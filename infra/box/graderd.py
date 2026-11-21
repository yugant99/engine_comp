#!/usr/bin/env python3
"""graderd: the grading daemon. Runs on the box as root.

Teams drop a request file into /srv/arena/queue (via the `arena` command). This daemon takes
requests one at a time, exports the requested commit from the team's git repository, runs the
grader on it while holding the GPU lock, and writes the verdict to /srv/arena/results.

The team never runs the grader itself and cannot read the grader's working files, so the hidden
prompts of a run stay hidden until the run is over.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUEUE = Path("/srv/arena/queue")
RESULTS = Path("/srv/arena/results")
SCOREBOARD = Path("/srv/arena/scoreboard.jsonl")
GPU_LOCK = Path("/srv/arena/gpu.lock")
RUNS = Path("/arena/runs")                 # root only: hidden prompts, references, logs
ENGINE_RUNS = Path("/home/engine/runs")    # the exported code, owned by the `engine` user
SRC = Path("/arena/src")
MODEL_DIR = "/arena/models/qwen2.5-1.5b-instruct"
CORPUS = "/arena/corpus.txt"
BASELINES = Path("/arena/baselines/baselines.json")
TEAM_REPO = "/home/team/repo.git"          # never taken from the request
ENGINE_CORES = "0-3"
GRADE_TIMEOUT_S = 45 * 60
KEEP_ENGINE_DIRS = 5


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def export_commit(commit, dest):
    """Copy the files of one commit out of the team's bare repo. Returns an error string or None."""
    if not all(c in "0123456789abcdef" for c in commit) or not 7 <= len(commit) <= 40:
        return "commit must be a hex sha"
    git = ["git", "-c", "safe.directory=*", f"--git-dir={TEAM_REPO}"]
    full = run(git + ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if full.returncode != 0:
        return f"commit {commit} is not in the team repository (did you push it?)"
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(git + ["archive", full.stdout.strip()], stdout=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    archive.wait()
    if archive.returncode != 0 or untar.returncode != 0:
        return "could not export the commit"
    subprocess.run(["chown", "-R", "engine:engine", str(dest.parent)])
    return None


def gpu_processes():
    out = run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def grade(req):
    rid = req["id"]
    mode = "quick" if req.get("mode") == "quick" else "full"
    work = RUNS / rid
    work.mkdir(parents=True, exist_ok=True)
    engine_dir = ENGINE_RUNS / rid / "engine"
    result_path = work / "result.json"

    err = export_commit(req.get("commit", ""), engine_dir)
    if err:
        return {"id": rid, "status": "fail", "fail_stage": "export", "fail_reason": err, "score": 0}

    cmd = [sys.executable, "-m", "grader.grade",
           "--engine-dir", str(engine_dir), "--model-dir", MODEL_DIR, "--corpus", CORPUS,
           "--out", str(result_path), "--commit", req["commit"],
           "--cores", ENGINE_CORES, "--run-as", "engine", "--port", "8000"]
    if mode == "quick":
        cmd.append("--quick")
    if BASELINES.exists():
        cmd += ["--baselines", str(BASELINES)]

    with open(GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)             # waits for any `arena gpu` command to finish
        strays = gpu_processes()
        started = time.time()
        try:
            with open(work / "grader.log", "w") as glog:
                # The grader itself stays off the engine's cores.
                proc = subprocess.run(["taskset", "-c", "4-7"] + cmd, cwd=SRC, stdout=glog,
                                      stderr=subprocess.STDOUT, timeout=GRADE_TIMEOUT_S)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        subprocess.run(["pkill", "-KILL", "-u", "engine"])   # nothing of the engine may survive
        fcntl.flock(lock, fcntl.LOCK_UN)

    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        tail = (work / "grader.log").read_text()[-3000:] if (work / "grader.log").exists() else ""
        result = {"status": "fail", "fail_stage": "grader", "score": 0,
                  "fail_reason": f"the grader did not produce a result (exit {rc})", "grader_log_tail": tail}
    result.update({"id": rid, "mode": mode, "commit": req["commit"], "agent": req.get("agent"),
                   "grade_seconds": round(time.time() - started, 1),
                   "gpu_processes_before_run": strays})
    return result


def summary_line(result):
    w = result.get("workloads") or {}
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": result["id"],
            "agent": result.get("agent"), "commit": result.get("commit", "")[:10],
            "mode": result.get("mode"), "status": result.get("status"),
            "fail_stage": result.get("fail_stage"), "score": result.get("score"),
            "cold_start_s": result.get("cold_start_s"),
            "tok_s": {k: v.get("tok_s") for k, v in w.items() if isinstance(v, dict)}}


def prune_engine_dirs():
    dirs = sorted((d for d in ENGINE_RUNS.glob("*") if d.is_dir()), key=lambda d: d.stat().st_mtime)
    for d in dirs[:-KEEP_ENGINE_DIRS]:
        shutil.rmtree(d, ignore_errors=True)


def main():
    for d in (QUEUE, RESULTS, RUNS, ENGINE_RUNS):
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "engine:engine", str(ENGINE_RUNS)])
    log("graderd started")
    while True:
        pending = sorted(QUEUE.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not pending:
            time.sleep(2)
            continue
        path = pending[0]
        try:
            req = json.loads(path.read_text())
            req["id"] = path.stem                      # the file name is the id; ignore any id inside
            log(f"grading {req['id']} commit={req.get('commit', '')[:10]} mode={req.get('mode')}")
            result = grade(req)
        except Exception as e:                         # a bad request must not stop the daemon
            result = {"id": path.stem, "status": "fail", "fail_stage": "request",
                      "fail_reason": f"{type(e).__name__}: {e}", "score": 0}
        (RESULTS / f"{result['id']}.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(RESULTS / f"{result['id']}.json.tmp", RESULTS / f"{result['id']}.json")
        with open(SCOREBOARD, "a") as f:
            f.write(json.dumps(summary_line(result)) + "\n")
        path.unlink(missing_ok=True)
        prune_engine_dirs()
        log(f"done {result['id']} status={result.get('status')} score={result.get('score')}")


if __name__ == "__main__":
    main()
