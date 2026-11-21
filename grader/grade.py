"""Grade one engine against SPEC v1.

    python -m grader.grade --engine-dir DIR --model-dir DIR --corpus FILE --out result.json \\
        [--quick] [--baselines baselines.json] [--commit SHA] [--cores 0-3] [--run-as USER]

Stages, in order: config, scan, prompts, reference (a subprocess), install, build,
installed_scan, cold_start, conformance, workloads (single, c8, c32, chat16), stop,
correctness, post_run_scan. The first failing stage ends the run. The result JSON is always
written, also on failure or interruption.

Without --baselines the run is in calibration mode: no ratios and no score, and the
correctness gates are reported but not enforced. Use this to measure vLLM itself
(with --baseline-engine, which also skips the banned-dependency scan).
"""

import argparse
import asyncio
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager

import aiohttp

from . import GRADER_VERSION, SPEC_VERSION
from . import check as checkmod
from .client import request, run_closed_loop
from .conformance import Conformance, failing
from .launch import Launcher, LaunchError, tail_lines
from .metrics import geometric_mean, workload_metrics
from .prompts import CLIENTS, REQUEST_COUNTS, WORKLOAD_ORDER, PromptError, build_plan, write_reference_requests
from .scan import scan_installed, scan_source_tree
from .tokenizer import make_tokenizer

PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_TAIL_LINES = 60
# Default overall budget: under the 45-minute SIGKILL of infra/box/graderd.py, so the grader
# always gets to stop the engine and write its result itself.
DEFAULT_TIME_BUDGET_S = 43 * 60
BUDGET_RESERVE_S = 60  # kept back for stopping the engine, correctness and writing the result


class StageFailure(Exception):
    def __init__(self, stage, reason):
        super().__init__("%s: %s" % (stage, reason))
        self.stage = stage
        self.reason = reason


class Interrupted(Exception):
    pass


def parse_cpu_list(spec):
    cpus = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def load_baselines(path, mode):
    """The entry for `mode` of a baselines file (format: grader/make_baselines.py).

    A file without an entry for this run's mode is a configuration error: grading against
    another mode's numbers, or silently not enforcing the gates, would both be wrong.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("spec_version")) != SPEC_VERSION:
        raise ValueError("baselines spec_version %r, grader implements %r" % (data.get("spec_version"), SPEC_VERSION))
    modes = data.get("modes") or {}
    if mode not in modes:
        raise ValueError("%s has no baselines for %s mode (it has: %s); add them with grader.make_baselines"
                         % (path, mode, ", ".join(sorted(modes)) or "none"))
    b = modes[mode]
    tok_s = b.get("tok_s") or {}
    for w in WORKLOAD_ORDER:
        if not isinstance(tok_s.get(w), (int, float)) or tok_s[w] <= 0:
            raise ValueError("baselines tok_s.%s missing or not positive" % w)
    gate = b.get("prefix_fraction_gate")
    if not isinstance(gate, (int, float)) or not 0.0 <= gate <= 1.0:
        raise ValueError("baselines prefix_fraction_gate missing or not in [0, 1]")
    return b


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Grader:
    def __init__(self, args):
        self.args = args
        self.seed = None
        self.tok = None
        self.plan = None
        self.refs = None
        self.launcher = None
        self.baselines = None
        self.loops = {}
        self.checks = {}
        self.private_dir = None
        self.work_dir = None
        self.remove_work_dir = False
        self.t_start = time.perf_counter()
        self.deadline = None if not args.time_budget else self.t_start + args.time_budget - BUDGET_RESERVE_S
        self.base_url = args.url.rstrip("/") if args.url else "http://%s:%d" % (args.host, args.port)
        self.log_path = args.engine_log or (os.path.splitext(args.out)[0] + ".engine.log")
        self.result = {
            "spec_version": SPEC_VERSION,
            "grader_version": GRADER_VERSION,
            "mode": "quick" if args.quick else "full",
            "commit": args.commit,
            "engine_dir": os.path.abspath(args.engine_dir) if args.engine_dir else None,
            "url": self.base_url,
            "baseline_engine": bool(args.baseline_engine),
            "reference_provider": args.reference,
            "calibration": args.baselines is None,
            "status": "fail",
            "fail_stage": None,
            "fail_reason": None,
            "seed": None,
            "started_at": utc_now(),
            "finished_at": None,
            "cold_start_s": None,
            "health_ready_s": None,
            "scan": {"skipped": None, "findings": [], "warnings": []},
            "conformance": [],
            "conformance_informational": [],
            "workloads": {},
            "gates": None,
            "ratios": None,
            "arena_score": 0.0,
            "stage_timings_s": {},
            "stop": None,
            "reference": None,
            "prompts": None,
            "environment": {},
            "engine_log": self.log_path,
            "engine_log_tail": [],
        }

    # ------------------------------------------------------------------ plumbing

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.result["stage_timings_s"][name] = round(time.perf_counter() - t0, 3)

    def run(self):
        try:
            self._run()
            self.result["status"] = "pass"
        except StageFailure as e:
            self._fail(e.stage, e.reason)
        except (Interrupted, KeyboardInterrupt) as e:
            self._fail("interrupted", str(e) or "interrupted")
        except Exception as e:
            self._fail("internal", "grader error: %s: %s" % (type(e).__name__, e))
            self.result["traceback"] = traceback.format_exc()
        finally:
            try:
                self._stop_engine()
            except Exception as e:  # never lose the result because of the stop
                self.result.setdefault("stop", {})
                self.result["stop"] = dict(self.result["stop"] or {}, error=str(e))
            self._finish()
        return 0 if self.result["status"] == "pass" else 1

    def _fail(self, stage, reason):
        if self.result["fail_stage"] is None:
            self.result["fail_stage"] = stage
            self.result["fail_reason"] = reason
        self.result["status"] = "fail"

    def _run(self):
        a = self.args
        with self.stage("config"):
            self._config()
        with self.stage("scan"):
            self._scan_source()
        with self.stage("prompts"):
            self._prompts()
        with self.stage("reference"):
            self._reference()
        if a.url:
            with self.stage("cold_start"):
                self._check_url()
        else:
            with self.stage("install"):
                self._launch_step(self.launcher.create_venv)
                self._launch_step(self.launcher.install_requirements)
            with self.stage("build"):
                self._launch_step(self.launcher.run_build)
            with self.stage("installed_scan"):
                self._scan_installed("installed_scan")
            with self.stage("cold_start"):
                self._cold_start()
        with self.stage("conformance"):
            self._conformance()
        workload_failure = None
        for name in WORKLOAD_ORDER:
            with self.stage("workload_" + name):
                workload_failure = self._workload(name)
            if workload_failure:
                break
        with self.stage("stop"):
            self._stop_engine()
        with self.stage("correctness"):
            gate_failure = self._evaluate()
        if workload_failure:
            raise StageFailure("workload", workload_failure)
        if not a.url:
            with self.stage("post_run_scan"):
                self._scan_installed("post_run_scan")
        if gate_failure and not self.result["calibration"]:
            raise StageFailure("correctness", gate_failure)

    def _launch_step(self, fn):
        try:
            fn()
        except LaunchError as e:
            raise StageFailure(e.stage, str(e))

    def budget(self, want_s, what):
        """want_s, capped by what is left of the grading time budget."""
        if self.deadline is None:
            return want_s
        left = self.deadline - time.perf_counter()
        if left <= 1:
            raise StageFailure("time_budget", "the grading time budget of %g s ran out before %s"
                               % (self.args.time_budget, what))
        return min(want_s, left)

    # ------------------------------------------------------------------ stages

    def _config(self):
        a = self.args
        mode = self.result["mode"]
        if a.reference == "hf" and not a.model_dir:
            raise StageFailure("config", "--model-dir is required with --reference hf")
        if not os.path.isfile(a.corpus):
            raise StageFailure("config", "corpus file %r not found" % a.corpus)
        if not a.url:
            if not a.engine_dir or not os.path.isdir(a.engine_dir):
                raise StageFailure("config", "--engine-dir %r is not a directory" % a.engine_dir)
        if a.baselines:
            try:
                self.baselines = load_baselines(a.baselines, mode)
            except (OSError, ValueError) as e:
                raise StageFailure("config", "baselines: %s" % e)
        out_dir = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(out_dir, exist_ok=True)
        env = self.result["environment"]
        env.update(python=sys.version.split()[0], platform=sys.platform, engine_cores=a.cores,
                   run_as=a.run_as, host=a.host, port=a.port, request_timeout_s=a.request_timeout)
        if a.cores:
            env["grader_cpu_affinity"] = self._pin_grader(a.cores)
        self.seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "big") >> 1
        self.private_dir = tempfile.mkdtemp(prefix="arena-private-")  # 0700: reference files
        if a.work_dir:
            self.work_dir = os.path.abspath(a.work_dir)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.work_dir = tempfile.mkdtemp(prefix="arena-work-")
            self.remove_work_dir = True
        if not a.url:
            try:
                self.launcher = Launcher(a.engine_dir, a.model_dir, self.work_dir, self.log_path, host=a.host,
                                         port=a.port, cores=a.cores, run_as=a.run_as, python=a.engine_python,
                                         stop_grace_s=a.stop_grace)
            except LaunchError as e:
                raise StageFailure("config", str(e))
            self.launcher.deadline = self.deadline

    def _pin_grader(self, cores):
        """Keep the grader off the engine's cores (Linux only)."""
        if not hasattr(os, "sched_setaffinity"):
            return "not supported on this platform"
        engine = parse_cpu_list(cores)
        rest = os.sched_getaffinity(0) - engine
        if not rest:
            return "no cores left for the grader; not pinned"
        os.sched_setaffinity(0, rest)
        return sorted(rest)

    def _scan_source(self):
        a = self.args
        scan = self.result["scan"]
        if a.baseline_engine:
            scan["skipped"] = "baseline engine"
            return
        if not a.engine_dir:
            scan["skipped"] = "no engine directory"
            return
        findings, warnings = scan_source_tree(a.engine_dir, skip_paths=[self.work_dir],
                                              exempt_tests=a.scan_exempt_tests)
        scan["findings"] += findings
        scan["warnings"] += warnings
        if findings:
            raise StageFailure("scan", "%d banned dependency finding(s): %s" % (len(findings), "; ".join(findings[:5])))

    def _scan_installed(self, stage_name):
        if self.args.baseline_engine:
            return
        try:
            dists = self.launcher.list_installed()
        except LaunchError as e:
            raise StageFailure("scan", "%s: %s" % (stage_name, e))
        findings, warnings = scan_installed(dists, self.launcher.local_roots())
        scan = self.result["scan"]
        scan["findings"] += [x for x in findings if x not in scan["findings"]]
        scan["warnings"] += [x for x in warnings if x not in scan["warnings"]]
        if findings:
            raise StageFailure("scan", "%s: %s" % (stage_name, "; ".join(findings[:5])))

    def _prompts(self):
        a = self.args
        try:
            self.tok = make_tokenizer(a.reference, a.model_dir)
            with open(a.corpus, encoding="utf-8") as f:
                corpus = f.read()
            self.plan = build_plan(self.seed, corpus, self.tok, self.result["mode"])
        except PromptError as e:
            raise StageFailure("prompts", str(e))
        self.result["prompts"] = self.plan.summary()

    def _reference(self):
        a = self.args
        req_path = os.path.join(self.private_dir, "requests.json")
        out_path = os.path.join(self.private_dir, "reference.json")
        log_path = os.path.join(self.private_dir, "reference.log")
        write_reference_requests(self.plan, req_path)
        cmd = [sys.executable, "-m", "grader.reference", "--provider", a.reference,
               "--requests", req_path, "--out", out_path]
        if a.reference == "hf":
            cmd += ["--model-dir", a.model_dir]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONPATH=PACKAGE_PARENT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        timeout = self.budget(a.reference_timeout, "the reference")
        with open(log_path, "wb") as log:
            try:
                p = subprocess.run(cmd, cwd=PACKAGE_PARENT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = "timeout after %d s" % timeout
        if code != 0:
            raise StageFailure("reference", "reference process failed (%s): %s"
                               % (code, " | ".join(tail_lines(log_path, 8))))
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        shutil.rmtree(self.private_dir, ignore_errors=True)  # nothing of it stays on disk for the engine
        self.private_dir = None
        refs = data["results"]
        for r in self.plan.all_reference_requests():
            if r.id not in refs:
                raise StageFailure("reference", "grader error: no reference for %s" % r.id)
            if refs[r.id]["prompt_tokens"] != r.prompt_tokens:
                raise StageFailure("reference", "grader error: %s has %d prompt tokens in the reference, %d planned"
                                   % (r.id, refs[r.id]["prompt_tokens"], r.prompt_tokens))
        self.refs = refs
        self.result["reference"] = data.get("meta")

    def _check_url(self):
        async def probe():
            async with aiohttp.ClientSession() as s:
                return await request(s, "GET", self.base_url + "/health", timeout_s=10)
        r = asyncio.run(probe())
        if r.status != 200:
            raise StageFailure("launch", "GET %s/health: %s %s" % (self.base_url, r.status, r.error or ""))

    def _cold_start(self):
        try:
            t_launch = self.launcher.start()
        except LaunchError as e:
            raise StageFailure(e.stage, str(e))
        health_s, ready_s = asyncio.run(self._wait_ready(t_launch))
        self.result["health_ready_s"] = round(health_s, 3)
        self.result["cold_start_s"] = round(ready_s, 3)

    async def _wait_ready(self, t_launch):
        a = self.args
        deadline = t_launch + self.budget(a.ready_timeout, "the cold start")
        body = {"prompt": "Hello", "max_tokens": 1, "temperature": 0}
        if a.model_name:
            body["model"] = a.model_name
        async with aiohttp.ClientSession() as s:
            while True:
                code = self.launcher.exit_code()
                if code is not None:
                    raise StageFailure("cold_start", "run.sh exited with code %d before /health returned 200" % code)
                if time.perf_counter() > deadline:
                    raise StageFailure("cold_start", "GET /health did not return 200 within %g s" % a.ready_timeout)
                r = await request(s, "GET", self.base_url + "/health", timeout_s=2)
                if r.status == 200:
                    break
                await asyncio.sleep(0.25)
            t_health = time.perf_counter()
            last = None
            end = t_health + a.request_timeout
            if self.deadline is not None:
                end = min(end, self.deadline)
            while True:
                remaining = end - time.perf_counter()
                if remaining <= 0:
                    raise StageFailure("cold_start", "no successful 1-token completion within %g s of /health "
                                       "returning 200 (last: %s)" % (a.request_timeout, last))
                r = await request(s, "POST", self.base_url + "/v1/completions", payload=body, timeout_s=remaining)
                ok = (r.status == 200 and isinstance(r.json, dict) and isinstance(r.json.get("choices"), list)
                      and r.json["choices"] and isinstance(r.json["choices"][0].get("text"), str))
                if ok:
                    return t_health - t_launch, time.perf_counter() - t_launch
                last = r.error or "HTTP %s %s" % (r.status, r.body[:200])
                if self.launcher.exit_code() is not None:
                    raise StageFailure("cold_start", "run.sh exited with code %d" % self.launcher.exit_code())
                await asyncio.sleep(0.25)

    def _conformance(self):
        a = self.args
        conf = Conformance(self.base_url, self.tok, self.plan.conformance, self.refs,
                           timeout_s=a.request_timeout, model_name=a.model_name)
        limit = self.budget(3600, "conformance")
        try:
            results = asyncio.run(asyncio.wait_for(conf.run(), limit))
        except asyncio.TimeoutError:
            raise StageFailure("time_budget", "conformance did not finish within the %g s left of the grading "
                               "time budget" % limit)
        # Informational checks go under their own key, so that tools listing failed conformance
        # checks (infra/box/arena) do not present them as fatal.
        self.result["conformance"] = [r.to_dict() for r in results if r.enforced]
        self.result["conformance_informational"] = [r.to_dict() for r in results if not r.enforced]
        bad = failing(results, baseline_engine=a.baseline_engine)
        if bad:
            raise StageFailure("conformance", "%d check(s) failed: %s"
                               % (len(bad), "; ".join("%s: %s" % (r.name, r.detail[:200]) for r in bad)))

    def _workload(self, name):
        a = self.args
        reqs = self.plan.workloads[name]
        n_clients = min(CLIENTS[name], len(reqs))
        stage_limit = self.budget((len(reqs) // n_clients + 2) * a.request_timeout + 60, "workload " + name)
        lr = asyncio.run(run_closed_loop(self.base_url, reqs, CLIENTS[name], a.request_timeout, a.model_name,
                                         stage_timeout_s=stage_limit))
        self.loops[name] = lr
        self.result["workloads"][name] = {
            "clients": CLIENTS[name],
            "requests_planned": len(reqs),
            "requests_completed": len(lr.records),
            "requests_failed": sum(1 for r in lr.records.values() if r.error),
            "failure": lr.failure,
        }
        if lr.failure:
            return "%s: %s" % (name, lr.failure)
        return None

    def _evaluate(self):
        """Correctness and metrics for every workload that ran. Returns the gate failure, if any."""
        all_checks = []
        for name in WORKLOAD_ORDER:
            lr = self.loops.get(name)
            if lr is None:
                continue
            ok_records = {k: v for k, v in lr.records.items() if not v.error}
            checks = checkmod.evaluate(self.plan.workloads[name], ok_records, self.refs, self.tok)
            self.checks[name] = checks
            credited = {c["id"]: c["credited_tokens"] for c in checks}
            w = self.result["workloads"][name]
            w.update(workload_metrics(ok_records, credited))
            w["requests_completed"] = len(lr.records)
            agg = checkmod.aggregate(checks)
            w.update({k: agg[k] for k in ("head_match_rate", "head_match_count", "prefix_fraction", "full_match_count",
                                          "usage_checked", "usage_failures", "granularity_failures",
                                          "min_events_per_credited_token", "overruns", "finish_reason_mismatches")})
            all_checks += checks
        if not all_checks:
            return None
        gate = self.baselines["prefix_fraction_gate"] if self.baselines else None
        g = checkmod.gates(all_checks, gate, enforced=not self.result["calibration"])
        self.result["gates"] = g
        return checkmod.describe_gate_failures(g) if g["failed"] else None

    def _stop_engine(self):
        if self.launcher is not None and self.launcher.proc is not None and self.result["stop"] is None:
            self.result["stop"] = self.launcher.stop()

    # ------------------------------------------------------------------ result

    def _finish(self):
        r = self.result
        a = self.args
        r["seed"] = self.seed
        r["finished_at"] = utc_now()
        if self.baselines:
            ratios = {}
            for w in WORKLOAD_ORDER:
                tps = (r["workloads"].get(w) or {}).get("output_tok_s")
                ratios[w] = None if tps is None else tps / self.baselines["tok_s"][w]
            r["ratios"] = ratios
            r["baselines"] = {"file": os.path.abspath(a.baselines), "tok_s": self.baselines["tok_s"],
                              "prefix_fraction_gate": self.baselines["prefix_fraction_gate"]}
        if r["status"] != "pass":
            r["arena_score"] = 0.0
        elif self.baselines:
            r["arena_score"] = geometric_mean([r["ratios"][w] for w in WORKLOAD_ORDER])
        else:
            r["arena_score"] = None
        # Aliases under the names infra/box/arena and graderd.py read.
        r["score"] = r["arena_score"]
        for w, m in r["workloads"].items():
            m["tok_s"] = m.get("output_tok_s")
            m["ratio"] = (r["ratios"] or {}).get(w)
            m["ttft_p50"] = m.get("ttft_p50_s")
            m["ttft_p95"] = m.get("ttft_p95_s")
        r["total_s"] = round(time.perf_counter() - self.t_start, 3)
        r["engine_log_tail"] = tail_lines(self.log_path, LOG_TAIL_LINES) if self.launcher else []
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, sort_keys=False, default=str)
        os.replace(tmp, a.out)
        if a.details:
            self._write_details(a.details)
        if self.private_dir:
            shutil.rmtree(self.private_dir, ignore_errors=True)
        if self.remove_work_dir and self.work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        print(summary(r, a.out))

    def _write_details(self, path):
        out = {"seed": self.seed, "requests": []}
        for name in WORKLOAD_ORDER:
            lr = self.loops.get(name)
            if lr is None:
                continue
            t0 = lr.t_first_send or 0.0
            checks = {c["id"]: c for c in self.checks.get(name, [])}
            for req in self.plan.workloads[name]:
                rec = lr.records.get(req.id)
                if rec is None:
                    continue
                ref = self.refs[req.id]
                out["requests"].append({
                    "workload": name, "kind": req.kind, "prompt_tokens": req.prompt_tokens,
                    "max_tokens": req.max_tokens, "record": rec.summary(t0), "check": checks.get(req.id),
                    "reference": {k: ref[k] for k in ("text", "head_text", "generated_tokens", "finish_reason")},
                })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


def _fmt(v, scale=1.0, digits=1):
    return "-" if v is None else "%.*f" % (digits, v * scale)


def summary(r, out_path):
    lines = []
    mode = r["mode"] + (" calibration" if r["calibration"] else "")
    lines.append("Engine Arena grader %s (spec %s), %s mode" % (r["grader_version"], r["spec_version"], mode))
    if r["status"] == "pass":
        lines.append("status: PASS")
    else:
        lines.append("status: FAIL at %s: %s" % (r["fail_stage"], (r["fail_reason"] or "")[:600]))
    lines.append("cold start: %s s" % _fmt(r["cold_start_s"], digits=2))
    enforced = r["conformance"]
    info = r["conformance_informational"]
    if enforced:
        lines.append("conformance: %d/%d passed%s" % (
            sum(c["passed"] for c in enforced), len(enforced),
            ("; informational: " + ", ".join("%s %s" % (c["name"], "ok" if c["passed"] else "not met") for c in info))
            if info else ""))
        for c in enforced + info:
            if not c["passed"]:
                lines.append("  %s %s: %s" % ("FAIL" if c["enforced"] else "info", c["name"], c["detail"][:300]))
    if r["workloads"]:
        lines.append("%-7s %9s %15s %15s %13s %6s %7s" % ("", "tok/s", "TTFT p50/p95ms", "TPOT p50/p95ms",
                                                         "e2e p50/p95s", "head", "prefix"))
        for name in WORKLOAD_ORDER:
            w = r["workloads"].get(name)
            if not w:
                continue
            lines.append("%-7s %9s %15s %15s %13s %6s %7s" % (
                name, _fmt(w.get("output_tok_s")),
                "%s/%s" % (_fmt(w.get("ttft_p50_s"), 1000, 0), _fmt(w.get("ttft_p95_s"), 1000, 0)),
                "%s/%s" % (_fmt(w.get("tpot_p50_s"), 1000, 1), _fmt(w.get("tpot_p95_s"), 1000, 1)),
                "%s/%s" % (_fmt(w.get("e2e_p50_s"), 1, 2), _fmt(w.get("e2e_p95_s"), 1, 2)),
                _fmt(w.get("head_match_rate"), 1, 3), _fmt(w.get("prefix_fraction"), 1, 4)))
    g = r["gates"]
    if g:
        parts = []
        for k in ("head_match", "prefix_fraction", "usage", "streaming_granularity"):
            v = g[k]
            state = {True: "ok", False: "FAILED", None: "n/a"}[v["passed"]]
            parts.append("%s %s" % (k, state))
        lines.append("gates (%s): %s" % ("enforced" if g["enforced"] else "not enforced", ", ".join(parts)))
    if r["ratios"]:
        lines.append("ratios: " + ", ".join("%s %s" % (w, _fmt(r["ratios"][w], 1, 3)) for w in WORKLOAD_ORDER))
    lines.append("arena score: %s" % ("-" if r["arena_score"] is None else "%.4f" % r["arena_score"]))
    lines.append("seed: %s" % r["seed"])
    lines.append("result: %s" % out_path)
    return "\n".join(lines)


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m grader.grade", description="Grade an Engine Arena submission.")
    ap.add_argument("--engine-dir", help="the submission (run.sh at its root)")
    ap.add_argument("--model-dir", help="Qwen2.5-1.5B-Instruct in Hugging Face layout")
    ap.add_argument("--corpus", required=True, help="text file prompts are cut from")
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--quick", action="store_true", help="quick mode: %s" % REQUEST_COUNTS["quick"])
    ap.add_argument("--seed", type=int, default=None,
                    help="default: fresh from os.urandom. Never pass this on a shared box: argv is world-readable")
    ap.add_argument("--reference", choices=["hf", "mock"], default="hf")
    ap.add_argument("--baselines", help="baselines JSON; without it the run is in calibration mode")
    ap.add_argument("--baseline-engine", action="store_true", help="grading vLLM itself: skip the scan")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cores", help="pin run.sh with taskset -c CORES (and keep the grader off them)")
    ap.add_argument("--run-as", help="run install, build and run.sh as this Unix user (grader must be root)")
    ap.add_argument("--url", help="grade an already-running server (no install, build, launch or cold start)")
    ap.add_argument("--model-name", help="send this as the 'model' field (default: omitted)")
    ap.add_argument("--engine-python", default=None, help="interpreter for the engine virtualenv (default: this one)")
    ap.add_argument("--work-dir", help="where the engine virtualenv and HOME go (default: a temp dir, removed)")
    ap.add_argument("--engine-log", help="engine log path (default: <out>.engine.log)")
    ap.add_argument("--details", help="also write per-request details (texts, timings) to this file")
    ap.add_argument("--scan-exempt-tests", action="store_true",
                    help="banned imports in test files (tests/, test_*.py, ...) only warn (default: they fail)")
    ap.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET_S,
                    help="overall seconds for the whole run; 0 = none (default %(default)s, under graderd's 45 min)")
    ap.add_argument("--request-timeout", type=float, default=120.0, help=argparse.SUPPRESS)
    ap.add_argument("--ready-timeout", type=float, default=300.0, help=argparse.SUPPRESS)
    ap.add_argument("--stop-grace", type=float, default=10.0, help=argparse.SUPPRESS)
    ap.add_argument("--reference-timeout", type=float, default=3600.0)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.url and not args.engine_dir:
        build_parser().error("--engine-dir is required unless --url is given")

    def on_sigterm(signum, frame):
        raise Interrupted("received signal %d" % signum)

    signal.signal(signal.SIGTERM, on_sigterm)
    return Grader(args).run()


if __name__ == "__main__":
    sys.exit(main())
