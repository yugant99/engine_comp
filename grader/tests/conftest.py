"""Helpers for the grader tests. Run from engine-arena/:  python -m pytest grader/tests"""

import json
import os
import random
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_corpus_text(n_words=30000, seed=0):
    """Pseudo-English text with punctuation, newlines and some multi-byte words."""
    rng = random.Random(seed)
    words = ["the", "of", "and", "river", "light", "stone", "garden", "winter", "said", "slowly",
             "über", "café", "naïve", "Ωmega", "東京", "Привет", "😀", "jalapeño", "north", "voice"]
    words += ["".join(rng.choice("abcdefghijklmnoprstuvwy") for _ in range(rng.randint(2, 9))) for _ in range(400)]
    out = []
    for i in range(n_words):
        w = rng.choice(words)
        if rng.random() < 0.08:
            w += rng.choice([".", ",", ";", "!"])
        out.append(w)
        if rng.random() < 0.02:
            out.append("\n\n")
    return " ".join(out)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    path.write_text(make_corpus_text(), encoding="utf-8")
    return str(path)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_engine(dirpath, mock_args="", run_sh=None, files=None):
    """An engine directory whose run.sh starts grader.mock_engine with the given flags."""
    os.makedirs(dirpath, exist_ok=True)
    if run_sh is None:
        run_sh = ("#!/bin/bash\nexport PYTHONPATH=%s\n"
                  'exec python -m grader.mock_engine --host "$HOST" --port "$PORT" --token-delay 0.001 %s\n'
                  % (ROOT, mock_args))
    with open(os.path.join(dirpath, "run.sh"), "w") as f:
        f.write(run_sh)
    for name, content in (files or {}).items():
        p = os.path.join(dirpath, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    return str(dirpath)


def quick_baselines(path, gate=0.9, modes=("quick",)):
    entry = {"tok_s": {"single": 100.0, "c8": 400.0, "c32": 800.0, "chat16": 400.0}, "prefix_fraction_gate": gate}
    with open(path, "w") as f:
        json.dump({"spec_version": "1", "modes": {m: entry for m in modes}}, f)
    return str(path)


def run_grade(tmp_path, corpus, engine_dir=None, extra=(), timeout=300):
    out = os.path.join(str(tmp_path), "result.json")
    cmd = [PY, "-m", "grader.grade", "--corpus", corpus, "--out", out, "--quick", "--reference", "mock",
           "--seed", "4242", "--port", str(free_port()), "--commit", "test"]
    if engine_dir:
        cmd += ["--engine-dir", engine_dir]
    cmd += list(extra)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert os.path.isfile(out), "no result written\nstdout:\n%s\nstderr:\n%s" % (p.stdout, p.stderr)
    with open(out) as f:
        result = json.load(f)
    result["_stdout"] = p.stdout
    result["_stderr"] = p.stderr
    result["_returncode"] = p.returncode
    result["_elapsed"] = time.time() - t0
    return result


def conformance_map(result):
    return {c["name"]: c for c in result["conformance"] + result["conformance_informational"]}
