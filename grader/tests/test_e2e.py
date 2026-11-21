"""End-to-end: the full grade pipeline (quick mode, mock reference) against grader.mock_engine.

Each fault must fail the run at the right stage. Faults that apply to every request are caught
by conformance; with --fault-min-tokens 100 they only hit the workloads (128 / 96 tokens), which
proves the workload-stage checks too. Gates are enforced only with a baselines file.
"""

import os
import subprocess
import time

import pytest

from conftest import PY, ROOT, conformance_map, free_port, make_engine, quick_baselines, run_grade


@pytest.fixture
def baselines(tmp_path):
    return quick_baselines(tmp_path / "baselines.json", gate=0.9)


def grade_mock(tmp_path, corpus, mock_args="", extra=(), **engine_kw):
    eng = make_engine(tmp_path / "engine", mock_args, **engine_kw)
    return run_grade(tmp_path, corpus, eng, extra)


def assert_fail(result, stage, *needles):
    assert result["status"] == "fail", result["_stdout"]
    assert result["fail_stage"] == stage, (result["fail_stage"], result["fail_reason"])
    for n in needles:
        assert n in result["fail_reason"], result["fail_reason"]
    assert result["arena_score"] == 0.0
    assert result["_returncode"] == 1
    assert result["seed"] == 4242


# ---------------------------------------------------------------- clean runs

def test_clean_pass_calibration(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, extra=["--details", str(tmp_path / "details.json")])
    assert r["status"] == "pass", r["_stdout"]
    assert r["calibration"] and r["ratios"] is None and r["arena_score"] is None
    conf = conformance_map(r)
    assert all(c["passed"] for c in conf.values()), [c for c in conf.values() if not c["passed"]]
    assert len(conf) == 18
    for w in ("single", "c8", "c32", "chat16"):
        m = r["workloads"][w]
        assert m["requests_completed"] == m["requests_planned"]
        assert m["head_match_rate"] == 1.0 and m["prefix_fraction"] == 1.0
        assert m["full_match_count"] == m["requests_planned"]
        assert m["usage_failures"] == 0 and m["granularity_failures"] == 0
        assert m["output_tok_s"] > 0 and m["ttft_p95_s"] >= m["ttft_p50_s"] > 0
    g = r["gates"]
    assert g["enforced"] is False and g["failed"] == [] and g["prefix_fraction"]["passed"] is None
    assert r["cold_start_s"] > 0 and r["stop"]["killed"] is False and r["stop"]["port_closed"]
    assert r["stage_timings_s"]["reference"] >= 0 and r["engine_log_tail"]
    assert os.path.isfile(tmp_path / "details.json")
    assert "status: PASS" in r["_stdout"]
    # Aliases read by infra/box/arena and graderd.py.
    assert r["score"] is None
    assert r["workloads"]["c8"]["tok_s"] == r["workloads"]["c8"]["output_tok_s"]
    assert r["workloads"]["c8"]["ttft_p50"] == r["workloads"]["c8"]["ttft_p50_s"]


def test_clean_pass_with_baselines_and_split_utf8(tmp_path, corpus, baselines):
    # --chunk-bytes 3 splits every event, and multi-byte characters, across many TCP writes.
    r = grade_mock(tmp_path, corpus, "--chunk-bytes 3", ["--baselines", baselines])
    assert r["status"] == "pass", r["_stdout"]
    assert not r["calibration"] and r["gates"]["enforced"] and r["gates"]["failed"] == []
    assert set(r["ratios"]) == {"single", "c8", "c32", "chat16"}
    assert r["arena_score"] > 0 and r["score"] == r["arena_score"]
    assert r["workloads"]["c32"]["ratio"] == r["ratios"]["c32"]
    assert r["gates"]["head_match"]["value"] == 1.0


def test_baselines_without_this_mode_fail_config(tmp_path, corpus):
    bl = quick_baselines(tmp_path / "full_only.json", modes=("full",))
    r = grade_mock(tmp_path, corpus, extra=["--baselines", bl])
    assert r["status"] == "fail" and r["fail_stage"] == "config"
    assert "no baselines for quick mode" in r["fail_reason"]


def test_url_mode(tmp_path, corpus):
    port = free_port()
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.Popen([PY, "-m", "grader.mock_engine", "--port", str(port), "--token-delay", "0.001"],
                         cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        r = run_grade(tmp_path, corpus, None, ["--url", "http://127.0.0.1:%d" % port])
    finally:
        p.terminate()
        p.wait(10)
    assert r["status"] == "pass", r["_stdout"]
    assert r["cold_start_s"] is None and r["scan"]["skipped"]


# ---------------------------------------------------------------- scan, install, build, start

def test_scan_banned_import(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, files={"engine/model.py": "import torch\nfrom vllm import LLM\n"})
    assert_fail(r, "scan", "vllm")
    assert r["stage_timings_s"].get("reference") is None  # nothing ran after the scan


def test_scan_banned_requirement(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, files={"requirements.txt": "numpy\ntransformers==4.57.0\n"})
    assert_fail(r, "scan", "transformers")


def test_baseline_engine_skips_scan(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, extra=["--baseline-engine"], files={"vendored.py": "import vllm\n"})
    assert r["status"] == "pass", r["_stdout"]
    assert r["scan"]["skipped"] == "baseline engine"


def test_build_failure(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, files={"build.sh": "echo compiling; exit 3\n"})
    assert_fail(r, "build", "code 3")
    assert any("compiling" in line for line in r["engine_log_tail"])


def test_run_sh_exits(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, run_sh="#!/bin/bash\necho starting; exit 7\n")
    assert_fail(r, "cold_start", "code 7")


def test_never_ready(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault no-health", ["--ready-timeout", "3"])
    assert_fail(r, "cold_start", "/health")


# ---------------------------------------------------------------- faults caught by conformance

@pytest.mark.parametrize("fault, check", [
    ("bursty", "streaming_granularity"),
    ("bad-usage", "stream_include_usage"),
    ("eos-ignore", "eos_stop"),
    ("stop-include", "stop_strings"),
])
def test_fault_fails_conformance(tmp_path, corpus, fault, check):
    r = grade_mock(tmp_path, corpus, "--fault " + fault)
    assert_fail(r, "conformance", check)
    assert conformance_map(r)[check]["passed"] is False
    assert r["workloads"] == {}  # workloads never ran


# ---------------------------------------------------------------- faults caught in the workloads

def test_wrong_text_early_fails_head_match(tmp_path, corpus, baselines):
    r = grade_mock(tmp_path, corpus, "--fault wrong-after:4 --fault-min-tokens 100", ["--baselines", baselines])
    assert_fail(r, "correctness", "head_match")
    assert all(c["passed"] for c in r["conformance"] if c["enforced"])


def test_late_drift_fails_prefix_gate(tmp_path, corpus, baselines):
    r = grade_mock(tmp_path, corpus, "--fault wrong-after:60 --fault-min-tokens 100", ["--baselines", baselines])
    assert_fail(r, "correctness", "prefix_fraction")
    assert r["gates"]["head_match"]["passed"] is True
    assert r["gates"]["prefix_fraction"]["value"] < 0.9


def test_cross_request_mixup_fails_correctness(tmp_path, corpus, baselines):
    r = grade_mock(tmp_path, corpus, "--fault mixup --fault-min-tokens 100", ["--baselines", baselines])
    assert_fail(r, "correctness", "head_match")


def test_mixup_on_every_request_never_passes(tmp_path, corpus, baselines):
    # The two-call conformance checks tolerate float16 rounding noise by retrying, so an engine that
    # answers with the previous request's prompt can get past them. It must still fail the run:
    # the comparison with the reference catches it.
    r = grade_mock(tmp_path, corpus, "--fault mixup", ["--baselines", baselines])
    assert r["status"] == "fail", r["_stdout"]
    assert r["arena_score"] == 0.0


def test_bursts_under_load_fail_granularity(tmp_path, corpus, baselines):
    r = grade_mock(tmp_path, corpus, "--fault bursty --fault-min-tokens 100", ["--baselines", baselines])
    assert_fail(r, "correctness", "streaming_granularity")


def test_wrong_usage_in_workloads(tmp_path, corpus, baselines):
    r = grade_mock(tmp_path, corpus, "--fault bad-usage --fault-min-tokens 100", ["--baselines", baselines])
    assert_fail(r, "correctness", "usage")


def test_http_500_fails_workload(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault error500:5 --fault-min-tokens 100")
    assert_fail(r, "workload", "HTTP 500")


def test_timeout_fails_workload(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault hang:10 --fault-min-tokens 100", ["--request-timeout", "3"])
    assert_fail(r, "workload", "timeout")
    assert r["_elapsed"] < 120


def test_time_budget_ends_a_slow_run_with_a_result(tmp_path, corpus):
    # 60 s of the budget are held back for stop/correctness/writing, so 72 s leaves ~12 s to grade.
    r = grade_mock(tmp_path, corpus, "--fault hang:10 --fault-min-tokens 100", ["--time-budget", "72"])
    assert r["status"] == "fail" and r["fail_stage"] in ("workload", "time_budget"), r["fail_reason"]
    assert r["_elapsed"] < 40
    assert r["stop"]["port_closed"]


def test_calibration_reports_but_does_not_enforce_gates(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault wrong-after:4 --fault-min-tokens 100")
    assert r["status"] == "pass"
    assert r["gates"]["enforced"] is False and "head_match" in r["gates"]["failed"]


# ---------------------------------------------------------------- informational checks and stop

def test_hold_all_is_informational(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault hold-all")
    assert r["status"] == "pass"
    spread = conformance_map(r)["streaming_spread"]
    assert spread["passed"] is False and spread["enforced"] is False


def test_sigterm_ignored_is_killed(tmp_path, corpus):
    r = grade_mock(tmp_path, corpus, "--fault ignore-sigterm", ["--stop-grace", "1"])
    assert r["status"] == "pass"
    assert r["stop"]["killed"] is True and r["stop"]["port_closed"] is True
