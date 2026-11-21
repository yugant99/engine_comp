"""Unit tests for the pieces that do not need a server."""

import json
import os
import random

import pytest

from grader import check, fakemodel, metrics, scan
from grader.client import Record, SSEParser, _handle_event, validate_stream
from grader.conformance import choose_stop_string
from grader.prompts import COMPLETION_PROMPT_TOKENS, PromptError, build_plan
from grader.reference import MockReference
from grader.tokenizer import MockTokenizer, strip_incomplete_utf8

from conftest import make_corpus_text


# ---------------------------------------------------------------- SSE parsing

def _sse_bytes():
    events = []
    for piece in ["Hello", " wörld", " 日本語", " 😀", " naïve", ""]:
        events.append(json.dumps({"choices": [{"index": 0, "text": piece, "finish_reason": None}]},
                                 ensure_ascii=False))
    events.append(json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}))
    raw = b""
    for i, e in enumerate(events):
        sep = b"\r\n" if i % 2 else b"\n"
        raw += b": keep-alive comment" + sep if i == 2 else b""
        raw += b"data: " + e.encode("utf-8") + sep + sep
    raw += b"data:[DONE]\n\n"
    return raw, events + ["[DONE]"]


def test_sse_parser_any_chunking():
    raw, expected = _sse_bytes()
    rng = random.Random(1)
    for trial in range(300):
        p = SSEParser()
        got = []
        i = 0
        while i < len(raw):
            n = rng.randint(1, 7) if trial % 2 else 1
            got += p.feed(raw[i:i + n])
            i += n
        assert got == expected
        assert not p.pending()


def test_sse_parser_unterminated_event_is_pending():
    p = SSEParser()
    assert p.feed(b'data: {"a": 1}\n') == []
    assert p.pending()


def test_sse_parser_multiline_data():
    p = SSEParser()
    assert p.feed(b"data: a\ndata: b\n\n") == ["a\nb"]


def test_stream_record_rules():
    rec = Record(kind="chat")
    for d in ['{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
              '{"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
              '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
              '{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
              "[DONE]"]:
        _handle_event(rec, d, 1.0)
    validate_stream(rec, want_usage=True)
    assert rec.error is None and rec.text == "Hi" and rec.n_text_events == 1
    assert rec.roles == ["assistant"] and rec.finish_reason == "stop" and rec.usage_last

    rec = Record()
    _handle_event(rec, '{"choices":[{"index":0,"text":"a","finish_reason":"length"}]}', 1.0)
    _handle_event(rec, '{"choices":[{"index":0,"text":"b","finish_reason":null}]}', 1.0)
    _handle_event(rec, "[DONE]", 1.0)
    validate_stream(rec, want_usage=False)
    assert "finish_reason" in rec.error  # the last event with choices must carry it

    rec = Record()
    _handle_event(rec, '{"choices":[{"index":0,"text":"a","finish_reason":"length"}]}', 1.0)
    validate_stream(rec, want_usage=False)
    assert "[DONE]" in rec.error


# ---------------------------------------------------------------- metrics and checks

def test_nearest_rank():
    v = list(range(1, 101))
    assert metrics.nearest_rank(v, 50) == 50
    assert metrics.nearest_rank(v, 95) == 95
    assert metrics.nearest_rank([3.0], 95) == 3.0
    assert metrics.nearest_rank([5, 1, 3, 2, 4], 50) == 3
    assert metrics.nearest_rank([], 50) is None


def test_workload_metrics():
    recs = {}
    for i in range(4):
        r = Record(id=str(i))
        r.t_send, r.t_first_text, r.t_last_text, r.t_end = 10.0 + i, 10.5 + i, 11.5 + i, 12.0 + i
        recs[r.id] = r
    m = metrics.workload_metrics(recs, {str(i): 11 for i in range(4)})
    assert m["credited_tokens"] == 44
    assert m["wall_s"] == pytest.approx(5.0)
    assert m["output_tok_s"] == pytest.approx(44 / 5.0)
    assert m["ttft_p50_s"] == pytest.approx(0.5)
    assert m["tpot_p50_s"] == pytest.approx(0.1)
    assert m["e2e_p95_s"] == pytest.approx(2.0)


def test_geometric_mean():
    assert metrics.geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    assert metrics.geometric_mean([1.0, None]) == 0.0


def test_prefix_fraction():
    assert check.prefix_fraction("", "") == 1.0
    assert check.prefix_fraction("x", "") == 0.0
    assert check.prefix_fraction("abcd", "abXY") == 0.5
    assert check.prefix_fraction("abcdef", "abc") == 1.0


def _ref(text, n, eos):
    return {"text": text, "head_text": text[:5], "generated_tokens": n, "eos_stopped": eos,
            "finish_reason": "stop" if eos else "length"}


def test_usage_tolerance():
    assert check.usage_matches(128, _ref("x", 128, False))
    assert check.usage_matches(130, _ref("x", 128, False))  # 2% of 128 = 2.56
    assert not check.usage_matches(131, _ref("x", 128, False))
    assert check.usage_matches(41, _ref("x", 41, True))  # EOS counted
    assert check.usage_matches(40, _ref("x", 41, True))  # EOS not counted
    assert not check.usage_matches(42, _ref("x", 41, True))
    assert not check.usage_matches(None, _ref("x", 41, True))


def test_check_request_overrun_and_granularity():
    from grader.prompts import Request
    req = Request(id="a", workload="chat16", kind="chat", max_tokens=96, ignore_eos=False)
    rec = Record(id="a")
    rec._parts = ["Hello world", " and more"]
    rec.n_text_events = 2
    rec.usage = {"completion_tokens": 3}
    c = check.check_request(req, rec, _ref("Hello world", 3, True), engine_tokens=5)
    assert c["head_match"] and c["prefix_fraction"] == 1.0 and not c["full_match"]
    assert c["overrun"] and c["usage_ok"] is None
    assert not c["granularity_ok"]  # 2 events for 5 tokens


def test_strip_incomplete_utf8():
    assert strip_incomplete_utf8("abc��") == "abc"
    assert strip_incomplete_utf8("a�b") == "a�b"


def test_choose_stop_string():
    text = " the river ran under the old stone bridge"
    s, idx = choose_stop_string(text)
    assert s and idx >= 1 and text.find(s) == idx and not any(c.isspace() for c in s)
    assert choose_stop_string("ab") == (None, None)


# ---------------------------------------------------------------- prompts and mock model

def test_plan_exact_lengths_and_determinism():
    tok = MockTokenizer()
    corpus = make_corpus_text()
    a = build_plan(7, corpus, tok, "quick")
    b = build_plan(7, corpus, MockTokenizer(), "quick")
    c = build_plan(8, corpus, MockTokenizer(), "quick")
    assert [r.prompt for r in a.workloads["c8"]] == [r.prompt for r in b.workloads["c8"]]
    assert [r.prompt for r in a.workloads["c8"]] != [r.prompt for r in c.workloads["c8"]]
    assert len(a.workloads["single"]) == 2 and len(a.workloads["c32"]) == 32
    for r in a.workloads["single"] + a.workloads["c8"] + a.workloads["c32"]:
        assert len(tok.encode(r.prompt)) == r.prompt_tokens
        assert r.prompt_tokens in COMPLETION_PROMPT_TOKENS and r.max_tokens == 128 and r.ignore_eos
    systems = {r.messages[0]["content"] for r in a.workloads["chat16"]}
    assert len(systems) == 1 and len(tok.encode(systems.pop())) == 300
    for r in a.workloads["chat16"]:
        assert r.max_tokens == 96 and not r.ignore_eos
        assert r.messages[1]["content"].startswith("Summarize the following passage in two sentences.\n\n")
    full = build_plan(7, corpus, MockTokenizer(), "full")
    assert {k: len(v) for k, v in full.workloads.items()} == {"single": 8, "c8": 48, "c32": 160, "chat16": 64}
    assert len(full.conformance["concurrent"]) == 64
    assert full.conformance["over_length_tokens"] > 4096


def test_plan_rejects_small_corpus():
    with pytest.raises(PromptError):
        build_plan(1, "a b c " * 100, MockTokenizer(), "quick")


def test_mock_reference_matches_fake_generation():
    ref = MockReference()
    reqs = [{"id": "x", "kind": "completion", "prompt": "a b c", "max_tokens": 40, "ignore_eos": True},
            {"id": "y", "kind": "chat", "messages": [{"role": "user", "content": "Reply with only the word: yes"}],
             "max_tokens": 32, "ignore_eos": False}]
    out, _ = ref.run(reqs)
    assert out["x"]["generated_tokens"] == 40 and not out["x"]["eos_stopped"]
    assert out["y"]["text"] == " yes" and out["y"]["eos_stopped"] and out["y"]["generated_tokens"] == 2
    words, _ = fakemodel.generate(["a", "b", "c"], 40, True)
    assert out["x"]["text"] == "".join(fakemodel.token_text(w) for w in words)


# ---------------------------------------------------------------- scan

def test_python_imports():
    src = ("import os, vllm.engine\nfrom transformers import AutoModel\nfrom . import transformers\n"
           "import importlib\nimportlib.import_module('sglang.srt')\n__import__('xformers')\n"
           "x = 'import peft'\n")
    names = scan.python_imports(src)
    assert {"os", "vllm", "transformers", "importlib", "sglang", "xformers"} <= names
    assert "peft" not in names


def test_scan_source_tree(tmp_path):
    (tmp_path / "ok.py").write_text("import torch\nimport triton\nimport flash_attn\n")
    (tmp_path / "bad.py").write_text("from vllm import LLM\n")
    (tmp_path / "requirements.txt").write_text("numpy\n-r more.txt\n# transformers\n")
    (tmp_path / "more.txt").write_text("Bits_And_Bytes>=0.4\ngit+https://github.com/vllm-project/vllm.git\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\ndependencies=["accelerate>=1", "safetensors"]\n')
    (tmp_path / "Cargo.toml").write_text('[package]\nname="k"\n[dependencies]\ncandle-core = "0.8"\nserde = "1"\n')
    (tmp_path / "build.sh").write_text("pip install optimum\n")
    (tmp_path / "third_party" / "llama.cpp").mkdir(parents=True)
    (tmp_path / "vllm-main").mkdir()
    (tmp_path / "accelerated").mkdir()
    (tmp_path / ".git" / "vllm").mkdir(parents=True)
    findings, warnings = scan.scan_source_tree(str(tmp_path))
    text = "\n".join(findings)
    assert "bad.py: imports vllm" in text
    assert "Bits_And_Bytes" in text and "'vllm'" in text  # via -r include, name and URL
    assert "accelerate" in text and "candle-core" in text
    assert "llama.cpp/" in text
    assert "ok.py" not in text and "accelerated" not in text and ".git" not in text
    assert "vllm-main" not in text and any("vllm-main" in w for w in warnings)
    assert any("build.sh" in w and "optimum" in w for w in warnings)  # shell heuristics only warn


def test_scan_test_files_strict_by_default(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "compare_hf.py").write_text("import transformers\n")
    (tmp_path / "test_numbers.py").write_text("from transformers import AutoModel\n")
    (tmp_path / "server.py").write_text("import transformers\n")
    findings, _ = scan.scan_source_tree(str(tmp_path))
    assert len(findings) == 3
    findings, warnings = scan.scan_source_tree(str(tmp_path), exempt_tests=True)
    assert len(findings) == 1 and "server.py" in findings[0]
    assert sum("test file, exempt" in w for w in warnings) == 2


def test_parse_pip_list_formats():
    cols = "Package    Version\n---------- -------\naiohttp    3.11.0\nvLLM       0.9.0\n"
    freeze = "aiohttp==3.11.0\ntransformers==5.0.0\nfoo @ file:///x\n"
    js = json.dumps([{"name": "numpy", "version": "2.0"}, {"name": "onnxruntime-gpu", "version": "1.2"}])
    assert scan.parse_pip_list(cols) == [("aiohttp", "3.11.0"), ("vLLM", "0.9.0")]
    assert ("foo", None) in scan.parse_pip_list(freeze)
    assert len(scan.scan_pip_list(cols)) == 1 and len(scan.scan_pip_list(freeze)) == 1
    assert "ONNX Runtime" in scan.scan_pip_list(js)[0]


def test_scan_installed_local_vs_system(tmp_path):
    venv = tmp_path / "venv"
    dists = [{"name": "transformers", "version": "5", "location": "/usr/lib/python3/site-packages/t.dist-info"},
             {"name": "xformers", "version": "1", "location": str(venv / "lib" / "x.dist-info")},
             {"name": "torch", "version": "2", "location": str(venv / "lib" / "t.dist-info")}]
    findings, warnings = scan.scan_installed(dists, [str(venv)])
    assert len(findings) == 1 and "xformers" in findings[0]
    assert len(warnings) == 1 and "transformers" in warnings[0]


def _calibration_run(mode, scale, pf, failed=()):
    return {"mode": mode, "status": "pass", "calibration": True, "seed": 1, "baseline_engine": True,
            "workloads": {w: {"output_tok_s": scale * k} for k, w in enumerate(["single", "c8", "c32", "chat16"], 1)},
            "gates": {"prefix_fraction": {"value": pf}, "head_match": {"value": 1.0}, "failed": list(failed)}}


def test_make_baselines_median_margin_and_merge(tmp_path):
    from grader.grade import load_baselines
    from grader.make_baselines import build, main

    mode, b = build([_calibration_run("full", 1.0, 0.97), _calibration_run("full", 3.0, 0.99),
                     _calibration_run("full", 2.0, 0.95)], margin=0.02)
    assert mode == "full"
    assert b["tok_s"] == {"single": 2.0, "c8": 4.0, "c32": 6.0, "chat16": 8.0}
    assert b["prefix_fraction_gate"] == pytest.approx(0.95)

    out = tmp_path / "baselines.json"
    runs = []
    for i, (m, s) in enumerate([("full", 1.0), ("full", 2.0), ("full", 3.0), ("quick", 5.0)]):
        p = tmp_path / ("run%d.json" % i)
        p.write_text(json.dumps(_calibration_run(m, s, 0.98)))
        runs.append(str(p))
    assert main(["--margin", "0.01", "--out", str(out)] + runs[:3]) == 0
    with pytest.raises(ValueError, match="no baselines for quick"):
        load_baselines(str(out), "quick")
    assert main(["--margin", "0.01", "--out", str(out), runs[3]]) == 0  # adds quick, keeps full
    assert load_baselines(str(out), "full")["tok_s"]["single"] == 2.0
    assert load_baselines(str(out), "quick")["tok_s"]["single"] == 5.0


def test_make_baselines_refuses_failed_runs_and_gates():
    from grader.make_baselines import build

    bad = _calibration_run("full", 1.0, 0.9)
    bad["status"] = "fail"
    with pytest.raises(ValueError, match="did not pass"):
        build([bad], margin=0.0)
    with pytest.raises(ValueError, match="failed the gates"):
        build([_calibration_run("full", 1.0, 0.9, failed=["head_match"])], margin=0.0)


def test_launcher_wraps_run_as_and_taskset(tmp_path, monkeypatch):
    from grader import launch

    monkeypatch.setattr(launch.shutil, "which", lambda name: "/usr/bin/" + name)
    l = launch.Launcher(str(tmp_path), "/models/m", str(tmp_path / "work"), str(tmp_path / "e.log"))
    l.run_as, l._uid, l._gid, l.cores = "engine", 1001, 1002, "0-3"
    assert l.wrap(["bash", "run.sh"], pin=True) == [
        "setpriv", "--reuid=1001", "--regid=1002", "--init-groups", "--", "taskset", "-c", "0-3", "bash", "run.sh"]
    assert l.wrap(["bash", "build.sh"]) == ["setpriv", "--reuid=1001", "--regid=1002", "--init-groups", "--",
                                            "bash", "build.sh"]
    monkeypatch.setattr(launch.shutil, "which", lambda name: None if name == "setpriv" else "/usr/bin/" + name)
    argv = l.wrap(["bash", "run.sh"], pin=True, env={"PORT": "8000", "HOME": "/h"})
    assert argv[:6] == ["sudo", "-n", "-u", "engine", "--", "env"]
    assert "PORT=8000" in argv and argv[-5:] == ["taskset", "-c", "0-3", "bash", "run.sh"]
    env = l.env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["MODEL_DIR"] == "/models/m" and env["HOME"] == l.home
    assert env["PATH"].startswith(l.venv)


def test_grader_pins_itself_off_engine_cores(monkeypatch):
    from grader import grade

    calls = []
    monkeypatch.setattr(grade.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(grade.os, "sched_setaffinity", lambda pid, cpus: calls.append(set(cpus)), raising=False)
    assert grade.parse_cpu_list("0-3,6") == {0, 1, 2, 3, 6}
    assert grade.Grader._pin_grader(None, "0-3") == [4, 5, 6, 7]
    assert calls == [{4, 5, 6, 7}]


def test_engine_env_drops_secrets_and_proxies():
    from grader.launch import engine_base_env
    env = engine_base_env({"PATH": "/bin", "HTTPS_PROXY": "x", "http_proxy": "y", "GITHUB_TOKEN": "t",
                           "MY_API_KEY": "k", "PYTHONPATH": "/g", "CUDA_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
