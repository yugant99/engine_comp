"""Correctness of workload requests against the reference (SPEC section 5, "Correctness").

Per request:
  head match       engine text starts with the reference's first 16 tokens (head_text)
  prefix fraction  common prefix chars / len(reference text); 1.0 when both are empty,
                   0.0 when only the reference is empty
  credited tokens  the grader's tokenization of the engine text, capped at max_tokens
  usage            only when the text matches the reference in full: usage.completion_tokens
                   within 2% of the reference token count. When the reference row ended on EOS
                   both conventions are accepted: counting the EOS token or not.
  granularity      text events >= 0.5 * credited tokens (no bursts)

Reported but not enforced:
  overrun          the reference ended on EOS, and the engine text strictly extends it
"""

import os

HEAD_MATCH_GATE = None   # set per run from the baselines file (calibrated from vLLM); None in calibration
USAGE_TOLERANCE = 0.02
GRANULARITY_MIN = 0.5


def common_prefix_len(a, b):
    return len(os.path.commonprefix([a, b]))


def prefix_fraction(engine_text, ref_text):
    if not ref_text:
        return 1.0 if not engine_text else 0.0
    return common_prefix_len(engine_text, ref_text) / len(ref_text)


def usage_matches(completion_tokens, ref):
    """Is usage.completion_tokens within 2% of the reference count?

    The reference count is the number of generated tokens. When the row ended on an EOS token,
    the SPEC does not say whether that token counts (vLLM counts it), so both are accepted.
    """
    if not isinstance(completion_tokens, int):
        return False
    counts = {ref["generated_tokens"]}
    if ref["eos_stopped"]:
        counts.add(ref["generated_tokens"] - 1)
    return any(abs(completion_tokens - c) <= USAGE_TOLERANCE * c for c in counts)


def check_request(req, rec, ref, engine_tokens):
    text = rec.text
    credited = min(engine_tokens, req.max_tokens)
    full = text == ref["text"]
    out = {
        "id": req.id,
        "workload": req.workload,
        "head_match": text.startswith(ref["head_text"]),
        "prefix_fraction": prefix_fraction(text, ref["text"]),
        "full_match": full,
        "credited_tokens": credited,
        "engine_tokens": engine_tokens,
        "ref_tokens": ref["generated_tokens"],
        "usage_completion_tokens": (rec.usage or {}).get("completion_tokens"),
        "usage_ok": None,
        "text_events": rec.n_text_events,
        "events_per_credited_token": (rec.n_text_events / credited) if credited else None,
        "granularity_ok": credited == 0 or rec.n_text_events >= GRANULARITY_MIN * credited,
        "finish_reason": rec.finish_reason,
        "ref_finish_reason": ref["finish_reason"],
        "overrun": bool(ref["eos_stopped"] and len(text) > len(ref["text"]) and text.startswith(ref["text"])),
    }
    if full:
        out["usage_ok"] = usage_matches(out["usage_completion_tokens"], ref)
    return out


def evaluate(requests, records, refs, tok):
    """Check every completed request; tokenization happens here, after the workload."""
    reqs = [r for r in requests if r.id in records]
    texts = [records[r.id].text for r in reqs]
    counts = [len(ids) for ids in tok.encode_batch(texts)]
    return [check_request(r, records[r.id], refs[r.id], n) for r, n in zip(reqs, counts)]


def aggregate(checks):
    n = len(checks)
    ratios = [c["events_per_credited_token"] for c in checks if c["events_per_credited_token"] is not None]
    return {
        "checked": n,
        "head_match_count": sum(c["head_match"] for c in checks),
        "head_match_rate": (sum(c["head_match"] for c in checks) / n) if n else None,
        "prefix_fraction": (sum(c["prefix_fraction"] for c in checks) / n) if n else None,
        "full_match_count": sum(c["full_match"] for c in checks),
        "usage_checked": sum(c["usage_ok"] is not None for c in checks),
        "usage_failures": sum(c["usage_ok"] is False for c in checks),
        "granularity_failures": sum(not c["granularity_ok"] for c in checks),
        "min_events_per_credited_token": min(ratios) if ratios else None,
        "overruns": sum(c["overrun"] for c in checks),
        "finish_reason_mismatches": sum(c["full_match"] and c["finish_reason"] != c["ref_finish_reason"]
                                        for c in checks),
    }


def gates(all_checks, prefix_gate, enforced, head_gate=None):
    """The correctness gates, pooled over every workload request of the run."""
    agg = aggregate(all_checks)
    g = {
        "enforced": enforced,
        "scope": "pooled over all workload requests",
        "head_match": {"value": agg["head_match_rate"], "threshold": head_gate},
        "prefix_fraction": {"value": agg["prefix_fraction"], "threshold": prefix_gate},
        "usage": {"value": agg["usage_failures"], "threshold": 0, "checked": agg["usage_checked"]},
        "streaming_granularity": {"value": agg["granularity_failures"], "threshold": 0,
                                  "min_events_per_credited_token": agg["min_events_per_credited_token"]},
    }
    hm = agg["head_match_rate"]
    pf = agg["prefix_fraction"]
    g["head_match"]["passed"] = None if head_gate is None else (hm is not None and hm >= head_gate)
    g["prefix_fraction"]["passed"] = None if prefix_gate is None else (pf is not None and pf >= prefix_gate)
    g["usage"]["passed"] = agg["usage_failures"] == 0
    g["streaming_granularity"]["passed"] = agg["granularity_failures"] == 0
    failed = [k for k in ("head_match", "prefix_fraction", "usage", "streaming_granularity")
              if g[k]["passed"] is False]
    g["failed"] = failed
    return g


def describe_gate_failures(g):
    parts = []
    for k in g["failed"]:
        v = g[k]
        if k in ("head_match", "prefix_fraction"):
            parts.append("%s %.4f < %.4f" % (k, v["value"] or 0.0, v["threshold"]))
        else:
            parts.append("%s: %d request(s) failed" % (k, v["value"]))
    return "; ".join(parts)
