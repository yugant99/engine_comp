"""Per-workload performance metrics (SPEC section 5, "Metrics").

output tok/s  total credited tokens / (last response finished - first request sent)
TTFT          first event with non-empty text - request sent
TPOT          (last text event - first text event) / (credited tokens - 1), credited >= 2 only
e2e           response finished - request sent
Percentiles use the nearest-rank method. Requests without text have no TTFT and requests with
fewer than 2 credited tokens have no TPOT; they are left out of those percentiles.
"""

import math


def nearest_rank(values, p):
    """The smallest value with at least p% of the values <= it."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(s)))
    return s[rank - 1]


def workload_metrics(records, credited):
    """records: request id -> client.Record; credited: request id -> credited tokens."""
    recs = list(records.values())
    if not recs:
        return {"requests_completed": 0}
    t0 = min(r.t_send for r in recs)
    t1 = max(r.t_end for r in recs)
    wall = t1 - t0
    total = sum(credited.get(r.id, 0) for r in recs)
    ttft = [r.t_first_text - r.t_send for r in recs if r.t_first_text is not None]
    tpot = []
    for r in recs:
        n = credited.get(r.id, 0)
        if n >= 2 and r.t_first_text is not None:
            tpot.append((r.t_last_text - r.t_first_text) / (n - 1))
    e2e = [r.t_end - r.t_send for r in recs]
    return {
        "requests_completed": len(recs),
        "credited_tokens": total,
        "wall_s": wall,
        "output_tok_s": (total / wall) if wall > 0 else None,
        "ttft_p50_s": nearest_rank(ttft, 50),
        "ttft_p95_s": nearest_rank(ttft, 95),
        "tpot_p50_s": nearest_rank(tpot, 50),
        "tpot_p95_s": nearest_rank(tpot, 95),
        "e2e_p50_s": nearest_rank(e2e, 50),
        "e2e_p95_s": nearest_rank(e2e, 95),
        "ttft_samples": len(ttft),
        "tpot_samples": len(tpot),
    }


def geometric_mean(values):
    if not values or any(v is None or v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))
