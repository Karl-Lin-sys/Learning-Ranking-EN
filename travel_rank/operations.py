"""Small operational diagnostics; inference is conditional on documented assumptions."""
import math
import random
import statistics


def analyze_ab(units, repetitions=1000, seed=7):
    """O(BN+B log B) time/O(N+B) space; ONE matured binary outcome per assigned unit.

    Includes non-searching/non-converting units (intention-to-treat). 50:50 allocation.
    SRM p uses a large-sample chi-square approximation; no automated launch decision.
    """
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    arms, seen = {"A": [], "B": []}, set()
    for row in units:
        if row["unit"] in seen or row["arm"] not in arms or row["converted"] not in (0, 1):
            raise ValueError("unique units, A/B arms and binary outcomes required")
        if not row["matured"]:
            raise ValueError("conversion window must close before analysis")
        seen.add(row["unit"])
        arms[row["arm"]].append(row["converted"])
    a, b = arms["A"], arms["B"]
    if min(len(a), len(b)) < 10:
        raise ValueError("at least 10 units per arm required; still may be underpowered")
    rng = random.Random(seed)
    deltas = sorted(statistics.mean(rng.choices(b, k=len(b))) - statistics.mean(rng.choices(a, k=len(a)))
                    for _ in range(repetitions))
    expected = (len(a) + len(b)) / 2
    chi2 = ((len(a) - expected) ** 2 + (len(b) - expected) ** 2) / expected
    return {"n_A": len(a), "n_B": len(b), "cvr_A": statistics.mean(a), "cvr_B": statistics.mean(b),
            "absolute_delta": statistics.mean(b) - statistics.mean(a),
            "ci95": [deltas[int(.025 * repetitions)], deltas[min(repetitions - 1, int(.975 * repetitions))]],
            "srm_p_approx": math.erfc(math.sqrt(chi2 / 2)),
            "caution": "Fixed horizon only; no significance claim for repeated peeking or sparse all-zero data."}


def log_summary(events):
    """O(N log N) time/O(N) space, fixed-window request logs; no raw query text."""
    if not events:
        raise ValueError("empty monitoring window")
    latency = []
    for event in events:
        value = event["latency_ms"]
        if not math.isfinite(value) or value < 0 or event["results"] < 0:
            raise ValueError("invalid event")
        latency.append(value)
    latency.sort()
    return {"requests": len(events),
            "zero_result_rate": sum(e["results"] == 0 for e in events) / len(events),
            "error_rate": sum(bool(e["error"]) for e in events) / len(events),
            "p95_ms": latency[max(0, math.ceil(.95 * len(latency)) - 1)]}
