"""Run with python -m travel_rank.demo. All observations are synthetic."""
import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import torch
from .core import (Product, BM25, dense_search, eligible, rrf, metrics,
                   temporal_split, paired_bootstrap, ips_mean)
from .model import fit, predict
from .core import experiment_arm
from .operations import analyze_ab, log_summary


def synthetic_queries(n=360, candidates=12, seed=23):
    """O(QM) time/space. No user data; label mechanism deliberately documented."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x, y = [], []
        for _ in range(candidates):
            features = [rng.random() for _ in range(6)]
            # Known toy law, not a travel relevance model; noisy judgments.
            latent = 1.8 * features[0] + 1.5 * features[1] + .6 * features[2] + .4 * features[3] + rng.gauss(0, .25)
            grade = 3 if latent > 3.2 else 2 if latent > 2.6 else 1 if latent > 2.0 else 0
            x.append(features)
            y.append(grade)
        rows.append({"qid": f"q{i}", "session": f"s{i}", "time": i * 10,
                     "feature_time": i * 10 - 1, "label_time": i * 10 + 5,
                     "slice": "tail" if i % 4 == 0 else "head", "x": x, "y": y})
    return rows


def catalog():
    """O(1) here; invented products and hand-authored 3D vectors, not embeddings."""
    return [Product("p1", "台北 親子 飯店 游泳池", "台北", 3200, True, (.9, .2, .4)),
            Product("p2", "台北 商務 飯店 捷運", "台北", 2200, True, (.2, .9, .3)),
            Product("p3", "台北 親子 飯店 已售完", "台北", 1800, False, (.95, .1, .4)),
            Product("p4", "高雄 親子 飯店", "高雄", 2500, True, (.8, .2, .5)),
            Product("p5", "台北 度假 住宿 兒童 遊戲室", "台北", 2800, True, (.95, .1, .5)),
            Product("p6", "台北 豪華 飯店", "台北", 9000, True, (.6, .5, .8))]


def search(query, vector, destination="台北", budget=4000, k=3, model_bundle=None):
    """End-to-end toy pipeline. O(T+ND+N log N+UMFH) with bounded toy N."""
    if not isinstance(query, str) or not query.strip() or len(query) > 256 or not 1 <= k <= 20:
        raise ValueError("query length 1..256 and k 1..20 required")
    products = catalog()
    pool = eligible(products, destination, budget)
    index = BM25(products)
    lexical = index.search(query, {p.id for p in pool}, 20)
    dense = dense_search(vector, pool, 20)
    candidates = rrf([lexical, dense], 20)
    ranked = candidates
    if model_bundle and candidates:
        # Features are illustrative adapters, not calibrated to the synthetic training law.
        lookup = {p.id: p for p in pool}
        lr, dr = {p: i for i, p in enumerate(lexical)}, {p: i for i, p in enumerate(dense)}
        features = [[1 / (1 + lr[p]) if p in lr else 0, 1 / (1 + dr[p]),
                     1 - lookup[p].price / max(budget or 1, 1), float(lookup[p].kind == "hotel"), 0., 0.]
                    for p in candidates]
        values = predict(*model_bundle, [features])[0]
        ranked = [pid for pid, score in sorted(zip(candidates, values), key=lambda z: (-z[1], z[0]))]
    # Final inventory check; real deployment must recheck against authoritative live inventory.
    allowed = {p.id for p in eligible(products, destination, budget)}
    final = [pid for pid in ranked if pid in allowed][:k]
    return {"synthetic": True, "lexical": lexical, "dense": dense, "candidates": candidates,
            "results": final, "model_reranked": model_bundle is not None,
            "note": "hand-authored vectors; no pretrained semantic encoder"}


def evaluate(rows, scores, k=5):
    """O(QM log M) time, O(QM) space; macro average across queries."""
    results = []
    for row, values in zip(rows, scores, strict=True):
        ranking = sorted(range(len(values)), key=lambda i: (-values[i], i))
        results.append(metrics(ranking, dict(enumerate(row["y"])), k))
    return {"queries": len(rows), "no_relevant_queries": sum(not r["has_relevant"] for r in results),
            **{name: statistics.mean(r[name] for r in results) for name in ("ndcg", "mrr", "recall")}}, results


def run(seed=7):
    """Train three losses, select using validation only, evaluate held-out once."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, valid, test = temporal_split(synthetic_queries(), 2200, 2900, 3600)
    fitted, validation = {}, {}
    for method in ("pointwise", "pairwise", "listwise"):
        model, mean, scale, history = fit(train, valid, method, seed=seed)
        fitted[method] = (model, mean, scale)
        summary, _ = evaluate(valid, predict(model, mean, scale, [r["x"] for r in valid]))
        validation[method] = {**summary, "epochs": len(history), "best_validation_loss": min(history)}
    selected = max(validation, key=lambda m: validation[m]["ndcg"])
    bundle = fitted[selected]
    scores = predict(*bundle, [r["x"] for r in test])
    summary, per_query = evaluate(test, scores)
    baseline, baseline_q = evaluate(test, [[x[0] for x in r["x"]] for r in test])
    slices = {}
    for name in ("head", "tail"):
        chosen = [i for i, r in enumerate(test) if r["slice"] == name]
        slices[name], _ = evaluate([test[i] for i in chosen], [scores[i] for i in chosen])
    # Known independent examination simulation: click = relevance * examination.
    rng = random.Random(91)
    outcomes, props = [], []
    for _ in range(10000):
        p = rng.choice([1., .5, .2])
        outcomes.append(int(rng.random() < .3 and rng.random() < p))
        props.append(p)
    latency = []
    for _ in range(5):
        search("親子 飯店", (.9, .1, .4), model_bundle=bundle)
    for _ in range(100):
        start = time.perf_counter()
        result = search("親子 飯店", (.9, .1, .4), model_bundle=bundle)
        latency.append((time.perf_counter() - start) * 1000)
    latency.sort()
    ab_units = [{"unit": str(i), "arm": experiment_arm(str(i)),
                 "converted": int(rng.random() < .1), "matured": True} for i in range(1000)]
    return {"synthetic_only": True, "seed": seed, "python": platform.python_version(),
            "torch": torch.__version__, "device": "cpu", "threads": 1,
            "split_sizes": [len(train), len(valid), len(test)], "validation": validation,
            "selected_on_validation": selected, "test_selected": summary, "test_baseline_feature0": baseline,
            "test_slices": slices,
            "paired_bootstrap": paired_bootstrap([a["ndcg"] - b["ndcg"] for a, b in zip(per_query, baseline_q)]),
            "bias_simulation": {"known_true_relevance": .3, "raw_ctr": statistics.mean(outcomes), **ips_mean(outcomes, props)},
            "ab_null_simulation": analyze_ab(ab_units),
            "monitor_example": log_summary([{"latency_ms": v, "results": len(result["results"]), "error": False} for v in latency]),
            "retrieval_example": result,
            "retrieval_qrels_example": metrics(result["results"], {"p1": 3, "p2": 0, "p5": 3}, 3),
            "latency_ms_6_products_local_sequential": {"p50": statistics.median(latency), "p95": latency[94], "p99": latency[98]},
            "limitations": ["No real traffic, business uplift, pretrained model, or ANN benchmark.",
                            "Synthetic training and search features differ; search integration is plumbing only.",
                            "One seed; bootstrap interval is conditional on the trained model.",
                            "No evidence of production throughput or absence of memory leaks."]}


def main():
    """CLI: run complexity plus O(output bytes) serialization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/demo.json")
    args = parser.parse_args()
    report = run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "selected": report["selected_on_validation"],
                      "test": report["test_selected"], "synthetic_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
