"""Auditable retrieval/evaluation utilities. Complexity: docs/complexity.md."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import random
import re


def tokens(text):
    """O(L) time/space; simple CJK-character tokenizer, not linguistic segmentation."""
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.casefold())


@dataclass(frozen=True)
class Product:
    id: str
    title: str
    destination: str
    price: float
    available: bool
    vector: tuple
    kind: str = "hotel"


def eligible(products, destination=None, budget=None):
    """O(N) time/space. Budget and price are TWD per synthetic product."""
    if budget is not None and (not math.isfinite(budget) or budget < 0):
        raise ValueError("budget must be finite and nonnegative")
    return [p for p in products if p.available
            and (destination is None or p.destination == destination)
            and (budget is None or p.price <= budget)]


class BM25:
    def __init__(self, products, k1=1.2, b=0.75):
        """O(T) build time/space; corpus statistics fixed at index construction."""
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        if len({p.id for p in products}) != len(products):
            raise ValueError("duplicate product IDs")
        self.k1, self.b = k1, b
        self.length, self.postings = {}, defaultdict(list)
        for p in products:
            terms = tokens(p.title)
            self.length[p.id] = len(terms)
            for term, count in Counter(terms).items():
                self.postings[term].append((p.id, count))
        self.n = len(products)
        self.avg = sum(self.length.values()) / max(1, self.n) or 1.0

    def search(self, query, allowed, k):
        """O(L+P+M log M) time, O(M+L) temporary space; exact filtered search."""
        if k < 1:
            raise ValueError("k must be positive")
        scores = defaultdict(float)
        for term in set(tokens(query)):
            posting = self.postings.get(term, ())
            idf = math.log1p((self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for pid, tf in posting:
                if pid in allowed:
                    norm = self.k1 * (1 - self.b + self.b * self.length[pid] / self.avg)
                    scores[pid] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores, key=lambda pid: (-scores[pid], pid))[:k]


def dense_search(query_vector, products, k):
    """Exact cosine baseline: O(ND+N log N) time, O(N) auxiliary space."""
    if k < 1 or not query_vector or not all(math.isfinite(v) for v in query_vector):
        raise ValueError("invalid query vector or k")
    qnorm = math.sqrt(sum(v * v for v in query_vector))
    if not qnorm:
        raise ValueError("zero query vector")
    scored = []
    for p in products:
        if len(p.vector) != len(query_vector) or not all(math.isfinite(v) for v in p.vector):
            raise ValueError("invalid product vector")
        norm = math.sqrt(sum(v * v for v in p.vector))
        if not norm:
            raise ValueError("zero product vector")
        score = sum(a * b for a, b in zip(query_vector, p.vector)) / (qnorm * norm)
        scored.append((p.id, score))
    return [pid for pid, _ in sorted(scored, key=lambda x: (-x[1], x[0]))[:k]]


def rrf(rankings, k=10, constant=60):
    """O(R+U log U) time, O(U) space; repeated IDs per route counted once."""
    if k < 1 or constant < 1:
        raise ValueError("k and constant must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for pid in ranking:
            if pid in seen:
                continue
            seen.add(pid)
            scores[pid] += 1 / (constant + len(seen))
    return sorted(scores, key=lambda pid: (-scores[pid], pid))[:k]


def metrics(ranking, qrels, k=10):
    """O(M+J log J) time, O(M+J) space. Complete judged eligible pool required."""
    if k < 1 or len(set(ranking)) != len(ranking):
        raise ValueError("invalid k or duplicate results")
    if any(not math.isfinite(v) or not 0 <= v <= 4 for v in qrels.values()):
        raise ValueError("relevance grades must be in [0,4]")
    if any(pid not in qrels for pid in ranking):
        raise ValueError("unjudged result: do not silently treat as irrelevant")
    top = ranking[:k]
    gains = [2 ** qrels[pid] - 1 for pid in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    positive = {pid for pid, g in qrels.items() if g > 0}
    return {"ndcg": dcg / idcg if idcg else 0.0,
            "mrr": next((1 / (i + 1) for i, pid in enumerate(top) if qrels[pid] > 0), 0.0),
            "recall": len(set(top) & positive) / len(positive) if positive else 0.0,
            "has_relevant": bool(positive), "zero_results": not ranking}


def temporal_split(rows, train_end, valid_end, observed_until):
    """O(N) expected time/space; discard sessions crossing partitions and immature labels.

    Times are integer synthetic hours. Each row is one whole query impression.
    feature_time includes every aggregate's latest contributing event.
    """
    if not train_end < valid_end < observed_until:
        raise ValueError("cutoffs must increase")
    buckets, session_parts, seen = [[], [], []], defaultdict(set), set()
    for row in rows:
        if row["qid"] in seen:
            raise ValueError("duplicate query impression")
        seen.add(row["qid"])
        if row["feature_time"] > row["time"]:
            raise ValueError("future feature leakage")
        if row["label_time"] < row["time"]:
            raise ValueError("invalid label timestamp")
        i = 0 if row["time"] < train_end else 1 if row["time"] < valid_end else 2
        cutoff = (train_end, valid_end, observed_until)[i]
        session_parts[row["session"]].add(i)
        if row["label_time"] < cutoff and row["time"] < cutoff:
            buckets[i].append(row)
    crossing = {s for s, parts in session_parts.items() if len(parts) > 1}
    return tuple([r for r in bucket if r["session"] not in crossing] for bucket in buckets)


def ips_mean(outcomes, propensities, clip=20.0):
    """O(N) time/space. Observed/examined positives; not a generic click debiaser."""
    if not outcomes or len(outcomes) != len(propensities) or not math.isfinite(clip) or clip < 1:
        raise ValueError("invalid IPS input")
    if any(y not in (0, 1) for y in outcomes) or any(not 0 < p <= 1 for p in propensities):
        raise ValueError("binary outcomes and positive propensities required")
    weights = [min(clip, 1 / p) for p in propensities]
    return {"ips": sum(w * y for w, y in zip(weights, outcomes)) / len(outcomes),
            "ess": sum(weights) ** 2 / sum(w * w for w in weights),
            "clipped_fraction": sum(1 / p > clip for p in propensities) / len(propensities)}


def paired_bootstrap(deltas, repetitions=1000, seed=7):
    """O(BQ+B log B) time, O(B+Q) space. Independent query/session deltas only."""
    if not deltas or repetitions < 2 or not all(math.isfinite(x) for x in deltas):
        raise ValueError("finite nonempty deltas required")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                   for _ in range(repetitions))
    return {"mean_delta": sum(deltas) / len(deltas),
            "ci95": [means[int(.025 * repetitions)], means[min(repetitions - 1, int(.975 * repetitions))]]}


def experiment_arm(unit, experiment="travel-v1"):
    """O(L) time/space; synthetic unit ID only. Stable across processes."""
    digest = hashlib.sha256((experiment + "\0" + unit).encode()).digest()
    return "A" if int.from_bytes(digest[:8], "big") % 2 == 0 else "B"
