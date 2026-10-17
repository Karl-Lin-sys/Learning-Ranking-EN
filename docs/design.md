# Technology selection and validation design

## Recommendation

Start with **availability filters + BM25** as the baseline. Add an independent dense retrieval route, combine results with RRF, and then evaluate a small PyTorch learning-to-rank model. Evaluate LLM reranking and RAG in separate experiments after comparing search quality and cost. This is a design judgement based on the job requirements, not a diagnosis of a measured company system.

The implementation is deliberately small for teaching: up to 20 results per retrieval route, 20 MLP candidates, 20 API results, a 256-character query, and a 4,096-byte request. These limits do not establish production performance at scale.

## Technology comparison

| Technology | Strengths and rationale | Structural limitations and checks |
|---|---|---|
| BM25 / inverted index | Explainable lexical matching and a low-cost baseline | Can miss synonyms and natural-language intent. This implementation tokenizes CJK text by character, without full linguistic segmentation |
| BGE-M3 candidate | Supports multilingual and multiple retrieval modes [P1] | Check weights, licensing, Traditional Chinese, place names, and training-data overlap. Not downloaded here |
| Exact cosine | Provides a reference set for checking ANN retrieval | Scans all products in O(ND). Hand-crafted vectors test integration only |
| Faiss / ANN candidate | Popular official implementation for scaling candidate retrieval | Measure recall–latency trade-offs, updates, deletion, and attribute filtering. Do not assume ANN is always O(log N) |
| RRF | Combines different score scales through ranks | Avoid duplicate contributions within a route. Cannot recover products missed by truncated retrieval. The constant 60 is a fixed baseline, not an optimum |
| Pointwise | Small baseline using independent binary cross-entropy | Binary labels discard graded relevance. Class imbalance can inflate accuracy |
| Pairwise | Learns to score better products above worse products within a query | All pairs cost O(M²). A click versus a non-click is not automatically a true preference |
| Listwise | Learns a gain distribution within each query using softmax | Depends on list composition. This loss is neither NDCG itself nor ListT5 |
| ListT5 / FIRST research candidates | Study list comparison and inference efficiency [P2, P3] | Distinguish papers from implementations meeting the high-star criterion. Defer adoption until GPU cost, Traditional Chinese, and ranking robustness are evaluated |
| PyTorch | Selected training and inference framework | Bounded batches, eval, inference_mode, and pinned dependencies. Speed and memory depend on the environment |
| TensorFlow Ranking | Comparison option for teams with existing TensorFlow assets | The official repository is archived; this does not mean TensorFlow itself is discontinued |
| OpenCV | Image-processing candidate if photo search is added | Not needed for this text-focused prototype. Installing every tool in the job posting is not the objective |

No live Elasticsearch, OpenSearch, or Solr system was provided, so no comparative performance ranking is claimed. Select an engine based on existing operations, analyzers, filtered ANN, index updates, licensing, observability, and p95/p99 under equivalent loads. This project uses official Elastic specifications as design references and implements a separate small baseline. [Sources](sources.md)

## Hypothesis 1: Selection and presentation bias turn position into apparent popularity

**Hypothesis:** Products near the top receive more attention and clicks. Features derived from earlier rankings further favour those products. Logs alone may not distinguish low interest from lack of observation. [P4]

**Evidence to inspect:** All candidates in each impression, display position, actual exposure, logging-policy version, position and relevance within the same intent, device, and time. Do not evaluate this using a dataset containing clicks only. Where lawful, introduce a small amount of randomized exposure and record assignment probabilities.

**Solution 1:** Establish a human-judged relevance baseline and correct for known or identifiable observation probabilities. In a simple position-based model, let C = R × E. When the examination probability p is correctly specified and the model's assumptions hold, E[C/p] = E[R]. Confounding, trust bias, or misleading clicks can invalidate this relationship. Do not equate position-specific CTR with p. [P4, P7]

`ips_mean` demonstrates this restricted assumption in a synthetic experiment. Clipping 1/p reduces variance but sacrifices unbiasedness. Report effective sample size (ESS) and the clipping fraction, and assess sensitivity to settings such as 5, 10, and 20 using training and validation data. Rejecting p = 0 does not create support for products that were never exposed. Doubly robust (DR) methods also require appropriate propensity or outcome models and procedures such as cross-fitting; DR is not implemented here. [P5, P7]

**Falsification and adoption criteria:** Check agreement with randomized or independent evaluation, confidence intervals for improvements, and weight variance. Success in the simulation does not demonstrate successful correction in production.

## Hypothesis 2: Future information and model selection inflate offline results

**Hypothesis:** Random row splits mix the same session across partitions; features include bookings from the test period; negative labels are finalized before bookings can occur; or encoder pretraining overlaps with evaluation data. Repeated model selection on the test set also overfits that set. Pretraining contamination in [P6] motivates the concern; the temporal rules below are this implementation's audit design.

**Solution 2:** Split whole query impressions chronologically and remove sessions crossing a boundary from every partition. Require feature_time <= impression_time and label_time < the relevant partition cutoff. Fit preprocessing means and standard deviations using training data only. Use validation loss for early stopping, validation NDCG to select the method, and evaluate the selected model on the test set once. Fix model and data-generation seeds.

The checks treat timestamps as trustworthy metadata. They cannot automatically detect a future aggregate carrying an incorrect feature_time. In production, audit point-in-time join SQL, contributing event times, and duplicate IDs. Shared product IDs are not necessarily leakage when evaluating future searches for existing products. Add independent holdouts for unseen users, unseen products, and new seasons. If an encoder's training set is unknown, do not claim that contamination has been eliminated.

**Falsification and adoption criteria:** Compare results before and after chronological splitting, entity holdouts, and deduplication. If the test results influence a change, reserve a new untouched period. Add rolling-window experiments and multiple seeds; do not claim business improvements before results are stable.

## Hypothesis 3: Retrieval omissions and the evaluation population hide behind reranking gains

**Hypothesis:** Products matching the destination or date never enter the candidate set. Computing IDCG from candidates alone fails to penalize missing relevant products and can inflate NDCG. Popular queries dominate aggregates, while saturated MRR or CTR hides quality differences. [P1, P2] suggest retrieval and reranking candidates; they do not establish that these problems occur at the company.

**Solution 3:** Apply hard constraints—dates, origin, party size, availability, currency, and price units—to each retrieval route, then combine the lexical and dense candidate sets. First measure Recall@K using judgements over the entire eligible corpus. Then measure NDCG@K and MRR@K using the same pool and its IDCG. Do not relabel omitted products as irrelevant. This implementation rejects unjudged results. Queries with no relevant products receive zero, with their count and denominator reported explicitly.

**Falsification and adoption criteria:** Run ablations on the same queries: BM25 only → dense only → RRF → reranking of the same candidates. Report route-specific candidate recall, zero-result rates, head/tail queries, language, product type, device, and new products separately. Synthetic head/tail labels use `qid % 4`; they do not reproduce actual frequency or long-tail behaviour. Report the wider uncertainty associated with small slices.

For class imbalance, do not rely on accuracy alone. Where relevant, assess precision–recall curves and calibration at the original production prevalence. If training negatives are downsampled, retain selection probabilities and do not downsample the evaluation set. Do not automatically label every non-click as negative. The best relevance ranking may differ from the best booking-probability ranking.

## Online evaluation and operations

1. Assign A/B variants using a stable unit, such as a user. `experiment_arm` demonstrates deterministic assignment of synthetic IDs. Hashing does not guarantee anonymization.
2. Define the primary metric as the proportion of assigned users who book within a specified period. Preserve the intention-to-treat (ITT) population and retain users who did not click. Also examine net conversion after cancellations, prices, and exposure.
3. Preregister the minimum detectable effect (MDE), sample size, duration, alpha, power, primary metric, guardrails, and stopping rule. Do not repeatedly inspect ordinary p-values during a fixed-horizon experiment and treat them as valid stopping decisions.
4. `analyze_ab` aggregates matured binary outcomes per unit and computes bootstrap intervals and an approximate 50:50 sample-ratio-mismatch (SRM) check. It does not solve multiple comparisons, sparse all-zero outcomes, network effects, interference, or seasonality. Do not launch automatically based on a confidence interval alone.
5. Monitor p95/p99, error rates, zero-result rates, cancellation rates, unavailable-product rates, missing features, distribution changes, and index freshness. `log_summary` covers only part of this over a fixed window. Set thresholds using existing service-level objectives (SLOs) and measurements.

These are explicit statistical and operational design choices, not a reproduction of a particular paper's online experiment. No real-customer A/B results are claimed.

For production expansion, attach versions and event times to product events: idempotent upsert → build a new index → validate quality → switch the alias, with rollback available. Key query caches by normalized query, filters, locale, and inventory/model/index versions, with capacity limits and time to live (TTL). Recheck price and availability against the authoritative system before responding. These distributed mechanisms are design proposals, not implemented features.

The demo HTTPServer processes requests serially and lacks authentication, TLS, horizontal scaling, a gateway, and rate limiting. In production, load the model once at worker startup, bound queues and deadlines, and fall back to RRF if reranking times out. Limit candidates, tokens, and batches instead of creating unlimited M² pairs. Ground RAG responses in retrieved product IDs and sources, use current retrieved prices rather than generating them, and do not execute instructions embedded in retrieved documents.

## Memory and performance

Training history retains Python floats, not computation graphs. Inference uses both inference_mode and eval. Training uses zero_grad(set_to_none=True) and bounded batches. Only one best checkpoint is retained. The demo rebuilds its small index for each search, and local timing includes that cost. Production should reuse an immutable index for each update version.

The absence of memory leaks has not been established. Long-running load tests, post-warm-up RSS/VRAM trends, concurrency, and out-of-memory recovery remain untested. Distinguish cached memory from unreachable leaked memory. No claim is made that clearing caches on every request improves speed.

Reference IDs are listed in [sources.md](sources.md). Complexity is documented in [complexity.md](complexity.md).
