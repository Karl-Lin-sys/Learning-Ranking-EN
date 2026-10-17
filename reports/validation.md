# Execution validation

Original run: October 17, 2026 (Asia/Taipei), on Windows with Python 3.12.14, PyTorch 2.14.1+cpu, and NumPy 2.5.4. The demo uses one CPU thread, model seed 7, and synthetic-data seed 23. GPU and other operating systems were not tested.

## Commands executed

```sh
python -m unittest discover -s tests -v
python -m travel_rank.demo
```

**18 tests passed** in the original version. Its final unittest-reported duration was 4.045 seconds. Coverage includes localhost HTTP requests and responses, malformed and oversized bodies, future features, sessions crossing partitions, delayed labels, duplicate query impressions, unjudged candidates, duplicate results, zero vectors, IPS support, loss ordering, inference, A/B aggregation and immature outcomes, and monitoring summaries.

The original run reported a missing-NumPy warning. NumPy was added and the final tests rerun. This is not evidence of a memory-leak fix established by a primary issue tracker. Dependencies are recorded in constraints-tested.txt.

## Synthetic results

Training / validation / test: **220 / 70 / 70 queries**, each with 12 candidates. Every observation below is synthetic. The complete original record is [demo.json](demo.json).

| Validation method | NDCG@5 | Epochs |
|---|---:|---:|
| Pointwise | 0.963080 | 19 |
| Pairwise | 0.963025 | 35 |
| Listwise | 0.964792 | 30 |

Listwise was selected using validation only. Its test NDCG@5 was **0.955208**, MRR@5 was 1.0, and Recall@5 was 0.705082. The simple feature-0 baseline achieved NDCG@5 of 0.774024. The paired query bootstrap 95% interval for the difference was [0.141446, 0.220559]. This interval concerns a fixed synthetic model and query sample; it does not cover business outcomes or variation across training seeds.

MRR = 1 indicates an easy task. These results are neither a comparison of retrieval systems nor a production semantic-model benchmark. Training features and the search-demo adapter have different distributions. Connecting a model does not establish that it improves search.

In the synthetic bias experiment, examination is independent: the true relevance rate is 0.3, raw CTR is 0.1717, and IPS is 0.2961. Do not transfer those numbers or assumptions directly to real logs.

The interval in the synthetic A/B null experiment, which assigns both arms the same underlying CVR, crossed zero. No real-user experiment was performed. Local timing covers only six products, 100 sequential runs, and five warm-up runs. It excludes server HTTP latency, concurrent requests, large indexes, networking, and embedding generation. Environment-specific measurements remain in the JSON.

## Publication checks and outstanding work

The repository contains no certificate files, identity numbers, verification codes, customer data, or complete job posting. Checks included compilation, Git whitespace validation, and inspection of the public file list. CI configuration is provided, but the 18 tests above are local results, not a claim of successful GitHub Actions execution.

Not performed: real-traffic A/B testing, pretrained-model evaluation, ANN, Elastic/OpenSearch/Solr integration, production deployment, load or long-running memory tests, a dependency-vulnerability audit, or a legal opinion. None is presented as complete.

## Canadian English edition

This edition translates the documentation and interview explanations while retaining executable identifiers, official paper titles, source URLs, legal licence text, numerical results, and Traditional Chinese test fixtures. Translation checks and rerun results are recorded in [translation-checks.json](translation-checks.json). The original demo.json remains a historical record; the English edition's rerun is saved separately as [demo-en.json](demo-en.json).
