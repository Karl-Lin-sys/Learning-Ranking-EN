# Learning-Ranking-EN: Travel Search and Ranking Lab

A Python prototype that breaks travel search into **retrieve → combine candidates → rank → evaluate properly**, with explanations suitable for a job interview. It is not a company's production system, a guarantee of employment, or evidence of improvements for real users.

**Synthetic data only.** This repository contains no customer logs, credentials, personal identifiers, complete job postings, or third-party model weights. Strong results here do not guarantee similar results in travel search.

This Canadian English edition translates [Learning-Ranking](https://github.com/Karl-Lin-sys/Learning-Ranking/tree/644c02ab97da4c3c61ce97059d395d56205fcba3). Research findings, assumptions, numerical results, and limitations retain their original meaning. Traditional Chinese queries and product names remain as test data for the Taiwan travel scenario; prices remain in New Taiwan dollars (TWD).

## Quick start

Use Python 3.12 and a CPU. Create and activate a virtual environment:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m unittest discover -s tests -v
python -m travel_rank.demo
python -m travel_rank.api
```

Training and evaluation results are written to `reports/demo.json`. The API binds only to `127.0.0.1:8000`. In another terminal:

```python
import json
from urllib.request import Request, urlopen

request = Request(
    'http://127.0.0.1:8000/search',
    data=json.dumps({'query': '親子 飯店', 'vector': [0.9, 0.1, 0.4],
                     'destination': '台北', 'budget': 4000, 'k': 3}).encode(),
    headers={'Content-Type': 'application/json'},
)
with urlopen(request, timeout=5) as response:
    print(json.load(response))
```

The query means “family-friendly hotel”; the destination is Taipei. This three-dimensional vector is a hand-crafted example, not a natural-language embedding. The API returns BM25 + exact cosine + RRF results. The demo experiment also connects a trained MLP. Synthetic training features and search features have different distributions, so this connection is not evidence of search quality.

## Implementation and reading guide

| Topic | File |
|---|---|
| Technology comparison, three causal hypotheses, evaluation and operational design | [Design](docs/design.md) |
| Child-friendly explanation and translated interview scripts | [Interview guide](docs/interview.md) |
| Peer-review status, official APIs, GitHub stars, and Taiwanese legal sources | [Sources](docs/sources.md) |
| Time and space complexity for all application functions | [Big-O](docs/complexity.md) |
| Filters, BM25, cosine, RRF, NDCG, MRR, recall, temporal splits, and IPS | [core.py](travel_rank/core.py) |
| Pointwise, pairwise, and listwise losses; training-only normalization; early stopping | [model.py](travel_rank/model.py) |
| Synthetic experiments, retrieval, ranking, bootstrap, and local timing | [demo.py](travel_rank/demo.py) |
| Bounded requests, health checks, and search API | [api.py](travel_rank/api.py) |
| A/B aggregation, SRM, CVR, and search-log monitoring | [operations.py](travel_rank/operations.py) |
| Executed checks and limitations | [Validation record](reports/validation.md) |

## What this project establishes

Tests cover behaviour on synthetic data, leakage detection, metric edge cases, and the API contract. There are 220 training queries, 70 validation queries, and 70 test queries. Model selection uses validation data only; the test set is not used to reselect the model.

Real-traffic A/B testing, Elastic clusters, pretrained encoders, GPU load testing, distributed streams, authentication, TLS, and production deployment have not been evaluated. This project does not establish CVR improvements, the absence of memory leaks, the elimination of all bias, or three years of professional experience. Explain the limitations of synthetic tests, including MRR saturation.

BM25 and RRF are not presented as new inventions. Research candidates were checked against peer-reviewed papers published on or after October 17, 2023, and APIs against official documentation. See [sources and publication scope](docs/sources.md) for copyright and personal-information considerations.
