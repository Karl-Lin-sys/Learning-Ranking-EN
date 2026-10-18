# Learning-Ranking-EN: Travel Search and Ranking Lab

A Python prototype that breaks travel search into **retrieve → combine candidates → rank → evaluate properly**, with explanations suitable for a job interview. It is not a company's production system, a guarantee of employment, or evidence of improvements for real users.

**Synthetic data only.** This repository contains no customer logs, credentials, personal identifiers, complete job postings, or third-party model weights. Strong results here do not guarantee similar results in travel search.

This Canadian English edition translates [Learning-Ranking](https://github.com/Karl-Lin-sys/Learning-Ranking/tree/644c02ab97da4c3c61ce97059d395d56205fcba3). Research findings, assumptions, numerical results, and limitations retain their original meaning. Traditional Chinese queries and product names remain as test data for the Taiwan travel scenario; prices remain in New Taiwan dollars (TWD).

## Quick start

Use Python 3.12 and a CPU. Run commands from the repository folder containing
`requirements.txt`, `constraints-tested.txt`, `tests`, and `travel_rank`.
Running from your home folder will produce missing-file or missing-module errors.

### Windows PowerShell

For a new checkout, first run:

```powershell
git clone https://github.com/Karl-Lin-sys/Learning-Ranking-EN.git
Set-Location .\Learning-Ranking-EN
```

If you already have a checkout or extracted ZIP, use `Set-Location` with that
folder's actual path instead. Do not clone into an existing checkout.
Then run these commands one at a time, stopping if a command fails:

```powershell
Get-Item .\requirements.txt, .\constraints-tested.txt, .\tests, .\travel_rank
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m travel_rank.demo
.\.venv\Scripts\python.exe -m travel_rank.api
```

These commands call the virtual environment's Python directly. Activation and
changes to PowerShell execution policy are unnecessary. Lines beginning with
`#` are comments and do not activate anything.

### macOS / Linux

For a new checkout:

```sh
git clone https://github.com/Karl-Lin-sys/Learning-Ranking-EN.git
cd Learning-Ranking-EN
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c constraints-tested.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m travel_rank.demo
.venv/bin/python -m travel_rank.api
```

For an existing checkout, first change to its folder and skip cloning.
The API stays running in the terminal; press **Ctrl+C** to stop it. In another
PowerShell window, check it with `Invoke-RestMethod http://127.0.0.1:8000/health`.

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
