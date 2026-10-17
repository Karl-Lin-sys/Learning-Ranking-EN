# Interview practice

## Explaining it to a five-year-old

“A travel shop has lots of cards. First, we collect cards for things in Taipei that cost no more than TWD 4,000 and are still available. One helper looks for matching names. Another looks for similar things. We put their cards together, keep only one copy of each card, and arrange them so the closest matches come first.”

“The card at the top is easy to see, so people click it more often. That does not always mean it is the best card. We also think about how easy it was to see. Getting full marks after looking at the answers does not show how much you know. That is why we keep practice questions separate from the final test.”

“The cards and customers in this project are made up. We have not yet tested whether it helps in a real shop.”

## 60-second introduction: translated from the Japanese script

“I built a Python project that separates travel search into candidate retrieval and ranking. It checks availability constraints first, then combines BM25 and vector-search results using RRF. In PyTorch, I compared pointwise, pairwise, and listwise methods. Model selection uses validation data only, with checks for sessions crossing partitions and features containing future information. I report recall, MRR, long-tail slices, and confidence intervals alongside NDCG. Because display position can bias clicks, I also built a synthetic experiment with known observation probabilities. So far, this is validation on synthetic data, not evidence of real CVR improvements. In practice, I would first confirm your log definitions and SLOs, then validate changes through a small A/B experiment.”

## Alternative 60-second introduction: translated from the Traditional Chinese script

“I divide travel search into retrieval, fusion, ranking, and validation. I first check destination, budget, and availability, then retrieve candidates through keyword and vector routes and combine them with RRF. I use PyTorch to compare pointwise, pairwise, and listwise approaches. Data is split chronologically, normalization parameters are calculated from training data only, and the model is selected on the validation set. Tests reject future features and duplicate query impressions. Evaluation covers NDCG, recall, MRR, and separate slices. Since exposure position affects clicks, I also created a synthetic experiment with known exposure probabilities. This project validates the workflow rather than performance on production traffic. On joining a team, I would first clarify logs, conversion definitions, and latency SLOs, then validate changes gradually through experiments that can be rolled back.”

## Common questions

| Question | Points to cover |
|---|---|
| Why not start with a large LLM? | Without a baseline and a judged reference set, we cannot measure whether gains justify the cost. Check candidate recall first |
| Why three losses? | Demonstrate the difference between judging one product, comparing two products, and comparing an entire list. Do not cherry-pick the highest score |
| Does higher NDCG mean more bookings? | Relevance and bookings differ. Price, availability, and seasonality matter. Use a real A/B primary metric and guardrails |
| Is a non-click a negative example? | The result may never have been seen. Check exposure, position, and the observation window |
| Do more stars mean the code is correct? | Stars indicate popularity. Pinned environments, reruns, data definitions, official checkpoints, and evaluation settings still matter |
| Why is MRR = 1 not the finish line? | This easy synthetic task places a relevant result first. MRR does not capture every lower-ranked ordering error or missed product |
| Do you have production experience? | Do not present this project as production experience. Distinguish actual professional work from what you learned here |
| What comes next? | Obtain properly authorized judgements and logs, then evaluate pretrained encoders, filtered retrieval, same-candidate ablations, and load tests |

## Five-minute demonstration

1. Run `python -m unittest discover -s tests -v` to demonstrate checks for invalid input, leakage, and metrics.
2. Run `python -m travel_rank.demo` and open the validation-based selection and held-out results.
3. Explain the synthetic-data flag, splits, MRR saturation, and IPS assumptions in `reports/demo.json`.
4. Run `python -m travel_rank.api`, set the budget to zero, and show that it correctly returns an empty result.
5. Identify the remaining production work yourself and explain the order in which you would validate it.

## Qualifications and supporting documents

Course completion, attendance, enrolment, a conferred degree, and professional experience are different facts. Do not claim a master's degree without evidence that it was awarded. A university's non-credit online course does not establish enrolment in a degree program or a degree from that university. If discussing CPE or similar results, stay within what the certificate actually supports and do not publish identity or student numbers.

This public guide contains no original certificates, verification codes, or lists of names. It cannot guarantee employment. A requirement for three or more years of experience is not automatically met by this new project or by course completion.
