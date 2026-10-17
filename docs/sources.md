# Primary sources and scope

Original verification date: October 17, 2026 (Asia/Taipei). The three-year technology-selection window starts on October 17, 2023; the five-year causal-analysis window starts on October 17, 2021. Blogs, forums, and Wikipedia appearing in search results were not used as evidence. An arXiv posting alone was not treated as proof of peer review.

This English edition preserves the original source checks and dated snapshots; translation does not constitute a new literature review.

## Peer-reviewed research

| ID | Paper and venue | What it supports here |
|---|---|---|
| P1 | [M3-Embedding, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.137/) | A multilingual dense/sparse/multi-vector retrieval candidate. Published in Findings, not the ACL main track; supplementary if the requirement strictly excludes Findings |
| P2 | [ListT5, ACL 2024 main](https://aclanthology.org/2024.acl-long.125/) | Research on comparing entire lists for reranking, using FiD and a tournament procedure. This repository's listwise loss is not a ListT5 reproduction |
| P3 | [FIRST, EMNLP 2024 main](https://aclanthology.org/2024.emnlp-main.491/) | A candidate for reducing LLM reranking inference cost. Relative performance remains undecided until tested on the target data |
| P4 | [Unbiased Learning-to-Rank Needs Unconfounded Propensity Estimation, SIGIR 2024](https://doi.org/10.1145/3626772.3657772) / [author's paper page](https://www.cse.lehigh.edu/~brian/pubs/2024/SIGIR/) | Relevance can confound position and clicks; position-specific CTR is not automatically propensity |
| P5 | [Fine-Grained Dynamic Framework for Bias-Variance Joint Optimization on Data Missing Not at Random, NeurIPS 2024 main](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc75254bc4b8b42f401d0ab5d6e9aa4b-Abstract-Conference.html) | Small propensities can produce large variance in unbiased estimators. Clipping involves a bias–variance trade-off |
| P6 | [Investigating Data Contamination in Modern Benchmarks for Large Language Models, NAACL 2024 main](https://aclanthology.org/2024.naacl-long.482/) | Motivates auditing pretraining/evaluation overlap. It does not directly establish a temporal-splitting method for travel logs |
| P7 | [Doubly Robust Estimation for Correcting Position Bias in Click Feedback for Unbiased Learning to Rank, ACM TOIS 2023](https://doi.org/10.1145/3569453) | Theoretical background for considering DR. Used within the five-year analysis window, not counted as a recent conference paper within the three-year window |

ACL, EMNLP, NAACL, and SIGIR are included among the relevant leading venues because the task concerns text retrieval. Unrelated computer-vision papers were not added merely to cite CVPR. Results from these studies are not extrapolated into improvement percentages for Taiwanese travel search.

## Official implementations and GitHub star snapshot

Rounded counts displayed on GitHub when checked on October 17, 2026. This is an ordering within a selected set of relevant official implementations, not an exhaustive ranking of GitHub. Stars indicate popularity, not reproducibility, quality, or causal superiority.

| Official repository | Displayed stars | Decision |
|---|---:|---|
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | Approximately 104.8k | Selected as the implementation framework |
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | Approximately 41.2k | Candidate for large-scale nearest-neighbour retrieval. This implementation uses exact cosine, not Faiss |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | Approximately 12.3k | Multilingual encoder candidate. No model weights downloaded here |
| [tensorflow/ranking](https://github.com/tensorflow/ranking) | Approximately 2.8k | Archived on May 21, 2025; not selected as the primary framework for a new implementation |

The [official ListT5 repository](https://github.com/soyoung97/ListT5) showed 51 stars. It does not meet the high-star selection criterion and was not added as a dependency. Its peer-reviewed paper, P2, is included in the research comparison, without describing the implementation as widely validated for reproducibility. Missing candidates that satisfy both criteria are not replaced with invented rankings.

## Official API references

- [PyTorch inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html): disables autograd during inference. eval() is still separately required.
- [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html): set_to_none=True.
- [Softplus](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softplus.html): a numerically stable loss applied to pairwise score differences.
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): a stable combination of sigmoid and binary cross-entropy; the implementation uses the functional equivalent.
- [TensorFlow Ranking overview](https://www.tensorflow.org/ranking/overview) / [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy): comparison references, not a claim that TensorFlow was run.
- [Official OpenCV Python tutorials](https://docs.opencv.org/4.13.0/d6/d00/tutorial_py_root.html): image processing; not installed because this text-search prototype does not require it.
- [Official Elastic RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) / [ranking](https://www.elastic.co/docs/solutions/search/ranking): retrieval design and BM25/RRF definitions. These official primary sources supplement framework API references.

When checked, PyTorch stable redirected to 2.14. The executed environment used 2.14.1+cpu. The TensorFlow API page displayed 2.16.1; that was not interpreted as its latest release in 2026. OpenCV redirected to 4.13.0. Consult the requirements and execution record for tested versions.

The code is an original educational implementation, not copied paper code. It does not claim that every function's effectiveness is guaranteed by a paper. Big-O is derived from the actual loops and arrays. No fix is attributed to an unverified “resolved issue”, and no claim is made that memory leaks are impossible.

## Taiwanese legal primary sources

These address the separate legal requirement, not the ML literature requirement.

- [Personal Data Protection Act, Laws & Regulations Database of the Republic of China](https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=I0050021): before introducing real data, check purpose, grounds for collection/processing/use, notice, and safeguards, including Articles 19 and 20. Some amendments have separately designated commencement dates; an amendment date alone does not establish that a provision is in force.
- [Copyright Act, Laws & Regulations Database](https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=J0070017) / [Taiwan Intellectual Property Office, Article 10-1](https://www.tipo.gov.tw/tw/copyright/694-17731.html): distinguishes ideas and principles from their expression. It does not authorize unrestricted copying of paper figures, full texts, or others' code.

The publication contains synthetic data, original code, short independently written summaries, and source links. It excludes certificates, identity numbers, student numbers, third-party names from private documents, customer logs, and the complete job posting. Licensing and privacy assessments depend on the actual use. This document is not a comprehensive legal opinion on compliance with Taiwanese law.
