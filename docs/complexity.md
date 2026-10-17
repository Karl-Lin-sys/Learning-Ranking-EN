# Big-O for each function

Derived from the implementation's loops and stored arrays, not from a paper's performance guarantee. Python hash-table operations are assumed to take O(1) on average. Sorting takes O(n log n); for sizes zero or one, interpret logarithmic terms as constant.

Notation: L = text length; T = total tokens; N = products; D = vector dimensions; P = total postings for query terms; M = candidates per query; U = unique candidates across routes; R = total route results; J = judged pool size; Q = queries; F = features; H = hidden width; B = bootstrap repetitions; E = epochs; b = training batch size; S = JSON bytes. Qt and Qv denote training and validation queries; Na denotes A/B units. Unless otherwise noted, space includes allocations beyond the inputs.

| File / function | Time | Space | Notes |
|---|---|---|---|
| core.tokens | O(L) | O(L) | Case folding and regular expression; CJK characters individually |
| Product construction | O(1) | O(1) | References existing strings and tuples; O(L+D) including inputs |
| eligible | O(N) | O(N) | Condition checks and a list of references |
| BM25.__init__ | O(T+N) | O(T+N) | Title tokenization, postings, and lengths |
| BM25.search | O(L+P+M log M) | O(L+M) | Sorts all matching candidates rather than using a top-k heap |
| dense_search | O(ND+N log N) | O(N) | Recomputes query/product norms; not ANN |
| rrf | O(R+U log U) | O(U) | Per-route deduplication and sorting of unique results |
| metrics | O(M+J log J) | O(M+J) | Validates rankings and sorts for IDCG; denominator is not limited to retrieved top-k items |
| temporal_split | O(Q) | O(Q) | References rows without copying their contents; detects sessions crossing partitions |
| ips_mean | O(N) | O(N) | Retains weights and computes ESS from weight sums |
| paired_bootstrap | O(BQ+B log B) | O(B+Q) | Q independent deltas; draws processed sequentially within a repetition |
| experiment_arm | O(L) | O(L) | Hashes concatenated experiment and unit strings |
| model.make_model | O(FH) | O(FH) | One-hidden-layer MLP; linear in F for fixed H |
| ranking_loss(pointwise) | O(bM) | O(bM) | BCE with logits; model forward-pass cost is separate |
| ranking_loss(pairwise) | O(bM²) | O(bM²) | All-pair differences, masks, and losses; bound b and M |
| ranking_loss(listwise) | O(bM) | O(bM) | Gain normalization and log_softmax |
| tensors | O(QMF) | O(QMF) | Materializes the small dataset on the CPU |
| fit | O(E[(Qt+Qv)MFH+(Qt+Qv)C]) | O((Qt+Qv)MF+FH+bMH+QvMH+max(b,Qv)C) | C=M² for pairwise, M otherwise. Validation runs as one batch. Best and Adam states take O(FH) |
| predict | O(QMFH) | O(QMF+QMH) | Includes the O(QM) Python output list |
| demo.synthetic_queries | O(QM) | O(QM) | F=6 is fixed; O(QMF) when generalized |
| catalog | O(1) | O(1) | Six fixed products; O(N(L+D)) when generalized |
| search | O(T+ND+N log N+R+U log U+UFH) | O(T+ND+U(F+H)) | With a model; fixed F, H, D. Includes rebuilding the index. Actual catalogue has six products |
| evaluate | O(QM log M) | O(QM) | Judgements for all M candidates and macro aggregation |
| run | Three fits + test O(QM(FH+log M)) + bootstrap O(BQ+B log B) + A/B O(BNa+B log B) + repeated searches | Peak retained dataset, models, and aggregates | Fixed counts: 10,000 bias observations, 100 timing runs, 1,000 A/B units |
| main | run cost + O(S) | run space + O(S) | JSON serialization and file output |
| Handler.do_GET | O(1) | O(1) | Fixed health response |
| Handler.do_POST | O(S+search) | O(S+search) | S<=4096, query<=256, three vector dimensions |
| Handler.respond | O(S) | O(S) | JSON encoding |
| Handler.log_message | O(1) | O(1) | Stores no data |
| analyze_ab | O(BN+B log B) | O(N+B) | Independent-arm bootstrap; N is assigned units |
| log_summary | O(N log N) | O(N) | Sorts latencies within a fixed window |

Single-batch validation and full dataset materialization suit small experiments. Scaling training requires a DataLoader or streaming and batched validation. inference_mode avoids storing the computation graph; it does not eliminate input or output storage. Measure elapsed milliseconds, peak RAM, and GPU throughput rather than inferring them from Big-O.

Tests use fixed-size fixtures and incur the corresponding costs above. HTTP waiting, thread shutdown, and I/O are outside CPU operation counts. __init__.py contains only a declaration and no additional application functions.
