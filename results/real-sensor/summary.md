## Real smartphone sensor data — 2026-10-20

**GroupMemory beats equally informed Qiskit in all three measured conditions,
including full process overhead, but does not beat the task-specific classical
classifiers on speed or accuracy.** It also beats direct classical adjoint
training of the identical quantum model in the two 512-example conditions;
the adjoint is faster in the 128-example condition.

The benchmark uses real [UCI smartphone activity data](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
128 or 512 training examples, 128 validation examples and 256 test examples.
Train/validation/test subjects are disjoint within each seed. Five seeds and two
timing repeats cover six methods: **180/180 successful attempts**.
The model has **8 or 16 independently trained quantum angles**, plus a jointly
trained softmax head. Its independent two-qubit patches can be evaluated on two
qubits sequentially; this is not a globally entangled 16-qubit benchmark.

| Training examples / quantum parameters | Method | Total process time | Test balanced accuracy |
|---|---|---:|---:|
| 128 / 8 | GroupMemory | 2.234 s | 59.06% |
| 128 / 8 | Equally informed Qiskit | 3.960 s | 59.06% |
| 128 / 8 | Same-model adjoint | 2.008 s | 59.06% |
| 128 / 8 | Logistic regression | 1.626 s | 87.80% |
| 128 / 8 | RBF SVM | 1.587 s | 87.34% |
| 128 / 8 | Classical MLP | 1.632 s | 84.91% |
| 512 / 8 | GroupMemory | 2.645 s | 57.59% |
| 512 / 8 | Equally informed Qiskit | 6.757 s | 57.59% |
| 512 / 8 | Same-model adjoint | 2.982 s | 57.59% |
| 512 / 8 | Logistic regression | 1.826 s | 92.08% |
| 512 / 8 | RBF SVM | 1.762 s | 92.53% |
| 512 / 8 | Classical MLP | 1.804 s | 85.87% |
| 512 / 16 | GroupMemory | 3.227 s | 58.50% |
| 512 / 16 | Equally informed Qiskit | 11.579 s | 58.50% |
| 512 / 16 | Same-model adjoint | 4.629 s | 58.50% |
| 512 / 16 | Logistic regression | 1.944 s | 92.08% |
| 512 / 16 | RBF SVM | 1.855 s | 92.53% |
| 512 / 16 | Classical MLP | 2.011 s | 91.30% |

Times are medians of seed medians; balanced accuracies are means across the five
seeds. Timing repeats do not create new test samples. GroupMemory's median paired
time savings against Qiskit are **42.9%, 60.4% and 72.1%** in table condition order.
The paired calculation differs from the ratio of marginal medians shown above.
A descriptive runtime win requires five complete paired seeds, two matching
repeats each, every seed median faster and median saving at least 5%.
A useful win over a classical control also requires no lower balanced accuracy
for every paired seed. There are **0/9 such wins** against the classical classifiers.
These are observed comparisons on overlapping subsets of one dataset, not
confidence intervals or universal performance claims.

The implementations use the same exact patch factorization and response support.
Qiskit receives the same cached diagonal blocks, compiler reuse, exact input keys,
response lifetime, validation-response warmup and off-grid correctness checks.
GroupMemory computes coefficients directly and reuses them across updates.
The adjoint is a direct classical implementation of the identical model.
Each classical learner fits both the same compressed inputs and all 561 released
features, selects the feature view by validation log loss, and pays for both fits.
Hyperparameters were fixed before the final campaign; test data does not select
models, views or checkpoints. MLP angle-view fits reached the 200-iteration
limit and emitted convergence warnings, including some selected fits; these
are retained as bounded fits. Classical controls may select all 561 features,
so the task comparison also involves different input representations.

**Learning is above the 16.7% chance baseline, but substantially weaker than the
classical classifiers.** All independent angles moved. More data/parameters did
not improve this quantum model's mean accuracy consistently. All selected quantum
checkpoints occur at the 60-update limit, so this is a bounded training result,
not a convergence or optimized-accuracy claim. Increasing 8 to 16 parameters also
changes PCA input width and head width. GroupMemory preserves the model's
predictions; faster reuse does not by itself improve predictive quality. No
frozen-quantum-feature ablation isolates the benefit of adapting the quantum
angles from training the classical head.

All 30 complete matched blocks passed the unchanged `1e-10`
trajectory/prediction tolerance; maximum error was **9.21e-15**.
**540 local tests passed.** The main campaign took **9.54 minutes**.
Per-method times include fresh interpreter startup, imports, prepared-data read
and verification, train-only preprocessing, construction/compilation,
calibration/checks, caching, training/validation/selection, prediction,
serialization and exit. The common dataset acquisition/preparation component
sum was **29.56 seconds**; standalone cold-start
illustrations add it once. No single external acquisition clock was recorded.
Original authors' sensor feature extraction, installation and implementation
are excluded. Preflight, pilot, parent orchestration, audit and report costs
are separately retained in the timing ledger. Pilot results are not pooled.

This is ideal local CPU simulation. It establishes no QPU, physical gate/shot,
energy, hardware training or practical quantum-learning advantage. UCI's current
listing and the older bundled README differ on data licensing; provenance
preserves both statements, and raw data is not redistributed.

- [Reproduction commands and scope](../../docs/real_sensor.md)
- [Executed notebook](../../examples/real_sensor_benchmark.ipynb)
- [All results](summary.csv), [paired comparisons](comparisons.csv), and [independent audit](independent_audit.json)
- [Raw attempts](raw_attempts.zip), [frozen sources](source_snapshot.zip), and [timing ledger](wall_accounting.json)

Historical benchmarks below use different data and models; their timings are
kept separate from this real-data campaign.
