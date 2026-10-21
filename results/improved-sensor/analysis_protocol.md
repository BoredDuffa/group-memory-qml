# Improved UCI-HAR quantum training: analysis contract

The prior real-sensor benchmark is frozen. New code and result folders are separate.

Question: can a validation-chosen input representation and longer joint training bring the quantum-feature model to classical sensor-classifier accuracy, and can GroupMemory achieve that accuracy faster including overhead?

Selection uses development seeds 966001 and 966002, training and subject-disjoint validation only. Candidate representations and 60/300/1000-update checkpoints are recorded before final tests. All computational development searches are retained and timed. This is a small exploratory search, not exhaustive optimization or a proof of the best architecture.

The final selected recipe is frozen before evaluation on 256 official-test windows excluding all previously evaluated quick/pilot/main rows and their immediate same-subject neighbors. Test subjects remain the same official held-out subjects as the earlier study; this is fresh-row evaluation, not a new dataset or independent cohort. Training and validation use only official-training subjects.

Final comparison: 5 matched seeds, 2 fresh-process repetitions, randomized serial method order, one CPU thread. GroupMemory and Qiskit receive the identical patch decomposition, exact parameter-response support, input keys and cache lifetime. Include a direct classical derivative implementation of the same model, a fixed-quantum-angle ablation, and independently trained logistic, RBF and MLP models offered the quantum input representation, all 561 standardized features, and the five uncompressed LDA axes. Validation chooses each classical feature view; all three fits are timed.

Primary quality metric: six-class balanced test accuracy. Chance is 1/6. Exact numerical equality between equivalent quantum methods requires maximum absolute saved matching-vector error <=1e-10. Do not relax this threshold for longer optimization. Accuracy differences are descriptive across overlapping samples from one dataset, not significance or equivalence tests.

Time: external parent perf_counter for each complete fresh subprocess, including startup/imports, dataset verification/read, train-only preprocessing, construction/compilation/response calibration, all training/validation selection, final test, serialization and shutdown. Report search and correctness preflight separately and include both in a whole-study clock. Common original acquisition preparation components sum to29.555855s and are separately identified; installed-package/runtime setup and historical data-owner feature extraction are not measured. Do not label summed acquisition components as a directly measured complete download wall.

Descriptive runtime win gate: at least5 complete matched seeds with both repetitions, every seed median faster, median paired saving >=5%. Runtime improvement at worse accuracy is not a useful win. Accuracy matching means observed point accuracy, not statistical noninferiority. All losses, errors, timeouts and missing coverage are retained. These are ideal CPU simulations of independently entangled two-qubit patches, not a QPU speedup or global16-qubit quantum-learning advantage.

## Recovery addendum, frozen before repaired test evaluation

The first 100-attempt recipe finished but FAILED the unchanged trajectory check (maximum error0.0340823), with a separate UTC-versus-monotonic elapsed discrepancy. Its results remain in the original folder, rejected for equivalent-implementation runtime claims. The new run uses a common learning rate.003 and Adam epsilon1e-4, selected through validation-only numerical-stability checks; it repeats all100 attempts from scratch. Test rows are now reused for this correctness-recovery retest. This addendum supersedes any suggestion above that the repaired run has a newly untouched holdout. No timings are pooled with the rejected run. Parent journal/checkpoint overhead and all measured repair/search/preflight costs are reported separately from complete worker process time. The original failure and repair evidence are included in the public package.
