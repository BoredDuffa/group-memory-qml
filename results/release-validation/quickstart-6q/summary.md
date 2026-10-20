# Sensor classifier: measured release run

6 qubits; 1 data seed(s); 1 timing repeat(s); 3 quantum updates.
Architecture symmetry enabled: True.

| Method | Mean seed accuracy | Median seed application time | Forward states | Reverse-gradient instances | Forward CNOT work |
|---|---:|---:|---:|---:|---:|
| GroupMemory | 75.00% | 0.414634 s | 40 | 24 | 480 |
| Qiskit + same symmetry | 75.00% | 0.342829 s | 40 | 24 | 480 |
| Qiskit exact cache | 75.00% | 1.302037 s | 160 | 96 | 1920 |
| Classical model | 100.00% | 0.007867 s | 0 | 0 | 0 |

## Paired descriptive timing

Ratios divide each baseline's median repeat time by GroupMemory's median repeat time within the same data seed, then take the median across seeds. Values above one favor GroupMemory. The symmetry-aware Qiskit comparison is primary.

- Qiskit + same symmetry / GroupMemory: **0.827×** (1 matched data seed(s)).
- Qiskit exact cache / GroupMemory: **3.140×** (1 matched data seed(s)).
- Classical model / GroupMemory: **0.019×** (1 matched data seed(s)).

These small descriptive measurements do not establish a reliable performance advantage. There are no confidence or significance claims. A repeat is not a new dataset.

## Interpretation

- Each test seed contains four independent histogram pairs, eight base patterns and 32 related views. The balanced chance level is 50%; related views are not independent test samples.
- All quantum methods use the same learner and are checked for matching predictions, weights and partial gradients. They train only two quantum weights.
- Forward CNOT work is forward evaluations × CNOTs per compiled circuit. Avoided evaluations are not a reduction in the gate count of one circuit.
- Reverse gradients run on the CPU. No QPU jobs, shot noise, or hardware training are measured.
- Application totals include data, compilation, canonicalization, cache, binding, execution, gradients, fitting and returned-record construction. Imports, validation, plot/file work and the remaining harness are reported in `wall_accounting.json`.
- Every invocation starts with a fresh engine. Cached results can be reused within that invocation; changing its weights clears the cache.

[Accuracy and runtime](comparison.png) · [Training trajectories](learning.png)

Inspect `results.json`, `validation.json`, `environment.json`, and `wall_accounting.json` before comparing results from different machines.
