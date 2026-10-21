# Eight experiments: assumptions and reproduction

This suite tests four ways to reuse work in small quantum-machine-learning workflows. Each application has two experiments. All runs use ideal CPU simulation; the suite submits no QPU jobs and measures no hardware shots.

The central comparison is **GroupMemory versus Qiskit given the same equivalence rule**. The implementation does not automatically discover arbitrary symmetries. Most classical controls solve the same designed prediction tasks using different models; both their quality and their time must be reported. Experiment 4B is a useful exception: its analytic classical control evaluates the same model exactly.

## Run the suite

Install the package as described in the main README, then run the correctness tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

Create an explicit condition file:

```bash
cat > conditions.json <<'JSON'
[
  {"id": "1A-small", "experiment": "1A", "config": {}},
  {"id": "1B-small", "experiment": "1B", "config": {}},
  {"id": "2A-small", "experiment": "2A", "config": {}},
  {"id": "2B-small", "experiment": "2B", "config": {}},
  {"id": "3A-small", "experiment": "3A", "config": {}},
  {"id": "3B-small", "experiment": "3B", "config": {}},
  {"id": "4A-small", "experiment": "4A", "config": {}},
  {"id": "4B-small", "experiment": "4B", "config": {}}
]
JSON
```

A bounded smoke invocation runs one complete four-method block from the deterministic shuffled plan:

```bash
python -m group_memory_qml.benchmark_suite run \
  --conditions conditions.json \
  --output results/eight-smoke \
  --seeds 930101 --repeats 1 \
  --budget-seconds 180 --limit 4
```

For a fresh planned comparison, explicitly choose a new output directory and budget:

```bash
python -m group_memory_qml.benchmark_suite run \
  --conditions conditions.json \
  --output results/eight-run \
  --seeds 930101 930102 930103 930104 930105 \
  --repeats 3 --budget-seconds 1800

python -m group_memory_qml.experiment_report results/eight-run
```

The full plan has 480 attempts: eight conditions × five data seeds × three timing repeats × four methods. The budget is a cap, not a completion-time estimate. Unfinished, failed, and timed-out attempts remain explicit. Generating the report only reads saved data; it does not run the experiments.

The repository also provides `benchmarks/eight_experiments.json`, with two workload sizes for each experiment: **16 conditions and 960 attempts** at the same seed/repeat settings. The smaller `benchmarks/eight_quick.json` exercises all eight APIs in 32 attempts with one seed and one repeat. The quick run is a functionality demonstration, not sufficient evidence for a descriptive performance win.

The harness saves an immutable configuration, source hashes, authorization deadline, per-attempt journal, and two alternating checkpoints. To resume before that deadline, run the **same command with the same output, condition file, seeds, repeats, and budget**, omitting `--limit` if a smoke limit stopped it. Resume continues the plan cursor and does not reset the deadline. Changed code or configuration is rejected; use a new directory for a genuinely new run.

## Four comparisons in every condition

| Method | Supplied capability |
|---|---|
| `group_memory` | Proven equivalence for this experiment, dictionary-based deduplication |
| `qiskit_symmetry` | The same equivalence, canonical jobs, circuit, simulator, and vectorized unique planning |
| `qiskit_exact` | Ordinary exact-input/query caching, without the extra group reduction |
| `classical` | An application-specific classical prediction model |

`qiskit_symmetry` is an explicitly programmed application wrapper, not a claim that a stock Qiskit pass discovers these relations. `qiskit_exact` is a secondary ablation. Reducing work relative to it does not establish an advantage over equally informed Qiskit.

## 1A — Symmetry-related sensor classification

The task distinguishes contiguous from alternating signed readings around an even-width ring. Paired classes have identical amplitude histograms. Underlying bases are generated before rotation/reflection views, and train/test exact orbits are disjoint.

The quantum circuit has a shared ring architecture. Its local-Z outputs transform with the input permutation; their mean and shared-weight derivatives are invariant. Canonicalizing a known orbit permits one evaluation to serve multiple views. Every change to the shared weights invalidates the cache.

The learner alternates fitting a standard scaler/logistic head and updating two shared quantum weights. The derivative holds the fitted scaler/head fixed; it does not differentiate through refitting them. Reverse derivatives are classical CPU computations. The classical control uses train-standardized Fourier power and circular correlations with logistic regression.

Report held-out accuracy, forward evaluations, reverse-gradient instances, and training time. The independent split units are histogram pairs, not augmented views. Set `symmetry` to `false` to disable orbit reuse for the circuit with site-dependent gains.

## 1B — Equivariant sensor denoising

Each independent base is a noisy periodic signal with a clean target. Every transformed view permutes both the input and target. The output is a vector, so reuse must undo the canonical input's permutation before fitting or predicting.

Quantum weights remain frozen. A shared ridge readout learns from the raw noisy reading and local features of the quantum outputs. The classical control uses a shared ridge readout on circular neighborhoods of the raw input. Neither method fits on the test set.

Report held-out mean-squared error, where lower is better, alongside raw-input and zero-predictor errors. This experiment trains a readout; it does not train quantum parameters. Independent test signals, not their views or individual sites, are the sampling units.

## 2A — Reuse valid derivatives of a tied quantum weight

Eight sites have uniform input preparation, ring entangling gates, and one rotation occurrence per site. The scalar observable is mean Z. At a tied weight, spatial symmetry makes several **individual occurrence** shift queries equivalent.

The implementation differentiates each occurrence with a valid parameter-shift rule and then sums the derivatives by the chain rule for the shared weight. It does not incorrectly apply the usual two-query rule directly to a parameter appearing in eight gates. A classical bias is trained alongside that weight.

The synthetic classifier distinguishes two ranges of a preparation angle. Its classical control is logistic regression on the angle and its sine/cosine features. Set `broken_symmetry` to `true` to break the spatial equivalence. Exact periodic relations remain valid where their assumptions still hold.

This is a controlled query-reuse experiment, not a claim of the best available derivative method. Generalized shared-parameter shift schedules and CPU adjoint/reverse derivatives are relevant alternatives but are not benchmarked here.

## 2B — Reuse gradient and Hessian requests during training

A six-qubit model has four parameters, each appearing in one Pauli rotation. Integer shift metadata records the derivative requests. Proven expectation-value periodicity identifies duplicate shifts without rounding nearby floating-point parameters together.

Training uses exact parameter-shift gradients and Hessians, a damped Newton direction, a bounded step, and a loss-decrease acceptance check. The target is an explicit smooth function of six input angles. The classical control is RBF kernel ridge regression.

Report held-out MSE, the full training trajectory, executed query count, and total time, including Hessian construction and solving for the update. Comparisons with generalized derivative schedules or CPU adjoint Hessians remain outside this experiment.

## 3A — Raw fidelity kernels with joint pair symmetry

A ring feature map has two input-encoding layers separated by entanglers. This avoids the degenerate construction in which a fixed trailing unitary cancels completely from every fidelity. An SVM learns from the resulting kernel.

For the supported symmetric map,

$$
K(x,y)=|\langle\phi(x)|\phi(y)\rangle|^2
      =K(Px,Py).
$$

The **same permutation acts on both inputs**. Independently canonicalizing `x` and `y` would generally change the raw fidelity kernel and is not allowed. Symmetric pair swapping is also valid. Breaking the map's architecture disables joint-group reuse.

All quantum methods cache statevectors by exact input, avoiding unnecessary preparation for each pair. Kernel overlaps are classical inner products. The classical model uses an RBF kernel on train-scaled invariant Fourier/correlation features.

## 3B — A fully group-averaged kernel

This experiment explicitly averages all ring rotations and reflections:

$$
\bar\rho(x)=\frac{1}{|G|}\sum_{g\in G}
|\phi(gx)\rangle\langle\phi(gx)|,\qquad
\bar K(x,y)=\operatorname{Tr}\bigl(\bar\rho(x)\bar\rho(y)\bigr).
$$

This density-feature inner product equals the full double-orbit average of raw fidelity. It is positive semidefinite and invariant under separate transformations of the two inputs. **Only for this explicitly averaged kernel** is independent input canonicalization valid. Its invariance follows from full averaging even when the underlying feature map is not equivariant.

Every quantum method uses the same averaging algorithm and exact state cache. Timings include state preparation, the entire average, classical inner products, diagonal normalization, PSD handling, and SVM fitting. The averaged diagonal is purity, not generally one: test diagonals are computed from self terms only. Raw-fidelity test diagonals in 3A are known to be one.

All methods reject a training-kernel eigenvalue below `-1e-9`. Tiny roundoff negativity receives only a matching diagonal jitter, not a silent projection of a materially incorrect kernel.

The held-out inputs receive an explicit orientation shift. Small sensor test sets remain synthetic and correlated by augmentation; the classical invariant-feature RBF control remains part of the comparison.

For 3A/3B, supported scaling configurations use `n_qubits` 6 or 8, `pairs` 2 or 4, and `views` 1, 2, or 4. Samples per split equal `2 * pairs * views`. These are bounded CPU experiments, not claims about scalable quantum-kernel advantage.

## 4A — Shared quantum filters for image patches

The task classifies small synthetic stripes and checkerboards after histogram-preserving pixel perturbations. Whole-image train/test orbits are disjoint. Four rotated views of an image remain related observations.

An image is split into non-overlapping 2×2 or 3×3 patches. A fixed quantum grid filter uses shared local gates and equal entangling couplings; its scalar mean-Z response is invariant under the eight rotations/reflections of a square. Exactly equal patches receive ordinary caching in every quantum method. The symmetry-aware methods can additionally identify equivalent orientations. Site-dependent gains disable that extra reuse.

Patch responses are pooled into image features; a training-only scaler and logistic readout are fitted. Quantum weights are frozen. The classical control uses local image correlations and logistic regression. The exact patch cache is shared across the input batches because the filter is fixed and unlabeled: reusing a patch value does not reuse a test label or fit preprocessing on the test set.

Report image accuracy, actual unique patch evaluations, compilation and cache cost, and complete runtime. Set `patch_side` to 2 or 3 for four or nine simulated qubits respectively. A grid of many patches does not imply one equally large entangled circuit.

## 4B — Exact local observables of a connected quantum ring

This experiment specifies a connected ring with product-state input preparation, one commuting ZZ entangling layer, and local rotations/readout. The global state can be entangled. For each local Z observable, gates outside its three-site backward cone cancel exactly in the Heisenberg picture.

Every quantum method uses the same exact cone decomposition. Only three-qubit circuits are simulated. Exact matching of cone data and parameters reuses repeated subproblems; the symmetry-aware methods can also identify reflected cones, exchanging endpoint data and the corresponding couplings together.

The classical control directly evaluates an analytic expression for the **same local observable**, then uses identical pooled features and the same fitted logistic readout. It is an especially strong baseline because this architecture admits a closed form. Every method also pays for a formula agreement check. For the quantum paths, this compares independently computed cone outputs with the formula; the classical path reevaluates the same formula. A deployment that omits validation would have a different cost.

The synthetic inputs contain contiguous or alternating motifs. Quantum weights remain fixed. Supported `n_global` values are multiples of eight from 8 through 1024; the default is 256. `periodic: false` removes the short repeated motif, and `symmetry: false` introduces site-dependent couplings/rotations and disables the extra reflection reduction.

`n_qubits` in this result describes the specified global circuit, while diagnostics record a maximum of three actually simulated qubits. Report both. This is exact simulation of supported local observables, not construction of a 256- or 1024-qubit statevector, execution on that many physical qubits, or a general solution for deep entangled circuits.

## Timing, matching, and interpretation

Each attempt runs in a fresh subprocess. Its primary clock begins in the parent before launch and ends after the subprocess exits. It therefore includes interpreter startup, dependency imports, source hashing, data generation, setup, compilation, canonicalization, execution, learning, prediction, result serialization, and interpreter shutdown.

The worker also records application time separately. Installation, independent correctness tests, parent orchestration, and report generation are separate costs. Report both the sum of attempt walls and campaign wall, preserving the difference; never describe a lower application time as a lower complete-process time without checking it.

These are **cold-process comparisons**, not measurements of a continuously running service. A classical scikit-learn model and a NumPy/Qiskit implementation can have different import costs. The parent clock charges the actual dependencies each needs. Some imports occur lazily inside the application function, so application-only time is not a uniform import-free boundary across experiments. Use fresh-process totals for the stated comparison; do not infer steady-state arithmetic, gradient, or hardware advantage from startup-dominated timings.

Within each condition/seed/repeat block, the quantum methods must agree on their saved predictions, kernels, or trajectories within absolute tolerance `1e-10`. The two equally informed implementations must also agree on their evaluation and CNOT counts. Altered specifications, source hashes, or task metrics are rejected.

A seed enters matched timing comparisons only when every configured repeat needed for the quantum block succeeds. The primary paired ratio is:

$$
\frac{\text{median Qiskit-symmetry process time for this seed}}
     {\text{median GroupMemory process time for this seed}}.
$$

The report then summarizes across complete seeds. A descriptive-win flag requires at least five complete seeds, every seed median faster, and median paired saving of at least 5%. This is a predefined descriptive rule, not a confidence interval, significance test, or universal crossover claim. Most classical time ratios compare different models and must be interpreted alongside their task metrics; the 4B analytic control computes the same model.

With partial coverage, a method's marginal summary can include a different subset of seeds from another method's summary. Paired ratios use their complete shared seeds. Check the seed counts and raw seed blocks before interpreting adjacent quality values or dividing marginal times yourself.

Counts have experiment-specific units. Sensor and derivative `evaluations` are executed quantum-circuit simulations; kernel `evaluations` are classical overlap calculations. Read each record's diagnostics before comparing counts. `total_cx` covers native gates in simulated preparations/evaluations actually performed; it is not a hardware receipt and does not include classical density overlaps or reverse differentiation.

CX counts use the `rz`, `sx`, `x`, `cx` basis without a physical coupling map. They are logical simulation work, not routed gate counts for a particular QPU. The `descriptive_win` field is a timing flag only: it does not incorporate prediction quality and is never, by itself, a quantum-advantage verdict.

Finite-shot hardware could break ideal equivariances and introduce correlations when estimates are reused. None of these experiments establishes hardware accuracy, cost, or quantum advantage.

## Saved artifacts

- `config.json` and `authorization.json`: frozen plan, source identity, and deadline.
- `attempts/*.json`: every measured attempt, including failures and timeouts.
- `checkpoints/`: resume cursor, configuration identity, and journal linkage.
- `attempts.csv` and `validation.csv`: complete planned coverage and matching status.
- `seed_blocks.csv`, `comparisons.csv`, and `summary.csv`: eligible paired statistics.
- `analysis.json`, `summary.md`, `comparison.png`, and `results.html`: saved-data readout.
- `report_wall.json`: separately measured readout cost.

When sharing a result, include the code revision, conditions, environment, complete coverage, correctness status, classical quality, and both timing boundaries. A missing result is not a zero-time win.

The published reference may bundle the raw attempt files in `raw_attempts.zip` to keep the review readable. Extract that archive in its result directory before rerunning `python -m group_memory_qml.experiment_report` on the reference. New experiment runs write `attempts/*.json` directly and need no extraction. The executed notebook reads the published summaries without extracting or launching any jobs.
