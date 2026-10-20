# Method, correctness, and measurement

## Supported computation

The demo builds an even-width ring circuit internally. For input angles $x$ and two shared weights $\theta=(\beta,\alpha)$, it applies:

1. $R_y(x_j+0.31)$ at each site.
2. $R_{ZZ}(\beta)$ on every nearest-neighbor ring edge.
3. $R_x(\alpha)$ at each site.
4. $R_y(0.47)$ at each site.

The initial state is $|0\rangle^{\otimes n}$. The forward result is the vector of local $Z$ expectations. Its mean is the classifier's quantum feature.

Rotations and reflections of the ring form the dihedral group $D_n$, with $2n$ site permutations. The local gates share parameters, and the $ZZ$ gates commute and cover the ring edges uniformly. These properties make the circuit equivariant under those permutations. If $P$ permutes an input vector, then

$$
z_\theta(Px)=Pz_\theta(x),\qquad
f_\theta(Px)=f_\theta(x),\qquad
f_\theta(x)=\frac{1}{n}\sum_j z_{\theta,j}(x).
$$

The scalar derivatives $\partial f_\theta/\partial\beta$ and $\partial f_\theta/\partial\alpha$ are therefore invariant under the same transformations.

This relation is built into a supported architecture. The implementation is not an arbitrary-circuit symmetry prover. Changing the gate order, connectivity, observables, weights, or input encoding can invalidate it.

## Canonical requests and cache scope

For each input, enumerate all ring rotations and reflections and choose the smallest contiguous float64 byte representation as its representative. Byte ordering only provides a deterministic representative; it does not assign a physical meaning to the ordering. No rounding, nearest-neighbor lookup, or approximate equality is used.

A request records the permutation that maps the original input to its representative. One representative forward evaluation supplies the local-observable vector for every equivalent input, restored using the inverse permutation. The scalar mean-feature gradient needs no output permutation.

The cache belongs to one engine with a fixed architecture and observable definition. It is cleared whenever the two-weight byte representation changes, including a one-representable-float change. This is important during training: equivalent inputs share work at the **current** weights; an old value is not reused after an update. A request needing derivatives also checks that the cached entry contains them.

Any extension that shares a cache across engines must include the full circuit, observable, backend semantics, precision, and derivative definition in its identity. The current scoped cache must not be treated as a general persistent quantum-result cache.

## Fair Qiskit comparisons

All quantum methods use ordinary template compilation, parameter binding, and exact-input caching. They share the Qiskit circuit, Aer CPU statevector backend, native gate basis, compilation settings, double precision, one simulation thread, and reverse-gradient algorithm.

- **GroupMemory** deduplicates canonical requests with a Python dictionary.
- **Symmetry-aware Qiskit** receives exactly the same representative calculation and deduplicates with a vectorized unique table.
- **Exact-cache Qiskit** deduplicates original input bytes without applying the group action.

Both symmetry-aware methods sort missing canonical jobs identically before executing them. Their predicted values, gradients, and trained weights should agree within the published numerical checks. Different speed comes from the surrounding implementation and measurement variation, not from giving one method a better symmetry.

The second method is the primary baseline. The exact-cache method is an ablation that answers how useful the supplied symmetry is; beating it does not establish an advantage over equally informed Qiskit.

## Training and the classical control

Each quantum training step evaluates the mean feature and its two shared-weight derivatives on the training split. A standard scaler and logistic head are fitted on those training features. The step computes the derivative of cross-entropy while **holding that fitted head and scaler fixed**, clips the gradient norm to at most one, and updates the two quantum weights. The head and scaler are refitted at the next step.

This is an alternating optimization procedure. It does not differentiate through the optimizer that refits the head or scaler, and it is not the derivative of the fully refitted learning procedure. Three updates are used by default. A final training-only fit supplies the reported held-out predictions.

The gradients use Qiskit's `ReverseEstimatorGradient`, a classical statevector derivative method. They do not represent parameter-shift circuits submitted to a QPU. The same reverse derivative is used by every quantum method.

The classical control trains logistic regression on the input mean, Fourier power, and circular autocorrelations. Those features are invariant to the same ring rotations and reflections. Their scaling and fitting use only training data. The control is deliberately useful for this task; it is not required to use the quantum model's much smaller feature space.

## Data and statistical limits

Each split contains four independent random amplitude draws. Each draw generates a pair of patterns with exactly matching signed-amplitude histograms: one contiguous and one alternating. Each base pattern has four related views. Separate random-number namespaces generate the training and test splits; the checks require disjoint base identifiers and exact orbit identities.

There are eight base patterns and 32 views per split, but only four independent histogram pairs per split. Test views are correlated. Three data seeds with two timing repeats are three datasets, not six independent learning experiments.

Balanced chance accuracy is 50%. A high observed accuracy on these tiny synthetic sets demonstrates that this implementation can learn the designed task; it does not establish population-level accuracy or practical quantum advantage. Runtime summaries are descriptive. Do not infer superiority from a small difference between marginal medians.

## Broken symmetry and validation

The broken-symmetry control replaces the shared $R_x$ angle with site-dependent gains times $\alpha$. Ring permutations no longer preserve that model. The engine disables orbit reuse and retains exact-input caching.

Correctness tests cover:

- Local-observable reconstruction against independent logical statevector evaluation.
- Shared-weight gradients against central finite differences.
- Agreement between the two symmetry-aware methods and direct exact-cache evaluation.
- Repeated-request hits and weight-change invalidation.
- Correct inverse-permutation restoration.
- Reuse disablement for the broken-symmetry architecture.
- Train/test base and orbit separation.

“Exact reuse” describes the identity rule, not an assertion of infinite-precision arithmetic. Simulators and derivatives still have explicit floating-point tolerances. Tests should retain or tighten these tolerances when fixing an implementation; changing a tolerance to hide a failure is not a repair.

## Timing and resource accounting

Application totals include the work needed for one fresh training/evaluation invocation: data construction, engine construction, compilation, representative search, cache handling, binding, simulation, derivatives, reconstruction, model fitting, prediction, and result construction. Each timed invocation creates a new engine. Within that invocation, normal warm-cache reuse is allowed and counted.

The benchmark also reports import time, validation time, harness/reporting overhead, and elapsed wall time from its first module-level clock to the accounting snapshot. This boundary includes result and plot writes, but excludes interpreter/package startup before the clock, the final accounting-file write, console output, and shutdown. Use an external process timer when those additional costs matter. The ledger states this boundary explicitly.

These costs are not silently folded into one method's application total or divided away. Whole-comparison wall time includes the competing methods and the comparison's checks; it is not the runtime of one model.

The avoided-forward-CNOT count is

$$
N_{\mathrm{forward\ evaluations}}\times N_{\mathrm{CX\ per\ compiled\ circuit}}.
$$

Reuse reduces the first factor. It does not claim to reduce the second factor, and this product excludes classical reverse-gradient work. In a statevector simulator, it is a circuit-work accounting measure rather than a hardware CNOT execution receipt.

The historical sensor result used 75% fewer forward and gradient instances when its four equivalent views collapsed to one representative. The equally symmetry-aware baseline had the same opportunity. Canonicalizing a full $D_n$ orbit itself costs work and can lose when requests are mostly unique.

## Limits of a hardware interpretation

This release runs ideal CPU simulation. On a QPU, device noise, layout, calibration, and finite-shot estimates may break or complicate the ideal permutation relation. Reusing a sampled estimate also introduces correlations between predictions that would otherwise use independent samples.

A hardware extension must state its noise assumptions and compare equal estimation precision or downstream quality. Counts of avoided ideal circuits alone do not establish QPU runtime, cost, or learning advantage.

## Related interfaces

The implementation builds on existing tools and established symmetry:

- [Qiskit commutative optimization](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.CommutativeOptimization) concerns circuit-level rewriting; request reuse here occurs around circuit evaluations.
- [Qiskit reverse estimator gradients](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.gradients.ReverseEstimatorGradient.html) supply the CPU derivatives used in the demo.
- [PennyLane QNode caching](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) is relevant prior functionality for evaluation reuse.
- [General parameter-shift rules for quantum gradients](https://quantum-journal.org/papers/q-2022-03-30-677/) discuss established gradient-evaluation reductions.

This release does not establish priority over related work. Feedback that identifies a closer existing implementation, a stronger comparison, or a useful extension is welcome.
