# Reusing exact parameter responses during training

The original GroupMemory engines reuse circuit outputs when inputs and weights
are equivalent under a known symmetry. A weight update changes that cache key.
The response engines instead cache a small, exact function of the trainable
weights. For the two supported shallow circuits, later updates evaluate that
function and its derivatives without another circuit execution.

This is **parameter-function memoization** for explicitly supported circuit
families. It uses established trigonometric structure of parameterized circuits.
It is not a new general theorem about group theory, an arbitrary-circuit
compiler, a quantum-memory device, or evidence of a quantum learning advantage.
All executions in this benchmark use ideal CPU simulation.

## 1. Sensor classifier: four coefficients per local observable

Let the input at ring site \(i\) be \(x_i\), and define
\[
a_i=x_i+0.31,\qquad c_i=\cos a_i,\qquad s_i=\sin a_i.
\]
The circuit prepares \(R_Y(a_i)|0\rangle\) at every site, applies commuting
\(R_{ZZ}(\gamma)\) gates on all ring edges, then applies
\(R_X(\beta)\) and \(R_Y(\delta)\), where \(\delta=0.47\). The two shared
trainable weights are \((\gamma,\beta)\). The readout contains all local
\(Z_i\) expectations; training uses their mean.

Conjugating a local observable through these gates gives
\[
R_Y(\delta)^\dagger ZR_Y(\delta)
  =Z\cos\delta-X\sin\delta,
\qquad
R_X(\beta)^\dagger ZR_X(\beta)
  =Z\cos\beta+Y\sin\beta.
\]
Only the two \(ZZ\) edges incident to site \(i\) affect this observable.
For the product input state, immediately after the entangling layer,
\[
\begin{aligned}
\langle Z_i\rangle&=c_i,\\
\langle X_i\rangle
 &=s_i\bigl(\cos^2\gamma-c_{i-1}c_{i+1}\sin^2\gamma\bigr),\\
\langle Y_i\rangle
 &=s_i(c_{i-1}+c_{i+1})\sin\gamma\cos\gamma.
\end{aligned}
\]
Indices wrap around the ring. Writing \(p_i=c_{i-1}c_{i+1}\), the complete
response is
\[
z_i(\gamma,\beta)
 =A_i\cos\beta+B_i\sin\beta\sin(2\gamma)+C_i+D_i\cos(2\gamma),
\]
where
\[
\begin{aligned}
A_i&=\cos\delta\,c_i,&
B_i&=\tfrac12\cos\delta\,s_i(c_{i-1}+c_{i+1}),\\
C_i&=-\tfrac12\sin\delta\,s_i(1-p_i),&
D_i&=-\tfrac12\sin\delta\,s_i(1+p_i).
\end{aligned}
\]
The exact local derivatives are
\[
\begin{aligned}
\partial_\gamma z_i
 &=2B_i\sin\beta\cos(2\gamma)-2D_i\sin(2\gamma),\\
\partial_\beta z_i
 &=-A_i\sin\beta+B_i\cos\beta\sin(2\gamma).
\end{aligned}
\]
The classifier's shared-weight gradient averages these local derivatives and
applies the same classical loss/head chain rule as the original experiment.
No two-query parameter-shift rule is incorrectly applied to a weight appearing
in many gates.

### Learning and checking the response

`SensorResponseEngine` obtains coefficients from the existing freshly compiled
Qiskit/Aer circuit, rather than using the closed-form coefficients above:

| Probe | \((\gamma,\beta)\) | Response |
|---|---|---|
| \(P_0\) | \((0,0)\) | \(A+C+D\) |
| \(P_1\) | \((0,\pi)\) | \(-A+C+D\) |
| \(P_2\) | \((\pi/2,\pi/2)\) | \(C-D\) |
| \(P_3\) | \((\pi/4,\pi/2)\) | \(B+C\) |

Thus
\[
A=(P_0-P_1)/2,\quad
C=((P_0+P_1)/2+P_2)/2,\quad
D=((P_0+P_1)/2-P_2)/2,\quad
B=P_3-C.
\]
A fifth, off-grid probe at \((0.317,-0.583)\) must agree with the reconstructed
local-output vector at absolute tolerance \(10^{-10}\). A failed batch check
raises an error and commits no coefficients from that batch. All five probe
executions count toward runtime and simulated CNOT work.

The architecture is equivariant under rotations and reflections of the ring.
The engine calibrates a canonical representative of each exact input orbit and
restores local outputs to the requested site order. Mean outputs and their
shared-weight derivatives are invariant under these permutations. Input keys
retain the float64 bytes; nearby inputs are not merged.

Calibration is triggered by a derivative request. A previously unseen input
requested only for forward prediction receives one direct execution rather
than five speculative probes. Calibrated coefficient functions survive weight
changes. Direct forward caches still invalidate when weights change.

Nonuniform site gains disable this supported response path and delegate to the
original engine. The implementation does not discover arbitrary circuit
symmetries. The analytic same-model control also handles the explicitly
supported nonuniform gains by replacing \(\beta\) with the site's gain times
\(\beta\) and including that gain in the derivative.

## 2. Shared-weight spin gradients: two coefficients per input

The eight-qubit spin circuit prepares \(R_Y(x)\) at each site, applies a ring
of \(R_{ZZ}(\phi)\) gates with fixed \(\phi=0.37\), then applies terminal
\(R_Y(\theta_j)\) gates. Its readout is mean \(Z\). There are no entangling
gates after these trainable rotations.

For an arbitrary fixed state before the terminal rotations, define
\[
A_j(x)=\langle Z_j\rangle,\qquad B_j(x)=-\langle X_j\rangle.
\]
The exact response is separable:
\[
f(x,\boldsymbol\theta)
 =\frac1n\sum_{j=1}^{n}
 \bigl[A_j(x)\cos\theta_j+B_j(x)\sin\theta_j\bigr].
\]
Uniform preparation and the ring symmetry make \(A_j=A\) and \(B_j=B\).
At tied weights \(\theta_j=\theta\),
\[
f(x,\theta)=A(x)\cos\theta+B(x)\sin\theta,
\qquad
\frac{\partial f}{\partial\theta_j}
 =\frac{-A(x)\sin\theta+B(x)\cos\theta}{n}.
\]
The derivative with respect to the single tied training parameter is the sum
over occurrences, so the \(1/n\) factor cancels. This conclusion follows from
the terminal rotations and additive observable. A generic circuit with many
occurrences of one parameter can have many frequencies.

`SpinResponseEngine` obtains \(A=f(x,0)\) and \(B=f(x,\pi/2)\) from two
global-angle probe executions. A third execution at \(\theta=0.431\) checks
the prediction at the same fixed \(10^{-10}\) absolute tolerance. Each
previously unseen derivative input therefore costs two coefficient probes and
one check. New forward-only inputs execute directly once. The engine reuses
the coefficients across later tied-weight updates.

Untied weights, the explicit broken-symmetry fixture, and Hessian requests
delegate to the original occurrence-shift implementation. An input failing the
response check is excluded from response reuse and evaluated using that
fallback. Integer shift metadata is reduced only by the proven \(2\pi\)
expectation period; no approximate numerical-key matching is used.

For this particular product preparation and shallow ring, the coefficients
also have an efficient classical formula:
\[
A(x)=\cos x,\qquad
B(x)=-\sin x\left[\cos^2\phi-\cos^2x\sin^2\phi\right].
\]
`AnalyticSpinEngine` evaluates the identical model directly, with zero circuit
probes. It additionally implements arbitrary occurrence weights, the supported
broken-site \(R_Z\) fixture, and the resulting diagonal occurrence Hessian.

## 3. What the comparisons establish

| Method | What it receives |
|---|---|
| `group_memory_previous` | Original exact-input/spatial-symmetry cache; invalidation when weights change |
| `qiskit_symmetry` | Original Qiskit wrapper with the same spatial symmetry, circuit and derivative rule |
| `group_memory_response` | Supported exact response basis, calibration, checks, persistent response cache |
| `qiskit_response` | The same response basis, probes, checks, cache lifetime and simulator |
| `classical_same_model` | Closed-form evaluation of the identical circuit model and gradients |
| `classical` | Existing application-specific classical learner |

The two response wrappers differ in dictionary versus NumPy-based unique-input
planning. Neither receives additional model information or a looser correctness
threshold. They both use Qiskit/Aer for their quantum-circuit probes.

A speedup over `qiskit_symmetry` shows the value of reusing parameter functions
instead of only equivalent values at fixed weights. It does not establish a
speedup over Qiskit given the same response algorithm. That stronger comparison
is `qiskit_response`. The analytic control is essential because these two shallow
models are efficiently classically evaluable, including their gradients.

The old sensor baseline already uses Qiskit's reverse-mode statevector
gradient. The old spin baseline uses exact occurrence-wise parameter shifts.
Qiskit also offers reverse-mode and other gradient methods; no benchmark should
describe one tested wrapper as every possible Qiskit implementation.

Matched runs must use the same generated data, split, initialization, update
count, optimizer and classifier head. Validation includes complete predictions
and training trajectories, not just final accuracy. Labels must not enter input
canonicalization or coefficient-cache construction.

Report fresh-process wall time as the primary complete cost, including imports,
data generation, preparation, compilation, probe checks, training, prediction,
serialization and exit. Report application stages separately, and keep any
loaded-dependency diagnostic separate from cold-process results. All actual
fit/check/direct circuit executions count toward simulated CNOT work; classical
reverse-mode and response arithmetic do not acquire fictional CNOT counts.
Validation, harness and report-generation costs should also be identifiable in
the campaign receipt.

The descriptive win criterion requires at least five complete paired seeds,
at least three matching timing repeats per seed, every eligible seed median faster,
and median paired savings of at least 5%. A small marginal-median difference is
not sufficient. Missing, failed or mismatched trials remain visible. Accuracy
and classical-baseline results accompany timing conclusions.

## 4. Reproducing the experiment

Install the repository's documented dependencies in a virtual environment,
then run from the repository root:

```bash
python -m pytest tests/test_sensor_response.py tests/test_spin_response.py
```

For a short fresh-process run with a separate output directory:

```bash
python -m group_memory_qml.response_suite run \
  --conditions benchmarks/response_quick.json \
  --output results/response-quick-local \
  --seeds 940101 --repeats 1 --budget-seconds 600
```

The full planned comparison uses six conditions: the 10-qubit sensor task
with 16 base inputs and eight views at 1, 5 and 20 updates; and the eight-qubit
spin task with 12 training and 12 test inputs at 1, 3 and 8 updates.
Five seeds, three repeats and six methods give 540 planned attempts:

```bash
python -m group_memory_qml.response_suite run \
  --conditions benchmarks/response_memory.json \
  --output results/response-memory-local \
  --seeds 941101 941102 941103 941104 941105 \
  --repeats 3 --budget-seconds 1800
```

The time budget is a deadline, not a prediction that every machine will finish
all attempts. Choose a suitable budget before starting; the runner preserves
missing coverage on exhaustion. Repeating the identical command resumes its
saved plan and original deadline, provided source hashes and configuration
are unchanged. It does not reset the allowance. Two alternating checkpoints
and their manifest live under the chosen output directory's `checkpoints/`.
Use `--limit 1` for a one-attempt interruption/resume smoke check on a separate
output directory; remove `--limit` to continue that same plan.

The common training API can run the same task with each method:

```python
from group_memory_qml.response_training import METHODS, run

for experiment, config in (
    ("1A", {"n_qubits": 6, "bases": 8, "views": 4, "updates": 3}),
    ("2A", {"n_train": 12, "n_test": 12, "steps": 3}),
):
    for method in METHODS:
        result = run(experiment, seed=940101, method=method, config=config)
        print(experiment, method, result["metric_value"],
              result["evaluations"], result["total_cx"])
```

This in-process example is a functional check, not a full-overhead timing
comparison. The frozen configuration, raw attempts, source/environment hashes,
wall-time receipt and matched result table belong with the measured results.
No performance numbers are inferred from the coefficient count.

## 5. Scope and prior work

The finite trigonometric response is exact for the documented architecture and
observable. The off-grid check protects implementation and numerical behavior;
one numerical check cannot prove the basis for an arbitrary circuit. The
derivation supplies that restriction. Independent source-circuit tests should
cover random inputs and off-grid weights, derivatives, output restoration,
weight updates, fallback cases, failed checks and mutation-safe outputs.

On hardware, finite-shot coefficient estimates would have uncertainty, and
reusing them would correlate errors across training steps. The current exact
CPU tests do not demonstrate a hardware noise model, calibration lifetime,
shot savings at a fixed statistical error, or a QPU training speedup. Larger
or deeper circuits can require many more response coefficients, potentially
eliminating this approach's benefit.

Relevant foundations and implementations:

- [Schuld, Sweke and Meyer, *Effect of data encoding on the expressive power of
  variational quantum-machine-learning models* (2021)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032430)
  develops the Fourier representation and frequency structure of quantum
  models.
- [Wierichs, Izaac, Wang and Lin, *General parameter-shift rules for quantum
  gradients* (2022)](https://quantum-journal.org/papers/q-2022-03-30-677/)
  connects finite Fourier structure, function reconstruction, repeated
  parameters and higher derivatives.
- [Ostaszewski, Grant and Benedetti, *Structure optimization for parameterized
  quantum circuits*](https://arxiv.org/abs/1905.09692) uses sinusoidal parameter
  dependence in circuit optimization.
- [Qiskit `ReverseEstimatorGradient`](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.gradients.ReverseEstimatorGradient.html)
  describes efficient reverse differentiation for classical statevector
  simulation, with exponential qubit-state storage.
- [Qiskit gradient methods](https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.gradients.html)
  documents parameter-shift, reverse, finite-difference and other alternatives.
- [Qiskit Aer parameter-binding implementation and API](https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aerbackend.py)
  supports batching parameter values on a parameterized circuit. Such binding
  optimizations are another implementation control, distinct from response
  reconstruction.

The contribution evaluated here is a concrete, checked implementation of
reusable responses with matched controls and full-cost evidence. A novelty
claim about Fourier reconstruction or quantum advantage would require different
evidence.
