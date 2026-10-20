# Matrix-free response memory with independent trainable angles

This experiment tests whether sharing response construction remains useful when
both the qubit count and the number of independently trainable circuit angles
increase. It is an **ideal classical CPU simulation** experiment. Matrix-free
execution removes dense block operators, but it still stores exponentially
large quantum statevectors.

This document describes the design and reproduction protocol. It does not
claim a measured speedup; completed result artifacts are needed for that.

## Circuit family and controlled comparison

Inputs are encoded as a product state,
\[
|\psi_0(x)\rangle=\bigotimes_{q=0}^{n-1}R_Y(x_q)|0\rangle.
\]
Trainable single-qubit \(R_Y\) gates are interleaved with fixed noncommuting
RX/RZ/CZ-ring blocks. The gate order and parameter-to-occurrence mapping are
defined in `src/group_memory_qml/multi_response.py`. The observable is mean
\(Z\), followed by a fitted classical logistic classifier.

There are \(p\) independently trainable angles, each appearing in \(r\) gates.
The planned comparison holds the number of trainable gate occurrences at
\(L=pr=8\). It therefore compares one angle repeated eight times, two angles
repeated four times, and four angles repeated twice without increasing the
number of trainable gates. It uses 12 training updates with the same initial
weights, optimizer, classifier, data and ordering for all same-model methods.

The planned conditions are:

| Task | Qubits \(n\) | Independent angles \(p\) | Occurrences per angle \(r\) | Total occurrences \(L\) |
|---|---:|---:|---:|---:|
| Sensor | 8 | 1 | 8 | 8 |
| Sensor | 8 | 2 | 4 | 8 |
| Sensor | 8 | 4 | 2 | 8 |
| Sensor | 12 | 2 | 4 | 8 |
| Sensor | 12 | 4 | 2 | 8 |
| Sensor | 14 | 2 | 4 | 8 |
| Shared-parameter classification | 12 | 2 | 4 | 8 |
| Shared-parameter classification | 12 | 4 | 2 | 8 |

The two tasks use different input distributions. Sensor data contain related
views of independent base patterns; base patterns are split before view
generation. The other task uniformly encodes a scalar input across the
register. Neither circuit assumes that rotating an arbitrary sensor input
preserves its output. Only exact input identities are reused.

Each main condition has **four training examples per method**. Sensor settings
use `bases=4, views=1`, giving four training and four held-out examples; the
held-out examples comprise two underlying histogram-matched pairs and are not
four independent draws. Shared-parameter settings use four training and sixteen
held-out examples. These deliberately small synthetic datasets bound execution
cost while exposing the cost of larger statevectors and response tables.
Accuracy is a task-quality check for this runtime/memory scaling toy, not
evidence of generalization or practical QML learning performance.

The final frozen configuration is authoritative. A separate pilot determines
whether the planned conditions fit the runtime and memory limits. Conditions
must not be selected according to whether GroupMemory wins; timeouts and
unattempted conditions remain visible in the completed plan.

## Exact multivariate response

For a trainable gate with angle \(\theta_j\),
\[
R_Y(\theta_j)
=e^{-i\theta_j/2}P_+ + e^{i\theta_j/2}P_-,
\qquad P_\pm=\frac{I\pm Y}{2}.
\]
Let \(r_j\) be the number of occurrences of that angle. The final state has a
finite coefficient representation
\[
|\psi(\boldsymbol\theta,x)\rangle
=\sum_{\boldsymbol a\in\mathcal A}
 e^{i(\boldsymbol a-\boldsymbol r/2)\cdot\boldsymbol\theta}
 |v_{\boldsymbol a}(x)\rangle,\qquad
\mathcal A=\prod_{j=1}^{p}\{0,\ldots,r_j\}.
\]
The projector branches update the index for the current parameter. Fixed
local gates then act on each active coefficient state. The implementation
applies local gates directly rather than materializing \(2^n\times2^n\)
fixed-block matrices.

For an observable \(O\), the response is
\[
f(x,\boldsymbol\theta)
=\sum_{\boldsymbol m\in\mathcal M}
 c_{\boldsymbol m}(x)e^{i\boldsymbol m\cdot\boldsymbol\theta},
\qquad
\mathcal M=\prod_{j=1}^{p}\{-r_j,\ldots,r_j\},
\]
where
\[
c_{\boldsymbol m}
=\sum_{\substack{\boldsymbol a,\boldsymbol a+\boldsymbol m\in\mathcal A}}
 \langle v_{\boldsymbol a}|O|v_{\boldsymbol a+\boldsymbol m}\rangle.
\]
The parameter derivatives are
\[
\frac{\partial f}{\partial\theta_j}
=\sum_{\boldsymbol m\in\mathcal M}
 im_j c_{\boldsymbol m}e^{i\boldsymbol m\cdot\boldsymbol\theta}.
\]

Mixed-sign frequencies such as \((1,-1)\) are necessary. Hermiticity gives
\(c_{-\boldsymbol m}=c_{\boldsymbol m}^{*}\) under simultaneous negation of
the whole frequency vector; it does not permit retaining only the
nonnegative orthant. A shift of one independent angle changes every gate
using it, so the ordinary two-query parameter-shift formula cannot simply
be applied to that shared angle.

The Fourier support follows from the specified circuit, rather than being
inferred from a few numerical observations. Off-grid checks validate the
implementation of that known support. They do not prove that the same support
is valid for arbitrary circuits.

## Controls and attribution

| Method | Work performed |
|---|---|
| `group_spectral` | Propagate coefficient states, construct the full signed-frequency response, and reuse it across updates |
| `group_grid` | Evaluate the same response grid using matrix-free batched state propagation, then reconstruct the same coefficients |
| `qiskit_response` | Compile the circuit and use native Aer runtime parameter binding for the same grid, response lifetime and correctness checks |
| `qiskit_shift` | Batch exact individual-occurrence parameter shifts and sum their derivatives for each independent angle, without constructing a response table |
| `classical_adjoint` | Evaluate the identical circuit model and all parameter derivatives by direct state/adjoint propagation without a persistent Fourier response |
| `classical` | Fit the task-specific classical features and logistic classifier |

The batched-grid ablation tests whether coefficient sharing adds value beyond
ordinary batching. The adjoint control avoids an artificially expensive
classical gradient baseline: it accumulates contributions from all occurrences
of each independent angle.

The occurrence-shift Qiskit control is useful when constructing a large
multivariate response costs more than directly computing the gradients needed
by the training loop. If \(\mathcal I_j\) is the set of gates using
\(\theta_j\), its exact derivative is
\[
\frac{\partial f}{\partial\theta_j}
=\frac12\sum_{\ell\in\mathcal I_j}
\left[f_{\ell,+\pi/2}-f_{\ell,-\pi/2}\right],
\]
where only gate occurrence \(\ell\) is shifted. It does not shift every gate
using \(\theta_j\) at once. All shift jobs for an update are batched, and exact
duplicate jobs may be reused.

All response methods receive the same frequency bounds, exact input keys,
response lifetime and off-grid validation requirement. Unseen forward-only
inputs execute directly once rather than reconstructing a function that will
not be reused. Their input data, predictions and complete training trajectories
must agree with the same-model controls at absolute tolerance \(10^{-10}\).
The task-specific classical learner has a different feature map, so its
comparison reports both task quality and time rather than requiring identical
predictions.

These named implementations do not exhaust every possible Qiskit optimization.
The response algorithms can be adopted by other libraries. A measured gain
would establish a software result for the tested workload, not an exclusive
capability of GroupMemory.

## The cost of more independent parameters

The state-coefficient support and exact tensor-product sampling grid contain
\[
S=\prod_{j=1}^{p}(r_j+1),\qquad
M=\prod_{j=1}^{p}(2r_j+1)
\]
entries respectively. With equal occurrence counts and \(L=8\):

| \(p\) | \(r\) | State coefficients \(S\) | Response grid points \(M\) |
|---:|---:|---:|---:|
| 1 | 8 | 9 | 17 |
| 2 | 4 | 25 | 81 |
| 4 | 2 | 81 | 625 |

Holding gate count fixed therefore does not hold response-construction cost
fixed. More independent angles can make an exact response table impractical.
For \(B\) inputs, one state-coefficient tensor needs
\(O(BS2^n)\) complex entries; a grid-state batch needs \(O(BM2^n)\).
Contraction, Fourier transforms and intermediate arrays add costs.

For example, at 12 qubits and \(p=4,r=2\), a complex128 coefficient tensor for
one input occupies about 5.06 MiB, while a full 625-point state batch occupies
39.06 MiB, before temporaries. At 14 qubits and \(p=2,r=4\), the corresponding
figures are 6.25 MiB and 20.25 MiB. Bounded chunks reduce peak allocation but
their processing and bookkeeping must be timed. The largest coefficient array
is not the same measurement as peak process RSS.

Matrix-free propagation removes \(O(4^n)\) dense operators. It does **not**
remove the \(O(2^n)\) statevector cost. These circuits do not demonstrate
general simulation of hundreds of entangled qubits, QPU memory compression,
or practical full-language-model training.

Native Aer circuit counts exclude classical coefficient and state arithmetic.
Reduced native calls are not evidence of fewer physical CNOTs or QPU shots.
The report should separate representative source/native circuit sizes,
simulated native calls, classical work, peak memory and total runtime.

## Measurement and correctness protocol

Each attempt runs in a fresh subprocess. The primary timer includes Python
startup, imports, input preparation, circuit construction, compilation,
response construction, validation, cache lookup, training, prediction,
serialization and shutdown. Method order is shuffled deterministically inside
matched condition/seed/repeat blocks, with one computational thread per worker.

Correctness checks compare full forward values, every parameter gradient,
optimizer trajectory and held-out predictions. Equal accuracy alone is
insufficient. Frozen source hashes, logical architecture identities and
compiled-circuit signatures are checked separately. Input and label hashes
must also agree with the task-specific classical control.
The shift control uses independently addressable occurrence parameters for
its operational circuit; its native hash can therefore differ from the
response circuits. It must retain the same canonical logical architecture,
and its operational signatures are checked within that method.

Five fresh data seeds and three timing repeats are planned. Seeds vary the
data, not the fixed architecture. Reported speed ratios are medians of paired
per-seed ratios, not ratios of marginal runtime medians. A descriptive win
requires at least five complete seeds, at least three matching repeats per
seed, every seed median faster, and median paired saving of at least 5%.
Observed seed ranges are not confidence intervals.
Eight conditions, five data seeds, three repeats and six methods produce
720 planned attempts. A conservative additional comparison can select the
faster of the two Qiskit control medians within each seed before calculating
the paired ratio. This is an analysis of the strongest measured control,
not a claim that choosing a runtime strategy has zero overhead.

Parent orchestration, correctness preflight, checkpoint/resume pilot and
reporting have separate timing receipts. Installation and implementation work
are excluded from benchmark runtime. Historical shallow and single-parameter
timings must not be pooled with this campaign.

## Reproduce and resume

Install the repository's development dependencies as described in the README.
The quick configuration is a functionality check:

```bash
python -m pytest tests/test_multi_response.py tests/test_multi_protocol.py
python -m group_memory_qml.multi_suite run \
  --conditions benchmarks/multi_quick.json \
  --output results/multi-quick-local \
  --seeds 943001 --repeats 1 --budget-seconds 600
python -m group_memory_qml.multi_report results/multi-quick-local
```

The full planned command uses the fixed configuration and a 70-minute ceiling,
chosen before launch from component-cost measurements. A separate bounded cost
pilot and quick functionality run each have a 600-second ceiling; their timings
are not pooled with the main comparison.

```bash
python -m group_memory_qml.multi_suite run \
  --conditions benchmarks/multi_response.json \
  --output results/multi-response-local \
  --seeds 944101 944102 944103 944104 944105 \
  --repeats 3 --budget-seconds 4200
python -m group_memory_qml.multi_report results/multi-response-local
```

The ceiling is an immutable deadline, not a runtime promise. Repeating the
identical command resumes the existing plan and deadline. Changed source,
configuration or budget is rejected. Two checkpoint slots and a manifest are
updated after each completed attempt. A pilot can use `--limit 1`, preserve
the first result and authorization hashes, and then rerun without that flag
to verify exact continuation. Failed, timed-out and unattempted work remains
visible; completion must not be inferred from a partially populated table.

The multivariate Fourier representation is established mathematical
structure. The contribution under test is its concrete implementation,
its cost relative to equally informed controls, and the boundary where reuse
stops paying for its construction and memory.
