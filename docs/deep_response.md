# Sharing response-construction work in deeper QML circuits

This experiment moves reuse into **classical circuit simulation**. It does not
claim that a physical quantum computer can reuse or clone the coefficient
states used by the implementation.

The previous response engines already let Qiskit and GroupMemory reuse an
exact function of the trainable weights. Their comparable execution paths left
little room for a reliable runtime difference. Here, GroupMemory propagates
the coefficients of that function together, instead of simulating every probe
angle separately. All old source files and historical results are preserved.

## Circuit and exact response

For an input vector \(x\), prepare
\[
|\psi_0(x)\rangle=\bigotimes_{j=0}^{n-1}R_Y(x_j)|0\rangle.
\]
Each of \(L\) layers applies one trainable \(R_Y(\theta)\) gate at site
\(\ell\bmod n\), followed by a fixed noncommuting block of single-qubit
rotations and ring entanglers. Two fixed blocks alternate; there are no
identity pairs inserted to inflate depth. The observable is mean \(Z\).
The exact block angles and gate order are defined in `deep_response.py`.

There is **one shared trainable angle**, appearing in \(L\) separate gates,
and a fitted classical classifier head. This is not an independent
trainable angle at every site/layer, nor the old shallow sensor model at a
larger numerical depth.

Let \(P_\pm=(I\pm Y)/2\). Since
\[
R_Y(\theta)=e^{-i\theta/2}P_+ + e^{i\theta/2}P_-,
\]
the state after \(\ell\) occurrences can be represented as
\[
|\psi_\ell(\theta,x)\rangle
=\sum_{k=0}^{\ell}e^{i(k-\ell/2)\theta}|v_{\ell,k}(x)\rangle.
\]
At the next rotation, the two projector branches shift the coefficient
index; the following fixed block acts on every active coefficient. Fixed
block operators are constructed once per worker and reused. Autocorrelations
of the final coefficient states give
\[
f(x,\theta)=\langle O\rangle
=\sum_{m=-L}^{L}c_m(x)e^{im\theta},\qquad
\partial_\theta f=\sum_{m=-L}^{L}im\,c_m(x)e^{im\theta}.
\]
No two-query shift rule is applied directly to the multiply occurring angle.

The Qiskit response controls reconstruct this same degree-\(L\) function
from \(2L+1\) uniformly spaced angles. The batched-grid ablation evaluates
that grid using dense batched states. Every response method checks its
reconstruction against two independent native Aer executions, at
\(\theta=0.317,-0.583\), with unchanged absolute tolerance \(10^{-10}\).
The architectural derivation establishes the response support; two numerical
checks alone cannot establish a basis for arbitrary circuits.

## Controls and costs

| Method | Implementation |
|---|---|
| `group_spectral` | Propagate Fourier coefficient states and form response autocorrelations |
| `group_grid` | Batched dense evolution of the same Fourier grid |
| `qiskit_response` | Aer runtime parameter binding, native compiled circuit, same grid and response |
| `qiskit_blocks` | Aer runtime-bound response probes with the same cached fixed-block unitaries |
| `classical_direct` | Direct exact state and tangent propagation; no persistent Fourier response |
| `classical` | Task-specific classical input features and logistic regression |

All response methods get the same degree bound, input identity, coefficient
cache lifetime, checks and training problem. No spatial invariance is assumed
for the new circuits. Cache keys retain exact input bytes; labels never enter
the cache. The sensor task splits independent base patterns before creating
related views. The shared-angle task uses uniformly encoded inputs.
An unseen forward-only input is executed directly once at the requested
angle; it does not pay to reconstruct an entire response that will not be
used. A derivative request constructs and checks the persistent response.

The installed Aer version produced an incorrect output when the tested
dense-unitary gates used read-only NumPy matrix buffers. Supplying writable
copies once at setup fixes that fixture and retains fast runtime parameter
binding. Copy construction is timed. The regression fixture and measured
discrepancy accompany the results; this is a statement about the tested
fixture/version, not every Aer circuit or parameter-binding operation.

The primary timer encloses a fresh subprocess, including imports, data,
operator construction, compilation, calibration/checks, caching, training,
prediction, result serialization and interpreter exit. Stage times identify
where savings occur. Parent overhead, correctness tests, checkpoint/resume
pilot and reporting are separately recorded.

Native Aer call counts do not count coefficient-state or dense-matrix
arithmetic. Reducing these calls is **not physical CNOT or shot savings**.
Report representative circuit depth and CNOT count separately from the
classical operations and memory needed by the implementation.

Dense block operators require \(O(4^n)\) storage, and coefficient states need
\(O(B(L+1)2^n)\) for \(B\) inputs. This restricts the implementation to small
qubit counts. More independent parameters can produce a large multidimensional
response table. Faster performance here is not evidence for full LLM training,
large-QPU execution, or an exclusive capability unavailable to Qiskit.

## Reproduce

Use the repository's Python environment and install `.[dev]` as in the README.

```bash
python -m pytest tests/test_deep_response.py tests/test_deep_protocol.py
python -m group_memory_qml.deep_suite run \
  --conditions benchmarks/deep_quick.json \
  --output results/deep-quick-local \
  --seeds 942001 --repeats 1 --budget-seconds 300
python -m group_memory_qml.deep_report results/deep-quick-local
```

Full planned comparison:

```bash
python -m group_memory_qml.deep_suite run \
  --conditions benchmarks/deep_response.json \
  --output results/deep-response-local \
  --seeds 942101 942102 942103 942104 942105 \
  --repeats 3 --budget-seconds 1800
python -m group_memory_qml.deep_report results/deep-response-local
```

The six conditions are sensor circuits at six qubits and shared-angle
circuits at eight qubits, each with 2, 8 and 16 interleaved layers and 12
training updates. Five fresh data seeds, three repeats and six methods give
540 attempts. Seeds vary the data, not the fixed architecture.

The budget is an immutable deadline rather than a runtime promise.
Reissuing the identical command resumes the saved plan and original deadline.
Source/configuration changes are rejected. Checkpoints are written after
each attempt, with the latest two and a manifest. On a separate pilot, use
`--limit 1` then remove that flag to verify continuation. Incomplete/failed
coverage remains visible. The quick run is a functionality check, not a
statistically supported timing claim.

A descriptive win requires five complete paired seeds, three matching
repeats each, every seed median faster, and at least 5% median paired saving.
Complete training trajectories and predictions must agree to \(10^{-10}\).
Observed seed ranges are not confidence intervals.

## Prior work and scope

The Fourier structure and reconstruction of parameterized quantum functions
are established ideas. See [Wierichs et al., General parameter-shift rules for
quantum gradients](https://quantum-journal.org/papers/q-2022-03-30-677/).
The control uses [Qiskit Aer runtime parameter
binding](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
instead of manufacturing one separately bound Python circuit per grid point.

The contribution tested here is a concrete implementation and full-cost
comparison of shared response construction. Its arithmetic and optimizations
can be adopted by other software. Hardware noise, finite-shot uncertainty,
calibration drift and their correlated effects across training are outside
this ideal CPU experiment.
