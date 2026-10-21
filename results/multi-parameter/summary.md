# Scaling response memory: qubits and independent parameters

Fresh local CPU simulations. The old dense-block measurements are not pooled with these matrix-free architectures. No QPU or finite-shot claim.

720/720 successful attempts; 120/120 matched blocks. Maximum complete-trajectory error: 1.77e-14; unchanged absolute tolerance 1e-10.

Each process clock encloses imports, data, preparation, compilation, response construction and native checks, cache planning, all parameter updates, head fitting, prediction, serialization and exit. Parent orchestration, preflight, resume smoke and reporting are separate measured costs. Installation and development are excluded.

| Condition | Qubits | Independent angles | Occurrences each | Layers | State coefficients | Response grid points |
|---|---:|---:|---:|---:|---:|---:|
| sensor-n8-p1 | 8 | 1 | 8 | 8 | 9 | 17 |
| sensor-n8-p2 | 8 | 2 | 4 | 8 | 25 | 81 |
| sensor-n8-p4 | 8 | 4 | 2 | 8 | 81 | 625 |
| sensor-n12-p2 | 12 | 2 | 4 | 8 | 25 | 81 |
| sensor-n12-p4 | 12 | 4 | 2 | 8 | 81 | 625 |
| sensor-n14-p2 | 14 | 2 | 4 | 8 | 25 | 81 |
| shared-n12-p2 | 12 | 2 | 4 | 8 | 25 | 81 |
| shared-n12-p4 | 12 | 4 | 2 | 8 | 81 | 625 |

| Condition | Method | Full process (s) | Accuracy | Aer circuit evaluations |
|---|---|---:|---:|---:|
| sensor-n8-p1 | GroupMemory spectral | 1.5832 | 100.00% | 8 |
| sensor-n8-p1 | GroupMemory batched grid | 1.4967 | 100.00% | 8 |
| sensor-n8-p1 | Qiskit diagonal response | 1.5232 | 100.00% | 80 |
| sensor-n8-p1 | Qiskit diagonal occurrence shifts | 1.7558 | 100.00% | 824 |
| sensor-n8-p1 | Direct state and adjoint | 1.2566 | 100.00% | 0 |
| sensor-n8-p1 | Classical task model | 1.2221 | 100.00% | 0 |
| sensor-n8-p2 | GroupMemory spectral | 1.5054 | 100.00% | 8 |
| sensor-n8-p2 | GroupMemory batched grid | 1.5628 | 100.00% | 8 |
| sensor-n8-p2 | Qiskit diagonal response | 1.6858 | 100.00% | 336 |
| sensor-n8-p2 | Qiskit diagonal occurrence shifts | 1.7064 | 100.00% | 824 |
| sensor-n8-p2 | Direct state and adjoint | 1.2598 | 100.00% | 0 |
| sensor-n8-p2 | Classical task model | 1.2157 | 100.00% | 0 |
| sensor-n8-p4 | GroupMemory spectral | 1.4910 | 100.00% | 8 |
| sensor-n8-p4 | GroupMemory batched grid | 1.9843 | 100.00% | 8 |
| sensor-n8-p4 | Qiskit diagonal response | 2.3083 | 100.00% | 2512 |
| sensor-n8-p4 | Qiskit diagonal occurrence shifts | 1.7732 | 100.00% | 824 |
| sensor-n8-p4 | Direct state and adjoint | 1.2548 | 100.00% | 0 |
| sensor-n8-p4 | Classical task model | 1.2076 | 100.00% | 0 |
| sensor-n12-p2 | GroupMemory spectral | 1.8529 | 100.00% | 8 |
| sensor-n12-p2 | GroupMemory batched grid | 3.6816 | 100.00% | 8 |
| sensor-n12-p2 | Qiskit diagonal response | 2.1471 | 100.00% | 336 |
| sensor-n12-p2 | Qiskit diagonal occurrence shifts | 3.0047 | 100.00% | 824 |
| sensor-n12-p2 | Direct state and adjoint | 1.6682 | 100.00% | 0 |
| sensor-n12-p2 | Classical task model | 1.2232 | 100.00% | 0 |
| sensor-n12-p4 | GroupMemory spectral | 2.6202 | 100.00% | 8 |
| sensor-n12-p4 | GroupMemory batched grid | 20.0096 | 100.00% | 8 |
| sensor-n12-p4 | Qiskit diagonal response | 6.3070 | 100.00% | 2512 |
| sensor-n12-p4 | Qiskit diagonal occurrence shifts | 3.3172 | 100.00% | 824 |
| sensor-n12-p4 | Direct state and adjoint | 1.9727 | 100.00% | 0 |
| sensor-n12-p4 | Classical task model | 1.4047 | 100.00% | 0 |
| sensor-n14-p2 | GroupMemory spectral | 3.2841 | 100.00% | 8 |
| sensor-n14-p2 | GroupMemory batched grid | 13.4201 | 100.00% | 8 |
| sensor-n14-p2 | Qiskit diagonal response | 4.1160 | 100.00% | 336 |
| sensor-n14-p2 | Qiskit diagonal occurrence shifts | 7.7802 | 100.00% | 824 |
| sensor-n14-p2 | Direct state and adjoint | 3.3362 | 100.00% | 0 |
| sensor-n14-p2 | Classical task model | 1.2409 | 100.00% | 0 |
| shared-n12-p2 | GroupMemory spectral | 1.9247 | 90.00% | 8 |
| shared-n12-p2 | GroupMemory batched grid | 3.7425 | 90.00% | 8 |
| shared-n12-p2 | Qiskit diagonal response | 2.1876 | 90.00% | 348 |
| shared-n12-p2 | Qiskit diagonal occurrence shifts | 3.0463 | 90.00% | 836 |
| shared-n12-p2 | Direct state and adjoint | 1.7132 | 90.00% | 0 |
| shared-n12-p2 | Classical task model | 1.2081 | 100.00% | 0 |
| shared-n12-p4 | GroupMemory spectral | 2.3721 | 90.00% | 8 |
| shared-n12-p4 | GroupMemory batched grid | 20.0826 | 90.00% | 8 |
| shared-n12-p4 | Qiskit diagonal response | 6.0203 | 90.00% | 2524 |
| shared-n12-p4 | Qiskit diagonal occurrence shifts | 3.0582 | 90.00% | 836 |
| shared-n12-p4 | Direct state and adjoint | 1.9677 | 90.00% | 0 |
| shared-n12-p4 | Classical task model | 1.5401 | 100.00% | 0 |

## Matched comparisons

Ratios use paired seed medians, not marginal median times. A descriptive win requires five complete paired seeds, three matched repeats each, every seed median faster and at least 5% median paired saving. Ranges are observed seeds, not confidence intervals. Missing/failed attempts never count as fast results.

| Condition | Baseline | Spectral speedup | Seed range | Meets rule |
|---|---|---:|---:|---|
| sensor-n8-p1 | GroupMemory batched grid | 1.000× | 0.915–1.066× | False |
| sensor-n8-p1 | Qiskit diagonal response | 1.032× | 0.923–1.062× | False |
| sensor-n8-p1 | Qiskit diagonal occurrence shifts | 1.152× | 1.109–1.283× | True |
| sensor-n8-p1 | Direct state and adjoint | 0.805× | 0.761–0.845× | False |
| sensor-n8-p1 | Classical task model | 0.792× | 0.753–0.835× | False |
| sensor-n8-p2 | GroupMemory batched grid | 1.018× | 0.898–1.137× | False |
| sensor-n8-p2 | Qiskit diagonal response | 1.098× | 0.943–1.136× | False |
| sensor-n8-p2 | Qiskit diagonal occurrence shifts | 1.136× | 1.035–1.146× | True |
| sensor-n8-p2 | Direct state and adjoint | 0.843× | 0.675–0.915× | False |
| sensor-n8-p2 | Classical task model | 0.808× | 0.749–0.819× | False |
| sensor-n8-p4 | GroupMemory batched grid | 1.301× | 1.206–1.344× | True |
| sensor-n8-p4 | Qiskit diagonal response | 1.509× | 1.386–1.580× | True |
| sensor-n8-p4 | Qiskit diagonal occurrence shifts | 1.133× | 1.075–1.216× | True |
| sensor-n8-p4 | Direct state and adjoint | 0.839× | 0.755–0.893× | False |
| sensor-n8-p4 | Classical task model | 0.797× | 0.715–0.841× | False |
| sensor-n12-p2 | GroupMemory batched grid | 1.987× | 1.839–2.063× | True |
| sensor-n12-p2 | Qiskit diagonal response | 1.159× | 1.136–1.198× | True |
| sensor-n12-p2 | Qiskit diagonal occurrence shifts | 1.608× | 1.476–1.706× | True |
| sensor-n12-p2 | Direct state and adjoint | 0.894× | 0.842–0.956× | False |
| sensor-n12-p2 | Classical task model | 0.670× | 0.633–0.694× | False |
| sensor-n12-p4 | GroupMemory batched grid | 7.676× | 7.006–7.797× | True |
| sensor-n12-p4 | Qiskit diagonal response | 2.402× | 2.281–2.471× | True |
| sensor-n12-p4 | Qiskit diagonal occurrence shifts | 1.217× | 1.166–1.266× | True |
| sensor-n12-p4 | Direct state and adjoint | 0.744× | 0.581–0.807× | False |
| sensor-n12-p4 | Classical task model | 0.550× | 0.448–0.585× | False |
| sensor-n14-p2 | GroupMemory batched grid | 4.064× | 3.883–4.255× | True |
| sensor-n14-p2 | Qiskit diagonal response | 1.264× | 1.176–1.366× | True |
| sensor-n14-p2 | Qiskit diagonal occurrence shifts | 2.355× | 2.231–2.537× | True |
| sensor-n14-p2 | Direct state and adjoint | 1.016× | 0.936–1.065× | False |
| sensor-n14-p2 | Classical task model | 0.378× | 0.352–0.442× | False |
| shared-n12-p2 | GroupMemory batched grid | 1.996× | 1.931–2.057× | True |
| shared-n12-p2 | Qiskit diagonal response | 1.160× | 1.104–1.240× | True |
| shared-n12-p2 | Qiskit diagonal occurrence shifts | 1.650× | 1.572–1.664× | True |
| shared-n12-p2 | Direct state and adjoint | 0.903× | 0.882–0.944× | False |
| shared-n12-p2 | Classical task model | 0.625× | 0.623–0.680× | False |
| shared-n12-p4 | GroupMemory batched grid | 8.307× | 7.795–8.883× | True |
| shared-n12-p4 | Qiskit diagonal response | 2.528× | 2.337–2.643× | True |
| shared-n12-p4 | Qiskit diagonal occurrence shifts | 1.305× | 1.289–1.333× | True |
| shared-n12-p4 | Direct state and adjoint | 0.830× | 0.668–0.870× | False |
| shared-n12-p4 | Classical task model | 0.598× | 0.499–0.833× | False |

## Faster of the Qiskit controls

The faster Qiskit seed median is selected separately for each seed. This deliberately favors Qiskit and is not a deployed adaptive selector.

| Condition | Paired speedup | Time saving | Meets rule |
|---|---:|---:|---|
| sensor-n8-p1 | 1.032× | 3.10% | False |
| sensor-n8-p2 | 1.077× | 7.14% | False |
| sensor-n8-p4 | 1.133× | 11.70% | True |
| sensor-n12-p2 | 1.159× | 13.70% | True |
| sensor-n12-p4 | 1.217× | 17.84% | True |
| sensor-n14-p2 | 1.264× | 20.88% | True |
| shared-n12-p2 | 1.160× | 13.77% | True |
| shared-n12-p4 | 1.305× | 23.35% | True |

## Scope and limitations

The independent angles really are updated separately. Their repeated gate occurrences are interleaved with fixed noncommuting rotations and ring entanglers. The response is a full multidimensional Fourier tensor, including mixed-sign frequencies. All response methods receive the same support, exact input identities, cache lifetime and native off-grid checks. Spatial input equivalence is not assumed.

The additional Qiskit control batches occurrence-local parameter shifts and sums them per independent parameter, using cached diagonals and runtime binding and exact job reuse. It avoids full response reconstruction and supplies a stronger alternative when the response table grows expensive.

Local-gate propagation removes the old dense 4^n block operators, but state storage still grows as 2^n. With p angles repeated r times each, the state coefficient count is (r+1)^p and response grid is (2r+1)^p. Chunking bounds temporary arrays; it does not remove exponential work. Library/import memory is additional to the algorithm's guarded array budget.

The batched-grid ablation tests ordinary batching. The direct adjoint control computes the identical model and all angle derivatives without persistent response reconstruction. The task-specific classical classifier uses different features and is compared on both accuracy and time. Sensor views share underlying base patterns; data seeds do not vary the architecture.

Classical coefficient propagation and native Aer calls are different costs. Fewer native calls do not establish physical CNOT, shot or hardware savings.

Worker process walls sum to 2425.526 s; last parent measurement invocation: 2427.431 s.
