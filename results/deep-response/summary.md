# Deeper QML: response construction and training

These are new interleaved architectures and fresh local CPU timings. No QPU execution, finite shots, or pooled historical timings.

540/540 successful attempts; 90/90 matched blocks; maximum trajectory error 6.69e-14, with unchanged absolute tolerance 1e-10.

Each primary timer encloses the entire fresh worker process, including imports, source and block construction, compilation, response construction and checking, data handling, cache planning, training, prediction, serialization and exit. Separate receipts record parent overhead, preflight, pilot and reports.

| Workload | Method | Full process (s) | Accuracy | Native-basis Aer evaluations only |
|---|---|---:|---:|---:|
| deep_sensor-2 | GroupMemory spectral | 1.6111 | 53.75% | 32 |
| deep_sensor-2 | GroupMemory batched grid | 1.6071 | 53.75% | 32 |
| deep_sensor-2 | Qiskit runtime-bound response | 1.5852 | 53.75% | 128 |
| deep_sensor-2 | Qiskit cached-block response | 1.7750 | 53.75% | 32 |
| deep_sensor-2 | Direct state and tangent | 1.3201 | 53.75% | 0 |
| deep_sensor-2 | Classical task model | 1.3060 | 100.00% | 0 |
| deep_sensor-8 | GroupMemory spectral | 1.5876 | 95.00% | 32 |
| deep_sensor-8 | GroupMemory batched grid | 1.5862 | 95.00% | 32 |
| deep_sensor-8 | Qiskit runtime-bound response | 1.6490 | 95.00% | 320 |
| deep_sensor-8 | Qiskit cached-block response | 1.6805 | 95.00% | 32 |
| deep_sensor-8 | Direct state and tangent | 1.3457 | 95.00% | 0 |
| deep_sensor-8 | Classical task model | 1.2688 | 100.00% | 0 |
| deep_sensor-16 | GroupMemory spectral | 1.6539 | 71.25% | 32 |
| deep_sensor-16 | GroupMemory batched grid | 1.5834 | 71.25% | 32 |
| deep_sensor-16 | Qiskit runtime-bound response | 1.9667 | 71.25% | 576 |
| deep_sensor-16 | Qiskit cached-block response | 1.8742 | 71.25% | 32 |
| deep_sensor-16 | Direct state and tangent | 1.3387 | 71.25% | 0 |
| deep_sensor-16 | Classical task model | 1.2685 | 100.00% | 0 |
| deep_shared-2 | GroupMemory spectral | 1.6120 | 100.00% | 32 |
| deep_shared-2 | GroupMemory batched grid | 1.5950 | 100.00% | 32 |
| deep_shared-2 | Qiskit runtime-bound response | 1.5911 | 100.00% | 144 |
| deep_shared-2 | Qiskit cached-block response | 1.8687 | 100.00% | 32 |
| deep_shared-2 | Direct state and tangent | 1.3527 | 100.00% | 0 |
| deep_shared-2 | Classical task model | 1.2825 | 100.00% | 0 |
| deep_shared-8 | GroupMemory spectral | 1.6081 | 97.50% | 32 |
| deep_shared-8 | GroupMemory batched grid | 1.6110 | 97.50% | 32 |
| deep_shared-8 | Qiskit runtime-bound response | 1.7232 | 97.50% | 336 |
| deep_shared-8 | Qiskit cached-block response | 2.7145 | 97.50% | 32 |
| deep_shared-8 | Direct state and tangent | 1.3626 | 97.50% | 0 |
| deep_shared-8 | Classical task model | 1.2738 | 100.00% | 0 |
| deep_shared-16 | GroupMemory spectral | 1.7127 | 100.00% | 32 |
| deep_shared-16 | GroupMemory batched grid | 1.7565 | 100.00% | 32 |
| deep_shared-16 | Qiskit runtime-bound response | 2.1565 | 100.00% | 592 |
| deep_shared-16 | Qiskit cached-block response | 5.9679 | 100.00% | 32 |
| deep_shared-16 | Direct state and tangent | 1.4616 | 100.00% | 0 |
| deep_shared-16 | Classical task model | 1.2922 | 100.00% | 0 |

## Matched comparisons

Speedups are medians of paired seed ratios, not ratios of displayed marginal times. The descriptive-win rule requires at least five complete seeds, at least three matching repeats per seed, every seed median faster and at least 5% median paired saving. Ranges describe observed seeds, not confidence intervals.

| Workload | Baseline | Group spectral speedup | Seed range | Meets rule |
|---|---|---:|---:|---|
| deep_sensor-2 | GroupMemory batched grid | 1.009× | 0.974–1.221× | False |
| deep_sensor-2 | Qiskit runtime-bound response | 0.996× | 0.947–1.180× | False |
| deep_sensor-2 | Qiskit cached-block response | 1.100× | 1.008–1.181× | True |
| deep_sensor-2 | Direct state and tangent | 0.819× | 0.797–0.853× | False |
| deep_sensor-2 | Classical task model | 0.783× | 0.775–0.846× | False |
| deep_sensor-8 | GroupMemory batched grid | 0.997× | 0.982–1.024× | False |
| deep_sensor-8 | Qiskit runtime-bound response | 1.029× | 1.008–1.079× | False |
| deep_sensor-8 | Qiskit cached-block response | 1.059× | 1.000–1.110× | True |
| deep_sensor-8 | Direct state and tangent | 0.853× | 0.795–0.935× | False |
| deep_sensor-8 | Classical task model | 0.813× | 0.757–0.826× | False |
| deep_sensor-16 | GroupMemory batched grid | 1.003× | 0.875–1.067× | False |
| deep_sensor-16 | Qiskit runtime-bound response | 1.231× | 1.116–1.276× | True |
| deep_sensor-16 | Qiskit cached-block response | 1.168× | 1.055–1.230× | True |
| deep_sensor-16 | Direct state and tangent | 0.811× | 0.757–0.853× | False |
| deep_sensor-16 | Classical task model | 0.780× | 0.760–0.969× | False |
| deep_shared-2 | GroupMemory batched grid | 0.975× | 0.765–1.030× | False |
| deep_shared-2 | Qiskit runtime-bound response | 1.006× | 0.745–1.020× | False |
| deep_shared-2 | Qiskit cached-block response | 1.059× | 0.861–1.194× | False |
| deep_shared-2 | Direct state and tangent | 0.839× | 0.587–1.011× | False |
| deep_shared-2 | Classical task model | 0.774× | 0.591–0.816× | False |
| deep_shared-8 | GroupMemory batched grid | 1.010× | 0.996–1.042× | False |
| deep_shared-8 | Qiskit runtime-bound response | 1.070× | 1.034–1.167× | True |
| deep_shared-8 | Qiskit cached-block response | 1.706× | 1.584–1.735× | True |
| deep_shared-8 | Direct state and tangent | 0.850× | 0.780–0.873× | False |
| deep_shared-8 | Classical task model | 0.797× | 0.710–0.885× | False |
| deep_shared-16 | GroupMemory batched grid | 1.024× | 0.993–1.031× | False |
| deep_shared-16 | Qiskit runtime-bound response | 1.236× | 1.196–1.341× | True |
| deep_shared-16 | Qiskit cached-block response | 3.495× | 3.409–4.104× | True |
| deep_shared-16 | Direct state and tangent | 0.871× | 0.835–1.011× | False |
| deep_shared-16 | Classical task model | 0.759× | 0.732–0.857× | False |

## Interpretation limits

The response has exact integer frequencies from -L through L for L repeated RY occurrences. All response methods receive this same bound and cache lifetime. The Qiskit controls reconstruct coefficients from 2L+1 grid points; GroupMemory spectral propagates coefficient states instead. GroupMemory batched grid is an ablation using dense batched state evolution. The stronger Qiskit control receives the same cached fixed-block matrices. All response methods pay for two independent native off-grid checks at 1e-10.

A measured gain is a specialized classical response-construction/simulation improvement over these named Qiskit/Aer implementations. The code and algorithm are portable; it does not establish an exclusive capability unavailable to Qiskit or a QPU speedup. Dense matrices and coefficient states scale exponentially with qubit count. One shared angle is trained, with a classical classifier head; this is not an independently trainable parameter at every layer.

The direct state/tangent control evaluates the identical model without a persistent Fourier response. The task classical model uses different features and is compared on accuracy as well as cost. Sensor augmented views are related, and the deeper circuit does not assume spatial invariance.

Native Aer calls and dense coefficient arithmetic are different operations. Fewer Aer calls must not be interpreted as fewer physical CNOTs or QPU shots.

Sum of worker process walls: 1024.383 s. Parent measurement loop: 1026.065 s.
