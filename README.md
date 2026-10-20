# GroupMemory QML

**Reuse equivalent forward and gradient computations in a small quantum machine learning experiment—and measure whether the bookkeeping pays for itself.**

GroupMemory is an experimental **classical cache around quantum computations**. This release supports a fixed, symmetric ring circuit and the known rotations and reflections of its inputs. It does not discover the symmetry of arbitrary circuits.

The runnable sensor classifier compares:

| Method | What it can reuse |
|---|---|
| GroupMemory | Exact inputs and symmetry-equivalent inputs, using a dictionary |
| Symmetry-aware Qiskit | The same symmetry and canonical jobs, using a vectorized unique table |
| Exact-cache Qiskit | Exact repeated inputs, without symmetry-based reuse |
| Classical model | Fourier power and circular-correlation features with logistic regression |

**The equally informed Qiskit implementation is the primary baseline.** In the original experiment, GroupMemory substantially reduced work relative to exact-input caching, but did not show a reliable advantage over symmetry-aware Qiskit. The classical model was faster and at least as accurate. This repository makes that result inspectable and reproducible.

## Try it

Use Python 3.11 or newer. Package installation needs an internet connection; the demo itself generates its data locally and needs no account, credentials, downloads, or QPU access.

```bash
git clone https://github.com/BoredDuffa/group-memory-qml.git
cd group-memory-qml
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m group_memory_qml.demo --output results/quickstart
pytest
```

On Windows, activate with `.venv\Scripts\activate` instead. The quickstart uses six qubits, one data seed, one timing repeat, and three quantum parameter updates. A small local CPU run is sufficient; runtime depends on your machine.

Run the larger sensor comparison explicitly:

```bash
python -m group_memory_qml.demo \
  --qubits 10 \
  --seeds 846101 846102 846103 \
  --repeats 2 \
  --output results/my-run
```

The command writes a new results directory with `summary.md`, `summary.csv`, accuracy/runtime and learning plots, full `results.json`, `validation.json`, `environment.json`, and `wall_accounting.json`. It refuses to overwrite a nonempty directory. For an interactive walkthrough, open [the sensor classifier notebook](examples/sensor_classifier.ipynb). Existing saved results are evidence, not a substitute for running the command on your machine.

Use `--broken-symmetry` with a new output directory to make the circuit's site gains unequal and verify that orbit reuse is disabled. `python -m group_memory_qml.demo --help` lists the bounded run options.

## What the demo learns

The binary task distinguishes contiguous positive/negative sensor readings from alternating readings around a ring. Each paired example has the same amplitude histogram, so spatial arrangement matters. Inputs are encoded as rotation angles.

Each split contains four independently generated histogram pairs: eight base patterns, expanded into 32 related views by rotations and reflections. Bases are split before augmentation, with explicit checks against train/test orbit leakage. The nominal balanced chance accuracy is 50%, but **32 augmented test views are not 32 independent test examples**.

The quantum model trains two shared circuit parameters for three updates, alternating with a fitted classical logistic head. The classical control uses invariant Fourier-power and circular-correlation features. This is a small synthetic learning experiment, not a large-model training benchmark.

## Original reference result

The following measurements come from the original **2026-10-19 local CPU experiment**, across three data seeds and two timing repeats. They are historical reference values, not promised runtimes or measurements of a fresh installation of this package.

| 10-qubit sensor classifier | Held-out accuracy¹ | Median application time |
|---|---:|---:|
| GroupMemory | 95.83% | 0.555 s |
| Symmetry-aware Qiskit | 95.83% | 0.564 s |
| Classical model | 100% | 0.00347 s |

¹ Mean accuracy across the three data seeds. Timing medians and paired speed ratios are different statistics.

Compared with the **secondary exact-cache Qiskit control**, GroupMemory used 75% fewer forward states and reverse-gradient instances, with a 4.04× paired speed ratio. Symmetry-aware Qiskit received the same reduction. The original paired timing interval for GroupMemory versus symmetry-aware Qiskit included a tie.

The avoided forward circuits also avoid their CNOTs. This is **not a reduction in the native CNOT count of each circuit**. Reverse-gradient instances are classical derivative computations and are not QPU executions.

## What is included in the clock

Every measured application starts with a fresh engine and includes data generation, circuit setup and compilation, canonicalization and cache lookup, parameter binding, forward simulation, gradients, output reconstruction, head fitting, prediction, and result construction.

Imports, correctness validation, benchmark orchestration, plotting, and file writing are reported separately, alongside elapsed wall time from the CLI module's first clock. The receipt explicitly excludes interpreter startup before that clock and its own final write, console output, and shutdown; an external process timer can measure those too. Inspect both the application comparison and the wall receipt. Repeated calls within an application can hit its cache; separate timing repeats do not share a trained engine or its results.

The quantum methods use the same circuit, compilation settings, double-precision CPU simulator, derivative algorithm, and canonical job ordering. “Symmetry-aware Qiskit” means an explicitly implemented application-aware wrapper; it does not imply that a stock transpiler pass automatically discovers this symmetry.

## Correctness comes first

The tests check output restoration against direct statevector evaluation, shared-parameter gradients against finite differences, cache invalidation after weight changes, broken-symmetry fallback, and train/test separation. Benchmark comparisons also check agreement between the quantum methods.

Reuse is exact with respect to the supplied input representation: the cache does not round similar inputs together. Floating-point simulation and differentiation are checked with explicit numerical tolerances.

Read [the method and limitations](docs/method.md) for the supported symmetry, gradient definition, and timing boundaries.

For the equations and proof, read the [two-page mathematical note](docs/groupmemory_math.pdf)
or its [LaTeX source](docs/groupmemory_math.tex).

## Verified release run

The packaged code was installed into a clean environment and passed **40 correctness tests**.
The [fresh 10-qubit run](results/release-validation/sensor-10q/summary.md) contains three
data seeds, two timing repeats, all four methods and matching quantum training trajectories.
GroupMemory and equally informed Qiskit both achieved **95.83%** mean accuracy, using
40 forward states and 24 reverse-gradient instances per application versus 160 and 96
for exact-input caching. Their application medians were **0.684 s** and **0.688 s**;
the median paired speed ratio was **0.994×**, so there is no demonstrated speed advantage
over the primary baseline. The classical model achieved **100%** in **0.00346 s**.

The complete fresh benchmark process took **27.40 seconds**, including startup, all 24
applications, validation, figures, files and shutdown. This aggregate process cost is
reported separately from per-method application time; it is not silently discarded or
arbitrarily assigned to one method. See its [external wall receipt](results/release-validation/sensor-10q/external_process_wall.json).
Installation time is excluded. These are machine-specific observations, not performance guarantees.

![Measured release accuracy and application time](results/release-validation/sensor-10q/comparison.png)

## Help improve it

Useful feedback includes a reproducible incorrect reuse, a stronger baseline, a workload with useful exact symmetry, or evidence that canonicalization costs more than it saves. Negative results are welcome.

- [Report a bug](https://github.com/BoredDuffa/group-memory-qml/issues/new?template=bug_report.yml)
- [Share a benchmark or use case](https://github.com/BoredDuffa/group-memory-qml/issues/new?template=benchmark_feedback.yml)
- [Contribution guide](CONTRIBUTING.md)

There is no claim here of new group theory, general automatic symmetry inference, QPU speedup, or quantum advantage over the classical model. The contribution is a small implementation and an auditable comparison that others can test and extend.

Released under the [MIT license](LICENSE).
