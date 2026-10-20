# Contributing

The most useful contributions make the supported behavior clearer, the comparison fairer, or the experiment easier to reproduce. Incorrect reuse, slower runtimes, and strong classical baselines are useful findings.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m group_memory_qml.demo --output results/contribution-check
```

Use a new output directory for a fresh run. Preserve the seed, configuration, package versions, and raw measurements when sharing a comparison. Do not commit credentials, machine-specific absolute paths, unrelated data, or large generated files.

## Report a result

Use the benchmark issue template and include:

- The exact command, commit, Python/package versions, operating system, and CPU.
- Whether the run was fresh, its seed/repeat counts, and any concurrent load.
- Both the symmetry-aware Qiskit comparison and the classical control.
- Correctness results, accuracy, application timings, and full-run overhead.
- Raw output files or a small reproducible extract, including failures.

Do not describe a speedup over exact-input caching as a speedup over equally symmetry-aware Qiskit. Multiple timing repeats on one dataset are not independent datasets. Label CPU simulation, derivative work, and any real hardware measurements separately.

## Change the method

Open an issue or draft pull request describing the supported equivalence before extending the cache. Explain how inputs and outputs transform, which parameters the rule assumes, how cache entries become invalid, and what happens when the symmetry is broken.

A useful test should compare the optimized output with an independently evaluated reference and fail for an incorrect reuse. Preserve explicit numerical tolerances. Add a counterexample or disablement test for unsupported inputs; do not broaden the claimed symmetry from a few passing examples.

Keep fair comparisons intact. If a new optimization is available to the Qiskit baseline, implement or account for it there too. Include its preprocessing and setup cost. Any approximate matching needs a separate opt-in API, error analysis, and clearly labeled results; do not silently change the exact cache.

## Pull requests

Keep each change focused. Describe the concrete behavior, how you checked correctness, and the benchmark scope. Link relevant related work without claiming novelty from an absence of references.

By submitting a contribution, you agree that it can be distributed under this project's MIT license.
