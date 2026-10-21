"""One diagnostic of library startup versus an already imported application.

This is separate from the frozen benchmark. It is not a repeated timing study
or a quantum-advantage test. Run with the package installed:
    python examples/startup_diagnostic.py --output results/startup_diagnostic.json
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import numpy  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.kernel_ridge import KernelRidge  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    import qiskit, qiskit_aer  # noqa: F401
    from group_memory_qml.experiments.derivatives import run
    from threadpoolctl import threadpool_limits
    import_s = time.perf_counter() - started
    measurements = []
    with threadpool_limits(limits=1):
        for experiment, config in [
            ("2A", {"steps": 5, "n_train": 24, "n_test": 24}),
            ("2B", {"steps": 3, "n_train": 12, "n_test": 12}),
        ]:
            for method in ("classical", "group_memory"):
                tick = time.perf_counter()
                result = run(experiment, 938001, method, config)
                elapsed = time.perf_counter() - tick
                measurements.append({
                    "experiment": experiment, "method": method,
                    "application_after_dependency_import_s": elapsed,
                    "metric_name": result["metric_name"],
                    "metric_value": result["metric_value"],
                })
    output = {
        "scope": "One independent diagnostic seed; dependencies imported before timing functions. "
                 "Not pooled with the frozen campaign and not statistical performance evidence. "
                 "Each run constructs a fresh model/cache. Interpreter startup is excluded.",
        "seed": 938001, "import_stage_s": import_s, "measurements": measurements,
        "diagnostic_clock_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
