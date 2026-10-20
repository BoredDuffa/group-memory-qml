"""Run a fresh, bounded sensor-classifier comparison and save every record."""
from __future__ import annotations

import time

_MODULE_STARTED = time.perf_counter()

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from datetime import datetime, timezone

_LABELS = {
    "group_memory": "GroupMemory",
    "qiskit_same_symmetry": "Qiskit + same symmetry",
    "qiskit_exact_cache": "Qiskit exact cache",
    "classical": "Classical model",
}
_COLORS = ["#187d64", "#3c72b4", "#9eabc0", "#bd7231"]


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[846101])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/quickstart"))
    parser.add_argument("--broken-symmetry", action="store_true")
    args = parser.parse_args(argv)
    if not 6 <= args.qubits <= 14 or args.qubits % 2:
        parser.error("--qubits must be an even number from 6 through 14")
    if not 1 <= args.repeats <= 5:
        parser.error("--repeats must be from 1 through 5")
    if not 1 <= args.updates <= 20:
        parser.error("--updates must be from 1 through 20")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be absent or empty; choose a new directory")
    return args


def _summarize(records, methods):
    seeds = sorted({row["seed"] for row in records})
    summaries, per_seed = [], {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        seed_stats = {}
        for seed in seeds:
            current = [row for row in rows if row["seed"] == seed]
            if not current:
                continue
            seed_stats[seed] = {
                key: statistics.median(row[key] for row in current)
                for key in ("pipeline_s", "heldout_accuracy", "forward_evaluations",
                            "gradient_instances", "cx_per_circuit", "forward_cx_work")
            }
        per_seed[method] = seed_stats
        summaries.append({
            "method": method, "data_seeds": len(seed_stats),
            "timed_runs": len(rows),
            "mean_seed_accuracy": statistics.mean(v["heldout_accuracy"] for v in seed_stats.values()),
            "median_seed_pipeline_s": statistics.median(v["pipeline_s"] for v in seed_stats.values()),
            "median_forward_evaluations": statistics.median(v["forward_evaluations"] for v in seed_stats.values()),
            "median_gradient_instances": statistics.median(v["gradient_instances"] for v in seed_stats.values()),
            "median_cx_per_circuit": statistics.median(v["cx_per_circuit"] for v in seed_stats.values()),
            "median_forward_cx_work": statistics.median(v["forward_cx_work"] for v in seed_stats.values()),
        })
    ratios = []
    for baseline in ("qiskit_same_symmetry", "qiskit_exact_cache", "classical"):
        shared = sorted(set(per_seed["group_memory"]) & set(per_seed[baseline]))
        values = [per_seed[baseline][seed]["pipeline_s"] / per_seed["group_memory"][seed]["pipeline_s"]
                  for seed in shared]
        ratios.append({
            "baseline": baseline, "candidate": "group_memory",
            "matched_data_seeds": len(shared),
            "median_paired_speed_ratio": statistics.median(values),
            "ratio_definition": "baseline seed-median time / GroupMemory seed-median time",
            "seed_ratios": [{"seed": seed, "ratio": ratio} for seed, ratio in zip(shared, values)],
        })
    return summaries, ratios


def _plots(output, summaries, records, plt):
    labels = [_LABELS[row["method"]] for row in summaries]
    x = list(range(len(summaries)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    accuracy = [100 * row["mean_seed_accuracy"] for row in summaries]
    durations = [row["median_seed_pipeline_s"] for row in summaries]
    axes[0].bar(x, accuracy, color=_COLORS)
    axes[0].axhline(50, linestyle="--", color="#777777", linewidth=1, label="Balanced chance: 50%")
    axes[0].set(ylim=(0, 110), ylabel="Held-out accuracy (%)", title="Quality on tiny synthetic test sets")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].scatter(x, durations, color=_COLORS, s=70)
    axes[1].set(yscale="log", ylabel="Seconds (log scale)", title="Fresh application time; lower is better")
    for i, (value, duration) in enumerate(zip(accuracy, durations)):
        axes[0].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
        axes[1].text(i, duration * 1.12, f"{duration:.4g}s", ha="center", fontsize=9)
    axes[1].set_ylim(top=max(durations) * 2.5)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=22, ha="right", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CPU simulation · application costs included · full harness wall reported separately", fontsize=11)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for method in ("qiskit_same_symmetry", "qiskit_exact_cache", "group_memory"):
        rows = [row for row in records if row["method"] == method]
        steps = sorted({point["step"] for row in rows for point in row["history"]})
        values = []
        for step in steps:
            # One value per data seed; timing repeats do not create new datasets.
            seed_values = []
            for seed in sorted({row["seed"] for row in rows}):
                losses = [point["train_loss"] for row in rows if row["seed"] == seed
                          for point in row["history"] if point["step"] == step]
                seed_values.append(statistics.median(losses))
            values.append(statistics.mean(seed_values))
        axis.plot([step + 1 for step in steps], values, marker="o", label=_LABELS[method],
                  color="#187d64" if method == "group_memory" else "#aab0ba",
                  linestyle="-" if method == "group_memory" else "--",
                  linewidth=2 if method == "group_memory" else 1)
    axis.set(xlabel="Alternating quantum update", ylabel="Training cross-entropy",
             title="Quantum training trajectories should overlap")
    axis.set_xticks([step + 1 for step in steps])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=9)
    fig.text(.5, -.02, "Loss after fitting each step's head, before its weight update; no derivative through head fitting.",
             ha="center", fontsize=8)
    fig.savefig(output / "learning.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _report(output, summaries, ratios, config):
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Sensor classifier: measured release run", "",
        f"{config['qubits']} qubits; {len(config['seeds'])} data seed(s); "
        f"{config['repeats']} timing repeat(s); {config['updates']} quantum updates.",
        f"Architecture symmetry enabled: {not config['broken_symmetry']}.", "",
        "| Method | Mean seed accuracy | Median seed application time | Forward states | Reverse-gradient instances | Forward CNOT work |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {_LABELS[row['method']]} | {100 * row['mean_seed_accuracy']:.2f}% | "
            f"{row['median_seed_pipeline_s']:.6f} s | {row['median_forward_evaluations']:g} | "
            f"{row['median_gradient_instances']:g} | {row['median_forward_cx_work']:g} |"
        )
    lines += ["", "## Paired descriptive timing", "",
              "Ratios divide each baseline's median repeat time by GroupMemory's median repeat time "
              "within the same data seed, then take the median across seeds. Values above one favor "
              "GroupMemory. The symmetry-aware Qiskit comparison is primary.", ""]
    for ratio in ratios:
        lines.append(f"- {_LABELS[ratio['baseline']]} / GroupMemory: "
                     f"**{ratio['median_paired_speed_ratio']:.3f}×** "
                     f"({ratio['matched_data_seeds']} matched data seed(s)).")
    lines += [
        "", "These small descriptive measurements do not establish a reliable performance advantage. "
        "There are no confidence or significance claims. A repeat is not a new dataset.", "",
        "## Interpretation", "",
        "- Each test seed contains four independent histogram pairs, eight base patterns and 32 related views. "
        "The balanced chance level is 50%; related views are not independent test samples.",
        "- All quantum methods use the same learner and are checked for matching predictions, weights and "
        "partial gradients. They train only two quantum weights.",
        "- Forward CNOT work is forward evaluations × CNOTs per compiled circuit. Avoided evaluations "
        "are not a reduction in the gate count of one circuit.",
        "- Reverse gradients run on the CPU. No QPU jobs, shot noise, or hardware training are measured.",
        "- Application totals include data, compilation, canonicalization, cache, binding, execution, "
        "gradients, fitting and returned-record construction. Imports, validation, plot/file work and "
        "the remaining harness are reported in `wall_accounting.json`.",
        "- Every invocation starts with a fresh engine. Cached results can be reused within that invocation; "
        "changing its weights clears the cache.",
        "", "[Accuracy and runtime](comparison.png) · [Training trajectories](learning.png)", "",
        "Inspect `results.json`, `validation.json`, `environment.json`, and `wall_accounting.json` "
        "before comparing results from different machines.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    """CLI entry point; no training or heavy imports occur merely on import."""
    args = _arguments(argv)
    started = _MODULE_STARTED
    invocation_started = time.perf_counter()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import_started = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
    from .training import METHODS, run_trial, validate_matched_block
    imports_s = time.perf_counter() - import_started

    args.output.mkdir(parents=True, exist_ok=True)
    config = {"qubits": args.qubits, "seeds": args.seeds, "repeats": args.repeats,
              "updates": args.updates, "broken_symmetry": args.broken_symmetry,
              "order_seed": 846919}
    environment = {
        "python": platform.python_version(), "system": platform.system(),
        "release": platform.release(), "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("group-memory-qml", "numpy", "qiskit", "qiskit-aer",
                      "qiskit-algorithms", "scikit-learn", "matplotlib", "threadpoolctl")},
        "thread_limits": {name: os.environ[name] for name in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    records, validations, order_receipts = [], [], []
    summaries, ratios = [], []
    validation_s = report_s = 0.
    status, error = "RUNNING", None
    order_rng = random.Random(config["order_seed"])
    block_order = [(seed, repeat) for repeat in range(args.repeats) for seed in args.seeds]
    order_rng.shuffle(block_order)
    initial_order = list(METHODS)
    order_rng.shuffle(initial_order)
    try:
        with threadpool_limits(limits=1):
            for index, (seed, repeat) in enumerate(block_order):
                # A shuffled cyclic design balances positions every four blocks.
                offset = index % len(METHODS)
                methods = initial_order[offset:] + initial_order[:offset]
                order_receipts.append({"seed": seed, "repeat": repeat, "methods": methods})
                block = []
                for method in methods:
                    record = run_trial(args.qubits, seed, method, args.updates,
                                       symmetry=not args.broken_symmetry)
                    record["repeat"] = repeat
                    records.append(record)
                    block.append(record)
                tick = time.perf_counter()
                try:
                    receipt = validate_matched_block(block)
                    validations.append({"seed": seed, "repeat": repeat, **receipt})
                finally:
                    validation_s += time.perf_counter() - tick
        summaries, ratios = _summarize(records, METHODS)
        tick = time.perf_counter()
        _plots(args.output, summaries, records, plt)
        _report(args.output, summaries, ratios, config)
        report_s += time.perf_counter() - tick
        status = "PASS"
    except Exception as exc:
        status = "FAILED"
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        tick = time.perf_counter()
        _write_json(args.output / "results.json", {
            "status": status, "config": config, "records": records,
            "method_order": order_receipts, "summary": summaries,
            "paired_comparisons": ratios, "error": error,
        })
        _write_json(args.output / "validation.json", {
            "status": status, "matched_blocks": validations,
            "expected_blocks": len(block_order), "passed_blocks": len(validations),
            "error": error, "prediction_and_gradient_atol": 1e-9,
            "independent_numerical_checks": "Run pytest; this receipt checks matched training results.",
        })
        _write_json(args.output / "environment.json", environment)
        report_s += time.perf_counter() - tick
        pipeline_s = sum(row["pipeline_s"] for row in records)
        wall_s = time.perf_counter() - started
        ledger = {
            "status": status,
            "wall_boundary": "From first demo-module clock to this receipt snapshot. "
                             "Includes standard and heavy imports, all methods, validation, plots and result files. "
                             "Excludes interpreter/package startup before that clock, final receipt write, stdout and shutdown.",
            "whole_module_wall_to_receipt_snapshot_s": wall_s,
            "module_to_main_entry_s": invocation_started - started,
            "heavy_import_s": imports_s, "application_pipeline_sum_s": pipeline_s,
            "matched_validation_s": validation_s,
            "plot_report_and_result_write_s": report_s,
            "other_harness_and_standard_import_s": wall_s - imports_s - pipeline_s - validation_s - report_s,
            "actual_application_runs": len(records),
            "expected_application_runs": len(block_order) * len(METHODS),
            "data_seed_count": len(args.seeds), "timing_repeats_per_seed": args.repeats,
            "thread_scope": "Thread environment set before heavy imports; threadpoolctl limits timed learner work to one.",
        }
        _write_json(args.output / "wall_accounting.json", ledger)
    print(f"PASS: {len(validations)} matched blocks, {len(records)} application runs.")
    print(f"Results: {args.output / 'summary.md'}")
    print(f"Wall accounting: {args.output / 'wall_accounting.json'}")
    print("CPU simulation only. Small descriptive timings do not establish a reliable speed advantage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
