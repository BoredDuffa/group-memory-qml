"""Audit saved parameter-response trials and report complete-process comparisons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .deep_suite import METHODS, atomic

TOLERANCE = 1e-10
REFERENCE = "qiskit_response"
PRIMARY = "group_spectral"
MATCHED = tuple(m for m in METHODS if m != "classical")
LABELS = {
    "group_spectral": "GroupMemory spectral",
    "group_grid": "GroupMemory batched grid",
    "qiskit_response": "Qiskit runtime-bound response",
    "qiskit_blocks": "Qiskit cached-block response",
    "classical_direct": "Direct state and tangent",
    "classical": "Classical task model",
}


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(directory):
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text())
    status = json.loads((directory/"status.json").read_text())
    rows, blocks = [], {}
    for index, spec in enumerate(config["plan"]):
        path = directory/"attempts"/f"{index:05d}.json"
        if not path.exists():
            row = {"spec": spec, "status": "UNATTEMPTED"}
        else:
            row = json.loads(path.read_text())
            if row["spec"] != spec:
                raise AssertionError("Attempt/plan mismatch")
        if row["status"] == "SUCCESS":
            if row["source_hashes"] != config["source_hashes"]:
                raise AssertionError("Frozen source mismatch")
            if row["method"] != spec["method"] or row["experiment"] != spec["experiment"]:
                raise AssertionError("Result identity mismatch")
            if row["seed"] != spec["seed"]:
                raise AssertionError("Seed mismatch")
            for field in ("process_wall_s", "application_s"):
                value = row[field]
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise AssertionError(f"Invalid measured timing: {field}")
            if row["application_s"] > row["process_wall_s"]:
                raise AssertionError("Application time exceeds full process time")
            digest = row["diagnostics"].get(
                "dataset_sha256", row["diagnostics"].get("data_sha256")
            )
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise AssertionError("Missing or invalid dataset hash")
            values = np.asarray(row["validation_vector"])
            if values.ndim != 1 or not np.isfinite(values).all():
                raise AssertionError("Invalid validation values")
            if not np.isfinite(row["metric_value"]):
                raise AssertionError("Nonfinite task metric")
        rows.append(row)
        key = (spec["condition"], spec["seed"], spec["repeat"])
        blocks.setdefault(key, {})[spec["method"]] = row

    validation, good = [], set()
    max_error = 0.0
    for key, block in blocks.items():
        receipt = dict(zip(("condition", "seed", "repeat"), key))
        if any(block.get(m, {}).get("status") != "SUCCESS" for m in MATCHED):
            receipt["status"] = "INCOMPLETE"
            validation.append(receipt)
            continue
        reference = block[REFERENCE]
        reference_data = reference["diagnostics"].get(
            "dataset_sha256", reference["diagnostics"].get("data_sha256")
        )
        worst = 0.
        for method in MATCHED:
            row = block[method]
            digest = row["diagnostics"].get(
                "dataset_sha256", row["diagnostics"].get("data_sha256")
            )
            if digest != reference_data:
                raise AssertionError("Unequal datasets")
            actual = np.asarray(row["validation_vector"])
            expected = np.asarray(reference["validation_vector"])
            if actual.shape != expected.shape:
                raise AssertionError(f"{key}: incompatible training trajectories")
            error = float(np.max(np.abs(actual-expected), initial=0))
            worst = max(worst, error)
            if error > TOLERANCE:
                raise AssertionError(f"{key}/{method}: correctness failure {error}")
            if row["metric_name"] != reference["metric_name"]:
                raise AssertionError("Unequal task metrics")
            if abs(row["metric_value"]-reference["metric_value"]) > TOLERANCE:
                raise AssertionError("Task quality mismatch")
        # Spectral propagation, batched states and native Aer calls are
        # different CPU algorithms. Their native-call counters need not match.
        # Validate the common circuit signature once provided by the engines.
        signatures = [block[m]["diagnostics"]["response"].get("source_sha256")
                      for m in MATCHED]
        if not all(signatures) or len(set(signatures)) != 1:
            raise AssertionError("Different source circuits across response methods")
        architectures = [block[m]["diagnostics"]["response"].get("architecture_sha256")
                         for m in MATCHED]
        if not all(architectures) or len(set(architectures)) != 1:
            raise AssertionError("Different architectures across same-model methods")
        native = [block[m]["diagnostics"]["response"].get("native_sha256")
                  for m in MATCHED if m != "classical_direct"]
        if not all(native) or len(set(native)) != 1:
            raise AssertionError("Different native circuits across response methods")
        fixed = [block[m]["diagnostics"]["response"].get("fixed_block_sha256")
                 for m in ("group_spectral", "group_grid", "qiskit_blocks")]
        if not all(fixed) or len({tuple(x) for x in fixed}) != 1:
            raise AssertionError("Different fixed block operators")
        if block.get("classical", {}).get("status") == "SUCCESS":
            classical = block["classical"]
            digest = classical["diagnostics"].get(
                "dataset_sha256", classical["diagnostics"].get("data_sha256")
            )
            if digest != reference_data:
                raise AssertionError("Classical task data mismatch")
        receipt.update(status="PASS", max_absolute_error=worst,
                       dataset_sha256=reference_data)
        max_error = max(max_error, worst)
        validation.append(receipt)
        good.add(key)

    seeds, comparisons, summary = [], [], []
    for condition in config["conditions"]:
        cid = condition["id"]
        for seed in config["seeds"]:
            keys = [(cid, seed, rep) for rep in range(config["repeats"])]
            if not all(key in good for key in keys):
                continue
            for method in METHODS:
                selected = [blocks[key][method] for key in keys]
                if any(row["status"] != "SUCCESS" for row in selected):
                    continue
                seeds.append({
                    "condition": cid, "seed": seed, "method": method,
                    "trials": len(selected),
                    "process_wall_s": statistics.median(r["process_wall_s"] for r in selected),
                    "application_s": statistics.median(r["application_s"] for r in selected),
                    "metric_value": statistics.mean(r["metric_value"] for r in selected),
                    "evaluations": statistics.median(r["evaluations"] for r in selected),
                    "total_cx": statistics.median(r["total_cx"] for r in selected),
                })
        for method in METHODS:
            selected = [r for r in seeds if r["condition"] == cid and r["method"] == method]
            if selected:
                summary.append({
                    "condition": cid, "method": method, "seeds": len(selected),
                    "process_wall_s": statistics.median(r["process_wall_s"] for r in selected),
                    "application_s": statistics.median(r["application_s"] for r in selected),
                    "accuracy": statistics.mean(r["metric_value"] for r in selected),
                    "evaluations": statistics.median(r["evaluations"] for r in selected),
                    "total_cx": statistics.median(r["total_cx"] for r in selected),
                })
        primary = {r["seed"]: r for r in seeds if r["condition"] == cid and r["method"] == PRIMARY}
        for method in METHODS[1:]:
            other = {r["seed"]: r for r in seeds if r["condition"] == cid and r["method"] == method}
            common = sorted(set(primary) & set(other))
            if not common:
                continue
            ratios = [other[s]["process_wall_s"]/primary[s]["process_wall_s"] for s in common]
            savings = [1-primary[s]["process_wall_s"]/other[s]["process_wall_s"] for s in common]
            applications = [other[s]["application_s"]/primary[s]["application_s"] for s in common]
            comparisons.append({
                "condition": cid, "baseline": method, "matched_seeds": len(common),
                "paired_speedup": statistics.median(ratios),
                "minimum_seed_speedup": min(ratios), "maximum_seed_speedup": max(ratios),
                "median_paired_saving": statistics.median(savings),
                "application_speedup": statistics.median(applications),
                "descriptive_win": bool(config["repeats"] >= 3 and len(common) >= 5 and min(ratios) > 1
                                        and statistics.median(savings) >= .05),
            })
    attempts = [{
        **r["spec"], "config": json.dumps(r["spec"].get("config", {}), sort_keys=True),
        "status": r["status"],
        **{k: r.get(k) for k in ("process_wall_s", "application_s", "metric_value",
                                "evaluations", "total_cx")},
    } for r in rows]
    return {
        "planned": len(rows), "successful": sum(r["status"] == "SUCCESS" for r in rows),
        "validation_planned": len(blocks), "validation_passed": len(good),
        "maximum_absolute_error": max_error, "tolerance": TOLERANCE,
        "conditions": config["conditions"], "status": status,
        "sum_process_wall_s": sum(r.get("process_wall_s", 0) for r in rows),
        "summary": summary, "comparisons": comparisons,
    }, attempts, validation, seeds


def report(directory):
    started = time.perf_counter()
    directory = Path(directory)
    result, attempts, validation, seeds = audit(directory)
    atomic(directory/"analysis.json", result)
    for name, records in (("attempts", attempts), ("validation", validation),
                          ("seed_blocks", seeds), ("summary", result["summary"]),
                          ("comparisons", result["comparisons"])):
        _write_csv(directory/f"{name}.csv", records)
    lines = [
        "# Deeper QML: response construction and training", "",
        "These are new interleaved architectures and fresh local CPU timings. "
        "No QPU execution, finite shots, or pooled historical timings.", "",
        f"{result['successful']}/{result['planned']} successful attempts; "
        f"{result['validation_passed']}/{result['validation_planned']} matched "
        f"blocks; maximum trajectory error {result['maximum_absolute_error']:.3g}, "
        "with unchanged absolute tolerance 1e-10.", "",
        "Each primary timer encloses the entire fresh worker process, including imports, "
        "source and block construction, compilation, response construction and checking, "
        "data handling, cache planning, training, prediction, serialization and exit. "
        "Separate receipts record parent overhead, preflight, pilot and reports.", "",
        "| Workload | Method | Full process (s) | Accuracy | Native-basis Aer evaluations only |",
        "|---|---|---:|---:|---:|",
    ]
    for r in result["summary"]:
        lines.append(f"| {r['condition']} | {LABELS[r['method']]} | "
                     f"{r['process_wall_s']:.4f} | {r['accuracy']:.2%} | "
                     f"{r['evaluations']:.0f} |")
    lines += [
        "", "## Matched comparisons", "",
        "Speedups are medians of paired seed ratios, not ratios of displayed marginal "
        "times. The descriptive-win rule requires at least five complete seeds, at least "
        "three matching repeats per seed, every seed median faster and at least 5% "
        "median paired saving. Ranges describe observed seeds, not confidence intervals.", "",
        "| Workload | Baseline | Group spectral speedup | Seed range | Meets rule |",
        "|---|---|---:|---:|---|",
    ]
    for r in result["comparisons"]:
        lines.append(f"| {r['condition']} | {LABELS[r['baseline']]} | "
                     f"{r['paired_speedup']:.3f}× | {r['minimum_seed_speedup']:.3f}–"
                     f"{r['maximum_seed_speedup']:.3f}× | {r['descriptive_win']} |")
    lines += [
        "", "## Interpretation limits", "",
        "The response has exact integer frequencies from -L through L for L repeated "
        "RY occurrences. All response methods receive this same bound and cache "
        "lifetime. The Qiskit controls reconstruct coefficients from 2L+1 grid points; "
        "GroupMemory spectral propagates coefficient states instead. GroupMemory "
        "batched grid is an ablation using dense batched state evolution. The stronger "
        "Qiskit control receives the same cached fixed-block matrices. All response "
        "methods pay for two independent native off-grid checks at 1e-10.",
        "",
        "A measured gain is a specialized classical response-construction/simulation "
        "improvement over these named Qiskit/Aer implementations. The code and algorithm "
        "are portable; it does not establish an exclusive capability unavailable to "
        "Qiskit or a QPU speedup. Dense matrices and coefficient states scale exponentially "
        "with qubit count. One shared angle is trained, with a classical classifier head; "
        "this is not an independently trainable parameter at every layer.",
        "",
        "The direct state/tangent control evaluates the identical model without a "
        "persistent Fourier response. The task classical model uses different features "
        "and is compared on accuracy as well as cost. Sensor augmented views are related, "
        "and the deeper circuit does not assume spatial invariance.",
        "",
        "Native Aer calls and dense coefficient arithmetic are different operations. "
        "Fewer Aer calls must not be interpreted as fewer physical CNOTs or QPU shots.",
        "",
        f"Sum of worker process walls: {result['sum_process_wall_s']:.3f} s. "
        f"Parent measurement loop: {result['status'].get('last_invocation_wall_s', 0):.3f} s.",
    ]
    (directory/"summary.md").write_text("\n".join(lines)+"\n")
    from matplotlib import pyplot as plt
    conditions = [c["id"] for c in result["conditions"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    relevant = [r for r in result["comparisons"]
                if r["baseline"] in ("qiskit_response", "qiskit_blocks")]
    limit = max([1.2]+[r["maximum_seed_speedup"] for r in relevant])*1.06
    for ax, baseline in zip(axes, ("qiskit_response", "qiskit_blocks")):
        for i, cid in enumerate(conditions):
            selected = [r for r in relevant
                        if r["condition"] == cid and r["baseline"] == baseline]
            if not selected:
                continue
            r = selected[0]
            ax.plot([r["minimum_seed_speedup"], r["maximum_seed_speedup"]], [i, i],
                    color="#2563eb", linewidth=2)
            ax.scatter(r["paired_speedup"], i, color="#163c80", s=38)
        ax.axvline(1, linestyle="--", color="#64748b", linewidth=1)
        ax.set(title=LABELS[baseline], xlabel="Full-process speedup of GroupMemory spectral",
               xlim=(0, limit))
        ax.set_yticks(range(len(conditions)), conditions)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Deeper QML — paired seed medians", fontsize=14)
    fig.text(.5, .01, "Ranges show observed seeds. Above 1 favors GroupMemory spectral.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/"paired_speedups.png", dpi=150)
    plt.close(fig)
    atomic(directory/"report_wall.json", {"seconds": time.perf_counter()-started})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    r = report(args.directory)
    print(json.dumps({k: r[k] for k in ("planned", "successful",
        "validation_planned", "validation_passed", "maximum_absolute_error",
        "sum_process_wall_s")}, indent=2))

