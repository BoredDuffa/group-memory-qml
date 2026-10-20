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

from .response_suite import METHODS, atomic

TOLERANCE = 1e-10
REFERENCE = "qiskit_symmetry"
PRIMARY = "group_memory_response"
MATCHED = tuple(m for m in METHODS if m != "classical")
LABELS = {
    "group_memory_response": "GroupMemory response",
    "qiskit_response": "Qiskit + same response",
    "group_memory_previous": "Previous GroupMemory",
    "qiskit_symmetry": "Previous symmetry Qiskit",
    "classical_same_model": "Analytic identical model",
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
        for field in ("evaluations", "total_cx"):
            if block[PRIMARY][field] != block["qiskit_response"][field]:
                raise AssertionError("Response baselines perform unequal native work")
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
    table = { (r["condition"], r["method"]): r for r in result["summary"] }
    lines = [
        "# Exact parameter-response memory: measured comparison", "",
        "Ideal CPU simulation of two supported shallow architectures. No QPU execution.",
        "",
        f"Successful attempts: {result['successful']}/{result['planned']}. "
        f"Matched blocks: {result['validation_passed']}/{result['validation_planned']}. "
        f"Maximum output/trajectory error: {result['maximum_absolute_error']:.3g}; "
        "the absolute tolerance remains 1e-10.",
        "",
        "Primary time includes the entire fresh process: imports, data, circuit compilation, "
        "all response probes and off-grid checks, cache planning, training, predictions, "
        "serialization and shutdown. The parent ledger and independent tests/reporting are separate. "
        "Application clocks may include lazy imports and are not uniformly warm-service timings.",
        "",
        "| Workload | Group response | Previous symmetry Qiskit | Qiskit + response | Analytic same model | Classical task | Accuracy: same model / task |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in result["conditions"]:
        cid = c["id"]
        def cell(method, key="process_wall_s", fmt=".3f"):
            row = table.get((cid, method))
            return format(row[key], fmt) if row else "missing"
        lines.append(f"| {cid} | {cell(PRIMARY)} s | {cell(REFERENCE)} s | "
                     f"{cell('qiskit_response')} s | {cell('classical_same_model')} s | "
                     f"{cell('classical')} s | {cell(PRIMARY, 'accuracy')} / "
                     f"{cell('classical', 'accuracy')} |")
    lines += [
        "", "## Paired runtime and native work", "",
        "Ratios are medians of matched per-seed ratios, not ratios of marginal medians. "
        "A descriptive win requires five complete seeds with at least three matching trials each, all seed medians faster, and at "
        "least 5% median paired saving. This is not a significance or universal claim.",
        "",
        "| Workload | Baseline | Paired speedup | Seed range | Meets rule |",
        "|---|---|---:|---:|---|",
    ]
    for r in result["comparisons"]:
        lines.append(f"| {r['condition']} | {LABELS[r['baseline']]} | "
                     f"{r['paired_speedup']:.3f}× | "
                     f"{r['minimum_seed_speedup']:.3f}–{r['maximum_seed_speedup']:.3f}× | "
                     f"{r['descriptive_win']} |")
    lines += [
        "", "| Workload | Group native CNOT work | Previous symmetry Qiskit | Qiskit + response |",
        "|---|---:|---:|---:|",
    ]
    for c in result["conditions"]:
        cid = c["id"]
        values = [str(table[(cid, m)]["total_cx"]) if (cid, m) in table else "missing"
                  for m in (PRIMARY, REFERENCE, "qiskit_response")]
        lines.append(f"| {cid} | {' | '.join(values)} |")
    lines += [
        "", "CNOT work counts actual native probe/direct forward simulations. Sensor reverse "
        "gradients are classical state propagations, so their work is included in time but "
        "not this forward CNOT count. Avoided executions do not shorten the underlying circuit.",
        "",
        "The identical-response Qiskit control receives the same basis, probes, verification "
        "policy and cache lifetime. A win over the previous baseline establishes the value "
        "of parameter-response reuse, not an exclusive GroupMemory advantage. The analytic "
        "control computes the identical circuit outputs and derivatives without any Qiskit probes. "
        "These architectures admit efficient classical formulas. Task-model accuracy and "
        "timing concern a different learner; all held-out views are not independent data.",
        "",
        f"Sum of method process walls: {result['sum_process_wall_s']:.3f} s. "
        f"Latest parent loop wall: {result['status'].get('last_invocation_wall_s', 0):.3f} s.",
    ]
    (directory/"summary.md").write_text("\n".join(lines)+"\n")
    from matplotlib import pyplot as plt
    conditions = [c["id"] for c in result["conditions"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, baseline, title in zip(
        axes, (REFERENCE, "qiskit_response"),
        ("Versus previous symmetry Qiskit", "Versus Qiskit with the same response reuse"),
    ):
        for index, cid in enumerate(conditions):
            matching = [r for r in result["comparisons"]
                        if r["condition"] == cid and r["baseline"] == baseline]
            if not matching:
                continue
            row = matching[0]
            ax.plot([row["minimum_seed_speedup"], row["maximum_seed_speedup"]],
                    [index, index], color="#2563eb", linewidth=2)
            ax.scatter(row["paired_speedup"], index, color="#163c80", s=38)
        ax.axvline(1, linestyle="--", color="#64748b", linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Full-process speedup of GroupMemory")
        ax.grid(axis="x", alpha=.2)
        ax.set_yticks(range(len(conditions)), conditions)
    relevant = [r for r in result["comparisons"] if r["baseline"] in (REFERENCE, "qiskit_response")]
    high = max([1.2]+[r["maximum_seed_speedup"] for r in relevant])
    for ax in axes:
        ax.set_xlim(0, high*1.06)
    axes[0].invert_yaxis()
    fig.suptitle("Exact parameter-response reuse — matched seed medians", fontsize=14)
    fig.text(.5, .01, "Ranges show observed seeds, not confidence intervals. Above 1 favors GroupMemory.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/"paired_speedups.png", dpi=150)
    plt.close(fig)
    body = html.escape("\n".join(lines))
    (directory/"results.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Parameter response results</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap;line-height:1.5}img{max-width:100%}</style></head><body>"
        "<h1>Parameter-response memory</h1><img src='paired_speedups.png' "
        "alt='Matched full-process speed ratios'><pre>"+body+"</pre></body></html>"
    )
    atomic(directory/"report_wall.json", {"seconds": time.perf_counter()-started})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    result = report(args.directory)
    print(json.dumps({k: result[k] for k in (
        "planned", "successful", "validation_planned", "validation_passed",
        "maximum_absolute_error", "sum_process_wall_s"
    )}, indent=2))
