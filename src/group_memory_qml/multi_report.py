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

from .multi_suite import METHODS, atomic

TOLERANCE = 1e-10
REFERENCE = "qiskit_response"
PRIMARY = "group_spectral"
MATCHED = tuple(m for m in METHODS if m != "classical")
LABELS = {
    "group_spectral": "GroupMemory spectral",
    "group_grid": "GroupMemory batched grid",
    "qiskit_response": "Qiskit diagonal response",
    "qiskit_shift": "Qiskit diagonal occurrence shifts",
    "classical_adjoint": "Direct state and adjoint",
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
    rows, blocks, operational_signatures = [], {}, {}
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
            requested = spec.get("config", {})
            actual_config = row.get("config")
            if not isinstance(actual_config, dict):
                raise AssertionError("Missing result configuration")
            if any(actual_config.get(key) != value for key, value in requested.items()):
                raise AssertionError("Requested/result configuration mismatch")
            for key, default in (("n_qubits", 8), ("n_parameters", 2), ("occurrences", 4)):
                expected = requested.get(key, default)
                if type(actual_config.get(key)) is not int or actual_config[key] != expected:
                    raise AssertionError(f"Incorrect circuit configuration: {key}")
                if key != "occurrences":
                    if type(row.get(key)) is not int or row[key] != expected:
                        raise AssertionError(f"Incorrect result circuit dimension: {key}")
            if spec["method"] != "classical":
                response = row.get("diagnostics", {}).get("response", {})
                for key in ("n_qubits", "n_parameters", "occurrences"):
                    if response.get(key) != actual_config[key]:
                        raise AssertionError(f"Incorrect response circuit dimension: {key}")
                parameters, repetitions = actual_config["n_parameters"], actual_config["occurrences"]
                expected_support = {
                    "layers": parameters*repetitions,
                    "response_degree_per_parameter": [repetitions]*parameters,
                    "response_shape": [2*repetitions+1]*parameters,
                    "response_grid_size": (2*repetitions+1)**parameters,
                    "response_state_support_size": (repetitions+1)**parameters,
                }
                if any(response.get(key) != value for key, value in expected_support.items()):
                    raise AssertionError("Incorrect exact response support/cardinality")
                if spec["method"] in ("group_spectral", "group_grid", "qiskit_response"):
                    if response.get("full_signed_frequency_tensor") is not True:
                        raise AssertionError("Missing full signed response support")
                if spec["method"] != "classical_adjoint":
                    digest = response.get("native_sha256")
                    if (not isinstance(digest, str) or len(digest) != 64
                            or any(c not in "0123456789abcdef" for c in digest)):
                        raise AssertionError("Missing or invalid operational native signature")
                    identity = spec["condition"], spec["method"]
                    if operational_signatures.setdefault(identity, digest) != digest:
                        raise AssertionError("Operational native circuit changed within method")
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
                  for m in MATCHED if m not in ("classical_adjoint", "qiskit_shift")]
        if not all(native) or len(set(native)) != 1:
            raise AssertionError("Different native circuits across response methods")
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


def strongest_qiskit(seeds, config):
    """Conservative readout: give Qiskit the faster control per paired seed."""
    output = []
    for c in config["conditions"]:
        cid = c["id"]
        indexed = {(r["seed"], r["method"]): r
                   for r in seeds if r["condition"] == cid}
        details = []
        for seed in config["seeds"]:
            methods = (PRIMARY, "qiskit_response", "qiskit_shift")
            if not all((seed, m) in indexed for m in methods):
                continue
            measured = {m: indexed[seed, m]["process_wall_s"] for m in methods}
            control = min(methods[1:], key=lambda m: measured[m])
            details.append(dict(seed=seed, times=measured, selected_control=control,
                                speedup=measured[control]/measured[PRIMARY]))
        if not details:
            continue
        ratios = [r["speedup"] for r in details]
        savings = [1-1/r for r in ratios]
        output.append(dict(condition=cid, matched_seeds=len(details),
                           paired_speedup=statistics.median(ratios),
                           minimum_seed_speedup=min(ratios),
                           maximum_seed_speedup=max(ratios),
                           median_paired_saving=statistics.median(savings),
                           descriptive_win=bool(config["repeats"] >= 3 and len(details) >= 5
                               and min(ratios) > 1 and statistics.median(savings) >= .05),
                           details=details,
                           scope="Faster Qiskit control chosen separately per seed; a conservative comparison, not a deployed selector"))
    return output


def report(directory):
    """Publish every planned condition, retaining incomplete/error coverage."""
    started = time.perf_counter()
    directory = Path(directory)
    result, attempts, validation, seeds = audit(directory)
    config = json.loads((directory/"config.json").read_text())
    strongest = strongest_qiskit(seeds, config)
    atomic(directory/"strongest_qiskit.json", strongest)
    atomic(directory/"analysis.json", result)
    for name, records in (("attempts", attempts), ("validation", validation),
                          ("seed_blocks", seeds), ("summary", result["summary"]),
                          ("comparisons", result["comparisons"])):
        _write_csv(directory/f"{name}.csv", records)
    lines = [
        "# Scaling response memory: qubits and independent parameters", "",
        "Fresh local CPU simulations. The old dense-block measurements are not pooled "
        "with these matrix-free architectures. No QPU or finite-shot claim.", "",
        f"{result['successful']}/{result['planned']} successful attempts; "
        f"{result['validation_passed']}/{result['validation_planned']} matched blocks. "
        f"Maximum complete-trajectory error: {result['maximum_absolute_error']:.3g}; "
        "unchanged absolute tolerance 1e-10.", "",
        "Each process clock encloses imports, data, preparation, compilation, response "
        "construction and native checks, cache planning, all parameter updates, head "
        "fitting, prediction, serialization and exit. Parent orchestration, preflight, "
        "resume smoke and reporting are separate measured costs. Installation and "
        "development are excluded.", "",
        "| Condition | Qubits | Independent angles | Occurrences each | Layers | State coefficients | Response grid points |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in result["conditions"]:
        cfg = c["config"]
        p, r = cfg["n_parameters"], cfg["occurrences"]
        lines.append(f"| {c['id']} | {cfg['n_qubits']} | {p} | {r} | {p*r} | "
                     f"{(r+1)**p} | {(2*r+1)**p} |")
    lines += [
        "", "| Condition | Method | Full process (s) | Accuracy | Aer circuit evaluations |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(f"| {row['condition']} | {LABELS[row['method']]} | "
                     f"{row['process_wall_s']:.4f} | {row['accuracy']:.2%} | "
                     f"{row['evaluations']:.0f} |")
    lines += [
        "", "## Matched comparisons", "",
        "Ratios use paired seed medians, not marginal median times. A descriptive win "
        "requires five complete paired seeds, three matched repeats each, every seed "
        "median faster and at least 5% median paired saving. Ranges are observed seeds, "
        "not confidence intervals. Missing/failed attempts never count as fast results.", "",
        "| Condition | Baseline | Spectral speedup | Seed range | Meets rule |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["comparisons"]:
        lines.append(f"| {row['condition']} | {LABELS[row['baseline']]} | "
                     f"{row['paired_speedup']:.3f}× | {row['minimum_seed_speedup']:.3f}–"
                     f"{row['maximum_seed_speedup']:.3f}× | {row['descriptive_win']} |")
    lines += ["", "## Faster of the Qiskit controls", "",
              "The faster Qiskit seed median is selected separately for each seed. "
              "This deliberately favors Qiskit and is not a deployed adaptive selector.", "",
              "| Condition | Paired speedup | Time saving | Meets rule |",
              "|---|---:|---:|---|"]
    for row in strongest:
        lines.append(f"| {row['condition']} | {row['paired_speedup']:.3f}× | "
                     f"{row['median_paired_saving']:.2%} | {row['descriptive_win']} |")
    lines += [
        "", "## Scope and limitations", "",
        "The independent angles really are updated separately. Their repeated gate "
        "occurrences are interleaved with fixed noncommuting rotations and ring "
        "entanglers. The response is a full multidimensional Fourier tensor, including "
        "mixed-sign frequencies. All response methods receive the same support, exact "
        "input identities, cache lifetime and native off-grid checks. Spatial input "
        "equivalence is not assumed.", "",
        "The additional Qiskit control batches occurrence-local parameter shifts and "
        "sums them per independent parameter, using cached diagonals and runtime binding and exact "
        "job reuse. It avoids full response reconstruction and supplies a stronger "
        "alternative when the response table grows expensive.", "",
        "Local-gate propagation removes the old dense 4^n block operators, but state "
        "storage still grows as 2^n. With p angles repeated r times each, the state "
        "coefficient count is (r+1)^p and response grid is (2r+1)^p. Chunking bounds "
        "temporary arrays; it does not remove exponential work. Library/import memory "
        "is additional to the algorithm's guarded array budget.", "",
        "The batched-grid ablation tests ordinary batching. The direct adjoint control "
        "computes the identical model and all angle derivatives without persistent "
        "response reconstruction. The task-specific classical classifier uses different "
        "features and is compared on both accuracy and time. Sensor views share "
        "underlying base patterns; data seeds do not vary the architecture.", "",
        "Classical coefficient propagation and native Aer calls are different costs. "
        "Fewer native calls do not establish physical CNOT, shot or hardware savings.", "",
        f"Worker process walls sum to {result['sum_process_wall_s']:.3f} s; "
        f"last parent measurement invocation: {result['status'].get('last_invocation_wall_s', 0):.3f} s.",
    ]
    (directory/"summary.md").write_text("\n".join(lines)+"\n")
    from matplotlib import pyplot as plt
    conditions = [c["id"] for c in result["conditions"]]
    baselines = ("qiskit_response", "classical_adjoint", "group_grid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    relevant = [r for r in result["comparisons"] if r["baseline"] in baselines]
    limit = max([1.2]+[r["maximum_seed_speedup"] for r in relevant])*1.06
    for ax, baseline in zip(axes, baselines):
        for i, cid in enumerate(conditions):
            selected = [r for r in relevant
                        if r["condition"] == cid and r["baseline"] == baseline]
            if not selected:
                continue
            row = selected[0]
            ax.plot([row["minimum_seed_speedup"], row["maximum_seed_speedup"]], [i, i],
                    color="#2563eb", linewidth=2)
            ax.scatter(row["paired_speedup"], i, color="#163c80", s=38)
        ax.axvline(1, linestyle="--", color="#64748b", linewidth=1)
        ax.set(title=LABELS[baseline], xlabel="GroupMemory spectral speedup", xlim=(0, limit))
        ax.set_yticks(range(len(conditions)), conditions)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Qubit and parameter scaling — full process, paired seed medians", fontsize=14)
    fig.text(.5, .01, "Observed seed ranges; above 1 favors GroupMemory. No confidence intervals.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/"paired_speedups.png", dpi=150)
    plt.close(fig)
    atomic(directory/"report_wall.json", {"seconds": time.perf_counter()-started})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    result = report(parser.parse_args().directory)
    print(json.dumps({k: result[k] for k in ("planned", "successful",
        "validation_planned", "validation_passed", "maximum_absolute_error")}))


if __name__ == "__main__":
    main()
