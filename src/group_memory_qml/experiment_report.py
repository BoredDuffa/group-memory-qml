"""Validate and summarize the eight-experiment fresh-process benchmark."""
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

NAMES = {
    "1A": "Sensor classification", "1B": "Equivariant sensor denoising",
    "2A": "Tied-weight shift gradients", "2B": "Gradient and Hessian training",
    "3A": "Joint-symmetry fidelity kernel", "3B": "Group-averaged kernel",
    "4A": "Shared quantum image patches", "4B": "Exact local quantum subcircuits",
}
METHODS = ("group_memory", "qiskit_symmetry", "qiskit_exact", "classical")


def write_csv(path, records):
    if not records:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def metric(row):
    return float(row.get("metric_value", row.get("value")))


def summarize(folder):
    started = time.perf_counter()
    folder = Path(folder)
    config = json.loads((folder/"config.json").read_text())
    status = json.loads((folder/"status.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((folder/"attempts").glob("*.json"))
            if not p.name.endswith(".worker.json")]
    planned = config["plan"]
    planned_by_key = {(s["condition"], s["seed"], s["repeat"], s["method"]):s for s in planned}
    if len(planned_by_key) != len(planned):
        raise AssertionError("Duplicate planned attempt")
    by_key = {}
    for row in rows:
        spec = row["spec"]
        key = (spec["condition"], spec["seed"], spec["repeat"], spec["method"])
        if key in by_key:
            raise AssertionError("Duplicate attempt")
        if key not in planned_by_key or spec != planned_by_key[key]:
            raise AssertionError("Unplanned or altered attempt spec")
        if "source_hashes" in row and row["source_hashes"] != config.get("source_hashes"):
            raise AssertionError("Worker source hash mismatch")
        by_key[key] = row
    validation = []
    flattened = []
    for spec in planned:
        key = (spec["condition"], spec["seed"], spec["repeat"], spec["method"])
        row = by_key.get(key, {"status": "UNATTEMPTED"})
        flat = dict(spec)
        flat["config"] = json.dumps(spec["config"], sort_keys=True)
        flat["status"] = row["status"]
        if row["status"] == "SUCCESS":
            flat.update(process_wall_s=row["process_wall_s"], application_s=row["application_s"],
                        import_and_setup_s=row["import_and_setup_s"],
                        metric_name=row["metric_name"], metric_value=metric(row),
                        higher_is_better=row.get("higher_is_better"),
                        evaluations=row.get("evaluations"), total_cx=row.get("total_cx"))
        flattened.append(flat)
    # A block only passes if all three quantum methods succeed and their entire
    # saved predictions/trajectories agree at the unchanged absolute tolerance.
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            for rep in range(config["repeats"]):
                block = [by_key.get((condition["id"], seed, rep, m)) for m in METHODS[:3]]
                result = dict(condition=condition["id"], seed=seed, repeat=rep)
                if any(r is None or r["status"] != "SUCCESS" for r in block):
                    result.update(status="MISSING", max_absolute_error=None)
                else:
                    vectors = [np.asarray(r["validation_vector"], dtype=float) for r in block]
                    if any(v.shape != vectors[0].shape or not np.isfinite(v).all() for v in vectors):
                        raise AssertionError(f"Invalid validation vector: {result}")
                    error = max(float(np.max(np.abs(v-vectors[0]))) for v in vectors[1:])
                    if error > 1e-10:
                        raise AssertionError(f"Quantum output mismatch {error}: {result}")
                    if block[0].get("evaluations") != block[1].get("evaluations"):
                        raise AssertionError(f"Unequal canonical workload in primary baseline: {result}")
                    if block[0].get("total_cx") != block[1].get("total_cx"):
                        raise AssertionError(f"Unequal CNOT workload in primary baseline: {result}")
                    if any(r["metric_name"] != block[0]["metric_name"]
                           or abs(metric(r)-metric(block[0])) > 1e-10
                           or r.get("higher_is_better") != block[0].get("higher_is_better")
                           for r in block[1:]):
                        raise AssertionError(f"Quantum task metric mismatch: {result}")
                    if condition["experiment"] == "4B":
                        classical = by_key.get((condition["id"],seed,rep,"classical"))
                        if classical is not None and classical["status"] == "SUCCESS":
                            oracle = np.asarray(classical["validation_vector"],dtype=float)
                            if oracle.shape != vectors[0].shape or not np.isfinite(oracle).all():
                                raise AssertionError(f"Invalid analytic local-model oracle: {result}")
                            oracle_error = float(np.max(np.abs(oracle-vectors[0])))
                            if oracle_error > 1e-10:
                                raise AssertionError(f"Analytic local-model mismatch {oracle_error}: {result}")
                            result["analytic_oracle_max_error"] = oracle_error
                    result.update(status="PASS", max_absolute_error=error)
                validation.append(result)
    valid_blocks = {(r["condition"], r["seed"], r["repeat"]) for r in validation if r["status"] == "PASS"}
    seed_rows, comparisons, summary = [], [], []
    for condition in config["conditions"]:
        cid = condition["id"]
        for seed in config["seeds"]:
            if not all((cid, seed, rep) in valid_blocks for rep in range(config["repeats"])):
                continue
            for m in METHODS:
                rr = [by_key.get((cid, seed, rep, m)) for rep in range(config["repeats"])]
                if any(r is None or r["status"] != "SUCCESS" for r in rr):
                    continue
                if max(metric(r) for r in rr)-min(metric(r) for r in rr) > 1e-10:
                    raise AssertionError("Nondeterministic task metric across timing repeats")
                seed_rows.append(dict(condition=cid, experiment=condition["experiment"],
                                      seed=seed, method=m,
                                      process_median_s=statistics.median(r["process_wall_s"] for r in rr),
                                      process_sum_s=sum(r["process_wall_s"] for r in rr),
                                      application_median_s=statistics.median(r["application_s"] for r in rr),
                                      metric_value=metric(rr[0]), metric_name=rr[0]["metric_name"],
                                      higher_is_better=rr[0].get("higher_is_better"),
                                      evaluations=rr[0].get("evaluations"), total_cx=rr[0].get("total_cx")))
        current = [r for r in seed_rows if r["condition"] == cid]
        lookup = {(r["seed"], r["method"]): r for r in current}
        for baseline in METHODS[1:]:
            pairs = [(lookup[(s, "group_memory")], lookup[(s, baseline)])
                     for s in config["seeds"] if (s, "group_memory") in lookup and (s, baseline) in lookup]
            if not pairs:
                continue
            ratios = [b["process_median_s"]/g["process_median_s"] for g,b in pairs]
            savings = [1-1/r for r in ratios]
            app = [b["application_median_s"]/g["application_median_s"] for g,b in pairs]
            # Classical uses a different model; report quality separately.
            comparisons.append(dict(
                condition=cid, experiment=condition["experiment"], baseline=baseline,
                complete_seeds=len(pairs), paired_process_speedup=statistics.median(ratios),
                paired_process_saving=statistics.median(savings),
                paired_application_speedup=statistics.median(app),
                seed_speedups=json.dumps(ratios),
                all_seed_medians_faster=all(r > 1 for r in ratios),
                descriptive_win=(len(pairs)>=5 and all(r>1 for r in ratios)
                                 and statistics.median(savings)>=.05),
                same_model=baseline!="classical" or condition["experiment"]=="4B"))
        for m in METHODS:
            rr = [r for r in current if r["method"] == m]
            if rr:
                summary.append(dict(condition=cid, experiment=condition["experiment"],
                                    method=m, seeds=len(rr),
                                    process_median_s=statistics.median(r["process_median_s"] for r in rr),
                                    application_median_s=statistics.median(r["application_median_s"] for r in rr),
                                    metric_mean=statistics.mean(r["metric_value"] for r in rr),
                                    metric_name=rr[0]["metric_name"],
                                    higher_is_better=rr[0].get("higher_is_better"),
                                    evaluations_median=statistics.median(r["evaluations"] for r in rr),
                                    total_cx_median=statistics.median(r["total_cx"] for r in rr)))
    write_csv(folder/"attempts.csv", flattened)
    write_csv(folder/"validation.csv", validation)
    write_csv(folder/"seed_blocks.csv", seed_rows)
    write_csv(folder/"comparisons.csv", comparisons)
    write_csv(folder/"summary.csv", summary)
    output = dict(status=status, attempts=len(rows), planned=len(planned),
                  successful=sum(r["status"]=="SUCCESS" for r in rows),
                  validation_passed=sum(r["status"]=="PASS" for r in validation),
                  validation_planned=len(validation),
                  maximum_absolute_error=max((r["max_absolute_error"] or 0 for r in validation), default=0),
                  process_wall_sum_s=sum(r.get("process_wall_s",0) for r in rows),
                  summary=summary, comparisons=comparisons)
    (folder/"analysis.json").write_text(json.dumps(output,indent=2,allow_nan=False))

    lines = ["# Eight QML experiments: measured results", "",
             "**Local ideal CPU simulation; no QPU or noisy-hardware speedup claim.**", "",
             f"Successful attempts: {output['successful']}/{len(planned)}. "
             f"Correctness-matched quantum blocks: {output['validation_passed']}/{len(validation)}. "
             f"Maximum saved-output disagreement: {output['maximum_absolute_error']:.3g}.", "",
             "Times below include a fresh interpreter, imports, data preparation, model construction, "
             "compilation, canonicalization, computation, fitting, predictions, JSON writing and shutdown. "
             "Installation is excluded. Independent correctness preflight, orchestration and reporting "
             "are separately recorded. Test views are related augmentations, not independent test samples.", "",
             "| Condition | Group time | Symmetry Qiskit time | Exact-cache Qiskit time | Classical time | Group / classical task metric |",
             "|---|---:|---:|---:|---:|---|"]
    for condition in config["conditions"]:
        lookup = {r["method"]:r for r in summary if r["condition"]==condition["id"]}
        def t(m): return f"{lookup[m]['process_median_s']:.3f} s" if m in lookup else "missing"
        g,c=lookup.get("group_memory"),lookup.get("classical")
        quality=f"{g['metric_name']}: {g['metric_mean']:.4g} / {c['metric_mean']:.4g}" if g and c else "missing"
        lines.append(f"| {condition['id']} | {t('group_memory')} | {t('qiskit_symmetry')} | {t('qiskit_exact')} | {t('classical')} | {quality} |")
    lines += ["", "## Paired comparisons", "",
              "Speedup >1 means GroupMemory is faster. Each seed contributes the median of all "
              "matching timing repeats. A descriptive win requires five complete seeds, every seed "
              "median faster and a median paired saving of at least 5%. This is not a confidence "
              "claim or a universal crossover. Classical quality can differ; its time ratio alone "
              "does not establish equal-task superiority. In 4B the analytic classical control "
              "computes the same features and trained readout, with explicit output checks.", "",
              "| Condition | Baseline | All-in speedup | Application speedup | Eligible descriptive win |",
              "|---|---|---:|---:|---|"]
    for r in comparisons:
        lines.append(f"| {r['condition']} | {r['baseline']} | {r['paired_process_speedup']:.3f}× | {r['paired_application_speedup']:.3f}× | {r['descriptive_win']} |")
    lines += ["", "## Boundaries", "",
              "- Equally informed Qiskit uses the same supplied symmetries, canonical quantum jobs, simulator and compilation settings; GroupMemory uses a dictionary and the baseline uses vectorized unique planning.",
              "- `qiskit_exact` is a secondary control with ordinary exact-input caches; it is not the primary comparison.",
              "- CNOT totals count gates in simulated preparations/evaluations actually performed. Avoided evaluations are not a reduction in each native circuit.",
              "- Kernel overlaps, reverse differentiation and the local closed-form classical control are classical computations. They are not measured QPU circuits.",
              "- Large conceptual width in 4B uses exact small backward cones and does not simulate the full global state.",
              "- Exact statevector equality does not prove identical noisy-device results. Hardware noise may break symmetry.",
              "- Missing/error/timeout attempts remain in attempts.csv and do not enter matched summaries.",
              "", f"Sum of worker process walls: {output['process_wall_sum_s']:.3f} s. "
              f"Campaign elapsed from its first start: {status.get('elapsed_since_first_start_s',0):.3f} s.",
              ""]
    markdown="\n".join(lines)
    (folder/"summary.md").write_text(markdown)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids=[c["id"] for c in config["conditions"]]
    fig, axes=plt.subplots(1,2,figsize=(14,max(5,len(ids)*.35)),constrained_layout=True)
    colors={"qiskit_symmetry":"#126e82","qiskit_exact":"#b56c21","classical":"#7b4587"}
    for offset,baseline in enumerate(colors):
        cc={r["condition"]:r for r in comparisons if r["baseline"]==baseline}
        ys=np.arange(len(ids))+(offset-1)*.23
        vals=[cc[c]["paired_process_speedup"] if c in cc else np.nan for c in ids]
        axes[0].scatter(vals,ys,label=baseline,color=colors[baseline],s=30)
    axes[0].set_yticks(range(len(ids)),ids);axes[0].invert_yaxis()
    axes[0].set_xscale("log");axes[0].axvline(1,color="black",linestyle="--")
    axes[0].set_xlabel("Baseline / GroupMemory total process time")
    axes[0].set_title("Right of 1: GroupMemory faster");axes[0].legend(fontsize=8)
    for offset,method in enumerate(METHODS):
        rr={r["condition"]:r for r in summary if r["method"]==method}
        vals=[rr[c]["process_median_s"] if c in rr else np.nan for c in ids]
        axes[1].scatter(vals,np.arange(len(ids))+(offset-1.5)*.18,label=method,s=24)
    axes[1].set_yticks(range(len(ids)),ids);axes[1].invert_yaxis()
    axes[1].set_xscale("log");axes[1].set_xlabel("Seconds, fresh process")
    axes[1].set_title("All classical overhead included");axes[1].legend(fontsize=8)
    fig.savefig(folder/"comparison.png",dpi=150);plt.close(fig)
    # Render the same complete table, avoiding a dependency on a Markdown parser.
    rendered=[]
    table=False
    for line in lines:
        if line.startswith("|---"):continue
        if line.startswith("|"):
            if not table:rendered.append("<table>");table=True
            rendered.append("<tr>"+"".join("<td>"+html.escape(cell.strip())+"</td>" for cell in line.strip("|").split("|"))+"</tr>")
            continue
        if table:rendered.append("</table>");table=False
        if line.startswith("# "):rendered.append("<h1>"+html.escape(line[2:])+"</h1>")
        elif line.startswith("## "):rendered.append("<h2>"+html.escape(line[3:])+"</h2>")
        elif line:rendered.append("<p>"+html.escape(line)+"</p>")
    if table:rendered.append("</table>")
    (folder/"results.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Eight QML experiments</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#123}"
        "table{border-collapse:collapse;width:100%;font-size:14px}td{padding:8px;border-bottom:1px solid #ccd}"
        "tr:first-child{font-weight:bold;background:#eef5f7}img{width:100%}p{line-height:1.5}</style>"
        +"".join(rendered[:2])+"<img src='comparison.png' alt='Matched runtime comparisons'>"
        +"".join(rendered[2:]))
    (folder/"report_wall.json").write_text(json.dumps({"report_wall_s":time.perf_counter()-started,
                                                    "includes":"validation, tables, plots and HTML; excludes this final receipt write"},indent=2))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    args=p.parse_args()
    r=summarize(args.folder)
    print(json.dumps({k:v for k,v in r.items() if k not in ("summary","comparisons")},indent=2))


if __name__=="__main__":
    main()
