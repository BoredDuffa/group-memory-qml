"""Improved sensor benchmark with fresh processes, fixed data identities and a resumable plan."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

METHODS = ("group_spectral", "qiskit_response", "classical_adjoint",
           "frozen_quantum", "classical_logistic", "classical_rbf", "classical_mlp")


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def worker(spec, output, data_dir):
    started = time.perf_counter()
    hashes = source_hashes()
    module = importlib.import_module("group_memory_qml.improved_sensor_training")
    from threadpoolctl import threadpool_limits
    import_s = time.perf_counter()-started
    with threadpool_limits(limits=1):
        tick = time.perf_counter()
        result = module.run(spec["seed"], spec["method"], spec.get("config", {}), data_dir)
        application_s = time.perf_counter()-tick
    try:
        import resource
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
            1 if sys.platform == "darwin" else 1024)
    except ImportError:
        peak_rss_bytes = None  # Windows: resource is unavailable.
    result.update(spec=spec, import_and_setup_s=import_s, application_s=application_s,
                  worker_before_write_s=time.perf_counter()-started,
                  python=platform.python_version(), source_hashes=hashes,
                  peak_rss_bytes=peak_rss_bytes)
    # The parent timer includes this serialization and the interpreter shutdown.
    atomic(output, result)


def build_plan(conditions, seeds, repeats):
    if not conditions or len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Conditions must have unique nonempty identifiers")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    rng = random.Random(20261025)
    blocks = [(c, seed, rep) for c in conditions for seed in seeds for rep in range(repeats)]
    rng.shuffle(blocks)
    plan = []
    for c, seed, rep in blocks:
        methods = list(c.get("methods", METHODS))
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
            raise ValueError("Each condition requires distinct supported methods")
        rng.shuffle(methods)
        for method in methods:
            plan.append({"condition": c["id"], "experiment": c["experiment"],
                         "config": c.get("config", {}), "seed": seed,
                         "repeat": rep, "method": method})
    return plan


def campaign(output, conditions, seeds, repeats, budget_s, limit=None,
             hard_deadline_unix=None, data_dir=None):
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Budget must be finite and positive")
    if hard_deadline_unix is not None:
        if not isinstance(hard_deadline_unix, (int, float)) or not math.isfinite(hard_deadline_unix) or hard_deadline_unix <= 0:
            raise ValueError("hard_deadline_unix must be a finite positive timestamp")
    data_dir = Path(data_dir).absolute()
    data_files = sorted(data_dir.glob("*.npz"))
    if not data_files:
        raise FileNotFoundError("Prepare the real dataset before benchmarking")
    data_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(conditions, seeds, repeats)
    config = {"conditions": conditions, "seeds": seeds, "repeats": repeats,
              "budget_s": budget_s, "plan": plan, "source_hashes": source_hashes(),
              "data_sha256": data_hashes, "method_timeout_s": 240}
    if hard_deadline_unix is not None:
        config["hard_deadline_unix"] = hard_deadline_unix
    config_path = output/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Resume requires unchanged plan, budget and source hashes")
        authorization = json.loads((output/"authorization.json").read_text())
    else:
        if any(output.iterdir()):
            raise RuntimeError("Refusing an unrecognized nonempty directory")
        now = time.time()
        authorization = {"start_unix": now, "deadline_unix": min(now+budget_s, hard_deadline_unix) if hard_deadline_unix is not None else now+budget_s,
                         "scope": "Local CPU-only benchmark; no remote VM or QPU"}
        atomic(config_path, config)
        atomic(output/"authorization.json", authorization)
        atomic(output/"environment.json", {
            "python": sys.version, "executable_name": Path(sys.executable).name,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "qiskit", "qiskit-aer", "scikit-learn", "threadpoolctl"]},
            "threads_per_worker": 1,
            "clock": "Parent perf_counter around subprocess.run; includes interpreter startup, imports, application, JSON write and shutdown",
            "exclusions": "Package installation; correctness preflight and reporting are separately timed."})
    deadline = authorization["deadline_unix"]
    manifest_path = output/"checkpoints"/"manifest.json"
    cursor = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cursor = manifest["completed_attempts"]
        for i in range(cursor):
            p = output/"attempts"/f"{i:05d}.json"
            if not p.exists() or json.loads(p.read_text())["spec"] != plan[i]:
                raise RuntimeError("Checkpoint journal is inconsistent")
        latest = output/"checkpoints"/manifest["latest"]
        if hashlib.sha256(latest.read_bytes()).hexdigest() != manifest["latest_sha256"]:
            raise RuntimeError("Checkpoint hash mismatch")
        checkpoint = json.loads(latest.read_text())
        if checkpoint["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
            raise RuntimeError("Checkpoint config mismatch")
        if checkpoint["deadline_unix"] != deadline or manifest["deadline_unix"] != deadline:
            raise RuntimeError("Checkpoint deadline mismatch")
        for name, expected in checkpoint["journal_sha256"].items():
            if hashlib.sha256((output/"attempts"/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Journal content hash mismatch")
        last_result = output/"attempts"/f"{cursor-1:05d}.json"
        if hashlib.sha256(last_result.read_bytes()).hexdigest() != checkpoint["last_result_sha256"]:
            raise RuntimeError("Last result hash mismatch")
    start_cursor = cursor
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0",
                       RAYON_NUM_THREADS="1")
    run_started = time.perf_counter()
    for i in range(cursor, len(plan)):
        remaining = deadline-time.time()
        if remaining <= 1 or (limit is not None and i-start_cursor >= limit):
            break
        spec = plan[i]
        attempt_dir = output/"attempts"
        attempt_dir.mkdir(exist_ok=True)
        raw = attempt_dir/f"{i:05d}.worker.json"
        attempt_start_unix = time.time()
        started = time.perf_counter()
        cmd = [sys.executable, "-m", "group_memory_qml.improved_sensor_suite", "worker",
               "--spec", json.dumps(spec), "--output", str(raw), "--data-dir", str(data_dir)]
        try:
            completed = subprocess.run(cmd, env=environment, capture_output=True,
                                       text=True, timeout=min(240, remaining))
            wall = time.perf_counter()-started
            if completed.returncode or not raw.exists():
                row = {"spec": spec, "status": "ERROR", "process_wall_s": wall,
                       "stderr": completed.stderr[-5000:], "stdout": completed.stdout[-2000:]}
            else:
                row = json.loads(raw.read_text())
                if row["source_hashes"] != config["source_hashes"]:
                    raise RuntimeError("Worker source changed during the frozen campaign")
                row.update(status="SUCCESS", process_wall_s=wall,
                           stderr=completed.stderr[-1500:])
                raw.unlink()
        except subprocess.TimeoutExpired as error:
            row = {"spec": spec, "status": "TIMEOUT",
                   "process_wall_s": time.perf_counter()-started,
                   "timeout_s": min(240, remaining)}
        row.update(attempt_start_unix=attempt_start_unix, attempt_end_unix=time.time())
        atomic(attempt_dir/f"{i:05d}.json", row)
        checkpoint = {"completed_attempts": i+1, "next_spec": plan[i+1] if i+1 < len(plan) else None,
                      "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      "deadline_unix": deadline, "journal_last_index": i,
                      "last_result_sha256": hashlib.sha256((attempt_dir/f"{i:05d}.json").read_bytes()).hexdigest(),
                      "journal_sha256": {f"{j:05d}.json": hashlib.sha256((attempt_dir/f"{j:05d}.json").read_bytes()).hexdigest() for j in range(i+1)}}
        cpname = f"checkpoint-{(i+1)%2}.json"
        cppath = output/"checkpoints"/cpname
        atomic(cppath, checkpoint)
        atomic(output/"checkpoints"/"most_complete.json", checkpoint)
        atomic(manifest_path, {"completed_attempts": i+1, "latest": cpname,
                              "previous": f"checkpoint-{i%2}.json" if i else None,
                              "latest_sha256": hashlib.sha256(cppath.read_bytes()).hexdigest(),
                              "deadline_unix": deadline})
        cursor = i+1
        atomic(output/"status.json", {"status": "RUNNING", "completed_attempts": cursor,
                                      "planned_attempts": len(plan), "latest": spec,
                                      "deadline_unix": deadline})
        print(json.dumps({"completed": cursor, "planned": len(plan), "status": row["status"],
                          "experiment": spec["experiment"], "condition": spec["condition"],
                          "method": spec["method"], "seconds": round(row["process_wall_s"], 3)}),
              flush=True)
    state = "COMPLETE" if cursor == len(plan) else "BUDGET_EXHAUSTED" if time.time() >= deadline-1 else "CHECKPOINTED"
    atomic(output/"status.json", {"status": state, "completed_attempts": cursor,
                                  "planned_attempts": len(plan), "deadline_unix": deadline,
                                  "last_invocation_wall_s": time.perf_counter()-run_started,
                                  "elapsed_since_first_start_s": time.time()-authorization["start_unix"],
                                  "unattempted": len(plan)-cursor})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    w.add_argument("--output", required=True)
    w.add_argument("--data-dir", required=True)
    c = sub.add_parser("run")
    c.add_argument("--conditions", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--seeds", nargs="+", type=int, default=[956101,956102,956103,956104,956105])
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument("--budget-seconds", type=float, default=1800)
    c.add_argument("--hard-deadline-unix", type=float, default=None,
                   help="Immutable enclosing deadline, including earlier revisions and preflight")
    c.add_argument("--limit", type=int, default=None, help="Stop at a checkpoint for resume smoke tests")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(json.loads(args.spec), args.output, args.data_dir)
    else:
        campaign(args.output, json.loads(Path(args.conditions).read_text()),
                 args.seeds, args.repeats, args.budget_seconds, args.limit,
                 args.hard_deadline_unix, args.data_dir)


if __name__ == "__main__":
    main()
