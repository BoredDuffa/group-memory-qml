"""Synthetic harness checks; these tests run no learner or performance benchmark."""
from collections import Counter
import csv
import json

import pytest

from group_memory_qml.benchmark_suite import METHODS, build_plan
from group_memory_qml.experiment_report import summarize


def _fixture(folder, seeds=(1,), repeats=2):
    conditions = [{"id": "synthetic-3A", "experiment": "3A", "config": {"n_qubits": 6}}]
    plan = build_plan(conditions, list(seeds), repeats)
    config = {"conditions": conditions, "seeds": list(seeds), "repeats": repeats, "plan": plan,
              "source_hashes": {"synthetic_fixture.py": "0" * 64}}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "completed_attempts": len(plan),
        "planned_attempts": len(plan), "elapsed_since_first_start_s": 100,
    }))
    (folder / "attempts").mkdir()
    paths = {}
    for i, spec in enumerate(plan):
        method = spec["method"]
        row = {
            "synthetic_fixture": True, "spec": spec, "status": "SUCCESS",
            "source_hashes": config["source_hashes"],
            "process_wall_s": {"group_memory": 2., "qiskit_symmetry": 3.,
                               "qiskit_exact": 4., "classical": .1}[method],
            "application_s": {"group_memory": 1., "qiskit_symmetry": 1.5,
                              "qiskit_exact": 2., "classical": .04}[method],
            "import_and_setup_s": .03,
            "metric_name": "heldout_accuracy", "higher_is_better": True,
            "value": .9 if method == "classical" else .75,
            "evaluations": 0 if method == "classical" else 32 if method == "qiskit_exact" else 8,
            "total_cx": 0 if method == "classical" else 64,
            "validation_vector": [.1, .2, 1.] if method != "classical" else [.4, .6, 0.],
        }
        path = folder / "attempts" / f"{i:05d}.json"
        path.write_text(json.dumps(row))
        paths[(spec["seed"], spec["repeat"], method)] = path
    return config, paths


def _change(path, update):
    row = json.loads(path.read_text())
    row.update(update)
    path.write_text(json.dumps(row))


@pytest.fixture
def no_plot_write(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)


def test_plan_is_deterministic_and_matches_all_methods_per_block():
    conditions = [
        {"id": "a", "experiment": "1A", "config": {"views": 4}},
        {"id": "b", "experiment": "3B", "config": {"n_qubits": 8}},
    ]
    plan = build_plan(conditions, [930101, 930102], 3)
    assert plan == build_plan(conditions, [930101, 930102], 3)
    assert len(plan) == len(conditions) * 2 * 3 * len(METHODS)
    keys = [(p["condition"], p["seed"], p["repeat"], p["method"]) for p in plan]
    assert len(set(keys)) == len(keys)
    blocks = Counter((p["condition"], p["seed"], p["repeat"]) for p in plan)
    assert set(blocks.values()) == {len(METHODS)}
    for block in blocks:
        rows = [p for p in plan if (p["condition"], p["seed"], p["repeat"]) == block]
        assert {p["method"] for p in rows} == set(METHODS)
        assert all(p["config"] == rows[0]["config"] for p in rows)


def test_synthetic_complete_blocks_and_paired_statistics(tmp_path, no_plot_write):
    _fixture(tmp_path, seeds=range(1, 6), repeats=3)
    result = summarize(tmp_path)
    assert result["successful"] == result["planned"] == 60
    assert result["validation_passed"] == 15
    assert result["maximum_absolute_error"] == 0
    comparison = next(r for r in result["comparisons"] if r["baseline"] == "qiskit_symmetry")
    assert comparison["complete_seeds"] == 5
    assert comparison["paired_process_speedup"] == pytest.approx(1.5)
    assert comparison["paired_process_saving"] == pytest.approx(1/3)
    assert comparison["descriptive_win"] is True
    classical = next(r for r in result["comparisons"] if r["baseline"] == "classical")
    assert classical["same_model"] is False
    assert classical["descriptive_win"] is False
    with (tmp_path / "seed_blocks.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    group = [r for r in rows if r["method"] == "group_memory"]
    assert all(float(r["process_sum_s"]) == 6. for r in group)


def test_report_rejects_changed_quantum_output(tmp_path):
    _, paths = _fixture(tmp_path)
    _change(paths[(1, 0, "qiskit_symmetry")], {"validation_vector": [.100001, .2, 1.]})
    with pytest.raises(AssertionError, match="Quantum output mismatch"):
        summarize(tmp_path)


def test_report_rejects_nonfinite_output(tmp_path):
    _, paths = _fixture(tmp_path)
    _change(paths[(1, 0, "group_memory")], {"validation_vector": [float("nan"), .2, 1.]})
    with pytest.raises(AssertionError, match="Invalid validation vector"):
        summarize(tmp_path)


def test_report_rejects_different_primary_canonical_workload(tmp_path):
    _, paths = _fixture(tmp_path)
    _change(paths[(1, 0, "qiskit_symmetry")], {"evaluations": 9})
    with pytest.raises(AssertionError, match="Unequal canonical workload"):
        summarize(tmp_path)


@pytest.mark.parametrize("update,match", [
    ({"total_cx": 65}, "Unequal CNOT workload"),
    ({"value": .5}, "Quantum task metric mismatch"),
    ({"metric_name": "unrelated_score"}, "Quantum task metric mismatch"),
    ({"higher_is_better": False}, "Quantum task metric mismatch"),
    ({"source_hashes": {"synthetic_fixture.py": "1" * 64}}, "Worker source hash mismatch"),
])
def test_report_rejects_changed_metric_count_or_source(tmp_path, update, match):
    _, paths = _fixture(tmp_path)
    _change(paths[(1, 0, "qiskit_symmetry")], update)
    with pytest.raises(AssertionError, match=match):
        summarize(tmp_path)


def test_report_rejects_altered_full_spec(tmp_path):
    _, paths = _fixture(tmp_path)
    path = paths[(1, 0, "group_memory")]
    row = json.loads(path.read_text())
    row["spec"]["config"]["n_qubits"] = 8
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError, match="Unplanned or altered attempt spec"):
        summarize(tmp_path)


def test_report_rejects_unplanned_extra_row(tmp_path):
    _, paths = _fixture(tmp_path)
    row = json.loads(paths[(1, 0, "group_memory")].read_text())
    row["spec"]["seed"] = 999
    (tmp_path / "attempts" / "extra.json").write_text(json.dumps(row))
    with pytest.raises(AssertionError, match="Unplanned or altered attempt spec"):
        summarize(tmp_path)


def test_report_rejects_duplicate_plan(tmp_path):
    config, _ = _fixture(tmp_path)
    config["plan"].append(config["plan"][0])
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(AssertionError, match="Duplicate planned attempt"):
        summarize(tmp_path)


def test_missing_or_failed_quantum_trial_excludes_whole_seed(tmp_path, no_plot_write):
    config, paths = _fixture(tmp_path, seeds=(1, 2, 3), repeats=2)
    paths[(2, 1, "qiskit_exact")].unlink()
    _change(paths[(3, 1, "qiskit_symmetry")], {"status": "TIMEOUT"})
    # An apparently attractive unmatched time must never enter the comparisons.
    _change(paths[(2, 0, "group_memory")], {"process_wall_s": .00001})
    result = summarize(tmp_path)
    assert result["attempts"] == len(config["plan"]) - 1
    assert result["validation_passed"] == 4
    assert all(row["seeds"] == 1 for row in result["summary"])
    assert all(row["complete_seeds"] == 1 for row in result["comparisons"])
    assert not any(row["descriptive_win"] for row in result["comparisons"])
    with (tmp_path / "attempts.csv").open() as handle:
        statuses = Counter(row["status"] for row in csv.DictReader(handle))
    assert statuses["UNATTEMPTED"] == 1
    assert statuses["TIMEOUT"] == 1
    with (tmp_path / "seed_blocks.csv").open() as handle:
        assert {int(row["seed"]) for row in csv.DictReader(handle)} == {1}


def test_missing_classical_trial_does_not_block_quantum_comparison(tmp_path, no_plot_write):
    _, paths = _fixture(tmp_path)
    paths[(1, 1, "classical")].unlink()
    result = summarize(tmp_path)
    assert result["validation_passed"] == 2
    assert {r["baseline"] for r in result["comparisons"]} == {"qiskit_symmetry", "qiskit_exact"}
    assert not any(row["method"] == "classical" for row in result["summary"])


def test_duplicate_attempt_is_rejected(tmp_path):
    _, paths = _fixture(tmp_path)
    original = paths[(1, 0, "group_memory")].read_text()
    (tmp_path / "attempts" / "duplicate.json").write_text(original)
    with pytest.raises(AssertionError, match="Duplicate attempt"):
        summarize(tmp_path)
