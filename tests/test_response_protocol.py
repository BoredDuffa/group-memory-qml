"""Small saved-record audits; these do not simulate or time quantum circuits."""

import hashlib
import json

import pytest

from group_memory_qml.response_report import audit
from group_memory_qml.response_suite import METHODS, build_plan


def saved_campaign(tmp_path, repeats=3):
    """Construct a complete, internally matched five-seed result journal."""
    directory = tmp_path / "campaign"
    attempts = directory / "attempts"
    attempts.mkdir(parents=True)
    seeds = list(range(941101, 941106))
    conditions = [{
        "id": "sensor-protocol",
        "experiment": "1A",
        "config": {"n_qubits": 10, "bases": 16, "views": 8, "updates": 5},
    }]
    plan = build_plan(conditions, seeds, repeats)
    source_hashes = {"engine.py": "a" * 64}
    config = {
        "conditions": conditions, "seeds": seeds, "repeats": repeats,
        "plan": plan, "source_hashes": source_hashes,
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "completed_attempts": len(plan),
        "planned_attempts": len(plan), "last_invocation_wall_s": 400000,
    }))
    primary = [1, 10, 100, 1000, 10000]
    baseline = [2, 11, 120, 1500, 20000]
    for index, spec in enumerate(plan):
        position = seeds.index(spec["seed"])
        method = spec["method"]
        duration = baseline[position] if method == "qiskit_symmetry" else primary[position]
        duration *= [0.8, 1.0, 1.2][spec["repeat"] % 3]
        analytic = method in ("classical", "classical_same_model")
        row = {
            "spec": spec, "status": "SUCCESS",
            "method": method, "experiment": spec["experiment"], "seed": spec["seed"],
            "source_hashes": source_hashes,
            "validation_vector": [0.25, 0.75, 0.53, -0.41, 0.03, -0.01],
            "metric_name": "heldout_accuracy", "metric_value": 0.75,
            "process_wall_s": duration, "application_s": duration / 2,
            "evaluations": 0 if analytic else 4,
            "total_cx": 0 if analytic else 40,
            "diagnostics": {
                "dataset_sha256": hashlib.sha256(str(spec["seed"]).encode()).hexdigest()
            },
        }
        (attempts / f"{index:05d}.json").write_text(json.dumps(row))
    return directory


def alter_record(directory, method, change, seed=941101, repeat=0):
    for path in sorted((directory / "attempts").glob("*.json")):
        row = json.loads(path.read_text())
        spec = row["spec"]
        if (spec["method"], spec["seed"], spec["repeat"]) == (method, seed, repeat):
            change(row)
            path.write_text(json.dumps(row))
            return path
    raise AssertionError("Test fixture did not contain the requested record")


def comparison(result, baseline="qiskit_symmetry"):
    return next(row for row in result["comparisons"] if row["baseline"] == baseline)


def test_paired_seed_ratios_are_not_ratios_of_marginal_medians(tmp_path):
    directory = saved_campaign(tmp_path)
    result, attempts, validation, seeds = audit(directory)
    assert result["successful"] == result["planned"] == 90
    assert result["validation_passed"] == result["validation_planned"] == 15
    assert result["maximum_absolute_error"] == 0
    assert all(row["trials"] == 3 for row in seeds)
    assert len(validation) == 15 and len(attempts) == 90
    paired = comparison(result)
    # Seed ratios are [2, 1.1, 1.2, 1.5, 2], whose median is 1.5.
    # Ratio of marginal time medians would incorrectly report 120/100 = 1.2.
    assert paired["paired_speedup"] == pytest.approx(1.5)
    assert paired["median_paired_saving"] == pytest.approx(1 - 1 / 1.5)
    assert paired["descriptive_win"] is True


def test_one_repeat_per_seed_cannot_meet_three_repeat_win_rule(tmp_path):
    result, _, _, _ = audit(saved_campaign(tmp_path, repeats=1))
    assert comparison(result)["matched_seeds"] == 5
    assert comparison(result)["descriptive_win"] is False


def test_missing_trial_stays_visible_and_removes_incomplete_seed(tmp_path):
    directory = saved_campaign(tmp_path)
    missing = alter_record(directory, "group_memory_response", lambda row: None)
    missing.unlink()
    result, attempts, validation, seeds = audit(directory)
    assert result["successful"] == 89
    assert result["validation_passed"] == 14
    assert sum(row["status"] == "UNATTEMPTED" for row in attempts) == 1
    assert sum(row["status"] == "INCOMPLETE" for row in validation) == 1
    assert {row["seed"] for row in seeds} == set(range(941102, 941106))
    assert comparison(result)["matched_seeds"] == 4
    assert comparison(result)["descriptive_win"] is False


def test_corrupted_training_trajectory_is_not_accepted_by_equal_accuracy(tmp_path):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "group_memory_response",
                 lambda row: row["validation_vector"].__setitem__(4, 0.03000001))
    with pytest.raises(AssertionError, match="correctness failure"):
        audit(directory)


def test_unequal_response_native_work_is_rejected(tmp_path):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "qiskit_response",
                 lambda row: row.__setitem__("total_cx", 41))
    with pytest.raises(AssertionError, match="unequal native work"):
        audit(directory)


def test_all_missing_data_hashes_cannot_count_as_matched_data(tmp_path):
    directory = saved_campaign(tmp_path)
    for method in METHODS:
        alter_record(directory, method,
                     lambda row: row["diagnostics"].pop("dataset_sha256"))
    with pytest.raises(AssertionError):
        audit(directory)


def test_different_classical_task_data_is_rejected(tmp_path):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "classical",
                 lambda row: row["diagnostics"].__setitem__("dataset_sha256", "b" * 64))
    with pytest.raises(AssertionError, match="data mismatch"):
        audit(directory)


def test_frozen_source_tampering_is_rejected(tmp_path):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "group_memory_response",
                 lambda row: row.__setitem__("source_hashes", {"engine.py": "b" * 64}))
    with pytest.raises(AssertionError, match="Frozen source mismatch"):
        audit(directory)


@pytest.mark.parametrize("field,value", [
    ("process_wall_s", 0.0), ("process_wall_s", -1.0),
    ("process_wall_s", float("nan")), ("process_wall_s", float("inf")),
    ("application_s", 0.0), ("application_s", -1.0),
    ("application_s", float("nan")), ("application_s", float("inf")),
])
def test_success_requires_finite_positive_measured_times(tmp_path, field, value):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "group_memory_response",
                 lambda row: row.__setitem__(field, value))
    with pytest.raises(AssertionError):
        audit(directory)


def test_application_clock_cannot_exceed_its_enclosing_process_clock(tmp_path):
    directory = saved_campaign(tmp_path)
    alter_record(directory, "group_memory_response",
                 lambda row: row.__setitem__("application_s", row["process_wall_s"] + 1))
    with pytest.raises(AssertionError):
        audit(directory)
