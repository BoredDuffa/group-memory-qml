"""Audit and resume protocol checks without circuit execution or timing claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from group_memory_qml import deep_suite
from group_memory_qml.deep_report import audit
from group_memory_qml.deep_training import _configuration, run


def saved_campaign(tmp_path, *, repeats=3, seed_count=5):
    directory = tmp_path / "saved"
    attempts = directory / "attempts"
    attempts.mkdir(parents=True)
    seeds = list(range(942101, 942101 + seed_count))
    conditions = [{
        "id": "sensor-L16", "experiment": "deep_sensor",
        "config": {"n_qubits": 6, "layers": 16, "updates": 12},
    }]
    plan = deep_suite.build_plan(conditions, seeds, repeats)
    frozen = {"engine.py": "a" * 64}
    (directory / "config.json").write_text(json.dumps({
        "conditions": conditions, "seeds": seeds, "repeats": repeats,
        "plan": plan, "source_hashes": frozen,
    }))
    (directory / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "completed_attempts": len(plan),
        "planned_attempts": len(plan),
    }))
    primary = [1., 10., 100., 1000., 10000.]
    baseline = [2., 11., 120., 1500., 20000.]
    for index, spec in enumerate(plan):
        position = seeds.index(spec["seed"])
        duration = (baseline if spec["method"] == "qiskit_response" else primary)[position]
        duration *= [.8, 1., 1.2][spec["repeat"] % 3]
        row = {
            "spec": spec, "status": "SUCCESS", "method": spec["method"],
            "experiment": spec["experiment"], "seed": spec["seed"],
            "source_hashes": frozen,
            "validation_vector": [.25, .75, .53, -.41, .03, -.01],
            "metric_name": "heldout_accuracy", "metric_value": .75,
            "process_wall_s": duration, "application_s": duration / 2,
            "evaluations": 0 if spec["method"] == "classical" else 4,
            "total_cx": 0 if spec["method"] == "classical" else 40,
            "diagnostics": {
                "dataset_sha256": hashlib.sha256(str(spec["seed"]).encode()).hexdigest(),
                "response": {
                    "source_sha256": "c" * 64, "architecture_sha256": "d" * 64,
                    "native_sha256": "e" * 64,
                    "fixed_block_sha256": ["f" * 64, "a" * 64],
                },
            },
        }
        (attempts / f"{index:05d}.json").write_text(json.dumps(row))
    return directory


def alter(directory, method, change, seed=942101, repeat=0):
    for path in (directory / "attempts").glob("*.json"):
        row = json.loads(path.read_text())
        spec = row["spec"]
        if (spec["method"], spec["seed"], spec["repeat"]) == (method, seed, repeat):
            change(row)
            path.write_text(json.dumps(row))
            return path
    raise AssertionError("Missing fixture record")


def primary_comparison(result):
    return next(r for r in result["comparisons"] if r["baseline"] == "qiskit_response")


def test_ratios_use_matched_seed_medians_not_marginal_ratios(tmp_path):
    result, attempts, validation, seeds = audit(saved_campaign(tmp_path))
    assert result["planned"] == result["successful"] == 90
    assert result["validation_planned"] == result["validation_passed"] == 15
    assert len(attempts) == 90 and len(validation) == 15
    assert all(r["trials"] == 3 for r in seeds)
    paired = primary_comparison(result)
    # The paired median is 1.5, while the ratio of marginal medians is 1.2.
    assert paired["paired_speedup"] == pytest.approx(1.5)
    assert paired["median_paired_saving"] == pytest.approx(1 - 1 / 1.5)
    assert paired["descriptive_win"] is True


@pytest.mark.parametrize("repeats,seed_count", [(1, 5), (2, 5), (3, 4)])
def test_win_requires_five_complete_seeds_with_three_trials(tmp_path, repeats, seed_count):
    result, *_ = audit(saved_campaign(tmp_path, repeats=repeats, seed_count=seed_count))
    assert primary_comparison(result)["descriptive_win"] is False


def test_single_slower_seed_prevents_descriptive_win(tmp_path):
    directory = saved_campaign(tmp_path)
    for repeat, duration in enumerate([.72, .9, 1.08]):
        alter(directory, "qiskit_response",
              lambda row, d=duration: row.update(process_wall_s=d, application_s=d / 2),
              repeat=repeat)
    result, *_ = audit(directory)
    paired = primary_comparison(result)
    assert paired["paired_speedup"] > 1.
    assert paired["minimum_seed_speedup"] < 1.
    assert not paired["descriptive_win"]


def test_less_than_five_percent_median_saving_prevents_win(tmp_path):
    directory = saved_campaign(tmp_path)
    for path in (directory / "attempts").glob("*.json"):
        row = json.loads(path.read_text())
        spec = row["spec"]
        if spec["method"] == "qiskit_response":
            duration = [1., 10., 100., 1000., 10000.][spec["seed"] - 942101]
            duration *= [.8, 1., 1.2][spec["repeat"]] * 1.04
            row.update(process_wall_s=duration, application_s=duration / 2)
            path.write_text(json.dumps(row))
    result, *_ = audit(directory)
    paired = primary_comparison(result)
    assert paired["minimum_seed_speedup"] > 1.
    assert paired["median_paired_saving"] < .05
    assert not paired["descriptive_win"]


def test_missing_attempt_remains_visible_and_excludes_the_seed(tmp_path):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral", lambda row: None).unlink()
    result, attempts, validation, seeds = audit(directory)
    assert result["successful"] == 89 and result["validation_passed"] == 14
    assert sum(r["status"] == "UNATTEMPTED" for r in attempts) == 1
    assert sum(r["status"] == "INCOMPLETE" for r in validation) == 1
    assert {r["seed"] for r in seeds} == set(range(942102, 942106))
    assert primary_comparison(result)["matched_seeds"] == 4
    assert not primary_comparison(result)["descriptive_win"]


@pytest.mark.parametrize("failure", ["TIMEOUT", "ERROR"])
def test_failed_attempt_is_preserved_and_never_counted_as_speedup(tmp_path, failure):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral", lambda row: row.update(status=failure))
    result, attempts, *_ = audit(directory)
    assert sum(r["status"] == failure for r in attempts) == 1
    assert result["successful"] == 89
    assert not primary_comparison(result)["descriptive_win"]


@pytest.mark.parametrize("field,value,match", [
    ("validation_vector", [.25, .75, .53, -.41, .03000001, -.01], "correctness failure"),
    ("source_hashes", {"engine.py": "b" * 64}, "Frozen source mismatch"),
    ("seed", 999999, "Seed mismatch"),
    ("method", "classical", "Result identity mismatch"),
])
def test_corrupt_record_rejected(tmp_path, field, value, match):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral", lambda row: row.update({field: value}))
    with pytest.raises(AssertionError, match=match):
        audit(directory)


@pytest.mark.parametrize("method", ["group_spectral", "qiskit_blocks", "classical"])
def test_different_data_is_rejected_even_for_task_classical(tmp_path, method):
    directory = saved_campaign(tmp_path)
    alter(directory, method,
          lambda row: row["diagnostics"].update(dataset_sha256="d" * 64))
    with pytest.raises(AssertionError):
        audit(directory)


def test_shared_missing_data_hash_is_not_equivalence(tmp_path):
    directory = saved_campaign(tmp_path)
    for method in deep_suite.METHODS:
        alter(directory, method, lambda row: row["diagnostics"].pop("dataset_sha256"))
    with pytest.raises(AssertionError):
        audit(directory)


def test_mismatched_source_circuit_signature_rejected(tmp_path):
    directory = saved_campaign(tmp_path)
    alter(directory, "qiskit_blocks",
          lambda row: row["diagnostics"]["response"].update(source_sha256="b" * 64))
    with pytest.raises(AssertionError, match="Different source circuits"):
        audit(directory)


@pytest.mark.parametrize("method,field,value", [
    ("classical_direct", "architecture_sha256", "b" * 64),
    ("group_spectral", "native_sha256", "b" * 64),
    ("qiskit_blocks", "fixed_block_sha256", ["b" * 64, "a" * 64]),
])
def test_unequal_architecture_native_or_dense_block_hash_rejected(tmp_path, method, field, value):
    directory = saved_campaign(tmp_path)
    alter(directory, method, lambda row: row["diagnostics"]["response"].update({field: value}))
    with pytest.raises(AssertionError):
        audit(directory)


@pytest.mark.parametrize("field,methods", [
    ("architecture_sha256", deep_suite.METHODS[:-1]),
    ("source_sha256", deep_suite.METHODS[:4]),
    ("native_sha256", deep_suite.METHODS[:4]),
    ("fixed_block_sha256", ("group_spectral", "group_grid", "qiskit_blocks")),
])
def test_shared_missing_signatures_cannot_be_accepted(tmp_path, field, methods):
    directory = saved_campaign(tmp_path)
    for method in methods:
        alter(directory, method, lambda row: row["diagnostics"]["response"].pop(field))
    with pytest.raises(AssertionError):
        audit(directory)


@pytest.mark.parametrize("field,value", [
    ("process_wall_s", 0.), ("process_wall_s", -1.),
    ("process_wall_s", float("nan")), ("process_wall_s", float("inf")),
    ("application_s", 0.), ("application_s", -1.),
    ("application_s", float("nan")), ("application_s", float("inf")),
])
def test_measured_times_must_be_positive_finite(tmp_path, field, value):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral", lambda row: row.update({field: value}))
    with pytest.raises(AssertionError):
        audit(directory)


def test_application_clock_must_fit_inside_process_clock(tmp_path):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral",
          lambda row: row.update(application_s=row["process_wall_s"] + 1.))
    with pytest.raises(AssertionError):
        audit(directory)


def test_native_call_counts_may_differ_between_cpu_algorithms(tmp_path):
    directory = saved_campaign(tmp_path)
    alter(directory, "group_spectral", lambda row: row.update(evaluations=2, total_cx=20))
    result, *_ = audit(directory)
    assert result["validation_passed"] == 15


@pytest.mark.parametrize("config", [
    {"unknown": 1}, {"n_qubits": 3}, {"n_qubits": 10}, {"n_qubits": 5},
    {"layers": 0}, {"layers": 33}, {"layers": True}, {"layers": 2.5},
    {"updates": 0}, {"updates": 41}, {"bases": 5},
    {"n_train": 5}, {"n_test": 5}, {"views": 13, "n_qubits": 6},
])
def test_configuration_rejects_unsupported_values(config):
    with pytest.raises(ValueError):
        _configuration("deep_sensor", config)


def test_configuration_valid_and_no_mutation():
    supplied = {"layers": 16, "updates": 12}
    assert _configuration("deep_shared", supplied)["n_qubits"] == 8
    assert supplied == {"layers": 16, "updates": 12}
    with pytest.raises(ValueError):
        _configuration("unsupported", {})


@pytest.mark.parametrize("method,seed", [("wrong", 1), ("group_spectral", -1),
                                        ("group_spectral", True)])
def test_invalid_worker_identity_rejected_before_loading_models(method, seed):
    with pytest.raises(ValueError):
        run("deep_sensor", seed, method)


def mock_workers(monkeypatch):
    frozen = {"engine.py": "f" * 64}
    calls = []
    monkeypatch.setattr(deep_suite, "source_hashes", lambda: frozen.copy())
    monkeypatch.setattr(deep_suite.importlib.metadata, "version", lambda package: "test")
    monkeypatch.setattr(deep_suite.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(deep_suite.platform, "processor", lambda: "test-processor")

    def subprocess_run(command, **kwargs):
        spec = json.loads(command[command.index("--spec") + 1])
        output = Path(command[command.index("--output") + 1])
        calls.append(spec)
        deep_suite.atomic(output, {"spec": spec, "source_hashes": frozen})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deep_suite.subprocess, "run", subprocess_run)
    return calls


def test_checkpoint_resume_keeps_plan_deadline_and_does_not_repeat_workers(tmp_path, monkeypatch):
    calls = mock_workers(monkeypatch)
    directory = tmp_path / "resume"
    conditions = [{"id": "smoke", "experiment": "deep_sensor", "config": {"layers": 2}}]
    args = directory, conditions, [942001], 1, 60
    assert deep_suite.campaign(*args, limit=1) == "CHECKPOINTED"
    authorization = (directory / "authorization.json").read_bytes()
    first_record = (directory / "attempts" / "00000.json").read_bytes()
    assert len(calls) == 1
    assert deep_suite.campaign(*args) == "COMPLETE"
    assert len(calls) == len(deep_suite.METHODS)
    assert len({json.dumps(c, sort_keys=True) for c in calls}) == len(calls)
    assert (directory / "authorization.json").read_bytes() == authorization
    assert (directory / "attempts" / "00000.json").read_bytes() == first_record
    manifest = json.loads((directory / "checkpoints" / "manifest.json").read_text())
    assert manifest["completed_attempts"] == len(calls)
    assert (directory / "checkpoints" / manifest["previous"]).exists()
    with pytest.raises(RuntimeError, match="unchanged"):
        deep_suite.campaign(directory, conditions, [942001], 1, 61)


def test_checkpoint_tamper_rejected_before_resume(tmp_path, monkeypatch):
    calls = mock_workers(monkeypatch)
    directory = tmp_path / "resume"
    args = directory, [{"id": "smoke", "experiment": "deep_sensor", "config": {}}], [1], 1, 60
    deep_suite.campaign(*args, limit=1)
    manifest = json.loads((directory / "checkpoints" / "manifest.json").read_text())
    checkpoint = directory / "checkpoints" / manifest["latest"]
    checkpoint.write_text(checkpoint.read_text() + " ")
    with pytest.raises(RuntimeError, match="Checkpoint hash mismatch"):
        deep_suite.campaign(*args)
    assert len(calls) == 1
