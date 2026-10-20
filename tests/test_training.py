"""A bounded end-to-end learner test; it does not assert timing superiority."""

import copy

import numpy as np
import pytest

from group_memory_qml.training import METHODS, run_trial, validate_matched_block


def test_one_update_matches_across_quantum_methods_and_reports_classical_control():
    records = [
        run_trial(n_qubits=6, seed=846101, method=method, updates=1)
        for method in METHODS
    ]
    check = validate_matched_block(records)
    assert check["status"] == "PASS"
    by = {row["method"]: row for row in records}
    for row in records:
        assert row["pipeline_s"] > 0
        assert all(seconds >= 0 for seconds in row["stages"].values())
        assert sum(row["stages"].values()) == pytest.approx(row["pipeline_s"])
        assert row["data"]["independent_histogram_pairs_per_split"] == 4
        assert len(row["predictions"]) == len(row["labels"]) == 32
        assert row["heldout_accuracy"] == np.mean(
            np.asarray(row["predictions"]) == row["labels"]
        )
        # A regression check for this fixed, small synthetic fixture only:
        # this is neither a statistical confidence test nor a hardware claim.
        assert row["heldout_accuracy"] > 0.5
    grouped = by["group_memory"]
    exact = by["qiskit_exact_cache"]
    assert grouped["forward_evaluations"] == 24
    assert grouped["gradient_instances"] == 8
    assert exact["forward_evaluations"] == 4 * grouped["forward_evaluations"]
    assert exact["gradient_instances"] == 4 * grouped["gradient_instances"]
    assert grouped["forward_cx_work"] == (
        grouped["forward_evaluations"] * grouped["cx_per_circuit"]
    )
    assert grouped["cache_invalidations"] == 1
    assert not np.array_equal(grouped["final_weights"], [0.53, -0.41])
    classical = by["classical"]
    assert classical["gradient_instances"] == classical["forward_evaluations"] == 0
    assert classical["final_weights"] is None
    # The validator must reject invalid comparisons instead of publishing a
    # speed ratio between different data or a changed numerical computation.
    bad_data = copy.deepcopy(records)
    bad_data[0]["data"]["dataset_sha256"] = "different-dataset"
    with pytest.raises(AssertionError):
        validate_matched_block(bad_data)
    bad_prediction = copy.deepcopy(records)
    bad_prediction[0]["probabilities"][0][0] += 0.01
    with pytest.raises(AssertionError):
        validate_matched_block(bad_prediction)
    with pytest.raises(ValueError):
        validate_matched_block(records[:-1])
