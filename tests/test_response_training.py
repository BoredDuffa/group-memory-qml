"""Independent end-to-end checks keep the historical learning task unchanged."""
import numpy as np
import pytest

from group_memory_qml.response_training import METHODS, run


@pytest.mark.parametrize("experiment,config", [
    ("1A", {"n_qubits": 6, "bases": 4, "views": 4, "updates": 2}),
    ("1A", {"n_qubits": 6, "bases": 4, "views": 2, "updates": 2, "symmetry": False}),
    ("2A", {"n_train": 4, "n_test": 4, "steps": 2}),
    ("2A", {"n_train": 4, "n_test": 4, "steps": 2, "broken_symmetry": True}),
])
def test_training_matches_original_and_analytic_model(experiment, config):
    records = {method: run(experiment, 940021, method, config) for method in METHODS}
    reference = records["qiskit_symmetry"]
    for method, row in records.items():
        if method == "classical":
            continue
        np.testing.assert_allclose(row["validation_vector"], reference["validation_vector"],
                                   atol=1e-10, rtol=0)
        assert row["metric_value"] == reference["metric_value"]
    for field in ("evaluations", "total_cx"):
        assert records["group_memory_response"][field] == records["qiskit_response"][field]
    assert records["classical_same_model"]["total_cx"] == 0


@pytest.mark.parametrize("experiment,config", [
    ("1A", {"n_qubits": 6, "bases": 4, "views": 4, "updates": 20}),
    ("2A", {"n_train": 4, "n_test": 4, "steps": 8}),
])
def test_longest_training_trajectory_unchanged(experiment, config):
    reference = run(experiment, 940022, "qiskit_symmetry", config)
    response = run(experiment, 940022, "group_memory_response", config)
    np.testing.assert_allclose(response["validation_vector"], reference["validation_vector"],
                               atol=1e-10, rtol=0)
    assert response["total_cx"] < reference["total_cx"]


def test_no_unrecognized_config_or_negative_budgeted_workload():
    with pytest.raises(ValueError):
        run("1A", 1, "group_memory_response", {"unknown": 1})
    with pytest.raises(ValueError):
        run("2A", 1, "group_memory_response", {"steps": 0})
    with pytest.raises(ValueError):
        run("1A", 1, "group_memory_response", {"updates": 21})
