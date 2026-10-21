import numpy as np
import pytest
from group_memory_qml.experiments.sensors import dataset, run


@pytest.mark.parametrize("experiment", ["1A", "1B"])
@pytest.mark.parametrize("symmetry", [True, False])
def test_matched_sensor_outputs(experiment, symmetry):
    cfg = {"n_qubits": 6, "views": 4, "bases": 4, "updates": 1, "symmetry": symmetry}
    rows = [run(experiment, 910101, method, cfg)
            for method in ("group_memory", "qiskit_symmetry", "qiskit_exact")]
    for row in rows[:2]:
        np.testing.assert_allclose(row["validation_vector"], rows[2]["validation_vector"],
                                   rtol=0, atol=1e-10)
    assert rows[0]["evaluations"] == rows[1]["evaluations"]
    if symmetry:
        assert rows[0]["evaluations"] < rows[2]["evaluations"]
    else:
        assert rows[0]["evaluations"] == rows[2]["evaluations"]


def test_denoising_split_and_joint_transform():
    (x, y, xt, yt), info = dataset("1B", 610101, n=6, views=4, bases=4)
    assert info["orbit_disjoint"]
    np.testing.assert_array_equal(x[1], x[0][::-1])
    np.testing.assert_array_equal(y[1], y[0][::-1])
    assert not np.array_equal(x, xt)


def test_classical_denooising_learns():
    r = run("1B", 610101, "classical", {"bases": 12})
    assert r["metric_value"] < r["diagnostics"]["raw_noisy_input_mse"]
