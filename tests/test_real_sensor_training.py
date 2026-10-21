"""Fast training/protocol tests using synthetic fixtures, never UCI performance claims."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from group_memory_qml import real_sensor_training as training


def fixture_dataset(n=12, qubits=8):
    rng = np.random.default_rng(881)
    fields = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        labels = np.arange(n) % 6
        centers = np.sin(np.arange(6)[:, None] + np.arange(qubits)[None, :] * 0.7)
        angles = centers[labels] + rng.normal(0, 0.08, (n, qubits))
        full = rng.normal(0, 0.1, (n, 561))
        full[np.arange(n), labels] += 2
        fields.update({
            f"{split}_angles": angles, f"{split}_full": full, f"y_{split}": labels,
            f"subjects_{split}": np.full(n, split_index + 1, dtype=int),
            f"indices_{split}": np.arange(n) + split_index * n,
        })
    return SimpleNamespace(
        **fields, class_names=tuple(f"class{i}" for i in range(6)),
        provenance={"fixture": "Synthetic training protocol test, not UCI"},
        preprocessing={"fit_split": "train", "features": qubits},
        timings={"total_preparation_s": 0.0},
    )


def config(steps=3):
    return dict(n_qubits=8, n_train=12, n_validation=12, n_test=12, steps=steps)


def install_dataset(monkeypatch, dataset):
    from group_memory_qml import real_sensor_data

    calls = []

    def load(data_dir, n_train, n_validation, n_test, seed, n_features):
        calls.append((data_dir, n_train, n_validation, n_test, seed, n_features))
        return dataset

    monkeypatch.setattr(real_sensor_data, "load_sensor_dataset", load)
    return calls


def analytic_features(x, theta):
    angles = x.reshape(len(x), -1, 2) + theta.reshape(-1, 2)[None, :, :]
    left, right = angles[:, :, 0], angles[:, :, 1]
    values = np.sin(left) * np.cos(right)
    jacobian = np.stack((np.cos(left) * np.cos(right), -np.sin(left) * np.sin(right)), axis=-1)
    return values, jacobian


def test_joint_softmax_and_angle_gradients_match_independent_finite_differences():
    rng = np.random.default_rng(978)
    x, labels = rng.normal(size=(12, 8)), np.arange(12) % 6
    theta = rng.normal(0, 0.3, 8)
    weights, bias = rng.normal(0, 0.2, (4, 6)), rng.normal(0, 0.1, 6)
    features, jacobian = analytic_features(x, theta)
    loss, _, gt, gw, gb = training._loss_gradients(features, jacobian, labels, weights, bias, 0.013)
    actual = training._pack(gt, gw, gb)
    parameters = training._pack(theta, weights, bias)

    def independent_loss(p):
        t, w, b = training._unpack(p, 8)
        f, _ = analytic_features(x, t)
        logits = f @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(logits).sum(axis=1))
        return np.mean(log_partition - logits[np.arange(len(labels)), labels]) + 0.013 * np.sum(w * w) / 2

    assert loss == pytest.approx(independent_loss(parameters), abs=1e-13)
    finite = np.empty_like(parameters)
    for i in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        finite[i] = (independent_loss(plus) - independent_loss(minus)) / 2e-6
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=0)


def test_actual_patch_methods_have_matching_complete_training_trajectories(monkeypatch):
    dataset = fixture_dataset()
    calls = install_dataset(monkeypatch, dataset)
    results = [training.run(9331, method, config(), "/fixture") for method in training.QML_METHODS]
    for result in results[1:]:
        np.testing.assert_allclose(
            result["validation_vector"], results[0]["validation_vector"], atol=1e-10, rtol=0
        )
        assert result["diagnostics"]["dataset_sha256"] == results[0]["diagnostics"]["dataset_sha256"]
        assert result["diagnostics"]["split_sha256"] == results[0]["diagnostics"]["split_sha256"]
        assert result["diagnostics"]["preprocessing_sha256"] == results[0]["diagnostics"]["preprocessing_sha256"]
    for result in results:
        json.dumps(result, allow_nan=False)
        diagnostic = result["diagnostics"]
        assert diagnostic["test_evaluations"] == 1
        assert diagnostic["parameter_training"]["independent_parameters"] == 8
        assert diagnostic["parameter_training"]["moved_parameters"] > 0
        assert len(np.unique(np.round(diagnostic["parameter_training"]["displacement"], 10))) > 1
        assert len(diagnostic["training_trajectory"]) == 3
        assert np.asarray(result["test_metrics"]["confusion_matrix"]).sum() == 12
        assert result["application_wall_s"] >= diagnostic["timing"]["train_select_test_s"] > 0
    assert len(calls) == 3


def test_adam_bias_correction_and_validation_checkpoint_are_reproducible(monkeypatch):
    install_dataset(monkeypatch, fixture_dataset())
    first = training.run(341, "classical_adjoint", config(), "/fixture")
    second = training.run(341, "classical_adjoint", config(), "/fixture")
    np.testing.assert_array_equal(first["validation_vector"], second["validation_vector"])
    diagnostic = first["diagnostics"]
    trace = diagnostic["training_trajectory"]
    g1, g2 = np.array(trace[0]["gradient"]), np.array(trace[1]["gradient"])
    expected_first = np.array(trace[0]["parameters_before"]) - 0.03 * g1 / (np.abs(g1) + 1e-8)
    np.testing.assert_allclose(trace[0]["parameters_after"], expected_first, atol=1e-12, rtol=0)
    m2, v2 = 0.9 * 0.1 * g1 + 0.1 * g2, 0.999 * 0.001 * g1 * g1 + 0.001 * g2 * g2
    expected_second = expected_first - 0.03 * (m2 / (1 - 0.9**2)) / (
        np.sqrt(v2 / (1 - 0.999**2)) + 1e-8
    )
    np.testing.assert_allclose(trace[1]["parameters_after"], expected_second, atol=1e-12, rtol=0)
    losses = [row["log_loss"] for row in diagnostic["validation_trajectory"]]
    assert diagnostic["selected_checkpoint_step"] == int(np.argmin(losses))
    assert first["validation_metrics"]["log_loss"] == min(losses)


@pytest.mark.parametrize("method", training.QML_METHODS)
@pytest.mark.parametrize("steps", (24, 25))
def test_validation_warmup_policy_and_final_test_only_once(monkeypatch, method, steps):
    from group_memory_qml import real_sensor_model

    dataset, calls = fixture_dataset(), []

    class SpyPatch:
        def __init__(self, n_parameters, method, occurrences):
            self.diagnostics = {"total_circuit_evaluations": 0, "native_cx_work": 0}

        def evaluate(self, x, theta, gradient=False):
            split = next(name for name in ("train", "validation", "test")
                         if x is getattr(dataset, f"{name}_angles"))
            calls.append((split, gradient, theta.copy()))
            features, jacobian = analytic_features(x, theta)
            return features, jacobian if gradient else None

    monkeypatch.setattr(real_sensor_model, "PatchResponseModel", SpyPatch)
    result = training._train_qml(dataset, 15, method, training._configuration(config(steps)))
    validation_calls = [row for row in calls if row[0] == "validation"]
    expected_warmup = steps >= 25 and method in ("group_spectral", "qiskit_response")
    assert validation_calls[0][1] is expected_warmup
    assert all(not row[1] for row in validation_calls[1:])
    assert len(validation_calls) == steps + 1
    assert len([row for row in calls if row[0] == "train" and row[1]]) == steps
    assert [row[0] for row in calls].count("test") == 1
    assert calls[-1][0] == "test" and calls[-1][1] is False
    np.testing.assert_array_equal(calls[-1][2], result["parameter_training"]["selected"])
    assert result["validation_response_strategy"]["warm_once"] is expected_warmup
    assert result["validation_response_strategy"]["threshold_steps"] == 25


def test_test_labels_cannot_select_qml_checkpoint(monkeypatch):
    dataset = fixture_dataset()
    install_dataset(monkeypatch, dataset)
    original = training.run(123, "classical_adjoint", config(), "/fixture")
    changed = deepcopy(dataset)
    changed.y_test = (changed.y_test + 1) % 6
    install_dataset(monkeypatch, changed)
    altered = training.run(123, "classical_adjoint", config(), "/fixture")
    assert original["diagnostics"]["selected_checkpoint_step"] == altered["diagnostics"]["selected_checkpoint_step"]
    assert original["diagnostics"]["parameter_training"] == altered["diagnostics"]["parameter_training"]
    assert original["diagnostics"]["training_trajectory"] == altered["diagnostics"]["training_trajectory"]
    np.testing.assert_array_equal(
        original["diagnostics"]["test_probabilities"], altered["diagnostics"]["test_probabilities"]
    )


def test_classical_family_selects_view_on_validation_and_charges_both_fits(monkeypatch):
    dataset, operations = fixture_dataset(), []

    class Candidate:
        classes_ = np.arange(6)

        def fit(self, x, labels):
            self.view = "angles" if x.shape[1] == 8 else "full561"
            assert x is getattr(dataset, f"train_{'angles' if self.view == 'angles' else 'full'}")
            assert labels is dataset.y_train
            operations.append(("fit", self.view))
            return self

        def predict_proba(self, x):
            suffix = "angles" if self.view == "angles" else "full"
            split = "validation" if x is getattr(dataset, f"validation_{suffix}") else "test"
            assert x is getattr(dataset, f"{split}_{suffix}")
            operations.append(("predict", self.view, split))
            correct = 0.8 if self.view == "full561" else 0.6
            p = np.full((len(x), 6), (1 - correct) / 5)
            p[np.arange(len(x)), np.arange(len(x)) % 6] = correct
            return p

    monkeypatch.setattr(training, "_make_classifier", lambda method, seed: Candidate())
    result = training._train_classical(dataset, 42, "classical_rbf", config())
    assert result["selected_feature_view"] == "full561"
    assert operations == [
        ("fit", "angles"), ("predict", "angles", "validation"),
        ("fit", "full561"), ("predict", "full561", "validation"),
        ("predict", "full561", "test"),
    ]
    assert result["test_evaluations"] == 1
    assert len(result["classical_candidates"]) == 2
    assert result["timings"]["candidate_fit_s"] == sum(c["fit_s"] for c in result["classical_candidates"])
    assert result["timings"]["candidate_validation_s"] == sum(
        c["validation_predict_s"] for c in result["classical_candidates"]
    )


@pytest.mark.parametrize("method", training.METHODS[3:])
def test_classical_family_smoke_on_tiny_synthetic_fixture(monkeypatch, method):
    install_dataset(monkeypatch, fixture_dataset())
    result = training.run(47, method, config(1), "/fixture")
    json.dumps(result, allow_nan=False)
    assert result["diagnostics"]["selected_feature_view"] in ("angles", "full561")
    assert result["diagnostics"]["test_evaluations"] == 1
    assert len(result["diagnostics"]["classical_candidates"]) == 2
    assert result["n_parameters"] == 0
    assert 0 <= result["test_metrics"]["balanced_accuracy"] <= 1
    assert result["test_metrics"]["log_loss"] >= 0


@pytest.mark.parametrize("bad", [
    {"n_train": 2049}, {"n_train": True}, {"n_qubits": 12},
    {"steps": 0}, {"steps": 61}, {"learning_rate": float("nan")},
    {"learning_rate": True}, {"head_l2": -0.01}, {"head_l2": -1e-16},
    {"unexpected": 1},
])
def test_unsupported_configuration_is_rejected(bad):
    with pytest.raises(ValueError):
        training._configuration(bad)


@pytest.mark.parametrize("method,seed", [("classical", 2), ("group_spectral", True),
                                        ("group_spectral", -1), ("classical_mlp", 2**32)])
def test_invalid_identity_is_rejected_before_data_loading(method, seed):
    with pytest.raises(ValueError):
        training.run(seed, method, config(), "/does-not-exist")


def test_subject_leakage_is_rejected_before_model_training(monkeypatch):
    dataset = fixture_dataset()
    dataset.subjects_validation[:] = 1
    install_dataset(monkeypatch, dataset)
    with pytest.raises(ValueError, match="subjects must be disjoint"):
        training.run(12, "group_spectral", config(), "/fixture")


def test_missing_class_and_nonfinite_features_fail_closed():
    dataset = fixture_dataset()
    dataset.y_train[:] = 0
    with pytest.raises(ValueError, match="six integer class labels"):
        training._dataset_metadata(dataset, training._configuration(config()))
    dataset = fixture_dataset()
    dataset.test_angles[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite test features"):
        training._dataset_metadata(dataset, training._configuration(config()))
