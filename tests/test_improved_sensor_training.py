"""Fast protocol/numerical tests; synthetic fixtures are not accuracy evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from group_memory_qml import improved_sensor_training as training


def dataset_fixture(n=12, qubits=8):
    rng, fields = np.random.default_rng(881), {}
    for index, split in enumerate(("train", "validation", "test")):
        labels = np.arange(n) % 6
        centers = np.sin(np.arange(6)[:, None] + np.arange(qubits)[None, :] * .7)
        angles = centers[labels] + rng.normal(0, .08, (n, qubits))
        full = rng.normal(0, .1, (n, 561))
        full[np.arange(n), labels] += 2
        fields.update({f"{split}_angles": angles, f"{split}_full": full, f"y_{split}": labels,
                       f"subjects_{split}": np.full(n, index+1, dtype=int),
                       f"indices_{split}": np.arange(n)+index*n})
    return SimpleNamespace(**fields, class_names=tuple(f"class{i}" for i in range(6)),
        provenance={"fixture": "Synthetic; not UCI evidence"},
        preprocessing={"fit_split": "train", "features": qubits}, timings={})


def config(steps=3, **updates):
    return dict(n_qubits=8, n_train=12, n_validation=12, n_test=12, steps=steps, **updates)


def install_data(monkeypatch, dataset):
    calls = []
    def load(data_dir, cfg, seed):
        calls.append((data_dir, cfg.copy(), seed))
        return dataset
    monkeypatch.setattr(training, "_load_dataset", load)
    return calls


def analytic_features(x, theta):
    angles = x.reshape(len(x), -1, 2) + theta.reshape(-1, 2)[None, :, :]
    left, right = angles[:, :, 0], angles[:, :, 1]
    return (np.sin(left)*np.cos(right),
            np.stack((np.cos(left)*np.cos(right), -np.sin(left)*np.sin(right)), axis=-1))


def install_spy_model(monkeypatch, dataset):
    from group_memory_qml import real_sensor_model
    calls = []
    class Spy:
        def __init__(self, n_parameters, method, occurrences):
            self.diagnostics = {"total_circuit_evaluations": 0}
            self.method = method
        def evaluate(self, x, theta, gradient=False):
            split = next(s for s in ("train", "validation", "test")
                         if x is getattr(dataset, s+"_angles"))
            calls.append((split, gradient, theta.copy(), self.method))
            f, j = analytic_features(x, theta)
            return f, j if gradient else None
    monkeypatch.setattr(real_sensor_model, "PatchResponseModel", Spy)
    return calls


def test_all_qml_engines_match_over_longer_than_old_sixty_step_limit(monkeypatch):
    install_data(monkeypatch, dataset_fixture())
    results = [training.run(9331, method, config(101, validation_only=True), "/fixture")
               for method in training.QML_METHODS]
    for result in results:
        json.dumps(result, allow_nan=False)
        assert result["diagnostics"]["completed_steps"] == 101
        assert result["diagnostics"]["test_evaluations"] == 0
        assert result["test_metrics"] is None
        assert result["diagnostics"]["parameter_training"]["moved_parameters"] == 8
        assert len(result["diagnostics"]["training_trajectory"]) == 101
        np.testing.assert_allclose(result["validation_vector"], results[0]["validation_vector"], atol=1e-10, rtol=0)
        assert result["diagnostics"]["split_sha256"] == results[0]["diagnostics"]["split_sha256"]
        assert result["application_wall_s"] >= result["diagnostics"]["timing"]["train_select_test_s"]


@pytest.mark.parametrize("epsilon", (1e-8, 1e-6, 1e-4))
def test_joint_gradient_and_adam_updates_are_auditable(monkeypatch, epsilon):
    dataset = dataset_fixture()
    install_data(monkeypatch, dataset)
    install_spy_model(monkeypatch, dataset)
    result = training.run(2, "group_spectral", config(adam_epsilon=epsilon), "/fixture")
    assert result["diagnostics"]["optimizer"]["epsilon"] == epsilon
    audits = result["diagnostics"]["optimizer_audit"]
    first, second = audits
    parameters = np.asarray(first["parameters_before"])
    def independent_loss(p):
        t, w, b = training._unpack(p, 8)
        f, _ = analytic_features(dataset.train_angles, t)
        logits = f @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(logits).sum(axis=1))-logits[np.arange(12), dataset.y_train])
                     + .5e-4*np.sum(w*w))
    finite = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite[index] = (independent_loss(plus)-independent_loss(minus))/2e-6
    g1, g2 = np.asarray(first["gradient"]), np.asarray(second["gradient"])
    np.testing.assert_allclose(g1, finite, atol=2e-8, rtol=0)
    p1 = parameters - .03*g1/(np.abs(g1)+epsilon)
    np.testing.assert_allclose(first["parameters_after"], p1, atol=1e-12, rtol=0)
    m2, v2 = .9*.1*g1+.1*g2, .999*.001*g1*g1+.001*g2*g2
    p2 = p1-.03*(m2/(1-.9**2))/(np.sqrt(v2/(1-.999**2))+epsilon)
    np.testing.assert_allclose(second["parameters_after"], p2, atol=1e-12, rtol=0)


def test_frozen_quantum_computes_features_once_and_never_changes_angles(monkeypatch):
    dataset = dataset_fixture()
    install_data(monkeypatch, dataset)
    calls = install_spy_model(monkeypatch, dataset)
    result = training.run(44, "frozen_quantum", config(1000), "/fixture")
    assert [(s, g, m) for s, g, _, m in calls] == [
        ("train", False, "classical_adjoint"), ("validation", False, "classical_adjoint"),
        ("test", False, "classical_adjoint")]
    p = result["diagnostics"]["parameter_training"]
    np.testing.assert_array_equal(p["initial"], p["final"])
    np.testing.assert_array_equal(p["initial"], p["selected"])
    assert p["independent_parameters"] == 0 and p["moved_parameters"] == 0
    assert not np.array_equal(p["initial_parameters"][8:], p["final_parameters"][8:])
    assert result["diagnostics"]["timings"]["frozen_feature_setup_s"] > 0
    assert result["diagnostics"]["test_evaluations"] == 1


class NoTestAccess(SimpleNamespace):
    def __getattribute__(self, name):
        if name.startswith("test_") or name.endswith("_test"):
            raise AssertionError("Search accessed the held-out test partition")
        return super().__getattribute__(name)


@pytest.mark.parametrize("method", ("classical_adjoint", "frozen_quantum", "classical_logistic"))
def test_validation_only_does_not_access_any_test_field(monkeypatch, method):
    data = NoTestAccess(**vars(dataset_fixture()))
    calls = install_data(monkeypatch, data)
    result = training.run(3, method, config(2, validation_only=True), "/fixture")
    assert calls[0][1]["validation_only"] is True
    assert result["metric_name"] == "validation_balanced_accuracy"
    assert result["test_metrics"] is None
    assert result["diagnostics"]["test_evaluations"] == 0
    assert "test_labels" not in result["diagnostics"]
    assert result["diagnostics"]["included_partitions"] == ["train", "validation"]


def test_test_labels_cannot_change_training_or_validation_selection(monkeypatch):
    data = dataset_fixture()
    install_data(monkeypatch, data)
    original = training.run(123, "classical_adjoint", config(7), "/fixture")
    changed = deepcopy(data)
    changed.y_test = (changed.y_test+1) % 6
    install_data(monkeypatch, changed)
    altered = training.run(123, "classical_adjoint", config(7), "/fixture")
    for field in ("selected_checkpoint_step", "parameter_training", "training_trajectory", "test_probabilities"):
        assert original["diagnostics"][field] == altered["diagnostics"][field]
    assert original["validation_metrics"] == altered["validation_metrics"]


@pytest.mark.parametrize("method", training.QML_METHODS)
@pytest.mark.parametrize("steps", (24, 25))
def test_validation_response_warmup_is_matched_and_test_is_forward_only(monkeypatch, method, steps):
    data = dataset_fixture()
    install_data(monkeypatch, data)
    calls = install_spy_model(monkeypatch, data)
    result = training.run(15, method, config(steps), "/fixture")
    val = [call for call in calls if call[0] == "validation"]
    warm = steps >= 25 and method in ("group_spectral", "qiskit_response")
    assert val[0][1] is warm
    assert all(not call[1] for call in val[1:])
    assert len(val) == steps+1
    assert [c[0] for c in calls].count("test") == 1
    assert calls[-1][0] == "test" and calls[-1][1] is False
    np.testing.assert_array_equal(calls[-1][2], result["diagnostics"]["parameter_training"]["selected"])


def test_optional_early_stopping_counts_only_validation_nonimprovements(monkeypatch):
    data = dataset_fixture()
    install_data(monkeypatch, data)
    calls = install_spy_model(monkeypatch, data)
    real_metrics = training._metrics
    def constant_loss(y, probabilities):
        return {**real_metrics(y, probabilities), "log_loss": 1.0}
    monkeypatch.setattr(training, "_metrics", constant_loss)
    result = training.run(3, "group_spectral", config(1000, early_stopping_patience=4), "/fixture")
    assert result["diagnostics"]["completed_steps"] == 4
    assert result["diagnostics"]["stop_reason"] == "validation_patience_exhausted"
    assert result["diagnostics"]["selected_checkpoint_step"] == 0
    assert [c[0] for c in calls].count("validation") == 5
    assert [c[0] for c in calls].count("test") == 1


@pytest.mark.parametrize("validation_only", (False, True))
def test_classical_fits_both_views_but_selects_without_test_feedback(monkeypatch, validation_only):
    data, operations = dataset_fixture(), []
    class Candidate:
        classes_ = np.arange(6)
        def fit(self, x, y):
            self.view = "angles" if x.shape[1] == 8 else "full"
            assert x is getattr(data, "train_"+self.view) and y is data.y_train
            operations.append(("fit", self.view))
            return self
        def predict_proba(self, x):
            split = "validation" if x is getattr(data, "validation_"+self.view) else "test"
            assert x is getattr(data, split+"_"+self.view)
            operations.append(("predict", self.view, split))
            correct = .8 if self.view == "full" else .6
            p = np.full((len(x), 6), (1-correct)/5)
            p[np.arange(len(x)), np.arange(len(x))%6] = correct
            return p
    monkeypatch.setattr(training, "_make_classifier", lambda method, seed: Candidate())
    result = training._train_classical(data, 2, "classical_rbf", training._configuration(config(validation_only=validation_only, representation="pca_original")))
    expected = [("fit", "angles"), ("predict", "angles", "validation"),
                ("fit", "full"), ("predict", "full", "validation")]
    if not validation_only:
        expected += [("predict", "full", "test")]
    assert operations == expected
    assert result["selected_feature_view"] == "full561"
    assert result["test_evaluations"] == int(not validation_only)
    assert result["timings"]["candidate_fit_s"] == sum(c["fit_s"] for c in result["classical_candidates"])
    assert result["timings"]["candidate_validation_s"] == sum(c["validation_predict_s"] for c in result["classical_candidates"])


def test_strong_classical_controls_keep_fixed_hyperparameters():
    logistic = training._make_classifier("classical_logistic", 7)
    rbf = training._make_classifier("classical_rbf", 7)
    mlp = training._make_classifier("classical_mlp", 7)
    assert logistic.C == 1 and logistic.max_iter == 1000
    assert rbf.C == 10 and rbf.gamma == "scale" and rbf.probability
    assert mlp.hidden_layer_sizes == (32,) and mlp.max_iter == 1000 and not mlp.early_stopping


@pytest.mark.parametrize("bad", [{"steps": 2001}, {"steps": True}, {"steps": 0},
    {"n_qubits": 12}, {"head_l2": -1e-16}, {"angle_scale": 0}, {"angle_scale": np.nan},
    {"validation_only": 1}, {"early_stopping_patience": -1}, {"representation": "test_fitted"},
    {"adam_epsilon": 0}, {"adam_epsilon": -1e-8}, {"adam_epsilon": True},
    {"adam_epsilon": float("nan")}, {"adam_epsilon": .11}])
def test_invalid_config_fails_closed(bad):
    with pytest.raises(ValueError):
        training._configuration(bad)


def test_subject_overlap_fails_before_any_training(monkeypatch):
    data = dataset_fixture()
    data.subjects_validation[:] = 1
    install_data(monkeypatch, data)
    with pytest.raises(ValueError, match="subjects must be disjoint"):
        training.run(1, "classical_adjoint", config(), "/fixture")


@pytest.mark.parametrize("validation_only", (False, True))
def test_lda_control_recovers_unique_axes_and_selects_by_validation(monkeypatch, validation_only):
    data = dataset_fixture()
    rng = np.random.default_rng(721)
    axes = {}
    for split in ("train", "validation", "test"):
        axes[split] = rng.normal(size=(12, 5))
        setattr(data, split+"_angles", np.pi/2 + .5*np.arctan(axes[split][:, np.arange(8)%5]))
    seen = []
    class Candidate:
        classes_ = np.arange(6)
        def fit(self, x, y):
            self.dim = x.shape[1]
            seen.append(("fit", self.dim))
            if self.dim == 5:
                np.testing.assert_allclose(x, axes["train"], atol=1e-13, rtol=0)
            return self
        def predict_proba(self, x):
            seen.append(("predict", self.dim))
            correct = .9 if self.dim == 5 else .4
            p = np.full((len(x), 6), (1-correct)/5)
            p[np.arange(len(x)), np.arange(len(x))%6] = correct
            return p
    monkeypatch.setattr(training, "_make_classifier", lambda method, seed: Candidate())
    cfg = training._configuration(config(validation_only=validation_only))
    result = training._train_classical(data, 2, "classical_logistic", cfg)
    assert [v[1] for v in seen if v[0] == "fit"] == [8, 561, 5]
    assert result["selected_feature_view"] == "lda_axes"
    assert result["test_evaluations"] == int(not validation_only)
    assert len(result["classical_candidates"]) == 3
    assert result["timings"]["candidate_fit_s"] == sum(c["fit_s"] for c in result["classical_candidates"])
