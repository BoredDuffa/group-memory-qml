"""Longer, validation-selected sensor training with unchanged patch physics.

Search runs can exclude the test partition entirely. The same quantum model,
optimizer and exact response information are shared by all three QML engines.
A frozen-angle ablation computes each feature matrix once, while each classical
family pays for the angle and full561 views plus the uncompressed five LDA
axes when available. Validation chooses among all candidate views.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter
import warnings

import numpy as np

from .real_sensor_training import (
    N_CLASSES, _hash_arrays, _jsonable, _loss_gradients, _metrics, _pack,
    _softmax, _unpack, _aligned_probabilities,
)

QML_METHODS = ("group_spectral", "qiskit_response", "classical_adjoint")
METHODS = (*QML_METHODS, "frozen_quantum", "classical_logistic", "classical_rbf", "classical_mlp")
REPRESENTATIONS = ("pca_original", "pca_shifted", "lda_tiled", "lda_pca")


def _configuration(config=None):
    cfg = dict(n_qubits=16, n_train=512, n_validation=128, n_test=256,
               steps=1000, learning_rate=0.03, adam_epsilon=1e-8, head_l2=1e-4,
               representation="lda_tiled", angle_scale=0.5,
               validation_only=False, early_stopping_patience=0)
    supplied = dict(config or {})
    if set(supplied) - set(cfg):
        raise ValueError("Unknown improved-sensor configuration field")
    cfg.update(supplied)
    for name, low, high in (("n_qubits", 8, 16), ("n_train", 12, 2048),
                            ("n_validation", 6, 2048), ("n_test", 6, 4096),
                            ("steps", 1, 2000), ("early_stopping_patience", 0, 2000)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"Invalid {name}")
    if cfg["n_qubits"] not in (8, 16):
        raise ValueError("Use 8 or 16 total independently parameterized patch qubits")
    if cfg["representation"] not in REPRESENTATIONS or type(cfg["validation_only"]) is not bool:
        raise ValueError("Invalid representation or validation_only")
    for name, high in (("learning_rate", 1.0), ("adam_epsilon", 0.1), ("head_l2", 1.0), ("angle_scale", 1.0)):
        value = cfg[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0 or value > high
                or (name != "head_l2" and value == 0)):
            raise ValueError(f"Invalid {name}")
        cfg[name] = float(value)
    return cfg


def _load_dataset(data_dir, cfg, seed):
    from .improved_sensor_data import load_improved_sensor_dataset
    return load_improved_sensor_dataset(
        data_dir, cfg["n_train"], cfg["n_validation"], cfg["n_test"], seed, cfg["n_qubits"],
        representation=cfg["representation"], angle_scale=cfg["angle_scale"],
        validation_only=cfg["validation_only"],
    )


def _dataset_metadata(dataset, cfg):
    # In search mode this function never even asks for a test attribute.
    splits = ("train", "validation") if cfg["validation_only"] else ("train", "validation", "test")
    identities, features, ordered, subject_sets = {}, {}, [], {}
    for split in splits:
        size = cfg["n_" + split]
        angles = np.asarray(getattr(dataset, split + "_angles"))
        full = np.asarray(getattr(dataset, split + "_full"))
        labels = np.asarray(getattr(dataset, "y_" + split))
        subjects = np.asarray(getattr(dataset, "subjects_" + split))
        indices = np.asarray(getattr(dataset, "indices_" + split))
        if angles.shape != (size, cfg["n_qubits"]) or full.shape != (size, 561):
            raise ValueError(f"Invalid {split} feature shapes")
        if any(a.shape != (size,) for a in (labels, subjects, indices)):
            raise ValueError(f"Invalid {split} identities")
        if not np.isfinite(angles).all() or not np.isfinite(full).all():
            raise ValueError(f"Nonfinite {split} features")
        if labels.dtype.kind not in "iu" or set(labels.tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"Every {split} subset must contain all six integer labels")
        if subjects.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
            raise ValueError(f"Invalid {split} subject or row identities")
        subject_sets[split] = set(subjects.tolist())
        identities[split] = _hash_arrays(indices, subjects, labels)
        features[split] = {"angles": _hash_arrays(angles), "full561": _hash_arrays(full)}
        ordered.extend((angles, full, labels, subjects, indices))
    for position, left in enumerate(splits):
        if any(subject_sets[left] & subject_sets[right] for right in splits[position+1:]):
            raise ValueError("Training, validation and test subjects must be disjoint")
    preprocessing = _jsonable(dataset.preprocessing)
    return dict(dataset_sha256=_hash_arrays(*ordered), split_sha256=identities,
                feature_sha256=features, preprocessing_sha256=hashlib.sha256(
                    json.dumps(preprocessing, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                subjects={s: sorted(v) for s, v in subject_sets.items()},
                class_names=list(dataset.class_names), provenance=_jsonable(dataset.provenance),
                preprocessing=preprocessing, dataset_timings=_jsonable(dataset.timings),
                subject_disjoint=True, included_partitions=list(splits))


def _train_qml(dataset, seed, method, cfg):
    started = perf_counter()
    from .real_sensor_model import PatchResponseModel
    frozen = method == "frozen_quantum"
    backend = "classical_adjoint" if frozen else method
    model = PatchResponseModel(cfg["n_qubits"], method=backend, occurrences=2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61527]))
    initial_theta = rng.uniform(-0.7, 0.7, cfg["n_qubits"])
    head = rng.normal(0, 0.1, (cfg["n_qubits"] // 2, N_CLASSES))
    parameters = _pack(initial_theta, head, np.zeros(N_CLASSES))
    initial = parameters.copy()
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    timings = dict(model_setup_s=perf_counter()-started, frozen_feature_setup_s=0.0,
                   train_forward_gradient_s=0.0, optimizer_s=0.0, validation_s=0.0,
                   final_test_s=0.0)
    fixed_train = fixed_validation = None
    if frozen:
        tick = perf_counter()
        fixed_train, _ = model.evaluate(dataset.train_angles, initial_theta, gradient=False)
        fixed_validation, _ = model.evaluate(dataset.validation_angles, initial_theta, gradient=False)
        timings["frozen_feature_setup_s"] = perf_counter()-tick
    warm = not frozen and method in ("group_spectral", "qiskit_response") and cfg["steps"] >= 25
    tick = perf_counter()
    validation_features = fixed_validation
    if validation_features is None:
        validation_features, _ = model.evaluate(dataset.validation_angles, initial_theta, gradient=warm)
    probabilities, _ = _softmax(validation_features @ head)
    initial_validation = _metrics(dataset.y_validation, probabilities)
    best_metrics, best_parameters = initial_validation, parameters.copy()
    best_probabilities = probabilities.copy()
    best_step, stale_steps = 0, 0
    initial_validation_s = perf_counter()-tick
    timings["validation_s"] += initial_validation_s
    validation_vector = [*initial.tolist(), *probabilities.ravel().tolist()]
    trajectory, audits = [], []
    stop_reason = "requested_steps_completed"
    for step in range(1, cfg["steps"]+1):
        before = parameters.copy()
        theta, head, bias = _unpack(parameters, cfg["n_qubits"])
        tick = perf_counter()
        if frozen:
            train_features = fixed_train
            jacobian = np.zeros((cfg["n_train"], cfg["n_qubits"]//2, 2))
        else:
            train_features, jacobian = model.evaluate(dataset.train_angles, theta, gradient=True)
        timings["train_forward_gradient_s"] += perf_counter()-tick
        tick = perf_counter()
        loss, probabilities, gt, gw, gb = _loss_gradients(
            train_features, jacobian, dataset.y_train, head, bias, cfg["head_l2"])
        gradient = _pack(gt, gw, gb)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise RuntimeError("Nonfinite optimizer state")
        first_moment = 0.9*first_moment + 0.1*gradient
        second_moment = 0.999*second_moment + 0.001*gradient*gradient
        parameters = parameters - cfg["learning_rate"] * (first_moment/(1-0.9**step)) / (
            np.sqrt(second_moment/(1-0.999**step)) + cfg["adam_epsilon"])
        if not np.isfinite(parameters).all():
            raise RuntimeError("Nonfinite updated parameters")
        if frozen and not np.array_equal(parameters[:cfg["n_qubits"]], initial_theta):
            raise RuntimeError("Frozen-angle ablation changed a quantum parameter")
        timings["optimizer_s"] += perf_counter()-tick
        theta, head, bias = _unpack(parameters, cfg["n_qubits"])
        tick = perf_counter()
        validation_features = fixed_validation
        if validation_features is None:
            validation_features, _ = model.evaluate(dataset.validation_angles, theta, gradient=False)
        validation_probabilities, _ = _softmax(validation_features @ head + bias)
        validation_metrics = _metrics(dataset.y_validation, validation_probabilities)
        improved = validation_metrics["log_loss"] < best_metrics["log_loss"]
        if improved:
            best_metrics, best_step = validation_metrics, step
            best_parameters, best_probabilities = parameters.copy(), validation_probabilities.copy()
            stale_steps = 0
        else:
            stale_steps += 1
        timings["validation_s"] += perf_counter()-tick
        training_accuracy = float(np.mean(probabilities.argmax(axis=1) == dataset.y_train))
        trajectory.append(dict(step=step, training_regularized_log_loss=loss,
                               training_accuracy=training_accuracy,
                               validation_log_loss=validation_metrics["log_loss"],
                               validation_balanced_accuracy=validation_metrics["balanced_accuracy"],
                               gradient_l2=float(np.linalg.norm(gradient)),
                               parameter_step_l2=float(np.linalg.norm(parameters-before))))
        # Preserve every exact gradient and post-update parameter vector once.
        # Compact display trajectories deliberately do not duplicate these arrays.
        validation_vector.extend([loss, validation_metrics["log_loss"], *gradient.tolist(), *parameters.tolist()])
        if step <= 2:
            audits.append(dict(step=step, parameters_before=before.tolist(), gradient=gradient.tolist(),
                               parameters_after=parameters.tolist()))
        if cfg["early_stopping_patience"] and stale_steps >= cfg["early_stopping_patience"]:
            stop_reason = "validation_patience_exhausted"
            break
    selected_theta, selected_head, selected_bias = _unpack(best_parameters, cfg["n_qubits"])
    test_metrics = test_probabilities = test_predictions = None
    test_evaluations = 0
    validation_vector.extend([float(best_step), *best_parameters.tolist(), *best_probabilities.ravel().tolist()])
    if not cfg["validation_only"]:
        tick = perf_counter()
        test_features, _ = model.evaluate(dataset.test_angles, selected_theta, gradient=False)
        test_probabilities, _ = _softmax(test_features @ selected_head + selected_bias)
        test_metrics = _metrics(dataset.y_test, test_probabilities)
        test_predictions = test_probabilities.argmax(axis=1).tolist()
        validation_vector.extend([*test_features.ravel().tolist(), *test_probabilities.ravel().tolist()])
        test_probabilities = test_probabilities.tolist()
        test_evaluations = 1
        timings["final_test_s"] = perf_counter()-tick
    return dict(validation_metrics=best_metrics, test_metrics=test_metrics,
                test_probabilities=test_probabilities, test_predictions=test_predictions,
                test_evaluations=test_evaluations, validation_vector=validation_vector,
                training_trajectory=trajectory, initial_validation_metrics=initial_validation,
                selected_checkpoint_step=best_step, completed_steps=step, stop_reason=stop_reason,
                optimizer_audit=audits, validation_vector_layout={
                    "initial": "initial parameters, initial validation probabilities",
                    "each_step": "training regularized loss, validation logloss, gradient, post-update parameters",
                    "final": "selected step, selected parameters, selected validation probabilities, optional test features/probabilities",
                    "parameter_order": "theta, row-major softmax weights, biases",
                    "parameter_count": len(parameters)},
                parameter_training=dict(independent_parameters=0 if frozen else cfg["n_qubits"],
                    circuit_parameters=cfg["n_qubits"], frozen=frozen,
                    initial=initial_theta.tolist(), final=parameters[:cfg["n_qubits"]].tolist(),
                    selected=selected_theta.tolist(), displacement=(parameters[:cfg["n_qubits"]]-initial_theta).tolist(),
                    selected_displacement=(selected_theta-initial_theta).tolist(),
                    moved_parameters=int(np.sum(np.abs(parameters[:cfg["n_qubits"]]-initial_theta)>1e-12)),
                    initial_parameters=initial.tolist(), final_parameters=parameters.tolist(),
                    selected_parameters=best_parameters.tolist()),
                optimizer=dict(name="Deterministic full-batch Adam", beta1=.9, beta2=.999, epsilon=cfg["adam_epsilon"],
                    learning_rate=cfg["learning_rate"], head_l2=cfg["head_l2"],
                    early_stopping_patience=cfg["early_stopping_patience"]),
                checkpoint_selection="Minimum validation log loss including initialization; earliest exact tie",
                validation_response_strategy=dict(warm_once=warm, threshold_steps=25,
                    initial_validation_s=initial_validation_s, timing_included=True),
                ablation_backend=backend if frozen else None,
                engine=_jsonable(model.diagnostics), timings=timings)


def _make_classifier(method, seed):
    if method == "classical_logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1, solver="lbfgs", max_iter=1000, tol=1e-8)
    if method == "classical_rbf":
        from sklearn.svm import SVC
        return SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=seed)
    if method == "classical_mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(32,), solver="adam", max_iter=1000,
                             random_state=seed, early_stopping=False)
    raise ValueError("Unsupported classical method")


def _classical_view(dataset, split, view, cfg):
    if view == "lda_axes":
        # Invert the declared monotone encoding; the first five columns are the
        # five unique LDA coordinates, before tiling or PCA concatenation.
        angles = getattr(dataset, split + "_angles")[:, :5]
        return np.tan((angles - np.pi / 2) / cfg["angle_scale"])
    suffix = "angles" if view == "angles" else "full"
    return getattr(dataset, split + "_" + suffix)


def _train_classical(dataset, seed, method, cfg):
    candidates, fitted = [], []
    views = ["angles", "full561"]
    if cfg["representation"] in ("lda_tiled", "lda_pca"):
        views.append("lda_axes")
    for view in views:
        tick = perf_counter()
        model = _make_classifier(method, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(_classical_view(dataset, "train", view, cfg), dataset.y_train)
        fit_s = perf_counter()-tick
        tick = perf_counter()
        probabilities = _aligned_probabilities(model, _classical_view(dataset, "validation", view, cfg))
        metrics = _metrics(dataset.y_validation, probabilities)
        predict_s = perf_counter()-tick
        candidates.append(dict(view=view, method=method, fit_s=fit_s,
            validation_predict_s=predict_s, validation_metrics=metrics,
            warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))
        fitted.append((model, probabilities))
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation_metrics"]["log_loss"])
    model, probabilities = fitted[selected]
    view = candidates[selected]["view"]
    vector = probabilities.ravel().tolist()
    test_metrics = test_probabilities = test_predictions = None
    test_s = 0.0
    if not cfg["validation_only"]:
        tick = perf_counter()
        test = _classical_view(dataset, "test", view, cfg)
        test_probabilities = _aligned_probabilities(model, test)
        test_metrics = _metrics(dataset.y_test, test_probabilities)
        test_predictions = test_probabilities.argmax(axis=1).tolist()
        vector.extend(test_probabilities.ravel().tolist())
        test_probabilities = test_probabilities.tolist()
        test_s = perf_counter()-tick
    return dict(validation_metrics=candidates[selected]["validation_metrics"], test_metrics=test_metrics,
        test_probabilities=test_probabilities, test_predictions=test_predictions,
        validation_vector=vector, test_evaluations=int(not cfg["validation_only"]),
        selected_feature_view=view, feature_view_selection="Lowest validation log loss within this fixed family; angles win exact ties",
        classical_candidates=candidates, parameter_training=None, training_trajectory=[], engine={},
        timings=dict(candidate_fit_s=sum(c["fit_s"] for c in candidates),
                     candidate_validation_s=sum(c["validation_predict_s"] for c in candidates), final_test_s=test_s))


def run(seed, method, config=None, data_dir=None):
    """One fit with all application costs; enclosing worker must charge startup/exit."""
    started = perf_counter()
    if method not in METHODS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid method or seed")
    if data_dir is None:
        raise ValueError("An explicit prepared sensor data directory is required")
    cfg = _configuration(config)
    tick = perf_counter()
    dataset = _load_dataset(data_dir, cfg, seed)
    metadata = _dataset_metadata(dataset, cfg)
    preparation_s = perf_counter()-tick
    tick = perf_counter()
    trained = (_train_qml(dataset, seed, method, cfg) if method in (*QML_METHODS, "frozen_quantum")
               else _train_classical(dataset, seed, method, cfg))
    training_s = perf_counter()-tick
    engine = trained["engine"]
    metric_scope = "validation" if cfg["validation_only"] else "test"
    result = dict(experiment="improved_sensor", seed=seed, method=method, config=cfg,
                  n_qubits=cfg["n_qubits"], n_parameters=cfg["n_qubits"] if method in QML_METHODS else 0,
                  metric_name=metric_scope+"_balanced_accuracy", higher_is_better=True,
                  metric_value=trained[metric_scope+"_metrics"]["balanced_accuracy"],
                  evaluations=int(engine.get("total_circuit_evaluations", 0)),
                  total_cx=int(engine.get("native_cx_work", 0)),
                  validation_metrics=trained["validation_metrics"], test_metrics=trained["test_metrics"],
                  validation_vector=trained.pop("validation_vector"),
                  diagnostics={**metadata, **trained, "response": engine,
                    "chance_balanced_accuracy": 1/N_CLASSES,
                    "timing": dict(load_split_preprocess_hash_s=preparation_s, train_select_test_s=training_s,
                                   method_stages=trained["timings"]),
                    "compute": "Ideal CPU simulation or classical learner; no QPU or finite shots",
                    "entanglement_scope": "Independent two-qubit patches; classical softmax head",
                    "timing_scope": "Includes loading, train-only preprocessing, all updates/fits, validation selection and optional final test. Worker startup/imports/JSON/exit and search cost must additionally be reported."})
    if not cfg["validation_only"]:
        result["diagnostics"]["test_labels"] = np.asarray(dataset.y_test).tolist()
    result = _jsonable(result)
    json.dumps(result, allow_nan=False)
    result["application_wall_s"] = perf_counter()-started
    return result
