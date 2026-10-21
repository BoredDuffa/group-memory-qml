"""Subject-disjoint UCI HAR training with complete, matched application costs.

The QML methods jointly train independent patch angles and a six-class softmax
head. Every update is deterministic full-batch Adam; the best validation
cross-entropy checkpoint is selected before one final test evaluation.

Each classical family is a separate benchmark method. It fits both the same
angle features and the train-standardized 561-feature view, charges both fits
and validation predictions, then selects the feature view by validation loss.
The test set does not select parameters, checkpoints, model families or views.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter
import warnings

import numpy as np


METHODS = (
    "group_spectral", "qiskit_response", "classical_adjoint",
    "classical_logistic", "classical_rbf", "classical_mlp",
)
QML_METHODS = METHODS[:3]
N_CLASSES = 6


def _configuration(config=None):
    cfg = dict(
        n_qubits=8, n_train=128, n_validation=128, n_test=256,
        steps=60, learning_rate=0.03, head_l2=1e-4,
    )
    config = dict(config or {})
    if set(config) - set(cfg):
        raise ValueError("Unknown real-sensor configuration field")
    cfg.update(config)
    for name, low, high in (
        ("n_qubits", 8, 16), ("n_train", 12, 2048),
        ("n_validation", 6, 2048), ("n_test", 6, 4096), ("steps", 1, 60),
    ):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"Invalid {name}")
    if cfg["n_qubits"] not in (8, 16):
        raise ValueError("The real-sensor plans support 8 or 16 total patch qubits")
    for name in ("learning_rate", "head_l2"):
        value = cfg[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid {name}")
        if (not np.isfinite(value) or value < 0 or value > 1
                or (name == "learning_rate" and value == 0)):
            raise ValueError(f"Invalid {name}")
        cfg[name] = float(value)
    return cfg


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(json.dumps({"shape": array.shape, "dtype": array.dtype.str}).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_metadata(dataset, cfg):
    split_hashes, feature_hashes = {}, {}
    ordered = []
    subject_sets = {}
    for split, count_key in (("train", "n_train"), ("validation", "n_validation"), ("test", "n_test")):
        angles = np.asarray(getattr(dataset, f"{split}_angles"))
        full = np.asarray(getattr(dataset, f"{split}_full"))
        y = np.asarray(getattr(dataset, f"y_{split}"))
        subjects = np.asarray(getattr(dataset, f"subjects_{split}"))
        indices = np.asarray(getattr(dataset, f"indices_{split}"))
        size = cfg[count_key]
        if angles.shape != (size, cfg["n_qubits"]) or full.shape != (size, 561):
            raise ValueError(f"Invalid {split} feature shapes")
        if any(array.shape != (size,) for array in (y, subjects, indices)):
            raise ValueError(f"Invalid {split} labels or split identities")
        if not np.isfinite(angles).all() or not np.isfinite(full).all():
            raise ValueError(f"Nonfinite {split} features")
        if y.dtype.kind not in "iu" or set(y.tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"Every {split} subset must contain all six integer class labels")
        subject_sets[split] = set(subjects.tolist())
        split_hashes[split] = _hash_arrays(indices, subjects, y)
        feature_hashes[split] = {
            "angles": _hash_arrays(angles), "full561": _hash_arrays(full),
        }
        ordered.extend((angles, full, y, subjects, indices))
    if any(subject_sets[left] & subject_sets[right] for left, right in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Training, validation and test subjects must be disjoint")
    preprocessing = _jsonable(dataset.preprocessing)
    return {
        "dataset_sha256": _hash_arrays(*ordered),
        "split_sha256": split_hashes,
        "feature_sha256": feature_hashes,
        "preprocessing_sha256": hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "subjects": {split: sorted(subjects) for split, subjects in subject_sets.items()},
        "class_names": list(dataset.class_names),
        "provenance": _jsonable(dataset.provenance),
        "preprocessing": preprocessing,
        "dataset_timings": _jsonable(dataset.timings),
        "subject_disjoint": True,
    }


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizers = np.sum(exponentials, axis=1, keepdims=True)
    return exponentials / normalizers, shifted - np.log(normalizers)


def _metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), N_CLASSES) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid class-probability array")
    if np.min(probabilities) < 0 or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Class probabilities are not normalized")
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    support = confusion.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("All six classes are required for balanced accuracy")
    return {
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(np.mean(np.diag(confusion) / support)),
        "log_loss": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1)))),
        "confusion_matrix": confusion.tolist(),
        "class_support": support.tolist(),
    }


def _loss_gradients(features, jacobian, labels, weights, bias, head_l2):
    probabilities, log_probabilities = _softmax(features @ weights + bias)
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1
    residual /= len(labels)
    feature_gradient = residual @ weights.T
    theta_gradient = np.sum(feature_gradient[:, :, None] * jacobian, axis=0).reshape(-1)
    weight_gradient = features.T @ residual + head_l2 * weights
    bias_gradient = residual.sum(axis=0)
    loss = -np.mean(log_probabilities[np.arange(len(labels)), labels])
    loss += 0.5 * head_l2 * np.sum(weights * weights)
    return float(loss), probabilities, theta_gradient, weight_gradient, bias_gradient


def _pack(theta, weights, bias):
    return np.concatenate((theta, weights.ravel(), bias))


def _unpack(parameters, n_qubits):
    theta = parameters[:n_qubits]
    stop = n_qubits + (n_qubits // 2) * N_CLASSES
    weights = parameters[n_qubits:stop].reshape(n_qubits // 2, N_CLASSES)
    return theta, weights, parameters[stop:]


def _train_qml(dataset, seed, method, cfg):
    started = perf_counter()
    from .real_sensor_model import PatchResponseModel
    model = PatchResponseModel(cfg["n_qubits"], method=method, occurrences=2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61527]))
    theta = rng.uniform(-0.7, 0.7, cfg["n_qubits"])
    weights = rng.normal(0, 0.1, (cfg["n_qubits"] // 2, N_CLASSES))
    bias = np.zeros(N_CLASSES)
    parameters = _pack(theta, weights, bias)
    initial = parameters.copy()
    first_moment, second_moment = np.zeros_like(parameters), np.zeros_like(parameters)
    setup_s = perf_counter() - started
    timing = dict(model_setup_s=setup_s, train_forward_gradient_s=0.0,
                  optimizer_s=0.0, validation_s=0.0, final_test_s=0.0)
    trace, validation_trace = [], []
    validation_grid_points = (2 * 2 + 1) ** 2
    warm_validation = (
        method in ("group_spectral", "qiskit_response")
        and cfg["steps"] >= validation_grid_points
    )
    tick = perf_counter()
    validation_features, _ = model.evaluate(
        dataset.validation_angles, theta, gradient=warm_validation
    )
    validation_probabilities, _ = _softmax(validation_features @ weights + bias)
    initial_validation = _metrics(dataset.y_validation, validation_probabilities)
    best_loss, best_step = initial_validation["log_loss"], 0
    best_parameters, best_validation = parameters.copy(), initial_validation
    best_validation_probabilities = validation_probabilities.copy()
    validation_trace.append({"step": 0, **initial_validation})
    initial_validation_s = perf_counter() - tick
    timing["validation_s"] += initial_validation_s
    validation_vector = [*initial.tolist(), best_loss, *validation_probabilities.ravel().tolist()]

    for step in range(1, cfg["steps"] + 1):
        before = parameters.copy()
        theta, weights, bias = _unpack(parameters, cfg["n_qubits"])
        tick = perf_counter()
        features, jacobian = model.evaluate(dataset.train_angles, theta, gradient=True)
        timing["train_forward_gradient_s"] += perf_counter() - tick
        tick = perf_counter()
        loss, probabilities, gt, gw, gb = _loss_gradients(
            features, jacobian, dataset.y_train, weights, bias, cfg["head_l2"]
        )
        gradient = _pack(gt, gw, gb)
        if not np.isfinite(gradient).all() or not np.isfinite(loss):
            raise RuntimeError("Nonfinite optimizer state")
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        corrected_first = first_moment / (1 - 0.9**step)
        corrected_second = second_moment / (1 - 0.999**step)
        parameters = parameters - cfg["learning_rate"] * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        if not np.isfinite(parameters).all():
            raise RuntimeError("Nonfinite updated parameters")
        timing["optimizer_s"] += perf_counter() - tick
        theta, weights, bias = _unpack(parameters, cfg["n_qubits"])
        tick = perf_counter()
        validation_features, _ = model.evaluate(dataset.validation_angles, theta, gradient=False)
        validation_probabilities, _ = _softmax(validation_features @ weights + bias)
        validation_metrics = _metrics(dataset.y_validation, validation_probabilities)
        if validation_metrics["log_loss"] < best_loss:
            best_loss, best_step = validation_metrics["log_loss"], step
            best_parameters = parameters.copy()
            best_validation = validation_metrics
            best_validation_probabilities = validation_probabilities.copy()
        validation_trace.append({"step": step, **validation_metrics})
        timing["validation_s"] += perf_counter() - tick
        trace.append({
            "step": step, "parameters_before": before.tolist(),
            "gradient": gradient.tolist(), "parameters_after": parameters.tolist(),
            "training_regularized_log_loss": loss,
            "training_accuracy": float(np.mean(probabilities.argmax(axis=1) == dataset.y_train)),
            "validation_log_loss": validation_metrics["log_loss"],
        })
        validation_vector.extend([
            *before.tolist(), *gradient.tolist(), loss, *parameters.tolist(),
            validation_metrics["log_loss"],
        ])

    final_theta, final_weights, final_bias = _unpack(parameters, cfg["n_qubits"])
    best_theta, best_weights, best_bias = _unpack(best_parameters, cfg["n_qubits"])
    tick = perf_counter()
    test_features, _ = model.evaluate(dataset.test_angles, best_theta, gradient=False)
    test_probabilities, _ = _softmax(test_features @ best_weights + best_bias)
    test_metrics = _metrics(dataset.y_test, test_probabilities)
    timing["final_test_s"] += perf_counter() - tick
    validation_vector.extend([
        float(best_step), *best_parameters.tolist(),
        *best_validation_probabilities.ravel().tolist(),
        *test_features.ravel().tolist(), *test_probabilities.ravel().tolist(),
    ])
    initial_theta, initial_weights, initial_bias = _unpack(initial, cfg["n_qubits"])
    return {
        "validation_metrics": best_validation, "test_metrics": test_metrics,
        "test_probabilities": test_probabilities.tolist(),
        "test_predictions": test_probabilities.argmax(axis=1).tolist(),
        "validation_vector": validation_vector,
        "training_trajectory": trace, "validation_trajectory": validation_trace,
        "selected_checkpoint_step": best_step,
        "checkpoint_selection": "Lowest validation log loss among initialization and every Adam update; earliest exact tie",
        "validation_response_strategy": {
            "warm_once": warm_validation,
            "threshold_steps": validation_grid_points,
            "threshold_formula": "(2 * occurrences + 1) ** parameters_per_patch = 25",
            "initial_validation_s": initial_validation_s,
            "initial_validation_timing_included_in_validation_s": True,
            "policy": "Warm the exact validation response only for response methods with at least 25 updates; test inputs are never warmed for selection",
        },
        "test_evaluations": 1,
        "parameter_training": {
            "independent_parameters": cfg["n_qubits"], "patches": cfg["n_qubits"] // 2,
            "steps": cfg["steps"], "initial": initial_theta.tolist(),
            "final": final_theta.tolist(), "selected": best_theta.tolist(),
            "displacement": (final_theta - initial_theta).tolist(),
            "selected_displacement": (best_theta - initial_theta).tolist(),
            "moved_parameters": int(np.sum(np.abs(final_theta - initial_theta) > 1e-12)),
            "initial_head_weights": initial_weights.tolist(), "initial_head_bias": initial_bias.tolist(),
            "final_head_weights": final_weights.tolist(), "final_head_bias": final_bias.tolist(),
            "selected_head_weights": best_weights.tolist(), "selected_head_bias": best_bias.tolist(),
        },
        "optimizer": {"name": "Full-batch joint Adam", "beta1": 0.9, "beta2": 0.999,
                      "epsilon": 1e-8, "learning_rate": cfg["learning_rate"], "head_l2": cfg["head_l2"]},
        "engine": _jsonable(model.diagnostics),
        "timings": timing,
    }


def _make_classifier(method, seed):
    if method == "classical_logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-8)
    if method == "classical_rbf":
        from sklearn.svm import SVC
        return SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=seed)
    if method == "classical_mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(32,), solver="adam", max_iter=200,
                             random_state=seed, early_stopping=False)
    raise ValueError("Unsupported classical family")


def _aligned_probabilities(model, features):
    raw = model.predict_proba(features)
    classes = np.asarray(model.classes_, dtype=int)
    if set(classes.tolist()) != set(range(N_CLASSES)):
        raise ValueError("Classifier did not fit all six classes")
    probabilities = np.empty((len(features), N_CLASSES), dtype=float)
    probabilities[:, classes] = raw
    return probabilities


def _train_classical(dataset, seed, method, cfg):
    candidates, fitted = [], []
    for view, train, validation in (
        ("angles", dataset.train_angles, dataset.validation_angles),
        ("full561", dataset.train_full, dataset.validation_full),
    ):
        tick = perf_counter()
        model = _make_classifier(method, seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(train, dataset.y_train)
        fit_s = perf_counter() - tick
        tick = perf_counter()
        probabilities = _aligned_probabilities(model, validation)
        metrics = _metrics(dataset.y_validation, probabilities)
        validation_predict_s = perf_counter() - tick
        candidates.append({
            "view": view, "method": method, "fit_s": fit_s,
            "validation_predict_s": validation_predict_s,
            "validation_metrics": metrics,
            "warnings": [{"category": warning.category.__name__, "message": str(warning.message)}
                         for warning in captured],
        })
        fitted.append((model, probabilities))
    selected = min(range(len(candidates)), key=lambda index: candidates[index]["validation_metrics"]["log_loss"])
    model, validation_probabilities = fitted[selected]
    view = candidates[selected]["view"]
    test_features = dataset.test_angles if view == "angles" else dataset.test_full
    tick = perf_counter()
    test_probabilities = _aligned_probabilities(model, test_features)
    test_metrics = _metrics(dataset.y_test, test_probabilities)
    test_predict_s = perf_counter() - tick
    return {
        "validation_metrics": candidates[selected]["validation_metrics"],
        "test_metrics": test_metrics,
        "test_probabilities": test_probabilities.tolist(),
        "test_predictions": test_probabilities.argmax(axis=1).tolist(),
        "validation_vector": [*validation_probabilities.ravel().tolist(), *test_probabilities.ravel().tolist()],
        "training_trajectory": [], "validation_trajectory": [],
        "selected_feature_view": view,
        "feature_view_selection": "Minimum validation log loss within this fixed classifier family; angles win exact ties",
        "classical_candidates": candidates,
        "test_evaluations": 1, "parameter_training": None, "engine": {},
        "timings": {
            "candidate_fit_s": sum(row["fit_s"] for row in candidates),
            "candidate_validation_s": sum(row["validation_predict_s"] for row in candidates),
            "final_test_s": test_predict_s,
        },
    }


def run(seed, method, config=None, data_dir=None):
    """Run one bounded method; enclosing worker timing must also charge imports/exit."""
    started = perf_counter()
    if method not in METHODS or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Invalid real-sensor method or seed")
    if data_dir is None:
        raise ValueError("An explicit prepared UCI HAR data directory is required")
    cfg = _configuration(config)
    tick = perf_counter()
    from .real_sensor_data import load_sensor_dataset
    dataset = load_sensor_dataset(
        data_dir, cfg["n_train"], cfg["n_validation"], cfg["n_test"], seed, cfg["n_qubits"]
    )
    metadata = _dataset_metadata(dataset, cfg)
    load_and_prepare_s = perf_counter() - tick
    tick = perf_counter()
    trained = (_train_qml(dataset, seed, method, cfg) if method in QML_METHODS
               else _train_classical(dataset, seed, method, cfg))
    train_select_test_s = perf_counter() - tick
    diagnostics = trained["engine"]
    result = {
        "experiment": "real_sensor", "seed": seed, "method": method, "config": cfg,
        "n_qubits": cfg["n_qubits"], "n_parameters": cfg["n_qubits"] if method in QML_METHODS else 0,
        "metric_name": "test_balanced_accuracy",
        "metric_value": trained["test_metrics"]["balanced_accuracy"], "higher_is_better": True,
        "evaluations": int(diagnostics.get("total_circuit_evaluations", 0)),
        "total_cx": int(diagnostics.get("native_cx_work", 0)),
        "validation_metrics": trained["validation_metrics"], "test_metrics": trained["test_metrics"],
        "validation_vector": trained.pop("validation_vector"),
        "diagnostics": {
            **metadata, **trained, "response": diagnostics,
            "test_labels": np.asarray(dataset.y_test).tolist(),
            "chance_balanced_accuracy": 1 / N_CLASSES,
            "timing": {
                "load_split_preprocess_hash_s": load_and_prepare_s,
                "train_select_test_s": train_select_test_s,
                "method_stages": trained["timings"],
            },
            "compute": "Ideal CPU simulation or classical learner; no QPU or finite shots",
            "entanglement_scope": "Independent two-qubit patches; classical softmax combines patch features",
            "selection_scope": "Validation data only; test evaluated once after checkpoint or feature-view selection",
            "timing_scope": "Function time includes dataset reading, train-only preprocessing, all candidates or updates, validation selection and final test; enclosing process must add interpreter startup/imports/serialization/exit. Dataset acquisition is separate.",
        },
    }
    # Reject hidden NumPy objects or nonfinite values before returning a result.
    result = _jsonable(result)
    json.dumps(result, allow_nan=False)
    result["application_wall_s"] = perf_counter() - started
    return result
