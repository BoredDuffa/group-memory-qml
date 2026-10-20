"""Identical quantum learners and a separate classical model with explicit costs."""
from __future__ import annotations

import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import make_sensor_dataset, classical_features
from .engine import QMLReuseEngine

METHODS = ("group_memory", "qiskit_same_symmetry", "qiskit_exact_cache", "classical")


def _head(features, labels, seed):
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=seed)
    classifier.fit(scaler.transform(features), labels)
    return scaler, classifier


def run_trial(n_qubits=6, seed=846101, method="group_memory", updates=3, symmetry=True):
    """Time complete application work; import/harness costs belong to the CLI ledger.

    Each quantum step refits the logistic head and scaler, then takes the partial
    quantum gradient holding that fitted head/scaler fixed. We do not differentiate
    through the classical fitting operation.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= 20:
        raise ValueError("updates must be an integer from 1 through 20")
    start = time.perf_counter()
    stages = {}
    tick = time.perf_counter()
    data = make_sensor_dataset(n_qubits, seed)
    stages["data_s"] = time.perf_counter() - tick
    record = {"n_qubits": n_qubits, "seed": seed, "method": method,
              "symmetry": symmetry, "updates": updates, "data": data.metadata()}
    if method == "classical":
        tick = time.perf_counter()
        train = classical_features(data.train_x)
        test = classical_features(data.test_x)
        scaler, classifier = _head(train, data.train_y, seed)
        probabilities = classifier.predict_proba(scaler.transform(test))
        stages["features_fit_predict_s"] = time.perf_counter() - tick
        record.update(history=[], final_weights=None, forward_evaluations=0,
                      gradient_instances=0, cx_per_circuit=0, forward_cx_work=0,
                      cache_invalidations=0, requested_examples=0, calls=[])
    else:
        tick = time.perf_counter()
        engine = QMLReuseEngine(n_qubits, method=method, symmetry=symmetry, gradients=True)
        stages["setup_compile_s"] = time.perf_counter() - tick
        stages["evaluate_s"] = 0.
        stages["fit_predict_s"] = 0.
        weights = np.array([.53, -.41])
        history = []
        for step in range(updates):
            tick = time.perf_counter()
            z, jacobian = engine.evaluate(data.train_x, weights, gradient=True)
            stages["evaluate_s"] += time.perf_counter() - tick
            tick = time.perf_counter()
            features = z.mean(axis=1, keepdims=True)
            scaler, classifier = _head(features, data.train_y, seed)
            probabilities = classifier.predict_proba(scaler.transform(features))[:, 1]
            partial_gradient = np.mean((probabilities - data.train_y)[:, None] * jacobian, axis=0)
            partial_gradient *= float(classifier.coef_[0, 0] / scaler.scale_[0])
            partial_gradient /= max(1., np.linalg.norm(partial_gradient))
            p = np.clip(probabilities, 1e-12, 1-1e-12)
            loss = float(-np.mean(data.train_y*np.log(p) + (1-data.train_y)*np.log(1-p)))
            history.append({"step": step, "weights": weights.tolist(), "train_loss": loss,
                            "partial_gradient": partial_gradient.tolist()})
            weights = weights - .05 * partial_gradient
            stages["fit_predict_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        train_z, _ = engine.evaluate(data.train_x, weights)
        test_z, _ = engine.evaluate(data.test_x, weights)
        stages["evaluate_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        scaler, classifier = _head(train_z.mean(axis=1, keepdims=True), data.train_y, seed)
        probabilities = classifier.predict_proba(scaler.transform(test_z.mean(axis=1, keepdims=True)))
        stages["fit_predict_s"] += time.perf_counter() - tick
        record.update(
            history=history, final_weights=weights.tolist(),
            forward_evaluations=engine.forward_count,
            gradient_instances=engine.gradient_count,
            cx_per_circuit=engine.gates.get("cx", 0),
            forward_cx_work=engine.forward_count * engine.gates.get("cx", 0),
            cache_invalidations=engine.invalidations, requested_examples=engine.requests,
            calls=engine.calls, model_signature=engine.signature,
        )
    predictions = classifier.classes_[probabilities.argmax(axis=1)]
    record.update(
        heldout_accuracy=float(np.mean(predictions == data.test_y)),
        probabilities=probabilities.tolist(), predictions=predictions.tolist(),
        labels=data.test_y.tolist(), stages=stages,
        compute="CPU statevector and classical reverse-mode gradients; no QPU",
    )
    # Returned records are primitive JSON values; disk output is in harness wall.
    json.dumps(record, allow_nan=False)
    elapsed = time.perf_counter() - start
    stages["record_and_dispatch_s"] = elapsed - sum(stages.values())
    record["pipeline_s"] = elapsed
    return record


def validate_matched_block(records):
    """Compare predictions, trained weights and every quantum update."""
    by = {r["method"]: r for r in records}
    if len(records) != len(METHODS) or set(by) != set(METHODS):
        raise ValueError("A matched block requires all four methods")
    for key in ("n_qubits", "seed", "updates", "symmetry"):
        if len({r[key] for r in records}) != 1:
            raise AssertionError(f"Methods used different {key}")
    if len({r["data"]["dataset_sha256"] for r in records}) != 1:
        raise AssertionError("Methods did not see the same dataset")
    reference = by["qiskit_exact_cache"]
    max_prediction_error, max_gradient_error = 0., 0.
    for name in ("group_memory", "qiskit_same_symmetry"):
        current = by[name]
        p_error = float(np.max(np.abs(np.asarray(current["probabilities"]) - reference["probabilities"])))
        max_prediction_error = max(max_prediction_error, p_error)
        if p_error > 1e-10:
            raise AssertionError(f"{name}: prediction mismatch {p_error}")
        np.testing.assert_allclose(current["final_weights"], reference["final_weights"], atol=1e-10, rtol=0)
        if current["predictions"] != reference["predictions"]:
            raise AssertionError("Reuse changed the predicted labels")
        for actual, expected in zip(current["history"], reference["history"], strict=True):
            error = float(np.max(np.abs(np.asarray(actual["partial_gradient"]) - expected["partial_gradient"])))
            max_gradient_error = max(max_gradient_error, error)
            np.testing.assert_allclose(actual["partial_gradient"], expected["partial_gradient"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual["train_loss"], expected["train_loss"], atol=1e-10, rtol=0)
    for key in ("forward_evaluations", "gradient_instances", "cx_per_circuit"):
        if by["group_memory"][key] != by["qiskit_same_symmetry"][key]:
            raise AssertionError(f"Same-symmetry job mismatch: {key}")
    return {"status": "PASS", "max_prediction_error": max_prediction_error,
            "max_partial_gradient_error": max_gradient_error}
