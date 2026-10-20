"""Small synthetic ring-pattern task; split independent bases before augmentation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def orbit_key(vector: np.ndarray) -> bytes:
    """Byte-exact dihedral representative; deliberately no approximate matching."""
    vector = np.asarray(vector, dtype="<f8")
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite one-dimensional vector")
    return min(
        np.ascontiguousarray(np.roll(v, shift)).tobytes()
        for v in (vector, vector[::-1]) for shift in range(len(vector))
    )


def four_views(vector: np.ndarray) -> np.ndarray:
    """Four label-preserving views, not four independent observations."""
    return np.stack([vector, np.roll(vector, 1), vector[::-1], np.roll(vector[::-1], 2)])


@dataclass
class SensorDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_base_ids: list[str]
    test_base_ids: list[str]
    dataset_sha256: str

    def metadata(self) -> dict:
        return {
            "dataset_sha256": self.dataset_sha256,
            "task": "Contiguous versus alternating signed-amplitude ring patterns",
            "source": "Deterministic synthetic data; no external download",
            "bases_per_split": 8,
            "independent_histogram_pairs_per_split": 4,
            "views_per_base": 4,
            "samples_per_split": 32,
            "train_base_ids": self.train_base_ids,
            "test_base_ids": self.test_base_ids,
            "split_unit": "Original bases; paired classes share amplitude histogram",
            "chance_accuracy": 0.5,
        }


def make_sensor_dataset(n_qubits: int = 6, seed: int = 846101) -> SensorDataset:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)) or n_qubits < 6 or n_qubits > 20 or n_qubits % 2:
        raise ValueError("n_qubits must be an even integer from 6 through 20")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    splits = {}
    seen = set()
    for split, namespace in [("train", 1103), ("test", 2207)]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_qubits, namespace]))
        inputs, labels, ids = [], [], []
        for pair in range(4):
            positive = .85 + rng.uniform(-.16, .16, n_qubits // 2)
            negative = -.85 + rng.uniform(-.16, .16, n_qubits // 2)
            contiguous = np.concatenate([positive, negative])
            alternating = np.empty(n_qubits)
            alternating[::2], alternating[1::2] = positive, negative
            if not np.array_equal(np.sort(contiguous), np.sort(alternating)):
                raise RuntimeError("Histogram construction mismatch")
            for label, base in enumerate((contiguous, alternating)):
                key = orbit_key(base)
                if key in seen:
                    raise RuntimeError("Unexpected duplicated orbit across independent bases")
                seen.add(key)
                inputs.extend(four_views(base))
                labels.extend([label] * 4)
                ids.extend([f"{split}-pair{pair}-class{label}"] * 4)
        splits[split] = (np.asarray(inputs, dtype=np.float64), np.asarray(labels), ids)
    digest = hashlib.sha256()
    for split in ("train", "test"):
        x, y, ids = splits[split]
        digest.update(x.astype("<f8").tobytes())
        digest.update(y.astype("<i8").tobytes())
        digest.update(json.dumps(ids).encode())
    return SensorDataset(*splits["train"][:2], *splits["test"][:2],
                         splits["train"][2], splits["test"][2], digest.hexdigest())


def classical_features(x: np.ndarray) -> np.ndarray:
    """An intentionally strong invariant classical comparator."""
    n = x.shape[1]
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2 / n
    correlations = np.stack([
        np.mean(x * np.roll(x, lag, axis=1), axis=1)
        for lag in range(1, n // 2 + 1)
    ], axis=1)
    return np.concatenate([np.mean(x, axis=1, keepdims=True), power, correlations], axis=1)

