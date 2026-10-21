"""Subject-disjoint UCI HAR data and training-only feature preparation.

No network access occurs in ``load_sensor_dataset``. Call ``prepare_sensor_data``
once to acquire the official archive and make a verified raw-feature NPZ. Data
and timing receipts belong outside the repository. The released 561 features
already contain the dataset authors' preprocessing; our scaler/PCA fits use
only the selected training rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any

DATASET_URL = "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
LICENSE_DISCREPANCY = ("Current UCI landing page lists CC BY 4.0; bundled 2013 README states "
                       "commercial use is prohibited. Both source statements are preserved; "
                       "no legal conclusion about the discrepancy is made.")
CLASS_NAMES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
ATTRIBUTION = ("Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). "
               "Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. "
               "https://doi.org/10.24432/C54S4K")


@dataclass(frozen=True)
class SensorDataset:
    train_angles: Any
    validation_angles: Any
    test_angles: Any
    train_full: Any
    validation_full: Any
    test_full: Any
    y_train: Any
    y_validation: Any
    y_test: Any
    subjects_train: Any
    subjects_validation: Any
    subjects_test: Any
    indices_train: Any
    indices_validation: Any
    indices_test: Any
    class_names: tuple[str, ...]
    provenance: dict
    timings: dict
    preprocessing: dict


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def prepare_sensor_data(data_dir):
    """Download once, verify a pinned archive, and prepare an unfitted NPZ.

    Returns a JSON-safe receipt separating download, extraction, text parsing,
    and NPZ serialization. Repeated calls verify existing artifacts and reuse
    them. No raw accelerometer/gyroscope windows are extracted.
    """
    from datetime import datetime, timezone
    import urllib.request
    import zipfile
    import numpy as np

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root/"uci_har_official.zip"
    receipt_path = root/"download_receipt.json"
    if not archive.exists():
        started = perf_counter()
        temporary = archive.with_suffix(".zip.part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=90) as response, temporary.open("wb") as stream:
            while block := response.read(1024*1024):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != ARCHIVE_SHA256:
            temporary.unlink()
            raise ValueError("Official UCI archive checksum changed; inspect before accepting a new version")
        temporary.replace(archive)
        _atomic_json(receipt_path, dict(source_url=DOWNLOAD_URL, archive_name=archive.name,
            archive_sha256=digest.hexdigest(), archive_bytes=archive.stat().st_size,
            download_seconds=perf_counter()-started,
            downloaded_utc=datetime.now(timezone.utc).isoformat()))
    if _sha(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("UCI source archive checksum mismatch")
    if not receipt_path.exists():
        raise ValueError("Archive exists without acquisition timing receipt; preserve provenance before using it")
    acquisition = json.loads(receipt_path.read_text())
    if acquisition["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Acquisition receipt does not identify the pinned source archive")
    prepared = root/"har_features.npz"
    manifest_path = root/"prepared_manifest.json"
    if prepared.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if _sha(prepared.read_bytes()) != manifest["npz_sha256"]:
            raise ValueError("Prepared raw-feature NPZ checksum mismatch")
        return manifest
    tick = perf_counter()
    paths = [f"{split}/{prefix}_{split}.txt" for split in ("train", "test")
             for prefix in ("X", "y", "subject")]
    paths += ["README.txt", "features.txt", "features_info.txt", "activity_labels.txt"]
    source_hashes = {}
    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for relative in paths:
            raw = inner.read("UCI HAR Dataset/"+relative)
            destination = root/"released"/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            source_hashes[relative] = _sha(raw)
    extraction_s = perf_counter()-tick
    tick = perf_counter()
    arrays = {}
    for split in ("train", "test"):
        arrays[f"X_{split}"] = np.loadtxt(root/"released"/split/f"X_{split}.txt", dtype=np.float64)
        arrays[f"y_{split}"] = np.loadtxt(root/"released"/split/f"y_{split}.txt", dtype=np.int64)-1
        arrays[f"subjects_{split}"] = np.loadtxt(root/"released"/split/f"subject_{split}.txt", dtype=np.int64)
    _validate_raw(arrays)
    text_parse_s = perf_counter()-tick
    tick = perf_counter()
    temporary = prepared.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(prepared)
    npz_write_s = perf_counter()-tick
    manifest = dict(dataset="UCI HAR", dataset_id=240, dataset_url=DATASET_URL,
        citation=ATTRIBUTION, doi="10.24432/C54S4K", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        license_source=DATASET_URL, license_verified_utc="2026-10-20",
        bundled_readme_license="Attribution required; commercial use prohibited",
        license_discrepancy=LICENSE_DISCREPANCY,
        archive_sha256=ARCHIVE_SHA256, acquisition=acquisition,
        released_file_sha256=source_hashes, npz_name=prepared.name,
        npz_sha256=_sha(prepared.read_bytes()), npz_bytes=prepared.stat().st_size,
        official_shapes={k:list(v.shape) for k,v in arrays.items()},
        official_train_subjects=sorted(np.unique(arrays["subjects_train"]).tolist()),
        official_test_subjects=sorted(np.unique(arrays["subjects_test"]).tolist()),
        class_names=list(CLASS_NAMES), labels="Zero-based 0..5, original labels 1..6",
        common_setup_timings=dict(download_s=acquisition["download_seconds"], extraction_s=extraction_s,
                                  raw_text_parse_s=text_parse_s, npz_write_s=npz_write_s),
        preparation_scope="Raw released 561-feature arrays only; no fitted preprocessing or train/validation/test selection",
        caveat="Released windows overlap 50%; subjects, not windows, define independent split units")
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_raw(arrays):
    import numpy as np
    for split, count in (("train", 7352), ("test", 2947)):
        x, y, subjects = [arrays[f"{prefix}_{split}"] for prefix in ("X", "y", "subjects")]
        if x.shape != (count, 561) or y.shape != (count,) or subjects.shape != (count,):
            raise ValueError("Unexpected official UCI HAR array dimensions")
        if not np.isfinite(x).all() or set(np.unique(y)) != set(range(6)):
            raise ValueError("Nonfinite features or invalid activity labels")
        if y.dtype.kind not in "iu" or subjects.dtype.kind not in "iu":
            raise ValueError("Activity and subject labels must be integers")
    a, b = set(arrays["subjects_train"]), set(arrays["subjects_test"])
    if len(a) != 21 or len(b) != 9 or a & b or a | b != set(range(1, 31)):
        raise ValueError("Official subject-disjoint split is invalid")


def _read_prepared(data_dir):
    import numpy as np
    root = Path(data_dir)
    if not (root/"prepared_manifest.json").exists() or not (root/"har_features.npz").exists():
        raise FileNotFoundError("Prepare UCI HAR first with prepare_sensor_data(data_dir); the loader never downloads implicitly")
    manifest = json.loads((root/"prepared_manifest.json").read_text())
    if manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Prepared metadata references an unexpected source archive")
    raw = (root/"har_features.npz").read_bytes()
    if _sha(raw) != manifest["npz_sha256"]:
        raise ValueError("Prepared raw-feature NPZ checksum mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        arrays = {name:saved[name] for name in saved.files}
    _validate_raw(arrays)
    return arrays, manifest


def _balanced_subset(labels, subjects, candidates, size, seed):
    """Nested, class-balanced prefixes with round-robin subjects per class."""
    import numpy as np
    if size < 6 or size > len(candidates):
        raise ValueError("Requested subset must contain at least six rows and fit its subject pool")
    rng = np.random.default_rng(seed)
    labels, subjects, candidates = np.asarray(labels), np.asarray(subjects), np.asarray(candidates)
    queues = {}
    for activity in range(6):
        rows = candidates[labels[candidates] == activity]
        if not len(rows):
            raise ValueError("A subject pool lacks one of the six activities")
        ids = rng.permutation(np.unique(subjects[rows]))
        per_subject = [rng.permutation(rows[subjects[rows] == subject]).tolist() for subject in ids]
        ordered = []
        for position in range(max(map(len, per_subject))):
            ordered.extend(group[position] for group in per_subject if position < len(group))
        queues[activity] = ordered
    class_order = rng.permutation(6).tolist()
    chosen, cursor = [], {activity:0 for activity in range(6)}
    # Exhausted classes are skipped only if a caller requests almost the whole
    # pool. Typical128/512 subsets remain balanced to within one row.
    while len(chosen) < size:
        progressed = False
        for activity in class_order:
            if cursor[activity] < len(queues[activity]):
                chosen.append(queues[activity][cursor[activity]])
                cursor[activity] += 1
                progressed = True
                if len(chosen) == size:
                    break
        if not progressed:
            raise ValueError("Insufficient rows in the requested class/subject pool")
    return np.asarray(chosen, dtype=np.int64)


def _hash_arrays(*arrays):
    import numpy as np
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def load_sensor_dataset(data_dir, n_train, n_validation, n_test, seed, n_features):
    """Load deterministic subjects/subsets and fit every transform on train only.

    Validation uses five subjects from the official training partition. Test
    uses only the nine official held-out subjects. Training-size changes retain
    the same validation/test selections and nested training-row prefixes.
    Subsets are class balanced and approximately subject balanced within each
    class; they are not samples preserving natural class prevalence.
    """
    started = perf_counter()
    for name, value in (("n_train",n_train), ("n_validation",n_validation), ("n_test",n_test),
                        ("seed",seed), ("n_features",n_features)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    if seed >= 2**32:
        raise ValueError("Seed must fit the deterministic PCA 32-bit random state")
    if min(n_train, n_validation, n_test) < 6 or n_features > min(561, n_train-1):
        raise ValueError("Need all six classes and PCA dimension no greater than min(561,n_train-1)")
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    tick = perf_counter()
    arrays, manifest = _read_prepared(data_dir)
    load_s = perf_counter()-tick
    tick = perf_counter()
    official_subjects = np.unique(arrays["subjects_train"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    validation_subjects = np.sort(rng.permutation(official_subjects)[:5])
    pool_validation = np.flatnonzero(np.isin(arrays["subjects_train"], validation_subjects))
    pool_train = np.flatnonzero(~np.isin(arrays["subjects_train"], validation_subjects))
    pool_test = np.arange(len(arrays["y_test"]))
    index = {"train":_balanced_subset(arrays["y_train"], arrays["subjects_train"], pool_train, n_train, np.random.SeedSequence([seed, 2001])),
             "validation":_balanced_subset(arrays["y_train"], arrays["subjects_train"], pool_validation, n_validation, np.random.SeedSequence([seed, 2002])),
             "test":_balanced_subset(arrays["y_test"], arrays["subjects_test"], pool_test, n_test, np.random.SeedSequence([seed, 2003]))}
    raw, labels, subjects = {}, {}, {}
    for split in index:
        original = "test" if split == "test" else "train"
        raw[split] = arrays[f"X_{original}"][index[split]].copy()
        labels[split] = arrays[f"y_{original}"][index[split]].copy()
        subjects[split] = arrays[f"subjects_{original}"][index[split]].copy()
    if any(set(subjects[a]) & set(subjects[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Selected subjects overlap across data partitions")
    split_s = perf_counter()-tick
    tick = perf_counter()
    scaler = StandardScaler().fit(raw["train"])
    full = {split:scaler.transform(values) for split,values in raw.items()}
    scale_s = perf_counter()-tick
    tick = perf_counter()
    pca = PCA(n_components=n_features, svd_solver="randomized", random_state=seed,
              iterated_power=4).fit(full["train"])
    projected = {split:pca.transform(values) for split,values in full.items()}
    pca_s = perf_counter()-tick
    tick = perf_counter()
    angle_scaler = StandardScaler().fit(projected["train"])
    angles = {split:2*np.arctan(angle_scaler.transform(values)) for split,values in projected.items()}
    angle_s = perf_counter()-tick
    tick = perf_counter()
    for values in (*full.values(), *angles.values()):
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite prepared sensor features")
        values.setflags(write=False)
    for values in (*labels.values(), *subjects.values(), *index.values()):
        values.setflags(write=False)
    split_hashes = {split:_hash_arrays(index[split], subjects[split], labels[split]) for split in index}
    features_hashes = {split:_hash_arrays(full[split], angles[split]) for split in index}
    transform_hash = _hash_arrays(scaler.mean_, scaler.scale_, pca.mean_, pca.components_,
                                  angle_scaler.mean_, angle_scaler.scale_)
    preprocessing = dict(fit_scope="selected_training_rows_only", raw_feature_count=561,
        n_features=n_features, standardizer_mean=scaler.mean_.tolist(), standardizer_scale=scaler.scale_.tolist(),
        pca_solver="randomized", pca_iterated_power=4, pca_seed=seed, pca_whiten=False,
        pca_components_sha256=_hash_arrays(pca.components_),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        angle_scaler_mean=angle_scaler.mean_.tolist(), angle_scaler_scale=angle_scaler.scale_.tolist(),
        angle_map="2*arctan(StandardScaler(PCA(StandardScaler(X))))", transform_sha256=transform_hash,
        fit_train_indices_sha256=_hash_arrays(index["train"]))
    provenance = dict(dataset="UCI HAR", dataset_id=240, dataset_url=DATASET_URL,
        citation=ATTRIBUTION, doi="10.24432/C54S4K", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        bundled_readme_license="Attribution required; commercial use prohibited",
        license_discrepancy=LICENSE_DISCREPANCY, archive_sha256=ARCHIVE_SHA256,
        prepared_npz_sha256=manifest["npz_sha256"], class_names=list(CLASS_NAMES),
        official_train_subjects=manifest["official_train_subjects"], official_test_subjects=manifest["official_test_subjects"],
        selected_subjects={s:sorted(np.unique(v).tolist()) for s,v in subjects.items()},
        validation_subject_pool=validation_subjects.tolist(),
        selected_indices={s:v.tolist() for s,v in index.items()},
        index_scope="train/validation indices reference official train; test indices reference official test",
        split_sha256=split_hashes, feature_sha256=features_hashes,
        dataset_sha256=_sha(json.dumps(dict(raw=manifest["npz_sha256"], split=split_hashes,
                                          features=features_hashes),sort_keys=True).encode()),
        class_counts={s:np.bincount(v,minlength=6).tolist() for s,v in labels.items()},
        common_setup_timings=manifest["common_setup_timings"],
        sampling="Class-balanced deterministic prefixes; round-robin subject sampling within each class",
        independence="Subjects are disjoint across train/validation/test; overlapping windows within subjects are not independent",
        test_policy="Official test subjects held out; no test values or labels fit preprocessing or select hyperparameters",
        released_preprocessing="Use UCI's released 561 time/frequency features; their historical normalization is inherited")
    hashing_s = perf_counter()-tick
    timings = dict(prepared_npz_read_verify_s=load_s, subject_split_and_subset_s=split_s,
        standardizer_fit_transform_s=scale_s, pca_fit_transform_s=pca_s,
        angle_fit_transform_s=angle_s, provenance_hash_s=hashing_s,
        total_preparation_s=perf_counter()-started)
    return SensorDataset(*(angles[s] for s in ("train","validation","test")),
                         *(full[s] for s in ("train","validation","test")),
                         *(labels[s] for s in ("train","validation","test")),
                         *(subjects[s] for s in ("train","validation","test")),
                         *(index[s] for s in ("train","validation","test")),
                         CLASS_NAMES, provenance, timings, preprocessing)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare the official UCI HAR features outside the source repository")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_sensor_data(args.data_dir)
    print(json.dumps(dict(status="PREPARED", manifest=str(args.data_dir/"prepared_manifest.json"),
                          archive_sha256=receipt["archive_sha256"], npz_sha256=receipt["npz_sha256"],
                          train_rows=receipt["official_shapes"]["X_train"][0],
                          test_rows=receipt["official_shapes"]["X_test"][0]), indent=2))


if __name__ == "__main__":
    main()
