"""Independent split and leakage checks using synthetic raw-feature fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from group_memory_qml import real_sensor_data as data


@pytest.fixture
def raw_fixture(monkeypatch):
    rng = np.random.default_rng(3701)
    arrays = {}
    for split, ids in (("train",range(1,22)),("test",range(22,31))):
        subjects = np.repeat(list(ids), 6*6)
        y = np.tile(np.repeat(np.arange(6),6),len(ids))
        x = rng.normal(size=(len(y),561))+subjects[:,None]/10+y[:,None]/3
        arrays.update({f"X_{split}":x, f"y_{split}":y, f"subjects_{split}":subjects})
    manifest = dict(npz_sha256="a"*64, official_train_subjects=list(range(1,22)),
                    official_test_subjects=list(range(22,31)), common_setup_timings={"download_s":1.})
    monkeypatch.setattr(data,"_read_prepared",lambda directory:(arrays,manifest))
    with threadpool_limits(limits=1):
        yield arrays


def load(**kwargs):
    config = dict(data_dir="unused", n_train=128,n_validation=128,n_test=256,seed=3702,n_features=16)
    config.update(kwargs)
    return data.load_sensor_dataset(**config)


def test_subjects_features_and_class_balance(raw_fixture):
    d = load()
    assert d.train_full.shape == (128,561)
    assert d.validation_full.shape == (128,561)
    assert d.test_full.shape == (256,561)
    assert d.train_angles.shape == (128,16)
    assert d.validation_angles.shape == (128,16)
    assert d.test_angles.shape == (256,16)
    for a,b in ((d.subjects_train,d.subjects_validation),(d.subjects_train,d.subjects_test),
                (d.subjects_validation,d.subjects_test)):
        assert not set(a)&set(b)
    assert len(set(d.subjects_train)) == 16
    assert len(set(d.subjects_validation)) == 5
    assert len(set(d.subjects_test)) == 9
    for y in (d.y_train,d.y_validation,d.y_test):
        counts = np.bincount(y,minlength=6)
        assert max(counts)-min(counts) <= 1
        assert set(y) == set(range(6))
    for x in (d.train_angles,d.validation_angles,d.test_angles):
        assert np.isfinite(x).all() and np.max(np.abs(x)) < np.pi
        assert not x.flags.writeable
    assert np.allclose(d.train_full.mean(axis=0),0,atol=1e-13)
    assert np.allclose(d.train_full.std(axis=0),1,atol=1e-13)
    assert d.class_names == data.CLASS_NAMES


def test_scaler_and_pca_fit_selected_training_only(raw_fixture):
    d = load()
    expected = raw_fixture["X_train"][d.indices_train]
    np.testing.assert_allclose(d.preprocessing["standardizer_mean"],expected.mean(axis=0),rtol=0,atol=1e-14)
    np.testing.assert_allclose(d.preprocessing["standardizer_scale"],expected.std(axis=0),rtol=0,atol=1e-14)
    # Alter every held-out feature strongly: neither fitted transform nor any
    # training feature may change. Check validation and test independently.
    validation_mask = np.isin(raw_fixture["subjects_train"],d.provenance["validation_subject_pool"])
    raw_fixture["X_train"][validation_mask] += 1e5
    raw_fixture["X_test"] *= -100
    other = load()
    assert other.preprocessing == d.preprocessing
    np.testing.assert_array_equal(other.train_full,d.train_full)
    np.testing.assert_array_equal(other.train_angles,d.train_angles)
    assert not np.allclose(other.validation_angles,d.validation_angles)
    assert not np.allclose(other.test_angles,d.test_angles)
    assert other.provenance["split_sha256"] == d.provenance["split_sha256"]


def test_nested_training_and_unchanged_validation_test(raw_fixture):
    small,large = load(n_train=128),load(n_train=512)
    np.testing.assert_array_equal(small.indices_train,large.indices_train[:128])
    np.testing.assert_array_equal(small.indices_validation,large.indices_validation)
    np.testing.assert_array_equal(small.indices_test,large.indices_test)
    assert small.provenance["validation_subject_pool"] == large.provenance["validation_subject_pool"]
    assert small.preprocessing["fit_train_indices_sha256"] != large.preprocessing["fit_train_indices_sha256"]
    assert small.preprocessing["transform_sha256"] != large.preprocessing["transform_sha256"]


def test_deterministic_hashes_and_no_global_rng_mutation(raw_fixture):
    np.random.seed(773)
    before = np.random.get_state()
    first,second = load(),load()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    assert first.provenance == second.provenance
    assert first.preprocessing == second.preprocessing
    np.testing.assert_array_equal(first.train_angles,second.train_angles)
    assert load(seed=3703).provenance["dataset_sha256"] != first.provenance["dataset_sha256"]
    for key,value in first.timings.items():
        assert np.isfinite(value) and value > 0
    assert sum(value for key,value in first.timings.items() if key != "total_preparation_s") <= first.timings["total_preparation_s"]
    json.dumps(first.provenance,allow_nan=False)
    json.dumps(first.preprocessing,allow_nan=False)


def test_input_count_guards_do_not_read_data(monkeypatch):
    monkeypatch.setattr(data,"_read_prepared",lambda path:pytest.fail("should validate arguments before I/O"))
    for kwargs in ({"n_train":True},{"n_train":5},{"n_features":128},{"n_features":0},
                   {"n_test":5},{"n_validation":0},{"seed":-1},{"seed":2**32}):
        with pytest.raises(ValueError):load(**kwargs)


def test_subset_guards_and_within_class_subject_balance():
    y=np.repeat(np.arange(6),9)
    subjects=np.tile(np.repeat([1,2,3],3),6)
    chosen=data._balanced_subset(y,subjects,np.arange(54),24,130)
    for label in range(6):
        counts=np.bincount(subjects[chosen[y[chosen]==label]],minlength=4)[1:]
        assert counts.max()-counts.min() <= 1
    for n in (5,55):
        with pytest.raises(ValueError):data._balanced_subset(y,subjects,np.arange(54),n,130)
    with pytest.raises(ValueError):data._balanced_subset(y,subjects,np.arange(9),6,130)


def test_loader_never_downloads_and_rejects_corrupted_npz(tmp_path):
    with pytest.raises(FileNotFoundError):data._read_prepared(tmp_path)
    (tmp_path/"har_features.npz").write_bytes(b"corrupt")
    (tmp_path/"prepared_manifest.json").write_text(json.dumps(dict(archive_sha256=data.ARCHIVE_SHA256,npz_sha256="0"*64)))
    with pytest.raises(ValueError,match="checksum"):
        data._read_prepared(tmp_path)


def test_raw_official_shape_validation_rejects_fixture(raw_fixture):
    with pytest.raises(ValueError,match="dimensions"):
        data._validate_raw(raw_fixture)


def test_preparation_cli_prints_compact_receipt(monkeypatch,capsys,tmp_path):
    import sys
    monkeypatch.setattr(sys,"argv",["real_sensor_data","--data-dir",str(tmp_path)])
    monkeypatch.setattr(data,"prepare_sensor_data",lambda path:dict(archive_sha256="a"*64,npz_sha256="b"*64,
        official_shapes={"X_train":[7352,561],"X_test":[2947,561]}))
    data.main()
    output=json.loads(capsys.readouterr().out)
    assert output["status"] == "PREPARED" and output["train_rows"] == 7352
    assert output["test_rows"] == 2947 and output["manifest"].endswith("prepared_manifest.json")
