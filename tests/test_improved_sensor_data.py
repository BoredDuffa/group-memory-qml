"""Train-only transformations, sealed search data and immutable fresh-row holdout."""
import copy
import hashlib
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from group_memory_qml import improved_sensor_data as data


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@pytest.fixture
def raw_fixture(monkeypatch):
    rng=np.random.default_rng(9162)
    subjects=np.repeat(np.arange(1,22),36)
    labels=np.tile(np.repeat(np.arange(6),6),21)
    x=rng.normal(size=(len(labels),561))+labels[:,None]*rng.normal(size=(1,561))
    arrays=dict(X_train=x,y_train=labels,subjects_train=subjects)
    test_subjects=np.repeat(np.arange(22,31),328)[:2947]
    test_labels=np.arange(2947)%6
    arrays.update(X_test=rng.normal(size=(2947,561)),y_test=test_labels,subjects_test=test_subjects)
    manifest=dict(npz_sha256='a'*64,official_train_subjects=list(range(1,22)),official_test_subjects=list(range(22,31)),common_setup_timings={})
    prior={0,100,327};purged=set(prior)
    for i in prior:
        for j in (i-1,i+1):
            if 0<=j<len(test_subjects) and test_subjects[j]==test_subjects[i]:purged.add(j)
    pool=np.array(sorted(set(range(2947))-purged))
    chosen=data._balanced_subset(test_labels,test_subjects,pool,256,37829)
    holdout=dict(schema=1,n_test=256,prior_evaluated_indices=sorted(prior),purged_indices=sorted(purged),
        final_test_indices=chosen.tolist(),prepared_npz_sha256='a'*64,
        final_split_sha256=data._hash_arrays(chosen,test_subjects[chosen],test_labels[chosen]),
        prior_evaluated_count=len(prior),purged_count=len(purged),index_scope='official test',limitations='row-level only')
    monkeypatch.setattr(data,'_FROZEN_HOLDOUT',holdout)
    monkeypatch.setattr(data,'HOLDOUT_MANIFEST_SHA256',digest(holdout))
    calls=[]
    def read_source(path,validation_only):
        calls.append(validation_only)
        selected={k:v for k,v in arrays.items() if not validation_only or k.endswith('_train')}
        return selected,manifest
    monkeypatch.setattr(data,'_read_source',read_source)
    with threadpool_limits(limits=1):
        yield arrays,manifest,calls


def load(**kwargs):
    config=dict(data_dir='unused',n_train=128,n_validation=128,n_test=256,seed=9163,n_features=16)
    config.update(kwargs)
    return data.load_improved_sensor_dataset(**config)


def test_frozen_real_manifest_and_explicit_previous_evidence():
    m=data.final_holdout_manifest()
    assert digest(m)==data.HOLDOUT_MANIFEST_SHA256
    assert m['prior_evaluated_count']==1248
    assert m['purged_count']==2376
    assert m['remaining_count']==571
    assert len(m['prior_attempt_evidence'])==192
    assert m['prior_runs']['quick']['attempt_count']==6
    assert m['prior_runs']['pilot']['attempt_count']==6
    assert m['prior_runs']['campaign']['attempt_count']==180
    assert sum(m['final_class_counts'])==256
    assert max(m['final_class_counts'])-min(m['final_class_counts'])==1
    assert not set(m['final_test_indices']) & set(m['purged_indices'])
    assert set(m['prior_evaluated_indices']) <= set(m['purged_indices'])
    m['final_test_indices'][0]=-1
    assert data.final_holdout_manifest()['final_test_indices'][0]>=0


@pytest.mark.parametrize('representation',['pca_original','pca_shifted','lda_tiled','lda_pca'])
def test_shapes_train_only_transform_and_subjects(raw_fixture,representation):
    arrays,_,_=raw_fixture
    d=load(representation=representation)
    assert d.train_angles.shape==(128,16) and d.test_angles.shape==(256,16)
    assert d.train_full.shape==(128,561)
    for a,b in ((d.subjects_train,d.subjects_validation),(d.subjects_train,d.subjects_test),(d.subjects_validation,d.subjects_test)):
        assert not set(a)&set(b)
    for x in (d.train_angles,d.validation_angles,d.test_angles,d.train_full):
        assert np.isfinite(x).all() and not x.flags.writeable
    np.testing.assert_allclose(d.preprocessing['standardizer_mean'],arrays['X_train'][d.indices_train].mean(axis=0),rtol=0,atol=1e-14)
    validation_mask=np.isin(arrays['subjects_train'],d.provenance['validation_subject_pool'])
    arrays['X_train'][validation_mask]+=1e5
    arrays['y_train'][validation_mask]=(arrays['y_train'][validation_mask]+1)%6
    arrays['X_test']*=100
    other=load(representation=representation)
    assert d.preprocessing==other.preprocessing
    np.testing.assert_array_equal(d.train_angles,other.train_angles)
    assert not np.allclose(d.validation_angles,other.validation_angles)


def test_search_never_receives_test_arrays_and_final_holdout_fixed(raw_fixture):
    _,_,calls=raw_fixture
    search=load(validation_only=True)
    for field in ('test_angles','test_full','y_test','subjects_test','indices_test'):
        assert getattr(search,field) is None
    assert search.provenance['class_counts']['test'] is None
    assert search.provenance['selected_indices']['test'] is None
    assert search.provenance['feature_sha256']['test'] is None
    first=load();second=load(seed=9164,n_train=256,n_features=8,representation='pca_original')
    np.testing.assert_array_equal(first.indices_test,second.indices_test)
    np.testing.assert_array_equal(search.train_angles,first.train_angles)
    assert search.preprocessing==first.preprocessing
    assert calls==[True,False,False]
    assert first.provenance['final_holdout_manifest_sha256']==second.provenance['final_holdout_manifest_sha256']


def test_fixed_lda_tiling_and_shift_formula(raw_fixture):
    d=load(representation='lda_tiled',angle_scale=.5)
    assert d.preprocessing['coordinate_schema']==[f'lda:{i%5}' for i in range(16)]
    assert d.preprocessing['intrinsic_lda_dimensions']==5
    assert d.preprocessing['lda']['shrinkage']=='auto'
    assert d.preprocessing['tiled_coordinates_add_information'] is False
    np.testing.assert_array_equal(d.train_angles[:,:5],d.train_angles[:,5:10])
    assert np.all(d.train_angles>np.pi/4) and np.all(d.train_angles<3*np.pi/4)
    scaled=load(representation='lda_tiled',angle_scale=1)
    np.testing.assert_allclose(scaled.train_angles-np.pi/2,2*(d.train_angles-np.pi/2),rtol=0,atol=1e-15)
    assert scaled.preprocessing['transform_sha256']!=d.preprocessing['transform_sha256']


def test_original_and_shifted_pca_share_basis(raw_fixture):
    a=load(representation='pca_original');b=load(representation='pca_shifted')
    assert a.preprocessing['pca']==b.preprocessing['pca']
    np.testing.assert_allclose(b.train_angles,np.pi/2+a.train_angles/4,rtol=0,atol=1e-15)


def test_manifest_and_dataset_identity_fail_closed(raw_fixture,monkeypatch):
    arrays,manifest,_=raw_fixture
    manifest['npz_sha256']='b'*64
    with pytest.raises(ValueError,match='different prepared'):load(validation_only=True)
    manifest['npz_sha256']='a'*64
    data._FROZEN_HOLDOUT['n_test']=255
    with pytest.raises(ValueError,match='checksum'):load()


def test_final_holdout_integrity_checks_neighbors_labels_and_duplicates(raw_fixture):
    arrays,manifest,_=raw_fixture
    m=data.final_holdout_manifest()
    altered=copy.deepcopy(m);altered['purged_indices'].append(200)
    with pytest.raises(ValueError,match='Neighbor'):data._validate_holdout(arrays,manifest,altered,False)
    altered=copy.deepcopy(m);altered['final_test_indices'][0]=altered['final_test_indices'][1]
    with pytest.raises(ValueError,match='duplicate'):data._validate_holdout(arrays,manifest,altered,False)
    arrays['y_test'][m['final_test_indices'][0]]=(arrays['y_test'][m['final_test_indices'][0]]+1)%6
    with pytest.raises(ValueError,match='signature'):load()


def test_invalid_arguments_fail_before_read(monkeypatch):
    monkeypatch.setattr(data,'_read_source',lambda *a:pytest.fail('must validate before data access'))
    for kw in ({'n_test':512},{'n_train':True},{'n_features':12},{'n_train':6},{'seed':2**32},
               {'representation':'unknown'},{'angle_scale':float('nan')},{'angle_scale':True},
               {'angle_scale':0},{'angle_scale':2},{'validation_only':1}):
        with pytest.raises(ValueError):load(**kw)


def test_search_io_opens_training_members_only(monkeypatch,tmp_path):
    # Spy on archive member access: reading/checksumming compressed bytes is
    # allowed, but loading any official-test arrays during search is forbidden.
    raw=b'fake NPZ bytes'
    (tmp_path/'har_features.npz').write_bytes(raw)
    manifest=dict(archive_sha256=data.ARCHIVE_SHA256,npz_sha256=hashlib.sha256(raw).hexdigest(),
        official_train_subjects=list(range(1,22)),official_test_subjects=list(range(22,31)))
    (tmp_path/'prepared_manifest.json').write_text(json.dumps(manifest))
    arrays={'X_train':np.zeros((7352,561)), 'y_train':np.arange(7352)%6,
            'subjects_train':np.tile(np.arange(1,22),351)[:7352]}
    seen=[]
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):
            seen.append(key)
            assert key.endswith('_train')
            return arrays[key]
    monkeypatch.setattr(np,'load',lambda *a,**k:Archive())
    got,_=data._read_source(tmp_path,True)
    assert set(seen)=={'X_train','y_train','subjects_train'}
    assert set(got)==set(seen)
