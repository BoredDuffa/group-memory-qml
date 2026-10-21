"""Independent circuit and reverse-gradient checks for parameter-response reuse."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from group_memory_qml.engine import QMLReuseEngine
from group_memory_qml.sensor_response import (SensorResponseEngine, ClassicalSensorEngine,
                                               METHODS)


def oracle(x, theta, symmetry=True):
    n = len(x)
    qc = QuantumCircuit(n)
    for q in range(n): qc.ry(float(x[q])+.31, q)
    for q in range(n): qc.rzz(float(theta[0]), q, (q+1)%n)
    gains = np.ones(n) if symmetry else np.linspace(.65,1.35,n)
    for q in range(n):
        qc.rx(float(theta[1]*gains[q]),q)
        qc.ry(.47,q)
    p=Statevector.from_instruction(qc).probabilities()
    idx=np.arange(2**n,dtype=np.uint64)
    return np.array([np.dot(p,1.-2*((idx>>q)&1)) for q in range(n)])


def examples(n=6):
    x=np.random.default_rng(991).normal(0,.7,n)
    return np.stack([x,np.roll(x,1),x[::-1],np.roll(x[::-1],2)])


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('n',[6,10])
def test_response_and_shared_derivatives_across_weights(method,n):
    x=examples(n)
    engine=SensorResponseEngine(n,method)
    reference=QMLReuseEngine(n)
    for theta in ([.53,-.41],[-.9,.2],[1.23,-1.11],[0.,0.],[np.pi/2,np.pi]):
        z,j=engine.evaluate(x,theta,True)
        expected=np.stack([oracle(row,theta) for row in x])
        _,jj=reference.evaluate(x,theta,True)
        np.testing.assert_allclose(z,expected,rtol=0,atol=1e-10)
        np.testing.assert_allclose(j,jj,rtol=0,atol=1e-10)
    assert engine.forward_count==5
    assert engine.gradient_count==engine.invalidations==0
    d=engine.diagnostics()
    assert d['response_orbits']==1 and d['plan_cache_hits']==4
    assert d['response_derivative_instances']==5
    assert d['max_probe_abs_error']<1e-10


def test_baseline_has_identical_probe_work_and_response_policy():
    x=np.concatenate([examples(),examples()*.73])
    engines=[SensorResponseEngine(6,m) for m in METHODS]
    for theta,grad in [([.3,.8],False),([.3,.8],True),([-.6,.2],True),([.2,.3],False)]:
        left,right=[e.evaluate(x,theta,grad) for e in engines]
        np.testing.assert_allclose(left[0],right[0],rtol=0,atol=1e-10)
        if grad: np.testing.assert_allclose(left[1],right[1],rtol=0,atol=1e-10)
        assert engines[0].forward_count==engines[1].forward_count
    assert engines[0].forward_count==12 # two direct states + ten calibration probes
    assert engines[0].diagnostics()['response_orbits']==2


def test_unseen_forward_only_input_uses_one_direct_circuit():
    e=SensorResponseEngine(6)
    x=examples()
    e.evaluate(x,[.4,.1],True)
    assert e.forward_count==5
    new=x*.81
    z,_=e.evaluate(new,[.4,.1])
    assert e.forward_count==6
    e.evaluate(new,[.4,.1])
    assert e.forward_count==6
    changed=np.array([.4,.1]);changed[0]=np.nextafter(changed[0],np.inf)
    e.evaluate(new,changed)
    assert e.forward_count==7 # exact weight-key distinction for direct cache
    np.testing.assert_allclose(z,np.stack([oracle(r,[.4,.1]) for r in new]),atol=1e-10,rtol=0)
    e.evaluate(new,changed,True)
    assert e.forward_count==12 # five probes; direct output is not a fit probe


def test_one_ulp_input_is_a_distinct_response_orbit_and_plan():
    e=SensorResponseEngine(6)
    x=examples()[0].copy()
    changed=x.copy(); changed[0]=np.nextafter(changed[0],np.inf)
    e.evaluate(np.stack([x,changed]),[.3,.2],True)
    assert e.forward_count==10
    assert e.diagnostics()['response_orbits']==2


def test_mutations_and_batch_reordering_do_not_poison_reuse():
    e=SensorResponseEngine(6)
    x=examples()
    theta=np.array([.3,.2])
    z,j=e.evaluate(x,theta,True)
    expected_z,expected_j=z.copy(),j.copy()
    z[:]=999;j[:]=-999
    e.source.x(0);e.native.x(0)
    gains=e.gains;gains[:]=18
    permutation=np.array([3,1,0,2])
    actual_z,actual_j=e.evaluate(x[permutation],theta,True)
    np.testing.assert_allclose(actual_z,expected_z[permutation],rtol=0,atol=1e-10)
    np.testing.assert_allclose(actual_j,expected_j[permutation],rtol=0,atol=1e-10)
    assert e.forward_count==5
    x[0,0]+=.001
    e.evaluate(x,theta,True)
    assert e.forward_count==10


def test_failed_validation_is_transactional_and_charges_probes(monkeypatch):
    e=SensorResponseEngine(6)
    simulate=e._simulate
    def corrupt(vectors,weights):
        result=simulate(vectors,weights)
        result[-1,0]+=.001
        return result
    monkeypatch.setattr(e,'_simulate',corrupt)
    with pytest.raises(RuntimeError,match='verification failed'):
        e.evaluate(examples(),[.3,.2],True)
    assert e.forward_count==5
    assert e.diagnostics()['response_orbits']==0
    assert e.diagnostics()['probe_failures']==1
    assert e.requests==0
    monkeypatch.setattr(e,'_simulate',simulate)
    e.evaluate(examples(),[.3,.2],True)
    assert e.forward_count==10
    assert e.diagnostics()['response_orbits']==1


@pytest.mark.parametrize('method',METHODS)
def test_nonuniform_gains_fall_back_without_response_reuse(method):
    e=SensorResponseEngine(6,method,symmetry=False)
    x=examples()
    z,j=e.evaluate(x,[.3,.2],True)
    ref=QMLReuseEngine(6,symmetry=False)
    zr,jr=ref.evaluate(x,[.3,.2],True)
    np.testing.assert_allclose(z,zr,rtol=0,atol=1e-10)
    np.testing.assert_allclose(j,jr,rtol=0,atol=1e-10)
    assert e.forward_count==e.gradient_count==4
    assert not e.group_enabled and not e.diagnostics()['response_supported']
    e.evaluate(x,[.31,.2],True)
    assert e.forward_count==8 and e.invalidations==1


@pytest.mark.parametrize('symmetry',[True,False])
def test_analytic_control_matches_independent_full_circuit_and_reverse(symmetry):
    x=examples()
    e=ClassicalSensorEngine(6,symmetry=symmetry)
    ref=QMLReuseEngine(6,symmetry=symmetry)
    for theta in ([.3,.2],[-1.2,1.5]):
        z,j=e.evaluate(x,theta,True)
        expected=np.stack([oracle(row,theta,symmetry) for row in x])
        _,jr=ref.evaluate(x,theta,True)
        np.testing.assert_allclose(z,expected,rtol=0,atol=1e-10)
        np.testing.assert_allclose(j,jr,rtol=0,atol=1e-10)
    assert e.forward_count==e.gradient_count==0


@pytest.mark.parametrize('cls',[SensorResponseEngine,ClassicalSensorEngine])
def test_empty_and_invalid_inputs_leave_counters_unchanged(cls):
    e=cls(6,gradients=False)
    z,j=e.evaluate(np.empty((0,6)),[.3,.2])
    assert z.shape==(0,6) and j is None
    for x,theta in [(np.zeros(6),[.3,.2]),(np.zeros((2,5)),[.3,.2]),
                    (np.full((1,6),np.nan),[.3,.2]),(np.zeros((1,6)),[.3]),
                    (np.zeros((1,6)),[np.inf,.2])]:
        with pytest.raises(ValueError): e.evaluate(x,theta)
    with pytest.raises(ValueError,match='gradients=True'):
        e.evaluate(np.zeros((1,6)),[.3,.2],True)
    assert e.forward_count==e.requests==0


def test_analytic_control_import_does_not_load_quantum_dependencies():
    import os
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from group_memory_qml.sensor_response import "
         "AnalyticSensorEngine; AnalyticSensorEngine(6); "
         "assert not any(k.startswith(('qiskit', 'qiskit_aer', 'qiskit_algorithms')) "
         "for k in sys.modules)"],
        env=os.environ.copy(), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
