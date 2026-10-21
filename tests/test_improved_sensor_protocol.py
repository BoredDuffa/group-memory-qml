"""Controller integrity tests with mocked workers; no benchmark timings."""
import json
from types import SimpleNamespace

import pytest

from group_memory_qml import improved_sensor_suite as suite


CONDITIONS = [{"id": "fixture", "experiment": "uci_har", "config": {}}]


def test_plan_has_all_matched_methods_and_deterministic_order():
    plan = suite.build_plan(CONDITIONS, [1, 2], 2)
    assert plan == suite.build_plan(CONDITIONS, [1, 2], 2)
    assert len(plan) == 28
    for seed in [1, 2]:
        for repeat in range(2):
            assert {r["method"] for r in plan if r["seed"] == seed
                    and r["repeat"] == repeat} == set(suite.METHODS)


@pytest.mark.parametrize("seeds,repeats", [([], 1), ([1, 1], 1), ([1], 0), ([1], True)])
def test_bad_plan_rejected(seeds, repeats):
    with pytest.raises(ValueError):
        suite.build_plan(CONDITIONS, seeds, repeats)


def fixture_campaign(tmp_path, monkeypatch):
    suite.platform.platform()  # Resolve macOS uname before mocking worker launch.
    data = tmp_path/"data"
    data.mkdir()
    (data/"fixture.npz").write_bytes(b"mock dataset identity")
    def fake_run(cmd, **kwargs):
        spec = json.loads(cmd[cmd.index("--spec")+1])
        target = cmd[cmd.index("--output")+1]
        suite.atomic(target, {"spec": spec, "source_hashes": suite.source_hashes()})
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    return data, tmp_path/"run"


def test_resume_preserves_journal_and_original_deadline(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    assert suite.campaign(out, CONDITIONS, [1], 1, 60, limit=2, data_dir=data) == "CHECKPOINTED"
    authorization = (out/"authorization.json").read_bytes()
    first = (out/"attempts/00000.json").read_bytes()
    first_cost = (out/"attempt_costs/00000.json").read_bytes()
    assert suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data) == "COMPLETE"
    assert (out/"authorization.json").read_bytes() == authorization
    assert (out/"attempts/00000.json").read_bytes() == first
    assert (out/"attempt_costs/00000.json").read_bytes() == first_cost
    assert json.loads((out/"checkpoints/most_complete.json").read_text())["completed_attempts"] == 7


def test_changed_data_prevents_resume(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    suite.campaign(out, CONDITIONS, [1], 1, 60, limit=1, data_dir=data)
    (data/"fixture.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="unchanged"):
        suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data)


def test_old_journal_tampering_prevents_resume(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    suite.campaign(out, CONDITIONS, [1], 1, 60, limit=3, data_dir=data)
    row = json.loads((out/"attempts/00000.json").read_text())
    row["process_wall_s"] = 0
    suite.atomic(out/"attempts/00000.json", row)
    with pytest.raises(RuntimeError, match="Journal content hash"):
        suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data)


def test_condition_specific_ablation_methods():
    conditions = [{"id": "ablation", "experiment": "improved_uci_har", "methods": ["group_spectral"]}]
    assert len(suite.build_plan(conditions, [1, 2], 2)) == 4
    conditions[0]["methods"] = ["unknown"]
    with pytest.raises(ValueError, match="supported methods"):
        suite.build_plan(conditions, [1], 1)


def test_monotonic_receipts_remain_valid_when_utc_moves_backwards(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    clock = SimpleNamespace(utc=1000.0, monotonic=100.0)
    monkeypatch.setattr(suite.time, 'time', lambda: clock.utc)
    monkeypatch.setattr(suite.time, 'perf_counter', lambda: clock.monotonic)

    def shifted_worker(cmd, **kwargs):
        spec = json.loads(cmd[cmd.index('--spec')+1])
        target = cmd[cmd.index('--output')+1]
        # A wall-clock correction is not a negative process duration.
        clock.monotonic += .3
        clock.utc -= .05
        suite.atomic(target, {'spec': spec, 'source_hashes': suite.source_hashes()})
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(suite.subprocess, 'run', shifted_worker)
    conditions = [{'id': 'clock', 'experiment': 'fixture', 'methods': ['group_spectral']}]
    assert suite.campaign(out, conditions, [1], 1, 60, data_dir=data) == 'COMPLETE'
    row = json.loads((out/'attempts/00000.json').read_text())
    assert row['clock_receipt_version'] == 2
    assert row['process_wall_s'] == pytest.approx(.3)
    assert row['attempt_start_monotonic_s'] == 100.0
    assert row['process_end_monotonic_s'] == pytest.approx(100.3)
    assert row['attempt_end_monotonic_s'] >= row['process_end_monotonic_s']
    assert row['attempt_wall_monotonic_s'] >= row['process_wall_s']
    assert row['utc_elapsed_s'] == pytest.approx(-.05)
    assert row['utc_minus_monotonic_s'] == pytest.approx(-.35)
    assert json.loads((out/'config.json').read_text())['clock_receipt_version'] == 2
    assert json.loads((out/'authorization.json').read_text())['deadline_unix'] == 1060.0


def test_timeout_preserves_monotonic_process_interval(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    clock = SimpleNamespace(utc=1000.0, monotonic=200.0)
    monkeypatch.setattr(suite.time, 'time', lambda: clock.utc)
    monkeypatch.setattr(suite.time, 'perf_counter', lambda: clock.monotonic)

    def timed_out(cmd, **kwargs):
        clock.utc += .2
        clock.monotonic += .25
        raise suite.subprocess.TimeoutExpired(cmd, kwargs['timeout'])

    monkeypatch.setattr(suite.subprocess, 'run', timed_out)
    conditions = [{'id': 'timeout', 'experiment': 'fixture', 'methods': ['group_spectral']}]
    suite.campaign(out, conditions, [1], 1, 60, data_dir=data)
    row = json.loads((out/'attempts/00000.json').read_text())
    assert row['status'] == 'TIMEOUT'
    assert row['process_wall_s'] == pytest.approx(.25)
    assert row['process_end_monotonic_s']-row['attempt_start_monotonic_s'] == row['process_wall_s']
    assert row['attempt_wall_monotonic_s'] == row['attempt_end_monotonic_s']-row['attempt_start_monotonic_s']
    assert row['timeout_s'] == 60


def test_worker_records_application_and_worker_monotonic_enclosures(tmp_path, monkeypatch):
    clock = SimpleNamespace(monotonic=100.0)
    monkeypatch.setattr(suite.time, 'perf_counter', lambda: clock.monotonic)
    actual_import = suite.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name != 'group_memory_qml.improved_sensor_training':
            return actual_import(name, *args, **kwargs)
        def fake_run(seed, method, config, data_dir):
            clock.monotonic += .5
            return {'method': method, 'application_wall_s': .5}
        return SimpleNamespace(run=fake_run)

    monkeypatch.setattr(suite.importlib, 'import_module', fake_import)
    spec = dict(seed=3, method='group_spectral', config={})
    output = tmp_path/'worker.json'
    suite.worker(spec, output, tmp_path)
    row = json.loads(output.read_text())
    assert row['worker_start_monotonic_s'] <= row['application_start_monotonic_s']
    assert row['application_start_monotonic_s'] <= row['application_end_monotonic_s']
    assert row['application_end_monotonic_s'] <= row['worker_end_monotonic_s']
    assert row['application_end_monotonic_s']-row['application_start_monotonic_s'] == row['application_s']
    assert row['worker_end_monotonic_s']-row['worker_start_monotonic_s'] == row['worker_before_write_s']


def test_attempt_cost_sidecars_charge_journal_work_and_reconcile_invocation(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    clock = SimpleNamespace(monotonic=100.0)
    monkeypatch.setattr(suite.time, 'perf_counter', lambda: clock.monotonic)
    original_atomic = suite.atomic

    def charged_write(path, value):
        original_atomic(path, value)
        clock.monotonic += .005

    monkeypatch.setattr(suite, 'atomic', charged_write)
    assert suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data) == 'COMPLETE'
    status = json.loads((out/'status.json').read_text())
    manifest = json.loads((out/'attempt_costs/manifest.json').read_text())
    assert manifest['completed_attempts'] == 7
    assert set(manifest['sidecar_sha256']) == {f'{i:05d}.json' for i in range(7)}
    summed = 0.0
    for i in range(7):
        path = out/'attempt_costs'/f'{i:05d}.json'
        cost = json.loads(path.read_text())
        result_path = out/'attempts'/f'{i:05d}.json'
        row = json.loads(result_path.read_text())
        assert suite.hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sidecar_sha256'][path.name]
        assert cost['result_sha256'] == suite.hashlib.sha256(result_path.read_bytes()).hexdigest()
        assert cost['spec'] == row['spec']
        assert cost['loop_start_monotonic_s'] <= row['attempt_start_monotonic_s']
        assert row['attempt_end_monotonic_s'] <= cost['loop_end_monotonic_s']
        assert cost['parent_loop_wall_s'] > cost['process_wall_s']
        assert cost['parent_overhead_s'] == pytest.approx(cost['pre_launch_parent_s']+cost['post_process_parent_s'])
        summed += cost['parent_loop_wall_s']
    assert status['summed_parent_loop_s'] == summed
    assert status['last_invocation_measured_s'] == pytest.approx(
        status['last_invocation_setup_s']+status['last_invocation_loop_s']+
        status['last_invocation_gap_s']+status['last_invocation_after_loop_s'])
    assert status['last_invocation_parent_remainder_s'] == pytest.approx(
        status['last_invocation_gap_s']+status['last_invocation_after_loop_s'])
    assert status['last_invocation_after_loop_s'] > 0
    assert status['attempt_cost_manifest_sha256'] == suite.hashlib.sha256((out/'attempt_costs/manifest.json').read_bytes()).hexdigest()


def test_attempt_cost_tampering_prevents_resume(tmp_path, monkeypatch):
    data, out = fixture_campaign(tmp_path, monkeypatch)
    suite.campaign(out, CONDITIONS, [1], 1, 60, limit=1, data_dir=data)
    path = out/'attempt_costs/00000.json'
    receipt = json.loads(path.read_text())
    receipt['parent_overhead_s'] += 10
    suite.atomic(path, receipt)
    with pytest.raises(RuntimeError, match='cost receipt hash mismatch'):
        suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data)
