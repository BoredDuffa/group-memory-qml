"""Controller integrity tests with mocked workers; no benchmark timings."""
import json
from types import SimpleNamespace

import pytest

from group_memory_qml import real_sensor_suite as suite


CONDITIONS = [{"id": "fixture", "experiment": "uci_har", "config": {}}]


def test_plan_has_all_matched_methods_and_deterministic_order():
    plan = suite.build_plan(CONDITIONS, [1, 2], 2)
    assert plan == suite.build_plan(CONDITIONS, [1, 2], 2)
    assert len(plan) == 24
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
    assert suite.campaign(out, CONDITIONS, [1], 1, 60, data_dir=data) == "COMPLETE"
    assert (out/"authorization.json").read_bytes() == authorization
    assert (out/"attempts/00000.json").read_bytes() == first
    assert json.loads((out/"checkpoints/most_complete.json").read_text())["completed_attempts"] == 6


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
