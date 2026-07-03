from pathlib import Path

import pytest

from gigaam_ui.cleanup import CleanupInProgressError, clear_target
from gigaam_ui.config import INPUTS_DIR, OUTPUTS_DIR, ensure_dirs
from gigaam_ui.jobs import JobStore


def test_clear_workspace(workspace, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    importlib.reload(config)

    import gigaam_ui.cleanup as cleanup_mod
    importlib.reload(cleanup_mod)

    ensure_dirs()
    store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    (config.INPUTS_DIR / "test.mp3").write_text("x")
    (config.OUTPUTS_DIR / "out.txt").write_text("y")
    store.set_job("done-job", status="done")

    result = cleanup_mod.clear_target("workspace", job_store=store)
    assert result["jobs_cleared"] is True
    assert not list(config.INPUTS_DIR.iterdir())
    assert store.get_job("done-job") is None


def test_clear_blocked_during_processing(workspace, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    importlib.reload(config)

    store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    store.set_job("active", status="running")
    import gigaam_ui.cleanup as cleanup_mod
    importlib.reload(cleanup_mod)
    with pytest.raises(cleanup_mod.CleanupInProgressError):
        cleanup_mod.clear_target("workspace", job_store=store)


def test_clear_outputs_does_not_touch_model_cache(workspace, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    importlib.reload(config)

    ensure_dirs()
    store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    (config.OUTPUTS_DIR / "result.txt").write_text("data")
    import gigaam_ui.cleanup as cleanup_mod
    importlib.reload(cleanup_mod)
    cleanup_mod.clear_target("outputs", job_store=store)
    assert not list(config.OUTPUTS_DIR.iterdir())
