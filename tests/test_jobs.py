import time

from gigaam_ui.jobs import JobStore


def test_active_job_tracking():
    store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    assert not store.has_active_jobs()
    store.set_job("j1", status="queued")
    assert store.has_active_jobs()
    store.set_job("j1", status="running")
    assert store.has_active_jobs()
    store.set_job("j1", status="done")
    assert not store.has_active_jobs()


def test_purge_expired_jobs():
    store = JobStore(ttl_seconds=1, ttl_check_interval=3600)
    store.set_job("old", status="done", message="x")
    time.sleep(1.1)
    removed = store.purge_expired()
    assert removed == 1
    assert store.get_job("old") is None


def test_batch_item_updates():
    store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    store.set_job("batch", status="running", items=[{"filename": "a.webm", "status": "queued"}])
    store.update_item("batch", 0, status="running", progress=50)
    job = store.get_job("batch")
    assert job["items"][0]["status"] == "running"
    assert job["items"][0]["progress"] == 50
