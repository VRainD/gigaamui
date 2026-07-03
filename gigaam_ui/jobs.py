import threading
import time
from datetime import datetime, timezone

ACTIVE_STATUSES = frozenset({"queued", "running"})


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.rstrip("Z"))
    except ValueError:
        return None


class JobStore:
    def __init__(self, ttl_seconds: int = 3600, ttl_check_interval: int = 60):
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._active_count = 0
        self._active_lock = threading.Lock()
        self.ttl_seconds = ttl_seconds
        self.ttl_check_interval = ttl_check_interval
        self._ttl_thread = threading.Thread(target=self._ttl_loop, daemon=True)
        self._ttl_thread.start()

    def set_job(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.setdefault(job_id, {})
            old_status = job.get("status")
            job.update(kwargs)
            job["updated_at"] = now_iso()
            new_status = job.get("status")
            self._sync_active_count(old_status, new_status)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update_item(self, batch_id: str, item_index: int, **kwargs):
        with self._lock:
            job = self._jobs.get(batch_id)
            if not job:
                return
            items = job.setdefault("items", [])
            if 0 <= item_index < len(items):
                items[item_index].update(kwargs)
            job["updated_at"] = now_iso()

    def has_active_jobs(self) -> bool:
        with self._active_lock:
            return self._active_count > 0

    def clear_history(self):
        with self._lock:
            self._jobs.clear()
            with self._active_lock:
                self._active_count = 0

    def purge_expired(self) -> int:
        if self.ttl_seconds <= 0:
            return 0
        cutoff = datetime.utcnow().timestamp() - self.ttl_seconds
        removed = 0
        with self._lock:
            to_delete = []
            for job_id, job in self._jobs.items():
                if job.get("status") not in ("done", "error"):
                    continue
                updated = parse_iso(job.get("updated_at", ""))
                if updated and updated.timestamp() < cutoff:
                    to_delete.append(job_id)
            for job_id in to_delete:
                del self._jobs[job_id]
                removed += 1
        return removed

    def _sync_active_count(self, old_status, new_status):
        with self._active_lock:
            if old_status in ACTIVE_STATUSES:
                self._active_count = max(0, self._active_count - 1)
            if new_status in ACTIVE_STATUSES:
                self._active_count += 1

    def _ttl_loop(self):
        while True:
            time.sleep(self.ttl_check_interval)
            if not self.has_active_jobs():
                self.purge_expired()
