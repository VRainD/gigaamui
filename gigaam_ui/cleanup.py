import shutil
from pathlib import Path

from gigaam_ui import config
from gigaam_ui.jobs import JobStore


class CleanupInProgressError(Exception):
    pass


def _remove_dir_contents(path: Path) -> int:
    count = 0
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return 0
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
            count += 1
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            count += 1
    return count


def clear_workspace(job_store: JobStore) -> dict:
    """Clear uploads, temp files, outputs, logs and job history. Does not touch the GigaAM model cache."""
    if job_store.has_active_jobs():
        raise CleanupInProgressError("Cannot clear workspace while processing is active")
    removed = {}
    for path in config.WORKSPACE_DIRS:
        removed[path.name] = _remove_dir_contents(path)
    job_store.clear_history()
    return {"target": "workspace", "removed": removed, "jobs_cleared": True}


def clear_target(target: str, job_store: JobStore | None = None) -> dict:
    if target == "workspace":
        if job_store is None:
            raise ValueError("job_store required for workspace cleanup")
        return clear_workspace(job_store)

    if target in ("outputs", "logs", "inputs", "temp"):
        if job_store and job_store.has_active_jobs():
            raise CleanupInProgressError("Cannot clear while processing is active")
        mapping = {
            "outputs": config.OUTPUTS_DIR,
            "logs": config.LOGS_DIR,
            "inputs": config.INPUTS_DIR,
            "temp": config.TEMP_DIR,
        }
        count = _remove_dir_contents(mapping[target])
        return {"target": target, "removed": count}

    if target == "cache":
        count = _remove_dir_contents(config.HF_CACHE_DIR)
        return {"target": target, "removed": count, "warning": "GigaAM model cache cleared; next run will re-download the model"}

    raise ValueError(f"Invalid target: {target}")
