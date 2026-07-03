import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock torch/gigaam for API tests without GPU stack installed.
if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.__version__ = "0.0.0+test"
    _mock_torch.version.cuda = None
    _mock_torch.cuda.is_available.return_value = False
    sys.modules["torch"] = _mock_torch

if "gigaam" not in sys.modules:
    sys.modules["gigaam"] = MagicMock()


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_TTL_SECONDS", "0")
    monkeypatch.setenv("JOB_TTL_CHECK_INTERVAL", "3600")

    for sub in ("inputs", "outputs", "logs", "temp"):
        (tmp_path / "data" / sub).mkdir(parents=True, exist_ok=True)

    yield tmp_path


@pytest.fixture()
def flask_app(workspace, monkeypatch):
    import importlib

    import gigaam_ui.config as config
    importlib.reload(config)

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "тестовая транскрипция"
    mock_result.segments = None
    mock_model.transcribe.return_value = mock_result

    import gigaam_ui.transcription as transcription
    importlib.reload(transcription)
    monkeypatch.setattr(transcription, "get_model", lambda job_store=None, job_id=None: mock_model)

    import app as app_module
    importlib.reload(app_module)
    from gigaam_ui.jobs import JobStore
    app_module.job_store = JobStore(ttl_seconds=0, ttl_check_interval=3600)
    app_module.app.config["TESTING"] = True
    yield app_module.app, mock_model


def make_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-ac", "1", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le", str(path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def make_webm(path: Path, duration: float = 2.0):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=880:duration={duration}",
            "-c:a", "libvorbis", str(path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
