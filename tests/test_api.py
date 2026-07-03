import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import make_webm, make_wav, requires_ffmpeg


def test_api_info(flask_app):
    app, _ = flask_app
    client = app.test_client()
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.get_json()
    assert "model_name" in data
    assert "job_ttl_seconds" in data


@requires_ffmpeg
def test_single_file_transcribe_backward_compat(flask_app, workspace, tmp_path):
    app, mock_model = flask_app
    client = app.test_client()
    wav = tmp_path / "short.wav"
    make_wav(wav, duration=2.0)

    with open(wav, "rb") as f:
        res = client.post("/api/transcribe", data={"file": (f, "short.wav")}, content_type="multipart/form-data")

    assert res.status_code == 200
    job_id = res.get_json()["job_id"]

    for _ in range(30):
        status = client.get(f"/api/status/{job_id}").get_json()
        if status.get("status") in ("done", "error"):
            break
        time.sleep(0.2)

    assert status["status"] == "done"
    assert "result" in status
    files = status["result"]["files"]
    assert "txt" in files
    assert "srt" in files
    assert "vtt" in files
    mock_model.transcribe.assert_called()


@requires_ffmpeg
def test_batch_upload_multiple_files(flask_app, workspace, tmp_path):
    app, mock_model = flask_app
    client = app.test_client()
    wav1 = tmp_path / "a.wav"
    wav2 = tmp_path / "b.wav"
    make_wav(wav1, duration=1.5)
    make_wav(wav2, duration=1.5)

    with open(wav1, "rb") as f1, open(wav2, "rb") as f2:
        res = client.post(
            "/api/batch",
            data={
                "files": [(f1, "a.wav"), (f2, "b.wav")],
                "order": json.dumps([1, 0]),
            },
            content_type="multipart/form-data",
        )

    assert res.status_code == 200
    data = res.get_json()
    assert data["batch"] is True
    job_id = data["job_id"]

    status = None
    for _ in range(60):
        status = client.get(f"/api/status/{job_id}").get_json()
        if status.get("status") in ("done", "error"):
            break
        time.sleep(0.3)

    assert status["status"] == "done"
    assert len(status["items"]) == 2
    result = status["result"]
    assert "zip" in result["files"]
    assert "srt" in result["files"]
    assert "vtt" in result["files"]
    assert mock_model.transcribe.call_count >= 2


@requires_ffmpeg
def test_integration_webm_processing(flask_app, workspace, tmp_path):
    """Mandatory WEBM scenario: upload WEBM → extract audio → transcribe → outputs."""
    app, mock_model = flask_app
    client = app.test_client()
    webm = tmp_path / "recording.webm"
    make_webm(webm, duration=2.0)

    with open(webm, "rb") as f:
        res = client.post("/api/transcribe", data={"file": (f, "recording.webm")}, content_type="multipart/form-data")

    assert res.status_code == 200
    job_id = res.get_json()["job_id"]

    status = None
    for _ in range(40):
        status = client.get(f"/api/status/{job_id}").get_json()
        if status.get("status") in ("done", "error"):
            break
        time.sleep(0.25)

    assert status["status"] == "done", status.get("message", status)
    assert status["result"]["text"] == "тестовая транскрипция"
    files = status["result"]["files"]
    for key in ("txt", "srt", "vtt", "json"):
        assert key in files

    import gigaam_ui.config as config
    for key in files.values():
        assert (config.OUTPUTS_DIR / key).exists()


def test_clear_workspace_blocked_during_job(flask_app):
    app, _ = flask_app
    client = app.test_client()

    import app as app_module
    app_module.job_store.set_job("x", status="running")

    res = client.post("/api/clear", json={"target": "workspace"})
    assert res.status_code == 409


def test_unsupported_extension_rejected(flask_app, tmp_path):
    app, _ = flask_app
    client = app.test_client()
    bad = tmp_path / "file.xyz"
    bad.write_text("data")
    with open(bad, "rb") as f:
        res = client.post("/api/transcribe", data={"file": (f, "file.xyz")}, content_type="multipart/form-data")
    assert res.status_code == 400
