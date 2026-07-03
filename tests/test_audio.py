import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from gigaam_ui.audio import ffmpeg_to_wav_mono_16k, is_allowed_extension, is_video_file
from gigaam_ui.config import TEMP_DIR, ensure_dirs
from tests.conftest import make_webm, requires_ffmpeg


def test_allowed_extensions():
    assert is_allowed_extension("audio.mp3")
    assert is_allowed_extension("video.WEBM")
    assert is_allowed_extension("clip.MP4")
    assert not is_allowed_extension("document.pdf")


def test_is_video_file():
    assert is_video_file("a.webm")
    assert is_video_file("b.mkv")
    assert not is_video_file("c.wav")


@requires_ffmpeg
def test_ffmpeg_to_wav_mono_16k_from_wav(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    import gigaam_ui.audio as audio_mod
    importlib.reload(config)
    importlib.reload(audio_mod)
    from gigaam_ui.config import ensure_dirs
    ensure_dirs()
    src = tmp_path / "src.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=1", str(src)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out = audio_mod.ffmpeg_to_wav_mono_16k(str(src))
    info = sf.info(out)
    assert info.samplerate == 16000
    assert info.channels == 1


@requires_ffmpeg
def test_ffmpeg_to_wav_mono_16k_from_webm(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    import gigaam_ui.audio as audio_mod
    importlib.reload(config)
    importlib.reload(audio_mod)
    from gigaam_ui.config import ensure_dirs
    ensure_dirs()
    src = tmp_path / "test.webm"
    make_webm(src, duration=1.5)
    out = audio_mod.ffmpeg_to_wav_mono_16k(str(src))
    info = sf.info(out)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.duration > 0
