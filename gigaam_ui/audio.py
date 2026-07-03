import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from gigaam_ui import config


def is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in config.ALLOWED_EXTENSIONS


def is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in config.VIDEO_EXTENSIONS


def ffmpeg_to_wav_mono_16k(src_path: str, dest_dir: Path | None = None) -> str:
    """Convert audio or video input to WAV, mono, PCM s16le, 16 kHz."""
    config.ensure_dirs()
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="gigaam_", dir=config.TEMP_DIR))
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "normalized.wav"
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", "-f", "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(out_path)


def cleanup_normalized_wav(normalized_path: str | None):
    if not normalized_path:
        return
    parent = Path(normalized_path).parent
    if parent.exists() and parent.name.startswith("gigaam_"):
        shutil.rmtree(parent, ignore_errors=True)
    elif Path(normalized_path).exists():
        Path(normalized_path).unlink(missing_ok=True)
