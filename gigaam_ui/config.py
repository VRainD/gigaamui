import os
from pathlib import Path

BASE_DIR = Path(os.getenv("GIGAAM_BASE_DIR", "/app"))
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOGS_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "temp"

MODEL_NAME = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "20"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "1000"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
JOB_TTL_CHECK_INTERVAL = int(os.getenv("JOB_TTL_CHECK_INTERVAL", "60"))
HF_CACHE_DIR = Path(os.getenv("HF_HOME", "/root/.cache/huggingface"))

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp2", ".amr",
}
VIDEO_EXTENSIONS = {
    ".webm", ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp", ".m4v",
}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

WORKSPACE_DIRS = (INPUTS_DIR, OUTPUTS_DIR, LOGS_DIR, TEMP_DIR)


def ensure_dirs():
    for path in WORKSPACE_DIRS:
        path.mkdir(parents=True, exist_ok=True)
