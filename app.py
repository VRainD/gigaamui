import json
import gc
import math
import os
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
import soundfile as sf
import torch
import gigaam
from subtitles import render_srt, result_segments

BASE_DIR = Path(os.getenv("GIGAAM_DATA_ROOT", "/app"))
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOGS_DIR = DATA_DIR / "logs"
for p in [INPUTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "20"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "5120"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
jobs = {}
job_inputs = {}


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def set_job(job_id, **kwargs):
    job = jobs.setdefault(job_id, {})
    job.update(kwargs)
    job["updated_at"] = now_iso()


def safe_unlink(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def ffmpeg_to_wav_mono_16k(src_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="gigaam_")
    out_path = os.path.join(tmp_dir, "normalized.wav")
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000", "-vn", out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def get_model(job_id=None):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if job_id:
                    set_job(job_id, stage="model_loading", progress=5, message=f"Loading model {MODEL_NAME}...")
                model_cache = Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "gigaam"
                model_cache.mkdir(parents=True, exist_ok=True)
                _model = gigaam.load_model(MODEL_NAME, download_root=str(model_cache))
                if torch.cuda.is_available():
                    try:
                        _model = _model.cuda()
                    except Exception:
                        pass
                if job_id:
                    set_job(job_id, stage="model_loaded", progress=15, message="Model loaded")
    return _model


def save_outputs(stem: str, text: str, segments: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{Path(stem).stem}_{ts}"
    txt_path = OUTPUTS_DIR / f"{base}.txt"
    json_path = OUTPUTS_DIR / f"{base}.json"
    srt_path = OUTPUTS_DIR / f"{base}.srt"

    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps({"text": text, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")

    srt_path.write_text(render_srt(segments), encoding="utf-8")
    return txt_path.name, json_path.name, srt_path.name


def reset_recognition_cache():
    """Caller holds _inference_lock; release all state from the previous model."""
    global _model
    with _model_lock:
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.route("/api/clear-recognition-cache", methods=["POST"])
def clear_recognition_cache():
    with _inference_lock:
        reset_recognition_cache()
    return jsonify({"cleared": True})


def transcribe_path(job_id: str, src_path: str):
    # The shared model must never run concurrent inference (UI, API and batch).
    with _inference_lock:
        _transcribe_path(job_id, src_path)


def _transcribe_path(job_id: str, src_path: str):
    normalized = None
    try:
        set_job(job_id, status="running", stage="preparing", progress=1, message="Preparing audio...")
        normalized = ffmpeg_to_wav_mono_16k(src_path)
        info = sf.info(normalized)
        duration_sec = float(info.frames) / float(info.samplerate)
        set_job(job_id, duration_sec=duration_sec, message=f"Audio prepared, duration {duration_sec:.1f}s")

        set_job(job_id, stage="clearing_cache", message="Clearing previous recognition state...")
        reset_recognition_cache()
        model = get_model(job_id=job_id)

        # <=25 sec use direct official transcribe path; longer audio uses chunking workaround.
        if duration_sec <= 25:
            set_job(job_id, stage="transcribing", progress=40, message="Transcribing short audio...")
            result = model.transcribe(normalized)
            segments = result_segments(result, 0.0, duration_sec)
            text = "\n".join(seg["text"] for seg in segments if seg["text"].strip())
        else:
            set_job(job_id, stage="chunking", progress=20, message="Splitting long audio into chunks...")
            audio, sr = sf.read(normalized, always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio[:, 0]
            chunk_size = CHUNK_SECONDS * sr
            total_chunks = math.ceil(len(audio) / chunk_size)
            texts = []
            segments = []
            current_time = 0.0
            for idx in range(total_chunks):
                start = idx * chunk_size
                end = min((idx + 1) * chunk_size, len(audio))
                chunk = audio[start:end]
                tmp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_chunk.close()
                sf.write(tmp_chunk.name, chunk, sr)
                pct = 20 + int(((idx + 1) / total_chunks) * 70)
                set_job(job_id, stage="transcribing", progress=pct, message=f"Transcribing chunk {idx+1}/{total_chunks}...")
                try:
                    chunk_result = model.transcribe(tmp_chunk.name)
                    chunk_duration = len(chunk) / sr
                    chunk_segments = result_segments(chunk_result, current_time, chunk_duration)
                    segments.extend(chunk_segments)
                    texts.extend(seg["text"].strip() for seg in chunk_segments)
                finally:
                    safe_unlink(Path(tmp_chunk.name))
                current_time += chunk_duration
            text = "\n".join([t for t in texts if t])

        txt_name, json_name, srt_name = save_outputs(job_id, text, segments)
        set_job(
            job_id,
            status="done",
            stage="finished",
            progress=100,
            message="Done",
            result={
                "text": text,
                "segments": segments,
                "primary_format": "srt",
                "timing": "model_segments_or_chunk_bounds",
                "files": {"srt": srt_name, "txt": txt_name, "json": json_name},
            },
        )
    except Exception as e:
        set_job(job_id, status="error", stage="failed", progress=100, message=str(e), traceback=traceback.format_exc())
    finally:
        if normalized and os.path.exists(normalized):
            shutil.rmtree(os.path.dirname(normalized), ignore_errors=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    gpu_name = "CPU"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA GPU"
    inputs = sorted([p.name for p in INPUTS_DIR.iterdir() if p.is_file()])
    return jsonify({
        "model_name": MODEL_NAME,
        "primary_format": "srt",
        "api_version": 1,
        "chunk_seconds": CHUNK_SECONDS,
        "max_file_mb": MAX_FILE_MB,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "inputs": inputs,
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    job_id = str(uuid.uuid4())
    dst = INPUTS_DIR / f"{job_id}_{Path(file.filename).name}"
    file.save(dst)
    job_inputs[job_id] = dst
    set_job(job_id, status="queued", stage="queued", progress=0, message="Queued", filename=file.filename)
    threading.Thread(target=transcribe_path, args=(job_id, str(dst)), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def cleanup_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"deleted": True})
    if job.get("status") not in ("done", "error"):
        return jsonify({"error": "Job is still active"}), 409
    # Delete only artifacts owned by this job, never shared cache or other meetings.
    for filename in job.get("result", {}).get("files", {}).values():
        path = OUTPUTS_DIR / filename
        if path.parent.resolve() == OUTPUTS_DIR.resolve():
            path.unlink(missing_ok=True)
    source = job_inputs.get(job_id)
    if source is not None:
        source.unlink(missing_ok=True)
    job_inputs.pop(job_id, None)
    jobs.pop(job_id, None)
    return jsonify({"deleted": True})


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)


@app.route("/api/batch", methods=["POST"])
def batch():
    files = sorted([p for p in INPUTS_DIR.iterdir() if p.is_file()])
    if not files:
        return jsonify({"error": "No files in data/inputs"}), 400
    batch_id = str(uuid.uuid4())
    set_job(batch_id, status="running", stage="batch", progress=0, message="Starting batch", batch=True, items=[])

    def run_batch():
        items = []
        for idx, path in enumerate(files, start=1):
            child_id = str(uuid.uuid4())
            set_job(batch_id, message=f"Processing {idx}/{len(files)}: {path.name}", progress=int(((idx-1)/len(files))*100))
            transcribe_path(child_id, str(path))
            items.append({"file": path.name, "job_id": child_id, "status": jobs[child_id].get("status")})
            set_job(batch_id, items=items, progress=int((idx/len(files))*100))
        set_job(batch_id, status="done", stage="finished", progress=100, message="Batch done", items=items)

    threading.Thread(target=run_batch, daemon=True).start()
    return jsonify({"job_id": batch_id})


@app.route("/api/clear", methods=["POST"])
def clear():
    target = request.json.get("target", "outputs")
    mapping = {
        "outputs": OUTPUTS_DIR,
        "logs": LOGS_DIR,
        "cache": Path("/root/.cache/huggingface"),
    }
    path = mapping.get(target)
    if path is None:
        return jsonify({"error": "Invalid target"}), 400
    count = 0
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
            count += 1
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            count += 1
    return jsonify({"target": target, "removed": count})


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
