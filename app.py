import threading
import uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from gigaam_ui import config
from gigaam_ui import __version__
from gigaam_ui.audio import is_allowed_extension
from gigaam_ui.cleanup import CleanupInProgressError, clear_target
from gigaam_ui.jobs import JobStore
from gigaam_ui.subtitles import create_results_zip, merge_segments, save_combined_outputs
from gigaam_ui.transcription import transcribe_path

config.ensure_dirs()

job_store = JobStore(
    ttl_seconds=config.JOB_TTL_SECONDS,
    ttl_check_interval=config.JOB_TTL_CHECK_INTERVAL,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_MB * 1024 * 1024


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA GPU"


def _run_batch(batch_id: str, file_paths: list[tuple[str, str]]):
    """Process files sequentially in the given order."""
    results = []
    total = len(file_paths)
    for idx, (child_id, path_str) in enumerate(file_paths):
        filename = Path(path_str).name
        if "_" in filename and len(filename.split("_", 1)[0]) == 36:
            display_name = filename.split("_", 1)[1]
        else:
            display_name = filename

        job_store.set_job(
            batch_id,
            message=f"Processing {idx + 1}/{total}: {display_name}",
            progress=int((idx / total) * 100),
        )
        job_store.update_item(
            batch_id, idx,
            status="queued", stage="queued", progress=0,
            message="Waiting...", filename=display_name, job_id=child_id,
        )
        result = transcribe_path(
            job_store, child_id, path_str,
            filename=display_name,
            batch_id=batch_id,
            item_index=idx,
        )
        if result:
            results.append(result)

    if len(results) > 1:
        combined_text, combined_segments = merge_segments(results)
        batch_stem = f"batch_{batch_id[:8]}"
        combined_files = save_combined_outputs(batch_stem, combined_text, combined_segments)
        zip_name = create_results_zip(batch_stem, results, combined_files)
        batch_result = {
            "text": combined_text,
            "segments": combined_segments,
            "files": {**combined_files, "zip": zip_name},
            "items": results,
        }
    elif len(results) == 1:
        batch_result = {
            "text": results[0]["text"],
            "segments": results[0]["segments"],
            "files": results[0]["files"],
            "items": results,
        }
    else:
        batch_result = {"text": "", "segments": [], "files": {}, "items": []}

    job_store.set_job(
        batch_id,
        status="done",
        stage="finished",
        progress=100,
        message="Batch done",
        result=batch_result,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    inputs = sorted(p.name for p in config.INPUTS_DIR.iterdir() if p.is_file())
    return jsonify({
        "version": __version__,
        "model_name": config.MODEL_NAME,
        "chunk_seconds": config.CHUNK_SECONDS,
        "max_file_mb": config.MAX_FILE_MB,
        "job_ttl_seconds": config.JOB_TTL_SECONDS,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": _gpu_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "active_jobs": job_store.has_active_jobs(),
        "inputs": inputs,
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Single-file upload (backward compatible)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not is_allowed_extension(file.filename):
        return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    job_id = str(uuid.uuid4())
    safe_name = secure_filename(Path(file.filename).name) or "upload"
    dst = config.INPUTS_DIR / f"{job_id}_{safe_name}"
    file.save(dst)
    job_store.set_job(
        job_id, status="queued", stage="queued", progress=0,
        message="Queued", filename=file.filename,
    )
    threading.Thread(
        target=transcribe_path,
        args=(job_store, job_id, str(dst)),
        kwargs={"filename": file.filename},
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/batch", methods=["POST"])
def batch():
    """
    Batch processing:
    - multipart `files` — upload and process in order (optional `order` JSON array of indices)
    - no files — legacy: process existing files in data/inputs
    """
    uploaded = request.files.getlist("files")
    if not uploaded or (len(uploaded) == 1 and not uploaded[0].filename):
        uploaded = request.files.getlist("file")
    uploaded = [f for f in uploaded if f.filename]

    if not uploaded:
        files = sorted(p for p in config.INPUTS_DIR.iterdir() if p.is_file())
        if not files:
            return jsonify({"error": "No files uploaded and no files in data/inputs"}), 400
        file_paths = [(str(uuid.uuid4()), str(p)) for p in files]
        filenames = [p.name for p in files]
    else:
        order_raw = request.form.get("order")
        if order_raw:
            import json
            try:
                order = json.loads(order_raw)
                uploaded = [uploaded[i] for i in order if 0 <= i < len(uploaded)]
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

        file_paths = []
        filenames = []
        for f in uploaded:
            if not is_allowed_extension(f.filename):
                return jsonify({"error": f"Unsupported file type: {f.filename}"}), 400
            child_id = str(uuid.uuid4())
            safe_name = secure_filename(Path(f.filename).name) or "upload"
            dst = config.INPUTS_DIR / f"{child_id}_{safe_name}"
            f.save(dst)
            file_paths.append((child_id, str(dst)))
            filenames.append(f.filename)

    batch_id = str(uuid.uuid4())
    items = [
        {"index": i, "filename": filenames[i], "status": "queued", "progress": 0, "message": "Queued"}
        for i in range(len(file_paths))
    ]
    job_store.set_job(
        batch_id, status="running", stage="batch", progress=0,
        message="Starting batch", batch=True, items=items,
    )
    threading.Thread(target=_run_batch, args=(batch_id, file_paths), daemon=True).start()
    return jsonify({"job_id": batch_id, "batch": True, "file_count": len(file_paths)})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = job_store.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(config.OUTPUTS_DIR, filename, as_attachment=True)


@app.route("/api/clear", methods=["POST"])
def clear():
    data = request.get_json(silent=True) or {}
    target = data.get("target", "workspace")
    try:
        result = clear_target(target, job_store=job_store)
        return jsonify(result)
    except CleanupInProgressError as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
