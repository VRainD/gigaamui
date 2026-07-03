import math
import tempfile
import threading
import traceback
from pathlib import Path

import soundfile as sf
import torch

from gigaam_ui.audio import cleanup_normalized_wav, ffmpeg_to_wav_mono_16k
from gigaam_ui.config import CHUNK_SECONDS, MODEL_NAME
from gigaam_ui.subtitles import save_individual_outputs

_model = None
_model_lock = threading.Lock()


def get_model(job_store=None, job_id=None):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if job_store and job_id:
                    job_store.set_job(
                        job_id,
                        stage="model_loading",
                        progress=5,
                        message=f"Loading model {MODEL_NAME}...",
                    )
                import gigaam
                _model = gigaam.load_model(MODEL_NAME)
                if torch.cuda.is_available():
                    try:
                        _model = _model.cuda()
                    except Exception:
                        pass
                if job_store and job_id:
                    job_store.set_job(
                        job_id,
                        stage="model_loaded",
                        progress=15,
                        message="Model loaded",
                    )
    return _model


def _parse_transcribe_result(result, duration_sec: float) -> tuple[str, list]:
    text = getattr(result, "text", None) or str(result)
    segments = []
    if hasattr(result, "segments") and result.segments:
        for seg in result.segments:
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": str(seg.text),
            })
    else:
        segments = [{"start": 0.0, "end": duration_sec, "text": text}]
    return text, segments


def transcribe_audio(model, normalized: str, duration_sec: float, on_progress=None) -> tuple[str, list]:
    if duration_sec <= 25:
        if on_progress:
            on_progress(stage="transcribing", progress=40, message="Transcribing short audio...")
        result = model.transcribe(normalized)
        return _parse_transcribe_result(result, duration_sec)

    if on_progress:
        on_progress(stage="chunking", progress=20, message="Splitting long audio into chunks...")
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
        if on_progress:
            on_progress(
                stage="transcribing",
                progress=pct,
                message=f"Transcribing chunk {idx + 1}/{total_chunks}...",
            )
        chunk_result = model.transcribe(tmp_chunk.name)
        chunk_text = getattr(chunk_result, "text", None) or str(chunk_result)
        texts.append(chunk_text.strip())
        chunk_duration = len(chunk) / sr
        segments.append({
            "start": round(current_time, 3),
            "end": round(current_time + chunk_duration, 3),
            "text": chunk_text.strip(),
        })
        current_time += chunk_duration
        Path(tmp_chunk.name).unlink(missing_ok=True)
    return "\n".join(t for t in texts if t), segments


def transcribe_path(
    job_store,
    job_id: str,
    src_path: str,
    filename: str | None = None,
    batch_id: str | None = None,
    item_index: int | None = None,
) -> dict | None:
    normalized = None
    display_name = filename or Path(src_path).name
    try:
        job_store.set_job(
            job_id,
            status="running",
            stage="preparing",
            progress=1,
            message="Preparing audio...",
            filename=display_name,
            batch_id=batch_id,
        )
        if batch_id is not None and item_index is not None:
            job_store.update_item(
                batch_id, item_index,
                status="running", stage="preparing", progress=1,
                message="Preparing audio...", job_id=job_id,
            )

        normalized = ffmpeg_to_wav_mono_16k(src_path)
        info = sf.info(normalized)
        duration_sec = float(info.frames) / float(info.samplerate)
        job_store.set_job(
            job_id,
            duration_sec=duration_sec,
            message=f"Audio prepared, duration {duration_sec:.1f}s",
        )

        def on_progress(**kwargs):
            job_store.set_job(job_id, **kwargs)
            if batch_id is not None and item_index is not None:
                job_store.update_item(batch_id, item_index, **kwargs)

        model = get_model(job_store, job_id)
        text, segments = transcribe_audio(model, normalized, duration_sec, on_progress)
        files = save_individual_outputs(display_name, text, segments)
        result = {"text": text, "segments": segments, "files": files, "filename": display_name}
        job_store.set_job(
            job_id,
            status="done",
            stage="finished",
            progress=100,
            message="Done",
            result=result,
        )
        if batch_id is not None and item_index is not None:
            job_store.update_item(
                batch_id, item_index,
                status="done", stage="finished", progress=100,
                message="Done", result=result, files=files,
            )
        return result
    except Exception as e:
        job_store.set_job(
            job_id,
            status="error",
            stage="failed",
            progress=100,
            message=str(e),
            traceback=traceback.format_exc(),
        )
        if batch_id is not None and item_index is not None:
            job_store.update_item(
                batch_id, item_index,
                status="error", stage="failed", progress=100,
                message=str(e), job_id=job_id,
            )
        return None
    finally:
        cleanup_normalized_wav(normalized)
