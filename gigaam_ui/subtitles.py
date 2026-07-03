import json
import zipfile
from datetime import datetime
from pathlib import Path

from gigaam_ui import config


def srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def vtt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def format_srt(segments: list, start_index: int = 1, time_offset: float = 0.0) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=start_index):
        start = float(seg["start"]) + time_offset
        end = float(seg["end"]) + time_offset
        blocks.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{seg['text']}\n")
    return "\n".join(blocks)


def format_vtt(segments: list, time_offset: float = 0.0) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg["start"]) + time_offset
        end = float(seg["end"]) + time_offset
        lines.append(f"{vtt_time(start)} --> {vtt_time(end)}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def merge_segments(file_results: list) -> tuple[str, list]:
    """Merge per-file segments with global time offsets."""
    combined_text_parts = []
    combined_segments = []
    offset = 0.0
    for item in file_results:
        text = item.get("text", "")
        segments = item.get("segments", [])
        if text:
            combined_text_parts.append(text)
        for seg in segments:
            combined_segments.append({
                "start": round(float(seg["start"]) + offset, 3),
                "end": round(float(seg["end"]) + offset, 3),
                "text": seg["text"],
                "source": item.get("filename"),
            })
        if segments:
            offset = combined_segments[-1]["end"]
        elif text:
            offset += max(len(text.split()) * 0.3, 1.0)
    return "\n\n".join(combined_text_parts), combined_segments


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_individual_outputs(stem: str, text: str, segments: list) -> dict:
    config.ensure_dirs()
    base = f"{Path(stem).stem}_{_timestamp()}"
    txt_path = config.OUTPUTS_DIR / f"{base}.txt"
    json_path = config.OUTPUTS_DIR / f"{base}.json"
    srt_path = config.OUTPUTS_DIR / f"{base}.srt"
    vtt_path = config.OUTPUTS_DIR / f"{base}.vtt"

    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps({"text": text, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    srt_path.write_text(format_srt(segments), encoding="utf-8")
    vtt_path.write_text(format_vtt(segments), encoding="utf-8")

    return {
        "txt": txt_path.name,
        "json": json_path.name,
        "srt": srt_path.name,
        "vtt": vtt_path.name,
    }


def save_combined_outputs(batch_stem: str, text: str, segments: list) -> dict:
    config.ensure_dirs()
    base = f"{Path(batch_stem).stem}_combined_{_timestamp()}"
    txt_path = config.OUTPUTS_DIR / f"{base}.txt"
    srt_path = config.OUTPUTS_DIR / f"{base}.srt"
    vtt_path = config.OUTPUTS_DIR / f"{base}.vtt"

    txt_path.write_text(text, encoding="utf-8")
    srt_path.write_text(format_srt(segments), encoding="utf-8")
    vtt_path.write_text(format_vtt(segments), encoding="utf-8")

    return {
        "txt": txt_path.name,
        "srt": srt_path.name,
        "vtt": vtt_path.name,
    }


def create_results_zip(batch_stem: str, file_entries: list, combined_files: dict) -> str:
    config.ensure_dirs()
    zip_name = f"{Path(batch_stem).stem}_all_{_timestamp()}.zip"
    zip_path = config.OUTPUTS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in file_entries:
            files = entry.get("files") or {}
            prefix = Path(entry.get("filename", "file")).stem
            for kind, name in files.items():
                src = config.OUTPUTS_DIR / name
                if src.is_file():
                    zf.write(src, arcname=f"individual/{prefix}/{kind}.{src.suffix.lstrip('.')}")
        for kind, name in combined_files.items():
            src = config.OUTPUTS_DIR / name
            if src.is_file():
                zf.write(src, arcname=f"combined/{kind}.{src.suffix.lstrip('.')}")
    return zip_name
