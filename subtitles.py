"""Subtitle serialization. Segment times are relative to the start of the recording."""
import math


def srt_time(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid subtitle timestamp")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def render_srt(segments):
    blocks = []
    for segment in segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if end < start:
            raise ValueError("Subtitle ends before it starts")
        blocks.append(f"{len(blocks)+1}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")
    return "".join(blocks)


def result_segments(result, offset, duration):
    """Keep model timestamps when available, otherwise use the audio chunk bounds."""
    raw = getattr(result, "segments", None)
    if raw:
        return [{"start": offset + max(0, min(duration, float(seg.start))),
                 "end": offset + max(0, min(duration, float(seg.end))),
                 "text": str(seg.text)} for seg in raw]
    text = getattr(result, "text", None)
    if text is None:
        text = str(result)
    return [{"start": offset, "end": offset + duration, "text": text.strip()}]
