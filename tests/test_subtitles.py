from gigaam_ui.subtitles import format_srt, format_vtt, merge_segments, save_combined_outputs, save_individual_outputs


def test_format_srt_with_offset():
    segments = [{"start": 0.0, "end": 2.5, "text": "Привет"}]
    srt = format_srt(segments, start_index=3, time_offset=10.0)
    assert srt.startswith("3\n")
    assert "00:00:10,000" in srt
    assert "00:00:12,500" in srt
    assert "Привет" in srt


def test_format_vtt():
    segments = [{"start": 1.0, "end": 3.0, "text": "Тест"}]
    vtt = format_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000" in vtt
    assert "Тест" in vtt


def test_merge_segments_global_offset():
    file_results = [
        {
            "filename": "a.webm",
            "text": "Первый",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Первый"}],
        },
        {
            "filename": "b.mp3",
            "text": "Второй",
            "segments": [{"start": 0.0, "end": 3.0, "text": "Второй"}],
        },
    ]
    text, segments = merge_segments(file_results)
    assert "Первый" in text
    assert "Второй" in text
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 5.0
    assert segments[1]["start"] == 5.0
    assert segments[1]["end"] == 8.0


def test_sequential_srt_numbering(workspace, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    import gigaam_ui.subtitles as subtitles_mod
    importlib.reload(config)
    importlib.reload(subtitles_mod)

    segments = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 0.0, "end": 2.0, "text": "B"},
    ]
    merged_text, merged = subtitles_mod.merge_segments([
        {"text": "A", "segments": [segments[0]]},
        {"text": "B", "segments": [segments[1]]},
    ])
    files = subtitles_mod.save_combined_outputs("batch_test", merged_text, merged)
    srt_content = (config.OUTPUTS_DIR / files["srt"]).read_text(encoding="utf-8")
    assert srt_content.index("1\n") < srt_content.index("2\n")
    assert "00:00:01,000" in srt_content


def test_individual_outputs_include_vtt(workspace, monkeypatch):
    monkeypatch.setenv("GIGAAM_BASE_DIR", str(workspace))
    import importlib
    import gigaam_ui.config as config
    import gigaam_ui.subtitles as subtitles_mod
    importlib.reload(config)
    importlib.reload(subtitles_mod)

    files = subtitles_mod.save_individual_outputs("clip.webm", "текст", [{"start": 0, "end": 1, "text": "текст"}])
    assert (config.OUTPUTS_DIR / files["vtt"]).exists()
    assert (config.OUTPUTS_DIR / files["srt"]).exists()
    assert (config.OUTPUTS_DIR / files["txt"]).exists()
