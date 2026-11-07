import unittest
from types import SimpleNamespace
from subtitles import render_srt, result_segments, srt_time


class SubtitleTests(unittest.TestCase):
    def test_offsets_for_later_audio_chunk(self):
        result = SimpleNamespace(segments=[SimpleNamespace(start=1.25, end=3.5, text="Решение встречи")])
        segments = result_segments(result, 20, 10)
        self.assertEqual(render_srt(segments), "1\n00:00:21,250 --> 00:00:23,500\nРешение встречи\n\n")

    def test_plain_text_uses_chunk_bounds_and_skips_silence(self):
        self.assertEqual(render_srt(result_segments("", 0, 20)), "")
        self.assertIn("00:00:20,000 --> 00:00:25,000", render_srt(result_segments("Конец", 20, 5)))

    def test_empty_model_text_is_not_object_representation(self):
        result = SimpleNamespace(text="", segments=[])
        self.assertEqual(render_srt(result_segments(result, 0, 10)), "")

    def test_hours_and_millisecond_rollover(self):
        self.assertEqual(srt_time(3661.9996), "01:01:02,000")

    def test_invalid_bounds_fail(self):
        with self.assertRaises(ValueError):
            render_srt([{"start": 2, "end": 1, "text": "bad"}])


if __name__ == "__main__":
    unittest.main()
