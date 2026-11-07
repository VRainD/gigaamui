"""API contract tests with lightweight model/audio stubs; no model download."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        os.environ["GIGAAM_DATA_ROOT"] = self.directory.name
        stubs = {name: types.ModuleType(name) for name in ("torch", "gigaam", "soundfile")}
        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location("api_under_test", Path(__file__).with_name("app.py"))
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.client = self.module.app.test_client()
        self.module.get_model = lambda **kwargs: types.SimpleNamespace(transcribe=lambda path: "Текст встречи")
        self.module.sf.info = lambda path: types.SimpleNamespace(frames=32000, samplerate=16000)
        def normalize(path):
            folder = tempfile.mkdtemp(dir=self.directory.name)
            out = Path(folder) / "audio.wav"
            out.write_bytes(b"stub")
            return str(out)
        self.module.ffmpeg_to_wav_mono_16k = normalize

    def tearDown(self):
        self.directory.cleanup()

    def test_upload_status_and_primary_srt_download(self):
        response = self.client.post("/api/transcribe", data={"file": (io.BytesIO(b"audio"), "meeting.mp3")})
        self.assertEqual(response.status_code, 200)
        job_id = response.json["job_id"]
        for _ in range(100):
            job = self.client.get("/api/status/" + job_id).json
            if job["status"] in ("done", "error"):
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["primary_format"], "srt")
        response = self.client.get("/download/" + job["result"]["files"]["srt"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "1\n00:00:00,000 --> 00:00:02,000\nТекст встречи\n\n")
        response.close()
        other = self.module.OUTPUTS_DIR / "other-meeting.srt"
        other.write_text("keep")
        self.assertEqual(self.client.delete("/api/jobs/" + job_id).status_code, 200)
        self.assertEqual(self.client.get("/api/status/" + job_id).status_code, 404)
        self.assertFalse(list(self.module.INPUTS_DIR.iterdir()))
        self.assertEqual(list(self.module.OUTPUTS_DIR.iterdir()), [other])
        self.assertEqual(self.client.delete("/api/jobs/" + job_id).status_code, 200)

    def test_active_job_cannot_be_cleaned(self):
        self.module.set_job("active", status="running")
        self.assertEqual(self.client.delete("/api/jobs/active").status_code, 409)
        self.assertIn("active", self.module.jobs)

    def test_missing_file_and_unknown_job(self):
        self.assertEqual(self.client.post("/api/transcribe").status_code, 400)
        self.assertEqual(self.client.get("/api/status/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
