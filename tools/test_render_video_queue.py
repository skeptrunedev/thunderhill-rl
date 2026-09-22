import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from video_jobs import enqueue_video
from render_video_queue import process_queue


class VideoQueueRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "episode.jsonl"
        self.source.write_text(
            json.dumps(
                {
                    "type": "episode",
                    "episode_id": "test",
                    "policy_display": {
                        "model_name": "test",
                        "generation": 3,
                        "rollout_number": 2,
                        "rollout_count": 4,
                    },
                }
            )
            + "\n"
        )
        self.job = enqueue_video(
            self.root, self.source, metadata={"stop_reason": "invalid_model_action"}
        )

    def run_queue(self, **kwargs):
        return process_queue(self.job.parent, godot="godot", ffmpeg="ffmpeg", **kwargs)

    def render(self, source, output, **kwargs):
        output.write_bytes(b"verified video fixture")
        return {"complete": True, "frames": 30}

    def test_lock_contention_waits_only_when_requested(self):
        with patch("render_video_queue.fcntl.flock", side_effect=[BlockingIOError(), None]) as lock, patch(
            "render_video_queue.time.sleep"
        ) as sleep, patch("render_video_queue.render_video", side_effect=self.render):
            self.assertTrue(self.run_queue(wait_for_lock=True)["complete"])
            self.assertEqual(lock.call_count, 2)
            sleep.assert_called_once_with(1)
        with patch("render_video_queue.fcntl.flock", side_effect=BlockingIOError()), patch(
            "render_video_queue.time.sleep"
        ) as sleep:
            with self.assertRaises(BlockingIOError):
                self.run_queue()
            sleep.assert_not_called()

    def test_waiting_does_not_hide_other_lock_errors(self):
        with patch("render_video_queue.fcntl.flock", side_effect=PermissionError("denied")), patch(
            "render_video_queue.time.sleep"
        ) as sleep:
            with self.assertRaises(PermissionError):
                self.run_queue(wait_for_lock=True)
            sleep.assert_not_called()

    def test_all_outcomes_indexed_and_completed_not_rendered_twice(self):
        with patch(
            "render_video_queue.render_video", side_effect=self.render
        ) as renderer:
            report = self.run_queue()
            self.assertTrue(report["complete"])
            self.assertEqual(report["queued"], 1)
            self.assertEqual(
                report["videos"][0]["metadata"]["stop_reason"], "invalid_model_action"
            )
            self.assertEqual(report["videos"][0]["policy_display"]["rollout_number"], 2)
            self.run_queue()
            self.assertEqual(renderer.call_count, 1)

    def test_failures_preserved_and_explicit_retry(self):
        with patch(
            "render_video_queue.render_video",
            side_effect=RuntimeError("render failure"),
        ) as renderer:
            self.assertFalse(self.run_queue()["complete"])
            self.assertFalse(self.run_queue()["complete"])
            self.assertEqual(renderer.call_count, 1)
        with patch("render_video_queue.render_video", side_effect=self.render):
            self.assertTrue(self.run_queue(retry_failed=True)["complete"])
        self.assertEqual(len(list((self.root / "videos").rglob("attempt-*"))), 2)
        self.assertEqual(len(list((self.root / "videos").rglob("error.json"))), 1)

    def test_changed_recording_rejected_before_render(self):
        self.source.write_text(self.source.read_text() + "{}\n")
        with patch("render_video_queue.render_video") as renderer:
            with self.assertRaisesRegex(ValueError, "Recording changed"):
                self.run_queue()
            renderer.assert_not_called()

    def test_corrupted_completed_video_rejected(self):
        with patch("render_video_queue.render_video", side_effect=self.render):
            report = self.run_queue()
        (self.root / report["videos"][0]["video"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            self.run_queue()

    def test_external_source_rejected(self):
        job = json.loads(self.job.read_text())
        job["source"] = "../outside.jsonl"
        self.job.write_text(json.dumps(job))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.run_queue()

    def test_unqueued_interrupted_episode_prevents_complete_archive(self):
        environment = self.root / "environment-0"
        environment.mkdir()
        header = json.loads(self.source.read_text())
        header["episode_id"] = "interrupted"
        (environment / "interrupted.jsonl").write_text(json.dumps(header) + "\n")
        with patch("render_video_queue.render_video", side_effect=self.render):
            report = self.run_queue()
        self.assertFalse(report["complete"])
        self.assertEqual(report["unqueued_episodes"], ["interrupted"])


if __name__ == "__main__":
    unittest.main()
