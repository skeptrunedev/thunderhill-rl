"""Exercise immutable episode identity and atomic queue publication."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from video_jobs import enqueue_video


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "run"
        self.out.mkdir()
        self.source = self.out / "episode.jsonl"
        self.header = {
            "type": "episode",
            "episode_id": "run-1",
            "policy_display": {"model_name": "gemma", "generation": 20},
        }
        self.metadata = {"reward": 12.5, "outcome": "horizon"}
        self.write()

    def write(self):
        self.source.write_text(json.dumps(self.header) + "\n")

    def enqueue(self):
        return enqueue_video(self.out, self.source, metadata=self.metadata)

    def test_portable_full_episode_and_idempotence(self):
        path = self.enqueue()
        data = json.loads(path.read_text())
        self.assertEqual(data["source"], "episode.jsonl")
        self.assertEqual(
            data["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest()
        )
        self.assertTrue(data["full_episode"])
        self.assertEqual(data["policy_display"], self.header["policy_display"])
        original = path.stat()
        self.assertEqual(self.enqueue(), path)
        self.assertEqual(original.st_ino, path.stat().st_ino)
        self.assertEqual(original.st_mtime_ns, path.stat().st_mtime_ns)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_same_episode_metadata_conflict(self):
        path = self.enqueue()
        original = path.read_bytes()
        self.metadata["reward"] = 20
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.enqueue()
        self.assertEqual(path.read_bytes(), original)

    def test_source_mutation_rejected(self):
        path = self.enqueue()
        original = path.read_bytes()
        with self.source.open("a") as stream:
            stream.write('{"type":"transition"}\n')
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.enqueue()
        self.assertEqual(path.read_bytes(), original)

    def test_duplicate_identity_in_different_source_rejected(self):
        self.enqueue()
        other = self.out / "other.jsonl"
        other.write_bytes(self.source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            enqueue_video(self.out, other, metadata=self.metadata)

    def test_missing_identity_and_non_episode_rejected(self):
        for header in (
            {"type": "transition"},
            {"type": "episode"},
            {"type": "episode", "episode_id": " "},
            [],
            None,
        ):
            self.source.write_text(json.dumps(header) + "\n")
            with self.assertRaises(ValueError):
                self.enqueue()
        self.assertFalse((self.out / "video_jobs").exists())

    def test_missing_display_and_partial_source_rejected(self):
        del self.header["policy_display"]
        self.write()
        with self.assertRaisesRegex(ValueError, "policy_display"):
            self.enqueue()
        self.header["policy_display"] = {}
        self.source.write_text(json.dumps(self.header))
        with self.assertRaisesRegex(ValueError, "partial"):
            self.enqueue()

    def test_outside_root_including_symlink_rejected(self):
        external = self.root / "outside.jsonl"
        external.write_bytes(self.source.read_bytes())
        link = self.out / "link.jsonl"
        link.symlink_to(external)
        for source in (external, link):
            with self.assertRaisesRegex(ValueError, "inside"):
                enqueue_video(self.out, source, metadata=self.metadata)

    def test_concurrent_enqueue_is_idempotent(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: self.enqueue(), range(16)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(list(paths[0].parent.iterdir()), [paths[0]])

    def test_failure_does_not_publish_partial_job(self):
        with patch("video_jobs.os.link", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.enqueue()
        self.assertEqual(list((self.out / "video_jobs").iterdir()), [])

    def test_nonfinite_metadata_rejected(self):
        self.metadata["reward"] = float("nan")
        with self.assertRaises(ValueError):
            self.enqueue()
        self.assertFalse((self.out / "video_jobs").exists())


if __name__ == "__main__":
    unittest.main()
