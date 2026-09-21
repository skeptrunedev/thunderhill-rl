"""Placement migration invariants independent of the production track geometry."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import migrate_plan_placements as migration


class PlacementMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old = {
            "origin": {},
            "coordinates": "x east, z south, y up",
            "length_m": 30,
            "samples": [
                {"s": s, "source_s": s, "p": [s, 0, 0], "left": [0, 0, 1], "width": 10}
                for s in [0, 10, 20]
            ],
        }
        self.new = copy.deepcopy(self.old)
        self.new["length_m"] = 32
        self.new["samples"][1].update(s=12, p=[11, 0, 1])
        self.new["samples"][2].update(s=22, p=[21, 0, 0])
        self.old_path = self.root / "old.json"
        self.new_path = self.root / "new.json"
        self.old_path.write_bytes(migration.encoded(self.old))
        self.new_path.write_bytes(migration.encoded(self.new))
        old_hash = migration.digest(self.old_path.read_bytes())
        corridor = {"points_local_xz": [[10, 9], [20, 9]], "tint": [0.4, 0.5, 0.6]}
        review = {
            "track_sha256": old_hash,
            "construction": {
                "requested_stations_m": [9, 21],
                "offset_inside_left_pavement_edge_m": 4,
                "method": "Nearest sample",
            },
            "corridor": corridor,
        }
        curb_review = {"track_sha256": old_hash, "baseline": {"immutable_evidence": 42}}
        self.write(migration.FIELD_REVIEW, review)
        self.write(migration.CURB_REVIEW, curb_review)
        self.write(
            migration.CURBS,
            {
                "track_sha256": old_hash,
                "sample_count": 3,
                "source_reference": migration.CURB_REVIEW,
                "source_reference_sha256": migration.digest(
                    migration.encoded(curb_review)
                ),
                "runs": [{"segments": [0, 2], "side": 1}],
                "width_m": 0.9,
            },
        )
        self.write(
            migration.FIELDS,
            {
                "track_sha256": old_hash,
                "corridors": [
                    {"points_local_xz": [[100, 101], [103, 104]], "feather": 1.2},
                    corridor,
                ],
                "source_references": [
                    {
                        "path": migration.FIELD_REVIEW,
                        "sha256": migration.digest(migration.encoded(review)),
                    }
                ],
            },
        )
        path = self.root / migration.SHADER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "// preserve me\nuniform vec2 paving_joint_station_m = vec2(5.0, 25.0);\n// and me\n"
        )

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(migration.encoded(value))

    def prepare(self):
        return migration.prepare(self.root, self.old_path, self.new_path)

    def test_preserves_authorship_and_maps_sample_identity(self):
        originals, outputs, report = self.prepare()
        self.assertTrue(
            all((self.root / p).read_bytes() == raw for p, raw in originals.items())
        )
        old_fields, fields = [
            json.loads(d[migration.FIELDS]) for d in (originals, outputs)
        ]
        self.assertEqual(fields["corridors"][0], old_fields["corridors"][0])
        self.assertEqual(fields["corridors"][1]["points_local_xz"], [[11, 10], [21, 9]])
        self.assertEqual(
            fields["corridors"][1]["tint"], old_fields["corridors"][1]["tint"]
        )
        old_curbs, curbs = [
            json.loads(d[migration.CURBS]) for d in (originals, outputs)
        ]
        self.assertEqual(curbs["runs"], old_curbs["runs"])
        self.assertEqual(curbs["width_m"], old_curbs["width_m"])
        review = json.loads(outputs[migration.CURB_REVIEW])
        review.pop("placement_migrations")
        self.assertEqual(review, json.loads(originals[migration.CURB_REVIEW]))
        field_review = json.loads(outputs[migration.FIELD_REVIEW])
        self.assertEqual(
            field_review["construction"]["selected_sample_indices"], [1, 2]
        )
        self.assertEqual(
            field_review["construction"]["requested_stations_m"], [10.8, 23]
        )
        self.assertEqual(
            field_review["original_placement"]["corridor"], old_fields["corridors"][1]
        )
        self.assertEqual(report["paving_joint_mapped_station_m"], [6, 27])
        self.assertEqual(
            outputs[migration.SHADER].decode(),
            "// preserve me\nuniform vec2 paving_joint_station_m = vec2(6.000000, 27.000000);\n// and me\n",
        )
        self.assertEqual(
            curbs["source_reference_sha256"],
            migration.digest(outputs[migration.CURB_REVIEW]),
        )
        self.assertEqual(
            fields["source_references"][0]["sha256"],
            migration.digest(outputs[migration.FIELD_REVIEW]),
        )
        migration.apply(self.root, originals, outputs)
        self.assertTrue(
            all((self.root / p).read_bytes() == raw for p, raw in outputs.items())
        )

    def test_rejects_count_identity_and_nonmonotone_station_changes(self):
        for kind in ("count", "identity", "station"):
            bad = copy.deepcopy(self.new)
            if kind == "count":
                bad["samples"].pop()
            elif kind == "identity":
                bad["samples"][1]["source_s"] = 11
            else:
                bad["samples"][1]["s"] = 23
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                migration.validate_tracks(self.old, bad)

    def test_rejects_unreproducible_field_selection(self):
        path = self.root / migration.FIELD_REVIEW
        review = json.loads(path.read_bytes())
        review["construction"]["requested_stations_m"][0] = 0
        self.write(migration.FIELD_REVIEW, review)
        fields = json.loads((self.root / migration.FIELDS).read_bytes())
        fields["source_references"][0]["sha256"] = migration.digest(path.read_bytes())
        self.write(migration.FIELDS, fields)
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            self.prepare()

    def test_rejects_stale_provenance_and_concurrent_edits(self):
        originals, outputs, _ = self.prepare()
        path = self.root / migration.CURB_REVIEW
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed after"):
            migration.apply(self.root, originals, outputs)
        self.assertEqual(
            (self.root / migration.CURBS).read_bytes(), originals[migration.CURBS]
        )
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.prepare()

    def test_station_endpoints_and_outside_rejection(self):
        self.assertEqual(migration.map_station(0, self.old, self.new), 0)
        self.assertEqual(migration.map_station(30, self.old, self.new), 32)
        for station in (-1, 31, float("nan")):
            with self.assertRaises(ValueError):
                migration.map_station(station, self.old, self.new)


if __name__ == "__main__":
    unittest.main()
