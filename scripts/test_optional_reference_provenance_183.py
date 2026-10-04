"""Independent literal offline declarations; no Julia/model execution."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import optional_reference_provenance_183 as reference


class ReferenceContract(unittest.TestCase):
    def test_capture_distinguishes_checkout_and_mixed_historical_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / reference.MANIFEST
            manifest.parent.mkdir(parents=True)
            manifest.write_text('julia_version = "1.13.1"\n')
            historical = root / "legacy/provenance.txt"
            historical.parent.mkdir()
            historical.write_text("julia=1.11.7\nC reference revision=622166a\n")
            c_only = root / "c/metadata.txt"
            c_only.parent.mkdir()
            c_only.write_text("Apple Clang; C Lanczos acquisition\n")
            paths = [manifest, historical, c_only]
            hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            report = reference.capture(paths)
            reference.validate(report, hashes)
            self.assertEqual(report["julia_runtime"],
                             {"status": "NotRun", "version": None, "blas": None})
            rows = {row["path"]: row for row in report["declarations"]}
            self.assertEqual(rows[str(manifest)]["julia_version"], "1.13.1")
            self.assertEqual(rows[str(historical)]["julia_versions"], ["1.11.7"])
            self.assertEqual(rows[str(c_only)]["julia_versions"], "NotRecorded")
            self.assertEqual(rows[str(c_only)]["text"], c_only.read_text())

    def test_missing_invalid_or_other_checkout_manifest_version_rejected(self):
        path = "/literal/" + reference.MANIFEST
        for text, expected in (("# absent\n", "Manifest Julia version missing/invalid"),
                               ('julia_version = true\n', "Manifest Julia version missing/invalid"),
                               ('julia_version = "1.11.7"\n', "checkout Manifest must declare reviewed Julia 1.13.1")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, expected):
                reference.row(path, text)
        with self.assertRaisesRegex(ValueError, "missing/duplicate/off-closure"):
            reference.validate(reference.report([]), {})


if __name__ == "__main__":
    unittest.main()
