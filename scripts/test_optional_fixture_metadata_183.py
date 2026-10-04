"""Synthetic reporting controls only; no model fixtures/oracle execution."""
import copy
from pathlib import Path
import tempfile
import unittest
import optional_fixture_metadata_183 as metadata


class FixtureReport(unittest.TestCase):
    def populate(self, root):
        for model in metadata.MODELS:
            path = root / f"extern/Julia-mVMC/test/integration/reference/{model}/physcal_ref/inputs/modpara.def"
            path.parent.mkdir(parents=True)
            path.write_text("Header ignored\n" + "\n".join(
                f"{key} {value} # report input" for key, value in metadata.expected(model).items()) + "\n")

    def test_declared_and_explicit_overrides_separate_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            report = metadata.capture(root)
            self.assertEqual(report["explicit_gate_overrides"], {"seed": 1, "modes": ["real", "cmp"]})
            self.assertEqual([row["declared"]["NVMCSample"] for row in report["fixtures"]], [100, 1000, 5000])
            closure = {row["path"]: row["sha256"] for row in report["fixtures"]}
            metadata.validate(report, closure)
            for variant in ("missing", "duplicate", "override", "override_bool", "hash", "bool"):
                bad = copy.deepcopy(report)
                if variant == "missing":
                    bad["fixtures"].pop()
                elif variant == "duplicate":
                    bad["fixtures"][1] = copy.deepcopy(bad["fixtures"][0])
                elif variant == "override":
                    bad["explicit_gate_overrides"]["seed"] = 2
                elif variant == "override_bool":
                    bad["explicit_gate_overrides"]["seed"] = True
                elif variant == "hash":
                    bad["fixtures"][0]["sha256"] = "0" * 64
                else:
                    bad["fixtures"][0]["declared"]["RndSeed"] = True
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    metadata.validate(bad, closure)

    def test_every_required_field_absent_duplicate_or_mismatched(self):
        for field in metadata.FIELDS:
            for variant in ("absent", "duplicate", "mismatch", "malformed"):
                with self.subTest(field=field, variant=variant), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.populate(root)
                    path = root / f"extern/Julia-mVMC/test/integration/reference/{metadata.MODELS[0]}/physcal_ref/inputs/modpara.def"
                    lines = path.read_text().splitlines()
                    index = next(i for i, line in enumerate(lines) if line.startswith(field + " "))
                    if variant == "absent":
                        lines.pop(index)
                    elif variant == "duplicate":
                        lines.append(lines[index])
                    elif variant == "mismatch":
                        lines[index] = f"{field} {metadata.expected(metadata.MODELS[0])[field] + 1}"
                    else:
                        lines[index] = f"{field} 1.0"
                    path.write_text("\n".join(lines) + "\n")
                    with self.assertRaises(ValueError):
                        metadata.capture(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            path = root / f"extern/Julia-mVMC/test/integration/reference/{metadata.MODELS[0]}/physcal_ref/inputs/modpara.def"
            path.write_bytes(path.read_bytes() + b"\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                metadata.capture(root)


if __name__ == "__main__":
    unittest.main()
