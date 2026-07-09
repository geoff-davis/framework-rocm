"""Tests for target-specific dependency-lock generation helpers."""

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.lock_dependencies import (
    TARGET_PLATFORMS,
    canonicalize_name,
    parse_exact_requirements,
    parse_hashed_lock,
    render_lock,
)


class LockDependenciesTests(unittest.TestCase):
    def test_platform_tags_do_not_exceed_container_glibc(self):
        self.assertEqual(TARGET_PLATFORMS[0], "manylinux_2_39_x86_64")
        self.assertNotIn("manylinux_2_40_x86_64", TARGET_PLATFORMS)

    def test_canonicalize_name(self):
        self.assertEqual(canonicalize_name("Typing_Extensions"), "typing-extensions")
        self.assertEqual(canonicalize_name("sentence.transformers"), "sentence-transformers")

    def test_parse_exact_requirements_rejects_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requirements.txt"
            path.write_text("demo>=1\n")
            with self.assertRaisesRegex(ValueError, "exact name==version"):
                parse_exact_requirements(path)

    def test_render_lock_uses_wheel_metadata_and_exact_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "demo-1.2.3-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "demo-1.2.3.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: Demo_Package\nVersion: 1.2.3\n",
                )
            expected_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
            lock = render_lock([("demo-package", "1.2.3")], [wheel])
            self.assertIn("demo-package==1.2.3", lock)
            self.assertIn(f"--hash=sha256:{expected_hash}", lock)

    def test_parse_hashed_lock_rejects_missing_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requirements.lock"
            path.write_text("demo==1.2.3 \\\n")
            with self.assertRaisesRegex(ValueError, "exactly one SHA-256 hash"):
                parse_hashed_lock(path)


if __name__ == "__main__":
    unittest.main()
