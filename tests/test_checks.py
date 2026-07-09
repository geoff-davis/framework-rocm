"""Hardware-free regression tests for the GPU check helpers."""

import math
import os
import unittest
from unittest.mock import patch

import check_gpu
import check_jax


class CheckHelperTests(unittest.TestCase):
    def test_scalar_validation_rejects_wrong_and_nonfinite_results(self):
        for module in (check_gpu, check_jax):
            with self.subTest(module=module.__name__):
                self.assertTrue(module.scalar_is_close(1024.0, 1024.0))
                self.assertFalse(module.scalar_is_close(1000.0, 1024.0))
                self.assertFalse(module.scalar_is_close(math.nan, 1024.0))
                self.assertFalse(module.scalar_is_close(math.inf, 1024.0))

    def test_boolean_environment_values_are_explicit(self):
        for module in (check_gpu, check_jax):
            for value in ("1", "true", "YES", "on"):
                with self.subTest(module=module.__name__, value=value):
                    with patch.dict(os.environ, {"TEST_FLAG": value}):
                        self.assertTrue(module.env_flag("TEST_FLAG"))
            for value in ("0", "false", "NO", "off", ""):
                with self.subTest(module=module.__name__, value=value):
                    with patch.dict(os.environ, {"TEST_FLAG": value}):
                        self.assertFalse(module.env_flag("TEST_FLAG"))

    def test_invalid_boolean_environment_value_is_rejected(self):
        for module in (check_gpu, check_jax):
            with self.subTest(module=module.__name__):
                with patch.dict(os.environ, {"TEST_FLAG": "sometimes"}):
                    with self.assertRaisesRegex(ValueError, "TEST_FLAG"):
                        module.env_flag("TEST_FLAG")

    def test_optional_performance_threshold_is_validated(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "25"}):
            self.assertEqual(check_gpu.optional_positive_float("TEST_LIMIT"), 25.0)
        with patch.dict(os.environ, {"TEST_LIMIT": ""}):
            self.assertIsNone(check_gpu.optional_positive_float("TEST_LIMIT"))
        for value in ("zero", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"TEST_LIMIT": value}):
                    with self.assertRaisesRegex(ValueError, "TEST_LIMIT"):
                        check_gpu.optional_positive_float("TEST_LIMIT")


if __name__ == "__main__":
    unittest.main()
