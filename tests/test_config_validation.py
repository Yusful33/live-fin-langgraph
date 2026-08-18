"""Tests for startup configuration validation.

_require_env lives in langgraph_fin_agent._config, which has no external
dependencies, so these tests run with the standard library alone and do not
need the full package virtualenv to be populated.
"""
import os
import sys
import unittest
from unittest.mock import patch

# Make sure the package root is on the path when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph_fin_agent._config import _require_env  # noqa: E402


class TestRequireEnv(unittest.TestCase):
    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"_TEST_VAR_XYZ": "my-secret-key"}):
            self.assertEqual(_require_env("_TEST_VAR_XYZ"), "my-secret-key")

    def test_raises_when_variable_is_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "_TEST_VAR_XYZ"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                _require_env("_TEST_VAR_XYZ")
        self.assertIn("_TEST_VAR_XYZ", str(ctx.exception))

    def test_raises_when_variable_is_empty_string(self):
        with patch.dict(os.environ, {"_TEST_VAR_XYZ": ""}):
            with self.assertRaises(EnvironmentError):
                _require_env("_TEST_VAR_XYZ")

    def test_raises_when_variable_is_whitespace_only(self):
        with patch.dict(os.environ, {"_TEST_VAR_XYZ": "   "}):
            with self.assertRaises(EnvironmentError):
                _require_env("_TEST_VAR_XYZ")

    def test_error_message_names_the_missing_variable(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                _require_env("OPENAI_API_KEY")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_raises_on_missing_openai_key(self):
        """Regression: the 401 failure mode is triggered by a missing/empty API key."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(EnvironmentError):
                _require_env("OPENAI_API_KEY")

    def test_accepts_valid_openai_key_format(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-proj-abc123"}):
            result = _require_env("OPENAI_API_KEY")
        self.assertEqual(result, "sk-proj-abc123")


if __name__ == "__main__":
    unittest.main()
