"""Regression checks for installed commands and import-safe plotting."""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from soma.cli import COMMANDS
from soma.paths import DEFAULT_CONFIG, DEFAULT_HEAD_CONFIG


class InterfaceTests(unittest.TestCase):
    def test_packaged_configs_exist(self):
        self.assertTrue(DEFAULT_CONFIG.is_file())
        self.assertTrue(DEFAULT_HEAD_CONFIG.is_file())

    def test_plotting_import_does_not_parse_arguments(self):
        with patch.object(sys, "argv", ["unrelated-app", "--not-a-soma-option"]):
            module = importlib.import_module("soma.plotting.learning_curves")
            self.assertTrue(callable(module.main))

    def test_command_help_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in COMMANDS:
                with self.subTest(command=name):
                    result = subprocess.run(
                        [sys.executable, "-m", "soma", name, "--help"],
                        cwd=directory,
                        env=os.environ.copy(),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
