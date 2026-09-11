import argparse
import tempfile
import unittest
from pathlib import Path

from analysis.evaluate_kinematic_selector_finalists_compact import build_command


class CompactFinalistCommandTest(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            device="cuda",
            latency_warmup_steps=5,
            projector_checkpoint=Path("projector"),
        )
        self.config = {
            "top_k": 4.0,
            "velocity_window": 5.0,
            "horizon_scale": 0.5,
            "acceleration_weight": 0.0,
            "uncertainty_deg": 0.0,
            "uncertainty_growth": 0.0,
        }

    def test_pure_nbs_is_compact_without_selector(self):
        command = build_command(
            self.args, Path("nbs"), Path("results"), None, Path("compact")
        )
        self.assertIn("compact", command)
        self.assertIn("--nbs-compact-output-dir", command)
        self.assertNotIn("--multimodal-mode", command)
        self.assertNotIn("--patch-selector-type", command)

    def test_finalist_reuses_compact_checkpoint_and_selector(self):
        command = build_command(
            self.args, Path("compact"), Path("results"), self.config
        )
        self.assertEqual(command[command.index("--model-path") + 1], "compact")
        self.assertEqual(
            command[command.index("--nbs-inference-mode") + 1], "compact"
        )
        self.assertEqual(
            command[command.index("--patch-selector-type") + 1], "kinematic"
        )
        self.assertEqual(command[command.index("--patch-top-k") + 1], "4")


if __name__ == "__main__":
    unittest.main()
