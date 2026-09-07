import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis.run_seed1_lora_range_audit import (
    REQUIRED_CHECKPOINT_FILES,
    checkpoint_missing,
    resolve_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adaptive_bitrate_streaming/analysis/run_seed1_lora_range_audit.py"
LOW_RANK = ROOT / "adaptive_bitrate_streaming/plm_special/models/low_rank.py"


class TestLoraRangeAudit(unittest.TestCase):
    @staticmethod
    def _complete(path: Path):
        path.mkdir(parents=True)
        for name in REQUIRED_CHECKPOINT_FILES:
            (path / name).write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"weights")

    def test_dry_run_uses_seed1_budget1536_without_compaction(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            env=os.environ.copy(),
        )
        output = result.stdout
        self.assertIn("--nbs-rank-budget 1536", output)
        self.assertIn("--data-seed 1", output)
        self.assertIn("--trace-num 1", output)
        self.assertNotIn("--nbs-compact-inference", output)

    def test_incomplete_best_falls_back_to_complete_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = root / "early_stop_-1_best_model"
            requested.mkdir()
            latest = root / "early_stop_-1_checkpoint/latest"
            self._complete(latest)
            selected, used_fallback = resolve_checkpoint(requested)
            self.assertEqual(selected, latest.resolve())
            self.assertTrue(used_fallback)
            self.assertIn(
                "adapter_model.safetensors or adapter_model.bin",
                checkpoint_missing(requested),
            )

    def test_audit_is_environment_gated_and_detect_only(self):
        source = LOW_RANK.read_text(encoding="utf-8")
        self.assertIn("ABR_LORA_RANGE_AUDIT_PATH", source)
        self.assertIn("'mode': 'detect-only'", source)
        self.assertIn("'would_clamp_calls'", source)
        self.assertIn("if not _RANGE_AUDIT_PATH:", source)


if __name__ == "__main__":
    unittest.main()
