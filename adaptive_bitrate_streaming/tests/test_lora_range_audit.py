import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adaptive_bitrate_streaming/analysis/run_seed1_lora_range_audit.py"
LOW_RANK = ROOT / "adaptive_bitrate_streaming/plm_special/models/low_rank.py"


class TestLoraRangeAudit(unittest.TestCase):
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

    def test_audit_is_environment_gated_and_detect_only(self):
        source = LOW_RANK.read_text(encoding="utf-8")
        self.assertIn("ABR_LORA_RANGE_AUDIT_PATH", source)
        self.assertIn("'mode': 'detect-only'", source)
        self.assertIn("'would_clamp_calls'", source)
        self.assertIn("if not _RANGE_AUDIT_PATH:", source)


if __name__ == "__main__":
    unittest.main()
