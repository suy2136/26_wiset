import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

from analysis.evaluate_cached_patch_selectors_compact import (
    build_command,
    run_streaming_command,
)
from analysis.evaluate_cached_patch_followups_compact import (
    QUALITY_CASES,
    SPEED_CASES,
    selector_command,
)


class CachedPatchFollowupPlanTest(unittest.TestCase):
    def test_all_group_contains_eight_unique_followups(self):
        cases = QUALITY_CASES + SPEED_CASES
        names = [name for name, _ in cases]
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(set(names)), 8)
        self.assertTrue(all(name.startswith("q_") for name, _ in QUALITY_CASES))
        self.assertTrue(all(name.startswith("s_") for name, _ in SPEED_CASES))

    def test_speed_group_contains_real_token_or_compute_reduction(self):
        configs = dict(SPEED_CASES)
        self.assertTrue(any(
            config["policy"].startswith("gated-")
            for config in configs.values()
        ))
        self.assertTrue(any(
            config.get("projector_cache") for config in configs.values()
        ))
        self.assertTrue(any(
            config.get("refresh", 1) > 1 for config in configs.values()
        ))

    def test_followup_case_uses_allowed_selector_inference_tag(self):
        args = argparse.Namespace(
            device="cuda:0", physical_rank=32, rank_budget=512,
            latency_warmup_steps=5, cache_dir=Path("cache"),
            projector_checkpoint=Path("projector"),
            projector_cache_max_entries=512,
            motion_threshold_deg=12.0,
        )
        command = selector_command(
            args, Path("compact"), Path("q_threshold6"),
            dict(QUALITY_CASES)["q_threshold6"],
        )
        self.assertEqual(
            command[command.index("--inference-tag") + 1], "selector"
        )

        original = build_command(
            args, Path("compact"), Path("cached_adaptive"), policy="adaptive"
        )
        self.assertEqual(
            original[original.index("--inference-tag") + 1], "selector"
        )

    def test_streaming_command_mirrors_output_to_console_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            console = io.StringIO()
            with redirect_stdout(console):
                run_streaming_command(
                    [sys.executable, "-c", "print('progress-now')"],
                    Path(directory), log_path,
                )
            self.assertIn("progress-now", console.getvalue())
            self.assertIn("progress-now", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
