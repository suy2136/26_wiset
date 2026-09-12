import argparse
import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "evaluate_vp_fixed_spec_patch_token_sweep.py"
SPEC = importlib.util.spec_from_file_location("fixed_spec_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def args():
    return argparse.Namespace(
        device="cuda:0", physical_rank=32, rank_budget=512,
        latency_warmup_steps=5, cache_dir=Path("cache"),
        projector_checkpoint=Path("projector"),
        projector_cache_max_entries=512,
    )


class VPFixedSpecPatchTokenSweepTest(unittest.TestCase):
    def test_sweep_grid(self):
        self.assertEqual(sweep.TOKEN_K_VALUES, (6, 8, 10))
        self.assertEqual(len(sweep.PATCH_CASES), 8)
        self.assertEqual(sweep.SPEC_GAMMA, 6)
        self.assertEqual(sweep.SPEC_THRESHOLD, 0.4)

    def test_commands_fix_speculative_and_change_only_token_patch(self):
        command = sweep.spec_token_command(
            args(), Path("compact"), Path("result"), 10
        )
        self.assertEqual(command[command.index("--selector-recent-k") + 1], "10")
        self.assertEqual(command[command.index("--speculative-gamma") + 1], "6")
        self.assertEqual(command[command.index("--speculative-threshold") + 1], "0.4")
        self.assertEqual(command[command.index("--inference-tag") + 1], "full_stack")
        self.assertNotIn("--multimodal-mode", command)

        patch = dict(sweep.PATCH_CASES)["gated_k1_t3_skip1_cache"]
        full = sweep.full_stack_command(
            args(), Path("compact"), Path("result"), patch, 8
        )
        self.assertEqual(full[full.index("--selector-recent-k") + 1], "8")
        self.assertEqual(full[full.index("--speculative-gamma") + 1], "6")
        self.assertEqual(full[full.index("--cached-patch-policy") + 1], "gated-k1")
        self.assertEqual(full[full.index("--inference-tag") + 1], "full_stack")

    def test_generated_inference_tags_are_accepted_by_run_plm_parser(self):
        tree = ast.parse((ROOT / "run_plm.py").read_text(encoding="utf-8"))
        accepted = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or first.value != "--inference-tag":
                continue
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(
                    keyword.value, (ast.List, ast.Tuple)
                ):
                    accepted = {
                        item.value for item in keyword.value.elts
                        if isinstance(item, ast.Constant)
                    }
        self.assertIsNotNone(accepted)
        self.assertIn("full_stack", accepted)
        command = sweep.spec_token_command(
            args(), Path("compact"), Path("result"), 8
        )
        self.assertIn(command[command.index("--inference-tag") + 1], accepted)

    def test_selection_uses_fastest_admissible_token(self):
        rows = [
            {"case": "k6", "family": "spec_token", "status": "complete",
             "recent_k": 6, "mae": 10.1, "latency_mean_ms": 80.0},
            {"case": "k8", "family": "spec_token", "status": "complete",
             "recent_k": 8, "mae": 10.2, "latency_mean_ms": 60.0},
            {"case": "k10", "family": "spec_token", "status": "complete",
             "recent_k": 10, "mae": 10.4, "latency_mean_ms": 50.0},
        ]
        selected = sweep.select_fastest(rows, baseline_mae=10.0, max_ratio=0.03)
        self.assertEqual(selected["recent_k"], 8)

    def test_speculative_wrapper_forwards_patch_statistics(self):
        source = (ROOT / "models" / "speculative_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def patch_selection_history(self):", source)
        self.assertIn("def cached_patch_visual_token_history(self):", source)
        self.assertIn("def cached_patch_runtime_stats(self):", source)

    def test_wrapper_covers_all_post_wrap_patch_statistics(self):
        run_source = (ROOT / "run_plm.py").read_text(encoding="utf-8")
        wrapper_source = (
            ROOT / "models" / "speculative_pipeline.py"
        ).read_text(encoding="utf-8")
        required = (
            "patch_selection_history",
            "cached_patch_visual_token_history",
            "cached_patch_runtime_stats",
        )
        for attribute in required:
            self.assertIn(f"pipeline.{attribute}", run_source)
            self.assertIn(f"def {attribute}(self):", wrapper_source)


if __name__ == "__main__":
    unittest.main()
