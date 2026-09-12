import argparse
import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "evaluate_vp_fixed_spec_quality_multiseed.py"
SPEC = importlib.util.spec_from_file_location("vp_quality_multiseed", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def args():
    return argparse.Namespace(
        device="cuda:0", physical_rank=32, rank_budget=512,
        latency_warmup_steps=5, cache_dir=Path("cache"),
        projector_checkpoint=Path("projector"),
        projector_cache_max_entries=512,
    )


class VPFixedSpecQualityMultiseedTest(unittest.TestCase):
    def test_exact_quality_configuration(self):
        self.assertEqual(module.SEEDS, (1, 2, 3))
        self.assertEqual(len(module.CONFIGURATIONS), 5)
        self.assertEqual(module.TOKEN_K, 8)
        self.assertEqual(module.SPEC_GAMMA, 6)
        self.assertEqual(module.SPEC_THRESHOLD, 0.4)
        self.assertEqual(module.QUALITY_PATCH, {
            "policy": "gated-k1", "threshold": 6.0,
            "max_skip": 0, "projector_cache": True,
        })

    def test_commands_keep_lora_seed_and_change_evaluation_seed(self):
        baseline = module.evaluation_command(
            args(), Path("compact"), Path("result"), 3, "baseline"
        )
        quality = module.evaluation_command(
            args(), Path("compact"), Path("result"), 2, "full_stack"
        )
        for command, seed in ((baseline, "3"), (quality, "2")):
            self.assertEqual(command[command.index("--seed") + 1], seed)
            self.assertEqual(command[command.index("--data-seed") + 1], seed)
            self.assertEqual(command[command.index("--lora-seed") + 1], "1")
        self.assertEqual(
            quality[quality.index("--inference-tag") + 1], "full_stack"
        )
        self.assertEqual(
            quality[quality.index("--cached-patch-policy") + 1], "gated-k1"
        )
        self.assertEqual(
            quality[quality.index("--cached-patch-motion-threshold-deg") + 1],
            "6.0",
        )
        self.assertEqual(quality[quality.index("--selector-recent-k") + 1], "8")
        self.assertEqual(quality[quality.index("--speculative-gamma") + 1], "6")

    def test_independent_module_commands_are_actually_independent(self):
        commands = {
            kind: module.evaluation_command(
                args(), Path("compact"), Path("result"), 2, kind
            )
            for kind in ("patch", "token", "speculative")
        }
        self.assertIn("--multimodal-mode", commands["patch"])
        self.assertNotIn("--selector-recent-k", commands["patch"])
        self.assertNotIn("--speculative-gamma", commands["patch"])
        self.assertIn("--selector-recent-k", commands["token"])
        self.assertNotIn("--multimodal-mode", commands["token"])
        self.assertNotIn("--speculative-gamma", commands["token"])
        self.assertIn("--speculative-gamma", commands["speculative"])
        self.assertNotIn("--multimodal-mode", commands["speculative"])
        self.assertNotIn("--selector-recent-k", commands["speculative"])

    def test_generated_tag_and_wrapper_contract_match_run_plm(self):
        tree = ast.parse((ROOT / "run_plm.py").read_text(encoding="utf-8"))
        accepted = set()
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
                    accepted.update(
                        item.value for item in keyword.value.elts
                        if isinstance(item, ast.Constant)
                    )
        self.assertIn("full_stack", accepted)
        wrapper = (ROOT / "models" / "speculative_pipeline.py").read_text(
            encoding="utf-8"
        )
        for attribute in (
            "patch_selection_history", "cached_patch_visual_token_history",
            "cached_patch_runtime_stats",
        ):
            self.assertIn(f"def {attribute}(self):", wrapper)

    def test_summary_uses_all_three_seeds(self):
        rows = []
        for config in module.CONFIGURATIONS:
            case = config["case"]
            for seed, mae, latency in ((1, 10.0, 100.0), (2, 11.0, 90.0), (3, 12.0, 80.0)):
                rows.append({
                    "case": case, "label": case, "status": "complete",
                    "evaluation_seed": seed, "mae": mae, "rmse": mae + 1,
                    "latency_mean_ms": latency,
                })
        summary = module.summarize(rows)
        self.assertEqual(len(summary), 5)
        self.assertEqual(summary[0]["seed_count"], 3)
        self.assertEqual(summary[0]["mae_mean"], 11.0)
        self.assertEqual(summary[0]["latency_mean_ms_mean"], 90.0)


if __name__ == "__main__":
    unittest.main()
