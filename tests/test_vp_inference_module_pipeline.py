import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "evaluate_vp_inference_module_pipeline.py"
SPEC = importlib.util.spec_from_file_location("vp_module_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def arguments():
    return argparse.Namespace(
        device="cuda:0", physical_rank=32, rank_budget=512,
        latency_warmup_steps=5, cache_dir=Path("cache"),
        projector_checkpoint=Path("projector"),
        projector_cache_max_entries=512,
    )


class VPInferenceModulePipelineTest(unittest.TestCase):
    def test_sweep_sizes_and_unique_patch_names(self):
        self.assertEqual(len(pipeline.PATCH_CASES), 16)
        self.assertEqual(len(pipeline.TOKEN_K_VALUES), 4)
        self.assertEqual(len(pipeline.SPECULATIVE_CONFIGS), 9)
        names = [name for name, _ in pipeline.PATCH_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_isolated_commands_use_only_the_requested_wrapper(self):
        args = arguments()
        token = pipeline.token_command(
            args, Path("compact"), Path("token"), 6
        )
        self.assertEqual(pipeline.option(token, "--inference-tag"), "selector")
        self.assertEqual(pipeline.option(token, "--selector-recent-k"), "6")
        self.assertNotIn("--speculative-gamma", token)
        self.assertNotIn("--multimodal-mode", token)

        speculative = pipeline.speculative_command(
            args, Path("compact"), Path("spec"), 4, 0.3
        )
        self.assertEqual(
            pipeline.option(speculative, "--inference-tag"), "speculative"
        )
        self.assertEqual(pipeline.option(speculative, "--speculative-gamma"), "4")
        self.assertNotIn("--selector-recent-k", speculative)
        self.assertNotIn("--multimodal-mode", speculative)

    def test_full_stack_composes_all_three_modules(self):
        args = arguments()
        config = dict(pipeline.PATCH_CASES)["p_adaptive_t12"]
        command = pipeline.full_stack_command(
            args, Path("compact"), Path("full"), config, 6, 4, 0.3
        )
        self.assertEqual(pipeline.option(command, "--inference-tag"), "full_stack")
        self.assertEqual(
            pipeline.option(command, "--multimodal-mode"),
            "cached-patch-selection",
        )
        self.assertEqual(pipeline.option(command, "--selector-recent-k"), "6")
        self.assertEqual(pipeline.option(command, "--speculative-gamma"), "4")
        self.assertEqual(pipeline.option(command, "--speculative-threshold"), "0.3")

    def test_selection_respects_mae_constraint(self):
        rows = [
            {"case": "a", "family": "patch", "status": "complete",
             "mae": 9.5, "latency_mean_ms": 100.0},
            {"case": "b", "family": "patch", "status": "complete",
             "mae": 10.2, "latency_mean_ms": 60.0},
            {"case": "c", "family": "patch", "status": "complete",
             "mae": 10.5, "latency_mean_ms": 40.0},
        ]
        accuracy, speed = pipeline.choose_accuracy_and_speed(
            rows, "patch", baseline_mae=10.0, max_ratio=0.03
        )
        self.assertEqual(accuracy["case"], "a")
        self.assertEqual(speed["case"], "b")


if __name__ == "__main__":
    unittest.main()
