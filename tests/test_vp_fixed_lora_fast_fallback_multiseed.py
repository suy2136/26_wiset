import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis import evaluate_vp_fixed_lora_fast_fallback_multiseed as pipeline


class FixedLoraFastFallbackPipelineTest(unittest.TestCase):
    def make_checkpoint(self, root, peft_type, rank, rank_pattern=None,
                        init_rank=None):
        checkpoint = Path(root)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_model.bin").write_bytes(b"weights")
        (checkpoint / "modules_except_plm.bin").write_bytes(b"head")
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({
                "peft_type": peft_type,
                "r": rank,
                "init_r": init_rank,
                "target_modules": ["q_proj", "v_proj"],
                "rank_pattern": rank_pattern or {},
            }),
            encoding="utf-8",
        )
        return checkpoint

    def test_checkpoint_budget_is_recorded_without_rejecting_preliminary(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.make_checkpoint(
                temporary,
                "ADALORA",
                8,
                {"a": [True] * 320, "b": [True] * 325},
                32,
            )
            description = pipeline.checkpoint_description("shapley", checkpoint)
        self.assertEqual(description["active_rank_total"], 645)
        self.assertEqual(description["budget_note"], "preliminary_nonmatching_645")

    def test_active_rank_accepts_integer_and_mask_encodings(self):
        self.assertEqual(pipeline.active_rank(8), 8)
        self.assertEqual(pipeline.active_rank([True, False, True]), 2)
        self.assertEqual(pipeline.active_rank([1, 0, 1, 1]), 3)

    def test_subset_methods_preserve_requested_execution_order(self):
        args = pipeline.parser().parse_args([
            "--methods", "shapley", "adalora",
            "--shapley-checkpoint", "shapley_ckpt",
            "--adalora-checkpoint", "adalora_ckpt",
            "--output-dir", "output",
        ])
        specs = pipeline.selected_method_specs(args)
        self.assertEqual([spec[0] for spec in specs], ["shapley", "adalora"])
        self.assertIsNone(args.uniform_checkpoint)
        self.assertIsNone(args.eva_checkpoint)

    def test_command_uses_fast_fallback_and_no_acceleration_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.make_checkpoint(
                root / "eva", "LORA", 32, {"a": 256, "b": 256}
            )
            description = pipeline.checkpoint_description("eva", checkpoint)
            args = SimpleNamespace(device="cuda:0", latency_warmup_steps=5)
            rank_config = pipeline.write_fixed_rank_config(
                description, checkpoint, root
            )
            command = pipeline.build_command(
                args, description, checkpoint, 3, root / "result", rank_config
            )
        self.assertIn("--vp-fp16-fallback", command)
        self.assertEqual(command[command.index("--evaluation-rng-mode") + 1],
                         "continuous")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertIn("--lora-rank-config", command)
        for option in (
            "--multimodal-mode", "--inference-tag", "--selector-recent-k",
            "--speculative-gamma", "--nbs-inference-mode",
        ):
            self.assertNotIn(option, command)

    def test_adalora_checkpoint_uses_shape_compatible_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.make_checkpoint(
                root / "adalora", "ADALORA", 8, {"a": 8}, 32
            )
            description = pipeline.checkpoint_description("adalora", checkpoint)
            args = SimpleNamespace(device="cuda:0", latency_warmup_steps=5)
            command = pipeline.build_command(
                args, description, checkpoint, 1, root / "result", None
            )
        self.assertIn("--use-adalora", command)
        self.assertEqual(command[command.index("--adalora-init-rank") + 1], "32")
        self.assertEqual(command[command.index("--adalora-allocator") + 1], "peft")

    def test_run_plm_exposes_opt_in_fallback(self):
        source = (pipeline.REPO_ROOT / "run_plm.py").read_text(encoding="utf-8")
        self.assertIn("'--vp-fp16-fallback'", source)
        self.assertIn("if args.vp_fp16_fallback:", source)
        self.assertIn(
            "args.nbs_inference_mode == 'compact' or args.vp_fp16_fallback",
            source,
        )

    def test_nonvisual_dataset_loading_does_not_require_top_level_opencv(self):
        source = (pipeline.REPO_ROOT / "dataset" / "load_dataset.py").read_text(
            encoding="utf-8"
        )
        prefix = source.split("def pack_data", 1)[0]
        self.assertNotIn("import cv2", prefix)
        self.assertIn("if for_track:\n        try:\n            import cv2", source)


if __name__ == "__main__":
    unittest.main()
