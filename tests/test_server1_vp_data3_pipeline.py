from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analysis import run_server1_vp_data3_pipeline as pipeline


class Server1VpData3PipelineTest(unittest.TestCase):
    def test_training_variants_and_stage_order(self):
        self.assertEqual(
            {value["variant"] for value in pipeline.METHODS.values()},
            {
                "uniform_r8_data3", "adalora_b512_data3",
                "shapley_b512_data3", "eva_b512_data3", "nbs_v19_data3",
            },
        )
        self.assertLess(
            pipeline.STAGE_ORDER.index("train_adalora"),
            pipeline.STAGE_ORDER.index("evaluate_adalora"),
        )
        self.assertLess(
            pipeline.STAGE_ORDER.index("compact_nbs"),
            pipeline.STAGE_ORDER.index("evaluate_nbs_modules"),
        )

    def test_module_command_uses_prescaled_qk_and_continuous_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                device="cuda:0", latency_warmup_steps=5,
                tuned_projector=root / "projector.pth",
                patch_cache=root / "cache",
            )
            command = pipeline.module_command(
                args, root / "compact", "full_stack", 3, root / "result",
            )
        self.assertIn("--vp-fp16-prescaled-qk", command)
        self.assertNotIn("--vp-fp16-fallback", command)
        self.assertEqual(command[command.index("--seed") + 1], "3")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertEqual(command[command.index("--lora-seed") + 1], "1")
        self.assertEqual(
            command[command.index("--evaluation-rng-mode") + 1], "continuous",
        )

    def test_budget_mismatch_is_recorded_without_rejection(self):
        original = pipeline.vp_lora.checkpoint_description
        pipeline.vp_lora.checkpoint_description = lambda method, checkpoint: {
            "method": method, "active_rank_total": 511,
        }
        try:
            description = pipeline.inspect_budget("adalora", Path("checkpoint"))
        finally:
            pipeline.vp_lora.checkpoint_description = original
        self.assertEqual(description["active_rank_total"], 511)
        self.assertFalse(description["budget_match"])
        self.assertEqual(description["target_rank_budget"], 512)
        self.assertEqual(description["budget_note"], "nonmatching_511_target_512")

    def test_shell_exposes_data3_and_opt_in_training_safety(self):
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        for variant in (value["variant"] for value in pipeline.METHODS.values()):
            self.assertIn(variant, shell)
        self.assertIn('VP_FP16_PRESCALED_QK="${VP_FP16_PRESCALED_QK:-0}"', shell)
        self.assertIn('TRAIN_CMD+=(--vp-fp16-prescaled-qk)', shell)

    def test_seed4_variants_and_separate_default_output(self):
        args = pipeline.parse_args(["--training-data-seed", "4", "--dry-run"])
        self.assertEqual(args.output_dir.name, "server1_vp_data4_pipeline")
        variants = {item["variant"] for item in pipeline.method_specs(4).values()}
        self.assertEqual(variants, {
            "uniform_r8_data4", "adalora_b512_data4", "shapley_b512_data4",
            "eva_b512_data4", "nbs_v19_data4",
        })
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        for variant in variants:
            self.assertIn(variant, shell)

    def test_failed_compaction_is_preserved_and_retried_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "nbs_compact/compact_checkpoint"
            original.mkdir(parents=True)
            (original / "equivalence_report.json").write_text('{"passed": false}')
            args = argparse.Namespace(output_dir=root, resume=True, dry_run=False)

            def fake_run(command, result_dir, resume):
                candidate = Path(command[command.index("--nbs-compact-output-dir") + 1])
                self.assertNotEqual(candidate, original)
                self.assertEqual(
                    command[command.index("--nbs-compaction-output-atol") + 1],
                    "0.01",
                )
                candidate.mkdir(parents=True)
                for name in ("compact_adapter.pt", "modules_except_plm.bin",
                             "compaction_metadata.json"):
                    (candidate / name).touch()
                (candidate / "equivalence_report.json").write_text('{"passed": true}')
                return {}

            with mock.patch.object(pipeline, "inspect_budget"), \
                    mock.patch.object(pipeline, "module_command", side_effect=
                        lambda args, compact, kind, seed, result_dir, source: [
                            "python", "run_plm.py", "--nbs-compact-output-dir", str(compact),
                        ]), \
                    mock.patch.object(pipeline, "run_case", side_effect=fake_run):
                candidate = pipeline.compact_nbs(args, root / "source")
            self.assertEqual(json.loads((original / "equivalence_report.json").read_text()),
                             {"passed": False})
            self.assertTrue(candidate.name.startswith("compact_checkpoint_fp16_retry_"))


if __name__ == "__main__":
    unittest.main()
