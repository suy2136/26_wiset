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

    def test_budget_scaling_arguments_and_shell_override(self):
        args = pipeline.parse_args([
            "--training-data-seed", "4", "--target-budget", "1536",
            "--lora-only", "--dry-run",
        ])
        self.assertEqual(args.target_budget, 1536)
        self.assertTrue(args.lora_only)
        self.assertEqual(args.output_dir.name, "server1_vp_data4_budget1536_pipeline")
        shell = (pipeline.REPO_ROOT / "scripts/run_netllm_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("VP_TOTAL_RANK_BUDGET", shell)
        self.assertIn('MODEL_TAG="${MODEL_TAG}_budget${RANK_BUDGET}"', shell)
        self.assertIn(
            'RANK_BUDGET="${VP_TOTAL_RANK_BUDGET:-512}"', shell,
        )

    def test_nbs_only_runs_pure_compact_pipeline(self):
        args = pipeline.parse_args([
            "--training-data-seed", "4", "--target-budget", "1024",
            "--nbs-only", "--dry-run",
        ])
        self.assertTrue(args.nbs_only)
        self.assertFalse(args.lora_only)

    def test_strict_nbs_budget_mismatch_is_rejected(self):
        original = pipeline.vp_lora.checkpoint_description
        original_budget = pipeline.TARGET_BUDGET
        pipeline.vp_lora.checkpoint_description = lambda method, checkpoint: {
            "method": method, "active_rank_total": 512,
        }
        pipeline.TARGET_BUDGET = 1024
        try:
            with self.assertRaisesRegex(ValueError, "active rank is 512.*target is 1024"):
                pipeline.inspect_budget(
                    "nbs", Path("checkpoint"), require_exact=True,
                )
        finally:
            pipeline.vp_lora.checkpoint_description = original
            pipeline.TARGET_BUDGET = original_budget

    def test_seed4_nbs_final_alias_recovers_physical_weights_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant = pipeline.method_specs(4)["nbs"]["variant"]
            run_dir = root / "run"
            run_dir.mkdir()
            (root / f"{variant}_latest.txt").write_text(str(run_dir))
            physical = root / "best_ar_model"
            physical.mkdir()
            for name in ("adapter_config.json", "adapter_model.bin",
                         "modules_except_plm.bin"):
                (physical / name).touch()
            alias = root / "final_nbs_model"
            alias.mkdir()
            (alias / "checkpoint_alias.json").write_text(
                json.dumps({"is_alias": True, "alias_of": "../best_ar_model"})
            )
            (run_dir / "metadata.env").write_text(
                f"final_nbs_model={alias}\nrank_budget=512\n"
            )
            state_path = root / "pipeline_state.json"
            state = {"checkpoints": {}}
            args = argparse.Namespace(resume=True, dry_run=False)
            with mock.patch.object(pipeline, "VP_RUN_ROOT", root), \
                    mock.patch.object(pipeline, "METHODS", pipeline.method_specs(4)), \
                    mock.patch.object(pipeline, "TRAINING_DATA_SEED", 4), \
                    mock.patch.object(pipeline, "inspect_budget"), \
                    mock.patch.object(pipeline.subprocess, "run") as run:
                recovered = pipeline.train_method(args, state, state_path, "nbs")
            self.assertEqual(recovered, physical.resolve())
            self.assertEqual(state["checkpoints"]["vp_nbs_data4"], str(physical.resolve()))
            run.assert_not_called()

    def test_incomplete_alias_target_is_not_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alias = root / "final_nbs_model"
            alias.mkdir()
            (alias / "checkpoint_alias.json").write_text(
                json.dumps({"alias_of": "../missing_best_ar_model"})
            )
            self.assertIsNone(pipeline.resolved_complete_checkpoint(alias))

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
                for name in ("compact_adapter.pt", "modules_except_plm.bin"):
                    (candidate / name).touch()
                (candidate / "compaction_metadata.json").write_text(json.dumps({
                    "compact_rank_total": pipeline.TARGET_BUDGET,
                    "source_checkpoint": str((root / "source").resolve()),
                }))
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
