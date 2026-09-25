import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_bitrate_streaming.analysis import (
    run_abr_nbs_rank_bound_ablation as ablation,
)


class ABRNBSRankBoundAblationTests(unittest.TestCase):
    def args(self, directory="output"):
        return ablation.parse_args([
            "--output-dir", directory,
            "--base-model-dir", "base",
            "--exp-pool-path", "pool.pkl",
        ])

    def test_four_feasible_rank_bound_specs(self):
        self.assertEqual(
            [(item["min_rank"], item["max_rank"]) for item in ablation.SPECS],
            [(8, 32), (16, 32), (2, 28), (2, 24)],
        )
        for item in ablation.SPECS:
            self.assertLessEqual(item["min_rank"] * 64, ablation.RANK_BUDGET)
            self.assertGreaterEqual(item["max_rank"] * 64, ablation.RANK_BUDGET)

    def test_configs_match_specs(self):
        for spec in ablation.SPECS:
            config = json.loads(
                (ablation.ABR_ROOT / spec["rank_config"]).read_text(encoding="utf-8")
            )
            for bounds in config.values():
                self.assertEqual(bounds["min_rank"], spec["min_rank"])
                self.assertEqual(bounds["max_rank"], spec["max_rank"])

    def test_train_and_eval_use_prescaled_qk_and_compaction(self):
        args = self.args()
        spec = ablation.SPECS[0]
        experiment = ablation.experiment_for(args, spec)
        train = ablation.add_prescaled_qk(
            ablation.training.build_training_command(
                ablation.training_args(args, spec), experiment,
            )
        )
        evaluation = {**experiment, "seed": 3, "data_seed": 3}
        test = ablation.add_prescaled_qk(
            ablation.training.build_test_command(
                ablation.training_args(args, spec), evaluation, Path("checkpoint"),
            )
        )
        for command in (train, test):
            self.assertIn("--fp16-attention-prescaled-qk", command)
            self.assertNotIn("--fp16-attention-fp32-scores", command)
            self.assertEqual(command[command.index("--nbs-rank-budget") + 1], "1536")
            self.assertEqual(command[command.index("--nbs-rank-config") + 1], spec["rank_config"])
        self.assertIn("--nbs-compact-inference", test)
        self.assertEqual(test[test.index("--seed") + 1], "3")
        self.assertEqual(test[test.index("--data-seed") + 1], "3")
        self.assertEqual(test[test.index("--lora-seed") + 1], "1")

    def test_run_tags_and_outputs_are_isolated(self):
        with TemporaryDirectory() as directory:
            args = self.args(directory)
            experiments = [ablation.experiment_for(args, spec) for spec in ablation.SPECS]
            self.assertEqual(len({item["run_tag"] for item in experiments}), 4)
            state_path, state = ablation.load_state(args)
            self.assertEqual(state_path, Path(directory).resolve() / "pipeline_state.json")
            self.assertEqual(state["signature"]["training_data_seed"], 4)


if __name__ == "__main__":
    unittest.main()
