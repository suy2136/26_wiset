import argparse
import csv
from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.analysis import (
    run_nbs_v19_lora_data_seed_pair as pipeline,
)


class NBSLoraDataSeedPairTest(unittest.TestCase):
    def legacy_rows(self):
        rows = []
        for index, spec in enumerate(pipeline.TARGET_SPECS):
            rows.append({
                "experiment": spec["name"], "data_seed": "1",
                "rank_budget": "1536", "evaluation_rng_mode": "per-episode",
                "metrics_path": f"legacy/seed1/{index}/selector_metrics.json",
                "mean_reward": str(0.8 + index * 0.01),
                "inference_latency_mean_ms": str(80 - index),
            })
        return rows

    def test_matrix_is_latest_five(self):
        self.assertEqual(len(pipeline.TARGET_SPECS), 5)
        self.assertEqual(pipeline.latest.TOKEN["token_offsets"], (0, 2, 3, 4, 7))
        self.assertEqual(pipeline.latest.SPECULATIVE["speculative_steps"], 5)

    def test_legacy_source_selects_only_seed1_and_rejects_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            pipeline.sweep.write_csv(path, self.legacy_rows())
            rows, mode = pipeline.legacy_seed1_rows(path, 1536)
            self.assertEqual(len(rows), 5)
            self.assertEqual(mode, "per-episode")
            rows[0]["metrics_path"] = "run_fp32_attention/metrics.json"
            pipeline.sweep.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "uses FP32 attention"):
                pipeline.legacy_seed1_rows(path, 1536)

    def test_seed3_commands_do_not_enable_fp32_attention(self):
        args = argparse.Namespace(
            data_seed=3, lora_seed=1, evaluation_rng_mode="per-episode",
            run_tag="lora_data_seed3_legacy_fp16",
            base_model_dir=Path("base"), checkpoint_dir=Path("checkpoint_seed3"),
            exp_pool_path=Path("pool"), physical_rank=32, rank_budget=1536,
            rank_config=Path("rank.json"), trace="fcc-test", trace_num=100,
            video="video1", device="cuda:0", fp16_numeric_safeguards=False,
            fp16_selective_clamp=False, fp16_attention_fp32_scores=False,
            nbs_compaction_rtol=0.05, nbs_compaction_atol=0.01,
        )
        for spec in pipeline.TARGET_SPECS:
            command = pipeline.sweep.best5.build_command(args, spec)
            self.assertNotIn("--fp16-attention-fp32-scores", command)
            self.assertNotIn("--fp16-numeric-safeguards", command)
            self.assertEqual(command[command.index("--data-seed") + 1], "3")
            self.assertEqual(command[command.index("--lora-seed") + 1], "1")
            self.assertEqual(
                command[command.index("--nbs-compaction-rtol") + 1], "0.05"
            )
            self.assertEqual(
                command[command.index("--nbs-compaction-atol") + 1], "0.01"
            )

    def test_summary_averages_two_trained_checkpoints(self):
        rows = []
        for spec in pipeline.TARGET_SPECS:
            for seed, reward, latency in ((1, 0.8, 80.0), (3, 1.0, 60.0)):
                rows.append({
                    "experiment": spec["name"],
                    "evaluation_data_seed": seed,
                    "mean_reward": reward,
                    "inference_latency_mean_ms": latency,
                })
        summary = pipeline.summarize(rows)
        self.assertAlmostEqual(summary[0]["mean_reward_mean"], 0.9)
        self.assertAlmostEqual(summary[0]["inference_latency_mean_ms_mean"], 70.0)
        self.assertEqual(summary[0]["num_trained_checkpoints"], 2)


if __name__ == "__main__":
    unittest.main()
