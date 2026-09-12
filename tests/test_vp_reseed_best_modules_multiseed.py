import argparse
from pathlib import Path
import unittest

from analysis import evaluate_vp_reseed_best_modules_multiseed as pipeline


def args():
    return argparse.Namespace(
        compact_checkpoint=Path("compact"), tuned_projector=Path("projector"),
        projector_checkpoint=Path("projector"), cache_dir=Path("cache"),
        device="cuda:0", physical_rank=32, rank_budget=512,
        latency_warmup_steps=5, projector_cache_max_entries=512,
    )


class VPReseedBestModulesTest(unittest.TestCase):
    def test_exact_four_configurations(self):
        self.assertEqual(len(pipeline.CONFIGURATIONS), 4)
        self.assertEqual(pipeline.PATCH_CONFIG["policy"], "gated-k1")
        self.assertEqual(pipeline.PATCH_CONFIG["threshold"], 6.0)
        self.assertEqual(pipeline.PATCH_CONFIG["max_skip"], 0)
        self.assertEqual(pipeline.TOKEN_K, 8)

    def test_every_command_uses_per_episode_reseeding(self):
        for _, _, kind in pipeline.CONFIGURATIONS:
            command = pipeline.build_command(args(), kind, Path("result"), 3)
            self.assertEqual(
                command[command.index("--evaluation-rng-mode") + 1],
                "per-episode",
            )
            self.assertEqual(command[command.index("--data-seed") + 1], "3")
            self.assertEqual(command[command.index("--lora-seed") + 1], "1")

    def test_full_stack_keeps_requested_parameters(self):
        command = pipeline.build_command(args(), "full_stack", Path("result"), 1)
        self.assertEqual(command[command.index("--cached-patch-policy") + 1], "gated-k1")
        self.assertEqual(command[command.index("--cached-patch-motion-threshold-deg") + 1], "6.0")
        self.assertEqual(command[command.index("--cached-patch-max-skip-calls") + 1], "0")
        self.assertEqual(command[command.index("--selector-recent-k") + 1], "8")
        self.assertEqual(command[command.index("--speculative-gamma") + 1], "6")
        self.assertEqual(command[command.index("--speculative-threshold") + 1], "0.4")


if __name__ == "__main__":
    unittest.main()
