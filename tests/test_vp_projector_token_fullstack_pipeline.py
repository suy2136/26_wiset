import argparse
from pathlib import Path
import unittest

from analysis import run_vp_projector_token_fullstack_pipeline as pipeline


def args():
    return argparse.Namespace(
        nbs_checkpoint=Path("nbs"), compact_checkpoint=Path("compact"),
        initial_projector=Path("initial"), cache_dir=Path("cache"),
        projector_checkpoint=Path("initial"), device="cuda:0",
        rank_budget=512, physical_rank=32, validation_samples=128,
        token_mae_tolerance_deg=0.1, latency_warmup_steps=5,
        projector_cache_max_entries=512,
    )


class VPProjectorTokenPipelineTest(unittest.TestCase):
    def test_training_candidate_matrix(self):
        self.assertEqual(pipeline.TRAINING_CANDIDATES, (
            ("projector_2ep_lr5e-5", 2, 5e-5),
            ("projector_2ep_lr2e-5", 2, 2e-5),
            ("projector_3ep_lr2e-5", 3, 2e-5),
        ))
        self.assertEqual(pipeline.TOKEN_K_VALUES, (2, 4, 6, 8, 10))
        self.assertTrue(all(
            value <= pipeline.VP_HISTORY_LENGTH
            for value in pipeline.TOKEN_K_VALUES
        ))

    def test_token_sweep_uses_validation(self):
        command = pipeline.full_command(
            args(), Path("projector"), Path("result"), 10, "valid", 1
        )
        self.assertEqual(command[command.index("--evaluation-split") + 1], "valid")
        self.assertEqual(command[command.index("--selector-recent-k") + 1], "10")
        self.assertEqual(command[command.index("--speculative-gamma") + 1], "6")
        self.assertEqual(command[command.index("--speculative-threshold") + 1], "0.4")

    def test_final_full_stack_uses_selected_projector_and_k(self):
        command = pipeline.final_command(
            args(), "full_stack", Path("best_projector.pth"), 12,
            Path("result"), 3,
        )
        self.assertEqual(command[command.index("--evaluation-split") + 1], "test")
        self.assertEqual(command[command.index("--selector-recent-k") + 1], "12")
        self.assertEqual(command[command.index("--data-seed") + 1], "3")
        self.assertEqual(
            command[command.index("--multimodal-projector-checkpoint") + 1],
            "best_projector.pth",
        )

    def test_final_comparison_has_five_cases(self):
        self.assertEqual([item[2] for item in pipeline.FINAL_CASES], [
            "baseline", "patch", "token", "speculative", "full_stack",
        ])


if __name__ == "__main__":
    unittest.main()
