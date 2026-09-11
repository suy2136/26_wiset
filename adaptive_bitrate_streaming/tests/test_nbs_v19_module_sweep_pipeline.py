import argparse
import importlib.util
from pathlib import Path
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / "analysis" / "run_nbs_v19_module_sweep_pipeline.py"
SPEC = importlib.util.spec_from_file_location("module_sweep", SCRIPT)
module_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module_sweep)


def arguments():
    return argparse.Namespace(
        checkpoint_dir=Path("checkpoint"), base_model_dir=Path("base"),
        exp_pool_path=Path("pool.pkl"), rank_budget=1536, physical_rank=32,
        rank_config=Path("configs/nbs_v19_rank_config.json"),
        trace="fcc-test", trace_num=100, video="video1", device="cuda:0",
    )


def option(command, name):
    return command[command.index(name) + 1]


class ModuleSweepPipelineTest(unittest.TestCase):
    def test_independent_matrix_size_and_isolation(self):
        specs = module_sweep.independent_specs()
        self.assertEqual(len(specs), 18)
        for spec in specs:
            args = arguments()
            args.data_seed = 1
            command = module_sweep.best5.build_command(args, spec)
            enabled = sum((
                option(command, "--temporal-selector") != "none",
                option(command, "--token-selector") != "none",
                option(command, "--speculative-draft-steps") != "0",
            ))
            self.assertEqual(enabled, 0 if spec["module"] == "baseline" else 1)
            self.assertIn("--nbs-compact-inference", command)

    def test_dynamic_parameters_reach_command(self):
        spec = {
            "name": "combined", "module": "combined", "temporal": True,
            "event_max_events": 4, "token": True,
            "token_offsets": (0, 2, 3, 7), "speculative": True,
            "speculative_steps": 5, "speculative_drafter": "repeat-last",
        }
        args = arguments()
        args.data_seed = 3
        command = module_sweep.best5.build_command(args, spec)
        self.assertEqual(option(command, "--event-max-events"), "4")
        self.assertEqual(option(command, "--speculative-draft-steps"), "5")
        start = command.index("--intra-token-keep-offsets") + 1
        self.assertEqual(command[start:start + 4], ["0", "2", "3", "7"])
        self.assertEqual(option(command, "--data-seed"), "3")

    def test_representatives_make_eight_combinations(self):
        specs = module_sweep.independent_specs()
        rows = []
        for index, spec in enumerate(specs):
            rows.append({
                "experiment": spec["name"], "mean_reward": 0.8 + index / 1000,
                "inference_latency_mean_ms": 80.0 - index,
                "llm_call_reduction_ratio": index / 100,
            })
        selected = {
            module: module_sweep.representative_specs(specs, rows, module)
            for module in ("temporal", "token", "speculative")
        }
        self.assertTrue(all(len(value) == 2 for value in selected.values()))
        combined = module_sweep.combined_specs(selected)
        self.assertEqual(len(combined), 8)
        self.assertEqual(len({spec["name"] for spec in combined}), 8)

    def test_finalists_are_unique_and_confirmation_aggregates(self):
        selected = {
            "temporal": [
                {"name": "t1", "event_max_events": 1},
                {"name": "t2", "event_max_events": 2},
            ],
            "token": [
                {"name": "o1", "token_offsets": (0, 7)},
                {"name": "o2", "token_offsets": (2, 6, 7)},
            ],
            "speculative": [
                {"name": "s1", "speculative_steps": 1},
                {"name": "s2", "speculative_steps": 3},
            ],
        }
        specs = module_sweep.combined_specs(selected)
        rows = [{
            "experiment": spec["name"], "mean_reward": 0.80 + index * 0.01,
            "inference_latency_mean_ms": 60.0 - index,
        } for index, spec in enumerate(specs)]
        finalists = module_sweep.combined_finalists(specs, rows, 0.8, 3)
        self.assertEqual(len(finalists), 3)
        self.assertEqual(len({spec["name"] for spec in finalists}), 3)

        confirmation = []
        names = ["nbs_compact_only", *[spec["name"] for spec in finalists]]
        for name in names:
            for seed in (1, 2, 3):
                confirmation.append({
                    "experiment": name, "data_seed": seed,
                    "mean_reward": 0.8 + seed * 0.01,
                    "inference_latency_mean_ms": 60 + seed,
                })
        summary = module_sweep.aggregate_confirmation(
            confirmation, [spec["name"] for spec in finalists]
        )
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(row["num_seeds"] == 3 for row in summary))


if __name__ == "__main__":
    unittest.main()
