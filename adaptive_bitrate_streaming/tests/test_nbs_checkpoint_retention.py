from pathlib import Path
import tempfile
import unittest

from adaptive_bitrate_streaming.plm_special.utils.checkpoints import (
    NBS_ADAPTER_FILES,
    NBS_REQUIRED_FILES,
    atomic_save_nbs_checkpoint,
    is_complete_nbs_checkpoint,
    prepare_best_latest_retention,
)


def write_complete(path, marker="complete"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in (*NBS_REQUIRED_FILES, NBS_ADAPTER_FILES[0]):
        (path / name).write_text(marker, encoding="utf-8")


class NBSCheckpointRetentionTest(unittest.TestCase):
    def test_corrupt_epoch_is_removed_and_newest_complete_becomes_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete(root / "0", "epoch0")
            write_complete(root / "6", "epoch6")
            (root / "8").mkdir()
            (root / "8" / "adapter_model.bin").write_text(
                "partial", encoding="utf-8"
            )
            (root / "nbs_rank_diagnostics.csv").write_text(
                "diagnostics", encoding="utf-8"
            )

            removed = prepare_best_latest_retention(root)

            self.assertEqual(removed, ["8"])
            self.assertTrue(is_complete_nbs_checkpoint(root / "latest"))
            self.assertEqual(
                (root / "latest" / "adapter_config.json").read_text(
                    encoding="utf-8"
                ),
                "epoch6",
            )
            self.assertFalse((root / "0").exists())
            self.assertTrue((root / "nbs_rank_diagnostics.csv").is_file())

    def test_failed_atomic_save_preserves_previous_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "latest"
            write_complete(target, "old")

            def incomplete_writer(path):
                path = Path(path)
                path.mkdir(parents=True)
                (path / "adapter_model.bin").write_text(
                    "partial", encoding="utf-8"
                )

            with self.assertRaises(RuntimeError):
                atomic_save_nbs_checkpoint(target, incomplete_writer)

            self.assertTrue(is_complete_nbs_checkpoint(target))
            self.assertEqual(
                (target / "adapter_config.json").read_text(encoding="utf-8"),
                "old",
            )

    def test_incomplete_latest_is_replaced_by_complete_numeric_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete(root / "6", "epoch6")
            (root / "latest").mkdir()
            (root / "latest" / "adapter_model.bin").write_text(
                "partial", encoding="utf-8"
            )
            removed = prepare_best_latest_retention(root)
            self.assertEqual(removed, ["latest"])
            self.assertTrue(is_complete_nbs_checkpoint(root / "latest"))

    def test_successful_atomic_save_replaces_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "latest"
            write_complete(target, "old")
            atomic_save_nbs_checkpoint(
                target, lambda path: write_complete(path, "new")
            )
            self.assertEqual(
                (target / "adapter_config.json").read_text(encoding="utf-8"),
                "new",
            )


if __name__ == "__main__":
    unittest.main()
