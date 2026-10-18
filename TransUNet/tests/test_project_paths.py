import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import DEFAULT_CHECKPOINT, build_snapshot_path


class ProjectPathTests(unittest.TestCase):
    def test_default_checkpoint_exists(self):
        if not DEFAULT_CHECKPOINT.exists():
            self.skipTest("Large model checkpoint is intentionally not committed")
        self.assertTrue(DEFAULT_CHECKPOINT.exists())

    def test_default_snapshot_name_matches_existing_checkpoint_directory(self):
        args = SimpleNamespace(
            dataset="Synapse",
            img_size=224,
            vit_name="R50-ViT-B_16",
            n_skip=3,
            vit_patches_size=16,
            max_epochs=150,
            batch_size=24,
            base_lr=0.01,
            seed=1234,
        )
        snapshot = build_snapshot_path(args)
        self.assertEqual(snapshot.name, DEFAULT_CHECKPOINT.parent.name)


if __name__ == "__main__":
    unittest.main()
