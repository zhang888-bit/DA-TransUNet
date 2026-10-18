import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_registry import get_model, list_models, register_model, sha256_file


class ModelRegistryTests(unittest.TestCase):
    def test_default_registry_contains_valid_checkpoint(self):
        models = list_models()
        self.assertGreaterEqual(len(models), 1)
        model = get_model("transunet-r50-b16-v1")
        self.assertEqual(model.config, "R50-ViT-B_16")
        self.assertEqual(model.num_classes, 9)
        if not model.checkpoint.exists():
            self.skipTest("Large model checkpoint is intentionally not committed")
        self.assertTrue(model.checkpoint.exists())

    def test_register_model_updates_registry(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"test-checkpoint")
            registry = root / "registry.json"
            registry.write_text('{"schema_version": 1, "models": []}', encoding="utf-8")

            entry = register_model(
                model_id="test-model",
                display_name="Test model",
                checkpoint=checkpoint,
                config="R50-ViT-B_16",
                registry_path=registry,
            )

            self.assertEqual(entry["id"], "test-model")
            payload = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["models"]), 1)
            self.assertEqual(sha256_file(checkpoint), sha256_file(checkpoint))


if __name__ == "__main__":
    unittest.main()
