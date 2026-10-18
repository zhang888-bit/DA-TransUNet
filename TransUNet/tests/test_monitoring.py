import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring import get_system_health


class MonitoringTests(unittest.TestCase):
    def test_health_contains_registry_logging_and_validation(self):
        health = get_system_health()
        self.assertIn("torch", health)
        self.assertGreaterEqual(health["registered_models"], 1)
        self.assertIn("logging", health)
        self.assertIn("validation", health)
        self.assertTrue(health["validation"]["available"])


if __name__ == "__main__":
    unittest.main()
