import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_logging import configure_logging, log_event, read_events, summarize_events


class LoggingTests(unittest.TestCase):
    def test_json_log_can_be_summarized(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_file = Path(temporary_dir) / "application.jsonl"
            logger = configure_logging(name="transunet-test", log_file=log_file)
            log_event(logger, "inference_completed", "done", duration_ms=123.5)
            logger.handlers.clear()

            events = read_events(log_file)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "inference_completed")
            summary = summarize_events(log_file)
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["average_duration_ms"], 123.5)
            self.assertEqual(summary["average_inference_duration_ms"], 123.5)


if __name__ == "__main__":
    unittest.main()
