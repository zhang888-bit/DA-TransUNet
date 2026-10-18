import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_validation import build_validation_report, parse_test_log


class ValidationTests(unittest.TestCase):
    def test_parse_existing_test_log(self):
        parsed = parse_test_log()
        self.assertEqual(len(parsed["classes"]), 8)
        self.assertAlmostEqual(parsed["summary"]["mean_dice"], 0.766565, places=6)
        self.assertAlmostEqual(parsed["summary"]["mean_hd95"], 27.533803, places=6)

    def test_report_has_explicit_clinical_boundary(self):
        report = build_validation_report("transunet-r50-b16-v1")
        self.assertTrue(report["passed"])
        self.assertEqual(report["report_type"], "technical_retrospective_validation")
        self.assertEqual(report["clinical_validation_status"], "not_performed")
        self.assertEqual(report["intended_use"], "research_education_only")
        self.assertGreaterEqual(len(report["limitations"]), 3)


if __name__ == "__main__":
    unittest.main()
