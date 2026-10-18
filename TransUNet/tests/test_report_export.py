import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_export import build_analysis_html, build_analysis_text, export_analysis_report


class ReportExportTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "timestamp": "2026-10-17T22:00:00",
            "model_id": "transunet-r50-b16-v1",
            "model_path": "model/checkpoint.pth",
            "device": "cpu",
            "elapsed_seconds": 0.321,
            "input_image": "case.jpg",
            "organ_metrics": [
                {"name": "肝脏", "percentage": 8.83, "color": [244, 35, 232]},
                {"name": "右肾", "percentage": 0.0, "color": [0, 0, 142]},
            ],
        }

    def test_text_report_contains_metrics_and_disclaimer(self):
        content = build_analysis_text(self.report)
        self.assertIn("腹部多器官分割分析报告", content)
        self.assertIn("肝脏", content)
        self.assertIn("8.83%", content)
        self.assertIn("不构成临床诊断", content)

    def test_html_report_contains_organ_table(self):
        content = build_analysis_html(self.report)
        self.assertIn("<table>", content)
        self.assertIn("右肾", content)
        self.assertIn("未检出", content)

    def test_txt_and_json_export(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            txt_path = export_analysis_report(self.report, root / "report.txt")
            json_path = export_analysis_report(self.report, root / "report.json")
            self.assertTrue(txt_path.exists())
            self.assertIn("肝脏", txt_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "transunet-r50-b16-v1")


if __name__ == "__main__":
    unittest.main()
