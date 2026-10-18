import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from project_paths import PROJECT_ROOT, TRANSUNET_ROOT, VALIDATION_ROOT


DEFAULT_TEST_LOG = (
    TRANSUNET_ROOT
    / "test_log"
    / "test_log_TU_Synapse224"
    / "TU_pretrain_R50-ViT-B_16_skip3_bs24_224.txt"
)
DEFAULT_REPORT_PATH = VALIDATION_ROOT / "validation_report.json"
DEFAULT_REPORT_HTML = VALIDATION_ROOT / "validation_report.html"

CLASS_LINE = re.compile(
    r"Mean class (?P<label>\d+) mean_dice (?P<dice>[0-9.]+) mean_hd95 (?P<hd95>[0-9.]+)"
)
SUMMARY_LINE = re.compile(
    r"mean_dice : (?P<dice>[0-9.]+) mean_hd95 : (?P<hd95>[0-9.]+)"
)


def parse_test_log(log_path=None):
    path = Path(log_path or DEFAULT_TEST_LOG)
    classes = {}
    summary = None
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            class_match = CLASS_LINE.search(line)
            if class_match:
                label = int(class_match.group("label"))
                classes[label] = {
                    "class_index": label,
                    "dice": float(class_match.group("dice")),
                    "hd95": float(class_match.group("hd95")),
                }
                continue
            summary_match = SUMMARY_LINE.search(line)
            if summary_match:
                summary = {
                    "mean_dice": float(summary_match.group("dice")),
                    "mean_hd95": float(summary_match.group("hd95")),
                }
    if summary is None:
        raise ValueError("No summary metrics found in {}".format(path))
    return {"summary": summary, "classes": [classes[key] for key in sorted(classes)]}


def build_validation_report(model_id, log_path=None, dice_threshold=0.70, hd95_threshold=30.0):
    path = Path(log_path or DEFAULT_TEST_LOG)
    try:
        source = str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        source = str(path)
    parsed = parse_test_log(path)
    summary = parsed["summary"]
    checks = {
        "mean_dice_at_least_{}".format(dice_threshold): summary["mean_dice"] >= dice_threshold,
        "mean_hd95_at_most_{}".format(hd95_threshold): summary["mean_hd95"] <= hd95_threshold,
    }
    class_at_least_050 = all(item["dice"] >= 0.50 for item in parsed["classes"])
    checks["each_class_dice_at_least_0.50"] = class_at_least_050

    return {
        "schema_version": 1,
        "report_type": "technical_retrospective_validation",
        "clinical_validation_status": "not_performed",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model_id": model_id,
        "dataset": "Synapse",
        "source": source,
        "metrics": summary,
        "class_metrics": parsed["classes"],
        "thresholds": {
            "mean_dice_min": dice_threshold,
            "mean_hd95_max": hd95_threshold,
            "per_class_dice_min": 0.50,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "limitations": [
            "This is a retrospective technical validation on a public research dataset.",
            "It is not a prospective clinical trial and does not establish diagnostic safety or efficacy.",
            "External-site validation, reader studies, and regulatory review have not been performed.",
        ],
        "intended_use": "research_education_only",
    }


def write_report(report, json_path=None, html_path=None):
    json_path = Path(json_path or DEFAULT_REPORT_PATH)
    html_path = Path(html_path or DEFAULT_REPORT_HTML)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path


def load_latest_report(report_path=None):
    path = Path(report_path or DEFAULT_REPORT_PATH)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_html(report):
    class_rows = "".join(
        "<tr><td>{class_index}</td><td>{dice:.4f}</td><td>{hd95:.2f}</td></tr>".format(**item)
        for item in report["class_metrics"]
    )
    check_rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(name, "通过" if passed else "未通过")
        for name, passed in report["checks"].items()
    )
    limitations = "".join("<li>{}</li>".format(item) for item in report["limitations"])
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>TransUNet 技术验证报告</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; margin: 36px; color: #1f2937; }
    h1 { font-size: 26px; margin-bottom: 8px; }
    .notice { background: #fff7ed; border-left: 4px solid #d97706; padding: 12px 16px; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; }
    th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }
    th { background: #f3f4f6; }
  </style>
</head>
<body>
  <h1>TransUNet 技术验证报告</h1>
  <p>模型：__MODEL_ID__　数据集：__DATASET__</p>
  <div class="notice">该报告为公开数据集的回顾性技术验证，不构成临床验证或诊断依据。</div>
  <h2>总体指标</h2>
  <p>Mean Dice: __MEAN_DICE__　Mean HD95: __MEAN_HD95__　结论：__PASSED__</p>
  <h2>分类指标</h2>
  <table><thead><tr><th>类别</th><th>Dice</th><th>HD95</th></tr></thead><tbody>__CLASS_ROWS__</tbody></table>
  <h2>验收检查</h2>
  <table><thead><tr><th>检查项</th><th>结果</th></tr></thead><tbody>__CHECK_ROWS__</tbody></table>
  <h2>适用范围与限制</h2>
  <ul>__LIMITATIONS__</ul>
</body>
</html>
"""
    replacements = {
        "__MODEL_ID__": report["model_id"],
        "__DATASET__": report["dataset"],
        "__MEAN_DICE__": "{:.4f}".format(report["metrics"]["mean_dice"]),
        "__MEAN_HD95__": "{:.2f}".format(report["metrics"]["mean_hd95"]),
        "__PASSED__": "通过" if report["passed"] else "未通过",
        "__CLASS_ROWS__": class_rows,
        "__CHECK_ROWS__": check_rows,
        "__LIMITATIONS__": limitations,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def main():
    parser = argparse.ArgumentParser(description="Build a retrospective technical validation report")
    parser.add_argument("--model-id", default="transunet-r50-b16-v1")
    parser.add_argument("--test-log", default=str(DEFAULT_TEST_LOG))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--html-output", default=str(DEFAULT_REPORT_HTML))
    args = parser.parse_args()

    report = build_validation_report(args.model_id, log_path=args.test_log)
    json_path, html_path = write_report(report, args.output, args.html_output)
    print("Validation report: {}".format(json_path))
    print("HTML report: {}".format(html_path))
    print("Passed: {}".format(report["passed"]))


if __name__ == "__main__":
    main()
