import html
import json
import time
from pathlib import Path


def _model_name(report):
    return report.get("model_id") or Path(report.get("model_path", "unknown-model")).name


def _organ_status(percentage):
    return "检出" if percentage > 0.1 else "未检出"


def build_analysis_text(report):
    lines = [
        "腹部多器官分割分析报告",
        "=" * 32,
        f"生成时间：{report.get('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S'))}",
        f"模型：{_model_name(report)}",
        f"设备：{report.get('device', 'unknown')}",
        f"推理耗时：{report.get('elapsed_seconds', 0.0):.3f} 秒",
        f"输入影像：{report.get('input_image', '')}",
        "",
        "器官面积占比",
        "-" * 32,
    ]
    for organ in report.get("organ_metrics", []):
        lines.append(
            f"{organ['name']:<6}  {organ['percentage']:>6.2f}%  {_organ_status(organ['percentage'])}"
        )
    lines.extend(
        [
            "",
            "说明：",
            "本报告仅用于科研、教学与算法验证，不构成临床诊断或医疗器械建议。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_analysis_html(report):
    rows = "".join(
        "<tr><td>{}</td><td>{:.2f}%</td><td>{}</td></tr>".format(
            html.escape(item["name"]),
            item["percentage"],
            _organ_status(item["percentage"]),
        )
        for item in report.get("organ_metrics", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>腹部多器官分割分析报告</title>
  <style>
    body {{ font-family: "Microsoft YaHei", sans-serif; color: #1f2937; line-height: 1.55; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    h2 {{ font-size: 16px; margin-top: 22px; }}
    .meta {{ color: #526675; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #cfdae1; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef4f6; }}
    .notice {{ margin-top: 24px; padding: 12px; background: #fff7ed; border-left: 4px solid #d97706; }}
  </style>
</head>
<body>
  <h1>腹部多器官分割分析报告</h1>
  <p class="meta">生成时间：{html.escape(report.get("timestamp", ""))}</p>
  <h2>推理信息</h2>
  <p>模型：{html.escape(_model_name(report))}</p>
  <p>设备：{html.escape(str(report.get("device", "unknown")))}</p>
  <p>推理耗时：{report.get("elapsed_seconds", 0.0):.3f} 秒</p>
  <p>输入影像：{html.escape(str(report.get("input_image", "")))}</p>
  <h2>器官面积占比</h2>
  <table>
    <thead><tr><th>器官</th><th>占比</th><th>状态</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="notice">本报告仅用于科研、教学与算法验证，不构成临床诊断或医疗器械建议。</div>
</body>
</html>
"""


def write_pdf_report(report, path, font_family="Microsoft YaHei"):
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPdfWriter, QTextDocument

    writer = QPdfWriter(str(path))
    writer.setResolution(144)
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(14, 14, 14, 14), QPageLayout.Millimeter)

    document = QTextDocument()
    document.setDefaultFont(QFont(font_family, 10))
    document.setHtml(build_analysis_html(report))
    document.setPageSize(QSizeF(writer.width(), writer.height()))
    document.print_(writer)


def export_analysis_report(report, path):
    path = Path(path)
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".pdf")
        suffix = ".pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".pdf":
        write_pdf_report(report, path)
    elif suffix == ".txt":
        path.write_text(build_analysis_text(report), encoding="utf-8")
    elif suffix == ".html" or suffix == ".htm":
        path.write_text(build_analysis_html(report), encoding="utf-8")
    elif suffix == ".json":
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
    else:
        raise ValueError(f"Unsupported report format: {suffix}")
    return path
