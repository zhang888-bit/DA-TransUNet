# Architecture

```text
Training data -> RandomGenerator -> TransUNet -> Dice/CE loss -> checkpoint
                                      |
                                      +-> training_manifest.json

Image upload -> PySide6 UI -> inference thread -> TransUNet
                                      |
                                      +-> overlay, organ metrics, PDF/TXT/JSON/HTML

Model registry -> UI model selection / checkpoint SHA256 / validation linkage
JSONL logs -> monitoring page / latency and error summaries
Test log -> technical validation report -> JSON and HTML
```

## Main Components

- `TransUNet/ui.py`: desktop workbench, asynchronous inference, monitoring, export.
- `TransUNet/networks/`: hybrid ResNet/Vision Transformer segmentation model.
- `TransUNet/train.py`, `trainer.py`: training loop and run manifest.
- `TransUNet/test.py`, `utils.py`: 3D volume evaluation and metrics.
- `TransUNet/model_registry.py`: model version registry.
- `TransUNet/app_logging.py`, `monitoring.py`: structured logs and health summary.
- `TransUNet/clinical_validation.py`: retrospective technical validation report.
- `TransUNet/report_export.py`: PDF, TXT, JSON, and HTML report generation.

## Data Boundary

Large datasets, checkpoints, generated overlays, and runtime logs are excluded
from Git. The public repository tracks source code, documentation, model
metadata, tests, and small screenshots only.
