import platform
import shutil
from datetime import datetime, timezone

import torch

from app_logging import summarize_events
from clinical_validation import load_latest_report
from model_registry import list_models, validate_model
from project_paths import PROJECT_ROOT


def get_system_health():
    models = list_models()
    valid_models = []
    for model in models:
        try:
            validate_model(model)
            valid_models.append(model.id)
        except (FileNotFoundError, ValueError):
            continue

    usage = shutil.disk_usage(PROJECT_ROOT)
    validation = load_latest_report()
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "registered_models": len(models),
        "valid_models": valid_models,
        "disk_free_gb": round(usage.free / (1024 ** 3), 2),
        "logging": summarize_events(limit=500),
        "validation": {
            "available": validation is not None,
            "passed": validation.get("passed") if validation else None,
            "mean_dice": validation.get("metrics", {}).get("mean_dice") if validation else None,
            "mean_hd95": validation.get("metrics", {}).get("mean_hd95") if validation else None,
            "clinical_validation_status": validation.get("clinical_validation_status") if validation else "not_available",
        },
    }
