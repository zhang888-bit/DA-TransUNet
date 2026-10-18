import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from project_paths import LOG_ROOT


APP_LOG_FILE = LOG_ROOT / "application.jsonl"


class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(name="transunet", log_file=None, level=logging.INFO):
    app_log_file = Path(log_file or APP_LOG_FILE)
    app_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if any(getattr(handler, "_transunet_json", False) for handler in logger.handlers):
        return logger

    file_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonLineFormatter())
    file_handler._transunet_json = True

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    stream_handler._transunet_json = True

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_event(logger, event, message="", level=logging.INFO, **fields):
    logger.log(level, message, extra={"event": event, "fields": fields})


def read_events(log_file=None, limit=None):
    app_log_file = Path(log_file or APP_LOG_FILE)
    if not app_log_file.exists():
        return []

    events = []
    with app_log_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events[-limit:] if limit else events


def summarize_events(log_file=None, limit=500):
    events = read_events(log_file=log_file, limit=limit)
    event_counts = Counter(item.get("event", "unknown") for item in events)
    latencies = []
    inference_latencies = []
    model_load_latencies = []
    for item in events:
        latency = item.get("fields", {}).get("duration_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
            if item.get("event") == "inference_completed":
                inference_latencies.append(float(latency))
            elif item.get("event") == "model_load_completed":
                model_load_latencies.append(float(latency))

    errors = [item for item in events if item.get("level") in {"ERROR", "CRITICAL"}]
    return {
        "event_count": len(events),
        "error_count": len(errors),
        "event_counts": dict(event_counts),
        "average_duration_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "average_inference_duration_ms": (
            round(sum(inference_latencies) / len(inference_latencies), 2)
            if inference_latencies else None
        ),
        "average_model_load_duration_ms": (
            round(sum(model_load_latencies) / len(model_load_latencies), 2)
            if model_load_latencies else None
        ),
        "last_event_at": events[-1].get("timestamp") if events else None,
        "recent_events": events[-20:],
    }
