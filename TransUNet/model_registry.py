import hashlib
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from project_paths import CONFIG_ROOT, PROJECT_ROOT


DEFAULT_REGISTRY_PATH = CONFIG_ROOT / "model_registry.json"


@dataclass(frozen=True)
class ModelVersion:
    id: str
    display_name: str
    checkpoint: Path
    config: str
    n_skip: int
    num_classes: int
    img_size: int
    dataset: str
    status: str
    metrics: dict
    validation_report: Path | None
    intended_use: str


def _resolve_registry_path(registry_path):
    return Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()


def load_registry(registry_path=None):
    path = _resolve_registry_path(registry_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_models(registry_path=None):
    registry = load_registry(registry_path)
    models = []
    for item in registry.get("models", []):
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        validation_report = item.get("validation_report")
        if validation_report:
            validation_report = Path(validation_report)
            if not validation_report.is_absolute():
                validation_report = PROJECT_ROOT / validation_report
        models.append(
            ModelVersion(
                id=item["id"],
                display_name=item.get("display_name", item["id"]),
                checkpoint=checkpoint.resolve(),
                config=item["config"],
                n_skip=int(item.get("n_skip", 3)),
                num_classes=int(item.get("num_classes", 9)),
                img_size=int(item.get("img_size", 224)),
                dataset=item.get("dataset", "Synapse"),
                status=item.get("status", "candidate"),
                metrics=dict(item.get("metrics", {})),
                validation_report=validation_report.resolve() if validation_report else None,
                intended_use=item.get("intended_use", "research_education_only"),
            )
        )
    return models


def get_model(model_id, registry_path=None):
    for model in list_models(registry_path):
        if model.id == model_id:
            return model
    raise KeyError("Unknown model id: {}".format(model_id))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(model):
    if not model.checkpoint.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(model.checkpoint))
    if model.num_classes < 1:
        raise ValueError("num_classes must be positive")
    if model.img_size < 16:
        raise ValueError("img_size must be at least 16")
    return True


def register_model(
    model_id,
    display_name,
    checkpoint,
    config,
    n_skip=3,
    num_classes=9,
    img_size=224,
    dataset="Synapse",
    status="candidate",
    metrics=None,
    validation_report=None,
    registry_path=None,
):
    path = _resolve_registry_path(registry_path)
    registry = load_registry(path)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    try:
        checkpoint_value = str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        checkpoint_value = str(checkpoint_path)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    entry = {
        "id": model_id,
        "display_name": display_name,
        "checkpoint": checkpoint_value,
        "config": config,
        "n_skip": int(n_skip),
        "num_classes": int(num_classes),
        "img_size": int(img_size),
        "dataset": dataset,
        "status": status,
        "registered_at": now,
        "file_size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "metrics": metrics or {},
        "validation_report": str(validation_report).replace("\\", "/") if validation_report else None,
        "intended_use": "research_education_only",
    }

    models = registry.setdefault("models", [])
    models = [item for item in models if item.get("id") != model_id]
    models.append(entry)
    registry["models"] = models
    registry["updated_at"] = now

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Manage registered TransUNet model versions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list registered models")

    register = subparsers.add_parser("register", help="register or update a checkpoint")
    register.add_argument("--id", required=True)
    register.add_argument("--display-name", required=True)
    register.add_argument("--checkpoint", required=True)
    register.add_argument("--config", default="R50-ViT-B_16")
    register.add_argument("--n-skip", type=int, default=3)
    register.add_argument("--num-classes", type=int, default=9)
    register.add_argument("--img-size", type=int, default=224)
    register.add_argument("--dataset", default="Synapse")
    register.add_argument("--status", default="candidate")
    register.add_argument("--metrics", default="{}", help="JSON object with evaluation metrics")
    register.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))

    args = parser.parse_args()
    if args.command == "list":
        for model in list_models():
            print(
                "{id} | {display_name} | {status} | {checkpoint}".format(
                    id=model.id,
                    display_name=model.display_name,
                    status=model.status,
                    checkpoint=model.checkpoint,
                )
            )
    elif args.command == "register":
        entry = register_model(
            model_id=args.id,
            display_name=args.display_name,
            checkpoint=args.checkpoint,
            config=args.config,
            n_skip=args.n_skip,
            num_classes=args.num_classes,
            img_size=args.img_size,
            dataset=args.dataset,
            status=args.status,
            metrics=json.loads(args.metrics),
            registry_path=args.registry,
        )
        print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
