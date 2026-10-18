# Productization Additions

## Model Version Registry

`configs/model_registry.json` records the active checkpoint, architecture,
input size, class count, technical metrics, checksum, status, and validation
report path.

List models:

```bash
python TransUNet/model_registry.py list
```

Register a checkpoint:

```bash
python TransUNet/model_registry.py register \
  --id transunet-candidate-v2 \
  --display-name "TransUNet candidate v2" \
  --checkpoint model/example/latest.pth \
  --status candidate \
  --metrics "{\"mean_dice\": 0.78}"
```

## Structured Logging and Monitoring

Application events are written as JSON Lines to `logs/application.jsonl`.
The desktop monitoring page summarizes model loading, inference latency,
errors, recent events, registered models, and validation status.

## Automated Tests

Run the standard-library test suite:

```bash
python TransUNet/run_tests.py
```

The suite covers model registry operations, path consistency, metric behavior,
validation report generation, and structured logging.

## Validation Report

Generate a report from the fixed Synapse test log:

```bash
python TransUNet/clinical_validation.py
```

Outputs:

- `TransUNet/validation/validation_report.json`
- `TransUNet/validation/validation_report.html`

The report is deliberately labeled as retrospective technical validation.
It does not claim clinical validation, regulatory approval, or diagnostic
safety.

## Analysis Report Export

The desktop UI exports completed inference results as PDF, TXT, JSON, or HTML.
PDF is the default format and uses Windows-compatible Chinese fonts.

## Training Manifest

Each training run writes `training_manifest.json` into its snapshot directory.
The manifest records the architecture, data paths, seed, optimizer, loss,
device, iteration count, checkpoint, and intended use.
