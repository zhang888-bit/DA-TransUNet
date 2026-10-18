# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-10-18

### Added

- Desktop workbench for abdominal multi-organ segmentation.
- PyTorch training and 3D volume evaluation entry points.
- Model version registry with checkpoint SHA256 metadata.
- Structured JSONL application logs and monitoring page.
- Automated tests for paths, metrics, validation, registry, logging, and exports.
- PDF, TXT, JSON, and HTML analysis report export.
- Retrospective technical validation report with clinical-use boundary.
- PyCharm shared run configurations.
- GitHub CI, contribution, security, and release documentation.

### Fixed

- Checkpoint path and device handling across CPU/GPU environments.
- False-positive Dice calculation.
- Missing configuration fields for alternate model variants.
- Shared decoder skip-channel configuration mutation.
- Vertical desktop navigation labels replaced with readable horizontal tabs.
