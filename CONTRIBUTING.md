# Contributing

## Development Setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install runtime dependencies:

```bash
python -m pip install -r TransUNet/requirements.txt
```

4. Run the automated tests:

```bash
python TransUNet/run_tests.py
```

5. Run a single-image UI smoke test:

```bash
python TransUNet/ui.py --device auto --smoke-test-image path/to/image.jpg
```

## Pull Requests

- Keep changes focused and describe the problem being solved.
- Add or update tests for behavior changes.
- Run `python -m compileall -q TransUNet` and `python TransUNet/run_tests.py`.
- Do not commit datasets, model weights, generated predictions, logs, or secrets.
- Do not claim clinical validation without an approved study and evidence.
- Update documentation when behavior or public interfaces change.

## Medical Data Policy

Public research data only. Contributors must not submit identifiable patient
data, private DICOM studies, clinical notes, access tokens, or credentials.
