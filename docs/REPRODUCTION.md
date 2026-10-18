# Reproduction Guide

## 1. Install

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install -r TransUNet/requirements.txt
```

## 2. Data

Prepare the Synapse data using `TransUNet/datasets/README.md`.

Expected paths:

```text
data/Synapse/train_npz/
data/Synapse/test_vol_h5/
```

The dataset is not redistributed by this repository.

## 3. Checkpoint

Place a compatible checkpoint at:

```text
model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_bs24_224/epoch_149.pth
```

Or register another checkpoint:

```bash
python TransUNet/model_registry.py register \
  --id transunet-candidate \
  --display-name "TransUNet candidate" \
  --checkpoint path/to/checkpoint.pth
```

## 4. Verify

```bash
python TransUNet/run_tests.py
python TransUNet/clinical_validation.py
python TransUNet/ui.py --device auto
```

## 5. Smoke Test

```bash
python TransUNet/ui.py \
  --device auto \
  --smoke-test-image data/Synapse/test_vol_h5_jpg_png/case0022/images/slice_070.jpg
```

## 6. Train and Test

```bash
python TransUNet/train.py --dataset Synapse --vit_name R50-ViT-B_16 --device auto
python TransUNet/test.py --dataset Synapse --vit_name R50-ViT-B_16 --device auto
```
