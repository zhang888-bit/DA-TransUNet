# TransUNet Module

> The primary project documentation is in
> [`../README.md`](../README.md). This file keeps the original TransUNet usage
> notes and dataset references.

This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Python 3.10 or newer is recommended. Install the dependencies with:

```bash
python -m pip install -r TransUNet/requirements.txt
```

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory, and both can reach similar performance.

```bash
python TransUNet/train.py --dataset Synapse --vit_name R50-ViT-B_16 --device auto
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python TransUNet/test.py --dataset Synapse --vit_name R50-ViT-B_16 --device auto
```

- Start the desktop segmentation UI with the bundled checkpoint:

```bash
python TransUNet/ui.py --device auto
```

All entry points resolve data, checkpoints, and UI assets from the project root,
so they can be started from either the repository root or PyCharm.

### 5. Productization Workflow

- Model versions are registered in `TransUNet/configs/model_registry.json`.
- Structured JSON logs are stored in `logs/application.jsonl`.
- The desktop UI includes a system-monitoring page and report export.
- Automated tests run with `python TransUNet/run_tests.py`.
- Technical validation reports are generated with
  `python TransUNet/clinical_validation.py`.
- Additional implementation notes are in `TransUNet/docs/PRODUCTIZATION.md`.

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
