from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSUNET_ROOT = PROJECT_ROOT / "TransUNet"
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "model"
UI_ROOT = TRANSUNET_ROOT / "images"
UI_ASSET_ROOT = UI_ROOT / "UI"
UI_TMP_ROOT = UI_ROOT / "tmp"
UI_RESULT_ROOT = TRANSUNET_ROOT / "results_ui"
PREDICTION_ROOT = PROJECT_ROOT / "predictions"
CONFIG_ROOT = TRANSUNET_ROOT / "configs"
LOG_ROOT = PROJECT_ROOT / "logs"
VALIDATION_ROOT = TRANSUNET_ROOT / "validation"

SYNAPSE_DATA_ROOT = DATA_ROOT / "Synapse"
SYNAPSE_TRAIN_ROOT = SYNAPSE_DATA_ROOT / "train_npz"
SYNAPSE_TEST_ROOT = SYNAPSE_DATA_ROOT / "test_vol_h5"
SYNAPSE_LIST_ROOT = TRANSUNET_ROOT / "lists" / "lists_Synapse"

DEFAULT_VIT_NAME = "R50-ViT-B_16"
DEFAULT_CHECKPOINT = (
    MODEL_ROOT
    / "TU_Synapse224"
    / "TU_pretrain_R50-ViT-B_16_skip3_bs24_224"
    / "epoch_149.pth"
)


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def resolve_transunet_path(path):
    return resolve_path(path, base=TRANSUNET_ROOT)


def resolve_torch_device(requested="auto"):
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build has no available CUDA device.")
    return device


def build_snapshot_path(args):
    exp = "TU_" + args.dataset + str(args.img_size)
    name = "TU_pretrain_" + args.vit_name + "_skip" + str(args.n_skip)
    if args.vit_patches_size != 16:
        name += "_vitpatch" + str(args.vit_patches_size)
    if args.max_epochs != 150:
        name += "_epo" + str(args.max_epochs)
    name += "_bs" + str(args.batch_size)
    if args.base_lr != 0.01:
        name += "_lr" + str(args.base_lr)
    name += "_" + str(args.img_size)
    if args.seed != 1234:
        name += "_s" + str(args.seed)
    return MODEL_ROOT / exp / name
