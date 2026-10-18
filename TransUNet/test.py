import argparse
import copy
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from project_paths import (
    PREDICTION_ROOT,
    SYNAPSE_LIST_ROOT,
    SYNAPSE_TEST_ROOT,
    TRANSUNET_ROOT,
    build_snapshot_path,
    resolve_path,
    resolve_torch_device,
)

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default=str(SYNAPSE_TEST_ROOT), help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default=str(SYNAPSE_LIST_ROOT), help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='epoch count used to locate the checkpoint')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default=str(PREDICTION_ROOT), help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='explicit checkpoint path; overrides checkpoint discovery')
parser.add_argument('--device', type=str, default='auto',
                    help='runtime device: auto, cpu, cuda, cuda:0, ...')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if args.dataset != 'Synapse':
        parser.error("This project currently supports only --dataset Synapse")

    args.volume_path = str(resolve_path(args.volume_path))
    args.list_dir = str(resolve_path(args.list_dir))
    args.device = resolve_torch_device(args.device)

    if args.device.type == "cuda" and not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    snapshot_path = build_snapshot_path(args)
    args.exp = snapshot_path.parent.name

    if args.checkpoint:
        snapshot = resolve_path(args.checkpoint)
    else:
        candidates = [
            snapshot_path / "latest.pth",
            snapshot_path / ("epoch_" + str(args.max_epochs - 1) + ".pth"),
            snapshot_path / "epoch_149.pth",
        ]
        snapshot = next((candidate for candidate in candidates if candidate.exists()), None)
        if snapshot is None:
            raise FileNotFoundError(
                "No checkpoint found for {}. Tried: {}".format(
                    snapshot_path, ", ".join(str(candidate.name) for candidate in candidates)
                )
            )

    config_vit = copy.deepcopy(CONFIGS_ViT_seg[args.vit_name])
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(args.device)
    net.load_state_dict(torch.load(snapshot, map_location=args.device, weights_only=True))
    snapshot_name = snapshot_path.name

    log_folder = TRANSUNET_ROOT / 'test_log' / ('test_log_' + args.exp)
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=str(log_folder / (snapshot_name + ".txt")), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = str(resolve_path(args.test_save_dir))
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

    print("Snapshot path is:", snapshot_path)
