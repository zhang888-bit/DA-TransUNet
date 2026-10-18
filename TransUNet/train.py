import argparse
import copy
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from project_paths import (
    SYNAPSE_LIST_ROOT,
    SYNAPSE_TRAIN_ROOT,
    build_snapshot_path,
    resolve_path,
    resolve_torch_device,
    resolve_transunet_path,
)
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=str(SYNAPSE_TRAIN_ROOT), help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default=str(SYNAPSE_LIST_ROOT), help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration cap; 0 means no extra cap')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--device', type=str, default='auto',
                    help='runtime device: auto, cpu, cuda, cuda:0, ...')
args = parser.parse_args()


if __name__ == "__main__":
    if args.dataset != 'Synapse':
        parser.error("This project currently supports only --dataset Synapse")

    args.root_path = str(resolve_path(args.root_path))
    args.list_dir = str(resolve_path(args.list_dir))
    args.device = resolve_torch_device(args.device)
    args.is_pretrain = True

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

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    snapshot_path = str(build_snapshot_path(args))

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = copy.deepcopy(CONFIGS_ViT_seg[args.vit_name])
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(args.device)
    pretrained_path = resolve_transunet_path(config_vit.pretrained_path)
    if not pretrained_path.exists():
        raise FileNotFoundError("Pretrained checkpoint not found: {}".format(pretrained_path))
    net.load_from(weights=np.load(pretrained_path))

    trainer = {'Synapse': trainer_synapse,}
    trainer[args.dataset](args, net, snapshot_path)
