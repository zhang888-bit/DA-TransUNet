import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_image(self, *args, **kwargs):
                pass

            def close(self):
                pass
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


def _write_training_manifest(snapshot_path, args, status, started_at, checkpoint=None, iterations=None):
    payload = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "iterations_completed": iterations,
        "intended_use": "research_education_only",
        "training_config": {
            "dataset": args.dataset,
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "vit_name": args.vit_name,
            "n_skip": args.n_skip,
            "num_classes": args.num_classes,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "n_gpu": args.n_gpu,
            "max_epochs": args.max_epochs,
            "max_iterations": args.max_iterations,
            "base_lr": args.base_lr,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "device": str(getattr(args, "device", "unknown")),
            "loss": "0.5 * CrossEntropyLoss + 0.5 * DiceLoss",
            "optimizer": "SGD(momentum=0.9, weight_decay=0.0001)",
        },
    }
    path = os.path.join(snapshot_path, "training_manifest.json")
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    _write_training_manifest(snapshot_path, args, "running", started_at)
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    device = torch.device(getattr(args, "device", "cuda"))
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    if getattr(args, "max_iterations", 0) > 0:
        max_iterations = min(max_iterations, args.max_iterations)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                visual_index = 1 if image_batch.size(0) > 1 else 0
                image = image_batch[visual_index, 0:1, :, :]
                image_range = image.max() - image.min()
                if image_range > 0:
                    image = (image - image.min()) / image_range
                else:
                    image = torch.zeros_like(image)
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[visual_index, ...] * 50, iter_num)
                labs = label_batch[visual_index, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num >= max_iterations:
                break

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or iter_num >= max_iterations:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_mode_path)
            torch.save(state_dict, os.path.join(snapshot_path, 'latest.pth'))
            _write_training_manifest(
                snapshot_path,
                args,
                "completed",
                started_at,
                checkpoint=os.path.join(snapshot_path, 'latest.pth'),
                iterations=iter_num,
            )
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
