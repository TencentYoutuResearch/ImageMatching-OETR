#!/usr/bin/env python
"""
@File    :   train.py
@Time    :   2021/06/29 17:19:36
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import argparse
import datetime
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config.default import get_cfg_defaults
from src.datasets import build_dataloader
from src.model import build_detectors
from src.utils.utils import (get_logger, loss_info,
                             visualize_centerness_overlap_gt,
                             visualize_overlap_gt)
from src.utils.validation import evaluate

# os.environ["OMP_NUM_THREADS"] = "1"
# seed = 42


def setup_seed(seed):
    """set random seed to protect the trainning results.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(opt):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config_path)
    setup_seed(opt.seed)
    # output folder init
    timestamp = datetime.datetime.now().strftime('%m-%d-%H:%M')
    if opt.debug:
        opt.save_path = Path(f'./OUTPUT/OETR/debug/{cfg.OUTPUT}' + timestamp)
    else:
        opt.save_path = Path(f'./OUTPUT/OETR/checkpoints/{cfg.OUTPUT}/' +
                             timestamp)

    opt.save_path.mkdir(exist_ok=True, parents=True)
    # pytorch init
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = (torch.device(f'cuda:{opt.local_rank}')
              if torch.cuda.is_available() else torch.device('cpu'))

    # build dataloader and detectors
    training_dataset = build_dataloader(cfg.DATASET.TRAIN,
                                        cfg.DATASET.DATA_ROOT)
    model = build_detectors(cfg.OETR).to(device)

    if cfg.OETR.CHECKPOINT:
        model.load_state_dict(
            torch.load(cfg.OETR.CHECKPOINT, map_location='cpu'))

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[opt.local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[15, 30],
                                                     gamma=0.1)

    if opt.local_rank == 0:
        writer = SummaryWriter(opt.save_path / 'logs')
        logger = get_logger(
            os.path.join(opt.save_path, ('{}.log'.format(timestamp))))
        logger.info(opt)
        logger.info(cfg)
        logger.info(model)

    if opt.validation:
        validation_dataset = build_dataloader(cfg.DATASET.VAL,
                                              cfg.DATASET.DATA_ROOT)
        validation_dataset.build_dataset()
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=False,
        )

    # start training
    for epoch in range(opt.epoch):
        model.float().train()
        training_dataset.build_dataset()

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            training_dataset)
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )

        for i, batch in enumerate(training_dataloader):
            data = model(batch)
            loss = sum(_value for _key, _value in data.items()
                       if 'loss' in _key)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # save info and visualization
            if opt.local_rank == 0 and i % 50 == 0:
                info = loss_info(data, writer,
                                 i + epoch * len(training_dataloader))

                logger.info(
                    'Epoch [{}][{}/{}], lr: {:E}, loss: {:.5f}, {}'.format(
                        epoch,
                        i,
                        len(training_dataloader),
                        scheduler.get_last_lr()[0],
                        loss,
                        info,
                    ))

                writer.add_scalar('Loss/train', loss.item(),
                                  i + epoch * len(training_dataloader))
                if cfg.DATASET.TRAIN.VIZ:
                    bbox1 = data['pred_bbox1'][0].detach().cpu().numpy(
                    ).astype(int)
                    bbox2 = data['pred_bbox2'][0].detach().cpu().numpy(
                    ).astype(int)
                    gt_bbox1 = batch['overlap_box1'][0].cpu().numpy().astype(
                        int)
                    gt_bbox2 = batch['overlap_box2'][0].cpu().numpy().astype(
                        int)
                    viz_name = os.path.join(
                        str(opt.save_path),
                        'train_{}_{}_'.format(epoch, i) +
                        batch['file_name'][0],
                    )
                    if 'pred_center1' in data.keys():
                        visualize_centerness_overlap_gt(
                            batch['image1'][0].cpu().numpy() * 255,
                            bbox1,
                            gt_bbox1,
                            data['pred_center1'][0].detach().cpu().numpy(),
                            batch['image2'][0].cpu().numpy() * 255,
                            bbox2,
                            gt_bbox2,
                            data['pred_center2'][0].detach().cpu().numpy(),
                            viz_name,
                        )
                    else:
                        visualize_overlap_gt(
                            batch['image1'][0].cpu().numpy() * 255,
                            bbox1,
                            gt_bbox1,
                            batch['image2'][0].cpu().numpy() * 255,
                            bbox2,
                            gt_bbox2,
                            viz_name,
                        )
        # validation results
        if opt.local_rank == 0 and opt.validation:
            model.eval()
            evaluate(
                model,
                validation_dataloader,
                logger,
                opt.save_path,
                iou_thrs=np.arange(0.5, 0.96, 0.05),
                epoch=epoch,
                oiou=cfg.DATASET.VAL.OIOU,
                viz=cfg.DATASET.VAL.VIZ,
            )
        scheduler.step()

        if opt.local_rank == 0:
            print('---------save weights----------')
            model_out_path = opt.save_path / 'model_epoch_{}.pth'.format(epoch)
            torch.save(model.module.state_dict(), model_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--config_path',
        type=str,
        default='assets/megadepth/config.py',
        help='configs of trainning',
    )
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num_workers')

    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Use less datasets')
    parser.add_argument('--validation',
                        action='store_true',
                        help='Use validation recalls')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epoch',
                        type=int,
                        default=30,
                        help='Number of epoches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
