from __future__ import print_function, division
import sys
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
print(torch.cuda.device_count())

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core.echopwc import Network
import evaluate_pwc as evaluate
import core.datasets_US_accuracy as datasets
from torch.utils.tensorboard import SummaryWriter


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
# VAL_FREQ = 1000
VAL_FREQ = 1000


# def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
#     """ Loss function defined over sequence of flow predictions """
#     n_predictions = len(flow_preds)
#     flow_loss = 0.0
#     # exlude invalid pixels and extremely large diplacements
#     # mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
#     # valid = (valid >= 0.5) & (mag < max_flow)
#     valid = valid > 0
#     for i in range(n_predictions):
#         i_weight = gamma ** (n_predictions - i - 1)
#         i_loss = (flow_preds[i] - flow_gt).abs()
#         flow_loss += i_weight * (valid[:, None] * i_loss).mean()
#     epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]
#     metrics = {
#         'epe': epe.mean().item(),
#         '0.1px': (epe < 0.1).float().mean().item(),
#         '0.5px': (epe < 0.5).float().mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#     }
#     return flow_loss, metrics
def sequence_loss(flow_pred, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    size = flow_pred[-1].shape[2:]
    valid = valid > 0
    n_predictions = len(flow_pred)
    flow_loss = 0.0
    # Bate = [1.0, 0.50, 0.25, 0.12, 0.06, 0.03, 0.015]
    Bate = [0.015, 0.03, 0.06, 0.12, 0.25, 0.05, 1.0]
    for i in range(n_predictions):
        flow = torch.nn.functional.interpolate(flow_pred[i], size, mode='bilinear', align_corners=False)
        # i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow - flow_gt).abs()
        flow_loss += Bate[i] * (valid[:, None] * i_loss).mean()
    epe = torch.sum((flow_pred[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '0.1px': (epe < 0.1).float().mean().item(),
        '0.5px': (epe < 0.5).float().mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
    }
    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
    #                                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(Network())
    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.train()
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('预训练权重加载成功')
    # print(next(model.parameters()).device)
    # if args.stage != 'chairs':
    #     model.module.freeze_bn()
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0 #改
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)
    add_noise = True
    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            flow_predictions = model(image1, image2)
            l1_lambda = 0.0001 # L1正则化系数
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            loss = loss + l1_lambda * l1_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = '/home/csj/GLS/RAFT_USDATA/checkpoints_echopwc/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model))
                    elif val_dataset == 'usdata':
                        results.update(evaluate.validate_usdata_all_cubic(model))
                logger.write_dict(results)
                model.train()
                # if args.stage != 'chairs':
                #     model.module.freeze_bn()
            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    logger.close()
    PATH = '/home/csj/GLS/RAFT_USDATA/checkpoints_echopwc/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pwc_usdata', help="name your experiment")
    parser.add_argument('--stage', default='usdata', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', default='usdata', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 448])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--wdecay', type=float, default=.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    # args.restore_ckpt = '/home/csj/GLS/RAFT_USDATA/checkpoints_echopwc/40000_pwc_usdata.pth'

    '''
    --name pwc_usdata \
                                                 --restore_ckpt /home/csj/GLS/RAFT_USDATA/checkpoints_echopwc/40000_pwc_usdata.pth \
                                                 --stage usdata \
                                                 --validation usdata \
                                                 --num_steps 40000 \
                                                 --batch_size 4 \
                                                 --lr 0.000001 \
                                                 --image_size 320 448 \
                                                 --wdecay 0.00001
    '''
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('/home/csj/GLS/RAFT_USDATA/checkpoints_echopwc'):
        os.mkdir('/home/csj/GLS/RAFT_USDATA/checkpoints_echopwc')

    train(args)


