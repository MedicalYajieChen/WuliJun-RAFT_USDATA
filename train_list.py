from __future__ import print_function, division
import sys
# sys.path.append('core')
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from core.raft_v2.raft_v2 import RAFT
from core.raft import RAFT
from torch.autograd import Variable
from core.utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
# from core.raft import RAFT
# from core.pwc_net import Network
import evaluate_v1 as evaluate
import core.dataset_US_list as datasets
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
print(torch.cuda.device_count())

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
SUM_FREQ = 300
VAL_FREQ = 1500


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1).cuda()
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def seg_loss(seg_mask_pr, seg_mask_1):
    criterion = nn.CrossEntropyLoss()
    c_loss = criterion(seg_mask_pr, seg_mask_1)
    d_loss = dice_loss(F.softmax(seg_mask_pr, dim=1).float(), F.one_hot(seg_mask_1, 4).permute(0, 3, 1, 2).float())
    loss = c_loss + d_loss
    metrics = {
        'd_loss': d_loss.item(),
        'c_loss': c_loss.item(),
        'loss': loss.item()
    }
    return loss, metrics

# def seg_loss_union(forward_mask_list, backward_mask_list, mask_list):
#     mid_loss = 0.0
#     for i in range(len(forward_mask_list)-1):
#         f_index = i + 1
#         b_index = i
#         forward_mask = forward_mask_list[f_index]
#         backward_mask = backward_mask_list[b_index]
#         d_loss = dice_loss(F.softmax(forward_mask, dim=1).float(), F.softmax(backward_mask, dim=1).float())
#         mid_loss = mid_loss + d_loss
#     criterion = nn.CrossEntropyLoss()
#     start_loss_c = criterion(forward_mask_list[0], mask_list[0])
#     end_loss_c = criterion(backward_mask_list[-1], mask_list[-1])
#     start_loss_d = dice_loss(F.softmax(forward_mask_list[0], dim=1), F.one_hot(mask_list[0], 4).permute(0, 3, 1, 2).float())
#     end_loss_d = dice_loss(F.softmax(backward_mask_list[-1], dim=1), F.one_hot(mask_list[-1], 4).permute(0, 3, 1, 2).float())
#     start_loss = start_loss_c+start_loss_d
#     end_loss = end_loss_d + end_loss_c
#     warp_gt_loss = 0.0
#     for i in range(len(forward_mask_list)):
#         f_d_loss = dice_loss(F.softmax(forward_mask_list[i], dim=1).float(), F.one_hot(mask_list[i], 4).permute(0, 3, 1, 2).float())
#         # f_c_loss = criterion(forward_mask_list[i], mask_list[i])
#         b_d_loss = dice_loss(F.softmax(backward_mask_list[i], dim=1).float(), F.one_hot(mask_list[i+1], 4).permute(0, 3, 1, 2).float())
#         # b_c_loss = criterion(backward_mask_list[i], mask_list[i+1])
#         warp_gt_loss = warp_gt_loss + f_d_loss  + b_d_loss
#
#     loss = warp_gt_loss*0.1 + 0.1* mid_loss +start_loss_c+end_loss_c +start_loss_d +end_loss_d
#     metrics = {
#         'start_loss_c': start_loss_c.item(),
#         'start_loss_d': start_loss_d.item(),
#         'mid_loss': mid_loss.item(),
#         'warp_gt_loss': warp_gt_loss.item(),
#         'loss': loss.item()
#     }
#     return loss, metrics

def seg_loss_union(forward_mask_list, backward_mask_list, seg_mask_1, seg_mask_2):
    mid_loss = 0.0
    for i in range(len(forward_mask_list)-1):
        f_index = i + 1
        b_index = i
        forward_mask = forward_mask_list[f_index]
        backward_mask = backward_mask_list[b_index]
        d_loss = dice_loss(F.softmax(forward_mask, dim=1).float(), F.softmax(backward_mask, dim=1).float())
        mid_loss = mid_loss + d_loss

    criterion = nn.CrossEntropyLoss()
    start_loss_c = criterion(forward_mask_list[0], seg_mask_1)
    end_loss_c = criterion(backward_mask_list[-1], seg_mask_2)
    start_loss_d = dice_loss(F.softmax(forward_mask_list[0], dim=1), F.one_hot(seg_mask_1, 4).permute(0, 3, 1, 2).float())
    end_loss_d = dice_loss(F.softmax(backward_mask_list[-1], dim=1), F.one_hot(seg_mask_2, 4).permute(0, 3, 1, 2).float())

    loss = start_loss_c + start_loss_d + end_loss_c + end_loss_d + 0.1 * mid_loss
    metrics = {
        'start_loss_c': start_loss_c.item(),
        'start_loss_d': start_loss_d.item(),
        'mid_loss': mid_loss.item(),
        'loss': loss.item()
    }
    return loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
    #                                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000, 8000, 12000, 15000, 20000], gamma=0.5)

    return optimizer, scheduler



class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        print(training_str, end=' ')
        for key in self.running_loss:
            print(key + ": %10.4f" % (self.running_loss[key] / SUM_FREQ), end=' ')
        print("\n", end='')

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
        self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(RAFT(args)).cuda()
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
        print('预训练权重加载成功')

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image_sequence, mask_sequence = data_blob
            image_sequence = [x.cuda() for x in image_sequence]
            mask_sequence =  [x.cuda() for x in mask_sequence]

            # if args.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            # 前向光流
            seg_mask_2_onehot = F.one_hot(mask_sequence[-1], 4).permute(0, 3, 1, 2).float()
            sequence_length = len(image_sequence)
            seg_mask_warp_forward = None
            forward_mask_list = []
            for i in range(sequence_length - 1):
                img1_index = sequence_length - 2 - i
                img2_index = sequence_length - 1 - i
                img1 = image_sequence[img1_index]
                img2 = image_sequence[img2_index]
                flow_predictions = model(img1, img2, iters=args.iters)
                if seg_mask_warp_forward is None:
                    seg_mask_warp_forward = warp(seg_mask_2_onehot, flow_predictions[-1])
                else:
                    seg_mask_warp_forward = warp(seg_mask_warp_forward, flow_predictions[-1])
                forward_mask_list.append(seg_mask_warp_forward)
            forward_mask_list.reverse()

            # 反向光流
            seg_mask_1_onehot = F.one_hot(mask_sequence[0], 4).permute(0, 3, 1, 2).float()
            sequence_length = len(image_sequence)
            seg_mask_warp_backward = None
            backward_mask_list = []
            for i in range(sequence_length - 1):
                img1_index = i
                img2_index = i + 1
                img1 = image_sequence[img1_index]
                img2 = image_sequence[img2_index]
                flow_predictions = model(img2, img1, iters=args.iters)
                if seg_mask_warp_backward is None:
                    seg_mask_warp_backward = warp(seg_mask_1_onehot, flow_predictions[-1])
                else:
                    seg_mask_warp_backward = warp(seg_mask_warp_backward, flow_predictions[-1])
                backward_mask_list.append(seg_mask_warp_backward)

            # loss, metrics = seg_loss(seg_mask_warp_forward, seg_mask_1)
            # loss, metrics = seg_loss_union(forward_mask_list, backward_mask_list, mask_sequence)
            loss, metrics = seg_loss_union(forward_mask_list, backward_mask_list, mask_sequence[0], mask_sequence[-1] )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = '/data/csj/GLS/checkpoints/RAFT_USDATA/list_all/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                print('保存权重')
                results = {}
                results.update(evaluate.validate_usdata_all_cubic_ba(model))
                logger.write_dict(results)
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = '/data/csj/GLS/checkpoints/RAFT_USDATA/list_all/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-usdata', help="name your experiment")
    parser.add_argument('--stage', default='usdata', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', default='usdata', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 384])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('/data/csj/GLS/checkpoints/RAFT_USDATA/list_all/'):
        os.mkdir('/data/csj/GLS/checkpoints/RAFT_USDATA/list_all/')

    train(args)
