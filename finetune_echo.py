"""
训练前向光流，有交叉熵损失
"""
from __future__ import print_function, division
import sys
# sys.path.append('core')
import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.cuda.device_count())
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from RAFT.core.raft import RAFT
# from core.KPAFlow import KPAFlow
# from core.pwc_net import Network
from core.echopwc import Network
from torch.autograd import Variable
import utils.stage3.dataset_baseline as dataset_camus
import utils.stage3.evaluate_v3 as evaluate
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
from hausdorff import hausdorff_distance

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
VAL_FREQ = 500


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # print(B)
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
    # print(seg_mask_pr.shape,  seg_mask_1.shape)
    criterion = nn.CrossEntropyLoss()
    c_loss = criterion(seg_mask_pr, seg_mask_1)
    # pred_classes = torch.argmax(seg_mask_pr, dim=1)
    # pred_classes = pred_classes.squeeze(0)
    # print(pred_classes.shape)
    # pred_classes = pred_classes.cpu().numpy()
    # cv2.imwrite("/home/csj/GLS/RAFT_finetune/temp/mask.png", pred_classes*50)
    
    d_loss = dice_loss(F.softmax(seg_mask_pr, dim=1).float(), F.one_hot(seg_mask_1, 2).permute(0, 3, 1, 2).float())
    loss = c_loss + d_loss
    metrics = {
        'd_loss': d_loss.item(),
        'c_loss': c_loss.item(),
        'loss': loss.item()
    }
    return loss, metrics

def get_contour_list(mask_gray, gap):
    th = np.max(mask_gray) - 10
    ret, thresh = cv2.threshold(mask_gray, th, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_temp = np.array(contours)
    contours_temp = np.squeeze(contours_temp)
    contour_list = []
    for i in range(len(contours_temp)):
        if i % gap == 0:
            contour_list.append(contours_temp[i])
    return contour_list

def get_hd95(contour_list_start, contour_list_end):
    """
    计算hd95
    :param contour_list_start: 起始轮廓点列表，list
    :param contour_list_end: 末尾轮廓点列表，list
    :return:hd95值
    """
    contour_list_start = np.array(contour_list_start)
    contour_list_end = np.array(contour_list_end)
    manhattan = hausdorff_distance(contour_list_start, contour_list_end, distance="manhattan")  # 曼哈顿距离
    euclidean = hausdorff_distance(contour_list_start, contour_list_end, distance="euclidean")  # 欧氏距离
    chebyshev = hausdorff_distance(contour_list_start, contour_list_end, distance="chebyshev")  # 切比雪夫距离
    cosine = hausdorff_distance(contour_list_start, contour_list_end, distance="cosine")  # 余弦距离
    return manhattan, euclidean, chebyshev, cosine
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # chairs
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000], gamma=0.5)
    # Flyingthings
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6000, 12000, 18000, 30000, 40000, 50000, 60000, 70000, 80000, 100000], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000, 10000], gamma=0.5)
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
        self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    # model = nn.DataParallel(KPAFlow(args)).cuda()
    model = nn.DataParallel(Network())
    print("Parameter Count: %d" % count_parameters(model))
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
        print('预训练权重加载成功')
    model.cuda()
    model.train()
    train_loader = dataset_camus.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)
    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image_sequence, seg_mask_1, seg_mask_2= data_blob
            # print(a)
            seg_mask_1 = seg_mask_1.cuda()
            # print(seg_mask_1.shape)
            seg_mask_2 = seg_mask_2.cuda()
            image_sequence = [x.cuda() for x in image_sequence]
            # print('image_sequence[0].shape:' + str(image_sequence[0].shape))
            # if args.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            seg_mask_1_onehot = F.one_hot(seg_mask_1, 2).permute(0, 3, 1, 2).float()
            #print(seg_mask_1_onehot.shape)
            sequence_length = len(image_sequence)
            seg_mask_warp = None
            mask_warp = None
            # seg_mask_1_dim4 = seg_mask_1.unsqueeze(0).float()
            # print('seg_mask_1_dim4:' + str(seg_mask_1_dim4.shape))
            for i in range(sequence_length - 1):
                img1_index = i
                img2_index = i + 1
                img1 = image_sequence[img1_index]
                img2 = image_sequence[img2_index]
                flow_predictions = model(img2, img1)
                if seg_mask_warp is None:
                    # mask_warp = warp(seg_mask_1_dim4, flow_predictions[-1])
                    seg_mask_warp = warp(seg_mask_1_onehot, flow_predictions[-1])
                else:
                    # mask_warp = warp(mask_warp, flow_predictions[-1])
                    seg_mask_warp = warp(seg_mask_warp, flow_predictions[-1])
            # print(seg_mask_warp.shape, seg_mask_2.shape)
            loss_backward, metrics = seg_loss(seg_mask_warp, seg_mask_2)
            scaler.scale(loss_backward).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = '/data/csj/GLS/checkpoints/RAFT_finetune/checkpoints_A_echopwc/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                print('保存权重')
                results = {}
                results.update(evaluate.validate_flow(model.module))
                logger.write_dict(results)
                model.train()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    logger.close()
    PATH = '/data/csj/GLS/checkpoints/RAFT_finetune/checkpoints_A_echopwc/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default='stage3', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
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
    if not os.path.isdir('/data/csj/GLS/checkpoints/RAFT_finetune/checkpoints_A_echopwc'):
        os.mkdir('/data/csj/GLS/checkpoints/RAFT_finetune/checkpoints_A_echopwc')
    train(args)
