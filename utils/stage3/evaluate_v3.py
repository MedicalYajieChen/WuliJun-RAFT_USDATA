# sys.path.append('core')
import numpy as np
import torch
import torch.nn.functional as F
# import utils.CAMUS_finetune.dataset_camus as dataset_camus
import utils.stage3.dataset_baseline as dataset_camus
from torch.utils.data import DataLoader
import utils.dice_score as dice_score
from core.utils.utils import InputPadder
from scipy import interpolate
import torch.nn as nn
from torch.autograd import Variable
import math
import cv2
import copy
# import utils.CAMUS_finetune.datasets_HMC as dataset_camus
from PIL import Image
# import matplotlib.pyplot as plt
# from core.forward_warping import ForwardWarp


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

def warp_forward(x, flo):
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
    vgrid = Variable(grid) - flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1).cuda()
    # print(x.shape, vgrid.shape)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


@torch.no_grad()
def validate_flow(raft_model, iters=4):
    # warp_forward = ForwardWarp().cuda()
    raft_model.eval()
    val_dataset = dataset_camus.STAGE3(test=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    # 测试前向一致性
    Dice_forward_list = []
    for val_id, data_blob in enumerate(val_dataloader):
        image_sequence, seg_mask_1, seg_mask_2 = data_blob
        seg_mask_1 = seg_mask_1.cuda()
        seg_mask_2 = seg_mask_2.cuda()
        image_sequence = [x.cuda() for x in image_sequence]
        seg_mask_2_onehot = F.one_hot(seg_mask_2, 2).permute(0, 3, 1, 2).float()
        sequence_length = len(image_sequence)
        seg_mask_warp = None
        flow_pre = None
        motion_pre = None
        for i in range(sequence_length - 1):
            img1_index = sequence_length - 2 - i
            img2_index = sequence_length - 1 - i
            img1 = image_sequence[img1_index]
            img2 = image_sequence[img2_index]
            #moa_v3
            # flow_predictions, flow, motion_features = raft_model(img1, img2, iters=iters, flow_init=flow_pre, mo_pre=motion_pre)
            # flow_pre = warp(flow, flow)
            # motion_pre = motion_features
            # flow_predictions, flow = raft_model(img1, img2, iters=iters, flow_init=flow_pre)
            # flow_pre = warp(flow, flow)

            flow_predictions = raft_model(img1, img2)

            if seg_mask_warp is None:
                seg_mask_warp = warp(seg_mask_2_onehot, flow_predictions[-1])
            else:
                seg_mask_warp = warp(seg_mask_warp, flow_predictions[-1])
        mask_true = F.one_hot(seg_mask_1, 2).permute(0, 3, 1, 2).float()
        mask_pr = seg_mask_warp.argmax(dim=1)
        mask_pr = F.one_hot(mask_pr, 2).permute(0, 3, 1, 2).float()
        # print(mask_true.shape, mask_pr.shape)
        d_score = dice_score.multiclass_dice_coeff(mask_pr[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
        # print(val_id, d_score)
        Dice_forward_list.append(d_score.cpu())
    Dice_forward_list = np.array(Dice_forward_list)
    Dice_forward_mean = np.mean(Dice_forward_list)
    print(Dice_forward_mean)

    # 测试反向一致性
    Dice_backward_list = []
    for val_id, data_blob in enumerate(val_dataloader):
        # image_sequence, seg_mask_1, seg_mask_2 = data_blob
        image_sequence, seg_mask_1, seg_mask_2 = data_blob
        seg_mask_1 = seg_mask_1.cuda()
        seg_mask_2 = seg_mask_2.cuda()
        image_sequence = [x.cuda() for x in image_sequence]
        seg_mask_1_onehot = F.one_hot(seg_mask_1, 2).permute(0, 3, 1, 2).float()
        seg_mask_warp = None
        sequence_length = len(image_sequence)
        flow_pre = None
        motion_pre = None
        for i in range(sequence_length - 1):
            img1_index = i
            img2_index = i + 1
            img1 = image_sequence[img1_index]
            img2 = image_sequence[img2_index]
            # flow_predictions, flow, motion_features = raft_model(img2, img1, iters=iters, flow_init=flow_pre, mo_pre=motion_pre)
            # flow_pre = warp(flow, flow)
            # motion_pre = motion_features
            # flow_predictions, flow = raft_model(img2, img1, iters=iters, flow_init=flow_pre)
            # flow_pre = warp(flow, flow)
            flow_predictions = raft_model(img2, img1)
            if seg_mask_warp is None:
                seg_mask_warp = warp(seg_mask_1_onehot, flow_predictions[-1])
            else:
                seg_mask_warp = warp(seg_mask_warp, flow_predictions[-1])

        mask_true = F.one_hot(seg_mask_2, 2).permute(0, 3, 1, 2).float()
        mask_pr = seg_mask_warp.argmax(dim=1)
        mask_pr = F.one_hot(mask_pr, 2).permute(0, 3, 1, 2).float()

        # print(mask_true.shape, mask_pr.shape)
        d_score = dice_score.multiclass_dice_coeff(mask_pr[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
        # print(val_id, d_score)
        Dice_backward_list.append(d_score.cpu())
    Dice_backward_list = np.array(Dice_backward_list)
    Dice_backward_mean = np.mean(Dice_backward_list)
    print(Dice_backward_mean)

    print('Dice_forward_mean:', Dice_forward_mean, 'Dice_backward_mean:', Dice_backward_mean)
    return {'Dice_forward_mean': Dice_forward_mean, 'Dice_backward_mean': Dice_backward_mean}


