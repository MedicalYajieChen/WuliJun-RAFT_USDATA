import sys
# sys.path.append('core')
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets as datasets
import core.datasets_US_accuracy as datasets_US_accuracy
from core.utils import flow_viz

from core.utils import frame_utils

# from raft import RAFT
import models
from core.utils.utils import InputPadder, forward_interpolate
from scipy import interpolate
import scipy.io as io
import math

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


def bilinear(x, y, z, x_a, y_a):
    """
    :param x: [1, 2]
    :param y: [1, 2]
    :param z: [[左上角, 右上角], [左下角, 右下角]] is [[Q11, Q21], [Q12, Q22]]
    :return:
    """
    f = interpolate.interp2d(x, y, z, kind='linear')
    out = f(x_a, y_a)
    return out


def cubic(x, y, z, x_a, y_a):
    """
    :param x: x坐标序列
    :param y: y坐标序列
    :param z: 值序列
    :param x_a: 插值x坐标
    :param y_a: 插值y坐标
    :return: 插值效果
    """
    f = interpolate.interp2d(x, y, z, kind='cubic', copy=True, bounds_error=False, fill_value=None)
    out = f(x_a, y_a)
    return out


@torch.no_grad()
def validate_usdata_all_bilinear(model, iters=12):
    model.eval()
    val_dataset = datasets_US_accuracy.USdata(test=True)
    epe_mask_list = []
    epe_pointsmask_list = []
    epe_bilinear_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, pointsmask, coordinate = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr).cpu()

        # 计算mask epe
        mask = valid_gt > 0
        epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
        epe_mask_list.append(epe[:, mask].mean().item())

        # 计算pointsmask epe
        pointsmask = pointsmask > 0
        epe_pointsmask_list.append(epe[:, pointsmask].mean().item())

        # 计算双线性插值epe
        coordinate_file = coordinate
        pr_flow = flow.squeeze(dim=0)
        pr_flow_ux = pr_flow[0, :, :]
        pr_flow_uy = pr_flow[1, :, :]

        points_epe = 0
        for i in range(len(coordinate_file)):
            # 真实数据
            P_x = coordinate_file[i][0]
            P_y = coordinate_file[i][1]
            P_data_ux = coordinate_file[i][2]
            P_data_uy = coordinate_file[i][3]

            # P点周围四个点
            Q11_x = int(coordinate_file[i][0])
            Q11_y = int(coordinate_file[i][1])
            Q11_data_ux = pr_flow_ux[Q11_y, Q11_x]
            Q11_data_uy = pr_flow_uy[Q11_y, Q11_x]

            Q12_x = Q11_x
            Q12_y = Q11_y + 1
            Q12_data_ux = pr_flow_ux[Q12_y, Q12_x]
            Q12_data_uy = pr_flow_uy[Q12_y, Q12_x]

            Q21_x = Q11_x + 1
            Q21_y = Q11_y
            Q21_data_ux = pr_flow_ux[Q21_y, Q21_x]
            Q21_data_uy = pr_flow_uy[Q21_y, Q21_x]

            Q22_x = Q11_x + 1
            Q22_y = Q11_y + 1
            Q22_data_ux = pr_flow_ux[Q22_y, Q22_x]
            Q22_data_uy = pr_flow_uy[Q22_y, Q22_x]

            # 获取插值得到的光流
            ux_x = [Q11_x, Q21_x]
            ux_y = [Q11_y, Q12_y]
            ux_z = [[Q11_data_ux, Q21_data_ux], [Q12_data_ux, Q22_data_ux]]
            pr_P_data_ux = bilinear(ux_x, ux_y, ux_z, P_x, P_y)

            uy_x = [Q11_x, Q21_x]
            uy_y = [Q11_y, Q12_y]
            uy_z = [[Q11_data_uy, Q21_data_uy], [Q12_data_uy, Q22_data_uy]]
            pr_P_data_uy = bilinear(uy_x, uy_y, uy_z, P_x, P_y)

            point_epe = math.sqrt((P_data_ux - pr_P_data_ux) ** 2 + (P_data_uy - pr_P_data_uy) ** 2)
            points_epe = points_epe + point_epe

        points_epe = points_epe / 180
        epe_bilinear_list.append(points_epe)

    epe_mask_list = np.array(epe_mask_list)
    epe_mask = np.mean(epe_mask_list)
    epe_std = np.std(epe_mask_list)

    epe_pointsmask_list = np.array(epe_pointsmask_list)
    epe_point = np.mean(epe_pointsmask_list)
    epe_point_std = np.std(epe_pointsmask_list)


    epe_bilinear_list = np.array(epe_bilinear_list)
    epe_bilinear = np.mean(epe_bilinear_list)
    epe_point_std = np.std(epe_pointsmask_list)

    print("Validation USDATA epe_mask: %f, epe_pointmask: %f, epe_bilinear: %f" % (epe_mask, epe_point, epe_bilinear))
    return {'epe-mask': epe_mask, 'epe-point': epe_point, 'epe-bilinear': epe_bilinear}


@torch.no_grad()
def validate_usdata_all_cubic(model, iters=4):
    model.eval()
    val_dataset = datasets_US_accuracy.USdata(test=True)
    epe_mask_list = []
    epe_pointsmask_list = []
    epe_cubic_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, pointsmask, coordinate = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr).cpu()
        # 计算mask epe
        mask = valid_gt > 0
        epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
        epe_mask_list.append(epe[:, mask].mean().item())
        # print(flow.shape, flow_gt.shape, epe.shape)

        # 计算pointsmask epe
        pointsmask = pointsmask > 0
        epe_pointsmask_list.append(epe[:, pointsmask].mean().item())

        # 计算cubic插值epe
        coordinate_file = coordinate
        pr_flow = flow.squeeze(dim=0)
        pr_flow_ux = pr_flow[0, :, :]
        pr_flow_uy = pr_flow[1, :, :]

        points_epe = 0
        for i in range(len(coordinate_file)):
            # 真实数据
            P_x = coordinate_file[i][0]
            P_y = coordinate_file[i][1]
            P_data_ux = coordinate_file[i][2]
            P_data_uy = coordinate_file[i][3]

            # P点周围16个点
            Q_x = int(P_x) - 1
            Q_y = int(P_y) - 1
            x = []
            y = []
            for m in range(4):
                for n in range(4):
                    x.append(m + Q_x)
                    y.append(n + Q_y)

            uz_x = []
            uz_y = []
            for k in range(len(x)):
                uz_x.append(pr_flow_ux[y[k], x[k]])
                uz_y.append(pr_flow_uy[y[k], x[k]])

            # 获取插值得到的光流
            pr_P_data_ux = cubic(x, y, uz_x, P_x, P_y)
            pr_P_data_uy = cubic(x, y, uz_y, P_x, P_y)

            point_epe = math.sqrt((P_data_ux - pr_P_data_ux) ** 2 + (P_data_uy - pr_P_data_uy) ** 2)
            points_epe = points_epe + point_epe

        points_epe = points_epe / 180
        epe_cubic_list.append(points_epe)

    k1 = 4.0625
    k2 = 3.7531

    epe_mask_list = np.array(epe_mask_list)
    print(len(epe_mask_list))
    for i in range(0, (2*len(epe_mask_list)/3)):
        epe_mask_list[i] = epe_mask_list[i]/k1 
    for i in range((2*len(epe_mask_list)/3), len(epe_mask_list)):
        epe_mask_list[i] = epe_mask_list[i]/k2 
   
    epe_mask = np.mean(epe_mask_list)
    epe_mask_std = np.std(epe_mask_list)

    epe_pointsmask_list = np.array(epe_pointsmask_list)
    epe_pointsmask_list = epe_pointsmask_list/k
    epe_point = np.mean(epe_pointsmask_list)
    epe_point_std = np.std(epe_pointsmask_list)

    epe_cubic_list = np.array(epe_cubic_list)
    epe_cubic_list = epe_cubic_list/k
    epe_cubic = np.mean(epe_cubic_list)
    epe_cubic_std = np.std(epe_cubic_list)

    print("Validation USDATA epe_mask: %f, epe_mask_std: %f, epe_pointmask: %f, epe_pointmask_std: %f, epe_cubic: %f, epe_cubic_std: %f" % (epe_mask, epe_mask_std, epe_point, epe_point_std, epe_cubic, epe_cubic_std))
    return {'epe-mask': epe_mask, 'epe-mask_std': epe_mask_std, 'epe-point': epe_point, 'epe_pointmask_std': epe_point_std, 'epe-cubic': epe_cubic, 'epe_cubic_std':epe_cubic_std}
