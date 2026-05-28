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
import core.frequency_domain.datasets_fre as datasets_fre
from core.utils.utils import InputPadder, forward_interpolate
from scipy import interpolate
import math
import cv2
import copy


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
def validate_flow(raft_model, iters=4):
    raft_model.eval()
    val_dataset = datasets_fre.USdata(test=True)
    images_list = val_dataset.test_image_list
    epe_mask_list = []
    epe_pointsmask_list = []
    epe_cubic_list = []
    for val_id in range(len(val_dataset)):
        img1, img2, flow_1, flow_2, mask_1, mask_2, seg_mask_1, seg_mask_2, pointsmask, coordinate = val_dataset[val_id]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()
        padder = InputPadder(img1.shape)
        img1 = padder.pad(img1)[0]
        img2 = padder.pad(img2)[0]

        # 光流
        flow_low, flow_pr = raft_model(img1, img2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr).cpu()

        # 计算mask epe
        mask = mask_1 > 0
        epe = torch.sum((flow - flow_1) ** 2, dim=1).sqrt()
        epe_mask_list.append(epe[:, mask].mean().item())

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

    epe_mask_list = np.array(epe_mask_list)
    epe_mask = np.mean(epe_mask_list)

    epe_pointsmask_list = np.array(epe_pointsmask_list)
    epe_point = np.mean(epe_pointsmask_list)

    epe_cubic_list = np.array(epe_cubic_list)
    epe_cubic = np.mean(epe_cubic_list)

    print("Validation USDATA epe_mask: %f, epe_pointmask: %f, epe_cubic: %f" % (epe_mask, epe_point, epe_cubic))
    return {'epe-mask': epe_mask, 'epe-point': epe_point, 'epe-cubic': epe_cubic}

