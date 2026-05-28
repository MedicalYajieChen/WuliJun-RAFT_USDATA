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


@torch.no_grad()
def validate_usdata_all_cubic(model):
    model.eval()
    val_dataset = datasets_US_accuracy.USdata(test=True)
    epe_mask_list = []
    epe_pointsmask_list = []
    epe_cubic_list = []

    for val_id in range(len(val_dataset)):
        # print(len(val_dataset))
        image1, image2, flow_gt, valid_gt, pointsmask, coordinate = val_dataset[val_id]
        # image1 = image1[None].to(device)
        # image2 = image2[None].to(device)
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)####改
        flows_pr = model(image1, image2)
        flow = padder.unpad(flows_pr[-1]).cpu()
        # 计算mask epe
        mask = valid_gt > 0
        epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
        # print(epe[:, mask].mean().item())
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

    #TO

    # k1 = 3.8154
    # k2 = 3.5429

    # # ESAOTE
    k1 = 4.0584
    k2 = 4.0584

    # #GE Vingmed Ultrasound
    # k1 = 3.4150
    # k2 = 3.4150

    # #'Siemens'
    # k1 = 4.2857
    # k2 = 4.2857

    # ##phlips
    # k1 = 4.0625
    # k2 = 3.7531

    # 'SAMSUNG MEDISON CO'
    # k1 = 2.58
    # k2 = 2.58
    # #'Hitachi Aloka Medical,Ltd'
    # k1 = 3.6
    # k2 = 3.6

    epe_mask_list = np.array(epe_mask_list)
    a = 245
    for i in range(0, a):
        epe_mask_list[i] = epe_mask_list[i]/k1 
    for i in range(a, 2*a):
        epe_mask_list[i] = epe_mask_list[i]/k2 
    for i in range(2*a, len(epe_mask_list)):
        epe_mask_list[i] = epe_mask_list[i]/k1 
    epe_mask = np.mean(epe_mask_list)
    epe_mask_std = np.std(epe_mask_list)

    epe_pointsmask_list = np.array(epe_pointsmask_list)
    for i in range(0, a):
        epe_pointsmask_list[i] = epe_pointsmask_list[i]/k1 
    for i in range(a, 2*a):
        epe_pointsmask_list[i] = epe_pointsmask_list[i]/k2
    for i in range(2*a, len(epe_pointsmask_list)):
        epe_pointsmask_list[i] = epe_pointsmask_list[i]/k1
    epe_point = np.mean(epe_pointsmask_list)
    epe_point_std = np.std(epe_pointsmask_list)

    epe_cubic_list = np.array(epe_cubic_list)
    for i in range(0, a):
        epe_cubic_list[i] = epe_cubic_list[i]/k1 
    for i in range(a, 2*a):
        epe_cubic_list[i] = epe_cubic_list[i]/k2
    for i in range(2*a, len(epe_cubic_list)):
        epe_cubic_list[i] = epe_cubic_list[i]/k1
    epe_cubic = np.mean(epe_cubic_list)
    epe_cubic_std = np.std(epe_cubic_list)

    print("Validation USDATA epe_mask: %f, epe_mask_std: %f, epe_pointmask: %f, epe_pointmask_std: %f, epe_cubic: %f, epe_cubic_std: %f" % (epe_mask, epe_mask_std, epe_point, epe_point_std, epe_cubic, epe_cubic_std))
    return {'epe-mask': epe_mask, 'epe-mask_std': epe_mask_std, 'epe-point': epe_point, 'epe_pointmask_std': epe_point_std, 'epe-cubic': epe_cubic, 'epe_cubic_std':epe_cubic_std}
