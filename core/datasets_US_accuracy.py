import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import math
import random
from glob import glob
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import pickle


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []

        self.test_image_list = []
        self.test_flow_list = []
        self.test_mask_list = []

        self.test_pointsmask_list = []
        self.test_coordinate_list = []

    def __getitem__(self, index):

        if self.is_test:
            # 读取灰度图
            img1 = frame_utils.read_gray_img(self.test_image_list[index][0])
            img2 = frame_utils.read_gray_img(self.test_image_list[index][1])
            Height = int(math.floor(math.ceil(img1.shape[0] / 8.0) * 8.0))##
            Width = int(math.floor(math.ceil(img1.shape[1] / 8.0) * 8.0))
            img1 = cv2.resize(img1, [Width, Height])
            img2 = cv2.resize(img2, [Width, Height])
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # 堆叠成三通道
            img1 = np.resize(img1, (img1.shape[0], img1.shape[1], 1))
            img1 = np.concatenate((img1, img1, img1), axis=2)
            img2 = np.resize(img2, (img2.shape[0], img2.shape[1], 1))
            img2 = np.concatenate((img2, img2, img2), axis=2)

            # 读取mask
            valid = frame_utils.read_gray_img(self.test_mask_list[index])
            valid = cv2.resize(valid, [Width, Height])
            valid = torch.from_numpy(valid)

            # 读取pointsmask
            pointsmask = frame_utils.read_gray_img(self.test_pointsmask_list[index])
            pointsmask = cv2.resize(pointsmask, [Width, Height])
            pointsmask = torch.from_numpy(pointsmask)

            # 读取光流
            flow = frame_utils.read_gen(self.test_flow_list[index])
            flow = cv2.resize(flow, [Width, Height])
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            
            # 读取坐标
            coordinate = joblib.load(self.test_coordinate_list[index])

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, flow, valid, pointsmask, coordinate

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            # 读取光流和mask
            flow = frame_utils.read_gen(self.flow_list[index])
            valid = frame_utils.read_gray_img(self.mask_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        # 读取灰度图
        img1 = frame_utils.read_gray_img(self.image_list[index][0])
        img2 = frame_utils.read_gray_img(self.image_list[index][1])

        # Height = int(math.floor(math.ceil(img1.shape[0] / 64.0) * 64.0))
        # Width = int(math.floor(math.ceil(img1.shape[1] / 64.0) * 64.0))
        # img1 = cv2.resize(img1, [Height, Width])
        # img2 = cv2.resize(img2, [Height, Width])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # 堆叠成三通道
        img1 = np.resize(img1, (img1.shape[0], img1.shape[1], 1))
        img1 = np.concatenate((img1, img1, img1), axis=2)
        img2 = np.resize(img2, (img2.shape[0], img2.shape[1], 1))
        img2 = np.concatenate((img2, img2, img2), axis=2)

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # if img1.shape != [3,384, 448]
        # print(img1.shape, img2.shape, flow.shape, valid.shape)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        if self.is_test:
            return len(self.test_image_list)
        else:
            return len(self.image_list)


class USdata(FlowDataset):
    def __init__(self, aug_params=None, test=False, root='/data/csj/data_accuracy/'):
        super(USdata, self).__init__(aug_params, sparse=True)
        validation='ESAOTE'
        train_images_list, train_flow_list, train_mask_list, test_images_list, test_flow_list, test_mask_list, test_pointsmask_list, test_coordinate_list = get_USData_list(root, validation)
        self.image_list = train_images_list
        self.flow_list = train_flow_list
        self.mask_list = train_mask_list

        self.is_test = test

        self.test_image_list = test_images_list
        self.test_flow_list = test_flow_list
        self.test_mask_list = test_mask_list
        self.test_pointsmask_list = test_pointsmask_list
        self.test_coordinate_list = test_coordinate_list


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'usdata':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.5, 'do_flip': True}
        train_dataset = USdata(aug_params)
    else:
        train_dataset = None

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def get_USData_list(root_path, validation):
    print('Validation:', validation)
    suppliers = ['ESAOTE', 'GE Vingmed Ultrasound', 'Hitachi Aloka Medical,Ltd', 'Philips Medical Systems', 'SAMSUNG MEDISON CO', 'Siemens', 'TOSHIBA_MEC_US']
    train_images_list = []
    train_flow_list = []
    train_mask_list = []
    test_images_list = []
    test_flow_list = []
    test_mask_list = []
    test_pointsmask_list = []
    test_coordinate_list = []

    for sp in suppliers:
        if sp == validation:
            images_list, flow_list, mask_list, pointsmask_list, coordinate_list = get_subset_list(root_path, validation, test=True)
            test_images_list = test_images_list + images_list
            test_flow_list = test_flow_list + flow_list
            test_mask_list = test_mask_list + mask_list
            test_pointsmask_list = test_pointsmask_list + pointsmask_list
            test_coordinate_list = test_coordinate_list + coordinate_list
            
            train_images_list = train_images_list + images_list
            train_flow_list = train_flow_list + flow_list
            train_mask_list = train_mask_list + mask_list

        else:
            images_list, flow_list, mask_list = get_subset_list(root_path, sp)
            train_images_list = train_images_list + images_list
            train_flow_list = train_flow_list + flow_list
            train_mask_list = train_mask_list + mask_list
    return train_images_list, train_flow_list, train_mask_list, test_images_list, test_flow_list, test_mask_list, test_pointsmask_list, test_coordinate_list


def get_subset_list(root_path, supplier, test=False):
    view_name = ['A4C', 'A2C', 'A3C']
    mode_name = ['laddist', 'ladprox', 'lcx', 'normal', 'rca']

    images_list = []
    flow_list = []
    mask_list = []
    pointsmask_list = []
    coordinate_list = []
    for vn in view_name:
        for mn in mode_name:
            image_dirs = os.path.join(root_path, supplier, vn, mn, 'frames')
            flow_dirs = os.path.join(root_path, supplier, vn, mn, 'flow')
            mask_dirs = os.path.join(root_path, supplier, vn, mn, 'mask')

            images = sorted(glob(os.path.join(image_dirs, '*.png')))
            flows = sorted(glob(os.path.join(flow_dirs, '*.flo')))
            masks = sorted(glob(os.path.join(mask_dirs, '*.png')))

            if test:
                pointsmask_dirs = os.path.join(root_path, supplier, vn, mn, 'pointsmask')
                pointsmasks = sorted(glob(os.path.join(pointsmask_dirs, '*.png')))
                pointsmask_list = pointsmask_list + pointsmasks

                coordinate_dirs = os.path.join(root_path, supplier, vn, mn, 'coordinate')
                coordinates = sorted(glob(os.path.join(coordinate_dirs, '*.pkl')))
                coordinate_list = coordinate_list + coordinates

            for i in range(len(images) - 1):
                images_list.append([images[i], images[i + 1]])
            flow_list = flow_list + flows
            mask_list = mask_list + masks

    if test:
        return images_list, flow_list, mask_list, pointsmask_list, coordinate_list
    else:
        return images_list, flow_list, mask_list


if __name__ == '__main__':
    root_path = 'F:/Dataset/USDATA/data_accuracy'
    validation = 'ESAOTE'
    train_images_list, train_flow_list, train_mask_list, test_images_list, test_flow_list, test_mask_list, test_pointsmask_list, test_coordinate_list = get_USData_list(root_path, validation)
    print(len(train_images_list))
    print(len(train_flow_list))
    print(len(train_mask_list))
    print(len(test_images_list))
    print(len(test_flow_list))
    print(len(test_mask_list))
    print(len(test_pointsmask_list))
    print(len(test_coordinate_list))
    print(test_images_list[0:10])
    print(test_flow_list[0:10])
    print(test_mask_list[0:10])
    print(test_pointsmask_list[0:10])
    print(test_coordinate_list[0:10])


