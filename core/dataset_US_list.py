import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import math
import random
from glob import glob
from core.utils import frame_utils
from core.utils.augmentor_list import FlowAugmentor, SparseFlowAugmentor
import cv2
import joblib


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
            imgs_list = []
            flow_list = []
            valid_list = []
            pointsmask_list = []
            coordinate_list = []
            # size = [464,384]
            img1 = frame_utils.read_gray_img(self.test_image_list[index][0])
            img2 = frame_utils.read_gray_img(self.test_image_list[index][1])
            Height = int(math.floor(math.ceil(img1.shape[0] / 64.0) * 64.0))
            Width = int(math.floor(math.ceil(img1.shape[1] / 64.0) * 64.0))
            size = [Height, Width]
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # 堆叠成三通道
            img1 = np.resize(img1, (img1.shape[0], img1.shape[1], 1))
            img1 = np.concatenate((img1, img1, img1), axis=2)
            img2 = np.resize(img2, (img2.shape[0], img2.shape[1], 1))
            img2 = np.concatenate((img2, img2, img2), axis=2)

            # 读取mask
            valid = frame_utils.read_gray_img(self.test_mask_list[index])
            valid = cv2.resize(valid, size)
            valid = torch.from_numpy(valid)

            # 读取pointsmask
            pointsmask = frame_utils.read_gray_img(self.test_pointsmask_list[index])
            pointsmask = cv2.resize(pointsmask, size)
            pointsmask = torch.from_numpy(pointsmask)

            # 读取光流
            flow = frame_utils.read_gen(self.test_flow_list[index])
            flow = cv2.resize(flow, size)
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            # 读取坐标
            coordinate = joblib.load(self.test_coordinate_list[index])

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, flow, valid, pointsmask, coordinate

        index = index % len(self.image_list)
        image_sequence = []
        mask_sequence = []
        # print(len(self.train_mask2_list[index]))
        img1 = frame_utils.read_gray_img(self.image_list[index][0])
        Height = int(math.floor(math.ceil(img1.shape[0] / 64.0) * 64.0))
        Width = int(math.floor(math.ceil(img1.shape[1] / 64.0) * 64.0))
        for i in range(len(self.image_list[index])):
            img = frame_utils.read_gray_img(self.image_list[index][i])
            img = cv2.resize(img, [Height, Width])
            img = np.array(img).astype(np.uint8)
            # 堆叠成三通道
            img = np.resize(img, (img.shape[0], img.shape[1], 1))
            img = np.concatenate((img, img, img), axis=2)
            image_sequence.append(img)
            # print(self.test_mask2_list[index])
            mask = frame_utils.read_gray_img(self.mask_list[index][i])
            unique_values = np.unique(mask)
            mask[mask==unique_values[1]]=1
            mask[mask == unique_values[2]] = 2

            mask = cv2.resize(mask, [Height, Width])
            # xinshi_flag = np.max(mask)

            # mask2 = torch.from_numpy(mask2)
            mask_sequence.append(mask)
        if self.augmentor is not None:
            image_sequence, mask_sequence = self.augmentor(image_sequence, mask_sequence)
        for i in range(len(mask_sequence)):
            mask = mask_sequence[i].copy()
            mask = torch.from_numpy(mask).long()
            mask_sequence[i] = mask
        for i in range(len(image_sequence)):
            img = image_sequence[i].copy()
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            image_sequence[i] = img
        return image_sequence, mask_sequence

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
        validation = 'ESAOTE'
        train_images_list, train_mask_list, test_images_list, test_flow_list, test_mask_list, test_pointsmask_list, test_coordinate_list = get_USData_list(root, validation)
        self.image_list = train_images_list
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
        else:
            images_list, mask_list = get_subset_list(root_path, sp)
            train_images_list = train_images_list + images_list
            train_mask_list = train_mask_list + mask_list
    return train_images_list, train_mask_list, test_images_list, test_flow_list, test_mask_list, test_pointsmask_list, test_coordinate_list


def get_subset_list(root_path, supplier, test=False):
    view_name = ['A4C', 'A2C', 'A3C']
    mode_name = ['laddist', 'ladprox', 'lcx', 'normal', 'rca']
    test_images_list = []
    images_list = []
    flow_list = []
    mask_list = []
    seg_mask_list = []
    pointsmask_list = []
    coordinate_list = []
    temp_img_list = []
    temp_mask_list = []
    for vn in view_name:
        for mn in mode_name:
            image_dirs = os.path.join(root_path, supplier, vn, mn, 'frames')
            seg_mask_dirs = os.path.join(root_path, supplier, vn, mn, 'seg_mask')
            images = sorted(glob(os.path.join(image_dirs, '*.png')))
            seg_masks = sorted(glob(os.path.join(seg_mask_dirs, '*.png')))

            SEQUENCE_LEN = 20
            step = (len(images) - 1) / (SEQUENCE_LEN - 1)
            temp_img_list = []
            temp_mask_list = []
            for k in range(SEQUENCE_LEN//2):
                temp_img_list.append(images[round(k * step)])
                temp_mask_list.append(seg_masks[round(k * step)])
            images_list.append(temp_img_list)
            seg_mask_list.append(temp_mask_list)
            # if supplier == 'GE Vingmed Ultrasound':
            #     print(vn, mn, supplier)
            temp_img_list = []
            temp_mask_list = []
            for k in range(SEQUENCE_LEN//2, SEQUENCE_LEN):
                temp_img_list.append(images[round(k * step)])
                temp_mask_list.append(seg_masks[round(k * step)])
            images_list.append(temp_img_list)
            seg_mask_list.append(temp_mask_list)

            if test:
                for i in range(len(images) - 1):
                    test_images_list.append([images[i], images[i + 1]])

                flow_dirs = os.path.join(root_path, supplier, vn, mn, 'flow')
                mask_dirs = os.path.join(root_path, supplier, vn, mn, 'mask')
                flows = sorted(glob(os.path.join(flow_dirs, '*.flo')))
                masks = sorted(glob(os.path.join(mask_dirs, '*.png')))
                flow_list = flow_list + flows
                mask_list = mask_list + masks
                pointsmask_dirs = os.path.join(root_path, supplier, vn, mn, 'pointsmask')
                pointsmasks = sorted(glob(os.path.join(pointsmask_dirs, '*.png')))
                pointsmask_list = pointsmask_list + pointsmasks

                coordinate_dirs = os.path.join(root_path, supplier, vn, mn, 'coordinate')
                coordinates = sorted(glob(os.path.join(coordinate_dirs, '*.pkl')))
                coordinate_list = coordinate_list + coordinates

    if test:
        return test_images_list, flow_list, mask_list, pointsmask_list, coordinate_list
    else:
        return images_list, seg_mask_list

