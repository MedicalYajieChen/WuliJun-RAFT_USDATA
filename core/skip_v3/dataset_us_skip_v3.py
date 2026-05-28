import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import math
from glob import glob
from core.utils import frame_utils
from core.skip_v3.augmentor_skip_v3 import FlowAugmentor_v3
import cv2
import joblib
import random
from scipy import interpolate


def get_dense_map(x, y, z, mask):
    """
    通过cubic插值获得稠密的光流图
    :param x: 已知数据的x坐标，水平方向
    :param y: 已知数据的y坐标，垂直方向
    :param z: 已知数据的值
    :param mask: 需要插值的位置的掩膜
    :return: 稠密的光流图
    """
    h = mask.shape[0]
    w = mask.shape[1]
    dense_map = np.zeros((h, w))
    f = interpolate.Rbf(x, y, z, function='cubic', copy=True, bounds_error=False, fill_value=None)
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                dense_map[i, j] = f(j, i)
    return dense_map


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = FlowAugmentor_v3(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.train_images_sequence_list = []
        self.train_flow_sequence_list = []
        self.train_mask_sequence_list = []
        self.train_pointsmask_sequence_list = []
        self.train_coordinate_sequence_list = []
        self.train_seg_mask_sequence_list = []
        self.train_boder_mask_list = []

        self.test_image_list = []
        self.test_flow_list = []
        self.test_mask_list = []
        self.test_seg_mask_list = []
        self.test_pointsmask_list = []
        self.test_coordinate_list = []

    def __getitem__(self, index):

        if self.is_test:
            # 读取灰度图
            img1 = frame_utils.read_gray_img(self.test_image_list[index][0])
            img2 = frame_utils.read_gray_img(self.test_image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # 堆叠成三通道
            img1 = np.resize(img1, (img1.shape[0], img1.shape[1], 1))
            img1 = np.concatenate((img1, img1, img1), axis=2)
            img2 = np.resize(img2, (img2.shape[0], img2.shape[1], 1))
            img2 = np.concatenate((img2, img2, img2), axis=2)

            # 读取flow mask
            mask_1 = frame_utils.read_gray_img(self.test_mask_list[index][0])
            mask_1 = torch.from_numpy(mask_1)
            mask_2 = frame_utils.read_gray_img(self.test_mask_list[index][1])
            mask_2 = torch.from_numpy(mask_2)

            # 读取分割mask
            seg_mask_1 = frame_utils.read_gray_img(self.test_seg_mask_list[index][0])
            seg_mask_2 = frame_utils.read_gray_img(self.test_seg_mask_list[index][1])
            # 转成类别标签，心室为2，心肌层为1，其余为0
            seg_mask_1[seg_mask_1 > 200] = 2
            seg_mask_1[seg_mask_1 > 100] = 1
            seg_mask_1 = torch.from_numpy(seg_mask_1).long()
            seg_mask_2[seg_mask_2 > 200] = 2
            seg_mask_2[seg_mask_2 > 100] = 1
            seg_mask_2 = torch.from_numpy(seg_mask_2).long()

            # 读取光流
            flow_1 = frame_utils.read_gen(self.test_flow_list[index][0])
            flow_1 = np.array(flow_1).astype(np.float32)
            flow_1 = torch.from_numpy(flow_1).permute(2, 0, 1).float()
            flow_2 = frame_utils.read_gen(self.test_flow_list[index][1])
            flow_2 = np.array(flow_2).astype(np.float32)
            flow_2 = torch.from_numpy(flow_2).permute(2, 0, 1).float()

            # 读取pointsmask
            pointsmask = frame_utils.read_gray_img(self.test_pointsmask_list[index])
            pointsmask = torch.from_numpy(pointsmask)

            # 读取坐标
            coordinate = joblib.load(self.test_coordinate_list[index])

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, flow_1, flow_2, mask_1, mask_2, seg_mask_1, seg_mask_2, pointsmask, coordinate

        index = index % len(self.train_images_sequence_list)
        # 最大跳帧数 以及 随机跳帧数
        max_skip = 16
        skip_gap = random.randint(1, max_skip)

        # 随机选择两帧
        sequence_frames = len(self.train_images_sequence_list[index])
        start_index = random.randint(0, sequence_frames - 1 - skip_gap)
        end_index = start_index + skip_gap
        # print(skip_gap, start_index, end_index)

        # 读取图像
        img1 = frame_utils.read_gray_img(self.train_images_sequence_list[index][start_index])
        img2 = frame_utils.read_gray_img(self.train_images_sequence_list[index][end_index])
        # 堆叠成三通道
        img1 = np.resize(img1, (img1.shape[0], img1.shape[1], 1))
        img1 = np.concatenate((img1, img1, img1), axis=2)
        img2 = np.resize(img2, (img2.shape[0], img2.shape[1], 1))
        img2 = np.concatenate((img2, img2, img2), axis=2)

        # 读取mask
        mask_1 = frame_utils.read_gray_img(self.train_mask_sequence_list[index][start_index])
        # mask_2 = frame_utils.read_gray_img(self.train_mask_sequence_list[index][end_index])
        boder_mask = frame_utils.read_gray_img(self.train_boder_mask_list[index])

        # 读取coordinate
        start_coordinate = joblib.load(self.train_coordinate_sequence_list[index][start_index])
        end_coordinate = joblib.load(self.train_coordinate_sequence_list[index][end_index])
        # print(start_coordinate[0])

        # 获取稀疏光流
        x = []
        y = []
        z_u = []
        z_v = []
        for i in range(start_coordinate.shape[0]):
            x.append(start_coordinate[i][0])
            y.append(start_coordinate[i][1])
            z_u.append(end_coordinate[i][0] - start_coordinate[i][0])
            z_v.append(end_coordinate[i][1] - start_coordinate[i][1])
        # 获取稠密光流图
        u_dense_map = get_dense_map(x, y, z_u, mask_1)
        # cv2.imshow('u_dense', u_dense_map)
        v_dense_map = get_dense_map(x, y, z_v, mask_1)
        # cv2.imshow('v_dense', v_dense_map)
        u_dense_map = np.resize(u_dense_map, (u_dense_map.shape[0], u_dense_map.shape[1], 1))
        v_dense_map = np.resize(v_dense_map, (v_dense_map.shape[0], v_dense_map.shape[1], 1))
        flow_1 = np.concatenate((u_dense_map, v_dense_map), axis=2)

        if self.augmentor is not None:
            img1, img2, flow_1, mask_1, boder_mask = self.augmentor(img1, img2, flow_1, mask_1, boder_mask)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow_1 = torch.from_numpy(flow_1).permute(2, 0, 1).float()
        mask_1 = torch.from_numpy(mask_1).float()
        boder_mask = torch.from_numpy(boder_mask).float()

        return img1, img2, flow_1, mask_1, boder_mask

    def __rmul__(self, v):
        self.train_images_sequence_list = v * self.train_images_sequence_list
        self.train_flow_sequence_list = v * self.train_flow_sequence_list
        return self

    def __len__(self):
        if self.is_test:
            return len(self.test_image_list)
        else:
            return len(self.train_images_sequence_list)


class USdata_skip(FlowDataset):
    def __init__(self, aug_params=None, test=False, root='/data/csj/data_accuracy/'):
        super(USdata_skip, self).__init__(aug_params, sparse=True)
        validation = 'ESAOTE'
        train_images_sequence_list, train_flow_sequence_list, train_mask_sequence_list, train_seg_mask_sequence_list, train_pointsmask_sequence_list, train_coordinate_sequence_list, train_boder_mask_list, test_images_list,                 test_flow_list, test_mask_list, test_seg_mask_list, test_pointsmask_list, test_coordinate_list = get_USData_list(root, validation)
        self.train_images_sequence_list = train_images_sequence_list
        self.train_flow_sequence_list = train_flow_sequence_list
        self.train_mask_sequence_list = train_mask_sequence_list
        self.train_seg_mask_sequence_list = train_seg_mask_sequence_list
        self.train_pointsmask_sequence_list = train_pointsmask_sequence_list
        self.train_coordinate_sequence_list = train_coordinate_sequence_list
        self.train_boder_mask_list = train_boder_mask_list

        self.is_test = test

        self.test_image_list = test_images_list
        self.test_flow_list = test_flow_list
        self.test_mask_list = test_mask_list
        self.test_seg_mask_list = test_seg_mask_list
        self.test_pointsmask_list = test_pointsmask_list
        self.test_coordinate_list = test_coordinate_list


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'usdata_skip':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.5, 'do_flip': True}
        train_dataset = USdata_skip(aug_params)
    else:
        train_dataset = None

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def get_USData_list(root_path, validation):
    suppliers = ['ESAOTE', 'GE Vingmed Ultrasound', 'Hitachi Aloka Medical,Ltd', 'Philips Medical Systems', 'SAMSUNG MEDISON CO', 'Siemens', 'TOSHIBA_MEC_US']
    train_images_sequence_list = []
    train_flow_sequence_list = []
    train_mask_sequence_list = []
    train_seg_mask_sequence_list = []
    train_pointsmask_sequence_list = []
    train_coordinate_sequence_list = []
    train_boder_mask_list = []

    test_images_list = []
    test_flow_list = []
    test_mask_list = []
    test_seg_mask_list = []
    test_pointsmask_list = []
    test_coordinate_list = []

    for sp in suppliers:
        if sp == validation:
            images_list, flow_list, mask_list, seg_mask_list, pointsmask_list, coordinate_list = get_subset_list(root_path, validation, test=True)
            test_images_list = test_images_list + images_list
            test_flow_list = test_flow_list + flow_list
            test_mask_list = test_mask_list + mask_list
            test_seg_mask_list = test_seg_mask_list + seg_mask_list
            test_pointsmask_list = test_pointsmask_list + pointsmask_list
            test_coordinate_list = test_coordinate_list + coordinate_list
        else:
            images_sequence_list, flow_sequence_list, mask_sequence_list, seg_mask_sequence_list, pointsmask_sequence_list, coordinate_sequence_list, boder_mask_list = get_subset_list(root_path, sp)
            train_images_sequence_list = train_images_sequence_list + images_sequence_list
            train_flow_sequence_list = train_flow_sequence_list + flow_sequence_list
            train_mask_sequence_list = train_mask_sequence_list + mask_sequence_list
            train_seg_mask_sequence_list = train_seg_mask_sequence_list + seg_mask_sequence_list
            train_pointsmask_sequence_list = train_pointsmask_sequence_list + pointsmask_sequence_list
            train_coordinate_sequence_list = train_coordinate_sequence_list + coordinate_sequence_list
            train_boder_mask_list = train_boder_mask_list + boder_mask_list

    return train_images_sequence_list, train_flow_sequence_list, train_mask_sequence_list, train_seg_mask_sequence_list, train_pointsmask_sequence_list, train_coordinate_sequence_list, train_boder_mask_list, test_images_list, test_flow_list, test_mask_list, test_seg_mask_list, test_pointsmask_list, test_coordinate_list


def get_subset_list(root_path, supplier, test=False):
    view_name = ['A4C', 'A3C', 'A2C']
    mode_name = ['laddist', 'ladprox', 'lcx', 'normal', 'rca']

    images_list = []
    flow_list = []
    mask_list = []
    pointsmask_list = []
    coordinate_list = []
    seg_mask_list = []

    images_sequence_list = []
    flow_sequence_list = []
    mask_sequence_list = []
    pointsmask_sequence_list = []
    coordinate_sequence_list = []
    seg_mask_sequence_list = []

    boder_mask_list = []

    for vn in view_name:
        for mn in mode_name:
            boder_mask_path = os.path.join(root_path, 'boder_mask', supplier, vn, mn, 'boder_mask', 'boder.png')

            image_dirs = os.path.join(root_path, supplier, vn, mn, 'frames')
            flow_dirs = os.path.join(root_path, supplier, vn, mn, 'flow')
            mask_dirs = os.path.join(root_path, supplier, vn, mn, 'mask')
            seg_mask_dirs = os.path.join(root_path, supplier, vn, mn, 'seg_mask')
            pointsmask_dirs = os.path.join(root_path, supplier, vn, mn, 'pointsmask')
            coordinate_dirs = os.path.join(root_path, supplier, vn, mn, 'coordinate')

            images = sorted(glob(os.path.join(image_dirs, '*.png')))
            flows = sorted(glob(os.path.join(flow_dirs, '*.flo')))
            masks = sorted(glob(os.path.join(mask_dirs, '*.png')))
            seg_masks = sorted(glob(os.path.join(seg_mask_dirs, '*.png')))

            if test:
                pointsmasks = sorted(glob(os.path.join(pointsmask_dirs, '*.png')))
                pointsmasks = pointsmasks[:-1]
                pointsmask_list = pointsmask_list + pointsmasks

                coordinates = sorted(glob(os.path.join(coordinate_dirs, '*.pkl')))
                coordinates = coordinates[:-1]
                coordinate_list = coordinate_list + coordinates

                for i in range(len(images) - 2):
                    images_list.append([images[i], images[i + 1]])
                    mask_list.append([masks[i], masks[i + 1]])
                    seg_mask_list.append([seg_masks[i], seg_masks[i + 1]])
                    flow_list.append([flows[i], flows[i + 1]])
            else:
                coordinates_sequence = sorted(glob(os.path.join(coordinate_dirs, '*.pkl')))
                pointsmask_sequence = sorted(glob(os.path.join(pointsmask_dirs, '*.png')))
                coordinates_num = len(coordinates_sequence)
                images_sequence = images[:coordinates_num]
                flow_sequence = flows
                mask_sequence = masks
                seg_mask_sequence = seg_masks[:coordinates_num]

                boder_mask_list.append(boder_mask_path)

                # print(len(coordinates_sequence))
                # print(len(images_sequence))
                # print(len(pointsmask_sequence))
                # print(len(flow_sequence))
                # print(len(mask_sequence))
                # print(len(seg_mask_sequence))

                images_sequence_list.append(images_sequence)
                flow_sequence_list.append(flow_sequence)
                mask_sequence_list.append(mask_sequence)
                pointsmask_sequence_list.append(pointsmask_sequence)
                coordinate_sequence_list.append(coordinates_sequence)
                seg_mask_sequence_list.append(seg_mask_sequence)

    if test:
        return images_list, flow_list, mask_list, seg_mask_list, pointsmask_list, coordinate_list
    else:
        return images_sequence_list, flow_sequence_list, mask_sequence_list, seg_mask_sequence_list, pointsmask_sequence_list, coordinate_sequence_list, boder_mask_list


if __name__ == '__main__':
    # root_path = 'F:/Dataset/USDATA/data_accuracy'
    # validation = 'ESAOTE'
    # images_sequence_list, flow_sequence_list, mask_sequence_list, seg_mask_sequence_list, pointsmask_sequence_list, coordinate_sequence_list = get_subset_list(root_path, validation)
    # print(len(images_sequence_list))
    # print(len(flow_sequence_list))
    # print(len(mask_sequence_list))
    # print(len(seg_mask_sequence_list))
    # print(len(pointsmask_sequence_list))
    # print(len(coordinate_sequence_list))
    # print(images_sequence_list[0:2])
    # print(flow_sequence_list[0:2])
    # print(mask_sequence_list[0:2])
    # print(seg_mask_sequence_list[0:2])
    # print(pointsmask_sequence_list[0:2])
    # print(coordinate_sequence_list[0:2])

    dataset = USdata_skip()
    for val_id in range(len(dataset)):
        _ = dataset[val_id]


