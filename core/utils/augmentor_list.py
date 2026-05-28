import numpy as np
import random
import math
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))
        # print('image_shape:', img1.shape[0], img2.shape[1])
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # print('image_shape_resize:', img1.shape[0], img2.shape[1])
        # print('crop_size', self.crop_size[0], self.crop_size[1])
        if (img1.shape[0] - self.crop_size[0]) == 0 | (img1.shape[1] - self.crop_size[1]) == 0:
            x0 = 0
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        # print(self.crop_size)
        # print(x0, y0)

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    # def random_intensity_reduction(self, img1, img2, max_reduction_ratio=0.5, max_reduction_size=0.3):
    #     """在图像中随机放置强度降低的区域
    #
    #     Args:
    #         img: 待增强的图像，要求通道数为3
    #         max_reduction_ratio: 强度降低比例的最大值
    #         max_reduction_size: 强度降低区域大小相对于图像尺寸的最大值
    #
    #     Returns:
    #         增强后的图像
    #     """
    #     height, width, _ = img1.shape
    #
    #     # 随机生成强度降低区域
    #     x1 = random.randint(0, width)
    #     y1 = random.randint(0, height)
    #     size = int(min(height, width) * max_reduction_size)
    #     x2 = min(x1 + size, width)
    #     y2 = min(y1 + size, height)
    #     ratio = random.uniform(0, max_reduction_ratio)
    #
    #     # 对区域内像素值进行随机降低
    #     new_value = img1[y1:y2, x1:x2, :] * (1 - ratio)
    #     img1[y1:y2, x1:x2, :] = new_value
    #
    #     # 随机生成强度降低区域
    #     x1 = random.randint(0, width)
    #     y1 = random.randint(0, height)
    #     size = int(min(height, width) * max_reduction_size)
    #     x2 = min(x1 + size, width)
    #     y2 = min(y1 + size, height)
    #     ratio = random.uniform(0, max_reduction_ratio)
    #
    #     # 对区域内像素值进行随机降低
    #     new_value = img2[y1:y2, x1:x2, :] * (1 - ratio)
    #     img2[y1:y2, x1:x2, :] = new_value
    #
    #     return img1, img2

    def __call__(self, imgs, flows):
        for i in range(len(imgs)):
            imgs[i][0], imgs[i][1] = self.color_transform(imgs[i][0], imgs[i][1])
            imgs[i][0], imgs[i][1] = self.eraser_transform(imgs[i][0], imgs[i][1])
            imgs[i][0], imgs[i][1], flows[i] = self.spatial_transform(imgs[i][0], imgs[i][1], flows[i])
        # img1, img2 = self.random_intensity_reduction(img1, img2)
            imgs[i][0] = np.ascontiguousarray(imgs[i][0])
            imgs[i][1] = np.ascontiguousarray(imgs[i][1])
            flows[i] = np.ascontiguousarray(flows[i])

        return imgs, flows


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    # def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
    #
    #     ht, wd = flow.shape[:2]
    #     coords = np.meshgrid(np.arange(wd), np.arange(ht))
    #     coords = np.stack(coords, axis=-1)
    #
    #     coords = coords.reshape(-1, 2).astype(np.float32)
    #     flow = flow.reshape(-1, 2).astype(np.float32)
    #     valid = valid.reshape(-1).astype(np.float32)
    #
    #     coords0 = coords[valid >= 1]
    #     flow0 = flow[valid >= 1]
    #
    #     ht1 = int(round(ht * fy))
    #     wd1 = int(round(wd * fx))
    #
    #     coords1 = coords0 * [fx, fy]
    #     flow1 = flow0 * [fx, fy]
    #
    #     xx = np.round(coords1[:, 0]).astype(np.int32)
    #     yy = np.round(coords1[:, 1]).astype(np.int32)
    #
    #     v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
    #     xx = xx[v]
    #     yy = yy[v]
    #     flow1 = flow1[v]
    #
    #     flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
    #     valid_img = np.zeros([ht1, wd1], dtype=np.int32)
    #
    #     flow_img[yy, xx] = flow1
    #     valid_img[yy, xx] = 1
    #
    #     return flow_img, valid_img

    def spatial_transform(self, img1, img2, mask1,mask2):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            mask1 = cv2.resize(mask1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            mask2 = cv2.resize(mask2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                mask1 = mask1[:, ::-1]
                mask2 = mask2[:, ::-1]
                # flow = flow[:, ::-1] * [-1.0, 1.0]
                # valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        a = -margin_x<(img1.shape[1] - self.crop_size[1] + margin_x)
        if a:
            x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)
        else:
            x0 = np.random.randint(img1.shape[1] - self.crop_size[1] + margin_x, -margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        mask1 = mask1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        mask2 = mask2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        # flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        # valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, mask1, mask2

    def random_intensity_reduction(self, img1, img2, max_reduction_ratio=0.5, max_reduction_size=0.3):
        """在图像中随机放置强度降低的区域

        Args:
            img: 待增强的图像，要求通道数为3
            max_reduction_ratio: 强度降低比例的最大值
            max_reduction_size: 强度降低区域大小相对于图像尺寸的最大值

        Returns:
            增强后的图像
        """
        height, width, _ = img1.shape

        # 随机生成强度降低区域
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        size = int(min(height, width) * max_reduction_size)
        x2 = min(x1 + size, width)
        y2 = min(y1 + size, height)
        ratio = random.uniform(0, max_reduction_ratio)

        # 对区域内像素值进行随机降低
        new_value = img1[y1:y2, x1:x2, :] * (1 - ratio)
        img1[y1:y2, x1:x2, :] = new_value
        # 随机生成强度降低区域
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        size = int(min(height, width) * max_reduction_size)
        x2 = min(x1 + size, width)
        y2 = min(y1 + size, height)
        ratio = random.uniform(0, max_reduction_ratio)

        # 对区域内像素值进行随机降低
        new_value = img2[y1:y2, x1:x2, :] * (1 - ratio)
        img2[y1:y2, x1:x2, :] = new_value

        return img1, img2

    def __call__(self, imgs, masks):
        for i in range(len(imgs)):
            if (i%2==0):
                imgs[i], imgs[i+1] = self.color_transform(imgs[i], imgs[i+1])
                imgs[i], imgs[i+1] = self.eraser_transform(imgs[i], imgs[i+1])
                imgs[i], imgs[i+1], masks[i], masks[i+1] = self.spatial_transform(imgs[i], imgs[i+1], masks[i], masks[i+1])
            # img1, img2 = self.random_intensity_reduction(img1, img2)
                imgs[i] = np.ascontiguousarray(imgs[i])
                imgs[i] = np.ascontiguousarray(imgs[i])
                masks[i] = np.ascontiguousarray(masks[i])

        return imgs, masks

