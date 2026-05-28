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

    def color_transform(self, image_sequence):
        """ Photometric augmentation """

        # # asymmetric
        # if np.random.rand() < self.asymmetric_color_aug_prob:
        #     img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
        #     img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        #
        # # symmetric
        # else:
        #     image_stack = np.concatenate([img1, img2], axis=0)
        #     image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        #     img1, img2 = np.split(image_stack, 2, axis=0)

        if np.random.rand() > self.asymmetric_color_aug_prob:
            image_stack = np.concatenate(image_sequence, axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            image_sequence = np.split(image_stack, len(image_sequence), axis=0)

        return image_sequence

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

    def spatial_transform(self, image_sequence, seg_mask_1, seg_mask_2):
        # randomly sample scale
        ht, wd = image_sequence[0].shape[:2]
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
            for i in range(len(image_sequence)):
                img = image_sequence[i]
                img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                image_sequence[i] = img
            # for i in range(len(mask_sequence)):
            #     img = mask_sequence[i]
            #     img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     mask_sequence[i] = img
            #     img = mask2_sequence[i]
            #     img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     mask2_sequence[i] = img
            seg_mask_1 = cv2.resize(seg_mask_1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            seg_mask_2 = cv2.resize(seg_mask_2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                for i in range(len(image_sequence)):
                    img = image_sequence[i]
                    img = img[:, ::-1]
                    image_sequence[i] = img
                seg_mask_1 = seg_mask_1[:, ::-1]
                seg_mask_2 = seg_mask_2[:, ::-1]
                # for i in range(len(mask_sequence)):
                #     img = mask_sequence[i]
                #     img = img[:, ::-1]
                #     mask_sequence[i] = img
                #     img = mask2_sequence[i]
                #     img = img[:, ::-1]
                #     mask2_sequence[i] = img

            if np.random.rand() < self.v_flip_prob:  # v-flip
                for i in range(len(image_sequence)):
                    img = image_sequence[i]
                    img = img[::-1, :]
                    image_sequence[i] = img
                # for i in range(len(mask_sequence)):
                #     img = mask_sequence[i]
                #     img = img[::-1, :]
                #     mask_sequence[i] = img
                #     img = mask2_sequence[i]
                #     img = img[::-1, :]
                #     mask2_sequence[i] = img
                seg_mask_1 = seg_mask_1[::-1, :]
                seg_mask_2 = seg_mask_2[::-1, :]
        # print('image_shape_resize:', img1.shape[0], img2.shape[1])
        # print('crop_size', self.crop_size[0], self.crop_size[1])
        if (image_sequence[0].shape[0] - self.crop_size[0]) == 0 | (image_sequence[0].shape[1] - self.crop_size[1]) == 0:
            x0 = 0
            y0 = 0
        else:
            y0 = np.random.randint(0, image_sequence[0].shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, image_sequence[0].shape[1] - self.crop_size[1])
        # print(self.crop_size)
        # print(x0, y0)
        # print('2:'+str(image_sequence[0].shape))
        # print('self.crop_size[0]:' + str(self.crop_size[0]))

        for i in range(len(image_sequence)):
            img = image_sequence[i]
            img = img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            image_sequence[i] = img
        # for i in range(len(mask_sequence)):
        #     img = mask_sequence[i]
        #     img = img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        #     mask_sequence[i] = img
        #     img = mask2_sequence[i]
        #     img = img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        #     mask2_sequence[i] = img

        seg_mask_1 = seg_mask_1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        seg_mask_2 = seg_mask_2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        # print('3:'+str(image_sequence[0].shape))

        return image_sequence, seg_mask_1, seg_mask_2
    def __call__(self, image_sequence, mask1, mask2):
        image_sequence = self.color_transform(image_sequence)
        # img1, img2 = self.eraser_transform(img1, img2)
        image_sequence, mask1, mask2= self.spatial_transform(image_sequence, mask1, mask2)

        for i in range(len(image_sequence)):
            img = image_sequence[i]
            img = np.ascontiguousarray(img)
            image_sequence[i] = img
        # for i in range(len(mask_sequence)):
        #     img = mask_sequence[i]
        #     img = np.ascontiguousarray(img)
        #     mask_sequence[i] = img
        #     img = mask2_sequence[i]
        #     img = np.ascontiguousarray(img)
        #     mask2_sequence[i] = img
        mask1 = np.ascontiguousarray(mask1)
        mask2 = np.ascontiguousarray(mask2)



        return image_sequence, mask1, mask2