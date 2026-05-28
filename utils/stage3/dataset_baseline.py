import os
import numpy as np
import torch
import torch.utils.data as data
from glob import glob
from utils import frame_utils
from utils.stage3.augmentor_ori import FlowAugmentor
import cv2
import copy


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)
        self.is_test = False
        self.init_seed = False
        self.train_sequence_list = []
        self.train_mask_list = []
        # self.train_mask2_list = []
        self.test_sequence_list = []
        self.test_mask_list = []
        # self.test_mask2_list = []

    def __getitem__(self, index):
        size = [512, 512]
        if self.is_test:
            # 读取mask
            seg_mask_1 = frame_utils.read_gray_img(self.test_mask_list[index][0])
            seg_mask_2 = frame_utils.read_gray_img(self.test_mask_list[index][-1])
            seg_mask_1[seg_mask_1>0]=1
            seg_mask_2[seg_mask_2>0]=1
            seg_mask_1 = cv2.resize(seg_mask_1, size)
            seg_mask_2 = cv2.resize(seg_mask_2, size)
            # 读取灰度图
            # mask2_sequence = []
            image_sequence = []
            for i in range(len(self.test_sequence_list[index])):
                img = frame_utils.read_gray_img(self.test_sequence_list[index][i])
                # mask2 = frame_utils.read_gray_img(self.test_mask2_list[index][i])
                # mask2 = cv2.resize(mask2, [64, 64])
                # mask2[mask2 > 0] = 5
                # mask2 = torch.from_numpy(mask2)
                img = cv2.resize(img, size)
                img = np.array(img).astype(np.uint8)
                # 堆叠成三通道
                img = np.resize(img, (img.shape[0], img.shape[1], 1))
                img = np.concatenate((img, img, img), axis=2)
                image_sequence.append(img)
                # mask2_sequence.append(mask2)
            seg_mask_1 = torch.from_numpy(seg_mask_1).long()
            seg_mask_2 = torch.from_numpy(seg_mask_2).long()
            for i in range(len(image_sequence)):
                img = image_sequence[i]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                image_sequence[i] = img
            return image_sequence, seg_mask_1, seg_mask_2

        index = index % len(self.train_sequence_list)
        # 读取mask
        if isinstance(self.train_mask_list[index][0], list):
            mask1_1 = frame_utils.read_gray_img(self.train_mask_list[index][0][0])
            mask2_1 = frame_utils.read_gray_img(self.train_mask_list[index][0][-1])
            seg_mask_1 = mask2_1-mask1_1
            mask1_2 = frame_utils.read_gray_img(self.train_mask_list[index][-1][0])
            mask2_2 = frame_utils.read_gray_img(self.train_mask_list[index][-1][-1])
            seg_mask_2 = mask2_2-mask1_2
        else:
            seg_mask_1 = frame_utils.read_gray_img(self.train_mask_list[index][0])
            seg_mask_2 = frame_utils.read_gray_img(self.train_mask_list[index][-1])
        # print(self.train_mask_list[index][0], self.train_mask_list[index])
        seg_mask_1[seg_mask_1>0]=1
        seg_mask_2[seg_mask_2>0]=1
        
        seg_mask_1 = cv2.resize(seg_mask_1, size)
        seg_mask_2 = cv2.resize(seg_mask_2, size)
        ##心室置为1

        # 读取灰度图
        image_sequence = []
        # mask2_sequence = []
        # print(len(self.train_mask2_list[index]))
        for i in range(len(self.train_sequence_list[index])):
            img = frame_utils.read_gray_img(self.train_sequence_list[index][i])
            # print(self.test_mask2_list[index])
            # mask2 = frame_utils.read_gray_img(self.train_mask2_list[index][i])
            # mask2 = cv2.resize(mask2, size)
            # mask2[mask2>0]=5
            # mask2 = torch.from_numpy(mask2)
            img = cv2.resize(img, size)
            img = np.array(img).astype(np.uint8)
            # 堆叠成三通道
            img = np.resize(img, (img.shape[0], img.shape[1], 1))
            img = np.concatenate((img, img, img), axis=2)
            image_sequence.append(img)
            # mask2_sequence.append(mask2)
        if self.augmentor is not None:
            image_sequence, seg_mask_1, seg_mask_2 = self.augmentor(image_sequence, seg_mask_1, seg_mask_2)
        seg_mask_1 = torch.from_numpy(seg_mask_1).long()
        seg_mask_2 = torch.from_numpy(seg_mask_2).long()
        # print(len(mask2_sequence))
        # mask2_size_list = copy.copy(mask2_sequence)
        # for i in range(len(mask2_sequence)):
        #     mask2_sequence[i] = torch.from_numpy(mask2_sequence[i]).long()
        # for i in range(len(mask2_sequence)):
        #     mask2_sequence[i] = cv2.resize(mask2_sequence[i], [45, 45])
        #     mask2_sequence[i] = torch.from_numpy(mask2_sequence[i])

        for i in range(len(image_sequence)):
            img = image_sequence[i]
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            image_sequence[i] = img

        return image_sequence, seg_mask_1, seg_mask_2#, self.train_sequence_list[index][0]##debug

    def __rmul__(self, v):
        self.train_sequence_list = v * self.train_sequence_list
        return self

    def __len__(self):
        if self.is_test:
            return len(self.test_sequence_list)
        else:
            return len(self.train_sequence_list)


class USdata(FlowDataset):
    def __init__(self, aug_params=None, test=False, root='/data/csj/CAMUS23/'):
        # def __init__(self, aug_params=None, test=False, root='/mnt/nas/fmj/CAMUS/training'):
        super(USdata, self).__init__(aug_params, sparse=True)
        train_sequence_list, train_mask_list, test_sequence_list, test_mask_list = get_CAMUS_list(root)
        self.train_sequence_list = train_sequence_list
        self.train_mask_list = train_mask_list
        # self.train_mask2_list = train_mask2_list
        self.is_test = test
        self.test_sequence_list = test_sequence_list
        self.test_mask_list = test_mask_list
        # self.test_mask2_list = test_mask2_list

class STAGE3(FlowDataset):
    def __init__(self, aug_params=None, test=False):
        super(STAGE3, self).__init__(aug_params, sparse=True)
        self.is_test = test
        ##camus   tmi     union 
        camus_root='/data/csj/CAMUS23/'
        train_sequence_list, train_mask_list, test_sequence_list, test_mask_list = get_CAMUS_list(camus_root)
        self.train_sequence_list = self.train_sequence_list + train_sequence_list
        self.train_mask_list = self.train_mask_list + train_mask_list
        
        self.test_sequence_list =  self.test_sequence_list + test_sequence_list
        self.test_mask_list = self.test_mask_list + test_mask_list
        union_root = '/data/csj/20230725_400pro2'
        train_sequence_list, train_mask_list = get_xiehe_list(union_root)
        self.train_sequence_list = self.train_sequence_list + train_sequence_list
        self.train_mask_list = self.train_mask_list + train_mask_list
        # print(len(self.train_sequence_list), len(self.train_mask_list))
       
        tmi_root = '/data/csj/TMI_DATA_pro/'
        train_sequence_list, train_mask_list = get_tmi_list(tmi_root)
        self.train_sequence_list = self.train_sequence_list + train_sequence_list
        self.train_mask_list = self.train_mask_list + train_mask_list
        # class_list = os.listdir(tmi_root)
        # for c in class_list:
        #     pa_list = os.listdir(os.path.join(tmi_root, c))
        #     for pa in pa_list:
        #         images = sorted(glob(osp.join(camus_root, pa, ch, 'frames',  '*.png')))
        #         for i in range(len(images)-1):
        #                 self.image_list += [[images[i], images[i+1]]]

       



def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'camus':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.2, 'do_flip': True}
        train_dataset = USdata(aug_params)
    elif args.stage == 'stage3':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.2, 'do_flip': True}
        train_dataset = STAGE3(aug_params)
    else:
        train_dataset = None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=1, drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def get_CAMUS_list(root_path):
    train_sequence_list = []
    train_mask_list = []
    test_sequence_list = []
    test_mask_list = []
    # train_mask2_list = []
    # test_mask2_list = []
    file_lists = sorted(os.listdir(root_path))
    # train_file_lists = file_lists[0:400]
    # test_file_lists = file_lists[400:]
    test_file_lists = ['patient0027', 'patient0047', 'patient0051', 'patient0052', 'patient0187', 'patient0189',
                   'patient0191', 'patient0194', 'patient0197', 'patient0199', 'patient0201', 'patient0208',
                   'patient0213', 'patient0214', 'patient0215', 'patient0217', 'patient0218', 'patient0219',
                   'patient0220', 'patient0221', 'patient0223', 'patient0224', 'patient0225', 'patient0226',
                   'patient0227', 'patient0228', 'patient0231', 'patient0234', 'patient0237', 'patient0238',
                   'patient0239', 'patient0240', 'patient0241', 'patient0242', 'patient0243', 'patient0246',
                   'patient0248', 'patient0251', 'patient0252', 'patient0254', 'patient0258', 'patient0260',
                   'patient0261', 'patient0262', 'patient0263', 'patient0266', 'patient0269', 'patient0273',
                   'patient0275', 'patient0276']
    val_file_lists = ['patient0386', 'patient0387','patient0388', 'patient0389', 'patient0390', 'patient0392', 'patient0393', 'patient0396',
                       'patient0397', 'patient0399', 'patient0400', 'patient0401', 'patient0402', 'patient0403',
                       'patient0404', 'patient0405', 'patient0406', 'patient0407', 'patient0408', 'patient0409',
                       'patient0410', 'patient0411', 'patient0412', 'patient0413', 'patient0414', 'patient0415',
                       'patient0416', 'patient0417', 'patient0418', 'patient0419', 'patient0420', 'patient0421',
                       'patient0422', 'patient0423', 'patient0424', 'patient0425', 'patient0426', 'patient0427',
                       'patient0428', 'patient0429', 'patient0430', 'patient0431', 'patient0433', 'patient0434',
                       'patient0437', 'patient0438', 'patient0439', 'patient0441', 'patient0442', 'patient0450']

    bad_list = []

    for i, fl in enumerate(file_lists):
        if fl not in test_file_lists:
            if fl in val_file_lists:
                file_path = os.path.join(root_path, fl)
                images_list, seg_mask_list = get_subset_list(file_path)
                test_sequence_list = test_sequence_list + images_list
                # test_mask2_list = test_mask2_list + seg_mask2_list
                test_mask_list = test_mask_list + seg_mask_list
            else:
                file_path = os.path.join(root_path, fl)
                images_list, seg_mask_list = get_subset_list(file_path)
                train_sequence_list = train_sequence_list + images_list
                # train_mask2_list = train_mask2_list + seg_mask2_list
                train_mask_list = train_mask_list + seg_mask_list
    # print(len(train_mask2_list), len(test_mask2_list))
    return train_sequence_list, train_mask_list, test_sequence_list, test_mask_list


def get_subset_list(file_path):
    images_list = []
    seg_mask_list = []
    # seg_mask2_list = []
    file_path_2CH = os.path.join(file_path, '2CH')
    file_path_4CH = os.path.join(file_path, '4CH')
    file_path_2CH_frame = os.path.join(file_path_2CH, 'frame')
    file_path_2CH_mask = os.path.join(file_path_2CH, 'mask2')
    # file_path_2CH_mask2 = os.path.join(file_path_2CH, 'mask')
    images_list_2CH = sorted(glob(os.path.join(file_path_2CH_frame, '*.png')))
    seg_mask_list_2CH = sorted(glob(os.path.join(file_path_2CH_mask, '*.png')))
    # seg_mask2_list_2CH = sorted(glob(os.path.join(file_path_2CH_mask2, '*.png')))
    file_path_4CH_frame = os.path.join(file_path_4CH, 'frame')
    file_path_4CH_mask = os.path.join(file_path_4CH, 'mask2')
    # file_path_4CH_mask2 = os.path.join(file_path_4CH, 'mask')
    images_list_4CH = sorted(glob(os.path.join(file_path_4CH_frame, '*.png')))
    seg_mask_list_4CH = sorted(glob(os.path.join(file_path_4CH_mask, '*.png')))
    # seg_mask2_list_4CH = sorted(glob(os.path.join(file_path_4CH_mask2, '*.png')))
    SEQUENCE_LEN = 10
    step = (len(images_list_2CH) - 1) / (SEQUENCE_LEN - 1)
    temp_image_list_2CH = []
    # temp_mask2_list_2CH = []
    for k in range(SEQUENCE_LEN):
        temp_image_list_2CH.append(images_list_2CH[round(k * step)])
        # temp_mask2_list_2CH.append(seg_mask2_list_2CH[round(k * step)])
    step = (len(images_list_4CH) - 1) / (SEQUENCE_LEN - 1)
    temp_image_list_4CH = []
    temp_mask2_list_4CH = []
    for k in range(SEQUENCE_LEN):
        temp_image_list_4CH.append(images_list_4CH[round(k * step)])
        # temp_mask2_list_4CH.append(seg_mask2_list_4CH[round(k * step)])
    images_list.append(temp_image_list_2CH)
    images_list.append(temp_image_list_4CH)
    seg_mask_list.append([seg_mask_list_2CH[0], seg_mask_list_2CH[-1]])
    seg_mask_list.append([seg_mask_list_4CH[0], seg_mask_list_4CH[-1]])
    # seg_mask2_list.append(temp_mask2_list_2CH)
    # seg_mask2_list.append(temp_mask2_list_4CH)
    return images_list, seg_mask_list

def get_xiehe_list(root_path):
    train_sequence_list = []
    train_mask_list = []
    bad_list = []
    ef_list = os.listdir(root_path)
    for ef in ef_list:
        file_lists = sorted(os.listdir(os.path.join(root_path, ef)))
        for i, fl in enumerate(file_lists):
            if fl not in bad_list:
                file_path = os.path.join(root_path, ef, fl)
                images_list, seg_mask_list = get_subset2_list(file_path)
                train_sequence_list = train_sequence_list + images_list
                train_mask_list = train_mask_list + seg_mask_list
        # print(len(train_mask2_list), len(test_mask2_list))
    return train_sequence_list, train_mask_list


def get_subset2_list(path):
    images_list = []
    mask_list = []
    mask2_list = []
    # seg_mask2_list = []
    ch_list = os.listdir(path)
    for ch in ch_list:
        file_path = os.path.join(path, ch)
        file_path_frame = os.path.join(file_path, 'frames')
        file_path_mask = os.path.join(file_path, 'mask1')
        file_path_mask2 = os.path.join(file_path, 'mask2')
        images_ori_list = sorted(glob(os.path.join(file_path_frame, '*.png')))
        seg_mask_list = sorted(glob(os.path.join(file_path_mask, '*.png')))
        seg_mask2_list = sorted(glob(os.path.join(file_path_mask2, '*.png')))
        masks_name = sorted(os.listdir(file_path_mask))
        # print(file_path_mask, len(images_list))
        ES_index = int(masks_name[-1][:-4])
        ED_index = int(masks_name[0][:-4])
        SEQUENCE_LEN = 10
        step = (ES_index - ED_index) / (SEQUENCE_LEN - 1)
        temp_image_list = []
        for k in range(SEQUENCE_LEN):
            temp_image_list.append(images_ori_list[int(ED_index + k * step)])
        images_list.append(temp_image_list)
        mask_list.append([[seg_mask_list[0], seg_mask2_list[0]], [seg_mask_list[-1], seg_mask2_list[-1]]])
    return images_list, mask_list


def get_tmi_list(root_path):
    train_sequence_list = []
    train_mask_list = []
    bad_list = []
    class_list = os.listdir(root_path)
    for c in class_list:
        file_lists = sorted(os.listdir(os.path.join(root_path, c)))
        for i, fl in enumerate(file_lists):
            if fl not in bad_list:
                file_path = os.path.join(root_path, c, fl)
                images_list, seg_mask_list = get_subset3_list(file_path)
                train_sequence_list = train_sequence_list + images_list
                train_mask_list = train_mask_list + seg_mask_list
    return train_sequence_list, train_mask_list


def get_subset3_list(path):
    images_list = []
    mask_list = []
    # seg_mask2_list = []
    file_path_frame = os.path.join(path, 'frames')
    file_path_mask = os.path.join(path, 'mask')
    images_ori_list = sorted(glob(os.path.join(file_path_frame, '*.png')))
    seg_mask_list = sorted(glob(os.path.join(file_path_mask, '*.png')))
    # print(file_path_mask, len(images_list))
    SEQUENCE_LEN = 10
    step = (len(images_ori_list)//2) / (SEQUENCE_LEN - 1)
    temp_image_list = []
    for k in range(SEQUENCE_LEN):
        temp_image_list.append(images_ori_list[int(k * step)])
    images_list.append(temp_image_list)
    mask_list.append([seg_mask_list[0], seg_mask_list[len(images_ori_list)//2]])
    return images_list, mask_list

