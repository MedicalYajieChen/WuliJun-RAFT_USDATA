import numpy as np
import cv2
import math
import numpy.matlib


def get_Gauss_shadow_mask(h, w):
    """
    生成高斯掩膜
    :param h: 掩膜高度
    :param w: 掩膜宽度
    :return:
    """
    IMAGE_WIDTH = h
    IMAGE_HEIGHT = w
    center_x = IMAGE_WIDTH / 2
    center_y = IMAGE_HEIGHT / 2
    R = np.sqrt(center_x ** 2 + center_y ** 2)
    # 直接利用矩阵运算实现
    mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)
    x1 = np.arange(IMAGE_WIDTH)
    x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)
    y1 = np.arange(IMAGE_HEIGHT)
    y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
    y_map = np.transpose(y_map)
    Gauss_shadow_mask = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)
    scale = np.random.uniform(0.3, 1)
    Gauss_shadow_mask = np.exp(-scale * Gauss_shadow_mask / R)
    Gauss_shadow_mask = 1 - Gauss_shadow_mask
    return Gauss_shadow_mask


def Gauss_mask(image1, image2, h=100, w=100):
    """
    使用高斯掩膜实现随机区域强度下降
    :param image1: 图像1
    :param image2: 图像2
    :param h: 强度降低区域高度
    :param w: 强度降低区域宽度
    :return: 随机区域强度降低后的图像
    """
    Gauss_shadow_mask = get_Gauss_shadow_mask(h, w)
    y = np.random.randint(0, image1.shape[0] - h)
    x = np.random.randint(0, image1.shape[1] - w)
    image1[y: y + h, x: x + w] = image1[y: y + h, x: x + w] * Gauss_shadow_mask
    image2[y: y + h, x: x + w] = image2[y: y + h, x: x + w] * Gauss_shadow_mask
    return image1, image2


def Random_mask(image1, image2, h=100, w=100):
    """
    使用随机掩膜实现随机区域强度下降
    :param image1: 图像1
    :param image2: 图像2
    :param h: 强度降低区域高度
    :param w: 强度降低区域宽度
    :return:
    """
    random_mask = np.random.random(size=(100, 100))
    y = np.random.randint(0, image1.shape[0] - h)
    x = np.random.randint(0, image1.shape[1] - w)
    image1[y: y + w, x: x + w] = image1[y: y + h, x: x + w] * random_mask
    image2[y: y + w, x: x + w] = image2[y: y + h, x: x + w] * random_mask
    return image1, image2


def Circle_mask(image1, image2, h=200, w=200):
    """
    使用圆形掩膜实现随机区域强度下降
    :param image1: 图像1
    :param image2: 图像2
    :param h: 强度降低区域的高度
    :param w: 强度降低区域的宽度
    :return:
    """
    R = min(h, w) / 2
    c_x = w / 2
    c_y = h / 2
    circle_mask = np.zeros((h, w))
    scale = np.random.uniform(0.6, 1)
    for i in range(h):
        for j in range(w):
            d = math.sqrt((i - c_x) ** 2 + (j - c_y) ** 2)
            if d <= R:
                circle_mask[i, j] = (R - d) * scale / R
    circle_mask = 1 - circle_mask
    y = np.random.randint(0, image1.shape[0] - h)
    x = np.random.randint(0, image1.shape[1] - w)
    image1[y: y + w, x: x + w] = image1[y: y + h, x: x + w] * circle_mask
    image2[y: y + w, x: x + w] = image2[y: y + h, x: x + w] * circle_mask
    return image1, image2


def Depth_attenuation(image1, image2):
    """
    沿径向方向强度衰减
    :param image1: 图像1
    :param image2: 图像2
    :return:
    """
    a = np.random.uniform(0.8, 1)
    b = np.random.uniform(0.3, 0.8)
    start = max(a, b)
    end = min(a, b)
    h = image1.shape[0]
    for i in range(h):
        image1[i, :] = image1[i, :] * ((end - start) / h * i + start)
        image2[i, :] = image2[i, :] * ((end - start) / h * i + start)
    return image1, image2


def Speckle_reduction(image1, image2):
    """
    双边滤波
    :param image1:
    :param image2:
    :return:
    """
    sigma_color = np.random.randint(10, 50)
    sigma_space = np.random.randint(10, 50)
    image1 = cv2.bilateralFilter(image1, 5, sigma_color, sigma_space)
    image2 = cv2.bilateralFilter(image2, 5, sigma_color, sigma_space)
    return image1, image2


def usdata_aug(image1, image2):
    """
    超声数据增强
    :param image1:
    :param image2:
    :return:
    """
    # 选择随机区域强度下降方式
    aug_mode = np.random.choice([0, 1, 2, 3], size=1, p=[0.25, 0.3, 0.25, 0.2])
    # print(aug_mode)
    if aug_mode[0] == 0:
        pass
    elif aug_mode[0] == 1:
        image1, image2 = Circle_mask(image1, image2)
    elif aug_mode[0] == 2:
        image1, image2 = Random_mask(image1, image2)
    elif aug_mode[0] == 3:
        image1, image2 = Gauss_mask(image1, image2)

    # 选择是否进行径向方向强度衰减
    aug_mode = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
    # print(aug_mode)
    if aug_mode[0] == 0:
        pass
    elif aug_mode[0] == 1:
        image1, image2 = Depth_attenuation(image1, image2)

    # 选择是否进行双边滤波
    aug_mode = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
    # print(aug_mode)
    if aug_mode[0] == 0:
        pass
    elif aug_mode[0] == 1:
        image1, image2 = Speckle_reduction(image1, image2)
    return image1, image2


if __name__ == '__main__':
    # out = np.random.choice([0, 1, 2, 3, 4], size=1, p=[0.1, 0.2, 0.3, 0.2, 0.2])
    # print(out)

    image1 = np.zeros((512, 512)).astype(np.uint8)
    image2 = np.zeros((512, 512)).astype(np.uint8)
    image1, image2 = usdata_aug(image1, image2)




