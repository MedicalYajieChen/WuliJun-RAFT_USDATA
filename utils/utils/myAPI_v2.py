import numpy as np
import cv2
import math
from hausdorff import hausdorff_distance


def get_full_contour_list(mask_gray, gap):
    """
    获取整个轮廓点坐标
    :param mask_gray: 没有三个点的mask灰度图，numpy
    :param gap: 点的稀疏程度，越小越密
    :return: 坐标列表，list，每一个元素是
    """
    th = np.max(mask_gray) - 10
    ret, thresh = cv2.threshold(mask_gray, th, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_temp = np.array(contours)
    contours_temp = np.squeeze(contours_temp)

    contour_list = []
    for i in range(len(contours_temp)):
        if i % gap == 0:
            contour_list.append(contours_temp[i])

    return contour_list


def get_keypoints(mask_gray, contour_list):
    """
    通过计算距离的方式获取三个关键点，顶点、左下角点、右下角点
    :param mask_gray: mask灰度图
    :param contour_list: 整个轮廓点坐标列表
    :return: 三个点的坐标，[w_index, h_index]，numpy数组
    """
    h, w = mask_gray.shape
    bottom_left_point = np.array([0, h])
    bottom_right_point = np.array([w, h])
    top_point = None
    left_point = None
    right_point = None
    h_min = None
    l_d_min = None
    r_d_min = None
    for c in contour_list:
        c_h = c[1]
        c_l_d = math.pow(c[0] - bottom_left_point[0], 2) + math.pow(c[1] - bottom_left_point[1], 2)
        c_r_d = math.pow(c[0] - bottom_right_point[0], 2) + math.pow(c[1] - bottom_right_point[1], 2)
        if h_min is None or c_h < h_min:
            h_min = c_h
            top_point = c
        if l_d_min is None or c_l_d < l_d_min:
            l_d_min = c_l_d
            left_point = c
        if r_d_min is None or c_r_d < r_d_min:
            r_d_min = c_r_d
            right_point = c
    return top_point, left_point, right_point


def get_partial_contour_list(top_point, left_point, right_point, contour_list):
    """
    获取内膜轮廓坐标列表
    :param top_point: 顶点坐标
    :param left_point: 左下角点坐标
    :param right_point: 右下角点坐标
    :param contour_list: 整个轮廓点坐标列表
    :return: 心肌轮廓坐标列表
    """
    left_point_index = None
    right_point_index = None
    # contour_list中索引0的元素即为顶点，然后逆时针旋转，因此排序较为简单，只需切割列表后拼接即可
    for i in range(len(contour_list)):
        if (contour_list[i] == left_point).all():
            left_point_index = i
        if (contour_list[i] == right_point).all():
            right_point_index = i
    left_partial_list = contour_list[:left_point_index+1]
    right_partial_list = contour_list[right_point_index:-1]
    partial_contour_list = right_partial_list + left_partial_list
    return partial_contour_list


def get_contour_list(mask_gray, gap=5):
    """
    调用前面的函数，获取内膜轮廓坐标列表
    :param mask_gray:
    :param gap:
    :return:
    """
    contour_list = get_full_contour_list(mask_gray, gap)
    top_point, left_point, right_point = get_keypoints(mask_gray, contour_list)
    partial_contour_list = get_partial_contour_list(top_point, left_point, right_point, contour_list)
    return partial_contour_list


def draw_contours_color(img, contours, color='red'):
    """
    :param img: 彩色图像
    :param contours: 绘制点坐标列表，[[1, 2], [5, 2], .....], index 0是w方向坐标, index 1是h方向坐标
    :param color: 颜色
    :return:
    """
    if color is 'red':
        ct = (0, 0, 255)
    elif color is 'green':
        ct = (0, 255, 0)
    else:
        ct = (255, 0, 0)
    for c in contours:
        cv2.circle(img, (int(c[0]), int(c[1])), 1, ct, 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
    return img


def length_calculate(list):
    """
    计算轮廓长度
    :param list：内膜轮廓坐标
    :return: length：轮廓总长度
    """
    length = 0.0
    for i in range(len(list) - 1):
        length = length + (math.sqrt((list[i][0] - list[i + 1][0]) ** 2 + (list[i][1] - list[i + 1][1]) ** 2))
    return length


def get_GLS(contour_list_start, contour_list_end):
    """
    计算GLS
    :param contour_list_start: 起始轮廓点列表，list
    :param contour_list_end: 末尾轮廓点列表，list
    :return:GLS值
    """
    L_start = length_calculate(contour_list_start)
    L_end = length_calculate(contour_list_end)
    GLS = (L_end - L_start) / L_start
    return GLS


################################
def get_keypoints_manual(mask_gray, full_contour_list):
    """
    获取人工标注的三个关键点，方法为提取三个轮廓取均值得到中心点，然后寻找与三个中心点的最近点
    :param mask_gray: 标注有三个关键点的灰度图
    :param full_contour_list: 完整的轮廓坐标列表
    :return: 三个关键点，np.array[w_index, h_index]
    """
    th = np.max(mask_gray) - 10
    mask_gray[mask_gray < th] = 0
    mask_gray[mask_gray > th] = 255

    ret, thresh = cv2.threshold(mask_gray, th, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    assert len(contours) == 3
    class1_point_array = contours[0].squeeze()
    class2_point_array = contours[1].squeeze()
    class3_point_array = contours[2].squeeze()
    # print(class1_point_array.shape)

    class1_point = np.mean(class1_point_array, axis=0)
    class2_point = np.mean(class2_point_array, axis=0)
    class3_point = np.mean(class3_point_array, axis=0)
    # print(class1_point, class2_point, class3_point)

    w_list = [class1_point[0], class2_point[0], class3_point[0]]
    h_list = [class1_point[1], class2_point[1], class3_point[1]]

    min_h_value = min(h_list)
    min_h_index = h_list.index(min_h_value)
    top_point_temp = np.array([w_list.pop(min_h_index), h_list.pop(min_h_index)])

    min_w_value = min(w_list)
    min_w_index = w_list.index(min_w_value)
    left_point_temp = np.array([w_list.pop(min_w_index), h_list.pop(min_w_index)])
    right_point_temp = np.array([w_list[0], h_list[0]])

    min_d_top = None
    min_d_left = None
    min_d_right = None

    top_point = None
    left_point = None
    right_point = None
    for c in full_contour_list:
        d_top = math.sqrt((c[0] - top_point_temp[0])**2 + (c[1] - top_point_temp[1])**2)
        d_left = math.sqrt((c[0] - left_point_temp[0])**2 + (c[1] - left_point_temp[1])**2)
        d_right = math.sqrt((c[0] - right_point_temp[0])**2 + (c[1] - right_point_temp[1])**2)
        if min_d_top is None or d_top < min_d_top:
            min_d_top = d_top
            top_point = c
        if min_d_left is None or d_left < min_d_left:
            min_d_left = d_left
            left_point = c
        if min_d_right is None or d_right < min_d_right:
            min_d_right = d_right
            right_point = c
    # print(top_point_temp, left_point_temp, right_point_temp)
    # print(top_point, left_point, right_point)
    return top_point, left_point, right_point


def get_contour_list_manual(mask1_gray, mask2_gray, gap=5):
    contour_list = get_full_contour_list(mask1_gray, gap)
    top_point, left_point, right_point = get_keypoints_manual(mask2_gray, contour_list)
    partial_contour_list = get_partial_contour_list(top_point, left_point, right_point, contour_list)
    return partial_contour_list
#########################################################


def get_hd95(contour_list_start, contour_list_end):
    """
    计算hd95
    :param contour_list_start: 起始轮廓点列表，list
    :param contour_list_end: 末尾轮廓点列表，list
    :return:hd95值
    """
    contour_list_start = np.array(contour_list_start)
    contour_list_end = np.array(contour_list_end)
    manhattan = hausdorff_distance(contour_list_start, contour_list_end, distance="manhattan")  # 曼哈顿距离
    euclidean = hausdorff_distance(contour_list_start, contour_list_end, distance="euclidean")  # 欧氏距离
    chebyshev = hausdorff_distance(contour_list_start, contour_list_end, distance="chebyshev")  # 切比雪夫距离
    cosine = hausdorff_distance(contour_list_start, contour_list_end, distance="cosine")  # 余弦距离
    return manhattan, euclidean, chebyshev, cosine


######################################
# EF 计算
def get_lr_contour_list(top_point, left_point, right_point, contour_list):
    """
    将点列表分成左右两半
    :param top_point: 顶部点
    :param left_point: 左边点
    :param right_point: 右边点
    :param contour_list: 完整轮廓点列表
    :return: 左边点列表，右边点列表
    """
    left_point_index = None
    right_point_index = None
    # contour_list中索引0的元素即为顶点，然后逆时针旋转，因此排序较为简单，只需切割列表后拼接即可
    for i in range(len(contour_list)):
        if (contour_list[i] == left_point).all():
            left_point_index = i
        if (contour_list[i] == right_point).all():
            right_point_index = i
    # left_partial_list = contour_list[:left_point_index+1]
    left_partial_list = contour_list[:right_point_index]
    right_partial_list = contour_list[right_point_index:-1]
    return left_partial_list, right_partial_list


def get_corresponding_pointlist(left_partial_list, right_partial_list, top_point, middle_point):
    """
    获取对应点，对应点就是左边点与右边点的连线与中线L垂直的点
    :param left_partial_list: 左边点列表
    :param right_partial_list: 右边点列表
    :param top_point: 顶部点
    :param middle_point: 底部中点
    :return: 左边点与其右边点列表
    """
    left_corresponding_point = []
    right_corresponding_point = []
    if (top_point[0] - middle_point[0]) == 0:
        # 如果L线垂直
        for i in range(len(left_partial_list)):
            for j in range(len(right_partial_list)):
                if left_partial_list[i][1] == right_partial_list[j][1]:
                    left_corresponding_point.append(left_partial_list[i])
                    right_corresponding_point.append(right_partial_list[j])
                    break
    else:
        k1 = (top_point[1] - middle_point[1]) / (top_point[0] - middle_point[0])
        # print(k1)
        th = 0.05
        for i in range(len(left_partial_list)):
            for j in range(len(right_partial_list)):
                if (left_partial_list[i][0] - right_partial_list[j][0]) != 0:
                    k2 = (left_partial_list[i][1] - right_partial_list[j][1]) / (
                                left_partial_list[i][0] - right_partial_list[j][0])
                    if (-1-th) < k1 * k2 < (-1+th):
                        left_corresponding_point.append(left_partial_list[i])
                        right_corresponding_point.append(right_partial_list[j])
                        break
    # print(len(left_corresponding_point), len(right_corresponding_point))
    return left_corresponding_point, right_corresponding_point


def distance(point_1, point_2):
    """
    计算两点的欧式距离
    :param point_1: [1, 2], np.array
    :param point_2: [1, 2], np.array
    :return: 欧式距离
    """
    d = math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)
    return d


def get_volume(left_corresponding_point, right_corresponding_point, top_point, middle_point):
    """
    计算容积
    :param left_corresponding_point: 左边点
    :param right_corresponding_point: 右边对应点
    :param top_point: 顶部点
    :param middle_point: 底部中点
    :return: 体积
    """
    assert len(left_corresponding_point) == len(right_corresponding_point)
    L = distance(top_point, middle_point)
    volume = 0.0
    for i in range(len(left_corresponding_point)):
        a_b = distance(left_corresponding_point[i], right_corresponding_point[i])
        volume = a_b**2 + volume
    volume = math.pi / 4 * volume * L / len(left_corresponding_point)
    return volume


def get_EF(ES_mask_1, ES_mask_2, ED_mask_1, ED_mask_2, gap=1):
    """
    计算EF
    :param ES_mask_1: ES mask 灰度图
    :param ES_mask_2: ES 关键点mask 灰度图
    :param ED_mask_1: ED mask 灰度图
    :param ED_mask_2: ED 关键点mask 灰度图
    :param gap: 取点稠密程度
    :return: EF
    """
    ES_full_contour_list = get_full_contour_list(ES_mask_1, gap=gap)
    top_point, left_point, right_point = get_keypoints_manual(ES_mask_2, ES_full_contour_list)
    middle_point = (left_point + right_point) / 2
    ES_left_contour_list, ES_right_contour_list = get_lr_contour_list(top_point, left_point, right_point, ES_full_contour_list)
    ES_left_correspoint_list, ES_right_correspoint_list = get_corresponding_pointlist(ES_left_contour_list,
                                                                                      ES_right_contour_list, top_point,
                                                                                      middle_point)
    ES_volume = get_volume(ES_left_correspoint_list, ES_right_correspoint_list, top_point, middle_point)

    # # 显示ES计算过程
    # ES_mask_1_color = cv2.cvtColor(ES_mask_1, cv2.COLOR_GRAY2RGB)
    # for i in range(len(ES_left_correspoint_list)):
    #     cv2.line(ES_mask_1_color, tuple(ES_left_correspoint_list[i]), tuple(ES_right_correspoint_list[i]), (255, 255, 0), 1)
    # cv2.circle(ES_mask_1_color, (int(top_point[0]), int(top_point[1])), 1, (255, 0, 0), 1)
    # cv2.circle(ES_mask_1_color, (int(middle_point[0]), int(middle_point[1])), 1, (255, 0, 0), 1)
    # cv2.imshow('ES_mask_1_color', ES_mask_1_color)

    ED_full_contour_list = get_full_contour_list(ED_mask_1, gap=gap)
    top_point, left_point, right_point = get_keypoints_manual(ED_mask_2, ED_full_contour_list)
    middle_point = (left_point + right_point) / 2
    ED_left_contour_list, ED_right_contour_list = get_lr_contour_list(top_point, left_point, right_point, ED_full_contour_list)
    ED_left_correspoint_list, ED_right_correspoint_list = get_corresponding_pointlist(ED_left_contour_list,
                                                                                      ED_right_contour_list, top_point,
                                                                                      middle_point)
    ED_volume = get_volume(ED_left_correspoint_list, ED_right_correspoint_list, top_point, middle_point)

    # # 显示ED计算过程
    # ED_mask_1_color = cv2.cvtColor(ED_mask_1, cv2.COLOR_GRAY2RGB)
    # for i in range(len(ED_left_correspoint_list)):
    #     cv2.line(ED_mask_1_color, tuple(ED_left_correspoint_list[i]), tuple(ED_right_correspoint_list[i]),
    #              (255, 255, 0), 1)
    # cv2.circle(ED_mask_1_color, (int(top_point[0]), int(top_point[1])), 1, (255, 0, 0), 1)
    # cv2.circle(ED_mask_1_color, (int(middle_point[0]), int(middle_point[1])), 1, (255, 0, 0), 1)
    # cv2.imshow('ED_mask_1_color', ED_mask_1_color)

    EF = (ES_volume - ED_volume) / ES_volume
    return EF
######################################


if __name__ == '__main__':
    # 使用自定义方法找关键点
    # ES_mask_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask1/00034.png'
    # ES_mask_gray = cv2.imread(ES_mask_path, cv2.IMREAD_GRAYSCALE)

    # ED_mask_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask1/00056.png'
    # ED_mask_gray = cv2.imread(ED_mask_path, cv2.IMREAD_GRAYSCALE)
    #
    # ES_partial_contour_list = get_contour_list(ES_mask_gray, gap=6)
    # ES_length = length_calculate(ES_partial_contour_list)
    # print(ES_length)
    # ES_mask_color = cv2.cvtColor(ES_mask_gray, cv2.COLOR_GRAY2RGB)
    # ES_img = draw_contours_color(ES_mask_color, ES_partial_contour_list)
    # cv2.imshow('ESimg', ES_img)
    #
    # ED_partial_contour_list = get_contour_list(ED_mask_gray, gap=6)
    # ED_length = length_calculate(ED_partial_contour_list)
    # print(ED_length)
    # ED_mask_color = cv2.cvtColor(ED_mask_gray, cv2.COLOR_GRAY2RGB)
    # ED_img = draw_contours_color(ED_mask_color, ED_partial_contour_list)
    # cv2.imshow('EDimg', ED_img)
    #
    # gls = (ED_length - ES_length) / ES_length
    # print(gls)
    #
    # cv2.waitKey()

    # 使用人工标注关键点
    # ES_mask_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask1/00034.png'
    # ES_mask_gray = cv2.imread(ES_mask_path, cv2.IMREAD_GRAYSCALE)
    # ES_mask_2_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask2/00034.png'
    # ES_mask_2_gray = cv2.imread(ES_mask_2_path, cv2.IMREAD_GRAYSCALE)
    #
    # ED_mask_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask1/00056.png'
    # ED_mask_gray = cv2.imread(ED_mask_path, cv2.IMREAD_GRAYSCALE)
    # ED_mask_2_path = 'F:/Dataset/xiehe_usdata_0/11/11-mask2/00056.png'
    # ED_mask_2_gray = cv2.imread(ED_mask_2_path, cv2.IMREAD_GRAYSCALE)

    ES_mask_path = 'F:/Dataset/xiehe_usdata_0/32/32-mask1/00027.png'
    ES_mask_gray = cv2.imread(ES_mask_path, cv2.IMREAD_GRAYSCALE)
    ES_mask_2_path = 'F:/Dataset/xiehe_usdata_0/32/32-mask2/00027.png'
    ES_mask_2_gray = cv2.imread(ES_mask_2_path, cv2.IMREAD_GRAYSCALE)

    ED_mask_path = 'F:/Dataset/xiehe_usdata_0/32/32-mask1/00044.png'
    ED_mask_gray = cv2.imread(ED_mask_path, cv2.IMREAD_GRAYSCALE)
    ED_mask_2_path = 'F:/Dataset/xiehe_usdata_0/32/32-mask2/00044.png'
    ED_mask_2_gray = cv2.imread(ED_mask_2_path, cv2.IMREAD_GRAYSCALE)

    # ES_partial_contour_list = get_contour_list_manual(ES_mask_gray, ES_mask_2_gray, gap=3)
    # ES_length = length_calculate(ES_partial_contour_list)
    # print(ES_length)
    # ES_mask_color = cv2.cvtColor(ES_mask_gray, cv2.COLOR_GRAY2RGB)
    # ES_img = draw_contours_color(ES_mask_color, ES_partial_contour_list)
    # cv2.imshow('ESimg', ES_img)
    #
    # ED_partial_contour_list = get_contour_list_manual(ED_mask_gray, ED_mask_2_gray, gap=3)
    # ED_length = length_calculate(ED_partial_contour_list)
    # print(ED_length)
    # ED_mask_color = cv2.cvtColor(ED_mask_gray, cv2.COLOR_GRAY2RGB)
    # ED_img = draw_contours_color(ED_mask_color, ED_partial_contour_list)
    # cv2.imshow('EDimg', ED_img)
    #
    # gls = (ED_length - ES_length) / ES_length
    # print('GLS:', gls)

    EF = get_EF(ES_mask_gray, ES_mask_2_gray, ED_mask_gray, ED_mask_2_gray)
    print('EF:', EF)
    # cv2.waitKey()

