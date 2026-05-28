import numpy as np
import cv2
import math
from hausdorff import hausdorff_distance
from skimage.morphology import skeletonize as skelt

def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    if len(area) ==0 :
        print('1')
    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

def get_full_contour_list(mask_gray, gap):
    """
    获取整个轮廓点坐标
    :param mask_gray: 没有三个点的mask灰度图，numpy
    :param gap: 点的稀疏程度，越小越密
    :return: 坐标列表，list，每一个元素是
    """
    mask_gray[mask_gray>0]=255
    mask_gray = find_max_region(mask_gray)
    th = np.max(mask_gray) - 10
    ret, thresh = cv2.threshold(mask_gray, th, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_temp = np.array(contours)
    contours_temp = np.squeeze(contours_temp)
    contour_list = []
    for i in range(len(contours_temp)):
        # if i % gap == 0:
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


def get_partial_contour_list(left_point, right_point, contour_list):
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
    output = img.copy()
    for c in contours:
        cv2.circle(output, (int(c[0]), int(c[1])), 1, ct, 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
    return output


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
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    # print(top_point, left_point, right_point)
    # mask1_color = cv2.cvtColor()
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
    euclidean_dis = get_distance_spacing(contour_list_start, contour_list_end)
    return manhattan, euclidean, chebyshev, cosine, euclidean_dis

def get_distance_spacing(XA, XB):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmin > cmax and np.inf > cmin:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmin > cmax and np.inf > cmin:
            cmax = cmin
    return cmax

def distance_function(A,B):
    x_di = (A[0]-B[0])*0.3
    y_di = (A[1]-B[1])*0.15
    distance = math.sqrt(x_di**2 + y_di**2)
    return distance

def get_threepoints(mask3_gray, mask3):
    # mass_x, mass_y = np.where(mask3_gray >= 255)
    # # mass_x and mass_y are the list of x indices and y indices of mass pixels
    # cent_x = np.average(mass_x)
    # cent_y = np.average(mass_y)
    # print(int(cent_x), int(cent_y))
    contour, hierarchy = cv2.findContours(mask3_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mu = cv2.moments(contour[0])
    cent = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
    cent = (int(cent[0]), int(cent[1]))
    contours = []
    # cent = (int(cent_x), int(cent_y))
    cv2.circle(mask3, (cent), 2, (0, 0, 255), -1, 0)
    cv2.imshow('1', mask3)
    cv2.waitKey()
    line =cv2.Canny(mask3_gray, 30, 150)
    for i in range(line.shape[0]):
        for j in range(line.shape[1]):
            if line[i][j] >= 127:
                a = [j, i]
                a = np.array(a)
                contours.append(a)
    dis_max_1 = 0.0
    dis_max_2 = 0.0
    dis_max_3 = 0.0
    for i in range(len(contours)):
        dis = math.sqrt(sum([(a - b)**2 for (a,b) in zip(contours[i], cent)]))
        if contours[i][1] < cent[1]:
            if dis>dis_max_1:
                dis_max_1 = dis
                x_max_1 = contours[i][0]
                y_max_1 = contours[i][1]
        else:
            if contours[i][0] < cent[0]-50:
                if dis>dis_max_2:
                    dis_max_2 = dis
                    x_max_2 = contours[i][0]
                    y_max_2 = contours[i][1]
            if contours[i][0] > cent[0]+50:
                if dis>dis_max_3:
                    dis_max_3 = dis
                    x_max_3 = contours[i][0]
                    y_max_3 = contours[i][1]

    top_point = (x_max_1, y_max_1)
    left_point = (x_max_2, y_max_2)
    right_point = (x_max_3, y_max_3)
    print(top_point, left_point, right_point)
    cv2.circle(mask3, (cent), 2, (0, 0, 255), -1, 0)
    cv2.circle(mask3, (top_point), 2, (0, 0, 255), -1, 0)
    cv2.circle(mask3, (left_point), 2, (0, 0, 255), -1, 0)
    cv2.circle(mask3, (right_point), 2, (0, 0, 255), -1, 0)
    cv2.imshow('points', mask3)
    cv2.waitKey()
    return cent, top_point, left_point, right_point

def uniform_sampling_curve(curve, N):
    """
    对包含(x, y)坐标的曲线进行均匀采样，返回N个采样点的坐标。

    参数：
    curve: 包含(x, y)坐标的列表或NumPy数组。
    N: 采样点数量。

    返回值：
    一个包含N个坐标点的列表或NumPy数组。
    """
    curve_length = len(curve)
    if N >= curve_length:
        return curve

    # 计算每个采样点之间的间距
    spacing = curve_length / (N - 1)

    # 创建一个空列表来存储采样点的坐标
    sampled_curve = []

    # 对曲线进行均匀采样
    for i in range(N-1):
        index = int(i * spacing)
        sampled_curve.append(curve[index])

    return sampled_curve

def dfs(x, y, visited, points, gray):
    # 将当前点加入坐标点列表
    points.append((x, y))

    # 遍历当前点的八连通域内的所有点
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            # 判断点是否在图像范围内，并且是否已经访问过
            if i >= 0 and i < gray.shape[1] and j >= 0 and j < gray.shape[0] and not visited[j, i]:
                # 判断点是否在曲线上
                if gray[j, i] != 0:
                    visited[j, i] = True
                    dfs(i, j, visited, points, gray)

def get_keypoints_manual_se(start_point, end_point, full_contour_list):
    """
    获取人工标注的三个关键点，方法为提取三个轮廓取均值得到中心点，然后寻找与三个中心点的最近点
    :param mask_gray: 标注有三个关键点的灰度图
    :param full_contour_list: 完整的轮廓坐标列表
    :return: 三个关键点，np.array[w_index, h_index]
    """
    class1_point = full_contour_list[0]
    class2_point = start_point
    class3_point = end_point
    # print(class1_point, class2_point, class3_point)
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

def get_line(mask1, mask2=None):
    if mask2 is not None:
        mask_temp = mask1 + mask2
        mask1_line = cv2.Canny(mask1, 30, 150)
        mask2_line = cv2.Canny(mask2, 30, 150)
        mask_temp = cv2.Canny(mask_temp, 30, 150)
        line = mask1_line + mask2_line -mask_temp
    else:
        line = cv2.Canny(mask1, 30, 150)
    _, thresh = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = 255 - thresh
    # thresh[thresh != 0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # 对图像进行闭运算
    img_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_closed = cv2.morphologyEx(img_closed, cv2.MORPH_CLOSE, kernel)
    img_closed[img_closed != 0] = 1
    thinned = skelt(img_closed)
    # thinned = cv2.ximgproc.thinning(thresh, cv2.ximgproc.THINNING_ZHANGSUEN)
    # thresh[thresh!=0]=1
    # thinned = skelt(thresh)
    # thinned = cv2.ximgproc.thinning(thresh, cv2.ximgproc.THINNING_ZHANGSUEN)
    thinned = thinned.astype(np.uint8) * 255
    height, width = thinned.shape
    # 找到曲线的起始点和结束点
    start_point = None
    end_point = None
    point_temp = None

    for y in range(height):
        for x in range(width):
            connected_count = 0
            if thresh[y, x] > 0:
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        if i >= 0 and i < width and j >= 0 and j < height:
                            if thinned[j, i] == 255 and (i, j) != (x, y):
                                connected_count += 1
                        if connected_count>1:
                            break
                    if connected_count>1:
                        break
            if connected_count==1:
                if start_point is None:
                    start_point = (x, y)
                else:
                    end_point = (x, y)
                    break
        if start_point is not None and end_point is not None:
            break
    # 初始化visited数组
    visited = np.zeros_like(thinned, dtype=bool)
    # 将起始点标记为已访问
    visited[start_point[1], start_point[0]] = True
    # 初始化坐标点列表
    points = []
    # 进行DFS搜索
    dfs(start_point[0], start_point[1], visited, points, thinned)
    # contours_full = get_full_contour_list(mask1, gap=1)
    # top_point, start_point, end_point = get_keypoints_manual_se(start_point, end_point, contours_full)
    
    # points = get_partial_contour_list(start_point, end_point, contours_full)
    points = uniform_sampling_curve(points, 100)
    return points
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

    ES_partial_contour_list = get_contour_list_manual(ES_mask_gray, ES_mask_2_gray, gap=3)
    ES_length = length_calculate(ES_partial_contour_list)
    print(ES_length)
    # l_temp = cv2.arcLength(np.array(ES_partial_contour_list), False)
    # print(l_temp)
    ES_mask_color = cv2.cvtColor(ES_mask_gray, cv2.COLOR_GRAY2RGB)
    ES_img = draw_contours_color(ES_mask_color, ES_partial_contour_list)
    #cv2.imshow('ESimg', ES_img)

    ED_partial_contour_list = get_contour_list_manual(ED_mask_gray, ED_mask_2_gray, gap=3)
    ED_length = length_calculate(ED_partial_contour_list)
    print(ED_length)
    ED_mask_color = cv2.cvtColor(ED_mask_gray, cv2.COLOR_GRAY2RGB)
    ED_img = draw_contours_color(ED_mask_color, ED_partial_contour_list)
    #cv2.imshow('EDimg', ED_img)

    gls = (ED_length - ES_length) / ES_length
    print('GLS:', gls)

    cv2.waitKey()




