import cv2
import numpy as np
import math


def get_points(gray):
    # 求质心点
    contour, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_index = 0
    max = 0
    for i in range(len(contour)):
        if len(contour[i])>max:
            max = len(contour[i])
            max_index = i
    mu = cv2.moments(contour[max_index])
    cent = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
    cent = (int(cent[0]), int(cent[1]))
    ##求轮廓
    line = cv2.Canny(gray, 30, 150)
    kernel = np.ones((2, 2), np.uint8)
    line = cv2.dilate(line, kernel, iterations=1)
    # cv2.imshow('line', line)
    # cv2.waitKey()
    contours = []
    for i in range(line.shape[0]):
        for j in range(line.shape[1]):
            if line[i][j] >= 127:
                a = [j, i]
                a = np.array(a)
                contours.append(a)
    dis_max_1 = 0.0
    y_max_2 = cent[1]
    y_max_3 = cent[1]
    top = 1000000
    for i in range(len(contours)):
        dis = math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(contours[i], cent)]))
        if contours[i][1] < cent[1]:
            if contours[i][1]<top:
                top = contours[i][1]
                x_max_1 = contours[i][0]
                y_max_1 = contours[i][1]
        #     if dis > dis_max_1:
        #         dis_max_1 = dis
        #         x_max_1 = contours[i][0]
        #         y_max_1 = contours[i][1]
        else:
            if contours[i][0] < cent[0]:
                if contours[i][1] > y_max_2:
                    x_max_2 = contours[i][0]
                    y_max_2 = contours[i][1]
            if contours[i][0] > cent[0]:
                if contours[i][1] > y_max_3:
                    x_max_3 = contours[i][0]
                    y_max_3 = contours[i][1]
    top_point = (x_max_1, y_max_1)
    left_point = (x_max_2, y_max_2)
    right_point = (x_max_3, y_max_3)
    # print(cent)
    # print(right_point)
    return cent, top_point, left_point, right_point, contours


def segment_6(mask, mask1, points=False):
    cent, top_point, left_point, right_point, contours = get_points(mask)
    ##等七分
    left_points_de = []
    right_points_de = []
    left_points_ma = []
    right_points_ma = []
    # cv2.circle(mask1, top_point, 2, (0, 0, 255), -1, 0)
    # cv2.circle(mask1, left_point, 2, (0, 0, 255), -1, 0)
    # cv2.circle(mask1, right_point, 2, (0, 0, 255), -1, 0)
    for k in [1, 3, 5]:
        left_point_s = (k * abs(top_point[0] - left_point[0]) // 7 + min(top_point[0], left_point[0]),
                        k * abs(top_point[1] - left_point[1]) // 7 + min(top_point[1], left_point[1]))
        right_point_s = (k * abs(top_point[0] - right_point[0]) // 7 + min(top_point[0], right_point[0]),
                         k * abs(top_point[1] - right_point[1]) // 7 + min(top_point[1], right_point[1]))
        left_points_de.append(left_point_s)
        right_points_de.append(right_point_s)

    for k in [2, 4, 6]:
        left_point_s = (k * abs(top_point[0] - left_point[0]) // 7 + min(top_point[0], left_point[0]),
                        k * abs(top_point[1] - left_point[1]) // 7 + min(top_point[1], left_point[1]))
        right_point_s = (k * abs(top_point[0] - right_point[0]) // 7 + min(top_point[0], right_point[0]),
                         k * abs(top_point[1] - right_point[1]) // 7 + min(top_point[1], right_point[1]))
        left_points_ma.append(left_point_s)
        right_points_ma.append(right_point_s)
    left_points_m = [[], [], []]
    right_points_m = [[], [], []]

    for i in range(len(contours)):
        if contours[i][0] < cent[0]:
            for m in range(len(left_points_ma)):
                if contours[i][1] == left_points_ma[m][1]:
                    left_points_m[m].append(contours[i])
        else:
            for n in range(len(right_points_ma)):
                if contours[i][1] == right_points_ma[n][1]:
                    right_points_m[n].append(contours[i])
    mass_points = []
    for i in range(3):
        mass_points.append(((left_points_m[i][0][0] + left_points_m[i][-1][0]) // 2,
                            ((left_points_m[i][0][1] + left_points_m[i][-1][1]) // 2)))
        mass_points.append(((right_points_m[i][0][0] + right_points_m[i][-1][0]) // 2,
                            ((right_points_m[i][0][1] + right_points_m[i][-1][1]) // 2)))

    left_points = [[], [], []]
    right_points = [[], [], []]

    for i in range(len(contours)):
        if contours[i][0] < cent[0]:
            for m in range(len(left_points_de)):
                if contours[i][1] == left_points_de[m][1]:
                    left_points[m].append(contours[i])
        else:
            for n in range(len(right_points_de)):
                if contours[i][1] == right_points_de[n][1]:
                    right_points[n].append(contours[i])

    point_color = (0, 0, 0)
    thickness = 1
    lineType = 4
    # cv2.line(mask1, left_points[0][0], right_points[0][-1], point_color, thickness, lineType)
    for i in range(len(left_points_de)):
        cv2.line(mask1, left_points[i][0], left_points[i][-1], point_color, thickness, lineType)
        cv2.line(mask1, right_points[i][0], right_points[i][-1], point_color, thickness, lineType)
    # 顶部分区
    k0 = (top_point[1] - cent[1]) / (top_point[0] - cent[0])
    b0 = top_point[1] - k0 * top_point[0]
    top_point2 = (0, 0)

    cv2.line(mask1, top_point, cent, point_color, thickness, lineType)
    if points:
        return [left_point, right_point, left_points[0][0], left_points[0][-1], right_points[0][0], right_points[0][-1],
                left_points[1][0], left_points[1][-1], right_points[1][0], right_points[1][-1], left_points[2][0],
                left_points[2][-1], right_points[2][0], right_points[2][-1],
                left_points_m[2][0], left_points_m[2][-1],
                left_points_m[1][0], left_points_m[1][-1],
                left_points_m[0][0], left_points_m[0][-1],
                right_points_m[0][0], right_points_m[0][-1],
                right_points_m[1][0], right_points_m[1][-1],
                right_points_m[2][0], right_points_m[2][-1]], mass_points, mask1
    else:
        return mask1


def get_6points_c(mask):
    cent, top_point, left_point, right_point, contours = get_points(mask)
    ##等七分
    left_points_de = []
    right_points_de = []

    for k in [1, 3, 5]:
        left_point_s = (k * abs(top_point[0] - left_point[0]) // 7 + min(top_point[0], left_point[0]),
                        k * abs(top_point[1] - left_point[1]) // 7 + min(top_point[1], left_point[1]))
        right_point_s = (k * abs(top_point[0] - right_point[0]) // 7 + min(top_point[0], right_point[0]),
                         k * abs(top_point[1] - right_point[1]) // 7 + min(top_point[1], right_point[1]))
        left_points_de.append(left_point_s)
        right_points_de.append(right_point_s)
    left_points = [[], [], []]
    right_points = [[], [], []]

    for i in range(len(contours)):
        if contours[i][0] < cent[0]:
            for m in range(len(left_points_de)):
                if contours[i][1] == left_points_de[m][1]:
                    left_points[m].append(contours[i])
        else:
            for n in range(len(right_points_de)):
                if contours[i][1] == right_points_de[n][1]:
                    right_points[n].append(contours[i])

    cen_point = []
    cen_point.append((((left_points[2][0][0] + left_points[2][-1][0]) / 2 + left_point[0]) / 2,
                      ((left_points[2][0][1] + left_points[2][-1][1]) / 2 + left_point[1]) / 2))
    cen_point.append(
        (((left_points[1][0][0] + left_points[1][-1][0]) / 2 + (left_points[2][0][0] + left_points[2][-1][0]) / 2) / 2,
         ((left_points[1][0][1] + left_points[1][-1][1]) / 2 + (left_points[2][0][1] + left_points[2][-1][1]) / 2) / 2))
    cen_point.append((
        ((left_points[1][0][0] + left_points[1][-1][0]) / 2 + left_points[0][0][0]) / 2,
        ((left_points[1][0][1] + left_points[1][-1][1]) / 2 + left_points[0][0][1]) / 2))
    cen_point.append((
        ((right_points[1][0][0] + right_points[1][-1][0]) / 2 + right_points[0][0][0]) / 2,
        ((right_points[1][0][1] + right_points[1][-1][1]) / 2 + right_points[0][0][1]) / 2))

    cen_point.append((
        ((right_points[1][0][0] + right_points[1][-1][0]) / 2 + (
                right_points[2][0][0] + right_points[2][-1][0]) / 2) / 2,
        ((right_points[1][0][1] + right_points[1][-1][1]) / 2 + (
                right_points[2][0][1] + right_points[2][-1][1]) / 2) / 2))
    cen_point.append((((right_points[2][0][0] + right_points[2][-1][0]) / 2 + right_point[0]) / 2,
                      ((right_points[2][0][1] + right_points[2][-1][1]) / 2 + right_point[1]) / 2))

    return cen_point

def get_points_in(mask1_gray, mask2_gray):
    # 求质心点
    contour, hierarchy = cv2.findContours(mask1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_index = 0
    max = 0
    for i in range(len(contour)):
        if len(contour[i])>max:
            max = len(contour[i])
            max_index = i
    mu = cv2.moments(contour[max_index])
    cent = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
    cent = (int(cent[0]), int(cent[1]))

    ##求轮廓
    line1 = cv2.Canny(mask1_gray, 30, 150)
    line2 = cv2.Canny(mask2_gray, 30, 150)
    mask = mask1_gray+mask2_gray
    mask[mask>0]=255
    line3 = cv2.Canny(mask, 30, 150)
    line = line1+line2-line3
    kernel = np.ones((2, 2), np.uint8)
    line = cv2.dilate(line, kernel, iterations=1)
    # cv2.imshow('line', line)
    # cv2.waitKey()
    contours = []
    for i in range(line.shape[0]):
        for j in range(line.shape[1]):
            if line[i][j] >= 127:
                a = [j, i]
                a = np.array(a)
                contours.append(a)
    dis_max_1 = 0.0
    y_max_2 = cent[1]
    y_max_3 = cent[1]
    top = 1000000
    for i in range(len(contours)):
        dis = math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(contours[i], cent)]))
        if contours[i][1] < cent[1]:
            if contours[i][1]<top:
                top = contours[i][1]
                x_max_1 = contours[i][0]
                y_max_1 = contours[i][1]
        else:
            if contours[i][0] < cent[0]:
                if contours[i][1] > y_max_2:
                    x_max_2 = contours[i][0]
                    y_max_2 = contours[i][1]
            if contours[i][0] > cent[0]:
                if contours[i][1] > y_max_3:
                    x_max_3 = contours[i][0]
                    y_max_3 = contours[i][1]
    top_point = (x_max_1, y_max_1)
    left_point = (x_max_2, y_max_2)
    right_point = (x_max_3, y_max_3)
    # print(cent)
    # print(right_point)
    return left_point, right_point