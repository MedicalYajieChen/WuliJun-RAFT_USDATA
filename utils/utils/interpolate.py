from scipy import interpolate


def bilinear(x, y, z, x_a, y_a):
    """
    :param x: [1, 2]
    :param y: [1, 2]
    :param z: [[左上角, 右上角], [左下角, 右下角]] is [[Q11, Q21], [Q12, Q22]]
    :return:
    """
    f = interpolate.interp2d(x, y, z, kind='linear')
    out = f(x_a, y_a)
    out = out[0]
    return out


def cubic(x, y, z, x_a, y_a):
    """
    :param x: x坐标序列
    :param y: y坐标序列
    :param z: 值序列
    :param x_a: 插值x坐标
    :param y_a: 插值y坐标
    :return: 插值效果
    """
    f = interpolate.interp2d(x, y, z, kind='cubic', copy=True, bounds_error=False, fill_value=None)
    out = f(x_a, y_a)
    return out

