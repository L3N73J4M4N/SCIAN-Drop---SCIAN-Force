import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import colorsys
from matplotlib.colors import LinearSegmentedColormap


def linear_gradient(ini, end, n):
    colors = [ini]
    for t in range(1, n):
        v = [ini[j] + (t / (n - 1)) * (end[j] - ini[j]) for j in range(3)]
        colors.append(v)
    return np.array(colors)


def get_color_gradient(c1, c2, c3, n):
    first = linear_gradient(c1, c2, n // 2)
    second = linear_gradient(c2, c3, n // 2)
    return np.concatenate([first, second]) if n % 2 == 0 else np.concatenate([first, [c2], second])


def get_scale(curves):
    m, n = np.shape(curves)
    fc = np.ndarray.flatten(curves)
    fcs = np.argsort(fc)
    count = 0
    for i in range(0, len(fc)):
        if fc[i] >= 0:
            count += 1
    cmap_1 = linear_gradient([0.00, 0.40, 1.00],
                             [1.00, 1.00, 1.00],
                             n * m - count)
    cmap_2 = linear_gradient([1.00, 1.00, 1.00],
                             [1.00, 0.15, 0.00],
                             count)
    cmap = np.concatenate([cmap_1, cmap_2])
    result = np.zeros((m * n, 3))
    for i in range(0, m * n):
        result[fcs[i]] = cmap[i]
    result = np.reshape(result, (m, n, 3))
    return result


def get_subscale(scale, k):
    subscale = scale
    values = k
    mini = np.argmin(k)
    maxi = np.argmax(k)
    c1, c2 = subscale[mini], subscale[maxi]
    if values[mini] < 0 < values[maxi]:
        cmap = LinearSegmentedColormap.from_list('temp', [c1, [1, 1, 1], c2])
    else:
        cmap = LinearSegmentedColormap.from_list('temp', [c1, c2])
    return cmap


def sort_colors(colors):
    hex_list = []
    for i in range(0, len(colors)):
        r, g, b = colors[i]
        color = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        if color not in hex_list:
            hex_list.append(color)
    rgb_list = []
    for j in range(0, len(hex_list)):
        h = hex_list[j]
        rgb_list.append([int(h[i:i + 2], 16) for i in range(1, 6, 2)])
    rgb_list = np.array(rgb_list, dtype=float)
    rgb_list /= 255
    rgb_sort = sorted(rgb_list, key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
    return rgb_sort


# c = np.array([(0.00, 0.40, 1.00),
#               (1.00, 1.00, 1.00),
#               [1.00, 0.15, 0.00]])
# num_points = 11
# values = [0, 4, 3, 7, -1]
#
# gradient = get_color_gradient(c[0], c[1], c[2], num_points)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(np.linspace(0, 100, num_points),
#            values, c=values, cmap=cl.ListedColormap(gradient))
# ax.set_facecolor((0.5, 0.5, 0.5))
# plt.show()