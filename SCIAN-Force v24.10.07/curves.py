import cv2 as cv
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import io
from skimage.util import img_as_ubyte


# retorna una parametrización del contorno de la gota
# a partir de una imagen 'img', con un grado de splines 'k',
# un suavizado 's'. El retorno corresponde a la derivada
# 'n'-ésima con una cantidad de datos de 'lenght'.
def get_splines(img, k=3, n=0, s=-1, length=-1):
    try:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except cv.error:
        pass
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    data = [c[0] for c in contours[1]] + [contours[1][0][0]]
    data = np.array(data)
    s = len(data) / 10 if s == -1 else s
    [tck, _] = splprep([data[:, 0], data[:, 1]], k=k, s=s, per=1)
    u = np.linspace(0, 1, 10000) if length == -1 else np.linspace(0, 1, length)
    spl = splev(u, tck, der=n)
    return spl[0], spl[1], u


# obtiene la curvatura principal a partir de las
# primeras y segundas derivadas del contorno de una imagen 'img'
def get_curvatures(img):
    dx, dy, _ = get_splines(img, n=1)
    d2x, d2y, _ = get_splines(img, n=2)
    k = (dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** (3 / 2)
    return k

# z = 45
# im = io.imread(r"C:\Users\matia\OneDrive - Universidad de Chile\Escritorio\test.tif")
# im = np.invert(img_as_ubyte(im[z]))
# colors = np.array([(0.00, 0.40, 1.00),   # blanco
#                    (1.00, 1.00, 1.00),
#                    [1.00, 0.15, 0.00]])
# cmap = LinearSegmentedColormap.from_list('temp', colors)
#
# x, y, u = get_splines(im)
# dx, dy, _ = get_splines(im, n=1)
# d2x, d2y, _ = get_splines(im, n=2)
# k = (dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** (3 / 2)
# for j in range(0, len(im)):
#     for i in range(0, len(im[0])):
#         if im[j, i] == 255:
#             im[j, i] = 128
# im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
# plt.imshow(im, cmap='gray')
# plt.scatter(x, y, c=k, marker='.', linewidths=0, cmap=cmap)
# plt.colorbar()
# plt.title('z = ' + str(z))
# plt.show()
