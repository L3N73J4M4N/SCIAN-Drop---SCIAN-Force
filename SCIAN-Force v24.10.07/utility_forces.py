import numpy as np
from tifffile import imwrite
import os
import plotly.graph_objects as go


# módulo con funciones de utilidad para poder definir la clase de fuerzas
############################################################################
# convierte un string con números (sin espacios al final y al inicio),
# separados por espacios, en números.
# ejemplo: str2num('1 2 3') -> [1.0, 2.0, 3.0]
def str2num(string, type_num='float'):
    string_list = []
    num_list = []
    string += ' '
    k = 0
    for i in range(0, len(string) - 1):
        if string[i + 1] == ' ' and string[i] != ' ':
            string_list.append(string[k: i + 1])
        if string[i] == ' ' and string[i + 1] != ' ':
            k = i + 1
    for s in string_list:
        if type_num == 'float':
            num_list.append(float(s))
        if type_num == 'int':
            num_list.append(int(s))
    return num_list


# en caso de tener un vector con elementos no deseados, los elimina.
# ejemplo: anti_nan([1, 2, np.nan, 3]) -> [1, 2, 3]
def anti_nan(vector):
    new_vector = []
    for p in vector:
        if p == np.nan or p == np.inf or p == -np.inf:
            continue
        new_vector.append(p)
    return new_vector


# cotangente, la inversa de la tangente
def cot(angle):
    return 1 / np.tan(angle)


############################################################################
# retorna la suma de las áreas baricéntricas de los triángulos
# formados por el punto i. Obtienes los lados de cada
# triángulo al cual pertenece i y se aplica la fórmula de Heron
def cell_area(i, vertices, triangles, three=True):
    area = 0
    a = vertices[i]
    tri = triangles[i]
    for j in range(0, len(tri)):
        b, c = tri[j]
        b, c = vertices[b], vertices[c]
        ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2
        area += (s * (s - ab) * (s - bc) * (s - ca)) ** (1 / 2)  # fórmula de Heron
    return (1 / 3) * area if three else area


def mixed_area(i, vertices, faces, triangles, angles):
    area = 0
    tri = triangles[i]
    for t in range(0, len(tri)):
        face = faces[tri[t]]
        k = np.where(face == i)[0]
        angles_tri = angles[tri[t]]
        if any(np.abs(angles_tri)) > np.pi / 2:  # no es obtuso
            p = vertices[i]
            k1, k2 = np.where(face != i)[0]
            print(k1, k2)
            q, r = vertices[face[k1]], vertices[face[k2]]
            # area de Voronoi
            area += (1 / 8) * (np.linalg.norm(p - r) ** 2 * cot(angles_tri[k1])
                               + np.linalg.norm(p - q) ** 2 * cot(angles_tri[k2]))
        else:  # es obtuso
            a, b, c = face
            a, b, c = vertices[a], vertices[b], vertices[c]
            ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
            s = (ab + bc + ca) / 2
            area_tri = (s * (s - ab) * (s - bc) * (s - ca)) ** (1 / 2)  # fórmula de Heron
            if angles_tri[k] > np.pi / 2:
                area += area_tri / 2
            else:
                area += area_tri / 4
    return area


# dados dos puntos, i y j, los cuales son vecinos y forman
# un borde. Calcula los ángulos opuestos en cada triángulo
def angles_triangles(i, j, neighbors, angles, faces):
    try:
        tri1, tri2 = neighbors[i, j]
    except KeyError:  # el orden en que se definen los bordes es desconocido
        tri1, tri2 = neighbors[j, i]
    angles1, angles2 = angles[tri1], angles[tri2]
    k1 = np.where(np.logical_and(faces[tri1] != i, faces[tri1] != j))
    k2 = np.where(np.logical_and(faces[tri2] != i, faces[tri2] != j))
    return angles1[k1], angles2[k2]


# retorna un arreglo donde cada índice indica una cara
# junto a sus 3 ángulos, en orden según el orden
# en que están sus vertices ordenados de menor a mayor
def get_angles(vertices, faces):
    angles = np.zeros((len(faces), 3))
    for i in range(0, len(faces)):
        a, b, c = faces[i]
        p, q, r = vertices[a], vertices[b], vertices[c]
        pq, qp = q - p, p - q
        pr, rp = r - p, p - r
        qr, rq = r - q, q - r
        angle1 = np.dot(pq, pr) / (np.linalg.norm(pq) * np.linalg.norm(pr))
        angle2 = np.dot(qp, qr) / (np.linalg.norm(qp) * np.linalg.norm(qr))
        angle3 = np.dot(rq, rp) / (np.linalg.norm(rq) * np.linalg.norm(rp))
        angle1, angle2, angle3 = np.arccos(angle1), np.arccos(angle2), np.arccos(angle3)
        #             a -> angle1
        #            / \
        # angle3 <- c _ b -> angle2
        angles[i] = [angle1, angle2, angle3]
    return angles


############################################################################
# se calculará el área y el ángulo a la vez para el caso gaussiano.
def angle_area(i, vertices, triangles):
    point = vertices[i]
    triangle = triangles[i]
    angle, area = 0, 0
    for j in range(0, len(triangle)):
        a, b = triangle[j]
        a, b = vertices[a], vertices[b]
        v1, v2 = a - point, b - point
        angle += np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        e1, e2, e3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(a - b)
        s = (e1 + e2 + e3) / 2
        area += (s * (s - e1) * (s - e2) * (s - e3)) ** (1 / 2)  # fórmula de Heron
    return angle, area


############################################################################
# descomposición en A = LDL^T, conocida como la descomposición de Cholesky.
# retorna bool de "es posible realizar la descomposición" y modifica diagonal D
def ldltdc(A, diag):
    N = len(A)
    # si no es matriz, nada que hacer
    if N < 1:
        return False
    # caso especial, qué sucede si N <= 3
    elif N <= 3:
        d0 = A[0, 0]
        diag[0] = 1 / d0
        if N == 1:
            return d0 != 0
        A[1, 0] = A[0, 1]
        l10 = diag[0] * A[1, 0]
        d1 = A[1, 1] - l10 * A[1, 0]
        diag[1] = 1 / d1
        if N == 2:
            return d0 != 0 and d1 != 0
        d2 = A[2, 2] - diag[0] * A[2, 0] ** 2 - diag[1] * A[2, 1] ** 2
        diag[2] = 1 / d2
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2] - l10 * A[2, 0]
        return d0 != 0 and d1 != 0 and d2 != 0
    v = np.zeros(N - 1)
    for i in range(0, N):
        for k in range(0, i):
            v[k] = A[i, k] * diag[k]
        for j in range(i, N):
            sum = A[i, j]
            for k in range(0, i):
                sum -= v[k] * A[j, k]
            if i == j:
                if sum == 0:
                    return False
                diag[i] = 1 / sum
            else:
                A[j, i] = sum
    return True


# resuelve el sistema de ecuaciones como una sustitución hacia
# adelante, seguida de una hacia atrás. Vale decir, dado el sistema
# Ax = B y A = LDLt, resuelve Ly = b -> Ltx = y. Si b es la salida
# tenemos que same=True y Ly = x ->  Ltx = y. Retorna x.
def ldltsl(A, rdiag, b, same=False):
    x = b if same is True else np.zeros(len(A))
    for i in range(0, len(A)):
        sum = b[i]
        for k in range(0, i):
            sum -= A[i, k] * x[k]
        x[i] = sum * rdiag[i]
    for i in range(len(A) - 1, -1, -1):
        sum = 0
        for k in range(i + 1, len(A)):
            sum += A[k, i] * x[k]
        x[i] -= sum * rdiag[i]
    return x


# rotador de coordenadas para ponerlas en su
# plano correspondiente, perpendicular.
def rot_coord(u_old, v_old, n_new):
    n_old = np.cross(u_old, v_old)
    n_dot = np.dot(n_old, n_new)
    u_new, v_new = u_old, v_old
    if n_dot <= -1:
        return -u_new, -v_new
    perp_old = n_new - n_dot * n_old
    d_perp = 1 / (1 + n_dot) * (n_old + n_new)
    u_new -= d_perp * np.dot(u_new, perp_old)
    v_new -= d_perp * np.dot(v_new, perp_old)
    return u_new, v_new


# proyección desde u_old y v_old hasta el u_new y v_new
def proj_curves(u_old, v_old, ku_old, kuv_old, kv_old, u_new, v_new):
    ru_new, rv_new = rot_coord(u_new, v_new, np.cross(u_old, v_old))
    u1, v1 = np.dot(ru_new, u_old), np.dot(ru_new, v_old)
    u2, v2 = np.dot(rv_new, u_old), np.dot(rv_new, v_old)
    # coeficientes de curvatura
    ku_new = ku_old * u1 * u1 + kuv_old * (2 * u1 * v1) + kv_old * v1 * v1
    kuv_new = ku_old * u1 * u2 + kuv_old * (u1 * v2 + u2 * v1) + kv_old * v1 * v2
    kv_new = ku_old * u2 * u2 + kuv_old * (2 * u2 * v2) + kv_old * v2 * v2
    return ku_new, kuv_new, kv_new


# dado el tensor de curvaturas, encuentra k1 y k2. A través
# de la diagonalización de sus componentes. También,
# retorna las direcciones principales p1 y p2.
def diagonalize_curves(u_old, v_old, ku, kuv, kv, n_new):
    ru_old, rv_old = rot_coord(u_old, v_old, n_new)
    c, s, tt = 1, 0, 0
    if kuv != 0:
        h = 0.5 * (kv - ku) / kuv
        tt = 1 / (h - np.sqrt(1 + h ** 2)) if h < 0 else 1 / (h + np.sqrt(1 + h ** 2))
        c = 1 / np.sqrt(1 + tt ** 2)
        s = tt * c
    k1 = ku - tt * kuv
    k2 = kv + tt * kuv
    if abs(k1) >= abs(k2):
        p1 = c * ru_old - s * rv_old
    else:
        k1, k2 = k2, k1
        p1 = s * ru_old + c * rv_old
    p2 = np.cross(n_new, p1)
    return k1, k2, p1, p2


# proyección para dcurves
def proj_dcurves(old_u, old_v, old_dcurv, new_u, new_v):
    r_new_u, r_new_v = rot_coord(new_u, new_v, np.cross(old_u, old_v))
    u1 = np.dot(r_new_u, old_u)
    v1 = np.dot(r_new_u, old_v)
    u2 = np.dot(r_new_v, old_u)
    v2 = np.dot(r_new_v, old_v)
    new_dcurv = np.zeros(4)
    new_dcurv[0] = old_dcurv[0] * u1 ** 3 + old_dcurv[1] * 3 * u1 ** 2 * v1 + \
        old_dcurv[2] * 3 * v1 ** 2 * u1 + old_dcurv[3] * v1 ** 3
    new_dcurv[1] = old_dcurv[0] * u1 ** 2 * u2 + old_dcurv[1] * (u1 ** 2 * v2 + 2.0 * u2 * u1 * v1) + \
        old_dcurv[2] * (u2 * v1 ** 2 + 2 * u1 * v1 * v2) + old_dcurv[3] * v1 ** 2 * v2
    new_dcurv[2] = old_dcurv[0] * u1 * u2 ** 2 + old_dcurv[1] * (u2 ** 2 * v1 + 2.0 * u1 * u2 * v2) + \
        old_dcurv[2] * (u1 * v2 ** 2 + 2.0 * u2 * v2 * v1) + old_dcurv[3] * v1 * v2 ** 2
    new_dcurv[3] = old_dcurv[0] * u2 ** 3 + old_dcurv[1] * 3 * u2 ** 2 * v2 + \
        old_dcurv[2] * 3 * u2 * v2 ** 2 + old_dcurv[3] * v2 ** 3
    return new_dcurv


def ldl_solve(a, b):
    l = np.linalg.cholesky(a)
    # a = l * d * lT
    y = np.linalg.lstsq(l, b)[0]
    x = np.linalg.lstsq(l.T, y)[0]
    return x


def get_point_area(vertices, faces):
    point_areas = np.zeros(len(vertices))
    corner_areas = np.zeros((len(faces), 3))
    for i in range(0, len(faces)):
        edges = np.array([vertices[faces[i, 2]] - vertices[faces[i, 1]],
                          vertices[faces[i, 0]] - vertices[faces[i, 2]],
                          vertices[faces[i, 1]] - vertices[faces[i, 0]]])
        # cálculo de áreas por punto a través de pesos
        area = 0.5 * np.linalg.norm(np.cross(edges[0], edges[1]))
        l2 = np.array([np.linalg.norm(edges[0]) ** 2,
                       np.linalg.norm(edges[1]) ** 2,
                       np.linalg.norm(edges[2]) ** 2])
        bcw = np.array([l2[0] * (l2[1] + l2[2] - l2[0]),
                        l2[1] * (l2[2] + l2[0] - l2[1]),
                        l2[2] * (l2[0] + l2[1] - l2[2])])
        if bcw[0] <= 0:
            corner_areas[i, 1] = -0.25 * l2[2] * area / np.dot(edges[0], edges[2])
            corner_areas[i, 2] = -0.25 * l2[1] * area / np.dot(edges[0], edges[1])
            corner_areas[i, 0] = area - corner_areas[i, 1] - corner_areas[i, 2]
        elif bcw[1] <= 0:
            corner_areas[i, 2] = -0.25 * l2[0] * area / np.dot(edges[1], edges[0])
            corner_areas[i, 0] = -0.25 * l2[2] * area / np.dot(edges[1], edges[2])
            corner_areas[i, 1] = area - corner_areas[i, 2] - corner_areas[i, 0]
        elif bcw[2] <= 0:
            corner_areas[i, 0] = -0.25 * l2[1] * area / np.dot(edges[2], edges[1])
            corner_areas[i, 1] = -0.25 * l2[0] * area / np.dot(edges[2], edges[0])
            corner_areas[i, 2] = area - corner_areas[i, 0] - corner_areas[i, 1]
        else:
            scale = 0.5 * area / (bcw[0] + bcw[1] + bcw[2])
            for j in range(0, 3):
                corner_areas[i, j] = scale * (bcw[(j + 1) % 3] + bcw[(j - 1) % 3])
        point_areas[faces[i, 0]] += corner_areas[i, 0]
        point_areas[faces[i, 1]] += corner_areas[i, 1]
        point_areas[faces[i, 2]] += corner_areas[i, 2]
    return point_areas, corner_areas


############################################################################
# retorna la escala de colores según la función function.
# En ella, las tonalidades anaranjadas indican positivo,
# las azuladas negativo, y el blanco cero.
def colormap(function):
    fun_color = function / abs(np.max(function))
    more0, less0, cero = 0, 0, 0
    for i in range(0, len(fun_color)):
        if fun_color[i] > 0:
            more0 += 1
        if fun_color[i] < 0:
            less0 += 1
        if fun_color[i] == 0:
            cero += 1
    if less0 + cero == 0:  # h > 0
        cmap = [(1.00, 1.00, 0.00),  # amarillo
                (1.00, 0.00, 0.00)]  # rojo
    elif more0 + cero == 0:  # h < 0
        cmap = [(0.10, 0.43, 0.86),  # celeste
                (0.63, 0.78, 1.00)]  # azul
    elif less0 == 0 and cero != 0:  # h >= 0
        cmap = [(1.00, 1.00, 1.00),  # blanco
                (1.00, 1.00, 0.00),  # amarillo
                (1.00, 0.00, 0.00)]  # rojo
    elif more0 == 0 and cero != 0:  # h <= 0
        cmap = [(0.10, 0.43, 0.86),  # azul
                (0.63, 0.78, 1.00),  # celeste
                (1.00, 1.00, 1.00)]  # blanco
    elif less0 < more0 / 2:  # hay muchos más h > 0 que h < 0
        cmap = [(0.63, 0.78, 1.00),  # celeste
                (1.00, 1.00, 1.00),  # blanco
                (1.00, 1.00, 0.00),  # amarillo
                (1.00, 0.00, 0.00)]  # rojo
    elif more0 < less0 / 2:  # hay muchos más h < 0 que h > 0
        cmap = [(0.10, 0.43, 0.86),  # azul
                (0.63, 0.78, 1.00),  # celeste
                (1.00, 1.00, 1.00),  # blanco
                (1.00, 1.00, 0.00)]  # amarillo
    else:
        cmap = [(0.10, 0.43, 0.86),  # azul
                (0.63, 0.78, 1.00),  # celeste
                (1.00, 1.00, 1.00),  # blanco
                (1.00, 1.00, 0.00),  # amarillo
                (1.00, 0.00, 0.00)]  # rojo
    return cmap


def cmap_go(curves, max_value=None, min_value=None):
    curves_norm = curves / np.max(np.abs(curves))
    index = np.zeros((5, len(curves)))
    for i in range(0, len(curves)):
        index[0, i] = abs(curves_norm[i] + 1)
        index[1, i] = abs(curves_norm[i] + 0.5)
        index[2, i] = abs(curves_norm[i])
        index[3, i] = abs(curves_norm[i] - 0.5)
        index[4, i] = abs(curves_norm[i] - 1)
    func = [curves[np.argmin(index[j])] for j in range(0, 5)]
    func_sort = sorted(func)
    if max_value is not None:
        func_sort[-1] = max_value
    if min_value is not None:
        func_sort[0] = min_value
    values = [(v - func_sort[0]) / (func_sort[-1] - func_sort[0]) for v in func_sort]
    for i in range(0, len(values)):
        if values[i] > 1.0:
            values[i] = 1.0
    return values


############################################################################
# crea un stack de un cilindro

def make_tube(path: str, r: int, h: int, dim=(128, 128, 64), one_tiff=True, name=False):
    x_dim, y_dim, z_dim = dim
    if r > x_dim:
        print('radio mayor a las dimensiones')
        x_dim = r
    if r > y_dim:
        y_dim = r
        print('radio mayor a las dimensiones')
    if h > z_dim:
        z_dim = h
        print('altura mayor a las dimensiones')
    zl, zh = z_dim // 2 - h / 2, z_dim // 2 + h / 2 - 1
    tiff = np.zeros((z_dim, y_dim, x_dim))
    for k in range(0, z_dim):
        if k > zh or k < zl:
            continue
        for i in range(0, x_dim):
            for j in range(0, y_dim):
                if (i - x_dim // 2) ** 2 + (j - y_dim // 2) ** 2 <= r ** 2:
                    tiff[k, j, i] = 255
    if not os.path.isdir(path):
        os.makedirs(path)
    path += r'\tube_' + f'r{r}_h{h}_x{x_dim}_y{y_dim}_z{z_dim}' if name is False else path
    if one_tiff is True:
        path += '.tif'
        imwrite(path, tiff.astype(np.uint8))
    else:
        for k in range(0, z_dim):
            sub_path = path + f'_t000_ch_000_z00{k}.tif'
            imwrite(sub_path, tiff[k].astype(np.uint8))
    return tiff


def make_sphere(r: int,
                dim=(-1, -1, -1),
                voxel_size=(1, 1, 1),
                c=None,
                one_tiff=True,
                path=r'C:\\',
                save=False):
    if 2 * r > dim[0] or 2 * r > dim[1] or 2 * r > dim[2]:
        if 2 * r + 1 % 2 == 1:
            dim = ((2 * r + 2) // voxel_size[0],
                   (2 * r + 2) // voxel_size[1],
                   (2 * r + 2) // voxel_size[2])
        else:
            dim = ((2 * r + 1) // voxel_size[0],
                   (2 * r + 1) // voxel_size[1],
                   (2 * r + 1) // voxel_size[2])
        dim = (int(dim[0]), int(dim[1]), int(dim[2]))
    tiff = np.zeros(dim)
    if c is None:
        c = (voxel_size[0] * dim[0] / 2,
             voxel_size[1] * dim[1] / 2,
             voxel_size[2] * dim[2] / 2)
    for z in range(dim[0]):
        for y in range(dim[1]):
            for x in range(dim[2]):
                if (x * voxel_size[2] - c[2]) ** 2 + \
                        (y * voxel_size[1] - c[1]) ** 2 + \
                        (z * voxel_size[0] - c[0]) ** 2 <= r ** 2:
                    tiff[z, y, x] = 255
    if save is True:
        if not os.path.isdir(path):
            os.makedirs(path)
        path += r'\sphere_' + f'r{r}'
        if one_tiff is True:
            path += '.tif'
            imwrite(path, tiff.astype(np.uint8))
        else:
            for k in range(0, dim[0]):
                sub_path = path + f'_t000_ch_000_z00{k}.tif'
                imwrite(sub_path, tiff[k].astype(np.uint8))
    return tiff


def save_a_lot_of_spheres(path, name, x, one_tiff=False):
    max_sphere = make_sphere(x[-1], voxel_size=(0.5, 0.163, 0.163))
    z, m, n = np.shape(max_sphere)
    spheres = np.zeros((z, m, n * len(x)))
    v = 0
    for rad in x:
        if rad == x[-1]:
            sphere = max_sphere
        else:
            sphere = make_sphere(rad, dim=(z, m, n), voxel_size=(0.5, 0.163, 0.163))
        for k in range(z):
            for j in range(m):
                for i in range(n):
                    spheres[k, j, i + v] = sphere[k, j, i]
        v += n
    if not os.path.isdir(path):
        os.makedirs(path)
    path += r'\\' + name
    if one_tiff is True:
        path += '.tif'
        imwrite(path, spheres.astype(np.uint8))
    else:
        for k in range(0, z):
            sub_path = path + f'_ch_000_z00{k}.tif'
            imwrite(sub_path, spheres[k].astype(np.uint8))
