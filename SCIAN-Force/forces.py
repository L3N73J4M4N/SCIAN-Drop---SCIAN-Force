import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
import pymeshfix
import trimesh

from utility_forces import str2num, anti_nan, \
    cot, cell_area, angles_triangles, angle_area, ldltsl, \
    ldltdc, proj_curves, diagonalize_curves, colormap, cmap_go, \
    get_angles, mixed_area


# clase para calcular fuerzas 3D (en un futuro). De momento, es capaz de obtener
# normales, vecinos, áreas y curvaturas. Para este último, existen distintos
# métodos y algoritmos, los cuales se pueden ver en la bibliografía. Además, permite
# leer archivos .off, graficarlos y colorear los objetos en función de sus
# propiedades.
class Forces3D:
    def __init__(self, path, interval='1x1x1', units='px'):
        self.path = path  # ruta del archivo
        self.colorFun = None  # array que le da color a cada cara
        self.normalsFaces = None  # array 3D con las normales unitarias de cada cara
        self.normalsPoints = None  # array 3D con las normales unitarias promedio en cada punto
        self.neighborsFaces = None  # dict - neighborsFaces[edge = (p, q)] -> [caras adyacentes a edge]
        self.neighborsPoints = None  # dict - neighborsFaces[point] -> [puntos conectados con point]
        self.meanCurvature = None  # array con la curvatura promedio para cada punto
        self.method = None  # str con el nombre del método utilizado para la curvatura
        self.trianglePoints = None  # dict - trianglePoints[v] -> [[otros v en face_i], ...]
        self.areas = None  # áreas de los triángulos / 3
        self.gaussCurvature = None  # array con la curvatura gaussiana para cada punto
        self.principals = None  # array con las curvaturas principales, k1 y k2
        self.name = os.path.split(path)[1]  # nombre del archivo
        self.type = 'obj' if path[-1] == 'j' else 'off'  # tipo de archivo
        # array con malla de la superficie, vertices y caras
        vertices, faces = self.off2mesh() if self.type == 'off' else self.obj2mesh()
        interval = interval.split('x')
        for i in range(0, 3):
            vertices[:, i] *= float(interval[i])
        self.mesh = vertices, faces
        self.units = units
        self.trimesh = None  # malla del método Trimesh
        self.volume = None  # volumen de la malla
        self.radius = None  # radio original de la gota
        self.stress_bool = False  # estrés normal, verifica si se ha calculado o no

    # convierte el archivo .off de la ruta en una malla triangular
    # de la superficie que definen los puntos con las caras
    def off2mesh(self):
        with open(self.path, 'r') as path:
            lines = path.readlines()
            if lines[0].strip() != 'OFF':
                raise ValueError('The file have not OFF format.')
            n, m, _ = list(map(int, lines[1].split()))
            vertices = np.zeros((n, 3))
            faces = np.zeros((m, 3), int)
            for i in range(2, n + 2):
                line = str2num(lines[i].strip())
                vertices[i - 2] = [float(line[0]), float(line[1]), float(line[2])]
            for i in range(n + 2, n + m + 2):
                line = str2num(lines[i].strip(), 'int')
                faces[i - n - 2] = [int(line[1]), int(line[2]), int(line[3])]
            return vertices, faces

    # convierte el archivo .obj de la ruta en una malla triangular
    # de la superficie que definen los puntos con las caras
    def obj2mesh(self):
        vertices, faces = [], []
        with open(self.path, 'r') as file:
            for lines in file:
                line = lines.split()
                if not line:
                    continue
                elif line[0] == 'v':
                    vertices.append([float(line[1]), float(line[2]), float(line[3])])
                elif line[0] == 'f':
                    faces.append([int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1])
        return np.array(vertices), np.array(faces)

    # grafica la malla superficial obtenida
    # - use_fun (bool) -> utilizar una función para los colores de la cara
    # - fun (array) -> arreglo del mismo largo que las caras, indica el color de cada una de ellas
    # - cmap (str) -> mapa de colores con lo que se va a pintar, 'paper' utiliza la escala
    #   del paper "dropletsOriginal"; "bw" escala blanco y negro; otro valor utiliza el mismo cmap
    #   los nombres posibles son similares o iguales a Matplotlib
    # - title (str) -> título de la gráfica, por predeterminado es la ruta
    # - edges (bool) -> mostrar los bordes de la malla
    # - view_method (bool) -> mostrar método en el título
    def plot(self, fun=None,
             use_fun=False,
             cmap='default',
             title='path',
             view_method=False,
             edges=False):
        fun = self.colorFun if fun is None else fun
        title = self.name if title == 'path' else title
        title += ' - ' + self.method if view_method is True else ''
        vertices, faces = self.mesh
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        if type(cmap) is str:
            if cmap == 'default':
                cmap = colormap(self.colorFun) if self.colorFun is not None else 'paper'
            if cmap == 'paper':
                cmap = [(0.10, 0.43, 0.86),
                        (0.63, 0.78, 1.00),
                        (1.00, 1.00, 1.00),
                        (1.00, 1.00, 0.00),
                        (1.00, 0.00, 0.00)]
            if cmap == 'bw':
                cmap = [(1, 1, 1),
                        (0, 0, 0)]
            if cmap == 'red':
                cmap = [(1, 0, 0),
                        (1, 0, 0)]
        else:
            cmap = cmap
        if use_fun is False:
            fig = ff.create_trisurf(x, y, z,
                                    simplices=faces,
                                    colormap=cmap,
                                    title=title,
                                    plot_edges=edges
                                    )
        else:
            fig = ff.create_trisurf(x, y, z,
                                    simplices=faces,
                                    colormap=cmap,
                                    color_func=fun,
                                    title=title,
                                    plot_edges=edges
                                    )
        fig.show()

    def plot_go(self, max_value=None, min_value=None, edges=False):
        title_colorbar = 'Mean<br>Curvatures' if self.stress_bool is False else 'Normal<br>Stress<br>[nN/μm²]'
        title = self.name + ' - ' + self.method
        vertices, faces = self.mesh
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        color = cmap_go(self.meanCurvature, max_value, min_value)
        fig = go.Figure(layout={'title': title})
        if min_value is None and max_value is None:
            fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                           intensity=self.meanCurvature,
                           colorbar={'title': title_colorbar},
                           colorscale=[[color[0], '#052861'],
                                       [color[1], '#009dff'],
                                       [color[2], '#ffffff'],
                                       [color[3], '#ffff00'],
                                       [color[4], '#ff0000']])
        elif min_value is None and max_value is not None:
            fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                           intensity=self.meanCurvature,
                           colorbar={'title': title_colorbar},
                           cmin=np.min(self.meanCurvature), cmax=max_value,
                           colorscale=[[color[0], '#052861'],
                                       [color[1], '#009dff'],
                                       [color[2], '#ffffff'],
                                       [color[3], '#ffff00'],
                                       [color[4], '#ff0000']])
        elif min_value is not None and max_value is None:
            fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                           intensity=self.meanCurvature,
                           colorbar={'title': title_colorbar},
                           cmin=min_value, cmax=np.max(self.meanCurvature),
                           colorscale=[[color[0], '#052861'],
                                       [color[1], '#009dff'],
                                       [color[2], '#ffffff'],
                                       [color[3], '#ffff00'],
                                       [color[4], '#ff0000']])
        elif min_value is not None and max_value is not None:
            fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                           intensity=self.meanCurvature,
                           colorbar={'title': title_colorbar},
                           cmin=min_value, cmax=max_value,
                           colorscale=[[color[0], '#052861'],
                                       [color[1], '#009dff'],
                                       [color[2], '#ffffff'],
                                       [color[3], '#ffff00'],
                                       [color[4], '#ff0000']])
        if edges is True:
            tri_points = vertices[faces]
            xe, ye, ze = [], [], []
            for t in tri_points:
                xe.extend([t[k % 3][0] for k in range(4)] + [None])
                ye.extend([t[k % 3][1] for k in range(4)] + [None])
                ze.extend([t[k % 3][2] for k in range(4)] + [None])
            fig.add_scatter3d(x=xe, y=ye, z=ze, mode='lines', name='',
                              line=dict(color='rgb(70,70,70)', width=1))
        fig.show()

    def normals_per_face(self):
        vertices, faces = self.mesh
        normals = np.zeros((len(faces), 3))
        for i in range(0, len(faces)):
            a, b, c = faces[i]
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            e1 = va - vb
            e2 = va - vc
            n = np.cross(e1, e2)
            n /= np.linalg.norm(n)
            normals[i] = n
        self.normalsFaces = normals
        return normals

    # busca los vecinos de cada cara. Para ello,
    # se agruparán los bordes, asumiendo que cada borde tiene
    # solamente dos caras adyacentes, vale decir
    # es una malla "regular". Retorna un dict con lo anterior
    def search_neighbors(self):
        neighbors = {}
        _, faces = self.mesh
        for i in range(0, len(faces)):
            a, b, c = faces[i]
            # primer borde
            e1 = [(a, b), (b, a)]
            r1, r2 = e1
            if r1 not in neighbors and r2 not in neighbors:
                neighbors[r1] = [i]
            elif r2 in neighbors and r1 not in neighbors:
                neighbors[r2] += [i]
            elif r2 not in neighbors and r1 in neighbors:
                neighbors[r1] += [i]

            # segundo borde
            e2 = [(a, c), (c, a)]
            r1, r2 = e2
            if r1 not in neighbors and r2 not in neighbors:
                neighbors[r1] = [i]
            elif r2 in neighbors and r1 not in neighbors:
                neighbors[r2] += [i]
            elif r2 not in neighbors and r1 in neighbors:
                neighbors[r1] += [i]

            # tercer borde
            e3 = [(b, c), (c, b)]
            r1, r2 = e3
            if r1 not in neighbors and r2 not in neighbors:
                neighbors[r1] = [i]
            elif r2 in neighbors and r1 not in neighbors:
                neighbors[r2] += [i]
            elif r2 not in neighbors and r1 in neighbors:
                neighbors[r1] += [i]
        self.neighborsFaces = neighbors
        return neighbors

    # retorna los puntos conectados al vértice i
    # es un diccionario, pues no todos tienen la misma
    # cantidad de puntos conectados. También, retorna los
    # vértices conectados agrupados en triángulos.
    def neighbors_points(self):
        _, faces = self.mesh
        connected = {}
        triangles = {}
        for i in range(0, len(faces)):
            a, b, c = faces[i]
            # primero, obtención de los puntos conectados
            # análisis del primer vértice
            if a not in connected:
                connected[a] = [b, c]
            else:
                if b not in connected[a]:
                    connected[a] += [b]
                if c not in connected[a]:
                    connected[a] += [c]
            # análisis del segundo vértice
            if b not in connected:
                connected[b] = [a, c]
            else:
                if a not in connected[b]:
                    connected[b] += [a]
                if c not in connected[b]:
                    connected[b] += [c]
            # análisis del tercer vértice
            if c not in connected:
                connected[c] = [a, b]
            else:
                if b not in connected[c]:
                    connected[c] += [b]
                if a not in connected[c]:
                    connected[c] += [a]
            # luego, obtención de los puntos asociados en cada triángulo
            # primero, para el vértice a
            if a not in triangles:
                triangles[a] = [i]
            else:
                triangles[a] += [i]
            # luego, para el vértice b
            if b not in triangles:
                triangles[b] = [i]
            else:
                triangles[b] += [i]
            # y por último, el vértice c
            if c not in triangles:
                triangles[c] = [i]
            else:
                triangles[c] += [i]
        self.neighborsPoints = connected
        self.trianglePoints = triangles
        return connected, triangles

    # le otorga un color a cada cara
    # dependiendo de los valores de fun. También,
    # permite colorear según la curvatura gaussiana
    # dependiendo del valor booleano de gauss
    def colormap(self, fun=None, gauss=False, normalize=False, max_value=None, type_plot='mean'):
        fun = self.meanCurvature if fun is None else fun
        fun = self.gaussCurvature if gauss is True else fun
        vertices, faces = self.mesh
        colors = np.zeros(len(faces))
        for i in range(0, len(faces)):
            a, b, c = faces[i]
            vect = anti_nan([fun[a], fun[b], fun[c]])
            # el máximo entre cada vértice
            if type_plot == 'max':
                colors[i] = max(vect)
            # el mínimo entre cada vértice
            if type_plot == 'min':
                colors[i] = min(vect)
            # el promedio de los vértices
            if type_plot == 'mean':
                colors[i] = np.mean(vect)
            # solo las caras con todos sus vertices no negativos
            if type_plot == 'one':
                if all(vect) >= 0:
                    colors[i] = 1
                if all(vect) <= 0:
                    colors[i] = -1
                if all(vect) == 0:
                    colors[i] = 0
        if normalize is True:
            max_value = np.max(colors) if max_value is None else max_value
            colors /= max_value
        self.colorFun = colors
        return colors

    # arregla la malla proporcionada. Es decir, repara
    # las caras que tienen más de 3 vecinos (y otros defectos) utilizando
    # el software de Marco Attene. Para más información ver:
    # https://pymeshfix.pyvista.org/
    def fix_mesh(self):
        vertices, faces = self.mesh
        meshfix = pymeshfix.MeshFix(vertices, faces)
        meshfix.repair()
        self.mesh = meshfix.v, meshfix.f
        return self.mesh

    # forma discreta de calcular la curvatura basada
    # en el algoritmo visto en "Keenan Crane’s lecture" disponible
    # en https://youtu.be/sokeN5VxBB8 y https://youtu.be/NlU1m-OfumE. El primer
    # link es para la fórmula de la curvatura. El segundo para el ángulo.
    # Desde el minuto 3:16 y 4:20 respectivamente.
    def discrete_mean_curvature(self):
        if self.method == 'Discrete Method':
            return self.meanCurvature
        if self.neighborsFaces is None:
            self.search_neighbors()
        if self.normalsFaces is None:
            self.normals_per_face()
        if self.neighborsPoints is None:
            self.neighbors_points()
        neighbors = self.neighborsFaces
        normals = self.normalsFaces
        connected = self.neighborsPoints
        vertices, faces = self.mesh
        mean_curvature = np.zeros(len(vertices))
        for i in range(0, len(vertices)):
            points = connected[i]
            v1 = vertices[i]
            hi = 0
            for p in points:
                v2 = vertices[p]
                length = np.linalg.norm(v1 - v2)
                edge = (v1 - v2) / length
                try:
                    f1, f2 = neighbors[i, p]
                except KeyError:
                    f1, f2 = neighbors[p, i]
                n1, n2 = normals[f1], normals[f2]
                phi = np.arctan2(np.dot(edge, np.cross(n1, n2)),
                                 np.dot(n1, n2))
                hi += phi * length
            mean_curvature[i] = hi / 4
        self.meanCurvature = mean_curvature
        self.method = 'Discrete Method'
        return mean_curvature

    # otra forma de calcular la curvatura promedio, basado en el algoritmo propuesto en
    # http://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
    def laplacian_curvature(self):
        if self.method == 'Laplacian Method':
            return self.meanCurvature
        if self.neighborsFaces is None:
            self.search_neighbors()
        if self.normalsPoints is None:
            self.normals_per_point()
        if self.neighborsPoints is None:
            self.neighbors_points()
        neighbors = self.neighborsFaces
        points = self.neighborsPoints
        normals = self.normalsPoints
        triangles = self.trianglePoints
        vertices, faces = self.mesh
        angles = get_angles(vertices, faces)  # ángulos para cada cara

        # obtengamos el laplaciano para cada punto, a través
        # del método de las cotangentes, o su expresión discreta.
        areas = np.zeros(len(vertices))  # áreas
        alpha = np.zeros((len(vertices), len(vertices)))  # ángulo 1
        beta = np.zeros((len(vertices), len(vertices)))  # ángulo 2
        for i in range(0, len(vertices)):
            areas[i] = mixed_area(i, vertices, faces, triangles, angles)  # función para áreas
            connected = points[i]
            for c in connected:
                alpha[i, c], beta[i, c] = angles_triangles(i, c, neighbors, angles, faces)  # función para ángulos
        self.areas = areas
        laplace = np.zeros((len(vertices), 3))
        for i in range(0, len(vertices)):
            laplace[i] = (1 / (2 * areas[i])) \
                         * sum((cot(alpha[i, j]) + cot(beta[i, j]))
                               * (vertices[j] - vertices[i]) for j in points[i])
        # sigue que, necesitamos determinar el signo de H, para esto
        # se realizará el producto punto entre la normal en el punto
        # y su laplaciano respectivo negativo
        sign = np.zeros(len(vertices))
        for i in range(0, len(vertices)):
            dot = np.dot(normals[i], -laplace[i])
            if dot >= 0:
                sign[i] = 1
            else:
                sign[i] = -1

        # por último, la curvatura promedio
        mean_curvature = np.zeros(len(vertices))
        for i in range(0, len(vertices)):
            mean_curvature[i] = sign[i] * np.linalg.norm(laplace[i]) / 2
        self.meanCurvature = mean_curvature
        self.method = 'Laplacian Method'
        return mean_curvature

    # retorna la curvatura gaussiana según lo visto en
    # http://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
    def gaussian_curvature(self, method=False):
        if self.gaussCurvature is not None:
            if method is True:
                self.method = 'Gaussian Method'
            return self.gaussCurvature
        if self.neighborsPoints is None:
            self.neighbors_points()
        triangles = self.trianglePoints
        vertices, faces = self.mesh
        gauss = np.zeros(len(vertices))
        for i in range(0, len(vertices)):
            th, ar = angle_area(i, vertices, triangles)
            gauss[i] = (2 * np.pi - th) / ar
        self.gaussCurvature = gauss
        if method is True:
            self.method = 'Gaussian Method'
        return gauss

    # obtención de las curvaturas principales
    # a través del método descrito por Taubin, 1995
    def taubin_method(self):
        if self.normalsPoints is None:
            self.normals_per_point()
        if self.neighborsPoints is None:
            self.neighbors_points()
        if self.method == 'Taubin Method':
            return self.meanCurvature
        vertices, faces = self.mesh
        normals = self.normalsPoints
        neighbors = self.neighborsPoints
        k1, k2 = np.zeros(len(vertices)), np.zeros(len(vertices))
        for i in range(0, len(vertices)):
            m = np.zeros((3, 3))
            for j in range(0, len(neighbors[i])):
                p = vertices[i] - vertices[neighbors[i][j]]
                k = 2 * np.dot(normals[i], p)
                k /= np.linalg.norm(p) ** 2
                m += k * np.dot(np.transpose(p), p)
            eigen = np.real(np.linalg.eigvals(m))
            eigen[np.argmin(np.abs(eigen))] = 0
            k1[i], k2[i] = np.min(eigen), np.max(eigen)
        self.meanCurvature = (k1 + k2) / 2
        self.principals = k1, k2
        self.method = 'Taubin Method'
        return self.meanCurvature

    def normals_per_point(self):
        if self.normalsFaces is None:
            self.normals_per_face()
        normals = self.normalsFaces
        vertices, faces = self.mesh
        self.normalsPoints = trimesh.geometry.mean_vertex_normals(len(vertices), faces, normals)
        return self.normalsPoints

    def rusinkiewicz_curvature(self):
        if self.method == 'Rusinkiewicz Method':
            return self.meanCurvature
        if self.normalsPoints is None:
            self.normals_per_point()
        vertices, faces = self.mesh
        normals = self.normalsPoints
        pdir1 = np.zeros((len(vertices), 3))
        pdir2 = np.zeros((len(vertices), 3))
        curve1, curve2, curve12 = np.zeros(len(vertices)), np.zeros(len(vertices)), np.zeros(len(vertices))

        # coordenadas iniciales:
        for i in range(0, len(faces)):
            face = faces[i]
            pdir1[face[0]] = vertices[face[1]] - vertices[face[0]]  # borde del triángulo
            pdir1[face[1]] = vertices[face[2]] - vertices[face[1]]  # ''
            pdir1[face[2]] = vertices[face[0]] - vertices[face[2]]  # ''
        for i in range(0, len(vertices)):
            pdir1[i] = np.cross(pdir1[i], normals[i])  # p x n
            pdir1[i] /= np.linalg.norm(pdir1[i])  # normalización
            pdir2[i] = np.cross(normals[i], pdir1[i])  # n x p

        # áreas
        point_areas = np.zeros(len(vertices))
        corner_areas = np.zeros((len(faces), 3))

        # cálculo de las curvaturas en cada cara:
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

            # N-T-B por cara junto a la matriz de Weingarten w
            t = edges[0] / np.linalg.norm(edges[0])
            n = np.cross(edges[0], edges[1])
            b = np.cross(n, t)
            m, w = np.zeros(3), np.zeros((3, 3))
            for j in range(0, 3):
                u, v = np.dot(edges[j], t), np.dot(edges[j], b)
                w[0, 0] += u * u
                w[0, 1] += u * v
                w[2, 2] += v * v
                dn = normals[faces[i, (j - 1) % 3]] - normals[faces[i, (j + 1) % 3]]
                dnu = np.dot(dn, t)
                dnv = np.dot(dn, b)
                m[0] += dnu * u
                m[1] += dnu * v + dnv * u
                m[2] += dnv * v
            w[1, 1], w[1, 2] = w[0, 0] + w[2, 2], w[0, 1]
            # ldl = sci.cholesky(w)
            # y = np.linalg.solve(ldl, m)
            # m = np.linalg.solve(ldl.T, y)
            diag = np.zeros(3)
            value = ldltdc(w, diag)
            if value is False:
                continue
            m = ldltsl(w, diag, m, same=True)
            for j in range(0, 3):
                vj = faces[i, j]
                c1, c12, c2 = proj_curves(t, b, m[0], m[1], m[2], pdir1[vj], pdir2[vj])
                wt = corner_areas[i, j] / point_areas[vj]
                curve1[vj] += wt * c1
                curve12[vj] += wt * c12
                curve2[vj] += wt * c2

        # diagonalización para obtener las direcciones y curvaturas,
        # aunque solo nos interesan las curvaturas
        for i in range(0, len(vertices)):
            curve1[i], curve2[i], pdir1[i], pdir2[i] = diagonalize_curves(pdir1[i], pdir2[i], curve1[i],
                                                                          curve12[i], curve2[i], normals[i])
        self.meanCurvature = (curve1 + curve2) / 2
        self.method = 'Rusinkiewicz Method'
        return curve1, curve2

    def trimesh_method(self, radius=1):
        vertices, faces = self.mesh
        if self.method == 'Trimesh Method':
            return self.meanCurvature
        if self.trimesh is None:
            self.trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.meanCurvature = trimesh.curvature.discrete_mean_curvature_measure(self.trimesh, vertices, radius)
        self.method = 'Trimesh Method'
        return self.meanCurvature

    # obtención del radio con el volumen, con la fórmula para esfera
    def get_radius(self):
        if self.volume is None:
            self.get_volume()
        self.radius = (3 * self.volume / (np.pi * 4)) ** (1 / 3)
        return self.radius

    # obtención del volumen: tomando un punto arbitrario o = (0, 0, 0)
    # el volumen total se puede ver como la suma de los volumenes
    # de cada tetrahedro formado por 4 puntos: o, p, q, r
    # donde p, q, r son los vértices que conforman la cara
    def get_volume(self):
        vertices, faces = self.mesh
        vol = 0
        for i in range(0, len(faces)):
            a, b, c = faces[i]
            p, q, r = vertices[a], vertices[b], vertices[c]
            vol += (1 / 6) * np.dot(p, np.cross(q, r))
        self.volume = abs(vol)
        return self.volume

    # obtención de las fuerzas
    def stress(self, gamma=1):
        if self.radius is None:
            self.get_radius()
        name_curvatures = self.method
        rad = np.ones(len(self.meanCurvature)) * (1 / self.radius)
        self.meanCurvature = 2 * gamma * (self.meanCurvature - rad)
        self.stress_bool = True
        self.method = 'Normal Stress with ' + name_curvatures
        return self.meanCurvature
