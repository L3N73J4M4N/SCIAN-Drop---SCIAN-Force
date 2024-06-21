from forces import Forces3D
import numpy as np
import trimesh
import plotly.graph_objects as go

r = 20
mesh = trimesh.creation.icosphere(subdivisions=5, radius=r)
vertices, faces = mesh.vertices, mesh.faces
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
fig = go.Figure(layout={'title': 'sphere radius='+str(r)+', faces='+str(len(faces))})
fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
fig.show()
vol = 0
for i in range(0, len(faces)):
    a, b, c = faces[i]
    p, q, r = vertices[a], vertices[b], vertices[c]
    vol += (1 / 6) * np.dot(p, np.cross(q, r))
volume = abs(vol)
radius = (3 * volume / (np.pi * 4)) ** (1 / 3)
print(radius)
