#  Copyright (c) 2022. Implementation of "RiemannNet"
#  by Oguzhan Yigit and Richard C. Wilson
import glob
from collections import defaultdict

import torch
import trimesh
import igl
from matplotlib.colors import LinearSegmentedColormap
# import vtk
from scipy import sparse
import numpy as np
import scipy.sparse.linalg as lg
import math
import matplotlib.pyplot as plt
#import meshplot as mp
import pyvista as pv
import CalculCurvature as CC
import os.path

from pointnet_util import square_distance

def plotEdges(v, f, c, m):
    verts = f.shape[0]
    b = np.ones((verts, 4)) * 3
    b[:, 1:] = f

    p = pv.Plotter()
    surf = pv.PolyData(v, b.astype("int"))
    edges = surf.extract_all_edges()

    ed = m.shape[0]
    edg = np.ones((ed, 3)) * 2
    edg[:, 1:] = m
    edg = edg.astype("int64").reshape(ed*3)
    edges.lines = edg
    edges.points = v
    #p.set_background('white')#'gray', top=
    #p.set_background('white', top='white')
    p.enable_anti_aliasing()
    p.add_mesh(edges, scalars=c, line_width=5, cmap='viridis', metallic=0.2, pbr=True)
    #edges.plot(scalars=c, line_width=1, cmap='jet', metallic=0.2,
    #           jupyter_backend='pythreejs', anti_aliasing=True)
    # import matplotlib
    # a = matplotlib.pyplot.get_cmap("viridis")
    # d = ((c - min(c)) / (max(c) - min(c))) * 255
    # edges.save("save3.ply", texture=(a(d.astype(np.int32)) * 255).astype(np.uint8))
    p.show()


def smooth_face_colors(mesh, face_colors, face_neighbors, iterations=2):
    """
    Smooth colors across faces of a mesh based on neighboring faces.

    Parameters:
    mesh : A mesh object (e.g., from PyVista or another library) with face data.
    face_colors : np.ndarray
        Initial colors for each face, shape (num_faces, color_channels).
    face_neighbors : list of lists
        A list where each entry contains indices of neighboring faces for each face.
    iterations : int, optional
        Number of smoothing iterations to perform, by default 2.

    Returns:
    np.ndarray
        Smoothed colors for each face, same shape as `face_colors`.
    """
    smoothed_colors = face_colors.copy()

    # Loop through specified number of iterations for smoothing
    for _ in range(iterations):
        new_colors = np.zeros_like(smoothed_colors)

        # Update each face's color based on its neighbors
        for i in range(mesh.n_faces):
            print(i)
            neighbor_colors = np.array([smoothed_colors[j] for j in face_neighbors[i]])
            avg_color = neighbor_colors.mean(axis=0)
            new_colors[i] = avg_color

        # Update the smoothed colors for the next iteration
        smoothed_colors[:] = new_colors

    return smoothed_colors

def smooth_vertex_colors(mesh, colors, neighbors, iterations=1):
    smoothed_colors = colors.copy()

    # Loop through specified number of iterations for smoothing
    for _ in range(iterations):
        new_colors = np.zeros_like(smoothed_colors)

        # Update each vertex's color based on its neighbors
        for i in range(mesh.n_points):
            neighbor_colors = np.array([smoothed_colors[j] for j in neighbors[i]])
            avg_color = neighbor_colors.mean(axis=0)
            new_colors[i] = avg_color

        # Update the smoothed colors for the next iteration
        smoothed_colors[:] = new_colors

    return smoothed_colors

def get_vertex_neighbors(mesh):
    neighbors = defaultdict(set)
    for face in mesh.faces.reshape(-1, 4)[:, 1:]:  # Ignore the first item in each face (triangle size)
        for i, vertex in enumerate(face):
            for neighbor in face:
                if vertex != neighbor:
                    neighbors[vertex].add(neighbor)
    return neighbors

def get_face_neighbors(faces, n):
    """
    Computes the neighboring faces for each face in the mesh using libigl.

    Parameters:
    faces : np.ndarray
        Array of shape (num_faces, 3) representing the triangular faces of the mesh.

    Returns:
    face_neighbors : list of lists
        A list where each entry contains indices of neighboring faces for each face.
    """
    # Get vertex-face adjacency
    vertex_to_faces = igl.vertex_triangle_adjacency(faces, n)[0]

    # Number of faces
    num_faces = faces.shape[0]

    # Create an empty list to store the neighbors
    face_neighbors = [[] for _ in range(num_faces)]

    # Populate face neighbors without using explicit loops
    for vertex in range(vertex_to_faces.shape[0]):
        connected_faces = vertex_to_faces[vertex]
        if connected_faces.size > 0:
            # Create a boolean mask for self-connections
            mask = np.arange(num_faces)[:, None] != connected_faces[None, :]
            # Get the neighboring faces
            neighbors = connected_faces[mask].reshape(-1)
            for face in connected_faces:
                face_neighbors[face].extend(neighbors[neighbors != face])

    return face_neighbors

def plotFacesAnimate(v, f, c, m, p, smooth=False, vector=None):
    surface = pv.PolyData(v, np.c_[(np.ones(f.shape[0], dtype=np.int32) * 3), f])
    surface.scale([0.5, 0.5, 0.5], inplace=True)
    p.set_background('white')  # 'gray', top=
    p.enable_anti_aliasing()
    vertex_neighbors = get_vertex_neighbors(surface)
    #igl.average_onto_vertices(v,f, vector[:,9:].transpose(1,0))
    if vector is not None:
        surface["vectors"] = igl.average_onto_vertices(v, f, vector.transpose(1, 0)) * 0.015
        surface.set_active_vectors("vectors")
        p.add_mesh(surface.arrows, lighting=False, show_scalar_bar=False)
    if smooth:
        c = smooth_vertex_colors(surface, c, vertex_neighbors)
    sargs = dict(interactive=True, color="black")
    custom_cmap = "turbo"
    p.add_mesh(surface, scalars=c, pbr=True, line_width=0.01,cmap=custom_cmap, scalar_bar_args=sargs, roughness=0.8, edge_color="gray", show_edges=False, metallic=0.0, lighting=True, interpolate_before_map=True, smooth_shading=True, ambient=1, specular=1, specular_power=20)

    light = pv.Light()
    light.position = (-5, -25, -5)  # Adjust position as needed
    light.intensity = 2.5  # Increase intensity for brighter light
    p.add_light(light)

def features(vertices, faces, color, name):
    c = plt.get_cmap('viridis')
    x = (color - np.min(color))
    x /= np.max(x)
    # x = np.array(np.round(x * 255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    shading = {"flat": True,  # Flat or smooth shading of triangles
               "wireframe": False, "wire_width": 0.01, "wire_color": "gray",  # Wireframe rendering
               "width": 800, "height": 800,  # Size of the viewer canvas
               "antialias": True,  # Antialising, might not work on all GPUs
               "scale": 2.0,  # Scaling of the model
               "side": "DoubleSide",  # FrontSide, BackSide or DoubleSide rendering of the triangles
               "colormap": "viridis", "normalize": [None, None],  # Colormap and normalization for colors
               "background": "#ffffff",  # Background color of the canvas
               "line_width": 0.1, "line_color": "black",  # Line properties of overlay lines
               "bbox": False,  # Enable plotting of bounding box
               "point_color": "red", "point_size": 0.05  # Point properties of overlay points
               }
    mp.plot(vertices.numpy(), faces.numpy(), c=C, shading=shading, filename=name)

def threejs(vertices, faces, color, frequency, name):
    c = plt.get_cmap('plasma')
    color = color[:, frequency]
    x = (color - np.min(color))
    x /= np.max(x)
    # x = np.array(np.round(x * 255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    shading = {"flat": True,  # Flat or smooth shading of triangles
               "wireframe": False, "wire_width": 0.01, "wire_color": "gray",  # Wireframe rendering
               "width": 800, "height": 800,  # Size of the viewer canvas
               "antialias": True,  # Antialising, might not work on all GPUs
               "scale": 2.0,  # Scaling of the model
               "side": "DoubleSide",  # FrontSide, BackSide or DoubleSide rendering of the triangles
               "colormap": "viridis", "normalize": [None, None],  # Colormap and normalization for colors
               "background": "#ffffff",  # Background color of the canvas
               "line_width": 0.1, "line_color": "black",  # Line properties of overlay lines
               "bbox": False,  # Enable plotting of bounding box
               "point_color": "red", "point_size": 0.05  # Point properties of overlay points
               }
    mp.plot(vertices.numpy(), faces.numpy(), c=C, shading=shading, filename=name)

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1
    from vtkmodules.vtkCommonCore import vtkLookupTable
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    import matplotlib

    # Set the fixed color limits
    vmin = 0
    vmax = 99

    # Normalize colors to a fixed range [0, 99]
    colors = np.clip(colors, vmin, vmax)  # Clip values outside the range
    colors = (colors - vmin) / (vmax - vmin)  # Normalize to [0, 1]


    cmap = matplotlib.cm.get_cmap('turbo')
    #colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    colors = cmap(colors)
    # write obja
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                               colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)


def get_curvature(mesh, path, plot=False, rewrite=True):
    edge = igl.edge_topology(mesh.vertices, mesh.faces)[0]
    if not os.path.isfile(path[:-4] + ".curvature2") or rewrite:
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = CC.GetCurvaturesAndDerivatives(mesh)
        k1 = np.nan_to_num(PrincipalCurvatures[0,:], nan=0)
        k2 = np.nan_to_num(PrincipalCurvatures[1, :], nan=0)

        perc01 = np.percentile(k1, 5, keepdims=True)
        perc99 = np.percentile(k1, 95, keepdims=True)
        k1 = np.clip(k1, a_min=perc01, a_max=perc99)


        perc01 = np.percentile(k2, 5, keepdims=True)
        perc99 = np.percentile(k2, 95, keepdims=True)
        k2 = np.clip(k2, a_min=perc01, a_max=perc99)


        curvature2 = 0.5 * (k1 + k2)
        #curvature2 = reject_outliers(curvature2)
        if (sum(curvature2 > 0) < (len(curvature2) / 2)):
            curvature2 = -curvature2
        #curvature2 = np.maximum(curvature2, torch.tensor(1e-8))
        curvature2 = torch.tensor(curvature2)
        perc01 = np.percentile(curvature2, 5, keepdims=True)
        perc99 = np.percentile(curvature2, 95, keepdims=True)
        curvature2 = np.clip(curvature2, a_min=perc01, a_max=perc99)
        curvature2 = (curvature2 - torch.mean(curvature2)) / torch.std(curvature2)
        curvature2 = (curvature2[edge[:, 0]] + curvature2[edge[:, 1]]) / 2

        gaussian = k1 * k2
        perc01 = np.percentile(gaussian, 5, keepdims=True)
        perc99 = np.percentile(gaussian, 95, keepdims=True)
        gaussian = np.clip(gaussian, a_min=perc01, a_max=perc99)
        gaussian = torch.tensor(gaussian)
        gaussian = (gaussian - torch.mean(gaussian)) / torch.std(gaussian)
        gaussian = (gaussian[edge[:, 0]] + gaussian[edge[:, 1]]) / 2



        perc01 = np.percentile(k1, 5, keepdims=True)
        perc99 = np.percentile(k1, 95, keepdims=True)
        k1 = np.clip(k1, a_min=perc01, a_max=perc99)
        k1 = torch.tensor(k1)
        k1 = (k1 - torch.mean(k1)) / torch.std(k1)
        k1 = (k1[edge[:, 0]] + k1[edge[:, 1]]) / 2


        perc01 = np.percentile(k2, 5, keepdims=True)
        perc99 = np.percentile(k2, 95, keepdims=True)
        k2 = np.clip(k2, a_min=perc01, a_max=perc99)
        k2 = torch.tensor(k2)
        k2 = (k2 - torch.mean(k2)) / torch.std(k2)
        k2 = (k2[edge[:, 0]] + k2[edge[:, 1]]) / 2



        np.savetxt(path[:-4] + ".curvature2", curvature2)
        np.savetxt(path[:-4] + ".k1", k1)
        np.savetxt(path[:-4] + ".k2", k2)
        np.savetxt(path[:-4] + ".gaussian", gaussian)

    else:
        curvature2 = np.loadtxt(path[:-4] + ".curvature2")
        k1 = np.loadtxt(path[:-4] + ".k1")
        k2 = np.loadtxt(path[:-4] + ".k2")
        gaussian = np.loadtxt(path[:-4] + ".gaussian")

    if plot:
        #plotEdges(mesh.vertices, mesh.faces, PrincipalDir1, edge)
        plotEdges(mesh.vertices, mesh.faces, curvature2, edge)
        plotEdges(mesh.vertices, mesh.faces, gaussian, edge)
        plotEdges(mesh.vertices, mesh.faces, k1, edge)
        plotEdges(mesh.vertices, mesh.faces, k2, edge)
        #plotEdges(mesh.vertices, mesh.faces, mc, edge)

    return curvature2, k1, k2, gaussian, edge#, k11, k22, mc

def normalize_features(features, a, b, c):
    #return ((features.reshape(i, b * p) - torch.mean(features.view(i, b * p), dim=1)[:,
    #                                            None]) / torch.std(features.view(i, b * p), dim=1)[:,
    #                                                    None]).reshape(i, b, p).transpose(1, 0)
    for i in range(a):
        a = features[:, i].reshape(-1)
        a = np.where(a > 0, a / a.max(),
                 np.where(a < 0, -a / a.min(), a))
        features[:, i] = torch.tensor(a).reshape(-1,c)
    return features
def sigmaclip(input):
    mean = np.median(input)
    d = 1.4826 * np.median(np.abs(input - np.median(input)))
    return np.clip(
        input,
        mean - 5 * d,
        mean + 5 * d
    )

def get_curvature2(mesh, path, plot=False, rewrite=False, mina=False, maxa=False):
    edge = igl.edge_topology(mesh.vertices, mesh.faces)[0]
    if not os.path.isfile(path[:-4] + ".curvature2") or rewrite:
        PrincipalCurvatures, PrincFace1, PrincFace2 = CC.GetCurvaturesAndDerivatives(mesh)
        #v1, v2, k1, k2 = igl.principal_curvature(mesh.vertices, mesh.faces)
        #PrincFace1, PrincFace2, Curv1, Curv2  = igl.principal_curvature(mesh.vertices, mesh.faces)
        PrincFace1 = igl.average_onto_faces(mesh.faces, PrincFace1)
        PrincFace2 = igl.average_onto_faces(mesh.faces, PrincFace2)
        PrincFace1 = (PrincFace1) / (np.sqrt(np.sum(PrincFace1 ** 2, axis=1)))[:, None]
        PrincFace2 = (PrincFace2) / (np.sqrt(np.sum(PrincFace2 ** 2, axis=1)))[:, None]

        edge1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
        edge1 = (edge1) / (np.sqrt(np.sum(edge1 ** 2, axis=1)))[:, None]
        edge2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]
        edge2 = (edge2) / (np.sqrt(np.sum(edge2 ** 2, axis=1)))[:, None]
        n = np.cross(edge1, edge2)
        n = (n)/(np.sqrt(np.sum(n**2, axis=1)))[:, None]

        Umax = PrincFace1 - n*np.sum(PrincFace1 * n, axis=1)[:, None]
        PrincFace1 = (Umax) / (np.sqrt(np.sum(Umax ** 2, axis=1)))[:, None]

        Umin = np.cross(n, Umax)
        PrincFace2 = (Umin) / (np.sqrt(np.sum(Umin ** 2, axis=1)))[:, None]
        k1 = (np.nan_to_num(PrincipalCurvatures[0,:], nan=0))
        k2 = (np.nan_to_num(PrincipalCurvatures[1, :], nan=0))



        if not mina:
            perc01 = np.percentile(k1, 3, keepdims=True)
            perc99 = np.percentile(k1, 97, keepdims=True)
            k1 = np.clip(k1, a_min=perc01, a_max=perc99)


            perc01 = np.percentile(k2, 3, keepdims=True)
            perc99 = np.percentile(k2, 97, keepdims=True)
            k2 = np.clip(k2, a_min=perc01, a_max=perc99)
        else:
            perc01 = np.percentile(k1, mina, keepdims=True)
            perc99 = np.percentile(k1, maxa, keepdims=True)
            k1 = np.clip(k1, a_min=perc01, a_max=perc99)

            perc01 = np.percentile(k2, mina, keepdims=True)
            perc99 = np.percentile(k2, maxa, keepdims=True)
            k2 = np.clip(k2, a_min=perc01, a_max=perc99)


        curvature2 =  0.5 * (k1+k2)#(2/np.pi)*np.arctan((k2+k1)/(k2-k1))
        curvature2 = torch.tensor(curvature2)
        curvature3 = (curvature2 - torch.min(curvature2)) / (torch.max(curvature2) - torch.min(curvature2))
        curvature2f = igl.average_onto_faces(mesh.faces, curvature3.numpy())


        curvature2P = (curvature2 - torch.min(curvature2)) / (torch.max(curvature2) - torch.min(curvature2))

        curvature2 = (curvature2[edge[:, 0]] + curvature2[edge[:, 1]]) / 2
        curvature2 = (curvature2 - torch.min(curvature2)) / (torch.max(curvature2) - torch.min(curvature2))


        gaussian = k1*k2#np.sqrt((k1**2 + k2**2)/2)

        gaussian = torch.tensor(gaussian)
        gaussian2 = (gaussian - torch.min(gaussian)) / (torch.max(gaussian) - torch.min(gaussian))
        gaussianf = igl.average_onto_faces(mesh.faces, gaussian2.numpy())

        gaussianP = (gaussian - torch.min(gaussian)) / (torch.max(gaussian) - torch.min(gaussian))

        gaussian = (gaussian[edge[:, 0]] + gaussian[edge[:, 1]]) / 2
        gaussian = (gaussian - torch.min(gaussian)) / (torch.max(gaussian) - torch.min(gaussian))

        k1 = torch.tensor(k1)
        k1r = (k1 - torch.min(k1)) / (torch.max(k1) - torch.min(k1))
        k1f = igl.average_onto_faces(mesh.faces, k1r.numpy())


        k1P = (k1 - torch.min(k1)) / (torch.max(k1) - torch.min(k1))

        k1 = (k1[edge[:, 0]] + k1[edge[:, 1]]) / 2
        k1 = (k1 - torch.min(k1)) / (torch.max(k1) - torch.min(k1))
        k2 = torch.tensor(k2)
        k2r = (k2 - torch.min(k2)) / (torch.max(k2) - torch.min(k2))
        k2f = igl.average_onto_faces(mesh.faces, k2r.numpy())


        k2P = (k2 - torch.min(k2)) / (torch.max(k2) - torch.min(k2))

        k2 = (k2[edge[:, 0]] + k2[edge[:, 1]]) / 2
        k2 = (k2 - torch.min(k2)) / (torch.max(k2) - torch.min(k2))


        np.savetxt(path[:-4] + ".curvature2", curvature2)
        np.savetxt(path[:-4] + ".curvature2f", curvature2f)
        np.savetxt(path[:-4] + ".k1", k1)
        np.savetxt(path[:-4] + ".k1f", k1f)
        np.savetxt(path[:-4] + ".k2", k2)
        np.savetxt(path[:-4] + ".k2f", k2f)
        np.savetxt(path[:-4] + ".gaussian", gaussian)
        np.savetxt(path[:-4] + ".gaussianf", gaussianf)
        np.savetxt(path[:-4] + ".curvature2P", curvature2P)
        np.savetxt(path[:-4] + ".k1P", k1P)
        np.savetxt(path[:-4] + ".k2P", k2P)
        np.savetxt(path[:-4] + ".gaussianP", gaussianP)
        np.savetxt(path[:-4] + ".p1", PrincFace1)
        np.savetxt(path[:-4] + ".p2", PrincFace2)
        np.savetxt(path[:-4] + ".n", n)
    else:
        curvature2 = np.loadtxt(path[:-4] + ".curvature2")
        curvature2f = np.loadtxt(path[:-4] + ".curvature2f")
        k1 = np.loadtxt(path[:-4] + ".k1")
        k1f = np.loadtxt(path[:-4] + ".k1f")
        k2 = np.loadtxt(path[:-4] + ".k2")
        k2f = np.loadtxt(path[:-4] + ".k2f")
        gaussian = np.loadtxt(path[:-4] + ".gaussian")
        gaussianf = np.loadtxt(path[:-4] + ".gaussianf")
        curvature2P = np.loadtxt(path[:-4] + ".curvature2P")
        k1P = np.loadtxt(path[:-4] + ".k1P")
        k2P = np.loadtxt(path[:-4] + ".k2P")
        gaussianP = np.loadtxt(path[:-4] + ".gaussianP")
        PrincFace1 = np.loadtxt(path[:-4] + ".p1")
        PrincFace2 = np.loadtxt(path[:-4] + ".p2")
        n = np.loadtxt(path[:-4] + ".n")

    if plot:
        #plotEdges(mesh.vertices, mesh.faces, PrincipalDir1, edge)
        plotEdges(mesh.vertices, mesh.faces, curvature2, edge)
        plotEdges(mesh.vertices, mesh.faces, gaussian, edge)
        plotEdges(mesh.vertices, mesh.faces, k1, edge)
        plotEdges(mesh.vertices, mesh.faces, k2, edge)
        #plotEdges(mesh.vertices, mesh.faces, mc, edge)

    return curvature2, k1, k2, gaussian, curvature2P, k1P, k2P, gaussianP, edge, PrincFace1, PrincFace2, curvature2f, gaussianf, k1f, k2f, n#, k11, k22, mc


def trainedSet(x, l):
    dist_matrix = torch.zeros(l, l, requires_grad=False)
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            dist_matrix[i, j] = torch.norm(x[i] - x[j], 2)
            dist_matrix[j, i] = dist_matrix[i, j]

    a = dist_matrix.detach().numpy()
    np.savetxt("trainSingle.txt", a, fmt="%s")
    b = plt.subplot(2, 2, 2)
    b.cla()
    plt.ion()
    plt.imshow(a)

    # plt.colorbar()
    plt.show()
    plt.pause(0.0001)
    del dist_matrix
    return (sum(sum(a)))


def calcDistMatrix(x, plot=False):
    a = np.sqrt(np.sum((x.cpu().numpy()[:, np.newaxis, :] - x.cpu().numpy()[np.newaxis, :, :]) ** 2, axis=-1))
    #a = dist_matrix.detach().numpy()
    np.savetxt("train2.txt", a, fmt="%s")
    print(sum(sum(a)))
    if plot:
        b = plt.subplot(2, 2, 3)
        b.cla()
        plt.ion()
        plt.imshow(a)
        # plt.colorbar()
        plt.show()
        plt.pause(0.0001)


def calcDistMatrix2(x, plot=False):

    a = np.sqrt(np.sum((x.cpu().numpy()[:, np.newaxis, :] - x.cpu().numpy()[np.newaxis, :, :]) ** 2, axis=-1))
    np.savetxt("reala.txt", a, fmt="%s")
    if plot:
        b = plt.subplot(2, 2, 4)
        b.cla()
        plt.ion()
        plt.imshow(a)
        plt.clim(-7, 7)
        # plt.colorbar()
        plt.show()
        plt.pause(0.0001)
    return a


def cot(x):
    return 1 / torch.tan(x)


def get_angle(a, b, c):
    alpha = torch.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    return alpha

def laplacianPartDebug(z, x, y, c, a, b):
    J = (-1 / 2) * ((b / ((b * c * torch.sin(x)) * torch.sin(y) ** 2)))
    return J

def laplacianPart(z, x, y, c, a, b):
    J = (-1 / 2) * ((b / ((b * c * torch.sin(x)) * torch.sin(y) ** 2)) - (
                (c * torch.cos(x)) / ((a * c * torch.sin(y)) * (torch.sin(z) ** 2))))
    return J


def laplacianPart2(z, x, y, c, a, b):
    J = (-1 / 2) * ((a / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)) - (
                (c * torch.cos(y)) / ((b * c * torch.sin(x)) * torch.sin(z) ** 2)))
    return J


def laplacianPartij(z, x, y, c, a, b):
    J = (1 / 2) * ((b / ((b * c * torch.sin(x)) * torch.sin(y) ** 2)))
    return J


def laplacianPart2ij(z, x, y, c, a, b):
    J = (1 / 2) * ((a / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)))
    return J


def laplacianPartk1(x, y, z, a, b, c):
    J = (1 / 2) * (((a * torch.cos(y)) / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)) + (b * torch.cos(x)) / (
                (b * c * torch.sin(x)) * torch.sin(y) ** 2))
    return J


def laplacianPartk2(x, y, z, a, b, c):
    J = (1 / 2) * (((a * torch.cos(z)) / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)) + (c * torch.cos(x)) / (
                (b * c * torch.sin(x)) * torch.sin(z) ** 2))
    return J

def laplacianPartk2A(x, y, z, a, b, c):
    J = (-1 / 2) * (((a * torch.cos(z)) / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)))
    return J


def laplacianPartk2B(x, y, z, a, b, c):
    J = (-1 / 2) * (((c * torch.cos(x)) / ((b * c * torch.sin(x)) * torch.sin(z) ** 2)))
    return J





def laplacianPartk1A(x, y, z, a, b, c):
    J = (-1 / 2) * (((a * torch.cos(y)) / ((b * c * torch.sin(x)) * torch.sin(x) ** 2)))
    return J


def laplacianPartk1B(x, y, z, a, b, c):
    J = (-1 / 2) * (((b * torch.cos(x)) / ((b * c * torch.sin(x)) * torch.sin(y) ** 2)))
    return J




def voronoiPartA(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)

    idx1 = torch.max(torch.column_stack((x, y, z)),1)[0] < torch.pi/2
    v[idx1] = (1 / 8) * ((-(c[idx1] ** 3 / ((torch.sin(z[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1])))) + cot(z[idx1]) * 2 * c[idx1]) - (-((a[idx1] ** 3 * torch.cos(y[idx1])) / ((torch.sin(x[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1]))))))

    idx2 = y > torch.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(z[idx2])


    idx3 = (y < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)),1)[0] > torch.pi/2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(z[idx3])
    
    return v

def voronoiPartB(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)),1)[0] < torch.pi/2
    v[idx1] = (1 / 8) * ((-(b[idx1] ** 3 / ((torch.sin(y[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1])))) + cot(y[idx1]) * 2 * b[idx1]) - (-((a[idx1] ** 3 * torch.cos(z[idx1])) / ((torch.sin(x[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1]))))))

    idx2 = z > math.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(y[idx2])


    idx3 = (z < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)),1)[0] > torch.pi/2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(y[idx3])

    return v



def voronoiPartImpact1(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)),1)[0] < torch.pi/2
    v[idx1] = (1 / 8) * (((a[idx1] ** 3 * torch.cos(y[idx1])) / ((torch.sin(x[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))) + ((b[idx1] ** 3 * torch.cos(x[idx1])) / ((torch.sin(y[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))))

    idx2 = z > math.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5) * torch.cos(z[idx2]))

    idx3 = (z < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)),1)[0] > torch.pi/2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5) * torch.cos(z[idx3]))
    return v


def voronoiPartImpact2(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)),1)[0] < torch.pi/2
    v[idx1] = (1 / 8) * (((c[idx1] ** 3 * torch.cos(x[idx1])) / ((torch.sin(z[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))) + ((a[idx1] ** 3 * torch.cos(z[idx1])) / ((torch.sin(x[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))))  # (1 / 8) * (((b**3 * torch.cos(y)) / ((torch.sin(y) ** 2)*(b*c*torch.sin(x)))) + ((a**3 * torch.cos(z)) / ((torch.sin(x) ** 2)*(b*c*torch.sin(x)))))#(1 / 8) * ((-(c**3 * torch.cos(y)) / ((torch.sin(z) ** 2)*(b*c*torch.sin(x)))))

    idx2 = y > torch.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(y[idx2])

    idx3 = (y < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)),1)[0] > torch.pi/2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(y[idx3])
    return v

def voronoiPartImpact1BU(x, y, z, a, b, c):
    if (max([x, y, z])) < math.pi / 2:
        dtheta = (1 / 8) * (((a ** 3 * torch.cos(y)) / ((torch.sin(x) ** 2) * (b * c * torch.sin(x)))) + (
                    (b ** 3 * torch.cos(x)) / ((torch.sin(y) ** 2) * (b * c * torch.sin(x)))))
    elif z > math.pi / 2:
        dtheta = (1 / 8) * ((a * b * c) / (b * c * torch.sin(x) * 0.5) * torch.cos(z))
    else:
        dtheta = (1 / 16) * ((a * b * c) / (b * c * torch.sin(x) * 0.5) * torch.cos(z))
    return dtheta


def voronoiPartImpact2BU(x, y, z, a, b, c):
    if (max([x, y, z])) < math.pi / 2:
        dtheta = (1 / 8) * (((c ** 3 * torch.cos(x)) / ((torch.sin(z) ** 2) * (b * c * torch.sin(x)))) + (
                    (a ** 3 * torch.cos(z)) / ((torch.sin(x) ** 2) * (b * c * torch.sin(
                x)))))  # (1 / 8) * (((b**3 * torch.cos(y)) / ((torch.sin(y) ** 2)*(b*c*torch.sin(x)))) + ((a**3 * torch.cos(z)) / ((torch.sin(x) ** 2)*(b*c*torch.sin(x)))))#(1 / 8) * ((-(c**3 * torch.cos(y)) / ((torch.sin(z) ** 2)*(b*c*torch.sin(x)))))
    elif y > math.pi / 2:
        dtheta = (1 / 8) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(y)
    else:
        dtheta = (1 / 16) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(y)
    return dtheta


def voronoiPartBBU(x, y, z, a, b, c):
    if (max([x, y, z])) < math.pi / 2:
        dtheta = (1 / 8) * ((-(b ** 3 / ((torch.sin(y) ** 2) * (a * b * torch.sin(z)))) + cot(y) * 2 * b) - (
            -((a ** 3 * torch.cos(z)) / ((torch.sin(x) ** 2) * (a * b * torch.sin(z))))))
    elif z > math.pi / 2:
        dtheta = (1 / 8) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(y)
    else:
        dtheta = (1 / 16) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(y)
    return dtheta

def voronoiPartABU(x, y, z, a, b, c):
    if (max([x, y, z])) < math.pi / 2:
        dtheta = (1 / 8) * ((-(c ** 3 / ((torch.sin(z) ** 2) * (a * b * torch.sin(z)))) + cot(z) * 2 * c) - (-(
                    (a ** 3 * torch.cos(y)) / ((torch.sin(x) ** 2) * (a * b * torch.sin(
                z))))))  # (1 / 8) * ((-(b**3 / ((torch.sin(y) ** 2)*(a*b*torch.sin(z)))) + cot(y) * 2 * b) - (-((c**3 * torch.cos(x)) / ((torch.sin(z) ** 2)*(a*b*torch.sin(z))))))
    elif y > math.pi / 2:
        dtheta = (1 / 8) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(z)
    else:
        dtheta = (1 / 16) * ((a * b * c) / (b * c * torch.sin(x) * 0.5)) * torch.cos(z)
    return dtheta

def get_laplace_gradient_edge(verts, ITris, edge, angles,cotAngles, edge_length, device, o):

    ts = torch.tensor(igl.edge_flaps(ITris.long().numpy())[2])
    corners = torch.tensor(igl.edge_flaps(ITris.long().numpy())[3])

    # calculate dL, dA
    k1 = torch.diag(ITris[ts[:, 0].long()][:, corners[:, 0].long()])  # ITris[ts[0]][corners[0]]
    k2 = torch.diag(ITris[ts[:, 1].long()][:, corners[:, 1].long()])  # ITris[ts[1]][corners[1]]

    k1_index = corners[:, 0]
    k2_index = corners[:, 1]
    i1_index = (ITris[ts[:, 0].long()] == edge[:, 0].unsqueeze(0).transpose(1, 0)).nonzero(as_tuple=True)[1]
    j1_index = (ITris[ts[:, 0].long()] == edge[:, 1].unsqueeze(0).transpose(1, 0)).nonzero(as_tuple=True)[1]
    i2_index = (ITris[ts[:, 1].long()] == edge[:, 0].unsqueeze(0).transpose(1, 0)).nonzero(as_tuple=True)[1]
    j2_index = (ITris[ts[:, 1].long()] == edge[:, 1].unsqueeze(0).transpose(1, 0)).nonzero(as_tuple=True)[1]

    dij = edge_length[ts[:, 0].long(), k1_index.long()]
    djk1 = edge_length[ts[:, 0].long(), i1_index.long()]
    dik1 = edge_length[ts[:, 0].long(), j1_index.long()]
    dik2 = edge_length[ts[:, 1].long(), i2_index.long()]
    djk2 = edge_length[ts[:, 1].long(), j2_index.long()]

    Thetaij1 = angles[ts[:, 0].long(), j1_index.long()]
    Thetaik1 = angles[ts[:, 0].long(), i1_index.long()]
    Thetajk1 = angles[ts[:, 0].long(), k1_index.long()]

    Thetaij2 = angles[ts[:, 1].long(), j2_index.long()]
    Thetaik2 = angles[ts[:, 1].long(), k2_index.long()]
    Thetajk2 = angles[ts[:, 1].long(), i2_index.long()]

    Lii = (laplacianPart(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij) + laplacianPart2(Thetaij2, Thetaik2, Thetajk2,
                                                                                         djk2, dij, dik2)).to(device)
    Ljj = (laplacianPart(Thetaik1, Thetaij1, Thetajk1, djk1, dik1, dij) + laplacianPart2(Thetajk2, Thetaik2, Thetaij2,
                                                                                         dik2, dij, djk2)).to(device)
    Lij = (laplacianPartij(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij) + laplacianPart2ij(Thetaij2, Thetaik2,
                                                                                             Thetajk2, djk2, dij,
                                                                                             dik2)).to(device)
    Lk2k2 = (laplacianPartk2(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2)).to(device)
    Lk1k1 = (laplacianPartk1(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij)).to(device)
    Lik2 = (laplacianPartk2A(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2)).to(device)
    Ljk2 = (laplacianPartk2B(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2)).to(device)
    Lik1 = (laplacianPartk1A(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij)).to(device)
    Ljk1 = (laplacianPartk1B(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij)).to(device)

    gL = torch.column_stack((Lii, Ljj, Lij, Lij, Lk2k2, Lk1k1, Lik2, Lik2, Ljk2, Ljk2, Lik1, Lik1, Ljk1, Ljk1))

    Vii = ((voronoiPartA(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij)) + (
        voronoiPartB(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2))).to(device)
    Vjj = ((voronoiPartA(Thetaik1, Thetaij1, Thetajk1, djk1, dik1, dij)) + (
        voronoiPartB(Thetajk2, Thetaik2, Thetaij2, dik2, dij, djk2))).to(device)
    Vk1k1 = voronoiPartImpact1(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij).to(device)
    Vk2k2 = voronoiPartImpact2(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2).to(device)

    gV = torch.column_stack((Vii, Vjj, Vk1k1, Vk2k2))

    i = edge[:, 0]
    j = edge[:, 1]

    indicesLx = torch.column_stack((i, j, i, j, k2, k1, i, k2, j, k2, i, k1, j, k1)).long().to(device)
    indicesLy = torch.column_stack((i, j, j, i, k2, k1, k2, i, k2, j, k1, i, k1, j)).long().to(device)

    # indicesV = [[i, i], [j, j], [k1, k1], [k2, k2]]
    indicesV = torch.column_stack((i, j, k1, k2)).long().to(device)
    N=verts.shape[0]
    dL = torch.zeros(N, N, device=device, dtype=torch.float64)
    dL[indicesLx[o, :], indicesLy[o, :]] = gL[o, :]

    dV = torch.zeros(N, N, device=device, dtype=torch.float64)
    dV[indicesV[o, :], indicesV[o, :]] = gV[o, :]

    return dL, dV

def get_cotan_laplacian(VPos, ITris, Riemannian, edges):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    #import time
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M * 6)
    J = np.zeros(M * 6)
    V = np.zeros(M * 6)
    A = np.zeros(N, dtype=np.float32)
    angles = torch.zeros(M, 3, dtype=torch.float64)
    cotAngle = torch.zeros(M, 3, dtype=torch.float64)
    e = torch.zeros(M, 3, dtype=torch.float64)
    z = torch.zeros(M, 3, dtype=torch.float64)
    squared_edge_length = torch.zeros(M, 3, dtype=torch.float64)
    #Riemannian = torch.clip(Riemannian, min=0.1)
    # project Riemannian onto edges
    pRiemann = sparse.coo_matrix((Riemannian.tolist(), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N))
    pRiemann = torch.tensor((pRiemann + pRiemann.transpose()).todense()).squeeze(0)
    summation = 1
    alpha = 0.00001
    while summation>=0.00001:
        summation = 0        # correct weightings to obey triangular inequality
        for rotation in range(3):
            [i, j, k] = [rotation, (rotation + 1) % 3, (rotation + 2) % 3]

            lij = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, j]])
            lik = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, k]])
            ljk = torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, j], ITris[:, k]])

            pRiemann[ITris[lij < alpha, i], ITris[lij < alpha, j]] += (alpha - lij[lij < alpha])
            pRiemann[ITris[lij < alpha, j], ITris[lij < alpha, i]] += (alpha - lij[lij < alpha])
            pRiemann[ITris[lik < alpha, i], ITris[lik < alpha, k]] += (alpha - lik[lik < alpha])
            pRiemann[ITris[lik < alpha, k], ITris[lik < alpha, i]] += (alpha - lik[lik < alpha])
            pRiemann[ITris[ljk < alpha, j], ITris[ljk < alpha, k]] += (alpha - ljk[ljk < alpha])
            pRiemann[ITris[ljk < alpha, k], ITris[ljk < alpha, j]] += (alpha - ljk[ljk < alpha])
            lij = torch.clip(lij, min=alpha)
            lik = torch.clip(lik, min=alpha)
            ljk = torch.clip(ljk, min=alpha)

            b = (lik + ljk - lij)

            mu = (1/3) * (e[:, i]-e[:, k]-e[:, j]-b)
            theta = torch.min(torch.stack((-mu, z[:, rotation])).transpose(1,0), dim=1)[0]

            e[:, k] -= theta
            e[:, j] -= theta
            e[:, i] += theta
            z[:, rotation] -= theta

            summation = summation + torch.sum(torch.abs(theta))
    e *= 1.01
    #print(len(e.nonzero()))
    # Step 1: Compute cotangent weights
    for shift in range(3):
        # For all 3 shifts of the roles of triangle vertices
        # to compute different cotangent weights
        [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]

        lij = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, j]]) + e[:, i]))
        lik = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, k]])) + e[:, k])
        ljk = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, j], ITris[:, k]])) + e[:, j])


        alpha2 = torch.acos((lik ** 2 + ljk ** 2 - lij ** 2) / (2 * lik * ljk))
        #if torch.isnan(alpha2).any():
        #    print("invalid state detected")
        cotAlpha = 1 / torch.tan(alpha2)

        I[shift * M * 2:shift * M * 2 + M] = ITris[:, i]
        J[shift * M * 2:shift * M * 2 + M] = ITris[:, j]
        V[shift * M * 2:shift * M * 2 + M] = cotAlpha

        I[shift * M * 2 + M:shift * M * 2 + 2 * M] = ITris[:, j]
        J[shift * M * 2 + M:shift * M * 2 + 2 * M] = ITris[:, i]
        V[shift * M * 2 + M:shift * M * 2 + 2 * M] = cotAlpha

        angles[:, k] = alpha2
        cotAngle[:, k] = cotAlpha

        squared_edge_length[:, k] = (lij) ** 2
    # print(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)))
    # print((torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) * (1 + pRiemann[
    #         ITris[:, i], ITris[:, j]]) + e[:, i])))
    #print(squared_edge_length[0,:])
    # Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N), dtype=np.float64).tocsr()
    # Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L

    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    Vvts1 = L.data.tolist()
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N), dtype=np.float64)

    rangle = angles
    el = squared_edge_length**(1/2)
    # Voronoi cell calculation
    cotAngle = cotAngle.cpu().numpy()
    angles = angles.cpu().numpy()
    squared_edge_length = squared_edge_length.cpu().numpy()
    faces_area = np.zeros(M, dtype=np.float64)
    for i in range(3):
        faces_area = faces_area + (1 / 4) * (squared_edge_length[:, i] * (cotAngle[:, i]))

    threshold = math.pi / 2
    tri = ITris.cpu().numpy()
    for edge in range(N):
        for shift in range(3):
            [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]
            indices = np.where(tri[:, i] == edge)
            for triangle in indices[0]:
                if max(angles[triangle, :]) <= threshold:
                    A[edge] += (1 / 8) * ((cotAngle[triangle, j] * squared_edge_length[triangle, j]) + (
                                cotAngle[triangle, k] * squared_edge_length[triangle, k]))
                elif angles[triangle, i] > threshold:
                    A[edge] += faces_area[triangle] / 2
                else:
                    A[edge] += faces_area[triangle] / 4
    A = sparse.diags(A)#sparse.coo_matrix((A, (list(range(N)), list(range(N)))), shape=(N, N))
    return A, L / 2, rangle, el

def get_cotan_laplacian_igl(VPos, ITris, Riemannian, edges):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    #import time
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M * 6)
    J = np.zeros(M * 6)
    V = np.zeros(M * 6)
    rangle = torch.zeros(M, 3, dtype=torch.float32)
    e = torch.zeros(M, 3, dtype=torch.float32)
    z = torch.zeros(M, 3, dtype=torch.float32)
    el = torch.zeros(M, 3, dtype=torch.float32)
    pRiemann = sparse.coo_matrix((Riemannian.tolist(), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N))
    pRiemann = torch.tensor((pRiemann + pRiemann.transpose()).todense()).squeeze(0)
    summation = 1
    alpha = 0.0001
    # first pass
    for rotation in range(3):
        [i, j, k] = [rotation, (rotation + 1) % 3, (rotation + 2) % 3]

        lij = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, j]])
        lik = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, k]])
        ljk = torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, j], ITris[:, k]])

        a = torch.cat((ITris[lij < alpha, i], ITris[lik < alpha, i], ITris[ljk < alpha, j])).cpu().numpy()
        #_, idx = np.unique(a, return_index=True)
        #a = (a[np.sort(idx)])

        b = torch.cat((ITris[lij < alpha, j], ITris[lik < alpha, k], ITris[ljk < alpha, k])).cpu().numpy()
        #_, idx = np.unique(b, return_index=True)
        #b = (b[np.sort(idx)])
        if len(a)>0:
            idx = torch.unique(torch.stack((torch.tensor(a),torch.tensor(b))), dim=0)
            pRiemann[idx[0,:], idx[1,:]] += alpha - (torch.sqrt(torch.sum(((VPos[idx[0,:], :]) - (VPos[idx[1,:], :])) ** 2, 1)) + pRiemann[idx[0,:], idx[1,:]])
            pRiemann[idx[1,:], idx[0,:]] = pRiemann[idx[0,:], idx[1,:]]
    # second pass
    beta = 0.0001
    while summation>=0.0001:
        summation = 0        # correct weightings to obey triangular inequality
        for rotation in range(3):
            [i, j, k] = [rotation, (rotation + 1) % 3, (rotation + 2) % 3]

            lij = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, j]])
            lik = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, k]])
            ljk = torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, j], ITris[:, k]])

            b = (lik + ljk - lij)
            #b[b < 0] = torch.clip(b[b < 0], max=-beta)
            b[torch.where((b>0) & (b<beta))] = torch.clip(b[torch.where((b>0) & (b<beta))], max=-beta)

            mu = (1/3) * (e[:, i]-e[:, k]-e[:, j]-b)
            theta = torch.min(torch.stack((-mu, z[:, rotation])).transpose(1,0), dim=1)[0]

            e[:, k] -= theta
            e[:, j] -= theta
            e[:, i] += theta
            z[:, rotation] -= theta
            summation = summation + torch.sum(torch.abs(theta))

    e[torch.where((e>0) & (e<beta))] = torch.clip(e[torch.where((e>0) & (e<beta))], min=beta)
    e[torch.where((e < 0) & (e > -beta))] = torch.clip(e[torch.where((e < 0) & (e > -beta))], max=-beta)
    e *= 0

    #print(len(e.nonzero()))
    # last pass
    for shift in range(3):
        # For all 3 shifts of the roles of triangle vertices
        # to compute different cotangent weights
        [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]

        lij = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, j]]) + e[:, i]))
        lik = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, k]])) + e[:, k])
        ljk = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, j], ITris[:, k]])) + e[:, j])

        rangle[:, k] = torch.acos((lik ** 2 + ljk ** 2 - lij ** 2) / (2 * lik * ljk))
        el[:, k] = (lij)

    L = -igl.cotmatrix(VPos.numpy(), ITris.numpy())
    A = igl.massmatrix_intrinsic(el.float().cpu().numpy(), ITris.numpy(), igl.MASSMATRIX_TYPE_VORONOI)
    A = np.diag(A.toarray())
    #A = np.diag(igl.massmatrix(VPos.numpy(), ITris.numpy()).toarray())
    return A, L, rangle, el



def get_cotan_laplacian_igl_default(VPos, ITris, edges):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    #import time
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M * 6)
    J = np.zeros(M * 6)
    V = np.zeros(M * 6)
    rangle = torch.zeros(M, 3, dtype=torch.float32)
    e = torch.zeros(M, 3, dtype=torch.float32)
    z = torch.zeros(M, 3, dtype=torch.float32)
    el = torch.zeros(M, 3, dtype=torch.float32)


    #print(len(e.nonzero()))
    # last pass
    for shift in range(3):
        # For all 3 shifts of the roles of triangle vertices
        # to compute different cotangent weights
        [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]

        lij = torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)))
        lik = torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)))
        ljk = torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)))

        rangle[:, k] = torch.acos((lik ** 2 + ljk ** 2 - lij ** 2) / (2 * lik * ljk))
        el[:, k] = (lij)


    L = -igl.cotmatrix(VPos.numpy(), ITris.numpy())
    A = igl.massmatrix_intrinsic(el.float().cpu().numpy(), ITris.numpy(), igl.MASSMATRIX_TYPE_VORONOI)
    A = np.diag(A.toarray())
    return A, L, rangle, el


def anisotropic_lbo_gradient(VPos, ITris, tsl, pRiemann, e, i1_index, j1_index, k1_index, maxCurvature, minCurvature, dij, dik1, djk1, Thetaij1, Thetaik1, Thetajk1, anisotropic1, anisotropic2, dotmin1, dotmin2, dotmax1, dotmax2, edge, k1, dotmin1s, dotmin2s, dotmax1s, dotmax2s, diagonalDotsMin1, diagonalDotsMin2, diagonalDotsMax1, diagonalDotsMax2, diagonalDotsMin1S, diagonalDotsMin2S, diagonalDotsMax1S, diagonalDotsMax2S):

    anisotropic1 = anisotropic1.cpu()
    anisotropic2 = anisotropic2.cpu()
    triArea = dij * djk1 * torch.sin(Thetaij1) * 0.5

    cotAlpha = 1 / torch.tan(Thetajk1)
    cotAlpha2 = 1 / torch.tan(Thetaij1)
    cotAlpha3 = 1 / torch.tan(Thetaik1)

    i1 = edge[:, 0]
    j1 = edge[:, 1]

    flipped_triangles = torch.logical_or((torch.logical_or((i1_index%2>j1_index), ((i1_index+1)<j1_index))), (i1_index-1==j1_index))

    deriv2 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))

    temp = ((dij) / (2 * triArea))
    derivative2 = deriv2
    derivative2[flipped_triangles] = temp[flipped_triangles]


    dotmaxe1 = diagonalDotsMax1[i1_index, tsl]
    dotmine1 = diagonalDotsMin1[i1_index, tsl]
    dotmaxe1s = diagonalDotsMax1S[i1_index, tsl]
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmine1s = diagonalDotsMin1S[i1_index, tsl]
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]

    fullDerivativeDiagonalA1 = -0.5 * (((anisotropic1[tsl] * (dotmaxe1) ** 2 + anisotropic2[tsl] * (dotmine1) ** 2)) * (dij / (2 * triArea) * 1 / torch.sin(Thetajk1) ** 2 + ((-((dik1 * torch.cos(Thetaik1)) / (2 * triArea))) * 1 / torch.sin(Thetaij1) ** 2)) + (anisotropic1[tsl]* 2 * torch.sin(torch.acos(dotmaxe1)) * (dotmaxe1) * torch.sign(dotmaxe1s) * derivative2 + anisotropic2[tsl] * 2 * torch.sin(torch.acos(dotmine1)) * dotmine1 * torch.sign(dotmine1s) * derivative2) * (cotAlpha + cotAlpha2))

    derivative3 = (-dij) / (2 * triArea)
    derivative3[flipped_triangles] = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))[flipped_triangles]

    dotmaxe2 = diagonalDotsMax1[j1_index, tsl]
    #dotmaxe2[flipped_triangles] = diagonalDotsMax1[j1_index, tsl][flipped_triangles]
    #dotmaxe2 = diagonalDotsMax1[j1_index, tsl]
    dotmine2 = diagonalDotsMin1[j1_index, tsl]
    #dotmine2[flipped_triangles] = diagonalDotsMin1[j1_index, tsl][flipped_triangles]
    #dotmine2 = diagonalDotsMin1[j1_index, tsl]
    dotmaxe2s = diagonalDotsMax1S[j1_index, tsl]
    dotmaxe2s[flipped_triangles] = diagonalDotsMax1S[j1_index, tsl].squeeze(0)[flipped_triangles]
    dotmine2s = diagonalDotsMin1S[j1_index, tsl]
    dotmine2s[flipped_triangles] = diagonalDotsMin1S[j1_index, tsl].squeeze(0)[flipped_triangles]
    fullDerivativeDiagonalA2 = -(0.5 * (((anisotropic1[tsl] * (dotmaxe2) ** 2 + anisotropic2[tsl] * (dotmine2) ** 2)) * (dij / (2 * triArea) * 1 / torch.sin(Thetajk1) ** 2 + (-((djk1 * torch.cos(Thetaij1)) / (2 * triArea)) * 1 / torch.sin(Thetaik1) ** 2)) + (anisotropic1[tsl] * 2 * torch.sin(torch.acos(dotmaxe2)) * (dotmaxe2) * -torch.sign(dotmaxe2s) * derivative3 + anisotropic2[tsl] * 2 * torch.sin(torch.acos(dotmine2)) * dotmine2 * -torch.sign(dotmine2s) * derivative3) * (cotAlpha + cotAlpha3)))

    dotmine1 = diagonalDotsMin1[k1_index, tsl]
    dotmine1s = diagonalDotsMin1S[k1_index, tsl]

    dotmaxe1 = diagonalDotsMax1[k1_index, tsl]
    dotmaxe1s = diagonalDotsMax1S[k1_index, tsl]

    derivative1 = (djk1 * torch.cos(Thetaij1)) / (2 * triArea)
    derivative1[flipped_triangles] = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))[flipped_triangles]

    fullDerivativeDiagonalA3 = (0.5 * (
            ((anisotropic1[tsl] * (dotmaxe1) ** 2 + anisotropic2[tsl] * (dotmine1) ** 2)) * (
            (djk1 * torch.cos(Thetaij1)) / (2 * triArea) * 1 / torch.sin(Thetaik1) ** 2 + (
            (((dik1 * torch.cos(Thetaik1)) / (2 * triArea))) * 1 / torch.sin(Thetaij1) ** 2)) + (
                    anisotropic1[tsl] * 2 * torch.sin(torch.acos(dotmaxe1)) * (
                dotmaxe1) * torch.sign(dotmaxe1s) * derivative1 + anisotropic2[tsl] * 2 * torch.sin(
                torch.acos(dotmine1)) * dotmine1 * torch.sign(dotmine1s) * derivative1) * (cotAlpha2 + cotAlpha3)))


    deriv1 = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    deriv2 = ((dij) / (2 * triArea))

    temp = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2

    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    dotmaxe1 = torch.tensor(dotmax1[k1, i1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[k1, i1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[k1, i1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[k1, i1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[k1, i1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[k1, i1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[k1, i1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[k1, i1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[i1, k1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (torch.sign(dotmaxe1s) * derivative1)) + (-torch.sin(torch.acos(dotmaxe2)) * (torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (torch.sign(dotmine1s) * derivative1)) + (-torch.sin(torch.acos(dotmine2)) * (torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA4 = (torch.sin(Thetaij1)*0.5 * (anisotropic1[tsl] * dOffDiagonal1 + anisotropic2[tsl] * dOffDiagonal2) - 0.5*(anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (dotmine2 * dotmine1)) * (((-dik1 * torch.cos(Thetaik1)) / (2 * triArea)) * torch.cos(Thetaij1)))/(torch.sin(Thetaij1)**2)


    deriv1 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))#((dij) / (2 * triArea))
    deriv2 = ((dij) / (2 * triArea))

    temp = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2

    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    flipped_triangles = torch.logical_not(flipped_triangles)
    dotmaxe1 = torch.tensor(dotmax1[k1, j1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[k1, j1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[k1, j1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[k1, j1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[k1, j1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[k1, j1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[k1, j1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[k1, j1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[j1, k1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (-torch.sign(dotmaxe1s) * derivative1)) + (-torch.sin(torch.acos(dotmaxe2)) * (-torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (-torch.sign(dotmine1s) * derivative1)) + (-torch.sin(torch.acos(dotmine2)) * (-torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA5 = (torch.sin(Thetaik1)*0.5 * (anisotropic2[tsl] * dOffDiagonal1 + anisotropic1[tsl] * dOffDiagonal2) - 0.5*(anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (dotmine2 * dotmine1)) * (((-djk1 * torch.cos(Thetaij1)) / (2 * triArea)) * torch.cos(Thetaik1)))/(torch.sin(Thetaik1)**2)

    deriv1 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    deriv2 = (-(djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    flipped_triangles = torch.logical_not(flipped_triangles)
    temp = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2
    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    dotmaxe1 = torch.tensor(dotmax1[i1, j1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[i1, j1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[i1, j1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[i1, j1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[i1, j1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[i1, j1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[i1, j1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[i1, j1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[j1, i1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (-torch.sign(dotmaxe1s) * derivative1)) + (-torch.sin(torch.acos(dotmaxe2)) * (-torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (-torch.sign(dotmine1s) * derivative1)) + (-torch.sin(torch.acos(dotmine2)) * (-torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA6 = -(torch.sin(Thetajk1)*0.5 * (anisotropic1[tsl] * dOffDiagonal1 + anisotropic2[tsl] * dOffDiagonal2) - 0.5*(anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (dotmine2 * dotmine1)) * (((-dij) / (2 * triArea)) * torch.cos(Thetajk1)))/(torch.sin(Thetajk1)**2)
    #fullDerivativeDiagonalA6
    return fullDerivativeDiagonalA1, fullDerivativeDiagonalA2, fullDerivativeDiagonalA3, fullDerivativeDiagonalA4, fullDerivativeDiagonalA5, fullDerivativeDiagonalA6




def anisotropic_lbo_gradient2(VPos, ITris, tsl, pRiemann, e, i1_index, j1_index, k1_index, maxCurvature, minCurvature, dij, dik1, djk1, Thetaij1, Thetaik1, Thetajk1, anisotropic1, anisotropic2, dotmin1, dotmin2, dotmax1, dotmax2, edge, k1, dotmin1s, dotmin2s, dotmax1s, dotmax2s, diagonalDotsMin1, diagonalDotsMin2, diagonalDotsMax1, diagonalDotsMax2, diagonalDotsMin1S, diagonalDotsMin2S, diagonalDotsMax1S, diagonalDotsMax2S):
    anisotropic1 = anisotropic1.cpu()
    anisotropic2 = anisotropic2.cpu()
    # Riemann Block Gradients
    # TODO cleanup and simplify expressions
    triArea = dij * djk1 * torch.sin(Thetaij1) * 0.5
    derivative1 = (djk1 * torch.cos(Thetaij1)) / (2 * triArea)
    derivative2 = (dik1 * torch.cos(Thetaik1)) / (2 * triArea)

    # TODO signs are swapped sometimes

    cotAlpha = 1 / torch.tan(Thetajk1)
    cotAlpha2 = 1 / torch.tan(Thetaij1)
    cotAlpha3 = 1 / torch.tan(Thetaik1)

    i1 = edge[:, 0]
    j1 = edge[:, 1]
    flipped_triangles = torch.logical_or((torch.logical_or((i1_index%2>j1_index), ((i1_index+1)<j1_index))), (i1_index-1==j1_index))

    #deriv1 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))#((dij) / (2 * triArea))
    deriv2 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))

    temp = ((dij) / (2 * triArea))
    derivative2 = deriv2
    derivative2[flipped_triangles] = temp[flipped_triangles]

    dotmaxe1 = diagonalDotsMax1[i1_index, tsl]
    dotmine1 = diagonalDotsMin1[i1_index, tsl]
    dotmaxe1s = diagonalDotsMax1S[i1_index, tsl]
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmine1s = diagonalDotsMin1S[i1_index, tsl]
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]

    fullDerivativeDiagonalA1 = -0.5 * (((anisotropic1[tsl] * (dotmaxe1) ** 2 + anisotropic2[tsl] * (dotmine1) ** 2)) * (dij / (2 * triArea) * 1 / torch.sin(Thetajk1) ** 2 + ((-((dik1 * torch.cos(Thetaik1)) / (2 * triArea))) * 1 / torch.sin(Thetaij1) ** 2)) + (anisotropic1[tsl] * 2 * torch.sin(torch.acos(dotmaxe1)) * (dotmaxe1) * torch.sign(dotmaxe1s) * derivative2 + anisotropic2[tsl] * 2 * torch.sin(torch.acos(dotmine1)) * dotmine1 * torch.sign(dotmine1s) * derivative2) * (cotAlpha + cotAlpha2))


    #todo fix
    dotmaxe2 = diagonalDotsMax1[j1_index, tsl]
    #dotmaxe2[flipped_triangles] = diagonalDotsMax1[j1_index, tsl][flipped_triangles]
    #dotmaxe2 = diagonalDotsMax1[j1_index, tsl]

    dotmine2 = diagonalDotsMin1[j1_index, tsl]
    #dotmine2[flipped_triangles] = diagonalDotsMin1[j1_index, tsl][flipped_triangles]
    #dotmine2 = diagonalDotsMin1[j1_index, tsl]
    dotmaxe2s = diagonalDotsMax1S[j1_index, tsl]
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2s[flipped_triangles] = diagonalDotsMax1S[j1_index, tsl][flipped_triangles]
    dotmine2s = diagonalDotsMin1S[j1_index, tsl]
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2s[flipped_triangles] = diagonalDotsMin1S[j1_index, tsl][flipped_triangles]
    derivative3 = (dij) / (2 * triArea)
    derivative3[flipped_triangles] = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))[flipped_triangles]

    fullDerivativeDiagonalA2 = -(0.5 * (((anisotropic1[tsl] * (dotmaxe2) ** 2 + anisotropic2[tsl] * (dotmine2) ** 2)) * (dij / (2 * triArea) * 1 / torch.sin(Thetajk1) ** 2 + (-((djk1 * torch.cos(Thetaij1)) / (2 * triArea)) * 1 / torch.sin(Thetaik1) ** 2)) + (anisotropic1[tsl] * 2 * torch.sin(torch.acos(dotmaxe2)) * (dotmaxe2) * -torch.sign(dotmaxe2s) * derivative3 + anisotropic2[tsl] * 2 * torch.sin(torch.acos(dotmine2)) * dotmine2 * -torch.sign(dotmine2s) * derivative3) * (cotAlpha + cotAlpha3)))

    dotmine1 = diagonalDotsMin1[k1_index, tsl]
    dotmine1s = diagonalDotsMin1S[k1_index, tsl]
    #dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmaxe1 = diagonalDotsMax1[k1_index, tsl]
    dotmaxe1s = diagonalDotsMax1S[k1_index, tsl]
    #dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    derivative1 = (djk1 * torch.cos(Thetaij1)) / (2 * triArea)
    derivative1[flipped_triangles] = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))[flipped_triangles]


    fullDerivativeDiagonalA3 = (0.5 * (
            ((anisotropic1[tsl] * (dotmaxe1) ** 2 + anisotropic2[tsl] * (dotmine1) ** 2)) * (
            (djk1 * torch.cos(Thetaij1)) / (2 * triArea) * 1 / torch.sin(Thetaik1) ** 2 + (
            (((dik1 * torch.cos(Thetaik1)) / (2 * triArea))) * 1 / torch.sin(Thetaij1) ** 2)) + (
                    anisotropic1[tsl] * 2 * torch.sin(torch.acos(dotmaxe1)) * (
                dotmaxe1) * torch.sign(dotmaxe1s) * derivative1 + anisotropic2[tsl] * 2 * torch.sin(
                torch.acos(dotmine1)) * dotmine1 * torch.sign(dotmine1s) * derivative1) * (cotAlpha2 + cotAlpha3)))


    deriv1 = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    deriv2 = ((dij) / (2 * triArea))

    temp = ((djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2

    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    dotmaxe1 = torch.tensor(dotmax1[k1, i1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[k1, i1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[k1, i1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[k1, i1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[k1, i1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[k1, i1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[i1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[k1, i1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[k1, i1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[i1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[i1, k1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (torch.sign(dotmaxe1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmaxe2)) * (torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (torch.sign(dotmine1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmine2)) * (torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA4 = (torch.sin(Thetaij1) * 0.5 * (
                anisotropic1[tsl] * dOffDiagonal1 + anisotropic2[tsl] * dOffDiagonal2) - 0.5 * (
                                            anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (
                                                dotmine2 * dotmine1)) * (
                                            ((-dik1 * torch.cos(Thetaik1)) / (2 * triArea)) * torch.cos(Thetaij1))) / (
                                           torch.sin(Thetaij1) ** 2)

    deriv1 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    deriv2 = ((dij) / (2 * triArea))

    temp = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2

    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    flipped_triangles = torch.logical_not(flipped_triangles)
    dotmaxe1 = torch.tensor(dotmax1[k1, j1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[k1, j1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[k1, j1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[k1, j1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[k1, j1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[k1, j1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[j1, k1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[k1, j1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[k1, j1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[j1, k1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[j1, k1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (-torch.sign(dotmaxe1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmaxe2)) * (-torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (-torch.sign(dotmine1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmine2)) * (-torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA5 = (torch.sin(Thetaik1) * 0.5 * (
                anisotropic2[tsl] * dOffDiagonal1 + anisotropic1[tsl] * dOffDiagonal2) - 0.5 * (
                                            anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (
                                                dotmine2 * dotmine1)) * (
                                            ((-djk1 * torch.cos(Thetaij1)) / (2 * triArea)) * torch.cos(Thetaik1))) / (
                                           torch.sin(Thetaik1) ** 2)

    deriv1 = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    deriv2 = (-(djk1 * torch.cos(Thetaij1)) / (2 * triArea))
    flipped_triangles = torch.logical_not(flipped_triangles)
    temp = ((dik1 * torch.cos(Thetaik1)) / (2 * triArea))
    derivative1 = deriv1
    derivative2 = deriv2
    derivative1[flipped_triangles] = deriv2[flipped_triangles]
    derivative2[flipped_triangles] = temp[flipped_triangles]

    dotmaxe1 = torch.tensor(dotmax1[i1, j1]).squeeze(0)
    dotmaxe1s = torch.tensor(dotmax1s[i1, j1]).squeeze(0)
    dotmaxe1s[torch.logical_not(flipped_triangles)] = -dotmaxe1s[torch.logical_not(flipped_triangles)]
    dotmaxe1[flipped_triangles] = torch.tensor(dotmax1[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe1s[flipped_triangles] = torch.tensor(dotmax1s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine1 = torch.tensor(dotmin1[i1, j1]).squeeze(0)
    dotmine1s = torch.tensor(dotmin1s[i1, j1]).squeeze(0)
    dotmine1s[torch.logical_not(flipped_triangles)] = -dotmine1s[torch.logical_not(flipped_triangles)]
    dotmine1[flipped_triangles] = torch.tensor(dotmin1[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine1s[flipped_triangles] = torch.tensor(dotmin1s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine2s = torch.tensor(dotmin2s[i1, j1]).squeeze(0)
    dotmine2s[torch.logical_not(flipped_triangles)] = -dotmine2s[torch.logical_not(flipped_triangles)]
    dotmine2 = torch.tensor(dotmin2[i1, j1]).squeeze(0)
    dotmine2[flipped_triangles] = torch.tensor(dotmin2[j1, i1]).squeeze(0)[flipped_triangles]
    dotmine2s[flipped_triangles] = torch.tensor(dotmin2s[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe2 = torch.tensor(dotmax2[i1, j1]).squeeze(0)
    dotmaxe2s = torch.tensor(dotmax2s[i1, j1]).squeeze(0)
    dotmaxe2s[torch.logical_not(flipped_triangles)] = -dotmaxe2s[torch.logical_not(flipped_triangles)]
    dotmaxe2[flipped_triangles] = torch.tensor(dotmax2[j1, i1]).squeeze(0)[flipped_triangles]
    dotmaxe2s[flipped_triangles] = torch.tensor(dotmax2s[j1, i1]).squeeze(0)[flipped_triangles]

    dOffDiagonal1 = dotmaxe2 * (-torch.sin(torch.acos(dotmaxe1)) * (-torch.sign(dotmaxe1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmaxe2)) * (-torch.sign(dotmaxe2s) * derivative2)) * dotmaxe1
    dOffDiagonal2 = dotmine2 * (-torch.sin(torch.acos(dotmine1)) * (-torch.sign(dotmine1s) * derivative1)) + (
                -torch.sin(torch.acos(dotmine2)) * (-torch.sign(dotmine2s) * derivative2)) * dotmine1
    fullDerivativeDiagonalA6 = -(torch.sin(Thetajk1) * 0.5 * (
                anisotropic1[tsl] * dOffDiagonal1 + anisotropic2[tsl] * dOffDiagonal2) - 0.5 * (
                                             anisotropic1[tsl] * (dotmaxe2 * dotmaxe1) + anisotropic2[tsl] * (
                                                 dotmine2 * dotmine1)) * (
                                             ((-dij) / (2 * triArea)) * torch.cos(Thetajk1))) / (
                                           torch.sin(Thetajk1) ** 2)

    return fullDerivativeDiagonalA1, fullDerivativeDiagonalA2, fullDerivativeDiagonalA3, fullDerivativeDiagonalA4, fullDerivativeDiagonalA5, fullDerivativeDiagonalA6


def get_anisotropic_laplacian(VPos, ITris, Riemannian, edges, minCurvature, maxCurvature, rotationNormal, anisotropy1, anisotropy2, Theta, corners, ts, voronoi):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    L = maxCurvature.shape[1]
    fOffset = (maxCurvature.shape[1] - M)
    maxCurvature = maxCurvature[:, (maxCurvature.shape[1] - M):].cpu()
    minCurvature = minCurvature[:, (minCurvature.shape[1] - M):].cpu()
    Theta = Theta[(Theta.shape[0] - M):].cpu()
    #Theta[10] += 0.001
    rotationNormal = rotationNormal[:, (rotationNormal.shape[1] - M):].transpose(1,0).cpu()

    # rotate prinicpal curvatures by Theta
    # compute rotation matrix
    rot = (torch.cos(Theta)[:, None] * torch.eye(3)[:, None]).transpose(0, 1) + (
                torch.sin(Theta)[None, :] * torch.stack((torch.zeros(rotationNormal.shape[0]), -rotationNormal[:, 2],
                                                          rotationNormal[:, 1], rotationNormal[:, 2],
                                                          torch.zeros(rotationNormal.shape[0]), -rotationNormal[:, 0],
                                                          -rotationNormal[:, 1], rotationNormal[:, 0],
                                                          torch.zeros(rotationNormal.shape[0])))).transpose(1, 0).reshape(
        rotationNormal.shape[0], 3, 3) + ((1 - np.cos(Theta))[:, None, None] * (
                rotationNormal[:, :, None] @ rotationNormal[:, None, :]))


    # rotate principal curvatures
    minCurvatureR = (minCurvature.transpose(1,0).unsqueeze(1) @ rot)[:,0,:].transpose(1,0)
    maxCurvatureR = (maxCurvature.transpose(1,0).unsqueeze(1) @ rot)[:,0,:].transpose(1,0)


    I = np.zeros(M * 9)
    J = np.zeros(M * 9)
    V = np.zeros(M * 9)
    rangle = torch.zeros(M, 3, dtype=torch.float64)
    e = torch.zeros(M, 3, dtype=torch.float64)
    z = torch.zeros(M, 3, dtype=torch.float64)
    el = torch.zeros(M, 3, dtype=torch.float64)

    ofD1 = torch.zeros(3, L, dtype=torch.float64)
    ofD2 = torch.zeros(3, L, dtype=torch.float64)
    ofDR = torch.zeros(3, L, dtype=torch.float64)

    alpha2A = torch.zeros(3, L - fOffset, dtype=torch.float64)
    alpha3A = torch.zeros(3, L - fOffset, dtype=torch.float64)
    diagonalDotsMax1 = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMax2 = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMin1 = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMin2 = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMax1S = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMax2S = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMin1S = torch.zeros(3, L-fOffset, dtype=torch.float64)
    diagonalDotsMin2S = torch.zeros(3, L-fOffset, dtype=torch.float64)
    dmax = torch.zeros(3, L, dtype=torch.float64)
    dmin = torch.zeros(3, L, dtype=torch.float64)
    drot = torch.zeros(3, L, dtype=torch.float64)


    pRiemann = sparse.coo_matrix((Riemannian.tolist(), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N))
    pDotmax1 = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmax1R = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmax1Sign = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmax2 = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmax2R = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmax2Sign = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin1 = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin1R = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin1Sign = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin2 = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin2R = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()
    pDotmin2Sign = sparse.coo_matrix((torch.zeros(edges[:, 0].shape), (edges[:, 0], edges[:, 1])),
                                 shape=(N, N), dtype=np.float64).todense()

    pRiemann = torch.tensor((pRiemann + pRiemann.transpose()).todense()).squeeze(0)


    tsl = ts[:, 0].cpu()
    tsr = ts[:, 1].cpu()
    k1_index = corners[:, 0].cpu()
    k2_index = corners[:, 1].cpu()
    i = edges[:, 0]
    j = edges[:, 1]
    k1 = torch.diag(ITris[tsl][:, k1_index])  # ITris[ts[0]][corners[0]]
    k2 = torch.diag(ITris[tsr][:, k2_index])  # ITris[ts[1]][corners[1]]
    i1_index = torch.max((ITris[tsl] == i.unsqueeze(1)), 1)[1]
    j1_index = torch.max((ITris[tsl] == j.unsqueeze(1)), 1)[1]
    i2_index = torch.max((ITris[tsr] == i.unsqueeze(1)), 1)[1]
    j2_index = torch.max((ITris[tsr] == j.unsqueeze(1)), 1)[1]
    if (len(tsl) != len(k1_index)) or (len(tsl) != len(i1_index)) or (
            len(tsl) != len(j1_index)) or (len(tsr) != len(i2_index)) or (
            len(tsr) != len(j2_index)):
        print("detected non-manifold, skipping")
        nograd = True

    summation = 1
    alpha = 0.001

    # first pass
    for rotation in range(3):
        [i, j, k] = [rotation, (rotation + 1) % 3, (rotation + 2) % 3]

        lij = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, j]])
        lik = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, k]])
        ljk = torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, j], ITris[:, k]])

        a = torch.cat((ITris[lij < alpha, i], ITris[lik < alpha, i], ITris[ljk < alpha, j])).cpu().numpy()
        #_, idx = np.unique(a, return_index=True)
        #a = (a[np.sort(idx)])

        b = torch.cat((ITris[lij < alpha, j], ITris[lik < alpha, k], ITris[ljk < alpha, k])).cpu().numpy()
        #_, idx = np.unique(b, return_index=True)
        #b = (b[np.sort(idx)])
        if len(a)>0:
            idx = torch.unique(torch.stack((torch.tensor(a),torch.tensor(b))), dim=0)
            pRiemann[idx[0,:], idx[1,:]] += alpha - (torch.sqrt(torch.sum(((VPos[idx[0,:], :]) - (VPos[idx[1,:], :])) ** 2, 1)) + pRiemann[idx[0,:], idx[1,:]])
            pRiemann[idx[1,:], idx[0,:]] = pRiemann[idx[0,:], idx[1,:]]

    # second pass
    beta = 0.0001
    while summation>=0.0001:
        summation = 0        # correct weightings to obey triangular inequality
        for rotation in range(3):
            [i, j, k] = [rotation, (rotation + 1) % 3, (rotation + 2) % 3]

            lij = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, j]])
            lik = torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, i], ITris[:, k]])
            ljk = torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
                ITris[:, j], ITris[:, k]])

            b = (lik + ljk - lij)
            #b[b < 0] = torch.clip(b[b < 0], max=-beta)
            b[torch.where((b>0) & (b<beta))] = torch.clip(b[torch.where((b>0) & (b<beta))], max=-beta)

            mu = (1/3) * (e[:, i]-e[:, k]-e[:, j]-b)
            theta = torch.min(torch.stack((-mu, z[:, rotation])).transpose(1,0), dim=1)[0]

            e[:, k] -= theta
            e[:, j] -= theta
            e[:, i] += theta
            z[:, rotation] -= theta
            summation = summation + torch.sum(torch.abs(theta))

    e[torch.where((e>0) & (e<beta))] = torch.clip(e[torch.where((e>0) & (e<beta))], min=beta)
    e[torch.where((e < 0) & (e > -beta))] = torch.clip(e[torch.where((e < 0) & (e > -beta))], max=-beta)
    e *= 1.2
    #print(e[torch.where((e > 0))])
    anisotropic1 = anisotropy1[(anisotropy1.shape[0] - M):]
    anisotropic2 = anisotropy2[(anisotropy2.shape[0] - M):]

    #print("Triangle Fixes", len(e[e>0]))
    for shift in range(3):
        # For all 3 shifts of the roles of triangle vertices
        # to compute different cotangent weights
        [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]

        lij = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, j]]) + e[:, i]))
        lik = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, i], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, i], ITris[:, k]])) + e[:, k])
        ljk = (torch.abs(torch.sqrt(torch.sum(((VPos[ITris[:, j], :]) - (VPos[ITris[:, k], :])) ** 2, 1)) + (pRiemann[
            ITris[:, j], ITris[:, k]])) + e[:, j])

        alpha2 = torch.acos((lik ** 2 + ljk ** 2 - lij ** 2) / (2 * lik * ljk))
        alpha3 = torch.acos((ljk ** 2 + lij ** 2 - lik ** 2) / (2 * ljk * lij))
        rangle[:, k] = alpha2
        el[:, k] = (lij)
        #
        riemannFactorij = (torch.norm(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :])), dim=1) + (pRiemann[ITris[:, i], ITris[:, j]] + e[:, i]))/torch.norm((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :]), dim=1)

        Np = np.cross((VPos[ITris[:, k], :]) - (VPos[ITris[:, i], :]), (VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :]))
        Np = Np / np.sqrt(np.sum(Np ** 2, 1))[:, None]

        rot = (torch.cos(alpha3)[:, None] * torch.eye(3)[:, None]).transpose(0,1) + (torch.sin(alpha3)[None, :] * torch.stack((torch.zeros(Np.shape[0]), torch.tensor(-Np[:,2]), torch.tensor(Np[:,1]), torch.tensor(Np[:,2]), torch.zeros(Np.shape[0]), torch.tensor(-Np[:, 0]), torch.tensor(-Np[:,1]), torch.tensor(Np[:,0]), torch.zeros(Np.shape[0])))).transpose(1,0).reshape(Np.shape[0], 3, 3) + ((1-np.cos(alpha3))[:, None, None]*(torch.tensor(Np)[:,:, None] @ torch.tensor(Np)[:, None, :]))
        nor = (torch.norm(((VPos[ITris[:, i], :]) - (VPos[ITris[:, j], :]))*riemannFactorij[:, None], dim=1))


        pointk = (((((VPos[ITris[:, i], :])*riemannFactorij[:, None] - (VPos[ITris[:, j], :])*riemannFactorij[:, None]) / nor[:, None])[:, None] @ rot).squeeze(1) * (ljk[:, None]))+(VPos[ITris[:, j], :])*riemannFactorij[:, None]

        #dotmaxe1 = (((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1,0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * maxCurvature.transpose(1,0)).sum(dim=-1)
        dotmaxe1R = torch.clip((((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1,0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * maxCurvatureR.transpose(1,0)).sum(dim=-1), min=-1, max=1)
        #dotmaxe2 = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * maxCurvature.transpose(1, 0)).sum(dim=-1)
        dotmaxe2R = torch.clip(((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * maxCurvatureR.transpose(1, 0)).sum(dim=-1), min=-1, max=1)
        #dotmine1 = (((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * minCurvature.transpose(1, 0)).sum(dim=-1)
        dotmine1R = torch.clip((((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * minCurvatureR.transpose(1, 0)).sum(dim=-1), min=-1, max=1)
        #dotmine2 = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * minCurvature.transpose(1, 0)).sum(dim=-1)
        dotmine2R = torch.clip(((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * minCurvatureR.transpose(1, 0)).sum(dim=-1), min=-1, max=1)

        pDotmax1[ITris[:, i], ITris[:, j]] = dotmaxe1R
        pDotmax1R[ITris[:, i], ITris[:, j]] = dotmaxe1R
        pDotmax1Sign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)))).transpose(1,0),  maxCurvatureR.transpose(1, 0))).sum(dim=-1)
        #pDotmax1RSign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)))).transpose(1,0),  maxCurvatureR.transpose(1, 0))).sum(dim=-1)

        diagonalDotsMax1[shift, :] = dotmaxe1R
        diagonalDotsMax1S[shift, :] = torch.tensor(pDotmax1Sign[ITris[:, i], ITris[:, j]])

        pDotmax2[ITris[:, i], ITris[:, j]] = dotmaxe2R
        pDotmax2R[ITris[:, i], ITris[:, j]] = dotmaxe2R
        pDotmax2Sign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1,0),  maxCurvatureR.transpose(1, 0))).sum(dim=-1)
        #pDotmax2RSign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1,0),  maxCurvatureR.transpose(1, 0))).sum(dim=-1)
        diagonalDotsMax2[shift, :] = dotmaxe2R
        diagonalDotsMax2S[shift, :] = torch.tensor(pDotmax2Sign[ITris[:, i], ITris[:, j]])
        pDotmin1[ITris[:, i], ITris[:, j]] = dotmine1R
        pDotmin1R[ITris[:, i], ITris[:, j]] = dotmine1R
        pDotmin1Sign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)))).transpose(1,0),  minCurvatureR.transpose(1, 0))).sum(dim=-1)
        #pDotmin1RSign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)))).transpose(1,0),  minCurvatureR.transpose(1, 0))).sum(dim=-1)
        diagonalDotsMin1[shift, :] = dotmine1R
        diagonalDotsMin1S[shift, :] = torch.tensor(pDotmin1Sign[ITris[:, i], ITris[:, j]])
        pDotmin2[ITris[:, i], ITris[:, j]] = dotmine2R
        pDotmin2R[ITris[:, i], ITris[:, j]] = dotmine2R
        pDotmin2Sign[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1,0),  minCurvatureR.transpose(1, 0))).sum(dim=-1)
        #pDotmin2SignR[ITris[:, i], ITris[:, j]] = (torch.tensor(Np) * torch.cross((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1,0),  minCurvatureR.transpose(1, 0))).sum(dim=-1)
        diagonalDotsMin2[shift, :] = dotmine2R
        diagonalDotsMin2S[shift, :] = torch.tensor(pDotmin2Sign[ITris[:, i], ITris[:, j]])
        alpha2A[shift, :] = alpha2
        alpha3A[shift, :] = alpha3

    for shift in range(3):
        # For all 3 shifts of the roles of triangle vertices
        # to compute different cotangent weights
        [i, j, k] = [shift, (shift + 1) % 3, (shift + 2) % 3]


        dotmaxe1 = torch.tensor(pDotmax1[ITris[:, i], ITris[:, j]])#diagonalDotsMax1[shift, :]
        diagonalDotsMax1[shift, :] = dotmaxe1
        dotmaxe2 = torch.tensor(pDotmax2[ITris[:, i], ITris[:, j]])#diagonalDotsMax2[shift, :]
        diagonalDotsMax2[shift, :] = dotmaxe2
        dotmine1 = torch.tensor(pDotmin1[ITris[:, i], ITris[:, j]])#diagonalDotsMin1[shift, :]
        diagonalDotsMin1[shift, :] = dotmine1
        dotmine2 = torch.tensor(pDotmin2[ITris[:, i], ITris[:, j]])#diagonalDotsMin1[shift, :]
        diagonalDotsMin2[shift, :] = dotmine2
        alpha2 = alpha2A[shift, :]
        alpha3 = alpha3A[shift, :]

        #print(dotmaxe1)
        offDiagonal1 = (dotmaxe1 * dotmaxe2) #anisotropic1.cpu()anisotropic2.cpu()
        offDiagonal2 = (dotmine1 * dotmine2)
        offDiagonal = anisotropic1.cpu() * offDiagonal1 + anisotropic2.cpu() * offDiagonal2

        cotAlpha = 1 / torch.tan(alpha2)
        cotAlpha2 = 1 / torch.tan(alpha3)

        ofD1[shift, fOffset:] = 0.5 * offDiagonal1/torch.sin(alpha2)
        ofD2[shift, fOffset:] = 0.5 * offDiagonal2/torch.sin(alpha2)

        partMax =  -(torch.sign(torch.tensor(pDotmax1Sign[ITris[:, i], ITris[:, j]])) * (torch.sign(torch.tensor(pDotmax2Sign[ITris[:, i], ITris[:, j]])))) * (anisotropic1.cpu() * (torch.sin(torch.acos(dotmaxe2)) * (torch.sign(torch.tensor(pDotmax1Sign[ITris[:, i], ITris[:, j]]))) * dotmaxe1 + (torch.sign(torch.tensor(pDotmax2Sign[ITris[:, i], ITris[:, j]]))) * dotmaxe2 * torch.sin(torch.acos(dotmaxe1))))
        partMin = -(torch.sign(torch.tensor(pDotmin1Sign[ITris[:, i], ITris[:, j]])) * (torch.sign(torch.tensor(pDotmin2Sign[ITris[:, i], ITris[:, j]]))))*(anisotropic2.cpu() * (torch.sin(torch.acos(dotmine2)) * (torch.sign(torch.tensor(pDotmin1Sign[ITris[:, i], ITris[:, j]]))) * dotmine1 + (torch.sign(torch.tensor(pDotmin2Sign[ITris[:, i], ITris[:, j]]))) * dotmine2 * torch.sin(torch.acos(dotmine1))))
        ofDR[shift, fOffset:] = -0.5 * (partMax + partMin) * (1/torch.sin(alpha2))


        dmax[shift, fOffset:] = 0.5 * (dotmaxe1)**2 * (cotAlpha + cotAlpha2)
        dmin[shift, fOffset:] = 0.5 * (dotmine1)**2 * (cotAlpha + cotAlpha2)
        drot[shift, fOffset:] = -(cotAlpha + cotAlpha2) * (anisotropic1.cpu() * torch.sin(torch.acos(dotmaxe1)) * -(torch.sign(torch.tensor(pDotmax1Sign[ITris[:, i], ITris[:, j]]))) * dotmaxe1 + anisotropic2.cpu() * torch.sin(torch.acos(dotmine1)) * -(torch.sign(torch.tensor(pDotmin1Sign[ITris[:, i], ITris[:, j]]))) * dotmine1)

        I[shift * M * 3:shift * M * 3 + M] = ITris[:, i]
        J[shift * M * 3:shift * M * 3 + M] = ITris[:, j]
        V[shift * M * 3:shift * M * 3 + M] = 0.5 * offDiagonal/torch.sin(alpha2)

        I[shift * M * 3 + M:shift * M * 3 + 2 * M] = ITris[:, j]
        J[shift * M * 3 + M:shift * M * 3 + 2 * M] = ITris[:, i]
        V[shift * M * 3 + M:shift * M * 3 + 2 * M] = 0.5 * offDiagonal/torch.sin(alpha2)

        I[shift * M * 3 + 2*M:shift * M * 3 + 3 * M] = ITris[:, i]
        J[shift * M * 3 + 2*M:shift * M * 3 + 3 * M] = ITris[:, i]
        V[shift * M * 3 + 2*M:shift * M * 3 + 3 * M] = 0.5 * ((anisotropic1.cpu() * (dotmaxe1)**2 + anisotropic2.cpu()  * (dotmine1)**2)) * (cotAlpha + cotAlpha2)

    L = sparse.coo_matrix((V, (I, J)), shape=(N, N), dtype=np.float64)

    nograd = False
    gV, gL = 0, 0
    if not nograd:
        dij = el[tsl, k1_index]
        dij2 = el[tsr, k2_index]
        djk1 = el[tsl, i1_index]
        dik1 = el[tsl, j1_index]
        dik2 = el[tsr, i2_index]
        djk2 = el[tsr, j2_index]

        Thetaij1 = rangle[tsl, j1_index]
        Thetaik1 = rangle[tsl, i1_index]
        Thetajk1 = rangle[tsl, k1_index]

        Thetaij2 = rangle[tsr, j2_index]
        Thetaik2 = rangle[tsr, k2_index]
        Thetajk2 = rangle[tsr, i2_index]

        A1, A2, A3, A4, A5, A6 = anisotropic_lbo_gradient(VPos, ITris, tsl, pRiemann, e, i1_index, j1_index, k1_index, maxCurvature,
                                 minCurvature, dij, dik1, djk1, Thetaij1, Thetaik1, Thetajk1, anisotropic1,
                                 anisotropic2, pDotmin1, pDotmin2, pDotmax1, pDotmax2, edges, k1, pDotmin1Sign, pDotmin2Sign, pDotmax1Sign, pDotmax2Sign, diagonalDotsMin1, diagonalDotsMin2, diagonalDotsMax1, diagonalDotsMax2, diagonalDotsMin1S, diagonalDotsMin2S, diagonalDotsMax1S, diagonalDotsMax2S)

        B1, B2, B3, B4, B5, B6 = anisotropic_lbo_gradient2(VPos, ITris, tsr, pRiemann, e, i2_index, j2_index, k2_index, maxCurvature,  minCurvature, dij2, djk2, dik2, Thetaij2, Thetajk2, Thetaik2, anisotropic1, anisotropic2, pDotmin1, pDotmin2, pDotmax1, pDotmax2, edges, k2, pDotmin1Sign, pDotmin2Sign, pDotmax1Sign, pDotmax2Sign, diagonalDotsMin1, diagonalDotsMin2, diagonalDotsMax1, diagonalDotsMax2, diagonalDotsMin1S, diagonalDotsMin2S, diagonalDotsMax1S, diagonalDotsMax2S)

        ALBO = True
        if ALBO:
            Lii = A1 + B1
            Ljj = A2 + B2
            Lij = A6 + B6
            Lk2k2 = B3
            Lk1k1 = A3

            Lik2 = B4
            Ljk2 = B5

            Lik1 = A4
            Ljk1 = A5
        else:
            Lii = (laplacianPart(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij) + laplacianPart2(Thetaij2, Thetaik2,Thetajk2, djk2, dij,dik2))#
            Ljj = (laplacianPart(Thetaik1, Thetaij1, Thetajk1, djk1, dik1, dij) + laplacianPart2(Thetajk2, Thetaik2,Thetaij2, dik2, dij, djk2))#
            Lij = (laplacianPartij(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij) + laplacianPart2ij(Thetaij2, Thetaik2,Thetajk2, djk2,dij, dik2))#
            Lk2k2 = (laplacianPartk2(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2))
            Lk1k1 = (laplacianPartk1(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij))

            Lik2 = (laplacianPartk2A(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2))
            Ljk2 = (laplacianPartk2B(Thetaij2, Thetaik2, Thetajk2, djk2, dij, dik2))

            Lik1 = (laplacianPartk1A(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij))
            Ljk1 = (laplacianPartk1B(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij))


        gL = torch.column_stack((Lii, Ljj, Lij, Lij, Lk2k2, Lk1k1, Lik2, Lik2, Ljk2, Ljk2, Lik1, Lik1, Ljk1, Ljk1))

        Vii = (((voronoiPartA(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij)) + (
            voronoiPartB(Thetaij2, Thetaik2, Thetajk2, djk2, dij2, dik2)))) * voronoi[0,ITris[tsl, i1_index]].cpu()
        Vjj = ((voronoiPartA(Thetaik1, Thetaij1, Thetajk1, djk1, dik1, dij)) + (
            voronoiPartB(Thetajk2, Thetaik2, Thetaij2, dik2, dij2, djk2))) * voronoi[0,ITris[tsl, j1_index]].cpu()
        Vk1k1 = voronoiPartImpact1(Thetaij1, Thetaik1, Thetajk1, dik1, djk1, dij) * voronoi[0,ITris[tsl, k1_index]].cpu()
        Vk2k2 = voronoiPartImpact2(Thetaij2, Thetaik2, Thetajk2, djk2, dij2, dik2) * voronoi[0,ITris[tsr, k2_index]].cpu()

        gV = torch.column_stack((Vii, Vjj, Vk1k1, Vk2k2))

    Au = torch.tensor(igl.massmatrix_intrinsic(el.cpu().numpy(), ITris.numpy(), igl.MASSMATRIX_TYPE_VORONOI).diagonal())

    A = (Au * voronoi.squeeze(0).cpu().numpy()).numpy()
    return A, L, rangle, el, ofD1[:, fOffset:], ofD2[:, fOffset:], ofDR[:, fOffset:], dmax[:, fOffset:], dmin[:, fOffset:], drot[:, fOffset:], gV, gL, Au