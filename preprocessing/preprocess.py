import os
import pickle
from multiprocessing import Pool
import pyvista as pv
import igl
import numpy as np
import point_cloud_utils as pcu
import torch
import trimesh

from LBONet import LBOSingle
from RiemannHelpers import get_curvature2
from RiemannNetRetrievalShapeNet import Minmesh


def iter_obj_paths(base_dir):
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()  # sort subdirectories in-placereverse=True
        files.sort()  # optional: also sort files
        if any(f.endswith('.obj') for f in files):
            target_filename = next((f for f in files if f.endswith('.obj')), None)
            parts = root.split(os.sep)
            yield os.path.join(root, target_filename)

def get_iter_obj_paths():
    pickle_file = 'iter_obj_pathsR.pkl'

    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        print("Mesh data is existing.")
        # Load the paths from the pickle file
        with open(pickle_file, 'rb') as f:
            paths = pickle.load(f)
    else:

        print("Building directory from mesh data...")
        # Run the function to generate the paths if pickle file doesn't exist
        paths = list(iter_obj_paths(r"E:\\ShapeNet"))

        # Save the paths to the pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(paths, f)
        print("Directory built.", len(paths))
    return paths
def process_mesh(pars, maxfaces=3000, resolution = 5000):
    try:
        i, file, device, modelSingle = pars
        parts = file.split(os.sep)
        category_id = parts[-4]
        model_id = parts[-3]
        filename = category_id + "_" + model_id
        vertices_count = 1010
        edge_count = 3020
        faces_count = 2010


        print(f"Processing {i}: {file}")
        v, f = igl.read_triangle_mesh(file, dtypef="float32")

        vOLD, fOLD = pcu.make_mesh_watertight(v, f, resolution*2, seed=5555)
        vOLD, fOLD = pcu.make_mesh_watertight(vOLD, fOLD, resolution, seed=5555)
        mesh = pv.PolyData(vOLD, np.c_[(np.ones(fOLD.shape[0], dtype=np.int32) * 3), fOLD])
        mesh = mesh.smooth(n_iter=10, relaxation_factor=0.01)
        v = mesh.points.copy()
        f = mesh.faces.reshape(-1, 4)[:, 1:].copy()

        print(len(f))
        if len(f)>maxfaces:
            _, v, f, decimated_f_idxs, decimated_i_idxs = igl.decimate(v, f, maxfaces)


        mesh = pv.PolyData(v, np.c_[(np.ones(f.shape[0], dtype=np.int32) * 3), f])
        mesh = mesh.smooth(n_iter=10, relaxation_factor=0.01)
        v = mesh.points.copy()
        f = mesh.faces.reshape(-1, 4)[:, 1:].copy()

        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        components = mesh.split(only_watertight=False)
        largest = trimesh.util.concatenate(components[0])
        v = largest.vertices
        f = largest.faces

        scale_factor = np.linalg.norm(v, axis=1).max()
        v = v / scale_factor
        centroid = v.mean(axis=0)
        v = v - centroid
        n = igl.per_vertex_normals(v, f)
        n[np.isnan(n)] = 0
        mesh = Minmesh(v, f)
        aleph, k1, k2, y, alephP, k1P, k2P, yP, edge, p1, p2, curvature2f, gaussianf, k1f, k2f, rn = get_curvature2(
            mesh, file, rewrite=True, plot=False, mina=5, maxa=95)

        y = np.pad(y, [(edge_count - y.shape[0], 0)])
        ec = (np.sum(v[edge], 1) / 2)
        ec = np.pad(ec, [(edge_count - k2.shape[0], 0), (0, 0)])
        aleph = np.pad(aleph, [(edge_count - aleph.shape[0], 0)])
        k1 = np.pad(k1, [(edge_count - k1.shape[0], 0)])
        k2 = np.pad(k2, [(edge_count - k2.shape[0], 0)])

        minCurvature1 = torch.cat([torch.zeros(3, faces_count - p1.shape[0]), (torch.tensor(p1).transpose(1, 0) / torch.norm(torch.tensor(p1), dim=1))], dim=1)
        maxCurvature1 = torch.cat([torch.zeros(3, faces_count - p2.shape[0]), (torch.tensor(p2).transpose(1, 0) / torch.norm(torch.tensor(p2), dim=1))], dim=1)
        rotatingNormal1 = torch.cat([torch.zeros(3, faces_count - p2.shape[0]),(torch.tensor(rn).transpose(1, 0) / torch.norm(torch.tensor(rn), dim=1))], dim=1)
        voronoi = np.diag(igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI).toarray())
        voronoi = np.pad(voronoi, [(vertices_count - n.shape[0], 0)])
        yP = np.pad(yP, [(vertices_count - yP.shape[0], 0)])
        alephP = np.pad(alephP, [(vertices_count - alephP.shape[0], 0)])
        k1P = np.pad(k1P, [(vertices_count - k1P.shape[0], 0)])
        k2P = np.pad(k2P, [(vertices_count - k2P.shape[0], 0)])
        n = np.pad(n, [(vertices_count - n.shape[0], 0), (0, 0)])

        el1 = torch.norm(torch.tensor(np.pad(mesh.vertices[edge[:, 0]] - mesh.vertices[edge[:, 1]], [
            (edge_count - (mesh.vertices[edge[:, 0]] - mesh.vertices[edge[:, 1]]).shape[0], 0)])), dim=1)

        feature_vector1 = torch.tensor(np.stack(
            (aleph, y, k1, k2, ec[:, 0], ec[:, 1], ec[:, 2])))

        fc = (np.sum(v[f], 1) / 3)
        fc = np.pad(fc, [(faces_count - k1f.shape[0], 0), (0, 0)])
        k1f = np.pad(k1f, [(faces_count - k1f.shape[0], 0)])
        k2f = np.pad(k2f, [(faces_count - k2f.shape[0], 0)])
        gaussianf = np.pad(gaussianf, [(faces_count - gaussianf.shape[0], 0)])
        curvature2f = np.pad(curvature2f, [(faces_count - curvature2f.shape[0], 0)])


        feature_vector1f = torch.tensor(np.stack(
            (curvature2f, gaussianf, k1f, k2f, fc[:, 0], fc[:, 1], fc[:, 2])))

        feature_vector1P = torch.tensor(np.stack(
            (alephP, yP, k1P, k2P, voronoi, n[:, 0], n[:, 1], n[:, 2])))

        edgesL = edge
        edgeOffset = edge_count - edgesL.shape[0]
        edgesL = np.pad(edgesL, [(edgeOffset, 0), (0, 0)])

        e1 = torch.tensor(edgesL).transpose(1, 0)
        faceOffset = faces_count - mesh.faces.shape[0]
        vertexOffset = vertices_count - mesh.vertices.shape[0]
        f1 = torch.tensor(np.pad(mesh.faces, [(faceOffset, 0), (0, 0)])).transpose(1, 0)
        v1 = torch.tensor(np.pad(mesh.vertices, [(vertexOffset, 0), (0, 0)])).transpose(1, 0)
        flaps = igl.edge_flaps(mesh.faces)
        ts1 = torch.tensor(flaps[2])
        corners1 = torch.tensor(flaps[3])
        hks, _, _, _, broken = modelSingle(v1.unsqueeze(0), f1.unsqueeze(0), e1.unsqueeze(0),
                                   feature_vector1.unsqueeze(0).to(device),
                                   feature_vector1P.unsqueeze(0).to(device),
                                   feature_vector1f.unsqueeze(0).to(device), el1.unsqueeze(0).to(device),
                                   ts1[edgeOffset:].unsqueeze(0), corners1[edgeOffset:].unsqueeze(0), minCurvature1.unsqueeze(0),
                                   maxCurvature1.unsqueeze(0), rotatingNormal1.unsqueeze(0))
        hks1 = hks.transpose(1, 0)
        facehks = igl.average_onto_faces(mesh.faces, hks1[:, faceOffset + 1:].transpose(1, 0).cpu().numpy()).transpose(1,0).T
        feature_vector1f = torch.cat((feature_vector1f, torch.tensor(np.pad(facehks, [(faces_count - facehks.shape[0], 0), (0, 0)])).transpose(1, 0)), dim=0)
        if broken:
            return None
        result = {
            'index': i,
            'vertices': v1,
            'faces': f1,
            'edges': e1,
            'el1': el1,
            'ts1': ts1,
            'cat': category_id,
            'corners1': corners1,
            'vertexFeatures': feature_vector1P,
            'faceFeatures': feature_vector1f,
            'edgeFeatures': feature_vector1,
            'maxCurvature': maxCurvature1,
            'minCurvature': minCurvature1,
            'rotationNormal': rotatingNormal1,
            'hks': hks1
        }
        torch.save(result, "/processed/" + filename)
        print(f"Results saved to {filename}")
        return True

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return False

if __name__ == '__main__':
    cache = r"BenchmarkFlag"
    isCached = os.path.isfile(cache)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelSingle = LBOSingle()
    modelSingle.eval()
    if not isCached:
        print("No cache file out there. Recalculating.")
        files = list(get_iter_obj_paths())

        indexed_files = [(i, file, device, modelSingle) for i, file in enumerate(files)]
        with Pool(processes=15) as pool:
            print("Spawning")
            results = pool.map(process_mesh,indexed_files)
