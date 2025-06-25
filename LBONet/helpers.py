import hashlib
import os
import pickle

import pyvista as pv
import torch
import torch.nn as nn
import numpy as np


CACHE_DIR = "precomputed"
os.makedirs(CACHE_DIR, exist_ok=True)

def tensor_hash(*tensors):
    m = hashlib.sha256()
    for t in tensors:
        arr = t.cpu().numpy().astype('float32')  # ensure consistent dtype
        m.update(arr.tobytes())
    return m.hexdigest()

def load_from_cache(hash_key):
    path = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def load_from_cache2(hash_key):
    path = os.path.join(CACHE_DIR, f"{hash_key}-lbo.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(hash_key, data):
    path = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_to_cache2(hash_key, data):
    path = os.path.join(CACHE_DIR, f"{hash_key}-lbo.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def plot_features(pc, feature):
    offset = torch.where((pc.sum(dim=1) != 0) == True)[0][0]
    point_cloud = pv.PolyData(pc[offset:].cpu().numpy())
    point_cloud['labels'] = np.array(feature[offset:])
    #point_cloud2['labels'] = np.array(feature[offset:])
    # Plot with color based on the labels
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars='labels', cmap='turbo', point_size=5, interpolate_before_map=False)
    plotter.show()

def plot_feature(pc, pc2, feature):
    offset = torch.where((pc.sum(dim=1) != 0) == True)[0][0]
    point_cloud = pv.PolyData(pc[offset:].cpu().numpy())
    point_cloud2 = pv.PolyData(pc2[offset:].cpu().numpy())
    point_cloud['labels'] = np.array(feature[offset:])
    #point_cloud2['labels'] = np.array(feature[offset:])
    # Plot with color based on the labels
    plotter = pv.Plotter()
    print(point_cloud, point_cloud2)
    plotter.add_mesh(point_cloud, scalars='labels', cmap='turbo', point_size=5, interpolate_before_map=False)
    plotter.add_mesh(point_cloud2, cmap='turbo', point_size=5, interpolate_before_map=False)
    plotter.show()

def plot_nn(pc, feature, nn):
        offset = torch.where((pc.sum(dim=1)!=0)==True)[0][0]
        point_cloud = pv.PolyData(pc[offset:].cpu().numpy())
        feature = feature * 0+1
        print(nn)
        feature[nn[offset+20]] = 0
        #print(len(feature[nn[offset+20]] ))
        point_cloud['labels'] = np.array(feature[offset:])
        # Plot with color based on the labels
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, scalars='labels', cmap='Set1', point_size=5, interpolate_before_map=False)
        plotter.show()

def plotMesh(vertices, faces, edge, map):#write_obj_pair("test.obj", vertices[0][vertOffset:,:].numpy(), faces[0][facsOffset:,:].numpy(), "texture.png")
    p = pv.Plotter()
    vertOffset = torch.where((torch.sum(vertices, 0) != 0) == True)[0][0]
    facsOffset = torch.where((torch.sum(faces, 0) != 0) == True)[0][0]
    edgeOffset = torch.where((torch.sum(edge, 0) != 0) == True)[0][0]
    plotFacesAnimate(vertices[0, vertOffset:,:].cpu().numpy(), faces[0, facsOffset:,:].cpu().numpy(),
                     (map.squeeze(0).squeeze(0)[vertOffset:].detach().cpu().numpy()), edge[edgeOffset:,:], p)
    p.show()