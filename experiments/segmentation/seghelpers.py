import math
import os
import pickle
from collections import defaultdict
from random import random
import torch
import igl
from torch.utils.data import Dataset, Sampler, DataLoader
import torch.nn.functional as F
def smooth_labels_igl(V, F, labels, lamb=1.0, iterations=5):
    """
    Smooth hard segmentation labels on mesh using majority voting.

    Args:
        V (np.ndarray): Nx3 vertices
        F (np.ndarray): Mx3 faces
        labels (np.ndarray): N labels (int)
        lamb (float): smoothing factor (0 = no smoothing, 1 = full smoothing)
        iterations (int): number of smoothing iterations

    Returns:
        np.ndarray: N smoothed hard labels
    """
    # Build vertex adjacency (list of neighbors for each vertex)
    # Using igl.vertex_triangle_adjacency to get adjacency matrix or igl.adjacency_matrix
    adj = igl.adjacency_matrix(F)
    adj = adj.tolil()  # For easy neighbor access

    new_labels = labels.copy()

    for _ in range(iterations):
        updated_labels = new_labels.copy()
        for vi in range(V.shape[0]):
            neighbors = adj.rows[vi]
            if len(neighbors) == 0:
                # isolated vertex keep original
                continue

            # Collect neighbor labels
            neighbor_labels = new_labels[neighbors]

            # Weighted voting:
            # Combine self-label with neighbors based on lambda
            # The effective voting pool:
            # - self label counts as weight (1 - lambda)
            # - each neighbor counts as weight lambda / len(neighbors)

            # Build label histogram
            label_counts = {}
            # Add self label weighted
            label_counts[new_labels[vi]] = label_counts.get(new_labels[vi], 0) + (1 - lamb)
            # Add neighbor labels weighted
            weight_per_neighbor = lamb / len(neighbors)
            for nl in neighbor_labels:
                label_counts[nl] = label_counts.get(nl, 0) + weight_per_neighbor

            # Majority voting by max weighted count
            max_label = max(label_counts.items(), key=lambda x: x[1])[0]

            # Prevent small region shrinkage: if current vertex label is unique in neighborhood (small region),
            # bias toward keeping current label by slightly increasing its weight
            if new_labels[vi] not in neighbor_labels:
                # Boost current label weight slightly
                label_counts[new_labels[vi]] += 0.1
                max_label = max(label_counts.items(), key=lambda x: x[1])[0]

            updated_labels[vi] = max_label

        new_labels = updated_labels

    return new_labels



class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, num_batches, mode="balanced", max_classes=None):
        """
        BalancedBatchSampler for creating batches with different sampling modes.

        Parameters:
        - data_source: List of file paths.
        - batch_size: Number of samples per batch.
        - num_batches: Number of batches to generate.
        - mode: Sampling mode. Options:
            - "balanced": Balanced sampling across all categories.
            - "all_categories": Ensures all categories appear in each batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.mode = mode
        self.max_classes = max_classes
        self.category_to_indices = self._get_category_to_indices()

    def _get_category_to_indices(self):
        # Create a mapping from category to indices
        category_to_indices = defaultdict(list)
        for idx, file_path in enumerate(self.data_source):
            category_id = file_path.split('ShapeNet\\')[1].split('\\')[0]  # Extract the category ID
            category_to_indices[category_id].append(idx)
        return category_to_indices

    def __iter__(self):
        for _ in range(self.num_batches):
            if self.mode == "balanced":
                yield self._generate_balanced_batch()
            elif self.mode == "all_categories":
                yield self._generate_all_categories_batch()
            elif self.mode == "max_n_classes":
                yield self._generate_max_n_classes_batch()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_balanced_batch(self):
        batch_indices = []
        batch_paths = []

        num_classes = len(self.category_to_indices)
        samples_per_class = max(1, self.batch_size // num_classes)

        for indices in self.category_to_indices.values():
            sampled_indices = random.sample(indices, min(samples_per_class, len(indices)))
            batch_indices.extend(sampled_indices)
            batch_paths.extend([self.data_source[i] for i in sampled_indices])

        # Fill remaining with random samples if the batch is too small
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            all_indices = [i for indices in self.category_to_indices.values() for i in indices]
            extra_indices = random.sample(all_indices, remaining)
            batch_indices.extend(extra_indices)
            batch_paths.extend([self.data_source[i] for i in extra_indices])

        random.shuffle(batch_indices)
        return batch_indices, batch_paths


    def _generate_max_n_classes_batch(self):
        if not self.max_classes:
            raise ValueError("max_classes must be set for 'max_n_classes' mode.")

        selected_categories = random.sample(list(self.category_to_indices.keys()),
                                            min(self.max_classes, len(self.category_to_indices)))
        samples_per_class = max(1, self.batch_size // len(selected_categories))

        batch_indices = []
        for cat in selected_categories:
            indices = self.category_to_indices[cat]
            sampled = random.sample(indices, min(samples_per_class, len(indices)))
            batch_indices.extend(sampled)

        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            extra_indices = [i for cat in selected_categories for i in self.category_to_indices[cat]]
            batch_indices.extend(random.sample(extra_indices, min(remaining, len(extra_indices))))

        random.shuffle(batch_indices)
        batch_paths = [self.data_source[i] for i in batch_indices]
        return batch_indices, batch_paths
    def _generate_all_categories_batch(self):
        batch_indices = []
        batch_paths = []

        for indices in self.category_to_indices.values():
            if indices:
                sampled_indices = random.choices(indices, k=1)
                batch_indices.extend(sampled_indices)
                batch_paths.extend([self.data_source[i] for i in sampled_indices])

        # Fill remaining with random samples if the batch is too small
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            all_indices = [i for indices in self.category_to_indices.values() for i in indices]
            extra_indices = random.sample(all_indices, remaining)
            batch_indices.extend(extra_indices)
            batch_paths.extend([self.data_source[i] for i in extra_indices])

        random.shuffle(batch_indices)
        return batch_indices, batch_paths

    def __len__(self):
        return self.num_batches
def collate_fn(batch):
    indices, files = batch[0]
    # The batch consists of indices, so we need to look up the actual file paths
    return indices



def load_batch(file_list, edge_count, baseline=True, default=False):
    vertices, faces, edges = [], [], []
    vertex_feats, face_feats, edge_feats = [], [], []
    segs, el1s, ts1s, corners = [], [], [], []
    min_curvs, max_curvs, rot_normals, hks, hksd = [], [], [], [], []
    normalize=1
    for f in file_list:
        pathsplit = (f.split(os.sep))
        finalPath = os.path.join("E:\\", "SHAPENET2", pathsplit[-4] + "_" + pathsplit[-3])
        if not os.path.isfile(finalPath):
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None
        data = torch.load(finalPath)

        try:
            vo = torch.max(torch.where((data['vertices'].sum(dim=0)==0)==True)[0])
        except:
            vo = -1
        fo = torch.max(torch.where((data['faces'].sum(dim=0) == 0) == True)[0])
        eo = torch.max(torch.where((torch.sum(data['edges'], dim=0))==0)[0])
        vertices.append(data['vertices'])
        faces.append(data['faces'])
        edges.append(data['edges'])
        vertex_feats.append(data['vertexFeatures'])
        if normalize:
            hksT = data['hks'].transpose(1,0)  # shape: [8, 1510]
            hksT = torch.clip(torch.log1p(hksT),
                             *torch.quantile(torch.log1p(hksT), torch.tensor([0.05, 0.95], device=hksT.device)))
            hksz = ((hksT - hksT.min(0, keepdim=True)[0]) / (
                        hksT.max(0, keepdim=True)[0] - hksT.min(0, keepdim=True)[0] + 1e-8)).transpose(1,0)

        else:
            hksz = data['hks']
        normals = data['vertexFeatures'][-3:]
        face_feats.append(torch.cat((data['faceFeatures'][:-8], F.pad(torch.tensor(igl.average_onto_faces(data['faces'][:, fo+1:].transpose(1,0).numpy(), normals[:, vo+1:].transpose(1,0).numpy())).transpose(1,0), (fo+1, 0)), F.pad(torch.tensor(igl.average_onto_faces(data['faces'][:, fo+1:].transpose(1,0).numpy(), hksz[:, vo+1:].transpose(1,0).cpu().numpy())).transpose(1,0), (fo+1, 0)), F.pad(torch.tensor(igl.average_onto_faces(data['faces'][:, fo+1:].transpose(1,0).numpy(), data['vertexFeatures'][4, vo+1:].cpu().numpy())), (fo+1, 0)).unsqueeze(0)), dim=0))
        edge_feats.append(torch.cat((data['edgeFeatures'], F.pad(((normals[:, vo+1:][:, data['edges'][0, eo+1:]] + normals[:, vo+1:][:, data['edges'][1, eo+1:]])/2), (eo+1, 0)), F.pad(((hksz[:, vo+1:][:, data['edges'][0, eo+1:]] + hksz[:, vo+1:][:, data['edges'][1, eo+1:]]).cpu()/2), (eo+1, 0))), dim=0))
        segs.append(data['labels'])
        el1s.append(data['el1'])
        ts1s.append(F.pad(data['ts1'], (0, 0, edge_count - data['ts1'].shape[0], 0)))
        corners.append(F.pad(data['corners1'], (0, 0, edge_count - data['corners1'].shape[0], 0)))
        min_curvs.append(data['minCurvature'])
        max_curvs.append(data['maxCurvature'])
        rot_normals.append(data['rotationNormal'])

        if baseline:
            hks.append(hksz.cpu())
            bh = data['hks']
            mean = bh.mean(dim=1, keepdim=True)  # Shape: [16, 8, 1]
            std = bh.std(dim=1, keepdim=True) + 1e-6  # avoid division by zero
            bh = (bh - mean) / std  # Shape remains [16, 8, 1510]
            hksd.append(bh)
        else:
            dataM = torch.load(os.path.join("D:\\", "SHAPENET11\\mhks", pathsplit[-4] + "_" + pathsplit[-3]))
            hksN = dataM['hks']
            hks.append(hksN.squeeze(1).transpose(1,0).cpu())

    if default:
        return (
            torch.stack(vertices),
            torch.stack(faces),
            torch.stack(edges),
            torch.stack(edge_feats),
            torch.stack(vertex_feats),
            torch.stack(face_feats),
            torch.stack(segs),
            torch.stack(el1s),
            torch.stack(ts1s),
            torch.stack(corners),
            torch.stack(min_curvs),
            torch.stack(max_curvs),
            torch.stack(rot_normals),
            torch.stack(hks),
            torch.stack(hksd)
        )
    else:
        return (
            torch.stack(vertices),
            torch.stack(faces),
            torch.stack(edges),
            torch.stack(edge_feats),
            torch.stack(vertex_feats),
            torch.stack(face_feats),
            torch.stack(segs),
            torch.stack(el1s),
            torch.stack(ts1s),
            torch.stack(corners),
            torch.stack(min_curvs),
            torch.stack(max_curvs),
            torch.stack(rot_normals),
            torch.stack(hks)
        )



def cache_hks(pars):
    try:
        i, file, device, model, cats, weights = pars
        v1, f1, e1, feature_vector1, feature_vector1P, feature_vector1f, segTrain, el1, ts1, corners1, minCurvature1, maxCurvature1, rotatingNormal1, hks1 = load_batch([file], 4520)
        parts = file.split(os.sep)
        category_id = parts[-4]
        model_id = parts[-3]
        filename = category_id + "_" + model_id
        weights = torch.load("paper-260")
        model.load_state_dict(weights, strict=False)
        model.eval()
        hks = model(0, [i], v1,
                   f1, e1, feature_vector1.to(device), torch.cat(((feature_vector1P), hks1), dim=1).to(device),
                   feature_vector1f.to(device),
                   el1.to(device),
                   ts1.to(device),
                   corners1.to(device),
                   minCurvature1.to(device),
                   maxCurvature1.to(device),
                   rotatingNormal1.to(device),
                   cats.unsqueeze(0).to(device),
                   hks1.to(device))

        hks = hks.transpose(1, 0)
        print(hks.shape)
        result = {
            'index': i,
            'hks': hks
        }
        torch.save(result, "D:/SHAPENET11/mhks/" + filename)
        print(f"Results saved to {filename}")
        return

    except Exception as e:
        print(f"Error processing {file}: {e}")
        print(i, file)
        return None



def anneal_log(t, base=101 ** (1 / 9)):
    return 1.0 / (1.0 + math.log(1.0 + t, base))



def compute_iou(predicted, ground_truth, num_classes, val=False):
    ious = []
    for cls in num_classes:
        pred_inds = predicted == cls
        gt_inds = ground_truth == cls

        intersection = (pred_inds & gt_inds).sum().float()
        union = (pred_inds | gt_inds).sum().float()

        if union == 0 or (len(gt_inds[gt_inds==True])>=1 and len(gt_inds[gt_inds==True])<10):
            iou = torch.tensor(1).cuda()  # Class not present in either prediction or gt
        elif not gt_inds.any():
            if val:
                print("Predicted wrong points: ", len(pred_inds[pred_inds==True]), " - ", end="")
            iou = intersection / union
        else:
            iou = intersection / union
        ious.append(iou)
    return torch.stack(ious)  # shape [num_classes]


def extract_ids(json_list):
    return set((path.split('/')[1], path.split('/')[2]) for path in json_list)




def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

def percentile_hard_mining(sim_matrix, labels, top_k_percent=0.1):
    N = sim_matrix.size(0)

    # Remove self-similarity
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(-float('inf'))

    # Create positive and negative masks
    labels = labels.view(-1, 1)  # (N, 1)
    mask_pos = labels == labels.T
    mask_neg = ~mask_pos

    sim_pos = sim_matrix[mask_pos]
    sim_neg = sim_matrix[mask_neg]

    # Compute percentile thresholds dynamically
    num_pos = sim_pos.numel()
    num_neg = sim_neg.numel()

    k_pos = max(1, int(top_k_percent * num_pos))
    k_neg = max(1, int(top_k_percent * num_neg))

    hard_pos_thresh = torch.kthvalue(sim_pos, k_pos).values
    hard_neg_thresh = torch.kthvalue(sim_neg, num_neg - k_neg + 1).values  # top k highest

    # Create masks
    hard_pos_mask = mask_pos & (sim_matrix <= hard_pos_thresh)
    hard_neg_mask = mask_neg & (sim_matrix >= hard_neg_thresh)

    return hard_pos_mask, hard_neg_mask