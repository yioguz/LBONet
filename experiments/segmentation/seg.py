#  Copyright (c) 2022. Implementation of "RiemannNet"
#  by Oguzhan Yigit and Richard C. Wilson

import matplotlib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch
import random
import numpy as np
from experiments.segmentation.segdata import ShapeNetDataset
from experiments.segmentation.seghelpers import smooth_labels_igl, BalancedBatchSampler, collate_fn, cache_hks, \
    load_batch, compute_iou, anneal_log
from preprocessing.preprocess import process_mesh
import pickle
from RiemannNetModelCorrespondence import LBOSingle
from RiemannNetModelSegmentation import RiemannNetModel, PointTransformerSeg, LBOLearning, LBOLearningCacheF, PointTransformerSeg
from torch.utils.data import Dataset, Sampler, DataLoader
from multiprocessing import Pool
from collections import defaultdict

matplotlib.use('Agg')
def get_iter_obj_paths():
    pickle_file = 'iter_obj_paths2.pkl'

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
        print("Directory built.")
    return paths
def iter_obj_paths(base_dir, target_filename="model_normalized.obj"):
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()  # sort subdirectories in-placereverse=True
        files.sort()  # optional: also sort files
        if target_filename in files:
            parts = root.split(os.sep)
            category_id = parts[-3]
            model_id = parts[-2]
            if (category_id, model_id) not in valid_ids:
                continue
            print(os.path.join(root, target_filename))
            yield os.path.join(root, target_filename)

fig, ax = plt.subplots(figsize=(7, 6))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    debug = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = 0
    classes = 50
    vertices_count = 1510
    edge_count = 4520
    faces_count = 3010

    channel = 8
    hks = True
    model = PointTransformerSeg(hks)
    #model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epoch = 5000
    label = 0
    setLength = 16881
    setLengthT = 18

    train = 1
    plotLoss = []
    testPureLoss = []
    cache = r"D:\SHAPENET2\WTF22"

    valid_ids = set()
    pickle_file = 'valid_ids.pkl'
    shapenet_pc_root = r"E:\ShapeNetPC"

    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        print("PC Directory exists. Loading...")
        # Load the valid_ids from the pickle file
        with open(pickle_file, 'rb') as f:
            valid_ids = pickle.load(f)
    else:
        print("PC Directory does not exist. Traversing..")
        for category_id in os.listdir(shapenet_pc_root):
            category_path = os.path.join(shapenet_pc_root, category_id)
            if not os.path.isdir(category_path):
                continue
            for fname in os.listdir(category_path):
                if fname.endswith(".txt"):
                    model_id = os.path.splitext(fname)[0]
                    valid_ids.add((category_id, model_id))

        # Save the valid_ids to a pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(valid_ids, f)

    print(f"Number of valid ids: {len(valid_ids)}")

    isCached = os.path.isfile(cache)

    # load model
    vertices = []
    faces = []
    edges = []
    curvature = []
    gaussian = []
    vertices2 = []
    faces2 = []
    edges2 = []
    curvature2 = []
    gaussian2 = []
    x2 = []

    with open("indices.pkl", "rb") as f:
        print("Loading PC data and test/train/val set indices")
        train_indices, test_indices, val_indices, categories, xyzL, labelL = pickle.load(f)
    bestLoss = 0

    cat_list = ['03261776', '03790512', '04099429',
                '02958343', '03642806', '02954340', '04225987',
                '03797390', '03467517', '02773838', '03636649',
                '04379243', '02691156', '03948459', '03001627',
                '03624134']

    seg_classes = [[16, 17, 18], [30, 31, 32, 33, 34, 35], [41, 42, 43],
                   [8, 9, 10, 11], [28, 29], [6, 7], [44, 45, 46],
                   [36, 37], [19, 20, 21], [4, 5], [24, 25, 26, 27],
                   [47, 48, 49], [0, 1, 2, 3], [38, 39, 40],[12, 13, 14, 15],
                   [22, 23]]

    cat_to_idx = {cat: idx for idx, cat in enumerate(cat_list)}
    indices = torch.tensor([cat_to_idx[cat] for cat in categories])
    one_hot = torch.nn.functional.one_hot(indices, num_classes=len(cat_list))

    seg_cats = ['Earphone', 'Motorbike', 'Rocket', 'Car', 'Laptop', 'Cap', 'Skateboard',
                   'Mug', 'Guitar', 'Bag', 'Lamp', 'Table', 'Airplane', 'Pistol', 'Chair', 'Knife']

    subL = [0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 15]#0, 1, 2, , 6, 10
    torch.manual_seed(4424658)

    segTrain = torch.ones(setLength, vertices_count, dtype=torch.long)*-1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    maxfaces = 7030
    modelSingle = LBOSingle()
    modelSingle.eval()
    if not isCached:
        print("No cache file out there. Recalculating.")
        files = list(get_iter_obj_paths())
        indexed_files = [(i, file, device, modelSingle) for i, file in enumerate(files)]

        with Pool(processes=16) as pool:
            print("Spawning")
            print(indexed_files)
            results = pool.map(process_mesh,indexed_files)

        results = [res for res in results if res is not None]
        hkslist = []
        labellist = []

    else:
        files = np.array(list(get_iter_obj_paths()))

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))

    print("Loading Model")

    mask = torch.isin(indices[train_indices], torch.tensor(subL))
    maskV = torch.isin(indices[val_indices], torch.tensor(subL))
    subsetV = torch.tensor(val_indices)#[maskV]
    subset = torch.tensor(train_indices)#[mask]

    train_files = files[subset]  # or any subset of your files
    val_files = files[subsetV]

    # Define the batch size
    batch_size = 16
    num_batches = len(train_files)  # You want to generate 178 batches

    batch = 50
    threshold = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
    model.trainingS = 1

    weights = torch.load("paper-260")
    model.cuda()

    model.train()
    debugPlot = 1
    batchLossM = 100
    batchloss = []
    testloss = []
    missingSegments = []
    previousLoss = 0
    resetted = False
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
    for t in range(epoch):
        print('\rEpoch [%d / %d]' % (t, epoch), end="")
        count = 0.0
        train_loss = 0.0

        subbatch = 1
        dataset = ShapeNetDataset(train_files)
        # Create the sampler
        balanced_sampler = BalancedBatchSampler(train_files, batch_size=batch_size, num_batches=num_batches)#, mode="all_categories"
        loss_fn2 = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=balanced_sampler, collate_fn=collate_fn)
        k = 0
        if os.path.exists("blacklist"):
            with open(pickle_file, 'rb') as f:
                blacklist = pickle.load(f)
        generate_hks = 0
        if generate_hks:
            modelCache = LBOLearningCacheF(50)
            weights = torch.load("paper-260")
            modelCache.cuda()
            ff = [item for item in files if "03261776" in item]
            indexed_files = [(i, file, device, modelCache, one_hot[i], weights) for i, file in enumerate(files)]
            with Pool(processes=10) as pool:
                results = pool.map(cache_hks, indexed_files)
        model.train()
        for l in range(subbatch):
            batchLoss = 0
            bbiou = 0
            biou = 0
            validate = 0
            valset = 1
            traino = 1
            sim_missings = 0
            sim_missingsc = 0
            num_total_pos = 0
            num_total_neg = 0
            num_hard_pos = 0
            separation_gap = 0
            if traino:
                for j in range(1):
                    torch.save(model.state_dict(), "fpaper-" + str(t) + str(k))
                    for batch in train_loader:
                        biou = 0
                        optimizer.zero_grad()
                        print('\rBatch [%d / %d]' % (k, len(train_loader)), end="")
                        print(k, end=",")
                        blist = torch.tensor([subset[x] for x in batch])#
                        category = [indices[x].item() for x in blist]

                        v1, f1, e1, feature_vector1, feature_vector1P, feature_vector1f, segTrain, el1, ts1, corners1, minCurvature1, maxCurvature1, rotatingNormal1, hks1 = load_batch(files[blist], edge_count)
                        if v1==None:
                            continue
                        cats = one_hot[blist]
                        y1, y2 = model(t, blist, v1,
                                   f1, e1, feature_vector1.to(device), torch.cat(((feature_vector1P),  hks1), dim=1).to(device), feature_vector1f.to(device),
                                   el1.to(device),
                                   ts1.to(device),
                                   corners1.to(device),
                                   minCurvature1.to(device),
                                   maxCurvature1.to(device),
                                   rotatingNormal1.to(device),
                                   cats.to(device),
                                   hks1.to(device))

                        losses = []
                        losses2 = []
                        y2 = y2.transpose(1, 2)
                        temperature = 0.15
                        for i in range(len(blist)):
                            simMask = False
                            lossfactor = 1
                            try:
                                plotOffsetV = (v1[i].transpose(1, 0).sum(dim=1) == 0).nonzero(as_tuple=True)[0][-1]+1
                            except:
                                plottOffsetV = 0
                            category = indices[blist[i]].item()
                            segment_indices = seg_classes[category]

                            plotOffsetF = np.argmin(f1[i].transpose(1, 0).numpy().sum(axis=1) == 0)
                            seg_data = segTrain[i][plotOffsetV:].long().cuda()  # [N]
                            seg_ids = seg_classes[category]#torch.unique(seg_data, sorted=True)  # [num_parts]
                            h = F.normalize(y1[i][plotOffsetV:], p=2, dim=1)

                            max_seg_id = max(seg_ids)
                            global_to_local = torch.full((max_seg_id + 1,), -1, dtype=torch.long, device=seg_data.device)
                            global_to_local[seg_ids] = torch.arange(len(seg_ids), device=seg_data.device)

                            mapped_labels = global_to_local[seg_data]

                            present_segments = torch.unique(mapped_labels)  # segments present in shape i, shape [num_present]
                            missing_segments = torch.tensor(
                                [s for s in global_to_local[global_to_local!=-1] if s not in present_segments.cpu().tolist()], device=mapped_labels.device)
                            if missing_segments.numel() > 0:
                                lossfactor = 2
                            lambda_missing = 2  # tune this weighting factor
                            sim_matrix = h @ h.T
                            labels = seg_data.unsqueeze(0)  # (1, N)
                            mask_pos = labels == labels.T  # (N, N) bool: True if same segment
                            #mask_neg = ~mask_pos  # different segments
                            mask_self = torch.eye(h.size(0), dtype=torch.bool, device=h.device)
                            mask_pos = mask_pos & ~mask_self  # exclude self-comparisons

                            sim_matrix = sim_matrix / temperature
                            sim_matrix_masked = sim_matrix.masked_fill(mask_self,
                                                                       float('-inf'))  # mask out diagonal by -inf
                            log_prob = sim_matrix - torch.logsumexp(sim_matrix_masked, dim=1,
                                                                    keepdim=True)  # stable log softmax denominator
                            mask = torch.zeros_like(y2[i], dtype=torch.bool)
                            mask[segment_indices] = 1
                            masked_segment = y2[i].clone()

                            masked_segment[~mask] = float('-inf')

                            # Get logits and labels
                            logits = y2[i][seg_ids, plotOffsetV:]  # [num_valid_parts, N]
                            mapped_labels = global_to_local[seg_data]  # [N]
                            predicted = torch.argmax(masked_segment, axis=0)[plotOffsetV:]
                            iou = compute_iou(predicted, (seg_data).cuda(), segment_indices).cpu()
                            biou += (torch.mean(iou.float()))
                            loss_n = loss_fn(logits.transpose(1,0), mapped_labels)  # shape: scalar
                            loss_per_point = F.cross_entropy(logits.transpose(1, 0), mapped_labels, reduction='none', ignore_index=-1)
                            _, hard_indices = torch.topk(loss_per_point, 100)
                            N = logits.shape[1]
                            hard_mask = torch.zeros(N, dtype=torch.bool)
                            hard_mask[hard_indices] = True
                            mask_pos_hard = mask_pos.clone()
                            mask_pos_hard[~hard_mask, :] = False
                            mask_pos_hard[:, ~hard_mask] = False
                            if t>1:
                                loss_i = anneal_log(t-2) * (-log_prob[mask_pos_hard].mean() + 0.3 * -log_prob[mask_pos].mean())# + lambda_missing * repulsion_loss
                            else:
                                loss_i = 0
                            losses.append(lossfactor * (0.2 * loss_i +loss_n))#

                        loss = torch.stack(losses).mean()
                        batchLoss = batchLoss + loss.item()
                        print("train loss", loss.item(), biou/len(blist))

                        bbiou = bbiou + biou/len(blist)
                        loss.backward()
                        optimizer.step()
                        k += 1
            print(batchLoss, bbiou)
            if bbiou>bestLoss:
                bestLoss = bbiou
                torch.save(model.state_dict(), str(bbiou.item()) + " IOU")
            if valset:
                ioum = 0
                acc = 0
                ioud = defaultdict(float)
                ioua = defaultdict(float)
                iouc = defaultdict(int)
                with torch.no_grad():
                    model.eval()
                    for rk in range(int(len(subsetV)/32)+1):
                        blist = subsetV[rk * 32:rk * 32 + 32]
                        v1, f1, e1, feature_vector1, feature_vector1P, feature_vector1f, segTrain, el1, ts1, corners1, minCurvature1, maxCurvature1, rotatingNormal1, hks1 = load_batch(files[blist], edge_count)
                        cats = one_hot[blist]
                        g1, y1 = model(t, blist, v1,
                                   f1, e1, feature_vector1.to(device),
                                   torch.cat(((feature_vector1P), hks1), dim=1).to(device), feature_vector1f.to(device),
                                   el1.to(device),
                                   ts1.to(device),
                                   corners1.to(device),
                                   minCurvature1.to(device),
                                   maxCurvature1.to(device),
                                   rotatingNormal1.to(device),
                                   cats.to(device),
                                   hks1.to(device))

                        y1 = y1.transpose(1, 2)
                        masked_y1 = []
                        for i in range(len(blist)):  # Iterate over each sample
                            category = indices[blist[i]].item()
                            plotOffsetV = np.argmin(v1[i].transpose(1, 0).numpy().sum(axis=1) == 0)
                            plotOffsetF = np.argmin(f1[i].transpose(1, 0).numpy().sum(axis=1) == 0)
                            segment_indices = seg_classes[category]
                            mask = torch.zeros_like(y1[i], dtype=torch.bool)
                            mask[segment_indices] = 1
                            masked_segment = y1[i].clone()
                            masked_segment[~mask] = float('-inf')


                            predicted = torch.argmax(masked_segment, axis=0)[plotOffsetV:]
                            predicted = torch.tensor(
                                smooth_labels_igl(v1[i][:, plotOffsetV:].transpose(1, 0).cpu().numpy(),
                                                  f1[i][:, plotOffsetF:].transpose(1, 0).cpu().numpy(),
                                                  predicted.cpu().numpy())).cuda()
                            ground_truth = segTrain[i][plotOffsetV:]
                            ground_truth = torch.tensor(
                                smooth_labels_igl(v1[i][:, plotOffsetV:].transpose(1, 0).cpu().numpy(),
                                                  f1[i][:, plotOffsetF:].transpose(1, 0).cpu().numpy(),
                                                  ground_truth.cpu().numpy()))
                            acc += torch.sum(predicted == torch.tensor(ground_truth).cuda()) / len(predicted)
                            iou = compute_iou(predicted, torch.tensor(ground_truth).cuda(), segment_indices).cpu()

                            ioud[category] += (np.mean(iou.numpy()))
                            iouc[category] += 1
                            ioum += np.mean(iou.numpy())
                for cat in ioud:
                    avg_iou = ioud[cat] / iouc[cat]
                    ioua[cat] = avg_iou
                    print(f"{seg_cats[cat]}: {avg_iou:.4f}", iouc[cat])
                model.train()
                mean_iouc = sum(ioua.values()) / 16
                print(f"Mean iouc: {mean_iouc:.4f}")
                print("mean ioum", ioum/(1860))
                print("mean acc", acc / (1860))
                plt.subplot(2, 2, 2)
                plt.clf()
                plotLoss.append(batchLoss / (758 * 16))
                batchloss.append(batchLoss / (758 * 16))
                plt.plot(plotLoss, 'b', label='Train Loss')
                plt.plot(testloss, 'r', label='Test Accuracy')
                plt.plot(testPureLoss, 'g', label='Test Loss')
                plt.legend()
                plt.draw()
                plt.pause(0.001)
                plt.savefig("LBONET")
        print(t, batchLoss)
