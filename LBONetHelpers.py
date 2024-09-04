#  Copyright (c) 2024. Implementation of "RiemannNet"
#  by Oguzhan Yigit and Richard C. Wilson
import glob

import torch
import trimesh
import igl
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

    idx1 = torch.max(torch.column_stack((x, y, z)), 1)[0] < torch.pi / 2
    v[idx1] = (1 / 8) * ((-(
                c[idx1] ** 3 / ((torch.sin(z[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1])))) + cot(
        z[idx1]) * 2 * c[idx1]) - (-((a[idx1] ** 3 * torch.cos(y[idx1])) / (
                (torch.sin(x[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1]))))))

    idx2 = y > torch.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(
        z[idx2])

    idx3 = (y < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)), 1)[0] > torch.pi / 2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(
        z[idx3])

    return v


def voronoiPartB(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)), 1)[0] < torch.pi / 2
    v[idx1] = (1 / 8) * ((-(
                b[idx1] ** 3 / ((torch.sin(y[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1])))) + cot(
        y[idx1]) * 2 * b[idx1]) - (-((a[idx1] ** 3 * torch.cos(z[idx1])) / (
                (torch.sin(x[idx1]) ** 2) * (a[idx1] * b[idx1] * torch.sin(z[idx1]))))))

    idx2 = z > math.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(
        y[idx2])

    idx3 = (z < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)), 1)[0] > torch.pi / 2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(
        y[idx3])

    return v


def voronoiPartImpact1(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)), 1)[0] < torch.pi / 2
    v[idx1] = (1 / 8) * (((a[idx1] ** 3 * torch.cos(y[idx1])) / (
                (torch.sin(x[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))) + (
                                     (b[idx1] ** 3 * torch.cos(x[idx1])) / (
                                         (torch.sin(y[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))))

    idx2 = z > math.pi / 2
    v[idx2] = (1 / 8) * (
                (a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5) * torch.cos(z[idx2]))

    idx3 = (z < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)), 1)[0] > torch.pi / 2)
    v[idx3] = (1 / 16) * (
                (a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5) * torch.cos(z[idx3]))
    return v


def voronoiPartImpact2(x, y, z, a, b, c, device=False):
    if device:
        v = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
    else:
        v = torch.zeros(x.shape[0], dtype=torch.float64)
    idx1 = torch.max(torch.column_stack((x, y, z)), 1)[0] < torch.pi / 2
    v[idx1] = (1 / 8) * (((c[idx1] ** 3 * torch.cos(x[idx1])) / (
                (torch.sin(z[idx1]) ** 2) * (b[idx1] * c[idx1] * torch.sin(x[idx1])))) + (
                                     (a[idx1] ** 3 * torch.cos(z[idx1])) / ((torch.sin(x[idx1]) ** 2) * (
                                         b[idx1] * c[idx1] * torch.sin(x[
                                                                           idx1])))))  # (1 / 8) * (((b**3 * torch.cos(y)) / ((torch.sin(y) ** 2)*(b*c*torch.sin(x)))) + ((a**3 * torch.cos(z)) / ((torch.sin(x) ** 2)*(b*c*torch.sin(x)))))#(1 / 8) * ((-(c**3 * torch.cos(y)) / ((torch.sin(z) ** 2)*(b*c*torch.sin(x)))))

    idx2 = y > torch.pi / 2
    v[idx2] = (1 / 8) * ((a[idx2] * b[idx2] * c[idx2]) / (b[idx2] * c[idx2] * torch.sin(x[idx2]) * 0.5)) * torch.cos(
        y[idx2])

    idx3 = (y < torch.pi / 2) & (torch.max(torch.column_stack((x, y, z)), 1)[0] > torch.pi / 2)
    v[idx3] = (1 / 16) * ((a[idx3] * b[idx3] * c[idx3]) / (b[idx3] * c[idx3] * torch.sin(x[idx3]) * 0.5)) * torch.cos(
        y[idx3])
    return v


def get_anisotropic_laplacian(VPos, ITris, Riemannian, edges, minCurvature, maxCurvature, rotationNormal, anisotropy1, anisotropy2, Theta, corners, ts, voronoi):
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
    beta = 0.0005
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
    e *= 1.1
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
        dotmaxe1R = (((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1,0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * maxCurvatureR.transpose(1,0)).sum(dim=-1)
        #dotmaxe2 = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * maxCurvature.transpose(1, 0)).sum(dim=-1)
        dotmaxe2R = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * maxCurvatureR.transpose(1, 0)).sum(dim=-1)
        #dotmine1 = (((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * minCurvature.transpose(1, 0)).sum(dim=-1)
        dotmine1R = (((((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk)).transpose(1, 0) / (torch.norm(((VPos[ITris[:, j], :]*riemannFactorij[:, None]) - pointk), dim=1))).transpose(1, 0) * minCurvatureR.transpose(1, 0)).sum(dim=-1)
        #dotmine2 = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * minCurvature.transpose(1, 0)).sum(dim=-1)
        dotmine2R = ((((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None]))).transpose(1, 0) / (torch.norm((pointk - (VPos[ITris[:, i], :]*riemannFactorij[:, None])), dim=1))).transpose(1, 0) * minCurvatureR.transpose(1, 0)).sum(dim=-1)

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


    Au = torch.tensor(np.diag(igl.massmatrix_intrinsic(el.cpu().numpy(), ITris.numpy(), igl.MASSMATRIX_TYPE_VORONOI).toarray()))
    A = (Au * voronoi.squeeze(0).cpu().numpy()).numpy()
    return A, L, rangle, el, ofD1[:, fOffset:], ofD2[:, fOffset:], ofDR[:, fOffset:], dmax[:, fOffset:], dmin[:, fOffset:], drot[:, fOffset:], gV, gL, Au