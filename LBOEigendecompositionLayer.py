#  Copyright (c) 2024. Implementation of "RiemannNet"
#  by Oguzhan Yigit and Richard C. Wilson
import time

import igl
import torch
from scipy import sparse
import numpy as np
import scipy.sparse.linalg as lg
from LBONetHelpers import get_cotan_laplacian, plotEdges, laplacianPart, laplacianPart2, \
    laplacianPartij, laplacianPart2ij, laplacianPartk2, laplacianPartk1, laplacianPartk2A, laplacianPartk2B, \
    laplacianPartk1A, laplacianPartk1B, voronoiPartA, voronoiPartB, voronoiPartImpact1, voronoiPartImpact2, plotFaces, \
    plotFacesAnimate, write_obj_with_colors, get_cotan_laplacian_igl, get_anisotropic_laplacian, plotFacesAnimate2
import pyvista as pv

class Spectral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vertices, edges, faces, x, el, ts, corners, minCurvature, maxCurvature, rotationNormal, anisotropy1, anisotropy2, Theta, voronoi):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.

        """

        # calculate standard Laplacian and Voronoi
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = vertices.shape[0]
        vertex_count = vertices.shape[1]
        faces_count = faces.shape[1]
        edges_count = edges.shape[1]
        frequency = 30



        eps = 1e-9

        fullPrecision = False

        if fullPrecision:
            dtype = torch.float64
        else:
            dtype = torch.float32

        eigvectors = torch.zeros(batch_size, vertex_count, frequency, dtype=dtype, device=device)
        eigvalues = torch.zeros(batch_size, frequency, dtype=dtype, device=device)
        Ab = torch.zeros(batch_size, vertex_count, vertex_count, dtype=dtype)
        Aub = torch.zeros(batch_size, vertex_count, dtype=dtype)
        Lb = torch.zeros(batch_size, vertex_count, vertex_count, dtype=dtype)
        vertOffset = torch.zeros(batch_size, device=device, dtype=torch.long)
        #regularization = torch.zeros(batch_size, edges_count, device=device, dtype=torch.float32)
        facsOffset = torch.zeros(batch_size, device=device, dtype=torch.long)
        edgeOffset = torch.zeros(batch_size, device=device, dtype=torch.long)
        angles = torch.zeros(batch_size, faces_count, 3, dtype=dtype, device=device)
        edge_length = torch.zeros(batch_size, faces_count, 3, dtype=dtype, device=device)
        gV = torch.zeros(batch_size, edges_count, 4, dtype=dtype, device=device)
        gL = torch.zeros(batch_size, edges_count, 14, dtype=dtype, device=device)
        ofD1 = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        ofD2 = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        ofDR = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        dmax = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        dmin = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        drot = torch.zeros(batch_size, 3, faces_count, dtype=dtype, device=device)
        baseline = 0
        if torch.sum(Theta) == 0:
            baseline = 1
        for i in range(batch_size):
            # clear padding
            vertOffset[i] = torch.where((torch.sum(vertices[i], 1) != 0)==True)[0][0]
            facsOffset[i] = torch.where((torch.sum(faces[i], 1) != 0)==True)[0][0]
            edgeOffset[i] = torch.where((torch.sum(edges[i], 1) != 0)==True)[0][0]

            verts = vertices[i][vertOffset[i]:, :]
            facs = faces[i][facsOffset[i]:, :]
            edge = edges[i][edgeOffset[i]:, :]

            ks = (x[i][0, torch.sum(edges[i], 1) != 0].detach())
            if not baseline:
                vor = voronoi[i][:,vertOffset[i]:].detach()
            aniso1 = anisotropy1[i].detach()
            aniso2 = anisotropy2[i].detach()
            Thet = Theta[i].detach()

            if not baseline:
                A, L, angles[i][facsOffset[i]:], edge_length[i][facsOffset[i]:], ofD1[i][:, facsOffset[i]:], ofD2[i][:, facsOffset[i]:], ofDR[i][:, facsOffset[i]:], dmax[i][:, facsOffset[i]:], dmin[i][:, facsOffset[i]:], drot[i][:, facsOffset[i]:], gV[i][edgeOffset[i]:], gL[i][edgeOffset[i]:], Aub[i][vertOffset[i]:] = get_anisotropic_laplacian(verts, facs, ks, edge, minCurvature[i], maxCurvature[i], rotationNormal[i], aniso1, aniso2, Thet, corners[i][edgeOffset[i]:], ts[i][edgeOffset[i]:], vor)
            else:
                A, L, angles[i][facsOffset[i]:], edge_length[i][facsOffset[i]:]= get_cotan_laplacian_igl(verts, facs, ks*0, edge)

            L = L + sparse.identity(L.shape[0]) * eps
            Ab[i] = torch.tensor(np.pad(np.diag(A), (vertex_count - verts.shape[0], 0)))
            Lb[i] = torch.tensor(np.pad(L.toarray(), (vertex_count - verts.shape[0], 0)))

            (eigvalue, eigvector) = lg.eigsh(L.tocsc(), frequency+1, sparse.diags(A), sigma=-0.01)

            eigvalue1 = torch.abs(torch.tensor(eigvalue[1:], requires_grad=False))
            eigvalue1 -= eps
            eigvalues[i], idx = torch.sort(eigvalue1)

            eigvectors[i] = torch.tensor(np.pad(eigvector[:, 1:], [(vertex_count - verts.shape[0], 0), (0, 0)]), requires_grad=False).index_select(-1, idx)  #

        timeframe = torch.logspace(-2, 0, 16)

        k = ((eigvectors[:, :, :, None] ** 2) * torch.exp(-eigvalues[:, None, :, None] * timeframe.to(device).flatten()[None, None, :]))
        k = torch.sum(k, 2).transpose(1, 2)


        ctx.save_for_backward(vertices, edges, faces, eigvectors[:, :, :], eigvalues[:, :], Ab, Lb, timeframe,
                              vertOffset, facsOffset, edgeOffset, angles, x.detach(), ts, corners, ofD1, ofD2, ofDR, dmax, dmin, drot, gV, gL, Aub, voronoi.detach())
        return k.float()

    @staticmethod
    def backward(ctx, hks):
        fullPrecision = False
        if fullPrecision:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        # print("Custom backward called")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vertices, edges, faces, eigvectorsB, eigvaluesB, Ab, Lb, timeframeB, vertOffset, facsOffset, edgeOffset, anglesB, edge_lengthB, tsB, cornersB, ofD1B, ofD2B, ofDRB, dmaxB, dminB, drotB, gVB, gLB, Aub, voronoiB = ctx.saved_tensors
        batch_size = edges.shape[0]
        dEdgeValue = torch.zeros(batch_size, len(edges[0]), dtype=torch.float32, device=device)
        dFaceValue1 = torch.zeros(batch_size, len(faces[0]), dtype=torch.float32, device=device)
        dFaceValue2 = torch.zeros(batch_size, len(faces[0]), dtype=torch.float32, device=device)
        dFaceValue3 = torch.zeros(batch_size, len(faces[0]), dtype=torch.float32, device=device)
        dVertexValue = torch.zeros(batch_size, len(vertices[0]), dtype=torch.float32, device=device)
        timeframe = timeframeB.to(device)
        frequency = eigvaluesB.shape[1]

        Ab = Ab.to(device)
        Lb = Lb.to(device)
        edges = edges.to(device)
        faces = faces.to(device)
        ldiags = [0, 1, 5, 4]

        plot = True
        if fullPrecision:
            dtype = torch.float64
        else:
            dtype = torch.float32
        ALBO = True
        ALBOR = True
        Riemann = True
        Voronoi = True

        for sample in torch.arange(batch_size, dtype=torch.int32, device=device):
            if(torch.sum(hks[sample])<0.001):
                continue

            offsetE = edgeOffset[sample]
            offset = vertOffset[sample]
            offsetF = facsOffset[sample]

            A = Ab[sample][offset:, offset:]
            Au = Aub[sample][offset:]


            VPos = vertices[sample][offset:, :]
            edge = edges[sample][offsetE:, :]
            eigvectors = eigvectorsB[sample][offset:]
            eigvalues = eigvaluesB[sample]

            M = edge.shape[0]
            N = A.shape[0]
            ts = tsB[sample][offsetE:]
            corners = cornersB[sample][offsetE:]
            edge_length = edge_lengthB[sample][0, offsetE:]
            tsl = ts[:, 0]
            tsr = ts[:, 1]
            k1_index = corners[:, 0]
            k2_index = corners[:, 1]
            i = edge[:, 0]
            j = edge[:, 1]



            maximum, ind = torch.max(torch.norm(eigvectors, dim=1), dim=0)
            LMatrix = Lb[sample][offset:, offset:] - (eigvalues[:,None, None] * A[None, :, :])
            LMatrix[:, ind, :] = 0
            LMatrix[:, :, ind] = 0
            LMatrix[:, ind, ind] = 1

            ITris = faces[sample][offsetF:, :]
            try:
                LMatrix = torch.inverse(LMatrix)
            except:
                continue

            ### HKS
            inner = torch.exp(-timeframe * eigvalues[:, None])
            partL = (-timeframe * inner * eigvectors[:, :, None] ** 2).transpose(1,2)
            partR = (2 * inner * eigvectors[:, :, None]).transpose(1,2)

            ATheta = (A @ eigvectors)

            #
            # #
            k1 = torch.diag(ITris[tsl][:, k1_index])  # ITris[ts[0]][corners[0]]
            k2 = torch.diag(ITris[tsr][:, k2_index])  # ITris[ts[1]][corners[1]]

            eigvectors = eigvectors.transpose(1, 0)
            if Riemann:
                gL = gLB[sample][offsetE:]
                gV = gVB[sample][offsetE:]


                indicesLx2 = torch.tensor((i[0], j[0], i[0], j[0], k2[0], k1[0], i[0], k2[0], j[0], k2[0], i[0], k1[0], j[0], k1[0]), device=device) #torch.column_stack((i, j, i, j, k2, k1, i, k2, j, k2, i, k1, j, k1)).to(device)
                indicesLy2 = torch.tensor((i[0], j[0], j[0], i[0], k2[0], k1[0], k2[0], i[0], k2[0], j[0], k1[0], i[0], k1[0], j[0]), device=device) #torch.column_stack((i, j, j, i, k2, k1, k2, i, k2, j, k1, i, k1, j)).to(device)

                uniqueized = torch.unique(indicesLx2, dim=0, sorted=False)

                # triangle ordering
                if uniqueized[2] == k1[0]:
                    index1 = 3
                    index2 = 2
                else:
                    index1 = 2
                    index2 = 3

                indicesLx2[indicesLx2 == uniqueized[0]] = -4
                indicesLx2[indicesLx2 == uniqueized[1]] = -3
                indicesLx2[indicesLx2 == uniqueized[3]] = -2
                indicesLx2[indicesLx2 == uniqueized[2]] = -1
                indicesLx2[indicesLx2 == -4] = 0
                indicesLx2[indicesLx2 == -3] = 1
                indicesLx2[indicesLx2 == -2] = index1
                indicesLx2[indicesLx2 == -1] = index2

                uniqueized = torch.unique(indicesLy2, dim=0, sorted=False)

                indicesLy2[indicesLy2 == uniqueized[0]] = -4
                indicesLy2[indicesLy2 == uniqueized[1]] = -3
                indicesLy2[indicesLy2 == uniqueized[3]] = -2
                indicesLy2[indicesLy2 == uniqueized[2]] = -1
                indicesLy2[indicesLy2 == -4] = 0
                indicesLy2[indicesLy2 == -3] = 1
                indicesLy2[indicesLy2 == -2] = index1
                indicesLy2[indicesLy2 == -1] = index2

                # #
                masterIndex = torch.cat((indicesLx2 * 4 + indicesLy2, torch.tensor([11, 14], device=device)))[None, None, :].repeat(M, frequency, 1)# torch.cat((indicesLx2 * 4 + indicesLy2, torch.tensor([11, 14]).to(device)))

                indicesV = torch.column_stack((i, j, k1, k2))


                mv = gL.unsqueeze(1).repeat(1, frequency, 1)
                mv[:, :, ldiags] = torch.sub(gL[:, ldiags], (eigvalues[:, None, None] * gV)).transpose(1, 0)

                dV = torch.zeros(M, N, device=device, dtype=dtype)
                dV.scatter_(1, indicesV, gV)

                final3 = torch.zeros(M, frequency, 16, device=device, dtype=dtype)
                final3.scatter_(2, masterIndex, torch.cat((mv, torch.zeros(M, frequency, 2, device=device)), 2))
                final3 = final3.view(M * frequency, 4, 4)
                right = eigvectors[:, indicesV].transpose(1,0).contiguous().view(M * frequency, 4)


                indicesVR = indicesV.unsqueeze(1).repeat(1, frequency, 1)
                F_iP2 = torch.zeros(M, frequency, N, device=device, dtype=dtype)
                F_iP2.scatter_(2, indicesVR,
                               torch.bmm(right.unsqueeze(1), final3).transpose(1, 2).view(M, frequency, 4))
                del masterIndex, mv, uniqueized, right, final3

                eigenvalue_gradient = torch.diagonal((eigvectors @ F_iP2.transpose(1, 2)), dim1=-2, dim2=-1).transpose(0,1).unsqueeze(0)
                vectorizedF = ((F_iP2.unsqueeze(2) @ eigvectors[:, :, None] * ATheta.transpose(1, 0)[:, :, None]).squeeze(3) - F_iP2)
                del F_iP2



                vectorizedMu = (LMatrix @ vectorizedF.permute(1,2,0)).permute(2, 0, 1)
                del vectorizedF


                unfoldedEigs = eigvectors.transpose(0, 1).unsqueeze(0).repeat(M, 1, 1)
                vectorizedC2 = torch.diagonal(0.5 * (torch.gather(unfoldedEigs, 1, indicesVR.transpose(1,2)).transpose(1, 2) @ torch.diag_embed(torch.gather(dV, 1, indicesV)) @ torch.gather(unfoldedEigs, 1, indicesVR.transpose(1,2))),dim1=1, dim2=2)
                del unfoldedEigs

                est = (vectorizedMu + ((torch.diagonal(-vectorizedMu @ ATheta, dim1=1, dim2=2) - vectorizedC2)[:, :, None] * eigvectors[None, :, :])).transpose(0,2)
                del vectorizedMu, vectorizedC2, indicesVR, indicesV
                if est.shape[2] < 15000:
                    dHks = (partL @ eigenvalue_gradient + partR @ est).permute(2, 0, 1)
                else:
                    dHksA = torch.zeros(est.shape[0], partL.shape[1], est.shape[2])
                    dHksB = torch.zeros(est.shape[0], partL.shape[1], est.shape[2])
                    for j in range(partL.shape[2]):
                        torch.cuda.empty_cache()
                        dHksA += (partL[:, :, j:j + 1] @ eigenvalue_gradient[:, j:j + 1, :]).cpu()
                        dHksB += (partR[:, :, j:j + 1] @ est[:, j:j + 1, :]).cpu()
                    dHks = (dHksA + dHksB).permute(2, 0, 1).cuda()


                dEdgeValueI = torch.sum(torch.diagonal((hks[sample, :, offset:]@torch.nan_to_num(dHks).float()), dim1=1, dim2=2), 1)

                perc01 = torch.quantile(dEdgeValueI, 0.03, keepdim=True)
                perc99 = torch.quantile(dEdgeValueI, 0.97, keepdim=True)
                dEdgeValueI = torch.clip(dEdgeValueI, min=perc01, max=perc99)
                dEdgeValue[sample, offsetE:] = dEdgeValueI
                del dEdgeValueI, dHks

            ofD1 = ofD1B[sample][:, offsetF:]
            ofD2 = ofD2B[sample][:, offsetF:]
            ofDR = ofDRB[sample][:, offsetF:]
            # anisotropic gradients
            masterIndexA = torch.tensor([0, 4, 8, 1, 5, 2, 3, 7, 6], device=device)[None, None, :].repeat(
                ITris.shape[0], frequency, 1)
            rightA = eigvectors[:, ITris].transpose(1, 0).contiguous().view(ITris.shape[0] * frequency, 3)
            if ALBO:
                gLA = torch.column_stack((dmaxB[sample][:, offsetF:].transpose(1, 0), ofD1.transpose(1, 0), ofD1.transpose(1, 0))).cuda()

                final3A = torch.zeros(ITris.shape[0], frequency, 9, device=device, dtype=dtype)
                final3A.scatter_(2, masterIndexA, gLA.unsqueeze(1).repeat(1, frequency, 1))
                final3A = final3A.view(ITris.shape[0] * frequency, 3, 3)

                F_iP2 = torch.zeros(ITris.shape[0], frequency, N, device=device, dtype=dtype)
                F_iP2.scatter_(2, ITris.unsqueeze(1).repeat(1, frequency, 1),
                               torch.bmm(rightA.unsqueeze(1), final3A).transpose(1, 2).view(ITris.shape[0], frequency, 3))
                del final3A, gLA

                eigenvalue_gradient = torch.diagonal((eigvectors @ F_iP2.transpose(1, 2)), dim1=-2, dim2=-1).transpose(0,
                                                                                                                       1).unsqueeze(
                    0)
                vectorizedF = ((F_iP2.unsqueeze(2) @ eigvectors[:, :, None] * ATheta.transpose(1, 0)[:, :, None]).squeeze(
                    3) - F_iP2)
                del F_iP2

                vectorizedMu = (LMatrix @ vectorizedF.permute(1, 2, 0)).permute(2, 0, 1)
                del vectorizedF

                est = (vectorizedMu + (torch.diagonal(-vectorizedMu @ ATheta, dim1=1, dim2=2)[:, :, None] * eigvectors[None, :, :])).transpose(0, 2)
                del vectorizedMu
                dFaceValueI = torch.nan_to_num(
                    torch.sum(torch.diagonal((hks[sample, :, offset:] @  (partL @ eigenvalue_gradient + partR @ est).permute(2, 0, 1).float()), dim1=1, dim2=2), 1))
                perc01 = torch.quantile(dFaceValueI, 0.01, keepdim=True)
                perc99 = torch.quantile(dFaceValueI, 0.99, keepdim=True)
                dFaceValueI = torch.clip(dFaceValueI, min=perc01, max=perc99)
                dFaceValue1[sample, offsetF:] = dFaceValueI

                gLA2 = torch.column_stack(
                    (dminB[sample][:, offsetF:].transpose(1, 0), ofD2.transpose(1, 0), ofD2.transpose(1, 0))).cuda()

                final3A = torch.zeros(ITris.shape[0], frequency, 9, device=device, dtype=dtype)
                final3A.scatter_(2, masterIndexA, gLA2.unsqueeze(1).repeat(1, frequency, 1))
                final3A = final3A.view(ITris.shape[0] * frequency, 3, 3)
                # rightA = eigvectors[:, ITris].transpose(1, 0).contiguous().view(ITris.shape[0] * frequency, 3)
                F_iP2 = torch.zeros(ITris.shape[0], frequency, N, device=device, dtype=dtype)
                F_iP2.scatter_(2, ITris.unsqueeze(1).repeat(1, frequency, 1),
                               torch.bmm(rightA.unsqueeze(1), final3A).transpose(1, 2).view(ITris.shape[0], frequency,
                                                                                            3))
                del final3A, gLA2

                eigenvalue_gradient = torch.diagonal((eigvectors @ F_iP2.transpose(1, 2)), dim1=-2, dim2=-1).transpose(
                    0,
                    1).unsqueeze(
                    0)
                vectorizedF = (
                            (F_iP2.unsqueeze(2) @ eigvectors[:, :, None] * ATheta.transpose(1, 0)[:, :, None]).squeeze(
                                3) - F_iP2)
                del F_iP2

                vectorizedMu = (LMatrix @ vectorizedF.permute(1, 2, 0)).permute(2, 0, 1)
                del vectorizedF

                est = (vectorizedMu + (
                            torch.diagonal(-vectorizedMu @ ATheta, dim1=1, dim2=2)[:, :, None] * eigvectors[None, :,
                                                                                                 :])).transpose(0, 2)
                del vectorizedMu
                dFaceValueI = torch.nan_to_num(
                    torch.sum(torch.diagonal((hks[sample, :, offset:] @ (
                            partL @ eigenvalue_gradient + partR @ est).permute(2, 0, 1).float()), dim1=1, dim2=2),
                              1))

                perc01 = torch.quantile(dFaceValueI, 0.01, keepdim=True)
                perc99 = torch.quantile(dFaceValueI, 0.99, keepdim=True)
                dFaceValueI = torch.clip(dFaceValueI, min=perc01, max=perc99)
                dFaceValue2[sample, offsetF:] = dFaceValueI
            #
                if ALBOR:
                    gLA2 = torch.column_stack((drotB[sample][:, offsetF:].transpose(1, 0), ofDR.transpose(1, 0), ofDR.transpose(1, 0))).cuda()

                    final3A = torch.zeros(ITris.shape[0], frequency, 9, device=device, dtype=dtype)
                    final3A.scatter_(2, masterIndexA, gLA2.unsqueeze(1).repeat(1, frequency, 1))
                    final3A = final3A.view(ITris.shape[0] * frequency, 3, 3)
                    F_iP2 = torch.zeros(ITris.shape[0], frequency, N, device=device, dtype=dtype)
                    F_iP2.scatter_(2, ITris.unsqueeze(1).repeat(1, frequency, 1),
                                   torch.bmm(rightA.unsqueeze(1), final3A).transpose(1, 2).view(ITris.shape[0], frequency, 3))
                    del final3A, gLA2

                    eigenvalue_gradient = torch.diagonal((eigvectors @ F_iP2.transpose(1, 2)), dim1=-2, dim2=-1).transpose(0,
                                                                                                                           1).unsqueeze(
                        0)
                    vectorizedF = ((F_iP2.unsqueeze(2) @ eigvectors[:, :, None] * ATheta.transpose(1, 0)[:, :, None]).squeeze(
                        3) - F_iP2)
                    del F_iP2

                    vectorizedMu = (LMatrix @ vectorizedF.permute(1, 2, 0)).permute(2, 0, 1)
                    del vectorizedF

                    est = (vectorizedMu + (torch.diagonal(-vectorizedMu @ ATheta, dim1=1, dim2=2)[:, :, None] * eigvectors[None, :, :])).transpose(0, 2)
                    del vectorizedMu

                    dFaceValueI = torch.nan_to_num(
                        torch.sum(torch.diagonal((hks[sample, :, offset:] @ (partL @ eigenvalue_gradient + partR @ est).permute(2, 0, 1).float()), dim1=1, dim2=2), 1))
                    perc01 = torch.quantile(dFaceValueI, 0.01, keepdim=True)
                    perc99 = torch.quantile(dFaceValueI, 0.99, keepdim=True)
                    dFaceValueI = torch.clip(dFaceValueI, min=perc01, max=perc99)
                    dFaceValue3[sample, offsetF:] = dFaceValueI

            if Voronoi:
                indicesV = torch.arange(N, device=device)[:, None]
                gV = Au[:, None].cuda()

                dV = torch.diag(Au.cuda())

                final3 = (-eigvalues[:, None, None]*gV.repeat(frequency, 1, 1)).transpose(1,0).contiguous()
                final3 = final3.view(N * frequency, 1, 1)
                right = eigvectors[:, indicesV].transpose(1, 0).contiguous().view(N * frequency, 1, 1)

                indicesVR = indicesV.unsqueeze(1).repeat(1, frequency, 1)
                F_iP2 = torch.zeros(N, frequency, N, device=device, dtype=dtype)
                F_iP2.scatter_(2, indicesVR, torch.bmm(right, final3).transpose(1, 2).view(N, frequency, 1))
                del right, final3

                eigenvalue_gradient = torch.diagonal((eigvectors @ F_iP2.transpose(1, 2)), dim1=-2, dim2=-1).transpose(0, 1).unsqueeze(0)#-eigvalues[:, None]*
                vectorizedF = ((F_iP2.unsqueeze(2) @ eigvectors[:, :, None] * ATheta.transpose(1, 0)[:, :, None]).squeeze(
                    3) - F_iP2)
                del F_iP2

                vectorizedMu = (LMatrix @ vectorizedF.permute(1, 2, 0)).permute(2, 0, 1)
                del vectorizedF

                unfoldedEigs = eigvectors.transpose(0, 1).unsqueeze(0).repeat(N, 1, 1)
                vectorizedC2 = torch.diagonal(0.5 * (torch.gather(unfoldedEigs, 1, indicesVR.transpose(1,2)).transpose(1, 2) @ torch.diag_embed(torch.gather(dV, 1, indicesV)) @ torch.gather(unfoldedEigs, 1, indicesVR.transpose(1,2))),dim1=1, dim2=2)
                del unfoldedEigs
                est = (vectorizedMu + ((torch.diagonal(-vectorizedMu @ ATheta, dim1=1, dim2=2) - vectorizedC2)[:, :,
                                       None] * eigvectors[None, :, :])).transpose(0, 2)
                del vectorizedMu, indicesVR, indicesV, ATheta

                dVertexValueI = torch.nan_to_num(
                    torch.sum(torch.diagonal((hks[sample, :, offset:] @ (partL @ eigenvalue_gradient + partR @ est).float().permute(2, 0, 1)), dim1=1, dim2=2), 1))
                perc01 = torch.quantile(dVertexValueI, 0.01, keepdim=True)
                perc99 = torch.quantile(dVertexValueI, 0.99, keepdim=True)
                dVertexValueI = torch.clip(dVertexValueI, min=perc01, max=perc99)
                dVertexValue[sample, offset:] = dVertexValueI
                del gV, dV


            del est, A, eigvectors, eigvalues, LMatrix, eigenvalue_gradient, partL

        return None, None, None, dEdgeValue.unsqueeze(1), None, None, None, None, None, None, dFaceValue1, dFaceValue2, dFaceValue3, dVertexValue.unsqueeze(1) # for unit test transfer grad to cpu
