import torch
import torch.nn as nn

from LBOEigendecompositionLayer import Spectral as SpectralImplicit
from LBONet.geometry import get_graph_feature
import numpy as np



class LBONetImplicitRetrieval(nn.Module):

    def __init__(self, Riemann, Anisotropy, Voronoi, debug):
        """
        Construct an LBONet.

        Parameters:
            Riemann (int):    Riemann dimension
            Anisotropy (int): Anisotropy dimension
            Voronoi (int):    Voronoi dimension
        """
        super(LBONetImplicitRetrieval, self).__init__()

        # --- Basic Parameters ---
        self.neighbours = 40
        self.RiemannFreq = Riemann
        self.VoronoiFreq = Voronoi
        self.AnisotropyFreq = Anisotropy
        self.debug = debug

        # --- Activation & Dropout ---
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.dropA = nn.Dropout(0.2)
        self.dropV = nn.Dropout(0.2)
        self.dropR = nn.Dropout(0.2)

        # --- Learnable Scale Parameters ---
        self.aniso_scale = nn.Parameter(torch.tensor(1.0))
        self.voronoi_scale = nn.Parameter(torch.tensor(1.0))
        self.riemann_scale = nn.Parameter(torch.tensor(0.01))

        # === Riemann Block ===
        # BatchNorm
        self.n0b = nn.BatchNorm1d(24)
        self.n2 = nn.BatchNorm1d(Riemann)
        self.n3 = nn.BatchNorm1d(Riemann)
        self.n4 = nn.BatchNorm1d(Riemann)
        self.n5 = nn.BatchNorm1d(Riemann)
        self.rcn1 = nn.BatchNorm2d(Riemann)
        self.rcn2 = nn.BatchNorm2d(Riemann)

        # Convolutions
        self.rconv0b = nn.Sequential(
            nn.Conv1d(18, 24, kernel_size=1, bias=False),
            self.n0b,
            self.activation
        )
        self.rdconv1b = nn.Sequential(
            nn.Conv2d(48, Riemann, kernel_size=1, bias=False),
            self.rcn1,
            self.activation
        )
        self.rconv2 = nn.Sequential(
            nn.Conv1d(Riemann, Riemann, kernel_size=1, bias=False),
            self.n2,
            self.activation
        )
        self.rconv3 = nn.Sequential(
            nn.Conv1d(Riemann * 2 + 16, Riemann, kernel_size=1, bias=False),
            self.n3,
            self.activation
        )
        self.rconv6 = nn.Sequential(
            nn.Conv1d(Riemann, 1, kernel_size=1, bias=True)
        )

        # === Voronoi Block ===
        # BatchNorm
        self.v0 = nn.BatchNorm1d(24)
        self.v1 = nn.BatchNorm1d(8)
        self.v2 = nn.BatchNorm1d(Voronoi * 2)
        self.v3 = nn.BatchNorm1d(Voronoi * 3)
        self.v4 = nn.BatchNorm1d(Voronoi)
        self.v5 = nn.BatchNorm1d(Voronoi)
        self.abn1 = nn.BatchNorm2d(Voronoi)
        self.abn2 = nn.BatchNorm2d(Voronoi)

        # Convolutions
        self.vvconv0 = nn.Sequential(
            nn.Conv1d(13, 24, kernel_size=1, bias=False),
            self.v0,
            self.activation
        )
        self.vdconv1 = nn.Sequential(
            nn.Conv2d(48, Voronoi, kernel_size=1, bias=False),
            self.abn1,
            self.activation
        )
        self.vconv2 = nn.Sequential(
            nn.Conv1d(Voronoi, Voronoi * 2, kernel_size=1, bias=False),
            self.v2,
            self.activation
        )
        self.vvconv3 = nn.Sequential(
            nn.Conv1d(Voronoi * 3, Voronoi * 3, kernel_size=1, bias=False),
            self.v3,
            self.activation
        )
        self.vvconv4 = nn.Sequential(
            nn.Conv1d(Voronoi * 6 + 32, Voronoi, kernel_size=1, bias=False),
            self.v4,
            self.activation
        )
        self.vconv6 = nn.Sequential(
            nn.Conv1d(Voronoi, 1, kernel_size=1, bias=True),
            self.activation
        )

        # === Anisotropy Block ===
        # BatchNorm
        self.an0 = nn.BatchNorm1d(23)
        self.an1 = nn.BatchNorm1d(8)
        self.an2 = nn.BatchNorm1d(Anisotropy * 2)
        self.an3 = nn.BatchNorm1d(Anisotropy * 3)
        self.an4 = nn.BatchNorm1d(Anisotropy)
        self.an5 = nn.BatchNorm1d(Anisotropy)
        self.an6 = nn.BatchNorm1d(2)
        self.acn1 = nn.BatchNorm2d(Anisotropy)
        self.acn2 = nn.BatchNorm2d(Anisotropy)

        # Convolutions
        self.aaconv0 = nn.Sequential(
            nn.Conv1d(13, 23, kernel_size=1, bias=False),
            self.an0,
            self.activation
        )
        self.adconv1 = nn.Sequential(
            nn.Conv2d(46, Anisotropy, kernel_size=1, bias=False),
            self.acn1,
            self.activation
        )
        self.aconv1 = nn.Sequential(
            nn.Linear(4, 8),
            self.an1,
            self.activation
        )
        self.aconv2 = nn.Sequential(
            nn.Conv1d(Anisotropy, Anisotropy * 2, kernel_size=1, bias=False),
            self.an2,
            self.activation
        )
        self.aaconv3 = nn.Sequential(
            nn.Conv1d(Anisotropy * 3, Anisotropy * 3, kernel_size=1, bias=False),
            self.an3,
            self.activation
        )
        self.aaconv4 = nn.Sequential(
            nn.Conv1d(Anisotropy * 6 + 32, Anisotropy, kernel_size=1, bias=False),
            self.an4,
            self.activation
        )
        self.aconv6 = nn.Sequential(
            nn.Conv1d(Anisotropy, 2, kernel_size=1, bias=True),
            self.activation
        )
        self.aconv7 = nn.Sequential(
            nn.Conv1d(Anisotropy, 1, kernel_size=1, bias=False),
            nn.Tanh()
        )

    def forward(self, vertices, faces, edges, feature_vector, feature_vectorP, feature_vectorf, el, ts, corners,
                minCurvature, maxCurvature, rotationNormal, cache=False, rewrite=False):
        batch_size = vertices.shape[0]

        # Transpose inputs
        vertices = vertices.transpose(1, 2)
        edges = edges.transpose(1, 2)
        faces = faces.transpose(1, 2)

        ## --- Riemann Branch --- ##
        xyzE = feature_vector[:, 4:7, :].transpose(1, 2)
        riemann = feature_vector.clone()
        riemann[:, 4:10, :] = 0  # zero out region
        riemann = self.rconv0b(riemann)

        riemann = get_graph_feature(xyzE, riemann, k=self.neighbours)
        riemann = self.rdconv1b(riemann)
        riemann = riemann.max(dim=-1)[0]

        riemann = self.rconv2(riemann)
        ax = torch.nn.functional.adaptive_avg_pool1d(riemann, 1).view(batch_size, -1)
        riemann = torch.cat((riemann, ax[:, :, None].repeat(1, 1, riemann.shape[2])), dim=1)
        riemann = self.rconv3(riemann)
        riemann = self.dropR(riemann)
        riemann = self.rconv6(riemann)
        riemann = torch.tanh(riemann) * 0.5
        riemann = (riemann.view(batch_size, -1, 1).transpose(1, 2) / 10) * self.riemann_scale

        ## --- Anisotropy Branch --- ##
        xyzF = feature_vectorf[:, 4:7, :].transpose(1, 2)
        ffwc = torch.cat((feature_vectorf[:, :4, :], feature_vectorf[:, 10:, :]), dim=1)

        anisotropy = self.aaconv0(ffwc)
        anisotropy = get_graph_feature(xyzF, anisotropy, k=self.neighbours)
        anisotropy = self.adconv1(anisotropy)
        anisotropy1 = anisotropy.max(dim=-1)[0]

        anisotropy = self.aconv2(anisotropy1)
        anisotropy = torch.cat((anisotropy1, anisotropy), dim=1)
        anisotropy = self.aaconv3(anisotropy)

        ax = torch.nn.functional.adaptive_avg_pool1d(anisotropy, 1).view(batch_size, -1)
        anisotropy = torch.cat((anisotropy, ax[:, :, None].repeat(1, 1, anisotropy.shape[2])), dim=1)

        anisotropyR = self.aaconv4(anisotropy)
        anisotropyR = self.dropA(anisotropyR)

        anisotropy = torch.clamp(self.aconv6(anisotropyR) / self.aniso_scale + 1, min=0.1)
        Theta = self.aconv7(anisotropyR) * np.pi

        ## --- Voronoi Branch --- ##
        verts = vertices
        vfwc = torch.cat((feature_vectorP[:, :4, :], feature_vectorP[:, 7:, :]), dim=1)

        voronoi = self.vvconv0(vfwc)
        voronoi = get_graph_feature(verts, voronoi, k=self.neighbours)
        voronoi = self.vdconv1(voronoi)
        voronoi1 = voronoi.max(dim=-1)[0]

        voronoi = self.vconv2(voronoi1)
        voronoi = torch.cat((voronoi1, voronoi), dim=1)
        voronoi = self.vvconv3(voronoi)

        vx = torch.nn.functional.adaptive_avg_pool1d(voronoi, 1).view(batch_size, -1)
        voronoi = torch.cat((voronoi, vx[:, :, None].repeat(1, 1, voronoi.shape[2])), dim=1)

        voronoi = self.vvconv4(voronoi)
        voronoi = self.dropV(voronoi)
        voronoi = self.vconv6(voronoi)
        voronoi = torch.clamp(voronoi / self.voronoi_scale + 1, min=0.01)

        ## --- Spectral Module --- ##
        hks = SpectralImplicit.apply(
            vertices, edges, faces, riemann, el, ts, corners,
            minCurvature, maxCurvature, rotationNormal,
            anisotropy[:, 0, :], anisotropy[:, 1, :],
            Theta[:, 0, :], voronoi, cache, rewrite
        )

        return hks



class LBONetImplicit(nn.Module):

    def __init__(self, Riemann, Anisotropy, Voronoi, debug):
        """
        Construct an LBONet.

        Parameters:
            Riemann (int):    Riemann dimension
            Anisotropy (int): Anisotropy dimension
            Voronoi (int):    Voronoi dimension
        """
        super(LBONetImplicit, self).__init__()

        # --- Basic Parameters ---
        self.neighbours = 40
        self.RiemannFreq = Riemann
        self.VoronoiFreq = Voronoi
        self.AnisotropyFreq = Anisotropy
        self.debug = debug

        # --- Activation & Dropout ---
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.dropA = nn.Dropout(0.2)
        self.dropV = nn.Dropout(0.2)
        self.dropR = nn.Dropout(0.2)

        # --- Learnable Scale Parameters ---
        self.aniso_scale = nn.Parameter(torch.tensor(1.0))
        self.voronoi_scale = nn.Parameter(torch.tensor(1.0))
        self.riemann_scale = nn.Parameter(torch.tensor(0.01))

        # === Riemann Block ===
        # BatchNorm
        self.n0b = nn.BatchNorm1d(24)
        self.n2 = nn.BatchNorm1d(Riemann)
        self.n3 = nn.BatchNorm1d(Riemann)
        self.n4 = nn.BatchNorm1d(Riemann)
        self.n5 = nn.BatchNorm1d(Riemann)
        self.rcn1 = nn.BatchNorm2d(Riemann)
        self.rcn2 = nn.BatchNorm2d(Riemann)

        # Convolutions
        self.rconv0b = nn.Sequential(
            nn.Conv1d(18, 24, kernel_size=1, bias=False),
            self.n0b,
            self.activation
        )
        self.rdconv1b = nn.Sequential(
            nn.Conv2d(48, Riemann, kernel_size=1, bias=False),
            self.rcn1,
            self.activation
        )
        self.rconv2 = nn.Sequential(
            nn.Conv1d(Riemann, Riemann, kernel_size=1, bias=False),
            self.n2,
            self.activation
        )
        self.rconv3 = nn.Sequential(
            nn.Conv1d(Riemann * 2 + 16, Riemann, kernel_size=1, bias=False),
            self.n3,
            self.activation
        )
        self.rconv6 = nn.Sequential(
            nn.Conv1d(Riemann, 1, kernel_size=1, bias=True)
        )

        # === Voronoi Block ===
        # BatchNorm
        self.v0 = nn.BatchNorm1d(24)
        self.v1 = nn.BatchNorm1d(8)
        self.v2 = nn.BatchNorm1d(Voronoi * 2)
        self.v3 = nn.BatchNorm1d(Voronoi * 3)
        self.v4 = nn.BatchNorm1d(Voronoi)
        self.v5 = nn.BatchNorm1d(Voronoi)
        self.abn1 = nn.BatchNorm2d(Voronoi)
        self.abn2 = nn.BatchNorm2d(Voronoi)

        # Convolutions
        self.vvconv0 = nn.Sequential(
            nn.Conv1d(13, 24, kernel_size=1, bias=False),
            self.v0,
            self.activation
        )
        self.vdconv1 = nn.Sequential(
            nn.Conv2d(48, Voronoi, kernel_size=1, bias=False),
            self.abn1,
            self.activation
        )
        self.vconv2 = nn.Sequential(
            nn.Conv1d(Voronoi, Voronoi * 2, kernel_size=1, bias=False),
            self.v2,
            self.activation
        )
        self.vvconv3 = nn.Sequential(
            nn.Conv1d(Voronoi * 3, Voronoi * 3, kernel_size=1, bias=False),
            self.v3,
            self.activation
        )
        self.vvconv4 = nn.Sequential(
            nn.Conv1d(Voronoi * 6 + 32, Voronoi, kernel_size=1, bias=False),
            self.v4,
            self.activation
        )
        self.vconv6 = nn.Sequential(
            nn.Conv1d(Voronoi, 1, kernel_size=1, bias=True),
            self.activation
        )

        # === Anisotropy Block ===
        # BatchNorm
        self.an0 = nn.BatchNorm1d(23)
        self.an1 = nn.BatchNorm1d(8)
        self.an2 = nn.BatchNorm1d(Anisotropy * 2)
        self.an3 = nn.BatchNorm1d(Anisotropy * 3)
        self.an4 = nn.BatchNorm1d(Anisotropy)
        self.an5 = nn.BatchNorm1d(Anisotropy)
        self.an6 = nn.BatchNorm1d(2)
        self.acn1 = nn.BatchNorm2d(Anisotropy)
        self.acn2 = nn.BatchNorm2d(Anisotropy)

        # Convolutions
        self.aaconv0 = nn.Sequential(
            nn.Conv1d(13, 23, kernel_size=1, bias=False),
            self.an0,
            self.activation
        )
        self.adconv1 = nn.Sequential(
            nn.Conv2d(46, Anisotropy, kernel_size=1, bias=False),
            self.acn1,
            self.activation
        )
        self.aconv1 = nn.Sequential(
            nn.Linear(4, 8),
            self.an1,
            self.activation
        )
        self.aconv2 = nn.Sequential(
            nn.Conv1d(Anisotropy, Anisotropy * 2, kernel_size=1, bias=False),
            self.an2,
            self.activation
        )
        self.aaconv3 = nn.Sequential(
            nn.Conv1d(Anisotropy * 3, Anisotropy * 3, kernel_size=1, bias=False),
            self.an3,
            self.activation
        )
        self.aaconv4 = nn.Sequential(
            nn.Conv1d(Anisotropy * 6 + 32, Anisotropy, kernel_size=1, bias=False),
            self.an4,
            self.activation
        )
        self.aconv6 = nn.Sequential(
            nn.Conv1d(Anisotropy, 2, kernel_size=1, bias=True),
            self.activation
        )
        self.aconv7 = nn.Sequential(
            nn.Conv1d(Anisotropy, 1, kernel_size=1, bias=False),
            nn.Tanh()
        )

    def forward(self, vertices, faces, edges, feature_vector, feature_vectorP, feature_vectorf, el, ts, corners,
                minCurvature, maxCurvature, rotationNormal, cats, cache=False, rewrite=False):
        batch_size = vertices.shape[0]

        # Transpose inputs
        vertices = vertices.transpose(1, 2)
        edges = edges.transpose(1, 2)
        faces = faces.transpose(1, 2)

        ## --- Riemann Branch --- ##
        xyzE = feature_vector[:, 4:7, :].transpose(1, 2)
        riemann = feature_vector.clone()
        riemann[:, 4:10, :] = 0  # zero out region
        riemann = self.rconv0b(riemann)

        riemann = get_graph_feature(xyzE, riemann, k=self.neighbours)
        riemann = self.rdconv1b(riemann)
        riemann = riemann.max(dim=-1)[0]

        riemann = self.rconv2(riemann)
        ax = torch.nn.functional.adaptive_avg_pool1d(riemann, 1).view(batch_size, -1)
        riemann = torch.cat((riemann, ax[:, :, None].repeat(1, 1, riemann.shape[2])), dim=1)
        riemann = torch.cat((riemann, cats.unsqueeze(2).repeat(1, 1, riemann.shape[2])), dim=1)
        riemann = self.rconv3(riemann)
        riemann = self.dropR(riemann)
        riemann = self.rconv6(riemann)
        riemann = torch.tanh(riemann) * 0.5
        riemann = (riemann.view(batch_size, -1, 1).transpose(1, 2) / 10) * self.riemann_scale

        ## --- Anisotropy Branch --- ##
        xyzF = feature_vectorf[:, 4:7, :].transpose(1, 2)
        ffwc = torch.cat((feature_vectorf[:, :4, :], feature_vectorf[:, 10:, :]), dim=1)

        anisotropy = self.aaconv0(ffwc)
        anisotropy = get_graph_feature(xyzF, anisotropy, k=self.neighbours)
        anisotropy = self.adconv1(anisotropy)
        anisotropy1 = anisotropy.max(dim=-1)[0]

        anisotropy = self.aconv2(anisotropy1)
        anisotropy = torch.cat((anisotropy1, anisotropy), dim=1)
        anisotropy = self.aaconv3(anisotropy)

        anisotropy = torch.cat((anisotropy, cats.unsqueeze(2).repeat(1, 1, anisotropy.shape[2])), dim=1)
        ax = torch.nn.functional.adaptive_avg_pool1d(anisotropy, 1).view(batch_size, -1)
        anisotropy = torch.cat((anisotropy, ax[:, :, None].repeat(1, 1, anisotropy.shape[2])), dim=1)

        anisotropyR = self.aaconv4(anisotropy)
        anisotropyR = self.dropA(anisotropyR)

        anisotropy = torch.clamp(self.aconv6(anisotropyR) / self.aniso_scale + 1, min=0.1)
        Theta = self.aconv7(anisotropyR) * np.pi

        ## --- Voronoi Branch --- ##
        verts = vertices
        vfwc = torch.cat((feature_vectorP[:, :4, :], feature_vectorP[:, 7:, :]), dim=1)

        voronoi = self.vvconv0(vfwc)
        voronoi = get_graph_feature(verts, voronoi, k=self.neighbours)
        voronoi = self.vdconv1(voronoi)
        voronoi1 = voronoi.max(dim=-1)[0]

        voronoi = self.vconv2(voronoi1)
        voronoi = torch.cat((voronoi1, voronoi), dim=1)
        voronoi = self.vvconv3(voronoi)

        voronoi = torch.cat((voronoi, cats.unsqueeze(2).repeat(1, 1, voronoi.shape[2])), dim=1)
        vx = torch.nn.functional.adaptive_avg_pool1d(voronoi, 1).view(batch_size, -1)
        voronoi = torch.cat((voronoi, vx[:, :, None].repeat(1, 1, voronoi.shape[2])), dim=1)

        voronoi = self.vvconv4(voronoi)
        voronoi = self.dropV(voronoi)
        voronoi = self.vconv6(voronoi)
        voronoi = torch.clamp(voronoi / self.voronoi_scale + 1, min=0.01)

        ## --- Spectral Module --- ##
        hks = SpectralImplicit.apply(
            vertices, edges, faces, riemann, el, ts, corners,
            minCurvature, maxCurvature, rotationNormal,
            anisotropy[:, 0, :], anisotropy[:, 1, :],
            Theta[:, 0, :], voronoi, cache, rewrite
        )

        return hks

