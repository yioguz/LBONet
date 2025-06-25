# LBONet: Supervised Spectral Descriptors for Shape Analysis

**GitHub:** [yioguz/LBONet](https://github.com/yioguz/LBONet)  
**Paper:** [arXiv:2411.08272](https://arxiv.org/abs/2411.08272)

LBONet is a PyTorch-based framework that learns task-specific modifications to the Laplace–Beltrami operator (LBO) for intrinsic shape analysis on 3D surfaces and meshes. It adapts the LBO eigenbasis to improve performance across various geometric learning tasks.

## 🔧 Key Modules

- **RiemannNet**: Learns edge-based metric weights to modify the stiffness matrix.
- **ALBONet / ALBO+Net**: Learns anisotropic diffusion directions and intensities on mesh faces.
- **VoronoiNet**: Learns per-vertex area weights, modifying the mass matrix.

These components adjust the generalized eigenproblem of the LBO in a differentiable way.

## 🧠 Technical Highlights

- Differentiable eigendecomposition with sparse matrix support.
- Modular neural network design based on geometric priors.
- Compatible with standard descriptor heads (e.g., DiffusionNet).
- Supports classification, segmentation, retrieval, and correspondence tasks.

## 📊 Benchmarks

LBONet achieves strong results on several datasets:
- **SHREC’11/14/15** – perfect or near-perfect retrieval/classification.
- **FAUST** – top-performing shape correspondence (low geodesic error).
- **COSEG, Human Segmentation** – outperforms baselines on segmentation.

## 🚀 Getting Started

Clone the repo and install dependencies:
```bash
git clone https://github.com/yioguz/LBONet.git
cd LBONet
# follow README instructions for environment setup
