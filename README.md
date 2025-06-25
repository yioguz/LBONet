# LBONet: Supervised Spectral Descriptors for Shape Analysis

![LBONet Architecture](arch.png)

**Authors**: [Oguzhan Yigit](https://oguzhanyigit.com) and [Richard C. Wilson](https://sites.google.com/york.ac.uk/richard-wilson/home)  
**GitHub**: [yioguz/LBONet](https://github.com/yioguz/LBONet)  
**Paper**: [arXiv:2411.08272](https://arxiv.org/abs/2411.08272)

LBONet is a PyTorch-based framework that learns task-specific modifications to the Laplace–Beltrami operator (LBO) for intrinsic shape analysis on 3D surfaces and meshes. It adapts the LBO eigenbasis to improve performance across various geometric learning tasks.

---

## 🧠 Overview

LBONet modifies the eigenbasis of the Laplace–Beltrami operator by learning geometric parameters over mesh elements. This improves spectral descriptors for tasks such as:

- Shape retrieval and classification  
- Shape segmentation  
- Dense shape correspondence  

It introduces **learnable modules** that deform the geometry and local operators to steer the spectral embedding.

---

## 🔧 Key Modules

- **RiemannNet** – learns edge-based metric weights (affects stiffness matrix)  
- **ALBONet / ALBO+Net** – learns anisotropic weights and directions on faces  
- **VoronoiNet** – learns vertex-based area weights (affects mass matrix)

These are combined to solve a modified LBO eigenproblem in a **differentiable** and **task-driven** way.

---

## 🔍 Features

- ✅ Differentiable eigendecomposition (custom CPU-based solver with PyTorch autograd)
- ✅ Modular architecture (easy to plug in new geometry learners)
- ✅ Supports classical and neural descriptor backends (e.g. DiffusionNet, Pointnet++, etc.)
- ✅ Efficient sparse matrix operations
- ✅ Strong performance on non-rigid shape benchmarks

---

## 📊 Benchmarks

LBONet achieves state-of-the-art or competitive results on:

- **SHREC’11/14/15 ShapeNetCore55** – classification and retrieval  
- **FAUST** – dense correspondence  
- **COSEG / Human Segmentation / ShapeNet Part** – segmentation  

Includes detailed ablation studies showing the effect of each module.

---

## 🚀 Getting Started

```bash
git clone https://github.com/yioguz/LBONet.git
cd LBONet
# Follow setup instructions in the README or environment.yml
