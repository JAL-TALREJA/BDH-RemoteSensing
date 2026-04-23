<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/e/e8/IIIT_Allahabad_Logo.svg/200px-IIIT_Allahabad_Logo.svg.png" width="100"/>

# 🐉 Dragon Hatchling (BDH) for Remote Sensing Image Classification

### A Biologically-Inspired Vision Architecture on EuroSAT Sentinel-2 Imagery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tj4rxBh_Cy7AdSS1mcWueK2ezBf2j1d-)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/IIIT-Allahabad-8B0000)](https://www.iiita.ac.in/)
[![Dataset: EuroSAT](https://img.shields.io/badge/Dataset-EuroSAT-2ea44f)](https://github.com/phelber/EuroSAT)
[![arXiv](https://img.shields.io/badge/arXiv-2509.26507-b31b1b)](https://arxiv.org/abs/2509.26507)

---

**🏆 Test Accuracy: 92.07% &nbsp;|&nbsp; F1 Macro: 91.87% &nbsp;|&nbsp; Params: 2.45M &nbsp;|&nbsp; Inference: 0.58 ms/img**

| Field | Details |
|:---|:---|
| **Authors** | Jal Kumar Talreja (IIT2023047) · Daksh Bhatti (IIT2023099) |
| **Supervisors** | Prof. Pavan Chakraborty (IIIT-A) · Dr. Snigdha Sen (MIT, MAHE) |
| **Architecture** | BDH-GPU — Sparse ReLU-LowRank FFN + Hebbian Linear Attention |
| **Dataset** | EuroSAT — 27,000 Sentinel-2 RGB images, 10 land-use classes |
| **Base Paper** | Kosowski et al. (2025) *The Dragon Hatchling*, arXiv:2509.26507 |
| **Framework** | PyTorch 2.x · Google Colab T4 GPU |

</div>

---

## 📌 Overview

This project is the **first independent extension** of the Dragon Hatchling (BDH) biologically-inspired architecture — originally designed for language modelling — to the **remote sensing vision domain**.

We adapt BDH-GPU (Sparse ReLU-LowRank FFN + Hebbian Linear Attention) for **satellite image land-use/land-cover classification** on the EuroSAT Sentinel-2 benchmark, introducing original contributions across architecture adaptation, interpretability, comparative evaluation, and systematic ablation.

> **Originality Statement:** The BDH-GPU architecture is theoretically inspired by Kosowski et al. (2025), which covers *language modelling only*. All vision-domain adaptations, code, experiments, visualisations, and analyses are the **sole original work** of the authors.

---

## 🏆 Results at a Glance

<div align="center">

| Metric | Score |
|:---|:---|
| **Test Accuracy** | **92.07%** |
| **F1 Macro** | **91.87%** |
| **F1 Weighted** | **91.80%** |
| **Precision** | **91.75%** |
| **Recall** | **92.12%** |
| **Parameters** | **2.45M** |
| **Inference** | **0.58 ms/image** |

</div>

---

## 📊 Experimental Results

### BDH vs. Baseline Models

| Model | Test Acc | F1 Macro | Precision | Recall | Params | Inference |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SimpleCNN | 94.3% | 94.1% | 94.0% | 94.3% | ~0.5M | 0.47ms |
| ResNet-18 | 90.6% | 90.4% | 90.5% | 90.4% | 11.7M | 0.78ms |
| ViT-Baseline | 91.5% | 91.3% | 91.3% | 91.4% | ~5M | 0.71ms |
| **BDH (Ours)** | **92.1%** | **91.9%** | **91.8%** | **92.1%** | **2.45M** | **0.58ms** |

> ✅ BDH achieves **ViT-competitive accuracy** with **sub-quadratic O(Nd) linear attention** — using fewer parameters than both ResNet-18 and ViT.

### Per-Class Performance (Confusion Matrix — Test Acc: 92.1%)

| Class | Recall | Class | Recall |
|:---|:---:|:---|:---:|
| SeaLake | **0.97** | AnnualCrop | **0.93** |
| Forest | **0.96** | Highway | **0.93** |
| Industrial | **0.96** | Residential | **0.92** |
| Pasture | **0.95** | River | **0.89** |
| HerbaceousVegetation | **0.85** | PermanentCrop | **0.85** |

### Ablation Study — 7 Component Variants

| Variant | Test Acc | Drop |
|:---|:---:|:---:|
| **BDH Full (Baseline)** | **90.9%** | — |
| No LayerNorm | 91.5% | −0.6% |
| No Positional Encoding | 92.0% | −1.1% |
| No Hebbian Memory (ρ) | 89.4% | +1.4% |
| Half Depth (3 layers) | 88.4% | +2.5% |
| No Sparse ReLU (GELU) | 88.3% | +2.6% |
| Half Dim (d=128) | 87.2% | +3.6% |
| **Single Head (heads=1)** | **83.1%** | **+7.7% ← most critical** |

### Activation Sparsity (% Non-zero per Layer)

| Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 46.8% | 44.4% | 44.2% | 42.3% | 37.9% | **30.9%** |

> Sparsity increases with depth — the final layer reaches ~31%, promoting monosemantic, interpretable Hebbian representations.

### Synaptic State Strength (Hebbian ρ, Layer 3)

| Class | Mean \|ρ\| | Class | Mean \|ρ\| |
|:---|:---:|:---|:---:|
| SeaLake | **8.772** | Highway | 8.078 |
| AnnualCrop | 8.415 | River | 8.036 |
| Pasture | 8.204 | HerbaceousVegetation | 7.789 |
| Forest | 8.185 | Industrial | 7.735 |
| — | — | Residential | 7.198 |

> Spectrally distinct classes (SeaLake, AnnualCrop) develop stronger Hebbian in-context memory — evidence of **class-selective monosemanticity**.

---

## 🧠 Architecture

```
Input Image (64×64×3)
       │
   ┌───▼──────────────────────────────────┐
   │  Patch Embedding  (8×8 patches)      │  → 64 tokens per image
   │  + Sinusoidal Positional Encoding    │
   └──────────────────┬───────────────────┘
                      │
   ┌──────────────────▼────────────────────────┐
   │           BDH Block × 6                   │
   │  ┌─────────────────────────────────────┐  │
   │  │  ReLU-LowRank FFN  (~31–47% sparse) │  │
   │  │  Hebbian Linear Attention (ρ state) │  │
   │  │  Layer Norm + Residual Connection   │  │
   │  └─────────────────────────────────────┘  │
   └──────────────────┬────────────────────────┘
                      │
   ┌──────────────────▼──────────────┐
   │  [CLS] Token Readout            │
   │  MLP Head → 10-class softmax    │
   └──────────────────┬──────────────┘
                      │
             Land-Use Prediction

   Config: d=256 | 6 layers | 4 heads | patch=8×8
   Parameters: 2.45M | Attention complexity: O(Nd)
```

### Vision Adaptation Mapping

| BDH Paper Concept | This Work — Vision Adaptation |
|:---|:---|
| Neuron particles in high-dim space | Patch tokens from 8×8 image regions |
| Positive orthant constraint (R⁺)ⁿ | ReLU activations → sparse positive features |
| ReLU-LowRank FFN (~5% sparsity) | Feature extraction with biological threshold dynamics |
| Hebbian synaptic state ρ | In-context patch-correlation memory across layers |
| Causal linear attention (Eq. 8) | **Non-causal (bidirectional)** spatial patch attention |
| Last-token readout | [CLS] token → MLP classification head |

---

## 📦 Dataset — EuroSAT (Sentinel-2)

| Property | Details |
|:---|:---|
| **Total Images** | 27,000 geo-referenced satellite images |
| **Resolution** | 64×64 pixels (RGB) |
| **Classes** | 10 land-use / land-cover categories |
| **Source** | Sentinel-2 (European Space Agency) |
| **Split** | 75% Train / 15% Val / 10% Test |
| **Normalization** | Mean [0.344, 0.380, 0.408] · Std [0.204, 0.137, 0.115] |

---

## 🗂️ Repository Structure

```
BDH-RemoteSensing-IIITA/
│
├── 📓 BDH_RemoteSensing_IIITA_Final__1_.ipynb   ← Main notebook
├── 📄 README.md
├── 📄 LICENSE
│
└── 📁 results/
    ├── bdh_final_dashboard.png       ← Complete results dashboard
    ├── bdh_training_curves.png       ← Loss + accuracy + LR schedule
    ├── bdh_confusion_matrix.png      ← Per-class normalised confusion
    ├── bdh_attention_maps.png        ← Patch spatial saliency maps
    ├── bdh_synapse_strength.png      ← Hebbian ρ strength by class
    ├── bdh_sparsity.png              ← Activation sparsity per layer
    ├── comparative_study.png         ← BDH vs 3 baselines (6 panels)
    ├── scalability_graph.png         ← Training time vs sample count
    ├── ablation_study.png            ← 7-variant component ablation
    ├── eurosat_samples.png           ← One sample per class
    ├── bdh_report.json               ← Full results JSON export
    └── bdh_best.pt                   ← Best model checkpoint
```

---

## 🚀 Quick Start

### ▶ Run in Google Colab (Zero Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tj4rxBh_Cy7AdSS1mcWueK2ezBf2j1d-)

1. `Runtime → Change runtime type → T4 GPU → Save`
2. `Runtime → Run all` &nbsp;(or `Ctrl+F9`)
3. Everything installs and runs automatically — total time ~40–50 min

### 💻 Run Locally

```bash
git clone https://github.com/JAL-TALREJA/BDH-RemoteSensing-IIITA.git
cd BDH-RemoteSensing-IIITA

pip install torch torchvision einops timm matplotlib seaborn scikit-learn tqdm grad-cam

jupyter notebook BDH_RemoteSensing_IIITA_Final__1_.ipynb
```

---

## 📋 Notebook Sections

| # | Section | Content |
|:---:|:---|:---|
| 1 | **Environment Setup** | GPU check, auto-install, global config |
| 2 | **Dataset** | EuroSAT download, augmentation, loaders, visual exploration |
| 3 | **BDH Architecture** | ReLU-LowRank FFN, Hebbian attention, full model definition |
| 4 | **Training** | AdamW + cosine warmup, label smoothing, best-model checkpoint |
| 5 | **Evaluation** | Confusion matrix, per-class metrics, test accuracy |
| 6 | **Interpretability** | Activation sparsity, synaptic state, patch attention maps |
| 7 | **Comparative Study** | BDH vs SimpleCNN / ResNet-18 / ViT |
| 8 | **Scalability** | Training time vs. sample count (linear & log scale) |
| 9 | **Ablation Study** | 7 variants, component-wise drop analysis |
| 10 | **Final Dashboard** | Dark-theme dashboard + JSON export + model download |

---

## 🔬 Differences from the Base Paper

| Dimension | Kosowski et al. (2025) | This Work |
|:---|:---|:---|
| **Domain** | Language modelling | Remote sensing classification |
| **Input** | Characters / subwords | 8×8 image patches |
| **Attention** | Causal (autoregressive) | Non-causal (bidirectional) |
| **Readout** | Last token | [CLS] token |
| **Evaluation** | WikiText, translation | EuroSAT Sentinel-2 |
| **Interpretability** | None | Patch attention maps, sparsity |
| **Ablation** | None | 7-variant systematic study |
| **Comparison** | None | 4-model comprehensive benchmark |

---

## 🔧 Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Image size | 64×64 |
| Patch size | 8×8 |
| Embedding dim | 256 |
| Depth | 6 layers |
| Heads | 4 |
| Epochs | 40 |
| Batch size | 64 |
| Optimizer | AdamW (lr=3e-4, wd=0.05) |
| Schedule | Cosine Annealing + 5-epoch warmup |
| Label smoothing | 0.1 |
| Gradient clip | 1.0 |

---

## 📖 References

1. **Kosowski et al. (2025).** *The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain.* [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
2. **Dosovitskiy et al. (2021).** *An Image is Worth 16×16 Words.* ICLR 2021.
3. **Helber et al. (2019).** *EuroSAT: A Novel Dataset and Deep Learning Benchmark.* IEEE JSTARS.
4. **He et al. (2016).** *Deep Residual Learning for Image Recognition.* CVPR 2016.
5. **Selvaraju et al. (2017).** *GradCAM: Visual Explanations from Deep Networks.* ICCV 2017.

---

## 👥 Authors

**Jal Kumar Talreja** · **Daksh Bhatti**

Supervised by **Prof. Pavan Chakraborty** (IIIT Allahabad) · **Dr. Snigdha Sen** (MIT, MAHE, Bengaluru)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*IIIT Allahabad · 2025*

**⭐ If this work is useful, please star the repository!**

</div>
