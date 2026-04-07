# Diverse Text-to-Image Generation via Contrastive Noise Optimization (ICLR 2026)

[![Paper](https://img.shields.io/badge/arXiv-2510.03813-b31b1b.svg)](https://arxiv.org/abs/2510.03813)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]Official PyTorch implementation of **"DIVERSE TEXT-TO-IMAGE GENERATION VIA CONTRASTIVE NOISE OPTIMIZATION"**[cite: 3, 4]. 
[cite_start]By Byungjun Kim, Soobin Um, and Jong Chul Ye. 
[cite_start]Published as a conference paper at ICLR 2026.

<p align="center">
  <img src="assets/teaser.png" alt="Teaser Image" width="100%">
  <br>
  <em>Figure 1: Example results from our diverse image generation approach. Standard DDIM exhibits pronounced mode collapse, producing repetitive images. [cite_start]Our method delivers markedly greater diversity and fidelity, generating a wide range of images that remain strongly aligned with the input text.</em>
</p>

## 📖 Abstract
[cite_start]Text-to-image (T2I) diffusion models have demonstrated impressive performance in generating high-fidelity images, largely enabled by text-guided inference[cite: 19]. [cite_start]However, this advantage often comes with a critical drawback: limited diversity, as outputs tend to collapse into similar modes under strong text guidance[cite: 20].

[cite_start]In this work, we introduce **Contrastive Noise Optimization (CNO)**, a simple yet effective method that addresses the diversity issue from a distinct perspective[cite: 22]. [cite_start]Unlike prior techniques that adapt intermediate latents, our approach shapes the initial noise to promote diverse outputs[cite: 23]. [cite_start]Specifically, we develop a contrastive loss defined in the Tweedie data space and optimize a batch of noise latents[cite: 24]. [cite_start]Extensive experiments across multiple T2I backbones demonstrate that our approach achieves a superior quality-diversity Pareto frontier while remaining robust to hyperparameter choices[cite: 27].

## 🚀 Getting Started

### Prerequisites
```bash
# Clone the repository
git clone [https://github.com/USERNAME/CNO.git](https://github.com/USERNAME/CNO.git)
cd CNO

# Create a conda environment
conda create -n cno python=3.10
conda activate cno

# Install dependencies
pip install -r requirements.txt
