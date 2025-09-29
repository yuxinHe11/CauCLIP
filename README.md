# CauCLIP

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Official implementation of our ICASSP 2026 paper:  
**Causality-inspired Augmentation and Suppression for Surgical Phase Recognition**

---

## 📌 Introduction
This repository implements our framework for **surgical phase recognition** under domain shift.  
We build upon **CLIP** to bridge the gap between Virtual Reality (VR) training data and Porcine testing data in the **MICCAI SurgVisDom challenge**.  

Key contributions:
- **Frequency-based augmentation** to perturb domain-specific low-level style while preserving semantics.  
- **Causal suppression loss** to reduce spurious domain-specific noise and focus on surgical semantics.  
- Achieves **state-of-the-art performance** in the challenge’s **hard mode** (cross-domain evaluation).  

---

## 📂 Dataset Preparation

1. **Extract frames from each video**  
   - For each surgical video, create a folder named after the video.  
   - Store all frames (`.jpg` or `.png`) of that video inside the folder.  

   Example:
