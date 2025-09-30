# Surgical Phase Recognition with CLIP

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
 
**CauCLIP: Bridging the Sim-to-real Gap in Surgical Video Understanding
Via Causality-inspired Vision-Language Modeling**

---

## 📌 Introduction
This repository implements our framework for **surgical phase recognition** under domain shift.  
We build upon **CLIP** to bridge the gap between Virtual Reality (VR) training data and Porcine testing data in the **MICCAI2020 SurgVisDom challenge**.  

Key contributions:
- **Frequency-based augmentation** to perturb domain-specific low-level style while preserving semantics.  
- **Causal suppression loss** to reduce spurious domain-specific noise and focus on surgical semantics.  
- Achieves **state-of-the-art performance** in the challenge’s **hard mode** (cross-domain evaluation).  

---

## 📂 Dataset Preparation

1. **Extract frames from each video**  
   - For each surgical video, create a folder named after the video.  
   - Store all frames (`.jpg` or `.png`) of that video inside the folder.  
Example：
   <pre>
data/
├── video_001/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── ...
├── video_002/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── ...
```
</pre>


3. **Create training lists**  
- In the `lists/` directory, create `.txt` files to specify the dataset split.  
- Each line in a `.txt` file should contain:
  ```
  <path> <num_frames> <label>
  ```
- Example:
  ```
  data/video_001 128 0
  data/video_002 64  1
  ```

where:
- `<path>`: relative path to the video frame folder  
- `<num_frames>`: total number of frames in the folder  
- `<label>`: ground-truth class label for the video  

---



