# Cherry Pit Detection using SWIR Hyperspectral Transmission Imaging (900–1700 nm)

<!-- =========================

========================= -->
[![Code](https://img.shields.io/badge/code-available-brightgreen.svg)](https://github.com/AgFood-Sensing-and-Intelligence-Lab/cherry-pit-detetction-HSI/tree/main/Code)
[![Data](https://img.shields.io/badge/data-coming%20soon-orange.svg)](../../)
[![License](https://img.shields.io/github/license/AgFood-Sensing-and-Intelligence-Lab/cherry-pit-detetction-HSI?label=license)](https://github.com/AgFood-Sensing-and-Intelligence-Lab/cherry-pit-detetction-HSI/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-Food%20Bioscience-6f42c1.svg)](https://www.sciencedirect.com/science/article/abs/pii/S221242922502379X)


This repository provides the **code and supporting materials** for our study on **non-destructive pit detection in cherries** using **shortwave infrared (SWIR) hyperspectral transmission imaging** (900–1700 nm).

**Paper:** *Non-destructive detection of pits in cherries using shortwave infrared hyperspectral transmission imaging*  
**Authors:** Naseeb Singh, Yuzhen Lu  
**Affiliation:** Department of Biosystems & Agricultural Engineering, Michigan State University, USA


---

## 1. Introduction

Pit fragments or residual pits in processed cherries reduce market value and can create food safety concerns. Conventional inspection methods are limited by speed, subjectivity, and/or the inability to reliably detect internal defects. This work investigates **SWIR hyperspectral imaging in transmission mode** as a rapid, non-destructive approach for detecting pits in cherries by learning discriminative spectral signatures linked to the presence or absence of pits.

---

## 2. What this work does

We developed a complete workflow to classify cherries as **intact** or **pitted** using SWIR hyperspectral transmission imaging:

- Acquire SWIR hyperspectral images in **transmittance mode** over **900–1700 nm**
- Scan cherries from **two cultivars**, captured in **two orientations (0° and 90°)**
- Collect images **before and after pitting** to generate intact vs pitted labels
- Segment cherry regions and extract **mean transmittance spectra** from the ROI
- Perform wavelength reduction using:
  - **biPLS** for informative **interval selection**
  - **CARS** and **SPA** for key wavelength selection
  - **Genetic Algorithm (GA)** for compact wavelength optimization
- Train and evaluate **Random Forest (RF)** and **Support Vector Machine (SVM)** models for classification

---

## 3. Main findings (from the paper)

- The **1007–1057 nm** interval was the most informative region and enabled **perfect classification** (accuracy, precision, recall, F1-score = **1.0**) using both RF and SVM.
- A compact wavelength set (**1007, 1014, 1032, 1236 nm**) achieved **the same perfect performance**, supporting reduced-band implementations.
- Longer wavelength regions **at and above 1436 nm** were substantially less informative, with performance dropping to approximately **50–67% accuracy** in some intervals.
- The approach demonstrated **orientation-invariant** performance across **0° and 90°** scanning orientations.

---

## 4. Repository contents

```
.
├─ data/
│  ├─ raw/                   # raw hyperspectral cubes (optional if shared)
│  ├─ processed/             # extracted spectra and labels
│  └─ README.md              # dataset notes (optional)
├─ src/
│  ├─ segmentation/          # ROI extraction scripts (e.g., SAM-based)
│  ├─ preprocessing/         # spectral preprocessing utilities
│  ├─ wavelength_selection/  # biPLS, CARS, SPA, GA scripts
│  ├─ modeling/              # RF/SVM training and evaluation
│  └─ utils/
├─ notebooks/                # optional experiments
├─ assets/                   # figures/screenshots used in this README
├─ requirements.txt
└─ README.md
```



---

## 5. Installation

### 5.1 Create a Python environment
```bash
conda create -n cherry_pit_hsi python=3.10 -y
conda activate cherry_pit_hsi
```

### 5.2 Install dependencies
```bash
pip install -r requirements.txt
```

Typical packages include:
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `scikit-learn`
- `opencv-python`
- `spectral`
- PyTorch + SAM (Segment Anything Model)

---

## 6. Data

Place raw hyperspectral data (if you are sharing cubes) under:
```
data/raw/
```

Recommended processed outputs:
```
data/processed/
  ├─ spectra.csv              # samples × wavelengths matrix
  ├─ labels.csv               # intact=0, pitted=1 (+ metadata columns if available)
  └─ qc/                      # optional ROI masks/overlays for quality checking
```

If raw cubes are too large to host, share extracted spectra (`spectra.csv`) and labels (`labels.csv`) and keep scripts for reproducing extraction from cubes.

---

## 7. How to run (end-to-end)

### Step 1 — ROI segmentation and spectral extraction
Example (update to your script name):
```bash
python src/segmentation/extract_roi_spectra.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --method sam
```

Outputs typically include:
- `data/processed/spectra.csv`
- `data/processed/labels.csv`
- optional QC masks/overlays under `data/processed/qc/`

---

### Step 2 — Interval selection using biPLS
```bash
python src/wavelength_selection/run_bipls.py \
  --spectra data/processed/spectra.csv \
  --labels data/processed/labels.csv \
  --out_dir results/bipls
```

---

### Step 3 — Key wavelength selection using CARS, SPA, and GA
```bash
python src/wavelength_selection/run_cars_spa_ga.py \
  --spectra data/processed/spectra.csv \
  --labels data/processed/labels.csv \
  --out_dir results/wavelengths
```

---

### Step 4 — Train and evaluate RF/SVM models

#### (A) Full spectrum
```bash
python src/modeling/train_eval.py \
  --spectra data/processed/spectra.csv \
  --labels data/processed/labels.csv \
  --input_type full \
  --model svm \
  --out_dir results/models
```

#### (B) Best interval (1007–1057 nm)
```bash
python src/modeling/train_eval.py \
  --spectra data/processed/spectra.csv \
  --labels data/processed/labels.csv \
  --input_type interval \
  --interval 1007 1057 \
  --model rf \
  --out_dir results/models
```

#### (C) Key wavelengths (1007, 1014, 1032, 1236 nm)
```bash
python src/modeling/train_eval.py \
  --spectra data/processed/spectra.csv \
  --labels data/processed/labels.csv \
  --input_type wavelengths \
  --wavelengths 1007 1014 1032 1236 \
  --model svm \
  --out_dir results/models
```

---

## 8. Practical significance

This study identifies a narrow SWIR region (**1007–1057 nm**) and a compact wavelength set (**1007, 1014, 1032, 1236 nm**) that can achieve perfect intact vs pitted classification under the experimental conditions. These results support the development of **reduced-band multispectral systems** (fewer wavelengths, lower cost, higher speed) for real-time industrial inspection.

---

## 9. Citation

If you use this repository, please cite our paper:

```bibtex
@article{SinghLu_CherryPit_SWIR_HSI,
  title   = {Non-destructive detection of pits in cherries using shortwave infrared hyperspectral transmission imaging},
  author  = {Singh, Naseeb and Lu, Yuzhen},
  journal = {Food Bioscience},
  pages   = {108201},
  DOI    = {10.1016/j.fbio.2025.108201}
}
```

---

## 10. License

This project is released under the **Apache License 2.0**. 

---

## 11. Acknowledgements

Supported by the **Michigan Cherry Committee**.  


---

## 12. Contact

For questions, suggestions, or issues, please open an Issue in this GitHub repository.

