# 🔬 ISIC 2024 — Skin Cancer Detection with 3D-TBP

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%2015%25-20BEFF?logo=kaggle)
![Competition](https://img.shields.io/badge/Competition-ISIC%202024-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

> **Kaggle:** [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)  
> **Final Ranking:** 🏆 **Top 15% of 3,410 competing teams**  
> **Primary Metric:** pAUC (partial AUC at TPR ≥ 80%)

---

## 📌 Problem Statement

Early detection of malignant skin lesions is critical — melanoma accounts for 75% of skin cancer deaths despite being the least common type. This competition challenged participants to build binary classifiers for distinguishing malignant from benign lesions using non-dermoscopic images extracted from 3D Total Body Photography (3D-TBP) systems, combined with rich patient metadata.

**Task:** Binary classification — `1` = Malignant, `0` = Benign  
**Challenge:** Extreme imbalance (~1,000:1 ratio — only 393 malignant out of 401,059 total images)  
**Metric:** Partial AUC (pAUC) at TPR ≥ 0.80 — prioritizes high-sensitivity detection

---

## 🧠 Approach — Inspired by 1st Place, With a Distinct Twist

> Research context: The actual 1st place winner (Ilya Novoselskiy, private LB pAUC = 0.1732) used multi-label classification across 5 cancer subtypes, combined image features from multiple lightweight architectures, and stacked them into a GBDT ensemble. The 2nd place team used 54 GBDT models with seed averaging and a fixed 1:3 or 1:5 negative:positive undersampling ratio per epoch.

### What We Borrowed from the Winners

- Multi-modal stacking: image model OOF predictions → LightGBM as meta-learner (used by all top-10 teams)
- Multi-label training targets (MEL, BCC, SCC, NV + binary target) for richer feature learning
- Patient-level Group K-Fold CV to prevent data leakage (same patient in both train and val)
- EfficientNetV2-S + Swin-Tiny ensemble for architectural diversity

### Our Distinct Contribution: Dynamic Undersampling Schedule

Instead of a fixed ratio (1:3 or 1:5 as used by 2nd place), we schedule the negative:positive ratio across epochs:

```
Epoch  1–5:   ratio 1:20  → Model sees abundant malignant examples early; learns their features
Epoch  6–15:  ratio 1:5   → Gradually introduce more benign diversity
Epoch 16–30:  ratio 1:3   → Final ratio matches realistic class distribution
```

**Why:** A fixed 1:3 ratio from epoch 1 means the first few batches still overwhelm malignant examples. Warming up with a high positive exposure rate helps the model build strong malignant representations before it must discriminate at scale.

### Patient-Relative Feature Engineering (Ugly Duckling Signal)

The GBDT layer uses patient-context features — features that compare a lesion *to the same patient's other lesions*:

```python
# Example engineered features
lesion_area_vs_patient_mean    = lesion_area / patient_mean_area
lesion_color_deviance          = abs(lesion_color - patient_mean_color)
lesion_size_rank_in_patient    = rank of this lesion's size among all patient's lesions
```

This captures the clinical "ugly duckling" principle: a suspicious lesion stands out from the patient's baseline.

---

## 📂 Dataset

- **Source:** [ISIC 2024 Kaggle Competition](https://www.kaggle.com/competitions/isic-2024-challenge/data)
- **Images:** 401,059 cropped lesion images (JPEG, ~128×128px) from 3D-TBP scans
- **Metadata:** 55 tabular features per record — demographics, lesion geometry, color statistics, anatomical site
- **Labels:** Binary (393 malignant vs. 400,666 benign)

> ⚠️ Data not included. Download via Kaggle API:
> ```bash
> kaggle competitions download -c isic-2024-challenge -p data/raw/
> ```

---

## 🏗️ Architecture: Dual-Stream + GBDT Stacking

```
                    ┌──────────────────────────┐
                    │    Image Stream           │
Images ─────────────► EfficientNetV2-S          │
                    │  + Swin-Tiny (diversity)  │
                    │  → OOF predictions        │
                    └────────────┬─────────────┘
                                 │  OOF preds as features
                    ┌────────────▼─────────────┐
Metadata + Engineered ► LightGBM Meta-Learner  │
Features  ────────────►  (5-fold Group K-Fold)  │
                    └────────────┬─────────────┘
                                 │
                         Final pAUC Score
```

**Image Model Training:**
- Architecture: EfficientNetV2-S (timm pretrained) + Swin-Tiny
- Input size: 128×128 (matches competition images)
- Loss: Binary Focal Loss (γ=2)
- Scheduler: CosineAnnealingLR
- Dynamic undersampling: epochs 1–5 (1:20), 6–15 (1:5), 16+ (1:3)
- CV: 5-Fold Patient-level Group K-Fold (no patient appears in both train and val)

**GBDT Layer:**
- Input: OOF image predictions + 55 raw metadata features + 12 engineered patient-relative features
- Models: LightGBM + CatBoost (averaged)
- CV: Same 5-fold Group K-Fold split

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Competition Rank** | **Top 15% (3,410 teams)** |
| Private LB pAUC | **0.168** |
| OOF AUC | **0.87** |

*pAUC is normalized to [0, 0.2] — random baseline ≈ 0.05*

---

## 📁 Project Structure

```
isic-2024-skin-cancer-detection/
│
├── data/
│   ├── raw/                          # Competition images & CSV (gitignored)
│   └── processed/                    # Fold splits, engineered features CSV
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Imbalance analysis, metadata exploration
│   ├── 02_Preprocessing.ipynb        # Image pipeline, feature engineering
│   ├── 03_Model_Training.ipynb       # CV training loop, dynamic undersampling
│   └── 04_Evaluation_Submission.ipynb # OOF eval, GBDT stacking, submission
│
├── src/
│   ├── dataset.py                    # PyTorch Dataset with dynamic undersampling
│   ├── model.py                      # EfficientNetV2-S / Swin-Tiny definitions
│   ├── train.py                      # Full 5-fold CV training loop (CLI)
│   ├── evaluate.py                   # pAUC, ROC curves, OOF scoring
│   ├── feature_engineering.py        # Patient-relative feature computation
│   ├── gbdt_stacking.py              # LightGBM + CatBoost meta-learner
│   └── utils.py                      # Seed, logging, augmentations
│
├── models/                           # Fold checkpoints (.pt)
├── results/
│   ├── oof_predictions.csv           # Out-of-fold predictions
│   ├── submission.csv                # Final Kaggle submission
│   └── figures/                      # ROC curves, feature importance
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone & set up
```bash
git clone https://github.com/YOUR_USERNAME/isic-2024-skin-cancer-detection.git
cd isic-2024-skin-cancer-detection

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data
```bash
kaggle competitions download -c isic-2024-challenge -p data/raw/
unzip data/raw/isic-2024-challenge.zip -d data/raw/
```

### 3. Run notebooks in order
```
01_EDA → 02_Preprocessing → 03_Model_Training → 04_Evaluation_Submission
```

### 4. Or run full training pipeline
```bash
# Train image models (5-fold CV)
python src/train.py --folds 5 --epochs 30 --img_size 128 --batch_size 64

# Train GBDT stacking layer
python src/gbdt_stacking.py --oof_path results/oof_predictions.csv
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch 2.x | Model training |
| timm | EfficientNetV2-S, Swin-Tiny pretrained weights |
| Albumentations | Image augmentation |
| LightGBM + CatBoost | GBDT meta-learner |
| Scikit-learn | CV splits, metrics |
| Pandas / NumPy | Tabular feature pipeline |
| Matplotlib / Seaborn | Visualization |

---

## 📈 Key Learnings

1. **Image-only models were not enough** — the extreme imbalance (1:1000) meant image models alone had very low pAUC; GBDT stacking on metadata unlocked the final score
2. **Patient-level Group K-Fold is non-negotiable** — standard random K-Fold leaks patient data and gives inflated CV scores that don't correlate with LB
3. **Dynamic undersampling scheduling outperformed fixed ratio** — the model converged faster and generalized better to the minority class
4. **Multi-label targets (MEL/BCC/SCC/NV) improve representations** — training the image model to predict cancer subtype forced richer feature learning than binary-only training
5. **pAUC requires calibrated models** — standard sigmoid outputs needed temperature scaling to improve pAUC in the ≥80% TPR region

---

## 🔗 References & Inspiration

- [1st Place Solution Discussion](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution) — Ilya Novoselskiy
- [2nd Place Solution + Code](https://github.com/uchiyama33/isic-2024-2nd-place) — Multi-GBDT ensemble with seed averaging
- [Top Solutions Summary (Medium)](https://medium.com/@nlztrk/my-competition-summary-isic-2024-825ab1b82711) — Patterns across top-10 solutions
- [ISIC 2024 SLICE-3D Dataset Paper](https://www.nature.com/articles/s41597-024-03743-w) — Kurtansky et al., Scientific Data
- [TIP: Tabular-Image Pre-training (ECCV 2024)](https://arxiv.org/abs/2307.04024) — Used by 2nd place for self-supervised tabular pretraining

---

## 📬 Contact

**Cruicil** | MS Data Science & Analytics, SUNY Polytechnic Institute  
🔗 [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | 🐙 [GitHub](https://github.com/YOUR_USERNAME)

---
*This project competed in the actual ISIC 2024 Kaggle competition (summer 2024), finishing in the Top 15% among 3,410 participating teams.*
