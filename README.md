# Glaucoma Detection Project — Data Preparation & Experimental Pipeline

This repository contains the complete, reproducible data preparation, auditing, patient-level cross-validation, preprocessing, quality control, visual verification, exploratory data analysis, and handcrafted texture feature extraction (GLCM + GLRLM) pipeline for binary glaucoma classification from fundus photography.

---

## 1. Project Objective & Task Specification

- **Task**: Binary Glaucoma Classification from Retinal Fundus Images.
- **Classification Target ($y$)**:
  - `0 = Normal` (Healthy retina)
  - `1 = Glaucoma` (Glaucomatous optic neuropathy)
- **Primary Design Principle**: Absolute patient-level data separation and zero data leakage. Training normalization statistics, feature scaling, model selection, hyperparameter tuning, and decision threshold selection are computed strictly within the development set, while evaluation is conducted on completely independent external test sets.

---

## 2. Experimental Dataset Roles

```
                              DRISHTI-GS1 (101 images)
                              68 Unique Patients
                                      |
                     StratifiedGroupKFold (5 Folds, seed=42)
                        /                       \
                       /                         \
           DRISHTI Train (4/5 Folds)        DRISHTI Validation (1/5 Fold)
           [Zero Patient Leakage]           [Zero Patient Leakage]
                       \                         /
                        \                       /
                          MODEL DEVELOPMENT POOL
                                /       \
                               /         \
                 HRF External Test      RIM-ONE DL Hospital External Test
                 (30 Images)            (485 Unique Images)
                 Baseline: 50.00%       Baseline: 64.54%
```

| Dataset | Experimental Role | Total Images | Normal (`0`) | Glaucoma (`1`) | Excluded | Majority Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DRISHTI-GS1** | **Development Pool (5-Fold CV)** | **101** | 31 | 70 | 0 | **69.31%** (Glaucoma) |
| **HRF** | **External Evaluation Test 1** | **30** | 15 | 15 | 15 (`_d`) | **50.00%** (Equal) |
| **RIM-ONE DL** | **External Evaluation Test 2** | **485** | 313 | 172 | 0 | **64.54%** (Normal) |

---

## 3. Preprocessing & Texture Feature Extraction Pipeline

### A. Preprocessing Pipeline
```
Raw Image File (Raw data remains 100% untouched)
       ↓
FOV Crop (Threshold=12, Safety Margin=2% -> Removes black border without clipping retina/optic disc)
       ↓
Aspect-Ratio Preserving Resize & Square Padding (Center canvas -> 224 × 224 × 3 RGB)
       ↓
CLAHE on GREEN CHANNEL ONLY (clip_limit = 2.0, tile_grid_size = (8, 8))
       ↓
Save as 8-bit RGB PNG in data/processed/<DATASET>/<class_name>/<image_id>.png
```

### B. Texture Feature Extraction (GLCM + GLRLM)
Quantized 8-bit Grayscale ($Y = 0.299R + 0.587G + 0.114B$) to 32 gray levels ($0..31$):
- **GLCM (12 Features)**: Distances `[1, 2, 4]`, Angles `[0, 45, 90, 135 deg]`, symmetric & normed. Features: `contrast`, `dissimilarity`, `homogeneity`, `energy`, `ASM`, `correlation` (means and stds).
- **GLRLM (11 Features)**: Directions `[0, 45, 90, 135 deg]`. Features: `sre`, `lre`, `gln`, `rln`, `rp`, `lglre`, `hglre`, `srlgle`, `srhgle`, `lrlgle`, `lrhgle` (directional means).
- **Combined (23 Features)**: 12 GLCM + 11 GLRLM features.

Saved files:
- `features/glcm_features.csv` (Shape: 616 x 24)
- `features/glrlm_features.csv` (Shape: 616 x 23)
- `features/glcm_glrlm_features.csv` (Shape: 616 x 35)

---

## 4. Repository Structure

```
glaucoma_detection/
├── README.md                          # Master project documentation
├── PROJECT_STATUS.md                  # Detailed handoff status and design record
├── GITHUB_SETUP.md                    # Instructions for pushing code to GitHub
├── requirements.txt                   # Required Python package dependencies
├── environment.yml                    # Conda environment specification file
├── .gitignore                         # Git exclusion rules
│
├── config/
│   ├── paths.yaml                     # Local dataset paths (git-ignored)
│   └── paths.example.yaml             # Repository template configuration
│
├── src/
│   ├── audit_datasets.py              # Audits raw datasets and checks integrity
│   ├── split_drishti.py               # Generates patient-level StratifiedGroupKFold splits
│   ├── analyze_rimone_overlap.py      # Performs MD5 checksum analysis on RIM-ONE DL
│   ├── create_manifest.py             # Generates metadata/master_manifest.csv
│   ├── preprocess_images.py           # Runs 224x224 Green-channel CLAHE pipeline
│   ├── quality_control.py             # Executes 8-point automated QC suite
│   ├── visual_checks.py               # Generates side-by-side BEFORE/AFTER visual grids
│   ├── eda.py                         # Computes EDA statistics and outputs plots
│   └── extract_texture_features.py    # Extracts GLCM (12) + GLRLM (11) texture features
│
├── metadata/
│   └── master_manifest.csv            # Authoritative project manifest (631 entries)
│
├── splits/
│   └── drishti_5fold_splits.csv       # DRISHTI 5-fold cross-validation assignments
│
├── features/
│   ├── glcm_features.csv              # 12 GLCM features + metadata (616 rows)
│   ├── glrlm_features.csv             # 11 GLRLM features + metadata (616 rows)
│   └── glcm_glrlm_features.csv        # 23 Combined features + metadata (616 rows)
│
└── results/
    ├── audit/                         # Raw dataset audit text and JSON reports
    ├── quality_control/               # QC test result logs (100% Passed)
    ├── visual_checks/                 # BEFORE/AFTER grid images
    ├── eda/                           # EDA text summaries and distribution charts
    └── texture_features/              # Feature stats, config JSON & extraction report
```

---

## 5. How to Reproduce the Pipeline

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Data Preparation & Feature Extraction
python src/audit_datasets.py
python src/split_drishti.py
python src/analyze_rimone_overlap.py
python src/create_manifest.py
python src/preprocess_images.py
python src/quality_control.py
python src/visual_checks.py
python src/eda.py
python src/extract_texture_features.py
```
