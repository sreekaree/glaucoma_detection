# Glaucoma Detection Project — Data Preparation & Experimental Pipeline

This repository contains the complete, reproducible data preparation, auditing, patient-level cross-validation, preprocessing, quality control, visual verification, and exploratory data analysis pipeline for binary glaucoma classification from fundus photography.

---

## 1. Project Objective & Task Specification

- **Task**: Binary Glaucoma Classification from Retinal Fundus Images.
- **Classification Target ($y$)**:
  - `0 = Normal` (Healthy retina)
  - `1 = Glaucoma` (Glaucomatous optic neuropathy)
- **Primary Design Principle**: Absolute patient-level data separation and zero data leakage. Training normalization statistics, model selection, hyperparameter tuning, and decision threshold selection are computed strictly within the development set, while evaluation is conducted on completely independent external test sets.

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

## 3. Dataset Details & Leakage Prevention

### A. DRISHTI-GS1 (Development Dataset)
- **Total Images**: 101 fundus photographs (50 from original `Training`, 51 from original `Test`).
- **Patient Count**: 68 unique Patient IDs.
- **Bilateral Discordant Cases**: 6 patients (12 images) have different eye-level diagnoses (e.g., Left Eye Normal, Right Eye Glaucoma). Their eye-level labels are strictly preserved (`0` and `1`), while both images are grouped into the **same fold** using `StratifiedGroupKFold`.
- **CV Strategy**: `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups = Patient ID`.
  - Fold 0: 13 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
  - Fold 1: 13 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
  - Fold 2: 14 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
  - Fold 3: 14 Patients | 21 Images (7 Normal, 14 Glaucoma, 4 Discordant)
  - Fold 4: 14 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)

### B. HRF (External Evaluation Test Set 1)
- **Primary Cohort**: 30 high-resolution fundus images (15 Normal `_h`, 15 Glaucoma `_g`).
- **Exclusion**: 15 Diabetic Retinopathy (`_d`) images are strictly excluded (`split = excluded`, `exclusion_reason = diabetic_retinopathy`).

### C. RIM-ONE DL (External Evaluation Test Set 2)
- **Primary Cohort**: All 485 unique physical images from `partitioned_by_hospital/` (`training_set/` + `test_set/`).
- **Hospital Site Breakdown**:
  - `r1` (Hospital 1 / RIM-ONE v1): 86 Normal, 12 Glaucoma (98 total)
  - `r2` (Hospital 2 / RIM-ONE v2): 142 Normal, 108 Glaucoma (250 total)
  - `r3` (Hospital 3 / RIM-ONE v3): 85 Normal, 52 Glaucoma (137 total)
- **Directory Exclusion**: `partitioned_randomly/` is **ignored** because MD5 binary checksum analysis proved it contains the exact same 485 physical images arranged into different subfolders.

---

## 4. Preprocessing Pipeline & Parameters

The exact same standardized preprocessing pipeline is applied to all datasets:

```
Raw Image File (Raw data remains 100% untouched)
       ↓
Validate Readability
       ↓
Non-Aggressive FOV Crop (Threshold=12, Safety Margin=2%)
  -> Crops black background margins without clipping retina, optic disc, macula, or vessels.
       ↓
Aspect-Ratio Preserving Resize & Square Padding (Center canvas -> 224 × 224 × 3 RGB)
  -> Prevents geometric stretching or optic disc distortion.
       ↓
CLAHE on GREEN CHANNEL ONLY (clip_limit = 2.0, tile_grid_size = (8, 8))
  -> Enhances retinal vessel and optic nerve head contrast while preserving Red/Blue channels.
       ↓
Save as 8-bit RGB PNG in data/processed/<DATASET>/<class_name>/<image_id>.png
```

---

## 5. Repository Structure

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
│   └── eda.py                         # Computes EDA statistics and outputs plots
│
├── metadata/
│   └── master_manifest.csv            # Authoritative project manifest (631 entries)
│
├── splits/
│   └── drishti_5fold_splits.csv       # DRISHTI 5-fold cross-validation assignments
│
└── results/
    ├── audit/                         # Raw dataset audit text and JSON reports
    ├── quality_control/               # QC test result logs (100% Passed)
    ├── visual_checks/                 # BEFORE/AFTER grid images
    └── eda/                           # EDA text summaries and distribution charts
```

---

## 6. How to Reproduce the Pipeline

1. **Clone the Repository & Configure Paths**:
   ```bash
   git clone <repository_url>
   cd glaucoma_detection
   cp config/paths.example.yaml config/paths.yaml
   ```
   Edit `config/paths.yaml` with your local dataset locations.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR using Conda:
   conda env create -f environment.yml
   conda activate glaucoma-env
   ```

3. **Execute Data Pipeline Step-by-Step**:
   ```bash
   python src/audit_datasets.py
   python src/split_drishti.py
   python src/analyze_rimone_overlap.py
   python src/create_manifest.py
   python src/preprocess_images.py
   python src/quality_control.py
   python src/visual_checks.py
   python src/eda.py
   ```

---

## 7. Future Machine Learning & Deep Learning Roadmap

1. **Classical ML (Handcrafted Texture Features)**:
   - GLCM (Gray-Level Co-occurrence Matrix): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
   - GLRLM (Gray-Level Run Length Matrix): SRE, LRE, GLN, RLN, RP.
   - Classifiers: Support Vector Machines (SVM), Random Forest, XGBoost trained on DRISHTI 5-fold CV.
2. **Deep Learning Architectures**:
   - Transfer learning using MobileNetV2 and ResNet50 fine-tuned on DRISHTI.
3. **Explainability & Model Transparency**:
   - Grad-CAM and Grad-CAM++ visualizations to verify model focus on optic nerve head / cup-to-disc ratio.
