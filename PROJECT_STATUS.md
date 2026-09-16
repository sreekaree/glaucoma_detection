# Glaucoma Detection Project — Project Status & Handoff Document

This document records the exact current state, completed milestones, verified metrics, design decisions, and future roadmap for the Glaucoma Detection Project.

---

## 1. Executive Summary & Current Status

| Pipeline Stage | Status | Verification Result |
| :--- | :---: | :--- |
| **Raw Dataset Audit** | **COMPLETE** | Audited DRISHTI (101), HRF (45), RIM-ONE (485). |
| **DRISHTI 5-Fold CV Split** | **COMPLETE** | StratifiedGroupKFold (n_splits=5, seed=42, zero leakage). |
| **Master Manifest Generation**| **COMPLETE** | `metadata/master_manifest.csv` created (631 entries). |
| **Image Preprocessing** | **COMPLETE** | 616 images FOV cropped + 224x224 RGB padded + Green CLAHE. |
| **Automated Quality Control** | **COMPLETE** | 100% Passed (0 errors across 8 verification tests). |
| **Visual Verification Grids**| **COMPLETE** | Generated side-by-side BEFORE/AFTER visual comparison grids. |
| **Exploratory Data Analysis** | **COMPLETE** | Summary tables, majority baselines, and distribution charts. |
| **Texture Feature Extraction** | **COMPLETE** | Extracted 12 GLCM + 11 GLRLM features (616 rows, 0 NaN/Inf). |
| **Classical Classifiers** | **NOT STARTED**| SVM / Random Forest / XGBoost training pending. |
| **Deep Learning Models** | **NOT STARTED**| MobileNetV2 / ResNet50 fine-tuning pending. |
| **Explainability (Grad-CAM)** | **NOT STARTED**| Grad-CAM / Grad-CAM++ visualizations pending. |

---

## 2. Key Experimental & Architectural Decisions

1. **Dataset Roles**:
   - **DRISHTI-GS1** (101 images, 68 patients): Used strictly as the **Development Set** for 5-fold cross-validation, model training, validation, and hyperparameter tuning.
   - **HRF** (30 usable images): Used strictly as **External Evaluation Test Set 1**.
   - **RIM-ONE DL** (485 unique images): Used strictly as **External Evaluation Test Set 2**.
   - **PAPILA**: **NOT USED** (excluded completely from the project).

2. **Data Leakage & Exclusion Rules**:
   - **HRF `_d` Exclusion**: 15 Diabetic Retinopathy images in HRF are excluded (`split = excluded`).
   - **RIM-ONE DL Scheme Selection**: `partitioned_randomly/` is ignored because MD5 checksum analysis proved it contains the exact same 485 physical images as `partitioned_by_hospital/`. All 485 unique images from `partitioned_by_hospital/` are included as `external_test`.
   - **Zero Leakage**: HRF and RIM-ONE DL are never used during training, validation, hyperparameter tuning, threshold selection, feature scaling (StandardScaler), or normalization statistics calculation.

3. **DRISHTI 5-Fold Stratified Group CV Design**:
   - **Label Target**: Individual eye/image diagnosis ($0 = \text{Normal}, 1 = \text{Glaucoma}$) from `Drishti-GS1_diagnosis.xlsx`.
   - **Leakage Prevention**: Grouped by `Patient ID` using `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
   - **Bilateral Discordant Patients**: 6 patients (12 images) have different eye-level diagnoses. Their eye-level labels are preserved, while both images are kept in the **same fold** (`discordant_patient = True`).

---

## 3. Verified Cohort Counts & Baselines

### DRISHTI-GS1 5-Fold Cross-Validation Breakdown:
- **Total Images**: 101 (31 Normal, 70 Glaucoma) | **Unique Patients**: 68
- **Fold 0**: 13 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
- **Fold 1**: 13 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
- **Fold 2**: 14 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
- **Fold 3**: 14 Patients | 21 Images (7 Normal, 14 Glaucoma, 4 Discordant)
- **Fold 4**: 14 Patients | 20 Images (6 Normal, 14 Glaucoma, 2 Discordant)
- **Development Majority-Class Baseline**: $70 / 101 = \mathbf{69.31\%}$ (Glaucoma)

### HRF External Test Set:
- **Usable Images**: 30 (15 Normal, 15 Glaucoma) | Excluded: 15 `_d`
- **Majority-Class Baseline**: $15 / 30 = \mathbf{50.00\%}$

### RIM-ONE DL External Test Set:
- **Total Unique Images**: 485 (313 Normal, 172 Glaucoma)
- **Site Breakdown**:
  - `r1` (Hospital 1 / v1): 86 Normal, 12 Glaucoma (98 total)
  - `r2` (Hospital 2 / v2): 142 Normal, 108 Glaucoma (250 total)
  - `r3` (Hospital 3 / v3): 85 Normal, 52 Glaucoma (137 total)
- **Majority-Class Baseline**: $313 / 485 = \mathbf{64.54\%}$ (Normal)

---

## 4. Texture Feature Extraction Summary (STEP 6)

Handcrafted texture features were extracted from the preprocessed 224x224 RGB images after deterministic 32-level grayscale quantization ($Y = 0.299R + 0.587G + 0.114B$):

1. **GLCM Features (12 Features)**:
   - Config: 32 gray levels, distances `[1, 2, 4]`, angles `[0, 45, 90, 135 deg]`, symmetric & normed.
   - Properties: `contrast`, `dissimilarity`, `homogeneity`, `energy`, `ASM`, `correlation` (mean and std across 12 combinations).
   - Saved to: `features/glcm_features.csv` (Shape: 616 x 24).

2. **GLRLM Features (11 Features)**:
   - Config: 32 gray levels, 4 directions (`0, 45, 90, 135 deg`).
   - Metrics: `sre`, `lre`, `gln`, `rln`, `rp`, `lglre`, `hglre`, `srlgle`, `srhgle`, `lrlgle`, `lrhgle` (directional means).
   - Saved to: `features/glrlm_features.csv` (Shape: 616 x 23).

3. **Combined GLCM + GLRLM Features (23 Features)**:
   - Saved to: `features/glcm_glrlm_features.csv` (Shape: 616 x 35).
   - QC Check: 0 NaN values, 0 Inf values, 0 constant columns across all 616 rows.

---

## 5. Next Steps for Upcoming ML Experiments

1. Train classical ML classifiers (SVM, Random Forest, XGBoost) on DRISHTI 5-fold CV using:
   - Feature Set A: GLCM only
   - Feature Set B: GLRLM only
   - Feature Set C: GLCM + GLRLM combined
2. Evaluate classical models on external test sets (HRF and RIM-ONE DL).
3. Fine-tune deep learning backbones (MobileNetV2, ResNet50) on DRISHTI.
4. Generate Grad-CAM heatmaps for explainability analysis.
