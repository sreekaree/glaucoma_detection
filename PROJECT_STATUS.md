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
| **Classical Classifiers** | **COMPLETE** | 9 experiments (SVM/RF/XGBoost x GLCM/GLRLM/Combined). Results in `results/classical_ml/`. |
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

## 5. Stage 7 — Classical ML Experiment Results (COMPLETE)

### Leakage Protocol Applied
- **Outer CV**: Pre-assigned patient-level folds (`StratifiedGroupKFold`, seed=42, 5 splits)
- **Inner CV**: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` with `groups=patient_id`
- **StandardScaler**: Inside `sklearn.Pipeline`, fitted only on outer training folds
- **Threshold**: Fixed at 0.5. Not tuned on HRF or RIM-ONE.
- **HRF/RIM-ONE**: Used for evaluation only — never fit, scaled, or tuned on.

### DRISHTI 5-Fold CV Results (mean ± std, AUC from probabilities)

| Experiment | AUC | Sensitivity | Specificity | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| GLCM + SVM | 0.460±0.110 | 0.957±0.096 | 0.095±0.147 | 0.811±0.061 | 0.694 |
| GLCM + RandomForest | **0.573±0.098** | 0.714±0.168 | 0.357±0.144 | 0.708±0.114 | 0.605 |
| GLCM + XGBoost | 0.527±0.107 | 0.786±0.087 | 0.257±0.145 | 0.742±0.039 | 0.624 |
| GLRLM + SVM | 0.479±0.147 | 0.986±0.032 | 0.000±0.000 | 0.812±0.026 | 0.684 |
| GLRLM + RandomForest | 0.551±0.064 | 0.714±0.087 | 0.414±0.164 | 0.723±0.048 | 0.624 |
| GLRLM + XGBoost | 0.499±0.110 | 0.800±0.117 | 0.133±0.139 | 0.730±0.078 | 0.596 |
| GLCM+GLRLM + SVM | 0.482±0.113 | 0.971±0.064 | 0.000±0.000 | 0.804±0.043 | 0.674 |
| GLCM+GLRLM + RandomForest | 0.554±0.080 | 0.686±0.108 | 0.324±0.120 | 0.687±0.060 | 0.574 |
| GLCM+GLRLM + XGBoost | 0.537±0.098 | 0.814±0.108 | 0.257±0.145 | 0.758±0.054 | 0.644 |

> **Best DRISHTI AUC**: GLCM + RandomForest (0.573±0.098)

### HRF External Test Results (pooled, 30 images)

| Experiment | AUC | Sensitivity | Specificity | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| GLCM + SVM | 0.631 | 1.000 | 0.000 | 0.667 | 0.500 |
| GLCM + RandomForest | 0.707 | 1.000 | 0.067 | 0.682 | 0.533 |
| GLCM + XGBoost | **0.720** | 1.000 | 0.067 | 0.682 | 0.533 |
| GLRLM + SVM | 0.702 | 1.000 | 0.000 | 0.667 | 0.500 |
| GLRLM + RandomForest | 0.680 | 0.933 | 0.333 | **0.718** | **0.633** |
| GLRLM + XGBoost | 0.729 | 1.000 | 0.200 | 0.714 | 0.600 |
| GLCM+GLRLM + SVM | 0.698 | 1.000 | 0.000 | 0.667 | 0.500 |
| GLCM+GLRLM + RandomForest | 0.667 | 1.000 | 0.067 | 0.682 | 0.533 |
| GLCM+GLRLM + XGBoost | 0.529 | 1.000 | 0.000 | 0.667 | 0.500 |

> **Best HRF AUC**: GLRLM + XGBoost (0.729)

### RIM-ONE External Test Results (pooled, 485 images)

| Experiment | AUC | Sensitivity | Specificity | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| GLCM + SVM | 0.537 | 1.000 | 0.000 | 0.524 | 0.355 |
| GLCM + RandomForest | **0.641** | 0.669 | 0.511 | 0.523 | 0.567 |
| GLCM + XGBoost | 0.583 | 0.988 | 0.003 | 0.520 | 0.353 |
| GLRLM + SVM | 0.523 | 1.000 | 0.000 | 0.524 | 0.355 |
| GLRLM + RandomForest | 0.473 | 0.523 | 0.435 | 0.410 | 0.466 |
| GLRLM + XGBoost | 0.473 | 0.965 | 0.026 | 0.516 | 0.359 |
| GLCM+GLRLM + SVM | 0.507 | 1.000 | 0.000 | 0.524 | 0.355 |
| GLCM+GLRLM + RandomForest | 0.626 | 0.698 | 0.498 | **0.535** | **0.569** |
| GLCM+GLRLM + XGBoost | 0.579 | 0.826 | 0.336 | 0.544 | 0.509 |

> **Best RIM-ONE AUC**: GLCM + RandomForest (0.641)

### RIM-ONE Per-Site AUC Summary (pooled)

| Experiment | r1 AUC | r2 AUC | r3 AUC |
|:---|:---:|:---:|:---:|
| GLCM + SVM | **0.701** | 0.567 | 0.558 |
| GLCM + RandomForest | 0.395 | **0.721** | **0.623** |
| GLCM + XGBoost | 0.370 | 0.657 | 0.565 |
| GLRLM + SVM | 0.592 | 0.592 | 0.591 |
| GLRLM + RandomForest | 0.523 | 0.476 | 0.555 |
| GLRLM + XGBoost | 0.519 | 0.504 | 0.516 |
| GLCM+GLRLM + SVM | 0.687 | 0.507 | 0.594 |
| GLCM+GLRLM + RandomForest | 0.411 | 0.721 | 0.598 |
| GLCM+GLRLM + XGBoost | 0.437 | 0.713 | 0.520 |

### Key Observations

- **SVM consistently predicts all-Glaucoma** (Specificity ≈ 0) across all feature sets — the class-balance correction drives it to the majority class on this small DRISHTI dataset.
- **RandomForest** is the most balanced classifier overall (best DRISHTI AUC, best RIM-ONE AUC).
- **GLRLM features alone underperform** GLCM on RIM-ONE (RF: 0.473 vs 0.641), suggesting GLCM captures more transferable texture.
- **Cross-site generalisation is difficult**: r1 (12 glaucoma / 86 normal — highly imbalanced) is hardest; r2 and r3 show higher AUC.
- **All AUCs are modest** (0.47–0.73), confirming that handcrafted GLCM+GLRLM features alone are insufficient — deep learning fine-tuning is the next planned step.
