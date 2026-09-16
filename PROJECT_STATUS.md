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
| **Classical Classifiers** | **READY** | Script written, validated, dry-run passed. Awaiting training authorisation. |
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

## 5. Stage 7 — Classical ML Experiment Plan (READY, NOT YET EXECUTED)

### Experiment Matrix

| Classifier | Feature Set A (GLCM, 12) | Feature Set B (GLRLM, 11) | Feature Set C (GLCM+GLRLM, 23) |
| :--- | :---: | :---: | :---: |
| **SVM** (RBF, class-balanced) | planned | planned | planned |
| **Random Forest** (class-balanced) | planned | planned | planned |
| **XGBoost** | planned | planned | planned |

**Total experiments**: 9 (3 classifiers × 3 feature sets × 5 DRISHTI folds)

### Leakage-Safe Protocol

- `StandardScaler` fitted ONLY on the 4 training DRISHTI folds per CV iteration.
- `GridSearchCV` (inner 3-fold stratified CV, scoring = `roc_auc`) used for hyperparameter selection within training folds only.
- HRF and RIM-ONE are NEVER touched during fitting, scaling, or tuning.
- `sklearn Pipeline` enforces fit-on-train-only.

### Hyperparameter Grids (Fixed, Reproducible)

| Classifier | Parameter | Values |
| :--- | :--- | :--- |
| SVM | C | [0.1, 1.0, 10.0] |
| SVM | gamma | ['scale', 'auto'] |
| Random Forest | n_estimators | [100, 200] |
| Random Forest | max_depth | [None, 10, 20] |
| Random Forest | min_samples_leaf | [1, 3] |
| XGBoost | n_estimators | [100, 200] |
| XGBoost | max_depth | [3, 5] |
| XGBoost | learning_rate | [0.05, 0.1] |
| XGBoost | subsample | [0.8, 1.0] |

### Output Structure

```
results/classical_ml/
  fold_metrics/       per-fold metric CSV for each (model, feature_set)
  predictions/        OOF predictions + external test predictions per fold
  confusion_matrices/ PNG confusion matrices for validation & external
  summaries/          aggregated mean+/-std CSV, JSON, final report
```

### Script

`src/train_classical_models.py` — written and dry-run validated. Run with:
```bash
python src/train_classical_models.py            # full training
python src/train_classical_models.py --dry-run  # config check only
```

---

## 6. Future Steps

1. (**PENDING USER AUTHORISATION**) Execute `src/train_classical_models.py`.
2. Fine-tune deep learning backbones (MobileNetV2, ResNet50) on DRISHTI.
3. Generate Grad-CAM heatmaps for explainability analysis.
