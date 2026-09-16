"""
validate_inputs.py — Stage 1 validation for classical ML experiment.
Checks counts, fold distribution, row alignment, feature dimensions,
NaN/Inf, and binary labels. Prints a full report. Exits with code 1
on any failure.
"""

import sys
import pandas as pd
import numpy as np

ROOT = "."  # run from project root

manifest  = pd.read_csv(f"{ROOT}/metadata/master_manifest.csv")
splits    = pd.read_csv(f"{ROOT}/splits/drishti_5fold_splits.csv")
glcm      = pd.read_csv(f"{ROOT}/features/glcm_features.csv")
glrlm     = pd.read_csv(f"{ROOT}/features/glrlm_features.csv")
combined  = pd.read_csv(f"{ROOT}/features/glcm_glrlm_features.csv")

PASS = "  [PASS]"
FAIL = "  [FAIL]"
errors = []

def check(condition, msg_pass, msg_fail):
    if condition:
        print(f"{PASS} {msg_pass}")
    else:
        print(f"{FAIL} {msg_fail}")
        errors.append(msg_fail)

# ─────────────────────────────────────────────
# 1. DRISHTI
# ─────────────────────────────────────────────
print("\n══ 1. DRISHTI VALIDATION ══")
d = manifest[manifest["dataset"] == "DRISHTI"]
check(len(d) == 101,
      f"DRISHTI images = {len(d)}",
      f"Expected 101 DRISHTI images, got {len(d)}")

n_normal   = (d["label"] == 0).sum()
n_glaucoma = (d["label"] == 1).sum()
check(n_normal == 31,   f"DRISHTI Normal = {n_normal}",   f"Expected 31 Normal, got {n_normal}")
check(n_glaucoma == 70, f"DRISHTI Glaucoma = {n_glaucoma}", f"Expected 70 Glaucoma, got {n_glaucoma}")

EXPECTED_FOLDS = {
    0: (20, 6, 14),
    1: (20, 6, 14),
    2: (20, 6, 14),
    3: (21, 7, 14),
    4: (20, 6, 14),
}
print("\n  Fold distribution (from splits CSV):")
for fold, (exp_n, exp_0, exp_1) in EXPECTED_FOLDS.items():
    fd = splits[splits["fold"] == fold]
    n  = len(fd)
    n0 = (fd["label"] == 0).sum()
    n1 = (fd["label"] == 1).sum()
    ok = (n == exp_n) and (n0 == exp_0) and (n1 == exp_1)
    msg = f"Fold {fold}: {n} total, {n0} Normal, {n1} Glaucoma"
    check(ok, msg, f"Fold {fold} mismatch — expected ({exp_n},{exp_0},{exp_1}) got ({n},{n0},{n1})")

all_folds = sorted(splits["fold"].unique())
check(all_folds == [0, 1, 2, 3, 4],
      f"All five folds present: {all_folds}",
      f"Missing folds — got {all_folds}")

# ─────────────────────────────────────────────
# 2. HRF
# ─────────────────────────────────────────────
print("\n══ 2. HRF VALIDATION ══")
hrf = manifest[manifest["dataset"] == "HRF"]
hrf_usable = hrf[hrf["label"].isin([0, 1])]
hrf_excl   = hrf[hrf["label"] == -1]
check(len(hrf_usable) == 30,
      f"HRF usable images = {len(hrf_usable)}",
      f"Expected 30 usable HRF, got {len(hrf_usable)}")
hrf_normal   = (hrf_usable["label"] == 0).sum()
hrf_glaucoma = (hrf_usable["label"] == 1).sum()
check(hrf_normal == 15,   f"HRF Normal = {hrf_normal}",   f"Expected 15 HRF Normal, got {hrf_normal}")
check(hrf_glaucoma == 15, f"HRF Glaucoma = {hrf_glaucoma}", f"Expected 15 HRF Glaucoma, got {hrf_glaucoma}")
check(len(hrf_excl) == 15,
      f"HRF excluded (DR) = {len(hrf_excl)}",
      f"Expected 15 HRF excluded, got {len(hrf_excl)}")

# ─────────────────────────────────────────────
# 3. RIM-ONE
# ─────────────────────────────────────────────
print("\n══ 3. RIM-ONE VALIDATION ══")
rim = manifest[manifest["dataset"] == "RIM-ONE"]
check(len(rim) == 485,
      f"RIM-ONE total = {len(rim)}",
      f"Expected 485 RIM-ONE images, got {len(rim)}")
rim_normal   = (rim["label"] == 0).sum()
rim_glaucoma = (rim["label"] == 1).sum()
check(rim_normal == 313,   f"RIM-ONE Normal = {rim_normal}",   f"Expected 313 RIM-ONE Normal, got {rim_normal}")
check(rim_glaucoma == 172, f"RIM-ONE Glaucoma = {rim_glaucoma}", f"Expected 172 RIM-ONE Glaucoma, got {rim_glaucoma}")
hosp_counts = rim["hospital"].value_counts().to_dict()
print(f"  Hospital breakdown: {hosp_counts}")
check(hosp_counts.get("r1", 0) == 98,  f"r1 = {hosp_counts.get('r1',0)}",  f"Expected r1=98,  got {hosp_counts.get('r1',0)}")
check(hosp_counts.get("r2", 0) == 250, f"r2 = {hosp_counts.get('r2',0)}", f"Expected r2=250, got {hosp_counts.get('r2',0)}")
check(hosp_counts.get("r3", 0) == 137, f"r3 = {hosp_counts.get('r3',0)}", f"Expected r3=137, got {hosp_counts.get('r3',0)}")

# ─────────────────────────────────────────────
# 4. FEATURE CSV SHAPES
# ─────────────────────────────────────────────
print("\n══ 4. FEATURE DIMENSIONS ══")
META_COLS = ["image_id", "dataset", "patient_id", "label", "class_name",
             "source_path", "processed_path", "split", "fold",
             "hospital", "source_partition", "discordant_patient"]

glcm_feat_cols     = [c for c in glcm.columns     if c not in META_COLS]
glrlm_feat_cols    = [c for c in glrlm.columns    if c not in META_COLS]
combined_feat_cols = [c for c in combined.columns  if c not in META_COLS]

check(len(glcm_feat_cols) == 12,
      f"GLCM feature columns = {len(glcm_feat_cols)}",
      f"Expected 12 GLCM features, got {len(glcm_feat_cols)}: {glcm_feat_cols}")
check(len(glrlm_feat_cols) == 11,
      f"GLRLM feature columns = {len(glrlm_feat_cols)}",
      f"Expected 11 GLRLM features, got {len(glrlm_feat_cols)}: {glrlm_feat_cols}")
check(len(combined_feat_cols) == 23,
      f"GLCM+GLRLM feature columns = {len(combined_feat_cols)}",
      f"Expected 23 combined features, got {len(combined_feat_cols)}: {combined_feat_cols}")
check(len(glcm) == 616,     f"GLCM rows = {len(glcm)}",     f"Expected 616 GLCM rows, got {len(glcm)}")
check(len(glrlm) == 616,    f"GLRLM rows = {len(glrlm)}",    f"Expected 616 GLRLM rows, got {len(glrlm)}")
check(len(combined) == 616, f"Combined rows = {len(combined)}", f"Expected 616 combined rows, got {len(combined)}")

print(f"\n  GLCM features:     {glcm_feat_cols}")
print(f"\n  GLRLM features:    {glrlm_feat_cols}")
print(f"\n  Combined features: {combined_feat_cols}")

# ─────────────────────────────────────────────
# 5. ROW ALIGNMENT
# ─────────────────────────────────────────────
print("\n══ 5. ROW ALIGNMENT ══")
check(list(glcm["image_id"]) == list(glrlm["image_id"]),
      "GLCM and GLRLM image_id order identical",
      "MISMATCH: GLCM vs GLRLM image_id order")
check(list(glcm["image_id"]) == list(combined["image_id"]),
      "GLCM and Combined image_id order identical",
      "MISMATCH: GLCM vs Combined image_id order")

# Check metadata columns match across all three
for col in ["dataset", "patient_id", "label", "fold", "hospital", "split"]:
    if col in glcm.columns and col in glrlm.columns:
        ok = glcm[col].equals(glrlm[col])
        check(ok, f"Column '{col}' consistent across GLCM/GLRLM",
              f"MISMATCH: column '{col}' differs between GLCM and GLRLM")
    if col in glcm.columns and col in combined.columns:
        ok = glcm[col].equals(combined[col])
        check(ok, f"Column '{col}' consistent across GLCM/Combined",
              f"MISMATCH: column '{col}' differs between GLCM and Combined")

# ─────────────────────────────────────────────
# 6. NaN / Inf / Constant
# ─────────────────────────────────────────────
print("\n══ 6. DATA QUALITY ══")
for name, df, feat_cols in [
    ("GLCM", glcm, glcm_feat_cols),
    ("GLRLM", glrlm, glrlm_feat_cols),
    ("Combined", combined, combined_feat_cols),
]:
    X = df[feat_cols].values.astype(float)
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    const_cols = [c for c in feat_cols if df[c].nunique() <= 1]
    check(nan_count == 0, f"{name}: no NaN", f"{name}: {nan_count} NaN values found")
    check(inf_count == 0, f"{name}: no Inf", f"{name}: {inf_count} Inf values found")
    check(len(const_cols) == 0,
          f"{name}: no constant features",
          f"{name}: constant features found: {const_cols}")

# Binary labels
label_vals = sorted(combined["label"].unique())
check(set(label_vals) == {0, 1},
      f"Labels are binary 0/1 across all 616 rows",
      f"Non-binary label values found: {label_vals}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n══ VALIDATION SUMMARY ══")
if errors:
    print(f"  FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"  ALL CHECKS PASSED — 0 errors.")
    print(f"\n  Feature dimensions confirmed:")
    print(f"    GLCM:          {len(glcm_feat_cols):>2} features × {len(glcm)} images")
    print(f"    GLRLM:         {len(glrlm_feat_cols):>2} features × {len(glrlm)} images")
    print(f"    GLCM + GLRLM:  {len(combined_feat_cols):>2} features × {len(combined)} images")
    sys.exit(0)
