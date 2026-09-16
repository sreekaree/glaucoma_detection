"""
src/split_drishti.py

Finalized patient-level stratified 5-fold cross-validation for DRISHTI-GS1.
Uses StratifiedGroupKFold to ensure zero patient leakage while preserving
image-level eye diagnoses. Flags discordant patients.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_drishti_5fold_split(seed=42):
    config = load_config()
    drishti_base = config["DRISHTI_PATH"]
    project_root = config["PROJECT_ROOT"]

    excel_diag = os.path.join(
        drishti_base, "Drishti-GS1_files", "Drishti-GS1_diagnosis.xlsx"
    )
    if not os.path.exists(excel_diag):
        excel_diag = os.path.join(drishti_base, "Drishti-GS1_diagnosis.xlsx")

    df_raw = pd.read_excel(excel_diag, header=None)
    header_idx = None
    for idx, row in df_raw.iterrows():
        if "Drishti-GS File" in row.values:
            header_idx = idx
            break

    cols = df_raw.iloc[header_idx].values
    df = df_raw.iloc[header_idx + 1:].copy()
    df.columns = cols
    df = df.dropna(subset=["Drishti-GS File"])

    df["image_id"] = (
        df["Drishti-GS File"]
        .astype(str)
        .str.strip()
        .str.replace("'", "")
        .str.replace("'", "")
    )
    df["patient_id"] = df["Patient ID"].astype(str).str.strip()
    df["total_label"] = df["Total"].astype(str).str.strip()

    def map_label(t):
        t_low = str(t).lower()
        if "glaucomatous" in t_low or "glaucoma" in t_low:
            return 1, "Glaucoma"
        elif "normal" in t_low:
            return 0, "Normal"
        else:
            raise ValueError(f"Unknown label: {t}")

    df["label"], df["class_name"] = zip(*df["total_label"].apply(map_label))

    # Identify discordant patients
    patient_group = df.groupby("patient_id")
    discordant_pids = set()

    for pid, group in patient_group:
        if len(group["label"].unique()) > 1:
            discordant_pids.add(pid)

    df["discordant_patient"] = df["patient_id"].isin(discordant_pids)

    print("=" * 60)
    print("DRISHTI-GS1 PATIENT-LEVEL STRATIFIED 5-FOLD CV SETUP")
    print("=" * 60)
    print(f"Total Images: {len(df)}")
    print(f"Total Unique Patients: {df['patient_id'].nunique()}")
    print(f"Discordant Patients Count: {len(discordant_pids)} ({sorted(list(discordant_pids))})")
    print(f"Discordant Images Count: {df['discordant_patient'].sum()}\n")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    df["fold"] = -1
    X = df["image_id"].values
    y = df["label"].values
    groups = df["patient_id"].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    # Verification Checks
    print("VERIFICATION CHECKS:")
    patient_fold_map = df.groupby("patient_id")["fold"].nunique()
    overlap_pats = patient_fold_map[patient_fold_map > 1]

    if len(overlap_pats) > 0:
        print(f"ERROR: Patients in multiple folds: {overlap_pats}")
        raise ValueError("Patient leakage detected!")
    else:
        print("  [OK] Zero Patient IDs appear in multiple folds!")

    assert (df["fold"] >= 0).all(), "Unassigned images found!"
    print(f"  [OK] All {len(df)} images assigned to exactly 1 fold!")

    print("\nDRISHTI 5-FOLD BREAKDOWN TABLE:")
    fold_reports = []
    for f in range(5):
        f_df = df[df["fold"] == f]
        n_pats = f_df["patient_id"].nunique()
        n_imgs = len(f_df)
        n_norm = len(f_df[f_df["label"] == 0])
        n_glau = len(f_df[f_df["label"] == 1])
        n_disc = f_df["discordant_patient"].sum()

        print(
            f"Fold {f}: {n_pats:2d} Patients | {n_imgs:2d} Images | "
            f"Normal: {n_norm:2d} | Glaucoma: {n_glau:2d} | Discordant Imgs: {n_disc:2d}"
        )

        fold_reports.append({
            "fold": f,
            "patient_count": n_pats,
            "total_images": n_imgs,
            "normal_images": n_norm,
            "glaucoma_images": n_glau,
            "discordant_images": n_disc
        })

    splits_dir = os.path.join(project_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    out_csv = os.path.join(splits_dir, "drishti_5fold_splits.csv")

    cols_to_save = [
        "image_id", "patient_id", "label", "class_name", "fold", "discordant_patient"
    ]
    df[cols_to_save].sort_values("image_id").to_csv(out_csv, index=False)
    print(f"\nFold assignments saved to: {out_csv}\n")

    return df, pd.DataFrame(fold_reports)


if __name__ == "__main__":
    create_drishti_5fold_split(seed=42)
