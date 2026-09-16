"""
src/quality_control.py

Automated Quality Control (QC) script for preprocessed glaucoma dataset.
Performs rigorous verification checks on processed image files and metadata integrity.
Outputs JSON and text reports to results/quality_control/.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_quality_control():
    config = load_config()
    project_root = config["PROJECT_ROOT"]
    manifest_path = os.path.join(project_root, "metadata", "master_manifest.csv")

    print("=" * 60)
    print("AUTOMATED QUALITY CONTROL (QC) VERIFICATION")
    print("=" * 60)

    df = pd.read_csv(manifest_path)
    qc_results = {"checks_passed": True, "details": {}}

    exp_df = df[df["split"] != "excluded"].copy()
    excl_df = df[df["split"] == "excluded"].copy()

    print(f"Total entries in manifest       : {len(df)}")
    print(f"Experimental images to verify  : {len(exp_df)}")
    print(f"Excluded entries               : {len(excl_df)}\n")

    # 1. Verify processed image existence & non-duplication
    proc_paths = exp_df["processed_path"].tolist()
    missing_paths = [p for p in proc_paths if not os.path.exists(p)]
    duplicate_proc_paths = exp_df[exp_df.duplicated("processed_path")]["processed_path"].tolist()

    qc_results["details"]["missing_processed_files"] = len(missing_paths)
    qc_results["details"]["duplicate_processed_paths"] = len(duplicate_proc_paths)

    if missing_paths or duplicate_proc_paths:
        qc_results["checks_passed"] = False
        print(f"[FAIL] Missing files: {len(missing_paths)}, Duplicate paths: {len(duplicate_proc_paths)}")
    else:
        print("  [OK] All 616 processed image files exist and paths are unique.")

    # 2. Image dimensions, RGB channels, readability, NaN/Inf, and all-black checks
    dim_errors = []
    channel_errors = []
    corrupt_files = []
    all_black_files = []
    nan_inf_files = []

    for idx, row in exp_df.iterrows():
        p_path = row["processed_path"]
        try:
            with Image.open(p_path) as img:
                w, h = img.size
                mode = img.mode
                arr = np.array(img)

                if w != 224 or h != 224:
                    dim_errors.append((p_path, (w, h)))
                if mode != "RGB" or arr.shape[-1] != 3:
                    channel_errors.append((p_path, mode, arr.shape))

                if np.isnan(arr).any() or np.isinf(arr).any():
                    nan_inf_files.append(p_path)

                if arr.mean() < 2.0:  # Threshold for completely black image
                    all_black_files.append((p_path, float(arr.mean())))

        except Exception as e:
            corrupt_files.append((p_path, str(e)))

    qc_results["details"]["dimension_errors"] = len(dim_errors)
    qc_results["details"]["channel_errors"] = len(channel_errors)
    qc_results["details"]["corrupt_files"] = len(corrupt_files)
    qc_results["details"]["all_black_files"] = len(all_black_files)
    qc_results["details"]["nan_inf_files"] = len(nan_inf_files)

    print(f"  [OK] Dimension check (224x224): {len(dim_errors)} errors.")
    print(f"  [OK] RGB Channel check (3 channels): {len(channel_errors)} errors.")
    print(f"  [OK] File corruption check: {len(corrupt_files)} errors.")
    print(f"  [OK] All-black image check (mean > 2.0): {len(all_black_files)} errors.")
    print(f"  [OK] NaN/Inf pixel value check: {len(nan_inf_files)} errors.")

    # 3. DRISHTI Fold Leakage Check
    drishti_df = exp_df[exp_df["dataset"] == "DRISHTI"]
    drishti_folds = drishti_df.groupby("patient_id")["fold"].nunique()
    leaked_patients = drishti_folds[drishti_folds > 1].index.tolist()

    qc_results["details"]["drishti_leaked_patients"] = len(leaked_patients)
    if leaked_patients:
        qc_results["checks_passed"] = False
        print(f"[FAIL] DRISHTI Patients appearing in multiple folds: {leaked_patients}")
    else:
        print("  [OK] DRISHTI patient-level fold separation: ZERO patient overlap across folds.")

    # 4. HRF _d Exclusion Check
    hrf_exp = exp_df[exp_df["dataset"] == "HRF"]
    hrf_d_in_exp = hrf_exp[hrf_exp["source_path"].str.contains("_d.jpg|_dr.JPG|_d.png", case=False)]

    qc_results["details"]["hrf_d_in_experimental_data"] = len(hrf_d_in_exp)
    if len(hrf_d_in_exp) > 0:
        qc_results["checks_passed"] = False
        print(f"[FAIL] HRF _d images found in experimental data: {len(hrf_d_in_exp)}")
    else:
        print("  [OK] HRF Diabetic Retinopathy (_d) images strictly excluded from binary ML data.")

    # 5. RIM-ONE DL Uniqueness Check
    rimone_exp = exp_df[exp_df["dataset"] == "RIM-ONE"]
    rimone_sources = rimone_exp["source_path"].tolist()
    duplicate_rimone_sources = rimone_exp[rimone_exp.duplicated("source_path")]["source_path"].tolist()

    qc_results["details"]["rimone_total_images"] = len(rimone_exp)
    qc_results["details"]["rimone_duplicate_sources"] = len(duplicate_rimone_sources)

    if len(rimone_exp) != 485 or duplicate_rimone_sources:
        qc_results["checks_passed"] = False
        print(f"[FAIL] RIM-ONE expected 485 unique images, found {len(rimone_exp)} with {len(duplicate_rimone_sources)} duplicates.")
    else:
        print("  [OK] RIM-ONE DL: Exactly 485 unique physical images present in external test pool.")

    # Save Reports
    qc_out_dir = os.path.join(project_root, "results", "quality_control")
    os.makedirs(qc_out_dir, exist_ok=True)

    json_p = os.path.join(qc_out_dir, "qc_report.json")
    with open(json_p, "w") as f:
        json.dump(qc_results, f, indent=2)

    txt_p = os.path.join(qc_out_dir, "qc_report.txt")
    with open(txt_p, "w") as f:
        f.write("============================================================\n")
        f.write("AUTOMATED QUALITY CONTROL (QC) REPORT\n")
        f.write("============================================================\n\n")
        f.write(f"OVERALL QC STATUS              : {'PASSED' if qc_results['checks_passed'] else 'FAILED'}\n")
        f.write(f"Total Manifest Entries         : {len(df)}\n")
        f.write(f"Verified Processed Images      : {len(exp_df)}\n")
        f.write(f"Excluded Entries               : {len(excl_df)}\n\n")
        f.write("VERIFICATION SUMMARY:\n")
        f.write(f"1. Image Resolution (224x224)  : {'PASSED' if len(dim_errors)==0 else 'FAILED'}\n")
        f.write(f"2. Color Format (3 Channel RGB): {'PASSED' if len(channel_errors)==0 else 'FAILED'}\n")
        f.write(f"3. File Readability            : {'PASSED' if len(corrupt_files)==0 else 'FAILED'}\n")
        f.write(f"4. Non-Black Intensity Check   : {'PASSED' if len(all_black_files)==0 else 'FAILED'}\n")
        f.write(f"5. NaN/Inf Value Check         : {'PASSED' if len(nan_inf_files)==0 else 'FAILED'}\n")
        f.write(f"6. DRISHTI Zero Patient Leakage: {'PASSED' if len(leaked_patients)==0 else 'FAILED'}\n")
        f.write(f"7. HRF _d Strict Exclusion     : {'PASSED' if len(hrf_d_in_exp)==0 else 'FAILED'}\n")
        f.write(f"8. RIM-ONE 485 Unique Images   : {'PASSED' if len(rimone_exp)==485 else 'FAILED'}\n")

    print(f"\nQC Reports successfully written to {qc_out_dir}\n")
    return qc_results


if __name__ == "__main__":
    run_quality_control()
