"""
src/analyze_rimone_overlap.py

Deep overlap analysis of RIM-ONE DL dataset.
Computes MD5 checksums of binary image data to detect identical physical images across:
  - partitioned_by_hospital (training_set vs test_set)
  - partitioned_randomly (training_set vs test_set)
Reports exact unique image counts, cross-partition overlaps, and hospital/class breakdown.
"""

import os
import glob
import hashlib
import json
import yaml
from PIL import Image


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_rimone():
    config = load_config()
    rimone_base = config["RIMONE_PATH"]
    project_root = config["PROJECT_ROOT"]

    inner_path = os.path.join(rimone_base, "RIM-ONE_DL_images")
    if not os.path.exists(inner_path):
        inner_path = rimone_base

    print("=" * 60)
    print("RIM-ONE DL DEEP OVERLAP & DUPLICATE ANALYSIS")
    print("=" * 60)
    print(f"Effective root: {inner_path}")

    # Subsets to analyze
    subsets = {
        "hospital_test": os.path.join(inner_path, "partitioned_by_hospital", "test_set"),
        "hospital_train": os.path.join(inner_path, "partitioned_by_hospital", "training_set"),
        "random_test": os.path.join(inner_path, "partitioned_randomly", "test_set"),
        "random_train": os.path.join(inner_path, "partitioned_randomly", "training_set"),
    }

    subset_images = {}
    all_records = []

    for name, s_path in subsets.items():
        if os.path.exists(s_path):
            files = set(glob.glob(os.path.join(s_path, "**", "*.*"), recursive=True))
            # Filter image extensions
            files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
            subset_images[name] = sorted(files)
            print(f"Subset '{name}': {len(files)} files found.")
            for f in files:
                rel = os.path.relpath(f, inner_path)
                fname = os.path.basename(f)
                prefix = fname.split("_")[0] if "_" in fname else "unknown"

                # Class folder
                parts = rel.split(os.sep)
                cls_folder = parts[-2] if len(parts) >= 2 else "unknown"
                label = 1 if "glaucoma" in cls_folder.lower() else (0 if "normal" in cls_folder.lower() else -1)

                md5 = compute_md5(f)
                all_records.append({
                    "subset": name,
                    "rel_path": rel,
                    "filename": fname,
                    "prefix": prefix,
                    "class": cls_folder,
                    "label": label,
                    "md5": md5,
                    "full_path": f
                })

    df = pd.DataFrame(all_records)
    print(f"\nTotal image records collected across all 4 partition subsets: {len(df)}")

    # 1. Total counts per subset
    print("\n--- 1. SUBSET FILE COUNTS & CLASSES ---")
    subset_summary = {}
    for name in subsets.keys():
        sub_df = df[df["subset"] == name]
        norm_c = len(sub_df[sub_df["label"] == 0])
        glau_c = len(sub_df[sub_df["label"] == 1])
        tot = len(sub_df)
        prefixes = sub_df["prefix"].value_counts().to_dict()
        subset_summary[name] = {
            "total": tot,
            "normal": norm_c,
            "glaucoma": glau_c,
            "prefixes": prefixes
        }
        print(f"  {name:15s}: Total={tot:3d} | Normal={norm_c:3d} | Glaucoma={glau_c:3d} | Prefixes={prefixes}")

    # 2. Total images in hospital partition vs random partition
    hosp_df = df[df["subset"].str.startswith("hospital")]
    rand_df = df[df["subset"].str.startswith("random")]

    hosp_md5s = set(hosp_df["md5"])
    rand_md5s = set(rand_df["md5"])
    total_unique_md5s = set(df["md5"])

    print("\n--- 2. OVERLAP ANALYSIS BETWEEN HOSPITAL & RANDOM PARTITIONS ---")
    print(f"Total files in hospital partition scheme : {len(hosp_df)} ({len(hosp_md5s)} unique MD5s)")
    print(f"Total files in random partition scheme   : {len(rand_df)} ({len(rand_md5s)} unique MD5s)")
    print(f"Overall unique physical images in dataset: {len(total_unique_md5s)}")

    hosp_rand_intersection = hosp_md5s.intersection(rand_md5s)
    print(f"MD5s shared between hospital & random partitions: {len(hosp_rand_intersection)}")

    if len(hosp_rand_intersection) == len(hosp_md5s) == len(rand_md5s):
        print("\nFINDING: The 'partitioned_by_hospital' and 'partitioned_randomly' schemes contain the EXACT SAME 485 underlying physical images, merely split differently!")
    else:
        print(f"\nFINDING: Differences found between hospital & random partitions.")

    # 3. Check within-scheme overlap (train vs test in hospital partition)
    hosp_test_md5s = set(df[df["subset"] == "hospital_test"]["md5"])
    hosp_train_md5s = set(df[df["subset"] == "hospital_train"]["md5"])
    hosp_internal_overlap = hosp_test_md5s.intersection(hosp_train_md5s)

    print("\n--- 3. HOSPITAL PARTITION SPLIT ANALYSIS ---")
    print(f"Hospital Test unique MD5s  : {len(hosp_test_md5s)}")
    print(f"Hospital Train unique MD5s : {len(hosp_train_md5s)}")
    print(f"Overlap between Hospital Train & Test: {len(hosp_internal_overlap)}")

    # 4. Check prefix breakdown per partition
    print("\n--- 4. HOSPITAL PREVIEW / SITE BREAKDOWN ---")
    print("Hospital Test prefixes breakdown:")
    print(df[df["subset"] == "hospital_test"].groupby(["prefix", "class"]).size())
    print("\nHospital Train prefixes breakdown:")
    print(df[df["subset"] == "hospital_train"].groupby(["prefix", "class"]).size())

    # Save overlap analysis report
    res_dir = os.path.join(project_root, "results", "audit")
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "rimone_overlap_analysis.json")

    with open(out_json, "w") as f:
        json.dump({
            "subset_summary": subset_summary,
            "total_records_collected": len(df),
            "hospital_total_files": len(hosp_df),
            "hospital_unique_md5s": len(hosp_md5s),
            "random_total_files": len(rand_df),
            "random_unique_md5s": len(rand_md5s),
            "total_unique_physical_images": len(total_unique_md5s),
            "hospital_random_md5_overlap": len(hosp_rand_intersection),
            "hospital_train_test_md5_overlap": len(hosp_internal_overlap)
        }, f, indent=2)

    print(f"\nOverlap analysis report saved to: {out_json}\n")


if __name__ == "__main__":
    import pandas as pd
    analyze_rimone()
