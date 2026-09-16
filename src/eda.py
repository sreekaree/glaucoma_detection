"""
src/eda.py

Exploratory Data Analysis (EDA) script for Glaucoma Detection Project.
Generates comprehensive summary tables, class balance figures, and source distribution charts.
Saves artifacts in results/eda/.
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_eda():
    config = load_config()
    project_root = config["PROJECT_ROOT"]
    manifest_path = os.path.join(project_root, "metadata", "master_manifest.csv")

    df = pd.read_csv(manifest_path)
    out_dir = os.path.join(project_root, "results", "eda")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Dataset & Split Overview
    print("\n--- 1. DATASET ROLES AND SPLITS OVERVIEW ---")
    dset_split = pd.crosstab(df["dataset"], df["split"], margins=True)
    print(dset_split)

    # 2. Class Distributions for Usable Experimental Data
    exp_df = df[df["split"] != "excluded"].copy()
    print("\n--- 2. USABLE EXPERIMENTAL CLASS DISTRIBUTIONS ---")
    class_dist = pd.crosstab(exp_df["dataset"], exp_df["class_name"], margins=True)
    print(class_dist)

    # Calculate baselines
    baselines = {}
    for dset in ["DRISHTI", "HRF", "RIM-ONE"]:
        sub = exp_df[exp_df["dataset"] == dset]
        n_tot = len(sub)
        n_norm = len(sub[sub["label"] == 0])
        n_glau = len(sub[sub["label"] == 1])
        maj_class = "Normal" if n_norm >= n_glau else "Glaucoma"
        maj_pct = (max(n_norm, n_glau) / float(n_tot)) * 100.0 if n_tot > 0 else 0.0
        baselines[dset] = {
            "total_usable": n_tot,
            "normal": n_norm,
            "glaucoma": n_glau,
            "majority_class": maj_class,
            "majority_baseline_pct": round(maj_pct, 2)
        }

    print("\n--- 3. MAJORITY-CLASS BASELINES ---")
    for dset, b_info in baselines.items():
        print(f"  {dset:8s}: {b_info['total_usable']:3d} images ({b_info['normal']} Normal, {b_info['glaucoma']} Glaucoma) -> Majority Baseline: {b_info['majority_baseline_pct']}% ({b_info['majority_class']})")

    # 4. RIM-ONE Hospital Prefix Breakdown
    rimone_df = exp_df[exp_df["dataset"] == "RIM-ONE"]
    rimone_hosp = pd.crosstab(rimone_df["hospital"], rimone_df["class_name"], margins=True)
    print("\n--- 4. RIM-ONE DL HOSPITAL SITE BREAKDOWN ---")
    print(rimone_hosp)

    # 5. DRISHTI 5-Fold Distribution
    drishti_df = exp_df[exp_df["dataset"] == "DRISHTI"]
    drishti_folds = pd.crosstab(drishti_df["fold"], drishti_df["class_name"], margins=True)
    print("\n--- 5. DRISHTI 5-FOLD CV DISTRIBUTION ---")
    print(drishti_folds)

    # Save summary text report
    txt_path = os.path.join(out_dir, "eda_summary.txt")
    with open(txt_path, "w") as f:
        f.write("============================================================\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("============================================================\n\n")
        f.write("1. DATASET ROLES & SPLITS\n")
        f.write(dset_split.to_string() + "\n\n")
        f.write("2. EXPERIMENTAL CLASS DISTRIBUTIONS\n")
        f.write(class_dist.to_string() + "\n\n")
        f.write("3. RIM-ONE SITE BREAKDOWN\n")
        f.write(rimone_hosp.to_string() + "\n\n")
        f.write("4. DRISHTI 5-FOLD CV DISTRIBUTION\n")
        f.write(drishti_folds.to_string() + "\n\n")
        f.write("5. MAJORITY-CLASS BASELINES\n")
        for dset, b_info in baselines.items():
            f.write(f"  {dset:8s}: {b_info['total_usable']} images ({b_info['normal']} Normal, {b_info['glaucoma']} Glaucoma) | Baseline: {b_info['majority_baseline_pct']}%\n")

    # Save summary JSON
    json_path = os.path.join(out_dir, "eda_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "baselines": baselines,
            "total_manifest_entries": len(df),
            "total_experimental_images": len(exp_df),
            "total_excluded_images": len(df[df["split"] == "excluded"])
        }, f, indent=2)

    # Generate Figures
    sns.set_theme(style="whitegrid")

    # Plot A: Class Balance per Dataset
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=exp_df, x="dataset", hue="class_name", palette="Set2")
    plt.title("Class Balance per Dataset (Usable Experimental Cohorts)", fontsize=14, pad=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    for p in ax.patches:
        h_val = p.get_height()
        if h_val is not None and not np.isnan(h_val) and h_val > 0:
            height = int(h_val)
            ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=11, color='white', weight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "class_distribution_per_dataset.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close()

    # Plot B: RIM-ONE Hospital Site Breakdown
    plt.figure(figsize=(9, 5))
    ax2 = sns.countplot(data=rimone_df, x="hospital", hue="class_name", palette="viridis")
    plt.title("RIM-ONE DL Site Breakdown (r1 = Hosp 1, r2 = Hosp 2, r3 = Hosp 3)", fontsize=13, pad=12)
    plt.xlabel("Hospital Site Identifier", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    for p in ax2.patches:
        h_val = p.get_height()
        if h_val is not None and not np.isnan(h_val) and h_val > 0:
            height = int(h_val)
            ax2.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height / 2.),
                         ha='center', va='center', fontsize=11, color='white', weight='bold')

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "rimone_site_breakdown.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()

    # Plot C: DRISHTI 5-Fold Class Balance
    plt.figure(figsize=(9, 5))
    ax3 = sns.countplot(data=drishti_df, x="fold", hue="class_name", palette="Set1")
    plt.title("DRISHTI-GS1 Stratified Group 5-Fold CV Class Balance", fontsize=13, pad=12)
    plt.xlabel("CV Fold Index", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    for p in ax3.patches:
        h_val = p.get_height()
        if h_val is not None and not np.isnan(h_val) and h_val > 0:
            height = int(h_val)
            ax3.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height / 2.),
                         ha='center', va='center', fontsize=11, color='white', weight='bold')

    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "drishti_fold_balance.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()

    print(f"\nEDA charts and reports saved to {out_dir}\n")


if __name__ == "__main__":
    run_eda()
