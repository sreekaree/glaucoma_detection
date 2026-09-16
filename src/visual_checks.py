"""
src/visual_checks.py

Generates side-by-side BEFORE (raw) vs AFTER (preprocessed) visual comparison grids
for sample images across all three datasets (DRISHTI, HRF, RIM-ONE) and classes.
Saves figure grids to results/visual_checks/.
"""

import os
import yaml
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_visual_checks():
    config = load_config()
    project_root = config["PROJECT_ROOT"]
    manifest_path = os.path.join(project_root, "metadata", "master_manifest.csv")

    df = pd.read_csv(manifest_path)
    exp_df = df[df["split"] != "excluded"].copy()

    out_dir = os.path.join(project_root, "results", "visual_checks")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING BEFORE/AFTER VISUAL COMPARISON GRIDS")
    print("=" * 60)

    # Select samples for each dataset and class
    sample_targets = [
        ("DRISHTI", "Normal", 2),
        ("DRISHTI", "Glaucoma", 2),
        ("HRF", "Normal", 2),
        ("HRF", "Glaucoma", 2),
        ("RIM-ONE", "Normal", 2),
        ("RIM-ONE", "Glaucoma", 2),
    ]

    selected_rows = []
    for dset, cls, n_samples in sample_targets:
        sub = exp_df[(exp_df["dataset"] == dset) & (exp_df["class_name"] == cls)]
        if len(sub) > 0:
            selected_rows.append(sub.head(n_samples))

    samples_df = pd.concat(selected_rows, ignore_index=True)

    # Plot 1: Combined 6x4 Grid (Raw vs Processed for each sample)
    n_rows = len(samples_df)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3 * n_rows))
    fig.suptitle("Before (Raw Original) vs After (FOV Cropped + Aspect-Padded + Green CLAHE)", fontsize=14, y=0.995)

    for idx, row in samples_df.iterrows():
        src_p = row["source_path"]
        proc_p = row["processed_path"]

        # Read RGB images
        img_raw = cv2.cvtColor(cv2.imread(src_p), cv2.COLOR_BGR2RGB)
        img_proc = cv2.cvtColor(cv2.imread(proc_p), cv2.COLOR_BGR2RGB)

        # Plot Raw
        ax_raw = axes[idx, 0]
        ax_raw.imshow(img_raw)
        ax_raw.set_title(f"RAW: {row['dataset']} | {row['class_name']}\n{row['image_id']} ({img_raw.shape[1]}x{img_raw.shape[0]})", fontsize=9)
        ax_raw.axis("off")

        # Plot Processed
        ax_proc = axes[idx, 1]
        ax_proc.imshow(img_proc)
        ax_proc.set_title(f"PROCESSED: {row['dataset']} | {row['class_name']}\n224x224x3 RGB (Green CLAHE)", fontsize=9)
        ax_proc.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "before_after_comparison_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved main visual comparison grid: {grid_path}")

    # Plot 2: Per-Dataset Individual Comparison Cards
    for dset in ["DRISHTI", "HRF", "RIM-ONE"]:
        d_sub = samples_df[samples_df["dataset"] == dset]
        fig_d, axes_d = plt.subplots(len(d_sub), 2, figsize=(8, 3 * len(d_sub)))
        fig_d.suptitle(f"{dset} Preprocessing Quality Comparison", fontsize=12)

        for i_d, (_, r_d) in enumerate(d_sub.iterrows()):
            r_img = cv2.cvtColor(cv2.imread(r_d["source_path"]), cv2.COLOR_BGR2RGB)
            p_img = cv2.cvtColor(cv2.imread(r_d["processed_path"]), cv2.COLOR_BGR2RGB)

            axes_d[i_d, 0].imshow(r_img)
            axes_d[i_d, 0].set_title(f"RAW ({r_d['class_name']}): {r_d['image_id']}", fontsize=8)
            axes_d[i_d, 0].axis("off")

            axes_d[i_d, 1].imshow(p_img)
            axes_d[i_d, 1].set_title(f"PROCESSED (224x224 RGB)", fontsize=8)
            axes_d[i_d, 1].axis("off")

        plt.tight_layout()
        dset_grid_path = os.path.join(out_dir, f"{dset}_before_after.png")
        plt.savefig(dset_grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Saved {dset} comparison grid: {dset_grid_path}")

    print("\nVisual check generation completed!\n")


if __name__ == "__main__":
    generate_visual_checks()
