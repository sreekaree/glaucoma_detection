"""
src/create_manifest.py

Generates the authoritative master manifest file (metadata/master_manifest.csv).
Captures image level metadata, source paths, labels, patient IDs, original partitions,
5-fold CV assignments for DRISHTI, external test roles for HRF and RIM-ONE, and
discordant patient flags.
"""

import os
import glob
import yaml
import pandas as pd
from PIL import Image


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_drishti_records(drishti_base, project_root):
    records = []
    drishti_inner = os.path.join(drishti_base, "Drishti-GS1_files", "Drishti-GS1_files")
    if not os.path.exists(drishti_inner):
        drishti_inner = os.path.join(drishti_base, "Drishti-GS1_files")

    # Load 5-fold splits
    splits_csv = os.path.join(project_root, "splits", "drishti_5fold_splits.csv")
    df_splits = pd.read_csv(splits_csv)
    splits_map = df_splits.set_index("image_id").to_dict("index")

    excel_notch = os.path.join(drishti_base, "Drishti-GS1_files", "notching_image_decisions.xls")
    if not os.path.exists(excel_notch):
        excel_notch = os.path.join(drishti_base, "notching_image_decisions.xls")

    notch_map = {}
    if os.path.exists(excel_notch):
        try:
            df_notch = pd.read_excel(excel_notch)
            num_col = [c for c in df_notch.columns if "IMAGE" in str(c).upper() or "NUMBER" in str(c).upper()][0]
            notch_col = [c for c in df_notch.columns if "NOTCHING" in str(c).upper() or "DIRECTION" in str(c).upper()][0]

            for _, row in df_notch.iterrows():
                num = row[num_col]
                n_dir = row[notch_col]
                if pd.notna(num):
                    try:
                        num_int = int(num)
                        formatted_id = f"drishtiGS_{num_int:03d}"
                        notch_map[formatted_id] = str(n_dir).strip()
                    except ValueError:
                        pass
        except Exception:
            pass

    train_files = glob.glob(os.path.join(drishti_inner, "Training", "Images", "*.*"))
    test_files = glob.glob(os.path.join(drishti_inner, "Test", "Images", "*.*"))

    for img_p in train_files + test_files:
        fname = os.path.basename(img_p)
        img_id = os.path.splitext(fname)[0]
        part = "training" if "Training" in img_p else "test"

        s_info = splits_map.get(img_id, {})
        pid = s_info.get("patient_id", "unknown")
        lbl = s_info.get("label", -1)
        cls = s_info.get("class_name", "Unknown")
        fold_idx = s_info.get("fold", -1)
        is_disc = s_info.get("discordant_patient", False)

        w, h = None, None
        try:
            with Image.open(img_p) as img:
                w, h = img.size
        except Exception:
            pass

        records.append({
            "image_id": img_id,
            "dataset": "DRISHTI",
            "patient_id": pid,
            "label": lbl,
            "class_name": cls,
            "source_path": img_p,
            "original_partition": part,
            "split": "development",
            "fold": fold_idx,
            "discordant_patient": is_disc,
            "hospital": "N/A",
            "source_partition": f"DRISHTI_{part}",
            "processed_path": os.path.join(project_root, "data", "processed", "DRISHTI", cls.lower(), f"{img_id}.png"),
            "image_width": w,
            "image_height": h,
            "preprocessing_status": "raw",
            "exclusion_reason": "none",
            "notching_direction": notch_map.get(img_id, "N/A")
        })

    return records


def build_hrf_records(hrf_base, project_root):
    records = []
    images_dir = os.path.join(hrf_base, "images")
    if not os.path.exists(images_dir):
        images_dir = hrf_base

    raw_jpgs = set()
    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg")):
            raw_jpgs.add(os.path.join(images_dir, f))

    for img_p in sorted(list(raw_jpgs)):
        fname = os.path.basename(img_p)
        stem = os.path.splitext(fname)[0]

        if stem.endswith("_g"):
            lbl = 1
            cls = "Glaucoma"
            split = "external_test"
            excl = "none"
            proc_p = os.path.join(project_root, "data", "processed", "HRF", "glaucoma", f"HRF_{stem}.png")
        elif stem.endswith("_h"):
            lbl = 0
            cls = "Normal"
            split = "external_test"
            excl = "none"
            proc_p = os.path.join(project_root, "data", "processed", "HRF", "normal", f"HRF_{stem}.png")
        elif stem.endswith("_d") or stem.endswith("_dr"):
            lbl = -1
            cls = "Diabetic_Retinopathy"
            split = "excluded"
            excl = "diabetic_retinopathy"
            proc_p = ""
        else:
            lbl = -1
            cls = "Unknown"
            split = "excluded"
            excl = "unrecognized_filename"
            proc_p = ""

        w, h = None, None
        try:
            with Image.open(img_p) as img:
                w, h = img.size
        except Exception:
            pass

        records.append({
            "image_id": f"HRF_{stem}",
            "dataset": "HRF",
            "patient_id": f"HRF_{stem.split('_')[0]}",
            "label": lbl,
            "class_name": cls,
            "source_path": img_p,
            "original_partition": "HRF_images",
            "split": split,
            "fold": -1,
            "discordant_patient": False,
            "hospital": "N/A",
            "source_partition": "HRF_primary",
            "processed_path": proc_p,
            "image_width": w,
            "image_height": h,
            "preprocessing_status": "raw" if split != "excluded" else "excluded",
            "exclusion_reason": excl,
            "notching_direction": "N/A"
        })

    return records


def build_rimone_records(rimone_base, project_root):
    records = []
    inner_path = os.path.join(rimone_base, "RIM-ONE_DL_images")
    if not os.path.exists(inner_path):
        inner_path = rimone_base

    hosp_path = os.path.join(inner_path, "partitioned_by_hospital")
    test_dir = os.path.join(hosp_path, "test_set")
    train_dir = os.path.join(hosp_path, "training_set")

    for s_name, s_dir in [("test_set", test_dir), ("training_set", train_dir)]:
        if os.path.exists(s_dir):
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        img_p = os.path.join(root, f)
                        fname = os.path.basename(f)
                        stem = os.path.splitext(fname)[0]
                        prefix = fname.split("_")[0] if "_" in fname else "unknown"

                        cls_folder = os.path.basename(root).lower()
                        if "glaucoma" in cls_folder:
                            lbl = 1
                            cls = "Glaucoma"
                        elif "normal" in cls_folder:
                            lbl = 0
                            cls = "Normal"
                        else:
                            lbl = -1
                            cls = "Unknown"

                        split = "external_test"  # ALL 485 RIM-ONE images are external_test
                        excl = "none"

                        w, h = None, None
                        try:
                            with Image.open(img_p) as img:
                                w, h = img.size
                        except Exception:
                            pass

                        records.append({
                            "image_id": f"RIMONE_{stem}",
                            "dataset": "RIM-ONE",
                            "patient_id": f"RIMONE_{stem}",
                            "label": lbl,
                            "class_name": cls,
                            "source_path": img_p,
                            "original_partition": f"hospital_{s_name}",
                            "split": split,
                            "fold": -1,
                            "discordant_patient": False,
                            "hospital": prefix,
                            "source_partition": f"partitioned_by_hospital/{s_name}",
                            "processed_path": os.path.join(project_root, "data", "processed", "RIM-ONE", cls.lower(), f"RIMONE_{stem}.png"),
                            "image_width": w,
                            "image_height": h,
                            "preprocessing_status": "raw",
                            "exclusion_reason": excl,
                            "notching_direction": "N/A"
                        })

    return records


def main():
    config = load_config()
    drishti_path = config["DRISHTI_PATH"]
    hrf_path = config["HRF_PATH"]
    rimone_path = config["RIMONE_PATH"]
    project_root = config["PROJECT_ROOT"]

    print("Generating Master Manifest...")

    d_records = build_drishti_records(drishti_path, project_root)
    h_records = build_hrf_records(hrf_path, project_root)
    r_records = build_rimone_records(rimone_path, project_root)

    all_records = d_records + h_records + r_records
    df_manifest = pd.DataFrame(all_records)

    meta_dir = os.path.join(project_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    manifest_path = os.path.join(meta_dir, "master_manifest.csv")

    df_manifest.to_csv(manifest_path, index=False)

    print(f"\nMaster manifest successfully created: {manifest_path}")
    print(f"Total rows in manifest: {len(df_manifest)}")
    print("\nDataset breakdown in manifest:")
    print(df_manifest["dataset"].value_counts())
    print("\nSplit breakdown in manifest:")
    print(df_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
