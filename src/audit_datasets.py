"""
src/audit_datasets.py

Comprehensive dataset audit script for Glaucoma Detection Project.
Audits DRISHTI-GS1, HRF Dataset, and RIM-ONE DL without altering raw data.
Outputs detailed metrics, sanity checks, and potential data issues.
"""

import os
import glob
import json
import yaml
import pandas as pd
from PIL import Image


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def audit_drishti(drishti_base_path):
    print("=" * 60)
    print("AUDITING DRISHTI-GS1 DATASET")
    print("=" * 60)

    audit_res = {}

    drishti_inner = os.path.join(
        drishti_base_path, "Drishti-GS1_files", "Drishti-GS1_files"
    )
    if not os.path.exists(drishti_inner):
        drishti_inner = os.path.join(drishti_base_path, "Drishti-GS1_files")
        if not os.path.exists(drishti_inner):
            drishti_inner = drishti_base_path

    excel_diag = os.path.join(
        drishti_base_path, "Drishti-GS1_files", "Drishti-GS1_diagnosis.xlsx"
    )
    if not os.path.exists(excel_diag):
        excel_diag = os.path.join(drishti_base_path, "Drishti-GS1_diagnosis.xlsx")

    excel_notch = os.path.join(
        drishti_base_path, "Drishti-GS1_files", "notching_image_decisions.xls"
    )
    if not os.path.exists(excel_notch):
        excel_notch = os.path.join(drishti_base_path, "notching_image_decisions.xls")

    train_img_dir = os.path.join(drishti_inner, "Training", "Images")
    test_img_dir = os.path.join(drishti_inner, "Test", "Images")

    train_files = sorted(list(set(glob.glob(os.path.join(train_img_dir, "*.*"))))) if os.path.exists(train_img_dir) else []
    test_files = sorted(list(set(glob.glob(os.path.join(test_img_dir, "*.*"))))) if os.path.exists(test_img_dir) else []

    all_img_paths = train_files + test_files

    audit_res["train_images_count"] = len(train_files)
    audit_res["test_images_count"] = len(test_files)
    audit_res["total_images_count"] = len(all_img_paths)

    unreadable = []
    dimensions = set()
    extensions = set()
    file_map = {}

    for img_path in all_img_paths:
        ext = os.path.splitext(img_path)[1].lower()
        extensions.add(ext)
        fname = os.path.basename(img_path)
        img_id = os.path.splitext(fname)[0]

        part = "training" if "Training" in img_path else ("test" if "Test" in img_path else "unknown")

        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                dimensions.add((w, h, mode))
        except Exception as e:
            unreadable.append((img_path, str(e)))

        file_map[img_id] = {
            "full_path": img_path,
            "filename": fname,
            "original_partition": part,
        }

    audit_res["unreadable_images"] = unreadable
    audit_res["dimensions"] = [list(d) for d in dimensions]
    audit_res["extensions"] = list(extensions)

    diag_records = {}
    if os.path.exists(excel_diag):
        df_raw = pd.read_excel(excel_diag, header=None)
        header_idx = None
        for idx, row in df_raw.iterrows():
            if "Drishti-GS File" in row.values:
                header_idx = idx
                break

        if header_idx is not None:
            cols = df_raw.iloc[header_idx].values
            df_diag = df_raw.iloc[header_idx + 1:].copy()
            df_diag.columns = cols
            df_diag = df_diag.dropna(subset=["Drishti-GS File"])

            df_diag["clean_id"] = (
                df_diag["Drishti-GS File"]
                .astype(str)
                .str.strip()
                .str.replace("'", "")
                .str.replace("'", "")
            )
            df_diag["clean_patient_id"] = (
                df_diag["Patient ID"].astype(str).str.strip()
            )
            df_diag["clean_total"] = (
                df_diag["Total"].astype(str).str.strip()
            )

            for _, row in df_diag.iterrows():
                cid = row["clean_id"]
                diag_records[cid] = {
                    "patient_id": row["clean_patient_id"],
                    "total_label": row["clean_total"],
                    "raw_filename": row["Drishti-GS File"],
                }

    audit_res["excel_records_count"] = len(diag_records)

    matched_ids = []
    unmatched_excel_records = []
    images_without_label = []

    for img_id, img_info in file_map.items():
        if img_id in diag_records:
            matched_ids.append(img_id)
            img_info.update(diag_records[img_id])
        else:
            images_without_label.append(img_id)

    for cid, rec in diag_records.items():
        if cid not in file_map:
            unmatched_excel_records.append(cid)

    audit_res["matched_count"] = len(matched_ids)
    audit_res["unmatched_excel_records"] = unmatched_excel_records
    audit_res["images_without_label"] = images_without_label

    normal_count = 0
    glaucoma_count = 0
    unknown_class_count = 0

    for img_id in matched_ids:
        tot = file_map[img_id]["total_label"].lower()
        if "glaucomatous" in tot or tot == "glaucoma":
            glaucoma_count += 1
            file_map[img_id]["class_name"] = "Glaucoma"
            file_map[img_id]["label"] = 1
        elif "normal" in tot:
            normal_count += 1
            file_map[img_id]["class_name"] = "Normal"
            file_map[img_id]["label"] = 0
        else:
            unknown_class_count += 1

    audit_res["normal_count"] = normal_count
    audit_res["glaucoma_count"] = glaucoma_count
    audit_res["unknown_class_count"] = unknown_class_count

    patient_map = {}
    patient_partitions = {}

    for img_id in matched_ids:
        pid = file_map[img_id]["patient_id"]
        part = file_map[img_id]["original_partition"]

        patient_map.setdefault(pid, []).append(img_id)
        patient_partitions.setdefault(pid, set()).add(part)

    unique_patients = len(patient_map)
    audit_res["unique_patients_count"] = unique_patients

    multi_img_patients = {pid: imgs for pid, imgs in patient_map.items() if len(imgs) > 1}
    audit_res["multi_image_patients_count"] = len(multi_img_patients)
    audit_res["multi_image_patients_details"] = multi_img_patients

    cross_partition_patients = {
        pid: list(parts)
        for pid, parts in patient_partitions.items()
        if len(parts) > 1
    }
    audit_res["cross_partition_patients"] = cross_partition_patients

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
        except Exception as e:
            pass

    audit_res["notching_records_matched"] = len(notch_map)

    print(f"Original Training images count: {len(train_files)}")
    print(f"Original Test images count: {len(test_files)}")
    print(f"Total Pooled DRISHTI images: {len(all_img_paths)}")
    print(f"Matched Excel Diagnosis records: {len(matched_ids)}")
    print(f"Normal count: {normal_count}")
    print(f"Glaucoma count: {glaucoma_count}")
    print(f"Unique Patient IDs: {unique_patients}")
    print(f"Patients appearing in BOTH original training/test: {len(cross_partition_patients)}")

    return audit_res


def audit_hrf(hrf_base_path):
    print("\n" + "=" * 60)
    print("AUDITING HRF DATASET")
    print("=" * 60)

    audit_res = {}
    images_dir = os.path.join(hrf_base_path, "images")
    if not os.path.exists(images_dir):
        images_dir = hrf_base_path

    all_files = glob.glob(os.path.join(hrf_base_path, "**", "*.*"), recursive=True)

    # De-duplicate case-insensitive matching on Windows
    raw_jpgs = set()
    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg")):
            raw_jpgs.add(os.path.join(images_dir, f))

    primary_jpgs = sorted(list(raw_jpgs))

    g_count = 0
    h_count = 0
    d_count = 0
    unrecognized = []
    unreadable = []
    dimensions = set()

    for f in primary_jpgs:
        fname = os.path.basename(f)
        stem = os.path.splitext(fname)[0]

        try:
            with Image.open(f) as img:
                w, h = img.size
                mode = img.mode
                dimensions.add((w, h, mode))
        except Exception as e:
            unreadable.append((f, str(e)))

        if stem.endswith("_g"):
            g_count += 1
        elif stem.endswith("_h"):
            h_count += 1
        elif stem.endswith("_d") or stem.endswith("_dr"):
            d_count += 1
        else:
            unrecognized.append(fname)

    root_extra = [
        f for f in os.listdir(hrf_base_path)
        if os.path.isfile(os.path.join(hrf_base_path, f)) and f.lower().endswith((".jpg", ".png", ".tif"))
    ]

    audit_res["total_files_in_dir"] = len(all_files)
    audit_res["primary_jpg_count"] = len(primary_jpgs)
    audit_res["g_count"] = g_count
    audit_res["h_count"] = h_count
    audit_res["d_count"] = d_count
    audit_res["unrecognized_count"] = len(unrecognized)
    audit_res["unrecognized_files"] = unrecognized
    audit_res["unreadable_images"] = unreadable
    audit_res["dimensions"] = [list(d) for d in dimensions]
    audit_res["root_extra_files_count"] = len(root_extra)

    print(f"Primary HRF images count (in images/): {len(primary_jpgs)}")
    print(f"  _g (Glaucoma, label 1): {g_count}")
    print(f"  _h (Healthy/Normal, label 0): {h_count}")
    print(f"  _d (Diabetic Retinopathy, EXCLUDED): {d_count}")
    print(f"Unrecognized filenames in images/: {len(unrecognized)}")
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Image dimensions: {dimensions}")

    return audit_res


def audit_rimone(rimone_base_path):
    print("\n" + "=" * 60)
    print("AUDITING RIM-ONE DL DATASET")
    print("=" * 60)

    audit_res = {}
    inner_path = os.path.join(rimone_base_path, "RIM-ONE_DL_images")
    if not os.path.exists(inner_path):
        inner_path = rimone_base_path

    partitions = [
        d for d in os.listdir(inner_path)
        if os.path.isdir(os.path.join(inner_path, d))
    ]
    audit_res["partitions_found"] = partitions

    # 1. Audit partitioned_by_hospital
    hosp_path = os.path.join(inner_path, "partitioned_by_hospital")
    hosp_audit = {}

    if os.path.exists(hosp_path):
        test_dir = os.path.join(hosp_path, "test_set")
        train_dir = os.path.join(hosp_path, "training_set")

        if os.path.exists(test_dir):
            norm_dir = os.path.join(test_dir, "normal")
            glau_dir = os.path.join(test_dir, "glaucoma")

            norm_files = sorted(list(set(glob.glob(os.path.join(norm_dir, "*.*"))))) if os.path.exists(norm_dir) else []
            glau_files = sorted(list(set(glob.glob(os.path.join(glau_dir, "*.*"))))) if os.path.exists(glau_dir) else []

            hosp_audit["test_normal_count"] = len(norm_files)
            hosp_audit["test_glaucoma_count"] = len(glau_files)
            hosp_audit["test_total_count"] = len(norm_files) + len(glau_files)

            prefix_breakdown = {}
            unreadable = []
            dimensions = set()

            for f in norm_files + glau_files:
                fname = os.path.basename(f)
                prefix = fname.split("_")[0] if "_" in fname else "unknown"
                prefix_breakdown[prefix] = prefix_breakdown.get(prefix, 0) + 1

                try:
                    with Image.open(f) as img:
                        w, h = img.size
                        mode = img.mode
                        dimensions.add((w, h, mode))
                except Exception as e:
                    unreadable.append((f, str(e)))

            hosp_audit["prefix_breakdown"] = prefix_breakdown
            hosp_audit["unreadable_images"] = unreadable
            hosp_audit["dimensions"] = [list(d) for d in dimensions]

            print("partitioned_by_hospital -> test_set (EXTERNAL TEST):")
            print(f"  Normal count (label 0): {len(norm_files)}")
            print(f"  Glaucoma count (label 1): {len(glau_files)}")
            print(f"  Total Hospital TEST: {len(norm_files) + len(glau_files)}")
            print(f"  Hospital Filename Prefixes: {prefix_breakdown}")
            print(f"  Unreadable images: {len(unreadable)}")
            print(f"  Image dimensions count: {len(dimensions)}")

        if os.path.exists(train_dir):
            norm_tr = sorted(list(set(glob.glob(os.path.join(train_dir, "normal", "*.*"))))) if os.path.exists(os.path.join(train_dir, "normal")) else []
            glau_tr = sorted(list(set(glob.glob(os.path.join(train_dir, "glaucoma", "*.*"))))) if os.path.exists(os.path.join(train_dir, "glaucoma")) else []
            hosp_audit["training_normal_count"] = len(norm_tr)
            hosp_audit["training_glaucoma_count"] = len(glau_tr)
            hosp_audit["training_total_count"] = len(norm_tr) + len(glau_tr)
            print("partitioned_by_hospital -> training_set (EXCLUDED FROM EXPERIMENT):")
            print(f"  Normal count: {len(norm_tr)}, Glaucoma count: {len(glau_tr)}, Total: {len(norm_tr) + len(glau_tr)}")

    audit_res["partitioned_by_hospital"] = hosp_audit

    # 2. Audit partitioned_randomly (FOR REFERENCE ONLY)
    rand_path = os.path.join(inner_path, "partitioned_randomly")
    rand_audit = {}

    if os.path.exists(rand_path):
        test_rand = os.path.join(rand_path, "test_set")
        train_rand = os.path.join(rand_path, "training_set")

        if os.path.exists(test_rand):
            rand_audit["test_set_files"] = len(set(glob.glob(os.path.join(test_rand, "**", "*.*"), recursive=True)))
        if os.path.exists(train_rand):
            rand_audit["training_set_files"] = len(set(glob.glob(os.path.join(train_rand, "**", "*.*"), recursive=True)))

        print("\npartitioned_randomly (FOR REFERENCE ONLY - DO NOT USE):")
        print(f"  test_set files: {rand_audit.get('test_set_files')}")
        print(f"  training_set files: {rand_audit.get('training_set_files')}")

    audit_res["partitioned_randomly"] = rand_audit

    return audit_res


def main():
    config = load_config()
    drishti_path = config["DRISHTI_PATH"]
    hrf_path = config["HRF_PATH"]
    rimone_path = config["RIMONE_PATH"]
    project_root = config["PROJECT_ROOT"]

    print("Starting Dataset Audit...")
    print(f"Project Root: {project_root}\n")

    drishti_res = audit_drishti(drishti_path)
    hrf_res = audit_hrf(hrf_path)
    rimone_res = audit_rimone(rimone_path)

    summary_report = {
        "DRISHTI": drishti_res,
        "HRF": hrf_res,
        "RIM_ONE": rimone_res,
    }

    audit_out_dir = os.path.join(project_root, "results", "audit")
    os.makedirs(audit_out_dir, exist_ok=True)

    json_path = os.path.join(audit_out_dir, "audit_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    txt_path = os.path.join(audit_out_dir, "audit_report.txt")
    with open(txt_path, "w") as f:
        f.write("============================================================\n")
        f.write("DATASET AUDIT REPORT - GLAUCOMA DETECTION PROJECT\n")
        f.write("============================================================\n\n")

        f.write("1. DRISHTI-GS1 AUDIT SUMMARY\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Original Training Images Count : {drishti_res.get('train_images_count')}\n")
        f.write(f"Original Test Images Count     : {drishti_res.get('test_images_count')}\n")
        f.write(f"Total Pooled DRISHTI Images    : {drishti_res.get('total_images_count')}\n")
        f.write(f"Matched Excel Diagnosis Images : {drishti_res.get('matched_count')}\n")
        f.write(f"Normal Count (label 0)         : {drishti_res.get('normal_count')}\n")
        f.write(f"Glaucoma Count (label 1)       : {drishti_res.get('glaucoma_count')}\n")
        f.write(f"Unique Patient IDs             : {drishti_res.get('unique_patients_count')}\n")
        f.write(f"Multi-image Patients Count     : {drishti_res.get('multi_image_patients_count')}\n")
        f.write(f"Patients in BOTH Train & Test   : {len(drishti_res.get('cross_partition_patients', {}))}\n")
        f.write(f"Unmatched Excel Records        : {len(drishti_res.get('unmatched_excel_records', []))}\n")
        f.write(f"Images without Label           : {len(drishti_res.get('images_without_label', []))}\n")
        f.write(f"Unreadable Images              : {len(drishti_res.get('unreadable_images', []))}\n")
        f.write(f"File Extensions                : {drishti_res.get('extensions')}\n\n")

        f.write("2. HRF AUDIT SUMMARY\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Primary JPG Images Count (images/): {hrf_res.get('primary_jpg_count')}\n")
        f.write(f"  _g (Glaucoma, label 1)          : {hrf_res.get('g_count')}\n")
        f.write(f"  _h (Healthy/Normal, label 0)    : {hrf_res.get('h_count')}\n")
        f.write(f"  _d (Diabetic Retinopathy)       : {hrf_res.get('d_count')} [EXCLUDE FROM BINARY ML]\n")
        f.write(f"Unrecognized Filenames            : {hrf_res.get('unrecognized_count')}\n")
        f.write(f"Unreadable Images                 : {len(hrf_res.get('unreadable_images', []))}\n")
        f.write(f"Dimensions Found                  : {hrf_res.get('dimensions')}\n\n")

        f.write("3. RIM-ONE DL AUDIT SUMMARY\n")
        f.write("------------------------------------------------------------\n")
        hosp = rimone_res.get("partitioned_by_hospital", {})
        f.write(f"Hospital TEST Normal Count        : {hosp.get('test_normal_count')}\n")
        f.write(f"Hospital TEST Glaucoma Count      : {hosp.get('test_glaucoma_count')}\n")
        f.write(f"Hospital TEST Total Count         : {hosp.get('test_total_count')}\n")
        f.write(f"Hospital TEST Filename Prefixes   : {hosp.get('prefix_breakdown')}\n")
        f.write(f"Unreadable Images in Hosp Test    : {len(hosp.get('unreadable_images', []))}\n")
        f.write(f"Hospital Training (Unused)        : Normal={hosp.get('training_normal_count')}, Glaucoma={hosp.get('training_glaucoma_count')}\n")

    print(f"\nAudit complete! Reports saved in {audit_out_dir}\n")


if __name__ == "__main__":
    main()
