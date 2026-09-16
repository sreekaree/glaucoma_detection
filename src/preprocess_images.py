"""
src/preprocess_images.py

Preprocessing pipeline for Glaucoma Detection Project.
Applies:
  1. Readability validation
  2. Non-aggressive black FOV background cropping (preserving retina & optic disc)
  3. Aspect-ratio preserving resize and square padding to 224x224x3 RGB
  4. CLAHE on GREEN CHANNEL ONLY (clip_limit=2.0, tile_grid_size=(8,8))
  5. Saves processed images and updates metadata/master_manifest.csv
"""

import os
import yaml
import cv2
import numpy as np
import pandas as pd


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "paths.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def crop_black_fov(img, thresh_val=12):
    """
    Detects and crops unnecessary black background while strictly preserving
    the full retinal field, optic disc, macula, and vessels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mask = gray > thresh_val

    if not np.any(mask):
        return img  # fallback if all black

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    h, w = gray.shape
    # Add a small 2% safety margin to ensure no retinal boundary clipping
    margin_y = int(h * 0.02)
    margin_x = int(w * 0.02)

    y0 = max(0, y0 - margin_y)
    y1 = min(h, y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(w, x1 + margin_x)

    cropped = img[y0:y1, x0:x1]
    return cropped


def resize_and_pad(img, target_size=224):
    """
    Resizes image while preserving aspect ratio, then pads symmetrically to target_size x target_size.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def apply_green_channel_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE ONLY to the Green channel of a BGR/RGB image.
    Retains original Red and Blue channels intact.
    """
    b, g, r = cv2.split(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g_clahe = clahe.apply(g)
    processed_bgr = cv2.merge([b, g_clahe, r])
    return processed_bgr


def preprocess_all(clip_limit=2.0, tile_grid_size=(8, 8)):
    config = load_config()
    project_root = config["PROJECT_ROOT"]
    manifest_path = os.path.join(project_root, "metadata", "master_manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    print("=" * 60)
    print("PREPROCESSING IMAGES")
    print("=" * 60)
    print(f"Total manifest entries: {len(df)}")
    print(f"CLAHE Parameters: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    print(f"Target Size: 224 x 224 x 3 (RGB)\n")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        if row["split"] == "excluded":
            df.loc[idx, "preprocessing_status"] = "excluded"
            skipped_count += 1
            continue

        src_p = row["source_path"]
        proc_p = row["processed_path"]

        if not os.path.exists(src_p):
            print(f"ERROR: Source file missing: {src_p}")
            df.loc[idx, "preprocessing_status"] = "source_missing"
            error_count += 1
            continue

        try:
            # Read BGR image
            img_bgr = cv2.imread(src_p, cv2.IMREAD_COLOR)
            if img_bgr is None or img_bgr.size == 0:
                raise ValueError("cv2.imread failed to decode image")

            # 1. FOV Crop (black background removal only)
            cropped_bgr = crop_black_fov(img_bgr)

            # 2. Aspect-ratio preserving resize + pad to 224x224
            padded_bgr = resize_and_pad(cropped_bgr, target_size=224)

            # 3. CLAHE on GREEN CHANNEL ONLY
            final_bgr = apply_green_channel_clahe(
                padded_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size
            )

            # Save processed image as RGB PNG
            os.makedirs(os.path.dirname(proc_p), exist_ok=True)

            # Convert BGR to RGB before saving or write BGR using cv2
            # cv2.imwrite expects BGR order
            success = cv2.imwrite(proc_p, final_bgr)

            if not success:
                raise IOError(f"cv2.imwrite failed to save {proc_p}")

            df.loc[idx, "preprocessing_status"] = "success"
            processed_count += 1

            if processed_count % 100 == 0 or processed_count == len(df) - skipped_count:
                print(f"Processed {processed_count} / {len(df) - skipped_count} images...")

        except Exception as e:
            print(f"ERROR processing {src_p}: {e}")
            df.loc[idx, "preprocessing_status"] = f"error: {str(e)}"
            error_count += 1

    df.to_csv(manifest_path, index=False)
    print("\nPREPROCESSING COMPLETED!")
    print(f"Successfully processed : {processed_count}")
    print(f"Skipped (Excluded)     : {skipped_count}")
    print(f"Errors                 : {error_count}")
    print(f"Updated manifest saved : {manifest_path}\n")

    return df


if __name__ == "__main__":
    preprocess_all(clip_limit=2.0, tile_grid_size=(8, 8))
