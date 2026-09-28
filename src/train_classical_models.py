"""
train_classical_models.py
=========================
Classical ML experiment — Stage 7: Training & Evaluation.

Experimental design
-------------------
Development set : DRISHTI (101 images, patient-level 5-fold CV)
External tests  : HRF (30 images), RIM-ONE (485 images from
                  partitioned_by_hospital only)

Feature sets
------------
  A  GLCM only          (12 features)
  B  GLRLM only         (11 features)
  C  GLCM + GLRLM       (23 features)

Classifiers
-----------
  1  SVM   (RBF kernel, class_weight='balanced')
  2  Random Forest      (class_weight='balanced')
  3  XGBoost

Leakage prevention
------------------
OUTER LOOP (5-fold):
  - Outer fold assignments are FIXED from splits/drishti_5fold_splits.csv.
  - Folds were produced by StratifiedGroupKFold(n_splits=5, shuffle=True,
    random_state=42) with groups = patient_id. NEVER regenerated here.

INNER LOOP (hyperparameter search):
  - Uses StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    with groups = patient_id of the TRAINING portion only.
  - This ensures no patient appears in both an inner train and inner val
    split, matching the outer grouping strategy.
  - GridSearchCV receives the training-fold patient_ids as groups via
    fit(X, y, groups=patient_ids_train).

SCALING:
  - StandardScaler is inside the sklearn Pipeline.
  - Pipeline.fit() is called on the outer training folds only.
  - Val and external sets are only ever passed through Pipeline.predict /
    Pipeline.predict_proba (transform-only path).

EXTERNAL SETS:
  - HRF and RIM-ONE are NEVER passed to any .fit() call.
  - No tuning, scaling fitting, or threshold selection uses external data.
  - Threshold is fixed at 0.5 for all external evaluation.

Hyperparameter grids  (small, reproducible)
--------------------
SVM:
  C     : [0.1, 1.0, 10.0]
  gamma : ['scale', 'auto']

Random Forest:
  n_estimators     : [100, 200]
  max_depth        : [None, 10, 20]
  min_samples_leaf : [1, 3]

XGBoost:
  n_estimators  : [100, 200]
  max_depth     : [3, 5]
  learning_rate : [0.05, 0.1]
  subsample     : [0.8, 1.0]

Outputs (results/classical_ml/)
-------------------------------
  fold_metrics/        per-fold metric CSV for every (model, feature_set)
  predictions/         OOF + external test predictions (all folds + averaged)
  confusion_matrices/  PNG confusion matrices
  roc_curves/          ROC curve data CSVs + PNG plots
  summaries/           aggregated mean+/-std CSV, JSON, final comparison table

Usage
-----
  python src/train_classical_models.py [--dry-run]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed — XGBoost classifier will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3      # inner StratifiedGroupKFold for hyperparameter search
SCORING       = "roc_auc"
THRESHOLD     = 0.5    # fixed classification threshold for external evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILES = {
    "GLCM":       PROJECT_ROOT / "features" / "glcm_features.csv",
    "GLRLM":      PROJECT_ROOT / "features" / "glrlm_features.csv",
    "GLCM+GLRLM": PROJECT_ROOT / "features" / "glcm_glrlm_features.csv",
}

RESULTS_ROOT = PROJECT_ROOT / "results" / "classical_ml"

# All columns that are NOT numeric feature values
META_COLS = {
    "image_id", "dataset", "patient_id", "label", "class_name",
    "source_path", "processed_path", "split", "fold",
    "hospital", "source_partition", "discordant_patient",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_classifiers():
    """Return list of (name, fresh_pipeline, param_grid) tuples.
    GridSearchCV clones the pipeline internally, so sharing one instance
    across folds is safe; we deepcopy anyway for clarity.
    """
    clfs = []

    clfs.append((
        "SVM",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_SEED, max_iter=5000)),
        ]),
        {
            "clf__C":     [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
    ))

    clfs.append((
        "RandomForest",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        {
            "clf__n_estimators":     [100, 200],
            "clf__max_depth":        [None, 10, 20],
            "clf__min_samples_leaf": [1, 3],
        },
    ))

    if XGBOOST_AVAILABLE:
        clfs.append((
            "XGBoost",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=0,
                )),
            ]),
            {
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample":     [0.8, 1.0],
            },
        ))

    return clfs


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_data(feature_set_name: str):
    """Load feature CSV; split by dataset.

    Returns
    -------
    df_drishti : development pool (101 rows, has 'fold' and 'patient_id')
    df_hrf     : HRF usable rows only (label in {0,1}, 30 rows)
    df_rimone  : RIM-ONE rows (485 rows, has 'hospital' in {r1,r2,r3})
    feat_cols  : list of numeric feature column names
    """
    path = FEATURE_FILES[feature_set_name]
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    df_drishti = (df[df["dataset"] == "DRISHTI"]
                  .copy().reset_index(drop=True))
    df_hrf     = (df[(df["dataset"] == "HRF") & df["label"].isin([0, 1])]
                  .copy().reset_index(drop=True))
    df_rimone  = (df[df["dataset"] == "RIM-ONE"]
                  .copy().reset_index(drop=True))

    return df_drishti, df_hrf, df_rimone, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Binary classification metrics at threshold=0.5."""
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "sensitivity":       recall_score(y_true, y_pred, zero_division=0),
        "specificity":       recall_score(y_true, y_pred,
                                          pos_label=0, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Glaucoma"],
                yticklabels=["Normal", "Glaucoma"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, title, png_path, csv_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    # Save CSV
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv(csv_path, index=False)
    # Save PNG
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PER-EXPERIMENT TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(feature_set_name: str,
                           clf_name: str,
                           pipeline,
                           param_grid: dict,
                           results_root: Path,
                           dry_run: bool = False):
    """
    Run 5-fold CV for one (feature_set, classifier) pair.

    INNER CV uses StratifiedGroupKFold(n_splits=3) with patient_id groups
    so that no patient leaks between inner train and inner val.
    """
    print(f"\n  [EXPERIMENT] {clf_name} | {feature_set_name}")

    df_drishti, df_hrf, df_rimone, feat_cols = load_feature_data(feature_set_name)

    X_hrf    = df_hrf[feat_cols].values.astype(float)
    y_hrf    = df_hrf["label"].values.astype(int)
    X_rimone = df_rimone[feat_cols].values.astype(float)
    y_rimone = df_rimone["label"].values.astype(int)
    hosp_rimone = df_rimone["hospital"].values  # r1 / r2 / r3

    exp_tag  = f"{clf_name}__{feature_set_name.replace('+', '_plus_')}"
    cm_dir   = results_root / "confusion_matrices" / exp_tag
    pred_dir = results_root / "predictions"        / exp_tag
    roc_dir  = results_root / "roc_curves"         / exp_tag
    fold_dir = results_root / "fold_metrics"

    for d in [cm_dir, pred_dir, roc_dir, fold_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"    [DRY-RUN] Would train {clf_name} on {feature_set_name}. Skipping.")
        return None

    # ── INNER CV: StratifiedGroupKFold with patient_id groups ─────────────────
    # This CV object is passed to GridSearchCV. The groups argument is passed
    # to GridSearchCV.fit() via fit_params, so the inner splits respect
    # patient-level grouping just as the outer splits do.
    inner_cv = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                    random_state=RANDOM_SEED)

    fold_metrics_rows = []
    oof_preds_rows    = []
    ext_hrf_probs_per_fold    = []   # list of length-30 arrays
    ext_rimone_probs_per_fold = []   # list of length-485 arrays
    best_params_per_fold = []

    for outer_fold in range(N_OUTER_FOLDS):
        # ── Outer split uses pre-computed fold column ─────────────────────────
        train_mask = df_drishti["fold"] != outer_fold
        val_mask   = df_drishti["fold"] == outer_fold

        X_train = df_drishti.loc[train_mask, feat_cols].values.astype(float)
        y_train = df_drishti.loc[train_mask, "label"].values.astype(int)
        g_train = df_drishti.loc[train_mask, "patient_id"].values  # groups for inner CV

        X_val   = df_drishti.loc[val_mask, feat_cols].values.astype(float)
        y_val   = df_drishti.loc[val_mask, "label"].values.astype(int)
        ids_val = df_drishti.loc[val_mask, "image_id"].values

        n_train_patients = len(np.unique(g_train))
        print(f"    Outer fold {outer_fold}: "
              f"train={len(X_train)} imgs / {n_train_patients} pts, "
              f"val={len(X_val)} imgs")

        # ── Hyperparameter search — patient-grouped inner CV ──────────────────
        gs = GridSearchCV(
            estimator=deepcopy(pipeline),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=SCORING,
            refit=True,
            n_jobs=-1,
            verbose=0,
            error_score="raise",
        )
        # groups= ensures StratifiedGroupKFold receives patient IDs
        gs.fit(X_train, y_train, groups=g_train)

        best_est      = gs.best_estimator_
        best_params   = gs.best_params_
        best_cv_score = gs.best_score_
        best_params_per_fold.append(best_params)

        print(f"      Best params: {best_params} | inner CV AUC={best_cv_score:.4f}")

        # ── Outer validation: held-out DRISHTI fold ───────────────────────────
        y_val_prob = best_est.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= THRESHOLD).astype(int)

        val_m = compute_metrics(y_val, y_val_pred, y_val_prob)
        val_m.update({
            "fold": outer_fold, "classifier": clf_name,
            "feature_set": feature_set_name, "split": "drishti_val",
            "best_params": json.dumps(best_params),
            "best_inner_cv_auc": best_cv_score,
            "n_train": len(X_train), "n_val": len(X_val),
        })
        fold_metrics_rows.append(val_m)

        save_confusion_matrix(
            y_val, y_val_pred,
            title=f"{clf_name} | {feature_set_name} | Fold {outer_fold} Val",
            filepath=cm_dir / f"fold{outer_fold}_drishti_val.png",
        )
        save_roc_curve(
            y_val, y_val_prob,
            title=f"{clf_name} | {feature_set_name} | Fold {outer_fold} DRISHTI Val",
            png_path=roc_dir / f"fold{outer_fold}_drishti_val_roc.png",
            csv_path=roc_dir / f"fold{outer_fold}_drishti_val_roc.csv",
        )

        for img_id, true, pred, prob in zip(ids_val, y_val, y_val_pred, y_val_prob):
            oof_preds_rows.append({
                "image_id": img_id, "outer_fold": outer_fold,
                "y_true": int(true), "y_pred": int(pred),
                "y_prob": float(prob),
            })

        # ── External HRF (transform only) ─────────────────────────────────────
        y_hrf_prob = best_est.predict_proba(X_hrf)[:, 1]
        y_hrf_pred = (y_hrf_prob >= THRESHOLD).astype(int)
        ext_hrf_probs_per_fold.append(y_hrf_prob)

        hrf_m = compute_metrics(y_hrf, y_hrf_pred, y_hrf_prob)
        hrf_m.update({
            "fold": outer_fold, "classifier": clf_name,
            "feature_set": feature_set_name, "split": "hrf_per_fold",
            "best_params": json.dumps(best_params),
            "best_inner_cv_auc": best_cv_score,
            "n_train": len(X_train), "n_val": len(X_hrf),
        })
        fold_metrics_rows.append(hrf_m)

        # ── External RIM-ONE (transform only) ─────────────────────────────────
        y_rimone_prob = best_est.predict_proba(X_rimone)[:, 1]
        y_rimone_pred = (y_rimone_prob >= THRESHOLD).astype(int)
        ext_rimone_probs_per_fold.append(y_rimone_prob)

        rimone_m = compute_metrics(y_rimone, y_rimone_pred, y_rimone_prob)
        rimone_m.update({
            "fold": outer_fold, "classifier": clf_name,
            "feature_set": feature_set_name, "split": "rimone_per_fold",
            "best_params": json.dumps(best_params),
            "best_inner_cv_auc": best_cv_score,
            "n_train": len(X_train), "n_val": len(X_rimone),
        })
        fold_metrics_rows.append(rimone_m)

        # RIM-ONE per-site (per fold)
        for site in ["r1", "r2", "r3"]:
            mask_s = hosp_rimone == site
            if mask_s.sum() == 0:
                continue
            ys_prob = y_rimone_prob[mask_s]
            ys_pred = (ys_prob >= THRESHOLD).astype(int)
            ys_true = y_rimone[mask_s]
            site_m = compute_metrics(ys_true, ys_pred, ys_prob)
            site_m.update({
                "fold": outer_fold, "classifier": clf_name,
                "feature_set": feature_set_name,
                "split": f"rimone_{site}_per_fold",
                "best_params": json.dumps(best_params),
                "best_inner_cv_auc": best_cv_score,
                "n_train": len(X_train), "n_val": int(mask_s.sum()),
            })
            fold_metrics_rows.append(site_m)

    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE EXTERNAL PREDICTIONS (mean probability across 5 folds)
    # ═══════════════════════════════════════════════════════════════════════

    hrf_avg_prob    = np.mean(ext_hrf_probs_per_fold,    axis=0)
    rimone_avg_prob = np.mean(ext_rimone_probs_per_fold, axis=0)
    hrf_avg_pred    = (hrf_avg_prob >= THRESHOLD).astype(int)
    rimone_avg_pred = (rimone_avg_prob >= THRESHOLD).astype(int)

    # Pooled HRF
    hrf_pooled_m = compute_metrics(y_hrf, hrf_avg_pred, hrf_avg_prob)
    hrf_pooled_m.update({
        "fold": "pooled", "classifier": clf_name,
        "feature_set": feature_set_name, "split": "hrf_pooled",
        "best_params": "mean_of_5_folds",
    })
    fold_metrics_rows.append(hrf_pooled_m)

    # Pooled RIM-ONE
    rimone_pooled_m = compute_metrics(y_rimone, rimone_avg_pred, rimone_avg_prob)
    rimone_pooled_m.update({
        "fold": "pooled", "classifier": clf_name,
        "feature_set": feature_set_name, "split": "rimone_pooled",
        "best_params": "mean_of_5_folds",
    })
    fold_metrics_rows.append(rimone_pooled_m)

    # Pooled RIM-ONE per site
    for site in ["r1", "r2", "r3"]:
        mask_s = hosp_rimone == site
        if mask_s.sum() == 0:
            continue
        ys_avg_prob = rimone_avg_prob[mask_s]
        ys_avg_pred = (ys_avg_prob >= THRESHOLD).astype(int)
        ys_true     = y_rimone[mask_s]
        site_pooled_m = compute_metrics(ys_true, ys_avg_pred, ys_avg_prob)
        site_pooled_m.update({
            "fold": "pooled", "classifier": clf_name,
            "feature_set": feature_set_name,
            "split": f"rimone_{site}_pooled",
            "best_params": "mean_of_5_folds",
        })
        fold_metrics_rows.append(site_pooled_m)

    # ── Confusion matrices and ROC curves for pooled predictions ──────────────
    save_confusion_matrix(
        y_hrf, hrf_avg_pred,
        title=f"{clf_name} | {feature_set_name} | HRF (pooled avg)",
        filepath=cm_dir / "hrf_pooled.png",
    )
    save_confusion_matrix(
        y_rimone, rimone_avg_pred,
        title=f"{clf_name} | {feature_set_name} | RIM-ONE (pooled avg)",
        filepath=cm_dir / "rimone_pooled.png",
    )
    save_roc_curve(
        y_hrf, hrf_avg_prob,
        title=f"{clf_name} | {feature_set_name} | HRF Pooled",
        png_path=roc_dir / "hrf_pooled_roc.png",
        csv_path=roc_dir / "hrf_pooled_roc.csv",
    )
    save_roc_curve(
        y_rimone, rimone_avg_prob,
        title=f"{clf_name} | {feature_set_name} | RIM-ONE Pooled",
        png_path=roc_dir / "rimone_pooled_roc.png",
        csv_path=roc_dir / "rimone_pooled_roc.csv",
    )

    # ── Save fold metrics CSV ──────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_metrics_rows)
    fold_df.to_csv(fold_dir / f"{exp_tag}.csv", index=False)

    # ── Save OOF predictions ───────────────────────────────────────────────────
    pd.DataFrame(oof_preds_rows).to_csv(
        pred_dir / "drishti_oof_predictions.csv", index=False)

    # ── Save HRF predictions (per-fold + pooled) ───────────────────────────────
    hrf_rows = []
    for fi, fold_probs in enumerate(ext_hrf_probs_per_fold):
        for img_id, true, prob in zip(df_hrf["image_id"].values, y_hrf, fold_probs):
            hrf_rows.append({
                "image_id": img_id, "outer_fold": fi,
                "y_true": int(true), "y_prob": float(prob),
                "y_pred": int(prob >= THRESHOLD),
            })
    # Add pooled row
    for img_id, true, avg_prob, avg_pred in zip(
            df_hrf["image_id"].values, y_hrf, hrf_avg_prob, hrf_avg_pred):
        hrf_rows.append({
            "image_id": img_id, "outer_fold": "pooled",
            "y_true": int(true), "y_prob": float(avg_prob),
            "y_pred": int(avg_pred),
        })
    pd.DataFrame(hrf_rows).to_csv(pred_dir / "hrf_predictions.csv", index=False)

    # ── Save RIM-ONE predictions (per-fold + pooled, with hospital column) ─────
    rimone_rows = []
    for fi, fold_probs in enumerate(ext_rimone_probs_per_fold):
        for img_id, true, hosp, prob in zip(
                df_rimone["image_id"].values, y_rimone,
                hosp_rimone, fold_probs):
            rimone_rows.append({
                "image_id": img_id, "outer_fold": fi,
                "hospital": hosp, "y_true": int(true),
                "y_prob": float(prob), "y_pred": int(prob >= THRESHOLD),
            })
    for img_id, true, hosp, avg_prob, avg_pred in zip(
            df_rimone["image_id"].values, y_rimone,
            hosp_rimone, rimone_avg_prob, rimone_avg_pred):
        rimone_rows.append({
            "image_id": img_id, "outer_fold": "pooled",
            "hospital": hosp, "y_true": int(true),
            "y_prob": float(avg_prob), "y_pred": int(avg_pred),
        })
    pd.DataFrame(rimone_rows).to_csv(
        pred_dir / "rimone_predictions.csv", index=False)

    # ── Save best-params log ───────────────────────────────────────────────────
    params_log = {f"fold_{i}": p for i, p in enumerate(best_params_per_fold)}
    with open(pred_dir / "best_hyperparams.json", "w") as f:
        json.dump(params_log, f, indent=2)

    return {
        "classifier":    clf_name,
        "feature_set":   feature_set_name,
        "fold_df":       fold_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION & COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "roc_auc", "sensitivity", "specificity",
    "f1", "precision", "accuracy", "balanced_accuracy",
]

def aggregate_and_save(all_results: list, results_root: Path):
    """Build mean±std summary across folds and a final comparison table."""
    summary_rows = []

    for res in all_results:
        if res is None:
            continue
        clf  = res["classifier"]
        fset = res["feature_set"]
        df   = res["fold_df"]

        # Mean+/-std for per-fold splits
        for split in ["drishti_val", "hrf_per_fold", "rimone_per_fold",
                      "rimone_r1_per_fold", "rimone_r2_per_fold",
                      "rimone_r3_per_fold"]:
            sub = df[df["split"] == split]
            if sub.empty:
                continue
            row = {"classifier": clf, "feature_set": fset, "split": split}
            for m in METRIC_COLS:
                if m in sub.columns:
                    row[f"{m}_mean"] = sub[m].mean()
                    row[f"{m}_std"]  = sub[m].std()
            summary_rows.append(row)

        # Pooled (single value — no std)
        for split in ["hrf_pooled", "rimone_pooled",
                      "rimone_r1_pooled", "rimone_r2_pooled",
                      "rimone_r3_pooled"]:
            sub = df[df["split"] == split]
            if sub.empty:
                continue
            row = {"classifier": clf, "feature_set": fset, "split": split}
            for m in METRIC_COLS:
                if m in sub.columns:
                    row[f"{m}_mean"] = sub[m].values[0]
                    row[f"{m}_std"]  = float("nan")
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "summaries" / "aggregated_metrics.csv",
                      index=False)
    return summary_df


def build_comparison_table(summary_df: pd.DataFrame, results_root: Path):
    """
    Build the 9-row × multi-column comparison table requested by the user.
    """
    experiments = [
        ("SVM",          "GLCM"),
        ("RandomForest", "GLCM"),
        ("XGBoost",      "GLCM"),
        ("SVM",          "GLRLM"),
        ("RandomForest", "GLRLM"),
        ("XGBoost",      "GLRLM"),
        ("SVM",          "GLCM+GLRLM"),
        ("RandomForest", "GLCM+GLRLM"),
        ("XGBoost",      "GLCM+GLRLM"),
    ]

    rows = []
    for clf, fset in experiments:
        sub = summary_df[(summary_df["classifier"] == clf) &
                         (summary_df["feature_set"] == fset)]

        def get(split, metric, stat="mean"):
            s = sub[sub["split"] == split]
            col = f"{metric}_{stat}"
            if s.empty or col not in s.columns:
                return float("nan")
            return s[col].values[0]

        def fmt_mean_sd(split, metric):
            m = get(split, metric, "mean")
            s = get(split, metric, "std")
            if np.isnan(m):
                return "N/A"
            if np.isnan(s):
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        def fmt_single(split, metric):
            m = get(split, metric, "mean")
            return f"{m:.4f}" if not np.isnan(m) else "N/A"

        row = {
            "Experiment": f"{fset} + {clf}",
            # DRISHTI 5-fold CV
            "DRISHTI AUC":       fmt_mean_sd("drishti_val", "roc_auc"),
            "DRISHTI Sens":      fmt_mean_sd("drishti_val", "sensitivity"),
            "DRISHTI Spec":      fmt_mean_sd("drishti_val", "specificity"),
            "DRISHTI F1":        fmt_mean_sd("drishti_val", "f1"),
            "DRISHTI Acc":       fmt_mean_sd("drishti_val", "accuracy"),
            # HRF pooled
            "HRF AUC":           fmt_single("hrf_pooled", "roc_auc"),
            "HRF Sens":          fmt_single("hrf_pooled", "sensitivity"),
            "HRF Spec":          fmt_single("hrf_pooled", "specificity"),
            "HRF F1":            fmt_single("hrf_pooled", "f1"),
            "HRF Acc":           fmt_single("hrf_pooled", "accuracy"),
            # RIM-ONE pooled
            "RIMONE AUC":        fmt_single("rimone_pooled", "roc_auc"),
            "RIMONE Sens":       fmt_single("rimone_pooled", "sensitivity"),
            "RIMONE Spec":       fmt_single("rimone_pooled", "specificity"),
            "RIMONE F1":         fmt_single("rimone_pooled", "f1"),
            "RIMONE Acc":        fmt_single("rimone_pooled", "accuracy"),
            # RIM-ONE per-site AUC
            "RIMONE r1 AUC":     fmt_single("rimone_r1_pooled", "roc_auc"),
            "RIMONE r2 AUC":     fmt_single("rimone_r2_pooled", "roc_auc"),
            "RIMONE r3 AUC":     fmt_single("rimone_r3_pooled", "roc_auc"),
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_path = results_root / "summaries" / "comparison_table.csv"
    comp_df.to_csv(comp_path, index=False)
    return comp_df


def write_final_report(comp_df: pd.DataFrame,
                       summary_df: pd.DataFrame,
                       results_root: Path,
                       start_time: str, end_time: str):
    """Write a complete plain-text report."""
    lines = [
        "=" * 72,
        "GLAUCOMA DETECTION — CLASSICAL ML EXPERIMENT REPORT",
        f"Started : {start_time}",
        f"Finished: {end_time}",
        "=" * 72,
        "",
        "INNER CV  : StratifiedGroupKFold(n_splits=3, shuffle=True, seed=42)",
        "           groups = patient_id  [PATIENT-LEVEL, LEAKAGE-FREE]",
        "OUTER CV  : Pre-assigned 5-fold column (StratifiedGroupKFold, seed=42)",
        "THRESHOLD : 0.5 (fixed, never tuned on external data)",
        "SCORING   : roc_auc",
        "",
        "Feature sets : GLCM (12), GLRLM (11), GLCM+GLRLM (23)",
        "Classifiers  : SVM (RBF, balanced), RandomForest (balanced), XGBoost",
        "Dev set      : DRISHTI 101 images, 68 patients, 5-fold CV",
        "External     : HRF (30), RIM-ONE (485 = r1:98 + r2:250 + r3:137)",
        "",
        "─" * 72,
        "COMPARISON TABLE",
        "─" * 72,
    ]

    # Print comparison table as fixed-width text
    col_order = [
        "Experiment",
        "DRISHTI AUC", "DRISHTI Sens", "DRISHTI Spec", "DRISHTI F1",
        "HRF AUC", "HRF Sens", "HRF Spec", "HRF F1",
        "RIMONE AUC", "RIMONE Sens", "RIMONE Spec", "RIMONE F1",
        "RIMONE r1 AUC", "RIMONE r2 AUC", "RIMONE r3 AUC",
    ]
    existing_cols = [c for c in col_order if c in comp_df.columns]
    lines.append(comp_df[existing_cols].to_string(index=False))

    lines += [
        "",
        "─" * 72,
        "DETAILED DRISHTI 5-FOLD RESULTS (mean +/- std)",
        "─" * 72,
    ]
    drishti_sub = summary_df[summary_df["split"] == "drishti_val"]
    for _, row in drishti_sub.iterrows():
        lines.append(
            f"  {row['classifier']:<14} | {row['feature_set']:<12} | "
            f"AUC={row.get('roc_auc_mean', float('nan')):.4f}"
            f"+/-{row.get('roc_auc_std', float('nan')):.4f} | "
            f"Sens={row.get('sensitivity_mean', float('nan')):.4f} | "
            f"Spec={row.get('specificity_mean', float('nan')):.4f} | "
            f"F1={row.get('f1_mean', float('nan')):.4f} | "
            f"Acc={row.get('accuracy_mean', float('nan')):.4f}"
        )

    lines += ["", "─" * 72, "RIM-ONE PER-SITE (pooled)", "─" * 72]
    for site in ["rimone_r1_pooled", "rimone_r2_pooled", "rimone_r3_pooled"]:
        site_sub = summary_df[summary_df["split"] == site]
        label = site.replace("_pooled", "").replace("rimone_", "RIM-ONE ")
        for _, row in site_sub.iterrows():
            lines.append(
                f"  {label} | {row['classifier']:<14} | {row['feature_set']:<12} | "
                f"AUC={row.get('roc_auc_mean', float('nan')):.4f}"
            )

    report_path = results_root / "summaries" / "final_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Config + data check only; no training.")
    args = parser.parse_args()

    print("=" * 65)
    print("GLAUCOMA DETECTION — CLASSICAL ML (STAGE 7)")
    print("=" * 65)
    if args.dry_run:
        print("*** DRY-RUN MODE ***\n")

    # Validate input files
    missing = [str(p) for p in FEATURE_FILES.values() if not p.exists()]
    if missing:
        print("ERROR: Missing feature files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("  Inner CV : StratifiedGroupKFold(n_splits=3, groups=patient_id)")
    print("  Outer CV : pre-assigned 5-fold column (patient-level)")
    print("  Threshold: 0.5 (fixed)")
    print()

    if not XGBOOST_AVAILABLE:
        print("  WARNING: XGBoost not available.")

    # Create output directories
    for sub in ["fold_metrics", "predictions", "confusion_matrices",
                "roc_curves", "summaries"]:
        (RESULTS_ROOT / sub).mkdir(parents=True, exist_ok=True)

    classifiers  = get_classifiers()
    feature_sets = list(FEATURE_FILES.keys())

    total = len(classifiers) * len(feature_sets)
    print(f"  Classifiers  : {[c[0] for c in classifiers]}")
    print(f"  Feature sets : {feature_sets}")
    print(f"  Experiments  : {total} ({total} x {N_OUTER_FOLDS} folds each)\n")

    if args.dry_run:
        for fset in feature_sets:
            df_d, df_h, df_r, fc = load_feature_data(fset)
            print(f"  [DRY-RUN] {fset}: "
                  f"DRISHTI={len(df_d)}, HRF={len(df_h)}, "
                  f"RIM-ONE={len(df_r)}, features={len(fc)}")
        print("\n  [DRY-RUN] OK — no training performed.")
        return

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = []

    for fset in feature_sets:
        for clf_name, pipeline, param_grid in classifiers:
            result = run_single_experiment(
                feature_set_name=fset,
                clf_name=clf_name,
                pipeline=pipeline,
                param_grid=param_grid,
                results_root=RESULTS_ROOT,
                dry_run=False,
            )
            all_results.append(result)

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n  Building summary tables...")
    summary_df = aggregate_and_save(all_results, RESULTS_ROOT)
    comp_df    = build_comparison_table(summary_df, RESULTS_ROOT)
    report_path = write_final_report(comp_df, summary_df, RESULTS_ROOT,
                                     start_time, end_time)

    # Save experiment config
    config = {
        "run_timestamp": start_time,
        "random_seed": RANDOM_SEED,
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "inner_cv_method": "StratifiedGroupKFold(groups=patient_id)",
        "outer_cv_method": "pre-assigned patient-level folds",
        "scoring": SCORING,
        "threshold": THRESHOLD,
        "classifiers": [c[0] for c in get_classifiers()],
        "feature_sets": feature_sets,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(RESULTS_ROOT / "summaries" / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Report: {report_path}")
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Results: {RESULTS_ROOT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
