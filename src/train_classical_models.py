"""
train_classical_models.py
=========================
Classical ML experiment — Stage 2: Training & Evaluation.

Experimental design
-------------------
Development set : DRISHTI (101 images, patient-level 5-fold CV)
External tests  : HRF (30 images), RIM-ONE (485 images)

Feature sets
------------
  A  GLCM only          (12 features)
  B  GLRLM only         (11 features)
  C  GLCM + GLRLM       (23 features)

Classifiers
-----------
  1  SVM   (RBF kernel)
  2  Random Forest
  3  XGBoost

Leakage prevention
------------------
  - StandardScaler is fitted ONLY on the DRISHTI training folds of each
    CV iteration. It is then applied (transform only) to:
      • the held-out DRISHTI validation fold
      • the HRF external test set
      • the RIM-ONE external test set
  - No test-set information is used in fitting, tuning, or scaling.
  - HRF and RIM-ONE are NEVER used in training, scaling, or
    hyperparameter selection.
  - sklearn Pipeline enforces the fit-on-train-only contract.

Hyperparameter grids
--------------------
Small, reproducible predefined grids — this is a methodological
comparison, not an unrestricted optimisation search.

SVM:
  C       : [0.1, 1.0, 10.0]
  gamma   : ['scale', 'auto']
  kernel  : ['rbf']

Random Forest:
  n_estimators : [100, 200]
  max_depth    : [None, 10, 20]
  min_samples_leaf : [1, 3]

XGBoost:
  n_estimators    : [100, 200]
  max_depth       : [3, 5]
  learning_rate   : [0.05, 0.1]
  subsample       : [0.8, 1.0]

Tuning method: GridSearchCV on training folds (inner CV = 3-fold
stratified), scoring = roc_auc.

Outputs (in results/classical_ml/)
------------------------------------
  fold_metrics/        : per-fold metric CSV for every (model, feature_set)
  predictions/         : per-fold OOF predictions + external test predictions
  confusion_matrices/  : PNG confusion matrices for validation & external
  summaries/           : aggregated mean±std CSV, JSON summary, final report

Usage
-----
  # From project root:
  python src/train_classical_models.py [--dry-run]

  --dry-run  : Validates config and data loading only. Does NOT train.

NOTE: This script was written and reviewed BEFORE execution. Do not run
until the user explicitly authorises training (Stage 2).
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

# XGBoost — optional but expected
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoost classifier will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_FOLDS     = 5
INNER_CV    = 3       # inner GridSearchCV folds
SCORING     = "roc_auc"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILES = {
    "GLCM":         PROJECT_ROOT / "features" / "glcm_features.csv",
    "GLRLM":        PROJECT_ROOT / "features" / "glrlm_features.csv",
    "GLCM+GLRLM":   PROJECT_ROOT / "features" / "glcm_glrlm_features.csv",
}

SPLITS_FILE   = PROJECT_ROOT / "splits"   / "drishti_5fold_splits.csv"
RESULTS_ROOT  = PROJECT_ROOT / "results"  / "classical_ml"

# Metadata columns that are NOT feature values
META_COLS = {
    "image_id", "dataset", "patient_id", "label", "class_name",
    "source_path", "processed_path", "split", "fold",
    "hospital", "source_partition", "discordant_patient",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_classifiers():
    """Return (name, estimator, param_grid) tuples.
    param_grid keys use the Pipeline step prefix 'clf__'.
    """
    classifiers = []

    # 1. SVM
    classifiers.append((
        "SVM",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_SEED)),
        ]),
        {
            "clf__C":     [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
    ))

    # 2. Random Forest
    classifiers.append((
        "RandomForest",
        Pipeline([
            ("scaler", StandardScaler()),   # RF doesn't need scaling,
            ("clf", RandomForestClassifier( # but kept for consistency
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        {
            "clf__n_estimators":    [100, 200],
            "clf__max_depth":       [None, 10, 20],
            "clf__min_samples_leaf":[1, 3],
        },
    ))

    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        classifiers.append((
            "XGBoost",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
            ]),
            {
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample":     [0.8, 1.0],
            },
        ))

    return classifiers


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_data(feature_set_name: str):
    """Load feature CSV and return (df_drishti, df_hrf, df_rimone, feat_cols).

    DRISHTI: development pool, will be used for 5-fold CV.
    HRF, RIM-ONE: external test sets, never used in training or scaling.
    """
    csv_path = FEATURE_FILES[feature_set_name]
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Identify feature columns (all non-metadata numeric columns)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # Split by dataset
    df_drishti = df[df["dataset"] == "DRISHTI"].copy().reset_index(drop=True)
    df_hrf     = df[(df["dataset"] == "HRF") & (df["label"].isin([0, 1]))].copy().reset_index(drop=True)
    df_rimone  = df[df["dataset"] == "RIM-ONE"].copy().reset_index(drop=True)

    return df_drishti, df_hrf, df_rimone, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Compute a standard set of binary classification metrics."""
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
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Glaucoma"],
                yticklabels=["Normal", "Glaucoma"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PER-FOLD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(feature_set_name: str, clf_name: str,
                           pipeline: Pipeline, param_grid: dict,
                           results_root: Path, dry_run: bool = False):
    """
    Run full 5-fold CV for one (feature_set, classifier) combination.

    Leakage-safe contract:
      - StandardScaler fitted on TRAIN portion of each fold only.
      - GridSearchCV uses the training portion only (inner 3-fold CV).
      - Validation and external test sets are only ever transformed.
    """
    print(f"\n  [EXPERIMENT] {clf_name} | {feature_set_name}")

    df_drishti, df_hrf, df_rimone, feat_cols = load_feature_data(feature_set_name)
    X_hrf    = df_hrf[feat_cols].values.astype(float)
    y_hrf    = df_hrf["label"].values.astype(int)
    X_rimone = df_rimone[feat_cols].values.astype(float)
    y_rimone = df_rimone["label"].values.astype(int)

    exp_tag   = f"{clf_name}__{feature_set_name.replace('+', '_plus_')}"
    cm_dir    = results_root / "confusion_matrices" / exp_tag
    pred_dir  = results_root / "predictions"        / exp_tag
    fold_dir  = results_root / "fold_metrics"
    cm_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"    [DRY-RUN] Would train {clf_name} on {feature_set_name}. Skipping.")
        return None

    inner_cv = StratifiedKFold(n_splits=INNER_CV, shuffle=True,
                               random_state=RANDOM_SEED)

    fold_metrics    = []      # one dict per fold
    oof_predictions = []      # out-of-fold predictions on DRISHTI
    ext_hrf_preds   = []      # external predictions per fold (averaged later)
    ext_rimone_preds = []

    for fold in range(N_FOLDS):
        # ── Split DRISHTI by fold assignment ──────────────────────────────────
        train_mask = df_drishti["fold"] != fold
        val_mask   = df_drishti["fold"] == fold

        X_train = df_drishti.loc[train_mask, feat_cols].values.astype(float)
        y_train = df_drishti.loc[train_mask, "label"].values.astype(int)
        X_val   = df_drishti.loc[val_mask,   feat_cols].values.astype(float)
        y_val   = df_drishti.loc[val_mask,   "label"].values.astype(int)

        ids_val = df_drishti.loc[val_mask, "image_id"].values

        print(f"    Fold {fold}: train={len(X_train)}, val={len(X_val)}")

        # ── Hyperparameter search (fit ONLY on training folds) ────────────────
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=SCORING,
            refit=True,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        best_estimator = gs.best_estimator_
        best_params    = gs.best_params_
        best_cv_score  = gs.best_score_

        print(f"      Best params: {best_params} | CV {SCORING}: {best_cv_score:.4f}")

        # ── Evaluate on HELD-OUT DRISHTI validation fold ─────────────────────
        y_val_pred = best_estimator.predict(X_val)
        y_val_prob = best_estimator.predict_proba(X_val)[:, 1]

        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
        val_metrics.update({
            "fold": fold,
            "classifier": clf_name,
            "feature_set": feature_set_name,
            "split": "drishti_val",
            "best_params": json.dumps(best_params),
            "best_cv_auc": best_cv_score,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })
        fold_metrics.append(val_metrics)

        # Save confusion matrix
        save_confusion_matrix(
            y_val, y_val_pred,
            title=f"{clf_name} | {feature_set_name} | Fold {fold} Val",
            filepath=cm_dir / f"fold{fold}_val.png",
        )

        # ── OOF predictions ────────────────────────────────────────────────────
        for img_id, true, pred, prob in zip(ids_val, y_val, y_val_pred, y_val_prob):
            oof_predictions.append({
                "image_id": img_id,
                "fold": fold,
                "y_true": true,
                "y_pred": pred,
                "y_prob": prob,
            })

        # ── External test — HRF (transform only, never fit) ───────────────────
        y_hrf_pred = best_estimator.predict(X_hrf)
        y_hrf_prob = best_estimator.predict_proba(X_hrf)[:, 1]
        ext_hrf_preds.append(y_hrf_prob)

        hrf_metrics = compute_metrics(y_hrf, y_hrf_pred, y_hrf_prob)
        hrf_metrics.update({
            "fold": fold,
            "classifier": clf_name,
            "feature_set": feature_set_name,
            "split": "hrf_external",
            "best_params": json.dumps(best_params),
            "best_cv_auc": best_cv_score,
            "n_train": len(X_train),
            "n_val": len(X_hrf),
        })
        fold_metrics.append(hrf_metrics)

        # ── External test — RIM-ONE (transform only, never fit) ───────────────
        y_rimone_pred = best_estimator.predict(X_rimone)
        y_rimone_prob = best_estimator.predict_proba(X_rimone)[:, 1]
        ext_rimone_preds.append(y_rimone_prob)

        rimone_metrics = compute_metrics(y_rimone, y_rimone_pred, y_rimone_prob)
        rimone_metrics.update({
            "fold": fold,
            "classifier": clf_name,
            "feature_set": feature_set_name,
            "split": "rimone_external",
            "best_params": json.dumps(best_params),
            "best_cv_auc": best_cv_score,
            "n_train": len(X_train),
            "n_val": len(X_rimone),
        })
        fold_metrics.append(rimone_metrics)

    # ── Ensemble / averaged external predictions (mean probability) ────────────
    hrf_avg_prob    = np.mean(ext_hrf_preds,    axis=0)
    rimone_avg_prob = np.mean(ext_rimone_preds, axis=0)
    hrf_avg_pred    = (hrf_avg_prob >= 0.5).astype(int)
    rimone_avg_pred = (rimone_avg_prob >= 0.5).astype(int)

    hrf_ensemble_metrics    = compute_metrics(y_hrf,    hrf_avg_pred,    hrf_avg_prob)
    rimone_ensemble_metrics = compute_metrics(y_rimone, rimone_avg_pred, rimone_avg_prob)

    save_confusion_matrix(
        y_hrf, hrf_avg_pred,
        title=f"{clf_name} | {feature_set_name} | HRF (ensemble avg)",
        filepath=cm_dir / "hrf_ensemble.png",
    )
    save_confusion_matrix(
        y_rimone, rimone_avg_pred,
        title=f"{clf_name} | {feature_set_name} | RIM-ONE (ensemble avg)",
        filepath=cm_dir / "rimone_ensemble.png",
    )

    # ── Save per-fold metrics CSV ──────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(fold_dir / f"{exp_tag}.csv", index=False)

    # ── Save OOF predictions ───────────────────────────────────────────────────
    oof_df = pd.DataFrame(oof_predictions)
    oof_df.to_csv(pred_dir / "oof_predictions.csv", index=False)

    # Save HRF external predictions (all folds)
    hrf_pred_rows = []
    for fold_i, fold_probs in enumerate(ext_hrf_preds):
        for img_id, true, prob in zip(df_hrf["image_id"].values, y_hrf, fold_probs):
            hrf_pred_rows.append({
                "image_id": img_id,
                "fold": fold_i,
                "y_true": true,
                "y_prob": prob,
                "y_pred": int(prob >= 0.5),
            })
    pd.DataFrame(hrf_pred_rows).to_csv(pred_dir / "hrf_predictions.csv", index=False)

    # Save RIM-ONE external predictions
    rimone_pred_rows = []
    for fold_i, fold_probs in enumerate(ext_rimone_preds):
        for img_id, true, prob in zip(df_rimone["image_id"].values, y_rimone, fold_probs):
            rimone_pred_rows.append({
                "image_id": img_id,
                "fold": fold_i,
                "y_true": true,
                "y_prob": prob,
                "y_pred": int(prob >= 0.5),
            })
    pd.DataFrame(rimone_pred_rows).to_csv(pred_dir / "rimone_predictions.csv", index=False)

    return {
        "classifier":    clf_name,
        "feature_set":   feature_set_name,
        "fold_metrics":  fold_df,
        "hrf_ensemble":  hrf_ensemble_metrics,
        "rimone_ensemble": rimone_ensemble_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "roc_auc",
    "f1", "precision", "recall", "sensitivity", "specificity",
]

def aggregate_results(all_results: list, results_root: Path):
    """Aggregate fold-level metrics into mean ± std summary."""
    summary_rows = []

    for res in all_results:
        if res is None:
            continue
        clf  = res["classifier"]
        fset = res["feature_set"]
        df   = res["fold_metrics"]

        for split in ["drishti_val", "hrf_external", "rimone_external"]:
            sub = df[df["split"] == split]
            if sub.empty:
                continue
            row = {"classifier": clf, "feature_set": fset, "split": split}
            for m in METRIC_COLS:
                if m in sub.columns:
                    row[f"{m}_mean"] = sub[m].mean()
                    row[f"{m}_std"]  = sub[m].std()
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_root / "summaries" / "aggregated_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Aggregated summary saved to: {summary_path}")

    # JSON summary
    json_summary = {}
    for _, row in summary_df.iterrows():
        key = f"{row['classifier']}|{row['feature_set']}|{row['split']}"
        json_summary[key] = {
            m: {"mean": row.get(f"{m}_mean"), "std": row.get(f"{m}_std")}
            for m in METRIC_COLS
            if f"{m}_mean" in row
        }

    with open(results_root / "summaries" / "results_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    return summary_df


def write_final_report(summary_df: pd.DataFrame, results_root: Path,
                       start_time: str, end_time: str):
    """Write a plain-text final report."""
    lines = [
        "=" * 70,
        "GLAUCOMA DETECTION — CLASSICAL ML EXPERIMENT REPORT",
        f"Started : {start_time}",
        f"Finished: {end_time}",
        "=" * 70,
        "",
        "Feature sets : GLCM (12), GLRLM (11), GLCM+GLRLM (23)",
        "Classifiers  : SVM, RandomForest, XGBoost",
        "Dev set      : DRISHTI (101 images, 5-fold CV, patient-level grouping)",
        "External test: HRF (30 images), RIM-ONE (485 images)",
        "",
    ]

    for split_label in ["drishti_val", "hrf_external", "rimone_external"]:
        lines.append(f"{'─'*70}")
        lines.append(f"Split: {split_label}")
        lines.append(f"{'─'*70}")
        sub = summary_df[summary_df["split"] == split_label]
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['classifier']:<16} | {row['feature_set']:<14} | "
                f"AUC={row.get('roc_auc_mean', float('nan')):.4f}±{row.get('roc_auc_std', float('nan')):.4f} | "
                f"F1={row.get('f1_mean', float('nan')):.4f}±{row.get('f1_std', float('nan')):.4f} | "
                f"Sens={row.get('sensitivity_mean', float('nan')):.4f} | "
                f"Spec={row.get('specificity_mean', float('nan')):.4f}"
            )
        lines.append("")

    report_path = results_root / "summaries" / "final_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Final report saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classical ML training pipeline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data loading and config only. Do not train.")
    args = parser.parse_args()

    print("=" * 65)
    print("GLAUCOMA DETECTION — CLASSICAL ML TRAINING PIPELINE")
    print("=" * 65)

    if args.dry_run:
        print("*** DRY-RUN MODE — No models will be trained ***\n")

    # ── Validate required files exist ──────────────────────────────────────────
    missing = []
    for name, path in FEATURE_FILES.items():
        if not path.exists():
            missing.append(str(path))
    if not SPLITS_FILE.exists():
        missing.append(str(SPLITS_FILE))
    if missing:
        print("ERROR: Required files not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("  All required input files found.")

    # ── Check XGBoost availability ─────────────────────────────────────────────
    if not XGBOOST_AVAILABLE:
        print("  WARNING: XGBoost not available. Install with: pip install xgboost")

    # ── Create output directories ──────────────────────────────────────────────
    for sub in ["fold_metrics", "predictions", "confusion_matrices", "summaries"]:
        (RESULTS_ROOT / sub).mkdir(parents=True, exist_ok=True)

    classifiers  = get_classifiers()
    feature_sets = list(FEATURE_FILES.keys())  # ["GLCM", "GLRLM", "GLCM+GLRLM"]

    total_experiments = len(classifiers) * len(feature_sets)
    print(f"\n  Classifiers  : {[c[0] for c in classifiers]}")
    print(f"  Feature sets : {feature_sets}")
    print(f"  Total experiments : {total_experiments}  ({total_experiments} × {N_FOLDS} folds)")
    print(f"  Inner CV folds    : {INNER_CV}")
    print(f"  Scoring           : {SCORING}")
    print(f"  Random seed       : {RANDOM_SEED}\n")

    if args.dry_run:
        # Verify data can be loaded for each feature set
        for fset in feature_sets:
            df_d, df_h, df_r, feat_cols = load_feature_data(fset)
            print(f"  [DRY-RUN] {fset}: "
                  f"DRISHTI={len(df_d)}, HRF={len(df_h)}, "
                  f"RIM-ONE={len(df_r)}, features={len(feat_cols)}")
        print("\n  [DRY-RUN] Config and data loading OK. No training performed.")
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

    # ── Aggregate and report ───────────────────────────────────────────────────
    summary_df = aggregate_results(all_results, RESULTS_ROOT)
    write_final_report(summary_df, RESULTS_ROOT, start_time, end_time)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Results saved to: {RESULTS_ROOT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
