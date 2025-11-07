#!/usr/bin/env python3
"""
Compute accuracy, precision, recall, f1 and AUC from existing predictions.csv files
and write a consolidated JSON to results/metrics_summary_from_preds.json.

This script is defensive: it handles constant predictions or single-class true labels
and uses zero_division=0 for precision/recall so we don't get NaNs from sklearn.
If ROC AUC cannot be computed (only one class present), it will set auc to null in JSON.
"""
import json
import math
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODEL_PATHS = {
    "model_1": RESULTS / "model 1" / "predictions.csv",
    "model_2": RESULTS / "model 2" / "predictions.csv",
    "model_3": RESULTS / "model 3" / "predictions.csv",
    "model_4": RESULTS / "model 4" / "predictions.csv",
    "model_6": RESULTS / "model6_handcrafted_ann_pytorch" / "predictions.csv",
}


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def compute_metrics_from_df(df):
    # Expect columns: true_label, predicted_label, malignant_prob (prob of class 1)
    # Try multiple probability column names for robustness
    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["predicted_label"].astype(int).to_numpy()

    # Find probability column for positive (malignant) class
    prob_col_candidates = ["malignant_prob", "prob_1", "positive_prob", "p_malignant"]
    prob_col = None
    for c in prob_col_candidates:
        if c in df.columns:
            prob_col = c
            break
    # fallback: if there are exactly 4 columns and last column looks numeric, pick last
    if prob_col is None:
        for c in df.columns[::-1]:
            try:
                _ = pd.to_numeric(df[c])
                prob_col = c
                break
            except Exception:
                continue

    probs = None
    if prob_col is not None:
        probs = pd.to_numeric(df[prob_col], errors="coerce").to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC: only if probs available and y_true has both classes
    auc = None
    if probs is not None and len(set(y_true)) > 1:
        try:
            auc_val = roc_auc_score(y_true, probs)
            if math.isfinite(auc_val):
                auc = float(auc_val)
        except Exception:
            auc = None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
    }


def main():
    out = {}
    for model_name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"WARNING: predictions file not found for {model_name}: {path}")
            out[model_name] = None
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            out[model_name] = None
            continue

        try:
            metrics = compute_metrics_from_df(df)
            out[model_name] = metrics
            print(f"Computed metrics for {model_name}: {metrics}")
        except Exception as e:
            print(f"ERROR computing metrics for {model_name}: {e}")
            out[model_name] = None

    out_path = RESULTS / "metrics_summary_from_preds.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote consolidated metrics to: {out_path}\n")


if __name__ == "__main__":
    main()
