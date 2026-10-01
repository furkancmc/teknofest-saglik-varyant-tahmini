"""Değerlendirme metrikleri ve karar eşiği taraması."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from .config import THRESHOLD_GRID


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "AUROC": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "AUPRC": average_precision_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def threshold_scan(y_true, proba, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """0,05-0,95 aralığında her eşik için metrikler."""
    return pd.DataFrame([compute_metrics(y_true, proba, t) for t in grid])


def best_threshold(y_true, proba, grid=THRESHOLD_GRID, metric: str = "F1") -> float:
    """Verilen metriği (varsayılan F1) maksimize eden eşik."""
    scan = threshold_scan(y_true, proba, grid)
    return float(scan.loc[scan[metric].idxmax(), "threshold"])


def format_table(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Raporlama için ondalıkları yuvarlar."""
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].round(digits)
    return out
