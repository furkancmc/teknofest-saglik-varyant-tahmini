"""SHAP tabanlı özellik önemi ve özellik seçimi (305 -> 50 / 59)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap

from .config import SELECTION_DIR, TOP_K, TOP_K_ALT
from .models import make_lgbm


def shap_values_binary(model, X: pd.DataFrame) -> np.ndarray:
    """Ağaç modeli için pozitif sınıfa (patojenik) ait SHAP matrisini döndürür.

    shap sürümüne göre çıktı liste, 2B veya 3B dizi olabilir; hepsi (n, p)'ye indirgenir.
    """
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Ortalama mutlak SHAP değeri (global önem), büyükten küçüğe sıralı."""
    sv = shap_values_binary(model, X)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)


def select_features(X_pre: pd.DataFrame, y, ks=(TOP_K, TOP_K_ALT)) -> tuple[dict, pd.Series]:
    """Ön işlenmiş eğitim verisinde temel LightGBM + SHAP ile top-k özellik setleri.

    Yalnızca kendisine verilen (eğitim) verisini kullanır; CV içinde her fold için
    ayrı çağrılarak seçim adımında veri sızıntısı önlenir.
    """
    selector = make_lgbm()
    selector.fit(X_pre, np.asarray(y).astype(int))
    imp = shap_importance(selector, X_pre)
    sets = {f"top{k}": imp.index[:k].tolist() for k in ks}
    sets["all"] = X_pre.columns.tolist()
    return sets, imp


def fold_feature_sets_path():
    return SELECTION_DIR / "fold_feature_sets.json"


def save_fold_feature_sets(fold_sets: dict[int, dict]) -> None:
    fold_feature_sets_path().write_text(
        json.dumps({str(k): v for k, v in fold_sets.items()}, indent=2), encoding="utf-8"
    )


def load_fold_feature_sets() -> dict[int, dict] | None:
    path = fold_feature_sets_path()
    if not path.exists():
        return None
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
