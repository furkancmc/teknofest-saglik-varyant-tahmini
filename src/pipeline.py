"""Ortak veri hazırlama ve fold-içi (sızıntısız) çapraz doğrulama yardımcıları."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ID_COL, N_SPLITS, SEED, TARGET
from .data import build_training_frame, load_panels
from .features import FeatureBuilder
from .preprocessing import FoldPreprocessor
from .selection import load_fold_feature_sets, save_fold_feature_sets, select_features


@dataclass
class Dataset:
    panels: dict[str, pd.DataFrame]
    df: pd.DataFrame          # tekilleştirilmiş ham tablo (3224 x ...)
    X: pd.DataFrame           # 305 özellikli ham (imputation öncesi) matris
    y: np.ndarray
    builder: FeatureBuilder


def prepare_dataset(verbose: bool = True) -> Dataset:
    panels = load_panels()
    df = build_training_frame(panels)
    builder = FeatureBuilder().fit(df)
    X = builder.transform(df)
    y = df[TARGET].astype(int).to_numpy()
    if verbose:
        print(
            f"Tekil varyant: {len(df)} | patojenik: {y.sum()} | benign: {(y == 0).sum()} | "
            f"özellik: {X.shape[1]} (sabit sütun çıkarıldı: {len(builder.dropped_constant_)})"
        )
    return Dataset(panels, df, X, y, builder)


def get_splits(y, n_splits: int = N_SPLITS, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def preprocess_fold(ds: Dataset, tr_idx, va_idx):
    pre = FoldPreprocessor(ds.builder.cat_cols_).fit(ds.X.iloc[tr_idx])
    return pre, pre.transform(ds.X.iloc[tr_idx]), pre.transform(ds.X.iloc[va_idx])


def get_fold_feature_sets(ds: Dataset, splits, recompute: bool = False) -> dict[int, dict]:
    """Her fold için yalnızca eğitim kısmından SHAP top-50 / top-59 setleri (diskte önbelleklenir)."""
    cached = None if recompute else load_fold_feature_sets()
    if cached is not None and len(cached) == len(splits):
        return cached
    fold_sets = {}
    for k, (tr, va) in enumerate(splits):
        _, X_tr, _ = preprocess_fold(ds, tr, va)
        fold_sets[k], _ = select_features(X_tr, ds.y[tr])
        print(f"  fold {k + 1}: SHAP özellik seçimi tamamlandı")
    save_fold_feature_sets(fold_sets)
    return fold_sets


FitPredict = Callable[[int, pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def run_oof(ds: Dataset, splits, fit_predict: FitPredict, name: str = "") -> np.ndarray:
    """Her fold'da ön işlemeyi eğitimden öğrenip modeli eğitir, OOF olasılıklarını döndürür.

    fit_predict(fold_no, X_train_pre, y_train, X_valid_pre) -> doğrulama olasılıkları
    """
    oof = np.zeros(len(ds.y), dtype=float)
    for k, (tr, va) in enumerate(splits):
        _, X_tr, X_va = preprocess_fold(ds, tr, va)
        oof[va] = fit_predict(k, X_tr, ds.y[tr], X_va)
        print(f"  [{name}] fold {k + 1}/{len(splits)} tamam")
    return oof


def save_oof(ds: Dataset, oof: dict[str, np.ndarray], path) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: ds.df[ID_COL], TARGET: ds.y, **oof})
    out.to_csv(path, index=False)
    return out
