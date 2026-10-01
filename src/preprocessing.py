"""Fold içi ön işleme: kategorik kodlama + medyan imputation.

Medyanlar ve kategori sözlükleri yalnızca eğitim fold'undan öğrenilir; doğrulama
fold'una aynı değerler uygulanır (veri sızıntısı önlemi). AL_ sütunlarında sıfır
doldurma kullanılmaz: eksiklik "gözlemlenmemiş", sıfır ise "frekans yok" demektir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKEN = "__NA__"


class FoldPreprocessor:
    def __init__(self, cat_cols: list[str]):
        self.cat_cols = list(cat_cols)

    def fit(self, X: pd.DataFrame, y=None) -> "FoldPreprocessor":
        self.columns_ = list(X.columns)
        self.num_cols_ = [c for c in self.columns_ if c not in self.cat_cols]
        self.cat_maps_ = {}
        for c in self.cat_cols:
            values = sorted(X[c].fillna(MISSING_TOKEN).astype(str).unique())
            self.cat_maps_[c] = {v: i for i, v in enumerate(values)}
        medians = X[self.num_cols_].median(numeric_only=True)
        # eğitim fold'unda tamamen boş kalan sütun olursa 0'a düş
        self.medians_ = medians.reindex(self.num_cols_).fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns_].copy()
        for c in self.cat_cols:
            X[c] = (
                X[c].fillna(MISSING_TOKEN).astype(str)
                .map(self.cat_maps_[c]).fillna(-1).astype(int)
            )
        X[self.num_cols_] = X[self.num_cols_].astype(float).fillna(self.medians_)
        return X.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
