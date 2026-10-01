"""Model fabrikaları ve 5 modelli soft-voting ensemble."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import (
    BASE_LGBM_PARAMS, CATBOOST_PARAMS, MANUAL_LGBM_PARAMS, OPTUNA_DIR, SEED, TOP_K, TOP_K_ALT,
)


def make_lgbm(params: dict | None = None, class_weight=None) -> LGBMClassifier:
    p = dict(BASE_LGBM_PARAMS if params is None else params)
    p.setdefault("subsample_freq", 1)
    return LGBMClassifier(
        **p, class_weight=class_weight, random_state=SEED, n_jobs=-1, verbose=-1,
    )


def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        **params, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )


def make_catboost(balanced: bool = False) -> CatBoostClassifier:
    return CatBoostClassifier(
        **CATBOOST_PARAMS,
        auto_class_weights="Balanced" if balanced else None,
        random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=-1,
    )


# ---------------------------------------------------------------------------
# Optuna sonuçları
# ---------------------------------------------------------------------------
def load_best_params(model: str) -> dict:
    path = OPTUNA_DIR / f"{model}_best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok. Önce scripts/03_optuna_lgbm.py ve scripts/04_optuna_xgb.py çalıştırılmalı."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ensemble: 2x CatBoost + 2x LightGBM + 1x XGBoost (soft voting)
# ---------------------------------------------------------------------------
# (üye adı, özellik seti anahtarı)
ENSEMBLE_MEMBERS = [
    ("catboost_unweighted", f"top{TOP_K}"),
    ("catboost_balanced", f"top{TOP_K}"),
    ("lgbm_manual", f"top{TOP_K}"),
    ("lgbm_optuna", f"top{TOP_K_ALT}"),
    ("xgb_optuna", f"top{TOP_K_ALT}"),
]


def make_member(name: str, lgbm_params: dict, xgb_params: dict):
    if name == "catboost_unweighted":
        return make_catboost(balanced=False)
    if name == "catboost_balanced":
        return make_catboost(balanced=True)
    if name == "lgbm_manual":
        return make_lgbm(MANUAL_LGBM_PARAMS)
    if name == "lgbm_optuna":
        return make_lgbm(lgbm_params)
    if name == "xgb_optuna":
        return make_xgb(xgb_params)
    raise ValueError(name)


class SoftVotingEnsemble:
    """Her üye kendi SHAP özellik setiyle eğitilir; olasılıkların eşit ağırlıklı ortalaması alınır."""

    def __init__(self, feature_sets: dict[str, list[str]], lgbm_params: dict, xgb_params: dict):
        self.feature_sets = feature_sets
        self.lgbm_params = lgbm_params
        self.xgb_params = xgb_params

    def fit(self, X: pd.DataFrame, y) -> "SoftVotingEnsemble":
        y = np.asarray(y).astype(int)
        self.models_ = {}
        for name, fs_key in ENSEMBLE_MEMBERS:
            model = make_member(name, self.lgbm_params, self.xgb_params)
            model.fit(X[self.feature_sets[fs_key]], y)
            self.models_[name] = model
        return self

    def predict_members(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                name: self.models_[name].predict_proba(X[self.feature_sets[fs_key]])[:, 1]
                for name, fs_key in ENSEMBLE_MEMBERS
            },
            index=X.index,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_members(X).mean(axis=1).to_numpy()
        return np.column_stack([1 - p, p])
