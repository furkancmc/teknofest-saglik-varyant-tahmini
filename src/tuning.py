"""Optuna TPE hiperparametre optimizasyonu (3-fold stratified CV, amaç: F1)."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import optuna
from sklearn.metrics import f1_score

from .config import OPTUNA_CV_SPLITS, OPTUNA_DIR, SEED
from .pipeline import Dataset, get_splits, preprocess_fold


def cv_f1(ds: Dataset, make_model: Callable, splits, features: list[str] | None = None) -> float:
    """Fold içinde ön işleme + model eğitimi; fold F1 ortalaması."""
    scores = []
    for tr, va in splits:
        _, X_tr, X_va = preprocess_fold(ds, tr, va)
        if features is not None:
            X_tr, X_va = X_tr[features], X_va[features]
        model = make_model()
        model.fit(X_tr, ds.y[tr])
        pred = (model.predict_proba(X_va)[:, 1] >= 0.5).astype(int)
        scores.append(f1_score(ds.y[va], pred))
    return float(np.mean(scores))


def run_study(ds: Dataset, name: str, suggest: Callable, build: Callable, n_trials: int,
              features: list[str] | None = None) -> optuna.Study:
    """suggest(trial) -> params ; build(params) -> model"""
    splits = get_splits(ds.y, n_splits=OPTUNA_CV_SPLITS, seed=SEED)

    def objective(trial):
        params = suggest(trial)
        return cv_f1(ds, lambda: build(params), splits, features)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    (OPTUNA_DIR / f"{name}_best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{name}_trials.csv", index=False)
    print(f"[{name}] en iyi 3-fold F1 = {study.best_value:.4f}")
    print(f"[{name}] en iyi parametreler: {study.best_params}")
    return study
