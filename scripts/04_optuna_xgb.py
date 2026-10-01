"""Bölüm 2.6 / Tablo 5 satır 8: XGBoost için Optuna TPE (40 trial, 3-fold CV, 305 özellik).

Arama uzayı:
    max_depth 3-8, n_estimators 200-600, learning_rate 0,01-0,15,
    subsample 0,6-1,0, colsample_bytree 0,5-1,0, reg_alpha / reg_lambda 1e-4-5,
    min_child_weight 1-10

Not: Bu adımda yalnızca en iyi parametreler kaydedilir; XGBoost tek başına 5-fold
CV ile raporlanmaz, final ensemble'da bileşen olarak kullanılır.

Çıktı: outputs/optuna/xgb_best_params.json, outputs/optuna/xgb_trials.csv,
       outputs/figures/optuna_xgb_gecmis.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history

from src.config import FIG_DIR, XGB_N_TRIALS
from src.models import make_xgb
from src.pipeline import prepare_dataset
from src.tuning import run_study


def suggest(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def main():
    ds = prepare_dataset()
    study = run_study(ds, "xgb", suggest, make_xgb, n_trials=XGB_N_TRIALS)
    plot_optimization_history(study)
    plt.savefig(FIG_DIR / "optuna_xgb_gecmis.png", bbox_inches="tight", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    main()
