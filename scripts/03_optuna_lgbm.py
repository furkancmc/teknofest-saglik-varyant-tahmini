"""Bölüm 2.6: LightGBM için Optuna TPE (60 trial, 3-fold CV).

Arama uzayı:
    n_estimators 200-800, num_leaves 15-63, learning_rate 0,01-0,15,
    min_child_samples 5-40, subsample 0,6-1,0, colsample_bytree 0,5-1,0,
    reg_alpha / reg_lambda 1e-4-10

Çıktı: outputs/optuna/lgbm_best_params.json, outputs/optuna/lgbm_trials.csv,
       outputs/figures/optuna_lgbm_gecmis.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history

from src.config import FIG_DIR, LGBM_N_TRIALS
from src.models import make_lgbm
from src.pipeline import prepare_dataset
from src.tuning import run_study


def suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def main():
    ds = prepare_dataset()
    study = run_study(ds, "lgbm", suggest, make_lgbm, n_trials=LGBM_N_TRIALS)
    plot_optimization_history(study)
    plt.savefig(FIG_DIR / "optuna_lgbm_gecmis.png", bbox_inches="tight", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    main()
