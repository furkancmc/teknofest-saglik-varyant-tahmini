"""Tüm analiz hattını rapordaki sırayla çalıştırır."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_eda.py",                   # Bölüm 2.1-2.2, Tablo 2, Şekil 1
    "02_imbalance_experiments.py",  # Bölüm 2.4, Tablo 5 (1-4)
    "03_optuna_lgbm.py",           # Bölüm 2.6
    "04_optuna_xgb.py",            # Bölüm 2.6, Tablo 5 (8)
    "05_shap_selection.py",        # Bölüm 2.8, 3.5, Tablo 5 (5-7), Şekil 5
    "06_ensemble_oof.py",          # Bölüm 3.1-3.2, 3.4, Tablo 5 (9), 6, 8, Şekil 2, 4
    "07_final_train_panels.py",    # Bölüm 3.3, Tablo 7, Şekil 3
    "08_pah_strategy.py",          # Bölüm 3.6, Tablo 9
    "09_error_analysis.py",        # Bölüm 4.2-4.3, local SHAP
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for step in STEPS:
        if only and not any(step.startswith(s) for s in only):
            continue
        print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], check=True, cwd=ROOT)
        print(f"<<< {step} tamamlandı ({time.time() - t0:.0f} sn)", flush=True)
