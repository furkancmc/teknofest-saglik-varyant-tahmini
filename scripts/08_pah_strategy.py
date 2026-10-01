"""Bölüm 3.6 / Tablo 9: PAH alt grubu özel strateji karşılaştırması.

PAH eğitim alt kümesi: 62 benign, 310 patojenik.

    1) Genel model, referans eşik (0,42)          -> tam eğitimli model panel skoru
    2) Genel model, PAH özel eşik                  -> tam eğitimli model panel skoru
    3) Eğitimde bootstrap (62->200) + PAH özel eşik -> PAH üzerinde 5-fold CV

Strateji 3'te PAH varyantları 5 katlı stratified olarak bölünür. Her fold'da
eğitim kümesi = PAH dışı tüm tekil varyantlar + PAH eğitim kısmı. PAH eğitim
kısmındaki benign örnekler yerine koymalı örnekleme (bootstrap) ile 62->200
oranında artırılır. Doğrulama fold'u (PAH) orijinal bırakılır; ön işleme,
SHAP seçimi ve ensemble eğitimi yalnız eğitim kısmından öğrenilir.

Çıktı: outputs/tables/tablo9_pah_strateji.csv, outputs/oof/oof_pah_bootstrap.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ID_COL, N_SPLITS, OOF_DIR, PAH_BOOTSTRAP_TARGET, PAH_REFERENCE_THRESHOLD, SEED, TABLE_DIR, TARGET,
)
from src.metrics import best_threshold, compute_metrics, format_table
from src.models import SoftVotingEnsemble, load_best_params
from src.pipeline import prepare_dataset
from src.preprocessing import FoldPreprocessor
from src.selection import select_features


def pah_bootstrap_cv(ds, lgbm_params, xgb_params) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pah_mask = ds.df[ID_COL].isin(set(ds.panels["PAH"][ID_COL])).to_numpy()
    pah_idx = np.where(pah_mask)[0]
    other_idx = np.where(~pah_mask)[0]
    y_pah = ds.y[pah_idx]
    n_benign_total = int((y_pah == 0).sum())
    print(f"PAH: {len(pah_idx)} varyant ({n_benign_total} benign, {int(y_pah.sum())} patojenik)")

    oof = np.zeros(len(pah_idx))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (tr_p, va_p) in enumerate(skf.split(pah_idx, y_pah)):
        tr_idx = np.concatenate([other_idx, pah_idx[tr_p]])
        va_idx = pah_idx[va_p]

        # yalnız eğitim fold'unda PAH benign bootstrap (62 -> 200 oranında)
        benign_tr = pah_idx[tr_p][y_pah[tr_p] == 0]
        target = int(round(PAH_BOOTSTRAP_TARGET * len(benign_tr) / n_benign_total))
        extra = rng.choice(benign_tr, size=max(target - len(benign_tr), 0), replace=True)
        tr_boot = np.concatenate([tr_idx, extra])

        pre = FoldPreprocessor(ds.builder.cat_cols_).fit(ds.X.iloc[tr_idx])
        X_tr = pre.transform(ds.X.iloc[tr_boot]).reset_index(drop=True)
        y_tr = ds.y[tr_boot]
        X_va = pre.transform(ds.X.iloc[va_idx])

        sets, _ = select_features(X_tr, y_tr)
        ens = SoftVotingEnsemble(sets, lgbm_params, xgb_params).fit(X_tr, y_tr)
        oof[va_p] = ens.predict_proba(X_va)[:, 1]
        print(f"  fold {k + 1}: PAH benign eğitim {len(benign_tr)} -> {len(benign_tr) + len(extra)}")

    return pd.DataFrame({ID_COL: ds.df[ID_COL].iloc[pah_idx].to_numpy(), TARGET: y_pah, "proba": oof})


def main():
    ds = prepare_dataset()

    # --- 1-2: tam eğitimli genel modelin PAH panel skorları ------------------------
    preds_path = TABLE_DIR / "panel_tahminleri.csv"
    if not preds_path.exists():
        raise FileNotFoundError("Önce scripts/07_final_train_panels.py çalıştırılmalı.")
    pah = pd.read_csv(preds_path).query("panel == 'PAH'")
    pah_thr = best_threshold(pah[TARGET], pah["proba"])

    rows = [
        ("Genel model, referans eşik", compute_metrics(pah[TARGET], pah["proba"], PAH_REFERENCE_THRESHOLD)),
        ("Genel model, PAH özel eşik", compute_metrics(pah[TARGET], pah["proba"], pah_thr)),
    ]

    # --- 3: bootstrap + PAH özel eşik (PAH 5-fold CV) ---------------------------------
    boot = pah_bootstrap_cv(ds, load_best_params("lgbm"), load_best_params("xgb"))
    boot.to_csv(OOF_DIR / "oof_pah_bootstrap.csv", index=False)
    boot_thr = best_threshold(boot[TARGET], boot["proba"])
    rows.append(("Eğitimde bootstrap (62→200) + PAH özel eşik",
                 compute_metrics(boot[TARGET], boot["proba"], boot_thr)))

    tablo9 = pd.DataFrame([{"Strateji": name, **m} for name, m in rows])
    tablo9 = format_table(
        tablo9[["Strateji", "threshold", "F1", "MCC", "AUROC", "AUPRC", "FP", "FN"]]
        .rename(columns={"threshold": "Eşik"})
    )
    tablo9.to_csv(TABLE_DIR / "tablo9_pah_strateji.csv", index=False)
    print(tablo9.to_string(index=False))


if __name__ == "__main__":
    main()
