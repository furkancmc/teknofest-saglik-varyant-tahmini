"""Bölüm 2.4 / Tablo 5 satır 1-4: sınıf dengesizliği stratejileri (LightGBM, 305 özellik).

    1) Baseline (ağırlıksız)
    2) class_weight="balanced"
    3) SMOTE 1:1   (azınlık = çoğunluk)
    4) SMOTE 2:1   (çoğunluk : azınlık = 2 : 1)

SMOTE yalnızca eğitim fold'una, medyan imputation'dan SONRA uygulanır
(SMOTE eksik değerle çalışamaz); doğrulama fold'u orijinal kalır.

Çıktı: outputs/tables/tablo5_dengesizlik.csv, outputs/oof/oof_dengesizlik.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import OOF_DIR, SEED, TABLE_DIR
from src.metrics import compute_metrics, format_table
from src.models import make_lgbm
from src.pipeline import get_splits, prepare_dataset, run_oof, save_oof

EXPERIMENTS = {
    "Baseline (ağırlıksız)": dict(class_weight=None, smote=None),
    "class_weight=balanced": dict(class_weight="balanced", smote=None),
    "SMOTE 1:1": dict(class_weight=None, smote=1.0),
    "SMOTE 2:1": dict(class_weight=None, smote=0.5),
}


def lgbm_fit_predict(class_weight=None, smote=None):
    def fit_predict(k, X_tr, y_tr, X_va):
        if smote is not None:
            X_tr, y_tr = SMOTE(sampling_strategy=smote, random_state=SEED, k_neighbors=5).fit_resample(X_tr, y_tr)
        model = make_lgbm(class_weight=class_weight)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1]
    return fit_predict


def main():
    ds = prepare_dataset()
    splits = get_splits(ds.y)

    rows, oofs = [], {}
    for i, (name, cfg) in enumerate(EXPERIMENTS.items(), start=1):
        oof = run_oof(ds, splits, lgbm_fit_predict(**cfg), name)
        oofs[f"lgbm_{i}"] = oof
        m = compute_metrics(ds.y, oof, 0.5)
        rows.append({"No": i, "Model": "LightGBM", "Özellik": ds.X.shape[1], "Strateji": name,
                     "CV F1": m["F1"], "CV MCC": m["MCC"], "CV AUPRC": m["AUPRC"]})
        print(f"{name}: F1={m['F1']:.4f} MCC={m['MCC']:.4f} AUPRC={m['AUPRC']:.4f}")

    table = format_table(pd.DataFrame(rows))
    table.to_csv(TABLE_DIR / "tablo5_dengesizlik.csv", index=False)
    save_oof(ds, oofs, OOF_DIR / "oof_dengesizlik.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
