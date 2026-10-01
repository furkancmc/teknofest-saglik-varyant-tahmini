"""Bölüm 2.7, 2.9, 3.1, 3.2, 3.4: 5 modelli soft-voting ensemble'ın 5-fold OOF değerlendirmesi.

Ensemble = 2x CatBoost (ağırlıksız + balanced, top-50)
         + 2x LightGBM (manuel top-50 + Optuna top-59)
         + 1x XGBoost  (Optuna, top-59)

Her fold'da: medyan imputation ve SHAP özellik seçimi yalnız eğitim kısmından öğrenilir.
Karar eşiği tüm fold'ların birleşik OOF olasılıkları üzerinde 0,05-0,95 aralığında
F1'i maksimize edecek şekilde seçilir (test seti kullanılmaz).

Çıktı:
    outputs/oof/oof_ensemble.csv
    outputs/tables/tablo5_model_konfigurasyonlari.csv  (Tablo 5, tüm satırlar)
    outputs/tables/tablo6_final_oof.csv
    outputs/tables/tablo8_esik_etkisi.csv, esik_taramasi.csv
    outputs/models/global_threshold.json
    outputs/figures/sekil2_karmasiklik_pr.png, sekil4_esik_etkisi.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import MODEL_DIR, OOF_DIR, REPORT_THRESHOLDS, TABLE_DIR, TOP_K, TOP_K_ALT
from src.metrics import compute_metrics, format_table, threshold_scan
from src.models import ENSEMBLE_MEMBERS, SoftVotingEnsemble, load_best_params
from src.pipeline import get_fold_feature_sets, get_splits, prepare_dataset, run_oof, save_oof
from src.plots import plot_confusion_and_pr, plot_threshold_effect


def build_table5(ens_metrics: dict, n_features: int) -> pd.DataFrame:
    """02 ve 05 scriptlerinin çıktılarını XGBoost ve ensemble satırlarıyla birleştirir."""
    parts = []
    for fname in ("tablo5_dengesizlik.csv", "tablo5_shap_secim.csv"):
        path = TABLE_DIR / fname
        if path.exists():
            parts.append(pd.read_csv(path))
        else:
            print(f"  uyarı: {fname} bulunamadı (ilgili script çalıştırılmamış)")
    parts.append(pd.DataFrame([
        {"No": 8, "Model": "XGBoost", "Özellik": n_features,
         "Strateji": "Optuna TPE 40 trial; tekil CV metrik kaydı yok",
         "CV F1": np.nan, "CV MCC": np.nan, "CV AUPRC": np.nan},
        {"No": 9, "Model": "5-Model Ensemble", "Özellik": f"{TOP_K}/{TOP_K_ALT}",
         "Strateji": "Soft voting (2×CatBoost, 2×LightGBM, 1×XGBoost)",
         "CV F1": ens_metrics["F1"], "CV MCC": ens_metrics["MCC"], "CV AUPRC": ens_metrics["AUPRC"]},
    ]))
    return format_table(pd.concat(parts, ignore_index=True).sort_values("No"))


def main():
    ds = prepare_dataset()
    splits = get_splits(ds.y)
    fold_sets = get_fold_feature_sets(ds, splits)
    lgbm_params, xgb_params = load_best_params("lgbm"), load_best_params("xgb")

    member_oof = {name: np.zeros(len(ds.y)) for name, _ in ENSEMBLE_MEMBERS}

    def fit_predict(k, X_tr, y_tr, X_va):
        ens = SoftVotingEnsemble(fold_sets[k], lgbm_params, xgb_params).fit(X_tr, y_tr)
        members = ens.predict_members(X_va)
        va_idx = splits[k][1]
        for name in member_oof:
            member_oof[name][va_idx] = members[name].to_numpy()
        return members.mean(axis=1).to_numpy()

    oof = run_oof(ds, splits, fit_predict, "ensemble")
    save_oof(ds, {"ensemble": oof, **member_oof}, OOF_DIR / "oof_ensemble.csv")

    # --- Karar eşiği taraması --------------------------------------------------------
    scan = threshold_scan(ds.y, oof)
    scan.to_csv(TABLE_DIR / "esik_taramasi.csv", index=False)
    best_thr = float(scan.loc[scan["F1"].idxmax(), "threshold"])
    (MODEL_DIR / "global_threshold.json").write_text(
        json.dumps({"global_threshold": best_thr}, indent=2), encoding="utf-8"
    )
    print(f"\nFinal global eşik (OOF, max F1): {best_thr:.2f}")

    # --- Tablo 6: final OOF başarımı ---------------------------------------------
    m = compute_metrics(ds.y, oof, best_thr)
    tablo6 = pd.DataFrame({
        "Metrik": ["F1", "MCC", "AUROC", "AUPRC", "Precision", "Recall (Duyarlılık)", "Specificity (Özgüllük)"],
        "Değer": [m["F1"], m["MCC"], m["AUROC"], m["AUPRC"], m["Precision"], m["Recall"], m["Specificity"]],
    })
    tablo6 = format_table(tablo6)
    tablo6.to_csv(TABLE_DIR / "tablo6_final_oof.csv", index=False)
    print(f"n={m['n']}, TP={m['TP']}, FN={m['FN']}, FP={m['FP']}, TN={m['TN']}")
    print(tablo6.to_string(index=False))

    # bileşen modellerin OOF skorları (bilgi amaçlı, konsola)
    for name, p in member_oof.items():
        mm = compute_metrics(ds.y, p, 0.5)
        print(f"  üye {name:<20} F1={mm['F1']:.4f} MCC={mm['MCC']:.4f} AUPRC={mm['AUPRC']:.4f}")

    # --- Tablo 8: farklı eşiklerin etkisi ------------------------------------------
    thr_list = sorted(set(REPORT_THRESHOLDS) | {best_thr})
    tablo8_raw = pd.DataFrame([compute_metrics(ds.y, oof, t) for t in thr_list])
    tablo8 = format_table(tablo8_raw[["threshold", "F1", "MCC", "Precision", "Recall", "FP", "FN"]]
                          .rename(columns={"threshold": "Eşik"}))
    tablo8.to_csv(TABLE_DIR / "tablo8_esik_etkisi.csv", index=False)
    print(tablo8.to_string(index=False))

    # --- Tablo 5 (tüm satırlar) ------------------------------------------------------
    tablo5 = build_table5(m, ds.X.shape[1])
    tablo5.to_csv(TABLE_DIR / "tablo5_model_konfigurasyonlari.csv", index=False)
    print(tablo5.to_string(index=False))

    # --- Şekiller ----------------------------------------------------------------------
    plot_confusion_and_pr(ds.y, oof, best_thr, m)
    plot_threshold_effect(tablo8_raw, best_thr)


if __name__ == "__main__":
    main()
