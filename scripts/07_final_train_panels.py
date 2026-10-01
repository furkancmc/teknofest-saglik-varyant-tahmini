"""Bölüm 3.3: final modelin tüm eğitim verisiyle eğitilmesi ve alt grup (panel) skorları.

- Ön işleyici, SHAP özellik setleri (top-50 / top-59) ve 5 modelli ensemble tüm
  tekil eğitim verisiyle (n=3224) eğitilip diske kaydedilir.
- Tam eğitimli model MASTER, KANSER, CFTR ve PAH panel dosyalarına uygulanır;
  her panel için F1'i maksimize eden alt gruba özgü eşik seçilir (Tablo 7).
  Bu skorlar bağımsız test / OOF değildir; paneller kısmen örtüştüğü için
  alt grup n toplamı genel n ile toplanmamalıdır.

Çıktı:
    outputs/models/final_model.joblib, panel_thresholds.json
    outputs/tables/tablo7_alt_grup.csv, panel_tahminleri.csv
    outputs/figures/sekil3_alt_grup_metrikleri.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.config import ID_COL, MODEL_DIR, SELECTION_DIR, TABLE_DIR, TARGET
from src.metrics import best_threshold, compute_metrics, format_table
from src.models import SoftVotingEnsemble, load_best_params
from src.pipeline import prepare_dataset
from src.plots import plot_subgroup_metrics
from src.preprocessing import FoldPreprocessor
from src.selection import select_features


def main():
    ds = prepare_dataset()

    pre = FoldPreprocessor(ds.builder.cat_cols_).fit(ds.X)
    X_pre = pre.transform(ds.X)

    gpath = SELECTION_DIR / "global_feature_sets.json"
    if gpath.exists():
        feature_sets = json.loads(gpath.read_text(encoding="utf-8"))
    else:
        feature_sets, _ = select_features(X_pre, ds.y)
        gpath.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")

    ensemble = SoftVotingEnsemble(
        feature_sets, load_best_params("lgbm"), load_best_params("xgb")
    ).fit(X_pre, ds.y)

    global_thr = json.loads((MODEL_DIR / "global_threshold.json").read_text(encoding="utf-8"))["global_threshold"]

    # --- Panel bazlı skorlar -------------------------------------------------------
    proba = ensemble.predict_proba(X_pre)[:, 1]
    preds = pd.DataFrame({ID_COL: ds.df[ID_COL], TARGET: ds.y, "proba": proba})

    rows, panel_thresholds, panel_preds = [], {}, []
    for panel, pdf in ds.panels.items():
        sub = preds[preds[ID_COL].isin(set(pdf[ID_COL]))]
        thr = best_threshold(sub[TARGET], sub["proba"])
        panel_thresholds[panel] = thr
        m = compute_metrics(sub[TARGET], sub["proba"], thr)
        rows.append({"Alt Grup": panel, **m})
        panel_preds.append(sub.assign(panel=panel))
        print(f"{panel:<7} n={m['n']:<5} eşik={thr:.2f} F1={m['F1']:.4f} MCC={m['MCC']:.4f} "
              f"AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")

    tablo7 = pd.DataFrame(rows).set_index("Alt Grup")
    tablo7_out = format_table(
        tablo7[["n", "threshold", "F1", "MCC", "AUROC", "AUPRC", "TP", "TN", "FP", "FN"]]
        .rename(columns={"threshold": "Eşik"})
    )
    tablo7_out.to_csv(TABLE_DIR / "tablo7_alt_grup.csv")
    pd.concat(panel_preds).to_csv(TABLE_DIR / "panel_tahminleri.csv", index=False)
    plot_subgroup_metrics(tablo7)

    # --- Kayıt -------------------------------------------------------------------------
    (MODEL_DIR / "panel_thresholds.json").write_text(json.dumps(panel_thresholds, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "feature_builder": ds.builder,
            "preprocessor": pre,
            "ensemble": ensemble,
            "feature_sets": feature_sets,
            "global_threshold": global_thr,
            "panel_thresholds": panel_thresholds,
        },
        MODEL_DIR / "final_model.joblib",
    )
    print(f"Final model kaydedildi: {MODEL_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
