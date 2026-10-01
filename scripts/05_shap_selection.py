"""Bölüm 2.8, 3.5 ve Tablo 5 satır 5-7: SHAP analizi ve SHAP tabanlı özellik seçimi.

A) Global SHAP (tüm eğitim verisi, 305 özellik):
   - ilk 15 özellik (Şekil 5) ve beeswarm özet grafiği
   - sıfır SHAP değerli özellik sayısı
   - grup katkıları (AL_, EK_, AA_ ve türevleri, Mühendislik, CAT_)
   - ilk 50 özelliğin toplam SHAP katkısındaki payı
B) Fold içi SHAP seçimi (her fold yalnız kendi eğitim kısmından top-50 / top-59)
   ve bu setlerle 5-fold CV deneyleri:
     5) LightGBM, SHAP top-50, num_leaves=40, lr=0,05
     6) CatBoost, SHAP top-50, class_weight=balanced
     7) CatBoost, SHAP top-50, ağırlıksız, depth=6

Çıktı: outputs/tables/shap_onem.csv, shap_grup_katkilari.csv, tablo5_shap_secim.csv,
       outputs/selection/global_feature_sets.json, fold_feature_sets.json,
       outputs/figures/sekil5_shap_top15.png, shap_beeswarm.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    FIG_DIR, MANUAL_LGBM_PARAMS, OOF_DIR, SELECTION_DIR, TABLE_DIR, TOP_K, TOP_K_ALT,
)
from src.features import feature_group
from src.metrics import compute_metrics, format_table
from src.models import make_catboost, make_lgbm
from src.pipeline import get_fold_feature_sets, get_splits, prepare_dataset, run_oof, save_oof
from src.plots import plot_shap_top
from src.preprocessing import FoldPreprocessor
from src.selection import shap_values_binary


def global_shap_analysis(ds):
    X_pre = FoldPreprocessor(ds.builder.cat_cols_).fit_transform(ds.X)
    # select_features ile aynı seçici model; SHAP matrisi beeswarm için de kullanılır
    selector = make_lgbm().fit(X_pre, ds.y)
    sv = shap_values_binary(selector, X_pre)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X_pre.columns).sort_values(ascending=False)
    sets = {f"top{k}": imp.index[:k].tolist() for k in (TOP_K, TOP_K_ALT)}
    sets["all"] = X_pre.columns.tolist()

    imp.rename("mean_abs_shap").to_frame().to_csv(TABLE_DIR / "shap_onem.csv", index_label="ozellik")
    (SELECTION_DIR / "global_feature_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")

    total = imp.sum()
    n_zero = int((imp == 0).sum())
    top50_share = imp.head(TOP_K).sum() / total
    groups = (imp.groupby(imp.index.map(feature_group)).sum() / total).sort_values(ascending=False)

    print(f"İlk 5 özellik: {imp.index[:5].tolist()}")
    print(f"Toplam {len(imp)} özelliğin {n_zero} tanesi sıfır SHAP değeri taşıyor")
    print(f"İlk {TOP_K} özelliğin toplam SHAP katkısındaki payı: %{top50_share * 100:.1f}")
    print("Grup katkıları (%):\n", (groups * 100).round(1).to_string())
    groups.rename("katki_orani").to_frame().to_csv(TABLE_DIR / "shap_grup_katkilari.csv", index_label="grup")

    plot_shap_top(imp, top_n=15)

    # beeswarm özet grafiği (ilk 20 özellik)
    shap.summary_plot(sv, X_pre, max_display=20, show=False)
    plt.savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.close("all")


def fold_fit_predict(fold_sets, make_model, fs_key=f"top{TOP_K}"):
    def fit_predict(k, X_tr, y_tr, X_va):
        cols = fold_sets[k][fs_key]
        model = make_model()
        model.fit(X_tr[cols], y_tr)
        return model.predict_proba(X_va[cols])[:, 1]
    return fit_predict


def main():
    ds = prepare_dataset()

    print("\n=== A) Global SHAP analizi ===")
    global_shap_analysis(ds)

    print("\n=== B) Fold içi SHAP seçimi ve CV deneyleri ===")
    splits = get_splits(ds.y)
    fold_sets = get_fold_feature_sets(ds, splits, recompute=True)

    experiments = [
        (5, "LightGBM", f"SHAP top-{TOP_K}; num_leaves=40; lr=0,05", lambda: make_lgbm(MANUAL_LGBM_PARAMS)),
        (6, "CatBoost", f"SHAP top-{TOP_K}, class_weight=balanced", lambda: make_catboost(balanced=True)),
        (7, "CatBoost", f"SHAP top-{TOP_K}, ağırlıksız, depth=6", lambda: make_catboost(balanced=False)),
    ]
    rows, oofs = [], {}
    for no, model_name, desc, factory in experiments:
        oof = run_oof(ds, splits, fold_fit_predict(fold_sets, factory), desc)
        oofs[f"exp_{no}"] = oof
        m = compute_metrics(ds.y, oof, 0.5)
        rows.append({"No": no, "Model": model_name, "Özellik": TOP_K, "Strateji": desc,
                     "CV F1": m["F1"], "CV MCC": m["MCC"], "CV AUPRC": m["AUPRC"]})
        print(f"{no}) {model_name} {desc}: F1={m['F1']:.4f} MCC={m['MCC']:.4f} AUPRC={m['AUPRC']:.4f}")

    table = format_table(pd.DataFrame(rows))
    table.to_csv(TABLE_DIR / "tablo5_shap_secim.csv", index=False)
    save_oof(ds, oofs, OOF_DIR / "oof_shap_secim.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
