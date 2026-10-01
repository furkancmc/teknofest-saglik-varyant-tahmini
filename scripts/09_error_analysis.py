"""Bölüm 2.9, 4.2, 4.3: hata profili ve hatalı örneklerde local SHAP analizi.

- OOF ensemble tahminleri final global eşikle TP / FP / FN / TN olarak ayrılır.
- Her grup için EK_ skor düzeyi ve AL_ doluluk/frekans profili çıkarılır
  (beklenti: FP -> yüksek EK_ skorlu benign, FN -> AL_ bilgisi eksik patojenik).
- En emin yanlış pozitif ve yanlış negatif örnekler için final CatBoost (ağırlıksız)
  üyesi üzerinden local SHAP waterfall grafikleri ve ilk 5 katkı özelliği üretilir.

Çıktı: outputs/tables/hata_profili.csv, local_shap_hatalar.csv,
       outputs/figures/local_shap_<FP|FN>_<Variant_ID>.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIG_DIR, ID_COL, MODEL_DIR, OOF_DIR, TABLE_DIR, TARGET
from src.models import ENSEMBLE_MEMBERS
from src.pipeline import prepare_dataset

EXPLAIN_MEMBER = "catboost_unweighted"
N_EXAMPLES = 3
PROFILE_COLS = ["ek_mean", "EK_7", "EK_8", "EK_9", "al_filled_count", "al_max_freq", "al_n_pops_common"]


def outcome(y, pred):
    return np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "FP", "FN"], default="TN",
    )


def main():
    ds = prepare_dataset()
    oof = pd.read_csv(OOF_DIR / "oof_ensemble.csv")
    assert (oof[ID_COL].to_numpy() == ds.df[ID_COL].to_numpy()).all()
    thr = json.loads((MODEL_DIR / "global_threshold.json").read_text(encoding="utf-8"))["global_threshold"]

    oof["tahmin"] = (oof["ensemble"] >= thr).astype(int)
    oof["sonuc"] = outcome(oof[TARGET].to_numpy(), oof["tahmin"].to_numpy())

    # --- Hata profili ----------------------------------------------------------------
    prof = ds.X[PROFILE_COLS].astype(float).copy()
    prof["al_eksik_orani"] = ds.X[ds.builder.al_cols_].isna().mean(axis=1)
    prof["sonuc"] = oof["sonuc"].to_numpy()
    profile = prof.groupby("sonuc").agg(["mean", "median"]).round(4)
    profile.insert(0, ("n", ""), prof["sonuc"].value_counts())
    profile.to_csv(TABLE_DIR / "hata_profili.csv")
    print(f"Eşik={thr:.2f} sonuç dağılımı:\n{prof['sonuc'].value_counts()}")
    print(profile.xs("mean", axis=1, level=1).to_string())

    # --- Local SHAP -----------------------------------------------------------------------
    bundle = joblib.load(MODEL_DIR / "final_model.joblib")
    fs_key = dict(ENSEMBLE_MEMBERS)[EXPLAIN_MEMBER]
    cols = bundle["feature_sets"][fs_key]
    model = bundle["ensemble"].models_[EXPLAIN_MEMBER]
    X_pre = bundle["preprocessor"].transform(ds.X)[cols]

    fp = oof[oof["sonuc"] == "FP"].nlargest(N_EXAMPLES, "ensemble")
    fn = oof[oof["sonuc"] == "FN"].nsmallest(N_EXAMPLES, "ensemble")
    errors = oof[oof["sonuc"].isin(["FP", "FN"])]

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X_pre.loc[errors.index]))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = np.ravel(explainer.expected_value)[-1]

    records = []
    for row_pos, idx in enumerate(errors.index):
        order = np.argsort(-np.abs(sv[row_pos]))[:5]
        rec = {ID_COL: errors.at[idx, ID_COL], "sonuc": errors.at[idx, "sonuc"],
               "olasilik": round(float(errors.at[idx, "ensemble"]), 4)}
        for r, j in enumerate(order, start=1):
            rec[f"ozellik_{r}"] = cols[j]
            rec[f"shap_{r}"] = round(float(sv[row_pos, j]), 4)
        records.append(rec)
    pd.DataFrame(records).to_csv(TABLE_DIR / "local_shap_hatalar.csv", index=False)

    pos = {idx: i for i, idx in enumerate(errors.index)}
    for label, sel in (("FP", fp), ("FN", fn)):
        for idx in sel.index:
            exp = shap.Explanation(
                values=sv[pos[idx]], base_values=base,
                data=X_pre.loc[idx].to_numpy(), feature_names=cols,
            )
            shap.plots.waterfall(exp, max_display=12, show=False)
            vid = sel.at[idx, ID_COL]
            plt.title(f"{label} örneği {vid} (olasılık={sel.at[idx, 'ensemble']:.3f})")
            plt.savefig(FIG_DIR / f"local_shap_{label}_{vid}.png", bbox_inches="tight", dpi=200)
            plt.close("all")
    print(f"Local SHAP: {len(errors)} hatalı örnek açıklandı, waterfall grafikleri {FIG_DIR} altında.")


if __name__ == "__main__":
    main()
