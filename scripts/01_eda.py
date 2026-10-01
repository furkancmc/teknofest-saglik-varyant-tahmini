"""Bölüm 2.1-2.2: veri yapısı, sınıf dağılımı, eksik/aykırı değer analizi ve Şekil 1.

Çıktılar:
    outputs/tables/tablo2_ozellik_gruplari.csv
    outputs/tables/sinif_dagilimi.csv
    outputs/tables/eksiklik_sutun_bazinda.csv
    outputs/tables/eda_ozet.json
    outputs/figures/sekil1_sinif_dagilimi_eksiklik.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ID_COL, TABLE_DIR, TARGET
from src.features import column_groups
from src.pipeline import prepare_dataset
from src.plots import plot_class_and_missingness

GROUP_DESC = {
    "AL_": ("Popülasyon alel frekansı (gnomAD, All of Us)", "Benign varyant sinyali"),
    "EK_": ("İn silico patojenite skorları", "Fonksiyonel risk sinyali"),
    "CAT_": ("Popülasyon/genotip/kalite bilgisi", "Veri kalitesi ve grup bilgisi"),
    "AA_": ("Referans / alternatif amino asit", "Biyokimyasal değişim sinyali"),
}


def main():
    ds = prepare_dataset()
    df, panels = ds.df, ds.panels

    # --- Tablo 2: özellik grupları -------------------------------------------------
    groups = column_groups(panels["MASTER"])
    n_cols = panels["MASTER"].shape[1]
    print(f"Her veri kümesinde sütun sayısı: {n_cols}")
    for name, p in panels.items():
        assert list(p.columns) == list(panels["MASTER"].columns), f"{name} sütunları farklı"
    tablo2 = pd.DataFrame(
        [(g, len(c), *GROUP_DESC[g]) for g, c in groups.items()],
        columns=["Grup", "Sütun", "İçerik", "Model Açısından Rolü"],
    )
    tablo2.to_csv(TABLE_DIR / "tablo2_ozellik_gruplari.csv", index=False)
    print(tablo2.to_string(index=False))

    # --- Sınıf dağılımı ----------------------------------------------------------------
    rows = {}
    for name, p in panels.items():
        vc = p[TARGET].value_counts()
        rows[name] = {"Benign": int(vc.get(0, 0)), "Patojenik": int(vc.get(1, 0))}
    vc = df[TARGET].value_counts()
    rows["GENEL (tekil)"] = {"Benign": int(vc.get(0, 0)), "Patojenik": int(vc.get(1, 0))}
    class_counts = pd.DataFrame(rows).T
    class_counts["n"] = class_counts.sum(axis=1)
    class_counts["Patojenik oranı"] = (class_counts["Patojenik"] / class_counts["n"]).round(3)
    class_counts.to_csv(TABLE_DIR / "sinif_dagilimi.csv")
    print(class_counts)

    total_rows = sum(len(p) for p in panels.values())
    print(f"Panel satır toplamı: {total_rows} -> Variant_ID tekilleştirme sonrası: {df[ID_COL].nunique()}")
    for name in ("KANSER", "CFTR", "PAH"):
        ov = panels[name][ID_COL].isin(panels["MASTER"][ID_COL]).sum()
        print(f"  {name}: {len(panels[name])} satırın {ov} tanesi MASTER ile örtüşüyor")

    # --- Eksik değer analizi -------------------------------------------------------
    raw_cols = [c for g in groups.values() for c in g]
    miss = df[raw_cols].isna().mean()
    miss_df = pd.DataFrame({"sutun": miss.index, "eksiklik_orani": miss.values})
    miss_df["grup"] = miss_df["sutun"].str.extract(r"^([A-Z]+_)")[0]
    miss_df.to_csv(TABLE_DIR / "eksiklik_sutun_bazinda.csv", index=False)
    group_missing = miss_df.groupby("grup")["eksiklik_orani"].mean()
    print("Grup bazında ortalama eksiklik:\n", (group_missing * 100).round(1))

    # --- Aykırı değer kontrolü -----------------------------------------------------
    b = ds.builder
    al_vals = df[b.al_cols_].apply(pd.to_numeric, errors="coerce")
    ek_vals = df[b.ek_cols_].apply(pd.to_numeric, errors="coerce")
    al_skew = al_vals.skew().median()
    print(f"Sabit (tek değerli) sütun sayısı: {len(b.dropped_constant_)} -> modelden çıkarıldı")
    print(f"AL_ [0,1] aralığı dışında değer içeren sütun: {len(b.al_out_of_range_)}")
    print(f"AL_ sütunlarının medyan çarpıklığı: {al_skew:.2f} (sağa çarpık -> log dönüşümü)")
    print("EK_ skor özetleri (uç değerler kırpılmadı):\n", ek_vals.describe().T.round(3))

    summary = {
        "sutun_sayisi": int(n_cols),
        "grup_sutun_sayilari": {g: len(c) for g, c in groups.items()},
        "panel_satir_toplami": int(total_rows),
        "tekil_varyant": int(len(df)),
        "grup_ortalama_eksiklik": group_missing.round(4).to_dict(),
        "sabit_sutunlar": b.dropped_constant_,
        "al_aralik_disi_sutunlar": b.al_out_of_range_,
        "al_medyan_carpiklik": round(float(al_skew), 3),
        "model_ozellik_sayisi": int(ds.X.shape[1]),
    }
    (TABLE_DIR / "eda_ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Şekil 1 -------------------------------------------------------------------------
    al_missing = miss_df.loc[miss_df["grup"] == "AL_"].set_index("sutun")["eksiklik_orani"]
    plot_class_and_missingness(
        class_counts.loc[list(panels), ["Benign", "Patojenik"]], al_missing, float(group_missing.get("EK_", 0))
    )


if __name__ == "__main__":
    main()
