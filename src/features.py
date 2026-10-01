"""Özellik grupları, sabit sütun temizliği, aykırı değer kontrolü ve özellik mühendisliği.

Model özellik seti:
    AL_  : 334 AL_ sütunundan sabit/tek değerli sütunlar çıkarıldıktan sonra kalanlar
    EK_  : 9 in silico skor
    CAT_ : 6 kategorik sütun (fold içinde kodlanır)
    AA_  : AA_1 / AA_2 (referans / alternatif amino asit, fold içinde kodlanır)
    AL_ türevi : al_filled_count, al_max_freq, al_log_max, al_n_pops_common
    EK_ türevi : ek_mean, ek7_x_ek9
    AA_ türevi : aa_hydro_abs_delta, aa_mw_abs_delta, aa_polarity_change, aa_stop_gain
    Panel      : is_PAH, is_CFTR, is_KANSER
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import (
    AA_PREFIX, AL_PREFIX, CAT_PREFIX, COMMON_AF, EK_PREFIX, FLAG_PANELS, ID_COL, TARGET,
)

# ---------------------------------------------------------------------------
# Sabit biyokimyasal ölçekler
# ---------------------------------------------------------------------------
# Kyte & Doolittle (1982) hidropati indeksi [14]
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Amino asit molekül ağırlıkları (Da)
AA_MOL_WEIGHT = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16, "Q": 146.15,
    "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23,
    "Y": 181.19, "V": 117.15,
}
# Polarite / yük sınıfı
AA_POLARITY = {
    **{a: "nonpolar" for a in "AVLIMFWPG"},
    **{a: "polar" for a in "STCYNQ"},
    **{a: "positive" for a in "KRH"},
    **{a: "negative" for a in "DE"},
}
STOP = "*"

AL_ENGINEERED = ["al_filled_count", "al_max_freq", "al_log_max", "al_n_pops_common"]
EK_ENGINEERED = ["ek_mean", "ek7_x_ek9"]
AA_ENGINEERED = ["aa_hydro_abs_delta", "aa_mw_abs_delta", "aa_polarity_change", "aa_stop_gain"]
FLAG_COLS = [f"is_{p}" for p in FLAG_PANELS]


# ---------------------------------------------------------------------------
# Sütun grupları
# ---------------------------------------------------------------------------
def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    return {
        "AL_": [c for c in cols if c.startswith(AL_PREFIX)],
        "EK_": [c for c in cols if c.startswith(EK_PREFIX)],
        "CAT_": [c for c in cols if c.startswith(CAT_PREFIX)],
        "AA_": [c for c in cols if c.startswith(AA_PREFIX)],
    }


def find_constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Eksik olmayan değerlerinde en fazla tek benzersiz değer taşıyan sütunlar.

    Bu sütunlar etiketten bağımsız (denetimsiz) olarak belirlendiği için
    veri sızıntısı oluşturmaz.
    """
    return [c for c in cols if df[c].nunique(dropna=True) <= 1]


def check_al_range(df: pd.DataFrame, al_cols: list[str]) -> list[str]:
    """AL_ frekans sütunlarının [0, 1] aralığında kaldığını doğrular."""
    vals = df[al_cols].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in al_cols if ((vals[c] < 0) | (vals[c] > 1)).any()]
    if bad:
        warnings.warn(f"[0,1] dışında değer içeren AL_ sütunları: {bad}")
    return bad


# ---------------------------------------------------------------------------
# Özellik mühendisliği
# ---------------------------------------------------------------------------
def _residue_props(seq) -> tuple[float, float, frozenset, int]:
    """Bir (veya birden çok harfli) amino asit kodunun ortalama özellikleri."""
    if not isinstance(seq, str) or seq == "":
        return np.nan, np.nan, frozenset(), 0
    seq = seq.strip().upper()
    has_stop = int(STOP in seq)
    letters = [a for a in seq if a in KYTE_DOOLITTLE]
    if not letters:
        return np.nan, np.nan, frozenset(), has_stop
    hydro = float(np.mean([KYTE_DOOLITTLE[a] for a in letters]))
    mw = float(np.mean([AA_MOL_WEIGHT[a] for a in letters]))
    classes = frozenset(AA_POLARITY[a] for a in letters)
    return hydro, mw, classes, has_stop


def aa_features(ref: pd.Series, alt: pd.Series) -> pd.DataFrame:
    """AA_1 (referans) ve AA_2 (alternatif) amino asitlerden biyokimyasal değişim özellikleri."""
    r = ref.map(_residue_props)
    a = alt.map(_residue_props)
    out = pd.DataFrame(index=ref.index)
    out["aa_hydro_abs_delta"] = [abs(ai[0] - ri[0]) for ri, ai in zip(r, a)]
    out["aa_mw_abs_delta"] = [abs(ai[1] - ri[1]) for ri, ai in zip(r, a)]
    out["aa_polarity_change"] = [
        float(ri[2] != ai[2]) if (ri[2] and ai[2]) else np.nan for ri, ai in zip(r, a)
    ]
    # stop kodon oluşumu: alternatif dizide stop (*) var, referansta yok
    out["aa_stop_gain"] = [
        float(ai[3] and not ri[3]) if isinstance(av, str) else np.nan
        for ri, ai, av in zip(r, a, alt)
    ]
    return out


def al_features(al: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=al.index)
    out["al_filled_count"] = al.notna().sum(axis=1)
    out["al_max_freq"] = al.max(axis=1, skipna=True)
    # sağa çarpık frekans dağılımı için log dönüşümü
    out["al_log_max"] = np.log10(out["al_max_freq"] + 1e-8)
    out["al_n_pops_common"] = (al >= COMMON_AF).sum(axis=1)
    return out


def ek_features(ek: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ek.index)
    out["ek_mean"] = ek.mean(axis=1, skipna=True)
    out["ek7_x_ek9"] = ek["EK_7"] * ek["EK_9"]
    return out


class FeatureBuilder:
    """Ham panel tablosundan 305 özellikli model matrisini üretir.

    `fit` yalnızca sabit sütunları belirler (etiket kullanılmaz). Eksik değer
    doldurma ve kategorik kodlama fold içinde `FoldPreprocessor` ile yapılır.
    """

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        groups = column_groups(df)
        self.dropped_constant_ = find_constant_columns(df, groups["AL_"] + groups["EK_"] + groups["CAT_"])
        self.al_cols_ = [c for c in groups["AL_"] if c not in self.dropped_constant_]
        self.ek_cols_ = [c for c in groups["EK_"] if c not in self.dropped_constant_]
        self.aa_cols_ = groups["AA_"]
        # fold içinde kategori koduna çevrilecek sütunlar: CAT_ + AA_
        self.cat_cols_ = [c for c in groups["CAT_"] if c not in self.dropped_constant_] + self.aa_cols_
        self.al_out_of_range_ = check_al_range(df, self.al_cols_)
        self.feature_names_ = (
            self.al_cols_ + self.ek_cols_ + self.cat_cols_
            + AL_ENGINEERED + EK_ENGINEERED + AA_ENGINEERED + FLAG_COLS
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        al = df[self.al_cols_].apply(pd.to_numeric, errors="coerce")
        ek = df[self.ek_cols_].apply(pd.to_numeric, errors="coerce")
        cat = df[self.cat_cols_].astype("object")  # CAT_ + AA_1/AA_2
        parts = [
            al, ek, cat,
            al_features(al),
            ek_features(ek),
            aa_features(df["AA_1"], df["AA_2"]),
            df[FLAG_COLS].astype(int),
        ]
        X = pd.concat(parts, axis=1)
        return X[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def feature_group(name: str) -> str:
    """SHAP grup katkısı için özellik -> grup eşlemesi."""
    if name.startswith(AL_PREFIX):
        return "AL_"
    if name.startswith(EK_PREFIX):
        return "EK_"
    if name.startswith(CAT_PREFIX):
        return "CAT_"
    if name.startswith(AA_PREFIX) or name.startswith("aa_"):
        return "AA_ ve türevleri"
    return "Mühendislik"
