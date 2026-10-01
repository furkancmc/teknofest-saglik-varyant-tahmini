"""Panel dosyalarının okunması, birleştirilmesi ve Variant_ID bazında tekilleştirilmesi."""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, FLAG_PANELS, ID_COL, PANEL_FILES, TARGET


def load_panels() -> dict[str, pd.DataFrame]:
    """Dört panel dosyasını (MASTER, KANSER, CFTR, PAH) okur."""
    panels = {}
    for name, fname in PANEL_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı. Veri dosyalarını {DATA_DIR} altına koyun.")
        panels[name] = pd.read_csv(path, low_memory=False)
    return panels


def panel_membership(panels: dict[str, pd.DataFrame]) -> dict[str, set]:
    """Her panelde yer alan Variant_ID kümeleri."""
    return {name: set(df[ID_COL]) for name, df in panels.items()}


def build_training_frame(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Panelleri birleştirip Variant_ID'ye göre tekilleştirir ve panel bayraklarını ekler.

    MASTER, KANSER, CFTR ve PAH dosyaları kısmi Variant_ID örtüşmesi içerir
    (toplam 3802 satır -> 3224 tekil varyant). Aynı varyantın farklı panellerde
    farklı etiket taşımadığı kontrol edilir. Bir varyant birden fazla panelde
    bulunuyorsa ilgili tüm `is_<PANEL>` bayrakları 1 olur.
    """
    stacked = pd.concat(
        [df.assign(_panel=name) for name, df in panels.items()], ignore_index=True
    )

    conflicts = stacked.groupby(ID_COL)[TARGET].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f"{len(conflicts)} varyant farklı panellerde çelişkili etikete sahip.")

    membership = panel_membership(panels)
    df = (
        stacked.drop_duplicates(subset=ID_COL, keep="first")
        .drop(columns="_panel")
        .reset_index(drop=True)
    )
    df = add_panel_flags(df, membership)
    return df


def add_panel_flags(df: pd.DataFrame, membership: dict[str, set]) -> pd.DataFrame:
    """is_PAH, is_CFTR, is_KANSER ikili değişkenlerini ekler."""
    df = df.copy()
    for p in FLAG_PANELS:
        df[f"is_{p}"] = df[ID_COL].isin(membership.get(p, set())).astype(int)
    return df


def set_panel_flags(df: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Test zamanında: dosyanın geldiği panele göre bayrakları sabitler."""
    df = df.copy()
    for p in FLAG_PANELS:
        df[f"is_{p}"] = int(panel.upper() == p)
    return df
