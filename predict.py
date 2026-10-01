"""Kaydedilmiş final model ile yeni (test) panel dosyası için tahmin üretir.

Kullanım:
    python predict.py --input YARISMA_TEST_PAH.csv --panel PAH --output tahmin_PAH.csv
    python predict.py --input YARISMA_TEST_MASTER.csv              # panel dosya adından çıkarılır

Eşik: varsayılan olarak alt gruba özgü eşik; `--threshold-mode global` ile OOF global eşiği.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from src.config import ID_COL, MODEL_DIR, PANEL_FILES
from src.data import set_panel_flags


def infer_panel(path: Path) -> str:
    name = path.stem.upper()
    for panel in PANEL_FILES:
        if panel in name:
            return panel
    raise ValueError("Panel dosya adından anlaşılamadı; --panel ile belirtin.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--panel", choices=list(PANEL_FILES), default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--threshold-mode", choices=["panel", "global"], default="panel")
    ap.add_argument("--model", type=Path, default=MODEL_DIR / "final_model.joblib")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    panel = args.panel or infer_panel(args.input)

    df = set_panel_flags(pd.read_csv(args.input, low_memory=False), panel)
    X = bundle["feature_builder"].transform(df)
    X_pre = bundle["preprocessor"].transform(X)
    proba = bundle["ensemble"].predict_proba(X_pre)[:, 1]

    thr = bundle["global_threshold"] if args.threshold_mode == "global" else bundle["panel_thresholds"][panel]
    out = pd.DataFrame({ID_COL: df[ID_COL], "proba": proba.round(6), "Label": (proba >= thr).astype(int)})

    output = args.output or Path(f"tahmin_{panel}.csv")
    out.to_csv(output, index=False)
    print(f"{panel}: {len(out)} varyant, eşik={thr:.2f}, patojenik tahmin={int(out['Label'].sum())} -> {output}")


if __name__ == "__main__":
    main()
