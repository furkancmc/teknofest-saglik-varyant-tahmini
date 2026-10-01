"""Proje genel ayarları: yollar, sabitler, tohum (seed) ve sabit hiperparametreler."""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Yarışma eğitim dosyalarının bulunduğu klasör. Orijinal klasör adı Türkçe
# karakter içerdiği için `data/` klasörü de alternatif olarak desteklenir.
_DATA_CANDIDATES = [ROOT / "EĞİTİM (TRAIN) SETLERİ", ROOT / "data"]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])

# Panel adı -> dosya adı. Sıra önemlidir: tekilleştirmede ilk görülen satır tutulur.
PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
}
# İkili panel değişkeni üretilecek paneller (MASTER referans paneldir).
FLAG_PANELS = ["PAH", "CFTR", "KANSER"]

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
OOF_DIR = OUTPUT_DIR / "oof"
OPTUNA_DIR = OUTPUT_DIR / "optuna"
SELECTION_DIR = OUTPUT_DIR / "selection"

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, OOF_DIR, OPTUNA_DIR, SELECTION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Veri sütunları
# ---------------------------------------------------------------------------
ID_COL = "Variant_ID"
TARGET = "Label"  # 1 = patojenik, 0 = benign

AL_PREFIX, EK_PREFIX, CAT_PREFIX, AA_PREFIX = "AL_", "EK_", "CAT_", "AA_"

# ---------------------------------------------------------------------------
# Deney ayarları
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5          # model değerlendirmesi: 5 katlı stratified CV
OPTUNA_CV_SPLITS = 3  # Optuna içi CV
LGBM_N_TRIALS = 60
XGB_N_TRIALS = 40

TOP_K = 50       # SHAP ile seçilen ana özellik sayısı
TOP_K_ALT = 59   # Optuna ile ayarlanan üyelerin kullandığı genişletilmiş set

# Karar eşiği taraması (OOF olasılıkları üzerinde)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
REPORT_THRESHOLDS = [0.20, 0.60]  # Tablo 8'de final eşikle birlikte raporlanan eşikler

# Yaygın varyant eşiği (al_n_pops_common için)
COMMON_AF = 0.01

# PAH özel strateji
PAH_BOOTSTRAP_TARGET = 200     # PAH benign örnek sayısı 62 -> 200
PAH_REFERENCE_THRESHOLD = 0.42  # Tablo 9 "genel model, referans eşik"

# ---------------------------------------------------------------------------
# Sabit model parametreleri
# ---------------------------------------------------------------------------
# Tablo 5 satır 1-4 (305 özellik) ve SHAP özellik seçimi için temel LightGBM
BASE_LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
)

# Tablo 5 satır 5: SHAP top-50; num_leaves=40; lr=0,05
MANUAL_LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
)

# CatBoost: depth=6, iterations=400, learning_rate=0,05, l2_leaf_reg=5 sabit
CATBOOST_PARAMS = dict(depth=6, iterations=400, learning_rate=0.05, l2_leaf_reg=5)
