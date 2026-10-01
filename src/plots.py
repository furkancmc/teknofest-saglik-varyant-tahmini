"""Rapordaki şekiller (Şekil 1-5)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR
from .features import feature_group
from .metrics import threshold_scan

PATHO_COLOR = "#D95B30"
BENIGN_COLOR = "#1F9E77"
BLUE = "#5DA0E0"
METRIC_COLORS = {"F1": "#4A90D9", "MCC": "#E07B4F", "AUPRC": "#3FA796"}
GROUP_COLORS = {"EK_": "#E07650", "AA_ ve türevleri": "#C48A3A", "AL_": "#579FE0",
                "CAT_": "#8C8C8C", "Mühendislik": "#7A68A6"}

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name: str):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  şekil kaydedildi: {path}")
    return path


def plot_class_and_missingness(class_counts: pd.DataFrame, al_missing: pd.Series, ek_mean_missing: float):
    """Şekil 1: panel bazında sınıf dağılımı (%) (sol) ve AL_ sütunları eksiklik dağılımı (sağ).

    class_counts: index = panel, sütunlar = [Benign, Patojenik]
    al_missing  : AL_ sütunu başına eksiklik oranı (0-1)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    pct = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
    x = np.arange(len(pct))
    ax.bar(x, pct["Patojenik"], 0.55, color=PATHO_COLOR, label="Patojenik")
    ax.bar(x, pct["Benign"], 0.55, bottom=pct["Patojenik"], color=BENIGN_COLOR, label="Benign")
    for i, (p, b) in enumerate(zip(pct["Patojenik"], pct["Benign"])):
        ax.text(i, p / 2, f"%{p:.0f}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, p + b / 2, f"%{b:.0f}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x, pct.index)
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("Sınıf Dağılımı", fontweight="bold")
    ax.legend(loc="upper right")

    ax = axes[1]
    bins = [0, 20, 40, 60, 80, 100.0001]
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    counts = pd.cut(al_missing * 100, bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    bars = ax.bar(labels, counts.values, 0.6, color=BLUE)
    ax.bar_label(bars, fontsize=9)
    ax.set_xlabel("Eksik Değer %")
    ax.set_ylabel("AL_ Sütun Sayısı")
    ax.set_title("AL_ Sütunları: Eksiklik Dağılımı", fontweight="bold")
    print(f"  AL_ ort. eksiklik = %{al_missing.mean() * 100:.1f}, EK_ ort. eksiklik = %{ek_mean_missing * 100:.1f}")
    return _save(fig, "sekil1_sinif_dagilimi_eksiklik.png")


def plot_confusion_and_pr(y, proba, threshold: float, metrics: dict):
    """Şekil 2: genel veri kümesi karmaşıklık matrisi ve Precision-Recall eğrisi (OOF)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    cm = np.array([[metrics["TP"], metrics["FN"]], [metrics["FP"], metrics["TN"]]])
    names = np.array([["TP", "FN"], ["FP", "TN"]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i, j]}\n{cm[i, j]}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Tahmin: Patojenik", "Tahmin: Benign"])
    ax.set_yticks([0, 1], ["Gerçek: Patojenik", "Gerçek: Benign"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Genel Karmaşıklık Matrisi (n={metrics['n']}, eşik={threshold:.2f})", fontweight="bold")

    ax = axes[1]
    grid = np.round(np.arange(0.20, 0.701, 0.05), 2)
    pts = threshold_scan(y, proba, sorted(set(grid) | {round(threshold, 2)}))
    ax.plot(pts["Recall"], pts["Precision"], "-o", color="#1F77B4", ms=4)
    ax.scatter([metrics["Recall"]], [metrics["Precision"]], s=80, color="#1F77B4", zorder=3,
               label=f"Final eşik = {threshold:.2f}")
    ax.annotate(f"t={threshold:.2f}\nP={metrics['Precision']:.4f} R={metrics['Recall']:.4f}",
                xy=(metrics["Recall"], metrics["Precision"]),
                xytext=(-40, -45), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Recall (Duyarlılık)")
    ax.set_ylabel("Precision (Kesinlik)")
    ax.set_title(f"Precision-Recall Eğrisi (AUPRC={metrics['AUPRC']:.4f})", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    return _save(fig, "sekil2_karmasiklik_pr.png")


def plot_subgroup_metrics(sub: pd.DataFrame):
    """Şekil 3: alt grup bazlı F1, MCC, AUPRC karşılaştırması. sub index = alt grup, 'n' sütunu içerir."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(sub))
    w = 0.26
    for i, (m, c) in enumerate(METRIC_COLORS.items()):
        bars = ax.bar(x + (i - 1) * w, sub[m], w, label=m, color=c)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(sub.index, sub["n"])])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor")
    ax.set_title("Alt Grup Bazlı F1, MCC ve AUPRC Karşılaştırması (alt gruba özgü optimum eşik)",
                 fontweight="bold")
    ax.legend(ncol=3, loc="upper center", frameon=False)
    fig.text(0.01, -0.02, "Not: MASTER alt grubu eğitim verisiyle kısmi örtüşme içerdiğinden referans niteliğindedir.",
             fontsize=8, color="gray", style="italic")
    return _save(fig, "sekil3_alt_grup_metrikleri.png")


def plot_threshold_effect(rows: pd.DataFrame, best_thr: float):
    """Şekil 4: Tablo 8 eşiklerinde F1, MCC (sol eksen) ve FP, FN (sağ eksen) değişimi."""
    rows = rows.sort_values("threshold")
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    ax1.plot(rows["threshold"], rows["F1"], "-o", color="#1F77B4", lw=2, label="F1")
    ax1.plot(rows["threshold"], rows["MCC"], "-s", color="#FF7F0E", lw=2, label="MCC")
    ax1.set_xlabel("Karar Eşiği (Threshold)")
    ax1.set_ylabel("F1 / MCC")
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(rows["threshold"], rows["FP"], "--^", color="#1F77B4", label="FP (yanlış alarm)")
    ax2.plot(rows["threshold"], rows["FN"], "--v", color="#FF7F0E", label="FN (kaçırılan)")
    ax2.set_ylabel("FP / FN örnek sayısı")
    ax2.set_ylim(bottom=0)
    ax1.axvline(best_thr, color="#1F77B4", ls=":")
    f1_best = rows.loc[rows["threshold"] == best_thr, "F1"]
    if len(f1_best):
        ax1.annotate(f"Final eşik = {best_thr:.2f}", xy=(best_thr, f1_best.iloc[0]),
                     xytext=(25, 5), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
    ax1.grid(alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    ax1.set_title("Karar Eşiğine Göre F1, MCC, FP ve FN Değişimi (Genel Veri Kümesi)", fontweight="bold")
    return _save(fig, "sekil4_esik_etkisi.png")


def plot_shap_top(importance: pd.Series, top_n: int = 15):
    """Şekil 5: SHAP analizinde ilk 15 özelliğin ortalama mutlak katkısı (gruba göre renkli)."""
    top = importance.head(top_n)[::-1]
    colors = [GROUP_COLORS[feature_group(f)] for f in top.index]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Ortalama |SHAP| değeri")
    ax.set_title(f"Top {top_n} Özellik Önemi\n(EK_=kırmızı, AA_=sarı, AL_=mavi)", fontweight="bold")
    return _save(fig, "sekil5_shap_top15.png")
