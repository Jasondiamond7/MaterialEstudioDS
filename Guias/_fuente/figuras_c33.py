"""Figuras de la Clase 33 (Feature engineering y selección de variables)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from figuras import GRID, INK, MUTED, save

BASE = Path(__file__).resolve().parents[2]
CARPETA = next(BASE.glob("Clase 33*"))
D = pd.read_excel(CARPETA / "Base de datos Churn.xlsx")

C33 = "#EA580C"
BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#059669"
PURPLE = "#7C3AED"
GRAY = "#6B7280"
AMBER = "#D97706"


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def mapa():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    W, H = 2.15, 1.15
    steps = [(1.2, "Datos crudos\n3.150 × 12", GRAY), (3.6, "1 · Limpiar\ncon criterio\nde negocio", C33),
             (6.0, "2 · Construir\nvariables\n12 → 50", C33), (8.4, "3 · Seleccionar\nfiltro · embedded\n· wrapper", C33),
             (10.9, "Modelo\n(logística, árboles…)", BLUE)]
    for x, t, c in steps:
        ax.add_patch(FancyBboxPatch((x - W / 2, 2.9 - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=c, ec=c))
        ax.text(x, 2.9, t, ha="center", va="center", fontsize=9.4, color="white", fontweight="bold")
    for (x1, *_), (x2, *_) in zip(steps, steps[1:]):
        ax.add_patch(FancyArrowPatch((x1 + W / 2, 2.9), (x2 - W / 2, 2.9), arrowstyle="-|>", mutation_scale=13,
                                     color=MUTED, lw=1.4))
    ax.add_patch(FancyBboxPatch((7.15, 2.05), 5.05, 1.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=RED, lw=1.8, ls="--"))
    ax.text(9.65, 4.0, "dentro de cada fold de la validación cruzada", ha="center", fontsize=9, color=RED,
            fontweight="bold")
    ax.text(3.6, 1.95, "no mira el target:\npuede ir antes del split", ha="center", fontsize=8.3, color=MUTED,
            style="italic")
    ax.text(6.0, 1.95, "no mira el target:\npuede ir antes del split", ha="center", fontsize=8.3, color=MUTED,
            style="italic")
    futures = [(3.6, "Clase 34 · Árboles\nganancia de información"), (6.6, "Clase 35 · Random Forest\nresuelve el XOR"),
               (9.6, "Clase 36 · Boosting\nimportancia / SHAP")]
    for x, t in futures:
        ax.add_patch(FancyBboxPatch((x - 1.3, 0.25), 2.6, 0.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=AMBER, lw=1.4, ls="--"))
        ax.text(x, 0.72, t, ha="center", va="center", fontsize=8.6, color=AMBER, fontweight="bold")
    save(fig, "c33_mapa.png")


def limpieza():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    cats = ["segundos < 0", "frec_llamadas < 0", "antigüedad < 0", "frec_sms < 0", "edad < 0", "edad 0–14", "edad > 100"]
    vals = [(D.segundos < 0).sum(), (D.frec_llamadas < 0).sum(), (D.antiguedad < 0).sum(), (D.frec_sms < 0).sum(),
            (D.edad < 0).sum(), D.edad.between(0, 14).sum(), (D.edad > 100).sum()]
    cols = [RED] * 5 + [AMBER, RED]
    a1.barh(cats[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        a1.text(v + 1.5, i, str(v), va="center", fontsize=8.5, color=INK)
    a1.set_title("Valores imposibles o dudosos (rojo = imposible, naranja = decisión de negocio)", loc="left",
                 fontsize=9.6, color=INK)
    style(a1, "x")
    e = D.edad
    a2.hist(e.clip(-20, 120), bins=60, color=GRAY, alpha=0.8, edgecolor="white")
    a2.axvspan(-20, 15, color=RED, alpha=0.08)
    a2.axvspan(100, 120, color=RED, alpha=0.08)
    a2.axvline(15, color=RED, ls="--", lw=1.2)
    a2.axvline(100, color=RED, ls="--", lw=1.2)
    a2.text(80, a2.get_ylim()[1] * 0.6, "regla de negocio:\nedad entre 15 y 100", ha="center", fontsize=8.5, color=RED)
    a2.text(108, a2.get_ylim()[1] * 0.3, "233 →", fontsize=8.5, color=RED, ha="center")
    a2.set_xlabel("Edad (recortada a [−20, 120] para graficar)")
    a2.set_title("Distribución de la edad", loc="left", fontsize=9.6, color=INK)
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "c33_limpieza.png")


def conjunto():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    labels = ["numero_fallas\nsola", "base de\n5 variables", "base +\nnumero_fallas", "base + n_fallas\n+ monto"]
    v = [0.5261, 0.9069, 0.9102, 0.9171]
    a1.bar(labels, v, color=[RED, GRAY, C33, C33])
    for i, x in enumerate(v):
        a1.text(i, x + 0.01, f"{x:.3f}", ha="center", fontsize=8.6, color=INK)
    a1.axhline(0.5, color=MUTED, ls=":", lw=1)
    a1.text(3.4, 0.51, "azar", fontsize=8, color=MUTED, ha="right")
    a1.set_ylim(0.45, 1.0)
    a1.set_ylabel("AUC (validación cruzada)")
    a1.set_title("Inútil sola, útil en conjunto: numero_fallas", loc="left", fontsize=9.6, color=INK)
    style(a1)
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=600), rng.normal(size=600)
    y = (np.sign(x1) != np.sign(x2)).astype(int)
    a2.scatter(x1[y == 0], x2[y == 0], s=9, color=BLUE, alpha=0.6, label="clase 0")
    a2.scatter(x1[y == 1], x2[y == 1], s=9, color=RED, alpha=0.6, label="clase 1")
    a2.axhline(0, color=INK, lw=0.8)
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_title("XOR: ninguna variable separa sola; juntas, perfecto", loc="left", fontsize=9.6, color=INK)
    a2.set_xlabel("x1")
    a2.set_ylabel("x2")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    style(a2, "both")
    fig.tight_layout(w_pad=3)
    save(fig, "c33_conjunto.png")


def lasso_y_resumen():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    C = [0.01, 0.05, 0.1, 0.5, 1.0]
    n = [13, 27, 33, 42, 45]
    a1.plot(C, n, marker="o", color=PURPLE, lw=2.2)
    a1.set_xscale("log")
    for c, k in zip(C, n):
        a1.text(c, k + 1.5, str(k), ha="center", fontsize=8.5, color=INK)
    a1.set_xlabel("C (escala log) — más grande = MENOS penalización")
    a1.set_ylabel("Variables vivas (de 50)")
    a1.set_ylim(5, 52)
    a1.set_title("Lasso: C controla cuántas variables sobreviven", loc="left", fontsize=9.6, color=INK)
    style(a1, "both")
    est = ["12 originales", "50 candidatas", "Lasso (27)", "Top-4 por MI", "4 elegidas\nsin redundancia"]
    auc = [0.9199, 0.9406, 0.9396, 0.8040, 0.8934]
    a2.barh(est[::-1], auc[::-1], color=[GREEN, RED, PURPLE, C33, GRAY][::-1])
    for i, x in enumerate(auc[::-1]):
        a2.text(x + 0.003, i, f"{x:.3f}", va="center", fontsize=8.5, color=INK)
    a2.set_xlim(0.75, 0.97)
    a2.set_xlabel("AUC (validación cruzada, 5 folds)")
    a2.set_title("Resumen de estrategias (clase)", loc="left", fontsize=9.6, color=INK)
    style(a2, "x")
    fig.tight_layout(w_pad=3)
    save(fig, "c33_lasso.png")


def leakage():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    lab = ["Selección FUERA del CV\n(con todos los datos)", "Selección DENTRO del pipeline", "La verdad\n(datos puro ruido)"]
    v = [0.882, 0.514, 0.500]
    ax.barh(lab[::-1], v[::-1], color=[GRAY, GREEN, RED][::-1])
    for i, x in enumerate(v[::-1]):
        ax.text(x + 0.008, i, f"AUC {x:.3f}", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.annotate("", xy=(0.514, 2.35), xytext=(0.882, 2.35), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(0.70, 2.48, "38 puntos de AUC inventados", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.6, 2.8)
    ax.set_title("100 observaciones × 1.000 variables aleatorias, target al azar: el AUC real es 0.5",
                 loc="left", fontsize=9.8, color=INK)
    style(ax, "x")
    save(fig, "c33_leakage.png")


if __name__ == "__main__":
    mapa()
    limpieza()
    conjunto()
    lasso_y_resumen()
    leakage()
    print("ok")
