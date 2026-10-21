"""Figuras de los apuntes de lectura de la sesión 0.1."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from figuras import GRID, INK, MUTED, save

E0 = "#475569"
RED = "#DC2626"
AMBER = "#D97706"
TEAL = "#0D9488"
BLUE = "#2563EB"


def arbol_tipos():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 3.7)
    ax.axis("off")

    def node(x, y, t, fc, tc="white", w=2.3, h=0.7):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=fc, ec=fc if fc != "white" else E0, lw=1.3))
        ax.text(x, y, t, ha="center", va="center", fontsize=9, color=tc, fontweight="bold")

    def arrow(p1, p2, lab):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, lab, ha="center",
                fontsize=8.3, color=MUTED, style="italic")

    node(1.3, 1.8, "¿Tiene sentido\nPROMEDIARLA?", E0, h=0.95)
    node(4.6, 2.85, "NUMÉRICA", TEAL)
    node(4.6, 0.75, "CATEGÓRICA", AMBER)
    arrow((2.45, 2.05), (3.45, 2.8), "sí")
    arrow((2.45, 1.55), (3.45, 0.8), "no")
    node(8.5, 3.3, "Continua: colesterol, salario", "white", INK, w=3.4, h=0.5)
    node(8.5, 2.45, "Discreta: n° de hijos", "white", INK, w=3.4, h=0.5)
    node(8.5, 1.15, "Ordinal (tiene orden): Joven < Adulto", "white", INK, w=3.4, h=0.5)
    node(8.5, 0.3, "Nominal (sin orden): industria, fuma", "white", INK, w=3.4, h=0.5)
    arrow((5.75, 2.95), (6.7, 3.3), "")
    ax.text(6.0, 3.45, "¿cualquier valor?", fontsize=8.3, color=MUTED, style="italic", ha="center")
    arrow((5.75, 2.75), (6.7, 2.45), "")
    arrow((5.75, 0.85), (6.7, 1.15), "")
    ax.text(6.0, 1.3, "¿tiene orden?", fontsize=8.3, color=MUTED, style="italic", ha="center")
    arrow((5.75, 0.65), (6.7, 0.3), "")
    save(fig, "ap01_tipos.png")


def media_mediana():
    s = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 6.0])
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.scatter(s, np.zeros_like(s), s=120, color=E0, zorder=3)
    ax.axvline(np.median(s), color=TEAL, lw=2.2)
    ax.axvline(s.mean(), color=RED, lw=2.2, ls="--")
    ax.text(np.median(s) + 0.05, 0.75, f"mediana = {np.median(s):.2f}\n(el «típico»)",
            color=TEAL, fontsize=9.5, fontweight="bold")
    ax.text(s.mean() + 0.08, 0.2, f"media = {s.mean():.2f}\n(arrastrada por el gerente)",
            color=RED, fontsize=9.5, fontweight="bold")
    ax.annotate("gerente: 6.0", xy=(6.0, 0.05), xytext=(5.0, -0.7), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_ylim(-1, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("Sueldo (millones)")
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "ap01_media.png")


def anatomia_boxplot():
    y = np.array([2, 4, 5, 5, 6, 7, 8, 30])
    q1, med, q3 = 4.75, 5.5, 7.25
    iqr = q3 - q1
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.boxplot(y, orientation="horizontal", widths=0.45, patch_artist=True,
               boxprops=dict(facecolor="#CBD5E1", edgecolor=E0),
               medianprops=dict(color=RED, lw=2.2),
               whiskerprops=dict(color=E0), capprops=dict(color=E0),
               flierprops=dict(marker="o", mfc=RED, mec="white", ms=9))
    ax.scatter(y, np.full_like(y, 0.62, dtype=float), s=28, color=INK, zorder=3)
    ax.text(0.5, 0.62, "datos:", va="center", fontsize=8.5, color=MUTED)
    for x, tx, lab in [(q1, 2.2, f"Q1 = {q1}"), (med, med, f"mediana = {med}"), (q3, 9.6, f"Q3 = {q3}")]:
        ax.annotate(lab, xy=(x, 1.22), xytext=(tx, 1.62), ha="center", fontsize=8.8,
                    color=RED if "mediana" in lab else INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("", xy=(q1, 1.42), xytext=(q3, 1.42),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.4))
    ax.text((q1 + q3) / 2, 1.95, f"IQR = {iqr}", ha="center", fontsize=9, color=TEAL, fontweight="bold")
    lim = q3 + 1.5 * iqr
    ax.axvline(lim, color=AMBER, ls="--", lw=1.4)
    ax.text(lim + 0.3, 1.45, f"límite superior\nQ3 + 1.5·IQR = {lim:g}", fontsize=8.6, color=AMBER)
    ax.text(29.5, 0.7, "outlier", ha="right", fontsize=9, color=RED, fontweight="bold")
    ax.text(12.0, 0.77, "el bigote llega hasta el último\ndato DENTRO del límite (8)", fontsize=8.3, color=MUTED)
    ax.set_ylim(0.4, 2.15)
    ax.set_xlim(0, 32)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "ap01_boxplot.png")


def groupby_diagrama():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    rows = [("Sí", 210), ("No", 180), ("Sí", 190), ("No", 200), ("No", 160)]

    def table(x, y, data, color_fn, title):
        ax.text(x + 0.8, y + 0.35, title, ha="center", fontsize=9.5, fontweight="bold", color=INK)
        for i, (a, b) in enumerate(data):
            yy = y - i * 0.42
            c = color_fn(a)
            ax.add_patch(Rectangle((x, yy - 0.36), 0.8, 0.36, fc=c, ec="white"))
            ax.add_patch(Rectangle((x + 0.8, yy - 0.36), 0.8, 0.36, fc="#F3F4F6", ec="white"))
            ax.text(x + 0.4, yy - 0.18, a, ha="center", va="center", fontsize=8.8,
                    color="white", fontweight="bold")
            ax.text(x + 1.2, yy - 0.18, str(b), ha="center", va="center", fontsize=8.8, color=INK)

    col = lambda a: BLUE if a == "Sí" else AMBER
    table(0.2, 2.6, rows, col, "1 · Datos")
    table(3.3, 2.9, [r for r in rows if r[0] == "Sí"], col, "2 · Separar")
    table(3.3, 1.6, [r for r in rows if r[0] == "No"], col, "")
    table(6.3, 2.6, [("Sí", 200), ("No", 180)], col, "3 · Aplicar mean()")
    ax.text(8.15, 2.05, "4 · Combinar\nen una tabla", fontsize=9, color=INK, fontweight="bold", va="center")
    for y0, y1 in [(2.4, 2.7), (1.8, 1.4)]:
        ax.add_patch(FancyArrowPatch((2.0, (y0 + y1) / 2 + 0.1), (3.2, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=MUTED))
    ax.add_patch(FancyArrowPatch((5.0, 2.7), (6.2, 2.45), arrowstyle="-|>", mutation_scale=11, color=MUTED))
    ax.add_patch(FancyArrowPatch((5.0, 1.0), (6.2, 2.05), arrowstyle="-|>", mutation_scale=11, color=MUTED))
    ax.text(0.2, 0.15, "d.groupby('FUMA')['COLESTEROL'].mean()   →   Sí: 200 · No: 180",
            fontsize=9.5, family="Consolas", color=INK)
    save(fig, "ap01_groupby.png")


if __name__ == "__main__":
    arbol_tipos()
    media_mediana()
    anatomia_boxplot()
    groupby_diagrama()
    print("ok")
