"""Genera las figuras de las guías visuales (roadmap + carpeta Clases 17-19).

Uso: ejecutar con un Python que tenga pandas, numpy, matplotlib y openpyxl.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)
C1719 = BASE / "Clase 17,18,19-Analisis de regresion (Profesora ana maria alvarado)"

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
C17 = "#2563EB"
C18 = "#0D9488"
C19 = "#7C3AED"
PRE = "#6B7280"
FUT = "#D97706"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- roadmap
def roadmap():
    etapas = [
        ("E0", "Nivelación estadística", 1, 2, "#475569", "A"),
        ("E1", "Regresión · C17–19", 3, 5, C17, "A"),
        ("E2", "GLM · C20–21", 6, 7, "#0891B2", "A/B"),
        ("E3", "A/B testing · C22", 8, 8, C18, "A"),
        ("E4", "Bayesiana · C23–24", 9, 10, "#059669", "A"),
        ("E5", "Series de tiempo · C25–27", 11, 12, "#65A30D", "B"),
        ("E6", "Fundamentos ML · C30–32", 13, 15, C19, "A"),
        ("E7", "Modelos · C33–35", 16, 18, "#C026D3", "A"),
        ("E8", "C28–29 y C36–40", 19, 21, "#9CA3AF", "?"),
    ]
    inicio = pd.Timestamp("2026-10-05")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (cod, nombre, a, b, color, nivel) in enumerate(etapas):
        y = len(etapas) - i
        hatch = "///" if cod == "E8" else None
        ax.barh(y, b - a + 1, left=a - 0.5, height=0.62, color=color,
                alpha=0.35 if cod == "E8" else 0.95, hatch=hatch,
                edgecolor="white" if hatch is None else "#6B7280", linewidth=0)
        ax.text(0.2, y, f"{cod}", ha="right", va="center", fontsize=10,
                fontweight="bold", color=INK)
        txt_x = b + 0.75
        ax.text(txt_x, y, f"{nombre}", va="center", fontsize=9.5, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(a - 0.5 + (b - a + 1) / 2, y, nivel, ha="center", va="center",
                fontsize=8.5, color="white" if cod != "E8" else INK, fontweight="bold")
    # project marker (week 5 = 2-8 nov)
    ax.axvline(5.0, color=FUT, lw=1.6, ls="--", zorder=1)
    ax.text(5.0, len(etapas) + 0.85, "Entrega Proyecto N°2 (~5 nov)", color=FUT,
            ha="center", fontsize=9, fontweight="bold")
    semanas = np.arange(1, 22)
    labels = [f"S{s}\n{(inicio + pd.Timedelta(weeks=int(s) - 1)).strftime('%d/%m')}"
              for s in semanas]
    ax.set_xticks(semanas)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlim(-0.6, 27.5)
    ax.set_ylim(0.3, len(etapas) + 1.3)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "roadmap.png")


# --------------------------------------------------------------- concept map
def concept_map():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 13.7)
    ax.set_ylim(0, 8)
    ax.axis("off")

    nodes = {
        "inf": (1.4, 7.0, "Inferencia\np-valor · H0/H1", PRE, False),
        "a1": (5.0, 7.0, "ANOVA\n1 factor", C17, False),
        "a2": (8.6, 7.0, "ANOVA 2 factores\n+ interacción", C17, False),
        "ab": (12.3, 7.0, "A/B testing\nClase 22", FUT, True),
        "cor": (1.4, 4.4, "Correlación\nPearson · Spearman", C18, False),
        "rls": (5.0, 4.4, "Regresión\nlineal simple", C18, False),
        "diag": (8.6, 4.4, "Diagnóstico\nde residuos", C18, False),
        "glm": (12.3, 4.4, "GLM\nClases 20–21", FUT, True),
        "dum": (1.4, 1.8, "Variables\ndummy", C19, False),
        "rlm": (5.0, 1.8, "Regresión\nlineal múltiple", C19, False),
        "vif": (8.6, 2.45, "Multicolinealidad\n(VIF)", C19, False),
        "sel": (8.6, 0.85, "Selección de variables\nAIC · BIC · stepwise", C19, False),
        "ml": (12.3, 1.6, "Machine Learning\nClases 30+", FUT, True),
    }
    w, h = 2.45, 1.05
    for key, (x, y, label, color, future) in nodes.items():
        hh = 0.95 if key in ("vif", "sel") else h
        box = FancyBboxPatch((x - w / 2, y - hh / 2), w, hh,
                             boxstyle="round,pad=0.02,rounding_size=0.18",
                             fc="white" if future else color,
                             ec=color, lw=1.8, ls="--" if future else "-")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.6,
                color=color if future else "white", fontweight="bold")

    def arrow(a, b, label=None, dashed=False, rad=0.0, lpos=0.5, loff=(0, 0.22)):
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        dx, dy = xb - xa, yb - ya
        # start/end on box borders
        def border(x, y, dx, dy, key):
            hh = (0.95 if key in ("vif", "sel") else h) / 2
            tx = (w / 2) / abs(dx) if dx else np.inf
            ty = hh / abs(dy) if dy else np.inf
            t = min(tx, ty)
            return x + dx * t, y + dy * t
        p1 = border(xa, ya, dx, dy, a)
        p2 = border(xb, yb, -dx, -dy, b)
        arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                              color=MUTED, lw=1.4, ls="--" if dashed else "-",
                              connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arr)
        if label:
            mx = p1[0] + (p2[0] - p1[0]) * lpos + loff[0]
            my = p1[1] + (p2[1] - p1[1]) * lpos + loff[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                    color=MUTED, style="italic",
                    bbox=dict(fc="white", ec="none", pad=0.5))

    arrow("inf", "a1", "compara medias")
    arrow("a1", "a2", "más factores")
    arrow("a2", "ab", "mismo marco de test", dashed=True)
    arrow("inf", "cor", "¿es significativa?", loff=(0.95, 0))
    arrow("cor", "rls", "R² = r²")
    arrow("rls", "diag", "valida supuestos")
    arrow("diag", "glm", "si Y no es normal", dashed=True)
    arrow("rls", "rlm", "k predictores", loff=(0.85, 0))
    arrow("dum", "rlm", "categóricas")
    arrow("rlm", "vif")
    arrow("rlm", "sel")
    arrow("sel", "ml", "regularización, árboles", dashed=True, loff=(0, 0.3))

    # legend
    items = [("Prerrequisito", PRE, False), ("Clase 17", C17, False),
             ("Clase 18", C18, False), ("Clase 19", C19, False),
             ("Tema futuro", FUT, True)]
    lx = 0.3
    for name, color, fut in items:
        ax.add_patch(FancyBboxPatch((lx, -0.55), 0.32, 0.26,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="white" if fut else color, ec=color, lw=1.4,
                                    ls="--" if fut else "-", clip_on=False))
        ax.text(lx + 0.45, -0.42, name, va="center", fontsize=8.8, color=INK)
        lx += 2.2
    save(fig, "mapa_c17_19.png")


# -------------------------------------------------------------- class 17
def fig_c17():
    d = pd.read_excel(C1719 / "salarios.xlsx")
    orden = list(d.groupby("Industria")["Salario"].mean().sort_values().index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    data = [d.loc[d.Industria == i, "Salario"] for i in orden]
    bp = ax1.boxplot(data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                     flierprops=dict(marker="o", mfc=C17, mec="white"))
    for b in bp["boxes"]:
        b.set(facecolor=C17, alpha=0.85, edgecolor=C17)
    rng = np.random.default_rng(1)
    for k, serie in enumerate(data, start=1):
        ax1.scatter(k + rng.uniform(-0.12, 0.12, len(serie)), serie, s=12,
                    color=INK, alpha=0.55, zorder=3)
    ax1.set_xticks(range(1, len(orden) + 1))
    ax1.set_xticklabels(orden)
    ax1.set_ylabel("Salario")
    ax1.set_title("¿Cambia el salario según la industria?", fontsize=10.5,
                  color=INK, loc="left")
    ax1.grid(axis="y", color=GRID, lw=0.6)

    medias = d.groupby(["Industria", "Sexo"])["Salario"].mean().unstack()
    medias = medias.loc[orden]
    colores = {"H": C17, "M": "#DB2777"}
    nombres = {"H": "Hombre (H)", "M": "Mujer (M)"}
    for sexo in medias.columns:
        ax2.plot(range(len(orden)), medias[sexo], marker="o", lw=2.2,
                 color=colores.get(sexo, MUTED), label=nombres.get(sexo, sexo))
    ax2.set_xticks(range(len(orden)))
    ax2.set_xticklabels(orden)
    ax2.set_ylabel("Salario promedio")
    ax2.set_title("Interacción: si las líneas se cruzan,\nel efecto del sexo depende de la industria",
                  fontsize=10.5, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout(w_pad=3)
    save(fig, "c17.png")


# -------------------------------------------------------------- class 18
def fig_c18():
    d = pd.read_csv(C1719 / "biomass.txt", sep="\t")
    x, y = d["Ph"].to_numpy(float), d["BIO"].to_numpy(float)
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    r = np.corrcoef(x, y)[0, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    for xi, yi, yh in zip(x, y, yhat):
        ax1.plot([xi, xi], [yi, yh], color=C18, alpha=0.35, lw=1)
    ax1.scatter(x, y, s=22, color=INK, alpha=0.75, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ax1.plot(xs, b0 + b1 * xs, color=C18, lw=2.6)
    ax1.set_xlabel("pH del suelo")
    ax1.set_ylabel("Biomasa (BIO)")
    ax1.set_title(f"BIO = {b0:.0f} + {b1:.0f}·pH     r = {r:.2f}   R² = {r**2:.2f}",
                  fontsize=10.5, color=INK, loc="left")
    ax1.text(0.02, 0.95, "líneas verticales = residuos (error)", transform=ax1.transAxes,
             fontsize=8.5, color=C18, va="top")
    ax1.grid(color=GRID, lw=0.6)

    ax2.scatter(yhat, y - yhat, s=20, color=C18, alpha=0.8)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xlabel("Valores ajustados")
    ax2.set_ylabel("Residuo")
    ax2.set_title("Residuos vs ajustados:\nbuscamos una nube sin forma", fontsize=10.5,
                  color=INK, loc="left")
    ax2.grid(color=GRID, lw=0.6)
    fig.tight_layout(w_pad=3)
    save(fig, "c18.png")
    return b0, b1, r


# -------------------------------------------------------------- class 19
def fig_c19():
    d = pd.read_csv(C1719 / "biomass.txt", sep="\t")
    num = d.drop(columns=["LOC", "TYPE", "BIO"]).astype(float)
    vifs = {}
    for col in num.columns:
        yv = num[col].to_numpy()
        X = np.column_stack([np.ones(len(num)), num.drop(columns=col).to_numpy()])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res = yv - X @ beta
        r2 = 1 - res.var() / yv.var()
        vifs[col] = 1 / (1 - r2)
    s = pd.Series(vifs).sort_values()
    colors = [C19 if v < 5 else ("#F59E0B" if v < 10 else "#DC2626") for v in s]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(s.index, s.values, color=colors)
    ax.set_xlim(-0.7, len(s) - 0.4)
    ax.axhline(5, color="#F59E0B", ls="--", lw=1.2)
    ax.axhline(10, color="#DC2626", ls="--", lw=1.2)
    ax.text(-0.6, 5.6, "5 = revisar", color="#B45309", fontsize=8.5, ha="left")
    ax.text(-0.6, 10.6, "10 = problema severo", color="#B91C1C", fontsize=8.5, ha="left")
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7.8, color=INK)
    ax.set_ylabel("VIF")
    ax.set_title("VIF de cada predictor numérico de biomass.txt (modelo con todas las variables)",
                 fontsize=10.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "c19.png")
    return s


if __name__ == "__main__":
    roadmap()
    concept_map()
    fig_c17()
    print("C18:", fig_c18())
    print("VIF:\n", fig_c19().round(1))
