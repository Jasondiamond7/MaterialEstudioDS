"""Figuras de la Clase 34 (árboles de decisión). Datos: base Churn de la Clase 33, limpiada igual que en clase."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from figuras import GRID, INK, MUTED, save

BASE = Path(__file__).resolve().parents[2]
D = pd.read_excel(next(BASE.glob("Clase 33*")) / "Base de datos Churn.xlsx")
for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
    D.loc[D[c] < 0, c] = np.nan
D.loc[(D.edad < 15) | (D.edad > 100), "edad"] = np.nan
D = D.fillna(D.median())
y = D.pop("fuga")
X = D

C34 = "#16A34A"
RED = "#DC2626"
BLUE = "#2563EB"
AMBER = "#D97706"
GRAY = "#6B7280"
PURPLE = "#7C3AED"


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def arbol():
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0).fit(X, y).tree_
    N = t.n_node_samples[0]
    pos = {}

    def layout(i, depth, x0, x1):
        pos[i] = ((x0 + x1) / 2, 3.2 - depth * 1.05)
        if t.children_left[i] != -1:
            m = (x0 + x1) / 2
            layout(t.children_left[i], depth + 1, x0, m)
            layout(t.children_right[i], depth + 1, m, x1)

    layout(0, 0, 0, 14)
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-0.3, 3.8)
    ax.axis("off")
    for i, (x, yy) in pos.items():
        n = t.n_node_samples[i]
        v = t.value[i][0]
        rate = v[1] / v.sum()
        leaf = t.children_left[i] == -1
        if leaf:
            txt = f"n = {n}\nfuga {100 * rate:.0f}%"
        else:
            var = X.columns[t.feature[i]]
            txt = f"{var} ≤ {t.threshold[i]:.1f}\nn = {n} · fuga {100 * rate:.0f}%"
        col = plt.cm.RdYlGn_r(min(rate / 0.9, 1.0))
        w = 1.55 if leaf else 2.3
        ax.add_patch(FancyBboxPatch((x - w / 2, yy - 0.33), w, 0.66, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=col, ec="white", lw=1.5, alpha=0.95))
        ax.text(x, yy, txt, ha="center", va="center", fontsize=8.2, color="white" if rate > 0.55 or rate < 0.08 else INK,
                fontweight="bold")
        if not leaf:
            for ch, lab in ((t.children_left[i], "sí"), (t.children_right[i], "no")):
                cx, cy = pos[ch]
                ax.add_patch(FancyArrowPatch((x, yy - 0.33), (cx, cy + 0.33), arrowstyle="-|>", mutation_scale=10,
                                             color=MUTED, lw=1.1))
                ax.text((x + cx) / 2 + (0.15 if lab == "no" else -0.15), (yy + cy) / 2, lab, fontsize=8, color=MUTED,
                        ha="center", style="italic")
    ax.text(0, 3.65, "Árbol de profundidad 3 (min_samples_leaf = 60) con los datos de la clase. Color: rojo = más fuga.",
            fontsize=9.5, color=INK)
    save(fig, "c34_arbol.png")


def impureza():
    p = np.linspace(0.001, 0.999, 300)
    gini = 1 - p ** 2 - (1 - p) ** 2
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    a1.plot(p, gini, color=C34, lw=2.2, label="Gini (máx 0.5)")
    a1.plot(p, ent, color=PURPLE, lw=2.2, label="Entropía (máx 1)")
    a1.axvline(0.157, color=RED, ls="--", lw=1.2)
    a1.text(0.17, 0.85, "raíz: 15.7% de fuga\nGini = 0.265", fontsize=8.5, color=RED)
    a1.set_xlabel("Proporción de fuga en el nodo (p)")
    a1.set_ylabel("Impureza")
    a1.set_title("Las dos medidas valen 0 si el nodo es puro", loc="left", fontsize=9.8, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a1)
    labels = ["Raíz\n3.150 clientes", "Reclamó\n278", "No reclamó\n2.872"]
    rates = [0.157, 210 / 278, 285 / 2872]
    ginis = [0.265, 0.370, 0.179]
    bars = a2.bar(labels, rates, color=[GRAY, RED, C34])
    for b, r, gv in zip(bars, rates, ginis):
        a2.text(b.get_x() + b.get_width() / 2, r + 0.02, f"fuga {100 * r:.1f}%\nGini {gv:.3f}", ha="center",
                fontsize=8.5, color=INK)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Tasa de fuga")
    a2.set_title("El primer corte: reclamos. Gini ponderado 0.196 → ganancia 0.069", loc="left", fontsize=9.8,
                 color=INK)
    style(a2, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "c34_impureza.png")


def profundidad():
    prof = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    tr = [0.6993, 0.8290, 0.8554, 0.8987, 0.9275, 0.9402, 0.9688, 0.9851, 0.9989, 1.0]
    va = [0.6993, 0.8107, 0.8495, 0.8792, 0.9058, 0.9114, 0.8917, 0.8817, 0.8379, 0.8267]
    labels = [str(p) for p in prof[:-1]] + ["sin\nlímite"]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    x = np.arange(len(prof))
    ax.plot(x, tr, marker="o", color=GRAY, lw=2, label="AUC entrenamiento")
    ax.plot(x, va, marker="o", color=C34, lw=2.4, label="AUC validación (CV)")
    ax.fill_between(x, va, tr, color=RED, alpha=0.08)
    ax.annotate("mejor: profundidad 6\nAUC 0.911", xy=(5, 0.9114), xytext=(3.3, 0.96), fontsize=8.8, color=C34,
                arrowprops=dict(arrowstyle="->", color=C34))
    ax.text(8.2, 0.9, "la brecha = sobreajuste\n224 hojas: memorizó", fontsize=8.8, color=RED, ha="center")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Profundidad máxima del árbol")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("Más profundidad: entrenamiento perfecto, validación peor", loc="left", fontsize=9.8, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax)
    save(fig, "c34_profundidad.png")


def poda_importancia():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    alpha = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    hojas = [224, 116, 42, 15, 5, 3]
    auc = [0.8267, 0.8506, 0.8996, 0.8832, 0.8345, 0.8053]
    xs = np.arange(len(alpha))
    a1.bar(xs, hojas, color=GRAY, alpha=0.5)
    for i, h in enumerate(hojas):
        a1.text(i, h + 4, f"{h} hojas", ha="center", fontsize=7.8, color=MUTED)
    a1.set_ylabel("Hojas", color=MUTED)
    a1.set_xticks(xs, [str(a) for a in alpha])
    a1.set_xlabel("ccp_alpha (más alto = más poda)")
    b = a1.twinx()
    b.plot(xs, auc, marker="o", color=C34, lw=2.4)
    b.set_ylabel("AUC validación", color=C34)
    b.set_ylim(0.78, 0.93)
    a1.set_title("Poda por costo-complejidad: el punto dulce", loc="left", fontsize=9.8, color=INK)
    a1.spines["right"].set_visible(True)
    imp = pd.DataFrame({"impureza": [0.2058, 0.1175, 0.1169, 0.1127, 0.1089, 0.0741, 0.0696, 0.0661, 0.0611, 0.0452],
                        "permutación": [0.0352, 0.0315, 0.0313, 0.0227, 0.0140, 0.0121, 0.0080, 0.0206, 0.0087, 0.0185]},
                       index=["reclamos", "segundos", "frec_llamadas", "antiguedad", "clv", "n_contactos", "edad",
                              "numero_fallas", "frec_sms", "estado"])
    imp = imp / imp.sum()
    yy = np.arange(len(imp))[::-1]
    a2.barh(yy + 0.2, imp["impureza"], height=0.4, color=GRAY, label="por impureza")
    a2.barh(yy - 0.2, imp["permutación"], height=0.4, color=C34, label="por permutación")
    a2.set_yticks(yy, imp.index)
    a2.set_xlabel("Importancia relativa (suma 1 en cada método)")
    a2.set_title("La impureza premia a las variables con muchos valores", loc="left", fontsize=9.8, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    style(a2, "x")
    fig.tight_layout(w_pad=3)
    save(fig, "c34_poda.png")


def bosque():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    n = [1, 5, 25, 100, 300]
    auc = [0.7973, 0.9311, 0.9677, 0.9729, 0.9736]
    a1.plot(range(len(n)), auc, marker="o", color=C34, lw=2.4)
    for i, v in enumerate(auc):
        a1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    a1.set_xticks(range(len(n)), [str(k) for k in n])
    a1.set_xlabel("Número de árboles en el bosque")
    a1.set_ylabel("AUC (CV)")
    a1.set_ylim(0.77, 1.0)
    a1.set_title("Promediar árboles estabiliza y mejora", loc="left", fontsize=9.8, color=INK)
    style(a1)
    mods = ["Árbol prof. 3", "Árbol prof. 5", "Árbol podado (CV)", "Regresión logística", "Random Forest (300)"]
    v = [0.8495, 0.9058, 0.9092, 0.9199, 0.9736]
    a2.barh(mods, v, color=[GRAY, GRAY, C34, BLUE, PURPLE])
    for i, x in enumerate(v):
        a2.text(x + 0.003, i, f"{x:.3f}", va="center", fontsize=8.5, color=INK)
    a2.set_xlim(0.8, 1.0)
    a2.set_title("Comparación final (AUC, validación cruzada)", loc="left", fontsize=9.8, color=INK)
    style(a2, "x")
    fig.tight_layout(w_pad=3)
    save(fig, "c34_bosque.png")


if __name__ == "__main__":
    arbol()
    impureza()
    profundidad()
    poda_importancia()
    bosque()
    print("ok")
