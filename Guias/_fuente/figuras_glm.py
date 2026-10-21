"""Figuras de la Carpeta 2 (GLM, Clases 20-21). Requiere statsmodels y scikit-learn."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
DATOS = Path(__file__).resolve().parents[1] / "datos"
C20 = "#0891B2"
BLUE = "#2563EB"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#059669"
PURPLE = "#7C3AED"
PINK = "#DB2777"
PRE = "#6B7280"

w = pd.read_csv(DATOS / "wcgs.csv")
w["chd01"] = (w.chd == "yes").astype(int)
cancer = pd.read_csv(DATOS / "cancer_pulmon.csv")
nes = pd.read_csv(DATOS / "nes96.csv")
nes["y"] = np.where(nes.PID.isin(["strDem", "weakDem", "indDem"]), 0,
                    np.where(nes.PID.isin(["strRep", "weakRep", "indRep"]), 1, 2))
ORDER = ["MS", "HSdrop", "HS", "Coll", "CCdeg", "BAdeg", "MAdeg"]
nes["educ_num"] = nes.educ.map({k: i for i, k in enumerate(ORDER)})


def style(ax, grid="both"):
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- mapa
def mapa():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    W, H = 2.5, 1.0

    def node(x, y, t, c, fut=False, w=W, h=H):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white" if fut else c, ec=c, lw=1.7, ls="--" if fut else "-"))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.3, fontweight="bold",
                color=c if fut else "white")

    def arr(p1, p2, lab=None, dashed=False, off=(0, 0.18)):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=MUTED,
                                     lw=1.3, ls="--" if dashed else "-"))
        if lab:
            ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], lab, ha="center",
                    fontsize=8, color=MUTED, style="italic",
                    bbox=dict(fc="white", ec="none", pad=0.4))

    node(1.4, 4.0, "Regresión lineal\n(Clases 17–19)", PRE)
    node(4.6, 4.0, "GLM\n3 componentes", C20, h=1.2)
    arr((2.65, 4.0), (3.35, 4.0))
    ax.text(3.0, 4.75, "Y no es normal", fontsize=8, color=MUTED, style="italic", ha="center")
    node(8.1, 6.6, "Logística\nY binaria · logit", BLUE)
    node(8.1, 4.6, "Poisson\nY conteo · log", GREEN)
    node(8.1, 2.6, "Multinomial\nY con 3+ categorías", PURPLE)
    node(8.1, 0.75, "Lineal clásica\nY normal · identidad", PRE, w=2.5, h=0.9)
    for y, lab in [(6.6, "0/1"), (4.6, "0, 1, 2…"), (2.6, "A/B/C"), (0.75, "continua")]:
        arr((5.85, 4.0), (6.85, y))
    node(11.6, 6.6, "Métricas de\nclasificación", BLUE)
    arr((9.35, 6.6), (10.35, 6.6))
    node(11.6, 4.6, "Offset → tasas\nsobredispersión", GREEN)
    arr((9.35, 4.6), (10.35, 4.6))
    node(11.6, 2.6, "ML clasificación\nNaive Bayes (C40)", AMBER, fut=True)
    arr((9.35, 2.6), (10.35, 2.6), dashed=True)
    node(11.6, 0.75, "A/B testing\n(Clase 22)", AMBER, fut=True, h=0.9)
    ax.text(11.6, 0.05, "usa odds y proporciones", ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "glm_mapa.png")


# ----------------------------------------------------------------- por qué
def por_que():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, 160)
    p = 1 / (1 + np.exp(-(x - 5) * 1.3))
    y = rng.binomial(1, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    a1.scatter(x, y + rng.uniform(-0.03, 0.03, len(y)), s=12, color=INK, alpha=0.5)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(-1, 11, 100)
    a1.plot(xs, b0 + b1 * xs, color=RED, lw=2, label="recta (regresión lineal)")
    a1.plot(xs, 1 / (1 + np.exp(-(xs - 5) * 1.3)), color=BLUE, lw=2.2, label="curva S (logística)")
    a1.axhspan(1, 1.4, color=RED, alpha=0.08)
    a1.axhspan(-0.4, 0, color=RED, alpha=0.08)
    a1.text(-0.8, 1.22, "probabilidad > 1: imposible", fontsize=8, color=RED)
    a1.text(5.5, -0.28, "probabilidad < 0: imposible", fontsize=8, color=RED)
    a1.set_ylim(-0.4, 1.4)
    a1.set_xlabel("X")
    a1.set_ylabel("P(Y = 1)")
    a1.set_title("Ilustrativo: la recta se sale de [0, 1]", loc="left", fontsize=10.3, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="center right")
    style(a1)

    m = smf.glm("chd01 ~ cigs", w, family=sm.families.Binomial()).fit()
    bins = [-1, 0, 10, 20, 30, 40, 100]
    g = w.groupby(pd.cut(w.cigs, bins), observed=True).agg(p=("chd01", "mean"), c=("cigs", "mean"), n=("chd01", "size"))
    a2.scatter(g.c, g.p, s=g.n / 6, color=BLUE, alpha=0.6, label="% real por grupo (tamaño = n)")
    xs = np.linspace(0, 60, 100)
    a2.plot(xs, m.predict(pd.DataFrame({"cigs": xs})), color=BLUE, lw=2, label="logística chd ~ cigs")
    a2.set_xlabel("Cigarrillos diarios")
    a2.set_ylabel("P(enfermedad coronaria)")
    a2.set_title("wcgs: la probabilidad sube con cada cigarrillo", loc="left", fontsize=10.3, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_ylim(0, 0.2)
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "glm_porque.png")


# ----------------------------------------------------------------- odds
def odds():
    p = np.linspace(0.005, 0.995, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.2))
    a1.plot(p, p / (1 - p), color=AMBER, lw=2.2)
    a1.set_ylim(0, 10)
    a1.set_xlabel("Probabilidad p")
    a1.set_ylabel("Odds = p / (1 − p)")
    for pp in (0.2, 0.5, 0.8):
        o = pp / (1 - pp)
        a1.plot(pp, o, "o", color=INK)
        a1.text(pp - 0.03, o + 0.9, f"p = {pp} → odds = {o:.2f}", fontsize=8.3, color=INK, ha="right" if pp < 0.8 else "left")
    a1.set_title("Odds: de 0 a infinito", loc="left", fontsize=10.3, color=INK)
    style(a1)
    a2.plot(p, np.log(p / (1 - p)), color=BLUE, lw=2.2)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    a2.text(0.52, -4.2, "p = 0.5 ↔ logit = 0", fontsize=8.3, color=MUTED)
    a2.set_xlabel("Probabilidad p")
    a2.set_ylabel("logit = log(odds)")
    a2.set_title("Logit: de −∞ a +∞ (ahí vive la recta)", loc="left", fontsize=10.3, color=INK)
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "glm_odds.png")


# ----------------------------------------------------------------- logística
def logistica():
    m = smf.glm("chd01 ~ cigs + age", w, family=sm.families.Binomial()).fit()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    xs = np.linspace(0, 60, 100)
    for age, c in [(40, "#93C5FD"), (50, BLUE), (59, "#1E3A8A")]:
        ax.plot(xs, m.predict(pd.DataFrame({"cigs": xs, "age": age})), color=c, lw=2.2, label=f"{age} años")
    ax.set_xlabel("Cigarrillos diarios")
    ax.set_ylabel("P(enfermedad coronaria)")
    ax.set_title("Modelo chd ~ cigs + age: probabilidad estimada según cigarrillos y edad",
                 loc="left", fontsize=10.3, color=INK)
    ax.legend(frameon=False, fontsize=8.5, title="Edad", title_fontsize=8.5)
    style(ax)
    save(fig, "glm_logistica.png")
    return m


# ----------------------------------------------------------------- clasificación
def clasificacion():
    m = smf.glm("chd01 ~ cigs + age", w, family=sm.families.Binomial()).fit()
    p = m.predict(w)
    fpr, tpr, thr = roc_curve(w.chd01, p)
    auc = roc_auc_score(w.chd01, p)
    k = np.argmax(tpr - fpr)
    t_y = thr[k]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    out = {}
    for ax, t, title in [(axes[0], 0.5, "Umbral 0.5"), (axes[1], t_y, f"Umbral {t_y:.3f} (Youden)")]:
        cm = confusion_matrix(w.chd01, (p >= t).astype(int), labels=[0, 1])
        out[title] = cm
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["pred. 0", "pred. 1"])
        ax.set_yticks([0, 1], ["real 0", "real 1"])
        sens = cm[1, 1] / cm[1].sum()
        esp = cm[0, 0] / cm[0].sum()
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{title}\nexact. {acc:.0%} · sens. {sens:.0%} · esp. {esp:.0%}", fontsize=9.3, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    a = axes[2]
    a.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"modelo (AUC = {auc:.2f})")
    a.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="azar (AUC = 0.5)")
    a.plot(fpr[k], tpr[k], "o", color=RED, ms=7, label="punto de Youden")
    a.set_xlabel("1 − especificidad (falsos positivos)")
    a.set_ylabel("Sensibilidad")
    a.set_title("Curva ROC", fontsize=9.6, color=INK)
    a.legend(frameon=False, fontsize=7.8, loc="lower right")
    style(a)
    fig.tight_layout(w_pad=2.5)
    save(fig, "glm_clasif.png")
    return auc, t_y, out


# ----------------------------------------------------------------- poisson
def poisson():
    c = cancer.copy()
    c["tasa"] = 1000 * c.casos / c.poblacion
    order = ["40-54", "55-59", "60-64", "65-69", "70-74", "Over 74"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    cols = {"Fredericia": RED, "Horsens": BLUE, "Kolding": GREEN, "Vejle": AMBER}
    for city, g in c.groupby("ciudad"):
        g = g.set_index("edad").loc[order]
        ax.plot(order, g.tasa, marker="o", lw=2, color=cols[city], label=city)
    ax.set_ylabel("Casos por 1000 habitantes")
    ax.set_xlabel("Grupo de edad")
    ax.set_title("Cáncer de pulmón: la tasa sube con la edad; las ciudades se parecen más entre sí",
                 loc="left", fontsize=10.3, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=4)
    style(ax)
    save(fig, "glm_poisson.png")


# ----------------------------------------------------------------- multinomial
def multinomial():
    m = smf.mnlogit("y ~ age + educ_num", nes).fit(disp=0)
    grid = pd.DataFrame({"age": 45, "educ_num": np.arange(7)})
    pr = m.predict(grid)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.stackplot(range(7), pr[0], pr[1], pr[2], colors=[BLUE, RED, "#9CA3AF"],
                 labels=["Demócrata (referencia)", "Republicano", "Independiente"], alpha=0.9)
    ax.set_xticks(range(7), ORDER)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nivel educacional (de menor a mayor)")
    ax.set_ylabel("Probabilidad")
    ax.set_title("nes96 · persona de 45 años: las 3 probabilidades suman 1 en cada nivel",
                 loc="left", fontsize=10.3, color=INK)
    ax.text(3, pr[0][3] / 2, "Demócrata (referencia)", ha="center", color="white", fontsize=10, fontweight="bold")
    ax.text(3, pr[0][3] + pr[1][3] / 2, "Republicano", ha="center", color="white", fontsize=10, fontweight="bold")
    ax.annotate("Independiente (~4%)", xy=(5.5, 0.985), xytext=(4.6, 1.08), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED), annotation_clip=False)
    save(fig, "glm_multinom.png")
    return pr.round(3)


if __name__ == "__main__":
    mapa()
    por_que()
    odds()
    m = logistica()
    print("cigs+age", m.params.round(5).to_dict())
    print("clasif", clasificacion())
    poisson()
    print("multinom grid\n", multinomial())
