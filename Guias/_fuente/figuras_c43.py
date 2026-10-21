"""Figuras de la Clase 43 (Evaluación estadística de modelos predictivos y bootstrap).
Fuentes: diapositivas clase.pdf, evaluacion_bootstrap.py / .R.txt del profesor Pedro Luiz Ramos y wisconsin.csv (569 muestras; 398 de entrenamiento con 5 pliegues y
171 de prueba; es la misma partición de wisconsin_clase.csv de la Clase 37, con las columnas extra id y fold). La carpeta de la clase se lee sin modificarla.
Todo sale de código; lo aleatorio usa las semillas del profesor (20261012, 20261013, 20261014, 20261017) o semillas fijas propias (43xxx).
Escribe c43_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c43.py   (o con nombres: python -X utf8 figuras_c43.py media auc)."""
import json
import warnings
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C43 = "#0E7490"      # color de la clase 43 (cian oscuro)
C43_BG = "#ECFEFF"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
TEAL = "#0D9488"
VIOLET = "#7C3AED"
GREEN = "#16A34A"
B = 2000
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA43 = next(p for p in BASE.glob("Clase 43*") if p.is_dir())
CARPETA37 = next(p for p in BASE.glob("Clase 37*") if p.is_dir())
_C = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def q(x, axis=None):
    return np.quantile(x, [0.025, 0.975], axis=axis)


# ----------------------------------------------------------------------------- datos y cálculos comunes (misma lógica que evaluacion_bootstrap.py)
def calc():
    if _C:
        return _C
    d = pd.read_csv(CARPETA43 / "wisconsin.csv")
    tr = d[d["conjunto"] == "entrenamiento"].copy()
    te = d[d["conjunto"] == "prueba"].copy()
    f_s, f_i, f_a = "maligno ~ radio + textura", "maligno ~ radio + textura + suavidad", "maligno ~ radio + textura + suavidad + concavidad"
    ms, mi, ma = (smf.glm(f, data=tr, family=sm.families.Binomial()).fit() for f in (f_s, f_i, f_a))
    y = te["maligno"].to_numpy()
    ps, pa = ms.predict(te).to_numpy(), ma.predict(te).to_numpy()
    _C.update(d=d, tr=tr, te=te, ms=ms, mi=mi, ma=ma, y=y, ps=ps, pa=pa, formulas=(f_s, f_i, f_a))

    # --- bootstrap del profesor (mismas semillas y mismo orden de llamadas que su script)
    valores = np.array([2, 4, 5, 7, 8])
    rng = np.random.default_rng(20261012)
    medias = np.empty(B)
    for b in range(B):
        medias[b] = np.mean(rng.choice(valores, size=len(valores), replace=True))
    n = len(y)
    rng = np.random.default_rng(20261013)
    ll_boot = np.empty((B, 2))
    for b in range(B):
        i = rng.choice(n, size=n, replace=True)
        ll_boot[b] = log_loss(y[i], ps[i]), log_loss(y[i], pa[i])
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(20261017)
    auc_boot = np.empty((B, 2))
    for b in range(B):
        i = np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])
        auc_boot[b] = roc_auc_score(y[i], ps[i]), roc_auc_score(y[i], pa[i])
    obs = np.array([10, 20, 30, 40, 50])
    pre = np.array([9, 22, 27, 44, 45])
    rng = np.random.default_rng(20261014)
    reg_boot = np.empty((B, 2))
    for b in range(B):
        i = rng.choice(5, size=5, replace=True)
        e = obs[i] - pre[i]
        reg_boot[b] = np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))
    _C.update(valores=valores, medias=medias, ll_boot=ll_boot, auc_boot=auc_boot, reg_boot=reg_boot, obs=obs, pre=pre, neg=neg, pos=pos)
    return _C


def delong(y, p1, p2):
    """Prueba de DeLong para dos AUC correlacionadas (mismos casos)."""
    pos, neg = y == 1, y == 0
    m, n_ = pos.sum(), neg.sum()
    v10, v01, auc = [], [], []
    for p in (p1, p2):
        a, b = p[pos][:, None], p[neg][None, :]
        psi = (a > b) + 0.5 * (a == b)
        v10.append(psi.mean(1))
        v01.append(psi.mean(0))
        auc.append(psi.mean())
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_
    z = (auc[1] - auc[0]) / np.sqrt(var)
    return dict(auc1=float(auc[0]), auc2=float(auc[1]), diff=float(auc[1] - auc[0]), se=float(np.sqrt(var)), z=float(z), p=float(2 * stats.norm.sf(abs(z))))


# ----------------------------------------------------------------------------- 1. el número de la prueba y su incertidumbre
def incertidumbre_fig():
    c = calc()
    rng = np.random.default_rng(43001)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    bins = np.linspace(0.75, 1.0, 26)
    sims = {}
    for nn, col in ((100, C43), (1000, AMBER)):
        a = rng.binomial(nn, 0.9, 20000) / nn
        sims[nn] = a
        ax.hist(a, bins=bins, density=False, weights=np.full(len(a), 1 / len(a)), alpha=0.65, color=col, label=f"n = {nn}")
    ax.axvline(0.9, color=INK, ls="--", lw=1)
    ax.set(xlabel="exactitud observada en una muestra de prueba", ylabel="frecuencia relativa")
    ax.legend(frameon=False, fontsize=8.5)
    titulo(ax, "Mismo modelo (acierto real 0.90), distintas pruebas")
    style(ax)
    ax = axs[1]
    nn = np.array([100, 1000])
    se = np.sqrt(0.9 * 0.1 / nn)
    lo, hi = 0.9 - 1.96 * se, 0.9 + 1.96 * se
    for i, (a, b_, s) in enumerate(zip(lo, hi, se)):
        ax.plot([a, b_], [i, i], color=[C43, AMBER][i], lw=6, solid_capstyle="round")
        ax.text(0.9, i + 0.2, f"SE = {s:.4f}   [{a:.3f}, {b_:.3f}]", ha="center", fontsize=8.5)
    ax.axvline(0.9, color=INK, ls="--", lw=1)
    ax.set(yticks=[0, 1], yticklabels=["90 de 100", "900 de 1 000"], xlim=(0.8, 1.0), ylim=(-0.5, 1.6), xlabel="intervalo normal aproximado para la exactitud")
    titulo(ax, "El mismo 90 % con distinta precisión")
    style(ax, "x")
    ax = axs[2]
    r2 = np.random.default_rng(43002)
    cubre = 0
    for e in range(40):
        x = r2.normal(0, 1, 30)
        m_ = x.mean()
        h = 1.96 / np.sqrt(30)
        ok = (m_ - h <= 0 <= m_ + h)
        cubre += ok
        ax.plot([m_ - h, m_ + h], [e, e], color=GREEN if ok else RED, lw=1.8)
        ax.plot(m_, e, "o", color=GREEN if ok else RED, ms=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set(xlabel="intervalo para la media (n = 30, σ conocida)", ylabel="experimento")
    ax.text(0.02, 0.02, f"{cubre} de 40 contienen la media verdadera", transform=ax.transAxes, fontsize=8.5, color=INK, bbox=dict(fc="white", ec=GRID))
    titulo(ax, "Un IC al 95 %: cobertura en repetición")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c43_incertidumbre.png")
    NUM["incert"] = dict(sd_n100=float(sims[100].std()), sd_n1000=float(sims[1000].std()), se=[float(s) for s in se], ic=[[float(a), float(b_)] for a, b_ in zip(lo, hi)], cubre=int(cubre))


# ----------------------------------------------------------------------------- 2. bootstrap: esquema de remuestreo y media
def esquema_fig():
    c = calc()
    v = c["valores"]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.6, 4.8)
    ax.axis("off")

    def fila(x0, y0, ids, vals, titulo_, color, resaltar=None, media=None):
        ax.text(x0, y0 + 0.62, titulo_, fontsize=8.8, fontweight="bold", color=color, ha="left")
        for k, (i, val) in enumerate(zip(ids, vals)):
            gris = resaltar is not None and i not in resaltar
            ax.add_patch(Rectangle((x0 + k * 0.78, y0), 0.7, 0.55, fc="#F3F4F6" if gris else "white", ec=color, lw=1.4))
            ax.text(x0 + k * 0.78 + 0.35, y0 + 0.37, f"ID {i}", ha="center", fontsize=6.8, color=MUTED)
            ax.text(x0 + k * 0.78 + 0.35, y0 + 0.14, f"{val}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
        if media is not None:
            ax.text(x0 + 3.95, y0 + 0.27, f"media = {media:.1f}", fontsize=9, color=color, va="center")

    fila(0.2, 3.9, [1, 2, 3, 4, 5], list(v), "Muestra original (n = 5)", INK, media=v.mean())
    reps = [([2, 2, 1, 5, 3], C43), ([5, 5, 4, 3, 5], AMBER), ([1, 1, 2, 3, 4], VIOLET)]
    medias = []
    for r, (ids, col) in enumerate(reps):
        vals = [int(v[i - 1]) for i in ids]
        medias.append(float(np.mean(vals)))
        fila(0.2, 2.5 - r * 1.2, ids, vals, f"Remuestra {r + 1}: se sortean 5 IDs con reposición", col, resaltar=set(ids), media=medias[-1])
    # histograma de las 3 medias dibujado a mano + texto
    ax2 = fig.add_axes([0.66, 0.14, 0.31, 0.70])
    ax2.hist(c["medias"], bins=np.arange(1.9, 8.2, 0.2), color=C43, alpha=0.8)
    ax2.axvline(v.mean(), color=RED, lw=1.8)
    lo, hi = q(c["medias"])
    for x in (lo, hi):
        ax2.axvline(x, color=INK, ls="--", lw=1.2)
    ax2.set(xlabel="media de la remuestra", ylabel="N.º de remuestras")
    titulo(ax2, f"B = {B} remuestras → IC percentil [{lo:.1f}; {hi:.1f}]", fs=8.8)
    style(ax2, "y")
    ax.text(0.2, -0.45, "Cada remuestra tiene el mismo tamaño que la muestra; un ID puede repetirse y otro quedar fuera (ID 4 en la remuestra 1).", fontsize=8.3, color=MUTED)
    save(fig, "c43_esquema_bootstrap.png")
    NUM["esquema"] = dict(medias_ejemplo=medias)


def media_fig():
    c = calc()
    v, m = c["valores"], c["medias"]
    # distribución exacta del bootstrap: 5^5 = 3 125 remuestras ordenadas
    todas = np.array([np.mean(np.array(t)) for t in product(v, repeat=5)])
    se_exacto = float(todas.std())               # varianza poblacional del bootstrap
    ic_exacto = q(todas)
    por_B = {}
    for bb in (100, 500, 2000):
        por_B[bb] = dict(se=float(np.std(m[:bb], ddof=1)), ic=[float(x) for x in q(m[:bb])])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    ax.hist(m, bins=np.arange(1.9, 8.2, 0.2), color=C43, alpha=0.85)
    ax.axvline(v.mean(), color=RED, lw=1.8, label="media original 5.2")
    for x in q(m):
        ax.axvline(x, color=INK, ls="--", lw=1.2)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set(xlabel="media de la remuestra", ylabel="N.º de remuestras")
    titulo(ax, f"Bootstrap de la media (B = {B}, semilla 20261012)")
    style(ax, "y")
    ax = axs[1]
    vals, cnt = np.unique(np.round(todas, 1), return_counts=True)
    ax.bar(vals, cnt / cnt.sum(), width=0.17, color=AMBER, alpha=0.9)
    for x in ic_exacto:
        ax.axvline(x, color=INK, ls="--", lw=1.2)
    ax.axvline(v.mean(), color=RED, lw=1.8)
    ax.set(xlabel="media de la remuestra", ylabel="probabilidad exacta")
    titulo(ax, "Las 3 125 remuestras posibles (distribución exacta)")
    style(ax, "y")
    ax = axs[2]
    bs = [100, 500, 2000]
    ax.errorbar(range(3), [np.mean(q(m[:bb])) for bb in bs], yerr=[[np.mean(q(m[:bb])) - q(m[:bb])[0] for bb in bs], [q(m[:bb])[1] - np.mean(q(m[:bb])) for bb in bs]], fmt="o", color=C43, capsize=5, lw=2)
    ax.axhline(5.2, color=RED, lw=1.2)
    ax.axhspan(ic_exacto[0], ic_exacto[1], color=AMBER, alpha=0.15)
    for i, bb in enumerate(bs):
        ax.text(i, 7.9, f"SE {por_B[bb]['se']:.3f}", ha="center", fontsize=8.5)
    ax.set(xticks=range(3), xticklabels=[f"B = {bb}" for bb in bs], ylim=(2, 8.3), ylabel="IC percentil de la media")
    titulo(ax, "Más remuestras ≠ más datos: el IC deja de cambiar")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c43_media.png")
    NUM["media"] = dict(media=float(v.mean()), se=float(np.std(m, ddof=1)), ic=[float(x) for x in q(m)], por_B={str(k): x for k, x in por_B.items()},
                        se_exacto=se_exacto, ic_exacto=[float(x) for x in ic_exacto], se_formula=float(np.sqrt(np.sum((v - v.mean()) ** 2) / 5) / np.sqrt(5)),
                        slide_R=dict(se=0.966, ic=[3.4, 7.0]), n_remuestras=int(len(todas)))


def oob_fig():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axs[0]
    nn = np.arange(2, 201)
    ax.plot(nn, (1 - 1 / nn) ** nn, color=C43, lw=2, label="P(un caso queda fuera) = (1 − 1/n)ⁿ")
    ax.axhline(np.exp(-1), color=RED, ls="--", lw=1.2)
    ax.text(150, np.exp(-1) + 0.02, "1/e ≈ 0.368", ha="center", color=RED, fontsize=8.5)
    for k in (5, 171):
        ax.scatter([k], [(1 - 1 / k) ** k], color=AMBER, zorder=3, s=45)
        ax.annotate(f"n = {k}: {(1 - 1 / k) ** k:.3f}", (k, (1 - 1 / k) ** k), xytext=(k + 10, (1 - 1 / k) ** k + 0.07), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set(xlabel="tamaño de la muestra n", ylabel="probabilidad", ylim=(0.2, 0.6))
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    titulo(ax, "¿Cuánto queda fuera de una remuestra?")
    style(ax)
    ax = axs[1]
    rng = np.random.default_rng(43003)
    n = 171
    dentro = np.array([len(np.unique(rng.integers(0, n, n))) / n for _ in range(2000)])
    ax.hist(dentro, bins=25, color=C43, alpha=0.85)
    ax.axvline(1 - (1 - 1 / n) ** n, color=RED, lw=1.8)
    ax.set(xlabel="fracción de casos distintos en la remuestra", ylabel="N.º de remuestras")
    titulo(ax, "Los 171 casos de prueba: ≈ 63.2 % entran, ≈ 36.8 % no")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c43_oob.png")
    NUM["oob"] = dict(p5=float((1 - 1 / 5) ** 5), p171=float((1 - 1 / 171) ** 171), limite=float(np.exp(-1)), dentro_media=float(dentro.mean()), dentro_sd=float(dentro.std()), distintos5=float(5 * (1 - (1 - 1 / 5) ** 5)))


# ----------------------------------------------------------------------------- 3. selección (AIC, BIC, OOF) frente a evaluación
def seleccion_fig():
    c = calc()
    tr, te, ms, mi, ma = c["tr"], c["te"], c["ms"], c["mi"], c["ma"]
    nombres = ["Simple", "Intermedio", "Amplio"]
    mods = [ms, mi, ma]
    crit = [dict(modelo=nm, p=int(len(m.params)), llf=float(m.llf), aic=float(m.aic), bic=float(m.bic_llf)) for nm, m in zip(nombres, mods)]
    # OOF dentro del entrenamiento con los 5 pliegues del profesor (cada modelo se reajusta sin su pliegue)
    oof = {}
    for nm, f in zip(nombres, c["formulas"]):
        p = np.zeros(len(tr))
        for k in sorted(tr.fold.unique()):
            v = (tr.fold == k).to_numpy()
            m = smf.glm(f, data=tr[~v], family=sm.families.Binomial()).fit()
            p[v] = m.predict(tr[v]).to_numpy()
        yt = tr["maligno"].to_numpy()
        oof[nm] = dict(log_loss=float(log_loss(yt, p)), auc=float(roc_auc_score(yt, p)), accuracy=float(np.mean((p >= 0.5) == yt)))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(Rectangle((0.2, 4.6), 9.6, 0.9, fc="white", ec=INK, lw=1.4))
    ax.text(5, 5.05, "569 muestras de Wisconsin", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.add_patch(Rectangle((0.2, 3.0), 6.7, 0.9, fc=C43_BG, ec=C43, lw=1.6))
    ax.text(3.55, 3.45, "398 entrenamiento", ha="center", va="center", fontsize=9, color=C43, fontweight="bold")
    ax.add_patch(Rectangle((7.1, 3.0), 2.7, 0.9, fc="#FEF3C7", ec=AMBER, lw=1.6))
    ax.text(8.45, 3.45, "171 prueba", ha="center", va="center", fontsize=9, color=AMBER, fontweight="bold")
    tam = tr.fold.value_counts().sort_index().tolist()
    for k, t in enumerate(tam):
        x0 = 0.2 + k * 1.34
        ax.add_patch(Rectangle((x0, 1.3), 1.34, 0.8, fc="#FDE68A" if k == 0 else "white", ec=C43, lw=1.1))
        ax.text(x0 + 0.67, 1.7, f"pliegue {k + 1}\n{t}", ha="center", va="center", fontsize=7.4)
    ax.text(0.2, 0.7, "AIC/BIC y OOF eligen la receta usando solo el entrenamiento;", fontsize=7.8, color=INK)
    ax.text(0.2, 0.25, "la prueba se mira al final, sin volver a elegir.", fontsize=7.8, color=INK)
    titulo(ax, "Seleccionar con entrenamiento, evaluar en prueba")
    ax = axs[1]
    x = np.arange(3)
    ax.bar(x - 0.2, [r["aic"] for r in crit], width=0.38, color=C43, label="AIC")
    ax.bar(x + 0.2, [r["bic"] for r in crit], width=0.38, color=AMBER, label="BIC")
    for i, r in enumerate(crit):
        ax.text(i - 0.2, r["aic"] + 3, f"{r['aic']:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r["bic"] + 3, f"{r['bic']:.1f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=[f"{r['modelo']}\np = {r['p']}" for r in crit], ylabel="menor es mejor", ylim=(0, 245))
    ax.legend(frameon=False, fontsize=8.5)
    titulo(ax, "AIC y BIC en entrenamiento (398 casos)")
    style(ax, "y")
    ax = axs[2]
    ax.bar(x - 0.2, [oof[n]["log_loss"] for n in nombres], width=0.38, color=C43, label="log loss OOF")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, [oof[n]["auc"] for n in nombres], "o-", color=RED, label="AUC OOF")
    ax2.set_ylim(0.9, 1.0)
    ax2.set_ylabel("AUC OOF", color=RED)
    for i, n in enumerate(nombres):
        ax.text(i - 0.2, oof[n]["log_loss"] + 0.008, f"{oof[n]['log_loss']:.3f}", ha="center", fontsize=8)
        ax2.text(i + 0.2, oof[n]["auc"] + 0.003, f"{oof[n]['auc']:.3f}", ha="center", fontsize=8, color=RED)
    ax.set(xticks=x, xticklabels=nombres, ylabel="log loss OOF (menor es mejor)")
    ax2.spines["right"].set_visible(True)
    titulo(ax, "Validación cruzada de 5 pliegues (OOF)")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c43_seleccion.png")
    NUM["criterios"] = crit
    NUM["oof"] = oof
    NUM["pliegues"] = [int(t) for t in tam]


# ----------------------------------------------------------------------------- 4. métricas de un modelo fijo: log loss, AUC, exactitud, sensibilidad
def metricas_prueba():
    c = calc()
    y, ps, pa = c["y"], c["ps"], c["pa"]
    comp = {}
    for nm, p in (("simple", ps), ("amplio", pa)):
        pred = p >= 0.5
        comp[nm] = dict(auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p)), exactitud=float(np.mean(pred == y)), sensibilidad=float(np.mean(pred[y == 1])),
                        especificidad=float(np.mean(~pred[y == 0])), aciertos=int(np.sum(pred == y)), positivos_detectados=int(np.sum(pred[y == 1])))
    comp["n"] = int(len(y))
    comp["n_neg"], comp["n_pos"] = int((y == 0).sum()), int((y == 1).sum())
    NUM["prueba"] = comp
    return comp


def auc_fig():
    c = calc()
    comp = metricas_prueba()
    ll, au = c["ll_boot"], c["auc_boot"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, dat, nm, col, ttl, xl in ((axs[0], ll[:, 0], "simple", BLUE, "Log loss: modelo simple", "log loss de la remuestra"), (axs[1], ll[:, 1], "amplio", C43, "Log loss: modelo amplio", "log loss de la remuestra"),
                                      (axs[2], au[:, 0], "simple", BLUE, "AUC: modelo simple", "AUC de la remuestra"), (axs[3], au[:, 1], "amplio", C43, "AUC: modelo amplio", "AUC de la remuestra")):
        ax.hist(dat, bins=25, color=col, alpha=0.8)
        est = comp[nm]["log_loss"] if "Log" in ttl else comp[nm]["auc"]
        ax.axvline(est, color=RED, lw=1.8)
        for x in q(dat):
            ax.axvline(x, color=INK, ls="--", lw=1.2)
        lo, hi = q(dat)
        ax.set(xlabel=xl, ylabel="N.º de remuestras")
        titulo(ax, f"{ttl}\n{est:.4f}  IC [{lo:.4f}; {hi:.4f}]", fs=8.6)
        style(ax, "y")
    fig.tight_layout()
    save(fig, "c43_auc_logloss.png")
    NUM["ic"] = dict(ll_simple=[float(x) for x in q(ll[:, 0])], ll_amplio=[float(x) for x in q(ll[:, 1])], auc_simple=[float(x) for x in q(au[:, 0])], auc_amplio=[float(x) for x in q(au[:, 1])],
                     se_ll=[float(ll[:, 0].std(ddof=1)), float(ll[:, 1].std(ddof=1))], se_auc=[float(au[:, 0].std(ddof=1)), float(au[:, 1].std(ddof=1))],
                     slide_R=dict(ll_simple=[0.2124, 0.4005], ll_amplio=[0.1066, 0.2798], auc_simple=[0.9004, 0.9708], auc_amplio=[0.9503, 0.9915]))


def intervalos_fig():
    """Exactitud y sensibilidad del modelo amplio: normal, Wilson, exacto, bayesiano y bootstrap (puente con C22 y C23)."""
    c = calc()
    comp = metricas_prueba()
    y, pa = c["y"], c["pa"]
    n = len(y)
    pred = pa >= 0.5
    rng = np.random.default_rng(43004)
    acc_b = np.empty(B)
    sen_b = np.empty(B)
    for b in range(B):
        i = rng.choice(n, size=n, replace=True)
        acc_b[b] = np.mean(pred[i] == y[i])
        yi, pi_ = y[i], pred[i]
        sen_b[b] = np.mean(pi_[yi == 1])
    res = {}
    for nm, k, nn, bt in (("exactitud", comp["amplio"]["aciertos"], n, acc_b), ("sensibilidad", comp["amplio"]["positivos_detectados"], comp["n_pos"], sen_b)):
        ph = k / nn
        met = {}
        for lab, mt in (("Normal (Wald)", "normal"), ("Wilson", "wilson"), ("Exacto (Clopper-Pearson)", "beta")):
            lo, hi = proportion_confint(k, nn, alpha=0.05, method=mt)
            met[lab] = [float(lo), float(min(hi, 1.0))]
        met["Bayes (Beta(1 + k, 1 + n − k))"] = [float(x) for x in stats.beta.ppf([0.025, 0.975], 1 + k, 1 + nn - k)]
        met["Bootstrap percentil"] = [float(x) for x in q(bt)]
        res[nm] = dict(k=int(k), n=int(nn), est=float(ph), se_wald=float(np.sqrt(ph * (1 - ph) / nn)), se_boot=float(bt.std(ddof=1)), metodos=met)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.7))
    cols = [BLUE, TEAL, VIOLET, AMBER, C43]
    for ax, nm, ttl in ((axs[0], "exactitud", "Exactitud del modelo amplio (161 de 171)"), (axs[1], "sensibilidad", "Sensibilidad del modelo amplio (58 de 64)")):
        r = res[nm]
        for i, (lab, (lo, hi)) in enumerate(r["metodos"].items()):
            ax.plot([lo, hi], [i, i], color=cols[i], lw=6, solid_capstyle="round")
            ax.text(hi + 0.004, i, f"[{lo:.3f}; {hi:.3f}]", va="center", fontsize=8)
        ax.axvline(r["est"], color=INK, ls="--", lw=1)
        ax.set(yticks=range(5), yticklabels=list(r["metodos"]), xlim=(min(v[0] for v in r["metodos"].values()) - 0.03, 1.06), xlabel="intervalo de confianza o creíble 95 %")
        ax.invert_yaxis()
        titulo(ax, ttl)
        style(ax, "x")
    fig.tight_layout()
    save(fig, "c43_intervalos_prop.png")
    NUM["prop"] = res


def roc_fig():
    c = calc()
    y, ps, pa, neg, pos = c["y"], c["ps"], c["pa"], c["neg"], c["pos"]
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(43005)
    Bb = 500
    tp = {"simple": np.empty((Bb, 101)), "amplio": np.empty((Bb, 101))}
    for b in range(Bb):
        i = np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])
        for nm, p in (("simple", ps), ("amplio", pa)):
            f, t, _ = roc_curve(y[i], p[i])
            tp[nm][b] = np.interp(grid, f, t)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axs[0]
    for nm, p, col in (("simple", ps, BLUE), ("amplio", pa, C43)):
        lo, hi = q(tp[nm], axis=0)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.2)
        f, t, _ = roc_curve(y, p)
        ax.plot(f, t, color=col, lw=2, label=f"{nm}: AUC {roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], ":", color=MUTED)
    ax.set(xlabel="1 − especificidad", ylabel="sensibilidad", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    titulo(ax, "Curva ROC y banda bootstrap 95 % (500 remuestras por clase)")
    style(ax)
    ax = axs[1]
    for nm, p, col in (("simple", ps, BLUE), ("amplio", pa, C43)):
        lo, hi = q(tp[nm], axis=0)
        f, t, _ = roc_curve(y, p)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.2)
        ax.plot(f, t, color=col, lw=2)
    ax.set(xlim=(0, 0.3), ylim=(0.55, 1.02), xlabel="1 − especificidad", ylabel="sensibilidad")
    titulo(ax, "Zona de interés: poca tasa de falsos positivos")
    style(ax)
    fig.tight_layout()
    save(fig, "c43_roc_banda.png")
    anch = {nm: float(np.mean(q(tp[nm], axis=0)[1] - q(tp[nm], axis=0)[0])) for nm in tp}
    NUM["roc"] = dict(B=Bb, ancho_medio=anch)


# ----------------------------------------------------------------------------- 5. comparación pareada
def pareado_fig():
    c = calc()
    comp = NUM.get("prueba") or metricas_prueba()
    ll, au = c["ll_boot"], c["auc_boot"]
    d_ll, d_au = ll[:, 0] - ll[:, 1], au[:, 1] - au[:, 0]
    dif_ll = comp["simple"]["log_loss"] - comp["amplio"]["log_loss"]
    dif_au = comp["amplio"]["auc"] - comp["simple"]["auc"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, d, est, ttl, xl in ((axs[0], d_ll, dif_ll, "Δ log loss = simple − amplio", "diferencia pareada de log loss"), (axs[1], d_au, dif_au, "Δ AUC = amplio − simple", "diferencia pareada de AUC")):
        ax.hist(d, bins=25, color=C43, alpha=0.8)
        ax.axvline(0, color=RED, lw=1.8, label="diferencia 0")
        ax.axvline(est, color=INK, lw=1.4, label="estimación original")
        lo, hi = q(d)
        for x in (lo, hi):
            ax.axvline(x, color=GREEN, ls="--", lw=1.4)
        ax.legend(frameon=False, fontsize=8)
        ax.set(xlabel=xl, ylabel="N.º de remuestras")
        titulo(ax, f"{ttl}\n{est:.4f}  IC [{lo:.4f}; {hi:.4f}]", fs=8.6)
        style(ax, "y")
    ax = axs[2]
    ax.scatter(au[:, 0], au[:, 1], s=6, alpha=0.35, color=VIOLET)
    lim = [min(au.min(), 0.88), 1.0]
    ax.plot(lim, lim, ":", color=MUTED)
    r = float(np.corrcoef(au[:, 0], au[:, 1])[0, 1])
    ax.text(0.03, 0.93, f"correlación = {r:.2f}\n(los mismos casos difíciles\nbajan las dos AUC)", transform=ax.transAxes, va="top", fontsize=8.3, bbox=dict(fc="white", ec=GRID))
    ax.set(xlabel="AUC simple en la remuestra", ylabel="AUC amplio en la remuestra")
    titulo(ax, "Por qué conviene emparejar")
    style(ax)
    fig.tight_layout()
    save(fig, "c43_pareado.png")
    NUM["dif"] = dict(ll=dif_ll, ic_ll=[float(x) for x in q(d_ll)], auc=dif_au, ic_auc=[float(x) for x in q(d_au)], corr_auc=r, corr_ll=float(np.corrcoef(ll[:, 0], ll[:, 1])[0, 1]),
                      frac_neg_ll=float(np.mean(d_ll <= 0)), frac_neg_auc=float(np.mean(d_au <= 0)),
                      primeras3=[dict(ll_s=float(ll[b, 0]), ll_a=float(ll[b, 1]), d=float(d_ll[b])) for b in range(3)],
                      slide_R=dict(ll=[0.0515, 0.1841], auc=[0.0123, 0.0624]))


def solapamiento_fig():
    c = calc()
    y, ps, pa, neg, pos = c["y"], c["ps"], c["pa"], c["neg"], c["pos"]
    au = c["auc_boot"]
    ic_s, ic_a = q(au[:, 0]), q(au[:, 1])
    d_pair = au[:, 1] - au[:, 0]
    rng = np.random.default_rng(43006)
    d_no = np.empty(B)
    for b in range(B):
        i1 = np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])
        i2 = np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])
        d_no[b] = roc_auc_score(y[i2], pa[i2]) - roc_auc_score(y[i1], ps[i1])
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.7))
    ax = axs[0]
    for i, (nm, ic, col, est) in enumerate((("simple", ic_s, BLUE, NUM["prueba"]["simple"]["auc"]), ("amplio", ic_a, C43, NUM["prueba"]["amplio"]["auc"]))):
        ax.plot(ic, [i, i], color=col, lw=7, solid_capstyle="round")
        ax.plot([est], [i], "o", color="white", ms=5)
        ax.text(ic[0] - 0.003, i, f"{ic[0]:.4f}", ha="right", va="center", fontsize=8)
        ax.text(ic[1] + 0.003, i, f"{ic[1]:.4f}", ha="left", va="center", fontsize=8)
    lo_o, hi_o = max(ic_s[0], ic_a[0]), min(ic_s[1], ic_a[1])
    ax.axvspan(lo_o, hi_o, color=AMBER, alpha=0.2)
    ax.text((lo_o + hi_o) / 2, 0.5, "los IC se\nsolapan", ha="center", va="center", fontsize=8.5, color=AMBER, fontweight="bold")
    ax.set(yticks=[0, 1], yticklabels=["AUC simple", "AUC amplio"], xlim=(0.87, 1.015), ylim=(-0.6, 1.6), xlabel="AUC (IC percentil 95 %)")
    titulo(ax, "Dos IC separados: se solapan…")
    style(ax, "x")
    ax = axs[1]
    for i, (nm, d, col) in enumerate((("pareado (mismas remuestras)", d_pair, C43), ("no pareado (remuestras independientes)", d_no, AMBER))):
        lo, hi = q(d)
        ax.plot([lo, hi], [i, i], color=col, lw=7, solid_capstyle="round")
        ax.text(hi + 0.003, i, f"[{lo:.4f}; {hi:.4f}]  ancho {hi - lo:.4f}", va="center", fontsize=8.3)
    ax.axvline(0, color=RED, lw=1.8)
    ax.set(yticks=[0, 1], yticklabels=["ΔAUC pareado", "ΔAUC no pareado"], xlim=(-0.05, 0.15), ylim=(-0.6, 1.6), xlabel="ΔAUC = amplio − simple")
    titulo(ax, "…pero la diferencia pareada no contiene 0")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c43_solapamiento.png")
    NUM["solap"] = dict(ic_simple=[float(x) for x in ic_s], ic_amplio=[float(x) for x in ic_a], solapan=bool(ic_a[0] < ic_s[1]), solape=[float(lo_o), float(hi_o)],
                        pareado=[float(x) for x in q(d_pair)], no_pareado=[float(x) for x in q(d_no)], ancho_pareado=float(np.diff(q(d_pair))[0]), ancho_no_pareado=float(np.diff(q(d_no))[0]),
                        no_pareado_contiene_0=bool(q(d_no)[0] <= 0 <= q(d_no)[1]), se_pareado=float(d_pair.std(ddof=1)), se_no_pareado=float(d_no.std(ddof=1)))


# ----------------------------------------------------------------------------- 6. alcance del intervalo: McNemar, DeLong, re-entrenar, tamaño
def mcnemar_fig():
    c = calc()
    y, ps, pa = c["y"], c["ps"], c["pa"]
    ok_s, ok_a = (ps >= 0.5) == y, (pa >= 0.5) == y
    a = int(np.sum(ok_s & ok_a))
    b = int(np.sum(ok_s & ~ok_a))
    cc = int(np.sum(~ok_s & ok_a))
    dd = int(np.sum(~ok_s & ~ok_a))
    stat = (abs(b - cc) - 1) ** 2 / (b + cc)
    p_chi = float(stats.chi2.sf(stat, 1))
    p_ex = float(stats.binomtest(min(b, cc), b + cc, 0.5).pvalue)
    dl = delong(y, ps, pa)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.7))
    ax = axs[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.axis("off")
    cel = [[a, b], [cc, dd]]
    for i in range(2):
        for j in range(2):
            col = ["#CFFAFE", "#FEF3C7"][(i + j) % 2] if (i, j) != (0, 0) and (i, j) != (1, 1) else ("#D1FAE5" if (i, j) == (0, 0) else "#FEE2E2")
            ax.add_patch(Rectangle((2 + j * 1.9, 2.4 - i * 1.3), 1.9, 1.3, fc=col, ec=INK, lw=1.2))
            ax.text(2.95 + j * 1.9, 3.05 - i * 1.3, str(cel[i][j]), ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(2.95, 4.0, "amplio acierta", ha="center", fontsize=8.5)
    ax.text(4.85, 4.0, "amplio falla", ha="center", fontsize=8.5)
    ax.text(1.9, 3.05, "simple\nacierta", ha="right", va="center", fontsize=8.5)
    ax.text(1.9, 1.75, "simple\nfalla", ha="right", va="center", fontsize=8.5)
    ax.text(0.2, 0.9, f"Solo cuentan los casos discordantes: b = {b} y c = {cc}.", fontsize=8.5)
    ax.text(0.2, 0.4, f"χ² = (|b − c| − 1)²/(b + c) = {stat:.2f};  p = {p_chi:.5f}  (exacto: {p_ex:.5f})", fontsize=8.5)
    titulo(ax, "McNemar: tabla 2 × 2 de aciertos, umbral 0.5 (extensión)")
    ax = axs[1]
    d_au = c["auc_boot"][:, 1] - c["auc_boot"][:, 0]
    lo, hi = q(d_au)
    ax.plot([lo, hi], [0, 0], color=C43, lw=7, solid_capstyle="round")
    ax.text(hi + 0.004, 0, f"bootstrap [{lo:.4f}; {hi:.4f}]", va="center", fontsize=8.3)
    z = 1.96 * dl["se"]
    ax.plot([dl["diff"] - z, dl["diff"] + z], [1, 1], color=VIOLET, lw=7, solid_capstyle="round")
    ax.text(dl["diff"] + z + 0.004, 1, f"DeLong [{dl['diff'] - z:.4f}; {dl['diff'] + z:.4f}]", va="center", fontsize=8.3)
    ax.axvline(0, color=RED, lw=1.8)
    ax.axvline(dl["diff"], color=INK, ls="--", lw=1)
    ax.set(yticks=[0, 1], yticklabels=["bootstrap pareado", "DeLong (analítico)"], xlim=(-0.01, 0.13), ylim=(-0.6, 1.6), xlabel="ΔAUC = amplio − simple")
    titulo(ax, "Dos caminos para el mismo ΔAUC (DeLong: extensión)")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c43_mcnemar.png")
    NUM["mcnemar"] = dict(a=a, b=b, c=cc, d=dd, stat=float(stat), p_chi=p_chi, p_exacto=p_ex, n_disc=b + cc)
    NUM["delong"] = dl


def reentrenar_fig():
    c = calc()
    d = c["d"]
    X = d[["radio", "textura", "suavidad", "concavidad"]].to_numpy()
    yy = d["maligno"].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=300, test_size=171, random_state=43007)
    dauc, dll, a_s, a_a = [], [], [], []
    for itr, ite in sss.split(X, yy):
        r = []
        for cols in ([0, 1], [0, 1, 2, 3]):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, solver="newton-cg", tol=1e-10, max_iter=1000)).fit(X[itr][:, cols], yy[itr])
            p = m.predict_proba(X[ite][:, cols])[:, 1]
            r.append((roc_auc_score(yy[ite], p), log_loss(yy[ite], p)))
        dauc.append(r[1][0] - r[0][0])
        dll.append(r[0][1] - r[1][1])
        a_s.append(r[0][0])
        a_a.append(r[1][0])
    dauc, dll, a_a, a_s = map(np.array, (dauc, dll, a_a, a_s))
    d_pair = c["auc_boot"][:, 1] - c["auc_boot"][:, 0]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axs[0]
    ax.hist(dauc, bins=25, color=VIOLET, alpha=0.8)
    ax.axvline(0, color=RED, lw=1.8)
    ax.axvline(NUM["dif"]["auc"], color=INK, lw=1.3)
    for x in q(dauc):
        ax.axvline(x, color=GREEN, ls="--", lw=1.4)
    ax.set(xlabel="ΔAUC = amplio − simple, en cada partición", ylabel="N.º de particiones")
    titulo(ax, f"300 particiones: se reentrena y se reevalúa\nIC [{q(dauc)[0]:.4f}; {q(dauc)[1]:.4f}]", fs=8.8)
    style(ax, "y")
    ax = axs[1]
    for i, (lab, dd_, col) in enumerate((("bootstrap de la prueba\n(modelos fijos)", d_pair, C43), ("repetir partición\n(reentrena y reevalúa)", dauc, VIOLET))):
        lo, hi = q(dd_)
        ax.plot([lo, hi], [i, i], color=col, lw=7, solid_capstyle="round")
        ax.text(hi + 0.003, i, f"[{lo:.4f}; {hi:.4f}]", va="center", fontsize=8.5)
    ax.axvline(0, color=RED, lw=1.8)
    ax.set(yticks=[0, 1], yticklabels=["bootstrap de la prueba\n(modelos fijos)", "repetir partición\n(reentrena y reevalúa)"], xlim=(-0.02, 0.1), ylim=(-0.6, 1.6), xlabel="ΔAUC")
    titulo(ax, "Dos preguntas distintas (diapositiva 17)")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c43_reentrenar.png")
    NUM["reentrenar"] = dict(R=300, media_dauc=float(dauc.mean()), sd_dauc=float(dauc.std(ddof=1)), ic_dauc=[float(x) for x in q(dauc)], ic_dll=[float(x) for x in q(dll)],
                             frac_amplio_mejor=float(np.mean(dauc > 0)), sd_auc_amplio=float(a_a.std(ddof=1)), auc_amplio_rango=[float(a_a.min()), float(a_a.max())],
                             ancho_boot=float(np.diff(q(d_pair))[0]), ancho_rep=float(np.diff(q(dauc))[0]))


def tamano_fig():
    rng = np.random.default_rng(43008)
    ns = np.array([25, 50, 100, 171, 400, 1000, 2500])
    p0 = 0.9
    anch_b, anch_w = [], []
    for nn in ns:
        wb = []
        for _ in range(30):
            x = rng.binomial(1, p0, nn)
            bm = np.array([rng.choice(x, nn, replace=True).mean() for _ in range(500)])
            wb.append(np.diff(q(bm))[0])
        anch_b.append(np.mean(wb))
        anch_w.append(2 * 1.96 * np.sqrt(p0 * (1 - p0) / nn))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axs[0]
    ax.plot(ns, anch_w, "o-", color=C43, lw=2, label="fórmula normal 2 · 1.96 √(p(1 − p)/n)")
    ax.plot(ns, anch_b, "s--", color=AMBER, lw=2, label="bootstrap (promedio de 30 muestras)")
    ax.set_xscale("log")
    ax.set(xlabel="N.º de casos de prueba (escala log)", ylabel="ancho del IC 95 %", ylim=(0, 0.32))
    ax.legend(frameon=False, fontsize=8.3)
    titulo(ax, "La incertidumbre baja como 1/√n (acierto real 0.90)")
    style(ax)
    ax = axs[1]
    comp = NUM["prueba"]
    ns2 = np.arange(20, 1001)
    ax.plot(ns2, np.sqrt(0.9 * 0.1 / ns2), color=C43, lw=2)
    ax.scatter([100, 1000, 171], [np.sqrt(0.09 / 100), np.sqrt(0.09 / 1000), NUM["prop"]["exactitud"]["se_wald"]], color=[AMBER, AMBER, RED], zorder=3, s=40)
    ax.annotate("n = 100: 0.030", (100, 0.03), xytext=(180, 0.045), fontsize=8.3, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("n = 1 000: 0.0095", (1000, np.sqrt(0.09 / 1000)), xytext=(520, 0.02), fontsize=8.3, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate(f"Wisconsin: n = 171, {comp['amplio']['exactitud']:.3f} → SE {NUM['prop']['exactitud']['se_wald']:.4f}", (171, NUM["prop"]["exactitud"]["se_wald"]), xytext=(260, 0.01), fontsize=8.3, color=RED, arrowprops=dict(arrowstyle="-", color=RED))
    ax.set(xlabel="N.º de casos de prueba", ylabel="error estándar de la exactitud", ylim=(0, 0.062))
    titulo(ax, "Para reducir el SE a la mitad se necesitan 4 veces más casos")
    style(ax)
    fig.tight_layout()
    save(fig, "c43_tamano.png")
    NUM["tamano"] = dict(ns=[int(x) for x in ns], ancho_boot=[float(x) for x in anch_b], ancho_normal=[float(x) for x in anch_w])


def regresion_fig():
    c = calc()
    rb, obs, pre = c["reg_boot"], c["obs"], c["pre"]
    e = obs - pre
    mae, rmse = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, k, nm, est in ((axs[0], 0, "MAE", mae), (axs[1], 1, "RMSE", rmse)):
        ax.hist(rb[:, k], bins=25, color=C43, alpha=0.8)
        ax.axvline(est, color=RED, lw=1.8)
        for x in q(rb[:, k]):
            ax.axvline(x, color=INK, ls="--", lw=1.2)
        ax.set(xlabel=f"{nm} de la remuestra", ylabel="N.º de remuestras")
        titulo(ax, f"{nm} = {est:.3f}   IC [{q(rb[:, k])[0]:.3f}; {q(rb[:, k])[1]:.3f}]  (5 casos: ilustrativo)", fs=8.8)
        style(ax, "y")
    fig.tight_layout()
    save(fig, "c43_regresion.png")
    NUM["regresion"] = dict(errores=[int(x) for x in e], mae=mae, rmse=rmse, ic_mae=[float(x) for x in q(rb[:, 0])], ic_rmse=[float(x) for x in q(rb[:, 1])], slide_R=dict(ic_rmse=[1.949, 4.313]))


# ----------------------------------------------------------------------------- 7. mapa de la clase
def mapa_fig():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    pasos = [("1. Definir", "unidad de separación\ny métrica relevante"), ("2. Seleccionar", "receta con entrena-\nmiento y validación\n(AIC, BIC, OOF)"), ("3. Reajustar", "la receta elegida con\ntodo el entrenamiento"),
             ("4. Evaluar", "en prueba, sin volver\na elegir nada"), ("5. Cuantificar", "incertidumbre de esa\nevaluación: bootstrap")]
    for i, (t, s) in enumerate(pasos):
        x0 = 0.1 + i * 2.5
        ax.add_patch(FancyBboxPatch((x0, 4.0), 2.25, 1.7, boxstyle="round,pad=0.03", fc="white" if i < 4 else C43_BG, ec=C43, lw=1.6 if i < 4 else 2.4))
        ax.text(x0 + 1.125, 5.35, t, ha="center", fontsize=10, fontweight="bold", color=C43)
        ax.text(x0 + 1.125, 4.65, s, ha="center", va="center", fontsize=8.3, color=INK)
    for xx in (2.4, 4.9, 7.4, 9.9):
        ax.annotate("", xy=(xx + 0.15, 4.85), xytext=(xx, 4.85), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 12.3, 3.6, boxstyle="round,pad=0.03", fc=C43_BG, ec=C43, lw=1.4))
    pr, ic, dif, md, sol, ro = NUM["prueba"], NUM["ic"], NUM["dif"], NUM["media"], NUM["solap"], NUM["reentrenar"]
    lineas = [
        ("Datos: wisconsin.csv · 569 muestras = 398 entrenamiento + 171 prueba (107 benignos y 64 malignos) · los mismos de la Clase 37 · B = 2 000 remuestras", C43, "bold"),
        (f"Modelo simple: AUC {pr['simple']['auc']:.4f} [{ic['auc_simple'][0]:.4f}; {ic['auc_simple'][1]:.4f}] · log loss {pr['simple']['log_loss']:.4f} [{ic['ll_simple'][0]:.4f}; {ic['ll_simple'][1]:.4f}]", INK, "normal"),
        (f"Modelo amplio: AUC {pr['amplio']['auc']:.4f} [{ic['auc_amplio'][0]:.4f}; {ic['auc_amplio'][1]:.4f}] · log loss {pr['amplio']['log_loss']:.4f} [{ic['ll_amplio'][0]:.4f}; {ic['ll_amplio'][1]:.4f}]", INK, "normal"),
        (f"Diferencia pareada (amplio mejor): ΔAUC {dif['auc']:.4f} [{dif['ic_auc'][0]:.4f}; {dif['ic_auc'][1]:.4f}] · Δ log loss {dif['ll']:.4f} [{dif['ic_ll'][0]:.4f}; {dif['ic_ll'][1]:.4f}] · ninguno contiene 0", INK, "normal"),
        (f"Los IC de cada AUC se solapan ({sol['solape'][0]:.3f}–{sol['solape'][1]:.3f}) y aun así la diferencia pareada es positiva: solapar no es 'no hay diferencia'", RED, "normal"),
        (f"Media de 5 valores: 5.2 · SE bootstrap {md['se']:.3f} (exacto {md['se_exacto']:.3f}) · IC percentil [{md['ic'][0]:.1f}; {md['ic'][1]:.1f}]", INK, "normal"),
        ("R ↔ Python:  sample(x, replace = TRUE) ↔ rng.choice(x, replace=True) · quantile ↔ np.quantile · sd ↔ np.std(ddof=1) · pROC::roc ↔ roc_auc_score · AIC/BIC ↔ .aic / .bic_llf", MUTED, "italic"),
    ]
    yy = 3.35
    for i, (t, col, w) in enumerate(lineas):
        ax.text(6.25, yy, t, ha="center", fontsize=8.3 if i < 6 else 7.9, color=col, fontweight="bold" if w == "bold" else "normal", style="italic" if w == "italic" else "normal")
        yy -= 0.5
    save(fig, "c43_esquema.png")


def cierre_calc():
    """Ejercicio de cierre del notebook: modelo intermedio frente a amplio en prueba (misma lógica que la solución del notebook)."""
    c = calc()
    y, pa, neg, pos = c["y"], c["pa"], c["neg"], c["pos"]
    pi_ = c["mi"].predict(c["te"]).to_numpy()
    n = len(y)
    rng = np.random.default_rng(43020)
    dauc = np.empty(B)
    dll = np.empty(B)
    for b in range(B):
        i = np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])
        dauc[b] = roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pi_[i])
        j = rng.choice(n, size=n, replace=True)
        dll[b] = log_loss(y[j], pi_[j]) - log_loss(y[j], pa[j])
    ok_i, ok_a = (pi_ >= 0.5) == y, (pa >= 0.5) == y
    bb, cc = int(np.sum(ok_i & ~ok_a)), int(np.sum(~ok_i & ok_a))
    NUM["cierre"] = dict(auc_i=float(roc_auc_score(y, pi_)), auc_a=float(roc_auc_score(y, pa)), ll_i=float(log_loss(y, pi_)), ll_a=float(log_loss(y, pa)),
                         acc_i=float(ok_i.mean()), acc_a=float(ok_a.mean()), dauc=float(roc_auc_score(y, pa) - roc_auc_score(y, pi_)), ic_dauc=[float(x) for x in q(dauc)],
                         dll=float(log_loss(y, pi_) - log_loss(y, pa)), ic_dll=[float(x) for x in q(dll)], b=bb, c=cc, p_exacto=float(stats.binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0)


PASOS = dict(incertidumbre=incertidumbre_fig, esquema_boot=esquema_fig, media=media_fig, oob=oob_fig, seleccion=seleccion_fig, auc=auc_fig, intervalos=intervalos_fig, roc=roc_fig,
             pareado=pareado_fig, solapamiento=solapamiento_fig, mcnemar=mcnemar_fig, reentrenar=reentrenar_fig, tamano=tamano_fig, regresion=regresion_fig, mapa=mapa_fig, cierre=cierre_calc)
ORDEN = ["incertidumbre", "media", "esquema_boot", "oob", "seleccion", "auc", "intervalos", "roc", "pareado", "solapamiento", "mcnemar", "reentrenar", "tamano", "regresion", "mapa", "cierre"]

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c43_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or ORDEN
    for k in ORDEN:
        if k in elegidos:
            PASOS[k]()
            print("hecho:", k, flush=True)
    # comprobaciones contra los resultados del profesor (diapositivas 27 y 39)
    c = calc()
    pr = NUM.get("prueba")
    if pr:
        assert round(pr["simple"]["auc"], 4) == 0.9394 and round(pr["amplio"]["auc"], 4) == 0.9746
        assert round(pr["simple"]["log_loss"], 4) == 0.2997 and round(pr["amplio"]["log_loss"], 4) == 0.1833
        assert round(pr["amplio"]["exactitud"], 4) == 0.9415 and round(pr["amplio"]["sensibilidad"], 4) == 0.9062
    NUM["datos"] = dict(n=int(len(c["d"])), n_tr=int(len(c["tr"])), n_te=int(len(c["te"])), n_neg=int((c["y"] == 0).sum()), n_pos=int((c["y"] == 1).sum()),
                        igual_c37=bool(np.allclose(c["d"][["radio", "textura", "suavidad", "concavidad"]].to_numpy(),
                                                   pd.read_csv(CARPETA37 / "wisconsin_clase.csv")[["radio", "textura", "suavidad", "concavidad"]].to_numpy())
                                       and (c["d"]["conjunto"].to_numpy() == pd.read_csv(CARPETA37 / "wisconsin_clase.csv")["conjunto"].to_numpy()).all()),
                        tr_malignos=int(c["tr"].maligno.sum()), tr_benignos=int((c["tr"].maligno == 0).sum()), B=B)
    out = Path(__file__).resolve().parent / "c43_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
