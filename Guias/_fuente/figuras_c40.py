"""Figuras de la Clase 40 (Naive Bayes y clasificación probabilística).
Fuentes: diapositivas DipDS_2026_S40.pdf, 40_naive_bayes.py / .R.txt del profesor Eloy Alvarado y spam_s40.csv (4 601 correos de Spambase,
54 frecuencias de palabras y caracteres; 3 680 de desarrollo con 5 pliegues y 921 de prueba histórica). La carpeta de la clase se lee sin modificarla.
Todo sale de código; lo aleatorio usa semilla fija. Escribe c40_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c40.py   (o con nombres: python -X utf8 figuras_c40.py bayes laplace calibracion)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C40 = "#BE123C"      # color de la clase 40 (carmesí)
C40_BG = "#FFF1F2"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
TEAL = "#0D9488"
VIOLET = "#7C3AED"
GREEN = "#16A34A"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA40 = next(p for p in BASE.glob("Clase 40*") if p.is_dir())
CARPETA38 = next(p for p in BASE.glob("Clase 38*") if p.is_dir())
_DAT = {}
CLS = ["legítimo", "spam"]


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


# ----------------------------------------------------------------------------- datos y funciones comunes
def datos():
    """Lee spam_s40.csv una sola vez con la misma partición del profesor (desarrollo / prueba y 5 pliegues)."""
    if _DAT:
        return _DAT
    d = pd.read_csv(CARPETA40 / "spam_s40.csv")
    var = [c for c in d if c.startswith(("word_freq_", "char_freq_"))]
    assert len(var) == 54
    tr = d[d.conjunto == "entrenamiento"].reset_index(drop=True)
    te = d[d.conjunto == "prueba"].reset_index(drop=True)
    _DAT.update(d=d, var=var, tr=tr, te=te, X=tr[var].to_numpy(), y=tr.spam.to_numpy(), fold=tr.fold.to_numpy(),
                Xte=te[var].to_numpy(), yte=te.spam.to_numpy())
    return _DAT


class Pred(np.ndarray):
    """Probabilidades con sus log-probabilidades (n×2) adjuntas, como en el script del profesor."""
    lp = None


def con_lp(p, lp):
    r = np.asarray(p).view(Pred)
    r.lp = lp
    return r


def medir(y, p, t=0.5):
    """Mismas medidas que `medir` del script del profesor: log loss y AUC desde las log-probabilidades cuando existen."""
    lp = getattr(p, "lp", None)
    q = np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    if lp is None:
        lp = np.column_stack([np.log1p(-q), np.log(q)])
    perdida = float(-np.maximum(lp[np.arange(len(y)), y], np.log(1e-15)).mean())
    pred = np.asarray(p) >= t
    fp, fn = int(np.sum((y == 0) & pred)), int(np.sum((y == 1) & ~pred))
    vn, vp = int(np.sum((y == 0) & ~pred)), int(np.sum((y == 1) & pred))
    return dict(log_loss=perdida, brier=float(np.mean((y - np.asarray(p)) ** 2)), auc=float(roc_auc_score(y, lp[:, 1] - lp[:, 0])),
                accuracy=float(np.mean(pred == y)), VN=vn, FP=fp, FN=fn, VP=vp, costo_1000=float(1000 * (5 * fp + fn) / len(y)))


def logprobs(m, x):
    """Log-probabilidades estables: desde decision_function si existe (logística) o predict_log_proba (Naive Bayes)."""
    if hasattr(m, "decision_function"):
        lo = m.decision_function(x)
        return np.column_stack([-np.logaddexp(0, lo), -np.logaddexp(0, -lo)])
    return m.predict_log_proba(x)


def oof(modelo, X, y, fold, transformar=None):
    """Predicciones fuera de pliegue con los 5 pliegues de la partición del profesor (con sus log-probabilidades)."""
    p = np.zeros(len(y))
    lp = np.zeros((len(y), 2))
    for k in np.unique(fold):
        v = fold == k
        xa, xv = (X[~v], X[v]) if transformar is None else (transformar(X[~v]), transformar(X[v]))
        m = clone(modelo).fit(xa, y[~v])
        lp[v] = logprobs(m, xv)
        p[v] = np.exp(lp[v][:, 1])
    return con_lp(p, lp)


def bern(alpha):
    return BernoulliNB(alpha=alpha, binarize=0.0)


def gauss_log():
    return GaussianNB(var_smoothing=1e-9)


def oof_receta(clave):
    """Las cinco recetas del profesor (misma lógica que 40_naive_bayes.py)."""
    d = datos()
    X, y, f = d["X"], d["y"], d["fold"]
    if clave == "FRECUENCIA":
        p = np.zeros(len(y))
        for k in np.unique(f):
            v = f == k
            p[v] = y[~v].mean()
        return con_lp(p, np.column_stack([np.log1p(-p), np.log(p)]))
    if clave.startswith("BERN"):
        a = {"BERN_05": 0.5, "BERN_1": 1.0, "BERN_2": 2.0}[clave]
        return oof(bern(a), X, y, f)
    return oof(gauss_log(), X, y, f, transformar=np.log1p)


def logistica():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))


def posterior_bin(prior, th1, th0, x):
    """Posterior de spam para indicadores x con probabilidades de presencia th1 (spam) y th0 (legítimo)."""
    s1 = prior * np.prod(np.where(x == 1, th1, 1 - th1))
    s0 = (1 - prior) * np.prod(np.where(x == 1, th0, 1 - th0))
    return s1 / (s0 + s1)


def recetas_oof():
    """Todas las OOF necesarias, calculadas una sola vez."""
    if "oof" in _DAT:
        return _DAT["oof"]
    d = datos()
    o = {c: oof_receta(c) for c in ("FRECUENCIA", "BERN_05", "BERN_1", "BERN_2", "GAUSS_LOG")}
    o["LOGIT54"] = oof(logistica(), d["X"], d["y"], d["fold"])
    # logística con las 57 entradas, como la Clase 38
    v57 = d["var"] + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
    o["LOGIT57"] = oof(logistica(), d["tr"][v57].to_numpy(), d["y"], d["fold"])
    _DAT["oof"] = o
    return o


# ----------------------------------------------------------------------------- 1. Bayes para clasificar
def bayes_fig():
    pi = np.linspace(0.001, 0.999, 300)
    ps, pl = 16 / 20, 8 / 80
    post = lambda g: g * ps / (g * ps + (1 - g) * pl)
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.8))
    ax = axs[0]
    ax.plot(pi, post(pi), color=C40, lw=2)
    ax.scatter([0.2, 0.02], [post(0.2), post(0.02)], color=AMBER, s=60, zorder=3)
    ax.annotate(f"20 % de spam → {post(0.2) * 100:.1f} %", (0.2, post(0.2)), xytext=(0.32, 0.55), fontsize=8.3, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate(f"2 % de spam → {post(0.02) * 100:.1f} %", (0.02, post(0.02)), xytext=(0.2, 0.18), fontsize=8.3, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=1)
    ax.set(xlabel="probabilidad previa de spam", ylabel="P(spam | «oferta»)", xlim=(0, 1), ylim=(0, 1.02))
    titulo(ax, "La misma palabra, dos puntos de partida")
    style(ax)
    ax = axs[1]
    nombres = ["P(oferta | spam)\n16/20", "P(oferta | legítimo)\n8/80", "P(spam | oferta)\n16/24"]
    vals = [ps, pl, 16 / 24]
    barras = ax.bar(range(3), vals, color=[BLUE, BLUE, C40], width=0.6)
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(nombres, fontsize=8)
    ax.set(ylim=(0, 1), ylabel="probabilidad")
    ax.text(0.5, 0.93, "dentro de cada clase", ha="center", fontsize=7.8, color=BLUE)
    ax.text(2, 0.9, "dentro de la señal", ha="center", fontsize=7.8, color=C40)
    titulo(ax, "Tres preguntas, tres respuestas distintas")
    style(ax, "y")
    ax = axs[2]
    casos = ["solo oferta", "oferta,\nsin enlace", "oferta\ny enlace"]
    v = [0.16 / 0.24, 0.048 / (0.064 + 0.048), 0.112 / (0.016 + 0.112)]
    barras = ax.bar(range(3), v, color=[TEAL, AMBER, C40], width=0.6)
    for b, vv in zip(barras, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + 0.02, f"{vv:.4f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(2.45, 0.51, "umbral 0.5", ha="right", fontsize=7.8, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(casos, fontsize=8.3)
    ax.set(ylim=(0, 1.05), ylabel="P(spam | señales observadas)")
    titulo(ax, "Cada señal observada cambia la probabilidad")
    style(ax, "y")
    NUM["bayes"] = dict(n=[80, 20], oferta=[8, 16], enlace=[16, 14], p_oferta_spam=ps, p_oferta_leg=pl, post_oferta=16 / 24, post_prior_002=float(post(0.02)),
                        post_oferta_sin_enlace=v[1], post_oferta_enlace=v[2], s0=0.064, s1=0.048, s0_enlace=0.016, s1_enlace=0.112,
                        denominador=0.24, numerador_spam=0.16, numerador_leg=0.08)
    fig.tight_layout()
    save(fig, "c40_bayes.png")


# ----------------------------------------------------------------------------- 2. independencia condicional y suavizado de la tabla
def indep_fig():
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.8))
    ax = axs[0]
    dd = np.arange(1, 21)
    ax.semilogy(dd, 2.0 ** dd - 1, "o-", color=RED, label="tabla conjunta: 2^d − 1 parámetros por clase", ms=4)
    ax.semilogy(dd, dd, "o-", color=TEAL, label="Naive Bayes: d parámetros por clase", ms=4)
    ax.text(19.6, 2.0 ** 20 * 0.06, f"d = 20: {2 ** 20 - 1:,}".replace(",", " "), ha="right", fontsize=8, color=RED)
    ax.text(20, 8, "d = 20: 20", ha="right", fontsize=8, color=TEAL)
    ax.set(xlabel="número de señales d", ylabel="parámetros por clase (escala log)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(ax, "La independencia evita contar todas las combinaciones")
    style(ax)
    ax = axs[1]
    # productos de la diapositiva 17 y 22
    pos = np.arange(2)
    s0 = [0.8 * 0.1 * 0.8, 0.8 * (9 / 82) * (65 / 82)]
    s1 = [0.2 * 0.8 * 0.3, 0.2 * (17 / 22) * (7 / 22)]
    ax.bar(pos - 0.18, s0, 0.34, color=BLUE, label="s0: legítimo")
    ax.bar(pos + 0.18, s1, 0.34, color=C40, label="s1: spam")
    for i in range(2):
        ax.text(pos[i] - 0.18, s0[i] + 0.002, f"{s0[i]:.4f}", ha="center", fontsize=8)
        ax.text(pos[i] + 0.18, s1[i] + 0.002, f"{s1[i]:.4f}", ha="center", fontsize=8)
        ax.text(pos[i], 0.085, f"P(spam) = {s1[i] / (s0[i] + s1[i]):.4f}", ha="center", fontsize=8.4, color=INK, fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels(["sin suavizado", "suavizado, α = 1"])
    ax.set(ylim=(0, 0.115), ylabel="producto (prior × factores)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    titulo(ax, "Oferta presente y enlace ausente (diap. 17 y 22)")
    style(ax, "y")
    ax = axs[2]
    th1, th0 = np.array([16 / 20, 14 / 20]), np.array([8 / 80, 16 / 80])
    pats = [(0, 0), (0, 1), (1, 0), (1, 1)]
    nb = [posterior_bin(0.2, th1, th0, np.array(p)) for p in pats]
    ax.bar(range(4), nb, color=[GRID, TEAL, AMBER, C40], edgecolor=MUTED)
    for i, v in enumerate(nb):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["sin oferta\nsin enlace", "sin oferta\ncon enlace", "oferta\nsin enlace", "oferta\ncon enlace"], fontsize=7.8)
    ax.set(ylim=(0, 1.05), ylabel="P(spam | patrón), Naive Bayes")
    titulo(ax, "Los cuatro patrones, con solo 4 probabilidades por clase")
    style(ax, "y")
    NUM["indep"] = dict(s0=s0, s1=s1, post=[s1[i] / (s0[i] + s1[i]) for i in range(2)], patrones=dict(zip(["00", "01", "10", "11"], nb)),
                        patrones_2_10=2 ** 10, d54=54, params_d20=2 ** 20 - 1)
    fig.tight_layout()
    save(fig, "c40_independencia.png")


# ----------------------------------------------------------------------------- 3. duplicar una señal
def dup_fig():
    d = datos()
    copias = np.array([1, 2, 4])
    s1c = 0.5 * 0.8 ** copias
    s0c = 0.5 * 0.2 ** copias
    tox = s1c / (s1c + s0c)
    # real: duplicar las 54 señales binarias, k veces
    Xb = (d["X"] > 0).astype(float)
    ks = [1, 2, 4, 8]
    fila = []
    for k in ks:
        p = oof(bern(1.0), np.hstack([Xb] * k), d["y"], d["fold"])
        m = medir(d["y"], p)
        fila.append(dict(k=k, log_loss=m["log_loss"], auc=m["auc"], accuracy=m["accuracy"], brier=m["brier"], p_media_extrema=float(np.mean((p < 0.01) | (p > 0.99)))))
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax = axs[0]
    ax.plot(copias, tox, "o-", color=C40, label="Naive Bayes: trata las copias como independientes")
    ax.axhline(0.8, color=BLUE, label="información disponible: la misma señal")
    for c, v in zip(copias, tox):
        ax.text(c, v + 0.015, f"{v:.4f}", ha="center", fontsize=8.3)
    ax.set(xlabel="número de copias de la señal", ylabel="P(clase 1)", ylim=(0.65, 1.04), xticks=copias)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(ax, "Ejemplo controlado de la clase (diap. 46 y 47)")
    style(ax)
    ax = axs[1]
    ax.plot(ks, [r["log_loss"] for r in fila], "o-", color=C40, label="log loss OOF")
    ax.set(xlabel="veces que se repiten las 54 señales", ylabel="log loss OOF", xscale="log", xticks=ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax2 = ax.twinx()
    ax2.plot(ks, [r["auc"] for r in fila], "s--", color=TEAL, label="AUC OOF")
    ax2.set(ylabel="AUC OOF", ylim=(0.9, 1.0))
    ax2.spines["right"].set_visible(True)
    ax.legend(loc="upper left", fontsize=7.8, frameon=False)
    ax2.legend(loc="lower right", fontsize=7.8, frameon=False)
    titulo(ax, "Con los correos reales: el orden casi no cambia; las probabilidades sí")
    style(ax)
    NUM["dup"] = dict(copias=copias.tolist(), post=tox.tolist(), correcto=0.8, real=fila)
    fig.tight_layout()
    save(fig, "c40_duplicacion.png")


# ----------------------------------------------------------------------------- 4. gaussiana
def gauss_fig():
    d = datos()
    X, y = d["X"], d["y"]
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.8))
    ax = axs[0]
    z = np.linspace(-3, 5, 400)
    f0 = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    f1 = np.exp(-(z - 2) ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(z, 0.8 * f0, color=BLUE, label="clase 0: 0.8·f₀(x)")
    ax.plot(z, 0.2 * f1, color=C40, label="clase 1: 0.2·f₁(x)")
    xs = (np.log(4) + 2) / 2
    ax.axvline(1.5, color=INK, ls=":")
    ax.axvline(xs, color=AMBER, ls="--")
    d0, d1 = np.exp(-1.5 ** 2 / 2) / np.sqrt(2 * np.pi), np.exp(-(1.5 - 2) ** 2 / 2) / np.sqrt(2 * np.pi)
    post = 0.2 * d1 / (0.8 * d0 + 0.2 * d1)
    ax.text(1.55, 0.31, f"x = 1.5\nP(clase 1) = {post:.4f}", fontsize=8, va="top")
    ax.text(xs + 0.05, 0.02, f"frontera\nx = {xs:.3f}", fontsize=7.6, color=AMBER)
    ax.set(xlabel="x", ylabel="densidad × previa")
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(ax, "Dos curvas normales y su previa (diap. 37)")
    style(ax)
    stats = {}
    for ax, nom, t in zip(axs[1:], ["word_freq_you", "char_freq_!"], ["Palabra «you»", "Carácter «!»"]):
        j = d["var"].index(nom)
        zc = np.log1p(X[:, j])
        bins = np.linspace(0, zc.max(), 36)
        info = {}
        for c, col in zip((0, 1), (BLUE, C40)):
            zz = zc[y == c]
            ax.hist(zz, bins=bins, density=True, alpha=0.45, color=col, label=CLS[c])
            mu, sd = zz.mean(), zz.std(ddof=0)
            g = np.linspace(0, zc.max(), 300)
            ax.plot(g, np.exp(-((g - mu) / sd) ** 2 / 2) / (sd * np.sqrt(2 * np.pi)), color=col, lw=2)
            info[CLS[c]] = dict(media=float(mu), sd=float(sd), ceros=float((zz == 0).mean()))
        ax.set(xlabel=f"log(1 + {nom})", ylabel="densidad", ylim=(0, 6 if "you" in nom else 5))
        ax.legend(fontsize=7.8, frameon=False)
        titulo(ax, f"{t}: histograma y curva normal por clase")
        style(ax)
        stats[nom] = info
    NUM["gauss"] = dict(x=1.5, densidad_0=float(d0), densidad_1=float(d1), producto_0=float(0.8 * d0), producto_1=float(0.2 * d1), post=float(post), frontera=float(xs), stats=stats)
    fig.tight_layout()
    save(fig, "c40_gaussiana.png")


# ----------------------------------------------------------------------------- 5. Bernoulli sobre los correos reales
def bernoulli_fig():
    d = datos()
    X, y = d["X"], d["y"]
    o = recetas_oof()
    m = bern(1.0).fit(X, y)
    i = int(np.argmin(d["tr"].id.to_numpy()))
    b = (X[i] > 0).astype(float)
    th = np.exp(m.feature_log_prob_)
    aportes = b * np.log(th[1] / th[0]) + (1 - b) * np.log((1 - th[1]) / (1 - th[0]))
    prior_lo = float(np.log(y.mean() / (1 - y.mean())))
    top = np.argsort(-np.abs(aportes), kind="stable")[:8]
    top = top[np.argsort(aportes[top])]
    p1 = np.where(b == 1, th[1], 1 - th[1])
    p0 = np.where(b == 1, th[0], 1 - th[0])
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    ax = axs[0]
    pos = np.arange(len(top))
    ax.barh(pos - 0.18, p0[top], 0.34, color=BLUE, label="dentro de legítimos")
    ax.barh(pos + 0.18, p1[top], 0.34, color=C40, label="dentro de spam")
    etiquetas = [d["var"][j].replace("word_freq_", "").replace("char_freq_", "carácter ") + (" (presente)" if b[j] else " (ausente)") for j in top]
    ax.set_yticks(pos)
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set(xlabel="P(estado observado | clase)", xlim=(0, 1.3))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(ax, "Correo ID 1: ocho de sus 54 señales (diap. 50)")
    style(ax, "x")
    ax = axs[1]
    pb = o["BERN_05"]
    bins = np.linspace(0, 1, 21)
    ax.hist(pb[y == 0], bins=bins, alpha=0.55, color=BLUE, label="legítimos")
    ax.hist(pb[y == 1], bins=bins, alpha=0.55, color=C40, label="spam")
    ax.set(xlabel="P(spam) OOF, Bernoulli α = 0.5", ylabel="correos", yscale="log")
    ax.legend(fontsize=7.8, frameon=False, loc="upper center")
    titulo(ax, "Casi todo se concentra en 0 y en 1")
    style(ax)
    ax = axs[2]
    th_s, th_l = th[1], th[0]
    # tasas de presencia por clase de las 54 variables
    ax.scatter(th_l, th_s, s=22, color=C40, alpha=0.8)
    ax.plot([0, 1], [0, 1], ":", color=MUTED)
    for nom in ["word_freq_free", "word_freq_your", "word_freq_remove", "word_freq_hp", "char_freq_$", "char_freq_!", "word_freq_george"]:
        j = d["var"].index(nom)
        ax.annotate(nom.replace("word_freq_", "").replace("char_freq_", "carácter "), (th_l[j], th_s[j]), xytext=(4, 3), textcoords="offset points", fontsize=7.6)
    ax.set(xlabel="P(presente | legítimo)", ylabel="P(presente | spam)", xlim=(0, 1), ylim=(0, 1))
    titulo(ax, "Las 54 señales, α = 1: sobre la diagonal favorecen spam")
    style(ax)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    cuenta0, cuenta1 = ((X > 0) == (b > 0))[y == 0].sum(0), ((X > 0) == (b > 0))[y == 1].sum(0)
    j_add = d["var"].index("word_freq_address")
    suma = float(aportes.sum())
    NUM["bern"] = dict(id=int(d["tr"].id.iloc[i]), log_odds_prior=prior_lo, suma_aportes=suma, log_odds_total=prior_lo + suma,
                       posterior=float(m.predict_proba(X[i:i + 1])[0, 1]), n=[n0, n1], address=dict(presente=int(b[j_add]), c0=int(cuenta0[j_add]), c1=int(cuenta1[j_add]), th0=float(p0[j_add]), th1=float(p1[j_add])),
                       free_valor=float(d["tr"].loc[i, "word_freq_free"]), money_valor=float(d["tr"].loc[i, "word_freq_money"]), excl_valor=float(d["tr"].loc[i, "char_freq_!"]),
                       prior_spam=float(y.mean()), extremos_p=float(np.mean((pb < 0.01) | (pb > 0.99))), top=[d["var"][j] for j in top],
                       n_aportes_positivos=int((aportes > 0).sum()), n_aportes_negativos=int((aportes < 0).sum()))
    fig.tight_layout()
    save(fig, "c40_bernoulli.png")
    _DAT["aportes"] = (aportes, prior_lo, b)


# ----------------------------------------------------------------------------- 6. variantes: Bernoulli, multinomial, gaussiana
def variantes_fig():
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.7))
    ax = axs[0]
    M = np.array([[1, 0, 1], [2, 0, 1]])
    ax.imshow(M, cmap="Reds", vmin=0, vmax=3, aspect="auto")
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=13, color=INK if v < 2 else "white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["oferta", "reunión", "hoy"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["«oferta hoy»", "«oferta oferta hoy»"], fontsize=8.6)
    ax.grid(False)
    titulo(ax, "Conteos de palabras (diap. 39)")
    ax = axs[1]
    cuentas = np.array([[0.0, 4.0, 2.0], [4.0, 0.0, 2.0]])
    theta = (cuentas + 1) / (cuentas.sum(1, keepdims=True) + 3)
    pm = []
    for msg in M:
        l = np.log(0.5) + msg @ np.log(theta).T
        z = l - l.max()
        pm.append(float(np.exp(z[1]) / np.exp(z).sum()))
    ax.bar([0, 1], pm, color=C40, width=0.5, label="multinomial")
    ax.bar([2, 3], [pm[0], pm[0]], color=TEAL, width=0.5, label="Bernoulli: ambos mensajes = (1, 0, 1)", alpha=0.8)
    for x_, v in zip([0, 1], pm):
        ax.text(x_, v + 0.02, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.text(2.5, pm[0] + 0.02, "misma\nrepresentación", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["oferta\nhoy", "oferta oferta\nhoy", "oferta\nhoy", "oferta oferta\nhoy"], fontsize=7.6)
    ax.set(ylim=(0, 1.45), ylabel="P(clase 1)")
    ax.legend(fontsize=7.2, frameon=False, loc="upper center")
    titulo(ax, "Repetir una palabra solo importa si se cuenta")
    style(ax, "y")
    ax = axs[2]
    ax.axis("off")
    filas = [["Variante", "Entrada que necesita", "En este caso"],
             ["Bernoulli", "presencias 0/1", "usar frecuencia > 0"],
             ["Gaussiana", "medidas numéricas", "probar log(1 + x)"],
             ["Multinomial", "conteos de términos", "falta información:\nlongitud del mensaje"]]
    tb = ax.table(cellText=filas[1:], colLabels=filas[0], loc="center", cellLoc="left", colWidths=[0.26, 0.38, 0.36])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 2.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(C40)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    titulo(ax, "¿Qué variante usa cada tipo de dato? (diap. 40)")
    NUM["variantes"] = dict(posterior_multinomial=pm, theta=theta.tolist(), mensajes=M.tolist())
    fig.tight_layout()
    save(fig, "c40_variantes.png")


# ----------------------------------------------------------------------------- 7. suavizado de Laplace
def laplace_fig():
    d = datos()
    X, y, f = d["X"], d["y"], d["fold"]
    al = np.linspace(0, 3, 121)
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.9))
    ax = axs[0]
    for cnt, n, col, et in [(0, 20, C40, "premio: 0 de 20"), (8, 80, BLUE, "oferta (legítimo): 8 de 80"), (16, 20, TEAL, "oferta (spam): 16 de 20")]:
        ax.plot(al, (cnt + al) / (n + 2 * al), color=col, label=et)
    ax.scatter([1], [1 / 22], color=C40, zorder=3)
    ax.text(1.05, 0.062, "0.0455", fontsize=8, color=C40)
    ax.set(xlabel="peso α sumado a presente y a ausente", ylabel="θ̂ = (conteo + α)/(n + 2α)", ylim=(-0.02, 0.9))
    ax.legend(fontsize=7.8, frameon=False, loc="center right")
    titulo(ax, "Un cero se vuelve pequeño; lo demás casi no cambia")
    style(ax)
    ax = axs[1]
    ps = []
    for a in al:
        s0 = 0.8 * ((8 + a) / (80 + 2 * a)) * ((64 + a) / (80 + 2 * a))
        s1 = 0.2 * ((16 + a) / (20 + 2 * a)) * ((6 + a) / (20 + 2 * a))
        ps.append(s1 / (s0 + s1))
    ax.plot(al, ps, color=C40, lw=2)
    ax.scatter([0, 1], [ps[0], ps[40]], color=AMBER, zorder=3)
    ax.text(0.05, ps[0] + 0.0006, f"{ps[0]:.4f}", fontsize=8)
    ax.text(1.05, ps[40] + 0.0006, f"{ps[40]:.4f}", fontsize=8)
    ax.set(xlabel="peso α", ylabel="P(spam | oferta, sin enlace)")
    titulo(ax, "En el ejemplo de 100 correos: 0.4286 → 0.4140")
    style(ax)
    ax = axs[2]
    alfas = [1e-10, 0.01, 0.1, 0.5, 1, 2, 5, 10, 25]
    filas = []
    for a in alfas:
        p = oof(bern(a), X, y, f)
        mm = medir(y, p)
        filas.append(dict(alpha=a, log_loss=mm["log_loss"], brier=mm["brier"], auc=mm["auc"], accuracy=mm["accuracy"]))
    ax.plot(range(len(alfas)), [r["log_loss"] for r in filas], "o-", color=C40)
    ax.set_xticks(range(len(alfas)))
    ax.set_xticklabels(["≈0", "0.01", "0.1", "0.5", "1", "2", "5", "10", "25"], fontsize=8)
    mejor = int(np.argmin([r["log_loss"] for r in filas]))
    ax.scatter([mejor], [filas[mejor]["log_loss"]], s=90, facecolors="none", edgecolors=AMBER, lw=2)
    ax.set(xlabel="α (Bernoulli, 54 señales)", ylabel="log loss OOF")
    titulo(ax, "Con 3 680 correos casi no importa (α ≤ 2)")
    style(ax)
    # celdas con conteo cero en el entrenamiento (dentro de clase y estado)
    Xb = X > 0
    ceros = int(sum(((Xb[y == c].sum(0) == 0).sum() + (Xb[y == c].sum(0) == (y == c).sum()).sum()) for c in (0, 1)))
    ceros_present = {CLS[c]: [d["var"][j] for j in np.where(Xb[y == c].sum(0) == 0)[0]] for c in (0, 1)}
    NUM["laplace"] = dict(premio=1 / 22, post_sin=ps[0], post_alpha1=ps[40], alfas=filas, celdas_cero=ceros, nunca_presente=ceros_present,
                          theta_oferta_leg=9 / 82, theta_oferta_spam=17 / 22, theta_noenlace_leg=65 / 82, theta_noenlace_spam=7 / 22)
    fig.tight_layout()
    save(fig, "c40_laplace.png")


# ----------------------------------------------------------------------------- 8. log-probabilidades
def logprob_fig():
    d = datos()
    aportes, prior_lo, b = _DAT["aportes"] if "aportes" in _DAT else (None, None, None)
    if aportes is None:
        bernoulli_fig()
        aportes, prior_lo, b = _DAT["aportes"]
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.9))
    ax = axs[0]
    dd = np.arange(1, 1201)
    prod = np.cumprod(np.full(len(dd), 0.3))
    logsum = dd * np.log10(0.3)
    ax.plot(dd, logsum, color=TEAL, label="suma de logaritmos")
    ax.plot(dd, np.where(prod > 0, np.log10(np.where(prod > 0, prod, 1)), np.nan), color=RED, lw=2.4, label="logaritmo del producto calculado")
    ax.axhline(-308, color=MUTED, ls=":")
    d_under = int(np.argmax(prod == 0)) + 1
    ax.axvline(d_under, color=AMBER, ls="--")
    ax.text(d_under + 15, -120, f"el producto se\nhace 0 en d = {d_under}", fontsize=8, color=AMBER)
    ax.set(xlabel="número de factores d (cada uno 0.3)", ylabel="log₁₀ del producto")
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    titulo(ax, "Muchos factores pequeños: el producto desborda")
    style(ax)
    ax = axs[1]
    orden = np.argsort(-np.abs(aportes), kind="stable")
    cum = prior_lo + np.concatenate([[0], np.cumsum(aportes[orden])])
    ax.plot(range(len(cum)), cum, "o-", ms=3, color=C40)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="señales incorporadas (de mayor a menor aporte absoluto)", ylabel="log-odds acumulado de spam")
    ax.text(len(cum) - 1, cum[-1] + 0.3, f"{cum[-1]:.2f}", ha="right", fontsize=8.5)
    ax.text(1.2, prior_lo - 0.1, f"previa: {prior_lo:.2f}", fontsize=8)
    titulo(ax, "Correo ID 1: la evidencia se suma")
    style(ax)
    ax = axs[2]
    t = np.linspace(-8, 8, 300)
    ax.plot(t, 1 / (1 + np.exp(-t)), color=INK)
    for v, et, col in [(prior_lo, "previa", AMBER), (cum[-1], "con las 54 señales", C40)]:
        pv = 1 / (1 + np.exp(-v))
        ax.scatter([v], [pv], color=col, s=60, zorder=3)
        ax.annotate(f"{et}: {pv:.4f}", (v, pv), xytext=(v - 7.4 if v > 0 else v + 0.6, pv - 0.22 if v > 0 else pv + 0.25), fontsize=8, color=col)
    ax.set(xlabel="log-odds = log P(spam | x)/P(legítimo | x)", ylabel="P(spam | x)")
    titulo(ax, "De log-odds a probabilidad: la función logística")
    style(ax)
    NUM["logprob"] = dict(d_desborde=d_under, log_odds_final=float(cum[-1]), prior_lo=float(prior_lo), p_prior=float(1 / (1 + np.exp(-prior_lo))), p_final=float(1 / (1 + np.exp(-cum[-1]))),
                          aporte_max=float(aportes[orden[0]]), var_max=d["var"][orden[0]])
    fig.tight_layout()
    save(fig, "c40_logprob.png")


# ----------------------------------------------------------------------------- 9. fronteras de decisión
def fronteras_fig():
    d = datos()
    j1, j2 = d["var"].index("char_freq_!"), d["var"].index("word_freq_your")
    cols = [j1, j2]
    Z = np.log1p(d["X"][:, cols])
    Zte = np.log1p(d["Xte"][:, cols])
    y, yte = d["y"], d["yte"]
    modelos = {"Naive Bayes gaussiano": GaussianNB(), "LDA (covarianza común)": LinearDiscriminantAnalysis(), "QDA (covarianza por clase)": QuadraticDiscriminantAnalysis(reg_param=0.0),
               "Regresión logística": LogisticRegression(C=1e4, max_iter=3000)}
    g1 = np.linspace(Z[:, 0].min() - 0.1, Z[:, 0].max() + 0.1, 220)
    g2 = np.linspace(Z[:, 1].min() - 0.1, Z[:, 1].max() + 0.1, 220)
    G1, G2 = np.meshgrid(g1, g2)
    rng = np.random.default_rng(RS)
    sub = rng.choice(len(y), 700, replace=False)
    fig, axs = plt.subplots(1, 4, figsize=(15.2, 3.9), sharex=True, sharey=True)
    res = {}
    for ax, (nom, m) in zip(axs, modelos.items()):
        m.fit(Z, y)
        P = m.predict_proba(np.c_[G1.ravel(), G2.ravel()])[:, 1].reshape(G1.shape)
        ax.contourf(G1, G2, P, levels=[0, 0.5, 1], colors=[BLUE, C40], alpha=0.16)
        ax.contour(G1, G2, P, levels=[0.5], colors=[INK], linewidths=1.6)
        for c, col in ((0, BLUE), (1, C40)):
            s = sub[y[sub] == c]
            ax.scatter(Z[s, 0], Z[s, 1], s=7, color=col, alpha=0.5)
        pte = m.predict_proba(Zte)[:, 1]
        mm = medir(yte, pte)
        res[nom] = dict(accuracy=mm["accuracy"], log_loss=mm["log_loss"], auc=mm["auc"])
        titulo(ax, nom, 9)
        ax.text(0.02, 0.97, f"prueba: exactitud {mm['accuracy'] * 100:.1f} %\nlog loss {mm['log_loss']:.3f} · AUC {mm['auc']:.3f}", transform=ax.transAxes, fontsize=7.6, va="top")
        ax.set_xlabel("log(1 + carácter «!»)")
        ax.set_xlim(-0.05, 1.9)
        ax.set_ylim(-0.05, 2.35)
        style(ax)
    axs[0].set_ylabel("log(1 + palabra «your»)")
    NUM["fronteras"] = dict(vars=["char_freq_!", "word_freq_your"], resultados=res)
    fig.tight_layout()
    save(fig, "c40_fronteras.png")


# ----------------------------------------------------------------------------- 10. comparación y calibración
def calibracion_fig():
    d = datos()
    y, X, f = d["y"], d["X"], d["fold"]
    o = recetas_oof()
    # NB + calibración sigmoide aprendida dentro de cada pliegue
    cal = np.zeros(len(y))
    for k in np.unique(f):
        v = f == k
        cal[v] = CalibratedClassifierCV(bern(1.0), method="sigmoid", cv=5).fit(X[~v], y[~v]).predict_proba(X[v])[:, 1]
    o["BERN_CAL"] = cal
    nombres = [("FRECUENCIA", "Frecuencia"), ("BERN_05", "NB Bernoulli α=0.5"), ("BERN_1", "NB Bernoulli α=1"), ("GAUSS_LOG", "NB gaussiano log(1+x)"),
               ("BERN_CAL", "NB Bernoulli + sigmoide"), ("LOGIT54", "Logística (54)"), ("LOGIT57", "Logística (57, C38)")]
    met = {c: medir(y, o[c]) for c, _ in nombres}
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ax = axs[0]
    cs = [GRID, C40, C40, VIOLET, TEAL, BLUE, BLUE]
    ax.barh([n for _, n in nombres], [met[c]["log_loss"] for c, _ in nombres], color=cs)
    for i, (c, _) in enumerate(nombres):
        ax.text(min(met[c]["log_loss"], 3.2) + 0.04, i, f"{met[c]['log_loss']:.3f}  (AUC {met[c]['auc']:.3f})", va="center", fontsize=7.6)
    ax.set_xlim(0, 6.9)
    ax.invert_yaxis()
    ax.set(xlabel="log loss OOF (menor es mejor)")
    titulo(ax, "Ordenar bien no es calibrar bien")
    style(ax, "x")
    ax = axs[1]
    ax.plot([0, 1], [0, 1], ":", color=MUTED)
    cal_tab = {}
    for c, nom, col in [("BERN_1", "NB Bernoulli α=1", C40), ("GAUSS_LOG", "NB gaussiano", VIOLET), ("BERN_CAL", "NB + sigmoide", TEAL), ("LOGIT57", "Logística (57)", BLUE)]:
        p = o[c]
        bins = np.minimum((p * 10).astype(int), 9)
        px, py, nn = [], [], []
        for bb in range(10):
            mk = bins == bb
            if mk.sum() >= 10:
                px.append(p[mk].mean()); py.append(y[mk].mean()); nn.append(int(mk.sum()))
        ax.plot(px, py, "o-", color=col, label=nom, ms=4)
        cal_tab[c] = dict(prob_media=px, frac=py, n=nn)
    ax.set(xlabel="probabilidad media predicha (OOF)", ylabel="fracción observada de spam", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    titulo(ax, "Diagrama de calibración (intervalos de 0.1)")
    style(ax)
    ax = axs[2]
    bins = np.linspace(0, 1, 21)
    ax.hist(o["LOGIT57"], bins=bins, alpha=0.55, color=BLUE, label="logística", density=False)
    ax.hist(o["BERN_1"], bins=bins, alpha=0.55, color=C40, label="NB Bernoulli α=1", density=False)
    ax.set(xlabel="probabilidad OOF de spam", ylabel="correos", yscale="log")
    ax.legend(fontsize=7.8, frameon=False, loc="upper center")
    titulo(ax, "NB empuja las probabilidades hacia 0 y 1")
    style(ax)
    # predicción alta: el grupo 0.9-1
    p1 = o["BERN_1"]
    mk = p1 >= 0.9
    NUM["comparacion"] = dict(oof={c: met[c] for c, _ in nombres}, calib=cal_tab, bern1_top=dict(n=int(mk.sum()), pred_media=float(p1[mk].mean()), spam_obs=float(y[mk].mean()), spam_n=int(y[mk].sum())),
                              bern1_ext=float(np.mean((p1 < 0.01) | (p1 > 0.99))), logit_ext=float(np.mean((o["LOGIT57"] < 0.01) | (o["LOGIT57"] > 0.99))),
                              c38=json.loads((Path(__file__).resolve().parent / "c38_numeros.json").read_text(encoding="utf-8"))["spam"]["ranking"])
    fig.tight_layout()
    save(fig, "c40_calibracion.png")
    _DAT["oof"] = o


# ----------------------------------------------------------------------------- 11. ROC y umbral por costo
def umbral_fig():
    d = datos()
    y, yte = d["y"], d["yte"]
    o = recetas_oof()
    modelos = {"NB Bernoulli α=0.5": o["BERN_05"], "NB gaussiano": o["GAUSS_LOG"], "Logística (57)": o["LOGIT57"]}
    umb = np.arange(1, 20) / 20
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    ax = axs[0]
    for (nom, p), col in zip(modelos.items(), [C40, VIOLET, BLUE]):
        fpr, tpr, _ = roc_curve(y, p.lp[:, 1] - p.lp[:, 0])
        ax.plot(fpr, tpr, color=col, label=f"{nom}: AUC {medir(y, p)['auc']:.3f}")
    ax.plot([0, 1], [0, 1], ":", color=MUTED)
    ax.set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(ax, "ROC OOF: el ordenamiento es parecido")
    style(ax)
    ax = axs[1]
    costos = {}
    for (nom, p), col in zip(list(modelos.items())[::2], [C40, BLUE]):
        c = [medir(y, p, t)["costo_1000"] for t in umb]
        costos[nom] = c
        ax.plot(umb, c, "o-", color=col, ms=3.5, label=nom)
        j = int(np.argmin(c[::-1]))
        jmin = len(c) - 1 - j      # empates: se elige el mayor umbral
        ax.scatter([umb[jmin]], [c[jmin]], s=110, facecolors="none", edgecolors=INK, lw=1.6)
        ax.text(umb[jmin], c[jmin] + 18, f"mín. {c[jmin]:.1f}\n(t = {umb[jmin]:.2f})", ha="center", fontsize=7.6, color=col)
    ax.axvline(5 / 6, color=AMBER, ls="--")
    ax.text(5 / 6 - 0.01, 470, "t* = 5/6\n(probabilidades\ncalibradas)", fontsize=7.4, color=AMBER, ha="right")
    ax.set(xlabel="umbral", ylabel="costo OOF por 1000 correos", ylim=(150, 560))
    ax.legend(fontsize=7.8, frameon=False, loc="upper right", bbox_to_anchor=(0.78, 1.0))
    titulo(ax, "Umbral por costo (FP = 5, FN = 1)")
    style(ax)
    ax = axs[2]
    # prueba histórica con el NB elegido: Bernoulli α=0.5 con umbral 0.95
    m = bern(0.5).fit(d["X"], y)
    lpt = m.predict_log_proba(d["Xte"])
    pte = con_lp(np.exp(lpt[:, 1]), lpt)
    rt = medir(yte, pte, 0.95)
    cm = np.array([[rt["VN"], rt["FP"]], [rt["FN"], rt["VP"]]])
    ax.imshow(cm, cmap="Reds", vmin=0, vmax=cm.max() * 1.4)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=14, color="white" if v > 300 else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred. legítimo", "pred. spam"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["real legítimo", "real spam"])
    ax.grid(False)
    base = 1000 * yte.sum() / len(yte)
    ax.set_xlabel(f"costo por 1000 = {rt['costo_1000']:.1f}  ·  dejar pasar todo = {base:.1f}", fontsize=8.5)
    titulo(ax, "921 correos históricos, α = 0.5 y umbral 0.95")
    # logística: umbral elegido en OOF y evaluado en prueba
    c_lo = costos["Logística (57)"]
    jl = len(c_lo) - 1 - int(np.argmin(c_lo[::-1]))
    tl = float(umb[jl])
    v57 = d["var"] + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
    ml = logistica().fit(d["tr"][v57].to_numpy(), y)
    lpl = logprobs(ml, d["te"][v57].to_numpy())
    pl = con_lp(np.exp(lpl[:, 1]), lpl)
    rl = medir(yte, pl, tl)
    c_nb = costos["NB Bernoulli α=0.5"]
    jn = len(c_nb) - 1 - int(np.argmin(c_nb[::-1]))
    NUM["umbral"] = dict(umbrales=umb.tolist(), costo_nb=c_nb, costo_logit=c_lo, t_nb=float(umb[jn]), c_nb_min=float(c_nb[jn]), t_logit=tl, c_logit_min=float(c_lo[jl]),
                         t_teorico=5 / 6, prueba_nb=rt, prueba_logit=rl, base_prueba=float(base),
                         fp_oof_nb=medir(y, o["BERN_05"], float(umb[jn]))["FP"], fn_oof_nb=medir(y, o["BERN_05"], float(umb[jn]))["FN"],
                         auc={n: float(medir(y, p)["auc"]) for n, p in modelos.items()}, prueba_nb_ll=float(medir(yte, pte)["log_loss"]), prueba_nb_brier=float(medir(yte, pte)["brier"]))
    fig.tight_layout()
    save(fig, "c40_umbral.png")


# ----------------------------------------------------------------------------- 12. Gaussiana sobre columnas 0/1
def gauss_bin_fig():
    d = datos()
    X, y, f = d["X"], d["y"], d["fold"]
    Xb = (X > 0).astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    # (a) una columna binaria rara: palabra 3d
    nom = "word_freq_3d"
    j = d["var"].index(nom)
    p0, p1 = Xb[y == 0, j].mean(), Xb[y == 1, j].mean()
    ax = axs[0]
    ax.bar([-0.12, 0.88], [1 - p0, p0], 0.22, color=BLUE, label="legítimo: Bernoulli (pmf)")
    ax.bar([0.12, 1.12], [1 - p1, p1], 0.22, color=C40, label="spam: Bernoulli (pmf)")
    gx = np.linspace(-0.4, 1.4, 400)
    sd0, sd1 = np.sqrt(p0 * (1 - p0) + 1e-9), np.sqrt(p1 * (1 - p1) + 1e-9)
    ax.plot(gx, np.exp(-((gx - p0) / sd0) ** 2 / 2) / (sd0 * np.sqrt(2 * np.pi)) / 40, color=BLUE, lw=1.6, label="legítimo: normal (÷40, solo forma)")
    ax.plot(gx, np.exp(-((gx - p1) / sd1) ** 2 / 2) / (sd1 * np.sqrt(2 * np.pi)) / 40, color=C40, lw=1.6, ls="--", label="spam: normal (÷40)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ausente (0)", "presente (1)"])
    ax.set(ylabel="probabilidad / densidad escalada", ylim=(0, 1.05))
    ax.legend(fontsize=6.9, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    titulo(ax, "Palabra «3d» (muy rara): la normal es un pico en p")
    style(ax, "y")
    dens_ll = lambda x, m, s2: -0.5 * np.log(2 * np.pi * s2) - (x - m) ** 2 / (2 * s2)
    s2_0, s2_1 = p0 * (1 - p0) + 1e-9 * Xb.var(0).max(), p1 * (1 - p1) + 1e-9 * Xb.var(0).max()
    # (b) log-razón por columna cuando la señal está presente
    ax = axs[1]
    th = np.vstack([Xb[y == 0].mean(0), Xb[y == 1].mean(0)])
    var_g = np.vstack([Xb[y == 0].var(0), Xb[y == 1].var(0)]) + 1e-9 * Xb.var(0).max()
    lr_b = np.log(th[1] + 1e-12) - np.log(th[0] + 1e-12)
    lr_g = dens_ll(1.0, th[1], var_g[1]) - dens_ll(1.0, th[0], var_g[0])
    ax.scatter(lr_b, lr_g, s=22, color=C40, alpha=0.85)
    lim = [-1000, 300]
    ax.plot(lim, lim, ":", color=MUTED)
    ax.set(xlabel="log-razón Bernoulli si la señal está presente", ylabel="log-razón normal si está presente", xscale="symlog", yscale="symlog")
    for nm in ["word_freq_3d", "word_freq_telnet", "word_freq_857"]:
        jj = d["var"].index(nm)
        ax.annotate(nm.replace("word_freq_", ""), (lr_b[jj], lr_g[jj]), xytext=(4, 3), textcoords="offset points", fontsize=7.6)
    titulo(ax, "Aporte de «presente» a favor de spam, por columna")
    style(ax)
    # (c) desempeño OOF
    ax = axs[2]
    recetas = [("Bernoulli α=1", bern(1.0), None), ("Gaussiano (0/1), var_smoothing 1e-9", GaussianNB(var_smoothing=1e-9), "b"),
               ("Gaussiano (0/1), var_smoothing 1e-2", GaussianNB(var_smoothing=1e-2), "b"), ("Gaussiano log(1 + x)", GaussianNB(var_smoothing=1e-9), "l")]
    res = []
    for nm, m, t in recetas:
        if t == "b":
            p = oof(m, Xb, y, f)
        elif t == "l":
            p = oof(m, np.log1p(X), y, f)
        else:
            p = oof(m, X, y, f)
        mm = medir(y, p)
        res.append(dict(modelo=nm, **{k: mm[k] for k in ("log_loss", "brier", "auc", "accuracy")}))
    nombres = [r["modelo"] for r in res]
    ax.barh(nombres, [min(r["log_loss"], 8) for r in res], color=[C40, VIOLET, VIOLET, VIOLET])
    for i, r in enumerate(res):
        ax.text(min(r["log_loss"], 8) + 0.08, i, f"{r['log_loss']:.2f} (AUC {r['auc']:.3f})", va="center", fontsize=7.8)
    ax.set_xlim(0, 11)
    ax.invert_yaxis()
    ax.set(xlabel="log loss OOF")
    titulo(ax, "El gaussiano sobre 0/1 ordena, pero calibra mal")
    style(ax, "x")
    NUM["gaussbin"] = dict(col=nom, p0=float(p0), p1=float(p1), n_presente=[int(Xb[y == 0, j].sum()), int(Xb[y == 1, j].sum())], var0=float(var_g[0][j]), var1=float(var_g[1][j]),
                           lr_bern_3d=float(lr_b[j]), lr_gauss_3d=float(lr_g[j]), lr_gauss_min=float(lr_g.min()), res=res,
                           media_var=[float(Xb[y == 0].var(0).mean()), float(Xb[y == 1].var(0).mean())])
    fig.tight_layout()
    save(fig, "c40_gauss_binarias.png")


# ----------------------------------------------------------------------------- 13. mapa de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    pasos = [("1. Representar", "54 frecuencias →\npresencia (Bernoulli)\no log(1 + x)"), ("2. Aprender", "prior por clase y\nfactores por señal\n(con suavizado α)"),
             ("3. Predecir", "multiplicar (sumar\nlogs) y normalizar:\nP(spam | señales)"), ("4. Comprobar", "OOF: log loss,\nBrier, AUC y\ncalibración"), ("5. Decidir", "umbral por costo\n(FP = 5, FN = 1)\ny prueba histórica")]
    for i, (t, s) in enumerate(pasos):
        x0 = 0.1 + i * 2.5
        ax.add_patch(FancyBboxPatch((x0, 4.1), 2.25, 1.75, boxstyle="round,pad=0.03", fc="white", ec=C40, lw=1.6))
        ax.text(x0 + 1.125, 5.5, t, ha="center", fontsize=10, fontweight="bold", color=C40)
        ax.text(x0 + 1.125, 4.85, s, ha="center", va="center", fontsize=8.4, color=INK)
    for xx in (2.4, 4.9, 7.4, 9.9):
        ax.annotate("", xy=(xx + 0.15, 4.98), xytext=(xx, 4.98), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 3.7, boxstyle="round,pad=0.03", fc=C40_BG, ec=C40, lw=1.4))
    bz, ig, ga, lp, cp, um, gb, fr = NUM["bayes"], NUM["indep"], NUM["gauss"], NUM["laplace"], NUM["comparacion"], NUM["umbral"], NUM["gaussbin"], NUM["dup"]
    ax.text(6.25, 3.5, "Datos de la clase: spam_s40.csv · 4 601 correos de Spambase, 54 frecuencias · 3 680 de desarrollo (5 pliegues) y 921 de prueba histórica", ha="center", fontsize=8.4, color=C40, fontweight="bold")
    ax.text(6.25, 3.05, f"Ejemplo de 100 correos: P(spam | oferta) = {bz['post_oferta']:.4f}; con 2 % de spam baja a {bz['post_prior_002']:.4f}; oferta sin enlace: {ig['post'][0]:.4f} sin suavizar y {ig['post'][1]:.4f} con α = 1", ha="center", fontsize=8.1, color=INK)
    o = cp["oof"]
    ax.text(6.25, 2.65, f"Log loss OOF: frecuencia {o['FRECUENCIA']['log_loss']:.3f} · Bernoulli α = 0.5 {o['BERN_05']['log_loss']:.3f} · gaussiana log(1 + x) {o['GAUSS_LOG']['log_loss']:.3f} · logística (57) {o['LOGIT57']['log_loss']:.3f}", ha="center", fontsize=8.1, color=INK)
    ax.text(6.25, 2.25, f"AUC OOF: Bernoulli {o['BERN_05']['auc']:.3f} frente a logística {o['LOGIT57']['auc']:.3f}: ordenan parecido, pero NB calibra peor (en el grupo ≥ 0.9 predice {cp['bern1_top']['pred_media'] * 100:.1f} % y observa {cp['bern1_top']['spam_obs'] * 100:.1f} %)", ha="center", fontsize=8.1, color=INK)
    ax.text(6.25, 1.85, f"Umbral por costo: {um['t_nb']:.2f} (costo OOF {um['c_nb_min']:.1f}); en 921 correos: FP {um['prueba_nb']['FP']}, FN {um['prueba_nb']['FN']}, costo {um['prueba_nb']['costo_1000']:.1f} frente a {um['base_prueba']:.1f} si todo pasa", ha="center", fontsize=8.1, color=INK)
    ax.text(6.25, 1.45, f"Duplicar las 54 señales 4 veces: log loss de {fr['real'][0]['log_loss']:.3f} a {fr['real'][2]['log_loss']:.3f}, AUC {fr['real'][0]['auc']:.3f} → {fr['real'][2]['auc']:.3f}", ha="center", fontsize=8.1, color=INK)
    ax.text(6.25, 0.95, "R ↔ Python:  naivebayes::bernoulli_naive_bayes() ↔ BernoulliNB(alpha) · gaussian_naive_bayes() ↔ GaussianNB · multinomial_naive_bayes() ↔ MultinomialNB · laplace ↔ alpha", ha="center", fontsize=7.8, color=MUTED, style="italic")
    ax.text(6.25, 0.5, "predict(type = 'prob') ↔ predict_proba · MASS::lda / qda ↔ LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis · glm(binomial) ↔ LogisticRegression", ha="center", fontsize=7.8, color=MUTED, style="italic")
    save(fig, "c40_esquema.png")


# ----------------------------------------------------------------------------- 14. cálculos extra para el notebook (sin figura)
def extras():
    """Una sola variable categórica (carácter «!» en tres tramos) con tres modelos distintos: sirve para mostrar el límite del gaussiano sobre dummies."""
    from sklearn.naive_bayes import CategoricalNB
    d = datos()
    j = d["var"].index("char_freq_!")
    cut = [1e-4, 0.3]
    c = np.digitize(d["X"][:, j], cut)
    cte = np.digitize(d["Xte"][:, j], cut)
    oh = np.eye(3)[c]
    ohte = np.eye(3)[cte]
    y = d["y"]
    emp = [float(y[c == k].mean()) for k in range(3)]
    cat = CategoricalNB(alpha=1.0).fit(c[:, None], y).predict_proba(np.arange(3)[:, None])[:, 1]
    be = BernoulliNB(alpha=1.0).fit(oh, y).predict_proba(np.eye(3))[:, 1]
    ga = GaussianNB().fit(oh, y).predict_proba(np.eye(3))[:, 1]
    res = {}
    for nom, m, xa, xt in [("CategoricalNB", CategoricalNB(alpha=1.0), c[:, None], cte[:, None]), ("BernoulliNB (dummies)", BernoulliNB(alpha=1.0), oh, ohte), ("GaussianNB (dummies)", GaussianNB(), oh, ohte)]:
        m.fit(xa, y)
        res[nom] = medir(d["yte"], m.predict_proba(xt)[:, 1])
    NUM["onehot"] = dict(n=[int((c == k).sum()) for k in range(3)], empirica=emp, categorical=[float(v) for v in cat], bernoulli=[float(v) for v in be], gauss=[float(v) for v in ga],
                         prueba={k: dict(log_loss=v["log_loss"], auc=v["auc"]) for k, v in res.items()})

PASOS = dict(bayes=bayes_fig, indep=indep_fig, dup=dup_fig, gauss=gauss_fig, bernoulli=bernoulli_fig, variantes=variantes_fig, laplace=laplace_fig,
             logprob=logprob_fig, fronteras=fronteras_fig, calibracion=calibracion_fig, umbral=umbral_fig, gaussbin=gauss_bin_fig, extras=extras, esquema=esquema_fig)

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c40_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or list(PASOS)
    if "esquema" in elegidos:
        for k in ("bayes", "indep", "gauss", "laplace", "calibracion", "umbral", "gaussbin", "dup"):
            if k not in elegidos and {"calibracion": "comparacion"}.get(k, k) not in NUM:
                elegidos.append(k)
    orden = [e for e in PASOS if e in elegidos and e != "esquema"]
    orden += (["esquema"] if "esquema" in elegidos else [])
    for k in orden:
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c40_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
