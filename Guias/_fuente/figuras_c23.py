"""Figuras de la Clase 23 (Estadística bayesiana: fundamentos, primera sesión).
Fuentes: diapositivas Clase 23.pdf, Clase23.ipynb y «Práctica de Estadística Bayesiana en R» de la profesora Maritza Márquez. La clase no trae datos: todo es analítico
(prueba diagnóstica, 14 aprobados de 20 con priori Beta(2, 2)). Para el puente con la Clase 22 se lee ab_checkout.csv (22 000 usuarios) de la carpeta «Clase 22 …» sin modificarla.
Todo sale de código; lo aleatorio usa semilla fija. Escribe c23_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c23.py   (o con nombres: python -X utf8 figuras_c23.py bayes betabinom)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from scipy.special import betaln, comb

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C23 = "#059669"      # color de la clase 23 (verde, igual que el roadmap)
C23_BG = "#ECFDF5"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA22 = next(p for p in BASE.glob("Clase 22*") if p.is_dir())

Y, N_ = 14, 20           # datos de la profesora: 14 aprobados de 20
A0, B0 = 2, 2            # priori Beta(2, 2)
AP, BP = A0 + Y, B0 + N_ - Y
TH = np.linspace(0.0005, 0.9995, 2000)


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def post_bayes(prev, sens, esp):
    """P(enfermo | positivo) por Bayes."""
    p_pos = sens * prev + (1 - esp) * (1 - prev)
    return sens * prev / p_pos


def hpd(a, b, masa=0.95):
    """Intervalo de mayor densidad posterior (HPD) de una Beta unimodal (a, b > 1) o con moda en el borde."""
    d = stats.beta(a, b)
    f = lambda l: d.ppf(l + masa) - d.ppf(l)
    l = optimize.minimize_scalar(f, bounds=(0, 1 - masa), method="bounded", options={"xatol": 1e-10}).x
    lo, hi = d.ppf(l), d.ppf(l + masa)
    # si el borde es mejor (moda en 0 o en 1), lo respeta
    if f(0) <= f(l) + 1e-12:
        lo, hi = d.ppf(0), d.ppf(masa)
    if f(1 - masa) <= f(l) + 1e-12 and f(1 - masa) < f(0):
        lo, hi = d.ppf(1 - masa), d.ppf(1)
    return float(lo), float(hi)


def resumen_beta(a, b):
    d = stats.beta(a, b)
    moda = (a - 1) / (a + b - 2) if (a > 1 and b > 1) else (0.0 if a <= 1 and b > 1 else (1.0 if b <= 1 and a > 1 else float("nan")))
    return dict(a=a, b=b, media=float(d.mean()), mediana=float(d.median()), moda=float(moda), sd=float(d.std()),
                ic=[float(d.ppf(0.025)), float(d.ppf(0.975))], hpd=list(hpd(a, b)))


# ------------------------------------------------------------------ 0. esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    def caja(x, y, w, h, color, titulo_, lineas, fs=7.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.2, titulo_, ha="center", va="center", fontsize=9.4, color=color, fontweight="bold")
        for i, t in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.52 - i * 0.28, t, ha="center", va="center", fontsize=fs, color=INK)

    caja(0.1, 2.75, 2.45, 1.85, MUTED, "1 · Dos enfoques", ["frecuentista: θ fijo,", "los datos varían", "bayesiano: θ incierto,", "los datos observados son fijos"])
    caja(2.85, 2.75, 2.35, 1.85, BLUE, "2 · Teorema de Bayes", ["P(A|B) = P(B|A)·P(A) / P(B)", "prueba diagnóstica:", "P(E|+) ≠ sensibilidad", "prevalencia = información previa"])
    caja(5.5, 2.75, 2.45, 1.85, AMBER, "3 · El modelo", ["posteriori proporcional a", "verosimilitud × priori", "la evidencia P(y) solo", "normaliza (integra 1)"])
    caja(8.25, 2.75, 2.1, 1.85, C23, "4 · Beta–Binomial", ["θ ~ Beta(α, β)", "y | θ ~ Bin(n, θ)", "θ | y ~ Beta(α+y, β+n−y)", "conjugación"])
    caja(10.65, 2.75, 1.75, 1.85, VIOLET, "5 · Resumir", ["media · moda (MAP)", "intervalo creíble", "P(θ > 0.6 | y)", ""])
    for x in (2.6, 5.25, 8.0, 10.4):
        ax.annotate("", xy=(x + 0.25, 3.65), xytext=(x, 3.65), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 2.35, boxstyle="round,pad=0.03", fc=C23_BG, ec=C23, lw=1.4))
    d = NUM["diagnostico"]
    r = NUM["betabinom"]
    ax.text(6.25, 2.15, "Datos de la clase: ninguno externo · prueba diagnóstica (prevalencia 1 %, sensibilidad 95 %, especificidad 90 %) · 14 aprobados de 20 con priori Beta(2, 2)", ha="center", va="center", fontsize=8.9, color=C23, fontweight="bold")
    ax.text(6.25, 1.7, f"Prueba diagnóstica: P(enfermo | positivo) = {d['post']:.4f} ({d['post'] * 100:.1f} %), no 95 %  ·  Beta–Binomial: posteriori Beta({AP}, {BP}), media {r['media']:.4f}, IC creíble 95 % [{r['ic'][0]:.4f}; {r['ic'][1]:.4f}]", ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(6.25, 1.32, f"P(θ > 0.6 | y) = {r['p_mayor_06']:.4f} (exacto) · la práctica en R, con 10 000 simulaciones, dio 0.7581 · puente a la Clase 22: A/B bayesiano con P(B > A) y P(mejora > MDE)", ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(6.25, 0.85, "R ↔ Python:  dbeta ↔ stats.beta.pdf  ·  pbeta ↔ stats.beta.cdf  ·  qbeta ↔ stats.beta.ppf  ·  rbeta ↔ stats.beta.rvs  ·  dbinom ↔ stats.binom.pmf", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    ax.text(6.25, 0.45, "set.seed(123) ↔ np.random.default_rng(123) (los números aleatorios de R y de NumPy no coinciden: se comparan con el valor exacto)", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    save(fig, "c23_esquema.png")


# ------------------------------------------------------------------ 1. Teorema de Bayes con frecuencias naturales
def bayes_fig():
    prev, sens, esp = 0.01, 0.95, 0.90
    n = 10000
    enf, san = int(n * prev), int(n * (1 - prev))
    tp, fn = int(round(enf * sens)), enf - int(round(enf * sens))
    fp = int(round(san * (1 - esp)))
    tn = san - fp
    post = tp / (tp + fp)
    p_pos = sens * prev + (1 - esp) * (1 - prev)
    post_ex = sens * prev / p_pos
    NUM["diagnostico"] = dict(prev=prev, sens=sens, esp=esp, n=n, enfermos=enf, sanos=san, vp=tp, fn=fn, fp=fp, vn=tn, positivos=tp + fp, p_pos=round(p_pos, 6),
                              post=round(post_ex, 7), post_nat=round(post, 7), p_neg_dado_neg=round(tn / (tn + fn), 6),
                              post_esp95=round(post_bayes(prev, sens, 0.95), 4), post_esp99=round(post_bayes(prev, sens, 0.99), 4),
                              post_prev10=round(post_bayes(0.10, sens, esp), 4), post_prev50=round(post_bayes(0.50, sens, esp), 4))
    # segundo test positivo independiente: la posteriori pasa a ser la nueva priori
    p2 = post_bayes(post_ex, sens, esp)
    NUM["diagnostico"]["post_segundo_test"] = round(p2, 4)
    NUM["diagnostico"]["post_tres_tests"] = round(post_bayes(p2, sens, esp), 4)
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.05], wspace=0.28)
    a = fig.add_subplot(gs[0])
    a.set_xlim(-0.3, 10.4)
    a.set_ylim(0, 8.2)
    a.axis("off")

    def caja(x, y, w, h, txt, col, fs=8.0, fc="white"):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=col, lw=1.8))
        a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=col, fontweight="bold")
    caja(3.3, 7.0, 3.4, 1.0, f"{n:,} personas".replace(",", " "), INK, 9)
    caja(0.2, 4.7, 3.6, 1.2, f"{enf} enfermos\n(prevalencia 1 %)", RED, 8.4)
    caja(6.2, 4.7, 3.6, 1.2, f"{san:,} sanos".replace(",", " "), BLUE, 8.4)
    caja(0.0, 2.4, 2.3, 1.0, f"{tp} dan +", RED, 8.2, "#FEF2F2")
    caja(2.55, 2.4, 2.3, 1.0, f"{fn} dan −", MUTED, 8.2)
    caja(5.2, 2.4, 2.3, 1.0, f"{fp} dan +", AMBER, 8.2, "#FFFBEB")
    caja(7.75, 2.4, 2.3, 1.0, f"{tn:,} dan −".replace(",", " "), MUTED, 8.2)
    a.text(1.2, 2.05, "sensibilidad 95 %", ha="center", fontsize=7.4, color=RED)
    a.text(6.35, 2.05, "1 − especificidad = 10 %", ha="center", fontsize=7.4, color=AMBER)
    for (x0, y0, x1, y1) in [(5, 7.0, 2.0, 5.95), (5, 7.0, 8.0, 5.95), (1.9, 4.7, 1.2, 3.45), (1.9, 4.7, 3.7, 3.45), (8.0, 4.7, 6.35, 3.45), (8.0, 4.7, 8.9, 3.45)]:
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    a.add_patch(FancyBboxPatch((0.0, 0.1), 10, 1.5, boxstyle="round,pad=0.05", fc=C23_BG, ec=C23, lw=1.6))
    a.text(5, 1.15, f"Entre los {tp + fp:,} positivos, solo {tp} están enfermos".replace(",", " "), ha="center", fontsize=8.4, color=C23, fontweight="bold")
    a.text(5, 0.6, f"P(E | +) = {tp} / {tp + fp} = {post * 100:.1f} %", ha="center", fontsize=9.2, color=INK, fontweight="bold")
    titulo(a, "Frecuencias naturales: contar personas, no multiplicar probabilidades")
    # (b) la barra de positivos
    b = fig.add_subplot(gs[1])
    b.barh([1], [tp], color=RED, alpha=0.85)
    b.barh([1], [fn], left=[tp], color=MUTED, alpha=0.5)
    b.barh([0], [tp], color=RED, alpha=0.85)
    b.barh([0], [fp], left=[tp], color=AMBER, alpha=0.85)
    b.set_yticks([0, 1])
    b.set_yticklabels(["entre quienes\ndan +", "entre los\nenfermos"], fontsize=8.4)
    b.text(tp + fn + 15, 1, f"{tp} de {tp + fn} dan +  =  {sens * 100:.0f} %  (sensibilidad)", va="center", fontsize=7.8, color=RED)
    b.text(tp + fp / 2, 0, f"{(1 - post) * 100:.1f} % sanos", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")
    b.text(tp + fp + 15, 0.0, f"{post * 100:.1f} %\nenfermos", va="center", fontsize=7.8, color=RED, fontweight="bold")
    b.set_xlabel("personas")
    b.set_xlim(0, 1700)
    b.set_ylim(-0.6, 1.6)
    titulo(b, "P(E|+) y P(+|E) miran grupos distintos")
    style(b, "x")
    # (c) posteriori frente a prevalencia
    c = fig.add_subplot(gs[2])
    pr = np.logspace(-3, np.log10(0.6), 200)
    for esp_, col in [(0.90, AMBER), (0.95, BLUE), (0.99, C23)]:
        c.plot(pr * 100, [post_bayes(p, sens, esp_) * 100 for p in pr], color=col, lw=2.2, label=f"especificidad {esp_ * 100:.0f} %")
        c.scatter([1], [post_bayes(0.01, sens, esp_) * 100], color=col, zorder=4, s=34)
    c.axhline(95, color=INK, ls=":", lw=1)
    c.text(0.12, 91, "sensibilidad = 95 %", fontsize=7.8, color=INK)
    c.set_xscale("log")
    c.set_xlabel("prevalencia (%) , escala log")
    c.set_ylabel("P(enfermo | positivo)  (%)")
    c.legend(fontsize=7.8, frameon=False, loc="lower right")
    titulo(c, "Misma prueba, distinta prevalencia (puntos: 1 %)")
    style(c)
    save(fig, "c23_bayes.png")


# ------------------------------------------------------------------ 2. Beta–Binomial: priori, verosimilitud, posteriori
def betabinom_fig():
    r = resumen_beta(AP, BP)
    d = stats.beta(AP, BP)
    p06 = float(d.sf(0.6))
    rng = np.random.default_rng(123)
    S = 10000
    sim = d.rvs(S, random_state=rng)
    mc = float((sim > 0.6).mean())
    evid = float(comb(N_, Y) * np.exp(betaln(AP, BP) - betaln(A0, B0)))     # p(y) = C(n, y) B(α+y, β+n−y) / B(α, β)
    like = stats.binom.pmf(Y, N_, TH)
    evid_grid = float(integrate.trapezoid(like * stats.beta(A0, B0).pdf(TH), TH))
    r.update(p_mayor_06=round(p06, 7), p_mayor_06_mc=round(mc, 4), mc_se=round(float(np.sqrt(p06 * (1 - p06) / S)), 4), mc_semilla="np.random.default_rng(123)",
             mle=Y / N_, peso_priori=(A0 + B0) / (A0 + B0 + N_), media_priori=A0 / (A0 + B0), sd_priori=float(stats.beta(A0, B0).std()),
             evidencia=evid, evidencia_grilla=evid_grid, prof_mc=0.7581, prof_ic=[0.4708083, 0.8362364], prof_media=0.6666667,
             ancho_ic=r["ic"][1] - r["ic"][0], ancho_ic_priori=float(stats.beta(A0, B0).ppf(0.975) - stats.beta(A0, B0).ppf(0.025)))
    NUM["betabinom"] = {k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}
    NUM["betabinom"]["ic"] = [round(v, 7) for v in r["ic"]]
    NUM["betabinom"]["hpd"] = [round(v, 7) for v in r["hpd"]]
    # el factor (n sobre y) y la forma de la verosimilitud
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    pri = stats.beta(A0, B0).pdf(TH)
    a.fill_between(TH, pri, color=BLUE, alpha=0.25)
    a.plot(TH, pri, color=BLUE, lw=2.4)
    a.axvline(0.5, color=BLUE, ls=":", lw=1.4)
    a.set_xlabel("θ (proporción que aprueba)")
    a.set_ylabel("densidad")
    a.set_ylim(0, 4.2)
    titulo(a, f"1. Priori Beta({A0}, {B0}): media 0.50, mucha incertidumbre")
    style(a)
    b = fig.add_subplot(gs[1])
    lik_n = like / integrate.trapezoid(like, TH)
    b.fill_between(TH, lik_n, color=AMBER, alpha=0.25)
    b.plot(TH, lik_n, color=AMBER, lw=2.4)
    b.axvline(Y / N_, color=AMBER, ls=":", lw=1.4)
    b.text(Y / N_ + 0.015, lik_n.max() * 0.95, "máximo en 14/20 = 0.70", fontsize=8, color=AMBER)
    b.set_xlabel("θ")
    b.set_ylabel("verosimilitud (reescalada para ver su forma)")
    b.set_ylim(0, 4.2)
    titulo(b, "2. Verosimilitud θ¹⁴(1 − θ)⁶: compatibilidad de los datos")
    style(b)
    c = fig.add_subplot(gs[2])
    c.plot(TH, pri, color=BLUE, lw=1.8, ls="--", label=f"priori Beta({A0}, {B0})")
    pos = d.pdf(TH)
    c.plot(TH, pos, color=C23, lw=2.8, label=f"posteriori Beta({AP}, {BP})")
    msk = (TH >= r["ic"][0]) & (TH <= r["ic"][1])
    c.fill_between(TH[msk], pos[msk], color=C23, alpha=0.20, label=f"IC creíble 95 %: [{r['ic'][0]:.2f}; {r['ic'][1]:.2f}]")
    m6 = TH >= 0.6
    c.fill_between(TH[m6], pos[m6], color=VIOLET, alpha=0.18, label=f"P(θ > 0.6 | y) = {p06:.3f}")
    for v, col, lab in [(r["media"], C23, "media"), (r["moda"], RED, "MAP")]:
        c.axvline(v, color=col, ls=":", lw=1.5)
    c.text(r["media"] - 0.012, 4.5, f"media {r['media']:.3f}", fontsize=7.6, color=C23, ha="right")
    c.text(r["moda"] + 0.012, 4.2, f"MAP {r['moda']:.3f}", fontsize=7.6, color=RED)
    c.set_xlabel("θ")
    c.set_ylim(0, 6.3)
    c.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(c, "3. Posteriori = verosimilitud × priori (normalizada): se desplaza y se estrecha")
    style(c)
    save(fig, "c23_betabinom.png")


# ------------------------------------------------------------------ 3. efecto de la priori y del tamaño de muestra
def prioris_fig():
    pri = [("Uniforme Beta(1, 1)", 1, 1, BLUE), ("Moderada Beta(2, 2)", 2, 2, C23), ("Optimista Beta(8, 2)", 8, 2, AMBER)]
    tabla = {}
    for y_ in (14, 8):
        filas = []
        for nom, a_, b_, _ in pri:
            ap, bp = a_ + y_, b_ + N_ - y_
            d_ = stats.beta(ap, bp)
            filas.append(dict(nombre=nom, a=a_, b=b_, a_post=ap, b_post=bp, media=round(float(d_.mean()), 7), li=round(float(d_.ppf(0.025)), 7), ls=round(float(d_.ppf(0.975)), 7),
                              p06=round(float(d_.sf(0.6)), 7), peso_priori=round((a_ + b_) / (a_ + b_ + N_), 4)))
        tabla[f"y{y_}"] = filas
    NUM["prioris"] = tabla
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    for k, (y_, ax_pos) in enumerate([(14, 0), (8, 1)]):
        a = fig.add_subplot(gs[ax_pos])
        for nom, a_, b_, col in pri:
            a.plot(TH, stats.beta(a_, b_).pdf(TH), color=col, lw=1.3, ls="--", alpha=0.8)
            a.plot(TH, stats.beta(a_ + y_, b_ + N_ - y_).pdf(TH), color=col, lw=2.6, label=f"{nom.split(' ')[0]} → media {(a_ + y_) / (a_ + b_ + N_):.3f}")
        a.axvline(y_ / N_, color=INK, ls=":", lw=1)
        a.set_xlabel("θ")
        a.set_ylabel("densidad")
        a.set_ylim(0, 6.2)
        a.legend(fontsize=7.6, frameon=False, loc="upper left")
        titulo(a, f"{y_} aprobados de {N_}: trazo fino = priori, grueso = posteriori")
        style(a)
    c = fig.add_subplot(gs[2])
    ns = np.unique(np.round(np.logspace(0.7, 3.3, 40)).astype(int))
    for nom, a_, b_, col in [("Uniforme Beta(1, 1)", 1, 1, BLUE), ("Optimista Beta(8, 2)", 8, 2, AMBER)]:
        yy = np.round(0.7 * ns)
        mu = (a_ + yy) / (a_ + b_ + ns)
        lo = stats.beta.ppf(0.025, a_ + yy, b_ + ns - yy)
        hi = stats.beta.ppf(0.975, a_ + yy, b_ + ns - yy)
        c.plot(ns, mu, color=col, lw=2.4, label=nom.split(" ")[0])
        c.fill_between(ns, lo, hi, color=col, alpha=0.15)
    c.axhline(0.7, color=INK, ls=":", lw=1)
    c.set_xscale("log")
    c.set_xlabel("tamaño de muestra n  (con 70 % de éxitos)")
    c.set_ylabel("media posteriori y IC creíble 95 %")
    c.legend(fontsize=7.8, frameon=False, loc="lower right")
    titulo(c, "Con más datos las prioris distintas convergen")
    style(c)
    save(fig, "c23_prioris.png")
    # comparación numérica de convergencia
    NUM["convergencia"] = {str(n_): dict(unif=round(float((1 + round(0.7 * n_)) / (2 + n_)), 4), optimista=round(float((8 + round(0.7 * n_)) / (10 + n_)), 4)) for n_ in (10, 20, 100, 1000)}


# ------------------------------------------------------------------ 4. actualización secuencial
def secuencial_fig():
    lotes = [(4, 3), (6, 4), (5, 3), (5, 4)]     # (n, éxitos) por llegada
    a, b = A0, B0
    camino = [dict(paso=0, n_acum=0, y_acum=0, a=a, b=b, media=a / (a + b), ic=[float(stats.beta(a, b).ppf(.025)), float(stats.beta(a, b).ppf(.975))])]
    n_ac = y_ac = 0
    for i, (n_, y_) in enumerate(lotes, 1):
        a, b = a + y_, b + n_ - y_
        n_ac, y_ac = n_ac + n_, y_ac + y_
        d_ = stats.beta(a, b)
        camino.append(dict(paso=i, n_acum=n_ac, y_acum=y_ac, a=a, b=b, media=float(d_.mean()), ic=[float(d_.ppf(.025)), float(d_.ppf(.975))]))
    assert (a, b) == (AP, BP)
    NUM["secuencial"] = dict(lotes=lotes, camino=[{k: (round(v, 4) if isinstance(v, float) else ([round(x, 4) for x in v] if isinstance(v, list) else v)) for k, v in c.items()} for c in camino],
                             final=[a, b], igual_a_lote=bool((a, b) == (AP, BP)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.9], wspace=0.3)
    cmap = plt.get_cmap("viridis")
    a_ = fig.add_subplot(gs[0])
    for i, c in enumerate(camino):
        col = cmap(i / (len(camino) - 1) * 0.85)
        a_.plot(TH, stats.beta(c["a"], c["b"]).pdf(TH), color=col, lw=2.2 if i in (0, len(camino) - 1) else 1.6,
                label=("priori" if i == 0 else f"tras el lote {i}  (n = {c['n_acum']}, y = {c['y_acum']})"))
    a_.axvline(Y / N_, color=INK, ls=":", lw=1)
    a_.set_xlabel("θ")
    a_.set_ylabel("densidad")
    a_.legend(fontsize=7.2, frameon=False, loc="upper left")
    titulo(a_, "La posteriori de un lote es la priori del siguiente")
    style(a_)
    b_ = fig.add_subplot(gs[1])
    xs = [c["n_acum"] for c in camino]
    b_.errorbar(xs, [c["media"] for c in camino], yerr=[[c["media"] - c["ic"][0] for c in camino], [c["ic"][1] - c["media"] for c in camino]], fmt="o-", color=C23, capsize=4, lw=2, ms=6)
    b_.axhline(Y / N_, color=INK, ls=":", lw=1)
    b_.set_xlabel("estudiantes observados (acumulado)")
    b_.set_ylabel("media posteriori e IC creíble 95 %")
    b_.set_xticks(xs)
    titulo(b_, "Más datos → menos incertidumbre")
    style(b_)
    c_ = fig.add_subplot(gs[2])
    c_.axis("off")
    filas = [["Lote", "n", "y", "(α, β)"]] + [["priori", "", "", f"({A0}, {B0})"]] + [[str(i), str(l[0]), str(l[1]), f"({camino[i]['a']}, {camino[i]['b']})"] for i, l in enumerate(lotes, 1)] + [["todo junto", str(N_), str(Y), f"({AP}, {BP})"]]
    tb = c_.table(cellText=filas, cellLoc="center", loc="center", colWidths=[0.34, 0.16, 0.16, 0.34])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.8)
    tb.scale(1, 1.75)
    for (r_, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r_ == 0:
            cell.set_facecolor(C23)
            cell.set_text_props(color="white", fontweight="bold")
        if r_ == len(filas) - 1:
            cell.set_text_props(fontweight="bold", color=C23)
    titulo(c_, "Por lotes o todo junto: la misma posteriori")
    save(fig, "c23_secuencial.png")


# ------------------------------------------------------------------ 5. resumir la posteriori: media, mediana, MAP, HPD
def resumenes_fig():
    asim = resumen_beta(2, 10)      # 1 éxito de 10 con priori uniforme
    sim = resumen_beta(AP, BP)
    cero = resumen_beta(1, 11)      # 0 éxitos de 10 con priori uniforme
    NUM["resumenes"] = dict(asimetrica=asim, simetrica=sim, cero_exitos=cero)
    NUM["resumenes"]["asimetrica"]["ic"] = [round(v, 4) for v in asim["ic"]]
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    for k, (r, nom, xmax) in enumerate([(asim, "1 éxito de 10, priori uniforme → Beta(2, 10)", 0.6), (sim, "14 de 20, priori Beta(2, 2) → Beta(16, 8)", 1.0)]):
        a = fig.add_subplot(gs[k])
        d_ = stats.beta(r["a"], r["b"])
        xs = np.linspace(0.0005, xmax, 1200)
        a.plot(xs, d_.pdf(xs), color=INK, lw=2)
        a.axvline(r["moda"], color=RED, lw=1.8, ls="-", label=f"MAP (moda) {r['moda']:.3f}")
        a.axvline(r["mediana"], color=BLUE, lw=1.8, ls="--", label=f"mediana {r['mediana']:.3f}")
        a.axvline(r["media"], color=C23, lw=1.8, ls=":", label=f"media {r['media']:.3f}")
        ylo = -0.07 * d_.pdf(xs).max()
        a.plot(r["ic"], [ylo, ylo], color=AMBER, lw=4, solid_capstyle="butt", label=f"IC por colas [{r['ic'][0]:.2f}; {r['ic'][1]:.2f}]")
        a.plot(r["hpd"], [2 * ylo, 2 * ylo], color=VIOLET, lw=4, solid_capstyle="butt", label=f"HPD [{r['hpd'][0]:.2f}; {r['hpd'][1]:.2f}]")
        a.set_xlabel("θ")
        a.set_ylabel("densidad")
        a.legend(fontsize=7.2, frameon=False, loc="upper right" if k == 0 else "upper left")
        titulo(a, nom)
        style(a)
    c = fig.add_subplot(gs[2])
    n = 10
    ys = np.arange(0, n + 1)
    c.plot(ys, ys / n, color=MUTED, lw=2, marker="o", ms=4, label="máx. verosimilitud y/n")
    c.plot(ys, [(1 + y_ - 1) / (2 + n - 2) if y_ > 0 else 0 for y_ in ys], color=RED, lw=2, marker="s", ms=4, ls="--", label="MAP, priori uniforme (= y/n)")
    c.plot(ys, (1 + ys) / (2 + n), color=C23, lw=2.4, marker="o", ms=4, label="media posteriori, priori uniforme")
    c.plot(ys, stats.beta.median(1 + ys, 1 + n - ys), color=BLUE, lw=1.6, ls=":", label="mediana posteriori")
    c.set_xlabel("éxitos y observados (n = 10)")
    c.set_ylabel("estimación de θ")
    c.legend(fontsize=7.2, frameon=False, loc="upper left")
    titulo(c, "MAP con priori plana = y/n; la media «suaviza» los extremos")
    style(c)
    save(fig, "c23_resumenes.png")


# ------------------------------------------------------------------ 6. intervalo creíble frente a intervalo de confianza
def ic_wald(y, n, z=1.959964):
    p = y / n
    s = np.sqrt(p * (1 - p) / n)
    return max(0, p - z * s), min(1, p + z * s)


def ic_wilson(y, n, z=1.959964):
    p = y / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def ic_cp(y, n):
    lo = 0.0 if y == 0 else stats.beta.ppf(0.025, y, n - y + 1)
    hi = 1.0 if y == n else stats.beta.ppf(0.975, y + 1, n - y)
    return lo, hi


def ic_bayes(y, n, a=1, b=1):
    return stats.beta.ppf(0.025, a + y, b + n - y), stats.beta.ppf(0.975, a + y, b + n - y)


def cobertura_exacta(metodo, theta, n):
    ys = np.arange(0, n + 1)
    pm = stats.binom.pmf(ys, n, theta)
    cub = np.array([(lambda iv: iv[0] <= theta <= iv[1])(metodo(y_, n)) for y_ in ys])
    return float((pm * cub).sum())


def intervalos_fig():
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.1], wspace=0.5)
    a = fig.add_subplot(gs[0])
    rng = np.random.default_rng(RS)
    th0, n = 0.65, 20
    ys = rng.binomial(n, th0, 40)
    miss_b = miss_w = 0
    for i, y_ in enumerate(ys):
        lb = ic_bayes(y_, n)
        lw_ = ic_wald(y_, n)
        ok_b, ok_w = lb[0] <= th0 <= lb[1], lw_[0] <= th0 <= lw_[1]
        miss_b += (not ok_b)
        miss_w += (not ok_w)
        a.plot(lb, [i + 0.15, i + 0.15], color=C23 if ok_b else RED, lw=2.4)
        a.plot(lw_, [i - 0.15, i - 0.15], color=MUTED if ok_w else RED, lw=2.4)
    a.axvline(th0, color=INK, ls="--", lw=1.4)
    a.text(th0 + 0.01, 40.6, "θ verdadero = 0.65", fontsize=8, color=INK)
    a.plot([], [], color=C23, lw=2.4, label="creíble Beta(1, 1)")
    a.plot([], [], color=MUTED, lw=2.4, label="confianza de Wald")
    a.set_xlim(0.2, 1.0)
    a.set_ylim(-7, 42)
    a.set_xlabel("θ")
    a.set_ylabel("40 muestras repetidas (n = 20)")
    a.legend(fontsize=7.6, frameon=False, loc="lower left")
    titulo(a, "El intervalo cambia con la muestra; θ no se mueve")
    style(a, "x")
    b = fig.add_subplot(gs[1])
    ths = np.linspace(0.04, 0.96, 231)
    metodos = [("Wald", ic_wald, RED), ("Wilson", ic_wilson, MUTED), ("Clopper–Pearson", ic_cp, BLUE), ("creíble Beta(1, 1)", ic_bayes, C23)]
    cob = {}
    for nom, m, col in metodos:
        cv = np.array([cobertura_exacta(m, t, n) for t in ths])
        cob[nom] = cv
        b.plot(ths, cv * 100, color=col, lw=1.6 if nom != "creíble Beta(1, 1)" else 2.4, label=nom)
    b.axhline(95, color=INK, ls=":", lw=1.2)
    b.set_ylim(50, 100.5)
    b.set_xlabel("θ verdadero")
    b.set_ylabel("cobertura frecuentista exacta (%)")
    b.legend(fontsize=7.4, frameon=False, loc="lower center", ncol=2)
    titulo(b, "n = 20: cobertura exacta de cada intervalo al 95 %")
    style(b)
    NUM["intervalos"] = dict(th_verdadero=th0, n=n, fallan_creible=int(miss_b), fallan_wald=int(miss_w), muestras=40,
                             cobertura_media={k: round(float(v.mean() * 100), 2) for k, v in cob.items()},
                             cobertura_min={k: round(float(v.min() * 100), 2) for k, v in cob.items()},
                             cobertura_en_065={k: round(cobertura_exacta(m, 0.65, n) * 100, 2) for k, m, _ in metodos},
                             para_14_de_20={"Wald": [round(float(v), 4) for v in ic_wald(14, 20)], "Wilson": [round(float(v), 4) for v in ic_wilson(14, 20)],
                                            "Clopper–Pearson": [round(float(v), 4) for v in ic_cp(14, 20)], "creíble Beta(1, 1)": [round(float(v), 4) for v in ic_bayes(14, 20)],
                                            "creíble Beta(2, 2)": [round(float(v), 4) for v in ic_bayes(14, 20, 2, 2)]})
    c = fig.add_subplot(gs[2])
    filas = list(NUM["intervalos"]["para_14_de_20"].items())
    for i, (nom, (lo, hi)) in enumerate(filas):
        col = C23 if nom.startswith("creíble") else (MUTED if i < 3 else C23)
        yy = len(filas) - 1 - i
        c.errorbar([(lo + hi) / 2], [yy], xerr=[[(hi - lo) / 2], [(hi - lo) / 2]], fmt="o", color=col, capsize=4, lw=2.2, ms=5)
        c.text(hi + 0.012, yy, f"[{lo:.3f}; {hi:.3f}]", va="center", fontsize=7.8, color=INK)
    c.axvline(0.7, color=INK, ls=":", lw=1)
    c.set_yticks(range(len(filas)))
    c.set_yticklabels([f[0] for f in filas][::-1], fontsize=8)
    c.set_xlim(0.38, 1.12)
    c.set_xlabel("θ")
    titulo(c, "Los mismos datos (14 de 20), cinco intervalos")
    style(c, "x")
    save(fig, "c23_intervalos.png")


# ------------------------------------------------------------------ 7. conjugadas típicas, verificadas con una grilla
def conjugadas_fig():
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3))
    res = {}

    def panel(ax, xs, prior_pdf, lik, post_pdf, nom, formula, col):
        un = lik * prior_pdf
        grilla = un / integrate.trapezoid(un, xs)
        cierre = post_pdf
        ax.plot(xs, prior_pdf, color=MUTED, lw=1.4, ls="--", label="priori")
        ax.plot(xs, grilla, color=col, lw=4, alpha=0.35, label="priori × verosimilitud (grilla)")
        ax.plot(xs, cierre, color=INK, lw=1.4, label="fórmula de la tabla")
        titulo(ax, nom, 8.6)
        ax.set_xlabel(formula, fontsize=8)
        style(ax)
        return float(np.abs(grilla - cierre).max())
    # Binomial–Beta
    xs = np.linspace(0.0005, 0.9995, 4000)
    res["Binomial–Beta"] = panel(axs[0], xs, stats.beta(2, 2).pdf(xs), stats.binom.pmf(14, 20, xs), stats.beta(16, 8).pdf(xs), "Binomial–Beta: 14 de 20", "Beta(2, 2) → Beta(16, 8)", C23)
    # Bernoulli–Beta (secuencia de 10 con 7 éxitos)
    seq = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    lik = np.prod([xs ** s * (1 - xs) ** (1 - s) for s in seq], axis=0)
    res["Bernoulli–Beta"] = panel(axs[1], xs, stats.beta(2, 2).pdf(xs), lik, stats.beta(2 + seq.sum(), 2 + len(seq) - seq.sum()).pdf(xs), "Bernoulli–Beta: 7 éxitos en 10", f"Beta(2, 2) → Beta({2 + seq.sum()}, {2 + len(seq) - seq.sum()})", BLUE)
    # Poisson–Gamma
    cts = np.array([3, 1, 4, 2, 5])
    xl = np.linspace(0.01, 8, 4000)
    lik = np.prod([stats.poisson.pmf(c_, xl) for c_ in cts], axis=0)
    res["Poisson–Gamma"] = panel(axs[2], xl, stats.gamma(2, scale=1 / 1).pdf(xl), lik, stats.gamma(2 + cts.sum(), scale=1 / (1 + len(cts))).pdf(xl), "Poisson–Gamma: 3,1,4,2,5", f"Gamma(2, 1) → Gamma({2 + cts.sum()}, {1 + len(cts)})", AMBER)
    # Exponencial–Gamma
    tm = np.array([0.8, 1.9, 0.4, 1.2])
    xe = np.linspace(0.01, 4, 4000)
    lik = np.prod([stats.expon.pdf(t, scale=1 / xe) for t in tm], axis=0)
    res["Exponencial–Gamma"] = panel(axs[3], xe, stats.gamma(2, scale=1).pdf(xe), lik, stats.gamma(2 + len(tm), scale=1 / (1 + tm.sum())).pdf(xe), "Exponencial–Gamma: 4 tiempos", f"Gamma(2, 1) → Gamma({2 + len(tm)}, {1 + tm.sum():.1f})", VIOLET)
    axs[0].legend(fontsize=6.6, frameon=False, loc="upper left")
    NUM["conjugadas"] = dict(max_dif_grilla={k: float(f"{v:.2e}") for k, v in res.items()},
                             poisson=dict(S=int(cts.sum()), m=len(cts), a_post=int(2 + cts.sum()), b_post=int(1 + len(cts)), media=round((2 + cts.sum()) / (1 + len(cts)), 4)),
                             exponencial=dict(S=round(float(tm.sum()), 2), m=len(tm), a_post=int(2 + len(tm)), b_post=round(float(1 + tm.sum()), 2), media=round((2 + len(tm)) / (1 + tm.sum()), 4)),
                             bernoulli=dict(S=int(seq.sum()), m=len(seq), a_post=int(2 + seq.sum()), b_post=int(2 + len(seq) - seq.sum())))
    fig.subplots_adjust(wspace=0.32)
    save(fig, "c23_conjugadas.png")


# ------------------------------------------------------------------ 8. puente con la Clase 22: A/B bayesiano
def ab_fig():
    ab = pd.read_csv(CARPETA22 / "ab_checkout.csv")
    ab["trat"] = (ab["grupo"] == "tratamiento").astype(int)
    c, t = ab[ab["trat"] == 0], ab[ab["trat"] == 1]
    n0, n1 = len(c), len(t)
    x0, x1 = int(c["convierte"].sum()), int(t["convierte"].sum())
    p0, p1 = x0 / n0, x1 / n1
    dif = p1 - p0
    se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    d0, d1 = stats.beta(1 + x0, 1 + n0 - x0), stats.beta(1 + x1, 1 + n1 - x1)
    rng = np.random.default_rng(RS)
    S = 200000
    s0, s1 = d0.rvs(S, random_state=rng), d1.rvs(S, random_state=rng)
    diff = s1 - s0
    NUM["ab"] = dict(n0=n0, n1=n1, x0=x0, x1=x1, p0=round(p0, 5), p1=round(p1, 5), dif_pp=round(dif * 100, 4), wald_ic_pp=[round((dif - 1.959964 * se) * 100, 3), round((dif + 1.959964 * se) * 100, 3)],
                     post0=[1 + x0, 1 + n0 - x0], post1=[1 + x1, 1 + n1 - x1], p_b_mayor=round(float((s1 > s0).mean()), 5), dif_media_pp=round(float(diff.mean() * 100), 4),
                     cri_pp=[round(float(np.percentile(diff, 2.5) * 100), 3), round(float(np.percentile(diff, 97.5) * 100), 3)], p_mejora_mayor_15pp=round(float((diff > 0.015).mean()), 4),
                     p_mejora_mayor_1pp=round(float((diff > 0.01).mean()), 4), sims=S)
    # submuestras anidadas: P(B>A) bayesiana frente a 1 − p unilateral
    rs = np.random.default_rng(RS + 1)
    c_idx, t_idx = rs.permutation(n0), rs.permutation(n1)
    cy, ty = c["convierte"].to_numpy()[c_idx], t["convierte"].to_numpy()[t_idx]
    grid = [200, 500, 1000, 2000, 5000, 10965]
    fila = []
    for m in grid:
        a_, b_ = int(cy[:m].sum()), int(ty[:m].sum())
        pa, pb = a_ / m, b_ / m
        pp = (a_ + b_) / (2 * m)
        z = (pb - pa) / np.sqrt(pp * (1 - pp) * 2 / m)
        p_uni = float(stats.norm.sf(z))
        da, db = stats.beta(1 + a_, 1 + m - a_).rvs(60000, random_state=rs), stats.beta(1 + b_, 1 + m - b_).rvs(60000, random_state=rs)
        fila.append(dict(m=m, xa=a_, xb=b_, dif_pp=round((pb - pa) * 100, 3), p_b_mayor=round(float((db > da).mean()), 4), uno_menos_p=round(1 - p_uni, 4)))
    NUM["ab_submuestras"] = fila
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1], wspace=0.3)
    a = fig.add_subplot(gs[0])
    xs = np.linspace(0.075, 0.125, 800)
    a.plot(xs * 100, d0.pdf(xs) / 100, color=MUTED, lw=2.4, label=f"control: Beta({1 + x0}, {1 + n0 - x0})")
    a.plot(xs * 100, d1.pdf(xs) / 100, color=TEAL, lw=2.4, label=f"tratamiento: Beta({1 + x1}, {1 + n1 - x1})")
    a.fill_between(xs * 100, d0.pdf(xs) / 100, color=MUTED, alpha=0.2)
    a.fill_between(xs * 100, d1.pdf(xs) / 100, color=TEAL, alpha=0.2)
    a.set_xlabel("tasa de conversión (%)")
    a.set_ylabel("densidad")
    a.legend(fontsize=7.6, frameon=False, loc="upper left")
    titulo(a, "Posterioris con priori uniforme (ab_checkout)")
    style(a)
    b = fig.add_subplot(gs[1])
    b.hist(diff * 100, bins=80, color=TEAL, alpha=0.7, density=True)
    lo, hi = NUM["ab"]["cri_pp"]
    b.axvline(lo, color=INK, ls="--", lw=1.2)
    b.axvline(hi, color=INK, ls="--", lw=1.2, label=f"IC creíble 95 % [{lo:.2f}; {hi:.2f}] pp")
    b.axvline(1.5, color=RED, lw=2, label=f"MDE = 1.5 pp: P(mejora > 1.5) = {NUM['ab']['p_mejora_mayor_15pp']:.2f}")
    b.axvline(0, color=INK, lw=0.8)
    b.set_xlabel("mejora de B sobre A (pp)")
    b.set_ylabel("densidad")
    b.set_ylim(0, 1.6)
    b.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(b, "Diferencia posteriori: P(B > A) > 0.9999")
    style(b)
    c_ = fig.add_subplot(gs[2])
    ms = [f["m"] for f in fila]
    c_.plot(ms, [f["p_b_mayor"] for f in fila], "o-", color=TEAL, lw=2.4, label="bayesiano: P(B > A | datos)")
    c_.plot(ms, [f["uno_menos_p"] for f in fila], "s--", color=AMBER, lw=1.8, label="frecuentista: 1 − p (unilateral)")
    c_.set_xscale("log")
    c_.set_xlabel("usuarios por grupo (submuestras de ab_checkout)")
    c_.set_ylim(0.3, 1.03)
    c_.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(c_, "Con priori plana, casi coinciden, pero se leen distinto")
    style(c_)
    save(fig, "c23_ab.png")


PASOS = dict(bayes=bayes_fig, betabinom=betabinom_fig, prioris=prioris_fig, secuencial=secuencial_fig, resumenes=resumenes_fig, intervalos=intervalos_fig,
             conjugadas=conjugadas_fig, ab=ab_fig, esquema=esquema_fig)

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c23_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or list(PASOS)
    # el esquema usa números de las demás figuras: se generan antes
    previos = [k for k in ("bayes", "betabinom") if "esquema" in elegidos and k not in elegidos]
    for k in previos + [e for e in elegidos if e != "esquema"] + (["esquema"] if "esquema" in elegidos else []):
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c23_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
