"""Figuras de la Clase 22 (Experimentación, A/B testing y causalidad aplicada).
Datos de la profesora Maritza Márquez (carpeta «Clase 22 …»): ab_checkout.csv (22 000 usuarios, experimento limpio), ab_srm.csv (11 476 usuarios,
experimento con Sample Ratio Mismatch) y did_app_regiones.csv (160 comunas × 6 meses, lanzamiento en la RM). Todo sale de código: nada se inventa.
Dos figuras usan simulaciones declaradas como tales (peeking y comparaciones múltiples) y una submuestra con «autoselección» construida a partir
de ab_checkout para ilustrar el sesgo de confusión. Todo el azar usa semilla fija.
Ejecutar con el venv que tiene scipy y statsmodels:  python -X utf8 figuras_c22.py   (o con nombres: python -X utf8 figuras_c22.py poder did)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C22 = "#0D9488"      # color de la clase 22 (verde azulado, igual que el roadmap)
C22_BG = "#F0FDFA"
CTRL = "#6B7280"     # control / variante A
TRAT = "#0D9488"     # tratamiento / variante B
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
GREEN = "#059669"
VIOLET = "#7C3AED"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA = next(p for p in BASE.glob("Clase 22*") if p.is_dir())

ab = pd.read_csv(CARPETA / "ab_checkout.csv")
ab["trat"] = (ab["grupo"] == "tratamiento").astype(int)
srm = pd.read_csv(CARPETA / "ab_srm.csv")
did = pd.read_csv(CARPETA / "did_app_regiones.csv")
did["trat"] = (did["zona"] == "RM").astype(int)

ctrl, tr = ab[ab["trat"] == 0], ab[ab["trat"] == 1]
n0, n1 = len(ctrl), len(tr)
x0, x1 = int(ctrl["convierte"].sum()), int(tr["convierte"].sum())
p0, p1 = x0 / n0, x1 / n1
dif = p1 - p0
se_w = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
Z = stats.norm.ppf(0.975)
ic = (dif - Z * se_w, dif + Z * se_w)
p_pool = (x0 + x1) / (n0 + n1)
se_p = np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1))
z_obs = dif / se_p
p_z = 2 * stats.norm.sf(abs(z_obs))


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def n_por_grupo(pa, pb, alfa=0.05, pot=0.80):
    """Misma fórmula que power.prop.test de R (dos colas, varianzas distintas bajo H1)."""
    za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(pot)
    pbar = (pa + pb) / 2
    return ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(pa * (1 - pa) + pb * (1 - pb))) / abs(pa - pb)) ** 2


def potencia(pa, pb, n, alfa=0.05):
    za = stats.norm.ppf(1 - alfa / 2)
    pbar = (pa + pb) / 2
    d = abs(pb - pa)
    return stats.norm.cdf((d * np.sqrt(n) - za * np.sqrt(2 * pbar * (1 - pbar))) / np.sqrt(pa * (1 - pa) + pb * (1 - pb)))


def n_arcoseno(pa, pb, alfa=0.05, pot=0.80):
    """Aproximación con el tamaño de efecto h de Cohen (arcoseno); la que da 6 131 en lugar de 6 138."""
    h = 2 * np.arcsin(np.sqrt(pb)) - 2 * np.arcsin(np.sqrt(pa))
    return 2 * ((stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(pot)) / h) ** 2


def mde_para_n(pa, n, pot=0.80):
    from scipy.optimize import brentq
    return brentq(lambda d: potencia(pa, pa + d, n) - pot, 1e-4, 0.2)


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

    caja(0.1, 2.75, 2.35, 1.85, CTRL, "1 · La pregunta", ["¿quién comprará? → predicción", "¿qué pasa si cambio algo?", "→ causalidad", "τᵢ = Yᵢ(1) − Yᵢ(0): nunca se", "observan los dos"])
    caja(2.75, 2.75, 2.55, 1.85, BLUE, "2 · El diseño", ["hipótesis · unidad · variantes", "métrica primaria (OEC)", "guardrails · duración", "MDE, α, potencia → n"])
    caja(5.6, 2.75, 2.3, 1.85, AMBER, "3 · Los chequeos", ["SRM (chi-cuadrado)", "peeking: no mirar antes", "comparaciones múltiples", "segmentos preespecificados"])
    caja(8.2, 2.75, 2.15, 1.85, C22, "4 · El análisis", ["p̂B − p̂A, IC y p-valor", "regresión y logit", "covariables y CUPED", "heterogeneidad (con cautela)"])
    caja(10.65, 2.75, 1.75, 1.85, VIOLET, "5 · Sin azar", ["DiD", "tendencias", "paralelas", "(cuasi-experimento)"])
    for x in (2.5, 5.35, 7.95, 10.4):
        ax.annotate("", xy=(x + 0.25, 3.65), xytext=(x, 3.65), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 2.35, boxstyle="round,pad=0.03", fc=C22_BG, ec=C22, lw=1.4))
    ax.text(6.25, 2.15, "Datos de la clase: ab_checkout (22 000 usuarios) · ab_srm (11 476, roto) · did_app_regiones (160 comunas × 6 meses)", ha="center", va="center", fontsize=9.0, color=C22, fontweight="bold")
    ax.text(6.25, 1.7, f"Experimento limpio: control {p0:.2%} frente a tratamiento {p1:.2%} → diferencia {dif * 100:.2f} pp, IC 95 % [{ic[0] * 100:.2f}; {ic[1] * 100:.2f}] pp, p = 1.4 × 10⁻⁶", ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(6.25, 1.32, "Tamaño: power.prop.test(0.09, 0.105) → 6 138 por grupo · SRM: p = 0.64 en ab_checkout y 1.5 × 10⁻⁸ en ab_srm · DiD en la RM: +5.76 puntos", ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(6.25, 0.85, "R ↔ Python:  prop.test ↔ proportions_ztest / scipy  ·  power.prop.test ↔ NormalIndPower  ·  chisq.test ↔ scipy.stats.chisquare", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    ax.text(6.25, 0.45, "lm ↔ statsmodels.formula.api.ols  ·  glm(family = binomial) ↔ smf.logit  ·  tidy(conf.int = TRUE) ↔ .summary() / .conf_int()", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    save(fig, "c22_esquema.png")


# ------------------------------------------------------------------ 1. correlación vs causalidad (confusor) y autoselección
def construir_autoseleccion():
    """Submuestra «observacional»: los usuarios con más gasto previo adoptan el checkout nuevo con mayor probabilidad."""
    rng = np.random.default_rng(RS)
    z = (ab["gasto_pre"] - ab["gasto_pre"].mean()) / ab["gasto_pre"].std()
    p_adopta = 1 / (1 + np.exp(-(0.6 * z)))                     # gasto alto → adopta
    quedan = np.where(ab["trat"] == 1, rng.random(len(ab)) < p_adopta, rng.random(len(ab)) < 1 - p_adopta)
    return ab[quedan].copy()


obs = construir_autoseleccion()


def causal_fig():
    o0, o1 = obs[obs["trat"] == 0]["convierte"], obs[obs["trat"] == 1]["convierte"]
    d_obs = o1.mean() - o0.mean()
    se_o = np.sqrt(o0.var() / len(o0) + o1.var() / len(o1))
    aj = smf.ols("convierte ~ trat + gasto_pre", obs).fit()
    aj2 = smf.ols("convierte ~ trat + sesiones_pre", obs).fit()
    NUM["autoseleccion"] = dict(n=len(obs), n_trat=int((obs["trat"] == 1).sum()), n_ctrl=int((obs["trat"] == 0).sum()), gasto_trat=round(float(obs[obs["trat"] == 1]["gasto_pre"].mean()), 2),
                                gasto_ctrl=round(float(obs[obs["trat"] == 0]["gasto_pre"].mean()), 2), dif_ingenua=round(float(d_obs * 100), 2), ic_ingenua=[round(float((d_obs - Z * se_o) * 100), 2), round(float((d_obs + Z * se_o) * 100), 2)],
                                ajustada_gasto=round(float(aj.params["trat"] * 100), 2), ic_ajustada=[round(float(v * 100), 2) for v in aj.conf_int().loc["trat"]],
                                ajustada_sesiones=round(float(aj2.params["trat"] * 100), 2))
    fig = plt.figure(figsize=(12.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.15], wspace=0.62)
    # (a) diagrama causal
    a = fig.add_subplot(gs[0])
    a.set_xlim(-0.7, 10.7)
    a.set_ylim(0, 8)
    a.axis("off")
    for (x, y, txt, col) in [(5, 6.6, "Z\nconfusor\n(gasto previo,\nmotivación…)", AMBER), (1.5, 1.8, "T\ncambio\n(checkout\nnuevo)", C22), (8.5, 1.8, "Y\nresultado\n(compra)", BLUE)]:
        a.add_patch(FancyBboxPatch((x - 1.5, y - 1.1), 3.0, 2.2, boxstyle="round,pad=0.06", fc="white", ec=col, lw=2))
        a.text(x, y, txt, ha="center", va="center", fontsize=8.3, color=col, fontweight="bold")
    a.annotate("", xy=(2.2, 3.0), xytext=(4.0, 5.4), arrowprops=dict(arrowstyle="->", color=AMBER, lw=2))
    a.annotate("", xy=(7.8, 3.0), xytext=(6.0, 5.4), arrowprops=dict(arrowstyle="->", color=AMBER, lw=2))
    a.annotate("", xy=(6.9, 1.8), xytext=(3.1, 1.8), arrowprops=dict(arrowstyle="->", color=C22, lw=2.4))
    a.text(5, 2.45, "efecto causal", ha="center", fontsize=7.6, color=C22)
    a.text(2.4, 4.7, "Z influye en quién\nadopta T", ha="right", fontsize=7.4, color=AMBER)
    a.text(7.6, 4.7, "Z también influye\nen Y", ha="left", fontsize=7.4, color=AMBER)
    a.text(5, 0.2, "Aleatorizar T borra la flecha Z → T", ha="center", fontsize=8.2, color=INK, fontweight="bold")
    titulo(a, "Correlación ≠ causalidad: un confusor abre un camino falso")
    # (b) conversión por nivel de gasto previo
    b = fig.add_subplot(gs[1])
    q = pd.qcut(ab["gasto_pre"], 5, labels=False)
    tasas = ab.groupby(q)["convierte"].mean()
    b.bar(range(5), tasas.values * 100, color=BLUE, alpha=0.85)
    for i, v in enumerate(tasas.values):
        b.text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", fontsize=8)
    b.set_xticks(range(5))
    b.set_xticklabels(["Q1\n(menor)", "Q2", "Q3", "Q4", "Q5\n(mayor)"], fontsize=8)
    b.set_xlabel("quintil de gasto previo")
    b.set_ylabel("conversión (%)")
    titulo(b, "En los datos, el gasto previo predice la compra")
    style(b, "y")
    NUM["conv_por_quintil_gasto"] = [round(float(v), 4) for v in tasas.values]
    # (c) tres estimaciones
    c = fig.add_subplot(gs[2])
    est = [("Experimento aleatorizado\n(ab_checkout, 22 000)", dif * 100, (ic[0] * 100, ic[1] * 100), TRAT),
           ("Autoselección:\ndiferencia ingenua", d_obs * 100, ((d_obs - Z * se_o) * 100, (d_obs + Z * se_o) * 100), RED),
           ("Autoselección ajustada\npor gasto_pre", aj.params["trat"] * 100, tuple(aj.conf_int().loc["trat"] * 100), AMBER)]
    for i, (nombre, v, (lo, hi), col) in enumerate(est):
        yy = 2 - i
        c.errorbar([v], [yy], xerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=4, lw=2, ms=7)
        c.text(v, yy + 0.22, f"{v:.2f} pp", ha="center", fontsize=8.2, color=col, fontweight="bold")
    c.axvline(dif * 100, color=TRAT, ls="--", lw=1)
    c.set_yticks([2, 1, 0])
    c.set_yticklabels([e[0] for e in est], fontsize=8)
    c.set_xlabel("diferencia de conversión (puntos porcentuales)")
    c.set_ylim(-0.6, 2.7)
    titulo(c, "La autoselección infla el efecto; ajustar lo medido ayuda")
    style(c, "x")
    save(fig, "c22_causal.png")


# ------------------------------------------------------------------ 2. aleatorización: balance de covariables
def smd_tabla(df):
    t, c_ = df[df["trat"] == 1], df[df["trat"] == 0]
    filas = {}
    for v in ["gasto_pre", "sesiones_pre"]:
        sp = np.sqrt((t[v].var() + c_[v].var()) / 2)
        filas[v] = (t[v].mean() - c_[v].mean()) / sp
    for col in ["dispositivo", "region"]:
        for lev in sorted(df[col].unique()):
            a_, b_ = (t[col] == lev).mean(), (c_[col] == lev).mean()
            filas[f"{col} = {lev}"] = (a_ - b_) / np.sqrt((a_ * (1 - a_) + b_ * (1 - b_)) / 2)
    return pd.Series(filas)


def balance_fig():
    s_alea, s_obs = smd_tabla(ab), smd_tabla(obs)
    NUM["balance"] = dict(max_smd_aleatorio=round(float(s_alea.abs().max()), 4), max_smd_autoseleccion=round(float(s_obs.abs().max()), 3),
                          smd_gasto_aleatorio=round(float(s_alea["gasto_pre"]), 4), smd_gasto_autoseleccion=round(float(s_obs["gasto_pre"]), 3),
                          medias_gasto=dict(control=round(float(ctrl["gasto_pre"].mean()), 2), tratamiento=round(float(tr["gasto_pre"].mean()), 2)),
                          medias_sesiones=dict(control=round(float(ctrl["sesiones_pre"].mean()), 3), tratamiento=round(float(tr["sesiones_pre"].mean()), 3)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.32)
    a = fig.add_subplot(gs[0])
    ys = np.arange(len(s_alea))[::-1]
    a.axvspan(-0.1, 0.1, color=GREEN, alpha=0.10)
    a.scatter(s_alea.values, ys, color=TRAT, s=42, label="aleatorizado (ab_checkout)", zorder=3)
    a.scatter(s_obs.values, ys, color=RED, s=42, marker="s", label="autoselección", zorder=3)
    a.axvline(0, color=INK, lw=0.8)
    a.set_yticks(ys)
    a.set_yticklabels(s_alea.index, fontsize=7.6)
    a.set_xlabel("diferencia estandarizada (tratamiento − control)")
    a.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(a, "Balance de covariables previas (zona verde: |d| < 0.1)")
    style(a, "x")
    b = fig.add_subplot(gs[1])
    bins = np.linspace(0, 200, 41)
    b.hist(ctrl["gasto_pre"].clip(upper=200), bins=bins, color=CTRL, alpha=0.6, density=True, label=f"control (media {ctrl['gasto_pre'].mean():.1f})")
    b.hist(tr["gasto_pre"].clip(upper=200), bins=bins, color=TRAT, alpha=0.6, density=True, label=f"tratamiento (media {tr['gasto_pre'].mean():.1f})")
    b.set_xlabel("gasto previo")
    b.set_ylabel("densidad")
    b.legend(fontsize=7.6, frameon=False)
    titulo(b, "Con azar: distribuciones casi idénticas")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    for df_, col, nom in [(obs[obs["trat"] == 0], CTRL, "control"), (obs[obs["trat"] == 1], RED, "tratamiento")]:
        c.hist(df_["gasto_pre"].clip(upper=200), bins=bins, color=col, alpha=0.6, density=True, label=f"{nom} (media {df_['gasto_pre'].mean():.1f})")
    c.set_xlabel("gasto previo")
    c.legend(fontsize=7.6, frameon=False)
    titulo(c, "Con autoselección: el tratamiento «ya era» distinto")
    style(c, "y")
    save(fig, "c22_balance.png")


# ------------------------------------------------------------------ 3. potencia, MDE y tamaño muestral
def poder_fig():
    n_prof = n_por_grupo(0.09, 0.105)
    n_h = n_arcoseno(0.09, 0.105)
    mde_real = mde_para_n(0.09, (n0 + n1) / 2)
    pot_real = potencia(p0, p1, (n0 + n1) / 2)
    NUM["poder"] = dict(n_power_prop_test=round(float(n_prof), 3), n_arcoseno=round(float(n_h), 3), n_redondeado=int(np.ceil(n_prof)), n_total=int(2 * np.ceil(n_prof)), n_total_diapositiva=12262,
                        n_mde_1pp=int(np.ceil(n_por_grupo(0.09, 0.10))), n_mde_2pp=int(np.ceil(n_por_grupo(0.09, 0.11))), n_mde_075pp=int(np.ceil(n_por_grupo(0.09, 0.0975))),
                        n_pot90=int(np.ceil(n_por_grupo(0.09, 0.105, pot=0.90))), mde_real_80=round(float(mde_real * 100), 3), n_real_por_grupo=int((n0 + n1) / 2),
                        potencia_real_efecto_observado=round(float(pot_real), 4), potencia_con_6138_y_1pp=round(float(potencia(0.09, 0.10, 6138)), 4))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    d = np.linspace(0.006, 0.03, 120)
    for pot, col in [(0.8, TRAT), (0.9, BLUE)]:
        a.plot(d * 100, [n_por_grupo(0.09, 0.09 + x, pot=pot) for x in d], color=col, lw=2, label=f"potencia {pot:.0%}")
    a.scatter([1.5], [n_prof], color=AMBER, s=60, zorder=5)
    a.annotate(f"MDE 1.5 pp → {int(np.ceil(n_prof)):,} por grupo".replace(",", " "), (1.5, n_prof), xytext=(1.9, n_prof * 1.5), fontsize=8, color=AMBER, arrowprops=dict(arrowstyle="->", color=AMBER))
    a.set_ylim(0, 22000)
    a.set_xlabel("efecto mínimo detectable, MDE (puntos porcentuales)")
    a.set_ylabel("usuarios por grupo")
    a.legend(fontsize=8, frameon=False)
    titulo(a, "Si el MDE baja a la mitad, n se cuadruplica (base 9 %)")
    style(a)
    b = fig.add_subplot(gs[1])
    ns = np.linspace(500, 16000, 200)
    for dd, col in [(0.01, RED), (0.015, TRAT), (0.02, BLUE)]:
        b.plot(ns, [potencia(0.09, 0.09 + dd, n) for n in ns], color=col, lw=2, label=f"efecto real {dd * 100:.1f} pp")
    b.axhline(0.8, color=INK, ls="--", lw=1)
    b.axvline(n_prof, color=AMBER, ls=":", lw=1.4)
    b.set_xlabel("usuarios por grupo")
    b.set_ylabel("potencia (prob. de detectar el efecto)")
    b.set_ylim(0, 1.02)
    b.legend(fontsize=8, frameon=False, loc="lower right")
    titulo(b, "Con n = 6 138: 80 % solo si el efecto es de 1.5 pp")
    style(b)
    c = fig.add_subplot(gs[2])
    ef = np.linspace(0.002, 0.03, 150)
    c.plot(ef * 100, [potencia(0.0888, 0.0888 + x, (n0 + n1) / 2) for x in ef], color=TRAT, lw=2)
    c.axhline(0.8, color=INK, ls="--", lw=1)
    c.axvline(mde_real * 100, color=AMBER, ls=":", lw=1.4)
    c.scatter([dif * 100], [pot_real], color=RED, zorder=5, s=50)
    c.text(mde_real * 100 + 0.1, 0.45, f"MDE real (80 %):\n{mde_real * 100:.2f} pp", fontsize=8, color=AMBER)
    c.text(dif * 100 - 0.2, pot_real - 0.12, f"efecto observado\n{dif * 100:.2f} pp", fontsize=8, color=RED, ha="right")
    c.set_xlabel("efecto real (puntos porcentuales)")
    c.set_ylabel("potencia")
    c.set_ylim(0, 1.02)
    titulo(c, "El experimento real (≈ 11 000 por grupo) sobraba")
    style(c)
    save(fig, "c22_poder.png")


# ------------------------------------------------------------------ 4. tasas de conversión con intervalos
def tasas_fig():
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=0.34)
    a = fig.add_subplot(gs[0])
    for i, (nombre, p_, n_, col) in enumerate([("A · control", p0, n0, CTRL), ("B · tratamiento", p1, n1, TRAT)]):
        se = np.sqrt(p_ * (1 - p_) / n_)
        a.bar(i, p_ * 100, color=col, alpha=0.85, width=0.55)
        a.errorbar(i, p_ * 100, yerr=Z * se * 100, color=INK, capsize=5, lw=1.6)
        a.text(i, p_ * 100 + Z * se * 100 + 0.25, f"{p_:.2%}", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks([0, 1])
    a.set_xticklabels([f"A · control\nn = {n0:,}".replace(",", " "), f"B · tratamiento\nn = {n1:,}".replace(",", " ")], fontsize=8)
    a.set_ylim(0, 13.5)
    a.set_ylabel("conversión (%) con IC 95 %")
    titulo(a, "Tasa de conversión por variante")
    style(a, "y")
    b = fig.add_subplot(gs[1])
    b.axvspan(ic[0] * 100, ic[1] * 100, color=TRAT, alpha=0.15)
    b.errorbar([dif * 100], [0], xerr=[[dif * 100 - ic[0] * 100], [ic[1] * 100 - dif * 100]], fmt="o", color=TRAT, capsize=6, lw=2.4, ms=8)
    b.axvline(0, color=INK, lw=1.2)
    b.axvline(1.5, color=AMBER, ls="--", lw=1.4)
    b.text(1.55, 0.35, "MDE\n1.5 pp", color=AMBER, fontsize=8)
    b.text(0.05, 0.35, "H₀: diferencia = 0", fontsize=8, color=INK)
    b.text(dif * 100, -0.28, f"{dif * 100:.2f} pp\n[{ic[0] * 100:.2f}; {ic[1] * 100:.2f}]", ha="center", fontsize=8.2, color=TRAT, fontweight="bold")
    b.set_ylim(-0.7, 0.8)
    b.set_xlim(-0.8, 3.3)
    b.set_yticks([])
    b.set_xlabel("p̂B − p̂A (puntos porcentuales)")
    titulo(b, "Diferencia e intervalo de confianza: no contiene el 0")
    style(b, "x")
    c = fig.add_subplot(gs[2])
    disp = ab.groupby(["dispositivo", "grupo"])["convierte"].agg(["mean", "size"]).reset_index()
    for j, dsp in enumerate(["desktop", "movil", "tablet"]):
        for k, (g, col) in enumerate([("control", CTRL), ("tratamiento", TRAT)]):
            r = disp[(disp["dispositivo"] == dsp) & (disp["grupo"] == g)].iloc[0]
            se = np.sqrt(r["mean"] * (1 - r["mean"]) / r["size"])
            c.bar(j + (k - 0.5) * 0.38, r["mean"] * 100, 0.36, color=col, alpha=0.85, label=g if j == 0 else None)
            c.errorbar(j + (k - 0.5) * 0.38, r["mean"] * 100, yerr=Z * se * 100, color=INK, capsize=3, lw=1.2)
    c.set_xticks(range(3))
    c.set_xticklabels(["desktop", "móvil", "tablet"], fontsize=8.5)
    c.set_ylabel("conversión (%)")
    c.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    c.set_ylim(0, 15)
    titulo(c, "Por dispositivo (IC más anchos: menos usuarios)")
    style(c, "y")
    save(fig, "c22_tasas.png")
    NUM["efecto"] = dict(n0=n0, n1=n1, x0=x0, x1=x1, p0=round(p0, 5), p1=round(p1, 5), dif_pp=round(float(dif * 100), 4), se_wald=round(float(se_w), 6), ic_pp=[round(float(ic[0] * 100), 3), round(float(ic[1] * 100), 3)],
                         se_pooled=round(float(se_p), 6), z=round(float(z_obs), 4), p_valor=float(f"{p_z:.4g}"), chi2=round(float(z_obs ** 2), 3), relativo=round(float(dif / p0), 4))


# ------------------------------------------------------------------ 5. p-valor: distribución nula, permutación y p bajo H0/H1
def pvalor_fig():
    rng = np.random.default_rng(RS)
    y = ab["convierte"].to_numpy()
    g = ab["trat"].to_numpy()
    B = 20000
    difs = np.empty(B)
    n_t = int(g.sum())
    for b in range(B):
        perm = rng.permutation(g)
        difs[b] = y[perm == 1].mean() - y[perm == 0].mean()
    extremos = int((np.abs(difs) >= abs(dif)).sum())
    NUM["permutacion"] = dict(B=B, sd=round(float(difs.std() * 100), 4), sd_teorica=round(float(se_p * 100), 4), extremos=extremos, maximo_abs=round(float(np.abs(difs).max() * 100), 3))
    # p-valores bajo H0 (A/A con el grupo control) y bajo H1 (remuestreo de ab_checkout, n pequeño)
    yc = ctrl["convierte"].to_numpy()
    n_aa = 3000
    sims = 4000
    p_h0, p_h1 = [], []
    yt = tr["convierte"].to_numpy()
    for _ in range(sims):
        a_, b_ = rng.choice(yc, n_aa), rng.choice(yc, n_aa)
        p_h0.append(_ztest(a_.sum(), n_aa, b_.sum(), n_aa))
        a_, b_ = rng.choice(yc, n_aa), rng.choice(yt, n_aa)
        p_h1.append(_ztest(a_.sum(), n_aa, b_.sum(), n_aa))
    p_h0, p_h1 = np.array(p_h0), np.array(p_h1)
    NUM["p_bajo_H0_H1"] = dict(n_por_grupo=n_aa, sims=sims, fp_H0=round(float((p_h0 < 0.05).mean()), 4), potencia_H1=round(float((p_h1 < 0.05).mean()), 4), p_h0_menor_10=round(float((p_h0 < 0.10).mean()), 3))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    xs = np.linspace(-6, 6, 400)
    a.plot(xs, stats.norm.pdf(xs), color=INK, lw=1.8)
    for lado in (xs <= -1.96, xs >= 1.96):
        a.fill_between(xs[lado], stats.norm.pdf(xs[lado]), color=AMBER, alpha=0.35)
    a.axvline(z_obs, color=RED, lw=2)
    a.text(z_obs - 0.15, 0.33, f"z observado\n= {z_obs:.2f}", color=RED, fontsize=8.4, ha="right")
    a.text(-3.3, 0.30, "95 % de los resultados\nbajo H₀ (sin efecto)", ha="center", fontsize=8, color=INK)
    a.set_xlabel("z = (p̂B − p̂A) / EE")
    a.set_ylabel("densidad bajo H₀")
    titulo(a, f"Si no hubiera efecto, z ≈ {z_obs:.1f} sería rarísimo (p = 1.4 × 10⁻⁶)")
    style(a, "y")
    b = fig.add_subplot(gs[1])
    b.hist(difs * 100, bins=60, color=CTRL, alpha=0.8, edgecolor="white")
    b.axvline(dif * 100, color=RED, lw=2)
    b.axvline(-dif * 100, color=RED, lw=1, ls="--")
    b.text(dif * 100 - 0.05, b.get_ylim()[1] * 0.55, f"observado\n{dif * 100:.2f} pp", color=RED, fontsize=8.4, ha="right")
    b.set_xlabel("diferencia al barajar los grupos (pp)")
    b.set_ylabel("remuestreos")
    titulo(b, f"Permutación: {extremos} de {B:,} barajados son tan extremos".replace(",", " "))
    style(b, "y")
    c = fig.add_subplot(gs[2])
    bins = np.linspace(0, 1, 21)
    c.hist(p_h0, bins=bins, color=CTRL, alpha=0.7, label="sin efecto (A/A)", edgecolor="white")
    c.hist(p_h1, bins=bins, color=TRAT, alpha=0.6, label="con el efecto real", edgecolor="white")
    c.axvline(0.05, color=RED, ls="--", lw=1.2)
    c.set_xlabel("p-valor")
    c.set_ylabel("experimentos simulados")
    c.legend(fontsize=8, frameon=False, loc="upper center")
    titulo(c, "Bajo H₀ el p-valor es uniforme: 5 % cae bajo 0.05")
    style(c, "y")
    save(fig, "c22_pvalor.png")


def _ztest(xa, na, xb, nb):
    pa_, pb_ = xa / na, xb / nb
    pp = (xa + xb) / (na + nb)
    se = np.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    return 2 * stats.norm.sf(abs((pb_ - pa_) / se)) if se > 0 else 1.0


# ------------------------------------------------------------------ 6. SRM
def srm_fig():
    c1 = np.array([n0, n1])
    e1 = np.full(2, c1.sum() / 2)
    chi1 = stats.chisquare(c1, e1)
    c2 = np.array([(srm["grupo"] == "control").sum(), (srm["grupo"] == "tratamiento").sum()])
    e2 = np.full(2, c2.sum() / 2)
    chi2 = stats.chisquare(c2, e2)
    ct = pd.crosstab(srm["grupo"], srm["dispositivo"])
    por_disp = {d: stats.chisquare(ct[d].values) for d in ct.columns}
    conv_srm = srm.groupby("grupo")["convierte"].mean()
    NUM["srm"] = dict(checkout=dict(obs=[int(v) for v in c1], esperado=float(e1[0]), chi2=round(float(chi1.statistic), 4), p=round(float(chi1.pvalue), 4)),
                      roto=dict(obs=[int(v) for v in c2], esperado=float(e2[0]), chi2=round(float(chi2.statistic), 3), p=float(f"{chi2.pvalue:.3g}"), residuo_control=int(c2[0] - e2[0])),
                      roto_por_dispositivo={d: dict(obs=[int(v) for v in ct[d].values], chi2=round(float(r.statistic), 3), p=float(f"{r.pvalue:.3g}")) for d, r in por_disp.items()},
                      roto_prop_dispositivo=pd.crosstab(srm["grupo"], srm["dispositivo"], normalize="index").round(4).to_dict(),
                      roto_conv={k: round(float(v), 4) for k, v in conv_srm.items()}, roto_n=len(srm))
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    for k, (cc, ee, ch, tit) in enumerate([(c1, e1, chi1, "ab_checkout: asignación sana"), (c2, e2, chi2, "ab_srm: asignación rota")]):
        a = fig.add_subplot(gs[k])
        a.bar([0, 1], cc, color=[CTRL, TRAT], alpha=0.85, width=0.55)
        a.plot([-0.4, 1.4], [ee[0], ee[0]], color=INK, ls="--", lw=1.4)
        a.text(1.42, ee[0], "esperado", fontsize=7.6, va="center", color=INK)
        for i, v in enumerate(cc):
            a.text(i, v * 1.015, f"{v:,}".replace(",", " "), ha="center", fontsize=8.6, fontweight="bold")
        a.set_ylim(0, cc.max() * 1.18)
        a.set_xticks([0, 1])
        a.set_xticklabels(["control", "tratamiento"])
        a.set_xlim(-0.5, 1.9)
        col = GREEN if ch.pvalue > 0.001 else RED
        a.text(0.5, cc.max() * 1.1, f"χ² = {ch.statistic:.2f} · p = {ch.pvalue:.2g}", ha="center", fontsize=8.6, color=col, fontweight="bold")
        titulo(a, tit)
        style(a, "y")
    c = fig.add_subplot(gs[2])
    ancho = 0.36
    for j, d in enumerate(ct.columns):
        for k, (g, col) in enumerate([("control", CTRL), ("tratamiento", TRAT)]):
            c.bar(j + (k - 0.5) * ancho, ct.loc[g, d], ancho * 0.95, color=col, alpha=0.85, label=g if j == 0 else None)
            c.text(j + (k - 0.5) * ancho, ct.loc[g, d] + 40, f"{ct.loc[g, d]:,}".replace(",", " "), ha="center", fontsize=7.6)
        r = por_disp[d]
        c.text(j, ct[d].max() * 1.16, f"p = {r.pvalue:.2g}", ha="center", fontsize=8.4, color=GREEN if r.pvalue > 0.001 else RED, fontweight="bold")
    c.set_xticks(range(len(ct.columns)))
    c.set_xticklabels(ct.columns)
    c.set_ylim(0, ct.values.max() * 1.3)
    c.legend(fontsize=8, frameon=False, loc="upper left")
    c.set_ylabel("usuarios")
    titulo(c, "ab_srm: el desbalance está en «móvil»")
    style(c, "y")
    save(fig, "c22_srm.png")


# ------------------------------------------------------------------ 7. peeking y comparaciones múltiples (simulación)
def peeking_fig():
    rng = np.random.default_rng(RS)
    n = 12000                      # usuarios por grupo al final
    bloque = 100
    nb = n // bloque
    sims = 4000
    pconv = float(p0)
    a_ = rng.binomial(bloque, pconv, size=(sims, nb)).cumsum(axis=1)
    b_ = rng.binomial(bloque, pconv, size=(sims, nb)).cumsum(axis=1)
    nn = bloque * np.arange(1, nb + 1)
    pp = (a_ + b_) / (2 * nn)
    z = (b_ / nn - a_ / nn) / np.sqrt(2 * pp * (1 - pp) / nn)
    rechaza = np.abs(z) > 1.96
    ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 20]
    sim_fp = []
    for k in ks:
        idx = [int(round(nb * i / k)) - 1 for i in range(1, k + 1)]
        sim_fp.append(float(rechaza[:, idx].any(axis=1).mean()))
    ind = [1 - 0.95 ** k for k in ks]
    cont = float(rechaza.any(axis=1).mean())
    NUM["peeking"] = dict(sims=sims, n_por_grupo=n, p_base=round(pconv, 4), ks=ks, sim=[round(v, 4) for v in sim_fp], independiente=[round(v, 4) for v in ind], continuo_cada_100=round(cont, 4),
                          fp_10_miradas_sim=round(sim_fp[ks.index(10)], 4), fp_10_miradas_indep=round(ind[ks.index(10)], 4))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    pvals = 2 * stats.norm.sf(np.abs(z))
    for i in range(8):
        a.plot(nn, pvals[i], lw=0.9, alpha=0.8)
    a.axhline(0.05, color=RED, ls="--", lw=1.4)
    a.set_ylim(0, 1)
    a.set_xlabel("usuarios por grupo acumulados")
    a.set_ylabel("p-valor del momento")
    titulo(a, "8 experimentos SIN efecto: el p-valor deambula")
    style(a)
    b = fig.add_subplot(gs[1])
    kk = np.arange(1, 21)
    b.plot(kk, 1 - 0.95 ** kk, color=AMBER, lw=2, label="si las miradas fueran independientes: 1 − 0.95ᵏ")
    b.plot(ks, sim_fp, "o-", color=RED, lw=2, label="simulado (mismos datos acumulados)")
    b.axhline(0.05, color=INK, ls="--", lw=1)
    b.set_xlabel("número de miradas intermedias (k)")
    b.set_ylabel("falso positivo acumulado")
    b.set_ylim(0, 0.7)
    b.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(b, "Peeking: de 5 % a ≈ 19 % con 10 miradas")
    style(b)
    c = fig.add_subplot(gs[2])
    m = np.arange(1, 41)
    c.plot(m, 1 - 0.95 ** m, color=RED, lw=2, label="sin corregir (α = 0.05 en cada una)")
    c.plot(m, 1 - (1 - 0.05 / m) ** m, color=GREEN, lw=2, label="Bonferroni (α/m en cada una)")
    c.axhline(0.05, color=INK, ls="--", lw=1)
    c.scatter([20], [1 - 0.95 ** 20], color=RED, zorder=5)
    c.text(21, 1 - 0.95 ** 20 - 0.07, f"20 métricas:\n{1 - 0.95 ** 20:.0%}", fontsize=8.4, color=RED)
    c.set_xlabel("número de métricas / segmentos probados (m)")
    c.set_ylabel("P(al menos un falso positivo)")
    c.set_ylim(0, 1.02)
    c.legend(fontsize=7.6, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.1))
    titulo(c, "Muchos cortes, una sola estrella")
    style(c)
    NUM["multiples"] = dict(m20=round(1 - 0.95 ** 20, 4), m7=round(1 - 0.95 ** 7, 4), bonf_m20=round(1 - (1 - 0.05 / 20) ** 20, 4))
    save(fig, "c22_peeking.png")


# ------------------------------------------------------------------ 8. regresión, covariables y CUPED
def cuped_fig():
    m = smf.ols("convierte ~ trat", ab).fit()
    mh = m.get_robustcov_results("HC1")
    ma = smf.ols("convierte ~ trat + gasto_pre + sesiones_pre + C(dispositivo)", ab).fit()
    theta = np.cov(ab["convierte"], ab["gasto_pre"])[0, 1] / ab["gasto_pre"].var()
    ab["y_cuped"] = ab["convierte"] - theta * (ab["gasto_pre"] - ab["gasto_pre"].mean())
    mc = smf.ols("y_cuped ~ trat", ab).fit()
    red = 1 - (mc.bse["trat"] / m.bse["trat"]) ** 2
    rho2 = float(np.corrcoef(ab["convierte"], ab["gasto_pre"])[0, 1] ** 2)
    l = smf.logit("convierte ~ trat", ab).fit(disp=0)
    ci_l = np.exp(l.conf_int().loc["trat"])
    NUM["modelos"] = dict(lpm=dict(b0=round(float(m.params["Intercept"]), 5), b1=round(float(m.params["trat"]), 5), se=round(float(m.bse["trat"]), 6), t=round(float(m.tvalues["trat"]), 3), p=float(f"{m.pvalues['trat']:.3g}"),
                                   ic=[round(float(v), 5) for v in m.conf_int().loc["trat"]], se_hc1=round(float(mh.bse[1]), 6)),
                          logit=dict(b0=round(float(l.params["Intercept"]), 4), b1=round(float(l.params["trat"]), 4), se=round(float(l.bse["trat"]), 4), z=round(float(l.tvalues["trat"]), 3), p=float(f"{l.pvalues['trat']:.3g}"),
                                     odds_ratio=round(float(np.exp(l.params["trat"])), 4), ic_or=[round(float(v), 4) for v in ci_l], odds_base=round(float(np.exp(l.params["Intercept"])), 5)),
                          ajustado=dict(b1=round(float(ma.params["trat"]), 5), se=round(float(ma.bse["trat"]), 6), p=float(f"{ma.pvalues['trat']:.3g}"), ic=[round(float(v), 5) for v in ma.conf_int().loc["trat"]],
                                        coef=dict(gasto_pre=round(float(ma.params["gasto_pre"]), 5), sesiones_pre=round(float(ma.params["sesiones_pre"]), 5), movil=round(float(ma.params["C(dispositivo)[T.movil]"]), 5), tablet=round(float(ma.params["C(dispositivo)[T.tablet]"]), 5))),
                          cuped=dict(theta=round(float(theta), 6), b1=round(float(mc.params["trat"]), 5), se=round(float(mc.bse["trat"]), 6), ic=[round(float(v), 5) for v in mc.conf_int().loc["trat"]], reduccion_var=round(float(red), 4),
                                     rho=round(float(np.sqrt(rho2)), 4), rho2=round(rho2, 4), ancho_ic_ratio=round(float(mc.bse["trat"] / m.bse["trat"]), 4)))
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.34)
    a = fig.add_subplot(gs[0])
    dec = pd.qcut(ab["gasto_pre"], 10, labels=False)
    g_ = ab.groupby(dec).agg(x=("gasto_pre", "mean"), y=("convierte", "mean"))
    a.plot(g_["x"], g_["y"] * 100, "o-", color=BLUE, lw=2)
    a.set_xlabel("gasto previo (media de cada decil)")
    a.set_ylabel("conversión (%)")
    a.text(0.04, 0.9, f"ρ = {np.sqrt(rho2):.3f} · ρ² = {rho2:.3f}", transform=a.transAxes, fontsize=8.6, color=BLUE, fontweight="bold")
    titulo(a, "El gasto previo explica parte de la conversión")
    style(a)
    b = fig.add_subplot(gs[1])
    filas = [("Sin ajuste (LPM)", m.params["trat"], m.conf_int().loc["trat"], CTRL), ("Con covariables", ma.params["trat"], ma.conf_int().loc["trat"], BLUE), ("CUPED (gasto_pre)", mc.params["trat"], mc.conf_int().loc["trat"], TRAT)]
    for i, (nombre, v, ci_, col) in enumerate(filas):
        yy = 2 - i
        b.errorbar([v * 100], [yy], xerr=[[(v - ci_[0]) * 100], [(ci_[1] - v) * 100]], fmt="o", color=col, capsize=5, lw=2.2, ms=7)
        b.text(v * 100, yy + 0.25, f"{v * 100:.2f} pp · ancho IC {(ci_[1] - ci_[0]) * 100:.2f}", ha="center", fontsize=8, color=col, fontweight="bold")
    b.axvline(0, color=INK, lw=1)
    b.set_yticks([2, 1, 0])
    b.set_yticklabels([f[0] for f in filas], fontsize=8.4)
    b.set_ylim(-0.6, 2.7)
    b.set_xlim(0.4, 3.4)
    b.set_xlabel("efecto estimado (puntos porcentuales)")
    titulo(b, "Misma estimación, intervalo algo más estrecho")
    style(b, "x")
    c = fig.add_subplot(gs[2])
    r = np.linspace(0, 0.9, 100)
    c.plot(r, r ** 2 * 100, color=VIOLET, lw=2)
    c.scatter([np.sqrt(rho2)], [red * 100], color=RED, zorder=5, s=55)
    c.text(np.sqrt(rho2) - 0.28, red * 100 + 14, f"ρ = {np.sqrt(rho2):.2f}\nreduce {red:.1%}", fontsize=8.4, color=RED)
    c.set_xlabel("correlación ρ entre Y y la covariable previa")
    c.set_ylabel("reducción de la varianza (%)")
    titulo(c, "CUPED reduce la varianza en ρ² (≈ 8.9 %)")
    style(c)
    save(fig, "c22_cuped.png")


# ------------------------------------------------------------------ 9. heterogeneidad
def hetero_fig():
    filas = []
    for col, niveles in [("dispositivo", ["desktop", "movil", "tablet"]), ("region", ["RM", "Valparaiso", "Biobio", "Otras"])]:
        for lev in niveles:
            d = ab[ab[col] == lev]
            c_, t_ = d[d["trat"] == 0]["convierte"], d[d["trat"] == 1]["convierte"]
            df_ = t_.mean() - c_.mean()
            se = np.sqrt(c_.var() / len(c_) + t_.var() / len(t_))
            zz = df_ / se
            filas.append((f"{col}: {lev}", df_ * 100, (df_ - Z * se) * 100, (df_ + Z * se) * 100, 2 * stats.norm.sf(abs(zz)), len(d)))
    h = smf.ols("convierte ~ trat * C(dispositivo)", ab).fit()
    h_null = smf.ols("convierte ~ trat + C(dispositivo)", ab).fit()
    ft = h.compare_f_test(h_null)
    hr = smf.ols("convierte ~ trat * C(region)", ab).fit()
    ft_r = hr.compare_f_test(smf.ols("convierte ~ trat + C(region)", ab).fit())
    NUM["hetero"] = dict(subgrupos={f[0]: dict(efecto=round(float(f[1]), 2), ic=[round(float(f[2]), 2), round(float(f[3]), 2)], p=round(float(f[4]), 4), n=int(f[5])) for f in filas},
                         modelo=dict(b_trat=round(float(h.params["trat"]) * 100, 2), int_movil=round(float(h.params["trat:C(dispositivo)[T.movil]"]) * 100, 2), int_tablet=round(float(h.params["trat:C(dispositivo)[T.tablet]"]) * 100, 2),
                                     p_movil=round(float(h.pvalues["trat:C(dispositivo)[T.movil]"]), 3), p_tablet=round(float(h.pvalues["trat:C(dispositivo)[T.tablet]"]), 3),
                                     ic_int_movil=[round(float(v) * 100, 2) for v in h.conf_int().loc["trat:C(dispositivo)[T.movil]"]], ic_int_tablet=[round(float(v) * 100, 2) for v in h.conf_int().loc["trat:C(dispositivo)[T.tablet]"]],
                                     F_conjunto=round(float(ft[0]), 3), p_conjunto=round(float(ft[1]), 3), F_region=round(float(ft_r[0]), 3), p_region=round(float(ft_r[1]), 3)))
    fig = plt.figure(figsize=(12.5, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    a = fig.add_subplot(gs[0])
    for i, (nombre, v, lo, hi, pv, n_) in enumerate(filas):
        yy = len(filas) - 1 - i
        col = TRAT if pv < 0.05 else CTRL
        a.errorbar([v], [yy], xerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=4, lw=2, ms=6)
        a.text(hi + 0.4, yy, f"{v:.2f} pp  (n = {n_:,}, p {'< 0.001' if pv < 0.001 else '= ' + format(pv, '.3f')})".replace(",", " "), va="center", fontsize=7.8, color=INK)
    a.axvline(0, color=INK, lw=1)
    a.axvline(dif * 100, color=TRAT, ls="--", lw=1)
    a.set_yticks(range(len(filas)))
    a.set_yticklabels([f[0] for f in filas][::-1], fontsize=8.2)
    a.set_xlim(-4.5, 12)
    a.set_xlabel("efecto del tratamiento en cada subgrupo (pp)")
    titulo(a, "Efecto por subgrupo: parecidos, con intervalos anchos")
    style(a, "x")
    b = fig.add_subplot(gs[1])
    b.axis("off")
    txt = (f"Modelo lm(convierte ~ trat * dispositivo)\n\n"
           f"efecto en desktop (referencia): {h.params['trat'] * 100:.2f} pp\n"
           f"interacción móvil: {h.params['trat:C(dispositivo)[T.movil]'] * 100:+.2f} pp (p = {h.pvalues['trat:C(dispositivo)[T.movil]']:.2f})\n"
           f"interacción tablet: {h.params['trat:C(dispositivo)[T.tablet]'] * 100:+.2f} pp (p = {h.pvalues['trat:C(dispositivo)[T.tablet]']:.2f})\n"
           f"prueba conjunta (F): p = {ft[1]:.2f}\n\n"
           "Conclusión: no hay evidencia suficiente de\nque el efecto cambie con el dispositivo.\n"
           "Pero «sin evidencia» ≠ «efecto igual»:\nel IC de cada interacción es ancho.")
    b.text(0.02, 0.97, txt, va="top", fontsize=8.6, color=INK, family="monospace", bbox=dict(fc=C22_BG, ec=C22, boxstyle="round,pad=0.5"))
    save(fig, "c22_hetero.png")


# ------------------------------------------------------------------ 10. DiD y tendencias paralelas
def did_fig():
    g = did.groupby(["zona", "mes"])["uso_app"].mean().unstack(0)
    meses = list(g.index)
    pre_m, post_m = meses[:3], meses[3:]
    cm = did.groupby(["trat", "post"])["uso_app"].mean()
    did_est = (cm[1, 1] - cm[1, 0]) - (cm[0, 1] - cm[0, 0])
    m = smf.ols("uso_app ~ trat * post", did).fit()
    grp = pd.factorize(did["comuna"])[0]
    mcl = smf.ols("uso_app ~ trat * post", did).fit(cov_type="cluster", cov_kwds={"groups": grp})
    fe = smf.ols("uso_app ~ trat:post + C(comuna) + C(mes)", did).fit()
    fecl = smf.ols("uso_app ~ trat:post + C(comuna) + C(mes)", did).fit(cov_type="cluster", cov_kwds={"groups": grp})
    did["t"] = did["mes"].map({mm: i for i, mm in enumerate(meses)})
    pre = did[did["post"] == 0]
    pt = smf.ols("uso_app ~ trat * t", pre).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pre["comuna"])[0]})
    brecha = (g["RM"] - g["otras"])
    otros = [mm for mm in meses if mm != "2026-03"]
    for mm in otros:
        did["ev_" + mm[-2:]] = ((did["mes"] == mm) & (did["trat"] == 1)).astype(int)
    formula = "uso_app ~ " + " + ".join("ev_" + mm[-2:] for mm in otros) + " + C(comuna) + C(mes)"
    ev = smf.ols(formula, did).fit(cov_type="cluster", cov_kwds={"groups": grp})
    evs = {}
    for mm in meses:
        key = "ev_" + mm[-2:]
        if key in ev.params.index:
            evs[mm] = (float(ev.params[key]), float(ev.conf_int().loc[key, 0]), float(ev.conf_int().loc[key, 1]))
        else:
            evs[mm] = (0.0, 0.0, 0.0)
    NUM["did"] = dict(n=len(did), comunas=int(did["comuna"].nunique()), tratadas=int(did[did["trat"] == 1]["comuna"].nunique()), controles=int(did[did["trat"] == 0]["comuna"].nunique()),
                      medias={f"trat{t}_post{p}": round(float(cm[t, p]), 3) for t in (0, 1) for p in (0, 1)}, did=round(float(did_est), 4),
                      mco=dict(b_trat=round(float(m.params["trat"]), 3), b_post=round(float(m.params["post"]), 3), b_int=round(float(m.params["trat:post"]), 4), se=round(float(m.bse["trat:post"]), 4), ic=[round(float(v), 3) for v in m.conf_int().loc["trat:post"]], p=float(f"{m.pvalues['trat:post']:.3g}")),
                      cluster=dict(se=round(float(mcl.bse["trat:post"]), 4), ic=[round(float(v), 3) for v in mcl.conf_int().loc["trat:post"]], p=float(f"{mcl.pvalues['trat:post']:.3g}")),
                      fe=dict(b=round(float(fe.params["trat:post"]), 4), se=round(float(fe.bse["trat:post"]), 4), ic=[round(float(v), 3) for v in fe.conf_int().loc["trat:post"]], se_cluster=round(float(fecl.bse["trat:post"]), 4), ic_cluster=[round(float(v), 3) for v in fecl.conf_int().loc["trat:post"]]),
                      pretendencia=dict(coef=round(float(pt.params["trat:t"]), 4), se=round(float(pt.bse["trat:t"]), 4), p=round(float(pt.pvalues["trat:t"]), 3)),
                      brecha_mensual={k: round(float(v), 3) for k, v in brecha.items()}, tabla_mensual={z_: {k: round(float(v), 3) for k, v in g[z_].items()} for z_ in g.columns},
                      evento={k: [round(v, 3) for v in vals] for k, vals in evs.items()})
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.9], wspace=0.3)
    a = fig.add_subplot(gs[0])
    xs = np.arange(6)
    a.plot(xs, g["RM"].values, "o-", color=TRAT, lw=2.2, label="RM (tratada)")
    a.plot(xs, g["otras"].values, "o-", color=CTRL, lw=2.2, label="otras regiones (control)")
    gap_pre = g["RM"].values[:3].mean() - g["otras"].values[:3].mean()
    cf = g["otras"].values[2:] + gap_pre
    a.plot(xs[2:], cf, "o--", color=AMBER, lw=1.8, label="RM sin lanzamiento (contrafactual)")
    a.axvline(2.5, color=INK, ls=":", lw=1.3)
    a.annotate("", xy=(5, g["RM"].values[5]), xytext=(5, cf[-1]), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    a.text(4.85, (g["RM"].values[5] + cf[-1]) / 2, f"brecha extra\nen junio\n≈ {g['RM'].values[5] - cf[-1]:.1f}", ha="right", fontsize=8.4, color=RED, fontweight="bold")
    a.text(2.57, a.get_ylim()[0] + 1, "lanzamiento", fontsize=8, color=INK)
    a.set_xticks(xs)
    a.set_xticklabels([mm[-2:] + "/26" for mm in meses])
    a.set_ylabel("uso promedio de la app")
    a.legend(fontsize=7.6, frameon=False, loc="upper left")
    titulo(a, "Antes del lanzamiento las líneas son paralelas")
    style(a)
    b = fig.add_subplot(gs[1])
    for i, mm in enumerate(meses):
        v, lo, hi = evs[mm]
        col = CTRL if i < 3 else TRAT
        b.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=4, lw=2, ms=6)
    b.axhline(0, color=INK, lw=1)
    b.axvline(2.5, color=INK, ls=":", lw=1.3)
    b.set_xticks(xs)
    b.set_xticklabels([mm[-2:] + "/26" for mm in meses])
    b.set_ylabel("brecha RM − otras, respecto de marzo")
    titulo(b, "Estudio de eventos: plano antes, salto después")
    style(b)
    c = fig.add_subplot(gs[2])
    c.axis("off")
    filas = [["", "Antes", "Después", "Δ"],
             ["RM", f"{cm[1, 0]:.2f}", f"{cm[1, 1]:.2f}", f"{cm[1, 1] - cm[1, 0]:.2f}"],
             ["Otras", f"{cm[0, 0]:.2f}", f"{cm[0, 1]:.2f}", f"{cm[0, 1] - cm[0, 0]:.2f}"],
             ["DiD", "", "", f"{did_est:.2f}"]]
    tb = c.table(cellText=filas, cellLoc="center", loc="center", colWidths=[0.22, 0.24, 0.27, 0.22])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.9)
    for (r, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(C22)
            cell.set_text_props(color="white", fontweight="bold")
        if r == 3:
            cell.set_text_props(fontweight="bold", color=RED)
    titulo(c, "2 × 2 de medias (antes = ene–mar)\nDiD = 10.90 − 5.14", fs=9.0)
    save(fig, "c22_did.png")


PASOS = dict(esquema=esquema_fig, causal=causal_fig, balance=balance_fig, poder=poder_fig, tasas=tasas_fig, pvalor=pvalor_fig, srm=srm_fig, peeking=peeking_fig,
             cuped=cuped_fig, hetero=hetero_fig, did=did_fig)

if __name__ == "__main__":
    import sys
    elegidos = sys.argv[1:] or list(PASOS)
    # el resumen del efecto se necesita siempre
    NUM["efecto"] = dict(n0=n0, n1=n1)
    for k in elegidos:
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c22_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
