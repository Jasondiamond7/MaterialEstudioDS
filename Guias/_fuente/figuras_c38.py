"""Figuras de la Clase 38 (Evaluación de modelos supervisados y selección de modelos).
Fuentes: diapositivas DipDS_2026_S38.pdf, 38_evaluacion_seleccion.py / .R.txt del profesor Eloy Alvarado y los dos CSV de la clase
(hormigon_s38.csv: regresión, 1 030 mediciones; spam_s38.csv: clasificación, 4 601 correos). La carpeta de la clase se lee sin modificarla.
Todo sale de código; lo aleatorio usa semilla fija. Escribe c38_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c38.py   (o con nombres: python -X utf8 figuras_c38.py metricas oof)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C38 = "#65A30D"      # color de la clase 38 (verde lima)
C38_BG = "#F7FEE7"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA38 = next(p for p in BASE.glob("Clase 38*") if p.is_dir())
V8 = ["cemento", "escoria", "ceniza", "agua", "superplastificante", "agregado_grueso", "agregado_fino", "edad"]
I7 = V8[:7]
_DAT = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def datos():
    """Lee los dos CSV de la clase (una sola vez) y deja las particiones del profesor."""
    if _DAT:
        return _DAT
    h = pd.read_csv(CARPETA38 / "hormigon_s38.csv")
    s = pd.read_csv(CARPETA38 / "spam_s38.csv")
    vs = [c for c in s if c.startswith(("word_freq_", "char_freq_", "capital_run_"))]
    _DAT.update(htr=h[h.conjunto == "entrenamiento"].reset_index(drop=True), hte=h[h.conjunto == "prueba"].reset_index(drop=True),
                str_=s[s.conjunto == "entrenamiento"].reset_index(drop=True), ste=s[s.conjunto == "prueba"].reset_index(drop=True), vs=vs)
    return _DAT


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def ridge_cuad(alpha, grado=2):
    """Receta del profesor: estandarizar, expandir términos, estandarizar de nuevo, ridge."""
    return make_pipeline(StandardScaler(), PolynomialFeatures(grado, include_bias=False), StandardScaler(), Ridge(alpha=alpha, solver="svd"))


def candidatos_reg():
    m = {"MEDIA": DummyRegressor(strategy="mean"), "LINEAL": make_pipeline(StandardScaler(), LinearRegression())}
    for a in (0.1, 1, 10, 100, 1000):
        m[f"Q_A{a:g}"] = ridge_cuad(a)
    return m


def oof_reg(modelo, x, y, fold):
    """Predicciones fuera de pliegue: cada fila se predice con un modelo que no la vio (ni a su composición)."""
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        v = fold == k
        oof[v] = clone(modelo).fit(x[~v], y[~v]).predict(x[v])
    return oof


def folds_por_grupo(grupos, K, rng):
    """Reparte GRUPOS completos en K pliegues al azar (una composición nunca queda a ambos lados)."""
    ug = np.unique(grupos)
    asign = dict(zip(rng.permutation(ug), np.arange(len(ug)) % K))
    return np.array([asign[g] for g in grupos])


def nombre(clave):
    return "Cuadrática α=" + clave[3:] if clave.startswith("Q_A") else {"MEDIA": "Media de entrenamiento", "LINEAL": "Regresión lineal"}[clave]


# ------------------------------------------------------------------ 1. pérdida y métricas: MAE, RMSE, R²
def metricas_fig():
    e = np.linspace(-4, 4, 200)
    A = np.array([0, 0, 0, 8.0])
    B = np.array([3, 3, 3, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    yh = y + 10
    NUM["metricas"] = dict(a_mae=float(np.mean(A)), a_rmse=float(np.sqrt(np.mean(A ** 2))), b_mae=float(np.mean(B)), b_rmse=float(np.sqrt(np.mean(B ** 2))),
                           corr=float(np.corrcoef(y, yh)[0, 1]), rmse_corr=rmse(y, yh), r2_corr=r2(y, yh), sse_corr=float(np.sum((y - yh) ** 2)), sst_corr=float(np.sum((y - y.mean()) ** 2)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.plot(e, np.abs(e), color=BLUE, lw=2.4, label="absoluta |e|  (MAE)")
    a.plot(e, e ** 2, color=AMBER, lw=2.4, label="cuadrática e²  (RMSE, R²)")
    a.annotate("error 2 → 2 vs 4", xy=(2, 4), xytext=(0.2, 11), fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    a.annotate("error 4 → 4 vs 16", xy=(4, 16), xytext=(1.2, 14.4), fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    a.set_xlabel("error  e = y − ŷ  (MPa)")
    a.set_ylabel("pérdida")
    a.set_ylim(0, 17)
    a.legend(fontsize=7.8, frameon=False, loc="upper center")
    titulo(a, "1. La pérdida define qué es equivocarse")
    style(a)
    b = fig.add_subplot(gs[1])
    x = np.arange(4)
    b.bar(x - 0.2, A, 0.38, color=RED, alpha=0.85, label=f"A: MAE = {A.mean():.0f}, RMSE = {np.sqrt((A ** 2).mean()):.0f}")
    b.bar(x + 0.2, B, 0.38, color=TEAL, alpha=0.85, label=f"B: MAE = {B.mean():.0f}, RMSE = {np.sqrt((B ** 2).mean()):.0f}")
    b.set_xticks(x)
    b.set_xticklabels(["caso 1", "caso 2", "caso 3", "caso 4"])
    b.set_ylabel("error absoluto (MPa)")
    b.set_ylim(0, 11.5)
    b.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(b, "2. Con MAE gana A; con RMSE gana B")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    c.plot([0, 14], [0, 14], color=MUTED, ls="--", lw=1.2, label="predicción perfecta")
    c.scatter(y, yh, color=RED, s=55, zorder=3, label="y = (1, 2, 3);  ŷ = y + 10")
    for xi, yi in zip(y, yh):
        c.plot([xi, xi], [xi, yi], color=RED, lw=1, alpha=0.5)
    c.text(0.4, 10.6, f"correlación = {NUM['metricas']['corr']:.0f}\nRMSE = {NUM['metricas']['rmse_corr']:.0f}\nR² = 1 − {NUM['metricas']['sse_corr']:.0f}/{NUM['metricas']['sst_corr']:.0f} = {NUM['metricas']['r2_corr']:.0f}", fontsize=8.6, color=INK, fontweight="bold")
    c.set_xlabel("observado y")
    c.set_ylabel("predicho ŷ")
    c.set_xlim(0, 4)
    c.set_ylim(0, 14)
    c.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(c, "3. Correlación 1 y R² = −149")
    style(c)
    save(fig, "c38_metricas.png")


# ------------------------------------------------------------------ 2. partición por grupos, pliegues y OOF
def oof_fig():
    d = datos()
    htr, hte = d["htr"], d["hte"]
    x, y, fold = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.fold.to_numpy()
    r = {}
    for nom, m in candidatos_reg().items():
        o = oof_reg(m, x, y, fold)
        r[nom] = dict(rmse=rmse(y, o), mae=mae(y, o), r2=r2(y, o), oof=o)
    por_fold = {k: [rmse(y[fold == f], r[k]["oof"][fold == f]) for f in range(1, 6)] for k in ("Q_A0.1", "Q_A1")}
    tam = [int((fold == f).sum()) for f in range(1, 6)]
    NUM["particion"] = dict(filas=int(len(htr) + len(hte)), grupos=int(htr.grupo.nunique() + hte.grupo.nunique()), filas_tr=int(len(htr)), grupos_tr=int(htr.grupo.nunique()),
                            filas_te=int(len(hte)), grupos_te=int(hte.grupo.nunique()), tam_pliegue=tam, pct_grupos_te=float(hte.grupo.nunique() / (htr.grupo.nunique() + hte.grupo.nunique())))
    NUM["cv"] = dict(rmse_oof_01=r["Q_A0.1"]["rmse"], rmse_oof_1=r["Q_A1"]["rmse"], por_fold_01=por_fold["Q_A0.1"], por_fold_1=por_fold["Q_A1"],
                     prom_rmse_01=float(np.mean(por_fold["Q_A0.1"])), prom_rmse_1=float(np.mean(por_fold["Q_A1"])),
                     gana_01_pliegues=int(sum(a < b for a, b in zip(por_fold["Q_A0.1"], por_fold["Q_A1"]))),
                     dif_01_1=r["Q_A0.1"]["rmse"] - r["Q_A1"]["rmse"], ajustes=7 * 5,
                     ejemplo_mse=[1, 9], ejemplo_prom_rmse=2.0, ejemplo_rmse_oof=float(np.sqrt(5)))
    fig = plt.figure(figsize=(12.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.75], wspace=0.32)
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 10)
    a.set_ylim(-0.1, 6.3)
    a.axis("off")
    a.text(0, 6.0, f"{NUM['particion']['filas']:,} mediciones = {NUM['particion']['grupos']} composiciones distintas".replace(",", " "), fontsize=8.6, color=INK, fontweight="bold")
    a.add_patch(Rectangle((0, 5.0), 8.0, 0.7, fc=BLUE, ec="white", alpha=0.85))
    a.add_patch(Rectangle((8.0, 5.0), 2.0, 0.7, fc=AMBER, ec="white", alpha=0.9))
    a.text(4.0, 5.35, f"entrenamiento: {NUM['particion']['filas_tr']} filas · {NUM['particion']['grupos_tr']} composiciones", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    a.text(9.0, 5.35, f"prueba\n{NUM['particion']['filas_te']} · {NUM['particion']['grupos_te']}", ha="center", va="center", fontsize=7.4, color="white", fontweight="bold")
    a.text(0, 4.5, "Validación cruzada de 5 pliegues por composición:", fontsize=8.2, color=INK)
    for it in range(5):
        for k in range(5):
            a.add_patch(Rectangle((k * 1.6, 3.7 - it * 0.62), 1.55, 0.5, fc=AMBER if k == it else BLUE, ec="white", alpha=0.9 if k == it else 0.55))
        a.text(8.2, 3.95 - it * 0.62, f"ajuste {it + 1}", fontsize=7.6, color=MUTED, va="center")
    a.text(0.0, 0.35, "azul: filas de ajuste · ámbar: pliegue de validación (OOF)\nla prueba no se toca hasta el final", fontsize=7.6, color=MUTED)
    titulo(a, "1. Se separan composiciones, no filas")
    b = fig.add_subplot(gs[1])
    xs = np.arange(1, 6)
    b.bar(xs - 0.2, por_fold["Q_A0.1"], 0.38, color=C38, alpha=0.9, label="α = 0.1")
    b.bar(xs + 0.2, por_fold["Q_A1"], 0.38, color=VIOLET, alpha=0.8, label="α = 1")
    b.axhline(r["Q_A0.1"]["rmse"], color=C38, ls="--", lw=1.4)
    b.text(0.55, r["Q_A0.1"]["rmse"] + 0.12, f"RMSE OOF {r['Q_A0.1']['rmse']:.3f}", ha="left", fontsize=7.8, color=C38, fontweight="bold")
    b.set_ylim(6, 10.3)
    b.set_xlabel("pliegue de composiciones excluidas")
    b.set_ylabel("RMSE por pliegue (MPa)")
    b.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(b, "2. Una cifra resume cinco pliegues distintos")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    c.bar([0, 1], [2.0, np.sqrt(5)], color=[BLUE, C38], width=0.55)
    c.text(0, 2.0 + 0.06, "2.000", ha="center", fontsize=8.6, fontweight="bold")
    c.text(1, np.sqrt(5) + 0.06, "2.236", ha="center", fontsize=8.6, fontweight="bold")
    c.set_xticks([0, 1])
    c.set_xticklabels(["promedio\nde RMSE", "RMSE\nOOF"], fontsize=8.2)
    c.set_ylim(0, 2.8)
    c.set_ylabel("MSE de los pliegues: 1 y 9")
    titulo(c, "3. La raíz no es lineal")
    style(c, "y")
    save(fig, "c38_oof.png")


# ------------------------------------------------------------------ 3. la elección de CV es una variable aleatoria: particiones repetidas y K
def repetido_fig():
    d = datos()
    htr = d["htr"]
    x, y, g = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.grupo.to_numpy()
    rng = np.random.default_rng(RS)
    R = 40
    alfas = (0.1, 1, 10)
    res = {a: [] for a in alfas}
    for _ in range(R):
        f = folds_por_grupo(g, 5, rng)
        for a in alfas:
            res[a].append(rmse(y, oof_reg(ridge_cuad(a), x, y, f)))
    res = {a: np.array(v) for a, v in res.items()}
    dif = res[0.1] - res[1]
    sweep = {}
    for K in (2, 3, 5, 10, 20):
        v = [rmse(y, oof_reg(ridge_cuad(0.1), x, y, folds_por_grupo(g, K, rng))) for _ in range(12)]
        sweep[K] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    ug = np.unique(g)
    logo = rmse(y, oof_reg(ridge_cuad(0.1), x, y, np.searchsorted(ug, g)))        # dejar un grupo fuera (342 ajustes)
    NUM["repetido"] = dict(R=R, media={str(a): float(res[a].mean()) for a in alfas}, sd={str(a): float(res[a].std(ddof=1)) for a in alfas},
                           min={str(a): float(res[a].min()) for a in alfas}, max={str(a): float(res[a].max()) for a in alfas},
                           p_gana_01=float((dif < 0).mean()), dif_media=float(dif.mean()), dif_sd=float(dif.std(ddof=1)),
                           sweep={str(k): list(v) for k, v in sweep.items()}, logo=logo, n_grupos=int(len(ug)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    cols = {0.1: C38, 1: VIOLET, 10: AMBER}
    for i, al in enumerate(alfas):
        a.scatter(np.full(R, i) + np.linspace(-0.18, 0.18, R), res[al], s=14, color=cols[al], alpha=0.8)
        a.plot([i - 0.25, i + 0.25], [res[al].mean()] * 2, color=INK, lw=2)
    a.axhline(NUM["cv"]["rmse_oof_01"], color=C38, ls=":", lw=1.2)
    a.set_xticks(range(3))
    a.set_xticklabels([f"α = {al}" for al in alfas])
    a.set_ylabel("RMSE OOF (MPa)")
    titulo(a, f"1. {R} particiones al azar: el RMSE OOF varía")
    style(a, "y")
    b = fig.add_subplot(gs[1])
    b.hist(dif, bins=14, color=C38, alpha=0.8, edgecolor="white")
    b.axvline(0, color=INK, lw=1.4)
    b.set_xlabel("RMSE(α = 0.1) − RMSE(α = 1)   (< 0 favorece 0.1)")
    b.set_ylabel("particiones")
    b.text(0.02, 0.95, f"α = 0.1 gana en {NUM['repetido']['p_gana_01'] * 100:.0f} %\nde las particiones", transform=b.transAxes, va="top", fontsize=8.6, color=INK, fontweight="bold")
    titulo(b, f"2. Gana casi siempre, pero por ~{abs(NUM['repetido']['dif_media']):.2f} MPa")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    ks = list(sweep)
    c.errorbar(range(len(ks)), [sweep[k][0] for k in ks], yerr=[sweep[k][1] for k in ks], fmt="o-", color=C38, lw=2, capsize=3, label="K pliegues (12 particiones)")
    c.scatter([len(ks)], [logo], color=RED, s=60, zorder=4, label=f"un grupo fuera ({NUM['repetido']['n_grupos']} ajustes)")
    c.set_xticks(range(len(ks) + 1))
    c.set_xticklabels([f"K={k}" for k in ks] + ["LOGO"], fontsize=8)
    c.set_ylabel("RMSE OOF (MPa), α = 0.1")
    c.legend(fontsize=7.6, frameon=False, loc="upper right")
    titulo(c, "3. K pequeño entrena con menos datos")
    style(c, "y")
    save(fig, "c38_cv_repetido.png")


# ------------------------------------------------------------------ 4. bootstrap por composición y estimador .632
def ajuste_final_reg():
    d = datos()
    htr, hte = d["htr"], d["hte"]
    m = ridge_cuad(0.1).fit(htr[V8].to_numpy(), htr.resistencia.to_numpy())
    base = DummyRegressor(strategy="mean").fit(htr[V8].to_numpy(), htr.resistencia.to_numpy())
    return m, base, m.predict(hte[V8].to_numpy()), base.predict(hte[V8].to_numpy())


def bootstrap_fig():
    d = datos()
    htr, hte = d["htr"], d["hte"]
    x, y, g = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.grupo.to_numpy()
    ug = np.unique(g)
    n = len(ug)
    por_g = {gg: np.where(g == gg)[0] for gg in ug}
    rng = np.random.default_rng(RS)
    B = 200
    mse_oob, dentro = [], []
    for _ in range(B):
        samp = rng.choice(ug, n, replace=True)
        filas = np.concatenate([por_g[s] for s in samp])
        fuera = np.setdiff1d(ug, np.unique(samp))
        ofilas = np.concatenate([por_g[s] for s in fuera])
        m = ridge_cuad(0.1).fit(x[filas], y[filas])
        mse_oob.append(float(np.mean((y[ofilas] - m.predict(x[ofilas])) ** 2)))
        dentro.append(1 - len(fuera) / n)
    full, base, pt, pb = ajuste_final_reg()
    mse_res = float(np.mean((y - full.predict(x)) ** 2))
    mse_b = float(np.mean(mse_oob))
    mse632 = 0.368 * mse_res + 0.632 * mse_b
    yt, gt = hte.resistencia.to_numpy(), hte.grupo.to_numpy()
    ugt = np.unique(gt)
    por_gt = {gg: np.where(gt == gg)[0] for gg in ugt}
    ra, rb, rd = [], [], []
    for _ in range(2000):
        f = np.concatenate([por_gt[s] for s in rng.choice(ugt, len(ugt), replace=True)])
        ra.append(rmse(yt[f], pt[f]))
        rb.append(rmse(yt[f], pb[f]))
        rd.append(rb[-1] - ra[-1])
    ci = [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]
    cid = [float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))]
    NUM["bootstrap"] = dict(B=B, n_grupos=int(n), frac_dentro=float(np.mean(dentro)), frac_teorica=float(1 - (1 - 1 / n) ** n), limite=float(1 - np.exp(-1)),
                            rmse_resub=float(np.sqrt(mse_res)), rmse_oob=float(np.sqrt(mse_b)), rmse_632=float(np.sqrt(mse632)), rmse_cv=NUM["cv"]["rmse_oof_01"],
                            rmse_test=rmse(yt, pt), ci_test=ci, rmse_base_test=rmse(yt, pb), ci_dif=cid, mse_resub=mse_res, mse_oob=mse_b, mse_632=mse632)
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 1.0], wspace=0.35)
    a = fig.add_subplot(gs[0])
    a.hist(dentro, bins=14, color=C38, alpha=0.85, edgecolor="white")
    a.axvline(0.632, color=INK, ls="--", lw=1.4)
    a.text(0.633, a.get_ylim()[1] * 0.92, " 1 − 1/e ≈ 0.632", fontsize=8, color=INK)
    a.set_xlabel("fracción de composiciones en la muestra")
    a.set_ylabel(f"remuestras (de {B})")
    titulo(a, "1. Cada remuestra deja ≈ 36.8 % fuera (OOB)")
    style(a, "y")
    b = fig.add_subplot(gs[1])
    et = ["resustitución\n(error de ajuste)", "bootstrap OOB", "CV 5 pliegues", "estimador .632", "prueba reservada"]
    vl = [NUM["bootstrap"]["rmse_resub"], NUM["bootstrap"]["rmse_oob"], NUM["bootstrap"]["rmse_cv"], NUM["bootstrap"]["rmse_632"], NUM["bootstrap"]["rmse_test"]]
    cl = [RED, AMBER, C38, VIOLET, BLUE]
    b.barh(range(5), vl, color=cl, alpha=0.9)
    for i, v in enumerate(vl):
        b.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=8.4, fontweight="bold")
    b.set_yticks(range(5))
    b.set_yticklabels(et, fontsize=8)
    b.invert_yaxis()
    b.set_xlim(0, 11.3)
    b.set_xlabel("RMSE (MPa), receta α = 0.1")
    titulo(b, "2. Optimista, pesimista y en el medio")
    style(b, "x")
    c = fig.add_subplot(gs[2])
    c.hist(ra, bins=40, color=BLUE, alpha=0.8, edgecolor="white")
    c.axvline(ci[0], color=INK, ls="--", lw=1.2)
    c.axvline(ci[1], color=INK, ls="--", lw=1.2)
    c.axvline(NUM["bootstrap"]["rmse_test"], color=RED, lw=1.8)
    c.text(0.03, 0.95, f"RMSE prueba {NUM['bootstrap']['rmse_test']:.2f}\nIC bootstrap 95 %\n[{ci[0]:.2f}; {ci[1]:.2f}]", transform=c.transAxes, va="top", fontsize=8.4, color=INK)
    c.set_xlabel("RMSE en prueba (MPa), 2 000 remuestras")
    c.set_ylabel("remuestras")
    titulo(c, "3. Incertidumbre de la cifra de prueba")
    style(c, "y")
    save(fig, "c38_bootstrap.png")


# ------------------------------------------------------------------ 5. sesgo, varianza y ruido (simulación)
def senal(x):
    return 1.5 * np.sin(2.2 * x) + 0.8 * x - 0.5


def sesgo_varianza_fig():
    rng = np.random.default_rng(RS)
    n, sig, reps = 30, 0.6, 400
    grados = np.arange(1, 9)
    xg = np.linspace(-0.9, 0.9, 101)
    pred = np.zeros((reps, len(grados), len(xg)))
    tr_err = np.zeros((reps, len(grados)))
    for r in range(reps):
        xs = np.sort(rng.uniform(-1, 1, n))
        ys = senal(xs) + rng.normal(0, sig, n)
        for j, gd in enumerate(grados):
            p = np.polynomial.Polynomial.fit(xs, ys, gd)
            pred[r, j] = p(xg)
            tr_err[r, j] = np.mean((ys - p(xs)) ** 2)
    f0 = senal(xg)
    sesgo2 = ((pred.mean(0) - f0) ** 2).mean(1)
    varia = pred.var(0, ddof=1).mean(1)
    total = sesgo2 + varia + sig ** 2
    NUM["sesgo_var"] = dict(n=n, sigma=sig, ruido=sig ** 2, reps=reps, grados=grados.tolist(), sesgo2=sesgo2.tolist(), varianza=varia.tolist(), total=total.tolist(),
                            rmse_train=np.sqrt(tr_err.mean(0)).tolist(), rmse_test=np.sqrt(total).tolist(), mejor_grado=int(grados[np.argmin(total)]))
    rng2 = np.random.default_rng(7)
    xs = np.sort(rng2.uniform(-1, 1, n))
    ys = senal(xs) + rng2.normal(0, sig, n)
    xn = rng2.uniform(-1, 1, 3000)
    yn = senal(xn) + rng2.normal(0, sig, 3000)
    uno = {}
    for gd in (2, 12):
        p = np.polynomial.Polynomial.fit(xs, ys, gd)
        uno[gd] = dict(train=rmse(ys, p(xs)), test=rmse(yn, p(xn)), curva=p(xg))
    NUM["sesgo_var"]["uno"] = {str(k): dict(train=v["train"], test=v["test"]) for k, v in uno.items()}
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.scatter(xs, ys, color=INK, s=16, alpha=0.7)
    a.plot(xg, f0, color=MUTED, ls="--", lw=1.6, label="señal verdadera")
    a.plot(xg, uno[2]["curva"], color=C38, lw=2.4, label=f"grado 2: ajuste {uno[2]['train']:.2f} · nuevos {uno[2]['test']:.2f}")
    a.plot(xg, uno[12]["curva"], color=RED, lw=2, label=f"grado 12: ajuste {uno[12]['train']:.2f} · nuevos {uno[12]['test']:.2f}")
    a.set_ylim(-3.2, 3.2)
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.legend(fontsize=7.4, frameon=False, loc="lower right")
    titulo(a, "1. Grado 12: mejor ajuste, peor en datos nuevos")
    style(a)
    b = fig.add_subplot(gs[1])
    b.stackplot(grados, sesgo2, varia, [sig ** 2] * len(grados), colors=[BLUE, AMBER, "#D1D5DB"], alpha=0.85, labels=["sesgo²", "varianza", "ruido σ² (no se elimina)"])
    b.plot(grados, total, color=INK, lw=1.6)
    b.set_ylim(0, 2.0)
    b.set_xlabel("grado del polinomio (complejidad)")
    b.set_ylabel("error cuadrático esperado")
    b.legend(fontsize=7.6, frameon=False, loc="upper left")
    titulo(b, "2. Error = sesgo² + varianza + ruido")
    style(b)
    c = fig.add_subplot(gs[2])
    c.plot(grados, NUM["sesgo_var"]["rmse_train"], "o-", color=VIOLET, lw=2, ms=4, label="error de ajuste")
    c.plot(grados, NUM["sesgo_var"]["rmse_test"], "o-", color=C38, lw=2.4, ms=4, label="error en datos nuevos")
    mg = NUM["sesgo_var"]["mejor_grado"]
    c.axvline(mg, color=INK, ls=":", lw=1.2)
    c.text(mg + 0.2, 1.25, f"mínimo: grado {mg}", fontsize=8, color=INK)
    c.set_ylim(0.4, 1.7)
    c.set_xlabel("grado del polinomio (complejidad)")
    c.set_ylabel("RMSE")
    c.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(c, "3. El ajuste siempre baja; lo nuevo hace una U")
    style(c)
    save(fig, "c38_sesgo_varianza.png")


# ------------------------------------------------------------------ 6. curvas de aprendizaje y de validación (hormigón)
def curvas_fig():
    d = datos()
    htr = d["htr"]
    x, y, g, fold = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.grupo.to_numpy(), htr.fold.to_numpy()
    rng = np.random.default_rng(RS)
    fr = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    mods = {"lineal": make_pipeline(StandardScaler(), LinearRegression()), "cuad": ridge_cuad(0.1)}
    out = {k: {"train": [], "valid": []} for k in mods}
    for f in fr:
        acum = {k: ([], []) for k in mods}
        for k in range(1, 6):
            v = fold == k
            pool = np.unique(g[~v])
            for _ in range(5 if f < 1 else 1):
                sel = rng.choice(pool, max(3, int(round(f * len(pool)))), replace=False)
                rows = np.where(~v & np.isin(g, sel))[0]
                for nm, m in mods.items():
                    ft = clone(m).fit(x[rows], y[rows])
                    acum[nm][0].append(np.mean((y[rows] - ft.predict(x[rows])) ** 2))
                    acum[nm][1].append(np.mean((y[v] - ft.predict(x[v])) ** 2))
        for nm in mods:
            out[nm]["train"].append(float(np.sqrt(np.mean(acum[nm][0]))))
            out[nm]["valid"].append(float(np.sqrt(np.mean(acum[nm][1]))))
    ntr = [int(round(f * 0.8 * len(htr))) for f in fr]
    alfas = np.logspace(-4, 3, 15)
    aj, ov = [], []
    for al in alfas:
        sse, nn = 0.0, 0
        oof = np.zeros(len(y))
        for k in range(1, 6):
            v = fold == k
            ft = ridge_cuad(al).fit(x[~v], y[~v])
            oof[v] = ft.predict(x[v])
            sse += np.sum((y[~v] - ft.predict(x[~v])) ** 2)
            nn += (~v).sum()
        aj.append(float(np.sqrt(sse / nn)))
        ov.append(rmse(y, oof))
    imin = int(np.argmin(ov))
    NUM["curvas"] = dict(frac=fr, n_entrena=ntr, lineal=out["lineal"], cuad=out["cuad"], alfas=alfas.tolist(), rmse_ajuste=aj, rmse_oof=ov, alfa_min=float(alfas[imin]), rmse_min=ov[imin],
                         rmse_oof_01=ov[int(np.argmin(abs(alfas - 0.1)))])
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    for i, (nm, tt, col) in enumerate([("lineal", "1. Regresión lineal: sesgo alto", RED), ("cuad", "2. Cuadrática α = 0.1: mejora con datos", C38)]):
        a = fig.add_subplot(gs[i])
        a.plot(ntr, out[nm]["train"], "o-", color=VIOLET, lw=2, ms=4, label="error de ajuste")
        a.plot(ntr, out[nm]["valid"], "o-", color=col, lw=2.4, ms=4, label="error de validación")
        a.set_ylim(3, 20)
        a.set_xlabel("filas usadas para ajustar")
        a.set_ylabel("RMSE (MPa)")
        a.legend(fontsize=7.8, frameon=False, loc="upper right")
        titulo(a, tt)
        style(a)
    c = fig.add_subplot(gs[2])
    c.plot(alfas, aj, "o-", color=VIOLET, lw=2, ms=4, label="ajuste")
    c.plot(alfas, ov, "o-", color=C38, lw=2.4, ms=4, label="validación OOF")
    c.axvspan(0.1, 1000, color="#E5E7EB", alpha=0.5)
    c.text(0.13, 12.2, "cuadrícula\ndel profesor", fontsize=7.8, color=MUTED)
    c.scatter([alfas[imin]], [ov[imin]], color=RED, zorder=5, s=40)
    c.set_xscale("log")
    c.set_ylim(6, 14)
    c.set_xlabel("α (más penalización a la derecha)")
    c.set_ylabel("RMSE (MPa)")
    c.legend(fontsize=7.8, frameon=False, loc="lower right")
    titulo(c, "3. Bajo α = 0.1 casi no hay ganancia")
    style(c)
    save(fig, "c38_curvas.png")


# ------------------------------------------------------------------ 7. hormigón: ranking, diferencias por pliegue y prueba
def hormigon_fig():
    d = datos()
    htr, hte = d["htr"], d["hte"]
    x, y, fold = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.fold.to_numpy()
    rk = []
    for nom, m in candidatos_reg().items():
        o = oof_reg(m, x, y, fold)
        rk.append(dict(modelo=nom, rmse=rmse(y, o), mae=mae(y, o), r2=r2(y, o)))
    rk = sorted(rk, key=lambda r: r["rmse"])
    m, base, pt, pb = ajuste_final_reg()
    yt = hte.resistencia.to_numpy()
    ev = dict(rmse=rmse(yt, pt), mae=mae(yt, pt), r2=r2(yt, pt))
    rf = dict(rmse=rmse(yt, pb), mae=mae(yt, pb), r2=r2(yt, pb))
    NUM["hormigon"] = dict(ranking=rk, prueba=ev, referencia=rf, mejora_rmse=float(1 - ev["rmse"] / rf["rmse"]), rmse_train_final=rmse(y, m.predict(x)),
                           ventaja_01_1=rk[1]["rmse"] - rk[0]["rmse"], n_prueba=int(len(hte)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.9, 1.0], wspace=0.5)
    a = fig.add_subplot(gs[0])
    ys_ = np.arange(len(rk))
    a.scatter([r["rmse"] for r in rk], ys_, color=C38, s=60, label="RMSE OOF (elige)", zorder=3)
    a.scatter([r["mae"] for r in rk], ys_, color=AMBER, s=40, marker="D", label="MAE OOF (describe)", zorder=3)
    for i, r in enumerate(rk):
        a.text(r["rmse"] + 0.5, i - 0.05, f"{r['rmse']:.2f}", fontsize=7.8, va="center")
    a.set_yticks(ys_)
    a.set_yticklabels([nombre(r["modelo"]) for r in rk], fontsize=8)
    a.invert_yaxis()
    a.set_xlim(4, 20)
    a.set_xlabel("MPa (menor es mejor)")
    a.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(a, "1. Siete candidatos, los mismos cinco pliegues")
    style(a, "x")
    b = fig.add_subplot(gs[1])
    dl = np.array(NUM["cv"]["por_fold_01"]) - np.array(NUM["cv"]["por_fold_1"])
    b.bar(range(1, 6), dl, color=np.where(dl < 0, C38, VIOLET))
    b.axhline(0, color=INK, lw=1)
    b.set_xlabel("pliegue")
    b.set_ylabel("RMSE(0.1) − RMSE(1)  (MPa)")
    b.text(0.5, 0.06, "bajo 0 favorece α = 0.1", transform=b.transAxes, ha="center", fontsize=7.8, color=MUTED)
    titulo(b, "2. Gana en 3 de 5 pliegues")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    c.scatter(yt, pt, color=BLUE, s=16, alpha=0.6)
    lim = [yt.min() - 2, yt.max() + 2]
    c.plot(lim, lim, "--", color=AMBER)
    c.text(0.04, 0.95, f"RMSE {ev['rmse']:.2f}  MAE {ev['mae']:.2f}\nR² {ev['r2']:.3f}\nmedia entrenamiento: RMSE {rf['rmse']:.2f}", transform=c.transAxes, va="top", fontsize=8, color=INK)
    c.set_xlabel("resistencia real (MPa)")
    c.set_ylabel("predicha (MPa)")
    titulo(c, f"3. Prueba reservada: {len(hte)} mediciones")
    style(c)
    save(fig, "c38_hormigon.png")


# ------------------------------------------------------------------ 8. diagnóstico: residuos y casi-duplicados
def diagnostico_fig():
    d = datos()
    htr, hte = d["htr"], d["hte"]
    m, base, pt, pb = ajuste_final_reg()
    yt = hte.resistencia.to_numpy()
    res = yt - pt
    sc = StandardScaler().fit(htr[I7])
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(htr[I7]))
    dist = nn.kneighbors(sc.transform(hte[I7]))[0][:, 0]
    q = pd.qcut(dist, 4, labels=False)
    por_q = [dict(n=int((q == k).sum()), dist=float(dist[q == k].mean()), rmse=float(np.sqrt(np.mean(res[q == k] ** 2)))) for k in range(4)]
    cerca = dist < 0.1
    NUM["dup"] = dict(por_cuartil=por_q, n_cerca=int(cerca.sum()), pct_cerca=float(cerca.mean()), rmse_cerca=float(np.sqrt(np.mean(res[cerca] ** 2))),
                      rmse_lejos=float(np.sqrt(np.mean(res[~cerca] ** 2))), umbral=0.1, mediana_dist=float(np.median(dist)),
                      media_res=float(res.mean()), pos=float((res > 0).mean()), max_abs=float(np.abs(res).max()))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.scatter(pt, res, color=BLUE, s=16, alpha=0.6)
    a.axhline(0, color=AMBER, ls="--")
    a.set_xlabel("predicha (MPa)")
    a.set_ylabel("real − predicha (MPa)")
    a.text(0.03, 0.05, "positivo = subestima", transform=a.transAxes, fontsize=7.8, color=MUTED)
    titulo(a, "1. ¿El promedio oculta un patrón?")
    style(a)
    b = fig.add_subplot(gs[1])
    b.scatter(hte.edad, np.abs(res), color=BLUE, s=16, alpha=0.6)
    b.set_xscale("log")
    b.set_xlabel("edad (días, escala log)")
    b.set_ylabel("error absoluto (MPa)")
    titulo(b, "2. Errores según edad")
    style(b)
    c = fig.add_subplot(gs[2])
    c.bar(range(4), [p["rmse"] for p in por_q], color=[C38, "#A3E635", AMBER, RED], alpha=0.9)
    for i, p in enumerate(por_q):
        c.text(i, p["rmse"] + 0.2, f"{p['rmse']:.1f}", ha="center", fontsize=8.6, fontweight="bold")
    c.set_xticks(range(4))
    c.set_xticklabels([f"Q{i + 1}\n(dist. {p['dist']:.2f})" for i, p in enumerate(por_q)], fontsize=8)
    c.set_xlabel("cuartil de distancia a la composición\nmás parecida del entrenamiento")
    c.set_ylabel("RMSE en prueba (MPa)")
    titulo(c, f"3. Casi-duplicados: {NUM['dup']['n_cerca']} filas, RMSE {NUM['dup']['rmse_cerca']:.1f}")
    style(c, "y")
    save(fig, "c38_diagnostico.png")


# ================================================================== CLASIFICACIÓN (spam)
_SP = {}


def modelos_spam():
    return {"FRECUENCIA": DummyClassifier(strategy="prior"),
            "LOGISTICA_C1": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, solver="lbfgs")),
            "BOSQUE_8": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, max_features="sqrt", random_state=20260924, n_jobs=1)}


NOM_SP = {"FRECUENCIA": "Frecuencia de spam", "LOGISTICA_C1": "Logística (C=1)", "BOSQUE_8": "Bosque (prof. 8)"}


def decision(y, p, t, cfp=5.0, cfn=1.0):
    vn, fp, fn, vp = confusion_matrix(y, np.asarray(p) >= t, labels=[0, 1]).ravel()
    n = vn + fp + fn + vp
    pr = vp / (vp + fp) if vp + fp else 0.0
    rc = vp / (vp + fn)
    return dict(umbral=float(t), vn=int(vn), fp=int(fp), fn=int(fn), vp=int(vp), accuracy=float((vn + vp) / n), precision=float(pr), recall=float(rc),
                f1=float(2 * vp / (2 * vp + fp + fn)), costo=float((cfp * fp + cfn * fn) / n))


def spam_oof():
    """Probabilidades OOF de los tres candidatos y del bosque recalibrado (isotónica anidada: la calibración se aprende solo con las filas de ajuste)."""
    if _SP:
        return _SP
    d = datos()
    tr, te, vs = d["str_"], d["ste"], d["vs"]
    x, y, fold, g = tr[vs].to_numpy(), tr.spam.to_numpy(), tr.fold.to_numpy(), tr.grupo.to_numpy()
    oof = {}
    for nm, m in modelos_spam().items():
        p = np.zeros(len(y))
        for k in range(1, 6):
            v = fold == k
            p[v] = clone(m).fit(x[~v], y[~v]).predict_proba(x[v])[:, 1]
        oof[nm] = p
    rng = np.random.default_rng(RS)
    pc = np.zeros(len(y))
    for k in range(1, 6):
        v = fold == k
        ids = folds_por_grupo(g[~v], 3, rng)
        splits = [(np.where(ids != j)[0], np.where(ids == j)[0]) for j in range(3)]
        cal = CalibratedClassifierCV(clone(modelos_spam()["BOSQUE_8"]), method="isotonic", cv=splits).fit(x[~v], y[~v])
        pc[v] = cal.predict_proba(x[v])[:, 1]
    oof["BOSQUE_CAL"] = pc
    _SP.update(x=x, y=y, fold=fold, oof=oof)
    return _SP


def probs(y, p):
    return dict(log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))


def spam_fig():
    sp = spam_oof()
    y = sp["y"]
    rk = {nm: probs(y, sp["oof"][nm]) for nm in ("FRECUENCIA", "LOGISTICA_C1", "BOSQUE_8")}
    NUM["spam"] = dict(n_tr=int(len(y)), prev=float(y.mean()), n0=int((y == 0).sum()), n1=int((y == 1).sum()), ranking=rk,
                       log_loss_1=float(-np.log(0.9)), log_loss_01=float(-np.log(0.1)), brier_09=0.01, brier_01=0.81)
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 1.2, 1.0], wspace=0.35)
    a = fig.add_subplot(gs[0])
    c = np.bincount(y)
    a.bar(["No spam (0)", "Spam (1)"], c, color=[BLUE, AMBER])
    for i, v in enumerate(c):
        a.text(i, v + 40, f"{v} ({v / len(y):.1%})", ha="center", fontsize=8.4)
    a.set_ylim(0, 2750)
    a.set_ylabel("correos de entrenamiento")
    titulo(a, "1. La referencia es la frecuencia")
    style(a, "y")
    b = fig.add_subplot(gs[1])
    nms = list(rk)
    x_ = np.arange(3)
    b.bar(x_ - 0.2, [rk[n]["log_loss"] for n in nms], 0.38, color=C38, label="log loss (elige)")
    b.bar(x_ + 0.2, [rk[n]["brier"] for n in nms], 0.38, color=VIOLET, label="Brier (describe)")
    for i, n in enumerate(nms):
        b.text(i - 0.2, rk[n]["log_loss"] + 0.012, f"{rk[n]['log_loss']:.3f}", ha="center", fontsize=8)
        b.text(i + 0.2, rk[n]["brier"] + 0.012, f"{rk[n]['brier']:.3f}", ha="center", fontsize=8)
    b.set_xticks(x_)
    b.set_xticklabels([NOM_SP[n] for n in nms], fontsize=8.2)
    b.set_ylim(0, 0.78)
    b.legend(fontsize=7.8, frameon=False)
    titulo(b, "2. Probabilidades OOF de 3 600+ correos")
    style(b, "y")
    cc = fig.add_subplot(gs[2])
    p = np.linspace(0.005, 0.995, 300)
    cc.plot(p, -np.log(p), color=C38, lw=2.4, label="log loss  −ln p")
    cc.plot(p, (1 - p) ** 2, color=VIOLET, lw=2.4, label="Brier  (1 − p)²")
    for pv in (0.9, 0.1):
        cc.scatter([pv], [-np.log(pv)], color=C38, zorder=4)
        cc.scatter([pv], [(1 - pv) ** 2], color=VIOLET, zorder=4)
    cc.text(0.12, 2.45, "p = 0.1 → 2.303", fontsize=8, color=C38)
    cc.text(0.5, 0.2, "p = 0.9 → 0.105 y 0.01", fontsize=8, color=INK)
    cc.set_ylim(0, 3.3)
    cc.set_xlabel("probabilidad predicha de spam (la clase real es spam)")
    cc.set_ylabel("pérdida")
    cc.legend(fontsize=7.8, frameon=False, loc="upper right")
    titulo(cc, "3. Penalizan distinto las probabilidades malas")
    style(cc)
    save(fig, "c38_spam.png")


def costos_fig():
    A = np.array([[890, 10], [40, 60]])
    B = np.array([[860, 40], [10, 90]])

    def mt(M):
        vn, fp, fn, vp = M.ravel()
        pr, rc = vp / (vp + fp), vp / (vp + fn)
        return dict(vn=int(vn), fp=int(fp), fn=int(fn), vp=int(vp), accuracy=(vn + vp) / M.sum(), precision=float(pr), recall=float(rc), f1=float(2 * pr * rc / (pr + rc)), costo=float((5 * fp + fn) / M.sum() * 1000))
    ma, mb = mt(A), mt(B)
    NUM["costos"] = dict(A=ma, B=mb, siempre_no=dict(accuracy=0.95, recall=0.0, spam=5, total=100),
                         prec_50=400 / 450, prec_5=40 / 135, vp_50=400, fp_50=50, vp_5=40, fp_5=95, umbral_teorico=5 / 6)
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 0.9, 0.9], wspace=0.45)
    for i, (M, mm, nm) in enumerate([(A, ma, "A"), (B, mb, "B")]):
        a = fig.add_subplot(gs[i])
        for r in range(2):
            for c in range(2):
                a.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=BLUE if r == c else AMBER, alpha=0.22))
                a.text(c, r, str(M[r, c]), ha="center", va="center", fontsize=15, color=INK)
        a.set_xticks([0, 1])
        a.set_xticklabels(["No spam", "Spam"], fontsize=8)
        a.set_yticks([0, 1])
        a.set_yticklabels(["No spam", "Spam"], fontsize=8)
        a.set_xlim(-0.5, 1.5)
        a.set_ylim(1.5, -0.5)
        a.set_xlabel("predicha")
        a.set_ylabel("real")
        a.set_aspect("equal")
        titulo(a, f"{nm}: accuracy 95 %, costo {mm['costo']:.0f}", 8.6)
    b = fig.add_subplot(gs[2])
    b.bar([0, 1], [ma["f1"], mb["f1"]], color=[C38, AMBER], width=0.5)
    for i, v in enumerate([ma["f1"], mb["f1"]]):
        b.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.6, fontweight="bold")
    b.set_xticks([0, 1])
    b.set_xticklabels(["A", "B"])
    b.set_ylim(0, 1.0)
    b.set_ylabel("F1 (mayor es mejor)")
    b2 = b.twinx()
    b2.plot([0, 1], [ma["costo"], mb["costo"]], "s--", color=RED, ms=7)
    for i, v in enumerate([ma["costo"], mb["costo"]]):
        b2.text(i + 0.06, v, f"costo {v:.0f}", color=RED, fontsize=8, va="center")
    b2.set_ylim(0, 260)
    b2.spines["right"].set_visible(True)
    b2.set_ylabel("costo por 1000 correos", color=RED)
    titulo(b, "F1 elige B; el costo elige A", 8.6)
    style(b, "y")
    c = fig.add_subplot(gs[3])
    c.bar([0, 1], [NUM["costos"]["prec_50"] * 100, NUM["costos"]["prec_5"] * 100], color=[C38, RED], width=0.5)
    for i, v in enumerate([NUM["costos"]["prec_50"] * 100, NUM["costos"]["prec_5"] * 100]):
        c.text(i, v + 2, f"{v:.1f} %", ha="center", fontsize=8.6, fontweight="bold")
    c.set_xticks([0, 1])
    c.set_xticklabels(["50 % spam", "5 % spam"])
    c.set_ylim(0, 105)
    c.set_ylabel("precision (misma sens. 80 %, FP 10 %)")
    titulo(c, "La prevalencia cambia la precision", 8.6)
    style(c, "y")
    save(fig, "c38_costos.png")


def roc_pr_fig():
    sp = spam_oof()
    y, oof = sp["y"], sp["oof"]
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    cols = {"FRECUENCIA": MUTED, "LOGISTICA_C1": BLUE, "BOSQUE_8": C38}
    a = fig.add_subplot(gs[0])
    for nm in cols:
        fp, tp, _ = roc_curve(y, oof[nm])
        a.plot(fp, tp, color=cols[nm], lw=2.2, label=f"{NOM_SP[nm]}  AUC {roc_auc_score(y, oof[nm]):.3f}")
    pts = {}
    for t, col in [(0.5, AMBER), (0.6, RED), (5 / 6, VIOLET)]:
        dd = decision(y, oof["BOSQUE_8"], t)
        pts[f"{t:.2f}"] = dict(fpr=dd["fp"] / (dd["fp"] + dd["vn"]), recall=dd["recall"], precision=dd["precision"])
        a.scatter([pts[f"{t:.2f}"]["fpr"]], [dd["recall"]], color=col, s=48, zorder=5, label=f"umbral {t:.2f}")
    a.plot([0, 1], [0, 1], "--", color="#9CA3AF")
    a.set_xlabel("tasa de falsos positivos")
    a.set_ylabel("recall (sensibilidad)")
    a.legend(fontsize=7.2, frameon=False, loc="lower right")
    titulo(a, "1. ROC: cada punto es un umbral")
    style(a)
    b = fig.add_subplot(gs[1])
    for nm in ("LOGISTICA_C1", "BOSQUE_8"):
        pr, rc, _ = precision_recall_curve(y, oof[nm])
        b.plot(rc, pr, color=cols[nm], lw=2.2, label=f"{NOM_SP[nm]}  AP {average_precision_score(y, oof[nm]):.3f}")
    b.axhline(y.mean(), color=MUTED, ls="--", lw=1.2, label=f"frecuencia de spam {y.mean():.3f}")
    for t, col in [(0.5, AMBER), (0.6, RED), (5 / 6, VIOLET)]:
        dd = decision(y, oof["BOSQUE_8"], t)
        b.scatter([dd["recall"]], [dd["precision"]], color=col, s=48, zorder=5)
    b.set_xlabel("recall")
    b.set_ylabel("precision")
    b.set_ylim(0.3, 1.02)
    b.legend(fontsize=7.4, frameon=False, loc="lower left")
    titulo(b, "2. Precision–recall (misma línea de umbrales)")
    style(b)
    c = fig.add_subplot(gs[2])
    bins = np.linspace(0, 1, 21)
    c.hist(oof["BOSQUE_8"][y == 0], bins=bins, color=BLUE, alpha=0.7, label="legítimos")
    c.hist(oof["BOSQUE_8"][y == 1], bins=bins, color=AMBER, alpha=0.7, label="spam")
    for t, col in [(0.5, AMBER), (0.6, RED), (5 / 6, VIOLET)]:
        c.axvline(t, color=col, ls="--", lw=1.4)
    c.set_xlabel("probabilidad OOF de spam (bosque)")
    c.set_ylabel("correos")
    c.legend(fontsize=7.8, frameon=False, loc="upper center")
    titulo(c, "3. El umbral corta dos distribuciones que se solapan")
    style(c, "y")
    save(fig, "c38_roc_pr.png")
    NUM["roc_pr"] = dict(puntos=pts, auc={n: float(roc_auc_score(y, oof[n])) for n in cols}, ap={n: float(average_precision_score(y, oof[n])) for n in cols})


def calibracion_fig():
    sp = spam_oof()
    y, oof = sp["y"], sp["oof"]

    def bines(p, k):
        b = pd.qcut(p, q=k, duplicates="drop")
        t = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b", observed=True).agg(p=("p", "mean"), f=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        return t
    q5 = bines(oof["BOSQUE_8"], 5)
    q10 = {nm: bines(oof[nm], 10) for nm in ("LOGISTICA_C1", "BOSQUE_8", "BOSQUE_CAL")}
    pr_ = {nm: probs(y, oof[nm]) for nm in ("BOSQUE_8", "BOSQUE_CAL", "LOGISTICA_C1")}
    NUM["calibracion"] = dict(quintil=q5.to_dict("list"), q4_prob=float(q5.p.iloc[3]), q4_frac=float(q5.f.iloc[3]), q4_n=int(q5.n.iloc[3]), metricas=pr_,
                              ece10={nm: float(np.sum(t.n * abs(t.p - t.f)) / t.n.sum()) for nm, t in q10.items()})
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.plot([0, 1], [0, 1], "--", color=MUTED, label="calibración ideal")
    a.plot(q5.p, q5.f, "o-", color=C38, lw=2.2, label="bosque OOF, 5 grupos")
    a.fill_between([0, 1], [0, 1], [1, 1], color=C38_BG, alpha=0.9)
    a.text(0.05, 0.9, "sobre la diagonal: subestima", fontsize=8, color=C38)
    a.text(0.55, 0.12, "bajo: sobreestima", fontsize=8, color=MUTED)
    a.annotate(f"4.º quintil: predice {q5.p.iloc[3] * 100:.1f} %\nobserva {q5.f.iloc[3] * 100:.1f} % de spam", xy=(q5.p.iloc[3], q5.f.iloc[3]), xytext=(0.43, 0.38), fontsize=8, arrowprops=dict(arrowstyle="->", color=INK))
    a.set_xlabel("probabilidad media predicha")
    a.set_ylabel("fracción observada de spam")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(a, "1. Buen ranking, probabilidades bajas")
    style(a)
    b = fig.add_subplot(gs[1])
    b.plot([0, 1], [0, 1], "--", color=MUTED)
    for nm, col, lb in [("LOGISTICA_C1", BLUE, "logística"), ("BOSQUE_8", C38, "bosque"), ("BOSQUE_CAL", RED, "bosque + isotónica")]:
        b.plot(q10[nm].p, q10[nm].f, "o-", color=col, lw=2, ms=4, label=lb)
    b.set_xlabel("probabilidad media predicha (10 grupos)")
    b.set_ylabel("fracción observada")
    b.legend(fontsize=7.8, frameon=False, loc="lower right")
    titulo(b, "2. Recalibrar acerca la curva a la diagonal")
    style(b)
    c = fig.add_subplot(gs[2])
    x_ = np.arange(2)
    c.bar(x_ - 0.2, [pr_["BOSQUE_8"]["log_loss"], pr_["BOSQUE_CAL"]["log_loss"]], 0.38, color=C38, label="log loss")
    c.bar(x_ + 0.2, [pr_["BOSQUE_8"]["brier"], pr_["BOSQUE_CAL"]["brier"]], 0.38, color=VIOLET, label="Brier")
    for i, nm in enumerate(("BOSQUE_8", "BOSQUE_CAL")):
        c.text(i - 0.2, pr_[nm]["log_loss"] + 0.006, f"{pr_[nm]['log_loss']:.3f}", ha="center", fontsize=8)
        c.text(i + 0.2, pr_[nm]["brier"] + 0.006, f"{pr_[nm]['brier']:.3f}", ha="center", fontsize=8)
    c.set_xticks(x_)
    c.set_xticklabels(["bosque", "bosque recalibrado\n(anidado)"], fontsize=8.4)
    c.set_ylim(0, 0.3)
    c.legend(fontsize=7.8, frameon=False)
    titulo(c, "3. Recalibrar es otra receta: se valida")
    style(c, "y")
    save(fig, "c38_calibracion.png")


def umbral_fig():
    sp = spam_oof()
    y, oof = sp["y"], sp["oof"]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    fino = np.round(np.arange(0.02, 0.99, 0.01), 2)

    def curva(p, g, r=5.0):
        return np.array([decision(y, p, t, r, 1.0)["costo"] * 1000 for t in g])
    c_raw, c_cal = curva(oof["BOSQUE_8"], grid), curva(oof["BOSQUE_CAL"], grid)
    imin = int(np.argmin(c_raw))
    tmin = float(grid[imin])
    ratios = [1, 2, 3, 5, 8, 10, 20]
    opt_raw, opt_cal = [], []
    for r in ratios:
        for p, lst in ((oof["BOSQUE_8"], opt_raw), (oof["BOSQUE_CAL"], opt_cal)):
            cc = curva(p, fino, r)
            m = cc.min()
            lst.append(float(fino[np.where(cc <= m + 1e-9)[0][-1]]))
    m05, m06 = decision(y, oof["BOSQUE_8"], 0.5), decision(y, oof["BOSQUE_8"], 0.6)
    d = datos()
    te, tr = d["ste"], d["str_"]
    fin = clone(modelos_spam()["BOSQUE_8"]).fit(sp["x"], y)
    pt = fin.predict_proba(te[d["vs"]].to_numpy())[:, 1]
    yt = te.spam.to_numpy()
    final = {**probs(yt, pt), **decision(yt, pt, tmin)}
    ref = decision(yt, np.zeros(len(yt)), 0.5)
    tcal = float(grid[int(np.argmin(c_cal))])
    NUM["umbral"] = dict(grid=grid.tolist(), costo_raw=c_raw.tolist(), costo_cal=c_cal.tolist(), t_min=tmin, costo_min=float(c_raw[imin]), t_min_cal=tcal, costo_min_cal=float(c_cal.min()),
                         c050=float(c_raw[list(grid).index(0.5)]), c055=float(c_raw[list(grid).index(0.55)]), c060=float(c_raw[list(grid).index(0.6)]), c065=float(c_raw[list(grid).index(0.65)]),
                         teorico=5 / 6, ratios=ratios, opt_raw=opt_raw, opt_cal=opt_cal, opt_teo=[r / (1 + r) for r in ratios],
                         m05=m05, m06=m06, ahorro_fp=m05["fp"] - m06["fp"], mas_fn=m06["fn"] - m05["fn"], ahorro_unid=5 * (m05["fp"] - m06["fp"]) - (m06["fn"] - m05["fn"]),
                         prueba=final, referencia=ref, n_prueba=int(len(yt)), costo_cal_en_083=float(curva(oof["BOSQUE_CAL"], np.array([0.85]))[0]),
                         costo_raw_en_083=float(curva(oof["BOSQUE_8"], np.array([0.85]))[0]))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9], wspace=0.32)
    a = fig.add_subplot(gs[0])
    a.plot(grid, c_raw, "o-", color=C38, lw=2.2, ms=4, label="bosque")
    a.plot(grid, c_cal, "o-", color=RED, lw=2, ms=4, label="bosque recalibrado")
    a.axvline(0.5, color=MUTED, ls="--", lw=1, label="0.50")
    a.axvline(5 / 6, color=VIOLET, ls=":", lw=1.6, label="teórico 5/6 = 0.833")
    a.scatter([tmin], [c_raw[imin]], color=C38, s=70, zorder=5)
    a.annotate(f"mínimo OOF {tmin:.2f}: {c_raw[imin]:.1f}", xy=(tmin, c_raw[imin]), xytext=(0.52, 330), fontsize=8, color=C38, arrowprops=dict(arrowstyle="->", color=C38))
    a.set_ylim(0, 600)
    a.set_xlabel("umbral para bloquear")
    a.set_ylabel("costo OOF por 1000 correos (FP = 5, FN = 1)")
    a.legend(fontsize=7.4, frameon=False, loc="upper right")
    titulo(a, "1. Costo frente al umbral")
    style(a)
    b = fig.add_subplot(gs[1])
    b.plot(ratios, NUM["umbral"]["opt_teo"], "s--", color=VIOLET, lw=1.6, label="teórico  r/(1 + r)")
    b.plot(ratios, opt_raw, "o-", color=C38, lw=2, label="empírico, bosque")
    b.plot(ratios, opt_cal, "o-", color=RED, lw=2, label="empírico, recalibrado")
    b.set_xscale("log")
    b.set_xticks(ratios)
    b.set_xticklabels([str(r) for r in ratios])
    b.set_xlabel("razón de costos r = costo FP / costo FN")
    b.set_ylabel("umbral que minimiza el costo OOF")
    b.set_ylim(0, 1.0)
    b.legend(fontsize=7.4, frameon=False, loc="lower right")
    titulo(b, "2. Más caro el FP, más alto el umbral")
    style(b)
    c = fig.add_subplot(gs[2])
    filas = [["", "FP", "FN", "costo"], ["t = 0.50", m05["fp"], m05["fn"], f"{m05['costo'] * 1000:.1f}"], ["t = 0.60", m06["fp"], m06["fn"], f"{m06['costo'] * 1000:.1f}"]]
    c.axis("off")
    tb = c.table(cellText=filas, loc="center", cellLoc="center", colWidths=[0.34, 0.2, 0.2, 0.26])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.8)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r_ == 0:
            cell.set_facecolor(C38_BG)
    c.text(0.5, 0.18, f"−{NUM['umbral']['ahorro_fp']} FP, +{NUM['umbral']['mas_fn']} FN\nahorro 5·{NUM['umbral']['ahorro_fp']} − {NUM['umbral']['mas_fn']} = {NUM['umbral']['ahorro_unid']} unidades", ha="center", fontsize=8.2, color=INK, transform=c.transAxes)
    titulo(c, "3. Menos aciertos, menos costo")
    save(fig, "c38_umbral.png")


# ================================================================== AIC / BIC
def aic_bic_fig():
    import statsmodels.api as sm
    d = datos()
    htr = d["htr"]
    Xs = StandardScaler().fit_transform(htr[V8])
    y, fold = htr.resistencia.to_numpy(), htr.fold.to_numpy()
    n = len(y)
    pf = PolynomialFeatures(2, include_bias=False).fit(Xs)
    Z = pf.transform(Xs)
    nm = pf.get_feature_names_out([f"x{i}" for i in range(8)])
    corr = np.array([abs(np.corrcoef(Z[:, j], y)[0, 1]) for j in range(Z.shape[1])])
    lin = [j for j in np.argsort(-corr) if "^" not in nm[j] and " " not in nm[j]]
    cua = [j for j in np.argsort(-corr) if "^2" in nm[j]]
    inter = [j for j in np.argsort(-corr) if " " in nm[j]]
    orden = lin + cua + inter
    ps = list(range(0, 45))
    aic, bic, cvr, sse_l = [], [], [], []
    for p in ps:
        cols = orden[:p]
        X = sm.add_constant(Z[:, cols], has_constant="add") if p else np.ones((n, 1))
        res = sm.OLS(y, X).fit()
        aic.append(float(res.aic))
        bic.append(float(res.bic))
        sse_l.append(float(res.ssr))
        oof = np.zeros(n)
        for k in range(1, 6):
            v = fold == k
            b = np.linalg.lstsq(X[~v], y[~v], rcond=None)[0]
            oof[v] = X[v] @ b
        cvr.append(rmse(y, oof))
    aic, bic, cvr = np.array(aic), np.array(bic), np.array(cvr)
    k = ps[-1] + 1
    sse = sse_l[-1]
    aic_man = n * np.log(sse / n) + n * (1 + np.log(2 * np.pi)) + 2 * k
    NUM["aic_bic"] = dict(n=n, p_aic=int(np.argmin(aic)), p_bic=int(np.argmin(bic)), p_cv=int(np.argmin(cvr)), aic_min=float(aic.min()), bic_min=float(bic.min()), cv_min=float(cvr.min()),
                          aic_nulo=float(aic[0]), bic_nulo=float(bic[0]), p_lineal=8, aic_lineal=float(aic[8]), bic_lineal=float(bic[8]), cv_lineal=float(cvr[8]), aic_cuad=float(aic[16]), bic_cuad=float(bic[16]), cv_cuad=float(cvr[16]),
                          aic_completo=float(aic[44]), bic_completo=float(bic[44]), cv_completo=float(cvr[44]), aic_manual=float(aic_man), ln_n=float(np.log(n)), sse_completo=float(sse), orden=[str(nm[j]) for j in orden[:8]],
                          penal_aic=2.0, penal_bic=float(np.log(n)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.plot(ps, aic - aic.min(), color=C38, lw=2.2, label="AIC − mínimo")
    a.plot(ps, bic - bic.min(), color=VIOLET, lw=2.2, label="BIC − mínimo")
    a.scatter([np.argmin(aic)], [0], color=C38, s=55, zorder=5)
    a.scatter([np.argmin(bic)], [0], color=VIOLET, s=55, zorder=5)
    a.set_ylim(-30, 600)
    a.set_xlabel("términos del modelo (sin contar el intercepto)")
    a.set_ylabel("Δ criterio (menor es mejor)")
    a.text(np.argmin(aic) + 0.8, 130, f"AIC elige {np.argmin(aic)}", fontsize=8, color=C38)
    a.text(np.argmin(bic) + 0.8, 70, f"BIC elige {np.argmin(bic)}", fontsize=8, color=VIOLET)
    a.legend(fontsize=7.8, frameon=False, loc="upper right")
    titulo(a, "1. AIC y BIC con todas las filas de ajuste")
    style(a)
    b = fig.add_subplot(gs[1])
    b.plot(ps, cvr, color=BLUE, lw=2.2)
    b.scatter([np.argmin(cvr)], [cvr.min()], color=RED, s=55, zorder=5)
    b.text(np.argmin(cvr) + 0.8, cvr.min() + 0.9, f"CV elige {np.argmin(cvr)}: {cvr.min():.2f}", fontsize=8, color=RED)
    b.set_ylim(7.5, 12.5)
    b.set_xlabel("términos del modelo (sin contar el intercepto)")
    b.set_ylabel("RMSE OOF (MPa), mínimos cuadrados")
    titulo(b, "2. La validación cruzada mira datos nuevos")
    style(b)
    c = fig.add_subplot(gs[2])
    kk = np.arange(0, 45)
    c.plot(kk, 2 * kk, color=C38, lw=2.2, label="AIC: 2k")
    c.plot(kk, np.log(n) * kk, color=VIOLET, lw=2.2, label=f"BIC: k·ln(n) = {np.log(n):.2f}k")
    c.set_xlabel("parámetros k")
    c.set_ylabel("penalización")
    c.legend(fontsize=7.8, frameon=False, loc="upper left")
    c.text(0.3, 0.55, "con n = 806, cada parámetro\ncuesta 6.69 en BIC y 2 en AIC", transform=c.transAxes, fontsize=8, color=INK)
    titulo(c, "3. El BIC penaliza más cuando n crece")
    style(c)
    save(fig, "c38_aic_bic.png")


# ================================================================== validación anidada
def anidada_fig():
    rng = np.random.default_rng(RS)
    reps, n, M = 300, 60, 100

    def nc_acc(Xt, yt, Xv, yv):
        m0, m1 = Xt[yt == 0].mean(0), Xt[yt == 1].mean(0)
        thr, sg = (m0 + m1) / 2, np.where(m1 > m0, 1, -1)
        return (((Xv - thr) * sg > 0).astype(int) == yv[:, None]).mean(0)

    def cv_feat(X, yv, K, rg):
        idx = rg.permutation(len(yv)) % K
        acc = np.zeros(X.shape[1])
        for k in range(K):
            v = idx == k
            acc += nc_acc(X[~v], yv[~v], X[v], yv[v]) * v.sum()
        return acc / len(yv)
    mejor, anid, nuevo = [], [], []
    for _ in range(reps):
        X = rng.normal(size=(n, M))
        yv = (np.arange(n) % 2).astype(int)
        rng.shuffle(yv)
        a = cv_feat(X, yv, 5, rng)
        j = int(np.argmax(a))
        mejor.append(float(a[j]))
        Xn = rng.normal(size=(2000, M))
        yn = rng.integers(0, 2, 2000)
        nuevo.append(float(nc_acc(X, yv, Xn, yn)[j]))
        idx = rng.permutation(n) % 5
        tot = 0.0
        for k in range(5):
            v = idx == k
            ai = cv_feat(X[~v], yv[~v], 4, rng)
            tot += nc_acc(X[~v], yv[~v], X[v], yv[v])[int(np.argmax(ai))] * v.sum()
        anid.append(tot / n)
    # hormigón: rejilla grado × α con validación anidada (afuera: los 5 pliegues de la clase; adentro: 5 pliegues por composición)
    d = datos()
    htr = d["htr"]
    x, y, g, fold = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.grupo.to_numpy(), htr.fold.to_numpy()
    rej = [(gr, al) for gr in (1, 2, 3) for al in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)]
    ext_mejor, ext_ext, elegidos = [], [], []
    for k in range(1, 6):
        v = fold == k
        xt, yt, gt = x[~v], y[~v], g[~v]
        fi = folds_por_grupo(gt, 5, np.random.default_rng(k))
        sc = [rmse(yt, oof_reg(ridge_cuad(al, gr), xt, yt, fi)) for gr, al in rej]
        j = int(np.argmin(sc))
        ft = ridge_cuad(rej[j][1], rej[j][0]).fit(xt, yt)
        ext_mejor.append(float(sc[j]))
        ext_ext.append(rmse(y[v], ft.predict(x[v])))
        elegidos.append(list(rej[j]))
    NUM["anidada"] = dict(reps=reps, n=n, candidatos=M, cv_mejor=float(np.mean(mejor)), anidada=float(np.mean(anid)), nuevo=float(np.mean(nuevo)), cv_mejor_sd=float(np.std(mejor, ddof=1)),
                          anidada_sd=float(np.std(anid, ddof=1)), rejilla=len(rej), hor_interna=ext_mejor, hor_externa=ext_ext, hor_elegidos=elegidos, hor_interna_media=float(np.mean(ext_mejor)),
                          hor_externa_media=rmse(np.zeros(1), np.array([0.0])) + float(np.sqrt(np.average(np.array(ext_ext) ** 2, weights=[int((fold == k).sum()) for k in range(1, 6)]))),
                          ajustes_anidados=int(5 * 5 * len(rej) + 5), ajustes_simple=int(5 * len(rej)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 10)
    a.set_ylim(0, 6.3)
    a.axis("off")
    a.text(0.0, 5.9, "Bucle externo: estima el desempeño", fontsize=8.6, color=INK, fontweight="bold")
    for k in range(5):
        a.add_patch(Rectangle((k * 2.0, 4.9), 1.9, 0.6, fc=AMBER if k == 2 else BLUE, ec="white", alpha=0.9 if k == 2 else 0.55))
    a.text(5, 4.6, "pliegue externo 3 = validación externa (no elige nada)", ha="center", fontsize=7.8, color=AMBER)
    a.text(0.0, 3.9, "Bucle interno (solo con los 4 pliegues azules): elige", fontsize=8.6, color=INK, fontweight="bold")
    for it in range(4):
        for k in range(4):
            a.add_patch(Rectangle((k * 2.0, 3.25 - it * 0.55), 1.9, 0.45, fc=C38 if k == it else BLUE, ec="white", alpha=0.9 if k == it else 0.4))
    a.text(8.15, 2.6, "elige α, grado,\nmodelo…", fontsize=8, color=C38, va="center")
    a.text(0.0, 0.55, "se reajusta lo elegido con los 4 pliegues azules y se\nmide en el externo; se promedian los 5 externos", fontsize=8, color=MUTED)
    titulo(a, "1. Dos bucles: elegir adentro, medir afuera")
    b = fig.add_subplot(gs[1])
    b.hist(mejor, bins=18, color=RED, alpha=0.7, label=f"mejor puntaje de CV (media {np.mean(mejor):.3f})")
    b.hist(anid, bins=18, color=C38, alpha=0.7, label=f"anidada (media {np.mean(anid):.3f})")
    b.axvline(0.5, color=INK, ls="--", lw=1.4)
    b.set_ylim(0, 112)
    b.set_xlabel(f"exactitud · etiquetas al azar, n = {n}, {M} candidatos")
    b.set_ylabel(f"simulaciones (de {reps})")
    b.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(b, "2. Sin señal alguna, elegir infla el puntaje")
    style(b, "y")
    c = fig.add_subplot(gs[2])
    xs = np.arange(1, 6)
    c.bar(xs - 0.2, ext_mejor, 0.38, color=VIOLET, alpha=0.85, label="mejor RMSE interno (elige)")
    c.bar(xs + 0.2, ext_ext, 0.38, color=C38, alpha=0.9, label="RMSE externo (mide)")
    c.set_ylim(5, 13.5)
    c.set_xlabel("pliegue externo")
    c.set_ylabel("RMSE (MPa)")
    c.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(c, f"3. Hormigón: {len(rej)} configuraciones, 5 + 5 pliegues")
    style(c, "y")
    save(fig, "c38_anidada.png")


# ================================================================== resumen de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    def caja(x, y, w, h, color, titulo_, lineas, fs=7.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.2, titulo_, ha="center", va="center", fontsize=9.2, color=color, fontweight="bold")
        for i, t in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.52 - i * 0.28, t, ha="center", va="center", fontsize=fs, color=INK)
    caja(0.1, 2.75, 2.35, 1.85, BLUE, "1 · Definir", ["qué se predice y para qué", "métrica y referencia", "fijadas ANTES", "MAE · RMSE · R² · log loss"])
    caja(2.7, 2.75, 2.35, 1.85, TEAL, "2 · Validar sin fuga", ["partición por grupos", "pliegues y OOF", "todo dentro de Pipeline", "K-fold · LOOCV · bootstrap"])
    caja(5.3, 2.75, 2.35, 1.85, AMBER, "3 · Comparar", ["sesgo · varianza · ruido", "curvas de aprendizaje", "AIC · BIC", "candidatos = recetas"])
    caja(7.9, 2.75, 2.2, 1.85, VIOLET, "4 · Elegir", ["regla fijada antes", "validación anidada", "reajustar con todo", "el entrenamiento"])
    caja(10.35, 2.75, 2.05, 1.85, C38, "5 · Evaluar", ["prueba, una vez", "ROC · PR · calibración", "umbral por costo", "límites y referencia"])
    for xx in (2.5, 5.1, 7.7, 10.15):
        ax.annotate("", xy=(xx + 0.22, 3.65), xytext=(xx, 3.65), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 2.35, boxstyle="round,pad=0.03", fc=C38_BG, ec=C38, lw=1.4))
    h, cv, sp, um = NUM["hormigon"], NUM["cv"], NUM["spam"], NUM["umbral"]
    ax.text(6.25, 2.15, "Datos de la clase: hormigon_s38.csv (1 030 mediciones, 8 predictores, resistencia en MPa) · spam_s38.csv (4 601 correos, 57 predictores)", ha="center", va="center", fontsize=8.7, color=C38, fontweight="bold")
    ax.text(6.25, 1.7, f"Hormigón: gana cuadrática α = 0.1 con RMSE OOF {cv['rmse_oof_01']:.3f} (α = 1: {cv['rmse_oof_1']:.3f}) · prueba: RMSE {h['prueba']['rmse']:.3f}, MAE {h['prueba']['mae']:.3f}, R² {h['prueba']['r2']:.3f} (media: {h['referencia']['rmse']:.3f})", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(6.25, 1.32, f"Spam: gana el bosque (log loss {sp['ranking']['BOSQUE_8']['log_loss']:.3f}, AUC {sp['ranking']['BOSQUE_8']['auc']:.3f}) · umbral {um['t_min']:.2f} (costo OOF {um['costo_min']:.1f}) · prueba: aciertos {um['prueba']['accuracy'] * 100:.1f} %, costo {um['prueba']['costo'] * 1000:.1f} por 1 000", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(6.25, 0.85, "R ↔ Python:  ranger ↔ RandomForestClassifier  ·  glm ↔ LogisticRegression  ·  lm + poly + ridge ↔ Pipeline(StandardScaler, PolynomialFeatures, Ridge)  ·  fold ↔ GroupKFold", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    ax.text(6.25, 0.45, "R y Python coinciden en las cifras de regresión; el bosque difiere un poco entre ranger y scikit-learn (implementación, no método)", ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")
    save(fig, "c38_esquema.png")


PASOS = dict(metricas=metricas_fig, oof=oof_fig, repetido=repetido_fig, bootstrap=bootstrap_fig, sesgo=sesgo_varianza_fig, curvas=curvas_fig, hormigon=hormigon_fig,
             diagnostico=diagnostico_fig, spam=spam_fig, costos=costos_fig, roc=roc_pr_fig, calibracion=calibracion_fig, umbral=umbral_fig, aic=aic_bic_fig, anidada=anidada_fig,
             esquema=esquema_fig)

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c38_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or list(PASOS)
    # el esquema y el resumen usan números de otras figuras: se generan antes
    if "esquema" in elegidos:
        for k in ("oof", "hormigon", "spam", "umbral"):
            if k not in elegidos and k not in NUM and k not in ("oof",):
                elegidos.append(k)
    orden = [e for e in PASOS if e in elegidos and e != "esquema"] + (["esquema"] if "esquema" in elegidos else [])
    for k in orden:
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c38_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
