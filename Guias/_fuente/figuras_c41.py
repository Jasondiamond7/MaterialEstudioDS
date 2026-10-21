"""Figuras de la Clase 41 (Detección de anomalías).
Fuentes: diapositivas deteccion_anomalias.pdf, anomalias.py / anomalias.R del profesor Pedro Luiz Ramos y los tres CSV de la clase
(puntos.csv: 206 puntos con 2 variables; canciones.csv: 197 canciones con 7 variables; cardio.csv: 1 831 registros con 21 variables y etiqueta).
La carpeta de la clase se lee sin modificarla. Todo sale de código; lo aleatorio usa semilla fija. Escribe c41_numeros.json
(los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c41.py   (o con nombres: python -X utf8 figuras_c41.py tipos ricz maha)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C41 = "#CA8A04"      # color de la clase 41 (dorado)
C41_BG = "#FEFCE8"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
OK = "#4B79B5"        # puntos normales
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA41 = next(p for p in BASE.glob("Clase 41*") if p.is_dir())
_DAT = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


# ----------------------------------------------------------------------------- datos y funciones comunes
def datos():
    """Lee los tres CSV de la clase (una sola vez), con las mismas transformaciones del script del profesor."""
    if _DAT:
        return _DAT
    pts = pd.read_csv(CARPETA41 / "puntos.csv")
    x = pts[["x1", "x2"]]
    z = ((x - x.mean()) / x.std(ddof=1)).to_numpy()
    can = pd.read_csv(CARPETA41 / "canciones.csv")
    xc = can.drop(columns="id")
    zc = ((xc - xc.mean()) / xc.std(ddof=1)).to_numpy()
    car = pd.read_csv(CARPETA41 / "cardio.csv")
    V = [f"v{j}" for j in range(1, 22)]
    tr, te = car[car.conjunto == "entrenamiento"], car[car.conjunto == "prueba"]
    mu, sd = tr[V].mean(), tr[V].std(ddof=1)
    _DAT.update(pts=pts, z=z, can=can, zc=zc, car=car, V=V, ztr=((tr[V] - mu) / sd).to_numpy(), zte=((te[V] - mu) / sd).to_numpy(),
                ytr=tr.anomalia.to_numpy(), yte=te.anomalia.to_numpy())
    return _DAT


def mahal2(a):
    """Distancia de Mahalanobis al cuadrado respecto de la media y la covarianza muestrales de a."""
    c = a - a.mean(axis=0)
    return np.sum((c @ np.linalg.inv(np.cov(a, rowvar=False))) * c, axis=1)


def elipse(ax, centro, cov, nivel, **kw):
    """Elipse de contorno chi-cuadrado (dos variables) para una media y covarianza dadas."""
    val, vec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    r = np.sqrt(chi2.ppf(nivel, 2))
    ax.add_patch(Ellipse(centro, 2 * r * np.sqrt(val[1]), 2 * r * np.sqrt(val[0]), angle=ang, fill=False, **kw))


def alertas_puntos():
    """Las cuatro alertas del script del profesor sobre puntos.csv (Mahalanobis 99 %, kNN k = 10 cuantil 95, LOF > 1.5, DBSCAN)."""
    d = datos()
    z = d["z"]
    dm = mahal2(z)
    a_m = dm > chi2.ppf(0.99, 2)
    sc_knn = NearestNeighbors(n_neighbors=10).fit(z).kneighbors()[0][:, -1]
    u_knn = float(np.quantile(sc_knn, 0.95))
    lof = -LocalOutlierFactor(n_neighbors=10).fit(z).negative_outlier_factor_
    db = DBSCAN(eps=0.45, min_samples=6).fit(z)
    return dict(dm=dm, a_m=a_m, sc_knn=sc_knn, u_knn=u_knn, a_knn=sc_knn > u_knn, lof=lof, a_lof=lof > 1.5, db=db, a_db=db.labels_ == -1)


def c_psi(n):
    """Longitud media de un camino fallido en un árbol binario de búsqueda (normalización del Isolation Forest)."""
    return 0.0 if n <= 1 else (1.0 if n == 2 else 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n)


def isoforest(Xtr, seed=RS, ms=256, ne=300, **kw):
    return IsolationForest(n_estimators=ne, max_samples=ms, random_state=seed, n_jobs=1, **kw).fit(Xtr)


def cardio_modelos():
    """Ajusta (solo con entrenamiento) el PCA con Q de la clase y el Isolation Forest; devuelve puntajes en entrenamiento y prueba."""
    d = datos()
    ztr, zte = d["ztr"], d["zte"]
    pca = PCA().fit(ztr)
    r = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1)
    P = pca.components_[:r]
    q = lambda Z: np.sum((Z - Z @ P.T @ P) ** 2, axis=1)
    m = isoforest(ztr)
    return dict(pca=pca, r=r, P=P, q_tr=q(ztr), q_te=q(zte), t2_te=np.sum((zte @ P.T) ** 2 / pca.explained_variance_[:r], axis=1),
                m_if=m, s_tr=-m.score_samples(ztr), s_te=-m.score_samples(zte))


# ----------------------------------------------------------------------------- 1. tipos de anomalías
def tipos_fig():
    d = datos()
    rng = np.random.default_rng(RS)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.6))
    # global: los ocho valores de la clase
    ax = axs[0]
    v = np.array([10, 11, 12, 12, 13, 14, 15, 30.0])
    ax.scatter(np.arange(1, 9), v, s=46, c=[OK] * 7 + [RED], zorder=3)
    ax.axhspan(8, 18, color=C41, alpha=0.12)
    ax.text(0.7, 8.6, "zona normal según la regla RIC", fontsize=7.8, color=INK)
    ax.set(xlabel="observación", ylabel="valor")
    titulo(ax, "Global: muy alejado de todos", 9)
    style(ax)
    # contextual: consumo por hora
    ax = axs[1]
    h = np.arange(24)
    base = 55 + 35 * np.exp(-((h - 14) ** 2) / 30) + 20 * np.exp(-((h - 20) ** 2) / 8) - 28 * np.exp(-((h - 3) ** 2) / 12)
    obs = base + rng.normal(0, 2.2, 24)
    obs[3] = 78
    ax.plot(h, base, color=MUTED, lw=1.2, ls="--", label="patrón habitual")
    ax.scatter(h, obs, s=26, c=[RED if i == 3 else OK for i in range(24)], zorder=3)
    ax.axhline(78, color=RED, lw=0.7, ls=":")
    ax.text(5.0, 79.5, "78: normal a las 14 h, anómalo a las 3 h", fontsize=7.8, color=RED)
    ax.set(xlabel="hora del día", ylabel="consumo (ilustración)")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    titulo(ax, "Contextual: depende del contexto", 9)
    style(ax)
    # conjunta / multivariada: puntos.csv
    ax = axs[2]
    z = d["z"]
    idx = [200, 201, 202]
    ax.scatter(z[:200, 0], z[:200, 1], s=14, c=OK, alpha=0.7)
    ax.scatter(z[[201, 202], 0], z[[201, 202], 1], s=46, c=RED, zorder=3)
    for i in (201, 202):
        ax.annotate(f"id {i + 1}", z[i], xytext=(5, 5), textcoords="offset points", fontsize=7.8)
    ax.axhspan(-2, 2, color=C41, alpha=0.10)
    ax.axvspan(-2, 2, color=C41, alpha=0.10)
    ax.set(xlabel="x1 estandarizada", ylabel="x2 estandarizada", xlim=(-4, 5), ylim=(-4, 5))
    titulo(ax, "Conjunta: valores comunes, combinación rara", 9)
    style(ax)
    NUM["tipos"] = dict(z_202=[float(z[201, 0]), float(z[201, 1])], z_203=[float(z[202, 0]), float(z[202, 1])],
                        z_max_marginal=float(np.abs(z[[201, 202]]).max()))
    fig.tight_layout()
    save(fig, "c41_tipos.png")


# ----------------------------------------------------------------------------- 2. RIC y puntaje z
def ricz_fig():
    v = np.array([10, 11, 12, 12, 13, 14, 15, 30.0])
    q1, q3 = np.quantile(v, [0.25, 0.75])
    ric = q3 - q1
    lo, hi = q1 - 1.5 * ric, q3 + 1.5 * ric
    zs = (v - v.mean()) / v.std(ddof=1)
    zs_pob = (v - v.mean()) / v.std(ddof=0)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    zmod = 0.6745 * (v - med) / mad
    nn = np.arange(3, 31)
    cota = (nn - 1) / np.sqrt(nn)
    n_min = int(nn[np.argmax(cota > 3)])
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7))
    ax = axs[0]
    ax.boxplot(v, orientation="horizontal", widths=0.5, patch_artist=True, boxprops=dict(facecolor=C41_BG, edgecolor=C41),
               medianprops=dict(color=INK), flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED, markersize=7))
    for xx, t in [(lo, f"L = {lo:g}"), (hi, f"U = {hi:g}")]:
        ax.axvline(xx, color=RED, ls="--", lw=1)
        ax.text(xx, 1.33, t, ha="center", fontsize=8.2, color=RED)
    ax.text(q1 - 0.3, 0.62, f"Q1 = {q1:.2f}", ha="right", fontsize=7.8)
    ax.text(q3 + 0.3, 0.62, f"Q3 = {q3:.2f}", ha="left", fontsize=7.8)
    ax.set(yticks=[], xlabel="valor", ylim=(0.5, 1.5), xlim=(5, 33))
    titulo(ax, "Regla del RIC: 30 queda fuera de [8; 18]", 9)
    style(ax, "x")
    ax = axs[1]
    ax.bar(np.arange(8), zs, color=[OK] * 7 + [RED], width=0.6)
    ax.axhline(3, color=RED, ls="--", lw=1)
    ax.axhline(-3, color=RED, ls="--", lw=1)
    ax.text(0.0, 3.12, "corte |z| = 3", fontsize=8, color=RED)
    ax.text(7, zs[-1] + 0.12, f"{zs[-1]:.3f}", ha="center", fontsize=8.4, fontweight="bold")
    ax.set(xticks=range(8), xticklabels=[f"{x:g}" for x in v], xlabel="valor", ylabel="puntaje z", ylim=(-3.6, 3.6))
    titulo(ax, "Puntaje z: el 30 no llega a 3", 9)
    style(ax, "y")
    ax = axs[2]
    ax.plot(nn, cota, color=C41, lw=2, label="|z| máximo posible, (n − 1)/√n")
    ax.axhline(3, color=RED, ls="--", lw=1)
    ax.scatter([8], [(8 - 1) / np.sqrt(8)], color=RED, s=48, zorder=3)
    ax.annotate("n = 8: máximo 2.47", (8, 2.47), xytext=(10, 1.5), fontsize=8, arrowprops=dict(arrowstyle="->", color=INK))
    ax.axvline(n_min, color=MUTED, ls=":", lw=1)
    ax.text(n_min + 0.4, 0.6, f"hace falta n ≥ {n_min}", fontsize=8, color=MUTED)
    ax.set(xlabel="tamaño de la muestra n", ylabel="|z| máximo", ylim=(0, 5.3))
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(ax, "Con pocos datos, |z| > 3 es imposible", 9)
    style(ax)
    NUM["ricz"] = dict(q1=float(q1), q3=float(q3), ric=float(ric), L=float(lo), U=float(hi), media=float(v.mean()), sd=float(v.std(ddof=1)),
                       sd_pob=float(v.std(ddof=0)), z30=float(zs[-1]), z30_pob=float(zs_pob[-1]), zmax_n8=float((8 - 1) / np.sqrt(8)),
                       zmax_n8_pob=float(np.sqrt(8 - 1)), n_min=n_min, mediana=float(med), mad=float(mad), zmod30=float(zmod[-1]),
                       media_sin30=float(v[:-1].mean()), sd_sin30=float(v[:-1].std(ddof=1)),
                       z_otros_max=float(np.abs(zs[:-1]).max()), p_2s=float(2 * (1 - __import__("scipy").stats.norm.cdf(2))),
                       p_3s=float(2 * (1 - __import__("scipy").stats.norm.cdf(3))))
    fig.tight_layout()
    save(fig, "c41_ricz.png")


# ----------------------------------------------------------------------------- 3. Mahalanobis
def maha_fig():
    d = datos()
    z, pts = d["z"], d["pts"]
    dm = mahal2(z)
    u99, u95 = chi2.ppf(0.99, 2), chi2.ppf(0.95, 2)
    mcd = MinCovDet(random_state=0).fit(z)
    dr = mcd.mahalanobis(z)
    raw = pts[["x1", "x2"]].to_numpy()
    inv_dif = float(np.abs(dm - mahal2(raw)).max())
    S_h = np.array([[1, 0.9], [0.9, 1]])
    d2_a = float(np.array([1, 1]) @ np.linalg.inv(S_h) @ np.array([1, 1]))
    d2_b = float(np.array([1, -1]) @ np.linalg.inv(S_h) @ np.array([1, -1]))
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    rng = np.random.default_rng(RS)
    sim = rng.multivariate_normal([0, 0], S_h, 300)
    ax.scatter(sim[:, 0], sim[:, 1], s=7, c=OK, alpha=0.45)
    for r in (np.sqrt(2),):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=MUTED, ls="--", lw=1.2))
    elipse(ax, (0, 0), S_h, 0.95, ec=C41, lw=1.8)
    ax.scatter([1], [1], c=BLUE, s=60, zorder=3)
    ax.scatter([1], [-1], c=RED, s=60, zorder=3)
    ax.annotate(f"(1, 1): D² = {d2_a:.2f}", (1, 1), xytext=(-3.6, 2.9), fontsize=8.2, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate(f"(1, −1): D² = {d2_b:.0f}", (1, -1), xytext=(0.8, -3.2), fontsize=8.2, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.text(-3.7, -3.8, "misma distancia euclidiana (círculo punteado),\nelipse = 95 % con correlación 0.9", fontsize=7.4, color=MUTED)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="x1", ylabel="x2")
    ax.set_aspect("equal")
    titulo(ax, "Euclidiana frente a Mahalanobis", 9)
    style(ax)
    ax = axs[1]
    a99 = dm > u99
    ax.scatter(z[~a99, 0], z[~a99, 1], s=14, c=OK, alpha=0.75)
    ax.scatter(z[a99, 0], z[a99, 1], s=40, c=RED, zorder=3)
    for q, ls in [(0.95, "--"), (0.99, "-")]:
        elipse(ax, z.mean(0), np.cov(z, rowvar=False), q, ec=C41, lw=1.3, ls=ls)
    for i in np.where(a99)[0]:
        ax.annotate(str(int(pts.id[i])), z[i], xytext=(4, 4), textcoords="offset points", fontsize=7.6)
    ax.set(xlabel="x1 estandarizada", ylabel="x2 estandarizada")
    titulo(ax, f"puntos.csv: {int(a99.sum())} alertas con el corte del 99 %", 9)
    style(ax)
    ax = axs[2]
    ids = np.arange(201, 207)
    w = 0.38
    ax.bar(np.arange(6) - w / 2, dm[200:], w, color=C41, label="clásica (media y covarianza de los 206)")
    ax.bar(np.arange(6) + w / 2, dr[200:], w, color=VIOLET, label="robusta (MCD)")
    ax.axhline(u99, color=RED, ls="--", lw=1)
    ax.text(5.45, u99 * 0.8, f"χ² 99 % = {u99:.2f}", ha="right", fontsize=8, color=RED, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_yscale("log")
    ax.set_ylim(2, 1500)
    ax.set(xticks=range(6), xticklabels=[str(i) for i in ids], xlabel="caso adicional (id)", ylabel="D² (escala log)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(ax, "Los extremos inflan la covarianza", 9)
    style(ax, "y")
    nflag = {f"{q:g}": int((dm > chi2.ppf(q, 2)).sum()) for q in (0.9, 0.95, 0.99, 0.999)}
    ids_flag = {f"{q:g}": [int(i) for i in pts.id[dm > chi2.ppf(q, 2)]] for q in (0.9, 0.95, 0.99, 0.999)}
    NUM["maha"] = dict(u90=float(chi2.ppf(0.9, 2)), u95=float(u95), u99=float(u99), u999=float(chi2.ppf(0.999, 2)), n_alertas=nflag, ids=ids_flag,
                       d2_hip_a=d2_a, d2_hip_b=d2_b, inv_dif=inv_dif, corr_todos=float(np.corrcoef(z.T)[0, 1]),
                       corr_200=float(np.corrcoef(z[:200].T)[0, 1]), corr_mcd=float(mcd.covariance_[0, 1] / np.sqrt(mcd.covariance_[0, 0] * mcd.covariance_[1, 1])),
                       d2_adic=[float(x) for x in dm[200:]], d2_adic_mcd=[float(x) for x in dr[200:]],
                       n_mcd=int((dr > u99).sum()), ids_mcd=[int(i) for i in pts.id[dr > u99]], n_p=int(len(z)),
                       esperadas_99=float(0.01 * 200), max_d2_200=float(dm[:200].max()))
    fig.tight_layout()
    save(fig, "c41_mahalanobis.png")


# ----------------------------------------------------------------------------- 4. kNN, LOF
def knn_lof_fig():
    d = datos()
    z, pts = d["z"], d["pts"]
    al = alertas_puntos()
    rng = np.random.default_rng(RS)
    # dos densidades: un grupo denso y otro disperso, con un punto local cerca del denso
    denso = rng.normal([0, 0], 0.25, (100, 2))
    disp = rng.normal([5, 0], 1.2, (100, 2))
    X2 = np.vstack([denso, disp, [[1.1, 0.0]]])
    k2 = 10
    knn2 = NearestNeighbors(n_neighbors=k2).fit(X2).kneighbors()[0][:, -1]
    lof2 = -LocalOutlierFactor(n_neighbors=k2).fit(X2).negative_outlier_factor_
    rk_knn = int((knn2 > knn2[-1]).sum() + 1)
    rk_lof = int((lof2 > lof2[-1]).sum() + 1)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.plot(np.sort(al["sc_knn"]), ".", color=OK)
    ax.axhline(al["u_knn"], color=RED, ls="--", lw=1.2)
    ax.text(5, al["u_knn"] + 0.08, f"cuantil 95 % = {al['u_knn']:.3f}", fontsize=8.2, color=RED)
    ax.set(xlabel="observaciones ordenadas", ylabel="distancia al vecino 10")
    titulo(ax, f"kNN: {int(al['a_knn'].sum())} de {len(z)} sobre el cuantil 95 %", 9)
    style(ax)
    ax = axs[1]
    ax.scatter(denso[:, 0], denso[:, 1], s=9, c=OK, alpha=0.8)
    ax.scatter(disp[:, 0], disp[:, 1], s=9, c=OK, alpha=0.8)
    ax.scatter(*X2[-1], s=70, c=RED, zorder=3)
    ax.annotate(f"punto local\nkNN: puesto {rk_knn} de {len(X2)}\nLOF = {lof2[-1]:.2f}: puesto {rk_lof}", X2[-1], xytext=(2.0, 2.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlabel="x1", ylabel="x2", xlim=(-1.2, 9), ylim=(-3.6, 3.6))
    titulo(ax, "Densidades distintas: LOF compara con el entorno", 9)
    style(ax)
    ax = axs[2]
    ks = [3, 5, 8, 10, 15, 20, 30, 40]
    cuenta, c206, c204 = [], [], []
    for k in ks:
        s = -LocalOutlierFactor(n_neighbors=k).fit(z).negative_outlier_factor_
        cuenta.append(int((s > 1.5).sum()))
        c206.append(float(s[205]))
        c204.append(float(s[203]))
    ax.plot(ks, cuenta, "o-", color=C41, label="alertas con LOF > 1.5")
    ax.set(xlabel="número de vecinos k", ylabel="alertas")
    ax2 = ax.twinx()
    ax2.plot(ks, c206, "s--", color=VIOLET, ms=5, label="LOF del caso 206")
    ax2.axhline(1.5, color=RED, ls=":", lw=1)
    ax2.set_ylabel("LOF del caso 206", color=VIOLET)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.6, frameon=True, loc="lower right")
    titulo(ax, "El corte 1.5 es sensible a k", 9)
    style(ax)
    NUM["knn"] = dict(k=10, umbral=al["u_knn"], n_alertas=int(al["a_knn"].sum()), ids=[int(i) for i in pts.id[al["a_knn"]]],
                      p_alertas=float(al["a_knn"].mean()), scores_adic=[float(x) for x in al["sc_knn"][200:]],
                      por_k={str(k): int((NearestNeighbors(n_neighbors=k).fit(z).kneighbors()[0][:, -1] > np.quantile(NearestNeighbors(n_neighbors=k).fit(z).kneighbors()[0][:, -1], 0.95)).sum()) for k in (5, 10, 20)})
    NUM["lof"] = dict(n_alertas=int(al["a_lof"].sum()), ids=[int(i) for i in pts.id[al["a_lof"]]], scores_adic=[float(x) for x in al["lof"][200:]],
                      ks=ks, cuenta_por_k=cuenta, lof206_por_k=c206, lof204_por_k=c204, toy_lof=float(lof2[-1]), toy_rank_knn=rk_knn, toy_rank_lof=rk_lof,
                      toy_knn=float(knn2[-1]), toy_knn_denso_max=float(knn2[:100].max()), toy_knn_disp_med=float(np.median(knn2[100:200])))
    fig.tight_layout()
    save(fig, "c41_knn_lof.png")


# ----------------------------------------------------------------------------- 5. DBSCAN y coincidencias
def dbscan_fig():
    d = datos()
    z, pts = d["z"], d["pts"]
    al = alertas_puntos()
    db = al["db"]
    lab = db.labels_
    nucleo = np.zeros(len(z), bool)
    nucleo[db.core_sample_indices_] = True
    borde = (lab >= 0) & ~nucleo
    ruido = lab < 0
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    ax.scatter(z[nucleo, 0], z[nucleo, 1], s=12, c=OK, alpha=0.7, label=f"núcleo ({nucleo.sum()})")
    ax.scatter(z[borde, 0], z[borde, 1], s=26, c=AMBER, label=f"borde ({borde.sum()})")
    ax.scatter(z[ruido, 0], z[ruido, 1], s=36, c=RED, marker="x", label=f"ruido ({ruido.sum()})", zorder=3)
    ax.legend(fontsize=7.8, frameon=True, loc="upper left")
    ax.set(xlabel="x1 estandarizada", ylabel="x2 estandarizada")
    titulo(ax, "DBSCAN: ε = 0.45, minPts = 6", 9)
    style(ax)
    ax = axs[1]
    eps_l = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]
    rr = [int((DBSCAN(eps=e, min_samples=6).fit(z).labels_ == -1).sum()) for e in eps_l]
    ax.plot(eps_l, rr, "o-", color=C41)
    ax.axvline(0.45, color=RED, ls="--", lw=1)
    ax.text(0.47, max(rr) * 0.8, "ε del profesor", color=RED, fontsize=8)
    ax.set(xlabel="radio ε", ylabel="puntos de ruido")
    titulo(ax, "El ruido depende mucho de ε", 9)
    style(ax)
    ax = axs[2]
    nombres = ["Mahalanobis", "kNN", "LOF", "DBSCAN"]
    M = np.column_stack([al["a_m"], al["a_knn"], al["a_lof"], al["a_db"]])[200:].astype(int)
    ax.imshow(M.T, cmap=matplotlib.colors.ListedColormap(["#F3F4F6", RED]), aspect="auto")
    for i in range(4):
        for j in range(6):
            ax.text(j, i, "Sí" if M[j, i] else "No", ha="center", va="center", fontsize=8.6, color="white" if M[j, i] else INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels([str(i) for i in range(201, 207)])
    ax.set_yticks(range(4))
    ax.set_yticklabels(nombres, fontsize=8.4)
    ax.set_xlabel("caso adicional (id)")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    titulo(ax, "Coincidencias en los 6 casos añadidos", 9)
    A = np.column_stack([al["a_m"], al["a_knn"], al["a_lof"], al["a_db"]])
    votos = A.sum(1)
    NUM["dbscan"] = dict(n_nucleo=int(nucleo.sum()), n_borde=int(borde.sum()), n_ruido=int(ruido.sum()), n_clusters=int(len(set(lab[lab >= 0]))),
                         ruido_por_eps={f"{e:g}": r for e, r in zip(eps_l, rr)}, ids_ruido=[int(i) for i in pts.id[ruido]])
    NUM["coinc"] = dict(totales=dict(zip(nombres, [int(x) for x in A.sum(0)])), votos_hist={str(k): int((votos == k).sum()) for k in range(5)},
                        matriz=[[bool(x) for x in fila] for fila in A[200:]], todos4=[int(i) for i in pts.id[votos == 4]],
                        al_menos1=int((votos >= 1).sum()), n=int(len(votos)), ids_3o4=[int(i) for i in pts.id[votos >= 3]])
    fig.tight_layout()
    save(fig, "c41_dbscan.png")


# ----------------------------------------------------------------------------- 6. PCA
def pca_fig():
    d = datos()
    zc, can = d["zc"], d["can"]
    pca = PCA().fit(zc)
    acum = np.cumsum(pca.explained_variance_ratio_)
    r = 5
    P = pca.components_[:r]
    T = zc @ P.T
    q = np.sum((zc - T @ P) ** 2, axis=1)
    umbral = float(np.quantile(q, 0.95))
    t2 = np.sum(T ** 2 / pca.explained_variance_[:r], axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    p1 = np.array([1, 1]) / np.sqrt(2)
    rng = np.random.default_rng(RS)
    t = rng.normal(0, 1.5, 120)
    base = np.outer(t, p1) + rng.normal(0, 0.25, (120, 2))
    ax.scatter(base[:, 0], base[:, 1], s=9, c=OK, alpha=0.6)
    ax.plot([-4.3, 5.2], [-4.3, 5.2], color=C41, lw=1.6, label="componente 1")
    for nm, pt, col in [("A", np.array([2.0, 0.0]), RED), ("B", np.array([4.0, 4.0]), BLUE)]:
        rec = (pt @ p1) * p1
        ax.scatter(*pt, c=col, s=60, zorder=3)
        ax.plot([pt[0], rec[0]], [pt[1], rec[1]], color=col, lw=1.4, ls=":")
        ax.annotate(f"{nm}: ({pt[0]:g}, {pt[1]:g}), Q = {np.sum((pt - rec) ** 2):.0f}", pt, xytext=(2.3, -1.4) if nm == "A" else (0.2, 4.6), fontsize=8, color=col)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax.set(xlim=(-4.5, 5.5), ylim=(-4.5, 5.5), xlabel="z1", ylabel="z2")
    ax.set_aspect("equal")
    titulo(ax, "Q mide la distancia al subespacio", 9)
    style(ax)
    ax = axs[1]
    ax.bar(range(1, 8), pca.explained_variance_ratio_ * 100, color=OK, alpha=0.75, label="por componente")
    ax.plot(range(1, 8), acum * 100, "o-", color=C41, label="acumulada")
    ax.axhline(80, color=MUTED, ls=":")
    ax.axvline(5, color=RED, ls="--", lw=1)
    ax.text(5.15, 50, f"5 componentes: {acum[4] * 100:.2f} %", color=RED, fontsize=8)
    ax.set(xlabel="componentes retenidas", ylabel="varianza explicada (%)", ylim=(0, 105))
    ax.legend(fontsize=7.8, frameon=False, loc="center right")
    titulo(ax, "Canciones: varianza explicada", 9)
    style(ax)
    ax = axs[2]
    a = q > umbral
    ax.scatter(can.id[~a], q[~a], s=14, c=OK, alpha=0.75)
    ax.scatter(can.id[a], q[a], s=30, c=RED, zorder=3)
    ax.axhline(umbral, color=RED, ls="--", lw=1)
    for i in np.argsort(-q)[:5]:
        ax.annotate(str(int(can.id[i])), (can.id[i], q[i]), xytext=(3, 3), textcoords="offset points", fontsize=7.6)
    ax.set(xlabel="canción (id)", ylabel="Q (residuos al cuadrado)")
    titulo(ax, f"Q con r = 5: {int(a.sum())} canciones sobre el cuantil 95 %", 9)
    style(ax)
    orden = np.argsort(-q)
    orden_t2 = np.argsort(-t2)
    corr = float(np.corrcoef(q, t2)[0, 1])
    NUM["pca"] = dict(acum=[float(x) for x in acum], var=[float(x) for x in pca.explained_variance_ratio_], eigen=[float(x) for x in pca.explained_variance_],
                      r=r, umbral=umbral, n_alertas=int(a.sum()), top_ids=[int(can.id[i]) for i in orden[:10]], top_q=[float(q[i]) for i in orden[:10]],
                      top_t2_ids=[int(can.id[i]) for i in orden_t2[:5]], top_t2=[float(t2[i]) for i in orden_t2[:5]], corr_q_t2=corr, n=int(len(q)),
                      inter_top10=len(set(int(can.id[i]) for i in orden[:10]) & set(int(can.id[i]) for i in orden_t2[:10])),
                      q_por_r={str(rr): float(np.sum((zc - zc @ pca.components_[:rr].T @ pca.components_[:rr]) ** 2, axis=1).mean()) for rr in range(1, 8)},
                      max_q_r7=float(np.sum((zc - zc @ pca.components_.T @ pca.components_) ** 2, axis=1).max()),
                      cols=[c for c in can.columns if c != "id"])
    fig.tight_layout()
    save(fig, "c41_pca.png")


# ----------------------------------------------------------------------------- 7. Isolation Forest
def _cortes_aleatorios(X, objetivo, rng, tope=60):
    """Camina un árbol de aislamiento (cortes al azar sobre una variable) y devuelve la profundidad a la que queda solo el objetivo."""
    idx = np.arange(len(X))
    prof = 0
    while len(idx) > 1 and prof < tope:
        j = rng.integers(0, X.shape[1])
        lo, hi = X[idx, j].min(), X[idx, j].max()
        if hi == lo:
            break
        c = rng.uniform(lo, hi)
        lado = X[idx, j] < c
        idx = idx[lado == (X[objetivo, j] < c)]
        prof += 1
    return prof


def iforest_fig():
    d = datos()
    z, pts = d["z"], d["pts"]
    rng = np.random.default_rng(RS)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    # panel 1: una secuencia de cortes aleatorios que aísla un punto extremo (id 201) y un punto del centro
    ax = axs[0]
    ax.scatter(z[:200, 0], z[:200, 1], s=9, c=OK, alpha=0.6)
    ax.scatter(*z[200], c=RED, s=55, zorder=3)
    cen = int(np.argmin(np.sum(z[:200] ** 2, axis=1)))
    ax.scatter(*z[cen], c=BLUE, s=55, zorder=3)
    ax.annotate("id 201", z[200], xytext=(-34, -2), textcoords="offset points", fontsize=8, color=RED)
    ax.annotate("punto central", z[cen], xytext=(6, -14), textcoords="offset points", fontsize=8, color=BLUE)
    xmin, xmax, ymin, ymax = -4.0, 5.3, -4.0, 5.3
    for semilla in range(100):
        rng_demo = np.random.default_rng(semilla)
        idx = np.arange(len(z))
        trazos = []
        while len(idx) > 1 and len(trazos) < 6:
            j = int(rng_demo.integers(0, 2))
            c = rng_demo.uniform(z[idx, j].min(), z[idx, j].max())
            trazos.append((j, c))
            idx = idx[(z[idx, j] < c) == (z[200, j] < c)]
        if len(idx) == 1 and len(trazos) == 3:
            break
    cortes = len(trazos)
    for j, c in trazos:
        if j == 0:
            ax.plot([c, c], [ymin, ymax], color=C41, lw=1.2)
        else:
            ax.plot([xmin, xmax], [c, c], color=C41, lw=1.2)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x1 estandarizada", ylabel="x2 estandarizada")
    titulo(ax, f"Una secuencia de {cortes} cortes al azar aísla al id 201", 8.8)
    style(ax)
    # panel 2: distribución de profundidades en muchos árboles de aislamiento (submuestra de 128 puntos)
    ax = axs[1]
    prof_ext, prof_cen = [], []
    for _ in range(2000):
        sub = rng.choice(np.arange(len(z)), 128, replace=False)
        sub = np.unique(np.concatenate([sub, [200, cen]]))
        zz = z[sub]
        prof_ext.append(_cortes_aleatorios(zz, int(np.where(sub == 200)[0][0]), rng))
        prof_cen.append(_cortes_aleatorios(zz, int(np.where(sub == cen)[0][0]), rng))
    bins = np.arange(0, 32, 1)
    ax.hist(prof_cen, bins=bins, color=BLUE, alpha=0.6, label=f"punto central (media {np.mean(prof_cen):.1f})", density=True)
    ax.hist(prof_ext, bins=bins, color=RED, alpha=0.7, label=f"id 201 (media {np.mean(prof_ext):.1f})", density=True)
    ax.set(xlabel="cortes necesarios para aislar", ylabel="proporción de árboles")
    ax.legend(fontsize=7.8, frameon=False)
    titulo(ax, "2 000 árboles aleatorios: los extremos se aíslan antes", 8.8)
    style(ax)
    # panel 3: puntaje en función de la profundidad media
    ax = axs[2]
    cc = c_psi(256)
    h = np.linspace(1, 20, 200)
    ax.plot(h, 2 ** (-h / cc), color=C41, lw=2, label="s = 2^(−E[h]/c(256))")
    for hh, col in [(2, RED), (cc, MUTED), (14, BLUE)]:
        ax.scatter([hh], [2 ** (-hh / cc)], c=col, s=50, zorder=3)
        ax.text(hh + 0.4, 2 ** (-hh / cc) + 0.015, f"h = {hh:.1f} → s = {2 ** (-hh / cc):.3f}", fontsize=8, color=col)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xlabel="longitud media del camino E[h]", ylabel="puntaje s", ylim=(0.2, 1.0))
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    titulo(ax, f"Puntaje con submuestra 256: c = {cc:.2f}", 9)
    style(ax)
    # verificación de la fórmula con scikit-learn (cardio)
    cm = cardio_modelos()
    m = cm["m_if"]
    from sklearn.ensemble._iforest import _average_path_length
    X = d["zte"]
    tot = np.zeros(len(X))
    for t, fs in zip(m.estimators_, m.estimators_features_):
        hoja = t.apply(X[:, fs])
        prof = t.decision_path(X[:, fs]).sum(axis=1).A1 - 1
        tot += prof + _average_path_length(t.tree_.n_node_samples[hoja])
    hmed = tot / len(m.estimators_)
    sc = 2 ** (-hmed / c_psi(256))
    NUM["iforest"] = dict(c256=float(cc), c8=float(c_psi(8)), s_h2_c8=float(2 ** (-2 / 8)), s_h2_c256=float(2 ** (-2 / cc)), s_h_c=0.5,
                          prof_ext_media=float(np.mean(prof_ext)), prof_cen_media=float(np.mean(prof_cen)), prof_ext_mediana=float(np.median(prof_ext)),
                          prof_cen_mediana=float(np.median(prof_cen)), dif_formula=float(np.abs(-sc - m.score_samples(X)).max()),
                          h_min_test=float(hmed.min()), h_max_test=float(hmed.max()), id_central=int(pts.id[cen]))
    fig.tight_layout()
    save(fig, "c41_iforest.png")


# ----------------------------------------------------------------------------- 8. One-Class SVM
def ocsvm_fig():
    d = datos()
    z, pts = d["z"], d["pts"]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    gx, gy = np.meshgrid(np.linspace(-4, 5.3, 220), np.linspace(-4, 5.3, 220))
    rejilla = np.c_[gx.ravel(), gy.ravel()]
    res = {}
    for ax, nu in zip(axs[:2], (0.05, 0.10)):
        o = OneClassSVM(nu=nu, gamma="scale").fit(z)
        f = o.decision_function(rejilla).reshape(gx.shape)
        pr = o.predict(z) == -1
        ax.contourf(gx, gy, f, levels=[f.min(), 0, f.max()], colors=["#FDE68A", "#FFFFFF"], alpha=0.7)
        ax.contour(gx, gy, f, levels=[0], colors=[C41], linewidths=1.6)
        ax.scatter(z[~pr, 0], z[~pr, 1], s=12, c=OK, alpha=0.7)
        ax.scatter(z[pr, 0], z[pr, 1], s=34, c=RED, zorder=3)
        ax.set(xlabel="x1 estandarizada", ylabel="x2 estandarizada", xlim=(-4, 5.3), ylim=(-4, 5.3))
        titulo(ax, f"ν = {nu:g}: {int(pr.sum())} alertas (casos 201–206: {int(pr[200:].sum())})", 9)
        style(ax)
        res[f"{nu:g}"] = dict(n=int(pr.sum()), adic=[bool(x) for x in pr[200:]], n_sv=int(len(o.support_)), gamma=float(1 / (z.shape[1] * z.var())))
    ax = axs[2]
    nus = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    ns = [int((OneClassSVM(nu=nu, gamma="scale").fit(z).predict(z) == -1).sum()) for nu in nus]
    nsv = [int(len(OneClassSVM(nu=nu, gamma="scale").fit(z).support_)) for nu in nus]
    ax.plot(nus, [n / len(z) for n in ns], "o-", color=C41, label="fracción de alertas")
    ax.plot(nus, [n / len(z) for n in nsv], "s--", color=VIOLET, label="fracción de vectores soporte")
    ax.plot([0, 0.3], [0, 0.3], color=MUTED, ls=":", label="y = ν")
    ax.set(xlabel="ν", ylabel="fracción de las 206 observaciones")
    ax.legend(fontsize=7.8, frameon=False)
    titulo(ax, "ν: cota inferior de los vectores soporte; alertas ≈ ν", 9)
    style(ax)
    NUM["ocsvm"] = dict(por_nu=res, nus=nus, n_alertas=ns, n_sv=nsv, n=int(len(z)))
    fig.tight_layout()
    save(fig, "c41_ocsvm.png")


# ----------------------------------------------------------------------------- 9. Cardio: ROC, PR, umbral
def cardio_metricas(y, clase, score):
    vn = int(((y == 0) & (clase == 0)).sum())
    fp = int(((y == 0) & (clase == 1)).sum())
    fn = int(((y == 1) & (clase == 0)).sum())
    vp = int(((y == 1) & (clase == 1)).sum())
    n = len(y)
    return dict(VN=vn, FP=fp, FN=fn, VP=vp, alertas=fp + vp, acc=(vn + vp) / n, recall=vp / (vp + fn), precision=vp / (vp + fp) if vp + fp else 0.0,
                auc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)))


def cardio_fig():
    d = datos()
    y, ytr = d["yte"], d["ytr"]
    cm = cardio_modelos()
    u_pca = float(np.quantile(cm["q_tr"], 0.95))
    u_if = float(np.quantile(cm["s_tr"], 0.95))
    c_pca, c_if = (cm["q_te"] > u_pca).astype(int), (cm["s_te"] > u_if).astype(int)
    m_pca, m_if = cardio_metricas(y, c_pca, cm["q_te"]), cardio_metricas(y, c_if, cm["s_te"])
    prev = float(y.mean())
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    for sc, nm, col in [(cm["q_te"], "PCA (Q)", BLUE), (cm["s_te"], "Isolation Forest", C41)]:
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{nm}: AUC {roc_auc_score(y, sc):.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":")
    ax.scatter([m_if["FP"] / (m_if["FP"] + m_if["VN"])], [m_if["recall"]], c=RED, s=50, zorder=3)
    ax.annotate("corte del cuantil 95 %", (m_if["FP"] / (m_if["FP"] + m_if["VN"]), m_if["recall"]), xytext=(0.2, 0.15), fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlabel="tasa de falsos positivos", ylabel="sensibilidad (recall)")
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    titulo(ax, "ROC: el orden de los puntajes", 9)
    style(ax)
    ax = axs[1]
    for sc, nm, col in [(cm["q_te"], "PCA (Q)", BLUE), (cm["s_te"], "Isolation Forest", C41)]:
        pr, rc, _ = precision_recall_curve(y, sc)
        ax.plot(rc, pr, color=col, lw=1.8, label=f"{nm}: AP {average_precision_score(y, sc):.3f}")
    ax.axhline(prev, color=MUTED, ls=":")
    ax.text(0.55, prev + 0.02, f"azar: precisión = prevalencia {prev:.3f}", fontsize=8, color=MUTED)
    ax.scatter([m_if["recall"]], [m_if["precision"]], c=RED, s=50, zorder=3)
    ax.set(xlabel="sensibilidad (recall)", ylabel="precisión", ylim=(0, 1.05))
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    titulo(ax, "Precisión–recall: pocas anomalías (9.8 %)", 9)
    style(ax)
    ax = axs[2]
    qs = [0.80, 0.85, 0.90, 0.95, 0.99]
    tabla = []
    for q in qs:
        u = float(np.quantile(cm["s_tr"], q))
        c = (cm["s_te"] > u).astype(int)
        tabla.append(dict(q=q, umbral=u, alertas=int(c.sum()), recall=float(recall_score(y, c)), precision=float(precision_score(y, c)), f1=float(f1_score(y, c)),
                          vp=int(((y == 1) & (c == 1)).sum())))
    ax.plot([t["q"] * 100 for t in tabla], [t["recall"] for t in tabla], "o-", color=BLUE, label="sensibilidad")
    ax.plot([t["q"] * 100 for t in tabla], [t["precision"] for t in tabla], "s-", color=RED, label="precisión")
    ax.plot([t["q"] * 100 for t in tabla], [t["f1"] for t in tabla], "^--", color=MUTED, label="F1")
    for t in tabla:
        ax.text(t["q"] * 100, 1.03, f"{t['alertas']}", ha="center", fontsize=7.6, color=INK)
    ax.text(79.2, 1.075, "alertas en prueba:", fontsize=7.6, color=INK)
    ax.set(xlabel="cuantil del corte (entrenamiento)", ylabel="valor", ylim=(0, 1.15))
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    titulo(ax, "Más alertas: más sensibilidad, menos precisión", 9)
    style(ax)
    # umbral por validación cruzada en entrenamiento (sin usar la prueba)
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    oof = np.zeros(len(ytr))
    for i, j in skf.split(d["ztr"], ytr):
        oof[j] = -isoforest(d["ztr"][i]).score_samples(d["ztr"][j])
    f1q = {}
    for q in qs:
        c = (oof > np.quantile(oof, q)).astype(int)
        f1q[f"{q:g}"] = float(f1_score(ytr, c))
    q_elegido = float(max(qs, key=lambda q: f1q[f"{q:g}"]))
    u_el = float(np.quantile(cm["s_tr"], q_elegido))
    c_el = (cm["s_te"] > u_el).astype(int)
    m_el = cardio_metricas(y, c_el, cm["s_te"])
    # corte automático de sklearn (predict)
    pred_auto = cm["m_if"].predict(d["zte"]) == -1
    m_auto = cardio_metricas(y, pred_auto.astype(int), cm["s_te"])
    todo_normal = float(np.mean(y == 0))
    # Hotelling T2 como complemento
    NUM["cardio"] = dict(n=int(len(d["car"])), n_tr=int(len(ytr)), n_te=int(len(y)), anom_tot=int(d["car"].anomalia.sum()), anom_tr=int(ytr.sum()), anom_te=int(y.sum()),
                         prev_tr=float(ytr.mean()), prev_te=prev, r=int(cm["r"]), u_pca=u_pca, u_if=u_if, pca=m_pca, iff=m_if, todo_normal=todo_normal,
                         auc_t2=float(roc_auc_score(y, cm["t2_te"])), tabla=tabla, f1_oof=f1q, q_elegido=q_elegido, u_elegido=u_el, elegido=m_el, auto=m_auto,
                         offset_auto=float(cm["m_if"].offset_), recall_max_5=float(min(1.0, 0.05 * len(y) / y.sum())))
    fig.tight_layout()
    save(fig, "c41_cardio.png")


# ----------------------------------------------------------------------------- 10. Robustez del resultado
def robustez_fig():
    d = datos()
    y, ytr, ztr, zte = d["yte"], d["ytr"], d["ztr"], d["zte"]
    cm = cardio_modelos()
    semillas = [RS, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    aucs, recs, precs = [], [], []
    for s in semillas:
        m = isoforest(ztr, seed=s)
        a, b = -m.score_samples(ztr), -m.score_samples(zte)
        c = (b > np.quantile(a, 0.95)).astype(int)
        aucs.append(float(roc_auc_score(y, b)))
        recs.append(float(recall_score(y, c)))
        precs.append(float(precision_score(y, c)))
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.scatter(np.zeros(len(aucs)) + np.linspace(-0.12, 0.12, len(aucs)), aucs, c=OK, s=34, alpha=0.85, label="Python, 10 semillas")
    ax.scatter([0], [aucs[0]], c=C41, s=70, zorder=3, label=f"semilla 2026: {aucs[0]:.4f}")
    ax.scatter([0.35], [0.9220], c=VIOLET, marker="D", s=60, label="R (isotree), diap. 47: 0.9220")
    ax.axhline(cm["pca"].explained_variance_ratio_[0] * 0 + 0.8503, color=BLUE, ls=":", lw=1)
    ax.text(-0.42, 0.853, "PCA: 0.8503 (determinista)", fontsize=8, color=BLUE)
    ax.set(xlim=(-0.5, 0.6), ylim=(0.84, 0.94), xticks=[], ylabel="AUC en prueba")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    titulo(ax, "La diferencia R–Python cabe en la variación por semilla", 8.6)
    style(ax, "y")
    ax = axs[1]
    P_all = cm["pca"].components_
    rs = [2, 3, 5, 8, 10, 12, 16, 20]
    auc_q = []
    for rr in rs:
        Pm = P_all[:rr]
        auc_q.append(float(roc_auc_score(y, np.sum((zte - zte @ Pm.T @ Pm) ** 2, axis=1))))
    ax.plot(rs, auc_q, "o-", color=BLUE, label="AUC con Q")
    ax.axvline(cm["r"], color=RED, ls="--", lw=1)
    ax.text(cm["r"] + 0.3, min(auc_q) + 0.003, f"r = {cm['r']} (≥ 80 %)", fontsize=8, color=RED)
    ax.axhline(roc_auc_score(y, cm["t2_te"]), color=TEAL, ls=":", lw=1.4)
    ax.text(rs[0], roc_auc_score(y, cm["t2_te"]) + 0.004, f"T² (r = {cm['r']}): {roc_auc_score(y, cm['t2_te']):.3f}", fontsize=8, color=TEAL)
    ax.set(xlabel="componentes retenidas r", ylabel="AUC en prueba")
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    titulo(ax, "El PCA depende de r y de Q frente a T²", 9)
    style(ax)
    ax = axs[2]
    m_n = isoforest(ztr[ytr == 0])
    s_n = -m_n.score_samples(zte)
    a_cont, a_norm = roc_auc_score(y, cm["s_te"]), roc_auc_score(y, s_n)
    ap_cont, ap_norm = average_precision_score(y, cm["s_te"]), average_precision_score(y, s_n)
    xs = np.arange(2)
    ax.bar(xs - 0.2, [a_cont, a_norm], 0.38, color=C41, label="AUC")
    ax.bar(xs + 0.2, [ap_cont, ap_norm], 0.38, color=BLUE, label="AP")
    for i, (a, b) in enumerate([(a_cont, ap_cont), (a_norm, ap_norm)]):
        ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set(xticks=xs, xticklabels=["entrenado con\nanomalías (9.5 %)", "entrenado solo\ncon normales"], ylim=(0, 1.1))
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(ax, "Contaminar el entrenamiento cuesta", 9)
    style(ax, "y")
    # n_estimators y max_samples
    ne_l = [50, 100, 300, 1000]
    ms_l = [64, 128, 256, 512, 1281]
    auc_ne = {str(n): float(roc_auc_score(y, -isoforest(ztr, ne=n).score_samples(zte))) for n in ne_l}
    auc_ms = {str(n): float(roc_auc_score(y, -isoforest(ztr, ms=n).score_samples(zte))) for n in ms_l}
    # escala: el Isolation Forest no cambia con la estandarización
    car = d["car"]
    tr, te = car[car.conjunto == "entrenamiento"], car[car.conjunto == "prueba"]
    m_raw = isoforest(tr[d["V"]].to_numpy())
    s_raw = -m_raw.score_samples(te[d["V"]].to_numpy())
    NUM["robustez"] = dict(semillas=semillas, auc=aucs, recall=recs, precision=precs, auc_min=min(aucs), auc_max=max(aucs), auc_media=float(np.mean(aucs)),
                           rec_min=min(recs), rec_max=max(recs), prec_min=min(precs), prec_max=max(precs), r_lista=rs, auc_q_por_r=auc_q,
                           auc_cont=float(a_cont), auc_norm=float(a_norm), ap_cont=float(ap_cont), ap_norm=float(ap_norm), auc_por_ne=auc_ne, auc_por_ms=auc_ms,
                           if_escala_igual=bool(np.allclose(s_raw, cm["s_te"])), corr_escala=float(np.corrcoef(s_raw, cm["s_te"])[0, 1]),
                           r_isotree=dict(acc=0.9127, rec=0.2963, prec=0.6154, auc=0.9220))
    fig.tight_layout()
    save(fig, "c41_robustez.png")


# ----------------------------------------------------------------------------- 11. Esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.4, 5.3))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def caja(x, y, w, h, color, tit, lineas, fs=7.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.45), w, 0.45, boxstyle="round,pad=0.03", fc=color, ec=color))
        ax.text(x + w / 2, y + h - 0.23, tit, ha="center", va="center", fontsize=9.2, color="white", fontweight="bold")
        for i, t in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.75 - i * 0.27, t, ha="center", va="center", fontsize=fs, color=INK)
    caja(0.1, 3.55, 2.3, 2.3, BLUE, "1 variable", ["RIC: [Q1 − 1.5·RIC, Q3 + 1.5·RIC]", "z: |z| > 3 (¡puede no alertar!)", "robusta: mediana y MAD", "exploración, no prueba"])
    caja(2.55, 3.55, 2.3, 2.3, TEAL, "Distancias", ["Mahalanobis: D² > χ²(p; 0.99)", "kNN: distancia al vecino k", "escala: estandarizar antes", "covarianza: sensible a extremos"])
    caja(5.0, 3.55, 2.3, 2.3, AMBER, "Densidad", ["LOF: densidad vs vecinos", "≈ 1 normal; muy sobre 1: alerta", "DBSCAN: ruido (sin región)", "depende de ε y minPts"])
    caja(7.45, 3.55, 2.3, 2.3, VIOLET, "Reconstrucción", ["PCA: Q = Σ residuos²", "T²: extremo dentro del plano", "r por varianza (≥ 80 %)", "ajustar solo con entrenamiento"])
    caja(9.9, 3.55, 2.5, 2.3, C41, "Aislamiento", ["Isolation Forest: caminos cortos", "s = 2^(−E[h]/c(ψ))", "One-Class SVM: frontera ν", "ν ≈ fracción de alertas"])
    for xx in (2.4, 4.85, 7.3, 9.75):
        ax.annotate("", xy=(xx + 0.18, 4.7), xytext=(xx, 4.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 3.2, boxstyle="round,pad=0.03", fc=C41_BG, ec=C41, lw=1.4))
    pc, ca, mh, ric = NUM["pca"], NUM["cardio"], NUM["maha"], NUM["ricz"]
    ax.text(6.25, 2.95, "Datos de la clase: puntos.csv (206 puntos, 2 variables) · canciones.csv (197 canciones, 7 variables) · cardio.csv (1 831 registros, 21 variables, etiqueta)", ha="center", fontsize=8.4, color=C41, fontweight="bold")
    ax.text(6.25, 2.5, f"Ocho valores: RIC marca el 30 (límites {ric['L']:g} y {ric['U']:g}); z = {ric['z30']:.3f} no pasa de 3 · puntos: Mahalanobis {mh['n_alertas']['0.99']}, kNN {NUM['knn']['n_alertas']}, LOF {NUM['lof']['n_alertas']}, DBSCAN {NUM['dbscan']['n_ruido']} alertas", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 2.1, f"Canciones: 5 componentes explican {pc['acum'][4] * 100:.2f} % · Q (cuantil 95 %) marca {pc['n_alertas']} canciones; la mayor es la {pc['top_ids'][0]} (Q = {pc['top_q'][0]:.3f})", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 1.7, f"Cardio ({ca['n_tr']} entrenamiento / {ca['n_te']} prueba): PCA con {ca['r']} componentes, AUC {ca['pca']['auc']:.3f}; Isolation Forest AUC {ca['iff']['auc']:.3f}, sensibilidad {ca['iff']['recall']:.3f}, precisión {ca['iff']['precision']:.3f}", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 1.3, f"Marcar todo como normal acierta {ca['todo_normal'] * 100:.2f} %: la exactitud ({ca['iff']['acc'] * 100:.2f} %) casi no supera a esa referencia", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 0.85, "R ↔ Python:  quantile ↔ np.quantile · mahalanobis ↔ cálculo con np.linalg.inv · kNNdist ↔ NearestNeighbors · lof (minPts = k + 1) ↔ LocalOutlierFactor(k)", ha="center", fontsize=7.8, color=MUTED, style="italic")
    ax.text(6.25, 0.45, "dbscan (ruido = 0) ↔ DBSCAN (ruido = −1) · prcomp ↔ PCA · isotree (puntaje alto = anómalo) ↔ IsolationForest (score_samples bajo = anómalo)", ha="center", fontsize=7.8, color=MUTED, style="italic")
    save(fig, "c41_esquema.png")


PASOS = dict(tipos=tipos_fig, ricz=ricz_fig, maha=maha_fig, knn=knn_lof_fig, dbscan=dbscan_fig, pca=pca_fig, iforest=iforest_fig, ocsvm=ocsvm_fig,
             cardio=cardio_fig, robustez=robustez_fig, esquema=esquema_fig)

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c41_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or list(PASOS)
    if "esquema" in elegidos:
        for k in ("ricz", "maha", "knn", "dbscan", "pca", "cardio"):
            if k not in elegidos and k not in NUM:
                elegidos.append(k)
    orden = [e for e in PASOS if e in elegidos and e != "esquema"] + (["esquema"] if "esquema" in elegidos else [])
    for k in orden:
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c41_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
