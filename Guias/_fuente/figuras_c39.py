"""Figuras de la Clase 39 (Clustering: K-means, K-modes y K-prototypes).
Fuentes: diapositivas clase_39_clustering.pdf, kmeans_numero_clusters.py / .R.txt del profesor Pedro Luiz Ramos y los CSV de la clase
(vinos.csv: 177 vinos con 13 mediciones; vinos_clases_reales.csv: la clase UCI de cada vino, solo para validación externa).
La carpeta de la clase se lee sin modificarla. Todo sale de código; lo aleatorio usa semilla fija. Escribe c39_numeros.json
(los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c39.py   (o con nombres: python -X utf8 figuras_c39.py iter codo silueta)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 2026
C39 = "#0E7490"      # color de la clase 39 (cian oscuro)
C39_BG = "#ECFEFF"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
OK = "#4B79B5"
G3 = ["#2563EB", "#D97706", "#0D9488"]            # grupos 1, 2, 3 (etiquetas de Python, como el script del profesor)
PAL = ["#4067A0", "#C66A25", "#247D72"]            # paleta del profesor para las clases reales 1, 2, 3
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA39 = next(p for p in BASE.glob("Clase 39*") if p.is_dir())
_DAT = {}
CAT_LAB = ["bajo", "medio", "alto"]


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


# ----------------------------------------------------------------------------- datos y funciones comunes
def datos():
    """Lee vinos.csv y las clases reales (una sola vez), con la misma estandarización del script del profesor (ddof = 1)."""
    if _DAT:
        return _DAT
    v = pd.read_csv(CARPETA39 / "vinos.csv")
    cl = pd.read_csv(CARPETA39 / "vinos_clases_reales.csv").set_index("codigo")
    y = v["codigo"].map(cl["clase_real"]).to_numpy()
    x = v.drop(columns="codigo")
    z = (x - x.mean()) / x.std(ddof=1)
    cat = pd.DataFrame({"alcohol": np.where(v.alcohol < 13, "bajo", "alto"),
                        "flavonoides": np.where(v.flavonoides < 1.5, "bajo", np.where(v.flavonoides < 3, "medio", "alto")),
                        "color": np.where(v.color < 4, "bajo", np.where(v.color < 6.5, "medio", "alto"))})
    _DAT.update(v=v, x=x, z=z, y=y, cat=cat)
    return _DAT


def kmeans(Z, k, n_init=50, seed=RS, **kw):
    """Los mismos parámetros del script del profesor: n_init = 50, max_iter = 100, Lloyd, tol = 1e-8."""
    return KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=seed, algorithm="lloyd", tol=1e-8, **kw).fit(Z)


def ajuste_k3():
    d = datos()
    km = kmeans(d["z"], 3)
    return km, km.labels_ + 1


def lloyd(Z, C0, tope=30):
    """K-means de Lloyd a mano: devuelve, por iteración, etiquetas, centros nuevos y WSS (calculado tras asignar)."""
    C = np.array(C0, float)
    hist, prev = [], None
    for _ in range(tope):
        D = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = D.argmin(1)
        wss = float(D[np.arange(len(Z)), lab].sum())
        Cn = np.array([Z[lab == g].mean(0) if (lab == g).any() else C[g] for g in range(len(C))])
        hist.append(dict(lab=lab, C_ini=C.copy(), C_nuevo=Cn, wss=wss))
        if prev is not None and np.array_equal(lab, prev):
            break
        prev, C = lab, Cn
    return hist


def tss_wss_bss(Z, lab):
    Z = np.asarray(Z, float)
    tss = float(((Z - Z.mean(0)) ** 2).sum())
    wss = float(sum(((Z[lab == g] - Z[lab == g].mean(0)) ** 2).sum() for g in np.unique(lab)))
    return tss, wss, tss - wss


def gap_stat(Z, B=30, seed=RS, n_init=20, kmax=8):
    """Estadístico Gap como en el script del profesor: referencia uniforme en el rango de cada columna, B bases simuladas."""
    wss = np.array([kmeans(Z, k).inertia_ for k in range(1, kmax + 1)])
    rng = np.random.default_rng(seed)
    lo, hi = Z.min(0), Z.max(0)
    ref = np.zeros((B, kmax))
    for b in range(B):
        R = rng.uniform(lo, hi, size=Z.shape)
        for k in range(1, kmax + 1):
            ref[b, k - 1] = np.log(KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=seed + b, algorithm="lloyd", tol=1e-8).fit(R).inertia_)
    gap = ref.mean(0) - np.log(wss)
    err = ref.std(0, ddof=1) * np.sqrt(1 + 1 / B)
    kg = kmax
    for k in range(1, kmax):
        if gap[k - 1] >= gap[k] - err[k]:
            kg = k
            break
    return gap, err, kg, ref, wss


# ----------------------------------------------------------------------------- 1. iteraciones de K-means
def iter_fig():
    d = datos()
    Z = d["z"][["flavonoides", "color"]].to_numpy()
    rng = np.random.default_rng(11)
    mejor = None
    for s in range(60):                      # un inicio "didáctico": converge en 4 a 6 iteraciones
        idx = np.random.default_rng(s).choice(len(Z), 3, replace=False)
        h = lloyd(Z, Z[idx])
        if 5 <= len(h) <= 7 and len(set(h[-1]["lab"])) == 3:
            mejor = (s, idx, h)
            break
    s_ini, idx, h = mejor
    ref = KMeans(3, init=Z[idx], n_init=1, algorithm="lloyd", tol=1e-8, max_iter=100).fit(Z)
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.7))
    mostrar = [0, 1, len(h) - 1]
    for ax, t in zip(axs[:3], mostrar):
        lab, C0, C1 = h[t]["lab"], h[t]["C_ini"], h[t]["C_nuevo"]
        for g in range(3):
            ax.scatter(Z[lab == g, 0], Z[lab == g, 1], s=12, c=G3[g], alpha=0.65)
        ax.scatter(C0[:, 0], C0[:, 1], marker="o", s=70, facecolors="white", edgecolors=INK, lw=1.4, zorder=4)
        ax.scatter(C1[:, 0], C1[:, 1], marker="X", s=90, c=INK, zorder=5)
        for a, b in zip(C0, C1):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        ax.set(xlabel="flavonoides (z)", ylabel="color (z)")
        titulo(ax, f"Iteración {t + 1}: WSS = {h[t]['wss']:.1f}" + ("  (sin cambios)" if t == len(h) - 1 else ""), 9)
        style(ax)
    ax = axs[3]
    ax.plot(range(1, len(h) + 1), [e["wss"] for e in h], "o-", color=C39)
    ax.axhline(ref.inertia_, color=RED, ls="--", lw=1)
    ax.text(len(h), ref.inertia_ * 1.01, f"final: {ref.inertia_:.1f}", ha="right", fontsize=8, color=RED)
    ax.set(xlabel="iteración", ylabel="WSS tras asignar", xticks=range(1, len(h) + 1))
    titulo(ax, "El costo baja o se mantiene", 9)
    style(ax)
    NUM["iter"] = dict(vars=["flavonoides", "color"], semilla_inicio=int(s_ini), idx_inicio=[int(i) for i in idx], n_iter=len(h),
                       wss=[e["wss"] for e in h], wss_sklearn=float(ref.inertia_), n_iter_sklearn=int(ref.n_iter_),
                       coincide=bool(np.allclose(ref.cluster_centers_, np.array(h[-1]["C_nuevo"]), atol=1e-6)),
                       tamanos=[int(c) for c in np.bincount(h[-1]["lab"])], centros_ini=[[float(a) for a in c] for c in Z[idx]])
    fig.tight_layout()
    save(fig, "c39_iteraciones.png")


# ----------------------------------------------------------------------------- 2. WSS, BSS y TSS
def wsbs_fig():
    d = datos()
    Z = d["z"].to_numpy()
    toy = np.array([1, 2, 3, 8, 9, 10.0])
    lab_t = np.array([0, 0, 0, 1, 1, 1])
    tss_t, wss_t, bss_t = tss_wss_bss(toy[:, None], lab_t)
    # K-means de la diapositiva 13 paso a paso con centros iniciales 1 y 8
    h = lloyd(toy[:, None], [[1.0], [8.0]])
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7))
    ax = axs[0]
    asign = h[0]["lab"]
    for g, col in zip(range(2), [BLUE, AMBER]):
        ax.scatter(toy[asign == g], np.zeros((asign == g).sum()), s=70, c=col, zorder=3)
    for xv in toy:
        ax.text(xv, 0.12, f"{xv:g}", ha="center", fontsize=8.5)
    ax.scatter([1, 8], [0, 0], marker="o", s=190, facecolors="none", edgecolors=INK, lw=1.4, zorder=2)
    ax.scatter([2, 9], [-0.32, -0.32], marker="^", s=70, c=INK, zorder=3)
    ax.text(2, -0.62, "μ₁ = 2", ha="center", fontsize=8.5)
    ax.text(9, -0.62, "μ₂ = 9", ha="center", fontsize=8.5)
    ax.text(0.4, 0.62, "círculos: centros iniciales (1 y 8)\ntriángulos: centros tras la iteración 1", fontsize=7.8, color=MUTED, va="top")
    ax.set(xlim=(0, 11), ylim=(-0.9, 0.7), yticks=[], xlabel="valor")
    ax.spines["left"].set_visible(False)
    titulo(ax, "Diapositiva 13: una iteración con k = 2", 9)
    style(ax, "x")
    ax = axs[1]
    ax.bar(["TSS\n77.5", "WSS + BSS\n4 + 73.5"][:1], [tss_t], color=C39, width=0.5, label="TSS")
    ax.bar(["WSS + BSS\n4 + 73.5"], [bss_t], color=G3[0], width=0.5, label="BSS (entre grupos)")
    ax.bar(["WSS + BSS\n4 + 73.5"], [wss_t], bottom=[bss_t], color=G3[1], width=0.5, label="WSS (dentro)")
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    ax.set(ylabel="suma de cuadrados", ylim=(0, 118))
    titulo(ax, "TSS = WSS + BSS (datos 1, 2, 3, 8, 9, 10)", 9)
    style(ax, "y")
    ax = axs[2]
    ks = list(range(1, 9))
    W = [kmeans(Z, k).inertia_ for k in ks]
    T = float((Z ** 2).sum())
    ax.bar(ks, [T - w for w in W], color=G3[0], label="BSS")
    ax.bar(ks, W, bottom=[T - w for w in W], color=G3[1], label="WSS")
    ax.axhline(T, color=INK, lw=0.8)
    ax.text(8.4, T + 30, f"TSS = {T:.0f}", ha="right", fontsize=8)
    ax.set(xlabel="k", ylabel="suma de cuadrados", ylim=(0, T * 1.25))
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", ncol=2)
    titulo(ax, "Vinos: al subir k, WSS baja y BSS sube", 9)
    style(ax, "y")
    mu_g = Z.mean(0)
    NUM["wsbs"] = dict(toy=dict(tss=tss_t, wss=wss_t, bss=bss_t, centros_iter1=[float(c[0]) for c in h[0]["C_nuevo"]], n_iter=len(h),
                                dist_c1=[float((v - 1) ** 2) for v in toy], dist_c2=[float((v - 8) ** 2) for v in toy], wss_por_iter=[e["wss"] for e in h],
                                media=float(toy.mean()), tss_partes=[float((v - toy.mean()) ** 2) for v in toy]),
                       vinos=dict(k=ks, wss=[float(w) for w in W], bss=[float(T - w) for w in W], tss=T, media_global_cero=bool(np.allclose(mu_g, 0, atol=1e-10))))
    fig.tight_layout()
    save(fig, "c39_wsbs.png")


# ----------------------------------------------------------------------------- 3. por qué estandarizar
def estandarizar_fig():
    d = datos()
    x, z, y = d["x"], d["z"], d["y"]
    var = x.var(ddof=1)
    cuota = var / var.sum()
    k_raw = kmeans(x.to_numpy(), 3)
    k_z = kmeans(z.to_numpy(), 3)
    ari_raw, ari_z = adjusted_rand_score(y, k_raw.labels_), adjusted_rand_score(y, k_z.labels_)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    orden = var.sort_values().index
    ax.barh([o.replace("_", " ") for o in orden], var[orden], color=[RED if o == "prolina" else OK for o in orden])
    ax.set_xscale("log")
    ax.set(xlabel="varianza (escala logarítmica)")
    ax.text(var["prolina"] * 0.9, 12, f"prolina: {var['prolina']:,.0f}".replace(",", " "), ha="right", va="center", fontsize=8, color=RED)
    titulo(ax, "Las unidades originales difieren mucho", 9)
    style(ax, "x")
    ax = axs[1]
    nombres = [o.replace("_", " ") for o in x.columns]
    pos = np.arange(len(nombres))
    ax.bar(pos - 0.2, cuota.to_numpy() * 100, 0.4, color=RED, label="sin estandarizar")
    ax.bar(pos + 0.2, np.full(len(nombres), 100 / len(nombres)), 0.4, color=C39, label="estandarizadas")
    ax.set_xticks(pos)
    ax.set_xticklabels(nombres, rotation=60, ha="right", fontsize=7.4)
    ax.set(ylabel="% de la varianza total")
    ax.legend(fontsize=7.8, frameon=False)
    titulo(ax, f"Prolina aporta el {cuota['prolina'] * 100:.1f} % de las distancias", 9)
    style(ax, "y")
    ax = axs[2]
    pca = PCA(2).fit(z)
    pc = pca.transform(z)
    ax.bar(["sin estandarizar", "estandarizadas"], [ari_raw, ari_z], color=[RED, C39], width=0.5)
    for i, v in enumerate([ari_raw, ari_z]):
        ax.text(i, v + 0.02, f"ARI {v:.3f}", ha="center", fontsize=9)
    ax.set(ylabel="índice de Rand ajustado con la clase real", ylim=(0, 1.1))
    titulo(ax, "K-means con k = 3: mismo algoritmo, distinta escala", 9)
    style(ax, "y")
    ss = StandardScaler().fit(x)
    k_ss = kmeans(ss.transform(x), 3)
    NUM["estandarizar"] = dict(var=var.round(4).to_dict(), cuota_prolina=float(cuota["prolina"]), cuota_magnesio=float(cuota["magnesio"]),
                               cuota_resto=float(1 - cuota["prolina"] - cuota["magnesio"]), ari_raw=float(ari_raw), ari_z=float(ari_z),
                               wss_raw=float(k_raw.inertia_), tamanos_raw=[int(c) for c in np.bincount(k_raw.labels_)],
                               ari_entre_raw_y_z=float(adjusted_rand_score(k_raw.labels_, k_z.labels_)),
                               wss_z_ddof1=float(k_z.inertia_), wss_z_ddof0=float(k_ss.inertia_), ari_ddof0_ddof1=float(adjusted_rand_score(k_z.labels_, k_ss.labels_)),
                               razon_ddof=float(177 / 176), ej_z=dict(x=14, media=13, sd=0.8, z=(14 - 13) / 0.8),
                               media_cero=bool(np.allclose(z.mean(), 0, atol=1e-10)), sd_uno=bool(np.allclose(z.std(ddof=1), 1)),
                               silueta_raw=float(silhouette_score(x, k_raw.labels_)), silueta_z=float(silhouette_score(z, k_z.labels_)))
    fig.tight_layout()
    save(fig, "c39_estandarizar.png")


# ----------------------------------------------------------------------------- 4. codo
def codo_fig():
    d = datos()
    Z = d["z"].to_numpy()
    ks = list(range(1, 9))
    W = [kmeans(Z, k).inertia_ for k in ks]
    red = [W[i - 1] - W[i] for i in range(1, len(W))]
    pct = [100 * (W[i - 1] - W[i]) / W[i - 1] for i in range(1, len(W))]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.plot(ks, W, "o-", color=C39)
    ax.scatter([3], [W[2]], s=110, facecolors="none", edgecolors=RED, lw=1.8, zorder=4)
    ax.annotate(f"k = 3: {W[2]:,.2f}".replace(",", " "), (3, W[2]), xytext=(4.0, 1850), fontsize=8.6, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlabel="número de grupos k", ylabel="WSS")
    titulo(ax, "Codo: WSS según k (13 variables estandarizadas)", 9)
    style(ax)
    ax = axs[1]
    ax.bar([f"{i}→{i + 1}" for i in range(1, 8)], red, color=[RED if i == 1 else OK for i in range(7)])
    for i, (r_, p_) in enumerate(zip(red, pct)):
        ax.text(i, r_ + 12, f"{r_:.0f}\n({p_:.1f} %)", ha="center", fontsize=7.6)
    ax.set(xlabel="paso de k a k + 1", ylabel="reducción de WSS", ylim=(0, max(red) * 1.25))
    titulo(ax, "Cuánto baja el WSS en cada paso", 9)
    style(ax, "y")
    ax = axs[2]
    k20 = list(range(1, 21))
    W20 = [kmeans(Z, k, n_init=10).inertia_ for k in k20]
    ax.plot(k20, W20, "o-", color=C39, ms=4)
    ax.axvline(3, color=RED, ls="--", lw=1)
    ax.text(3.4, W20[0] * 0.8, "k = 3", color=RED, fontsize=8.4)
    ax.set(xlabel="número de grupos k (hasta 20)", ylabel="WSS")
    titulo(ax, "Con k hasta 20 el WSS sigue bajando", 9)
    style(ax)
    n = len(Z)
    NUM["codo"] = dict(k=ks, wss=[float(w) for w in W], reduccion=[float(r) for r in red], pct=[float(p) for p in pct], k20=k20,
                       wss20=[float(w) for w in W20], wss_k_igual_n=0.0, n=int(n), wss_k3=float(W[2]), tss=float(W[0]))
    fig.tight_layout()
    save(fig, "c39_codo.png")


# ----------------------------------------------------------------------------- 5. silueta
def silueta_fig():
    d = datos()
    Z = d["z"].to_numpy()
    km, grupo = ajuste_k3()
    sv = silhouette_samples(Z, km.labels_)
    ks = list(range(2, 9))
    S = [silhouette_score(Z, kmeans(Z, k).labels_) for k in ks]
    # a(i) y b(i) a mano
    D = squareform(pdist(Z))
    a = np.zeros(len(Z))
    b = np.zeros(len(Z))
    for i in range(len(Z)):
        mi = km.labels_ == km.labels_[i]
        a[i] = D[i, mi].sum() / (mi.sum() - 1)
        b[i] = min(D[i, km.labels_ == g].mean() for g in range(3) if g != km.labels_[i])
    s_manual = (b - a) / np.maximum(a, b)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    y0 = 0
    for g in range(3):
        v = np.sort(sv[km.labels_ == g])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=G3[g], alpha=0.85, lw=0)
        ax.text(-0.06, y0 + len(v) / 2, f"grupo {g + 1}\n(n = {len(v)})", fontsize=7.6, ha="right", va="center")
        y0 += len(v) + 6
    ax.axvline(sv.mean(), color=RED, ls="--", lw=1.2)
    ax.text(sv.mean() + 0.01, y0 - 14, f"media {sv.mean():.3f}", color=RED, fontsize=8.2)
    ax.set(xlabel="silueta s(i)", yticks=[], xlim=(-0.35, 0.8))
    ax.spines["left"].set_visible(False)
    titulo(ax, "Gráfico de silueta con k = 3", 9)
    style(ax, "x")
    ax = axs[1]
    ax.plot(ks, S, "o-", color=AMBER)
    ax.scatter([3], [S[1]], s=110, c=GREEN if False else TEAL, zorder=4)
    ax.axvline(3, color=MUTED, ls="--", lw=1)
    for kk, ss_ in zip(ks, S):
        ax.text(kk, ss_ + 0.006, f"{ss_:.3f}", ha="center", fontsize=7.8)
    ax.set(xlabel="número de grupos k", ylabel="silueta media", ylim=(0.1, 0.32))
    titulo(ax, "Silueta media: el máximo está en k = 3", 9)
    style(ax)
    ax = axs[2]
    for g in range(3):
        m = km.labels_ == g
        ax.scatter(a[m], b[m], s=12, c=G3[g], alpha=0.7, label=f"grupo {g + 1}")
    lim = max(a.max(), b.max()) * 1.03
    ax.plot([0, lim], [0, lim], color=INK, lw=1)
    neg = s_manual < 0
    ax.scatter(a[neg], b[neg], s=60, facecolors="none", edgecolors=RED, lw=1.4, label=f"s < 0 ({int(neg.sum())})")
    ax.set(xlabel="a(i): distancia media a su grupo", ylabel="b(i): distancia media al grupo vecino", xlim=(1.5, lim), ylim=(1.5, lim))
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    titulo(ax, "Bajo la diagonal: b < a, silueta negativa", 9)
    style(ax)
    por_g = [float(sv[km.labels_ == g].mean()) for g in range(3)]
    NUM["silueta"] = dict(k=ks, media=[float(s) for s in S], k_max=int(ks[int(np.argmax(S))]), media_k3=float(sv.mean()), por_grupo=por_g,
                          n_negativas=int(neg.sum()), negativas_por_grupo=[int((neg & (km.labels_ == g)).sum()) for g in range(3)],
                          manual_igual=bool(np.allclose(s_manual, sv)), s_min=float(sv.min()), s_max=float(sv.max()),
                          ej=dict(a=2, b=6, s=(6 - 2) / 6), idx_peor=int(sv.argmin()), a_peor=float(a[sv.argmin()]), b_peor=float(b[sv.argmin()]),
                          s_peor=float(sv.min()), idx_mejor=int(sv.argmax()), a_mejor=float(a[sv.argmax()]), b_mejor=float(b[sv.argmax()]),
                          codigo_peor=str(d["v"].codigo[sv.argmin()]), codigo_mejor=str(d["v"].codigo[sv.argmax()]))
    fig.tight_layout()
    save(fig, "c39_silueta.png")


GREEN = "#059669"


# ----------------------------------------------------------------------------- 6. gap
def gap_fig():
    d = datos()
    Z = d["z"].to_numpy()
    gap, err, kg, ref, wss = gap_stat(Z)
    ks = np.arange(1, 9)
    # sensibilidad a la simulación: otras semillas y otro B
    otros = {}
    for s in (1, 2, 3, 4):
        g2, e2, k2, _, _ = gap_stat(Z, seed=s)
        otros[str(s)] = int(k2)
    g60, e60, k60, _, _ = gap_stat(Z, B=60, seed=RS)
    # regla en el ejemplo hipotético de la diapositiva 23
    gh = np.array([0.40, 0.62, 0.64])
    sh = np.array([0.04, 0.05, 0.04])
    umbrales = gh[1:] - sh[1:]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    ax.plot(ks, np.log(wss), "o-", color=RED, label="log W observado")
    ax.plot(ks, ref.mean(0), "s--", color=C39, label="media de log W de las 30 referencias")
    ax.fill_between(ks, np.log(wss), ref.mean(0), color=C39, alpha=0.12)
    ax.text(5.2, (np.log(wss)[4] + ref.mean(0)[4]) / 2, "Gap", color=C39, fontsize=9)
    ax.set(xlabel="k", ylabel="log WSS")
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    titulo(ax, "Gap(k) = promedio log W* − log W", 9)
    style(ax)
    ax = axs[1]
    ax.errorbar(ks, gap, yerr=err, fmt="o-", capsize=4, color=C39)
    ax.axvline(kg, color=AMBER, ls="--", lw=1.2)
    ax.text(kg + 0.1, gap.min() + 0.02, f"regla de 1 error estándar: k = {kg}", color=AMBER, fontsize=8.2)
    ax.scatter([int(np.argmax(gap)) + 1], [gap.max()], s=90, facecolors="none", edgecolors=RED, lw=1.6)
    ax.text(int(np.argmax(gap)) + 1 - 0.2, gap.max() + 0.03, f"máximo: k = {int(np.argmax(gap)) + 1}", color=RED, fontsize=8, ha="right")
    ax.set(xlabel="número de grupos k", ylabel="Gap", ylim=(gap.min() - 0.05, gap.max() + 0.1))
    titulo(ax, "Gap con B = 30 y la regla Gap(k) ≥ Gap(k+1) − s(k+1)", 8.6)
    style(ax)
    ax = axs[2]
    ax.bar([1, 2, 3], gh, color=OK, width=0.5, label="Gap(k)")
    ax.errorbar([1, 2, 3], gh, yerr=sh, fmt="none", ecolor=INK, capsize=4)
    ax.scatter([1, 2], umbrales, marker="_", s=900, c=RED, zorder=4, label="Gap(k+1) − s(k+1)")
    ax.text(1, 0.74, "0.40 < 0.57:\nno cumple", ha="center", fontsize=8)
    ax.text(2, 0.76, "0.62 ≥ 0.60:\nelige k = 2", ha="center", fontsize=8, color=GREEN)
    ax.set(xticks=[1, 2, 3], xlabel="k (ejemplo hipotético, diapositiva 23)", ylabel="Gap", ylim=(0.2, 1.0))
    ax.legend(fontsize=7.6, frameon=False, loc="upper center", ncol=2)
    titulo(ax, "La regla puede elegir un k menor que el máximo", 9)
    style(ax, "y")
    NUM["gap"] = dict(k=[int(k) for k in ks], gap=[float(g) for g in gap], err=[float(e) for e in err], k_gap=int(kg), k_max_gap=int(np.argmax(gap)) + 1,
                      B=30, k_otras_semillas=otros, k_B60=int(k60), gap_B60=[float(g) for g in g60], umbral_k3=float(gap[3] - err[3]),
                      gap_k3=float(gap[2]), gap_k4=float(gap[3]), err_k4=float(err[3]), umbral_k2=float(gap[2] - err[2]), gap_k2=float(gap[1]),
                      hip=dict(gap=[float(g) for g in gh], s=[float(s) for s in sh], umbrales=[float(u) for u in umbrales], k=2),
                      logw=[float(v) for v in np.log(wss)], logw_ref=[float(v) for v in ref.mean(0)])
    fig.tight_layout()
    save(fig, "c39_gap.png")


# ----------------------------------------------------------------------------- 7. perfiles de los grupos
def perfiles_fig():
    d = datos()
    x, z, y, v = d["x"], d["z"], d["y"], d["v"]
    km, grupo = ajuste_k3()
    cent = pd.DataFrame(km.cluster_centers_, columns=z.columns, index=[1, 2, 3])
    med = x.groupby(grupo).mean()
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1.5, 0.9, 1.1]})
    ax = axs[0]
    M = cent.T.to_numpy()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_yticks(range(13))
    ax.set_yticklabels([c.replace("_", " ") for c in cent.columns], fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Grupo {g}\n(n = {int((grupo == g).sum())})" for g in (1, 2, 3)], fontsize=8.4)
    for i in range(13):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.8, color="white" if abs(M[i, j]) > 0.9 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="media estandarizada")
    titulo(ax, "Perfiles estandarizados (diapositiva 29)", 9)
    ax = axs[1]
    bp = ax.boxplot([x.alcohol[grupo == g] for g in (1, 2, 3)], patch_artist=True, widths=0.55, medianprops=dict(color="white", lw=2))
    for b_, c_ in zip(bp["boxes"], G3):
        b_.set(facecolor=c_, edgecolor=c_)
    ax.set(xticklabels=["1", "2", "3"], xlabel="grupo", ylabel="alcohol")
    titulo(ax, "Alcohol por grupo", 9)
    style(ax, "y")
    ax = axs[2]
    for g in (1, 2, 3):
        m = grupo == g
        ax.scatter(x.alcohol[m], x.flavonoides[m], s=16, c=G3[g - 1], alpha=0.75, label=f"Grupo {g}")
    ax.set(xlabel="alcohol", ylabel="flavonoides")
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(ax, "Dos de las 13 variables", 9)
    style(ax)
    sel = ["alcohol", "flavonoides", "color", "matiz", "prolina"]
    NUM["perfiles"] = dict(n=[int((grupo == g).sum()) for g in (1, 2, 3)], centros={str(g): {c: float(cent.loc[g, c]) for c in cent.columns} for g in (1, 2, 3)},
                           medias={str(g): {c: float(med.loc[g, c]) for c in x.columns} for g in (1, 2, 3)},
                           global_media={c: float(x[c].mean()) for c in x.columns}, sel=sel,
                           mayor_por_grupo={str(g): [c for c in cent.columns if cent.loc[g, c] == cent.loc[:, c].max()] for g in (1, 2, 3)},
                           menor_por_grupo={str(g): [c for c in cent.columns if cent.loc[g, c] == cent.loc[:, c].min()] for g in (1, 2, 3)},
                           mapeo_R_a_python={"1": 3, "2": 2, "3": 1}, n_R=[65, 50, 62])
    fig.tight_layout()
    save(fig, "c39_perfiles.png")


# ----------------------------------------------------------------------------- 8. PCA, clases reales y tabla de contingencia
def pca_fig():
    d = datos()
    z, y, v = d["z"], d["y"], d["v"]
    km, grupo = ajuste_k3()
    tabla = pd.crosstab(pd.Series(y, name="Clase real"), pd.Series(grupo, name="Grupo"))
    clase_del_grupo = tabla.idxmax(axis=0)
    asign = pd.Series(grupo).map(clase_del_grupo).to_numpy()
    discrepa = asign != y
    pca = PCA(2).fit(z)
    co = pca.transform(z)
    var = 100 * pca.explained_variance_ratio_
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, ttl, cols in [(axs[0], "Clases reales de los vinos", y), (axs[1], "Grupos encontrados por K-means", asign)]:
        for c in (1, 2, 3):
            m = cols == c
            nm = f"Clase {c}" if ax is axs[0] else f"Grupo {', '.join(str(g) for g in clase_del_grupo.index[clase_del_grupo == c])} (clase {c})"
            ax.scatter(co[m, 0], co[m, 1], c=PAL[c - 1], s=22, label=nm)
        ax.set(xlabel=f"PC1 ({var[0]:.1f} % de varianza)", ylabel=f"PC2 ({var[1]:.1f} % de varianza)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.4, frameon=False)
        style(ax)
    axs[1].scatter(co[discrepa, 0], co[discrepa, 1], facecolors="none", edgecolors="black", s=130, lw=1.5)
    ax = axs[2]
    T = tabla.to_numpy()
    ax.imshow(T, cmap="Blues", vmin=0, vmax=75)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Grupo {g}" for g in tabla.columns], fontsize=8.4)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Clase {c}" for c in tabla.index], fontsize=8.4)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(T[i, j]), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if T[i, j] > 40 else INK)
    ax.grid(False)
    ax.set_title(f"Coinciden {int((~discrepa).sum())} de {len(y)} vinos", loc="left", fontsize=9.2, pad=26)
    fig.suptitle(f"Pureza {100 * np.mean(~discrepa):.1f} % · ARI {adjusted_rand_score(y, grupo):.3f} · círculos negros: {int(discrepa.sum())} discrepancias", y=1.0, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # K-means sobre las dos componentes
    k2 = kmeans(co, 3)
    tabla2 = pd.crosstab(y, k2.labels_)
    pur2 = float(tabla2.max(axis=0).sum() / len(y))
    pca_all = PCA().fit(z)
    NUM["pca"] = dict(var=[float(a) for a in pca_all.explained_variance_ratio_], acum2=float(pca_all.explained_variance_ratio_[:2].sum()),
                      acum=[float(a) for a in np.cumsum(pca_all.explained_variance_ratio_)],
                      tabla=[[int(a) for a in fila] for fila in T], clase_del_grupo={str(a): int(b) for a, b in clase_del_grupo.items()},
                      coincidencias=int((~discrepa).sum()), discrepancias=int(discrepa.sum()), pureza=float(np.mean(~discrepa)),
                      ari=float(adjusted_rand_score(y, grupo)), ids_discrepancia=[str(c) for c in v.codigo[discrepa]],
                      clases_discrepancia=[int(c) for c in y[discrepa]], grupo_discrepancia=[int(g) for g in grupo[discrepa]],
                      ari_2pc=float(adjusted_rand_score(y, k2.labels_)), pureza_2pc=pur2, discrep_2pc=int(len(y) - tabla2.max(axis=0).sum()),
                      acuerdo_13_vs_2pc=float(adjusted_rand_score(grupo, k2.labels_)), wss_2pc=float(k2.inertia_),
                      clases_n=[int((y == c).sum()) for c in (1, 2, 3)], n_iter=int(km.n_iter_))
    save(fig, "c39_pca.png")


# ----------------------------------------------------------------------------- 9. validación externa: ARI, silueta de los errores
def validar_fig():
    d = datos()
    z, y = d["z"].to_numpy(), d["y"]
    km, grupo = ajuste_k3()
    sv = silhouette_samples(z, km.labels_)
    tabla = pd.crosstab(y, grupo)
    asign = pd.Series(grupo).map(tabla.idxmax(axis=0)).to_numpy()
    discrepa = asign != y
    ks = list(range(2, 9))
    ari, pur, sil = [], [], []
    for k in ks:
        kk = kmeans(z, k)
        ari.append(adjusted_rand_score(y, kk.labels_))
        t = pd.crosstab(y, kk.labels_)
        pur.append(float(t.max(axis=0).sum() / len(y)))
        sil.append(silhouette_score(z, kk.labels_))
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.plot(ks, ari, "o-", color=C39, label="ARI con la clase real")
    ax.plot(ks, sil, "s--", color=AMBER, label="silueta media (sin etiquetas)")
    ax.axvline(3, color=MUTED, ls=":")
    ax.set(xlabel="k", ylabel="valor", ylim=(0, 1.0))
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    titulo(ax, "ARI vs k: solo se puede calcular si hay etiquetas", 9)
    style(ax)
    ax = axs[1]
    ax.hist(sv[~discrepa], bins=np.linspace(-0.2, 0.6, 17), color=OK, alpha=0.8, label=f"coinciden ({int((~discrepa).sum())})")
    ax.hist(sv[discrepa], bins=np.linspace(-0.2, 0.6, 17), color=RED, alpha=0.9, label=f"discrepan ({int(discrepa.sum())})")
    ax.set(xlabel="silueta s(i)", ylabel="vinos")
    ax.legend(fontsize=7.8, frameon=False)
    titulo(ax, "Los vinos «mal agrupados» tienen silueta baja", 9)
    style(ax, "y")
    ax = axs[2]
    pares = [("13 variables\nestandarizadas", adjusted_rand_score(y, grupo)),
             ("2 componentes\nprincipales", NUM["pca"]["ari_2pc"] if "pca" in NUM else np.nan),
             ("13 variables\nsin estandarizar", adjusted_rand_score(y, kmeans(d["x"].to_numpy(), 3).labels_))]
    ax.bar([p[0] for p in pares], [p[1] for p in pares], color=[C39, VIOLET, RED], width=0.55)
    for i, p in enumerate(pares):
        ax.text(i, p[1] + 0.02, f"{p[1]:.3f}", ha="center", fontsize=9)
    ax.set(ylabel="ARI con la clase real", ylim=(0, 1.1))
    ax.tick_params(axis="x", labelsize=8)
    titulo(ax, "¿Dónde agrupar? Escala y componentes", 9)
    style(ax, "y")
    NUM["validar"] = dict(k=ks, ari=[float(a) for a in ari], pureza=[float(p) for p in pur], sil=[float(s) for s in sil],
                          sil_media_discrepan=float(sv[discrepa].mean()), sil_media_coinciden=float(sv[~discrepa].mean()),
                          sil_discrepan=[float(s) for s in sv[discrepa]], n_discrepan_sil_neg=int((sv[discrepa] < 0).sum()),
                          ari_13=float(pares[0][1]), ari_2pc=float(pares[1][1]), ari_raw=float(pares[2][1]))
    fig.tight_layout()
    save(fig, "c39_validar.png")


# ----------------------------------------------------------------------------- 10. inicialización
def init_fig():
    d = datos()
    Z = d["z"].to_numpy()
    w_rand = np.array([KMeans(3, init="random", n_init=1, max_iter=100, algorithm="lloyd", tol=1e-8, random_state=s).fit(Z).inertia_ for s in range(25)])
    w_pp = np.array([KMeans(3, init="k-means++", n_init=1, max_iter=100, algorithm="lloyd", tol=1e-8, random_state=s).fit(Z).inertia_ for s in range(25)])
    mejor50 = kmeans(Z, 3).inertia_
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.scatter(range(1, 26), w_rand, s=34, c=OK, label="inicio aleatorio, 1 ajuste")
    ax.axhline(mejor50, color=RED, ls="--", lw=1.2, label=f"mejor de 50 inicios: {mejor50:.2f}")
    ax.set(xlabel="inicialización", ylabel="WSS final", ylim=(1240, w_rand.max() * 1.03))
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    titulo(ax, "Diapositiva 15: mismos datos y k = 3", 9)
    style(ax)
    ax = axs[1]
    vals = np.round(np.concatenate([w_rand, w_pp]), 1)
    uniq = np.unique(vals)
    pos = np.arange(len(uniq))
    cr = [int((np.round(w_rand, 1) == u).sum()) for u in uniq]
    cp = [int((np.round(w_pp, 1) == u).sum()) for u in uniq]
    ax.bar(pos - 0.2, cr, 0.4, color=OK, label="init = random")
    ax.bar(pos + 0.2, cp, 0.4, color=C39, label="init = k-means++")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{u:.1f}" for u in uniq], fontsize=7.4)
    ax.set(xlabel="WSS final (mínimo local)", ylabel="de 25 ajustes")
    ax.legend(fontsize=7.8, frameon=False)
    titulo(ax, "Cada inicio cae en un mínimo local", 9)
    style(ax, "y")
    ax = axs[2]
    ninit = [1, 2, 5, 10, 20, 50]
    rng = np.random.default_rng(RS)
    sim = []
    for m in ninit:
        sim.append([w_rand_boot.min() for w_rand_boot in (rng.choice(w_rand, m, replace=True) for _ in range(300))])
    bp = ax.boxplot(sim, patch_artist=True, widths=0.55, medianprops=dict(color="white", lw=2), flierprops=dict(markersize=3))
    for b_ in bp["boxes"]:
        b_.set(facecolor=C39, edgecolor=C39)
    ax.set_xticklabels([str(m) for m in ninit])
    ax.axhline(mejor50, color=RED, ls="--", lw=1)
    ax.set(xlabel="n_init (número de inicios por ajuste)", ylabel="WSS conservado", ylim=(1265, 1285))
    ax.text(0.6, 1283.5, "el ajuste de 1 604 queda fuera del gráfico", fontsize=7.6, color=MUTED)
    titulo(ax, "Conservar el mejor de varios inicios", 9)
    style(ax, "y")
    NUM["init"] = dict(wss_random=[float(w) for w in w_rand], min_random=float(w_rand.min()), max_random=float(w_rand.max()),
                       n_unicos_random=int(len(np.unique(np.round(w_rand, 2)))), n_optimo_random=int((np.round(w_rand, 2) == round(mejor50, 2)).sum()),
                       min_pp=float(w_pp.min()), max_pp=float(w_pp.max()), n_unicos_pp=int(len(np.unique(np.round(w_pp, 2)))),
                       n_optimo_pp=int((np.round(w_pp, 2) == round(mejor50, 2)).sum()), mejor50=float(mejor50),
                       dif_pct_peor=float(100 * (w_rand.max() / mejor50 - 1)), ninit=ninit, mediana_por_ninit=[float(np.median(s)) for s in sim])
    fig.tight_layout()
    save(fig, "c39_inicializacion.png")


# ----------------------------------------------------------------------------- 11. jerárquico (complemento)
def jerarquico_fig():
    d = datos()
    z, y = d["z"].to_numpy(), d["y"]
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [1.35, 1.0, 0.9]})
    aris, sils, tam, cor = {}, {}, {}, {}
    Ls = {}
    for met in ("single", "complete", "average", "ward"):
        L = linkage(z, method=met)
        lab = fcluster(L, 3, criterion="maxclust")
        Ls[met] = L
        aris[met] = float(adjusted_rand_score(y, lab))
        sils[met] = float(silhouette_score(z, lab))
        tam[met] = sorted((int(c) for c in np.bincount(lab)[1:]), reverse=True)
        cor[met] = float(cophenet(L, pdist(z))[0])
    L = Ls["ward"]
    corte = (L[-3, 2] + L[-2, 2]) / 2
    ax = axs[0]
    dendrogram(L, truncate_mode="lastp", p=30, ax=ax, color_threshold=corte, above_threshold_color=MUTED, no_labels=True, leaf_rotation=90)
    ax.axhline(corte, color=RED, ls="--", lw=1)
    ax.text(0.5, corte * 1.03, "corte en 3 grupos", color=RED, fontsize=8)
    ax.set(ylabel="distancia de fusión (Ward)", xlabel="30 ramas finales")
    titulo(ax, "Dendrograma con enlace de Ward", 9)
    style(ax, "y")
    ax = axs[1]
    names = ["single", "complete", "average", "ward"]
    km_ari = adjusted_rand_score(y, ajuste_k3()[1])
    vals = [aris[m] for m in names] + [km_ari]
    ax.bar(["single", "complete", "average", "Ward", "K-means"], vals, color=[RED, AMBER, AMBER, C39, VIOLET], width=0.6)
    for i, vv in enumerate(vals):
        ax.text(i, max(vv, 0) + 0.02, f"{vv:.2f}", ha="center", fontsize=8.4)
        if names[i if i < 4 else 3] and i < 4:
            ax.text(i, -0.1, str(tam[names[i]]), ha="center", fontsize=6.6, color=MUTED)
    ax.set(ylabel="ARI con la clase real (k = 3)", ylim=(-0.15, 1.05))
    ax.tick_params(axis="x", labelsize=8)
    titulo(ax, "Tamaños bajo cada barra: single encadena", 8.8)
    style(ax, "y")
    ax = axs[2]
    dendrogram(Ls["single"], truncate_mode="lastp", p=30, ax=ax, color_threshold=0, above_threshold_color=RED, no_labels=True)
    ax.set(ylabel="distancia de fusión (single)", xlabel="30 ramas finales")
    titulo(ax, "Single: efecto cadena (173 + 3 + 1)", 9)
    style(ax, "y")
    NUM["jerarq"] = dict(ari=aris, silueta=sils, tamanos=tam, cofenetica=cor, ari_kmeans=float(km_ari), altura_corte3=float(corte),
                         altura_ward_final=float(L[-1, 2]))
    fig.tight_layout()
    save(fig, "c39_jerarquico.png")
    # ejemplo pequeño para los ejercicios (cinco valores en una recta)
    xs = np.array([0.0, 1.0, 2.5, 4.1, 5.8])
    nombres = list("ABCDE")
    Dm = squareform(pdist(xs[:, None]))
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.2))
    res = {}
    for ax, met, ttl in zip(axs, ("single", "complete", "ward"), ("enlace simple (single)", "enlace completo (complete)", "enlace de Ward")):
        Lt = linkage(xs[:, None], method=met)
        dendrogram(Lt, labels=nombres, ax=ax, color_threshold=0, above_threshold_color=C39)
        ax.set(ylabel="altura de fusión")
        titulo(ax, ttl, 9.4)
        style(ax, "y")
        res[met] = [[int(a), int(b), round(float(h), 4), int(n)] for a, b, h, n in Lt]
    fig.tight_layout()
    save(fig, "c39_jerarquico_toy.png")
    NUM["jerarq_toy"] = dict(x=[float(a) for a in xs], D=[[float(a) for a in fila] for fila in Dm], merges=res)


# ----------------------------------------------------------------------------- 12. K-modes
def kmodes_fig():
    from kmodes.kmodes import KModes
    d = datos()
    cat = d["cat"]
    ks = list(range(1, 9))
    costos = []
    for k in ks:
        m = KModes(n_clusters=k, init="Huang", n_init=20, max_iter=100, random_state=RS).fit(cat)
        costos.append(float(m.cost_))
    m4 = KModes(n_clusters=4, init="Huang", n_init=20, max_iter=100, random_state=RS).fit(cat)
    g4 = m4.labels_ + 1
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axs[0]
    ax.plot(ks, costos, "o-", color=C39)
    ax.scatter([4], [costos[3]], s=110, facecolors="none", edgecolors=RED, lw=1.8, zorder=4)
    ax.annotate(f"k = 4: {costos[3]:.0f}", (4, costos[3]), xytext=(4.8, 190), fontsize=8.6, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlabel="número de grupos k", ylabel="total de desacuerdos")
    titulo(ax, "Costo de K-modes: sin codo claro", 9)
    style(ax)
    ax = axs[1]
    cen = m4.cluster_centroids_
    cols = ["alcohol", "flavonoides", "color"]
    val = {"bajo": 0, "medio": 1, "alto": 2}
    M = np.array([[val[c] for c in fila] for fila in cen])
    ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=2.3, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(cols, fontsize=8.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Grupo {g} (n = {int((g4 == g).sum())})" for g in range(1, 5)], fontsize=8.2)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, cen[i, j], ha="center", va="center", fontsize=9.4, color="white" if M[i, j] == 2 else INK)
    ax.grid(False)
    titulo(ax, "Modas por grupo (centros de K-modes, k = 4)", 9)
    ax = axs[2]
    g2 = np.argmax([((g4 == g) & (cat.color == "alto")).sum() for g in range(1, 5)]) + 1
    cnt = cat.color[g4 == g2].value_counts().reindex(CAT_LAB[::-1]).fillna(0).astype(int)
    ax.bar(cnt.index, cnt.values, color=[C39, OK, MUTED])
    for i, (c_, vv) in enumerate(cnt.items()):
        ax.text(i, vv + 0.8, f"{vv}/{int(cnt.sum())}\n{100 * vv / cnt.sum():.1f} %", ha="center", fontsize=8.2)
    ax.set(ylabel="vinos del grupo", ylim=(0, cnt.max() * 1.3), xlabel=f"color en el grupo {g2} de K-modes")
    titulo(ax, "Un perfil resume frecuencias, no cada vino", 9)
    style(ax, "y")
    # ejemplo de disimilitud de las diapositivas 33 y 34
    cliente = ("Gold", "App", "Sur")
    A, B = ("Gold", "App", "Norte"), ("Silver", "Web", "Sur")
    dis = lambda a, b: sum(u != w for u, w in zip(a, b))
    filas = [("Gold", "App", "Norte"), ("Gold", "Web", "Norte"), ("Silver", "App", "Sur")]
    moda = tuple(max(set(c), key=list(c).count) for c in zip(*filas))
    NUM["kmodes"] = dict(k=ks, costo=costos, k4=4, costo_k4=costos[3], n_k4=[int((g4 == g).sum()) for g in range(1, 5)], centros_k4=[[str(c) for c in fila] for fila in cen],
                         g_color_alto=int(g2), n_g=int(cnt.sum()), color_g={c: int(vv) for c, vv in cnt.items()},
                         cats={c: {k_: int(v_) for k_, v_ in cat[c].value_counts().items()} for c in cat.columns},
                         dis_cliente_A=int(dis(cliente, A)), dis_cliente_B=int(dis(cliente, B)), moda_ej=list(moda),
                         costo_moda=int(sum(dis(f_, moda) for f_ in filas)), costos_por_cliente=[int(dis(f_, moda)) for f_ in filas])
    fig.tight_layout()
    save(fig, "c39_kmodes.png")


# ----------------------------------------------------------------------------- 13. K-prototypes
def kproto_fig():
    from kmodes.kprototypes import KPrototypes
    d = datos()
    z, cat = d["z"], d["cat"]
    mix = pd.concat([z[["ceniza", "matiz"]], cat], axis=1)
    ks = list(range(1, 9))
    costos = []
    for k in ks:
        m = KPrototypes(n_clusters=k, init="Huang", n_init=20, gamma=1, max_iter=100, random_state=RS).fit(mix.to_numpy(), categorical=[2, 3, 4])
        costos.append(float(m.cost_))
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.7))
    ax = axs[0]
    ax.plot(ks, costos, "o-", color=C39)
    ax.scatter([3], [costos[2]], s=110, facecolors="none", edgecolors=RED, lw=1.8, zorder=4)
    ax.annotate(f"k = 3: {costos[2]:.2f}", (3, costos[2]), xytext=(4.2, 520), fontsize=8.4, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlabel="número de grupos k", ylabel="costo mixto (γ = 1)")
    titulo(ax, "Costo de K-prototypes", 9)
    style(ax)
    res = {}
    labs = {}
    for ax, gam in zip(axs[1:], (0.25, 1, 4)):
        m = KPrototypes(n_clusters=3, init="Huang", n_init=20, gamma=gam, max_iter=100, random_state=RS).fit(mix.to_numpy(), categorical=[2, 3, 4])
        lab = m.labels_
        # numeración alineada por el centro de matiz, como la diapositiva 50
        orden = np.argsort([z.matiz[lab == g].mean() for g in range(3)])
        remap = {int(o): i for i, o in enumerate(orden)}
        lab2 = np.array([remap[int(a)] for a in lab])
        labs[gam] = lab2
        for g in range(3):
            ax.scatter(z.ceniza[lab2 == g], z.matiz[lab2 == g], s=14, c=G3[g], alpha=0.75)
        ax.set(xlabel="ceniza estandarizada", ylabel="matiz estandarizado")
        titulo(ax, f"γ = {gam:g}: grupos de {sorted(int(c) for c in np.bincount(lab))[::-1]}", 8.8)
        style(ax)
        res[f"{gam:g}"] = dict(costo=float(m.cost_), tamanos=[int(c) for c in np.bincount(lab)], n_iter=int(m.n_iter_))
    m1 = KPrototypes(n_clusters=3, init="Huang", n_init=20, gamma=1, max_iter=100, random_state=RS).fit(mix.to_numpy(), categorical=[2, 3, 4])
    cn = m1.cluster_centroids_
    NUM["kproto"] = dict(k=ks, costo=costos, k3=3, costo_k3=costos[2], tamanos_k3=[int(c) for c in np.bincount(m1.labels_)],
                         centros_k3=[[(float(a) if i < 2 else str(a)) for i, a in enumerate(fila)] for fila in cn], por_gamma=res,
                         ari_025_vs_1=float(adjusted_rand_score(labs[0.25], labs[1])), ari_1_vs_4=float(adjusted_rand_score(labs[1], labs[4])),
                         ari_025_vs_4=float(adjusted_rand_score(labs[0.25], labs[4])),
                         ej_costo_A_025=1 + 0.25, ej_costo_A_05=1 + 0.5, ej_costo_A_3=1 + 3, ej_costo_B=2,
                         n_color_alto_g2=30)
    fig.tight_layout()
    save(fig, "c39_kprototypes.png")


# ----------------------------------------------------------------------------- 14. limitaciones de K-means
def limites_fig():
    Xm, ym = make_moons(300, noise=0.06, random_state=RS)
    km = KMeans(2, n_init=10, random_state=RS).fit(Xm)
    db = DBSCAN(eps=0.2, min_samples=5).fit(Xm)
    ari_km, ari_db = adjusted_rand_score(ym, km.labels_), adjusted_rand_score(ym, db.labels_)
    Xb, yb = make_blobs([80, 80, 80], centers=[[0, 0], [4, 0], [2, 3.4]], cluster_std=0.8, random_state=RS)
    X_out = np.vstack([Xb, [[30.0, 30.0]]])
    k_sin = KMeans(3, n_init=10, random_state=RS).fit(Xb)
    k_con = KMeans(3, n_init=10, random_state=RS).fit(X_out)
    rng = np.random.default_rng(RS)
    Xd = np.vstack([rng.normal([0, 0], 2.2, (300, 2)), rng.normal([6, 0], 0.4, (40, 2))])
    yd = np.r_[np.zeros(300, int), np.ones(40, int)]
    kd = KMeans(2, n_init=10, random_state=RS).fit(Xd)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axs[0]
    ax.scatter(Xm[:, 0], Xm[:, 1], c=[G3[l] for l in km.labels_], s=12)
    ax.scatter(*km.cluster_centers_.T, marker="X", s=100, c=INK, zorder=4)
    ax.set(xlabel="x1", ylabel="x2")
    titulo(ax, f"Formas curvas: K-means ARI {ari_km:.2f} · DBSCAN ARI {ari_db:.2f}", 8.8)
    style(ax)
    ax = axs[1]
    ax.scatter(X_out[:, 0], X_out[:, 1], c=[G3[l] for l in k_con.labels_], s=12)
    ax.scatter(*k_con.cluster_centers_.T, marker="X", s=100, c=INK, zorder=4)
    ax.set(xlabel="x1", ylabel="x2")
    titulo(ax, f"Un solo extremo: ARI {adjusted_rand_score(yb, k_sin.labels_):.2f} → {adjusted_rand_score(yb, k_con.labels_[:-1]):.2f}", 9)
    style(ax)
    ax = axs[2]
    ax.scatter(Xd[:, 0], Xd[:, 1], c=[G3[l] for l in kd.labels_], s=12)
    ax.scatter(*kd.cluster_centers_.T, marker="X", s=100, c=INK, zorder=4)
    ax.set(xlabel="x1", ylabel="x2")
    titulo(ax, f"Tamaños y dispersiones distintos: ARI {adjusted_rand_score(yd, kd.labels_):.2f}", 9)
    style(ax)
    NUM["limites"] = dict(ari_moons_kmeans=float(ari_km), ari_moons_dbscan=float(ari_db), ari_blobs=float(adjusted_rand_score(yb, k_sin.labels_)),
                          ari_blobs_outlier=float(adjusted_rand_score(yb, k_con.labels_[:-1])), tamanos_con_outlier=sorted((int(c) for c in np.bincount(k_con.labels_)), reverse=True),
                          ari_desigual=float(adjusted_rand_score(yd, kd.labels_)))
    fig.tight_layout()
    save(fig, "c39_limites.png")


# ----------------------------------------------------------------------------- 15. esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.4, 5.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def caja(x, y, w, h, color, tit, lineas, fs=7.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=color, lw=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.45), w, 0.45, boxstyle="round,pad=0.03", fc=color, ec=color))
        ax.text(x + w / 2, y + h - 0.23, tit, ha="center", va="center", fontsize=9.2, color="white", fontweight="bold")
        for i, t in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.75 - i * 0.27, t, ha="center", va="center", fontsize=fs, color=INK)
    caja(0.1, 3.75, 2.35, 2.35, BLUE, "Sin etiquetas", ["agrupar ≠ clasificar", "no hay Y que predecir", "el «grupo 1» no es «mejor»", "los números son etiquetas"])
    caja(2.6, 3.75, 2.35, 2.35, TEAL, "K-means", ["variables numéricas", "centro = media del grupo", "minimiza WSS (cuadrados)", "estandarizar antes"])
    caja(5.1, 3.75, 2.35, 2.35, AMBER, "K-modes", ["variables categóricas", "centro = moda", "disimilitud = n.º de", "desacuerdos"])
    caja(7.6, 3.75, 2.35, 2.35, VIOLET, "K-prototypes", ["variables mixtas", "medias + modas", "peso γ de lo categórico", "γ cambia los grupos"])
    caja(10.1, 3.75, 2.3, 2.35, C39, "Elegir k", ["codo: WSS", "silueta: máximo", "Gap: regla de 1 error", "estabilidad e interpretación"])
    for xx in (2.45, 4.95, 7.45, 9.95):
        ax.annotate("", xy=(xx + 0.15, 4.95), xytext=(xx, 4.95), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 3.45, boxstyle="round,pad=0.03", fc=C39_BG, ec=C39, lw=1.4))
    pc, pf, si, gp, ce, km, kp = NUM["pca"], NUM["perfiles"], NUM["silueta"], NUM["gap"], NUM["estandarizar"], NUM["kmodes"], NUM["kproto"]
    wk = NUM["codo"]["wss_k3"]
    ax.text(6.25, 3.2, "Datos de la clase: vinos.csv (177 vinos, 13 mediciones químicas) · vinos_clases_reales.csv (clase UCI, solo para validar al final)", ha="center", fontsize=8.5, color=C39, fontweight="bold")
    ax.text(6.25, 2.75, f"K-means (13 variables estandarizadas): k = 3 da WSS = {wk:.2f}, silueta media {si['media_k3']:.3f}, grupos de {pf['n'][0]}, {pf['n'][1]} y {pf['n'][2]} vinos", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 2.35, f"Codo y silueta apuntan a k = 3; Gap (30 simulaciones) elige k = {gp['k_gap']}; con las clases reales: pureza {pc['pureza'] * 100:.1f} % y ARI {pc['ari']:.3f}", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 1.95, f"Sin estandarizar, la prolina aporta el {ce['cuota_prolina'] * 100:.1f} % de las distancias y el ARI cae de {ce['ari_z']:.3f} a {ce['ari_raw']:.3f}", ha="center", fontsize=8.2, color=INK)
    ax.text(6.25, 1.55, f"K-modes (3 variables categorizadas): costo {km['costo_k4']:.0f} con k = 4 · K-prototypes (2 numéricas + 3 categóricas, γ = 1): costo {kp['costo_k3']:.2f} con k = 3 · los costos de métodos distintos no se comparan", ha="center", fontsize=8.0, color=INK)
    ax.text(6.25, 1.05, "R ↔ Python:  scale ↔ (x − media)/std(ddof=1) · kmeans(nstart) ↔ KMeans(n_init) · tot.withinss ↔ inertia_ · silhouette ↔ silhouette_score · clusGap ↔ cálculo propio", ha="center", fontsize=7.8, color=MUTED, style="italic")
    ax.text(6.25, 0.55, "klaR::kmodes ↔ kmodes.KModes · clustMixType::kproto (lambda) ↔ kmodes.KPrototypes (gamma) · prcomp ↔ PCA · table() ↔ pd.crosstab", ha="center", fontsize=7.8, color=MUTED, style="italic")
    save(fig, "c39_esquema.png")


PASOS = dict(iter=iter_fig, wsbs=wsbs_fig, estandarizar=estandarizar_fig, codo=codo_fig, silueta=silueta_fig, gap=gap_fig, perfiles=perfiles_fig,
             pca=pca_fig, validar=validar_fig, init=init_fig, jerarquico=jerarquico_fig, kmodes=kmodes_fig, kproto=kproto_fig, limites=limites_fig,
             esquema=esquema_fig)

if __name__ == "__main__":
    import sys
    _prev = Path(__file__).resolve().parent / "c39_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or list(PASOS)
    if "esquema" in elegidos:
        for k in ("pca", "perfiles", "silueta", "gap", "estandarizar", "kmodes", "kproto", "codo"):
            if k not in elegidos and k not in NUM:
                elegidos.append(k)
    if "validar" in elegidos and "pca" not in NUM and "pca" not in elegidos:
        elegidos.append("pca")
    orden = [e for e in PASOS if e in elegidos and e != "esquema"]
    orden = ([e for e in orden if e == "pca"] + [e for e in orden if e != "pca"])
    orden += (["esquema"] if "esquema" in elegidos else [])
    for k in orden:
        PASOS[k]()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c39_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
