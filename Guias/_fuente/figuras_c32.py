"""Figuras de la Clase 32 (Reducción de dimensionalidad y representación de variables).
Datos: (1) el dataset de churn SINTÉTICO de la profesora Laura Sánchez (1.500 clientes, np.random.default_rng(32)), que se genera
en su propio notebook y aquí se reproduce idéntico; (2) la base real «Base de datos Churn-Actividad repaso.xlsx» (3.150 clientes)
de la carpeta de la clase, con la misma limpieza del notebook. Nada se inventa: todo sale de código.
Ejecutar con el venv que tiene scikit-learn:  python figuras_c32.py   (o con un nombre: python figuras_c32.py scree)."""
import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 32
C32 = "#DB2777"      # color de la clase 32 (rosa)
C32_BG = "#FDF2F8"
BLUE = "#2563EB"
RED = "#DC2626"
GRAY = "#6B7280"
AMBER = "#D97706"
TEAL = "#0D9488"
GREEN = "#059669"
VIOLET = "#7C3AED"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA = next(BASE.glob("Clase 32*"))


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ datos (idénticos al notebook de la profesora)
def generar_churn_sintetico(n=1500, seed=RS):
    rng = np.random.default_rng(seed)
    insatisfaccion = rng.normal(0, 1, n)
    uso = rng.normal(0, 1, n)
    antiguedad = np.clip(rng.normal(24, 12, n), 1, 72).round(0)
    tipo_plan = rng.choice([1, 2], size=n, p=[0.55, 0.45])
    estado = rng.choice([1, 2], size=n, p=[0.85, 0.15])
    edad = np.clip(rng.normal(38, 12, n), 18, 80).round(0)
    numero_fallas = np.clip(6 + 4 * insatisfaccion + rng.normal(0, 2, n), 0, 35).round(0)
    reclamos = (rng.uniform(0, 1, n) < (0.05 + 0.12 * (insatisfaccion > 0.5))).astype(int)
    monto = np.clip((4 + 2 * uso + rng.normal(0, 1.5, n)).round(0), 0, 9)
    segundos = np.clip(4000 + 3000 * uso - 800 * insatisfaccion + rng.normal(0, 900, n), 50, None).round(0)
    frec_llamadas = np.clip((segundos / 65) + rng.normal(0, 8, n), 1, None).round(0)
    frec_sms = np.clip(60 + 90 * uso + rng.normal(0, 40, n), 0, None).round(0)
    n_contactos = np.clip((frec_llamadas * 0.35) + rng.normal(0, 5, n), 0, None).round(0)
    clv = np.clip(300 + 220 * uso - 90 * insatisfaccion + rng.normal(0, 120, n), 10, None).round(0)
    logit = (-2.2 + 0.9 * insatisfaccion + 0.05 * numero_fallas + 1.1 * reclamos
             - 0.02 * antiguedad - 0.15 * uso)
    prob_fuga = 1 / (1 + np.exp(-logit))
    fuga = (rng.uniform(0, 1, n) < prob_fuga).astype(int)
    return pd.DataFrame({
        "numero_fallas": numero_fallas, "reclamos": reclamos, "antiguedad": antiguedad,
        "monto": monto, "segundos": segundos, "frec_llamadas": frec_llamadas,
        "frec_sms": frec_sms, "n_contactos": n_contactos, "tipo_plan": tipo_plan,
        "estado": estado, "edad": edad, "clv": clv, "fuga": fuga,
    })


def limpieza_clase(df):
    """Limpieza del notebook de la profesora: edad imposible -> NaN y mediana; reclamos con la moda."""
    df = df.copy()
    df.loc[(df["edad"] < 0) | (df["edad"] > 100), "edad"] = np.nan
    for col in ["segundos", "frec_llamadas", "frec_sms", "n_contactos", "antiguedad", "edad"]:
        df[col] = df[col].fillna(df[col].median())
    df["reclamos"] = df["reclamos"].fillna(df["reclamos"].mode()[0])
    return df


def preparar(df):
    """Feature engineering + one-hot, igual que el notebook de la clase. Devuelve (X_codificada, y, X_cruda)."""
    y = df["fuga"]
    X_raw = df.drop(columns=["fuga"])
    X_fe = X_raw.copy()
    X_fe["duracion_prom_llamada"] = X_fe["segundos"] / X_fe["frec_llamadas"].replace(0, np.nan)
    X_fe["sms_por_antiguedad"] = X_fe["frec_sms"] / X_fe["antiguedad"].replace(0, np.nan)
    X_fe["intensidad_contacto"] = X_fe["n_contactos"] / X_fe["frec_llamadas"].replace(0, np.nan)
    X_fe = X_fe.fillna(0)
    categoricas = ["tipo_plan", "estado"]
    numericas = [c for c in X_fe.columns if c not in categoricas]
    enc = OneHotEncoder(drop="if_binary", sparse_output=False)
    cat = enc.fit_transform(X_fe[categoricas].astype(str))
    X_enc = pd.concat([X_fe[numericas].reset_index(drop=True),
                       pd.DataFrame(cat, columns=enc.get_feature_names_out(categoricas))], axis=1)
    return X_enc, y.reset_index(drop=True), X_raw.reset_index(drop=True)


def dividir(X_enc, y):
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_enc, y, test_size=0.25, random_state=RS, stratify=y)
    esc = StandardScaler()
    return Xtr_raw, Xte_raw, ytr, yte, esc.fit_transform(Xtr_raw), esc.transform(Xte_raw), esc


df_s = generar_churn_sintetico()
X_enc, y, X_raw = preparar(df_s)
Xtr_raw, Xte_raw, ytr, yte, Xtr, Xte, esc = dividir(X_enc, y)
COLS = list(X_enc.columns)
pca_full = PCA(random_state=RS).fit(Xtr)
var_exp = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_exp)
K80 = int(np.argmax(var_acum >= 0.80) + 1)

df_r = limpieza_clase(pd.read_excel(CARPETA / "Base de datos Churn-Actividad repaso.xlsx"))
XR_enc, yr, XR_raw = preparar(df_r)
XRtr_raw, XRte_raw, yrtr, yrte, XRtr, XRte, escR = dividir(XR_enc, yr)
pca_R = PCA(random_state=RS).fit(XRtr)
varR = pca_R.explained_variance_ratio_
acumR = np.cumsum(varR)

FAMILIA = {"numero_fallas": AMBER, "reclamos": AMBER, "antiguedad": GRAY, "monto": C32, "segundos": C32,
           "frec_llamadas": C32, "frec_sms": C32, "n_contactos": C32, "edad": GRAY, "clv": C32,
           "duracion_prom_llamada": BLUE, "sms_por_antiguedad": BLUE, "intensidad_contacto": BLUE,
           "tipo_plan_2": GRAY, "estado_2": GRAY}


# ------------------------------------------------------------------ 0. esquema
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 2.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    cajas = [("1 · Diagnóstico", "¿por qué demasiadas\nvariables estorban?"), ("2 · Representación", "escalar, codificar\ny construir variables"),
             ("3 · PCA", "resumir variables\ncorrelacionadas"), ("4 · t-SNE", "mirar la estructura\nde vecindad en 2D"),
             ("5 · Decidir", "selección o extracción;\n¿cuántos componentes?")]
    cols = [GRAY, AMBER, C32, BLUE, GREEN]
    w = 2.25
    for i, ((t, q), c) in enumerate(zip(cajas, cols)):
        x = 0.12 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, 0.85), w, 1.35, boxstyle="round,pad=0.04", fc="white", ec=c, lw=1.8))
        ax.text(x + w / 2, 1.88, t, ha="center", va="center", fontsize=10.2, color=c, fontweight="bold")
        ax.text(x + w / 2, 1.32, q, ha="center", va="center", fontsize=8.7, color=INK, linespacing=1.3)
        if i < len(cajas) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 1.52), (x + 2.5 - 0.03, 1.52), arrowstyle="-|>", mutation_scale=11, color=GRAY))
    ax.add_patch(FancyBboxPatch((0.12, 0.12), 12.25, 0.5, boxstyle="round,pad=0.02", fc=C32_BG, ec=C32, lw=1.2))
    ax.text(6.25, 0.37, "Imputar → escalar → codificar → PCA → modelo: todo se APRENDE solo con entrenamiento, dentro de un Pipeline",
            ha="center", va="center", fontsize=9.2, color=C32, fontweight="bold")
    save(fig, "c32_esquema.png")


# ------------------------------------------------------------------ 1. problema de la dimensionalidad
def dimensionalidad_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.95, 0.95], wspace=0.3)

    # (a) matriz de correlación de las 12 variables originales
    a = fig.add_subplot(gs[0])
    num12 = [c for c in X_raw.columns]
    corr = X_raw.corr()
    im = a.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(12))
    a.set_yticks(range(12))
    a.set_xticklabels(num12, rotation=60, ha="right", fontsize=7)
    a.set_yticklabels(num12, fontsize=7)
    for i in range(12):
        for j in range(12):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.6:
                a.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.6, color="white")
    a.grid(False)
    a.spines[:].set_visible(False)
    a.set_title("Variables de uso casi duplicadas (|r| ≥ 0.6)", loc="left", fontsize=9.2, color=INK)
    uso = ["segundos", "frec_llamadas", "n_contactos", "frec_sms", "monto", "clv"]
    cu = corr.loc[uso, uso].values
    NUM["corr_uso_min"] = round(float(cu[np.triu_indices(6, 1)].min()), 2)
    NUM["corr_uso_media"] = round(float(cu[np.triu_indices(6, 1)].mean()), 2)
    NUM["corr_seg_llam"] = round(float(corr.loc["segundos", "frec_llamadas"]), 2)
    NUM["corr_llam_cont"] = round(float(corr.loc["frec_llamadas", "n_contactos"]), 2)
    NUM["corr_seg_clv"] = round(float(corr.loc["segundos", "clv"]), 2)
    # VIF de las variables de uso
    Z = (X_raw - X_raw.mean()) / X_raw.std()
    vif = {}
    for c in uso:
        otros = [k for k in X_raw.columns if k != c]
        r2 = LinearRegression().fit(Z[otros], Z[c]).score(Z[otros], Z[c])
        vif[c] = round(float(1 / (1 - r2)), 1)
    NUM["vif"] = vif

    # (b) concentración de distancias
    b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(1)
    ds = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    razon = []
    for d in ds:
        P = rng.uniform(size=(400, d))
        r = []
        for q in range(15):
            dist = np.linalg.norm(np.delete(P, q, axis=0) - P[q], axis=1)
            r.append((dist.max() - dist.min()) / dist.min())
        razon.append(np.mean(r))
    b.plot(ds, razon, marker="o", color=C32, lw=2)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("número de dimensiones d (400 puntos uniformes)")
    b.set_ylabel("(más lejano − más cercano) / más cercano")
    b.set_title("Todos quedan a la misma distancia", loc="left", fontsize=9.2, color=INK)
    style(b)
    NUM["conc"] = {str(d): round(float(r), 2) for d, r in zip(ds, razon) if d in (2, 10, 100, 1000)}

    # (c) variables ruido con pocas observaciones
    c = fig.add_subplot(gs[2])
    ks = [0, 10, 25, 50, 100, 200, 400]
    nt = 100
    res_te, res_tr = [], []
    for k in ks:
        te, tr = [], []
        for s in range(25):
            r = np.random.default_rng(100 + s)
            idx = r.choice(len(Xtr_raw), nt, replace=False)
            while ytr.values[idx].sum() < 8:
                idx = r.choice(len(Xtr_raw), nt, replace=False)
            A = Xtr_raw.values[idx]
            T = Xte_raw.values
            if k:
                A = np.hstack([A, r.normal(size=(nt, k))])
                T = np.hstack([T, r.normal(size=(len(T), k))])
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(A, ytr.values[idx])
            te.append(roc_auc_score(yte, m.predict_proba(T)[:, 1]))
            tr.append(roc_auc_score(ytr.values[idx], m.predict_proba(A)[:, 1]))
        res_te.append((np.mean(te), np.std(te)))
        res_tr.append(np.mean(tr))
    mu = np.array([m for m, _ in res_te])
    sd = np.array([s for _, s in res_te])
    c.plot(ks, res_tr, ls="--", color=GRAY, marker="s", ms=4, label="entrenamiento (100 filas)")
    c.plot(ks, mu, color=C32, marker="o", lw=2, label="prueba (375 filas)")
    c.fill_between(ks, mu - sd, mu + sd, color=C32, alpha=0.15)
    c.set_xlabel("columnas de ruido agregadas a las 15 variables")
    c.set_ylabel("AUC")
    c.set_ylim(0.5, 1.02)
    c.legend(fontsize=8, frameon=False, loc="lower left")
    c.set_title("Con pocas filas, más columnas = más varianza", loc="left", fontsize=9.2, color=INK)
    style(c)
    NUM["ruido"] = {"ks": ks, "auc_prueba": [round(float(m), 3) for m in mu], "auc_entren": [round(float(m), 3) for m in res_tr]}
    save(fig, "c32_dimensionalidad.png")
    print("dimensionalidad", perf_counter() - t0)


# ------------------------------------------------------------------ 2. representación
def representacion_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 0.72, 0.72, 1.25], wspace=0.42)

    # (a) la escala manda
    a = fig.add_subplot(gs[0])
    vars_ = ["segundos", "clv", "frec_sms", "frec_llamadas", "monto", "reclamos"]
    sd = Xtr_raw[vars_].std()
    ypos = np.arange(len(vars_))
    a.barh(ypos + 0.2, sd.values, height=0.38, color=C32, label="datos originales")
    a.barh(ypos - 0.2, np.ones(len(vars_)), height=0.38, color="#9CA3AF", label="estandarizados")
    a.set_yticks(ypos)
    a.set_yticklabels(vars_)
    a.invert_yaxis()
    a.set_xscale("log")
    a.set_xlabel("desviación estándar (escala log)")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("«segundos» pesa miles de veces más que «reclamos»", loc="left", fontsize=9, color=INK)
    style(a, "x")
    NUM["sd"] = {k: round(float(v), 2) for k, v in sd.items()}

    # (b) ordinal vs one-hot: distancias entre 4 categorías
    cats = ["Norte", "Sur", "Este", "Oeste"]
    ordinal = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
    onehot = np.where(np.eye(4) == 1, 0, np.sqrt(2))
    for k, (M, t) in enumerate([(ordinal, "ordinal (0,1,2,3)"), (onehot, "one-hot")]):
        b = fig.add_subplot(gs[1 + k])
        b.imshow(M, cmap="Purples" if k == 0 else "Greens", vmin=0, vmax=3)
        b.set_xticks(range(4))
        b.set_yticks(range(4))
        b.set_xticklabels(cats, rotation=60, ha="right", fontsize=7.5)
        b.set_yticklabels(cats if k == 0 else [], fontsize=7.5)
        for i in range(4):
            for j in range(4):
                b.text(j, i, f"{M[i, j]:.1f}" if k else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if M[i, j] > 1.8 else INK)
        b.grid(False)
        b.spines[:].set_visible(False)
        b.set_title(t, fontsize=8.8, color=C32 if k == 0 else GREEN)
    fig.text(0.405, 0.97, "Distancia entre categorías según la codificación", ha="center", fontsize=9, color=INK)
    NUM["dist_onehot"] = round(float(np.sqrt(2)), 3)

    # (c) codificación por target: media del target con y sin validación cruzada interna
    c = fig.add_subplot(gs[3])
    rng = np.random.default_rng(7)
    comuna_tr = rng.choice([f"c{i}" for i in range(60)], len(Xtr_raw))     # 60 comunas SIN relación con la fuga
    comuna_te = rng.choice([f"c{i}" for i in range(60)], len(Xte_raw))
    ytr_a, yte_a = ytr.values, yte.values
    medias = pd.Series(ytr_a).groupby(comuna_tr).mean()
    naive_tr = pd.Series(comuna_tr).map(medias).values
    naive_te = pd.Series(comuna_te).map(medias).fillna(ytr_a.mean()).values
    te = TargetEncoder(smooth="auto", target_type="binary", cv=5, random_state=RS)
    cv_tr = te.fit_transform(comuna_tr.reshape(-1, 1), ytr_a).ravel()
    cv_te = te.transform(comuna_te.reshape(-1, 1)).ravel()
    oh = OneHotEncoder(handle_unknown="ignore")
    Otr = oh.fit_transform(comuna_tr.reshape(-1, 1))
    Ote = oh.transform(comuna_te.reshape(-1, 1))
    lr = LogisticRegression(max_iter=2000).fit(Otr, ytr_a)
    res = {"media del target\n(sin cuidado)": (roc_auc_score(ytr_a, naive_tr), roc_auc_score(yte_a, naive_te)),
           "one-hot\n(60 columnas)": (roc_auc_score(ytr_a, lr.predict_proba(Otr)[:, 1]), roc_auc_score(yte_a, lr.predict_proba(Ote)[:, 1])),
           "TargetEncoder\n(validación interna)": (roc_auc_score(ytr_a, cv_tr), roc_auc_score(yte_a, cv_te))}
    xs = np.arange(3)
    c.bar(xs - 0.19, [v[0] for v in res.values()], width=0.36, color="#9CA3AF", label="entrenamiento")
    c.bar(xs + 0.19, [v[1] for v in res.values()], width=0.36, color=C32, label="prueba")
    for i, v in enumerate(res.values()):
        c.text(i - 0.19, v[0] + 0.012, f"{v[0]:.2f}", ha="center", fontsize=8)
        c.text(i + 0.19, v[1] + 0.012, f"{v[1]:.2f}", ha="center", fontsize=8)
    c.axhline(0.5, color=INK, ls="--", lw=1)
    c.set_xticks(xs)
    c.set_xticklabels(list(res), fontsize=7.6)
    c.set_ylim(0.4, 0.82)
    c.set_ylabel("AUC con 60 comunas inventadas")
    c.legend(fontsize=7.8, frameon=False, loc="upper right")
    c.set_title("Una variable SIN señal: ¿quién se engaña?", loc="left", fontsize=9, color=INK)
    style(c, "y")
    NUM["target_enc"] = {k.replace("\n", " "): [round(float(v[0]), 3), round(float(v[1]), 3)] for k, v in res.items()}
    save(fig, "c32_representacion.png")
    print("representacion", perf_counter() - t0)


# ------------------------------------------------------------------ 3. geometría de PCA
def geometria_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.25)
    par = ["segundos", "clv"]
    Z = ((X_raw[par] - X_raw[par].mean()) / X_raw[par].std()).values
    p2 = PCA(random_state=RS).fit(Z)
    r = float(np.corrcoef(Z.T)[0, 1])
    NUM["geo_r"] = round(r, 2)
    NUM["geo_var"] = [round(float(v), 3) for v in p2.explained_variance_ratio_]
    NUM["geo_eig"] = [round(float(v), 3) for v in p2.explained_variance_]
    lim = 3.6

    a = fig.add_subplot(gs[0])
    a.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.35, color="#9CA3AF")
    for k, (col, nm) in enumerate([(C32, "PC1"), (BLUE, "PC2")]):
        v = p2.components_[k] * 2.4 * np.sqrt(p2.explained_variance_[k])
        a.add_patch(FancyArrowPatch((0, 0), tuple(v), arrowstyle="-|>", mutation_scale=15, color=col, lw=2.4))
        a.text(v[0] * 1.13, v[1] * 1.13, nm, color=col, fontsize=10, fontweight="bold", ha="center", va="center")
    a.axhline(0, color=GRID, lw=1)
    a.axvline(0, color=GRID, lw=1)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("segundos (estandarizada)")
    a.set_ylabel("clv (estandarizada)")
    a.set_title(f"PC1 = dirección de mayor varianza (r = {r:.2f})", loc="left", fontsize=9.2, color=INK)
    style(a)

    b = fig.add_subplot(gs[1])
    u = p2.components_[0]
    proy = (Z @ u)[:, None] * u[None, :]
    b.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.2, color="#9CA3AF")
    for i in range(0, len(Z), 12):
        b.plot([Z[i, 0], proy[i, 0]], [Z[i, 1], proy[i, 1]], color=AMBER, lw=0.6, alpha=0.7)
    b.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]], color=C32, lw=2.2)
    b.scatter(proy[:, 0], proy[:, 1], s=7, color=C32, alpha=0.7)
    mse = float(((Z - proy) ** 2).sum(axis=1).mean())
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    b.set_aspect("equal")
    b.set_xlabel("segundos (estandarizada)")
    b.set_title(f"Proyectar sobre PC1: se conserva el {p2.explained_variance_ratio_[0]:.0%}", loc="left", fontsize=9.2, color=INK)
    style(b)
    NUM["geo_perdida"] = round(mse, 3)

    c = fig.add_subplot(gs[2], projection="3d")
    tri = ["segundos", "clv", "numero_fallas"]
    Z3 = ((X_raw[tri] - X_raw[tri].mean()) / X_raw[tri].std()).values
    p3 = PCA(random_state=RS).fit(Z3)
    NUM["geo3_var"] = [round(float(v), 3) for v in p3.explained_variance_ratio_]
    c.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=4, alpha=0.35, color="#6B7280")
    g = np.linspace(-3.5, 3.5, 2)
    A, B = np.meshgrid(g, g)
    P_ = A[..., None] * p3.components_[0] + B[..., None] * p3.components_[1]
    c.plot_surface(P_[..., 0], P_[..., 1], P_[..., 2], alpha=0.25, color=C32)
    c.set_xlabel("segundos", fontsize=8)
    c.set_ylabel("clv", fontsize=8)
    c.set_zlabel("fallas", fontsize=8)
    c.tick_params(labelsize=6)
    c.view_init(elev=18, azim=-55)
    c.set_title(f"3 variables ≈ 1 plano: PC1 + PC2 = {p3.explained_variance_ratio_[:2].sum():.0%}", loc="left", fontsize=9.2, color=INK)
    save(fig, "c32_geometria.png")
    print("geometria", perf_counter() - t0)


# ------------------------------------------------------------------ 4. scree y número de componentes
def scree_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    def scree(ax, ve, ac, k, titulo):
        comps = np.arange(1, len(ve) + 1)
        ax.bar(comps, ve * 100, color=C32, alpha=0.85)
        ax.plot(comps, ac * 100, color=AMBER, marker="o", ms=4, lw=2)
        ax.axhline(80, color=GRAY, ls=":", lw=1)
        ax.axhline(90, color=GRAY, ls=":", lw=1)
        ax.axvline(k, color=INK, ls="--", lw=1)
        ax.text(15.3, 81, "80 %", fontsize=7.5, color=GRAY, ha="right")
        ax.text(15.3, 91, "90 %", fontsize=7.5, color=GRAY, ha="right")
        ax.annotate(f"{ac[k - 1] * 100:.1f} %", (k, ac[k - 1] * 100), textcoords="offset points", xytext=(-6, -16), fontsize=8.5,
                    ha="right", color=INK, fontweight="bold")
        ax.set_xlabel("componente principal")
        ax.set_ylabel("% de varianza")
        ax.set_xticks(comps[::2])
        ax.set_title(titulo, loc="left", fontsize=9.2, color=INK)
        style(ax, "y")

    scree(fig.add_subplot(gs[0]), var_exp, var_acum, K80, f"Datos de la clase: {K80} componentes para ≥ 80 %")
    kR = int(np.argmax(acumR >= 0.80) + 1)
    scree(fig.add_subplot(gs[1]), varR, acumR, kR, f"Base real (3.150): {kR} componentes para ≥ 80 %")

    c = fig.add_subplot(gs[2])
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    ks = list(range(1, 16))
    medias, desv = [], []
    for k in ks:
        pipe = Pipeline([("esc", StandardScaler()), ("pca", PCA(n_components=k, random_state=RS)),
                         ("lr", LogisticRegression(max_iter=2000))])
        sc = cross_val_score(pipe, Xtr_raw, ytr, cv=cv, scoring="roc_auc")
        medias.append(sc.mean())
        desv.append(sc.std())
    medias, desv = np.array(medias), np.array(desv)
    c.plot(ks, medias, color=C32, marker="o", lw=2)
    c.fill_between(ks, medias - desv, medias + desv, color=C32, alpha=0.15)
    c.axhline(medias[-1], color=GRAY, ls="--", lw=1)
    c.text(1.2, medias[-1] + 0.004, "las 15 variables", fontsize=8, color=GRAY)
    c.axvline(K80, color=INK, ls="--", lw=1)
    c.set_xlabel("n_components dentro del Pipeline")
    c.set_ylabel("AUC (5-Fold, solo entrenamiento)")
    c.set_xticks(ks[::2])
    c.set_title("Elegir k con validación cruzada", loc="left", fontsize=9.2, color=INK)
    style(c)
    NUM["scree_s"] = {"var": [round(float(v) * 100, 1) for v in var_exp], "acum": [round(float(v) * 100, 1) for v in var_acum], "k80": K80,
                      "k85": int(np.argmax(var_acum >= 0.85) + 1), "k90": int(np.argmax(var_acum >= 0.90) + 1)}
    NUM["scree_r"] = {"var": [round(float(v) * 100, 1) for v in varR], "acum": [round(float(v) * 100, 1) for v in acumR], "k80": kR,
                      "k90": int(np.argmax(acumR >= 0.90) + 1)}
    NUM["cv_auc"] = {"medias": [round(float(m), 3) for m in medias], "desv": [round(float(d), 3) for d in desv]}
    save(fig, "c32_scree.png")
    print("scree", perf_counter() - t0)


# ------------------------------------------------------------------ 5. cargas y círculo de correlaciones
def loadings_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.35)

    a = fig.add_subplot(gs[0])
    L = pca_full.components_[:4].T
    im = a.imshow(L, cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    a.set_xticks(range(4))
    a.set_xticklabels([f"PC{i + 1}\n({var_exp[i] * 100:.1f} %)" for i in range(4)], fontsize=8)
    a.set_yticks(range(len(COLS)))
    a.set_yticklabels(COLS, fontsize=7.4)
    for i in range(len(COLS)):
        for j in range(4):
            if abs(L[i, j]) >= 0.3:
                a.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(L[i, j]) > 0.45 else INK)
    a.grid(False)
    a.spines[:].set_visible(False)
    a.set_title("Cargas (vectores propios): la «receta»", loc="left", fontsize=9.2, color=INK)
    NUM["cargas"] = {f"PC{j + 1}": {COLS[i]: round(float(L[i, j]), 2) for i in np.argsort(-np.abs(L[:, j]))[:5]} for j in range(4)}

    b = fig.add_subplot(gs[1])
    corr = pca_full.components_[:2].T * np.sqrt(pca_full.explained_variance_[:2])
    th = np.linspace(0, 2 * np.pi, 200)
    b.plot(np.cos(th), np.sin(th), color="#D1D5DB")
    b.axhline(0, color=GRID, lw=1)
    b.axvline(0, color=GRID, lw=1)
    for i, col in enumerate(COLS):
        vx, vy = corr[i]
        b.add_patch(FancyArrowPatch((0, 0), (vx, vy), arrowstyle="-|>", mutation_scale=9, color=FAMILIA[col], lw=1.4))
        if np.hypot(vx, vy) > 0.35 and FAMILIA[col] != C32:
            b.text(vx * 1.1, vy * 1.1, col, fontsize=6.8, color=FAMILIA[col], ha="center", va="center")
    b.text(0.62, 0.2, "6 variables de uso (segundos,\nllamadas, contactos, sms, monto, clv)", fontsize=6.8, color=C32, ha="center")
    b.set_xlim(-1.25, 1.25)
    b.set_ylim(-1.25, 1.25)
    b.set_aspect("equal")
    b.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f} %)")
    b.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f} %)")
    b.set_title("Correlación de cada variable con PC1 y PC2", loc="left", fontsize=9.2, color=INK)
    NUM["circ_max"] = round(float(np.linalg.norm(corr, axis=1).max()), 3)
    NUM["circ"] = {COLS[i]: [round(float(corr[i, 0]), 2), round(float(corr[i, 1]), 2)] for i in range(len(COLS))}

    c = fig.add_subplot(gs[2])
    sc = pca_full.transform(Xtr)
    for val, col, lab in [(0, "#B9C4CC", "activo (0)"), (1, C32, "fuga (1)")]:
        m = ytr.values == val
        c.scatter(sc[m, 0], sc[m, 1], s=6, alpha=0.5, color=col, label=lab)
    c.legend(fontsize=8, frameon=False, markerscale=2)
    c.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f} %)")
    c.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f} %)")
    c.set_title("Clientes en 2 componentes (el color no se usó)", loc="left", fontsize=9.2, color=INK)
    style(c)
    NUM["pc_fuga_corr"] = [round(float(np.corrcoef(sc[:, j], ytr.values)[0, 1]), 3) for j in range(5)]
    fig.legend(handles=[plt.Line2D([0], [0], color=C32, lw=2, label="uso del servicio"), plt.Line2D([0], [0], color=AMBER, lw=2, label="fallas y reclamos"),
                        plt.Line2D([0], [0], color=BLUE, lw=2, label="variables construidas"), plt.Line2D([0], [0], color=GRAY, lw=2, label="resto")],
               loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, "c32_loadings.png")
    print("loadings", perf_counter() - t0)


# ------------------------------------------------------------------ 6. qué pasa si no se estandariza
def escala_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
    p_raw = PCA(random_state=RS).fit(Xtr_raw.values)
    var_raw = Xtr_raw.var()
    cuota = (var_raw / var_raw.sum()).sort_values(ascending=False)

    a = fig.add_subplot(gs[0])
    top = cuota.head(5)
    a.barh(range(len(top)), top.values * 100, color=C32)
    a.set_yticks(range(len(top)))
    a.set_yticklabels(top.index)
    a.invert_yaxis()
    for i, v in enumerate(top.values):
        a.text(v * 100 + 1, i, f"{v * 100:.1f} %", va="center", fontsize=8.5)
    a.set_xlim(0, 118)
    a.set_xlabel("% de la varianza total (sin estandarizar)")
    a.set_title("Una sola columna es casi toda la varianza", loc="left", fontsize=9.2, color=INK)
    style(a, "x")

    b = fig.add_subplot(gs[1])
    comps = np.arange(1, 6)
    b.bar(comps - 0.2, p_raw.explained_variance_ratio_[:5] * 100, width=0.38, color=RED, label="sin estandarizar")
    b.bar(comps + 0.2, var_exp[:5] * 100, width=0.38, color=C32, label="estandarizado")
    for x, v in zip(comps - 0.2, p_raw.explained_variance_ratio_[:5] * 100):
        b.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    for x, v in zip(comps + 0.2, var_exp[:5] * 100):
        b.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    b.set_xlabel("componente principal")
    b.set_ylabel("% de varianza")
    b.set_ylim(0, 108)
    b.legend(fontsize=8, frameon=False)
    b.set_title("El «PC1 de 99 %» es un artefacto de las unidades", loc="left", fontsize=9.2, color=INK)
    style(b, "y")

    c = fig.add_subplot(gs[2])
    sc_raw = p_raw.transform(Xtr_raw.values)
    c.scatter(Xtr_raw["segundos"], sc_raw[:, 0], s=5, alpha=0.4, color=RED)
    r = abs(np.corrcoef(Xtr_raw["segundos"], sc_raw[:, 0])[0, 1])
    c.set_xlabel("segundos")
    c.set_ylabel("PC1 sin estandarizar")
    c.set_title(f"PC1 ≈ «segundos» (|r| = {r:.3f})", loc="left", fontsize=9.2, color=INK)
    style(c)
    NUM["escala"] = {"cuota": {k: round(float(v) * 100, 1) for k, v in cuota.head(4).items()},
                     "pc1_raw": round(float(p_raw.explained_variance_ratio_[0]) * 100, 2),
                     "pc2_raw": round(float(p_raw.explained_variance_ratio_[1]) * 100, 2),
                     "r_pc1_seg": round(float(r), 4),
                     "carga_seg": round(float(abs(p_raw.components_[0][COLS.index("segundos")])), 3),
                     "k90_raw": int(np.argmax(np.cumsum(p_raw.explained_variance_ratio_) >= 0.90) + 1)}
    save(fig, "c32_escala.png")
    print("escala", perf_counter() - t0)


# ------------------------------------------------------------------ 7. t-SNE frente a PCA
def tsne_fig():
    t0 = perf_counter()
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.1))
    cols = {0: "#B9C4CC", 1: C32}

    def pintar(ax, emb, titulo):
        for val in (0, 1):
            m = ytr.values == val
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.55, color=cols[val], label="fuga" if val else "activo")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titulo, loc="left", fontsize=9, color=INK)

    emb_pca = PCA(n_components=2, random_state=RS).fit_transform(Xtr)
    pintar(axes[0, 0], emb_pca, "PCA (2 componentes)")
    axes[0, 0].legend(fontsize=7.5, frameon=False, markerscale=2, loc="upper left")
    tw = {"PCA": trustworthiness(Xtr, emb_pca, n_neighbors=10)}
    kl, tiempos = {}, {}
    for ax, perp in zip(axes[0, 1:], [5, 30, 80]):
        t1 = perf_counter()
        ts = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=RS)
        e = ts.fit_transform(Xtr)
        tiempos[perp] = round(perf_counter() - t1, 1)
        pintar(ax, e, f"t-SNE · perplexity = {perp}")
        tw[f"perp {perp}"] = trustworthiness(Xtr, e, n_neighbors=10)
        kl[perp] = round(float(ts.kl_divergence_), 3)
    for ax, sem in zip(axes[1, :3], [0, 1, 2]):
        e = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto", random_state=sem).fit_transform(Xtr)
        pintar(ax, e, f"perplexity = 30 · init aleatoria · semilla {sem}")
    ax = axes[1, 3]
    nombres = list(tw)
    ax.bar(range(len(nombres)), [tw[k] for k in nombres], color=[GRAY, C32, C32, C32])
    for i, k in enumerate(nombres):
        ax.text(i, tw[k] + 0.004, f"{tw[k]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=7.8)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("trustworthiness (10 vecinos)")
    ax.set_title("¿Conserva a los vecinos?", loc="left", fontsize=9, color=INK)
    style(ax, "y")
    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    NUM["tsne"] = {"trustworthiness": {k: round(float(v), 3) for k, v in tw.items()}, "kl": kl, "segundos": tiempos}
    save(fig, "c32_tsne.png")
    print("tsne", perf_counter() - t0)


# ------------------------------------------------------------------ 8. fuga de datos con PCA
def leakage_fig():
    t0 = perf_counter()
    res = {"sel_fuera": [], "sel_pipe": [], "pca_fuera": [], "pca_pipe": []}
    for s in range(40):
        rng = np.random.default_rng(s)
        Xn = rng.normal(size=(100, 1000))
        yn = rng.integers(0, 2, 100)
        lr = LogisticRegression(max_iter=1000)
        res["sel_fuera"].append(cross_val_score(lr, SelectKBest(f_classif, k=10).fit_transform(Xn, yn), yn, cv=5).mean())
        res["sel_pipe"].append(cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=10)), ("m", lr)]), Xn, yn, cv=5).mean())
        Z = PCA(n_components=10, random_state=RS).fit_transform(StandardScaler().fit_transform(Xn))
        res["pca_fuera"].append(cross_val_score(lr, Z, yn, cv=5).mean())
        res["pca_pipe"].append(cross_val_score(Pipeline([("esc", StandardScaler()), ("pca", PCA(n_components=10, random_state=RS)), ("m", lr)]), Xn, yn, cv=5).mean())
    NUM["fuga"] = {k: dict(media=round(float(np.mean(v)), 3), min=round(float(np.min(v)), 3), max=round(float(np.max(v)), 3)) for k, v in res.items()}

    fig = plt.figure(figsize=(12.5, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95])
    a = fig.add_subplot(gs[0])
    datos = [res["sel_fuera"], res["sel_pipe"], res["pca_fuera"], res["pca_pipe"]]
    bp = a.boxplot(datos, widths=0.55, patch_artist=True, medianprops=dict(color=INK, lw=1.8),
                   tick_labels=["SelectKBest\nfuera (supervisado)", "SelectKBest\nen Pipeline", "PCA fuera\n(no supervisado)", "PCA\nen Pipeline"])
    for patch, c in zip(bp["boxes"], [RED, C32, AMBER, C32]):
        patch.set(facecolor=c, alpha=0.55)
    a.axhline(0.5, color=INK, ls="--", lw=1.2)
    a.text(4.45, 0.505, "azar", fontsize=8.4, color=INK, ha="right")
    a.set_ylabel("exactitud en validación cruzada")
    a.set_title("Ruido puro (100 filas, 1.000 columnas, etiquetas al azar)", loc="left", fontsize=9.2, color=INK)
    a.tick_params(axis="x", labelsize=8)
    style(a, "y")

    b = fig.add_subplot(gs[1])
    b.set_xlim(0, 10)
    b.set_ylim(0, 4.4)
    b.axis("off")
    filas = [("MAL", RED, "ajustar scaler / imputador / PCA / encoder con todas las filas", "→ luego dividir o validar", 3.2),
             ("BIEN", C32, "dividir primero; Pipeline.fit(X_train, y_train)", "→ cada paso aprende solo con train (y con cada pliegue)", 1.7)]
    for etiqueta, c, t1, t2, yy in filas:
        b.add_patch(FancyBboxPatch((0.05, yy - 0.5), 9.85, 1.15, boxstyle="round,pad=0.04", fc="white", ec=c, lw=1.8))
        b.text(0.35, yy + 0.05, etiqueta, fontsize=12, color=c, fontweight="bold", va="center")
        b.text(1.65, yy + 0.3, t1, fontsize=8.8, color=INK, va="center")
        b.text(1.65, yy - 0.15, t2, fontsize=8.8, color=c, va="center", fontweight="bold")
    b.text(0.05, 0.55, "PCA no ve «y», por eso su fuga es discreta; la selección y el target encoding sí ven «y».", fontsize=8.6, color=GRAY)
    fig.tight_layout(w_pad=2)
    save(fig, "c32_leakage.png")
    print("leakage", perf_counter() - t0)


if __name__ == "__main__":
    import sys
    pasos = dict(esquema=esquema_fig, dimensionalidad=dimensionalidad_fig, representacion=representacion_fig, geometria=geometria_fig,
                 scree=scree_fig, loadings=loadings_fig, escala=escala_fig, tsne=tsne_fig, leakage=leakage_fig)
    elegidos = sys.argv[1:] or list(pasos)
    for k in elegidos:
        pasos[k]()
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
