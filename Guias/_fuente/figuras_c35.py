"""Figuras de la Clase 35 (Bagging y Random Forest). Datos de la clase: Heart.xlsx (clasificación) y ames_housing.csv
(regresión) del profesor Bastian Galasso, más el conjunto de dígitos de scikit-learn para el ejemplo de imágenes.
Ejecutar con el venv que tiene sklearn (sin necesidad de xgboost/lightgbm)."""
import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, KFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[2]
CLASE35 = next(BASE.glob("Clase*35*"))
RS = 42

C35 = "#0D9488"      # color de la clase 35 (verde azulado)
BLUE = "#2563EB"
RED = "#DC2626"
GRAY = "#6B7280"
AMBER = "#D97706"
PURPLE = "#7C3AED"


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ datos
def prep(X, escalar=False):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    pasos_num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if escalar else [])
    return ColumnTransformer([
        ("num", Pipeline(pasos_num), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


heart = pd.read_excel(CLASE35 / "Heart.xlsx")
Xh, yh = heart.drop(columns="AHD"), heart["AHD"].map({"No": 0, "Yes": 1})
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.25, stratify=yh, random_state=RS)
# Igual que el notebook del profesor: se conservan todas las columnas de Ames salvo el índice.
ames = pd.read_csv(CLASE35 / "ames_housing.csv").drop(columns=["Unnamed: 0"])
Xa, ya = ames.drop(columns="Sale_Price"), ames["Sale_Price"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.20, random_state=RS)
# validación DENTRO del train de Ames para las curvas (el test queda intacto)
Xa_fit, Xa_val, ya_fit, ya_val = train_test_split(Xa_tr, ya_tr, test_size=0.25, random_state=RS)
_pa = prep(Xa_tr).fit(Xa_fit)
A_fit, A_val = _pa.transform(Xa_fit), _pa.transform(Xa_val)
_ph = prep(Xh_tr).fit(Xh_tr)
H_tr, H_te = _ph.transform(Xh_tr), _ph.transform(Xh_te)
H_names = _ph.get_feature_names_out()
yf, yv = ya_fit.values, ya_val.values


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def variable_original(nombre, cols_cat):
    """'cat__ChestPain_asymptomatic' -> 'ChestPain'; 'num__Age' -> 'Age' (agrupa las dummies de una misma variable)."""
    if nombre.startswith("num__"):
        return nombre[5:]
    resto = nombre[5:]
    mejores = [c for c in cols_cat if resto.startswith(c + "_")]
    return max(mejores, key=len) if mejores else resto


NUM = {}


# ------------------------------------------------------------------ 1. bootstrap, OOB e inestabilidad
def bootstrap_fig():
    t0 = perf_counter()
    rng = np.random.default_rng(RS)
    n = len(Xh_tr)
    S = 2000
    cuentas = np.zeros((S, n), dtype=int)
    for s in range(S):
        cuentas[s] = np.bincount(rng.integers(0, n, n), minlength=n)
    oob_pct = (cuentas == 0).mean(axis=1)
    obs = np.array([(cuentas == k).mean() for k in range(5)] + [(cuentas >= 5).mean()])
    teo = np.array([poisson.pmf(k, 1) for k in range(5)] + [1 - poisson.cdf(4, 1)])

    # inestabilidad: variable de la raíz en 200 árboles bootstrap sin podar
    cols_cat = Xh_tr.select_dtypes(exclude=np.number).columns.tolist()
    rng = np.random.default_rng(RS)       # mismo flujo que el notebook (A2)
    raices, pred_arboles = [], []
    for b in range(200):
        idx = rng.integers(0, n, n)
        t = DecisionTreeClassifier(random_state=b).fit(H_tr[idx], yh_tr.values[idx])
        raices.append(variable_original(H_names[t.tree_.feature[0]], cols_cat))
        pred_arboles.append(t.predict(H_te))
    frec = pd.Series(raices).value_counts(normalize=True)
    P = np.array(pred_arboles)
    desacuerdo = float(np.mean([np.mean(P[i] != P[j]) for i in range(0, 200, 2) for j in range(1, 200, 2) if i != j]))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    a = axes[0]
    ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(20000), 200)).astype(int))
    a.plot(ns, (1 - 1 / ns) ** ns, color=C35, lw=2.4)
    a.axhline(np.exp(-1), color=GRAY, ls=":", lw=1.2)
    a.text(2.2, np.exp(-1) + 0.012, "e⁻¹ ≈ 0.368", fontsize=8.5, color=GRAY)
    for nn, dy in [(8, -0.05), (227, -0.05)]:
        v = (1 - 1 / nn) ** nn
        a.plot(nn, v, "o", color=INK, ms=6)
        a.annotate(f"n = {nn}: {v:.3f}", (nn, v), xytext=(nn * 1.5, v + dy), fontsize=8.5, color=INK)
    a.set_xscale("log")
    a.set_ylim(0.2, 0.4)
    a.set_xlabel("Tamaño de la muestra n")
    a.set_ylabel("P(una fila queda fuera) = (1 − 1/n)ⁿ")
    a.set_title("La fracción OOB converge a 36.8 %", loc="left", fontsize=9.8, color=INK)
    style(a)
    b = axes[1]
    xs = np.arange(6)
    b.bar(xs - 0.2, obs, width=0.4, color=C35, label="simulado (Heart, n = 227)")
    b.bar(xs + 0.2, teo, width=0.4, color=GRAY, alpha=0.7, label="Poisson(1)")
    for x, v in zip(xs, obs):
        b.text(x - 0.2, v + 0.008, f"{v:.0%}", ha="center", fontsize=8)
    b.set_xticks(xs, ["0\n(OOB)", "1", "2", "3", "4", "5+"])
    b.set_xlabel("Veces que una fila entra a la muestra")
    b.set_ylabel("Proporción de filas")
    b.set_title("Una fila entra 0, 1, 2… veces", loc="left", fontsize=9.8, color=INK)
    b.legend(frameon=False, fontsize=8.2, loc="upper right")
    style(b, "y")
    c = axes[2]
    top = frec.head(6)
    c.barh(top.index[::-1], top.values[::-1], color=BLUE)
    for i, v in enumerate(top.values[::-1]):
        c.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8.5)
    c.set_xlim(0, top.max() + 0.12)
    c.set_xlabel("Proporción de árboles")
    c.set_title("Variable de la raíz en 200 árboles bootstrap", loc="left", fontsize=9.8, color=INK)
    style(c, "x")
    fig.tight_layout(w_pad=2)
    save(fig, "c35_bootstrap.png")
    NUM["boot"] = dict(oob_media=float(oob_pct.mean()), oob_sd=float(oob_pct.std()), oob8=(1 - 1 / 8) ** 8,
                       oob227=(1 - 1 / 227) ** 227, oob2344=(1 - 1 / 2344) ** 2344,
                       obs=[round(float(v), 3) for v in obs], raices={k: round(float(v), 3) for k, v in frec.head(5).items()},
                       n_raices=int(frec.size), desacuerdo=desacuerdo)
    print("bootstrap", perf_counter() - t0)


# ------------------------------------------------------------------ 2. varianza y correlación
def estimar_rho(mf, K=60, B=100, frac=0.5, semilla=0):
    """Estima σ² (varianza de un árbol), ρ (correlación entre dos árboles) y Var(B) del promedio, como en ESL 15.4:
    la aleatoriedad que se repite es la MUESTRA de entrenamiento (K submuestras del train) y el azar del algoritmo."""
    rng = np.random.default_rng(semilla)
    m = int(len(A_fit) * frac)
    P = np.empty((K, B, len(A_val)))
    for k in range(K):
        idx = rng.choice(len(A_fit), m, replace=False)
        rf = RandomForestRegressor(n_estimators=B, max_features=mf, min_samples_leaf=1, n_jobs=-1,
                                   random_state=int(rng.integers(1e9))).fit(A_fit[idx], yf[idx])
        P[k] = np.array([e.predict(A_val) for e in rf.estimators_])
    C = P - P.mean(axis=0, keepdims=True)
    cov = np.einsum("kbx,kcx->bcx", C, C) / (K - 1)
    diag = np.einsum("bbx->bx", cov)
    sigma2 = float(diag.mean())
    off = (cov.sum(axis=(0, 1)) - diag.sum(axis=0)) / (B * (B - 1))
    rho = float(off.mean() / sigma2)
    bs = [1, 2, 5, 10, 25, 50, 100]
    var_b = [float(P[:, :b, :].mean(axis=1).var(axis=0, ddof=1).mean()) for b in bs]
    return dict(sigma2=sigma2, rho=rho, bs=bs, var_b=var_b)


def varianza_fig():
    t0 = perf_counter()
    res = {}
    for nombre, mf in [("Bagging (todas)", 1.0), ("RF (1/3)", 1 / 3), ("RF (10 %)", 0.10)]:
        res[nombre] = estimar_rho(mf)
        print("  ", nombre, round(res[nombre]["rho"], 3), round(res[nombre]["sigma2"] ** 0.5, 0))
    cols = {"Bagging (todas)": BLUE, "RF (1/3)": C35, "RF (10 %)": PURPLE}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = axes[0]
    Bs = np.arange(1, 301)
    for nombre, r in res.items():
        teo = r["rho"] * r["sigma2"] + (1 - r["rho"]) * r["sigma2"] / Bs
        a.plot(Bs, np.sqrt(teo) / 1000, color=cols[nombre], lw=2.2, label=f"{nombre}: ρ̂ = {r['rho']:.2f}")
        a.plot(r["bs"], np.sqrt(r["var_b"]) / 1000, "o", color=cols[nombre], ms=6, mec="white")
        a.axhline(np.sqrt(r["rho"] * r["sigma2"]) / 1000, color=cols[nombre], ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("Número de árboles B")
    a.set_ylabel("Desvío del promedio de B árboles (miles USD)")
    a.set_title("Líneas: fórmula ρσ² + (1 − ρ)σ²/B · puntos: medido · punteado: el piso √(ρσ²)", loc="left",
                fontsize=9.2, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a)
    b = axes[1]
    nombres = list(res)
    x = np.arange(3)
    sig = [np.sqrt(res[n]["sigma2"]) / 1000 for n in nombres]
    piso = [np.sqrt(res[n]["rho"] * res[n]["sigma2"]) / 1000 for n in nombres]
    b.bar(x - 0.2, sig, width=0.4, color=GRAY, alpha=0.8, label="un árbol (σ)")
    b.bar(x + 0.2, piso, width=0.4, color=[cols[n] for n in nombres], label="bosque con B → ∞ (√ρσ²)")
    for i, (s_, p_) in enumerate(zip(sig, piso)):
        b.text(i - 0.2, s_ + 0.4, f"{s_:.1f}", ha="center", fontsize=8.5)
        b.text(i + 0.2, p_ + 0.4, f"{p_:.1f}", ha="center", fontsize=8.5)
    b.set_ylim(0, 52)
    b.set_xticks(x, ["Bagging\n(todas las var.)", "RF\nmax_features = 1/3", "RF\nmax_features = 10 %"])
    b.set_ylabel("Desvío de la predicción (miles USD)")
    b.set_title("Menos variables por nodo: ρ baja y σ sube", loc="left", fontsize=9.8, color=INK)
    b.legend(frameon=False, fontsize=8.3, loc="upper right")
    style(b, "y")
    fig.tight_layout(w_pad=2)
    save(fig, "c35_varianza.png")
    NUM["var"] = {k: dict(rho=round(v["rho"], 3), sigma=round(v["sigma2"] ** 0.5), piso=round((v["rho"] * v["sigma2"]) ** 0.5),
                          medido={b: round(x ** 0.5) for b, x in zip(v["bs"], v["var_b"])},
                          formula={b: round((v["rho"] * v["sigma2"] + (1 - v["rho"]) * v["sigma2"] / b) ** 0.5) for b in v["bs"]})
                  for k, v in res.items()}
    print("varianza", perf_counter() - t0)


# ------------------------------------------------------------------ 3. max_features
def max_features_fig():
    t0 = perf_counter()
    fr = [0.01, 0.02, 0.05, 0.10, 0.20, 1 / 3, 0.50, 0.75, 1.0]
    vals, oobs, tiempos = [], [], []
    for f in fr:
        tt = perf_counter()
        rf = RandomForestRegressor(n_estimators=300, max_features=f, oob_score=True, n_jobs=-1, random_state=RS).fit(A_fit, yf)
        tiempos.append(perf_counter() - tt)
        vals.append(rmse(yv, rf.predict(A_val)))
        oobs.append(rmse(yf, rf.oob_prediction_))
        print("   ames", round(f, 3), round(vals[-1]), round(oobs[-1]), round(tiempos[-1], 1))
    p = H_tr.shape[1]
    ms = [1, 2, 3, 4, 6, 9, 13, p]
    ms = sorted(set(m for m in ms if m <= p))
    h_oob, h_cv = [], []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    for m in ms:
        rf = RandomForestClassifier(n_estimators=300, max_features=m, oob_score=True, n_jobs=-1, random_state=RS).fit(H_tr, yh_tr)
        h_oob.append(roc_auc_score(yh_tr, rf.oob_decision_function_[:, 1]))
        pipe = Pipeline([("p", prep(Xh_tr)), ("m", RandomForestClassifier(n_estimators=300, max_features=m, n_jobs=-1, random_state=RS))])
        h_cv.append(cross_val_score(pipe, Xh_tr, yh_tr, cv=cv, scoring="roc_auc").mean())
        print("   heart", m, round(h_oob[-1], 4), round(h_cv[-1], 4))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.5))
    a = axes[0]
    a.plot(fr, np.array(vals) / 1000, "-o", color=C35, lw=2.2, label="validación")
    a.plot(fr, np.array(oobs) / 1000, "--s", color=BLUE, lw=1.8, ms=5, label="OOB")
    k = int(np.argmin(vals))
    a.plot(fr[k], vals[k] / 1000, "o", color=RED, ms=11, mfc="none", mew=2)
    a.annotate(f"mínimo: {fr[k]:.2f} de las columnas\n({vals[k] / 1000:.1f}k)", (fr[k], vals[k] / 1000),
               xytext=(0.16, 31.5), fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    a.annotate("bagging\n(todas)", (1.0, vals[-1] / 1000), xytext=(0.62, 21.4), fontsize=8.5, color=GRAY)
    a.set_ylim(21, 35)
    a.set_xscale("log")
    a.set_xlabel(f"max_features (fracción de las {A_fit.shape[1]} columnas, escala log)")
    a.set_ylabel("RMSE (miles USD)")
    a.set_title("Ames: demasiado poco o demasiado, cuesta", loc="left", fontsize=9.8, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a)
    b = axes[1]
    b.plot(ms, h_oob, "--s", color=BLUE, lw=1.8, ms=5, label="AUC OOB")
    b.plot(ms, h_cv, "-o", color=C35, lw=2.2, label="AUC validación cruzada (5×3)")
    b.axvline(np.sqrt(13), color=GRAY, ls=":", lw=1.2)
    b.text(np.sqrt(13) + 0.1, min(h_cv + h_oob) + 0.0015, "√13 ≈ 3\n(valor por\ndefecto)", fontsize=8.3, color=GRAY)
    b.set_xscale("log")
    b.set_xticks(ms, [str(m) for m in ms])
    b.minorticks_off()
    b.set_xlabel(f"max_features (número de columnas, de {p})")
    b.set_ylabel("AUC en el train de Heart")
    b.set_title("Heart: la curva es casi plana", loc="left", fontsize=9.8, color=INK)
    b.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(b)
    fig.tight_layout(w_pad=2.5)
    save(fig, "c35_max_features.png")
    NUM["mf"] = dict(ames={f"{f:.3f}": (round(v), round(o), round(t, 1)) for f, v, o, t in zip(fr, vals, oobs, tiempos)},
                     ncols_ames=int(A_fit.shape[1]), best_ames=round(fr[k], 3),
                     heart={m: (round(o, 4), round(c, 4)) for m, o, c in zip(ms, h_oob, h_cv)}, ncols_heart=int(p))
    print("max_features", perf_counter() - t0)


# ------------------------------------------------------------------ 4. curvas con el número de árboles
def curva_oob(est, X, y, muestras):
    """RMSE OOB acumulado: para cada B usa, en cada fila, solo los árboles que NO la vieron."""
    n = len(y)
    pred = np.array([e.predict(X[:, f]) if f is not None else e.predict(X) for e, f in est])
    mask = np.ones((len(est), n), dtype=bool)
    for b, idx in enumerate(muestras):
        mask[b, np.unique(idx)] = False
    sums = np.cumsum(pred * mask, axis=0)
    cnt = np.cumsum(mask, axis=0)
    out = []
    for b in range(len(est)):
        ok = cnt[b] > 0
        out.append(rmse(y[ok], sums[b][ok] / cnt[b][ok]) if ok.sum() > 0.95 * n else np.nan)
    return out


def curvas_fig():
    t0 = perf_counter()
    B = 300
    tree = DecisionTreeRegressor(random_state=RS).fit(A_fit, yf)
    t_val = rmse(yv, tree.predict(A_val))
    bag = BaggingRegressor(DecisionTreeRegressor(), n_estimators=B, oob_score=True, n_jobs=-1, random_state=RS).fit(A_fit, yf)
    rf = RandomForestRegressor(n_estimators=B, max_features=1 / 3, oob_score=True, n_jobs=-1, random_state=RS).fit(A_fit, yf)
    n_ = np.arange(1, B + 1).reshape(-1, 1)
    pb = np.array([e.predict(A_val[:, f]) for e, f in zip(bag.estimators_, bag.estimators_features_)])
    pr = np.array([e.predict(A_val) for e in rf.estimators_])
    cb, cr = [rmse(yv, c) for c in np.cumsum(pb, 0) / n_], [rmse(yv, c) for c in np.cumsum(pr, 0) / n_]
    ob = curva_oob(list(zip(bag.estimators_, bag.estimators_features_)), A_fit, yf, bag.estimators_samples_)
    orf = curva_oob([(e, None) for e in rf.estimators_], A_fit, yf, rf.estimators_samples_)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.5))
    a = axes[0]
    a.axhline(t_val / 1000, color=GRAY, ls=":", lw=1.6)
    a.text(B, t_val / 1000 + 0.6, f"un árbol sin podar: {t_val / 1000:.1f}k", ha="right", fontsize=8.5, color=GRAY)
    a.plot(range(1, B + 1), np.array(cb) / 1000, color=BLUE, lw=2.2, label="Bagging (todas las variables)")
    a.plot(range(1, B + 1), np.array(cr) / 1000, color=C35, lw=2.4, label="Random Forest (max_features = 1/3)")
    a.set_ylim(15, 50)
    a.set_xlabel("Número de árboles B")
    a.set_ylabel("RMSE en validación (miles USD)")
    a.set_title("Ames: pocos árboles bastan para casi todo el beneficio", loc="left", fontsize=9.8, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a)
    b = axes[1]
    b.plot(range(1, B + 1), np.array(cr) / 1000, color=C35, lw=2.4, label="validación (hold-out)")
    b.plot(range(1, B + 1), np.array(orf) / 1000, color=AMBER, lw=2.0, ls="--", label="OOB (sin validación aparte)")
    b.set_ylim(18, 40)
    b.set_xlim(0, B)
    b.set_xlabel("Número de árboles B")
    b.set_ylabel("RMSE (miles USD)")
    b.set_title("El error OOB se estabiliza igual que el de validación (más alto)", loc="left", fontsize=9.8, color=INK)
    b.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(b)
    fig.tight_layout(w_pad=2.5)
    save(fig, "c35_curvas.png")
    NUM["curvas"] = dict(tree=t_val, bag={k: cb[k - 1] for k in (1, 10, 50, 100, 300)}, rf={k: cr[k - 1] for k in (1, 10, 50, 100, 300)},
                         oob_rf={k: orf[k - 1] for k in (50, 100, 300)}, oob_bag300=ob[-1],
                         b95_rf=int(next(i + 1 for i, v in enumerate(cr) if v <= cr[-1] * 1.02)),
                         b95_bag=int(next(i + 1 for i, v in enumerate(cb) if v <= cb[-1] * 1.02)),
                         rf_oob_r2=float(rf.oob_score_), bag_oob_r2=float(bag.oob_score_))
    print("curvas", perf_counter() - t0)


# ------------------------------------------------------------------ 5. importancia: impureza vs permutación
def importancia_fig():
    t0 = perf_counter()
    rng = np.random.default_rng(7)
    Xh2 = Xh.copy()
    Xh2["Ruido_continuo"] = rng.normal(size=len(Xh2))
    Xh2["Ruido_binario"] = rng.integers(0, 2, size=len(Xh2))
    a_tr, a_te, b_tr, b_te = train_test_split(Xh2, yh, test_size=0.25, stratify=yh, random_state=RS)
    pipe = Pipeline([("p", prep(a_tr)), ("m", RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1))]).fit(a_tr, b_tr)
    cols_cat = a_tr.select_dtypes(exclude=np.number).columns.tolist()
    nm = pipe.named_steps["p"].get_feature_names_out()
    mdi = pd.Series(pipe.named_steps["m"].feature_importances_, nm)
    mdi = mdi.groupby([variable_original(n, cols_cat) for n in mdi.index]).sum().sort_values(ascending=False)
    perm = permutation_importance(pipe, a_te, b_te, scoring="roc_auc", n_repeats=30, random_state=RS, n_jobs=1)
    perm = pd.Series(perm.importances_mean, a_te.columns)
    perm_tr = permutation_importance(pipe, a_tr, b_tr, scoring="roc_auc", n_repeats=10, random_state=RS, n_jobs=1)
    perm_tr = pd.Series(perm_tr.importances_mean, a_tr.columns)
    orden = mdi.index.tolist()
    rows = orden[:7] + [c for c in ["Ruido_continuo", "Ruido_binario"] if c not in orden[:7]]
    rows = rows[::-1]
    colr = [RED if r.startswith("Ruido") else C35 for r in rows]
    # Ames
    pa = Pipeline([("p", prep(Xa_tr)), ("m", RandomForestRegressor(n_estimators=200, max_features=1 / 3, n_jobs=-1, random_state=RS))]).fit(Xa_tr, ya_tr)
    cat_a = Xa_tr.select_dtypes(exclude=np.number).columns.tolist()
    nma = pa.named_steps["p"].get_feature_names_out()
    mdi_a = pd.Series(pa.named_steps["m"].feature_importances_, nma)
    mdi_a = mdi_a.groupby([variable_original(n, cat_a) for n in mdi_a.index]).sum()
    pr_a = permutation_importance(pa, Xa_te, ya_te, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=RS, n_jobs=1)
    pr_a = pd.Series(pr_a.importances_mean, Xa_te.columns)
    top_a = pr_a.sort_values(ascending=False).head(7).index.tolist()[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    a = axes[0]
    a.barh(rows, [mdi[r] for r in rows], color=colr)
    a.set_title("Heart · impureza (MDI), en el train", loc="left", fontsize=9.6, color=INK)
    a.set_xlabel("Importancia (suma 1)")
    style(a, "x")
    b = axes[1]
    b.barh(rows, [perm[r] for r in rows], color=colr)
    b.axvline(0, color=INK, lw=0.8)
    b.set_yticklabels([])
    b.set_title("Heart · permutación, en el test", loc="left", fontsize=9.6, color=INK)
    b.set_xlabel("Caída del AUC al desordenar la variable")
    style(b, "x")
    c = axes[2]
    y_ = np.arange(len(top_a))
    c.barh(y_ + 0.2, [mdi_a[v] / mdi_a.sum() for v in top_a], height=0.4, color=BLUE, label="impureza (MDI)")
    c.barh(y_ - 0.2, [pr_a[v] / pr_a.sum() for v in top_a], height=0.4, color=C35, label="permutación (test)")
    c.set_yticks(y_, top_a)
    c.set_xlabel("Participación en el total")
    c.set_title("Ames · mismas dos medidas, 7 mayores", loc="left", fontsize=9.6, color=INK)
    c.legend(frameon=False, fontsize=8.3, loc="lower right")
    style(c, "x")
    fig.tight_layout(w_pad=1.5)
    save(fig, "c35_importancia.png")
    NUM["imp"] = dict(mdi_top=mdi.head(6).round(3).to_dict(), mdi_ruido=mdi[["Ruido_continuo", "Ruido_binario"]].round(3).to_dict(),
                      rank_ruido_cont=int(orden.index("Ruido_continuo")) + 1, rank_ruido_bin=int(orden.index("Ruido_binario")) + 1,
                      perm_ruido=perm[["Ruido_continuo", "Ruido_binario"]].round(4).to_dict(),
                      perm_top=perm.sort_values(ascending=False).head(6).round(3).to_dict(),
                      perm_tr_ruido=perm_tr[["Ruido_continuo", "Ruido_binario"]].round(4).to_dict(),
                      ames_mdi=(mdi_a / mdi_a.sum()).sort_values(ascending=False).head(5).round(3).to_dict(),
                      ames_perm=(pr_a / pr_a.sum()).sort_values(ascending=False).head(5).round(3).to_dict(),
                      ames_perm_rmse=pr_a.sort_values(ascending=False).head(5).round(0).to_dict())
    print("importancia", perf_counter() - t0)


# ------------------------------------------------------------------ 6. OOB vs validación cruzada vs test
def oob_cv_fig():
    t0 = perf_counter()
    # Heart
    tt = perf_counter()
    rf_h = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=RS).fit(H_tr, yh_tr)
    t_oob_h = perf_counter() - tt
    h_oob = rf_h.oob_score_
    tt = perf_counter()
    pipe_h = Pipeline([("p", prep(Xh_tr)), ("m", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RS))])
    cvr = cross_validate(pipe_h, Xh_tr, yh_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), scoring="accuracy")
    t_cv_h = perf_counter() - tt
    h_cv = cvr["test_score"].mean()
    h_te = accuracy_score(yh_te, rf_h.predict(H_te))
    # Ames
    tt = perf_counter()
    rf_a = RandomForestRegressor(n_estimators=300, max_features=1 / 3, oob_score=True, n_jobs=-1, random_state=RS)
    pre_a = prep(Xa_tr).fit(Xa_tr)
    Xa_t, Xa_s = pre_a.transform(Xa_tr), pre_a.transform(Xa_te)
    rf_a.fit(Xa_t, ya_tr)
    t_oob_a = perf_counter() - tt
    a_oob = rmse(ya_tr, rf_a.oob_prediction_)
    tt = perf_counter()
    pipe_a = Pipeline([("p", prep(Xa_tr)), ("m", RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS))])
    cva = cross_validate(pipe_a, Xa_tr, ya_tr, cv=KFold(5, shuffle=True, random_state=RS), scoring="neg_root_mean_squared_error")
    t_cv_a = perf_counter() - tt
    a_cv = -cva["test_score"].mean()
    a_te = rmse(ya_te, rf_a.predict(Xa_s))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    a = axes[0]
    vals = [h_oob, h_cv, h_te]
    a.bar(["OOB\n(1 ajuste)", "CV 5 bloques\n(5 ajustes)", "Test\n(76 pacientes)"], vals, color=[AMBER, BLUE, C35], width=0.6)
    for i, v in enumerate(vals):
        a.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    a.set_ylim(0.7, 0.9)
    a.set_ylabel("Exactitud")
    a.set_title(f"Heart · RF de 500 árboles ({t_oob_h:.1f} s vs {t_cv_h:.1f} s)", loc="left", fontsize=9.8, color=INK)
    style(a, "y")
    b = axes[1]
    vals = [a_oob, a_cv, a_te]
    b.bar(["OOB\n(1 ajuste)", "CV 5 bloques\n(5 ajustes)", "Test\n(586 casas)"], np.array(vals) / 1000, color=[AMBER, BLUE, C35], width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v / 1000 + 0.3, f"{v / 1000:.1f}k", ha="center", fontsize=9)
    b.set_ylim(15, 35)
    b.set_ylabel("RMSE (miles USD)")
    b.set_title(f"Ames · RF de 300 árboles ({t_oob_a:.1f} s vs {t_cv_a:.1f} s)", loc="left", fontsize=9.8, color=INK)
    style(b, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "c35_oob_cv.png")
    NUM["oobcv"] = dict(heart=dict(oob=h_oob, cv=h_cv, test=h_te, t_oob=t_oob_h, t_cv=t_cv_h, cv_sd=cvr["test_score"].std()),
                        ames=dict(oob=a_oob, cv=a_cv, test=a_te, t_oob=t_oob_a, t_cv=t_cv_a))
    print("oob_cv", perf_counter() - t0)


# ------------------------------------------------------------------ 7. imágenes (dígitos de scikit-learn)
def desplazar(X, dx=1):
    """Desplaza cada imagen 8×8 dx píxeles a la derecha (rellena con ceros)."""
    im = X.reshape(-1, 8, 8)
    out = np.zeros_like(im)
    out[:, :, dx:] = im[:, :, :-dx]
    return out.reshape(len(X), -1)


def imagenes_fig():
    t0 = perf_counter()
    d = load_digits()
    X, y = d.data / 16.0, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)
    rf = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=RS).fit(Xtr, ytr)
    acc_te = accuracy_score(yte, rf.predict(Xte))
    acc_sh = accuracy_score(yte, rf.predict(desplazar(Xte)))
    perm = np.random.default_rng(RS).permutation(64)           # mezclar los píxeles de forma fija
    rf_p = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS).fit(Xtr[:, perm], ytr)
    acc_mez = accuracy_score(yte, rf_p.predict(Xte[:, perm]))
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.15])
    sub = gs[0].subgridspec(3, 6, wspace=0.1, hspace=0.35)
    for k in range(18):
        ax = fig.add_subplot(sub[k // 6, k % 6])
        ax.imshow(d.images[k], cmap="gray_r")
        ax.set_title(str(d.target[k]), fontsize=8, color=INK, pad=2)
        ax.axis("off")
    fig.text(0.015, 0.99, "Dígitos 8 × 8 = 64 variables (una por píxel)", fontsize=9.8, color=INK, va="top")
    a = fig.add_subplot(gs[1])
    im = a.imshow(rf.feature_importances_.reshape(8, 8), cmap="Greens")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title("Importancia (MDI) de cada píxel", loc="left", fontsize=9.8, color=INK)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    b = fig.add_subplot(gs[2])
    nombres = ["OOB", "Test", "Test con\nimágenes\ndesplazadas\n1 píxel", "Píxeles\nmezclados\n(test)"]
    vals = [rf.oob_score_, acc_te, acc_sh, acc_mez]
    b.bar(nombres, vals, color=[AMBER, C35, RED, GRAY], width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    b.set_ylim(0, 1.08)
    b.set_ylabel("Exactitud")
    b.set_title("RF no sabe qué píxel está al lado de cuál", loc="left", fontsize=9.8, color=INK)
    style(b, "y")
    fig.subplots_adjust(left=0.02, right=0.99, wspace=0.28, bottom=0.2, top=0.86)
    save(fig, "c35_imagenes.png")
    NUM["img"] = dict(oob=rf.oob_score_, test=acc_te, shifted=acc_sh, shuffled=acc_mez, n_tr=len(Xtr), n_te=len(Xte),
                      top_pixeles=[int(i) for i in np.argsort(rf.feature_importances_)[::-1][:5]])
    print("imagenes", perf_counter() - t0)


# ------------------------------------------------------------------ 8. esquema bagging vs random forest
def esquema(ax, titulo, rf):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    col = C35 if rf else BLUE
    ax.text(0, 5.95, titulo, fontsize=10.5, color=INK, fontweight="bold", va="top")
    rng = np.random.default_rng(3)
    for i in range(3):
        yy = 4.0 - i * 1.55
        ax.add_patch(FancyBboxPatch((0.1, yy), 1.9, 0.9, boxstyle="round,pad=0.03", fc="#F0FDFA" if rf else "#EFF6FF", ec=col, lw=1.3))
        ax.text(1.05, yy + 0.45, f"bootstrap {i + 1}", ha="center", va="center", fontsize=8.3)
        ax.add_patch(FancyArrowPatch((2.05, yy + 0.45), (2.7, yy + 0.45), arrowstyle="-|>", mutation_scale=9, color=MUTED))
        ax.add_patch(FancyBboxPatch((2.75, yy - 0.05), 3.4, 1.0, boxstyle="round,pad=0.03", fc="white", ec=col, lw=1.3))
        ax.text(3.0, yy + 0.7, f"árbol {i + 1}", ha="left", va="center", fontsize=8.5, color=col, fontweight="bold")
        elegidas = sorted(rng.choice(8, 3, replace=False)) if rf else list(range(8))
        for j in range(8):
            ax.add_patch(Rectangle((2.95 + j * 0.34, yy + 0.05), 0.28, 0.32, fc=col if j in elegidas else "white",
                                   ec=col, lw=0.9))
        ax.add_patch(FancyArrowPatch((6.2, yy + 0.45), (7.1, 2.7), arrowstyle="-|>", mutation_scale=9, color=MUTED))
    ax.add_patch(FancyBboxPatch((7.2, 2.2), 2.7, 1.0, boxstyle="round,pad=0.03", fc=col, ec=col))
    ax.text(8.55, 2.7, "promedio\n/ voto", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.text(0.1, 0.45, "Cuadritos = variables candidatas en cada nodo" if True else "", fontsize=8.3, color=MUTED)
    ax.text(0.1, 0.05, "todas" if not rf else "solo un subconjunto al azar, distinto en cada nodo", fontsize=8.3, color=col,
            fontweight="bold")


def esquema_fig():
    fig = plt.figure(figsize=(12.5, 2.9))
    gs = fig.add_gridspec(1, 2)
    esquema(fig.add_subplot(gs[0]), "BAGGING con árboles", False)
    esquema(fig.add_subplot(gs[1]), "RANDOM FOREST", True)
    fig.tight_layout(w_pad=1.5)
    save(fig, "c35_esquema.png")


# ------------------------------------------------------------------ 9. comparación árbol / logística / bagging / RF
def comparacion_fig():
    t0 = perf_counter()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    modelos_h = {
        "Árbol": Pipeline([("p", prep(Xh)), ("m", DecisionTreeClassifier(random_state=RS))]),
        "Reg. logística": Pipeline([("p", prep(Xh, True)), ("m", LogisticRegression(max_iter=2000))]),
        "Bagging de\nlogísticas": Pipeline([("p", prep(Xh, True)), ("m", BaggingClassifier(LogisticRegression(max_iter=2000), n_estimators=100, random_state=RS))]),
        "Bagging de\nárboles": Pipeline([("p", prep(Xh)), ("m", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=RS, n_jobs=-1))]),
        "Random Forest": Pipeline([("p", prep(Xh)), ("m", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS))]),
    }
    res_h = {}
    for nm, m in modelos_h.items():
        s = cross_val_score(m, Xh, yh, cv=cv, scoring="roc_auc")
        res_h[nm] = (s.mean(), s.std())
    modelos_a = {
        "Árbol": DecisionTreeRegressor(random_state=RS),
        "Bagging": BaggingRegressor(DecisionTreeRegressor(), n_estimators=300, n_jobs=-1, random_state=RS),
        "Random Forest\n(1/3)": RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS),
    }
    res_a = {}
    pre = prep(Xa_tr).fit(Xa_tr)
    At, As = pre.transform(Xa_tr), pre.transform(Xa_te)
    for nm, m in modelos_a.items():
        tt = perf_counter()
        m.fit(At, ya_tr)
        res_a[nm] = (rmse(ya_te, m.predict(As)), rmse(ya_tr, m.predict(At)), perf_counter() - tt)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = axes[0]
    nombres = list(res_h)
    cols = [GRAY, AMBER, "#FCD9A8", BLUE, C35]
    m_ = [res_h[n][0] for n in nombres]
    s_ = [res_h[n][1] for n in nombres]
    a.bar(nombres, m_, yerr=s_, color=cols, error_kw=dict(ecolor=INK, lw=1, capsize=3), width=0.65)
    for i, (v, s) in enumerate(zip(m_, s_)):
        a.text(i, v + s + 0.006, f"{v:.3f}", ha="center", fontsize=8.8)
    a.set_ylim(0.7, 1.0)
    a.set_ylabel("AUC (validación cruzada 5×3)")
    a.set_title("Heart (303 pacientes): bagging ayuda donde el modelo es inestable", loc="left", fontsize=9.6, color=INK)
    style(a, "y")
    b = axes[1]
    nombres = list(res_a)
    cols = [GRAY, BLUE, C35]
    t_ = [res_a[n][0] / 1000 for n in nombres]
    b.bar(nombres, t_, color=cols, width=0.6)
    for i, n in enumerate(nombres):
        b.text(i, t_[i] + 0.8, f"{t_[i]:.1f}k", ha="center", fontsize=9)
    b.set_ylim(0, 52)
    b.set_ylabel("RMSE en test (miles USD)")
    b.set_title("Ames (2.930 casas)", loc="left", fontsize=9.6, color=INK)
    style(b, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "c35_comparacion.png")
    NUM["cmp_h"] = {k.replace("\n", " "): (round(v[0], 4), round(v[1], 4)) for k, v in res_h.items()}
    NUM["cmp_a"] = {k.replace("\n", " "): (round(v[0]), round(v[1]), round(v[2], 2)) for k, v in res_a.items()}
    print("comparacion", perf_counter() - t0)


if __name__ == "__main__":
    import sys
    pasos = dict(esquema=esquema_fig, bootstrap=bootstrap_fig, varianza=varianza_fig, max_features=max_features_fig,
                 curvas=curvas_fig, importancia=importancia_fig, oob_cv=oob_cv_fig, imagenes=imagenes_fig,
                 comparacion=comparacion_fig)
    elegidos = sys.argv[1:] or list(pasos)
    for k in elegidos:
        pasos[k]()
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
