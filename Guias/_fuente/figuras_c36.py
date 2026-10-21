"""Figuras de la Clase 36 (Boosting). Datos de la clase: Heart.xlsx (clasificación) y ames_housing.csv (regresión),
los mismos de la Clase 35 (profesor Bastian Galasso). Ejecutar con el venv que tiene sklearn, xgboost y lightgbm."""
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[2]
CLASE35 = next(BASE.glob("Clase*35*"))
RS = 42

C36 = "#EA580C"      # color de la clase 36 (naranja)
BLUE = "#2563EB"
GREEN = "#16A34A"
RED = "#DC2626"
PURPLE = "#7C3AED"
GRAY = "#6B7280"
AMBER = "#D97706"


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ datos
def prep(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


def cargar():
    heart = pd.read_excel(CLASE35 / "Heart.xlsx")
    Xh, yh = heart.drop(columns="AHD"), heart["AHD"].map({"No": 0, "Yes": 1})
    hsplit = train_test_split(Xh, yh, test_size=0.25, stratify=yh, random_state=RS)
    ames = pd.read_csv(CLASE35 / "ames_housing.csv").drop(columns=["Unnamed: 0", "Longitude", "Latitude"])
    Xa, ya = ames.drop(columns="Sale_Price"), ames["Sale_Price"]
    asplit = train_test_split(Xa, ya, test_size=0.20, random_state=RS)
    return heart, hsplit, ames, asplit


heart, (Xh_tr, Xh_te, yh_tr, yh_te), ames, (Xa_tr, Xa_te, ya_tr, ya_te) = cargar()

# Ames: una partición de validación DENTRO del entrenamiento para las curvas (el test queda intacto)
Xa_fit, Xa_val, ya_fit, ya_val = train_test_split(Xa_tr, ya_tr, test_size=0.25, random_state=RS)
_pa = prep(Xa_tr).fit(Xa_fit)
A_fit, A_val = _pa.transform(Xa_fit), _pa.transform(Xa_val)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


NUM = {}  # números que se citan en los PDF y el notebook


# ------------------------------------------------------------------ 1. residuos secuenciales
def secuencial():
    d = ames[["Gr_Liv_Area", "Sale_Price"]].copy()
    d = d[d.Gr_Liv_Area < 4000]
    x = d.Gr_Liv_Area.values.reshape(-1, 1)
    y = d.Sale_Price.values / 1000
    grid = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.3, random_state=RS).fit(x, y)
    base = y.mean()
    stg_grid = list(gb.staged_predict(grid))
    stg_x = list(gb.staged_predict(x))
    pasos = [0, 1, 5, 50]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4), sharex=True)
    out = []
    for j, m in enumerate(pasos):
        fg = np.full(300, base) if m == 0 else stg_grid[m - 1]
        fx = np.full(len(y), base) if m == 0 else stg_x[m - 1]
        res = y - fx
        a = axes[0, j]
        a.scatter(x, y, s=4, color=GRAY, alpha=0.35)
        a.plot(grid, fg, color=C36, lw=2.4)
        a.set_title("F₀ = promedio" if m == 0 else f"F con {m} árbol{'es' if m > 1 else ''}", fontsize=10, color=INK,
                    loc="left")
        a.set_ylim(0, 600)
        style(a)
        b = axes[1, j]
        b.scatter(x, res, s=4, color=BLUE, alpha=0.35)
        b.axhline(0, color=INK, lw=0.9)
        b.set_ylim(-250, 450)
        b.text(0.97, 0.93, f"RMSE = {rmse(y, fx):.1f} mil", transform=b.transAxes, ha="right", va="top", fontsize=9.5,
               color=RED, fontweight="bold")
        style(b)
        out.append(rmse(y, fx))
        if j == 0:
            a.set_ylabel("Precio (miles de USD)")
            b.set_ylabel("Residuo y − F")
        b.set_xlabel("Superficie habitable (pies²)")
    fig.suptitle("Cada árbol se ajusta a lo que el modelo anterior no explicó: los residuos se achican (η = 0.3, profundidad 2)",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "c36_secuencial.png")
    NUM["secuencial_rmse"] = [round(v, 1) for v in out]


# ------------------------------------------------------------------ 2. bagging vs boosting
def esquema(ax, titulo, secuencial_):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0, 5.9, titulo, fontsize=10.5, color=INK, fontweight="bold", va="top")
    col = C36 if secuencial_ else BLUE
    if not secuencial_:
        for i in range(3):
            yy = 4.0 - i * 1.55
            ax.add_patch(FancyBboxPatch((0.2, yy), 2.2, 0.9, boxstyle="round,pad=0.03", fc="#EFF6FF", ec=col, lw=1.3))
            ax.text(1.3, yy + 0.45, f"muestra {i + 1}", ha="center", va="center", fontsize=8.5)
            ax.add_patch(FancyBboxPatch((3.4, yy), 2.0, 0.9, boxstyle="round,pad=0.03", fc="white", ec=col, lw=1.3))
            ax.text(4.4, yy + 0.45, f"árbol {i + 1}", ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
            ax.add_patch(FancyArrowPatch((2.45, yy + 0.45), (3.35, yy + 0.45), arrowstyle="-|>", mutation_scale=9, color=MUTED))
            ax.add_patch(FancyArrowPatch((5.45, yy + 0.45), (6.6, 2.7), arrowstyle="-|>", mutation_scale=9, color=MUTED))
        ax.add_patch(FancyBboxPatch((6.7, 2.2), 3.2, 1.0, boxstyle="round,pad=0.03", fc=col, ec=col))
        ax.text(8.3, 2.7, "promedio\n/ voto", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.2, 0.15, "En paralelo e independientes → reduce varianza", fontsize=8.8, color=MUTED)
    else:
        xs = [0.2, 2.9, 5.6]
        for i, xx in enumerate(xs):
            ax.add_patch(FancyBboxPatch((xx, 3.3), 2.2, 1.0, boxstyle="round,pad=0.03", fc="white", ec=col, lw=1.5))
            ax.text(xx + 1.1, 3.8, f"árbol {i + 1}", ha="center", va="center", fontsize=9, color=col, fontweight="bold")
            ax.text(xx + 1.1, 2.55, "errores" if i < 2 else "…", ha="center", va="center", fontsize=8.5, color=RED)
            if i < 2:
                ax.add_patch(FancyArrowPatch((xx + 2.25, 3.8), (xs[i + 1] - 0.05, 3.8), arrowstyle="-|>",
                                             mutation_scale=10, color=col, lw=1.4))
        ax.add_patch(FancyBboxPatch((0.2, 0.85), 9.4, 0.8, boxstyle="round,pad=0.03", fc=col, ec=col))
        ax.text(4.9, 1.25, "F = F₀ + η·h₁ + η·h₂ + …", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(0.2, 5.0, "Todos ven el dataset completo", fontsize=8.8, color=MUTED)
        ax.text(0.2, 0.15, "En secuencia, cada uno corrige al anterior → reduce sesgo y error", fontsize=8.8, color=MUTED)


def bagging_boosting():
    t0 = perf_counter()
    rf = RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS).fit(A_fit, ya_fit)
    cum = np.cumsum([e.predict(A_val) for e in rf.estimators_], axis=0) / np.arange(1, 301).reshape(-1, 1)
    rf_curve = [rmse(ya_val, c) for c in cum]
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RS).fit(A_fit, ya_fit)
    gb_curve = [rmse(ya_val, p) for p in gb.staged_predict(A_val)]
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_depth=3, random_state=RS).fit(A_fit, ya_fit)
    t_curve = rmse(ya_val, tree.predict(A_val))
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45])
    esquema(fig.add_subplot(gs[0]), "BAGGING / Random Forest", False)
    esquema(fig.add_subplot(gs[1]), "BOOSTING", True)
    a = fig.add_subplot(gs[2])
    n = np.arange(1, 301)
    a.plot(n, rf_curve, color=BLUE, lw=2.2, label="Random Forest (árboles profundos)")
    a.plot(n, gb_curve, color=C36, lw=2.4, label="Gradient Boosting (profundidad 3, η = 0.1)")
    a.axhline(t_curve, color=GRAY, ls=":", lw=1.4)
    a.text(300, t_curve + 800, "un árbol de profundidad 3", ha="right", fontsize=8.3, color=GRAY)
    a.set_ylim(15000, 50000)
    a.set_xlabel("Número de árboles")
    a.set_ylabel("RMSE en validación (USD)")
    a.set_title("Datos de la clase (Ames): más árboles, distinto efecto", loc="left", fontsize=9.8, color=INK)
    a.legend(frameon=False, fontsize=8.3, loc="upper right")
    style(a)
    fig.tight_layout(w_pad=1.5)
    save(fig, "c36_bagging_boosting.png")
    NUM["bb"] = dict(rf1=rf_curve[0], rf300=rf_curve[-1], gb1=gb_curve[0], gb300=gb_curve[-1], tree3=t_curve,
                     gb_best=int(np.argmin(gb_curve)) + 1, gb_min=min(gb_curve), rf_min=min(rf_curve))
    print("bagging_boosting", perf_counter() - t0)


# ------------------------------------------------------------------ 3. AdaBoost a mano
def adaboost_mano(X, y, rondas):
    """AdaBoost con stumps. y en {-1,+1}. Devuelve lista con pesos, error, alpha de cada ronda."""
    n = len(y)
    w = np.full(n, 1 / n)
    hist = []
    for m in range(rondas):
        st = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        pred = st.predict(X)
        err = float(np.sum(w[pred != y]) / np.sum(w))
        alpha = 0.5 * np.log((1 - err) / err)
        hist.append(dict(w=w.copy(), pred=pred, err=err, alpha=alpha, stump=st))
        w = w * np.exp(-alpha * y * pred)
        w = w / w.sum()
    return hist


def adaboost_fig():
    d = heart[["Age", "MaxHR", "AHD"]].sample(70, random_state=3).reset_index(drop=True)
    X = d[["Age", "MaxHR"]].values.astype(float)
    y = np.where(d.AHD == "Yes", 1, -1)
    hist = adaboost_mano(X, y, 30)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5), sharex=True, sharey=True)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200),
                         np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 200))
    G = np.c_[gx.ravel(), gy.ravel()]
    colores = np.where(y == 1, RED, BLUE)
    for j in range(3):
        h = hist[j]
        a = axes[j]
        a.scatter(X[:, 0], X[:, 1], s=h["w"] * 70 * 380, color=colores, alpha=0.8, edgecolor="white", lw=0.6)
        mal = h["pred"] != y
        a.scatter(X[mal, 0], X[mal, 1], s=h["w"][mal] * 70 * 380 + 60, facecolor="none", edgecolor="#F4B64E", lw=1.8)
        st = h["stump"].tree_
        thr, f = st.threshold[0], st.feature[0]
        if f == 0:
            a.axvline(thr, color=C36, lw=2.2)
        else:
            a.axhline(thr, color=C36, lw=2.2)
        a.set_title(f"Ronda {j + 1}: error ε = {h['err']:.3f}\nα = {h['alpha']:.2f}", fontsize=9.6, color=INK, loc="left")
        style(a)
    F = sum(h["alpha"] * h["stump"].predict(G) for h in hist)
    a = axes[3]
    a.contourf(gx, gy, F.reshape(gx.shape), levels=[-1e9, 0, 1e9], colors=["#DBEAFE", "#FEE2E2"], alpha=0.9)
    a.scatter(X[:, 0], X[:, 1], s=22, color=colores, edgecolor="white", lw=0.5)
    Ftr = sum(h["alpha"] * h["stump"].predict(X) for h in hist)
    acc = float(np.mean(np.sign(Ftr) == y))
    a.set_title(f"Voto ponderado de 30 stumps\nexactitud en estos 70 = {acc:.2f}", fontsize=9.6, color=INK, loc="left")
    a.set_xlim(gx.min(), gx.max())
    a.set_ylim(gy.min(), gy.max())
    for a in axes:
        a.set_xlabel("Edad")
    axes[0].set_ylabel("Frecuencia cardíaca máx.")
    fig.text(0.01, -0.04, "Tamaño del punto = peso al comenzar la ronda · borde amarillo = error del stump · rojo = enfermedad (AHD = Yes), azul = No",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "c36_adaboost.png")
    NUM["ada"] = dict(err=[round(h["err"], 3) for h in hist[:3]], alpha=[round(h["alpha"], 2) for h in hist[:3]],
                      acc30=acc, n_err1=int((hist[0]["pred"] != y).sum()))


# ------------------------------------------------------------------ 4. pérdida y gradiente
def gradiente_fig():
    F = np.linspace(2, 22, 300)
    y0 = 12
    r = y0 - F
    cuad = 0.5 * r ** 2
    absl = np.abs(r)
    delta = 3.0
    hub = np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))
    g_cuad = r
    g_abs = np.sign(r)
    g_hub = np.clip(r, -delta, delta)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.3))
    a1.plot(F, cuad, color=C36, lw=2.3, label="cuadrática ½(y − F)²")
    a1.plot(F, absl, color=BLUE, lw=2.3, label="absoluta |y − F|")
    a1.plot(F, hub, color=GREEN, lw=2.3, label="Huber (δ = 3)")
    a1.axvline(y0, color=INK, lw=0.8, ls=":")
    a1.text(y0 + 0.2, 20, "y = 12", fontsize=8.5)
    a1.set_xlabel("Predicción actual F")
    a1.set_ylabel("Pérdida L(y, F)")
    a1.set_ylim(0, 40)
    a1.set_title("Qué significa «equivocarse» lo define la pérdida", loc="left", fontsize=9.8, color=INK)
    a1.legend(frameon=True, framealpha=1, edgecolor="white", fontsize=8.3, loc="upper center")
    style(a1)
    a2.plot(F, g_cuad, color=C36, lw=2.3, label="cuadrática: residuo y − F")
    a2.plot(F, g_abs * 3, color=BLUE, lw=2.3, label="absoluta: signo del residuo (×3)")
    a2.plot(F, g_hub, color=GREEN, lw=2.3, label="Huber: residuo recortado en ±3")
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xlabel("Predicción actual F")
    a2.set_ylabel("Gradiente negativo −∂L/∂F")
    a2.set_title("Lo que el siguiente árbol intenta predecir", loc="left", fontsize=9.8, color=INK)
    a2.legend(frameon=False, fontsize=8.3, loc="upper right")
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "c36_gradiente.png")


# ------------------------------------------------------------------ 5. learning rate vs n_estimators
def lr_nest():
    t0 = perf_counter()
    lrs = [1.0, 0.3, 0.1, 0.03]
    cols = [RED, AMBER, C36, GREEN]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    best = []
    for lr, c in zip(lrs, cols):
        gb = GradientBoostingRegressor(n_estimators=800, max_depth=3, learning_rate=lr, random_state=RS).fit(A_fit, ya_fit)
        curve = [rmse(ya_val, p) for p in gb.staged_predict(A_val)]
        k = int(np.argmin(curve))
        best.append((lr, k + 1, curve[k], curve[-1]))
        a1.plot(np.arange(1, 801), curve, color=c, lw=2.1, label=f"η = {lr}")
        a1.plot(k + 1, curve[k], marker="o", color=c, ms=7, mec="white")
    a1.set_ylim(17000, 40000)
    a1.set_xlabel("Número de árboles")
    a1.set_ylabel("RMSE en validación (USD)")
    a1.set_title("Menor η: curva más lenta, pero llega más abajo (● = mejor n)", loc="left", fontsize=9.8, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a1)
    xs = np.arange(len(lrs))
    a2.bar(xs, [b[1] for b in best], color=cols, alpha=0.85)
    for i, b in enumerate(best):
        a2.text(i, b[1] + 15, f"{b[1]} árboles\nRMSE {b[2] / 1000:.1f}k", ha="center", fontsize=8.3, color=INK)
    a2.set_xticks(xs, [f"η = {b[0]}" for b in best])
    a2.set_ylim(0, 1050)
    a2.set_ylabel("Árboles que minimizan la validación")
    a2.set_title("η bajo: hacen falta más árboles", loc="left", fontsize=9.8, color=INK)
    style(a2, "y")
    fig.tight_layout(w_pad=2.5)
    save(fig, "c36_lr_nest.png")
    NUM["lr"] = [dict(lr=b[0], n=b[1], best=round(b[2]), last=round(b[3])) for b in best]
    print("lr_nest", perf_counter() - t0)


# ------------------------------------------------------------------ 6. XGBoost / LightGBM
def arbol_esquema(ax, titulo, niveles):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0, 5.95, titulo, fontsize=10.2, color=INK, fontweight="bold", va="top")
    nodos = {"R": (5, 4.7), "L": (2.6, 3.4), "Rr": (7.4, 3.4), "LL": (1.3, 2.1), "LR": (3.9, 2.1), "RL": (6.1, 2.1),
             "RR": (8.7, 2.1), "RLL": (5.4, 0.85), "RLR": (6.8, 0.85)}
    aristas = [("R", "L"), ("R", "Rr"), ("L", "LL"), ("L", "LR"), ("Rr", "RL"), ("Rr", "RR")]
    if niveles == "hoja":
        usados = ["R", "L", "Rr", "RL", "RR", "RLL", "RLR"]
        aristas = [("R", "L"), ("R", "Rr"), ("Rr", "RL"), ("Rr", "RR"), ("RL", "RLL"), ("RL", "RLR")]
        hojas = {"L", "RR", "RLL", "RLR"}
    else:
        usados = ["R", "L", "Rr", "LL", "LR", "RL", "RR"]
        hojas = {"LL", "LR", "RL", "RR"}
    nodos["RL"] = nodos["RL"] if niveles == "hoja" else nodos["RL"]
    for a, b in aristas:
        ax.add_patch(FancyArrowPatch((nodos[a][0], nodos[a][1] - 0.22), (nodos[b][0], nodos[b][1] + 0.22), arrowstyle="-",
                                     color=MUTED, lw=1.2))
    for k in usados:
        x, y = nodos[k]
        leaf = k in hojas
        ax.add_patch(Circle((x, y), 0.34, fc=(C36 if leaf else "white"), ec=C36, lw=1.6))
    ax.text(5, 0.05, "Expande un nivel completo" if niveles == "nivel" else "Expande la hoja con mayor ganancia",
            ha="center", fontsize=8.8, color=MUTED)


def xgb_lgbm():
    t0 = perf_counter()
    pre = _pa
    mod = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                       early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=RS)
    mod.fit(A_fit, ya_fit, eval_set=[(A_fit, ya_fit), (A_val, ya_val)], verbose=False)
    ev = mod.evals_result()
    tr, va = ev["validation_0"]["rmse"], ev["validation_1"]["rmse"]
    bi = mod.best_iteration
    fig = plt.figure(figsize=(12.5, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    arbol_esquema(fig.add_subplot(gs[0]), "Por nivel (XGBoost)", "nivel")
    arbol_esquema(fig.add_subplot(gs[1]), "Por hoja (LightGBM)", "hoja")
    a = fig.add_subplot(gs[2])
    a.plot(tr, color=GRAY, lw=2, label="entrenamiento")
    a.plot(va, color=C36, lw=2.3, label="validación")
    a.axvline(bi, color=GREEN, ls="--", lw=1.4)
    a.annotate(f"early stopping:\nmejor ronda {bi}", xy=(bi, va[bi]), xytext=(bi + 80, va[bi] + 18000), fontsize=8.6,
               color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    a.set_ylim(8000, 100000)
    a.set_yscale("log")
    a.set_xlabel("Ronda (árbol)")
    a.set_ylabel("RMSE (USD, escala log)")
    a.set_title(f"XGBoost en Ames: se detiene a las {len(va)} rondas de {1500}", loc="left", fontsize=9.8, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(a)
    fig.tight_layout(w_pad=1.5)
    save(fig, "c36_xgb_lgbm.png")
    NUM["es"] = dict(best=bi, parada=len(va), val=va[bi], train=tr[bi])
    print("xgb_lgbm", perf_counter() - t0)


# ------------------------------------------------------------------ 7. comparación
def modelos_clf():
    return {
        "AdaBoost": AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=2, random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", n_jobs=1, random_state=RS),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbosity=-1, n_jobs=1, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RS),
    }


def modelos_reg():
    return {
        "AdaBoost": AdaBoostRegressor(n_estimators=350, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=0.05, max_depth=2, random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, random_state=RS),
        "LightGBM": LGBMRegressor(n_estimators=350, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbosity=-1, n_jobs=-1, random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS),
    }


def comparacion():
    t0 = perf_counter()
    Xh_all = heart.drop(columns="AHD")                      # los 303 pacientes, en el orden original
    yh_all = heart["AHD"].map({"No": 0, "Yes": 1})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    res_h = {}
    for nm, m in modelos_clf().items():
        pipe = Pipeline([("p", prep(Xh_all)), ("m", m)])
        s = cross_val_score(pipe, Xh_all, yh_all, cv=cv, scoring="roc_auc", n_jobs=-1)
        res_h[nm] = (s.mean(), s.std())
    res_a = {}
    for nm, m in modelos_reg().items():
        pipe = Pipeline([("p", prep(Xa_tr)), ("m", m)])
        t = perf_counter()
        pipe.fit(Xa_tr, ya_tr)
        dt = perf_counter() - t
        res_a[nm] = (rmse(ya_te, pipe.predict(Xa_te)), rmse(ya_tr, pipe.predict(Xa_tr)), dt)
    nombres = list(res_h)
    cols = [RED, AMBER, C36, GREEN, BLUE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    m_ = [res_h[n][0] for n in nombres]
    s_ = [res_h[n][1] for n in nombres]
    a1.barh(nombres[::-1], m_[::-1], xerr=s_[::-1], color=cols[::-1], alpha=0.9, error_kw=dict(ecolor=INK, lw=1, capsize=3))
    for i, (v, s) in enumerate(zip(m_[::-1], s_[::-1])):
        a1.text(v + s + 0.004, i, f"{v:.3f}", va="center", fontsize=8.6)
    a1.set_xlim(0.8, 0.97)
    a1.set_xlabel("AUC (validación cruzada 5×3)")
    a1.set_title("Heart (303 pacientes): todos muy parecidos", loc="left", fontsize=9.8, color=INK)
    style(a1, "x")
    t_ = [res_a[n][0] for n in nombres]
    a2.barh(nombres[::-1], t_[::-1], color=cols[::-1], alpha=0.9)
    for i, n in enumerate(nombres[::-1]):
        a2.text(res_a[n][0] + 400, i, f"${res_a[n][0] / 1000:.1f}k · {res_a[n][2]:.1f}s", va="center", fontsize=8.6)
    a2.set_xlim(0, 56000)
    a2.set_xlabel("RMSE en test (USD)")
    a2.set_title("Ames (2.930 casas): AdaBoost queda atrás", loc="left", fontsize=9.8, color=INK)
    style(a2, "x")
    fig.tight_layout(w_pad=3)
    save(fig, "c36_comparacion.png")
    NUM["cmp_h"] = {k: (round(v[0], 4), round(v[1], 4)) for k, v in res_h.items()}
    NUM["cmp_a"] = {k: (round(v[0]), round(v[1]), round(v[2], 2)) for k, v in res_a.items()}
    print("comparacion", perf_counter() - t0)


if __name__ == "__main__":
    secuencial()
    bagging_boosting()
    adaboost_fig()
    gradiente_fig()
    lr_nest()
    xgb_lgbm()
    comparacion()
    import json
    print(json.dumps(NUM, indent=1, default=float))
    print("ok")
