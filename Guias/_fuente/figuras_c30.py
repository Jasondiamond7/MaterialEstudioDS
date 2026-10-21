"""Figuras de la Clase 30 (Preparación para Machine Learning: variables, datos y validación).
Datos: los de la profesora Laura Sánchez, que son SINTÉTICOS y se generan en el propio notebook de la clase
(viviendas con np.random.seed(42) y fraude con make_classification). No hay archivos externos: aquí se
reproducen exactamente, en el mismo orden de llamadas al generador aleatorio.
Ejecutar con el venv que tiene scikit-learn:  python figuras_c30.py   (o con un nombre: python figuras_c30.py kfold)."""
import json
import warnings
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 42
C30 = "#7C3AED"      # color de la clase 30 (violeta)
C30_BG = "#F5F3FF"
BLUE = "#2563EB"
RED = "#DC2626"
GRAY = "#6B7280"
AMBER = "#D97706"
TEAL = "#0D9488"
GREEN = "#059669"
NUM = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ datos (idénticos al notebook de la profesora)
np.random.seed(42)
n = 300
viviendas = pd.DataFrame({
    "superficie_m2": np.random.normal(90, 25, n).round(1),
    "n_habitaciones": np.random.choice([1, 2, 3, 4, 5], n),
    "barrio": np.random.choice(["Centro", "Norte", "Sur", "Este"], n),
    "antiguedad_años": np.random.randint(0, 50, n),
    "tiene_ascensor": np.random.choice(["Sí", "No"], n),
})
viviendas["precio_clp"] = (2_200_000 * viviendas["superficie_m2"] + 8_000_000 * viviendas["n_habitaciones"]
                           - 900_000 * viviendas["antiguedad_años"] + np.random.normal(0, 15_000_000, n)).round(0)
viviendas_sucias = viviendas.copy()
faltantes = np.random.choice(viviendas_sucias.index, size=20, replace=False)
viviendas_sucias.loc[faltantes, "antiguedad_años"] = np.nan

X = viviendas[["superficie_m2", "n_habitaciones", "antiguedad_años"]]
y = viviendas["precio_clp"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RS)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RS)

X_reg = X.copy()
X_reg["ruido_aleatorio"] = np.random.default_rng(RS).normal(0, 1, len(X_reg))
y_reg = y / 1_000_000
Xr_tr, Xr_tmp, yr_tr, yr_tmp = train_test_split(X_reg, y_reg, test_size=0.30, random_state=RS)
Xr_val, Xr_te, yr_val, yr_te = train_test_split(Xr_tmp, yr_tmp, test_size=0.50, random_state=RS)

Xf, yf = make_classification(n_samples=2000, n_features=5, weights=[0.95, 0.05], random_state=RS)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.25, stratify=yf, random_state=RS)


# ------------------------------------------------------------------ 0. esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 2.7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 2.7)
    ax.axis("off")
    cajas = [("1 · Variables", "¿qué tipo es\ncada columna?"), ("2 · Calidad", "¿faltan datos?\n¿son confiables?"),
             ("3 · División", "entrenar, validar\ny probar"), ("4 · Regularización", "¿el modelo es\ndemasiado complejo?"),
             ("5 · K-Fold", "¿el resultado\nes estable?"), ("6 · Métricas", "¿mide lo que\nimporta?")]
    cols = [GRAY, AMBER, C30, BLUE, TEAL, RED]
    w = 1.85
    for i, ((t, q), c) in enumerate(zip(cajas, cols)):
        x = 0.1 + i * 2.07
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.35, boxstyle="round,pad=0.04", fc="white", ec=c, lw=1.8))
        ax.text(x + w / 2, 1.78, t, ha="center", va="center", fontsize=10.2, color=c, fontweight="bold")
        ax.text(x + w / 2, 1.2, q, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.3)
        if i < len(cajas) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 1.42), (x + 2.07 - 0.03, 1.42), arrowstyle="-|>", mutation_scale=11, color=GRAY))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 0.42, boxstyle="round,pad=0.02", fc=C30_BG, ec=C30, lw=1.2))
    ax.text(6.25, 0.33, "Todo el preprocesamiento (imputar, escalar, codificar) se APRENDE solo con los datos de entrenamiento: Pipeline",
            ha="center", va="center", fontsize=9.2, color=C30, fontweight="bold")
    save(fig, "c30_esquema.png")


# ------------------------------------------------------------------ 1. calidad de datos
def calidad_fig():
    t0 = perf_counter()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    a = axes[0]
    faltan = viviendas_sucias.isna().sum()
    a.barh(faltan.index, faltan.values, color=[AMBER if v > 0 else "#D1D5DB" for v in faltan.values], height=0.6)
    for i, v in enumerate(faltan.values):
        a.text(v + 0.4, i, str(v), va="center", fontsize=9)
    a.set_xlim(0, 26)
    a.invert_yaxis()
    a.set_xlabel("valores faltantes (de 300 filas)")
    a.set_title("Completitud: 20 de 300 sin antigüedad (6.7 %)", loc="left", fontsize=9.6, color=INK)
    style(a, "x")

    b = axes[1]
    real = viviendas["antiguedad_años"]
    parcial = viviendas_sucias["antiguedad_años"]
    imp = parcial.fillna(parcial.mean())
    bins = np.arange(0, 51, 5)
    b.hist(parcial.dropna(), bins=bins, color="#D1D5DB", ec="white", label=f"con faltantes (n = 280)")
    b.hist(imp[parcial.isna()], bins=bins, color=AMBER, ec="white", label="20 valores imputados con la media")
    b.set_xlabel("antigüedad (años)")
    b.set_ylabel("viviendas")
    b.legend(fontsize=8, frameon=False, loc="upper right")
    b.set_title("Imputar con la media apila 20 filas en un valor", loc="left", fontsize=9.6, color=INK)
    style(b, "y")
    NUM["sd_antes"] = round(float(parcial.std()), 2)
    NUM["sd_despues"] = round(float(imp.std()), 2)
    NUM["media_antig"] = round(float(parcial.mean()), 2)
    NUM["sd_real"] = round(float(real.std()), 2)

    c = axes[2]
    ing = np.random.default_rng(7).lognormal(mean=13.4, sigma=0.55, size=2000) / 1e6   # SINTÉTICO, solo ilustra el sesgo
    media, mediana = ing.mean(), np.median(ing)
    c.hist(ing, bins=60, color="#DDD6FE", ec="white")
    c.axvline(media, color=RED, lw=2, label=f"media = {media:.2f}")
    c.axvline(mediana, color=C30, lw=2, label=f"mediana = {mediana:.2f}")
    c.set_xlim(0, 7)
    c.set_xlabel("ingreso mensual (millones de CLP, simulado)")
    c.legend(fontsize=8.5, frameon=False)
    c.set_title("Con asimetría y atípicos: mediana", loc="left", fontsize=9.6, color=INK)
    style(c, "y")
    NUM["ingreso_media"], NUM["ingreso_mediana"] = round(float(media), 2), round(float(mediana), 2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "c30_calidad.png")
    print("calidad", perf_counter() - t0)


# ------------------------------------------------------------------ 2. división y sobreajuste
def division_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 10)
    a.set_ylim(0, 4.6)
    a.axis("off")
    a.text(0, 4.35, "300 viviendas → 210 / 45 / 45 (70 / 15 / 15)", fontsize=10, color=INK, fontweight="bold")
    tramos = [(0, 7, "Entrenamiento\n210", C30, "aprende los\nparámetros"), (7, 8.5, "Validación\n45", BLUE, "elige alpha,\nmodelo"),
              (8.5, 10, "Prueba\n45", RED, "veredicto\nfinal")]
    for x0, x1, t, c, rol in tramos:
        a.add_patch(Rectangle((x0, 2.55), x1 - x0 - 0.06, 1.1, fc=c, ec="white"))
        a.text((x0 + x1) / 2 - 0.03, 3.1, t, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        a.text((x0 + x1) / 2 - 0.03, 2.0, rol, ha="center", va="top", color=c, fontsize=8.4)
    a.add_patch(FancyArrowPatch((0, 1.15), (10, 1.15), arrowstyle="-|>", mutation_scale=12, color=GRAY))
    a.text(0, 0.8, "tiempo de desarrollo", fontsize=8.4, color=GRAY)
    a.text(10, 0.8, "se mira UNA vez", fontsize=8.4, color=RED, ha="right", fontweight="bold")
    a.text(0, 0.1, "Las proporciones son una referencia, no una regla: con pocos datos se usa validación cruzada.", fontsize=8.4, color=GRAY)

    b = fig.add_subplot(gs[1])
    prof = list(range(1, 13))
    tr, va = [], []
    for d in prof:
        arbol = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
        tr.append(arbol.score(X_train, y_train))
        va.append(arbol.score(X_val, y_val))
    b.plot(prof, tr, "-o", color=C30, ms=4, lw=2, label="entrenamiento")
    b.plot(prof, va, "-o", color=BLUE, ms=4, lw=2, label="validación")
    mejor = int(np.argmax(va))
    b.axvline(prof[mejor], color=GREEN, ls="--", lw=1.4)
    b.text(prof[mejor] + 0.15, 0.55, f"mejor validación\nprofundidad {prof[mejor]}", fontsize=8.4, color=GREEN)
    b.annotate("la brecha crece:\nmemoriza ruido", xy=(10, 0.9), xytext=(7.3, 0.35), fontsize=8.4, color=RED,
               arrowprops=dict(arrowstyle="->", color=RED))
    b.set_xlabel("complejidad del modelo (profundidad del árbol)")
    b.set_ylabel("R²")
    b.set_ylim(0.2, 1.03)
    b.legend(fontsize=8.5, frameon=False, loc="lower right")
    b.set_title("Sobreajuste en las viviendas de la clase", loc="left", fontsize=9.6, color=INK)
    style(b)
    NUM["arbol_train"] = [round(float(v), 3) for v in tr]
    NUM["arbol_val"] = [round(float(v), 3) for v in va]
    NUM["arbol_mejor_prof"] = prof[mejor]
    fig.tight_layout(w_pad=2.5)
    save(fig, "c30_division.png")
    print("division", perf_counter() - t0)


# ------------------------------------------------------------------ 3. regularización y estandarización
def regularizacion_fig():
    t0 = perf_counter()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    nombres = list(X_reg.columns)
    mods = {"Sin regularizar": make_pipeline(StandardScaler(), Ridge(alpha=1e-9)),
            "Ridge (alpha=10)": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "Lasso (alpha=2)": make_pipeline(StandardScaler(), Lasso(alpha=2, max_iter=10000))}
    coefs = {}
    for k, m in mods.items():
        m.fit(Xr_tr, yr_tr)
        coefs[k] = m[-1].coef_
    NUM["coef_std"] = {k: [round(float(v), 3) for v in c] for k, c in coefs.items()}
    a = axes[0]
    w = 0.26
    for i, (k, c) in enumerate(coefs.items()):
        a.bar(np.arange(4) + (i - 1) * w, c, width=w, color=[GRAY, BLUE, C30][i], label=k)
    a.set_xticks(range(4))
    a.set_xticklabels(["superficie", "habitaciones", "antigüedad", "ruido"], fontsize=8.4)
    a.axhline(0, color=INK, lw=0.8)
    a.legend(fontsize=8, frameon=False)
    a.set_ylabel("coeficiente (variables estandarizadas)")
    a.set_title("Lasso lleva el ruido a cero; Ridge solo lo achica", loc="left", fontsize=9.6, color=INK)
    style(a, "y")

    b = axes[1]
    alphas = np.logspace(-2, 1.9, 40)
    rutas = np.array([make_pipeline(StandardScaler(), Lasso(alpha=al, max_iter=50000)).fit(Xr_tr, yr_tr)[-1].coef_ for al in alphas])
    cols = [C30, BLUE, AMBER, RED]
    for j in range(4):
        b.plot(alphas, rutas[:, j], color=cols[j], lw=2, label=nombres[j])
    b.set_xscale("log")
    b.axhline(0, color=INK, lw=0.8)
    b.set_xlabel("alpha (más castigo →)")
    b.set_ylabel("coeficiente estandarizado")
    b.legend(fontsize=8, frameon=False, loc="upper right")
    b.set_title("Camino de Lasso: se apagan de a una", loc="left", fontsize=9.6, color=INK)
    style(b)
    orden = []
    for j in range(4):
        cero = np.where(np.abs(rutas[:, j]) < 1e-9)[0]
        orden.append((nombres[j], round(float(alphas[cero[0]]), 3) if len(cero) else None))
    NUM["alpha_cero"] = orden

    c = axes[2]
    Xs_tr, Xs_val = Xr_tr.copy(), Xr_val.copy()
    Xs_tr["superficie_m2"] /= 100
    Xs_val["superficie_m2"] /= 100
    res = {}
    for etiqueta, (A, B) in {"superficie en m²": (Xr_tr, Xr_val), "superficie en\ncentenas de m²": (Xs_tr, Xs_val)}.items():
        raw = Ridge(alpha=30).fit(A, yr_tr)
        std = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(A, yr_tr)
        res[etiqueta] = (raw.score(B, yr_val), std.score(B, yr_val))
    xs = np.arange(2)
    c.bar(xs - 0.18, [v[0] for v in res.values()], width=0.34, color=RED, label="sin estandarizar")
    c.bar(xs + 0.18, [v[1] for v in res.values()], width=0.34, color=C30, label="con estandarizar")
    for i, v in enumerate(res.values()):
        c.text(i - 0.18, v[0] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=8.6)
        c.text(i + 0.18, v[1] + 0.02, f"{v[1]:.2f}", ha="center", fontsize=8.6)
    c.set_xticks(xs)
    c.set_xticklabels(list(res), fontsize=8.6)
    c.set_ylim(0, 1.05)
    c.set_ylabel("R² en validación (Ridge, alpha = 30)")
    c.legend(fontsize=8, frameon=False, loc="upper right")
    c.set_title("Cambiar las unidades cambia el modelo", loc="left", fontsize=9.6, color=INK)
    style(c, "y")
    NUM["unidades_r2"] = {k: [round(float(a_), 3), round(float(b_), 3)] for k, (a_, b_) in res.items()}
    fig.tight_layout(w_pad=2.5)
    save(fig, "c30_regularizacion.png")
    print("regularizacion", perf_counter() - t0)


# ------------------------------------------------------------------ 4. K-Fold
def kfold_fig():
    t0 = perf_counter()
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=10))
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    puntajes = cross_val_score(modelo, X_dev, y_dev, cv=kf, scoring="r2")
    NUM["kfold_dev"] = [round(float(v), 3) for v in puntajes]
    NUM["kfold_dev_media"], NUM["kfold_dev_sd"] = round(float(puntajes.mean()), 3), round(float(puntajes.std()), 3)
    # lo que hace el notebook de la profesora (usa las 300 filas, incluida la prueba)
    p300 = cross_val_score(modelo, X, y, cv=kf, scoring="r2")
    NUM["kfold_300"] = [round(float(v), 3) for v in p300]
    NUM["kfold_300_media"], NUM["kfold_300_sd"] = round(float(p300.mean()), 3), round(float(p300.std()), 3)
    modelo.fit(X_dev, y_dev)
    NUM["r2_prueba"] = round(float(modelo.score(X_test, y_test)), 3)

    # R² de 300 particiones simples entrenamiento/validación (85/15) para ver cuánto depende de la partición
    simples = []
    for s in range(300):
        a_, b_, c_, d_ = train_test_split(X_dev, y_dev, test_size=0.18, random_state=s)
        simples.append(make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(a_, c_).score(b_, d_))
    simples = np.array(simples)
    NUM["simple_min_max"] = [round(float(simples.min()), 3), round(float(simples.max()), 3)]
    NUM["simple_sd"] = round(float(simples.std()), 3)

    fig = plt.figure(figsize=(12.5, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 0.95])
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 10.6)
    a.set_ylim(0, 6.5)
    a.axis("off")
    a.text(0, 6.15, "K-Fold (K = 5) sobre las 255 viviendas de desarrollo", fontsize=9.6, color=INK, fontweight="bold")
    for f in range(5):
        for j in range(5):
            val = (f == j)
            a.add_patch(Rectangle((1.1 + j * 1.45, 4.75 - f * 0.85), 1.38, 0.68, fc=BLUE if val else "#DDD6FE", ec="white"))
        a.text(0.95, 4.75 - f * 0.85 + 0.34, f"Pliegue {f + 1}", ha="right", va="center", fontsize=8.3, color=INK)
        a.text(1.1 + f * 1.45 + 0.69, 4.75 - f * 0.85 + 0.34, "valida", ha="center", va="center", color="white", fontsize=7.8, fontweight="bold")
    a.add_patch(Rectangle((1.1, 0.0), 7.18, 0.6, fc=RED, ec="white"))
    a.text(4.7, 0.3, "Prueba (45): fuera de todo el K-Fold", ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
    a.text(1.1, 1.0, "los 4 pliegues claros entrenan; se promedian los 5 resultados", fontsize=8.4, color=GRAY)

    b = fig.add_subplot(gs[1])
    b.bar(range(1, 6), puntajes, color=BLUE, width=0.62)
    b.axhline(puntajes.mean(), color=C30, lw=2)
    b.axhspan(puntajes.mean() - puntajes.std(), puntajes.mean() + puntajes.std(), color=C30, alpha=0.15)
    for i, v in enumerate(puntajes):
        b.text(i + 1, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.4)
    b.set_ylim(0.8, 0.98)
    b.set_xlabel("pliegue de validación")
    b.set_ylabel("R²")
    b.set_title(f"media {puntajes.mean():.3f} ± {puntajes.std():.3f}", loc="left", fontsize=9.6, color=INK)
    style(b, "y")

    c = fig.add_subplot(gs[2])
    c.hist(simples, bins=24, color="#D1D5DB", ec="white")
    c.axvline(puntajes.mean(), color=C30, lw=2.2, label=f"media K-Fold = {puntajes.mean():.3f}")
    c.axvline(simples.min(), color=RED, ls=":", lw=1.4)
    c.axvline(simples.max(), color=RED, ls=":", lw=1.4)
    c.set_xlabel("R² de UNA sola partición de validación")
    c.set_ylabel("frecuencia (300 particiones)")
    c.legend(fontsize=8.2, frameon=False, loc="upper left")
    c.set_title(f"Una partición oscila entre {simples.min():.2f} y {simples.max():.2f}", loc="left", fontsize=9.6, color=INK)
    style(c, "y")
    fig.tight_layout(w_pad=2)
    save(fig, "c30_kfold.png")
    print("kfold", perf_counter() - t0)


# ------------------------------------------------------------------ 5. estratificación
def estratificado_fig():
    t0 = perf_counter()
    prop_n, prop_s = [], []
    for s in range(1000):
        prop_n.append(train_test_split(Xf, yf, test_size=0.25, random_state=s)[3].mean())
        prop_s.append(train_test_split(Xf, yf, test_size=0.25, stratify=yf, random_state=s)[3].mean())
    prop_n, prop_s = np.array(prop_n), np.array(prop_s)
    idx = np.random.default_rng(0).choice(2000, 200, replace=False)
    Xs, ys = Xf[idx], yf[idx]
    cn, cs = [], []
    for s in range(1000):
        cn.append(train_test_split(Xs, ys, test_size=0.25, random_state=s)[3].sum())
        cs.append(train_test_split(Xs, ys, test_size=0.25, stratify=ys, random_state=s)[3].sum())
    cn, cs = np.array(cn), np.array(cs)
    NUM["fraude_total"] = [int(yf.sum()), round(float(yf.mean()), 5)]
    NUM["prop_train_test"] = [round(float(yf_tr.mean()), 5), round(float(yf_te.mean()), 5), int(yf_tr.sum()), int(yf_te.sum())]
    NUM["prop_sin_estr"] = [round(float(prop_n.mean()), 4), round(float(prop_n.std()), 4), round(float(prop_n.min()), 4), round(float(prop_n.max()), 4)]
    NUM["prop_estr"] = [round(float(prop_s.mean()), 4), round(float(prop_s.std()), 6)]
    NUM["chico_positivos"] = int(ys.sum())
    NUM["chico_cuenta_sin_estr"] = np.bincount(cn, minlength=9).tolist()
    NUM["chico_cuenta_estr"] = np.bincount(cs, minlength=9).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.3))
    a = axes[0]
    a.hist(prop_n * 100, bins=np.arange(2.4, 8.8, 0.2), color="#D1D5DB", ec="white", label="aleatoria simple")
    a.axvline(yf.mean() * 100, color=C30, lw=2.2, label=f"estratificada: siempre {prop_s.mean() * 100:.1f} %")
    a.set_xlabel("% de fraude en la prueba (500 filas)")
    a.set_ylabel("particiones (de 1.000)")
    a.legend(fontsize=8.4, frameon=False)
    a.set_title(f"n = 2.000: sin estratificar, de {prop_n.min() * 100:.1f} % a {prop_n.max() * 100:.1f} %", loc="left", fontsize=9.6, color=INK)
    style(a, "y")
    b = axes[1]
    ks = np.arange(0, 9)
    b.bar(ks - 0.2, np.bincount(cn, minlength=9) / 10, width=0.4, color="#9CA3AF", label="aleatoria simple")
    b.bar(ks + 0.2, np.bincount(cs, minlength=9) / 10, width=0.4, color=C30, label="estratificada")
    b.set_xlabel(f"casos de fraude en la prueba (50 filas; hay {int(ys.sum())} en las 200)")
    b.set_ylabel("% de las particiones")
    b.set_xticks(ks)
    b.legend(fontsize=8.4, frameon=False)
    b.set_title(f"n = 200: sin estratificar, {np.mean(cn == 0) * 100:.1f} % de las veces cero fraudes", loc="left", fontsize=9.6, color=INK)
    style(b, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "c30_estratificado.png")
    print("estratificado", perf_counter() - t0)


# ------------------------------------------------------------------ 6. métricas
def metricas_fig():
    t0 = perf_counter()
    lr = LogisticRegression(max_iter=1000).fit(Xf_tr, yf_tr)
    pred = lr.predict(Xf_te)
    tn, fp, fn, tp = confusion_matrix(yf_te, pred).ravel()
    NUM["matriz"] = dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    acc = (tp + tn) / len(yf_te)
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    NUM["metricas_0.5"] = dict(acc=round(acc, 3), prec=round(prec, 3), rec=round(rec, 3), f1=round(f1, 3))
    # umbral elegido con validación (sale del entrenamiento), la prueba no se toca
    Xa, Xv, ya, yv = train_test_split(Xf_tr, yf_tr, test_size=0.25, stratify=yf_tr, random_state=RS)
    lr2 = LogisticRegression(max_iter=1000).fit(Xa, ya)
    p = lr2.predict_proba(Xv)[:, 1]
    umbrales = np.linspace(0.02, 0.8, 40)
    P = [precision_score(yv, p >= u, zero_division=0) for u in umbrales]
    R = [recall_score(yv, p >= u) for u in umbrales]
    F = [f1_score(yv, p >= u, zero_division=0) for u in umbrales]
    mejor = int(np.argmax(F))
    NUM["umbral_val"] = dict(umbral=round(float(umbrales[mejor]), 3), f1=round(float(F[mejor]), 3), positivos_val=int(yv.sum()))
    # con ese umbral, en la prueba
    pt = lr2.predict_proba(Xf_te)[:, 1] >= umbrales[mejor]
    NUM["prueba_umbral"] = dict(prec=round(float(precision_score(yf_te, pt)), 3), rec=round(float(recall_score(yf_te, pt)), 3),
                                f1=round(float(f1_score(yf_te, pt)), 3), matriz=confusion_matrix(yf_te, pt).ravel().tolist())

    fig = plt.figure(figsize=(12.5, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.25, 0.85])
    a = fig.add_subplot(gs[0])
    M = np.array([[tn, fp], [fn, tp]])
    colores = np.array([[GREEN, AMBER], [RED, GREEN]])
    for i in range(2):
        for j in range(2):
            a.add_patch(Rectangle((j, 1 - i), 0.96, 0.96, fc=colores[i, j], alpha=0.85, ec="white"))
            a.text(j + 0.48, 1 - i + 0.55, f"{M[i, j]}", ha="center", va="center", fontsize=17, color="white", fontweight="bold")
            a.text(j + 0.48, 1 - i + 0.2, [["TN", "FP"], ["FN", "TP"]][i][j], ha="center", va="center", fontsize=9, color="white")
    a.set_xlim(-0.35, 2)
    a.set_ylim(-0.1, 2.2)
    a.axis("off")
    a.text(0.48, 2.08, "predice\nno fraude", ha="center", fontsize=8.4, color=INK)
    a.text(1.48, 2.08, "predice\nfraude", ha="center", fontsize=8.4, color=INK)
    a.text(-0.05, 1.45, "real\nno", ha="right", va="center", fontsize=8.4, color=INK)
    a.text(-0.05, 0.45, "real\nfraude", ha="right", va="center", fontsize=8.4, color=INK)
    a.set_title(f"Prueba (500): precision {prec:.2f}, recall {rec:.2f}", loc="left", fontsize=9.4, color=INK, y=1.06)

    b = fig.add_subplot(gs[1])
    b.plot(umbrales, P, color=BLUE, lw=2, label="precision")
    b.plot(umbrales, R, color=RED, lw=2, label="recall")
    b.plot(umbrales, F, color=C30, lw=2.4, label="F1")
    b.axvline(0.5, color=GRAY, ls=":", lw=1.3)
    b.text(0.51, 0.04, "umbral por\ndefecto 0.5", fontsize=8, color=GRAY)
    b.axvline(umbrales[mejor], color=GREEN, ls="--", lw=1.4)
    b.text(umbrales[mejor] + 0.01, 0.9, f"mejor F1 en\nvalidación: {umbrales[mejor]:.2f}", fontsize=8, color=GREEN)
    b.set_xlabel("umbral de decisión (validación, 375 filas)")
    b.set_ylim(0, 1.03)
    b.legend(fontsize=8.4, frameon=False, loc="lower right")
    b.set_title("Mover el umbral cambia precision y recall", loc="left", fontsize=9.4, color=INK)
    style(b)

    c = fig.add_subplot(gs[2])
    barras = {"accuracy": (0.946, acc), "recall": (0.0, rec)}
    xs = np.arange(2)
    c.bar(xs - 0.18, [barras[k][0] for k in barras], width=0.34, color="#9CA3AF", label="«siempre no fraude»")
    c.bar(xs + 0.18, [barras[k][1] for k in barras], width=0.34, color=C30, label="regresión logística")
    for i, k in enumerate(barras):
        c.text(i - 0.18, barras[k][0] + 0.02, f"{barras[k][0]:.3f}", ha="center", fontsize=8.5)
        c.text(i + 0.18, barras[k][1] + 0.02, f"{barras[k][1]:.3f}", ha="center", fontsize=8.5)
    c.set_xticks(xs)
    c.set_xticklabels(list(barras))
    c.set_ylim(0, 1.15)
    c.legend(fontsize=8, frameon=False, loc="upper right")
    c.set_title("El accuracy casi no distingue", loc="left", fontsize=9.4, color=INK)
    style(c, "y")
    fig.tight_layout(w_pad=2)
    save(fig, "c30_metricas.png")
    print("metricas", perf_counter() - t0)


# ------------------------------------------------------------------ 7. fuga de datos
def leakage_fig():
    t0 = perf_counter()
    fuga, honesto = [], []
    for s in range(40):
        rng = np.random.default_rng(s)
        Xn = rng.normal(size=(100, 1000))          # 1.000 columnas de ruido puro
        yn = rng.integers(0, 2, 100)               # etiquetas al azar: el modelo NO puede acertar más de 50 %
        Xsel = SelectKBest(f_classif, k=10).fit_transform(Xn, yn)          # selecciona con TODAS las filas (fuga)
        fuga.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=5).mean())
        pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("m", LogisticRegression(max_iter=1000))])
        honesto.append(cross_val_score(pipe, Xn, yn, cv=5).mean())     # la selección se re-aprende en cada pliegue
    fuga, honesto = np.array(fuga), np.array(honesto)
    NUM["fuga"] = dict(fuga_media=round(float(fuga.mean()), 3), fuga_min=round(float(fuga.min()), 3), fuga_max=round(float(fuga.max()), 3),
                       pipe_media=round(float(honesto.mean()), 3), pipe_min=round(float(honesto.min()), 3), pipe_max=round(float(honesto.max()), 3))

    fig = plt.figure(figsize=(12.5, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15])
    a = fig.add_subplot(gs[0])
    bp = a.boxplot([fuga, honesto], widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.8),
                   tick_labels=["selecciona con TODO\ny luego valida (fuga)", "dentro del Pipeline\n(sin fuga)"])
    for patch, c in zip(bp["boxes"], [RED, C30]):
        patch.set(facecolor=c, alpha=0.55)
    a.axhline(0.5, color=INK, ls="--", lw=1.2)
    a.text(2.42, 0.505, "azar", fontsize=8.4, color=INK, ha="right")
    a.set_ylabel("exactitud en validación cruzada")
    a.set_title("Datos de puro ruido (100 filas, 1.000 columnas)", loc="left", fontsize=9.4, color=INK)
    style(a, "y")

    b = fig.add_subplot(gs[1])
    b.set_xlim(0, 10)
    b.set_ylim(0, 4.4)
    b.axis("off")
    filas = [("MAL", RED, "escalar / imputar / seleccionar con todas las filas", "→ luego dividir", 3.1),
             ("BIEN", C30, "dividir primero", "→ fit solo con train → transform en val/prueba", 1.6)]
    for etiqueta, c, t1, t2, yy in filas:
        b.add_patch(FancyBboxPatch((0.05, yy - 0.5), 9.85, 1.15, boxstyle="round,pad=0.04", fc="white", ec=c, lw=1.8))
        b.text(0.35, yy + 0.05, etiqueta, fontsize=12, color=c, fontweight="bold", va="center")
        b.text(1.65, yy + 0.3, t1, fontsize=9.4, color=INK, va="center")
        b.text(1.65, yy - 0.15, t2, fontsize=9.4, color=c, va="center", fontweight="bold")
    b.text(0.05, 0.55, "Pipeline = el «dividir primero» automático dentro de cada pliegue.", fontsize=9, color=GRAY)
    fig.tight_layout(w_pad=2)
    save(fig, "c30_leakage.png")
    print("leakage", perf_counter() - t0)


if __name__ == "__main__":
    import sys
    pasos = dict(esquema=esquema_fig, calidad=calidad_fig, division=division_fig, regularizacion=regularizacion_fig,
                 kfold=kfold_fig, estratificado=estratificado_fig, metricas=metricas_fig, leakage=leakage_fig)
    elegidos = sys.argv[1:] or list(pasos)
    for k in elegidos:
        pasos[k]()
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
