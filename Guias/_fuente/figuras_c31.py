"""Figuras de la Clase 31 (Introducción al Machine Learning).
Datos: (1) el caso «Club Activo» SINTÉTICO de la profesora Laura Sánchez (800 clientes, np.random.seed(42)), que se genera en su
propio notebook y aquí se reproduce idéntico (incluido el orden de los sorteos aleatorios); (2) la base real «Base de datos
Churn-Actividad repaso.xlsx» (3.150 clientes) de la carpeta de la Clase 32, que la actividad de repaso usa para la parte de la Sesión 31
(¿supervisado o no supervisado? ¿clasificación o regresión?). Nada se inventa: todo sale de código.
Ejecutar con el venv que tiene scikit-learn:  python figuras_c31.py   (o con un nombre: python figuras_c31.py kmeans)."""
import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, lasso_path
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 42
C31 = "#4F46E5"      # color de la clase 31 (índigo)
C31_BG = "#EEF2FF"
BLUE = "#2563EB"
RED = "#DC2626"
GRAY = "#6B7280"
AMBER = "#D97706"
TEAL = "#0D9488"
GREEN = "#059669"
SEG = [C31, AMBER, TEAL, "#DB2777"]
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA32 = next(BASE.glob("Clase 32*"))


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ datos: Club Activo (idéntico al notebook de la profesora)
def club_activo(n=800):
    """Reproduce, en el mismo orden, TODOS los sorteos del notebook de la profesora (datos, ruido de la curva de ajuste y variables ruido)."""
    np.random.seed(42)
    edad = np.random.normal(38, 12, n).clip(18, 75)
    antiguedad_meses = np.random.exponential(14, n).clip(1, 60)
    uso_mensual = np.random.poisson(8, n).clip(0, 30)
    quejas_soporte = np.random.poisson(0.6, n).clip(0, 8)
    gasto_mensual = (25 + 0.9 * antiguedad_meses + 3.2 * uso_mensual + np.random.normal(0, 12, n)).clip(15, None)
    logit = (-0.7 + 1.35 * quejas_soporte - 0.14 * uso_mensual - 0.05 * antiguedad_meses + np.random.normal(0, 0.5, n))
    prob_cancela = 1 / (1 + np.exp(-logit))
    cancelo = (np.random.rand(n) < prob_cancela).astype(int)
    df = pd.DataFrame({"edad": edad.round(1), "antiguedad_meses": antiguedad_meses.round(1), "uso_mensual": uso_mensual,
                       "quejas_soporte": quejas_soporte, "gasto_mensual": gasto_mensual.round(0), "cancelo": cancelo})
    for col in ["edad", "antiguedad_meses"]:                       # faltantes (3 %), como en la clase
        idx = df.sample(frac=0.03, random_state=1).index
        df.loc[idx, col] = np.nan
    # sorteos posteriores del notebook (Secciones 7 y 8), en el mismo orden
    x_dato = np.linspace(0, 10, 60)
    y_dato = 3 + 2.5 * x_dato - 0.18 * x_dato ** 2 + np.random.normal(0, 1.6, 60)
    ruido1 = np.random.normal(0, 1, n)
    ruido2 = np.random.normal(0, 1, n)
    return df, x_dato, y_dato, ruido1, ruido2


df, X_DATO, Y_DATO, RUIDO1, RUIDO2 = club_activo()
FEATS_CLF = ["edad", "antiguedad_meses", "uso_mensual", "quejas_soporte"]
FEATS_REG = ["antiguedad_meses", "uso_mensual"]
FEATS_CLU = ["antiguedad_meses", "uso_mensual", "gasto_mensual"]


def prep_clf():
    imp = ColumnTransformer([("edad", SimpleImputer(strategy="mean"), ["edad"]),
                             ("antig", SimpleImputer(strategy="median"), ["antiguedad_meses"]),
                             ("resto", "passthrough", ["uso_mensual", "quejas_soporte"])])
    return imp


def modelo_clf(cw="balanced"):
    return Pipeline([("imp", prep_clf()), ("esc", StandardScaler()), ("lr", LogisticRegression(class_weight=cw))])


Xtr, Xte, ytr, yte = train_test_split(df[FEATS_CLF], df["cancelo"], test_size=0.30, random_state=RS, stratify=df["cancelo"])
clf = modelo_clf().fit(Xtr, ytr)
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]
clf_sin = modelo_clf(None).fit(Xtr, ytr)


# ------------------------------------------------------------------ datos: base real de la actividad de repaso
def cargar_real():
    raw = pd.read_excel(CARPETA32 / "Base de datos Churn-Actividad repaso.xlsx")
    d = raw.copy()
    for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad"]:
        d.loc[d[c] < 0, c] = np.nan                    # negativos imposibles -> faltante (se imputa DENTRO del pipeline)
    d.loc[(d["edad"] < 0) | (d["edad"] > 100), "edad"] = np.nan
    return raw, d


RAW, REAL = cargar_real()
USO = ["segundos", "frec_llamadas", "n_contactos", "frec_sms", "monto", "clv"]
yR = REAL["fuga"]
XR = REAL.drop(columns=["fuga"])
XR_tr, XR_te, yR_tr, yR_te = train_test_split(XR, yR, test_size=0.25, random_state=31, stratify=yR)


# ------------------------------------------------------------------ 0. esquema
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    # IA > ML > DL
    for (x, y, w, h, c, t, ty) in [(0.1, 0.95, 3.5, 3.35, GRAY, "Inteligencia artificial", 4.05),
                                   (0.35, 1.05, 3.0, 2.45, C31, "Machine Learning", 3.22),
                                   (0.7, 1.15, 2.3, 1.15, AMBER, "Deep Learning", 1.95)]:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=c, lw=2))
        ax.text(x + w / 2, ty, t, ha="center", va="center", fontsize=9.5, color=c, fontweight="bold")
    ax.text(1.85, 1.55, "redes neuronales\ncon muchas capas", ha="center", va="center", fontsize=7.8, color=INK)
    ax.text(1.85, 2.72, "aprende de datos\nmejora con la experiencia", ha="center", va="center", fontsize=7.6, color=INK)
    # tres paradigmas
    cols = [("Supervisado", BLUE, "hay etiqueta (y)", [("Clasificación", "¿cancela? sí / no", "regresión logística"),
                                                          ("Regresión", "¿cuánto gastará?", "regresión lineal")]),
            ("No supervisado", TEAL, "no hay etiqueta", [("Clustering", "segmentos de clientes", "K-Means"),
                                                           ("Reducción de dim.", "resumir variables", "PCA")]),
            ("Por refuerzo", AMBER, "entorno + recompensa", [("Agente", "acción → recompensa", "ensayo y error")])]
    x0 = 4.0
    for i, (nombre, c, sub, tareas) in enumerate(cols):
        x = x0 + i * 2.85
        ax.add_patch(FancyBboxPatch((x, 3.45), 2.65, 0.85, boxstyle="round,pad=0.03", fc=c, ec=c, alpha=0.95))
        ax.text(x + 1.325, 4.0, nombre, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        ax.text(x + 1.325, 3.66, sub, ha="center", va="center", fontsize=8, color="white")
        for j, (t, q, algo) in enumerate(tareas):
            y = 2.45 - j * 1.1
            ax.add_patch(FancyBboxPatch((x, y), 2.65, 0.95, boxstyle="round,pad=0.03", fc="white", ec=c, lw=1.5))
            ax.text(x + 1.325, y + 0.72, t, ha="center", va="center", fontsize=9.2, color=c, fontweight="bold")
            ax.text(x + 1.325, y + 0.44, q, ha="center", va="center", fontsize=8, color=INK)
            ax.text(x + 1.325, y + 0.19, algo, ha="center", va="center", fontsize=7.8, color=GRAY, style="italic")
    ax.add_patch(FancyBboxPatch((4.0, 0.12), 8.45, 0.72, boxstyle="round,pad=0.02", fc=C31_BG, ec=C31, lw=1.3))
    ax.text(8.2, 0.48, "Ciclo: datos → modelo → evaluación (error) → mejora · todo se APRENDE con entrenamiento",
            ha="center", va="center", fontsize=7.9, color=C31, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 3.5, 0.72, boxstyle="round,pad=0.02", fc="white", ec=GRAY, lw=1))
    ax.text(1.85, 0.48, "No siempre hace falta IA:\n¿basta una regla o un modelo estadístico?", ha="center", va="center", fontsize=7.8, color=GRAY)
    save(fig, "c31_esquema.png")


# ------------------------------------------------------------------ 1. clasificación: regresión logística
def logistica_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)
    lr = clf.named_steps["lr"]
    Zte = clf[:-1].transform(Xte)
    z = lr.decision_function(Zte)
    NUM["coef_std"] = {k: round(float(v), 3) for k, v in zip(["edad", "antiguedad_meses", "uso_mensual", "quejas_soporte"], lr.coef_[0])}
    NUM["intercepto_std"] = round(float(lr.intercept_[0]), 3)

    a = fig.add_subplot(gs[0])
    g = np.linspace(-5, 5, 300)
    a.plot(g, 1 / (1 + np.exp(-g)), color=C31, lw=2.4)
    for val, col, lab in [(0, "#9CA3AF", "se queda"), (1, AMBER, "canceló")]:
        m = yte.values == val
        a.scatter(z[m], 1 / (1 + np.exp(-z[m])), s=12, alpha=0.6, color=col, label=lab)
    a.axhline(0.5, color=INK, ls="--", lw=1)
    a.axvline(0, color=INK, ls=":", lw=1)
    a.text(-4.9, 0.55, "umbral 0.5 = puntaje 0", fontsize=8, color=INK)
    a.set_xlabel("puntaje de riesgo (logit)")
    a.set_ylabel("probabilidad de cancelar")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("La sigmoide convierte un puntaje en probabilidad", loc="left", fontsize=9.2, color=INK)
    style(a)
    NUM["clientes_p_mayor_05"] = int((proba >= 0.5).sum())

    b = fig.add_subplot(gs[1])
    p_, r_, th = precision_recall_curve(yte, proba)
    b.plot(th, p_[:-1], color=BLUE, lw=2, label="precisión")
    b.plot(th, r_[:-1], color=AMBER, lw=2, label="recall")
    b.axvline(0.5, color=INK, ls="--", lw=1)
    pr05 = (precision_score(yte, pred), recall_score(yte, pred))
    b.scatter([0.5, 0.5], list(pr05), color=[BLUE, AMBER], zorder=5)
    b.text(0.52, pr05[1] + 0.03, f"recall {pr05[1]:.0%}", fontsize=8, color=AMBER)
    b.text(0.52, pr05[0] - 0.09, f"precisión {pr05[0]:.1%}", fontsize=8, color=BLUE)
    b.axhline(float(yte.mean()), color=GRAY, ls=":", lw=1)
    b.text(0.02, float(yte.mean()) + 0.02, f"prevalencia {yte.mean():.1%} (azar)", fontsize=7.6, color=GRAY)
    b.set_xlabel("umbral de decisión")
    b.set_ylim(0, 1.03)
    b.legend(fontsize=8, frameon=False, loc="upper center")
    b.set_title("Subir el umbral: más precisión, menos recall", loc="left", fontsize=9.2, color=INK)
    style(b)
    NUM["clf"] = dict(acc=round(accuracy_score(yte, pred), 3), prec=round(pr05[0], 3), rec=round(pr05[1], 3), f1=round(f1_score(yte, pred), 3),
                      auc=round(roc_auc_score(yte, proba), 3), cm=confusion_matrix(yte, pred).tolist(),
                      prev_test=round(float(yte.mean()), 3), n_test=int(len(yte)), base_acc=round(1 - float(yte.mean()), 3))
    ps = clf_sin.predict(Xte)
    NUM["clf_sin_peso"] = dict(acc=round(accuracy_score(yte, ps), 3), prec=round(precision_score(yte, ps), 3), rec=round(recall_score(yte, ps), 3),
                               f1=round(f1_score(yte, ps), 3), auc=round(roc_auc_score(yte, clf_sin.predict_proba(Xte)[:, 1]), 3))
    NUM["pesos_clase"] = {"0": round(len(ytr) / (2 * (ytr == 0).sum()), 3), "1": round(len(ytr) / (2 * (ytr == 1).sum()), 3)}

    c = fig.add_subplot(gs[2])
    f2 = ["antiguedad_meses", "quejas_soporte"]
    m2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced")).fit(Xtr[f2], ytr)
    gx, gy = np.meshgrid(np.linspace(0, 60, 220), np.linspace(-0.5, 6.5, 220))
    pp = m2.predict_proba(pd.DataFrame({"antiguedad_meses": gx.ravel(), "quejas_soporte": gy.ravel()}))[:, 1].reshape(gx.shape)
    cf = c.contourf(gx, gy, pp, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1)
    c.contour(gx, gy, pp, levels=[0.5], colors=[INK], linewidths=2)
    rng = np.random.default_rng(1)
    for val, col in [(0, "#6B7280"), (1, "#7C2D12")]:
        m = ytr.values == val
        c.scatter(Xtr["antiguedad_meses"][m], Xtr["quejas_soporte"][m] + rng.uniform(-0.18, 0.18, m.sum()), s=7, color=col, alpha=0.55)
    c.set_xlim(0, 60)
    c.set_ylim(-0.5, 6.5)
    c.set_xlabel("antigüedad (meses)")
    c.set_ylabel("quejas a soporte (con jitter)")
    c.set_title("Frontera de decisión (2 variables): una recta", loc="left", fontsize=9.2, color=INK)
    fig.colorbar(cf, ax=c, fraction=0.045, pad=0.02, label="P(cancela)")
    save(fig, "c31_logistica.png")
    print("logistica", perf_counter() - t0)


# ------------------------------------------------------------------ 2. regresión
def regresion_fig():
    t0 = perf_counter()
    Xr, yr = df[FEATS_REG], df["gasto_mensual"]
    Xrt, Xre, yrt, yre = train_test_split(Xr, yr, test_size=0.30, random_state=RS)
    reg = Pipeline([("imp", SimpleImputer(strategy="median")), ("lr", LinearRegression())]).fit(Xrt, yrt)
    pr = reg.predict(Xre)
    lr = reg.named_steps["lr"]
    mse = mean_squared_error(yre, pr)
    NUM["reg"] = dict(coef=dict(zip(FEATS_REG, [round(float(c), 3) for c in lr.coef_])), intercepto=round(float(lr.intercept_), 2), mse=round(mse, 1),
                      rmse=round(float(np.sqrt(mse)), 1), r2_test=round(r2_score(yre, pr), 3), r2_train=round(r2_score(yrt, reg.predict(Xrt)), 3),
                      gasto_media=round(float(yr.mean()), 1), gasto_sd=round(float(yr.std()), 1), n_test=len(yre))
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1], wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.scatter(yre, pr, s=14, alpha=0.6, color=C31)
    lims = [min(yre.min(), pr.min()) - 3, yre.max() + 3]
    a.plot(lims, lims, color=RED, ls="--", lw=1.5, label="predicción perfecta")
    a.set_xlabel("gasto real (miles de CLP)")
    a.set_ylabel("gasto predicho")
    a.text(0.04, 0.9, f"R² = {NUM['reg']['r2_test']:.2f}\nRMSE = {NUM['reg']['rmse']:.1f}", transform=a.transAxes, fontsize=9, color=INK, va="top")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("Predicción vs. realidad (prueba, 240 clientes)", loc="left", fontsize=9.2, color=INK)
    style(a)

    b = fig.add_subplot(gs[1])
    xs = np.arange(2)
    reales = [0.9, 3.2]
    apr = [float(c) for c in lr.coef_]
    b.bar(xs - 0.19, reales, width=0.36, color="#9CA3AF", label="regla oculta")
    b.bar(xs + 0.19, apr, width=0.36, color=C31, label="aprendido")
    for i in range(2):
        b.text(i - 0.19, reales[i] + 0.06, f"{reales[i]}", ha="center", fontsize=8.5)
        b.text(i + 0.19, apr[i] + 0.06, f"{apr[i]:.2f}", ha="center", fontsize=8.5)
    b.set_xticks(xs)
    b.set_xticklabels(["antigüedad\n(por mes)", "uso mensual\n(por uso)"])
    b.set_ylim(0, 3.9)
    b.legend(fontsize=8, frameon=False, loc="upper left")
    b.set_title(f"El modelo «redescubre» la regla (intercepto {lr.intercept_:.1f} vs 25)", loc="left", fontsize=8.6, color=INK)
    style(b, "y")

    c = fig.add_subplot(gs[2])
    res = yre - pr
    c.hist(res, bins=22, color=C31, alpha=0.8)
    c.axvline(0, color=INK, lw=1.2)
    c.set_xlabel("residuo = real − predicho")
    c.set_ylabel("clientes")
    c.set_title(f"Residuos alrededor de 0 (sd ≈ {res.std():.1f})", loc="left", fontsize=9.2, color=INK)
    style(c, "y")
    NUM["reg"]["resid_media"] = round(float(res.mean()), 2)
    NUM["reg"]["resid_sd"] = round(float(res.std()), 1)
    save(fig, "c31_regresion.png")
    print("regresion", perf_counter() - t0)


# ------------------------------------------------------------------ 3. K-Means
def kmeans_fig():
    t0 = perf_counter()
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = pre.fit_transform(df[FEATS_CLU])
    ks = list(range(2, 9))
    inercia, sil = [], []
    for k in ks:
        km = KMeans(k, random_state=RS, n_init=10).fit(Z)
        inercia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_))
    km = KMeans(3, random_state=RS, n_init=10).fit(Z)
    lab = km.labels_
    d = df.assign(seg=lab, antiguedad_meses=df["antiguedad_meses"].fillna(df["antiguedad_meses"].median()))
    centros_std = pd.DataFrame(km.cluster_centers_, columns=FEATS_CLU).round(2)
    NUM["km"] = dict(inercia={k: round(float(v), 1) for k, v in zip(ks, inercia)}, silueta={k: round(float(v), 3) for k, v in zip(ks, sil)},
                     tamanos=[int((lab == i).sum()) for i in range(3)], centros_std=centros_std.to_dict("index"),
                     perfil=d.groupby("seg")[FEATS_CLU + ["cancelo"]].mean().round(2).to_dict("index"))
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 0.9, 1.05], wspace=0.62)
    a = fig.add_subplot(gs[0])
    sc = pre.named_steps["standardscaler"]
    cent = sc.inverse_transform(km.cluster_centers_)
    for s in range(3):
        m = lab == s
        a.scatter(d.loc[m, "antiguedad_meses"], d.loc[m, "uso_mensual"], s=10, alpha=0.5, color=SEG[s], label=f"segmento {s}")
        a.scatter(cent[s, 0], cent[s, 1], marker="*", s=230, color=SEG[s], edgecolor=INK, lw=1.2, zorder=5)
    a.set_xlabel("antigüedad (meses)")
    a.set_ylabel("uso mensual")
    a.legend(fontsize=7.6, frameon=False, loc="upper right")
    a.set_title("K-Means (k = 3) en 2 de las 3 variables", loc="left", fontsize=9, color=INK)
    style(a)

    b = fig.add_subplot(gs[1])
    p2 = PCA(2, random_state=RS).fit(Z)
    E = p2.transform(Z)
    for s in range(3):
        m = lab == s
        b.scatter(E[m, 0], E[m, 1], s=10, alpha=0.5, color=SEG[s])
    NUM["km_pca"] = dict(var=[round(float(v), 3) for v in p2.explained_variance_ratio_], suma=round(float(p2.explained_variance_ratio_.sum()), 3),
                         cargas={f"PC{i + 1}": dict(zip(FEATS_CLU, [round(float(c), 2) for c in p2.components_[i]])) for i in range(2)})
    b.set_xlabel(f"PC1 ({p2.explained_variance_ratio_[0]:.1%})")
    b.set_ylabel(f"PC2 ({p2.explained_variance_ratio_[1]:.1%})")
    b.set_title("Los mismos clientes con PCA (solo para ver)", loc="left", fontsize=9, color=INK)
    style(b)

    c = fig.add_subplot(gs[2])
    c.plot(ks, inercia, marker="o", color=C31, lw=2)
    c.set_xlabel("k (número de grupos)")
    c.set_ylabel("inercia", color=C31)
    c.set_xticks(ks)
    c2 = c.twinx()
    c2.plot(ks, sil, marker="s", color=AMBER, lw=2)
    c2.set_ylabel("silueta media", color=AMBER)
    c2.spines["right"].set_visible(True)
    c.set_title("Elegir k: codo y silueta", loc="left", fontsize=9, color=INK)
    style(c)

    e = fig.add_subplot(gs[3])
    vals = centros_std.values
    im = e.imshow(vals, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    e.set_xticks(range(3))
    e.set_xticklabels(["antigüedad", "uso", "gasto"], fontsize=8)
    e.set_yticks(range(3))
    tasas = [d.loc[d.seg == s, "cancelo"].mean() for s in range(3)]
    e.set_yticklabels([f"seg. {s}\ncancela {tasas[s]:.0%}" for s in range(3)], fontsize=8)
    for i in range(3):
        for j in range(3):
            e.text(j, i, f"{vals[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(vals[i, j]) > 1 else INK)
    e.grid(False)
    e.set_title("Centros (en desv. estándar)", loc="left", fontsize=9, color=INK)
    NUM["km"]["tasas_cancela"] = [round(float(t), 3) for t in tasas]
    # estabilidad frente a la semilla y a no escalar
    from sklearn.metrics import adjusted_rand_score
    ari = [adjusted_rand_score(lab, KMeans(3, random_state=s, n_init=1).fit_predict(Z)) for s in range(10)]
    raw = KMeans(3, random_state=RS, n_init=10).fit_predict(SimpleImputer(strategy="median").fit_transform(df[FEATS_CLU]))
    NUM["km"]["ari_semillas_min"] = round(float(min(ari)), 2)
    NUM["km"]["ari_sin_escalar"] = round(float(adjusted_rand_score(lab, raw)), 2)
    from scipy.stats import chi2_contingency
    NUM["km"]["chi2_p"] = float(chi2_contingency(pd.crosstab(d.seg, d.cancelo))[1])
    save(fig, "c31_kmeans.png")
    print("kmeans", perf_counter() - t0)


# ------------------------------------------------------------------ 4. subajuste, ajuste óptimo, sobreajuste
def ajuste_fig():
    t0 = perf_counter()
    xt, xe, yt, ye = train_test_split(X_DATO.reshape(-1, 1), Y_DATO, test_size=0.3, random_state=7)

    def poly(g, escalar=True):
        pasos = [StandardScaler()] if escalar else []
        return make_pipeline(*pasos, PolynomialFeatures(g, include_bias=False), LinearRegression())

    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.42)
    xl = np.linspace(0, 10, 200).reshape(-1, 1)
    res = {}
    for k, (tit, g) in enumerate([("Subajuste (grado 1)", 1), ("Ajuste adecuado (grado 2)", 2), ("Sobreajuste (grado 15)", 15)]):
        a = fig.add_subplot(gs[k])
        m = poly(g).fit(xt, yt)
        r2tr, r2te = r2_score(yt, m.predict(xt)), r2_score(ye, m.predict(xe))
        res[g] = (round(r2tr, 3), round(r2te, 3))
        a.scatter(xt, yt, s=16, color=C31, label="entrenamiento")
        a.scatter(xe, ye, s=16, color=AMBER, label="prueba")
        a.plot(xl, m.predict(xl), color=INK, lw=1.6)
        a.set_ylim(Y_DATO.min() - 5, Y_DATO.max() + 5)
        a.set_title(f"{tit}\nR² entren. {r2tr:.2f} · R² prueba {r2te:.2f}", loc="left", fontsize=8.8, color=INK)
        if k == 0:
            a.legend(fontsize=7.4, frameon=False, loc="lower center")
        style(a)
    NUM["ajuste"] = {f"grado{g}": v for g, v in res.items()}
    b = fig.add_subplot(gs[3])
    grados = list(range(1, 16))
    tr, te = [], []
    for g in grados:
        m = poly(g).fit(xt, yt)
        tr.append(r2_score(yt, m.predict(xt)))
        te.append(r2_score(ye, m.predict(xe)))
    b.plot(grados, tr, marker="o", ms=4, color=C31, lw=2, label="entrenamiento")
    b.plot(grados, te, marker="o", ms=4, color=AMBER, lw=2, label="prueba")
    sin = poly(15, escalar=False).fit(xt, yt)
    r2_sin = (r2_score(yt, sin.predict(xt)), r2_score(ye, sin.predict(xe)))
    b.scatter([15], [r2_sin[0]], marker="x", s=60, color=RED, zorder=6, label="grado 15 SIN escalar")
    b.set_ylim(-1.2, 1.05)
    b.axhline(0, color=GRAY, lw=1, ls=":")
    b.set_xlabel("grado del polinomio (complejidad)")
    b.set_ylabel("R²")
    b.legend(fontsize=7.4, frameon=False, loc="lower left")
    b.text(8.2, -1.05, "grados 13-15: R² de prueba < −2 (fuera de escala)", fontsize=6.6, color=GRAY)
    b.set_title("La brecha crece con la complejidad", loc="left", fontsize=8.8, color=INK)
    style(b)
    mejor = int(grados[int(np.argmax(te))])
    NUM["ajuste"].update(dict(mejor_grado=mejor, r2_train_curva=[round(float(v), 3) for v in tr], r2_test_curva=[round(float(v), 3) for v in te],
                              grado15_sin_escalar=[round(float(v), 3) for v in r2_sin]))
    # lo que hace el notebook de la profesora (sin escalar)
    sin_g = {}
    for g in (1, 2, 15):
        m = make_pipeline(PolynomialFeatures(g), LinearRegression()).fit(xt, yt)
        sin_g[g] = (round(r2_score(yt, m.predict(xt)), 3), round(r2_score(ye, m.predict(xe)), 3))
    NUM["ajuste"]["profesora_sin_escalar"] = {str(k): v for k, v in sin_g.items()}
    save(fig, "c31_ajuste.png")
    print("ajuste", perf_counter() - t0)


# ------------------------------------------------------------------ 5. Ridge vs Lasso
def regulariza_fig():
    t0 = perf_counter()
    d = df.assign(ruido_1=RUIDO1, ruido_2=RUIDO2)
    cols = ["antiguedad_meses", "uso_mensual", "edad", "quejas_soporte", "ruido_1", "ruido_2"]
    Xa, ya = d[cols], d["gasto_mensual"]
    Xt, Xe, yt, ye = train_test_split(Xa, ya, test_size=0.3, random_state=RS)
    imp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Xt)
    At, Ae = imp.transform(Xt), imp.transform(Xe)
    colores = [C31, C31, GRAY, GRAY, RED, RED]
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    alphas_l = np.logspace(-2, 1.4, 60)
    alphas_r = np.logspace(-1, 4, 60)
    cl = np.array([Lasso(alpha=a, max_iter=20000).fit(At, yt).coef_ for a in alphas_l])
    cr = np.array([Ridge(alpha=a).fit(At, yt).coef_ for a in alphas_r])
    for k, (al, co, tit, nombre) in enumerate([(alphas_l, cl, "Lasso (L1): los coeficientes llegan a 0", "alpha"),
                                              (alphas_r, cr, "Ridge (L2): se encogen sin anularse", "alpha")]):
        a = fig.add_subplot(gs[k])
        for j, c in enumerate(cols):
            a.plot(al, co[:, j], color=colores[j], lw=1.9 if colores[j] != GRAY else 1.4, label=c)
        a.set_xscale("log")
        a.axhline(0, color=INK, lw=0.8)
        a.set_xlabel(f"{nombre} (fuerza de la penalización)")
        a.set_ylabel("coeficiente (variables estandarizadas)")
        a.set_title(tit, loc="left", fontsize=9.2, color=INK)
        if k == 0:
            a.legend(fontsize=7, frameon=False, loc="center right", ncol=1)
        style(a)
    c = fig.add_subplot(gs[2])
    r2l = [r2_score(ye, Lasso(alpha=a, max_iter=20000).fit(At, yt).predict(Ae)) for a in alphas_l]
    r2r = [r2_score(ye, Ridge(alpha=a).fit(At, yt).predict(Ae)) for a in alphas_r]
    c.plot(alphas_l, r2l, color=AMBER, lw=2, label="Lasso")
    c.plot(alphas_r, r2r, color=BLUE, lw=2, label="Ridge")
    base = r2_score(ye, LinearRegression().fit(At, yt).predict(Ae))
    c.axhline(base, color=GRAY, ls="--", lw=1, label=f"sin penalizar ({base:.3f})")
    c.set_xscale("log")
    c.set_xlabel("alpha")
    c.set_ylabel("R² en prueba")
    c.legend(fontsize=8, frameon=False, loc="lower left")
    c.set_title("Un alpha moderado no daña; uno enorme sí", loc="left", fontsize=9.2, color=INK)
    style(c)
    # los valores de la profesora (alpha = 5 y 0.5) con su mismo split (scaler antes de dividir, test_size 0.3, random_state 42)
    Xs = StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
    A1, A2, b1, b2 = train_test_split(Xs, ya, test_size=0.3, random_state=42)
    prof = {nm: dict(zip(cols, [round(float(v), 2) for v in m.fit(A1, b1).coef_])) for nm, m in
            [("OLS", LinearRegression()), ("Ridge5", Ridge(alpha=5.0)), ("Lasso0.5", Lasso(alpha=0.5))]}
    NUM["reg_pen"] = dict(base_r2=round(float(base), 3), prof=prof,
                          ceros_lasso_alpha1=[c_ for c_, v in zip(cols, Lasso(alpha=1.0).fit(At, yt).coef_) if abs(v) < 1e-9],
                          r2_lasso_05=round(float(r2_score(ye, Lasso(alpha=0.5).fit(At, yt).predict(Ae))), 3),
                          r2_ridge_5=round(float(r2_score(ye, Ridge(alpha=5.0).fit(At, yt).predict(Ae))), 3),
                          r2_lasso_alto=round(float(r2l[-1]), 3), r2_ridge_alto=round(float(r2r[-1]), 3))
    # el atajo de los ejercicios: con columnas ortonormales Ridge = b/(1+a) y Lasso = umbral suave
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(300, 4)))
    beta = np.array([4.0, -2.0, 0.8, 0.3])
    yy = Q @ beta
    n_ = len(yy)
    lam = 0.5
    rid = Ridge(alpha=lam, fit_intercept=False).fit(Q, yy).coef_
    las = Lasso(alpha=lam / n_, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Q, yy).coef_
    NUM["atajo_ortonormal"] = dict(ridge=[round(float(v), 3) for v in rid], lasso=[round(float(v), 3) for v in las],
                                   ridge_formula=[round(float(v), 3) for v in beta / (1 + lam)],
                                   lasso_formula=[round(float(np.sign(b) * max(abs(b) - lam, 0)), 3) for b in beta])
    save(fig, "c31_regulariza.png")
    print("regulariza", perf_counter() - t0)


# ------------------------------------------------------------------ 6. la trampa del accuracy
def accuracy_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.35, 0.9], wspace=0.32)
    # (a) modelo perezoso, 5 % de riesgo
    a = fig.add_subplot(gs[0])
    y5 = np.zeros(1000, dtype=int)
    y5[:50] = 1
    z5 = np.zeros(1000, dtype=int)
    vals = [accuracy_score(y5, z5), recall_score(y5, z5, zero_division=0), f1_score(y5, z5, zero_division=0)]
    a.bar(["accuracy", "recall", "F1"], vals, color=[GRAY, RED, RED])
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    a.set_ylim(0, 1.12)
    a.set_title("«Siempre sin riesgo» (5 % de casos)", loc="left", fontsize=9.2, color=INK)
    style(a, "y")
    NUM["perezoso"] = [round(float(v), 3) for v in vals]
    # (b) base real de churn
    b = fig.add_subplot(gs[1])
    modelos = {"siempre «no fuga»": DummyClassifier(strategy="most_frequent"),
               "logística": Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]),
               "logística\nclass_weight='balanced'": Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler()),
                                                               ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])}
    met = ["accuracy", "precisión", "recall", "F1"]
    colores = [GRAY, C31, AMBER]
    out = {}
    for k, (nm, m) in enumerate(modelos.items()):
        m.fit(XR_tr, yR_tr)
        p = m.predict(XR_te)
        v = [accuracy_score(yR_te, p), precision_score(yR_te, p, zero_division=0), recall_score(yR_te, p), f1_score(yR_te, p, zero_division=0)]
        out[nm.replace("\n", " ")] = dict(acc=round(v[0], 3), prec=round(v[1], 3), rec=round(v[2], 3), f1=round(v[3], 3), cm=confusion_matrix(yR_te, p).tolist())
        if hasattr(m, "predict_proba") and nm != "siempre «no fuga»":
            out[nm.replace("\n", " ")]["auc"] = round(roc_auc_score(yR_te, m.predict_proba(XR_te)[:, 1]), 3)
        w = 0.26
        xs = np.arange(4) + (k - 1) * w
        b.bar(xs, v, width=w, color=colores[k], label=nm)
        for x_, v_ in zip(xs, v):
            b.text(x_, v_ + 0.015, f"{v_:.2f}".replace("0.", "."), ha="center", fontsize=6.8)
    b.set_xticks(range(4))
    b.set_xticklabels(met)
    b.set_ylim(0, 1.3)
    b.legend(fontsize=7.4, frameon=False, loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.2)
    b.set_title(f"Base real: {yR_te.mean():.1%} de fuga en la prueba (788 clientes)", loc="left", fontsize=9.2, color=INK)
    style(b, "y")
    NUM["real_clf"] = out
    NUM["real_split"] = dict(n_train=len(XR_tr), n_test=len(XR_te), fuga=round(float(yR.mean()), 4), fuga_test=round(float(yR_te.mean()), 4))
    # (c) matriz de confusión de Club Activo
    c = fig.add_subplot(gs[2])
    cm = confusion_matrix(yte, pred)
    c.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
    etiquetas = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            c.text(j, i, f"{etiquetas[i][j]}\n{cm[i, j]}", ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontweight="bold")
    c.set_xticks([0, 1])
    c.set_xticklabels(["pred. se queda", "pred. cancela"], fontsize=8)
    c.set_yticks([0, 1])
    c.set_yticklabels(["real se queda", "real cancela"], fontsize=8)
    c.grid(False)
    c.set_title("Matriz de confusión (Club Activo)", loc="left", fontsize=9.2, color=INK)
    save(fig, "c31_accuracy.png")
    print("accuracy", perf_counter() - t0)


# ------------------------------------------------------------------ 7. base real: segmentos (Marketing)
def real_fig():
    t0 = perf_counter()
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(XR_tr[USO])
    Ztr, Zte = pre.transform(XR_tr[USO]), pre.transform(XR_te[USO])
    ks = list(range(2, 9))
    inercia, sil = [], []
    for k in ks:
        km = KMeans(k, random_state=31, n_init=10).fit(Ztr)
        inercia.append(km.inertia_)
        sil.append(silhouette_score(Ztr, km.labels_))
    km = KMeans(3, random_state=31, n_init=10).fit(Ztr)
    # ordenar segmentos por uso (mediana de CLV) para etiquetas estables
    orden = np.argsort(pd.Series(XR_tr["clv"].values).groupby(km.labels_).median().values)
    remap = {int(o): i for i, o in enumerate(orden)}
    lab_tr = np.array([remap[v] for v in km.labels_])
    lab_te = np.array([remap[v] for v in km.predict(Zte)])
    fuga_te = [float(yR_te.values[lab_te == s].mean()) for s in range(3)]
    fuga_tr = [float(yR_tr.values[lab_tr == s].mean()) for s in range(3)]
    perfil = XR_tr[USO].assign(seg=lab_tr).groupby("seg").median().round(1)
    NUM["real_km"] = dict(inercia={k: round(float(v), 0) for k, v in zip(ks, inercia)}, silueta={k: round(float(v), 3) for k, v in zip(ks, sil)},
                          n_train=[int((lab_tr == s).sum()) for s in range(3)], n_test=[int((lab_te == s).sum()) for s in range(3)],
                          fuga_train=[round(f, 3) for f in fuga_tr], fuga_test=[round(f, 3) for f in fuga_te], perfil=perfil.to_dict("index"))
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1], wspace=0.3)
    a = fig.add_subplot(gs[0])
    p2 = PCA(2, random_state=31).fit(Ztr)
    E = p2.transform(Ztr)
    nombres = ["uso bajo", "llamadas intensivas", "SMS y valor altos"]
    for s in range(3):
        m = lab_tr == s
        a.scatter(E[m, 0], E[m, 1], s=7, alpha=0.5, color=SEG[s], label=f"{nombres[s]} ({m.sum()})")
    a.legend(fontsize=7.6, frameon=False, markerscale=1.8)
    a.set_xlabel(f"PC1 ({p2.explained_variance_ratio_[0]:.1%})")
    a.set_ylabel(f"PC2 ({p2.explained_variance_ratio_[1]:.1%})")
    a.set_title("Segmentos de la base real (6 variables de uso)", loc="left", fontsize=9, color=INK)
    style(a)
    NUM["real_km"]["pca_var"] = [round(float(v), 3) for v in p2.explained_variance_ratio_]
    b = fig.add_subplot(gs[1])
    b.bar(range(3), [f * 100 for f in fuga_te], color=SEG[:3])
    for i, f in enumerate(fuga_te):
        b.text(i, f * 100 + 0.8, f"{f:.1%}", ha="center", fontsize=9)
    b.axhline(float(yR_te.mean()) * 100, color=INK, ls="--", lw=1)
    b.text(2.45, float(yR_te.mean()) * 100 + 0.8, "toda la base", fontsize=7.6, ha="right", color=INK)
    b.set_xticks(range(3))
    b.set_xticklabels([n.replace(" ", chr(10), 1) for n in nombres], fontsize=8)
    b.set_ylabel("% de fuga en la prueba")
    b.set_ylim(0, max(fuga_te) * 100 + 7)
    b.set_title("La fuga cambia mucho entre segmentos", loc="left", fontsize=9, color=INK)
    style(b, "y")
    c = fig.add_subplot(gs[2])
    c.plot(ks, sil, marker="s", color=AMBER, lw=2, label="silueta")
    c.set_xlabel("k")
    c.set_ylabel("silueta media", color=AMBER)
    c.set_xticks(ks)
    c2 = c.twinx()
    c2.plot(ks, inercia, marker="o", color=C31, lw=2)
    c2.set_ylabel("inercia", color=C31)
    c2.spines["right"].set_visible(True)
    c.set_title("k = 3 tiene la mejor silueta después de k = 2", loc="left", fontsize=9, color=INK)
    style(c)
    save(fig, "c31_real.png")
    print("real", perf_counter() - t0)


if __name__ == "__main__":
    import sys
    pasos = dict(esquema=esquema_fig, logistica=logistica_fig, regresion=regresion_fig, kmeans=kmeans_fig, ajuste=ajuste_fig,
                 regulariza=regulariza_fig, accuracy=accuracy_fig, real=real_fig)
    elegidos = sys.argv[1:] or list(pasos)
    for k in elegidos:
        pasos[k]()
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
