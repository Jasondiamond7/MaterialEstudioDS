"""Figuras de la Clase 37 (SVM y clasificadores clásicos).
Datos: «wisconsin_clase.csv» (569 muestras del Breast Cancer Wisconsin Diagnostic, 4 predictores: radio, textura, suavidad y concavidad
medias; la columna «conjunto» ya trae la partición del profesor Pedro Luiz Ramos: 398 de entrenamiento y 171 de prueba). Todo sale de código:
nada se inventa. En los gráficos de 2 variables (radio y concavidad) los datos se estandarizan con la media y la desviación del ENTRENAMIENTO.
Ejecutar con el venv que tiene scikit-learn:  python figuras_c37.py   (o con nombres: python figuras_c37.py margen gamma)."""
import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
RS = 42
C37 = "#0284C7"      # color de la clase 37 (azul cielo)
C37_BG = "#F0F9FF"
BEN = "#2563EB"      # benigno
MAL = "#DC2626"      # maligno
GRAY = "#6B7280"
AMBER = "#D97706"
TEAL = "#0D9488"
GREEN = "#059669"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CARPETA = next(BASE.glob("Clase 37*"))
V4 = ["radio", "textura", "suavidad", "concavidad"]
V2 = ["radio", "concavidad"]

datos = pd.read_csv(CARPETA / "wisconsin_clase.csv")
tr = datos[datos["conjunto"] == "entrenamiento"]
te = datos[datos["conjunto"] == "prueba"]
ytr, yte = tr["maligno"].to_numpy(), te["maligno"].to_numpy()

# --- los tres modelos de la clase (4 variables, estandarizadas con el entrenamiento) tal como en clasificadores.py
esc4 = StandardScaler().fit(tr[V4])
Xtr, Xte = esc4.transform(tr[V4]), esc4.transform(te[V4])
logistica = LogisticRegression(penalty=None, max_iter=2000).fit(Xtr, ytr)
svm_lineal = SVC(kernel="linear", C=0.1).fit(Xtr, ytr)
svm_radial = SVC(kernel="rbf", C=10, gamma=0.1).fit(Xtr, ytr)
p_log = logistica.predict_proba(Xte)[:, 1]
f_lin = svm_lineal.decision_function(Xte)
f_rad = svm_radial.decision_function(Xte)

# --- 2 variables (radio, concavidad) para dibujar fronteras
esc2 = StandardScaler().fit(tr[V2])
Z2tr, Z2te = esc2.transform(tr[V2]), esc2.transform(te[V2])
LIM = (-2.1, 4.6, -1.5, 4.4)           # xmin, xmax, ymin, ymax en unidades estandarizadas


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def malla(n=260):
    gx, gy = np.meshgrid(np.linspace(LIM[0], LIM[1], n), np.linspace(LIM[2], LIM[3], n))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def puntos(ax, Z, y, s=14, alpha=0.7):
    for val, col in [(0, BEN), (1, MAL)]:
        m = y == val
        ax.scatter(Z[m, 0], Z[m, 1], s=s, color=col, alpha=alpha, lw=0, zorder=3)


def ejes2(ax, titulo, fs=9.2):
    ax.set_xlim(LIM[0], LIM[1])
    ax.set_ylim(LIM[2], LIM[3])
    ax.set_xlabel("radio (estandarizado)")
    ax.set_ylabel("concavidad (estandarizada)")
    ax.set_title(titulo, loc="left", fontsize=fs, color=INK)
    style(ax)


def frontera(ax, modelo, niveles=(0,), margen=True, relleno=True):
    gx, gy, G = malla()
    f = modelo.decision_function(G).reshape(gx.shape)
    if relleno:
        ax.contourf(gx, gy, f, levels=[-1e9, 0, 1e9], colors=[BEN, MAL], alpha=0.10, zorder=1)
    ax.contour(gx, gy, f, levels=[0], colors=[INK], linewidths=2, zorder=4)
    if margen:
        ax.contour(gx, gy, f, levels=[-1, 1], colors=[INK], linewidths=1, linestyles="--", zorder=4)
    return f


def anillos(ax, modelo, Z=None):
    sv = modelo.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=70, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=5)


# ------------------------------------------------------------------ 0. esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def caja(x, y, w, h, color, titulo, lineas, relleno="white", fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=relleno, ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.2, titulo, ha="center", va="center", fontsize=9.6, color=color, fontweight="bold")
        for i, t in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.52 - i * 0.28, t, ha="center", va="center", fontsize=fs, color=INK)

    caja(0.1, 2.75, 2.7, 1.65, GRAY, "Datos Wisconsin", ["569 muestras · 4 predictores", "benigno 0 / maligno 1", "entrenamiento 398", "prueba 171 (no se toca)"])
    caja(3.15, 2.75, 2.55, 1.65, GREEN, "Pipeline", ["StandardScaler: media y sd", "del ENTRENAMIENTO", "luego el clasificador", "(nada se aprende de prueba)"])
    # modelos
    caja(6.05, 3.2, 3.15, 1.2, C37, "Regresión logística", ["probabilidad p(x) = 1/(1+e^−η)", "clase 1 si p ≥ umbral t"])
    caja(6.05, 1.75, 3.15, 1.2, MAL, "SVM (kernel lineal o RBF)", ["puntaje f(x) = w·x + b", "clase 1 si f(x) ≥ 0"])
    caja(9.55, 1.75, 2.85, 2.65, AMBER, "Hiperparámetros", ["C (cost): peso de las violaciones", "kernel: lineal / poly / RBF", "γ (gamma): alcance del RBF", "se eligen con validación", "cruzada SOLO en entrenamiento"], fs=7.8)
    ax.annotate("", xy=(3.1, 3.55), xytext=(2.85, 3.55), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.0, 3.8), xytext=(5.75, 3.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.0, 2.35), xytext=(5.75, 3.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(9.5, 2.75), xytext=(9.25, 2.45), arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 12.3, 1.35, boxstyle="round,pad=0.03", fc=C37_BG, ec=C37, lw=1.4))
    ax.text(6.25, 1.28, "Se evalúa UNA vez en prueba: matriz de confusión → exactitud · sensibilidad · especificidad · AUC", ha="center", va="center",
            fontsize=9.2, color=C37, fontweight="bold")
    ax.text(6.25, 0.88, "VN  FP  /  FN  VP   (filas = real, columnas = predicho; clase positiva = maligno)", ha="center", va="center", fontsize=8.4, color=INK)
    ax.text(6.25, 0.5, "R ↔ Python:  glm(family = binomial) ↔ LogisticRegression(penalty=None)  ·  svm(cost = C, kernel = «radial») ↔ SVC(C=C, kernel=«rbf»)",
            ha="center", va="center", fontsize=8.0, color=GRAY, style="italic")
    save(fig, "c37_esquema.png")


# ------------------------------------------------------------------ 1. hiperplano y margen
def margen_fig():
    t0 = perf_counter()
    rng = np.random.default_rng(3)
    ben = tr[(tr["maligno"] == 0) & (tr["radio"] < 13) & (tr["concavidad"] < 0.06)]
    mal = tr[(tr["maligno"] == 1) & (tr["radio"] > 15.5) & (tr["concavidad"] > 0.08)]
    sub = pd.concat([ben.sample(18, random_state=3), mal.sample(14, random_state=3)])
    Zs = esc2.transform(sub[V2])
    ys = sub["maligno"].to_numpy()
    duro = SVC(kernel="linear", C=1e6).fit(Zs, ys)
    w, b = duro.coef_[0], float(duro.intercept_[0])
    NUM["margen_duro"] = dict(n=len(sub), acc=round(float(duro.score(Zs, ys)), 3), w=[round(float(v), 3) for v in w], b=round(b, 3),
                              norma_w=round(float(np.linalg.norm(w)), 3), ancho=round(float(2 / np.linalg.norm(w)), 3), n_sv=int(len(duro.support_)))
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.28)

    a = fig.add_subplot(gs[0])
    gx, gy, G = malla()
    f = duro.decision_function(G).reshape(gx.shape)
    a.contourf(gx, gy, f, levels=[-1e9, 0, 1e9], colors=[BEN, MAL], alpha=0.10)
    a.contour(gx, gy, f, levels=[0], colors=[INK], linewidths=2)
    a.contour(gx, gy, f, levels=[-1, 1], colors=[INK], linewidths=1, linestyles="--")
    puntos(a, Zs, ys, s=22, alpha=0.85)
    anillos(a, duro)
    # ancho del margen = 2/||w||, medido perpendicular a la frontera
    sv0 = duro.support_vectors_[0]
    p0 = sv0 - duro.decision_function([sv0])[0] * w / (w @ w)
    u = w / (w @ w)
    a.annotate("", xy=p0 + u, xytext=p0 - u, arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2))
    a.text(p0[0] - 0.05, p0[1] + 0.22, f"2/‖w‖ = {2 / np.linalg.norm(w):.2f}", fontsize=8.6, color=AMBER, ha="right", fontweight="bold")
    a.set_aspect("equal")
    ejes2(a, f"Margen duro: {len(duro.support_)} vectores de soporte")
    a.text(0.03, 0.04, "caso separable (32 casos)\nf = 0 continua · f = ±1 discontinuas", transform=a.transAxes, fontsize=7.6, color=INK)

    soft = SVC(kernel="linear", C=1).fit(Z2tr, ytr)
    ws = soft.coef_[0]
    fy = (2 * ytr - 1) * soft.decision_function(Z2tr)
    NUM["margen_suave"] = dict(C=1, n_sv=int(len(soft.support_)), norma_w=round(float(np.linalg.norm(ws)), 3), ancho=round(float(2 / np.linalg.norm(ws)), 3),
                               acc_train=round(float(soft.score(Z2tr, ytr)), 3), acc_test=round(float(soft.score(Z2te, yte)), 3),
                               dentro=int(((fy > 0) & (fy < 1)).sum()), mal=int((fy < 0).sum()))
    b_ = fig.add_subplot(gs[1])
    frontera(b_, soft)
    puntos(b_, Z2tr, ytr, s=10, alpha=0.55)
    anillos(b_, soft)
    ejes2(b_, f"Margen suave (C = 1): {len(soft.support_)} vectores de soporte")
    b_.text(0.97, 0.97, f"entrenamiento (398): {NUM['margen_suave']['mal']} mal clasificados,\n{NUM['margen_suave']['dentro']} dentro del margen", transform=b_.transAxes, fontsize=7.6, color=INK, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.85))

    # holgura en el modelo del profesor (4 variables, lineal, C = 0.1)
    c = fig.add_subplot(gs[2])
    fy4 = (2 * ytr - 1) * svm_lineal.decision_function(Xtr)
    bins = np.linspace(-3.5, 4.5, 41)
    c.axvspan(-3.5, 0, color=MAL, alpha=0.10)
    c.axvspan(0, 1, color=AMBER, alpha=0.14)
    c.axvspan(1, 4.5, color=GREEN, alpha=0.08)
    c.hist(np.clip(fy4, -3.5, 4.5), bins=bins, color=C37, alpha=0.85, edgecolor="white")
    c.axvline(0, color=INK, lw=1.2)
    c.axvline(1, color=INK, lw=1.2, ls="--")
    n_mal, n_dentro, n_fuera = int((fy4 < 0).sum()), int(((fy4 >= 0) & (fy4 < 1)).sum()), int((fy4 >= 1).sum())
    top = c.get_ylim()[1]
    c.text(-1.7, top * 0.88, f"mal clasificado\n{n_mal} casos", ha="center", fontsize=8, color=MAL)
    c.text(0.5, top * 0.88, f"dentro del\nmargen\n{n_dentro}", ha="center", fontsize=7.6, color=AMBER)
    c.text(2.9, top * 0.88, f"correcto y fuera\ndel margen: {n_fuera}", ha="center", fontsize=8, color=GREEN)
    c.set_xlabel("yᵢ·f(xᵢ)  (y = −1 o +1)")
    c.set_ylabel("casos de entrenamiento")
    c.set_title("Holgura ξ = máx(0, 1 − y·f): modelo de la clase", loc="left", fontsize=9.2, color=INK)
    style(c)
    NUM["holgura_clase"] = dict(mal=n_mal, dentro=n_dentro, fuera=n_fuera, n_sv=int(len(svm_lineal.support_)), sv_contados=int((fy4 <= 1 + 1e-6).sum()),
                                suma_xi=round(float(np.maximum(0, 1 - fy4).sum()), 2), w=[round(float(v), 3) for v in svm_lineal.coef_[0]],
                                b=round(float(svm_lineal.intercept_[0]), 3), norma_w=round(float(np.linalg.norm(svm_lineal.coef_[0])), 3))
    save(fig, "c37_margen.png")
    print("margen", perf_counter() - t0)


# ------------------------------------------------------------------ 2. efecto de C (margen suave)
def c_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.3)
    NUM["C2d"] = {}
    for i, Cv in enumerate([0.01, 1, 100]):
        m = SVC(kernel="linear", C=Cv).fit(Z2tr, ytr)
        a = fig.add_subplot(gs[i])
        frontera(a, m)
        puntos(a, Z2tr, ytr, s=8, alpha=0.5)
        ejes2(a, f"C = {Cv:g}: {len(m.support_)} SV · ancho {2 / np.linalg.norm(m.coef_[0]):.2f}", fs=8.8)
        if i:
            a.set_ylabel("")
        NUM["C2d"][str(Cv)] = dict(n_sv=int(len(m.support_)), ancho=round(float(2 / np.linalg.norm(m.coef_[0])), 3), acc_test=round(float(m.score(Z2te, yte)), 3))
    Cs = np.logspace(-3, 3, 13)
    accs_tr, accs_cv, accs_te, nsv = [], [], [], []
    skf = StratifiedKFold(5)
    for Cv in Cs:
        pipe = make_pipeline(StandardScaler(), SVC(kernel="linear", C=Cv))
        accs_cv.append(cross_val_score(pipe, tr[V4], ytr, cv=skf).mean())
        pipe.fit(tr[V4], ytr)
        accs_tr.append(pipe.score(tr[V4], ytr))
        accs_te.append(pipe.score(te[V4], yte))
        nsv.append(len(pipe[-1].support_))
    d = fig.add_subplot(gs[3])
    d.semilogx(Cs, accs_tr, color=GRAY, lw=2, marker="o", ms=3.5, label="entrenamiento")
    d.semilogx(Cs, accs_cv, color=C37, lw=2.4, marker="o", ms=3.5, label="validación cruzada (5)")
    d.set_xlabel("C (escala logarítmica)")
    d.set_ylabel("exactitud")
    d.set_ylim(0.88, 1.0)
    d.legend(fontsize=7.8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    d2 = d.twinx()
    d2.semilogx(Cs, nsv, color=AMBER, lw=1.6, ls="--")
    d2.set_ylabel("vectores de soporte", color=AMBER)
    d2.spines["right"].set_visible(True)
    d2.tick_params(axis="y", colors=AMBER)
    d.set_title("4 variables: C no siempre «mientras más, mejor»", loc="left", fontsize=8.8, color=INK)
    style(d)
    j = int(np.argmax(accs_cv))
    NUM["C4d"] = dict(C=[round(float(c), 4) for c in Cs], acc_train=[round(float(a), 4) for a in accs_tr], acc_cv=[round(float(a), 4) for a in accs_cv],
                      acc_test=[round(float(a), 4) for a in accs_te], n_sv=[int(n) for n in nsv], mejor_C_cv=round(float(Cs[j]), 3), mejor_acc_cv=round(float(accs_cv[j]), 4))
    save(fig, "c37_C.png")
    print("C", perf_counter() - t0)


# ------------------------------------------------------------------ 3. kernels
def kernels_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.3)
    modelos = [("Lineal", SVC(kernel="linear", C=1)),
               ("Polinomial (grado 3)", SVC(kernel="poly", degree=3, coef0=1, gamma="scale", C=1)),
               ("RBF (γ = 0.5)", SVC(kernel="rbf", gamma=0.5, C=1))]
    NUM["kernels2d"] = {}
    for i, (nombre, m) in enumerate(modelos):
        m.fit(Z2tr, ytr)
        a = fig.add_subplot(gs[i])
        frontera(a, m, margen=False)
        puntos(a, Z2tr, ytr, s=8, alpha=0.5)
        ejes2(a, f"{nombre}: prueba {m.score(Z2te, yte):.1%}", fs=8.8)
        if i:
            a.set_ylabel("")
        NUM["kernels2d"][nombre] = dict(acc_train=round(float(m.score(Z2tr, ytr)), 3), acc_test=round(float(m.score(Z2te, yte)), 3), n_sv=int(len(m.support_)))
    d = fig.add_subplot(gs[3])
    dist = np.linspace(0, 4, 200)
    for gam, col in [(0.1, BEN), (0.5, C37), (1, AMBER), (5, MAL)]:
        d.plot(dist, np.exp(-gam * dist ** 2), color=col, lw=2, label=f"γ = {gam:g}")
    d.axvline(2, color=INK, ls=":", lw=1)
    d.scatter([2, 2], [np.exp(-0.4), np.exp(-4)], color=[BEN, AMBER], zorder=5, s=24)
    d.text(2.07, np.exp(-0.4) + 0.02, f"{np.exp(-0.4):.4f}", fontsize=8, color=BEN)
    d.text(2.07, np.exp(-4) + 0.04, f"{np.exp(-4):.4f}", fontsize=8, color=AMBER)
    d.set_xlabel("distancia ‖x − z‖ (espacio estandarizado)")
    d.set_ylabel("similitud K(x, z) = exp(−γ‖x − z‖²)")
    d.legend(fontsize=8, frameon=False, loc="upper right")
    d.set_title("RBF: γ controla el alcance", loc="left", fontsize=8.8, color=INK)
    style(d)
    NUM["k_dist2"] = dict(g01=round(float(np.exp(-0.4)), 4), g1=round(float(np.exp(-4)), 4))
    save(fig, "c37_kernels.png")
    print("kernels", perf_counter() - t0)


# ------------------------------------------------------------------ 4. efecto de gamma
def gamma_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.3)
    NUM["gamma2d"] = {}
    for i, gam in enumerate([0.01, 1, 50]):
        m = SVC(kernel="rbf", C=1, gamma=gam).fit(Z2tr, ytr)
        a = fig.add_subplot(gs[i])
        frontera(a, m, margen=False)
        puntos(a, Z2tr, ytr, s=8, alpha=0.5)
        ejes2(a, f"γ = {gam:g}: entrenamiento {m.score(Z2tr, ytr):.1%} · prueba {m.score(Z2te, yte):.1%}", fs=7.9)
        if i:
            a.set_ylabel("")
        NUM["gamma2d"][str(gam)] = dict(acc_train=round(float(m.score(Z2tr, ytr)), 3), acc_test=round(float(m.score(Z2te, yte)), 3), n_sv=int(len(m.support_)))
    gs_ = np.logspace(-3, 2, 16)
    skf = StratifiedKFold(5)
    a_tr, a_cv, a_te = [], [], []
    for gam in gs_:
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma=gam))
        a_cv.append(cross_val_score(pipe, tr[V4], ytr, cv=skf).mean())
        pipe.fit(tr[V4], ytr)
        a_tr.append(pipe.score(tr[V4], ytr))
        a_te.append(pipe.score(te[V4], yte))
    d = fig.add_subplot(gs[3])
    d.semilogx(gs_, a_tr, color=GRAY, lw=2, marker="o", ms=3.5, label="entrenamiento")
    d.semilogx(gs_, a_cv, color=C37, lw=2.4, marker="o", ms=3.5, label="validación cruzada (5)")
    j = int(np.argmax(a_cv))
    d.axvline(gs_[j], color=AMBER, ls="--", lw=1.2)
    d.text(gs_[j] * 1.15, 0.78, f"mejor γ ≈ {gs_[j]:.3f}", color=AMBER, fontsize=8)
    d.axvspan(10, 100, color=MAL, alpha=0.08)
    d.text(14, 0.72, "sobreajuste", color=MAL, fontsize=8)
    d.axvspan(0.001, 0.003, color=GRAY, alpha=0.1)
    d.text(0.0011, 0.72, "subajuste", color=GRAY, fontsize=8)
    d.set_xlabel("γ (escala logarítmica), C = 1")
    d.set_ylabel("exactitud")
    d.legend(fontsize=7.8, frameon=False, loc="lower center")
    d.set_title("4 variables: el valor intermedio gana", loc="left", fontsize=8.8, color=INK)
    style(d)
    NUM["gamma4d"] = dict(gamma=[round(float(g), 4) for g in gs_], acc_train=[round(float(a), 4) for a in a_tr], acc_cv=[round(float(a), 4) for a in a_cv],
                          acc_test=[round(float(a), 4) for a in a_te], mejor_gamma_cv=round(float(gs_[j]), 4), mejor_acc_cv=round(float(a_cv[j]), 4))
    save(fig, "c37_gamma.png")
    print("gamma", perf_counter() - t0)


# ------------------------------------------------------------------ 5. búsqueda en rejilla (validación cruzada solo en entrenamiento)
def rejilla_fig():
    t0 = perf_counter()
    modelo = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    prof = GridSearchCV(modelo, {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}, cv=5, scoring="accuracy").fit(tr[V4], ytr)
    Cs, Gs = [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]
    amplia = GridSearchCV(modelo, {"svc__C": Cs, "svc__gamma": Gs}, cv=5, scoring="accuracy").fit(tr[V4], ytr)
    NUM["rejilla_prof"] = dict(mejor=prof.best_params_, acc_cv=round(float(prof.best_score_), 4),
                               tabla=[dict(C=float(r["param_svc__C"]), gamma=float(r["param_svc__gamma"]), media=round(float(r["mean_test_score"]), 4),
                                           sd=round(float(r["std_test_score"]), 4), rank=int(r["rank_test_score"])) for _, r in pd.DataFrame(prof.cv_results_).iterrows()])
    pe = prof.best_estimator_
    NUM["rejilla_prof"]["prueba_mejor"] = dict(acc=round(float(pe.score(te[V4], yte)), 4), cm=confusion_matrix(yte, pe.predict(te[V4])).tolist())
    ma = amplia.cv_results_["mean_test_score"].reshape(len(Cs), len(Gs))
    NUM["rejilla_amplia"] = dict(mejor=amplia.best_params_, acc_cv=round(float(amplia.best_score_), 4),
                                 prueba_mejor=dict(acc=round(float(amplia.best_estimator_.score(te[V4], yte)), 4), cm=confusion_matrix(yte, amplia.predict(te[V4])).tolist()),
                                 maximo_celdas_empate=int((ma >= ma.max() - 1e-9).sum()))
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.2, 1.1], wspace=0.34)
    a = fig.add_subplot(gs[0])
    m3 = np.array(prof.cv_results_["mean_test_score"]).reshape(3, 3)
    im = a.imshow(m3, cmap="Blues", vmin=0.6, vmax=0.95, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{m3[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if m3[i, j] > 0.85 else INK)
    a.set_xticks(range(3))
    a.set_xticklabels(["0.01", "0.1", "1"])
    a.set_yticks(range(3))
    a.set_yticklabels(["0.1", "1", "10"])
    a.set_xlabel("gamma")
    a.set_ylabel("C")
    a.set_title("Rejilla del profesor (3×3, 5 folds)", loc="left", fontsize=9.0, color=INK)
    bi = prof.best_index_
    a.add_patch(plt.Rectangle((bi % 3 - 0.5, bi // 3 - 0.5), 1, 1, fill=False, ec=AMBER, lw=2.6))
    b = fig.add_subplot(gs[1])
    b.imshow(ma, cmap="Blues", vmin=0.6, vmax=0.95, origin="lower", aspect="auto")
    for i in range(len(Cs)):
        for j in range(len(Gs)):
            b.text(j, i, f"{ma[i, j]:.2f}", ha="center", va="center", fontsize=7.8, color="white" if ma[i, j] > 0.85 else INK)
    b.set_xticks(range(len(Gs)))
    b.set_xticklabels([f"{g:g}" for g in Gs])
    b.set_yticks(range(len(Cs)))
    b.set_yticklabels([f"{c:g}" for c in Cs])
    b.set_xlabel("gamma")
    b.set_ylabel("C")
    bj = amplia.best_index_
    b.add_patch(plt.Rectangle((bj % len(Gs) - 0.5, bj // len(Gs) - 0.5), 1, 1, fill=False, ec=AMBER, lw=2.6))
    b.set_title("Rejilla ampliada (6×5): exactitud media", loc="left", fontsize=9.0, color=INK)
    c = fig.add_subplot(gs[2])
    tab = pd.DataFrame(prof.cv_results_).sort_values("rank_test_score")
    etq = [f"C={r.param_svc__C:g}, γ={r.param_svc__gamma:g}" for r in tab.itertuples()]
    y = np.arange(len(tab))[::-1]
    c.errorbar(tab["mean_test_score"], y, xerr=tab["std_test_score"], fmt="o", color=C37, ecolor="#9CA3AF", capsize=3, ms=5)
    c.set_yticks(y)
    c.set_yticklabels(etq, fontsize=7.6)
    c.set_xlabel("exactitud media ± 1 sd entre folds")
    c.axvline(tab["mean_test_score"].iloc[0] - tab["std_test_score"].iloc[0], color=AMBER, ls=":", lw=1.2)
    c.set_title("Varias combinaciones empatan dentro del ruido", loc="left", fontsize=9.0, color=INK)
    style(c, "x")
    save(fig, "c37_rejilla.png")
    print("rejilla", perf_counter() - t0)


# ------------------------------------------------------------------ 6. logística frente a SVM, probabilidad frente a puntaje
def comparacion_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.3)
    lr2 = LogisticRegression(penalty=None, max_iter=2000).fit(Z2tr, ytr)
    lin2 = SVC(kernel="linear", C=1).fit(Z2tr, ytr)
    rbf2 = SVC(kernel="rbf", C=10, gamma=0.1).fit(Z2tr, ytr)
    gx, gy, G = malla()
    a = fig.add_subplot(gs[0])
    p = lr2.predict_proba(G)[:, 1].reshape(gx.shape)
    cf = a.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
    a.contour(gx, gy, p, levels=[0.5], colors=[INK], linewidths=2)
    puntos(a, Z2te, yte, s=11, alpha=0.8)
    ejes2(a, f"Logística: prueba {lr2.score(Z2te, yte):.1%}", fs=8.8)
    b = fig.add_subplot(gs[1])
    frontera(b, lin2)
    puntos(b, Z2te, yte, s=11, alpha=0.8)
    ejes2(b, f"SVM lineal: prueba {lin2.score(Z2te, yte):.1%}", fs=8.8)
    b.set_ylabel("")
    c = fig.add_subplot(gs[2])
    frontera(c, rbf2, margen=False)
    puntos(c, Z2te, yte, s=11, alpha=0.8)
    ejes2(c, f"SVM radial: prueba {rbf2.score(Z2te, yte):.1%}", fs=8.8)
    c.set_ylabel("")
    NUM["comp2d"] = dict(logistica=round(float(lr2.score(Z2te, yte)), 3), lineal=round(float(lin2.score(Z2te, yte)), 3), radial=round(float(rbf2.score(Z2te, yte)), 3),
                         coef_log=[round(float(v), 3) for v in lr2.coef_[0]], w_svm=[round(float(v), 3) for v in lin2.coef_[0]])
    # probabilidad (Platt) frente a puntaje en el modelo radial del profesor
    pl = SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=0).fit(Xtr, ytr)
    f = pl.decision_function(Xte)
    pp = pl.predict_proba(Xte)[:, 1]
    pred = pl.predict(Xte)
    disc = (pp >= 0.5) != (pred == 1)
    d = fig.add_subplot(gs[3])
    for val, col in [(0, BEN), (1, MAL)]:
        m = yte == val
        d.scatter(f[m], pp[m], s=14, color=col, alpha=0.6, lw=0)
    d.scatter(f[disc], pp[disc], s=70, facecolors="none", edgecolors=INK, lw=1.2, label=f"predict ≠ (p ≥ 0.5): {int(disc.sum())} casos")
    d.axvline(0, color=INK, ls="--", lw=1)
    d.axhline(0.5, color=INK, ls=":", lw=1)
    d.set_xlabel("puntaje f(x) = decision_function")
    d.set_ylabel("p con probability=True (Platt)")
    d.legend(fontsize=7.6, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.75))
    d.set_title("SVM radial: puntaje ≠ probabilidad", loc="left", fontsize=8.8, color=INK)
    style(d)
    NUM["platt"] = dict(discrepancias=int(disc.sum()), acc_predict=round(float(accuracy_score(yte, pred)), 4), acc_p05=round(float(accuracy_score(yte, (pp >= 0.5).astype(int))), 4),
                        auc_f=round(float(roc_auc_score(yte, f)), 4), auc_p=round(float(roc_auc_score(yte, pp)), 4),
                        cm_predict=confusion_matrix(yte, pred).tolist(), cm_p05=confusion_matrix(yte, (pp >= 0.5).astype(int)).tolist())
    save(fig, "c37_logistica_svm.png")
    print("comparacion", perf_counter() - t0)


# ------------------------------------------------------------------ 7. umbral, sensibilidad/especificidad y ROC
def umbral_fig():
    t0 = perf_counter()
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    p_cv = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=2000)), tr[V4], ytr, cv=skf, method="predict_proba")[:, 1]
    ts = np.linspace(0.02, 0.98, 97)
    sens = [recall_score(ytr, (p_cv >= t).astype(int)) for t in ts]
    espe = [recall_score(ytr, (p_cv >= t).astype(int), pos_label=0) for t in ts]
    acc = [accuracy_score(ytr, (p_cv >= t).astype(int)) for t in ts]
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.3)
    a = fig.add_subplot(gs[0])
    a.plot(ts, sens, color=MAL, lw=2.2, label="sensibilidad (VP / malignos)")
    a.plot(ts, espe, color=BEN, lw=2.2, label="especificidad (VN / benignos)")
    a.plot(ts, acc, color=GRAY, lw=1.4, ls="--", label="exactitud")
    for t in (0.5, 0.8):
        a.axvline(t, color=INK, ls=":", lw=1)
    j = int(np.argmin(np.abs(np.array(sens) - np.array(espe))))
    a.set_xlabel("umbral t sobre la probabilidad")
    a.set_ylim(0.5, 1.02)
    a.legend(fontsize=7.6, frameon=False, loc="lower center")
    a.set_title("Subir el umbral: más especificidad, menos sensibilidad", loc="left", fontsize=8.6, color=INK)
    style(a)
    NUM["umbral_cv"] = {f"{t}": dict(sens=round(float(sens[int(np.argmin(abs(ts - t)))]), 3), espe=round(float(espe[int(np.argmin(abs(ts - t)))]), 3)) for t in (0.3, 0.5, 0.8)}
    NUM["umbral_cv"]["cruce"] = dict(t=round(float(ts[j]), 2), sens=round(float(sens[j]), 3), espe=round(float(espe[j]), 3))
    b = fig.add_subplot(gs[1])
    curvas = [("Logística", p_log, BEN), ("SVM lineal", f_lin, AMBER), ("SVM radial", f_rad, MAL)]
    NUM["auc"] = {}
    for nombre, s, col in curvas:
        fpr, tpr, _ = roc_curve(yte, s)
        b.plot(fpr, tpr, color=col, lw=2, label=f"{nombre} (AUC {auc(fpr, tpr):.4f})")
        NUM["auc"][nombre] = round(float(auc(fpr, tpr)), 4)
    b.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    for t, m in [(0.5, "o"), (0.8, "s")]:
        cm = confusion_matrix(yte, (p_log >= t).astype(int)).ravel()
        tn, fp, fn, tp = cm
        b.scatter([fp / (fp + tn)], [tp / (tp + fn)], marker=m, s=60, color=BEN, edgecolor=INK, zorder=6)
        b.text(fp / (fp + tn) + 0.03, tp / (tp + fn) - 0.06, f"t = {t}", fontsize=8, color=BEN)
    b.set_xlabel("1 − especificidad (falsos positivos)")
    b.set_ylabel("sensibilidad")
    b.legend(fontsize=7.6, frameon=False, loc="lower right")
    b.set_title("Curva ROC en prueba (puntajes, no clases)", loc="left", fontsize=8.8, color=INK)
    style(b)
    c = fig.add_subplot(gs[2])
    bins = np.linspace(0, 1, 26)
    c.hist(p_log[yte == 0], bins=bins, color=BEN, alpha=0.75, label="benignos")
    c.hist(p_log[yte == 1], bins=bins, color=MAL, alpha=0.75, label="malignos")
    for t in (0.5, 0.8):
        c.axvline(t, color=INK, ls=":", lw=1.2)
        c.text(t + 0.01, c.get_ylim()[1] * 0.9, f"t = {t}", fontsize=8)
    c.set_yscale("symlog", linthresh=5)
    c.set_xlabel("probabilidad estimada de malignidad")
    c.set_ylabel("casos de prueba (escala simétrica)")
    c.legend(fontsize=8, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.62))
    c.set_title("La probabilidad no cambia con el umbral", loc="left", fontsize=8.8, color=INK)
    style(c)
    for t in (0.3, 0.5, 0.8):
        tn, fp, fn, tp = confusion_matrix(yte, (p_log >= t).astype(int)).ravel()
        NUM.setdefault("umbral_prueba", {})[str(t)] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), sens=round(tp / (tp + fn), 4), espe=round(tn / (tn + fp), 4),
                                                           acc=round((tp + tn) / len(yte), 4), errores=int(fp + fn))
    save(fig, "c37_umbral.png")
    print("umbral", perf_counter() - t0)


# ------------------------------------------------------------------ 8. estandarización
def escala_fig():
    t0 = perf_counter()
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.2, 1.15], wspace=0.32)
    a = fig.add_subplot(gs[0])
    sd = tr[V4].std()
    a.barh(range(4), sd.values, color=[C37, C37, AMBER, AMBER])
    a.set_xscale("log")
    a.set_yticks(range(4))
    a.set_yticklabels(V4)
    for i, v in enumerate(sd.values):
        a.text(v * 1.15, i, f"{v:.3f}", va="center", fontsize=8.4, color=INK)
    a.set_xlim(0.005, 40)
    a.invert_yaxis()
    a.set_xlabel("desviación estándar (escala logarítmica)")
    a.set_title(f"Escalas muy distintas (×{sd.max() / sd.min():.0f})", loc="left", fontsize=9, color=INK)
    style(a, "x")
    NUM["sd"] = {k: round(float(v), 4) for k, v in sd.items()}
    # frontera RBF con radio y suavidad: sin y con estandarizar
    par = ["radio", "suavidad"]
    esc = StandardScaler().fit(tr[par])
    gx = np.linspace(tr["radio"].min() - 1, tr["radio"].max() + 1, 240)
    gy = np.linspace(tr["suavidad"].min() - 0.01, tr["suavidad"].max() + 0.01, 240)
    GX, GY = np.meshgrid(gx, gy)
    Gdf = pd.DataFrame({"radio": GX.ravel(), "suavidad": GY.ravel()})
    b = fig.add_subplot(gs[1])
    crudo = SVC(kernel="rbf", C=10, gamma=0.1).fit(tr[par].to_numpy(), ytr)
    escalado = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.1)).fit(tr[par], ytr)
    b.contour(GX, GY, crudo.decision_function(Gdf.to_numpy()).reshape(GX.shape), levels=[0], colors=[AMBER], linewidths=2.4)
    b.contour(GX, GY, escalado.decision_function(Gdf).reshape(GX.shape), levels=[0], colors=[C37], linewidths=2.4)
    for val, col in [(0, BEN), (1, MAL)]:
        m = ytr == val
        b.scatter(tr["radio"][m], tr["suavidad"][m], s=8, color=col, alpha=0.45, lw=0)
    b.plot([], [], color=AMBER, lw=2.4, label=f"sin estandarizar: prueba {crudo.score(te[par].to_numpy(), yte):.1%}")
    b.plot([], [], color=C37, lw=2.4, label=f"estandarizado: prueba {escalado.score(te[par], yte):.1%}")
    b.legend(fontsize=7.8, frameon=False, loc="upper right")
    b.set_xlabel("radio (unidades originales)")
    b.set_ylabel("suavidad (unidades originales)")
    b.set_title("RBF con 2 variables: la frontera ignora la suavidad", loc="left", fontsize=8.8, color=INK)
    style(b)
    NUM["escala2d"] = dict(crudo=round(float(crudo.score(te[par].to_numpy(), yte)), 3), escalado=round(float(escalado.score(te[par], yte)), 3))
    c = fig.add_subplot(gs[2])
    configs = [("Lineal\nC = 0.1", dict(kernel="linear", C=0.1)), ("RBF\nC = 10, γ = 0.1", dict(kernel="rbf", C=10, gamma=0.1)), ("RBF\nC = 10, γ = «scale»", dict(kernel="rbf", C=10, gamma="scale"))]
    ancho = 0.36
    NUM["escala4d"] = {}
    for i, (nombre, kw) in enumerate(configs):
        crudo4 = SVC(**kw).fit(tr[V4].to_numpy(), ytr)
        esc4p = make_pipeline(StandardScaler(), SVC(**kw)).fit(tr[V4], ytr)
        a_c, a_e = crudo4.score(te[V4].to_numpy(), yte), esc4p.score(te[V4], yte)
        c.bar(i - ancho / 2, a_c, ancho, color=AMBER, label="sin estandarizar" if i == 0 else None)
        c.bar(i + ancho / 2, a_e, ancho, color=C37, label="con StandardScaler" if i == 0 else None)
        c.text(i - ancho / 2, a_c + 0.004, f"{a_c:.1%}", ha="center", fontsize=8)
        c.text(i + ancho / 2, a_e + 0.004, f"{a_e:.1%}", ha="center", fontsize=8)
        NUM["escala4d"][nombre.replace("\n", " ")] = dict(sin=round(float(a_c), 4), con=round(float(a_e), 4),
                                                         cm_sin=confusion_matrix(yte, crudo4.predict(te[V4].to_numpy())).tolist(),
                                                         cm_con=confusion_matrix(yte, esc4p.predict(te[V4])).tolist())
    c.set_xticks(range(3))
    c.set_xticklabels([n for n, _ in configs], fontsize=8)
    c.set_ylim(0.8, 1.01)
    c.set_ylabel("exactitud en prueba (4 variables)")
    c.legend(fontsize=8, frameon=False, loc="upper left")
    c.set_title("Estandarizar importa, sobre todo con RBF", loc="left", fontsize=8.8, color=INK)
    style(c, "y")
    save(fig, "c37_escala.png")
    print("escala", perf_counter() - t0)


def numeros_base():
    cm = {}
    for nombre, c in [("Logística", (p_log >= 0.5).astype(int)), ("SVM lineal", svm_lineal.predict(Xte)), ("SVM radial", svm_radial.predict(Xte))]:
        tn, fp, fn, tp = confusion_matrix(yte, c).ravel()
        cm[nombre] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), acc=round((tp + tn) / len(yte), 4), sens=round(tp / (tp + fn), 4), espe=round(tn / (tn + fp), 4),
                          prec=round(tp / (tp + fp), 4), errores=int(fp + fn))
    NUM["modelos_clase"] = cm
    NUM["split"] = dict(train=len(tr), test=len(te), train_ben=int((ytr == 0).sum()), train_mal=int((ytr == 1).sum()), test_ben=int((yte == 0).sum()), test_mal=int((yte == 1).sum()),
                        base_acc=round(float((yte == 0).mean()), 4))
    NUM["sv"] = dict(lineal=[int(v) for v in svm_lineal.n_support_], radial=[int(v) for v in svm_radial.n_support_])
    nuevo = pd.DataFrame({"radio": [14], "textura": [20], "suavidad": [0.10], "concavidad": [0.08]})
    zn = esc4.transform(nuevo)
    NUM["nuevo"] = dict(p_log=round(float(logistica.predict_proba(zn)[0, 1]), 4), radial=int(svm_radial.predict(zn)[0]), lineal=int(svm_lineal.predict(zn)[0]),
                        f_radial=round(float(svm_radial.decision_function(zn)[0]), 4), f_lineal=round(float(svm_lineal.decision_function(zn)[0]), 4))
    glm = LogisticRegression(penalty=None, max_iter=100000, tol=1e-10).fit(tr[V4], ytr)
    NUM["glm_como_R"] = dict(intercepto=round(float(glm.intercept_[0]), 4), coef=dict(zip(V4, [round(float(v), 4) for v in glm.coef_[0]])),
                             p_nuevo=round(float(glm.predict_proba(nuevo)[0, 1]), 4))
    NUM["logistica_std"] = dict(intercepto=round(float(logistica.intercept_[0]), 4), coef=dict(zip(V4, [round(float(v), 4) for v in logistica.coef_[0]])))
    NUM["sklearn_defecto"] = dict(cm=confusion_matrix(yte, LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict(Xte)).tolist())


if __name__ == "__main__":
    import sys
    pasos = dict(esquema=esquema_fig, margen=margen_fig, c=c_fig, kernels=kernels_fig, gamma=gamma_fig, rejilla=rejilla_fig,
                 comparacion=comparacion_fig, umbral=umbral_fig, escala=escala_fig)
    elegidos = sys.argv[1:] or list(pasos)
    numeros_base()
    for k in elegidos:
        pasos[k]()
    print(json.dumps(NUM, indent=1, default=float, ensure_ascii=False))
    print("ok")
