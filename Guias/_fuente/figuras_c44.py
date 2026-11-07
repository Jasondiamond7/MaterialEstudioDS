"""Figuras de la Clase 44 (Aplicaciones de IA y panorama actual del Deep Learning · Prof. Eloy Alvarado).
Fuentes: diapositivas DipDS_2026_S44.pdf, transcripción de la clase y la partición del profesor (columna `conjunto` de digitos.csv, carpeta de la Clase 45:
1 077 entrenamiento, 360 validación y 360 prueba; las filas están en el mismo orden que sklearn.datasets.load_digits()).
Reproducción propia del bloque 1 (dígitos): regresión logística y MLPClassifier con semillas fijas. Los números de las diapositivas (corrida del profesor) se guardan
aparte en NUM["diapos"]; los de esta reproducción, en el resto del JSON. Todo lo aleatorio usa semillas fijas.
Escribe c44_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c44.py"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C44 = "#4F46E5"
C44_BG = "#EEF2FF"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
TEAL = "#0D9488"
GREEN = "#16A34A"
GRAY = "#94A3B8"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CSV = next(BASE.glob("Clase 45*/digitos.csv"))
SEMILLA_RED = 20261020      # semilla del profesor en la Clase 45
SEMILLA_BOOT = 44001
B = 2000
_C = {}


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def perdidas(y, p):
    """-log de la probabilidad asignada a la clase real, imagen por imagen."""
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))


def desplazar(X):
    """Una columna a la derecha: la primera se rellena con 0 y la última original se descarta."""
    im = X.reshape(-1, 8, 8)
    out = np.zeros_like(im)
    out[:, :, 1:] = im[:, :, :-1]
    return out.reshape(-1, 64)


def entrenar_red(h, X, y):
    return MLPClassifier(hidden_layer_sizes=h, activation="relu", solver="adam", learning_rate_init=0.002, batch_size=64, alpha=1e-4,
                         max_iter=120, tol=0, n_iter_no_change=1000, random_state=SEMILLA_RED).fit(X, y)


# ----------------------------------------------------------------------------- cálculo común
def calc():
    if _C:
        return _C
    dig = load_digits()
    d = pd.read_csv(CSV)
    assert np.array_equal(d.filter(regex="^px").to_numpy(float), dig.data) and np.array_equal(d.digito.to_numpy(), dig.target)
    Xi = dig.data.astype(int)                 # intensidades 0-16
    X = dig.data / 16
    y = dig.target
    tr, va, te = (d.conjunto.eq(k).to_numpy() for k in ("entrenamiento", "validacion", "prueba"))
    dev = tr | va
    # regresión logística de referencia (configuración fijada, ajustada con las 1 437 imágenes de desarrollo)
    lg = LogisticRegression(C=1, max_iter=3000, tol=1e-8).fit(X[dev], y[dev])
    # red: dos candidatas se comparan por log loss de validación (ajustadas solo con entrenamiento); la elegida se reajusta con desarrollo
    cand = {}
    for h in ((16,), (32,)):
        m = entrenar_red(h, X[tr], y[tr])
        cand[h] = float(log_loss(y[va], m.predict_proba(X[va]), labels=np.arange(10)))
    h_el = min(cand, key=cand.get)
    red = entrenar_red(h_el, X[dev], y[dev])
    P = {"log": lg.predict_proba(X[te]), "red": red.predict_proba(X[te])}
    Xs = desplazar(X[te])
    Ps = {"log": lg.predict_proba(Xs), "red": red.predict_proba(Xs)}
    yt = y[te]
    L = {k: perdidas(yt, P[k]) for k in P}
    _C.update(dig=dig, d=d, Xi=Xi, X=X, y=y, tr=tr, va=va, te=te, dev=dev, lg=lg, red=red, cand=cand, h_el=h_el, P=P, Ps=Ps, Xs=Xs, yt=yt, L=L)
    return _C


def resumen(k, P, y):
    cl = P.argmax(1)
    return dict(k=int((cl == y).sum()), n=int(len(y)), acc=float((cl == y).mean()), ll=float(perdidas(y, P).mean()), errores=int((cl != y).sum()))


# ----------------------------------------------------------------------------- 1. números del bloque 1
def numeros_dig():
    c = calc()
    yt, X, Xi = c["yt"], c["X"], c["Xi"]
    NUM["datos"] = dict(n=int(len(c["y"])), n_tr=int(c["tr"].sum()), n_va=int(c["va"].sum()), n_te=int(c["te"].sum()), n_dev=int(c["dev"].sum()),
                        n_px=64, vmax=int(Xi.max()), vmin=int(Xi.min()), por_clase_te=np.bincount(yt).tolist())
    NUM["datos"]["cand"] = {str(k): v for k, v in c["cand"].items()}
    NUM["datos"]["elegida"] = int(c["h_el"][0])
    NUM["datos"]["n_params_elegida"] = int(sum(w.size + b.size for w, b in zip(c["red"].coefs_, c["red"].intercepts_)))
    NUM["datos"]["n_params_log"] = int(c["lg"].coef_.size + c["lg"].intercept_.size)
    NUM["log"] = resumen("log", c["P"]["log"], yt)
    NUM["red"] = resumen("red", c["P"]["red"], yt)
    # diapositivas (corrida del profesor): se citan, no se recalculan
    NUM["diapos"] = dict(log=dict(k=353, acc=98.06, ll=0.1353), red=dict(k=354, acc=98.33, ll=0.0657), ic=[0.0478, 0.0901], dif=0.0696,
                         desp=dict(log=dict(acc=39.72, ll=2.3826), red=dict(acc=45.00, ll=3.1314)), errores_p=[0.85, 0.80, 0.91, 0.77, 0.76, 0.94], n_err_red=6)
    # imagen de la diapositiva 11 (índice 8 de load_digits: un 8 de la partición de entrenamiento)
    i8 = 8
    im = Xi[i8].reshape(8, 8)
    NUM["ejemplo"] = dict(idx=i8, etiqueta=int(c["y"][i8]), matriz=im.tolist(), fila0=im[0].tolist(), fila0_esc=[round(v / 16, 4) for v in im[0]],
                          n_ceros=int((im == 0).sum()), n_16=int((im == 16).sum()), suma=int(im.sum()))
    # una imagen de prueba bien clasificada con probabilidades de ambos modelos (para "leer una predicción paso a paso")
    ok = np.where((c["P"]["log"].argmax(1) == yt) & (c["P"]["red"].argmax(1) == yt) & (yt == 3))[0][0]
    NUM["paso"] = dict(i=int(ok), real=3, p_log=float(c["P"]["log"][ok, 3]), p_red=float(c["P"]["red"][ok, 3]),
                       l_log=float(c["L"]["log"][ok]), l_red=float(c["L"]["red"][ok]))
    # errores
    for k in ("log", "red"):
        cl = c["P"][k].argmax(1)
        idx = np.where(cl != yt)[0]
        filas = [dict(i=int(i), real=int(yt[i]), pred=int(cl[i]), p_pred=float(c["P"][k][i, cl[i]]), p_real=float(c["P"][k][i, yt[i]]), perdida=float(c["L"][k][i]))
                 for i in idx]
        filas.sort(key=lambda r: -r["p_pred"])
        NUM[k]["lista_errores"] = filas
        NUM[k]["n_conf90"] = int(sum(r["p_pred"] >= 0.9 for r in filas))
        NUM[k]["perdida_errores"] = float(sum(r["perdida"] for r in filas))
        NUM[k]["perdida_total"] = float(c["L"][k].sum())
        NUM[k]["perdida_aciertos"] = float(c["L"][k].sum() - sum(r["perdida"] for r in filas))
        NUM[k]["pmax_media"] = float(c["P"][k].max(1).mean())
    # estrés: desplazar una columna
    NUM["desp"] = dict(log=resumen("log", c["Ps"]["log"], yt), red=resumen("red", c["Ps"]["red"], yt))
    for k in ("log", "red"):
        NUM["desp"][k]["pmax_media"] = float(c["Ps"][k].max(1).mean())
        NUM["desp"][k]["n_conf90_errores"] = int(((c["Ps"][k].argmax(1) != yt) & (c["Ps"][k].max(1) >= 0.9)).sum())
    NUM["desp"]["chance"] = 0.1
    # otros desplazamientos (ejercicio del notebook): 2 columnas a la derecha y 1 a la izquierda
    def mover(Xv, k):
        im = Xv.reshape(-1, 8, 8)
        out = np.zeros_like(im)
        if k > 0:
            out[:, :, k:] = im[:, :, :-k]
        else:
            out[:, :, :k] = im[:, :, -k:]
        return out.reshape(-1, 64)
    NUM["desp_k"] = {}
    for k in (2, -1):
        Xk = mover(c["X"][c["te"]], k)
        NUM["desp_k"][str(k)] = {m: resumen(m, (c["lg"] if m == "log" else c["red"]).predict_proba(Xk), yt) for m in ("log", "red")}
    # regla de umbral: aceptar sin revisión si la probabilidad máxima >= t
    NUM["umbral"] = {}
    for m in ("log", "red"):
        pm = c["P"][m].max(1)
        mal = c["P"][m].argmax(1) != yt
        NUM["umbral"][m] = {}
        for t in (0.9, 0.99):
            ok = pm >= t
            NUM["umbral"][m][str(t)] = dict(auto=int(ok.sum()), err_auto=int((ok & mal).sum()), revision=int((~ok).sum()), err_rev=int((~ok & mal).sum()))


# ----------------------------------------------------------------------------- 2. bootstrap de la diferencia de pérdidas
def numeros_boot():
    c = calc()
    n = len(c["yt"])
    dl = c["L"]["log"] - c["L"]["red"]                        # positivo = la red pierde menos
    rng = np.random.default_rng(SEMILLA_BOOT)
    bs = np.empty(B)
    for b in range(B):
        j = rng.choice(n, size=n, replace=True)
        bs[b] = dl[j].mean()
    ic = np.quantile(bs, [0.025, 0.975])
    NUM["boot"] = dict(B=B, semilla=SEMILLA_BOOT, dif=float(dl.mean()), ic=[float(ic[0]), float(ic[1])], se=float(bs.std(ddof=1)),
                       sd_casos=float(dl.std(ddof=1)), se_formula=float(dl.std(ddof=1) / np.sqrt(n)), prop_pos=float((bs > 0).mean()),
                       n_red_gana=int((dl > 0).sum()), n_log_gana=int((dl < 0).sum()), n_empate=int((dl == 0).sum()),
                       max_dl=float(dl.max()), min_dl=float(dl.min()))
    c["bs"] = bs
    c["dl"] = dl
    # IC bootstrap de la exactitud de la red (ejercicio del notebook), semilla propia
    ok = (c["P"]["red"].argmax(1) == c["yt"]).astype(float)
    rng2 = np.random.default_rng(44002)
    ba = np.array([ok[rng2.choice(n, size=n, replace=True)].mean() for _ in range(B)])
    NUM["boot_acc_red"] = dict(semilla=44002, acc=float(ok.mean()), ic=[float(x) for x in np.quantile(ba, [0.025, 0.975])], se=float(ba.std(ddof=1)), se_formula=float(np.sqrt(ok.mean() * (1 - ok.mean()) / n)))


# ----------------------------------------------------------------------------- 3. números pequeños: log loss, embeddings, costos, split por envase
def numeros_pequenos():
    ps = [0.90, 0.60, 0.01]
    NUM["logloss"] = dict(ps=ps, vals=[float(-np.log(p)) for p in ps], p_05=float(-np.log(0.5)), p_099=float(-np.log(0.99)), p_0001=float(-np.log(0.001)))
    q, a, b = np.array([0.20, 0.80]), np.array([0.35, 0.70]), np.array([0.80, 0.20])
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    NUM["emb"] = dict(q=q.tolist(), a=a.tolist(), b=b.tolist(), d_qa=float(np.linalg.norm(q - a)), d_qb=float(np.linalg.norm(q - b)),
                      cos_qa=cos(q, a), cos_qb=cos(q, b), n_q=float(np.linalg.norm(q)), n_a=float(np.linalg.norm(a)), n_b=float(np.linalg.norm(b)),
                      dot_qa=float(q @ a), dot_qb=float(q @ b))
    # costos y revisión (diapositiva 41)
    n, d, fn, fp, cfn, cfp, cap = 1000, 20, 4, 30, 50, 1, 20
    det = d - fn
    NUM["costo"] = dict(n=n, defectos=d, fn=fn, fp=fp, c_fn=cfn, c_fp=cfp, costo_fn=fn * cfn, costo_fp=fp * cfp, costo=fn * cfn + fp * cfp, detectados=det,
                        alertas=det + fp, capacidad=cap, faltan=det + fp - cap, sanos=n - d,
                        acc=(n - fn - fp) / n, recall=det / d, precision=det / (det + fp),
                        siempre_sano_acc=(n - d) / n, siempre_sano_costo=d * cfn)
    # separación por envase (caso integrador, diapositiva 46)
    NUM["envases"] = dict(fotos=6000, envases=1200, vistas=5, tr=840, va=180, te=180, fotos_tr=4200, fotos_va=900, fotos_te=900)
    # pronóstico (diapositiva 24)
    NUM["demanda"] = dict(pred=120, real=100, ref=80, err_modelo=20, err_ref=20)
    # contar palabras (diapositiva 9)
    NUM["bolsa"] = dict(texto="crédito gratis crédito", cuentas=[2, 1, 0], palabras=["crédito", "gratis", "urgente"], largo=3)
    # cajas (diapositiva 38)
    NUM["cajas"] = dict(cajas=12, por_caja=24, total=288)


# ----------------------------------------------------------------------------- figuras
def f_esquema():
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    pasos = [("1 · Tarea", "¿clasificar, pronosticar,\nrecuperar o generar?", C44),
             ("2 · Datos", "¿qué entra y cómo se\nrepresenta con números?", BLUE),
             ("3 · Modelo", "referencia simple\n+ candidato", TEAL),
             ("4 · Salida", "clase, probabilidad,\ncantidad o texto", AMBER),
             ("5 · Comprobar", "métrica, costo y\ncarga de revisión", RED),
             ("6 · Uso", "¿otra cámara, otra fecha,\nquién decide?", GREEN)]
    w, g = 14.6, 2.48
    for i, (t, s, col) in enumerate(pasos):
        x = i * (w + g)
        ax.add_patch(FancyBboxPatch((x + 0.3, 6), w - 0.6, 18, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=col, lw=1.8))
        ax.text(x + w / 2, 20, t, ha="center", va="center", fontsize=9.6, fontweight="bold", color=col)
        ax.text(x + w / 2, 12.5, s, ha="center", va="center", fontsize=6.9, color=INK, linespacing=1.3)
        if i < len(pasos) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.2, 15), (x + w + g - 0.2, 15), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.text(50, 1.8, "El objetivo (paso 1) va antes que los datos; el paso 6 vuelve a preguntar por la tarea cuando cambian las condiciones.", ha="center", fontsize=8, color=MUTED)
    save(fig, "c44_esquema.png")


def f_anidado():
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.25, fc="#E0E7FF", ec=C44, lw=1.8))
    ax.add_patch(Circle((0, -0.2), 0.92, fc="#C7D2FE", ec=C44, lw=1.8))
    ax.add_patch(Circle((0, -0.42), 0.46, fc="#A5B4FC", ec=C44, lw=1.8))
    ax.text(0, 1.02, "Inteligencia artificial", ha="center", fontsize=9.2, fontweight="bold", color=C44)
    ax.text(0, 0.84, "(incluye reglas escritas a mano)", ha="center", fontsize=7, color=INK)
    ax.text(0, 0.45, "Aprendizaje automático", ha="center", fontsize=8.8, fontweight="bold", color=C44)
    ax.text(0, 0.27, "logística, Naive Bayes, árboles", ha="center", fontsize=6.8, color=INK)
    ax.text(0, -0.40, "Aprendizaje\nprofundo", ha="center", va="center", fontsize=8.2, fontweight="bold", color=C44, linespacing=1.15)
    save(fig, "c44_anidado.png")


def f_digitos():
    c = calc()
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.3))
    for k, ax in enumerate(axes.flat):
        i = int(np.where(c["y"] == k)[0][0])
        ax.imshow(c["Xi"][i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"dígito {k}", fontsize=8.4, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "c44_digitos.png")


def f_matriz():
    c = calc()
    E = NUM["ejemplo"]
    im = np.array(E["matriz"])
    fig = plt.figure(figsize=(10.4, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.55, 1.25], wspace=0.12)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    a0.imshow(im, cmap="gray_r", vmin=0, vmax=16)
    a0.set_xticks([])
    a0.set_yticks([])
    titulo(a0, f"Lo que vemos: etiqueta {E['etiqueta']}")
    a1.imshow(im, cmap="gray_r", vmin=0, vmax=16, alpha=0.35)
    for r in range(8):
        for k in range(8):
            a1.text(k, r, str(im[r, k]), ha="center", va="center", fontsize=8.6, color=INK if im[r, k] < 12 else "white", fontweight="bold" if im[r, k] >= 12 else "normal")
    a1.add_patch(Rectangle((-0.5, -0.5), 8, 1, fill=False, ec=C44, lw=2.2))
    a1.set_xticks([])
    a1.set_yticks([])
    titulo(a1, "Lo que ve el modelo: 8 × 8 = 64 números (0 a 16)")
    a2.axis("off")
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 10)
    titulo(a2, "Primera fila → vector → escala 0 a 1")
    f0 = E["fila0"]
    a2.text(0.1, 8.7, "fila 1:", fontsize=8.6, color=MUTED)
    a2.text(1.9, 8.7, str(f0), fontsize=8.6, color=INK)
    a2.text(0.1, 7.4, "se lee por filas, siempre en el mismo orden:", fontsize=8, color=MUTED)
    a2.text(0.1, 6.5, "[" + ", ".join(map(str, f0[:6])) + ", …] → 64 números", fontsize=8.2, color=INK)
    for j, (raw, esc) in enumerate(zip(f0[2:5], E["fila0_esc"][2:5])):
        a2.text(0.1, 4.6 - 1.25 * j, f"{raw} ÷ 16 = {esc:g}", fontsize=9.4, color=C44, fontweight="bold")
    a2.text(0.1, 0.6, "Entrada: 64 números entre 0 y 1.\nLa etiqueta se guarda aparte.", fontsize=8.4, color=INK, linespacing=1.3)
    save(fig, "c44_imagen_numeros.png")


def f_probs():
    c = calc()
    yt = c["yt"]
    E = NUM["red"]["lista_errores"][0]
    ok = NUM["paso"]["i"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), sharey=True)
    for ax, i, t in ((axes[0], ok, "Un acierto"), (axes[1], E["i"], "El error más confiado de la red")):
        x = np.arange(10)
        ax.bar(x - 0.2, c["P"]["log"][i], 0.4, color=GRAY, label="logística")
        ax.bar(x + 0.2, c["P"]["red"][i], 0.4, color=C44, label="red neuronal")
        ax.axvspan(yt[i] - 0.5, yt[i] + 0.5, color="#FEF3C7", zorder=0)
        ax.set_xticks(x)
        ax.set_xlabel("dígito")
        titulo(ax, f"{t} (real = {yt[i]}, zona amarilla)")
        style(ax, "y")
    axes[0].set_ylabel("probabilidad")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "c44_probabilidades.png")


def f_logloss():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    p = np.linspace(0.004, 1, 400)
    ax.plot(p, -np.log(p), color=C44, lw=2.2)
    for pp, off in zip(NUM["logloss"]["ps"], ((0.03, 0.55), (0.04, 0.5), (0.06, 0.4))):
        v = -np.log(pp)
        ax.scatter([pp], [v], color=RED, zorder=3)
        ax.annotate(f"p = {pp:g} → {v:.3f}", (pp, v), xytext=(pp + off[0], v + off[1]), fontsize=8.4, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("probabilidad asignada al dígito real")
    ax.set_ylabel("−log(p)")
    ax.set_ylim(0, 5.6)
    titulo(ax, "Costo de cada predicción: −log(p)")
    style(ax)
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9, "Idea clave", fontsize=9.4, fontweight="bold", color=C44)
    ax.text(0, 8.2, "p cerca de 1  →  costo casi 0\np = 0.5  →  0.693\np cerca de 0  →  costo enorme", fontsize=9, color=INK, linespacing=1.6, va="top")
    ax.text(0, 4.2, "Un error muy confiado\n(p real = 0.01) cuesta 4.605:\n44 veces el de un acierto\ncon p = 0.90 (0.105).", fontsize=9, color=RED, linespacing=1.5, va="top")
    fig.tight_layout()
    save(fig, "c44_logloss.png")


def f_resultados():
    L, R = NUM["log"], NUM["red"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for ax, key, tit, fmt in ((axes[0], "errores", "Imágenes mal clasificadas (de 360)", "{:d}"), (axes[1], "ll", "Log loss (menor es mejor)", "{:.4f}")):
        v = [L[key], R[key]]
        ax.bar(["logística", "red"], v, color=[GRAY, C44], width=0.55)
        for x, y in enumerate(v):
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylim(0, max(v) * 1.2)
        titulo(ax, tit)
        style(ax, "y")
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 8.8, "Misma exactitud, distinta calidad", fontsize=9.4, fontweight="bold", color=C44)
    ax.text(0, 6.2, f"exactitud: {L['acc'] * 100:.2f} % vs {R['acc'] * 100:.2f} %\n({L['errores']} errores en cada una)\n\nlog loss: {L['ll']:.4f} vs {R['ll']:.4f}\n(la red pierde menos)", fontsize=8.8, color=INK, linespacing=1.45, va="center")
    fig.tight_layout()
    save(fig, "c44_resultados.png")


def f_errores():
    c = calc()
    errs = NUM["red"]["lista_errores"][:6]
    n = len(errs)
    fig, axes = plt.subplots(1, 6, figsize=(10.4, 2.3))
    for ax in axes:
        ax.axis("off")
    for ax, e in zip(axes, errs):
        ax.imshow(c["X"][c["te"]][e["i"]].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Real {e['real']} · Pred. {e['pred']}\np = {e['p_pred']:.2f}", fontsize=8.6, color=RED if e["p_pred"] >= 0.9 else INK)
    fig.tight_layout()
    save(fig, "c44_errores_red.png")


def f_desplazadas():
    c = calc()
    yt = c["yt"]
    Xt = c["X"][c["te"]]
    idx = [int(np.where(yt == k)[0][0]) for k in (3, 4, 8, 5)]
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 3.4))
    for j, i in enumerate(idx):
        axes[0, j].imshow(Xt[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].imshow(c["Xs"][i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_title(f"Original · real {yt[i]}", fontsize=8.2)
        axes[1, j].set_title(f"Desplazada · real {yt[i]}", fontsize=8.2)
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    fig.tight_layout()
    save(fig, "c44_desplazadas.png")


def f_estres():
    L, R, DL, DR = NUM["log"], NUM["red"], NUM["desp"]["log"], NUM["desp"]["red"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0))
    x = np.arange(2)
    for ax, key, tit, fmt in ((axes[0], "acc", "Exactitud (%)", "{:.1f}"), (axes[1], "ll", "Log loss", "{:.2f}")):
        sc = 100 if key == "acc" else 1
        o = [L[key] * sc, R[key] * sc]
        s = [DL[key] * sc, DR[key] * sc]
        ax.bar(x - 0.2, o, 0.38, color=C44, label="original")
        ax.bar(x + 0.2, s, 0.38, color=AMBER, label="desplazada 1 columna")
        for xx, v in zip(x - 0.2, o):
            ax.text(xx, v, fmt.format(v), ha="center", va="bottom", fontsize=8.4)
        for xx, v in zip(x + 0.2, s):
            ax.text(xx, v, fmt.format(v), ha="center", va="bottom", fontsize=8.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["logística", "red"])
        titulo(ax, tit)
        style(ax, "y")
    axes[0].set_ylim(0, 115)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].set_ylim(0, max(DL["ll"], DR["ll"]) * 1.2)
    fig.tight_layout()
    save(fig, "c44_estres.png")


def f_bootstrap():
    c = calc()
    bs, bo = c["bs"], NUM["boot"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    ax.hist(bs, bins=40, color=C44, alpha=0.85)
    ax.axvline(bo["dif"], color=RED, lw=2, label=f"observada {bo['dif']:.4f}")
    for x in bo["ic"]:
        ax.axvline(x, color="black", ls="--", lw=1.2)
    ax.axvline(0, color=GREEN, lw=2, label="0 = sin diferencia")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("pérdida media logística − pérdida media red (en cada remuestra)")
    titulo(ax, f"{B} remuestras de las 360 pérdidas · IC 95 % [{bo['ic'][0]:.4f}; {bo['ic'][1]:.4f}]")
    style(ax, "y")
    ax = axes[1]
    dl = c["dl"]
    ax.hist(dl, bins=40, color=GRAY)
    ax.set_yscale("log")
    ax.set_xlabel("diferencia de pérdida por imagen")
    titulo(ax, "Las 360 diferencias (escala log en y)")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c44_bootstrap.png")


def f_embedding():
    E = NUM["emb"]
    q, a, b = (np.array(E[k]) for k in ("q", "a", "b"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    for v, nm, col in ((q, "Q: «cobro duplicado»", RED), (a, "A: «reembolso»", GREEN), (b, "B: «envíos»", BLUE)):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.scatter(*v, color=col, s=55, zorder=3)
        ax.text(v[0] + 0.02, v[1] + 0.035, nm, fontsize=8.4, color=col)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("primer número")
    ax.set_ylabel("segundo número")
    titulo(ax, "Cada texto es un punto (un vector) · números inventados")
    style(ax)
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.2, "¿Qué tan cerca está cada uno de Q?", fontsize=9.2, fontweight="bold", color=C44)
    ax.text(0, 6.4, f"distancia  Q–A = {E['d_qa']:.3f}\ndistancia  Q–B = {E['d_qb']:.3f}", fontsize=9.2, color=INK, linespacing=1.7)
    ax.text(0, 3.2, f"similitud coseno  Q–A = {E['cos_qa']:.3f}\nsimilitud coseno  Q–B = {E['cos_qb']:.3f}", fontsize=9.2, color=INK, linespacing=1.7)
    ax.text(0, 0.6, "A está más cerca de Q que B,\npor distancia y por ángulo.", fontsize=8.8, color=GREEN, linespacing=1.4)
    fig.tight_layout()
    save(fig, "c44_embedding.png")


def f_costo():
    K = NUM["costo"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axes[0]
    ax.barh(["costo total"], [K["costo_fn"]], color=RED, label=f"{K['fn']} omitidos × {K['c_fn']} = {K['costo_fn']}")
    ax.barh(["costo total"], [K["costo_fp"]], left=[K["costo_fn"]], color=AMBER, label=f"{K['fp']} falsas alarmas × {K['c_fp']} = {K['costo_fp']}")
    ax.text(K["costo"] + 4, 0, f"= {K['costo']}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(0, K["costo"] * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    titulo(ax, "Costo de los errores (unidades)")
    style(ax, "x")
    ax = axes[1]
    ax.barh(["revisiones\nnecesarias"], [K["detectados"]], color=GREEN, label=f"{K['detectados']} defectos detectados")
    ax.barh(["revisiones\nnecesarias"], [K["fp"]], left=[K["detectados"]], color=AMBER, label=f"{K['fp']} falsas alarmas")
    ax.axvline(K["capacidad"], color=RED, lw=2, ls="--")
    ax.text(K["capacidad"] + 0.5, 0.42, f"capacidad = {K['capacidad']}", color=RED, fontsize=8.6)
    ax.text(K["alertas"] + 0.8, 0, f"= {K['alertas']}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(0, K["alertas"] * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    titulo(ax, "Carga de revisión: no alcanza")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c44_costo.png")


def f_split():
    rng = np.random.default_rng(44002)
    cols = {0: BLUE, 1: AMBER, 2: GREEN}
    nombres = {0: "entrenamiento", 1: "validación", 2: "prueba"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.9))
    nenv, nv = 10, 5
    # por foto: cada foto cae en un conjunto al azar
    ax = axes[0]
    pf = rng.choice(3, size=(nenv, nv), p=[0.6, 0.2, 0.2])
    # garantizar que se vea la mezcla en al menos tres envases
    for e in range(nenv):
        for v in range(nv):
            ax.add_patch(Rectangle((v, nenv - 1 - e), 0.86, 0.8, fc=cols[int(pf[e, v])], ec="white"))
        ax.text(-0.15, nenv - 1 - e + 0.4, f"E{e + 1}", ha="right", va="center", fontsize=7.4, color=MUTED)
    mezclados = int(sum(len(set(pf[e])) > 1 for e in range(nenv)))
    ax.set_xlim(-0.9, nv)
    ax.set_ylim(0, nenv)
    ax.axis("off")
    titulo(ax, f"Mal: separar por foto ({mezclados} de {nenv} envases con fotos en varios conjuntos)")
    # por envase
    ax = axes[1]
    pe = rng.choice(3, size=nenv, p=[0.6, 0.2, 0.2])
    for e in range(nenv):
        for v in range(nv):
            ax.add_patch(Rectangle((v, nenv - 1 - e), 0.86, 0.8, fc=cols[int(pe[e])], ec="white"))
        ax.text(-0.15, nenv - 1 - e + 0.4, f"E{e + 1}", ha="right", va="center", fontsize=7.4, color=MUTED)
    ax.set_xlim(-0.9, nv)
    ax.set_ylim(0, nenv)
    ax.axis("off")
    titulo(ax, "Bien: separar por envase (las 5 vistas juntas)")
    handles = [Rectangle((0, 0), 1, 1, fc=cols[k]) for k in range(3)]
    fig.legend(handles, [nombres[k] for k in range(3)], loc="lower center", ncol=3, frameon=False, fontsize=8.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "c44_split_envases.png")
    NUM["envases"]["demo_mezclados_por_foto"] = mezclados
    NUM["envases"]["demo_n"] = nenv


def f_texto():
    """Mapa de aplicaciones con texto, imagen y otras tareas (diapositivas 26, 29-35)."""
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")
    filas = [("Clasificar", "consulta del cliente", "categoría", "¿es la correcta?", C44),
             ("Recuperar", "consulta + archivo", "fuentes ordenadas", "¿relevantes y vigentes?", BLUE),
             ("Resumir", "documento extenso", "texto breve", "¿conserva lo importante?", TEAL),
             ("Redactar", "instrucción + antecedentes", "borrador", "¿cada afirmación tiene respaldo?", AMBER)]
    hdr = ["Tarea", "Entra", "Sale", "Se comprueba"]
    xs = [2, 22, 50, 74]
    for x, h in zip(xs, hdr):
        ax.text(x, 29.5, h, fontsize=9, fontweight="bold", color=INK)
    for r, (t, e, s, cmp_, col) in enumerate(filas):
        y = 23.2 - r * 7
        ax.add_patch(FancyBboxPatch((0.5, y - 2.8), 99, 5.6, boxstyle="round,pad=0.1,rounding_size=1", fc="#F8FAFC", ec=GRID))
        ax.text(xs[0], y, t, fontsize=9.2, fontweight="bold", color=col, va="center")
        for x, s_ in zip(xs[1:], (e, s, cmp_)):
            ax.text(x, y, s_, fontsize=8.6, color=INK, va="center")
    save(fig, "c44_tareas_texto.png")


def f_adaptar():
    fig, ax = plt.subplots(figsize=(10.4, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 27)
    ax.axis("off")
    cols = [("Instrucciones y contexto", "cambia la información de entrada;\nel modelo no se vuelve a entrenar", C44),
            ("Reutilizar representaciones", "se conservan los vectores y se\najusta un clasificador encima", TEAL),
            ("Ajuste fino", "se modifica el modelo con datos\netiquetados de la tarea local", AMBER)]
    for i, (t, s, col) in enumerate(cols):
        x = 1 + i * 33.3
        ax.add_patch(FancyBboxPatch((x, 5), 31, 19, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=col, lw=1.8))
        ax.text(x + 15.5, 20, t, ha="center", fontsize=9.4, fontweight="bold", color=col)
        ax.text(x + 15.5, 12, s, ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.4)
    ax.text(50, 1.2, "Menos cambio en el modelo  →  más cambio en el modelo", ha="center", fontsize=8.2, color=MUTED)
    ax.add_patch(FancyArrowPatch((8, 3.4), (92, 3.4), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
    save(fig, "c44_adaptar.png")


def f_diffusion():
    rng = np.random.default_rng(44003)
    base = np.zeros((16, 16))
    base[5:11, 3:8] = 1.0          # "caja azul"
    base[11:13, 1:15] = 0.5        # "mesa"
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.0))
    for ax, (t, a) in zip(axes, (("Ruido", 1.0), ("Formas iniciales", 0.65), ("Más definición", 0.3), ("Imagen final", 0.0))):
        ax.imshow(np.clip((1 - a) * base + a * rng.random((16, 16)), 0, 1), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(t, fontsize=8.2)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "c44_difusion.png")


def f_mapa_resumen():
    """Gráfico de dispersión de lo que aparece en la clase 'dónde está cada idea' (resumen visual de preguntas)."""
    fig, ax = plt.subplots(figsize=(10.4, 2.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.axis("off")
    qs = [("Tarea", "¿Qué hay que producir?", C44), ("Datos", "¿Qué recibe el sistema?", BLUE), ("Resultado", "¿Cómo reconozco un acierto?", TEAL), ("Uso", "¿Qué cambia al llegar datos nuevos?", AMBER)]
    for i, (t, s, col) in enumerate(qs):
        x = 1 + i * 25
        ax.add_patch(FancyBboxPatch((x, 6), 23, 16, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=col, lw=1.8))
        ax.text(x + 11.5, 18, t, ha="center", fontsize=10, fontweight="bold", color=col)
        ax.text(x + 11.5, 11, s, ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.3)
    save(fig, "c44_cuatro_preguntas.png")


PASOS = dict(esquema=f_esquema, anidado=f_anidado, digitos=f_digitos, matriz=f_matriz, probs=f_probs, logloss=f_logloss, resultados=f_resultados, errores=f_errores,
             desplazadas=f_desplazadas, estres=f_estres, bootstrap=f_bootstrap, embedding=f_embedding, costo=f_costo, split=f_split, texto=f_texto, adaptar=f_adaptar,
             difusion=f_diffusion, preguntas=f_mapa_resumen)

if __name__ == "__main__":
    numeros_dig()
    numeros_boot()
    numeros_pequenos()
    for k, fn in PASOS.items():
        fn()
        print("hecho:", k, flush=True)
    NUM["datos"]["B"] = B
    out = Path(__file__).resolve().parent / "c44_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
