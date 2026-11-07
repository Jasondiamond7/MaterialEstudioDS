"""Figuras de la Clase 46 (Deep Learning, autoencoders y representaciones · Prof. Eloy Alvarado).
Fuentes: diapositivas autoencoders.pdf (53), transcripción de la clase (cubre las diapositivas 1 a 14; el video termina en el recreo) y el programa
del profesor autoencoders.py (carpeta de la Clase 46).
Reproducción: se ejecuta la MISMA receta del profesor (64-8-64 y 64-32-8-32-64, tanh oculta + salida lineal, Adam tasa 0.002, lote 64, alpha 0.0001,
150 épocas, semilla 20261022, elección por MSE de validación) con scikit-learn 1.9.1; las cifras salen idénticas a las de las diapositivas.
Los números de las diapositivas se guardan en NUM["comparacion"] junto a los reproducidos; lo demás (ejemplos de juguete, códigos redondeados a
signos, barrido de k en PCA, AE 64-8-64 sobre prueba) es cálculo propio y se rotula «nuestra reproducción» o «ejemplo propio».
Escribe c46_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c46.py"""
import json
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C46 = "#6D28D9"
C46_BG = "#F5F3FF"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
GREEN = "#16A34A"
GRAY = "#94A3B8"
TEAL = "#0F766E"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CSV = next(BASE.glob("Clase 46*/digitos.csv"))
SEMILLA = 20261022
EPOCAS = 150
_C = {}
W_FIG = 7.4


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.6):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ------------------------------------------------------------------ entrenamiento con la receta del profesor
def entrenar(capas, X_tr, X_va):
    red = MLPRegressor(hidden_layer_sizes=capas, activation="tanh", solver="adam", learning_rate_init=0.002, batch_size=64, alpha=0.0001,
                       random_state=SEMILLA, early_stopping=False)
    mejor, be, copia, tl, vl = np.inf, 0, None, [], []
    with threadpool_limits(limits=1):
        for e in range(1, EPOCAS + 1):
            red.partial_fit(X_tr, X_tr)
            a, b = mse(X_tr, red.predict(X_tr)), mse(X_va, red.predict(X_va))
            tl.append(a)
            vl.append(b)
            if b < mejor:
                mejor, be, copia = b, e, deepcopy(red)
    return dict(mejor=mejor, epoca=be, modelo=copia, tl=tl, vl=vl)


def nparams(m):
    return int(sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_)))


def codificar(m, X):
    """Capas del encoder con tanh hasta la capa central (misma lógica que la sección 06 del profesor)."""
    H = X.copy()
    for j in range(len(m.hidden_layer_sizes) // 2 + 1):
        H = np.tanh(H @ m.coefs_[j] + m.intercepts_[j])
    return H


def decodificar(m, H):
    """Capas posteriores a la central: tanh en las ocultas y salida lineal."""
    n = len(m.coefs_)
    for j in range(len(m.hidden_layer_sizes) // 2 + 1, n):
        H = H @ m.coefs_[j] + m.intercepts_[j]
        if j < n - 1:
            H = np.tanh(H)
    return H


def calc():
    if _C:
        return _C
    d = pd.read_csv(CSV)
    X = d.filter(regex="^px").to_numpy(float) / 16
    y = d.digito.to_numpy(int)
    ids = d.id.to_numpy(int)
    tr, va, te = (d.conjunto.eq(k).to_numpy() for k in ("entrenamiento", "validacion", "prueba"))
    assert (tr.sum(), va.sum(), te.sum()) == (1077, 360, 360)
    R = {"AE8": entrenar((8,), X[tr], X[va]), "AE32_8_32": entrenar((32, 8, 32), X[tr], X[va])}
    prof = R["AE32_8_32"]["modelo"]   # elegida por el profesor (menor MSE de validación)
    ae8 = R["AE8"]["modelo"]
    pca = PCA(n_components=8, svd_solver="full").fit(X[tr])
    media = X[tr].mean(0)
    _C.update(d=d, X=X, y=y, ids=ids, tr=tr, va=va, te=te, R=R, prof=prof, ae8=ae8, pca=pca, media=media)
    assert np.allclose(decodificar(prof, codificar(prof, X)), prof.predict(X))
    assert np.allclose(decodificar(ae8, codificar(ae8, X)), ae8.predict(X))
    return _C


def rec_pca(c, X, k=8):
    p = PCA(n_components=k, svd_solver="full").fit(c["X"][c["tr"]]) if k != 8 else c["pca"]
    return p.inverse_transform(p.transform(X))


def rec_media(c, X):
    return np.tile(c["media"], (len(X), 1))


# ------------------------------------------------------------------ números
def numeros():
    c = calc()
    X, y, ids, tr, va, te, R = c["X"], c["y"], c["ids"], c["tr"], c["va"], c["te"], c["R"]
    prof, ae8 = c["prof"], c["ae8"]
    S = {}  # resultados internos reutilizados por las figuras
    NUM["datos"] = dict(n=int(len(y)), n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum()), px=64)
    NUM["curvas"] = {k: dict(tr=[round(v, 6) for v in R[k]["tl"]], va=[round(v, 6) for v in R[k]["vl"]], epoca=int(R[k]["epoca"]), val=float(R[k]["mejor"]),
                             params=nparams(R[k]["modelo"]), tr_1=float(R[k]["tl"][0]), va_1=float(R[k]["vl"][0]), tr_150=float(R[k]["tl"][-1]), va_150=float(R[k]["vl"][-1]))
                      for k in R}
    # parámetros por capa
    def por_capa(m):
        return [dict(entra=int(w.shape[0]), sale=int(w.shape[1]), pesos=int(w.size), sesgos=int(b.size), total=int(w.size + b.size)) for w, b in zip(m.coefs_, m.intercepts_)]
    NUM["params"] = dict(ae8=por_capa(ae8), prof=por_capa(prof), total_ae8=nparams(ae8), total_prof=nparams(prof),
                         ae4=64 * 4 + 4 + 4 * 64 + 64, prof4=(64 * 32 + 32) + (32 * 4 + 4) + (4 * 32 + 32) + (32 * 64 + 64),
                         ae16=64 * 16 + 16 + 16 * 64 + 64)
    # MSE de las cuatro reconstrucciones
    m = {}
    for nom, f in (("media", lambda Z: rec_media(c, Z)), ("pca8", lambda Z: rec_pca(c, Z)), ("ae8", ae8.predict), ("prof", prof.predict)):
        m[nom] = dict(val=mse(X[va], f(X[va])), prueba=mse(X[te], f(X[te])), entrenamiento=mse(X[tr], f(X[tr])))
    NUM["mse"] = m
    NUM["mse"]["reduccion_pca_vs_media"] = 1 - m["pca8"]["val"] / m["media"]["val"]
    NUM["mse"]["reduccion_prof_vs_media"] = 1 - m["prof"]["val"] / m["media"]["val"]
    NUM["mse"]["reduccion_prof_vs_pca"] = 1 - m["prof"]["val"] / m["pca8"]["val"]
    # PCA con distintos k (diapositiva 30; nuestra reproducción)
    NUM["pca_k"] = {str(k): mse(X[va], rec_pca(c, X[va], k)) for k in (1, 2, 4, 8, 16, 32, 64)}
    # imagen del ejemplo (id 4, diapositivas 5, 29 y 41)
    i4 = int(np.where(ids == 4)[0][0])
    r4 = prof.predict(X)[i4]
    x4 = X[i4]
    err4 = (x4 - r4) ** 2
    des = np.zeros((8, 8))
    des[:, 1:] = x4.reshape(8, 8)[:, :-1]
    des = des.ravel()
    rdes = prof.predict(des[None])[0]
    NUM["pixel"] = dict(id=4, indice=i4, digito=int(y[i4]), fila=3, columna=4, x=float(x4[2 * 8 + 3]), xhat=float(r4[2 * 8 + 3]), dif=float(r4[2 * 8 + 3] - x4[2 * 8 + 3]),
                        cuadrado=float(err4[2 * 8 + 3]), mse_imagen=float(err4.mean()), suma=float(err4.sum()), mayor_pixel=int(err4.argmax()), mayor_valor=float(err4.max()),
                        mse_desplazada=mse(des, rdes))
    S.update(i4=i4, x4=x4, r4=r4, err4=err4, des=des, rdes=rdes)
    # código del «8» de la diapositiva 9 (id 9)
    i9 = int(np.where(ids == 9)[0][0])
    H = codificar(prof, X)
    S["H"] = H
    NUM["codigo8"] = dict(id=9, indice=i9, digito=int(y[i9]), conjunto=str(c["d"].conjunto.iloc[i9]), h=H[i9].tolist(), suma=float(H[i9].sum()),
                          mse=mse(X[i9], prof.predict(X[i9][None])[0]), h4=H[i4].tolist())
    S["i9"] = i9
    # ejemplos de juguete
    p, q = np.array([0.2, 0.8]), np.array([0.8, 0.2])
    def enc_media(x):
        return (x[0] + x[1]) / 2
    def dec_igual(h):
        return np.array([h, h])
    def enc_dif(x):
        return x[0] - x[1]
    def dec_dif(h):
        return np.array([0.5 + h / 2, 0.5 - h / 2])
    NUM["toy"] = dict(
        media=[dict(x=v.tolist(), h=float(enc_media(v)), xhat=dec_igual(enc_media(v)).tolist(), mse=mse(v, dec_igual(enc_media(v)))) for v in (p, q)],
        dif=[dict(x=v.tolist(), h=float(enc_dif(v)), xhat=dec_dif(enc_dif(v)).tolist(), mse=mse(v, dec_dif(enc_dif(v)))) for v in (p, q)],
        propuestas=[dict(xhat=z, mse=mse(p, np.array(z))) for z in ([0.2, 0.8], [0.3, 0.7], [0.5, 0.5])],
        mismo_cuadrado=dict(dif=[-0.3, 0.3], cuadrado=[0.09, 0.09]))
    NUM["tanh"] = {str(v): float(np.tanh(v)) for v in (-2, -1, -0.5, 0, 0.5, 1, 2, 3)}
    # 8 reales frente a 8 bits (ejemplo propio): el código se reemplaza por su signo (±1)
    Hva = H[va]
    Hs = np.where(Hva >= 0, 1.0, -1.0)
    rec_real = decodificar(prof, Hva)
    rec_bits = decodificar(prof, Hs)
    NUM["bits"] = dict(val_real=mse(X[va], rec_real), val_bits=mse(X[va], rec_bits), val_media=NUM["mse"]["media"]["val"], val_pca8=NUM["mse"]["pca8"]["val"],
                       patrones_posibles=2 ** 8, patrones_usados=int(len({tuple(r) for r in Hs})), coord_min=float(H.min()), coord_max=float(H.max()))
    for dec in (1, 2):
        NUM["bits"][f"val_redondeo{dec}"] = mse(X[va], decodificar(prof, np.round(Hva, dec)))
    # perturbar una coordenada del código del «3» (id 4) en 0.05: el decoder responde de forma continua
    h = H[i4].copy()
    h2 = h.copy()
    h2[0] += 0.05
    S["pert"] = (decodificar(prof, h[None])[0], decodificar(prof, h2[None])[0])
    NUM["bits"]["perturbar"] = dict(delta=0.05, mse_original=mse(x4, S["pert"][0]), cambio_medio=float(np.mean((S["pert"][1] - S["pert"][0]) ** 2) ** 0.5),
                                    cambio_max=float(np.abs(S["pert"][1] - S["pert"][0]).max()))
    # umbral: percentil 95 del error de validación (diapositivas 38-40)
    rv = prof.predict(X[va])
    ev = np.mean((X[va] - rv) ** 2, axis=1)
    thr = float(np.quantile(ev, 0.95))
    et = np.mean((X[te] - prof.predict(X[te])) ** 2, axis=1)
    Xs = np.zeros((int(te.sum()), 8, 8))
    Xs[:, :, 1:] = X[te].reshape(-1, 8, 8)[:, :, :-1]
    Xs = Xs.reshape(-1, 64)
    es = np.mean((Xs - prof.predict(Xs)) ** 2, axis=1)
    S.update(ev=ev, et=et, es=es, thr=thr, Xs=Xs)
    NUM["umbral"] = dict(cuantil=0.95, umbral=thr, n_val=int(va.sum()), val_sobre=int((ev > thr).sum()), val_pct=float((ev > thr).mean()), prueba_sobre=int((et > thr).sum()),
                         prueba_pct=float((et > thr).mean()), desplazadas_sobre=int((es > thr).sum()), desplazadas_pct=float((es > thr).mean()),
                         mediana_val=float(np.median(ev)), max_val=float(ev.max()), mediana_prueba=float(np.median(et)), mediana_desplazadas=float(np.median(es)))
    ale = et > thr
    NUM["umbral"]["por_digito"] = [dict(digito=k, alertas=int(ale[y[te] == k].sum()), imagenes=int((y[te] == k).sum())) for k in range(10)]
    o = np.argsort(-et)[:6]
    NUM["peores"] = [dict(id=int(ids[te][k]), digito=int(y[te][k]), mse=float(et[k]), alerta=bool(et[k] > thr)) for k in o]
    S["peores_idx"] = o
    # utilidad de la representación (diapositivas 34-35, sección 08 del profesor)
    Z = c["pca"].transform(X)
    util = {}
    for nom, rep in (("pixeles", X), ("pca8", Z), ("ae", H)):
        with threadpool_limits(limits=1):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, tol=1e-8)).fit(rep[tr], y[tr])
            pr = clf.predict_proba(rep[te])
            pd_ = clf.predict(rep[te])
        util[nom] = dict(log_loss=float(log_loss(y[te], pr)), exactitud=float((pd_ == y[te]).mean()), errores=int((pd_ != y[te]).sum()))
    NUM["utilidad"] = util
    # vecinos en el espacio latente (diapositiva 48)
    q8 = int(np.where(ids == 29)[0][0])
    dd = np.linalg.norm(H - H[q8], axis=1)
    dd[q8] = np.inf
    dd[~tr] = np.inf
    nn = np.argsort(dd)[:3]
    NUM["vecinos"] = dict(consulta_id=29, digito=int(y[q8]), dist=[float(dd[k]) for k in nn], digitos=[int(y[k]) for k in nn], ids=[int(ids[k]) for k in nn])
    S["q8"], S["nn"] = q8, nn
    # diapositiva 24: ids 4, 25, 29, 34 → códigos de 8 valores (segunda matriz de la diapositiva)
    NUM["ids_diapo24"] = {str(i): dict(digito=int(y[int(np.where(ids == i)[0][0])]), h=H[int(np.where(ids == i)[0][0])].tolist()) for i in (4, 25, 29, 34)}
    # primeros casos de cada dígito 0..5 en prueba (diapositiva 28)
    pri = [int(np.where(te & (y == k))[0][0]) for k in range(6)]
    S["pri"] = pri
    NUM["primeros_prueba"] = [dict(digito=k, id=int(ids[j])) for k, j in enumerate(pri)]
    # práctica de la diapositiva 53 (latente 4) y un ancho extra (nuestra reproducción; los usa el notebook)
    ex = {}
    for nom, capas in (("ae4", (4,)), ("prof4", (32, 4, 32)), ("ae16", (16,))):
        rr = entrenar(capas, X[tr], X[va])
        ex[nom] = dict(val=float(rr["mejor"]), epoca=int(rr["epoca"]), params=nparams(rr["modelo"]), prueba=mse(X[te], rr["modelo"].predict(X[te])))
    ex["pca4_val"] = mse(X[va], rec_pca(c, X[va], 4))
    ex["pca16_val"] = mse(X[va], rec_pca(c, X[va], 16))
    NUM["extra"] = ex
    # comparación diapositivas vs reproducción
    L = []
    def cmp(item, diap, rep, igual=None):
        L.append(dict(item=item, diapositiva=str(diap), reproducido=str(rep), igual=bool(diap == rep) if igual is None else bool(igual)))
    f4 = lambda v: f"{v:.4f}"
    f5 = lambda v: f"{v:.5f}"
    cmp("MSE validación 64-8-64 (mejor)", "0.02403 · época 150", f"{NUM['curvas']['AE8']['val']:.5f} · época {NUM['curvas']['AE8']['epoca']}")
    cmp("MSE validación 64-32-8-32-64 (mejor)", "0.01931 · época 150", f"{NUM['curvas']['AE32_8_32']['val']:.5f} · época {NUM['curvas']['AE32_8_32']['epoca']}")
    cmp("Imagen media: validación / prueba", "0.0729 / 0.0736", f"{f4(m['media']['val'])} / {f4(m['media']['prueba'])}")
    cmp("PCA 8: validación / prueba", "0.0235 / 0.0242", f"{f4(m['pca8']['val'])} / {f4(m['pca8']['prueba'])}")
    cmp("AE elegido: validación / prueba", "0.0193 / 0.0198", f"{f4(m['prof']['val'])} / {f4(m['prof']['prueba'])}")
    cmp("Parámetros 64-8-64", "520+576 = 1096", f"{NUM['params']['ae8'][0]['total']}+{NUM['params']['ae8'][1]['total']} = {NUM['params']['total_ae8']}")
    pp = NUM["params"]["prof"]
    cmp("Parámetros 64-32-8-32-64", "2080+264+288+2112 = 4744", "+".join(str(q["total"]) for q in pp) + f" = {NUM['params']['total_prof']}")
    cmp("Píxel fila 3, col. 4 (entrada → reconstrucción; cuadrado)", "0.8125 → 0.4089; 0.1629", f"{NUM['pixel']['x']:.4f} → {NUM['pixel']['xhat']:.4f}; {NUM['pixel']['cuadrado']:.4f}")
    cmp("MSE de esa imagen (3, id 4)", "0.017220", f"{NUM['pixel']['mse_imagen']:.6f}")
    cmp("Código del 8 (diap. 9), valores ordenados", "−0.56 −0.52 −0.34 −0.27 −0.14 −0.07 −0.01 0.04",
        " ".join(f"{v:.2f}".replace("-", "−") for v in sorted(NUM["codigo8"]["h"])))
    cmp("Umbral = percentil 95 de validación", "0.033108", f"{thr:.6f}")
    cmp("Validación sobre el umbral", "18 de 360", f"{NUM['umbral']['val_sobre']} de 360")
    cmp("Alertas en prueba: original / desplazada", "9.72 % / 88.61 %", f"{NUM['umbral']['prueba_pct'] * 100:.2f} % / {NUM['umbral']['desplazadas_pct'] * 100:.2f} %")
    cmp("MSE del 3 desplazado (diap. 41)", "0.0418", f"{NUM['pixel']['mse_desplazada']:.4f}")
    cmp("Clasificación log loss: píxeles / PCA 8 / AE", "0.0984 / 0.2692 / 0.2567", " / ".join(f4(util[k]["log_loss"]) for k in ("pixeles", "pca8", "ae")))
    cmp("Clasificación: errores de 360", "11 / 25 / 25", " / ".join(str(util[k]["errores"]) for k in ("pixeles", "pca8", "ae")))
    cmp("Vecinos del 8 (id 29): distancias", "0.246 · 0.284 · 0.311", " · ".join(f"{v:.3f}" for v in NUM["vecinos"]["dist"]))
    t = NUM["toy"]
    cmp("Ejemplo pequeño, referencia (0.2, 0.8): MSE con (0.5, 0.5) y con (0.3, 0.7)", "0.09 y 0.01", f"{t['propuestas'][2]['mse']:.2f} y {t['propuestas'][1]['mse']:.2f}")
    # el código del 8: la diapositiva da los 8 valores con 2 decimales; la igualdad se juzga por redondeo
    for r_ in L:
        if r_["item"].startswith("Código del 8"):
            r_["igual"] = bool(sorted(round(v, 2) for v in NUM["codigo8"]["h"]) == sorted([-0.56, -0.52, -0.34, -0.27, -0.14, -0.07, -0.01, 0.04]))
    NUM["comparacion"] = L
    NUM["todo_igual"] = all(r["igual"] for r in L)
    NUM["no_reproducible"] = ["Diapositiva 20 (MSE baja de 0.0400 a 0.0298): el programa no incluye ese paso aislado", "Diapositivas 12 y 36: ejemplos conceptuales sin código"]
    _C["S"] = S


# ------------------------------------------------------------------ helpers de dibujo
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def pix(ax, img, x, y, s, vmax=1.0, ec=INK, lw=1.0, cmap="gray_r"):
    """Dibuja una imagen cuadrada de lado s (unidades de 0.1 pulgada) con esquina inferior izquierda en (x, y)."""
    img = np.asarray(img, float)
    if img.ndim == 1:
        img = img.reshape(8, 8)
    ax.imshow(img, extent=(x, x + s, y, y + s), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=ec, lw=lw, zorder=3))


def flecha(ax, p, q, color=MUTED, lw=1.5, ms=12, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=4))


def caja(ax, x, y, w, h, t, col, fs=9, fc="white", bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0", fc=fc, ec=col, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc or INK, fontweight="bold" if bold else "normal", linespacing=1.3, zorder=3)


def casilla(ax, x, y, w, h, valor, fs=9, fmt="{:g}", vmax=1.0, ec=INK):
    g = 1 - float(np.clip(valor / vmax, 0, 1))
    ax.add_patch(Rectangle((x, y), w, h, fc=(g, g, g), ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, fmt.format(valor), ha="center", va="center", fontsize=fs, color="white" if g < 0.5 else INK, fontweight="bold", zorder=3)


# ------------------------------------------------------------------ figuras
def f_esquema():
    c = calc()
    S = c["S"]
    i9 = S["i9"]
    x = c["X"][i9]
    r = c["prof"].predict(c["X"])[i9]
    h = S["H"][i9]
    fig, ax = canvas(W_FIG, 4.1)
    ax.text(1, 39.0, "Autoencoder: la entrada se resume en 8 números y se vuelve a armar", fontsize=10.5, fontweight="bold", color=C46, va="center")
    # imágenes
    pix(ax, x, 2, 15, 15)
    ax.text(9.5, 11.5, "Entrada x", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(9.5, 8.2, "64 valores\n(8 × 8 píxeles)", ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.3)
    pix(ax, r, 57, 15, 15)
    ax.text(64.5, 11.5, r"Reconstrucción $\hat{x}$", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(64.5, 8.2, "64 valores\n(casi la misma imagen)", ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.3)
    # encoder: trapecio que se angosta
    ax.add_patch(Polygon([(20, 14), (20, 31), (35, 24.5), (35, 20.5)], closed=True, fc=C46_BG, ec=C46, lw=1.8, zorder=2))
    ax.text(27, 22.5, "Encoder\nf", ha="center", va="center", fontsize=10, fontweight="bold", color=C46, linespacing=1.3)
    ax.text(27, 11.5, "codificador", ha="center", fontsize=8.8, color=MUTED)
    # decoder: trapecio que se ensancha
    ax.add_patch(Polygon([(52, 14), (52, 31), (37, 24.5), (37, 20.5)][::-1], closed=True, fc=C46_BG, ec=C46, lw=1.8, zorder=2))
    ax.text(45, 22.5, "Decoder\ng", ha="center", va="center", fontsize=10, fontweight="bold", color=C46, linespacing=1.3)
    ax.text(45, 11.5, "decodificador", ha="center", fontsize=8.8, color=MUTED)
    # código en el centro (8 casillas apiladas entre los dos trapecios): se dibuja sobre el hueco
    flecha(ax, (17.3, 22.5), (19.6, 22.5))
    flecha(ax, (52.4, 22.5), (56.6, 22.5))
    # vector latente: 8 casillas coloreadas
    ax.text(36, 33.0, "Vector latente h", ha="center", fontsize=9.6, fontweight="bold", color=AMBER)
    # tira horizontal de 8 coordenadas bajo el título
    y0 = 4.0
    for j, v in enumerate(h):
        col = plt.cm.RdBu((np.clip(v, -1, 1) + 1) / 2)
        ax.add_patch(Rectangle((19.5 + j * 4.1, y0 + 0.2), 3.8, 3.8, fc=col, ec=INK, lw=0.9, zorder=2))
        ax.text(19.5 + j * 4.1 + 1.9, y0 - 1.3, f"{v:.2f}".replace("-", "−"), ha="center", fontsize=7.6, color=INK, zorder=3)
    ax.text(36, 8.4, "los 8 números de la imagen (cuello de botella)", ha="center", fontsize=8.6, color=AMBER, fontweight="bold")
    flecha(ax, (36, 20.0), (36, 9.4), color=AMBER, lw=1.4, ms=10, ls="--")
    # nota inferior
    ax.text(37, 0.6, "Ambas mitades aprenden juntas con un solo error: qué tan distinta quedó x̂ de x. La etiqueta del dígito no interviene.", ha="center", fontsize=8.6, color=INK)
    # contador de dimensiones
    ax.text(10, 33.6, "64", ha="center", fontsize=13, fontweight="bold", color=BLUE)
    ax.text(36, 28.3, "8", ha="center", fontsize=13, fontweight="bold", color=AMBER)
    ax.text(64.5, 33.6, "64", ha="center", fontsize=13, fontweight="bold", color=BLUE)
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(-0.5, 40.5)
    save(fig, "c46_esquema.png")


def f_codigo():
    c = calc()
    S = c["S"]
    fig = plt.figure(figsize=(W_FIG, 3.6))
    casos = [(S["i9"], "Un «8» (entrenamiento, id 9): su código"), (S["i4"], "Un «3» (prueba, id 4): otro código")]
    for r, (idx, tt) in enumerate(casos):
        top = 0.97 - r * 0.5
        a1 = fig.add_axes([0.01, top - 0.38, 0.2, 0.38])
        a1.imshow(c["X"][idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        a1.set_xticks([])
        a1.set_yticks([])
        a1.set_title("imagen de entrada", fontsize=9, color=MUTED)
        a2 = fig.add_axes([0.30, top - 0.42, 0.68, 0.36])
        h = S["H"][idx]
        a2.bar(range(1, 9), h, color=[BLUE if v >= 0 else RED for v in h], width=0.62)
        for j, v in enumerate(h):
            a2.text(j + 1, v + (0.07 if v >= 0 else -0.07), f"{v:.2f}".replace("-", "−"), ha="center", va="bottom" if v >= 0 else "top", fontsize=8.6)
        a2.axhline(0, color=INK, lw=0.8)
        a2.axhline(1, color=GRAY, lw=0.9, ls=":")
        a2.axhline(-1, color=GRAY, lw=0.9, ls=":")
        a2.set_ylim(-1.18, 1.18)
        a2.set_xticks(range(1, 9))
        a2.set_ylabel("valor de h", fontsize=9)
        if r == 1:
            a2.set_xlabel("coordenada latente", fontsize=9)
        titulo(a2, tt, 9.6)
        a2.tick_params(labelsize=8.6)
        style(a2, "y")
    fig.text(0.985, 0.935, "tanh: todo queda entre −1 y 1 (líneas punteadas)", ha="right", fontsize=8.4, color=MUTED)
    save(fig, "c46_codigo.png")


def f_clasif():
    c = calc()
    S = c["S"]
    P = NUM["pixel"]
    fig, ax = canvas(W_FIG, 4.1)
    ax.text(1, 39.0, "La misma imagen de entrada, dos tareas distintas", fontsize=10.5, fontweight="bold", color=C46, va="center")
    pix(ax, S["x4"], 2, 9, 17)
    ax.add_patch(Rectangle((2 + 3 * 17 / 8, 9 + (7 - 2) * 17 / 8), 17 / 8, 17 / 8, fill=False, ec=AMBER, lw=2.4, zorder=5))
    ax.text(10.5, 5.8, "Entrada: dígito 3\n64 intensidades", ha="center", va="top", fontsize=8.8, color=INK, linespacing=1.3)
    # clasificar (arriba)
    flecha(ax, (20.5, 21.5), (27.5, 30.5), color=BLUE, lw=1.6)
    caja(ax, 28, 28, 12, 8, "Red de\nclasificación", BLUE, fs=8.8)
    flecha(ax, (40.5, 32), (46.5, 32), color=BLUE, lw=1.6)
    for k in range(10):
        ax.add_patch(Rectangle((47 + k * 2.55, 28.8), 2.3, 6.4, fc="white", ec=BLUE, lw=1.1, zorder=2))
        ax.text(47 + k * 2.55 + 1.15, 32, str(k), ha="center", va="center", fontsize=8.6, color=BLUE, zorder=3)
    ax.text(47, 26.2, "10 salidas: una probabilidad por dígito", ha="left", fontsize=8.8, color=BLUE, fontweight="bold")
    ax.text(47, 23.6, "se compara con la etiqueta", ha="left", fontsize=8.4, color=MUTED)
    # reconstruir (abajo)
    flecha(ax, (20.5, 16.5), (27.5, 12.5), color=C46, lw=1.6)
    caja(ax, 28, 8.5, 12, 8, "Autoencoder", C46, fs=8.8)
    flecha(ax, (40.5, 12.5), (46.5, 12.5), color=C46, lw=1.6)
    pix(ax, S["r4"], 47, 5.5, 14)
    ax.add_patch(Rectangle((47 + 3 * 14 / 8, 5.5 + (7 - 2) * 14 / 8), 14 / 8, 14 / 8, fill=False, ec=AMBER, lw=2.4, zorder=5))
    ax.text(63, 12.5, "64 salidas:\nuna intensidad por píxel,\nse compara con la\npropia entrada", ha="left", va="center", fontsize=8.8, color=C46, fontweight="bold", linespacing=1.35)
    ax.text(40, 1.6, f"Mismo píxel (fila 3, columna 4): entrada {P['x']:.4f} → reconstrucción {P['xhat']:.4f}", ha="center", fontsize=8.8, color=AMBER, fontweight="bold")
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(0, 41)
    save(fig, "c46_clasificar.png")


def f_toy():
    T = NUM["toy"]
    fig, ax = canvas(W_FIG, 5.5)
    ax.text(1, 53.4, "Ejemplo pequeño de la clase: 2 píxeles → 1 número → 2 píxeles", fontsize=11, fontweight="bold", color=C46, va="center")
    def fila(y, d, enc_txt, dec_txt, col, nota, mark):
        ax.add_patch(Rectangle((1, y - 1.2), 72.5, 10.4, fc="#FAFAFA", ec=GRID, lw=1.0, zorder=0))
        casilla(ax, 3, y + 1.0, 5.4, 6, d["x"][0], fmt="{:.1f}", fs=10)
        casilla(ax, 8.4, y + 1.0, 5.4, 6, d["x"][1], fmt="{:.1f}", fs=10)
        ax.text(8.4, y + 7.9, "entrada x", ha="center", fontsize=9, color=MUTED)
        flecha(ax, (14.8, y + 4), (18.6, y + 4))
        caja(ax, 19, y + 0.8, 14, 6.4, enc_txt, col, fs=9.4)
        flecha(ax, (33.6, y + 4), (37.6, y + 4))
        ax.add_patch(Circle((41.4, y + 4), 3.4, fc="white", ec=AMBER, lw=2.0, zorder=2))
        ax.text(41.4, y + 4, f"{d['h']:.1f}".replace("-", "−"), ha="center", va="center", fontsize=10.4, fontweight="bold", color=AMBER, zorder=3)
        ax.text(41.4, y + 8.0, "código h", ha="center", fontsize=9, color=AMBER)
        flecha(ax, (45.2, y + 4), (49.2, y + 4))
        caja(ax, 49.6, y + 0.8, 11.5, 6.4, dec_txt, col, fs=9.4)
        flecha(ax, (61.6, y + 4), (65.2, y + 4))
        casilla(ax, 65.6, y + 1.0, 3.6, 6, d["xhat"][0], fmt="{:.1f}", fs=9)
        casilla(ax, 69.2, y + 1.0, 3.6, 6, d["xhat"][1], fmt="{:.1f}", fs=9)
        ax.text(69.2, y + 7.9, r"reconstrucción $\hat{x}$", ha="center", fontsize=9, color=MUTED)
        ax.text(74.2, y + 4, f"{nota}\nMSE = {d['mse']:.2f}", ha="left", va="center", fontsize=9.2, color=mark, fontweight="bold", linespacing=1.4)
    ax.text(1, 48.6, "Encoder fijo h = (x₁ + x₂)/2 y decoder fijo (h, h), como en la diapositiva 11:", fontsize=9.8, color=INK, va="center")
    fila(37.2, T["media"][0], "h = (x₁+x₂)/2", "(h, h)", C46, "mismo\ncódigo", RED)
    fila(26.8, T["media"][1], "h = (x₁+x₂)/2", "(h, h)", C46, "mismo\ncódigo", RED)
    ax.text(1, 22.4, "Ejemplo propio (no es de la clase): otro encoder fijo, h = x₁ − x₂, y decoder (0.5 + h/2, 0.5 − h/2):", fontsize=9.8, color=INK, va="center")
    fila(11.6, T["dif"][0], "h = x₁ − x₂", "(0.5+h/2,\n0.5−h/2)", TEAL, "código\ndistinto", GREEN)
    fila(1.2, T["dif"][1], "h = x₁ − x₂", "(0.5+h/2,\n0.5−h/2)", TEAL, "código\ndistinto", GREEN)
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(0, 55)
    save(fig, "c46_toy.png")


def f_bits():
    c = calc()
    S = c["S"]
    B = NUM["bits"]
    X = c["X"]
    prof = c["prof"]
    Hall = S["H"]
    pri = S["pri"]
    Xi = X[pri]
    Hr = Hall[pri]
    Rr = decodificar(prof, Hr)
    Rb = decodificar(prof, np.where(Hr >= 0, 1.0, -1.0))
    fig, axes = plt.subplots(3, 6, figsize=(W_FIG, 4.1))
    filas = [("Original", Xi, None), ("Código real\n(8 números)", Rr, "r"), ("Código de signos\n(8 bits: ±1)", Rb, "b")]
    for r, (nom, imgs, _) in enumerate(filas):
        for k in range(6):
            ax = axes[r, k]
            ax.imshow(np.clip(imgs[k].reshape(8, 8), 0, 1), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nom, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
            if r > 0:
                ax.set_xlabel(f"MSE {mse(Xi[k], imgs[k]):.3f}", fontsize=8.4, color=INK if r == 1 else RED)
            else:
                ax.set_title(f"dígito {k}", fontsize=9, color=MUTED)
    fig.suptitle(f"8 números reales no son 8 bits · MSE de validación: código real {B['val_real']:.4f} frente a solo signos {B['val_bits']:.4f}",
                 fontsize=9.4, x=0.01, ha="left", color=INK, y=1.0)
    fig.tight_layout(rect=(0.1, 0, 1, 0.95))
    save(fig, "c46_bits.png")


def f_mse_toy():
    T = NUM["toy"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.3), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ax = axes[0]
    d = np.linspace(-1, 1, 200)
    ax.plot(d, d ** 2, color=C46, lw=2.4)
    for dv in (-0.3, 0.3):
        ax.scatter([dv], [dv ** 2], color=AMBER, s=60, zorder=3)
        ax.annotate(f"({dv:g}; {dv ** 2:.2f})".replace("-", "−"), (dv, dv ** 2), xytext=(dv * 0.95, 0.5), fontsize=9, color=AMBER, ha="center",
                    arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.9))
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("diferencia = reconstrucción − referencia", fontsize=9)
    ax.set_ylabel("diferencia al cuadrado", fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.03, 1.05)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Elevar al cuadrado: −0.3 y +0.3 cuestan igual", 9.4)
    style(ax)
    ax = axes[1]
    nombres = ["(0.2; 0.8)", "(0.3; 0.7)", "(0.5; 0.5)"]
    vals = [p["mse"] for p in T["propuestas"]]
    ax.bar(range(3), vals, color=[GREEN, AMBER, RED], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.2f}", ha="center", fontsize=9.6, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(nombres, fontsize=9)
    ax.set_ylim(0, 0.115)
    ax.set_xlabel("reconstrucción propuesta", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Referencia (0.2; 0.8): tres reconstrucciones", 9.4)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c46_mse_toy.png")


def f_pixel():
    c = calc()
    S = c["S"]
    P = NUM["pixel"]
    fig = plt.figure(figsize=(W_FIG, 3.1))
    axs = [fig.add_axes([0.01 + i * 0.27, 0.13, 0.25, 0.7]) for i in range(3)]
    imgs = [(S["x4"], "Original x", "gray_r", 0, 1), (np.clip(S["r4"], 0, 1), "Reconstrucción", "gray_r", 0, 1), (S["err4"], "Error² por píxel", "magma_r", 0, 0.16)]
    for a, (im, tt, cm, lo, hi) in zip(axs, imgs):
        h = a.imshow(im.reshape(8, 8), cmap=cm, vmin=lo, vmax=hi)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(tt, fontsize=9.6, color=INK)
        a.add_patch(Rectangle((2.5, 1.5), 1, 1, fill=False, ec=AMBER, lw=2.6))
    cb = fig.colorbar(h, ax=axs[2], fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    # etiquetas de los píxeles de mayor error
    big = np.argsort(-S["err4"])[:4]
    for k in big:
        r_, c_ = divmod(int(k), 8)
        axs[2].text(c_, r_, f"{S['err4'][k]:.2f}", ha="center", va="center", fontsize=7.4, color="white" if S["err4"][k] > 0.08 else INK)
    fig.text(0.01, 0.93, f"Dígito 3 (id 4): MSE de la imagen = promedio de los 64 cuadrados = {P['mse_imagen']:.6f}", fontsize=9.6, fontweight="bold", color=C46)
    fig.text(0.01, 0.03, f"Cuadro naranja (fila 3, columna 4): ({P['xhat']:.4f} − {P['x']:.4f})² = ({str(round(P['dif'], 4)).replace('-', '−')})² = {P['cuadrado']:.4f}", fontsize=9.4, color=AMBER, fontweight="bold")
    save(fig, "c46_pixel.png")


def f_reconstrucciones():
    c = calc()
    S = c["S"]
    X = c["X"]
    pri = S["pri"]
    Xi = X[pri]
    filas = [("Original", Xi), ("Imagen media", rec_media(c, Xi)), ("PCA 8", rec_pca(c, Xi)), ("AE 64–8–64", c["ae8"].predict(Xi)), ("AE profundo\n64–32–8–32–64", c["prof"].predict(Xi))]
    fig, axes = plt.subplots(5, 6, figsize=(W_FIG, 5.6))
    for r, (nom, imgs) in enumerate(filas):
        for k in range(6):
            ax = axes[r, k]
            ax.imshow(np.clip(imgs[k].reshape(8, 8), 0, 1), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nom, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
            if r == 0:
                ax.set_title(f"dígito {k}", fontsize=9.4, color=MUTED)
            else:
                ax.set_xlabel(f"MSE {mse(Xi[k], imgs[k]):.3f}", fontsize=8.4)
    fig.suptitle("Primer caso de cada dígito 0 a 5 en prueba (diapositiva 28) · se muestra recortado a [0, 1]; el MSE usa las salidas completas", fontsize=8.8, x=0.01, ha="left", y=1.0, color=INK)
    fig.tight_layout(rect=(0.1, 0, 1, 0.97))
    save(fig, "c46_reconstrucciones.png")


def f_mse_barras():
    m = NUM["mse"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    nombres = ["Imagen\nmedia", "PCA 8", "AE\n64–8–64", "AE profundo\n64–32–8–32–64"]
    ks = ["media", "pca8", "ae8", "prof"]
    x = np.arange(4)
    ax.bar(x - 0.19, [m[k]["val"] for k in ks], 0.36, color=C46, label="Validación")
    ax.bar(x + 0.19, [m[k]["prueba"] for k in ks], 0.36, color=GRAY, label="Prueba histórica")
    for i, k in enumerate(ks):
        ax.text(i - 0.19, m[k]["val"] + 0.0015, f"{m[k]['val']:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.text(i + 0.19, m[k]["prueba"] + 0.0015, f"{m[k]['prueba']:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=8.6)
    ax.set_ylabel("MSE por píxel", fontsize=9)
    ax.set_ylim(0, 0.098)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")
    titulo(ax, "Menor MSE es mejor", 9.6)
    style(ax, "y")
    ax = axes[1]
    ks2 = [int(k) for k in NUM["pca_k"]]
    ax.plot(ks2, [NUM["pca_k"][str(k)] for k in ks2], "o-", color=C46, lw=2, label="PCA con k componentes")
    ax.scatter([8], [m["ae8"]["val"]], color=AMBER, s=70, marker="D", zorder=3, label="AE 64–8–64")
    ax.scatter([8], [m["prof"]["val"]], color=GREEN, s=60, zorder=3, label="AE profundo")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks2)
    ax.set_xticklabels([str(k) for k in ks2], fontsize=8.6)
    ax.set_xlabel("dimensión del resumen k (validación)", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    titulo(ax, "Más dimensiones, menos error", 9.6)
    style(ax)
    fig.tight_layout()
    save(fig, "c46_mse_barras.png")


def f_arquitecturas():
    P = NUM["params"]
    fig, ax = canvas(W_FIG, 4.8)
    def red(y0, tam, colores, etiquetas, tit, params, total, ancho):
        ax.text(1, y0 + 17.2, tit, fontsize=10, fontweight="bold", color=C46)
        n = len(tam)
        xs = np.linspace(8, 60, n)
        for i, (t, x) in enumerate(zip(tam, xs)):
            h = 4 + 10 * np.sqrt(t / 64)
            ax.add_patch(FancyBboxPatch((x - 3.2, y0 + 8 - h / 2), 6.4, h, boxstyle="round,pad=0.1,rounding_size=0.8", fc=colores[i][1], ec=colores[i][0], lw=1.8, zorder=2))
            ax.text(x, y0 + 8, str(t), ha="center", va="center", fontsize=10, fontweight="bold", color=colores[i][0], zorder=3)
            ax.text(x, y0 - 0.2, etiquetas[i], ha="center", va="top", fontsize=8.2, color=MUTED, linespacing=1.2)
        for i in range(n - 1):
            x0, x1 = xs[i] + 3.6, xs[i + 1] - 3.6
            flecha(ax, (x0, y0 + 8), (x1, y0 + 8), ms=10)
            ax.text((x0 + x1) / 2, y0 + 10.2, f"{params[i]:,}".replace(",", " "), ha="center", fontsize=8.8, fontweight="bold", color=AMBER)
        ax.text(66, y0 + 8, f"Total\n{total:,}".replace(",", " ") + "\nparámetros", ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK, linespacing=1.4)
    col = {"e": (BLUE, "#EFF6FF"), "o": (TEAL, "#F0FDFA"), "l": (AMBER, "#FFFBEB"), "s": (GREEN, "#F0FDF4")}
    red(21.5, [64, 8, 64], [col["e"], col["l"], col["s"]], ["entrada", "latente\n(tanh)", "salida\n(lineal)"], "64 → 8 → 64", [q["total"] for q in P["ae8"]], P["total_ae8"], 0)
    red(0, [64, 32, 8, 32, 64], [col["e"], col["o"], col["l"], col["o"], col["s"]], ["entrada", "oculta\n(tanh)", "latente\n(tanh)", "oculta\n(tanh)", "salida\n(lineal)"],
        "64 → 32 → 8 → 32 → 64", [q["total"] for q in P["prof"]], P["total_prof"], 0)
    ax.text(1, -7.4, "Sobre cada flecha: pesos + sesgos de esa capa (unidades × (entradas + 1)). Ambas redes resumen cada imagen en 8 valores; la profunda tiene más parámetros.", fontsize=8.4, color=INK)
    ax.set_xlim(0, W_FIG * 10)
    ax.set_ylim(-9, 39)
    save(fig, "c46_arquitecturas.png")


def f_curvas():
    cu = NUM["curvas"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.3), sharey=True)
    for ax, k, t in ((axes[0], "AE8", "64–8–64"), (axes[1], "AE32_8_32", "64–32–8–32–64")):
        e = np.arange(1, EPOCAS + 1)
        ax.plot(e, cu[k]["tr"], color=C46, lw=2.0, label="Entrenamiento")
        ax.plot(e, cu[k]["va"], color=AMBER, lw=2.0, label="Validación")
        ax.scatter([cu[k]["epoca"]], [cu[k]["val"]], color=RED, s=48, zorder=4)
        ax.annotate(f"mínimo {cu[k]['val']:.5f}\nen la época {cu[k]['epoca']} (la última)", (cu[k]["epoca"], cu[k]["val"]), xytext=(45, 0.0385), fontsize=8.8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
        ax.text(148, 0.0425, f"época 1: {cu[k]['va_1']:.3f}\n(fuera del gráfico)", ha="right", fontsize=8.2, color=MUTED)
        ax.set_xlabel("época", fontsize=9)
        ax.tick_params(labelsize=8.6)
        ax.set_ylim(0.015, 0.046)
        titulo(ax, t, 9.8)
        style(ax)
    axes[0].set_ylabel("MSE por píxel", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.8, loc="lower left")
    fig.tight_layout()
    save(fig, "c46_curvas.png")


def f_tanh():
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 2.9))
    a = np.linspace(-3, 3, 200)
    ax = axes[0]
    ax.plot(a, np.tanh(a), color=AMBER, lw=2.4)
    pts = [-2, -1, 0, 1, 2]
    ax.scatter(pts, np.tanh(pts), color=INK, s=26, zorder=3)
    ax.text(0.97, 0.06, "tanh(−1) = −0.7616\ntanh(1) = 0.7616\ntanh(2) = 0.9640", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8, color=INK, linespacing=1.4,
            bbox=dict(fc="white", ec=GRID, pad=3))
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("combinación a (pesos · entradas + sesgo)", fontsize=9)
    ax.set_ylabel("activación h", fontsize=9)
    ax.set_ylim(-1.25, 1.25)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Capas ocultas: tanh acota a (−1, 1)", 9.4)
    style(ax)
    ax = axes[1]
    ax.plot(a, a, color=TEAL, lw=2.4)
    ax.axhspan(0, 1, color=C46_BG, zorder=0)
    ax.text(-2.8, 0.5, "[0, 1]: rango\nde las intensidades", fontsize=8.2, color=C46, va="center")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("combinación s (capa de salida)", fontsize=9)
    ax.set_ylabel("intensidad reconstruida", fontsize=9)
    ax.set_ylim(-3.1, 3.1)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Salida lineal: puede salirse de [0, 1]", 9.4)
    style(ax)
    fig.tight_layout()
    save(fig, "c46_tanh.png")


def f_codigos2d():
    c = calc()
    S = c["S"]
    te, y = c["te"], c["y"]
    Zp = c["pca"].transform(c["X"][te])[:, :2]
    Za = S["H"][te][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.7))
    cmap = plt.get_cmap("tab10")
    for ax, Z, t in ((axes[0], Zp, "PCA: coordenadas 1 y 2"), (axes[1], Za, "Autoencoder: coordenadas 1 y 2")):
        for k in range(10):
            m = y[te] == k
            ax.scatter(Z[m, 0], Z[m, 1], s=20, color=cmap(k), label=str(k), edgecolor="white", linewidth=0.3)
        ax.set_xlabel("coordenada 1", fontsize=9)
        ax.set_ylabel("coordenada 2", fontsize=9)
        ax.tick_params(labelsize=8.6)
        titulo(ax, t, 9.6)
        style(ax)
    axes[1].legend(title="dígito", frameon=False, fontsize=8, title_fontsize=8.6, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.suptitle("360 imágenes de prueba · colores = etiquetas añadidas solo para mirar; el autoencoder no las usó (se ven 2 de 8 coordenadas)", fontsize=8.6, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "c46_codigos2d.png")


def f_utilidad():
    U = NUM["utilidad"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 2.7))
    nom = ["Píxeles", "PCA 8", "Latente AE"]
    ks = ["pixeles", "pca8", "ae"]
    cols = [GRAY, C46, AMBER]
    ax = axes[0]
    ax.bar(range(3), [U[k]["log_loss"] for k in ks], color=cols, width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, U[k]["log_loss"] + 0.006, f"{U[k]['log_loss']:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(nom, fontsize=9)
    ax.set_ylabel("log loss (prueba)", fontsize=9)
    ax.set_ylim(0, 0.32)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Log loss de la logística", 9.6)
    style(ax, "y")
    ax = axes[1]
    ax.bar(range(3), [U[k]["errores"] for k in ks], color=cols, width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, U[k]["errores"] + 0.5, str(U[k]["errores"]), ha="center", fontsize=9.6, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(nom, fontsize=9)
    ax.set_ylabel("errores de 360", fontsize=9)
    ax.set_ylim(0, 31)
    ax.tick_params(labelsize=8.6)
    titulo(ax, "Errores de clasificación", 9.6)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c46_utilidad.png")


def f_umbral():
    c = calc()
    S = c["S"]
    U = NUM["umbral"]
    ev, thr = S["ev"], S["thr"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.4))
    ax = axes[0]
    o = np.sort(ev)
    n = len(o)
    ax.bar(range(n), o, color=[RED if v > thr else C46 for v in o], width=1.0, linewidth=0)
    ax.axhline(thr, color=GREEN, ls="--", lw=1.8)
    ax.text(3, thr + 0.0025, f"percentil 95 = {thr:.5f}", fontsize=8.8, color=GREEN, fontweight="bold")
    ax.set_xlabel("posición del error ordenado (de menor a mayor)", fontsize=9)
    ax.set_ylabel("MSE por imagen", fontsize=9)
    ax.tick_params(labelsize=8.6)
    titulo(ax, f"{U['val_sobre']} de {U['n_val']} errores de validación superan el corte", 9.4)
    style(ax, "y")
    ax = axes[1]
    ax.hist(ev, bins=24, color=C46, alpha=0.85, edgecolor="white")
    ax.axvline(thr, color=GREEN, ls="--", lw=1.8, label=f"umbral {thr:.5f}")
    ax.set_xlabel("MSE por imagen (validación)", fontsize=9)
    ax.set_ylabel("imágenes", fontsize=9)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.8)
    titulo(ax, "Casi todas se reconstruyen bien; hay una cola", 9.4)
    style(ax)
    fig.tight_layout()
    save(fig, "c46_umbral.png")


def f_peores():
    c = calc()
    S = c["S"]
    X, te, y, ids = c["X"], c["te"], c["y"], c["ids"]
    Xt = X[te]
    Rt = c["prof"].predict(Xt)
    o = S["peores_idx"]
    fig, axes = plt.subplots(2, 6, figsize=(W_FIG, 2.9))
    for k, j in enumerate(o):
        for r, (im, nm) in enumerate(((Xt[j], "original"), (Rt[j], "reconstrucción"))):
            ax = axes[r, k]
            ax.imshow(np.clip(im.reshape(8, 8), 0, 1), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"dígito {y[te][j]} · id {ids[te][j]}\nMSE {S['et'][j]:.3f}", fontsize=8.6, color=RED)
            if k == 0:
                ax.set_ylabel(nm, fontsize=8.8)
    fig.tight_layout()
    save(fig, "c46_peores.png")


def f_estres():
    c = calc()
    S = c["S"]
    U = NUM["umbral"]
    P = NUM["pixel"]
    fig = plt.figure(figsize=(W_FIG, 3.9))
    ims = [(S["x4"], "original"), (S["r4"], f"su reconstrucción\nMSE {P['mse_imagen']:.4f}"), (S["des"], "desplazada 1 columna\na la derecha"), (S["rdes"], f"su reconstrucción\nMSE {P['mse_desplazada']:.4f}")]
    for k, (im, tt) in enumerate(ims):
        a = fig.add_axes([0.02 + k * 0.245, 0.55, 0.22, 0.28])
        a.imshow(np.clip(im.reshape(8, 8), 0, 1), cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(tt, fontsize=8.8, color=INK)
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.31])
    bins = np.linspace(0, 0.2, 41)
    ax.hist(np.clip(S["et"], 0, 0.2), bins=bins, alpha=0.8, color=C46, label=f"Originales ({U['prueba_pct'] * 100:.2f} % sobre el umbral)")
    ax.hist(np.clip(S["es"], 0, 0.2), bins=bins, alpha=0.7, color=AMBER, label=f"Desplazadas ({U['desplazadas_pct'] * 100:.2f} % sobre el umbral)")
    ax.axvline(S["thr"], color=GREEN, ls="--", lw=1.8, label=f"umbral de validación {S['thr']:.5f}")
    ax.set_xlabel("MSE por imagen (prueba; valores sobre 0.2 se agrupan en el último tramo)", fontsize=9)
    ax.set_ylabel("imágenes", fontsize=9)
    ax.tick_params(labelsize=8.6)
    ax.legend(frameon=False, fontsize=8.8)
    style(ax)
    fig.text(0.02, 0.995, "Prueba de estrés artificial: cada salida se compara con su propia entrada", fontsize=9.8, fontweight="bold", color=C46, va="top")
    save(fig, "c46_estres.png")


PASOS = dict(esquema=f_esquema, codigo=f_codigo, clasificar=f_clasif, toy=f_toy, bits=f_bits, mse_toy=f_mse_toy, pixel=f_pixel, reconstrucciones=f_reconstrucciones,
             mse_barras=f_mse_barras, arquitecturas=f_arquitecturas, curvas=f_curvas, tanh=f_tanh, codigos2d=f_codigos2d, utilidad=f_utilidad, umbral=f_umbral,
             peores=f_peores, estres=f_estres)

if __name__ == "__main__":
    numeros()
    for k, fn in PASOS.items():
        fn()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c46_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("todas las cifras de las diapositivas reproducidas:", NUM["todo_igual"])
    for r in NUM["comparacion"]:
        print(("OK  " if r["igual"] else "DIF ") + r["item"], "|", r["diapositiva"], "|", r["reproducido"])
    print("ok")
