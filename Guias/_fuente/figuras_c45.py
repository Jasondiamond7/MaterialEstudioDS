"""Figuras de la Clase 45 (Redes neuronales · Prof. Eloy Alvarado).
Fuentes: diapositivas redes_neuronales.pdf (48), transcripción de la clase y el programa del profesor redes_neuronales.py (carpeta de la Clase 45).
Reproducción: se ejecuta la MISMA receta del profesor (64-16-10 y 64-32-10, ReLU + softmax, Adam tasa 0.002, lote 64, alpha 0.0001, 120 épocas,
semilla 20261020, elección por log loss de validación) con scikit-learn 1.9.1 y las cifras salen idénticas a las de las diapositivas.
Los números de las diapositivas se guardan en NUM["diapos"] junto a los reproducidos; lo demás (barridos de ancho y de alpha, Adam,
ejemplos) es cálculo propio y se rotula «nuestra reproducción».
Escribe c45_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c45.py"""
import json
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C45 = "#0F766E"
C45_BG = "#F0FDFA"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
GREEN = "#16A34A"
GRAY = "#94A3B8"
PURPLE = "#7C3AED"
NUM = {}
BASE = Path(__file__).resolve().parents[2]
CSV = next(BASE.glob("Clase 45*/digitos.csv"))
SEMILLA = 20261020
EPOCAS = 120
_C = {}
W_FIG = 7.4


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def ll(y, p):
    """Promedio de -log(probabilidad de la clase real) (misma definición que el programa del profesor)."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


def sigm(a):
    return 1 / (1 + np.exp(-a))


# ------------------------------------------------------------------ entrenamiento con la receta del profesor
def entrenar(h, X, y, tr, va, alpha=1e-4, epocas=EPOCAS, lr=0.002, seed=SEMILLA):
    red = MLPClassifier(hidden_layer_sizes=h, activation="relu", solver="adam", learning_rate_init=lr, batch_size=64, alpha=alpha,
                        random_state=seed, shuffle=True, early_stopping=False)
    mejor, be, copia, tl, vl = np.inf, 0, None, [], []
    with threadpool_limits(limits=1):
        for e in range(1, epocas + 1):
            red.partial_fit(X[tr], y[tr], classes=np.arange(10))
            a, b = ll(y[tr], red.predict_proba(X[tr])), ll(y[va], red.predict_proba(X[va]))
            tl.append(a)
            vl.append(b)
            if b < mejor:
                mejor, be, copia = b, e, deepcopy(red)
    return dict(mejor=mejor, epoca=be, modelo=copia, ultimo=red, tl=tl, vl=vl)


def nparams(m):
    return int(sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_)))


def calc():
    if _C:
        return _C
    d = pd.read_csv(CSV)
    X = d.filter(regex="^px").to_numpy(float) / 16
    y = d.digito.to_numpy(int)
    tr, va, te = (d.conjunto.eq(k).to_numpy() for k in ("entrenamiento", "validacion", "prueba"))
    assert (tr.sum(), va.sum(), te.sum()) == (1077, 360, 360)
    R = {16: entrenar((16,), X, y, tr, va), 32: entrenar((32,), X, y, tr, va)}
    el = min(R, key=lambda k: R[k]["mejor"])
    red = R[el]["modelo"]
    with threadpool_limits(limits=1):
        lg = LogisticRegression(C=1, max_iter=3000, tol=1e-8).fit(X[tr], y[tr])
    _C.update(d=d, X=X, y=y, tr=tr, va=va, te=te, R=R, el=el, red=red, lg=lg)
    return _C


# ------------------------------------------------------------------ números
def numeros_basicos():
    # neurona (diapositiva 6)
    x, w, b = np.array([0.5, 1.0]), np.array([0.8, -0.4]), 0.2
    esc = []
    for nombre, xx, bb in (("base", x, b), ("x2 = 2", np.array([0.5, 2.0]), b), ("b = 0.5", x, 0.5)):
        a = float(w @ xx + bb)
        esc.append(dict(nombre=nombre, x=xx.tolist(), b=bb, prod=(w * xx).tolist(), a=a, h=max(0.0, a)))
    NUM["neurona"] = dict(x=x.tolist(), w=w.tolist(), b=b, esc=esc)
    # activaciones (diapositivas 5 y 7)
    pts = [-2.0, -1.0, 0.0, 1.0, 3.0]
    NUM["activ"] = dict(pts=pts, relu=[max(0, v) for v in pts], sig=[float(sigm(v)) for v in pts], tanh=[float(np.tanh(v)) for v in pts])
    # capa y propagación (diapositivas 8, 12, 14, 15)
    W1 = np.array([[1.0, -1.0], [0.5, 0.5]])
    x1 = np.array([1.0, 2.0])
    a1 = x1 @ W1
    h1 = np.maximum(a1, 0)
    W2, b2 = np.array([0.5, -1.0]), -0.5
    s = float(h1 @ W2 + b2)
    Xb = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 1.0]])
    Ab = Xb @ W1
    NUM["capa"] = dict(x=x1.tolist(), W1=W1.tolist(), a=a1.tolist(), h=h1.tolist(), W2=W2.tolist(), b2=b2, s=s, p=float(sigm(s)),
                       Xb=Xb.tolist(), Ab=Ab.tolist(), Hb=np.maximum(Ab, 0).tolist())
    # parámetros (diapositiva 13)
    NUM["params"] = dict(W1=64 * 32, b1=32, W2=32 * 10, b2=10, total32=64 * 32 + 32 + 32 * 10 + 10, w16_1=64 * 16, b16_1=16, w16_2=16 * 10, b16_2=10,
                         total16=64 * 16 + 16 + 16 * 10 + 10, logistica=64 * 10 + 10, doble32=64 * 32 + 32 + 32 * 32 + 32 + 32 * 10 + 10)
    # no linealidad (diapositiva 11)
    pts4 = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    h_ = np.column_stack([np.maximum(pts4[:, 0] - pts4[:, 1], 0), np.maximum(pts4[:, 1] - pts4[:, 0], 0)])
    NUM["xor"] = dict(pts=pts4.tolist(), clase=[0, 0, 1, 1], h=h_.tolist(), suma=h_.sum(1).tolist())
    # composición lineal sin activación (diapositiva 10), ejemplo numérico propio
    Wa, ba, Wb, bb = np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([1.0, 0.0]), np.array([[2.0], [1.0]]), np.array([0.5])
    Wc = Wa @ Wb
    bc = ba @ Wb + bb
    xx = np.array([3.0, -1.0])
    NUM["compone"] = dict(W1=Wa.tolist(), b1=ba.tolist(), W2=Wb.tolist(), b2=bb.tolist(), Wc=Wc.tolist(), bc=bc.tolist(), x=xx.tolist(),
                          dos_capas=float(((xx @ Wa + ba) @ Wb + bb)[0]), una_capa=float((xx @ Wc + bc)[0]))
    # softmax (diapositiva 17)
    sc = np.array([0, np.log(2), 0])
    ex = np.exp(sc)
    NUM["softmax"] = dict(s=sc.tolist(), e=ex.tolist(), suma=float(ex.sum()), p=(ex / ex.sum()).tolist())
    s2 = np.array([100.0, 100 + np.log(2), 100.0])
    e2 = np.exp(s2 - s2.max())
    NUM["softmax"]["p_desplazado"] = (e2 / e2.sum()).tolist()
    # pérdida de una observación (diapositiva 18)
    NUM["perdida"] = {str(p): float(-np.log(p)) for p in (0.8, 0.5, 0.1, 0.99, 0.49, 0.51)}
    # descenso de gradiente 1-D (diapositivas 19, 20, 27)
    J = lambda w: (w - 3) ** 2
    w0, eta = 1.0, 0.1
    camino = [w0]
    for _ in range(12):
        camino.append(camino[-1] - eta * 2 * (camino[-1] - 3))
    NUM["gd"] = dict(w0=w0, eta=eta, J0=J(w0), grad0=2 * (w0 - 3), w1=camino[1], J1=J(camino[1]), camino=camino, Jcamino=[J(v) for v in camino])
    tasas = {}
    for e in (0.05, 0.1, 0.4, 1.05):
        w, ls = 0.0, [9.0]
        for _ in range(10):
            w -= e * 2 * (w - 3)
            ls.append(J(w))
        tasas[str(e)] = dict(perdida=ls, factor=(1 - 2 * e) ** 2)
    NUM["gd"]["tasas"] = tasas
    # Adam sobre el mismo juguete (ilustración propia)
    def adam(escala, w0=0.0, eta=0.1, pasos=25, b1=0.9, b2=0.999, eps=1e-8):
        w, m, v = w0, 0.0, 0.0
        tray = [w]
        info = []
        for t in range(1, pasos + 1):
            g = escala * 2 * (w - 3)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
            paso = eta * mh / (np.sqrt(vh) + eps)
            w -= paso
            tray.append(float(w))
            info.append(dict(t=t, g=float(g), m=float(m), v=float(v), mh=float(mh), vh=float(vh), paso=float(paso), w=float(w)))
        return tray, info
    def gdesc(escala, w0=0.0, eta=0.1, pasos=25):
        w = w0
        tray = [w]
        for _ in range(pasos):
            w -= eta * escala * 2 * (w - 3)
            tray.append(float(w))
        return tray
    ta1, ia1 = adam(1.0)
    ta100, ia100 = adam(100.0)
    NUM["adam"] = dict(eta=0.1, beta1=0.9, beta2=0.999, eps=1e-8, tray_adam1=ta1, tray_adam100=ta100, tray_gd1=gdesc(1.0), tray_gd100=gdesc(100.0),
                       pasos1=ia1[:3], pasos100=ia100[:3])
    # ejemplo completo (diapositivas 21-24)
    th = np.array([0.4, 0.1, 0.8, -0.2])
    xe, ye = 0.5, 1
    def fwd(t):
        a1 = t[0] * xe + t[1]
        h = max(a1, 0)
        a2 = t[2] * h + t[3]
        p = sigm(a2)
        return a1, h, a2, p, float(-np.log(p))
    a1_, h_, a2_, p_, l_ = fwd(th)
    d2 = p_ - ye
    d1 = d2 * th[2] * float(a1_ > 0)
    g = np.array([d1 * xe, d1, d2 * h_, d2])
    eps = 1e-6
    num = np.array([(fwd(th + eps * v)[4] - fwd(th - eps * v)[4]) / (2 * eps) for v in np.eye(4)])
    th2 = th - 0.1 * g
    a1n, hn, a2n, pn, ln = fwd(th2)
    NUM["ejemplo"] = dict(x=xe, y=ye, theta=th.tolist(), a1=a1_, h=h_, a2=a2_, p=float(p_), l=l_, delta2=float(d2), delta1=float(d1), grad=g.tolist(), grad_num=num.tolist(),
                          dif_max=float(np.abs(g - num).max()), eta=0.1, theta2=th2.tolist(), a1n=a1n, hn=hn, a2n=a2n, pn=float(pn), ln=ln)
    # lotes y épocas (diapositiva 26)
    n, B = 1077, 64
    NUM["lotes"] = dict(n=n, B=B, completos=n // B, resto=n % B, lotes=-(-n // B), epocas=120, actualiz=120 * (-(-n // B)), por_candidata=120 * (-(-n // B)))
    # por qué difieren las métricas (diapositiva 38)
    NUM["metricas"] = dict(A=dict(p=[0.51, 0.51], perdida=float(np.mean([-np.log(0.51)] * 2)), errores=0),
                           B=dict(p=[0.99, 0.49], perdida=float(np.mean([-np.log(0.99), -np.log(0.49)])), errores=1))
    # inicialización simétrica (diapositiva 28): tres unidades con los mismos pesos dan lo mismo
    xs = np.array([[1.0, 2.0], [0.0, 1.0]])
    Wi = np.full((2, 3), 0.5)
    NUM["simetria"] = dict(x=xs.tolist(), W=Wi.tolist(), h=np.maximum(xs @ Wi, 0).tolist())


def numeros_entrenamiento():
    c = calc()
    X, y, tr, va, te, R, lg = c["X"], c["y"], c["tr"], c["va"], c["te"], c["R"], c["lg"]
    red = c["red"]
    NUM["datos"] = dict(n=int(len(y)), n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum()), por_clase_te=np.bincount(y[te], minlength=10).tolist(),
                        px=64, clases=10)
    NUM["curvas"] = {f"Red{k}": dict(tr=[round(v, 6) for v in R[k]["tl"]], va=[round(v, 6) for v in R[k]["vl"]], epoca=int(R[k]["epoca"]), val=float(R[k]["mejor"]),
                                     params=nparams(R[k]["modelo"]), tr_1=float(R[k]["tl"][0]), va_1=float(R[k]["vl"][0]), tr_120=float(R[k]["tl"][-1]), va_120=float(R[k]["vl"][-1]),
                                     tr_20=float(R[k]["tl"][19]), va_20=float(R[k]["vl"][19])) for k in R}
    NUM["elegida"] = f"Red{c['el']}"
    NUM["epoca_elegida"] = int(R[c["el"]]["epoca"])
    # prueba: logística vs red (diapositivas 37-41)
    out = {}
    for nm, m in (("log", lg), ("red", red)):
        res = {}
        for cj, mk in (("validacion", va), ("prueba", te)):
            P = m.predict_proba(X[mk])
            cl = P.argmax(1)
            res[cj] = dict(ll=ll(y[mk], P), errores=int((cl != y[mk]).sum()), acc=float((cl == y[mk]).mean()), n=int(mk.sum()))
        out[nm] = res
    NUM["eval"] = out
    P = red.predict_proba(X[te])
    cl = P.argmax(1)
    yt = y[te]
    ids = c["d"].loc[te, "id"].to_numpy()
    errs = [dict(pos=int(i), id=int(ids[i]), real=int(yt[i]), pred=int(cl[i]), pmax=float(P[i].max()), p_real=float(P[i, yt[i]]), perdida=float(-np.log(P[i, yt[i]])))
            for i in np.where(cl != yt)[0]]
    NUM["errores"] = errs
    NUM["errores_resumen"] = dict(n=len(errs), n_conf90=int(sum(e["pmax"] >= 0.9 for e in errs)), perdida_errores=float(sum(e["perdida"] for e in errs)),
                                  perdida_total=float(-np.log(P[np.arange(len(yt)), yt]).sum()))
    conf = np.zeros((10, 10), int)
    np.add.at(conf, (yt, cl), 1)
    NUM["confusion"] = conf.tolist()
    # logística: errores y confusiones (para comparar)
    Pl = lg.predict_proba(X[te])
    NUM["errores_log"] = [dict(pos=int(i), real=int(yt[i]), pred=int(Pl[i].argmax()), pmax=float(Pl[i].max())) for i in np.where(Pl.argmax(1) != yt)[0]]
    # barrido de ancho (nuestra reproducción, misma receta)
    anchos = {}
    for h in (2, 4, 8, 16, 32, 64):
        o = R[h] if h in R else entrenar((h,), X, y, tr, va)
        anchos[str(h)] = dict(val=float(o["mejor"]), epoca=int(o["epoca"]), params=nparams(o["modelo"]), tr_120=float(o["tl"][-1]))
    NUM["anchos"] = anchos
    # barrido de alpha con la red 64-32-10 (nuestra reproducción)
    alphas = {}
    for a in (0.0, 1e-4, 1e-2, 1e-1, 1.0, 3.0):
        o = R[32] if a == 1e-4 else entrenar((32,), X, y, tr, va, alpha=a)
        ult = o["ultimo"]
        alphas[str(a)] = dict(val_120=float(o["vl"][-1]), tr_120=float(o["tl"][-1]), mejor=float(o["mejor"]), epoca=int(o["epoca"]),
                              suma_w2=float(sum((w ** 2).sum() for w in ult.coefs_)), acc_va=float((ult.predict(X[va]) == y[va]).mean()))
    NUM["alphas"] = alphas
    # con el programa del profesor: ¿reproduce lo de las diapositivas? (comparación)
    r16, r32 = NUM["curvas"]["Red16"], NUM["curvas"]["Red32"]
    ev = NUM["eval"]
    NUM["diapos"] = dict(
        val16=dict(valor=0.0894, epoca=116), val32=dict(valor=0.0633, epoca=118), params=dict(p32=2410, p16=1210, w1=2048, w2=320),
        prueba=dict(log_ll=0.1511, red_ll=0.0932, log_err=8, red_err=10, log_acc=97.78, red_acc=97.22), lotes=dict(lotes=17, actualiz=2040, ultimo=53),
        gd=dict(J0=4, grad0=-4, w1=1.4, J1=2.56),
        ejemplo=dict(l=0.673347, p=0.509999, delta2=-0.4900, grad=[-0.1960, -0.3920, -0.1470, -0.4900], theta2=[0.4196, 0.1392, 0.8147, -0.1510], pn=0.533283, ln=0.628702),
        capa=dict(h=[2, 0], s=0.5, p=0.6225), metricas=dict(A=0.6733, B=0.3617),
        errores6=[("9", "8", 0.64), ("1", "6", 0.70), ("1", "6", 0.75), ("7", "8", 0.52), ("5", "6", 0.88), ("6", "1", 0.99)],
        errores10=["9→8", "1→6", "1→6", "7→8", "5→6", "6→1", "8→4", "9→3", "9→3", "3→2"], n_conf90=1)
    D = NUM["diapos"]
    rep = lambda t, nd: t
    NUM["comparacion"] = [
        dict(item="Log loss de validación, Red16 (época)", diapositiva="0.0894 (116)", reproducido=f"{r16['val']:.4f} ({r16['epoca']})", igual=bool(round(r16['val'], 4) == 0.0894 and r16['epoca'] == 116)),
        dict(item="Log loss de validación, Red32 (época)", diapositiva="0.0633 (118)", reproducido=f"{r32['val']:.4f} ({r32['epoca']})", igual=bool(round(r32['val'], 4) == 0.0633 and r32['epoca'] == 118)),
        dict(item="Parámetros Red32 / Red16", diapositiva="2 410 / 1 210", reproducido=f"{r32['params']} / {r16['params']}", igual=bool(r32['params'] == 2410 and r16['params'] == 1210)),
        dict(item="Prueba: log loss logística / red", diapositiva="0.1511 / 0.0932", reproducido=f"{ev['log']['prueba']['ll']:.4f} / {ev['red']['prueba']['ll']:.4f}",
             igual=bool(round(ev['log']['prueba']['ll'], 4) == 0.1511 and round(ev['red']['prueba']['ll'], 4) == 0.0932)),
        dict(item="Prueba: errores logística / red (de 360)", diapositiva="8 / 10", reproducido=f"{ev['log']['prueba']['errores']} / {ev['red']['prueba']['errores']}",
             igual=bool(ev['log']['prueba']['errores'] == 8 and ev['red']['prueba']['errores'] == 10)),
        dict(item="Prueba: exactitud logística / red", diapositiva="97.78 % / 97.22 %", reproducido=f"{ev['log']['prueba']['acc'] * 100:.2f} % / {ev['red']['prueba']['acc'] * 100:.2f} %",
             igual=bool(round(ev['log']['prueba']['acc'] * 100, 2) == 97.78 and round(ev['red']['prueba']['acc'] * 100, 2) == 97.22)),
        dict(item="Primeros 6 errores (real→pred, p máx.)", diapositiva="9→8 0.64 · 1→6 0.70 · 1→6 0.75 · 7→8 0.52 · 5→6 0.88 · 6→1 0.99",
             reproducido=" · ".join(f"{e['real']}→{e['pred']} {e['pmax']:.2f}" for e in errs[:6]),
             igual=bool([(str(e['real']), str(e['pred']), round(e['pmax'], 2)) for e in errs[:6]] == [(a, b, p) for a, b, p in D['errores6']])),
        dict(item="Los 10 errores (real→pred)", diapositiva=" ".join(D['errores10']), reproducido=" ".join(f"{e['real']}→{e['pred']}" for e in errs),
             igual=bool([f"{e['real']}→{e['pred']}" for e in errs] == D['errores10'])),
        dict(item="Errores con p máx. ≥ 0.90", diapositiva="1 de 10", reproducido=f"{NUM['errores_resumen']['n_conf90']} de {len(errs)}", igual=bool(NUM['errores_resumen']['n_conf90'] == 1 and len(errs) == 10)),
        dict(item="Lotes por época / actualizaciones", diapositiva="17 / 2 040", reproducido=f"{NUM['lotes']['lotes']} / {NUM['lotes']['actualiz']}", igual=bool(NUM['lotes']['lotes'] == 17 and NUM['lotes']['actualiz'] == 2040)),
        dict(item="Ejemplo completo: pérdida antes → después", diapositiva="0.673347 → 0.628702", reproducido=f"{NUM['ejemplo']['l']:.6f} → {NUM['ejemplo']['ln']:.6f}",
             igual=bool(round(NUM['ejemplo']['l'], 6) == 0.673347 and round(NUM['ejemplo']['ln'], 6) == 0.628702)),
        dict(item="Ejemplo completo: p antes → después", diapositiva="0.509999 → 0.533283", reproducido=f"{NUM['ejemplo']['p']:.6f} → {NUM['ejemplo']['pn']:.6f}",
             igual=bool(round(NUM['ejemplo']['p'], 6) == 0.509999 and round(NUM['ejemplo']['pn'], 6) == 0.533283)),
    ]
    NUM["todo_igual"] = bool(all(r["igual"] for r in NUM["comparacion"]))
    # ¿la receta reproduciría los números de la Clase 44 (354 de 360, log loss 0.0657)?
    NUM["c44_check"] = dict(diapos_c44=dict(aciertos=354, ll=0.0657), receta_c45_prueba=dict(aciertos=360 - ev["red"]["prueba"]["errores"], ll=ev["red"]["prueba"]["ll"]))


# ------------------------------------------------------------------ figuras
def flecha(ax, p, q, color=MUTED, lw=1.4, ms=11, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", ls=ls))


def caja(ax, x, y, w, h, t, col, fs=8.4, fc="white", bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0", fc=fc, ec=col, lw=1.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc or INK, fontweight="bold" if bold else "normal", linespacing=1.3)


def f_ciclo():
    fig, ax = plt.subplots(figsize=(W_FIG, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.axis("off")
    pasos = [("1 · Propagar", "entrada → capas\n→ probabilidades", C45), ("2 · Medir", "pérdida: −log de la\nprob. de la clase real", BLUE),
             ("3 · Retropropagar", "cuánto influye cada\npeso en la pérdida", AMBER), ("4 · Actualizar", "mover cada parámetro\n un poquito (Adam)", RED)]
    w, g = 19.5, 6.0
    for i, (t, s, col) in enumerate(pasos):
        x = 2 + i * (w + g)
        caja(ax, x, 9, w, 16, "", col)
        ax.text(x + w / 2, 21.3, t, ha="center", fontsize=8.8, fontweight="bold", color=col)
        ax.text(x + w / 2, 14.2, s, ha="center", va="center", fontsize=7.9, color=INK, linespacing=1.3)
        if i < 3:
            flecha(ax, (x + w + 0.3, 17), (x + w + g - 0.3, 17), ms=10)
    x_end = 2 + 3 * (w + g) + w / 2
    x_ini = 2 + w / 2
    ax.plot([x_end, x_end], [8.6, 5], color=MUTED, lw=1.3, ls="--")
    ax.plot([x_end, x_ini], [5, 5], color=MUTED, lw=1.3, ls="--")
    flecha(ax, (x_ini, 5), (x_ini, 8.6), ms=10, ls="--")
    ax.text(50, 5.9, "se repite con el lote siguiente", ha="center", fontsize=8, color=MUTED)
    ax.text(50, 0.8, "Tras cada época (una pasada por todas las imágenes de entrenamiento) se mira la validación.", ha="center", fontsize=8.2, color=MUTED)
    save(fig, "c45_ciclo.png")


def f_neurona():
    E = NUM["neurona"]
    fig = plt.figure(figsize=(W_FIG, 3.3))
    ax = fig.add_axes([0.0, 0.04, 0.56, 0.86])
    ax.set_xlim(0, 100)
    ax.set_ylim(-14.5, 54.5)
    ax.axis("off")
    ax.set_title("Una neurona: suma ponderada + sesgo + activación", loc="left", fontsize=9.2, color=INK, x=0.02)
    x, w, b = E["x"], E["w"], E["b"]
    pos = [(8, 29), (8, 11)]
    for (px, py), xv, wv, pr, nm in zip(pos, x, w, E["esc"][0]["prod"], ("x₁", "x₂")):
        ax.add_patch(Circle((px, py), 5, fc="white", ec=BLUE, lw=1.7))
        ax.text(px, py, f"{xv:g}", ha="center", va="center", fontsize=9, fontweight="bold", color=BLUE)
        ax.text(px, py + 7.2, nm, ha="center", fontsize=8, color=MUTED)
        flecha(ax, (px + 5.2, py), (38.6, 20 + (py - 20) * 0.3), color=GRAY, ms=9)
        ax.text(px + 14, 20 + (py - 20) * 0.78 + (3.6 if py > 20 else -3.6), f"× {wv:g} = {pr:g}", ha="center", fontsize=8.2, color=INK)
    ax.add_patch(Circle((45, 20), 6, fc=C45_BG, ec=C45, lw=1.9))
    ax.text(45, 20, "Σ", ha="center", va="center", fontsize=13, fontweight="bold", color=C45)
    ax.text(45, 29.5, f"sesgo b = {b:g}", ha="center", fontsize=8.2, color=AMBER)
    flecha(ax, (45, 28.5), (45, 26.3), color=AMBER, ms=8)
    ax.text(45, 7.2, f"a = {E['esc'][0]['prod'][0]:g} + ({E['esc'][0]['prod'][1]:g}) + {b:g} = {E['esc'][0]['a']:g}", ha="center", fontsize=8.2, color=C45)
    flecha(ax, (51.4, 20), (63, 20), color=GRAY, ms=9)
    caja(ax, 63.5, 14.5, 17, 11, "ReLU\nmáx(0, a)", PURPLE, fs=8)
    flecha(ax, (81, 20), (90, 20), color=GRAY, ms=9)
    ax.add_patch(Circle((95, 20), 4.7, fc="white", ec=GREEN, lw=1.9))
    ax.text(95, 20, f"{E['esc'][0]['h']:g}", ha="center", va="center", fontsize=9, fontweight="bold", color=GREEN)
    ax.text(95, 27.5, "h", ha="center", fontsize=8, color=MUTED)
    ax = fig.add_axes([0.66, 0.17, 0.32, 0.72])
    a = np.linspace(-1, 1, 100)
    ax.plot(a, np.maximum(a, 0), color=PURPLE, lw=2.2)
    for esc, col, xt in zip(E["esc"], (C45, RED, AMBER), ((-0.95, 0.62), (-0.95, 0.27), (-0.95, 0.92))):
        ax.scatter([esc["a"]], [esc["h"]], color=col, s=48, zorder=3)
        ax.annotate(f"{esc['nombre']}: a = {esc['a']:g}, h = {esc['h']:g}", (esc["a"], esc["h"]), xytext=xt, fontsize=7.8, color=col, ha="left", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("suma a")
    ax.set_ylabel("activación h = ReLU(a)")
    titulo(ax, "Mismos pesos, tres casos")
    style(ax)
    save(fig, "c45_neurona.png")


def f_activaciones():
    A = NUM["activ"]
    a = np.linspace(-3, 3, 200)
    fig, axes = plt.subplots(1, 3, figsize=(W_FIG, 2.7), sharex=True)
    datos = [("ReLU = máx(0, a)", np.maximum(a, 0), "relu", PURPLE, "deja pasar lo positivo,\nlleva lo negativo a 0"),
             ("Sigmoide = 1/(1+e⁻ᵃ)", sigm(a), "sig", BLUE, "comprime todo a (0, 1)"),
             ("Tangente hiperbólica", np.tanh(a), "tanh", AMBER, "comprime todo a (−1, 1)")]
    for ax, (t, yv, k, col, nota) in zip(axes, datos):
        ax.plot(a, yv, color=col, lw=2.2)
        ax.scatter(A["pts"], A[k], color=INK, s=22, zorder=3)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        titulo(ax, t, 8.8)
        ax.text(0.04, 0.95, nota, transform=ax.transAxes, fontsize=7.8, va="top", color=MUTED)
        ax.set_xlabel("a")
        style(ax)
    axes[0].set_ylabel("activación")
    axes[0].set_ylim(-0.3, 3.2)
    axes[1].set_ylim(-0.1, 1.1)
    axes[2].set_ylim(-1.2, 1.2)
    fig.tight_layout()
    save(fig, "c45_activaciones.png")


def matriz(ax, x0, y0, M, color, titulo_=None, cw=5.2, ch=4.6, fs=8.6, fmt="{:g}"):
    M = np.array(M, float)
    if M.ndim == 1:
        M = M[None, :]
    r, c = M.shape
    ax.add_patch(FancyBboxPatch((x0 - 0.8, y0 - 0.5), c * cw + 1.6, r * ch + 1.0, boxstyle="round,pad=0.1,rounding_size=0.8", fc="white", ec=color, lw=1.7))
    for i in range(r):
        for j in range(c):
            ax.text(x0 + j * cw + cw / 2, y0 + (r - 1 - i) * ch + ch / 2, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs, color=INK)
    if titulo_:
        ax.text(x0 + c * cw / 2, y0 + r * ch + 2.2, titulo_, ha="center", fontsize=7.8, color=color)
    return x0 + c * cw + 0.8


def f_capa():
    C = NUM["capa"]
    fig, ax = plt.subplots(figsize=(W_FIG, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    # una observación
    y1 = 36
    ax.text(0, 49, "Una observación", fontsize=9, fontweight="bold", color=C45)
    xe = matriz(ax, 2, y1 + 1.5, C["x"], BLUE, "x  (1×2)")
    ax.text(xe + 2, y1 + 3.8, "×", fontsize=13, ha="center", color=INK)
    xe = matriz(ax, xe + 5, y1 - 2.5, C["W1"], AMBER, "W₁  (2×2)", cw=5.6)
    ax.text(xe + 2, y1 + 3.8, "=", fontsize=13, ha="center", color=INK)
    xe = matriz(ax, xe + 5, y1 + 1.5, C["a"], C45, "a  (1×2)")
    flecha(ax, (xe + 1.5, y1 + 3.8), (xe + 7.5, y1 + 3.8), ms=9)
    ax.text(xe + 4.5, y1 + 6.2, "ReLU", fontsize=8, ha="center", color=PURPLE)
    xe = matriz(ax, xe + 9, y1 + 1.5, C["h"], GREEN, "h  (1×2)")
    ax.text(xe + 3, y1 + 3.8, "(sesgos = 0)", fontsize=7.8, color=MUTED, va="center")
    ax.text(2, y1 - 6.5, "cada columna de W₁ = los pesos de una unidad", fontsize=7.8, color=AMBER)
    ax.text(0, 26.5, f"a₁ = 1·1 + 2·0.5 = 2   ·   a₂ = 1·(−1) + 2·0.5 = 0   →   h = (2, 0)", fontsize=8.2, color=INK)
    # tres observaciones
    y2 = 1
    ax.text(0, 22, "Tres observaciones a la vez (cada fila usa los mismos pesos y sesgos)", fontsize=9, fontweight="bold", color=C45)
    xe = matriz(ax, 2, y2 + 1, C["Xb"], BLUE, "X  (3×2)")
    ax.text(xe + 2, y2 + 8.2, "×", fontsize=13, ha="center", color=INK)
    xe = matriz(ax, xe + 5, y2 + 4, C["W1"], AMBER, "W₁  (2×2)", cw=5.6)
    ax.text(xe + 2, y2 + 8.2, "=", fontsize=13, ha="center", color=INK)
    xe = matriz(ax, xe + 5, y2 + 1, C["Ab"], C45, "A  (3×2)")
    flecha(ax, (xe + 1.5, y2 + 8), (xe + 7.5, y2 + 8), ms=9)
    ax.text(xe + 4.5, y2 + 10.4, "ReLU", fontsize=8, ha="center", color=PURPLE)
    xe = matriz(ax, xe + 9, y2 + 1, C["Hb"], GREEN, "H  (3×2)")
    ax.text(xe + 3, y2 + 8, "el −1.5 pasa a 0", fontsize=7.8, color=RED, va="center")
    save(fig, "c45_capa.png")


def f_red():
    P = NUM["params"]
    fig = plt.figure(figsize=(W_FIG, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-4, 104)
    ax.set_ylim(-1, 56)
    ax.axis("off")
    xs = (13, 50, 87)
    ys_in = np.linspace(46, 13, 7)
    ys_h = np.linspace(46, 12, 6)
    ys_o = np.linspace(48, 10, 10)
    ys_hid = ys_h
    for yi in ys_in:
        for yh in ys_hid:
            ax.plot([xs[0], xs[1]], [yi, yh], color="#CBD5E1", lw=0.5, zorder=1)
    for yh in ys_hid:
        for yo in ys_o:
            ax.plot([xs[1], xs[2]], [yh, yo], color="#CBD5E1", lw=0.5, zorder=1)
    for yi in ys_in:
        ax.add_patch(Circle((xs[0], yi), 1.8, fc="white", ec=BLUE, lw=1.4, zorder=3))
    for yh in ys_hid:
        ax.add_patch(Circle((xs[1], yh), 1.8, fc="white", ec=C45, lw=1.4, zorder=3))
    for k, yo in enumerate(ys_o):
        ax.add_patch(Circle((xs[2], yo), 1.8, fc="white", ec=GREEN, lw=1.4, zorder=3))
        ax.text(xs[2] + 3.6, yo, str(k), fontsize=7.6, va="center", color=MUTED)
    for xx, cc in ((xs[0], BLUE), (xs[1], C45)):
        ax.scatter([xx] * 3, [9.2, 7.4, 5.6], s=5, color=cc, zorder=3)
    ax.text(xs[0] + 4, 7.4, "(hay 64)", fontsize=7.8, color=MUTED, va="center")
    ax.text(xs[1] + 4, 7.4, "(hay 32)", fontsize=7.8, color=MUTED, va="center")
    ax.text(xs[0], 52.5, "Entrada: 64", fontsize=9.2, fontweight="bold", color=BLUE, ha="center")
    ax.text(xs[0], 49.4, "píxeles ÷ 16", fontsize=7.8, color=MUTED, ha="center")
    ax.text(xs[1], 52.5, "Capa oculta: 32 unidades", fontsize=9.2, fontweight="bold", color=C45, ha="center")
    ax.text(xs[1], 49.4, "ReLU", fontsize=7.8, color=MUTED, ha="center")
    ax.text(xs[2] - 3, 52.5, "Salida: 10", fontsize=9.2, fontweight="bold", color=GREEN, ha="center")
    ax.text(xs[2] - 3, 49.4, "softmax", fontsize=7.8, color=MUTED, ha="center")
    ax.text(31.5, 29, f"W₁: 64×32\n= {P['W1']:,}".replace(",", " ") + f"\n+ b₁: {P['b1']}", ha="center", fontsize=8, color=AMBER, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5, linespacing=1.3)
    ax.text(68.5, 29, f"W₂: 32×10\n= {P['W2']}\n+ b₂: {P['b2']}", ha="center", fontsize=8, color=AMBER, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5, linespacing=1.3)
    ax.text(50, 0.5, f"Parámetros: {P['W1']:,} + {P['b1']} + {P['W2']} + {P['b2']} = {P['total32']:,}  (con 16 unidades: 64·16+16 + 16·10+10 = {P['total16']:,})".replace(",", " "),
            ha="center", fontsize=8.4, fontweight="bold", color=INK)
    save(fig, "c45_red.png")


def f_xor():
    X = NUM["xor"]
    pts, cl, H = np.array(X["pts"]), np.array(X["clase"]), np.array(X["h"])
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.1))
    for ax, Z, t, xl, yl in ((axes[0], pts, "Entradas: ninguna recta separa las clases", "primera coordenada x₁", "segunda x₂"),
                             (axes[1], H, "Después de la capa oculta: sí hay recta", "h₁ = máx(0, x₁−x₂)", "h₂ = máx(0, x₂−x₁)")):
        for c, mk, col, nm in ((0, "o", GRAY, "clase 0"), (1, "^", C45, "clase 1")):
            m = cl == c
            ax.scatter(Z[m, 0], Z[m, 1], marker=mk, s=130, color=col, edgecolor="white", linewidth=1.2, label=nm, zorder=3)
        ax.set_xlim(-0.25, 1.3)
        ax.set_ylim(-0.25, 1.3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        titulo(ax, t, 8.8)
        style(ax)
    for (x, y), tx in zip(pts, ["(0,0)", "(1,1)", "(1,0)", "(0,1)"]):
        axes[0].text(x + 0.04, y + 0.06, tx, fontsize=8, color=INK)
    xx = np.linspace(-0.25, 1.3, 10)
    axes[1].plot(xx, 0.5 - xx, color=RED, ls="--", lw=1.6, label="h₁ + h₂ = 0.5")
    axes[1].text(0.08, 0.62, "(0,0) y (1,1) caen\nen el mismo punto (0,0)", fontsize=7.8, color=INK)
    axes[1].legend(frameon=False, fontsize=7.8, loc="upper right")
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper center")
    fig.tight_layout()
    save(fig, "c45_xor.png")


def f_softmax():
    S = NUM["softmax"]
    fig, axes = plt.subplots(1, 3, figsize=(W_FIG, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    lab = ["clase 1", "clase 2", "clase 3"]
    ax = axes[0]
    ax.bar(lab, S["e"], color=[GRAY, C45, GRAY], width=0.6)
    for i, v in enumerate(S["e"]):
        ax.text(i, v + 0.04, f"{v:g}", ha="center", fontsize=8.6)
    ax.set_ylim(0, 2.5)
    titulo(ax, "1) Exponenciar: eˢ", 8.8)
    style(ax, "y")
    ax = axes[1]
    ax.bar(lab, S["p"], color=[GRAY, C45, GRAY], width=0.6)
    for i, v in enumerate(S["p"]):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.6)
    ax.set_ylim(0, 0.62)
    titulo(ax, "2) Dividir por la suma (4)", 8.8)
    style(ax, "y")
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.3, "puntajes s = (0, ln 2, 0)", fontsize=8.4, color=INK)
    ax.text(0, 7.6, "eˢ = (1, 2, 1) · suma = 4", fontsize=8.4, color=INK)
    ax.text(0, 5.9, "p = (¼, ½, ¼)  suma = 1", fontsize=8.6, color=C45, fontweight="bold")
    ax.text(0, 3.4, "Truco numérico: se resta\nel máximo c antes de\nexponenciar. Sumar 100 a\ntodos los puntajes da\nlas mismas p.", fontsize=7.9, color=MUTED, va="top", linespacing=1.3)
    fig.tight_layout()
    save(fig, "c45_softmax.png")


def f_perdida():
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    p = np.linspace(0.02, 1, 300)
    ax.plot(p, -np.log(p), color=C45, lw=2.2)
    for pp, off in ((0.8, (0.0, 0.9)), (0.5, (0.1, 0.8)), (0.1, (0.1, 0.7))):
        v = -np.log(pp)
        ax.scatter([pp], [v], color=RED, zorder=3)
        ax.annotate(f"p = {pp:g} → {v:.3f}", (pp, v), xytext=(pp + off[0] - (0.0 if pp != 0.8 else 0.05), v + off[1]), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("probabilidad de la clase REAL")
    ax.set_ylabel("−log(p)")
    ax.set_ylim(0, 4.2)
    titulo(ax, "Pérdida de una observación", 9)
    style(ax)
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.2, "Cómo se lee", fontsize=9.2, fontweight="bold", color=C45)
    ax.text(0, 8.2, "p cerca de 1 → pérdida\ncasi 0 (acierto seguro)\n\np = 0.5 → 0.693\n\np cerca de 0 → pérdida\nenorme (error seguro)", fontsize=8.2, color=INK, va="top", linespacing=1.4)
    fig.tight_layout()
    save(fig, "c45_perdida.png")


def f_gd():
    G = NUM["gd"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    w = np.linspace(0, 6, 200)
    ax.plot(w, (w - 3) ** 2, color=C45, lw=2.2, label="J(w) = (w − 3)²")
    cam = G["camino"][:7]
    ax.plot(cam, [(v - 3) ** 2 for v in cam], "o-", color=RED, ms=5, lw=1, label="pasos con η = 0.1")
    ww = np.linspace(0, 2, 10)
    ax.plot(ww, 4 + G["grad0"] * (ww - 1), color=AMBER, lw=1.4, ls="--", label="tangente en w = 1 (pendiente −4)")
    ax.annotate("w = 1\nJ = 4", (1, 4), xytext=(2.3, 5.6), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"w = {G['w1']:g}\nJ = {G['J1']:g}", (G["w1"], G["J1"]), xytext=(3.3, 3.2), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12.5)
    ax.set_xlabel("parámetro w")
    ax.set_ylabel("pérdida J")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center")
    titulo(ax, "Bajar por la pendiente", 9)
    style(ax)
    ax = axes[1]
    for k, col in (("0.05", BLUE), ("0.1", C45), ("0.4", GREEN), ("1.05", RED)):
        v = np.array(G["tasas"][k]["perdida"])
        ax.semilogy(range(len(v)), v, "o-", ms=3.2, color=col, label=f"η = {k}")
    ax.set_xlabel("actualización")
    ax.set_ylabel("pérdida (escala log)")
    ax.legend(frameon=False, fontsize=7.8, loc="lower left", ncol=2)
    titulo(ax, "El tamaño del paso importa (inicio w = 0)", 9)
    ax.set_ylim(1e-14, 1e3)
    style(ax)
    fig.tight_layout()
    save(fig, "c45_gd.png")


def f_backprop():
    E = NUM["ejemplo"]
    fig, ax = plt.subplots(figsize=(W_FIG, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    xs = [10, 29, 48, 67, 86]
    nom = [("x", f"{E['x']:g}"), ("a₁ = w₁x + b₁", f"{E['a1']:.2f}"), ("h = ReLU(a₁)", f"{E['h']:.2f}"), ("a₂ = w₂h + b₂", f"{E['a2']:.2f}"), ("p = σ(a₂)", f"{E['p']:.4f}")]
    ax.text(0, 57.5, "Hacia adelante (propagación): se calculan los valores", fontsize=8.8, fontweight="bold", color=C45)
    for x, (n, v) in zip(xs, nom):
        caja(ax, x - 8, 41, 16, 11, "", C45)
        ax.text(x, 49.2, n, ha="center", fontsize=7.4, color=MUTED)
        ax.text(x, 44.3, v, ha="center", fontsize=10, fontweight="bold", color=INK)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        flecha(ax, (x0 + 8.3, 46.5), (x1 - 8.3, 46.5), ms=9)
    ax.text(94, 37.2, f"pérdida ℓ = −log p = {E['l']:.4f}   (y = 1)", ha="right", fontsize=8.4, color=RED, fontweight="bold")
    ax.text(0, 31, "Hacia atrás (retropropagación): se calcula cuánto influye cada cosa en ℓ", fontsize=8.8, fontweight="bold", color=AMBER)
    d1, d2 = E["delta1"], E["delta2"]
    bw = [(xs[3], f"δ₂ = p − y\n= {d2:.4f}", "ℓ respecto a a₂"), (xs[2], f"δ₂·w₂\n= {d2 * E['theta'][2]:.4f}", "ℓ respecto a h"), (xs[1], f"δ₁ = δ₂·w₂·ReLU′\n= {d1:.4f}", "ℓ respecto a a₁ (ReLU′ = 1)")]
    for x, t, s in bw:
        caja(ax, x - 9, 17.5, 18, 10.5, t, AMBER, fs=8)
        ax.text(x, 14.6, s, ha="center", fontsize=7.4, color=MUTED)
    for (xa, _, _), (xb, _, _) in zip(bw[:-1], bw[1:]):
        flecha(ax, (xa - 9.3, 22.7), (xb + 9.3, 22.7), color=AMBER, ms=9)
    g = E["grad"]
    ax.text(0, 9.2, "Sensibilidades de los cuatro parámetros (θ = w₁, b₁, w₂, b₂):", fontsize=8.8, fontweight="bold", color=RED)
    ax.text(0, 3.6, f"∂ℓ/∂w₁ = δ₁·x = {g[0]:.4f}   ·   ∂ℓ/∂b₁ = δ₁ = {g[1]:.4f}   ·   ∂ℓ/∂w₂ = δ₂·h = {g[2]:.4f}   ·   ∂ℓ/∂b₂ = δ₂ = {g[3]:.4f}", fontsize=8.1, color=INK)
    save(fig, "c45_backprop.png")


def f_adam():
    A = NUM["adam"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.1), sharey=True)
    for ax, esc, ta, tg, t in ((axes[0], 1, A["tray_adam1"], A["tray_gd1"], "J = (w − 3)²"), (axes[1], 100, A["tray_adam100"], A["tray_gd100"], "J = 100·(w − 3)²  (gradientes 100 veces mayores)")):
        ax.semilogy(range(len(ta)), np.abs(np.array(ta) - 3) + 1e-12, "o-", ms=3.2, color=C45, label="Adam, η = 0.1")
        ax.semilogy(range(len(tg)), np.abs(np.array(tg) - 3) + 1e-12, "s-", ms=3.2, color=RED, label="descenso simple, η = 0.1")
        ax.set_xlabel("actualización")
        titulo(ax, t, 8.6)
        style(ax)
    axes[0].set_ylabel("distancia al óptimo |w − 3|")
    axes[0].legend(frameon=False, fontsize=7.8, loc="lower left")
    axes[0].set_ylim(1e-3, 1e8)
    fig.tight_layout()
    save(fig, "c45_adam.png")


def f_lotes():
    L = NUM["lotes"]
    fig, ax = plt.subplots(figsize=(W_FIG, 2.4))
    ax.set_xlim(0, 1077)
    ax.set_ylim(-0.4, 3)
    ax.axis("off")
    x = 0
    for k in range(L["lotes"]):
        n = 64 if k < L["completos"] else L["resto"]
        ax.add_patch(Rectangle((x, 1.0), n, 1.1, fc=C45 if k < L["completos"] else AMBER, ec="white", lw=1.2))
        ax.text(x + n / 2, 1.55, str(k + 1), ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
        x += n
    ax.text(0, 2.6, f"Una época = una pasada por las {L['n']:,} imágenes de entrenamiento".replace(",", " "), fontsize=8.8, fontweight="bold", color=INK)
    ax.text(0, 0.55, f"{L['completos']} lotes de 64 imágenes  +  1 lote final de {L['resto']}  =  {L['lotes']} actualizaciones por época", fontsize=8.4, color=INK)
    ax.text(0, -0.15, f"{L['lotes']} actualizaciones × {L['epocas']} épocas = {L['actualiz']:,} actualizaciones en total".replace(",", " "), fontsize=8.4, color=C45, fontweight="bold")
    save(fig, "c45_lotes.png")


def f_curvas():
    cv = NUM["curvas"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.3), sharey=True)
    for ax, k in zip(axes, ("Red16", "Red32")):
        e = np.arange(1, EPOCAS + 1)
        ax.semilogy(e, cv[k]["tr"], color=BLUE, lw=1.8, label="entrenamiento")
        ax.semilogy(e, cv[k]["va"], color=AMBER, lw=1.8, label="validación")
        ax.axvline(cv[k]["epoca"], color=GRAY, ls="--", lw=1.2, label=f"época elegida: {cv[k]['epoca']}")
        ax.set_xlabel("época")
        titulo(ax, f"{k}: validación mínima {cv[k]['val']:.4f}", 9)
        ax.legend(frameon=False, fontsize=7.8, loc="center right")
        style(ax)
    axes[0].set_ylabel("log loss sin penalización (escala log)")
    fig.tight_layout()
    save(fig, "c45_curvas.png")


def f_seleccion():
    cv, an = NUM["curvas"], NUM["anchos"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.1), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    nombres = ["Red16", "Red32"]
    v = [cv[k]["val"] for k in nombres]
    ax.barh(nombres, v, color=[GRAY, C45], height=0.5)
    for i, k in enumerate(nombres):
        ax.text(v[i] + 0.002, i, f"{v[i]:.4f} · época {cv[k]['epoca']}", va="center", fontsize=8.4)
    ax.set_xlim(0, 0.16)
    titulo(ax, "Diapositiva 35: log loss mínimo de validación", 8.8)
    style(ax, "x")
    ax = axes[1]
    ks = list(an)
    xs = [int(k) for k in ks]
    ax.plot(range(len(ks)), [an[k]["val"] for k in ks], "o-", color=C45, lw=1.8)
    for i, k in enumerate(ks):
        ax.text(i, an[k]["val"] * (1.25 if an[k]["val"] > 0.2 else 1.1), f"{an[k]['val']:.3f}", ha="center", fontsize=7.8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k}\n({an[k]['params']:,})".replace(",", " ") for k in ks], fontsize=7.8)
    ax.set_xlabel("unidades ocultas (parámetros)")
    ax.set_ylabel("validación mínima")
    ax.set_ylim(0.04, 2)
    titulo(ax, "Nuestra reproducción: más anchos, misma receta", 8.8)
    style(ax)
    fig.tight_layout()
    save(fig, "c45_seleccion.png")


def f_alpha():
    al = NUM["alphas"]
    ks = list(al)
    lab = ["0" if float(k) == 0 else f"{float(k):g}" for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 3.1))
    ax = axes[0]
    x = np.arange(len(ks))
    ax.plot(x, [al[k]["tr_120"] for k in ks], "o-", color=BLUE, label="entrenamiento (época 120)")
    ax.plot(x, [al[k]["val_120"] for k in ks], "s-", color=AMBER, label="validación (época 120)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("log loss sin penalización")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    titulo(ax, "Demasiada penalización: la red no aprende", 8.8)
    style(ax)
    ax = axes[1]
    ax.bar(x, [al[k]["suma_w2"] for k in ks], color=C45, width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, al[k]["suma_w2"] + 6, f"{al[k]['suma_w2']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("suma de los pesos² (W₁, W₂)")
    titulo(ax, "La penalización encoge los pesos", 8.8)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c45_alpha.png")


def f_comparacion():
    ev = NUM["eval"]
    fig, axes = plt.subplots(1, 3, figsize=(W_FIG, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, key, tit, fmt in ((axes[0], "ll", "Log loss (menor es mejor)", "{:.4f}"), (axes[1], "errores", "Errores en 360 imágenes", "{:d}")):
        v = [ev["log"]["prueba"][key], ev["red"]["prueba"][key]]
        ax.bar(["logística", "red"], v, color=[GRAY, C45], width=0.55)
        for i, y_ in enumerate(v):
            ax.text(i, y_, fmt.format(y_), ha="center", va="bottom", fontsize=8.6)
        ax.set_ylim(0, max(v) * 1.22)
        titulo(ax, tit, 8.8)
        style(ax, "y")
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.0, "Prueba histórica", fontsize=9.2, fontweight="bold", color=C45)
    ax.text(0, 7.6, f"exactitud: {ev['log']['prueba']['acc'] * 100:.2f} % (logística)\n{ev['red']['prueba']['acc'] * 100:.2f} % (red)\n\nLa red reduce el log loss;\nla logística comete menos\nerrores.", fontsize=8.2, color=INK, va="top", linespacing=1.35)
    fig.tight_layout()
    save(fig, "c45_comparacion.png")


def f_discrepancia():
    M = NUM["metricas"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    x = np.arange(2)
    ax.bar(x - 0.2, M["A"]["p"], 0.38, color=GRAY, label="modelo A")
    ax.bar(x + 0.2, M["B"]["p"], 0.38, color=C45, label="modelo B")
    for i in range(2):
        ax.text(i - 0.2, M["A"]["p"][i] + 0.015, f"{M['A']['p'][i]:g}", ha="center", fontsize=8.6)
        ax.text(i + 0.2, M["B"]["p"][i] + 0.015, f"{M['B']['p'][i]:g}", ha="center", fontsize=8.6)
    ax.axhline(0.5, color=RED, ls="--", lw=1.3)
    ax.text(1.5, 0.5, "umbral\n0.5", fontsize=7.8, color=RED, ha="left", va="center")
    ax.set_xlim(-0.5, 1.95)
    ax.set_xticks(x)
    ax.set_xticklabels(["caso 1 (y = 1)", "caso 2 (y = 1)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("probabilidad de la clase 1")
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", ncol=2)
    titulo(ax, "Dos modelos, dos casos", 9)
    style(ax, "y")
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.1, "A: 0 errores", fontsize=9, fontweight="bold", color=INK)
    ax.text(0, 7.7, f"(−log 0.51 − log 0.51) ÷ 2\n= {M['A']['perdida']:.4f}", fontsize=8.4, color=INK, va="top", linespacing=1.4)
    ax.text(0, 4.6, "B: 1 error", fontsize=9, fontweight="bold", color=C45)
    ax.text(0, 3.2, f"(−log 0.99 − log 0.49) ÷ 2\n= {M['B']['perdida']:.4f}", fontsize=8.4, color=INK, va="top", linespacing=1.4)
    ax.text(0, 0.4, "B se equivoca más y pierde menos.", fontsize=8.2, color=AMBER)
    fig.tight_layout()
    save(fig, "c45_discrepancia.png")


def f_confusion():
    M = np.array(NUM["confusion"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(M, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8.6, color="white" if M[i, j] > M.max() / 2 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("dígito predicho")
    ax.set_ylabel("dígito real")
    ax.spines[:].set_visible(False)
    titulo(ax, "Matriz de confusión · 360 imágenes de prueba", 9)
    fig.tight_layout()
    save(fig, "c45_confusion.png")


def f_errores_imgs():
    c = calc()
    errs = NUM["errores"]
    Xt = c["X"][c["te"]]
    fig, axes = plt.subplots(2, 5, figsize=(W_FIG, 3.5))
    for ax, e in zip(axes.flat, errs):
        ax.imshow(Xt[e["pos"]].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"real {e['real']} → pred. {e['pred']}\np máx. = {e['pmax']:.2f}", fontsize=8.2, color=RED if e["pmax"] >= 0.9 else INK)
    fig.tight_layout()
    save(fig, "c45_errores_imgs.png")


def f_errores_p():
    errs = NUM["errores"]
    fig, ax = plt.subplots(figsize=(W_FIG, 2.7))
    lab = [f"{e['real']}→{e['pred']}" for e in errs]
    pm = [e["pmax"] for e in errs]
    ax.bar(range(len(errs)), pm, color=[RED if p >= 0.9 else C45 for p in pm], width=0.65)
    ax.axhline(0.9, color=AMBER, ls="--", lw=1.4)
    ax.text(len(errs) - 0.5, 0.915, "corte ilustrativo 0.90", fontsize=8, color=AMBER, ha="right")
    for i, p in enumerate(pm):
        ax.text(i, p + 0.012, f"{p:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(errs)))
    ax.set_xticklabels(lab, fontsize=8.4)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("dígito real → predicho en cada error")
    ax.set_ylabel("probabilidad máxima")
    titulo(ax, "Los 10 errores de la red: ¿con cuánta seguridad se equivocó?", 9)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c45_errores_p.png")


PASOS = dict(ciclo=f_ciclo, neurona=f_neurona, activaciones=f_activaciones, capa=f_capa, red=f_red, xor=f_xor, softmax=f_softmax, perdida=f_perdida, gd=f_gd,
             backprop=f_backprop, adam=f_adam, lotes=f_lotes, curvas=f_curvas, seleccion=f_seleccion, alpha=f_alpha, comparacion=f_comparacion,
             discrepancia=f_discrepancia, confusion=f_confusion, errores_imgs=f_errores_imgs, errores_p=f_errores_p)

if __name__ == "__main__":
    numeros_basicos()
    numeros_entrenamiento()
    for k, fn in PASOS.items():
        fn()
        print("hecho:", k, flush=True)
    out = Path(__file__).resolve().parent / "c45_numeros.json"
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("todas las cifras de las diapositivas reproducidas:", NUM["todo_igual"])
    print("ok")
