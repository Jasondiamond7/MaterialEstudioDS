"""Ayudas comunes para los generadores de la Clase 34 (sobre dsl.py)."""
import json
from pathlib import Path

from dsl import Bc, Bx, K, L, N, P, R, T

AQUI = Path(__file__).resolve().parent
D = json.loads((AQUI.parent / "datos34.json").read_text(encoding="utf-8"))
FUENTE = Path(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")


def out(nombre):
    return str(FUENTE / nombre)


def mil(n):
    """3150 -> '3.150' (miles con punto, como en el material de la clase)."""
    return f"{int(n):,}".replace(",", ".")


def pct(p, d=1):
    return f"{100 * p:.{d}f} %"


def tarjetas(items, y=520, w=500, h=380, tam_t=56, tam_d=38, sombra=True):
    """items = [(x, color, titulo, detalle, en)] -> caja + titulo + detalle."""
    acts = []
    for x, c, tit, det, k in items:
        acts += [Bc(x, y, w, h, "blanco", c, 5, 30, sombra=sombra, en=k, ent="subir"),
                 T(tit, x, y - h * 0.22, tam=tam_t, c=c, mw=w - 60, en=k, t=0.1),
                 T(det, x, y + h * 0.14, tam=tam_d, c="tinta", w="sb", mw=w - 70, en=k, t=0.2)]
    return acts


def flujo(pasos, y=500, w=340, h=190, x0=330, paso=450, tam=46):
    """pasos = [(texto, color, en)] -> cajas en fila unidas por flechas."""
    acts = []
    for i, (txt, c, k) in enumerate(pasos):
        x = x0 + paso * i
        acts += [Bc(x, y, w, h, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(txt, x, y, tam=tam, c=c, mw=w - 40, en=k, t=0.1)]
        if i:
            acts.append(L(x - paso + w / 2 + 10, y, x - w / 2 - 10, y, "gris", 6, True, en=k, t=0.2))
    return acts


def siguiente(num, titulo):
    """Tarjeta de 'siguiente parada'."""
    return [Bc(960, 480, 760, 260, "ambarclaro", "ambar", 6, 40, ent="pop", en="a"),
            T(str(num), 680, 480, tam=120, c="ambar", en="a", t=0.1),
            T(titulo, 1060, 480, tam=40, c="tinta", mw=520, en="a", t=0.15)]


def ilustrativo(y=1000, en=None, t=0.6, texto="ejemplo ilustrativo"):
    return T(texto, 1860, y, tam=26, c="gris", w="sb", anc="rm", en=en, t=t)


def real(y=1050, en=None, t=0.6, texto="datos reales de la clase (recalculados)"):
    return T(texto, 1860, y, tam=26, c="teal", w="sb", anc="rm", en=en, t=t)


def codigo(lineas, y=420, w=1640, tam=36, c="violeta", en=None, t=0.0, dt=0.25, alto=None, x=960, ids=None):
    """Caja de codigo monoespaciado; las lineas aparecen una tras otra (alineadas a la izquierda)."""
    paso = tam * 1.55
    h = alto or paso * len(lineas) + 70
    acts = [Bc(x, y, w, h, "blanco", c, 4, 24, sombra=True, en=en, t=t, ent="subir")]
    y0 = y - paso * (len(lineas) - 1) / 2
    for i, ln in enumerate(lineas):
        acts.append(T(ln, x - w / 2 + 50, y0 + i * paso, tam=tam, c="tinta", w="m", anc="lm", en=en, t=t + 0.1 + dt * i,
                      id=(ids[i] if ids else None)))
    return acts


def caja_formula(texto, y=300, c="violeta", tam=60, en=None, t=None, w=1400, wf="b"):
    return [Bc(960, y, w, tam * 2.1, "blanco", c, 4, 26, sombra=True, en=en, t=t, ent="subir"),
            T(texto, 960, y, tam=tam, c=c, w=wf, en=en, t=(t or 0) + 0.1)]


def alerta(y=200, en=None, t=None, texto="alerta del material"):
    return P(texto, 960, y, "ambar", 48, ent="subir", en=en, t=t)


def fila(y, cols, tams=None, colores=None, pesos=None, en=None, t=0.0, fondo=None, x0=260, x1=1660, h=86, ancl=None):
    """Fila de tabla: cols = [(texto, x)] ; fondo opcional (caja)."""
    acts = []
    if fondo:
        acts.append(Bc((x0 + x1) / 2, y, x1 - x0, h, fondo, None, 0, 16, en=en, t=t, ent="subir"))
    for j, (tx, x) in enumerate(cols):
        acts.append(T(tx, x, y, tam=(tams[j] if tams else 36), c=(colores[j] if colores else "tinta"),
                      w=(pesos[j] if pesos else "sb"), anc=(ancl[j] if ancl else "mm"), en=en, t=t + 0.08 * j))
    return acts


# ------------------------------------------------------------------ el arbol real de profundidad 3
NODOS = {nd["id"]: nd for nd in D["arbol3_nodos"]}
POS = {0: (960, 270), 1: (510, 450), 8: (1410, 450), 2: (285, 640), 5: (735, 640), 9: (1185, 640), 12: (1635, 640),
       3: (172, 840), 4: (397, 840), 6: (622, 840), 7: (847, 840), 10: (1072, 840), 11: (1297, 840)}
PROF = {0: 0, 1: 1, 8: 1, 2: 2, 5: 2, 9: 2, 12: 2, 3: 3, 4: 3, 6: 3, 7: 3, 10: 3, 11: 3}
NOMBRE_CORTO = {"frec_llamadas": "frec_llamadas", "numero_fallas": "numero_fallas"}


def color_tasa(t):
    if t >= 0.6:
        return "rojo"
    if t >= 0.3:
        return "ambar"
    if t < 0.12:
        return "verde"
    return "gris"


def etiqueta_nodo(i):
    nd = NODOS[i]
    if nd["var"]:
        u = nd["umbral"]
        us = f"{u:g}" if u < 1000 else mil(u)
        return f"{nd['var']} ≤ {us}"
    return None


def arbol(en=None, pre="ar", t0=0.0, dt=0.12, hasta=3, escala=1.0):
    """Dibuja el arbol real (profundidad 3, min_samples_leaf=60). en = {profundidad: marca}.
    Ids: {pre}b{i} caja, {pre}v{i} regla, {pre}n{i} n/tasa, {pre}l{i} linea al padre."""
    en = en or {}
    acts = []
    padre = {}
    for i, nd in NODOS.items():
        if nd["izq"] != -1:
            padre[nd["izq"]], padre[nd["der"]] = (i, "sí"), (i, "no")
    orden = sorted(NODOS, key=lambda i: (PROF[i], POS[i][0]))
    for k, i in enumerate(orden):
        d = PROF[i]
        if d > hasta:
            continue
        nd = NODOS[i]
        x, y = POS[i]
        hoja = nd["var"] is None
        w, h = (205, 118) if hoja else (370, 118)
        tasa = nd["fugas"] / nd["n"]
        c = color_tasa(tasa)
        mk = en.get(d)
        tt = t0 + dt * (k % 4)
        if i in padre:
            px, py = POS[padre[i][0]]
            acts.append(L(px, py + 59, x, y - 59, "grisclaro", 5, en=mk, t=tt, id=f"{pre}l{i}"))
            acts.append(T(padre[i][1], (px + x) / 2 + (-26 if padre[i][1] == "sí" else 26), (py + y) / 2, tam=24, c="gris", w="sb",
                          en=mk, t=tt + 0.2, id=f"{pre}s{i}"))
        acts.append(Bc(x, y, w, h, "blanco", c, 5, 20, sombra=True, en=mk, t=tt + 0.1, ent="pop", id=f"{pre}b{i}"))
        if hoja:
            acts.append(T(f"n {mil(nd['n'])}", x, y - 26, tam=28, c="gris", w="sb", en=mk, t=tt + 0.2, id=f"{pre}v{i}"))
            acts.append(T(pct(tasa), x, y + 22, tam=42, c=c, w="b", en=mk, t=tt + 0.25, id=f"{pre}n{i}"))
        else:
            acts.append(T(etiqueta_nodo(i), x, y - 24, tam=32, c="tinta", w="b", en=mk, t=tt + 0.2, id=f"{pre}v{i}"))
            acts.append(T(f"n {mil(nd['n'])} · fuga {pct(tasa)}", x, y + 26, tam=27, c=c, w="sb", en=mk, t=tt + 0.25,
                          id=f"{pre}n{i}"))
    return acts


def ids_arbol(pre="ar"):
    out_ = []
    for i in NODOS:
        out_ += [f"{pre}b{i}", f"{pre}v{i}", f"{pre}n{i}"]
        if i:
            out_ += [f"{pre}l{i}", f"{pre}s{i}"]
    return out_
