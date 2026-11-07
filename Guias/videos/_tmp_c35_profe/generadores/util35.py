"""Ayudas comunes para los generadores de la Clase 35 (sobre dsl.py)."""
import json
from pathlib import Path

from dsl import Bc, Bx, K, L, N, P, R, T

AQUI = Path(__file__).resolve().parent
D = json.loads((AQUI.parent / "datos35.json").read_text(encoding="utf-8"))
FG = D["fig"]
FUENTE = Path(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")
NB = "\u202f"          # espacio fino: separador de miles en la narracion (la voz lo junta: "2 344" -> dos mil...)


def out(nombre):
    return str(FUENTE / nombre)


def mil(n):
    """3150 -> '3.150' (miles con punto, en pantalla, como en el material de la clase)."""
    return f"{int(round(n)):,}".replace(",", ".")


def mils(n):
    """Para la narracion (subtitulo + voz): 2344 -> '2 344' con espacio fino."""
    return f"{int(round(n)):,}".replace(",", NB)


def k1(v):
    """RMSE en dolares -> '23.1 mil' (texto en pantalla)."""
    return f"{v / 1000:.1f} mil"


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


def casillas(valores, y, x0=None, paso=150, w=120, h=110, c="azul", f="blanco", en=None, t=0.0, dt=0.12, tam=54, pre="cs",
             colores=None, rellenos=None):
    """Fila de casillas numeradas (filas de datos, muestras bootstrap). Ids {pre}b{i} y {pre}t{i}."""
    n = len(valores)
    x0 = x0 if x0 is not None else 960 - paso * (n - 1) / 2
    acts = []
    for i, v in enumerate(valores):
        cc = colores[i] if colores else c
        ff = rellenos[i] if rellenos else f
        acts += [Bc(x0 + paso * i, y, w, h, ff, cc, 4, 18, en=en, t=t + dt * i, ent="pop", id=f"{pre}b{i}"),
                 T(str(v), x0 + paso * i, y, tam=tam, c=cc, en=en, t=t + dt * i + 0.05, id=f"{pre}t{i}")]
    return acts


def barras_h(items, x0=640, y0=300, paso=110, escala=5000, vmin=0.0, fmt="{:.3f}", en=None, tam=36, h=60, xl=620, ids=None):
    """Barras horizontales: items = [(etiqueta, valor, color, marca)]."""
    acts = []
    for i, (lab, val, c, m) in enumerate(items):
        y = y0 + paso * i
        acts += [T(lab, xl, y, tam=tam, c="tinta", w="sb", anc="rm", en=m or en),
                 Bx(x0, y - h / 2, x0 + (val - vmin) * escala, y + h / 2, c, None, 0, 10, grow="x", en=m or en, t=0.1),
                 N(val, x0 + (val - vmin) * escala + 20, y, fmt, 44, c, anc="lm", en=m or en, t=0.4)]
    return acts
