"""Ayudas comunes para los generadores de la Clase 32 (sobre dsl.py)."""
import json
from pathlib import Path

from dsl import Bc, Bx, L, P, T

AQUI = Path(__file__).resolve().parent
D = json.loads((AQUI.parent / "datos32.json").read_text(encoding="utf-8"))
FUENTE = Path(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")


def out(nombre):
    return str(FUENTE / nombre)


def tarjetas(items, y=520, w=500, h=380, tam_t=56, tam_d=38, sombra=True):
    """items = [(x, color, titulo, detalle, en)] -> caja + titulo + detalle."""
    acts = []
    for x, c, tit, det, k in items:
        acts += [Bc(x, y, w, h, "blanco", c, 5, 30, sombra=sombra, en=k, ent="subir"),
                 T(tit, x, y - h * 0.22, tam=tam_t, c=c, mw=w - 60, en=k, t=0.1),
                 T(det, x, y + h * 0.14, tam=tam_d, c="tinta", w="sb", mw=w - 70, en=k, t=0.2)]
    return acts


def flujo(pasos, y=500, w=340, h=190, x0=330, paso=450, tam=46, en_flechas=None):
    """pasos = [(texto, color, en)] -> cajas en fila unidas por flechas."""
    acts = []
    for i, (txt, c, k) in enumerate(pasos):
        x = x0 + paso * i
        acts += [Bc(x, y, w, h, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(txt, x, y, tam=tam, c=c, mw=w - 40, en=k, t=0.1)]
        if i:
            acts.append(L(x - paso + w / 2 + 10, y, x - w / 2 - 10, y, "gris", 6, True, en=k, t=0.2))
    return acts


def siguiente(num, titulo, say_extra=""):
    """Tarjeta de 'siguiente parada' (como en C31)."""
    return [Bc(960, 480, 760, 260, "ambarclaro", "ambar", 6, 40, ent="pop", en="a"),
            T(str(num), 680, 480, tam=120, c="ambar", en="a", t=0.1),
            T(titulo, 1060, 480, tam=40, c="tinta", mw=520, en="a", t=0.15)]


def barras(vals, x0, ybase, ancho, escala, colores, etiquetas=None, fmt="{:.1f}", suf=" %", en=None, t0=0.0, dt=0.12,
           tam_v=34, tam_e=30, sep=None, id_pre=None, base_tam=None):
    """Barras verticales que crecen desde ybase; vals en unidades -> alto = val*escala px."""
    from dsl import N
    acts = []
    sep = sep or ancho * 1.35
    for i, v in enumerate(vals):
        x = x0 + i * sep
        c = colores[i] if isinstance(colores, list) else colores
        acts.append(Bx(x - ancho / 2, ybase - v * escala, x + ancho / 2, ybase, c, None, 0, 8, grow="y", en=en, t=t0 + dt * i,
                       id=(f"{id_pre}{i}" if id_pre else None)))
        if fmt:
            acts.append(N(v, x, ybase - v * escala - 28, fmt, tam_v, c, suf=suf, en=en, t=t0 + dt * i + 0.2, dur=0.6,
                          id=(f"{id_pre}v{i}" if id_pre else None)))
        if etiquetas:
            acts.append(T(etiquetas[i], x, ybase + 34, tam=tam_e, c="gris", w="sb", en=en, t=t0 + dt * i,
                          id=(f"{id_pre}e{i}" if id_pre else None)))
    return acts


def caja_formula(texto, y=300, c="violeta", tam=60, en=None, t=None, w=1400):
    return [Bc(960, y, w, tam * 2.1, "blanco", c, 4, 26, sombra=True, en=en, t=t, ent="subir"),
            T(texto, 960, y, tam=tam, c=c, en=en, t=(t or 0) + 0.1)]


def ilustrativo(y=1000, en=None, t=0.6, texto="ejemplo ilustrativo"):
    return T(texto, 1860, y, tam=26, c="gris", w="sb", anc="rm", en=en, t=t)
