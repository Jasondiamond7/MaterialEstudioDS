"""DSL compacto (copia C36) para escribir guiones del motor video_profe (genera el JSON)."""
import itertools
import json
from pathlib import Path

_n = itertools.count(1)
CLARO = {"violeta": "violetaclaro", "azul": "azulclaro", "rojo": "rojoclaro", "verde": "verdeclaro",
         "ambar": "ambarclaro", "teal": "tealclaro", "gris": "linea"}


def _fin(d, id, en, t, dur, ent, desde=None, z=None, ease=None):
    d["id"] = id or f"e{next(_n)}"
    if ent:
        d["entrada"] = ent
    if en:
        d["en"] = en
    if t:
        d["t"] = t
    if dur:
        d["dur"] = dur
    if desde:
        d["desde"] = desde
    if z is not None:
        d["z"] = z
    if ease:
        d["ease"] = ease
    return d


def T(text, x, y, tam=48, c="tinta", w="b", id=None, en=None, t=None, ent="pop", anc="mm", fondo=None, mw=0,
      dur=None, pad=None, inter=None, **kw):
    d = dict(tipo="texto", texto=text, x=x, y=y, tam=tam, peso=w, color=c, ancla=anc)
    if fondo:
        d["fondo_txt"] = CLARO.get(fondo, fondo)
        if pad:
            d["pad"] = pad
    if mw:
        d["ancho_max"] = mw
    if inter:
        d["interlinea"] = inter
    d.update(kw)
    return _fin(d, id, en, t, dur, ent)


def P(text, x, y, c="violeta", tam=44, **kw):
    """Pildora: texto de color sobre fondo claro del mismo color."""
    return T(text, x, y, tam=tam, c=c, fondo=c, **kw)


def N(v, x, y, fmt="{:.0f}", tam=60, c="tinta", suf="", pre="", id=None, en=None, t=None, dur=1.0, anc="mm", **kw):
    d = dict(tipo="numero", valor=v, fmt=fmt, sufijo=suf, prefijo=pre, x=x, y=y, tam=tam, color=c, ancla=anc)
    d.update(kw)
    return _fin(d, id, en, t, dur, "fade", desde={"valor": 0}, ease="lineal")


def Bx(x0, y0, x1, y1, f="blanco", b=None, g=3, r=18, id=None, en=None, t=None, ent="subir", grow=None, dur=None,
       guion=False, **kw):
    d = dict(tipo="caja", x0=x0, y0=y0, x1=x1, y1=y1, radio=r, relleno=f, borde=b, grosor=g)
    if guion:
        d["guion"] = True
    d.update(kw)
    desde = None
    if grow == "x":
        desde, ent = {"x1": x0}, None
    elif grow == "y":
        desde, ent = {"y0": y1}, None
    elif grow == "-y":
        desde, ent = {"y1": y0}, None
    return _fin(d, id, en, t, dur or (0.7 if grow else None), ent, desde)


def Bc(x, y, w, h, f="blanco", b=None, g=3, r=18, id=None, en=None, t=None, ent="pop", dur=None, sombra=False, **kw):
    d = dict(tipo="caja", x=x, y=y, w=w, h=h, radio=r, relleno=f, borde=b, grosor=g)
    if sombra:
        d["sombra"] = True
    d.update(kw)
    return _fin(d, id, en, t, dur, ent)


def R(x, y, w, h, c="ambar", r=0, g=6, id=None, en=None, t=None, ent="fade", pulso=True, dur=None, **kw):
    d = dict(tipo="anillo", x=x, y=y, w=w, h=h, radio=r, color=c, grosor=g, pulso=pulso)
    d.update(kw)
    return _fin(d, id, en, t, dur, ent)


def L(x0, y0, x1, y1, c="gris", g=6, punta=False, guion=False, id=None, en=None, t=None, dur=0.7, **kw):
    d = dict(tipo="linea", x0=x0, y0=y0, x1=x1, y1=y1, color=c, grosor=g, punta=punta, guion=guion)
    d.update(kw)
    return _fin(d, id, en, t, dur, "ninguna", desde={"progreso": 0})


def K(x, y, forma="check", tam=60, c=None, id=None, en=None, t=None, dur=0.5, **kw):
    d = dict(tipo="marca", x=x, y=y, tam=tam, forma=forma, color=c or ("verde" if forma == "check" else "rojo"))
    d.update(kw)
    return _fin(d, id, en, t, dur, "ninguna", desde={"progreso": 0})


def Pt(grupos, radio=10, visibles=None, id=None, en=None, t=None, dur=None, **kw):
    """grupos=[(n, rect, color)]. Con visibles se hace aparecer de a poco (ola)."""
    d = dict(tipo="puntos", grupos=[dict(n=n, rect=r_, color=c) for n, r_, c in grupos], radio=radio)
    d.update(kw)
    if visibles is not None:
        d["visibles"] = visibles
        return _fin(d, id, en, t, dur or 1.5, "ninguna", desde={"visibles": 0}, ease="lineal")
    return _fin(d, id, en, t, dur, "fade")


def Cand(x, y, tam=110, c="rojo", abierto=0, disco=True, id=None, en=None, t=None, **kw):
    d = dict(tipo="candado", x=x, y=y, tam=tam, color=c, abierto=abierto, disco=disco)
    d.update(kw)
    return _fin(d, id, en, t, None, "pop")


def G(x0, y0, x1, y1, xmin, xmax, ymin, ymax, series, id=None, en=None, t=None, dur=0.7, **kw):
    d = dict(tipo="grafico", x0=x0, y0=y0, x1=x1, y1=y1, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, series=series)
    d.update(kw)
    return _fin(d, id, en, t, dur, "fade")


def U(id, en=None, t=None, dur=None, ease=None, **props):
    d = dict(props)
    d["id"] = id
    if en:
        d["en"] = en
    if t:
        d["t"] = t
    if dur:
        d["dur"] = dur
    if ease:
        d["ease"] = ease
    return d


def X(*ids, en=None, t=None, dur=None):
    d = {"quitar": list(ids)}
    if en:
        d["en"] = en
    if t:
        d["t"] = t
    if dur:
        d["dur"] = dur
    return d


def CL(mant=None, en=None, dur=0.5, t=None):
    d = {"limpiar": True, "dur": dur}
    if mant:
        d["mantener"] = mant
    if en:
        d["en"] = en
    if t:
        d["t"] = t
    return d


def flat(xs):
    out = []
    for x in xs:
        if isinstance(x, list):
            out += flat(x)
        elif x:
            out.append(x)
    return out


class V:
    def __init__(self, titulo, cabecera=None):
        self.titulo = titulo
        self.cab = cabecera or f"Clase 36  ·  {titulo}"
        self.beats = []

    def b(self, say, *acts, sec=None, pausa=None):
        import re as _re
        say = _re.sub(r"(?<=\d) (?=\d{3}(?!\d))", " ", say)     # "26 300" -> miles con espacio fino (la voz lo une)
        d = {}
        if sec:
            d["seccion"] = sec
        d["say"] = say
        if pausa:
            d["pausa"] = pausa
        d["acciones"] = flat(acts)
        self.beats.append(d)
        return d

    def portada(self, titulo, sub, say, sec="Introducción"):
        """Tarjeta de titulo igual a la del video de muestra."""
        return self.b(say, [
            dict(id="t_fondo", tipo="caja", x=960, y=540, w=1920, h=1080, radio=0, relleno="navy", entrada="ninguna"),
            dict(id="t_franja", tipo="caja", x=12, y=540, w=24, h=1080, radio=0, relleno="#EA580C", entrada="ninguna"),
            dict(id="t_clase", tipo="texto", texto="CLASE 36  ·  BOOSTING", x=140, y=330, ancla="lm",
                 tam=40, peso="b", color="#96D6DC", entrada="subir"),
            dict(id="t_tit", tipo="texto", texto=titulo, x=140, y=450, ancla="lm", tam=100 if len(titulo) < 34 else 84,
                 peso="b", color="blanco", entrada="subir", t=0.1),
            dict(id="t_lin", tipo="caja", x0=140, y0=540, x1=420, y1=548, radio=0, relleno="teal", desde={"x1": 140}, t=0.2),
            dict(id="t_sub", tipo="texto", texto=sub, x=140, y=610, ancla="lm", tam=46, color="#E2EAF4", entrada="subir",
                 t=0.25),
            dict(id="t_pie", tipo="texto", texto="Diplomado Data Science UC 2026  ·  Prof. Bastián Galasso", x=140, y=960,
                 ancla="lm", tam=32, color="#96AAC8", t=0.3),
        ], sec=sec)

    def head(self, texto, c="tinta", y=140, id="hd"):
        return T(texto, 960, y, tam=54, c=c, id=id, ent="subir")

    def guardar(self, ruta):
        for b in self.beats:                       # series de puntos (sin linea): ordenar por x para que 'revela' no los recorte
            for a in b["acciones"]:
                for se in a.get("series", []) if isinstance(a.get("series"), list) else []:
                    if se.get("linea") is False and se["xs"] != sorted(se["xs"]):
                        par = sorted(zip(se["xs"], se["ys"]))
                        se["xs"], se["ys"] = [p[0] for p in par], [p[1] for p in par]
        g = {"clase": "Clase 36", "titulo": self.titulo, "cabecera": self.cab, "profesora": "Prof. Bastián Galasso",
             "voz": "es-MX-DaliaNeural", "rate": "+0%", "fps": 25, "beats": self.beats}
        Path(ruta).write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(self.beats)


GR = {}


def g(name, *acts):
    """Agrupa elementos para quitarlos juntos despues con XG(name)."""
    fl = flat(acts)
    GR.setdefault(name, []).extend(a["id"] for a in fl if "tipo" in a)
    return fl


def XG(name, en=None, dur=None, t=None):
    return X(*GR.pop(name, []), en=en, dur=dur, t=t)


def Tj(x, y, w, h, titulo, items, c="violeta", id=None, en=None, t=None, ent="subir", dur=None, **kw):
    d = dict(tipo="tarjeta", x=x, y=y, w=w, h=h, titulo=titulo, items=items, color=c)
    d.update(kw)
    return _fin(d, id, en, t, dur, ent)
