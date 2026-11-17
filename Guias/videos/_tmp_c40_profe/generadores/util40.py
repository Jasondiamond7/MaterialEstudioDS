"""Ayudas comunes para los generadores de la Clase 40 (sobre dsl.py)."""
import json
from pathlib import Path

from dsl import Bc, Bx, K, L, N, P, R, T, G, U

AQUI = Path(__file__).resolve().parent
D = json.loads((AQUI.parent / "datos40.json").read_text(encoding="utf-8"))
FUENTE = Path(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")
NB = "\u202f"
GCOL = ["azul", "ambar", "teal", "violeta", "rojo"]      # colores de los grupos 1, 2, 3, ...


def out(nombre):
    return str(FUENTE / nombre)


def f2(v):
    return f"{v:.2f}"


def f3(v):
    return f"{v:.3f}"


# ---------------------------------------------------------------- etiquetas de cada visual
def etiqueta(texto, y=1050, en=None, t=0.6, c="gris"):
    return T(texto, 1860, y, tam=26, c=c, w="sb", anc="rm", en=en, t=t)


def diapo(n, y=1050, en=None, t=0.6):
    return etiqueta(f"Ejemplo de la diapositiva {n}", y, en, t)


def ilustrativo(y=1050, en=None, t=0.6, texto="Ejemplo ilustrativo"):
    return etiqueta(texto, y, en, t)


def analogia(y=1050, en=None, t=0.6):
    return etiqueta("Analogía", y, en, t, c="violeta")


def real(y=1050, en=None, t=0.6, texto="Datos reales de la clase (recalculados con el script)"):
    return etiqueta(texto, y, en, t, c="teal")


def adicional(y=1050, en=None, t=0.6, texto="Cálculo adicional (no está en el material de la clase)"):
    return etiqueta(texto, y, en, t, c="violeta")


def simulado(y=1050, en=None, t=0.6, texto="Ejemplo simulado"):
    return etiqueta(texto, y, en, t)


def grabacion(hms, y=1050, en=None, t=0.6):
    return etiqueta(f"Grabación de la clase, {hms}", y, en, t, c="ambar")


def cita(texto, y=330, en=None, t=None, tam=42, id=None, mw=1500):
    return T(f"«{texto}»", 960, y, tam=tam, c="tinta", w="sb", mw=mw, en=en, t=t, id=id)


def alerta(y=170, en=None, t=None, texto="alerta del material"):
    return P(texto, 960, y, "ambar", 46, ent="subir", en=en, t=t)


# ---------------------------------------------------------------- bloques
def tarjetas(items, y=520, w=500, h=380, tam_t=56, tam_d=38, sombra=True):
    """items = [(x, color, titulo, detalle, en)] -> caja + titulo + detalle."""
    acts = []
    for x, c, tit, det, k in items:
        acts += [Bc(x, y, w, h, "blanco", c, 5, 30, sombra=sombra, en=k, ent="subir"),
                 T(tit, x, y - h * 0.22, tam=tam_t, c=c, mw=w - 60, en=k, t=0.1),
                 T(det, x, y + h * 0.14, tam=tam_d, c="tinta", w="sb", mw=w - 70, en=k, t=0.2)]
    return acts


def flujo(pasos, y=500, w=340, h=190, x0=330, paso=450, tam=46):
    acts = []
    for i, (txt, c, k) in enumerate(pasos):
        x = x0 + paso * i
        acts += [Bc(x, y, w, h, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(txt, x, y, tam=tam, c=c, mw=w - 40, en=k, t=0.1)]
        if i:
            acts.append(L(x - paso + w / 2 + 10, y, x - w / 2 - 10, y, "gris", 6, True, en=k, t=0.2))
    return acts


def siguiente(num, titulo):
    return [Bc(960, 480, 760, 260, "ambarclaro", "ambar", 6, 40, ent="pop", en="a"),
            T(str(num), 720, 480, tam=110, c="ambar", en="a", t=0.1),
            T(titulo, 1090, 480, tam=40, c="tinta", mw=460, en="a", t=0.15)]


def codigo(lineas, y=420, w=1640, tam=36, c="violeta", en=None, t=0.0, dt=0.25, alto=None, x=960, ids=None):
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


def fila(y, cols, tams=None, colores=None, pesos=None, en=None, t=0.0, fondo=None, x0=260, x1=1660, h=86, ancl=None):
    acts = []
    if fondo:
        acts.append(Bc((x0 + x1) / 2, y, x1 - x0, h, fondo, None, 0, 16, en=en, t=t, ent="subir"))
    for j, (tx, x) in enumerate(cols):
        acts.append(T(tx, x, y, tam=(tams[j] if tams else 36), c=(colores[j] if colores else "tinta"),
                      w=(pesos[j] if pesos else "sb"), anc=(ancl[j] if ancl else "mm"), en=en, t=t + 0.08 * j))
    return acts


def casillas(valores, y, x0=None, paso=150, w=120, h=110, c="azul", f="blanco", en=None, t=0.0, dt=0.12, tam=54, pre="cs",
             colores=None, rellenos=None):
    n = len(valores)
    x0 = x0 if x0 is not None else 960 - paso * (n - 1) / 2
    acts = []
    for i, v in enumerate(valores):
        cc = colores[i] if colores else c
        ff = rellenos[i] if rellenos else f
        acts += [Bc(x0 + paso * i, y, w, h, ff, cc, 4, 18, en=en, t=t + dt * i, ent="pop", id=f"{pre}b{i}"),
                 T(str(v), x0 + paso * i, y, tam=tam, c=cc, en=en, t=t + dt * i + 0.05, id=f"{pre}t{i}")]
    return acts


def barras_v(items, x0=330, paso=230, base=860, escala=10.0, w=160, fmt="{:.1f}", tam=34, en=None, dt=0.12):
    acts = []
    for i, (lab, val, c, m) in enumerate(items):
        x = x0 + paso * i
        acts += [Bx(x - w / 2, base - val * escala, x + w / 2, base, c, None, 0, 12, grow="y", en=m or en, t=dt * i),
                 T(fmt.format(val), x, base - val * escala - 34, tam=tam, c="tinta", w="b", en=m or en, t=dt * i + 0.4),
                 T(lab, x, base + 40, tam=30, c="gris", w="sb", en=m or en, t=dt * i)]
    return acts


# ---------------------------------------------------------------- nubes de puntos
def serie_puntos(xs, ys, color, mk=7, alfa=1.0, revela=0.0):
    """Serie de dispersion ordenada por x, sin linea, con un punto final repetido apenas a la derecha del ultimo (para que
    'revela' recorra todos los puntos de izquierda a derecha). Empieza oculta (revela 0) y se anima a 1 con U."""
    par = sorted(zip(xs, ys))
    ancho = (par[-1][0] - par[0][0]) or 1.0
    xs2 = [p[0] for p in par] + [par[-1][0] + 1e-3 * ancho]
    ys2 = [p[1] for p in par] + [par[-1][1]]
    return dict(xs=xs2, ys=ys2, color=color, linea=None, marcadores=mk, revela=revela, lapiz=False, alfa=alfa)


def grupos_series(xs, ys, labs, colores=GCOL, mk=7, revela=0.0, orden=None):
    """Una serie por grupo (labs 0..k-1)."""
    ks = orden or sorted(set(labs))
    return [serie_puntos([x for x, l in zip(xs, labs) if l == k], [y for y, l in zip(ys, labs) if l == k], colores[i % len(colores)],
                         mk, revela=revela) for i, k in enumerate(ks)]


def revelar(gid, n, en=None, t=None, dur=1.2, desde=0):
    """U que anima revela -> 1 en las series desde..n-1 del grafico gid."""
    return U(gid, en=en, t=t, dur=dur, **{f"series.{i}.revela": 1.0 for i in range(desde, n)})


def cruz(x, y, c="tinta", tam=46, ref=None, en=None, t=None, id=None):
    """Centro de un grupo: una cruz trazada (en coordenadas de datos si ref)."""
    d = K(x, y, "cruz", tam, c, en=en, t=t, id=id)
    if ref:
        d["ref"] = ref
    return d


# ---------------------------------------------------------------- C40: tablas y alertas
def tabla(filas, xs, y0, paso=84, en=None, tam=38, hdr="violetaclaro", x0=None, x1=None, colores=None, pesos=None, ens=None, dt=0.15):
    """filas[0] = encabezado (sobre fondo claro), resto = filas. xs = centro de cada columna. ens = en por fila."""
    acts = []
    x0 = x0 if x0 is not None else xs[0] - 230
    x1 = x1 if x1 is not None else xs[-1] + 150
    for i, fl in enumerate(filas):
        k = ens[i] if ens else en
        acts += fila(y0 + paso * i, list(zip(fl, xs)), tams=[tam] * len(fl),
                     colores=(colores[i] if colores else ["tinta"] * len(fl)),
                     pesos=(["b"] * len(fl) if i == 0 else (pesos[i] if pesos else ["sb"] * len(fl))),
                     en=k, t=(0.0 if ens else dt * i), fondo=(hdr if i == 0 else None), x0=x0, x1=x1, h=paso - 10)
    return acts


def error_material(hms, dicho, correcto, y_dicho=360, y_ok=600, en_ok=None, tam=40):
    """Alerta suave: lo que se dijo en la grabacion (con su marca de tiempo) y el valor correcto."""
    return [alerta(170), T(f"En la grabación ({hms}):", 960, y_dicho - 80, tam=34, c="gris", w="sb"),
            cita(dicho, y=y_dicho, tam=tam),
            P(correcto, 960, y_ok, "verde", 42, en=en_ok), grabacion(hms)]
