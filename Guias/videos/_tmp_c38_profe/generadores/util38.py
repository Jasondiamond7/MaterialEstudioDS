"""Ayudas comunes para los generadores de la Clase 38 (sobre dsl.py)."""
import json
from pathlib import Path

from dsl import Bc, Bx, K, L, N, P, R, T

AQUI = Path(__file__).resolve().parent
D = json.loads((AQUI.parent / "datos38.json").read_text(encoding="utf-8"))
FG = {}
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


def cien(x):
    """dolares -> texto redondeado a centenas como se narra: 23081.0 -> '23 100' (V.b lo pasa a espacio fino)."""
    v = int(round(x / 100.0)) * 100
    return f"{v:,}".replace(",", " ")


def r3(x):
    return f"{x:.3f}"


def barras_v(items, x0=330, paso=230, base=860, escala=10.0, w=160, fmt="{:.1f}", tam=34, en=None, dt=0.12, ids=None):
    """Barras verticales: items = [(etiqueta, valor, color, marca)]; altura = valor * escala px."""
    acts = []
    for i, (lab, val, c, m) in enumerate(items):
        x = x0 + paso * i
        acts += [Bx(x - w / 2, base - val * escala, x + w / 2, base, c, None, 0, 12, grow="y", en=m or en, t=dt * i),
                 T(fmt.format(val), x, base - val * escala - 34, tam=tam, c="tinta", w="b", en=m or en, t=dt * i + 0.4),
                 T(lab, x, base + 40, tam=30, c="gris", w="sb", en=m or en, t=dt * i)]
    return acts


def nodo(x, y, hoja=False, c="ambar", en=None, t=0.0, r=34, id=None):
    return Bc(x, y, 2 * r, 2 * r, (c if hoja else "blanco"), c, 5, r, en=en, t=t, id=id)


def arista(a, b, en=None, t=0.0, r=34):
    return L(a[0], a[1] + r, b[0], b[1] - r, "gris", 5, en=en, t=t, dur=0.4)


# ---------------------------------------------------------------- C37: nubes de puntos y fronteras sobre un grafico
def serie_puntos(xs, ys, color, mk=6, alfa=1.0):
    """Serie de dispersion: ordenada por x, sin linea, con un punto ficticio fuera del lienzo (x=50) al final,
    para que 'revela' recorra todos los puntos. Empieza con revela 0.01 (se anima a 1 con U)."""
    par = sorted(zip(xs, ys))
    xs2 = [p[0] for p in par] + [50.0]
    ys2 = [p[1] for p in par] + [par[-1][1]]
    return dict(xs=xs2, ys=ys2, color=color, linea=None, marcadores=mk, revela=0.01, lapiz=False, alfa=alfa)


def serie_curva(xs, ys, color, g=5, alfa=0.0):
    """Curva cualquiera (contorno no monotono): revela enorme para que se vea entera; aparece animando su alfa."""
    return dict(xs=list(xs), ys=list(ys), color=color, grosor=g, revela=100.0, lapiz=False, alfa=alfa)


def recta(w, b, nivel, x0, x1):
    """Recta w1*x + w2*y + b = nivel entre x0 y x1 (en coordenadas de datos)."""
    return [x0, x1], [(nivel - b - w[0] * x0) / w[1], (nivel - b - w[0] * x1) / w[1]]


def nube(pts, clases, col0="azul", col1="rojo", mk=6):
    """Dos series (benignos, malignos) a partir de listas x, y y clases 0/1."""
    x, y = pts
    a = serie_puntos([xi for xi, c in zip(x, clases) if c == 0], [yi for yi, c in zip(y, clases) if c == 0], col0, mk)
    b = serie_puntos([xi for xi, c in zip(x, clases) if c == 1], [yi for yi, c in zip(y, clases) if c == 1], col1, mk)
    return [a, b]


def adicional(y=1050, en=None, t=0.6, texto="cálculo adicional (no está en el material de la clase)"):
    return T(texto, 1860, y, tam=26, c="violeta", w="sb", anc="rm", en=en, t=t)


def matriz(cx, cy, vals, en=None, t=0.0, c="azul", tam=56, cel=170, etiquetas=True, ids=None, tit=None, colores=None):
    """Matriz de confusion 2x2 (filas = real 0/1, columnas = predicha 0/1). vals = [[VN, FP], [FN, VP]]."""
    acts = []
    cols = colores or [["verde", "rojo"], ["rojo", "verde"]]
    if tit:
        acts.append(T(tit, cx, cy - cel - 80, tam=38, c=c, w="b", en=en, t=t))
    if etiquetas:
        acts += [T("pred. 0", cx - cel / 2, cy - cel - 28, tam=28, c="gris", w="sb", en=en, t=t),
                 T("pred. 1", cx + cel / 2, cy - cel - 28, tam=28, c="gris", w="sb", en=en, t=t),
                 T("real 0", cx - cel - 18, cy - cel / 2, tam=28, c="gris", w="sb", anc="rm", en=en, t=t),
                 T("real 1", cx - cel - 18, cy + cel / 2, tam=28, c="gris", w="sb", anc="rm", en=en, t=t)]
    k = 0
    for i in range(2):
        for j in range(2):
            x, y = cx + (j - 0.5) * cel, cy + (i - 0.5) * cel
            cc = cols[i][j]
            acts += [Bc(x, y, cel - 10, cel - 10, CLARO_C.get(cc, "blanco"), cc, 4, 18, en=en, t=t + 0.08 * k,
                        id=(f"{ids}b{i}{j}" if ids else None)),
                     T(str(vals[i][j]), x, y, tam=tam, c=cc, w="b", en=en, t=t + 0.08 * k + 0.05, id=(f"{ids}t{i}{j}" if ids else None))]
            k += 1
    return acts


CLARO_C = {"verde": "verdeclaro", "rojo": "rojoclaro", "azul": "azulclaro", "ambar": "ambarclaro", "gris": "linea",
           "violeta": "violetaclaro", "teal": "tealclaro"}


def rh(num, den, d=3):
    """Fraccion num/den redondeada hacia arriba en el medio (como en el material): 52/64 = 0.8125 -> '0.813'."""
    from decimal import Decimal, ROUND_HALF_UP
    q = (Decimal(num) / Decimal(den)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:.{d}f}"


def recta_caja(w, b, nivel, xmin, xmax, ymin, ymax):
    """Segmento de la recta w1*x + w2*y + b = nivel recortado a la caja [xmin, xmax] x [ymin, ymax] (x creciente)."""
    pts = []
    for x in (xmin, xmax):
        y = (nivel - b - w[0] * x) / w[1]
        if ymin - 1e-9 <= y <= ymax + 1e-9:
            pts.append((x, y))
    for y in (ymin, ymax):
        if abs(w[0]) > 1e-12:
            x = (nivel - b - w[1] * y) / w[0]
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                pts.append((x, y))
    pts = sorted(set((round(x, 4), round(y, 4)) for x, y in pts))
    a, c = pts[0], pts[-1]
    return [a[0], c[0]], [a[1], c[1]]


# ---------------------------------------------------------------- C38
def diapo(n, y=1050, en=None, t=0.6):
    """Etiqueta de un ejemplo tomado de la diapositiva n."""
    return T(f"ejemplo de la diapositiva {n}", 1860, y, tam=26, c="gris", w="sb", anc="rm", en=en, t=t)


def etiqueta(texto, y=1050, en=None, t=0.6, c="gris"):
    return T(texto, 1860, y, tam=26, c=c, w="sb", anc="rm", en=en, t=t)


def cita(texto, y=330, en=None, t=None, tam=44, id=None):
    """Cita textual de la grabacion de la clase, en cursiva visual (comillas latinas)."""
    return T(f"«{texto}»", 960, y, tam=tam, c="tinta", w="sb", mw=1480, en=en, t=t, id=id)


def m2(cx, cy, vals, en=None, t=0.0, cel=170, tam=52, filas=("no spam", "spam"), cols=("no spam", "spam"), tit=None, c="azul"):
    """Matriz 2x2 con etiquetas propias: filas reales, columnas predichas. vals = [[VN, FP], [FN, VP]]."""
    acts = []
    if tit:
        acts.append(T(tit, cx, cy - cel - 90, tam=36, c=c, w="b", en=en, t=t))
    acts += [T("predicha →", cx, cy - cel - 50, tam=24, c="gris", w="sb", en=en, t=t),
             T(cols[0], cx - cel / 2, cy - cel - 16, tam=26, c="gris", w="sb", en=en, t=t),
             T(cols[1], cx + cel / 2, cy - cel - 16, tam=26, c="gris", w="sb", en=en, t=t),
             T(f"real {filas[0]}", cx - cel - 14, cy - cel / 2, tam=26, c="gris", w="sb", anc="rm", en=en, t=t),
             T(f"real {filas[1]}", cx - cel - 14, cy + cel / 2, tam=26, c="gris", w="sb", anc="rm", en=en, t=t)]
    cl = [["verde", "rojo"], ["rojo", "verde"]]
    k = 0
    for i in range(2):
        for j in range(2):
            x, y = cx + (j - 0.5) * cel, cy + (i - 0.5) * cel
            cc = cl[i][j]
            acts += [Bc(x, y, cel - 10, cel - 10, CLARO_C[cc], cc, 4, 18, en=en, t=t + 0.08 * k),
                     T(str(vals[i][j]), x, y, tam=tam, c=cc, w="b", en=en, t=t + 0.08 * k + 0.05)]
            k += 1
    return acts
