import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_02_tipos.json"
v = V("Tipos de variables")

v.portada("Tipos de variables", "Antes de modelar: qué clase de dato hay en cada columna",
          "Tipos de variables. Antes de entrenar cualquier modelo, hay que saber qué clase de dato hay en cada columna.")

# ---------------- 1. El problema
v.b("Imagina que le entregas a un modelo la palabra [[a]]Sur. ¿Qué puede hacer con ella? [[b]]No puede sumarla, ni promediarla, ni multiplicarla.",
    CL(),
    P("Sur", 480, 480, "teal", 90, en="a", ent="pop"),
    L(680, 480, 960, 480, "gris", 8, True, en="a", t=0.4),
    Bc(1180, 480, 400, 240, "blanco", "violeta", 5, 30, sombra=True, id="md", en="a", t=0.5),
    T("modelo", 1180, 480, tam=64, c="violeta", en="a", t=0.55),
    T("Sur + 2 = ?", 960, 760, tam=70, c="gris", id="suma", en="b"),
    K(960, 760, "cruz", 190, "rojo", en="b", t=0.6),
    sec="1. El problema")
v.b("Los modelos solo entienden números. [[a]]Por eso lo primero es saber de qué tipo es cada variable, [[b]]porque de eso depende cómo la vamos a preparar.",
    CL(),
    Bc(520, 460, 480, 260, "violetaclaro", "violeta", 5, 36, ent="subir", t=0.2),
    T("modelo", 520, 400, tam=56, c="violeta", t=0.3),
    T("12   0.5   3", 520, 520, tam=60, c="tinta", w="m", t=0.5),
    P("solo números", 520, 700, "verde", 46, en="a"),
    Bc(1400, 460, 480, 260, "blanco", "grisclaro", 5, 36, ent="subir", en="b"),
    T("tipo de variable", 1400, 430, tam=48, c="teal", en="b", t=0.1),
    T("→ cómo prepararla", 1400, 510, tam=40, c="gris", w="sb", en="b", t=0.4))

# ---------------- 2. Numéricas y categóricas
v.b("Hay dos grandes familias. [[a]]Las numéricas, [[b]]y las categóricas.",
    CL(),
    Bc(960, 230, 520, 110, "blanco", "tinta", 4, 30, id="nv", ent="subir", t=0.1),
    T("variable", 960, 230, tam=56, c="tinta", t=0.2),
    L(840, 290, 560, 420, "gris", 6, True, en="a"), L(1080, 290, 1360, 420, "gris", 6, True, en="b"),
    Bc(520, 520, 620, 200, "tealclaro", "teal", 5, 30, id="fn", en="a", t=0.3),
    T("numérica", 520, 520, tam=66, c="teal", en="a", t=0.4),
    Bc(1400, 520, 620, 200, "ambarclaro", "ambar", 5, 30, id="fc", en="b", t=0.3),
    T("categórica", 1400, 520, tam=66, c="ambar", en="b", t=0.4),
    sec="2. Numéricas y categóricas")
v.b("¿Cómo distinguirlas? Con una prueba rápida: [[a]]¿tiene sentido promediarla? Si la respuesta es sí, es numérica. [[b]]Si son nombres o niveles, es categórica.",
    P("¿tiene sentido promediarla?", 960, 860, "violeta", 58, en="a"),
    T("sí", 520, 700, tam=64, c="teal", en="a", t=0.7), T("no: nombres o niveles", 1400, 700, tam=52, c="ambar", en="b"))

# tabla del seguro
COLS = [("edad", ["23", "41", "67"], 270), ("n_hijos", ["0", "1", "3"], 610), ("region", ["Norte", "Sur", "Este"], 950),
        ("nivel_estudios", ["básico", "medio", "superior"], 1290), ("tiene_deuda", ["Sí", "No", "Sí"], 1630)]


def tabla(t0=0.0):
    out = []
    for k, (h, vals, x) in enumerate(COLS):
        tt = t0 + 0.12 * k
        out.append(Bx(x - 160, 250, x + 160, 340, "tealclaro", "teal", 3, 16, id=f"th{k}", en="a", t=tt))
        out.append(T(h, x, 295, tam=34 if len(h) < 12 else 30, c="teal", w="m", id=f"tht{k}", en="a", t=tt + 0.05))
        for r, val in enumerate(vals):
            y = 410 + r * 90
            out.append(Bx(x - 160, y - 38, x + 160, y + 38, "blanco", "linea", 3, 12, id=f"tc{k}{r}", en="a", t=tt + 0.1 + 0.05 * r))
            out.append(T(val, x, y, tam=38, c="tinta", w="sb", id=f"tv{k}{r}", en="a", t=tt + 0.12 + 0.05 * r))
    return out


def col_ring(k, en=None):
    x = COLS[k][2]
    return R(x, 430, 350, 430, "ambar", 28, id="rc", en=en)


v.b("Veamos un ejemplo: [[a]]los clientes de un seguro. Tenemos cinco columnas, y vamos a clasificarlas una por una.",
    CL(), tabla(0.1), T("clientes de un seguro", 960, 150, tam=54, c="tinta", id="tit", ent="subir", t=0.1), sec="3. Clasificar columnas")
v.b("Empecemos por la edad. [[a]]Veintitrés, cuarenta y uno, sesenta y siete. Se puede promediar, así que es numérica.",
    col_ring(0, "a"), P("numérica", 270, 760, "teal", 48, en="a", t=0.6, id="tg0"))
v.b("El número de hijos también es numérica, [[a]]pero solo toma valores enteros: cero, uno, tres. [[b]]No existe un cliente con uno y medio hijos. A eso se le llama discreta.",
    X("rc", dur=0.2), col_ring(1), P("numérica", 610, 760, "teal", 48, id="tg1", en="a"),
    T("¿1.5 hijos?", 610, 950, tam=44, c="gris", id="xh", en="b"), K(610, 950, "cruz", 100, "rojo", id="kx", en="b", t=0.5),
    P("discreta", 610, 850, "teal", 48, id="tg1b", en="b", t=1.0))
v.b("Lo contrario es una variable continua, [[a]]como la superficie de una casa: puede valer cualquier valor intermedio.",
    X("rc", "tg1", "tg1b", "xh", "kx", dur=0.25),
    L(300, 880, 1620, 880, "teal", 6, True, id="cl_l", en="a"), Bc(520, 880, 40, 40, "teal", None, 0, 20, id="dot", en="a", t=0.2),
    U("dot", x=1380, dur=2.6, en="a", t=0.4, ease="inout"),
    T("superficie: cualquier valor intermedio", 960, 970, tam=42, c="teal", w="sb", id="cl_t", en="a", t=0.5))
v.b("La región tiene nombres: [[a]]Norte, Sur, Este. No se promedian, así que es categórica. [[b]]Y como no hay ningún orden entre ellas, se llama nominal.",
    X("cl_l", "dot", "cl_t", dur=0.25), col_ring(2, "a"), P("categórica", 950, 760, "ambar", 48, id="tg2", en="a"), P("nominal", 950, 870, "ambar", 44, id="tg2b", en="b"))
v.b("El nivel de estudios también son etiquetas, [[a]]pero aquí sí hay un orden natural: básico, luego medio, luego superior. [[b]]Esa es una categórica ordinal.",
    X("rc", dur=0.2), col_ring(3), P("categórica", 1290, 760, "ambar", 48, id="tg3", en="a"), P("ordinal", 1290, 870, "ambar", 44, id="tg3b", en="b"))
v.b("Y tiene deuda, con sí o no, [[a]]es categórica con solo dos opciones. [[b]]Se llama binaria.",
    X("rc", dur=0.2), col_ring(4), P("categórica", 1630, 760, "ambar", 48, id="tg4", en="a"), P("binaria", 1630, 870, "ambar", 44, id="tg4b", en="b"))
v.b("Piensa un segundo: en las viviendas de la clase, ¿de qué tipo es el número de habitaciones? [[a]]¿Y el barrio?",
    CL(), T("n_habitaciones  →  ?", 260, 400, tam=64, c="teal", w="m", t=0.3, anc="lm"),
    T("barrio  →  ?", 260, 600, tam=64, c="ambar", w="m", en="a", anc="lm"), pausa=0.5)
v.b("El número de habitaciones es numérica discreta: uno, dos, tres, cuatro o cinco. [[a]]Y el barrio es categórica nominal: Centro, Norte, Sur o Este.",
    P("numérica discreta", 1350, 400, "teal", 48), P("categórica nominal", 1350, 600, "ambar", 48, en="a"),
    K(1700, 400, "check", 60, en=None), K(1700, 600, "check", 60, en="a"))
v.b("Resumiendo, es un árbol con cuatro ramas. [[a]]Las numéricas pueden ser continuas o discretas. [[b]]Y las categóricas pueden ser nominales u ordinales.",
    CL(),
    Bc(960, 230, 460, 100, "blanco", "tinta", 4, 30, ent="subir", t=0.1), T("variable", 960, 230, tam=52, t=0.2),
    L(880, 285, 600, 390, "gris", 5, True, en="a"), L(1040, 285, 1320, 390, "gris", 5, True, en="b"),
    Bc(580, 450, 460, 120, "tealclaro", "teal", 4, 28, en="a", t=0.2), T("numérica", 580, 450, tam=50, c="teal", en="a", t=0.3),
    Bc(1340, 450, 460, 120, "ambarclaro", "ambar", 4, 28, en="b", t=0.2), T("categórica", 1340, 450, tam=50, c="ambar", en="b", t=0.3),
    L(500, 515, 370, 640, "teal", 4, True, en="a", t=0.5), L(660, 515, 790, 640, "teal", 4, True, en="a", t=0.7),
    L(1260, 515, 1130, 640, "ambar", 4, True, en="b", t=0.5), L(1420, 515, 1550, 640, "ambar", 4, True, en="b", t=0.7),
    P("continua", 350, 710, "teal", 44, en="a", t=0.6), P("discreta", 810, 710, "teal", 44, en="a", t=0.8),
    P("nominal", 1110, 710, "ambar", 44, en="b", t=0.6), P("ordinal", 1570, 710, "ambar", 44, en="b", t=0.8),
    T("superficie", 350, 810, tam=34, c="gris", w="sb", en="a", t=1.0), T("n.º de hijos", 810, 810, tam=34, c="gris", w="sb", en="a", t=1.2),
    T("región", 1110, 810, tam=34, c="gris", w="sb", en="b", t=1.0), T("básico, medio, alto", 1570, 810, tam=34, c="gris", w="sb", en="b", t=1.2))

# ---------------- 4. One-hot
v.b("Ya sabemos qué es cada columna. Pero queda el problema del principio. [[a]]¿Cómo le damos la palabra Sur a un modelo que solo entiende números?",
    CL(), P("Sur", 600, 480, "ambar", 90, ent="subir", t=0.2), L(800, 480, 1080, 480, "gris", 8, True, en="a"),
    T("?", 1280, 480, tam=200, c="violeta", en="a", t=0.4), sec="4. One-hot")
v.b("La idea ingenua sería numerarlas: [[a]]Norte uno, [[b]]Sur dos, [[c]]Este tres. Pero eso inventa un orden que no existe.",
    CL(), P("Norte = 1", 480, 440, "teal", 56, en="a"), P("Sur = 2", 960, 440, "teal", 56, en="b"), P("Este = 3", 1440, 440, "teal", 56, en="c"),
    R(960, 560, 1300, 230, "rojo", 60, en="c", t=0.6),
    T("el modelo creería que Este vale el triple que Norte", 960, 640, tam=48, c="rojo", en="c", t=0.9), K(960, 800, "cruz", 120, "rojo", en="c", t=1.5))


def toggle(x, y, on, id, en, t=0):
    c = "verde" if on else "grisclaro"
    kx = x + 34 if on else x - 34
    return [Bc(x, y, 140, 76, c, None, 0, 38, id=f"{id}b", en=en, t=t, ent="fade"),
            Bc(kx, y, 58, 58, "blanco", None, 0, 29, id=f"{id}k", en=en, t=t + 0.05, ent="fade")]


v.b("La solución más usada se llama one-hot: [[a]]una columna por categoría. [[b]]Piensa en una fila de interruptores, donde solo uno está encendido.",
    CL(),
    P("one-hot = una columna por categoría", 960, 190, "violeta", 52, en="a"),
    T("Norte", 480, 400, tam=50, c="tinta", en="b"), T("Sur", 960, 400, tam=50, c="tinta", en="b", t=0.1), T("Este", 1440, 400, tam=50, c="tinta", en="b", t=0.2),
    toggle(480, 520, False, "s1", "b", 0.2), toggle(960, 520, True, "s2", "b", 0.4), toggle(1440, 520, False, "s3", "b", 0.6),
    T("0", 480, 640, tam=70, c="gris", en="b", t=0.6), T("1", 960, 640, tam=70, c="verde", en="b", t=0.8), T("0", 1440, 640, tam=70, c="gris", en="b", t=1.0),
    P("un cliente del Sur  =  (0, 1, 0)", 960, 840, "verde", 54, en="b", t=1.6))
HC = [("region_Norte", 1060), ("region_Sur", 1330), ("region_Este", 1600)]
CLI = [("Norte", (1, 0, 0)), ("Sur", (0, 1, 0)), ("Este", (0, 0, 1))]
v.b("Mira cómo queda la tabla. [[a]]La columna región se reemplaza por tres columnas nuevas. [[b]]Cada una vale uno si el cliente es de esa región, y cero si no.",
    CL(),
    T("antes", 400, 190, tam=44, c="gris"), T("después", 1330, 190, tam=44, c="gris", en="a"),
    Bx(250, 250, 550, 340, "ambarclaro", "ambar", 3, 16, id="oh0"), T("region", 400, 295, tam=36, c="ambar", w="m", id="oh0t"),
    [[Bx(250, 395 + r * 100, 550, 465 + r * 100, "blanco", "linea", 3, 12, en="a", t=0.05 * r), T(CLI[r][0], 400, 430 + r * 100, tam=38, c="tinta", w="sb", en="a", t=0.05 * r)] for r in range(3)],
    L(600, 480, 760, 480, "gris", 7, True, en="a", t=0.3),
    [[Bx(x - 130, 250, x + 130, 340, "violetaclaro", "violeta", 3, 16, en="a", t=0.4 + 0.15 * i), T(h, x, 295, tam=29, c="violeta", w="m", en="a", t=0.45 + 0.15 * i)] for i, (h, x) in enumerate(HC)],
    [[Bx(x - 130, 395 + r * 100, x + 130, 465 + r * 100, "verdeclaro" if CLI[r][1][i] else "blanco", "verde" if CLI[r][1][i] else "linea", 3, 12, en="b", t=0.1 * (r * 3 + i)),
      T(str(CLI[r][1][i]), x, 430 + r * 100, tam=44, c="verde" if CLI[r][1][i] else "gris", en="b", t=0.1 * (r * 3 + i) + 0.05)] for r in range(3) for i, (h, x) in enumerate(HC)])
v.b("Fíjate: cada fila tiene un solo uno. [[a]]El cliente del Sur es cero, uno, cero. [[b]]Y ya no hay orden inventado: las tres columnas valen lo mismo.",
    R(1330, 530, 830, 90, "verde", 30, id="rS", en="a"), T("(0, 1, 0)", 1330, 760, tam=60, c="verde", en="a", t=0.3),
    T("tres columnas, mismo peso", 1330, 860, tam=44, c="violeta", w="sb", en="b"))
v.b("Con las ordinales pasa distinto. [[a]]Como sí hay un orden, tiene sentido numerarlas respetándolo: básico uno, medio dos, superior tres.",
    CL(),
    Bx(380, 600, 700, 760, "ambarclaro", "ambar", 4, 18, en="a", dur=0.6), Bx(740, 480, 1060, 760, "ambarclaro", "ambar", 4, 18, en="a", t=0.4, dur=0.6),
    Bx(1100, 360, 1420, 760, "ambarclaro", "ambar", 4, 18, en="a", t=0.8, dur=0.6),
    T("básico", 540, 680, tam=42, c="ambar", en="a", t=0.7), T("medio", 900, 620, tam=42, c="ambar", en="a", t=1.1), T("superior", 1260, 560, tam=42, c="ambar", en="a", t=1.5),
    T("1", 540, 810, tam=64, c="tinta", en="a", t=0.8), T("2", 900, 810, tam=64, c="tinta", en="a", t=1.2), T("3", 1260, 810, tam=64, c="tinta", en="a", t=1.6))
v.b("Y para tiene deuda, con solo dos opciones, [[a]]basta una columna: uno si hay deuda, cero si no.",
    CL(), T("tiene_deuda", 960, 260, tam=50, c="ambar", w="m"),
    P("Sí", 640, 480, "rojo", 66, en="a"), L(780, 480, 1000, 480, "gris", 7, True, en="a", t=0.2), T("1", 1140, 480, tam=100, c="rojo", en="a", t=0.4),
    P("No", 640, 680, "verde", 66, en="a", t=0.6), L(780, 680, 1000, 680, "gris", 7, True, en="a", t=0.8), T("0", 1140, 680, tam=100, c="verde", en="a", t=1.0))
v.b("En pandas se hace con una sola línea: [[a]]get dummies. [[b]]En las viviendas de la clase, el barrio tiene cuatro valores, así que se crean cuatro columnas.",
    CL(),
    T("pd.get_dummies(viviendas, columns=['barrio'])", 960, 320, tam=40, c="tinta", w="m", fondo="#E8EDF4", pad=22, en="a"),
    [P(n, 340 + 420 * i, 640, "violeta", 46, en="b", t=0.25 * i) for i, n in enumerate(["barrio_Centro", "barrio_Norte", "barrio_Sur", "barrio_Este"])],
    N(4, 960, 800, "{:.0f}", 90, "violeta", suf=" columnas nuevas", en="b", dur=1.0))
v.b("Un error típico es dejar una categoría como texto, o numerarla sin pensar. [[a]]En el primer caso el modelo falla. En el segundo, aprende un orden falso.",
    CL(), P("error típico", 960, 250, "ambar", 56, ent="subir"),
    T("Sur", 560, 480, tam=80, c="tinta", w="sb"), K(560, 620, "cruz", 120, "rojo", en="a"), T("el modelo no lo entiende", 560, 760, tam=38, c="rojo", w="sb", en="a", t=0.3),
    T("Norte = 1, Sur = 2", 1360, 480, tam=70, c="tinta", w="sb"), K(1360, 620, "cruz", 120, "rojo", en="a", t=0.7), T("inventa un orden", 1360, 760, tam=38, c="rojo", w="sb", en="a", t=1.0))
v.b("Para llevarte: [[a]]si tiene sentido promediarla, es numérica. [[b]]Si son nombres o niveles, es categórica, [[c]]y a las categóricas se las convierte en columnas de ceros y unos.",
    CL(), P("¿se puede promediar?  →  numérica", 960, 330, "teal", 56, en="a"), P("¿nombres o niveles?  →  categórica", 960, 500, "ambar", 56, en="b"),
    P("categórica  →  columnas 0 y 1", 960, 670, "violeta", 56, en="c"), sec="5. Para llevar")

print(v.guardar(OUT), "beats")
