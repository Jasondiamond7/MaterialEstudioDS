import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_05_regularizacion.json"
v = V("Regularización: Ridge, Lasso y estandarizar")

v.portada("Regularización: Ridge, Lasso y estandarizar", "Cómo frenar a un modelo demasiado complejo",
          "Regularización. Ridge, Lasso, y por qué hay que estandarizar. Es la herramienta para frenar a los modelos que memorizan.")

# ---------------- 1. La idea
XS_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
YS_ = [0.42, 0.33, 0.55, 0.47, 0.66, 0.52, 0.74, 0.69, 0.62, 0.86]
v.b("Te acuerdas del sobreajuste: [[a]]un modelo que memoriza hasta el ruido de los datos. [[b]]¿Cómo lo frenamos?",
    CL(),
    G(380, 240, 1540, 880, 0, 11, 0.2, 1.0, [
        {"xs": XS_, "ys": YS_, "color": "tinta", "linea": False, "marcadores": 13, "revela": 1},
        {"xs": XS_, "ys": YS_, "color": "rojo", "suave": True, "grosor": 6, "revela": 0}], id="g0", rejilla=False, t=0.2),
    U("g0", **{"series.1.revela": 1}, dur=2.4, ease="lineal", en="a"),
    T("memoriza el ruido", 960, 300, tam=48, c="rojo", en="a", t=1.4), P("¿cómo lo frenamos?", 960, 980, "violeta", 56, en="b"),
    sec="1. La idea: castigar la complejidad")
v.b("La idea se parece a un impuesto. [[a]]Al error del modelo le sumamos una multa, [[b]]que crece cuando los coeficientes son grandes.",
    CL(),
    Bc(430, 480, 520, 260, "violetaclaro", "violeta", 5, 32, ent="subir", t=0.2), T("error del modelo", 430, 480, tam=50, c="violeta", t=0.3),
    T("+", 800, 480, tam=110, c="gris", en="a", t=0.0),
    Bc(1170, 480, 520, 260, "ambarclaro", "ambar", 5, 32, en="a", t=0.1), T("multa", 1170, 440, tam=60, c="ambar", en="a", t=0.2),
    T("por coeficientes grandes", 1170, 530, tam=34, c="ambar", w="sb", en="b"),
    P("regularizar = castigar la complejidad", 960, 800, "teal", 56, en="b", t=0.6))
v.b("¿Y qué es un coeficiente? [[a]]Es el número que le dice al modelo cuánto pesa cada variable. Por ejemplo, cuánto sube el precio por cada metro cuadrado.",
    CL(),
    T("precio  =", 280, 480, tam=64, c="tinta", anc="lm", t=0.2),
    P("β₁", 560, 480, "violeta", 70, en="a"), T("× superficie", 760, 480, tam=56, c="tinta", anc="lm", en="a", t=0.2),
    T("+", 1100, 480, tam=60, c="gris", en="a", t=0.4), P("β₂", 1220, 480, "azul", 70, en="a", t=0.5), T("× habitaciones   + …", 1400, 480, tam=50, c="tinta", anc="lm", en="a", t=0.6),
    T("coeficiente = peso de cada variable", 960, 750, tam=50, c="violeta", w="sb", en="a", t=0.9))

# ---- barras de coeficientes
BASE = 760
SCL = 6.2
GX = [330, 760, 1190, 1620]
VARS = ["superficie", "habitaciones", "antigüedad", "ruido"]
COEF = {"sin": [56.402, 11.942, -12.082, -0.4], "ridge": [53.801, 11.379, -11.309, -0.44], "lasso": [54.311, 9.708, -9.694, 0.0]}
COL = {"sin": "#94A3B8", "ridge": "azul", "lasso": "violeta"}
OFF = {"sin": -100, "ridge": 0, "lasso": 100}


def barras(kind, en, t0=0.0, etiquetas=(0, 3)):
    out = []
    for j, val in enumerate(COEF[kind]):
        x = GX[j] + OFF[kind]
        h = abs(val) * SCL
        if val >= 0:
            out.append(Bx(x - 42, BASE - max(h, 3), x + 42, BASE, COL[kind], None, 0, 6, id=f"b_{kind}{j}", grow="y", en=en, t=t0 + 0.12 * j, dur=0.6))
        else:
            out.append(Bx(x - 42, BASE, x + 42, BASE + max(h, 3), COL[kind], None, 0, 6, id=f"b_{kind}{j}", grow="-y", en=en, t=t0 + 0.12 * j, dur=0.6))
        if j in etiquetas:
            txt = f"{val:.1f}" if j == 0 else (f"{val:.2f}".replace("-", "−") if val != 0 else "0")
            ly = BASE - h - 36 if val > 0 else BASE - 40
            out.append(T(txt, x, ly, tam=34, c="tinta" if kind != "lasso" or j == 0 else "rojo", id=f"l_{kind}{j}", en=en, t=t0 + 0.6 + 0.1 * j))
    return out


def base_barras(en=None):
    return [L(160, BASE, 1800, BASE, "tinta", 4, id="bbase", dur=0.5)] + \
           [T(n, GX[j], 880, tam=40, c="tinta", w="sb", id=f"bn{j}", t=0.1 * j + 0.2) for j, n in enumerate(VARS)]


v.b("Veámoslo con los datos de la clase. [[a]]Predecimos el precio de las viviendas con la superficie, las habitaciones y la antigüedad. [[b]]Y le agregamos una columna de ruido aleatorio, que no tiene nada que ver con el precio.",
    CL(),
    [P(n, 330 + 420 * i, 450, c, 52, en="a", t=0.25 * i) for i, (n, c) in enumerate([("superficie", "violeta"), ("habitaciones", "azul"), ("antigüedad", "ambar")])],
    P("ruido aleatorio", 1590, 450, "rojo", 52, en="b"), T("sin relación con el precio", 1590, 560, tam=36, c="rojo", w="sb", en="b", t=0.4),
    sec="2. Ridge y Lasso en las viviendas")
v.b("Primero, sin regularizar. [[a]]La superficie tiene un coeficiente de cincuenta y seis coma cuatro. [[b]]El ruido, apenas menos cero coma cuatro.",
    CL(), base_barras(), barras("sin", "a", 0.0, (0, 3)),
    T("sin regularizar", 960, 190, tam=54, c="gris", w="sb", id="ttl", ent="subir", t=0.2))
v.b("Con Ridge, [[a]]la superficie baja a cincuenta y tres coma ocho. [[b]]Y el ruido queda en menos cero coma cuarenta y cuatro: achicado, pero no es cero.",
    barras("ridge", "a", 0.0, (0, 3)), U("l_sin0", alfa=0.35, dur=0.3, en="a"), U("l_sin3", alfa=0.35, dur=0.3, en="a"),
    U("ttl", texto="Ridge", color="azul", en="a"), R(GX[3], BASE - 30, 260, 150, "ambar", 30, en="b", id="rr", ent="fade"))
v.b("Con Lasso, [[a]]la superficie queda en cincuenta y cuatro coma tres. [[b]]¿Y el ruido? Exactamente cero. [[c]]Lasso la descartó.",
    X("rr", dur=0.2), barras("lasso", "a", 0.0, (0, 3)), U("l_ridge0", alfa=0.35, dur=0.3, en="a"), U("l_ridge3", alfa=0.35, dur=0.3, en="a"),
    U("ttl", texto="Lasso", color="violeta", en="a"), R(GX[3] + 100, BASE - 30, 110, 110, "rojo", 0, en="b", id="r0"),
    P("descartada", 1620, 640, "rojo", 48, en="c"))
v.b("Esa es la gran diferencia. Ridge penaliza la suma de los coeficientes al cuadrado: [[a]]los achica a todos, pero normalmente sin llegar a cero. Conserva todas las variables.",
    CL(),
    Bc(960, 300, 1500, 170, "azulclaro", "azul", 5, 36, ent="subir", t=0.2),
    T("error  +  α · Σ β²", 960, 300, tam=80, c="azul", t=0.3),
    P("Ridge · L2", 960, 150, "azul", 44, t=0.1),
    [[Bx(300 + 400 * i, 640 - h, 300 + 400 * i + 150, 640, "#94A3B8", None, 0, 6, id=f"ra{i}", grow="y", en="a", t=0.1 * i, dur=0.5)] for i, h in enumerate([300, 180, 120, 40])],
    [U(f"ra{i}", y0=640 - h2, dur=1.0, en="a", t=0.7) for i, h2 in enumerate([260, 150, 100, 30])],
    T("todas más pequeñas, ninguna en cero", 960, 780, tam=44, c="azul", w="sb", en="a", t=0.8))
v.b("Lasso, en cambio, penaliza la suma de los valores absolutos. [[a]]Eso sí puede llevar coeficientes exactamente a cero, [[b]]así que además elige variables.",
    CL(),
    Bc(960, 300, 1500, 170, "violetaclaro", "violeta", 5, 36, ent="subir", t=0.2),
    T("error  +  α · Σ |β|", 960, 300, tam=80, c="violeta", t=0.3),
    P("Lasso · L1", 960, 150, "violeta", 44, t=0.1),
    [[Bx(300 + 400 * i, 640 - h, 300 + 400 * i + 150, 640, "#94A3B8", None, 0, 6, id=f"la{i}", grow="y", en="a", t=0.1 * i, dur=0.5)] for i, h in enumerate([300, 180, 120, 40])],
    [U("la3", alfa=0, dur=0.8, en="a", t=0.5)], K(1590, 600, "cruz", 90, "rojo", en="a", t=0.9),
    P("selecciona variables", 960, 780, "verde", 50, en="b"))
v.b("Una forma de recordarlo: Ridge le pide a todo el equipo que trabaje un poco menos. [[a]]Lasso, en cambio, despide a los que no aportan.",
    CL(),
    [[Bc(330 + 420 * i, 460, 300, 180, "tealclaro", "teal", 4, 26, ent="subir", t=0.1 * i), T(n, 330 + 420 * i, 460, tam=40, c="teal", t=0.1 * i + 0.1)]
     for i, n in enumerate(["variable 1", "variable 2", "variable 3", "ruido"])],
    P("Ridge: todos trabajan menos", 960, 700, "azul", 48, t=0.6),
    P("Lasso: despide a los que no aportan", 960, 830, "violeta", 48, en="a"),
    K(1590, 460, "cruz", 110, "rojo", en="a", t=0.3))
v.b("¿Y cuánto castigo? Eso lo controla alfa. [[a]]Con alfa cero, no hay castigo: es la regresión de siempre. [[b]]Mientras más grande, más castigo.",
    CL(),
    L(300, 520, 1620, 520, "gris", 8, True, t=0.2), T("α", 960, 400, tam=110, c="violeta", t=0.3),
    Bc(380, 520, 44, 44, "teal", None, 0, 22, id="knob", en="a"), T("α = 0: sin castigo", 380, 620, tam=40, c="teal", w="sb", en="a", t=0.2),
    U("knob", x=1500, dur=2.2, ease="inout", en="b"), T("más castigo", 1500, 620, tam=40, c="rojo", w="sb", en="b", t=1.8))
v.b("Alfa es un hiperparámetro: [[a]]un número que eliges tú antes de entrenar. [[b]]Y lo eliges con la validación, nunca con la prueba.",
    CL(), P("alfa = hiperparámetro", 960, 330, "violeta", 62, ent="subir"),
    Bx(260, 560, 1100, 700, "violeta", None, 0, 20, grow="x", en="a", dur=0.7), T("entrenamiento", 680, 630, tam=44, c="blanco", en="a", t=0.5),
    Bx(1120, 560, 1390, 700, "azul", None, 0, 20, grow="x", en="b", dur=0.4), Bx(1410, 560, 1660, 700, "rojo", None, 0, 20, grow="x", en="b", t=0.5, dur=0.4),
    T("elige alfa", 1255, 760, tam=40, c="azul", en="b", t=0.5), K(1535, 760, "cruz", 50, "rojo", en="b", t=1.0),
    T("validación", 1255, 630, tam=34, c="blanco", en="b", t=0.6), T("prueba", 1535, 630, tam=34, c="blanco", en="b", t=0.7))
v.b("Hay un tercer castigo, Elastic Net, [[a]]que simplemente mezcla los dos.",
    CL(), P("Ridge", 560, 480, "azul", 70), T("+", 800, 480, tam=100, c="gris", en="a"), P("Lasso", 1040, 480, "violeta", 70, en="a", t=0.2),
    T("=", 1280, 480, tam=100, c="gris", en="a", t=0.4), P("Elastic Net", 1560, 480, "teal", 70, en="a", t=0.6))

# ---- camino de Lasso
LA = [0.01, 0.016, 0.025, 0.04, 0.063, 0.1, 0.158, 0.251, 0.398, 0.631, 1.0, 1.585, 2.512, 3.981, 6.31, 10.0, 15.849, 25.119, 39.811, 63.096, 79.433]
import math
LX = [round(math.log10(a), 3) for a in LA]
RUT = [[56.39, 56.39, 56.38, 56.36, 56.34, 56.3, 56.24, 56.15, 56.0, 55.75, 55.36, 54.75, 53.77, 52.23, 49.77, 45.89, 40.05, 30.78, 16.09, 0.0, 0.0],
       [11.93, 11.92, 11.91, 11.9, 11.87, 11.83, 11.76, 11.65, 11.48, 11.22, 10.81, 10.17, 9.14, 7.52, 4.94, 0.86, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-12.07, -12.06, -12.05, -12.03, -12.01, -11.96, -11.89, -11.78, -11.6, -11.33, -10.89, -10.19, -9.08, -7.33, -4.56, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-0.39, -0.38, -0.37, -0.36, -0.33, -0.3, -0.24, -0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
PC = ["violeta", "azul", "ambar", "rojo"]
v.b("Mira ahora el camino de Lasso. [[a]]Vamos subiendo alfa, de menos castigo a más castigo, [[b]]y los coeficientes se van apagando de a uno.",
    CL(),
    G(300, 230, 1480, 850, -2.2, 2.1, -15, 62, [{"xs": LX, "ys": RUT[j], "color": PC[j], "grosor": 7, "revela": 0} for j in range(4)],
      id="gp", yticks=[0, 20, 40, 60], yfmt="{:.0f}", xlabel="alpha (escala logarítmica: más castigo →)", ylabel="coef.", t=0.2, rejilla=True),
    [T(lbl, lx, -15, tam=30, c="gris", ref="gp", dy=34, id=f"gx{k}", en="a", t=0.05 * k) for k, (lx, lbl) in enumerate([(-2, "0.01"), (-1, "0.1"), (0, "1"), (1, "10"), (2, "100")])],
    [[Bc(1560, 330 + 80 * j, 40, 14, PC[j], None, 0, 7, en="a", t=0.1 * j), T(n, 1600, 330 + 80 * j, tam=36, c=PC[j], anc="lm", en="a", t=0.1 * j)] for j, n in enumerate(VARS)],
    U("gp", **{f"series.{j}.revela": 1 for j in range(4)}, dur=4.5, ease="lineal", en="b"))
v.b("El ruido es el primero en apagarse: [[a]]con alfa cerca de cero coma cuatro. [[b]]Después, la antigüedad y las habitaciones. [[c]]Y la superficie, la que de verdad importa, aguanta hasta el final.",
    L(-0.4, -15, -0.4, 62, "rojo", 4, guion=True, ref="gp", id="v04", en="a"),
    T("alfa ≈ 0.4", -0.4, 58, tam=36, c="rojo", ref="gp", dx=12, anc="lm", id="v04t", en="a", t=0.3),
    R(-0.4, 0, 70, 70, "rojo", 0, ref="gp", id="rr0", en="a", t=0.2), X("rr0", en="b", dur=0.3),
    R(1.1, 0, 80, 80, "azul", 0, ref="gp", id="rr1", en="b", t=0.2), X("rr1", en="c", dur=0.3),
    R(1.8, 0, 90, 90, "violeta", 0, ref="gp", id="rr2", en="c"))
v.b("Ojo: que Lasso anule una variable depende de alfa y de los datos. [[a]]No es magia, es el resultado de cuánto castigo aplicas.",
    CL(), P("depende de alfa y de los datos", 960, 430, "ambar", 62, ent="subir"), T("no es magia", 960, 600, tam=56, c="gris", w="sb", en="a"))
v.b("Piensa un segundo: si lo que quieres es quedarte con pocas variables, [[a]]¿usas Ridge o Lasso?",
    CL(), P("Ridge", 600, 480, "azul", 90, t=0.3), P("Lasso", 1320, 480, "violeta", 90, t=0.4), T("o", 960, 480, tam=60, c="gris", t=0.35), pausa=0.5)
v.b("Lasso. [[a]]Porque puede llevar coeficientes exactamente a cero, y esas variables salen del modelo.",
    R(1320, 480, 380, 190, "verde", 50, id="rl", en=None), K(1320, 640, "check", 90, "verde", en="a"))

# ---------------- 3. Estandarizar
v.b("Ahora, la pregunta clave: ¿por qué hay que estandarizar? [[a]]Porque la penalización mira el tamaño de los coeficientes, [[b]]y ese tamaño depende de las unidades.",
    CL(), P("la multa mira el tamaño del coeficiente", 960, 330, "ambar", 56, en="a"), P("el tamaño depende de las unidades", 960, 520, "rojo", 56, en="b"),
    sec="3. Por qué estandarizar")
v.b("Mira este ejemplo. Un mismo efecto, expresado en dos unidades. [[a]]Tres por cada metro cuadrado, [[b]]o trescientos por cada cien metros cuadrados.",
    CL(),
    Bc(520, 440, 640, 260, "tealclaro", "teal", 5, 30, ent="subir", en="a"), T("3", 520, 410, tam=110, c="teal", en="a", t=0.1), T("por cada m²", 520, 510, tam=40, c="teal", w="sb", en="a", t=0.2),
    Bc(1400, 440, 640, 260, "ambarclaro", "ambar", 5, 30, ent="subir", en="b"), T("300", 1400, 410, tam=110, c="ambar", en="b", t=0.1), T("por cada 100 m²", 1400, 510, tam=40, c="ambar", w="sb", en="b", t=0.2),
    T("=", 960, 440, tam=120, c="gris", en="b", t=0.4), P("el mismo efecto", 960, 760, "violeta", 56, en="b", t=0.6))
v.b("Con Ridge y alfa igual a uno, la multa es el coeficiente al cuadrado. [[a]]Tres al cuadrado es nueve. [[b]]Trescientos al cuadrado es noventa mil.",
    T("multa  =  α · β²   con α = 1", 960, 860, tam=44, c="gris", w="sb", t=0.2),
    T("3²  =  9", 520, 640, tam=70, c="teal", en="a"), T("300²  =  90.000", 1400, 640, tam=70, c="ambar", en="b"))
v.b("Es exactamente el mismo efecto, [[a]]pero castigado diez mil veces más.",
    CL(),
    Bx(300, 720, 400, 780, "teal", None, 0, 8, grow="x", dur=0.5, t=0.2), T("9", 450, 750, tam=44, c="teal", anc="lm", t=0.5),
    Bx(300, 820, 1500, 900, "ambar", None, 0, 8, grow="x", en="a", dur=1.2), T("90.000", 1560, 860, tam=44, c="ambar", anc="lm", en="a", t=1.0),
    T("× 10.000", 960, 400, tam=150, c="rojo", en="a"), T("veces más castigo", 960, 520, tam=54, c="rojo", w="sb", en="a", t=0.8))
v.b("Piensa un segundo: si pasas la superficie de metros cuadrados a centenas de metros cuadrados, [[a]]¿Ridge castiga más o menos a esa variable?",
    CL(), P("m²", 560, 480, "teal", 90, t=0.3), L(760, 480, 1160, 480, "gris", 8, True, t=0.5), P("centenas de m²", 1500, 480, "ambar", 80, t=0.7),
    T("¿más castigo o menos castigo?", 960, 760, tam=60, c="violeta", en="a"), pausa=0.5)
v.b("Más. [[a]]El coeficiente se multiplica por cien, así que la multa se multiplica por diez mil. [[b]]Cambiar la unidad cambió el modelo.",
    P("más castigo", 960, 880, "rojo", 66), T("coeficiente × 100  →  multa × 10.000", 960, 960, tam=44, c="rojo", w="sb", en="a"),
    T("misma realidad, otro modelo", 960, 620, tam=50, c="gris", w="sb", en="b"))
v.b("La solución es estandarizar: [[a]]a cada valor le restas la media, [[b]]y lo divides por la desviación estándar. El resultado se llama puntaje z.",
    CL(),
    T("z", 360, 450, tam=150, c="violeta", t=0.2), T("=", 490, 450, tam=100, c="gris", t=0.3),
    T("x − media", 840, 400, tam=80, c="tinta", id="znum", en="a"), L(600, 450, 1080, 450, "tinta", 6, id="zlin", en="b", t=0.1, dur=0.4),
    T("desviación estándar", 840, 510, tam=64, c="azul", id="zden", en="b", t=0.3),
    P("queda todo en una escala comparable", 960, 800, "teal", 56, en="b", t=1.0))
v.b("Por ejemplo, una casa de ciento veinte metros cuadrados, [[a]]con media ochenta y desviación veinte. [[b]]Ciento veinte menos ochenta, entre veinte: z igual a dos.",
    CL(),
    L(300, 560, 1620, 560, "gris", 6, True, t=0.2),
    [T(lbl, 300 + (xv - 40) * 16.5, 620, tam=36, c="gris", w="sb", t=0.3) for xv, lbl in ((40, "40"), (60, "60"), (80, "80"), (100, "100"), (120, "120"))],
    Bc(300 + 40 * 16.5, 560, 28, 28, "teal", None, 0, 14, en="a"), T("media 80", 960, 470, tam=44, c="teal", en="a", t=0.1),
    Bc(300 + 80 * 16.5, 560, 40, 40, "ambar", None, 0, 20, en="b"), T("120 m²", 1620, 470, tam=44, c="ambar", en="b", t=0.1),
    T("(120 − 80) / 20  =  2", 960, 810, tam=70, c="tinta", en="b", t=0.5), P("2 desviaciones sobre el promedio", 960, 940, "violeta", 46, en="b", t=1.2))
v.b("Con las habitaciones pasa igual. [[a]]Cuatro coma cinco habitaciones, con media tres y desviación uno coma cinco. [[b]]Cuatro coma cinco menos tres, entre uno coma cinco: z igual a uno.",
    CL(),
    T("habitaciones: 4.5  ·  media 3  ·  desviación 1.5", 960, 330, tam=48, c="azul", w="sb", en="a"),
    T("(4.5 − 3) / 1.5  =  1", 960, 520, tam=90, c="tinta", en="b"), P("1 desviación sobre el promedio", 960, 740, "violeta", 52, en="b", t=0.8))
v.b("Ahora la superficie y las habitaciones hablan en el mismo idioma: [[a]]cuántas desviaciones se alejan de su promedio.",
    CL(), P("superficie", 520, 480, "violeta", 70), P("habitaciones", 1400, 480, "azul", 70, t=0.2), T("↔", 960, 480, tam=100, c="gris", en="a"),
    P("misma escala", 960, 720, "teal", 66, en="a", t=0.4))

# ---- R2 grupos
RB = 880
RS_ = 520


def barra_r2(x, val, c, ident, en, t):
    return [Bx(x - 70, RB - val * RS_, x + 70, RB, c, None, 0, 8, id=ident, grow="y", en=en, t=t, dur=0.9),
            T(f"{val:.3f}", x, RB - val * RS_ - 36, tam=38, c="tinta", id=ident + "l", en=en, t=t + 0.8)]


v.b("Con las viviendas de la clase, probamos Ridge con alfa igual a treinta. [[a]]Con la superficie en metros cuadrados, el R cuadrado de validación es cero coma novecientos dieciséis.",
    CL(),
    T("Ridge con α = 30  ·  R² en validación", 960, 170, tam=48, c="gris", w="sb", ent="subir"),
    L(300, RB, 1620, RB, "tinta", 4, t=0.2), T("superficie en m²", 600, 940, tam=42, c="tinta", w="sb", t=0.3), T("superficie en centenas de m²", 1320, 940, tam=42, c="tinta", w="sb", t=0.3),
    barra_r2(520, 0.916, "rojo", "r2a", "a", 0.0), P("sin estandarizar", 1560, 260, "rojo", 40, t=0.5), P("estandarizando", 1560, 340, "teal", 40, t=0.7),
    sec="4. El efecto en las viviendas")
v.b("Si medimos la superficie en centenas de metros cuadrados, [[a]]cae a cero coma quinientos treinta y dos. [[b]]Solo por cambiar la unidad.",
    barra_r2(1240, 0.532, "rojo", "r2b", "a", 0.0), R(1240, RB - 150, 220, 340, "rojo", 30, id="rr3", en="b"), P("¡solo cambió la unidad!", 1320, 450, "rojo", 46, en="b", t=0.3))
v.b("Con estandarización, [[a]]el R cuadrado queda en cero coma ochocientos ochenta y siete [[b]]en los dos casos.",
    X("rr3", dur=0.2), barra_r2(680, 0.887, "teal", "r2c", "a", 0.0), barra_r2(1400, 0.887, "teal", "r2d", "b", 0.0), P("iguales", 1040, 560, "verde", 54, en="b", t=0.9))
v.b("Algo parecido pasa con los coeficientes. [[a]]Sin estandarizar, son dos coma dieciocho, ocho coma diecinueve y menos cero coma ochenta y seis. [[b]]Estandarizados, cincuenta y seis coma cuatro, once coma nueve y menos doce coma uno.",
    CL(),
    T("sin estandarizar", 520, 250, tam=48, c="rojo", t=0.2), T("estandarizados", 1400, 250, tam=48, c="teal", t=0.2),
    [T(t_, 520, 400 + 120 * i, tam=70, c="rojo", w="b", en="a", t=0.2 * i) for i, t_ in enumerate(["2.18", "8.19", "−0.86"])],
    [T(t_, 1400, 400 + 120 * i, tam=70, c="teal", w="b", en="b", t=0.2 * i) for i, t_ in enumerate(["56.4", "11.9", "−12.1"])],
    T("⇄", 960, 520, tam=100, c="gris", en="b", t=0.3), P("no se pueden comparar: cambian las unidades", 960, 850, "ambar", 48, en="b", t=0.8))
v.b("Un error típico: estandarizar con todos los datos, incluida la prueba. [[a]]El estandarizador se ajusta solo con el entrenamiento, y lo más cómodo es hacerlo dentro de un Pipeline.",
    CL(),
    P("error: estandarizar con todo", 960, 220, "rojo", 52, ent="subir"),
    T("make_pipeline(StandardScaler(), Ridge(alpha=10))", 960, 470, tam=42, c="tinta", w="m", fondo="#E8EDF4", pad=24, en="a"),
    T("make_pipeline(StandardScaler(), Lasso(alpha=2))", 960, 590, tam=42, c="tinta", w="m", fondo="#E8EDF4", pad=24, en="a", t=0.5),
    P("fit solo con el entrenamiento", 960, 800, "teal", 52, en="a", t=1.0))
v.b("Y un recordatorio: estandarizar no arregla un modelo malo. [[a]]Solo hace comparable la penalización.",
    CL(), P("no arregla un modelo malo", 960, 440, "ambar", 64, ent="subir"), T("solo hace comparable la penalización", 960, 620, tam=52, c="gris", w="sb", en="a"))
v.b("Para llevarte: [[a]]Ridge achica, Lasso puede anular. [[b]]Estandariza siempre antes de Ridge o Lasso. [[c]]Y alfa se elige con la validación.",
    CL(),
    P("Ridge achica  ·  Lasso puede anular", 960, 330, "azul", 56, en="a"), P("estandariza antes de Ridge o Lasso", 960, 520, "teal", 56, en="b"),
    P("alfa se elige con la validación", 960, 710, "violeta", 56, en="c"), sec="5. Para llevar")

print(v.guardar(OUT), "beats")
