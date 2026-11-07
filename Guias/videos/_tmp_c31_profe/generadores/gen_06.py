import sys, json, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_06_ajuste.json"
D = json.load(open(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe\datos31.json", encoding="utf-8"))
A = D["ajuste"]
v = V("Subajuste, sobreajuste y regularización")

v.portada("Subajuste, sobreajuste y regularización", "Cuándo un modelo aprende poco, cuándo memoriza, y cómo frenarlo",
          "Subajuste, sobreajuste y regularización. Cómo saber si un modelo aprendió de verdad, o solo se memorizó los datos.")

# ---- 1. Hook
v.b("Dos estudiantes se preparan para un examen. [[a]]Uno no estudia casi nada. [[b]]El otro se memoriza las respuestas del examen del año pasado, palabra por palabra. [[c]]¿Cuál de los dos va a tener problemas?",
    CL(), Bc(560, 520, 620, 380, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("no estudia", 560, 520, tam=60, c="rojo", en="a", t=0.1),
    Bc(1360, 520, 620, 380, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir"), T("memoriza", 1360, 460, tam=60, c="ambar", en="b", t=0.1), T("las respuestas viejas", 1360, 560, tam=38, c="gris", w="sb", en="b", t=0.3),
    T("¿cuál tiene problemas?", 960, 820, tam=60, c="violeta", en="c"), sec="1. Aprender poco, aprender de memoria")
v.b("Los dos. [[a]]El primero, porque no aprendió nada. Eso es subajuste. [[b]]El segundo, porque le va bien con lo conocido, y mal con preguntas nuevas. Eso es sobreajuste.",
    CL(), P("subajuste", 560, 400, "rojo", 66, en="a"), T("no aprendió el patrón", 560, 560, tam=44, c="gris", w="sb", en="a", t=0.3), K(560, 700, "cruz", 90, en="a", t=0.5),
    P("sobreajuste", 1360, 400, "ambar", 66, en="b"), T("memorizó el ruido", 1360, 560, tam=44, c="gris", w="sb", en="b", t=0.3), K(1360, 700, "cruz", 90, c="rojo", en="b", t=0.5))

# ---- 2. Definiciones
v.b("En un modelo, se ve así. [[a]]El subajuste es cuando funciona mal, incluso con los datos de entrenamiento. [[b]]El sobreajuste, cuando funciona muy bien en entrenamiento, y mucho peor en prueba.",
    CL(),
    Bc(560, 480, 700, 420, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("subajuste", 560, 330, tam=52, c="rojo", en="a", t=0.1), T("entrenamiento: mal", 560, 470, tam=40, c="tinta", w="sb", en="a", t=0.2), T("prueba: mal", 560, 550, tam=40, c="tinta", w="sb", en="a", t=0.3),
    Bc(1360, 480, 700, 420, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir"), T("sobreajuste", 1360, 330, tam=52, c="ambar", en="b", t=0.1), T("entrenamiento: muy bien", 1360, 470, tam=40, c="tinta", w="sb", en="b", t=0.2), T("prueba: mucho peor", 1360, 550, tam=40, c="tinta", w="sb", en="b", t=0.3),
    sec="2. Cómo se ve cada uno")
v.b("La causa de fondo es la complejidad del modelo, frente a la cantidad de datos disponibles. [[a]]Poca complejidad, y no alcanza para capturar el patrón. [[b]]Demasiada, y el modelo se ajusta hasta al ruido.",
    CL(), P("complejidad  vs  datos disponibles", 960, 270, "violeta", 56, ent="subir", t=0.2),
    L(380, 560, 1540, 560, "gris", 8, True, t=0.4), T("poca", 380, 640, tam=44, c="rojo", en="a"), T("mucha", 1540, 640, tam=44, c="ambar", en="b"),
    P("no captura el patrón", 380, 480, "rojo", 40, en="a"), P("se ajusta hasta al ruido", 1540, 480, "ambar", 40, en="b"), T("complejidad del modelo", 960, 640, tam=40, c="gris", w="sb", t=0.6))

# ---- 3. Ajuste visual
GX0, GY0, GX1, GY1 = 240, 200, 1240, 880
YMIN, YMAX = -3, 16
cl = lambda ys: [max(YMIN + 0.2, min(YMAX - 0.2, y)) for y in ys]  # noqa: E731


def nube(en=None):
    return G(GX0, GY0, GX1, GY1, 0, 10, YMIN, YMAX,
             [dict(xs=A["xt"], ys=A["yt"], color="violeta", linea=False, marcadores=11), dict(xs=A["xe"], ys=A["ye"], color="ambar", linea=False, marcadores=11),
              dict(xs=A["xl"], ys=cl(A["c1"]), color="tinta", grosor=7, revela=0.0, lapiz=False)],
             xticks=[0, 2, 4, 6, 8, 10], yticks=[0, 5, 10, 15], yfmt="{:.0f}", xlabel="x", id="gj", en=en, t=0.1)


v.b("Veámoslo con datos. Esta es una nube de sesenta puntos, [[a]]cuarenta y dos para entrenar, en violeta, [[b]]y dieciocho para probar, en ámbar. Son datos simulados de la clase, con ruido.",
    CL(), nube(), T("y", GX0, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.3),
    P("entrenamiento · 42", 1560, 400, "violeta", 40, en="a"), P("prueba · 18", 1560, 500, "ambar", 40, en="b"), sec="3. Tres ajustes sobre la misma nube")
v.b("Primero, un modelo muy simple: una recta, grado uno. [[a]]No sigue la forma de la nube. Falla incluso con los puntos de entrenamiento. [[b]]El R cuadrado es cero coma cincuenta y dos en entrenamiento, y apenas cero coma diez en prueba.",
    U("gj", **{"series.2.revela": 1.0}, dur=1.8), g("r2", P("grado 1 · subajuste", 1560, 600, "rojo", 40, mw=560, en="a")),
    g("r2", P("entrenamiento 0.52", 1560, 700, "violeta", 40, en="b"), P("prueba 0.10", 1560, 790, "ambar", 40, en="b", t=0.4)))
v.b("Ahora, un polinomio de grado dos, una curva suave. [[a]]Sigue la tendencia sin forzar nada. [[b]]El R cuadrado sube a cero coma ochenta en entrenamiento, y cero coma cincuenta y uno en prueba.",
    U("gj", **{"series.2.ys": cl(A["c2"])}, dur=1.5), XG("r2", dur=0.3), g("r2", P("grado 2 · ajuste adecuado", 1560, 600, "verde", 40, mw=560, en="a")),
    g("r2", P("entrenamiento 0.80", 1560, 700, "violeta", 40, en="b"), P("prueba 0.51", 1560, 790, "ambar", 40, en="b", t=0.4)))
v.b("Y con grado quince, la curva se retuerce para pasar por casi todos los puntos. [[a]]En entrenamiento, el R cuadrado llega a cero coma noventa. [[b]]Pero en prueba cae a menos cuatro coma setenta y nueve: peor que predecir el promedio.",
    U("gj", **{"series.2.ys": cl(A["c15"]), "series.2.color": "rojo"}, dur=1.8), XG("r2", dur=0.3), g("r2", P("grado 15 · sobreajuste", 1560, 600, "rojo", 40, mw=560, en="a")),
    g("r2", P("entrenamiento 0.90", 1560, 700, "violeta", 40, en="a"), P("prueba −4.79", 1560, 790, "rojo", 40, en="b", t=0.4)))
v.b("Fíjate qué engañoso es esto. [[a]]El grado quince tiene el mejor resultado en entrenamiento. [[b]]Si solo miraras ese número, lo elegirías. Y sería un error.",
    R(1560, 700, 340, 80, "violeta", 30, id="rr1", en="a", ent="fade"), X("rr1", en="b", dur=0.3), K(1760, 700, "cruz", 60, c="rojo", en="b"))
v.b("Piensa un segundo: [[a]]si el R cuadrado de entrenamiento es alto y el de prueba es muy bajo, ¿qué tienes?",
    CL(), T("entrenamiento: 0.90", 960, 380, tam=64, c="violeta", en="a"), T("prueba: −4.79", 960, 500, tam=64, c="rojo", en="a", t=0.3), T("¿qué tienes?", 960, 700, tam=90, c="tinta", en="a", t=0.7), pausa=0.6)
v.b("Sobreajuste. [[a]]La brecha entre entrenamiento y prueba es el síntoma.",
    CL(), P("sobreajuste", 960, 400, "ambar", 100, ent="subir"), P("la brecha es el síntoma", 960, 680, "violeta", 60, en="a"))

# ---- 4. Detectar con la curva
v.b("Para detectarlo, se compara entrenamiento y prueba, al subir la complejidad. [[a]]En entrenamiento, el R cuadrado casi siempre sube. [[b]]En prueba, sube un poco, y luego se cae.",
    CL(),
    G(240, 200, 1240, 880, 1, 15, -1.2, 1.05, [dict(xs=list(range(1, 16)), ys=A["tr"], color="violeta", grosor=7, marcadores=8, revela=0.0), dict(xs=list(range(1, 16)), ys=[max(-1.15, y) for y in A["te"]], color="ambar", grosor=7, marcadores=8, revela=0.0)],
      xticks=[1, 3, 5, 7, 9, 11, 13, 15], yticks=[-1, -0.5, 0, 0.5, 1], yfmt="{:g}", xlabel="grado del polinomio (complejidad)", id="gc", t=0.1), T("R²", 240, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.2),
    U("gc", **{"series.0.revela": 1.0}, en="a", dur=2.0), U("gc", **{"series.1.revela": 1.0}, en="b", dur=2.0),
    P("entrenamiento", 1560, 420, "violeta", 40, en="a"), P("prueba", 1560, 520, "ambar", 40, en="b"), sec="4. Detectarlo con una curva")
v.b("La brecha crece con la complejidad. [[a]]Con grados trece a quince, el R cuadrado de prueba baja de menos dos, y se sale de la escala. [[b]]Por eso no se elige el grado de mayor entrenamiento, sino uno sencillo con buena prueba.",
    R(1170, 850, 330, 100, "rojo", 30, id="rg", en="a", ent="fade"), P("grados 13 a 15: fuera de escala", 1560, 640, "rojo", 36, en="a", mw=560), P("elige simple, con buena prueba", 1560, 780, "verde", 40, en="b", mw=560))
v.b("El grado dos da cero coma cincuenta y uno en prueba, [[a]]y algunos grados intermedios dan algo parecido. [[b]]Con tan pocos puntos de prueba, se prefiere el más simple.",
    X("rg", dur=0.3), P("grado 2: prueba 0.51", 1560, 280, "verde", 40), P("otros grados ≈ 0.4 a 0.5", 1560, 360, "gris", 36, en="a"), P("se prefiere el más simple", 1560, 440, "teal", 40, en="b"))
v.b("Hay un detalle del notebook de la clase. [[a]]Con grado quince, sin escalar los datos, el R cuadrado de entrenamiento no fue altísimo: fue cero coma treinta. [[b]]Menor que el cero coma ochenta del grado dos.",
    CL(), P("alerta del material", 960, 250, "ambar", 54, ent="subir", t=0.2), T("grado 15 · sin escalar", 960, 380, tam=52, c="tinta", en="a"),
    Bc(560, 640, 520, 260, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("entrenamiento", 560, 560, tam=40, c="gris", w="sb", en="a", t=0.1), N(0.30, 560, 660, "{:.2f}", 120, "violeta", en="a", t=0.2, dur=0.9),
    Bc(1360, 640, 520, 260, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("grado 2 (referencia)", 1360, 560, tam=40, c="gris", w="sb", en="b", t=0.1), N(0.80, 1360, 660, "{:.2f}", 120, "verde", en="b", t=0.2, dur=0.9))
v.b("Eso no es solo sobreajuste: es un problema numérico. [[a]]Elevar números a la quince produce columnas enormes, y la matriz queda mal condicionada. [[b]]Escalando los datos, vuelve a verse el patrón: cero coma noventa en entrenamiento, y menos cuatro coma setenta y nueve en prueba.",
    CL(), P("problema numérico", 960, 270, "rojo", 60, ent="subir", t=0.2), P("x elevado a 15: números enormes", 960, 470, "ambar", 48, en="a"), P("matriz mal condicionada", 960, 600, "rojo", 48, en="a", t=0.4),
    P("escalando: 0.90 y −4.79", 960, 790, "verde", 52, en="b"), sec=None)

# ---- 5. Regularización
v.b("¿Cómo frenar el sobreajuste? [[a]]Una forma es la regularización. [[b]]Le agrega al error una penalización por tener coeficientes grandes.",
    CL(), P("¿cómo frenar el sobreajuste?", 960, 260, "violeta", 54, ent="subir", t=0.2), T("error  +  penalización", 960, 520, tam=84, c="tinta", en="b"),
    P("regularización", 960, 400, "teal", 56, en="a"), P("castiga coeficientes grandes", 960, 720, "ambar", 54, en="b", t=0.5), sec="5. Regularización: Ridge y Lasso")
v.b("Es como un impuesto a los coeficientes. [[a]]Cada coeficiente grande paga más. [[b]]Así el modelo solo los mantiene si de verdad ayudan a reducir el error.",
    CL(), P("un impuesto a los coeficientes", 960, 270, "violeta", 56, ent="subir", t=0.2), Bc(960, 600, 1100, 300, "blanco", "teal", 5, 30, sombra=True, ent="subir", t=0.3),
    T("coeficiente grande  →  paga más", 960, 560, tam=48, c="tinta", en="a"), T("solo lo conserva si ayuda de verdad", 960, 650, tam=44, c="teal", w="sb", en="b"))
v.b("Hay dos sabores. Ridge, o L dos, [[a]]suma el cuadrado de los coeficientes. [[b]]Lasso, o L uno, suma sus valores absolutos.",
    CL(),
    Bc(520, 520, 780, 460, "blanco", "azul", 5, 30, sombra=True, ent="subir", t=0.2), T("Ridge  (L2)", 520, 340, tam=56, c="azul", t=0.3), T("error + α · Σ β²", 520, 480, tam=46, c="tinta", en="a"),
    Bc(1400, 520, 780, 460, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir"), T("Lasso  (L1)", 1400, 340, tam=56, c="ambar", en="b", t=0.1), T("error + α · Σ |β|", 1400, 480, tam=46, c="tinta", en="b", t=0.3),
    P("α = fuerza de la penalización", 960, 860, "violeta", 46, en="b", t=0.6))
v.b("Y se comportan distinto. [[a]]Ridge encoge todos los coeficientes, pero rara vez los anula. [[b]]Lasso puede llevar a cero los coeficientes poco útiles. Es una selección automática de variables.",
    CL(), Bc(520, 520, 780, 400, "blanco", "azul", 5, 30, sombra=True, ent="subir", t=0.1), T("Ridge", 520, 400, tam=56, c="azul"), P("encoge todos", 520, 540, "azul", 48, en="a"), T("rara vez anula", 520, 650, tam=38, c="gris", w="sb", en="a", t=0.3),
    Bc(1400, 520, 780, 400, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir"), T("Lasso", 1400, 400, tam=56, c="ambar", en="b", t=0.1), P("puede llevar a cero", 1400, 540, "ambar", 48, en="b", t=0.2), T("selección automática", 1400, 650, tam=38, c="gris", w="sb", en="b", t=0.4))

# ---- 6. Ejemplo resuelto
v.b("Veamos las cuentas con el ejemplo del apunte. [[a]]Tres coeficientes sin penalizar: cinco, menos uno coma cuatro, y cero coma seis. [[b]]Con alfa igual a uno.",
    CL(), T("variable", 420, 280, tam=38, c="gris", w="sb"), T("sin penalizar", 880, 280, tam=38, c="gris", w="sb"),
    *[[T(n, 420, 400 + 130 * i, tam=54, c="tinta"), T(val, 880, 400 + 130 * i, tam=60, c="violeta", en="a", t=0.2 * i)] for i, (n, val) in enumerate((("x₁", "5.0"), ("x₂", "−1.4"), ("x₃", "0.6")))],
    P("α = 1", 1500, 400, "ambar", 70, en="b"), T("columnas ortonormales", 1500, 520, tam=34, c="gris", w="sb", en="b", t=0.3), sec="6. Un ejemplo resuelto")
v.b("Ridge, en este caso simple, divide cada coeficiente entre uno más alfa. [[a]]Es decir, entre dos. [[b]]Quedan dos coma cinco, menos cero coma siete, y cero coma tres. Ninguno se anula.",
    T("Ridge:  β / (1 + α)", 1500, 640, tam=46, c="azul", en="a"), *[T(val, 1250, 400 + 130 * i, tam=60, c="azul", en="b", t=0.2 * i) for i, val in enumerate(("2.50", "−0.70", "0.30"))],
    T("Ridge", 1250, 280, tam=38, c="azul", w="sb", en="a"))
v.b("Lasso, en cambio, le resta alfa a cada valor absoluto, y no baja de cero. [[a]]Cinco menos uno: cuatro. [[b]]Uno coma cuatro menos uno: cero coma cuatro, con su signo negativo. [[c]]Y cero coma seis menos uno daría menos cero coma cuatro, así que queda en cero.",
    T("Lasso", 1620, 280, tam=38, c="ambar", w="sb"), T("máx(|β| − α, 0)", 1500, 760, tam=46, c="ambar"),
    T("4.0", 1620, 400, tam=60, c="ambar", en="a"), T("−0.4", 1620, 530, tam=60, c="ambar", en="b"), T("0", 1620, 660, tam=60, c="rojo", en="c"), P("anulado", 1620, 740, "rojo", 34, en="c", t=0.2))
v.b("Mira la suma de los valores absolutos. [[a]]Sin penalizar: siete. Con Ridge: tres coma cinco. [[b]]Con Lasso: cuatro coma cuatro, y una variable menos.",
    CL(), T("suma de |coeficientes|", 960, 260, tam=46, c="gris", w="sb"),
    *[[Bc(x, 560, 420, 340, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 440, tam=44, c=c, en=k, t=0.1), N(val, x, 580, "{:.1f}", 110, c, en=k, t=0.2, dur=0.9)]
      for x, c, n, val, k in ((440, "gris", "sin penalizar", 7.0, "a"), (960, "azul", "Ridge", 3.5, "a"), (1480, "ambar", "Lasso", 4.4, "b"))],
    P("una variable anulada", 1480, 780, "rojo", 38, en="b", t=0.6))
v.b("Piensa un segundo: [[a]]si el coeficiente de x tres fuera cero coma seis, y alfa fuera uno, ¿en cuánto lo deja Lasso?",
    CL(), T("β = 0.6   ·   α = 1", 960, 400, tam=80, c="tinta", en="a"), T("Lasso lo deja en…", 960, 600, tam=72, c="ambar", en="a", t=0.4), pausa=0.6)
v.b("En cero. [[a]]Porque su valor absoluto, cero coma seis, es menor que alfa. Lasso solo conserva lo que supera la penalización.",
    CL(), N(0, 960, 420, "{:.0f}", 250, "rojo", dur=0.3), P("|0.6| < 1  →  se anula", 960, 740, "rojo", 56, en="a"))

# ---- 7. Club Activo
VARS = ["antigüedad", "uso", "edad", "quejas", "ruido 1", "ruido 2"]
RC = D["reg_coef"]
SC, BASE = 26, 700
XV = [360 + 250 * i for i in range(6)]
MET = [("OLS", "#94A3B8", -62), ("Ridge5", "azul", 0), ("Lasso05", "ambar", 62)]
v.b("Ahora, el caso de la clase. [[a]]Se predice el gasto con seis variables. Dos son reales, antigüedad y uso. [[b]]Dos aportan poco, edad y quejas. [[c]]Y dos son puro ruido, que la profesora agregó a propósito.",
    CL(),
    *[[Bc(XV[i], 520, 220, 250, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, XV[i], 520, tam=36, c=c, mw=190, en=k, t=0.1)]
      for i, (n, c, k) in enumerate(zip(VARS, ["azul", "azul", "gris", "gris", "rojo", "rojo"], ["a", "a", "b", "b", "c", "c"]))],
    sec="7. Ridge y Lasso en Club Activo")


def barra(m, i, en=None, t=0.0):
    nom, col, dx = MET[m]
    val = RC[nom][i]
    x = XV[i] + dx
    h = max(abs(val) * SC, 4)
    y0, y1 = (BASE - h, BASE) if val >= 0 else (BASE, BASE + h)
    return Bx(x - 24, y0, x + 24, y1, col, None, 0, 4, en=en, t=t, ent="subir")


v.b("Primero, la regresión sin penalizar, en gris. [[a]]Los dos reales tienen coeficientes grandes. [[b]]Pero incluso el ruido recibe un coeficiente, aunque pequeño.",
    CL(), L(250, BASE, 1700, BASE, "tinta", 4, t=0.1), *[T(n, XV[i], 770, tam=30, c="tinta", w="sb", t=0.1) for i, n in enumerate(VARS)],
    [barra(0, i, t=0.2 + 0.05 * i) for i in range(6)], P("sin penalizar", 1560, 300, "gris", 38, en="a"),
    R(XV[4] - 30, BASE, 520, 120, "rojo", 30, id="rrn", en="b", ent="fade"), T("coeficiente de un ruido", 1300, 880, tam=32, c="rojo", w="sb", en="b"), sec=None)
v.b("Ahora agrega Ridge, con alfa cinco. [[a]]Casi no cambia nada: apenas encoge los coeficientes. El ruido sigue ahí.",
    X("rrn", dur=0.3), [barra(1, i, en="a", t=0.05 * i) for i in range(6)], P("Ridge · α = 5", 1560, 380, "azul", 38, en="a"))
v.b("Y ahora Lasso, con alfa cero coma cinco. [[a]]Anula las quejas y los dos ruidos. [[b]]Y deja la edad en cero coma veintidós.",
    [barra(2, i, en="a", t=0.05 * i) for i in range(6)], P("Lasso · α = 0.5", 1560, 460, "ambar", 38, en="a"),
    *[T("0", XV[i] + 62, BASE - 26, tam=34, c="rojo", en="a", t=0.3) for i in (3, 4, 5)], T("0.22", XV[2] + 62, BASE - 40, tam=30, c="ambar", w="sb", en="b"))
v.b("Lasso hizo una selección automática: dejó lo importante, y descartó el ruido. [[a]]Y el R cuadrado de prueba no empeoró: pasó de cero coma quinientos cuarenta y cuatro, a cero coma quinientos cincuenta y dos con Lasso.",
    CL(), P("Lasso descartó el ruido", 960, 330, "ambar", 60, ent="subir", t=0.2), T("R² en prueba", 960, 520, tam=40, c="gris", w="sb", en="a"),
    T("0.544", 600, 680, tam=110, c="gris", en="a"), L(780, 680, 1000, 680, "gris", 8, True, en="a", t=0.3), T("0.552", 1260, 680, tam=110, c="ambar", en="a", t=0.4), T("sin penalizar  →  Lasso 0.5", 960, 820, tam=36, c="gris", w="sb", en="a", t=0.5))
v.b("Cuidado con alfa. [[a]]Si lo subes demasiado, los dos métodos encogen tanto los coeficientes, [[b]]que el modelo termina prediciendo siempre la media. El R cuadrado se acerca a cero.",
    CL(), P("alfa demasiado grande", 960, 270, "rojo", 60, ent="subir", t=0.2),
    G(300, 340, 1100, 820, 0, 10, 0, 0.7, [dict(xs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ys=[0.55, 0.55, 0.54, 0.53, 0.5, 0.42, 0.3, 0.18, 0.08, 0.03, 0.0], color="violeta", grosor=7, revela=0.0, lapiz=False, suave=True)],
      xticks=[], yticks=[0, 0.5], yfmt="{:g}", xlabel="alfa  →  cada vez más grande", id="gal", en="a", t=0.1),
    U("gal", **{"series.0.revela": 1.0}, en="a", t=0.4, dur=2.0), T("R² de prueba", 300, 300, tam=34, c="tinta", w="sb", anc="lm", en="a"),
    P("tiende a predecir la media", 1560, 700, "rojo", 40, en="b", mw=560), T("forma ilustrativa de la curva", 1560, 800, tam=30, c="gris", w="sb", en="b", t=0.4))
v.b("Y un recordatorio de la clase treinta. [[a]]Ridge y Lasso penalizan por tamaño, así que hay que estandarizar antes. Y alfa se elige con la validación, nunca con la prueba. [[b]]Dentro de un Pipeline, para no filtrar información.",
    CL(), P("estandarizar antes de penalizar", 960, 330, "teal", 56, en="a"), P("alfa se elige con la validación", 960, 520, "azul", 56, en="a", t=0.5), P("todo dentro de un Pipeline", 960, 710, "violeta", 56, en="b"))

# ---- 8. Cierre
v.b("Para llevarte: [[a]]subajuste es fallar incluso en entrenamiento. Sobreajuste es brillar en entrenamiento y caer en prueba. [[b]]Ridge encoge, y Lasso puede anular. [[c]]Y la brecha entre entrenamiento y prueba te avisa.",
    CL(), P("subajuste: falla en entrenamiento", 960, 260, "rojo", 48, en="a"), P("sobreajuste: cae en prueba", 960, 400, "ambar", 48, en="a", t=0.4),
    P("Ridge encoge · Lasso puede anular", 960, 590, "azul", 48, en="b"), P("la brecha te avisa", 960, 770, "violeta", 56, en="c"), sec="8. Para llevar")
v.b("Vamos con la sexta parada: [[a]]la trampa del accuracy.",
    CL(), Bc(960, 480, 620, 260, "rojoclaro", "rojo", 6, 40, id="p1", ent="pop", en="a"),
    T("6", 720, 480, tam=120, c="rojo", en="a", t=0.1), T("La trampa del accuracy", 1040, 480, tam=44, c="tinta", mw=420, en="a", t=0.15), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
