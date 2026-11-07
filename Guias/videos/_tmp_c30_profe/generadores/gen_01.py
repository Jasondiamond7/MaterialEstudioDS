import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_01_introduccion.json"
v = V("Introducción: qué necesita un modelo de tus datos")

v.portada("Qué necesita un modelo de tus datos", "La pregunta de la clase y el mapa de este recorrido",
          "Hola. Bienvenido a la serie de videos de la clase treinta. Aquí vas a ver, paso a paso, cómo se preparan los datos antes de entrenar un modelo.")

# ---- 1. La pregunta
v.b("Todo en Machine Learning parte de una pregunta: [[a]]¿cómo sé que mi modelo aprendió, [[b]]y no solo memorizó?",
    CL(),
    T("¿Cómo sé que mi modelo aprendió…", 960, 330, tam=70, c="tinta", id="q1", en="a"),
    T("…y no solo memorizó?", 960, 430, tam=70, c="violeta", id="q2", en="b"),
    P("aprendió", 560, 700, "verde", 64, en="b", t=0.5), K(560, 830, "check", 110, en="b", t=0.9),
    P("memorizó", 1360, 700, "rojo", 64, en="b", t=0.7), K(1360, 830, "cruz", 110, en="b", t=1.2),
    sec="1. La pregunta")
v.b("Un modelo no se juzga por lo bien que responde a lo que ya vio. [[a]]Se juzga por cómo se porta con datos nuevos.",
    CL(),
    Bc(560, 520, 620, 420, "blanco", "grisclaro", 4, 28, sombra=True, id="v1", ent="subir", t=0.2),
    T("datos que ya vio", 560, 380, tam=44, c="gris", ent="subir", t=0.3),
    N(0.99, 560, 520, "{:.2f}", 150, "gris", dur=1.0, t=0.4),
    T("no prueba nada", 560, 640, tam=38, c="gris", w="sb", ent="subir", t=1.2),
    Bc(1360, 520, 620, 420, "azulclaro", "azul", 5, 28, sombra=True, id="v2", ent="subir", en="a"),
    T("datos nuevos", 1360, 380, tam=44, c="azul", en="a", t=0.1),
    T("?", 1360, 530, tam=190, c="azul", en="a", t=0.25),
    T("aquí se decide", 1360, 650, tam=38, c="azul", w="sb", ent="subir", en="a", t=0.5))
v.b("Para poder responder, hay reglas del juego que se aplican [[a]]antes de elegir cualquier algoritmo. [[b]]Y de esas reglas trata toda esta clase.",
    CL(),
    Bc(520, 520, 560, 220, "tealclaro", "teal", 5, 36, id="rg", t=0.2),
    T("reglas del juego", 520, 520, tam=56, c="teal", t=0.35),
    L(820, 520, 1060, 520, "gris", 8, True, en="a"),
    Bc(1360, 520, 560, 220, "violetaclaro", "violeta", 5, 36, id="alg", en="a", t=0.3),
    T("algoritmo", 1360, 520, tam=56, c="violeta", en="a", t=0.45),
    T("antes", 940, 440, tam=38, c="gris", w="sb", en="a", t=0.3),
    P("preparar bien los datos", 960, 820, "ambar", 52, en="b"))

# ---- 2. El mapa
ETQ = [("Tipos de variables", "teal"), ("Calidad e imputación", "teal"), ("Entrenar, validar, probar", "violeta"),
       ("Ridge y Lasso", "azul"), ("Validación cruzada K-Fold", "violeta"), ("Muestreo estratificado", "violeta"),
       ("Matriz de confusión y métricas", "rojo"), ("Fuga de datos y Pipeline", "ambar")]
XS = [270, 730, 1190, 1650]


def pos(i):
    return XS[i % 4], 360 if i < 4 else 660


def tarjeta(i, t):
    x, y = pos(i)
    c = ETQ[i][1]
    return [Bc(x, y, 410, 190, "blanco", c, 5, 28, sombra=True, id=f"m{i}", en="a", t=t),
            T(str(i + 1), x - 150, y, tam=84, c=c, id=f"mn{i}", en="a", t=t + 0.05),
            T(ETQ[i][0], x + 35, y, tam=33, c="tinta", mw=270, id=f"ml{i}", en="a", t=t + 0.1)]


flechas = [L(pos(i)[0] + 208, pos(i)[1], pos(i)[0] + 252, pos(i)[1], "grisclaro", 5, True, id=f"mf{i}", en="a",
             t=0.15 * i + 0.9, dur=0.4) for i in (0, 1, 2, 4, 5, 6)]
v.b("Este es el mapa de la clase. [[a]]Son ocho paradas, y cada una tiene su propio video.",
    CL(), [tarjeta(i, 0.17 * i) for i in range(8)], flechas, sec="2. El mapa de la clase")
v.b("Empezamos por conocer el material: [[a]]el tipo de cada variable, [[b]]y la calidad de los datos.",
    R(*pos(0), 470, 245, "ambar", 40, id="rA", en="a"),
    X("rA", en="b", dur=0.3), R(*pos(1), 470, 245, "ambar", 40, id="rB", en="b"))
v.b("Después, [[a]]cómo dividir los datos en entrenamiento, validación y prueba, y cómo descubrir el sobreajuste.",
    X("rB", dur=0.3), R(*pos(2), 470, 245, "ambar", 40, id="rC", en="a"))
v.b("Luego, [[a]]Ridge y Lasso, que castigan a los modelos demasiado complejos. [[b]]Y la validación cruzada y la estratificación, para validar con más estabilidad.",
    X("rC", dur=0.3), R(*pos(3), 470, 245, "ambar", 40, id="rD", en="a"),
    X("rD", en="b", dur=0.3), R(*pos(4), 470, 245, "ambar", 40, id="rE", en="b"),
    X("rE", en="b", t=2.3, dur=0.3), R(*pos(5), 470, 245, "ambar", 40, id="rF", en="b", t=2.3))
v.b("Y al final, [[a]]las métricas para medir un modelo, [[b]]y la fuga de datos: el error más común, y cómo evitarlo con un Pipeline.",
    X("rF", dur=0.3), R(*pos(6), 470, 245, "ambar", 40, id="rG", en="a"),
    X("rG", en="b", dur=0.3), R(*pos(7), 470, 245, "ambar", 40, id="rH", en="b"))

# ---- 3. Los datos
v.b("Para practicar, la profesora usó dos conjuntos de datos. Los dos son simulados, y [[a]]se generan dentro del propio cuaderno: no hay archivos que descargar.",
    CL(),
    Bc(520, 470, 700, 400, "blanco", "teal", 5, 30, sombra=True, id="d1", ent="subir", t=0.2),
    T("Datos 1", 520, 330, tam=48, c="teal", t=0.3), T("viviendas", 520, 440, tam=64, c="tinta", t=0.4),
    T("para predecir un precio", 520, 540, tam=36, c="gris", w="sb", ent="subir", t=0.5),
    Bc(1400, 470, 700, 400, "blanco", "rojo", 5, 30, sombra=True, id="d2", ent="subir", t=0.5),
    T("Datos 2", 1400, 330, tam=48, c="rojo", t=0.6), T("fraude", 1400, 440, tam=64, c="tinta", t=0.7),
    T("para detectar casos raros", 1400, 540, tam=36, c="gris", w="sb", ent="subir", t=0.8),
    P("sin archivos que descargar", 960, 850, "teal", 46, en="a"), sec="3. Los datos de la clase")
v.b("El primero tiene [[a]]trescientas viviendas, [[b]]cada una con su precio en pesos chilenos.",
    CL(),
    Pt([(300, [160, 280, 1760, 700], "#94A3B8")], 12, visibles=300, id="pv", en="a", dur=2.0),
    N(300, 960, 190, "{:.0f}", 70, "tinta", suf=" viviendas", en="a", dur=2.0),
    P("precio en pesos chilenos (CLP)", 960, 830, "teal", 46, en="b"))
v.b("El segundo tiene [[a]]dos mil transacciones, y [[b]]solo el cinco por ciento son fraudes.",
    CL(),
    Pt([(1891, [160, 280, 1540, 700], "#94A3B8"), (109, [1640, 280, 1760, 700], "rojo")], 6, id="pf", en="a"),
    N(2000, 960, 190, "{:,.0f}".replace(",", ""), 70, "tinta", suf=" transacciones", en="a", dur=1.6),
    P("≈ 5 % son fraudes", 960, 830, "rojo", 50, en="b"))
v.b("Fíjate en esa diferencia: [[a]]cuando algo es raro, medir bien se vuelve más difícil. [[b]]Lo vamos a ver con calma.",
    R(1700, 490, 200, 480, "rojo", 40, en="a", id="rf", ent="fade"), X("rf", en="b", dur=0.3))

# ---- 4. Cómo usar los videos
v.b("Cada video sigue la misma receta. Primero, [[a]]una pregunta. Después, [[b]]una analogía. Luego, [[c]]un ejemplo con números. Y al final, [[d]]el error típico y una frase para llevarte.",
    CL(),
    *[[Bc(x, 480, 390, 300, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"),
       T(n, x, 380, tam=90, c=c, en=k, t=0.1), T(tx, x, 520, tam=38, c="tinta", mw=320, en=k, t=0.2)]
      for x, c, n, tx, k in ((330, "violeta", "1", "una pregunta", "a"), (780, "azul", "2", "una analogía", "b"),
                              (1230, "teal", "3", "un ejemplo con números", "c"), (1680, "ambar", "4", "el error típico y una frase para llevarte", "d"))],
    sec="4. Cómo estudiar con estos videos")
v.b("Mi consejo: míralos con la guía al lado, [[a]]pausa cuando algo no quede claro, [[b]]y cuando te haga una pregunta, [[c]]piensa tu respuesta antes de seguir.",
    CL(),
    P("con la guía al lado", 960, 330, "teal", 56, ent="subir", t=0.3),
    P("pausa cuando algo no quede claro", 960, 500, "azul", 56, en="a"),
    P("piensa tu respuesta antes de seguir", 960, 670, "violeta", 56, en="c"))
v.b("Probemos una. [[a]]Un modelo saca cero coma noventa y nueve con los datos con que aprendió. Piensa un segundo: ¿ya sabemos que es bueno?",
    CL(),
    Bc(960, 480, 460, 300, "blanco", "violeta", 5, 30, sombra=True, id="mm", ent="subir", t=0.2),
    N(0.99, 960, 480, "{:.2f}", 170, "violeta", dur=1.1, t=0.3, id="nn"),
    T("con los datos que ya vio", 960, 700, tam=38, c="gris", w="sb", en="a", t=0.5),
    T("¿bueno?", 1500, 480, tam=90, c="ambar", en="a", t=0.8), pausa=0.6)
v.b("Todavía no. Solo sabemos que se aprendió esos datos. [[a]]Hay que medirlo con datos que no vio, y de eso se trata esta clase.",
    U("nn", color="gris", dur=0.4),
    K(960, 480, "cruz", 260, c="rojo", en="a", id="xx"),
    P("medir con datos nuevos", 960, 850, "teal", 56, en="a", t=0.4))
v.b("Vamos con la primera parada: [[a]]los tipos de variables.",
    CL(),
    Bc(960, 480, 620, 260, "tealclaro", "teal", 6, 40, id="p1", ent="pop", en="a"),
    T("1", 740, 480, tam=120, c="teal", en="a", t=0.1), T("Tipos de variables", 1040, 480, tam=52, c="tinta", mw=400, en="a", t=0.15),
    sec="5. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
