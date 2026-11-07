import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_01_introduccion.json"
v = V("Introducción: de qué trata Machine Learning")

v.portada("Introducción al Machine Learning", "Qué problema tengo, con qué datos y cómo sabré que el modelo sirve",
          "Hola. Bienvenido a la serie de videos de la clase treinta y uno. Aquí vas a ver, paso a paso, qué es el {Machine Learning|mashín lérnin} y cómo se usa con criterio.")

# ---- 1. La pregunta guía
v.b("Esta clase gira en torno a tres preguntas. [[a]]Primera: ¿qué tipo de problema tengo? [[b]]Segunda: ¿qué datos tengo? [[c]]Y tercera: ¿cómo sabré que el modelo sirve?",
    CL(),
    [[Bc(x, 520, 520, 420, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(n, x, 360, tam=110, c=c, en=k, t=0.1),
      T(tx, x, 540, tam=46, c="tinta", mw=420, en=k, t=0.2)]
     for x, c, n, tx, k in ((360, "violeta", "1", "¿Qué tipo de problema es?", "a"), (960, "teal", "2", "¿Qué datos tengo?", "b"),
                            (1560, "ambar", "3", "¿Cómo sabré que el modelo sirve?", "c"))],
    sec="1. Las tres preguntas")
v.b("Si respondes bien esas tres, ya tienes la mitad del trabajo hecho. [[a]]La otra mitad es elegir una herramienta, y eso casi siempre es lo más fácil.",
    R(960, 520, 1700, 480, "violeta", 40, id="rq", ent="fade", t=0.2), X("rq", en="a", dur=0.3),
    P("la herramienta viene después", 960, 900, "azul", 54, en="a"))

# ---- 2. La gran idea
v.b("La gran idea de la clase es simple. [[a]]Antes, un programador escribía a mano las reglas de decisión. [[b]]Ahora le entregas ejemplos a un modelo, [[c]]y es el modelo quien ajusta sus parámetros para encontrar la regla.",
    CL(),
    Bc(500, 480, 640, 460, "blanco", "grisclaro", 5, 30, sombra=True, id="b1", ent="subir", en="a"), T("Antes", 500, 300, tam=52, c="gris", en="a", t=0.1),
    T("si quejas > 2", 500, 430, tam=44, c="tinta", w="sb", en="a", t=0.3), T("entonces cancela", 500, 500, tam=44, c="tinta", w="sb", en="a", t=0.45),
    P("reglas escritas a mano", 500, 640, "gris", 38, en="a", t=0.6),
    Bc(1400, 480, 640, 460, "blanco", "violeta", 5, 30, sombra=True, id="b2", ent="subir", en="b"), T("Ahora", 1400, 300, tam=52, c="violeta", en="b", t=0.1),
    Pt([(30, [1160, 370, 1640, 560], "#94A3B8"), (8, [1160, 590, 1640, 650], "rojo")], 9, id="pej", en="b", t=0.3),
    P("ejemplos", 1400, 700, "violeta", 38, en="b", t=0.6),
    L(830, 480, 1070, 480, "violeta", 8, True, en="c"), P("el modelo ajusta parámetros", 960, 860, "teal", 50, en="c", t=0.3))
v.b("Y esos ejemplos pueden venir de tres formas. [[a]]Con la respuesta correcta: es aprendizaje supervisado. [[b]]Sin respuesta: es no supervisado. [[c]]O con un entorno que da premios y castigos: es aprendizaje por refuerzo.",
    CL(),
    [[Bc(x, 500, 520, 400, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(n, x, 380, tam=50, c=c, en=k, t=0.1), T(d, x, 530, tam=38, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "violeta", "Supervisado", "ejemplos con la respuesta correcta", "a"), (960, "teal", "No supervisado", "ejemplos sin respuesta", "b"),
                           (1560, "ambar", "Refuerzo", "entorno con recompensas", "c"))])
v.b("Y el ciclo siempre es el mismo: [[a]]datos, [[b]]modelo, [[c]]evaluar el error, [[d]]y mejorar. Y vuelta a empezar.",
    CL(),
    *[[Bc(x, 480, 340, 190, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 480, tam=50, c=c, en=k, t=0.1)]
      for x, c, n, k in ((330, "teal", "datos", "a"), (780, "violeta", "modelo", "b"), (1230, "rojo", "error", "c"), (1680, "verde", "mejorar", "d"))],
    [L(510 + 450 * i, 480, 600 + 450 * i, 480, "gris", 6, True, en=k, t=0.2) for i, k in enumerate("bcd")],
    L(1680, 620, 330, 620, "grisclaro", 6, True, en="d", t=0.5, dur=1.0), P("otra vuelta", 1000, 700, "gris", 40, en="d", t=1.0))

# ---- 3. Ferrari
v.b("Antes de seguir, una advertencia de la profesora. [[a]]No siempre se necesita inteligencia artificial. [[b]]Hay gente que quiere un Ferrari para ir a comprar pan a la esquina.",
    CL(),
    P("no siempre se necesita IA", 960, 330, "ambar", 66, en="a"),
    Bc(520, 640, 560, 280, "blanco", "rojo", 5, 30, sombra=True, id="fe", en="b", ent="subir"), T("un Ferrari", 520, 640, tam=64, c="rojo", en="b", t=0.1),
    L(820, 640, 1060, 640, "gris", 8, True, en="b", t=0.3), Bc(1400, 640, 560, 280, "blanco", "verde", 5, 30, sombra=True, en="b", t=0.4, ent="subir"),
    T("pan en la esquina", 1400, 640, tam=54, c="verde", en="b", t=0.5), sec="2. Una advertencia")
v.b("Si una resta o una tabla de reglas ya resuelve el problema, [[a]]no hace falta Machine Learning. Piensa un segundo: ¿cuándo sí vale la pena usarlo?",
    CL(),
    T("¿cuándo sí vale la pena?", 960, 460, tam=76, c="violeta", ent="subir", t=0.3), pausa=0.6)
v.b("Cuando las reglas son difíciles de escribir, [[a]]cambian con el tiempo, [[b]]o dependen de muchas variables a la vez. En esos casos, los ejemplos hablan mejor que las reglas.",
    P("reglas difíciles de escribir", 960, 640, "teal", 54, t=0.1), P("cambian con el tiempo", 960, 770, "azul", 54, en="a"),
    P("muchas variables a la vez", 960, 900, "violeta", 54, en="b"))

# ---- 4. El mapa
ETQ = [("Reglas fijas y ciclo del ML", "teal"), ("Clasificación: regresión logística", "violeta"), ("Regresión: predecir un número", "azul"),
       ("Sin etiquetas: K-Means y PCA", "verde"), ("Subajuste y regularización", "ambar"), ("La trampa del accuracy", "rojo"),
       ("Elegir el enfoque", "teal")]
XS = [270, 730, 1190, 1650]


def pos(i):
    return XS[i % 4], 360 if i < 4 else 660


def tarjeta(i, t):
    x, y = pos(i)
    c = ETQ[i][1]
    return [Bc(x, y, 410, 190, "blanco", c, 5, 28, sombra=True, id=f"m{i}", en="a", t=t),
            T(str(i + 1), x - 150, y, tam=84, c=c, id=f"mn{i}", en="a", t=t + 0.05),
            T(ETQ[i][0], x + 35, y, tam=31, c="tinta", mw=270, id=f"ml{i}", en="a", t=t + 0.1)]


v.b("Este es el mapa de la clase. [[a]]Son siete paradas, y cada una tiene su propio video. Al final, un repaso.",
    CL(), [tarjeta(i, 0.17 * i) for i in range(7)],
    [L(pos(i)[0] + 208, pos(i)[1], pos(i)[0] + 252, pos(i)[1], "grisclaro", 5, True, id=f"mf{i}", en="a", t=0.15 * i + 0.9, dur=0.4) for i in (0, 1, 2, 4, 5)],
    sec="3. El mapa de la clase")
v.b("Primero, [[a]]cómo pasar de un problema real a una tarea de Machine Learning. [[b]]Después, los dos grandes modelos supervisados: clasificación y regresión.",
    R(*pos(0), 470, 245, "ambar", 40, id="rA", en="a"), X("rA", en="b", dur=0.3),
    R(*pos(1), 470, 245, "ambar", 40, id="rB", en="b"), X("rB", en="b", t=2.2, dur=0.3), R(*pos(2), 470, 245, "ambar", 40, id="rC", en="b", t=2.2))
v.b("Luego, [[a]]qué se hace cuando no hay etiquetas. [[b]]Cómo saber si un modelo aprendió demasiado poco, o demasiado. [[c]]Y por qué el accuracy puede engañarte.",
    X("rC", dur=0.3), R(*pos(3), 470, 245, "ambar", 40, id="rD", en="a"), X("rD", en="b", dur=0.3), R(*pos(4), 470, 245, "ambar", 40, id="rE", en="b"),
    X("rE", en="c", dur=0.3), R(*pos(5), 470, 245, "ambar", 40, id="rF", en="c"))
v.b("Para cerrar, [[a]]cómo elegir el enfoque según el problema, y cuándo conviene Deep Learning.",
    X("rF", dur=0.3), R(*pos(6), 470, 245, "ambar", 40, id="rG", en="a"))

# ---- 5. Lo que necesitas antes
v.b("Esta clase usa piezas que ya conoces. [[a]]La regresión lineal y la logística, de las clases diecisiete a veintiuna, [[b]]y todo lo de la clase treinta: dividir los datos, el Pipeline, Ridge, Lasso y las métricas.",
    CL(),
    Bc(520, 440, 700, 260, "blanco", "azul", 5, 30, sombra=True, ent="subir", en="a"), T("Clases 17 a 21", 520, 400, tam=50, c="azul", en="a", t=0.1), T("regresión lineal y logística", 520, 480, tam=36, c="gris", w="sb", en="a", t=0.2),
    Bc(1400, 440, 700, 260, "blanco", "violeta", 5, 30, sombra=True, ent="subir", en="b"), T("Clase 30", 1400, 400, tam=50, c="violeta", en="b", t=0.1), T("división, Pipeline, Ridge, Lasso, métricas", 1400, 485, tam=32, c="gris", w="sb", mw=600, en="b", t=0.2),
    P("aquí las verás como modelos de ML", 960, 800, "teal", 50, en="b", t=0.6), sec="4. Lo que necesitas antes")
v.b("Y lo que sigue: [[a]]la clase treinta y dos explica con calma la reducción de dimensionalidad. Aquí solo usaremos {PCA|pe ce a} para mirar los datos.",
    Bc(960, 650, 1500, 130, "tealclaro", "teal", 4, 26, ent="subir", en="a"), T("C30  →  C31  →  C32", 960, 650, tam=60, c="teal", en="a", t=0.2))

# ---- 6. Los datos
v.b("Para practicar, la profesora usó dos conjuntos de datos. [[a]]Primero, Club Activo: ochocientos clientes de un gimnasio, simulados dentro del propio cuaderno. [[b]]Un veinte coma nueve por ciento canceló.",
    CL(),
    Pt([(633, [160, 280, 1760, 760], "#94A3B8"), (167, [160, 800, 1760, 900], "rojo")], 8, visibles=800, id="pc", en="a", dur=2.0),
    N(800, 960, 190, "{:.0f}", 70, "tinta", suf=" clientes", en="a", dur=2.0), P("20.9 % cancela", 960, 990, "rojo", 46, en="b"),
    sec="5. Los datos de la clase")
v.b("Segundo, una base real de tres mil ciento cincuenta clientes, de la actividad de repaso. [[a]]Aquí la fuga es de quince coma siete por ciento. Sirve para el desafío de decidir si un problema es supervisado o no.",
    CL(),
    Pt([(2655, [160, 280, 1560, 880], "#94A3B8"), (495, [1620, 280, 1760, 880], "rojo")], 5, id="pr", t=0.2),
    N(3150, 960, 190, "{:.0f}", 70, "tinta", suf=" clientes", t=0.2, dur=1.6), P("15.7 % de fuga", 960, 970, "rojo", 46, en="a"))
v.b("Fíjate en algo. En los dos casos, el cliente que se va es la minoría. [[a]]Eso va a complicar la forma de medir un modelo, y lo veremos con calma.",
    R(1690, 580, 200, 620, "rojo", 40, id="rr", ent="fade", t=0.2), X("rr", en="a", dur=0.3))

# ---- 7. Cómo estudiar
v.b("Cada video sigue la misma receta. Primero, [[a]]una pregunta. Después, [[b]]una analogía. Luego, [[c]]un ejemplo con números. Y al final, [[d]]el error típico y una frase para llevarte.",
    CL(),
    *[[Bc(x, 480, 390, 300, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 380, tam=90, c=c, en=k, t=0.1), T(tx, x, 520, tam=38, c="tinta", mw=320, en=k, t=0.2)]
      for x, c, n, tx, k in ((330, "violeta", "1", "una pregunta", "a"), (780, "azul", "2", "una analogía", "b"),
                              (1230, "teal", "3", "un ejemplo con números", "c"), (1680, "ambar", "4", "el error típico y una frase para llevarte", "d"))],
    sec="6. Cómo estudiar con estos videos")
v.b("Mi consejo: míralos con la guía al lado, [[a]]pausa cuando algo no quede claro, [[b]]y cuando te haga una pregunta, [[c]]piensa tu respuesta antes de seguir.",
    CL(),
    P("con la guía al lado", 960, 330, "teal", 56, ent="subir", t=0.3), P("pausa cuando algo no quede claro", 960, 500, "azul", 56, en="a"),
    P("piensa tu respuesta antes de seguir", 960, 670, "violeta", 56, en="c"))
v.b("Probemos una. [[a]]Un modelo acierta el setenta y tres por ciento de las veces. Otro, que siempre responde «nadie cancela», [[b]]acierta el setenta y nueve. Piensa un segundo: ¿cuál es mejor?",
    CL(),
    Bc(560, 480, 560, 360, "blanco", "violeta", 5, 30, sombra=True, id="m1", ent="subir", t=0.2), T("el modelo", 560, 340, tam=44, c="violeta", t=0.3),
    N(73.3, 560, 480, "{:.1f}", 130, "violeta", suf=" %", dur=1.0, t=0.4, id="n1"),
    Bc(1360, 480, 560, 360, "blanco", "ambar", 5, 30, sombra=True, en="b", ent="subir", id="m2"), T("«nadie cancela»", 1360, 340, tam=44, c="ambar", en="b", t=0.1),
    N(79.2, 1360, 480, "{:.1f}", 130, "ambar", suf=" %", dur=1.0, en="b", t=0.3, id="n2"), pausa=0.6)
v.b("Todavía no lo sabemos. [[a]]El que siempre dice «nadie cancela» gana en accuracy, pero no detecta a ningún cliente que se va. [[b]]Por eso el accuracy no basta, y de eso trata uno de los videos.",
    K(1360, 640, "cruz", 90, c="rojo", en="a", id="x1"), P("0 clientes detectados", 1360, 740, "rojo", 40, en="a", t=0.4),
    P("accuracy no basta", 960, 900, "teal", 56, en="b"))
v.b("Vamos con la primera parada: [[a]]cómo pasar de un problema a una tarea de Machine Learning.",
    CL(),
    Bc(960, 480, 620, 260, "tealclaro", "teal", 6, 40, id="p1", ent="pop", en="a"),
    T("1", 740, 480, tam=120, c="teal", en="a", t=0.1), T("Reglas fijas y ciclo del ML", 1040, 480, tam=46, c="tinta", mw=400, en="a", t=0.15),
    sec="7. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
