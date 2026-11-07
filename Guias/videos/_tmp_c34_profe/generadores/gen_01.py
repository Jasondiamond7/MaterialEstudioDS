import random
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, arbol, mil, pct, NODOS

OUT = out("guion_c34_01_introduccion.json")
v = V("Introducción: un modelo que se puede dibujar")
H10 = NODOS[10]

v.portada("Un modelo que se puede dibujar", "Cómo parte el espacio un árbol de decisión, y por qué es simple y peligroso a la vez",
          "Hola. Bienvenido a la serie de videos de la clase treinta y cuatro, con el profesor Eduardo Rodríguez. Hoy conoces el modelo más fácil de explicar de todo el diplomado: el árbol de decisión.")

# ---- 1. Hook
v.b("El profesor lo presentó con una frase que conviene recordar. [[a]]El árbol de decisión es la técnica más simple de todas, [[b]]y a la vez una de las más peligrosas. [[c]]Simple, porque se dibuja y se explica a cualquiera. [[d]]Peligrosa, porque sobreajusta muchísimo.",
    CL(), T("«la técnica más simple de todas…", 960, 300, tam=64, c="tinta", en="a"),
    T("…y a la vez una de las más peligrosas»", 960, 410, tam=64, c="rojo", en="b"),
    tarjetas([(560, "verde", "simple", "se dibuja · se explica a cualquiera", "c"), (1360, "rojo", "peligrosa", "sobreajusta muchísimo", "d")], y=700, w=640, h=300),
    sec="1. Simple y peligroso")
v.b("Peligrosa en serio. [[a]]Un árbol sobreajustado puede funcionar uno o dos meses en producción, [[b]]y fallar al cuarto mes. [[c]]Por eso, dice el profesor, separar entrenamiento y prueba aquí es obligatorio.",
    CL(), T("un árbol sobreajustado en producción", 960, 200, tam=46, c="gris", w="sb", ent="subir"),
    *[[Bc(420 + 360 * i, 480, 300, 200, "verdeclaro" if i < 2 else ("blanco" if i == 2 else "rojoclaro"), "verde" if i < 2 else ("linea" if i == 2 else "rojo"), 4, 24,
          en=("a" if i < 2 else "b"), t=0.3 * (i % 2), ent="pop"),
       T(f"mes {i + 1}", 420 + 360 * i, 450, tam=40, c="tinta", en=("a" if i < 2 else "b"), t=0.3 * (i % 2) + 0.1),
       K(420 + 360 * i, 530, "check" if i < 2 else "cruz", 50, en=("a" if i < 2 else "b"), t=0.3 * (i % 2) + 0.3) if i != 2 else None] for i in range(4)],
    P("entrenamiento y prueba separados: obligatorio", 960, 760, "violeta", 48, en="c"))

# ---- 2. Dónde estamos
v.b("¿Dónde estamos en el diplomado? [[a]]La clase treinta preguntó si los datos eran confiables. [[b]]La treinta y uno, cómo aprende un modelo. [[c]]La treinta y dos, en qué forma se los entregamos. [[d]]La treinta y tres, con cuáles variables me quedo. [[e]]Y hoy, la treinta y cuatro pregunta: ¿cómo lo explico?",
    CL(), *[[Bc(230 + 365 * i, 520, 330, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"),
             T(f"Clase {n}", 230 + 365 * i, 410, tam=44, c=c, en=k, t=0.1),
             T(q, 230 + 365 * i, 560, tam=32, c="tinta", w="sb", mw=270, en=k, t=0.2)]
            for i, (n, c, q, k) in enumerate(((30, "gris", "¿son datos confiables?", "a"), (31, "verde", "¿cómo aprende un modelo?", "b"),
                                              (32, "azul", "¿en qué forma se los entrego?", "c"), (33, "violeta", "¿con cuáles me quedo?", "d"),
                                              (34, "ambar", "¿cómo lo explico?", "e")))],
    R(1690, 520, 360, 390, "ambar", 32, en="e", t=0.4), sec="2. Dónde estamos")
v.b("Los datos son los mismos de la clase treinta y tres: [[a]]la base de fuga de clientes de una telco. [[b]]Tres mil ciento cincuenta clientes, [[c]]doce variables originales, [[d]]y cuatrocientos noventa y cinco que se fueron: el quince coma siete por ciento.",
    CL(), P("Base de datos Churn.xlsx · limpiada igual que en la clase 33", 960, 210, "violeta", 44, ent="subir", en="a"),
    *[[Bc(x, 510, 440, 300, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(val, x, 480, tam=110, c=c, en=k, t=0.15),
       T(lab, x, 600, tam=34, c="gris", w="sb", en=k, t=0.25)]
      for x, c, val, lab, k in ((440, "violeta", mil(D["n"]), "clientes", "b"), (960, "azul", str(D["p"]), "variables originales", "c"),
                                (1480, "rojo", pct(D["tasa"]), f"se fuga · {D['n_fuga']} clientes", "d"))],
    real(800, en="d", t=0.8))

# ---- 3. Analogía: adivina quién
v.b("Para entender cómo piensa un árbol, juguemos a adivinar un personaje con preguntas de sí o no. [[a]]Empiezas con todos los candidatos. [[b]]Preguntas: ¿usa lentes? La mitad queda afuera. [[c]]Preguntas otra cosa, y otra, [[d]]hasta que el grupo que queda es casi de un solo tipo.",
    CL(), T("el juego de las preguntas de sí o no", 960, 180, tam=48, c="tinta", ent="subir"),
    Pt([(24, [360, 300, 1560, 520], "#94A3B8")], 20, id="aq", en="a"),
    P("¿usa lentes?", 960, 610, "violeta", 46, en="b"),
    U("aq", grupos=[dict(n=12, rect=[260, 300, 860, 520], color="#CBD5E1"), dict(n=12, rect=[1060, 300, 1660, 520], color="azul")], en="b", t=0.3, dur=1.0),
    P("¿tiene sombrero?", 1360, 720, "violeta", 42, en="c"),
    U("aq", grupos=[dict(n=12, rect=[260, 300, 860, 520], color="#CBD5E1"), dict(n=8, rect=[1060, 300, 1460, 520], color="#CBD5E1"),
                    dict(n=4, rect=[1500, 330, 1660, 490], color="azul")], en="c", t=0.3, dur=1.0),
    P("grupo casi puro", 1580, 850, "verde", 44, en="d"), ilustrativo(1010, texto="analogía"), sec="3. Analogía: preguntas de sí o no")
v.b("Un árbol de decisión hace exactamente eso con los clientes. [[a]]Cada pregunta parte al grupo en dos. [[b]]Y al final, en cada grupo, [[c]]predice la proporción de fuga que observó ahí.",
    CL(), Bc(960, 300, 680, 130, "blanco", "violeta", 5, 26, sombra=True, ent="pop"), T("¿pregunta sobre una variable?", 960, 300, tam=40, c="violeta"),
    L(860, 365, 620, 520, "grisclaro", 6, en="a"), L(1060, 365, 1300, 520, "grisclaro", 6, en="a"),
    T("sí", 700, 430, tam=32, c="gris", w="sb", en="a", t=0.3), T("no", 1220, 430, tam=32, c="gris", w="sb", en="a", t=0.3),
    Bc(620, 600, 420, 150, "blanco", "rojo", 5, 26, sombra=True, en="b", ent="pop"), Bc(1300, 600, 420, 150, "blanco", "verde", 5, 26, sombra=True, en="b", ent="pop"),
    T("grupo A", 620, 570, tam=34, c="gris", w="sb", en="b", t=0.1), T("grupo B", 1300, 570, tam=34, c="gris", w="sb", en="b", t=0.1),
    T("predice su tasa", 620, 625, tam=38, c="rojo", en="c"), T("predice su tasa", 1300, 625, tam=38, c="verde", en="c", t=0.2),
    P("la predicción de una hoja = la tasa de fuga de su grupo", 960, 840, "teal", 44, en="c", t=0.6))

# ---- 4. Paso a paso: partir el espacio (ejemplo ilustrativo con puntos generados)
rnd = random.Random(34)
PTS = []
for _ in range(70):
    x, y = rnd.uniform(0.3, 9.7), rnd.uniform(0.3, 9.7)
    p = 0.85 if (x < 4 and y > 5) else (0.3 if x < 4 else 0.07)
    PTS.append((round(x, 2), round(y, 2), rnd.random() < p))
roj = [(x, y) for x, y, f in PTS if f]
ver = [(x, y) for x, y, f in PTS if not f]


def cuenta(cond):
    s = [f for x, y, f in PTS if cond(x, y)]
    return len(s), sum(s)


RA, RB, RC = cuenta(lambda x, y: x < 4 and y > 5), cuenta(lambda x, y: x < 4 and y <= 5), cuenta(lambda x, y: x >= 4)
GX = dict(x0=520, y0=230, x1=1240, y1=900)
v.b("Veámoslo en un dibujo. [[a]]Cada punto es un cliente, ubicado según dos variables: cuánto usa el servicio y su antigüedad. [[b]]Los rojos se fueron. Los grises se quedaron.",
    CL(), G(GX["x0"], GX["y0"], GX["x1"], GX["y1"], 0, 10, 0, 10,
            [dict(xs=[p[0] for p in ver] + [50], ys=[p[1] for p in ver] + [5], color="#94A3B8", linea=None, marcadores=13, revela=1.0, alfa=0.0),
             dict(xs=[p[0] for p in roj] + [50], ys=[p[1] for p in roj] + [5], color="rojo", linea=None, marcadores=13, revela=1.0, alfa=0.0)],
            xticks=[0, 5, 10], yticks=[0, 5, 10], yfmt="{:.0f}", xlabel="uso del servicio", id="g1"),
    T("antigüedad", GX["x0"] - 10, GX["y0"] - 40, tam=32, c="tinta", w="sb", anc="lm"),
    U("g1", **{"series.0.alfa": 1.0, "series.1.alfa": 1.0}, en="a", dur=1.0),
    P("rojo = se fue", 1560, 300, "rojo", 44, en="b", id="lg1"), P("gris = se quedó", 1560, 400, "gris", 44, en="b", t=0.3, id="lg2"),
    ilustrativo(1010), sec="4. Paso a paso: partir el espacio")
v.b("El árbol hace su primera pregunta: [[a]]¿usa menos de cuatro? Es un corte vertical. [[b]]A la derecha casi todos se quedan. [[c]]A la izquierda hay mezcla, así que el árbol sigue preguntando solo ahí: [[d]]¿antigüedad mayor que cinco? Un corte horizontal.",
    X("lg1", "lg2", en="a"), L(4, 0, 4, 10, "violeta", 7, ref="g1", en="a", dur=0.8, id="c1"), P("¿uso < 4?", 1560, 300, "violeta", 44, en="a", t=0.3, id="q1"),
    Bx(4, 10, 10, 0, "verdeclaro", None, 0, 0, ref="g1", en="b", alfa=0.45, id="zc", ent="fade"),
    L(0, 5, 4, 5, "violeta", 7, ref="g1", en="d", dur=0.8, id="c2"), P("¿antigüedad > 5?", 1560, 400, "violeta", 44, en="d", t=0.3, id="q2"))
v.b("Quedaron tres rectángulos. [[a]]Arriba a la izquierda: poco uso y mucha antigüedad. [[b]]Abajo a la izquierda: poco uso y poca antigüedad. [[c]]Y a la derecha, los que usan mucho. [[d]]En cada rectángulo el árbol predice una constante: la proporción de rojos que hay adentro.",
    Bx(0, 10, 4, 5, "rojoclaro", None, 0, 0, ref="g1", en="a", alfa=0.5, id="za", ent="fade"),
    Bx(0, 5, 4, 0, "ambarclaro", None, 0, 0, ref="g1", en="b", alfa=0.5, id="zb", ent="fade"),
    T(f"arriba-izq: {RA[1]} de {RA[0]} = {100 * RA[1] / RA[0]:.0f} %", 1560, 560, tam=38, c="rojo", w="b", en="d", id="k1"),
    T(f"abajo-izq: {RB[1]} de {RB[0]} = {100 * RB[1] / RB[0]:.0f} %", 1560, 640, tam=38, c="ambar", w="b", en="d", t=0.4, id="k2"),
    T(f"derecha: {RC[1]} de {RC[0]} = {100 * RC[1] / RC[0]:.0f} %", 1560, 720, tam=38, c="verde", w="b", en="d", t=0.8, id="k3"))
v.b("Compáralo con la regresión logística de la carpeta anterior. [[a]]La logística ajusta una sola frontera, global y lineal, para todo el espacio. [[b]]El árbol, en cambio, dibuja muchas fronteras locales, [[c]]y todas perpendiculares a los ejes.",
    X("za", "zb", "zc", "q1", "q2", "k1", "k2", "k3"), L(0.5, 3.2, 6.6, 10, "azul", 6, ref="g1", en="a", guion=True, id="lg", dur=0.9),
    P("logística: una frontera, global y lineal", 1560, 300, "azul", 36, en="a", t=0.4, mw=640),
    U("c1", grosor=11, en="b", dur=0.4), U("c2", grosor=11, en="b", dur=0.4),
    P("árbol: muchas fronteras locales", 1560, 420, "violeta", 36, en="b", t=0.3, mw=640),
    P("y perpendiculares a los ejes", 1560, 520, "violeta", 36, en="c", mw=640))
v.b("Esa forma de partir trae tres consecuencias. [[a]]Captura la no linealidad sola, sin que nadie construya un logaritmo. [[b]]Captura interacciones solas: una rama puede usar variables distintas que la otra. [[c]]Pero solo corta en ángulo recto: una frontera diagonal le cuesta muchos escalones.",
    CL(), T("tres consecuencias de partir en rectángulos", 960, 200, tam=48, c="tinta", ent="subir"),
    tarjetas([(360, "verde", "no linealidad", "sin construir logs ni cuadrados", "a"), (960, "verde", "interacciones", "cada rama usa sus propias variables", "b"),
              (1560, "rojo", "solo ángulo recto", "una diagonal = muchos escalones", "c")], y=560, h=380))

# ---- 5. Ejemplo resuelto: un cliente baja por el árbol real
v.b("Ahora con el árbol real de la clase, de profundidad tres. [[a]]Sigamos a un cliente: reclamó, hizo cuarenta llamadas al mes, y no tuvo fallas técnicas.",
    CL(), arbol(pre="ar"), P("cliente: reclamó · 40 llamadas · 0 fallas", 960, 140, "violeta", 40, en="a", ent="subir", id="pcli"),
    real(1050), sec="5. Ejemplo: un cliente baja por el árbol")
v.b("Primera pregunta, en la raíz: [[a]]¿reclamos menor o igual a cero coma cinco? Él sí reclamó, así que la respuesta es no: va a la derecha. [[b]]Segunda pregunta: ¿frecuencia de llamadas menor o igual a cincuenta y siete coma cinco? Cuarenta, sí: izquierda. [[c]]Tercera: ¿fallas menor o igual a cero coma cinco? Cero fallas, sí: izquierda.",
    R(960, 270, 400, 140, "violeta", 22, en="a"), R(1410, 450, 400, 140, "violeta", 22, en="b"), R(1185, 640, 400, 140, "violeta", 22, en="c"))
v.b("Llegó a una hoja. [[a]]En esa hoja cayeron sesenta y siete clientes del entrenamiento, [[b]]y se fugó el sesenta y siete coma dos por ciento. Esa es su predicción. [[c]]Comparado con la tasa general de quince coma siete, es un lift de cuatro coma tres: [[d]]cuatro veces más riesgo que un cliente promedio.",
    R(1072, 840, 235, 140, "rojo", 22, en="a"), X("pcli", en="b"),
    Bc(960, 140, 1300, 90, "blanco", "rojo", 4, 24, en="b", ent="subir"), T(f"hoja: {H10['n']} clientes · {H10['fugas']} se fugan · {pct(H10['tasa'])}", 960, 140, tam=40, c="rojo", en="b", t=0.1),
    Bc(1640, 830, 470, 230, "blanco", "violeta", 5, 26, sombra=True, en="c", ent="pop"),
    T(f"lift = {H10['tasa']:.3f} / {D['tasa']:.3f}", 1640, 780, tam=34, c="tinta", w="sb", en="c", t=0.1),
    N(H10["tasa"] / D["tasa"], 1640, 860, "{:.1f}", 76, "violeta", suf=" ×", en="c", t=0.4, dur=1.0),
    P("4 veces el riesgo promedio", 1640, 975, "violeta", 34, en="d"))

# ---- 6. Lo que viene de la clase 33
v.b("Tres cosas de la clase treinta y tres hoy se explican solas. [[a]]Uno: la ganancia de información que usamos como filtro es exactamente el criterio con que el árbol elige cada corte. [[b]]Dos: la importancia de un árbol es selección embedded: el árbol elige variables mientras se entrena. [[c]]Tres: reclamos era la variable más fuerte, y hoy la vas a ver siempre arriba de todo.",
    CL(), T("tres cosas de la clase 33 que hoy se explican solas", 960, 190, tam=46, c="tinta", ent="subir"),
    tarjetas([(360, "violeta", "ganancia", "el filtro = el criterio de cada corte", "a"), (960, "azul", "embedded", "el árbol selecciona al entrenar", "b"),
              (1560, "rojo", "reclamos", "siempre en la raíz", "c")], y=560, h=380), sec="6. Lo que viene de la clase 33")
v.b("Y por eso hoy no construimos las cincuenta variables de la clase pasada. [[a]]El árbol puede llegar a lo mismo partiendo el espacio. [[b]]Con las doce originales, además, el árbol se puede leer. Y que se pueda leer es justamente la razón para elegir un árbol.",
    CL(), T("50 variables construidas", 500, 380, tam=56, c="gris", en=None, id="v50"), U("v50", tachado=1.0, en="a", dur=0.6),
    T("12 originales", 1360, 380, tam=64, c="verde", en="a", t=0.4), L(820, 380, 1130, 380, "gris", 6, True, en="a", t=0.2),
    P("el árbol parte el espacio por su cuenta", 960, 560, "teal", 46, en="a", t=0.8), P("y con 12 se puede leer", 960, 700, "verde", 52, en="b"))

# ---- 7. Error típico
v.b("El error típico de esta clase: [[a]]pensar que, como el árbol es fácil, no hace falta validarlo. [[b]]Es al revés. Como sobreajusta tanto, el árbol es el modelo que más exige validación cruzada y poda.",
    CL(), T("«es fácil, no hace falta validarlo»", 960, 340, tam=64, c="tinta", en="a", id="ef"), U("ef", tachado=1.0, color="gris", en="b", t=0.1, dur=0.6),
    P("el más fácil de sobreajustar: validación cruzada y poda", 960, 560, "verde", 48, en="b", t=0.5), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]En la clase treinta y tres construimos logaritmo de segundos para ayudar a la logística. [[b]]¿Por qué un árbol no lo necesita?",
    CL(), P("la clase 33 construyó log_segundos", 960, 320, "violeta", 52, en="a"),
    T("¿por qué el árbol no lo necesita?", 960, 520, tam=64, c="tinta", en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Porque el árbol solo pregunta si una variable está por encima o por debajo de un umbral. [[a]]Puede poner ese umbral donde quiera, y cortar la curva en escalones. [[b]]Un logaritmo mueve el número del umbral, pero no cambia quién queda a cada lado. Lo vas a comprobar con un número en el video nueve.",
    CL(), P("solo pregunta: ¿x ≤ umbral?", 960, 300, "teal", 60, ent="subir"),
    T("el umbral va donde haga falta", 960, 470, tam=48, c="tinta", w="sb", en="a"),
    P("log mueve el umbral, no la partición", 960, 640, "violeta", 54, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]un árbol parte el espacio con preguntas de sí o no, en rectángulos. [[b]]Cada hoja predice la tasa de fuga de su grupo. [[c]]Y es simple de explicar, pero fácil de sobreajustar.",
    CL(), P("preguntas sí / no → rectángulos", 960, 300, "violeta", 54, en="a"),
    P("cada hoja = la tasa de fuga de su grupo", 960, 480, "teal", 54, en="b"), P("simple de explicar · fácil de sobreajustar", 960, 660, "rojo", 52, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]leer el árbol completo de la clase, y convertir sus hojas en reglas de negocio.",
    CL(), siguiente(2, "Leer el árbol: reglas y segmentos"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
