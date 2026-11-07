import sys, json, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_05_kmeans.json"
D = json.load(open(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe\datos31.json", encoding="utf-8"))
K_ = D["km"]
v = V("Sin etiquetas: K-Means y PCA")

v.portada("Sin etiquetas: K-Means y PCA", "Encontrar estructura cuando nadie te dice la respuesta",
          "Aprendizaje no supervisado. Cómo encontrar grupos y resumir variables cuando no hay etiquetas.")

# ---- 1. Hook
v.b("Imagina que heredas una biblioteca sin catálogo. [[a]]Nadie te dice qué libros son de qué tema. [[b]]Tu trabajo es ordenarlos en estantes, mirando cuáles se parecen.",
    CL(), P("biblioteca sin catálogo", 960, 270, "violeta", 56, en="a"),
    Pt([(12, [260, 420, 700, 640], "azul"), (12, [760, 420, 1200, 640], "ambar"), (12, [1260, 420, 1700, 640], "teal")], 11, id="bib", en="b", t=0.2), P("ordenar por parecido", 960, 800, "teal", 54, en="b", t=0.8),
    sec="1. Sin respuesta correcta")
v.b("Eso es el aprendizaje no supervisado. [[a]]No hay respuesta correcta que adivinar. [[b]]Lo que se busca es estructura: grupos, o una forma más simple de mirar los datos.",
    CL(), P("no supervisado", 960, 270, "teal", 60, ent="subir", t=0.2), P("no hay respuesta correcta", 960, 480, "gris", 54, en="a"), P("se busca estructura", 960, 640, "violeta", 54, en="b"))
v.b("En esta parada vemos dos herramientas. K-Means [[a]]forma grupos alrededor de centros. [[b]]Y PCA resume variables parecidas en pocas. Importante: PCA no forma grupos.",
    CL(), Bc(520, 500, 700, 380, "blanco", "verde", 5, 30, sombra=True, en="a", ent="subir"), T("K-Means", 520, 420, tam=64, c="verde", en="a", t=0.1), T("forma grupos", 520, 540, tam=44, c="tinta", w="sb", en="a", t=0.2),
    Bc(1400, 500, 700, 380, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("PCA", 1400, 420, tam=64, c="azul", en="b", t=0.1), T("resume variables", 1400, 540, tam=44, c="tinta", w="sb", en="b", t=0.2),
    P("PCA no forma grupos", 1400, 760, "rojo", 44, en="b", t=0.8))

# ---- 2. K-Means paso a paso
v.b("Veamos K-Means paso a paso. [[a]]Uno: eliges cuántos grupos quieres, k, y pones k centros al azar. [[b]]Dos: cada punto se asigna al centro más cercano. [[c]]Tres: cada centro se mueve al promedio de su grupo. [[d]]Y cuatro: repites, hasta que nada cambie.",
    CL(),
    *[[Bc(x, 500, 400, 340, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 360, tam=100, c=c, en=k, t=0.1), T(tx, x, 540, tam=36, c="tinta", w="sb", mw=330, en=k, t=0.2)]
      for x, c, n, tx, k in ((260, "violeta", "1", "elegir k centros", "a"), (740, "azul", "2", "asignar cada punto al centro más cercano", "b"),
                              (1220, "teal", "3", "mover cada centro al promedio de su grupo", "c"), (1700, "ambar", "4", "repetir hasta que nada cambie", "d"))],
    sec="2. K-Means paso a paso")
v.b("Es como repartir niños en dos equipos para un juego. [[a]]Dos capitanes se ubican. Cada niño se va con el capitán que tiene más cerca. [[b]]Luego cada capitán se mueve al centro de su equipo, y los niños vuelven a elegir.",
    CL(), P("dos capitanes", 960, 270, "violeta", 54, en="a"),
    Pt([(10, [260, 400, 760, 700], "azul"), (10, [1160, 400, 1660, 700], "ambar")], 11, id="ni", t=0.2),
    Bc(500, 330, 60, 60, "azulclaro", "azul", 6, 30, ent="pop", en="a", t=0.5), Bc(1400, 330, 60, 60, "ambarclaro", "ambar", 6, 30, ent="pop", en="a", t=0.7),
    P("cada niño con el capitán más cercano", 960, 850, "teal", 46, en="a", t=1.0), P("los capitanes se mueven al centro", 960, 950, "ambar", 44, en="b"))

# ---- 3. Ejemplo resuelto: seis puntos
PUN = {"A": (1, 2), "B": (2, 1), "C": (2, 4), "D": (7, 7), "E": (8, 9), "F": (9, 6)}


def fx(v_):
    return 240 + v_ * 100


def fy(v_):
    return 880 - v_ * 68


def px(n):
    return fx(PUN[n][0]), fy(PUN[n][1])


def punto(n, en=None, t=0.0):
    x, y = px(n)
    return [Bc(x, y, 44, 44, "grisclaro", "tinta", 3, 22, id=f"pt{n}", en=en, t=t), T(n, x + 34, y - 30, tam=36, c="tinta", id=f"pl{n}", en=en, t=t + 0.1)]


def pinta(n, c, en=None, t=0.0):
    return U(f"pt{n}", relleno=c + "claro", borde=c, dur=0.5, en=en, t=t)


v.b("Hagámoslo con seis puntos del apunte. [[a]]Cada punto tiene una antigüedad y un uso. A es uno y dos, B es dos y uno, C es dos y cuatro, [[b]]D es siete y siete, E es ocho y nueve, y F es nueve y seis. Queremos dos grupos.",
    CL(),
    G(240, 200, 1240, 880, 0, 10, 0, 10, [], xticks=[0, 2, 4, 6, 8, 10], yticks=[0, 2, 4, 6, 8, 10], yfmt="{:.0f}", xlabel="antigüedad", id="gk", t=0.1), T("uso", 240, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.2),
    [punto(n, "a", 0.1 * i) for i, n in enumerate("ABC")], [punto(n, "b", 0.1 * i) for i, n in enumerate("DEF")],
    g("tab", T("A (1, 2)   B (2, 1)   C (2, 4)", 1560, 330, tam=36, c="tinta", w="sb", en="a"), T("D (7, 7)   E (8, 9)   F (9, 6)", 1560, 400, tam=36, c="tinta", w="sb", en="b")),
    sec="3. K-Means a mano: una iteración")
v.b("Los centros iniciales son los puntos A y F. [[a]]El centro uno, en azul, empieza en A. [[b]]El centro dos, en ámbar, empieza en F.",
    Bc(*px("A"), 62, 62, "azulclaro", "azul", 6, 12, id="c1", en="a", ent="pop"), T("c₁", *px("A"), tam=30, c="azul", id="c1t", en="a", t=0.1),
    Bc(*px("F"), 62, 62, "ambarclaro", "ambar", 6, 12, id="c2", en="b", ent="pop"), T("c₂", *px("F"), tam=30, c="ambar", id="c2t", en="b", t=0.1),
    pinta("A", "azul", en="a"), pinta("F", "ambar", en="b"), XG("tab", dur=0.3),
    g("tab", P("c₁ = A    c₂ = F", 1560, 330, "violeta", 44, en="a")))
v.b("Ahora asignamos. Para cada punto medimos la distancia al cuadrado hasta cada centro. [[a]]El punto B está a dos del centro uno, y a setenta y cuatro del dos. [[b]]Gana el uno.",
    XG("tab", dur=0.3), g("tab", T("distancia² al centro", 1560, 300, tam=34, c="gris", w="sb")),
    L(*px("B"), *px("A"), "azul", 5, id="lB1", en="a"), L(*px("B"), *px("F"), "ambar", 4, guion=True, id="lB2", en="a", t=0.3),
    g("tab", T("B:   c₁ = 2    c₂ = 74", 1560, 400, tam=42, c="tinta", en="a", t=0.2), T("B → c₁", 1560, 480, tam=60, c="azul", en="b")), pinta("B", "azul", en="b"))
v.b("El punto C está a cinco del centro uno, y a cincuenta y tres del dos. [[a]]También se va con el uno.",
    X("lB1", "lB2", dur=0.3), L(*px("C"), *px("A"), "azul", 5, id="lC1", t=0.2), L(*px("C"), *px("F"), "ambar", 4, guion=True, id="lC2", t=0.4),
    g("tab", T("C:   c₁ = 5    c₂ = 53", 1560, 580, tam=42, c="tinta", t=0.3), T("C → c₁", 1560, 660, tam=60, c="azul", en="a")), pinta("C", "azul", en="a"))
v.b("Con D y E pasa lo contrario. [[a]]D está a sesenta y uno del centro uno, y a cinco del dos. [[b]]E, a noventa y ocho del uno, y a diez del dos. Los dos se van con el centro dos.",
    X("lC1", "lC2", dur=0.3), L(*px("D"), *px("F"), "ambar", 5, id="lD2", t=0.2), L(*px("E"), *px("F"), "ambar", 5, id="lE2", en="b", t=0.1),
    XG("tab", dur=0.3), g("tab", T("D:   c₁ = 61    c₂ = 5", 1560, 400, tam=42, c="tinta", en="a"), T("E:   c₁ = 98    c₂ = 10", 1560, 480, tam=42, c="tinta", en="b"), T("D, E → c₂", 1560, 580, tam=60, c="ambar", en="b", t=0.3)),
    pinta("D", "ambar", en="a"), pinta("E", "ambar", en="b"))
v.b("Quedan dos grupos: A, B y C con el uno, y D, E y F con el dos. [[a]]Antes de mover nada, la inercia, que es la suma de las distancias al cuadrado de cada punto a su centro, vale veintidós.",
    X("lD2", "lE2", dur=0.3), XG("tab", dur=0.3), g("tab", P("grupo 1: A, B, C", 1560, 330, "azul", 42), P("grupo 2: D, E, F", 1560, 430, "ambar", 42), T("inercia = 0 + 2 + 5 + 5 + 10 + 0", 1560, 560, tam=34, c="tinta", w="sb", en="a"), T("= 22", 1560, 650, tam=80, c="rojo", en="a", t=0.3)))
v.b("Segundo paso: mover cada centro al promedio de su grupo. [[a]]Para el uno: uno, dos y dos, entre tres, es uno coma sesenta y siete. Y dos, uno y cuatro, entre tres, es dos coma treinta y tres. [[b]]Para el dos: ocho en x, y siete coma treinta y tres en y.",
    XG("tab", dur=0.3), g("tab", T("c₁ = ((1+2+2)/3 , (2+1+4)/3)", 1560, 330, tam=34, c="azul", w="sb", en="a"), T("= (1.67 , 2.33)", 1560, 400, tam=48, c="azul", en="a", t=0.3),
                          T("c₂ = ((7+8+9)/3 , (7+9+6)/3)", 1560, 560, tam=34, c="ambar", w="sb", en="b"), T("= (8.00 , 7.33)", 1560, 630, tam=48, c="ambar", en="b", t=0.3)),
    U("c1", x=fx(5 / 3), y=fy(7 / 3), en="a", dur=1.2), U("c1t", x=fx(5 / 3), y=fy(7 / 3), en="a", dur=1.2),
    U("c2", x=fx(8), y=fy(22 / 3), en="b", dur=1.2), U("c2t", x=fx(8), y=fy(22 / 3), en="b", dur=1.2))
v.b("Con los nuevos centros, ¿cambia alguna asignación? [[a]]No: cada punto sigue más cerca de su centro. [[b]]El algoritmo terminó. Y la inercia bajó de veintidós a doce.",
    XG("tab", dur=0.3), g("tab", T("¿cambia alguna asignación?", 1560, 340, tam=44, c="tinta", mw=480), P("no cambia", 1560, 460, "verde", 56, en="a"), K(1560, 580, "check", 80, en="a", t=0.3),
                          T("inercia", 1400, 760, tam=40, c="gris", w="sb", anc="rm", en="b"), T("22 → 12", 1700, 760, tam=80, c="verde", en="b")))
v.b("Piensa un segundo: [[a]]si las asignaciones hubieran cambiado, ¿qué tendrías que hacer?",
    XG("tab", dur=0.3), g("tab", T("¿y si cambian las asignaciones?", 1560, 400, tam=44, c="tinta", mw=500, en="a"), T("?", 1560, 560, tam=140, c="violeta", en="a", t=0.3)), pausa=0.6)
v.b("Volver a mover los centros, y repetir. [[a]]Se continúa hasta que nada cambie. En un caso real, se hace varias veces con distintos puntos de partida, y se queda la mejor inercia.",
    XG("tab", dur=0.3), g("tab", P("repetir hasta que no cambie", 1560, 420, "teal", 40, mw=500), P("varios puntos de partida: n_init", 1560, 560, "violeta", 38, en="a", mw=500), P("gana la menor inercia", 1560, 700, "verde", 40, en="a", mw=500, t=0.5)))

# ---- 4. Escalar
v.b("Un detalle crucial: K-Means usa distancias. [[a]]Imagina que la antigüedad se mide en días, de trescientos sesenta y cinco a tres mil seiscientos cincuenta, [[b]]y el uso, en veces al mes, de cero a treinta.",
    CL(), P("K-Means usa distancias", 960, 250, "violeta", 54, ent="subir"),
    Bc(560, 560, 640, 260, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("antigüedad en días", 560, 500, tam=44, c="azul", en="a", t=0.1), T("365 a 3.650", 560, 590, tam=52, c="tinta", en="a", t=0.2),
    Bc(1360, 560, 640, 260, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"), T("uso mensual", 1360, 500, tam=44, c="teal", en="b", t=0.1), T("0 a 30", 1360, 590, tam=52, c="tinta", en="b", t=0.2),
    sec="4. Escalar antes de agrupar")
v.b("La variable con números enormes domina toda la distancia. [[a]]El agrupamiento solo reflejaría la antigüedad, [[b]]y el uso ni se notaría.",
    CL(), Bx(300, 400, 1500, 480, "azul", None, 0, 12, grow="x", t=0.2), T("antigüedad", 200, 440, tam=36, c="azul", anc="rm"),
    Bx(300, 560, 340, 640, "teal", None, 0, 12, t=0.4), T("uso", 200, 600, tam=36, c="teal", anc="rm"),
    P("domina la distancia", 1500, 330, "azul", 46, en="a"), P("el uso ni se nota", 700, 700, "teal", 46, en="b"))
v.b("La solución es estandarizar antes, [[a]]para que todas las variables pesen parecido. Así lo hicimos con Club Activo. [[b]]Sin estandarizar, los segmentos cambian: la coincidencia con los estandarizados es de cero coma treinta y nueve.",
    CL(), P("estandarizar antes de agrupar", 960, 300, "teal", 60, en="a"),
    T("coincidencia (ARI)", 960, 520, tam=44, c="gris", w="sb", en="b"), N(0.39, 960, 680, "{:.2f}", 190, "ambar", en="b", t=0.2, dur=1.0), T("1 = idénticos", 1500, 680, tam=34, c="gris", w="sb", en="b", t=0.8))
v.b("Ese número se llama ARI, y mide qué tan parecidas son dos formas de agrupar. [[a]]Con cero coma treinta y nueve son bastante distintas. Escalar no es un detalle: cambia tus segmentos.",
    CL(), P("ARI 0.39: segmentos distintos", 960, 420, "ambar", 70, ent="subir"), P("escalar cambia tus segmentos", 960, 660, "violeta", 56, en="a"))

# ---- 5. Elegir k
v.b("¿Y cuántos grupos elegir? [[a]]No existe un k verdadero. [[b]]Se mira el codo de la inercia, la silueta, y sobre todo, si los grupos tienen sentido para el negocio.",
    CL(), T("¿cuántos grupos, k?", 960, 270, tam=72, c="violeta", ent="subir"), P("no hay un k verdadero", 960, 470, "rojo", 54, en="a"),
    P("el codo de la inercia", 560, 700, "azul", 46, en="b"), P("la silueta", 1060, 700, "teal", 46, en="b", t=0.3), P("el sentido para el negocio", 1500, 800, "verde", 46, en="b", t=0.6), sec="5. Elegir k")
KS = [2, 3, 4, 5, 6, 7, 8]
v.b("La inercia siempre baja al agregar grupos. [[a]]Se busca el codo: donde dejar de ganar mucho. [[b]]En Club Activo, el codo no es nítido.",
    CL(),
    G(300, 220, 1140, 840, 2, 8, 0, 1700, [dict(xs=KS, ys=[K_["inercia"][str(k)] for k in KS], color="violeta", grosor=7, marcadores=10, revela=0.0)],
      xticks=KS, yticks=[0, 500, 1000, 1500], yfmt="{:.0f}", xlabel="k (número de grupos)", id="ga", t=0.1), T("inercia", 300, 170, tam=34, c="tinta", w="sb", anc="lm", t=0.2),
    U("ga", **{"series.0.revela": 1.0}, en="a", dur=2.2), P("el codo no es nítido", 1560, 460, "ambar", 46, en="b", mw=560))
v.b("La silueta mide qué tan separados están los grupos. [[a]]Más alta es mejor, y va de menos uno a uno. [[b]]Con k igual a dos da cero coma trescientos cuarenta y ocho, con tres, cero coma trescientos treinta y ocho, y con cuatro, cero coma trescientos veintisiete.",
    CL(),
    *[[Bc(x, 560, 400, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(f"k = {n}", x, 420, tam=56, c=c, en=k, t=0.1), N(val, x, 590, "{:.3f}", 100, c, en=k, t=0.2, dur=0.9)]
      for x, c, n, val, k in ((460, "teal", 2, 0.348, "b"), (960, "violeta", 3, 0.338, "b"), (1460, "azul", 4, 0.327, "b"))],
    T("silueta: más alta = grupos más separados", 960, 280, tam=44, c="gris", w="sb", en="a"))
v.b("Casi no hay diferencia. [[a]]Sin un k claro, decidimos con el negocio. [[b]]Tres segmentos son fáciles de leer y de actuar, así que se usa k igual a tres.",
    CL(), P("casi iguales: no hay un k claro", 960, 330, "ambar", 60, ent="subir"), P("decide el negocio", 960, 560, "verde", 60, en="a"), P("k = 3: tres segmentos fáciles de leer", 960, 780, "violeta", 54, en="b"))

# ---- 6. Club Activo
CEN = K_["cent"]
COL = ["azul", "ambar", "teal"]
xmin, xmax, ymin, ymax = -2.5, 5.5, -3.0, 4.0
v.b("Veamos los clientes de Club Activo. [[a]]Se agrupan con tres variables estandarizadas: antigüedad, uso y gasto. [[b]]Aquí se muestran dos de las tres, y una muestra de ciento sesenta clientes.",
    CL(),
    G(240, 200, 1240, 880, xmin, xmax, ymin, ymax, [dict(xs=K_["x"], ys=K_["y"], color="#94A3B8", linea=False, marcadores=9)], xticks=[-2, 0, 2, 4], yticks=[-2, 0, 2], yfmt="{:.0f}", xlabel="antigüedad (estandarizada)", id="gc", t=0.2),
    T("uso (estandarizado)", 240, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.3), sec="6. K-Means en Club Activo")
v.b("K-Means los reparte en tres segmentos, [[a]]de doscientos cincuenta y dos, cuatrocientos treinta y nueve, y ciento nueve clientes. [[b]]Los círculos oscuros grandes son los centros de cada grupo.",
    *[G(240, 200, 1240, 880, xmin, xmax, ymin, ymax, [dict(xs=[x for x, l in zip(K_["x"], K_["lab"]) if l == s], ys=[y for y, l in zip(K_["y"], K_["lab"]) if l == s], color=COL[s], linea=False, marcadores=9)],
        id=f"gs{s}", en="a", ejes=0.0, t=0.2 * s) for s in range(3)],
    U("gc", **{"series.0.alfa": 0.0}, en="a", dur=0.4),
    G(240, 200, 1240, 880, xmin, xmax, ymin, ymax, [dict(xs=[c[0] for c in CEN], ys=[c[1] for c in CEN], color="tinta", linea=False, marcadores=18)], id="gcn", en="b", ejes=0.0),
    g("pn", P("segmento 0 · 252", 1560, 380, "azul", 42, en="a"), P("segmento 1 · 439", 1560, 480, "ambar", 42, en="a", t=0.3), P("segmento 2 · 109", 1560, 580, "teal", 42, en="a", t=0.6)))
v.b("Fíjate en el segmento dos, en verde azulado: [[a]]son los clientes más antiguos, a casi dos desviaciones sobre el promedio, [[b]]y también los de mayor gasto, uno coma cuarenta y ocho.",
    R(927, 591, 620, 520, "teal", 40, id="rs2", en="a", ent="fade"),
    XG("pn", dur=0.3), g("pn", T("segmento 2", 1560, 330, tam=46, c="teal"), N(1.99, 1560, 460, "{:+.2f}", 90, "teal", suf=" sd", en="a", dur=0.9), T("antigüedad", 1560, 550, tam=34, c="gris", w="sb", en="a", t=0.2),
                          N(1.48, 1560, 680, "{:+.2f}", 90, "teal", suf=" sd", en="b", dur=0.9), T("gasto", 1560, 770, tam=34, c="gris", w="sb", en="b", t=0.2)))
v.b("Los segmentos cero y uno se separan sobre todo por el uso. [[a]]El cero usa más que el promedio: más cero coma noventa y nueve. [[b]]El uno, menos: menos cero coma cincuenta y seis.",
    X("rs2", dur=0.3), XG("pn", dur=0.3), g("pn", T("segmentos 0 y 1", 1560, 330, tam=46, c="gris", w="sb"), T("uso", 1560, 430, tam=40, c="gris", w="sb"),
                                         P("segmento 0: +0.99", 1560, 540, "azul", 46, en="a"), P("segmento 1: −0.56", 1560, 660, "ambar", 46, en="b")))
v.b("Y ahora, la validación. [[a]]La variable cancelo no se usó para agrupar. [[b]]Aun así, el segmento dos tiene solo siete por ciento de cancelación, frente a veintiuno y veinticuatro de los otros.",
    CL(), P("«cancelo» no se usó para agrupar", 960, 250, "gris", 48, en="a"),
    *[[Bc(x, 560, 400, 360, "blanco", c, 5, 28, sombra=True, en="b", ent="subir"), T(f"segmento {s}", x, 420, tam=44, c=c, en="b", t=0.1), N(val, x, 590, "{:.0f}", 120, c, suf=" %", en="b", t=0.2, dur=0.9), T("cancela", x, 700, tam=34, c="gris", w="sb", en="b", t=0.3)]
      for x, c, s, val in ((460, "azul", 0, 21), (960, "ambar", 1, 24), (1460, "teal", 2, 7))])
v.b("Eso demuestra que los grupos tienen sentido. Pero ojo: en el notebook se esperaba que el segmento dos fuera el de más cancelación. [[a]]Ocurre lo contrario. [[b]]Y se decía que el gasto era la variable que más lo distingue, pero es la antigüedad, con uno coma noventa y nueve, contra uno coma cuarenta y ocho del gasto.",
    CL(), P("alerta del material", 960, 250, "ambar", 54, ent="subir"), P("se esperaba: más cancelación", 560, 480, "gris", 42, en="a"), K(560, 580, "cruz", 80, en="a", t=0.3), P("realidad: 7 % frente a 21-24 %", 1360, 480, "verde", 42, en="a", t=0.3),
    P("lo que más lo distingue: antigüedad, no gasto", 960, 800, "azul", 46, en="b"))
v.b("Los grupos de K-Means no traen nombre. [[a]]Tú los lees mirando el perfil de cada uno, y les pones el nombre que tenga sentido. Por ejemplo: «antiguos y de alto gasto», «usuarios frecuentes» y «de poco uso».",
    CL(), P("los grupos no traen nombre", 960, 280, "violeta", 54, ent="subir"), P("antiguos y de alto gasto", 960, 520, "teal", 48, en="a"), P("usuarios frecuentes", 960, 650, "azul", 48, en="a", t=0.4), P("de poco uso", 960, 780, "ambar", 48, en="a", t=0.8),
    T("nombres sugeridos, no oficiales", 960, 900, tam=30, c="gris", w="sb", en="a"))

# ---- 7. PCA
v.b("Ahora, PCA. [[a]]Es la técnica para resumir variables correlacionadas en pocas. Como una foto en dos dimensiones de un objeto de tres: eliges el ángulo que muestra más.",
    CL(), P("PCA: resumir variables", 960, 270, "azul", 60, ent="subir", t=0.2), Bc(560, 600, 520, 300, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("3 variables", 560, 600, tam=56, c="azul", en="a", t=0.1),
    L(850, 600, 1070, 600, "gris", 8, True, en="a", t=0.4), Bc(1360, 600, 520, 300, "azulclaro", "azul", 5, 30, en="a", ent="subir", t=0.5), T("2 componentes", 1360, 600, tam=56, c="azul", en="a", t=0.6), sec="7. PCA: un nuevo sistema de ejes")
v.b("Cada componente es un eje nuevo, que mezcla las tres variables. [[a]]El primero captura cincuenta y ocho coma dos por ciento de la varianza. [[b]]El segundo, treinta y tres coma seis. [[c]]Juntos, noventa y uno coma ocho.",
    CL(),
    *[[T(n, x, 310, tam=44, c=c, en=k), Bx(x - 90, 800 - val * 5.8, x + 90, 800, c, None, 0, 10, en=k, grow="-y" if False else None, ent="subir", t=0.1), N(val, x, 800 - val * 5.8 - 40, "{:.1f}", 54, c, suf=" %", en=k, t=0.3, dur=0.8)]
      for x, c, n, val, k in ((560, "azul", "PC1", 58.2, "a"), (960, "teal", "PC2", 33.6, "b"), (1360, "gris", "PC3 (resto)", 8.2, "c"))],
    L(300, 810, 1620, 810, "tinta", 4, t=0.1))
v.b("Fíjate que dije varianza, no información. [[a]]Noventa y uno coma ocho por ciento de la varianza no es noventa y uno coma ocho de la información. Y eliminar componentes no es quitar ruido: es descartar direcciones de poca variación.",
    CL(), N(91.8, 960, 400, "{:.1f}", 200, "azul", suf=" %", dur=1.0, t=0.2), T("de la varianza", 960, 580, tam=56, c="tinta", w="sb"), P("no es «información»", 960, 740, "rojo", 54, en="a"), P("y descartar ≠ quitar ruido", 960, 860, "ambar", 48, en="a", t=0.8))
v.b("Aquí están los mismos clientes, proyectados con PCA. [[a]]Los colores vienen de K-Means. PCA no los formó. Solo cambió los ejes, para mirarlos mejor.",
    CL(),
    G(240, 200, 1240, 880, -3.5, 3.5, -3.5, 3.5, [dict(xs=[x for x, l in zip(K_["pcx"], K_["lab"]) if l == s], ys=[y for y, l in zip(K_["pcy"], K_["lab"]) if l == s], color=COL[s], linea=False, marcadores=9) for s in range(3)],
      xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-3, -2, -1, 0, 1, 2, 3], yfmt="{:.0f}", xlabel="PC1 (58.2 %)", id="gp", t=0.2), T("PC2 (33.6 %)", 240, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.3),
    P("colores: K-Means", 1560, 420, "verde", 44, en="a"), P("ejes: PCA", 1560, 540, "azul", 44, en="a", t=0.4))
v.b("Es un error común decir «PCA separa tres clústeres». [[a]]Los clústeres los formó K-Means. [[b]]La Clase treinta y dos explica PCA en profundidad: cómo se calcula, cuántos componentes elegir, y qué es t-SNE.",
    CL(), P("«PCA separa tres clústeres»", 960, 300, "rojo", 54, ent="subir"), K(1620, 300, "cruz", 80, en=None, t=0.5), P("los formó K-Means", 960, 520, "verde", 54, en="a"), P("más detalle de PCA: Clase 32", 960, 760, "violeta", 54, en="b"))
v.b("Una última nota. Con la base real de tres mil ciento cincuenta clientes, y seis variables de uso, [[a]]K-Means con k igual a tres tiene una silueta de cero coma cuatrocientos setenta y tres. [[b]]Y la fuga en la prueba fue de veinticuatro coma siete por ciento, cero, y dos coma tres por ciento, según el grupo.",
    CL(), P("base real · 3.150 clientes · 6 variables de uso", 960, 250, "gris", 40, ent="subir"), T("silueta (k = 3)", 960, 400, tam=44, c="gris", w="sb", en="a"), N(0.473, 960, 520, "{:.3f}", 130, "violeta", en="a", dur=0.9),
    *[[Bc(x, 800, 380, 180, "blanco", c, 5, 28, sombra=True, en="b", ent="subir"), N(val, x, 800, "{:.1f}", 80, c, suf=" %", en="b", t=0.2, dur=0.9)] for x, c, val in ((460, "rojo", 24.7), (960, "verde", 0.0), (1460, "ambar", 2.3))],
    T("fuga en la prueba, por grupo", 960, 680, tam=36, c="gris", w="sb", en="b"))

# ---- 8. cierre
v.b("Para llevarte: K-Means [[a]]asigna cada punto al centro más cercano, y mueve los centros al promedio. [[b]]Estandariza antes de agrupar. [[c]]No hay un k verdadero: decide con el negocio. [[d]]Y los grupos se interpretan con su perfil.",
    CL(), P("asignar  →  promediar  →  repetir", 960, 250, "verde", 52, en="a"), P("estandarizar antes de agrupar", 960, 430, "teal", 52, en="b"), P("no hay un k verdadero", 960, 610, "violeta", 52, en="c"), P("interpretar con el perfil", 960, 790, "ambar", 52, en="d"), sec="8. Para llevar")
v.b("Vamos con la quinta parada: [[a]]el subajuste, el sobreajuste y la regularización.",
    CL(), Bc(960, 480, 620, 260, "ambarclaro", "ambar", 6, 40, id="p1", ent="pop", en="a"),
    T("5", 720, 480, tam=120, c="ambar", en="a", t=0.1), T("Subajuste, sobreajuste y regularización", 1040, 480, tam=40, c="tinta", mw=420, en="a", t=0.15), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
