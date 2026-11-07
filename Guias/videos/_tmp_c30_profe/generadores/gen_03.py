import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_03_calidad.json"
v = V("Calidad de datos e imputación")

v.portada("Calidad de datos e imputación", "¿Se puede confiar en los datos? ¿Y qué hacemos con los que faltan?",
          "Calidad de datos e imputación. Antes de modelar, hay que preguntarse si los datos son confiables, y qué hacer cuando faltan.")

# ---------------- 1. Cuatro dimensiones
v.b("Piensa en un cocinero. Aunque sea el mejor del mundo, [[a]]si los ingredientes están malos, [[b]]el plato sale malo.",
    CL(),
    Bc(430, 480, 420, 230, "ambarclaro", "ambar", 5, 30, id="ing", ent="subir", t=0.2), T("ingredientes", 430, 480, tam=50, c="ambar", t=0.3),
    L(660, 480, 860, 480, "gris", 7, True, t=0.5),
    Bc(1000, 480, 360, 230, "violetaclaro", "violeta", 5, 30, ent="subir", t=0.6), T("receta", 1000, 480, tam=50, c="violeta", t=0.7),
    L(1200, 480, 1400, 480, "gris", 7, True, t=0.9),
    Bc(1580, 480, 340, 230, "blanco", "grisclaro", 5, 30, id="plato", ent="subir", t=1.0), T("plato", 1580, 480, tam=50, c="gris", t=1.1),
    U("ing", relleno="rojoclaro", borde="rojo", dur=0.5, en="a"), P("malos", 430, 650, "rojo", 46, en="a", t=0.2),
    U("plato", relleno="rojoclaro", borde="rojo", dur=0.5, en="b"), K(1580, 480, "cruz", 120, "rojo", en="b", t=0.3),
    sec="1. Las cuatro dimensiones de calidad")
v.b("Con los modelos es igual: [[a]]un modelo no puede ser mejor que sus datos. [[b]]Un buen modelo no repara datos malos.",
    CL(), T("calidad del modelo", 960, 400, tam=80, c="violeta", en="a"), T("≤", 960, 520, tam=120, c="gris", en="a", t=0.3),
    T("calidad de los datos", 960, 640, tam=80, c="teal", en="a", t=0.5),
    P("un buen modelo no repara datos malos", 960, 850, "rojo", 52, en="b"))

DIM = [("Exactitud", "¿refleja la realidad?", "teal"), ("Completitud", "¿faltan datos?", "azul"),
       ("Coherencia", "¿las fuentes concuerdan?", "violeta"), ("Frescura", "¿está vigente?", "ambar")]
CX = [270, 730, 1190, 1650]


def dim_card(i, en, t=0):
    n, q, c = DIM[i]
    return [Bc(CX[i], 330, 410, 270, "blanco", c, 5, 28, sombra=True, id=f"dc{i}", en=en, t=t, ent="subir"),
            T(n, CX[i], 270, tam=50, c=c, en=en, t=t + 0.1), T(q, CX[i], 380, tam=34, c="tinta", w="sb", mw=330, en=en, t=t + 0.2)]


v.b("La clase propone cuatro dimensiones para revisar la calidad. [[a]]Son cuatro preguntas que le haces a tus datos.",
    CL(), [dim_card(i, "a", 0.18 * i) for i in range(4)])
v.b("Primera, la exactitud: [[a]]¿el dato refleja la realidad? Si un cliente de cuarenta años aparece con cuatrocientos, falla la exactitud.",
    R(CX[0], 330, 450, 310, "ambar", 34, id="rd", en="a", ent="fade"),
    g("ex", P("edad:  400 años", 480, 770, "rojo", 54, en="a", t=0.2), K(480, 900, "cruz", 100, "rojo", en="a", t=0.6),
      T("no es real", 480, 990, tam=36, c="rojo", w="sb", en="a", t=0.7)))
v.b("Segunda, la completitud: [[a]]¿faltan datos? Una celda vacía en la tabla es un problema de completitud.",
    XG("ex", dur=0.25), R(CX[1], 330, 450, 310, "ambar", 34, id="rd", en="a", ent="fade"),
    g("ex", [Bx(580, 740 + r * 80, 880, 810 + r * 80, "blanco", "linea", 3, 12, en="a", t=0.15 * r) for r in range(3)],
      T("23", 730, 775, tam=40, c="tinta", w="sb", en="a"), T("41", 730, 935, tam=40, c="tinta", w="sb", en="a", t=0.3),
      Bx(580, 820, 880, 890, "ambarclaro", "ambar", 4, 12, en="a", t=0.4, guion=True), T("?", 730, 855, tam=48, c="ambar", en="a", t=0.5)))
v.b("Tercera, la coherencia: [[a]]¿las distintas fuentes concuerdan entre sí? Si la web dice una cosa y el contrato dice otra, no hay coherencia.",
    XG("ex", dur=0.25), R(CX[2], 330, 450, 310, "ambar", 34, id="rd", en="a", ent="fade"),
    g("ex", Bc(1020, 800, 250, 150, "azulclaro", "azul", 4, 22, en="a", t=0.1), T("web", 1020, 770, tam=40, c="azul", en="a", t=0.2), T("dato A", 1020, 830, tam=36, c="tinta", w="sb", en="a", t=0.3),
      Bc(1360, 800, 250, 150, "verdeclaro", "verde", 4, 22, en="a", t=0.3), T("contrato", 1360, 770, tam=40, c="verde", en="a", t=0.4), T("dato B", 1360, 830, tam=36, c="tinta", w="sb", en="a", t=0.5),
      T("≠", 1190, 800, tam=90, c="rojo", en="a", t=0.7)))
v.b("Y cuarta, la frescura: [[a]]¿el dato sigue vigente? Un teléfono de hace diez años probablemente ya no sirve.",
    XG("ex", dur=0.25), R(CX[3], 330, 450, 310, "ambar", 34, id="rd", en="a", ent="fade"),
    P("teléfono de hace diez años", 1500, 790, "ambar", 44, en="a"), T("¿sigue vigente?", 1500, 900, tam=46, c="ambar", w="sb", en="a", t=0.6))
v.b("Pueden fallar varias a la vez. [[a]]Y hay algo más serio: los sesgos que traen los datos se convierten en sesgos del modelo. [[b]]En clase se mencionaron los generadores de imágenes como ejemplo.",
    CL(),
    Bc(480, 480, 480, 240, "ambarclaro", "ambar", 5, 30, ent="subir", t=0.2), T("datos con sesgo", 480, 480, tam=50, c="ambar", t=0.3),
    L(730, 480, 1010, 480, "gris", 8, True, en="a"), Bc(1290, 480, 480, 240, "rojoclaro", "rojo", 5, 30, en="a", t=0.4), T("modelo con sesgo", 1290, 480, tam=50, c="rojo", en="a", t=0.5),
    T("el sesgo pasa de los datos al modelo", 960, 760, tam=46, c="gris", w="sb", en="b"))
v.b("Piensa un segundo: si la web y el contrato no coinciden, [[a]]¿qué dimensión falla primero?",
    CL(), Bc(560, 480, 380, 200, "azulclaro", "azul", 4, 24, ent="subir"), T("web", 560, 480, tam=56, c="azul"),
    T("≠", 960, 480, tam=100, c="rojo", t=0.3), Bc(1360, 480, 380, 200, "verdeclaro", "verde", 4, 24, ent="subir", t=0.2), T("contrato", 1360, 480, tam=56, c="verde", t=0.2),
    g("pq", P("¿qué dimensión?", 960, 780, "violeta", 60, en="a")), pausa=0.5)
v.b("La coherencia: las fuentes no concuerdan. [[a]]Y para decidir cuál de las dos es la verdad, ahí entra la exactitud. Conviene responder las dos.",
    XG("pq", dur=0.3), P("coherencia", 960, 780, "violeta", 60, t=0.2), P("+ exactitud: ¿cuál es la verdad?", 960, 900, "teal", 46, en="a"))

# ---------------- 2. Faltantes
v.b("Hablemos de la completitud, porque es la que más vas a enfrentar. [[a]]En las trescientas viviendas de la clase, [[b]]la antigüedad tiene veinte valores faltantes.",
    CL(),
    Pt([(300, [160, 280, 1760, 760], "#94A3B8")], 11, visibles=300, id="dv", en="a", dur=1.8),
    T("300 viviendas", 960, 190, tam=64, c="tinta", id="n300", en="a"),
    U("dv", grupos=[dict(n=280, rect=[160, 280, 1440, 760], color="#94A3B8"), dict(n=20, rect=[1560, 280, 1760, 760], color="ambar")], dur=1.4, en="b"),
    P("20 sin antigüedad", 1660, 200, "ambar", 40, en="b", t=0.8),
    sec="2. Datos faltantes")
v.b("¿Cuánto es eso? [[a]]Faltantes dividido por filas: veinte entre trescientos, [[b]]seis coma siete por ciento.",
    T("% faltante  =  faltantes / filas", 960, 880, tam=50, c="gris", w="sb", id="f1", en="a"),
    T("20 / 300", 760, 960, tam=64, c="tinta", id="f2", en="a", t=0.4), T("=  6.7 %", 1150, 960, tam=74, c="ambar", id="f3", en="b"))
v.b("En pandas, lo cuentas con una línea: [[a]]isna, suma. [[b]]Solo la antigüedad tiene faltantes.",
    CL(),
    T("viviendas.isna().sum()", 960, 170, tam=44, c="tinta", w="m", fondo="#E8EDF4", pad=22, en="a"),
    [[T(n, 700, 330 + 105 * i, tam=40, c="tinta", w="m", anc="rm", en="a", t=0.12 * i + 0.3),
      Bx(740, 300 + 105 * i, 740 + max(8, c * 24), 360 + 105 * i, "ambar" if c else "grisclaro", None, 0, 10, grow="x", en="b", t=0.1 * i + (0.3 if c else 0), dur=0.6),
      T(str(c), 790 + max(8, c * 24), 330 + 105 * i, tam=44, c="ambar" if c else "gris", anc="lm", en="b", t=0.1 * i + (0.7 if c else 0.2))]
     for i, (n, c) in enumerate([("superficie_m2", 0), ("n_habitaciones", 0), ("barrio", 0), ("antiguedad_años", 20), ("tiene_ascensor", 0), ("precio_clp", 0)])])
v.b("Cuando faltan datos, se pueden completar con un valor razonable. [[a]]A eso se le llama imputar.",
    CL(), P("imputar", 960, 400, "violeta", 110, en="a"), T("completar lo que falta con un valor razonable", 960, 600, tam=52, c="gris", w="sb", en="a", t=0.5))
v.b("Pero hay un límite. [[a]]Si falta un tercio de los datos, o más, no se imputa: [[b]]se revisa cómo se recogieron.",
    CL(),
    Bx(260, 400, 1660, 520, "grisclaro", None, 0, 20, ent="subir", t=0.1), Bx(260, 400, 727, 520, "ambar", None, 0, 20, grow="x", en="a", dur=0.9),
    T("1/3", 493, 460, tam=60, c="blanco", en="a", t=0.6), T("falta un tercio", 960, 650, tam=60, c="ambar", en="a", t=0.8),
    P("no se imputa: se revisa la recolección", 960, 820, "rojo", 54, en="b"))

# ---------------- 3. Media, mediana, moda
v.b("Y para completar, hay tres reglas. [[a]]Variable simétrica y sin valores extremos: la media. [[b]]Con atípicos, o asimétrica: la mediana. [[c]]Categórica: la moda.",
    CL(),
    [[Bc(x, 430, 500, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 340, tam=64, c=c, en=k, t=0.1),
      T(d, x, 470, tam=34, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "teal", "media", "simétrica, sin extremos", "a"), (960, "violeta", "mediana", "con atípicos o asimétrica", "b"), (1560, "ambar", "moda", "categórica", "c"))],
    sec="3. Media, mediana o moda")
v.b("Recordemos qué es cada una. [[a]]La media es el promedio. [[b]]La mediana es el valor que queda en el medio cuando ordenas los datos. [[c]]Y la moda es el valor que más se repite.",
    T("promedio", 360, 700, tam=46, c="teal", w="sb", en="a"), T("el valor del medio", 960, 700, tam=46, c="violeta", w="sb", en="b"),
    T("el que más se repite", 1560, 700, tam=46, c="ambar", w="sb", en="c"))

BX = [420, 620, 820, 1020, 1220]
VAL = [1.2, 1.4, 1.5, 1.6, 9.0]
BASE = 900
SC = 48
v.b("Veamos un ejemplo con seis clientes. A uno le falta el ingreso. Los otros cinco ganan, en millones de pesos: [[a]]uno coma dos, uno coma cuatro, uno coma cinco, uno coma seis, [[b]]y nueve.",
    CL(),
    T("ingresos (millones de pesos)", 960, 160, tam=48, c="gris", w="sb", ent="subir"),
    L(300, BASE, 1500, BASE, "tinta", 4, id="ejeb", en="a", dur=0.5),
    [[Bx(x - 60, BASE - val * SC, x + 60, BASE, "azul" if val < 5 else "rojo", None, 0, 10, id=f"ib{i}", grow="y", en="a" if i < 4 else "b", t=0.5 + 0.5 * i if i < 4 else 0.1, dur=0.5),
      T(f"{val:.1f}", x, BASE - val * SC - 38, tam=40, c="tinta", id=f"iv{i}", en="a" if i < 4 else "b", t=0.9 + 0.5 * i if i < 4 else 0.5)] for i, (x, val) in enumerate(zip(BX, VAL))],
    Bx(1360, BASE - 150, 1480, BASE, "ambarclaro", "ambar", 4, 12, guion=True, id="faltaB", en="b", t=0.3),
    T("falta", 1420, BASE - 75, tam=36, c="ambar", id="faltaT", en="b", t=0.4))
v.b("La media: sumamos todo, catorce coma siete, y dividimos entre cinco. [[a]]Da dos coma noventa y cuatro.",
    T("(1.2 + 1.4 + 1.5 + 1.6 + 9.0) / 5  =  14.7 / 5", 960, 270, tam=44, c="tinta", w="sb", id="fm", en=None, t=0.2),
    L(300, BASE - 2.94 * SC, 1500, BASE - 2.94 * SC, "teal", 6, guion=True, id="lm", en="a"),
    T("media = 2.94", 1520, BASE - 2.94 * SC, tam=42, c="teal", anc="lm", id="lmt", en="a", t=0.5))
v.b("Pero mira: [[a]]casi todos ganan menos de dos. [[b]]Ese nueve empuja la media hacia arriba, a más del doble de lo que gana la mayoría.",
    R(1220, BASE - 216, 200, 460, "rojo", 30, id="r9", en="a"), R(720, BASE - 40, 820, 180, "azul", 30, id="r9b", en="a", t=0.3),
    X("r9", "r9b", en="b", dur=0.3), L(1220, BASE - 9 * SC - 30, 1220, BASE - 2.94 * SC - 25, "rojo", 8, True, id="empuja", en="b"),
    T("empuja la media", 1330, 520, tam=42, c="rojo", anc="lm", id="empt", en="b", t=0.3))
v.b("La mediana, en cambio: ordenamos, y el valor del medio es [[a]]uno coma cinco.",
    X("empuja", "empt", dur=0.3), T("ordenados:  1.2  ·  1.4  ·  1.5  ·  1.6  ·  9.0", 960, 340, tam=40, c="gris", w="sb", id="ord", ent="subir", t=0.2),
    L(300, BASE - 1.5 * SC, 1500, BASE - 1.5 * SC, "violeta", 6, guion=True, id="lmed", en="a"),
    R(820, BASE - 40, 140, 110, "violeta", 24, id="rmed", en="a"), T("mediana = 1.5", 1520, BASE - 1.5 * SC, tam=42, c="violeta", anc="lm", id="lmedt", en="a", t=0.4))
v.b("¿Cuál representa mejor al cliente típico? [[a]]La mediana. Por eso, para imputar este ingreso, usamos uno coma cinco.",
    X("faltaT", en="a", dur=0.2), N(1.5, 1420, BASE - 75, "{:.1f}", 54, "violeta", id="fill", en="a", t=0.2, dur=0.8),
    U("faltaB", relleno="violetaclaro", borde="violeta", guion=False, dur=0.6, en="a", t=0.3),
    P("con atípicos  →  mediana", 960, 980, "violeta", 52, en="a", t=0.6))
v.b("¿Y la región de otro cliente? [[a]]Tenemos Norte, Sur, Sur, Este, Sur. [[b]]¿Cuál se repite más? Sur, tres de cinco veces. Esa es la moda.",
    CL(),
    [P(n, 300 + 330 * i, 440, "teal" if n != "Sur" else "ambar", 60, en="a", t=0.2 * i) for i, n in enumerate(["Norte", "Sur", "Sur", "Este", "Sur"])],
    R(630, 440, 230, 120, "ambar", 40, id="m1", en="b"), R(960, 440, 230, 120, "ambar", 40, id="m2", en="b", t=0.3), R(1620, 440, 230, 120, "ambar", 40, id="m3", en="b", t=0.6),
    N(3, 960, 700, "{:.0f}", 130, "ambar", suf=" de 5", en="b", t=0.9, dur=1.0), P("moda = Sur", 960, 900, "ambar", 64, en="b", t=1.8))
v.b("Y si la variable es simétrica, como estas edades, [[a]]treinta, treinta y dos, treinta y cuatro, treinta y seis y treinta y ocho, [[b]]la media es ciento setenta entre cinco: treinta y cuatro.",
    CL(),
    [[Bc(360 + 300 * i, 430, 200, 120, "tealclaro", "teal", 4, 22, en="a", t=0.2 * i), T(str(e), 360 + 300 * i, 430, tam=64, c="teal", en="a", t=0.2 * i + 0.1)] for i, e in enumerate([30, 32, 34, 36, 38])],
    T("(30 + 32 + 34 + 36 + 38) / 5  =  170 / 5", 960, 680, tam=48, c="tinta", w="sb", en="b"), T("= 34", 960, 800, tam=110, c="teal", en="b", t=0.6),
    P("simétrica  →  media", 960, 950, "teal", 50, en="b", t=1.0))
v.b("En las viviendas de la clase pasa lo mismo con los ingresos simulados. [[a]]La media es setecientos cincuenta mil pesos, [[b]]pero la mediana es seiscientos cuarenta mil: los atípicos tiran la media.",
    CL(),
    T("ingresos simulados, cola larga", 960, 190, tam=48, c="gris", w="sb"),
    Bx(380, 800 - 0.75 * 500, 760, 800, "teal", None, 0, 12, grow="y", en="a", dur=0.8), T("media 0.75 M", 570, 800 - 0.75 * 500 - 40, tam=44, c="teal", en="a", t=0.7),
    Bx(1160, 800 - 0.64 * 500, 1540, 800, "violeta", None, 0, 12, grow="y", en="b", dur=0.8), T("mediana 0.64 M", 1350, 800 - 0.64 * 500 - 40, tam=44, c="violeta", en="b", t=0.7),
    L(300, 800, 1620, 800, "tinta", 4))
v.b("Piensa un segundo. En el caso del banco: el ingreso tiene ocho por ciento de nulos y sesgo a la derecha; la edad, dos por ciento y simétrica; [[a]]y la comuna, cinco por ciento. ¿Con qué imputas cada una?",
    CL(),
    T("ingreso   8 %  ·  sesgo a la derecha", 260, 340, tam=46, c="tinta", w="sb", anc="lm", t=0.2), T("edad   2 %  ·  simétrica", 260, 540, tam=46, c="tinta", w="sb", anc="lm", t=0.4),
    T("comuna   5 %  ·  categórica", 260, 740, tam=46, c="tinta", w="sb", anc="lm", en="a"),
    T("?", 1500, 340, tam=90, c="gris", id="q1", t=0.4), T("?", 1500, 540, tam=90, c="gris", id="q2", t=0.6), T("?", 1500, 740, tam=90, c="gris", id="q3", en="a", t=0.2), pausa=0.6)
v.b("[[a]]El ingreso, con la mediana. [[b]]La edad, con la media. [[c]]Y la comuna, con la moda.",
    X("q1", en="a", dur=0.2), P("mediana", 1500, 340, "violeta", 60, en="a", t=0.1),
    X("q2", en="b", dur=0.2), P("media", 1500, 540, "teal", 60, en="b", t=0.1),
    X("q3", en="c", dur=0.2), P("moda", 1500, 740, "ambar", 60, en="c", t=0.1))

# ---------------- 4. Errores típicos
HB = [26, 33, 21, 22, 35, 35, 28, 28, 25, 27]
hx = lambda i: 340 + 120 * i
v.b("Ahora, dos errores típicos. El primero: [[a]]imputar con la media hace que la variable varíe menos. [[b]]Mira la antigüedad de las viviendas: los veinte valores imputados caen todos en el mismo lugar.",
    CL(),
    T("antigüedad (años)  ·  280 viviendas con dato", 960, 180, tam=44, c="gris", w="sb", ent="subir"),
    L(280, 820, 1560, 820, "tinta", 4),
    [[Bx(hx(i) - 52, 820 - c * 6, hx(i) + 52, 820, "grisclaro", None, 0, 6, grow="y", en="a", t=0.06 * i, dur=0.5)] for i, c in enumerate(HB)],
    [T(f"{5 * i}", hx(i) - 60, 850, tam=30, c="gris", w="sb", anc="mm", en="a", t=0.3) for i in range(11)],
    Bx(hx(4) - 52, 820 - 35 * 6 - 20 * 6, hx(4) + 52, 820 - 35 * 6, "ambar", None, 0, 6, grow="y", id="apil", en="b", dur=0.8),
    T("20 filas apiladas", hx(4), 820 - 55 * 6 - 40, tam=42, c="ambar", id="apilt", en="b", t=0.6),
    sec="4. Errores típicos")
v.b("Como todas valen lo mismo, la variable pierde variación. [[a]]La desviación estándar baja de catorce coma diecinueve [[b]]a trece coma setenta y uno.",
    N(14.19, 1380, 330, "{:.2f}", 90, "gris", id="sd1", en="a", dur=0.8), T("→", 1560, 330, tam=70, c="gris", en="b"),
    N(13.71, 1730, 330, "{:.2f}", 90, "rojo", id="sd2", en="b", dur=0.8), T("desviación estándar", 1560, 240, tam=36, c="gris", w="sb", en="a"))
v.b("El segundo error: [[a]]calcular la media o la mediana con todas las filas. Esos valores se aprenden solo con el entrenamiento. [[b]]Si no, es fuga de datos, y lo vamos a ver al final.",
    CL(),
    Bx(260, 400, 1100, 540, "violeta", None, 0, 20, grow="x", en="a", dur=0.8), T("entrenamiento", 680, 470, tam=52, c="blanco", en="a", t=0.5),
    Bx(1120, 400, 1390, 540, "azul", None, 0, 20, grow="x", en="a", t=0.7, dur=0.4), Bx(1410, 400, 1660, 540, "rojo", None, 0, 20, grow="x", en="a", t=1.0, dur=0.4),
    T("validación", 1255, 470, tam=34, c="blanco", en="a", t=1.0), T("prueba", 1535, 470, tam=34, c="blanco", en="a", t=1.1),
    T("imputer.fit(solo entrenamiento)", 680, 660, tam=44, c="violeta", w="m", en="a", t=1.2),
    P("fuga de datos", 960, 850, "rojo", 56, en="b"))
v.b("En scikit-learn, el imputador se llama simple impúter. [[a]]Le dices la estrategia: media, mediana o la más frecuente. [[b]]Y lo ajustas solo con el entrenamiento.",
    CL(),
    T("SimpleImputer(strategy='median')", 960, 360, tam=46, c="tinta", w="m", fondo="#E8EDF4", pad=26, en="a"),
    T("# 'mean'  ·  'median'  ·  'most_frequent'", 960, 480, tam=36, c="gris", w="m", en="a", t=0.4),
    T("media   ·   mediana   ·   moda", 960, 640, tam=48, c="violeta", w="sb", en="a", t=0.6),
    P("fit: solo con el entrenamiento", 960, 820, "teal", 54, en="b"))
v.b("Una opción más avanzada: [[a]]si la columna se relaciona con otras, puedes predecir el valor que falta con un modelo de regresión.",
    CL(), P("imputar con un modelo de regresión", 960, 450, "azul", 56, en="a"), T("cuando la columna se relaciona con otras", 960, 600, tam=42, c="gris", w="sb", en="a", t=0.5))
v.b("Para llevarte: [[a]]media si es simétrica, mediana si hay atípicos, moda si es categórica. [[b]]Si falta un tercio, no se imputa. [[c]]Y los valores se aprenden solo con el entrenamiento.",
    CL(),
    P("simétrica → media  ·  atípicos → mediana  ·  categórica → moda", 960, 320, "violeta", 46, en="a"),
    P("falta un tercio → no se imputa", 960, 520, "ambar", 56, en="b"),
    P("se aprende solo con el entrenamiento", 960, 720, "teal", 56, en="c"), sec="5. Para llevar")

print(v.guardar(OUT), "beats")
