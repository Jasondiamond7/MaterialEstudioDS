import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, flujo, siguiente, ilustrativo

OUT = out("guion_c32_01_introduccion.json")
v = V("Introducción: representar antes de modelar")

v.portada("Reducción de dimensionalidad", "Cuántas variables necesitas de verdad, y cómo entregárselas al modelo",
          "Hola. Bienvenido a la serie de videos de la clase treinta y dos. Hoy vas a aprender a preparar las variables para un modelo, y a resumirlas cuando sobran.")

# ---- 1. Hook
v.b("Imagina que vas a cocinar. [[a]]Tienes veinte ingredientes en la mesa, [[b]]pero seis de ellos son casi la misma sal con distinto nombre. [[c]]Y unos vienen en gramos, otros en kilos.",
    CL(), P("vas a cocinar", 960, 230, "violeta", 56, ent="subir"),
    Pt([(14, [300, 360, 900, 640], "#94A3B8")], 14, id="ing", visibles=14, en="a", dur=1.0),
    Pt([(6, [1060, 400, 1620, 600], "ambar")], 16, id="sal", visibles=6, en="b", dur=0.8), P("seis «sales» casi iguales", 1340, 690, "ambar", 44, en="b", t=0.5),
    P("gramos", 600, 820, "azul", 48, en="c"), P("kilos", 1340, 820, "rojo", 48, en="c", t=0.3), L(760, 820, 1220, 820, "gris", 5, False, guion=True, en="c", t=0.2),
    sec="1. La gran idea")
v.b("Antes de cocinar, un buen chef ordena: [[a]]mide todo en la misma unidad, [[b]]corta cada ingrediente de la forma correcta, [[c]]y saca de la mesa lo que repite sabor.",
    CL(), tarjetas([(360, "teal", "misma unidad", "medir todo igual", "a"), (960, "violeta", "cortar bien", "la forma correcta", "b"),
                    (1560, "ambar", "quitar repetidos", "lo que repite sabor", "c")]))
v.b("Con los datos pasa exactamente lo mismo. [[a]]La gran idea de hoy es esta: un modelo no recibe datos. [[b]]Recibe una representación de los datos.",
    CL(), T("un modelo no recibe datos", 960, 380, tam=76, c="gris", en="a"), K(1560, 380, "cruz", 70, en="a", t=0.5),
    T("recibe una representación", 960, 600, tam=86, c="violeta", en="b"), R(960, 600, 1300, 170, "violeta", 40, id="rr", en="b", t=0.5))
v.b("Y esa representación se arma en dos pasos. [[a]]Primero, representar: escalar, codificar y construir variables. [[b]]Después, si sobran columnas, reducir: resumirlas en pocas.",
    CL(), Bc(560, 520, 700, 420, "blanco", "teal", 5, 30, sombra=True, en="a", ent="subir"), T("1 · Representar", 560, 380, tam=56, c="teal", en="a", t=0.1),
    T("escalar · codificar · construir", 560, 540, tam=38, c="tinta", w="sb", mw=600, en="a", t=0.2),
    L(930, 520, 1010, 520, "gris", 7, True, en="b"),
    Bc(1380, 520, 700, 420, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("2 · Reducir", 1380, 380, tam=56, c="azul", en="b", t=0.1),
    T("resumir muchas columnas en pocas", 1380, 540, tam=38, c="tinta", w="sb", mw=600, en="b", t=0.2))
v.b("La profesora lo dijo así: [[a]]las matemáticas buscan el camino más eficiente. Reducir dimensiones es eficiente, [[b]]pero no es reducir por reducir.",
    CL(), Bc(960, 480, 1500, 420, "violetaclaro", None, 0, 40, ent="pop"),
    T("«Las matemáticas buscan el camino más eficiente: reducir dimensiones es eficiente,", 960, 420, tam=46, c="violeta", w="sb", mw=1300, en="a"),
    T("pero no es reducir por reducir»", 960, 580, tam=54, c="violeta", en="b"), T("Prof. Laura Sánchez", 960, 760, tam=34, c="gris", w="sb", en="b", t=0.5))

# ---- 2. De dónde venimos
v.b("¿De dónde venimos? [[a]]En la clase treinta, nos aseguramos de que los datos fueran confiables. [[b]]En la treinta y uno, vimos cómo aprende un modelo. [[c]]Hoy, en la treinta y dos, le entregamos variables bien representadas, y resumidas.",
    CL(),
    tarjetas([(360, "gris", "Clase 30", "¿los datos son confiables?", "a"), (960, "verde", "Clase 31", "¿cómo aprende un modelo?", "b"),
              (1560, "violeta", "Clase 32", "¿en qué forma se los entrego?", "c")], y=500, h=400),
    L(620, 500, 700, 500, "grisclaro", 6, True, en="b"), L(1220, 500, 1300, 500, "grisclaro", 6, True, en="c"), sec="2. De dónde venimos")
v.b("Fíjate en la diferencia. [[a]]Un dato puede estar perfecto, sin errores ni faltantes, [[b]]y aun así estar mal representado. Si eso pasa, el modelo resume una distorsión.",
    CL(), P("dato correcto", 560, 400, "verde", 56, en="a"), K(560, 520, "check", 90, en="a", t=0.3),
    P("mal representado", 1360, 400, "rojo", 56, en="b"), K(1360, 520, "cruz", 90, en="b", t=0.3),
    T("el modelo resume una distorsión", 960, 760, tam=56, c="tinta", en="b", t=0.8))

# ---- 3. Los datos
v.b("Vamos a trabajar con clientes de una empresa de telecomunicaciones. [[a]]La pregunta de negocio es la fuga: ¿qué cliente va a cancelar su suscripción?",
    CL(), P("telco · fuga de clientes", 960, 260, "violeta", 60, ent="subir"),
    Pt([(85, [360, 380, 1560, 760], "#94A3B8"), (15, [360, 800, 1560, 880], "rojo")], 12, id="cli", visibles=100, en="a", dur=1.6),
    T("cada punto, un 1 % de los clientes", 960, 960, tam=30, c="gris", w="sb", en="a", t=1.2), sec="3. Los datos de la clase")
v.b("El notebook de la profesora genera mil quinientos clientes simulados. [[a]]Tienen doce variables, [[b]]y el quince coma cero siete por ciento se fuga.",
    CL(), T("datos simulados de la profesora", 960, 220, tam=40, c="gris", w="sb"),
    *[[Bc(x, 520, 460, 300, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), (T("1.500", x, 490, tam=110, c=c, en=k, t=0.2) if val == D["n"] else N(val, x, 490, fmt, 110, c, en=k, t=0.2, dur=0.9, suf=suf)), T(lab, x, 610, tam=36, c="gris", w="sb", en=k, t=0.3)]
      for x, c, val, fmt, suf, lab, k in ((420, "violeta", D["n"], "{:,.0f}", "", "clientes", None), (960, "azul", 12, "{:.0f}", "", "variables", "a"),
                                          (1500, "rojo", D["fuga"], "{:.2f}", " %", "se fuga", "b"))],
    T("generados con default_rng(32)", 960, 820, tam=32, c="gris", w="sb", en="b", t=0.6))
VARS = ["numero_fallas", "reclamos", "antiguedad", "monto", "segundos", "frec_llamadas", "frec_sms", "n_contactos", "tipo_plan", "estado", "edad", "clv"]
v.b("Estas son las doce variables. [[a]]Hay fallas y reclamos, [[b]]antigüedad y monto del plan, [[c]]segundos, llamadas, mensajes y contactos, [[d]]y el tipo de plan, el estado, la edad y el valor del cliente.",
    CL(), T("las 12 variables", 960, 190, tam=44, c="gris", w="sb", ent="subir"),
    *[[Bc(330 + 420 * (i % 4), 330 + 150 * (i // 4), 380, 110, "blanco", c, 4, 22, en=k, ent="subir", t=0.05 * (i % 4)),
       T(nm, 330 + 420 * (i % 4), 330 + 150 * (i // 4), tam=32, c="tinta", w="sb", en=k, t=0.05 * (i % 4) + 0.1)]
      for i, (nm, c, k) in enumerate(zip(VARS, ["ambar", "ambar", "gris", "violeta", "violeta", "violeta", "violeta", "violeta", "gris", "gris", "gris", "violeta"],
                                         ["a", "a", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d"]))],
    P("objetivo: fuga (1 = se fue · 0 = sigue)", 960, 820, "rojo", 44, en="d", t=0.6))
v.b("Ojo con un detalle de las diapositivas: [[a]]dicen «dataset real de mil quinientos clientes». [[b]]Esos mil quinientos son simulados. La base real es otra: tres mil ciento cincuenta clientes, con quince coma siete por ciento de fuga.",
    CL(), P("alerta del material", 960, 220, "ambar", 50, ent="subir"),
    T("«dataset real de 1.500 clientes»", 960, 400, tam=52, c="gris", t=0.4, id="qreal"), U("qreal", tachado=1.0, en="b", dur=0.6),
    P("1.500 = simulados", 560, 600, "violeta", 52, en="b"), P("base real: 3.150 · fuga 15.7 %", 1300, 600, "verde", 52, en="b", t=0.5),
    T("la base real se usa en el caso aplicado (video 9)", 960, 780, tam=36, c="gris", w="sb", en="b", t=1.0))

# ---- 4. Ejemplo en números
v.b("Veamos la ruta de hoy, en números. [[a]]Partimos con doce variables. [[b]]Construimos tres más, con cocientes. Doce más tres, quince columnas.",
    CL(), T("la ruta en números", 960, 200, tam=48, c="gris", w="sb", ent="subir"),
    Bc(380, 500, 360, 260, "blanco", "gris", 5, 28, sombra=True, en="a", ent="subir"), N(12, 380, 470, "{:.0f}", 120, "gris", en="a", t=0.1), T("variables", 380, 580, tam=34, c="gris", w="sb", en="a", t=0.2),
    L(580, 500, 680, 500, "gris", 7, True, en="b"), T("+ 3 construidas", 640, 320, tam=34, c="azul", w="sb", en="b", t=0.1),
    Bc(900, 500, 360, 260, "blanco", "azul", 5, 28, sombra=True, en="b", ent="subir", t=0.2), N(15, 900, 470, "{:.0f}", 120, "azul", en="b", t=0.3), T("columnas", 900, 580, tam=34, c="azul", w="sb", en="b", t=0.4),
    sec="4. La ruta en números")
v.b("Después, PCA resume esas quince columnas. [[a]]Con ocho componentes conserva el ochenta y cinco por ciento de la varianza. [[b]]Ocho de quince: usamos poco más de la mitad de las columnas.",
    L(1100, 500, 1200, 500, "gris", 7, True, en="a"), Bc(1440, 500, 440, 260, "blanco", "violeta", 5, 28, sombra=True, en="a", ent="subir"),
    N(8, 1440, 450, "{:.0f}", 120, "violeta", en="a", t=0.1), T("componentes · 85.0 %", 1440, 570, tam=34, c="violeta", w="sb", en="a", t=0.3),
    P("8 / 15 = 53 % de las columnas", 960, 800, "teal", 52, en="b"))

# ---- 5. Error típico
v.b("Y aquí viene el error más común: [[a]]creer que más columnas es más información.",
    CL(), T("más columnas = más información", 960, 440, tam=80, c="tinta", en="a", id="mcmi"), sec="5. El error típico")
v.b("No es así. [[a]]Si dos columnas cuentan la misma historia, la segunda no agrega casi nada. [[b]]Solo agrega costo, inestabilidad y ruido.",
    U("mcmi", tachado=1.0, color="gris", dur=0.6, t=0.2),
    P("la misma historia dos veces", 960, 640, "ambar", 54, en="a"), P("costo · inestabilidad · ruido", 960, 800, "rojo", 54, en="b"))

# ---- 6. Piensa un segundo
v.b("Piensa un segundo. [[a]]En los datos de la clase, los segundos hablados y el número de llamadas tienen una correlación de cero coma noventa y nueve. ¿Cuánta información nueva aporta la segunda?",
    CL(), T("segundos  ↔  frec_llamadas", 960, 330, tam=60, c="violeta", en="a"), N(D["corr_seg_llam"], 960, 500, "r = {:.2f}", 130, "rojo", en="a", t=0.3),
    T("¿cuánta información nueva aporta la segunda?", 960, 720, tam=50, c="tinta", en="a", t=0.8), sec="6. Piensa un segundo", pausa=0.6)
v.b("Casi ninguna. [[a]]Son prácticamente la misma variable. Por eso, antes de modelar, hay que decidir cuántas dimensiones necesitas de verdad.",
    CL(), P("casi ninguna", 960, 330, "verde", 70, ent="subir"), P("¿cuántas dimensiones necesito de verdad?", 960, 560, "violeta", 56, en="a"),
    P("¿y cómo se las entrego al modelo sin hacer trampa?", 960, 720, "teal", 48, en="a", t=0.8))

# ---- 7. Mapa
ETQ = [("El problema de la dimensionalidad", "rojo"), ("Escalar: estandarizar y normalizar", "teal"), ("Codificar y construir variables", "violeta"),
       ("PCA: la idea y una 2×2 a mano", "azul"), ("Varianza explicada: cuántos componentes", "azul"), ("Cargas y efecto de la escala", "azul"),
       ("t-SNE: un mapa para explorar", "verde"), ("Pipeline sin fuga y caso Churn", "ambar"), ("Repaso final", "gris")]


def tj(i, t):
    x, y = 330 + 630 * (i % 3), 300 + 230 * (i // 3)
    c = ETQ[i][1]
    return [Bc(x, y, 580, 190, "blanco", c, 5, 26, sombra=True, en="a", t=t), T(str(i + 2), x - 225, y, tam=72, c=c, en="a", t=t + 0.05),
            T(ETQ[i][0], x + 45, y, tam=32, c="tinta", w="sb", mw=430, en="a", t=t + 0.1)]


v.b("Este es el mapa de la serie. [[a]]Después de esta introducción vienen nueve paradas. Empezamos por el problema, seguimos con la representación, luego PCA y t-SNE, y cerramos con el caso real.",
    CL(), [tj(i, 0.18 * i) for i in range(9)], sec="7. El mapa de la serie")

# ---- 8. Para llevar
v.b("Para llevarte de esta introducción: [[a]]un modelo recibe una representación. [[b]]Primero se representa, después se reduce. [[c]]Y más columnas no es más información.",
    CL(), P("el modelo recibe una representación", 960, 300, "violeta", 54, en="a"), P("representar  →  reducir", 960, 480, "teal", 54, en="b"),
    P("más columnas ≠ más información", 960, 660, "rojo", 54, en="c"), sec="8. Para llevar")
v.b("Vamos con la primera parada: [[a]]el problema de la dimensionalidad.",
    CL(), siguiente(2, "El problema de la dimensionalidad"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
