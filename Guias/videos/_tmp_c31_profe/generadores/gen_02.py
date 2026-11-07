import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
import numpy as np

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_02_reglas.json"
v = V("Del problema a la tarea de ML")

v.portada("Del problema a la tarea de ML", "Reglas fijas, el ciclo del aprendizaje y las familias de problemas",
          "Del problema a la tarea de {Machine Learning|mashín lérnin}. Antes de elegir un modelo, hay que saber qué problema tienes.")

# ---------------- 1. Reglas fijas
v.b("Empecemos por una pregunta que la profesora hizo en clase: [[a]]¿qué decisión toma tu empresa hoy basándose en reglas fijas?",
    CL(), T("¿qué decisión toma tu empresa hoy", 960, 400, tam=62, c="tinta", en="a"), T("basándose en reglas fijas?", 960, 500, tam=62, c="violeta", en="a", t=0.4),
    sec="1. Las reglas fijas")
v.b("Una regla fija es una condición que alguien escribió a mano. [[a]]Si el pedido pasa de cierto monto, se revisa. [[b]]Si la nota es menor que la mínima, se reprueba. [[c]]Siempre igual, para todos.",
    CL(),
    Bc(960, 500, 1100, 330, "blanco", "teal", 5, 30, sombra=True, id="rf", ent="subir", t=0.2),
    T("si monto > límite  →  revisar", 960, 410, tam=52, c="tinta", en="a"), T("si nota < mínima  →  reprobar", 960, 520, tam=52, c="tinta", en="b"),
    P("siempre igual, para todos", 960, 640, "teal", 46, en="c"), T("ejemplos ilustrativos", 960, 860, tam=32, c="gris", w="sb", en="c", t=0.4))
v.b("Y funcionan muy bien, cuando la regla es clara. [[a]]El problema aparece cuando las variables son difíciles de medir con exactitud, [[b]]o cuando el mundo cambia y la regla se queda vieja.",
    CL(),
    P("funcionan bien si la regla es clara", 960, 300, "verde", 52, ent="subir", t=0.2), K(1560, 300, "check", 70, en=None, t=0.5),
    Bc(520, 640, 640, 300, "blanco", "ambar", 5, 30, sombra=True, en="a", ent="subir"), T("variables difíciles de medir", 520, 640, tam=42, c="ambar", mw=520, en="a", t=0.2),
    Bc(1400, 640, 640, 300, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("el mundo cambió, la regla no", 1400, 640, tam=42, c="rojo", mw=520, en="b", t=0.2))
v.b("Es como una receta que escribiste hace diez años. [[a]]Si cambió el horno, el ingrediente o el gusto de la gente, [[b]]la receta ya no sale igual, y alguien tiene que reescribirla a mano.",
    CL(), P("una receta de hace diez años", 960, 330, "violeta", 56, ent="subir", t=0.2),
    P("cambió el horno, el ingrediente, el gusto", 960, 560, "ambar", 50, en="a"), P("alguien la reescribe a mano", 960, 760, "rojo", 54, en="b"))
v.b("El Machine Learning propone otra cosa: [[a]]en lugar de escribir la regla, le muestras ejemplos al modelo, [[b]]y él descubre la regla ajustando sus parámetros.",
    CL(),
    Bc(520, 500, 560, 340, "blanco", "grisclaro", 5, 30, sombra=True, ent="subir", en="a"), T("ejemplos", 520, 420, tam=56, c="violeta", en="a", t=0.1),
    Pt([(14, [330, 480, 710, 560], "#94A3B8"), (6, [330, 590, 710, 630], "rojo")], 8, id="pe2", en="a", t=0.3),
    L(820, 500, 1100, 500, "violeta", 8, True, en="b"), Bc(1400, 500, 560, 340, "violetaclaro", "violeta", 5, 30, sombra=True, en="b", ent="subir", t=0.3),
    T("modelo", 1400, 420, tam=56, c="violeta", en="b", t=0.5), P("regla aprendida", 1400, 560, "violeta", 44, en="b", t=0.8))

# ---------------- 2. El modelo ajusta parámetros (ilustrativo)
PXX = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
PYY = np.array([2.1, 3.0, 3.4, 4.6, 5.1, 6.3, 6.6, 7.9])
b1, b0 = np.polyfit(PXX, PYY, 1)
xl = [0.5, 8.5]
mal = [4.0, 4.0]
medio = [1.5, 5.5]
bien = [float(b0 + b1 * 0.5), float(b0 + b1 * 8.5)]


def mse(a, b):
    ys = a + (b - a) * (PXX - 0.5) / 8.0
    return float(np.mean((PYY - ys) ** 2))


m1, m2, m3 = mse(*mal), mse(*medio), mse(*bien)
GX = dict(x0=240, y0=260, x1=1280, y1=860, xmin=0, xmax=9, ymin=0, ymax=9, xticks=[], yticks=[], tam_txt=30)


def grafico(en=None):
    return G(240, 260, 1280, 860, 0, 9, 0, 9,
             [dict(xs=[float(a) for a in PXX], ys=[float(a) for a in PYY], color="violeta", linea=False, marcadores=13),
              dict(xs=xl, ys=mal, color="rojo", grosor=7, lapiz=False)],
             xticks=[], yticks=[], id="gf", en=en, ent="fade")


v.b("Veámoslo con un ejemplo muy simple. [[a]]Estos puntos son los datos: cada uno es un caso con su resultado. [[b]]El modelo parte con una recta cualquiera, casi plana.",
    CL(), grafico("a"), T("ejemplo ilustrativo", 1600, 300, tam=34, c="gris", w="sb", en="a", t=0.2),
    g("pr", P("una primera recta cualquiera", 1600, 480, "rojo", 40, en="b", mw=480)), sec="2. El modelo ajusta parámetros")
v.b("Ahora mide el error: [[a]]qué tan lejos queda cada punto de la recta. El error cuadrático medio de esta recta es [[b]]" + f"{m1:.2f}" + ". Muy alto.",
    *[L(float(240 + (x - 0) / 9 * 1040), float(860 - (y) / 9 * 600), float(240 + (x) / 9 * 1040), float(860 - 4.0 / 9 * 600), "ambar", 5, id=f"er{i}", en="a", t=0.08 * i, dur=0.4) for i, (x, y) in enumerate(zip(PXX, PYY))],
    XG("pr", dur=0.3), P("error medio", 1600, 640, "ambar", 40, en="a"), N(m1, 1600, 760, "{:.2f}", 90, "rojo", en="b", dur=0.9, id="n1"))
v.b("Entonces el modelo ajusta sus parámetros, la pendiente y el punto de partida, [[a]]y la recta se acerca a los datos. [[b]]El error baja a " + f"{m2:.2f}" + ".",
    X("er0", "er1", "er2", "er3", "er4", "er5", "er6", "er7", dur=0.3), U("gf", **{"series.1.ys": medio}, en="a", dur=1.2),
    U("n1", valor=m2, color="ambar", en="b", dur=0.9))
v.b("Sigue ajustando, [[a]]hasta que casi no puede mejorar. [[b]]Ahora el error es " + f"{m3:.2f}" + ". Esa es la recta que mejor describe los ejemplos.",
    U("gf", **{"series.1.ys": bien, "series.1.color": "verde"}, en="a", dur=1.2), U("n1", valor=m3, color="verde", en="b", dur=0.9))
v.b("Fíjate que nadie le dijo la regla. [[a]]Solo vio ejemplos y fue corrigiendo su error. [[b]]Eso es aprender, en Machine Learning.",
    P("nadie le dio la regla", 960, 940, "teal", 46, en="a"), P("aprender = reducir el error", 1600, 930, "violeta", 40, en="b", mw=420))

# ---------------- 3. El ciclo
v.b("Ese ciclo de ajustar y medir se repite siempre. [[a]]Primero, los datos. [[b]]Luego, el modelo hace sus predicciones. [[c]]Se mide el error. [[d]]Y el modelo mejora. Después, otra vuelta.",
    CL(),
    *[[Bc(x, 480, 340, 190, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 480, tam=50, c=c, en=k, t=0.1)]
      for x, c, n, k in ((330, "teal", "datos", "a"), (780, "violeta", "modelo", "b"), (1230, "rojo", "error", "c"), (1680, "verde", "mejorar", "d"))],
    [L(510 + 450 * i, 480, 600 + 450 * i, 480, "gris", 6, True, en=k, t=0.2) for i, k in enumerate("bcd")],
    L(1680, 620, 330, 620, "grisclaro", 6, True, en="d", t=0.5, dur=1.0), sec="3. El ciclo del aprendizaje")
v.b("Piensa un segundo: [[a]]en este ciclo, ¿con qué datos conviene medir el error final, con los mismos que el modelo ya vio, o con otros nuevos?",
    T("¿medir con los mismos datos…", 960, 810, tam=54, c="tinta", en="a"), T("…o con datos nuevos?", 960, 900, tam=54, c="violeta", en="a", t=0.4), pausa=0.6)
v.b("Con datos nuevos. Si mides con los mismos, solo compruebas que se los aprendió. [[a]]Esto lo vimos en la clase treinta y lo vamos a usar todo el rato.",
    CL(), P("datos nuevos", 960, 400, "verde", 80, ent="subir"), K(960, 560, "check", 130, en=None, t=0.3), P("lo vimos en la clase 30", 960, 800, "teal", 50, en="a"))

# ---------------- 4. Las familias
v.b("Según los datos que tengas, el problema cae en una familia. [[a]]Si tienes la respuesta correcta para cada ejemplo, es supervisado. [[b]]Si no la tienes, es no supervisado. [[c]]Y si hay un entorno que da recompensas, es refuerzo.",
    CL(),
    [[Bc(x, 500, 520, 400, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(n, x, 360, tam=50, c=c, en=k, t=0.1), T(d, x, 530, tam=36, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "violeta", "Supervisado", "tengo la respuesta correcta", "a"), (960, "teal", "No supervisado", "no hay respuesta", "b"),
                           (1560, "ambar", "Refuerzo", "hay un entorno con recompensas", "c"))],
    sec="4. Las familias de problemas")
v.b("En el supervisado, [[a]]si la respuesta es una categoría, como spam o no spam, es clasificación. [[b]]Si es un número, como cuánto gastará un cliente, es regresión.",
    CL(),
    Bc(960, 250, 640, 130, "violetaclaro", "violeta", 5, 30, ent="subir"), T("Supervisado", 960, 250, tam=56, c="violeta"),
    L(800, 320, 560, 470, "gris", 6, True, en="a"), L(1120, 320, 1360, 470, "gris", 6, True, en="b"),
    Bc(560, 650, 640, 330, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("Clasificación", 560, 540, tam=50, c="violeta", en="a", t=0.1), T("categoría", 560, 640, tam=40, c="gris", w="sb", en="a", t=0.2), P("¿spam o no spam?", 560, 760, "violeta", 40, en="a", t=0.3),
    Bc(1360, 650, 640, 330, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("Regresión", 1360, 540, tam=50, c="azul", en="b", t=0.1), T("número", 1360, 640, tam=40, c="gris", w="sb", en="b", t=0.2), P("¿cuánto gastará?", 1360, 760, "azul", 40, en="b", t=0.3))
v.b("En clase, la profesora lo explicó con los captcha. [[a]]Cuando marcas cuáles imágenes tienen un semáforo, [[b]]le estás dando al modelo ejemplos con la respuesta correcta. Eso es aprendizaje supervisado.",
    CL(),
    Bc(960, 470, 760, 470, "blanco", "grisclaro", 5, 30, sombra=True, ent="subir", t=0.2),
    *[Bc(700 + 260 * (i % 3), 360 + 190 * (i // 3), 200, 160, "grisclaro" if i not in (1, 4) else "ambarclaro", "linea" if i not in (1, 4) else "ambar", 3, 14, en="a", t=0.08 * i) for i in range(6)],
    P("marca los semáforos", 960, 770, "ambar", 44, en="a", t=0.3), K(960, 880, "check", 70, en="b"), P("ejemplos etiquetados", 960, 940, "violeta", 40, en="b", t=0.4))
v.b("Sin respuesta correcta, [[a]]el modelo busca patrones ocultos. Por ejemplo, agrupar clientes parecidos, [[b]]o resumir muchas variables en pocas. Es como ordenar un cajón de calcetines sin que nadie te diga qué grupos hay.",
    CL(),
    Pt([(14, [300, 320, 700, 520], "azul"), (14, [820, 320, 1220, 520], "ambar"), (14, [1340, 320, 1740, 520], "teal")], 10, id="pg", en="a", t=0.2),
    P("clustering: formar grupos", 960, 640, "teal", 50, en="a", t=0.5), P("reducción de dimensionalidad: resumir variables", 960, 780, "azul", 46, en="b"),
    sec=None)
v.b("Y el refuerzo es como entrenar a un perrito. [[a]]Haces una acción, [[b]]recibes un premio o un castigo, [[c]]y ajustas lo que haces la próxima vez.",
    CL(),
    *[[Bc(x, 480, 380, 200, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 480, tam=46, c=c, en=k, t=0.1)]
      for x, c, n, k in ((380, "violeta", "acción", "a"), (960, "verde", "premio o castigo", "b"), (1540, "teal", "ajusto", "c"))],
    [L(580, 480, 760, 480, "gris", 6, True, en="b"), L(1160, 480, 1340, 480, "gris", 6, True, en="c")], P("aprende por ensayo y error", 960, 800, "ambar", 50, en="c", t=0.4))
v.b("Y un sistema de recomendación, como el de Netflix, mezcla familias. [[a]]Una parte usa etiquetas, lo que tú eliges, [[b]]y otra descubre grupos de películas parecidas. Por eso se dice que es un modelo mixto.",
    CL(), Bc(960, 300, 620, 140, "tealclaro", "teal", 5, 30, ent="subir"), T("Netflix", 960, 300, tam=60, c="teal"),
    Bc(560, 620, 640, 260, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("supervisado", 560, 580, tam=48, c="violeta", en="a", t=0.1), T("lo que eliges", 560, 670, tam=36, c="gris", w="sb", en="a", t=0.2),
    Bc(1360, 620, 640, 260, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"), T("no supervisado", 1360, 580, tam=48, c="teal", en="b", t=0.1), T("grupos parecidos", 1360, 670, tam=36, c="gris", w="sb", en="b", t=0.2))

# ---------------- 5. Ejemplo resuelto: cuatro situaciones de una tienda
SIT = [("¿Es fraude esta compra online? (hay historial marcado)", "Clasificación", "violeta", "la etiqueta es sí / no"),
       ("¿Cuánto gastará un cliente en diciembre?", "Regresión", "azul", "la salida es un número"),
       ("Descubrir tipos de clientes sin categorías previas", "Clustering", "teal", "no hay etiqueta"),
       ("Un sistema prueba precios y aprende de las ventas", "Refuerzo", "ambar", "acción, recompensa, ajuste")]


def sit(i, en, t=0.0):
    txt, tipo, c, por = SIT[i]
    y = 330 + 150 * i
    return [Bc(560, y, 940, 120, "blanco", "grisclaro", 4, 24, sombra=True, id=f"s{i}", en=en, ent="subir", t=t), T(txt, 560, y, tam=33, c="tinta", w="sb", mw=860, en=en, t=t + 0.1),
            L(1050, y, 1160, y, "gris", 5, True, en=en, t=t + 0.5, dur=0.3), Bc(1480, y, 480, 120, c + "claro", c, 4, 24, en=en, t=t + 0.7, ent="pop"),
            T(tipo, 1480, y - 22, tam=40, c=c, en=en, t=t + 0.8), T(por, 1480, y + 28, tam=28, c="gris", w="sb", en=en, t=t + 0.9)]


v.b("Pongamos todo a prueba con cuatro situaciones de una tienda. [[a]]Primera: saber si una compra online es fraude, y hay historial marcado. [[b]]Hay etiqueta, y es sí o no: clasificación.",
    CL(), P("cuatro situaciones de una tienda", 960, 170, "teal", 46), sit(0, "a"), sec="5. Un ejemplo resuelto")
v.b("Segunda: estimar cuánto gastará un cliente en diciembre. [[a]]Hay etiqueta, pero es un número: regresión.",
    sit(1, "a"))
v.b("Tercera: descubrir tipos de clientes, sin categorías previas. [[a]]No hay etiqueta, y queremos formar grupos: clustering.",
    sit(2, "a"))
v.b("Y cuarta: un sistema prueba precios y aprende de las ventas. [[a]]Acción, recompensa y ajuste: refuerzo.",
    sit(3, "a"))
v.b("Ahora tú. Piensa un segundo: [[a]]resumir cuarenta indicadores de una encuesta en tres factores, para poder graficarlos, ¿qué tipo de problema es?",
    CL(), T("40 indicadores  →  3 factores", 960, 420, tam=72, c="tinta", en="a"), T("¿qué tipo de problema?", 960, 600, tam=72, c="violeta", en="a", t=0.4), pausa=0.6)
v.b("Es reducción de dimensionalidad. [[a]]No hay etiqueta, y lo que buscamos es resumir muchas variables en pocas.",
    CL(), P("reducción de dimensionalidad", 960, 420, "azul", 70, ent="subir"), T("sin etiqueta  ·  resumir variables", 960, 620, tam=50, c="gris", w="sb", en="a"))

# ---------------- 6. El error típico
v.b("El error típico: [[a]]elegir primero el algoritmo, y recién después preguntarse qué se quiere predecir. [[b]]Al revés. Primero el problema, luego la herramienta.",
    CL(), P("primero el algoritmo", 560, 420, "rojo", 56, en="a"), K(560, 560, "cruz", 100, en="a", t=0.3),
    P("primero el problema", 1360, 420, "verde", 56, en="b"), K(1360, 560, "check", 100, en="b", t=0.3), sec="6. Error típico y para llevar")
v.b("Una pregunta práctica: [[a]]¿qué etiqueta tendrías que registrar hoy, para poder entrenar un modelo mañana? [[b]]Si no la guardas, no habrá ejemplos con respuesta.",
    CL(), T("¿qué etiqueta registro hoy…", 960, 420, tam=64, c="tinta", en="a"), T("…para entrenar mañana?", 960, 530, tam=64, c="violeta", en="a", t=0.4), P("sin etiqueta guardada, no hay ejemplos", 960, 780, "ambar", 48, en="b"))
v.b("Para llevarte: [[a]]el tipo de problema decide el modelo. [[b]]Etiqueta categórica, clasificación. Etiqueta numérica, regresión. [[c]]Sin etiqueta, clustering o reducción. Y con recompensas, refuerzo.",
    CL(), P("el tipo de problema decide el modelo", 960, 250, "violeta", 54, en="a"),
    *[P(n, 960, y, c, 44, en="b" if i < 2 else "c") for i, (n, y, c) in enumerate([("categoría  →  clasificación", 440, "violeta"), ("número  →  regresión", 560, "azul"), ("sin etiqueta  →  clustering o reducción", 680, "teal"), ("recompensas  →  refuerzo", 800, "ambar")])])
v.b("Vamos con la segunda parada: [[a]]la clasificación, con la regresión logística.",
    CL(), Bc(960, 480, 620, 260, "violetaclaro", "violeta", 6, 40, id="p1", ent="pop", en="a"),
    T("2", 720, 480, tam=120, c="violeta", en="a", t=0.1), T("Clasificación: regresión logística", 1040, 480, tam=44, c="tinta", mw=420, en="a", t=0.15), sec="7. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats", "mse", m1, m2, m3)
