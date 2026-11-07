import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_08_enfoque.json"
v = V("Elegir el enfoque y el flujo honesto")

v.portada("Elegir el enfoque y el flujo honesto", "Primero el problema, luego el algoritmo, y todo dentro de un Pipeline",
          "Cómo elegir el enfoque, la diferencia entre {Machine Learning|mashín lérnin} y {Deep Learning|dip lérnin}, y cómo armar un flujo honesto.")

# ---- 1. Primero el problema
v.b("Recuerdas la advertencia del Ferrari para ir a comprar pan. [[a]]La forma de no caer en eso es un orden de preguntas. [[b]]Primero el problema. Después el algoritmo.",
    CL(), P("primero el problema", 560, 420, "verde", 66, en="a"), L(840, 420, 1080, 420, "gris", 8, True, en="b"), P("después el algoritmo", 1400, 420, "violeta", 66, en="b"),
    T("no uses IA porque está de moda", 960, 700, tam=52, c="gris", w="sb", en="b", t=0.6), sec="1. Primero el problema, luego el algoritmo")
RUTA = [("¿Tengo ejemplos con la respuesta correcta?", "sí, y es una categoría", "Clasificación", "violeta"), ("", "sí, y es un número", "Regresión", "azul"),
        ("¿No tengo etiquetas?", "quiero descubrir grupos", "Clustering, p. ej. K-Means", "teal"), ("", "quiero simplificar muchas variables", "Reducción, p. ej. PCA", "teal"),
        ("¿Decisiones secuenciales con entorno?", "ensayo y error con recompensa", "Refuerzo", "ambar"), ("¿Imágenes, audio, texto, gran volumen?", "datos complejos", "Deep Learning", "rojo")]


def fila(i, en, t=0.0):
    q, resp, enf, c = RUTA[i]
    y = 290 + 110 * i
    out = [T(resp, 760, y, tam=34, c="tinta", w="sb", mw=480, en=en, t=t), L(1010, y, 1090, y, "gris", 5, True, en=en, t=t + 0.2, dur=0.3),
           Bc(1440, y, 560, 90, c + "claro", c, 4, 22, en=en, t=t + 0.3, ent="pop"), T(enf, 1440, y, tam=32, c=c, mw=520, en=en, t=t + 0.4)]
    if q:
        out.append(T(q, 380, y, tam=32, c="violeta" if c != "rojo" else "rojo", mw=340, id=f"q{i}", en=en, t=t))
    return out


v.b("La tabla de decisión es esta. [[a]]¿Tienes ejemplos con la respuesta correcta? Si es una categoría, clasificación. Si es un número, regresión. [[b]]¿No tienes etiquetas? Si quieres descubrir grupos, clustering. Si quieres simplificar muchas variables, reducción de dimensionalidad.",
    CL(), fila(0, "a"), fila(1, "a", 0.5), fila(2, "b"), fila(3, "b", 0.5))
v.b("¿Hay decisiones secuenciales con un entorno? [[a]]Ensayo y error con recompensa: refuerzo. [[b]]¿Imágenes, audio, texto, gran volumen? Datos complejos: Deep Learning.",
    fila(4, "a"), fila(5, "b"))
v.b("Fíjate en algo. Un problema puede encajar en más de un enfoque. [[a]]La función de la profesora que devuelve un solo enfoque lo trata como si fueran excluyentes, [[b]]pero no lo son: un sistema real puede combinar varios.",
    CL(), P("un solo enfoque", 560, 400, "rojo", 56, en="a"), K(560, 540, "cruz", 90, en="a", t=0.3), P("pueden combinarse", 1360, 400, "verde", 56, en="b"), K(1360, 540, "check", 90, en="b", t=0.3),
    T("(la función recomendar_enfoque del notebook)", 960, 800, tam=32, c="gris", w="sb", en="a", t=0.4))

# ---- 2. Ejemplo resuelto
PROP = [("Predecir ventas semanales con 3 años de historia en una planilla", "Regresión (o series de tiempo)", "pocos miles de filas: ML clásico basta", "azul"),
        ("Clasificar 2 millones de fotos de productos", "Deep Learning", "datos complejos y gran volumen", "rojo"),
        ("Calcular un descuento fijo con una tabla de reglas", "Sin ML", "una regla ya resuelve el problema", "gris")]


def prop_l(i, en=None, t=0.0):
    txt = PROP[i][0]
    y = 360 + 200 * i
    return [Bc(520, y, 880, 150, "blanco", "grisclaro", 4, 24, sombra=True, en=en, ent="subir", t=t), T(txt, 520, y, tam=34, c="tinta", w="sb", mw=800, en=en, t=t + 0.1)]


def prop_r(i, en=None, t=0.0):
    _, enf, por, c = PROP[i]
    y = 360 + 200 * i
    return [L(980, y, 1080, y, "gris", 5, True, en=en, t=t, dur=0.3), Bc(1480, y, 640, 150, c + "claro" if c != "gris" else "linea", c, 4, 24, en=en, t=t + 0.2, ent="pop"),
            T(enf, 1480, y - 25, tam=40, c=c, en=en, t=t + 0.3), T(por, 1480, y + 32, tam=28, c="gris", w="sb", mw=580, en=en, t=t + 0.4)]


v.b("Pongámoslo a prueba con tres propuestas de la dirección de una empresa. Primera: predecir las ventas semanales, con tres años de historia en una planilla. [[a]]¿Qué enfoque dirías?",
    CL(), P("tres propuestas de la dirección", 960, 170, "teal", 44), prop_l(0), sec="2. Ejemplo resuelto: tres propuestas", pausa=0.5)
v.b("Es un valor continuo, con pocos miles de filas tabulares. [[a]]Regresión, o series de tiempo. El ML clásico basta.",
    prop_r(0, "a"))
v.b("Segunda: clasificar dos millones de fotos de productos. [[a]]Son datos complejos, con gran volumen. Ahí sí conviene Deep Learning.",
    prop_l(1), prop_r(1, "a"))
v.b("Tercera: calcular un descuento fijo según una tabla de reglas. [[a]]Una regla ya resuelve el problema. No hace falta Machine Learning.",
    prop_l(2), prop_r(2, "a"))
v.b("Piensa un segundo: [[a]]si una empresa propone Deep Learning para predecir la demanda de quinientos productos con tres años de historia semanal en una planilla, ¿qué le recomiendas?",
    CL(), T("500 productos · 3 años semanales · planilla", 960, 380, tam=52, c="tinta", en="a"), T("¿Deep Learning?", 960, 580, tam=90, c="rojo", en="a", t=0.5), pausa=0.6)
v.b("Que no lo necesita. [[a]]Con datos tabulares y una historia corta, conviene regresión, series de tiempo o modelos de árboles. [[b]]Y como línea base, la demanda de la semana anterior, o un promedio móvil. El modelo debe superarla en el error de prueba.",
    CL(), P("regresión · series de tiempo · árboles", 960, 300, "azul", 52, en="a"), P("línea base: la semana anterior", 960, 520, "ambar", 52, en="b"), P("superarla en el error de prueba", 960, 700, "verde", 52, en="b", t=0.6))

# ---- 3. ML vs DL
v.b("Hablemos de Machine Learning clásico y Deep Learning. [[a]]El Deep Learning es un subconjunto del Machine Learning, basado en redes con varias capas. [[b]]Aprende representaciones directamente de datos complejos, como imágenes, audio o texto.",
    CL(), Bc(960, 520, 1300, 560, "blanco", "azul", 6, 50, ent="subir", t=0.2), T("Machine Learning", 960, 300, tam=52, c="azul", t=0.3),
    Bc(960, 600, 700, 340, "violetaclaro", "violeta", 6, 40, en="a", ent="pop"), T("Deep Learning", 960, 600, tam=54, c="violeta", en="a", t=0.1), T("redes con varias capas", 960, 680, tam=34, c="gris", w="sb", en="a", t=0.2),
    P("imágenes · audio · texto", 960, 880, "teal", 46, en="b"), sec="3. Machine Learning clásico frente a Deep Learning")
CMP = [("Volumen de datos", "suele bastar con miles de registros", "suele necesitar muchísimos más"), ("Hardware", "CPU estándar", "frecuentemente GPU"),
       ("Atributos", "manuales, con conocimiento del negocio", "automáticos, por capas ocultas"), ("Interpretabilidad", "en general mayor", "en general menor"),
       ("Casos de uso", "datos tabulares, scoring", "visión, audio, lenguaje, IA generativa")]


def cmp_(i, en, t=0.0):
    n, a_, b_ = CMP[i]
    y = 390 + 120 * i
    return [T(n, 300, y, tam=34, c="gris", w="sb", anc="lm", en=en, t=t), T(a_, 960, y, tam=32, c="azul", w="sb", mw=420, en=en, t=t + 0.1), T(b_, 1520, y, tam=32, c="violeta", w="sb", mw=420, en=en, t=t + 0.2)]


v.b("Se diferencian en varios criterios. [[a]]El Machine Learning clásico suele bastar con miles de registros, y el Deep Learning necesita muchísimos más. [[b]]El primero corre en una CPU estándar, el segundo frecuentemente en una GPU.",
    CL(), T("ML clásico", 960, 270, tam=48, c="azul", t=0.1), T("Deep Learning", 1520, 270, tam=48, c="violeta", t=0.1), cmp_(0, "a"), cmp_(1, "b"))
v.b("Los atributos, en el clásico, se construyen a mano, con conocimiento del negocio. [[a]]En el Deep Learning los aprenden las capas ocultas. [[b]]Y el clásico suele ser más interpretable.",
    cmp_(2, "a"), cmp_(3, "b"))
v.b("Por eso, para datos tabulares y scoring, el clásico es lo normal. [[a]]El Deep Learning brilla en visión, audio, lenguaje, e inteligencia artificial generativa.",
    cmp_(4, "a"))
v.b("Pero ojo: eso son tendencias, no reglas. [[a]]Un modelo de boosting no es una caja blanca. [[b]]Y existen redes con pocos datos. [[c]]Además, dividir en entrenamiento, validación y prueba vale para cualquier modelo, no solo para redes.",
    CL(), P("tendencias, no reglas", 960, 260, "ambar", 60, ent="subir"), P("boosting no es caja blanca", 960, 470, "rojo", 48, en="a"), P("hay redes con pocos datos", 960, 620, "azul", 48, en="b"),
    P("entrenamiento / validación / prueba: para todo modelo", 960, 800, "verde", 48, en="c"))

# ---- 4. Flujo honesto
v.b("Ahora, el flujo honesto. [[a]]Todo lo que aprende de los datos debe ir dentro de un Pipeline. [[b]]Imputar, estandarizar y ajustar el modelo, en ese orden.",
    CL(),
    *[[Bc(x, 480, 400, 220, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 480, tam=44, c=c, mw=340, en=k, t=0.1)] for x, c, n, k in ((380, "teal", "imputar", "b"), (960, "azul", "estandarizar", "b"), (1540, "violeta", "regresión logística balanced", "b"))],
    [L(590, 480, 750, 480, "gris", 6, True, en="b", t=0.3), L(1170, 480, 1330, 480, "gris", 6, True, en="b", t=0.3)],
    R(960, 480, 1700, 300, "teal", 40, id="rp", en="a", ent="fade"), P("Pipeline", 960, 290, "teal", 54, en="a", t=0.2), sec="4. El flujo honesto")
v.b("Y el notebook de la clase tiene un detalle. [[a]]Imputa con media y mediana, y escala, antes de dividir los datos. [[b]]Es una fuga leve: la prueba influye en el promedio.",
    CL(), P("alerta del material", 960, 240, "ambar", 48, ent="subir"),
    *[[Bc(x, 560, 400, 180, "blanco", c, 5, 28, sombra=True, en="a", ent="subir"), T(n, x, 560, tam=42, c=c, en="a", t=0.1)] for x, c, n in ((420, "teal", "imputar y escalar"), (960, "violeta", "dividir"))],
    L(630, 560, 760, 560, "gris", 6, True, en="a", t=0.3), P("fuga leve: la prueba influye", 1500, 560, "rojo", 40, en="b", mw=500))
v.b("Dentro de un Pipeline, cada paso aprende solo con el entrenamiento. [[a]]Y las métricas de la clase no cambian: setenta y tres coma tres, treinta y nueve coma cuatro, y cincuenta y dos.",
    CL(), P("dentro de un Pipeline: cada paso aprende solo con entrenamiento", 960, 300, "teal", 44, ent="subir", mw=1500),
    *[[Bc(x, 600, 440, 300, "blanco", c, 5, 28, sombra=True, en="a", ent="subir"), T(n, x, 500, tam=38, c="gris", w="sb", en="a", t=0.1), N(val, x, 620, "{:.1f}", 110, c, suf=" %", en="a", t=0.2, dur=0.9)] for x, c, n, val in ((420, "violeta", "accuracy", 73.3), (960, "azul", "precisión", 39.4), (1500, "ambar", "recall", 52.0))])

# ---- 5. Checklist
CHK = [("Defino qué predecir y qué decisión mejora", "etiqueta o no, categoría o número", "violeta"), ("Divido antes de imputar, escalar o ajustar", "todo lo que aprende va en un Pipeline", "teal"),
       ("Comparo contra una línea base", "siempre la clase mayoritaria, o la regla actual", "ambar"), ("Elijo la métrica según el costo del error", "no solo accuracy; matriz completa", "rojo"),
       ("Escalo antes de usar distancias o penalizaciones", "K-Means, Ridge, Lasso", "azul"), ("Interpreto los segmentos con su perfil", "los grupos no traen nombre", "verde")]


def chk(i, en, t=0.0):
    n, c_, col = CHK[i]
    y = 270 + 128 * i
    return [Bc(960, y, 1640, 104, "blanco", col, 4, 24, en=en, ent="subir", t=t), T(n, 220, y - 18, tam=36, c="tinta", w="sb", anc="lm", mw=1100, en=en, t=t + 0.1),
            T(c_, 220, y + 24, tam=28, c="gris", w="sb", anc="lm", mw=1100, en=en, t=t + 0.2), K(1740, y, "check", 56, c=col, en=en, t=t + 0.5)]


v.b("Resumamos el flujo en una lista de verificación. [[a]]Defino qué quiero predecir, y qué decisión mejora. [[b]]Divido antes de imputar, escalar o ajustar. [[c]]Y comparo contra una línea base.",
    CL(), P("checklist del flujo honesto", 960, 150, "violeta", 44), chk(0, "a"), chk(1, "b"), chk(2, "c"), sec="5. La lista de verificación")
v.b("Sigue con [[a]]elegir la métrica según el costo del error. [[b]]Escalar antes de usar distancias o penalizaciones. [[c]]E interpretar los segmentos con su perfil.",
    chk(3, "a"), chk(4, "b"), chk(5, "c"))

# ---- 6. Error y cierre
v.b("Un último error del material. [[a]]El cierre del notebook dice que la próxima sesión, la treinta y dos, es de redes neuronales. [[b]]Pero la Clase treinta y dos es reducción de dimensionalidad: PCA y t-SNE.",
    CL(), P("alerta del material", 960, 230, "ambar", 48, ent="subir"), P("«sesión 32: redes neuronales»", 960, 440, "rojo", 56, en="a"), K(1620, 440, "cruz", 80, en="a", t=0.3),
    P("Clase 32: reducción de dimensionalidad", 960, 700, "verde", 56, en="b"), sec="6. Para llevar")
v.b("Para llevarte: [[a]]primero el problema, luego el algoritmo. [[b]]Para datos tabulares, el ML clásico suele bastar. [[c]]Y todo lo que aprende de los datos va dentro de un Pipeline, con una línea base de comparación.",
    CL(), P("primero el problema, luego el algoritmo", 960, 300, "violeta", 52, en="a"), P("datos tabulares: ML clásico", 960, 500, "azul", 52, en="b"), P("Pipeline + línea base", 960, 700, "teal", 52, en="c"))
v.b("Con esto cerramos las siete paradas. [[a]]Ahora, un repaso final para conectar todo.",
    CL(), Bc(960, 480, 620, 260, "violetaclaro", "violeta", 6, 40, id="p1", ent="pop", en="a"), T("Repaso final", 960, 480, tam=64, c="violeta", en="a", t=0.15), sec="7. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
