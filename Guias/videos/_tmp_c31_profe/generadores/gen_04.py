import sys, json, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_04_regresion.json"
D = json.load(open(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe\datos31.json", encoding="utf-8"))
v = V("Regresión: predecir un número")

v.portada("Regresión: predecir un número", "Cuánto gastará un cliente, y cómo medir el error",
          "Regresión. Cómo predecir un número, y cómo saber cuánto se equivoca tu modelo.")

# ---- 1. Hook
v.b("Un cliente nuevo se une al gimnasio. [[a]]No te preguntas si cancelará. [[b]]Te preguntas cuánto va a gastar al mes. [[c]]La respuesta ya no es sí o no: es un número.",
    CL(), T("cliente nuevo", 960, 300, tam=60, c="gris", w="sb", ent="subir", t=0.2), P("¿cancelará?  sí / no", 560, 560, "violeta", 54, en="a"),
    P("¿cuánto gastará?", 1360, 560, "azul", 64, en="b"), N(63, 1360, 740, "{:.0f}", 130, "azul", pre="$ ", suf=" mil", en="c", dur=1.0), sec="1. Predecir un número")
v.b("Cuando lo que predices es un valor continuo, el problema es de regresión. [[a]]Es como estimar el precio de un taxi: [[b]]depende de los kilómetros y de los minutos de espera.",
    CL(), P("valor continuo  =  regresión", 960, 300, "azul", 62, ent="subir", t=0.2),
    Bc(560, 600, 600, 300, "blanco", "teal", 5, 30, sombra=True, en="a", ent="subir"), T("taxi", 560, 520, tam=56, c="teal", en="a", t=0.1), T("precio = base + km + espera", 560, 630, tam=36, c="tinta", w="sb", mw=520, en="b"),
    Bc(1360, 600, 600, 300, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("cliente", 1360, 520, tam=56, c="azul", en="b", t=0.1), T("gasto = base + antigüedad + uso", 1360, 630, tam=36, c="tinta", w="sb", mw=520, en="b", t=0.3))

# ---- 2. El modelo
v.b("En el caso de Club Activo, se predice el gasto mensual, en miles de pesos chilenos. [[a]]Con dos variables: la antigüedad, en meses, [[b]]y el uso mensual del gimnasio.",
    CL(), T("gasto mensual", 960, 260, tam=64, c="azul", ent="subir", t=0.2), P("en miles de CLP", 960, 360, "gris", 40, t=0.4),
    Bc(560, 640, 520, 240, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("antigüedad", 560, 600, tam=52, c="violeta", en="a", t=0.1), T("meses", 560, 680, tam=36, c="gris", w="sb", en="a", t=0.2),
    Bc(1360, 640, 520, 240, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"), T("uso", 1360, 600, tam=52, c="teal", en="b", t=0.1), T("veces al mes", 1360, 680, tam=36, c="gris", w="sb", en="b", t=0.2), sec="2. El modelo de regresión")
v.b("El modelo ajusta una función de esas variables. [[a]]Un punto de partida, [[b]]más un peso por la antigüedad, [[c]]más un peso por el uso.",
    CL(), T("gasto  =  b₀  +  b₁·antigüedad  +  b₂·uso", 960, 360, tam=64, c="tinta", t=0.2),
    P("punto de partida", 360, 560, "gris", 40, en="a"), P("peso de la antigüedad", 960, 560, "violeta", 40, en="b"), P("peso del uso", 1560, 560, "teal", 40, en="c"),
    P("los pesos los aprende el modelo con ejemplos", 960, 820, "azul", 46, en="c", t=0.6))
v.b("Y aquí hay un detalle curioso de este ejemplo. [[a]]Los datos no son reales: se generaron con una regla oculta. [[b]]El gasto es veinticinco, más cero coma nueve por la antigüedad, más tres coma dos por el uso, más ruido.",
    CL(), P("datos simulados con una regla oculta", 960, 270, "ambar", 52, en="a"),
    T("gasto = 25 + 0.9·antigüedad + 3.2·uso + ruido", 960, 470, tam=60, c="tinta", en="b"), T("el modelo no conoce esta fórmula", 960, 640, tam=44, c="gris", w="sb", en="b", t=1.0))
v.b("El ruido es lo que hace distintos a dos clientes con la misma antigüedad y el mismo uso. [[a]]Aquí, el ruido tiene desviación estándar de doce mil pesos.",
    CL(), T("mismo uso, misma antigüedad", 960, 300, tam=52, c="tinta", ent="subir", t=0.2), Bc(560, 560, 380, 220, "blanco", "azul", 5, 28, sombra=True, ent="subir", t=0.3), T("$ 58 mil", 560, 560, tam=64, c="azul", t=0.4),
    Bc(1360, 560, 380, 220, "blanco", "azul", 5, 28, sombra=True, ent="subir", t=0.5), T("$ 71 mil", 1360, 560, tam=64, c="azul", t=0.6), T("≠", 960, 560, tam=90, c="ambar", t=0.5),
    P("ruido: desviación estándar ≈ 12", 960, 850, "ambar", 50, en="a"), T("cifras ilustrativas", 960, 700, tam=30, c="gris", w="sb", t=0.8))
v.b("Después de aprender, el modelo descubre los pesos, sin que nadie se los diga. [[a]]El punto de partida: veintiséis coma ocho, contra veinticinco real. [[b]]Antigüedad: cero coma noventa y tres, contra cero coma nueve. [[c]]Y uso: dos coma noventa y ocho, contra tres coma dos.",
    CL(),
    T("regla oculta", 900, 260, tam=40, c="gris", w="sb"), T("lo que aprendió", 1420, 260, tam=40, c="violeta", w="sb"),
    *[[T(n, 360, y, tam=44, c="tinta", w="sb", anc="lm", en=k), N(real, 900, y, fmt, 80, "gris", en=k, dur=0.8), N(apr, 1420, y, fmt, 80, "violeta", en=k, dur=0.9, t=0.3)]
      for n, y, real, apr, fmt, k in (("punto de partida", 420, 25, 26.76, "{:.1f}", "a"), ("antigüedad", 600, 0.9, 0.929, "{:.2f}", "b"), ("uso", 780, 3.2, 2.981, "{:.2f}", "c"))])
v.b("Se parecen mucho. [[a]]Pero ojo: «redescubrir» la fórmula solo es posible porque los datos se simularon con ella. [[b]]En un caso real no existe una fórmula verdadera con la que comparar.",
    CL(), P("se parecen mucho", 960, 330, "verde", 64, ent="subir", t=0.2), K(1560, 330, "check", 80, en=None, t=0.5),
    P("solo se puede porque se simuló", 960, 560, "ambar", 54, en="a"), P("en la vida real no hay fórmula verdadera", 960, 760, "rojo", 54, en="b"))

# ---- 3. Cómo medir el error
v.b("¿Y cómo sabemos si el modelo predice bien? [[a]]Mirando el error: la diferencia entre lo real y lo predicho. [[b]]Un error negativo significa que predijo de más. Uno positivo, que predijo de menos.",
    CL(), T("error  =  real  −  predicho", 960, 270, tam=64, c="tinta", en="a"),
    Bc(560, 560, 500, 200, "blanco", "rojo", 5, 28, sombra=True, en="b", ent="subir"), T("error negativo", 560, 520, tam=44, c="rojo", en="b", t=0.1), T("predijo de más", 560, 600, tam=36, c="gris", w="sb", en="b", t=0.2),
    Bc(1360, 560, 500, 200, "blanco", "azul", 5, 28, sombra=True, en="b", ent="subir", t=0.5), T("error positivo", 1360, 520, tam=44, c="azul", en="b", t=0.6), T("predijo de menos", 1360, 600, tam=36, c="gris", w="sb", en="b", t=0.7),
    sec="3. Cómo medir el error")
v.b("Si sumas errores con signo, unos cancelan a otros. [[a]]Por eso se elevan al cuadrado, y luego se promedian. [[b]]Eso es el error cuadrático medio, el {MSE|eme ese e}.",
    CL(), T("menos 3  +  más 3  =  0", 960, 330, tam=64, c="tinta", ent="subir", t=0.2), P("¡parece que no hay error!", 960, 480, "rojo", 48, t=0.6),
    T("MSE = promedio de (error)²", 960, 700, tam=60, c="violeta", en="a"), P("al cuadrado: ya no se cancelan", 960, 840, "teal", 46, en="b"))
v.b("Su raíz cuadrada es el {RMSE|erre eme ese e}. [[a]]Y tiene una ventaja: está en las mismas unidades del gasto, [[b]]en este caso, miles de pesos. Se lee como el error típico.",
    CL(), T("RMSE = √MSE", 960, 330, tam=80, c="ambar", ent="subir", t=0.2), P("mismas unidades que el gasto", 960, 560, "ambar", 56, en="a"), P("miles de CLP", 960, 720, "gris", 50, en="b"))
v.b("Y el R cuadrado. [[a]]Compara tu error contra el que tendrías si siempre predijeras el promedio. [[b]]Es uno menos el error del modelo, entre el error total. Es la proporción de la variación que se explica.",
    CL(), T("R² = 1 − SSE / SST", 960, 300, tam=80, c="violeta", ent="subir", t=0.2),
    Bc(560, 600, 640, 300, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("SSE", 560, 520, tam=56, c="rojo", en="a", t=0.1), T("error del modelo", 560, 620, tam=36, c="gris", w="sb", en="a", t=0.2),
    Bc(1360, 600, 640, 300, "blanco", "gris", 5, 30, sombra=True, en="b", ent="subir"), T("SST", 1360, 520, tam=56, c="gris", en="b", t=0.1), T("error de predecir siempre el promedio", 1360, 620, tam=34, c="gris", w="sb", mw=520, en="b", t=0.2))
v.b("R cuadrado igual a uno sería perfecto. Cero significa que no mejoras al promedio. [[a]]Y puede ser negativo, si tu modelo es peor que predecir el promedio.",
    CL(), *[[T(str(x), 360 + i * 600, 420, tam=100, c=c), T(d, 360 + i * 600, 560, tam=40, c="gris", w="sb", mw=420)] for i, (x, c, d) in enumerate((("1", "verde", "perfecto"), ("0", "ambar", "igual al promedio"), ("< 0", "rojo", "peor que el promedio")))],
    P("R² puede ser negativo", 960, 800, "rojo", 50, en="a"))

# ---- 4. Ejemplo resuelto a mano
v.b("Hagamos un ejemplo pequeño, con cuatro clientes, y números inventados para practicar. [[a]]Los valores reales: cincuenta, sesenta, setenta y ochenta. [[b]]Las predicciones: cincuenta y dos, sesenta y tres, sesenta y ocho y ochenta y tres.",
    CL(),
    T("ejemplo ilustrativo", 1620, 190, tam=32, c="gris", w="sb", t=0.2),
    T("real", 360, 330, tam=40, c="gris", w="sb", anc="rm", t=0.2), T("predicho", 360, 440, tam=40, c="violeta", w="sb", anc="rm", t=0.3),
    *[[T(str(r), 560 + 300 * i, 330, tam=56, c="tinta", en="a", t=0.1 * i), T(str(p), 560 + 300 * i, 440, tam=56, c="violeta", en="b", t=0.1 * i)] for i, (r, p) in enumerate(((50, 52), (60, 63), (70, 68), (80, 83)))],
    sec="4. Un ejemplo resuelto a mano")
ERR = [-2, -3, 2, -3]
v.b("Los errores, real menos predicho, son: [[a]]menos dos, menos tres, dos y menos tres. [[b]]Al cuadrado: cuatro, nueve, cuatro y nueve. [[c]]La suma, el SSE, es veintiséis.",
    T("error", 360, 550, tam=40, c="rojo", w="sb", anc="rm"), *[T(str(e).replace("-", "−"), 560 + 300 * i, 550, tam=56, c="rojo", en="a", t=0.15 * i) for i, e in enumerate(ERR)],
    T("error²", 360, 660, tam=40, c="ambar", w="sb", anc="rm"), *[T(str(e * e), 560 + 300 * i, 660, tam=56, c="ambar", en="b", t=0.15 * i) for i, e in enumerate(ERR)],
    T("SSE = 26", 960, 800, tam=80, c="rojo", en="c"))
v.b("El MSE es veintiséis entre cuatro: seis coma cinco. [[a]]El RMSE es la raíz de seis coma cinco: dos coma cincuenta y cinco. [[b]]En promedio, el modelo se equivoca en unos dos coma seis puntos.",
    T("MSE = 26 / 4 = 6.5", 960, 910, tam=56, c="violeta", t=0.1), T("RMSE = √6.5 = 2.55", 1500, 800, tam=48, c="ambar", en="a"), P("error típico ≈ 2.6", 1500, 910, "ambar", 40, en="b"))
v.b("Para el R cuadrado, necesitamos el error total. [[a]]El promedio de los reales es sesenta y cinco. Las distancias al cuadrado suman quinientos. [[b]]Entonces R cuadrado es uno menos veintiséis entre quinientos: cero coma noventa y cinco.",
    CL(), T("promedio real = 65", 960, 270, tam=56, c="gris", w="sb", en="a"), T("SST = 225 + 25 + 25 + 225 = 500", 960, 420, tam=56, c="tinta", en="a", t=0.4),
    T("R² = 1 − 26 / 500", 960, 620, tam=64, c="violeta", en="b"), N(0.948, 960, 800, "{:.2f}", 140, "violeta", en="b", t=0.5, dur=0.9))
v.b("Piensa un segundo: [[a]]si el modelo siempre predijera sesenta y cinco, el promedio, ¿cuánto sería el R cuadrado?",
    CL(), T("predecir siempre 65", 960, 400, tam=72, c="tinta", en="a"), T("R² = ?", 960, 600, tam=100, c="violeta", en="a", t=0.5), pausa=0.6)
v.b("Cero. [[a]]Porque entonces el error del modelo es igual al error total, y uno menos uno da cero.",
    CL(), N(0, 960, 420, "{:.0f}", 220, "ambar", dur=0.3), P("1 − SST/SST = 0", 960, 700, "ambar", 56, en="a"))

# ---- 5. Resultados de la clase
v.b("Volvamos a Club Activo. Se entrenó con una parte de los clientes, y se evaluó con otros doscientos cuarenta, que el modelo no vio. [[a]]Estos puntos comparan el gasto real con el predicho. Si fuera perfecto, caerían sobre la diagonal.",
    CL(),
    G(300, 200, 1160, 880, 20, 120, 20, 120, [dict(xs=[20, 120], ys=[20, 120], color="rojo", grosor=5, guion=True),
                                              dict(xs=sorted(D["reg"]["real"]), ys=[p for _, p in sorted(zip(D["reg"]["real"], D["reg"]["pred"]))], color="azul", linea=False, marcadores=8)],
      xticks=[20, 40, 60, 80, 100], yticks=[20, 40, 60, 80, 100], yfmt="{:.0f}", xlabel="gasto real (miles de CLP)", id="gp", t=0.2),
    T("gasto predicho", 300, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.3),
    P("predicción perfecta", 1560, 380, "rojo", 40, en="a"), sec="5. Los resultados de la clase")
v.b("Los puntos se reparten alrededor de la diagonal, [[a]]pero con bastante dispersión. [[b]]El error cuadrático medio es ciento cuarenta y cuatro coma nueve, [[c]]y el RMSE, doce coma cero mil pesos.",
    R(730, 540, 900, 160, "ambar", 60, id="rd", en="a", ent="fade"), N(144.9, 1560, 560, "{:.1f}", 80, "violeta", pre="MSE ", en="b", dur=0.9), N(12.0, 1560, 700, "{:.1f}", 80, "ambar", pre="RMSE ", en="c", dur=0.9))
v.b("El gasto medio es sesenta y tres coma dos mil pesos. [[a]]Así que un error típico de doce es cerca del diecinueve por ciento del gasto medio.",
    X("rd", dur=0.3), P("gasto medio: 63.2", 1560, 840, "teal", 40, mw=500), P("RMSE ≈ 19 % del gasto medio", 1560, 930, "ambar", 40, en="a", mw=560))
v.b("Y el R cuadrado en la prueba es cero coma quinientos cuarenta y ocho. [[a]]En entrenamiento fue cero coma quinientos setenta y dos, casi igual. [[b]]Eso indica que no hay sobreajuste.",
    CL(), T("R² en prueba", 560, 330, tam=46, c="gris", w="sb"), N(0.548, 560, 480, "{:.3f}", 160, "violeta", dur=1.0, t=0.2), T("R² en entrenamiento", 1360, 330, tam=46, c="gris", w="sb", en="a"), N(0.572, 1360, 480, "{:.3f}", 160, "azul", en="a", dur=1.0),
    P("casi iguales: sin sobreajuste", 960, 780, "verde", 54, en="b"))
v.b("¿Es poco, un R cuadrado de cero coma cincuenta y cinco? [[a]]No necesariamente. El ruido que pusimos tiene desviación de doce, y el gasto tiene desviación de diecinueve. [[b]]Con tanto ruido, el techo del R cuadrado ronda cero coma sesenta.",
    CL(), T("ruido  sd 12", 560, 330, tam=56, c="ambar", en="a"), T("gasto  sd 19", 1360, 330, tam=56, c="azul", en="a", t=0.3),
    T("1 − (12 / 19)²  ≈  0.60", 960, 560, tam=72, c="violeta", en="b"), P("cálculo propio, para dimensionar", 960, 700, "gris", 36, en="b", t=0.4), P("el techo lo pone el ruido, no el modelo", 960, 860, "teal", 52, en="b", t=0.6))
v.b("Y mira los residuos, que son los errores de cada cliente. [[a]]Se centran alrededor de cero. [[b]]Tienen una desviación estándar cercana a doce, justo el ruido que pusimos.",
    CL(),
    *[Bx(300 + i * 70, 860 - h * 11, 300 + i * 70 + 62, 860, "azul" if 4 <= i <= 9 else "azulclaro", None, 0, 6, en="a", t=0.05 * i, grow="-y" if False else None, ent="subir") for i, h in enumerate(D["reg"]["hist"])],
    L(300, 870, 1280, 870, "tinta", 4, t=0.1), T("−42", 300, 910, tam=32, c="gris"), T("0", 300 + 7 * 70, 910, tam=32, c="gris"), T("+42", 1280, 910, tam=32, c="gris"),
    L(300 + 7 * 70, 360, 300 + 7 * 70, 870, "rojo", 4, guion=True, en="a"), T("error del cliente (real − predicho)", 790, 970, tam=34, c="gris", w="sb"),
    P("centrados en 0", 1560, 460, "verde", 48, en="a"), P("sd ≈ 12", 1560, 580, "ambar", 48, en="b"))

# ---- 6. Error típico y para llevar
v.b("El error típico en regresión es dejarse llevar por un solo número. [[a]]Un R cuadrado alto no basta. [[b]]Mira también el error en las unidades del problema: ¿doce mil pesos es mucho o poco para este negocio?",
    CL(), P("un R² alto no basta", 960, 330, "ambar", 70, en="a"), P("¿12 mil pesos es mucho o poco?", 960, 600, "azul", 60, en="b"), T("depende del negocio", 960, 760, tam=48, c="gris", w="sb", en="b", t=0.5), sec="6. Error típico y para llevar")
v.b("Y la buena noticia: el flujo es el mismo que en clasificación. [[a]]Dividir en entrenamiento y prueba, ajustar solo con el entrenamiento, [[b]]y evaluar en datos nuevos. Solo cambia la métrica.",
    CL(),
    Bc(520, 480, 640, 420, "blanco", "violeta", 5, 30, sombra=True, ent="subir", t=0.2), T("clasificación", 520, 330, tam=48, c="violeta", t=0.3), T("matriz de confusión", 520, 470, tam=42, c="tinta", w="sb", en="b"),
    Bc(1400, 480, 640, 420, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("regresión", 1400, 330, tam=48, c="azul", en="b", t=0.1), T("MSE, RMSE y R²", 1400, 470, tam=42, c="tinta", w="sb", en="b", t=0.3),
    P("dividir y evaluar en datos no vistos no cambia", 960, 800, "teal", 46, en="a"))
v.b("Para llevarte: [[a]]regresión es predecir un número. [[b]]El RMSE te dice el error típico, en las unidades del problema. [[c]]Y un R cuadrado moderado no es una falla, si el ruido es grande.",
    CL(), P("regresión = predecir un número", 960, 330, "azul", 54, en="a"), P("RMSE: error típico en tus unidades", 960, 540, "ambar", 54, en="b"), P("R² moderado, con mucho ruido, es normal", 960, 750, "violeta", 54, en="c"))
v.b("Vamos con la cuarta parada: [[a]]qué pasa cuando no hay etiquetas, con K-Means y PCA.",
    CL(), Bc(960, 480, 620, 260, "verdeclaro", "verde", 6, 40, id="p1", ent="pop", en="a"),
    T("4", 720, 480, tam=120, c="verde", en="a", t=0.1), T("Sin etiquetas: K-Means y PCA", 1040, 480, tam=44, c="tinta", mw=420, en="a", t=0.15), sec="7. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
