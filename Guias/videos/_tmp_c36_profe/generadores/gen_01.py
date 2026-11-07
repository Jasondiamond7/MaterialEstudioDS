import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, flujo, barras_v

OUT = out("guion_c36_01_intuicion.json")
v = V("Aprender de los errores")
SQ = FG["secuencial_rmse"]          # [F0, 1 arbol, 5 arboles, 50 arboles] en miles de USD

v.portada("Aprender de los errores", "Muchos modelos simples, en secuencia, forman uno fuerte",
          "Clase treinta y seis: boosting. La idea es simple y poderosa: muchos modelos débiles, uno detrás de otro, cada uno corrigiendo al anterior.")

# ---- 1. Hook
v.b("Un árbol de decisión con solo dos niveles predice bastante mal el precio de una casa. [[a]]Pero imagina que entrenas un segundo árbol, que se dedica solo a corregir lo que el primero hizo mal. [[b]]Y un tercero, que corrige lo que todavía queda. [[c]]Esa cadena de correcciones se llama boosting.",
    CL(), Bc(420, 420, 340, 170, "blanco", "gris", 5, 26, sombra=True), T("árbol 1", 420, 395, tam=48, c="gris"), T("predice mal", 420, 455, tam=32, c="rojo", w="sb"),
    L(610, 420, 760, 420, "gris", 6, True, en="a"), Bc(960, 420, 340, 170, "blanco", "azul", 5, 26, sombra=True, en="a", t=0.2),
    T("árbol 2", 960, 395, tam=48, c="azul", en="a", t=0.3), T("corrige al 1", 960, 455, tam=32, c="azul", w="sb", en="a", t=0.4),
    L(1150, 420, 1300, 420, "gris", 6, True, en="b"), Bc(1500, 420, 340, 170, "blanco", "violeta", 5, 26, sombra=True, en="b", t=0.2),
    T("árbol 3", 1500, 395, tam=48, c="violeta", en="b", t=0.3), T("corrige lo que queda", 1500, 455, tam=32, c="violeta", w="sb", en="b", t=0.4),
    P("una cadena de correcciones = BOOSTING", 960, 700, "ambar", 58, en="c"), sec="1. La idea")

# ---- 2. Analogía: el golf
GX0, GX1 = 260, 1660


def gx(m):
    return GX0 + (GX1 - GX0) * m / 200


v.b("Piensa en el golf. [[a]]El hoyo está a doscientos metros. Nadie intenta embocar desde la salida con un golpe perfecto. [[b]]El primer golpe deja la bola a cuarenta metros. [[c]]El segundo no vuelve a empezar: apunta solo a esos cuarenta metros que faltan, y deja la bola a diez.",
    CL(), T("el golf: cada golpe apunta a lo que falta", 960, 160, tam=52, c="tinta", ent="subir"),
    L(GX0, 560, GX1, 560, "verdeclaro", 22), Bc(gx(200), 560, 34, 34, "tinta", None, 0, 17, en="a"), T("hoyo · 200 m", gx(200), 630, tam=34, c="tinta", w="sb", en="a"),
    Bc(gx(0), 530, 30, 30, "blanco", "tinta", 4, 15, id="bola"), T("salida", gx(0), 630, tam=32, c="gris", w="sb"),
    U("bola", x=gx(160), en="b", dur=1.0), T("golpe 1 · faltan 40 m", gx(160), 460, tam=34, c="azul", w="b", en="b", t=0.9),
    U("bola", x=gx(190), en="c", t=1.2, dur=0.8), T("golpe 2 · faltan 10 m", gx(190) - 40, 380, tam=34, c="violeta", w="b", en="c", t=2.0),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el golf")
v.b("El tercer golpe apunta a esos diez metros, y deja la bola a dos. [[a]]Cada golpe es modesto, pero cada uno trabaja sobre el error que dejó el anterior. [[b]]Y la posición final es la suma de todos los golpes.",
    U("bola", x=gx(198), dur=0.6), T("golpe 3 · faltan 2 m", gx(198) - 60, 300, tam=34, c="teal", w="b", t=0.5),
    P("cada golpe trabaja sobre el error que dejó el anterior", 960, 790, "azul", 44, en="a"),
    P("posición final = suma de todos los golpes", 960, 900, "verde", 44, en="b"))

# ---- 3. El modelo aditivo
v.b("En boosting, el primer golpe es una predicción base: [[a]]por ejemplo, el promedio de los precios. A eso lo llamamos efe cero. [[b]]Cada golpe siguiente es un árbol pequeño, hache uno, hache dos, que apunta al error que queda: el residuo. [[c]]Y la predicción final es la suma.",
    CL(), T("el modelo aditivo", 960, 170, tam=54, c="tinta", ent="subir"),
    caja_formula("F(x) = F₀ + η·h₁(x) + η·h₂(x) + …", y=420, c="ambar", tam=64, w=1300),
    T("F₀ = predicción base (el promedio)", 960, 600, tam=44, c="teal", w="sb", en="a"),
    T("hₘ = árbol pequeño que apunta al residuo", 960, 690, tam=44, c="azul", w="sb", en="b"),
    T("F = la suma de todo", 960, 780, tam=44, c="verde", w="sb", en="c"), sec="3. El modelo aditivo")
v.b("¿Y esa letra griega? Es eta, la tasa de aprendizaje. [[a]]Cada árbol no se suma completo: se suma solo una fracción, por ejemplo un treinta por ciento. [[b]]La diapositiva tres lo dice así: eta controla cuánto aporta cada nuevo modelo.",
    R(960, 420, 120, 110, "ambar", r=20, id="anil"), P("η = tasa de aprendizaje", 960, 880, "ambar", 50, en="a"),
    T("cada árbol aporta solo una fracción: η = 0.3 → un 30 %", 960, 970, tam=38, c="tinta", w="sb", en="a", t=0.5),
    X("anil", en="b"), T("diapositiva 3", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="b"))

# ---- 4. Ejemplo resuelto: tres árboles (apunte)
v.b("Hagamos la suma a mano, con el ejemplo del apunte. [[a]]El precio base, efe cero, es 160 mil dólares. [[b]]Para una casa, los tres árboles predicen más 50, más 20 y menos 10. [[c]]Y eta vale 0.3.",
    CL(), T("predecir una casa con tres árboles · miles de USD", 960, 150, tam=46, c="tinta", ent="subir"),
    P("F₀ = 160", 420, 290, "teal", 52, en="a"), P("h₁ = +50", 820, 290, "azul", 48, en="b"), P("h₂ = +20", 1140, 290, "azul", 48, en="b", t=0.3),
    P("h₃ = −10", 1460, 290, "azul", 48, en="b", t=0.6), P("η = 0.3", 1720, 290, "ambar", 48, en="c"),
    ilustrativo(1000, texto="ejemplo del apunte"), sec="4. Ejemplo: tres árboles")
v.b("Etapa uno: 160 más 0.3 por 50, [[a]]o sea 160 más 15: 175. [[b]]Etapa dos: 175 más 0.3 por 20, más 6: 181. [[c]]Etapa tres: 181 más 0.3 por menos 10, menos 3: 178.",
    T("etapa 1:  160 + 0.3 · 50 = 160 + 15 =", 1150, 460, tam=46, c="tinta", w="sb", anc="rm"), N(175, 1180, 460, tam=60, c="verde", anc="lm", en="a"),
    T("etapa 2:  175 + 0.3 · 20 = 175 + 6 =", 1150, 590, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(181, 1180, 590, tam=60, c="verde", anc="lm", en="b", t=0.6),
    T("etapa 3:  181 + 0.3 · (−10) = 181 − 3 =", 1150, 720, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(178, 1180, 720, tam=60, c="verde", anc="lm", en="c", t=0.8))
v.b("Fíjate en dos cosas. [[a]]Cada árbol aporta solo un treinta por ciento de lo que predice: pasos chicos. [[b]]Y el tercer árbol corrige hacia abajo: si los anteriores se pasaron, el siguiente lo compensa.",
    P("pasos chicos: solo el 30 % de cada árbol", 960, 860, "ambar", 44, en="a"), P("h₃ negativo: corrige si los anteriores se pasaron", 960, 960, "azul", 42, en="b"))

# ---- 5. Datos reales: los residuos se achican
v.b("Ahora con datos reales. [[a]]Tomamos las casas de Ames y una sola variable: la superficie habitable. [[b]]Predecir con el promedio deja un error típico, un RMSE, de 78 500 dólares.",
    CL(), T("Ames · solo la superficie habitable · RMSE en miles de USD", 960, 150, tam=44, c="tinta", ent="subir"),
    T("Gradient Boosting · árboles de profundidad 2 · η = 0.3", 960, 220, tam=34, c="gris", w="sb", en="a"),
    barras_v([("F₀ promedio", SQ[0], "gris", "b")], x0=450, escala=7.0, base=860, w=220), real(1050), sec="5. Con datos reales")
v.b("Con un árbol, [[a]]el error baja a 68 400. [[b]]Con cinco árboles, a 55 400. [[c]]Y con cincuenta, a 50 900. Cada árbol trabajó sobre los residuos del anterior, y el error se fue achicando.",
    barras_v([("1 árbol", SQ[1], "azul", "a"), ("5 árboles", SQ[2], "violeta", "b"), ("50 árboles", SQ[3], "verde", "c")], x0=780, paso=330, escala=7.0, base=860, w=220),
    T("cada árbol, sobre los residuos del anterior", 1300, 275, tam=38, c="verde", w="sb", en="c", t=1.0))
v.b("¿Y por qué no baja a cero? [[a]]Porque una sola variable no explica todo el precio: dos casas del mismo tamaño pueden valer muy distinto. [[b]]Lo que queda es ruido, o información que no le dimos al modelo.",
    P("una sola variable no explica todo el precio", 960, 960, "ambar", 44, en="a"), T("lo que queda: ruido o variables que faltan", 960, 1030, tam=32, c="gris", w="sb", en="b"))

# ---- 6. El ciclo general (diapositiva 5)
v.b("Todos los modelos de esta clase siguen el mismo ciclo de cinco pasos, el de la diapositiva cinco. [[a]]Uno, inicializar con una predicción base. [[b]]Dos, medir el error. Tres, ajustar un nuevo aprendiz a ese error. [[c]]Cuatro, actualizar, sumando su contribución. Y cinco, detener: después de eme etapas, o cuando la validación deja de mejorar.",
    CL(), T("el ciclo de boosting · diapositiva 5", 960, 160, tam=50, c="tinta", ent="subir"),
    flujo([("1. inicializar", "teal", "a"), ("2. medir error", "rojo", "b"), ("3. ajustar", "azul", "b"), ("4. actualizar", "violeta", "c"),
           ("5. detener", "verde", "c")], y=460, w=300, h=170, x0=230, paso=365, tam=38),
    L(1690, 560, 1690, 700, "gris", 5, en="c", t=0.6), L(1690, 700, 595, 700, "gris", 5, en="c", t=0.8), L(595, 700, 595, 560, "gris", 5, True, en="c", t=1.0),
    T("repetir mientras la validación mejora", 1140, 740, tam=32, c="gris", w="sb", en="c", t=1.0), sec="6. El ciclo general")
v.b("Y la diapositiva remata con la frase clave de toda la clase: [[a]]la definición exacta de error separa a AdaBoost de Gradient Boosting. [[b]]AdaBoost mide el error con pesos. Gradient Boosting, con residuos.",
    P("la definición de «error» separa a los algoritmos", 960, 860, "ambar", 46, en="a"),
    T("AdaBoost: pesos  ·  Gradient Boosting: residuos", 960, 960, tam=42, c="tinta", w="sb", en="b"))

# ---- 7. La ruta de la clase
v.b("La ruta de la clase tiene cuatro etapas. [[a]]Intuición: aprender de los errores. [[b]]Fundamentos: AdaBoost y los gradientes. [[c]]Implementaciones: XGBoost, LightGBM y CatBoost. Y decisión: comparar y ajustar.",
    CL(), T("la ruta de la clase · diapositiva 2", 960, 160, tam=50, c="tinta", ent="subir"),
    tarjetas([(330, "teal", "Intuición", "aprender de los errores", "a"), (750, "azul", "Fundamentos", "AdaBoost y gradientes", "b"),
              (1170, "violeta", "Implementaciones", "XGBoost, LightGBM, CatBoost", "c"), (1590, "verde", "Decisión", "comparar y ajustar", "c")],
             y=470, w=400, h=320, tam_t=36, tam_d=32), sec="7. La ruta de la clase")
v.b("Y una pregunta guía para todo el recorrido: [[a]]¿cómo equilibramos capacidad predictiva, velocidad, datos categóricos e interpretabilidad?",
    P("¿capacidad · velocidad · categóricas · interpretabilidad?", 960, 820, "ambar", 44, en="a"))

# ---- 8. Error típico
v.b("El error típico al empezar: [[a]]pensar que boosting es entrenar muchos árboles en paralelo y promediarlos. Eso es bagging, la clase anterior. [[b]]En boosting, cada árbol depende del anterior, y todos ven los mismos datos de entrenamiento. [[c]]Lo que cambia de un árbol a otro no es la muestra: es el objetivo, los pesos o los residuos.",
    CL(), K(330, 300, "cruz", 80, en="a"), T("«muchos árboles en paralelo, promediados»", 420, 300, tam=44, c="rojo", anc="lm", en="a"),
    T("eso es bagging (clase 35)", 420, 370, tam=36, c="gris", w="sb", anc="lm", en="a", t=0.6),
    K(330, 520, "check", 80, en="b"), T("boosting: en secuencia, todos ven los mismos datos", 420, 520, tam=44, c="verde", anc="lm", en="b"),
    P("lo que cambia es el OBJETIVO: pesos o residuos", 960, 740, "azul", 50, en="c"), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]En el ejemplo de las tres etapas, ¿cuánto valdría la predicción si eta fuera uno, es decir, si cada árbol se sumara completo?",
    CL(), P("F₀ = 160 · h = +50, +20, −10", 960, 300, "azul", 52, en="a"), T("¿y si η = 1?", 960, 470, tam=80, c="tinta", en="a", t=0.6),
    sec="9. Piensa un segundo", pausa=0.6)
v.b("Con eta igual a uno, sumas todo: 160 más 50, más 20, menos 10, [[a]]igual a 220. [[b]]Con eta igual a 0.3 llegamos a 178. [[c]]Con eta igual a uno se avanza más rápido, pero cada árbol se toma demasiado en serio. Más adelante veremos por qué los pasos chicos suelen ganar.",
    CL(), T("η = 1:  160 + 50 + 20 − 10 =", 1050, 330, tam=52, c="tinta", w="sb", anc="rm"), N(220, 1080, 330, tam=66, c="rojo", anc="lm", en="a"),
    T("η = 0.3:", 1050, 470, tam=52, c="tinta", w="sb", anc="rm", en="b"), T("178", 1080, 470, tam=66, c="verde", anc="lm", en="b", t=0.3),
    P("η = 1: rápido, pero cada árbol pesa demasiado", 960, 660, "ambar", 46, en="c"), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]boosting entrena modelos simples en secuencia, cada uno sobre el error del anterior. [[b]]La predicción final es una suma: efe cero, más eta por cada árbol. [[c]]Y la definición de error es lo que distingue a cada algoritmo.",
    CL(), P("modelos simples EN SECUENCIA, cada uno sobre el error anterior", 960, 300, "azul", 46, en="a"),
    P("F = F₀ + η·h₁ + η·h₂ + …", 960, 470, "ambar", 52, en="b"), P("la definición de «error» distingue a cada algoritmo", 960, 640, "violeta", 46, en="c"),
    sec="10. Para llevar")
v.b("Siguiente parada: [[a]]boosting frente a bagging. Mismos árboles, filosofías opuestas.",
    CL(), siguiente(2, "Bagging vs boosting"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
