import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, matriz

OUT = out("guion_c37_02_matriz.json")
v = V("La matriz de confusión y las métricas")
LG = D["script"]["Logistica"]

v.portada("La matriz de confusión", "Cuatro números que dicen lo que la exactitud esconde",
          "Segunda parada: la matriz de confusión. Cuatro números, y de ellos salen todas las métricas que vamos a usar para comparar modelos.")

# ---- 1. Hook
v.b("Dos modelos tienen la misma exactitud: noventa por ciento. [[a]]Uno se equivoca diciéndole maligno a personas sanas. [[b]]El otro se equivoca mandando a casa a personas enfermas. [[c]]La exactitud no distingue entre los dos. Para eso existe la matriz de confusión.",
    CL(), T("misma exactitud: 90 %", 960, 170, tam=58, c="tinta", ent="subir"),
    Bc(520, 470, 640, 300, "blanco", "ambar", 5, 30, sombra=True, en="a"), T("modelo A", 520, 380, tam=46, c="ambar", en="a", t=0.1),
    T("alarma a personas sanas", 520, 490, tam=38, c="tinta", w="sb", en="a", t=0.2),
    Bc(1400, 470, 640, 300, "blanco", "rojo", 5, 30, sombra=True, en="b"), T("modelo B", 1400, 380, tam=46, c="rojo", en="b", t=0.1),
    T("manda a casa a enfermos", 1400, 490, tam=38, c="tinta", w="sb", en="b", t=0.2),
    P("la exactitud no los distingue → matriz de confusión", 960, 760, "violeta", 50, en="c"), ilustrativo(1000), sec="1. Misma exactitud")

# ---- 2. Analogía: la tabla de cruce
v.b("Piensa en un profesor que corrige un verdadero o falso. [[a]]Para cada pregunta mira dos cosas: qué era lo correcto, [[b]]y qué marcó el alumno. [[c]]Con esas dos cosas, cada respuesta cae en una de cuatro casillas. La matriz de confusión es exactamente esa tabla, pero para un modelo.",
    CL(), T("el profesor que corrige", 960, 160, tam=54, c="tinta", ent="subir"),
    P("¿qué era lo correcto?", 560, 340, "teal", 48, en="a"), P("¿qué marcó el alumno?", 1360, 340, "violeta", 48, en="b"),
    *[Bc(760 + 200 * (k % 2), 560 + 170 * (k // 2), 180, 150, "blanco", "gris", 4, 18, en="c", t=0.1 * k) for k in range(4)],
    T("4 casillas", 1260, 645, tam=46, c="tinta", anc="lm", en="c", t=0.5), ilustrativo(1000, texto="analogía"), sec="2. Analogía: corregir una prueba")

# ---- 3. La tabla de 2 × 2
v.b("La diapositiva ocho la arma así. [[a]]Las filas son la clase real: arriba cero, abajo uno. [[b]]Las columnas, la clase predicha: cero a la izquierda, uno a la derecha. [[c]]En la diagonal están los aciertos: verdaderos negativos y verdaderos positivos.",
    CL(), T("la matriz de confusión · diapositiva 8", 960, 150, tam=50, c="tinta", ent="subir"),
    T("real 0", 560, 420, tam=40, c="teal", anc="rm", en="a"), T("real 1", 560, 640, tam=40, c="teal", anc="rm", en="a", t=0.2),
    T("predicha 0", 770, 290, tam=40, c="violeta", en="b"), T("predicha 1", 1150, 290, tam=40, c="violeta", en="b", t=0.2),
    Bc(770, 420, 360, 200, "verdeclaro", "verde", 4, 20, en="c", id="c00"), T("VN", 770, 395, tam=60, c="verde", en="c", t=0.1),
    T("verdadero negativo", 770, 465, tam=28, c="verde", w="sb", en="c", t=0.15),
    Bc(1150, 640, 360, 200, "verdeclaro", "verde", 4, 20, en="c", t=0.3), T("VP", 1150, 615, tam=60, c="verde", en="c", t=0.4),
    T("verdadero positivo", 1150, 685, tam=28, c="verde", w="sb", en="c", t=0.45), sec="3. La tabla de 2 × 2")
v.b("Y fuera de la diagonal, los errores. [[a]]Arriba a la derecha, el falso positivo: un caso negativo que el modelo declara positivo. Aquí, un benigno que se clasifica como maligno. [[b]]Abajo a la izquierda, el falso negativo: un maligno que se clasifica como benigno. [[c]]En este problema, ese es el error grave.",
    Bc(1150, 420, 360, 200, "rojoclaro", "rojo", 4, 20, en="a"), T("FP", 1150, 395, tam=60, c="rojo", en="a", t=0.1),
    T("falso positivo", 1150, 465, tam=28, c="rojo", w="sb", en="a", t=0.15),
    Bc(770, 640, 360, 200, "rojoclaro", "rojo", 4, 20, en="b"), T("FN", 770, 615, tam=60, c="rojo", en="b", t=0.1),
    T("falso negativo", 770, 685, tam=28, c="rojo", w="sb", en="b", t=0.15),
    T("FP: benigno → «maligno» (falsa alarma)", 960, 840, tam=36, c="tinta", w="sb", en="a", t=1.0),
    P("FN: maligno → «benigno»: el error grave", 960, 940, "rojo", 42, en="c"))
v.b("Por eso la diapositiva insiste: [[a]]siempre hay que declarar cuál es la clase positiva, y en qué orden van las filas y las columnas. [[b]]Si no, los mismos cuatro números se pueden leer al revés.",
    CL(), P("declarar SIEMPRE la clase positiva", 960, 380, "violeta", 54, en="a"), P("y el orden de filas y columnas", 960, 520, "violeta", 54, en="a", t=0.6),
    T("si no, los mismos números se leen al revés", 960, 680, tam=42, c="gris", w="sb", en="b"))

# ---- 4. Ejemplo de la diapositiva 9
EJ = [[56, 14], [6, 24]]
v.b("Ahora, el ejemplo resuelto de la diapositiva nueve. [[a]]Cien casos: 56 verdaderos negativos, 14 falsos positivos, [[b]]6 falsos negativos y 24 verdaderos positivos. [[c]]Primera métrica, la exactitud: los aciertos sobre el total. 24 más 56, sobre cien: 0.80.",
    CL(), T("ejemplo resuelto · diapositiva 9", 960, 140, tam=48, c="tinta", ent="subir"),
    matriz(520, 520, EJ, en="a", cel=200, tam=64, ids="m"),
    T("exactitud = (VP + VN) / n", 1400, 400, tam=42, c="tinta", w="sb", en="c"), T("= (24 + 56) / 100 =", 1380, 490, tam=42, c="tinta", w="sb", anc="rm", en="c", t=0.5),
    N(0.80, 1400, 490, "{:.2f}", 60, "azul", anc="lm", en="c", t=1.0), ilustrativo(1000, texto="ejemplo de la diapositiva 9"), sec="4. Ejemplo de la diapositiva 9")
v.b("Segunda, la sensibilidad: de los positivos reales, ¿cuántos detecté? [[a]]Verdaderos positivos sobre verdaderos positivos más falsos negativos: [[b]]24 sobre 30, 0.80. [[c]]Tercera, la especificidad: de los negativos reales, ¿cuántos dejé bien? 56 sobre 70: también 0.80.",
    R(520, 620, 410, 210, "verde", r=20, id="rs", en="a"),
    T("sensibilidad = VP / (VP + FN) = 24 / 30 =", 1500, 640, tam=38, c="tinta", w="sb", anc="rm", en="a", t=0.3), N(0.80, 1520, 640, "{:.2f}", 56, "verde", anc="lm", en="b", t=0.3),
    X("rs", en="c"), R(520, 420, 410, 210, "azul", r=20, id="re", en="c"),
    T("especificidad = VN / (VN + FP) = 56 / 70 =", 1500, 780, tam=38, c="tinta", w="sb", anc="rm", en="c", t=0.4), N(0.80, 1520, 780, "{:.2f}", 56, "azul", anc="lm", en="c", t=1.2))

# ---- 5. Precisión: la red del pescador
v.b("Para la cuarta métrica, una imagen: un pescador con una red. [[a]]La sensibilidad pregunta: de todos los peces del lago, ¿qué fracción quedó en la red? [[b]]La precisión pregunta otra cosa: de todo lo que hay en la red, ¿qué fracción son peces, y no botas viejas?",
    CL(), T("la red del pescador", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "sensibilidad", "de los peces del lago, ¿cuántos atrapé?", "a"), (1360, "violeta", "precisión", "de lo que hay en la red, ¿cuánto son peces?", "b")],
             y=500, w=640, h=360, tam_t=52, tam_d=38), ilustrativo(1000, texto="analogía"), sec="5. Precisión: la red del pescador")
v.b("En el ejemplo: [[a]]verdaderos positivos sobre verdaderos positivos más falsos positivos. [[b]]24 sobre 38: 0.632. [[c]]Las otras tres métricas daban 0.80, pero la precisión baja, porque los catorce falsos positivos se mezclan con los veinticuatro aciertos.",
    CL(), matriz(520, 540, EJ, cel=200, tam=64), R(620, 540, 230, 420, "violeta", r=20, en="a"),
    T("precisión = VP / (VP + FP) = 24 / 38 =", 1460, 450, tam=38, c="tinta", w="sb", anc="rm", en="a", t=0.3), N(0.632, 1480, 450, "{:.3f}", 56, "violeta", anc="lm", en="b", t=0.3),
    P("14 FP se mezclan con 24 VP", 1380, 640, "rojo", 44, en="c"), ilustrativo(1000, texto="ejemplo de la diapositiva 9"))

# ---- 6. F1 y exactitud balanceada
v.b("Dos resúmenes más, de la diapositiva diez. [[a]]El F1 junta precisión y sensibilidad: dos veces los verdaderos positivos, sobre dos veces los verdaderos positivos más los falsos positivos y los falsos negativos. [[b]]48 sobre 68: 0.706. [[c]]Ojo: el F1 no usa los verdaderos negativos.",
    CL(), T("F1 y exactitud balanceada · diapositiva 10", 960, 150, tam=48, c="tinta", ent="subir"),
    caja_formula("F1 = 2·VP / (2·VP + FP + FN)", y=320, c="violeta", tam=54, w=1100, en="a"),
    T("= 48 / (48 + 14 + 6) = 48 / 68 =", 1080, 480, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(0.706, 1110, 480, "{:.3f}", 60, "violeta", anc="lm", en="b", t=0.6),
    P("el F1 NO usa los VN", 960, 610, "ambar", 46, en="c"), sec="6. F1 y exactitud balanceada")
v.b("La exactitud balanceada es el promedio de sensibilidad y especificidad: [[a]]0.80 más 0.80, sobre dos: 0.80. [[b]]Le da el mismo peso a las dos clases, sin importar cuántos casos tenga cada una. [[c]]Con la línea base de siempre benigno, daría uno más cero, sobre dos: 0.5. Ahí se ve que esa regla no sabe nada.",
    T("exactitud balanceada = (0.80 + 0.80) / 2 =", 1080, 740, tam=42, c="tinta", w="sb", anc="rm", en="a"), N(0.80, 1110, 740, "{:.2f}", 56, "azul", anc="lm", en="a", t=0.6),
    T("mismo peso a las dos clases", 960, 830, tam=38, c="gris", w="sb", en="b"),
    P("«siempre benigno»: (1 + 0) / 2 = 0.5", 960, 930, "rojo", 44, en="c"))

# ---- 7. Con datos reales
v.b("Ahora, con datos reales: la regresión logística de la clase, en los 171 casos de prueba. [[a]]103 verdaderos negativos, 4 falsos positivos, [[b]]6 falsos negativos y 58 verdaderos positivos. [[c]]Sensibilidad: 58 sobre 64, 0.906. Y precisión: 58 sobre 62, 0.935.",
    CL(), T("la logística de la clase · 171 casos de prueba", 960, 140, tam=48, c="tinta", ent="subir"),
    matriz(520, 540, [[LG["tn"], LG["fp"]], [LG["fn"], LG["tp"]]], en="a", cel=200, tam=64),
    T("sensibilidad = 58 / 64 =", 1440, 440, tam=44, c="tinta", w="sb", anc="rm", en="c"), N(LG["sens"], 1460, 440, "{:.3f}", 58, "verde", anc="lm", en="c", t=0.5),
    T("precisión = 58 / 62 =", 1440, 580, tam=44, c="tinta", w="sb", anc="rm", en="c", t=1.4), N(LG["prec"], 1460, 580, "{:.3f}", 58, "violeta", anc="lm", en="c", t=1.9),
    real(1050, en="a"), sec="7. Con datos reales")
v.b("En el código, la tabla sale con confusion matrix, con las filas como clase real y el orden cero, uno. [[a]]Python cuenta desde cero: la posición cero cero es el verdadero negativo, y la uno uno, el verdadero positivo. [[b]]En R, la función table cuenta desde uno: el verdadero negativo está en la posición uno uno, y el verdadero positivo, en la dos dos.",
    CL(), T("la misma tabla en Python y en R", 960, 150, tam=50, c="tinta", ent="subir"),
    Bc(520, 470, 760, 380, "blanco", "azul", 4, 24, sombra=True), T("Python", 520, 330, tam=46, c="azul"),
    T("confusion_matrix(y, ŷ, labels=[0, 1])", 520, 420, tam=30, c="tinta", w="m"),
    T("[0, 0] = VN    [1, 1] = VP", 520, 520, tam=40, c="azul", w="sb", en="a"),
    Bc(1400, 470, 760, 380, "blanco", "teal", 4, 24, sombra=True, en="b"), T("R", 1400, 330, tam=46, c="teal", en="b", t=0.1),
    T("table(Real = y, Prediccion = ŷ)", 1400, 420, tam=30, c="tinta", w="m", en="b", t=0.2),
    T("[1, 1] = VN    [2, 2] = VP", 1400, 520, tam=40, c="teal", w="sb", en="b", t=0.4), sec="8. En Python y en R")

# ---- 8. El error típico
v.b("El error típico: [[a]]tomar la clase equivocada como positiva. [[b]]Con la misma matriz de la logística, si alguien toma benigno como positivo, su «sensibilidad» sale 103 sobre 107: 0.963. [[c]]Suena excelente, pero es la especificidad de verdad. De los malignos, el modelo detecta 0.906.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("tomar la clase equivocada como positiva", 410, 230, tam=46, c="rojo", anc="lm", en="a"),
    T("positivo = benigno →  «sensibilidad» = 103 / 107 =", 1260, 420, tam=42, c="tinta", w="sb", anc="rm", en="b"), N(0.963, 1290, 420, "{:.3f}", 56, "rojo", anc="lm", en="b", t=0.6),
    T("eso es la especificidad real", 960, 540, tam=40, c="gris", w="sb", en="c"),
    T("positivo = maligno →  sensibilidad = 58 / 64 =", 1260, 660, tam=42, c="tinta", w="sb", anc="rm", en="c", t=0.8), N(LG["sens"], 1290, 660, "{:.3f}", 56, "verde", anc="lm", en="c", t=1.4),
    real(1050, en="b"), sec="9. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un modelo tiene 90 verdaderos negativos, 10 falsos positivos, 20 falsos negativos y 80 verdaderos positivos. [[b]]¿Cuál es su sensibilidad? ¿Y su precisión?",
    CL(), matriz(760, 540, [[90, 10], [20, 80]], en="a", cel=200, tam=64), P("sensibilidad = ?", 1450, 450, "verde", 50, en="b"), P("precisión = ?", 1450, 590, "violeta", 50, en="b", t=0.4),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("Sensibilidad: 80 sobre 100, [[a]]0.80. [[b]]Precisión: 80 sobre 90, [[c]]0.889. Y de regalo: la especificidad es 90 sobre 100, 0.90, y la exactitud, 170 sobre 200, 0.85.",
    CL(), T("sensibilidad = 80 / (80 + 20) =", 1100, 300, tam=48, c="tinta", w="sb", anc="rm"), N(0.80, 1130, 300, "{:.2f}", 62, "verde", anc="lm", en="a"),
    T("precisión = 80 / (80 + 10) =", 1100, 430, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(0.889, 1130, 430, "{:.3f}", 62, "violeta", anc="lm", en="c"),
    T("especificidad = 90 / 100 = 0.90   ·   exactitud = 170 / 200 = 0.85", 960, 600, tam=40, c="gris", w="sb", en="c", t=1.0), ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]filas, la clase real; columnas, la predicha; y siempre declarando la clase positiva. [[b]]Sensibilidad, de los positivos reales cuántos detecto; especificidad, de los negativos cuántos dejo bien. [[c]]Y precisión: de los que declaro positivos, cuántos lo eran.",
    CL(), P("filas = real · columnas = predicha · clase positiva declarada", 960, 300, "violeta", 44, en="a"),
    P("sensibilidad = VP/(VP+FN) · especificidad = VN/(VN+FP)", 960, 460, "verde", 44, en="b"),
    P("precisión = VP/(VP+FP)", 960, 620, "azul", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la regresión logística, el primer modelo de la clase, que convierte un puntaje en una probabilidad.",
    CL(), siguiente(3, "Regresión logística: del puntaje a la probabilidad"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
