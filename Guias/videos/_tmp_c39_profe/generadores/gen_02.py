import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, barras_v)

OUT = out("guion_c39_02_escala.json")
v = V("Los vinos y la estandarización")
DT = D["datos"]
ME, SD, VA, CU = DT["media"], DT["sd"], DT["var"], DT["cuota_var"]
PZ = DT["primer"]

v.portada("Los vinos y la estandarización", "Por qué las escalas deciden los grupos",
          "Segunda parada: los datos de la clase, y el paso que va antes de cualquier distancia: estandarizar.")

# ---- 1. Hook
v.b("Mira estas dos mediciones de un mismo vino. [[a]]La ceniza vale alrededor de 2.37. [[b]]La prolina, alrededor de 748. [[c]]Si calculas distancias con esos números tal cual, una diferencia de 100 en prolina pesa muchísimo más que cualquier diferencia en ceniza. Y los grupos los terminaría decidiendo una sola variable.",
    CL(), T("dos mediciones, dos escalas", 960, 150, tam=56, c="tinta", ent="subir"),
    N(ME["ceniza"], 560, 400, "{:.2f}", 110, "azul", en="a"), T("ceniza (media)", 560, 500, tam=38, c="gris", w="sb", en="a", t=0.3),
    N(ME["prolina"], 1360, 400, "{:.0f}", 110, "rojo", en="b"), T("prolina (media)", 1360, 500, tam=38, c="gris", w="sb", en="b", t=0.3),
    P("sin estandarizar, la prolina decide los grupos", 960, 720, "rojo", 48, en="c"), real(1050, en="a"), sec="1. Dos escalas")

# ---- 2. Los datos
v.b("Los datos de la clase son vinos. [[a]]Ciento setenta y siete vinos, y trece mediciones químicas de cada uno: alcohol, ácido málico, ceniza, alcalinidad, magnesio, fenoles, flavonoides, color, matiz, prolina, y otras. [[b]]Hay además una columna, código, que solo identifica a cada vino: se excluye del ajuste. [[c]]Y no hay ninguna etiqueta de clase. Vamos a buscar grupos solo con la química.",
    CL(), T("datos: 177 vinos y 13 mediciones · diapositiva 6", 960, 150, tam=50, c="tinta", ent="subir"),
    N(DT["n"], 560, 360, "{:.0f}", 120, "violeta", en="a"), T("vinos (filas)", 560, 460, tam=38, c="gris", w="sb", en="a", t=0.3),
    N(DT["p"], 1360, 360, "{:.0f}", 120, "teal", en="a", t=0.4), T("mediciones químicas", 1360, 460, tam=38, c="gris", w="sb", en="a", t=0.7),
    P("código identifica la fila: no entra al ajuste", 960, 640, "ambar", 44, en="b"),
    P("sin etiquetas de clase", 960, 770, "rojo", 44, en="c"), diapo(6, en="a"), sec="2. Los vinos")
v.b("Un dato de contexto. [[a]]La base es una adaptación docente de la base de vinos del repositorio UCI, que tiene ciento setenta y ocho filas: el archivo del curso trae ciento setenta y siete. [[b]]Según la ficha de UCI, son vinos de una misma región de Italia, elaborados con tres variedades de uva distintas. [[c]]En la grabación se mencionaron vinos blancos, un Merlot y un Cabernet; esos nombres no están en los datos. El archivo no dice qué variedad es cada vino.",
    CL(), alerta(160, texto="precisión sobre la grabación"),
    T("UCI: 178 filas · archivo del curso: 177 filas", 960, 300, tam=44, c="tinta", w="sb", en="a"),
    T("tres variedades de uva de una misma región de Italia", 960, 410, tam=44, c="teal", w="sb", en="b"),
    cita("no me acuerdo si es Merlot, no sé; ... un vino blanco y otro tipo de vino, Cabernet", y=560, en="c", tam=40),
    T("el archivo no trae el nombre de la variedad", 960, 700, tam=42, c="rojo", w="sb", en="c", t=2.5),
    grabacion("1:06:23", en="c"), etiqueta("Diapositiva 6 y ficha UCI Wine (doi 10.24432/C5PC7J)", 1010, en="a", c="gris"))

# ---- 3. Analogía
v.b("Una analogía. [[a]]Quieres comparar personas por su estatura, en metros, y su sueldo, en pesos. [[b]]Dos personas que difieren treinta centímetros de estatura, pero mil pesos de sueldo, parecen lejísimos solo por el sueldo. [[c]]El problema no es la persona: es la unidad de medida. Estandarizar es poner todas las variables en la misma unidad: desviaciones estándar.",
    CL(), T("metros contra pesos", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "estatura", "diferencia: 0.3 metros", "a"), (1360, "rojo", "sueldo", "diferencia: 1000 pesos", "b")],
             y=430, w=620, h=280, tam_t=52, tam_d=40),
    P("el sueldo domina solo por su unidad", 960, 700, "rojo", 46, en="b", t=1.5),
    P("estandarizar: todo en desviaciones estándar", 960, 830, "teal", 46, en="c"), analogia(1050), sec="3. Analogía: metros contra pesos")

# ---- 4. La fórmula
v.b("La fórmula de la diapositiva siete. [[a]]A cada valor le restamos la media de su variable, y lo dividimos por la desviación estándar de esa variable. [[b]]El ejemplo hipotético: un vino con alcohol 14, si la media es 13 y la desviación 0.8. [[c]]Catorce menos trece es uno; uno dividido por 0.8 da 1.25. Ese vino está 1.25 desviaciones estándar sobre la media de alcohol.",
    CL(), T("estandarización · diapositiva 7", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("z = (x − media) / desviación estándar", y=310, c="violeta", tam=56, w=1300, en="a"),
    T("x = 14   ·   media = 13   ·   desviación = 0.8", 960, 500, tam=44, c="tinta", w="sb", en="b"),
    T("z = (14 − 13) / 0.8 =", 1080, 640, tam=54, c="tinta", w="sb", anc="rm", en="c"),
    N(1.25, 1110, 640, "{:.2f}", 80, "violeta", anc="lm", en="c", t=1.5),
    P("1.25 desviaciones sobre la media de alcohol", 960, 800, "teal", 44, en="c", t=3.0), diapo(7, en="a"), sec="4. La fórmula")
v.b("Con un vino real. [[a]]El primer vino del archivo tiene 14.23 de alcohol. La media de los ciento setenta y siete es 13.00, y la desviación, 0.814. [[b]]Catorce coma veintitrés menos trece es 1.23; dividido por 0.814, da 1.511. [[c]]Con la media y la desviación sin redondear, sale 1.509; el script imprime 1.51. Después de estandarizar, todas las variables tienen media cero y desviación uno.",
    CL(), T("el primer vino del archivo, estandarizado", 960, 150, tam=50, c="tinta", ent="subir"),
    T(f"alcohol = {PZ['alcohol']:.2f}   ·   media = {ME['alcohol']:.2f}   ·   desviación = {SD['alcohol']:.3f}", 960, 300, tam=42, c="tinta", w="sb", en="a"),
    T("z = (14.23 − 13.00) / 0.814 = 1.23 / 0.814 =", 1150, 450, tam=44, c="tinta", w="sb", anc="rm", en="b"),
    N(1.511, 1180, 450, "{:.3f}", 66, "violeta", anc="lm", en="b", t=2.0),
    T("sin redondear: 1.509 · el script imprime 1.51", 960, 590, tam=40, c="teal", w="sb", en="c"),
    P("tras estandarizar: media 0 y desviación 1 en las 13 variables", 960, 760, "azul", 42, en="c", t=2.0), real(1050, en="a"))

# ---- 5. Cuánto pesa cada escala
BAR = [("prolina", VA["prolina"]), ("magnesio", VA["magnesio"]), ("alcalinidad", VA["alcalinidad"]), ("color", VA["color"])]
v.b("¿Cuánto domina la prolina, de verdad? [[a]]La varianza de la prolina es casi noventa y nueve mil trescientos. La del magnesio, 204. La de la alcalinidad, 11.2. La del color, 5.27. [[b]]La suma de distancias al cuadrado entre vinos se reparte según estas varianzas. Y la prolina se lleva el 99.8 por ciento. [[c]]Sin estandarizar, K-means prácticamente agrupa por prolina, y las otras doce mediciones casi no cuentan.",
    CL(), T("varianza de cada medición (sin estandarizar)", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(290, [(n, 300 + 440 * i) for i, (n, _) in enumerate(BAR)], tams=[36] * 4, colores=["gris"] * 4, en="a", x0=120, x1=1800),
    fila(380, [(f"{VA['prolina']:.0f}", 300), (f"{VA['magnesio']:.0f}", 740), (f"{VA['alcalinidad']:.1f}", 1180), (f"{VA['color']:.2f}", 1620)],
         tams=[52] * 4, colores=["rojo", "ambar", "azul", "teal"], en="a", t=0.4, fondo="linea", x0=120, x1=1800, h=100),
    N(100 * CU["prolina"], 960, 600, "{:.1f}", 120, "rojo", suf=" %", en="b"),
    T("de la varianza total es de la prolina", 960, 710, tam=40, c="gris", w="sb", en="b", t=0.5),
    P("sin estandarizar: K-means agrupa casi solo por prolina", 960, 860, "rojo", 44, en="c"), adicional(1050, en="a"), sec="5. Cuánto pesa cada escala")
v.b("Y eso cambia el resultado. [[a]]Con los datos sin estandarizar, los tres grupos de K-means coinciden con la variedad real en ciento veinticinco de ciento setenta y siete vinos. [[b]]Con los datos estandarizados, en ciento setenta y uno. Lo veremos en detalle más adelante; por ahora, la lección es clara: la escala decide los grupos.",
    CL(), T("los mismos vinos, dos escalas, k = 3", 960, 150, tam=52, c="tinta", ent="subir"),
    N(DT["raw_coinc"], 560, 400, "{:.0f}", 120, "rojo", suf=" / 177", en="a"), T("sin estandarizar", 560, 510, tam=40, c="gris", w="sb", en="a", t=0.3),
    N(D["pca"]["coincidencias"], 1360, 400, "{:.0f}", 120, "verde", suf=" / 177", en="b"), T("estandarizados", 1360, 510, tam=40, c="gris", w="sb", en="b", t=0.3),
    P("la escala decide los grupos", 960, 720, "ambar", 50, en="b", t=2.0), adicional(1050, en="a",
    texto="Cálculo adicional: coincidencia con la variedad real, que no se usa para ajustar"))
v.b("Una precisión de la diapositiva siete: [[a]]estandarizar iguala las escalas, pero no elimina los atípicos ni las correlaciones. Un vino extremo sigue siendo extremo, ahora medido en desviaciones estándar.",
    P("iguala escalas · no elimina atípicos ni correlaciones", 960, 900, "violeta", 44, en="a"))

# ---- 6. En código
v.b("En código. [[a]]En R, la función scale resta la media y divide por la desviación muestral. [[b]]En Python, la clase lo escribe a mano: se quita la columna código con drop, y se resta la media y se divide por la desviación con ddof igual a uno. [[c]]Ese ddof igual a uno es el que hace que Python use la misma desviación muestral que R.",
    CL(), T("datos y escala en R y en Python · diapositivas 8 y 9", 960, 140, tam=46, c="tinta", ent="subir"),
    codigo(["x <- vinos[, -1]", "z <- scale(x)"], y=320, w=1500, tam=38, c="azul", en="a"),
    codigo(['x = vinos.drop(columns="codigo")', "z = (x - x.mean()) / x.std(ddof=1)"], y=560, w=1500, tam=38, c="violeta", en="b"),
    P("ddof=1: la misma desviación muestral que scale() de R", 960, 760, "teal", 42, en="c"), diapo(9, en="a"), sec="6. En código")
v.b("¿Y si usaras la desviación con ddof igual a cero, la de NumPy? [[a]]Todos los valores estandarizados se multiplican por la misma constante, así que los grupos son exactamente los mismos. [[b]]Lo único que cambia es la escala del costo: el WSS pasa de 1267.37 a 1274.57, que es 1267.37 por ciento setenta y siete sobre ciento setenta y seis. Más adelante veremos qué es ese WSS.",
    CL(), T("ddof = 1 o ddof = 0", 960, 150, tam=56, c="tinta", ent="subir"),
    P("mismos grupos (se multiplica todo por una constante)", 960, 320, "verde", 44, en="a"),
    T("WSS con ddof = 1:", 900, 480, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(D["km"]["wss_k3"], 930, 480, "{:.2f}", 60, "azul", anc="lm", en="b"),
    T("WSS con ddof = 0:", 900, 590, tam=44, c="tinta", w="sb", anc="rm", en="b", t=1.5), N(DT["ddof0_wss"], 930, 590, "{:.2f}", 60, "violeta", anc="lm", en="b", t=1.5),
    T("1267.37 × 177 / 176 = 1274.57", 960, 720, tam=40, c="gris", w="sb", en="b", t=4.0), adicional(1050, en="a"))

# ---- 7. Error típico
v.b("Dos errores típicos. [[a]]El primero, dejar la columna código dentro del ajuste: es un identificador, y K-means lo trataría como si fuera química. [[b]]El segundo lo advierte la diapositiva nueve: una etiqueta de clúster que obtuvo el algoritmo no debe volver a entrar como predictor, ni como variable para agrupar de nuevo.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("agrupar con la columna código", 410, 300, tam=46, c="rojo", anc="lm", en="a"),
    K(330, 470, "cruz", 70, en="b"), T("reusar la etiqueta del clúster como predictor", 410, 470, tam=46, c="rojo", anc="lm", en="b"),
    K(330, 650, "check", 70, en="b", t=2.5), T("solo las 13 mediciones, estandarizadas", 410, 650, tam=46, c="verde", anc="lm", en="b", t=2.5),
    diapo(9, en="b"), sec="7. Errores típicos")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un vino tiene color 7.6. La media del color es 5.03, y su desviación, 2.30. [[b]]¿Cuánto vale su color estandarizado?",
    CL(), T("color = 7.6   ·   media = 5.03   ·   desviación = 2.30", 960, 320, tam=50, c="tinta", w="b", en="a"),
    P("z = ?", 960, 520, "violeta", 60, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Siete coma seis menos cinco coma cero tres es 2.57. [[a]]Dividido por 2.30, da [[b]]1.117. Ese vino está algo más de una desviación estándar sobre el color medio.",
    CL(), T("z = (7.6 − 5.03) / 2.30 = 2.57 / 2.30 =", 1120, 360, tam=50, c="tinta", w="sb", anc="rm", en="a"),
    N(1.117, 1150, 360, "{:.3f}", 72, "violeta", anc="lm", en="b"), P("un poco más de una desviación sobre la media", 960, 560, "teal", 44, en="b", t=1.0),
    ilustrativo(1050, texto="Ejemplo ilustrativo con la media y la desviación reales del color"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]las distancias dependen de las unidades; sin estandarizar, la prolina decide casi todo. [[b]]Estandarizar es restar la media y dividir por la desviación, y en Python se usa ddof igual a uno para coincidir con R. [[c]]Y solo entran las mediciones: ni el código, ni etiquetas obtenidas por el propio algoritmo.",
    CL(), P("las unidades deciden las distancias", 960, 300, "rojo", 48, en="a"),
    P("z = (x − media) / desviación · ddof = 1", 960, 460, "violeta", 48, en="b"),
    P("solo las 13 mediciones", 960, 620, "teal", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]la distancia euclidiana, el criterio WSS que minimiza K-means, y cómo se descompone la variabilidad total.",
    CL(), siguiente(3, "Distancia, WSS y TSS = WSS + BSS"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
