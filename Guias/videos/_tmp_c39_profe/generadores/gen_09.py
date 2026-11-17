import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, grupos_series, serie_puntos, revelar, cruz, simulado, GCOL)

OUT = out("guion_c39_09_proyeccion.json")
v = V("Proyección, clases reales y límites de K-means")
PC = D["pca"]
P1, P2, GR, CL_ = PC["pc1"], PC["pc2"], PC["grupo"], PC["clase"]
CDG = {int(k): c for k, c in PC["clase_del_grupo"].items()}
CCOL = {1: "azul", 2: "teal", 3: "ambar"}          # color de cada variedad (= color de su grupo mayoritario)
asig = [CDG[g] for g in GR]
disc = [i for i in range(len(GR)) if asig[i] != CL_[i]]
TB = PC["tabla"]
LU, AT = D["sim_lunas"], D["sim_atipico"]

v.portada("Proyección y límites", "Dibujar trece dimensiones, mirar las variedades reales y saber cuándo K-means falla",
          "Novena parada: vamos a dibujar los grupos, a compararlos con las variedades reales de los vinos, y a ver cuándo K-means no es la herramienta adecuada.")

# ---- 1. Hook
v.b("Los grupos se formaron con trece variables. [[a]]Nadie puede dibujar un espacio de trece dimensiones. [[b]]Si dibujas solo dos variables, por ejemplo alcohol y flavonoides, ves una parte muy pequeña de la historia. [[c]]Necesitamos una forma de resumir las trece en dos ejes que conserven lo más posible. Para eso está el análisis de componentes principales, PCA.",
    CL(), T("¿cómo se dibujan trece dimensiones?", 960, 150, tam=54, c="tinta", ent="subir"),
    P("13 variables: imposible de dibujar", 960, 330, "rojo", 46, en="a"),
    P("2 variables sueltas: una parte muy pequeña", 960, 480, "ambar", 46, en="b"),
    P("PCA: resumir 13 variables en 2 ejes", 960, 640, "teal", 50, en="c"), sec="1. Trece dimensiones")

# ---- 2. Analogía
v.b("Una analogía. [[a]]La sombra de una mano en la pared. Si pones bien la luz, la sombra muestra un perro, un pájaro: conserva mucho de la forma. [[b]]Pero pierde la profundidad. Dos dedos que están lejos, uno delante del otro, en la sombra pueden verse juntos. [[c]]PCA busca la mejor posición de la luz: la proyección que conserva la mayor variabilidad. Aun así, es una sombra.",
    CL(), T("la sombra en la pared", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "conserva", "la forma general", "a"), (1360, "rojo", "pierde", "la profundidad", "b")], y=430, w=600, h=260, tam_t=52, tam_d=42),
    P("PCA: la sombra que conserva más variabilidad", 960, 720, "teal", 46, en="c"), analogia(1050), sec="2. Analogía: la sombra")

# ---- 3. La proyección
GP = "gp9"
v.b("Esta es la proyección de los vinos en los dos primeros componentes, recalculada con el script. [[a]]El primer componente conserva el 36.0 por ciento de la variabilidad; el segundo, el 19.4. Juntos, el 55.3 por ciento. [[b]]Cada vino tiene el color de su grupo de K-means. [[c]]Los tres grupos ocupan zonas distintas de la sombra, aunque en las fronteras se tocan. Ojo: K-means se ajustó con las trece variables, no con estas dos coordenadas.",
    CL(), T("una proyección de los grupos · diapositiva 30", 960, 130, tam=48, c="tinta", ent="subir"),
    G(360, 220, 1180, 860, -4.6, 4.6, -3.6, 4.0, grupos_series(P1, P2, [g - 1 for g in GR], mk=7), xticks=[-4, -2, 0, 2, 4], yticks=[-2, 0, 2],
      yfmt="{:.0f}", xlabel="componente principal 1", ylabel="PC2", id=GP, tam_txt=26),
    T("PC1: 36.0 %", 1500, 330, tam=44, c="tinta", w="sb", en="a"), T("PC2: 19.4 %", 1500, 410, tam=44, c="tinta", w="sb", en="a", t=0.8),
    T("juntos: 55.3 %", 1500, 500, tam=48, c="violeta", w="b", en="a", t=1.6),
    revelar(GP, 3, en="b", dur=1.5),
    [T(f"grupo {g + 1}", 1500, 620 + 60 * g, tam=40, c=GCOL[g], w="b", en="b", t=0.3 * g) for g in range(3)],
    P("ajustado con 13 variables, no con estas 2", 1500, 860, "ambar", 32, en="c", t=2.0), real(1050, en="a"), sec="3. La proyección PCA")
v.b("Una precisión menor sobre la grabación: [[a]]se dijo que los dos componentes conservan casi el cincuenta y seis por ciento de la varianza. [[b]]El valor exacto es 55.3 por ciento: más cerca de cincuenta y cinco. Lo importante se mantiene: casi la mitad de la información queda fuera del dibujo.",
    CL(), alerta(160, texto="precisión sobre la grabación"),
    cita("yo tengo ahí casi 56% de la variabilidad explicada por estos componentes", y=320, en="a"),
    N(PC["var2"], 960, 500, "{:.1f}", 110, "violeta", suf=" %", en="b"), T("valor recalculado (PC1 + PC2)", 960, 600, tam=38, c="gris", w="sb", en="b", t=0.5),
    P("casi la mitad de la información queda fuera del dibujo", 960, 760, "ambar", 44, en="b", t=2.5),
    grabacion("1:07:58", en="a"))

# ---- 4. Comparar con las variedades reales
GQ = "gq9"
v.b("Ahora una comparación que en un problema real casi nunca se puede hacer. [[a]]Este archivo tiene, aparte, la variedad real de cada vino. No se usó para ajustar K-means, ni para elegir k. [[b]]A la izquierda, los vinos coloreados por su variedad real. A la derecha, por el grupo de K-means, pintado con el color de la variedad más frecuente en ese grupo. [[c]]Los círculos negros marcan los vinos donde no coinciden.",
    CL(), T("grupos de K-means frente a variedades reales", 960, 130, tam=48, c="tinta", ent="subir"),
    G(200, 230, 900, 840, -4.6, 4.6, -3.6, 4.0, grupos_series(P1, P2, CL_, colores=[CCOL[1], CCOL[2], CCOL[3]], mk=6, orden=[1, 2, 3]),
      xticks=[-4, 0, 4], yticks=[-2, 0, 2], yfmt="{:.0f}", xlabel="PC1", ylabel="PC2", id=GQ, tam_txt=24, en="a"),
    T("variedades reales", 550, 200, tam=36, c="tinta", w="b", en="b"),
    G(1080, 230, 1780, 840, -4.6, 4.6, -3.6, 4.0, grupos_series(P1, P2, asig, colores=[CCOL[1], CCOL[2], CCOL[3]], mk=6, orden=[1, 2, 3]),
      xticks=[-4, 0, 4], yticks=[-2, 0, 2], yfmt="{:.0f}", xlabel="PC1", ylabel="", id="gq9b", tam_txt=24, en="b"),
    T("grupos de K-means", 1430, 200, tam=36, c="tinta", w="b", en="b"),
    revelar(GQ, 3, en="b", dur=1.2), revelar("gq9b", 3, en="b", t=1.5, dur=1.2),
    [R(P1[i], P2[i], 34, 34, "tinta", r=17, ref="gq9b", g=3, pulso=False, en="c", t=0.1 * k) for k, i in enumerate(disc)],
    real(1050, en="a"), sec="4. Las variedades reales")
v.b("La tabla cruzada, del script de la clase. [[a]]Los cincuenta y nueve vinos de la variedad uno caen todos en el grupo uno. Los cuarenta y siete de la variedad tres, todos en el grupo dos. [[b]]De los setenta y uno de la variedad dos, sesenta y cinco caen en el grupo tres, tres en el uno y tres en el dos. [[c]]En total, ciento setenta y uno de ciento setenta y siete vinos coinciden con la variedad mayoritaria de su grupo: un 96.6 por ciento. Las seis discrepancias son todas de la variedad dos.",
    CL(), T("variedad real × grupo de K-means (sección 7 del script)", 960, 130, tam=46, c="tinta", ent="subir"),
    fila(260, [("", 500), ("grupo 1", 820), ("grupo 2", 1120), ("grupo 3", 1420)], tams=[34] * 4, colores=["gris", GCOL[0], GCOL[1], GCOL[2]], en="a"),
    [fila(360 + 100 * c, [(f"variedad {c + 1}", 500)] + [(str(TB[c][g]), 820 + 300 * g) for g in range(3)], tams=[38, 48, 48, 48],
          colores=["tinta"] + [("verde" if TB[c][g] == max(TB[c]) else ("rojo" if TB[c][g] else "gris")) for g in range(3)],
          en=("a" if c != 1 else "b"), t=(0.4 * c if c != 1 else 0.0), fondo="linea", x0=360, x1=1560, h=86) for c in range(3)],
    N(PC["coincidencias"], 760, 760, "{:.0f}", 100, "verde", suf=" / 177", en="c"),
    N(100 * PC["pureza"], 1300, 760, "{:.1f}", 100, "verde", suf=" %", en="c", t=0.6),
    T("6 discrepancias, todas de la variedad 2", 960, 880, tam=40, c="rojo", w="sb", en="c", t=3.0), real(1050, en="a"))

# ---- 5. Alerta
v.b("Una precisión sobre la grabación. [[a]]Se dijo que debía haber alrededor de ocho o nueve puntos mal clasificados de ciento setenta y siete, y una clasificación buena de más del noventa, noventa y dos por ciento. [[b]]El script entrega seis discrepancias, y ciento setenta y uno de ciento setenta y siete: 96.6 por ciento. El propio profesor había contado seis en el gráfico. [[c]]Y una advertencia de fondo: esto no es exactitud en datos nuevos. Es una comparación descriptiva de estos mismos vinos, con una etiqueta que normalmente no existe.",
    CL(), alerta(160),
    cita("debe tener alguna cosa como 8, 9 puntos de 177 mal clasificados ... más de 90%, 92%", y=310, en="a"),
    P("script: 6 discrepancias · 171 / 177 = 96.6 %", 960, 470, "verde", 46, en="b"),
    T("no es exactitud en datos nuevos: es una comparación descriptiva", 960, 620, tam=42, c="ambar", w="sb", en="c"),
    grabacion("1:24:39", en="a"), etiqueta("Script, sección 7: coincidencias = 171, pureza = 96.6 %", 1010, en="b"), sec="5. Alerta: cuántos discrepan")

# ---- 6. Agrupar con las 13 o con 2
v.b("En la clase, Matías hizo una gran pregunta: [[a]]¿la agrupación se hizo con las trece variables, o con los dos componentes del PCA? [[b]]Con las trece. La proyección es solo para mirar. [[c]]Agrupar sobre dos componentes descarta casi la mitad de la variabilidad, y con datos categóricos, que son la mayoría en una empresa, el PCA ni siquiera aplica directamente. La recomendación del profesor: agrupar con la base original.",
    CL(), T("¿agrupar con las 13 variables o con 2 componentes?", 960, 150, tam=50, c="tinta", ent="subir"),
    P("con las 13 variables", 960, 340, "verde", 52, en="b"),
    T("la proyección es solo para mirar", 960, 470, tam=44, c="tinta", w="sb", en="b", t=1.5),
    T("2 componentes descartan casi la mitad de la variabilidad", 960, 590, tam=42, c="rojo", w="sb", en="c"),
    grabacion("1:28:33", en="a"), sec="6. ¿13 variables o 2 componentes?")

# ---- 7. Límites: formas curvas
GL = "gl9"
v.b("Ahora los límites de K-means, de la diapositiva treinta y uno. [[a]]Estos datos simulados forman dos medias lunas, una dentro de la otra. [[b]]K-means busca grupos compactos alrededor de un centro, así que corta el dibujo con una recta: mitad izquierda, mitad derecha. [[c]]Solo el setenta y cinco por ciento de los puntos queda en el grupo de su propia luna. K-means puede dividir estructuras curvas en grupos compactos que no siguen su forma.",
    CL(), T("limitaciones de K-means · diapositiva 31", 960, 130, tam=50, c="tinta", ent="subir"),
    G(200, 230, 900, 790, -1.4, 2.4, -0.8, 1.3, grupos_series(LU["x"], LU["y"], LU["real"], colores=["violeta", "teal"], mk=6),
      xticks=[-1, 0, 1, 2], yticks=[0, 1], yfmt="{:.0f}", id=GL, tam_txt=24),
    T("forma real", 550, 880, tam=36, c="tinta", w="b", en="a"), revelar(GL, 2, en="a", dur=1.2),
    G(1080, 230, 1780, 790, -1.4, 2.4, -0.8, 1.3, grupos_series(LU["x"], LU["y"], LU["lab"], colores=["azul", "ambar"], mk=6),
      xticks=[-1, 0, 1, 2], yticks=[0, 1], yfmt="{:.0f}", id="gl9b", tam_txt=24, en="b"),
    T("K-means, k = 2", 1430, 880, tam=36, c="tinta", w="b", en="b"), revelar("gl9b", 2, en="b", dur=1.2),
    [cruz(cx, cy, "tinta", 44, ref="gl9b", en="b", t=1.4) for cx, cy in zip(LU["cx"], LU["cy"])],
    P(f"solo {100 * LU['acuerdo']:.0f} % de los puntos en el grupo de su luna", 960, 975, "rojo", 38, en="c"),
    simulado(1050, en="a", texto="Ejemplo simulado (cálculo adicional)"), sec="7. Formas curvas")
v.b("Para formas así existen otros métodos. [[a]]El profesor mencionó DBSCAN, que no busca centros sino zonas de alta densidad de puntos: sigue la forma de cada luna. [[b]]K-means, K-modes y K-prototypes no son los únicos métodos de agrupamiento; son los más sencillos y conocidos.",
    CL(), P("DBSCAN: agrupa por densidad, no por centros", 960, 380, "teal", 48, en="a"),
    P("K-means no es el único método", 960, 560, "violeta", 48, en="b"), grabacion("1:04:06", en="a"))

# ---- 8. Límites: atípicos
GA_ = "ga9"
v.b("Otro límite, que el profesor dibujó al final de la clase: los atípicos. [[a]]Aquí hay dos nubes claras, y un solo punto muy lejos, a la derecha. [[b]]Pedimos dos grupos, esperando una nube en cada uno. [[c]]Pero K-means deja al punto lejano solo, en un grupo de un elemento, y junta las dos nubes en el otro grupo, de ochenta puntos. El cuadrado de su distancia pesa tanto, que le conviene darle un centro propio.",
    CL(), T("un atípico puede robarse un grupo", 960, 130, tam=52, c="tinta", ent="subir"),
    G(260, 250, 1660, 760, -4, 26.5, -2.2, 2.2, [serie_puntos(AT["x"], AT["y"], "gris", 7)], xticks=[0, 10, 20], yticks=[-2, 0, 2], yfmt="{:.0f}",
      id=GA_, tam_txt=26),
    revelar(GA_, 1, en="a", dur=1.2), R(25, 0, 60, 60, "rojo", r=30, ref=GA_, en="a", t=1.5),
    G(260, 250, 1660, 760, -4, 26.5, -2.2, 2.2, grupos_series(AT["x"], AT["y"], AT["lab"], colores=(["azul", "rojo"] if AT["lab"][-1] == 1 else ["rojo", "azul"]), mk=7, revela=1.0),
      xticks=[], yticks=[], id="ga9b", ejes=0.0, en="c", dur=1.0),
    [cruz(cx, cy, "tinta", 44, ref=GA_, en="c", t=0.8) for cx, cy in zip(AT["cx"], AT["cy"])],
    T(f"tamaños: {AT['tamanos'][1]} y {AT['tamanos'][0]}", 960, 850, tam=44, c="rojo", w="b", en="c", t=1.2),
    simulado(1050, en="a", texto="Ejemplo simulado (cálculo adicional)"), grabacion("2:03:26", en="a", y=1010), sec="8. Atípicos")
v.b("La recomendación: [[a]]revisar atípicos antes de agrupar, por ejemplo con una proyección o con métodos de detección de atípicos, que verán en otra clase. [[b]]Y la diapositiva agrega más cosas que revisar: escalas, variables redundantes, y grupos de tamaños o dispersiones muy distintos. K-means funciona mejor con grupos compactos y de tamaño parecido.",
    CL(), P("revisar atípicos antes de agrupar", 960, 330, "rojo", 48, en="a"),
    P("también: escalas, variables redundantes, tamaños y dispersiones distintas", 960, 510, "ambar", 40, en="b"),
    diapo(31, en="b"))

# ---- 9. Error típico
v.b("El error típico de esta parada: [[a]]reportar el 96.6 por ciento como si fuera la exactitud de un clasificador. [[b]]En un problema real no hay variedad conocida: no se puede calcular. Aquí se pudo solo porque el archivo docente la trae aparte. [[c]]Sirve como ilustración de que K-means puede recuperar estructura real, no como métrica para elegir k ni para vender el modelo.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("«el clustering tiene 96.6 % de exactitud»", 410, 300, tam=46, c="rojo", anc="lm", en="a"),
    T("en la práctica no hay variedad conocida", 960, 450, tam=44, c="tinta", w="sb", en="b"),
    K(330, 620, "check", 70, en="c"), T("es una ilustración, no una métrica de selección", 410, 620, tam=46, c="verde", anc="lm", en="c"),
    sec="9. El error típico")

# ---- 10. Piensa
v.b("Piensa un segundo. [[a]]Un grupo A tiene cuarenta vinos de la variedad uno y diez de la variedad dos. Un grupo B tiene cinco de la variedad uno y cuarenta y cinco de la variedad dos. [[b]]¿Cuántos vinos coinciden con la variedad mayoritaria de su grupo, y qué porcentaje es?",
    CL(), T("grupo A: 40 de la variedad 1 y 10 de la variedad 2", 960, 300, tam=48, c="tinta", w="b", en="a"),
    T("grupo B: 5 de la variedad 1 y 45 de la variedad 2", 960, 400, tam=48, c="tinta", w="b", en="a", t=1.0),
    P("¿cuántos coinciden? ¿qué porcentaje?", 960, 580, "violeta", 54, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("En A, la mayoritaria es la variedad uno: [[a]]son cuarenta vinos. En B, la variedad dos: son cuarenta y cinco vinos. [[b]]Cuarenta más cuarenta y cinco son ochenta y cinco, de cien vinos: un ochenta y cinco por ciento.",
    CL(), T("A: 40 coinciden   ·   B: 45 coinciden", 960, 330, tam=54, c="tinta", w="sb", en="a"),
    T("(40 + 45) / 100 =", 940, 480, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(85, 970, 480, "{:.0f}", 76, "verde", suf=" %", anc="lm", en="b", t=1.0),
    ilustrativo(1050))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]PCA sirve para dibujar los grupos, pero es una sombra: aquí conserva el 55.3 por ciento de la variabilidad, y el ajuste se hace con las trece variables. [[b]]Con la variedad real, que no se usó, ciento setenta y uno de ciento setenta y siete vinos coinciden con su variedad: el algoritmo K-means recuperó la estructura. [[c]]Y K-means sufre con formas curvas, atípicos y grupos de tamaños muy distintos.",
    CL(), P("PCA: una sombra para mirar (55.3 %)", 960, 300, "teal", 48, en="a"),
    P("171 / 177 coinciden con la variedad real", 960, 460, "verde", 48, en="b"),
    P("límites: formas curvas, atípicos, tamaños distintos", 960, 620, "rojo", 46, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]¿y si las variables no son números, sino categorías? K-modes.",
    CL(), siguiente(10, "K-modes para variables categóricas"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
