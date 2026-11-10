import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, fila, adicional, barras_h, matriz, rh)

OUT = out("guion_c37_10_comparacion.json")
v = V("La comparación de los clasificadores")
SC = D["script"]
LG, LI, RA = SC["Logistica"], SC["SVM lineal"], SC["SVM radial"]
OT = D["otros"]
KN, LD, QD = OT["kNN (k = 5)"], OT["LDA"], OT["QDA"]


def cm(m):
    return [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]


def costo(m):
    return 10 * m["fn"] + m["fp"]


v.portada("La comparación", "Tres modelos de la clase, tres clásicos más, la misma prueba",
          "Décima parada: la comparación. Las matrices de confusión de los modelos de la clase en la misma prueba, cómo leerlas, y otros clasificadores clásicos.")

# ---- 1. Hook
v.b("Tres modelos, los mismos 171 casos de prueba. [[a]]En la clase, el profesor lo resumió así: la SVM radial solo tiene cinco errores; [[b]]la regresión logística, diez; y la SVM lineal, catorce. [[c]]¿Basta eso para elegir? Vamos a mirarlo con cuidado.",
    CL(), T("errores en los 171 casos de prueba", 960, 150, tam=52, c="tinta", ent="subir"),
    barras_h([("SVM radial", RA["fn"] + RA["fp"], "verde", "a"), ("logística", LG["fn"] + LG["fp"], "azul", "b"),
              ("SVM lineal", LI["fn"] + LI["fp"], "rojo", "b")], x0=700, y0=330, paso=150, escala=60, fmt="{:.0f}", tam=42, h=90, xl=660),
    P("¿basta eso para elegir?", 960, 840, "violeta", 50, en="c"), real(1050, en="a"), sec="1. Cinco, diez y catorce")

# ---- 2. Analogía
v.b("Piensa en comparar tres médicos. [[a]]Solo es justo si atienden a los mismos pacientes: si uno atiende casos fáciles y otro difíciles, la comparación no dice nada. [[b]]Aquí se cumple: los tres modelos usan exactamente la misma partición, en R y en Python. [[c]]Y aun así, con 171 pacientes, un caso de diferencia es poco más de medio punto porcentual.",
    CL(), T("tres médicos, los mismos pacientes", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "mismos pacientes", "si no, la comparación no dice nada", "a"), (960, "verde", "misma partición", "los tres modelos, R y Python", "b"),
              (1520, "ambar", "171 casos", "1 caso = 0.58 puntos", "c")], y=500, w=480, h=330, tam_t=46, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: tres médicos")

# ---- 3. Las tres matrices
XS = [380, 960, 1540]
MOD = [("logística (t = 0.5)", LG, "azul"), ("SVM lineal (C = 0.1)", LI, "ambar"), ("SVM radial (C = 10, γ = 0.1)", RA, "verde")]
v.b("Estas son las tres matrices de confusión. [[a]]La logística: 103, 4, 6 y 58. [[b]]La SVM lineal: 103, 4, 10 y 54. [[c]]Y la SVM radial: 106, 1, 4 y 60. Fíjate: la logística y la lineal tienen los mismos falsos positivos. Se diferencian en los malignos que se les escapan.",
    CL(), T("matrices de confusión · 171 casos de prueba", 960, 140, tam=48, c="tinta", ent="subir"),
    *[matriz(x, 470, cm(m), en=k, cel=150, tam=46, tit=nom, c=c, etiquetas=(i == 0)) for i, ((nom, m, c), x, k) in enumerate(zip(MOD, XS, "abc"))],
    P("mismos FP; distintos FN", 960, 760, "ambar", 44, en="c", t=1.5), real(1050, en="a"), sec="3. Las tres matrices")

# ---- 4. La tabla de la diapositiva 52
v.b("Con esas matrices sale la tabla de la diapositiva cincuenta y dos. [[a]]Exactitud: 0.9415, 0.9181 y 0.9708. [[b]]Sensibilidad: 0.9062, 0.8438 y 0.9375. [[c]]Especificidad: 0.9626, 0.9626 y 0.9907. Recalculamos todo desde los datos, y coincide exactamente con la diapositiva.",
    CL(), T("resultados en prueba · diapositiva 52", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(270, [("modelo", 440), ("exactitud", 900), ("sensibilidad", 1200), ("especificidad", 1520)], tams=[36] * 4, colores=["gris"] * 4),
    *[fila(370 + 110 * i, [(n.split(" (")[0], 440), (f"{m['acc']:.4f}", 900), (f"{m['sens']:.4f}", 1200), (f"{m['espe']:.4f}", 1520)],
           tams=[42, 46, 46, 46], colores=[c, "tinta", "tinta", "tinta"], en="a", t=0.2 * i, fondo=c + "claro", x0=240, x1=1700, h=96)
      for i, (n, m, c) in enumerate(MOD)],
    R(1200, 480, 260, 330, "violeta", r=20, en="b"), R(1520, 480, 300, 330, "teal", r=20, en="c"),
    P("recalculado desde los datos: coincide con la diapositiva", 960, 800, "verde", 44, en="c", t=1.2), real(1050, en="a"), sec="4. La tabla de la diapositiva 52")

# ---- 5. La cuenta de la SVM radial
v.b("Hagamos a mano la cuenta de la SVM radial, de la diapositiva cincuenta y tres. [[a]]Exactitud: 106 más 60, sobre 171: 0.9708. [[b]]Sensibilidad: 60 sobre 64: 0.9375. Especificidad: 106 sobre 107: 0.9907. [[c]]Y precisión: 60 sobre 61: 0.9836. En esta prueba hubo cuatro falsos negativos y un falso positivo.",
    CL(), T("SVM radial: la cuenta final · diapositiva 53", 960, 140, tam=48, c="tinta", ent="subir"),
    matriz(480, 520, cm(RA), cel=180, tam=58),
    T("exactitud = (106 + 60) / 171 =", 1460, 330, tam=40, c="tinta", w="sb", anc="rm", en="a"), N(RA["acc"], 1480, 330, "{:.4f}", 50, "azul", anc="lm", en="a", t=0.5),
    T("sensibilidad = 60 / 64 =", 1460, 440, tam=40, c="tinta", w="sb", anc="rm", en="b"), N(RA["sens"], 1480, 440, "{:.4f}", 50, "verde", anc="lm", en="b", t=0.4),
    T("especificidad = 106 / 107 =", 1460, 550, tam=40, c="tinta", w="sb", anc="rm", en="b", t=1.2), N(RA["espe"], 1480, 550, "{:.4f}", 50, "teal", anc="lm", en="b", t=1.6),
    T("precisión = 60 / 61 =", 1460, 660, tam=40, c="tinta", w="sb", anc="rm", en="c"), N(RA["prec"], 1480, 660, "{:.4f}", 50, "violeta", anc="lm", en="c", t=0.4),
    P("4 falsos negativos · 1 falso positivo", 1180, 800, "rojo", 44, en="c", t=1.5), real(1050, en="a"), sec="5. La cuenta de la SVM radial")

# ---- 6. Sensibilidad
v.b("La diapositiva cincuenta y cuatro mira la sensibilidad, porque aquí los errores no cuestan lo mismo. [[a]]De los 64 malignos, la logística detecta 58, [[b]]la SVM lineal, 54, [[c]]y la SVM radial, 60.",
    CL(), T("malignos detectados (de 64) · diapositiva 54", 960, 150, tam=50, c="tinta", ent="subir"),
    barras_h([("logística", LG["tp"], "azul", "a"), ("SVM lineal", LI["tp"], "ambar", "b"), ("SVM radial", RA["tp"], "verde", "c")],
             x0=700, y0=330, paso=150, escala=14, fmt="{:.0f}", tam=42, h=90, xl=660),
    L(700 + 64 * 14, 260, 700 + 64 * 14, 690, "gris", 3, guion=True), T("64", 700 + 64 * 14, 720, tam=32, c="gris", w="sb"),
    real(1050, en="a"), sec="6. La sensibilidad")
v.b("Pero cuidado con leer demasiado. [[a]]La diferencia entre la logística y la radial son dos malignos más detectados, y tres falsos positivos menos. [[b]]La propia diapositiva lo advierte: es una comparación docente de una sola partición. No implica superioridad universal, ni validación clínica.",
    P("radial frente a logística: +2 detectados, −3 falsas alarmas", 960, 840, "violeta", 42, en="a"),
    T("una sola partición: no implica superioridad universal (diapositiva 52)", 960, 950, tam=36, c="gris", w="sb", en="b"))

# ---- 7. Otros clasificadores clásicos
v.b("Hay otros clasificadores clásicos que vale la pena conocer. [[a]]Ya vimos kNN, los {k|k,} vecinos. [[b]]El análisis discriminante lineal, LDA, supone que cada clase es una nube normal, con la misma forma: por eso su frontera es una recta, o un hiperplano. [[c]]Y su versión cuadrática, QDA, deja que cada clase tenga su propia forma: la frontera se curva.",
    CL(), T("otros clasificadores clásicos", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "violeta", "kNN", "votan los k vecinos más cercanos", "a"), (960, "teal", "LDA", "nubes normales con la misma forma → frontera lineal", "b"),
              (1520, "ambar", "QDA", "cada clase con su forma → frontera curva", "c")], y=500, w=480, h=360, tam_t=56, tam_d=34),
    adicional(1050, en="a"), sec="7. Otros clasificadores clásicos")
OTROS = [("kNN (k = 5)", KN, "violeta"), ("LDA", LD, "teal"), ("QDA", QD, "ambar")]
v.b("Los ajustamos con la misma partición y las mismas cuatro variables estandarizadas. [[a]]kNN acierta 92.4 por ciento, con 9 falsos negativos. [[b]]LDA, 91.2, con la mejor especificidad, 0.981, pero 13 falsos negativos. [[c]]Y QDA, 92.4, con exactamente la misma matriz que kNN: 103, 4, 9 y 55.",
    CL(), T("misma partición, mismas 4 variables · prueba", 960, 140, tam=48, c="tinta", ent="subir"),
    *[matriz(x, 470, cm(m), en=k, cel=150, tam=46, tit=nom, c=c, etiquetas=(i == 0)) for i, ((nom, m, c), x, k) in enumerate(zip(OTROS, XS, "abc"))],
    *[T(f"exactitud {100 * m['acc']:.1f} % · espe {m['espe']:.3f}", x, 700, tam=32, c="tinta", w="sb", en=k, t=0.5) for (nom, m, c), x, k in zip(OTROS, XS, "abc")],
    adicional(1050, en="a"))

# ---- 8. La tabla completa
TODOS = [("SVM radial", RA, "verde"), ("logística", LG, "azul"), ("kNN (k = 5)", KN, "violeta"), ("QDA", QD, "ambar"),
         ("SVM lineal", LI, "ambar"), ("LDA", LD, "teal")]
v.b("Todos juntos, ordenados por sensibilidad. [[a]]Arriba, la SVM radial, con 60 de 64. Después, la logística, con 58. [[b]]kNN y QDA, con 55. La SVM lineal, con 54, y LDA, con 51. [[c]]Y la línea base, siempre benigno, con cero. Todos superan con creces a la línea base, y entre ellos las diferencias son de pocos casos.",
    CL(), T("seis clasificadores en la misma prueba, por sensibilidad", 960, 140, tam=46, c="tinta", ent="subir"),
    fila(240, [("modelo", 440), ("detectados", 860), ("FP", 1140), ("exactitud", 1420)], tams=[34] * 4, colores=["gris"] * 4),
    *[fila(320 + 86 * i, [(n, 440), (f"{m['tp']} / 64", 860), (str(m["fp"]), 1140), (f"{100 * m['acc']:.1f} %", 1420)], tams=[38, 42, 42, 42],
           colores=[c, "tinta", "tinta", "tinta"], en=("a" if i < 2 else "b"), t=0.15 * i, fondo="linea" if i % 2 else None, x0=240, x1=1640, h=78)
      for i, (n, m, c) in enumerate(TODOS)],
    fila(320 + 86 * 6, [("siempre benigno", 440), ("0 / 64", 860), ("0", 1140), ("62.6 %", 1420)], tams=[38, 42, 42, 42], colores=["rojo"] * 4, en="c", fondo="rojoclaro", x0=240, x1=1640, h=78),
    T("kNN, LDA y QDA: cálculo adicional", 1860, 1050, tam=26, c="violeta", w="sb", anc="rm", en="b"), sec="8. La tabla completa")

# ---- 9. Con costos
v.b("¿Cómo decidir entonces? Una forma es ponerle precio a cada error. [[a]]Supón, como ejemplo, que un falso negativo cuesta diez veces más que un falso positivo. El costo es diez por los falsos negativos, más los falsos positivos. [[b]]La SVM radial: diez por cuatro, más uno, cuarenta y uno. La logística: sesenta, más cuatro, sesenta y cuatro. [[c]]Y la SVM lineal: cien más cuatro, ciento cuatro. Con ese criterio, la diferencia entre modelos se ve mucho más grande.",
    CL(), T("ponerle precio a cada error", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("costo = 10 · FN + FP", y=300, c="rojo", tam=56, w=800, en="a"),
    T("SVM radial: 10 × 4 + 1 =", 1000, 450, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(costo(RA), 1030, 450, "{:.0f}", 58, "verde", anc="lm", en="b", t=0.5),
    T("logística: 10 × 6 + 4 =", 1000, 560, tam=46, c="tinta", w="sb", anc="rm", en="b", t=1.5), N(costo(LG), 1030, 560, "{:.0f}", 58, "azul", anc="lm", en="b", t=2.0),
    T("SVM lineal: 10 × 10 + 4 =", 1000, 670, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(costo(LI), 1030, 670, "{:.0f}", 58, "rojo", anc="lm", en="c", t=0.5),
    ilustrativo(1000, texto="ejemplo ilustrativo: costo 10 a 1 (los conteos son reales)"), sec="9. Ponerle precio a los errores")
v.b("Y la diapositiva cincuenta y siete agrega más criterios. [[a]]La logística entrega probabilidades y coeficientes interpretables como odds. [[b]]La SVM radial capta patrones curvos, pero es más difícil de interpretar, y su puntaje no es una probabilidad. [[c]]La elección combina la validación, la interpretabilidad, el costo de los errores y el costo computacional.",
    CL(), T("logística o SVM · diapositiva 57", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "logística", "probabilidad · coeficientes y odds · interpretable", "a"),
              (1360, "verde", "SVM radial", "patrones curvos · puntaje, no probabilidad · más difícil de leer", "b")], y=460, w=640, h=340, tam_t=52, tam_d=34),
    P("validación + interpretabilidad + costo de errores + costo computacional", 960, 790, "violeta", 38, en="c"))

# ---- 10. El error típico
v.b("El error típico: [[a]]comparar modelos evaluados con particiones distintas, y atribuirle toda la diferencia al algoritmo. [[b]]La diapositiva cincuenta y ocho lo pone en su lista de errores frecuentes. [[c]]Con otra partición al azar, la logística podría quedar por encima de la radial. Por eso aquí todos usan la misma columna conjunto.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("comparar con particiones distintas", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("diapositiva 58: errores frecuentes al implementar", 960, 400, tam=42, c="tinta", w="sb", en="b"),
    P("misma partición para todos (columna «conjunto»)", 960, 560, "verde", 48, en="c"), sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]El modelo X tiene 4 falsos negativos y 10 falsos positivos. El modelo Y, 8 falsos negativos y 2 falsos positivos. [[b]]Los dos se equivocan catorce veces. Con un falso negativo diez veces más caro, ¿cuál conviene?",
    CL(), P("X: FN = 4 · FP = 10", 960, 290, "azul", 52, en="a"), P("Y: FN = 8 · FP = 2", 960, 400, "ambar", 52, en="a", t=0.5),
    P("ambos: 14 errores · ¿cuál, con FN 10 veces más caro?", 960, 560, "violeta", 46, en="b"), ilustrativo(1000), sec="11. Piensa un segundo", pausa=0.6)
v.b("El modelo X cuesta diez por cuatro más diez: [[a]]cincuenta. [[b]]El modelo Y, diez por ocho más dos: [[c]]ochenta y dos. Con los mismos catorce errores, X es mucho mejor, porque se le escapan menos malignos.",
    CL(), T("X: 10 × 4 + 10 =", 960, 330, tam=52, c="tinta", w="sb", anc="rm"), N(50, 990, 330, "{:.0f}", 64, "azul", anc="lm", en="a"),
    T("Y: 10 × 8 + 2 =", 960, 460, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(82, 990, 460, "{:.0f}", 64, "ambar", anc="lm", en="c"),
    P("mismos errores, distinto costo: gana X", 960, 640, "verde", 50, en="c", t=0.8), ilustrativo(1000))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]con la misma partición, la SVM radial comete cinco errores; la logística, diez; la lineal, catorce. Y los números coinciden con la diapositiva. [[b]]kNN, LDA y QDA quedan en el mismo rango, entre 91 y 92 por ciento. [[c]]Y para elegir, no basta la exactitud: hay que mirar sensibilidad, el costo de cada error y cuánto se puede interpretar.",
    CL(), P("radial 5 · logística 10 · lineal 14 errores (= diapositiva 52)", 960, 300, "verde", 44, en="a"),
    P("kNN, LDA y QDA: entre 91 % y 92 %", 960, 460, "violeta", 46, en="b"),
    P("elegir: sensibilidad + costo de errores + interpretabilidad", 960, 620, "ambar", 44, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]cómo elegir {C|, ce,} y gamma sin tocar la prueba, con validación cruzada, y qué mide realmente la AUC.",
    CL(), siguiente(11, "Validación cruzada, búsqueda y AUC"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
