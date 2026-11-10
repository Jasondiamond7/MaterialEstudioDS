import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, fila, codigo, adicional, barras_h, matriz)

OUT = out("guion_c37_09_escalar.json")
v = V("Estandarizar: SVM y kNN")
SD = D["sd"]
ES = D["escala"]
E4 = FG["escala4d"]
OT = D["otros"]
VE, VS = D["vecinos"], D["vecinos_sin_escalar"]
NU = D["nuevo"]

v.portada("Estandarizar: SVM y kNN", "Cuando las unidades de medida deciden por ti",
          "Novena parada: estandarizar. Vas a ver con números por qué una SVM o los k vecinos más cercanos sin estandarizar terminan escuchando a una sola variable.")

# ---- 1. Hook
v.b("En la base de Wisconsin, la textura varía en unidades de a cuatro. [[a]]La suavidad, en cambio, varía en centésimas. [[b]]Si mides distancias sin hacer nada, una diferencia de una unidad de textura pesa lo mismo que una diferencia de cien veces todo el rango de la suavidad. [[c]]Sin estandarizar, la suavidad prácticamente no existe para el modelo.",
    CL(), T("las unidades deciden", 960, 160, tam=54, c="tinta", ent="subir"),
    P("textura: desviación estándar ≈ 4.2", 960, 330, "teal", 50),
    P("suavidad: desviación estándar ≈ 0.014", 960, 460, "violeta", 50, en="a", t=0.6),
    P("sin estandarizar, la suavidad «no existe»", 960, 650, "rojo", 52, en="c"), real(1050, en="a"), sec="1. Las unidades deciden")

# ---- 2. Analogía
v.b("Piensa en comparar dos departamentos por precio y por superficie. [[a]]Si el precio va en pesos y la superficie en metros cuadrados, una diferencia de mil pesos parece enorme frente a una de diez metros. [[b]]Pero mil pesos no es nada, y diez metros es una pieza entera. [[c]]Antes de sumar diferencias, hay que ponerlas en la misma escala.",
    CL(), T("precio y superficie", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(560, "ambar", "1 000 pesos", "número grande… diferencia mínima", "a"), (1360, "teal", "10 m²", "número chico… una pieza entera", "b")],
             y=480, w=640, h=320, tam_t=56, tam_d=38),
    P("antes de sumar diferencias: la misma escala", 960, 800, "violeta", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: pesos y metros")

# ---- 3. Las escalas reales
v.b("Estas son las desviaciones estándar de las cuatro variables, en el entrenamiento. [[a]]Radio, 3.67; textura, 4.22. [[b]]Concavidad, 0.080; y suavidad, 0.014. [[c]]Entre la textura y la suavidad hay un factor de casi trescientos.",
    CL(), T("desviación estándar en entrenamiento (398 casos)", 960, 150, tam=48, c="tinta", ent="subir"),
    barras_h([("radio", SD["radio"], "azul", "a"), ("textura", SD["textura"], "teal", "a"), ("concavidad", SD["concavidad"], "ambar", "b"),
              ("suavidad", SD["suavidad"], "violeta", "b")], x0=700, y0=320, paso=130, escala=200, fmt="{:.3f}", tam=40, h=80, xl=660),
    P("textura / suavidad ≈ 293 veces", 960, 900, "rojo", 48, en="c"), real(1050, en="a"), sec="3. Las escalas reales")

# ---- 4. Por qué importa: distancias
v.b("Veamos por qué importa, con dos casos reales. [[a]]El caso nuevo de la clase tiene radio 14 y concavidad 0.08. Un caso del entrenamiento tiene radio 14.26 y concavidad 0.30. [[b]]Sin estandarizar, las diferencias son 0.26 en radio y 0.22 en concavidad: parecen igual de pequeñas. [[c]]Pero en desviaciones estándar, el radio difiere 0.07, y la concavidad, 2.75. Son casos muy distintos.",
    CL(), T("la misma diferencia, en dos escalas", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(290, [("", 380), ("radio", 860), ("concavidad", 1360)], tams=[38, 40, 40], colores=["gris", "azul", "ambar"]),
    fila(390, [("caso nuevo", 380), ("14", 860), ("0.08", 1360)], tams=[38, 46, 46], en="a"),
    fila(480, [("caso del entrenamiento", 380), ("14.26", 860), ("0.3003", 1360)], tams=[38, 46, 46], en="a", t=0.4),
    fila(600, [("diferencia cruda", 380), ("0.26", 860), ("0.22", 1360)], tams=[38, 50, 50], colores=["tinta", "tinta", "tinta"], en="b", fondo="linea", x0=180, x1=1640, h=90),
    fila(720, [("en desviaciones estándar", 380), ("0.26 / 3.67 = 0.07", 860), ("0.22 / 0.080 = 2.75", 1360)], tams=[38, 42, 42],
         colores=["tinta", "azul", "rojo"], en="c", fondo="rojoclaro", x0=180, x1=1640, h=90),
    real(1050, en="a"), sec="4. Por qué importa")
v.b("La SVM radial y los {k|k,} vecinos miden distancias. [[a]]Si no estandarizas, la concavidad y la suavidad quedan aplastadas por el radio y la textura, solo por sus unidades.",
    P("distancias sin estandarizar → mandan las variables con números grandes", 960, 880, "ambar", 42, en="a"))

# ---- 5. Estandarizar
v.b("La receta de la diapositiva catorce: [[a]]a cada valor se le resta la media del entrenamiento, y se divide por la desviación estándar del entrenamiento. [[b]]Para el caso nuevo: radio 14 menos 14.179, dividido por 3.669, da menos 0.049. [[c]]Suavidad: 0.10 menos 0.09618, dividido por 0.0144, da 0.265.",
    CL(), T("estandarizar · diapositiva 14", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("z = (x − media del entrenamiento) / sd del entrenamiento", y=300, c="violeta", tam=48, w=1500, en="a"),
    T("radio:  (14 − 14.179) / 3.669 =", 1000, 470, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(NU["z"][0], 1030, 470, "{:.3f}", 58, "azul", anc="lm", en="b", t=0.6),
    T("suavidad:  (0.10 − 0.09618) / 0.0144 =", 1000, 590, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(NU["z"][2], 1030, 590, "{:.3f}", 58, "violeta", anc="lm", en="c", t=0.6),
    real(1050, en="b"), sec="5. Estandarizar")
v.b("En Python se hace con StandardScaler. [[a]]fit transform aprende la media y la escala con el entrenamiento, y lo transforma. [[b]]transform aplica esa misma media y escala a la prueba, y a cualquier caso nuevo. [[c]]En R, la función svm lo hace sola con scale igual a TRUE.",
    codigo(["escala = StandardScaler()", "x_entrenamiento_escalado = escala.fit_transform(x_entrenamiento)",
            "x_prueba_escalado = escala.transform(x_prueba)"], y=760, w=1700, tam=30, c="azul", en="a"),
    T("R: svm(…, scale = TRUE)", 960, 960, tam=38, c="teal", w="sb", en="c"))

# ---- 6. Efecto real en la SVM
KE = list(E4.keys())
v.b("¿Cuánto cambia en la SVM? Con las cuatro variables, en la prueba. [[a]]La SVM lineal de la clase: sin estandarizar, 86.0 por ciento; estandarizada, 91.8. [[b]]La radial de la clase, con C igual a diez y gamma 0.1: sin estandarizar, 90.1; estandarizada, 97.1. [[c]]Siete puntos de diferencia, solo por las unidades.",
    CL(), T("SVM en prueba: sin y con StandardScaler (4 variables)", 960, 150, tam=46, c="tinta", ent="subir"),
    fila(290, [("modelo", 520), ("sin estandarizar", 1040), ("estandarizado", 1460)], tams=[36, 36, 36], colores=["gris"] * 3),
    fila(400, [("lineal, C = 0.1", 520), (f"{100 * E4[KE[0]]['sin']:.1f} %", 1040), (f"{100 * E4[KE[0]]['con']:.1f} %", 1460)], tams=[42, 52, 52],
         colores=["tinta", "rojo", "verde"], en="a", fondo="linea", x0=280, x1=1700, h=100),
    fila(530, [("radial, C = 10, γ = 0.1", 520), (f"{100 * E4[KE[1]]['sin']:.1f} %", 1040), (f"{100 * E4[KE[1]]['con']:.1f} %", 1460)], tams=[42, 52, 52],
         colores=["tinta", "rojo", "verde"], en="b", fondo="linea", x0=280, x1=1700, h=100),
    fila(660, [("radial, C = 10, γ = «scale»", 520), (f"{100 * E4[KE[2]]['sin']:.1f} %", 1040), (f"{100 * E4[KE[2]]['con']:.1f} %", 1460)], tams=[42, 52, 52],
         colores=["tinta", "rojo", "verde"], en="c", fondo="linea", x0=280, x1=1700, h=100),
    P("hasta 7 puntos solo por las unidades", 960, 830, "ambar", 48, en="c", t=0.8), real(1050, en="a"), sec="6. El efecto en la SVM")

# ---- 7. kNN: el clasificador de los vecinos
v.b("Hay un clasificador clásico que depende todavía más de las distancias: los {k|k,} vecinos más cercanos, kNN. [[a]]No ajusta ningún coeficiente. Para clasificar un caso nuevo, busca los {k|k,} casos de entrenamiento más cercanos, [[b]]y vota: gana la clase más frecuente entre ellos. [[c]]En la clase se mencionó al pasar; aquí lo aplicamos a la misma base, como cálculo adicional.",
    CL(), T("los k vecinos más cercanos (kNN)", 960, 150, tam=52, c="tinta", ent="subir"),
    P("no ajusta coeficientes", 960, 300, "gris", 48, en="a"), P("busca los k casos más cercanos", 960, 420, "azul", 48, en="a", t=0.6),
    P("vota: gana la clase más frecuente", 960, 560, "violeta", 50, en="b"), adicional(1050, en="c"), sec="7. kNN: votar con los vecinos")

# ---- 8. Votación a mano
ROWS = [(f"{n['d']:.3f}", "maligno" if n["y"] else "benigno", "rojo" if n["y"] else "azul") for n in VE]
v.b("Votemos a mano, con el caso nuevo de la clase y k igual a cinco, en la escala estandarizada. [[a]]Estos son sus cinco vecinos más cercanos, con su distancia. [[b]]El primero, a 0.223, es maligno. Después, un benigno, dos malignos y un benigno. [[c]]Tres votos contra dos: kNN dice maligno, con una proporción de tres quintos, 0.6.",
    CL(), T("los 5 vecinos del caso nuevo (estandarizado)", 960, 150, tam=48, c="tinta", ent="subir"),
    fila(260, [("vecino", 560), ("distancia", 960), ("clase", 1360)], tams=[36, 36, 36], colores=["gris"] * 3),
    *[fila(350 + 90 * i, [(str(i + 1), 560), (d, 960), (c, 1360)], tams=[40, 44, 44], colores=["tinta", "tinta", col],
           en="a", t=0.2 * i, fondo=col + "claro", x0=420, x1=1500, h=78) for i, (d, c, col) in enumerate(ROWS)],
    P("3 malignos contra 2 benignos → maligno (3/5 = 0.6)", 960, 860, "rojo", 46, en="c"),
    adicional(1050, en="a"), sec="8. Votar a mano")
v.b("Compáralo con los otros modelos para el mismo caso. [[a]]La logística le da probabilidad 0.392: clase cero. [[b]]La SVM radial le da un puntaje de apenas 0.046: clase uno, pero casi sobre la frontera. [[c]]Es un caso límite, y por eso los modelos discrepan.",
    CL(), T("el mismo caso nuevo, tres modelos", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(330, [("logística", 600), (f"p = {NU['p_log']:.3f}", 1000), ("clase 0", 1400)], tams=[44, 46, 46], colores=["azul", "tinta", "azul"], en="a", fondo="azulclaro", x0=380, x1=1620, h=96),
    fila(460, [("SVM radial", 600), (f"f = {NU['f_radial']:.3f}", 1000), ("clase 1", 1400)], tams=[44, 46, 46], colores=["rojo", "tinta", "rojo"], en="b", fondo="rojoclaro", x0=380, x1=1620, h=96),
    fila(590, [("kNN (k = 5)", 600), ("3 de 5", 1000), ("clase 1", 1400)], tams=[44, 46, 46], colores=["violeta", "tinta", "rojo"], en="b", t=0.8, fondo="violetaclaro", x0=380, x1=1620, h=96),
    P("caso límite: los modelos discrepan", 960, 760, "ambar", 48, en="c"), real(1050, en="a"))

# ---- 9. Sin estandarizar, el voto cambia
ROWS2 = [(f"{n['d']:.3f}", f"{n['x'][3]:.3f}", "maligno" if n["y"] else "benigno", "rojo" if n["y"] else "azul") for n in VS]
v.b("Ahora repite la búsqueda sin estandarizar. [[a]]Los cinco vecinos cambian. El segundo es aquel caso con concavidad 0.30, que en desviaciones estándar estaba lejísimos. [[b]]Ahora el voto es dos malignos contra tres benignos. [[c]]kNN pasa a decir benigno, solo por las unidades de medida.",
    CL(), T("los 5 vecinos SIN estandarizar", 960, 150, tam=48, c="tinta", ent="subir"),
    fila(260, [("vecino", 480), ("distancia cruda", 860), ("concavidad", 1200), ("clase", 1500)], tams=[34, 34, 34, 34], colores=["gris"] * 4),
    *[fila(350 + 90 * i, [(str(i + 1), 480), (d, 860), (cc, 1200), (c, 1500)], tams=[40, 42, 42, 42], colores=["tinta", "tinta", ("rojo" if i == 1 else "tinta"), col],
           en="a", t=0.2 * i, fondo=col + "claro", x0=340, x1=1640, h=78) for i, (d, cc, c, col) in enumerate(ROWS2)],
    P("2 malignos contra 3 benignos → benigno", 960, 860, "azul", 46, en="b"), P("el voto cambió solo por las unidades", 960, 960, "ambar", 42, en="c"),
    adicional(1050, en="a"), sec="9. Sin estandarizar, el voto cambia")

# ---- 10. kNN en la prueba
KS, KR = OT["kNN (k = 5)"], OT["kNN sin escalar"]
v.b("En los 171 casos de prueba: [[a]]kNN con k igual a cinco, estandarizado, acierta el 92.4 por ciento, y detecta 55 de los 64 malignos. [[b]]Sin estandarizar, baja a 91.2, y detecta solo 52. [[c]]Y su AUC cae de 0.978 a 0.910. El orden de los puntajes empeora mucho más que la exactitud.",
    CL(), T("kNN (k = 5) en prueba", 960, 150, tam=50, c="tinta", ent="subir"),
    matriz(560, 520, [[KS["tn"], KS["fp"]], [KS["fn"], KS["tp"]]], en="a", cel=160, tam=50, tit="estandarizado", c="verde"),
    matriz(1360, 520, [[KR["tn"], KR["fp"]], [KR["fn"], KR["tp"]]], en="b", cel=160, tam=50, tit="sin estandarizar", c="rojo"),
    T(f"exactitud {100 * KS['acc']:.1f} % · AUC {KS['auc']:.3f}", 560, 750, tam=36, c="verde", w="sb", en="a", t=0.6),
    T(f"exactitud {100 * KR['acc']:.1f} % · AUC {KR['auc']:.3f}", 1360, 750, tam=36, c="rojo", w="sb", en="b", t=0.6),
    P("la AUC cae mucho más que la exactitud", 960, 880, "ambar", 44, en="c"), adicional(1050, en="a"), sec="10. kNN en la prueba")

# ---- 11. Sin fugas
v.b("Un cuidado clave, de la diapositiva trece: [[a]]la media y la desviación se calculan solo con los datos de ajuste. [[b]]Si se calculan con toda la base y después se divide, la escala ya vio la prueba: eso es fuga de información. [[c]]En validación cruzada, la escala se reestima dentro de cada fold. Por eso se pone StandardScaler dentro de un Pipeline.",
    CL(), T("sin fugas · diapositiva 13", 960, 150, tam=50, c="tinta", ent="subir"),
    P("media y sd: SOLO con los datos de ajuste", 960, 310, "teal", 48, en="a"),
    P("escalar toda la base antes de dividir = fuga", 960, 460, "rojo", 48, en="b"),
    codigo(["make_pipeline(StandardScaler(), SVC(kernel=\"rbf\"))"], y=640, w=1400, tam=34, c="violeta", en="c"),
    T("el Pipeline reestima la escala en cada fold", 960, 780, tam=40, c="gris", w="sb", en="c", t=0.5), sec="11. Sin fugas")

# ---- 12. El error típico
v.b("El error típico en el código: [[a]]usar fit transform también en la prueba. [[b]]Así la prueba se estandariza con su propia media y desviación, y no con las del entrenamiento. [[c]]Para la prueba, y para cada caso nuevo, siempre transform, con el objeto escala que se ajustó en el entrenamiento.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("escala.fit_transform(x_prueba)", 410, 230, tam=48, c="rojo", w="m", anc="lm", en="a"),
    T("la prueba usa su propia media y sd", 960, 420, tam=44, c="tinta", w="sb", en="b"),
    P("prueba y casos nuevos: escala.transform(…)", 960, 580, "verde", 48, en="c"), sec="12. El error típico")

# ---- 13. Piensa un segundo
v.b("Piensa un segundo. [[a]]La media de la textura en el entrenamiento es 19.29, y su desviación estándar, 4.21. [[b]]¿Cuánto vale la textura estandarizada de un caso con textura 25?",
    CL(), P("media = 19.29   ·   sd = 4.21", 960, 330, "teal", 52, en="a"), P("textura = 25  →  z = ?", 960, 480, "violeta", 54, en="b"),
    real(1050, en="a"), sec="13. Piensa un segundo", pausa=0.6)
v.b("25 menos 19.29 es 5.71. [[a]]Dividido por 4.21: 1.36. [[b]]Ese caso está algo más de una desviación estándar y un tercio por encima del promedio de entrenamiento.",
    CL(), T("z = (25 − 19.29) / 4.21 = 5.71 / 4.21 =", 1060, 380, tam=48, c="tinta", w="sb", anc="rm"), N(1.36, 1090, 380, "{:.2f}", 62, "violeta", anc="lm", en="a"),
    P("1.36 desviaciones estándar sobre la media", 960, 560, "teal", 48, en="b"))

# ---- 14. Para llevar
v.b("Para llevarte: [[a]]la SVM radial y kNN miden distancias, así que sin estandarizar mandan las variables con números grandes. [[b]]Se estandariza con la media y la desviación del entrenamiento, dentro de un Pipeline. [[c]]Y kNN vota con sus k vecinos: en la prueba, 92.4 por ciento estandarizado, frente a 91.2 sin estandarizar.",
    CL(), P("distancias sin estandarizar → mandan los números grandes", 960, 300, "ambar", 46, en="a"),
    P("media y sd del entrenamiento, dentro de un Pipeline", 960, 460, "teal", 46, en="b"),
    P("kNN: votan los k vecinos (92.4 % frente a 91.2 %)", 960, 620, "violeta", 46, en="c"), sec="14. Para llevar")
v.b("Siguiente parada: [[a]]la gran comparación. Logística, SVM lineal, SVM radial, y otros clasificadores clásicos, con sus matrices de confusión en la prueba.",
    CL(), siguiente(10, "La comparación de los clasificadores"), sec="15. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
