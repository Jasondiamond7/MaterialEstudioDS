import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, alerta, k1, fila

OUT = out("guion_c35_09_importancia.json")
v = V("Importancia y límites")
IM = FG["imp"]
IMG = FG["img"]
EX = D["prop_extrap"]

# importancia por impureza a mano (ejemplo del apunte), con intermedios redondeados como en pantalla
gA = round(1 - 0.9 ** 2 - 0.1 ** 2, 4)                       # 0.18
gD = round(1 - 0.26 ** 2 - 0.74 ** 2, 4)                     # 0.3848
pondA = round((30 * gA + 50 * gD) / 80, 3)                   # 0.308
dA = round(0.5 - pondA, 3)                                   # 0.192
gB1 = round(1 - 0.55 ** 2 - 0.45 ** 2, 4)                    # 0.495
gB2 = round(1 - (2 / 30) ** 2 - (28 / 30) ** 2, 4)           # 0.1244
pondB = round((20 * gB1 + 30 * gB2) / 50, 4)                 # 0.2726
dB = round(gD - pondB, 4)                                    # 0.1122
aB = round(dB * 50 / 80, 4)                                  # 0.0701
tot = round(dA + aB, 4)                                      # 0.2621
iA, iB = round(dA / tot, 3), round(aB / tot, 3)

v.portada("Importancia y límites", "Cómo explicar un bosque, y dónde no conviene usarlo",
          "Novena parada: un bosque de trescientos árboles no se puede dibujar. Vas a ver cómo medir qué variables usa, por qué una de las medidas engaña, y cuáles son sus límites.")

# ---- 1. Hook
v.b("Un árbol de profundidad tres cabe en una lámina. [[a]]Un bosque de trescientos árboles no se puede dibujar. [[b]]Entonces, ¿cómo le explicas a alguien qué aprendió el modelo? [[c]]La respuesta habitual es la importancia de las variables. Pero cuidado: hay dos formas de medirla, y una engaña.",
    CL(), T("1 árbol: cabe en una lámina", 560, 300, tam=48, c="verde", en=None, ent="subir"), T("300 árboles: no se pueden dibujar", 1360, 300, tam=48, c="rojo", en="a"),
    T("¿cómo explicas qué aprendió?", 960, 480, tam=62, c="tinta", en="b"), P("importancia de variables · dos formas · una engaña", 960, 660, "ambar", 46, en="c"),
    sec="1. Explicar un bosque")

# ---- 2. Analogía
v.b("Piensa en un equipo de fútbol. [[a]]Una forma de medir a un jugador es contar cuántas veces tocó la pelota. [[b]]Otra es sacarlo de la cancha, y ver cuánto empeora el equipo. [[c]]La primera premia al que corre mucho aunque no aporte; la segunda mide lo que de verdad hace falta.",
    CL(), T("¿cuánto vale un jugador?", 960, 180, tam=54, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "contar toques", "premia al que corre mucho, aporte o no", "a"), (1360, "verde", "sacarlo de la cancha", "mide cuánto empeora el equipo sin él", "b")],
             y=470, w=660, h=320),
    P("toques ≈ impureza · sacarlo ≈ permutación", 960, 780, "violeta", 46, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: el jugador")

# ---- 3. Impureza a mano
v.b("La primera medida es la importancia por impureza, MDI. [[a]]Suma, para cada variable, cuánto bajó el Gini en los cortes que hizo, pesado por la fracción de datos que llegó a ese nodo. [[b]]Hagamos un ejemplo del apunte: ochenta casos, cuarenta de cada clase. El Gini de la raíz es 0.5.",
    CL(), T("importancia por impureza (MDI)", 960, 160, tam=50, c="tinta", ent="subir"),
    caja_formula("MDI(variable) = Σ (N_t / N) · ΔGini en sus cortes", y=290, c="azul", tam=46, w=1300, en="a"),
    Bc(960, 470, 300, 100, "blanco", "tinta", 4, 20, en="b", ent="pop"), T("raíz · 80 (40/40)", 960, 450, tam=30, c="tinta", w="sb", en="b", t=0.1),
    T("Gini 0.5", 960, 490, tam=32, c="tinta", w="b", en="b", t=0.2), ilustrativo(1000, texto="ejemplo del apunte"), sec="3. Impureza, a mano")
v.b("La raíz se corta con la variable A. [[a]]A la izquierda quedan treinta casos, veintisiete contra tres: Gini 0.18. A la derecha, cincuenta casos, trece contra treinta y siete: Gini 0.3848. [[b]]El Gini ponderado es 0.308, y la bajada por A es 0.5 menos 0.308: 0.192.",
    L(960, 520, 640, 610, "grisclaro", 5), L(960, 520, 1280, 610, "grisclaro", 5), T("A", 790, 545, tam=30, c="azul", w="b"),
    Bc(640, 660, 300, 100, "blanco", "azul", 4, 20, en="a", ent="pop"), T("30 (27/3)", 640, 640, tam=30, c="tinta", w="sb", en="a", t=0.1), T(f"Gini {gA:.2f}", 640, 680, tam=30, c="azul", w="b", en="a", t=0.2),
    Bc(1280, 660, 300, 100, "blanco", "azul", 4, 20, en="a", t=0.4, ent="pop"), T("50 (13/37)", 1280, 640, tam=30, c="tinta", w="sb", en="a", t=0.5), T(f"Gini {gD:.4f}", 1280, 680, tam=30, c="azul", w="b", en="a", t=0.6),
    T(f"ΔA = 0.5 − (30 · {gA:.2f} + 50 · {gD:.4f}) / 80 = 0.5 − {pondA:.3f} = {dA:.3f}", 960, 800, tam=38, c="azul", w="b", en="b"))
v.b("El nodo derecho se corta con la variable B, en veinte y treinta casos, con Gini 0.495 y 0.1244. [[a]]La bajada en ese nodo es 0.3848 menos 0.2726: 0.1122. [[b]]Pero a ese nodo llegó solo cincuenta de ochenta casos, así que se pesa por 0.625: aporta 0.0701.",
    T(f"ΔB = {gD:.4f} − (20 · {gB1:.3f} + 30 · {gB2:.4f}) / 50 = {gD:.4f} − {pondB:.4f} = {dB:.4f}", 960, 880, tam=34, c="violeta", w="b", en="a"),
    T(f"peso 50/80 = 0.625  →  {dB:.4f} × 0.625 = {aB:.4f}", 960, 950, tam=36, c="violeta", w="b", en="b"))
v.b("Total: 0.192 más 0.0701, igual a 0.2621. [[a]]Normalizando, A vale 0.733, y B vale 0.267. [[b]]La variable que cortó primero, con más datos, se lleva casi tres cuartos de la importancia.",
    CL(), T(f"total = {dA:.3f} + {aB:.4f} = {tot:.4f}", 960, 300, tam=54, c="tinta", w="sb", ent="subir"),
    T(f"A = {dA:.3f} / {tot:.4f} =", 900, 450, tam=52, c="tinta", w="sb", anc="rm", en="a"), N(iA, 930, 450, "{:.3f}", 70, "azul", anc="lm", en="a", t=0.3),
    T(f"B = {aB:.4f} / {tot:.4f} =", 900, 570, tam=52, c="tinta", w="sb", anc="rm", en="a", t=0.6), N(iB, 930, 570, "{:.3f}", 70, "violeta", anc="lm", en="a", t=0.9),
    P("cortar primero, con más datos → casi 3/4 de la importancia", 960, 740, "ambar", 44, en="b"), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 4. La trampa: ruido
v.b("¿Dónde está la trampa? [[a]]Agregamos a Heart dos columnas de ruido puro, números al azar sin ninguna relación con la enfermedad: una continua y una binaria. [[b]]Y entrenamos un bosque de quinientos árboles. Por impureza, el ruido continuo queda en el puesto siete de quince, con un 6.9 % de la importancia: por encima de variables clínicas reales como el colesterol o la presión.",
    CL(), T("Heart + 2 columnas de RUIDO puro", 960, 160, tam=50, c="tinta", ent="subir"),
    T("impureza (MDI), en el entrenamiento", 560, 260, tam=38, c="azul", en="b"),
    *[[T(nm, 400, 340 + 70 * i, tam=32, c=("rojo" if nm.startswith("Ruido") else "tinta"), w="sb", anc="rm", en="b", t=0.06 * i),
       Bx(420, 340 + 70 * i - 22, 420 + val * 2400, 340 + 70 * i + 22, ("rojo" if nm.startswith("Ruido") else "azul"), None, 0, 8, grow="x", en="b", t=0.06 * i + 0.1),
       T(f"{val * 100:.1f} %", 440 + val * 2400, 340 + 70 * i, tam=28, c="tinta", w="sb", anc="lm", en="b", t=0.06 * i + 0.3)]
      for i, (nm, val) in enumerate(list(IM["mdi_top"].items()) + [("Ruido_continuo", IM["mdi_ruido"]["Ruido_continuo"])])],
    T(f"puesto {IM['rank_ruido_cont']} de 15", 560, 820, tam=36, c="rojo", w="b", en="b", t=1.0), real(1050), sec="4. La trampa: ruido puro")
v.b("¿Por qué? [[a]]La impureza se calcula con los datos de entrenamiento, y una variable continua tiene cientos de cortes posibles: siempre encuentra alguno que, por azar, separa un poco. [[b]]La segunda medida, la de permutación, desordena una variable en datos que el modelo no vio, y mide cuánto empeora. [[c]]El ruido continuo: apenas 0.001 de AUC. Prácticamente cero, como corresponde.",
    T("permutación, en el test", 1360, 260, tam=38, c="verde", en="b"),
    *[[T(nm, 1240, 340 + 70 * i, tam=32, c=("rojo" if nm.startswith("Ruido") else "tinta"), w="sb", anc="rm", en="b", t=0.06 * i),
       Bx(1260, 340 + 70 * i - 22, 1260 + max(val, 0.0005) * 9000, 340 + 70 * i + 22, ("rojo" if nm.startswith("Ruido") else "verde"), None, 0, 8, grow="x", en="b", t=0.06 * i + 0.1),
       T(f"{val:.3f}", 1280 + max(val, 0.0005) * 9000, 340 + 70 * i, tam=28, c="tinta", w="sb", anc="lm", en="b", t=0.06 * i + 0.3)]
      for i, (nm, val) in enumerate(list(IM["perm_top"].items())[:5] + [("Ruido_continuo", IM["perm_ruido"]["Ruido_continuo"])])],
    T("caída del AUC al desordenar", 1360, 820, tam=30, c="gris", w="sb", en="b", t=0.6),
    P("muchos cortes posibles + datos de entrenamiento = sesgo de la impureza", 960, 900, "ambar", 36, en="a"),
    P(f"ruido por permutación: {IM['perm_ruido']['Ruido_continuo']:.3f} ≈ 0", 1360, 970, "verde", 34, en="c"))

# ---- 5. Ames
v.b("En Ames las dos medidas tampoco coinciden. [[a]]Por permutación, la variable más importante es el área habitable, con un 22.5 % del total: si la desordenas, el error sube 11 900 dólares. [[b]]Por impureza, el área habitable vale solo un 12.2 %, y el primer lugar se lo lleva el número de autos que caben en el garaje.",
    CL(), T("Ames · las dos medidas no coinciden", 960, 170, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "permutación", f"Gr_Liv_Area: {IM['ames_perm']['Gr_Liv_Area'] * 100:.1f} % (+{k1(IM['ames_perm_rmse']['Gr_Liv_Area'])} de RMSE)", "a"),
              (1360, "azul", "impureza", f"Garage_Cars primero ({IM['ames_mdi']['Garage_Cars'] * 100:.1f} %) · Gr_Liv_Area {IM['ames_mdi']['Gr_Liv_Area'] * 100:.1f} %", "b")],
             y=470, w=700, h=330, tam_t=50, tam_d=34),
    P("para explicar: permutación, en datos no vistos", 960, 780, "violeta", 46, en="b", t=0.8), real(1050), sec="5. En Ames")

# ---- 6. Límites: no extrapola
v.b("Ahora los límites. El primero: un bosque de regresión no extrapola. [[a]]Lo probamos: entrenamos un bosque solo con casas de menos de 1 800 pies cuadrados. La casa más cara que vio costaba 392 000 dólares. [[b]]Y le pedimos predecir 133 casas grandes, de más de 2 400 pies cuadrados.",
    CL(), T("límite 1: no extrapola", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "entrena con", f"casas < 1.800 pies² · la más cara: {EX['max_y_entrenado'] / 1000:.0f} mil", "a"),
              (1360, "violeta", "predice", f"{EX['n_grandes']} casas > 2.400 pies²", "b")], y=420, w=660, h=280),
    real(1050, texto="cálculo propio con Ames"), sec="6. Límite: no extrapola")
v.b("El precio real promedio de esas casas grandes es 316 700 dólares. [[a]]El bosque predice, en promedio, 222 900: unos 94 000 dólares menos. [[b]]¿Por qué? Cada hoja promedia precios que vio. Ninguna predicción puede superar el precio más alto del entrenamiento: la mayor fue 337 500.",
    T("real promedio", 560, 680, tam=38, c="tinta", w="sb"), N(EX["media_real_grandes"] / 1000, 560, 770, "{:.1f}", 70, "tinta", suf=" mil"),
    T("predicho promedio", 1360, 680, tam=38, c="rojo", w="sb", en="a"), N(EX["media_pred_grandes"] / 1000, 1360, 770, "{:.1f}", 70, "rojo", suf=" mil", en="a", t=0.3),
    P(f"cada hoja promedia lo visto: máxima predicción {EX['max_pred_grandes'] / 1000:.1f} mil < {EX['max_y_entrenado'] / 1000:.0f} mil", 960, 920, "ambar", 38, en="b"))

# ---- 7. Imágenes
v.b("El segundo límite aparece con imágenes. El notebook del profesor usa CIFAR-10, que requiere descargar unos 170 megas. [[a]]Aquí usamos un reemplazo liviano: los dígitos escritos a mano de scikit-learn, imágenes de ocho por ocho píxeles, una variable por píxel. [[b]]El bosque acierta el 96.9 % en el test. Impresionante.",
    CL(), T("límite 2: imágenes", 960, 160, tam=52, c="tinta", ent="subir"),
    T("reemplazo de CIFAR-10: dígitos de scikit-learn (8 × 8 = 64 variables)", 960, 270, tam=38, c="gris", w="sb", en="a"),
    Pt([(64, [760, 340, 1160, 740], "gris")], 18, en="a", t=0.2),
    T("test", 1500, 470, tam=40, c="teal", en="b"), N(IMG["test"], 1500, 560, "{:.3f}", 80, "teal", en="b", t=0.3),
    real(1050, texto="reemplazo de CIFAR-10 (dígitos de scikit-learn)"), sec="7. Límite: imágenes")
v.b("Ahora dos experimentos. [[a]]Primero, desordenamos los píxeles de todas las imágenes, siempre con la misma permutación, y reentrenamos: el bosque acierta 97.3 %. Igual que antes, con imágenes que para ti son ruido. [[b]]Segundo, al modelo original le mostramos las imágenes de prueba corridas un solo píxel a la derecha: cae a 50.4 %.",
    CL(), T("dos experimentos con los mismos dígitos", 960, 170, tam=48, c="tinta", ent="subir"),
    *[[Bx(x - 140, 860 - val * 520, x + 140, 860, c, None, 0, 12, grow="y", en=m), N(val, x, 860 - val * 520 - 44, "{:.3f}", 56, c, en=m, t=0.3),
       T(lab, x, 910, tam=32, c=c, w="sb", mw=360, en=m)]
      for x, val, lab, c, m in ((560, IMG["test"], "test original", "teal", None), (960, IMG["shuffled"], "píxeles desordenados", "gris", "a"),
                                (1360, IMG["shifted"], "corridas 1 píxel", "rojo", "b"))],
    L(380, 860, 1540, 860, "tinta", 4), real(1050, texto="reemplazo de CIFAR-10 (dígitos de scikit-learn)"))
v.b("La conclusión: [[a]]el bosque no sabe qué píxel está al lado de cuál. Para él, cada píxel es una columna más. [[b]]Por eso la diapositiva dieciséis dice que en imágenes, texto o series largas, los modelos especializados suelen ser más adecuados. Donde el bosque brilla es en datos tabulares.",
    CL(), P("el bosque no sabe qué píxel está al lado de cuál", 960, 300, "rojo", 48, ent="subir"),
    T("cada píxel = una columna más", 960, 420, tam=44, c="gris", w="sb", en="a"),
    P("imágenes, texto, series largas → modelos especializados", 960, 580, "violeta", 44, en="b"), P("brilla en datos TABULARES", 960, 720, "verde", 50, en="b", t=0.8))

# ---- 8. Ventajas y limitaciones
v.b("La diapositiva quince lo resume. [[a]]Ventajas: buen desempeño sin suponer relaciones lineales, menos varianza que un árbol, interacciones y umbrales automáticos, poco sensible a la escala de las variables, y el OOB incluido. [[b]]Limitaciones: menos interpretable, más memoria y tiempo de predicción, la importancia por impureza puede engañar, y no extrapola en regresión.",
    CL(), T("diapositiva 15 · ventajas y limitaciones", 960, 150, tam=46, c="tinta", ent="subir"),
    Tj(530, 560, 780, 640, "ventajas", ["sin suponer linealidad", "menos varianza que un árbol", "interacciones y umbrales", "poco sensible a la escala", "OOB incluido"],
       c="verde", en="a", tam=34),
    Tj(1390, 560, 780, 640, "limitaciones", ["menos interpretable", "más memoria y tiempo", "la impureza puede engañar", "no extrapola (regresión)"],
       c="rojo", en="b", tam=34), sec="8. Ventajas y limitaciones")

# ---- 9. Error típico
v.b("El error típico, también en la diapositiva diecisiete: [[a]]interpretar la importancia como causalidad. [[b]]Que el bosque use mucho una variable para predecir no significa que esa variable cause el resultado. [[c]]La importancia describe al modelo, no al mundo.",
    CL(), T("«Thal es importante → Thal causa la enfermedad»", 960, 280, tam=54, c="tinta", en="a", id="e1"),
    T("usar una variable para predecir ≠ causar el resultado", 960, 420, tam=44, c="gris", w="sb", en="b"),
    U("e1", tachado=1.0, color="gris", en="c", dur=0.6), P("la importancia describe al MODELO, no al mundo", 960, 600, "verde", 50, en="c", t=0.4),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si desordenaste los píxeles y el bosque acertó igual, 97.3 %, ¿qué te dice eso sobre lo que aprendió?",
    CL(), P(f"píxeles desordenados: {IMG['shuffled']:.3f}", 960, 300, "gris", 56, en="a"), T("¿qué aprendió, entonces?", 960, 480, tam=70, c="tinta", en="a", t=0.8),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("Que aprendió qué píxeles concretos suelen estar encendidos para cada dígito, [[a]]como una lista de casillas, no como una forma. [[b]]Por eso un corrimiento de un píxel lo desarma: las casillas encendidas ya no son las mismas, aunque para ti el dígito sea idéntico.",
    CL(), T("aprendió casillas encendidas, no formas", 960, 300, tam=56, c="tinta", ent="subir"),
    T("una lista de píxeles concretos por dígito", 960, 420, tam=44, c="gris", w="sb", en="a"),
    P(f"corrido 1 píxel → otras casillas → {IMG['shifted']:.3f}", 960, 600, "rojo", 50, en="b"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]la importancia por impureza se calcula en el entrenamiento y premia variables con muchos cortes, incluso ruido. [[b]]Para explicar, usa la de permutación, sobre datos no vistos, y no la leas como causalidad. [[c]]Y conoce los límites: no extrapola, y en imágenes no ve la estructura.",
    CL(), P("impureza: entrenamiento · premia muchos cortes, incluso ruido", 960, 300, "rojo", 44, en="a"),
    P("explicar: permutación en datos no vistos · no es causalidad", 960, 470, "verde", 44, en="b"), P("límites: no extrapola · no ve estructura espacial", 960, 640, "ambar", 46, en="c"),
    sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la comparación final entre árbol, logística, bagging y Random Forest. ¿Quién gana?",
    CL(), siguiente(10, "La comparación final"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
