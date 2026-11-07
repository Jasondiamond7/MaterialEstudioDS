import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, alerta, k1, cien, r3

OUT = out("guion_c36_10_comparacion.json")
v = V("La comparación y cómo elegir")
CH, CA = FG["cmp_h"], FG["cmp_a"]
NH, NA = D["nb_heart"], D["nb_ames"]
MOD = ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "Random Forest"]
COL = {"AdaBoost": "rojo", "Gradient Boosting": "ambar", "XGBoost": "violeta", "LightGBM": "verde", "Random Forest": "azul"}

v.portada("La comparación", "Cinco modelos, los mismos datos: ¿quién gana, y cuánto importa?",
          "Décima parada: la comparación. Los modelos de la clase, con los mismos datos y la misma validación. Y cómo elegir.")

# ---- 1. Hook
v.b("Llegó el momento de la verdad. [[a]]AdaBoost, Gradient Boosting, XGBoost, LightGBM, y el Random Forest de la clase anterior como referencia. [[b]]Los mismos datos, el mismo preprocesamiento. [[c]]¿Quién gana? Y, más importante: ¿la diferencia es real?",
    CL(), *[P(m, x, 330, COL[m], 36, en="a", t=0.25 * i) for i, (m, x) in enumerate(zip(MOD, [300, 620, 930, 1180, 1500]))],
    T("mismos datos · mismo preprocesamiento", 960, 500, tam=48, c="tinta", w="sb", en="b"),
    T("¿quién gana? ¿la diferencia es real?", 960, 660, tam=66, c="tinta", en="c"), sec="1. El momento de la verdad")

# ---- 2. Heart con validación cruzada
orden_h = sorted(MOD, key=lambda m: -CH[m][0])
v.b("Primero Heart, 303 pacientes, clasificación. [[a]]Usamos validación cruzada de cinco pliegues, repetida tres veces: quince evaluaciones por modelo, y el AUC promedio. [[b]]AdaBoost, " + r3(CH["AdaBoost"][0]) + ". Random Forest, " + r3(CH["Random Forest"][0]) +
    ". [[c]]XGBoost, " + r3(CH["XGBoost"][0]) + ". LightGBM, " + r3(CH["LightGBM"][0]) + ". Y Gradient Boosting, " + r3(CH["Gradient Boosting"][0]) + ".",
    CL(), T("Heart · AUC con validación cruzada 5 × 3 (media ± desvío)", 960, 140, tam=44, c="tinta", ent="subir"),
    T("303 pacientes · 15 evaluaciones por modelo", 960, 205, tam=32, c="gris", w="sb", en="a"),
    *[[T(m, 560, 300 + 120 * i, tam=40, c=COL[m], anc="rm", en=("b" if i < 2 else "c"), t=0.4 * (i % 2 if i < 2 else (i - 2))),
       Bx(600, 300 + 120 * i - 32, 600 + (CH[m][0] - 0.80) * 6000, 300 + 120 * i + 32, COL[m], None, 0, 12, grow="x",
          en=("b" if i < 2 else "c"), t=0.4 * (i % 2 if i < 2 else (i - 2)) + 0.1),
       L(600 + (CH[m][0] - CH[m][1] - 0.80) * 6000, 300 + 120 * i, 600 + (CH[m][0] + CH[m][1] - 0.80) * 6000, 300 + 120 * i, "tinta", 3,
         en=("b" if i < 2 else "c"), t=0.4 * (i % 2 if i < 2 else (i - 2)) + 0.5),
       T(f"{r3(CH[m][0])} ± {CH[m][1]:.3f}", 1560, 300 + 120 * i, tam=36, c="tinta", w="b", anc="lm", en=("b" if i < 2 else "c"),
         t=0.4 * (i % 2 if i < 2 else (i - 2)) + 0.6)] for i, m in enumerate(orden_h)],
    T("eje desde 0.80", 600, 855, tam=26, c="gris", w="sb", anc="lm", en="a"), real(1050), sec="2. Heart con validación cruzada")
v.b("¿Ganó AdaBoost? [[a]]Mira las rayas negras: el desvío entre las quince evaluaciones va de 0.035 a 0.044. [[b]]La distancia entre el primero y el último es " + r3(CH["AdaBoost"][0] - CH["Gradient Boosting"][0]) +
    ": cabe dentro del desvío. [[c]]Con 303 pacientes, los cinco modelos están prácticamente empatados.",
    P("desvíos: de 0.035 a 0.044", 960, 920, "gris", 40, en="a"),
    T(f"primero − último = {r3(CH['AdaBoost'][0])} − {r3(CH['Gradient Boosting'][0])} = {r3(CH['AdaBoost'][0] - CH['Gradient Boosting'][0])}", 960, 990, tam=34, c="tinta", w="sb", en="b"),
    P("prácticamente empatados", 1600, 920, "ambar", 36, en="c"))

# ---- 3. Ames con partición de test
orden_a = sorted(MOD, key=lambda m: CA[m][0])
v.b("Ahora Ames, 2 930 casas, regresión. [[a]]RMSE en el test, con la configuración del notebook: XGBoost, " + cien(CA["XGBoost"][0]) + " dólares. Gradient Boosting, " + cien(CA["Gradient Boosting"][0]) +
    ". [[b]]LightGBM, " + cien(CA["LightGBM"][0]) + ". Random Forest, " + cien(CA["Random Forest"][0]) + ". [[c]]Y AdaBoost, muy atrás: " + cien(CA["AdaBoost"][0]) + ".",
    CL(), T("Ames · RMSE en test (miles de USD) · menor es mejor", 960, 140, tam=44, c="tinta", ent="subir"),
    *[[T(m, 560, 280 + 120 * i, tam=40, c=COL[m], anc="rm", en=("a" if i < 2 else ("b" if i < 4 else "c")), t=0.6 * (i % 2)),
       Bx(600, 280 + 120 * i - 32, 600 + CA[m][0] / 40, 280 + 120 * i + 32, COL[m], None, 0, 12, grow="x", en=("a" if i < 2 else ("b" if i < 4 else "c")),
          t=0.6 * (i % 2) + 0.1),
       T(k1(CA[m][0]), 620 + CA[m][0] / 40, 280 + 120 * i, tam=38, c="tinta", w="b", anc="lm", en=("a" if i < 2 else ("b" if i < 4 else "c")),
         t=0.6 * (i % 2) + 0.6)] for i, m in enumerate(orden_a)],
    real(1050), sec="3. Ames con test")
v.b("Aquí sí hay diferencias claras. [[a]]Con más datos y relaciones más complejas, los métodos de gradiente despegan, y AdaBoost queda atrás. [[b]]La idea general: a problema simple, modelo simple; a problema complejo, modelo más complejo. Y el consejo de siempre: probar, y validar.",
    P("más datos y más complejidad: los de gradiente despegan", 960, 900, "violeta", 40, en="a"),
    T("problema simple → modelo simple · complejo → más complejo · y probar", 960, 990, tam=34, c="tinta", w="sb", en="b"))

# ---- 4. El notebook: una sola partición
v.b("El notebook del profesor compara con una sola partición de test: 76 pacientes. [[a]]Lo replicamos: XGBoost y LightGBM aciertan 65 de 76; AdaBoost y Gradient Boosting, 64. [[b]]Un solo paciente de diferencia. [[c]]Y un paciente de 76 son 1.3 puntos de exactitud: no es evidencia de que uno sea mejor.",
    CL(), T("el notebook: una sola partición de test (76 pacientes)", 960, 150, tam=44, c="tinta", ent="subir"),
    *[[T(m, 760, 300 + 100 * i, tam=40, c=COL[m], anc="rm", en="a", t=0.3 * i),
       T(f"{NH[m]['aciertos']} de 76 · AUC {r3(NH[m]['auc'])}", 800, 300 + 100 * i, tam=40, c="tinta", w="b", anc="lm", en="a", t=0.3 * i + 0.2)]
      for i, m in enumerate(["XGBoost", "LightGBM", "AdaBoost", "Gradient Boosting"])],
    P("1 paciente de diferencia", 960, 760, "ambar", 48, en="b"), T("1 / 76 = 1.3 puntos de exactitud: no es evidencia", 960, 860, tam=42, c="tinta", w="sb", en="c"),
    real(1050, texto="réplica del notebook (FAST_MODE = True)"), sec="4. Una sola partición")
v.b("El propio notebook lo reconoce al final: [[a]]una sola partición no determina un ganador universal. [[b]]Para una evaluación sólida, validación cruzada anidada, o varias semillas.",
    CL(), P("una sola partición no determina un ganador", 960, 380, "rojo", 50, en="a"),
    P("CV anidada o varias semillas", 960, 540, "verde", 50, en="b"))

# ---- 5. La tabla de la diapositiva 16 (alerta)
FILAS = [("AdaBoost", "pondera casos difíciles", "codificar", "media", "ruido y outliers"),
         ("Gradient Boosting", "gradiente de una pérdida", "codificar", "media-baja", "ajuste sensible"),
         ("XGBoost", "regularización, escalable", "depende", "alta", "complejidad del tuning"),
         ("LightGBM", "histogramas y leaf-wise", "nativas", "muy alta", "sobreajuste en datos chicos")]
CX = [150, 470, 900, 1150, 1390]
v.b("La diapositiva dieciséis resume los modelos en una tabla: [[a]]su sello distintivo, cómo tratan las categorías, su velocidad y su riesgo principal. [[b]]AdaBoost pondera casos difíciles; Gradient Boosting usa el gradiente de una pérdida. [[c]]XGBoost, regularización; LightGBM, histogramas y crecimiento por hoja.",
    CL(), T("comparación de modelos · diapositiva 16", 960, 140, tam=48, c="tinta", ent="subir"),
    *[T(h, x, 250, tam=30, c="gris", w="sb", anc="lm", en="a") for h, x in zip(["modelo", "sello", "categorías", "velocidad", "riesgo"], CX)],
    *[[T(cel, x, 340 + 100 * i, tam=32 if j else 34, c=(COL[f[0]] if j == 0 else "tinta"), w=("b" if j == 0 else "sb"), anc="lm",
         en=("b" if i < 2 else "c"), t=0.3 * (i % 2)) for j, (cel, x) in enumerate(zip(f, CX))] for i, f in enumerate(FILAS)],
    sec="5. La tabla de la diapositiva 16")
v.b("Pero ojo, una alerta del material. [[a]]El título de la diapositiva dice comparación de los cinco modelos, y la tabla trae solo cuatro: falta CatBoost, que sí aparece en la diapositiva dieciocho. [[b]]Y la numeración impresa salta del trece al quince: el archivo tiene diecinueve láminas. En estos videos usamos la numeración del archivo.",
    alerta(760, en="a", texto="alerta del material"), T("título: «cinco modelos» · la tabla trae 4 (falta CatBoost)", 960, 860, tam=40, c="rojo", w="sb", en="a", t=0.4),
    T("numeración impresa: salta del 13/20 al 15/20 (19 láminas)", 960, 950, tam=36, c="gris", w="sb", en="b"))

# ---- 6. Cómo elegir (diapositiva 18)
v.b("¿Cómo elegir? La diapositiva dieciocho propone preguntarte qué domina tu problema. [[a]]Una línea base simple: AdaBoost. Control didáctico: Gradient Boosting. [[b]]Rendimiento general: XGBoost. Gran escala: LightGBM. [[c]]Y muchas variables categóricas: CatBoost.",
    CL(), T("¿qué domina tu problema? · diapositiva 18", 960, 140, tam=48, c="tinta", ent="subir"),
    *[[Bc(960, 260 + 115 * i, 1300, 95, "blanco", c, 4, 20, en=m, t=0.3 * k, ent="subir"),
       T(q, 360, 260 + 115 * i, tam=36, c="tinta", w="sb", anc="lm", en=m, t=0.3 * k + 0.1),
       T(r, 1560, 260 + 115 * i, tam=40, c=c, w="b", anc="rm", en=m, t=0.3 * k + 0.2)]
      for i, (q, r, c, m, k) in enumerate([("línea base simple", "AdaBoost", "rojo", "a", 0), ("control didáctico", "Gradient Boosting", "ambar", "a", 1),
                                           ("rendimiento general", "XGBoost", "violeta", "b", 0), ("gran escala", "LightGBM", "verde", "b", 1),
                                           ("muchas categorías", "CatBoost", "teal", "c", 0)])],
    sec="6. Cómo elegir")
v.b("Y después, validar la hipótesis: [[a]]misma partición, métrica adecuada, tiempo y memoria, y estabilidad.",
    P("validar: misma partición · métrica adecuada · tiempo y memoria · estabilidad", 960, 900, "verde", 38, en="a"))

# ---- 7. Alerta: la importancia de variables
v.b("Una alerta más del notebook, en la importancia de variables. [[a]]Grafica la importancia de XGBoost en Heart, y la de LightGBM en Ames, como si fueran lo mismo. [[b]]Pero en XGBoost, la importancia por defecto es la ganancia: cuánto mejora la pérdida cada variable. [[c]]Y en LightGBM, es el número de cortes: cuántas veces se usó la variable. Lo comprobamos en las dos bibliotecas.",
    CL(), alerta(170, texto="alerta del notebook"),
    T("feature_importances_ de XGBoost y de LightGBM", 960, 300, tam=44, c="tinta", w="m", en="a"),
    tarjetas([(560, "violeta", "XGBoost", "ganancia (gain): cuánto mejora la pérdida", "b"),
              (1360, "verde", "LightGBM", "cortes (split): cuántas veces se usa", "c")], y=560, w=680, h=320, tam_t=56),
    sec="7. Alerta: la importancia")
v.b("No son comparables. [[a]]Y ninguna demuestra causalidad. Para comparar en serio, usa la importancia por permutación sobre el test.",
    P("no son comparables · ninguna es causal", 960, 840, "rojo", 46),
    T("mejor: permutation_importance sobre el test", 960, 940, tam=40, c="verde", w="sb", en="a"),
    real(1050, texto="comprobado con XGBoost 3.4 y LightGBM 4.7"))

# ---- 8. Aplicaciones (diapositiva 17)
v.b("¿Dónde se usa boosting? La diapositiva diecisiete da seis áreas: [[a]]riesgo, como impago, fraude o churn. Demanda: ventas, energía, inventario. [[b]]Salud, marketing, operaciones y precios. [[c]]Boosting destaca con datos tabulares, relaciones no lineales e interacciones.",
    CL(), T("aplicaciones · diapositiva 17", 960, 150, tam=50, c="tinta", ent="subir"),
    *[P(t_, 620 + 680 * (i % 2), 290 + 130 * (i // 2), c, 40, en=m, t=0.2 * (i % 2)) for i, (t_, c, m) in enumerate(
        [("riesgo: impago, fraude, churn", "rojo", "a"), ("demanda: ventas, energía", "ambar", "a"), ("salud: riesgo clínico", "teal", "b"),
         ("marketing: propensión", "violeta", "b"), ("operaciones: fallas", "azul", "b"), ("precios: valoración", "verde", "b")])],
    P("datos tabulares · no linealidades · interacciones", 960, 740, "gris", 44, en="c"), sec="8. Aplicaciones")

# ---- 9. Error típico
v.b("El error típico de toda comparación: [[a]]declarar un ganador con una sola partición, o comparando modelos con preprocesamientos distintos. [[b]]Compara siempre bajo el mismo pipeline, la misma partición y la misma métrica. [[c]]Y reporta la media con su desvío.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("un ganador con una sola partición", 410, 260, tam=48, c="rojo", anc="lm", en="a"),
    T("mismo pipeline · misma partición · misma métrica", 960, 420, tam=46, c="tinta", w="sb", en="b"),
    P("reporta media ± desvío", 960, 580, "verde", 54, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]En Ames, XGBoost sacó " + cien(CA["XGBoost"][0]) + " dólares de RMSE en test, y Gradient Boosting, " + cien(CA["Gradient Boosting"][0]) +
    ". [[b]]¿Puedes afirmar que XGBoost es mejor, con la misma seguridad que en Heart?",
    CL(), P(f"Ames: XGBoost {k1(CA['XGBoost'][0])} · Gradient Boosting {k1(CA['Gradient Boosting'][0])}", 960, 300, "violeta", 48, en="a"),
    T("¿XGBoost es mejor?", 960, 470, tam=72, c="tinta", en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Con más cautela de la que parece. [[a]]Es una sola partición de test, y no tenemos su desvío. [[b]]La diferencia es de unos 2 400 dólares, cerca de un nueve por ciento, y va en la misma dirección que la curva de validación: es una señal razonable. [[c]]Pero para afirmarlo en serio, repetirías con validación cruzada o con varias semillas, como hicimos con Heart.",
    CL(), T("una sola partición · sin desvío", 960, 280, tam=52, c="tinta", w="sb", en="a"),
    P(f"diferencia ≈ {cien(CA['Gradient Boosting'][0] - CA['XGBoost'][0]).replace(' ', '.')} USD (~9 %): señal razonable", 960, 440, "ambar", 44, en="b"),
    P("para afirmarlo: CV o varias semillas", 960, 600, "verde", 48, en="c"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]en Heart, con pocos datos, los cinco modelos empatan dentro del desvío. [[b]]En Ames, con más datos, XGBoost y los de gradiente se separan, y AdaBoost queda atrás. [[c]]Elige según lo que domina tu problema, y valida bajo la misma partición y la misma métrica.",
    CL(), P("Heart (303): empate dentro del desvío", 960, 300, "azul", 48, en="a"),
    P("Ames (2.930): XGBoost adelante, AdaBoost atrás", 960, 460, "violeta", 48, en="b"), P("elige por lo que domina · valida igual para todos", 960, 620, "verde", 46, en="c"),
    sec="11. Para llevar")
v.b("Última parada: [[a]]el repaso final de la clase, con un checklist para entrenar cualquier modelo de boosting.",
    CL(), siguiente(11, "Repaso final y checklist"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
