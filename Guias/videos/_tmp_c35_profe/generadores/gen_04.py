import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, flujo, caja_formula, codigo, alerta, k1

OUT = out("guion_c35_04_bagging.json")
v = V("Bagging: votar y promediar")
CH, CA = FG["cmp_h"], FG["cmp_a"]
PR = [0.80, 0.30, 0.70, 0.45, 0.20]
PRECIOS = [180, 210, 195, 240, 175]

v.portada("Bagging: votar y promediar", "Bootstrap aggregating: el nombre ya es la receta",
          "Cuarta parada: el bagging. Vas a ver el algoritmo en cuatro pasos, y vas a agregar votos y probabilidades a mano.")

# ---- 1. Hook
v.b("El nombre bagging viene de dos palabras en inglés: [[a]]bootstrap, que ya conoces, [[b]]y aggregating, que significa agregar, juntar. [[c]]El nombre ya es la receta: muestras bootstrap, y luego juntar las respuestas.",
    CL(), T("bagging =", 960, 220, tam=64, c="tinta", ent="subir"),
    P("bootstrap", 640, 400, "azul", 64, en="a"), T("+", 960, 400, tam=64, c="gris", en="b"), P("aggregating", 1290, 400, "violeta", 64, en="b"),
    T("muestras bootstrap  →  juntar las respuestas", 960, 600, tam=46, c="tinta", w="sb", en="c"), sec="1. El nombre es la receta")

# ---- 2. Analogía
v.b("Una analogía clásica: adivinar cuántos caramelos hay en un frasco. [[a]]Cada persona se equivoca bastante: unos dicen muy poco, otros demasiado. [[b]]Pero el promedio de muchas personas suele quedar sorprendentemente cerca, [[c]]porque los errores para arriba y para abajo se cancelan.",
    CL(), T("¿cuántos caramelos hay en el frasco?", 960, 170, tam=50, c="tinta", ent="subir"),
    Bc(400, 560, 300, 420, "blanco", "gris", 5, 40, sombra=True), Pt([(40, [290, 420, 510, 740], "ambar")], 12),
    *[T(str(g), 760 + 150 * i, 400, tam=46, c="gris", en="a", t=0.12 * i) for i, g in enumerate([310, 520, 380, 700, 450, 260])],
    T("cada uno se equivoca bastante", 1130, 490, tam=36, c="gris", w="sb", en="a", t=0.8),
    T("promedio:", 1000, 640, tam=46, c="verde", anc="rm", en="b"), N(436.7, 1020, 640, "{:.0f}", 70, "verde", anc="lm", en="b", t=0.3),
    P("los errores hacia arriba y hacia abajo se cancelan", 1130, 790, "violeta", 40, en="c"), ilustrativo(1000, texto="analogía · números ilustrativos"),
    sec="2. Analogía: el frasco de caramelos")

# ---- 3. El algoritmo
v.b("El algoritmo de bagging tiene cuatro pasos, como en la diapositiva ocho. [[a]]Uno, muestrear: crear B muestras bootstrap. [[b]]Dos, entrenar: ajustar un árbol en cada muestra. [[c]]Tres, predecir: para un caso nuevo x, obtener B respuestas. [[d]]Y cuatro, agregar: votar o promediar.",
    CL(), T("el algoritmo de bagging · diapositiva 8", 960, 170, tam=48, c="tinta", ent="subir"),
    flujo([("1. muestrear\nB bootstrap", "azul", "a"), ("2. entrenar\nun árbol c/u", "violeta", "b"), ("3. predecir\nB respuestas", "teal", "c"),
           ("4. agregar\nvotar o promediar", "verde", "d")], y=470, w=360, h=230, x0=330, paso=420, tam=38), sec="3. El algoritmo")
v.b("Dos detalles importantes. [[a]]Cada árbol suele crecer profundo, sin poda: queremos árboles de bajo sesgo, y el promedio se encarga de la varianza. [[b]]Y los árboles no dependen unos de otros: se pueden entrenar al mismo tiempo, en paralelo, repartidos entre los núcleos del computador.",
    P("árboles profundos, sin poda: bajo sesgo", 960, 730, "violeta", 44, en="a"),
    P("independientes → se entrenan en paralelo (n_jobs)", 960, 850, "teal", 44, en="b"))

# ---- 4. Votar: diapositiva 13
v.b("Primero, clasificación por votos. El ejemplo de la diapositiva trece: [[a]]cinco árboles estiman si un cliente responderá a una campaña. [[b]]Dicen: Sí, No, Sí, Sí, No.",
    CL(), T("diapositiva 13 · ¿responderá a la campaña?", 960, 160, tam=46, c="tinta", ent="subir"),
    *[[Bc(360 + 300 * i, 380, 240, 220, "blanco", "violeta", 4, 24, sombra=True, en="a", t=0.1 * i, ent="pop"), T(f"árbol {i + 1}", 360 + 300 * i, 320, tam=34, c="violeta", en="a", t=0.1 * i + 0.05),
       T(vv, 360 + 300 * i, 410, tam=58, c=("verde" if vv == "Sí" else "rojo"), en="b", t=0.15 * i, id=f"vt{i}")] for i, vv in enumerate(["Sí", "No", "Sí", "Sí", "No"])],
    ilustrativo(1000, texto="ejemplo de la diapositiva 13"), sec="4. Votar")
v.b("Contamos: [[a]]tres votos Sí, dos votos No. [[b]]La mayoría dice Sí, y la probabilidad estimada por votos es tres quintos: 0.60.",
    T("3 votos Sí · 2 votos No", 960, 620, tam=56, c="tinta", en="a"), P("predicción: Sí · P = 3/5 = 0.60", 960, 760, "verde", 54, en="b"))
v.b("La diapositiva deja una pregunta: [[a]]¿cambiaría la decisión si el umbral fuera 0.70? [[b]]Sí: 0.60 no alcanza 0.70, así que la predicción pasaría a No. [[c]]El umbral es una decisión de negocio, no del algoritmo.",
    CL(), P("¿y con umbral 0.70?", 960, 280, "ambar", 56, en="a"),
    T("0.60 < 0.70  →  predicción: No", 960, 450, tam=60, c="rojo", en="b"), T("el umbral es una decisión de negocio", 960, 600, tam=44, c="gris", w="sb", en="c"),
    ilustrativo(1000, texto="ejemplo de la diapositiva 13"))

# ---- 5. Votar no es lo mismo que promediar probabilidades
v.b("Pero hay una sutileza, y la nota de la misma diapositiva la menciona: [[a]]muchas implementaciones no cuentan votos, sino que promedian probabilidades. [[b]]Veamos por qué importa. Cinco árboles dan estas probabilidades de Sí: 0.80, 0.30, 0.70, 0.45 y 0.20.",
    CL(), alerta(170, texto="aclaración del material"),
    T("nota de la diapositiva 13: «muchas implementaciones promedian probabilidades»", 960, 280, tam=38, c="tinta", w="sb", en="a"),
    *[[Bc(360 + 300 * i, 470, 240, 150, "blanco", "azul", 4, 24, en="b", t=0.12 * i, ent="pop"), T(f"{p:.2f}", 360 + 300 * i, 470, tam=56, c="azul", en="b", t=0.12 * i + 0.05)]
      for i, p in enumerate(PR)],
    ilustrativo(1000, texto="ejemplo del apunte"), sec="5. Votos contra probabilidades")
v.b("Promedio de probabilidades: [[a]]la suma da 2.45, y dividida por cinco, 0.49. [[b]]Votos, con umbral 0.5: Sí, No, Sí, No, No. Dos votos de cinco: 0.40.",
    T("promedio:  (0.80 + 0.30 + 0.70 + 0.45 + 0.20) / 5 = 2.45 / 5 =", 1260, 640, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(sum(PR) / 5, 1290, 640, "{:.2f}", 60, "azul", anc="lm", en="a", t=0.8),
    T("votos (P > 0.5):  Sí · No · Sí · No · No  →  2/5 =", 1260, 760, tam=40, c="tinta", w="sb", anc="rm", en="b"),
    N(0.40, 1290, 760, "{:.2f}", 60, "violeta", anc="lm", en="b", t=0.8))
v.b("Con umbral 0.5, los dos métodos dicen No. [[a]]Pero con un umbral de 0.45, el promedio, 0.49, dice Sí, y los votos, 0.40, dicen No. [[b]]No son lo mismo. Y scikit-learn, en el Random Forest, promedia probabilidades: el conteo de votos de la diapositiva es una simplificación.",
    CL(), T("umbral 0.5:   promedio 0.49 → No   ·   votos 0.40 → No", 960, 300, tam=46, c="tinta", w="sb", ent="subir"),
    T("umbral 0.45:   promedio 0.49 → Sí   ·   votos 0.40 → No", 960, 420, tam=46, c="rojo", w="b", en="a"),
    P("scikit-learn promedia probabilidades (no cuenta votos)", 960, 600, "violeta", 46, en="b"), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 6. Regresión
v.b("En regresión es aún más simple: se promedian los números. [[a]]Cinco árboles predicen el precio de una casa, en miles de dólares: 180, 210, 195, 240 y 175. [[b]]Suman mil, y el promedio es 200 mil dólares.",
    CL(), T("regresión: promediar las predicciones", 960, 170, tam=50, c="tinta", ent="subir"),
    *[[Bc(360 + 300 * i, 400, 240, 150, "blanco", "teal", 4, 24, en="a", t=0.12 * i, ent="pop"), T(str(p), 360 + 300 * i, 400, tam=58, c="teal", en="a", t=0.12 * i + 0.05)]
      for i, p in enumerate(PRECIOS)],
    T("miles de dólares", 960, 510, tam=32, c="gris", w="sb", en="a", t=0.7),
    T("(180 + 210 + 195 + 240 + 175) / 5 = 1000 / 5 =", 1180, 650, tam=44, c="tinta", w="sb", anc="rm", en="b"),
    N(sum(PRECIOS) / 5, 1210, 650, "{:.0f}", 72, "verde", anc="lm", en="b", t=0.6), ilustrativo(1000), sec="6. Regresión: promediar")

# ---- 7. Datos reales
v.b("¿Cuánto ayuda en la práctica? [[a]]En Heart, con validación cruzada repetida, un árbol solo tiene un AUC de 0.753. [[b]]Un bagging de árboles sube a 0.893. [[c]]En Ames, el error de prueba de un árbol es 45 800 dólares, y el de un bagging de 300 árboles, 28 900.",
    CL(), T("un árbol contra un bagging de árboles", 960, 160, tam=48, c="tinta", ent="subir"),
    T("Heart · AUC (CV 5×3)", 560, 290, tam=40, c="teal", en="a"),
    T("árbol", 380, 400, tam=40, c="gris", w="sb", en="a"), N(CH["Árbol"][0], 740, 400, "{:.3f}", 60, "gris", en="a", t=0.3),
    T("bagging", 380, 520, tam=40, c="azul", w="sb", en="b"), N(CH["Bagging de árboles"][0], 740, 520, "{:.3f}", 60, "azul", en="b", t=0.3),
    T("Ames · RMSE de prueba", 1360, 290, tam=40, c="teal", en="c"),
    T("árbol", 1180, 400, tam=40, c="gris", w="sb", en="c"), N(CA["Árbol"][0] / 1000, 1540, 400, "{:.1f}", 60, "gris", suf=" mil", en="c", t=0.3),
    T("bagging", 1180, 520, tam=40, c="azul", w="sb", en="c", t=0.5), N(CA["Bagging"][0] / 1000, 1540, 520, "{:.1f}", 60, "azul", suf=" mil", en="c", t=0.8),
    real(1050), sec="7. Con datos reales")
v.b("En los dos problemas, el bagging corrige mucho al árbol suelto: [[a]]0.14 más de AUC, y casi 17 000 dólares menos de error.",
    P(f"+{CH['Bagging de árboles'][0] - CH['Árbol'][0]:.2f} de AUC  ·  −{(CA['Árbol'][0] - CA['Bagging'][0]) / 1000:.1f} mil dólares de error", 960, 720, "verde", 48, en="a"))

# ---- 8. En Python
v.b("En scikit-learn, así lo arma el notebook del profesor: [[a]]un BaggingClassifier, con un árbol de decisión como modelo base, [[b]]120 estimadores, bootstrap igual a True, y oob_score igual a True para tener el error fuera de bolsa.",
    CL(), T("bagging en Python (notebook del profesor)", 960, 170, tam=48, c="tinta", ent="subir"),
    codigo(["BaggingClassifier(", "    estimator=DecisionTreeClassifier(random_state=42),", "    n_estimators=120,", "    bootstrap=True,",
            "    oob_score=True,", "    n_jobs=6)"], y=500, w=1300, tam=38, en="a", dt=0.3), sec="8. En Python")

# ---- 9. Error típico
v.b("El error típico: [[a]]hacer bagging de un modelo que ya es estable, esperando una mejora. [[b]]En Heart, una regresión logística tiene un AUC de 0.913, y un bagging de cien logísticas, 0.912. [[c]]Nada. El bagging reduce varianza, y la logística casi no tiene: ayuda a modelos inestables, como los árboles profundos.",
    CL(), T("bagging de un modelo estable", 960, 220, tam=56, c="tinta", en="a"),
    T("regresión logística", 700, 400, tam=42, c="ambar", w="sb", en="b"), N(CH["Reg. logística"][0], 1200, 400, "{:.3f}", 60, "ambar", en="b", t=0.3),
    T("bagging de 100 logísticas", 700, 510, tam=42, c="ambar", w="sb", en="b", t=0.5), N(CH["Bagging de logísticas"][0], 1200, 510, "{:.3f}", 60, "ambar", en="b", t=0.8),
    P("ayuda a modelos INESTABLES (árboles profundos)", 960, 700, "verde", 48, en="c"), real(1050), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]¿Por qué en bagging se dejan crecer los árboles profundos, sin podar, si un árbol profundo sobreajusta?",
    CL(), T("¿por qué árboles profundos, sin podar?", 960, 380, tam=64, c="tinta", en="a"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Porque el sobreajuste de un árbol profundo es varianza, y la varianza es justo lo que el promedio elimina. [[a]]Lo que el promedio NO arregla es el sesgo: si cada árbol fuera simple y sesgado, el promedio seguiría sesgado. [[b]]Por eso se buscan árboles de bajo sesgo, y se deja que el promedio haga el resto.",
    CL(), P("sobreajuste del árbol profundo = varianza → el promedio la elimina", 960, 300, "verde", 44, ent="subir"),
    P("el promedio NO arregla el sesgo", 960, 450, "rojo", 48, en="a"), T("árboles de bajo sesgo + el promedio hace el resto", 960, 600, tam=46, c="tinta", w="sb", en="b"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]bagging es entrenar un árbol en cada muestra bootstrap y agregar: votos o probabilidades en clasificación, promedio en regresión. [[b]]Votar y promediar probabilidades no siempre coinciden; scikit-learn promedia. [[c]]Y ayuda a modelos inestables: un árbol en Heart pasa de 0.753 a 0.893.",
    CL(), P("bagging = árbol por muestra bootstrap + agregar", 960, 300, "violeta", 48, en="a"),
    P("votos ≠ promedio de probabilidades (scikit-learn promedia)", 960, 470, "azul", 44, en="b"),
    P(f"ayuda a lo inestable: {CH['Árbol'][0]:.3f} → {CH['Bagging de árboles'][0]:.3f}", 960, 640, "verde", 50, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la matemática de por qué promediar funciona, y cuál es su límite.",
    CL(), siguiente(5, "Por qué promediar reduce la varianza"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
