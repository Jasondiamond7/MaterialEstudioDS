import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, alerta, k1, flujo

OUT = out("guion_c35_08_hiperparametros.json")
v = V("Número de árboles e hiperparámetros")
CU = FG["curvas"]
GR = D["nb_heart_grid_rf"]
NH = D["nb_heart"]

v.portada("Árboles e hiperparámetros", "Cuántos árboles, qué ajustar primero, y cómo no hacer trampa",
          "Octava parada: los hiperparámetros. Cuántos árboles usar, qué ajustar primero, y dos errores del notebook de la clase en la búsqueda.")

# ---- 1. Hook
v.b("¿Cien árboles, quinientos, cinco mil? [[a]]Y una duda que casi todos tienen: [[b]]si pongo demasiados árboles, ¿el bosque se sobreajusta?",
    CL(), T("¿100, 500 o 5.000 árboles?", 960, 300, tam=66, c="tinta", en="a"), P("¿demasiados árboles = sobreajuste?", 960, 500, "rojo", 56, en="b"),
    sec="1. ¿Cuántos árboles?")

# ---- 2. Analogía
v.b("Piensa en un promedio de encuestas. [[a]]Con tres encuestas, una más cambia bastante el promedio. [[b]]Con cien, una más ya casi no lo mueve: solo cuesta dinero. [[c]]Y agregar encuestas nunca empeora el promedio. Con los árboles pasa igual.",
    CL(), T("un promedio de encuestas", 960, 180, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "azul", "3 encuestas", "una más cambia mucho el promedio", "a"), (960, "teal", "100 encuestas", "una más casi no lo mueve: solo cuesta", "b"),
              (1520, "verde", "y nunca empeora", "agregar no daña el promedio", "c")], y=490, w=480, h=330, tam_t=46, tam_d=34),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: las encuestas")

# ---- 3. La curva en Ames
BS = [1, 10, 50, 100, 300]
v.b("Lo medimos en Ames, con un conjunto de validación. [[a]]Un solo árbol sin podar tiene un error de 36 100 dólares. [[b]]Un Random Forest con un árbol, 37 300: igual de malo. Con diez árboles, 24 400. [[c]]Con cincuenta, 23 300. Y con cien y con trescientos, 23 100: ya no se mueve.",
    CL(), T("Ames · RMSE de validación según el número de árboles", 960, 160, tam=44, c="tinta", ent="subir"),
    G(300, 260, 1380, 800, -0.3, 4.3, 20, 40, [dict(xs=list(range(5)), ys=[CU["rf"][str(b)] / 1000 for b in BS], color="teal", grosor=7, marcadores=11, revela=0.0),
                                               dict(xs=[-0.3, 4.3], ys=[CU["tree"] / 1000] * 2, color="gris", grosor=4, revela=0.0, lapiz=False)],
      xticks=[], yticks=[20, 25, 30, 35, 40], yfmt="{:.0f} mil", id="gc"),
    *[T(str(b), i, 18.6, tam=30, c="gris", w="sb", ref="gc") for i, b in enumerate(BS)], T("número de árboles B", 840, 880, tam=32, c="tinta", w="sb"),
    U("gc", **{"series.1.revela": 1.0}, en="a", dur=0.8), T(f"un árbol sin podar: {k1(CU['tree'])}", 1660, 330, tam=36, c="gris", w="b", en="a", t=0.3),
    U("gc", **{"series.0.revela": 0.01}, en="b", dur=0.1), U("gc", **{"series.0.revela": 0.25}, en="b", t=2.2, dur=0.6),
    T(f"B = 1: {k1(CU['rf']['1'])}", 1660, 420, tam=38, c="rojo", w="b", en="b", t=0.3), T(f"B = 10: {k1(CU['rf']['10'])}", 1660, 500, tam=38, c="tinta", w="b", en="b", t=2.4),
    U("gc", **{"series.0.revela": 1.0}, en="c", dur=1.6), T(f"B = 50: {k1(CU['rf']['50'])}", 1660, 580, tam=38, c="tinta", w="b", en="c", t=0.5),
    T(f"B = 100 y 300: {k1(CU['rf']['300'])}", 1660, 660, tam=36, c="verde", w="b", en="c", t=1.4), real(1050), sec="3. La curva en Ames")
v.b("Dos lecciones. [[a]]Con unos treinta y cuatro árboles, el bosque ya está a menos de un 2 % de su error final: la mejora se aplana pronto. [[b]]Y la curva nunca vuelve a subir: más árboles no sobreajustan, solo cuestan tiempo y memoria.",
    P(f"B = {CU['b95_rf']}: a menos de 2 % del final", 1620, 780, "teal", 30, en="a"),
    P("más árboles NO sobreajustan: solo cuestan", 840, 975, "verde", 40, en="b"))
v.b("¿Por qué no sobreajustan? [[a]]Vuelve a la fórmula: agregar árboles solo achica el término uno menos ro, sigma al cuadrado, sobre B. [[b]]No cambia cómo es cada árbol. El sobreajuste se controla con la complejidad de cada árbol, no con cuántos hay.",
    CL(), caja_formula("Var = ρσ² + (1 − ρ)σ² / B", y=300, c="azul", tam=60, w=1100, en="a"),
    T("más árboles: solo achican el 2.º término", 960, 460, tam=46, c="verde", w="sb", en="a", t=0.5),
    P("el sobreajuste se controla con la complejidad de CADA árbol", 960, 620, "violeta", 46, en="b"))

# ---- 4. Los hiperparámetros (diapositiva 14)
v.b("Los hiperparámetros principales, según la diapositiva catorce. [[a]]n_estimators, el número de árboles: más árboles estabilizan y aumentan el costo. [[b]]max_features, las variables candidatas por nodo: menos variables reducen la correlación y pueden aumentar el sesgo. [[c]]Y max_depth y min_samples_leaf, la complejidad de cada árbol: controlan hojas pequeñas y la suavidad de la predicción.",
    CL(), T("hiperparámetros principales · diapositiva 14", 960, 150, tam=46, c="tinta", ent="subir"),
    *[[Bc(960, 290 + 150 * i, 1500, 120, "blanco", c, 4, 22, en=m, ent="subir"), T(nm, 300, 290 + 150 * i, tam=40, c=c, w="m", anc="lm", en=m, t=0.1),
       T(desc, 1640, 290 + 150 * i, tam=34, c="tinta", w="sb", anc="rm", en=m, t=0.2)]
      for i, (nm, desc, c, m) in enumerate([("n_estimators", "número de árboles · estabiliza y cuesta", "azul", "a"),
                                            ("max_features", "candidatas por nodo · menos ρ, quizá más sesgo", "violeta", "b"),
                                            ("max_depth / min_samples_leaf", "complejidad de cada árbol · hojas y suavidad", "teal", "c")])],
    sec="4. Los hiperparámetros")
v.b("Dos más: [[a]]bootstrap y max_samples, que definen si hay reemplazo y cuántas filas usa cada árbol. [[b]]Y class_weight, el peso de las clases, que ayuda cuando la variable objetivo está desbalanceada. [[c]]Y la prioridad práctica de la diapositiva: primero valida n_estimators, max_features y el tamaño mínimo de hoja.",
    *[[Bc(960, 740 + 120 * i, 1500, 100, "blanco", c, 4, 22, en=m, ent="subir"), T(nm, 300, 740 + 120 * i, tam=38, c=c, w="m", anc="lm", en=m, t=0.1),
       T(desc, 1640, 740 + 120 * i, tam=32, c="tinta", w="sb", anc="rm", en=m, t=0.2)]
      for i, (nm, desc, c, m) in enumerate([("bootstrap / max_samples", "reemplazo · cuántas filas por árbol", "gris", "a"),
                                            ("class_weight", "peso de las clases · desbalance", "ambar", "b")])],
    P("prioridad: n_estimators · max_features · tamaño mínimo de hoja", 960, 1000, "verde", 38, en="c"))

# ---- 5. Ejemplo: el costo de una grilla
v.b("Para ajustarlos, el notebook usa GridSearchCV. Calculemos cuánto trabajo es. [[a]]La grilla del Random Forest prueba tres valores de n_estimators, dos de max_features, dos de max_depth y dos de min_samples_leaf. [[b]]Tres por dos por dos por dos: 24 combinaciones. [[c]]Con validación cruzada de cinco bloques, 24 por 5: 120 bosques entrenados, más uno final con la mejor combinación.",
    CL(), T("¿cuánto trabajo es una grilla? (notebook, Heart)", 960, 150, tam=46, c="tinta", ent="subir"),
    codigo(["n_estimators:      [100, 300, 600]     → 3", "max_features:      ['sqrt', 0.7]       → 2", "max_depth:         [None, 6]           → 2",
            "min_samples_leaf:  [1, 3]              → 2"], y=380, w=1300, tam=38, en="a", dt=0.3),
    T("3 × 2 × 2 × 2 =", 900, 620, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(24, 930, 620, "{:.0f}", 72, "violeta", suf=" combinaciones", anc="lm", en="b", t=0.4),
    T("24 × 5 bloques =", 900, 750, tam=56, c="tinta", w="sb", anc="rm", en="c"), N(120, 930, 750, "{:.0f}", 72, "azul", suf=" ajustes (+1 final)", anc="lm", en="c", t=0.4),
    sec="5. El costo de una grilla")
v.b("Y el resultado, replicando el notebook. [[a]]La configuración base, con cien árboles y sin límites, tenía un AUC de validación cruzada de 0.897. [[b]]La mejor de la grilla, seiscientos árboles con hojas de al menos tres pacientes, llega a 0.908. [[c]]Una ganancia de una centésima, menor que el desvío entre particiones. La búsqueda pule; no transforma un modelo.",
    CL(), T("Heart · GridSearchCV del Random Forest · AUC de validación cruzada", 960, 160, tam=42, c="tinta", ent="subir"),
    T("base (100 árboles, sin límites)", 860, 320, tam=40, c="gris", w="sb", anc="rm", en="a"), N(GR["base_cv"], 900, 320, "{:.3f}", 64, "gris", anc="lm", en="a", t=0.3),
    T("mejor (600 árboles, hoja ≥ 3)", 860, 450, tam=40, c="verde", w="sb", anc="rm", en="b"), N(GR["best"], 900, 450, "{:.3f}", 64, "verde", anc="lm", en="b", t=0.3),
    P(f"+{GR['best'] - GR['base_cv']:.3f}: la búsqueda pule, no transforma", 960, 640, "violeta", 48, en="c"), real(1050, texto="celdas del notebook replicadas"))

# ---- 6. Dos errores del notebook en la búsqueda
v.b("Ahora, dos detalles del notebook que conviene corregir. [[a]]El primero: el Random Forest base lleva class_weight igual a balanced, y el bagging no. [[b]]Así, la comparación no es pareja: un modelo tiene un ajuste que el otro no. [[c]]Y Heart está casi equilibrado, 46 % de positivos, así que el peso ni siquiera era necesario.",
    CL(), alerta(160, texto="detalle del notebook de la clase"),
    codigo(["BaggingClassifier(estimator=DecisionTreeClassifier(), ...)", "RandomForestClassifier(..., class_weight=\"balanced\", ...)   # ← solo aquí"],
           y=340, w=1600, tam=34, c="ambar", en="a", dt=0.4),
    P("comparación no pareja: un ajuste que el otro no tiene", 960, 520, "rojo", 44, en="b"),
    T(f"Heart: {D['nb_heart_pos_tr'] * 100:.0f} % de positivos → casi equilibrado", 960, 650, tam=42, c="tinta", w="sb", en="c"), sec="6. Dos detalles del notebook")
v.b("El segundo es más sutil. [[a]]En la búsqueda aleatoria, al bagging se le deja sortear max_features del árbol base: raíz cuadrada, o 0.7. [[b]]Pero un árbol de decisión con max_features sortea variables en cada nodo. [[c]]Si sale ese valor, ese bagging optimizado es, en realidad, un Random Forest disfrazado.",
    CL(), codigo(["bag_clf_random = {", "    ...", "    \"model__estimator__max_features\": [None, \"sqrt\", 0.7],   # ← sorteo por nodo", "}"],
                 y=330, w=1600, tam=34, c="ambar", en="a", dt=0.25),
    T("DecisionTreeClassifier(max_features=…) sortea variables en CADA nodo", 960, 560, tam=40, c="tinta", w="sb", en="b"),
    P("«Bagging optimizado» = Random Forest disfrazado", 960, 720, "rojo", 50, en="c"))

# ---- 7. Error típico (diapositiva 17)
v.b("El error típico, y la diapositiva diecisiete lo lista: [[a]]ajustar hiperparámetros mirando repetidamente el conjunto de prueba. [[b]]Cada vez que eliges mirando el test, el test se vuelve parte del entrenamiento, y deja de medir la generalización. [[c]]Se ajusta con validación cruzada u OOB, y el test se mira una sola vez, al final.",
    CL(), T("ajustar mirando el test una y otra vez", 960, 280, tam=56, c="tinta", en="a", id="e1"),
    T("el test pasa a ser parte del entrenamiento", 960, 410, tam=44, c="gris", w="sb", en="b"), U("e1", tachado=1.0, color="gris", en="c", dur=0.6),
    P("ajustar con CV u OOB · el test, UNA vez, al final", 960, 590, "verde", 50, en="c", t=0.4), T("diapositiva 17", 960, 720, tam=32, c="gris", w="sb", en="a", t=0.6),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tu bosque de quinientos árboles tiene exactitud 1.000 en entrenamiento: [[b]]acierta a todos los pacientes que vio. Y en el test, 0.855. [[c]]¿Está sobreajustado? ¿Hay que podarlo?",
    CL(), P(f"exactitud en entrenamiento: {D['prop_rf_train_acc']:.3f}", 960, 270, "rojo", 56, en="a"),
    P(f"en el test: {D['prop_rf_test_acc']:.3f}", 960, 410, "teal", 56, en="b"), T("¿sobreajustado? ¿hay que podarlo?", 960, 590, tam=64, c="tinta", en="c"),
    real(1050), sec="8. Piensa un segundo", pausa=0.6)
v.b("No necesariamente. [[a]]Un bosque con hojas de un solo paciente siempre memoriza su entrenamiento: esa exactitud de 1.000 no dice nada. [[b]]Lo que importa es el error en datos no vistos: el OOB da 0.811, y el test, 0.855. [[c]]Si ese error es el menor entre tus alternativas, la brecha con el entrenamiento no es un problema.",
    CL(), T("hojas de 1 paciente → siempre memoriza el entrenamiento", 960, 260, tam=48, c="tinta", ent="subir"),
    P(f"importa lo no visto: OOB {D['prop_rf_oob_acc']:.3f} · test {D['prop_rf_test_acc']:.3f}", 960, 420, "teal", 50, en="b"),
    P("si es el menor error de tus alternativas, la brecha no es problema", 960, 590, "verde", 44, en="c"), real(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]más árboles estabilizan y nunca sobreajustan; en Ames, con unos treinta y cuatro, ya casi está. [[b]]Primero se validan n_estimators, max_features y el tamaño mínimo de hoja, con validación cruzada u OOB. [[c]]La búsqueda pule unas centésimas, y el test se mira una sola vez.",
    CL(), P(f"más árboles: estabilizan, no sobreajustan (≈ {CU['b95_rf']} ya casi basta)", 960, 300, "teal", 46, en="a"),
    P("validar: n_estimators · max_features · hoja mínima", 960, 470, "violeta", 48, en="b"), P("la búsqueda pule · el test, una vez", 960, 640, "verde", 50, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]cómo explicar lo que aprendió un bosque que no se puede dibujar, y dónde falla.",
    CL(), siguiente(9, "Importancia y límites"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
