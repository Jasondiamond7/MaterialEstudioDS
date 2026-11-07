import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, caja_formula, alerta

OUT = out("guion_c33_09_trampa.json")
v = V("La trampa: dónde vive la selección")
RU, RF = D["ruido"], D["real_fuera_dentro"]
NB_FOLDS = [0.8948, 0.9138, 0.8897, 0.9229, 0.8866]      # salida impresa del notebook de la clase (celda 48)
NB_MEDIA = 0.9015

v.portada("La trampa", "Dónde vive la selección: todo lo que mira el target va dentro de cada fold",
          "Octava parada: la trampa. Todo lo que hicimos hoy tiene un problema, y lo vamos a medir.")

# ---- 1. Hook: la prueba hecha con las respuestas
v.b("Imagina un profesor que arma la prueba final eligiendo justo las preguntas que sus alumnos ya respondieron bien en clase. [[a]]El curso saca un siete. [[b]]¿Aprendieron, [[c]]o el profesor eligió las preguntas mirando las respuestas?",
    CL(), P("una prueba armada con las respuestas", 960, 220, "violeta", 52, ent="subir"),
    Tj(960, 560, 620, 460, "Prueba final", ["pregunta 3  (ya la sabían)", "pregunta 7  (ya la sabían)", "pregunta 9  (ya la sabían)"], c="violeta", id="tj",
       marcas=[{"forma": "check", "progreso": 0}] * 3),
    U("tj", **{"marcas": [{"forma": "check", "progreso": 1}] * 3}, en="a", dur=0.8), T("nota 7.0", 1450, 560, tam=80, c="verde", en="a", t=0.6),
    T("¿aprendieron?", 360, 480, tam=50, c="tinta", en="b"), T("¿o se eligió mirando?", 360, 600, tam=50, c="rojo", en="c"),
    ilustrativo(1000, texto="analogía"), sec="1. Una prueba armada con las respuestas")
v.b("Eso mismo hicimos toda la clase. [[a]]La clase treinta y dos dejó escrita esta regla: el escalador se ajusta solo con entrenamiento; ajustarlo con todos los datos es data leakage. [[b]]Si eso vale para el escalador, ¿por qué no valdría para la selección de variables?",
    CL(), Bc(960, 380, 1500, 260, "azulclaro", None, 0, 36, ent="pop"),
    T("«El escalador se ajusta (fit) únicamente con entrenamiento y luego se aplica (transform) a test. Ajustarlo con todos los datos produciría Data Leakage.»",
      960, 380, tam=40, c="azul", w="sb", mw=1350, en="a"), T("regla de la clase 32", 960, 560, tam=32, c="gris", w="sb", en="a", t=0.6),
    T("¿y la selección de variables?", 960, 720, tam=70, c="tinta", en="b"))
v.b("Toda esta clase seleccionamos mirando las tres mil ciento cincuenta filas: [[a]]el filtro de información mutua, el desempate de los pares redundantes, el RFE, y el Lasso. [[b]]Y después medimos el desempeño sobre esas mismas filas. [[c]]Todos esos AUC están inflados. La pregunta es cuánto.",
    CL(), *[P(tx, x, 330, "violeta", 40, en="a", t=0.4 * j) for j, (tx, x) in enumerate((("filtro MI", 400), ("desempate", 760), ("RFE", 1100), ("Lasso", 1420)))],
    T("mirando las 3.150 filas", 960, 450, tam=44, c="gris", w="sb", en="a", t=1.6),
    T("medimos sobre esas mismas filas", 960, 590, tam=50, c="tinta", en="b"), P("los AUC están inflados · ¿cuánto?", 960, 760, "rojo", 58, en="c"),
    sec="2. Lo que hicimos toda la clase")

# ---- 3. El experimento del ruido puro
v.b("Para medirlo, la clase arma un caso donde sabemos la respuesta de antemano. [[a]]Cien observaciones, [[b]]mil variables generadas al azar, [[c]]y un target lanzado a cara o sello. No hay ninguna señal: cualquier medición honesta tiene que dar AUC cero coma cinco.",
    CL(), T("el experimento del ruido puro", 960, 190, tam=48, c="tinta", ent="subir"),
    *[[Bc(x, 450, 440, 280, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(val, x, 420, tam=96, c=c, en=k, t=0.1), T(lab, x, 520, tam=34, c=c, w="sb", en=k, t=0.2)]
      for x, val, lab, c, k in ((440, "100", "observaciones", "azul", "a"), (960, "1.000", "variables al azar", "violeta", "b"), (1480, "50/50", "target: cara o sello", "ambar", "c"))],
    P("sin señal → AUC honesto ≈ 0.5", 960, 760, "verde", 56, en="c", t=1.0), sec="3. El experimento del ruido puro")
v.b("El procedimiento que casi todos escriben: [[a]]calcular el estadístico F de las mil variables contra el target, [[b]]quedarse con las veinte mejores, [[c]]y recién ahí hacer validación cruzada de cinco folds sobre esas veinte.",
    CL(), T("el procedimiento que casi todos escriben", 960, 190, tam=44, c="rojo", ent="subir"),
    *[[Bc(330 + 630 * i, 470, 540, 260, "blanco", "rojo", 5, 30, sombra=True, en=k, ent="subir"), T(str(i + 1), 330 + 630 * i, 390, tam=60, c="rojo", en=k, t=0.1),
       T(tx, 330 + 630 * i, 500, tam=36, c="tinta", w="sb", mw=470, en=k, t=0.2)] for i, (tx, k) in enumerate((("F de las 1.000 variables con todos los datos", "a"),
                                                                                                         ("quedarse con las 20 mejores", "b"), ("validación cruzada sobre esas 20", "c")))],
    *[L(600 + 630 * i, 470, 690 + 630 * i, 470, "gris", 6, True, en="bc"[i]) for i in range(2)],
    T("SelectKBest(f_classif, k=20).fit_transform(Xr, yr)  →  cross_val_score(...)", 960, 720, tam=34, c="tinta", w="m", en="c", t=0.6))
v.b("Resultado: [[a]]AUC cero coma ochocientos ochenta y dos. Con datos que no contienen absolutamente nada.",
    Bc(960, 880, 700, 150, "rojoclaro", "rojo", 4, 30, en="a"), N(RU["mal"], 960, 880, "AUC {:.3f}", 80, "rojo", en="a", t=0.2, dur=1.0))
v.b("El procedimiento correcto: [[a]]armar un Pipeline que encadene la selección y el modelo, y pasarle el pipeline completo a cross val score. [[b]]Así, en cada fold, las veinte variables se eligen usando solo los ochenta casos de entrenamiento. Los veinte de validación no participan.",
    CL(), T("el procedimiento correcto", 960, 190, tam=44, c="verde", ent="subir"),
    codigo(["pipe = Pipeline([('sel', SelectKBest(f_classif, k=20)),", "                 ('clf', LogisticRegression(max_iter=2000))])",
            "cross_val_score(pipe, Xr, yr, cv=5, scoring='roc_auc')"], y=380, tam=36, c="verde", en="a", dt=0.4),
    *[Bc(380 + 300 * j, 700, 260, 120, "verdeclaro" if j < 4 else "ambarclaro", "verde" if j < 4 else "ambar", 3, 18, en="b", t=0.1 * j) for j in range(5)],
    T("80 casos eligen las 20 variables", 830, 790, tam=34, c="verde", w="sb", en="b", t=0.6), T("20 validan", 1580, 790, tam=34, c="ambar", w="sb", en="b", t=0.8),
    T("(en cada uno de los 5 folds)", 960, 860, tam=30, c="gris", w="sb", en="b", t=1.0))
v.b("Resultado: [[a]]cero coma quinientos catorce. La verdad. [[b]]Mismo dato, mismo modelo, misma validación. Lo único que cambió es dónde vive la selección.",
    CL(), T("AUC · mismos datos de ruido puro", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    *[[Bx(x - 140, 860 - (a - 0.3) * 900, x + 140, 860, c, None, 0, 12, grow="y", en=k, t=t_), N(a, x, 860 - (a - 0.3) * 900 - 45, "{:.3f}", 64, c, en=k, t=t_ + 0.3, dur=0.8),
       T(lab, x, 905, tam=32, c=c, w="sb", en=k, t=t_)]
      for x, a, c, lab, k, t_ in ((480, RU["mal"], "rojo", "selección FUERA del CV", None, 0.0), (960, RU["bien"], "verde", "selección DENTRO del pipeline", "a", 0.0),
                                  (1440, 0.5, "gris", "la verdad", "a", 0.6))],
    L(300, 860, 1620, 860, "tinta", 4), T("eje desde 0.30", 1740, 860, tam=26, c="gris", w="sb"),
    P("misma métrica, mismos datos · solo cambia dónde vive la selección", 960, 990, "violeta", 38, en="b"))
v.b("Cero coma ochocientos ochenta y dos menos cero coma cinco: [[a]]treinta y ocho puntos de AUC inventados, a partir de datos que no contienen nada.",
    CL(), T("0.882 − 0.500 =", 820, 420, tam=90, c="tinta", ent="subir"), N(RU["inventado"], 1260, 420, "{:.3f}", 110, "rojo", en="a", anc="lm"),
    P("38 puntos de AUC inventados", 960, 640, "rojo", 70, en="a", t=0.6))

# ---- 4. Por qué pasa: α · p
v.b("¿Por qué pasa? [[a]]Con mil variables de ruido, algunas van a coincidir con el target por pura casualidad. [[b]]Si usas un test al cinco por ciento, el número esperado de falsos positivos es alfa por p.",
    CL(), T("¿por qué pasa?", 960, 200, tam=50, c="tinta", ent="subir"),
    Pt([(950, [300, 300, 1620, 640], "#CBD5E1"), (50, [300, 660, 1620, 700], "rojo")], 7, visibles=1000, en="a", dur=1.6),
    caja_formula("falsos positivos esperados = α · p", y=840, c="rojo", tam=56, w=1200, en="b"), sec="4. Por qué pasa: α · p")
FP = [("1.000", "0.05", "50"), ("200", "0.05", "10"), ("1.000", "0.01", "10")]
v.b("Tres casos del apunte. [[a]]Mil variables al cinco por ciento: cincuenta falsos positivos. [[b]]Doscientas al cinco por ciento: diez. [[c]]Mil al uno por ciento: también diez.",
    CL(), T("ejemplo del apunte", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    T("p (variables de ruido)", 500, 290, tam=32, c="gris", w="sb"), T("α", 960, 290, tam=32, c="gris", w="sb"), T("α · p", 1400, 290, tam=32, c="gris", w="sb"),
    *[[Bc(960, 400 + 140 * i, 1300, 110, "blanco", "rojo", 4, 22, en=k, ent="subir"), T(p_, 500, 400 + 140 * i, tam=52, c="tinta", en=k, t=0.1),
       T(a_, 960, 400 + 140 * i, tam=52, c="tinta", en=k, t=0.2), T(r_, 1400, 400 + 140 * i, tam=60, c="rojo", en=k, t=0.4)] for i, ((p_, a_, r_), k) in enumerate(zip(FP, "abc"))])
v.b("¿Y en los datos de ruido del notebook? Lo verifiqué. [[a]]Cincuenta y siete de las mil variables tienen p-valor bajo cero coma cero cinco: cerca de los cincuenta esperados. [[b]]Y diez bajo cero coma cero uno: justo lo esperado.",
    CL(), T("verificado con los datos de ruido de la clase (semilla 42)", 960, 200, tam=38, c="gris", w="sb", ent="subir"),
    *[[Bc(x, 480, 600, 300, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), N(val, x, 450, "{:.0f}", 120, c, en=k, t=0.1), T(lab, x, 570, tam=34, c=c, w="sb", en=k, t=0.3)]
      for x, val, lab, c, k in ((560, RU["signif05"], "con p < 0.05 · esperadas 50", "rojo", "a"), (1360, RU["signif01"], "con p < 0.01 · esperadas 10", "ambar", "b"))])
v.b("Elegimos las veinte que mejor coincidieron en estas cien observaciones concretas. [[a]]Después validamos sobre esas mismas cien: el fold de validación contiene justo los casos que las hicieron parecer buenas. [[b]]El modelo no generaliza: recuerda. Y la validación cruzada no lo detecta, porque la trampa ocurrió antes de partir los datos.",
    CL(), P("las 20 «mejores» = las que coincidieron por azar en estas 100", 960, 260, "violeta", 44, ent="subir"),
    T("el fold de validación ya influyó en la elección", 960, 440, tam=50, c="tinta", w="sb", en="a"),
    T("no generaliza: recuerda", 960, 620, tam=80, c="rojo", en="b"), T("la trampa ocurrió antes de partir los datos", 960, 760, tam=40, c="gris", w="sb", en="b", t=1.4))

# ---- 5. Qué va dentro de cada fold
TAB = [("eliminar edad 233", "No", "puede ir antes", "verde"), ("ratios y logaritmos", "No", "puede ir antes", "verde"), ("one-hot de categóricas", "No", "puede ir antes", "verde"),
       ("escalar (StandardScaler)", "No, pero usa la media", "DENTRO del fold", "rojo"), ("imputar por mediana", "No, pero usa la mediana", "DENTRO del fold", "rojo"),
       ("filtrar por información mutua", "Sí", "DENTRO del fold", "rojo"), ("elegir el par redundante", "Sí", "DENTRO del fold", "rojo"), ("RFE y Lasso", "Sí", "DENTRO del fold", "rojo")]
v.b("La regla de oro: si la decisión mira el target, va dentro de cada fold. [[a]]Borrar una edad imposible, construir ratios y logaritmos, y el one-hot, no miran nada: pueden ir antes. [[b]]Escalar e imputar no miran el target, pero usan la media y la mediana del conjunto: van dentro. [[c]]Y el filtro de información mutua, el desempate, el RFE y el Lasso, miran el target: van dentro.",
    CL(), T("qué va dentro de cada fold", 960, 170, tam=44, c="tinta", ent="subir"),
    T("paso", 450, 240, tam=30, c="gris", w="sb"), T("¿mira el target?", 1020, 240, tam=30, c="gris", w="sb"), T("¿dónde va?", 1480, 240, tam=30, c="gris", w="sb"),
    *[[Bc(960, 310 + 82 * i, 1500, 70, "verdeclaro" if c == "verde" else "rojoclaro", None, 0, 16, en=k, ent="subir", t=t_), T(a, 450, 310 + 82 * i, tam=32, c="tinta", w="sb", en=k, t=t_ + 0.05),
       T(b, 1020, 310 + 82 * i, tam=32, c="tinta", en=k, t=t_ + 0.1), T(r, 1480, 310 + 82 * i, tam=32, c=c, w="b", en=k, t=t_ + 0.15)]
      for i, ((a, b, r, c), k, t_) in enumerate(zip(TAB, "aaabbccc", (0, 0.3, 0.6, 0, 0.4, 0, 0.3, 0.6)))],
    sec="5. Qué va dentro de cada fold")
v.b("Regla de bolsillo del apunte: [[a]]si un paso necesita fit, va dentro del Pipeline. Si es una fórmula fija, fila por fila, puede ir antes. [[b]]Y una precisión sobre el propio notebook: los cuartiles, y los indicadores que usan la mediana o el percentil setenta y cinco de toda la base, sí usan estadísticas del conjunto. [[c]]Por la misma regla, en rigor, van dentro del fold. Aquí no medimos cuánto cambia, pero la regla es la regla.",
    CL(), P("¿necesita fit? → dentro del Pipeline", 960, 240, "rojo", 54, ent="subir", en="a"), P("¿fórmula fija fila por fila? → puede ir antes", 960, 380, "verde", 50, en="a", t=0.8),
    P("precisión sobre el notebook", 960, 530, "ambar", 44, en="b"),
    T("pd.qcut(d.clv, 4) · uso_alto (mediana) · clv_alto (percentil 75)", 960, 640, tam=38, c="tinta", w="sb", en="b", t=0.4),
    T("usan estadísticas de toda la base → en rigor, dentro del fold", 960, 760, tam=42, c="ambar", w="b", en="c"))

# ---- 6. El pipeline honesto con la base real
v.b("Ahora, rehacer la clase entera, esta vez bien. [[a]]Un Pipeline con cuatro pasos, en este orden: imputar con la mediana, escalar, seleccionar doce variables por información mutua, y la regresión logística. [[b]]Todo dentro del mismo objeto, para que la validación cruzada lo repita entero en cada fold.",
    CL(), T("el pipeline honesto de la clase", 960, 170, tam=44, c="tinta", ent="subir"),
    codigo(["pipe = Pipeline([", "    ('imp', SimpleImputer(strategy='median')),", "    ('sc',  StandardScaler()),",
            "    ('sel', SelectKBest(mutual_info_classif, k=12)),", "    ('clf', LogisticRegression(max_iter=3000))])",
            "cv = StratifiedKFold(5, shuffle=True, random_state=1)", "cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')"], y=520, tam=34, c="verde", en="a", dt=0.35),
    P("imputar → escalar → seleccionar → modelar", 960, 900, "verde", 46, en="b"), sec="6. El pipeline honesto")
v.b("El resultado del notebook, fold por fold: [[a]]cero coma ocho nueve cinco, cero coma nueve uno cuatro, cero coma ocho nueve, cero coma nueve dos tres, y cero coma ocho ocho siete. [[b]]Promedio: cero coma nueve cero uno. [[c]]Al repetirlo da lo mismo hasta la tercera cifra: la información mutua dentro del pipeline no tiene semilla, y cambia en la cuarta.",
    CL(), T("AUC por fold · notebook de la clase", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    *[[Bx(330 + 280 * i - 100, 800 - (a - 0.8) * 3000, 330 + 280 * i + 100, 800, "verde", None, 0, 10, grow="y", en="a", t=0.4 * i),
       N(a, 330 + 280 * i, 800 - (a - 0.8) * 3000 + 36, "{:.4f}", 36, "blanco", en="a", t=0.4 * i + 0.3), T(f"fold {i + 1}", 330 + 280 * i, 840, tam=30, c="gris", w="sb", en="a", t=0.4 * i)]
      for i, a in enumerate(NB_FOLDS)],
    L(200, 800, 1520, 800, "tinta", 4), L(200, 800 - (NB_MEDIA - 0.8) * 3000, 1520, 800 - (NB_MEDIA - 0.8) * 3000, "violeta", 4, guion=True, en="b"),
    N(NB_MEDIA, 1700, 800 - (NB_MEDIA - 0.8) * 3000, "media {:.4f}", 44, "violeta", en="b", t=0.3),
    T(f"repetido: {D['honesto']['media']:.4f}", 1700, 570, tam=34, c="gris", w="sb", en="c"), T("eje desde 0.80", 1700, 840, tam=26, c="gris", w="sb"))
v.b("¿Y cuánto inflaba la trampa con la base real? Lo medí con la misma selección, y la misma semilla. [[a]]Seleccionando las doce afuera del CV: cero coma nueve cero uno tres. [[b]]Adentro del pipeline: cero coma ocho nueve nueve seis. [[c]]Con tres mil filas y señal fuerte, la trampa infla poco: menos de dos décimas de punto. Con cien filas y mil variables, treinta y ocho puntos. La regla es la misma; el daño depende de cuántas variables tengas por fila.",
    CL(), T("base real · top-12 por MI (random_state = 0)", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    *[[Bc(x, 430, 560, 240, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(lab, x, 360, tam=36, c=c, w="sb", en=k, t=0.1),
       N(a, x, 450, "{:.4f}", 90, c, en=k, t=0.2)] for x, lab, a, c, k in ((560, "selección FUERA del CV", RF["fuera"], "rojo", "a"), (1360, "selección DENTRO", RF["dentro"], "verde", "b"))],
    P(f"base real: +{(RF['fuera'] - RF['dentro']) * 100:.2f} puntos  ·  ruido puro: +{RU['inventado'] * 100:.0f} puntos", 960, 680, "violeta", 46, en="c"),
    T("el daño crece con las variables por fila", 960, 800, tam=40, c="tinta", w="sb", en="c", t=2.0))

# ---- 7. StratifiedKFold, AUC y la ventana temporal
v.b("Dos decisiones del pipeline, explicadas. [[a]]StratifiedKFold conserva el quince coma siete por ciento de fuga en cada fold; una partición simple podría dejar un fold casi sin fugados. [[b]]Y el AUC: decir «nadie se va» acierta el ochenta y cuatro coma tres por ciento, pero su AUC es cero coma cinco.",
    CL(), tarjetas([(560, "teal", "StratifiedKFold", "cada fold conserva el 15.7 % de fuga", "a"), (1360, "violeta", "AUC, no exactitud", "«nadie se va» = 84.3 % de exactitud, AUC 0.5", "b")],
                   y=480, w=700, h=360), sec="7. Estratificar, AUC y la ventana temporal")
v.b("Y la trampa que ningún pipeline detecta: la ventana temporal. [[a]]Si quieres predecir la fuga de agosto, las variables se miden de junio hacia atrás. [[b]]Una variable como «llamó para dar de baja el servicio» casi siempre ocurre después de la decisión que quieres predecir. Es leakage, aunque no uses la columna fuga.",
    CL(), L(260, 450, 1660, 450, "tinta", 5), *[[Bc(x, 450, 22, 22, c, None, 0, 11, t=0.15 * j), T(m, x, 500, tam=36, c=c, w="sb", t=0.15 * j)]
                                                for j, (x, m, c) in enumerate(((400, "abril", "teal"), (700, "mayo", "teal"), (1000, "junio", "teal"), (1300, "julio", "gris"), (1600, "agosto", "rojo")))],
    Bx(260, 400, 1080, 420, "teal", None, 0, 6, grow="x", en="a"), T("variables: de junio hacia atrás", 670, 360, tam=38, c="teal", w="sb", en="a", t=0.4),
    T("target: fuga de agosto", 1600, 360, tam=38, c="rojo", w="sb", en="a", t=0.6),
    P("«llamó para dar de baja» → información del futuro = leakage", 960, 680, "rojo", 46, en="b"))
v.b("Por eso el profesor advierte: [[a]]en la industria, un AUC de cero coma ochenta y cuatro a cero coma ochenta y cinco en fuga ya es extremadamente bueno. [[b]]Un número demasiado bueno, de libro, suele esconder leakage. Desconfía de las métricas perfectas.",
    CL(), P("AUC 0.84–0.85 en fuga = «extremadamente bueno»", 960, 330, "verde", 52, en="a"),
    P("demasiado bueno → sospecha leakage", 960, 540, "rojo", 56, en="b"))

# ---- 8. Error típico
v.b("El error típico de esta parada: [[a]]elegir las variables con todos los datos, y después validar. [[b]]La validación sale preciosa, y es mentira. La selección tiene que vivir dentro del Pipeline.",
    CL(), T("seleccionar con todos los datos → validar", 960, 330, tam=64, c="tinta", en="a", id="sv"), U("sv", tachado=1.0, color="gris", en="b", t=0.1, dur=0.6),
    P("validación preciosa · y mentirosa", 960, 520, "rojo", 56, en="b", t=0.4), P("la selección vive dentro del Pipeline", 960, 690, "verde", 56, en="b", t=1.4),
    sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Imputas los nulos con la mediana de toda la base, antes de la validación cruzada. No miraste la fuga. [[b]]¿Hay leakage?",
    CL(), T("fillna(mediana de toda la base) → después el CV", 960, 330, tam=56, c="violeta", w="sb", en="a"), T("no miraste la fuga", 960, 470, tam=50, c="gris", w="sb", en="a", t=0.8),
    T("¿hay leakage?", 960, 640, tam=80, c="tinta", en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("Sí, aunque sea leve. [[a]]La mediana se calculó también con los casos de validación. [[b]]La solución es la misma de siempre: SimpleImputer dentro del Pipeline, para que cada fold use solo la mediana de su entrenamiento.",
    CL(), P("sí, aunque sea leve", 960, 260, "rojo", 60, ent="subir"),
    T("la mediana usó los casos de validación", 960, 430, tam=50, c="tinta", w="sb", en="a"),
    T("('imp', SimpleImputer(strategy='median'))  dentro del Pipeline", 960, 620, tam=40, c="verde", w="m", en="b"))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]toda decisión que mire el target, o que use estadísticas del conjunto, va dentro de cada fold. [[b]]Con ruido puro, seleccionar afuera inventó treinta y ocho puntos de AUC. [[c]]Y además del pipeline, cuida la ventana temporal: nada del futuro.",
    CL(), P("mira el target o usa el conjunto → dentro del fold", 960, 300, "violeta", 50, en="a"),
    P(f"ruido puro: {RU['mal']:.3f} afuera · {RU['bien']:.3f} adentro · verdad 0.500", 960, 480, "rojo", 46, en="b"),
    P("ventana temporal: de junio hacia atrás", 960, 660, "teal", 50, en="c"), sec="10. Para llevar")
v.b("Última parada: [[a]]el repaso de la clase, y el checklist del proyecto.",
    CL(), siguiente(10, "Repaso final y checklist"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
