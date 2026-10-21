"""Clase 34 · Árboles de decisión: guía visual, apuntes y cuaderno."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_glm import code, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C34 = colors.HexColor("#16A34A")
C34_BG = colors.HexColor("#F0FDF4")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + notebook Python del profesor"


def topic_page(s, titulo, idea, conceptos, figura, pie, formulas, py, salida, cuidado):
    s.append(band(f"Clase 34 · {titulo}", FUENTE, C34))
    s.append(Spacer(1, 7))
    s.append(box([P(idea, "big")], bg=SOFT, border=C34))
    s.append(Spacer(1, 6))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C34))
    s.append(Spacer(1, 4))
    if figura:
        s.append(img(figura, width=CONTENT_W * 0.97))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 5))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF),
            Spacer(1, 5), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C34), Spacer(1, 5),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER,
                 width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 34 · Árboles de decisión",
                  "Profesor Eduardo Rodríguez · Etapa 7 del roadmap · misma base de fuga de la Clase 33", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Un árbol parte el espacio con preguntas de sí o no y, en cada hoja, predice la tasa de su grupo. Es «la técnica "
                    "más simple y a la vez más peligrosa», en palabras del profesor: la más fácil de explicar y la más fácil de "
                    "<b>sobreajustar</b>. Se elige cuando la explicación vale más que unos puntos de AUC.", "body")],
                 bg=C34_BG, border=C34))
    s.append(Spacer(1, 6))
    s.append(P("El árbol de la clase", "h2"))
    s.append(img("c34_arbol.png", width=CONTENT_W))
    s.append(Spacer(1, 4))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Limpieza y ganancia de información (Clase 33)",
                     "Validación cruzada y AUC (Clases 30–33)",
                     "Regresión logística, para comparar (Carpeta 2)",
                     "Proporciones y logaritmo en base 2"], "cell", C34)]
    right = [P("Tres cosas de la Clase 33 que hoy se explican solas", "h3"),
             bullets(["La <b>ganancia de información</b> que usamos como filtro es el criterio con que el árbol elige cada corte",
                      "La <b>importancia de un árbol</b> es selección embebida: el árbol elige variables al entrenar",
                      "<b>reclamos</b> domina: es la raíz del árbol, con una ganancia que más que duplica a la segunda"], "cell", C34)]
    s.append(two_cols(left, right))
    s.append(PageBreak())

    topic_page(
        s, "Cómo parte el espacio un árbol",
        "La regresión logística ajusta un solo plano para todo el espacio. El árbol no ajusta ninguna fórmula: corta el espacio "
        "en rectángulos y en cada uno predice la proporción de fuga que observó.",
        [("Regiones", "Cada hoja es una región R<sub>m</sub> y su predicción es la tasa de fuga de los clientes que cayeron ahí."),
         ("No linealidad e interacciones solas", "Una rama puede usar variables distintas a otra. Por eso la clase NO construye las 50 variables de la Clase 33: el árbol llega a lo mismo partiendo el espacio, y con 12 variables se puede leer."),
         ("Segmentos, no coeficientes", "La hoja más roja tiene 149 clientes con 94% de fuga: «reclamó, usa poco el servicio y tuvo fallas». Es un segmento al que se le puede mandar una campaña mañana."),
         ("Campañas de precisión", "La hoja de 69 clientes (2.2% de la base) con 35% de fuga sirve para una campaña de bajo presupuesto. Cruzarla con el CLV dice si vale la pena retener a cada cliente."),
         ("Sin escalar", "El árbol solo pregunta «¿x ≤ umbral?». No necesita estandarizar ni transformar las variables.")],
        None, None,
        ["Logística: log(p / (1 − p)) = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + …",
         "Árbol: p̂(x) = Σ<sub>m</sub> c<sub>m</sub> · 1{x ∈ R<sub>m</sub>}",
         "c<sub>m</sub> = tasa de fuga en la hoja m",
         "lift = tasa de la hoja / tasa general"],
        ["arbol = DecisionTreeClassifier(",
         "    max_depth=3, min_samples_leaf=60,",
         "    random_state=0).fit(X, y)",
         "export_text(arbol, feature_names=list(X.columns))",
         "hoja = arbol.apply(X)   # hoja de cada cliente"],
        ["7 hojas, profundidad 3, mínimo 60 clientes por hoja",
         "Segmentos: 149 → 94% · 67 → 67% · 340 → 43% · 62 → 40% · 69 → 35% · 104 → 9% · 2.359 → 4.5%",
         "La hoja de 94% tiene un lift de 6: seis veces la tasa general (15.7%)"],
        ["min_samples_leaf = 60 no es un detalle: con hojas de 5 clientes, una tasa de 100% puede ser puro ruido de muestra chica."])

    topic_page(
        s, "El criterio de corte: Gini y entropía",
        "En cada nodo, el árbol prueba todas las variables y todos los umbrales, y se queda con el corte que más reduce la mezcla "
        "de clases. Es un algoritmo voraz: elige lo mejor ahora, sin mirar dos pasos adelante.",
        [("Impureza", "Mide qué tan mezclado está un nodo. Vale 0 si el nodo es puro y es máxima con 50/50 (Gini 0.5, entropía 1)."),
         ("Los p<sub>i</sub> son del nodo", "Se recalculan en cada nodo con los clientes que caen ahí. En la raíz: 495 de 3.150 se fugan → Gini 0.265."),
         ("Ganancia", "Impureza del padre menos la impureza ponderada de los hijos. Es la misma fórmula que usamos como filtro en la Clase 33."),
         ("El barrido", "Para la raíz se evaluaron 378 combinaciones de variable y umbral. Y el barrido completo se repite en CADA nodo."),
         ("Gini o entropía", "Suelen dar árboles parecidos; la entropía viene de la teoría de la información (la «medida del desorden»). Aquí, con profundidad 5: Gini 0.906 vs entropía 0.920.")],
        "c34_impureza.png",
        "Izquierda: las dos medidas según la proporción de fuga del nodo. Derecha: el primer corte real, reclamos, separa un grupo de 76% de fuga de otro de 10%.",
        ["Gini = 1 − Σ p<sub>i</sub>²",
         "Entropía = − Σ p<sub>i</sub> · log<sub>2</sub> p<sub>i</sub>",
         "ganancia = imp(padre) − Σ (n<sub>j</sub>/n) · imp(hijo<sub>j</sub>)",
         "raíz: 1 − 0.843² − 0.157² = 0.265"],
        ["def gini(p): return 1 - p**2 - (1 - p)**2",
         "DecisionTreeClassifier(criterion='entropy')",
         "cross_val_score(arbol, X, y, cv=skf,",
         "    scoring='roc_auc')"],
        ["Reclamó: 278 clientes, 210 fugan (75.5%), Gini 0.370",
         "No reclamó: 2.872, 285 fugan (9.9%), Gini 0.179",
         "Ponderado 0.196 → ganancia 0.069, más del doble que la segunda (estado, 0.027)",
         "Edad apenas gana 0.0005: sus hijos se parecen al padre"],
        ["Voraz significa que no garantiza el árbol óptimo, solo uno bueno y rápido. Un corte malo arriba condiciona todo lo de abajo."])

    topic_page(
        s, "Sobreajuste y validación cruzada",
        "Un árbol sin límite crece hasta que cada hoja es pura: acierta todo el entrenamiento y memoriza. La curva de profundidad "
        "muestra exactamente dónde deja de aprender y empieza a memorizar.",
        [("Train vs validación", "El AUC de entrenamiento siempre sube con la profundidad. El de validación sube hasta un punto y después baja: esa brecha es el sobreajuste."),
         ("El punto de quiebre", "Con los datos de la clase, la mejor profundidad es 6 (AUC 0.911). Sin límite, el árbol tiene 224 hojas, AUC de entrenamiento 1.000 y de validación 0.827."),
         ("Validación cruzada", "5 bloques de 630 clientes: se entrena con 2.520 y se mide en el bloque restante, rotando. Importa el promedio Y el desvío entre rondas."),
         ("Separar al azar, no por fecha", "Partir train y test por fecha sin querer es «un error gravísimo, pero pasa». Estratifica para mantener el 15.7% de fuga en cada bloque."),
         ("Tolerancia práctica", "El profesor acepta como mucho unos 2 puntos de diferencia entre entrenamiento y validación. En la industria, un AUC de 0.81–0.85 en fuga ya es bueno.")],
        "c34_profundidad.png",
        "Curva real con los datos de la clase. El área rosada es la brecha entre lo que el árbol memoriza y lo que generaliza.",
        ["sobreajuste = AUC train − AUC validación",
         "k-fold: k modelos, cada uno validado en 1/k de los datos",
         "elegir la profundidad con el MÁXIMO AUC de validación"],
        ["skf = StratifiedKFold(5, shuffle=True, random_state=0)",
         "cross_validate(arbol, X, y, cv=skf,",
         "    scoring='roc_auc', return_train_score=True)"],
        ["Rondas (profundidad 5): 0.901 · 0.920 · 0.911 · 0.918 · 0.880",
         "Promedio 0.906 · desvío 0.015",
         "Profundidad 3 → 0.850 · 6 → 0.911 · 15 → 0.838"],
        ["Un AUC de 0.90+ «demasiado bueno» en datos reales suele ser leakage, no un gran modelo. Si una sola variable explica casi "
         "todo, el profesor sugiere evaluar submodelos por segmento: la población puede no ser homogénea."])

    topic_page(
        s, "Poda e importancia de variables",
        "Hay dos formas de frenar al árbol: antes (pre-poda, con límites) o después (post-poda, recortando ramas que no compensan). "
        "Y para saber qué variables importan, la importancia por impureza engaña: hay que usar permutación.",
        [("Pre-poda", "Limita el crecimiento durante el entrenamiento: max_depth, min_samples_leaf, min_samples_split, max_leaf_nodes."),
         ("Post-poda por costo-complejidad", "Deja crecer el árbol y lo recorta: cada hoja debe «pagar» α. Con α = 0 sobrevive todo; al subir α quedan menos hojas. α se elige por validación cruzada."),
         ("El punto dulce", "Con los datos de la clase, α = 0.001 deja 42 hojas y AUC 0.900; la búsqueda del notebook elige α = 0.0013, con 36 hojas y AUC 0.909."),
         ("Importancia por impureza (sesgada)", "Premia a las variables con muchos valores distintos: tienen más umbrales para probar y, por azar, alguno reduce la impureza. clv tiene 2.654 valores."),
         ("Importancia por permutación", "Desordena una variable en datos no vistos y mide cuánto cae el AUC. Ahí clv baja y estado (binaria) sube: la impureza premiaba la cardinalidad.")],
        "c34_poda.png",
        "Izquierda: hojas y AUC según α (datos de la clase). Derecha: importancia relativa de un Random Forest por los dos métodos.",
        ["costo<sub>α</sub>(T) = R(T) + α · |hojas(T)|",
         "α = 0 → árbol completo · α ↑ → menos hojas",
         "imp. permutación = AUC − AUC(con X<sub>j</sub> desordenada)"],
        ["path = arbol.cost_complexity_pruning_path(X, y)",
         "GridSearchCV(DecisionTreeClassifier(),",
         "    {'ccp_alpha': path.ccp_alphas[:-1]},",
         "    cv=skf, scoring='roc_auc')",
         "permutation_importance(modelo, X_test, y_test,",
         "    n_repeats=20, scoring='roc_auc')"],
        ["α: 0 → 224 hojas, 0.827 · 0.001 → 42, 0.900 · 0.01 → 3, 0.805",
         "Impureza: reclamos 0.206 · segundos 0.118 · clv 0.109 · estado 0.045",
         "Permutación: reclamos 0.035 · segundos 0.032 · clv 0.014 · estado 0.019"],
        ["La importancia por permutación se mide SIEMPRE sobre datos que el modelo no vio. Sobre el entrenamiento hereda el sobreajuste."])

    topic_page(
        s, "Fortalezas, límites y el salto al bosque",
        "El árbol se dibuja y se explica a cualquiera, pero es inestable y un árbol solo suele predecir peor que una logística. "
        "La solución a la inestabilidad es promediar muchos árboles: el Random Forest.",
        [("Invariante a transformaciones monótonas", "Aplicar log o raíz a una variable mueve el umbral pero no cambia quién queda a cada lado: el AUC del árbol queda idéntico (0.906). La logística sí cambia (0.920 → 0.923)."),
         ("Inestable", "Con 6 muestras distintas, la raíz siempre es reclamos, pero el segundo nivel cambia (clv, segundos o frec_llamadas). Con reentrenamientos trimestrales, eso es explicar un árbol distinto cada vez."),
         ("Otros límites", "Solo corta perpendicular a los ejes (una diagonal le cuesta muchos escalones) y predice constantes por región: no extrapola."),
         ("El bosque", "Muchos árboles, cada uno con una muestra bootstrap y un subconjunto de variables por corte (√12 ≈ 3), promediados. Con 300 árboles: AUC 0.974."),
         ("¿Para qué es el modelo?", "El árbol de profundidad 3 cabe en una lámina; el bosque no. Es el mismo trade-off de la Clase 32: se cede interpretabilidad a cambio de desempeño.")],
        "c34_bosque.png",
        "Izquierda: el AUC del bosque según el número de árboles. Derecha: todos los modelos de la clase, con validación cruzada.",
        ["Var(promedio) = ρσ² + (1 − ρ)σ² / B",
         "B = n° de árboles · ρ = correlación entre árboles",
         "max_features = √p decorrelaciona los árboles"],
        ["RandomForestClassifier(n_estimators=300,",
         "    max_features='sqrt', n_jobs=-1,",
         "    random_state=0)",
         "np.log1p(X.segundos)   # no cambia el árbol"],
        ["Árboles en el bosque: 1 → 0.797 · 5 → 0.931 · 25 → 0.968 · 300 → 0.974",
         "Solo bootstrap 0.969 · bootstrap + subconjunto 0.974",
         "Logística 0.920 · árbol podado 0.909 · árbol prof. 3 0.850"],
        ["El profesor confiesa su preferencia entre los modelos clásicos: Gradient Boosting (Clase 36). Hoy los ensambles son tan "
         "baratos de entrenar que un árbol solo se elige casi solo por explicabilidad."])

    s.append(band("Chuleta, hiperparámetros y checklist", "Lo que vas a ajustar en cualquier modelo de árboles", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["Hiperparámetro", "Qué controla", "Rango típico", "Efecto si lo subes"],
                    ["max_depth", "preguntas encadenadas", "3–8", "más complejo, más sobreajuste"],
                    ["min_samples_leaf", "clientes mínimos por hoja", "5–60 (clase: 60)", "más simple, tasas más estables"],
                    ["min_samples_split", "clientes mínimos para partir", "10–50", "más simple"],
                    ["max_leaf_nodes", "hojas máximas", "10–50", "más complejo"],
                    ["ccp_alpha", "precio de cada hoja (post-poda)", "0–0.05", "más poda, menos hojas"],
                    ["criterion", "medida de impureza", "gini / entropy", "—"],
                    ["class_weight", "peso de la clase minoritaria", "None / 'balanced'", "más sensibilidad a la fuga"]],
                   [32 * mm, 50 * mm, 34 * mm, CONTENT_W - 116 * mm]))
    s.append(Spacer(1, 4))
    s.append(P("Rangos típicos tomados de la presentación genérica de la clase («Arbol de decision.pptx»).", "small"))
    s.append(Spacer(1, 8))
    s.append(P("Checklist para entrenar un árbol", "h2"))
    checks = [("Separa train y test al azar (y estratificado)", "nunca por fecha sin querer"),
              ("Fija min_samples_leaf con criterio", "hojas muy chicas dan tasas de ruido"),
              ("Traza la curva de profundidad", "elige donde el AUC de validación es máximo"),
              ("Poda con ccp_alpha validado", "GridSearchCV, no a ojo"),
              ("Revisa el desvío entre folds", "un buen promedio con mucho desvío no es estable"),
              ("Mide importancia por permutación", "sobre el test, nunca sobre el entrenamiento"),
              ("Traduce las hojas a reglas", "n, tasa y lift de cada segmento"),
              ("Compara con una logística y un bosque", "y decide según para qué es el modelo")]
    s.append(table([["", "Paso", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 66 * mm, CONTENT_W - 74 * mm]))
    s.append(PageBreak())

    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales, preguntas clave y cuándo pasar a la Clase 35", NAVY))
    s.append(Spacer(1, 8))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Errores y diferencias en las diapositivas (verificados)</b>", "cellb"),
                  bullets(["<b>Diapositiva 13, nota de respuesta:</b> dice que el candidato A (reclamos) da «Gini ponderado ~0,168 y ganancia "
                           "~0,097». Con los datos de la misma diapositiva da <b>0.195 y 0.070</b>, lo mismo que la diapositiva 12. Además, "
                           "esa diapositiva usa 211 y 284 fugas, y los datos reales son 210 y 285.",
                           "<b>Diapositiva 19:</b> muestra como resultado {'ccp_alpha': 0.001}, AUC 0.8996 y 42 hojas. El notebook, ejecutado con "
                           "los datos, elige <b>α = 0.0013, AUC 0.909 y 36 hojas</b>: usa otra grilla de valores. El 0.8996 corresponde a la fila "
                           "α = 0.001 de la tabla de la diapositiva 18.",
                           "<b>Diapositivas 25 y 27:</b> una dice «el mejor árbol solo 0,911» y la otra «solo llega a 0,906». No se contradicen: "
                           "0.911 es la profundidad 6 y 0.906 la profundidad 5.",
                           "<b>Diapositiva 11:</b> la fila «84% / 16% ← nuestra raíz» muestra Gini 0.269 porque redondea. El valor exacto es 0.265."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Otras observaciones</b>", "cellb"),
                  bullets(["El comentario del notebook dice que Gini y entropía «dan árboles casi idénticos», pero con profundidad 5 la "
                           "entropía da 1.4 puntos más de AUC (0.920 vs 0.906). Aquí sí hay diferencia.",
                           "El notebook carga <font name='Mono'>churn_telco.csv</font>, que no existe. Tu notebook guiado usa el Excel de la Clase 33.",
                           "<b>La transcripción de esta clase tiene ruido:</b> bucles de Whisper (líneas 855–857 y 893–901), un fragmento ajeno a "
                           "la clase sobre cómo pausar OBS Studio (líneas 1085–1101) y términos mal reconocidos: «Spirman» (Spearman), «la nova» "
                           "(ANOVA), «vernugis» (Bernoulli), «araso» (Lasso), «Scorchart» (scorecard) y «Attention y Solve Unit» (el paper "
                           "«Attention Is All You Need», 2017)."], "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 8))
    s.append(P("Preguntas de repaso", "h2"))
    preg = [("¿Por qué se dice que el árbol es «voraz»?", "Elige el mejor corte en cada nodo sin mirar los siguientes: no garantiza el árbol óptimo."),
            ("Gini de un nodo con 15 de clase A y 5 de B", "p = 0.75 → 1 − 0.5625 − 0.0625 = 0.375 (entropía 0.811)."),
            ("¿Qué diferencia la pre-poda de la post-poda?", "Pre-poda limita al crecer (max_depth, min_samples_leaf); post-poda recorta después con ccp_alpha."),
            ("¿Por qué engaña la importancia por impureza?", "Premia variables con muchos valores; la permutación sobre el test lo corrige."),
            ("99% en train y 71% en test, ¿qué haces?", "Regularizar (profundidad, hojas mínimas), podar con α validado y solo después probar un bosque."),
            ("¿Por qué el árbol no necesita escalar?", "Solo pregunta x ≤ umbral: una transformación monótona mueve el umbral, no la partición.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 70 * mm, CONTENT_W - 77 * mm]))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 35</b>", "cellb"), Spacer(1, 3),
                  bullets(["Calculo a mano Gini, entropía y la ganancia de un corte",
                           "Leo un árbol y traduzco sus hojas a reglas con n, tasa y lift",
                           "Diagnostico sobreajuste con la curva de profundidad",
                           "Podo con ccp_alpha elegido por validación cruzada",
                           "Explico por qué uso importancia por permutación",
                           "Resuelvo los bloques 1 a 8 del cuaderno con al menos 80% de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C34_Guia_ArbolesDecision.pdf", s, "Clase 34 · Árboles de decisión")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#16A34A">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 34 · Árboles de decisión", "Para leer antes de los ejercicios · lectura de ~50 min", C34))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("Rectángulos", "El árbol corta el espacio con preguntas sí/no; cada hoja predice la tasa de fuga de su grupo."),
                      ("Impureza", "Gini o entropía miden la mezcla de un nodo. El mejor corte es el de mayor ganancia."),
                      ("Voraz", "Elige lo mejor en cada nodo, sin mirar adelante."),
                      ("Sobreajuste", "Sin límite, memoriza. Se controla con pre-poda (límites) y post-poda (ccp_alpha validado)."),
                      ("Importancia", "Usa permutación sobre el test; la impureza premia a las variables con muchos valores."),
                      ("Al bosque", "Un árbol es explicable pero inestable; promediar muchos (Random Forest) predice mucho mejor."),
                  ], C34)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C34_ArbolesDecision.ipynb</font> o el cuaderno "
               "<font name='Mono'>C34_Ejercicios_Cuaderno_ArbolesDecision.pdf</font>.", "small"))

    s.append(h("1", "Qué hace un árbol"))
    s += para("Un árbol de decisión hace preguntas encadenadas: «¿reclamó?», «¿usa menos de 934 segundos?»… Cada respuesta te lleva por "
              "una rama hasta una <b>hoja</b>. La predicción es la proporción de fuga de los clientes de entrenamiento que llegaron a esa "
              "hoja. A diferencia de la logística, no hay una fórmula con coeficientes: hay <b>regiones</b>.")
    s.append(img("c34_arbol.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(ejemplo("recorrer el árbol con un cliente", [
        P("Cliente: reclamó, hizo 40 llamadas, tuvo 0 fallas.", "cell"),
        pasos([["Nodo", "Pregunta", "Respuesta → rama"],
               ["Raíz", "reclamos ≤ 0.5", "no (sí reclamó) → derecha"],
               ["Nivel 2", "frec_llamadas ≤ 57.5", "sí (40) → izquierda"],
               ["Nivel 3", "numero_fallas ≤ 0.5", "sí (0 fallas) → izquierda"],
               ["Hoja", "67 clientes", "probabilidad de fuga 67%"]]),
        P("Lift de esa hoja: 0.67 / 0.157 = 4.3 veces la tasa general.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("En cada nodo, «sí» significa que la condición x ≤ umbral se cumple y vas a la izquierda. Escribe la ruta como regla: "
                      "«reclamó Y llamadas ≤ 57.5 Y fallas ≤ 0.5».", "cell")]))

    s.append(h("2", "Impureza: Gini y entropía"))
    s += para("Un nodo es <b>puro</b> si todos sus clientes son de la misma clase. La impureza mide cuánto se aleja de eso. Con dos "
              "clases y p = proporción de fuga en el nodo:")
    s.append(box([P("Gini = 1 − p² − (1 − p)²   ·   Entropía = −p·log<sub>2</sub>(p) − (1 − p)·log<sub>2</sub>(1 − p)", "formula")], bg=SOFT))
    s.append(Spacer(1, 4))
    s.append(ejemplo("tres nodos", [
        pasos([["Nodo", "p (fuga)", "Gini", "Entropía"],
               ["Puro", "0", "1 − 0 − 1 = 0", "0"],
               ["20% de fuga", "0.2", "1 − 0.04 − 0.64 = 0.32", "−0.2·(−2.32) − 0.8·(−0.32) = 0.722"],
               ["Mitad y mitad", "0.5", "1 − 0.25 − 0.25 = 0.5", "1 (el máximo)"]],
              [30 * mm, 22 * mm, 50 * mm, CONTENT_W - 102 * mm - 18]),
        P("Para la entropía necesitas log<sub>2</sub>: log<sub>2</sub>(x) = ln(x) / ln(2). Por ejemplo, log<sub>2</sub>(0.2) = −1.609 / 0.693 = −2.32.", "cell")]))

    s.append(h("3", "Elegir el corte: la ganancia"))
    s += para("Para cada variable y cada umbral, el árbol calcula la impureza de los dos hijos, la pondera por su tamaño y la compara con "
              "la del padre. La diferencia es la <b>ganancia</b>. Gana el corte con mayor ganancia, y el proceso se repite en cada hijo.")
    s.append(ejemplo("padre de 100 clientes con 20 fugas (Gini 0.32)", [
        pasos([["Corte", "Hijo izquierdo", "Hijo derecho", "Gini ponderado", "Ganancia"],
               ["A", "40 clientes, 16 fugan: Gini 0.48", "60 clientes, 4 fugan: Gini 0.124", "0.4·0.48 + 0.6·0.124 = 0.267", "0.32 − 0.267 = 0.053"],
               ["B", "50 clientes, 10 fugan: Gini 0.32", "50 clientes, 10 fugan: Gini 0.32", "0.32", "0"]],
              [14 * mm, 44 * mm, 44 * mm, 40 * mm, CONTENT_W - 142 * mm - 18]),
        P("El árbol elige A. B produce hijos idénticos al padre: ganancia cero. Es lo que le pasa a <b>edad</b> en la clase (ganancia 0.0005).", "cell")]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte B del notebook y los Bloques 2 a 4 del cuaderno",
                   [P("Ordena el cálculo siempre igual: (1) proporción de cada hijo, (2) Gini de cada hijo, (3) ponderar por n<sub>hijo</sub>/n, "
                      "(4) restar al Gini del padre.", "cell")]))

    s.append(h("4", "Sobreajuste y validación cruzada"))
    s += para("Si lo dejas crecer sin límite, el árbol sigue partiendo hasta que cada hoja es pura: 224 hojas con los datos de la clase. "
              "Acierta todo el entrenamiento (AUC 1.000), pero en datos nuevos cae a 0.827: memorizó el ruido. La <b>curva de "
              "profundidad</b> compara el AUC de entrenamiento con el de validación cruzada para cada profundidad, y la mejor es la que "
              "maximiza el de validación (aquí, 6, con 0.911).")
    s.append(img("c34_profundidad.png", width=CONTENT_W * 0.92))

    s.append(h("5", "Pre-poda y post-poda"))
    s += para("La <b>pre-poda</b> limita al árbol mientras crece (max_depth, min_samples_leaf…). La <b>post-poda</b> lo deja crecer y después "
              "recorta las ramas que no compensan su costo. sklearn usa la poda por costo-complejidad: cada hoja «cuesta» α.")
    s.append(ejemplo("¿qué árbol gana según α?", [
        P("Árbol grande T1: error 0.10 con 20 hojas · árbol chico T2: error 0.13 con 5 hojas. Costo = error + α · hojas.", "cell"),
        pasos([["α", "Costo T1", "Costo T2", "Gana"],
               ["0.001", "0.10 + 0.020 = 0.120", "0.13 + 0.005 = 0.135", "T1 (el grande)"],
               ["0.005", "0.10 + 0.100 = 0.200", "0.13 + 0.025 = 0.155", "T2 (el chico)"]],
              [20 * mm, 50 * mm, 50 * mm, CONTENT_W - 120 * mm - 18]),
        P("Empatan cuando 0.10 + 20α = 0.13 + 5α → α = 0.002. Con α mayor, conviene el árbol podado. El α final se elige con validación "
          "cruzada, no a ojo.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte C del notebook y los Bloques 5 y 6 del cuaderno",
                   [P("La pregunta de cada hoja es: ¿reduce el error en más de α? Si no, se recorta.", "cell")]))

    s.append(h("6", "Importancia de variables: impureza vs permutación"))
    s += para("La importancia «por impureza» (<font name='Mono'>feature_importances_</font>) suma cuánto redujo la impureza cada variable en "
              "todos sus cortes. Tiene un sesgo: las variables con muchos valores distintos tienen más umbrales para probar y, por azar, "
              "alguno reduce la impureza. La importancia <b>por permutación</b> desordena una variable en datos no vistos y mide cuánto cae "
              "el AUC: si el modelo no la necesitaba, casi no cae.")
    s.append(ejemplo("leer una importancia por permutación", [
        pasos([["Variable", "AUC original", "AUC con la variable desordenada", "Importancia"],
               ["reclamos", "0.900", "0.865", "0.035"],
               ["una variable ruidosa", "0.900", "0.899", "0.001"]],
              [36 * mm, 30 * mm, 60 * mm, CONTENT_W - 126 * mm - 18]),
        P("En la clase, clv (2.654 valores distintos) parecía importante por impureza (0.109) pero no por permutación (0.014). estado "
          "(binaria) era baja por impureza (0.045) y casi igual que clv por permutación (0.019).", "cell")]))

    s.append(h("7", "Fortalezas, límites y el bosque"))
    s.append(table([["Fortalezas", "Límites"],
                    ["Se dibuja y se explica a cualquiera", "Inestable: pocas observaciones distintas cambian el árbol"],
                    ["Captura no linealidad e interacciones solas", "Solo corta perpendicular a los ejes"],
                    ["No hay que escalar: invariante a log, raíz…", "Predice constantes por región: no extrapola"],
                    ["Mezcla variables numéricas y categóricas", "Sobreajusta con facilidad"],
                    ["Selecciona variables mientras entrena", "Un árbol solo suele rendir menos que una logística"]],
                   [CONTENT_W / 2, CONTENT_W / 2]))
    s.append(Spacer(1, 4))
    s.append(ejemplo("por qué promediar árboles reduce el error", [
        P("Si cada árbol tiene varianza σ² = 1 y la correlación entre árboles es ρ, la varianza del promedio de B árboles es "
          "ρσ² + (1 − ρ)σ²/B.", "cell"),
        pasos([["ρ", "B", "Varianza del promedio"],
               ["0.3", "10", "0.3 + 0.7/10 = 0.37"], ["0.3", "100", "0.3 + 0.7/100 = 0.307"], ["0", "100", "1/100 = 0.01"]],
              [20 * mm, 20 * mm, CONTENT_W - 40 * mm - 18]),
        P("Más árboles ayudan, pero el piso es ρσ². Por eso el bosque además <b>decorrelaciona</b> los árboles: cada corte mira solo "
          "√12 ≈ 3 variables al azar. Es el tema de la Clase 35.", "cell")]))

    s.append(PageBreak())
    s.append(h("8", "Chuleta de código"))
    s.append(table([
        ["Quiero…", "Código"],
        ["Entrenar un árbol", "DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0).fit(X, y)"],
        ["Ver las reglas", "export_text(arbol, feature_names=list(X.columns))"],
        ["Dibujarlo", "plot_tree(arbol, feature_names=X.columns, filled=True)"],
        ["Hoja de cada cliente", "arbol.apply(X)"],
        ["Curva de profundidad", "cross_validate(..., return_train_score=True) en un loop de max_depth"],
        ["Alphas candidatos", "arbol.cost_complexity_pruning_path(X, y).ccp_alphas"],
        ["Elegir α", "GridSearchCV(DecisionTreeClassifier(), {'ccp_alpha': alphas}, cv=skf, scoring='roc_auc')"],
        ["Importancia sesgada", "modelo.feature_importances_"],
        ["Importancia correcta", "permutation_importance(modelo, X_test, y_test, n_repeats=20, scoring='roc_auc')"],
        ["Bosque", "RandomForestClassifier(n_estimators=300, max_features='sqrt', n_jobs=-1)"],
    ], [42 * mm, CONTENT_W - 42 * mm]))
    s.append(Spacer(1, 8))
    s.append(h("9", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Dejar el árbol sin límite", "Memoriza (224 hojas, AUC 0.83)", "Curva de profundidad + poda"],
        ["Hojas de muy pocos clientes", "Tasas de ruido", "min_samples_leaf con criterio"],
        ["Elegir α a ojo", "Puede sobrepodar o quedarse corto", "GridSearchCV con validación cruzada"],
        ["Confiar en feature_importances_", "Premia la cardinalidad", "Permutación sobre el test"],
        ["Escalar variables para un árbol", "No cambia nada", "Ahorra ese paso"],
        ["Mirar solo el AUC promedio", "Puede esconder mucho desvío", "Reportar promedio y desvío"],
        ["Elegir árbol cuando solo importa predecir", "Un bosque predice mucho mejor", "Preguntar para qué es el modelo"],
    ], [50 * mm, 52 * mm, CONTENT_W - 102 * mm], header_color=RED))
    s.append(Spacer(1, 8))
    s.append(recuerda(["Orden sugerido: lee estos apuntes → Bloques 1–4 del cuaderno → Partes A y B del notebook → Bloques 5–6 → "
                       "Parte C → Bloques 7–8 → Partes D y E → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/C34_Apuntes_ArbolesDecision.pdf", s, "Apuntes · Clase 34 · Árboles de decisión")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#16A34A">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 3)]
    flow.append(numbered([("", t) for t in items], C34))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 34 · Árboles de decisión",
                  "Para resolver en papel, con calculadora (teclas ln y log) · soluciones paso a paso al final", C34))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Para log<sub>2</sub> usa log<sub>2</sub>(x) = ln(x) / ln(2) = ln(x) / 0.693.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · "
                           "<font name='Sym'>★★★</font> integra varios conceptos.",
                           "Los Bloques 1 y 4 usan el árbol y los números de la guía (página 1)."], "cell", C34)], bg=SOFT, border=C34))
    s.append(Spacer(1, 4))
    s += bloque(1, "Recorrer el árbol de la clase", "★", "Usa el árbol de profundidad 3 de la guía.", [
        "Cliente A: no reclamó, 500 segundos, 40 meses de antigüedad. ¿En qué hoja cae y cuál es su probabilidad de fuga?",
        "Cliente B: reclamó, 30 llamadas, 2 fallas. ¿Hoja y probabilidad?",
        "Cliente C: no reclamó, 3.000 segundos, CLV 500. ¿Hoja y probabilidad?",
        "Calcula el lift de la hoja del cliente B (tasa general 15.7%) y escribe su regla en una frase de negocio.",
    ])
    s += bloque(2, "Impureza a mano", "★", None, [
        "Calcula Gini y entropía de un nodo con 30 clientes, de los cuales 6 se fugan.",
        "Calcula Gini y entropía de un nodo con 20 clientes, 15 de clase A y 5 de clase B.",
        "¿Cuánto valen Gini y entropía en un nodo puro? ¿Y en uno 50/50?",
    ])
    s += bloque(3, "Elegir el corte", "★★", "Padre: 200 clientes, 40 se fugan. Corte A: izquierda 80 clientes (32 fugan), derecha 120 (8 fugan). "
                "Corte B: izquierda 100 (20 fugan), derecha 100 (20 fugan).", [
        "Calcula el Gini del padre.",
        "Calcula el Gini de cada hijo, el Gini ponderado y la ganancia de los cortes A y B.",
        "¿Cuál elige el árbol? ¿Qué tienen de especial los hijos del corte B?",
        "<font name='Sym'>★★★</font> Repite el corte A con entropía en vez de Gini. ¿Cambia la elección?",
    ])
    s += bloque(4, "El primer corte real", "★★", "Raíz: 3.150 clientes, 495 fugan. Reclamó: 278 clientes, 210 fugan. No reclamó: 2.872, 285 fugan.", [
        "Calcula el Gini de la raíz y de cada hijo.",
        "Calcula el Gini ponderado y la ganancia.",
        "La diapositiva 13 dice que la ganancia es ~0.097. ¿Está bien? Justifica.",
    ])
    s += bloque(5, "Sobreajuste", "★★", "AUC (entrenamiento / validación): prof. 3 → 0.855 / 0.850 · prof. 6 → 0.940 / 0.911 · prof. 10 → 0.985 / 0.882 · "
                "sin límite → 1.000 / 0.827.", [
        "¿Qué profundidad eliges y por qué?",
        "Calcula la brecha entre entrenamiento y validación para cada profundidad. ¿Qué te dice?",
        "Con el criterio del profesor (diferencia máxima de unos 2 puntos), ¿qué profundidades aceptarías?",
    ])
    s += bloque(6, "Poda por costo-complejidad", "★★", "T1: error 0.08 con 30 hojas · T2: error 0.12 con 6 hojas.", [
        "Calcula el costo de T1 y T2 con α = 0.001 y con α = 0.004. ¿Cuál gana en cada caso?",
        "¿Para qué α empatan?",
        "Si GridSearchCV elige α = 0.0013 con 36 hojas y la tabla de la clase muestra α = 0.002 con 15 hojas y menor AUC, ¿cuál "
        "elegirías para una presentación a gerencia? Justifica.",
    ])
    s += bloque(7, "Importancia de variables", "★★", None, [
        "Un modelo tiene AUC 0.92 en el test. Al desordenar «segundos» baja a 0.89 y al desordenar «edad» baja a 0.915. Calcula ambas "
        "importancias por permutación.",
        "En la clase, clv tiene importancia por impureza 0.109 y por permutación 0.014. Explica la diferencia usando sus 2.654 valores distintos.",
        "¿Por qué la permutación se mide sobre el test y no sobre el entrenamiento?",
    ])
    s += bloque(8, "Invariancia y bosque", "★★★", None, [
        "El árbol corta en segundos ≤ 934. Si reemplazas segundos por log(1 + segundos), ¿cuál sería el nuevo umbral? ¿Cambia algún cliente de lado?",
        "Con σ² = 1 y ρ = 0.4, calcula la varianza del promedio de 10 y de 200 árboles. ¿Cuál es el piso con infinitos árboles?",
        "Con 12 variables y max_features = 'sqrt', ¿cuántas variables mira cada corte? ¿Para qué sirve eso?",
    ])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "reclamos ≤ 0.5 sí → segundos ≤ 934 sí → antigüedad ≤ 28.5 no → hoja de 340 clientes: 43%."],
        ["2", "reclamos ≤ 0.5 no → frec_llamadas ≤ 57.5 sí → numero_fallas ≤ 0.5 no → hoja de 149 clientes: 94%."],
        ["3", "reclamos ≤ 0.5 sí → segundos ≤ 934 no → clv ≤ 61.3 no → hoja de 2.359 clientes: 4.5%."],
        ["4", "Lift = 0.94 / 0.157 = 6.0. Regla: «clientes que reclamaron, usan poco el servicio (57 llamadas o menos) y tuvieron "
              "fallas técnicas: 94% se fuga». Prioridad 1 para retención."]])
    s += sol("Bloque 2", [
        ["1", "p = 6/30 = 0.2 → Gini = 1 − 0.04 − 0.64 = 0.32 · entropía = −0.2·log<sub>2</sub>0.2 − 0.8·log<sub>2</sub>0.8 = 0.464 + 0.258 = 0.722."],
        ["2", "p = 0.75 → Gini = 1 − 0.5625 − 0.0625 = 0.375 · entropía = 0.311 + 0.5 = 0.811."],
        ["3", "Puro: 0 y 0 · 50/50: Gini 0.5 y entropía 1 (los máximos)."]])
    s += sol("Bloque 3", [
        ["1", "p = 40/200 = 0.2 → Gini padre = 0.32."],
        ["2", "A: izquierda p = 0.4 → 0.48 · derecha p = 8/120 = 0.067 → 0.124 · ponderado 0.4·0.48 + 0.6·0.124 = 0.267 · ganancia 0.053. "
              "B: los dos hijos con p = 0.2 → 0.32 · ponderado 0.32 · ganancia 0."],
        ["3", "Elige A. Los hijos de B tienen la misma proporción que el padre: no separan nada."],
        ["4", "Entropía padre 0.722 · izquierda 0.971 · derecha 0.353 · ponderada 0.4·0.971 + 0.6·0.353 = 0.600 · ganancia 0.122. "
              "B sigue con ganancia 0: la elección no cambia."]])
    s.append(PageBreak())
    s += sol("Bloque 4", [
        ["1", "Raíz: p = 0.157 → 0.265 · reclamó: p = 210/278 = 0.755 → 0.370 · no reclamó: p = 285/2.872 = 0.099 → 0.179."],
        ["2", "Ponderado = (278·0.370 + 2.872·0.179) / 3.150 = 0.196 · ganancia = 0.265 − 0.196 = 0.069."],
        ["3", "No: con esos datos (incluso con los 211/284 de la diapositiva) la ganancia es 0.070, no 0.097. La propia diapositiva 12 "
              "muestra 0.0693. La nota de respuesta tiene un error."]])
    s += sol("Bloque 5", [
        ["1", "Profundidad 6: tiene el mayor AUC de validación (0.911)."],
        ["2", "Brechas: prof. 3 → 0.005 · 6 → 0.029 · 10 → 0.103 · sin límite → 0.173. La brecha crece con la profundidad: más sobreajuste."],
        ["3", "Solo la profundidad 3 (0.5 puntos). La 6 tiene casi 3 puntos: hay que decidir si esa ganancia de AUC justifica el sobreajuste, "
              "o podar."]])
    s += sol("Bloque 6", [
        ["1", "α = 0.001: T1 = 0.08 + 0.030 = 0.110 · T2 = 0.12 + 0.006 = 0.126 → gana T1. α = 0.004: T1 = 0.08 + 0.12 = 0.20 · "
              "T2 = 0.12 + 0.024 = 0.144 → gana T2."],
        ["2", "0.08 + 30α = 0.12 + 6α → 24α = 0.04 → α = 0.00167."],
        ["3", "Para gerencia, probablemente el de 15 hojas: se entiende en una lámina y pierde poco AUC (0.883 vs 0.909). Si el objetivo es "
              "predecir, el de 36 hojas. Depende de para qué es el modelo."]])
    s += sol("Bloque 7", [
        ["1", "segundos: 0.92 − 0.89 = 0.03 · edad: 0.92 − 0.915 = 0.005. segundos importa seis veces más."],
        ["2", "Con 2.654 valores, clv ofrece muchísimos umbrales; por azar alguno reduce la impureza, y la importancia por impureza lo premia. "
              "Al desordenarla en datos nuevos, el AUC casi no cae: el modelo no la necesitaba tanto."],
        ["3", "Porque sobre el entrenamiento el modelo ya memorizó: desordenar una variable que memorizó hace caer el AUC aunque no generalice."]])
    s += sol("Bloque 8", [
        ["1", "Nuevo umbral: log(1 + 934) = 6.84. Ningún cliente cambia de lado: la transformación es monótona y conserva el orden."],
        ["2", "B = 10: 0.4 + 0.6/10 = 0.46 · B = 200: 0.4 + 0.6/200 = 0.403 · piso: ρσ² = 0.4."],
        ["3", "√12 ≈ 3 variables por corte. Así los árboles se parecen menos entre sí (baja ρ) y el promedio reduce más la varianza."]])
    build("apuntes/C34_Ejercicios_Cuaderno_ArbolesDecision.pdf", s, "Ejercicios para el cuaderno · Clase 34")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
