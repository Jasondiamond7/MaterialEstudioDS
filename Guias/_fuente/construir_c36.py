"""Clase 36 · Boosting: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_glm import code, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C36 = colors.HexColor("#EA580C")
C36_BG = colors.HexColor("#FFF7ED")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + notebook Python del profesor"


def topic_page(s, titulo, idea, conceptos, figura, pie, formulas, py, salida, cuidado, fig_w=0.97):
    s.append(band(f"Clase 36 · {titulo}", FUENTE, C36))
    s.append(Spacer(1, 7))
    s.append(box([P(idea, "big")], bg=SOFT, border=C36))
    s.append(Spacer(1, 6))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C36))
    s.append(Spacer(1, 4))
    if figura:
        s.append(img(figura, width=CONTENT_W * fig_w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 5))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF),
            Spacer(1, 5), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C36), Spacer(1, 5),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER,
                 width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 36 · Boosting",
                  "Profesor Bastian Galasso · Etapa 7 del roadmap · continúa Árboles (C34) y Random Forest (C35)", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Un bosque entrena muchos árboles <b>independientes</b> y los promedia. Boosting hace lo contrario: entrena los "
                    "árboles <b>uno tras otro</b> y cada uno se concentra en lo que los anteriores no lograron explicar. "
                    "«Si un modelo se equivocó en una parte de los datos, queremos que el siguiente aprenda de ese error y no "
                    "tropiece dos veces con la misma piedra.» El resultado es un modelo <b>aditivo</b>: la suma ponderada de "
                    "todos los árboles.", "body")],
                 bg=C36_BG, border=C36))
    s.append(Spacer(1, 6))
    s.append(P("Boosting en una imagen: ajustar residuos", "h2"))
    s.append(img("c36_secuencial.png", width=CONTENT_W))
    s.append(Spacer(1, 4))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Árboles de decisión: profundidad, hojas, sobreajuste (Clase 34)",
                     "Bagging y Random Forest: por qué promediar reduce la varianza (Clase 35)",
                     "Validación cruzada, AUC y RMSE (Clases 30–33)",
                     "Pipeline con imputación y one-hot (preprocesamiento dentro del pipeline)"], "cell", C36)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Intuición</b> → <b>AdaBoost</b> (pesos) → <b>Gradient Boosting</b> (gradientes) → "
                      "<b>XGBoost y LightGBM</b> → comparar y elegir (CatBoost solo se menciona)",
                      "<b>Heart</b> (303 pacientes, clasificación AHD) y <b>Ames Housing</b> (2.930 casas, regresión del precio): "
                      "los mismos datos de la Clase 35",
                      "Pregunta guía: ¿cómo equilibramos capacidad predictiva, velocidad, categóricas e interpretabilidad?"],
                     "cell", C36)]
    s.append(two_cols(left, right))
    s.append(PageBreak())

    topic_page(
        s, "Bagging vs boosting y el ciclo general",
        "Los dos son ensambles de árboles, pero resuelven problemas distintos. Bagging promedia modelos independientes para bajar "
        "la <b>varianza</b>; boosting los encadena para bajar el <b>sesgo</b> y el error.",
        [("Paralelo vs secuencial", "En el bosque cada árbol se entrena solo, con su muestra, «sin importarle el de al lado». En boosting el árbol 2 necesita el feedback del árbol 1: no se puede paralelizar entre árboles."),
         ("Mismo dataset completo", "Cada etapa ve todos los datos de entrenamiento. El error del árbol anterior viaja como pesos (AdaBoost) o como residuos (Gradient Boosting); remuestrear lo destruiría."),
         ("Suma, no promedio", "El modelo final es aditivo: F(x) = F₀ + η·h₁(x) + η·h₂(x) + … La contribución de cada árbol se pondera; en AdaBoost, según su calidad."),
         ("Ciclo de 5 pasos", "1) Inicializar con una predicción base · 2) medir el error (pérdida o casos difíciles) · 3) ajustar un nuevo aprendiz · 4) sumar su contribución · 5) detener tras M etapas o cuando la validación deja de mejorar."),
         ("Lo que se espera del árbol 2", "No que cometa menos errores en todo, sino que <b>no cometa los mismos</b>: se concentra en lo difícil y descuida un poco lo que ya aprendió bien.")],
        "c36_bagging_boosting.png",
        "Izquierda: el esquema de cada familia. Derecha: error en validación de Ames al agregar árboles. El bosque se estanca; boosting sigue mejorando.",
        ["F<sub>M</sub>(x) = F<sub>0</sub>(x) + η · Σ h<sub>m</sub>(x), m = 1…M",
         "bagging → paralelo, promedio o voto, ↓ varianza",
         "boosting → secuencia, suma ponderada, ↓ sesgo y error",
         "tiempo con 1.000 árboles y 10 núcleos: bagging ≈ 100 · boosting ≈ 1.000"],
        ["RandomForestRegressor(n_estimators=300,",
         "    n_jobs=-1)          # paraleliza el bosque",
         "GradientBoostingRegressor(n_estimators=300,",
         "    learning_rate=0.1, max_depth=3)",
         "list(gb.staged_predict(X_val))  # F₁, F₂, …"],
        ["Árbol solo (profundidad 3): RMSE de validación 41.2k USD",
         "Bosque de 300 árboles: 23.1k, ya estancado hacia los 100 árboles",
         "Gradient Boosting de 300 árboles (η = 0.1): 21.8k, mejor ronda ≈ 250",
         "Con un solo árbol el boosting va peor (69.0k) a propósito: da pasos chicos"],
        ["Por ser secuencial, boosting es más sensible a <b>ruido</b>, hiperparámetros y <b>filtración</b> de información: una etiqueta mal puesta se convierte en el caso «más difícil» y el modelo se obsesiona con ella."])

    topic_page(
        s, "AdaBoost: enfocar los casos difíciles",
        "AdaBoost sube el peso de las observaciones mal clasificadas, de modo que el siguiente árbol (normalmente un <i>stump</i> "
        "de un solo corte) las tome más en serio, y combina los árboles con un voto ponderado por su calidad.",
        [("Pesos iguales al inicio", "w<sub>i</sub> = 1/n. El árbol 1 se entrena con todos los pesos iguales."),
         ("Error del modelo ε", "Suma de los pesos de los casos mal clasificados, entre la suma de todos los pesos: es un número entre 0 y 1. Perfecto: 0; errar todo: 1."),
         ("Peso del modelo α", "α = ½ ln((1 − ε)/ε). Con ε = 0.5 (azar) α = 0: no aporta. Con ε &gt; 0.5, α &lt; 0. «Mayor error del árbol, menor influencia.»"),
         ("Reponderar", "Los casos mal clasificados multiplican su peso por e<sup>α</sup>; los bien clasificados, por e<sup>−α</sup>. Luego se normaliza para que sumen 1."),
         ("Voto final", "H(x) = signo de Σ α<sub>m</sub>·h<sub>m</sub>(x), con clases −1/+1. Cada stump corrige una región; el voto produce una frontera «escalonada».")],
        "c36_adaboost.png",
        "Con 70 pacientes de Heart y dos variables (edad y frecuencia cardíaca máxima). Cada ronda agranda los puntos mal clasificados.",
        ["ε<sub>m</sub> = Σ w<sub>i</sub>·1[h<sub>m</sub>(x<sub>i</sub>) ≠ y<sub>i</sub>] / Σ w<sub>i</sub>",
         "α<sub>m</sub> = ½ · ln((1 − ε<sub>m</sub>) / ε<sub>m</sub>)",
         "w<sub>i</sub> ← w<sub>i</sub> · exp(−α<sub>m</sub> · y<sub>i</sub> · h<sub>m</sub>(x<sub>i</sub>)), luego normalizar",
         "H(x) = sign( Σ α<sub>m</sub> · h<sub>m</sub>(x) )"],
        ["AdaBoostClassifier(n_estimators=250,",
         "    learning_rate=0.05, random_state=42)",
         "# estimador base por defecto: stump (prof. 1)",
         "AdaBoostClassifier(estimator=",
         "    DecisionTreeClassifier(max_depth=2))"],
        ["Rondas 1 a 3: ε = 0.271 → 0.338 → 0.453; α = 0.49 → 0.34 → 0.09",
         "A medida que quedan solo casos difíciles, ε se acerca a 0.5 y α a 0",
         "Heart (CV 5×3): AdaBoost 0.912 de AUC, el mejor de los cinco, pero a menos de un desvío (≈ 0.04) del último",
         "Ames: AdaBoost es el peor (RMSE 36.2k vs 24.4k de XGBoost)"],
        ["<b>AdaBoost es «Adaptive Boosting»</b>; en clase se dijo «additive». Es sensible a ruido y etiquetas incorrectas: "
         "úsalo como línea base fuerte con datos limpios."])

    topic_page(
        s, "Gradient Boosting y las tres perillas",
        "En vez de pesar casos, Gradient Boosting ajusta cada árbol a la dirección que más reduce la pérdida. Con error "
        "cuadrático, esa dirección son exactamente los <b>residuos</b> y − F. Tres perillas controlan todo: pérdida, tasa η y "
        "complejidad del árbol.",
        [("La pérdida define el error", "Cuadrática, absoluta, Huber o logística: es la forma en que un árbol le dice al siguiente qué corregir. Otra pérdida, otro resultado."),
         ("Gradiente negativo = residuo", "El nuevo árbol h<sub>m</sub> predice −∂L/∂F. Para L = ½(y − F)² eso es y − F; para otras pérdidas ya no es el residuo a secas."),
         ("Tasa de aprendizaje η", "Cuánto de cada árbol se suma. Menor η suele requerir más árboles: «estudiar un poco cada día» rinde más que de golpe la noche anterior."),
         ("η y n_estimators van juntos", "Más árboles con η bajo da modelos más estables, pero más caros. Se afinan en pareja, nunca solos."),
         ("Complejidad del árbol", "max_depth, número de hojas, muestras mínimas por hoja: limitan lo que aprende cada árbol (los árboles de boosting suelen ser chicos, de profundidad 2 a 8).")],
        "c36_lr_nest.png",
        "Gradient Boosting con profundidad 3 sobre Ames, validación separada del test. Los puntos marcan el número de árboles óptimo para cada η.",
        ["F<sub>m</sub> = F<sub>m−1</sub> + η · h<sub>m</sub>",
         "r<sub>im</sub> = −∂L/∂F evaluado en F<sub>m−1</sub> (cuadrática: y − F)",
         "caso ideal: residuo tras m pasos = r<sub>0</sub> · (1 − η)<sup>m</sup>",
         "con η = 1 cada árbol corrige todo su residuo (¡y su ruido!)"],
        ["GradientBoostingRegressor(",
         "    n_estimators=500, learning_rate=0.05,",
         "    max_depth=2, loss='squared_error')",
         "# loss: 'huber', 'absolute_error' (regresión)",
         "# clasificación: log_loss"],
        ["η = 1.0: mejor a los 9 árboles, RMSE 30.3k; luego empeora",
         "η = 0.3: 462 árboles, 22.5k",
         "η = 0.1: 587 árboles, 21.5k",
         "η = 0.03: ≈ 790 árboles (casi el tope de 800), 21.5k: igual de bueno, mucho más lento"],
        ["Con η = 0.03 y pocos árboles el modelo queda <b>subajustado</b>. «A menor η, mejor» solo vale si das árboles suficientes. "
         "Y los residuos son el gradiente solo con pérdida cuadrática."])

    topic_page(
        s, "XGBoost y LightGBM",
        "Las implementaciones modernas son Gradient Boosting con ingeniería: XGBoost agrega regularización y velocidad; LightGBM "
        "acelera la búsqueda de cortes con histogramas y crece el árbol hoja por hoja.",
        [("XGBoost: objetivo regularizado", "Pérdida de entrenamiento + penalización por complejidad (premia la simplicidad). Usa gradientes de 1.er y 2.º orden, submuestreo de filas y columnas, manejo eficiente de dispersión y de nulos. Está escrito en C++ y se llama desde Python."),
         ("LightGBM: histogramas", "Agrupa los valores continuos en bins y busca los cortes ahí: menos memoria y tiempo. GOSS conserva gradientes grandes y muestrea los pequeños; EFB agrupa variables dispersas casi excluyentes."),
         ("Por nivel vs por hoja", "XGBoost suele expandir un nivel completo. LightGBM expande solo la hoja con <b>mayor ganancia</b>: árbol asimétrico, mucho más eficiente con muchos datos."),
         ("Riesgo del leaf-wise", "Puede sobreajustar datos pequeños si no se limita (num_leaves, min_data_in_leaf, max_depth). LightGBM está pensado para muchos datos."),
         ("Early stopping", "Se monitorea el error en un conjunto de validación y se detiene cuando deja de mejorar durante k rondas: elige n_estimators automáticamente.")],
        "c36_xgb_lgbm.png",
        "Izquierda: dos formas de hacer crecer un árbol. Derecha: XGBoost con early stopping en Ames; el entrenamiento sigue bajando, la validación se aplana.",
        ["Obj = Σ L(y<sub>i</sub>, ŷ<sub>i</sub>) + Σ Ω(f<sub>k</sub>)",
         "Ω(f) = γ·T + ½·λ·Σ w<sub>j</sub>²   (T hojas, λ = L2)",
         "hojas máximas con profundidad d: 2<sup>d</sup>",
         "LightGBM: num_leaves &lt; 2<sup>max_depth</sup>"],
        ["XGBRegressor(n_estimators=1500,",
         "    learning_rate=0.05, max_depth=3,",
         "    subsample=0.8, colsample_bytree=0.8,",
         "    early_stopping_rounds=50)",
         "LGBMClassifier(num_leaves=15,",
         "    subsample=0.8, subsample_freq=1)"],
        ["XGBoost con early stopping en Ames: mejor ronda 754, se detiene en la 805 de 1.500 posibles",
         "En esa ronda: RMSE de validación 21.0k y de entrenamiento 9.4k",
         "Tiempos con 350 árboles en Ames: XGBoost 0.5 s, LightGBM 0.6 s, Gradient Boosting de sklearn 3.2 s, AdaBoost 6.3 s",
         "LightGBM (31 hojas) en Ames: RMSE de entrenamiento 8.0k vs test 27.0k: sobreajuste"],
        ["El early stopping necesita su propio conjunto de validación, <b>nunca el test</b>. En LightGBM, <font name='Mono'>subsample</font> no hace nada "
         "mientras <font name='Mono'>subsample_freq</font> sea 0 (el valor por defecto)."])

    topic_page(
        s, "Comparar y elegir",
        "No hay un ganador universal. La mejor opción depende del tamaño de los datos, las categóricas, el presupuesto de cómputo "
        "y, sobre todo, de la validación: mismo pipeline, misma partición, métrica adecuada.",
        [("Qué domina tu problema", "Línea base simple: AdaBoost · control didáctico: Gradient Boosting · rendimiento general: XGBoost · gran escala: LightGBM · muchísimas categóricas: CatBoost."),
         ("Categóricas", "AdaBoost, Gradient Boosting y (normalmente) XGBoost piden codificarlas; LightGBM las admite de forma nativa; CatBoost es el especialista."),
         ("«A problema simple, modelo simple»", "Con pocos datos (Heart, 303 filas) los modelos simples compiten con los avanzados. Con más datos (Ames, 2.930) XGBoost y los de gradiente despegan."),
         ("Mide bien", "Clasificación: AUC, F1, recall (¿cuánto cuesta un falso negativo?). Regresión: RMSE, MAE y R². Compara train y test para ver el sobreajuste, y reserva el test para el final."),
         ("Dónde se usan", "Riesgo (impago, fraude, churn), demanda, salud, marketing, operaciones, precios: datos tabulares con no linealidades e interacciones.")],
        "c36_comparacion.png",
        "Los cinco modelos con la configuración del notebook del profesor (más Random Forest como referencia). Heart: validación cruzada 5×3. Ames: partición de test.",
        ["AUC con CV: reportar media ± desvío",
         "RMSE = √( promedio de (ŷ − y)² ) · en la misma unidad que y",
         "1 paciente de 76 en test = 1.3 puntos de exactitud",
         "ganador = diferencia mayor que el desvío entre rondas"],
        ["Pipeline([('prep', ColumnTransformer(...)),",
         "          ('modelo', XGBClassifier(...))])",
         "cross_val_score(pipe, X, y, cv=cv,",
         "    scoring='roc_auc')",
         "scoring='neg_root_mean_squared_error'"],
        ["Heart, AUC (CV 5×3): AdaBoost 0.912 · RF 0.906 · XGBoost 0.894 · LightGBM 0.892 · GB 0.891; desvíos ≈ 0.04",
         "Ames, RMSE de test: XGBoost 24.4k · Gradient Boosting 26.8k · LightGBM 27.0k · Random Forest 27.5k · AdaBoost 36.2k",
         "Cuanto más cerca de 0 el RMSE, mejor; 25k significa errar ≈ 25 mil dólares por casa"],
        ["Una sola partición no declara un ganador: en Heart las diferencias entre modelos caben dentro del desvío. Para una conclusión "
         "más sólida: validación cruzada anidada o varias semillas."])

    # ------------------------------------------------------------ chuleta
    s.append(band("Chuleta, hiperparámetros y checklist", "Lo que vas a ajustar en cualquier modelo de boosting", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["Función", "Parámetros", "Efecto si lo subes", "Valor de partida"],
                    ["Capacidad", "n_estimators (sklearn, XGBoost, LightGBM) · iterations (CatBoost)", "más etapas: menos sesgo, más costo y más riesgo de sobreajuste", "100–1.000 con early stopping"],
                    ["Paso", "learning_rate (η)", "más rápido, pero más ruidoso y menos estable", "0.03–0.1"],
                    ["Árbol", "max_depth · num_leaves (LightGBM)", "más complejidad e interacciones", "profundidad 2–6; hojas 15–31"],
                    ["Regularización", "min_child_weight · min_data_in_leaf (min_child_samples) · reg_lambda (L2)", "hojas más robustas y árboles más simples", "L2 = 1 (el valor típico)"],
                    ["Aleatoriedad", "subsample · colsample_bytree", "valores &lt; 1: más diversidad y menos sobreajuste", "0.7–1.0"],
                    ["Parada", "early_stopping_rounds · n_iter_no_change", "más paciencia, más rondas", "20–100"]],
                   [24 * mm, 56 * mm, 56 * mm, CONTENT_W - 136 * mm]))
    s.append(Spacer(1, 3))
    s.append(P("Función y parámetros: diapositiva 15. Los valores de partida son orientativos (ver notebook); el profesor aconseja entender cada "
               "regularización antes de meterla a un buscador automático.", "small"))
    s.append(Spacer(1, 6))
    s.append(table([["Modelo", "Sello distintivo", "Categóricas", "Velocidad", "Riesgo principal"],
                    ["AdaBoost", "Pondera casos difíciles", "Codificar", "Media", "Ruido y outliers"],
                    ["Gradient Boosting", "Gradiente de una pérdida", "Codificar", "Media-baja", "Ajuste sensible"],
                    ["XGBoost", "Regularización y sistema escalable", "Depende de la implementación", "Alta", "Complejidad del tuning"],
                    ["LightGBM", "Histogramas y leaf-wise", "Nativas", "Muy alta", "Sobreajuste en datos pequeños"],
                    ["CatBoost", "Pensado para categóricas (solo mencionado en clase)", "Nativas, es su fuerte", "—", "—"]],
                   [30 * mm, 50 * mm, 36 * mm, 22 * mm, CONTENT_W - 138 * mm]))
    s.append(Spacer(1, 6))
    s.append(P("Checklist para entrenar un modelo de boosting", "h2"))
    checks = [("Parte train/test (estratificado si es clasificación)", "en regresión no se estratifica"),
              ("Preprocesa dentro del Pipeline", "imputación y one-hot se aprenden solo con el train"),
              ("Empieza con una línea base", "AdaBoost o un bosque, para saber qué hay que superar"),
              ("Fija η bajo y deja que early stopping elija n", "con validación aparte del test"),
              ("Limita la complejidad del árbol", "profundidad, hojas mínimas; cuidado con LightGBM en datos chicos"),
              ("Compara train vs test", "una brecha grande es sobreajuste"),
              ("Incluye tu configuración base en la búsqueda", "si no, la búsqueda puede elegir algo peor"),
              ("Reporta media ± desvío y compara bajo la misma partición", "una sola partición no basta")]
    s.append(table([["", "Paso", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 80 * mm, CONTENT_W - 88 * mm]))
    s.append(PageBreak())

    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales, preguntas clave y cuándo pasar a la siguiente clase", NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Diapositivas y clase (verificados)</b>", "cellb"),
                  bullets(["<b>«AdaBoost = additive boosting»</b> (clase): el nombre viene de <i>Adaptive Boosting</i> (Freund y Schapire, 1997).",
                           "<b>Diapositiva 16:</b> el título dice «cinco modelos», pero la tabla trae cuatro (falta CatBoost, que sí aparece en la diapositiva 18). "
                           "La numeración impresa salta del 13 al 15: el archivo tiene 19 láminas y falta la «14/20».",
                           "<b>«Gradient Boosting se detiene con tolerancia ≈ 0.001»</b> (clase): la teoría de optimización lo dice, pero "
                           "<font name='Mono'>GradientBoosting*</font> de sklearn entrena exactamente n_estimators árboles salvo que actives "
                           "<font name='Mono'>n_iter_no_change</font> (su tol por defecto es 1e-4).",
                           "<b>«Boosting trabaja sobre y − F»</b>: eso es el gradiente negativo solo con pérdida cuadrática. Y «menor η, mejor» solo si aumentas los árboles."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Notebook del profesor (Ejemplo Boosting.ipynb)</b>", "cellb"),
                  bullets(["<font name='Mono'>LGBM…(subsample=0.8)</font> sin <font name='Mono'>subsample_freq</font> ignora el submuestreo: "
                           "en el notebook guiado se comprueba. XGBoost sí lo aplica.",
                           "El GB de Ames usa <font name='Mono'>loss='huber'</font>, pero se evalúa y optimiza con RMSE (pérdida cuadrática): objetivo y métrica no coinciden.",
                           "La «importancia» compara cosas distintas: en XGBoost es por <b>ganancia</b>; en LightGBM, por <b>número de cortes</b> (<i>split</i>).",
                           "En clase se dijo que se eliminaban Latitude y Longitude de Ames; el notebook entregado no lo hace.",
                           "Las dos comparaciones finales usan una sola partición (76 pacientes de test): que AdaBoost «gane» por un paciente no es evidencia."],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("«bugging» (bagging), «Roundoff Forest» (Random Forest), «Caldus» (CatBoost), «XzBus/Xtbus» (XGBoost), «Light DBM» (LightGBM), "
                    "«KDNN» (KNN), «Cycle Learn» (scikit-learn), «Nameless Housing» (Ames Housing), «KERX» (Heart). Bucles de Whisper en las líneas "
                    "216–244 y 1337–1341; comentario ajeno a la clase en las líneas 508–512.", "cell")], bg=SOFT, border=GRAY if False else NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Preguntas de repaso", "h2"))
    preg = [("¿Qué cambia entre AdaBoost y Gradient Boosting?", "AdaBoost pasa el error como pesos de los casos; Gradient Boosting como el gradiente de la pérdida (residuos si es cuadrática)."),
            ("¿Por qué η y n_estimators interactúan?", "Menor η da pasos chicos, así que hacen falta más árboles para llegar al mismo punto."),
            ("¿Cuándo priorizarías LightGBM?", "Muchos datos y variables, o presupuesto de cómputo ajustado; con datos pequeños, con cuidado."),
            ("ε = 0.5 en una ronda de AdaBoost, ¿qué α recibe?", "α = ½ ln(1) = 0: el árbol no aporta al voto."),
            ]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 66 * mm, CONTENT_W - 73 * mm]))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"), Spacer(1, 2),
                  bullets(["Hago a mano una ronda de AdaBoost: ε, α y nuevos pesos",
                           "Hago a mano una etapa de Gradient Boosting con error cuadrático",
                           "Explico por qué η y el número de árboles se ajustan juntos",
                           "Uso early stopping con un conjunto de validación propio",
                           "Distingo XGBoost (por nivel, regularizado) de LightGBM (por hoja, histogramas)",
                           "Resuelvo los bloques 1 a 9 del cuaderno con al menos 80% de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C36_Guia_Boosting.pdf", s, "Clase 36 · Boosting")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#EA580C">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 36 · Boosting", "Para leer antes de los ejercicios · lectura de ~45 min", C36))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("Secuencia", "Los árboles se entrenan uno tras otro; cada uno recibe información del error del anterior."),
                      ("Aditivo", "El modelo final es una suma ponderada: F = F₀ + η·h₁ + η·h₂ + …"),
                      ("AdaBoost", "Sube el peso de los casos mal clasificados y vota con peso α = ½ ln((1 − ε)/ε)."),
                      ("Gradient Boosting", "Cada árbol predice el gradiente negativo de la pérdida (el residuo, si es cuadrática)."),
                      ("η y n_estimators", "Van juntos: menor η, más árboles. Se elige n con early stopping."),
                      ("XGBoost y LightGBM", "Gradient Boosting regularizado y rápido; LightGBM crece por hoja y con histogramas."),
                  ], C36)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C36_Boosting.ipynb</font> o el cuaderno "
               "<font name='Mono'>C36_Ejercicios_Cuaderno_Boosting.pdf</font>.", "small"))

    s.append(h("1", "Bagging y boosting: dos ideas distintas"))
    s += para("En el bosque (Clase 35) cada árbol se entrena con su propia muestra, sin importarle los demás: por eso se pueden "
              "entrenar en paralelo, y promediarlos baja la <b>varianza</b>. En boosting el árbol 2 necesita saber dónde falló el árbol 1, "
              "así que se entrena <b>después</b>, sobre los <b>mismos datos completos</b>, y sus contribuciones se <b>suman</b>. El objetivo típico "
              "es bajar el sesgo y el error.")
    s.append(ejemplo("el costo de ser secuencial", [
        P("Un modelo de 600 árboles tarda 1 s por árbol y dispones de 12 procesadores.", "cell"),
        pasos([["Familia", "Tiempo aproximado", "Razón"],
               ["Bosque", "600 / 12 = 50 s", "los árboles son independientes: se reparten"],
               ["Boosting", "600 s", "cada árbol espera al anterior"]],
              [26 * mm, 40 * mm, CONTENT_W - 66 * mm - 18]),
        P("Por eso XGBoost invierte tanto en paralelizar <i>dentro</i> de cada árbol (la búsqueda de cortes), aunque entre árboles siga siendo secuencial.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c36_bagging_boosting.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Compara las curvas de error al agregar árboles: el bosque se estanca pronto; boosting sigue bajando. Pregúntate "
                      "qué ve cada árbol y qué hace con ello.", "cell")]))

    s.append(h("2", "El modelo aditivo y el ciclo de boosting"))
    s += para("Todos los modelos de la clase siguen cinco pasos: <b>inicializar</b> con una predicción base F₀, <b>medir el error</b>, "
              "<b>ajustar</b> un nuevo aprendiz, <b>actualizar</b> el modelo sumando su contribución y <b>detener</b> tras M etapas o cuando la "
              "validación deja de mejorar. Lo que separa a AdaBoost de Gradient Boosting es solo la definición de «error».")
    s.append(ejemplo("predecir una casa con tres árboles", [
        P("F₀ = 160 (miles de USD, el promedio). Para una casa, los tres árboles predicen h₁ = +50, h₂ = +20, h₃ = −10. η = 0.3.", "cell"),
        pasos([["Etapa", "Cálculo", "Predicción F"],
               ["0", "—", "160"],
               ["1", "160 + 0.3 · 50", "175"],
               ["2", "175 + 0.3 · 20", "181"],
               ["3", "181 + 0.3 · (−10)", "178"]],
              [20 * mm, 60 * mm, CONTENT_W - 80 * mm - 18]),
        P("Cada árbol aporta solo una fracción η de lo que predice: es la «tasa de aprendizaje».", "cell")]))

    s.append(h("3", "AdaBoost: pesos, error y voto"))
    s += para("AdaBoost trabaja con clases −1/+1. Empieza con pesos iguales 1/n. Cada ronda: entrena un árbol (normalmente un <i>stump</i>) con "
              "esos pesos, calcula su error <b>ponderado</b> ε, su peso α = ½ ln((1 − ε)/ε) y actualiza los pesos: los errores se multiplican por "
              "e<sup>α</sup> y los aciertos por e<sup>−α</sup>. Después se normaliza.")
    s.append(ejemplo("una ronda con 10 observaciones", [
        P("Pesos iniciales 0.1. El stump falla en 2 observaciones: ε = 0.2.", "cell"),
        pasos([["Paso", "Cálculo", "Resultado"],
               ["α", "½ ln(0.8/0.2) = ½ ln 4", "0.693"],
               ["Factor de los errores", "e<sup>0.693</sup>", "2"],
               ["Factor de los aciertos", "e<sup>−0.693</sup>", "0.5"],
               ["Pesos sin normalizar", "errores 0.1·2 = 0.2 · aciertos 0.1·0.5 = 0.05", "suma = 2·0.2 + 8·0.05 = 0.8"],
               ["Pesos normalizados", "errores 0.2/0.8 · aciertos 0.05/0.8", "0.25 y 0.0625"]],
              [38 * mm, 70 * mm, CONTENT_W - 108 * mm - 18]),
        P("Fíjate: tras normalizar, los 2 errores concentran el 50% del peso total (2 · 0.25). Siempre ocurre: el siguiente árbol ve los errores del anterior como «la mitad del problema». "
          "El voto final: tres árboles con α = 0.9, 0.4, 0.3 votan −1, +1, +1; la suma es −0.9 + 0.7 = −0.2, así que gana la clase −1 aunque haya más votos +1.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c36_adaboost.png", width=CONTENT_W * 0.97))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook y los Bloques 2 y 3 del cuaderno",
                   [P("Ordena el cálculo siempre igual: (1) ε sumando pesos de los errores, (2) α, (3) factor de errores e<sup>α</sup> "
                      "y de aciertos e<sup>−α</sup>, (4) normalizar. Si ε &gt; 0.5, α sale negativo: el árbol se «invierte».", "cell")]))

    s.append(h("4", "Gradient Boosting: ajustar residuos"))
    s += para("Se empieza con F₀ = promedio de y (en pérdida cuadrática). En cada etapa se calculan los <b>residuos</b> r = y − F, "
              "se ajusta un árbol pequeño para predecirlos y se actualiza F = F + η·h.")
    s.append(ejemplo("una etapa con 4 observaciones", [
        P("y = [20, 14, 18, 24] para A, B, C, D. F₀ = promedio = 19. El árbol separa {A, B} de {C, D}.", "cell"),
        pasos([["Obs.", "y", "F₀", "r = y − F₀", "h (media de r por hoja)", "F₁ con η = 0.5", "y − F₁"],
               ["A", "20", "19", "+1", "−2 (media de +1, −5)", "19 + 0.5·(−2) = 18", "+2"],
               ["B", "14", "19", "−5", "−2", "18", "−4"],
               ["C", "18", "19", "−1", "+2 (media de −1, +5)", "19 + 0.5·2 = 20", "−2"],
               ["D", "24", "19", "+5", "+2", "20", "+4"]],
              [11 * mm, 9 * mm, 10 * mm, 22 * mm, 42 * mm, 36 * mm, CONTENT_W - 130 * mm - 18]),
        P("RMSE antes: √((1 + 25 + 1 + 25)/4) = √13 = 3.61. Después con η = 0.5: √((4 + 16 + 4 + 16)/4) = √10 = 3.16. "
          "Con η = 1 los nuevos residuos serían [+3, −3, −3, +3], RMSE 3.0: baja más rápido, pero «de golpe». El siguiente árbol estudiará los residuos [+2, −4, −2, +4].", "cell")]))
    s.append(Spacer(1, 4))
    s += para("<b>¿Qué es el «gradiente»?</b> La pérdida L(y, F) mide qué tan mal predice F. El gradiente negativo −∂L/∂F indica hacia dónde mover F para "
              "reducir la pérdida; el nuevo árbol aprende a predecir esa dirección. Para L = ½(y − F)² es y − F. Para la pérdida absoluta es el signo del "
              "residuo, y para Huber, el residuo recortado: robustos a valores extremos.")
    s.append(ejemplo("qué predice el siguiente árbol según la pérdida", [
        pasos([["Pérdida", "y, F", "−∂L/∂F"],
               ["Cuadrática ½(y − F)²", "y = 12, F = 9", "y − F = +3"],
               ["Absoluta |y − F|", "y = 12, F = 9", "signo(y − F) = +1"],
               ["Huber (δ = 2)", "y = 12, F = 9", "residuo recortado en ±2 → +2"],
               ["Logística (y = 0 ó 1, p = 1/(1+e<sup>−F</sup>))", "y = 1, F = 0 → p = 0.5", "y − p = +0.5"],
               ["Logística", "y = 1, F = 2 → p = 0.881", "y − p = +0.119"]],
              [50 * mm, 46 * mm, CONTENT_W - 96 * mm - 18]),
        P("Con la logística, a medida que el modelo se acerca a la clase correcta el gradiente se achica solo.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c36_gradiente.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook y los Bloques 4 a 6 del cuaderno",
                   [P("En el notebook programas Gradient Boosting <b>desde cero</b> con DecisionTreeRegressor y compruebas que coincide con sklearn. "
                      "Antes de empezar un ejercicio, escribe F₀, los residuos y la regla de actualización.", "cell")]))

    s.append(h("5", "Tasa de aprendizaje y número de árboles"))
    s += para("η es el tamaño del paso. El profesor lo explica con una analogía: estudiar poco cada día (η bajo, muchos «días» = árboles) deja más "
              "aprendizaje que estudiar todo la noche anterior. Con η bajo hacen falta más árboles, el costo sube, pero el modelo suele ser más estable. "
              "Un modelo idealizado ayuda a ver el efecto: si cada árbol corrigiera exactamente la fracción η del residuo, tras m árboles quedaría r₀(1 − η)<sup>m</sup>.")
    s.append(ejemplo("¿cuántos árboles hacen falta?", [
        P("r₀ = 10. Queremos bajar el residuo a menos de 1 (a menos del 10% del inicial).", "cell"),
        pasos([["η", "Residuo tras 1, 2, 3 árboles", "Árboles para bajar de 1"],
               ["0.5", "5 · 2.5 · 1.25", "0.5<sup>m</sup> &lt; 0.1 → m ≥ 4 (0.0625)"],
               ["0.1", "9 · 8.1 · 7.29", "0.9<sup>m</sup> &lt; 0.1 → m ≥ 22 (0.098)"]],
              [16 * mm, 56 * mm, CONTENT_W - 72 * mm - 18]),
        P("Dividir η entre 5 multiplica por ≈ 5.5 los árboles necesarios. En datos reales el óptimo se busca con validación: en Ames, η = 0.3 necesitó ≈ 460 árboles y η = 0.1, ≈ 590, "
          "con casi el mismo error mínimo (22.5k y 21.5k).", "cell")]))

    s.append(h("6", "XGBoost y LightGBM"))
    s += para("<b>XGBoost</b> (<i>Extreme Gradient Boosting</i>) es Gradient Boosting con ingeniería: una función objetivo con "
              "<b>regularización</b> (penaliza árboles complejos), gradientes de primer y segundo orden, submuestreo de filas y columnas como el bosque, "
              "tratamiento de valores nulos y de dispersión, y un núcleo en C++ (no está en scikit-learn: se instala aparte, aunque ofrece una interfaz compatible). "
              "<b>LightGBM</b> (<i>Light Gradient Boosting Machine</i>) usa <b>histogramas</b> (agrupa valores continuos en bins) y crece el árbol <b>hoja por hoja</b>: "
              "siempre expande la hoja con mayor ganancia, así que el árbol queda asimétrico y cuesta menos con muchos datos. Con datos pequeños el crecimiento "
              "hoja por hoja puede sobreajustar.")
    s.append(ejemplo("hojas y profundidad", [
        P("¿Cuántas hojas tiene como máximo un árbol por nivel con max_depth = 4? 2<sup>4</sup> = 16. "
          "En LightGBM con num_leaves = 15 el árbol puede ser asimétrico: 15 hojas implican 14 cortes, y en el caso extremo una rama de profundidad 14 "
          "(por eso conviene fijar también max_depth o min_data_in_leaf).", "cell"),
        P("Regla práctica: num_leaves &lt; 2<sup>max_depth</sup>. Con max_depth = 5, num_leaves debe ser menor que 32.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c36_xgb_lgbm.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s += para("<b>Early stopping.</b> Se reserva un conjunto de validación (distinto del test), se mide el error después de cada árbol y se detiene "
              "cuando no mejora durante k rondas consecutivas. El número de árboles se elige solo, y se conserva la mejor ronda.")
    s.append(ejemplo("early stopping con paciencia 2", [
        P("RMSE de validación por ronda: 30, 25, 22, 21, 21.4, 21.9, 22.5.", "cell"),
        P("El mínimo es 21 en la ronda 4. Las rondas 5 y 6 no mejoran (2 seguidas): con paciencia 2 se detiene tras la 6 y la mejor ronda es la 4.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook y los Bloques 7 y 8 del cuaderno",
                   [P("Reserva siempre un conjunto de validación propio para early stopping: si usas el test, tu estimación final deja de ser honesta.", "cell")]))

    s.append(PageBreak())
    s.append(h("7", "Cómo elegir entre los cinco modelos"))
    s.append(table([
        ["Si lo que domina tu problema es…", "Elige", "Porque"],
        ["Una línea base simple, datos limpios", "AdaBoost", "intuición clara y pocos hiperparámetros"],
        ["Control didáctico del proceso", "Gradient Boosting", "puedes seguir F₀, r₁, F₁… paso a paso"],
        ["Rendimiento general (candidato por defecto)", "XGBoost", "regularizado y muy optimizado"],
        ["Gran escala de filas y variables", "LightGBM", "histogramas y crecimiento por hoja"],
        ["Muchísimas variables categóricas", "CatBoost", "las maneja de forma nativa (no se vio en la práctica)"],
    ], [66 * mm, 36 * mm, CONTENT_W - 102 * mm]))
    s.append(Spacer(1, 4))
    s += para("Con los datos de la clase: en Heart (303 filas) los cinco modelos quedan entre 0.89 y 0.91 de AUC, dentro del desvío; en Ames (2.930 casas) "
              "XGBoost fue el mejor (RMSE 24.4k), seguido de Gradient Boosting (26.8k), LightGBM (27.0k) y el bosque (27.5k), y AdaBoost el peor (36.2k). "
              "En clase el profesor lo resumió así: «a problemas complejos, modelos complejos; a problemas simples, modelos simples», y «el consejo es probar».")
    s.append(img("c36_comparacion.png", width=CONTENT_W * 0.93))
    s.append(Spacer(1, 6))
    s.append(h("8", "Chuleta de código"))
    s.append(table([
        ["Quiero…", "Código"],
        ["AdaBoost", "AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=42)"],
        ["Gradient Boosting", "GradientBoostingRegressor(n_estimators=350, learning_rate=0.05, max_depth=2)"],
        ["Ver F₁, F₂, … paso a paso", "for pred in gb.staged_predict(X_val): ..."],
        ["XGBoost con early stopping", "XGBRegressor(n_estimators=1500, learning_rate=0.05, early_stopping_rounds=50).fit(X, y, eval_set=[(Xv, yv)])"],
        ["LightGBM", "LGBMClassifier(num_leaves=15, subsample=0.8, subsample_freq=1, verbosity=-1)"],
        ["Importancia correcta", "permutation_importance(modelo, X_test, y_test, scoring='roc_auc', n_repeats=20)"],
        ["Optimizar", "RandomizedSearchCV(pipe, {'model__max_depth': [...]}, n_iter=20, scoring='roc_auc', cv=cv)"],
    ], [46 * mm, CONTENT_W - 46 * mm]))
    s.append(Spacer(1, 6))
    s.append(h("9", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Usar boosting con etiquetas dudosas", "Persigue los errores: se obsesiona con el ruido", "Limpia y revisa etiquetas; prueba con η bajo"],
        ["η bajo con pocos árboles", "Subajusta", "Más árboles o early stopping"],
        ["Early stopping contra el test", "El test deja de ser honesto", "Validación aparte"],
        ["Fiarse de feature_importances_ entre librerías", "XGBoost = ganancia, LightGBM = cortes", "Permutación sobre el test"],
        ["LightGBM subsample sin subsample_freq", "No hace nada", "subsample_freq = 1"],
        ["Declarar ganador con una partición", "Diferencias dentro del desvío", "CV repetida y media ± desvío"],
    ], [52 * mm, 54 * mm, CONTENT_W - 106 * mm], header_color=RED))
    s.append(Spacer(1, 6))
    s.append(recuerda(["Orden sugerido: estos apuntes → Bloques 1–3 del cuaderno → Partes A y B del notebook → Bloques 4–6 → Parte C → "
                       "Bloques 7–9 → Parte D → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/C36_Apuntes_Boosting.pdf", s, "Apuntes · Clase 36 · Boosting")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#EA580C">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 3)]
    flow.append(numbered([("", t) for t in items], C36))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 36 · Boosting",
                  "Para resolver en papel, con calculadora (tecla ln y exp) · soluciones paso a paso al final", C36))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Para AdaBoost usa α = ½ ln((1 − ε)/ε) y los factores e<sup>α</sup> y e<sup>−α</sup>.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · "
                           "<font name='Sym'>★★★</font> integra varios conceptos.",
                           "Los bloques 8 y 9 usan los números de la guía (páginas 3 a 7)."], "cell", C36)], bg=SOFT, border=C36))
    s.append(Spacer(1, 4))
    s += bloque(1, "Bagging vs boosting", "★", None, [
        "Un bosque de 1.000 árboles tarda 100 unidades con 10 procesadores. ¿Cuánto tardaría un boosting de 1.000 árboles en las mismas condiciones y por qué?",
        "Completa: en bagging los árboles se entrenan ________ con ________; en boosting se entrenan ________ y cada uno usa el ________ del anterior.",
        "¿Por qué no tiene sentido que cada árbol de AdaBoost use una muestra bootstrap distinta?",
    ])
    s += bloque(2, "Una ronda de AdaBoost", "★★", "Diez observaciones con pesos iniciales 0.1. El primer stump clasifica mal 3 de ellas.", [
        "Calcula el error ponderado ε.",
        "Calcula α = ½ ln((1 − ε)/ε).",
        "Calcula los factores e<sup>α</sup> y e<sup>−α</sup> y los pesos sin normalizar de errores y aciertos.",
        "Normaliza. ¿Cuánto pesan en total las 3 observaciones mal clasificadas?",
        "<font name='Sym'>★★★</font> ¿Qué α recibe un stump con ε = 0.5? ¿Y uno con ε = 0.8? Interpreta.",
    ])
    s += bloque(3, "El voto final de AdaBoost", "★", "Cuatro stumps con α = 0.8, 0.5, 0.45 y 0.2 votan +1, −1, −1 y +1.", [
        "Calcula Σ α·h y la clase predicha.",
        "¿Gana la clase con más votos? Explica.",
    ])
    s += bloque(4, "Una etapa de Gradient Boosting", "★★", "Tabla de la clase: y = [12, 7, 11, 10] para A, B, C, D; F₀ = 10. El árbol separa {A, C} de {B, D}.", [
        "Calcula los residuos r₁ = y − F₀ y el RMSE inicial.",
        "Calcula las predicciones del árbol (media de r₁ por hoja).",
        "Con η = 1 calcula F₁, los nuevos residuos y el RMSE. ¿Mejora cada observación?",
        "Repite con η = 0.5.",
    ])
    s += bloque(5, "Dos etapas completas", "★★★", "y = [10, 4, 9, 17] para A, B, C, D; η = 0.5; pérdida cuadrática. Árbol 1: {A, B} | {C, D}. Árbol 2: {B} | {A, C, D}.", [
        "Calcula F₀ y r₁. Ajusta el árbol 1 y calcula F₁ y r₂.",
        "Ajusta el árbol 2 sobre r₂ y calcula F₂ y r₃.",
        "Calcula el RMSE de F₀, F₁ y F₂. ¿Qué observación sigue peor explicada?",
    ])
    s += bloque(6, "Pérdida y gradiente", "★★", None, [
        "y = 20 y F = 26. Calcula −∂L/∂F para la pérdida cuadrática ½(y − F)², la absoluta y Huber con δ = 4.",
        "Clasificación logística, p = 1/(1 + e<sup>−F</sup>): calcula y − p para (y = 0, p = 0.8) y para (y = 1, p = 0.2).",
        "¿Cuál de las tres pérdidas del punto 1 es menos sensible a un valor extremo de y? Justifica.",
    ])
    s += bloque(7, "Tasa de aprendizaje y número de árboles", "★★", "Modelo idealizado: cada árbol corrige la fracción η del residuo, que parte en r₀ = 8.", [
        "Calcula el residuo tras 1, 2 y 3 árboles con η = 0.2.",
        "¿Cuántos árboles hacen falta para bajar de 0.5 con η = 0.2 y con η = 0.05?",
        "Un compañero usa η = 0.01 con 100 árboles y obtiene un modelo malo. ¿Qué pasó y qué le recomiendas?",
    ])
    s += bloque(8, "Curvas, early stopping y sobreajuste", "★★", "XGBoost en Ames: ronda 100 → RMSE train 20.2k, val 23.7k · ronda 400 → 12.6k, 21.4k · "
                "ronda 754 → 9.4k, 21.0k. Validación por ronda: 23, 22, 21.5, 21.2, 21.3, 21.6, 22.", [
        "Calcula la brecha train-validación en cada ronda. ¿Qué indica?",
        "Con early_stopping_rounds = 2 y la secuencia de validación dada (rondas 1 a 7), ¿en qué ronda se detiene y cuál es la mejor?",
        "¿Por qué la validación del early stopping no puede ser el conjunto de test?",
    ])
    s += bloque(9, "Elegir y leer resultados", "★★★", None, [
        "Heart: AUC (CV 5×3) AdaBoost 0.912 ± 0.039 y XGBoost 0.894 ± 0.035. ¿Puedes afirmar que AdaBoost es mejor? Justifica.",
        "Ames: RMSE de test 24.4k (XGBoost), 36.2k (AdaBoost). ¿Cuánto se equivoca en promedio cada uno en dólares y cuál elegirías?",
        "LightGBM en Ames: RMSE de entrenamiento 8.0k y de test 27.0k. Diagnostica y propón dos hiperparámetros para corregirlo.",
        "Elige modelo y justifica: (a) 50 millones de filas, tiempo limitado; (b) 2.000 filas con 30 variables categóricas de muchas categorías; (c) base limpia, línea base rápida.",
    ])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "10 veces más: 1.000 unidades. Cada árbol necesita el error del anterior, así que no hay árboles que repartir entre procesadores."],
        ["2", "En paralelo e independientes, con su muestra · en secuencia, error (o residuo) del árbol anterior."],
        ["3", "Porque los pesos de los casos difíciles se aplican sobre las mismas observaciones. Con muestras distintas, el árbol 2 no vería los casos que el árbol 1 erró."]])
    s += sol("Bloque 2", [
        ["1", "ε = 3 · 0.1 / 1 = 0.3."],
        ["2", "α = ½ ln(0.7/0.3) = ½ ln 2.333 = ½ · 0.847 = 0.424."],
        ["3", "e<sup>0.424</sup> = 1.528 (= √(7/3)); e<sup>−0.424</sup> = 0.655. Errores: 0.1·1.528 = 0.1528 cada uno. Aciertos: 0.1·0.655 = 0.0655 cada uno. Suma = 3·0.1528 + 7·0.0655 = 0.458 + 0.458 = 0.917."],
        ["4", "Errores: 0.1528/0.917 = 0.1667 cada uno; aciertos: 0.0655/0.917 = 0.0714 cada uno. Las 3 observaciones mal clasificadas pesan 3 · 0.1667 = 0.5: la mitad del total."],
        ["5", "ε = 0.5: α = ½ ln 1 = 0, el árbol no aporta al voto. ε = 0.8: α = ½ ln(0.2/0.8) = ½ ln 0.25 = −0.693, es peor que el azar y su voto se invierte (equivale a predecir lo contrario)."]])
    s += sol("Bloque 3", [
        ["1", "0.8·(+1) + 0.5·(−1) + 0.45·(−1) + 0.2·(+1) = 0.8 − 0.5 − 0.45 + 0.2 = 0.05 &gt; 0: clase +1."],
        ["2", "No: hay 2 votos +1 y 2 votos −1, y el voto ponderado desempata a favor de +1 porque el primer stump pesa más (α = 0.8)."]])
    s += sol("Bloque 4", [
        ["1", "r₁ = [2, −3, 1, 0]. RMSE = √((4 + 9 + 1 + 0)/4) = √3.5 = 1.87."],
        ["2", "{A, C}: (2 + 1)/2 = 1.5 · {B, D}: (−3 + 0)/2 = −1.5. Predicciones h = [1.5, −1.5, 1.5, −1.5]."],
        ["3", "η = 1: F₁ = [11.5, 8.5, 11.5, 8.5]; y − F₁ = [0.5, −1.5, −0.5, 1.5]; RMSE = √((0.25 + 2.25 + 0.25 + 2.25)/4) = √1.25 = 1.12. "
              "No mejora cada una: D pasa de residuo 0 a 1.5, porque comparte hoja con B."],
        ["4", "η = 0.5: F₁ = [10.75, 9.25, 10.75, 9.25]; y − F₁ = [1.25, −2.25, 0.25, 0.75]; RMSE = √((1.5625 + 5.0625 + 0.0625 + 0.5625)/4) = √1.8125 = 1.35. Baja menos, pero el siguiente árbol puede corregir D."]])
    s += sol("Bloque 5", [
        ["1", "F₀ = (10 + 4 + 9 + 17)/4 = 10; r₁ = [0, −6, −1, 7]. Hojas: {A, B} → (0 − 6)/2 = −3; {C, D} → (−1 + 7)/2 = 3. "
              "F₁ = 10 + 0.5·h = [8.5, 8.5, 11.5, 11.5]; r₂ = y − F₁ = [1.5, −4.5, −2.5, 5.5]."],
        ["2", "Árbol 2: {B} → −4.5; {A, C, D} → (1.5 − 2.5 + 5.5)/3 = 1.5. F₂ = F₁ + 0.5·h₂ = [9.25, 6.25, 12.25, 12.25]; r₃ = [0.75, −2.25, −3.25, 4.75]."],
        ["3", "RMSE₀ = √((0 + 36 + 1 + 49)/4) = √21.5 = 4.64 · RMSE₁ = √((2.25 + 20.25 + 6.25 + 30.25)/4) = √14.75 = 3.84 · RMSE₂ = √((0.5625 + 5.0625 + 10.5625 + 22.5625)/4) = √9.6875 = 3.11. "
              "La observación D (residuo 4.75) es la que sigue peor explicada."]])
    s += sol("Bloque 6", [
        ["1", "Residuo y − F = −6. Cuadrática: −6. Absoluta: signo(−6) = −1. Huber (δ = 4): residuo recortado a ±4 → −4."],
        ["2", "(y = 0, p = 0.8): y − p = −0.8. (y = 1, p = 0.2): y − p = +0.8. El modelo está muy seguro y equivocado: el gradiente es grande y empuja fuerte hacia la clase correcta."],
        ["3", "La absoluta (gradiente acotado en ±1) y Huber (acotado en ±δ). La cuadrática crece con el residuo, así que un valor extremo domina el ajuste."]])
    s += sol("Bloque 7", [
        ["1", "r₀(1 − η)<sup>m</sup> = 8·0.8, 8·0.64, 8·0.512 = 6.4 · 5.12 · 4.096."],
        ["2", "Se busca residuo &lt; 0.5, o sea 8(1 − η)<sup>m</sup> &lt; 0.5 → (1 − η)<sup>m</sup> &lt; 0.0625. η = 0.2: m &gt; ln 0.0625 / ln 0.8 = (−2.77)/(−0.223) = 12.4 → 13 árboles. "
              "η = 0.05: m &gt; (−2.77)/(−0.0513) = 54.1 → 55 árboles."],
        ["3", "Subajuste: con η = 0.01 el residuo tras 100 árboles sigue en 0.99<sup>100</sup> = 37% del inicial. Recomendación: subir n_estimators (≈ 300 para el 5%) o usar early stopping, o aumentar η."]])
    s += sol("Bloque 8", [
        ["1", "Brechas: ronda 100 → 3.5k; ronda 400 → 8.8k; ronda 754 → 11.6k. El entrenamiento sigue bajando y la validación se aplana: la brecha crece (más memoria, poca ganancia en validación)."],
        ["2", "El mínimo es 21.2 en la ronda 4. Las rondas 5 (21.3) y 6 (21.6) no mejoran: se detiene tras la 6 y la mejor ronda es la 4."],
        ["3", "Porque se usa para elegir la ronda: dejaría de ser independiente y la estimación final del error sería optimista."]])
    s += sol("Bloque 9", [
        ["1", "No. La diferencia (0.018) es menor que cualquiera de los desvíos (0.039 y 0.035). Con 303 pacientes no se distinguen: hacen falta más repeticiones o más datos."],
        ["2", "≈ 24 mil dólares por casa (XGBoost) y ≈ 36 mil (AdaBoost). Se elige XGBoost: 12 mil menos de error promedio, una diferencia grande."],
        ["3", "Sobreajuste: gran brecha. Corregir con menos hojas (num_leaves 15), más min_child_samples (50), y además subsample con subsample_freq = 1, colsample_bytree o reg_lambda, y early stopping."],
        ["4", "(a) LightGBM: histogramas y hojas, ideal para gran escala. (b) CatBoost: muchas categóricas; con 2.000 filas ojo con el sobreajuste. (c) AdaBoost o Gradient Boosting con pocos hiperparámetros."]])
    build("apuntes/C36_Ejercicios_Cuaderno_Boosting.pdf", s, "Ejercicios para el cuaderno · Clase 36")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
