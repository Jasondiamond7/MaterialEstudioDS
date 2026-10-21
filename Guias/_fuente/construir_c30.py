"""Clase 30 · Preparación para Machine Learning: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C30 = colors.HexColor("#7C3AED")
C30_BG = colors.HexColor("#F5F3FF")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + apunte + notebook y caso de la profesora Laura Sánchez"


def topic_page(s, titulo, idea, conceptos, figura, pie, formulas, py, salida, cuidado, fig_w=0.97):
    s.append(band(f"Clase 30 · {titulo}", FUENTE, C30))
    s.append(Spacer(1, 6))
    s.append(box([P(idea, "big")], bg=SOFT, border=C30))
    s.append(Spacer(1, 5))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C30))
    s.append(Spacer(1, 3))
    if figura:
        s.append(img(figura, width=CONTENT_W * fig_w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 4))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF),
            Spacer(1, 4), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C30), Spacer(1, 4),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER,
                 width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 30 · Preparación para Machine Learning: variables, datos y validación",
                  "Profesora Laura Sánchez · inicio del bloque de Machine Learning · precede a Introducción al ML (C31)", NAVY))
    s.append(Spacer(1, 7))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("En Machine Learning el modelo se juzga por <b>cómo se comporta con datos que nunca vio</b>. Por eso hay reglas del juego "
                    "antes de cualquier algoritmo: conocer el <b>tipo</b> de cada variable, revisar la <b>calidad</b> de los datos, <b>dividir</b> "
                    "en entrenamiento / validación / prueba, <b>regularizar</b> sobre variables comparables, <b>validar de forma estable</b> (K-Fold, "
                    "estratificación) y elegir <b>métricas</b> acordes al costo de equivocarse. «Si entendemos las piezas, podremos armar el engranaje.»", "body")],
                 bg=C30_BG, border=C30))
    s.append(Spacer(1, 5))
    s.append(P("El camino de los datos en una imagen", "h2"))
    s.append(img("c30_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Regresión lineal y R² (Clases 17–19); regresión logística (Clases 20–21)",
                     "Media, mediana, moda y desviación estándar (Etapa 0)",
                     "Probabilidad condicional: precision y recall solo cambian el «total» (explicación de la clase)",
                     "Python básico con pandas (Colab o Jupyter)"], "cell", C30)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Variables</b> → <b>calidad e imputación</b> → <b>entrenamiento / validación / prueba</b> → <b>Ridge y Lasso</b> antes y "
                      "después de estandarizar → <b>K-Fold</b> → <b>estratificación</b> → <b>métricas</b>",
                      "Datos: <b>300 viviendas simuladas</b> (precio en CLP) y <b>2.000 casos de fraude simulados</b> (5 % de positivos); "
                      "ambos se generan en el notebook de la profesora, no hay archivos que descargar",
                      "Caso de repaso: <b>scoring de crédito de consumo</b> (100.000 clientes, 2 % de impago)",
                      "Pregunta guía: ¿cómo sé que mi modelo aprendió y no solo memorizó?"], "cell", C30)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Primera vez con scikit-learn</b>", "cellb"),
                  P("La API completa cabe en tres verbos: <font name='Mono'>fit</font> (aprender de los datos), <font name='Mono'>transform</font> "
                    "(aplicar lo aprendido) y <font name='Mono'>predict</font> (predecir). Regla de oro: <b>fit solo con entrenamiento</b>. "
                    "Resumen en la página 7.", "cell")], bg=SOFT, border=C30))
    s.append(PageBreak())

    topic_page(
        s, "Variables y calidad de los datos",
        "Los modelos solo entienden números: hay que saber qué <b>tipo</b> es cada variable y si los datos son <b>confiables</b> antes de modelar. "
        "Un dato faltante se completa (imputa) con una regla que depende del tipo y de la distribución.",
        [("Tipos de variables", "Numéricas (continuas o discretas) y categóricas (nominales sin orden, ordinales con orden). <font name='Mono'>n_habitaciones</font> es numérica discreta; <font name='Mono'>barrio</font>, nominal."),
         ("Codificar categóricas", "Antes de entrenar se convierten a números, por ejemplo con one-hot (<font name='Mono'>pd.get_dummies</font> u <font name='Mono'>OneHotEncoder</font>): una columna 0/1 por categoría."),
         ("Imputar", "Normal y sin atípicos: <b>media</b>. Con atípicos o asimetría: <b>mediana</b>. Categórica: <b>moda</b>. Si se relaciona con otras columnas: un modelo de regresión. Si falta un tercio, no se imputa: se revisa la recolección."),
         ("Cuatro dimensiones de calidad", "<b>Exactitud</b> (refleja la realidad), <b>completitud</b> (no faltan datos), <b>coherencia</b> (las fuentes concuerdan), <b>frescura</b> (está vigente). Pueden fallar varias a la vez."),
         ("Los datos mandan", "Sesgos en los datos se vuelven sesgos del modelo (ejemplo de la clase: generadores de imágenes). Un buen modelo no repara datos malos.")],
        "c30_calidad.png",
        "Izquierda: faltantes de las 300 viviendas (solo antigüedad: 20). Centro: tras imputar con la media, 20 filas quedan apiladas en un mismo valor (la desviación baja de 14.19 a 13.71). Derecha: ingresos simulados con cola larga.",
        ["imputación: media · mediana · moda · regresión",
         "% faltante = faltantes / filas = 20/300 = 6.7 %",
         "media imputada ⇒ la desviación estándar baja",
         "imputer.fit(solo entrenamiento)"],
        ["viviendas.dtypes · .isna().sum()",
         "pd.get_dummies(df, columns=['barrio'])",
         "SimpleImputer(strategy='median')",
         "# 'mean' · 'median' · 'most_frequent'"],
        ["300 viviendas, 6 columnas; antigüedad con 20 faltantes (6.7 %), media observada 24.56 años",
         "Ingresos simulados con cola: media 0.75 M vs mediana 0.64 M de CLP (los atípicos tiran la media)",
         "Caso del banco: ingreso (8 % nulos, sesgo a la derecha) → mediana; edad (2 %, simétrica) → media; comuna (5 %) → moda"],
        ["La media y la mediana no se calculan con todo el dataset antes de dividir: se aprenden con el entrenamiento (fuga de datos, página 7)."])

    topic_page(
        s, "División de los datos y sobreajuste",
        "Se divide el dataset en tres: <b>entrenamiento</b> (aprende parámetros), <b>validación</b> (compara alternativas y ajusta hiperparámetros) "
        "y <b>prueba</b> (veredicto final, se usa una sola vez). Así se detecta el <b>sobreajuste</b>.",
        [("Tres roles distintos", "Entrenamiento: el modelo aprende. Validación: tú eliges (alpha, profundidad, modelo). Prueba: se reserva y se mira al final; por eso es «sagrada»."),
         ("Sobreajuste (overfitting)", "El modelo aprende ruido y particularidades del entrenamiento: rinde muy bien ahí y peor con datos nuevos. La señal es la <b>brecha</b> entre entrenamiento y validación."),
         ("Subajuste (underfitting)", "Modelo demasiado simple: rinde mal en ambos. El punto bueno está en el medio, donde la validación es máxima."),
         ("Proporciones", "70/15/15 es solo una referencia; depende del tamaño del problema. Con pocos datos se usa validación cruzada."),
         ("Datos con tiempo", "Si hay orden temporal, se divide cronológicamente (entrenar con lo anterior, probar con lo posterior): un corte al azar filtra el futuro.")],
        "c30_division.png",
        "Izquierda: la división de la clase (210 / 45 / 45). Derecha: árboles de profundidad creciente sobre esas viviendas; el entrenamiento llega a 0.999 y la validación se estanca en ≈ 0.80 tras su máximo (0.847).",
        ["n_entrenamiento = 0.70 · 300 = 210",
         "n_temp = 0.30 · 300 = 90 → 45 y 45",
         "brecha = métrica_entrenamiento − métrica_validación",
         "la prueba no decide nada: solo informa"],
        ["X_train, X_temp, y_train, y_temp =",
         "    train_test_split(X, y, test_size=0.30,",
         "                     random_state=42)",
         "X_val, X_test, y_val, y_test =",
         "    train_test_split(X_temp, y_temp, test_size=0.5)"],
        ["Dos llamadas a train_test_split: 70 % y luego el 30 % restante a la mitad → 210 / 45 / 45",
         "Árbol de profundidad 12: R² 0.999 en entrenamiento y 0.801 en validación; el mejor punto fue profundidad 7 (0.847)",
         "Caso del banco: el 90/10 del equipo de TI no deja dónde ajustar sin tocar la prueba",
         "En 120 de 300 particiones simples el R² de validación superó al de entrenamiento (≈ 45 filas de validación)"],
        ["En clase se dijo que una validación mejor que el entrenamiento «es raro». Con pocas filas es normal; lo preocupante es una brecha grande a favor del entrenamiento."])

    topic_page(
        s, "Regularización: Ridge, Lasso y estandarizar",
        "Regularizar es <b>castigar la complejidad</b>: se suma una penalización a los coeficientes. Como el castigo depende del tamaño de cada coeficiente, "
        "las variables deben estar en una <b>escala comparable</b> (estandarizar).",
        [("Ridge (L2)", "Penaliza la suma de coeficientes al cuadrado: <b>los achica</b>, normalmente sin llegar a cero. Conserva todas las variables."),
         ("Lasso (L1)", "Penaliza la suma de valores absolutos: puede llevar coeficientes <b>exactamente a cero</b> y actúa como selección de variables. Elastic Net mezcla ambos."),
         ("alpha", "Es un hiperparámetro: mayor alpha, más castigo. Se elige con validación, no con la prueba. Que Lasso anule una variable depende de alpha y de los datos."),
         ("Estandarizar", "Restar la media y dividir por la desviación estándar (z-score). Sin eso, el castigo depende de las <b>unidades</b>: una variable en cientos de m² paga mucho más que una en m²."),
         ("No arregla un modelo malo", "Solo hace comparable la penalización. Y el StandardScaler se aprende con el entrenamiento, dentro de un Pipeline.")],
        "c30_regularizacion.png",
        "Izquierda: coeficientes del modelo estandarizado (la prueba de la profesora). Centro: camino de Lasso, el ruido se anula primero (alpha ≈ 0.4). Derecha: la misma regresión Ridge con la superficie en m² o en centenas de m².",
        ["Ridge: minimiza RSS + α·Σ βⱼ²",
         "Lasso: minimiza RSS + α·Σ |βⱼ|",
         "z = (x − media) / desviación estándar",
         "α = 0 ⇒ regresión sin regularizar"],
        ["make_pipeline(StandardScaler(),",
         "               Ridge(alpha=10))",
         "make_pipeline(StandardScaler(),",
         "               Lasso(alpha=2))",
         "modelo[-1].coef_   # coeficientes"],
        ["Estandarizado: superficie 56.4 (sin regularizar) → 53.8 (Ridge) → 54.3 (Lasso); el ruido: −0.40, −0.44 y exactamente 0",
         "Sin estandarizar los coeficientes (2.18, 8.19, −0.86) no son comparables con los estandarizados (56.4, 11.9, −12.1): cambian las unidades",
         "Ridge con alpha = 30: superficie en m² R² 0.916; en centenas de m² cae a 0.532; estandarizado siempre 0.887"],
        ["Con alpha = 2 Lasso ya anula el ruido incluso sin estandarizar (−0.000): la demo de clase no muestra el daño de no escalar; el panel derecho sí."])

    topic_page(
        s, "Validación cruzada K-Fold y estratificación",
        "Una sola partición puede ser «fácil» o «difícil» por azar. <b>K-Fold</b> repite la validación K veces y promedia; la <b>estratificación</b> "
        "conserva la proporción de clases al dividir. Ninguna reemplaza al conjunto de prueba.",
        [("K-Fold", "Se parten los datos en K pliegues; cada uno actúa una vez como validación y los otros K − 1 entrenan. Se promedian los K resultados."),
         ("Media y dispersión", "No solo importa el promedio: la desviación entre pliegues dice cuán estable es el modelo. Dispersión grande: el resultado depende de la muestra."),
         ("K-Fold sirve en el desarrollo", "Se hace con los datos de entrenamiento (o entrenamiento + validación). La prueba queda fuera hasta el final."),
         ("Muestreo estratificado", "Con clases raras (5 % de fraude) conserva la proporción en cada partición. No aumenta los fraudes: solo evita que una partición quede casi sin ellos."),
         ("Otros muestreos", "Aleatorio simple (sin estructura ni desbalance) y consciente del tiempo (orden cronológico).")],
        "c30_kfold.png",
        "Izquierda: esquema de 5 pliegues (la prueba queda aparte). Centro: R² por pliegue con la media y ± una desviación. Derecha: R² de 300 particiones simples; una sola puede dar de 0.82 a 0.96.",
        ["tamaño del pliegue = n / K",
         "resultado = promedio de los K puntajes",
         "dispersión = desviación entre pliegues",
         "stratify=y ⇒ misma proporción de clases"],
        ["kf = KFold(n_splits=5, shuffle=True,",
         "            random_state=42)",
         "cross_val_score(pipe, X_dev, y_dev,",
         "                cv=kf, scoring='r2')",
         "train_test_split(X, y, stratify=y)"],
        ["255 viviendas de desarrollo, 5 pliegues: R² 0.941 · 0.931 · 0.918 · 0.895 · 0.933 → media 0.924, desviación 0.016",
         "Fraude: 109 positivos en 2.000 (5.45 %); entrenamiento 5.467 %, prueba 5.400 % con stratify",
         "Sin estratificar (n = 2.000) la prueba varía de 2.8 % a 8.4 %; con n = 200 hay 2.2 % de particiones sin ningún fraude en la prueba"],
        ["El notebook de la clase hace el K-Fold con las 300 filas (0.921 ± 0.010), o sea, incluye la prueba. Aquí se usa solo entrenamiento + validación. En clasificación desbalanceada usa <font name='Mono'>StratifiedKFold</font>."])

    topic_page(
        s, "Métricas de evaluación",
        "Con clases desbalanceadas el <b>accuracy engaña</b>: un modelo que siempre dice «no fraude» acierta el 94.6 % y no detecta nada. "
        "Se mira la matriz de confusión y precision, recall y F1, según <b>qué error cuesta más</b>.",
        [("Matriz de confusión", "TP (fraude detectado), FP (falsa alarma), FN (fraude que se escapó) y TN. Todas las métricas salen de estas cuatro cifras."),
         ("Precision", "De todo lo que marqué como positivo, ¿cuánto lo era? Mide la confiabilidad de las alertas: precision alta = pocas falsas alarmas."),
         ("Recall (exhaustividad)", "De todos los positivos reales, ¿cuántos detecté? Recall alto = se escapan pocos. Mismo numerador que precision, <b>otro total</b>."),
         ("F1", "Media armónica de precision y recall: resume el equilibrio cuando ambos errores importan. Subir una suele bajar la otra (umbral)."),
         ("Regresión", "MAE (error medio en las unidades del precio), RMSE (castiga más los errores grandes) y R² (proporción de variabilidad explicada). Un R² alto no basta.")],
        "c30_metricas.png",
        "Izquierda: matriz de confusión de la regresión logística en la prueba (500 casos). Centro: precision, recall y F1 al variar el umbral, calculados en validación. Derecha: accuracy de «siempre no fraude» frente al modelo.",
        ["accuracy = (TP + TN) / n",
         "precision = TP / (TP + FP)",
         "recall = TP / (TP + FN)",
         "F1 = 2·P·R / (P + R)"],
        ["precision_score(y, pred)",
         "recall_score(y, pred)",
         "f1_score(y, pred)",
         "confusion_matrix(y, pred)",
         "modelo.predict_proba(X)[:, 1] >= umbral"],
        ["Matriz: TN 466 · FP 7 · FN 17 · TP 10 → accuracy 0.952, precision 0.588, recall 0.370, F1 0.455",
         "«Siempre no fraude»: accuracy 0.946 y recall 0: el 0.952 apenas lo supera",
         "Regresión de precios (prueba): MAE 13.9 M, RMSE 16.2 M de CLP, R² 0.904",
         "Caso del banco (2 % de impago): un falso negativo es costoso, así que importa el recall, sin olvidar la precision"],
        ["En clase se dijo F1 «50 %»; la salida real del notebook es 0.455. Con 27 fraudes en la prueba, mover el umbral cambia mucho el resultado: elígelo en validación."])

    # ------------------------------------------------------------ API sklearn + fuga
    s.append(band("scikit-learn, Pipeline y fuga de datos", "Lo que más se rompe en la primera clase de Machine Learning", NAVY))
    s.append(Spacer(1, 6))
    s.append(img("c30_leakage.png", width=CONTENT_W * 0.97))
    s.append(P("Datos de puro ruido: con 1.000 columnas sin relación alguna con la etiqueta, seleccionar las 10 «mejores» con todas las filas y luego validar da ≈ 0.79 de exactitud "
               "(40 repeticiones); hacerlo dentro de un Pipeline da ≈ 0.51, que es lo honesto.", "small"))
    s.append(Spacer(1, 4))
    s.append(table([["Verbo", "Qué hace", "Con qué datos", "Ejemplo"],
                    ["fit", "Aprende (media, coeficientes, categorías…)", "<b>solo entrenamiento</b>", "scaler.fit(X_train)"],
                    ["transform", "Aplica lo aprendido", "entrenamiento, validación, prueba", "scaler.transform(X_val)"],
                    ["fit_transform", "fit + transform seguidos", "solo entrenamiento", "scaler.fit_transform(X_train)"],
                    ["predict", "Predice con un modelo ya ajustado", "validación, prueba, datos nuevos", "modelo.predict(X_test)"],
                    ["Pipeline", "Encadena pasos: hace fit de cada uno solo con lo que recibe", "se llama con <font name='Mono'>.fit(X_train, y_train)</font>", "make_pipeline(StandardScaler(), Ridge())"],
                    ["ColumnTransformer", "Un tratamiento por tipo de columna", "dentro del Pipeline", "num → imputar + escalar; cat → moda + one-hot"]],
                   [28 * mm, 50 * mm, 46 * mm, CONTENT_W - 124 * mm]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Fuga de datos (data leakage)</b>", "cellb"),
                  P("Es usar, directa o indirectamente, información del conjunto de prueba (o de la validación) durante el desarrollo: para entrenar, imputar, escalar, "
                    "seleccionar variables o elegir hiperparámetros. El resultado se ve mejor de lo que será en producción. La profesora lo menciona en el caso de repaso; "
                    "aquí la forma práctica de evitarlo es el <b>Pipeline</b>.", "cell")], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    checks = [("Divido primero", "antes de imputar, escalar, codificar o seleccionar"),
              ("Uso un Pipeline", "así cada pliegue de K-Fold re-aprende sus propios pasos"),
              ("Estratifico si hay clases raras", "stratify=y y StratifiedKFold"),
              ("Elijo hiperparámetros con validación o K-Fold", "alpha, profundidad, umbral"),
              ("Miro la prueba una sola vez", "con el modelo ya decidido"),
              ("Reporto varias métricas", "accuracy no basta con clases desbalanceadas")]
    s.append(table([["", "Checklist del flujo", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 66 * mm, CONTENT_W - 74 * mm]))
    s.append(PageBreak())

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene corregir (verificado con código)", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Notebook, diapositivas y apunte</b>", "cellb"),
                  bullets(["<b>K-Fold con las 300 filas</b> (Bloque 5): <font name='Mono'>cross_val_score(modelo, X, y)</font> incluye la prueba, aunque el texto dice que sigue reservada. "
                           "Con entrenamiento + validación: 0.924 ± 0.016; con las 300: 0.921 ± 0.010.",
                           "<b>La validación no se usa</b>: <font name='Mono'>X_val</font> se crea y no se vuelve a tocar; el clasificador de fraude se evalúa directo en la prueba, sin validar antes.",
                           "<b>«Sin estandarizar es difícil llevar a cero»</b> (clase): con alpha = 2, Lasso anula el ruido en ambos casos. El daño real es de unidades: Ridge (alpha 30) pasa de R² 0.916 a 0.532 si la superficie se mide en centenas de m². Además las dos tablas de coeficientes tienen unidades distintas: no se pueden comparar entre sí.",
                           "<b>Diapositiva «Prueba (61 %)»</b>: la curva de la brecha 98 % → 61 % es ilustrativa y es de <b>validación</b> (la profesora lo corrige en clase: «esto no es prueba, es validación»).",
                           "<b>COVID y la prueba</b>: «el conjunto de prueba simula ese futuro desconocido». Una prueba tomada al azar del mismo periodo <b>no</b> detecta un cambio de contexto; para eso hace falta una división temporal (el apunte sí lo matiza).",
                           "<b>«Validación mejor que entrenamiento = algo falla»</b>: con 45 filas ocurre en el 40 % de las particiones (120 de 300); es una señal para revisar, no un error.",
                           "<b>Texto vs código</b>: <font name='Mono'>n_habitaciones</font> se describe entre 1 y 6 y el código genera 1 a 5.",
                           "<b>F1 «50 %»</b> dicho en clase; la salida real es 0.455. <b>K-Fold:</b> <font name='Mono'>.std()</font> de NumPy usa ddof = 0 (0.016; con ddof = 1, 0.018) y los pliegues no son independientes."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Caso de repaso (scoring de crédito)</b>", "cellb"),
                  bullets(["<b>Pregunta 1.2</b>: la respuesta oficial dice <b>Exactitud</b>. Con las definiciones del propio apunte (coherencia: «las fuentes concuerdan»), que la web y el contrato no coincidan es primero un problema de <b>coherencia</b>; "
                           "la exactitud entra al elegir cuál es la verdad (el contrato). Conviene responder ambas y justificar.",
                           "<b>Pregunta 1.1</b>: imputar con mediana, media o moda es correcto, pero el valor se calcula solo con el entrenamiento; con 5 % de comunas vacías también cabe una categoría «desconocida».",
                           "<b>Pregunta 3.3</b>: recall es la respuesta esperada, pero un banco real también fija un umbral con validación y mira precision, el costo de rechazar a un buen pagador y métricas como PR-AUC."],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper confundió Ridge (Rich, reach, pitch), Lasso (laso, lazo), overfitting (overpicking), exactitud (exagitu), exhaustividad (exageridad), "
                    "K-Fold (casco holding) y DALL-E. Corregido: «200/45/35 viviendas» era 210/45/45. Hay texto irrecuperable hacia 0:28–0:55, un recreo hacia 1:12–1:24 "
                    "y bucles de repetición, ya omitidos en el .txt y el .srt corregidos.", "cell")],
                 bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Preguntas de repaso", "h2"))
    preg = [("¿Para qué sirve cada conjunto: entrenamiento, validación y prueba?", "Aprender parámetros; elegir alpha o modelo; veredicto final una sola vez."),
            ("¿Por qué el accuracy engaña con 95 % de no fraude?", "Un modelo que siempre dice «no» ya acierta 95 % y su recall es 0."),
            ("¿Qué diferencia hay entre Ridge y Lasso y por qué estandarizar?", "Ridge achica, Lasso puede anular; el castigo depende de las unidades.")]

    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 76 * mm, CONTENT_W - 83 * mm]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"), Spacer(1, 2),
                  bullets(["Clasifico variables y elijo media, mediana o moda con una justificación",
                           "Calculo a mano precision, recall, F1 y accuracy desde una matriz de confusión",
                           "Explico por qué se estandariza antes de Ridge o Lasso y por qué el scaler se ajusta solo con entrenamiento",
                           "Calculo el promedio y la desviación de un K-Fold y reparto una clase rara con estratificación",
                           "Escribo un Pipeline con ColumnTransformer sin filtrar información de la prueba",
                           "Resuelvo los bloques 1 a 8 del cuaderno con al menos 80 % de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C30_Guia_PreparacionML.pdf", s, "Clase 30 · Preparación para Machine Learning")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#7C3AED">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 30 · Preparación para Machine Learning", "Para leer antes de los ejercicios · lectura de ~35 min", C30))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([
                      ("Tipos y calidad", "Cada variable tiene un tipo (numérica o categórica) y los datos se juzgan por exactitud, completitud, coherencia y frescura."),
                      ("Imputar con criterio", "Media (simétrica), mediana (atípicos), moda (categórica); siempre aprendida con el entrenamiento."),
                      ("Tres conjuntos", "Entrenamiento aprende, validación elige, prueba da el veredicto final una sola vez."),
                      ("Regularizar y estandarizar", "Ridge achica y Lasso puede anular; el castigo depende de las unidades, por eso se estandariza."),
                      ("K-Fold y estratificación", "Un promedio y una dispersión en vez de una partición suerte; la clase rara se reparte proporcionalmente."),
                      ("Métricas y fuga", "Precision, recall y F1 según el costo del error; todo el preprocesamiento va dentro de un Pipeline."),
                  ], C30)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C30_PreparacionML.ipynb</font> o el cuaderno "
               "<font name='Mono'>C30_Ejercicios_Cuaderno_PreparacionML.pdf</font>.", "small"))

    s.append(h("1", "Tipos de variables y one-hot"))
    s += para("Una variable <b>numérica</b> se puede sumar y promediar (continua: superficie; discreta: número de hijos). Una <b>categórica</b> son etiquetas: "
              "<b>nominal</b> si no hay orden (región) y <b>ordinal</b> si lo hay (básico, medio, alto). Los modelos solo reciben números, así que las "
              "categóricas se convierten con <b>one-hot</b>: una columna 0/1 por categoría.")
    s.append(ejemplo("clientes de un seguro", [
        pasos([["Columna", "Valores", "Tipo"],
               ["edad", "23, 41, 67", "numérica (se promedia)"],
               ["n_hijos", "0, 1, 3", "numérica discreta"],
               ["region", "Norte, Sur, Este", "categórica nominal → one-hot: 3 columnas 0/1"],
               ["nivel_estudios", "básico, medio, superior", "categórica ordinal (hay orden)"],
               ["tiene_deuda", "Sí, No", "categórica binaria → una columna 0/1"]],
              [30 * mm, 46 * mm, CONTENT_W - 76 * mm - 18]),
        P("Con one-hot, <font name='Mono'>region</font> deja de ser una columna y pasa a tres: <font name='Mono'>region_Norte</font>, <font name='Mono'>region_Sur</font>, "
          "<font name='Mono'>region_Este</font>. Un cliente del Sur queda como (0, 1, 0).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Pide <font name='Mono'>viviendas.dtypes</font> y clasifica cada columna; después aplica <font name='Mono'>pd.get_dummies</font> a <font name='Mono'>barrio</font> y cuenta las columnas nuevas.", "cell")]))

    s.append(h("2", "Calidad de datos e imputación"))
    s += para("Un dato faltante se puede completar con un estadístico. <b>Media</b> si la variable es simétrica y sin atípicos; <b>mediana</b> si hay atípicos o asimetría; "
              "<b>moda</b> si es categórica. Si el faltante es muy grande (un tercio), no se imputa: se revisa cómo se recogieron los datos. Imputar con la media deja "
              "la <b>desviación estándar más pequeña</b>, porque todos los rellenos caen en el mismo valor.")
    s.append(ejemplo("seis clientes, un ingreso faltante", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Ingresos (M CLP): 1.2, 1.4, 1.5, 1.6, 9.0, falta · media de los 5 observados", "(1.2 + 1.4 + 1.5 + 1.6 + 9.0)/5 = 14.7/5", "2.94"],
               ["Mediana de los 5 observados", "valor central ordenado", "1.5"],
               ["Región: Norte, Sur, Sur, Este, Sur, falta · moda", "la que más se repite (3 de 5)", "Sur"],
               ["Edad: 30, 32, 34, 36, 38 (simétrica) · media", "170/5", "34"]],
              [64 * mm, 66 * mm, CONTENT_W - 130 * mm - 18]),
        P("El 9.0 empuja la media a 2.94, más del doble de lo que gana la mayoría: la mediana (1.5) representa mejor al cliente típico. "
          "Las cuatro dimensiones de calidad: <b>exactitud</b> (¿refleja la realidad?), <b>completitud</b> (¿falta?), <b>coherencia</b> (¿las fuentes concuerdan?) y <b>frescura</b> (¿está vigente?).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c30_calidad.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 2 del cuaderno",
                   [P("En A2 cuenta los faltantes con <font name='Mono'>isna().sum()</font>, imputa con <font name='Mono'>SimpleImputer</font> y compara la desviación antes y después. "
                      "Recuerda: el imputador se ajusta con el entrenamiento.", "cell")]))

    s.append(h("3", "Entrenamiento, validación, prueba y sobreajuste"))
    s += para("El <b>entrenamiento</b> ajusta los parámetros; la <b>validación</b> sirve para comparar alternativas (alpha, profundidad); la <b>prueba</b> se reserva para el final y se "
              "mira una sola vez. El <b>sobreajuste</b> aparece como una brecha: rinde mucho mejor en entrenamiento que en datos que no vio.")
    s.append(ejemplo("1.000 registros con una división 60/20/20", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Tamaños de cada conjunto", "0.6·1.000 · 0.2·1.000 · 0.2·1.000", "600 · 200 · 200"],
               ["Cómo hacerlo con train_test_split", "test_size = 0.40 y luego test_size = 0.50", "600 y 400 → 200 y 200"],
               ["Modelo A: R² 0.99 en entrenamiento y 0.70 en validación", "brecha = 0.99 − 0.70", "0.29 → sobreajuste"],
               ["Modelo B: R² 0.86 y 0.82", "brecha = 0.04", "estable: B generaliza mejor"]],
              [64 * mm, 66 * mm, CONTENT_W - 130 * mm - 18]),
        P("Se elige B por su validación (0.82 &gt; 0.70), aunque A tenga mejor entrenamiento. La prueba se reserva para informar el modelo ya elegido.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c30_division.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook y el Bloque 3 del cuaderno",
                   [P("Reproduce el 70/15/15 de la profesora (210/45/45) y luego usa la validación para elegir la profundidad de un árbol; mira la prueba solo al final.", "cell")]))

    s.append(h("4", "Ridge, Lasso y estandarización"))
    s += para("Ridge minimiza el error más <b>α·Σβ²</b> y Lasso el error más <b>α·Σ|β|</b>. Como la penalización mira el tamaño de los coeficientes, una variable en unidades "
              "pequeñas necesita un coeficiente grande y paga más. <b>Estandarizar</b> (z = (x − media)/desviación) deja todo en escala comparable.")
    s.append(ejemplo("un mismo efecto en dos unidades", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Superficie: media 80 m², desviación 20 · z de una casa de 120 m²", "(120 − 80)/20", "z = 2"],
               ["Habitaciones: media 3, desviación 1.5 · z de 4.5", "(4.5 − 3)/1.5", "z = 1"],
               ["Efecto de 3 por m² frente a 300 por centena de m² (penalidad Ridge, α = 1)", "3² frente a 300²", "9 frente a 90.000"],
               ["Veces más castigado", "90.000 / 9", "10.000"]],
              [66 * mm, 52 * mm, CONTENT_W - 118 * mm - 18]),
        P("Es el mismo efecto con otras unidades, pero la penalización lo castiga 10.000 veces más. Con las viviendas de la clase, Ridge (alpha 30) pasa de R² 0.916 a 0.532 al cambiar a centenas de m²; "
          "con estandarización queda en 0.887 en ambos casos. Lasso, además, puede llevar coeficientes a cero: en la clase anuló <font name='Mono'>ruido_aleatorio</font>.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c30_regularizacion.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook y el Bloque 4 del cuaderno",
                   [P("En C1 compara los coeficientes con y sin <font name='Mono'>StandardScaler</font> y repite con la superficie en centenas de m². "
                      "Pregunta clave: ¿qué cambia al cambiar las unidades?", "cell")]))

    s.append(h("5", "K-Fold y estratificación"))
    s += para("K-Fold divide los datos en K pliegues y usa cada uno una vez como validación. El resultado es un <b>promedio</b> y una <b>desviación entre pliegues</b>. "
              "La <b>estratificación</b> reparte las clases en proporción: con 5 % de fraude, la prueba también tiene ≈ 5 %.")
    s.append(ejemplo("K = 4 con 100 filas, y una clase rara", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Filas por pliegue (validación) y de entrenamiento", "100/4 · 100 − 25", "25 · 75"],
               ["Puntajes: 0.80, 0.84, 0.78, 0.86 · promedio", "(0.80 + 0.84 + 0.78 + 0.86)/4", "0.82"],
               ["Desviación (ddof = 0, como NumPy)", "√((0.02² + 0.02² + 0.04² + 0.04²)/4) = √0.001", "0.032"],
               ["400 filas con 5 % de fraude, prueba de 25 % · fraudes en la prueba (estratificada)", "20 · 0.25 = 5 de 100 filas", "5 casos"]],
              [66 * mm, 66 * mm, CONTENT_W - 132 * mm - 18]),
        P("Sin estratificar, el número de fraudes en la prueba varía alrededor de 5 (desviación ≈ 1.9), y puede salir 2 o 8. Con la estratificación siempre son 5. "
          "En el notebook de la clase: 2.000 casos con 5.45 % de fraude → entrenamiento 5.467 % y prueba 5.400 %.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c30_estratificado.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C (K-Fold) y la Parte D (estratificación) del notebook y los Bloques 5 y 6 del cuaderno",
                   [P("Calcula el K-Fold con entrenamiento + validación (no con las 300 filas) y repite 1.000 particiones con y sin <font name='Mono'>stratify</font> para ver la dispersión.", "cell")]))

    s.append(h("6", "Matriz de confusión y métricas"))
    s += para("Todo sale de cuatro números. <b>Precision</b> = TP/(TP+FP): de lo que marqué como positivo, cuánto lo era. <b>Recall</b> = TP/(TP+FN): de los positivos reales, cuántos detecté. "
              "Son como probabilidades condicionales: cambia el «total». <b>F1</b> es su media armónica. <b>Accuracy</b> = (TP+TN)/n puede engañar si una clase domina.")
    s.append(ejemplo("1.000 transacciones, 50 fraudes", [
        P("TP = 30, FP = 10, FN = 20, TN = 940.", "cell"),
        pasos([["Métrica", "Cálculo", "Resultado"],
               ["Accuracy", "(30 + 940)/1.000", "0.970"],
               ["Precision", "30/(30 + 10)", "0.750"],
               ["Recall", "30/(30 + 20)", "0.600"],
               ["F1", "2·0.75·0.60/(0.75 + 0.60) = 0.90/1.35", "0.667"],
               ["Accuracy de «siempre no fraude»", "950/1.000", "0.950"]],
              [40 * mm, 76 * mm, CONTENT_W - 116 * mm - 18]),
        P("El accuracy 0.970 parece excelente pero el modelo deja escapar 20 de 50 fraudes. Si bajar el umbral sube el recall a 0.8 con más falsas alarmas, el F1 puede mejorar o no: depende del costo de cada error.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c30_metricas.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook y el Bloque 7 del cuaderno",
                   [P("Calcula las métricas a mano desde <font name='Mono'>confusion_matrix</font>, compáralas con scikit-learn y luego mueve el umbral con "
                      "<font name='Mono'>predict_proba</font> usando la validación.", "cell")]))

    s.append(h("7", "Fuga de datos y Pipeline"))
    s += para("Un estandarizador ajustado con <b>todas</b> las filas conoce la media y desviación de la prueba: eso es fuga. Lo correcto es <b>fit con entrenamiento</b> y <b>transform</b> en el resto. "
              "Un <b>Pipeline</b> lo hace solo, y dentro de K-Fold repite el proceso en cada pliegue.")
    s.append(ejemplo("un atípico en la prueba", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Entrenamiento: 10, 12, 14 · media y desviación", "media = 12; √((4 + 0 + 4)/3)", "12 · 1.633"],
               ["Prueba: 30 · z con estadísticos del entrenamiento (correcto)", "(30 − 12)/1.633", "z = 11.0 (atípico claro)"],
               ["z con estadísticos de las 4 filas (fuga)", "media 16.5 · desviación 7.92 → (30 − 16.5)/7.92", "z = 1.70 (parece normal)"]],
              [66 * mm, 66 * mm, CONTENT_W - 132 * mm - 18]),
        P("La fuga ocultó el atípico: el modelo «ya vio» la prueba. Con datos de puro ruido, seleccionar variables con todas las filas dio ≈ 0.79 de exactitud; dentro del Pipeline, ≈ 0.51 (azar).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(table([["Quiero…", "Código"],
                    ["Escalar y modelar sin fuga", "make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X_train, y_train)"],
                    ["Tratar columnas por tipo", "ColumnTransformer([('num', pipe_num, cols_num), ('cat', pipe_cat, cols_cat)])"],
                    ["Validar con Pipeline", "cross_val_score(pipe, X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=42))"],
                    ["Evaluar una vez", "pipe.fit(X_dev, y_dev); pipe.predict(X_test)"]],
                   [44 * mm, CONTENT_W - 44 * mm]))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook (B3) y el Bloque 8 del cuaderno",
                   [P("Reproduce el experimento de ruido con y sin Pipeline y explica con una frase por qué difieren.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(recuerda(["Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A y B del notebook → Bloques 5–8 → Partes C y D → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/C30_Apuntes_PreparacionML.pdf", s, "Apuntes · Clase 30 · Preparación para Machine Learning")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#7C3AED">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C30))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 30 · Preparación para Machine Learning",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C30))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: precision = TP/(TP+FP) · recall = TP/(TP+FN) · F1 = 2PR/(P+R) · z = (x − media)/desviación.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."],
                          "cell", C30)], bg=SOFT, border=C30))
    s.append(Spacer(1, 2))
    s += bloque(1, "Tipos de variables", "★", None, [
        "Clasifica: (a) número de cuotas pagadas, (b) comuna, (c) saldo de la cuenta, (d) segmento «bronce/plata/oro», (e) ¿tiene tarjeta? Sí/No.",
        "¿Cuántas columnas genera el one-hot de «comuna» si hay 12 comunas distintas? ¿Y de la variable binaria (e)?"])
    s += bloque(2, "Calidad e imputación", "★★", None, [
        "Ingresos observados (M CLP): 2.1, 2.3, 2.4, 2.6, 12.0 y un faltante. Calcula media y mediana y elige con qué imputar.",
        "Elige media, mediana o moda: (a) edad, simétrica; (b) región; (c) gasto con atípicos. Una base de 600 filas tiene 220 sin dato de ingreso: ¿imputas?",
        "Identifica la dimensión que falla: (a) dos sistemas con direcciones distintas; (b) un 15 % sin teléfono; (c) edad = 190; (d) tarifa de hace 3 años."])
    s += bloque(3, "División de datos", "★", None, [
        "2.400 registros, división 70/15/15: ¿cuántos en cada conjunto? Y con 5.000 registros y 80/10/10.",
        "Un modelo tiene R² 0.97 en entrenamiento y 0.64 en validación. ¿Qué pasa y qué no debes hacer para «arreglarlo»?"])
    s += bloque(4, "Estandarización y regularización", "★★", None, [
        "Superficie: media 90, desviación 25. Calcula z para 140 y para 65. Habitaciones: media 3, desviación 1.5; z para 6.",
        "Ridge con α = 2: un coeficiente 0.5 por m² frente a 50 por centena de m². Calcula α·β² en cada caso. ¿Cuál se castiga más y por qué conviene estandarizar?",
        "¿Cuándo preferirías Lasso y cuándo Ridge?"])
    s += bloque(5, "K-Fold", "★★", "Puntajes R² de un K-Fold de 5 pliegues: 0.71, 0.78, 0.74, 0.69, 0.78.", [
        "Calcula el promedio y la desviación (ddof = 0).",
        "¿Cuántas filas valida y entrena cada pliegue con 500 filas? ¿Qué te dice la desviación y qué conjunto no se usa en el K-Fold?"])
    s += bloque(6, "Estratificación", "★★", None, [
        "800 casos con 4 % de fraude y prueba de 25 %. ¿Cuántos fraudes hay en total, en la prueba y en el entrenamiento si se estratifica?",
        "Sin estratificar, ¿puede la prueba quedar sin fraudes? Explica por qué esto es grave para el recall.",
        "¿Estratificar aumenta la cantidad de fraudes? Justifica."])
    s += bloque(7, "Métricas", "★★★", "1.000 casos: TP = 24, FP = 16, FN = 6, TN = 954.", [
        "Calcula accuracy, precision, recall y F1.",
        "Calcula el accuracy de un modelo que siempre predice «no fraude» y compáralo con el anterior.",
        "El costo de un fraude no detectado es 50 veces el de una falsa alarma. ¿Qué métrica priorizas y qué harías con el umbral?"])
    s += bloque(8, "Fuga de datos", "★★★", None, [
        "Entrenamiento: 50, 55, 60 (media 55, desviación 4.08). Prueba: 90. Calcula z correcto y z con las 4 filas (media 63.75, desviación 15.56).",
        "¿Fuga o no? (a) imputar la mediana con todo el dataset y luego dividir; (b) ajustar el scaler solo con entrenamiento; (c) elegir alpha mirando la prueba; (d) hacer K-Fold con un Pipeline.",
        "Escribe en una frase por qué un Pipeline evita la fuga en K-Fold."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 4))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 3)]

    s += sol("Bloque 1", [
        ["1", "(a) numérica discreta · (b) categórica nominal · (c) numérica continua · (d) categórica ordinal (hay orden) · (e) categórica binaria."],
        ["2", "Comuna: 12 columnas (una por categoría; con <font name='Mono'>drop='first'</font> serían 11). La binaria: 1 columna 0/1."]])
    s += sol("Bloque 2", [
        ["1", "Media = (2.1 + 2.3 + 2.4 + 2.6 + 12.0)/5 = 21.4/5 = 4.28. Mediana = 2.4. El 12.0 arrastra la media: se imputa con la mediana 2.4."],
        ["2", "(a) media · (b) moda · (c) mediana. 220 de 600 = 36.7 %, más de un tercio: no se imputa a ciegas; se revisa la recolección o se analiza con otro enfoque."],
        ["3", "(a) coherencia · (b) completitud · (c) exactitud · (d) frescura."]])
    s += sol("Bloque 3", [
        ["1", "2.400: 0.70 → 1.680 · 0.15 → 360 · 360. 5.000: 4.000 · 500 · 500."],
        ["2", "Brecha de 0.33: sobreajuste. No se «arregla» mirando la prueba (fuga); se simplifica el modelo, se regulariza o se consiguen más datos, decidiendo con la validación."]])
    s += sol("Bloque 4", [
        ["1", "z(140) = (140 − 90)/25 = 2.0 · z(65) = −25/25 = −1.0 · z(6) = (6 − 3)/1.5 = 2.0."],
        ["2", "0.5 por m²: 2·0.5² = 0.5. 50 por centena: 2·50² = 5.000. Se castiga 10.000 veces más el mismo efecto: sin escala común la penalización depende de las unidades; se estandariza."],
        ["3", "Lasso: sospecho muchas variables irrelevantes y quiero un modelo más simple (puede anular coeficientes). Ridge: casi todas aportan algo y quiero estabilizar sin eliminar."]])
    s += sol("Bloque 5", [
        ["1", "Promedio = 3.70/5 = 0.74. Desviaciones: −0.03, 0.04, 0, −0.05, 0.04 → cuadrados 0.0009 + 0.0016 + 0 + 0.0025 + 0.0016 = 0.0066; /5 = 0.00132 → desviación 0.036."],
        ["2", "Valida 100 y entrena 400 en cada pliegue. Una desviación de 0.036 sobre 0.74 indica estabilidad moderada. El conjunto de prueba no se usa en el K-Fold."]])
    s += sol("Bloque 6", [
        ["1", "0.04·800 = 32 fraudes · prueba 0.25·800 = 200 filas → 0.04·200 = 8 · entrenamiento 600 filas → 24."],
        ["2", "Sí, el azar puede dejar muy pocos o ninguno (en un ejemplo con 200 filas y 12 fraudes, 2.2 % de las particiones). Sin positivos el recall no se puede calcular."],
        ["3", "No: la cantidad total es la misma; solo se conserva la proporción en cada parte."]])
    s += sol("Bloque 7", [
        ["1", "Accuracy = (24 + 954)/1.000 = 0.978 · precision = 24/40 = 0.600 · recall = 24/30 = 0.800 · F1 = 2·0.6·0.8/1.4 = 0.96/1.4 = 0.686."],
        ["2", "Siempre «no fraude»: 970/1.000 = 0.970 (hay 30 fraudes). El modelo solo mejora 0.008 en accuracy, aunque detecta 24 de 30 fraudes."],
        ["3", "Recall: un fraude perdido cuesta 50 veces una falsa alarma. Se baja el umbral para subir el recall, aceptando más falsas alarmas, y el umbral se elige en validación."]])
    s += sol("Bloque 8", [
        ["1", "Correcto: (90 − 55)/4.08 = 8.57. Con fuga: (90 − 63.75)/15.56 = 1.69. La fuga esconde el atípico."],
        ["2", "(a) fuga · (b) no · (c) fuga · (d) no."],
        ["3", "En cada pliegue el Pipeline hace fit solo con el entrenamiento de ese pliegue y transform en el de validación; nada del pliegue de validación influye en los pasos aprendidos."]])
    build("apuntes/C30_Ejercicios_Cuaderno_PreparacionML.pdf", s, "Ejercicios para el cuaderno · Clase 30")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
