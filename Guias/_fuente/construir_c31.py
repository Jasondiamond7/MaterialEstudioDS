"""Clase 31 · Introducción al Machine Learning: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C31 = colors.HexColor("#4F46E5")
C31_BG = colors.HexColor("#EEF2FF")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + apunte + notebook y caso de la profesora Laura Sánchez"


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado):
    """figuras: lista de (archivo, ancho_relativo, pie)."""
    s.append(band(f"Clase 31 · {titulo}", FUENTE, C31))
    s.append(Spacer(1, 5))
    s.append(box([P(idea, "big")], bg=SOFT, border=C31))
    s.append(Spacer(1, 4))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C31))
    s.append(Spacer(1, 2))
    for f, w, pie in figuras:
        s.append(img(f, width=CONTENT_W * w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 3))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF)]
    if py:
        left += [Spacer(1, 4), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C31), Spacer(1, 4),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER, width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


M = lambda t: f"<font name='Mono'>{t}</font>"


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 31 · Introducción al Machine Learning",
                  "Profesora Laura Sánchez · bloque de Machine Learning · sigue a Preparación para ML (C30) y precede a Reducción de dimensionalidad (C32)", NAVY))
    s.append(Spacer(1, 7))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("En lugar de escribir a mano las reglas de decisión, se le entregan <b>ejemplos</b> a un modelo y él ajusta sus parámetros. "
                    "Según lo que se tenga, el problema cae en una familia: con <b>etiqueta</b> (<b>supervisado</b>: clasificación o regresión), sin etiqueta "
                    "(<b>no supervisado</b>: clustering o reducción de dimensionalidad) o con un entorno y recompensas (<b>refuerzo</b>). El ciclo es siempre el mismo: "
                    "datos, modelo, <b>evaluación del error</b> y mejora. «No siempre se necesita inteligencia artificial: quieren un Ferrari para ir a comprar pan a la esquina.»", "body")],
                 bg=C31_BG, border=C31))
    s.append(Spacer(1, 5))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c31_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Regresión lineal (Clases 17–19) y logística (Clases 20–21): aquí aparecen como «modelos de ML»",
                     "Train / validación / prueba, Pipeline, Ridge y Lasso, accuracy / precision / recall / F1 (Clase 30)",
                     "Imputar con media o mediana (Clase 30), ahora dentro de un Pipeline",
                     "Distancia euclidiana y promedio (para K-Means); la idea de varianza (para PCA)"], "cell", C31)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Reglas fijas</b> → <b>ciclo del ML</b> → <b>clasificación</b> → <b>regresión</b> → <b>K-Means y PCA</b> → <b>ajuste y regularización</b> → <b>trampa del accuracy</b> → <b>elegir el enfoque</b>",
                      "Datos: <b>Club Activo</b>, 800 clientes simulados (el notebook usa <font name='Mono'>np.random.seed(42)</font>; 20.9 % cancela) y la <b>base real de 3.150 clientes</b> "
                      "de la actividad de repaso (15.7 % de fuga), que sirve para el desafío «supervisado o no supervisado»",
                      "Pregunta guía: ¿qué tipo de problema es, qué datos tengo y cómo sabré que el modelo sirve?"], "cell", C31)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Puentes con las clases vecinas</b>", "cellb"),
                  P("<b>C30 → C31</b>: lo que allí se vio por piezas (división, regularización, métricas) aquí se aplica a un caso completo. "
                    "<b>C31 → C32</b>: PCA aparece como «resumen visual» de 3 variables (58.2 % + 33.6 %); la Clase 32 explica cómo se calcula, cuántos componentes elegir y qué es t-SNE. "
                    "Para no repetir: aquí solo se usa PCA para <i>mirar</i> segmentos, no para modelar.", "cell")], bg=SOFT, border=C31))
    s.append(PageBreak())

    topic_page(
        s, "Clasificación: la regresión logística",
        "Si lo que se predice es una <b>categoría</b> (¿cancela, sí o no?), el problema es de <b>clasificación</b>. La regresión logística calcula un <b>puntaje</b> (logit) "
        "y lo convierte en <b>probabilidad</b> con la función sigmoide; el umbral decide la clase.",
        [("Puntaje y probabilidad", "logit = b₀ + b₁x₁ + … ; p = 1 / (1 + e<super>−logit</super>). Puntaje 0 equivale a p = 0.5; cada unidad adicional de x<sub>j</sub> multiplica las <i>odds</i> por e<super>b<sub>j</sub></super>."),
         ("Aprende con entrenamiento, se juzga con prueba", "Flujo de la Clase 30: dividir (estratificado), ajustar con <font name='Mono'>fit</font> solo en entrenamiento, evaluar en datos que el modelo nunca vio."),
         ("class_weight='balanced'", "Pondera cada clase de forma inversa a su frecuencia (aquí 0.63 y 2.39): el modelo se vuelve «alarmista», gana recall y pierde precisión. Es una decisión de modelado, no un arreglo gratuito."),
         ("El umbral es una decisión de negocio", "Con 0.5 se detecta el 52 %; subirlo da más precisión y menos recall. Se elige según el costo de cada error.")],
        [("c31_logistica.png", 0.92,
          "Izquierda: los 240 clientes de prueba sobre la sigmoide (el color real no se usó para ubicarlos). Centro: precisión y recall según el umbral. "
          "Derecha: frontera de decisión con solo antigüedad y quejas (una recta, porque el logit es lineal).")],
        ["p = 1 / (1 + e<super>−z</super>)",
         "z = b₀ + b₁x₁ + b₂x₂ + …",
         "odds = p / (1 − p) = e<super>z</super>"],
        ["Pipeline([('imp', ColumnTransformer(...)),",
         "          ('esc', StandardScaler()),",
         "          ('lr', LogisticRegression())])"],
        ["Prueba de 240 clientes (20.8 % cancela): accuracy 73.3 %, precisión 39.4 %, recall 52.0 %, F1 44.8 %, AUC 0.685",
         "Coeficientes (estandarizados): quejas +1.08, antigüedad −0.80, uso −0.22, edad +0.08; intercepto −0.39",
         "Sin class_weight: precisión 69.2 %, recall 18.0 %"],
        ["El 73.3 % de accuracy es <b>menor</b> que el 79.2 % de «nadie cancela». Con un AUC de 0.685 el modelo es modesto, no «una moneda» ni «excelente»."])

    topic_page(
        s, "Regresión: predecir un número",
        "Si lo que se predice es un <b>valor continuo</b> (¿cuánto gastará?), el problema es de <b>regresión</b>. El modelo ajusta una función de las variables y se evalúa con el "
        "<b>error</b> entre lo predicho y lo real (MSE, RMSE, R²).",
        [("Métricas de error", "MSE = promedio de errores²; RMSE = su raíz, en las unidades del gasto (miles de CLP); R² = 1 − SSE/SST: la proporción de la variación explicada."),
         ("Redescubrir una regla", "Los datos se generaron con gasto = 25 + 0.9·antigüedad + 3.2·uso + ruido. El modelo, sin conocer la fórmula, aprende 26.8, 0.93 y 2.98."),
         ("R² moderado no es falla", "Con ruido (sd 12) sobre un gasto de sd 19, un R² de 0.55 es lo esperable: el techo lo pone el ruido, no el modelo."),
         ("Mismo flujo, otra métrica", "En clasificación, matriz de confusión; en regresión, error cuadrático y R². Dividir y evaluar en datos no vistos no cambia.")],
        [("c31_regresion.png", 0.92,
          "Izquierda: predicción vs. gasto real en la prueba (R² 0.55, RMSE 12.0). Centro: coeficientes ocultos y aprendidos. Derecha: los residuos se centran en 0 (sd ≈ 12).")],
        ["MSE = (1/n) Σ (yᵢ − ŷᵢ)²",
         "RMSE = √MSE",
         "R² = 1 − SSE / SST"],
        ["LinearRegression().fit(X_train, y_train)",
         "mean_squared_error(y_test, pred)",
         "r2_score(y_test, pred)"],
        ["MSE 144.9 · RMSE 12.0 · R² 0.548 en la prueba (0.572 en entrenamiento)",
         "Coeficientes: antigüedad 0.929 (real 0.9), uso 2.981 (real 3.2), intercepto 26.76 (real 25)",
         "El gasto medio es 63.2 miles de CLP: un RMSE de 12 es ≈ 19 % del gasto medio"],
        ["«Redescubrir» la fórmula es posible solo porque los datos se simularon con ella; en un caso real no existe una fórmula verdadera que comparar."])

    topic_page(
        s, "Sin etiquetas: K-Means y PCA",
        "En <b>aprendizaje no supervisado</b> no hay respuesta correcta; se busca estructura. <b>K-Means</b> forma <b>grupos</b> alrededor de centroides; <b>PCA</b> "
        "<b>resume</b> variables correlacionadas en pocos componentes. PCA no forma grupos: solo cambia el sistema de ejes.",
        [("K-Means paso a paso", "Elegir k centroides → asignar cada punto al más cercano → recalcular cada centroide como la media de su grupo → repetir hasta que nada cambie. Minimiza la suma de distancias² a los centroides (inercia)."),
         ("Escalar antes de agrupar", "Se usan distancias: la variable de mayor escala domina. Sin estandarizar, los segmentos cambian (ARI 0.39 respecto de los estandarizados)."),
         ("Elegir k", "No hay un k verdadero: se mira el codo de la inercia, la silueta y, sobre todo, si los grupos tienen sentido para el negocio."),
         ("Interpretar y validar", "Los grupos se leen mirando el perfil de cada uno. Aunque cancelo no se usó, un grupo (antiguos y de alto gasto) tiene 7 % de cancelación frente a 21–24 %.")],
        [("c31_kmeans.png", 0.97,
          "K-Means con antigüedad, uso y gasto estandarizados (Club Activo). Izquierda: dos de las tres variables; segundo: los mismos clientes proyectados con PCA (58.2 % + 33.6 % = 91.8 % de la varianza); "
          "tercero: codo y silueta; derecha: perfil de cada segmento y su tasa de cancelación.")],
        ["inercia = Σ ‖xᵢ − μ<sub>c(i)</sub>‖²",
         "μ<sub>c</sub> = media de los puntos del grupo c",
         "silueta ∈ [−1, 1]; más alta = grupos más separados"],
        ["KMeans(n_clusters=3, n_init=10,",
         "       random_state=42).fit_predict(Z)",
         "silhouette_score(Z, etiquetas)"],
        ["Segmentos de 252, 439 y 109 clientes; silueta 0.348 (k = 2), 0.338 (k = 3), 0.327 (k = 4): sin un k claro",
         "Segmento 2: antigüedad +1.99 sd, gasto +1.48 sd, cancela 7 %; segmentos 0 y 1: 21 % y 24 %",
         "Base real, 6 variables de uso: k = 3, silueta 0.473; fuga de 24.7 %, 0.0 % y 2.3 % en la prueba"],
        ["«Con dos de las tres variables llegamos a los mismos clústeres» no es exacto: los componentes mezclan las tres variables, y los clústeres los formó K-Means, no PCA."])

    topic_page(
        s, "Ajuste del modelo: subajuste, sobreajuste y regularización",
        "Un modelo demasiado simple <b>no aprende</b> el patrón (subajuste); uno demasiado complejo <b>memoriza el ruido</b> (sobreajuste). Se detecta comparando "
        "entrenamiento y prueba. La <b>regularización</b> penaliza coeficientes grandes: Ridge los encoge, Lasso puede llevarlos a cero.",
        [("Subajuste y sobreajuste", "Subajuste: mal incluso en entrenamiento. Sobreajuste: muy bien en entrenamiento y mucho peor en prueba. La causa es la complejidad frente a los datos disponibles."),
         ("Ridge (L2)", "Penaliza Σβ²: encoge todos los coeficientes, rara vez los anula."),
         ("Lasso (L1)", "Penaliza Σ|β|: lleva a cero los coeficientes poco útiles (selección automática).")],
        [("c31_ajuste.png", 0.7,
          "Tres ajustes sobre la misma nube (60 puntos): grado 1, 2 y 15, y el R² de entrenamiento y prueba según el grado. La X roja es lo que ocurre sin escalar."),
         ("c31_regulariza.png", 0.7,
          "Caminos de coeficientes de Lasso y Ridge al subir alpha (antigüedad y uso en azul; ruido en rojo) y R² de prueba. Con alpha moderado el R² no cambia (0.544 → 0.552 con Lasso 0.5).")],
        ["Ridge: min Σ(y − ŷ)² + α Σ β²",
         "Lasso: min Σ(y − ŷ)² + α Σ |β|",
         "si las columnas son ortonormales: Ridge = β/(1+α); Lasso = signo(β)·máx(|β| − α, 0)"],
        None,
        ["Grado 1: R² 0.52 / 0.10 · grado 2: 0.80 / 0.51 · grado 15 (escalado): 0.90 / −4.79 · (sin escalar: 0.30 / −0.56)",
         "Lasso (alpha 0.5) anula quejas, ruido 1 y ruido 2 y deja edad en 0.22; Ridge (alpha 5) apenas las encoge",
         "Con alpha enorme los dos tienden a predecir la media (R² ≈ 0)"],
        ["En el notebook de la clase el grado 15 sin escalar da R² de entrenamiento 0.30 (no «altísimo»): hay un problema numérico, no solo sobreajuste."])

    topic_page(
        s, "Evaluar bien: la trampa del accuracy",
        "Con clases desbalanceadas, un modelo que siempre responde «no pasa nada» obtiene un accuracy alto y un recall de 0. Hay que reportar la <b>matriz de confusión</b> "
        "y mirar <b>precisión</b>, <b>recall</b> y <b>F1</b> junto con una <b>línea base</b>.",
        [("Matriz de confusión", "Filas = real, columnas = predicho: VN y FP arriba, FN y VP abajo (convención de scikit-learn). Todas las métricas salen de estas cuatro cifras."),
         ("Accuracy vs. recall", "Accuracy = (VP + VN)/total. Recall = VP/(VP + FN): de los que sí cancelan, cuántos se detectan. Precisión = VP/(VP + FP)."),
         ("Siempre comparar con la línea base", "En la base real, «nadie se fuga» ya logra 84.3 % de accuracy: un 88.5 % no es tan impresionante como parece."),
         ("El costo manda", "Si dejar escapar un caso cuesta mucho más que una falsa alarma, se prioriza el recall (y se acepta menos precisión).")],
        [("c31_accuracy.png", 0.93,
          "Izquierda: 1.000 casos con 5 % de riesgo y un modelo «perezoso». Centro: base real (prueba de 788 clientes) con la línea base, la logística y la logística balanceada. "
          "Derecha: matriz de confusión del modelo de Club Activo.")],
        ["precisión = VP / (VP + FP)",
         "recall = VP / (VP + FN)",
         "F1 = 2·precisión·recall / (precisión + recall)"],
        ["confusion_matrix(y_test, pred)",
         "DummyClassifier(strategy='most_frequent')",
         "classification_report(y_test, pred)"],
        ["Club Activo: VN 150, FP 40, FN 24, VP 26 → precisión 26/66 = 39.4 %, recall 26/50 = 52.0 %",
         "Base real: logística 88.5 % acc, 71.4 % prec, 44.4 % recall, F1 0.547; balanceada 82.1 %, 46.4 %, 87.9 %, F1 0.607",
         "Mismo AUC (0.908 y 0.910): cambia el punto de operación, no la capacidad de ordenar"],
        ["El F1 y el accuracy no se pueden leer como un ranking único: depende de qué error cuesta más."])

    # ------------------------------------------------------------ elegir enfoque
    s.append(band("Cómo elegir el enfoque, ML vs. Deep Learning y el flujo honesto", "Primero el problema, luego el algoritmo", NAVY))
    s.append(Spacer(1, 6))
    s.append(table([["Pregunta", "Si la respuesta es…", "Enfoque a considerar"],
                    ["¿Tengo ejemplos con la respuesta correcta?", "sí, y es una categoría", "Clasificación (supervisado)"],
                    ["", "sí, y es un número", "Regresión (supervisado)"],
                    ["¿No tengo etiquetas?", "quiero descubrir grupos", "Clustering, p. ej. K-Means"],
                    ["", "quiero simplificar muchas variables", "Reducción de dimensionalidad, p. ej. PCA"],
                    ["¿Decisiones secuenciales con entorno?", "ensayo y error con recompensa", "Aprendizaje por refuerzo"],
                    ["¿Imágenes, audio, texto, gran volumen?", "datos complejos", "Deep Learning (redes neuronales)"]],
                   [58 * mm, 56 * mm, CONTENT_W - 114 * mm]))
    s.append(Spacer(1, 4))
    s.append(table([["Criterio", "Machine Learning clásico", "Deep Learning"],
                    ["Volumen de datos", "suele bastar con miles de registros", "suele necesitar muchísimos más"],
                    ["Hardware", "CPU estándar", "frecuentemente GPU"],
                    ["Extracción de atributos", "manual, con conocimiento del negocio", "automática, por capas ocultas"],
                    ["Interpretabilidad", "en general mayor (la regresión lo es; otros modelos no)", "en general menor"],
                    ["Casos de uso", "datos tabulares, scoring", "visión, audio, lenguaje, IA generativa"]],
                   [44 * mm, 66 * mm, CONTENT_W - 110 * mm]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>El Pipeline de la clase, honesto</b>", "cellb"),
                  P("Pipeline([('imp', ColumnTransformer(...media / mediana...)), ('esc', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced'))]).fit(X_train, y_train)", "mono"),
                  P("El notebook de la clase imputa (media, mediana) y escala <b>antes</b> de dividir: es una fuga leve. Dentro de un Pipeline, cada paso aprende solo con entrenamiento y "
                    "las métricas de la clase no cambian (73.3 / 39.4 / 52.0).", "cell")],
                 bg=C31_BG, border=C31))
    s.append(Spacer(1, 4))
    checks = [("Defino qué quiero predecir y qué decisión mejora", "etiqueta o no, categoría o número"),
              ("Divido antes de imputar, escalar o ajustar", "todo lo que aprende de los datos va en un Pipeline"),
              ("Comparo contra una línea base", "«siempre la clase mayoritaria» o la regla fija actual"),
              ("Elijo la métrica según el costo del error", "no solo accuracy; matriz de confusión completa"),
              ("Escalo antes de usar distancias o penalizaciones", "K-Means, Ridge, Lasso"),
              ("Interpreto los segmentos con su perfil", "los grupos de K-Means no traen nombre ni etiqueta")]
    s.append(table([["", "Checklist del flujo", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 74 * mm, CONTENT_W - 82 * mm]))
    s.append(PageBreak())

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene corregir (verificado con código)", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Diapositivas, apunte y notebook</b>", "cellb"),
                  bullets(["<b>Nota del notebook sobre K-Means</b>: «el gasto mensual es la variable que más distingue al Segmento 2 y queda invisible». El que más lo distingue es la <b>antigüedad</b> (+1.99 sd; gasto +1.48) y sí se ve en el eje x. Los segmentos 0 y 1 se separan por uso (+0.99 frente a −0.56), también visible. Además se espera un segmento de <b>mayor</b> cancelación; ocurre lo contrario: el 2 cancela 7 % frente a 21 % y 24 %.",
                           "<b>Sobreajuste con grado 15</b>: el texto habla de R² de entrenamiento «altísimo». Con <font name='Mono'>PolynomialFeatures(15)</font> sin escalar el R² de entrenamiento es 0.30 (menor que el 0.80 del grado 2) y el de prueba −0.56: la matriz está mal condicionada. Escalando, 0.90 y −4.79. El diagnóstico es correcto; las cifras de la celda no lo muestran.",
                           "<b>K-Means (diapositiva 8)</b>: «encuentra centros (K) que minimizan la distancia entre los puntos de cada grupo». K es el número de grupos; se minimiza la suma de distancias² de cada punto a <b>su centroide</b>. Y PCA «elimina ruido visual»: descarta direcciones de poca varianza, que no es lo mismo que ruido.",
                           "<b>Diapositiva 12</b>: «Modelos neuronales requieren conjuntos… divididos en entrenamiento, validación y prueba»: la división vale para <b>cualquier</b> modelo. <b>Diapositiva 13</b>: ML = «cajas blancas» y DL = «millones de registros» son tendencias, no reglas (un boosting no es caja blanca; hay redes con pocos datos).",
                           "<b>Cierre del notebook</b>: «Próxima sesión (32): redes neuronales». La Clase 32 es reducción de dimensionalidad (PCA y t-SNE). Y <font name='Mono'>recomendar_enfoque</font> devuelve un solo enfoque como si fueran excluyentes.",
                           "<b>Apunte</b>: dice que el modelo de clasificación usó «antigüedad, uso, edad, quejas y gasto» y que se predijo «un cliente nuevo»; usa cuatro variables (sin gasto) y predice el conjunto de prueba. En clase salió 22.8 % de cancelación; el notebook final da 20.9 %."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets(["«Precisión 51 % es como lanzar una moneda» y «recall 52 %… lo mismo que una moneda»: la comparación es con la <b>prevalencia</b> (20.8 %), no con 50 %. Un clasificador al azar tiene precisión igual a la prevalencia; este logra 39.4 % y AUC 0.685 (azar = 0.5): modesto, pero no es una moneda.",
                           "«Esto es 149 verdaderos positivos»: la celda superior izquierda de scikit-learn son los <b>verdaderos negativos</b> (150 en el notebook final); los verdaderos positivos son 26.",
                           "«Quitar quejas_soporte desbalanceó el modelo»: no cambia el balance de clases (20.8 %). Efecto real: accuracy 73.3 → 52.5 %, precisión 39.4 → 24.6 %, recall 52 → 62 %, F1 44.8 → 35.2 % y AUC 0.685 → 0.575. Subió solo el recall (lo dijo bien una participante).",
                           "<b>90/10 «corre el riesgo de sobreajuste»</b>: un 10 % de prueba da una estimación ruidosa del error; el sobreajuste depende de la complejidad, no del porcentaje.",
                           "<b>K-Means</b> «toma un punto, ve los más cercanos y mueve el centro»: no; asigna cada punto a su centroide más cercano y recalcula medias. <b>PCA «separa tres clústeres»</b>: los formó K-Means; PCA proyecta. «91.8 % de la <i>información</i>» es 91.8 % de la <i>varianza</i>.",
                           "<b>Caso de repaso, pregunta 1.2</b>: predecir «cuántos meses le quedan antes de cancelar» no es sin más una regresión lineal; es un problema de tiempo hasta el evento con datos censurados (los que aún no cancelan), que se trata con análisis de supervivencia."],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper confundió KYC (KIC), K-Means (camins, Kamins, Cummings), accuracy (cura), recall (recol), PCA (PSA, IPCA), CAPTCHA (CAPCHAT), «redes neuronales» (jornales) y «Machine Learning» (Machilas, Machete). "
                    "El volumen cayó a ≈ −65 dB desde ≈ 35:00 y a ≈ −78 dB desde ≈ 85:00 (se transcribió una copia amplificada); el recreo (≈ 0:59:00) no está en el video. Frases dudosas: «la gragasa» (omitida), «un platón de riesgo».", "cell")],
                 bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"),
                  bullets(["Clasifico un problema como clasificación, regresión, clustering, reducción de dimensionalidad o refuerzo",
                           "Calculo a mano una probabilidad con la sigmoide y una iteración de K-Means",
                           "Leo una curva de entrenamiento / prueba y explico qué hacen Ridge y Lasso",
                           "Calculo accuracy, precisión, recall y F1 y comparo con la línea base",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C31_Guia_IntroML.pdf", s, "Clase 31 · Introducción al Machine Learning")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#4F46E5">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 31 · Introducción al Machine Learning",
                  "Para leer antes de los ejercicios · lectura de ~30 min", C31))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([
                      ("El tipo de problema decide el modelo", "Etiqueta categórica → clasificación; etiqueta numérica → regresión; sin etiqueta → clustering o reducción; recompensas → refuerzo."),
                      ("Probabilidad antes que clase", "La logística calcula un puntaje, la sigmoide lo vuelve probabilidad y el umbral decide."),
                      ("Siempre se mide el error en datos nuevos", "Entrenar con un conjunto, evaluar con otro; todo lo que se aprende, solo con entrenamiento."),
                      ("K-Means: asignar y promediar", "Cada punto al centroide más cercano; cada centroide pasa a ser la media de su grupo; se repite."),
                      ("Complejidad y penalización", "Mucha complejidad memoriza ruido; Ridge encoge y Lasso anula coeficientes."),
                      ("Accuracy no basta", "Con clases desbalanceadas se mira la matriz de confusión, el recall, la precisión y una línea base."),
                  ], C31)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C31_IntroML.ipynb</font> o el cuaderno "
               "<font name='Mono'>C31_Ejercicios_Cuaderno_IntroML.pdf</font>.", "small"))

    s.append(h("1", "Del problema a la tarea de ML"))
    s += para("Antes de elegir un algoritmo hay que preguntarse qué se quiere predecir y con qué datos. Una <b>regla fija</b> dice qué hacer ante ciertas condiciones; un "
              "<b>modelo</b> ajusta sus parámetros con ejemplos. Si hay variable objetivo, es <b>supervisado</b> (<b>clasificación</b> si es categoría, <b>regresión</b> si es número); si no, "
              "<b>no supervisado</b> (<b>clustering</b> para formar grupos, <b>reducción de dimensionalidad</b> para resumir variables); y con un entorno que da recompensas, <b>refuerzo</b>.")
    s.append(ejemplo("cuatro situaciones de una tienda", [
        pasos([["Situación", "Tipo de problema", "Por qué"],
               ["Saber si una compra online es fraude (hay historial marcado)", "supervisado · clasificación", "la etiqueta es sí / no"],
               ["Estimar cuánto dinero gastará un cliente en diciembre", "supervisado · regresión", "la salida es un número"],
               ["Descubrir tipos de clientes sin categorías previas", "no supervisado · clustering", "no hay etiqueta"],
               ["Un sistema prueba precios y aprende de las ventas", "refuerzo", "acción → recompensa → ajuste"]],
              [78 * mm, 44 * mm, CONTENT_W - 122 * mm - 18]),
        P("Y si el problema se resuelve con una resta o una tabla de reglas que ya funciona, no hace falta ML: «no uses IA porque está de moda».", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c31_esquema.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Clasifica los casos, usa <font name='Mono'>recomendar_enfoque</font> con tu regla fija y pregúntate qué etiqueta tendrías que registrar hoy para entrenar un modelo mañana.", "cell")]))

    s.append(h("2", "Regresión logística y la sigmoide"))
    s += para("La regresión logística calcula un <b>puntaje</b> z = b₀ + b₁x₁ + … y lo transforma en una probabilidad con <b>p = 1/(1 + e<super>−z</super>)</b>. "
              "Si z = 0, p = 0.5; z grande y positivo acerca p a 1; z muy negativo, a 0. Con un umbral de 0.5 se predice «cancela» cuando z &gt; 0. "
              "Cada coeficiente multiplica las <i>odds</i> (p/(1−p)) por e<super>b</super> por cada unidad adicional de la variable.")
    s.append(ejemplo("z = −1.0 + 1.5·quejas − 0.2·uso", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Cliente con 2 quejas y 6 usos: puntaje", "−1.0 + 1.5·2 − 0.2·6", "z = 0.8"],
               ["Su probabilidad de cancelar", "1/(1 + e<super>−0.8</super>) = 1/(1 + 0.449)", "p = 0.690"],
               ["Cliente con 0 quejas y 8 usos", "z = −1.0 + 0 − 1.6 = −2.6 → 1/(1 + e<super>2.6</super>)", "p = 0.069"],
               ["Efecto de una queja más", "e<super>1.5</super>", "odds × 4.48"],
               ["Umbral 0.5: clases predichas", "z &gt; 0 ⇔ p &gt; 0.5", "cancela / se queda"]],
              [66 * mm, 64 * mm, CONTENT_W - 130 * mm - 18]),
        P("Un cliente con p = 0.69 «cancela» con umbral 0.5, pero con umbral 0.7 ya no. Bajar el umbral detecta más cancelaciones (más recall) y genera más falsas alarmas.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c31_logistica.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook (B1 y B2) y el Bloque 2 del cuaderno",
                   [P("Ajusta la logística dentro de un Pipeline, imprime <font name='Mono'>predict_proba</font> de dos clientes y comprueba a mano que 1/(1 + e<super>−z</super>) coincide.", "cell")]))

    s.append(h("3", "Matriz de confusión y métricas"))
    s += para("Se cuentan cuatro tipos de resultado: verdaderos positivos (VP), falsos positivos (FP), falsos negativos (FN) y verdaderos negativos (VN). "
              "<b>Accuracy</b> = (VP+VN)/total; <b>precisión</b> = VP/(VP+FP); <b>recall</b> = VP/(VP+FN); <b>F1</b> = media armónica de ambas. "
              "Con clases muy desiguales el accuracy engaña: hay que compararlo con la <b>línea base</b> «siempre la clase mayoritaria».")
    s.append(ejemplo("500 transacciones, 20 fraudes", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Modelo: VP 12, FN 8, FP 30, VN 450 · accuracy", "(12 + 450)/500", "92.4 %"],
               ["Precisión", "12/(12 + 30)", "28.6 %"],
               ["Recall", "12/(12 + 8)", "60.0 %"],
               ["F1", "2·0.286·0.60/(0.286 + 0.60)", "0.387"],
               ["Línea base «nunca hay fraude»: accuracy · recall", "480/500 · 0/20", "96.0 % · 0 %"]],
              [72 * mm, 62 * mm, CONTENT_W - 134 * mm - 18]),
        P("El modelo «pierde» en accuracy contra la línea base (92.4 % frente a 96 %) pero es el único que detecta fraudes. En la base real de churn, «nadie se fuga» ya da 84.3 % de accuracy.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c31_accuracy.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook (D3) y el Bloque 6 del cuaderno",
                   [P("Usa <font name='Mono'>DummyClassifier</font> como línea base, imprime la matriz de confusión con <font name='Mono'>confusion_matrix</font> y calcula a mano las cuatro métricas.", "cell")]))

    s.append(h("4", "K-Means a mano: una iteración"))
    s += para("K-Means necesita k, un punto de partida (los centroides) y distancias. Cada iteración: (1) <b>asignar</b> cada punto al centroide más cercano; (2) <b>recalcular</b> cada centroide como la media de sus puntos. "
              "Se detiene cuando las asignaciones ya no cambian. Se compara con la <b>inercia</b>: suma de distancias² de cada punto a su centroide. Y se escala antes: es un método basado en distancias.")
    s.append(ejemplo("seis puntos (antigüedad, uso) y k = 2", [
        P("A(1, 2), B(2, 1), C(2, 4), D(7, 7), E(8, 9), F(9, 6); centroides iniciales c1 = A y c2 = F.", "cell"),
        pasos([["Pregunta", "Cálculo (distancia²)", "Resultado"],
               ["B a c1 y c2", "(1² + 1²) = 2 · (7² + 5²) = 74", "B → c1"],
               ["C a c1 y c2", "(1² + 2²) = 5 · (7² + 2²) = 53", "C → c1"],
               ["D, E a c1 y c2", "D: 61 vs 5 · E: 98 vs 10", "D, E → c2"],
               ["Nuevo c1", "((1 + 2 + 2)/3, (2 + 1 + 4)/3)", "(1.67, 2.33)"],
               ["Nuevo c2", "((7 + 8 + 9)/3, (7 + 9 + 6)/3)", "(8.00, 7.33)"]],
              [46 * mm, 70 * mm, CONTENT_W - 116 * mm - 18]),
        P("Con los nuevos centroides las asignaciones no cambian: el algoritmo terminó. En un caso real se repite con varios puntos de partida (<font name='Mono'>n_init</font>) y se queda con la menor inercia.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c31_kmeans.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook y el Bloque 3 del cuaderno",
                   [P("Reproduce una iteración con NumPy, compara la inercia antes y después y prueba K-Means con y sin <font name='Mono'>StandardScaler</font> (ARI 0.39 entre ambos).", "cell")]))

    s.append(h("5", "Subajuste, sobreajuste, Ridge y Lasso"))
    s += para("El <b>subajuste</b> aparece cuando el modelo falla incluso en entrenamiento; el <b>sobreajuste</b>, cuando acierta en entrenamiento y falla en prueba. "
              "La <b>regularización</b> añade al error una penalización: Ridge (L2) suma α·Σβ² y Lasso (L1), α·Σ|β|. En el caso simple de columnas ortonormales, "
              "Ridge deja β/(1 + α) y Lasso aplica un <i>umbral suave</i>: signo(β)·máx(|β| − α, 0). Por eso Lasso anula coeficientes pequeños y Ridge no.")
    s.append(ejemplo("coeficientes de una regresión con α = 1", [
        pasos([["Variable", "OLS", "Ridge β/(1 + 1)", "Lasso máx(|β| − 1, 0)"],
               ["x₁", "5.0", "2.50", "4.0"],
               ["x₂", "−1.4", "−0.70", "−0.4"],
               ["x₃", "0.6", "0.30", "0 (anulado)"],
               ["Suma de |coef.|", "7.0", "3.50", "4.4"]],
              [30 * mm, 24 * mm, 46 * mm, CONTENT_W - 100 * mm - 18]),
        P("Con α = 1, Ridge divide todo por 2 (ninguno se anula); Lasso resta 1 y deja en cero x₃ (|0.6| &lt; 1). Un α demasiado grande hace que los dos converjan a predecir la media (R² ≈ 0).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c31_ajuste.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(img("c31_regulariza.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook (D1 y D2) y los Bloques 4 y 5 del cuaderno",
                   [P("Ajusta polinomios de grado 1 a 15 con y sin escalar, dibuja el R² de entrenamiento y prueba, y compara los coeficientes de OLS, Ridge y Lasso con ruido añadido.", "cell")]))

    s.append(h("6", "Elegir el enfoque: ML o Deep Learning"))
    s += para("El Deep Learning es un subconjunto del ML basado en redes con varias capas: aprende representaciones directamente de datos complejos (imágenes, audio, texto), "
              "pero necesita más datos, más cómputo y es menos interpretable. Para datos tabulares con miles de filas, un modelo clásico suele bastar. "
              "El orden de preguntas: ¿hay etiqueta?, ¿es categoría o número?, ¿hay entorno con recompensas?, ¿es un problema de datos complejos?, ¿qué métrica demuestra que supera a la regla actual?")
    s.append(ejemplo("tres propuestas de la dirección", [
        pasos([["Propuesta", "Enfoque razonable", "Por qué"],
               ["Predecir ventas semanales con 3 años de historia tabular", "regresión (o series de tiempo)", "pocos miles de filas; ML clásico basta"],
               ["Clasificar 2 millones de fotos de productos", "Deep Learning", "datos complejos y gran volumen"],
               ["Calcular un descuento fijo según una tabla de reglas", "sin ML", "una regla ya resuelve el problema"]],
              [70 * mm, 46 * mm, CONTENT_W - 116 * mm - 18])]))
    s.append(Spacer(1, 3))
    s.append(recuerda(["Sesión 32 trata PCA y t-SNE en profundidad; aquí solo se usó PCA para mirar los segmentos.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–C del notebook → Bloques 5–7 → Parte D → ejercicio de cierre."]))
    build("apuntes/C31_Apuntes_IntroML.pdf", s, "Apuntes · Clase 31 · Introducción al Machine Learning")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#4F46E5">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C31))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 31 · Introducción al Machine Learning",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C31))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: p = 1/(1 + e<super>−z</super>) · accuracy = (VP+VN)/total · precisión = VP/(VP+FP) · recall = VP/(VP+FN) · F1 = 2PR/(P+R) · R² = 1 − SSE/SST · "
                           "Ridge: β/(1+λ) · Lasso: signo(β)·máx(|β|−λ, 0).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."],
                          "cell", C31)], bg=SOFT, border=C31))
    s.append(Spacer(1, 2))
    s += bloque(1, "Clasificar problemas", "★", "Indica el tipo de problema: clasificación, regresión, clustering, reducción de dimensionalidad, refuerzo o Deep Learning.", [
        "(a) Decidir si un correo es spam · (b) estimar los minutos que demorará un pedido · (c) agrupar tiendas según su perfil de ventas, sin categorías previas · "
        "(d) resumir 40 indicadores de una encuesta en 3 factores para graficarlos · (e) un robot aprende a caminar con premios por avanzar · (f) reconocer objetos en 2 millones de fotos.",
        "Para los problemas supervisados, nombra la variable objetivo y una métrica adecuada."])
    s += bloque(2, "La sigmoide a mano", "★★", "Modelo: z = −1.2 + 0.8·quejas − 0.1·uso. Cliente A: 3 quejas y 5 usos. Cliente B: 0 quejas y 10 usos.", [
        "Calcula el puntaje y la probabilidad de cancelar de A y de B.",
        "¿Cuánto se multiplican las odds por cada queja adicional?",
        "Con umbral 0.5 y luego con 0.3, ¿qué clase predice el modelo para cada cliente?"])
    s += bloque(3, "Una iteración de K-Means", "★★★", "Puntos (antigüedad, uso): P1(1, 2), P2(2, 4), P3(3, 3), P4(6, 5), P5(7, 8), P6(8, 6). Centroides iniciales c1 = (1, 1) y c2 = (8, 8).", [
        "Asigna cada punto con la distancia euclidiana al cuadrado y calcula la inercia.",
        "Recalcula los centroides y la nueva inercia. ¿Cambian las asignaciones?",
        "Si «antigüedad» estuviera en días (valores de 365 a 3.650) y «uso» en veces, ¿qué problema aparecería? ¿Cómo se evita?"])
    s += bloque(4, "Leer subajuste y sobreajuste", "★★", "R² de entrenamiento / prueba según el grado de un polinomio: grado 1: 0.52 / 0.50 · grado 2: 0.80 / 0.78 · grado 4: 0.84 / 0.74 · grado 8: 0.93 / 0.31 · grado 15: 0.99 / −0.60.", [
        "¿Qué grado muestra subajuste, cuál un ajuste adecuado y cuál sobreajuste? Justifica con la brecha.",
        "¿Qué grado elegirías y por qué no el de mayor R² de entrenamiento?",
        "Nombra dos acciones que reduzcan el sobreajuste del grado 15."])
    s += bloque(5, "Ridge frente a Lasso", "★★", "Coeficientes de una regresión sin penalizar con variables ortonormales: β = (4.0, −2.0, 0.8, 0.3). Penalización λ = 0.5.", [
        "Calcula los coeficientes de Ridge (β/(1+λ)) y de Lasso (signo(β)·máx(|β| − λ, 0)).",
        "¿Qué variables anula cada método? ¿Cuál usarías para «limpiar» variables sospechosas?",
        "Repite con λ = 1 solo para Lasso. ¿Cuántos coeficientes quedan distintos de cero?"])
    s += bloque(6, "Accuracy contra recall", "★★★", "2.000 transacciones, 40 fraudes. Modelo A: VP 10, FN 30, FP 20, VN 1.940. Modelo B: siempre «no fraude». Modelo C: VP 34, FN 6, FP 150, VN 1.810.", [
        "Calcula accuracy, precisión, recall y F1 de A, B y C.",
        "Ordena los modelos por accuracy y por recall. ¿Coinciden?",
        "Un fraude no detectado cuesta 100 y una falsa alarma cuesta 1. Calcula el costo total de cada modelo y elige."])
    s += bloque(7, "Métricas de regresión y elegir el enfoque", "★★", "Valores reales (60, 70, 80, 90) y predichos (62, 66, 83, 89).", [
        "Calcula los errores, el MSE, el RMSE y el R².",
        "Una empresa propone Deep Learning para predecir la demanda de 500 productos con 3 años de historia semanal en una planilla. ¿Qué le recomiendas y qué línea base usarías?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 4))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 3)]

    s += sol("Bloque 1", [
        ["1", "(a) clasificación · (b) regresión · (c) clustering · (d) reducción de dimensionalidad · (e) refuerzo · (f) Deep Learning (clasificación de imágenes, supervisado con gran volumen)."],
        ["2", "(a) y = spam / no spam: precisión, recall, F1 · (b) y = minutos: RMSE, R² · (f) y = objeto: accuracy y matriz de confusión. Los problemas (c), (d), (e) no tienen variable objetivo en el sentido supervisado."]])
    s += sol("Bloque 2", [
        ["1", "A: z = −1.2 + 2.4 − 0.5 = 0.7 → p = 1/(1 + e<super>−0.7</super>) = 0.668. B: z = −1.2 + 0 − 1.0 = −2.2 → p = 1/(1 + e<super>2.2</super>) = 0.100."],
        ["2", "e<super>0.8</super> = 2.23: cada queja multiplica las odds por ≈ 2.2."],
        ["3", "Umbral 0.5: A cancela, B se queda. Umbral 0.3: A cancela (0.668 ≥ 0.3), B se queda (0.100 &lt; 0.3). Bajar el umbral aumenta los positivos solo si hay clientes entre 0.3 y 0.5."]])
    s += sol("Bloque 3", [
        ["1", "Distancia² a c1 / c2: P1 1 / 85 · P2 10 / 52 · P3 8 / 50 · P4 41 / 13 · P5 85 / 1 · P6 74 / 4. Asignación: P1, P2, P3 → c1; P4, P5, P6 → c2. Inercia = 1 + 10 + 8 + 13 + 1 + 4 = 37."],
        ["2", "c1 = (2, 3); c2 = (7, 6.33). Nueva inercia: P1 2, P2 1, P3 1, P4 2.78, P5 2.78, P6 1.11 = 10.67. Las asignaciones no cambian: terminó."],
        ["3", "La variable en días domina la distancia: el agrupamiento solo reflejaría antigüedad. Se evita estandarizando (o llevando todo a una misma escala) antes de ejecutar K-Means."]])
    s += sol("Bloque 4", [
        ["1", "Subajuste: grado 1 (bajo en entrenamiento y en prueba, 0.52 / 0.50). Adecuado: grado 2 (0.80 / 0.78, brecha 0.02). Sobreajuste: grados 8 y 15 (brechas 0.62 y 1.59)."],
        ["2", "Grado 2: el R² de prueba es el más alto y la brecha es pequeña. El grado 15 tiene el mejor entrenamiento pero memoriza ruido."],
        ["3", "Más datos; menos complejidad (grado bajo); regularización (Ridge o Lasso con alpha elegido en validación); validación cruzada para escoger el grado."]])
    s += sol("Bloque 5", [
        ["1", "Ridge: (4.0, −2.0, 0.8, 0.3)/1.5 = (2.67, −1.33, 0.53, 0.20). Lasso: (3.5, −1.5, 0.3, 0)."],
        ["2", "Ridge no anula ninguno; Lasso anula el cuarto (|0.3| &lt; 0.5). Para limpiar variables sospechosas, Lasso."],
        ["3", "Con λ = 1: (3.0, −1.0, 0, 0): quedan 2 distintos de cero."]])
    s += sol("Bloque 6", [
        ["1", "A: accuracy 1.950/2.000 = 97.5 %; precisión 10/30 = 33.3 %; recall 10/40 = 25.0 %; F1 = 0.286. B: accuracy 1.960/2.000 = 98.0 %; precisión 0/0 (se define como 0); recall 0 %; F1 0. "
              "C: accuracy 1.844/2.000 = 92.2 %; precisión 34/184 = 18.5 %; recall 34/40 = 85.0 %; F1 = 0.304."],
        ["2", "Por accuracy: B (98.0) &gt; A (97.5) &gt; C (92.2). Por recall: C (85) &gt; A (25) &gt; B (0). El orden se invierte."],
        ["3", "A: 30·100 + 20·1 = 3.020. B: 40·100 = 4.000. C: 6·100 + 150·1 = 750. Se elige C: el modelo con peor accuracy es el más barato."]])
    s += sol("Bloque 7", [
        ["1", "Errores: 60 − 62 = −2, 70 − 66 = 4, 80 − 83 = −3, 90 − 89 = 1. SSE = 4 + 16 + 9 + 1 = 30; MSE = 7.5; RMSE = 2.74. Media real 75; SST = 225 + 25 + 25 + 225 = 500; R² = 1 − 30/500 = 0.94."],
        ["2", "Con datos tabulares y una historia corta, no se justifica Deep Learning: conviene regresión o series de tiempo (y modelos de árboles). Línea base: «demanda de la semana anterior» o el promedio móvil; el modelo debe superarla en el error de prueba."]])
    build("apuntes/C31_Ejercicios_Cuaderno_IntroML.pdf", s, "Ejercicios para el cuaderno · Clase 31")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
