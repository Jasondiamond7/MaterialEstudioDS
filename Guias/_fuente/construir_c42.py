"""Clase 42 · Taller #3 (Prof. Eloy Alvarado): guía de estudio en PDF (≈ 6 páginas).
Solo orienta (qué se pide, dónde se estudió, trampas, checklist, aclaraciones del video). NO contiene respuestas ni
números de la base del banco. Usa los mismos helpers que construir_c41.py -> Guias/C42_Guia_Taller3.pdf"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG, SOFT,
                           band, box, build, bullets, table, two_cols)

C42 = colors.HexColor("#0F766E")
C42_BG = colors.HexColor("#F0FDFA")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
GUIAS = OUT                               # carpeta Guias/
NB = "Guias/notebooks/"


def mono(t):
    return f"<font name='Mono' size='7.6'>{t}</font>"


def nb(codigo, nombre, partes=""):
    return mono(f"{nombre}.ipynb") + (f" <b>{partes}</b>" if partes else "")


# Archivos referenciados: se verifica que existan (el script falla si falta alguno)
REFS = [f"{NB}C{n}.ipynb" for n in (
    "30_PreparacionML", "31_IntroML", "32_ReduccionDimensionalidad", "33_FeatureEngineering", "34_ArbolesDecision",
    "35_RandomForest", "36_Boosting", "37_SVM", "38_EvaluacionModelos", "40_NaiveBayes", "41_Anomalias", "23_Bayesiana")]
ROOT = GUIAS.parent
EXTRA = ["Guias/C40_Guia_NaiveBayes.pdf", "Clase 42 Taller #3/Taller03_2026.pdf", "Clase 42 Taller #3/Taller03_Python.ipynb",
         "Clase 40 Naive Bayes y clasificaciones probabilistica Eloy Alvarado/DipDS_2026_S40.pdf",
         "Clase 40 Naive Bayes y clasificaciones probabilistica Eloy Alvarado/40_naive_bayes.py",
         "Clase 39 Clustering Pedro Luiz Ramos/clase_39_clustering.pdf",
         "Clase 39 Clustering Pedro Luiz Ramos/kmeans_numero_clusters.py",
         "Guias/C41_Guia_Anomalias.pdf", "Guias/C38_Guia_EvaluacionModelos.pdf", "Guias/notebooks/C42_Practica_TallerEspejo.ipynb"]


def verificar():
    faltan = [r for r in REFS if not (GUIAS / r.replace("Guias/", "", 1)).exists()]
    faltan += [r for r in EXTRA if not (ROOT / r).exists()]
    # guías y apuntes por clase
    for n, nom in [(30, "PreparacionML"), (31, "IntroML"), (32, "ReduccionDimensionalidad"), (33, "FeatureEngineering"),
                   (34, "ArbolesDecision"), (35, "RandomForest"), (36, "Boosting"), (37, "SVM"), (38, "EvaluacionModelos")]:
        for pat in (f"C{n}_Guia_{nom}.pdf", f"apuntes/C{n}_Apuntes_{nom}.pdf"):
            if not (GUIAS / pat).exists():
                faltan.append("Guias/" + pat)
    if faltan:
        raise SystemExit("Faltan archivos referenciados:\n" + "\n".join(faltan))


def tabla_preguntas(filas, color):
    rows = [["Pregunta", "En palabras simples", "Qué evalúa", "Dónde lo estudiaste"]] + filas
    w = [CONTENT_W * 0.095, CONTENT_W * 0.325, CONTENT_W * 0.26, CONTENT_W * 0.32]
    return table(rows, w, header_color=color)


def guia():
    verificar()
    s = []
    # ------------------------------------------------------------------ PÁGINA 1
    s.append(band("Clase 42 · Taller #3: respuesta a una campaña bancaria",
                  "Prof. Eloy Alvarado · 60 puntos · entrega miércoles 21-oct-2026, 23:59 · grupos de 3 a 5 · R o Python", C42))
    s.append(Spacer(1, 5))
    s.append(box([P("Cómo usar esta guía", "cellb"), Spacer(1, 2),
                  P("Dice <b>qué se pide</b> en cada pregunta, <b>qué concepto evalúa</b>, <b>en qué archivo tuyo ya lo estudiaste</b>, qué trampas hacen perder puntos y qué revisar antes de entregar. "
                    "<b>No contiene respuestas ni cifras de la base del banco</b>: el taller es una evaluación calificada del grupo. Para ver cada técnica resuelta de punta a punta hay un "
                    "notebook espejo con <b>otra base</b> (churn de telefonía, Clase 33): " + mono("Guias/notebooks/C42_Practica_TallerEspejo.ipynb") +
                    ", con la misma estructura 1(a)–5(b) y, en cada paso, una nota «En el Taller 3 esto corresponde a…».", "cell")],
                 bg=C42_BG, border=C42, pad=5))
    s.append(Spacer(1, 4))
    left = [P("El caso", "h3"),
            bullets(["Banco portugués que ofrece <b>depósitos a plazo</b> por teléfono; contactar a toda la cartera tiene costo.",
                     "Hay que <b>anticipar qué registros contratarán</b> (<font name='Mono' size='8'>contrata = 1</font>, clase positiva) y recomendar a quién llamar.",
                     "La predicción es <b>antes de la llamada registrada</b>. Cierra con perfiles (PCA, K-means) y registros inusuales (Mahalanobis).",
                     "Muestra pública <i>Bank Marketing</i> (UCI), 4 521 registros: <b>3 617</b> de entrenamiento y <b>904</b> de prueba.",
                     "Cubre el módulo de aprendizaje automático: <b>Clases 30 a 41</b> (el profesor dijo «desde la 30 en adelante, incluyendo los repasos»)."],
                    "cell", C42)]
    right = [P("Reglas de entrega", "h3"),
             bullets(["Un solo archivo <font name='Mono' size='8'>Grupo_XX_Taller3.zip</font> por grupo (sube <b>una</b> persona) con el <b>notebook ejecutado</b> y los <b>dos .csv</b>.",
                      "<b>Todo en el mismo lenguaje</b> (R o Python). Identifica a todos los integrantes. Rutas <b>relativas</b>; indica las <b>versiones de los paquetes</b>.",
                      "Cada respuesta con <b>código, resultado e interpretación</b>. Si entregan HTML/PDF, adjunten también la <b>fuente</b> (.ipynb, .Rmd o .qmd).",
                      "Plantillas del profesor: " + mono("Taller03_Python.ipynb / .qmd") + " y las equivalentes en R.",
                      "Semilla <b>2026</b>; <b>conservar las filas</b>; <b>5 pliegues entregados</b> (columna <font name='Mono' size='8'>pliegue</font>); la prueba solo se usa en <b>3(b)</b>."],
                     "cell", C42)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    left = [P("Nota y reparto de puntos", "h3"),
            table([["Sección", "Pts", "%"], ["1 · Problema, preparación y representación", "12", "20 %"],
                   ["2 · Selección de variables y modelos", "24", "40 %"], ["3 · Evaluación y decisión", "10", "17 %"],
                   ["4 · Componentes principales y agrupamiento", "8", "13 %"], ["5 · Detección de anomalías", "6", "10 %"],
                   ["Total (Nota = 1 + P/10)", "60", "100 %"]], [HALF * 0.66, HALF * 0.14, HALF * 0.2], header_color=C42),
            Spacer(1, 2),
            P("Se evalúan <b>procedimiento, resultados e interpretación</b>. Crédito parcial y <b>sin descontar dos veces un mismo error arrastrado</b>; se aceptan diferencias numéricas y ganadores distintos "
              "si la implementación y la regla de selección están justificadas.", "small")]
    right = [P("Plan para los días que quedan (hoy es domingo 18)", "h3"),
             table([["Día", "Meta del grupo"],
                    ["Dom 18", "Leer enunciado y esta guía; repartir roles; correr el notebook espejo; <b>Sección 1</b> y la carga de datos"],
                    ["Lun 19", "<b>Sección 2</b> completa (OOF de los 9 modelos) y 3(a); redactar 2(c) con la tabla en mano"],
                    ["Mar 20", "3(b), <b>Secciones 4 y 5</b>; ejecutar el notebook <b>de cero</b> y corregir"],
                    ["Mié 21", "Revisión cruzada con el checklist, zip, subir <b>antes de las 18:00</b> (margen de 6 h)"]],
                   [HALF * 0.16, HALF * 0.84], header_color=C42)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Orden de estudio sugerido (≈ 3 h):</b> Guía C30 → C33 (selección dentro del pliegue) → C38 (OOF, log loss, umbral por costo) → resto de las clases según la tabla de las páginas 2–3 → "
                    "notebook espejo (un ejercicio por pregunta) → recién entonces abrir la plantilla del taller.", "cell")], bg=SOFT, border=C42, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Dudas para el foro (el enunciado no las precisa):</b> ¿cómo tratar <i>dia</i> y <i>mes</i> (numéricas o categorías)? · ¿qué valores toma <i>pliegue</i> (0–4 o 1–5)? · "
                    "¿las columnas <i>mora/hipoteca/prestamo</i> entran como 0/1? · ¿se acepta Boosting de <i>HistGradientBoosting</i> o <i>XGBoost</i>? · ¿pueden repetirse los grupos de los talleres anteriores? "
                    "(la pregunta del chat quedó cortada en el video y no se alcanza a oír la respuesta).", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    # ------------------------------------------------------------------ PÁGINAS 2-3: pregunta por pregunta
    s.append(band("Pregunta por pregunta: qué piden y dónde lo estudiaste",
                  "Los archivos son de tu carpeta Guias/ (notebooks y guías PDF). «Partes» = secciones del notebook. Cada clase tiene su guía C3x_Guia_*.pdf y apuntes en Guias/apuntes/.", C42))
    s.append(Spacer(1, 4))
    s.append(P("Sección 1 · Problema, preparación y representación (12 pts)", "h2"))
    s.append(tabla_preguntas([
        ["<b>1(a)</b><br/>3 pts",
         "Decir qué tipo de problema es, qué es una fila y qué se predice. Explicar por qué <b>duracion</b> no puede usarse antes de la llamada y sacarla de los predictores (junto con <i>id_cliente, pliegue, contrata</i>).",
         "Clasificación supervisada, unidad de observación, respuesta. Fuga por <b>momento de medición</b>: ¿existe la variable cuando hay que decidir?",
         nb("", "C31_IntroML", "A1") + " (del problema a la tarea)<br/>" + nb("", "C30_PreparacionML", "B3") + " (fuga de datos)"],
        ["<b>1(b)</b><br/>4 pts",
         "Revisar tipos, faltantes, categorías <i>unknown</i>, duplicados y balance. Interpretar los <b>saldo negativos</b> y el código <b>−1</b> de dias_previos. ¿Bastaría la exactitud de predecir siempre la clase mayoritaria?",
         "Calidad de datos: valor <i>imposible</i> vs. <i>posible</i> vs. <i>código</i>; exactitud vs. sensibilidad; línea base.",
         nb("", "C30_PreparacionML", "A1–A3, D2") + "<br/>" + nb("", "C33_FeatureEngineering", "A1–A4") + " (imposibles, duplicados, no borrar filas)<br/>" + nb("", "C31_IntroML", "B3")],
        ["<b>1(c)</b><br/>5 pts",
         "Crear <i>contactado_antes</i> y <i>log_contactos_previos</i> y explicar su sentido; cambiar −1 por 0 conservando el indicador; mantener <i>unknown</i> como categoría; one-hot; <b>describir y aplicar el escalamiento que necesita cada método</b>.",
         "Variables derivadas, códigos especiales, codificación, qué métodos requieren escala (logística y SVM sí; árboles y NB no), todo dentro del <i>Pipeline</i>.",
         nb("", "C30_PreparacionML", "A1, B4, C1") + "<br/>" + nb("", "C32_ReduccionDimensionalidad", "B1–B4") + "<br/>" + nb("", "C33_FeatureEngineering", "B2") + "; " + nb("", "C34_ArbolesDecision", "D2") + "; " + nb("", "C37_SVM", "B4")],
    ], C42))
    s.append(Spacer(1, 2))
    s.append(P("Sección 2 · Selección de variables y modelos supervisados (24 pts)", "h2"))
    s.append(tabla_preguntas([
        ["<b>2(a)</b><br/>5 pts",
         "Logística <b>sin penalización</b>: todas las columnas preparadas vs. las <b>10 de mayor F de ANOVA</b> (elegidas en cada pliegue). Informar las columnas por pliegue y comparar métricas OOF. ¿Diferencia con PCA? ¿Limitación del filtro univariado?",
         "Selección <b>dentro</b> del pliegue; filtro supervisado vs. extracción no supervisada; redundancia e interacciones.",
         nb("", "C33_FeatureEngineering", "C2–C5, E1–E4") + " (ANOVA F, irrelevante sola, XOR, redundancia, pipeline honesto)<br/>" + nb("", "C32_ReduccionDimensionalidad", "Parte C") + " (PCA)<br/>" + nb("", "C37_SVM", "A2") + " (penalty=None)"],
        ["<b>2(b)</b><br/>11 pts",
         "Comparar <b>9 filas</b> con predicciones OOF: referencia, logística (2), árbol prof. 3 y 6 (mín. 20 por hoja), Random Forest (200, √p, mín. 10), Boosting (≤ 200, prof. 3; informar variante y tasa), SVM radial (C = 1, γ = 1/p) y Naive Bayes gaussiano. Log loss, Brier, AUC; <b>SVM solo AUC</b>; árboles: log loss medio de entrenamiento. Elegir el probabilístico de <b>menor log loss OOF</b>.",
         "Validación cruzada con pliegues dados, OOF, métricas probabilísticas, configuración exacta de hiperparámetros, puntaje de decisión vs. probabilidad.",
         nb("", "C38_EvaluacionModelos", "A3, B2, B3") + " (OOF, log loss/Brier, AUC)<br/>" + nb("", "C34_ArbolesDecision", "C1–C2") + "; " + nb("", "C35_RandomForest", "B1, B3") + "<br/>" + nb("", "C36_Boosting", "C3, D1") + "; " + nb("", "C37_SVM", "B3, C3") + "<br/><b>Naive Bayes:</b> " + nb("", "C40_NaiveBayes", "B3–B4, C3–C4, D2")],
        ["<b>2(c)</b><br/>8 pts",
         "Interpretar con los números de la tabla: profundidad y sobreajuste; bagging vs. RF vs. boosting; independencia condicional y límite de la gaussiana con columnas binarias; efecto de C, γ y escala en SVM; <b>¿mejor orden implica mejores probabilidades?</b>",
         "Sesgo–varianza, brecha entrenamiento–validación, supuestos de Naive Bayes, hiperparámetros del kernel, ranking (AUC) vs. calibración (log loss).",
         nb("", "C34_ArbolesDecision", "C2") + "; " + nb("", "C35_RandomForest", "A3, B2–B3") + "; " + nb("", "C36_Boosting", "A3") + "<br/>" + nb("", "C37_SVM", "B2–B4, C3") + "; " + nb("", "C38_EvaluacionModelos", "B2–B4") + " (calibración)"],
    ], C42))
    s.append(Spacer(1, 2))
    s.append(P("Sección 3 · Evaluación y decisión (10 pts)", "h2"))
    s.append(tabla_preguntas([
        ["<b>3(a)</b><br/>4 pts",
         "Con las probabilidades OOF del modelo elegido, evaluar umbrales <b>0.05, 0.10, …, 0.95</b>, predecir contratación si <b>p ≥ t</b> y elegir el que minimiza <b>C(t) = FP + 5·FN</b>. Tabla o curva e interpretación de ambos errores.",
         "Matriz de confusión por umbral, costos asimétricos, qué significan FP y FN para el banco.",
         nb("", "C38_EvaluacionModelos", "B5, C7") + " (costo y umbral; umbral teórico)<br/>" + nb("", "C30_PreparacionML", "D2–D3") + "; " + nb("", "C37_SVM", "A3–A4")],
        ["<b>3(b)</b><br/>6 pts",
         "<b>Reajustar con todo el entrenamiento</b> y evaluar en la prueba: log loss, Brier, AUC y curva ROC; umbral elegido vs. 0.50 con matriz de confusión, sensibilidad, especificidad y <b>C/n</b>; incluir la referencia. Recomendar, concluir y decir qué falta verificar.",
         "Evaluación final honesta (prueba una vez), ROC, comparación de reglas por costo, lectura crítica y límites.",
         nb("", "C38_EvaluacionModelos", "A5, B3, B6") + "<br/>" + nb("", "C37_SVM", "A4") + "; " + nb("", "C31_IntroML", "B3")],
    ], C42))
    s.append(Spacer(1, 2))
    s.append(P("Secciones 4 y 5 · Componentes principales, agrupamiento y anomalías (14 pts)", "h2"))
    s.append(P("Se usan solo <b>edad, saldo, contactos_campana y contactos_previos</b> del entrenamiento, <b>estandarizadas con media y desviación muestral</b>.", "small"))
    s.append(tabla_preguntas([
        ["<b>4(a)</b><br/>3 pts",
         "PCA: varianza explicada por PC1 y PC2, su suma, las cargas y el gráfico de las observaciones en ese plano. ¿Qué resume la proyección y qué omite?",
         "Autovalores/varianza explicada, interpretación de cargas, límites de una proyección (no usa la respuesta).",
         nb("", "C32_ReduccionDimensionalidad", "C2, C4") + "<br/>" + nb("", "C31_IntroML", "C3")],
        ["<b>4(b)</b><br/>5 pts",
         "K-means con <b>k = 2, 3, 4</b> y <b>≥ 30 inicializaciones</b>: comparar WSS y silueta media, elegir el k de mayor silueta, informar tamaños, perfiles, grupos en el plano PCA y la proporción de contratación por grupo. ¿Por qué los grupos no son clases predichas?",
         "Aprendizaje no supervisado, WSS vs. silueta, perfilamiento, diferencia entre segmentar y predecir.",
         nb("", "C31_IntroML", "C1–C2") + "<br/>Clase 39 (profesor): " + mono("clase_39_clustering.pdf") + ", " + mono("kmeans_numero_clusters.py")],
        ["<b>5(a)</b><br/>3 pts",
         "Calcular la distancia de <b>Mahalanobis al cuadrado</b>; corte = cuantil 0.99 de una <b>χ² con 4 g. l.</b>; informar corte, cantidad y % marcados; presentar los <b>dos registros</b> de mayor distancia con sus valores originales.",
         "Distancia que usa la covarianza, χ² como referencia bajo normalidad, lectura de registros.",
         nb("", "C41_Anomalias", "A3") + "<br/>Guía " + mono("C41_Guia_Anomalias.pdf") + " (página de Mahalanobis)"],
        ["<b>5(b)</b><br/>3 pts",
         "¿Por qué son inusuales esos dos? ¿Demuestra que son errores? ¿El corte <b>garantiza</b> marcar el 1 %? Fundamentar y decir qué se revisaría antes de eliminar un registro.",
         "Atípico ≠ error; supuestos de la χ²; criterio de revisión (unidades, duplicados, contexto, con/sin el caso).",
         nb("", "C41_Anomalias", "A3, A7, D1") + "<br/>Guía C41: recuadro «¿atípico, error o evento real?»"],
    ], C42))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Naive Bayes (Clase 40).</b> Para 2(b) y 2(c) estudia " + mono("C40_NaiveBayes.ipynb") + " y " + mono("C40_Guia_NaiveBayes.pdf") +
                    " (independencia condicional, copias de una variable tratadas como independientes, <i>GaussianNB</i> vs. <i>BernoulliNB</i>); material del profesor, como apoyo secundario: "
                    + mono("Clase 40 Naive Bayes…/DipDS_2026_S40.pdf") + " y " + mono("40_naive_bayes.py") + ". "
                    "La base conceptual (teorema de Bayes, P(y|x) proporcional a P(x|y)·P(y)) está en " + mono("C23_Bayesiana.ipynb") +
                    ", pero esa clase trata inferencia bayesiana, <b>no</b> Naive Bayes. El notebook espejo muestra cómo se ve un Naive Bayes gaussiano con muchas columnas redundantes.", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    # ------------------------------------------------------------------ PÁGINA 5: trampas
    s.append(band("Trampas que hacen perder puntos", "Cada fila es un error frecuente; la columna derecha dice cómo evitarlo", RED))
    s.append(Spacer(1, 4))
    tr = [["#", "Trampa", "Cómo evitarla"],
          ["1", "Usar <b>duracion</b>, <i>id_cliente</i> o <i>pliegue</i> como predictores.", "Excluir <i>duracion, id_cliente, pliegue, contrata</i> de X desde el comienzo y dejar la lista escrita."],
          ["2", "Ajustar escalador, one-hot, imputación o selección con <b>todo</b> el entrenamiento y recién después validar (fuga).", "Un <i>Pipeline/ColumnTransformer</i> que se ajusta <b>solo con las filas de ajuste</b> de cada pliegue (R: <i>recipes</i> dentro del bucle)."],
          ["3", "No usar la columna <b>pliegue</b> dada (<i>cross_val_score</i>, <i>KFold</i> propio).", "Bucle sobre <i>sorted(pliegue.unique())</i>; ajuste = pliegue ≠ k, validación = pliegue = k. Mira primero los valores de la columna."],
          ["4", "F de ANOVA calculado fuera del pliegue; no informar las columnas seleccionadas <b>por pliegue</b>.", "<i>SelectKBest</i> dentro del Pipeline; guardar <i>get_support()</i> de cada ajuste y mostrar una tabla pliegue × columnas."],
          ["5", "SVM: pedir probabilidades, calcular log loss/Brier o promediar puntajes entre pliegues como si fueran probabilidades.", "<i>decision_function</i> OOF y <b>solo AUC</b>; C = 1; <b>γ = 1/p</b> con p = columnas <b>ya preparadas</b> (después del one-hot)."],
          ["6", "Referencia mal definida (0.5, o la tasa de todo el entrenamiento dentro de la CV).", "En cada pliegue: tasa positiva de las <b>filas de ajuste</b>; en la evaluación final: tasa de <b>todo</b> el entrenamiento."],
          ["7", "Elegir el modelo por AUC, o dejar competir a la SVM o a la referencia.", "Regla escrita: <b>menor log loss OOF entre los probabilísticos</b>. Un ganador distinto al de otros grupos es válido si está justificado."],
          ["8", "Árboles: olvidar el <b>log loss medio de entrenamiento de los cinco ajustes</b>; confundir mínimo por hoja con mínimo para dividir.", "Calcular log loss sobre las filas de ajuste en cada pliegue y promediar; <i>min_samples_leaf</i> = 20 (árboles) y 10 (RF)."],
          ["9", "Umbral con <b>p &gt; t</b>, o costo 5·FP + FN; tocar la prueba antes de 3(b) o elegir el umbral con ella.", "Regla <b>p ≥ t</b> con t = 0.05, 0.10, …, 0.95; C(t) = FP + 5·FN; el umbral sale de los <b>OOF</b>; la prueba se usa una sola vez."],
          ["10", "Confundir <b>ordenar</b> con <b>dar buenas probabilidades</b> (AUC alto ≠ log loss bajo).", "En 2(c) contrasta ambas métricas con la propia tabla; el AUC no cambia con transformaciones monótonas de p."],
          ["11", "Naive Bayes gaussiano sobre columnas 0/1 «porque lo pide el enunciado»: se interpreta mal el resultado.", "Hacerlo como se pide y <b>comentar</b> el límite (una dummy no es normal; independencia condicional falsa; probabilidades extremas)."],
          ["12", "Perder el indicador al reemplazar −1 por 0, o tratar −1 como un número de días.", "Crear <i>contactado_antes</i> <b>antes</b> de reemplazar; explicar que −1 es un código y que <b>saldo</b> negativo es un valor posible."],
          ["13", "PCA/K-means con desviación poblacional (<i>StandardScaler</i> divide por n) o sobre variables sin estandarizar; K-means con una sola inicialización.", "Z = (x − media)/sd con <b>ddof = 1</b>; <i>n_init ≥ 30</i>, semilla 2026; elegir k por <b>silueta</b> y reportar WSS."],
          ["14", "Mahalanobis: decir que el corte «marca el 1 %» o que lo marcado «son errores».", "El 1 % vale solo con normalidad multivariada y parámetros conocidos; una alerta es una <b>pista de revisión</b> (C41)."],
          ["15", "Mezclar lenguajes, rutas absolutas, no declarar versiones, no ejecutar de cero, entregar solo código sin interpretación.", "Reiniciar kernel y ejecutar todo; rutas relativas; imprimir versiones; <b>una interpretación bajo cada resultado</b>."]]
    s.append(table(tr, [CONTENT_W * 0.05, CONTENT_W * 0.45, CONTENT_W * 0.50], header_color=RED))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Para grupos que elijan R.</b> Mismas decisiones, otras funciones: <i>glm(family = binomial)</i>, <i>rpart</i> (<i>minbucket</i> = 20, <i>cp</i> = 0), <i>ranger</i> (<i>num.trees</i> = 200, <i>mtry</i> = floor(√p), <i>min.node.size</i> = 10), "
                    "<i>e1071::svm</i> (<i>decision.values</i> = TRUE; ojo con el signo), <i>naiveBayes</i>, <i>prcomp</i>, <i>kmeans(nstart = 30)</i>, <i>mahalanobis</i> y <i>qchisq(0.99, 4)</i>. "
                    "La tabla completa R ↔ Python está al final del notebook espejo. El profesor aceptó <b>diferencias numéricas</b> entre lenguajes.", "cell")],
                 bg=SOFT, border=NAVY, pad=5))
    s.append(PageBreak())

    # ------------------------------------------------------------------ PÁGINA 6: checklist + video
    s.append(band("Checklist de entrega y lo que dijo el profesor en la clase",
                  "Transcripción corregida de «Clase 42 taller 3» (≈ 19 min); los tiempos son mm:ss del video", GREEN))
    s.append(Spacer(1, 4))
    ck_a = ["Notebook ejecutado <b>de punta a punta</b> en un kernel limpio, sin errores",
            "Nombres de los <b>integrantes</b> en el encabezado",
            "Un solo lenguaje; <b>rutas relativas</b>; versiones de paquetes impresas",
            "Semilla <b>2026</b> en todo lo aleatorio; clase positiva = 1",
            "<i>duracion, id_cliente, pliegue, contrata</i> fuera de los predictores",
            "Transformaciones y selección <b>solo con filas de ajuste</b> de cada pliegue",
            "Se usó la columna <b>pliegue</b> entregada (5 pliegues)",
            "1(b): faltantes, <i>unknown</i>, duplicados, balance, saldo negativo y −1 interpretados",
            "1(c): variables nuevas explicadas y escalamiento descrito por método"]
    ck_b = ["2(a): columnas elegidas <b>por pliegue</b> + métricas OOF + diferencia con PCA + limitación",
            "2(b): 9 filas; SVM solo AUC; árboles con log loss de entrenamiento; modelo elegido con regla escrita",
            "2(c): las cinco preguntas respondidas con los números de la tabla",
            "3(a): umbrales 0.05–0.95, p ≥ t, C = FP + 5·FN, tabla o curva",
            "3(b): reajuste con todo el entrenamiento; prueba <b>una vez</b>; ROC; matrices en t* y 0.50; C/n; referencia; recomendación",
            "4–5: cuatro variables estandarizadas (sd muestral); PCA, k-means (n_init ≥ 30), Mahalanobis con χ²(0.99; 4)",
            "Interpretación bajo <b>cada</b> resultado; gráficos con ejes y título",
            "Si hay HTML/PDF: <b>fuente</b> adjunta. Zip <font name='Mono' size='8'>Grupo_XX_Taller3.zip</font> con notebook + 2 .csv",
            "Subido <b>antes de las 18:00 del miércoles</b> (límite 23:59); una sola persona sube"]
    s.append(two_cols([bullets(ck_a, "cell", GREEN, marker=SYM)], [bullets(ck_b, "cell", GREEN, marker=SYM)]))
    s.append(Spacer(1, 5))
    s.append(P("Qué aclaró el profesor al leer el enunciado (con tiempos)", "h2"))
    vid = [["Tiempo", "Lo que dijo", "Qué cambia para ti"],
           ["01:07–01:17", "Grupos de tres a cinco; admite uno algo menor, pero recomienda al menos tres.", "Define el grupo hoy; solo uno sube el archivo."],
           ["02:07–02:21", "El taller abarca el módulo de aprendizaje automático, <b>desde la sesión 30 en adelante, incluidos los repasos</b>.", "Las Clases 30–41 son la base; la Clase 43 (bootstrap) no se menciona en el enunciado."],
           ["04:37–05:30", "El zip se llama <b>Grupo_XX_Taller3</b>; si compilan a PDF/HTML, deben <b>incorporar los archivos fuente</b> (el .qmd o el notebook).", "No basta el HTML: adjunta la fuente."],
           ["06:03–06:25", "Entrenar solo con <i>clientes_entrenamiento</i>; la prueba está <b>reservada para 3(b)</b>; conservar las filas; semilla 2026; contrata = 1 es la clase positiva.", "Sin eliminar filas; sin tocar la prueba antes."],
           ["06:31–07:10", "<b>5 pliegues</b>; aprender transformaciones y selección <b>solo con las filas de ajuste</b>; reunir predicciones fuera de pliegue: probabilidades y, para SVM, puntajes de decisión. La referencia es la proporción positiva de las filas de ajuste; en la evaluación final, la de <b>todo</b> el entrenamiento.", "Es la trampa 2, 3, 5 y 6."],
           ["08:16–08:30", "Las particiones las hizo el profesor por <b>reproducibilidad y para tener un recurso común entre lenguajes</b>.", "No rehacer la separación ni los pliegues."],
           ["10:47–11:18", "Para SVM, <b>solo AUC</b>; incluir la referencia y, para ambos árboles, el log loss medio de entrenamiento de los cinco ajustes.", "Tabla de 2(b) con esos detalles."],
           ["11:39–11:48", "«¿Un mejor ordenamiento implica mejores probabilidades? Tienen que fundamentarlo.»", "Pregunta explícita de 2(c); no se responde con un sí/no."],
           ["12:44–13:13", "3(b): informar log loss, Brier, AUC y ROC; comparar el umbral elegido con 0.5; <b>«traten de explayarse lo suficiente»</b> en las repercusiones.", "La recomendación al banco merece extensión y matices."],
           ["14:36–15:12", "5(b): ¿demuestra errores? ¿el corte garantiza el 1 %? ¿qué revisarían antes de eliminar? <b>Un error no arrastra descuento sustancial</b> al resto; <b>pueden existir diferencias numéricas</b> entre lenguajes.", "No te bloquees por un número distinto; justifica."],
           ["16:07–16:37", "Entrega el <b>miércoles 21</b>; la idea es empezar hoy y resolver dudas; habrá un <b>foro</b> en el módulo como en el Taller 2.", "Usa el foro para dudas de interpretación."]]
    s.append(table(vid, [CONTENT_W * 0.12, CONTENT_W * 0.58, CONTENT_W * 0.30], header_color=GREEN))
    build("C42_Guia_Taller3.pdf", s, "Guía · Clase 42 · Taller #3 (sin respuestas del taller)")


if __name__ == "__main__":
    guia()
    print("OK")
