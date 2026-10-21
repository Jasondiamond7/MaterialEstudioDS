"""Clase 42 · Taller #3: PDF de ESTUDIO que explica, pregunta por pregunta, la solución del Grupo 36.
Solo explica lo que ya está en su notebook (no crea respuestas nuevas ni modifica sus carpetas).
Reutiliza los helpers de construir_pdf.py / construir_c42.py -> Guias/apuntes/C42_Explicacion_Solucion_Taller3_Grupo36.pdf"""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer, Table, TableStyle

from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, P, RED, RED_BG, SOFT,
                           band, box, build, bullets, table, two_cols)
from construir_c42 import C42, C42_BG, mono, nb

SALIDA = "apuntes/C42_Explicacion_Solucion_Taller3_Grupo36.pdf"
LBL_W = 27 * mm


PEND = []


def enc(x):
    PEND.append(x)


def bloque(titulo, filas, color=C42, antes=()):
    """Un recuadro por pregunta: cabecera de color + filas (etiqueta | texto). Se mantiene unido en una página."""
    data = [[P(f'<font color="white">{titulo}</font>', "cellb"), ""]]
    for et, tx in filas:
        data.append([P(et, "cellb"), P(tx, "cell")])
    t = Table(data, colWidths=[LBL_W, CONTENT_W - LBL_W])
    style = [("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (-1, 0), color),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
             ("BACKGROUND", (0, 1), (0, -1), C42_BG)]
    t.setStyle(TableStyle(style))
    pre = list(antes) + PEND[:]
    PEND.clear()
    return KeepTogether(pre + [t, Spacer(1, 5)])


def pregunta_prof(q, a):
    return f"<i>{q}</i><br/><b>Respuesta corta:</b> {a}"


def contenido():
    s = []
    # ================================================================== PÁGINA 1
    s.append(band("Cómo resolvió el Grupo 36 el Taller #3, explicado fácil",
                  "Guía de estudio para entender y defender la solución · Diplomado Data Science UC 2026 · Clase 42", C42))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Material de estudio personal para entender la solución del grupo. No forma parte de la entrega.</b> "
                    "Explica lo que ya está en el notebook <font name='Mono' size='8'>Grupo_36_Taller3.ipynb</font>; no agrega análisis nuevos. "
                    "Todas las cifras salen de las salidas guardadas o del texto de ese notebook.", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 4))
    left = [P("El caso en 5 líneas", "h3"),
            bullets(["Un banco portugués llama por teléfono para ofrecer <b>depósitos a plazo</b>. Llamar cuesta y muchos dicen que no.",
                     "Hay 4 521 registros: <b>3 617</b> para entrenar y <b>904</b> para la prueba final.",
                     "Se quiere predecir quién <b>contrata</b> (1) o no (0) <b>antes</b> de llamar. Solo 11,5 % contrata.",
                     "Se comparan 9 modelos, se elige uno y se decide a partir de qué probabilidad vale la pena llamar.",
                     "Al final se buscan <b>perfiles</b> de clientes (PCA, K-means) y <b>registros raros</b> (Mahalanobis)."],
                    "cell", C42)]
    flow = ["1. Preparar datos", "2. Validar con pliegues", "3. Comparar modelos", "4. Elegir umbral (costo)",
            "5. Evaluar en prueba", "6. PCA y K-means", "7. Anomalías"]
    cap = ["1(a)–1(c)", "2(a)", "2(b), 2(c)", "3(a)", "3(b)", "4(a), 4(b)", "5(a), 5(b)"]
    fl = [[P(f"<b>{a}</b><br/><font size='7.4' color='#6B7280'>{b}</font>", "cell")] for a, b in zip(flow, cap)]
    ft = Table([[c[0]] for c in fl], colWidths=[HALFW])
    ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C42_BG), ("LINEBELOW", (0, 0), (-1, -2), 0.6, colors.white),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LINEBEFORE", (0, 0), (0, -1), 3, C42)]))
    right = [P("La solución como un flujo", "h3"), ft]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    gl = [["Término", "Qué significa, en simple"],
          ["<b>Pliegue</b>", "Cada uno de los 5 trozos en que viene partido el entrenamiento. Se entrena con 4 y se examina con el 5.º, y se repite 5 veces."],
          ["<b>OOF</b> (fuera de pliegue)", "La predicción de cada cliente hecha por un modelo que <b>no lo vio</b> al entrenar. Es como corregir un examen con preguntas nuevas."],
          ["<b>Log loss</b>", "Nota de las probabilidades: castiga fuerte decir «99 % seguro» y equivocarse. <b>Menor es mejor.</b>"],
          ["<b>Brier</b>", "Promedio de (probabilidad − realidad)²; también menor es mejor. Es más suave con los errores extremos que el log loss."],
          ["<b>AUC</b>", "Mide si el modelo <b>ordena</b> bien: de un contratante y un no contratante al azar, ¿a cuál le da más probabilidad? 0,5 = azar; 1 = perfecto."],
          ["<b>Umbral</b>", "Probabilidad desde la cual se decide llamar. Si p ≥ umbral, se llama."],
          ["<b>Costo</b> C(t)", "FP + 5·FN. Una falsa alarma (FP: llamar a quien no contrata) cuesta 1; una oportunidad perdida (FN: no llamar a quien sí contrataba) cuesta 5."],
          ["<b>One-hot</b>", "Convertir una categoría (p. ej. mes) en columnas 0/1, una por categoría."],
          ["<b>Estandarizar</b>", "Restar la media y dividir por la desviación, para que todas las variables queden en la misma escala."],
          ["<b>Silueta</b>", "Cuán bien separados están los grupos de K-means (de −1 a 1; más alto = grupos más nítidos)."],
          ["<b>Mahalanobis</b>", "Distancia de un registro al «centro» de los datos, teniendo en cuenta cómo se relacionan las variables. Más grande = más raro."]]
    s.append(box([P("Glosario express", "h3"), table(gl, [CONTENT_W * 0.22, CONTENT_W * 0.74 - 6], header_color=C42)],
                 bg=colors.white, border=C42, pad=4))
    s.append(Spacer(1, 3))
    s.append(P("Cada pregunta está en un recuadro con: <b>Qué pedía · Qué hicieron · Por qué así · Cómo leer el resultado · Pregunta que podría hacer el profesor · Dónde lo estudiaste</b>. "
               "Al final, lo que conviene corregir antes de entregar y una autoevaluación oral.", "small"))
    s.append(PageBreak())

    # ================================================================== PREGUNTA 1
    enc(P("Pregunta 1 · Problema, preparación y representación (12 puntos)", "h2"))
    s.append(bloque("1(a) · Tarea, unidad, respuesta y por qué no usar «duracion» (3 pts)", [
        ("Qué pedía", "Decir qué tipo de problema es, qué es una fila y qué se predice; explicar por qué <i>duracion</i> no sirve para predecir antes de llamar y sacarla de los predictores."),
        ("Qué hicieron", "Dijeron: <b>clasificación binaria supervisada</b>; cada fila es un cliente con su llamada registrada (3 617 en entrenamiento); la respuesta es <i>contrata</i>. "
                         "Excluyeron <i>duracion, id_cliente, pliegue</i> y <i>contrata</i>: quedan <b>15 predictores</b>. Mostraron que la duración media es <b>554 s</b> en quienes contratan y <b>228 s</b> en quienes no."),
        ("Por qué así", "La duración solo se sabe <b>al colgar</b>, y una llamada larga suele ser una venta. Usarla sería como corregir un examen mirando la nota: da resultados lindos "
                        "pero inútiles, porque lo que el banco quiere es decidir <i>a quién llamar</i>. Eso se llama <b>fuga de información</b>. "
                        "<i>id_cliente</i> es solo un número de fila y <i>pliegue</i> es parte del diseño de validación."),
        ("Cómo leer el resultado", "Los 554 s contra 228 s prueban que la duración «delata» la respuesta: por eso se saca."),
        ("Pregunta del profesor", pregunta_prof("¿Qué pasaría si dejaran duracion?",
                                                 "El modelo se vería excelente en validación, pero en la vida real no tendríamos ese dato al momento de decidir a quién llamar.")),
        ("Dónde lo estudiaste", nb("", "C31_IntroML", "A1") + " · " + nb("", "C30_PreparacionML", "B3") + " (fuga de datos)")]))
    s.append(bloque("1(b) · Calidad de los datos y exactitud de la clase mayoritaria (4 pts)", [
        ("Qué pedía", "Revisar tipos, faltantes, <i>unknown</i>, duplicados y balance; interpretar los saldos negativos y el −1 de <i>dias_previos</i>; decir si basta la exactitud de «siempre no contrata»."),
        ("Qué hicieron", "Contaron 6 variables numéricas y 9 categóricas; <b>0 faltantes</b> y <b>0 duplicados</b>. <i>Unknown</i>: <i>resultado_previo</i> 82,4 %, <i>canal</i> 30,1 %, <i>educacion</i> 4,3 %, <i>ocupacion</i> 1,0 %. "
                         "Balance: 3 200 no contratan (88,5 %) y 417 sí (11,5 %), casi igual en los 5 pliegues. Saldo negativo: 302 registros (mínimo −3 313 €). "
                         "<i>dias_previos</i> = −1 en 2 979 registros, justo los que tienen <i>contactos_previos</i> = 0."),
        ("Por qué así", "Un <i>unknown</i> no siempre es «dato perdido»: aquí 82,4 % en <i>resultado_previo</i> significa simplemente «nunca lo contactaron antes». "
                        "El saldo negativo es un cliente sobregirado (posible, no error). El −1 es un <b>código</b>, no «menos un día». "
                        "Y predecir siempre «no» acierta 88,5 % sin encontrar a un solo cliente interesado: la exactitud engaña cuando una clase es mucho más grande."),
        ("Cómo leer el resultado", "Exactitud 0,8847 con <b>sensibilidad 0</b> (de los que sí contratan, detecta 0 %). Por eso se miden log loss, Brier, AUC y costos."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué no basta la exactitud?",
                                                 "Porque con 88,5 % de «no», un modelo que nunca llama ya tiene 88,5 % y no sirve al banco; hay que mirar cuántos contratantes se detectan y cuánto cuestan los errores.")),
        ("Dónde lo estudiaste", nb("", "C30_PreparacionML", "A1–A3, D2") + " · " + nb("", "C33_FeatureEngineering", "A1–A4") + " · " + nb("", "C31_IntroML", "B3")),
    ]))
    s.append(bloque("1(c) · Variables nuevas, one-hot y escalamiento (5 pts)", [
        ("Qué pedía", "Crear <i>contactado_antes</i> y <i>log_contactos_previos</i>, cambiar −1 por 0, codificar categorías (one-hot) y explicar qué escalamiento necesita cada método."),
        ("Qué hicieron", "<i>contactado_antes</i> = 1 si <i>dias_previos</i> ≠ −1; <i>log_contactos_previos</i> = log(1 + contactos previos). Reemplazaron −1 por 0 y mantuvieron <i>unknown</i> como categoría. "
                         "One-hot dejando una categoría de referencia. Resultado: <b>p = 42 columnas</b> (6 numéricas, 1 indicador, 35 dummies). "
                         "Todo se aprende <b>solo con las filas de ajuste</b> de cada pliegue."),
        ("Por qué así", "El indicador conserva la información que el −1 escondía: contratan <b>22,4 %</b> los contactados antes vs <b>9,2 %</b> el resto. "
                        "El logaritmo «aplasta» los valores muy grandes (asimetría 5,85 → 2,51): pasar de 0 a 1 contacto importa más que de 10 a 11. "
                        "Escalar: la SVM mide distancias, así que sin escalar <i>saldo</i> mandaría sobre todo; logística lo agradece pero no cambia sus predicciones; árboles, bosques, boosting y Naive Bayes no lo necesitan."),
        ("Cómo leer el resultado", "Tras estandarizar, las numéricas quedan con media ≈ 0 y desviación 1. Las dummies se dejan en 0/1."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué aprender el one-hot y el escalado solo con las filas de ajuste?",
                                                 "Si se aprenden con todo, el pliegue de validación «se cuela» en el entrenamiento (fuga) y la nota sale inflada.")),
        ("Dónde lo estudiaste", nb("", "C30_PreparacionML", "A1, B4, C1") + " · " + nb("", "C32_ReduccionDimensionalidad", "B1–B4") + " · " + nb("", "C33_FeatureEngineering", "B2") + " · " + nb("", "C37_SVM", "B4")),
    ]))

    # ================================================================== PREGUNTA 2
    enc(P("Pregunta 2 · Selección de variables y modelos supervisados (24 puntos)", "h2"))
    enc(P("Cómo se hace la validación: se toma un pliegue como «examen», se entrena con los otros cuatro (aprendiendo preparación y selección solo ahí) y se guardan las predicciones del examen. "
               "Tras los 5 pliegues, cada cliente tiene una predicción hecha sin haberlo visto: son las <b>OOF</b>.", "small"))
    s.append(bloque("2(a) · Logística con todas las columnas vs 10 columnas por F de ANOVA (5 pts)", [
        ("Qué pedía", "Comparar la logística con las 42 columnas y con las 10 de mayor F; informar las columnas por pliegue y las métricas OOF; diferenciar de PCA; dar una limitación del filtro."),
        ("Qué hicieron", "En cada pliegue eligieron las 10 columnas con mayor F. Salieron casi siempre las mismas: <i>dias_previos, log_contactos_previos, contactado_antes, resultado_previo_success, resultado_previo_unknown, canal_unknown, hipoteca_yes, mes_may, mes_oct</i> (5 de 5) y <i>mes_mar</i> (4 de 5); en el pliegue 5 entró <i>estado_civil_married</i>. "
                         "Métricas OOF: 42 columnas = log loss <b>0,3151</b>, Brier 0,0895, AUC <b>0,7250</b>; 10 columnas = 0,3203, 0,0914, 0,7011."),
        ("Por qué así", "La selección F es un filtro: puntúa cada columna sola y se queda con las 10 mejores. Es estable, pero escoge columnas que dicen casi lo mismo (haber sido contactado antes). "
                        "Con 3 617 filas, el modelo completo puede usar las 42 columnas sin pasarse de ajuste, por eso gana por ~0,024 de AUC."),
        ("Cómo leer el resultado", "Gana el modelo completo en las tres métricas. La selección gana solo en simplicidad."),
        ("Diferencia con PCA / limitación", "<b>Selección:</b> conserva columnas originales (interpretables) y usa la respuesta. <b>PCA:</b> inventa combinaciones nuevas de todas las columnas y no mira la respuesta. "
                                            "<b>Limitación del filtro:</b> mira cada columna por separado, así que no ve redundancia ni efectos conjuntos; además k = 10 se fijó sin optimizar."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué la selección se hace dentro del pliegue?",
                                                 "Porque si se elige con todos los datos, la respuesta del pliegue de validación ya influyó en qué columnas quedan; la nota saldría optimista.")),
        ("Dónde lo estudiaste", nb("", "C33_FeatureEngineering", "C2–C5, E1–E4") + " · " + nb("", "C32_ReduccionDimensionalidad", "Parte C") + " · " + nb("", "C37_SVM", "A2") + " (sin penalización)"),
    ]))
    s.append(bloque("2(b) · Tabla de nueve modelos y elección del mejor (11 pts)", [
        ("Qué pedía", "Comparar todas las configuraciones con sus OOF (log loss, Brier, AUC; SVM solo AUC), incluir la referencia y el log loss de entrenamiento de los árboles, y elegir el probabilístico de menor log loss OOF."),
        ("Qué hicieron", "Ajustaron cada modelo en los 5 pliegues y juntaron las OOF. Resultado (log loss / Brier / AUC): <b>Gradient Boosting 0,311 / 0,0896 / 0,744</b>; Random Forest 0,312 / 0,0898 / 0,740; "
                         "Logística (42) 0,315 / 0,0895 / 0,725; Logística (10 F) 0,320 / 0,0914 / 0,701; Árbol prof. 3 0,330 / 0,0932 / 0,636; Árbol prof. 6 0,339 / 0,0921 / 0,664; "
                         "Referencia 0,357 / 0,1020 / 0,50; Naive Bayes 2,256 / 0,1648 / 0,702; SVM solo AUC 0,685. "
                         "Boosting elegido: <b>Gradient Boosting, 200 árboles, profundidad 3, tasa de aprendizaje 0,05</b>."),
        ("Por qué así", "La regla se fijó antes de mirar: gana el <b>menor log loss OOF</b> entre los modelos que dan probabilidades. La <b>referencia</b> (predecir siempre la tasa de contratación de las filas de ajuste) es el «modelo tonto»: todo lo que no la supere no aporta. "
                        "La SVM da un puntaje, no una probabilidad, por eso solo entra con AUC."),
        ("Cómo leer el resultado", "Casi todos superan a la referencia, pero la mejora es moderada (0,357 → 0,311, unos 13 %): sin <i>duracion</i> hay poca señal (AUC ≈ 0,74). "
                                   "Las diferencias entre los tres primeros son pequeñas (0,001–0,004), pero la regla decide. Naive Bayes es peor que la referencia."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué eligieron por log loss y no por AUC?",
                                                 "Porque luego usamos las probabilidades para decidir con costos; el AUC solo mira el orden, el log loss también mira si las probabilidades son creíbles.")),
        ("Dónde lo estudiaste", nb("", "C38_EvaluacionModelos", "A3, B2, B3") + " · " + nb("", "C34_ArbolesDecision", "C1–C2") + " · " + nb("", "C35_RandomForest", "B1, B3") + " · " + nb("", "C36_Boosting", "C3, D1") + " · " + nb("", "C37_SVM", "B3, C3") + " · " + nb("", "C40_NaiveBayes", "B3–B4")),
    ]))
    enc(P("2(c) · Interpretación: cinco ideas (8 pts)", "h3"))
    s.append(bloque("2(c)-1 y 2 · Árboles y sobreajuste; bagging, Random Forest y boosting", [
        ("Qué pedía", "Explicar con los números de la tabla qué pasa con la profundidad de los árboles y en qué se diferencian bagging, Random Forest y boosting."),
        ("Árboles: qué hicieron", "Compararon el log loss con que el árbol se «examina a sí mismo» (entrenamiento) y el OOF. Árbol prof. 3: 0,320 → 0,330 (brecha <b>0,010</b>). Árbol prof. 6: 0,302 → 0,339 (brecha <b>0,037</b>)."),
        ("Árboles: por qué", "<b>Sobreajuste</b> = aprenderse de memoria los datos de estudio. El árbol profundo se aprende mejor el entrenamiento (0,302) pero rinde peor con datos nuevos: la brecha crece. "
                             "Sus hojas chicas dan probabilidades extremas e inestables, que el log loss castiga. Aun así ordena un poco mejor (AUC 0,664 vs 0,636): un árbol de 3 niveles solo da ≤ 8 probabilidades distintas."),
        ("Bagging, RF, boosting", "<b>Bagging:</b> muchos árboles profundos, cada uno con una muestra al azar, y se promedian; reduce la <b>varianza</b> (la inestabilidad), pero los árboles se parecen entre sí. "
                                   "<b>Random Forest:</b> igual, pero en cada corte solo mira √p ≈ 6 columnas al azar; así los árboles se <b>decorrelacionan</b> y el promedio mejora. "
                                   "<b>Boosting:</b> árboles chicos puestos <b>uno tras otro</b>, cada uno corrige los errores del conjunto anterior (con tasa de aprendizaje 0,05); reduce sobre todo el <b>sesgo</b>. "
                                   "Analogía: bagging/RF = un comité que vota; boosting = un alumno que repasa sus propios errores."),
        ("Cómo leer", "Boosting (0,311) y RF (0,312) quedan primeros y muy cerca: captan relaciones no lineales que la logística no ve."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué el árbol de profundidad 6 es peor que el de 3 en log loss si ordena mejor?",
                                                 "Porque se sobreajusta: su log loss de entrenamiento mejora pero el OOF empeora (brecha 0,037 vs 0,010); ordenar mejor no es lo mismo que dar buenas probabilidades.")),
        ("Dónde lo estudiaste", nb("", "C34_ArbolesDecision", "C2") + " · " + nb("", "C35_RandomForest", "A3, B2–B3") + " · " + nb("", "C36_Boosting", "A3")),
    ]))
    s.append(bloque("2(c)-3, 4 y 5 · Naive Bayes, SVM y orden vs probabilidades", [
        ("Naive Bayes: qué pasó", "Log loss <b>2,26</b> (peor que la referencia, 0,357) y Brier 0,165, aunque el AUC es 0,70. El <b>75 %</b> de sus probabilidades OOF es &lt; 0,01 y el <b>11 %</b> es &gt; 0,99; su probabilidad media es 0,16 (la real es 0,115)."),
        ("Naive Bayes: por qué", "Supone <b>independencia condicional</b>: que cada columna aporta información propia sin repetir a las otras. Aquí <i>dias_previos, contactado_antes, log_contactos_previos</i> y <i>resultado_previo_unknown</i> son casi la misma información; al multiplicarlas, la evidencia se cuenta varias veces "
                                 "(como que cuatro amigos repitan el mismo chisme y parezca verdad firme) y las probabilidades se van a los extremos. Además, la versión gaussiana trata columnas 0/1 como si fueran campanas normales, lo que no corresponde y genera razones enormes."),
        ("SVM: C, gamma, escala", "<b>C</b> = cuánta multa se pone a los errores (alto: ajusta más; bajo: margen amplio). <b>Gamma</b> = alcance de cada punto (alto: fronteras muy locales; bajo: casi rectas). "
                                   "En su grilla de AUC OOF: γ = 0,1/p da 0,63–0,68 y γ = 1/p o 10/p da 0,69–0,70; el mejor valor es 0,700 y con C = 1, γ = 1/p el AUC es <b>0,685</b>. "
                                   "Sin estandarizar las numéricas, el AUC cae a <b>0,599</b> porque <i>saldo</i> domina las distancias. La SVM entrega un puntaje (distancia a la frontera), no una probabilidad: por eso solo AUC."),
        ("Orden vs probabilidades", "<b>No:</b> mejor ordenamiento no implica mejores probabilidades. El AUC solo mira el <b>orden</b> (no cambia si estiras los puntajes); log loss y Brier miran si «p = 0,2» realmente ocurre ~20 % de las veces. "
                                    "Prueba: Naive Bayes tiene AUC 0,70 (≈ la logística de 10 columnas, 0,701) pero el peor log loss; y el árbol 6 tiene mejor AUC que el 3 pero peor log loss."),
        ("Pregunta del profesor", pregunta_prof("Si dos modelos tienen igual AUC, ¿sirven igual para decidir con costos?",
                                                 "No necesariamente: para fijar un umbral por costo importan las probabilidades; Naive Bayes y la logística tienen AUC parecido, pero Naive Bayes está muy mal calibrado.")),
        ("Dónde lo estudiaste", nb("", "C40_NaiveBayes", "C3–C4, D2") + " · " + nb("", "C37_SVM", "B2–B4, C3") + " · " + nb("", "C38_EvaluacionModelos", "B2–B4") + " (calibración)"),
    ]))

    # ================================================================== PREGUNTA 3
    enc(P("Pregunta 3 · Evaluación y decisión (10 puntos)", "h2"))
    s.append(bloque("3(a) · Umbral que minimiza el costo (4 pts)", [
        ("Qué pedía", "Con las OOF del modelo elegido probar umbrales 0,05; 0,10; …; 0,95, y quedarse con el que minimiza C(t) = FP + 5·FN."),
        ("Qué hicieron", "Para cada umbral contaron TP, FP, FN y TN y calcularon el costo. El mínimo está en <b>t = 0,15</b>: costo <b>1 492</b> (0,412 por registro), con 192 TP, 367 FP, 225 FN, 2 833 TN; sensibilidad 46 % y especificidad 88,5 %; se llamaría a 559 de 3 617 clientes. "
                         "Referencias: no llamar a nadie cuesta 2 085 (5·417) y llamar a todos 3 200; el umbral 0,50 cuesta 1 816."),
        ("Por qué así", "Equivocarse llamando (FP) cuesta 1, pero dejar ir a un comprador (FN) cuesta 5. Entonces conviene llamar aunque la probabilidad sea baja: aprox. cuando p &gt; 1/(1+5) ≈ 0,167. "
                        "El 0,15 empírico calza con eso, señal de que las probabilidades del modelo están razonablemente calibradas."),
        ("Cómo leer el resultado", "Bajar el umbral aumenta los FP (917 con 0,10); subirlo aumenta los FN. La curva es plana entre 0,15 y 0,25 (1 492 a 1 597), así que la elección es robusta."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué el umbral es 0,15 y no 0,50?",
                                                 "Porque perder un cliente cuesta 5 veces más que una llamada inútil; con esa asimetría conviene llamar a más gente, aunque su probabilidad sea baja.")),
        ("Dónde lo estudiaste", nb("", "C38_EvaluacionModelos", "B5, C7") + " (costo y umbral teórico) · " + nb("", "C30_PreparacionML", "D2–D3")),
    ]))
    s.append(bloque("3(b) · Evaluación final en la prueba y recomendación (6 pts)", [
        ("Qué pedía", "Reajustar con todo el entrenamiento y evaluar una vez en la prueba (log loss, Brier, AUC, ROC); comparar t = 0,15 con 0,50 (confusión, sensibilidad, especificidad, C/n) incluyendo la referencia; recomendar y decir qué falta verificar."),
        ("Qué hicieron", "Registraron modelo y umbral <b>antes</b> de abrir la prueba. Reajustaron Gradient Boosting con los 3 617 registros y lo evaluaron en 904. Prueba: log loss <b>0,309</b>, Brier <b>0,0886</b>, AUC <b>0,750</b> (referencia 0,357 / 0,102 / 0,50). "
                         "Con t = 0,15: 55 TP, 111 FP, 49 FN, 689 TN, sensibilidad 52,9 %, especificidad 86,1 %, C/n = <b>0,394</b>. Con t = 0,50: 21 TP, 16 FP, 83 FN, sensibilidad 20,2 %, C/n = 0,477. Referencia: no llama a nadie, C/n = 0,575. Llamar a todos: 0,885."),
        ("Por qué así", "La prueba se usa una sola vez para no «hacerse trampa». Sus métricas son parecidas a las OOF (0,311 / 0,0895 / 0,744): el procedimiento de validación estimó bien. "
                        "Entre los llamados con t = 0,15, contrata 55 de 166 (≈ 33 %), casi el triple de la tasa base (11,5 %); en el 10 % de clientes con mayor probabilidad contrata el 44 %."),
        ("Recomendación", "Ordenar clientes por probabilidad y llamar a quienes tengan p ≥ 0,15 (o a los deciles más altos si falta capacidad). Falta verificar: los <b>costos reales</b> (el 1:5 es hipotético) y la capacidad del call center; "
                                    "<b>validez en el tiempo</b> (datos de campañas pasadas; <i>contactos_campana</i> incluye la llamada registrada); <b>causalidad</b> (el modelo dice quién contrata <i>si lo llaman</i>, no si llamar causa la compra); "
                                    "<b>incertidumbre</b> (solo 104 positivos en prueba) → conviene un piloto."),
        ("Pregunta del profesor", pregunta_prof("¿Qué le dirían al gerente?",
                                                 "Que con el modelo y t = 0,15 se detecta a la mitad de los compradores llamando al 18 % de la cartera y se baja el costo hipotético frente a no usar modelo (0,394 vs 0,575 por registro), pero hay que validar costos reales y probar en un piloto.")),
        ("Dónde lo estudiaste", nb("", "C38_EvaluacionModelos", "A5, B3, B6") + " · " + nb("", "C37_SVM", "A4") + " · " + nb("", "C31_IntroML", "B3")),
    ]))
    enc(P("Pregunta 4 · Componentes principales y agrupamiento (8 puntos)", "h2"))
    enc(P("Aquí solo se usan <i>edad, saldo, contactos_campana</i> y <i>contactos_previos</i>, estandarizadas con media y desviación muestral.", "small"))
    s.append(bloque("4(a) · PCA (3 pts)", [
        ("Qué pedía", "Hacer PCA; dar la varianza explicada por PC1 y PC2, su suma y las cargas; graficar; decir qué resume y qué omite el plano."),
        ("Qué hicieron", "PC1 explica <b>27,7 %</b> y PC2 <b>26,7 %</b>: juntas <b>54,4 %</b> (PC3 23,4 %, PC4 22,3 %). Cargas de PC1: edad 0,58, saldo 0,63, contactos_campana −0,38, contactos_previos 0,35. "
                         "PC2: edad −0,43, saldo −0,31, contactos_campana −0,58, contactos_previos 0,62."),
        ("Por qué así", "PCA busca los ejes sobre los que los datos se «estiran» más. Aquí las cuatro variables casi no se correlacionan (la mayor es 0,097 entre edad y saldo), así que no hay nada que comprimir: "
                        "cada componente explica ~25 %, justo lo esperable si fueran independientes. Es como intentar resumir cuatro mapas distintos en una sola foto."),
        ("Cómo leer el resultado", "El plano conserva 54 %; se pierde 46 %, y las distancias del gráfico pueden engañar. PCA no usa <i>contrata</i> (las dos clases se superponen) ni las variables categóricas, que son las más predictivas."),
        ("Pregunta del profesor", pregunta_prof("¿Por qué PC1 + PC2 explican tan poco?",
                                                 "Porque las variables son casi independientes entre sí (correlación máxima 0,097); PCA solo comprime bien cuando hay variables que se parecen.")),
        ("Dónde lo estudiaste", nb("", "C32_ReduccionDimensionalidad", "C2, C4") + " · " + nb("", "C31_IntroML", "C3")),
    ]))
    s.append(bloque("4(b) · K-means (5 pts)", [
        ("Qué pedía", "K-means con k = 2, 3, 4 y ≥ 30 inicializaciones; comparar WSS y silueta, elegir k, dar tamaños y perfiles, graficar en el plano PCA, tasa de contratación por grupo y explicar por qué un grupo no es una clase predicha."),
        ("Qué hicieron", "Con 30 inicializaciones y semilla 2026: k = 2 silueta 0,301 (WSS 11 893); k = 3 silueta 0,334 (WSS 9 685); k = 4 silueta <b>0,363</b> (WSS 7 665). Eligieron <b>k = 4</b>. "
                         "Grupos: 1) jóvenes de saldo bajo (n = 2 004, edad 34, contratación 10,8 %); 2) mayores con más saldo (n = 1 289, edad 52, 11,4 %); 3) con historial previo (n = 177, 6,6 contactos previos, <b>26,6 %</b>); "
                         "4) muy insistidos (n = 147, 14,5 contactos en esta campaña, <b>4,8 %</b>). Tasa global: 11,5 %."),
        ("Por qué así", "El WSS siempre baja al aumentar k, así que no sirve para elegir; la silueta premia grupos compactos y bien separados. Una silueta de 0,36 es de estructura débil a moderada, esperable con variables poco relacionadas. "
                        "K-means <b>no usa</b> <i>contrata</i>: arma grupos por parecido en las 4 variables. Después se mira cuánto contrata cada grupo, como descripción."),
        ("Cómo leer el resultado", "Ningún grupo pasa de 27 % de contratación; una regla «grupo → clase» predeciría siempre «no contrata». Sirven para segmentar (priorizar el grupo con historial, no insistir con el muy llamado), no para clasificar."),
        ("Pregunta del profesor", pregunta_prof("¿Un grupo de K-means es un cliente «que contrata»?",
                                                 "No: el algoritmo nunca vio la respuesta; en cada grupo hay compradores y no compradores. Sirve para describir perfiles, no para predecir.")),
        ("Dónde lo estudiaste", nb("", "C39_Clustering", "") + " · " + nb("", "C31_IntroML", "C1–C2") + " · Guía " + mono("C39_Guia_Clustering.pdf")),
    ]))
    enc(P("Pregunta 5 · Detección de anomalías (6 puntos)", "h2"))
    s.append(bloque("5(a) · Distancia de Mahalanobis y corte chi-cuadrado (3 pts)", [
        ("Qué pedía", "Calcular d² de Mahalanobis, usar como corte el cuantil 0,99 de una χ² con 4 grados de libertad, informar corte, cantidad y %, y mostrar los dos registros más lejanos."),
        ("Qué hicieron", "Corte χ²(0,99; 4) = <b>13,277</b>. Marcados: <b>189 de 3 617 (5,23 %)</b>. Los dos más lejanos: id 1477 (60 años, saldo <b>71 188</b>, d² = 518,6) e id 1542 (35 años, <b>50</b> contactos en la campaña, d² = 224,3); ambos tienen contrata = 0."),
        ("Por qué así", "Mahalanobis es una «distancia con sentido»: mide cuántas desviaciones está un punto del centro, descontando cómo se mueven juntas las variables. Si los datos fueran normales, d² seguiría una χ² con 4 g.l., y el cuantil 0,99 sería un corte que marcaría ~1 %."),
        ("Cómo leer el resultado", "Se marcó 5,2 %, cinco veces lo esperado: la señal de que los datos no son normales."),
        ("Pregunta del profesor", pregunta_prof("¿De dónde sale el 13,277?",
                                                 "Es el valor de una χ² con 4 grados de libertad (una por variable) que deja 99 % de probabilidad por debajo.")),
        ("Dónde lo estudiaste", nb("", "C41_Anomalias", "A3") + " · Guía " + mono("C41_Guia_Anomalias.pdf")),
    ]))
    s.append(bloque("5(b) · ¿Errores, 1 % garantizado y qué revisar? (3 pts)", [
        ("Qué pedía", "Explicar por qué son inusuales los dos registros, si eso prueba errores, si el corte garantiza marcar 1 %, y qué revisar antes de eliminar."),
        ("Qué hicieron", "id 1477: el saldo es el máximo de la muestra, a 22,8 desviaciones de la media (mediana 438 €; percentil 99 ≈ 14 200 €). id 1542: 50 contactos, a 14,9 desviaciones (mediana 2; percentil 99 = 16). "
                         "En ambos, una sola variable es extrema. Mostraron asimetría (saldo 7,1; contactos_campana 4,9; contactos_previos 5,9) y curtosis en exceso (99, 40, 51)."),
        ("Por qué así", "<b>Atípico ≠ error</b>: un jubilado con una indemnización puede tener 71 188 €; alguien pudo ser llamado 50 veces por política de insistencia. El corte de 1 % solo vale si los datos son normales multivariados y con parámetros conocidos; "
                        "aquí hay colas pesadas y la media y la covarianza se calculan con los mismos atípicos. Por eso salió 5,23 % y no 1 %."),
        ("Qué revisarían", "(1) coherencia con otras fuentes (contabilidad, call center); (2) si el valor es posible según el negocio; (3) resultados con y sin el registro; (4) si los atípicos se relacionan con la respuesta (los marcados contratan 14,3 % vs 11,4 %); "
                           "(5) documentar la decisión. Solo eliminar si hay error comprobado."),
        ("Pregunta del profesor", pregunta_prof("¿Eliminarían el registro 1477?",
                                                 "No, no mientras no se compruebe que es un error: es un valor posible y quitarlo sin revisar podría sesgar el análisis; primero se contrasta con otras fuentes y se comparan resultados con y sin él.")),
        ("Dónde lo estudiaste", nb("", "C41_Anomalias", "A3, A7, D1") + " · Guía C41, recuadro «¿atípico, error o evento real?»"),
    ]))

    # ================================================================== ÚLTIMA PÁGINA
    s.append(PageBreak())
    s.append(band("Lo que conviene corregir antes de entregar", "Son correcciones que el grupo debe redactar por su cuenta; aquí solo se describe qué ajustar y por qué", RED))
    s.append(Spacer(1, 4))
    corr = [
        ["Dónde", "Qué conviene arreglar", "Por qué"],
        ["<b>2(c)</b> SVM", "Agregar por qué la SVM (AUC 0,685) queda <b>por debajo de la logística (0,725)</b>.",
         "Con C y gamma fijos el AUC es 0,685 y el máximo de la grilla es ≈ 0,70: igual no alcanza a la logística. Falta decirlo explícitamente."],
        ["<b>1(c)</b> escala", "Corregir la frase que dice que la selección por F de ANOVA es sensible a la escala. Lo mismo para Mahalanobis.",
         "El estadístico F univariado y la distancia de Mahalanobis <b>no cambian</b> si se reescalan las variables (son invariantes a la escala). Los que sí dependen de la escala son SVM, PCA y K-means."],
        ["<b>3(b)</b> perfiles", "La frase sobre «canal celular, ciertos meses» <b>no está respaldada</b> por ninguna salida del notebook: mostrar las tasas de contratación por canal y mes, o quitar la frase.",
         "Una afirmación sin número que la respalde puede no aguantar una pregunta del profesor."],
        ["<b>4(b)</b> k = 4", "Reconocer que k = 4 está en el <b>borde</b> del rango 2–4 (la silueta sube con k: 0,301 → 0,334 → 0,363), y que el 4,8 % del grupo de 147 clientes se apoya en unos <b>7 contratos</b>.",
         "No se sabe si con k = 5 la silueta seguiría subiendo; y una tasa basada en ~7 casos es muy inestable."],
        ["Notebook", "Hacer <b>«Reiniciar y ejecutar todo»</b> para que ninguna celda (la 14 en particular) conserve una salida de error guardada.",
         "El enunciado pide un notebook ejecutado; una celda con error o salida vieja resta credibilidad."],
        ["Cifras del texto", "Tras re-ejecutar, copiar al texto las cifras de las salidas: hoy el texto de 2(b) dice log loss 0,3110 y AUC 0,7443 (la salida: 0,3111 y 0,7442); el de 3(b) dice FP 112, TN 688, 167 contactados, C = 357 (la salida: 111, 689, 166, 356) y log loss 0,3088 (salida 0,3087); el de 3(a) dice 2 355 FP con t = 0,05 (salida 2 356).",
         "Diferencias mínimas, pero conviene que texto y salida coincidan."],
        ["Entrega", "Armar <b>Grupo_36_Taller3.zip</b> solo con el notebook y los 2 .csv. <b>No</b> incluir los dos PDF de estudio.",
         "Eso pide el enunciado: notebook ejecutado y dos .csv."],
    ]
    s.append(table(corr, [CONTENT_W * 0.12, CONTENT_W * 0.50, CONTENT_W * 0.38], header_color=RED))
    s.append(Spacer(1, 5))
    s.append(P("Autoevaluación oral (10 preguntas) · respuestas en la página siguiente", "h2"))
    qs = ["¿Por qué se excluye <i>duracion</i> y qué es la fuga de información?",
          "¿Qué significan el −1 de <i>dias_previos</i> y los saldos negativos? ¿Por qué no basta una exactitud de 88,5 %?",
          "¿Por qué la selección F y el escalado se aprenden solo en las filas de ajuste de cada pliegue?",
          "¿Qué es una predicción OOF y por qué se usa?",
          "¿Por qué la logística de 42 columnas le gana a la de 10 en 2(a)?",
          "¿Por qué se elige Gradient Boosting y por qué el árbol de prof. 6 se sobreajusta más?",
          "¿Por qué Naive Bayes tiene AUC 0,70 pero log loss 2,26?",
          "¿Por qué el umbral es 0,15 y no 0,50? ¿Qué rol tiene el 1/(1+5)?",
          "¿Qué se concluye de la prueba (log loss 0,309; AUC 0,750; C/n 0,394) y qué falta verificar?",
          "¿Por qué PC1 + PC2 solo explican 54 %, un grupo de K-means no es una clase y Mahalanobis no garantiza 1 %?"]
    s.append(bullets([f"<b>{i}.</b> {q}" for i, q in enumerate(qs, 1)], "cell", C42, marker="•"))
    s.append(PageBreak())
    s.append(band("Respuestas breves a la autoevaluación", "Para chequearse después de intentar responder en voz alta", GREEN))
    s.append(Spacer(1, 4))
    ans = ["Porque la duración se conoce al colgar y delata la respuesta (554 s vs 228 s); usarla da un modelo que no se puede aplicar antes de llamar.",
           "−1 es un código para «sin campaña previa» (coincide con 0 contactos previos); saldo negativo = cliente sobregirado, posible. Con 88,5 % de «no», siempre decir «no» acierta mucho y detecta a 0 compradores.",
           "Para que el pliegue de examen no influya en el entrenamiento; si no, las métricas salen infladas.",
           "La predicción de un cliente hecha por un modelo que no lo vio al entrenar; da una nota honesta usando todos los datos.",
           "Porque la selección descarta información útil (otros meses, edad, <i>prestamo</i>…) y filtra columna por columna; con 3 617 filas el modelo completo no se sobreajusta (AUC 0,725 vs 0,701).",
           "Tiene el menor log loss OOF (0,311). El árbol 6 se aprende el entrenamiento (0,302) pero rinde peor fuera (0,339): brecha 0,037 vs 0,010.",
           "Ordena bien pero sus probabilidades son extremas (75 % &lt; 0,01; 11 % &gt; 0,99) por contar evidencia repetida y tratar 0/1 como normales.",
           "Porque un FN cuesta 5 FP: conviene llamar si p &gt; 1/6 ≈ 0,167; el mínimo empírico (0,15) coincide.",
           "El modelo discrimina moderadamente y el umbral 0,15 rebaja el costo (0,394 vs 0,477 con 0,50 y 0,575 sin modelo). Falta: costos reales, capacidad, validez temporal, causalidad, incertidumbre (104 positivos).",
           "Las 4 variables casi no se correlacionan (máx. 0,097), así que no se comprimen bien; K-means no usa la respuesta, solo agrupa por parecido; el corte χ² solo da 1 % si los datos son normales (aquí marcó 5,23 %)."]
    s.append(table([["#", "Respuesta breve"]] + [[str(i), a] for i, a in enumerate(ans, 1)],
                   [CONTENT_W * 0.06, CONTENT_W * 0.94], header_color=GREEN))
    s.append(Spacer(1, 5))
    s.append(box([P("Recordatorio: este documento es para <b>entender</b> la solución; no se entrega y no reemplaza las respuestas del grupo.", "cell")], bg=GREEN_BG, border=GREEN, pad=5))
    return s


HALFW = (CONTENT_W - 6 * mm) * 0.5

if __name__ == "__main__":
    build(SALIDA, contenido(), "Explicación de la solución · Taller #3 · Grupo 36 (material de estudio, no es entrega)")
    print("OK")
