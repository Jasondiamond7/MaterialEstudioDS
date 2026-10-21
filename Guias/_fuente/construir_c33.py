"""Clase 33 · Feature engineering y selección de variables: guía, apuntes y cuaderno."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_glm import code, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C33 = colors.HexColor("#EA580C")
C33_BG = colors.HexColor("#FFF7ED")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + notebook Python del profesor"


def topic_page(s, titulo, idea, conceptos, figura, pie, formulas, py, salida, cuidado):
    s.append(band(f"Clase 33 · {titulo}", FUENTE, C33))
    s.append(Spacer(1, 7))
    s.append(box([P(idea, "big")], bg=SOFT, border=C33))
    s.append(Spacer(1, 6))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C33))
    s.append(Spacer(1, 4))
    if figura:
        s.append(img(figura, width=CONTENT_W * 0.97))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 5))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF),
            Spacer(1, 5), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C33), Spacer(1, 5),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER,
                 width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 33 · Feature engineering y selección de variables",
                  "Profesor Eduardo Rodríguez · Etapa 7 del roadmap · dataset: fuga de clientes de una telco", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Más variables no es un mejor modelo. El trabajo real, que según el profesor es el 70–80% de un proyecto, es "
                    "<b>limpiar con criterio de negocio</b>, <b>construir variables con sentido</b> y <b>seleccionar sin hacer "
                    "trampa</b>. Y la trampa más común es invisible: si una selección mira el target fuera de la validación "
                    "cruzada, el AUC que reportas es inventado.", "body")],
                 bg=C33_BG, border=C33))
    s.append(Spacer(1, 6))
    s.append(P("El flujo de trabajo", "h2"))
    s.append(img("c33_mapa.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 4))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Regresión logística y lectura de AUC (Carpeta 2)",
                     "Validación cruzada y data leakage del escalador (Clases 30–32)",
                     "Correlación, ANOVA F y p-valor (Etapa 0 y Carpeta 1)",
                     "pandas: groupby, describe, isna"], "cell", C33)]
    right = [P("Hacia dónde lleva", "h3"),
             bullets(["<b>Clase 34 · árboles:</b> la ganancia de información es la información mutua de hoy",
                      "<b>Clase 35 · Random Forest:</b> resuelve el XOR que la logística no ve",
                      "<b>Clase 36 · Boosting:</b> importancia de variables y SHAP como selección embebida",
                      "<b>Proyecto:</b> el checklist de esta clase aplica a cualquier modelo"], "cell", C33)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 5))
    s.append(table([["Variable", "Qué es", "Variable", "Qué es"],
                    ["numero_fallas", "fallas de llamada", "frec_sms", "n° de SMS"],
                    ["reclamos", "si reclamó (0/1)", "n_contactos", "n° de contactos distintos"],
                    ["antiguedad", "meses como cliente", "tipo_plan", "1 o 2 (no se define)"],
                    ["monto", "monto cobrado", "estado", "1 o 2 (no se define)"],
                    ["segundos", "segundos de uso", "edad", "edad del cliente"],
                    ["frec_llamadas", "n° de llamadas", "clv", "valor del cliente (customer lifetime value)"],
                    ["fuga", "TARGET: 1 = el cliente se fue (15.7%)", "", ""]],
                   [28 * mm, 52 * mm, 26 * mm, CONTENT_W - 106 * mm]))
    s.append(PageBreak())

    topic_page(
        s, "Limpiar con criterio de negocio",
        "Ninguna técnica de selección arregla un dato imposible. La limpieza va antes, y el umbral lo pone el negocio, "
        "no la estadística.",
        [("Valores imposibles", "Segundos negativos (mínimo −14.645), llamadas negativas, antigüedad −40, edades de −13 y 233 años. Los datos los traen a propósito."),
         ("El umbral es de negocio", "«Edad entre 15 y 100 no sale de los datos: sale de saber que una telco no contrata a alguien de 233 años». Los 108 clientes de 0–14 años podrían ser cuentas a nombre de un menor: es una decisión, no un error."),
         ("No borrar filas", "Es la advertencia más repetida del profesor: borrar una fila borra información de TODAS sus variables. Mejor: marcar como NaN, imputar o crear una categoría propia."),
         ("Mediana, no media", "La mediana es robusta a los extremos. Un solo −14.645 arrastra la media de segundos; la mediana casi no se mueve."),
         ("Revisa también", "Nulos (6 columnas con faltantes), 19 filas duplicadas y la ventana temporal de los datos antes de imputar mecánicamente.")],
        "c33_limpieza.png",
        "Conteos reales de la base Churn. La barra naranja (edad 0–14) no es un error evidente: tratarla es una decisión de negocio.",
        ["imposible → NaN → imputar con la mediana",
         "edad válida: 15 ≤ edad ≤ 100",
         "variables: solo información anterior al target"],
        ["d.describe().T[['min', 'max']]",
         "d.loc[d.segundos < 0, 'segundos'] = np.nan",
         "d['edad'] = d.edad.where(",
         "    d.edad.between(15, 100))",
         "d.fillna(d.median(numeric_only=True))"],
        ["3.150 clientes · 495 se fugan (15.71%)",
         "Imposibles: segundos 18 · llamadas 12 · antigüedad 9 · SMS 1 · edad 3",
         "Nulos: segundos 40 · llamadas 19 · reclamos 13 · antigüedad 5 · SMS 4 · contactos 2"],
        ["Un «pájaro sobre el sensor» (la anécdota del profesor sobre Anglo American) genera outliers falsos; un fraude "
         "genera outliers reales. Antes de tocar un valor, pregunta de dónde viene."])

    topic_page(
        s, "Construir variables: de 12 a 50",
        "Con 12 columnas no hay nada que seleccionar. El feature engineering fabrica candidatas con sentido de negocio, "
        "y con ellas aparece el problema real.",
        [("Ratios", "Llamadas por mes, CLV por antigüedad, segundos por llamada… En la experiencia del profesor son «algo bien potente» y discriminan mejor que la variable cruda."),
         ("Épsilon en el denominador", "x / (y + 1e-6): un solo infinito por dividir por cero rompe todo el pipeline que viene después."),
         ("Otras familias", "Logaritmos (colas largas), indicadores binarios (tiene_reclamo, cliente_nuevo), cuartiles (clv_q1…q4), interacciones (reclamo × antigüedad) y one-hot (estado_2, plan_2)."),
         ("Redundancia por construcción", "reclamos y tiene_reclamo valen lo mismo (r = 1.000). Crear variables sin después seleccionar no mejora el modelo: lo infla."),
         ("WOE (adelanto)", "El profesor pide anotar el Weight of Evidence, usado en banca para discretizar antes de la logística. Se mencionó pero no se calculó en el notebook.")],
        None, None,
        ["ratio = numerador / (denominador + 1e-6)",
         "log(x + 1) para variables con cola larga",
         "13 ratios + 3 interacciones + 5 logs",
         "+ 5 binarias + 12 cuartiles + 2 one-hot",
         "+ 10 originales = 50 candidatas"],
        ["eps = 1e-6",
         "d['llamadas_por_mes'] = d.frec_llamadas / (d.antiguedad + eps)",
         "d['log_clv'] = np.log1p(d.clv)",
         "d['clv_q'] = pd.qcut(d.clv, 4, labels=False)",
         "pd.get_dummies(d, columns=['estado'], drop_first=True)"],
        ["12 originales → AUC 0.920 · 50 candidatas → AUC 0.941",
         "11 pares con |r| &gt; 0.90 entre las 50",
         "Pares idénticos: reclamos–tiene_reclamo, clv_alto–clv_q4"],
        ["Construir variables no mira el target, así que puede ir antes del split. Pero ojo con la ventana temporal: una "
         "variable que usa información posterior al momento de predecir es leakage aunque no use la columna fuga."])

    topic_page(
        s, "Irrelevante, redundante o útil solo en conjunto",
        "Una variable puede sobrar por tres motivos distintos, y cada uno pide una decisión distinta. El más traicionero: "
        "la que es inútil sola, pero útil acompañada.",
        [("Irrelevante", "No aporta nada, ni sola ni acompañada → se descarta."),
         ("Redundante", "Repite la información de otra (r &gt; 0.90) → se conserva una. Se desempata por información mutua y, a igualdad, por la más simple de mantener en producción, nunca por «la primera de la lista»."),
         ("Útil solo en conjunto", "numero_fallas tiene r = −0.009, información mutua 0 y p = 0.60. Sola da AUC 0.526 (azar), pero sumada a otras sube el modelo de 0.907 a 0.917."),
         ("El caso XOR", "Dos variables que solas no separan nada (logística AUC 0.510) separan perfecto juntas: con la interacción x1·x2, o con un Random Forest, AUC = 1.000."),
         ("La lección", "«Un p-valor alto no significa que la variable no sirve; significa que no sirve SOLA». Los filtros univariados matan estas variables.")],
        "c33_conjunto.png",
        "Izquierda: AUC real con validación cruzada. Derecha: datos sintéticos XOR; ninguna recta horizontal ni vertical separa los colores.",
        ["Redundancia: |r| &gt; 0.90 → quedarse con una",
         "Desempate: mayor información mutua",
         "y a igualdad, la más simple de explicar"],
        ["corr = X.corr().abs()",
         "pares = corr.where(np.triu(",
         "    np.ones(corr.shape), 1).astype(bool))",
         "pares.stack().loc[lambda s: s &gt; 0.9]"],
        ["Correlación con fuga: reclamos 0.51 · estado_2 0.32 · frec_llamadas −0.30",
         "numero_fallas: r −0.009 · p ANOVA 0.60",
         "XOR: logística 0.510 · + interacción 1.000 · RF 1.000"],
        ["Un árbol poco profundo también falla en XOR: es voraz y evalúa un corte a la vez. Lo verás en la Clase 34."])

    topic_page(
        s, "Filter, wrapper y embedded",
        "Hay tres familias de selección. No compiten: son etapas de un embudo. Filtro para bajar de miles a cientos, embedded "
        "para bajar a decenas y wrapper solo si sobra cómputo.",
        [("Filter", "Mide cada variable con una métrica, sin entrenar el modelo: varianza, correlación, información mutua (MI), ANOVA F, χ². Barato, pero no ve interacciones."),
         ("Información mutua", "Detecta relaciones no lineales (es 0 solo si hay independencia). En sklearn se estima con vecinos: cambia con la semilla, así que fija random_state."),
         ("Wrapper", "Entrena el modelo con distintos subconjuntos (RFE, forward, backward). Ve interacciones, pero es caro: con 50 variables hay 2<super>50</super> ≈ 10<super>15</super> subconjuntos, unos 35.702 años a 1.000 modelos por segundo."),
         ("Embedded: Lasso", "La penalización L1 lleva coeficientes exactamente a 0. En sklearn, C es el INVERSO de la penalización: C chico deja menos variables. Ridge (L2) no produce ceros."),
         ("Top-k ingenuo", "Las 4 mejores por MI dan AUC 0.804 porque «gastan» dos cupos en segundos y log_segundos. Cuatro elegidas sin redundancia dan 0.893. Con k ≥ 6 la ventaja se diluye.")],
        "c33_lasso.png",
        "Izquierda: variables que sobreviven en el Lasso según C. Derecha: AUC con validación cruzada de cada estrategia en la clase.",
        ["Var(binaria) = π(1 − π)",
         "I(X; Y) ≥ 0 · = 0 solo si son independientes",
         "F = CM entre / CM dentro",
         "n° de subconjuntos = 2<super>p</super> − 1",
         "Lasso: C chico → más coeficientes en 0"],
        ["VarianceThreshold(threshold=0.01)",
         "mutual_info_classif(X, y, random_state=0)",
         "f_classif(X, y)",
         "RFECV(LogisticRegression(), scoring='roc_auc')",
         "LogisticRegression(penalty='l1',",
         "    solver='liblinear', C=0.05)"],
        ["VarianceThreshold: 0 descartadas",
         "RFECV elige 42 de 50 · Lasso (C = 0.05) deja 27",
         "C: 0.01 → 13 · 0.1 → 33 · 1.0 → 45 variables",
         "Lasso es inestable: en 5 particiones solo 16 variables aparecen siempre"],
        ["Escala las variables (StandardScaler) antes de RFE o Lasso: si no, el tamaño del coeficiente depende de las unidades y "
         "no de la importancia.", "El Lasso requiere solver='liblinear' o 'saga'; el solver por defecto no admite L1."])

    topic_page(
        s, "La trampa: dónde vive la selección",
        "Todo lo que mira el target (filtros, desempates, RFE, Lasso) debe hacerse DENTRO de cada fold de la validación "
        "cruzada. Si no, el modelo no generaliza: recuerda.",
        [("El experimento", "100 clientes × 1.000 variables de puro ruido, con un target al azar. El AUC verdadero es 0.5."),
         ("Selección fuera del CV", "Elegir las 20 mejores con todos los datos y después validar da AUC 0.882: 38 puntos inventados de la nada."),
         ("Selección dentro del pipeline", "Con SelectKBest dentro de un Pipeline pasado a cross_val_score, el AUC es 0.514: la verdad."),
         ("Por qué pasa", "Con 1.000 variables y α = 0.05 se esperan α·p = 50 falsos positivos. El fold de validación contiene justo los casos que hicieron parecer buenas a esas variables."),
         ("Extiende la regla de la Clase 32", "Ya sabías que el escalador se ajusta solo con train. La misma regla aplica a cualquier decisión que use el target o estadísticas del conjunto (media, mediana).")],
        "c33_leakage.png",
        "Resultado real del notebook de la clase. La misma métrica, los mismos datos: lo único que cambia es dónde vive la selección.",
        ["Falsos positivos esperados = α · p",
         "ANTES del split: limpieza, ratios, logs, one-hot",
         "DENTRO del fold: imputar, escalar, filtrar,",
         "    desempatar, RFE, Lasso"],
        ["pipe = Pipeline([",
         "    ('imp', SimpleImputer(strategy='median')),",
         "    ('esc', StandardScaler()),",
         "    ('sel', SelectKBest(mutual_info_classif, k=12)),",
         "    ('log', LogisticRegression(max_iter=1000))])",
         "cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')"],
        ["Ruido: fuera del CV 0.882 · dentro 0.514 · verdad 0.500",
         "Pipeline honesto con los datos reales: AUC medio ≈ 0.90",
         "Por fold: entre 0.887 y 0.923"],
        ["El profesor advierte: en la industria, un AUC de 0.84–0.85 en fuga ya es «extremadamente bueno». Un 0.90+ «de libro» "
         "suele esconder leakage. Desconfía de las métricas demasiado buenas."])

    # traductor / checklist
    s.append(band("Chuleta de Python y checklist del proyecto", "Lo que vas a usar en cualquier modelo de aquí en adelante", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["Quiero…", "En Python (scikit-learn)"],
                    ["Quitar variables casi constantes", "VarianceThreshold(threshold=0.01)"],
                    ["Ranking por información mutua", "mutual_info_classif(X, y, random_state=0)"],
                    ["Ranking por ANOVA F", "f_classif(X, y) → (F, p)"],
                    ["Quedarme con las k mejores", "SelectKBest(score_func, k=12)  (dentro del Pipeline)"],
                    ["Eliminación recursiva con CV", "RFECV(estimador, step=1, cv=5, scoring='roc_auc')"],
                    ["Lasso (selección embebida)", "LogisticRegression(penalty='l1', solver='liblinear', C=…)"],
                    ["Imputar dentro del fold", "SimpleImputer(strategy='median')"],
                    ["Encadenar pasos sin leakage", "Pipeline([...]) + cross_val_score(pipe, X, y, cv=skf)"],
                    ["Folds que respetan el 15.7% de fuga", "StratifiedKFold(5, shuffle=True, random_state=42)"]],
                   [62 * mm, CONTENT_W - 62 * mm]))
    s.append(Spacer(1, 10))
    s.append(P("Checklist metodológico", "h2"))
    checks = [("Limpia con criterio de negocio", "umbrales que explicarías a un gerente, no solo a un estadístico"),
              ("Separa el test ANTES de todo", "el test no se toca hasta el final"),
              ("Define la ventana temporal", "las variables vienen de antes del período del target («junio hacia atrás»)"),
              ("Construye variables con sentido", "ratios, logs, binarias; con épsilon en los denominadores"),
              ("Mide la redundancia", "pares con |r| &gt; 0.90; desempata por MI y simplicidad"),
              ("No mates variables por un p-valor alto", "pueden servir en conjunto"),
              ("Mete la selección en el Pipeline", "imputar, escalar, filtrar, RFE y Lasso dentro del fold"),
              ("Usa AUC, no exactitud", "con 15.7% de fuga, decir «nadie se va» acierta 84.3%"),
              ("Desconfía de métricas perfectas", "un AUC demasiado alto suele ser leakage"),
              ("Documenta por qué quedó cada variable", "es lo que más modelos salva en producción")]
    s.append(table([["", "Paso", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 62 * mm, CONTENT_W - 70 * mm]))
    s.append(PageBreak())

    # alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales, preguntas clave y cuándo pasar a la Clase 34", NAVY))
    s.append(Spacer(1, 8))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Diapositivas y notebook no siempre coinciden</b>", "cellb"),
                  bullets(["<b>Lasso:</b> la diapositiva 25 dice «25 de 50 sobreviven» y la 26 dice «0.938 con 25 variables». El notebook "
                           "ejecutado da <b>27 de 50</b> y <b>AUC 0.9396</b>. Lo reproduje: el valor correcto es el del notebook.",
                           "<b>Ranking de información mutua (diapositiva 22):</b> difiere levemente del notebook. No es un error: la MI se "
                           "estima con vecinos y cambia con la semilla y con el orden de las columnas.",
                           "<b>«6 vs 6» (diapositiva 23):</b> dice top-6 univariado AUC 0.8105 vs 6 elegidas a mano 0.9041. El notebook no tiene esa "
                           "celda: su top-6 por MI da <b>0.897</b> (no 0.81), y su selección sin redundancia automática da 0.888. Toma la "
                           "conclusión de la diapositiva con cautela: con 6 variables la ventaja de evitar redundancia ya no es clara.",
                           "<b>Correlación segundos–frec_llamadas:</b> 0.907 en la diapositiva y 0.935 en el notebook. No es un error: 0.907 es "
                           "con los datos crudos y 0.935 después de limpiar."], "cell", RED)],
                 bg=RED_BG, border=RED))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Otras observaciones</b>", "cellb"),
                  bullets(["El notebook de la clase carga <font name='Mono'>churn_telco.csv</font>, que no está en la carpeta. Los datos "
                           "son los mismos del Excel <font name='Mono'>Base de datos Churn.xlsx</font>; tu notebook guiado ya lee el Excel.",
                           "<b>tipo_plan</b> y <b>estado</b> nunca se definen (valen 1 o 2). Es justo la falta de diccionario de datos que el "
                           "profesor critica.",
                           "WOE / IV y KS / Gini se mencionan, pero no se calculan en esta clase. Están como adelanto en los apuntes.",
                           "La transcripción dice «ocho tipos de técnicas» en una frase confusa sobre redes neuronales. No corresponde a "
                           "ninguna lista de la clase: la taxonomía real es filter / wrapper / embedded."], "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 8))
    s.append(P("Preguntas de repaso", "h2"))
    preg = [("¿Por qué la limpieza va antes de la selección?", "Ninguna técnica corrige un dato imposible, y el criterio para limpiar es de negocio."),
            ("numero_fallas tiene p = 0.60. ¿La descartas?", "No sin probarla en conjunto: sola da AUC 0.53, pero sube el modelo de 0.907 a 0.917."),
            ("¿Qué diferencia filter, wrapper y embedded?", "Filter mide sin entrenar; wrapper entrena con subconjuntos; embedded selecciona mientras entrena (Lasso)."),
            ("¿Por qué el top-4 por MI rinde peor que 4 elegidas a mano?", "Gasta cupos en variables redundantes (segundos y log_segundos) y deja fuera reclamos."),
            ("¿Por qué seleccionar fuera del CV infla el AUC?", "Los casos de validación ya influyeron en qué variables se eligieron: el modelo recuerda, no generaliza."),
            ("En sklearn, ¿C chico en Lasso deja más o menos variables?", "Menos: C es el inverso de la penalización.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 78 * mm, CONTENT_W - 85 * mm]))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 34</b>", "cellb"), Spacer(1, 3),
                  bullets(["Detecto valores imposibles y justifico la regla de negocio que aplico",
                           "Construyo ratios con épsilon y sé qué familias de variables existen",
                           "Distingo irrelevante, redundante y útil en conjunto, con un ejemplo de cada una",
                           "Explico filter, wrapper y embedded, y cuándo usar cada uno",
                           "Armo un Pipeline donde toda la selección vive dentro del fold",
                           "Resuelvo los bloques 1 a 8 del cuaderno con al menos 80% de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C33_Guia_FeatureEngineering.pdf", s, "Clase 33 · Feature engineering y selección de variables")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#EA580C">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 33 · Feature engineering y selección de variables",
                  "Para leer antes de los ejercicios · lectura de ~60 min", C33))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("Limpiar primero", "Los valores imposibles se tratan con reglas de negocio, antes de seleccionar. Imputa con la mediana; no borres filas."),
                      ("Construir con sentido", "Ratios, logs, binarias, cuartiles. Épsilon en todo denominador."),
                      ("Tres razones para sobrar", "Irrelevante (se va), redundante (queda una), útil solo en conjunto (no la mates por un p-valor)."),
                      ("Tres familias", "Filter (rápido, univariado), wrapper (caro, ve combinaciones), embedded (Lasso: selecciona al entrenar)."),
                      ("La trampa", "Todo lo que mira el target va dentro de cada fold. Si no, el AUC es inventado."),
                      ("Desconfía", "Un AUC demasiado bueno suele ser leakage, también por la ventana temporal."),
                  ], C33)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C33_FeatureEngineering.ipynb</font> o el cuaderno "
               "<font name='Mono'>C33_Ejercicios_Cuaderno_FeatureEngineering.pdf</font>.", "small"))

    s.append(h("1", "Qué es feature engineering y por qué importa"))
    s += para("Un modelo solo puede aprender de lo que le entregas. El <b>feature engineering</b> es el trabajo de convertir datos "
              "crudos en variables útiles: limpiarlas, transformarlas y crear nuevas. La <b>selección de variables</b> decide cuáles "
              "de esas candidatas entran al modelo final.")
    s += para("El profesor insiste en que no es una receta lineal («primero esto, después esto»), sino un trabajo iterativo que "
              "depende del problema de negocio. Su ejemplo de la clase: predecir qué clientes de una telco se van a fugar, con "
              "3.150 clientes y 12 variables.")

    s.append(h("2", "Limpiar con criterio de negocio"))
    s += para("Antes de cualquier técnica estadística, revisa mínimos y máximos con <font name='Mono'>describe()</font>. Un valor "
              "<b>imposible</b> (segundos negativos, 233 años) no es un outlier interesante: es un error de captura. Se marca como "
              "faltante (NaN) y luego se imputa.")
    s.append(ejemplo("¿media o mediana para imputar?", [
        P("Segundos de 5 clientes: 3.000, 3.200, 2.900, 3.100 y −14.645 (imposible).", "cell"),
        pasos([["Paso", "Qué haces", "Resultado"],
               ["1", "Media con el error incluido", "(12.200 − 14.645) / 5 = −489"],
               ["2", "Mediana con el error incluido", "3.000 (casi no se mueve)"],
               ["3", "Marcar −14.645 como NaN y calcular la mediana", "3.050"],
               ["4", "Imputar el NaN con esa mediana", "el cliente queda con 3.050 segundos"]]),
        P("Un solo dato malo vuelve negativa la media; la mediana es robusta. Por eso la clase imputa con la mediana.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>La regla la pone el negocio</b>", "cellb"),
                  P("Edad entre 15 y 100 es una regla de negocio. ¿Y los 108 clientes de 0 a 14 años? Podrían ser cuentas a nombre de un "
                    "menor. No hay respuesta estadística: hay que preguntar al negocio y documentar la decisión. Y no borres la fila: "
                    "perderías la información de sus otras 11 variables.", "cell")], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Para cada variable pregúntate: ¿qué rango es físicamente posible? Todo lo que esté fuera, a NaN. Después cuenta "
                      "cuántos marcaste por variable antes de imputar.", "cell")]))

    s.append(h("3", "Construir variables"))
    s += para("Las variables crudas muchas veces no capturan lo importante. Un cliente con 60 llamadas en 20 meses no es igual a uno con "
              "60 llamadas en 2 meses: lo que discrimina es el <b>ratio</b> llamadas por mes.")
    s.append(table([["Familia", "Ejemplo", "Para qué sirve"],
                    ["Ratios", "frec_llamadas / antigüedad", "normalizar por exposición; según el profesor, las más potentes"],
                    ["Logaritmos", "log(1 + clv)", "aplastar colas largas"],
                    ["Binarias", "tiene_reclamo, cliente_nuevo", "marcar una condición de negocio"],
                    ["Cuartiles", "clv_q1 … clv_q4", "capturar relaciones no lineales"],
                    ["Interacciones", "reclamo × antigüedad", "efectos que dependen de otra variable"],
                    ["One-hot", "estado_2, plan_2", "categóricas para modelos lineales"]],
                   [28 * mm, 52 * mm, CONTENT_W - 80 * mm]))
    s.append(Spacer(1, 4))
    s.append(ejemplo("ratios con épsilon", [
        pasos([["Cliente", "Cálculo", "Resultado"],
               ["20 meses, 60 llamadas", "60 / (20 + 0.000001)", "3.0 llamadas por mes"],
               ["60 llamadas, 4.500 segundos", "4.500 / (60 + 0.000001)", "75 segundos por llamada"],
               ["0 meses (cliente nuevo), 60 llamadas", "60 / (0 + 0.000001)", "60.000.000: enorme pero finito"]],
              [52 * mm, 50 * mm, CONTENT_W - 102 * mm - 18]),
        P("Sin el épsilon, la tercera fila daría infinito y rompería el escalado y el modelo. Fíjate también que el resultado es "
          "gigante: conviene revisar y tratar esos casos, por ejemplo con un log o un tope.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte B del notebook y el Bloque 2 del cuaderno",
                   [P("Escribe primero el ratio en palabras («llamadas por cada mes de antigüedad») y después en código. Al terminar, "
                      "verifica que no queden infinitos con <font name='Mono'>np.isinf(X).sum().sum()</font>.", "cell")]))

    s.append(h("4", "Irrelevante, redundante o útil en conjunto"))
    s.append(table([["Tipo", "Qué pasa", "Decisión", "Ejemplo de la clase"],
                    ["Irrelevante", "no aporta ni sola ni acompañada", "descartar", "ninguna clara en este dataset"],
                    ["Redundante", "repite otra variable", "conservar una", "reclamos y tiene_reclamo (r = 1)"],
                    ["Útil en conjunto", "sola no sirve, acompañada sí", "NO descartar", "numero_fallas, el caso XOR"]],
                   [28 * mm, 50 * mm, 26 * mm, CONTENT_W - 104 * mm]))
    s.append(Spacer(1, 4))
    s.append(ejemplo("el XOR con 4 puntos", [
        pasos([["Punto", "x1", "x2", "Clase (signos distintos)"],
               ["A", "+1", "+1", "0"], ["B", "+1", "−1", "1"], ["C", "−1", "+1", "1"], ["D", "−1", "−1", "0"]],
              [20 * mm, 20 * mm, 20 * mm, CONTENT_W - 60 * mm - 18]),
        P("Mira solo x1: con +1 hay un 0 y un 1; con −1 también. x1 sola no dice nada, y lo mismo pasa con x2. Pero x1·x2 vale +1 en "
          "las clases 0 y −1 en las clases 1: juntas separan perfecto. Un filtro univariado habría descartado las dos.", "cell")]))

    s.append(h("5", "Métodos de filtro"))
    s += para("Un <b>filtro</b> mide cada variable por separado contra el target, sin entrenar el modelo. Es rápido, pero no ve "
              "combinaciones.")
    s.append(bullets(["<b>Varianza:</b> una variable casi constante no informa. Para una binaria, Var = π(1 − π); con threshold = 0.01 se exige "
                      "que la categoría menor tenga más de ~1% de los casos.",
                      "<b>Correlación de Pearson:</b> mide solo relación lineal. Con y = x² (una U perfecta), r = 0.",
                      "<b>Información mutua:</b> detecta cualquier dependencia, también no lineal. Sirve para ordenar dentro de un dataset. "
                      "Cambia con la semilla: fija random_state.",
                      "<b>ANOVA F:</b> compara las medias de la variable entre fugados y no fugados. Falla con relaciones tipo U."], "body", C33))
    s.append(Spacer(1, 4))
    s.append(ejemplo("varianza de una variable binaria", [
        pasos([["% de unos (π)", "Var = π(1 − π)", "¿Pasa threshold = 0.01?"],
               ["0.5%", "0.005 · 0.995 = 0.005", "No: casi constante"],
               ["1%", "0.01 · 0.99 = 0.0099", "No, por muy poco"],
               ["10%", "0.1 · 0.9 = 0.09", "Sí"]],
              [34 * mm, 60 * mm, CONTENT_W - 94 * mm - 18])]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte C del notebook y los Bloques 3 y 4 del cuaderno",
                   [P("Ordena las variables por información mutua y después mira la matriz de correlación entre las primeras: si dos de "
                      "arriba tienen r &gt; 0.9, son la misma información.", "cell")]))

    s.append(h("6", "Wrapper y embedded"))
    s += para("Un <b>wrapper</b> entrena el modelo con distintos subconjuntos y compara el desempeño. RFE parte con todas y va quitando "
              "la de menor coeficiente. El problema es el costo: con p variables hay 2<super>p</super> − 1 subconjuntos posibles.")
    s.append(ejemplo("¿cuánto cuesta probar todo? (1.000 modelos por segundo)", [
        pasos([["p", "Subconjuntos (2<super>p</super> − 1)", "Tiempo"],
               ["10", "1.023", "1 segundo"],
               ["20", "1.048.575", "17 minutos"],
               ["30", "≈ 1.07 · 10<super>9</super>", "12 días"],
               ["50", "≈ 1.13 · 10<super>15</super>", "35.702 años"]],
              [20 * mm, 60 * mm, CONTENT_W - 80 * mm - 18])]))
    s.append(Spacer(1, 4))
    s += para("Un método <b>embedded</b> selecciona mientras entrena. El <b>Lasso</b> (penalización L1) empuja coeficientes a exactamente 0: "
              "las variables con coeficiente 0 quedan fuera. En sklearn, <b>C es el inverso de la penalización</b>: C = 0.01 dejó 13 "
              "variables y C = 1 dejó 45. Antes del Lasso hay que <b>escalar</b>: si no, el coeficiente depende de las unidades.")

    s.append(h("7", "La trampa: selección fuera de la validación cruzada"))
    s += para("Si eliges variables mirando el target con <b>todos</b> los datos y después validas, los casos de validación ya influyeron "
              "en la elección. Con suficientes variables, algunas parecen buenas por puro azar, y la validación no lo detecta porque "
              "esos mismos casos las hicieron parecer buenas.")
    s.append(ejemplo("¿cuántas variables «buenas» aparecen por azar?", [
        pasos([["Variables de puro ruido (p)", "α", "Falsos positivos esperados = α · p"],
               ["1.000", "0.05", "50"], ["200", "0.05", "10"], ["1.000", "0.01", "10"]],
              [56 * mm, 20 * mm, CONTENT_W - 76 * mm - 18]),
        P("En la clase, elegir 20 de 1.000 variables de ruido fuera del CV dio AUC 0.882. Dentro del pipeline, 0.514. La verdad: 0.5.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(table([["Paso", "¿Mira el target?", "¿Usa estadísticas del conjunto?", "¿Dónde va?"],
                    ["Borrar imposibles, construir ratios, one-hot", "No", "No", "Puede ir antes del split"],
                    ["Imputar con la mediana, escalar", "No", "Sí", "Dentro del fold"],
                    ["Filtro por MI, desempate de redundantes", "Sí", "Sí", "Dentro del fold"],
                    ["RFE, Lasso, SelectKBest", "Sí", "Sí", "Dentro del fold"]],
                   [62 * mm, 28 * mm, 42 * mm, CONTENT_W - 132 * mm]))
    s.append(Spacer(1, 4))
    s.append(pista("la Parte E del notebook y el Bloque 7 del cuaderno",
                   [P("Regla de bolsillo: si un paso necesita <font name='Mono'>fit</font>, va dentro del Pipeline. Si es una fórmula fija "
                      "fila por fila (un ratio, un log), puede ir antes.", "cell")]))

    s.append(h("8", "La ventana temporal («junio hacia atrás»)"))
    s += para("En un modelo <b>predictivo</b>, las variables deben medirse <b>antes</b> del período en que ocurre el target. Si quieres "
              "predecir la fuga de agosto, usa información hasta junio. Si construyes una variable con datos de julio o agosto, estás "
              "usando el futuro: es leakage, aunque no uses la columna fuga.")
    s.append(box([P("<b>Señal de alarma</b>", "cellb"),
                  P("Una variable con poder predictivo «demasiado bueno», como «llamó para dar de baja el servicio», casi siempre ocurre "
                    "después de la decisión que quieres predecir. En la industria, un AUC de 0.84–0.85 en fuga ya es muy bueno.", "cell")],
                 bg=AMBER_BG, border=AMBER))

    s.append(h("9", "Adelanto: WOE e Information Value"))
    s += para("El profesor pidió anotar esta técnica de banca y riesgo de crédito. No se calcula en esta clase. Se discretiza la variable en "
              "tramos y, para cada tramo, se compara la proporción de «buenos» con la de «malos».")
    s.append(ejemplo("una variable con 2 tramos (800 buenos y 200 malos en total)", [
        pasos([["Tramo", "% buenos", "% malos", "WOE = ln(%buenos / %malos)", "Aporte al IV"],
               ["A", "500/800 = 62.5%", "60/200 = 30%", "ln(0.625/0.30) = 0.734", "(0.625 − 0.30)·0.734 = 0.239"],
               ["B", "300/800 = 37.5%", "140/200 = 70%", "ln(0.375/0.70) = −0.624", "(0.375 − 0.70)·(−0.624) = 0.203"]],
              [14 * mm, 30 * mm, 28 * mm, 46 * mm, CONTENT_W - 118 * mm - 18]),
        P("IV = 0.239 + 0.203 = <b>0.44</b> → poder predictivo fuerte. Umbrales usuales: &lt; 0.02 nulo · 0.02–0.1 débil · 0.1–0.3 medio · "
          "0.3–0.5 fuerte · &gt; 0.5 sospechoso (posible leakage).", "cell")]))

    s.append(PageBreak())
    s.append(h("10", "Chuleta de código"))
    s.append(table([
        ["Quiero…", "Código"],
        ["Ver mínimos y máximos", "d.describe().T[['min', 'max']]"],
        ["Marcar imposibles", "d.loc[d.x < 0, 'x'] = np.nan"],
        ["Contar duplicados", "d.duplicated().sum()"],
        ["Ratio seguro", "d.a / (d.b + 1e-6)"],
        ["Cuartiles", "pd.qcut(d.x, 4, labels=False)"],
        ["Pares redundantes", "corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()"],
        ["Información mutua", "mutual_info_classif(X, y, random_state=0)"],
        ["Lasso", "LogisticRegression(penalty='l1', solver='liblinear', C=0.05)"],
        ["Pipeline honesto", "Pipeline([('imp', SimpleImputer(strategy='median')), ('esc', StandardScaler()), ('sel', SelectKBest(...)), ('log', ...)])"],
        ["Validar", "cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True), scoring='roc_auc')"],
    ], [42 * mm, CONTENT_W - 42 * mm]))
    s.append(Spacer(1, 8))
    s.append(h("11", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Borrar filas con un valor raro", "Pierdes las otras variables de esa fila", "NaN + imputar, o una categoría propia"],
        ["Imputar con la media", "Los extremos la arrastran", "Mediana"],
        ["Dividir sin épsilon", "Un infinito rompe el pipeline", "x / (y + 1e-6)"],
        ["Descartar por un p-valor alto", "Puede servir en conjunto", "Probar en el modelo con otras"],
        ["Top-k por un ranking univariado", "Gasta cupos en redundantes", "Revisar la correlación entre las elegidas"],
        ["Seleccionar antes del CV", "AUC inflado (0.88 con puro ruido)", "Selección dentro del Pipeline"],
        ["Variables del futuro", "Leakage temporal", "Ventana «junio hacia atrás»"],
        ["Usar exactitud con 15.7% de fuga", "Decir «nadie se va» da 84.3%", "AUC, sensibilidad, precisión"],
    ], [48 * mm, 52 * mm, CONTENT_W - 100 * mm], header_color=RED))
    s.append(Spacer(1, 8))
    s.append(recuerda(["Orden sugerido: lee estos apuntes → Bloques 1–3 del cuaderno → Partes A y B del notebook → Bloques 4–6 → "
                       "Partes C y D → Bloques 7–8 → Parte E → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/C33_Apuntes_FeatureEngineering.pdf", s, "Apuntes · Clase 33 · Feature engineering y selección de variables")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#EA580C">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 3)]
    flow.append(numbered([("", t) for t in items], C33))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 33 · Feature engineering",
                  "Para resolver en papel, con calculadora (tecla ln) · soluciones paso a paso al final", C33))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu razonamiento completo: aquí importan tanto las decisiones como "
                           "los números.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · "
                           "<font name='Sym'>★★★</font> integra varios conceptos.",
                           "Corrige con las soluciones y marca con color lo que tuviste mal: eso va al repaso del domingo."], "cell", C33)],
                 bg=SOFT, border=C33))
    s.append(Spacer(1, 4))
    s += bloque(1, "Limpiar con criterio", "★★", "Seis clientes: (a) edad 34, segundos 3.100 · (b) edad 233 · (c) segundos −250 · "
                "(d) antigüedad 0 meses · (e) edad 12 · (f) frec_llamadas 0.", [
        "Clasifica cada caso como imposible, dudoso (decisión de negocio) o válido. Justifica en una línea.",
        "¿Qué harías con cada uno: NaN + imputar, preguntar al negocio, o dejarlo?",
        "Segundos de 6 clientes: 2.800, 3.000, 3.100, 3.300, 3.500 y −9.000. Calcula la media y la mediana con el error, y la mediana "
        "después de marcarlo como NaN.",
        "<font name='Sym'>★★★</font> Un compañero borra las 111 filas con edad fuera de [15, 100]. Explica dos problemas de esa decisión.",
    ])
    s += bloque(2, "Construir variables", "★", "Cliente: antigüedad 25 meses, 50 llamadas, 4.000 segundos, 200 SMS, CLV 600.", [
        "Calcula llamadas por mes, segundos por llamada y SMS por llamada.",
        "Calcula log(1 + CLV). ¿Para qué sirve el logaritmo en una variable como CLV?",
        "Si la antigüedad fuera 0, ¿qué pasa con llamadas / antigüedad sin épsilon? ¿Y con ε = 10<super>−6</super>?",
        "La variable categórica estado tiene 2 niveles y tipo_plan tiene 2. ¿Cuántas columnas one-hot agregas con drop_first?",
    ])
    s += bloque(3, "Varianza y correlación a mano", "★★", None, [
        "Una binaria vale 1 en el 2% de los clientes. Calcula su varianza. ¿La descarta VarianceThreshold(0.01)?",
        "Calcula r de Pearson para x = 1, 2, 3, 4, 5 · y = 1, 3, 2, 5, 4.",
        "Calcula r para x = −2, −1, 0, 1, 2 · y = x². ¿Significa que no hay relación?",
        "Dos variables tienen r = 0.96 entre sí. Una tiene MI 0.10 con el target y la otra 0.08. ¿Con cuál te quedas y por qué?",
    ])
    s += bloque(4, "Irrelevante, redundante o útil en conjunto", "★★", None, [
        "Clasifica: (a) log_segundos cuando ya tienes segundos · (b) el color favorito del cliente · (c) numero_fallas, que sola da AUC "
        "0.53 pero sube el modelo de 0.907 a 0.910.",
        "Para los 4 puntos (x1, x2, clase): (+1, +1, 1), (+1, −1, 0), (−1, +1, 0), (−1, −1, 1), ¿x1 sola separa las clases? Calcula x1·x2 "
        "para cada punto. ¿Qué observas?",
        "<font name='Sym'>★★★</font> ¿Por qué un filtro univariado (ANOVA F) descartaría x1 y x2 en el ejercicio anterior? ¿Qué método no las descartaría?",
    ])
    s += bloque(5, "Filter, wrapper o embedded", "★", None, [
        "Clasifica: VarianceThreshold · RFE · Lasso · información mutua · forward selection · importancia de un árbol · ANOVA F.",
        "Tienes 5.000 variables candidatas y poco cómputo. Propón el orden en que aplicarías las tres familias.",
    ])
    s += bloque(6, "Costo y Lasso", "★★", None, [
        "¿Cuántos subconjuntos hay con 15 variables? ¿Cuánto tardarías a 1.000 modelos por segundo?",
        "En sklearn, C = 0.01 deja 13 variables y C = 1 deja 45. ¿Qué C usarías si quieres un modelo simple de explicar? ¿Qué arriesgas?",
        "¿Por qué hay que escalar las variables antes de un Lasso? Responde con un ejemplo de segundos (miles) vs reclamos (0/1).",
    ])
    s += bloque(7, "Leakage en la selección", "★★★", None, [
        "Tienes 400 variables de puro ruido y usas α = 0.05. ¿Cuántas esperas que parezcan significativas por azar?",
        "Clasifica cada paso como «antes del split» o «dentro del fold»: (a) quitar edades imposibles · (b) imputar con la mediana · "
        "(c) calcular llamadas por mes · (d) SelectKBest · (e) StandardScaler · (f) Lasso · (g) one-hot de estado.",
        "Un compañero reporta AUC 0.97 en fuga. Nombra dos preguntas que le harías antes de creerle.",
        "Quieres predecir la fuga de agosto. ¿Puedes usar «n° de reclamos en julio»? ¿Y «n° de reclamos en agosto»? Explica.",
    ])
    s += bloque(8, "WOE e IV", "★★★", "Variable con 2 tramos; total 1.000 buenos y 250 malos. Tramo A: 700 buenos y 50 malos. Tramo B: 300 buenos y 200 malos.", [
        "Calcula % de buenos y % de malos de cada tramo.",
        "Calcula el WOE de cada tramo.",
        "Calcula el IV e interprétalo con los umbrales de los apuntes.",
    ])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el razonamiento, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "(a) válido · (b) imposible · (c) imposible · (d) válido: cliente nuevo, pero ojo con los ratios · (e) dudoso: podría ser la "
              "cuenta de un menor · (f) válido: un cliente sin llamadas existe."],
        ["2", "(b) y (c): NaN + imputar con la mediana · (e): preguntar al negocio y documentar · (a), (d) y (f): dejar; (d) necesita "
              "épsilon en los ratios."],
        ["3", "Media con el error = (15.700 − 9.000) / 6 = 1.116.7 · mediana con el error = (3.000 + 3.100) / 2 = 3.050 · mediana sin el "
              "error (5 valores) = 3.100."],
        ["4", "Pierde la información de las otras 11 variables de esas filas, y los 108 casos de 0–14 años podrían ser clientes reales "
              "(cuentas de menores). Además puede sesgar la muestra si esos clientes se fugan distinto."]])
    s += sol("Bloque 2", [
        ["1", "50 / 25 = 2 llamadas por mes · 4.000 / 50 = 80 segundos por llamada · 200 / 50 = 4 SMS por llamada."],
        ["2", "ln(601) = 6.40. El log aplasta la cola larga: los clientes de CLV muy alto no dominan el modelo."],
        ["3", "Sin épsilon: división por cero → infinito, y el pipeline se rompe. Con épsilon: 50 / 10<super>−6</super> = 50.000.000, finito "
              "pero enorme: conviene revisarlo o aplicarle un log."],
        ["4", "2 − 1 = 1 por estado y 2 − 1 = 1 por plan: 2 columnas (estado_2, plan_2)."]])
    s += sol("Bloque 3", [
        ["1", "0.02 · 0.98 = 0.0196 &gt; 0.01 → no la descarta."],
        ["2", "x̄ = 3, ȳ = 3. Desvíos x: −2, −1, 0, 1, 2 · desvíos y: −2, 0, −1, 2, 1. S<sub>xy</sub> = 4 + 0 + 0 + 2 + 2 = 8 · S<sub>xx</sub> = 10 · "
              "S<sub>yy</sub> = 10 → r = 8 / 10 = 0.8."],
        ["3", "r = 0, pero la relación es perfecta (y = x²). Pearson solo mide relaciones lineales: grafica siempre."],
        ["4", "Son redundantes: me quedo con la de MI 0.10, salvo que la otra sea mucho más simple de mantener en producción."]])
    s.append(PageBreak())
    s += sol("Bloque 4", [
        ["1", "(a) redundante → conservar una · (b) irrelevante → descartar · (c) útil en conjunto → no descartarla."],
        ["2", "x1 = +1 tiene una clase 1 y una 0; con −1, igual: x1 sola no separa. x1·x2 = +1, −1, −1, +1 coincide con la clase (+1 → 1, "
              "−1 → 0): juntas separan perfecto."],
        ["3", "ANOVA F compara la media de cada variable entre clases, y esas medias son iguales: F ≈ 0. Un modelo con la interacción "
              "x1·x2, un Random Forest o un árbol profundo sí las aprovechan."]])
    s += sol("Bloque 5", [
        ["1", "Filter: VarianceThreshold, información mutua, ANOVA F · Wrapper: RFE, forward selection · Embedded: Lasso, importancia de un árbol."],
        ["2", "Filtro (varianza + MI) para bajar a unas 200 → embedded (Lasso) para bajar a unas 30 → wrapper (RFE) solo si sobra cómputo."]])
    s += sol("Bloque 6", [
        ["1", "2<super>15</super> − 1 = 32.767 subconjuntos → 33 segundos a 1.000 modelos por segundo."],
        ["2", "Un C chico, por ejemplo 0.01 (13 variables). Arriesgas perder variables útiles y algo de AUC; además, entre variables "
              "redundantes el Lasso elige una casi al azar."],
        ["3", "La penalización castiga el tamaño del coeficiente. Segundos en miles tendría un coeficiente diminuto aunque fuera importante, "
              "y reclamos (0/1) uno grande: el Lasso decidiría por las unidades, no por la información."]])
    s += sol("Bloque 7", [
        ["1", "α · p = 0.05 · 400 = 20 variables falsamente significativas."],
        ["2", "Antes del split: (a), (c), (g) · Dentro del fold: (b), (d), (e), (f)."],
        ["3", "¿La selección o la imputación se hizo dentro del CV? ¿Alguna variable usa información posterior al momento de predecir "
              "(ventana temporal)? También: ¿está evaluado en un test que nunca se tocó?"],
        ["4", "Julio sí: es anterior al período del target. Agosto no: es el mismo período que quieres predecir (leakage temporal)."]])
    s += sol("Bloque 8", [
        ["1", "A: 700/1.000 = 70% buenos · 50/250 = 20% malos · B: 300/1.000 = 30% buenos · 200/250 = 80% malos."],
        ["2", "WOE<sub>A</sub> = ln(0.70 / 0.20) = 1.253 · WOE<sub>B</sub> = ln(0.30 / 0.80) = −0.981."],
        ["3", "IV = (0.70 − 0.20)·1.253 + (0.30 − 0.80)·(−0.981) = 0.626 + 0.490 = 1.12 &gt; 0.5 → sospechoso: tanto poder predictivo "
              "suele indicar leakage. Hay que revisar cómo se construyó la variable."]])
    build("apuntes/C33_Ejercicios_Cuaderno_FeatureEngineering.pdf", s, "Ejercicios para el cuaderno · Clase 33")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
