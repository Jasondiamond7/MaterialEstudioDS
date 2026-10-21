"""Clase 32 · Reducción de dimensionalidad y representación de variables: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C32 = colors.HexColor("#DB2777")
C32_BG = colors.HexColor("#FDF2F8")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + apunte + notebook y caso de la profesora Laura Sánchez"
M = lambda t: f"<font name='Mono'>{t}</font>"


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado):
    """figuras: lista de (archivo, ancho_relativo, pie)."""
    s.append(band(f"Clase 32 · {titulo}", FUENTE, C32))
    s.append(Spacer(1, 5))
    s.append(box([P(idea, "big")], bg=SOFT, border=C32))
    s.append(Spacer(1, 4))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C32))
    s.append(Spacer(1, 2))
    for f, w, pie in figuras:
        s.append(img(f, width=CONTENT_W * w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 3))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF)]
    if py:
        left += [Spacer(1, 4), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C32), Spacer(1, 4),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER, width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 32 · Reducción de dimensionalidad y representación de variables",
                  "Profesora Laura Sánchez · bloque de Machine Learning · sigue a Preparación para ML (C30) e Introducción al ML (C31)", NAVY))
    s.append(Spacer(1, 7))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Un modelo no recibe «datos»: recibe una <b>representación</b>. Antes de modelar hay que <b>escalar</b>, <b>codificar</b> y <b>construir</b> variables, y cuando hay "
                    "demasiadas columnas (redundantes o ruidosas) conviene <b>resumirlas</b>. <b>PCA</b> las resume en pocas combinaciones lineales que conservan la mayor varianza; "
                    "<b>t-SNE</b> dibuja en 2D quién es vecino de quién para <b>explorar</b>, no para producir variables. «Las matemáticas buscan el camino más eficiente: "
                    "reducir dimensiones es eficiente, pero no es reducir por reducir.»", "body")], bg=C32_BG, border=C32))
    s.append(Spacer(1, 5))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c32_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Correlación, varianza y desviación estándar (Etapa 0); regresión múltiple y multicolinealidad (Clases 17–19)",
                     "Train / validación / prueba, fuga de datos y Pipeline (Clase 30)",
                     "Supervisado vs no supervisado: PCA y t-SNE no usan la etiqueta (Clase 31)",
                     "Vectores y matrices: basta la idea de «dirección» y «proporcional» (la profesora evita las fórmulas)"], "cell", C32)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Diagnóstico</b> → <b>representación</b> (escalar, codificar, construir) → <b>PCA</b> (varianza explicada, cargas) → <b>t-SNE</b> (perplexity) → decidir",
                      "Datos: <b>1.500 clientes de telco simulados</b> (el notebook de la profesora los genera con <font name='Mono'>default_rng(32)</font>, 15 % de fuga) y la "
                      "<b>base real de 3.150 clientes</b> de la actividad de repaso (15.7 % de fuga)",
                      "Pregunta guía: ¿cuántas dimensiones necesito de verdad y cómo las entrego al modelo sin hacer trampa?"], "cell", C32)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo mínimo de scikit-learn para hoy</b>", "cellb"),
                  P("Los tres verbos de la Clase 30 (<font name='Mono'>fit</font>, <font name='Mono'>transform</font>, <font name='Mono'>predict</font>) valen igual para "
                    "<font name='Mono'>StandardScaler</font> y <font name='Mono'>PCA</font>: ambos <b>aprenden</b> con <font name='Mono'>fit(X_train)</font>. "
                    "PCA tiene además <font name='Mono'>explained_variance_ratio_</font> (varianza explicada) y <font name='Mono'>components_</font> (las cargas). "
                    "Todo va dentro de un <font name='Mono'>Pipeline</font> (página 7). Sobre la API: Clase 30, página 7.", "cell")], bg=SOFT, border=C32))
    s.append(PageBreak())

    topic_page(
        s, "El problema de la dimensionalidad",
        "Más columnas no son más información. Las variables sobran por cuatro razones: <b>costo</b>, <b>multicolinealidad</b>, <b>ruido</b> y <b>sesgo vs. varianza</b>. "
        "Se reduce por <b>selección</b> (quitar columnas) o por <b>extracción</b> (combinarlas).",
        [("Costo computacional", "Más columnas, más memoria y más tiempo de entrenamiento; más difícil de poner en producción, depurar y explicar."),
         ("Multicolinealidad", "Variables que cuentan la misma historia. En el notebook, <font name='Mono'>segundos</font> y <font name='Mono'>frec_llamadas</font> tienen r = 0.99; el VIF de ambas supera 40 (regla práctica: > 10 es problema)."),
         ("Variables ruido", "Columnas sin señal. Con 100 filas, agregar 400 columnas de ruido baja el AUC en prueba de 0.72 a 0.57 mientras en entrenamiento llega a 1.00: sobreajuste."),
         ("Sesgo vs. varianza", "Con pocas observaciones, más variables aumentan la varianza de las predicciones sin compensar el sesgo; además, en dimensiones altas las distancias se «aplanan».")],
        [("c32_dimensionalidad.png", 0.9,
          "Izquierda: correlaciones de las 12 variables originales (se rotulan |r| ≥ 0.6): seis variables de uso casi duplicadas. Centro: con puntos uniformes, la razón (más lejano − más cercano)/más cercano cae de 41 (d = 2) a 0.1 (d = 1.000). "
          "Derecha: AUC de una regresión logística con 100 filas al agregar columnas de ruido (25 repeticiones).")],
        ["VIFⱼ = 1 / (1 − R²ⱼ)",
         "filas por columna = n / p",
         "selección: quita columnas · extracción: las combina"],
        ["df.corr()  ·  df.shape",
         "# VIF: 1/(1-R²) con LinearRegression",
         "X = df.drop(columns='fuga')"],
        ["1.500 clientes, 12 variables, fuga 15.07 %; las 6 de uso correlacionan entre 0.66 y 0.99",
         "VIF: segundos 40.8, frec_llamadas 48.3, n_contactos 12.2",
         "«No hay un número ideal» de variables (profesora)"],
        ["Correlacionadas no es lo mismo que ruido: el ruido no aporta nada aunque no se parezca a ninguna."])

    topic_page(
        s, "Representación de variables: escalar, codificar y construir",
        "La Clase 30 preguntó «¿son confiables los datos?». Hoy: «¿en qué <b>forma</b> se los entrego al modelo?». Un dato correcto pero mal representado hace que PCA o t-SNE resuman <b>una distorsión</b>.",
        [("Escalas distintas", "<font name='Mono'>segundos</font> llega a miles y <font name='Mono'>reclamos</font> es 0 o 1. Sin escalar, la variable de mayor magnitud domina varianzas y distancias."),
         ("Estandarizar vs. normalizar", "<b>Estandarización</b> (z-score): cuántas desviaciones se aleja de la media; centro 0 y desviación 1; es lo habitual para PCA. <b>Min-max</b>: comprime a [0, 1]. No hace falta que la variable sea normal."),
         ("Codificar categóricas", "<b>Ordinal</b>: enteros cuando hay orden real (monto 0 a 9). <b>One-hot</b>: una columna 0/1 por categoría, sin imponer orden. <b>Target / frequency</b>: la categoría se reemplaza por una tasa o una frecuencia; sirve con muchas categorías."),
         ("Construir variables", "La señal puede estar en un cociente: <font name='Mono'>segundos / frec_llamadas</font>, <font name='Mono'>frec_sms / antiguedad</font>, <font name='Mono'>n_contactos / frec_llamadas</font>. Se crean <b>antes</b> de reducir.")],
        [("c32_representacion.png", 0.9,
          "Izquierda: desviación estándar de seis variables (escala log): 2.969 para segundos y 0.27 para reclamos. Centro: con códigos 0, 1, 2, 3, «Norte» queda a 3 de «Oeste» y a 1 de «Sur» sin motivo; con one-hot todas distan √2 = 1.41. "
          "Derecha: 60 comunas inventadas, sin relación con la fuga: la media del target y el one-hot «logran» 0.70 de AUC en entrenamiento y 0.51 en prueba.")],
        ["z = (x − media) / desviación estándar",
         "min-max = (x − mín) / (máx − mín)",
         "one-hot: 1 columna por categoría (drop evita una)"],
        ["StandardScaler().fit(X_train)",
         "OneHotEncoder(drop='if_binary')",
         "TargetEncoder(cv=5)  # tasa por categoría"],
        ["Variables construidas: 12 → 15 columnas; one-hot de tipo_plan y estado (drop if_binary): sigue en 15",
         "Target encoding «a mano», 60 categorías sin señal: AUC 0.70 en entrenamiento, 0.51 en prueba; <font name='Mono'>TargetEncoder</font>: 0.53 y 0.51"],
        ["La diapositiva dice «Ordinal: Nominal → entero»; es para variables <b>ordinales</b>. El target encoding con todas las filas es fuga."])

    topic_page(
        s, "PCA: la idea, la varianza explicada y cuántos componentes",
        "PCA busca la <b>dirección en que los datos varían más</b> (PC1), luego la siguiente dirección <b>perpendicular</b> (PC2), etc. Cada componente es una <b>receta</b> (combinación lineal) y "
        "no usa la etiqueta. Se eligen los primeros k que acumulan suficiente varianza.",
        [("Direcciones ortogonales", "Los componentes son los <b>vectores propios</b> de la matriz de covarianza de los datos estandarizados; el <b>valor propio</b> (λ) es la varianza que captura cada uno. Se ordenan de mayor a menor."),
         ("Varianza explicada", "<font name='Mono'>explained_variance_ratio_</font> = λ / Σλ. Con p variables estandarizadas, Σλ = p. El gráfico de barras es el scree plot y la línea, la acumulada."),
         ("¿Cuántos componentes?", "No hay un número mágico: se busca el codo, un umbral (80 o 90 %) o, mejor, el desempeño del modelo con validación cruzada dentro de un Pipeline."),
         ("No supervisado", "Varianza no es relevancia: un componente chico puede ser el que separa la fuga.")],
        [("c32_geometria.png", 0.72,
          "Con <font name='Mono'>segundos</font> y <font name='Mono'>clv</font> estandarizadas (r = 0.84): PC1 tiene λ = 1.84 y PC2 λ = 0.16. Proyectar sobre PC1 conserva el 91.8 % de la varianza. Derecha: tres variables que casi caben en un plano (PC1 + PC2 = 95 %)."),
         ("c32_scree.png", 0.72,
          "Scree plot de los datos de la clase (8 componentes = 85.0 %), de la base real (7 = 80.4 %) y AUC con validación cruzada de un Pipeline (escalar → PCA → logística) según k.")],
        ["Σ v = λ v  ·  λ₁ ≥ λ₂ ≥ … ≥ λₚ",
         "varianza explicada de j = λⱼ / Σ λ",
         "puntaje = Σ cargaⱼ · zⱼ"],
        None,
        ["Con 7 componentes 78.7 %; con 8, 85.0 %; con 9, 90.5 %. Base real: 7 → 80.4 %",
         "Logística en prueba: 15 variables F1 0.270 / MCC 0.265; PCA 8: 0.267 / 0.252. Base real: 0.574 / 0.546 vs PCA 7: 0.548 / 0.534"],
        ["AUC en 5-Fold: 3 componentes 0.773 > 15 variables 0.746; F1 0.270 vs 0.267 es ruido (57 fugas en la prueba)."])

    topic_page(
        s, "Leer los componentes: cargas y escala",
        "Una carga dice <b>cuánto pesa cada variable</b> en un componente. Se lee con cuidado: el signo de un componente es arbitrario, y sin <b>estandarizar</b> el primer componente solo refleja la unidad de la variable más grande.",
        [("Cargas", "Cada fila de <font name='Mono'>components_</font> tiene norma 1. Las cargas grandes (en valor absoluto) definen de qué trata el componente."),
         ("Círculo de correlaciones", "Con datos estandarizados, carga × √λ es la <b>correlación</b> entre la variable y el componente. Flechas largas y juntas: variables redundantes."),
         ("Interpretar con cautela", "PC1 = «uso del servicio». PC2 es antigüedad y SMS por antigüedad; fallas y reclamos están en PC3. El nombre lo pone quien analiza."),
         ("Sin estandarizar", "PCA maximiza varianza: la variable en miles se lleva casi todo; el resultado refleja las unidades.")],
        [("c32_loadings.png", 0.68,
          "Cargas de PC1 a PC4, círculo de correlaciones y clientes en PC1–PC2 (el color de la fuga no se usó para construirlos: PC1–PC2 no la separan)."),
         ("c32_escala.png", 0.6,
          "Sin estandarizar: «segundos» concentra el 99.2 % de la varianza, PC1 explica 99.71 % y coincide con «segundos» (|r| ≈ 1.000).")],
        ["correlación(variable, PCⱼ) = cargaⱼ · √λⱼ",
         "norma de un componente = 1; son ⟂ entre sí",
         "k componentes ⇒ n × k puntajes"],
        None,
        ["PC1 (34.9 %): segundos, llamadas, contactos, CLV, SMS (0.38 a 0.42) · PC2 (8.9 %): antiguedad 0.71, sms_por_antiguedad −0.60",
         "PC3 (7.8 %): numero_fallas 0.72, reclamos 0.50; correlación con la fuga 0.33 (PC1: −0.06)",
         "Sin estandarizar, el 90 % de la varianza se alcanza con 1 componente"],
        ["La diapositiva pone «fallas y reclamos» en PC2; en el notebook son PC3. Comprueba con <font name='Mono'>components_</font>."])

    topic_page(
        s, "t-SNE: un mapa no lineal para explorar",
        "t-SNE dibuja los clientes en 2D de modo que los <b>vecinos</b> del espacio original queden cerca. Sirve para <b>mirar</b> estructura; no es una transformación para producción ni un algoritmo de clústeres.",
        [("Qué conserva", "PCA conserva la varianza global (lineal y determinista). t-SNE conserva vecindades locales (no lineal); las distancias entre grupos y el tamaño de las islas no se interpretan."),
         ("Perplexity", "Aproximadamente «a cuántos vecinos le presto atención»: baja → grupos chicos y locales; alta → vista más global. Se prueba entre 5 y 50 y debe ser menor que n."),
         ("No determinista, sin transform", "Otra semilla, otro dibujo; <font name='Mono'>TSNE</font> solo tiene <font name='Mono'>fit_transform</font> (no proyecta clientes nuevos) y es costoso. Modelar o producir: PCA. Explorar: t-SNE.")],
        [("c32_tsne.png", 0.88,
          "Fila 1: PCA en 2D y t-SNE con perplexity 5, 30 y 80 sobre el entrenamiento (1.125 clientes). Fila 2: tres semillas distintas con perplexity 30 y la «trustworthiness» (qué tanto se conservan los 10 vecinos más cercanos).")],
        ["trustworthiness ∈ [0, 1]: 1 = vecinos conservados",
         "perplexity < n_observaciones",
         "t-SNE no es K-means: no hay centroides"],
        ["TSNE(perplexity=30, init='pca',",
         "     random_state=32).fit_transform(X)",
         "trustworthiness(X, emb, n_neighbors=10)"],
        ["Trustworthiness (10 vecinos): PCA 2D 0.799; t-SNE 0.977 (perplexity 5), 0.982 (30), 0.979 (80); 2–4 s por ajuste",
         "Perplexity 5 parte los grupos en piezas; con otra semilla cambia la posición de las islas",
         "La fuga no forma un grupo propio: t-SNE no usa la etiqueta"],
        ["Ver «islas» no prueba que existan clústeres; tamaño y distancia entre islas no se interpretan."])

    # ------------------------------------------------------------ Pipeline y fuga
    s.append(band("Pipeline, fuga de datos y cómo decidir", "Todo lo que aprende de los datos se ajusta solo con entrenamiento", NAVY))
    s.append(Spacer(1, 6))
    s.append(img("c32_leakage.png", width=CONTENT_W * 0.97))
    s.append(P("Datos de puro ruido (100 filas, 1.000 columnas, etiquetas al azar; 40 repeticiones): seleccionar las 10 «mejores» con todas las filas y luego validar da ≈ 0.79 de exactitud; en un Pipeline, ≈ 0.51. "
               "PCA ajustado fuera de la validación da ≈ 0.51 (igual que dentro, 0.52): no ve la etiqueta, así que no la filtra, pero igual contamina (la prueba influye en los componentes) y el hábito es peligroso.", "small"))
    s.append(Spacer(1, 4))
    s.append(table([["Paso", "Qué aprende de los datos", "Dónde va"],
                    ["Imputador (mediana, moda)", "la mediana / la moda", "Pipeline, fit con entrenamiento"],
                    ["StandardScaler", "media y desviación por columna", "Pipeline, fit con entrenamiento"],
                    ["OneHotEncoder / OrdinalEncoder", "la lista de categorías", "ColumnTransformer dentro del Pipeline"],
                    ["TargetEncoder", "la tasa del target por categoría (mira «y»)", "Pipeline; usa validación cruzada interna"],
                    ["PCA", "los componentes (vectores propios)", "Pipeline, después del escalador"],
                    ["t-SNE", "no tiene transform", "solo para explorar, fuera del modelo"]],
                   [50 * mm, 70 * mm, CONTENT_W - 120 * mm]))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>El Pipeline de la clase, honesto</b>", "cellb"),
                  P("make_pipeline(StandardScaler(), PCA(n_components=8), LogisticRegression()).fit(X_train, y_train)", "mono"),
                  P("Con <font name='Mono'>GridSearchCV</font> o <font name='Mono'>cross_val_score</font>, el escalador y el PCA se reajustan en cada pliegue: así se elige k sin mirar la prueba.", "cell")],
                 bg=C32_BG, border=C32))
    s.append(Spacer(1, 4))
    checks = [("Construyo variables antes de reducir", "los cocientes no se obtienen desde los componentes"),
              ("Divido antes de escalar", "el scaler y el PCA se ajustan solo con entrenamiento"),
              ("Estandarizo antes de PCA", "si no, PC1 es la variable de mayor unidad"),
              ("Elijo k con criterio y con validación", "umbral de varianza, codo y AUC/F1 en CV"),
              ("Interpreto los componentes con sus cargas", "no con el nombre de la diapositiva"),
              ("Uso t-SNE solo para explorar", "no como entrada de un modelo en producción")]
    s.append(table([["", "Checklist del flujo", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 66 * mm, CONTENT_W - 74 * mm]))
    s.append(PageBreak())

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene corregir (verificado con código)", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Diapositivas, apunte y notebook</b>", "cellb"),
                  bullets(["<b>«7 componentes (≈ 80 %)»</b> (diapositiva 21) pero la tabla dice 8. Con los datos del notebook: 7 componentes = 78.7 % (no llega a 80), 8 = 85.0 %. En la base real, 7 = 80.4 %.",
                           "<b>PC2 = fallas y reclamos</b> (diapositiva 19 y apunte): en el notebook PC2 es antigüedad (0.71) y sms_por_antiguedad (−0.60); fallas (0.72) y reclamos (0.50) forman PC3. En la base real PC2 es SMS, CLV y sms_por_antiguedad.",
                           "<b>«Dataset real de 1.500 clientes»</b> (diapositiva 16): los 1.500 son sintéticos; la base real tiene 3.150. <b>USE_REAL_DATA = True</b> falla: <font name='Mono'>RUTA_CHURN</font> no está definida y se lee con <font name='Mono'>read_csv(sep=';')</font> un archivo .xlsx.",
                           "<b>Ejemplo de valores propios</b>: usa la matriz no simétrica [[10, −18], [6, −11]] (autovalores 1 y −2, vectores con producto 0.99, no ortogonales). PCA diagonaliza una matriz de <b>covarianzas, simétrica</b>: sus vectores son ortogonales (<font name='Mono'>np.linalg.eigh</font>).",
                           "<b>Cuánto cuesta PCA</b>: F1 0.270 vs 0.267 con 57 fugas en la prueba no es una diferencia; con AUC en 5-Fold, 3 componentes (0.773 ± 0.019) superan a las 15 variables (0.746 ± 0.017) y 8 empatan (0.745). «Se cede algo de desempeño» no es una regla.",
                           "<b>Ordinal encoding</b>: la diapositiva dice «Nominal → entero» y en clase «es para las nominales»; es para variables <b>ordinales</b>. <b>Target / frequency encoding</b> son dos técnicas, y la primera exige validación interna (0.70 → 0.51 en el experimento).",
                           "<b>Limpieza de la base real</b>: el código de la clase solo corrige edad y faltantes; quedan negativos imposibles (segundos hasta −14.645, frec_llamadas −176…) y edades 0–14; además monto llega a 10 (7 filas) y la diapositiva dice 0 a 9."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets(["«<b>PCA es un clustering</b>»: no; PCA es una proyección lineal sin grupos. «<b>El 85 % de fuga</b>… con 8 componentes cubrimos a los que se fugan»: el 85 % de varianza y el 85 % de clientes que NO se fugan son cantidades sin relación (la fuga es 15 %).",
                           "<b>Varianza</b> como «promedio de distancias a la media»: son distancias al <b>cuadrado</b>. Y desbalance de clases no es el «sesgo» del par sesgo-varianza.",
                           "«Con train/test sale la misma descomposición que con tres conjuntos»: no; PCA depende de las filas con que se ajusta.",
                           "<b>Caso de repaso, pregunta 3.2</b>: dice que fallas y reclamos apuntan opuestos al uso. En la base real, fallas se correlaciona <b>positivamente</b> con segundos (+0.49), llamadas (+0.57) y monto (+0.59; el propio caso lo reporta), carga +0.61 en PC1 vs 0.84 de segundos; solo reclamos es ligeramente negativo (−0.12). La interpretación «más fallas, menos uso» no se sostiene.",
                           "<b>Hinton &amp; van der Maaten</b> (2008): el orden de autores es van der Maaten y Hinton."],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper confundió t-SNE (TSN, TSM, TESEC, DCNE), PCA (pcea, pcl, PFA), Lasso (laso, lazo), Ridge (reach), Feature Engineering (Future), K-means (camins), "
                    "Matthews (Mathieu's), Ñuñoa (yu-nyua) y eigenvalues (agent values). Hubo bucles de repetición (ya omitidos) y el recreo (≈ 1:03:29) no está en el video. "
                    "Frases dudosas: «la paridad de clientes» (¿fuga?), «para qué voy a hacer todos el astre», «la que anita en la RAE».", "cell")],
                 bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"),
                  bullets(["Calculo a mano la varianza explicada desde los valores propios y elijo k para un umbral",
                           "Diagonalizo una matriz de covarianza 2×2 y proyecto un punto",
                           "Leo cargas y escribo qué significa un componente sin inventar",
                           "Elijo ordinal, one-hot o target encoding según el tipo de variable",
                           "Escribo un Pipeline con ColumnTransformer, StandardScaler y PCA sin fuga",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C32_Guia_ReduccionDimensionalidad.pdf", s, "Clase 32 · Reducción de dimensionalidad y representación de variables")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#DB2777">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 32 · Reducción de dimensionalidad y representación de variables",
                  "Para leer antes de los ejercicios · lectura de ~35 min", C32))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([
                      ("Más columnas no es más información", "Costo, multicolinealidad, ruido y varianza: se reduce por selección o por extracción."),
                      ("Representar antes de reducir", "Estandarizar, codificar y construir variables; lo aprendido (media, categorías, tasas) solo con entrenamiento."),
                      ("PCA = direcciones de máxima varianza", "Vectores propios de la covarianza; el valor propio λ es la varianza que explica cada uno."),
                      ("Elegir k con criterio", "Umbral de varianza, codo y validación cruzada dentro de un Pipeline."),
                      ("Cargas y correlaciones", "Carga × √λ = correlación; los componentes se nombran leyendo sus cargas."),
                      ("t-SNE explora, PCA produce", "t-SNE conserva vecindades locales, no es determinista y no transforma datos nuevos."),
                  ], C32)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C32_ReduccionDimensionalidad.ipynb</font> o el cuaderno "
               "<font name='Mono'>C32_Ejercicios_Cuaderno_ReduccionDimensionalidad.pdf</font>.", "small"))

    s.append(h("1", "El problema de la dimensionalidad"))
    s += para("Más variables aumentan el costo, introducen <b>redundancia</b> (multicolinealidad) y <b>ruido</b>, y con pocas filas disparan la varianza del modelo. "
              "La multicolinealidad se mide con el <b>VIF</b> = 1/(1 − R²), donde R² sale de explicar esa variable con todas las demás: VIF &gt; 10 es una señal fuerte. "
              "Reducir puede ser <b>seleccionar</b> (quitar columnas) o <b>extraer</b> (combinarlas).")
    s.append(ejemplo("un modelo de 40 columnas con 200 filas", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Filas por columna", "200 / 40", "5 (muy pocas)"],
               ["VIF si R² = 0.96 al explicar una variable con las demás", "1/(1 − 0.96) = 1/0.04", "25"],
               ["VIF si R² = 0.50", "1/0.50", "2 (sin problema)"],
               ["Tras quitar 25 columnas redundantes: filas por columna", "200 / 15", "13.3"]],
              [64 * mm, 66 * mm, CONTENT_W - 130 * mm - 18]),
        P("Un VIF de 25 dice que esa variable casi se puede calcular con las otras: sobra. En el notebook de la clase, segundos y frec_llamadas tienen VIF 40.8 y 48.3. "
          "Selección = eliminar una de las dos; extracción = resumirlas en un componente.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_dimensionalidad.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Reproduce la matriz de correlaciones, calcula el VIF de las seis variables de uso y repite el experimento de columnas de ruido con 100 filas.", "cell")]))

    s.append(h("2", "Escalar, codificar y construir"))
    s += para("La <b>estandarización</b> resta la media y divide por la desviación: z = (x − media)/desviación. Después de ella, todas las variables tienen media 0 y desviación 1. "
              "La <b>normalización min-max</b> lleva a [0, 1]. Para variables <b>categóricas</b>: ordinal si hay orden, one-hot si no; target o frequency encoding con muchas categorías. "
              "Las variables construidas (cocientes) se crean antes de reducir.")
    s.append(ejemplo("llamadas de un cliente", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["segundos: media 4.000, desviación 3.000 · z de 10.000", "(10.000 − 4.000)/3.000", "z = 2"],
               ["reclamos: media 0.1, desviación 0.3 · z de un 1", "(1 − 0.1)/0.3", "z = 3"],
               ["min-max de 10.000 con mínimo 50 y máximo 17.000", "(10.000 − 50)/(17.000 − 50)", "0.587"],
               ["Cuánto más varianza tiene segundos que reclamos (sin escalar)", "3.000² / 0.3² = 9.000.000/0.09", "100.000.000 veces"],
               ["Región con 5 categorías, canal con 4 y estado binario: columnas one-hot con drop = 'first'", "4 + 3 + 1", "8 (sin drop: 5 + 4 + 2 = 11)"],
               ["duracion_prom_llamada de 7.200 segundos en 90 llamadas", "7.200 / 90", "80 s"]],
              [66 * mm, 60 * mm, CONTENT_W - 126 * mm - 18]),
        P("Sin escalar, un cliente que se desvía 2 sd en segundos pesa igual que otro 3 sd en reclamos solo si estandarizas. Con target encoding: comuna A con 40 clientes y 8 fugas, "
          "tasa global 0.15, suavizado m = 20: (8 + 20·0.15)/(40 + 20) = 11/60 = 0.183, no el 0.20 crudo.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_representacion.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook y los Bloques 2 y 5 del cuaderno",
                   [P("Estandariza con <font name='Mono'>fit</font> solo en entrenamiento, compara PCA con y sin estandarizar y arma un <font name='Mono'>ColumnTransformer</font> con ordinal, one-hot y target encoding.", "cell")]))

    s.append(h("3", "PCA a mano con una matriz 2×2"))
    s += para("Con datos estandarizados, la matriz de covarianza es la matriz de correlaciones. Sus <b>valores propios</b> λ son las varianzas de los componentes y sus "
              "<b>vectores propios</b>, las direcciones. Para [[1, r], [r, 1]] los valores propios son <b>1 + r</b> y <b>1 − r</b>, con vectores (1, 1)/√2 y (1, −1)/√2. "
              "Un punto z se proyecta con el producto punto: puntaje = v · z.")
    s.append(ejemplo("dos variables con r = 0.6", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Valores propios", "1 + 0.6 · 1 − 0.6", "λ₁ = 1.6 · λ₂ = 0.4"],
               ["Varianza explicada", "1.6/2 · 0.4/2", "80 % · 20 %"],
               ["Vectores propios", "(1, 1)/√2 · (1, −1)/√2", "(0.707, 0.707) · (0.707, −0.707)"],
               ["Punto z = (1.0, 0.2): puntaje en PC1", "(1.0 + 0.2)/√2", "0.849"],
               ["Puntaje en PC2", "(1.0 − 0.2)/√2", "0.566"],
               ["Reconstruir con solo PC1 y error", "0.849·(0.707, 0.707) = (0.6, 0.6) · (0.4² + 0.4²)", "(0.6, 0.6) · 0.32"]],
              [58 * mm, 66 * mm, CONTENT_W - 124 * mm - 18]),
        P("PC1 es la diagonal («ambas altas o ambas bajas»); PC2, lo que las diferencia. Conservar solo PC1 mantiene el 80 % de la varianza. Los dos vectores son perpendiculares: (0.707·0.707 + 0.707·(−0.707)) = 0.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_geometria.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook (C1) y el Bloque 3 del cuaderno",
                   [P("Calcula la matriz de covarianza del entrenamiento escalado, diagonalízala con <font name='Mono'>np.linalg.eigh</font> y compara con <font name='Mono'>PCA</font>. Ojo: los signos pueden ser opuestos.", "cell")]))

    s.append(h("4", "Varianza explicada y cuántos componentes"))
    s += para("La varianza explicada del componente j es λⱼ/Σλ. Con p variables estandarizadas Σλ = p. La acumulada indica cuánta información conservas con k componentes. "
              "Reglas de apoyo: el <b>umbral</b> (80 o 90 %), el <b>codo</b> del scree plot y el criterio de Kaiser (λ &gt; 1); la decisión final se contrasta con el desempeño en validación.")
    s.append(ejemplo("seis variables estandarizadas", [
        P("λ = 2.4, 1.5, 1.0, 0.6, 0.3, 0.2 (suman 6.0).", "cell"),
        pasos([["Componente", "Cálculo", "Varianza · acumulada"],
               ["PC1", "2.4/6", "40.0 % · 40.0 %"],
               ["PC2", "1.5/6", "25.0 % · 65.0 %"],
               ["PC3", "1.0/6", "16.7 % · 81.7 %"],
               ["PC4", "0.6/6", "10.0 % · 91.7 %"],
               ["Para 80 % · para 90 % · Kaiser (λ &gt; 1)", "primer k con acumulada ≥ umbral", "k = 3 · k = 4 · k = 2"]],
              [44 * mm, 60 * mm, CONTENT_W - 104 * mm - 18]),
        P("Los criterios no coinciden y está bien: son referencias, no verdades (la profesora insistió en bajarlas al contexto). En los datos de la clase, 8 componentes dan 85.0 % y en la base real 7 dan 80.4 %.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_scree.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook (C2 y C3) y el Bloque 1 y 2 del cuaderno",
                   [P("Dibuja el scree plot, calcula k para 80, 85 y 90 % y elige k con <font name='Mono'>GridSearchCV</font> sobre un Pipeline.", "cell")]))

    s.append(h("5", "Cargas y correlaciones con los componentes"))
    s += para("Cada fila de <font name='Mono'>components_</font> es un componente; sus cargas (norma 1) dicen cuánto pesa cada variable. El puntaje de un cliente es Σ carga·z. "
              "La correlación entre una variable y el componente es <b>carga · √λ</b>. El signo global de un componente es arbitrario; lo que importa es la relación entre signos.")
    s.append(ejemplo("PC1 con cuatro variables y λ = 1.6", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Cargas de PC1: (0.5, 0.5, 0.5, −0.5): norma", "√(4 · 0.25)", "1 (correcto)"],
               ["Cliente z = (2, 1, 0, 1): puntaje", "0.5·2 + 0.5·1 + 0.5·0 − 0.5·1", "1.0"],
               ["Correlación de la variable 1 con PC1", "0.5 · √1.6 = 0.5 · 1.265", "0.63"],
               ["Cómo se llama el componente", "tres variables suben juntas y la cuarta baja", "«A, B y C altas / D baja»"]],
              [62 * mm, 66 * mm, CONTENT_W - 128 * mm - 18]),
        P("El nombre es una interpretación: ponerlo exige leer las cargas, no confiar en la diapositiva. En los datos de la clase PC1 (34.9 %) reúne segundos, llamadas, contactos, SMS, monto y CLV: «uso del servicio».", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_loadings.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(img("c32_escala.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook (C4) y el Bloque 4 del cuaderno",
                   [P("Imprime <font name='Mono'>components_</font> como tabla, grafica el círculo de correlaciones y repite el PCA sin estandarizar para ver qué cambia.", "cell")]))

    s.append(h("6", "t-SNE, PCA y la decisión final"))
    s += para("<b>t-SNE</b> calcula qué tan probable es que cada cliente sea vecino de otro y mueve puntos en 2D hasta reproducir esas vecindades. <b>Perplexity</b> controla el tamaño de la vecindad. "
              "No es determinista, no tiene <font name='Mono'>transform</font> y no se interpretan distancias entre islas. Para modelar se usa PCA (o selección).")
    s.append(ejemplo("elegir la técnica", [
        pasos([["Situación", "Técnica", "Por qué"],
               ["Seis variables de uso casi iguales y un modelo liviano para producción", "PCA (o seleccionar una)", "reduce y transforma datos nuevos"],
               ["Ver si hay estructuras locales no lineales entre clientes", "t-SNE", "explorar y visualizar"],
               ["Tengo segundos y llamadas y quiero la duración media", "feature engineering", "cociente, antes de reducir"],
               ["Una columna no aporta y complica la explicación", "selección", "conserva variables originales"]],
              [74 * mm, 40 * mm, CONTENT_W - 114 * mm - 18]),
        P("Un modelo que aprendió con 8 componentes necesita el mismo PCA (con el mismo escalador) al predecir: por eso todo va en un Pipeline. "
          "Con t-SNE no se puede: cada vez que llega un cliente habría que recalcular el mapa.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c32_tsne.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook y el Bloque 6 y 7 del cuaderno",
                   [P("Prueba perplexity 5, 30 y 80, cambia la semilla, calcula la trustworthiness y decide qué técnica usarías en cada caso.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(recuerda(["Sesión 31 (Introducción al ML) todavía no está en este material: aquí solo se usó lo que la clase 32 recuerda de ella (supervisado vs no supervisado).",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–C del notebook → Bloques 5–7 → Parte D → ejercicio de cierre."]))
    build("apuntes/C32_Apuntes_ReduccionDimensionalidad.pdf", s, "Apuntes · Clase 32 · Reducción de dimensionalidad")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#DB2777">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C32))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 32 · Reducción de dimensionalidad",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C32))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: varianza explicada = λⱼ/Σλ · [[1, r], [r, 1]] tiene λ = 1 ± r, vectores (1, ±1)/√2 · puntaje = v · z · correlación = carga · √λ · VIF = 1/(1 − R²).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."],
                          "cell", C32)], bg=SOFT, border=C32))
    s.append(Spacer(1, 2))
    s += bloque(1, "Varianza explicada desde los valores propios", "★", "Seis variables estandarizadas; los valores propios son 2.7, 1.5, 0.8, 0.5, 0.3 y 0.2.", [
        "Comprueba que suman p y calcula la varianza explicada y la acumulada de cada componente.",
        "¿Cuántos componentes necesitas para al menos 80 % y para 90 %? ¿Y según Kaiser (λ &gt; 1)?"])
    s += bloque(2, "Elegir el número de componentes", "★★", "Ocho variables estandarizadas; λ = 3.2, 1.6, 1.1, 0.8, 0.6, 0.4, 0.2, 0.1.", [
        "Calcula la acumulada y el k para 80 % y 90 %.",
        "Con validación cruzada obtienes AUC 0.71 (k = 2), 0.78 (4), 0.79 (5) y 0.79 (8). ¿Qué k elegirías y por qué?"])
    s += bloque(3, "PCA a mano con 2 variables", "★★★", "Matriz de correlación [[1, 0.8], [0.8, 1]].", [
        "Calcula los valores propios, los vectores propios y la varianza explicada de cada componente.",
        "Proyecta z = (1.2, 0.8) en PC1 y PC2. Reconstruye con solo PC1 y calcula el error.",
        "Sin estandarizar, la covarianza es [[400, 20], [20, 4]] (valores propios 401.0 y 3.0). ¿Qué fracción explica PC1 y qué variable domina? ¿Por qué se estandariza?"])
    s += bloque(4, "Leer cargas", "★★", "PC1 con λ = 2.5. Cargas (A, B, C, D, E) = (0.48, 0.46, 0.45, 0.05, −0.59).", [
        "¿Qué variables definen PC1 y qué significa el signo de E? Verifica que la norma sea aproximadamente 1.",
        "Calcula la correlación de A y E con PC1, y el puntaje de un cliente z = (1, 2, 0.5, −1, −0.5).",
        "Un colega dice «PC1 es la variable más importante para predecir la fuga». ¿Qué le respondes?"])
    s += bloque(5, "Elegir la codificación", "★★", None, [
        "Elige ordinal, one-hot, target/frequency o numérica: (a) tipo de plan (2 categorías), (b) nivel educativo (básico, medio, superior), (c) región (5 sin orden), (d) comuna (300 categorías), (e) monto 0 a 9.",
        "La comuna B tiene 10 clientes y 4 fugas; la tasa global es 0.15. Calcula la tasa cruda y la suavizada con m = 20, y explica por qué la cruda es riesgosa.",
        "¿Por qué la media del target calculada con todas las filas es fuga de datos?"])
    s += bloque(6, "Dimensiones después de codificar", "★★", "8 variables numéricas; categóricas: plan (2), región (5), estado civil (4), canal (6); nivel (ordinal con 3 niveles) y 2 variables construidas.", [
        "¿Cuántas columnas hay con one-hot sin drop (nivel ordinal en 1 columna)? ¿Y con drop = 'first'?",
        "Si PCA conserva 90 % con 9 componentes, ¿qué fracción de columnas se ahorra con el esquema drop = 'first'?"])
    s += bloque(7, "Pipeline, fuga y técnica adecuada", "★★★", None, [
        "¿Fuga o no? (a) StandardScaler ajustado con todas las filas, luego dividir; (b) PCA dentro de un Pipeline evaluado con K-Fold; (c) elegir k mirando la prueba; (d) crear duracion_prom_llamada antes de dividir.",
        "Elige PCA, t-SNE, selección o feature engineering: (a) modelo liviano con seis variables de uso correlacionadas, (b) ver estructura local no lineal, (c) una columna sin información, (d) distinguir llamadas largas de cortas.",
        "Explica por qué t-SNE no sirve como paso de un modelo en producción (dos razones)."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 4))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 3)]

    s += sol("Bloque 1", [
        ["1", "Σλ = 2.7 + 1.5 + 0.8 + 0.5 + 0.3 + 0.2 = 6.0 = p. Varianza: 45.0, 25.0, 13.3, 8.3, 5.0, 3.3 %. Acumulada: 45.0, 70.0, 83.3, 91.7, 96.7, 100 %."],
        ["2", "80 %: k = 3 (83.3). 90 %: k = 4 (91.7). Kaiser (λ &gt; 1): k = 2."]])
    s += sol("Bloque 2", [
        ["1", "Σλ = 8. Acumulada: 40.0, 60.0, 73.8, 83.8, 91.3, 96.3, 98.8, 100 %. 80 %: k = 4. 90 %: k = 5."],
        ["2", "k = 4: ya llega a 0.78 y 5 o 8 casi no mejoran (+0.01). Se elige el más pequeño que rinde igual (parsimonia), y se compara dentro del Pipeline."]])
    s += sol("Bloque 3", [
        ["1", "λ = 1 + 0.8 = 1.8 y 1 − 0.8 = 0.2. Vectores (0.707, 0.707) y (0.707, −0.707). Varianza explicada: 1.8/2 = 90 % y 10 %."],
        ["2", "PC1 = (1.2 + 0.8)/√2 = 1.414 · PC2 = (1.2 − 0.8)/√2 = 0.283. Reconstrucción con PC1: 1.414·(0.707, 0.707) = (1.0, 1.0). Error: (0.2² + 0.2²) = 0.08 (= 0.283²)."],
        ["3", "PC1 explica 401.0/404 = 99.3 % y casi coincide con la variable de varianza 400 (carga ≈ 0.999). Es un efecto de la unidad: se estandariza para que cada variable pese igual."]])
    s += sol("Bloque 4", [
        ["1", "A, B y C (≈ 0.46) definen PC1; E pesa −0.59: cuando A, B y C suben, E baja. Norma² = 0.2304 + 0.2116 + 0.2025 + 0.0025 + 0.3481 = 0.995 ≈ 1."],
        ["2", "√2.5 = 1.581: A: 0.48·1.581 = 0.76; E: −0.59·1.581 = −0.93. Puntaje = 0.48 + 0.92 + 0.225 − 0.05 + 0.295 = 1.87."],
        ["3", "PCA no usa la etiqueta: PC1 es la dirección de mayor varianza, no necesariamente la que predice. Hay que contrastar con la fuga (aquí PC3, no PC1, es la que más se correlaciona)."]])
    s += sol("Bloque 5", [
        ["1", "(a) one-hot con drop if_binary (una columna 0/1) · (b) ordinal · (c) one-hot · (d) target/frequency con validación interna · (e) ya es ordinal numérica: se deja como número (o se estandariza)."],
        ["2", "Cruda: 4/10 = 0.40. Suavizada: (4 + 20·0.15)/(10 + 20) = 7/30 = 0.233. Con 10 clientes la cruda es muy ruidosa y el modelo la memorizaría; la suavizada se acerca a la tasa global."],
        ["3", "Porque usa la etiqueta de las filas de prueba para construir una variable de entrada; el modelo «ve» la respuesta. Debe calcularse con entrenamiento (idealmente con validación cruzada interna)."]])
    s += sol("Bloque 6", [
        ["1", "Sin drop: 8 + (2 + 5 + 4 + 6) + 1 + 2 = 28. Con drop = 'first': 8 + (1 + 4 + 3 + 5) + 1 + 2 = 24."],
        ["2", "Con 24 columnas y 9 componentes: 9/24 = 37.5 % de las columnas; se ahorra 62.5 %. (Con las 28 columnas: 9/28 = 32 %; se ahorra 68 %.)"]])
    s += sol("Bloque 7", [
        ["1", "(a) fuga · (b) no · (c) fuga · (d) no (es una operación fila a fila que no aprende nada del conjunto)."],
        ["2", "(a) PCA (o seleccionar una) · (b) t-SNE · (c) selección · (d) feature engineering (segundos / frec_llamadas)."],
        ["3", "(1) No tiene transform: no proyecta clientes nuevos sin recalcular todo. (2) No es determinista y es costoso; el resultado cambia con la semilla."]])
    build("apuntes/C32_Ejercicios_Cuaderno_ReduccionDimensionalidad.pdf", s, "Ejercicios para el cuaderno · Clase 32")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
