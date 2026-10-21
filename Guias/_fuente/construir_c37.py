"""Clase 37 · SVM y clasificadores clásicos: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C37 = colors.HexColor("#0284C7")
C37_BG = colors.HexColor("#F0F9FF")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + scripts clasificadores.py / .R + wisconsin_clase.csv del profesor Pedro Luiz Ramos"


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado):
    """figuras: lista de (archivo, ancho_relativo, pie)."""
    s.append(band(f"Clase 37 · {titulo}", FUENTE, C37))
    s.append(Spacer(1, 5))
    s.append(box([P(idea, "big")], bg=SOFT, border=C37))
    s.append(Spacer(1, 4))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C37))
    s.append(Spacer(1, 2))
    for f, w, pie in figuras:
        s.append(img(f, width=CONTENT_W * w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 3))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF)]
    if py:
        left += [Spacer(1, 4), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C37), Spacer(1, 4),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER, width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 37 · SVM y clasificadores clásicos",
                  "Profesor Pedro Luiz Ramos · regresión logística y máquinas de vectores de soporte · sigue a la Clase 36 (Boosting) y precede a la Clase 38", NAVY))
    s.append(Spacer(1, 7))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Clasificar es construir una regla que, a partir de las variables <b>X</b>, asigne una clase (0 o 1). La <b>regresión logística</b> estima una <b>probabilidad</b> y "
                    "decide con un <b>umbral</b>; la <b>SVM</b> busca el <b>hiperplano de margen máximo</b>, entrega un <b>puntaje</b> f(x) y decide con su signo. "
                    "El <b>kernel</b> permite fronteras curvas y los <b>hiperparámetros</b> (C y gamma) se eligen con validación cruzada <b>solo en entrenamiento</b>. "
                    "«No hay un modelo que siempre gane»: se comparan en datos que no se usaron para ajustar.", "body")],
                 bg=C37_BG, border=C37))
    s.append(Spacer(1, 5))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c37_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Regresión logística y odds (Clases 20–21 y 31) y la matriz de confusión (Clases 30–31)",
                     "Train / prueba, validación cruzada y fuga de información (Clase 30): aquí se usan en serio",
                     "Estandarizar (z = (x − media)/sd) y el uso de un Pipeline (Clase 31)",
                     "Idea de regularización (Ridge y Lasso, Clases 30–31): C de la SVM juega un papel parecido"], "cell", C37)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Clasificación supervisada</b> → <b>métricas</b> → <b>validación y fuga</b> → <b>regresión logística</b> → <b>SVM: margen, C, kernel, gamma</b> → <b>comparar con prueba</b> → <b>AUC</b>",
                      "Datos: <b>Breast Cancer Wisconsin</b>, 569 muestras (357 benignas, 212 malignas), cuatro predictores (radio, textura, suavidad, concavidad) y la partición ya hecha: "
                      "<b>398 de entrenamiento y 171 de prueba</b>. Clase positiva: maligno (1)",
                      "Pregunta guía: ¿qué regla de decisión usar, con qué hiperparámetros y cómo sabré que funciona en casos nuevos?"], "cell", C37)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Puentes con las clases vecinas</b>", "cellb"),
                  P("<b>C31 → C37</b>: la logística y las métricas de la Clase 31 aquí se comparan con la SVM en una misma partición. "
                    "<b>C34–C36 → C37</b>: árboles, bosques y boosting son otros clasificadores de la «familia de 15 métodos» que menciona el profesor; no se repiten aquí. "
                    "<b>C37 → C38</b>: la evaluación de modelos supervisados (Eloy Alvarado) profundiza AUC, umbrales y validación; aquí solo se usa lo necesario para comparar tres modelos.", "cell")],
                 bg=SOFT, border=C37))
    s.append(PageBreak())

    topic_page(
        s, "Regresión logística: de la probabilidad a la clase",
        "La regresión logística modela <b>directamente la probabilidad</b> de la clase positiva con un predictor lineal η = β₀ + β·x y la función logística. "
        "El <b>umbral</b> convierte la probabilidad en clase; cambiarlo mueve la decisión, <b>no</b> la probabilidad.",
        [("Puntaje, probabilidad y odds", "p = 1/(1 + e<super>−η</super>); η = 0 equivale a p = 0.5. Cada unidad de x<sub>j</sub> multiplica los <i>odds</i> p/(1 − p) por e<super>β<sub>j</sub></super> (asociación condicional, no efecto causal)."),
         ("Umbral t", "Clase 1 si p ≥ t. Con t = 0.5 es lo habitual, pero se puede fijar con un criterio previo (costos, validación cruzada), nunca mirando la prueba."),
         ("Sensibilidad y especificidad", "Sensibilidad = VP/(VP + FN) (malignos detectados); especificidad = VN/(VN + FP) (benignos bien clasificados). Subir t gana especificidad y pierde sensibilidad."),
         ("R frente a Python", "<font name='Mono'>glm(family = binomial)</font> no penaliza; <font name='Mono'>LogisticRegression()</font> <b>sí</b> penaliza por defecto: hay que poner <font name='Mono'>penalty=None</font>.")],
        [("c37_umbral.png", 0.86,
          "Izquierda: sensibilidad, especificidad y exactitud según el umbral (validación cruzada en entrenamiento; líneas en 0.5 y 0.8). Centro: curvas ROC en prueba (los puntos son la logística con t = 0.5 y 0.8). "
          "Derecha: la probabilidad de cada caso de prueba no cambia con el umbral; cambia a cuántos se deja de un lado.")],
        ["p = 1 / (1 + e<super>−η</super>),  η = β₀ + β·x",
         "odds = p / (1 − p) = e<super>η</super>",
         "odds nuevos / odds antiguos = e<super>β<sub>j</sub></super>",
         "frontera con t = 0.5:  β₀ + β·x = 0 (un hiperplano)"],
        ["LogisticRegression(penalty=None,",
         "                   max_iter=2000)",
         "p = modelo.predict_proba(X)[:, 1]",
         "clase = (p >= 0.5).astype(int)"],
        ["Prueba de 171 casos con t = 0.5: VN 103, FP 4, FN 6, VP 58 → exactitud 94.15 %, sensibilidad 90.62 %, especificidad 96.26 %",
         "Con t = 0.8: VN 105, FP 2, FN 12, VP 52 → 14 errores en vez de 10 (como se vio en clase); sensibilidad 81.25 %, especificidad 98.13 %",
         "Coeficientes de R: intercepto −42.2511, radio 1.3399, textura 0.4421 (e<super>0.4421</super> = 1.56), suavidad 124.9794, concavidad 21.4077; caso nuevo (14, 20, 0.10, 0.08): p = 0.3919 → clase 0"],
        ["Coeficientes de Python (estandarizados) y de R (originales) difieren en escala pero dan las <b>mismas probabilidades</b>. «+1» de suavidad (rango 0.05–0.16) no tiene sentido; mejor «+0.01»: odds × 3.49."])

    topic_page(
        s, "SVM: hiperplano, margen y vectores de soporte",
        "La SVM busca la frontera f(x) = w·x + b = 0 que <b>separa las clases con el mayor margen</b>. El signo de f(x) decide la clase y solo los "
        "<b>vectores de soporte</b> (los casos sobre el margen o dentro de él) determinan la solución.",
        [("Hiperplano y puntaje", "f(x) = w·x + b. w da la orientación, b desplaza la frontera. La distancia geométrica de un punto a la frontera es |f(x)|/‖w‖."),
         ("Margen máximo", "Con yᵢ f(xᵢ) ≥ 1 el ancho total del margen es 2/‖w‖; minimizar ‖w‖² lo maximiza. Los puntos con yᵢ f(xᵢ) = 1 son los vectores de soporte."),
         ("Margen suave", "Si las clases se superponen se permite una holgura ξᵢ = máx(0, 1 − yᵢ f(xᵢ)): 0 si el caso está bien y fuera del margen; entre 0 y 1 si está dentro; mayor que 1 si está mal clasificado."),
         ("Etiquetas −1 y +1", "Son una convención para explicar la geometría; en <font name='Mono'>SVC</font> y <font name='Mono'>svm()</font> las clases pueden seguir siendo 0 y 1.")],
        [("c37_margen.png", 0.95,
          "Izquierda: margen duro en un subconjunto separable de 32 casos (3 vectores de soporte; ancho 2/‖w‖ = 1.18 en unidades estandarizadas). Centro: margen suave con C = 1 sobre los 398 casos de entrenamiento "
          "(radio y concavidad): 83 vectores de soporte. Derecha: holgura del modelo de la clase (4 variables, lineal, C = 0.1).")],
        ["f(x) = w·x + b;  frontera f(x) = 0",
         "distancia = |f(x)| / ‖w‖;  ancho del margen = 2 / ‖w‖",
         "ξᵢ = máx(0, 1 − yᵢ f(xᵢ))",
         "mín ½‖w‖² + C Σ ξᵢ"],
        ["svm = SVC(kernel='linear', C=0.1)",
         "svm.fit(X_train_esc, y_train)",
         "svm.decision_function(X)  # f(x)",
         "svm.support_vectors_, svm.n_support_"],
        ["Modelo de la clase (lineal, C = 0.1, 4 variables): 93 vectores de soporte (47 + 46); ‖w‖ = 1.73",
         "Entrenamiento (398): 20 mal clasificados, 70 dentro del margen y 308 correctos y fuera del margen",
         "Matriz en prueba: VN 103, FP 4, FN 10, VP 54 → 14 errores; exactitud 91.81 %, sensibilidad 84.38 %, especificidad 96.26 %",
         "Cuenta de la diapositiva: w = (1, −1), b = −0.5, x = (2, 1) → f = 0.5, ‖w‖ = √2, distancia 0.354"],
        ["Un margen amplio <b>no garantiza</b> por sí solo buen desempeño en datos nuevos. Y f(x) <b>no es una probabilidad</b>: sus valores no están entre 0 y 1."])

    topic_page(
        s, "Margen suave y el hiperparámetro C",
        "C es el <b>peso de las violaciones del margen</b> en la función que se minimiza. <b>C pequeño</b>: se toleran más violaciones, margen ancho y regularización más fuerte; "
        "<b>C grande</b>: se castigan casi todas, margen estrecho y frontera más pegada al entrenamiento. No existe «el mejor C» sin validación.",
        [("Qué controla", "En mín ½‖w‖² + C Σ ξᵢ, C mide cuánto pesa Σ ξᵢ frente a ‖w‖². Con Σ ξᵢ = 4: contribución 0.4 si C = 0.1 y 40 si C = 10."),
         ("Se ve en el margen y en los soportes", "Al subir C el ancho 2/‖w‖ baja y quedan menos vectores de soporte (menos casos violan el margen)."),
         ("Tampoco «mientras más, mejor»", "Con C muy pequeño hay subajuste; con C muy grande la frontera se ajusta a particularidades del entrenamiento. La validación cruzada decide."),
         ("C de Python = cost de R", "Ambos son el mismo parámetro de libsvm; por eso R y Python entregan las mismas matrices de confusión con los mismos datos y escala.")],
        [("c37_C.png", 1.0,
          "Tres valores de C con un kernel lineal sobre radio y concavidad: el margen se estrecha de 2.24 a 0.86 y 0.81 y los soportes bajan de 188 a 83 y 78. "
          "Derecha: con las 4 variables, la exactitud de validación cruzada sube hasta C ≈ 0.1 y luego se mantiene o baja ligeramente.")],
        ["objetivo = ½‖w‖² + C · Σ ξᵢ",
         "C ↓ : margen ancho, más violaciones, más simple",
         "C ↑ : margen estrecho, menos violaciones, más flexible",
         "ancho = 2 / ‖w‖"],
        ["SVC(kernel='linear', C=0.1)",
         "# R: svm(..., kernel='linear', cost=0.1)",
         "GridSearchCV(pipe, {'svc__C': [0.1, 1, 10]},",
         "             cv=5)"],
        ["2 variables: C = 0.01 → 188 soportes y ancho 2.24; C = 1 → 83 y 0.86; C = 100 → 78 y 0.81",
         "4 variables: validación cruzada (5 folds) 0.917 con C = 0.01, 0.940 con C = 0.1 y 0.930 con C = 1000; mejor C ≈ 0.1",
         "Exactitud de prueba del lineal: 0.918 con C = 0.1 y 0.936 con C ≥ 0.3: la prueba no se usa para elegir C",
         "El profesor fija C = 0.1 (lineal) y C = 10 (radial) <b>antes</b> de ver la prueba"],
        ["«Con C pequeño no dejo que acepte errores» es lo contrario: C <b>pequeño</b> tolera <b>más</b> violaciones. Y elegir C mirando la prueba convierte la prueba en validación."])

    topic_page(
        s, "Kernels y gamma: fronteras curvas",
        "Un <b>kernel</b> K(x, z) = ⟨φ(x), φ(z)⟩ calcula un producto interno en un espacio de características sin construirlo (<i>truco del kernel</i>): la frontera es lineal allí y curva en el espacio original. "
        "En el kernel <b>radial (RBF)</b> el parámetro <b>γ</b> fija el alcance de la similitud.",
        [("Tres kernels", "Lineal x·z; polinomial (γ x·z + r)<super>d</super> (hiperparámetros γ, r y grado d); radial exp(−γ‖x − z‖²). El lineal no tiene γ; el polinomial sí."),
         ("γ pequeño y γ grande", "γ pequeño: la similitud decae lento y la frontera se parece a una recta (subajuste). γ grande: influencia muy local y fronteras irregulares que memorizan (sobreajuste)."),
         ("γ y C juntos", "Con γ muy grande ningún C salva el modelo; con γ muy pequeño el RBF se parece al lineal. Se buscan en rejilla (GridSearchCV, tune.svm). No hay un kernel mejor en general.")],
        [("c37_kernels.png", 0.78,
          "Lineal, polinomial (grado 3) y RBF sobre radio y concavidad (exactitud de prueba 88.3 %, 88.9 % y 90.1 %). Derecha: la similitud RBF cae con la distancia; a distancia 2, γ = 0.1 da 0.6703 y γ = 1 da 0.0183."),
         ("c37_gamma.png", 0.78,
          "RBF con C = 1: γ = 0.01 (casi recta), 1 (curva suave) y 50 (islas alrededor de casos aislados). Derecha: con 4 variables la exactitud de entrenamiento sube siempre, la de validación cruzada tiene un máximo y cae.")],
        ["K(x, z) = exp(−γ ‖x − z‖²)",
         "‖x − z‖² = 4: γ = 0.1 → 0.6703; γ = 1 → 0.0183",
         "K<sub>poli</sub> = (γ x·z + r)<super>d</super>"],
        ["SVC(kernel='rbf', C=10, gamma=0.1)"],
        ["2 variables (C = 1): γ = 0.01 → entrenamiento 92.2 %, prueba 90.6 %; γ = 1 → 93.2 % / 90.6 %; γ = 50 → 96.0 % / 88.9 %",
         "4 variables (C = 1): validación cruzada máxima 0.945 con γ ≈ 0.046; con γ = 100, 0.633 (entrenamiento 100 %)"],
        ["«Gamma entre 0.1 y 0.5 es lo mejor» no es una regla: depende de la escala de los datos (por eso se estandariza) y se decide con validación."])

    topic_page(
        s, "Estandarizar, validar y no filtrar información",
        "La SVM usa <b>distancias</b>: si una variable tiene una escala mucho mayor, domina. Se estandariza con la media y la desviación del <b>entrenamiento</b> y se dejan en un "
        "<font name='Mono'>Pipeline</font> para que dentro de cada fold se aprendan solo con los folds de ajuste. Los hiperparámetros se eligen en <b>entrenamiento</b>; la prueba se mira <b>una vez</b>.",
        [("Tres funciones de los datos", "Entrenamiento: estima coeficientes, escalas y vectores de soporte. Validación: compara hiperparámetros y reglas de decisión. Prueba: evalúa el procedimiento ya elegido."),
         ("Validación cruzada de k folds", "Se divide el entrenamiento en k partes; cada una valida una vez con un modelo ajustado en las otras k − 1. Con 3 valores de C, 3 de γ y 5 folds son 45 ajustes."),
         ("Fuga de información", "Calcular media y sd (o imputar, o seleccionar variables) con toda la base antes de dividir contamina la validación. Tampoco se elige C, γ o t mirando la prueba."),
         ],
        [("c37_escala.png", 0.64,
          "Izquierda: las desviaciones estándar van de 0.014 (suavidad) a 4.2 (textura), ×293. Centro: RBF con radio y suavidad: sin estandarizar la frontera es casi una recta vertical en el radio. "
          "Derecha: exactitud de prueba con 4 variables, sin y con StandardScaler."),
         ("c37_rejilla.png", 0.64,
          "Izquierda: la rejilla 3×3 del profesor (mejor C = 10, γ = 0.1, exactitud media 0.9396). Centro: rejilla ampliada 6×5 (mejor C = 100, γ = 0.01, 0.9447). "
          "Derecha: las nueve combinaciones con ± 1 sd entre folds: varias empatan dentro del ruido.")],
        ["z = (x − media<sub>entr</sub>) / sd<sub>entr</sub>",
         "exactitud = 1 − error",
         "ajustes = |C| · |γ| · folds"],
        ["make_pipeline(StandardScaler(), SVC())",
         "GridSearchCV(modelo, {'svc__C': [...],",
         "   'svc__gamma': [...]}, cv=5)"],
        ["Sin estandarizar la prueba cae de 91.8 % a 86.0 % (lineal) y de 97.1 % a 90.1 % (RBF)",
         "Rejilla de la diapositiva 62: mejor C = 10, γ = 0.1; la ampliada elige C = 100, γ = 0.01 (prueba 94.7 %, menos que el 97.1 % del profesor)"],
        ["Una diferencia de 1 a 3 puntos <b>no es</b> evidencia de superioridad: en 171 casos, uno vale 0.58 puntos y los folds varían ± 2 a 4 puntos."])

    # ------------------------------------------------------------ comparar modelos
    s.append(band("Comparar logística y SVM: fronteras, probabilidad y R ↔ Python", "Primero el problema y el costo de los errores, luego el modelo", NAVY))
    s.append(Spacer(1, 5))
    s.append(img("c37_logistica_svm.png", width=CONTENT_W))
    s.append(P("Con radio y concavidad: logística (probabilidad y frontera lineal), SVM lineal (recta con margen) y SVM radial (frontera curva), en prueba (88.9 %, 88.3 % y 90.1 %). "
               "Derecha: con <font name='Mono'>probability=True</font> la SVM calcula una probabilidad por ajuste de Platt a partir del puntaje; en 3 casos de prueba <font name='Mono'>predict</font> y «p ≥ 0.5» discrepan.", "small"))
    s.append(Spacer(1, 3))
    s.append(table([["Modelo (prueba, 171 casos)", "VN · FP · FN · VP", "Exactitud", "Sensib.", "Especif.", "Precisión", "AUC"],
                    ["Logística (t = 0.5)", "103 · 4 · 6 · 58", "0.9415", "0.9062", "0.9626", "0.9355", "0.9746"],
                    ["SVM lineal (C = 0.1)", "103 · 4 · 10 · 54", "0.9181", "0.8438", "0.9626", "0.9310", "0.9746"],
                    ["SVM radial (C = 10, γ = 0.1)", "106 · 1 · 4 · 60", "0.9708", "0.9375", "0.9907", "0.9836", "0.9834"],
                    ["Línea base «siempre benigno»", "107 · 0 · 64 · 0", "0.6257", "0.0000", "1.0000", "—", "0.5"]],
                   [50 * mm, 32 * mm, 18 * mm, 16 * mm, 16 * mm, 18 * mm, CONTENT_W - 150 * mm]))
    s.append(Spacer(1, 4))
    s.append(table([["Criterio", "Regresión logística", "SVM lineal", "SVM radial"],
                    ["Salida", "probabilidad p(x)", "puntaje f(x)", "puntaje f(x)"],
                    ["Frontera aquí", "lineal", "lineal", "no lineal"],
                    ["Interpretación", "coeficientes y odds", "pesos y margen", "más difícil"],
                    ["Ajustes", "penalización (opcional)", "C y escala", "C, γ y escala"],
                    ["R ↔ Python", "glm(family = binomial) ↔ LogisticRegression(penalty=None)", "svm(kernel = «linear», cost) ↔ SVC(kernel=«linear», C)", "svm(kernel = «radial», cost, gamma) ↔ SVC(kernel=«rbf», C, gamma)"]],
                   [28 * mm, 55 * mm, 45 * mm, CONTENT_W - 128 * mm]))
    s.append(Spacer(1, 4))
    checks = [("Fijo la clase positiva (maligno = 1) y el orden de la matriz", "filas = real, columnas = predicho, orden 0, 1"),
              ("Divido antes de estandarizar, imputar o ajustar", "todo lo que aprende de los datos va en un Pipeline"),
              ("Elijo C, γ y el umbral con validación cruzada en entrenamiento", "la prueba se mira una sola vez"),
              ("Comparo con la línea base y mido sensibilidad además de exactitud", "los errores (FN y FP) casi nunca cuestan lo mismo"),
              ("Uso puntajes (no clases 0/1) para la AUC", "decision_function o predict_proba"),
              ("Documento las limitaciones", "base histórica de una región: no es una herramienta clínica")]
    s.append(table([["", "Checklist del flujo", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 78 * mm, CONTENT_W - 86 * mm]))
    s.append(PageBreak())

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets(["<b>«Con la base de 357 vamos a dividir en 70 y 30»</b>: 357 son solo los benignos; la base tiene 569 casos y la partición es 398 / 171 (250 + 148 y 107 + 64). La grabación dice además «559 muestras».",
                           "<b>Regla «siempre uno» con 107 aciertos</b>: la regla que acierta 107 de 171 es la que predice <b>benigno (0)</b>, no 1. Siempre 1 acertaría 64 (37.4 %). La diapositiva 7 lo dice bien (0.626).",
                           "<b>«Para SVM hay que usar −1 y 1»</b>: es solo la convención para la geometría; <font name='Mono'>SVC</font> y <font name='Mono'>svm()</font> aceptan 0/1 y dan las mismas predicciones (los scripts usan 0 y 1; lo aclara la diapositiva 31).",
                           "<b>«El estimador de máxima verosimilitud tiene garantizado que la solución es única»</b>: no siempre. Con <b>separación perfecta</b> la verosimilitud sigue creciendo al aumentar los coeficientes y no hay máximo finito; con colinealidad perfecta tampoco es única (se ve en el notebook).",
                           "<b>Falsos negativos</b>: «la persona va a ser mal pagador y predecimos buen pagador» es un falso <b>positivo</b> con la codificación 1 = buen pagador. Un falso negativo es lo contrario (real 1, predicho 0).",
                           "<b>Kernels e hiperparámetros</b>: «el polinomial tiene un hiperparámetro, el radial gamma, el lineal ninguno». El polinomial (γ x·z + r)<super>d</super> tiene tres: γ, r y el grado d; solo el lineal no tiene γ.",
                           "<b>«Cuando voy a hacer inferencia, siempre me voy con regularización»</b>: la penalización <b>sesga</b> los coeficientes y no entrega los intervalos de confianza clásicos; es una herramienta de predicción y selección. Para inferir hay que tener cuidado (inferencia tras selección).",
                           "<b>«C = 1 y gamma = 0.1 … los dos valores iguales a 10»</b> (Python): el script fija C = 10 y γ = 0.1 (diapositiva 43); la frase de la grabación está mal transcrita o es un error al hablar."],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Diapositivas y scripts</b>", "cellb"),
                  bullets(["<b>Diapositiva 28</b>: la interpretación «una unidad de suavidad multiplica los odds por e<super>124.98</super>» no tiene sentido práctico: el rango de suavidad es 0.05 a 0.16. Mejor «+0.01» (×3.49) o usar variables estandarizadas. "
                           "Los coeficientes de R (−42.2511, 1.3399, 0.4421, 124.9794, 21.4077) <b>se reproducen exactamente</b> con <font name='Mono'>LogisticRegression(penalty=None)</font> sin estandarizar (p = 0.3919 para el caso nuevo).",
                           "<b>Diapositiva 63 (R: C = 1, γ = 0.1; Python: C = 10, γ = 0.1)</b>: no es una contradicción; en Python (10, 0.1) gana con 0.9396 y (1, 0.1) queda segunda con 0.9347, dentro de 1 sd. Y la rejilla ampliada elige otra (100, 0.01) con prueba 94.7 % frente al 97.1 % de la configuración del profesor: ningún número de prueba justifica elegir.",
                           "<b>AUC</b>: la logística y la SVM lineal dan <b>la misma AUC</b> (0.9746) con distinta exactitud; la SVM radial 0.9834. La AUC mide el orden de los puntajes, no el punto de corte (diapositiva 64).",
                           "<b>Probabilidad de la SVM</b>: <font name='Mono'>decision_function</font> no es probabilidad; <font name='Mono'>SVC(probability=True)</font> la calibra (Platt, con validación cruzada interna) y puede discrepar de <font name='Mono'>predict</font>: 3 casos y exactitud 95.3 % frente a 97.1 %.",
                           "<b>Diapositiva 16</b>: escribe «p = 0,731» con coma y en el resto del material con punto; es solo formato. <b>Script Python</b>: <font name='Mono'>os.chdir(r\"D:\\codigos\")</font> y <font name='Mono'>setwd(\"D:/codigos\")</font> son rutas del profesor; hay que cambiarlas."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper confundió Wisconsin («Luis Colzin»), glm / glmnet («RLM», «relíem net»), e1071 («e de 71»), Tibshirani, Breiman («Breeman»), k-folds («CAFOD»), Leave-One-Out, "
                    "kernel («crânio», «k-neu»), ROC («rock», «roca»), chance («iance», «yance»), ChatGPT («chate PT»). El volumen cayó ≈ 12 dB desde ≈ 65 min (se transcribió una copia amplificada); "
                    "el recreo (≈ 1:26:14) no está en el video. Frases dudosas de la clase se dejaron como el profesor las dijo.", "cell")],
                 bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"),
                  bullets(["Calculo a mano la probabilidad logística, los odds y la clase para un umbral dado",
                           "Calculo exactitud, sensibilidad, especificidad y precisión desde una matriz de confusión y la comparo con la línea base",
                           "Calculo f(x), la distancia a la frontera y el ancho del margen 2/‖w‖, e identifico vectores de soporte",
                           "Explico el efecto de subir C o γ y por qué se eligen con validación cruzada en entrenamiento",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C37_Guia_SVM.pdf", s, "Clase 37 · SVM y clasificadores clásicos")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#0284C7">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 37 · SVM y clasificadores clásicos",
                  "Para leer antes de los ejercicios · lectura de ~30 min", C37))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([
                      ("Probabilidad y decisión son cosas distintas", "La logística da p(x); el umbral t decide. Cambiar t no cambia p, solo la clase."),
                      ("La exactitud no basta", "Con la matriz de confusión se calculan sensibilidad, especificidad y precisión; siempre contra la línea base «clase mayoritaria»."),
                      ("SVM = margen máximo", "f(x) = w·x + b; el ancho del margen es 2/‖w‖ y solo los vectores de soporte importan. f(x) no es una probabilidad."),
                      ("C y γ regulan la flexibilidad", "C pequeño tolera violaciones (suave); γ grande hace fronteras locales (riesgo de sobreajuste)."),
                      ("Estandarizar y validar sin fugas", "Media y sd del entrenamiento, dentro de un Pipeline; C, γ y t se eligen con validación cruzada en entrenamiento."),
                      ("Se evalúa una sola vez", "La prueba compara el procedimiento ya elegido; mirarla para decidir la convierte en validación."),
                  ], C37)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 4))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C37_SVM.ipynb</font> o el cuaderno "
               "<font name='Mono'>C37_Ejercicios_Cuaderno_SVM.pdf</font>.", "small"))

    s.append(h("1", "Regresión logística: de η a la clase"))
    s += para("La regresión logística calcula un puntaje <b>η = β₀ + β₁z₁ + …</b> y lo convierte en probabilidad con <b>p = 1/(1 + e<super>−η</super>)</b>. Si η = 0, p = 0.5. "
              "Cada coeficiente multiplica los <i>odds</i> p/(1 − p) por e<super>β</super> al subir una unidad (con variables estandarizadas, una desviación estándar). "
              "Con umbral t, se predice 1 si p ≥ t; un t más alto da más especificidad y menos sensibilidad.")
    s.append(ejemplo("η = −2 + 0.9·z₁ + 0.5·z₂", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Caso A (z₁ = 2, z₂ = 1): puntaje", "−2 + 0.9·2 + 0.5·1", "η = 0.30"],
               ["Su probabilidad", "1/(1 + e<super>−0.3</super>) = 1/(1 + 0.741)", "p = 0.574"],
               ["Caso B (z₁ = −1, z₂ = 0.5)", "η = −2 − 0.9 + 0.25 = −2.65 → 1/(1 + e<super>2.65</super>)", "p = 0.066"],
               ["Efecto de +1 en z₁", "e<super>0.9</super>", "odds × 2.46"],
               ["Clase con t = 0.5 y con t = 0.7 (caso A)", "0.574 ≥ 0.5 · 0.574 &lt; 0.7", "1 · 0"]],
              [64 * mm, 66 * mm, CONTENT_W - 130 * mm - 18]),
        P("La probabilidad del caso A es 0.574 con cualquier umbral; solo cambia la decisión. Con t = 0.7 el modelo es más exigente: menos falsos positivos y más falsos negativos.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c37_umbral.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Ajusta <font name='Mono'>LogisticRegression(penalty=None)</font>, calcula a mano 1/(1 + e<super>−η</super>) para un caso de prueba y compara con <font name='Mono'>predict_proba</font>. Prueba t = 0.5 y 0.8.", "cell")]))

    s.append(h("2", "Matriz de confusión y métricas"))
    s += para("Con la clase positiva declarada (maligno = 1) y filas = real, columnas = predicho: VN y FP arriba, FN y VP abajo. "
              "<b>Exactitud</b> = (VP + VN)/n; <b>sensibilidad</b> = VP/(VP + FN); <b>especificidad</b> = VN/(VN + FP); <b>precisión</b> = VP/(VP + FP); "
              "<b>F1</b> = 2VP/(2VP + FP + FN); <b>exactitud balanceada</b> = (sensibilidad + especificidad)/2. La AUC evalúa el orden de los puntajes para todos los umbrales.")
    s.append(ejemplo("VN = 70, FP = 10, FN = 15, VP = 55 (n = 150)", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Exactitud", "(55 + 70)/150", "83.3 %"],
               ["Sensibilidad · especificidad", "55/70 · 70/80", "78.6 % · 87.5 %"],
               ["Precisión · F1", "55/65 · 110/(110 + 10 + 15)", "84.6 % · 0.815"],
               ["Exactitud balanceada", "(0.786 + 0.875)/2", "0.830"],
               ["Línea base «siempre negativo»", "80/150 · sensibilidad 0/70", "53.3 % · 0 %"]],
              [56 * mm, 64 * mm, CONTENT_W - 120 * mm - 18]),
        P("El modelo supera a la línea base en exactitud (83.3 % frente a 53.3 %). Si dejar escapar un positivo cuesta mucho más que una falsa alarma, importa la sensibilidad (78.6 %), no la exactitud.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook (A2) y el Bloque 2 del cuaderno",
                   [P("Calcula la matriz con <font name='Mono'>confusion_matrix(y, pred, labels=[0, 1])</font>, saca las métricas a mano y con <font name='Mono'>recall_score(..., pos_label=0)</font> para la especificidad.", "cell")]))

    s.append(h("3", "SVM: hiperplano, distancia y margen"))
    s += para("El modelo es <b>f(x) = w·x + b</b>; la clase es el signo de f(x). La distancia de un punto a la frontera es <b>|f(x)|/‖w‖</b> y el ancho total del margen es <b>2/‖w‖</b> (bordes en f = ±1). "
              "Para un caso con etiqueta y = ±1 se mira <b>y·f(x)</b>: ≥ 1 está bien y fuera del margen; entre 0 y 1, dentro del margen; &lt; 0, mal clasificado. La holgura es <b>ξ = máx(0, 1 − y·f(x))</b>, "
              "y la función objetivo ½‖w‖² + C·Σξ. Son vectores de soporte los casos con y·f(x) ≤ 1.")
    s.append(ejemplo("w = (3, −4), b = 2", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Norma y ancho del margen", "‖w‖ = √(9 + 16) = 5 · 2/5", "‖w‖ = 5 · 0.4"],
               ["x = (2, 1): puntaje y clase", "3·2 − 4·1 + 2", "f = 4 → +1"],
               ["Distancia a la frontera", "|4|/5", "0.8"],
               ["x = (−1, 2)", "−3 − 8 + 2 = −9 · |−9|/5", "−1 · 1.8"],
               ["Holguras con y·f = 1.6, 0.4 y −0.5", "máx(0, 1 − y·f)", "ξ = 0 · 0.6 · 1.5"],
               ["Costo de holgura con C = 0.1 y C = 10", "Σξ = 2.1: 0.1·2.1 · 10·2.1", "0.21 · 21"]],
              [60 * mm, 66 * mm, CONTENT_W - 126 * mm - 18]),
        P("Los tres casos de holgura son: correcto fuera del margen, correcto dentro del margen (0.4 &lt; 1) y mal clasificado (−0.5 &lt; 0). Los dos últimos son vectores de soporte.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c37_margen.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte B del notebook (B1) y el Bloque 3 del cuaderno",
                   [P("Ajusta <font name='Mono'>SVC(kernel='linear')</font>, lee <font name='Mono'>coef_</font>, <font name='Mono'>intercept_</font> y <font name='Mono'>decision_function</font>, y comprueba a mano w·x + b y 2/‖w‖.", "cell")]))

    s.append(h("4", "C, kernel y gamma"))
    s += para("<b>C</b> pondera las holguras: C pequeño tolera más violaciones (margen ancho, más regularización); C grande las castiga (margen estrecho, riesgo de sobreajuste). "
              "El <b>kernel</b> define la forma de la frontera: lineal, polinomial o radial K = exp(−γ‖x − z‖²). Con γ pequeño la similitud decae lento (frontera casi lineal); con γ grande es local y la frontera memoriza. "
              "Se necesita validar: «C o γ más altos» no siempre mejoran.")
    s.append(ejemplo("holguras y similitud con otros valores", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Σξ = 5: contribución con C = 0.1 y C = 10", "0.1·5 · 10·5", "0.5 · 50"],
               ["Similitud RBF a distancia 3 con γ = 0.05", "e<super>−0.05·9</super> = e<super>−0.45</super>", "0.638"],
               ["Misma distancia con γ = 0.5", "e<super>−0.5·9</super> = e<super>−4.5</super>", "0.011"],
               ["Ajustes en una rejilla de 4 C × 3 γ con 5 folds", "4·3·5", "60 (más 1 reajuste)"]],
              [66 * mm, 62 * mm, CONTENT_W - 128 * mm - 18]),
        P("Con γ = 0.5 dos puntos a distancia 3 casi no se «ven» (0.011): la frontera se vuelve muy local. Con γ = 0.05 todavía se parecen (0.638): frontera más suave.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c37_C.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(img("c37_gamma.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte C del notebook y el Bloque 5 del cuaderno",
                   [P("Dibuja las fronteras para tres valores de C y de γ con dos variables y mira las curvas de entrenamiento frente a validación cruzada para elegir un valor intermedio.", "cell")]))

    s.append(h("5", "Estandarizar, validación cruzada y búsqueda en rejilla"))
    s += para("Se estandariza con la <b>media y la sd del entrenamiento</b> (z = (x − media)/sd) y la misma transformación se aplica a prueba y a casos nuevos. Dentro de la validación cruzada de k folds, la escala se aprende solo con los k − 1 folds de ajuste "
              "(un <font name='Mono'>Pipeline</font>). <font name='Mono'>GridSearchCV</font> prueba todas las combinaciones de C y γ, promedia la exactitud entre folds y reajusta la mejor con todo el entrenamiento.")
    s.append(ejemplo("tres combinaciones en 5 folds", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Radio 21 con media 14.0 y sd 3.5", "(21 − 14.0)/3.5", "z = 2.0"],
               ["Exactitud por fold de (C = 1, γ = 0.1): 0.94, 0.92, 0.95, 0.93, 0.96", "(0.94 + 0.92 + 0.95 + 0.93 + 0.96)/5", "media 0.940"],
               ["Otra combinación: 0.90, 0.95, 0.97, 0.92, 0.93", "suma 4.67/5", "media 0.934"],
               ["¿Cuál se elige?", "mayor media en validación; la prueba no participa", "la primera (0.940)"]],
              [66 * mm, 62 * mm, CONTENT_W - 128 * mm - 18]),
        P("Las diferencias de 0.6 puntos son menores que la variación entre folds; en un caso real se puede preferir la combinación más simple. Lo que no se hace es decidir con la prueba.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c37_escala.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(img("c37_rejilla.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook y los Bloques 6 y 7 del cuaderno",
                   [P("Ajusta un <font name='Mono'>Pipeline</font> con <font name='Mono'>GridSearchCV</font> solo en entrenamiento, mira <font name='Mono'>cv_results_</font> y evalúa el mejor una sola vez en prueba.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(recuerda(["Probabilidad con la logística (o <font name='Mono'>probability=True</font>, calibrada); puntaje con <font name='Mono'>decision_function</font>: no son lo mismo.",
                       "R ↔ Python: <font name='Mono'>glm(family = binomial)</font> ↔ <font name='Mono'>LogisticRegression(penalty=None)</font>; <font name='Mono'>cost</font> ↔ <font name='Mono'>C</font>; <font name='Mono'>'radial'</font> ↔ <font name='Mono'>'rbf'</font>; <font name='Mono'>scale = TRUE</font> ↔ <font name='Mono'>StandardScaler</font>.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–C del notebook → Bloques 5–7 → Parte D → ejercicio de cierre."]))
    build("apuntes/C37_Apuntes_SVM.pdf", s, "Apuntes · Clase 37 · SVM y clasificadores clásicos")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0284C7">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C37))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 37 · SVM y clasificadores clásicos",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C37))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: p = 1/(1 + e<super>−η</super>) · odds = e<super>η</super> · exactitud = (VP+VN)/n · sensibilidad = VP/(VP+FN) · especificidad = VN/(VN+FP) · precisión = VP/(VP+FP) · "
                           "F1 = 2VP/(2VP+FP+FN) · f(x) = w·x + b · distancia = |f|/‖w‖ · ancho = 2/‖w‖ · ξ = máx(0, 1 − y·f) · K = exp(−γ d²).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."],
                          "cell", C37)], bg=SOFT, border=C37))
    s.append(Spacer(1, 2))
    s += bloque(1, "Probabilidad logística y umbral", "★★", "Modelo con variables estandarizadas: η = −1.5 + 0.7·z₁ + 1.2·z₂. Caso A: z₁ = 1, z₂ = 0.5. Caso B: z₁ = −1, z₂ = −0.5.", [
        "Calcula η y la probabilidad p de A y de B.",
        "¿Por cuánto se multiplican los odds al subir z₁ en una unidad? ¿Y z₂?",
        "Indica la clase de cada caso con t = 0.5 y con t = 0.4. ¿Qué cambió y qué no?"])
    s += bloque(2, "Matriz de confusión y costos", "★★★", "Prueba de 200 casos (100 positivos). Modelo A: VN 88, FP 12, FN 20, VP 80. Modelo B: VN 95, FP 5, FN 35, VP 65.", [
        "Calcula exactitud, sensibilidad, especificidad, precisión y F1 de A y de B.",
        "Compara con la línea base «siempre negativo». ¿Qué modelo tiene mejor exactitud y cuál mejor sensibilidad?",
        "Un FN cuesta 10 y un FP cuesta 1. Calcula el costo total de cada modelo y elige."])
    s += bloque(3, "Hiperplano, distancia y margen", "★★", "w = (2, −1), b = −3. Puntos: P1 = (4, 1), P2 = (1, 2), P3 = (2.5, 1.5).", [
        "Calcula f(x) y la clase de cada punto.",
        "Calcula ‖w‖, la distancia de cada punto a la frontera y el ancho del margen 2/‖w‖.",
        "Si la etiqueta real de P3 es +1, ¿cuánto vale y·f? ¿Está dentro del margen? ¿Cuál es su holgura ξ?"])
    s += bloque(4, "Vectores de soporte y holguras", "★★", "Seis casos de entrenamiento con valores de y·f(x): 2.3, 1.0, 0.4, 1.0, −0.2 y 3.1. Se usa C = 2.", [
        "Calcula la holgura ξ de cada caso.",
        "¿Qué casos son vectores de soporte? ¿Cuáles están mal clasificados?",
        "Calcula C·Σξ y la nueva contribución con C = 0.5. ¿Qué pasa con la tolerancia a violaciones al bajar C?"])
    s += bloque(5, "Probabilidad y odds desde los coeficientes", "★★★", "Coeficientes de glm (unidades originales): intercepto −42.2511; radio 1.3399; textura 0.4421; suavidad 124.9794; concavidad 21.4077. Caso: radio 16, textura 18, suavidad 0.09, concavidad 0.05.", [
        "Calcula η y p. ¿Qué clase predice con t = 0.5?",
        "¿Por cuánto se multiplican los odds al subir la textura en 5 unidades? ¿Y la concavidad en 0.1?",
        "Si la textura sube de 18 a 23, calcula la nueva probabilidad y el cociente de odds. ¿Coincide con e<super>5·0.4421</super>?"])
    s += bloque(6, "Efecto de C y de γ", "★★", "Σξ = 1.8. Dos puntos a distancia ‖x − z‖ = 1.5 (d² = 2.25).", [
        "Calcula C·Σξ con C = 0.5 y con C = 2. Ordena de menos a más tolerante a violaciones.",
        "Calcula la similitud RBF con γ = 0.02 y con γ = 2. ¿Con cuál la frontera es más local?",
        "Un modelo con γ = 50 tiene 100 % de exactitud en entrenamiento y 63 % en validación cruzada. Diagnostica y propón una acción."])
    s += bloque(7, "Estandarizar, validar y elegir", "★★", "Radio con media 14.0 y sd 3.5 (calculadas con el entrenamiento). Rejilla: 4 valores de C × 3 de γ, validación cruzada de 5 folds.", [
        "Estandariza un radio de 21 y uno de 11.2. Si en la prueba la media fuera 15.1 y se usara esa media, ¿qué error se comete?",
        "¿Cuántos ajustes hace la búsqueda, sin contar el reajuste final?",
        "El mejor par (C, γ) por validación tiene validación 0.94 y prueba 0.92; otro tiene validación 0.93 y prueba 0.95. ¿Cuál eliges y por qué?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 4))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 3)]

    s += sol("Bloque 1", [
        ["1", "A: η = −1.5 + 0.7 + 0.6 = −0.2 → p = 1/(1 + e<super>0.2</super>) = 0.450. B: η = −1.5 − 0.7 − 0.6 = −2.8 → p = 1/(1 + e<super>2.8</super>) = 0.057."],
        ["2", "e<super>0.7</super> = 2.01 (odds × 2.01 por unidad de z₁); e<super>1.2</super> = 3.32 (odds × 3.32 por unidad de z₂)."],
        ["3", "t = 0.5: A → 0 (0.450 &lt; 0.5) y B → 0. t = 0.4: A → 1 (0.450 ≥ 0.4) y B → 0. Cambió la clase de A; las probabilidades (0.450 y 0.057) no cambiaron."]])
    s += sol("Bloque 2", [
        ["1", "A: exactitud 168/200 = 84.0 %; sensibilidad 80/100 = 80.0 %; especificidad 88/100 = 88.0 %; precisión 80/92 = 87.0 %; F1 = 160/(160 + 12 + 20) = 0.833. "
              "B: exactitud 160/200 = 80.0 %; sensibilidad 65 %; especificidad 95 %; precisión 65/70 = 92.9 %; F1 = 130/(130 + 5 + 35) = 0.765."],
        ["2", "Línea base: exactitud 100/200 = 50 %, sensibilidad 0 %. A tiene la mejor exactitud (84 % frente a 80 %) y la mejor sensibilidad (80 % frente a 65 %); B tiene mejor especificidad y precisión."],
        ["3", "A: 20·10 + 12·1 = 212. B: 35·10 + 5·1 = 355. Se elige A: deja escapar menos positivos."]])
    s += sol("Bloque 3", [
        ["1", "P1: f = 8 − 1 − 3 = 4 → +1. P2: f = 2 − 2 − 3 = −3 → −1. P3: f = 5 − 1.5 − 3 = 0.5 → +1."],
        ["2", "‖w‖ = √5 = 2.236. Distancias: P1 4/2.236 = 1.789; P2 3/2.236 = 1.342; P3 0.5/2.236 = 0.224. Ancho del margen = 2/2.236 = 0.894."],
        ["3", "y·f = 1·0.5 = 0.5: está dentro del margen (0 &lt; 0.5 &lt; 1), bien clasificado. ξ = máx(0, 1 − 0.5) = 0.5."]])
    s += sol("Bloque 4", [
        ["1", "ξ = máx(0, 1 − y·f): 0, 0, 0.6, 0, 1.2, 0."],
        ["2", "Vectores de soporte: los de y·f ≤ 1, es decir, los casos 2, 3, 4 y 5 (1.0, 0.4, 1.0, −0.2). Mal clasificado: solo el 5 (−0.2 &lt; 0)."],
        ["3", "Σξ = 1.8: C·Σξ = 2·1.8 = 3.6; con C = 0.5, 0.9. Al bajar C las violaciones «cuestan» menos: el modelo las tolera más (margen más ancho)."]])
    s += sol("Bloque 5", [
        ["1", "η = −42.2511 + 1.3399·16 + 0.4421·18 + 124.9794·0.09 + 21.4077·0.05 = −42.2511 + 21.4384 + 7.9578 + 11.2481 + 1.0704 = −0.5364. p = 1/(1 + e<super>0.5364</super>) = 0.369. Clase 0."],
        ["2", "Textura +5: e<super>5·0.4421</super> = e<super>2.2105</super> = 9.12. Concavidad +0.1: e<super>2.14077</super> = 8.51."],
        ["3", "Nuevo η = −0.5364 + 5·0.4421 = 1.674 → p = 0.842. Cociente de odds = (0.842/0.158)/(0.369/0.631) = 5.33/0.585 = 9.12 = e<super>2.2105</super>: coincide. El odds se multiplica por 9.12 pero la probabilidad pasó de 0.369 a 0.842, no se multiplica por 9.12."]])
    s += sol("Bloque 6", [
        ["1", "C = 0.5: 0.5·1.8 = 0.9. C = 2: 2·1.8 = 3.6. De menos a más tolerante a violaciones: C = 2 (penaliza más) y luego C = 0.5."],
        ["2", "γ = 0.02: e<super>−0.02·2.25</super> = e<super>−0.045</super> = 0.956. γ = 2: e<super>−2·2.25</super> = e<super>−4.5</super> = 0.011. Con γ = 2 es más local (la similitud casi desaparece a esa distancia)."],
        ["3", "Sobreajuste: la brecha entrenamiento–validación es de 37 puntos. Acciones: bajar γ (y/o C) y elegir ambos con validación cruzada solo en entrenamiento; no mirar la prueba."]])
    s += sol("Bloque 7", [
        ["1", "z = (21 − 14.0)/3.5 = 2.0; z = (11.2 − 14.0)/3.5 = −0.8. Usar la media de la prueba (15.1) da z = 1.69 y −1.11: es fuga de información (se usan datos de prueba) y además cada conjunto queda en otra escala. "
              "Siempre se usa la media y sd del entrenamiento."],
        ["2", "4·3·5 = 60 ajustes (más un reajuste final con todo el entrenamiento)."],
        ["3", "Se elige el de mayor validación (0.94): la prueba no participa en la elección. La diferencia de 0.01 está dentro del ruido y la prueba se mira una vez, después de decidir."]])
    build("apuntes/C37_Ejercicios_Cuaderno_SVM.pdf", s, "Ejercicios para el cuaderno · Clase 37")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
