"""Clase 35 · Bagging y Random Forest: guía visual, apuntes y cuaderno de ejercicios en papel."""
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from construir_glm import code, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

C35 = colors.HexColor("#0D9488")
C35_BG = colors.HexColor("#F0FDFA")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción de la clase + diapositivas + notebooks Python del profesor"


def topic_page(s, titulo, idea, conceptos, figura, pie, formulas, py, salida, cuidado, fig_w=0.97):
    s.append(band(f"Clase 35 · {titulo}", FUENTE, C35))
    s.append(Spacer(1, 7))
    s.append(box([P(idea, "big")], bg=SOFT, border=C35))
    s.append(Spacer(1, 6))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos, C35))
    s.append(Spacer(1, 4))
    if figura:
        s.append(img(figura, width=CONTENT_W * fig_w))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 5))
    left = [P("Fórmulas y reglas", "h3"), box([P(f, "formula") for f in formulas], bg=SOFT, width=HALF),
            Spacer(1, 5), P("En Python", "h3"), box([P(f, "mono") for f in py], bg=SOFT, width=HALF)]
    right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C35), Spacer(1, 5),
             box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER,
                 width=HALF)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 35 · Bagging y Random Forest",
                  "Profesor Bastian Galasso · Etapa 7 del roadmap · continúa Árboles de decisión (C34) y precede a Boosting (C36)", NAVY))
    s.append(Spacer(1, 8))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Un árbol profundo es <b>inestable</b>: si cambian unas pocas filas, cambian sus preguntas. La solución es entrenar "
                    "<b>muchos árboles, cada uno con una muestra distinta</b> (bootstrap: muestras con reemplazo) y <b>agregar</b> sus "
                    "predicciones por voto o promedio: eso es <b>bagging</b>. <b>Random Forest</b> agrega un segundo azar: en cada nodo "
                    "solo se evalúa un subconjunto de variables, así los árboles se parecen menos entre sí. «Si varios modelos cometen "
                    "errores distintos y luego votan, el error que depende de una muestra particular se reduce.»", "body")],
                 bg=C35_BG, border=C35))
    s.append(Spacer(1, 6))
    s.append(P("Bagging y Random Forest en una imagen", "h2"))
    s.append(img("c35_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 4))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["Árboles de decisión: preguntas, profundidad, sobreajuste (Clase 34)",
                     "Validación cruzada, AUC, exactitud y RMSE (Clases 30–33)",
                     "Varianza y correlación de variables aleatorias (Etapa 0)",
                     "Pipeline con imputación y one-hot, sin filtrar información al test"], "cell", C35)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Inestabilidad</b> → <b>bootstrap y OOB</b> → <b>bagging</b> → <b>varianza del promedio</b> → "
                      "<b>Random Forest</b> (<font name='Mono'>max_features</font>) → hiperparámetros → límites y usos",
                      "<b>Heart</b> (303 pacientes, clasificación AHD) y <b>Ames Housing</b> (2.930 casas, regresión del precio)",
                      "Ejemplo de imágenes: el profesor usa CIFAR-10; aquí, los dígitos 8×8 de scikit-learn (sin descargas)",
                      "Pregunta guía: ¿cómo reducir la inestabilidad de un árbol sin perder su capacidad no lineal?"],
                     "cell", C35)]
    s.append(two_cols(left, right))
    s.append(PageBreak())

    topic_page(
        s, "Árbol inestable, ensamblado y bootstrap",
        "Un ensamblado combina varios modelos base en una predicción. Para que sirva, los modelos deben ser <b>distintos</b>; "
        "el bootstrap fabrica muestras distintas con los mismos datos, sacando <b>n filas con reemplazo</b>.",
        [("Árboles inestables", "Con otra muestra cambian las variables, los umbrales y hasta la profundidad: bajo sesgo, <b>alta varianza</b>. Con 200 muestras bootstrap de Heart la raíz fue Thal en 48.5 %, ChestPain en 30 % y Ca en 20 %."),
         ("Ensamblar", "Se entrena un modelo por muestra y una función de agregación entrega la predicción final: voto o promedio de probabilidades en clasificación, promedio en regresión."),
         ("Bootstrap", "Se saca una fila, se anota, <b>se devuelve</b> y se repite n veces. Las repeticiones crean conjuntos distintos sin recolectar datos nuevos; cada extracción tiene siempre la misma probabilidad."),
         ("63.2 % únicas, 36.8 % fuera", "Una fila queda fuera con (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368. Esas filas son el <b>OOB</b> (<i>out of bag</i>) de ese árbol."),
         ("OOB: validación gratis", "Cada fila se predice solo con los árboles que no la vieron: sirve para estimar el error sin reservar otro conjunto. <b>No reemplaza</b> al test final.")],
        "c35_bootstrap.png",
        "Izquierda: fracción OOB teórica. Centro: veces que cada fila entra a una muestra (simulado con Heart, n = 227). Derecha: la raíz de 200 árboles bootstrap.",
        ["P(fila fuera) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368",
         "filas únicas esperadas ≈ n·(1 − e⁻¹) ≈ 0.632·n",
         "veces que entra una fila ~ Poisson(1): 0 → 36.8 %, 1 → 36.8 %, 2 → 18.4 %",
         "bagging: ŷ = moda de h₁…h_B (o promedio de probabilidades)"],
        ["BaggingClassifier(DecisionTreeClassifier(),",
         "    n_estimators=120, bootstrap=True,",
         "    oob_score=True, n_jobs=-1)",
         "rng.integers(0, n, n)   # bootstrap a mano",
         "modelo.oob_score_"],
        ["n = 8: 34.4 % fuera · n = 227 (Heart): 36.7 % · n = 2.344 (Ames): 36.8 %",
         "Simulado con 2.000 muestras: 36.8 % fuera de media (desvío 2.0 puntos)",
         "Dos árboles bootstrap cualquiera discrepan en 26 % de los pacientes del test",
         "Un árbol suelto: AUC ≈ 0.73; 100 árboles promediados: ≈ 0.92 (notebook A3)"],
        ["En clase se dijo «500 de 1.000 registros» por muestra: por defecto son <b>n</b> extracciones (≈ 632 filas distintas). "
         "Con <font name='Mono'>max_samples</font> se cambia. Las muestras se parecen entre sí: no son independientes."])

    topic_page(
        s, "Bagging y por qué reduce la varianza",
        "Bagging = <i>bootstrap aggregating</i>: entrena el mismo tipo de modelo en muestras bootstrap distintas y agrega. "
        "Reduce la varianza porque promediar suaviza errores individuales, <b>si los árboles no son idénticos</b>.",
        [("Algoritmo de 4 pasos", "1) crear B muestras bootstrap · 2) ajustar un árbol en cada una · 3) obtener B respuestas para x · 4) votar o promediar."),
         ("Entrenamiento en paralelo", "Cada árbol solo ve su muestra: se reparten entre procesadores (<font name='Mono'>n_jobs</font>). En boosting (Clase 36) no se puede: el árbol 2 necesita al 1."),
         ("Varianza del promedio", "Con varianza σ² por árbol y correlación ρ entre árboles: Var = ρσ² + (1 − ρ)σ²/B. Más árboles borran el segundo término; <b>el primero queda</b>."),
         ("ρ alto = no se gana", "Si todos los árboles se parecen, repiten los mismos errores. Señal en la práctica: un bagging que casi no mejora a un solo árbol."),
         ("Cualquier modelo base", "Se puede hacer bagging de una regresión, pero ayuda solo a modelos <b>inestables</b> (árboles profundos), no a los estables.")],
        "c35_varianza.png",
        "Ames, validación. Se repitió K = 60 veces el entrenamiento con otra submuestra. Los puntos (medidos) caen sobre la fórmula: ρ̂ = 0.05 (bagging), 0.02 (RF 1/3), 0.01 (RF 10 %).",
        ["Var(promedio de B) = ρσ² + (1 − ρ)σ²/B",
         "B → ∞: Var → ρσ² (el piso que no se cruza)",
         "ρ = 1: Var = σ² · ρ = 0: Var = σ²/B",
         "supone varianzas iguales y ρ común (nota de la diapositiva 9)"],
        ["BaggingRegressor(DecisionTreeRegressor(),",
         "    n_estimators=300, n_jobs=-1)",
         "RandomForestRegressor(n_estimators=300,",
         "    n_jobs=-1, random_state=42)",
         "np.mean([a.predict(X) for a in arboles], 0)"],
        ["σ (un árbol) ≈ 36.6 mil USD; con B = 100 el promedio de bagging baja a ≈ 8.8 mil (piso √(ρσ²) ≈ 8.1 mil)",
         "Fórmula y medición coinciden casi al dólar (ej.: B = 10 → 13.9 mil medido, 13.9 mil fórmula)",
         "Ames, 300 árboles: un árbol 36.1k · bagging 24.4k · Random Forest (1/3) 23.1k (RMSE en validación)",
         "Heart: bagging de árboles AUC 0.893 (CV 5×3) frente a 0.753 de un árbol"],
        ["En clase se dijo «ρ es la correlación entre errores y no se mide». Es la correlación entre las <b>predicciones</b> de dos árboles bajo el azar de datos y "
         "algoritmo, y sí se estima (notebook B2)."])

    topic_page(
        s, "Random Forest y max_features",
        "Random Forest = bagging de árboles + <b>selección aleatoria de variables en cada nodo</b>. Si una variable domina, no puede "
        "mandar en todos los árboles: los árboles se parecen menos (ρ baja) y el promedio mejora.",
        [("Dos fuentes de azar", "Filas: cada árbol recibe una muestra bootstrap. Variables: en cada <b>nodo</b> solo se evalúan <i>mtry</i> = <font name='Mono'>max_features</font> candidatas al azar entre las p disponibles."),
         ("Por nodo, no por árbol", "El mejor corte se busca <b>dentro del subconjunto</b> sorteado en ese nodo; en el siguiente nodo se sortea otro. (La nota de la diapositiva 10 lo dice; en clase se explicó por árbol.)"),
         ("Qué se compra y qué cuesta", "Menos variables por nodo = menor ρ, pero cada árbol es algo peor (σ² mayor, algo más de sesgo). El óptimo es intermedio y se <b>valida</b>."),
         ("Valores de partida", "Clasificación: <font name='Mono'>'sqrt'</font>. Regresión: probar 0.1–0.5 (el RF de regresión de scikit-learn usa 1.0 por defecto, es decir, <b>un bagging</b>)."),
         ("Bagging vs Random Forest", "Mismo bootstrap, misma agregación y OOB en ambos; la diferencia clave es <font name='Mono'>max_features</font>. «Un parámetro para controlar la similitud entre árboles.»")],
        "c35_max_features.png",
        "Ames (izquierda): RMSE de validación y OOB al variar max_features con 300 árboles. Heart (derecha): AUC en el train, OOB y validación cruzada 5×3.",
        ["candidatas por nodo = max_features",
         "'sqrt' → √p · 'log2' → log₂p · fracción → ⌊f·p⌋",
         "P(una variable dada es candidata en un nodo) = m/p",
         "m = p ⇒ Random Forest = bagging"],
        ["RandomForestClassifier(n_estimators=300,",
         "    max_features='sqrt', oob_score=True)",
         "RandomForestRegressor(max_features=0.2)",
         "# por árbol (otra cosa): BaggingRegressor(",
         "#     max_features=0.1)"],
        ["Ames (341 columnas tras el one-hot): 1 % de columnas → RMSE 27.7k · 10 % → 22.5k (mínimo) · 1/3 → 23.1k · todas (bagging) → 24.3k",
         "Mismo 10 %: por nodo (RF) 22.5k vs por árbol (random subspace) 27.1k",
         "Heart (18 columnas): m = 1 → AUC 0.905 · m = 3 → 0.902 · m = 18 → 0.882; desvío ≈ 0.03: curva casi plana",
         "Con el one-hot, 13 variables son 18 columnas: √18 ≈ 4 (en clase: √13 ≈ 3)"],
        ["«Con max_features = 1 salen modelos muchísimo mejores» (clase): en Heart es indistinguible; en Ames es lo <b>peor</b> del barrido. "
         "Se valida, no se recita."])

    topic_page(
        s, "OOB, número de árboles e hiperparámetros",
        "Más árboles estabilizan el resultado y cuestan tiempo y memoria; el OOB estima el error con <b>un solo ajuste</b>; "
        "la complejidad de cada árbol y el muestreo se afinan con validación cruzada, nunca mirando el test.",
        [("n_estimators", "Más árboles bajan el término (1 − ρ)σ²/B y estabilizan, <b>sin sobreajustar</b>; el costo sube casi lineal. La mejora se aplana pronto."),
         ("max_features", "Menos variables = menor correlación y quizá algo más de sesgo. Se prioriza junto con n_estimators y el tamaño mínimo de hoja."),
         ("max_depth / min_samples_leaf", "Controlan la complejidad de cada árbol. Con hojas de 1 observación el bosque memoriza el train (exactitud 1.000) pero igual generaliza."),
         ("bootstrap / max_samples / class_weight", "Definen si hay reemplazo y cuántas filas ve cada árbol; <font name='Mono'>class_weight</font> ayuda con una clase rara (p. ej. 2 % de positivos)."),
         ("Búsqueda y OOB", "<font name='Mono'>GridSearchCV</font> prueba una grilla; <font name='Mono'>RandomizedSearchCV</font>, n_iter combinaciones. El OOB sirve para comparar configuraciones; el test queda para el final.")],
        "c35_curvas.png",
        "Ames, validación (hold-out). Izquierda: error al agregar árboles. Derecha: el OOB (sin validación aparte) se estabiliza igual, pero más alto: usa ≈ 37 % de los árboles.",
        ["grilla: nº combinaciones × nº bloques = ajustes",
         "24 comb. × 5 bloques = 120 ajustes por modelo",
         "tiempo en paralelo ≥ tiempo serie / nº de núcleos",
         "OOB: cada fila usa ≈ 0.368·B árboles"],
        ["RandomForestClassifier(oob_score=True)",
         "rf.oob_score_ · rf.oob_decision_function_",
         "GridSearchCV(pipe, grilla, cv=5,",
         "    scoring='roc_auc', n_jobs=-1)",
         "RandomizedSearchCV(pipe, dist, n_iter=12)"],
        ["Ames: a las 34 árboles el RF queda a menos de 2 % de su valor final (RMSE 23.1k con 100 y con 300 árboles)",
         "OOB vs CV vs test: Heart 0.811 · 0.798 · 0.855 (exactitud); Ames 26.0k · 26.3k · 27.0k. OOB: 1 ajuste; CV: 5",
         "Heart: RF base AUC CV 0.899; tras Grid o Random, ≈ 0.911: la búsqueda ayuda ≈ 0.01",
         "n_jobs = −1 acelera ≈ 6.6× con 16 núcleos lógicos (no 16×)"],
        ["El OOB es levemente pesimista con pocos árboles y <b>no reemplaza</b> al test independiente. Ajustar mirando el test repetidamente destruye su validez. "
         "Incluye tu configuración base en la grilla."])

    topic_page(
        s, "Importancia, límites y aplicaciones",
        "Rinde en <b>datos tabulares</b> no lineales, sin escalar variables; pierde interpretabilidad, no extrapola en regresión y no ve estructura espacial. "
        "La importancia por impureza puede engañar: prefiere la de permutación.",
        [("Ventajas", "Buen desempeño sin suponer linealidad, menos varianza que un árbol, interacciones y umbrales automáticos, poco sensible a la escala, OOB incluido."),
         ("Limitaciones", "Menos interpretable, más memoria y tiempo de predicción, <b>no extrapola</b> fuera del rango visto y pide variables numéricas (codificar las categóricas)."),
         ("Importancia por impureza (MDI)", "Se calcula en el <b>train</b> y favorece variables con muchos cortes posibles, aunque sean ruido. Mejor: <b>permutación</b> sobre datos no vistos. Ninguna es causalidad."),
         ("Flujo recomendado", "1) separar test · 2) elegir la métrica (clases raras: recall, F1, PR-AUC) · 3) ajustar con CV u OOB · 4) evaluar una vez y explicar."),
         ("Dónde encaja", "Riesgo y fraude, salud, operaciones, marketing. En imágenes, texto o series muy largas: modelos especializados.")],
        "c35_importancia.png",
        "Heart con dos columnas de ruido puro agregadas (rojo): la impureza (train) premia al ruido continuo; la permutación (test) lo deja en ≈ 0. A la derecha, las dos medidas en Ames.",
        ["MDI = Σ (N_t/N)·ΔGini en los nodos de la variable",
         "perm = métrica_base − métrica_con_la_variable_desordenada",
         "RF regresión: ŷ queda entre el mín y el máx de y (train)"],
        ["permutation_importance(pipe, X_test, y_test,",
         "    scoring='roc_auc', n_repeats=30)",
         "rf.feature_importances_   # MDI (train)",
         "from sklearn.metrics import recall_score,",
         "    f1_score, average_precision_score"],
        ["Ruido continuo: puesto 7 de 15 por impureza (6.9 %, sobre Chol, RestBP, Sex…) puesto 10 por permutación (≈ 0.001 de AUC)",
         "Ames: Gr_Liv_Area vale 22.5 % por permutación (+11.9 mil USD de RMSE) y 12.2 % por impureza",
         "Entrenado con casas &lt; 1.800 pies², el bosque no pasa de ≈ 158 mil USD en las &gt; 2.400 pies²"],
        ["Con 76 pacientes y 5 repeticiones (notebook del profesor) la permutación es ruidosa. En clase se dijo que la impureza engaña «porque hay menos variables»: "
         "la causa es el sesgo hacia variables con muchos cortes."], fig_w=0.9)

    # ------------------------------------------------------------ chuleta
    s.append(band("Chuleta, hiperparámetros y checklist", "Lo que vas a ajustar en cualquier Random Forest", NAVY))
    s.append(Spacer(1, 8))
    s.append(table([["Hiperparámetro", "Qué controla", "Efecto si lo subes", "Valor de partida"],
                    ["n_estimators", "Número de árboles (B)", "estabiliza; más tiempo y memoria; se aplana pronto", "300–500"],
                    ["max_features", "Variables candidatas por nodo", "mayor ρ; menos sesgo individual (1.0 = bagging)", "'sqrt' (clasif.); 0.1–0.5 (regr.)"],
                    ["max_depth", "Profundidad máxima de cada árbol", "más complejidad; riesgo de memorizar", "None o validar 6–20"],
                    ["min_samples_leaf", "Observaciones mínimas por hoja", "hojas y predicciones más suaves", "1–5"],
                    ["bootstrap · max_samples", "Muestreo de filas (con reemplazo, fracción)", "max_samples menor: más rápido y diverso", "True · None (n filas)"],
                    ["class_weight", "Peso de cada clase", "da más peso a la clase rara", "None; 'balanced' si hay desbalance"],
                    ["oob_score · n_jobs", "Error interno · paralelismo", "OOB gratis · acelera (no linealmente)", "True · −1"]],
                   [30 * mm, 48 * mm, 62 * mm, CONTENT_W - 140 * mm]))
    s.append(Spacer(1, 3))
    s.append(P("Hiperparámetros: diapositiva 14 (nombres de la API de scikit-learn). Los valores de partida son orientativos; el profesor pide validarlos "
               "con CV u OOB y una métrica alineada con el problema.", "small"))
    s.append(Spacer(1, 6))
    s.append(table([["Modelo", "Datos de entrenamiento", "Varianza", "Interpretable", "Paralelizable"],
                    ["Árbol de decisión", "Todos", "Alta", "Sí", "—"],
                    ["Bagging de árboles", "Muestras bootstrap", "Baja", "No", "Sí, entre árboles"],
                    ["Random Forest", "Bootstrap + variables al azar por nodo", "Más baja (menor ρ)", "No", "Sí, entre árboles"],
                    ["Boosting (Clase 36)", "Todos, con pesos o residuos", "—, baja sesgo", "No", "No entre árboles"]],
                   [34 * mm, 58 * mm, 32 * mm, 26 * mm, CONTENT_W - 150 * mm]))
    s.append(Spacer(1, 6))
    s.append(P("Checklist para entrenar un Random Forest", "h2"))
    checks = [("Parte train/test (estratificado si es clasificación)", "en regresión no se estratifica"),
              ("Preprocesa dentro del Pipeline", "imputación y one-hot se aprenden solo con el train; el RF necesita variables numéricas"),
              ("Elige la métrica antes de ajustar", "con clases raras no uses exactitud"),
              ("Compara con un árbol y con un bagging", "para ver cuánto aportan max_features y el promedio"),
              ("Valida max_features, n_estimators y min_samples_leaf", "CV u OOB; el test no se toca"),
              ("Mira train vs OOB/CV vs test", "una brecha grande no es mala si el error de test es el menor"),
              ("Interpreta con permutación sobre datos no vistos", "la impureza (MDI) puede engañar"),
              ("Reporta media ± desvío", "con 76 pacientes de test una sola partición no basta")]
    s.append(table([["", "Paso", "Cómo sé que lo hice bien"]] + [[SYM, a, b] for a, b in checks],
                   [8 * mm, 80 * mm, CONTENT_W - 88 * mm]))
    s.append(PageBreak())

    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales, preguntas clave y cuándo pasar a la siguiente clase", NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Diapositivas y clase (verificados con código)</b>", "cellb"),
                  bullets(["<b>«Cada árbol toma una muestra de las variables (la 1, 5 y 8; el otro la 7, 15 y 20)»</b> (clase): la selección es <b>por nodo</b>, como dice la nota de la diapositiva 10. "
                           "Por árbol sería <i>random subspace</i> y rinde mucho peor (27.1k vs 22.5k de RMSE).",
                           "<b>«De 1.000 registros se toman 500»</b>: la muestra bootstrap tiene n extracciones por defecto. <b>«Las muestras son independientes»</b>: se parecen entre sí.",
                           "<b>«max_features = 1 da modelos muchísimo mejores»</b>: en Heart es indistinguible; en Ames es lo peor. <b>«Casi elimina el sobreajuste»</b>: con hojas de 1 el bosque llega a exactitud 1.000 en el train.",
                           "<b>«Con 10 procesadores, 10 veces menos tiempo»</b> es el límite ideal (real ≈ 6.6× con 16 núcleos). <b>«√13 ≈ 3 variables»</b>: con el one-hot son 18 columnas (√18 ≈ 4).",
                           "<b>«Los bosques aceptan categóricas»</b>: en scikit-learn hay que codificarlas. <b>«Boosting pierde el OOB»</b>: con <font name='Mono'>subsample &lt; 1</font> existe <font name='Mono'>oob_improvement_</font>. <b>«AUC siempre &gt; 0.5»</b>: puede ser &lt; 0.5 si ordena al revés.",
                           "<b>Diapositiva 9</b>: ρ es la correlación entre <b>predicciones</b>, no «errores». <b>Diapositiva 13</b>: scikit-learn promedia probabilidades (≠ voto con hojas &gt; 1)."],
                          "cell", RED)], bg=RED_BG, border=RED))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Notebooks del profesor</b>", "cellb"),
                  bullets(["<font name='Mono'>RandomForestRegressor(max_features=1.0)</font> <b>es un bagging</b>: casi idéntico al <font name='Mono'>BaggingRegressor</font> (RMSE 29.1k vs 29.3k); con 1/3 baja a 26.7k.",
                           "<font name='Mono'>class_weight='balanced'</font> solo en el bosque; y en la búsqueda aleatoria el «Bagging» puede sortear <font name='Mono'>estimator__max_features</font>, lo que lo convierte en un Random Forest.",
                           "Permutación sobre 76 pacientes con 5 repeticiones: ruidosa. Ruta de datos fija y <font name='Mono'>FAST_MODE = False</font> (en clase se recomendó True).",
                           "<b>Imágenes</b> usa CIFAR-10 (descarga ≈ 170 MB) y <b>no se vio en clase</b>; aquí se reemplaza por los dígitos de scikit-learn. Rinde bien con píxeles reordenados al azar (0.973) pero cae a 0.504 si se desplaza la imagen 1 píxel: no entiende la posición."
                           ],
                          "cell", AMBER)], bg=AMBER_BG, border=AMBER))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("«busstrap/boost trap» (bootstrap), «Vallin/Bind/Baguí» (bagging), «random forks/Randolph Farris» (Random Forest), «obb/OVE/auto bag» (OOB), "
                    "«rock aoc» (ROC AUC), «aquira» (accuracy), «ricó» (recall), «min sample lift» (<font name='Mono'>min_samples_leaf</font>), «her.xlsx» (Heart.xlsx). "
                    "Bucle de Whisper en las líneas 231–252; comentarios ajenos en 319–329, 453 y 840–848. La clase termina antes de Ames y de las imágenes.", "cell")],
                 bg=SOFT, border=NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Preguntas de repaso", "h2"))
    preg = [("¿Por qué aparecen filas repetidas en el bootstrap?", "Se extrae con reemplazo; con n grande ≈ 63.2 % únicas y ≈ 36.8 % OOB."),
            ("¿Qué problema busca resolver max_features?", "Reducir la correlación ρ entre árboles, el piso ρσ² de la varianza.")            ]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(preg, 1)],
                   [7 * mm, 66 * mm, CONTENT_W - 73 * mm]))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Criterio de dominio para pasar a la siguiente clase</b>", "cellb"), Spacer(1, 2),
                  bullets(["Calculo a mano la probabilidad OOB (1 − 1/n)ⁿ y leo una muestra bootstrap",
                           "Aplico ρσ² + (1 − ρ)σ²/B y explico por qué B → ∞ no basta",
                           "Distingo bagging (todas las variables) de Random Forest (subconjunto por nodo)",
                           "Calculo a mano el OOB, un voto y una importancia por impureza",
                           "Elijo max_features, n_estimators y min_samples_leaf con CV u OOB, sin tocar el test",
                           "Resuelvo los bloques 1 a 9 del cuaderno con al menos 80% de aciertos"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C35_Guia_RandomForest.pdf", s, "Clase 35 · Bagging y Random Forest")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#0D9488">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 35 · Bagging y Random Forest", "Para leer antes de los ejercicios · lectura de ~45 min", C35))
    s.append(Spacer(1, 8))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 3),
                  numbered([
                      ("Árbol inestable", "Un árbol profundo cambia mucho con pequeños cambios en los datos: alta varianza."),
                      ("Bootstrap", "n filas con reemplazo: ≈ 63.2 % únicas y ≈ 36.8 % fuera (OOB, validación interna)."),
                      ("Bagging", "Un árbol por muestra bootstrap; se agrega por voto o promedio. Se entrena en paralelo."),
                      ("Varianza", "Var = ρσ² + (1 − ρ)σ²/B: más árboles no pueden cruzar el piso ρσ²."),
                      ("Random Forest", "Bagging + variables al azar en cada nodo (max_features): baja ρ."),
                      ("Validar, no recitar", "max_features, n_estimators y hojas se eligen con CV u OOB; el test queda para el final."),
                  ], C35)], bg=SOFT, border=NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una "
               "<b>pista</b> para el notebook <font name='Mono'>C35_RandomForest.ipynb</font> o el cuaderno "
               "<font name='Mono'>C35_Ejercicios_Cuaderno_RandomForest.pdf</font>.", "small"))

    s.append(h("1", "Bootstrap y fracción OOB"))
    s += para("Una muestra bootstrap saca <b>n filas con reemplazo</b>. Cada extracción deja fuera a una fila dada con probabilidad 1 − 1/n; tras n extracciones "
              "la fila sigue fuera con probabilidad <b>(1 − 1/n)ⁿ</b>, que converge a e⁻¹ ≈ 0.368. Esas filas son el <b>OOB</b> de ese árbol.")
    s.append(ejemplo("cuántas filas quedan fuera", [
        pasos([["Caso", "Cálculo", "Resultado"],
               ["n = 6", "(5/6)⁶", "0.335 (el ejemplo de la clase, n = 8: 0.344)"],
               ["n = 25", "(24/25)²⁵ = 0.96²⁵", "0.360"],
               ["n = 40: filas distintas esperadas", "40 · (1 − 0.975⁴⁰) = 40 · (1 − 0.363)", "25.5 (63.7 %)"],
               ["B = 10 árboles: filas sin ninguna predicción OOB", "0.632¹⁰", "1.0 % (con 30 árboles, ≈ 0.0001 %)"]],
              [58 * mm, 60 * mm, CONTENT_W - 118 * mm - 18]),
        P("Con n pequeño la fracción fuera es algo menor que 36.8 %, pero converge muy rápido: con n = 227 (Heart) es 36.7 %. "
          "Por eso hacen falta ≈ 30 árboles para que casi toda fila tenga alguna predicción OOB.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_bootstrap.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno",
                   [P("Escribe una muestra bootstrap de 8 números, tacha los que salieron y anota el OOB. Luego simula 2.000 muestras y compara con "
                      "(1 − 1/n)ⁿ. Recuerda: la muestra tiene <b>n</b> extracciones, no una fracción de n.", "cell")]))

    s.append(h("2", "Bagging: voto y promedio"))
    s += para("Bagging entrena un árbol por muestra bootstrap y agrega: en <b>clasificación</b>, la moda de los votos o el promedio de probabilidades; en "
              "<b>regresión</b>, el promedio. Los árboles no se comunican entre sí, así que se pueden entrenar en paralelo.")
    s.append(ejemplo("agregar cinco árboles", [
        pasos([["Árbol", "1", "2", "3", "4", "5", "Agregación"],
               ["P(Sí)", "0.80", "0.30", "0.70", "0.45", "0.20", "promedio = 0.49"],
               ["Voto (P &gt; 0.5)", "Sí", "No", "Sí", "No", "No", "2 Sí de 5 → 0.40"],
               ["Precio (miles USD)", "180", "210", "195", "240", "175", "promedio = 200 (mediana 195)"]],
              [30 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, CONTENT_W - 90 * mm - 18]),
        P("Con umbral 0.5 ambas reglas dicen No; con un umbral 0.45 la probabilidad media (0.49) dice <b>Sí</b> y el voto (0.40) dice No: no son lo mismo. "
          "scikit-learn promedia probabilidades; coincide con el voto solo si las hojas son puras (<font name='Mono'>min_samples_leaf=1</font>).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(pista("la parte A3 del notebook y el Bloque 3 del cuaderno",
                   [P("En el notebook programas el bagging desde cero (bootstrap + árboles + promedio + OOB a mano) y lo comparas con "
                      "<font name='Mono'>BaggingClassifier</font>.", "cell")]))

    s.append(h("3", "Por qué promediar reduce la varianza"))
    s += para("Si cada árbol tiene varianza σ² y dos árboles cualesquiera tienen correlación ρ, el promedio de B árboles tiene "
              "<b>Var = ρσ² + (1 − ρ)σ²/B</b>. Con B grande el segundo término desaparece, pero el primero queda: el <b>piso</b> ρσ².")
    s.append(ejemplo("σ² = 64 con dos correlaciones", [
        pasos([["B", "ρ = 0.5", "ρ = 0.1"],
               ["1", "32 + 32 = 64", "6.4 + 57.6 = 64"],
               ["4", "32 + 8 = 40", "6.4 + 14.4 = 20.8"],
               ["16", "32 + 2 = 34", "6.4 + 3.6 = 10"],
               ["∞", "32 (desvío 5.66)", "6.4 (desvío 2.53)"]],
              [20 * mm, 60 * mm, CONTENT_W - 80 * mm - 18]),
        P("Con ρ = 0.5 pasar de 16 a infinitos árboles solo baja de 34 a 32; con ρ = 0.1 el piso es cinco veces menor. "
          "Árboles que se parecen ya no aportan: hay que bajar ρ, no solo subir B.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_varianza.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s += para("En Ames se midieron σ y ρ repitiendo el entrenamiento con distintas submuestras: ρ̂ = 0.05 en el bagging y 0.02 en un Random Forest, y las varianzas "
              "medidas coinciden con la fórmula. La ρ de la fórmula es la correlación entre las <b>predicciones</b> de los árboles (no «de los errores»).")
    s.append(pista("la parte B2 del notebook y el Bloque 4 del cuaderno",
                   [P("Usa la fórmula con papel antes de abrir el notebook; después verifica con la estimación de ρ y σ. El piso √(ρσ²) te dice cuántos "
                      "árboles son «suficientes».", "cell")]))

    s.append(h("4", "Random Forest y max_features"))
    s += para("Random Forest añade <b>variables al azar en cada nodo</b>: de las p disponibles solo se evalúan m = <font name='Mono'>max_features</font>, y el mejor corte "
              "se busca ahí. Una variable dominante no puede controlar todos los árboles: se parecen menos y ρ baja. A cambio cada árbol es algo peor. "
              "Con m = p el Random Forest <b>es</b> un bagging.")
    s.append(ejemplo("una variable dominante", [
        P("p = 10 variables, m = 4 candidatas por nodo (el valor por defecto √10 daría 3).", "cell"),
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["P(la dominante es candidata en un nodo)", "m/p = 4/10", "0.40"],
               ["P(NO es candidata en la raíz)", "1 − 0.4", "0.60"],
               ["P(ausente en las raíces de 3 árboles)", "0.6³", "0.216"],
               ["P(ausente en las raíces de 10 árboles)", "0.6¹⁰", "0.006"]],
              [66 * mm, 40 * mm, CONTENT_W - 106 * mm - 18]),
        P("En el 60 % de los nodos la mejor variable ni siquiera compite: los demás árboles aprenden de otras variables y se vuelven distintos.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_max_features.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s += para("<b>Por nodo, no por árbol.</b> Si las variables se sortearan una sola vez por árbol (<font name='Mono'>BaggingRegressor(max_features=…)</font>), "
              "un árbol con variables malas sería malo en todo su recorrido: en Ames, con el 10 % de las columnas, eso da un RMSE de 27.1k contra 22.5k del bosque "
              "que sortea por nodo. Y el valor óptimo se valida: en Ames fue ≈ 10 %; en Heart la curva es casi plana.")
    s.append(pista("la Parte B del notebook y los Bloques 5 y 6 del cuaderno",
                   [P("En B3 compara «por árbol» y «por nodo» con la misma fracción. Luego haz tu propio barrido de <font name='Mono'>max_features</font> y "
                      "mira la brecha entre validación y OOB.", "cell")]))

    s.append(h("5", "OOB, número de árboles y búsqueda de hiperparámetros"))
    s += para("El <b>OOB</b> predice cada fila solo con los árboles que no la vieron: es una validación interna con <b>un solo ajuste</b>. Sirve para comparar "
              "configuraciones, pero el informe final necesita un test intacto. Más árboles estabilizan sin sobreajustar y cuestan tiempo; los demás hiperparámetros "
              "(max_features, min_samples_leaf, max_depth) se buscan con <font name='Mono'>GridSearchCV</font> o <font name='Mono'>RandomizedSearchCV</font>.")
    s.append(ejemplo("costo de una búsqueda y del paralelismo", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Ajustes de una grilla 3 × 2 × 2 × 4 con 5 bloques", "48 combinaciones × 5", "240 ajustes"],
               ["Bosque de 400 árboles a 0.5 s por árbol, 8 núcleos: tiempo ideal", "400 · 0.5 / 8", "25 s"],
               ["Si en la práctica tarda 35 s: aceleración y eficiencia", "200 / 35 · (5.7 / 8)", "5.7× · 71 %"],
               ["Árboles en los que cada fila es OOB (B = 200)", "0.368 · 200", "≈ 74"]],
              [76 * mm, 46 * mm, CONTENT_W - 122 * mm - 18]),
        P("«Se tarda B/N» es el límite ideal: la aceleración real queda bajo el número de núcleos (en Ames, 6.6× con 16 núcleos lógicos).", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_oob_cv.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 3))
    s += para("OOB, validación cruzada y test dan estimaciones en el mismo rango (Heart 0.811 · 0.798 · 0.855; Ames 26.0k · 26.3k · 27.0k), y el OOB cuesta una quinta parte. "
              "Y la búsqueda ayuda poco: en Heart pasó de AUC 0.899 (RF base) a ≈ 0.911 con Grid o Random, por debajo del desvío (≈ 0.03). El profesor lo resumió: "
              "la optimización «puede subir unos puntos», no convertir un modelo de 0.70 en uno de 0.95; y se justifica con CV y test, no por el nombre del algoritmo.")
    s.append(pista("la Parte C del notebook y el Bloque 7 del cuaderno",
                   [P("En C1 calcula el OOB a mano con <font name='Mono'>estimators_samples_</font>. En C3 incluye tu configuración base en la grilla y mira el test <b>una sola vez</b>.", "cell")]))

    s.append(h("6", "Importancia: impureza y permutación"))
    s += para("La importancia por <b>impureza</b> (MDI) suma, para cada variable, la reducción de Gini de sus cortes ponderada por la fracción de datos que llega al nodo. "
              "Se calcula con el train y premia a las variables con muchos cortes posibles. La de <b>permutación</b> desordena una variable en datos no vistos y mide cuánto "
              "empeora la métrica.")
    s.append(ejemplo("importancia por impureza a mano", [
        P("N = 80 (40 de cada clase). Raíz (Gini 0.5), corte por A: izquierda 30 (27/3, Gini 0.18), derecha 50 (13/37, Gini 0.3848). "
          "El nodo derecho se corta por B: 20 (11/9, Gini 0.495) y 30 (2/28, Gini 0.1244).", "cell"),
        pasos([["Corte", "ΔGini del nodo", "Peso N_t/N", "Aporte"],
               ["A (raíz)", "0.5 − (30·0.18 + 50·0.3848)/80 = 0.5 − 0.308 = 0.192", "80/80 = 1", "0.192"],
               ["B (nodo derecho)", "0.3848 − (20·0.495 + 30·0.1244)/50 = 0.3848 − 0.2727 = 0.1121", "50/80 = 0.625", "0.0701"]],
              [26 * mm, 80 * mm, 26 * mm, CONTENT_W - 132 * mm - 18]),
        P("Total 0.2621 → importancia normalizada: A = 0.192/0.2621 = <b>0.733</b> y B = <b>0.267</b>. "
          "Permutación: si el AUC base es 0.88 y al desordenar Edad baja a 0.85, Colesterol a 0.879 y una columna de ruido a 0.881, las importancias son 0.03, 0.001 y −0.001.", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_importancia.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook (D1) y el Bloque 8 del cuaderno",
                   [P("Agrega columnas de ruido a Heart y compara las dos importancias: el ruido continuo suele colarse por impureza y desaparecer por permutación. "
                      "Recuerda que ninguna importancia es causalidad.", "cell")]))

    s.append(h("7", "Límites: extrapolación, imágenes y modelos estables"))
    s += para("<b>No extrapola.</b> Cada predicción de regresión es un promedio de valores de hojas, así que queda dentro del rango de entrenamiento. "
              "<b>Imágenes:</b> cada píxel es una columna; el bosque clasifica bien los dígitos (exactitud 0.969), pero no usa qué píxel está al lado de cuál. "
              "<b>Modelos estables:</b> el bagging ayuda donde el modelo base cambia mucho con los datos (árboles), no en una regresión logística.")
    s.append(ejemplo("extrapolar con un bosque", [
        P("Entrenas con casas de 100 a 500 millones. Llega una casa que debería valer 700. Un bosque promedia valores de hojas de entrenamiento: su predicción nunca supera "
          "≈ 500. Una regresión lineal sí puede pasar de 500, bien o mal. En Ames, entrenado con casas &lt; 1.800 pies² el bosque no pasó de ≈ 158 mil USD en las de &gt; 2.400 pies² "
          "(RMSE 208 mil) y la lineal llegó a predecir ≈ 647 mil (RMSE 120 mil).", "cell")]))
    s.append(Spacer(1, 4))
    s.append(img("c35_imagenes.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 3))
    s.append(pista("la Parte D del notebook (D2 a D4) y el Bloque 9 del cuaderno",
                   [P("En D3 mezcla los píxeles al azar (el bosque rinde igual: 0.973) y desplaza la imagen un píxel (cae a 0.504). Es la mejor demostración de qué «ve» un bosque.", "cell")]))

    s.append(h("8", "Cómo elegir y comparar"))
    s.append(table([
        ["Si lo que domina tu problema es…", "Elige", "Porque"],
        ["Interpretar cada decisión", "Un árbol (o regresión)", "se lee de punta a punta"],
        ["Datos tabulares con no linealidades, sin tiempo para ajustar mucho", "Random Forest", "buen desempeño con poco ajuste, sin escalar"],
        ["Relaciones casi lineales, pocos datos", "Regresión logística / lineal", "con 303 pacientes igualó o superó al bosque (AUC 0.913)"],
        ["Mejor error posible y puedes afinar", "Boosting (Clase 36)", "reduce sesgo en secuencia"],
        ["Valores fuera del rango de entrenamiento", "Un modelo que extrapole", "los árboles se aplanan"],
    ], [66 * mm, 40 * mm, CONTENT_W - 106 * mm]))
    s.append(Spacer(1, 4))
    s.append(img("c35_comparacion.png", width=CONTENT_W * 0.88))
    s.append(Spacer(1, 3))
    s += para("Heart (CV 5×3): árbol 0.753 · logística 0.913 · bagging de logísticas 0.912 · bagging de árboles 0.893 · Random Forest 0.906, con desvíos ≈ 0.04: salvo el árbol, "
              "no hay diferencias claras. Ames (test): árbol 45.8k · bagging 28.9k · Random Forest 27.0k. Con una sola partición de 76 pacientes no se declara un ganador.")
    s.append(Spacer(1, 4))
    s.append(h("9", "Chuleta de código"))
    s.append(table([
        ["Quiero…", "Código"],
        ["Bagging con árboles", "BaggingClassifier(DecisionTreeClassifier(), n_estimators=120, oob_score=True, n_jobs=-1)"],
        ["Random Forest", "RandomForestClassifier(n_estimators=300, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)"],
        ["Random Forest de regresión", "RandomForestRegressor(n_estimators=300, max_features=0.2, min_samples_leaf=2, n_jobs=-1)"],
        ["Error OOB", "rf.oob_score_ · rf.oob_decision_function_ · rf.oob_prediction_ (regresión)"],
        ["Importancia correcta", "permutation_importance(pipe, X_test, y_test, scoring='roc_auc', n_repeats=30)"],
        ["Optimizar", "RandomizedSearchCV(pipe, {'model__max_features': [...]}, n_iter=12, scoring='roc_auc', cv=cv)"],
    ], [42 * mm, CONTENT_W - 42 * mm]))
    s.append(Spacer(1, 4))
    s.append(h("10", "Errores comunes"))
    s.append(table([
        ["Error", "Por qué está mal", "Qué hacer"],
        ["Llamar «Random Forest» a max_features=1.0", "Con todas las variables es un bagging", "Probar max_features &lt; 1"],
        ["Usar solo más árboles para mejorar", "El piso ρσ² no baja", "Bajar ρ (max_features) o cambiar de modelo"],
        ["Fiarse de la impureza con ruido o categóricas largas", "Premia variables con muchos cortes", "Permutación sobre el test"],
        ["Declarar ganador con 76 pacientes", "1 paciente = 1.3 puntos de exactitud", "CV repetida y media ± desvío"],
        ["Buscar hiperparámetros contra el test", "El test deja de ser honesto", "CV u OOB; test al final"],
        ["Esperar que el bosque extrapole", "Promedia hojas del rango visto", "Otro modelo o modelar la tendencia"],
    ], [58 * mm, 54 * mm, CONTENT_W - 112 * mm], header_color=RED))
    s.append(Spacer(1, 4))
    s.append(recuerda(["Orden sugerido: estos apuntes → Bloques 1–4 del cuaderno → Partes A y B del notebook → Bloques 5–7 → Parte C → "
                       "Bloques 8–9 → Parte D → ejercicio de cierre.",
                       "Si te trabas más de 10 minutos, mira la solución, entiéndela y vuelve a hacerlo sin mirar."]))
    build("apuntes/C35_Apuntes_RandomForest.pdf", s, "Apuntes · Clase 35 · Bagging y Random Forest")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0D9488">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 3)]
    flow.append(numbered([("", t) for t in items], C35))
    return flow


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 35 · Bagging y Random Forest",
                  "Para resolver en papel, con calculadora · soluciones paso a paso al final", C35))
    s.append(Spacer(1, 7))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: P(fuera) = (1 − 1/n)ⁿ · Var = ρσ² + (1 − ρ)σ²/B · Gini = 1 − Σ p².",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · "
                           "<font name='Sym'>★★★</font> integra varios conceptos.",
                           "Los bloques 8 y 9 usan los números de la guía (páginas 2 a 6)."], "cell", C35)], bg=SOFT, border=C35))
    s.append(Spacer(1, 4))
    s += bloque(1, "Bootstrap y fracción OOB", "★", None, [
        "Calcula P(una fila queda fuera) para n = 4, 10 y 50. ¿Hacia qué valor converge?",
        "n = 200: ¿cuántas filas distintas se esperan en una muestra bootstrap y cuántas quedan fuera?",
        "Datos 1 a 10 y muestra bootstrap 3, 3, 4, 6, 6, 6, 8, 9, 9, 10. ¿Cuántas filas únicas tiene? Escribe el OOB y su porcentaje.",
    ])
    s += bloque(2, "OOB a mano", "★★", "Cuatro árboles, seis observaciones (A–F, clase real entre paréntesis: A 1, B 0, C 1, D 0, E 1, F 0). "
                "Cada árbol predice solo sus observaciones OOB: <b>árbol 1</b> {A→1, C→0, E→1, F→0} · <b>árbol 2</b> {B→0, C→0, F→1} · "
                "<b>árbol 3</b> {A→1, B→0, D→0, F→0} · <b>árbol 4</b> {B→1, C→1, D→0, E→1}.", [
        "¿En cuántos árboles es OOB cada observación?",
        "Calcula la predicción OOB de cada observación por mayoría de votos.",
        "Calcula la exactitud OOB y el error OOB.",
    ])
    s += bloque(3, "Voto y promedio", "★", None, [
        "Siete árboles votan Sí, No, Sí, Sí, No, Sí, No. ¿Predicción y probabilidad estimada por votos? Con umbral 0.70, ¿cuál es la predicción?",
        "Probabilidades de cinco árboles: 0.90, 0.20, 0.60, 0.55, 0.35. Calcula la probabilidad media y la fracción de votos (P &gt; 0.5). ¿Coinciden?",
        "Cinco árboles de regresión predicen 210, 195, 230, 220 y 205 (miles USD). Calcula el promedio y la mediana.",
    ])
    s += bloque(4, "Varianza de un promedio", "★★", "σ² = 100 (σ = 10).", [
        "Con ρ = 0.2 calcula Var para B = 1, 5, 25 y B → ∞.",
        "Repite con ρ = 0.6. ¿Cuánto ganas pasando de 5 a 25 árboles en cada caso?",
        "<font name='Sym'>★★★</font> ¿Desde qué B el término (1 − ρ)σ²/B es menor que ρσ² con ρ = 0.2? ¿Y con ρ = 0.6? Interpreta.",
    ])
    s += bloque(5, "max_features", "★", None, [
        "Calcula las candidatas por nodo con p = 16 para 'sqrt', 'log2' y max_features = 1/3 (redondea hacia abajo).",
        "p = 8 y m = 3: ¿probabilidad de que la variable dominante sea candidata en un nodo? ¿Y de que falte en las raíces de 5 árboles?",
        "¿Qué algoritmo es un Random Forest con max_features = p? ¿Por qué?",
    ])
    s += bloque(6, "Importancia por impureza (Gini)", "★★★", "N = 100 (50 y 50). Raíz: corte por X₁ → izquierda 40 (35 de la clase 1, 5 de la 0), derecha 60 (15 de la 1, 45 de la 0). "
                "El nodo derecho se corta por X₂ → 20 (12 de la 1, 8 de la 0) y 40 (3 de la 1, 37 de la 0).", [
        "Calcula el Gini de la raíz, de los dos hijos del primer corte y la reducción de impureza ponderada.",
        "Calcula la reducción del segundo corte y multiplícala por N_t/N.",
        "Normaliza para obtener la importancia de X₁ y X₂.",
    ])
    s += bloque(7, "OOB, CV y costo", "★★", None, [
        "Un bosque de 600 árboles: ¿en cuántos es OOB una fila (en promedio)? Con B = 8 árboles, ¿qué proporción de filas no tiene ninguna predicción OOB?",
        "Una grilla de 3 × 3 × 2 combinaciones con CV de 5 bloques: ¿cuántos ajustes? Si cada ajuste dura 4 s en serie y paralelizas en 6 núcleos (ideal), ¿cuánto tarda?",
        "El OOB de un bosque da exactitud 0.84 y el test 0.79 con 60 pacientes. ¿Es una contradicción? Justifica con el error de una proporción (≈ √(p(1 − p)/n)).",
    ])
    s += bloque(8, "Importancia por permutación y lectura", "★★", "AUC base 0.90. Al desordenar cada variable en el test: X₁ → 0.74 · X₂ → 0.89 · X₃ (ruido) → 0.90 · X₄ → 0.905.", [
        "Calcula la importancia por permutación de cada variable.",
        "¿Qué variables eliminarías? ¿Qué significa un valor negativo pequeño?",
        "Una variable continua de ruido aparece con 7 % de importancia por impureza. ¿Qué dos razones lo explican y qué harías?",
    ])
    s += bloque(9, "Elegir y leer resultados", "★★★", None, [
        "Heart: logística AUC 0.913 ± 0.035 y Random Forest 0.906 ± 0.036 (CV 5×3). ¿Puedes afirmar que la logística es mejor? ¿Qué modelo elegirías si importa interpretar?",
        "Entrenas con precios de 100 a 500 mil y el bosque predice 498 mil para una casa que vale 700 mil. ¿Hay un error de código? Explica y propón una alternativa.",
        "Un bosque tiene exactitud de train 1.000 y de test 0.85. ¿Es sobreajuste que debes corregir? ¿Qué mirarías antes de decidir?",
        "Elige modelo y justifica: (a) clasificar fraude con 2 % de positivos; (b) clasificar fotos 256×256; (c) 300 filas casi lineales.",
    ])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 6))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [12 * mm, CONTENT_W - 12 * mm], header_color=GREEN_C),
                Spacer(1, 5)]

    s += sol("Bloque 1", [
        ["1", "n = 4: (3/4)⁴ = 0.316 · n = 10: 0.9¹⁰ = 0.349 · n = 50: 0.98⁵⁰ = 0.364. Converge a e⁻¹ ≈ 0.368."],
        ["2", "(1 − 1/200)²⁰⁰ = 0.367 fuera → filas distintas = 200 · (1 − 0.367) ≈ 126.6 y fuera ≈ 73.4."],
        ["3", "Únicas: 3, 4, 6, 8, 9, 10 → 6 filas (60 %). OOB = {1, 2, 5, 7} → 4 filas = 40 %."]])
    s += sol("Bloque 2", [
        ["1", "A: árboles 1 y 3 · B: 2, 3 y 4 · C: 1, 2 y 4 · D: 3 y 4 · E: 1 y 4 · F: 1, 2 y 3."],
        ["2", "A: 1, 1 → 1 · B: 0, 0, 1 → 0 · C: 0, 0, 1 → 0 · D: 0, 0 → 0 · E: 1, 1 → 1 · F: 0, 1, 0 → 0."],
        ["3", "Comparando con la clase real (1, 0, 1, 0, 1, 0): aciertan A, B, D, E y F; falla C. Exactitud OOB = 5/6 = 0.833; error OOB = 1/6 = 0.167."]])
    s += sol("Bloque 3", [
        ["1", "Cuatro Sí de siete → Sí. Probabilidad por votos 4/7 = 0.571. Con umbral 0.70, 0.571 &lt; 0.70 → No."],
        ["2", "Promedio = (0.90 + 0.20 + 0.60 + 0.55 + 0.35)/5 = 2.60/5 = 0.52. Votos con P &gt; 0.5: 0.90, 0.60 y 0.55 → 3/5 = 0.60. No coinciden (0.52 vs 0.60)."],
        ["3", "Promedio = (210 + 195 + 230 + 220 + 205)/5 = 1.060/5 = 212. Mediana = 210."]])
    s += sol("Bloque 4", [
        ["1", "ρ = 0.2: B = 1 → 20 + 80 = 100 · B = 5 → 20 + 16 = 36 · B = 25 → 20 + 3.2 = 23.2 · B → ∞ → 20."],
        ["2", "ρ = 0.6: B = 1 → 100 · B = 5 → 60 + 8 = 68 · B = 25 → 60 + 1.6 = 61.6 · B → ∞ → 60. De 5 a 25 árboles: con ρ = 0.2 baja 12.8 (36 → 23.2); con ρ = 0.6, solo 6.4 (68 → 61.6)."],
        ["3", "(1 − ρ)/B &lt; ρ ⇒ B &gt; (1 − ρ)/ρ. ρ = 0.2: B &gt; 4 · ρ = 0.6: B &gt; 0.67, o sea desde el primer árbol. Con ρ alto el término de la correlación domina enseguida: más árboles ya no ayudan."]])
    s += sol("Bloque 5", [
        ["1", "'sqrt' → 4 · 'log2' → 4 · 1/3 → ⌊5.33⌋ = 5."],
        ["2", "P(candidata) = 3/8 = 0.375. P(falta en una raíz) = 5/8 = 0.625 → en 5 raíces: 0.625⁵ = 0.095."],
        ["3", "Un bagging: con todas las variables en cada nodo no hay selección aleatoria, solo bootstrap."]])
    s += sol("Bloque 6", [
        ["1", "Raíz: 1 − (0.5² + 0.5²) = 0.5. Izquierda (35/5): 1 − (0.875² + 0.125²) = 0.21875. Derecha (15/45): 1 − (0.25² + 0.75²) = 0.375. "
              "Ponderado: 0.4·0.21875 + 0.6·0.375 = 0.3125. Reducción = 0.5 − 0.3125 = 0.1875 (peso 100/100 = 1)."],
        ["2", "Nodo derecho: 0.375. Hijos: (12/8) 1 − (0.36 + 0.16) = 0.48; (3/37) 1 − (0.075² + 0.925²) = 0.13875. Ponderado: (20·0.48 + 40·0.13875)/60 = 0.2525. "
              "Reducción = 0.375 − 0.2525 = 0.1225; por N_t/N = 60/100: 0.0735."],
        ["3", "Total = 0.1875 + 0.0735 = 0.261. X₁ = 0.1875/0.261 = 0.718 · X₂ = 0.0735/0.261 = 0.282."]])
    s += sol("Bloque 7", [
        ["1", "0.368 · 600 ≈ 221 árboles. Con B = 8: 0.632⁸ = 0.025 → 2.5 % de las filas sin ninguna predicción OOB."],
        ["2", "3·3·2 = 18 combinaciones × 5 = 90 ajustes · 90·4 = 360 s en serie → 360/6 = 60 s (ideal; en la práctica algo más)."],
        ["3", "No es contradicción: el error estándar de una proporción de 0.8 con n = 60 es √(0.8·0.2/60) = 0.052, así que 0.84 y 0.79 difieren menos de un error estándar. Con tan pocos pacientes el test es muy ruidoso."]])
    s += sol("Bloque 8", [
        ["1", "X₁: 0.90 − 0.74 = 0.16 · X₂: 0.01 · X₃: 0.00 · X₄: 0.90 − 0.905 = −0.005."],
        ["2", "Solo X₁ importa claramente; X₂ aporta poco y X₃, X₄ nada. Un valor negativo pequeño es ruido: desordenar la variable «mejoró» por azar, o sea no aporta."],
        ["3", "(a) La impureza se calcula con el train y premia variables con muchos cortes posibles, aunque no tengan relación con el objetivo; (b) el bosque memoriza el train, y ahí el ruido «sirve». Calcular la permutación sobre datos no vistos (con más repeticiones) y comparar."]])
    s += sol("Bloque 9", [
        ["1", "No: la diferencia (0.007) es mucho menor que cualquiera de los desvíos (≈ 0.035). Si importa interpretar, la logística (coeficientes, odds) o un árbol, no el bosque."],
        ["2", "No es un error: el bosque promedia valores de hojas de entrenamiento y no pasa de ≈ 500 mil. Alternativas: regresión lineal, modelar la tendencia aparte o transformar el objetivo (p. ej. precio por pie²)."],
        ["3", "No necesariamente. Un bosque con hojas de una observación memoriza el train, y aun así generaliza: lo que importa es el error de test o de CV frente a otras configuraciones (min_samples_leaf mayor, otro max_features) y la brecha entre OOB y test."],
        ["4", "(a) Random Forest con class_weight o un modelo de boosting, evaluado con recall, F1 o PR-AUC, no con exactitud. (b) Una red convolucional (el bosque no ve la estructura espacial). (c) Regresión logística: casi lineal y pocos datos."]])
    build("apuntes/C35_Ejercicios_Cuaderno_RandomForest.pdf", s, "Ejercicios para el cuaderno · Clase 35")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
