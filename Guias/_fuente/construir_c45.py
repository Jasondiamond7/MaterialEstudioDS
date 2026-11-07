"""Clase 45 · Redes neuronales (Prof. Eloy Alvarado): guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c45_numeros.json (lo escribe figuras_c45.py); los de los ejemplos y ejercicios se calculan aquí.
Los números de las diapositivas (corrida del profesor) se citan como «diapositivas». Con el programa del profesor, nuestra reproducción coincide
con todas ellas; lo que se rotula «nuestra reproducción» (otros anchos, barrido de alpha, Adam) es cálculo propio."""
import json
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import CondPageBreak, KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c45_numeros.json").read_text(encoding="utf-8"))
C45 = colors.HexColor("#0F766E")
C45_BG = colors.HexColor("#F0FDFA")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas redes_neuronales.pdf y programa redes_neuronales.py del profesor Eloy Alvarado (reproducido con scikit-learn)"
EN, NE, CA, PR, EJ, CV = N["neurona"], N["neurona"], N["capa"], N["params"], N["ejemplo"], N["curvas"]
EV, ER, AN, AL, LO, GD, AD = N["eval"], N["errores_resumen"], N["anchos"], N["alphas"], N["lotes"], N["gd"], N["adam"]
SM, ME = N["softmax"], N["metricas"]


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{x:,}".replace(",", " ")


def pc(x, d=2):
    return f"{x * 100:.{d}f} %"


def sigm(a):
    return 1 / (1 + exp(-a))


# ======================================================================= GUÍA
def tema(s, num, titulo, idea, conceptos, figuras, simple, formulas, salida, cuidado, minimo=50):
    """Un tema: título, idea central, conceptos, figuras (a ancho casi completo), analogía, números de la clase y advertencia."""
    s.append(CondPageBreak(minimo * mm))
    s.append(P(f'<font color="#0F766E">{num}</font>  {titulo}', "h2"))
    s.append(box([P(idea, "big")], bg=SOFT, border=C45, pad=5))
    s.append(Spacer(1, 3))
    s.append(numbered(conceptos, C45))
    s.append(Spacer(1, 2))
    for fig, pie, *w in figuras:
        s.append(KeepTogether([img(fig, width=CONTENT_W * (w[0] if w else 0.8)), P(pie, "small"), Spacer(1, 3)]))
    left = [P("En palabras simples", "h3"), box([P(simple, "cell")], bg=C45_BG, border=C45, width=HALF, pad=5)]
    if formulas:
        left += [Spacer(1, 3), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida, "cell", C45)]
    if cuidado:
        right += [Spacer(1, 3), box([P("<b>Cuidado.</b> " + cuidado, "cell")], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
    s.append(KeepTogether([two_cols(left, right)]))
    s.append(Spacer(1, 6))


def guia():
    s = []
    s.append(band("Clase 45 · Redes neuronales",
                  "Profesor Eloy Alvarado · primera mirada al interior de una red: neurona, capas, aprendizaje por gradiente y evaluación, con los dígitos de la Clase 44", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Una red neuronal es una cadena de operaciones muy simples.</b> Cada neurona hace una suma ponderada de sus entradas, le suma un sesgo y aplica una función de activación; "
                    "las salidas de una capa son las entradas de la siguiente; y al final <b>softmax</b> convierte los números en 10 probabilidades (una por dígito). "
                    "Para <b>aprender</b>, la red mide su error con la pérdida logarítmica, calcula cuánto influye cada peso en ese error (<b>retropropagación</b>) y mueve cada peso un poco en la dirección que lo reduce "
                    "(<b>descenso de gradiente</b>, aquí con <b>Adam</b>), lote tras lote y época tras época. "
                    "El número de capas y neuronas, el tamaño del paso, el lote, alpha y las épocas son <b>hiperparámetros</b>: se fijan antes de entrenar y se eligen con la <b>validación</b>, nunca con la prueba.", "body")],
                 bg=C45_BG, border=C45, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("El ciclo de entrenamiento en una imagen", "h2"))
    s.append(img("c45_ciclo.png", width=CONTENT_W * 0.97))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Regresión logística y probabilidades</b> (Clases 20–21 y 31): una probabilidad por clase",
                     "<b>Log loss y exactitud</b> (Clases 38 y 44): −log de la probabilidad de la clase real",
                     "<b>Validación y prueba</b> (Clases 30 y 38): elegir con una, evaluar al final con otra",
                     "<b>Idea de derivada</b> (pendiente de una curva) y <b>multiplicar matrices</b>: se repasan aquí con números pequeños"], "cell", C45)]
    right = [P("Ruta de esta guía", "h3"),
             bullets(["Neurona → activaciones → capa y matrices → arquitectura 64–32–10 y parámetros",
                      "Salida softmax → pérdida → objetivo con penalización → descenso de gradiente → retropropagación",
                      "Lotes, épocas, Adam e inicialización → curvas, ancho y alpha",
                      "Evaluación: log loss vs errores, matriz de confusión, errores seguros y límites"], "cell", C45)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Cómo leer los números de este material</b>", "cellb"),
                  P("Las <b>diapositivas</b> muestran la corrida del profesor. Aquí se ejecutó <b>su mismo programa</b> (<font name='Mono'>redes_neuronales.py</font>, scikit-learn 1.9.1, misma semilla) y "
                    f"<b>las {len(N['comparacion'])} cifras comparadas coinciden exactamente</b> con las diapositivas (tabla en la última página). "
                    "Lo que se rotula «nuestra reproducción» (otros anchos, barrido de alpha, Adam sobre el juguete) es cálculo propio, no de la clase.", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))

    # ---------------------------------------------------------------- 1 neurona
    nb, n2, n3 = NE["esc"]
    tema(s, "1", "La neurona: una suma ponderada con un interruptor",
         "<b>Una neurona recibe números, los multiplica por pesos, suma un sesgo y pasa el resultado por una función de activación.</b> Eso es todo: a = w₁x₁ + … + wₚxₚ + b, y la salida es h = φ(a).",
         [("Entradas x", "Los números que llegan: en la clase, las 64 intensidades de una imagen (entre 0 y 1) o las salidas de la capa anterior."),
          ("Pesos w", "Cuánta importancia da la neurona a cada entrada. Un peso positivo hace subir la suma; uno negativo, bajarla. Son <b>parámetros</b>: se aprenden."),
          ("Sesgo b", "Un número que desplaza toda la suma. Permite que la neurona responda aunque todas las entradas sean 0. También es un parámetro."),
          ("Activación h = φ(a)", "El resultado de aplicar una función a la suma. Con ReLU, φ(a) = máx(0, a): si a es negativo, la neurona entrega 0.")],
         [("c45_neurona.png", "Diapositivas 5 y 6: una neurona con x = (0.5, 1), pesos (0.8, −0.4) y sesgo 0.2; a la derecha, lo que cambia si cambia x₂ o el sesgo.")],
         "Piensa en un <b>jurado de dos personas</b>: cada una da su opinión (x), tú le crees más a una que a otra (los pesos), partes con una predisposición (el sesgo) y decides «sí» solo si el balance es positivo (ReLU). "
         "Los pesos no cambian entre una imagen y otra; lo que cambia es la opinión que entra, y por eso cambia la respuesta.",
         ["a = w₁x₁ + w₂x₂ + b", "h = ReLU(a) = máx(0, a)"],
         [f"Caso base: a = 0.8(0.5) − 0.4(1) + 0.2 = {nb['a']:g} → h = {nb['h']:g}",
          f"Con x₂ = 2: a = 0.4 − 0.8 + 0.2 = {n2['a']:g} → ReLU lo lleva a h = {n2['h']:g}",
          f"Con b = 0.5 y x₂ = 1: a = 0.4 − 0.4 + 0.5 = {n3['a']:g} → h = {n3['h']:g}",
          "ReLU(−2) = 0 y ReLU(3) = 3 (diapositiva 5)"],
         "«Activación» no significa «distinta de cero»: una activación de 0 es un resultado válido. Y las activaciones <b>no son probabilidades</b> ni representan un dígito en particular.")

    # ---------------------------------------------------------------- 2 activaciones y no linealidad
    C = N["compone"]
    tema(s, "2", "Funciones de activación y por qué hace falta la no linealidad",
         "<b>La suma ponderada es una función lineal; sin una activación no lineal entre capas, apilar capas no aporta nada.</b> Dos capas lineales seguidas equivalen a una sola capa lineal.",
         [("ReLU", "Deja pasar lo positivo y lleva lo negativo a 0. Puede superar 1. Es la habitual en capas ocultas."),
          ("Sigmoide", "Comprime cualquier valor al intervalo (0, 1): sirve para salidas binarias que se leen como probabilidad (σ(0) = 0.5)."),
          ("Tangente hiperbólica", "Comprime al intervalo (−1, 1); tanh(0) = 0."),
          ("Transformar el espacio", "Con una capa oculta no lineal, puntos que ninguna recta separaba pasan a poder separarse (ejemplo de la diapositiva 11).")],
         [("c45_activaciones.png", "Diapositivas 5 y 7: las tres funciones; los puntos negros son a = −2, −1, 0, 1 y 3."),
          ("c45_xor.png", "Diapositiva 11: clase 1 si las dos coordenadas son distintas. Con h₁ = máx(0, x₁−x₂) y h₂ = máx(0, x₂−x₁), una recta (h₁ + h₂ = 0.5) separa las clases. Los pesos se fijaron a mano.")],
         "Imagina que tienes <b>reglas transparentes</b>: poner varias reglas rectas una encima de otra sigue dando una línea recta. La activación es como <b>doblar</b> la regla en un punto: con dobleces se puede formar curvas. "
         "Y en el ejemplo, la capa oculta «pliega» el cuadrado de modo que dos esquinas opuestas quedan juntas y las otras dos separadas.",
         ["s = (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂)", "→ una sola capa lineal con W* y b*"],
         [f"Sin activación, con x = (3, −1): dos capas dan {C['dos_capas']:g} y la capa única equivalente también da {C['una_capa']:g} (ejemplo propio)",
          "Tabla de la diapositiva 11: (0,0) → (0,0); (1,1) → (0,0); (1,0) → (1,0); (0,1) → (0,1); h₁ + h₂ vale 0, 0, 1, 1",
          "Cuándo usar cuál: ReLU en capas intermedias; sigmoide (o softmax) en la salida de clasificación; salida lineal si se predice un número"],
         "La función de activación de la <b>salida</b> depende de la tarea (diapositiva 16): ReLU no sirve para producir probabilidades porque no está acotada.")

    # ---------------------------------------------------------------- 3 capa y matrices
    tema(s, "3", "Una capa: muchas neuronas a la vez, con matrices",
         "<b>Una capa reúne varias neuronas que reciben la misma entrada pero tienen pesos y sesgos propios.</b> Calcularlas todas es un producto de matrices: X(n×p) · W(p×q) = A(n×q), más el sesgo y la activación.",
         [("Cada columna de W es una neurona", "Con 2 entradas y 2 unidades, W es 2×2: la primera columna tiene los pesos de la unidad 1; la segunda, los de la unidad 2."),
          ("Cada fila de X es una observación", "Se pueden procesar 3 imágenes (o 360) de una vez con una sola multiplicación: cada fila usa los mismos pesos y sesgos."),
          ("Dimensiones", "n observaciones × p entradas, por p × q unidades, da n × q activaciones. El sesgo (q números) se suma a todas las filas."),
          ("Con imágenes", "En la clase, X es 360 × 64 (imágenes de validación), W₁ es 64 × 32 y H es 360 × 32.")],
         [("c45_capa.png", "Diapositivas 8, 12 y 14: una observación x = (1, 2) y tres observaciones a la vez; sesgos en cero. En la fila 2, a₂ = −1.5 pasa a 0 por ReLU.")],
         "Es como una <b>planilla de cálculo</b>: cada columna de W es una «receta» con sus propios ingredientes (pesos) y cada fila de X es un «pedido». La multiplicación aplica todas las recetas a todos los pedidos de una sola vez.",
         ["a₁ = 1·1 + 2·0.5 = 2 → h₁ = 2", "a₂ = 1·(−1) + 2·0.5 = 0 → h₂ = 0"],
         ["x = (1, 2) con W₁ = [[1, −1], [0.5, 0.5]] y b₁ = (0, 0) da h = (2, 0)",
          "Fila 2 de X, (2, 1): a = (2·1 + 1·0.5, 2·(−1) + 1·0.5) = (2.5, −1.5) → h = (2.5, 0)",
          "Fila 3, (0, 1): a = (0.5, 0.5) → h = (0.5, 0.5)"],
         "Hay que usar siempre la misma convención: aquí cada observación es una <b>fila</b> y se multiplica x·W (no W·x).")

    # ---------------------------------------------------------------- 4 arquitectura y parámetros
    tema(s, "4", "Leer una arquitectura y contar sus parámetros: 64–32–10",
         f"<b>Se lee de izquierda a derecha: 64 entradas, 32 unidades ocultas, 10 salidas.</b> Al ajustar, la red 64–32–10 tiene {mil(PR['total32'])} parámetros (pesos y sesgos); la 64–16–10, {mil(PR['total16'])}.",
         [("Capa oculta y ancho", "Las capas entre la entrada y la salida. El número de unidades de una capa es su <b>ancho</b>; el número de capas, su <b>profundidad</b>."),
          ("Hiperparámetro", "Se fija antes de ajustar con los datos (32, 16, número de capas, activación…). No hay valor óptimo conocido de antemano: hay que probar."),
          ("Parámetro", "Se aprende al entrenar: los pesos W y los sesgos b. No se cuentan las activaciones ni los píxeles."),
          ("Red densa", "Cada unidad recibe las 64 entradas. Otras arquitecturas (convolucionales) conectan solo partes de la imagen.")],
         [("c45_red.png", "Diapositivas 9 y 13: se dibujan 7 de las 64 entradas y 6 de las 32 unidades ocultas; las 10 salidas son los dígitos 0 a 9.")],
         "Como en una <b>cadena de montaje</b>: 64 piezas entran, 32 estaciones intermedias las combinan cada una a su manera y 10 estaciones finales entregan un puntaje por dígito. "
         "Cada flecha del dibujo es un peso; cada estación tiene además su sesgo. Más estaciones = más capacidad, pero también más cosas que ajustar y más tiempo.",
         [f"Ancho 32: 32(64 + 1) + 10(32 + 1) = {PR['W1'] + PR['b1']} + {PR['W2'] + PR['b2']} = {mil(PR['total32'])}",
          f"Ancho 16: 16(64 + 1) + 10(16 + 1) = {PR['w16_1'] + PR['b16_1']} + {PR['w16_2'] + PR['b16_2']} = {mil(PR['total16'])}"],
         [f"W₁ = 64×32 = {mil(PR['W1'])} · b₁ = {PR['b1']} · W₂ = 32×10 = {PR['W2']} · b₂ = {PR['b2']} (diapositiva 13)",
          f"La regresión logística sobre los mismos píxeles tiene 64·10 + 10 = {PR['logistica']} parámetros (nuestro cálculo)",
          f"Agregar una segunda capa oculta de 32 a la 64–32–10 da {mil(PR['doble32'])} parámetros, no el doble (ejemplo propio)",
          "Los modelos actuales tienen millones o miles de millones de parámetros ya ajustados (comentado en clase)"],
         "Contar parámetros es contar columnas: cada unidad tiene (entradas + 1) parámetros, donde el +1 es el sesgo.")

    # ---------------------------------------------------------------- 5 salida softmax y pérdida
    tema(s, "5", "La salida: softmax y la pérdida de una observación",
         "<b>Para clases excluyentes, la última capa usa softmax: convierte 10 puntajes cualquiera en 10 probabilidades que suman 1.</b> La pérdida de una imagen es −log de la probabilidad que se dio a la clase real.",
         [("Tarea y salida", "Regresión: un número (pérdida: error cuadrático). Binaria: una sigmoide (entropía cruzada binaria). Clases excluyentes: softmax (entropía cruzada multiclase). Salida y pérdida deben corresponder a la tarea."),
          ("Softmax", "Se exponencia cada puntaje y se divide por la suma de los exponenciales. Todos quedan entre 0 y 1 y suman 1."),
          ("Pérdida −log p", "Usa la probabilidad de la clase <b>real</b>, aunque otra clase tenga mayor probabilidad. Castiga fuerte los errores seguros."),
          ("Truco numérico", "Restar el máximo c a los puntajes antes de exponenciar no cambia las probabilidades y evita números enormes.")],
         [("c45_softmax.png", "Diapositiva 17: puntajes s = (0, ln 2, 0) → exponenciales (1, 2, 1), suma 4 → probabilidades (0.25, 0.50, 0.25)."),
          ("c45_perdida.png", "Diapositiva 18: pérdida = −log p para p = 0.8, 0.5 y 0.1.")],
         "Softmax es como <b>repartir 100 puntos de confianza</b> entre las 10 opciones según cuánto destaca cada puntaje (los exponenciales exageran las diferencias). "
         "La pérdida es la <b>multa</b> por haber apostado poco a la respuesta correcta: con 80 % de apuesta la multa es pequeña; con 10 %, grande.",
         ["pₖ = e^sₖ ÷ Σⱼ e^sⱼ", "pérdida = −log p(clase real)"],
         [f"s = (0, ln 2, 0) → p = (¼, ½, ¼): {', '.join(f'{v:.2f}' for v in SM['p'])}",
          f"−log 0.8 = {f(N['perdida']['0.8'], 3)} · −log 0.5 = {f(N['perdida']['0.5'], 3)} · −log 0.1 = {f(N['perdida']['0.1'], 3)} (diapositiva 18)",
          "Sumar 100 a todos los puntajes da las mismas probabilidades (diapositiva 17)",
          "Ejemplo binario de la diapositiva 15: s = 0.5 → σ(0.5) = 0.6225 (con 10 clases no basta una sigmoide)"],
         "Las salidas de softmax se <b>interpretan</b> como probabilidades; no garantizan estar bien calibradas (la clase pasada ya mostró errores confiados).")

    # ---------------------------------------------------------------- 6 objetivo y gradiente
    tema(s, "6", "Qué es «aprender»: minimizar J con el descenso de gradiente",
         "<b>Aprender es buscar los pesos y sesgos θ que minimizan el objetivo J(θ) = pérdida media + penalización por pesos grandes.</b> El gradiente indica hacia dónde sube J; se avanza en sentido contrario.",
         [("Objetivo J", "Primer término: el promedio de las pérdidas de las imágenes (ajuste a los datos). Segundo: λ/2 por la suma de los pesos al cuadrado (preferencia por pesos pequeños). En scikit-learn el control es <b>alpha</b>."),
          ("Gradiente", "La derivada en varias dimensiones: una pendiente por cada parámetro. Si la pendiente es negativa, aumentar el parámetro baja J."),
          ("Regla de actualización", "w⁺ = w − η · J′(w). η (eta) es la <b>tasa de aprendizaje</b>, el tamaño del paso."),
          ("Tamaño del paso", "Con η muy pequeño se avanza lento; con η muy grande se pasa de largo y J puede empezar a crecer.")],
         [("c45_gd.png", "Diapositivas 20 y 27: J(w) = (w−3)². Izquierda: pasos de η = 0.1 desde w = 1. Derecha: pérdida tras cada actualización con cuatro tasas (inicio w = 0).")],
         "Es como <b>bajar un cerro con niebla</b>: no ves el valle, pero sientes con los pies hacia dónde baja el suelo y das un paso en esa dirección. "
         "Si los pasos son diminutos, tardas horas; si son enormes, saltas al otro lado del valle y terminas más arriba.",
         ["J′(w) = 2(w − 3)", "w⁺ = w − η·J′(w)"],
         [f"w = 1: J = (1−3)² = {GD['J0']:g}; J′(1) = 2(1−3) = {GD['grad0']:g}; w⁺ = 1 − 0.1(−4) = {GD['w1']:g}; J(1.4) = {GD['J1']:g}",
          "Restar una derivada negativa aumenta el parámetro, y aquí eso reduce la pérdida",
          f"Pérdida tras 10 actualizaciones desde w = 0: η = 0.05 → {GD['tasas']['0.05']['perdida'][-1]:.2f}; η = 0.1 → {GD['tasas']['0.1']['perdida'][-1]:.3f}; η = 0.4 → {GD['tasas']['0.4']['perdida'][-1]:.1e}; η = 1.05 → {GD['tasas']['1.05']['perdida'][-1]:.1f} (cálculo propio: factor (1−2η)² por paso)",
          "(−w)² = w² y (2w)² = 4w²: la penalización mira la magnitud y crece al cuadrado"],
         "J(w) = (w − 3)² es un <b>juguete</b>: la pérdida real de la red depende de miles de parámetros. Con η = 1.05 la pérdida crece en vez de bajar.")

    # ---------------------------------------------------------------- 7 retropropagación
    tema(s, "7", "Retropropagación: repartir la «culpa» del error hacia atrás",
         "<b>La retropropagación aplica la regla de la cadena para calcular cuánto cambia la pérdida si se mueve cada parámetro.</b> Primero se calcula hacia adelante (predicción y pérdida); luego se vuelve hacia atrás obteniendo una sensibilidad por parámetro.",
         [("Hacia adelante", "Con x = 0.5, y = 1 y θ = (w₁, b₁, w₂, b₂) = (0.4, 0.1, 0.8, −0.2): a₁ = 0.3, h = 0.3, a₂ = 0.04, p = σ(0.04) ≈ 0.51, ℓ = −log p ≈ 0.6733."),
          ("Error en la salida", "Con sigmoide y entropía cruzada binaria, δ₂ = ∂ℓ/∂a₂ = p − y. Aquí δ₂ = 0.51 − 1 ≈ −0.49."),
          ("Hacia atrás", "δ₁ = δ₂·w₂·ReLU′(a₁) = (−0.49)(0.8)(1) ≈ −0.392. Luego: ∂ℓ/∂w₂ = δ₂h, ∂ℓ/∂b₂ = δ₂, ∂ℓ/∂w₁ = δ₁x, ∂ℓ/∂b₁ = δ₁."),
          ("Qué hace el optimizador", "Backpropagation solo calcula los gradientes. El <b>optimizador</b> decide cómo cambiar los parámetros con ellos.")],
         [("c45_backprop.png", "Diapositivas 21 a 23: el ejemplo completo, con la propagación (arriba) y la retropropagación (abajo).")],
         "Es una <b>cadena de responsabilidades</b>: al final alguien dice «salió mal por 0.49». Esa persona le avisa a la anterior cuánto pesó su parte (multiplicando por su peso w₂), ella avisa a la anterior, y así hasta los primeros pesos. "
         "Cada uno aprende cuánto le tocó de la culpa y corrige solo su parte.",
         ["δ₂ = p − y", "δ₁ = δ₂·w₂·ReLU′(a₁)", "∂ℓ/∂w = δ · (entrada de ese peso)"],
         [f"∇ℓ = ({EJ['grad'][0]:.4f}, {EJ['grad'][1]:.4f}, {EJ['grad'][2]:.4f}, {EJ['grad'][3]:.4f})",
          f"Paso con η = 0.1: θ⁺ = ({EJ['theta2'][0]:.4f}, {EJ['theta2'][1]:.4f}, {EJ['theta2'][2]:.4f}, {EJ['theta2'][3]:.4f})",
          f"Nueva predicción: a₁ ≈ {EJ['a1n']:.4f}, a₂ ≈ {EJ['a2n']:.4f}, p ≈ {EJ['pn']:.6f}; pérdida {EJ['l']:.6f} → {EJ['ln']:.6f} (baja)",
          f"Comprobación numérica (diferencias finitas): el gradiente coincide con diferencia máxima {EJ['dif_max']:.0e}"],
         "Un paso que baja la pérdida de este ejemplo <b>no demuestra</b> que la red generalice. Las cifras de la diapositiva están redondeadas.")

    # ---------------------------------------------------------------- 8 entrenar: lotes, épocas, Adam
    a1, a100 = AD["pasos1"], AD["pasos100"]
    tema(s, "8", "Entrenar en la práctica: lotes, épocas, Adam e inicialización",
         "<b>No se actualiza con una sola imagen ni con todas a la vez: se reparten en lotes.</b> Un lote produce una actualización; recorrer todo el entrenamiento es una época. El optimizador Adam decide el tamaño de cada paso.",
         [("Lote, actualización y época", "Lote: un grupo de imágenes. Actualización: un cambio de parámetros con un lote. Época: una pasada por todas las imágenes de entrenamiento."),
          ("Adam", "Un optimizador que adapta el paso de cada parámetro: promedia los gradientes recientes (inercia) y los divide por su tamaño típico. La clase solo lo nombró; aquí se ilustra con el juguete."),
          ("Inicialización", "Hay que partir de valores iniciales. Si todas las unidades parten con los mismos pesos, calculan lo mismo y se mantienen iguales: conviene que difieran."),
          ("Cuándo parar", "Cuando la pérdida de validación deja de mejorar o se agota el presupuesto de épocas (120 en la clase).")],
         [("c45_lotes.png", "Diapositiva 26: 1 077 = 16·64 + 53; 17 actualizaciones por época; 120 épocas × 17 = 2 040 actualizaciones."),
          ("c45_adam.png", "Ilustración propia (no está en las diapositivas): Adam con η = 0.1 avanza ≈ 0.1 por actualización aunque el gradiente sea 100 veces mayor; el descenso simple se desborda.")],
         "Los lotes son como <b>corregir un examen por montones</b> de 64 hojas y ajustar tu criterio después de cada montón, en vez de esperar a corregir las 1 077. "
         "Adam es un <b>conductor con inercia</b>: recuerda hacia dónde venía y frena o acelera según qué tan fuerte es el terreno de cada parámetro.",
         ["lotes por época = ⌈1 077 ÷ 64⌉ = 17", "Adam: paso ≈ η · (gradiente medio) ÷ (tamaño típico del gradiente)"],
         [f"1 077 = 16(64) + 53: el último lote usa las {LO['resto']} imágenes restantes",
          f"{LO['lotes']} lotes por época × {LO['epocas']} épocas = {mil(LO['actualiz'])} actualizaciones por candidata",
          f"Adam desde w = 0 en J = (w−3)²: primer paso {abs(a1[0]['paso']):.3f}; con J = 100(w−3)² el gradiente inicial es {abs(a100[0]['g']):.0f} en vez de {abs(a1[0]['g']):.0f} y el primer paso sigue siendo {abs(a100[0]['paso']):.3f} (cálculo propio)",
          f"Tres unidades con pesos idénticos 0.5 y x = (1, 2) dan h = {N['simetria']['h'][0]}: las tres activaciones son iguales (diapositiva 28)"],
         "Receta del profesor: Adam, tasa 0.002, lote 64, 120 épocas. <b>Alpha no es la tasa de aprendizaje</b> (en la clase se mencionó «alpha … ese eta»): alpha controla la penalización; la tasa es <font name='Mono'>learning_rate_init</font>.")

    # ---------------------------------------------------------------- 9 curvas y selección
    tema(s, "9", "Leer las curvas y elegir el ancho con la validación",
         f"<b>Cada candidata se entrena 120 épocas y se conserva la época con menor log loss de validación.</b> La elegida fue {N['elegida']} (log loss de validación {f(CV['Red32']['val'], 4)}, época {CV['Red32']['epoca']}) frente a Red16 ({f(CV['Red16']['val'], 4)}, época {CV['Red16']['epoca']}).",
         [("Dos curvas", "Entrenamiento y validación, ambas como log loss sin penalización. Con el tiempo, la de entrenamiento sigue bajando; la de validación se aplana."),
          ("Qué indica una brecha", "Si entrenamiento baja pero validación no, la red empieza a memorizar los datos (sobreajuste)."),
          ("Elegir por validación", "Se comparan los mínimos de validación de cada candidata. No se mira la prueba para elegir, y no se reajusta con entrenamiento + validación."),
          ("Sin afirmar convergencia", "Las épocas elegidas (116 y 118) están cerca del límite de 120: no se afirma que la red haya convergido.")],
         [("c45_curvas.png", "Diapositiva 36: log loss por época (escala logarítmica para ver los detalles). Línea punteada: época elegida."),
          ("c45_seleccion.png", "Izquierda, diapositiva 35. Derecha, nuestra reproducción: misma receta con 2, 4, 8, 16, 32 y 64 unidades ocultas.")],
         "Estudiar para un examen usando <b>solo el libro de ejercicios</b> hace que cada vez lo sepas mejor (curva azul), pero lo que importa es cómo te va en <b>problemas nuevos</b> (curva naranja). "
         "Cuando el libro sigue mejorando y los problemas nuevos no, estás memorizando.",
         None,
         [f"Red16: entrenamiento {f(CV['Red16']['tr_120'], 4)} y validación {f(CV['Red16']['va_120'], 4)} en la época 120 (nuestro cálculo con el programa del profesor)",
          f"Red32: entrenamiento {f(CV['Red32']['tr_120'], 4)} y validación {f(CV['Red32']['va_120'], 4)} en la época 120",
          f"Nuestra reproducción: 2 unidades → {f(AN['2']['val'], 3)}; 4 → {f(AN['4']['val'], 3)}; 8 → {f(AN['8']['val'], 3)}; 16 → {f(AN['16']['val'], 3)}; 32 → {f(AN['32']['val'], 3)}; 64 → {f(AN['64']['val'], 3)} (el mínimo está en 32)",
          "Una semilla y una cuadrícula pequeña no establecen estabilidad general (diapositiva 43)"],
         "Que 32 gane a 16 en esta corrida no prueba que 32 sea mejor siempre; con otra semilla o con más épocas el resultado podría cambiar.")

    # ---------------------------------------------------------------- 10 alpha
    A0, A4, A2, A1, A_, A3 = (AL[k] for k in ("0.0", "0.0001", "0.01", "0.1", "1.0", "3.0"))
    tema(s, "10", "Regularización: el papel de alpha",
         "<b>La penalización de pesos grandes (alpha en scikit-learn) controla cuánto se prefiere una red «suave» frente al ajuste puro a los datos.</b> Poco alpha deja a la red libre; demasiado alpha le impide aprender.",
         [("Qué penaliza", "La suma de los pesos al cuadrado (los sesgos no). Duplicar un peso cuadruplica su aporte a la penalización."),
          ("Poco alpha", "La red se ajusta casi sin freno: el entrenamiento llega a pérdidas muy bajas."),
          ("Mucho alpha", "Los pesos se encogen tanto que la red ya no representa bien los datos: sube la pérdida de entrenamiento y la de validación."),
          ("Se elige con validación", "Como el ancho y la tasa: es un hiperparámetro. La receta del profesor usa alpha = 0.0001 (diapositiva 33).")],
         [("c45_alpha.png", "Nuestra reproducción (no está en las diapositivas): red 64–32–10, misma receta, solo cambia alpha. Izquierda: log loss sin penalización tras 120 épocas. Derecha: suma de los pesos al cuadrado.")],
         "Es como el <b>consejo de una persona prudente</b> que dice «no exageres»: con poco consejo, la red exagera sus pesos para calzar con cada detalle de los datos; con demasiado, la red ya no se atreve a aprender nada. "
         "El punto bueno está en medio.",
         ["J(θ) = promedio de pérdidas + (λ/2)·Σ‖W‖²", "scikit-learn: penalización = α/(2·lote)·ΣW²"],
         [f"alpha = 0: validación {f(A0['val_120'], 4)}; entrenamiento {f(A0['tr_120'], 4)}; suma de W² = {A0['suma_w2']:.0f}",
          f"alpha = 0.0001 (receta): validación {f(A4['val_120'], 4)}; entrenamiento {f(A4['tr_120'], 4)}; suma de W² = {A4['suma_w2']:.0f}",
          f"alpha = 0.01: validación {f(A2['val_120'], 4)}; entrenamiento {f(A2['tr_120'], 4)}; suma de W² = {A2['suma_w2']:.0f}",
          f"alpha = 0.1: validación {f(A1['val_120'], 4)}; alpha = 1: {f(A_['val_120'], 4)}; alpha = 3: {f(A3['val_120'], 4)} (la red casi no aprende)"],
         "En scikit-learn 1.9.1 el término es α/(2·tamaño del lote)·ΣW² (se verificó en el código fuente): el λ de la diapositiva 19 equivale a alpha ÷ lote. La selección usa log loss sin penalización.")

    # ---------------------------------------------------------------- 11 evaluar
    D = EV["red"]["prueba"]
    L_ = EV["log"]["prueba"]
    tema(s, "11", "Evaluar: la red pierde menos, pero la logística se equivoca menos",
         f"<b>En las 360 imágenes de prueba la red tiene menor log loss ({f(D['ll'], 4)} contra {f(L_['ll'], 4)}) y la logística comete menos errores ({L_['errores']} contra {D['errores']}).</b> Las dos métricas miden cosas distintas.",
         [("Prueba histórica", "Estas imágenes ya se mostraron en clase: la comparación es <b>retrospectiva</b>. Para decidir de verdad, la prueba se mira una sola vez, después de fijar la receta."),
          ("Log loss", "Mira las probabilidades: premia la seguridad correcta y castiga la seguridad equivocada."),
          ("Exactitud", "Cuenta aciertos con el umbral que decide la clase (0.5 en binario, el máximo en 10 clases). No mira qué tan segura estaba la predicción."),
          ("Por qué discrepan", "Un modelo puede equivocarse en más casos pero con probabilidades menos extremas, y pierde menos en promedio.")],
         [("c45_comparacion.png", "Diapositiva 37: log loss y errores de la logística y de la red en las mismas 360 imágenes."),
          ("c45_discrepancia.png", "Diapositiva 38: en ambos casos y = 1; A acierta los dos con 0.51 (0 errores) y B acierta uno con 0.99 y falla otro con 0.49 (1 error).")],
         "Dos alumnos responden «verdadero o falso» con su <b>nivel de seguridad</b>. Uno responde todo con «51 % seguro» y acierta todo; el otro dice «99 % seguro» en uno y «49 %» en otro (falla). "
         "El segundo se equivocó una vez, pero su apuesta total estuvo mejor repartida: perdió menos.",
         ["ℓ_A = (−log 0.51 − log 0.51) ÷ 2 ≈ 0.6733", "ℓ_B = (−log 0.99 − log 0.49) ÷ 2 ≈ 0.3617"],
         [f"Exactitud: logística {pc(L_['acc'])} ({L_['errores']} errores); red {pc(D['acc'])} ({D['errores']} errores)",
          f"Log loss: logística {f(L_['ll'], 4)}; red {f(D['ll'], 4)} (diapositiva 37: 0.1511 y 0.0932)",
          f"Validación: log loss logística {f(EV['log']['validacion']['ll'], 4)} y red {f(EV['red']['validacion']['ll'], 4)}",
          f"Ejemplo binario: ℓ_A = {f(ME['A']['perdida'], 4)} con 0 errores; ℓ_B = {f(ME['B']['perdida'], 4)} con 1 error"],
         "El umbral 0.5 decide la clase; la pérdida evalúa las probabilidades. Elegir según una sola métrica puede llevar a conclusiones distintas.")

    # ---------------------------------------------------------------- 12 errores y límites
    tema(s, "12", "Dónde se equivoca la red y qué no demuestra este experimento",
         f"<b>Los {ER['n']} errores de la red se reparten entre pocos pares de dígitos y {ER['n_conf90']} de ellos tiene probabilidad máxima de al menos 0.90.</b> Esos errores seguros son los que más pesan en el log loss.",
         [("Matriz de confusión", "Cada fila es el dígito real y cada columna el predicho; la diagonal son los aciertos. Permite ver qué pares se confunden (9 con 8 y 3; 1 con 6)."),
          ("Errores seguros", f"El error 6 → 1 tiene p máx. {N['errores'][5]['pmax']:.2f}: la pérdida de esa imagen es {N['errores'][5]['perdida']:.2f}. Los {ER['n']} errores suman {f(ER['perdida_errores'], 1)} de los {f(ER['perdida_total'], 1)} de pérdida total ({f(ER['perdida_errores'] / ER['perdida_total'] * 100, 0)} %)."),
          ("Alcance", "Una semilla y una cuadrícula chica no dan estabilidad; la prueba reutilizada es retrospectiva; separar por imagen no acredita escritores nuevos; una red densa no es invariante a desplazamientos."),
          ("Diagnóstico", "Pérdida no finita → revisar datos, escala, tasa. No mejora el entrenamiento → gradientes, capacidad, tasa, objetivo. Mejora solo el entrenamiento → regularización, época, datos. Falla en otro contexto → distribución y partición.")],
         [("c45_confusion.png", "Diapositiva 39: matriz de confusión de la red en la prueba (360 imágenes).", 0.48),
          ("c45_errores_imgs.png", "Diapositivas 40 y 41: los 10 errores de la red en el orden de las filas de prueba (rojo: p máx. ≥ 0.90)."),
          ("c45_errores_p.png", "Probabilidad máxima de cada error: solo el 6 → 1 supera el corte ilustrativo de 0.90.")],
         "Una red que <b>se equivoca con seguridad</b> es como un guía que te indica el camino equivocado con total convicción: por eso, además de contar errores, hay que mirar las imágenes mal clasificadas y su probabilidad. "
         "A veces el error está en la etiqueta del dato, no en el modelo.",
         None,
         [f"Errores (real → predicho): " + ", ".join(f"{e['real']}→{e['pred']}" for e in N["errores"]),
          f"Primeros seis, p máx.: " + ", ".join(f"{e['pmax']:.2f}" for e in N["errores"][:6]) + " (diapositiva 40: 0.64, 0.70, 0.75, 0.52, 0.88, 0.99)",
          "Procedimiento a criticar (diapositiva 45): estandarizar antes de separar, elegir la época con mayor exactitud de <i>prueba</i>, comunicar solo la pérdida de entrenamiento y concluir que generaliza a escritores nuevos",
          "Corrección (diapositiva 46): separar por la unidad relevante antes de estimar transformaciones; ajustar con entrenamiento; elegir con validación; evaluar la receta fijada"],
         "Una probabilidad alta también puede acompañar un error: no sirve como garantía individual.", minimo=50)

    # ---------------------------------------------------------------- repaso
    s.append(PageBreak())
    s.append(band("Repaso, alertas del material y criterio de dominio", "Lo que debes poder explicar sin apuntes antes de la Clase 46 (autoencoders y representaciones)", NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Qué conviene saber al leer las diapositivas y la transcripción</b>", "cellb"),
                  bullets(["<b>La transcripción tenía errores de reconocimiento de voz</b> («redes neuronales» como «reenhorarías» o «regla oral»; «hiperparámetro» como «Iberparámetro»; los decimales sin coma, 0.5 como «05»…). Se corrigieron en la versión limpia; el original se conserva aparte.",
                           "<b>«Alpha» no es la tasa de aprendizaje.</b> En la clase se dijo que alpha «hace referencia a ese eta»; en scikit-learn alpha es la penalización de los pesos y la tasa es <font name='Mono'>learning_rate_init</font> (0.002).",
                           "<b>Agregar una capa no «duplica exponencialmente» los parámetros</b> (comentario oral): añadir una segunda capa de 32 a la 64–32–10 suma 1 056 parámetros (de 2 410 a 3 466).",
                           "<b>En la clase se dijo «64 unidades»</b> al contar las unidades de la capa oculta; son 32 (las 64 son las entradas). Y «todas las capas reciben la misma entrada» debe leerse «todas las unidades de una capa».",
                           "<b>El valor 0.51</b> de la probabilidad del ejemplo (0.509999) se oyó como «0.599»; los números de las diapositivas son los correctos. El paso a «0.41» es 0.4196 redondeado.",
                           "<b>Adam solo se nombró.</b> La explicación de esta guía (inercia + paso adaptativo) es complemento propio, no de la clase.",
                           "<b>Las cifras de R son otra corrida.</b> Las diapositivas usan la ejecución de Python; R entrena por separado y puede diferir."], "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 5))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    pregs = [("Una neurona recibe x = (2, −1), pesos (0.5, 1) y sesgo 0.3. ¿Cuál es a y cuál es la activación ReLU?", f"a = 0.5·2 + 1·(−1) + 0.3 = 0.3; ReLU(0.3) = 0.3."),
             ("¿Cuántos parámetros tiene una red 64–16–10? ¿Y una 64–32–10?", f"1 210 y 2 410: 16(64+1) + 10(16+1) y 32(64+1) + 10(32+1)."),
             ("Softmax de (0, ln 2, 0). ¿Qué probabilidades da?", "Exponenciales (1, 2, 1), suma 4: (0.25, 0.50, 0.25)."),
             ("Si la clase real recibe p = 0.1, ¿cuánto vale la pérdida? ¿Y con p = 0.8?", f"−log 0.1 = {f(N['perdida']['0.1'], 3)}; −log 0.8 = {f(N['perdida']['0.8'], 3)}."),
             ("En J(w) = (w−3)², con η = 0.1 y w = 1, ¿cuánto vale la actualización?", "J′(1) = −4; w⁺ = 1 − 0.1(−4) = 1.4."),
             ("¿Qué calcula la retropropagación y qué hace el optimizador?", "Calcula los gradientes (cuánto influye cada parámetro en la pérdida); el optimizador decide cómo cambiar los parámetros con ellos."),
             ("1 077 imágenes y lote 64. ¿Cuántas actualizaciones hay por época y cuántas en 120 épocas?", "17 (16 lotes de 64 + 1 de 53) y 120 × 17 = 2 040."),
             ("¿Con qué conjunto se elige entre Red16 y Red32, y qué métrica se usa?", "Con validación, por menor log loss sin penalización; la prueba no participa."),
             ("La red tiene menor log loss pero más errores que la logística. ¿Es contradictorio?", "No: el log loss mira las probabilidades y la exactitud solo la clase elegida; pueden diferir.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(pregs, 1)], [7 * mm, 82 * mm, CONTENT_W - 89 * mm]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 46</b>", "cellb"), Spacer(1, 3),
                  bullets(["Calculo a mano la salida de una neurona y de una capa pequeña, con ReLU",
                           "Cuento los parámetros de una arquitectura y explico por qué la activación debe ser no lineal",
                           "Calculo softmax de tres puntajes y la pérdida −log p de la clase real",
                           "Hago varios pasos de descenso de gradiente en una función de una variable y explico el papel de η",
                           "Explico qué hace la retropropagación y por qué hay lotes, épocas y un optimizador",
                           "Elijo entre dos redes con la validación, leo las curvas y explico por qué log loss y exactitud pueden discrepar"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    s.append(Spacer(1, 5))
    s.append(P("Cifras de las diapositivas frente a nuestra reproducción (programa del profesor)", "h3"))
    filas = [["Cifra", "Diapositivas", "Reproducido", "¿Igual?"]] + [[r["item"], r["diapositiva"], r["reproducido"], "sí" if r["igual"] else "NO"] for r in N["comparacion"]]
    s.append(table(filas, [60 * mm, 50 * mm, 50 * mm, CONTENT_W - 160 * mm]))
    s.append(Spacer(1, 3))
    c44 = N["c44_check"]
    s.append(P(f"Nota: la receta de esta clase evaluada en la prueba da {c44['receta_c45_prueba']['aciertos']} aciertos de 360 y log loss {f(c44['receta_c45_prueba']['ll'], 4)}; "
               f"las cifras de la red de la Clase 44 ({c44['diapos_c44']['aciertos']} de 360 y {c44['diapos_c44']['ll']}) <b>no</b> salen de esta receta: son otra corrida.", "small"))
    build("C45_Guia_RedesNeuronales.pdf", s, "Clase 45 · Redes neuronales")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#0F766E">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 45 · Redes neuronales", "Para leer antes de los ejercicios · lectura de ~40 min · pensado para quien mira por primera vez dentro de una red neuronal", C45))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 8 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Una neurona", "es una suma ponderada más un sesgo, pasada por una función de activación."),
                            ("Una capa", "son muchas neuronas a la vez: X·W + b y la activación."),
                            ("La activación no lineal", "es lo que impide que varias capas se reduzcan a una sola."),
                            ("Softmax y −log p", "convierten puntajes en probabilidades y miden el error de una imagen."),
                            ("Aprender es minimizar J", "con descenso de gradiente: w⁺ = w − η·J′(w)."),
                            ("Retropropagación", "calcula los gradientes de todos los pesos con la regla de la cadena."),
                            ("Lotes, épocas y Adam", "organizan el entrenamiento; el optimizador decide el paso."),
                            ("Ancho, alpha y épocas", "son hiperparámetros: se eligen con validación; la prueba se mira al final.")], C45)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación en palabras simples, un <b>ejemplo resuelto con números distintos</b> a los de las diapositivas y los ejercicios, y un resumen «para recordar». Ejercicios en <font name='Mono'>C45_Ejercicios_Cuaderno_RedesNeuronales.pdf</font> y notebook <font name='Mono'>C45_RedesNeuronales.ipynb</font>.", "small"))

    # 1 neurona
    x = np.array([2.0, -1.0, 3.0])
    w = np.array([0.5, 1.0, -0.2])
    b = 0.1
    a = float(w @ x + b)
    s.append(h("1", "La neurona: suma ponderada, sesgo y activación"))
    s += para("Una <b>neurona artificial</b> es una calculadora diminuta con tres piezas: unos <b>pesos</b> (uno por cada entrada), un <b>sesgo</b> (un número que se suma al final) y una <b>función de activación</b>. "
              "Primero multiplica cada entrada por su peso y suma todo: a = w₁x₁ + … + wₚxₚ + b. Luego aplica la función: h = φ(a). Ese h es la <b>activación</b>. "
              "Los pesos y el sesgo son los <b>parámetros</b>: se ajustan al entrenar. Los x son datos y no cambian. Un peso positivo hace que la entrada empuje la suma hacia arriba; uno negativo, hacia abajo; el sesgo permite que la neurona responda aunque todas las entradas valgan 0.")
    s.append(ejemplo("una neurona con 3 entradas", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Productos", "0.5·2; 1·(−1); (−0.2)·3", "1; −1; −0.6"],
               ["Suma + sesgo", "1 − 1 − 0.6 + 0.1", f"a = {a:g}"],
               ["ReLU", "máx(0, −0.5)", "h = 0"],
               ["Sigmoide", "1 ÷ (1 + e^0.5)", f"{sigm(a):.4f}"],
               ["Tangente hiperbólica", "tanh(−0.5)", f"{np.tanh(a):.4f}"]],
              [38 * mm, 62 * mm, CONTENT_W - 100 * mm - 18]),
        P("Lectura: la misma suma a = −0.5 da resultados distintos según la función. ReLU la convierte en 0; la sigmoide la deja en (0, 1); tanh, en (−1, 1). <b>Error común</b>: pensar que h = 0 significa que la neurona «no hizo nada»: calculó un valor y ese valor es 0.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_activaciones.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["a = Σ wⱼxⱼ + b; h = φ(a)", "Pesos y sesgo = parámetros que se aprenden; x = datos", "ReLU(a) = máx(0, a); sigmoide en (0, 1); tanh en (−1, 1)"]))

    # 2 capas y matrices
    X = np.array([[2.0, 1.0, 1.0], [0.0, 3.0, 1.0]])
    W = np.array([[0.5, -1.0], [1.0, 2.0], [-0.5, 0.3]])
    bb = np.array([0.1, 0.2])
    A = X @ W + bb
    Hh = np.maximum(A, 0)
    s.append(h("2", "Una capa: muchas neuronas y un producto de matrices"))
    s += para("Una <b>capa</b> reúne varias neuronas que miran la misma entrada, cada una con sus propios pesos y sesgo. En vez de calcular una por una, se ordenan los pesos en una matriz <b>W</b> con una <b>columna por neurona</b> y se multiplica: A = X·W + b. "
              "Cada <b>fila de X es una observación</b> (una imagen), así que una sola multiplicación procesa muchas imágenes. Las dimensiones deben encajar: (n × p) · (p × q) = (n × q), donde n son las observaciones, p las entradas y q las unidades. "
              "Se mantiene siempre la misma convención (fila por observación), porque cambiarla a mitad de camino daría resultados inconsistentes.")
    s.append(ejemplo("2 observaciones, 3 entradas, 2 unidades", [
        pasos([["Paso", "Fila 1: x = (2, 1, 1)", "Fila 2: x = (0, 3, 1)"],
               ["Unidad 1: pesos (0.5, 1, −0.5), b = 0.1", f"1 + 1 − 0.5 + 0.1 = {A[0, 0]:.1f}", f"0 + 3 − 0.5 + 0.1 = {A[1, 0]:.1f}"],
               ["Unidad 2: pesos (−1, 2, 0.3), b = 0.2", f"−2 + 2 + 0.3 + 0.2 = {A[0, 1]:.1f}", f"0 + 6 + 0.3 + 0.2 = {A[1, 1]:.1f}"],
               ["ReLU", f"h = ({Hh[0, 0]:.1f}, {Hh[0, 1]:.1f})", f"h = ({Hh[1, 0]:.1f}, {Hh[1, 1]:.1f})"]],
              [64 * mm, 50 * mm, CONTENT_W - 114 * mm - 18]),
        P("Lectura: X es 2×3, W es 3×2 y el resultado A es 2×2 (una fila por imagen, una columna por unidad). Con 64 píxeles y 32 unidades sería X(360×64)·W(64×32) = A(360×32). <b>Error común</b>: olvidar que el sesgo se suma a <b>todas</b> las filas.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_capa.png", width=CONTENT_W * 0.88))
    s.append(Spacer(1, 2))
    s += para("¿Y por qué se necesita la activación? Porque sin ella, dos capas seguidas se pueden juntar en una: (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂), que es otra función lineal. "
              "En el ejemplo de la guía, con x = (3, −1), dos capas lineales y la capa única equivalente dan el mismo 13.5. La activación <b>no lineal</b> rompe esa equivalencia: es la que permite que una red aprenda formas que ninguna recta captura, como separar los puntos de la diapositiva 11.")
    s.append(recuerda(["Una columna de W = una neurona; una fila de X = una observación", "(n×p)·(p×q) = (n×q); el sesgo se suma a todas las filas", "Sin activación no lineal, varias capas equivalen a una"]))

    # 3 arquitectura y parámetros
    s.append(h("3", "Arquitectura, hiperparámetros y conteo de parámetros"))
    s += para("La <b>arquitectura</b> describe cuántas capas hay y cuántas neuronas tiene cada una, leyendo de izquierda a derecha. «64–32–10» significa: 64 entradas, una capa oculta de 32 unidades y 10 salidas. "
              "El número de unidades es el <b>ancho</b>; el de capas, la <b>profundidad</b>. Estos números son <b>hiperparámetros</b>: hay que fijarlos <i>antes</i> de ajustar con los datos y no existe una regla que dé el valor óptimo; se prueban. "
              "No se confunden con los <b>parámetros</b> (pesos y sesgos), que sí se ajustan. Contar parámetros es fácil: cada unidad tiene (número de entradas + 1) parámetros, donde el +1 es el sesgo.")
    n1 = 8 * 5 + 5
    n2 = 5 * 3 + 3
    s.append(ejemplo("una red 8–5–3", [
        pasos([["Capa", "Cálculo", "Parámetros"],
               ["Entrada → oculta (5 unidades)", "5 · (8 + 1) = 5 · 9", f"{n1}"],
               ["Oculta → salida (3 unidades)", "3 · (5 + 1) = 3 · 6", f"{n2 - 0}"],
               ["Total", f"{n1} + {n2}", f"{n1 + n2}"]],
              [60 * mm, 56 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: ocurre lo mismo que con 64–32–10 (2 410 parámetros): una capa más ancha o una capa adicional aumenta muchísimo el número de cosas por ajustar, y con ello el tiempo y el riesgo de sobreajuste. <b>Error común</b>: contar las 64 entradas o las 32 activaciones como parámetros; son valores que se calculan, no se aprenden.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_red.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Parámetros: pesos y sesgos (se aprenden). Hiperparámetros: capas, ancho, tasa, lote, alpha, épocas (se fijan antes)", "Por unidad: (entradas + 1) parámetros", "64–32–10 tiene 2 410; 64–16–10, 1 210"]))

    # 4 softmax y pérdida
    sc = np.array([2.0, 1.0, 0.0])
    ex = np.exp(sc)
    pr = ex / ex.sum()
    s.append(h("4", "La salida: softmax y la pérdida −log p"))
    s += para("Para clasificar en 10 dígitos excluyentes, la última capa entrega 10 puntajes (uno por clase) que pueden ser cualquier número. <b>Softmax</b> los convierte en probabilidades: exponencia cada puntaje (así todos quedan positivos) y divide por la suma. "
              "Resultado: números entre 0 y 1 que suman 1. Para hacerlo estable se resta el máximo c de los puntajes antes de exponenciar; las probabilidades no cambian. "
              "Después, la <b>pérdida de una imagen</b> es −log de la probabilidad que se asignó a la clase <b>real</b> (no a la más probable). La pérdida del conjunto es el promedio.")
    s.append(ejemplo("puntajes s = (2, 1, 0); la clase real es la tercera", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Exponenciales", "e², e¹, e⁰", f"{ex[0]:.3f}; {ex[1]:.3f}; {ex[2]:.3f}"],
               ["Suma", f"{ex[0]:.3f} + {ex[1]:.3f} + {ex[2]:.3f}", f"{ex.sum():.3f}"],
               ["Probabilidades", "cada exponencial ÷ suma", f"{pr[0]:.3f}; {pr[1]:.3f}; {pr[2]:.3f}"],
               ["Comprobar", "suma de las tres", f"{pr.sum():.1f}"],
               ["Pérdida (real = 3.ª)", f"−log({pr[2]:.3f})", f"{-log(pr[2]):.3f}"],
               ["Con el truco", "restar 2: (0, −1, −2)", f"mismas probabilidades"]],
              [36 * mm, 58 * mm, CONTENT_W - 94 * mm - 18]),
        P(f"Lectura: el modelo apostó {pr[2]:.0%} a la clase real y por eso la pérdida es alta ({-log(pr[2]):.3f}); si hubiera sido la primera clase, habría sido {-log(pr[0]):.3f}. <b>Error común</b>: calcular la pérdida con la probabilidad máxima en vez de la de la clase real.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_perdida.png", width=CONTENT_W * 0.82))
    s.append(Spacer(1, 2))
    s += para("La elección de salida y pérdida depende de la tarea: <b>regresión</b> (un número; error cuadrático), <b>clasificación binaria</b> (una sigmoide; entropía cruzada binaria) y <b>clases excluyentes</b> (softmax; entropía cruzada multiclase). "
              "Una función ReLU no sirve como salida de probabilidades porque no está acotada entre 0 y 1.")
    s.append(recuerda(["softmax: pₖ = e^sₖ ÷ Σ e^sⱼ; suma 1", "pérdida = −log p(clase real); promedio = log loss", "La salida y la pérdida deben corresponder a la tarea"]))

    # 5 aprender: gradiente
    s.append(h("5", "Aprender: minimizar J con descenso de gradiente"))
    s += para("Aprender significa encontrar los pesos y sesgos θ que hacen pequeña una función objetivo J(θ). En la clase, J es la <b>pérdida media</b> más una <b>penalización</b> por pesos grandes (λ/2 por la suma de los pesos al cuadrado). "
              "Como J depende de miles de parámetros, se usa el <b>gradiente</b>: el vector con la pendiente de J respecto de cada parámetro. Si la pendiente de un parámetro es negativa, aumentar ese parámetro baja J; si es positiva, hay que disminuirlo. "
              "La regla es <b>w⁺ = w − η·J′(w)</b>, donde η es la <b>tasa de aprendizaje</b>: cuánto se avanza en cada paso.")
    eta, w_ = 0.2, 0.0
    fila = []
    J = lambda v: (v - 5) ** 2
    for k in range(4):
        g = 2 * (w_ - 5)
        wn = w_ - eta * g
        fila.append([str(k + 1), f"{w_:g}", f"{J(w_):g}", f"2({w_:g} − 5) = {g:g}", f"{w_:g} − 0.2({g:g}) = {wn:g}", f"{J(wn):.4f}".rstrip("0").rstrip(".")])
        w_ = wn
    s.append(ejemplo("J(w) = (w − 5)², η = 0.2, empezando en w = 0", [
        pasos([["Paso", "w", "J(w)", "J′(w)", "nuevo w", "J(nuevo w)"]] + fila, [12 * mm, 14 * mm, 20 * mm, 38 * mm, 48 * mm, CONTENT_W - 132 * mm - 18]),
        P("Lectura: en cada paso el error |w − 5| se multiplica por 1 − 2η = 0.6, y J por 0.36: baja de forma regular. Con η = 0.5 llegaría de golpe al mínimo y con η > 1 se alejaría (J crece). <b>Error común</b>: creer que un η más grande siempre acelera: pasado cierto punto, empeora.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_gd.png", width=CONTENT_W * 0.85))
    s.append(Spacer(1, 2))
    s += para("La penalización mira la <b>magnitud</b> de los pesos: como (−w)² = w² y (2w)² = 4w², el signo no importa y duplicar un peso cuadruplica su aporte. El parámetro que controla cuánto pesa esa preferencia frente al ajuste a los datos se llama λ en la diapositiva y <b>alpha</b> en scikit-learn "
              "(el código de scikit-learn 1.9.1 suma α/(2·tamaño del lote) por la suma de pesos al cuadrado, es decir, λ ≈ α ÷ lote). Alpha no es la tasa de aprendizaje.")
    s.append(recuerda(["Gradiente = pendiente de J respecto de cada parámetro", "w⁺ = w − η·J′(w): η es el tamaño del paso", "J = pérdida media + penalización de pesos (alpha); alpha ≠ η"]))

    # 6 retropropagación
    x_, y_ = 1.0, 0
    th = np.array([0.6, -0.1, 0.8, -0.3])
    def fwd(t):
        a1 = t[0] * x_ + t[1]
        hh = max(a1, 0)
        a2 = t[2] * hh + t[3]
        p = 1 / (1 + np.exp(-a2))
        return a1, hh, a2, p, float(-np.log(1 - p if y_ == 0 else p))
    a1, hh, a2, p, l0 = fwd(th)
    d2 = p - y_
    d1 = d2 * th[2] * float(a1 > 0)
    gr = np.array([d1 * x_, d1, d2 * hh, d2])
    th2 = th - 0.1 * gr
    _, _, _, p2, l2 = fwd(th2)
    s.append(h("6", "Retropropagación: calcular todos los gradientes con la regla de la cadena"))
    s += para("La pérdida depende de la probabilidad p, esta de a₂, esta de h, esta de a₁ y esta de los pesos: es una <b>cadena</b> de funciones. La <b>regla de la cadena</b> dice que la derivada de una cadena es el producto de las derivadas de cada eslabón. "
              "La <b>retropropagación</b> aprovecha eso: primero calcula hacia adelante (propagación) y luego recorre la cadena hacia atrás, reutilizando cada δ para el eslabón anterior. "
              "Con sigmoide y entropía cruzada binaria, δ₂ = p − y. Para la capa oculta, δ₁ = δ₂·w₂·ReLU′(a₁), donde ReLU′ vale 1 si a₁ > 0 y 0 si no. Y el gradiente de cada peso es el δ de su unidad por la entrada de ese peso.")
    s.append(ejemplo("x = 1, y = 0 y θ = (w₁, b₁, w₂, b₂) = (0.6, −0.1, 0.8, −0.3)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["a₁, h", "0.6·1 − 0.1; ReLU", f"{a1:.2f}; {hh:.2f}"],
               ["a₂, p", f"0.8·{hh:.2f} − 0.3; σ(a₂)", f"{a2:.2f}; {p:.4f}"],
               ["Pérdida (y = 0)", f"−log(1 − {p:.4f})", f"{l0:.4f}"],
               ["δ₂ = p − y", f"{p:.4f} − 0", f"{d2:.4f}"],
               ["∂ℓ/∂w₂; ∂ℓ/∂b₂", f"δ₂·h; δ₂", f"{gr[2]:.4f}; {gr[3]:.4f}"],
               ["δ₁", f"{d2:.4f}·0.8·1", f"{d1:.4f}"],
               ["∂ℓ/∂w₁; ∂ℓ/∂b₁", "δ₁·x; δ₁", f"{gr[0]:.4f}; {gr[1]:.4f}"],
               ["Actualizar (η = 0.1)", "θ − 0.1·gradiente", "(" + "; ".join(f"{v:.4f}" for v in th2) + ")"],
               ["Nueva pérdida", f"p⁺ = {p2:.4f}", f"{l2:.4f} (baja)"]],
              [38 * mm, 58 * mm, CONTENT_W - 96 * mm - 18]),
        P("Lectura: como y = 0, se quiere bajar p; los cuatro gradientes son positivos, así que restarlos reduce los cuatro parámetros y la pérdida baja de " + f"{l0:.4f} a {l2:.4f}. <b>Error común</b>: olvidar la entrada (x o h) al calcular el gradiente de un peso; el del sesgo, en cambio, es solo δ.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_backprop.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Primero hacia adelante (valores), luego hacia atrás (sensibilidades δ)", "δ₂ = p − y; δ₁ = δ₂·w₂·ReLU′(a₁); gradiente del peso = δ × entrada", "Backpropagation calcula; el optimizador actualiza"]))

    # 7 lotes y Adam
    n_, B_ = 1500, 32
    comp, resto = n_ // B_, n_ % B_
    nb_ = comp + (1 if resto else 0)
    s.append(h("7", "Lotes, épocas, optimizador Adam e inicialización"))
    s += para("Calcular el gradiente con todas las imágenes en cada paso es caro; hacerlo con una sola imagen es ruidoso. El punto medio son los <b>lotes</b>. Un <b>lote</b> es un grupo de imágenes; una <b>actualización</b> cambia los parámetros usando un lote; "
              "una <b>época</b> es una pasada por todas las imágenes de entrenamiento. Si el número de imágenes no es múltiplo del lote, el último lote es más pequeño.")
    s.append(ejemplo(f"{mil(n_)} imágenes de entrenamiento, lote {B_}, 30 épocas", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Lotes completos", f"{n_} ÷ {B_}", f"{comp} lotes (sobran {resto})"],
               ["Lote final", f"{n_} − {comp}·{B_}", f"{resto} imágenes"],
               ["Actualizaciones por época", f"{comp} + 1", f"{nb_}"],
               ["Actualizaciones en 30 épocas", f"30 × {nb_}", f"{30 * nb_}"]],
              [50 * mm, 52 * mm, CONTENT_W - 102 * mm - 18]),
        P("Lectura: con 1 077 imágenes y lote 64 de la clase son 16 lotes de 64 más uno de 53, es decir, 17 actualizaciones por época y 2 040 en 120 épocas. <b>Error común</b>: confundir lote con época: un lote es una parte; la época es una pasada completa.", "cell")]))
    s.append(Spacer(1, 3))
    a1_, a2_ = AD["pasos1"], AD["pasos100"]
    s += para("<b>Adam</b> es un optimizador: la regla que usa los gradientes para decidir el cambio de cada parámetro. En vez de restar η por el gradiente, lleva dos promedios móviles: m (el gradiente reciente, la «inercia») y v (el cuadrado del gradiente, su «tamaño típico»), y da el paso η·m̂ ÷ (√v̂ + ε). "
              "Resultado: cada parámetro avanza aproximadamente η por actualización, sin importar cuán grande o pequeño sea su gradiente. La clase solo nombró a Adam (tasa 0.002); este ejemplo es una ilustración propia.")
    s.append(ejemplo("primer paso de Adam desde w = 0, η = 0.1, con J = (w−3)² y con J = 100(w−3)²", [
        pasos([["Cantidad", "J = (w−3)²", "J = 100(w−3)²"],
               ["Gradiente g", f"{a1_[0]['g']:.0f}", f"{a2_[0]['g']:.0f}"],
               ["m̂ (inercia corregida)", f"{a1_[0]['mh']:.0f}", f"{a2_[0]['mh']:.0f}"],
               ["√v̂ (tamaño típico)", f"{sqrt(a1_[0]['vh']):.0f}", f"{sqrt(a2_[0]['vh']):.0f}"],
               ["Paso de Adam = η·m̂/√v̂", f"{abs(a1_[0]['paso']):.3f}", f"{abs(a2_[0]['paso']):.3f}"],
               ["Paso del descenso simple = η·g", f"{abs(0.1 * a1_[0]['g']):.1f}", f"{abs(0.1 * a2_[0]['g']):.1f}"]],
              [60 * mm, 46 * mm, CONTENT_W - 106 * mm - 18]),
        P("Lectura: Adam da el mismo paso (≈ 0.1) en ambos casos; el descenso simple pasa de 0.6 a 60, y se desborda en la segunda función (figura de la guía). <b>Error común</b>: pensar que Adam siempre converge más rápido: en la función benigna avanza más lento que el descenso simple.", "cell")]))
    s.append(Spacer(1, 3))
    s += para("<b>Inicialización.</b> Hay que partir de algún valor. Si todos los pesos de varias unidades son iguales, esas unidades calculan lo mismo y reciben el mismo gradiente: nunca se diferencian. Por ejemplo, con tres unidades de pesos 0.5 y x = (1, 2), las tres activaciones valen 1.5. "
              "Por eso se inicializa con valores distintos (la semilla fija la inicialización y el orden de las imágenes). Los pesos iniciales malos pueden hacer que la red tarde más o termine en un mínimo local; el descenso de gradiente es una búsqueda local, y no garantiza un óptimo global.")
    s.append(recuerda(["Lote → actualización; época = pasada completa; el último lote puede ser menor", "Adam ≈ paso de tamaño η por parámetro, adaptado a la escala del gradiente", "Pesos iniciales iguales → unidades idénticas; la semilla fija la inicialización"]))

    # 8 curvas, ancho, alpha
    s.append(h("8", "Leer curvas, elegir el ancho y ajustar alpha"))
    s += para("En cada época se mide el log loss <b>sin penalización</b> en entrenamiento y en validación. Normalmente ambas bajan al comienzo; después la de entrenamiento sigue bajando mientras la de validación se aplana (y puede subir). La red se entrena 120 épocas y se <b>conserva la época con menor pérdida de validación</b> (se guarda una copia). "
              "Con dos anchos candidatos (16 y 32), se comparan sus mínimos de validación y se elige el menor, sin tocar la prueba ni reajustar con entrenamiento + validación.")
    r16, r32 = CV["Red16"], CV["Red32"]
    s.append(ejemplo("las curvas de la corrida del profesor (log loss)", [
        pasos([["Época", "Red16 · entr.", "Red16 · valid.", "Red32 · entr.", "Red32 · valid."],
               ["1", f(r16["tr_1"], 3), f(r16["va_1"], 3), f(r32["tr_1"], 3), f(r32["va_1"], 3)],
               ["20", f(r16["tr_20"], 3), f(r16["va_20"], 3), f(r32["tr_20"], 3), f(r32["va_20"], 3)],
               ["120", f(r16["tr_120"], 4), f(r16["va_120"], 4), f(r32["tr_120"], 4), f(r32["va_120"], 4)],
               ["Mínimo de validación", "", f"{r16['val']:.4f} (época {r16['epoca']})", "", f"{r32['val']:.4f} (época {r32['epoca']})"]],
              [34 * mm, 34 * mm, 38 * mm, 34 * mm, CONTENT_W - 140 * mm - 18]),
        P("Lectura: en la época 120 el entrenamiento de Red32 llega a 0.0122 y la validación se queda en 0.0635: la brecha crece, pero la validación sigue bajando lentamente hasta la época 118. Se elige Red32 por su menor mínimo (0.0633 frente a 0.0894). <b>Error común</b>: afirmar convergencia cuando el mínimo está justo al final del presupuesto de épocas.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_curvas.png", width=CONTENT_W * 0.88))
    s.append(Spacer(1, 3))
    filas = [["alpha", "Entr. (ép. 120)", "Valid. (ép. 120)", "Suma de W²"]]
    for k in ("0.0", "0.0001", "0.01", "0.1", "1.0", "3.0"):
        v = AL[k]
        filas.append(["0" if float(k) == 0 else f"{float(k):g}", f(v["tr_120"], 4), f(v["val_120"], 4), f"{v['suma_w2']:.0f}"])
    s += para("<b>El efecto de alpha</b> (nuestra reproducción con la red 64–32–10, no es de las diapositivas): una penalización moderada reduce los pesos y mantiene o mejora la validación; una excesiva impide que la red aprenda.")
    s.append(table(filas, [30 * mm, 42 * mm, 42 * mm, CONTENT_W - 114 * mm - 18]))
    s.append(Spacer(1, 3))
    s += para("Con el ancho ocurre algo parecido: en nuestra reproducción, con 2 unidades la red aprende muy poco (validación 0.965), con 8 llega a 0.117, con 32 a 0.063 y con 64 vuelve a subir un poco (0.067). Elegir el ancho es un compromiso entre capacidad y ajuste, y se decide con validación.")
    s.append(recuerda(["Elegir con validación (menor log loss sin penalización); la prueba, al final", "Una brecha creciente entrenamiento–validación indica sobreajuste", "Alpha moderado ayuda; demasiado impide aprender"]))

    # 9 evaluar
    s.append(h("9", "Evaluar: log loss, errores, confusión y alcance"))
    s += para("Después de elegir la receta, se evalúa con la prueba y se compara con una referencia (la regresión logística). En la clase, la red tuvo log loss 0.0932 y la logística 0.1511, pero la red cometió 10 errores y la logística 8. "
              "La <b>exactitud</b> cuenta aciertos con el umbral que decide la clase; el <b>log loss</b> evalúa qué probabilidad se dio a la clase real. Un modelo puede equivocarse más veces pero con menos seguridad en sus errores y perder menos.")
    pa, pb = [0.60, 0.60], [0.95, 0.45]
    la, lb = np.mean([-log(v) for v in pa]), np.mean([-log(v) for v in pb])
    s.append(ejemplo("dos modelos en dos casos con y = 1 (umbral 0.5)", [
        pasos([["Modelo", "Probabilidades de la clase 1", "Errores", "Cálculo de la pérdida", "Pérdida"],
               ["A", "0.60 y 0.60", "0", "(−log 0.60 − log 0.60) ÷ 2", f"{la:.4f}"],
               ["B", "0.95 y 0.45", "1", "(−log 0.95 − log 0.45) ÷ 2", f"{lb:.4f}"]],
              [14 * mm, 44 * mm, 18 * mm, 58 * mm, CONTENT_W - 134 * mm - 18]),
        P(f"Lectura: A no se equivoca nunca y B se equivoca una vez, pero la pérdida de B ({lb:.4f}) es menor que la de A ({la:.4f}): B está muy seguro cuando acierta. Por eso elegir solo por exactitud o solo por log loss puede dar conclusiones distintas. <b>Error común</b>: creer que menos errores implica siempre menos pérdida.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c45_comparacion.png", width=CONTENT_W * 0.85))
    s.append(Spacer(1, 3))
    s += para("La <b>matriz de confusión</b> detalla qué se confunde con qué: la fila es el dígito real, la columna el predicho. En la red hay 10 errores en 360 imágenes: 9→8, 1→6, 1→6, 7→8, 5→6, 6→1, 8→4, 9→3, 9→3, 3→2. "
              "Un error seguro como el 6→1 (probabilidad 0.99) pesa mucho en el log loss; por eso conviene mirar las imágenes mal clasificadas. También puede haber errores en la etiqueta del dato.")
    s += para("<b>Alcance.</b> Una semilla y una cuadrícula pequeña no establecen estabilidad; habría que repetir con varias semillas. La prueba usada ya se mostró en clase, así que la comparación es retrospectiva. Separar por imagen no demuestra desempeño con escritores nuevos. "
              "Una red densa no es invariante a desplazamientos: correr la imagen una columna cambia por completo las coordenadas que recibe. Y la receta correcta (diapositiva 46) es: separar por la unidad relevante antes de estimar transformaciones, ajustar con entrenamiento, elegir receta y época con validación, y evaluar lo fijado.")
    s.append(recuerda(["Log loss (probabilidades) y exactitud (clase elegida) pueden discrepar", "Mirar la matriz de confusión y los errores seguros", "La prueba reutilizada es retrospectiva; la evaluación debe ajustarse al uso real"]))

    # glosario
    s.append(h("10", "Glosario rápido"))
    gl = [["Término", "Qué es"],
          ["Neurona / unidad", "Calcula a = Σwx + b y entrega h = φ(a)."],
          ["Peso, sesgo", "Parámetros que se aprenden; el sesgo desplaza la suma."],
          ["Activación", "El resultado h de una unidad para una entrada. No es una probabilidad."],
          ["Capa oculta, ancho, profundidad", "Capas intermedias; unidades por capa; número de capas."],
          ["Hiperparámetro", "Se fija antes de entrenar: ancho, capas, η, lote, alpha, épocas."],
          ["Softmax", "Convierte puntajes en probabilidades que suman 1."],
          ["Pérdida, log loss", "−log p de la clase real; promedio sobre las imágenes."],
          ["Gradiente, η", "Pendiente de J por parámetro; η es el tamaño del paso."],
          ["Retropropagación", "Cálculo de gradientes con la regla de la cadena."],
          ["Lote, actualización, época", "Grupo de imágenes; un cambio de parámetros; una pasada completa."],
          ["Adam", "Optimizador con inercia y paso adaptado a la escala del gradiente."],
          ["Alpha", "Penalización L2 de los pesos en scikit-learn (no es la tasa)."],
          ["Sobreajuste", "El entrenamiento mejora, pero la validación no."]]
    s.append(table(gl, [52 * mm, CONTENT_W - 52 * mm - 18]))
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 3]))
            i += 3
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C45_Apuntes_RedesNeuronales.pdf", out, "Apuntes · Clase 45 · Redes neuronales")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0F766E">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C45))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 45 · Redes neuronales", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C45))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: a = Σ wⱼxⱼ + b · ReLU(a) = máx(0, a) · σ(a) = 1 ÷ (1 + e⁻ᵃ) · softmax: pₖ = e^sₖ ÷ Σ e^sⱼ · pérdida = −ln p(clase real) · parámetros por capa = unidades × (entradas + 1) · w⁺ = w − η·J′(w) · gradiente de un peso = δ × entrada, con δ₂ = p − y.",
                           "Valores útiles: ln 2 = 0.6931 · ln 10 = 2.3026 · e⁰·⁵ = 1.6487 · e⁻⁰·⁵ = 0.6065 · e¹ = 2.7183.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C45)], bg=SOFT, border=C45, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "Una neurona", "★", "Una neurona tiene pesos w = (2, −1, 0.5), sesgo b = −0.5 y usa ReLU.", [
        "Para x = (1, 3, 2) calcula a y la activación h.",
        "Para x = (0.5, 1, 0) calcula a y h. ¿Qué pasó con la activación?",
        "¿Qué sesgo haría que, con x = (0.5, 1, 0), la activación fuera exactamente 0.3?"])
    s += bloque(2, "Propagar por una red pequeña", "★★", "Red de 2 entradas, 2 unidades ocultas con ReLU y 1 salida sigmoide. W₁ = [[1, 2], [−1, 1]] (columnas = unidades), b₁ = (0, −1), W₂ = (1, −1)ᵀ, b₂ = 0.2. Entrada x = (2, 1).", [
        "Calcula a₁ y la capa oculta h = ReLU(a₁).",
        "Calcula el puntaje s = hW₂ + b₂ y la probabilidad p = σ(s).",
        "Si la clase real es 1, ¿cuánto vale la pérdida −ln p?"])
    s += bloque(3, "Softmax", "★★", "Una red de 3 clases entrega los puntajes s = (1, 0, −1).", [
        "Calcula los exponenciales y su suma.",
        "Calcula las tres probabilidades y verifica que suman 1.",
        "Resta el máximo a los puntajes y comprueba que las probabilidades no cambian."])
    s += bloque(4, "Pérdida de una predicción", "★", "Un clasificador da la probabilidad indicada a la clase REAL de cada imagen.", [
        "Calcula −ln(p) para p = 0.9, 0.4 y 0.05.",
        "Para cuatro imágenes con p real = 0.9, 0.7, 0.4 y 0.05, calcula el log loss.",
        "¿Cuántas veces mayor es la pérdida con p = 0.05 que con p = 0.9? ¿Qué enseña?"])
    s += bloque(5, "Contar parámetros", "★★", None, [
        "Una red 10–6–3: ¿cuántos parámetros tiene?",
        "Una red 64–20–10: ¿cuántos? Compárala con la 64–32–10 (2 410).",
        "Una red 8–5–4–3 (dos capas ocultas): ¿cuántos parámetros hay en total?"])
    s += bloque(6, "Descenso de gradiente a mano", "★★", "Se minimiza J(w) = (w − 4)² con tasa η = 0.25, empezando en w = 0.", [
        "Calcula J′(w) y haz tres actualizaciones: w₁, w₂, w₃.",
        "Calcula J en cada paso. ¿Por qué factor se multiplica el error |w − 4| en cada actualización?",
        "Repite un paso con η = 1.2 desde w = 0 y con η = 0.5. ¿Qué pasa con cada uno?"])
    s += bloque(7, "Lotes y épocas", "★", "El entrenamiento tiene 1 077 imágenes y el lote es 64.", [
        "¿Cuántos lotes completos hay y cuántas imágenes tiene el último lote?",
        "¿Cuántas actualizaciones hay por época y en 120 épocas?",
        "Con 900 imágenes y lote 32: ¿cuántas actualizaciones por época? ¿Y en 50 épocas?"])
    s += bloque(8, "Un paso de retropropagación", "★★★", "Red con una entrada, una unidad oculta ReLU y una salida sigmoide. x = 2, y = 0, parámetros (w₁, b₁, w₂, b₂) = (0.5, −0.5, 1, 0).", [
        "Propaga: a₁, h, a₂, p y la pérdida −ln(1 − p).",
        "Calcula δ₂ = p − y, δ₁ = δ₂·w₂·ReLU′(a₁) y los cuatro gradientes.",
        "Actualiza con η = 0.1 y recalcula p y la pérdida: ¿bajó?"])
    s += bloque(9, "Leer curvas y elegir", "★★", "Dos redes se entrenan 100 épocas. Log loss de validación en su mejor época y en la época 100: A: 0.20 en la época 60 y 0.25 en la 100; B: 0.15 en la época 100 (aún bajando). Entrenamiento en la época 100: A 0.02, B 0.10.", [
        "¿Qué conclusión sacas de A? ¿Qué señal de sobreajuste hay?",
        "¿Con qué red te quedarías según la validación y qué época conservarías en cada una?",
        "¿Qué no puedes afirmar sobre B?"])
    s += bloque(10, "Exactitud frente a log loss", "★★", "Dos modelos, A y B, evalúan 4 imágenes (clase real = 1). Probabilidad que dan a la clase 1: A: 0.55, 0.55, 0.55, 0.55. B: 0.99, 0.99, 0.99, 0.30.", [
        "Calcula los errores y la exactitud de cada modelo con umbral 0.5.",
        "Calcula el log loss de cada uno.",
        "¿Cuál es mejor según cada métrica? Explica la discrepancia."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C45))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [KeepTogether([P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C45), Spacer(1, 2)])]

    # 1
    w1 = np.array([2.0, -1.0, 0.5])
    a_1 = float(w1 @ np.array([1, 3, 2]) - 0.5)
    a_2 = float(w1 @ np.array([0.5, 1, 0]) - 0.5)
    s += sol("Bloque 1", [
        ["1", f"a = 2·1 + (−1)·3 + 0.5·2 − 0.5 = 2 − 3 + 1 − 0.5 = {a_1:g}; h = máx(0, {a_1:g}) = 0."],
        ["2", f"a = 2·0.5 − 1·1 + 0 − 0.5 = 1 − 1 − 0.5 = {a_2:g}; h = máx(0, {a_2:g}) = 0. La suma es negativa y ReLU la lleva a 0: la neurona entrega 0."],
        ["3", "Se necesita a = 0.3 (con a ≥ 0 ReLU deja pasar a): 2·0.5 − 1·1 + 0 + b = 0.3 → b = 0.3. Con b = 0.3 la activación es 0.3."]])
    # 2
    xx = np.array([2.0, 1.0])
    W1 = np.array([[1.0, 2.0], [-1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    a1v = xx @ W1 + b1
    h1 = np.maximum(a1v, 0)
    s_ = float(h1 @ np.array([1.0, -1.0]) + 0.2)
    p_ = sigm(s_)
    s += sol("Bloque 2", [
        ["1", f"a₁ = (2·1 + 1·(−1) + 0, 2·2 + 1·1 − 1) = ({a1v[0]:g}, {a1v[1]:g}). h = ReLU = ({h1[0]:g}, {h1[1]:g})."],
        ["2", f"s = {h1[0]:g}·1 + {h1[1]:g}·(−1) + 0.2 = {s_:g}; p = σ({s_:g}) = 1 ÷ (1 + e^−{s_:g}) = {p_:.4f}."],
        ["3", f"−ln({p_:.4f}) = {-log(p_):.4f}."]])
    # 3
    ex = np.exp(np.array([1.0, 0.0, -1.0]))
    pr = ex / ex.sum()
    s += sol("Bloque 3", [
        ["1", f"e¹ = {ex[0]:.4f}; e⁰ = {ex[1]:.4f}; e⁻¹ = {ex[2]:.4f}. Suma = {ex.sum():.4f}."],
        ["2", f"p = ({pr[0]:.4f}, {pr[1]:.4f}, {pr[2]:.4f}); suma = {pr.sum():.1f}."],
        ["3", "Restando 1: (0, −1, −2) → exponenciales (1, 0.3679, 0.1353), suma 1.5032 → p = (0.6652, 0.2447, 0.0900): las mismas."]])
    # 4
    l1 = [-log(p) for p in (0.9, 0.4, 0.05)]
    l2 = [-log(p) for p in (0.9, 0.7, 0.4, 0.05)]
    s += sol("Bloque 4", [
        ["1", f"−ln 0.9 = {l1[0]:.4f}; −ln 0.4 = {l1[1]:.4f}; −ln 0.05 = {l1[2]:.4f}."],
        ["2", f"Pérdidas: {', '.join(f'{v:.4f}' for v in l2)}; suma = {sum(l2):.4f}; log loss = {sum(l2):.4f} ÷ 4 = {np.mean(l2):.4f}."],
        ["3", f"{l1[2]:.4f} ÷ {l1[0]:.4f} = {l1[2] / l1[0]:.1f} veces. Un error muy seguro cuesta muchísimo más que un acierto razonable."]])
    # 5
    p1 = 6 * (10 + 1) + 3 * (6 + 1)
    p2 = 20 * (64 + 1) + 10 * (20 + 1)
    p3 = 5 * (8 + 1) + 4 * (5 + 1) + 3 * (4 + 1)
    s += sol("Bloque 5", [
        ["1", f"6·(10 + 1) + 3·(6 + 1) = 66 + 21 = {p1}."],
        ["2", f"20·65 + 10·21 = 1 300 + 210 = {mil(p2)}; menos que la 64–32–10 ({mil(2410)}): {mil(2410 - p2)} menos."],
        ["3", f"5·(8 + 1) + 4·(5 + 1) + 3·(4 + 1) = 45 + 24 + 15 = {p3}."]])
    # 6
    ws = [0.0]
    for _ in range(3):
        ws.append(ws[-1] - 0.25 * 2 * (ws[-1] - 4))
    def paso6(w0, e):
        return w0 - e * 2 * (w0 - 4)
    s += sol("Bloque 6", [
        ["1", f"J′(w) = 2(w − 4). w₁ = 0 − 0.25·2(0 − 4) = {ws[1]:g}; w₂ = {ws[1]:g} − 0.25·2({ws[1]:g} − 4) = {ws[2]:g}; w₃ = {ws[2]:g} − 0.25·2({ws[2]:g} − 4) = {ws[3]:g}."],
        ["2", f"J: {', '.join(f'{(v - 4) ** 2:g}' for v in ws)}. El error |w − 4| pasa de 4 a 2, 1, 0.5: se multiplica por 1 − 2η = 0.5 (y J por 0.25)."],
        ["3", f"η = 1.2: w₁ = {paso6(0, 1.2):g} (error pasa de 4 a 5.6: se aleja; factor |1 − 2.4| = 1.4 > 1). η = 0.5: w₁ = {paso6(0, 0.5):g}, justo en el mínimo (factor 0)."]])
    # 7
    s += sol("Bloque 7", [
        ["1", f"1 077 ÷ 64 = 16.83: 16 lotes completos (1 024 imágenes) y el último tiene 1 077 − 1 024 = 53."],
        ["2", "16 + 1 = 17 actualizaciones por época; 120 × 17 = 2 040."],
        ["3", f"900 ÷ 32 = 28.125: 28 lotes completos (896) + 1 de 4 imágenes = 29 por época; 50 × 29 = {50 * 29}."]])
    # 8
    x8, w1_, b1_, w2_, b2_ = 2.0, 0.5, -0.5, 1.0, 0.0
    a1 = w1_ * x8 + b1_
    h8 = max(a1, 0)
    a2 = w2_ * h8 + b2_
    p8 = sigm(a2)
    l8 = -log(1 - p8)
    d2 = p8
    d1 = d2 * w2_ * float(a1 > 0)
    g8 = [d1 * x8, d1, d2 * h8, d2]
    t2 = [w1_ - 0.1 * g8[0], b1_ - 0.1 * g8[1], w2_ - 0.1 * g8[2], b2_ - 0.1 * g8[3]]
    a1n = t2[0] * x8 + t2[1]
    hn = max(a1n, 0)
    pn = sigm(t2[2] * hn + t2[3])
    ln_ = -log(1 - pn)
    s += sol("Bloque 8", [
        ["1", f"a₁ = 0.5·2 − 0.5 = {a1:g}; h = {h8:g}; a₂ = 1·{h8:g} + 0 = {a2:g}; p = σ({a2:g}) = {p8:.4f}; pérdida = −ln(1 − {p8:.4f}) = {l8:.4f}."],
        ["2", f"δ₂ = p − y = {d2:.4f}; δ₁ = {d2:.4f}·1·1 = {d1:.4f}. Gradientes: ∂w₁ = δ₁x = {g8[0]:.4f}; ∂b₁ = δ₁ = {g8[1]:.4f}; ∂w₂ = δ₂h = {g8[2]:.4f}; ∂b₂ = δ₂ = {g8[3]:.4f}."],
        ["3", f"θ⁺ = ({t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f}, {t2[3]:.4f}). Nueva: a₁ = {a1n:.4f}, h = {hn:.4f}, p = {pn:.4f}, pérdida = {ln_:.4f}: bajó (de {l8:.4f})."]])
    # 9
    s += sol("Bloque 9", [
        ["1", "A: su mejor validación (0.20) fue en la época 60 y después empeoró a 0.25 mientras el entrenamiento bajó a 0.02: señal clara de sobreajuste."],
        ["2", "Según la validación: B (0.15 contra 0.20). De A se conserva la época 60; de B, la 100 (la última, que es la mejor hasta ahora)."],
        ["3", "Que B haya convergido: su validación sigue bajando al final del presupuesto, y con más épocas podría mejorar (o empezar a subir). Tampoco que sea mejor en la prueba."]])
    # 10
    la = -log(0.55)
    lb = np.mean([-log(0.99)] * 3 + [-log(0.30)])
    s += sol("Bloque 10", [
        ["1", "A: 0 errores (todas ≥ 0.5), exactitud 100 %. B: 1 error (0.30 < 0.5), exactitud 3 ÷ 4 = 75 %."],
        ["2", f"A: −ln 0.55 = {la:.4f} (las cuatro iguales). B: (3·(−ln 0.99) + (−ln 0.30)) ÷ 4 = (3·{-log(0.99):.4f} + {-log(0.3):.4f}) ÷ 4 = {lb:.4f}."],
        ["3", f"Exactitud: A es mejor (100 % contra 75 %). Log loss: B es menor ({lb:.4f} contra {la:.4f}). B está muy seguro en tres casos y falla en uno con p = 0.30; A siempre duda (0.55). La pérdida mira cuánta probabilidad se dio a la clase real; la exactitud, solo si pasó el umbral."]])
    build("apuntes/C45_Ejercicios_Cuaderno_RedesNeuronales.pdf", s, "Ejercicios para el cuaderno · Clase 45")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
