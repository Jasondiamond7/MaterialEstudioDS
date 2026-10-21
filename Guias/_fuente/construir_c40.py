"""Clase 40 · Naive Bayes y clasificación probabilística: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c40_numeros.json (lo escribe figuras_c40.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c40_numeros.json").read_text(encoding="utf-8"))
C40 = colors.HexColor("#BE123C")
C40_BG = colors.HexColor("#FFF1F2")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas DipDS_2026_S40.pdf + 40_naive_bayes.py / .R del profesor Eloy Alvarado + spam_s40.csv"
BZ, IG, DP, GS, BR, VR, LP, LG, FR, CM, UM, GB = (N[k] for k in (
    "bayes", "indep", "dup", "gauss", "bern", "variantes", "laplace", "logprob", "fronteras", "comparacion", "umbral", "gaussbin"))
OO = CM["oof"]


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.78):
    s.append(band(f"Clase 40 · {titulo}", FUENTE, C40))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C40, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C40))
    s.append(Spacer(1, 1))
    for fig, pie in figuras:
        s.append(img(fig, width=CONTENT_W * fw))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C40)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ---------------------------------------------------------------- cálculos de apoyo (ejemplos y ejercicios)
def posterior_nb(prior1, th1, th0, x, alpha=0.0, n1=None, n0=None, c1=None, c0=None):
    """Posterior de la clase 1 con indicadores x; si se dan conteos (c1, c0 de n1, n0) usa suavizado alpha."""
    x = np.asarray(x)
    if c1 is not None:
        th1 = (np.asarray(c1, float) + alpha) / (n1 + 2 * alpha)
        th0 = (np.asarray(c0, float) + alpha) / (n0 + 2 * alpha)
    s1 = prior1 * np.prod(np.where(x == 1, th1, 1 - th1))
    s0 = (1 - prior1) * np.prod(np.where(x == 1, th0, 1 - th0))
    return s1 / (s0 + s1), s1, s0


def normal(x, mu, sd):
    return exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * sqrt(2 * pi))


def logloss(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def auc_manual(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 40 · Naive Bayes y clasificación probabilística",
                  "Profesor Eloy Alvarado · usa Bayes (Clase 23), retoma OOF, log loss, calibración y umbral por costo (Clase 38) y prepara el Taller 3 (pregunta 2)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Un clasificador probabilístico entrega una probabilidad, no solo una etiqueta</b>, y <b>predecir no es decidir</b>: primero se construye P(clase | señales) y después un umbral, "
                    "elegido con los costos de cada error, convierte esa probabilidad en una acción. <b>Naive Bayes</b> usa el teorema de Bayes y supone que, <b>dentro de cada clase, las señales son independientes</b>: "
                    "así multiplica un factor por señal (o suma logaritmos) en vez de estimar toda la tabla conjunta. Hay que <b>suavizar</b> para evitar ceros, "
                    "<b>comprobar con predicciones fuera de pliegue</b> y mirar la <b>calibración</b>: ordena bien, pero con señales dependientes sus probabilidades salen demasiado seguras.", "body")], bg=C40_BG, border=C40, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c40_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Bayes y priors</b> (Clase 23): P(A | B) = P(B | A) P(A) / P(B)",
                     "<b>Regresión logística y clasificadores</b> (Clases 31 y 37): probabilidad de la clase 1 y umbral",
                     "<b>OOF, log loss, Brier, calibración y umbral por costo</b> (Clase 38), con los mismos correos de spam",
                     "<b>R ↔ Python</b>: el profesor trae el mismo flujo en los dos lenguajes"], "cell", C40)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Ejemplo de 100 correos</b> → <b>independencia y suavizado</b> → <b>54 señales reales</b> → <b>variantes</b> (Bernoulli, gaussiana, multinomial) → <b>OOF y calibración</b> → <b>umbral por costo</b>",
                      "<b>spam_s40.csv</b>: 4 601 correos de Spambase y 54 frecuencias (48 palabras y 6 caracteres); 3 680 de desarrollo en 5 pliegues y 921 de prueba histórica. Es el mismo spam de la Clase 38",
                      "Pregunta guía: ¿qué probabilidad de spam tiene este correo y conviene bloquearlo?"], "cell", C40)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con otras clases (sin repetirlas)</b>", "cellb"),
                  P("<b>C23 → C40</b>: allí el prior y la verosimilitud actualizaban un parámetro; hoy actualizan la <b>clase</b> de un correo. "
                    f"<b>C37 → C40</b>: la regresión logística es el clasificador de referencia; con los mismos 57 predictores de la Clase 38 da log loss {f(OO['LOGIT57']['log_loss'], 3)}, frente a {f(OO['BERN_05']['log_loss'], 3)} de Naive Bayes Bernoulli, con AUC parecido ({f(OO['LOGIT57']['auc'], 3)} y {f(OO['BERN_05']['auc'], 3)}). "
                    "<b>C38 → C40</b>: OOF, log loss, Brier, curva de calibración y umbral con costo FP = 5 y FN = 1 se aplican igual; aquí se agrega la lectura de por qué las probabilidades salen extremas. "
                    "<b>Taller 3, pregunta 2</b>: esta clase respalda el Naive Bayes del taller y su análisis (independencia condicional, variables 0/1, AUC frente a log loss); aquí solo hay ejemplos con spam, no con los datos del taller.", "cell")], bg=SOFT, border=C40, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: ¿una probabilidad alta es certeza?</b>", "cellb"),
                  P(f"No. En el correo ID 1 el modelo da {f(BR['posterior'] * 100, 2)} % de spam, pero entre los {mil(CM['bern1_top']['n'])} correos con probabilidad ≥ 0.9 la predicción media es {f(CM['bern1_top']['pred_media'] * 100, 1)} % y la fracción real de spam es {f(CM['bern1_top']['spam_obs'] * 100, 1)} %. "
                    "Una probabilidad alta expresa una preferencia del modelo; su fiabilidad se comprueba con la calibración (diapositiva 58).", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Bayes para clasificar: previa, condicional y posterior",
        "Hay <b>tres preguntas distintas</b>: la <b>previa</b> (¿qué fracción de los correos es spam?), la <b>condicional</b> (entre los spam, ¿qué fracción trae «oferta»?) y la <b>posterior</b> (entre los correos con «oferta», ¿qué fracción es spam?). Bayes pasa de la segunda a la tercera.",
        [("La tabla de 100 correos", "80 legítimos y 20 spam; «oferta» aparece en 8 legítimos y en 16 spam. P(oferta | spam) = 16/20 = 0.80 (se mira la fila), pero P(spam | oferta) = 16/24 = 0.667 (se mira la columna)."),
         ("Normalizar", f"Producto de cada clase: legítimo 0.80 × 0.10 = 0.08; spam 0.20 × 0.80 = 0.16. Se divide por la suma (0.24): {f(BZ['post_oferta'], 4)}. Los dos resultados suman 1."),
         ("La prevalencia importa", f"Con las mismas condicionales, si el spam fuera 2 % en vez de 20 %, P(spam | oferta) baja a {f(BZ['post_prior_002'], 4)}. La palabra ayuda, pero no decide por sí sola.")],
        [("c40_bayes.png", "Izquierda: la posterior de «oferta» según la previa (diapositiva 10). Centro: las tres preguntas. Derecha: con «enlace» se agrega información: solo oferta 0.6667, oferta sin enlace 0.4286 y oferta con enlace 0.8750.")],
        ["P(spam | x) = π₁ f₁(x) / [π₀ f₀(x) + π₁ f₁(x)]",
         "π_k: frecuencia de la clase k (previa)",
         "f_k(x): probabilidad de las señales en la clase k",
         "con K clases: se suman K productos abajo"],
        ["s1 = 0.20 * 0.80", "s0 = 0.80 * 0.10", "p = s1 / (s0 + s1)      # 0.6667"],
        ["Con oferta presente y enlace ausente: s0 = 0.80 × 0.10 × 0.80 = 0.064 y s1 = 0.20 × 0.80 × 0.30 = 0.048 → 0.4286 (diap. 17)",
         "Con oferta y enlace presentes: 0.112 y 0.016 → 0.8750 (diap. 18)",
         "Con umbral 0.5, solo «oferta» y «oferta con enlace» se clasificarían como spam"],
        ["Confundir P(oferta | spam) con P(spam | oferta) es el error clásico: el 80 % de los spam trae «oferta», pero solo dos tercios de los correos con «oferta» son spam."], fw=0.97)

    topic_page(
        s, "Independencia condicional y el costo de ignorarla",
        "No se puede contar todas las combinaciones de señales: con d señales hay 2^d patrones por clase. <b>Naive Bayes supone independencia dentro de cada clase</b> y multiplica un factor por señal. Es <b>un supuesto, no un hecho</b>: si falla, el modelo cuenta dos veces la misma evidencia.",
        [("Por qué «naive»", f"Con 10 señales hay {mil(IG['patrones_2_10'])} patrones por clase y con 54 hay 2^54: muchos no aparecerían nunca en los datos. Con independencia bastan d probabilidades por clase."),
         ("El supuesto", "P(oferta, sin enlace | spam) = P(oferta | spam) × P(sin enlace | spam) = 0.80 × 0.30 = 0.24. Nada garantiza que se cumpla: se asume y se comprueba con los resultados."),
         ("Duplicar una señal", f"Si una señal (80 % en spam y 20 % en legítimos) se copia, la probabilidad sube de {f(DP['post'][0], 2)} a {f(DP['post'][1], 4)} con 2 copias y a {f(DP['post'][2], 4)} con 4, sin información nueva.")],
        [("c40_independencia.png", "Izquierda: parámetros por clase con y sin independencia. Centro: los productos s0 y s1 de la diapositiva 17 y los de la 22 con suavizado. Derecha: las cuatro combinaciones de oferta y enlace."),
         ("c40_duplicacion.png", f"Izquierda: el experimento controlado (diapositivas 46 y 47). Derecha: con los correos reales, repetir las 54 señales 1, 2, 4 y 8 veces lleva el log loss OOF de {f(DP['real'][0]['log_loss'], 3)} a {f(DP['real'][3]['log_loss'], 3)} mientras el AUC se mantiene en {f(DP['real'][0]['auc'], 3)}.")],
        ["f_k(x) = Π_j f_jk(x_j)",
         "P(C_k | x) = π_k Π_j f_jk(x_j) / (suma sobre las clases)",
         "d factores en lugar de 2^d patrones",
         "copias: el factor se repite y la certeza sube"],
        ["# duplicar columnas", "Xdup = np.hstack([Xb, Xb])", "BernoulliNB(alpha=1).fit(Xdup, y)"],
        [f"Con las 54 señales el 81 % de las probabilidades OOF queda fuera de (0.01, 0.99); con 4 repeticiones, el {f(DP['real'][2]['p_media_extrema'] * 100, 0)} %",
         f"El orden no cambia (AUC {f(DP['real'][0]['auc'], 3)} → {f(DP['real'][2]['auc'], 3)}) pero el log loss se multiplica por {f(DP['real'][2]['log_loss'] / DP['real'][0]['log_loss'], 1)}",
         "En la regresión, dos predictores idénticos no «suman» información; en Naive Bayes sí aumentan la certeza"],
        ["Duplicar una señal no mejora el modelo: lo hace más seguro de sí mismo. No es lo mismo que el sobreajuste (ajustar el ruido); es <b>sobreconfianza por dependencia</b>."])

    topic_page(
        s, "Bernoulli sobre correos reales y suavizado de Laplace",
        "En la variante <b>Bernoulli</b> cada una de las 54 frecuencias se convierte en <b>presente (1) o ausente (0)</b>; se aprende, por clase, la probabilidad de presencia, y se combinan presencias y ausencias. El <b>suavizado de Laplace</b> suma α a presente y a ausente para que ningún conteo cero anule el producto.",
        [("Aprender y predecir", f"Se cuentan correos por clase (legítimos {mil(BR['n'][0])}, spam {mil(BR['n'][1])}) y presencias por señal. Para «address» presente: legítimo {BR['address']['c0']}/{mil(BR['n'][0])} ≈ {f(BR['address']['th0'], 4)}, spam {BR['address']['c1']}/{mil(BR['n'][1])} ≈ {f(BR['address']['th1'], 4)} (con α = 1, diap. 49)."),
         ("Suavizado", f"θ̂ = (conteo + α)/(n + 2α). Una palabra («premio») con 0 de 20 pasa de 0 a 1/22 = {f(LP['premio'], 4)}. En el ejemplo de 100 correos, 0.4286 → 0.4140 (diap. 22)."),
         ("Con datos reales", f"El log loss OOF es casi igual para α entre 0 y 2 ({f(LP['alfas'][0]['log_loss'], 4)} con α ≈ 0; {f(LP['alfas'][3]['log_loss'], 4)} con 0.5; {f(LP['alfas'][5]['log_loss'], 4)} con 2): con 3 680 correos ninguna señal queda con cero casos. El profesor eligió α = 0.5.")],
        [("c40_bernoulli.png", f"Izquierda: ocho de las 54 señales del correo ID 1 (diap. 50). Centro: las probabilidades OOF se concentran en 0 y 1 ({f(BR['extremos_p'] * 100, 0)} % fuera de (0.01, 0.99)). Derecha: probabilidad de presencia por clase de las 54 señales."),
         ("c40_laplace.png", "Izquierda: cómo cambia θ̂ con α. Centro: la posterior del ejemplo de 100 correos. Derecha: el log loss OOF con 3 680 correos según α.")],
        ["θ̂_jk = (c_jk + α)/(n_k + 2α)",
         "presente: θ̂; ausente: 1 − θ̂",
         "α = 1 es el suavizado clásico; puede ser 0.5",
         "el suavizado no corrige la dependencia"],
        ["m = BernoulliNB(alpha=0.5, binarize=0.0)", "m.fit(X, y)", "m.predict_proba(X_nuevo)[:, 1]"],
        [f"Correo ID 1: previa de spam {f(BR['prior_spam'], 4)}; suma de aportes {f(BR['suma_aportes'], 3)} en log-odds; P(spam) = {f(BR['posterior'], 4)}",
         f"Entre las 54 señales, {BR['n_aportes_positivos']} empujan hacia spam y {BR['n_aportes_negativos']} hacia legítimo para ese correo",
         "En el ejemplo de la diapositiva 25: oferta entre spam 17/22 = 0.7727 y enlace 15/22 = 0.6818"],
        ["Suavizar y decidir son pasos distintos: α cambia el cálculo de la probabilidad; el umbral cambia la acción. Y el suavizado importa sobre todo con pocos datos o señales muy raras."])

    topic_page(
        s, "Gaussiana, log-probabilidades y variantes",
        "Cuando la señal es una medición, <b>Naive Bayes gaussiano</b> aprende una <b>media y una desviación por clase y variable</b> y multiplica <b>alturas</b> de curvas normales (densidades). En la práctica los productos se calculan como <b>sumas de logaritmos</b>: así no se pierde precisión y el resultado es una función logística de la evidencia.",
        [("Densidad, no probabilidad", f"Ejemplo de la clase: x = 1.5, medias 0 y 2, desviaciones 1 y previas 0.8 y 0.2: densidades {f(GS['densidad_0'], 4)} y {f(GS['densidad_1'], 4)}; productos {f(GS['producto_0'], 4)} y {f(GS['producto_1'], 4)}; P(clase 1) = {f(GS['post'], 4)}. La altura no es una probabilidad."),
         ("Transformar", f"El profesor usa z = log(1 + x) para comprimir valores grandes y conservar los ceros. No garantiza normalidad: en «you» hay {f(GS['stats']['word_freq_you']['legítimo']['ceros'] * 100, 0)} % de ceros en legítimos y en «!» {f(GS['stats']['char_freq_!']['legítimo']['ceros'] * 100, 0)} %; la curva normal no calza."),
         ("Sumar logaritmos", f"Con 54 factores el producto no es problema, pero con {LG['d_desborde']} factores de 0.3 ya vale 0 en doble precisión. Los log-odds se suman: previa {f(LG['prior_lo'], 2)} + aportes = {f(LG['log_odds_final'], 2)} → {f(LG['p_final'], 4)}.")],
        [("c40_gaussiana.png", "Izquierda: dos curvas normales ponderadas por su previa (diapositiva 37 y script 04). Centro y derecha: histogramas de log(1 + x) por clase con la normal ajustada."),
         ("c40_logprob.png", "Izquierda: el producto de muchos factores pequeños desborda; la suma de logaritmos no. Centro: la evidencia del correo ID 1 se acumula. Derecha: log-odds → probabilidad.")],
        ["f(x) = exp(−(x − μ)²/2σ²)/(σ√2π)",
         "log P(C_k | x) = log π_k + Σ_j log f_jk(x_j) + cte",
         "log-odds = log(π₁/π₀) + Σ_j log(f_j1/f_j0)",
         "P(spam | x) = 1/(1 + e^(−log-odds))"],
        ["GaussianNB(var_smoothing=1e-9)", ".fit(np.log1p(X), y)", ".predict_log_proba(...)   # sumas de logs"],
        [f"Gaussiana log(1 + x): log loss OOF {f(OO['GAUSS_LOG']['log_loss'], 3)} y exactitud {f(OO['GAUSS_LOG']['accuracy'] * 100, 1)} % (con umbral 0.5); Bernoulli α = 0.5: {f(OO['BERN_05']['log_loss'], 3)}",
         f"El AUC de la gaussiana es {f(OO['GAUSS_LOG']['auc'], 3)}: ordena bien pero muchas probabilidades quedan en 0 o 1",
         "Multinomial: necesita conteos de términos (cuántas veces aparece cada palabra); un porcentaje no basta sin la longitud del mensaje"],
        ["La exactitud no ve la sobreconfianza: la gaussiana acierta 80.8 % y su log loss es nueve veces el del Bernoulli. Con evidencia extrema la probabilidad se redondea a 0 o a 1: conviene mirar también los log-odds."])

    topic_page(
        s, "Gaussiano sobre columnas 0/1 y fronteras de decisión",
        "Aplicar un Naive Bayes <b>gaussiano a una columna 0/1</b> la trata como una medición normal con media p y varianza p(1 − p). Es una aproximación torpe: la densidad no es una probabilidad, la varianza puede ser casi cero y el aporte de «presente» se exagera. Con las <b>fronteras</b> se compara Naive Bayes con LDA, QDA y regresión logística.",
        [("Qué se aprende", f"Para «3d» (presente en {GB['n_presente'][0]} de 2 230 legítimos y {GB['n_presente'][1]} de 1 450 spam): media {f(GB['p0'], 4)} y {f(GB['p1'], 4)}, varianzas {f(GB['var0'], 4)} y {f(GB['var1'], 4)}. La normal es un pico muy angosto en p."),
         ("Qué se distorsiona", f"Cuando la señal está presente, el aporte en log-razón es {f(GB['lr_bern_3d'], 2)} con Bernoulli y {f(GB['lr_gauss_3d'], 0)} con la normal; para señales casi siempre ausentes llega a {f(GB['lr_gauss_min'], 0)}. Una sola columna domina el producto."),
         ("Cómo se lee", f"Log loss OOF: Bernoulli {f(GB['res'][0]['log_loss'], 2)}; gaussiano sobre 0/1 {f(GB['res'][1]['log_loss'], 2)} (con var_smoothing = 1e-2: {f(GB['res'][2]['log_loss'], 2)}); AUC casi igual. Ordena bien y calibra mal; la causa es el modelo de la columna, no el algoritmo.")],
        [("c40_gauss_binarias.png", "Izquierda: una columna 0/1 rara; la pmf de Bernoulli frente a la normal (reescalada). Centro: aporte de «presente» por columna (escala simétrica logarítmica). Derecha: log loss OOF de las cuatro versiones."),
         ("c40_fronteras.png", "Dos señales (log(1 + «!») y log(1 + «your»)) con regiones P = 0.5 de cuatro modelos. Naive Bayes y QDA dan fronteras curvas; LDA y logística, rectas. El resultado en los 921 correos de prueba está en cada panel.")],
        ["Normal en 0/1: μ = p,  σ² = p(1 − p)",
         "NB gaussiano = QDA con covarianza diagonal",
         "LDA: covarianza común → frontera lineal",
         "logística: modela P(clase | x) directamente"],
        ["GaussianNB().fit((X > 0).astype(float), y)", "LinearDiscriminantAnalysis().fit(Z, y)", "QuadraticDiscriminantAnalysis().fit(Z, y)"],
        [f"En la prueba con dos señales: NB {f(FR['resultados']['Naive Bayes gaussiano']['accuracy'] * 100, 1)} % y log loss {f(FR['resultados']['Naive Bayes gaussiano']['log_loss'], 3)}; LDA {f(FR['resultados']['LDA (covarianza común)']['accuracy'] * 100, 1)} %; QDA {f(FR['resultados']['QDA (covarianza por clase)']['accuracy'] * 100, 1)} %; logística {f(FR['resultados']['Regresión logística']['accuracy'] * 100, 1)} % y {f(FR['resultados']['Regresión logística']['log_loss'], 3)}",
         "Variables dummy 0/1 no son normales: con una sola dummy la «campana» no existe",
         "Una variable con media 0 en una clase (siempre ausente) deja varianza casi nula: la densidad explota"],
        ["Bernoulli es la variante natural para 0/1; un gaussiano sobre dummies <b>funciona como aproximación</b>, pero con probabilidades poco confiables. Aquí se explica solo con los correos de spam."])

    topic_page(
        s, "Calibración y decisión por costo",
        "<b>Ordenar bien no es calibrar bien.</b> Naive Bayes y la regresión logística tienen un AUC parecido, pero las probabilidades de Naive Bayes son más extremas y el log loss es el doble. Una vez que se tiene una probabilidad, el <b>umbral</b> se elige con los costos de cada error (FP = 5, FN = 1), con predicciones OOF.",
        [("Comparar con la logística", f"Log loss OOF: frecuencia {f(OO['FRECUENCIA']['log_loss'], 3)}, Bernoulli α = 0.5 {f(OO['BERN_05']['log_loss'], 3)}, logística (54 señales) {f(OO['LOGIT54']['log_loss'], 3)} y logística (57, Clase 38) {f(OO['LOGIT57']['log_loss'], 3)}; Brier {f(OO['BERN_05']['brier'], 3)} frente a {f(OO['LOGIT57']['brier'], 3)}; AUC {f(OO['BERN_05']['auc'], 3)} y {f(OO['LOGIT57']['auc'], 3)}."),
         ("Recalibrar", f"Una sigmoide ajustada dentro de cada pliegue lleva el log loss del Naive Bayes Bernoulli de {f(OO['BERN_1']['log_loss'], 3)} a {f(OO['BERN_CAL']['log_loss'], 3)}: el orden casi no cambia, las probabilidades sí."),
         ("Umbral por costo", f"Con probabilidades calibradas, bloquear conviene si 5(1 − p) &lt; p, es decir p &gt; 5/6 ≈ 0.833. Con OOF el menor costo está en 0.95 ({f(UM['c_nb_min'], 1)} por 1000); en los 921 correos: FP {UM['prueba_nb']['FP']}, FN {UM['prueba_nb']['FN']} y costo {f(UM['prueba_nb']['costo_1000'], 1)}, frente a {f(UM['base_prueba'], 1)} si todo pasa.")],
        [("c40_calibracion.png", "Izquierda: log loss OOF y AUC de cada receta. Centro: diagrama de calibración. Derecha: distribución de las probabilidades OOF de Naive Bayes y de la logística."),
         ("c40_umbral.png", "Izquierda: ROC OOF. Centro: costo por 1000 correos según el umbral para Naive Bayes y la logística; el umbral teórico es 5/6. Derecha: matriz de confusión en los 921 correos históricos con α = 0.5 y umbral 0.95.")],
        ["costo bloquear: 5 (1 − p)",
         "costo dejar pasar: 1 · p",
         "bloquear si p ≥ 5/(5 + 1) = 0.833",
         "costo por 1000 = 1000 (5 FP + FN)/n"],
        ["costo = 1000 * (5 * fp + fn) / n", "umbrales = np.arange(1, 20) / 20", "# elegir en OOF; probar una vez en prueba"],
        [f"p = 0.90: bloquear cuesta 0.50 y dejar pasar 0.90 → bloquear; p = 0.70: 1.50 frente a 0.70 → dejar pasar (diap. 51)",
         f"Con la logística (57), la prueba de 921 correos da FP {UM['prueba_logit']['FP']}, FN {UM['prueba_logit']['FN']} y costo {f(UM['prueba_logit']['costo_1000'], 1)} con umbral {f(UM['t_logit'], 2)}; con Naive Bayes, {f(UM['prueba_nb']['costo_1000'], 1)}",
         "El 0.95 está en el borde de la rejilla: con otra rejilla podría haber un umbral mejor"],
        ["Los 921 correos ya se usaron al comparar clasificadores: es una <b>comprobación retrospectiva</b>, no una prueba independiente. Elegir modelo y umbral en desarrollo, y probar una sola vez."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase (transcripción)</b>", "cellb"),
                  bullets(["<b>Enlace en la tabla</b>: se dijo «16 correos legítimos tienen enlace y 14 spam no tienen enlace». La diapositiva 11 dice: 16 legítimos con enlace y <b>14 spam con enlace</b> (6 sin enlace). Después se usa bien.",
                           "<b>«6 correos sin enlace dentro de los 20 legítimos»</b>: son los 20 <b>spam</b> (los legítimos son 80, con 64 sin enlace). Y «20 de los spam observados son 80 de los 100» debe leerse 20 de 100 (denominador 20 + 2).",
                           f"<b>Costo de umbral</b>: la transcripción habló de un costo «216»; las diapositivas y el código dan <b>{f(UM['c_nb_min'], 1)}</b> por 1000 (78 FP y 416 FN) con umbral 0.95. El log loss de la frecuencia es 0.671 (no 0.61) y el total son 4 601 correos (no 6 000).",
                           "<b>«Es sobre ajuste»</b> (duplicar una columna): es <b>sobreconfianza por dependencia</b>; en regresión, columnas idénticas no suman información, en Naive Bayes sí aumentan la certeza.",
                           "<b>Umbral y calibración</b>: con probabilidades calibradas el umbral teórico sería 5/6; el óptimo empírico (0.95) está en el borde de la rejilla de 19 valores. No demuestra que 0.95 sea el mejor umbral posible.",
                           "<b>Suavizar no arregla la dependencia</b> (se dijo, y es correcto). Lo que sí corrige un desbalance fuerte es remuestrear y luego «reponer» la prevalencia real: las probabilidades quedan sobreestimadas si no se compensa."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, scripts y verificación</b>", "cellb"),
                  bullets([f"<b>Reproducido con el script</b>: ejemplo de 100 correos (0.6667, 0.1404, 0.4286, 0.4140), suavizado, duplicación (0.80, 0.9412, 0.9961), gaussiana (0.4046), multinomial (0.8333 y 0.9615) y OOF: log loss {f(OO['FRECUENCIA']['log_loss'], 3)}, {f(OO['BERN_05']['log_loss'], 3)}, {f(OO['BERN_1']['log_loss'], 3)}, 0.558 y {f(OO['GAUSS_LOG']['log_loss'], 3)}; AUC {f(OO['BERN_05']['auc'], 3)} y {f(OO['GAUSS_LOG']['auc'], 3)}.",
                           f"<b>Prueba histórica</b>: 541 / 17 / 89 / 274, costo {f(UM['prueba_nb']['costo_1000'], 1)}, log loss {f(UM['prueba_nb']['log_loss'], 3)} y Brier {f(UM['prueba_nb']['brier'], 3)}; el correo ID 1 da {f(BR['posterior'], 4)}.",
                           "<b>Qué no hace el script</b>: no compara con la regresión logística ni usa LDA o QDA; esas comparaciones son de esta guía (la logística con la receta de la Clase 38, 57 entradas, StandardScaler, C = 1).",
                           "<b>Multinomial</b>: solo se vio un ejemplo construido; con frecuencias en % no se puede aplicar sin conocer la longitud del mensaje.",
                           "<b>Transcripción con ruido</b>: «Nahid Valles / naivayes / NUGI» = Naive Bayes y Bernoulli, «suavizamiento de la plaza» = de Laplace, «el láser / el lazo» = el enlace, «cátera» = cátedra. El recreo (hacia 1:26:33) quedó marcado y el video continúa sin hueco."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Calculo P(spam | señales) a mano desde una tabla de frecuencias y distingo previa, condicional y posterior",
                           "Explico qué supone Naive Bayes y qué pasa si dos señales repiten la misma información",
                           "Aplico el suavizado de Laplace a un conteo cero y digo cuándo importa",
                           "Calculo la verosimilitud gaussiana con medias y desviaciones dadas y explico por qué se usan logaritmos",
                           "Explico por qué un gaussiano sobre columnas 0/1 es una aproximación débil",
                           "Comparo Naive Bayes con la logística: mismo orden, distinta calibración (AUC frente a log loss)",
                           "Elijo el umbral con costos y lo pruebo una sola vez",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Siguiente</b>: Clase 41 (anomalías) y el Taller 3, pregunta 2 (Naive Bayes)."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C40_Guia_NaiveBayes.pdf", s, "Clase 40 · Naive Bayes")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#BE123C">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 40 · Naive Bayes y clasificación probabilística", "Para leer antes de los ejercicios · lectura de ~30 min", C40))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Bayes invierte la pregunta", "P(spam | señales) sale de P(señales | spam) y de la previa; no son lo mismo."),
                            ("Naive = independencia dentro de la clase", "Se multiplica un factor por señal; si falla, el modelo se sobreconfía."),
                            ("Suaviza los ceros", "(conteo + α)/(n + 2α): ningún cero anula el producto."),
                            ("Suma logaritmos", "Evita el desborde y muestra la evidencia como log-odds."),
                            ("Ordenar no es calibrar", "NB y la logística pueden tener el mismo AUC y distinto log loss."),
                            ("Predecir ≠ decidir", "El umbral sale de los costos de FP y FN, no de 0.5 por costumbre.")], C40)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C40_NaiveBayes.ipynb</font> o el cuaderno <font name='Mono'>C40_Ejercicios_Cuaderno_NaiveBayes.pdf</font>.", "small"))

    # 1 Bayes con tabla
    s.append(h("1", "Bayes con una tabla de frecuencias"))
    s += para("Con una tabla de conteos se calculan tres cosas: la <b>previa</b> (fracción de cada clase), la <b>condicional</b> (dentro de cada clase, fracción con la señal) y la <b>posterior</b>, que se obtiene multiplicando previa por condicional en cada clase y dividiendo por la suma de ambos productos.")
    n0, n1, c0, c1 = 150, 50, 10, 35
    pr1 = n1 / (n0 + n1)
    t1, t0 = c1 / n1, c0 / n0
    post = pr1 * t1 / (pr1 * t1 + (1 - pr1) * t0)
    post2 = 0.5 * t1 / (0.5 * t1 + 0.5 * t0)
    s.append(ejemplo("200 correos (150 legítimos y 50 spam); «gratis» aparece en 10 legítimos y en 35 spam", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Previa", "50/200", f"{f(pr1, 2)}"],
               ["Condicionales", "35/50 y 10/150", f"{f(t1, 2)} y {f(t0, 4)}"],
               ["Productos", "0.25 × 0.70 y 0.75 × 0.0667", f"s1 = {f(pr1 * t1, 4)}; s0 = {f((1 - pr1) * t0, 4)}"],
               ["Posterior", "s1/(s0 + s1)", f"{f(post, 4)}"],
               ["Con otra previa (50 %)", "0.5 × 0.70/(0.5 × 0.70 + 0.5 × 0.0667)", f"{f(post2, 4)}"]],
              [38 * mm, 74 * mm, CONTENT_W - 112 * mm - 18]),
        P(f"Lectura: de los spam, el 70 % trae «gratis», pero de los correos con «gratis» el {f(post * 100, 1)} % es spam; y si la previa fuera 50 % subiría a {f(post2 * 100, 1)} %. <b>Error común</b>: tomar 0.70 como probabilidad de spam o ignorar la previa.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_bayes.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A1 y A2) y los Bloques 1 y 2 del cuaderno", [P("Reproduce 0.6667, 0.1404, 0.4286 y 0.4140 con NumPy y con <font name='Mono'>BernoulliNB</font>.", "cell")]))

    # 2 independencia y suavizado
    s.append(h("2", "Dos señales, independencia y suavizado"))
    s += para("Con dos señales se multiplica un factor por señal dentro de cada clase, usando el estado observado (presente o ausente). El <b>suavizado de Laplace</b> reemplaza cada conteo c por c + α y cada total n por n + 2α.")
    x = [1, 0]
    c1v, c0v = [35, 5], [10, 60]       # gratis, reunión (presencias)
    th1 = np.array([35 / 50, 5 / 50])
    th0 = np.array([10 / 150, 60 / 150])
    pa, a1, a0 = posterior_nb(0.25, th1, th0, x)
    pb, b1, b0 = posterior_nb(0.25, None, None, x, alpha=1, n1=50, n0=150, c1=c1v, c0=c0v)
    s.append(ejemplo("mismos 200 correos; «gratis» presente y «reunión» ausente (reunión: 60 legítimos, 5 spam)", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Factores spam", "0.25 × (35/50) × (1 − 5/50)", f"s1 = {f(a1, 4)}"],
               ["Factores legítimo", "0.75 × (10/150) × (1 − 60/150)", f"s0 = {f(a0, 4)}"],
               ["Posterior sin suavizar", "s1/(s0 + s1)", f"{f(pa, 4)}"],
               ["Con α = 1", "(35 + 1)/52, (45 + 1)/52, (10 + 1)/152, (90 + 1)/152", f"s1 = {f(b1, 4)}; s0 = {f(b0, 4)}"],
               ["Posterior suavizado", "s1/(s0 + s1)", f"{f(pb, 4)}"],
               ["Un conteo cero", "«ganador»: 0 de 50 spam; α = 1", "0 → 1/52 = 0.0192"]],
              [38 * mm, 80 * mm, CONTENT_W - 118 * mm - 18]),
        P("Lectura: con 200 correos el suavizado cambia poco el resultado, pero evita que una palabra nunca vista en spam anule todo el producto. <b>Error común</b>: sumar solo 1 al denominador, o suavizar y creer que se corrigió la dependencia entre señales.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_independencia.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(img("c40_laplace.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A3 y A4) y los Bloques 1 y 2 del cuaderno", [P("Calcula los productos s0 y s1 a mano y compáralos con <font name='Mono'>predict_proba</font> de <font name='Mono'>BernoulliNB</font>.", "cell")]))

    # 3 gaussiano
    s.append(h("3", "Naive Bayes gaussiano y las log-probabilidades"))
    s += para("Para una medición se aprende una media y una desviación por clase; la «verosimilitud» es la altura de la curva normal en el valor observado. Con varias variables se multiplican (se suman logaritmos), por la previa, y se normaliza.")
    mu0, sd0, mu1, sd1, pr1g, xo = 1.0, 0.8, 3.0, 1.2, 0.3, 2.5
    d0, d1 = normal(xo, mu0, sd0), normal(xo, mu1, sd1)
    q1, q0 = pr1g * d1, (1 - pr1g) * d0
    lo = log(q1) - log(q0)
    s.append(ejemplo("una medición x = 2.5; clase 0: media 1, desviación 0.8; clase 1: media 3, desviación 1.2; previas 0.7 y 0.3", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Altura en clase 0", "N(2.5; 1, 0.8)", f"{f(d0, 4)}"],
               ["Altura en clase 1", "N(2.5; 3, 1.2)", f"{f(d1, 4)}"],
               ["Productos con previa", "0.7 × altura₀ y 0.3 × altura₁", f"{f(q0, 4)} y {f(q1, 4)}"],
               ["Posterior de la clase 1", "q₁/(q₀ + q₁)", f"{f(q1 / (q0 + q1), 4)}"],
               ["Log-odds", "ln q₁ − ln q₀", f"{f(lo, 3)} → 1/(1 + e^(−log-odds)) = {f(1 / (1 + exp(-lo)), 4)}"]],
              [44 * mm, 66 * mm, CONTENT_W - 110 * mm - 18]),
        P("Lectura: aunque x = 2.5 está más cerca de la media de la clase 1, la previa pequeña (0.3) compensa esa cercanía; aun así gana la clase 1. La decisión depende de la previa y de las dos curvas. <b>Error común</b>: leer la altura como si fuera una probabilidad, o olvidar la previa.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_gaussiana.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(img("c40_logprob.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B1 y B2) y el Bloque 3 y 4 del cuaderno", [P("Calcula las alturas con <font name='Mono'>scipy.stats.norm.pdf</font> y compáralas con <font name='Mono'>GaussianNB</font>; luego suma logaritmos.", "cell")]))

    # 4 variantes y gaussiano en 0/1
    s.append(h("4", "Variantes, y el gaussiano sobre columnas 0/1"))
    s += para("<b>Bernoulli</b> usa presencias (0/1); <b>multinomial</b>, conteos de términos; <b>gaussiana</b>, mediciones continuas. Con frecuencias en porcentaje no hay conteos: se usa Bernoulli (frecuencia &gt; 0) o gaussiana (log(1 + x)). Aplicar la gaussiana a una columna 0/1 la trata como normal con media p y varianza p(1 − p).")
    p0v, p1v = 0.02, 0.15
    v0, v1 = p0v * (1 - p0v), p1v * (1 - p1v)
    s.append(ejemplo("una palabra presente en el 2 % de los legítimos y en el 15 % de los spam; el correo la trae (x = 1)", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Bernoulli: log-razón", "ln(0.15/0.02)", f"{f(log(p1v / p0v), 3)}"],
               ["Varianzas normales", "p(1 − p) por clase", f"{f(v0, 4)} y {f(v1, 4)}"],
               ["Alturas en x = 1", "N(1; 0.02, √0.0196) y N(1; 0.15, √0.1275)", f"{normal(1, p0v, sqrt(v0)):.2e} y {f(normal(1, p1v, sqrt(v1)), 4)}"],
               ["Gaussiano: log-razón", "ln(altura₁/altura₀)", f"{f(log(normal(1, p1v, sqrt(v1)) / normal(1, p0v, sqrt(v0))), 2)}"]],
              [44 * mm, 76 * mm, CONTENT_W - 120 * mm - 18]),
        P(f"Lectura: con Bernoulli el aporte es {f(log(p1v / p0v), 2)} (la señal es 7.5 veces más frecuente en spam); con la normal sale {f(log(normal(1, p1v, sqrt(v1)) / normal(1, p0v, sqrt(v0))), 1)}, unas diez veces más, porque la varianza minúscula del grupo legítimo exagera la evidencia. <b>Error común</b>: creer que un gaussiano sobre dummies «no funciona»; funciona como aproximación, pero con probabilidades poco fiables.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_variantes.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(img("c40_gauss_binarias.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C1 y C2) y el Bloque 3 del cuaderno", [P("Aplica <font name='Mono'>GaussianNB</font> y <font name='Mono'>BernoulliNB</font> a las presencias de los 54 correos y compara log loss y AUC.", "cell")]))

    # 5 independencia violada y calibración
    s.append(h("5", "Dependencia entre señales: sobreconfianza"))
    s += para("Si una señal se repite, Naive Bayes la multiplica de nuevo. La clase predicha puede no cambiar, pero la probabilidad sí: sube sin información nueva. Por eso Naive Bayes ordena bien y calibra mal.")
    ps_, pl_ = 0.6, 0.15
    rows = [["Copias", "Cálculo", "Posterior"]]
    for k in (1, 2, 3):
        a = 0.5 * ps_ ** k
        b = 0.5 * pl_ ** k
        rows.append([str(k), f"0.5·0.60^{k} / (0.5·0.60^{k} + 0.5·0.15^{k})", f"{f(a / (a + b), 4)}"])
    s.append(ejemplo("una señal en el 60 % de los spam y el 15 % de los legítimos; previas iguales", [
        pasos(rows, [26 * mm, 90 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: con una copia, 0.8000; con dos, 0.9412; con tres, 0.9846. La información real es la misma señal, pero el modelo cree tener tres evidencias independientes. <b>Error común</b>: verlo como «sobreajuste»; es sobreconfianza por dependencia.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_duplicacion.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(img("c40_calibracion.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C3 y C4) y el Bloque 5 y 6 del cuaderno", [P("Repite las 54 señales 1, 2, 4 y 8 veces y mira log loss y AUC; compara con la logística.", "cell")]))

    # 6 umbral
    s.append(h("6", "Decidir: umbral y costos"))
    s += para("Si bloquear un legítimo (FP) cuesta a y dejar pasar un spam (FN) cuesta b, con una probabilidad p bien calibrada se bloquea cuando a (1 − p) &lt; b p, es decir p &gt; a/(a + b). En la práctica el umbral se busca en una rejilla con predicciones OOF y se prueba una vez.")
    a_, b_ = 4, 1
    tcrit = a_ / (a_ + b_)
    ps2 = [0.9, 0.85, 0.6]
    rows = [["p", "bloquear: 4 (1 − p)", "dejar pasar: p", "decisión"]]
    for p in ps2:
        bl, dp = a_ * (1 - p), b_ * p
        rows.append([f"{p:.2f}", f"{f(bl, 2)}", f"{f(dp, 2)}", "bloquear" if bl < dp else "dejar pasar"])
    s.append(ejemplo("costos FP = 4 y FN = 1", [
        pasos(rows, [24 * mm, 54 * mm, 44 * mm, CONTENT_W - 122 * mm - 18]),
        P(f"Lectura: el umbral teórico es 4/5 = {f(tcrit, 2)}. En un conjunto de 1 000 correos con FP = 30 y FN = 120 el costo es (4 × 30 + 120)/1 000 = 0.24, o 240 por cada 1 000 correos. <b>Error común</b>: usar 0.5 por costumbre aunque los errores no cuesten lo mismo, o elegir el umbral con el conjunto de prueba.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c40_umbral.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D1 y D2) y el Bloque 7 del cuaderno", [P("Barre 19 umbrales con las OOF, elige el de menor costo y pruébalo una vez en los 921 correos históricos.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Previa, condicional y posterior son tres cantidades distintas; la prevalencia cambia la respuesta.",
                       "Naive Bayes supone independencia dentro de la clase: el suavizado evita ceros, pero no corrige la dependencia.",
                       "Usa logaritmos; el log-odds suma la evidencia y la probabilidad es su función logística.",
                       "AUC mide el orden; log loss y la calibración miden las probabilidades. Naive Bayes ordena bien y calibra peor.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    # el título, su párrafo y el ejemplo resuelto van juntos (sin títulos solos al pie de página)
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 4]))
            i += 4
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C40_Apuntes_NaiveBayes.pdf", out, "Apuntes · Clase 40 · Naive Bayes")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#BE123C">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C40))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 40 · Naive Bayes", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C40))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: P(clase | x) = π_k Π f_jk(x_j) / Σ_ℓ π_ℓ Π f_jℓ(x_j) · θ̂ = (c + α)/(n + 2α) · N(x; μ, σ) = exp(−(x − μ)²/2σ²)/(σ√2π) · log loss = −media[y ln p + (1 − y) ln(1 − p)] · umbral = a/(a + b).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C40)], bg=SOFT, border=C40, pad=5))
    s.append(Spacer(1, 2))

    # datos de los ejercicios
    A_n0, A_n1 = 240, 60
    A_urg = (12, 30)       # urgente: legítimos, spam
    A_enl = (48, 27)       # enlace
    s += bloque(1, "Naive Bayes desde una tabla de frecuencias", "★★", "300 correos: 240 legítimos y 60 spam. «urgente» aparece en 12 legítimos y 30 spam; «enlace», en 48 legítimos y 27 spam. Un correo trae «urgente» y no trae «enlace».", [
        "Calcula la previa y las probabilidades condicionales de cada señal en cada clase (incluidas las ausencias).",
        "Calcula s0 y s1 con independencia y la probabilidad de spam.",
        "Con umbral 0.5, ¿se bloquea? ¿Y si «urgente» fuera la única señal?"])
    s += bloque(2, "Suavizado de Laplace a mano", "★★", "En los mismos correos, «ganador» no aparece en ninguno de los 60 spam y aparece en 4 de los 240 legítimos. Llega un correo con «ganador» y «urgente» (y sin «enlace»).", [
        "Calcula P(ganador | spam) sin suavizar y con α = 1. ¿Qué pasa con el producto de spam sin suavizar?",
        "Calcula P(ganador | legítimo) con α = 1 y α = 0.5.",
        "Con α = 1, calcula la probabilidad de spam del correo (usa las tres señales con los conteos del Bloque 1 suavizados)."])
    s += bloque(3, "Naive Bayes gaussiano", "★★★", "Dos mediciones: x₁ (clase 0: media 10, desviación 2; clase 1: media 14, desviación 3) y x₂ (clase 0: media 1.0, desviación 0.5; clase 1: media 2.5, desviación 1.0). Previas 0.6 y 0.4. Un caso: x₁ = 12 y x₂ = 1.5.",
             ["Calcula las cuatro alturas normales.",
              "Calcula la posterior de la clase 1 multiplicando los factores de las dos variables.",
              "Si x₂ se reemplazara por una columna 0/1 con media 0.05 y desviación 0.22 en la clase 0, ¿qué problema aparecería?"])
    s += bloque(4, "Por qué logaritmos", "★★", "Un filtro multiplica 500 factores, cada uno cercano a 0.1 (por clase).", [
        "Calcula el orden de magnitud del producto. ¿Qué ocurre en doble precisión (mínimo ≈ 1e−308)?",
        "En log natural, ln s1 = −812.4 y ln s0 = −815.1. Calcula el log-odds y la probabilidad de spam.",
        "¿Por qué se puede restar logaritmos sin pasar por los productos?"])
    s += bloque(5, "Independencia violada: señales duplicadas", "★★★", "Una señal aparece en 70 de 100 spam y en 30 de 100 legítimos; previas iguales.", [
        "Calcula la probabilidad de spam con una copia y con dos copias de la señal.",
        "Calcula con tres copias. ¿Cuánta información nueva hay respecto de una copia?",
        "Explica con una frase por qué esto es sobreconfianza y no sobreajuste."])
    y6 = np.array([1, 1, 0, 1, 0, 0])
    plg = np.array([0.90, 0.75, 0.60, 0.55, 0.30, 0.10])
    pnb = 1 / (1 + np.exp(-3 * np.log(plg / (1 - plg))))
    rows_ = "; ".join(f"{i + 1}: y = {y6[i]}, logística {plg[i]:.2f}, Naive Bayes {pnb[i]:.3f}" for i in range(6))
    s += bloque(6, "Ordenar bien frente a calibrar bien", "★★★", "Seis correos. " + rows_ + ".", [
        "Ordena los correos por probabilidad con cada modelo. ¿Coinciden los órdenes? ¿Cuántos pares (spam, legítimo) quedan bien ordenados (AUC)?",
        "Calcula el log loss de cada modelo. ¿Cuál es mejor y por qué?",
        "¿Qué modelo usarías para ordenar la bandeja y cuál para estimar el costo esperado de bloquear?"])
    s += bloque(7, "Umbral por costo", "★★★", "Costos: FP = 3 y FN = 1. En 500 correos de desarrollo, con umbral 0.5: FP = 40, FN = 50; con 0.75: FP = 20, FN = 85; con 0.90: FP = 8, FN = 140.", [
        "Calcula el umbral teórico y explica qué supone.",
        "Calcula el costo por 1000 correos de cada umbral y elige el mejor.",
        "Si el mejor umbral empírico difiere del teórico, ¿qué dice eso de las probabilidades? ¿Qué más harías antes de usarlo?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C40))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C40), Spacer(1, 2)]

    # bloque 1
    pr1 = A_n1 / (A_n0 + A_n1)
    u1, u0 = A_urg[1] / A_n1, A_urg[0] / A_n0
    e1, e0 = A_enl[1] / A_n1, A_enl[0] / A_n0
    s1_ = pr1 * u1 * (1 - e1)
    s0_ = (1 - pr1) * u0 * (1 - e0)
    pst = s1_ / (s0_ + s1_)
    only_u = pr1 * u1 / (pr1 * u1 + (1 - pr1) * u0)
    s += sol("Bloque 1", [
        ["1", f"Previa spam 60/300 = {f(pr1, 2)} (legítimo {f(1 - pr1, 2)}). «urgente»: spam 30/60 = {f(u1, 2)}, legítimo 12/240 = {f(u0, 2)}; «enlace»: spam 27/60 = {f(e1, 2)}, legítimo 48/240 = {f(e0, 2)}. Ausencias: sin enlace {f(1 - e1, 2)} en spam y {f(1 - e0, 2)} en legítimos."],
        ["2", f"s1 = 0.20 × {f(u1, 2)} × {f(1 - e1, 2)} = {f(s1_, 4)}; s0 = 0.80 × {f(u0, 2)} × {f(1 - e0, 2)} = {f(s0_, 4)}. P(spam) = {f(s1_, 4)}/({f(s0_, 4)} + {f(s1_, 4)}) = {f(pst, 4)}."],
        ["3", f"{f(pst, 3)} &gt; 0.5: se bloquea. Con solo «urgente»: 0.20 × 0.50 = 0.10 y 0.80 × 0.05 = 0.04 → {f(only_u, 4)}: también se bloquea y con más certeza, porque la ausencia de enlace (más frecuente en legítimos) baja la probabilidad de spam."]])
    # bloque 2
    g1_ = (0 + 1) / (60 + 2)
    g0a = (4 + 1) / (240 + 2)
    g0b = (4 + 0.5) / (240 + 1)
    th1 = np.array([(30 + 1) / 62, (27 + 1) / 62, g1_])
    th0 = np.array([(12 + 1) / 242, (48 + 1) / 242, g0a])
    x_ = np.array([1, 0, 1])
    pst2, s1b, s0b = posterior_nb(0.2, th1, th0, x_)
    s += sol("Bloque 2", [
        ["1", f"Sin suavizar: 0/60 = 0 y el producto de spam vale 0 (la probabilidad de spam sería 0 aunque otras señales indiquen spam). Con α = 1: (0 + 1)/(60 + 2) = {f(g1_, 4)}."],
        ["2", f"α = 1: (4 + 1)/(240 + 2) = {f(g0a, 4)}; α = 0.5: (4 + 0.5)/(240 + 1) = {f(g0b, 4)}. El suavizado cambia poco una señal con conteos grandes."],
        ["3", f"Con α = 1: urgente 31/62 = {f(th1[0], 3)} y 13/242 = {f(th0[0], 4)}; sin enlace 1 − 28/62 = {f(1 - th1[1], 4)} y 1 − 49/242 = {f(1 - th0[1], 4)}. s1 = 0.2 × {f(th1[0], 3)} × {f(1 - th1[1], 4)} × {f(g1_, 4)} = {f(s1b, 6)}; s0 = 0.8 × {f(th0[0], 4)} × {f(1 - th0[1], 4)} × {f(g0a, 4)} = {f(s0b, 6)}. P(spam) = {f(pst2, 4)}."]])
    # bloque 3
    xs = [12, 1.5]
    d = {"a0": normal(12, 10, 2), "a1": normal(12, 14, 3), "b0": normal(1.5, 1.0, 0.5), "b1": normal(1.5, 2.5, 1.0)}
    q0 = 0.6 * d["a0"] * d["b0"]
    q1 = 0.4 * d["a1"] * d["b1"]
    s += sol("Bloque 3", [
        ["1", f"x₁ = 12: clase 0 → {f(d['a0'], 4)}; clase 1 → {f(d['a1'], 4)}. x₂ = 1.5: clase 0 → {f(d['b0'], 4)}; clase 1 → {f(d['b1'], 4)}."],
        ["2", f"q₀ = 0.6 × {f(d['a0'], 4)} × {f(d['b0'], 4)} = {f(q0, 4)}; q₁ = 0.4 × {f(d['a1'], 4)} × {f(d['b1'], 4)} = {f(q1, 4)}. P(clase 1) = {f(q1 / (q0 + q1), 4)}."],
        ["3", "Una columna 0/1 con media 0.05 tiene desviación 0.22: un valor 1 queda a (1 − 0.05)/0.22 ≈ 4.3 desviaciones; la altura es minúscula y arrastra el producto hacia la otra clase. Además la altura no es una probabilidad (puede ser mayor que 1) y la columna no es normal: Bernoulli es la variante natural."]])
    # bloque 4
    l1, l0 = -812.4, -815.1
    lo4 = l1 - l0
    s += sol("Bloque 4", [
        ["1", "500 × log₁₀(0.1) = −500: el producto es 1e−500, menor que el mínimo ≈ 1e−308: en doble precisión se redondea a 0 y el cociente s1/(s0 + s1) queda 0/0."],
        ["2", f"log-odds = ln s1 − ln s0 = {f(l1, 1)} − ({f(l0, 1)}) = {f(lo4, 1)}; P(spam) = 1/(1 + e^(−{f(lo4, 1)})) = {f(1 / (1 + exp(-lo4)), 4)}."],
        ["3", "Porque ln(s1/s0) = ln s1 − ln s0: la razón de productos es una diferencia de sumas de logaritmos; solo hace falta exponenciar el resultado final, que ya es moderado."]])
    # bloque 5
    r5 = [0.5 * 0.7 ** k / (0.5 * 0.7 ** k + 0.5 * 0.3 ** k) for k in (1, 2, 3)]
    s += sol("Bloque 5", [
        ["1", f"1 copia: 0.7/(0.7 + 0.3) = {f(r5[0], 4)}. 2 copias: 0.49/(0.49 + 0.09) = {f(r5[1], 4)}."],
        ["2", f"3 copias: 0.343/(0.343 + 0.027) = {f(r5[2], 4)}. La información nueva es nula: las copias son la misma señal."],
        ["3", "El modelo cuenta tres veces la misma evidencia (independencia falsa) y se vuelve más seguro sin motivo; sobreajustar sería ajustar el ruido de la muestra de entrenamiento."]])
    # bloque 6
    ll_l, ll_n = logloss(y6, plg), logloss(y6, pnb)
    s += sol("Bloque 6", [
        ["1", f"Ambos modelos ordenan 1, 2, 3, 4, 5, 6 (el orden de las probabilidades coincide, porque Naive Bayes aquí es una transformación creciente de la logística). Pares spam–legítimo: 3 × 3 = 9; mal ordenado: el correo 3 (legítimo, 0.60) supera al 4 (spam, 0.55) → 8 de 9 bien: AUC = {f(auc_manual(y6, plg), 4)} en los dos."],
        ["2", f"Logística: −media[ln 0.90, ln 0.75, ln 0.40, ln 0.55, ln 0.70, ln 0.90] = {f(ll_l, 4)}. Naive Bayes: {f(ll_n, 4)}. La logística es mejor: Naive Bayes es más seguro y el error (correo 3, correo 4) se castiga mucho más."],
        ["3", "Para ordenar, cualquiera (mismo AUC); para estimar costos esperados o fijar un umbral por costo, el de mejores probabilidades (menor log loss y mejor calibración), o recalibrar el Naive Bayes."]])
    # bloque 7
    tc = 3 / (3 + 1)
    costs = {t: 1000 * (3 * fp + fn) / 500 for t, fp, fn in [(0.5, 40, 50), (0.75, 20, 85), (0.9, 8, 140)]}
    mejor = min(costs, key=costs.get)
    s += sol("Bloque 7", [
        ["1", f"Se bloquea si 3(1 − p) &lt; p, es decir p &gt; 3/4 = {f(tc, 2)}. Supone que las probabilidades están calibradas y que los costos son los dados."],
        ["2", f"Costo por 1000 = 1000 (3 FP + FN)/500: 0.50 → {f(costs[0.5], 0)}; 0.75 → {f(costs[0.75], 0)}; 0.90 → {f(costs[0.9], 0)}. El menor es el de {mejor:.2f}."],
        ["3", "Si coincide con el teórico (0.75), las probabilidades están razonablemente calibradas; si el óptimo se desplaza (como pasa con Naive Bayes sobreconfiado), hay que revisar la calibración. Antes de usarlo: afinar la rejilla de umbrales en OOF, probar una vez en un conjunto aparte y comprobar la prevalencia."]])
    build("apuntes/C40_Ejercicios_Cuaderno_NaiveBayes.pdf", s, "Ejercicios para el cuaderno · Clase 40")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
