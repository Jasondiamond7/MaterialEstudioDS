"""Clase 23 · Estadística bayesiana (fundamentos): guía visual, apuntes y cuaderno de ejercicios en papel.
Todos los números de la guía vienen de c23_numeros.json (lo escribe figuras_c23.py); los de los ejemplos y ejercicios se calculan aquí con scipy."""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer
from scipy import stats

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c23_numeros.json").read_text(encoding="utf-8"))
C23 = colors.HexColor("#059669")
C23_BG = colors.HexColor("#ECFDF5")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas + Clase23.ipynb / práctica en R de la profesora Maritza Márquez"
D, BB, IV, AB, RS_, PR = N["diagnostico"], N["betabinom"], N["intervalos"], N["ab"], N["resumenes"], N["prioris"]
SQ, CJ = N["secuencial"], N["conjugadas"]
B = stats.beta


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{int(round(x)):,}".replace(",", " ")


def wald(y, n, z=1.959964):
    p = y / n
    h_ = z * np.sqrt(p * (1 - p) / n)
    return p - h_, p + h_


def wilson(y, n, z=1.959964):
    p = y / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h_ = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h_, c + h_


def post_bayes(prev, sens, esp):
    return sens * prev / (sens * prev + (1 - esp) * (1 - prev))


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.77):
    s.append(band(f"Clase 23 · {titulo}", FUENTE, C23))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C23, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    dos = len(figuras) > 1
    s.append(numbered(conceptos[:3] if dos else conceptos, C23))
    s.append(Spacer(1, 1))
    for fig, w, pie in figuras:
        s.append(img(fig, width=CONTENT_W * (fw if dos else w * 0.92)))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py and not dos:
        left += [Spacer(1, 3), P("En Python", "h3"), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    if dos:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)] if py else []
        right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C23)]
        s.append(two_cols(left, right))
    else:
        right = [P("Con los números de la clase", "h3"), bullets(salida, "cell", C23), Spacer(1, 3),
                 box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
        s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 23 · Estadística bayesiana: fundamentos",
                  "Profesora Maritza Márquez · primera de dos sesiones · sigue al A/B testing (Clase 22) y precede a predicción, Monte Carlo y MCMC (Clase 24)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("La estadística bayesiana trata el parámetro desconocido θ como una cantidad <b>incierta</b> y describe esa incertidumbre con una <b>distribución de probabilidad</b>. "
                    "Se parte de una <b>priori</b> (lo que se sabía), se mide con la <b>verosimilitud</b> qué tan compatibles son los datos con cada valor de θ y el <b>teorema de Bayes</b> las combina en una <b>posteriori</b> (lo que se sabe ahora): "
                    "<b>creencia inicial + evidencia nueva = creencia actualizada</b>. Con proporciones, el modelo <b>Beta–Binomial</b> lo resuelve con una suma: éxitos previos + éxitos observados, fracasos previos + fracasos observados. "
                    "Y el <b>intervalo creíble</b> sí se lee como suena: «hay 95 % de probabilidad de que θ esté aquí, dados el modelo y los datos».", "body")], bg=C23_BG, border=C23, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c23_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Probabilidad, eventos y proporciones</b> (Etapa 0): P(A), complemento, «A y B»",
                     "<b>Distribución binomial</b> y <b>logística</b> (Clases 20–21): Y ~ Binomial(n, θ) es el modelo de datos de hoy",
                     "<b>Intervalo de confianza y p-valor</b> (Clases 17–19 y 22): se comparan con el intervalo creíble",
                     "Calcular con scipy: <font name='Mono'>stats.beta</font> y <font name='Mono'>stats.binom</font> (equivalentes de <font name='Mono'>dbeta, qbeta, dbinom</font> de R)"], "cell", C23)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Dos enfoques</b> → <b>probabilidad condicional y Bayes</b> → <b>modelo: priori × verosimilitud</b> → <b>Beta–Binomial</b> → <b>resumir la posteriori</b> → <b>intervalo creíble vs de confianza</b>",
                      "No hay archivos de datos: la clase usa una <b>prueba diagnóstica</b> (prevalencia 1 %, sensibilidad 95 %, especificidad 90 %) y <b>14 aprobados de 20</b> con priori Beta(2, 2)",
                      "Pregunta guía: ¿cómo cambia lo que creo sobre θ cuando llegan datos, y cuánta incertidumbre queda?"], "cell", C23)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con las clases vecinas</b>", "cellb"),
                  P("<b>C17–21 → C23</b>: en regresión y GLM los parámetros (β) eran fijos y se estimaban con máxima verosimilitud; ahora la <b>verosimilitud</b> es la misma pero se <i>multiplica</i> por una priori. "
                    "<b>C22 → C23</b>: el p-valor y el IC del A/B test son la respuesta frecuentista; la bayesiana contesta «¿qué probabilidad hay de que B supere a A?» y «¿de que la mejora supere el MDE?» (se calcula en la página 7). "
                    "<b>C23 → C24</b>: la posteriori completa permite predecir y simular (Monte Carlo); cuando no tiene fórmula cerrada se usa MCMC.", "cell")], bg=SOFT, border=C23, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: probabilidad condicional, sin fórmulas</b>", "cellb"),
                  P("P(A | B) se lee «probabilidad de A <b>sabiendo que</b> B ya ocurrió»: se mira solo el grupo de los casos con B. P(enfermo | positivo) mira a quienes dieron positivo; P(positivo | enfermo) mira a los enfermos. "
                    "Son grupos distintos, así que <b>no tienen por qué ser iguales</b>: la sensibilidad es la segunda, y lo que quiere saber un paciente es la primera.", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Dos enfoques: frecuentista y bayesiano",
        "Los dos enfoques modelan los mismos fenómenos; cambia <b>qué es aleatorio</b>. Frecuentista: el parámetro es fijo y los datos varían de muestra en muestra. Bayesiano: los datos observados son fijos y la incertidumbre está en el parámetro.",
        [("Frecuentista", "θ fijo pero desconocido. La probabilidad describe la repetición de experimentos. Se estima y se reporta error estándar, p-valor e intervalo de confianza (todo depende de la muestra que salió)."),
         ("Bayesiano", "θ se describe con una distribución. La probabilidad cuantifica incertidumbre y se puede escribir P(θ &gt; 0.6 | datos) = 0.82. La inferencia sale de la posteriori."),
         ("Preguntas distintas", "Frecuentista: «¿qué pasaría si repitiéramos el experimento muchas veces?». Bayesiano: «dado lo que ya sabía y lo observado, ¿qué valores de θ son más plausibles?»."),
         ("¿Cuál es mejor?", "Ninguno: son enfoques distintos, con supuestos distintos. El bayesiano ayuda cuando hay pocos datos, hay información previa o los datos llegan en secuencia.")],
        [("c23_intervalos.png", 0.98, f"Izquierda: 40 muestras repetidas de n = 20 con θ = 0.65; los intervalos cambian, θ no ({IV['fallan_wald']} intervalo de Wald lo falla, en rojo). Centro: cobertura frecuentista exacta; el creíble con priori uniforme cubre en promedio {f(IV['cobertura_media']['creíble Beta(1, 1)'], 1)} %, Wald solo {f(IV['cobertura_media']['Wald'], 1)} %. "
                                    f"Derecha: con los mismos datos (14 de 20) los cinco intervalos se parecen, pero se interpretan distinto.")],
        ["P(θ <font name='Sym'>∈</font> [a, b] | y) = 0.95  →  intervalo creíble",
         "IC de confianza: el <b>procedimiento</b> contiene a θ el 95 % de las veces",
         "posteriori <font name='Sym'>∝</font> verosimilitud × priori"],
        ["from scipy import stats", "stats.beta(16, 8).interval(0.95)   # creíble", "stats.beta(16, 8).sf(0.6)        # P(θ > 0.6 | y)"],
        [f"Interpretación bayesiana: dado el modelo y los datos, hay 95 % de probabilidad de que θ esté en [{f(BB['ic'][0])}; {f(BB['ic'][1])}]",
         "Interpretación frecuentista de un IC 95 %: si repitiéramos el muestreo muchas veces, el 95 % de los intervalos construidos así contendría el parámetro; este intervalo en particular lo contiene o no",
         "El error típico de estudiantes: leer un IC de confianza como «95 % de probabilidad de que θ esté dentro»"],
        ["El intervalo creíble depende de la <b>priori</b>; el de confianza, no. Y «95 % de los datos dentro del intervalo» no es ninguna de las dos lecturas: ambos intervalos hablan del <b>parámetro</b>."])

    topic_page(
        s, "Teorema de Bayes y frecuencias naturales",
        "Bayes invierte una condicional: pasa de P(datos | hipótesis) a P(hipótesis | datos). La forma más fácil de entenderlo es <b>contar personas</b>: de cada 10 000, ¿cuántas dan positivo y cuántas de ellas están enfermas?",
        [("Probabilidad condicional", "P(A | B) = P(A ∩ B) / P(B), con P(B) &gt; 0. Como P(A ∩ B) = P(B | A)·P(A), se despeja Bayes: P(A | B) = P(B | A)·P(A) / P(B)."),
         ("Cuatro piezas", "P(A) = priori (prevalencia); P(B | A) = verosimilitud (sensibilidad); P(B) = evidencia (probabilidad total de dar positivo); P(A | B) = posteriori."),
         ("Probabilidad total", "P(+) = P(+ | E)·P(E) + P(+ | Eᶜ)·P(Eᶜ). Suma las dos formas de dar positivo: siendo enfermo y siendo sano (falso positivo)."),
         ("Prevalencia manda", "Con una enfermedad rara, los falsos positivos (de un grupo enorme de sanos) superan a los verdaderos positivos. Nunca se interpreta un test sin la prevalencia.")],
        [("c23_bayes.png", 0.98, f"Izquierda: de 10 000 personas, {D['vp']} de los {D['positivos']} positivos están enfermos → {f(D['post'] * 100, 1)} %. Centro: P(E | +) y P(+ | E) miran grupos distintos. "
                                 f"Derecha: la posteriori sube con la prevalencia y con la especificidad (a 1 %: {f(D['post'] * 100, 1)} %, {f(D['post_esp95'] * 100, 1)} % y {f(D['post_esp99'] * 100, 1)} %).")],
        ["P(A | B) = P(B | A)·P(A) / P(B)",
         "P(E | +) = 0.95·0.01 / (0.95·0.01 + 0.10·0.99)",
         "= 0.0095 / 0.1085 = 0.0876",
         "posteriori = verosimilitud × priori / evidencia"],
        ["prev, sens, esp = 0.01, 0.95, 0.90", "p_pos = sens*prev + (1 - esp)*(1 - prev)", "sens*prev / p_pos          # 0.0875576"],
        [f"10 000 personas: {D['enfermos']} enfermos ({D['vp']} dan + y {D['fn']} dan −) y {mil(D['sanos'])} sanos ({D['fp']} dan + y {mil(D['vn'])} dan −)",
         f"Positivos: {D['vp']} + {D['fp']} = {mil(D['positivos'])}; P(E | +) = {D['vp']}/{mil(D['positivos'])} = {f(D['post'], 4)} (la práctica en R da 0.0875576)",
         f"Con un segundo test positivo independiente la posteriori sube a {f(D['post_segundo_test'] * 100, 1)} %; con un tercero, a {f(D['post_tres_tests'] * 100, 1)} %: la posteriori de hoy es la priori de mañana"],
        ["Dos cosas se confunden: «el test es 95 % sensible» (P(+ | E)) y «si doy positivo, tengo 95 % de estar enfermo» (P(E | +)). Con prevalencia 1 % la segunda es 8.8 %."])

    topic_page(
        s, "Modelo bayesiano básico y Beta–Binomial",
        "Un modelo bayesiano tiene tres ingredientes: <b>priori</b> p(θ), <b>verosimilitud</b> p(y | θ) y <b>posteriori</b> p(θ | y) <font name='Sym'>∝</font> p(y | θ)·p(θ). Con datos binomiales y priori Beta, la posteriori es otra Beta: <b>conjugación</b>.",
        [("Beta como priori", "Una proporción vive en (0, 1): θ ~ Beta(α, β). Beta(1, 1) es uniforme; Beta(2, 2) es una campana suave centrada en 0.5. La media es α/(α + β) y α + β mide cuán informativa es."),
         ("Verosimilitud", "Con y éxitos en n intentos, L(θ; y) = C(n, y)·θʸ(1 − θ)ⁿ⁻ʸ. No es una distribución de θ: mide qué tan compatibles son los datos con cada θ."),
         ("Actualización conjugada", "θ | y ~ Beta(α + y, β + n − y). «Éxitos previos + éxitos observados; fracasos previos + fracasos observados». La evidencia p(y) solo normaliza (la posteriori integra 1)."),
         ("Otras conjugadas", "Bernoulli–Beta, Poisson–Gamma, Exponencial–Gamma. Se buscan en una tabla: no hace falta deducirlas cada vez.")],
        [("c23_betabinom.png", 0.98, f"Priori Beta(2, 2) (media 0.50) × verosimilitud de 14 aprobados de 20 (máximo en 0.70) = posteriori Beta(16, 8): media {f(BB['media'], 3)}, MAP {f(BB['moda'], 3)}, IC creíble 95 % [{f(BB['ic'][0], 3)}; {f(BB['ic'][1], 3)}], P(θ &gt; 0.6 | y) = {f(BB['p_mayor_06'], 3)}."),
         ("c23_conjugadas.png", 0.98, f"Verificación numérica: la posteriori calculada multiplicando priori × verosimilitud y normalizando en una grilla (trazo ancho) coincide con la fórmula de la tabla (línea negra). Diferencia máxima entre 1.9 × 10⁻¹⁴ y {CJ['max_dif_grilla']['Exponencial–Gamma']:.1e}.")],
        ["θ ~ Beta(α, β),   y | θ ~ Bin(n, θ)",
         "θ | y ~ Beta(α + y, β + n − y)",
         "Poisson–Gamma: λ | y ~ Gamma(a + S, b + m)",
         "Exponencial–Gamma: λ | y ~ Gamma(a + m, b + S)"],
        ["alpha_p, beta_p = 2 + 14, 2 + 20 - 14", "post = stats.beta(alpha_p, beta_p)   # Beta(16, 8)", "post.mean()   # 0.6667"],
        [f"Priori Beta(2, 2) y 14 aprobados de 20: α = 2 + 14 = 16, β = 2 + 20 − 14 = 8 → media 16/24 = {f(BB['media'], 4)}",
         f"La media posteriori es un promedio ponderado: {f(BB['peso_priori'] * 100, 1)} % de peso a la priori (media 0.50) y {f((1 - BB['peso_priori']) * 100, 1)} % a los datos (14/20 = 0.70)",
         f"Evidencia p(y) = {f(BB['evidencia'], 5)} (fórmula cerrada y grilla coinciden)"],
        ["La profesora dijo que α = β = 2 «no significa dos aprobados y dos reprobados»; la diapositiva 23 sí los llama «éxitos y fracasos previos». Ambas lecturas son útiles: para actualizar, Beta(2, 2) se comporta <b>como si</b> se hubieran visto 2 éxitos y 2 fracasos más."])

    topic_page(
        s, "Resumir la posteriori: media, MAP e intervalo creíble",
        "La posteriori completa es el resultado; un solo número (media, mediana o MAP) la resume pero <b>oculta la incertidumbre</b>. El <b>intervalo creíble</b> añade lo que falta y se lee de forma directa.",
        [("Resúmenes puntuales", "Media posteriori α/(α + β); mediana; moda o MAP (máximo a posteriori) (α − 1)/(α + β − 2) si α, β &gt; 1. Con posterioris simétricas casi coinciden; con asimetría se separan."),
         ("Intervalo creíble 95 %", "[a, b] con P(a &lt; θ &lt; b | y) = 0.95. El «por colas» deja 2.5 % a cada lado (<font name='Mono'>qbeta</font>); el HPD es el más corto con 95 % de masa. Ambos son válidos y distintos."),
         ("Probabilidades de decisión", "P(θ &gt; 0.6 | y) se calcula directo con la cola de la Beta (o con rbeta y la media de un indicador, por simulación)."),
         ("MAP y máxima verosimilitud", "Con priori plana, el MAP es y/n, el estimador de máxima verosimilitud. La media posteriori suaviza los extremos: con 0 éxitos de 10 da 0.083, no 0.")],
        [("c23_resumenes.png", 0.98, f"Izquierda: posteriori asimétrica Beta(2, 10) (1 éxito de 10): MAP {f(RS_['asimetrica']['moda'], 3)}, mediana {f(RS_['asimetrica']['mediana'], 3)}, media {f(RS_['asimetrica']['media'], 3)}; el HPD {f(RS_['asimetrica']['hpd'][0], 2)}–{f(RS_['asimetrica']['hpd'][1], 2)} es más corto que el de colas iguales. "
                                      f"Centro: Beta(16, 8), casi simétrica. Derecha: estimaciones de θ con n = 10 según y.")],
        ["media = α/(α + β)",
         "MAP = (α − 1)/(α + β − 2)",
         "IC creíble por colas = [qbeta(0.025, α, β); qbeta(0.975, α, β)]",
         "P(θ &gt; c | y) = 1 − pbeta(c, α, β)"],
        ["post.ppf([0.025, 0.975])   # qbeta", "post.sf(0.6)               # 1 - pbeta", "(16 - 1) / (16 + 8 - 2)     # MAP"],
        [f"Beta(16, 8): media {f(BB['media'], 4)}, mediana {f(BB['mediana'], 4)}, MAP {f(BB['moda'], 4)}, desviación {f(BB['sd'], 4)}",
         f"IC creíble 95 % por colas [{f(BB['ic'][0], 4)}; {f(BB['ic'][1], 4)}] (R: 0.4708083 y 0.8362364); HPD [{f(BB['hpd'][0], 4)}; {f(BB['hpd'][1], 4)}]",
         f"P(θ &gt; 0.6 | y) = {f(BB['p_mayor_06'], 4)} exacto; con 10 000 simulaciones la práctica obtuvo 0.7581 (error de simulación ≈ {f(BB['mc_se'], 4)}); el ancho del IC baja de {f(BB['ancho_ic_priori'], 2)} (priori) a {f(BB['ancho_ic'], 2)}"],
        ["En clase se dijo «0.47 y 0.83»; el valor exacto es 0.8362, que la práctica escribe 0.84. Y <font name='Mono'>qbeta(0.025, ·)</font> da <b>cuantiles</b> (percentiles 2.5 % y 97.5 %), no «cuartiles»."])

    topic_page(
        s, "Priori, tamaño de muestra y actualización secuencial",
        "La priori pesa como una muestra previa de tamaño α + β: <b>cuanto más datos, menos importa</b>. Y como la posteriori de hoy es la priori de mañana, los datos pueden llegar de a poco y el resultado es el mismo que si llegaran juntos.",
        [("Sensibilidad a la priori", "Con las mismas observaciones, prioris distintas dan posterioris distintas. Se reporta con varias prioris razonables y se ve si la conclusión cambia."),
         ("Peso de la priori", "Peso = (α + β)/(α + β + n). Beta(2, 2) con n = 20: 4/24 = 17 %. Beta(8, 2): 10/30 = 33 %, no es «poco informativa»."),
         ("Más datos, convergencia", "Al crecer n, las posterioris de prioris distintas se parecen entre sí y se acercan a y/n; el intervalo se estrecha en ≈ 1/√n."),
         ("Actualización secuencial", "(α, β) → (α + y₁, β + n₁ − y₁) → … Da lo mismo que procesar todo junto: Beta(16, 8) en ambos casos.")],
        [("c23_prioris.png", 0.98, f"Tres prioris de la práctica con 14 de 20: medias {f(PR['y14'][0]['media'], 3)}, {f(PR['y14'][1]['media'], 3)} y {f(PR['y14'][2]['media'], 3)}; con 8 de 20 (ejercicio de la profesora): {f(PR['y8'][0]['media'], 3)}, {f(PR['y8'][1]['media'], 3)} y {f(PR['y8'][2]['media'], 3)}. Derecha: con 70 % de éxitos, las medias de Beta(1, 1) y Beta(8, 2) pasan de {f(N['convergencia']['10']['unif'], 2)} y {f(N['convergencia']['10']['optimista'], 2)} (n = 10) a {f(N['convergencia']['1000']['unif'], 3)} y {f(N['convergencia']['1000']['optimista'], 3)} (n = 1 000)."),
         ("c23_secuencial.png", 0.98, "Cuatro llegadas de estudiantes (4, 6, 5 y 5; con 3, 4, 3 y 4 aprobados): los parámetros pasan por (5, 3), (9, 5), (12, 7) y (16, 8), igual que con todos los datos de una vez.")],
        ["peso de la priori = (α + β)/(α + β + n)",
         "media posteriori = w·media priori + (1 − w)·y/n",
         "w = (α + β)/(α + β + n)"],
        ["for a, b in [(1, 1), (2, 2), (8, 2)]:", "    post = stats.beta(a + y, b + n - y)", "    print(post.mean(), post.sf(0.6))"],
        [f"y = 14: posterioris Beta({PR['y14'][0]['a_post']}, {PR['y14'][0]['b_post']}), Beta(16, 8), Beta({PR['y14'][2]['a_post']}, {PR['y14'][2]['b_post']}); P(θ &gt; 0.6) = {f(PR['y14'][0]['p06'], 3)}, {f(PR['y14'][1]['p06'], 3)}, {f(PR['y14'][2]['p06'], 3)}",
         f"y = 8: Beta(9, 13), Beta(10, 14), Beta(16, 14); P(θ &gt; 0.6) = {f(PR['y8'][0]['p06'], 3)}, {f(PR['y8'][1]['p06'], 3)}, {f(PR['y8'][2]['p06'], 3)}. La priori optimista pesa más cuando los datos la contradicen",
         "Ejercicio de la práctica: la media pasa de 0.67 a 0.42 con Beta(2, 2); influye más la más informativa, Beta(8, 2)"],
        ["Beta(8, 2) equivale a 10 observaciones previas: con 20 datos es un tercio de la información. Hay que <b>justificarla</b> (datos históricos, expertos), no elegirla por el resultado."], fw=0.6)

    topic_page(
        s, "Puente con la Clase 22: A/B testing bayesiano",
        "El experimento de checkout de la Clase 22 se puede analizar con Beta–Binomial en cada grupo. La respuesta ya no es un p-valor sino una probabilidad directa: <b>P(B &gt; A | datos)</b> y <b>P(mejora &gt; MDE | datos)</b>. (La clase no lo hizo: es una extensión de este material.)",
        [("Un modelo por grupo", "Cada grupo: conversiones ~ Binomial(n, θ) con priori uniforme Beta(1, 1). Posterioris: control Beta(975, 9 992), tratamiento Beta(1 195, 9 842)."),
         ("Diferencia por simulación", "Se simulan 200 000 pares (θ_A, θ_B) de las Betas y se mira θ_B − θ_A: la media, el intervalo creíble y la fracción &gt; 0 o &gt; MDE."),
         ("Qué responde", f"P(B &gt; A) &gt; 0.9999; IC creíble 95 % de la mejora [{f(AB['cri_pp'][0])}; {f(AB['cri_pp'][1])}] pp; P(mejora &gt; 1.5 pp) = {f(AB['p_mejora_mayor_15pp'])}. La Clase 22 solo podía decir que el extremo bajo del IC (1.15) no supera el MDE."),
         ("Parentesco", "Con priori plana y mucho n, el IC creíble ≈ IC de Wald y P(B &gt; A) ≈ 1 − p unilateral. Los números casi coinciden; cambia la lectura.")],
        [("c23_ab.png", 0.98, f"Izquierda: posterioris de las dos tasas. Centro: la diferencia simulada, con su IC creíble y el MDE de 1.5 pp. Derecha: con submuestras de ab_checkout, P(B &gt; A) y 1 − p (unilateral) casi coinciden (por ejemplo, con {mil(N['ab_submuestras'][2]['m'])} por grupo: {f(N['ab_submuestras'][2]['p_b_mayor'], 3)} y {f(N['ab_submuestras'][2]['uno_menos_p'], 3)}).")],
        ["θ_A ~ Beta(1 + x_A, 1 + n_A − x_A)",
         "θ_B ~ Beta(1 + x_B, 1 + n_B − x_B)",
         "P(B &gt; A) = promedio de (θ_B &gt; θ_A) en las simulaciones",
         "mejora = θ_B − θ_A"],
        ["sa = stats.beta(975, 9992).rvs(200000, random_state=rng)", "sb = stats.beta(1195, 9842).rvs(200000, random_state=rng)", "(sb - sa > 0.015).mean()   # P(mejora > 1.5 pp)"],
        [f"Control {mil(AB['x0'])}/{mil(AB['n0'])} = {f(AB['p0'] * 100)} %; tratamiento {mil(AB['x1'])}/{mil(AB['n1'])} = {f(AB['p1'] * 100)} %; diferencia {f(AB['dif_pp'], 3)} pp",
         f"IC de Wald (Clase 22) [{f(AB['wald_ic_pp'][0])}; {f(AB['wald_ic_pp'][1])}] frente a creíble [{f(AB['cri_pp'][0])}; {f(AB['cri_pp'][1])}]",
         f"P(mejora &gt; 1.0 pp) = {f(AB['p_mejora_mayor_1pp'], 3)}; P(mejora &gt; 1.5 pp) = {f(AB['p_mejora_mayor_15pp'], 3)}"],
        ["«P(B &gt; A) = 0.97» no dice que B sea <b>bueno</b>: dice cuán probable es que sea algo mejor. La decisión necesita el tamaño de la mejora, el costo y los guardrails. Y con una priori muy informativa, los resultados dejan de coincidir con el frecuentista."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets(["<b>«gamma de alfa más gamma de beta»</b> (densidad Beta): el denominador es el <b>producto</b> Γ(α)Γ(β); la diapositiva 23 lo escribe bien. Se verificó contra <font name='Mono'>stats.beta.pdf</font>.",
                           "<b>«Los parámetros α y β no son dos aprobados y dos reprobados»</b> frente a la diapositiva 23 («éxitos y fracasos previos»): para <i>actualizar</i> funcionan como pseudo-conteos; la media posteriori usa (α + y)/(α + β + n).",
                           f"<b>«La posteriori es la sensibilidad por la priori sobre la posteriori»</b> (práctica en R): el denominador es la <b>evidencia</b> P(+) = {f(D['p_pos'], 4)}, no la posteriori.",
                           "<b>«Los valores 0.025 y 0.975 son cuartiles»</b>: son <b>cuantiles</b> (percentiles 2.5 % y 97.5 %). Cuartiles son 25, 50 y 75 %.",
                           "<b>«95 % de las veces ese parámetro queda dentro del intervalo»</b> (frecuentista): el parámetro es fijo; lo que se repite y varía es el <b>intervalo</b>. La profesora lo corrigió en el momento.",
                           "<b>«Función de densidad de la binomial»</b>: es una función de <b>masa</b> de probabilidad; como función de θ se llama verosimilitud y no integra 1."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, práctica en R y cuaderno</b>", "cellb"),
                  bullets([f"<b>P(θ &gt; 0.6 | y)</b>: el cuaderno entrega 0.7581 con <font name='Mono'>rbeta</font> (10 000 simulaciones, <font name='Mono'>set.seed(123)</font>); el valor exacto es {f(BB['p_mayor_06'], 4)} (<font name='Mono'>1 − pbeta</font>, 0.7627291 en la tabla de la práctica). La diferencia es ruido de Monte Carlo (error estándar ≈ {f(BB['mc_se'], 4)}).",
                           "<b>Intervalo</b>: la práctica reporta «entre 0.47 y 0.84» y en clase se leyó 0.83; el extremo exacto es 0.8362 (se redondea a 0.84). Es un intervalo <b>por colas</b>; el HPD [0.48; 0.84] también es un intervalo creíble.",
                           "<b>Beta(8, 2) «optimista»</b>: es informativa (equivale a 10 observaciones, 33 % del peso con n = 20). No es una prueba de «prioridad suave».",
                           "<b>Diapositiva 8</b>: «P(θ &gt; 0.6 | datos) = 0.82» es un ejemplo ilustrativo; el valor de la práctica es 0.76.",
                           "<b>Diapositiva 32</b>: anuncia una práctica «en R y Python» pero solo se entrega el archivo en R (<font name='Mono'>Clase23_Bayes_R.Rmd</font>); este material añade la versión en Python."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper escribió «vallenciana», «valleciana», «vallesiana», «valleseano», «vayaciana» y «Valles» por <i>bayesiana</i> y <i>Bayes</i>, «cobi» por COVID, «cuartilas/cubeta» por cuantiles y qbeta, «Colap» por Colab. "
                    "El recreo (≈ 1:24:30 del video) no está en el video; la corrección dejó marcado el punto.", "cell")], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Distingo P(A | B) de P(B | A) y calculo P(enfermo | positivo) con frecuencias naturales y con la fórmula",
                           "Explico priori, verosimilitud, posteriori y para qué sirve la evidencia",
                           "Actualizo una Beta con datos binomiales, y calculo media, MAP e intervalo creíble",
                           "Leo un intervalo creíble y uno de confianza sin confundirlos, y explico cómo influyen la priori y n",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Clase 24</b> (próxima sesión): modelo Normal con varianza conocida, distribución predictiva, simulación de Monte Carlo e introducción a MCMC (Metropolis–Hastings)."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C23_Guia_Bayesiana.pdf", s, "Clase 23 · Estadística bayesiana: fundamentos")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#059669">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 23 · Estadística bayesiana: fundamentos", "Para leer antes de los ejercicios · lectura de ~30 min", C23))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("El parámetro es incierto", "Se describe con una distribución; la inferencia sale de la posteriori."),
                            ("Bayes invierte condicionales", "P(A | B) = P(B | A)·P(A)/P(B): no es lo mismo que P(B | A)."),
                            ("Posteriori <font name='Sym'>∝</font> verosimilitud × priori", "La evidencia solo normaliza."),
                            ("Beta–Binomial = sumar conteos", "Beta(α + y, β + n − y): éxitos y fracasos previos más observados."),
                            ("El intervalo creíble se lee directo", "P(θ <font name='Sym'>∈</font> intervalo | datos) = 0.95; el de confianza habla del procedimiento."),
                            ("La priori pesa como α + β datos", "Con más datos pierde peso; se justifica y se compara con otras.")], C23)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C23_Bayesiana.ipynb</font> o el cuaderno <font name='Mono'>C23_Ejercicios_Cuaderno_Bayesiana.pdf</font>.", "small"))

    s.append(h("1", "Teorema de Bayes con frecuencias naturales"))
    s += para("Contar personas es más fácil que multiplicar probabilidades. Se imagina una población grande, se reparte según la prevalencia y se cuentan los positivos de cada grupo. "
              "La posteriori es la fracción de positivos que son verdaderos: P(E | +) = verdaderos positivos / todos los positivos.")
    prev, sens, esp, tot = 0.02, 0.90, 0.95, 10000
    enf, san = int(tot * prev), int(tot * (1 - prev))
    vp, fp = int(enf * sens), int(round(san * (1 - esp)))
    s.append(ejemplo("prevalencia 2 %, sensibilidad 90 %, especificidad 95 %", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Reparto de 10 000", "2 % enfermos · 98 % sanos", f"{enf} enfermos · {mil(san)} sanos"],
               ["Positivos entre enfermos", "90 % de 200", f"{vp} verdaderos positivos"],
               ["Positivos entre sanos", "5 % de 9 800 (1 − especificidad)", f"{fp} falsos positivos"],
               ["Todos los positivos", f"{vp} + {fp}", f"{vp + fp}"],
               ["Posteriori", f"{vp} / {vp + fp}", f"P(E | +) = {f(vp / (vp + fp), 4)}"],
               ["Con la fórmula", "0.9·0.02 / (0.9·0.02 + 0.05·0.98)", f"{f(post_bayes(prev, sens, esp), 4)}"]],
              [44 * mm, 70 * mm, CONTENT_W - 114 * mm - 18]),
        P(f"Lectura: aun con una prueba buena, solo ≈ {f(vp / (vp + fp) * 100, 0)} % de los positivos están enfermos, porque la enfermedad es poco frecuente. Si la prevalencia fuera 20 %, la posteriori sería {f(post_bayes(0.2, sens, esp) * 100, 0)} %. "
          "<b>Error común</b>: responder «90 %» (la sensibilidad) a la pregunta «¿qué probabilidad hay de estar enfermo si di positivo?».", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_bayes.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno", [P("Simula 1 millón de personas, cuenta los positivos y comprueba que la fracción de enfermos entre ellos es la posteriori. Después encadena dos tests.", "cell")]))

    s.append(h("2", "Priori, verosimilitud y posteriori con Beta–Binomial"))
    s += para("Se quiere estimar la proporción θ. Los datos son y éxitos en n intentos (Binomial). La priori Beta(α, β) resume lo que se sabía; su media es α/(α + β) y α + β mide su peso. "
              "Multiplicar la verosimilitud por la priori da otra Beta con parámetros (α + y, β + n − y). La media posteriori es un promedio ponderado entre la media de la priori y y/n.")
    a0, b0, y_, n_ = 3, 7, 18, 40
    ap, bp = a0 + y_, b0 + n_ - y_
    pd_ = B(ap, bp)
    s.append(ejemplo("tasa de adherencia a un tratamiento: priori Beta(3, 7) y 18 de 40", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Priori", "Beta(3, 7): media 3/10", "0.30 (peso α + β = 10)"],
               ["Datos", "18 adhieren de 40", "y/n = 0.45"],
               ["Parámetros posteriori", f"α = 3 + 18 · β = 7 + 40 − 18", f"Beta({ap}, {bp})"],
               ["Media posteriori", f"{ap}/({ap} + {bp})", f"{f(pd_.mean(), 4)}"],
               ["Como promedio ponderado", "w = 10/50 = 0.20 → 0.2·0.30 + 0.8·0.45", f"{f(0.2 * 0.3 + 0.8 * 0.45, 4)}"],
               ["Moda (MAP)", f"({ap} − 1)/({ap} + {bp} − 2)", f"{f((ap - 1) / (ap + bp - 2), 4)}"],
               ["IC creíble 95 %", "qbeta(0.025; 0.975)", f"[{f(pd_.ppf(0.025), 3)}; {f(pd_.ppf(0.975), 3)}]"]],
              [48 * mm, 62 * mm, CONTENT_W - 110 * mm - 18]),
        P(f"Lectura: la priori (0.30) tira la estimación hacia abajo: la media posteriori {f(pd_.mean(), 3)} queda entre 0.30 y 0.45. P(θ &gt; 0.40 | y) = {f(pd_.sf(0.4), 3)}. "
          "<b>Error común</b>: decir «la verosimilitud es la probabilidad de θ»: es la compatibilidad de los datos con cada θ y no suma 1 sobre θ.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_betabinom.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook y el Bloque 2 del cuaderno", [P("Reproduce los resultados de la profesora (Beta(16, 8), 0.6667, [0.4708; 0.8362]), verifica la conjugación con una grilla y calcula P(θ &gt; 0.6) exacto y por simulación.", "cell")]))

    s.append(h("3", "Intervalo creíble frente a intervalo de confianza"))
    s += para("El <b>creíble</b> sale de la posteriori: P(a &lt; θ &lt; b | y) = 0.95. El de <b>confianza</b> es un procedimiento: repetido muchas veces, contiene a θ en el 95 % de las muestras; una vez calculado, θ está dentro o no lo está. "
              "También conviene saber que hay varios intervalos creíbles de 95 %: el de colas iguales (el que da <font name='Mono'>qbeta</font>) y el HPD (el más corto).")
    y2, n2 = 9, 30
    d2 = B(1 + y2, 1 + n2 - y2)
    s.append(ejemplo("9 éxitos en 30 con priori uniforme", [
        pasos([["Intervalo", "Cálculo", "Resultado"],
               ["Posteriori", "Beta(1 + 9, 1 + 21)", "Beta(10, 22): media 0.3125"],
               ["Creíble 95 % (colas)", "qbeta(0.025, 10, 22) y qbeta(0.975, 10, 22)", f"[{f(d2.ppf(0.025), 3)}; {f(d2.ppf(0.975), 3)}]"],
               ["Confianza de Wald", "0.30 ± 1.96·√(0.3·0.7/30)", f"[{f(wald(y2, n2)[0], 3)}; {f(wald(y2, n2)[1], 3)}]"],
               ["Confianza de Wilson", "corrige el centro y el ancho", f"[{f(wilson(y2, n2)[0], 3)}; {f(wilson(y2, n2)[1], 3)}]"]],
              [44 * mm, 70 * mm, CONTENT_W - 114 * mm - 18]),
        P("Lectura: los intervalos se parecen, pero se leen distinto. Creíble: «con estos datos y esta priori, hay 95 % de probabilidad de que θ esté entre 0.17 y 0.48». "
          "Confianza: «el método captura a θ en el 95 % de las muestras». <b>Error común</b>: dar la segunda lectura con la frase de la primera.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_intervalos.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C3) y el Bloque 4 del cuaderno", [P("Simula miles de muestras, mide la cobertura de cada intervalo y observa por qué Wald falla con n pequeño o θ cerca de 0 o 1.", "cell")]))

    s.append(h("4", "Resumir la posteriori: media, MAP y HPD"))
    s += para("La posteriori completa es el resultado. Para resumirla: la <b>media</b> (α/(α + β)), la <b>mediana</b>, el <b>MAP</b> o moda ((α − 1)/(α + β − 2)) y un intervalo. "
              "Con posterioris simétricas casi coinciden; con asimetría se separan. Con priori plana el MAP es y/n (puede ser 0), y la media nunca llega a los extremos. El <b>HPD</b> es el intervalo creíble más corto.")
    y4, n4 = 2, 15
    d4 = B(1 + y4, 1 + n4 - y4)
    from figuras_c23 import hpd as _hpd
    h4 = _hpd(1 + y4, 1 + n4 - y4)
    s.append(ejemplo("2 éxitos en 15 con priori uniforme: Beta(3, 14)", [
        pasos([["Resumen", "Cálculo", "Resultado"],
               ["Media", "3/(3 + 14)", f"{f(d4.mean(), 4)}"],
               ["MAP (moda)", "(3 − 1)/(3 + 14 − 2) = 2/15", f"{f(2 / 15, 4)} (= y/n)"],
               ["Mediana", "qbeta(0.5; 3, 14)", f"{f(d4.median(), 4)}"],
               ["IC creíble por colas 95 %", "qbeta(0.025) y qbeta(0.975)", f"[{f(d4.ppf(0.025), 3)}; {f(d4.ppf(0.975), 3)}]"],
               ["HPD 95 %", "el intervalo más corto con 95 %", f"[{f(h4[0], 3)}; {f(h4[1], 3)}]"],
               ["Probabilidad de decisión", "1 − pbeta(0.25; 3, 14)", f"P(θ &gt; 0.25 | y) = {f(d4.sf(0.25), 3)}"]],
              [50 * mm, 62 * mm, CONTENT_W - 112 * mm - 18]),
        P(f"Lectura: con la posteriori asimétrica, MAP &lt; mediana &lt; media ({f(2 / 15, 3)}, {f(d4.median(), 3)}, {f(d4.mean(), 3)}). No hay «el» estimador correcto: se reporta la posteriori y, si hace falta un número, se dice cuál es. "
          "<b>Error común</b>: reportar solo la media y olvidar la dispersión: la media sola no muestra cuánta incertidumbre queda.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_resumenes.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C4) y el Bloque 6 del cuaderno", [P("Compara MAP, mediana y media de Beta(16, 8) y de una posteriori asimétrica, y calcula el HPD con una búsqueda numérica.", "cell")]))

    s.append(h("5", "Efecto de la priori, del tamaño y actualización secuencial"))
    s += para("Con las mismas observaciones, una priori más informativa mueve más la posteriori. Se compara con varias prioris razonables y se informa si la conclusión cambia. "
              "Además, la posteriori de un lote es la priori del siguiente: los datos pueden llegar de a poco y el resultado es el mismo que con todos juntos.")
    pr = [("Uniforme Beta(1, 1)", 1, 1), ("Beta(2, 18) pesimista", 2, 18), ("Beta(30, 70) con peso", 30, 70)]
    y3, n3 = 6, 25
    filas = [["Priori", "Posteriori", "Media", "Peso"]]
    for nom, a_, b_ in pr:
        filas.append([nom, f"Beta({a_ + y3}, {b_ + n3 - y3})", f"{f((a_ + y3) / (a_ + b_ + n3), 3)}", f"{f((a_ + b_) / (a_ + b_ + n3) * 100, 0)} %"])
    s.append(ejemplo("6 éxitos de 25 con tres prioris; y dos lotes con priori Beta(2, 2)", [
        pasos(filas, [50 * mm, 40 * mm, 30 * mm, CONTENT_W - 120 * mm - 18]),
        P("Con y/n = 0.24, la uniforme da 0.259; la pesimista, 0.160 (tira hacia 0.10); la de peso 100, 0.256 (cerca de su propia media 0.30). "
          "<b>Por lotes</b>: priori Beta(2, 2); lote 1 con 3 éxitos de 5 → Beta(5, 4); lote 2 con 6 de 10 → Beta(11, 8). Todo junto (9 de 15): Beta(2 + 9, 2 + 6) = Beta(11, 8). Misma posteriori. "
          "<b>Error común</b>: escoger la priori mirando qué resultado se prefiere.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_prioris.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(img("c23_secuencial.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C1 y C2) y el Bloque 5 del cuaderno", [P("Repite el análisis con 8 aprobados de 20 (ejercicio de la profesora) y comprueba que lotes y datos completos dan lo mismo.", "cell")]))

    s.append(h("6", "Otras conjugadas y simulación de Monte Carlo"))
    s += para("La tabla de la diapositiva 29 reúne cuatro pares conjugados. Para <b>Poisson–Gamma</b> (conteos con tasa λ y priori Gamma(a, b) en forma–tasa): λ | y ~ Gamma(a + S, b + m), con S = Σyᵢ y m el número de observaciones. "
              "Cuando la fórmula del intervalo o de una probabilidad es incómoda, se <b>simula</b>: se sacan muchos valores de la posteriori (<font name='Mono'>rvs</font>) y se cuentan (esto es lo que hace <font name='Mono'>rbeta</font> en la práctica).")
    cts = [2, 0, 3, 1, 4, 2]
    ag, bg = 3 + sum(cts), 2 + len(cts)
    dg = stats.gamma(ag, scale=1 / bg)
    sim_ = stats.gamma(ag, scale=1 / bg).rvs(100000, random_state=np.random.default_rng(5))
    s.append(ejemplo("fallas por semana: 2, 0, 3, 1, 4, 2 con priori Gamma(3, 2)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Resumen de datos", "S = 2 + 0 + 3 + 1 + 4 + 2 · m = 6", "S = 12, m = 6"],
               ["Posteriori", "Gamma(3 + 12, 2 + 6)", f"Gamma({ag}, {bg})"],
               ["Media posteriori", f"{ag}/{bg}", f"{f(dg.mean(), 4)} fallas por semana"],
               ["Moda (MAP)", f"({ag} − 1)/{bg}", f"{f((ag - 1) / bg, 4)}"],
               ["IC creíble 95 %", "cuantiles 2.5 % y 97.5 %", f"[{f(dg.ppf(0.025), 3)}; {f(dg.ppf(0.975), 3)}]"],
               ["P(λ &gt; 2) exacto y por simulación", "100 000 valores de la Gamma", f"{f(dg.sf(2), 4)} y {f(float((sim_ > 2).mean()), 4)}"]],
              [50 * mm, 62 * mm, CONTENT_W - 112 * mm - 18]),
        P("Lectura: la priori (media 1.5) y los datos (media 2.0) dan una media posteriori de 1.875. La probabilidad exacta y la simulada difieren solo por el error de Monte Carlo (≈ 0.0015 con 100 000 valores). "
          "<b>Error común</b>: sumar S a la tasa: en Gamma forma–tasa, S se suma a la forma (a + S) y m a la tasa (b + m).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_conjugadas.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D2) y el Bloque 2 del cuaderno", [P("Verifica la conjugación Poisson–Gamma con una grilla (priori × verosimilitud, normalizado) y compara con la fórmula.", "cell")]))

    s.append(h("7", "Puente con la Clase 22: A/B bayesiano"))
    s += para("Con priori Beta(1, 1) en cada grupo, la posteriori de cada tasa es Beta(1 + x, 1 + n − x). Se simula la diferencia θ_B − θ_A y se calculan P(B &gt; A), el intervalo creíble y la probabilidad de superar el MDE.")
    rng = np.random.default_rng(1)
    da, db = B(49, 353).rvs(400000, random_state=rng), B(67, 335).rvs(400000, random_state=rng)
    pb = float((db > da).mean())
    pp = 114 / 800
    zz = (66 / 400 - 48 / 400) / np.sqrt(pp * (1 - pp) * 2 / 400)
    dif = db - da
    s.append(ejemplo("A: 48 de 400 · B: 66 de 400", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Posterioris", "Beta(1 + 48, 1 + 352) y Beta(1 + 66, 1 + 334)", "Beta(49, 353) y Beta(67, 335)"],
               ["Diferencia observada", "66/400 − 48/400", "16.5 % − 12.0 % = +4.5 pp"],
               ["P(B &gt; A)", "400 000 pares simulados", f"{f(pb, 3)}"],
               ["IC creíble 95 % de la mejora", "cuantiles de θ_B − θ_A", f"[{f(float(np.percentile(dif, 2.5)) * 100, 2)}; {f(float(np.percentile(dif, 97.5)) * 100, 2)}] pp"],
               ["Frecuentista (z unilateral)", f"z = {f(zz, 2)}", f"p = {f(float(stats.norm.sf(zz)), 3)} (1 − p = {f(1 - float(stats.norm.sf(zz)), 3)})"],
               ["P(mejora &gt; 2 pp)", "fracción de simulaciones", f"{f(float((dif > 0.02).mean()), 3)}"]],
              [50 * mm, 62 * mm, CONTENT_W - 112 * mm - 18]),
        P("Lectura: P(B &gt; A) y 1 − p casi coinciden con priori plana, pero la primera es una probabilidad sobre la mejora y la segunda no. Con n pequeño el IC incluye el 0 y resulta amplio: 0.97 de probabilidad de que B sea mejor <b>no</b> es lo mismo que «B es claramente mejor». "
          "<b>Error común</b>: tomar P(B &gt; A) como si fuera el tamaño del efecto.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c23_ab.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D1) y el Bloque 7 del cuaderno", [P("Haz el A/B bayesiano con ab_checkout y calcula P(mejora &gt; 1.5 pp), que la Clase 22 no podía responder directamente.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["La posteriori de hoy es la priori de mañana: los datos pueden llegar por lotes.",
                       "R ↔ Python: <font name='Mono'>dbeta</font> ↔ <font name='Mono'>stats.beta.pdf</font>; <font name='Mono'>pbeta</font> ↔ <font name='Mono'>.cdf</font>; <font name='Mono'>qbeta</font> ↔ <font name='Mono'>.ppf</font>; <font name='Mono'>rbeta</font> ↔ <font name='Mono'>.rvs</font>; <font name='Mono'>1 − pbeta</font> ↔ <font name='Mono'>.sf</font>.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    build("apuntes/C23_Apuntes_Bayesiana.pdf", s, "Apuntes · Clase 23 · Estadística bayesiana: fundamentos")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#059669">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C23))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 23 · Estadística bayesiana", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C23))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: P(A | B) = P(B | A)·P(A)/P(B) · P(+) = P(+ | E)P(E) + P(+ | Eᶜ)P(Eᶜ) · θ | y ~ Beta(α + y, β + n − y) · media = α/(α + β) · MAP = (α − 1)/(α + β − 2) · "
                           "sd = √[αβ / ((α + β)²(α + β + 1))] · aproximación normal del IC: media ± 1.96·sd.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C23)], bg=SOFT, border=C23, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "Teorema de Bayes a mano: la paradoja del falso positivo", "★★★", "Un examen de detección tiene sensibilidad 98 % y especificidad 95 %; la enfermedad afecta a 0.5 % de la población.", [
        "Con 100 000 personas, completa el reparto: enfermos, sanos, verdaderos positivos, falsos positivos.",
        "Calcula P(enfermo | positivo) como fracción de positivos y con la fórmula de Bayes.",
        "Explica en una frase por qué el resultado es mucho menor que la sensibilidad.",
        "Si la prevalencia fuera 20 %, ¿cuánto vale la posteriori? ¿Y si una persona con 0.5 % de prevalencia da positivo en dos tests independientes?"])
    s += bloque(2, "Beta–Binomial: parámetros y media posteriori", "★★", "Adherencia a un tratamiento: priori θ ~ Beta(4, 6); en una muestra, 27 de 50 pacientes adhieren.", [
        "Calcula la media y el peso (α + β) de la priori.",
        "Escribe los parámetros de la posteriori y calcula su media y su moda (MAP).",
        "Expresa la media posteriori como promedio ponderado de la media de la priori y y/n.",
        "Con la aproximación normal, calcula la desviación y un intervalo de ≈ 95 %. ¿P(θ &gt; 0.5 | y) es mayor o menor que 0.5?"])
    s += bloque(3, "Distribución predictiva (adelanto de la Clase 24)", "★★★", "Con la posteriori Beta(31, 29) del bloque anterior, llegan 10 pacientes nuevos.", [
        "¿Cuál es el número esperado de pacientes que adhieren? (la media de la predictiva es m·α/(α + β))",
        "¿Por qué la predictiva es más dispersa que una Binomial(10, 0.5167)? Dilo con una frase.",
        "Con la distribución beta-binomial, P(al menos 6 de 10) ≈ 0.425. ¿Qué compararías si usaras solo la estimación puntual 0.5167?"])
    s += bloque(4, "Intervalo creíble frente a intervalo de confianza", "★★", "Indica verdadero o falso y justifica brevemente.", [
        "«Un intervalo creíble del 95 % contiene a θ con probabilidad 0.95, dados los datos y la priori.»",
        "«Un intervalo de confianza del 95 % ya calculado contiene a θ con probabilidad 0.95.»",
        "«Si repetimos el muestreo muchas veces, el 95 % de los intervalos de confianza contiene a θ.»",
        "«El intervalo creíble no depende de la priori.»",
        "«Un intervalo creíble del 95 % significa que el 95 % de los datos están dentro de él.»"])
    s += bloque(5, "Efecto de la priori", "★★", "Se observan 12 éxitos en 20. Tres analistas usan prioris Beta(1, 1), Beta(5, 5) y Beta(2, 18).", [
        "Calcula los parámetros y la media posteriori de cada analista.",
        "Calcula el peso de cada priori, (α + β)/(α + β + n). ¿Cuál influye más?",
        "Con 1 200 éxitos en 2 000 (misma proporción), ¿cuánto valen las tres medias? ¿Qué pasó con las diferencias?"])
    s += bloque(6, "MAP frente a media posteriori", "★★", "Se observan 0 éxitos en 8 intentos.", [
        "Con priori uniforme: posteriori, MAP y media.",
        "Con priori Beta(2, 2): posteriori, MAP y media.",
        "¿Por qué decir «θ = 0» sería arriesgado? ¿Qué resumen prefieres y por qué?"])
    s += bloque(7, "Ejercicio de la diapositiva 16 (finalización de cursos)", "★★", "Tasa histórica de finalización: 40 %. En una nueva cohorte, 12 de 20 finalizan. Priori razonable: Beta(8, 12) (equivale a 20 cursos previos, 8 finalizados).", [
        "Interpretación frecuentista directa: proporción observada y diferencia respecto de 0.40.",
        "Posteriori, media y P(θ &gt; 0.40 | y).",
        "¿Qué incertidumbre queda? Calcula la desviación y un intervalo aproximado del 95 %."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C23))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C23), Spacer(1, 2)]

    # Bloque 1
    pv_ = post_bayes(0.005, 0.98, 0.95)
    p20 = post_bayes(0.20, 0.98, 0.95)
    p2t = post_bayes(pv_, 0.98, 0.95)
    s += sol("Bloque 1", [
        ["1", "Enfermos 500, sanos 99 500. Verdaderos positivos 0.98·500 = 490. Falsos positivos 0.05·99 500 = 4 975."],
        ["2", f"Positivos 490 + 4 975 = 5 465. P(E | +) = 490/5 465 = {f(490 / 5465, 4)}. Fórmula: 0.98·0.005 / (0.98·0.005 + 0.05·0.995) = 0.0049/0.05465 = {f(pv_, 4)}."],
        ["3", "Hay muy pocos enfermos y muchos sanos: el 5 % de falsos positivos sobre 99 500 sanos (4 975) supera a los 490 verdaderos positivos."],
        ["4", f"Con 20 %: 0.98·0.2/(0.98·0.2 + 0.05·0.8) = 0.196/0.236 = {f(p20, 4)}. Con dos tests positivos (priori 0.5 %): la primera posteriori ({f(pv_, 4)}) pasa a ser la priori: {f(p2t, 4)}."]])
    # Bloque 2
    ap, bp = 4 + 27, 6 + 50 - 27
    d = B(ap, bp)
    sd = float(np.sqrt(ap * bp / ((ap + bp) ** 2 * (ap + bp + 1))))
    s += sol("Bloque 2", [
        ["1", "Media 4/10 = 0.40; peso α + β = 10."],
        ["2", f"α = 4 + 27 = {ap}; β = 6 + 50 − 27 = {bp}. Media {ap}/{ap + bp} = {f(d.mean(), 4)}. MAP = ({ap} − 1)/({ap + bp} − 2) = 30/58 = {f(30 / 58, 4)}."],
        ["3", f"w = 10/60 = 0.1667: 0.1667·0.40 + 0.8333·0.54 = {f(10 / 60 * 0.4 + 50 / 60 * 0.54, 4)} (y/n = 27/50 = 0.54)."],
        ["4", f"sd = √[31·29/(60²·61)] = {f(sd, 4)}. Intervalo ≈ {f(d.mean(), 3)} ± 1.96·{f(sd, 3)} = [{f(d.mean() - 1.96 * sd, 3)}; {f(d.mean() + 1.96 * sd, 3)}] (exacto: [{f(d.ppf(0.025), 3)}; {f(d.ppf(0.975), 3)}]). La media 0.517 &gt; 0.5, así que P(θ &gt; 0.5 | y) &gt; 0.5; vale {f(d.sf(0.5), 3)}."]])
    pb = stats.betabinom(10, ap, bp)
    s += sol("Bloque 3", [
        ["1", f"10·31/60 = {f(10 * 31 / 60, 3)} pacientes (≈ 5.2)."],
        ["2", "Porque además de la variabilidad del azar de los 10 pacientes (Binomial), arrastra la incertidumbre sobre θ (la posteriori). La varianza de la predictiva es mayor."],
        ["3", f"Con la estimación puntual 0.5167 se usaría una Binomial(10, 0.5167) y P(≥ 6) = {f(float(stats.binom.sf(5, 10, ap / (ap + bp))), 3)}; la predictiva da {f(float(pb.sf(5)), 3)}. Ambas son parecidas en el centro, pero la predictiva tiene colas más pesadas (más probabilidad de 0–2 y de 9–10)."]])
    s += sol("Bloque 4", [
        ["1", "Verdadero: es la lectura directa de la posteriori (condicionada a priori, modelo y datos)."],
        ["2", "Falso: una vez calculado, el intervalo de confianza contiene a θ o no; la probabilidad 0.95 describe al procedimiento."],
        ["3", "Verdadero: es la interpretación frecuentista de largo plazo."],
        ["4", "Falso: depende de la priori (por eso se compara con varias); con muchos datos la dependencia es pequeña."],
        ["5", "Falso: el intervalo habla del parámetro θ, no de los datos."]])
    rows5 = []
    for nom, a_, b_ in [("Beta(1, 1)", 1, 1), ("Beta(5, 5)", 5, 5), ("Beta(2, 18)", 2, 18)]:
        rows5.append((nom, a_, b_))
    t1 = "; ".join(f"{nom}: Beta({a_ + 12}, {b_ + 8}) → {f((a_ + 12) / (a_ + b_ + 20), 3)}" for nom, a_, b_ in rows5)
    t2 = "; ".join(f"{nom}: {f((a_ + b_) / (a_ + b_ + 20) * 100, 0)} %" for nom, a_, b_ in rows5)
    t3 = ", ".join(f"{f((a_ + 1200) / (a_ + b_ + 2000), 4)}" for nom, a_, b_ in rows5)
    s += sol("Bloque 5", [
        ["1", t1 + "."],
        ["2", t2 + ". Influye más Beta(2, 18), la de mayor α + β: arrastra la media hacia 0.10."],
        ["3", f"Medias con 1 200 de 2 000: {t3}. Las diferencias casi desaparecen: con más datos la priori pesa menos (peso ≤ 1 %)."]])
    s += sol("Bloque 6", [
        ["1", "Beta(1, 9). MAP = (1 − 1)/(1 + 9 − 2) = 0. Media 1/10 = 0.10."],
        ["2", "Beta(2, 10). MAP = (2 − 1)/(2 + 10 − 2) = 0.10. Media 2/12 = 0.167."],
        ["3", "Con 8 observaciones, θ = 0 es una conclusión muy fuerte (descarta cualquier probabilidad pequeña pero positiva). La media (o la mediana) y el intervalo creíble muestran que θ pudo ser 0.1 o incluso 0.3. Prefiero la media o el intervalo creíble."]])
    d7 = B(20, 20)
    s += sol("Bloque 7", [
        ["1", f"p̂ = 12/20 = 0.60; diferencia = +0.20 respecto de 0.40. Una prueba binomial unilateral da p = {f(float(stats.binom.sf(11, 20, 0.4)), 3)}: poca evidencia frente a 0.05."],
        ["2", f"Beta(8 + 12, 12 + 8) = Beta(20, 20); media 0.50. P(θ &gt; 0.40 | y) = {f(d7.sf(0.4), 3)}."],
        ["3", f"sd = √[20·20/(40²·41)] = {f(d7.std(), 3)}. Intervalo ≈ 0.50 ± 1.96·{f(d7.std(), 3)} = [{f(0.5 - 1.96 * d7.std(), 3)}; {f(0.5 + 1.96 * d7.std(), 3)}] (exacto [{f(d7.ppf(0.025), 3)}; {f(d7.ppf(0.975), 3)}]). Sigue siendo amplio: la priori (20 cursos) tira la estimación hacia 0.40."]])
    build("apuntes/C23_Ejercicios_Cuaderno_Bayesiana.pdf", s, "Ejercicios para el cuaderno · Clase 23")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
