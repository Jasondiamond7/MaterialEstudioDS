"""Clase 24 · Estadística bayesiana, parte 2: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c24_numeros.json (lo escribe figuras_c24.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from math import sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer
from scipy import stats

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)
from figuras_c24 import normal_normal

N = json.loads((Path(__file__).resolve().parent / "c24_numeros.json").read_text(encoding="utf-8"))
C24 = colors.HexColor("#047857")
C24_BG = colors.HexColor("#ECFDF5")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + Clase 24.pdf + Clase24.ipynb (R, rstanarm) de la profesora Maritza Márquez"
NO, PD, MC, PO, TZ, BI, AC, RH = (N[k] for k in ("normal", "pred", "mc", "pima", "traza", "burnin", "acf", "rhat"))
PP, VF, PS = N["pred_pima"], N["verif"], N["pasos"]
R_ = PO["res"]


def f(x, d=2):
    return f"{x:.{d}f}"


def ci(v, d=3):
    return f"[{v[0]:.{d}f}; {v[1]:.{d}f}]"


def mil(x):
    return f"{int(round(x)):,}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.78):
    s.append(band(f"Clase 24 · {titulo}", FUENTE, C24))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C24, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C24))
    s.append(Spacer(1, 1))
    for fig, pie, *w in figuras:
        s.append(img(fig, width=CONTENT_W * (w[0] if w else fw)))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C24)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 24 · Estadística bayesiana, parte 2: posteriori, predicción, Monte Carlo y MCMC",
                  "Profesora Maritza Márquez · continúa la Clase 23 (priori, verosimilitud, Beta–Binomial) y se conecta con GLM (C20–21), A/B testing (C22) y bootstrap (C43)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Cuando la posteriori <b>tiene fórmula</b> (Beta–Binomial, Normal–Normal) todo se calcula con álgebra: la media posteriori es un promedio ponderado por <b>precisiones</b>. "
                    "La <b>predicción</b> no usa un solo valor de θ, sino todos, ponderados por la posteriori. Cuando una cantidad es una integral difícil se <b>simula</b> (Monte Carlo: integral = promedio de muestras). "
                    "Y cuando ni siquiera se sabe simular de la posteriori, porque solo se conoce <b>proporcional</b> a priori × verosimilitud, se construye una <b>cadena de Markov</b> (MCMC, p. ej. Metropolis–Hastings) "
                    "cuya distribución de equilibrio es la posteriori. Una cadena <b>no se acepta a ciegas</b>: se revisan la <b>traza</b>, el <b>burn-in</b>, la <b>autocorrelación</b>, el <b>tamaño efectivo</b> y el <b>R-hat</b> de varias cadenas.", "body")],
                 bg=C24_BG, border=C24, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c24_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Priori × verosimilitud <font name='Sym'>∝</font> posteriori</b>, Beta–Binomial e intervalo creíble (Clase 23)",
                     "<b>Regresión logística y odds ratio</b> (Clases 20–21): hoy se ajusta con el enfoque bayesiano",
                     "<b>IC, valor p y bootstrap</b> (Clases 22 y 43): se comparan con la posteriori",
                     "<b>scipy y numpy</b>: <font name='Mono'>stats.norm</font>, <font name='Mono'>stats.beta</font>, <font name='Mono'>rng.normal</font>; el muestreador se escribe a mano"], "cell", C24)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Normal–Normal</b> → <b>predictiva</b> → <b>Monte Carlo</b> → <b>por qué MCMC</b> → <b>Metropolis paso a paso</b> → <b>traza, burn-in, ESS, R-hat</b> → <b>regresión logística bayesiana</b>",
                      f"<b>Pima.tr</b> (paquete MASS de R): {PO['n']} mujeres, {PO['n_si']} con diabetes y {PO['n_no']} sin ella; predictores glucosa, IMC y edad (estandarizados). Está en <font name='Mono'>Guias/datos/pima_tr.csv</font>",
                      "Pregunta guía: ¿qué dicen los datos sobre el riesgo de diabetes y cuánta incertidumbre queda en cada coeficiente?"], "cell", C24)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con otras clases (sin repetirlas)</b>", "cellb"),
                  P(f"<b>C22 → C24</b>: en lugar de «¿p &lt; 0.05?» se pregunta P(θ &gt; c | y); aquí P(μ &gt; 65 | y) = {f(NO['p65'], 3)}. "
                    f"<b>C17–21 → C24</b>: la regresión logística es la de GLM; con 200 datos y prioris suaves la posteriori casi coincide con la máxima verosimilitud (glu_z: clásico {f(R_['glu_z']['mle'], 3)}, bayesiano {f(R_['glu_z']['larga_media'], 3)}), pero se interpreta como probabilidad. "
                    f"<b>C23 → C24</b>: la Beta(16, 8) de la clase pasada sirve para predecir la próxima cohorte (P(menos de 5 aprueben) = {f(PD['p_menos5'], 3)}). "
                    f"<b>C43 → C24</b>: el bootstrap mide la variación del estimador al remuestrear datos (sd {f(PP['boot']['sd'], 3)} para glu_z); la posteriori mide la incertidumbre dado un modelo y una priori (sd {f(PP['post']['sd'], 3)}).", "cell")], bg=SOFT, border=C24, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: «cadena de Markov» sin fórmulas</b>", "cellb"),
                  P("Una cadena de Markov es una secuencia de valores donde <b>el siguiente depende solo del actual</b> (no de todo el pasado). Metropolis–Hastings la diseña para que, a la larga, visite cada valor de θ "
                    "con frecuencia proporcional a la posteriori. Los valores consecutivos <b>no son independientes</b>: por eso hay que medir cuánta información real aportan (tamaño efectivo).", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Inferencia posteriori y el modelo Normal–Normal",
        "Con datos normales de varianza conocida y priori normal, la posteriori es normal: su precisión es la suma de precisiones y su media, un <b>promedio ponderado</b> por precisiones. Con más datos, la priori pesa cada vez menos pero nunca desaparece.",
        [("La posteriori, objeto central", "En vez de un estimador y un p-valor, el resultado es p(θ | y): de ahí salen estimaciones puntuales, intervalos creíbles, probabilidades posteriores P(θ &gt; c | y), predicciones y decisiones."),
         ("Normal–Normal", "Yᵢ | μ ~ N(μ, σ²) con σ² conocida y μ ~ N(m₀, s₀²). Entonces μ | y ~ N(mₙ, sₙ²): 1/sₙ² = 1/s₀² + n/σ² y mₙ = sₙ²·(m₀/s₀² + n·ȳ/σ²)."),
         ("Precisión = 1/varianza", "La priori aporta 1/s₀²; los datos, n/σ². Gana quien tiene más precisión. Si σ² fuera desconocida, el problema es otro (priori Normal–Gamma inversa; la marginal de μ es una t).")],
        [("c24_normal.png", f"Ejemplo de la clase: priori N(60, 15²), n = 25, ȳ = 66, σ = 10. La priori vale {f(NO['prec_priori'], 4)} de precisión y los datos {f(NO['prec_datos'], 2)}: la posteriori N({f(NO['media'], 2)}, {f(NO['sd'], 2)}²) queda casi sobre ȳ. "
                         f"Derecha: n = 1, 5, 25 y 100 con el mismo ȳ; la media pasa de {f(NO['por_n']['1'][0], 2)} a {f(NO['por_n']['100'][0], 2)} y la desviación baja de {f(NO['por_n']['1'][1], 2)} a {f(NO['por_n']['100'][1], 2)}.", 0.98)],
        ["1/sₙ² = 1/s₀² + n/σ²", "mₙ = sₙ²·(m₀/s₀² + n·ȳ/σ²)", "mₙ = w·m₀ + (1 − w)·ȳ,  w = (1/s₀²)/(1/sₙ²)", "P(μ &gt; c | y) = 1 − Φ((c − mₙ)/sₙ)"],
        ["from figuras_c24 import normal_normal", "m, v = normal_normal(60, 15, 10, 25, 66)", "stats.norm(m, v**0.5).sf(65)   # P(μ > 65 | y)"],
        [f"sₙ² = 1/(1/225 + 25/100) = {f(NO['var'], 4)}; sₙ = {f(NO['sd'], 4)}; mₙ = {f(NO['media'], 4)}",
         f"Peso de la priori w = {f(NO['w_priori'], 4)} ({f(NO['w_priori'] * 100, 1)} %); de los datos, {f((1 - NO['w_priori']) * 100, 1)} %",
         f"P(μ &gt; 65 | y) = {f(NO['p65'], 4)}; intervalo creíble 95 % {ci(NO['ic'], 2)}",
         "Pregunta de la clase: al aumentar n, la priori <b>no desaparece</b>, pierde peso relativo frente a la verosimilitud"],
        ["En el notebook la función se llama <font name='Mono'>normal_normal(m0, s0, sigma, n, ybar)</font>; s₀ y σ son desviaciones (no varianzas)."])

    topic_page(
        s, "Distribución predictiva y simulación de Monte Carlo",
        "Para predecir un dato nuevo Ỹ se promedia sobre <b>todos</b> los valores de θ según la posteriori: la predictiva tiene <b>más dispersión</b> que usar un θ fijo. Y casi cualquier integral posterior se aproxima con un <b>promedio de simulaciones</b>.",
        [("Predictiva posterior", "p(ỹ | y) = ∫ p(ỹ | θ) p(θ | y) dθ. Dos fuentes de variación: el azar del dato nuevo y la incertidumbre sobre θ. Se simula: θ⁽ˢ⁾ ~ p(θ | y) y luego Ỹ⁽ˢ⁾ | θ⁽ˢ⁾."),
         ("Monte Carlo", "I = E[g(θ) | y] ≈ (1/M) Σ g(θ⁽ᵐ⁾) con muestras independientes de la posteriori. Una probabilidad es un promedio de indicadores: P(θ &gt; 0.6 | y) ≈ fracción de θ⁽ᵐ⁾ &gt; 0.6."),
         ("Error de simulación", "El error estándar baja como 1/√M: para dividirlo por 10 hacen falta 100 veces más simulaciones. Es error de cálculo, no incertidumbre sobre θ (esa la da la posteriori).")],
        [("c24_predictiva.png", f"Izquierda: con Beta(16, 8) y m = 10 la predictiva (beta-binomial, sd {f(PD['sd'], 2)}) es más ancha que la Binomial(10, 0.667) con θ fijo (sd {f(PD['sd_plug'], 2)}): P(menos de 5) = {f(PD['p_menos5'], 3)} frente a {f(PD['p_menos5_plug'], 3)}. "
                                f"Centro: en el caso normal las varianzas se suman (σ² + sₙ²). Derecha: 20 000 pares simulados dan P(Ỹ &lt; 5) ≈ {f(PD['p_menos5_sim'], 3)}.", 0.92),
         ("c24_montecarlo.png", f"Con M = {MC['M']} simulaciones de la Beta(16, 8), {MC['cuenta']} superan 0.6: P ≈ {f(MC['estimado'], 3)} (exacta {f(MC['exacta'], 4)}, error estándar ≈ {f(MC['se'], 4)}). Centro: el error baja como 1/√M. "
                               "Derecha: esperanzas y probabilidades, todas como promedios.", 0.92)],
        ["Ỹ | θ ~ Binomial(m, θ),  θ | y ~ Beta(α, β)", "Var(Ỹ) = m·αβ(α + β + m) / ((α + β)²(α + β + 1))", "Var(Ỹ_normal) = σ² + sₙ²", "SE_MC ≈ √(p̂(1 − p̂)/M)"],
        ["th = stats.beta(16, 8).rvs(20000)", "y_nuevo = rng.binomial(10, th)", "(y_nuevo < 5).mean()   # P(Ỹ < 5)"],
        [f"Beta(16, 8), m = 10: media {f(PD['media'], 3)}; varianza {f(PD['var'], 3)} (con θ fijo: {f(PD['var_plug'], 3)})",
         f"P(Ỹ ≥ 7) = {f(PD['p_7_mas'], 3)} (parecido al plug-in {f(PD['p_7_mas_plug'], 3)}): en el centro coinciden, las colas difieren",
         f"Normal: sd predictiva {f(NO['pred_sd'], 3)} frente a σ = 10; P(nuevo estudiante &gt; 65) = {f(NO['p_nueva_mayor_65'], 3)}",
         "La clase dijo «no basta decir que aprobarán 7; conviene decir qué tan probable es que aprueben menos de 5»"],
        ["Un promedio de Monte Carlo con M = 1 000 tiene error de ≈ 0.013 en una probabilidad cercana a 0.76: no reportes más de dos decimales."], fw=0.8)

    topic_page(
        s, "Por qué MCMC y Metropolis paso a paso",
        "Si la posteriori solo se conoce <b>salvo una constante</b>, no se puede simular de ella directamente ni integrar con una grilla. <b>Metropolis</b> propone un valor cerca del actual, compara su plausibilidad según la posteriori y lo <b>acepta o rechaza</b>: la cadena resultante recorre la posteriori.",
        [("El problema", "p(θ | y) = p(y | θ)p(θ)/p(y), y el denominador p(y) = ∫ p(y | θ)p(θ) dθ es una integral de d dimensiones. En la regresión logística no hay familia conocida y una grilla cuesta 100^d evaluaciones."),
         ("Algoritmo de Metropolis", "Desde θ: propone θ* ~ N(θ, τ²); calcula r = p(θ* | y)/p(θ | y) (la constante se cancela); acepta con probabilidad α = min(1, r). Si acepta, θ ← θ*; si no, la cadena se repite en θ."),
         ("Qué se obtiene", "Una cadena θ⁽¹⁾, θ⁽²⁾, … cuya distribución de largo plazo es la posteriori; se usa como una muestra, pero con valores <b>dependientes</b>.")],
        [("c24_porque_mcmc.png", "Izquierda: contorno de la posteriori de (β₀, β_glu) en la regresión de Pima con puntos de la cadena. Centro: una grilla de 100 valores por parámetro explota con d. Derecha: qué herramienta usar.", 0.8),
         ("c24_metropolis.png", f"Seis pasos reales con la Beta(16, 8) como objetivo (núcleo θ¹⁵(1 − θ)⁷), desde θ = 0.55 y paso τ = {N['pasos_paso']}: sube la cadena cuando r &gt; 1 (acepta siempre), y baja a veces (r &lt; 1 acepta con probabilidad r); los pasos 4 y 6 se rechazan y la cadena <b>se repite</b>.", 0.58)],
        ["r = p(θ* | y)/p(θ | y) = [p(y | θ*)p(θ*)]/[p(y | θ)p(θ)]", "α = min(1, r);  aceptar si u &lt; α, con u ~ U(0, 1)", "propuesta asimétrica: r = [p(θ*)q(θ | θ*)]/[p(θ)q(θ* | θ)]"],
        ["u = rng.uniform()", "if np.log(u) < logp(prop) - logp(x):", "    x = prop          # acepta; si no, x queda igual"],
        [f"Paso 1: r = {f(PS[0]['razon'], 2)}, u = {f(PS[0]['u'], 2)} → acepta (θ: {f(PS[0]['actual'], 2)} → {f(PS[0]['propuesta'], 2)})",
         f"Paso 3 (bajada): r = {f(PS[2]['razon'], 2)} &lt; 1, u = {f(PS[2]['u'], 2)} &lt; α → acepta igual: así la cadena explora las colas",
         f"Paso 4: r = {f(PS[3]['razon'], 2)}, u = {f(PS[3]['u'], 2)} ≥ α → rechaza y se queda en {f(PS[3]['actual'], 2)}",
         "Una propuesta fuera del soporte (θ ≤ 0 o ≥ 1) tiene densidad 0: r = 0 y siempre se rechaza"],
        ["Las diapositivas describen la propuesta «cercana»; con una propuesta <b>asimétrica</b> hay que incluir el cociente de las densidades de propuesta (corrección de Hastings), o la cadena converge a otra distribución (ver alerta)."], fw=0.8)

    topic_page(
        s, "Diagnóstico visual: traza, tamaño de paso y burn-in",
        "Una cadena <b>no es un resultado</b> hasta revisarla. La <b>traza</b> debe oscilar rápido alrededor de una zona estable, sin tendencias ni tramos pegados; el <b>tamaño de paso</b> decide si lo hace; y el comienzo (<b>burn-in</b>) se descarta.",
        [("Traza", "Gráfico del valor simulado frente a la iteración. Buena: «oruga peluda» estable. Mala: tendencia, saltos largos, periodos inmóviles. Se acompaña del histograma o la densidad."),
         ("Mezcla y tamaño de paso", "Paso muy pequeño: casi todo se acepta pero la cadena avanza a paso de tortuga. Paso muy grande: casi todo se rechaza y se queda pegada. El punto medio mezcla bien (aceptación ≈ 25–50 % en dimensiones moderadas)."),
         ("Burn-in (calentamiento)", "Las primeras iteraciones dependen del punto de partida; se descartan. En el cuaderno: 4 cadenas, 2 000 iteraciones, las primeras 1 000 de «warmup». No arregla un modelo mal especificado ni una muestra pequeña.")],
        [("c24_traza.png", "Misma posteriori Beta(16, 8) y tres tamaños de paso (2 000 iteraciones desde 0.5). Con τ = 0.005 la cadena tarda cientos de iteraciones en recorrer lo que una buena cadena recorre en diez; con τ = 1.5 se queda pegada (92 % de rechazos).", 0.6),
         ("c24_burnin.png", f"Dos cadenas que parten de 0.03 y 0.97 se encuentran en la zona de la posteriori. Derecha: sesgo medio de la media de la cadena según las iteraciones descartadas (promedio de {BI['R']} cadenas de {BI['L']} iteraciones desde 0.03): {f(BI['sesgo']['0'], 4)} sin descartar y ≈ 0 después de unas 50–100.", 0.7)],
        ["aceptación = aceptadas/iteraciones", "descarta las primeras b iteraciones: cadena[b:]", "estimación = promedio de cadena[b:]"],
        ["ch, acep = metropolis(logp, 0.5, 0.12, 2000, rng)", "ch = ch[150:]   # sin burn-in"],
        [f"τ = 0.005: aceptación {f(TZ[0]['aceptacion'] * 100, 0)} %, ESS {f(TZ[0]['ess'], 0)} de 2 000; media {f(TZ[0]['media'], 3)} (exacta 0.667)",
         f"τ = 0.12: aceptación {f(TZ[1]['aceptacion'] * 100, 0)} %, ESS {f(TZ[1]['ess'], 0)}; media {f(TZ[1]['media'], 3)}",
         f"τ = 1.5: aceptación {f(TZ[2]['aceptacion'] * 100, 0)} %, ESS {f(TZ[2]['ess'], 0)}; media {f(TZ[2]['media'], 3)}",
         "Aceptar mucho no es buena señal: aceptación 98 % con paso minúsculo da una cadena casi inmóvil"],
        ["Una traza «bonita» a ojo no basta; la misma cadena con paso minúsculo parece suave. Se cuantifican la autocorrelación, el ESS y el R-hat (página siguiente)."], fw=0.8)

    topic_page(
        s, "Autocorrelación, tamaño efectivo y R-hat",
        "Los valores de una cadena están correlacionados: 5 000 iteraciones pueden valer solo unas decenas de muestras independientes. El <b>tamaño efectivo (ESS)</b> lo mide, y el <b>R-hat</b> compara varias cadenas con puntos de partida distintos para detectar falta de convergencia.",
        [("Autocorrelación y ESS", "ρₖ = correlación entre θ⁽ᵗ⁾ y θ⁽ᵗ⁺ᵏ⁾. ESS ≈ M / (1 + 2 Σ ρₖ): cuántas muestras independientes «vale» la cadena. El error de Monte Carlo de la media es sd/√ESS, no sd/√M."),
         ("Thinning (adelgazar)", "Guardar 1 de cada k reduce la autocorrelación del resultado pero <b>no crea información</b>: el ESS no sube (baja si se tira información). Solo ahorra memoria; con ESS bajo hay que mejorar el muestreador o correr más."),
         ("Varias cadenas y R-hat", "Se parten de valores dispersos. R-hat compara la varianza entre cadenas con la de dentro: ≈ 1 si todas exploran lo mismo; &gt; 1.01 sugiere no convergencia. Es necesario, no suficiente.")],
        [("c24_autocorr.png", f"5 000 iteraciones desde 0.68. La cadena con paso razonable pierde la memoria en ≈ 10 rezagos (ESS {f(AC['res'][1]['ess'], 0)}); la de paso minúsculo conserva autocorrelación {f(AC['res'][0]['ac50'], 2)} a 50 rezagos (ESS {f(AC['res'][0]['ess'], 0)}). Derecha: adelgazar la cadena razonable de {AC['thin'][0]['n']} a {AC['thin'][1]['n']} valores baja el ESS de {f(AC['thin'][0]['ess'], 0)} a {f(AC['thin'][1]['ess'], 0)}.", 0.74),
         ("c24_rhat.png", f"Cuatro cadenas desde 0.05, 0.3, 0.7 y 0.95. Con 120 iteraciones R-hat = {f(RH['rhat_120'], 3)} (no han convergido); con 3 000, {f(RH['rhat_3000'], 3)} y ESS ≈ {f(RH['ess_3000'], 0)}. A la derecha, R-hat decrece hacia 1 al alargar las cadenas.", 0.74)],
        ["ESS ≈ M / (1 + 2 Σₖ ρₖ)", "SE_MCMC ≈ sd / √ESS", "R-hat = √[(varianza combinada)/(varianza dentro de cadenas)]", "criterio habitual: R-hat &lt; 1.01 y ESS <font name='Sym'>≳</font> 400 en total"],
        ["ess, rhat = ess_rhat(cadenas)   # (m, n)", "autocorr(cadena, 50)[10]          # ρ₁₀"],
        [f"Cuaderno de la profesora (Stan): R-hat 1.000, 1.000, 1.002 y 1.001; n_eff entre 3 668 y 4 139 de 4 000",
         f"Mi Metropolis a mano con la misma configuración (4 × 1 000 válidas): ESS entre {f(min(r['ess'] for r in R_.values()), 0)} y {f(max(r['ess'] for r in R_.values()), 0)} y R-hat hasta {f(max(r['rhat'] for r in R_.values()), 3)}",
         f"Con 4 × 25 000 iteraciones: ESS {f(min(r['larga_ess'] for r in R_.values()), 0)}–{f(max(r['larga_ess'] for r in R_.values()), 0)} y R-hat {f(max(r['larga_rhat'] for r in R_.values()), 4)}",
         "Un Metropolis de paseo aleatorio es mucho menos eficiente que el NUTS de Stan"],
        ["R-hat cercano a 1 no demuestra convergencia si todas las cadenas se quedan en la misma región equivocada, y un ESS alto no corrige un modelo mal planteado."], fw=0.8)

    topic_page(
        s, "Regresión logística bayesiana con los datos Pima",
        "La regresión logística de GLM se ajusta con el enfoque bayesiano asignando <b>prioris a los coeficientes</b>; la posteriori no tiene fórmula, así que se muestrea con MCMC y <b>todo</b> (medianas, intervalos creíbles, odds ratios, predicciones) sale de las muestras.",
        [("El modelo", "diabetesᵢ ~ Bernoulli(pᵢ), logit(pᵢ) = β₀ + β₁·glu_z + β₂·bmi_z + β₃·age_z. Los predictores se <b>estandarizan</b> para que una misma priori (aquí N(0, 1) en los coeficientes y N(0, 2.5²) en el intercepto) sea razonable para todos."),
         ("Ajuste", "4 cadenas, 2 000 iteraciones, 1 000 de calentamiento y semilla fija (rstanarm usa NUTS; aquí se reproduce con Metropolis). Antes de interpretar: trazas, R-hat y ESS."),
         ("Resúmenes posteriores", "Media, mediana, intervalo creíble 95 % por cuantiles, P(β &gt; 0 | y) como promedio de indicadores y odds ratio = exp(β). Con glucosa estandarizada, OR compara a pacientes que difieren en <b>una desviación estándar</b>.")],
        [("c24_regresion.png", f"Izquierda: la glucosa mediana es {f(PO['med_glu'][0], 0)} sin diabetes y {f(PO['med_glu'][1], 0)} con ella. Centro: las tres posterioris frente a la priori N(0, 1). Derecha: mediana e IC creíble bayesianos y estimación e IC de Wald de glm: casi coinciden (n = 200 y priori suave).", 0.74),
         ("c24_predicciones.png", f"Perfil A (glu 100, IMC 25, edad 30): {f(PP['perfiles'][0]['media'], 3)} {ci([PP['perfiles'][0]['lo'], PP['perfiles'][0]['hi']])}; Perfil B (160, 35, 55): {f(PP['perfiles'][1]['media'], 3)} {ci([PP['perfiles'][1]['lo'], PP['perfiles'][1]['hi']])}. Centro: la cantidad observada de diabéticos ({PP['obs']}) cae en el centro de lo simulado. Derecha: bootstrap del glm y posteriori para glu_z.", 0.74)],
        ["logit(p) = β₀ + β₁z₁ + β₂z₂ + β₃z₃", "OR = exp(β);  IC creíble de OR = exp(IC creíble de β)", "P(β &gt; 0 | y) = mean(muestras &gt; 0)"],
        ["m = cadenas.reshape(-1, 4)       # 4 000 × 4", "np.quantile(m[:, 1], [0.025, 0.975])", "np.exp(np.median(m[:, 1]))        # OR de glu_z"],
        [f"glu_z: mediana {f(R_['glu_z']['larga_mediana'], 3)}, IC creíble {ci([R_['glu_z']['larga_q025'], R_['glu_z']['larga_q975']])}, OR {f(np.exp(R_['glu_z']['larga_mediana']), 2)}; rstanarm: 0.966 y [0.580; 1.386]",
         f"bmi_z: {f(R_['bmi_z']['larga_mediana'], 3)} (rstanarm 0.548); age_z: {f(R_['age_z']['larga_mediana'], 3)} (rstanarm 0.573); intercepto {f(R_['(Intercept)']['larga_mediana'], 3)} (rstanarm −0.921)",
         f"P(β &gt; 0 | y): {f(R_['glu_z']['p_pos'], 3)}, {f(R_['bmi_z']['p_pos'], 3)} y {f(R_['age_z']['p_pos'], 3)}: los tres intervalos quedan a la derecha de cero",
         f"Exactitud aparente {f(PP['exactitud'], 2)} (113, 19, 29 y 39 en la matriz): es dentro de la muestra y optimista"],
        ["«Intervalo de confianza para los odds ratios» (se dijo en clase y luego se corrigió): son intervalos <b>creíbles</b>. Y los resultados de rstanarm y de este muestreador difieren en el segundo decimal por error de Monte Carlo."], fw=0.8)

    # --- página de cierre
    s.append(band("Clase 24 · Alertas, verificación y cierre", FUENTE, C24))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase y en las diapositivas (verificado)</b>", "cellb"),
                  bullets([f"<b>El método del cuaderno no es Metropolis–Hastings</b>: <font name='Mono'>stan_glm</font> de rstanarm usa <b>HMC/NUTS</b>, no el algoritmo explicado en clase ni en la diapositiva 24. Además, «warmup» en Stan también <b>adapta</b> el paso (no es solo burn-in). La profesora lo dijo al final: MH «se utilizó para crear las cadenas»; en el cuaderno no es así.",
                           f"<b>La corrección de Hastings</b>: la diapositiva describe la propuesta como «cercana» y acepta según la razón de posterioris. Con propuesta <b>asimétrica</b> falta el cociente de propuestas: en el experimento de este material (Beta(16, 8), propuesta con deriva +0.08) la media sin corrección sale {f(VF['asimetrica']['media_sin_correccion'], 3)} y con corrección {f(VF['asimetrica']['media_con_correccion'], 3)} (exacta {f(VF['asimetrica']['exacta'], 3)}).",
                           "<b>«Más simulaciones o cadenas puede hacer que no converja»</b>: impreciso. En Monte Carlo ordinario el error baja siempre como 1/√M; en MCMC más iteraciones no empeoran la estimación (solo el tiempo); lo que no mejora con más cadenas ni más iteraciones es un <b>modelo mal especificado</b> o una priori inadecuada (la diapositiva 26 lo dice bien).",
                           "<b>«Monte Carlo es cuando no puedo calcular la integral; MCMC cuando ni siquiera normalizada puedo tener la expresión»</b>: más preciso: el Monte Carlo ordinario exige <b>saber simular</b> de la posteriori; MCMC solo exige evaluarla salvo una constante.",
                           f"<b>Diapositiva 20, «mientras mayor sea M, menor el error»</b>: vale con muestras independientes; con una cadena depende del ESS (5 000 iteraciones con ESS {f(AC['res'][0]['ess'], 0)} no sirven de nada).",
                           "<b>Diapositiva 28</b> anuncia una práctica <font name='Mono'>Clase24_Bayes_R.Rmd</font> con Normal–Normal, predictiva, A/B por Monte Carlo y Metropolis (y el título dice «R y Python»); el material entregado es el cuaderno <font name='Mono'>Clase24.ipynb</font> (solo R: regresión logística con rstanarm). Esta guía cubre ambas cosas en Python.",
                           f"<b>Cuaderno de la profesora</b>: <font name='Mono'>posterior_linpred(transform = TRUE)</font> está obsoleto (usar <font name='Mono'>posterior_epred</font>); la exactitud 0.76 es <b>aparente</b> (mismos datos); «ejemplo con datos real» es un error de tipeo. «R ajustado / reajustado» en la transcripción es el R-hat.",
                           "<b>«Python es una caja negra y R está bien documentado»</b>: es una opinión; Stan, PyMC y brms están igual de documentados. Lo importante es saber cuántas cadenas, iteraciones y qué diagnósticos se usaron."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Verificación de este material</b>", "cellb"),
                  bullets([f"Normal–Normal y predictiva por fórmula y por simulación (20 000 pares): P(Ỹ &lt; 5) {f(PD['p_menos5'], 4)} exacta frente a {f(PD['p_menos5_sim'], 4)} simulada.",
                           f"Cuaderno de la profesora reproducido con un Metropolis por bloques en numpy (sin Stan): con 4 × 25 000 iteraciones las medianas son {f(R_['(Intercept)']['larga_mediana'], 3)}, {f(R_['glu_z']['larga_mediana'], 3)}, {f(R_['bmi_z']['larga_mediana'], 3)} y {f(R_['age_z']['larga_mediana'], 3)} frente a −0.921, 0.966, 0.548 y 0.573 de rstanarm; glm clásico: {f(R_['(Intercept)']['mle'], 4)}, {f(R_['glu_z']['mle'], 4)}, {f(R_['bmi_z']['mle'], 4)} y {f(R_['age_z']['mle'], 4)} (los mismos de la profesora).",
                           f"Datos: Pima.tr, {PO['n']} filas, {PO['n_si']} «Yes» y {PO['n_no']} «No» (68 y 132 como en clase); media de aceptación del muestreador {f(PO['aceptacion_media'], 2)}.",
                           "<b>Transcripción con ruido</b> (ya corregido): «vallesiana/valleciana» = bayesiana, «Burkin/burin» = burn-in, «Geek Sample» = Gibbs sampler, «Nux» = NUTS, «SAEN» = SAEM, «math SD» = MAD_SD, «optradio» = odds ratio. Se marcó el recreo entre 1:26:49 y 1:48:15 (el video no tiene audio en ese tramo)."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Calculo a mano la media y la varianza posteriores Normal–Normal y las leo como promedio ponderado por precisiones",
                           "Distingo la posteriori de la predictiva y calculo la varianza predictiva (Normal y beta-binomial)",
                           "Aproximo probabilidades y esperanzas por Monte Carlo y calculo su error estándar",
                           "Explico por qué se necesita MCMC y hago un paso de Metropolis a mano (razón, α, u)",
                           "Leo una traza, el burn-in, la autocorrelación, el ESS y el R-hat y decido si confiar en la cadena",
                           "Resumo una posteriori con muestras: mediana, intervalo creíble, P(β &gt; 0), odds ratio",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Siguiente</b>: series de tiempo (Clases 25–27) y, más adelante, la evaluación de modelos con bootstrap (Clase 43)."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C24_Guia_Bayesiana2.pdf", s, "Clase 24 · Estadística bayesiana, parte 2")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#047857">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 24 · Estadística bayesiana, parte 2", "Para leer antes de los ejercicios · lectura de ~30 min", C24))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Normal–Normal pondera por precisiones", "mₙ = promedio de m₀ y ȳ con pesos 1/s₀² y n/σ²; la precisión posterior es la suma."),
                            ("La predictiva suma incertidumbres", "Var(Ỹ) = σ² + sₙ² (Normal): más ancha que usar θ fijo."),
                            ("Monte Carlo: integral = promedio", "El error baja como 1/√M; con dependencia, como 1/√ESS."),
                            ("MCMC cuando solo hay una proporcionalidad", "Metropolis: propone, calcula r, acepta con min(1, r)."),
                            ("Revisa la cadena antes de usarla", "Traza, burn-in, autocorrelación, ESS y R-hat de varias cadenas."),
                            ("Todo sale de las muestras", "Mediana, intervalo creíble, P(β &gt; 0) y odds ratio de la regresión logística.")], C24)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C24_Bayesiana2.ipynb</font> o el cuaderno <font name='Mono'>C24_Ejercicios_Cuaderno_Bayesiana2.pdf</font>.", "small"))

    # 1
    s.append(h("1", "Modelo Normal con varianza conocida"))
    s += para("Se observa una media muestral ȳ de n datos con desviación σ conocida y se tiene una priori N(m₀, s₀²) para μ. Se convierte todo a <b>precisiones</b> (1/varianza): la de la priori es 1/s₀² y la de los datos n/σ². La precisión posterior es su suma; la media posterior, el promedio de m₀ y ȳ ponderado por precisiones.")
    m0, s0, sg, n1, yb = 100, 20, 15, 9, 88
    mn, vn = normal_normal(m0, s0, sg, n1, yb)
    w = (1 / s0 ** 2) / (1 / vn)
    s.append(ejemplo("tiempo de respuesta: priori N(100, 20²), σ = 15, n = 9 y ȳ = 88", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Precisión de la priori", "1/20² = 1/400", "0.0025"],
               ["Precisión de los datos", "9/15² = 9/225", "0.0400"],
               ["Precisión posterior", "0.0025 + 0.0400", f"{f(1 / vn, 4)}  →  sₙ² = {f(vn, 3)}, sₙ = {f(sqrt(vn), 3)}"],
               ["Peso de la priori", "0.0025/0.0425", f"w = {f(w, 4)}"],
               ["Media posterior", "w·100 + (1 − w)·88", f"{f(mn, 3)}"],
               ["Intervalo creíble 95 %", "mₙ ± 1.96·sₙ", ci(stats.norm.interval(0.95, mn, sqrt(vn)), 2)],
               ["P(μ &lt; 90 | y)", "Φ((90 − mₙ)/sₙ)", f"{f(stats.norm.cdf(90, mn, sqrt(vn)), 3)}"]],
              [46 * mm, 56 * mm, CONTENT_W - 102 * mm - 18]),
        P(f"Lectura: los datos pesan {f((1 - w) * 100, 1)} % y la priori {f(w * 100, 1)} %; la media posterior queda entre 100 y 88, muy cerca de 88. <b>Error común</b>: promediar m₀ y ȳ sin pesos, o usar σ² donde la fórmula pide σ/√n.", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_normal.png", width=CONTENT_W * 0.86), Spacer(1, 2), pista("la Parte A del notebook y el Bloque 1 del cuaderno", [P("Calcula mₙ y sₙ con la función <font name='Mono'>normal_normal</font>, verifícalos multiplicando priori × verosimilitud en una grilla y mira cómo cambian con n.", "cell")])]))

    # 2
    s.append(h("2", "Distribución predictiva posterior"))
    s += para("Para predecir una observación nueva se promedian las predicciones de todos los θ posibles, ponderadas por la posteriori. El resultado tiene la variabilidad del azar del dato nuevo <b>más</b> la incertidumbre sobre θ. En el caso Beta–Binomial es una distribución beta-binomial; en el Normal–Normal, N(mₙ, σ² + sₙ²).")
    a2, b2, m2 = 9, 6, 8
    bb = stats.betabinom(m2, a2, b2)
    pl = stats.binom(m2, a2 / (a2 + b2))
    s.append(ejemplo(f"posteriori Beta({a2}, {b2}) y {m2} pacientes nuevos", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Media de θ", f"{a2}/({a2} + {b2})", f"{f(a2 / (a2 + b2), 3)}"],
               ["Media predictiva", f"{m2}·{a2}/({a2} + {b2})", f"{f(bb.mean(), 3)} pacientes"],
               ["Varianza predictiva", f"{m2}·{a2}·{b2}·({a2 + b2 + m2})/(({a2 + b2})²·{a2 + b2 + 1})", f"{f(bb.var(), 3)}"],
               ["Varianza con θ fijo", f"{m2}·0.6·0.4", f"{f(pl.var(), 3)}"],
               ["P(al menos 6 responden)", "predictiva", f"{f(bb.sf(5), 3)}"],
               ["P(al menos 6)", f"plug-in Binomial({m2}, 0.6)", f"{f(pl.sf(5), 3)}"]],
              [46 * mm, 74 * mm, CONTENT_W - 120 * mm - 18]),
        P(f"Lectura: el centro casi no cambia, pero la varianza sube de {f(pl.var(), 2)} a {f(bb.var(), 2)} y las colas pesan más. <b>Error común</b>: predecir con la media de θ como si fuera conocida; subestima la incertidumbre.", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_predictiva.png", width=CONTENT_W * 0.86), Spacer(1, 2), pista("la Parte A del notebook (A3 y A4) y el Bloque 2 del cuaderno", [P("Simula θ y luego Ỹ, compara con la beta-binomial exacta y con el plug-in, y calcula la predictiva Normal.", "cell")])]))

    # 3
    s.append(h("3", "Simulación de Monte Carlo"))
    s += para("Si se pueden generar M muestras independientes de la posteriori, cualquier esperanza E[g(θ) | y] se aproxima con el promedio de g(θ⁽ᵐ⁾). Una probabilidad es el caso g = indicador. El error estándar de una proporción simulada es √(p̂(1 − p̂)/M).")
    M3, k3 = 4000, 2920
    p3 = k3 / M3
    se3 = sqrt(p3 * (1 - p3) / M3)
    s.append(ejemplo("de 4 000 simulaciones de la posteriori, 2 920 superan el umbral 0.5", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Estimación", "2 920/4 000", f"p̂ = {f(p3, 3)}"],
               ["Error estándar", "√(0.73·0.27/4 000)", f"{f(se3, 4)}"],
               ["Intervalo de simulación 95 %", "p̂ ± 1.96·SE", ci([p3 - 1.96 * se3, p3 + 1.96 * se3], 3)],
               ["M para SE ≤ 0.002", "0.73·0.27/0.002²", f"M ≥ {mil(p3 * (1 - p3) / 0.002 ** 2)}"],
               ["Si hay autocorrelación", "ESS = 800 en vez de 4 000", f"SE = {f(sqrt(p3 * (1 - p3) / 800), 4)}"]],
              [48 * mm, 56 * mm, CONTENT_W - 104 * mm - 18]),
        P("Lectura: dividir el error por 2 exige 4 veces más simulaciones. Con una cadena, el tamaño que cuenta es el efectivo. <b>Error común</b>: confundir el error de simulación (cálculo) con la incertidumbre sobre θ (la posteriori).", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_montecarlo.png", width=CONTENT_W * 0.86), Spacer(1, 2), pista("la Parte B del notebook y el Bloque 3 del cuaderno", [P("Reproduce 0.7627 con la cola exacta y por simulación y mide cómo baja el error con M; añade la comparación A/B por simulación.", "cell")])]))

    # 4
    s.append(h("4", "Metropolis a mano"))
    s += para("Basta el núcleo de la posteriori (sin constante). Para un paso: proponer θ*, calcular r = núcleo(θ*)/núcleo(θ), α = min(1, r); si u &lt; α se acepta, si no la cadena repite el valor actual. Una propuesta fuera del soporte tiene núcleo 0 y se rechaza.")
    ker = lambda t: t ** 8 * (1 - t) ** 3 if 0 < t < 1 else 0.0
    th0, th1, u1 = 0.60, 0.72, 0.30
    th2 = 0.45
    r1 = ker(th1) / ker(th0)
    r2 = ker(th2) / ker(th1) if r1 else 0
    s.append(ejemplo("núcleo θ⁸(1 − θ)³ (una Beta(9, 4)); actual 0.60", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Propuesta θ* = 0.72", "0.72⁸·0.28³ / (0.60⁸·0.40³)", f"r = {f(r1, 3)}"],
               ["Probabilidad de aceptar", "min(1, r)", f"α = {f(min(1, r1), 3)}"],
               ["Decisión con u = 0.30", f"u {'&lt;' if u1 < min(1, r1) else '≥'} α", "acepta: θ pasa a 0.72" if u1 < min(1, r1) else "rechaza"],
               ["Siguiente propuesta θ* = 0.45", "0.45⁸·0.55³ / (0.72⁸·0.28³)", f"r = {f(r2, 3)}"],
               ["Con u = 0.40", f"u {'&lt;' if 0.4 < min(1, r2) else '≥'} {f(min(1, r2), 3)}", "acepta" if 0.4 < min(1, r2) else "rechaza: la cadena se queda en 0.72"],
               ["Propuesta θ* = 1.10", "fuera de (0, 1): núcleo 0", "r = 0 → rechaza"]],
              [46 * mm, 62 * mm, CONTENT_W - 108 * mm - 18]),
        P("Lectura: se sube siempre que la propuesta es más plausible; se baja con probabilidad r. Rechazar no es un error: la repetición del valor es parte de la cadena. <b>Error común</b>: omitir los rechazos al guardar la cadena.", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_metropolis.png", width=CONTENT_W * 0.78), Spacer(1, 2), pista("la Parte C del notebook (C2 y C3) y el Bloque 4 del cuaderno", [P("Programa tu propio paso, repítelo 5 000 veces y compara el histograma con la Beta exacta; luego prueba tres tamaños de paso.", "cell")])]))

    # 5
    s.append(h("5", "Leer una cadena: traza, burn-in, ESS y R-hat"))
    s += para("Un buen ajuste cumple a la vez: trazas estables que se superponen entre cadenas, R-hat ≈ 1 (&lt; 1.01), ESS suficiente (cientos como mínimo) y una aceptación razonable. El burn-in se descarta; el thinning no mejora la información.")
    s.append(ejemplo("tres ajustes del mismo coeficiente (4 cadenas × 2 000 iteraciones)", [
        pasos([["Ajuste", "R-hat · ESS · aceptación · traza", "Decisión"],
               ["A", "1.003 · 1 450 · 33 % · oruga estable, cadenas superpuestas", "Usar; ESS ≈ 18 % del total, aceptable"],
               ["B", "1.12 · 28 · 95 % · tramos suaves y cadenas separadas", "No usar: paso pequeño y sin convergencia; aumentar el paso y correr más"],
               ["C", "1.002 · 90 · 4 % · tramos planos y saltos", "Cuidado: R-hat bueno pero ESS bajo y casi todo rechazado; reducir el paso"]],
              [18 * mm, 86 * mm, CONTENT_W - 104 * mm - 18]),
        P("Lectura: ningún criterio basta solo. R-hat compara cadenas; el ESS mide información; la aceptación orienta el tamaño de paso. <b>Error común</b>: fiarse de R-hat cuando el ESS es minúsculo.", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_rhat.png", width=CONTENT_W * 0.86), Spacer(1, 2), pista("la Parte C del notebook (C4 a C7) y los Bloques 5 y 6 del cuaderno", [P("Calcula ESS y R-hat con la función entregada, cambia el tamaño de paso y observa cómo cambian las cuatro cifras.", "cell")])]))

    # 6
    s.append(h("6", "Resumir la posteriori con muestras: regresión logística bayesiana"))
    s += para("Con las muestras de un coeficiente β: la <b>mediana</b> es el estimador, los cuantiles 2.5 % y 97.5 % dan el intervalo creíble, la fracción de muestras &gt; 0 es P(β &gt; 0 | y) y el <b>odds ratio</b> sale de exp(β). Con predictores estandarizados, el OR es por una desviación estándar.")
    s.append(ejemplo("coeficiente β con mediana 0.42 e intervalo creíble 95 % [0.10; 0.75]; el 99.2 % de las muestras es positivo", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["Odds ratio", "exp(0.42)", f"{f(np.exp(0.42), 3)}"],
               ["IC creíble del OR", "exp(0.10) y exp(0.75)", ci([np.exp(0.10), np.exp(0.75)])],
               ["P(β &gt; 0 | y)", "fracción de muestras &gt; 0", "0.992"],
               ["Lectura del OR", "exp(0.42) = 1.52", "+1 sd en el predictor multiplica los odds por 1.52"],
               ["Interpretación del IC", "95 % de probabilidad", "dados el modelo, la priori y los datos"]],
              [42 * mm, 56 * mm, CONTENT_W - 98 * mm - 18]),
        P("Lectura: el intervalo no contiene 0 ni (para el OR) 1: hay asociación positiva. <b>Error común</b>: decir que el OR es el cambio en la <i>probabilidad</i>, o llamar «de confianza» a un intervalo creíble.", "cell")]))
    s.append(KeepTogether([Spacer(1, 2), img("c24_regresion.png", width=CONTENT_W * 0.86), Spacer(1, 2), pista("la Parte D del notebook y el Bloque 7 del cuaderno", [P("Reproduce los resultados de la profesora (medianas −0.921, 0.966, 0.548 y 0.573) y compara con glm y con el bootstrap.", "cell")])]))

    # 7 puentes
    s.append(h("7", "Puentes: los mismos datos, dos lecturas"))
    s += para("Con 200 observaciones y prioris suaves, la posteriori y la máxima verosimilitud casi coinciden; cambia <b>qué significa</b> el intervalo y qué se puede preguntar. Los números son los de este material con los datos Pima (4 cadenas de 25 000 iteraciones, descartando las 5 000 primeras).")
    filas = [["Coeficiente", "glm clásico: estimación [IC 95 % de Wald]", "bayesiano: mediana [IC creíble 95 %]", "valor p · P(β &gt; 0 | y)"]]
    for nm in ("glu_z", "bmi_z", "age_z"):
        r = R_[nm]
        filas.append([nm, f"{f(r['mle'], 3)} [{f(r['mle_lo'], 3)}; {f(r['mle_hi'], 3)}]", f"{f(r['larga_mediana'], 3)} [{f(r['larga_q025'], 3)}; {f(r['larga_q975'], 3)}]", f"{r['mle_p']:.4f} · {f(r['p_pos'], 3)}"])
    s.append(ejemplo("regresión logística de diabetes con Pima.tr", [
        pasos(filas, [24 * mm, 58 * mm, 58 * mm, CONTENT_W - 140 * mm - 18]),
        P(f"Lectura: el valor p responde «¿qué tan raro sería este dato si β fuera 0?»; P(β &gt; 0 | y) responde «¿qué tan probable es que β sea positivo?». El bootstrap del glm (B = {PP['boot']['B']}) da para glu_z una desviación de {f(PP['boot']['sd'], 3)} y la posteriori {f(PP['post']['sd'], 3)}: dos formas de medir la incertidumbre que aquí coinciden. <b>Error común</b>: pensar que una priori «débil» deja de influir en muestras pequeñas; con n = 200 casi no se nota.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(KeepTogether([img("c24_predicciones.png", width=CONTENT_W * 0.86), Spacer(1, 2),
                           pista("la Parte D del notebook (D8)", [P("Compara las posterioris con el glm y con el bootstrap y repite con una priori más fuerte, N(0, 0.25²), para ver cuándo la priori sí cambia el resultado.", "cell")])]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Normal–Normal: sumar precisiones; media = promedio ponderado por precisiones.",
                       "Predictiva: Var = (variabilidad del dato) + (incertidumbre de θ).",
                       "Monte Carlo: integral → promedio; error <font name='Sym'>∝</font> 1/√M (o 1/√ESS en una cadena).",
                       "Metropolis: r = núcleo(θ*)/núcleo(θ); α = min(1, r); u &lt; α → acepta; si no, repite.",
                       "Cadena: traza, burn-in, autocorrelación, ESS, varias cadenas y R-hat &lt; 1.01.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 4]))
            i += 4
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C24_Apuntes_Bayesiana2.pdf", out, "Apuntes · Clase 24 · Estadística bayesiana, parte 2")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#047857">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C24))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 24 · Bayesiana, parte 2", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C24))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: 1/sₙ² = 1/s₀² + n/σ² · mₙ = sₙ²(m₀/s₀² + nȳ/σ²) · Var(Ỹ) = σ² + sₙ² · SE_MC = √(p̂(1 − p̂)/M) · r = núcleo(θ*)/núcleo(θ), α = min(1, r) · SE = sd/√ESS · OR = exp(β).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C24)], bg=SOFT, border=C24, pad=5))
    s.append(Spacer(1, 2))

    # datos
    m0, s0, sg, n1, yb = 50, 10, 8, 16, 56
    mn1, vn1 = normal_normal(m0, s0, sg, n1, yb)
    mn4, vn4 = normal_normal(m0, s0, sg, 64, yb)
    a2, b2, m2 = 12, 8, 5
    bb = stats.betabinom(m2, a2, b2)
    M3, k3 = 5000, 3850
    g6 = [0.20, 0.50, 0.40, 0.90, 0.60, 0.30]
    kern = lambda t: t ** 5 * (1 - t) ** 2 if 0 < t < 1 else 0.0
    rng = np.random.default_rng(2403)
    dr = np.sort(np.round(rng.normal(0.45, 0.4, 40), 2))
    s += bloque(1, "Normal–Normal a mano", "★", f"Puntajes N(μ, {sg}²) con σ = {sg} conocida. Priori μ ~ N({m0}, {s0}²). Se observan n = {n1} estudiantes con ȳ = {yb}.", [
        "Calcula las precisiones de la priori y de los datos, y los pesos de cada una.",
        "Calcula sₙ², sₙ y mₙ, y el intervalo creíble de 95 %.",
        "Calcula P(μ &gt; 55 | y). Repite mₙ y sₙ con n = 64 y el mismo ȳ: ¿qué cambia?"])
    s += bloque(2, "Distribución predictiva", "★★", f"(a) Con los resultados del Bloque 1 llega un estudiante nuevo. (b) La posteriori de una tasa es Beta({a2}, {b2}) y se observarán m = {m2} casos nuevos.", [
        "(a) Calcula la media y la desviación de la predictiva de un nuevo puntaje y compárala con σ.",
        "(a) Calcula P(nuevo &gt; 60) con la predictiva y con N(mₙ, σ²).",
        f"(b) Calcula la media y la varianza de la predictiva (beta-binomial) y compárala con la de una Binomial({m2}, {a2}/{a2 + b2}).",
        f"(b) Calcula P(Ỹ = {m2}) con la predictiva y con el plug-in. ¿Cuál es mayor y por qué?"])
    s += bloque(3, "Monte Carlo", "★★", f"De M = {M3} simulaciones de una posteriori, {k3} superan 0.5. Además, 6 simulaciones de θ son {', '.join(f'{x:.2f}' for x in g6)}.", [
        "Estima P(θ &gt; 0.5 | y), su error estándar y un intervalo de simulación del 95 %.",
        "¿Cuántas simulaciones harían falta para un error estándar de 0.002?",
        "Con las 6 simulaciones estima E[θ | y] y E[θ² | y], y deduce la varianza posterior estimada."])
    s += bloque(4, "Un paso de Metropolis", "★★★", "Núcleo de la posteriori θ⁵(1 − θ)² (una Beta(6, 3)). La cadena está en θ = 0.50 y la propuesta es θ* = 0.65.", [
        "Calcula r y α. Con u = 0.80, ¿se acepta?",
        "Desde θ = 0.65 se propone θ* = 0.40. Calcula r y α y decide con u = 0.20 y con u = 0.60.",
        "Desde θ = 0.65 se propone θ* = 1.05. ¿Qué pasa y qué se guarda en la cadena?"])
    s += bloque(5, "Leer diagnósticos", "★★", "Tres ajustes con 4 cadenas × 1 000 iteraciones válidas. P: R-hat 1.001, ESS 2 600, aceptación 35 %. Q: R-hat 1.30, ESS 15, aceptación 90 %. R: R-hat 1.004, ESS 40, aceptación 3 %.", [
        "¿Cuál se puede usar sin más cambios? Justifica con las tres cifras.",
        "Para Q y para R, diagnostica la causa más probable y la acción (paso, iteraciones, reparametrizar).",
        "Para R: si la desviación posterior es 0.5, ¿cuál es el error de Monte Carlo de la media? ¿Y si el ESS fuera 2 000?"])
    s += bloque(6, "Burn-in y thinning", "★★", "Se corren 4 cadenas de 6 000 iteraciones, con 1 000 de burn-in. El ESS total resulta 1 200.", [
        "¿Cuántos valores quedan sin thinning? ¿Y con thinning = 5?",
        "¿Cuál es el ESS aproximado con thinning = 5 y por qué no puede superar al original?",
        "Si el sd posterior es 0.4, calcula el error de Monte Carlo de la media con ESS = 1 200 y con 24 000 muestras independientes."])
    s += bloque(7, "Resumir con muestras", "★★★", f"Se tienen 40 simulaciones ordenadas de un coeficiente β: {', '.join(f'{x:.2f}' for x in dr)}.", [
        "Construye el intervalo creíble del 90 % descartando 2 valores en cada cola y la mediana.",
        "Calcula P(β &gt; 0 | y) como fracción de muestras positivas.",
        "Calcula el odds ratio mediano y su intervalo (con exp) e interprétalo; ¿qué se haría con un predictor estandarizado?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C24))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C24), Spacer(1, 2)]

    w0 = (1 / s0 ** 2) / (1 / vn1)
    lo1, hi1 = stats.norm.interval(0.95, mn1, sqrt(vn1))
    s += sol("Bloque 1", [
        ["1", f"Priori: 1/{s0}² = {f(1 / s0 ** 2, 4)}; datos: {n1}/{sg}² = {f(n1 / sg ** 2, 4)}. Pesos: {f(w0, 3)} para la priori y {f(1 - w0, 3)} para los datos."],
        ["2", f"sₙ² = 1/({f(1 / s0 ** 2, 4)} + {f(n1 / sg ** 2, 4)}) = {f(vn1, 3)}; sₙ = {f(sqrt(vn1), 3)}; mₙ = {f(w0, 3)}·{m0} + {f(1 - w0, 3)}·{yb} = {f(mn1, 3)}. IC 95 %: {ci([lo1, hi1], 2)}."],
        ["3", f"P(μ &gt; 55) = 1 − Φ(({55} − {f(mn1, 3)})/{f(sqrt(vn1), 3)}) = {f(stats.norm.sf(55, mn1, sqrt(vn1)), 3)}. Con n = 64: mₙ = {f(mn4, 3)} y sₙ = {f(sqrt(vn4), 3)}: la media se acerca a ȳ = {yb} y la desviación baja a casi la mitad."]])
    sd_p = sqrt(sg ** 2 + vn1)
    pb = stats.betabinom(m2, a2, b2)
    pg = stats.binom(m2, a2 / (a2 + b2))
    s += sol("Bloque 2", [
        ["1", f"Media {f(mn1, 3)}; varianza σ² + sₙ² = {sg ** 2} + {f(vn1, 3)} = {f(sg ** 2 + vn1, 3)}; desviación {f(sd_p, 3)}, algo mayor que σ = {sg}."],
        ["2", f"Predictiva: 1 − Φ((60 − {f(mn1, 3)})/{f(sd_p, 3)}) = {f(stats.norm.sf(60, mn1, sd_p), 4)}; con N(mₙ, σ²): {f(stats.norm.sf(60, mn1, sg), 4)}. Cambia poco porque sₙ² es pequeña frente a σ²."],
        ["3", f"Media = {m2}·{a2}/{a2 + b2} = {f(pb.mean(), 3)}; varianza = {m2}·{a2}·{b2}·({a2 + b2 + m2})/(({a2 + b2})²·{a2 + b2 + 1}) = {f(pb.var(), 3)}. Binomial: {m2}·0.6·0.4 = {f(pg.var(), 3)}."],
        ["4", f"P(Ỹ = {m2}) predictiva = {f(pb.pmf(m2), 4)}; plug-in = 0.6⁵ = {f(pg.pmf(m2), 4)}. La predictiva es una mezcla de Binomiales: los θ más altos aportan mucho al extremo «todos» y los más bajos aportan poco, y como P(Ỹ = 5) crece muy rápido con θ gana la mezcla. En general la predictiva tiene colas más pesadas."]])
    p3 = k3 / M3
    se3 = sqrt(p3 * (1 - p3) / M3)
    e1, e2 = np.mean(g6), np.mean(np.square(g6))
    s += sol("Bloque 3", [
        ["1", f"p̂ = {k3}/{M3} = {f(p3, 3)}; SE = √({f(p3, 2)}·{f(1 - p3, 2)}/{M3}) = {f(se3, 4)}; intervalo {ci([p3 - 1.96 * se3, p3 + 1.96 * se3], 3)}."],
        ["2", f"M = {f(p3, 2)}·{f(1 - p3, 2)}/0.002² = {mil(p3 * (1 - p3) / 0.002 ** 2)}."],
        ["3", f"E[θ] ≈ {f(e1, 4)}; E[θ²] ≈ {f(e2, 4)}; varianza ≈ {f(e2, 4)} − {f(e1, 4)}² = {f(e2 - e1 ** 2, 4)} (con solo 6 simulaciones es muy imprecisa)."]])
    r1_, r2_, r3_ = kern(0.65) / kern(0.50), kern(0.40) / kern(0.65), 0
    s += sol("Bloque 4", [
        ["1", f"r = 0.65⁵·0.35² / (0.50⁵·0.50²) = {f(r1_, 3)} &gt; 1, α = 1; como u = 0.80 &lt; 1, <b>acepta</b> (la cadena pasa a 0.65)."],
        ["2", f"r = 0.40⁵·0.60² / (0.65⁵·0.35²) = {f(r2_, 3)}, α = {f(min(1, r2_), 3)}. u = 0.20 {'&lt;' if 0.2 < r2_ else '≥'} α → {'acepta' if 0.2 < r2_ else 'rechaza'}; u = 0.60 {'&lt;' if 0.6 < r2_ else '≥'} α → {'acepta' if 0.6 < r2_ else 'rechaza'}."],
        ["3", "θ* = 1.05 está fuera de (0, 1): el núcleo vale 0, r = 0, α = 0 y siempre se rechaza. La cadena <b>repite</b> 0.65 (se guarda otra vez 0.65)."]])
    s += sol("Bloque 5", [
        ["1", "Solo P: R-hat ≈ 1, ESS = 2 600 (65 % de 4 000) y aceptación 35 % (zona sana)."],
        ["2", "Q: R-hat 1.30 (sin convergencia), ESS 15 y aceptación 90 % → paso demasiado pequeño y pocas iteraciones: aumentar el paso y correr mucho más. R: R-hat bueno pero ESS 40 y aceptación 3 % → paso demasiado grande, la cadena se pega: reducir el paso (o reparametrizar)."],
        ["3", f"SE = 0.5/√40 = {f(0.5 / sqrt(40), 3)}; con ESS = 2 000: 0.5/√2 000 = {f(0.5 / sqrt(2000), 3)}. Menos de 3 decimales fiables en el primer caso."]])
    s += sol("Bloque 6", [
        ["1", "Cada cadena guarda 5 000 válidas: 4 × 5 000 = 20 000. Con thinning = 5: 4 000."],
        ["2", "Quedan 4 000 valores, pero el ESS no sube: es ≤ 1 200 (algo menos, porque se tira información). Adelgazar ahorra memoria, no mejora la información."],
        ["3", f"SE con ESS = 1 200: 0.4/√1 200 = {f(0.4 / sqrt(1200), 4)}; con 24 000 independientes: 0.4/√24 000 = {f(0.4 / sqrt(24000), 4)} (casi 4.5 veces menor)."]])
    lo7, hi7 = dr[2], dr[-3]
    med7 = np.median(dr)
    pos = (dr > 0).mean()
    s += sol("Bloque 7", [
        ["1", f"Se descartan 2 menores ({dr[0]:.2f}, {dr[1]:.2f}) y 2 mayores ({dr[-2]:.2f}, {dr[-1]:.2f}): IC 90 % = [{lo7:.2f}; {hi7:.2f}]; mediana = {f(med7, 3)} (promedio de los dos centrales)."],
        ["2", f"{int((dr > 0).sum())} de 40 son positivas: P(β &gt; 0 | y) = {f(pos, 3)}."],
        ["3", f"OR mediano = exp({f(med7, 3)}) = {f(np.exp(med7), 3)}; intervalo [exp({lo7:.2f}); exp({hi7:.2f})] = [{f(np.exp(lo7), 3)}; {f(np.exp(hi7), 3)}]. Si el predictor está estandarizado, es el efecto sobre los odds de +1 desviación estándar; como el intervalo {'contiene' if np.exp(lo7) <= 1 <= np.exp(hi7) else 'no contiene'} 1, {'no hay asociación clara' if np.exp(lo7) <= 1 <= np.exp(hi7) else 'hay asociación'}."]])
    build("apuntes/C24_Ejercicios_Cuaderno_Bayesiana2.pdf", s, "Ejercicios para el cuaderno · Clase 24")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
