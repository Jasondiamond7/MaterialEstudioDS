"""Clases 25-27 · Modelos predictivos de series de tiempo (Prof. Cristian Vásquez): guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c25_27_numeros.json (lo escribe figuras_c25_27.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer
from scipy import stats

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image

from construir_pdf import IMG
from figuras_c25_27 import acf_manual, descomponer_aditiva

N = json.loads((Path(__file__).resolve().parent / "c25_27_numeros.json").read_text(encoding="utf-8"))
C25 = colors.HexColor("#4D7C0F")
C25_BG = colors.HexColor("#F7FEE7")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: 3 transcripciones + presentaciones «Clase 1, 2 y 4» + cuadernos en R (Clases 1 y 2) + cuaderno de cierre (Clase 28) · Prof. Cristian Vásquez"
EJ, RU, DE, RG, BC = N["ejemplos"], N["ruido"], N["descomp"], N["regresion"], N["boxcox"]
AR_, MA_, AB, MO, DG, PR = N["ar"], N["ma"], N["arbol"], N["modelos"], N["diag"], N["pron"]
SA, US, TE, AD, RT, VA, CB, PU = N["sar"], N["usacc"], N["tend"], N["adf"], N["retornos"], N["valid"], N["citybike"], N["puentes"]


def imgc(name, width, maxh):
    """Imagen con ancho dado pero con altura máxima (reduce el ancho si la figura es alta)."""
    iw, ih = ImageReader(str(IMG / name)).getSize()
    w = min(width, maxh * iw / ih)
    return Image(str(IMG / name), width=w, height=w * ih / iw)


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ")


def ci(v, d=3):
    return f"[{v[0]:.{d}f}; {v[1]:.{d}f}]"


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.9):
    s.append(band(f"Clases 25-27 · {titulo}", FUENTE, C25))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C25, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C25))
    s.append(Spacer(1, 1))
    for fig, pie, *w in figuras:
        s.append(imgc(fig, CONTENT_W * (w[0] if w else fw), (82 if len(figuras) == 1 else 50) * mm))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida[:3] + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C25)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clases 25-27 · Modelos predictivos de series de tiempo: de la descomposición a ARIMA y SARIMA",
                  "Prof. Cristian Vásquez (PUC) · puentes con regresión (C17–21), validación (C30, C38) y bootstrap (C43)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de las tres clases", "cellb"), Spacer(1, 3),
                  P("En una serie de tiempo los datos <b>no son independientes</b>: lo de hoy se parece a lo de ayer. El trabajo consiste en separar lo que se puede explicar (tendencia, estacionalidad) de lo que queda, y en <b>modelar esa dependencia</b> con modelos "
                    "<b>AR, MA y ARMA</b> (Box-Jenkins) para series <b>estacionarias</b>. Se identifican con la <b>ACF y la PACF</b>, se eligen por <b>AIC</b> y parsimonia y se aceptan solo si los <b>residuos son ruido blanco</b> (Ljung-Box). "
                    "Si la serie tiene tendencia estocástica se <b>diferencia</b> (ARIMA) tras un test de <b>Dickey-Fuller</b>; si tiene ciclos regulares se agrega una <b>parte estacional</b> (SARMA/SARIMA). "
                    "Un pronóstico se entrega con su <b>intervalo de predicción</b> y se valida <b>respetando el orden del tiempo</b>.", "body")],
                 bg=C25_BG, border=C25, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("Las tres clases en una imagen", "h2"))
    s.append(img("c25_27_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Qué material es de cuál clase", "h3"),
            table([["Clase", "Fecha / tema", "Material"],
                   ["C25", "Introducción: ruido blanco, ACF, descomposición, regresión con dummies, armónicas, Box-Cox", "Presentación «Clase 1» (portada dice «Clase 25») + «Código Clase 1» (R)"],
                   ["C26", "Estacionarias: AR, MA, ARMA, ACF/PACF, estimación, AIC, Ljung-Box, pronóstico", "Presentación «Clase 2» (portada repite «Clase 25») + «Código Clase 2» (R)"],
                   ["—", "Sesión de repaso del sábado (caminatas aleatorias): sin transcripción ni archivos", "no incluida en la carpeta (no existe «Clase 3»)"],
                   ["C27", "SARMA, ARIMA, raíz unitaria (Dickey-Fuller), retornos", "Presentación «Clase 4» (portada dice «Clase 27»); su cuaderno en R no se entregó"],
                   ["C28", "Cierre del módulo: CityBike con ARIMA, SARIMA y SARIMAX (Python)", "<font name='Mono'>DS26_28_Cierre_de_Módulo.ipynb</font> (sin transcripción)"]],
                  [14 * mm, 40 * mm, HALF - 54 * mm - 12], header_color=C25)]
    right = [P("Lo que necesitas ANTES y los datos", "h3"),
             bullets(["<b>Regresión lineal y dummies</b> (C17–19): la descomposición de la Clase 25 es una regresión; aquí el error <b>no</b> es independiente",
                      "<b>Probabilidad básica</b>: varianza, covarianza, correlación, normal y chi-cuadrado",
                      "<b>AIC</b> y comparación de modelos (C20–21), <b>validación</b> (C30, C38) y <b>bootstrap</b> (C43)",
                      f"<b>Datos</b>: pasajeros EE. UU. ({DE['n']} meses desde 2010), anillos de un árbol de Idaho ({AB['n']} años), IPC, IPSA, PM2.5, desocupación, accidentes EE. UU. (72 meses), CityBike ({CB['n']} días) e IAC (120 meses). Copias en <font name='Mono'>Guias/datos/series_tiempo/</font>"], "cell", C25)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con otras clases (sin repetirlas)</b>", "cellb"),
                  P(f"<b>C17–21 → C25</b>: el ajuste con tendencia y dummies mensuales es una regresión lineal (R² ajustado {f(RG['r2aj'], 3)}), pero sus residuos tienen Durbin-Watson {f(RG['dw'], 2)} y ACF(1) = {f(RG['acf1'], 2)}: el error estándar de la tendencia pasa de {mil(PU['bse_ols'])} (OLS) a {mil(PU['bse_hac'])} (robusto), ×{f(PU['razon'], 2)}. "
                    f"<b>C30 y C38 → C26–27</b>: en series la validación <b>es temporal</b> (corte o origen móvil), nunca al azar; en los pasajeros el SARIMA tiene MAE {f(VA['mae']['SARIMA(0,1,1)(0,1,1)₁₂'], 2)} frente a {f(VA['mae']['ingenuo estacional'], 2)} del ingenuo estacional. "
                    f"<b>C43 → C26</b>: el intervalo de predicción del modelo para h = 10 mide {f(PU['ancho_modelo_h10'], 3)} de ancho y el de un bootstrap de residuos, {f(PU['ancho_boot_h10'], 3)}: dos caminos para la misma incertidumbre.", "cell")], bg=SOFT, border=C25, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Qué es una serie de tiempo, ruido blanco y ACF (C25)",
        "Una serie de tiempo es una secuencia de variables aleatorias {Yₜ} indexada por tiempos <b>equidistantes</b>. Al perder la independencia aparece la <b>autocorrelación</b>. El <b>ruido blanco</b> (media 0, varianza constante, covarianza 0) es el modelo de «lo que no se puede predecir».",
        [("Objetivos", "Describir, pronosticar, retro-predecir e imputar datos faltantes. Aplica a IPC, IPSA, contaminación o desempleo; si los tiempos no son equidistantes hacen falta otras técnicas."),
         ("Ruido blanco ≠ independencia", "Ruido blanco pide E(εₜ) = 0, Var(εₜ) = σ² y Cov(εₜ, εₛ) = 0. Independencia implica covarianza 0, no al revés: ruido blanco es más débil. Tendencia o varianza creciente lo descartan a simple vista."),
         ("ACF y bandas", "ρ(k) = Cov(Yₜ, Yₜ₊ₖ)/Var(Yₜ) está entre −1 y 1 (no depende de la escala). La estimada usa una media global; las bandas ±1.96/√n marcan qué es significativo a ≈ 5 %. En k = 0 vale 1 siempre.")],
        [("c25_27_ruido_blanco.png", f"Tres series simuladas de n = {RU['n']}: la que tiene tendencia (ACF(1) = {f(RU['acf1_tend'], 2)}) y la de varianza creciente no son ruido blanco; la tercera sí: {RU['fuera_c']} de {RU['de_c']} autocorrelaciones fuera de las bandas (±{f(RU['banda'], 3)}).", 0.86)],
        ["Yₜ ~ ruido blanco: E = 0, Var = σ², Cov(εₜ, εₛ) = 0", "r(k) = Σₜ (Yₜ − Ȳ)(Yₜ₊ₖ − Ȳ) / Σₜ (Yₜ − Ȳ)²", "bandas: ±1.96/√n"],
        ["acf_manual(y, 20)               # igual a acf() de R", "from statsmodels.graphics.tsaplots import plot_acf"],
        [f"Con n = {RU['n']} la banda es ±{f(RU['banda'], 3)}; con 20 rezagos se esperan ≈ 1 fuera por azar (5 %)",
         "Estacionaria: media constante, varianza constante y Cov(Yₜ, Yₜ₊ₖ) = γ(k) solo depende de k",
         "«Covarianza 0 no implica independencia; independencia sí implica covarianza 0»",
         "Se necesitan al menos 50–60 datos para que la ACF sea confiable"],
        [f"El código en R del IPC lee <font name='Mono'>ipc_var.txt</font> (tabla 12 × 22: meses × años) <b>por filas</b>, mezclando años y meses; leída por columnas la ACF(1) es {f(EJ['ipc_acf1_ok'], 2)} y no {f(EJ['ipc_acf1_codigo_R'], 2)}."], fw=0.86)

    topic_page(
        s, "Descomposición, regresión con dummies, armónicas y Box-Cox (C25)",
        "La primera aproximación es aditiva: <b>Yₜ = Tₜ + Sₜ + Cₜ + εₜ</b>. Tendencia y estacionalidad se estiman con funciones deterministas (polinomio, dummies); si lo que queda es ruido blanco, no hay más que hacer. Casi nunca lo es.",
        [("Descomposición clásica", "decompose() usa una media móvil centrada para la tendencia (se pierden s/2 datos en cada extremo), promedia por mes para la estacionalidad y deja el ruido = observada − tendencia − estacional."),
         ("Regresión lineal con dummies", "Yₜ = β₀ + β₁t + Σⱼ αⱼ·dⱼₜ + εₜ con un mes como referencia (si hay 12 dummies y constante, el modelo no se identifica). Estacional («saltos» en meses puntuales) se captura con dummies; ciclos suaves con senos y cosenos."),
         ("Estabilizar la varianza", "La transformación de Box-Cox (Y^α − 1)/α, o log si α = 0, se usa cuando la dispersión crece con el nivel. No quita tendencia ni estacionalidad.")],
        [("c25_27_descomposicion.png", f"Pasajeros internacionales, {DE['ini'][:7]} a {DE['fin'][:7]} (n = {DE['n']}). El mes de mayor efecto es el {DE['mes_max']} (julio, {f(DE['efecto_max'] / 1e6, 2)} millones) y el menor {f(DE['efecto_min'] / 1e6, 2)} millones. Se pierden {DE['na_ini']} valores de tendencia (6 por lado).", 0.62)],
        ["Yₜ = Tₜ + Sₜ + Cₜ + εₜ  (aditivo);  Yₜ = Tₜ·Sₜ·εₜ  (multiplicativo)", "Sₜ = Σⱼ₌₂ᴿ αⱼ·dⱼₜ,  Tₜ = β₀ + β₁t + … + βₘtᵐ", "Cₜ = Σⱼ [αⱼ sen(ωⱼt) + βⱼ cos(ωⱼt)]", "Xₜ = (Yₜᵅ − 1)/α  si α ≠ 0;  log Yₜ  si α = 0"],
        ["descomponer_aditiva(y, 12)", "sm.OLS.from_formula('y ~ t + C(mes)', df).fit()", "stats.boxcox_normmax(y, method='mle')"],
        [f"{RG['n_signif']} de {RG['n_coef']} coeficientes son significativos; no lo son octubre (p = {f(RG['p_oct'], 2)}) ni septiembre (p = {f(RG['p_sep'], 2)})",
         f"Con el corte temporal (entrenar sin los últimos 12 meses) el MAE de ese mismo modelo es {mil(RG['mae_ultimos12'])} pasajeros ({f(RG['mape_ultimos12'], 1)} %)",
         "Ciclo ≠ estacionalidad: la estacionalidad tiene un periodo fijo (12 meses) y se capta con dummies; el ciclo es sinusoidal y se capta con armónicas",
         "Un componente cíclico suele fundirse con el estacional (la clase lo separó para explicar la diferencia)"],
        ["La regresión de la diapositiva 31 se llama «regresión lineal simple» pero tiene 12 regresores: es <b>múltiple</b>. Y un R² de 0.978 no demuestra que el modelo sea adecuado: los residuos siguen correlacionados."], fw=0.86)

    topic_page(
        s, "Modelos AR, MA y ARMA: identificación con ACF y PACF (C26)",
        "Los modelos de Box y Jenkins aproximan cualquier fenómeno estacionario de memoria corta. <b>AR(p)</b>: Yₜ depende de sus p rezagos; <b>MA(q)</b>: Yₜ es una suma ponderada de q choques; <b>ARMA(p, q)</b> combina ambos y necesita menos parámetros.",
        [("Qué dibuja cada modelo", "AR: ACF decae (ρ(k) = φᵏ en AR(1)) y PACF se corta en p. MA: ACF se corta en q y PACF decae. ARMA: las dos decaen. La PACF mide la correlación entre Yₜ y Yₜ₊ₖ quitando el efecto de los rezagos intermedios."),
         ("Estacionariedad e invertibilidad", "AR(1): |φ| < 1; AR(2): φ₂ + φ₁ < 1, φ₂ − φ₁ < 1 y |φ₂| < 1. Un MA siempre es estacionario; se exige |θ| < 1 (invertibilidad) para poder escribirlo como AR y pronosticar."),
         ("Signos", "Las diapositivas escriben Yₜ = εₜ − θεₜ₋₁. R (arima.sim, arima) y statsmodels usan + θεₜ₋₁: un «ma = +0.9» de R es θ = −0.9 en las diapositivas. Hay que fijarse en la convención antes de comparar números.")],
        [("c25_27_ma_arma.png", f"MA(1) con θ = −0.9: ρ(1) = −θ/(1 + θ²) = {f(N['ma_teo']['rho1_theta09'], 3)}, y después 0; la PACF oscila y decae. En el ARMA(1,1) ambas decaen y el orden no se lee a simple vista.", 0.86)],
        ["AR(p): Yₜ = μ + φ₁Yₜ₋₁ + … + φₚYₜ₋ₚ + εₜ", "MA(q): Yₜ = μ + εₜ − θ₁εₜ₋₁ − … − θ_qεₜ₋_q", "AR(1): ρ(k) = φᵏ;  AR(2): ρ(1) = φ₁/(1 − φ₂)", "MA(1): ρ(1) = −θ/(1 + θ²),  ρ(k) = 0 si k ≥ 2"],
        ["from statsmodels.tsa.arima_process import ArmaProcess", "simular_arma([0.8], [], 500, rng)       # AR(1)", "pacf_yw(y, 25)"],
        ["Con φ = 0.8: ρ = 0.80, 0.64, 0.51, 0.41, 0.33 (decae geométricamente)",
         f"MA(2) con θ = (−0.7, −0.2): ACF teórica {f(MA_[1]['acf'][0], 2)} y {f(MA_[1]['acf'][1], 2)}, luego 0",
         "Los MA siempre son estacionarios; los AR no: φ = 1.5 deja de serlo",
         "Para ARMA se evalúan varias configuraciones y se elige con AIC y parsimonia (un AR(3) tiene 3 parámetros; un ARMA(1,1), 2)"],
        ["Un ARMA no se identifica «mirando»: ACF y PACF solo <b>sugieren</b> candidatos. Con series reales los patrones son difusos (ver el árbol de la página siguiente)."], fw=0.86)

    topic_page(
        s, "Estimar, elegir por AIC y diagnosticar (C26)",
        "Con la serie de anillos de árbol se ajustan cinco candidatos, se comparan por <b>AIC</b> (menor mejor, castigando parámetros) y se comprueba que los <b>residuos sean ruido blanco</b>. Si no lo son, el modelo no sirve aunque su AIC sea bajo.",
        [("Estimación", "Máxima verosimilitud con la normal multivariada (la matriz de covarianzas se construye con γ), mínimos cuadrados condicionales (tseries::arma), Yule-Walker (solo AR), Whittle o espacio de estados. Dan valores parecidos pero reportan cosas distintas."),
         ("AIC y significancia", "AIC = −2 log L + 2k. Solo se comparan AIC calculados con el mismo método y los mismos datos. Además se exige que cada parámetro sea significativo; un modelo con un coeficiente borderline (MA(3): p = 0.058) pierde frente a uno parsimonioso."),
         ("Diagnóstico de residuos", "Residuos estandarizados sin patrón, ACF dentro de las bandas, test de Ljung-Box (H₀: ρ₁ = … = ρₘ = 0) con valor p > 0.05 en varios m, y normalidad aproximada (histograma, QQ-plot) para fiarse de los intervalos.")],
        [("c25_27_arbol_modelos.png", f"Replicamos las diapositivas: con tseries::arma el menor AIC es el {MO['menor_aic_css']} ({f(MO['AR(2)']['css_aic'], 2)}); con máxima verosimilitud es el {MO['menor_aic_mle']} ({f(MO['ARMA(1,1)']['mle_aic'], 2)}); por BIC también el {MO['menor_bic_mle']}.", 0.86)],
        ["AIC = −2 log L + 2k;  BIC = −2 log L + k log n", "Q(m) = n(n + 2) Σᵣ r̂ᵣ²/(n − r)  ~  χ²(m − p − q)", "ARMA(1,1): Yₜ − μ = φ(Yₜ₋₁ − μ) + εₜ − θεₜ₋₁"],
        ["m = ARIMA(y, order=(1, 0, 1)).fit();  m.aic", "arma_css(y, 1, 1)         # réplica de tseries::arma", "ljung_box(m.resid, 10, ddof=2)"],
        [f"MA(3) por CSS: ma = {f(MO['MA(3)']['theta'][0], 3)}, {f(MO['MA(3)']['theta'][1], 3)}, {f(MO['MA(3)']['theta'][2], 3)}; AIC {f(MO['MA(3)']['css_aic'], 2)} y σ² = {f(MO['MA(3)']['css_s2'], 4)} (la diapositiva: 0.23819, 0.17289, 0.08113; 96.37)",
         f"ARMA(1,1) por CSS: φ = {f(MO['ARMA(1,1)']['phi'][0], 4)}, θ = {f(MO['ARMA(1,1)']['theta'][0], 4)}; por MLE: φ = {f(MO['ARMA(1,1)']['mle_params'][0], 4)}, ma = {f(MO['ARMA(1,1)']['mle_params'][1], 4)}",
         f"Tabla de la diapositiva 49 (arima): 102.77, 104.77, 105.98, 100.48, 100.11 — coincide con nuestros {', '.join(f(MO[k]['mle_aic'], 2) for k in ('MA(3)', 'MA(2)', 'AR(1)', 'AR(2)', 'ARMA(1,1)'))}",
         f"Normalidad: Shapiro-Wilk p = {f(DG['sw_p'], 3)} (colas algo pesadas); Kolmogorov-Smirnov sale p = {f(DG['ks_p'], 2)} pero con μ y σ estimados exige Lilliefors (p = {f(DG['lil_p'], 3)})"],
        ["Comparar 94.64 (tseries, AR(2)) con 100.11 (arima, ARMA(1,1)) no tiene sentido: son AIC de métodos distintos. La diapositiva 49 (arima) y las 44–48 (tseries) dan «ganadores» distintos."], fw=0.86)

    topic_page(
        s, "Pronóstico, intervalos de predicción y validación temporal (C26–27)",
        "El mejor predictor a h pasos es la <b>esperanza condicional</b> E[Yₙ₊ₕ | Y₁, …, Yₙ]. En un ARMA estacionario converge a la media cuando h crece (memoria corta). Debe entregarse con un <b>intervalo de predicción</b> y evaluarse con datos que el modelo <b>no vio</b>, respetando el tiempo.",
        [("Pronóstico y su intervalo", "Un AR(1) predice μ + φʰ(yₙ − μ) con varianza σ²(1 − φ²ʰ)/(1 − φ²): el intervalo se ensancha y se estabiliza en el de la serie. Aquí se llamó «intervalo de confianza» al de predicción: la diferencia importa (es sobre un valor futuro, no sobre un parámetro)."),
         ("Partición temporal", "Entrenar con el pasado y probar con el futuro: corte único (últimos h) u <b>origen móvil</b> (varios cortes). Separar al azar filtra información del futuro hacia el entrenamiento y entrega errores optimistas. Es el mismo principio de C30 y C38."),
         ("Comparar con referencias simples", "Un modelo útil debe ganarle al <b>ingenuo</b> (último valor) y al <b>ingenuo estacional</b> (valor de hace un año), y a la media si la serie es estacionaria.")],
        [("c25_27_pronostico.png", f"ARMA(1,1) del árbol: pronóstico a 1 año {f(PR['h1'], 3)} con IC 95 % [{f(PR['lo95_h1'], 3)}; {f(PR['hi95_h1'], 3)}] (la diapositiva 60: 1.1328, 0.6122 y 1.6534). A 10 años: {f(PR['h10'], 3)}; el ancho pasa de {f(PR['ancho95_h1'], 2)} a {f(PR['ancho95_h10'], 2)}, casi 2·1.96·sd de la serie ({f(PR['z95_sd_serie'], 2)}).", 0.88)],
        ["ŷₙ₊ₕ = E[yₙ₊ₕ | yₙ, yₙ₋₁, …]", "AR(1): ŷₙ₊ₕ = μ + φʰ(yₙ − μ);  Var = σ²(1 − φ²ʰ)/(1 − φ²)", "IC 95 % de predicción: ŷₙ₊ₕ ± 1.96·√Var del error", "MAE = media |y − ŷ|;  RMSE = √media (y − ŷ)²"],
        ["fc = m.get_forecast(10);  fc.conf_int(alpha=0.05)", "train, test = particion_temporal(y, 24)", "ingenuo(train, 24, 12)"],
        [f"Árbol, validación a ciegas de 40 años: MAE {f(PR['mae']['ARMA(1,1)'], 4)} (ARMA(1,1)), {f(PR['mae']['media del entrenamiento'], 4)} (la media) y {f(PR['mae']['ingenuo (último valor)'], 3)} (ingenuo): con ACF(1) = {f(AB['acf'][0], 2)} el ARMA casi no supera a la media",
         f"Con un solo corte de 12 meses el SARIMA acertó el intervalo del mes 12: [{f(VA['ic95_h12'][0], 2)}; {f(VA['ic95_h12'][1], 2)}] y real {f(VA['real_h12'], 2)}",
         "Un pronóstico a muchos pasos de un ARMA es «la media»: no sirve para mirar el largo plazo",
         "Los intervalos suponen residuos normales y parámetros conocidos: si fallan, el 95 % real es menor"],
        ["El cuaderno de CityBike (C28) alimenta el pronóstico con la temperatura <b>real</b> del periodo de prueba (SARIMAX): es un pronóstico condicionado, no de un futuro desconocido."], fw=0.88)

    topic_page(
        s, "Modelos estacionales SARMA (C27)",
        "Si la serie es mensual, hoy se parece también a hace 12, 24 y 36 meses. En vez de un ARMA con 36 coeficientes se añaden polinomios en <b>Bˢ</b>: <b>SARMA(p, q)(P, Q)ₛ</b>. La ACF muestra picos en los múltiplos de s.",
        [("Operador de rezago B", "B·Yₜ = Yₜ₋₁, Bʲ·Yₜ = Yₜ₋ⱼ, Δ = 1 − B (diferencia) y los polinomios en B se multiplican como polinomios comunes. Con él un ARMA se escribe φ(B)(Yₜ − μ) = θ(B)εₜ."),
         ("Modelo estacional", "φ(B)·Φ(Bˢ)(Yₜ − μ) = θ(B)·Θ(Bˢ)εₜ. Un AR estacional de orden 1 con s = 12, Yₜ = ΦYₜ₋₁₂ + εₜ, tiene ρ(12j) = Φʲ y cero en el resto. P y Q rara vez superan 3 o 4; p y q, 5 o 6."),
         ("Buscar el orden", "A mano es difícil: se prueba una rejilla y se elige el menor AIC (auto.arima en R). Los máximos de p y q deben ser menores que s para que no compitan con la parte estacional.")],
        [("c25_27_usacc.png", f"Muertes accidentales en EE. UU. (n = {US['n']}): SARMA(1,0)(1,0)₁₂ con φ = {f(US['ar1'], 4)}, Φ = {f(US['sar1'], 4)}, media {mil(US['media'], 1)} y AIC {f(US['aic'], 2)} (diapositiva 15: 0.7579, 0.8501, 9214.7102, 1074.89). Ljung-Box p = {f(US['p_lb12'], 2)} (m = 12).", 0.88)],
        ["φ(B)Φ(Bˢ)(Yₜ − μ) = θ(B)Θ(Bˢ)εₜ", "φ(B) = 1 − φ₁B − … − φₚBᵖ;  Φ(Bˢ) = 1 − Φ₁Bˢ − …", "AR(1)ₛ: ρ(k) = Φ^(k/s) si k es múltiplo de s; 0 en otro caso"],
        ["ARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))", "# rejilla p, q ≤ 2; P, Q ≤ 1 y menor AIC"],
        [f"Rejilla de {US['rejilla_n']} modelos: menor AIC es SARMA{tuple(US['mejor_rejilla'][0][:2])}x{tuple(US['mejor_rejilla'][0][2:])} con {f(US['mejor_rejilla'][1], 2)}; el sencillo AR(1)xAR(1) está a ≈ {f(US['aic'] - US['mejor_rejilla'][1], 1)} puntos (AIC similares: se prefiere el parsimonioso)",
         f"Pronóstico a 24 meses: primer mes {mil(US['pron_h1'])} (IC {mil(US['ic_h1'][0])}–{mil(US['ic_h1'][1])}); el máximo es en julio, como en los datos",
         f"Con 12 meses de prueba: MAE {f(US['mae_sarma_12'], 0)} (SARMA), {f(US['mae_ing_est_12'], 0)} (ingenuo estacional) y {f(US['mae_media_12'], 0)} (la media)",
         "El cuaderno de R de la Clase 27 no se entregó; el SARMA se replicó con la serie USAccDeaths de R"],
        [f"La función de R usa σ² con otra convención (132 733 frente a nuestro {mil(US['sigma2'])}); log-verosimilitud y AIC coinciden. El optimizador por defecto de statsmodels se detiene antes en la media (por eso se pule con Nelder-Mead)."], fw=0.86)

    topic_page(
        s, "Series no estacionarias: diferenciación, ARIMA y Dickey-Fuller (C27)",
        "Casi ninguna serie real es estacionaria. Con <b>tendencia estocástica</b> (caminata aleatoria) o determinística, una diferencia <b>ΔYₜ = Yₜ − Yₜ₋₁</b> deja una serie estacionaria; un <b>ARIMA(p, d, q)</b> es un ARMA sobre la serie diferenciada d veces. El test de <b>Dickey-Fuller</b> decide si hace falta.",
        [("Determinística vs estocástica", "Determinística: la serie oscila alrededor de una recta fija (mejor ajustar un polinomio y modelar los errores con ARMA). Estocástica: la tendencia cambia en el tiempo (caminata aleatoria Yₜ = Yₜ₋₁ + εₜ = ARIMA(0,1,0))."),
         ("Cuánto diferenciar", "Casi siempre d = 1; d = 2 en casos extremos. Diferenciar de más aumenta la varianza: para ruido blanco Var(Δᵈε) = C(2d, d)σ² = 1, 2, 6, 20 σ² para d = 0, 1, 2, 3 (en clase se dijo «3σ²» para tres veces, y no es así)."),
         ("Dickey-Fuller", "Regresión Δyₜ = c + γ·yₜ₋₁ + ε con H₀: γ = 0 (φ = 1, raíz unitaria) frente a H₁: γ < 0. El estadístico t <b>no</b> sigue una t de Student: se compara con valores críticos propios (≈ −2.86 al 5 % con constante). Valor p grande → no se rechaza raíz unitaria → hay que diferenciar.")],
        [("c25_27_adf.png", f"Simulación: el 5 % crítico del estadístico es {f(AD['crit5_sim'], 2)}, no −1.65. Índice de precios reconstruido: p = {f(AD['res']['Índice de precios (reconstruido de ipc_var)']['p'], 3)}; log IPSA: p = {f(AD['res']['log IPSA 1980-2000']['p'], 3)}; retorno del IPSA: p ≈ 0.", 0.9)],
        ["ΔYₜ = (1 − B)Yₜ;  Δ²Yₜ = Yₜ − 2Yₜ₋₁ + Yₜ₋₂", "ARIMA(p,d,q): φ(B)(1 − B)ᵈ(Yₜ − μ) = θ(B)εₜ", "retorno log: log Pₜ − log Pₜ₋₁ ≈ (Pₜ − Pₜ₋₁)/Pₜ₋₁", "DF: Δyₜ = c + γyₜ₋₁ + e;  H₀: γ = 0"],
        ["adfuller(y, regression='c', autolag='AIC')", "adf_regresion(y)        # γ̂ y su t", "ARIMA(y, order=(p, 1, q))"],
        [f"Slide 35 (IPC empalmado, no entregado): p = 0.9599 y 0.9438; con el índice reconstruido: p = {f(AD['res']['Índice de precios (reconstruido de ipc_var)']['p'], 3)} con rezagos elegidos por AIC",
         f"La regresión DF sin rezagos dio t = {f(AD['t_indice'], 2)} (p ≈ 0.006): por eso el test <b>aumentado</b> agrega rezagos de ΔY; con mucha autocorrelación el DF simple es engañoso",
         f"Caminata simulada: p = {f(AD['res']['Caminata aleatoria simulada']['p'], 2)} (no rechaza); AR(1) con φ = 0.5: p ≈ 0 (rechaza)",
         "El profesor aplica el test a los retornos de Falabella; como ese archivo no se entregó, aquí se usa el IPSA"],
        ["Las diapositivas 28–29 escriben H₁: φ &lt; 1 y luego H₁: φ ≠ 1; el DF clásico es unilateral. Y «existe evidencia de presencia de tendencia» (diapositiva 35) debe leerse «no se rechaza la raíz unitaria». La diapositiva 39 divide por Pₜ en vez de Pₜ₋₁."], fw=0.88)

    topic_page(
        s, "Cierre del módulo (C28): CityBike, ARIMA, SARIMA y SARIMAX en Python",
        "El cuaderno de cierre aplica lo visto a una serie diaria de viajes de CityBike. Se separan los últimos 30 días como prueba (<b>sin barajar</b>), se ajusta un ARIMA sencillo, se ve que ignora la <b>semana</b> y se pasa a SARIMA con s = 7 y SARIMAX con una variable exógena.",
        [("Qué se hizo", "ARIMA(2,0,2) → ARIMA(7,1,7) (usar 7 rezagos «para explicar hoy») → SARIMA(1,0,1)(1,0,1)₇ → SARIMAX con la temperatura promedio. Se comparan por MAE en el periodo de prueba."),
         ("Qué se aprende", "El patrón recurrente de la serie (semanal) lo capta mejor la parte estacional que 7 rezagos simples; un regresor ayuda si está relacionado con la respuesta."),
         ("Qué hay que mirar con cuidado", f"Los coeficientes AR del SARIMA salen {f(CB['sarima_ar1'], 3)} y {f(CB['sarima_ars'], 3)}: casi raíces unitarias, porque la serie tiene media ≈ {f(CB['media_train'], 0)} y el modelo no tiene intercepto ni diferencia. Y el SARIMAX usa la temperatura <b>real</b> de la prueba.")],
        [("c25_27_citybike.png", f"Reproducido con los datos del cuaderno ({CB['ini']} a {CB['fin']}, {CB['n']} días; entrenamiento {CB['n_train']}): MAE {f(CB['mae']['ARIMA(2,0,2)'], 2)}, {f(CB['mae']['ARIMA(7,1,7)'], 2)}, {f(CB['mae']['SARIMA(1,0,1)(1,0,1)₇'], 2)} y {f(CB['mae']['SARIMAX + temperatura'], 2)} (el cuaderno: 25.47, 25.15, 21.85 y 15.08; el ARIMA(7,1,7) no converge y varía).", 0.9)],
        ["SARIMAX: yₜ = β·xₜ + ARMA estacional en los errores", "MAE = (1/h) Σ |yₜ − ŷₜ|", "error en prueba ≠ AIC del entrenamiento"],
        ["SARIMAX(train, exog=X_train, order=(1,0,1), seasonal_order=(1,0,1,7))", "modelo.forecast(30, exog=X_test)"],
        [f"AIC: {mil(CB['aic']['a202'], 1)} (ARIMA 2,0,2), {mil(CB['aic']['a717'], 1)}, {mil(CB['aic']['sarima'], 1)} y {mil(CB['aic']['sarimax'], 1)} (SARIMAX)",
         f"Temperatura y viajes: correlación {f(CB['temp_corr'], 2)}; la temperatura baja de {f(CB['temp_ini'], 0)} a {f(CB['temp_fin'], 0)} °C mientras los viajes pasan de ≈ {f(CB['v_ini'], 0)} a ≈ {f(CB['v_fin'], 0)} diarios: es en parte una variable de tendencia",
         f"Referencias simples en la prueba: ingenuo estacional {f(CB['mae']['ingenuo estacional (s = 7)'], 1)} y media de 7 días {f(CB['mae']['media de los últimos 7 días'], 1)}",
         "El cuaderno resume Ljung-Box solo en el rezago 1; hay que mirar más rezagos"],
        ["El cuaderno carga los datos desde hojas de Google (URL): si el enlace deja de existir no se puede reproducir. Aquí hay una copia local en <font name='Mono'>citybike_serie_diaria.csv</font>."], fw=0.9)

    # --- página de cierre
    s.append(band("Clases 25-27 · Hoja de referencia, alertas y cierre", FUENTE, C25))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Hoja de referencia (cheat sheet)</b>", "cellb"),
                  table([["Pregunta", "Herramienta", "Cómo se decide"],
                         ["¿Es ruido blanco?", "gráfico + ACF + Ljung-Box", "sin tendencia, varianza constante y ACF dentro de ±1.96/√n"],
                         ["¿Tendencia o estacionalidad?", "descomposición, regresión con dummies, Δ y Δₛ", "residuos sin patrón; en la ACF, picos en múltiplos de s"],
                         ["¿Estacionaria?", "Dickey-Fuller aumentado", "p &lt; 0.05 rechaza raíz unitaria; si no, diferenciar (d = 1)"],
                         ["¿Qué orden?", "ACF y PACF", "AR: PACF corta / MA: ACF corta / ARMA: ambas decaen; rejilla + menor AIC"],
                         ["¿Es bueno el ajuste?", "residuos: ACF, Ljung-Box (gl = m − p − q), QQ", "p &gt; 0.05 en varios m; si no, otro modelo"],
                         ["¿Cuál modelo?", "AIC/BIC (mismo método), parsimonia, error fuera de muestra", "lo que mejor pronostica en <b>prueba temporal</b> y supera al ingenuo"],
                         ["¿Qué entregar?", "pronóstico puntual + intervalo de predicción", "indicar horizonte, supuestos y qué pasa si cambia el régimen"]],
                        [40 * mm, 56 * mm, CONTENT_W - 96 * mm - 12], header_color=C25)], bg=SOFT, border=C25, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Lo dicho en clase y en las diapositivas (verificado)</b>", "cellb"),
                  bullets([f"<b>Código del IPC</b>: lee la tabla por filas y mezcla los meses (verificado: ACF(1) {f(EJ['ipc_acf1_codigo_R'], 2)} frente a {f(EJ['ipc_acf1_ok'], 2)} bien leída). <b>PM2.5</b>: la diapositiva dice 1990–2000, los datos son {EJ['pm_ini'][:4]}–{EJ['pm_fin'][:4]}. <b>Pasajeros</b>: la diapositiva dice desde 2010 pero el archivo parte en 2002-10 (el profesor lo corrigió en clase).",
                           f"<b>Diapositiva 50</b>: «Yₜ = 1.0 − 0.652Yₜ₋₁ + εₜ + 0.415εₜ₋₁» tiene los signos al revés: el ajuste (MLE) es φ = +{f(MO['ARMA(1,1)']['mle_params'][0], 3)} y, en la notación de la diapositiva 25, θ = +{f(-MO['ARMA(1,1)']['mle_params'][1], 3)} (εₜ − 0.415εₜ₋₁); además la media es 1.0, no el intercepto. <b>Diapositiva 47</b>: el MA(3) fijado tiene ma₂ = ma₃ = −0.0863 en la convención de R, no +0.086.",
                           f"<b>Ljung-Box</b> (diapositiva 53): al compararlo con χ² hay que restar los parámetros ARMA estimados (gl = m − p − q); en el árbol con m = 10 el valor p pasa de {f(DG['p10_arma_sin_gl'], 3)} a {f(DG['p10_arma'], 3)}. <b>Kolmogorov-Smirnov</b> con μ y σ estimados (diapositivas 56 y 46) da valores p inflados; corresponde Lilliefors ({f(DG['ks_p'], 2)} frente a {f(DG['lil_p'], 3)}).",
                           "<b>Intervalos</b>: se les llama «de confianza» (clase y cuaderno de cierre) pero son de <b>predicción</b>. En la transcripción «75 %» es el nivel 80 % que dibuja forecast.",
                           f"<b>Varianza de la diferenciación</b>: no es «tres veces σ²» tras tres diferencias sino 20σ² (verificado: {', '.join(f(v, 2) for v in TE['var_emp'])} para d = 0 a 3). <b>Diapositiva 39</b>: (Pₜ − Pₜ₋₁)/Pₜ₋₁, no /Pₜ. <b>Diapositivas 6 y 11</b>: θ(B) con + y con −. <b>Diapositiva 10</b>: Φ(B²) debe ser Φ(Bˢ).",
                           "<b>Dickey-Fuller</b>: las diapositivas 28–29 mezclan H₁: φ &lt; 1 y φ ≠ 1; el estadístico no es una t. «Tendencia determinística + ARIMA siempre sirve» es impreciso: diferenciar una recta crea un MA(1) con θ = 1 y más varianza; conviene ajustar la recta.",
                           f"<b>Cuaderno de cierre</b>: el SARIMA converge a raíces casi unitarias (φ = {f(CB['sarima_ar1'], 3)}, Φ = {f(CB['sarima_ars'], 3)}) y el SARIMAX usa temperatura real del futuro. <b>Dato</b>: «IPSA = 50 empresas» (clase); según mi conocimiento externo hoy son 30 (no verificable con los materiales)."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Errores comunes, preguntas de repaso y criterio de dominio</b>", "cellb"),
                  bullets(["Errores: confundir ruido blanco con independencia · leer la ACF de una serie con tendencia como «dependencia» · comparar AIC de métodos distintos · usar una t de Student en Dickey-Fuller · diferenciar de más · validar al azar · olvidar el intervalo de predicción · pronosticar a largo plazo con un ARMA",
                           "Repaso: ¿qué es ρ(k) y por qué ±1.96/√n? · ¿cómo se ve un AR(2) en ACF y PACF? · ¿por qué Δ²ε tiene varianza 6σ²? · ¿qué hace 1 − B¹²? · ¿cuándo falla Ljung-Box y qué se hace? · ¿qué entrega un intervalo de predicción que no entrega uno de confianza?",
                           "Dominio (marca ✓): " + SYM + " dibujo ACF/PACF de un AR(1), MA(1) y ARMA(1,1)  " + SYM + " calculo una autocorrelación muestral a mano  " + SYM + " diferencio y diferencio estacionalmente a mano  " + SYM + " hago un pronóstico AR(1) con su intervalo  " + SYM + " leo una salida de Dickey-Fuller  " + SYM + " elijo entre modelos con AIC y diagnóstico  " + SYM + " valido en el tiempo y gano al ingenuo  " + SYM + " resuelvo los bloques del cuaderno con ≥ 80 %",
                           "<b>Siguiente</b>: Machine Learning (C30–32), validación (C38) y bootstrap (C43)."], "cell", GREEN)], bg=GREEN_BG, border=GREEN, pad=5))
    build("04_SeriesTiempo_C25-27.pdf", s, "Clases 25-27 · Modelos predictivos de series de tiempo")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#4D7C0F">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clases 25-27 · Modelos predictivos de series de tiempo", "Para leer antes de los ejercicios · lectura de ~40 min", C25))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 8 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Ruido blanco: media 0, varianza constante, covarianza 0", "Es el residuo que buscamos; ni tendencia ni dependencia."),
                            ("ACF mide dependencia y no depende de la escala", "r(k) con bandas ±1.96/√n."),
                            ("Descomponer = regresión", "Y = T + S + ε con dummies; los residuos aún tienen estructura."),
                            ("AR: PACF corta · MA: ACF corta · ARMA: ambas decaen", "Son pistas; luego se estima y se compara por AIC."),
                            ("Residuos ruido blanco o el modelo no sirve", "ACF, Ljung-Box con gl = m − p − q."),
                            ("Pronóstico = esperanza condicional + intervalo de predicción", "Un ARMA vuelve a la media."),
                            ("Tendencia estocástica → diferenciar (d = 1); raíz unitaria → Dickey-Fuller", "Δᵈ y 1 − Bˢ; no diferenciar de más."),
                            ("Validar en el tiempo", "Corte u origen móvil, siempre contra el ingenuo.")], C25)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C25-27_SeriesTiempo.ipynb</font> o el cuaderno <font name='Mono'>04_Ejercicios_Cuaderno_SeriesTiempo.pdf</font>.", "small"))

    # 1
    s.append(h("1", "Ruido blanco y autocorrelación muestral"))
    s += para("La autocorrelación muestral a rezago k divide la suma de productos de desviaciones (a distancia k) por la suma de cuadrados de todas las desviaciones, con la media de <b>toda</b> la serie. Si la serie es ruido blanco, cada r(k) cae en ±1.96/√n con probabilidad ≈ 95 %.")
    y1 = np.array([12, 15, 14, 18, 17, 21, 20, 24], float)
    r1 = acf_manual(y1, 2)
    d1 = y1 - y1.mean()
    s.append(ejemplo("serie de 8 valores: 12, 15, 14, 18, 17, 21, 20, 24", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Media y desviaciones", f"ȳ = {f(y1.mean(), 2)}", "−5.25, −2.25, −3.25, 0.75, −0.25, 3.75, 2.75, 6.75"],
               ["Suma de cuadrados", "Σ(yₜ − ȳ)²", f"{f(np.sum(d1 ** 2), 3)}"],
               ["Producto a rezago 1", "Σ (yₜ − ȳ)(yₜ₊₁ − ȳ), 7 términos", f"{f(np.sum(d1[:-1] * d1[1:]), 3)}  →  r₁ = {f(r1[1], 3)}"],
               ["Producto a rezago 2", "6 términos", f"{f(np.sum(d1[:-2] * d1[2:]), 3)}  →  r₂ = {f(r1[2], 3)}"],
               ["Banda al 95 %", "±1.96/√8", f"±{f(1.96 / sqrt(8), 3)}"]],
              [40 * mm, 56 * mm, CONTENT_W - 96 * mm - 18]),
        P(f"Lectura: r₁ = {f(r1[1], 2)} no sale de la banda (con n = 8 la banda es enorme); la serie sube con el tiempo, así que la ACF se interpreta mal: antes hay que quitar la tendencia. <b>Error común</b>: dividir por n − k en el denominador o usar medias distintas en cada tramo.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_ejemplos.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), imgc("c25_27_acf_idea.png", CONTENT_W * 0.86, 52 * mm), Spacer(1, 2), imgc("c25_27_ruido_blanco.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), pista("la Parte A del notebook (A2 y A3)", [P("Simula las tres series, compara tu <font name='Mono'>acf_manual</font> con <font name='Mono'>statsmodels</font> y cuenta cuántas barras salen de la banda.", "cell")])])

    # 2
    s.append(h("2", "Descomposición aditiva a mano"))
    s += para("Con datos trimestrales (s = 4) la tendencia es una media móvil centrada con pesos (½, 1, 1, 1, ½)/4; se pierden 2 datos por lado. Se resta la tendencia, se promedia por trimestre y se recentra el efecto estacional para que sume 0; el ruido es lo que queda.")
    y2 = np.array([30, 24, 36, 48, 36, 30, 42, 54, 42, 36, 48, 60], float)
    t2, s2_, e2 = descomponer_aditiva(y2, 4)
    s.append(ejemplo("ventas trimestrales de 3 años: 30, 24, 36, 48 | 36, 30, 42, 54 | 42, 36, 48, 60", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Tendencia t = 3", "(0.5·30 + 24 + 36 + 48 + 0.5·36)/4", f"{f(t2[2], 3)}"],
               ["Tendencia t = 4", "(0.5·24 + 36 + 48 + 36 + 0.5·30)/4", f"{f(t2[3], 3)}"],
               ["Efecto sin recentrar T1", "media de (y − tendencia) en los trimestres 1", f"{f(np.nanmean((y2 - t2)[0::4]), 3)}"],
               ["Efectos estacionales", "recentrados (suman 0)", ", ".join(f(v, 3) for v in s2_[:4])],
               ["Ruido en t = 3", "36 − tendencia − estacional T3", f"{f(e2[2], 3)}"],
               ["Datos perdidos", "s/2 por extremo", "2 al inicio y 2 al final"]],
              [40 * mm, 64 * mm, CONTENT_W - 104 * mm - 18]),
        P("Lectura: el trimestre 4 siempre es el más alto (+) y el 2, el más bajo (−). <b>Error común</b>: usar una media móvil no centrada para s par, o no recentrar la estacionalidad.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_descomposicion.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), imgc("c25_27_regresion.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), imgc("c25_27_armonica_boxcox.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), pista("la Parte A del notebook (A4 y A5)", [P("Reproduce con <font name='Mono'>descomponer_aditiva</font> los coeficientes de la diapositiva 33 y mira la ACF de los residuos de la regresión.", "cell")])])

    # 3
    s.append(h("3", "ACF teórica de AR y MA, y condiciones"))
    s += para("Para un AR(1), ρ(k) = φᵏ. Para un AR(2), ρ(1) = φ₁/(1 − φ₂) y ρ(2) = φ₂ + φ₁²/(1 − φ₂); desde k = 3, ρ(k) = φ₁ρ(k − 1) + φ₂ρ(k − 2). Para un MA(1) con Yₜ = εₜ − θεₜ₋₁, ρ(1) = −θ/(1 + θ²) y ρ(k) = 0 para k ≥ 2.")
    p1, p2 = 0.4, 0.2
    r1_ = p1 / (1 - p2)
    r2_ = p2 + p1 ** 2 / (1 - p2)
    r3_ = p1 * r2_ + p2 * r1_
    th = 0.5
    s.append(ejemplo("AR(2) con φ = (0.4, 0.2) y MA(1) con θ = 0.5", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["¿Estacionario?", "φ₂ + φ₁ = 0.6 &lt; 1; φ₂ − φ₁ = −0.2 &lt; 1; |φ₂| &lt; 1", "sí"],
               ["ρ(1) del AR(2)", "0.4/(1 − 0.2)", f"{f(r1_, 3)}"],
               ["ρ(2) del AR(2)", "0.2 + 0.16/0.8", f"{f(r2_, 3)}"],
               ["ρ(3) del AR(2)", "0.4·ρ(2) + 0.2·ρ(1)", f"{f(r3_, 3)}"],
               ["ρ(1) del MA(1)", "−0.5/(1 + 0.25)", f"{f(-th / (1 + th ** 2), 3)}"],
               ["ρ(2) del MA(1)", "k ≥ 2", "0"],
               ["PACF del AR(2)", "se corta en 2", "α(1) = ρ(1) = 0.5; α(2) = φ₂ = 0.2; α(3) = 0"]],
              [40 * mm, 72 * mm, CONTENT_W - 112 * mm - 18]),
        P("Lectura: el AR(2) deja dos valores en la PACF; el MA(1) uno en la ACF. <b>Error común</b>: usar el signo de R (+θ) con las fórmulas de las diapositivas (−θ).", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_ar.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), imgc("c25_27_ma_arma.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), imgc("c25_27_identificacion.png", CONTENT_W * 0.86, 52 * mm), Spacer(1, 2), pista("la Parte B del notebook (B1 y B2)", [P("Simula AR(1), AR(2), MA(1), MA(2) y ARMA(1,1), compara la ACF/PACF muestral con la teórica y revisa las raíces del polinomio.", "cell")])])

    # 4
    s.append(h("4", "Comparar modelos: AIC, BIC y parsimonia"))
    s += para("AIC = −2 log L + 2k y BIC = −2 log L + k·log n. El menor es mejor; BIC castiga más los parámetros. Una diferencia de menos de 2 puntos es «empate». Solo es válido comparar valores calculados con el mismo método y los mismos datos.")
    ll = {"AR(1)": (-250.0, 2), "AR(2)": (-246.0, 3), "ARMA(1,1)": (-245.5, 3), "ARMA(2,2)": (-244.8, 5)}
    n4 = 120
    filas = [["Modelo", "log L", "k", "AIC", "BIC"]]
    for nm, (l, k) in ll.items():
        filas.append([nm, f(l, 1), str(k), f(-2 * l + 2 * k, 1), f(-2 * l + k * np.log(n4), 1)])
    s.append(ejemplo("n = 120 observaciones y cuatro modelos ya ajustados por máxima verosimilitud (k incluye la media y σ²)", [
        pasos(filas, [38 * mm, 28 * mm, 20 * mm, 30 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: el AR(2) y el ARMA(1,1) quedan a menos de 2 puntos de AIC; el ARMA(2,2) agrega 2 parámetros para mejorar 0.7 de log L y lo pierde. Se elige entre los empatados el de residuos limpios y menos parámetros. <b>Error común</b>: comparar el AIC de tseries (CSS) con el de arima (MLE).", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_arbol_serie.png", CONTENT_W * 0.86, 52 * mm), Spacer(1, 2), imgc("c25_27_diagnostico.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), imgc("c25_27_arbol_modelos.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), pista("la Parte B del notebook (B4)", [P("Ajusta los cinco modelos del árbol por CSS (réplica de tseries) y por MLE (statsmodels) y verifica las dos tablas de AIC de las diapositivas.", "cell")])])

    # 5
    s.append(h("5", "Pronóstico AR(1) con intervalo de predicción"))
    s += para("Para un AR(1) con media μ, el pronóstico a h pasos es μ + φʰ(yₙ − μ) y la varianza del error es σ²(1 − φ²ʰ)/(1 − φ²). El intervalo del 95 % es el pronóstico ± 1.96 veces la raíz de esa varianza.")
    mu5, phi5, sg5, y5 = 50.0, 0.7, 4.0, 58.0
    filas = [["h", "pronóstico", "sd del error", "IC 95 %"]]
    for hh in (1, 2, 3, 10):
        pt = mu5 + phi5 ** hh * (y5 - mu5)
        sd = sg5 * sqrt((1 - phi5 ** (2 * hh)) / (1 - phi5 ** 2))
        filas.append([str(hh), f(pt, 3), f(sd, 3), ci([pt - 1.96 * sd, pt + 1.96 * sd], 2)])
    s.append(ejemplo("AR(1): μ = 50, φ = 0.7, σ = 4 y último valor yₙ = 58", [
        pasos(filas, [20 * mm, 36 * mm, 40 * mm, CONTENT_W - 96 * mm - 18]),
        P(f"Lectura: el pronóstico desciende hacia 50 y la sd del error sube de 4 hacia σ/√(1 − φ²) = {f(sg5 / sqrt(1 - phi5 ** 2), 3)} (la sd de la serie). <b>Error común</b>: dejar el intervalo con ancho constante o calcularlo solo con σ.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_pronostico.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), imgc("c25_27_puentes.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), pista("la Parte B del notebook (B6)", [P("Calcula tu pronóstico AR(1) con <font name='Mono'>ar1_pronostico</font>, compáralo con <font name='Mono'>get_forecast</font> y mide el MAE en los últimos 40 años.", "cell")])])

    # 6
    s.append(h("6", "Diferenciación simple y estacional"))
    s += para("La primera diferencia ΔYₜ = Yₜ − Yₜ₋₁ elimina tendencias lineales; la diferencia estacional ΔₛYₜ = Yₜ − Yₜ₋ₛ elimina un patrón de periodo s. Se pierden d + s·D datos. Diferenciar de más aumenta la varianza.")
    y6 = np.array([20, 24, 30, 26, 32, 37, 43, 38, 44, 49, 56, 50], float)
    d1_ = np.diff(y6)
    d4 = y6[4:] - y6[:-4]
    s.append(ejemplo("serie trimestral de 12 valores: 20, 24, 30, 26 | 32, 37, 43, 38 | 44, 49, 56, 50", [
        pasos([["Transformación", "Cálculo", "Resultado"],
               ["ΔY (11 valores)", "24 − 20, 30 − 24, …", ", ".join(f"{v:.0f}" for v in d1_)],
               ["Δ₄Y (8 valores)", "32 − 20, 37 − 24, …", ", ".join(f"{v:.0f}" for v in d4)],
               ["Varianza de Y, ΔY y Δ₄Y", "muestral", f"{f(y6.var(ddof=1), 1)}, {f(d1_.var(ddof=1), 1)} y {f(d4.var(ddof=1), 2)}"],
               ["ΔΔ₄Y (7 valores)", "diferencia de Δ₄Y", ", ".join(f"{v:.0f}" for v in np.diff(d4))],
               ["Operador", "(1 − B)(1 − B⁴)Y = Yₜ − Yₜ₋₁ − Yₜ₋₄ + Yₜ₋₅", "se pierden 1 + 4 = 5 datos"]],
              [42 * mm, 66 * mm, CONTENT_W - 108 * mm - 18]),
        P("Lectura: la diferencia estacional deja casi constante (≈ 12) y la simple todavía muestra el patrón de cuatro trimestres. <b>Error común</b>: aplicar la diferencia estacional con el s equivocado o repetirla sin necesidad.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_tendencias.png", CONTENT_W * 0.8, 52 * mm), Spacer(1, 2), pista("la Parte C del notebook (C4)", [P("Diferencia las dos series simuladas y comprueba la varianza de Δᵈ del ruido blanco: 1, 2, 6 y 20.", "cell")])])

    # 7
    s.append(h("7", "Leer un test de Dickey-Fuller"))
    s += para("El test estima Δyₜ = c + γ·yₜ₋₁ + e (con rezagos de Δy en la versión aumentada) y contrasta H₀: γ = 0 (hay raíz unitaria) con H₁: γ &lt; 0. Se rechaza H₀ solo si el estadístico es <b>menor</b> que el valor crítico (más negativo) o si p &lt; 0.05. El valor crítico con constante es ≈ −2.86 al 5 %.")
    s.append(ejemplo("tres salidas con constante, valor crítico 5 % = −2.87", [
        pasos([["Serie", "Estadístico · valor p", "Decisión y acción"],
               ["Índice de ventas", "−1.52 · 0.52", "No se rechaza H₀ (−1.52 &gt; −2.87): hay raíz unitaria → diferenciar y repetir"],
               ["Ventas diferenciadas", "−6.80 · 0.00", "Se rechaza H₀: estacionaria; modelar con ARMA (ARIMA con d = 1)"],
               ["Tasa de interés", "−3.10 · 0.03", "Se rechaza al 5 % (p &lt; 0.05) pero no al 1 % (crítico ≈ −3.45): evidencia moderada; mirar el gráfico"]],
              [32 * mm, 40 * mm, CONTENT_W - 72 * mm - 18]),
        P("Lectura: «p grande» significa <b>no</b> poder descartar la raíz unitaria, no probar que la haya. Con el parámetro del cuaderno, γ = φ − 1. <b>Error común</b>: comparar el estadístico con ±1.96 o con una t de Student.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_adf.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), imgc("c25_27_retornos.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2), pista("la Parte C del notebook (C5)", [P("Reproduce la regresión DF a mano con <font name='Mono'>adf_regresion</font> y compárala con <font name='Mono'>adfuller</font> y con una simulación de la distribución del estadístico.", "cell")])])

    # 8
    s.append(h("8", "Modelos estacionales y validación temporal"))
    s += para("El modelo AR estacional Yₜ = ΦYₜ₋ₛ + εₜ tiene ρ(k) = Φ^(k/s) en los múltiplos de s. En la práctica se multiplican polinomios: (1 − φB)(1 − ΦBˢ) = 1 − φB − ΦBˢ + φΦBˢ⁺¹. Para validar, se separa el final de la serie como prueba y se comparan los errores con los del ingenuo estacional.")
    Phi, ph = 0.5, 0.6
    real = np.array([100, 120, 90, 110], float)
    p_mod = np.array([104, 115, 95, 108], float)
    p_ing = np.array([95, 118, 85, 100], float)
    train8 = np.array([80, 100, 70, 90, 90, 112, 80, 100], float)
    esc = np.mean(np.abs(train8[4:] - train8[:-4]))
    s.append(ejemplo("AR estacional con s = 12 y Φ = 0.5; y una prueba de 4 meses", [
        pasos([["Pregunta", "Cálculo", "Resultado"],
               ["ρ(12), ρ(24), ρ(36)", "0.5, 0.5², 0.5³", f"{f(Phi, 3)}, {f(Phi ** 2, 3)}, {f(Phi ** 3, 3)}"],
               ["ρ(5) del AR estacional", "k no es múltiplo de 12", "0"],
               ["Polinomio (1 − 0.6B)(1 − 0.5B¹²)", "multiplicar", f"1 − 0.6B − 0.5B¹² + {f(ph * Phi, 2)}B¹³"],
               ["MAE del modelo", "(4 + 5 + 5 + 2)/4", f"{f(np.mean(np.abs(real - p_mod)), 2)}"],
               ["MAE del ingenuo estacional", "(5 + 2 + 5 + 10)/4", f"{f(np.mean(np.abs(real - p_ing)), 2)}"],
               ["MASE (escala = MAE ingenuo estacional del entrenamiento)", f"{f(np.mean(np.abs(real - p_mod)), 2)}/{f(esc, 1)}", f"{f(np.mean(np.abs(real - p_mod)) / esc, 3)}"]],
              [48 * mm, 54 * mm, CONTENT_W - 102 * mm - 18]),
        P("Lectura: MASE &lt; 1 significa que el modelo gana a la referencia estacional. <b>Error común</b>: evaluar con los mismos datos del ajuste o con una partición aleatoria.", "cell")]))
    s.extend([Spacer(1, 2), imgc("c25_27_sar.png", CONTENT_W * 0.86, 52 * mm), Spacer(1, 2), imgc("c25_27_validacion.png", CONTENT_W * 0.9, 52 * mm), Spacer(1, 2),
                           pista("la Parte C (C2, C3 y C7) y el ejercicio de cierre", [P("Ajusta SARMA a los accidentes y SARIMA a los pasajeros, compara contra el ingenuo estacional con origen móvil y repite con el IAC.", "cell")])])
    s.append(Spacer(1, 2))
    s.append(recuerda(["r(k) usa la media global y la suma de cuadrados total; banda ±1.96/√n.",
                       "Descomposición: tendencia (media móvil centrada), estacionalidad (media por mes recentrada), ruido.",
                       "AR: ACF decae y PACF corta en p · MA: ACF corta en q y PACF decae · ARMA: ambas decaen.",
                       "AIC = −2 log L + 2k (mismo método); residuos ruido blanco con gl = m − p − q.",
                       "ARIMA(p, d, q): diferenciar d veces; Dickey-Fuller no usa una t de Student.",
                       "Valida con corte u origen móvil y compara con el ingenuo; entrega un intervalo de predicción.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–8 → Parte C → ejercicio de cierre."]))
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 4]))
            i += 4
        else:
            out.append(s[i])
            i += 1
    build("apuntes/04_Apuntes_SeriesTiempo_C25-27.pdf", out, "Apuntes · Clases 25-27 · Series de tiempo")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#4D7C0F">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C25))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clases 25-27 · Series de tiempo", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C25))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: r(k) = Σ(yₜ − ȳ)(yₜ₊ₖ − ȳ)/Σ(yₜ − ȳ)² · banda ±1.96/√n · AR(1): ρ(k) = φᵏ · MA(1): ρ(1) = −θ/(1 + θ²) · ŷₙ₊ₕ = μ + φʰ(yₙ − μ) · Var = σ²(1 − φ²ʰ)/(1 − φ²) · AIC = −2 log L + 2k · BIC = −2 log L + k log n.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C25)], bg=SOFT, border=C25, pad=5))
    s.append(Spacer(1, 2))

    # datos
    yb2 = np.array([10, 14, 12, 16, 15, 19, 17, 22], float)
    rb2 = acf_manual(yb2, 2)
    s += bloque(1, "Identificar componentes en un gráfico", "★", "Se describen cuatro series. Para cada una indica tendencia (no / lineal / estocástica), estacionalidad (periodo), si la descomposición es aditiva o multiplicativa y si la transformarías antes de modelar.", [
        "A: ventas mensuales de helados; las alzas de verano crecen cada año en proporción al nivel y hay un aumento lento del nivel.",
        "B: temperatura media diaria de una ciudad durante 10 años; oscila alrededor de un valor con amplitud constante.",
        "C: precio de cierre de una acción; sube y baja sin un patrón regular y a veces se queda arriba por largos periodos.",
        "D: pasajeros mensuales de un aeropuerto, con un salto brusco en marzo de 2020 que luego se recupera."])
    s += bloque(2, "Autocorrelación muestral a mano", "★★", f"Serie de n = 8 valores: {', '.join(f'{v:.0f}' for v in yb2)}.", [
        "Calcula la media y la suma de cuadrados de las desviaciones.",
        "Calcula r₁ y r₂.",
        "Calcula la banda ±1.96/√n y concluye si r₁ y r₂ son significativas. ¿Qué advertencia agregas por el tamaño de n?"])
    s += bloque(3, "Identificar AR / MA / ARMA", "★★", "Con n = 200 (banda ±0.139) se obtienen: caso A, ACF = 0.62, 0.38, 0.24, 0.15, 0.09 y PACF = 0.62, 0.02, −0.03, 0.05, 0.01. Caso B, ACF = −0.45, 0.04, 0.02, −0.05, 0.03 y PACF = −0.45, −0.28, −0.17, −0.12, −0.05. Caso C, ACF = 0.50, 0.30, 0.17, 0.09, 0.04 y PACF = 0.50, 0.07, 0.03, −0.01, 0.02.", [
        "Para cada caso, indica qué modelo y orden sugieren la ACF y la PACF.",
        "En el caso A, estima φ y predice ρ(3) con la fórmula de un AR(1); compara con 0.24.",
        "En el caso B, ¿qué signo tiene θ en la notación de las diapositivas? Calcula θ a partir de ρ(1) = −θ/(1 + θ²) (toma la raíz con |θ| &lt; 1)."])
    y4 = np.array([40, 44, 52, 48, 51, 56, 63, 57, 61, 66, 74, 67], float)
    s += bloque(4, "Diferencias a mano", "★★", f"Serie trimestral: {', '.join(f'{v:.0f}' for v in y4)}.", [
        "Calcula ΔY y su varianza muestral.",
        "Calcula Δ₄Y y su varianza muestral; ¿cuál elimina mejor el patrón?",
        "¿Cuántos datos quedan en (1 − B)(1 − B⁴)Y? Escribe el polinomio expandido y calcula el primer valor."])
    s += bloque(5, "Pronóstico AR(1)", "★★★", "AR(1) estimado: μ = 120, φ = 0.8, σ = 6, con último valor yₙ = 132.", [
        "Calcula el pronóstico y la sd del error para h = 1, 2 y 5.",
        "Construye el intervalo de predicción del 95 % para h = 1 y h = 5.",
        "¿Hacia qué valor convergen el pronóstico y la sd del error cuando h crece? ¿Qué significa para un pronóstico a largo plazo?"])
    s += bloque(6, "Leer Dickey-Fuller", "★★", "Test aumentado con constante, valor crítico al 5 % = −2.87. Serie P: estadístico −0.95, p = 0.77. Serie Q: −4.20, p = 0.001. Serie R: −2.60, p = 0.09. Además γ̂ = −0.05 con error estándar 0.02 en una regresión de Δy sobre y₋₁.", [
        "Para P, Q y R escribe H₀, la decisión y qué harías con cada serie.",
        "Calcula el estadístico t = γ̂/error estándar y compáralo con −1.96 y con −2.87. ¿Por qué da decisiones distintas?",
        "Si en la serie R el gráfico muestra una tendencia lineal clara, ¿qué opciones tienes además de diferenciar?"])
    s += bloque(7, "Comparar modelos con AIC", "★★★", "n = 150. Resultados de máxima verosimilitud: AR(1): log L = −310.0, k = 3; AR(2): −305.2, k = 4; ARMA(1,1): −304.8, k = 4; ARMA(2,2): −303.9, k = 6.", [
        "Calcula AIC y BIC de cada modelo.",
        "¿Qué modelo elegiría el AIC? ¿Y el BIC? Justifica con la parsimonia.",
        "Los residuos del AR(2) tienen Ljung-Box con m = 10: Q = 14.1. Calcula los grados de libertad y decide con χ²₀.₉₅(8) = 15.51. ¿Qué cambia si no restas los parámetros (χ²₀.₉₅(10) = 18.31)?"])
    s += bloque(8, "Elegir un esquema de validación", "★★", "Tres situaciones: (i) 8 años de ventas mensuales para pronosticar 12 meses; (ii) 90 días de demanda diaria para pronosticar 7 días y recalibrar cada semana; (iii) un conjunto de 500 clientes sin orden temporal. Además, en la situación (i) las predicciones de dos modelos sobre 4 meses de prueba son: real 50, 56, 61, 58; modelo A: 52, 55, 58, 60; modelo B (ingenuo estacional): 47, 52, 66, 52.", [
        "Indica para cada situación qué esquema usarías (corte, origen móvil o validación cruzada usual) y por qué.",
        "Calcula MAE y RMSE de A y de B en los 4 meses.",
        "En (i) con corte de 12 meses y origen móvil con paso de 12, ¿cuántos cortes caben si se exige un entrenamiento mínimo de 4 años?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C25))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C25), Spacer(1, 2)]

    s += sol("Bloque 1", [
        ["A", "Tendencia lineal al alza, estacionalidad de 12 meses y amplitud que crece con el nivel → <b>multiplicativa</b>; se aplica log (o Box-Cox) para volverla aditiva y estabilizar la varianza."],
        ["B", "Sin tendencia, estacionalidad anual de periodo 365 (o armónicas), amplitud constante → <b>aditiva</b>; se puede usar armónicas en vez de 365 dummies."],
        ["C", "Tendencia estocástica (caminata aleatoria), sin estacionalidad; no hay descomposición aditiva útil: se trabaja con el log-retorno log Pₜ − log Pₜ₋₁."],
        ["D", "Tendencia y estacionalidad de 12 meses más una <b>intervención</b> (marzo 2020); se agrega una variable indicadora o se modela el tramo afectado por separado."]])
    d2 = yb2 - yb2.mean()
    s += sol("Bloque 2", [
        ["1", f"ȳ = {f(yb2.mean(), 3)}; desviaciones {', '.join(f(v, 3) for v in d2)}; suma de cuadrados = {f(np.sum(d2 ** 2), 3)}."],
        ["2", f"Σ rezago 1 = {f(np.sum(d2[:-1] * d2[1:]), 3)} → r₁ = {f(rb2[1], 3)}; Σ rezago 2 = {f(np.sum(d2[:-2] * d2[2:]), 3)} → r₂ = {f(rb2[2], 3)}."],
        ["3", f"Banda = 1.96/√8 = {f(1.96 / sqrt(8), 3)}: ni r₁ ni r₂ salen de la banda. Con n = 8 la ACF casi no tiene potencia (y la serie tiene tendencia), así que no se concluye ruido blanco: faltan datos."]])
    phiA = 0.62
    th_ = (1 - sqrt(1 - 4 * 0.45 ** 2)) / (2 * 0.45)
    s += sol("Bloque 3", [
        ["1", "A: ACF decae y la PACF tiene un solo valor significativo (0.62; los demás &lt; 0.139) → <b>AR(1)</b>. B: la ACF tiene un solo valor significativo (−0.45) y la PACF decae con alternancia → <b>MA(1)</b>. C: la PACF tiene un solo valor significativo (0.50) → sugiere <b>AR(1)</b>; la ACF decae un poco más lento que 0.5ᵏ (0.30 y 0.17 frente a 0.25 y 0.125), diferencia dentro del error de muestreo: se ajustan AR(1) y ARMA(1,1) y se decide por AIC y residuos"],
        ["2", f"φ̂ = ρ(1) = 0.62; ρ(3) = 0.62³ = {f(0.62 ** 3, 3)}, frente a 0.24 observada (cercano: la diferencia está dentro del error de muestreo)."],
        ["3", f"θ es <b>positivo</b>: ρ(1) &lt; 0 con ρ(1) = −θ/(1 + θ²). Resolviendo 0.45θ² − θ + 0.45 = 0 → θ = {f(th_, 3)} (la otra raíz es {f(1 / th_, 3)} y no cumple |θ| &lt; 1)."]])
    d1y, d4y = np.diff(y4), y4[4:] - y4[:-4]
    s += sol("Bloque 4", [
        ["1", f"ΔY = {', '.join(f'{v:.0f}' for v in d1y)}; varianza muestral = {f(d1y.var(ddof=1), 2)}."],
        ["2", f"Δ₄Y = {', '.join(f'{v:.0f}' for v in d4y)}; varianza = {f(d4y.var(ddof=1), 2)}. La estacional deja valores casi constantes (≈ 11): elimina el patrón de cuatro trimestres y la tendencia a la vez."],
        ["3", f"Se pierden 1 + 4 = 5 datos: quedan 7. (1 − B)(1 − B⁴)Yₜ = Yₜ − Yₜ₋₁ − Yₜ₋₄ + Yₜ₋₅. Primer valor (t = 6): {y4[5]:.0f} − {y4[4]:.0f} − {y4[1]:.0f} + {y4[0]:.0f} = {y4[5] - y4[4] - y4[1] + y4[0]:.0f}."]])
    mu5, ph5, sg5, y5 = 120, 0.8, 6, 132
    pun = lambda hh: mu5 + ph5 ** hh * (y5 - mu5)
    sdh = lambda hh: sg5 * sqrt((1 - ph5 ** (2 * hh)) / (1 - ph5 ** 2))
    s += sol("Bloque 5", [
        ["1", "; ".join(f"h = {hh}: {f(pun(hh), 3)} y sd {f(sdh(hh), 3)}" for hh in (1, 2, 5)) + "."],
        ["2", f"h = 1: {f(pun(1), 2)} ± 1.96·{f(sdh(1), 3)} = {ci([pun(1) - 1.96 * sdh(1), pun(1) + 1.96 * sdh(1)], 2)}. h = 5: {f(pun(5), 2)} ± 1.96·{f(sdh(5), 3)} = {ci([pun(5) - 1.96 * sdh(5), pun(5) + 1.96 * sdh(5)], 2)}."],
        ["3", f"El pronóstico converge a μ = {mu5} y la sd a σ/√(1 − φ²) = {f(sg5 / sqrt(1 - ph5 ** 2), 3)} (la de la serie): a largo plazo el modelo solo dice «la media», con la incertidumbre de la serie completa."]])
    gam = -0.05 / 0.02
    s += sol("Bloque 6", [
        ["1", "H₀: raíz unitaria (γ = 0). P: −0.95 &gt; −2.87 y p = 0.77 → no se rechaza → diferenciar. Q: −4.20 &lt; −2.87 y p = 0.001 → se rechaza → estacionaria; se modela con ARMA. R: −2.60 &gt; −2.87 y p = 0.09 → no se rechaza al 5 % (evidencia débil); se mira el gráfico y se prueba también la versión con tendencia."],
        ["2", f"t = −0.05/0.02 = {f(gam, 2)}: con la t de Student (−1.96) se rechazaría H₀, con el valor crítico correcto (−2.87) <b>no</b>. Por eso no se usa la t de Student: bajo H₀ el estadístico no es simétrico ni centrado en 0 (su 5 % está en ≈ −2.87)."],
        ["3", "Ajustar una tendencia lineal y modelar los residuos con ARMA (si es determinística), usar la versión del test con tendencia o agregar un regresor; no diferenciar de más, porque diferenciar una recta aumenta la varianza y crea un MA(1) con θ = 1."]])
    mods = {"AR(1)": (-310.0, 3), "AR(2)": (-305.2, 4), "ARMA(1,1)": (-304.8, 4), "ARMA(2,2)": (-303.9, 6)}
    n7 = 150
    aic7 = {k: -2 * l + 2 * kk for k, (l, kk) in mods.items()}
    bic7 = {k: -2 * l + kk * np.log(n7) for k, (l, kk) in mods.items()}
    s += sol("Bloque 7", [
        ["1", "; ".join(f"{k}: AIC {f(aic7[k], 1)}, BIC {f(bic7[k], 1)}" for k in mods) + "."],
        ["2", f"AIC: menor {min(aic7, key=aic7.get)} ({f(min(aic7.values()), 1)}), con ARMA(1,1) e AR(2) a menos de 1 punto (empate); BIC: menor {min(bic7, key=bic7.get)} ({f(min(bic7.values()), 1)}). Se elige uno de los dos de 4 parámetros y se desempata con los residuos y la parsimonia; el ARMA(2,2) agrega 2 parámetros para ganar muy poco."],
        ["3", "gl = m − p − q = 10 − 2 − 0 = 8; Q = 14.1 &lt; 15.51 → no se rechaza ruido blanco. Sin restar (gl = 10), el valor crítico es 18.31 y la conclusión sería la misma, pero con otro valor p; en general omitir la corrección hace al test <b>demasiado permisivo</b>."]])
    r8 = np.array([50, 56, 61, 58.0])
    A, B = np.array([52, 55, 58, 60.0]), np.array([47, 52, 66, 52.0])
    s += sol("Bloque 8", [
        ["1", "(i) <b>Corte</b> de 12 meses (o origen móvil si se quiere evaluar la estabilidad del ranking): se entrena con el pasado y se prueba el último año, sin barajar. (ii) <b>Origen móvil</b> semanal: cada semana se reentrena con todo lo anterior y se pronostica la siguiente. (iii) Sin orden temporal, la <b>validación cruzada usual</b> (aquí no hay fuga hacia el pasado)."],
        ["2", f"A: errores {', '.join(f'{v:.0f}' for v in np.abs(r8 - A))} → MAE = {f(np.mean(np.abs(r8 - A)), 2)}, RMSE = {f(np.sqrt(np.mean((r8 - A) ** 2)), 3)}. B: errores {', '.join(f'{v:.0f}' for v in np.abs(r8 - B))} → MAE = {f(np.mean(np.abs(r8 - B)), 2)}, RMSE = {f(np.sqrt(np.mean((r8 - B) ** 2)), 3)}. Gana A en ambos."],
        ["3", "96 meses; entrenamiento mínimo 48; prueba de 12 con paso 12: orígenes en 48, 60, 72 y 84 → <b>4 cortes</b> (el último origen que deja 12 meses de prueba es 84)."]])
    build("apuntes/04_Ejercicios_Cuaderno_SeriesTiempo.pdf", s, "Ejercicios para el cuaderno · Clases 25-27")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
