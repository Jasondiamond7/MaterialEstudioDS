"""Clase 22 · Experimentación, A/B testing y causalidad aplicada: guía visual, apuntes y cuaderno de ejercicios en papel.
Todos los números de la guía vienen de c22_numeros.json (lo escribe figuras_c22.py); los de los ejemplos y ejercicios se calculan aquí."""
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

N = json.loads((Path(__file__).resolve().parent / "c22_numeros.json").read_text(encoding="utf-8"))
C22 = colors.HexColor("#0D9488")
C22_BG = colors.HexColor("#F0FDFA")
GREEN_C = colors.HexColor("#059669")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas + Clase22_ABtesting.R/.ipynb + datos de la profesora Maritza Márquez"
E, M, PO, SR, PK = N["efecto"], N["modelos"], N["poder"], N["srm"], N["peeking"]
HT, DD = N["hetero"], N["did"]


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{int(round(x)):,}".replace(",", " ")


def pv(p):
    if p < 0.001:
        base, exp = f"{p:.2e}".split("e")
        return f"{float(base):.2f} × 10<super>{int(exp)}</super>"
    return f"{p:.3f}"


# ------------------------------------------------------------ cálculos reutilizables
def z2(x1, n1, x2, n2):
    """Dos proporciones (2 − 1): estimador, EE de Wald, EE agrupado, z, p y IC 95 %."""
    a, b = x1 / n1, x2 / n2
    d = b - a
    se_w = np.sqrt(a * (1 - a) / n1 + b * (1 - b) / n2)
    pp = (x1 + x2) / (n1 + n2)
    se_p = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = d / se_p
    return dict(pa=a, pb=b, d=d, se_w=se_w, pool=pp, se_p=se_p, z=z, p=2 * stats.norm.sf(abs(z)), lo=d - 1.959964 * se_w, hi=d + 1.959964 * se_w)


def n_pg(p1, p2, alfa=0.05, pot=0.80):
    za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(pot)
    pb = (p1 + p2) / 2
    return ((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / abs(p1 - p2)) ** 2


def n_simple(p1, p2, alfa=0.05, pot=0.80):
    """Fórmula simplificada: n = (z_α/2 + z_β)² · [p1(1−p1) + p2(1−p2)] / (p1 − p2)²."""
    return (stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(pot)) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado):
    s.append(band(f"Clase 22 · {titulo}", FUENTE, C22))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C22, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3] if len(figuras) > 1 else conceptos, C22))
    s.append(Spacer(1, 1))
    dos = len(figuras) > 1
    for fig, w, pie in figuras:
        s.append(img(fig, width=CONTENT_W * (0.77 if dos else w * 0.92)))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py and not dos:
        left += [Spacer(1, 3), P("En Python", "h3"), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    if dos:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)] if py else []
        right = [P("Con los datos de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C22)]
        s.append(two_cols(left, right))
    else:
        right = [P("Con los datos de la clase", "h3"), bullets(salida, "cell", C22), Spacer(1, 3),
                 box([P("<b>Cuidado</b>", "cellb")] + [P(c, "cell") for c in cuidado], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
        s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 22 · Experimentación, A/B testing y causalidad aplicada",
                  "Profesora Maritza Márquez · de la predicción a la decisión · sigue a los GLM (Clases 20–21) y precede a la estadística bayesiana (Clases 23–24)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Un modelo predictivo responde «¿quién comprará?»; una decisión necesita otra pregunta: <b>«¿qué pasa si cambio algo?»</b>. Esa es una pregunta <b>causal</b> y no se contesta con una correlación. "
                    "El <b>A/B testing</b> la contesta <b>asignando al azar</b> quién recibe el cambio: así los dos grupos son comparables en promedio y la diferencia de resultados se atribuye al cambio. "
                    "Después hay que <b>diseñar</b> (hipótesis, métrica primaria, guardrails, tamaño de muestra), <b>vigilar</b> (SRM, peeking, comparaciones múltiples), <b>analizar</b> (diferencia de tasas, intervalo de confianza, p-valor, regresión, CUPED) y, "
                    "cuando no se puede aleatorizar, recurrir a <b>diferencias en diferencias</b>.", "body")], bg=C22_BG, border=C22, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c22_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Test de hipótesis, p-valor e intervalo de confianza</b> (Etapa 0 y Clases 17–19): aquí se usan de verdad y se explican con calma",
                     "<b>Regresión lineal</b> y su lectura de coeficientes (Clases 17–19): el efecto del tratamiento es un coeficiente",
                     "<b>Regresión logística y odds</b> (Clases 20–21): el modelo logit de la práctica",
                     "Proporciones, desviación estándar y la normal: ver recuadro de abajo"], "cell", C22)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Causalidad</b> → <b>anatomía del experimento</b> → <b>tamaño de muestra</b> → <b>chequeos (SRM, peeking)</b> → <b>efecto principal</b> → <b>covariables y CUPED</b> → <b>heterogeneidad</b> → <b>DiD</b>",
                      f"<b>ab_checkout</b>: 22 000 usuarios (control {mil(E['n0'])}, tratamiento {mil(E['n1'])}); <b>ab_srm</b>: {mil(SR['roto_n'])} usuarios con la asignación rota; <b>did_app_regiones</b>: 160 comunas × 6 meses (64 en la RM, tratadas)",
                      "Pregunta guía: ¿el nuevo checkout aumenta la conversión, y cómo sé que puedo creerle al experimento?"], "cell", C22)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con las clases vecinas</b>", "cellb"),
                  P("<b>C17–19 → C22</b>: el modelo lineal con una variable binaria T es el test de dos grupos: β₁ es la diferencia de medias. <b>C20–21 → C22</b>: con Y = 1 si convierte, el modelo lineal es el «modelo lineal de probabilidad» y la logística da los odds. "
                    "<b>C22 → C23–24</b>: la estadística bayesiana mira las mismas preguntas (¿cuánto mejora B?) con una probabilidad directa; el p-valor y el IC de hoy son la versión frecuentista.", "cell")], bg=SOFT, border=C22, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: p-valor e intervalo de confianza, sin fórmulas</b>", "cellb"),
                  P("<b>p-valor</b> = «si el cambio no hiciera nada (H₀), ¿qué tan raro sería ver una diferencia al menos así de grande solo por el azar del muestreo?». Un p-valor pequeño dice que <i>ese azar no alcanza</i> para explicar lo visto. "
                    "<b>No</b> es la probabilidad de que H₀ sea cierta, ni dice qué tan grande o importante es el efecto. "
                    "<b>Intervalo de confianza del 95 %</b> = rango de efectos que los datos «toleran»: si repitiéramos el experimento muchas veces, el 95 % de los intervalos así construidos contendría el efecto verdadero. "
                    "Si no contiene el 0, el p-valor es menor que 0.05; pero el IC además dice <b>cuánto</b> (y permite compararlo con lo que al negocio le importa).", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Causalidad y aleatorización",
        "Una asociación entre «cambio» y «mejora» <b>no prueba</b> que el cambio la causó: los usuarios expuestos pueden ser distintos de los no expuestos. "
        "Asignar <b>al azar</b> rompe esa asociación sistemática y deja grupos comparables <b>en promedio</b>.",
        [("Resultados potenciales", "Cada usuario tendría dos resultados: Yᵢ(1) con el tratamiento y Yᵢ(0) sin él. El efecto individual τᵢ = Yᵢ(1) − Yᵢ(0) <b>nunca se observa</b>: del otro resultado (el contrafactual) no hay dato."),
         ("Efecto promedio (ATE)", "ATE = E[Y(1) − Y(0)]. Con aleatorización se estima con la diferencia de promedios Ȳ<sub>T=1</sub> − Ȳ<sub>T=0</sub>: no hace falta que los grupos sean idénticos persona a persona, sino comparables en promedio."),
         ("Confusores", "Una variable Z (gasto previo, motivación) que influye en quién adopta el cambio y en el resultado crea una diferencia que <b>no</b> es causal. Aleatorizar elimina la flecha Z → T."),
         ("Predecir ≠ intervenir", "Un buen predictor de compra no dice qué cambiar. Predicción: ¿quién…? Causalidad: ¿qué ocurre si cambiamos…?")],
        [("c22_causal.png", 0.98, f"Izquierda: diagrama del confusor. Centro: en ab_checkout la conversión sube de {f(N['conv_por_quintil_gasto'][0] * 100, 1)} % a {f(N['conv_por_quintil_gasto'][4] * 100, 1)} % del quintil 1 al 5 de gasto previo. "
                                     f"Derecha: con una submuestra de «autoselección» (más gasto → más probable adoptar), la diferencia ingenua es {f(CA_['dif_ingenua'])} pp frente a {f(E['dif_pp'])} pp del experimento; ajustar por gasto_pre la baja a {f(CA_['ajustada_gasto'])} pp."),
         ("c22_balance.png", 0.98, f"Con aleatorización las covariables previas quedan equilibradas (máxima diferencia estandarizada {f(N['balance']['max_smd_aleatorio'], 3)}); en la submuestra con autoselección, el gasto previo difiere en {f(N['balance']['smd_gasto_autoseleccion'], 2)} desviaciones.")],
        ["τᵢ = Yᵢ(1) − Yᵢ(0)  (no observable)",
         "ATE = E[Y(1)] − E[Y(0)]",
         "ATE estimado = Ȳ<sub>B</sub> − Ȳ<sub>A</sub> (si hay azar)",
         "balance: d = (media<sub>T</sub> − media<sub>C</sub>) / sd"],
        None,
        [f"Control {f(E['p0'] * 100)} % frente a tratamiento {f(E['p1'] * 100)} %: ATE estimado {f(E['dif_pp'])} pp ({f(E['relativo'] * 100, 1)} % más, en relativo)",
         "El gasto previo medio es 50.2 (control) y 50.0 (tratamiento); sesiones previas, 3.96 y 3.98: equilibrados",
         f"La autoselección se construyó con los propios datos (submuestra de {mil(CA_['n'])} usuarios): es una ilustración, no un hallazgo de la clase"],
        ["Balancear lo <b>observado</b> es una buena señal, no una prueba: el azar equilibra también lo <b>no observado</b>, y de eso se trata aleatorizar."])

    topic_page(
        s, "Efecto principal: diferencia de tasas, IC y p-valor",
        "Con respuesta binaria (Yᵢ = 1 si convierte), el efecto es la <b>diferencia de tasas</b> p̂<sub>B</sub> − p̂<sub>A</sub>. El <b>intervalo de confianza</b> dice qué tan grande puede ser y el <b>p-valor</b> mide cuán compatible es lo visto con «sin efecto».",
        [("Estimador y error estándar", "p̂ = x/n en cada grupo; EE(p̂<sub>B</sub> − p̂<sub>A</sub>) = √[p̂<sub>A</sub>(1 − p̂<sub>A</sub>)/n<sub>A</sub> + p̂<sub>B</sub>(1 − p̂<sub>B</sub>)/n<sub>B</sub>]. Mide cuánto variaría la diferencia de un experimento a otro."),
         ("Intervalo de confianza", "diferencia ± 1.96 · EE (95 %). Se lee como el rango de efectos plausibles; si no contiene el 0 y supera el MDE, el efecto es estadística y prácticamente interesante."),
         ("Test z y p-valor", "z = diferencia / EE<sub>agrupado</sub> (bajo H₀ ambas tasas son iguales, así que se usa la tasa conjunta). El p-valor es la probabilidad, bajo H₀, de un |z| al menos tan grande."),
         ("Misma conclusión, tres rutas", "El test de proporciones, el modelo lineal (β₁ = diferencia) y la logística (β₁ = log-odds) llegan a la misma decisión; los números difieren un poco porque son modelos distintos.")],
        [("c22_tasas.png", 0.98, f"Izquierda y derecha: tasas con IC 95 %. Centro: la diferencia de {f(E['dif_pp'])} pp con IC [{f(E['ic_pp'][0])}; {f(E['ic_pp'][1])}] no contiene el 0 y supera el MDE de 1.5 pp."),
         ("c22_pvalor.png", 0.98, f"Izquierda: bajo H₀, z observado = {f(E['z'])} cae muy lejos. Centro: al barajar 20 000 veces quién es tratamiento, ninguna diferencia llega a {f(E['dif_pp'])} pp. Derecha: simulación (n = 3 000 por grupo): sin efecto, {f(N['p_bajo_H0_H1']['fp_H0'] * 100, 1)} % de p-valores bajo 0.05; con el efecto real, {f(N['p_bajo_H0_H1']['potencia_H1'] * 100, 0)} %.")],
        ["EE = √[ p̂<sub>A</sub>(1−p̂<sub>A</sub>)/n<sub>A</sub> + p̂<sub>B</sub>(1−p̂<sub>B</sub>)/n<sub>B</sub> ]",
         "IC 95 %: (p̂<sub>B</sub> − p̂<sub>A</sub>) ± 1.96 · EE",
         "z = (p̂<sub>B</sub> − p̂<sub>A</sub>) / √[ p̂(1−p̂)(1/n<sub>A</sub> + 1/n<sub>B</sub>) ],  p̂ = (x<sub>A</sub>+x<sub>B</sub>)/(n<sub>A</sub>+n<sub>B</sub>)"],
        ["proportions_ztest([x_B, x_A], [n_B, n_A])", "smf.ols('convierte ~ trat', ab).fit()", "smf.logit('convierte ~ trat', ab).fit()"],
        [f"Control {mil(E['x0'])}/{mil(E['n0'])} y tratamiento {mil(E['x1'])}/{mil(E['n1'])}: diferencia {f(E['dif_pp'], 3)} pp, EE {f(E['se_wald'] * 100, 3)} pp",
         f"IC 95 % [{f(E['ic_pp'][0])}; {f(E['ic_pp'][1])}] pp; z = {f(E['z'])}, χ² = {f(E['chi2'])}, p = {pv(E['p_valor'])}",
         f"Modelo lineal: β₀ = {f(M['lpm']['b0'] * 100)} %, β₁ = {f(M['lpm']['b1'] * 100)} pp, p = {pv(M['lpm']['p'])}. Logit: β₁ = {f(M['logit']['b1'], 4)}, odds ratio {f(M['logit']['odds_ratio'], 3)} [{f(M['logit']['ic_or'][0], 3)}; {f(M['logit']['ic_or'][1], 3)}], p = {pv(M['logit']['p'])}"],
        ["Un p-valor pequeño <b>no</b> dice que el efecto sea «real y relevante»: dice que el azar no lo explica. La relevancia se juzga con el IC y el MDE. Y p = 0.03 <b>no</b> significa «3 % de que H₀ sea cierta»."])

    topic_page(
        s, "Potencia, MDE y tamaño de muestra",
        "Antes de empezar se fija el <b>efecto mínimo detectable (MDE)</b>, el nivel de significancia α y la <b>potencia</b> 1 − β. De ahí sale cuántos usuarios hay que reunir. <b>Si el MDE se divide por 2, n se multiplica por ≈ 4.</b>",
        [("α y potencia", "α = probabilidad tolerada de falso positivo (decir «hay efecto» cuando no lo hay, error tipo I). Potencia = probabilidad de detectar un efecto real del tamaño del MDE (1 − β; β es el error tipo II)."),
         ("MDE", "Es el menor efecto que al negocio le importa detectar, no una predicción del efecto. Se acuerda <b>antes</b> de mirar resultados."),
         ("Tamaño por grupo", "n crece con la variabilidad p(1 − p), con el nivel de confianza y con la potencia; baja con el cuadrado del MDE. Es el n <b>por grupo</b>: el total es el doble."),
         ("Potencia ex post", "Con n fijo se puede calcular qué MDE alcanzaba el experimento: aquí, ≈ 11 000 por grupo detectan ≈ 1.1 pp con 80 % de potencia.")],
        [("c22_poder.png", 0.98, f"Izquierda: n por grupo según el MDE (línea base 9 %). Centro: potencia según n para tres efectos reales; con n = {mil(PO['n_redondeado'])} y un efecto de 1.0 pp la potencia es solo {f(PO['potencia_con_6138_y_1pp'] * 100, 0)} %. Derecha: el experimento real tenía potencia {f(PO['potencia_real_efecto_observado'] * 100, 1)} % para el efecto observado.")],
        ["n = [ z<sub>1−α/2</sub>·√(2p̄(1−p̄)) + z<sub>1−β</sub>·√(p₁(1−p₁)+p₂(1−p₂)) ]² / (p₂ − p₁)²",
         "z<sub>0.975</sub> = 1.960,  z<sub>0.80</sub> = 0.842,  z<sub>0.90</sub> = 1.282",
         "n total = 2 · n por grupo"],
        ["from statsmodels.stats.power import NormalIndPower",
         "h = proportion_effectsize(0.105, 0.09)",
         "NormalIndPower().solve_power(h, power=0.8, alpha=0.05)"],
        [f"p₁ = 0.09, p₂ = 0.105 (MDE 1.5 pp), α = 0.05, potencia 80 %: <b>{f(PO['n_power_prop_test'], 1)} → {mil(PO['n_redondeado'])} por grupo</b> ({mil(PO['n_total'])} en total) con la fórmula de <font name='Mono'>power.prop.test</font>",
         f"Otros MDE: 1.0 pp → {mil(PO['n_mde_1pp'])}; 2.0 pp → {mil(PO['n_mde_2pp'])}; 0.75 pp → {mil(PO['n_mde_075pp'])}. Con potencia 90 % y 1.5 pp: {mil(PO['n_pot90'])}",
         f"La diapositiva dice 6 131 (12 262 en total): es la aproximación del arcoseno (h de Cohen, {f(PO['n_arcoseno'], 1)}), no <font name='Mono'>power.prop.test</font>"],
        ["Tomar el efecto <b>observado</b> como MDE después del experimento (potencia «post hoc») no aporta información: el MDE se acuerda antes."])

    topic_page(
        s, "Errores frecuentes: SRM, peeking y comparaciones múltiples",
        "Un experimento bonito se puede destruir de tres formas: <b>asignación rota</b> (SRM), <b>mirar el p-valor antes de tiempo</b> (peeking) y <b>probar muchas cosas y mostrar la que salió</b>. Todo se define <b>antes</b> de lanzar.",
        [("SRM (Sample Ratio Mismatch)", "La proporción observada de usuarios por grupo no coincide con la planeada (p. ej. 50/50). Se prueba con chi-cuadrado sobre los conteos: χ² = Σ (obs − esp)²/esp. Regla práctica: p &lt; 0.001 → no se reporta el efecto, se audita la instrumentación."),
         ("Peeking", "Mirar el p-valor varias veces y parar cuando baja de 0.05. Cada mirada es otra oportunidad de falso positivo: el error real supera el 5 %."),
         ("Comparaciones múltiples", "Con m pruebas independientes al 5 %, P(al menos un falso positivo) = 1 − 0.95<super>m</super>. Con 20 métricas, 64 %. Se corrige (Bonferroni: α/m) o se preespecifica la métrica primaria."),
         ("Disciplina mínima", "Métrica primaria, segmentos principales, guardrails, duración y plan de análisis se escriben <b>antes</b>. Lo exploratorio se etiqueta como exploratorio.")],
        [("c22_srm.png", 0.98, f"ab_checkout: {mil(SR['checkout']['obs'][0])} frente a {mil(SR['checkout']['obs'][1])} (esperado 11 000), χ² = {f(SR['checkout']['chi2'])}, p = {f(SR['checkout']['p'])}. ab_srm: {mil(SR['roto']['obs'][0])} frente a {mil(SR['roto']['obs'][1])} (esperado {mil(SR['roto']['esperado'])}), χ² = {f(SR['roto']['chi2'], 1)}, p = {pv(SR['roto']['p'])}; el desbalance está en móvil."),
         ("c22_peeking.png", 0.98, f"Simulación de experimentos sin efecto (tasa 8.88 %, 12 000 por grupo, {mil(PK['sims'])} repeticiones). Con 10 miradas el falso positivo llega a {f(PK['fp_10_miradas_sim'] * 100, 1)} %; mirando cada 100 usuarios, {f(PK['continuo_cada_100'] * 100, 0)} %.")],
        ["χ² = Σ (observado − esperado)² / esperado,  gl = k − 1",
         "P(≥ 1 falso positivo) = 1 − (1 − α)<super>m</super>",
         "Bonferroni: usar α/m en cada prueba"],
        None,
        [f"ab_srm: χ² = {f(SR['roto']['chi2'], 1)} con 1 gl, p = {pv(SR['roto']['p'])}: se audita. Por dispositivo, desktop p = {f(SR['roto_por_dispositivo']['desktop']['p'], 2)} y móvil p = {pv(SR['roto_por_dispositivo']['movil']['p'])}",
         f"20 métricas sin corregir: {f(N['multiples']['m20'] * 100, 0)} % de falso positivo; con Bonferroni, {f(N['multiples']['bonf_m20'] * 100, 1)} %"],
        ["Un SRM no significativo (p = 0.64) <b>no demuestra</b> aleatorización perfecta. Con ab_srm no se reporta la conversión ingenua (12.5 % y 12.7 %)."])

    topic_page(
        s, "Covariables, CUPED y heterogeneidad",
        "En un experimento bien aleatorizado, ajustar por covariables <b>previas</b> no corrige sesgo: <b>gana precisión</b>. CUPED hace lo mismo restando la parte de Y que predice el comportamiento previo. La <b>heterogeneidad</b> se mira con cautela.",
        [("Covariables", "Yᵢ = β₀ + β₁Tᵢ + γXᵢ + εᵢ con Xᵢ medido <b>antes</b> del tratamiento. Nunca se ajusta por variables que el tratamiento afecta (estarían «después»). β₁ sigue siendo el efecto promedio."),
         ("CUPED", "Y<sup>cuped</sup> = Y − θ(X − X̄), con θ = Cov(Y, X)/Var(X). Misma media de efecto, menos varianza: la reducción es ≈ ρ², con ρ = Cor(Y, X). No arregla un SRM ni una mala aleatorización."),
         ("Heterogeneidad", "Yᵢ = β₀ + β₁Tᵢ + β₂Dᵢ + β₃(TᵢDᵢ) + εᵢ. β₃ es la <b>diferencia de efecto</b> entre segmentos. Si no estaba preespecificada, es exploratoria."),
         ("Ausencia de evidencia", "Un IC de interacción que incluye el 0 y es ancho no prueba que el efecto sea igual: solo que los datos no alcanzan para distinguirlo.")],
        [("c22_cuped.png", 0.98, f"CUPED con gasto_pre (ρ = {f(M['cuped']['rho'], 3)}): reduce la varianza en {f(M['cuped']['reduccion_var'] * 100, 1)} %, igual que ρ² = {f(M['cuped']['rho2'] * 100, 1)} %; el IC se estrecha un {f((1 - M['cuped']['ancho_ic_ratio']) * 100, 1)} %."),
         ("c22_hetero.png", 0.98, "Efecto por dispositivo y por región: todos parecidos al promedio; los intervalos de tablet, Valparaíso y Biobío incluyen el 0 por tener pocos usuarios, no porque el efecto desaparezca.")],
        ["θ = Cov(Y, X) / Var(X)",
         "Y<sup>cuped</sup> = Y − θ(X − X̄)",
         "reducción de varianza ≈ ρ²",
         "interacción: β₃ = efecto<sub>móvil</sub> − efecto<sub>desktop</sub>"],
        ["smf.ols('convierte ~ trat + gasto_pre + sesiones_pre + C(dispositivo)', ab)", "theta = np.cov(y, x)[0, 1] / x.var()", "smf.ols('convierte ~ trat * C(dispositivo)', ab)"],
        [f"Sin ajuste {f(M['lpm']['b1'] * 100)} pp (EE {f(M['lpm']['se'] * 100, 3)}); con covariables {f(M['ajustado']['b1'] * 100)} pp (EE {f(M['ajustado']['se'] * 100, 3)}); CUPED {f(M['cuped']['b1'] * 100)} pp (EE {f(M['cuped']['se'] * 100, 3)}), θ = {f(M['cuped']['theta'], 4)}",
         f"Efecto por dispositivo: desktop {f(HT['subgrupos']['dispositivo: desktop']['efecto'])}, móvil {f(HT['subgrupos']['dispositivo: movil']['efecto'])}, tablet {f(HT['subgrupos']['dispositivo: tablet']['efecto'])} pp",
         f"Interacciones: móvil {f(HT['modelo']['int_movil'])} pp (p = {f(HT['modelo']['p_movil'])}), tablet {f(HT['modelo']['int_tablet'])} pp (p = {f(HT['modelo']['p_tablet'])}); prueba conjunta p = {f(HT['modelo']['p_conjunto'])}"],
        ["Mirando 7 subgrupos, cuatro salen «significativos» y tres no, aunque el efecto es casi el mismo: la significancia depende de n. Compara <b>efectos</b>, no solo p-valores."])

    topic_page(
        s, "Cuando no se puede aleatorizar: diferencias en diferencias",
        "Si el cambio no se puede asignar al azar (campañas regionales, políticas), se compara cuánto cambió el grupo tratado con cuánto cambió un grupo de control en el mismo período. Es creíble solo si, sin el cambio, <b>ambos habrían evolucionado en paralelo</b>.",
        [("Idea", "DiD = (Ȳ<sub>T,post</sub> − Ȳ<sub>T,pre</sub>) − (Ȳ<sub>C,post</sub> − Ȳ<sub>C,pre</sub>). La resta del control quita lo que habría pasado de todos modos (la tendencia)."),
         ("Regresión", "Y<sub>it</sub> = β₀ + β₁Trat<sub>i</sub> + β₂Post<sub>t</sub> + β₃(Trat<sub>i</sub> × Post<sub>t</sub>) + u<sub>it</sub>. El efecto causal es β₃ y coincide con el DiD de las medias."),
         ("Tendencias paralelas", "Es el supuesto clave y no se prueba, solo se respalda: gráfico antes del lanzamiento, estudio de eventos y coeficientes previos cercanos a cero."),
         ("Errores estándar", "Cada comuna aparece 6 veces: los errores deben agruparse por comuna (cluster). Aquí el modelo simple del script es, si acaso, conservador.")],
        [("c22_did.png", 0.98, f"Izquierda: uso medio de la app; la línea punteada es la RM «sin lanzamiento» (la brecha previa se mantiene). Centro: la brecha respecto de marzo es ≈ 0 antes y ≈ 6 después. Derecha: DiD = {f(DD['mco']['b_int'], 2)}.")],
        ["DiD = (63.37 − 52.48) − (54.78 − 49.64) = 5.76",
         "uso ~ trat * post:  β₃ = coeficiente de trat:post",
         "efectos fijos: uso ~ trat:post + C(comuna) + C(mes)"],
        ["smf.ols('uso_app ~ trat * post', did).fit(cov_type='cluster', cov_kwds={'groups': comuna})"],
        [f"Modelo simple: β₃ = {f(DD['mco']['b_int'])}, IC 95 % [{f(DD['mco']['ic'][0])}; {f(DD['mco']['ic'][1])}], p = {pv(DD['mco']['p'])} (como la diapositiva 30)",
         f"Con errores agrupados por comuna: EE {f(DD['cluster']['se'], 3)} frente a {f(DD['mco']['se'], 3)}; IC [{f(DD['cluster']['ic'][0])}; {f(DD['cluster']['ic'][1])}]. Con efectos fijos de comuna y mes: {f(DD['fe']['b'])}, EE {f(DD['fe']['se'], 3)}",
         f"Pre-tendencia: brecha RM − otras de {f(DD['brecha_mensual']['2026-01'])}, {f(DD['brecha_mensual']['2026-02'])} y {f(DD['brecha_mensual']['2026-03'])} antes; {f(DD['brecha_mensual']['2026-04'])}, {f(DD['brecha_mensual']['2026-05'])} y {f(DD['brecha_mensual']['2026-06'])} después. Interacción con el tiempo: {f(DD['pretendencia']['coef'], 2)} (p = {f(DD['pretendencia']['p'], 2)})"],
        ["Un buen gráfico previo <b>respalda</b> pero no <b>prueba</b> las tendencias paralelas. Y la RM pudo recibir otras cosas al mismo tiempo (campañas, feriados): el DiD solo identifica el efecto si nada más cambió solo en la RM."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets(["<b>«El p-valor es bajito, entonces el efecto parece real y relevante»</b>: el p-valor solo mide compatibilidad con H₀; la relevancia se ve en el IC frente al MDE.",
                           "<b>«Si el p-valor del SRM es grande, el experimento está completamente aleatorizado»</b>: un p grande no prueba aleatorización; solo no hay evidencia de desbalance en los conteos.",
                           "<b>«La idea es que las proporciones sean distintas»</b>: el test es neutro; no se busca rechazar H₀. Buscarlo es la raíz del peeking y de las comparaciones múltiples.",
                           "<b>«6131 usuarios por grupo» con <font name='Mono'>power.prop.test</font></b>: esa función da 6 137.95 → 6 138 (12 276 en total). 6 131 es la aproximación del arcoseno.",
                           "<b>«P2 = 0.15»</b> (dicho al explicar los parámetros): el código usa p₂ = 0.105 (MDE de 1.5 pp). Con 0.15 harían falta solo 459 por grupo."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas y scripts</b>", "cellb"),
                  bullets([f"<b>Diapositiva 21 (peeking)</b>: «con 10 miradas independientes al 5 %, el falso positivo se acerca a 40 %». 1 − 0.95<super>10</super> = {f(PK['fp_10_miradas_indep'] * 100, 1)} % solo vale si las miradas fueran independientes; con datos que se acumulan <b>no</b> lo son. Simulado: {f(PK['fp_10_miradas_sim'] * 100, 1)} % con 10 miradas; ≈ 40 % exige mirar de forma casi continua.",
                           f"<b>Diapositiva 30 y script (DiD)</b>: <font name='Mono'>lm(uso_app ~ trat * post)</font> ignora que cada comuna se repite 6 veces. El IC [{f(DD['mco']['ic'][0])}; {f(DD['mco']['ic'][1])}] es más ancho que con errores agrupados [{f(DD['cluster']['ic'][0])}; {f(DD['cluster']['ic'][1])}]; el punto estimado (5.76) no cambia.",
                           "<b>Diapositiva 29</b>: «si antes ya corrían en direcciones distintas, el DiD queda a medio camino» es impreciso: el DiD queda <b>sesgado</b> en la magnitud de la diferencia de tendencias, sin un «medio camino» definido.",
                           "<b>Diapositiva 15 (regresión = test)</b>: coinciden las conclusiones y el efecto del modelo lineal, pero no los números exactos (EE del test con tasa conjunta; la logística da odds ratio, no diferencia). El modelo lineal debería usar errores robustos (aquí son idénticos a 4 decimales).",
                           "<b>Diapositivas 5 y 6</b>: «estima lo ocurrido si ¿cambiamos…?» y «27 %:,» son erratas de redacción. Las cifras históricas (1747, 1948, Obama 2008, Bing 12 %) no se verificaron aquí."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper confundió Wardrails (guardrails), picking (peeking), Coupet / Cuper (CUPED), DIT (DiD), «Yesu-i» (Y sub i), «Tao Tzu Y» (τᵢ), ATG (ATE), VIM (Bing), SAS (SaaS), «Valleciana» (Bayesiana), «cita» (srm_test). "
                    "El recreo (≈ 1:21:15 del video) no está en el video; la corrección de este material dejó las frases dudosas como las dijo la profesora.", "cell")], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist y criterio de dominio</b>", "cellb"),
                  bullets(["Distingo una pregunta predictiva de una causal y explico por qué aleatorizar",
                           "Calculo a mano la diferencia de tasas, su IC y el z, y explico qué dice y qué no dice el p-valor",
                           "Calculo n por grupo con la fórmula y explico cómo cambia con el MDE, α y la potencia",
                           "Hago un chi-cuadrado de SRM y sé qué hacer si falla; explico por qué no se «mira» el p-valor antes de tiempo",
                           "Estimo un DiD con una tabla 2 × 2 y discuto las tendencias paralelas",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos"], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C22_Guia_ABTesting.pdf", s, "Clase 22 · Experimentación, A/B testing y causalidad aplicada")


CA_ = N["autoseleccion"]


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#0D9488">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 22 · Experimentación, A/B testing y causalidad aplicada", "Para leer antes de los ejercicios · lectura de ~35 min", C22))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Predecir no es intervenir", "Una pregunta causal pide qué pasaría con el cambio; la correlación no la responde."),
                            ("El azar hace comparables los grupos", "El efecto se estima con la diferencia de promedios; el confusor deja de importar."),
                            ("El p-valor y el IC se leen juntos", "p: ¿es compatible con «sin efecto»?; IC: ¿de qué tamaño puede ser el efecto?"),
                            ("El tamaño se calcula antes", "n depende de α, potencia y MDE; mitad de MDE ≈ cuádruple de n."),
                            ("Antes de creer el efecto, chequeo el diseño", "SRM, peeking y comparaciones múltiples pueden invalidarlo."),
                            ("Sin azar, DiD con supuestos", "Efecto = cambio del tratado − cambio del control; exige tendencias paralelas.")], C22)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C22_ABTesting.ipynb</font> o el cuaderno <font name='Mono'>C22_Ejercicios_Cuaderno_ABTesting.pdf</font>.", "small"))

    s.append(h("1", "Pregunta predictiva o causal"))
    s += para("Una pregunta es <b>predictiva</b> si pide anticipar un resultado con la información disponible («¿quién abandonará la app?»). Es <b>causal</b> si pide el efecto de <i>cambiar</i> algo («¿qué pasa con el abandono si ofrecemos 14 días de prueba en vez de 7?»). "
              "Para la segunda necesitamos el contrafactual: lo que habría ocurrido <b>sin</b> el cambio. Como nunca lo observamos en la misma persona, lo reemplazamos por un grupo control obtenido por <b>azar</b>.")
    s.append(ejemplo("clasificar tres preguntas", [
        pasos([["Pregunta", "Tipo", "Por qué"],
               ["¿Qué clientes tienen mayor riesgo de no pagar el mes próximo?", "predictiva", "anticipa con datos actuales; no cambia nada"],
               ["¿Cuántas ventas más tendríamos si rebajamos el envío a 0?", "causal", "compara dos mundos: con y sin la rebaja"],
               ["¿Los usuarios con más sesiones previas compran más?", "asociación", "describe una relación; no dice que sesiones causen compras"]],
              [78 * mm, 24 * mm, CONTENT_W - 102 * mm - 18]),
        P("Regla práctica: si la respuesta sirve para <i>decidir qué hacer</i> (y no solo para saber qué pasará), hay que pensar en causalidad y diseñar un experimento.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno", [P("Simula en A1 los dos resultados potenciales y comprueba que la diferencia de promedios del experimento recupera el ATE.", "cell")]))

    s.append(h("2", "Diferencia de tasas, IC y p-valor, paso a paso"))
    s += para("Idea: la tasa observada de cada grupo es una estimación con ruido. El <b>error estándar</b> (EE) cuantifica ese ruido. La diferencia dividida por su EE (el <b>z</b>) cuenta «cuántos ruidos» se aleja del 0. "
              "Si |z| &gt; 1.96, el azar solo explicaría esa diferencia menos del 5 % de las veces (p &lt; 0.05). El <b>IC</b> es diferencia ± 1.96·EE: el rango de efectos que los datos toleran.")
    ex = z2(300, 5000, 365, 5000)
    s.append(ejemplo("A: 300/5 000 · B: 365/5 000", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Tasas y diferencia", "300/5 000 = 0.0600 · 365/5 000 = 0.0730", f"diferencia = {f(ex['d'] * 100)} pp"],
               ["Tasa conjunta", "(300 + 365)/10 000", f"p̂ = {f(ex['pool'], 4)}"],
               ["EE agrupado (para el test)", "√[0.0665·0.9335·(1/5 000 + 1/5 000)]", f"{f(ex['se_p'], 5)}"],
               ["z y p-valor", f"{f(ex['d'], 4)} / {f(ex['se_p'], 5)}", f"z = {f(ex['z'])} · p = {f(ex['p'], 4)}"],
               ["EE de Wald (para el IC)", "√[0.06·0.94/5 000 + 0.073·0.927/5 000]", f"{f(ex['se_w'], 5)}"],
               ["IC 95 %", f"{f(ex['d'] * 100)} ± 1.96·{f(ex['se_w'] * 100, 3)}", f"[{f(ex['lo'] * 100)}; {f(ex['hi'] * 100)}] pp"]],
              [44 * mm, 70 * mm, CONTENT_W - 114 * mm - 18]),
        P(f"Lectura: el efecto observado es +{f(ex['d'] * 100)} pp y el IC no contiene el 0 (p = {f(ex['p'], 3)} &lt; 0.05). Pero el IC llega casi a {f(ex['lo'] * 100, 1)} pp: el efecto podría ser pequeño. "
          "<b>Error común</b>: decir «hay 97 % de probabilidad de que B sea mejor» (eso es bayesiano y no es el p-valor).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c22_tasas.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(img("c22_pvalor.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B2) y los Bloques 2, 3 y 7 del cuaderno", [P("Calcula el z a mano, compáralo con <font name='Mono'>proportions_ztest</font> y con el modelo lineal, y haz el test de permutación para <i>ver</i> qué es un p-valor.", "cell")]))

    s.append(h("3", "Potencia, MDE y tamaño de muestra"))
    s += para("Hay dos errores posibles: decir «hay efecto» cuando no lo hay (<b>falso positivo</b>, probabilidad α) y no ver un efecto que sí existe (<b>falso negativo</b>, β). La <b>potencia</b> 1 − β es la probabilidad de detectar un efecto del tamaño del MDE. "
              "Se fijan α (0.05), la potencia (0.80 o 0.90) y el MDE, y se despeja n: crece con la variabilidad y baja con el cuadrado del MDE.")
    n1_ = n_pg(0.12, 0.14)
    n2_ = n_pg(0.12, 0.13)
    n3_ = n_pg(0.12, 0.14, pot=0.90)
    s.append(ejemplo("línea base 12 %, queremos detectar +2 pp", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Constantes (α = 0.05, 80 %)", "z₀.₉₇₅ = 1.960 · z₀.₈₀ = 0.842", "(1.960 + 0.842)² = 7.85"],
               ["Varianzas", "0.12·0.88 + 0.14·0.86", f"{f(0.12 * 0.88 + 0.14 * 0.86, 4)}"],
               ["Fórmula simplificada", f"7.85 · {f(0.12 * 0.88 + 0.14 * 0.86, 4)} / 0.02²", f"n ≈ {mil(n_simple(0.12, 0.14))} por grupo"],
               ["Fórmula de power.prop.test", "con p̄ = 0.13", f"n = {f(n1_, 1)} → {mil(np.ceil(n1_))}"],
               ["Mitad del MDE (+1 pp)", "p₂ = 0.13", f"{mil(np.ceil(n2_))} por grupo (≈ ×4)"],
               ["Potencia 90 %", "z₀.₉₀ = 1.282", f"{mil(np.ceil(n3_))} por grupo"]],
              [48 * mm, 62 * mm, CONTENT_W - 110 * mm - 18]),
        P("La fórmula simplificada es una aproximación; <font name='Mono'>power.prop.test</font> usa la varianza exacta bajo H₁ y da un valor algo mayor. Siempre se redondea <b>hacia arriba</b>, y el total es el doble.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c22_poder.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A3) y el Bloque 4 del cuaderno", [P("Reproduce 6 138 con la fórmula, descubre de dónde sale 6 131 y dibuja la curva n frente al MDE.", "cell")]))

    s.append(h("4", "SRM y chequeos del diseño"))
    s += para("Si el experimento asigna 50/50, los conteos deberían parecerse a 50/50 salvo por azar. El <b>chi-cuadrado</b> compara lo observado con lo esperado: χ² = Σ (obs − esp)²/esp, con (grupos − 1) grados de libertad. "
              "Con 2 grupos: χ² &gt; 3.84 ↔ p &lt; 0.05; χ² &gt; 10.83 ↔ p &lt; 0.001 (umbral usual para SRM).")
    c1 = stats.chisquare([5300, 4700])
    c2 = stats.chisquare([5100, 4900])
    s.append(ejemplo("dos experimentos 50/50 con 10 000 usuarios", [
        pasos([["Caso", "Cálculo", "Resultado"],
               ["Observado 5 300 / 4 700 (esperado 5 000)", "(300²/5 000) + (300²/5 000) = 18 + 18", f"χ² = {f(c1.statistic, 1)} · p = {pv(c1.pvalue)}"],
               ["Observado 5 100 / 4 900", "(100²/5 000) · 2 = 2 + 2", f"χ² = {f(c2.statistic, 1)} · p = {f(c2.pvalue, 3)}"]],
              [60 * mm, 62 * mm, CONTENT_W - 122 * mm - 18]),
        P(f"El primero es un SRM claro (p ≪ 0.001): se audita y no se reporta el efecto. El segundo tiene p = {f(c2.pvalue, 3)}: bajo 0.05 pero sobre 0.001, así que con la regla práctica no se declara SRM; "
          "conviene igual revisar. <b>Peeking</b>: mirar el p-valor 10 veces no da 5 % sino ≈ 19 % de falsos positivos (simulado), y con 20 métricas sin corregir, 64 %.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c22_srm.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(img("c22_peeking.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C1, C3, C4) y el Bloque 5 del cuaderno", [P("Haz el SRM de ab_checkout y de ab_srm, cruza por dispositivo para localizar el problema y simula el peeking tú mismo.", "cell")]))

    s.append(h("5", "Regresión, CUPED y DiD"))
    s += para("<b>Regresión</b>: con T = 0/1, β₁ del modelo lineal es la diferencia de tasas; agregar covariables previas reduce el error estándar pero no corrige un experimento mal hecho. "
              "<b>CUPED</b> resta θ(X − X̄) a Y y reduce la varianza en ≈ ρ². <b>DiD</b>: efecto = cambio del tratado − cambio del control, bajo tendencias paralelas.")
    s.append(ejemplo("tabla 2 × 2 de medias (uso diario en minutos)", [
        pasos([["Grupo", "Antes", "Después", "Cambio"],
               ["Tratado", "40", "52", "+12"],
               ["Control", "38", "44", "+6"],
               ["DiD", "", "", "12 − 6 = +6"]], [44 * mm, 36 * mm, 36 * mm, CONTENT_W - 116 * mm - 18]),
        P("Con CUPED: si ρ = 0.5 entre Y y la covariable previa, la varianza baja ≈ 0.5² = 25 % y el EE a √0.75 = 87 % del original. "
          "El 6 es el efecto <i>si</i> el tratado habría subido 6 como el control (tendencias paralelas); comparar solo 52 − 40 = 12 mezclaría efecto y tendencia.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c22_cuped.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(img("c22_did.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(pista("las Partes C y D del notebook y los Bloques 6 y 7 del cuaderno", [P("Estima el DiD con la tabla, con <font name='Mono'>trat * post</font> y con efectos fijos, y agrupa los errores por comuna.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["La causalidad sale del <b>diseño</b>, no del algoritmo: «buen experimento primero, modelo elegante después».",
                       "R ↔ Python: <font name='Mono'>prop.test</font> ↔ <font name='Mono'>proportions_ztest</font>; <font name='Mono'>power.prop.test</font> ↔ <font name='Mono'>NormalIndPower</font>; <font name='Mono'>chisq.test</font> ↔ <font name='Mono'>scipy.stats.chisquare</font>; <font name='Mono'>lm</font> ↔ <font name='Mono'>smf.ols</font>; <font name='Mono'>glm(binomial)</font> ↔ <font name='Mono'>smf.logit</font>.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    build("apuntes/C22_Apuntes_ABTesting.pdf", s, "Apuntes · Clase 22 · Experimentación, A/B testing y causalidad aplicada")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0D9488">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C22))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 22 · A/B testing y causalidad aplicada", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C22))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: EE = √[p̂<sub>A</sub>(1−p̂<sub>A</sub>)/n<sub>A</sub> + p̂<sub>B</sub>(1−p̂<sub>B</sub>)/n<sub>B</sub>] · IC = d ± 1.96·EE · z = d/√[p̂(1−p̂)(1/n<sub>A</sub>+1/n<sub>B</sub>)] · "
                           "n = (z<sub>α/2</sub> + z<sub>β</sub>)²·[p₁(1−p₁) + p₂(1−p₂)]/(p₂−p₁)² · χ² = Σ(obs−esp)²/esp · DiD = Δ<sub>tratado</sub> − Δ<sub>control</sub>. Constantes: z₀.₉₇₅ = 1.96, z₀.₈₀ = 0.842, z₀.₉₀ = 1.282; χ²(1 gl) crítico: 3.84 (5 %) y 10.83 (0.1 %).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C22)], bg=SOFT, border=C22, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "Pregunta predictiva o causal", "★", "Clasifica cada pregunta como predictiva, causal o de asociación y explica en una frase.", [
        "«¿Qué usuarios cancelarán su suscripción el próximo mes?»",
        "«¿Aumentan las compras si mostramos las reseñas arriba de la página?»",
        "«¿Los clientes que usan la app compran más que los que no la usan?»",
        "«Si la tienda A tuvo más ventas que la B después de una promoción, ¿la promoción fue la causa?» ¿Qué faltaría para decirlo?"])
    a, b = z2(320, 4000, 372, 4000), z2(275, 2500, 325, 2500)
    s += bloque(2, "Test de dos proporciones a mano", "★★★", "Un experimento 50/50: control 320 compras de 4 000 usuarios; tratamiento 372 compras de 4 000.", [
        "Calcula las dos tasas y la diferencia (en puntos porcentuales).",
        "Calcula la tasa conjunta p̂, el error estándar agrupado y el estadístico z.",
        "Con α = 0.05 (|z| crítico = 1.96), ¿rechazas H₀: p<sub>B</sub> = p<sub>A</sub>? Calcula el p-valor aproximado con la tabla normal o una calculadora.",
        "Redacta la conclusión en una frase para un gerente (sin decir «H₀»)."])
    s += bloque(3, "Intervalo de confianza para la diferencia", "★★", "Otro experimento: A 275 conversiones de 2 500 usuarios; B 325 de 2 500.", [
        "Calcula las tasas, la diferencia y el error estándar de Wald.",
        "Construye el IC del 95 % para la diferencia. ¿Contiene el 0?",
        "El negocio solo lanzará el cambio si el efecto es al menos +1 pp. ¿Qué dice el IC?",
        "Explica con tus palabras qué significa «95 % de confianza» (y una cosa que NO significa)."])
    s += bloque(4, "Tamaño de muestra por grupo", "★★★", "Línea base de conversión 20 %; se quiere detectar 23 % (MDE = 3 pp) con α = 0.05.", [
        "Calcula con la fórmula simplificada n por grupo para potencia 80 % y para potencia 90 %.",
        "¿Cuántos usuarios hacen falta en total para potencia 80 %?",
        "Si el MDE fuera la mitad (1.5 pp), ¿cuánto vale n con potencia 80 %? ¿Cuántas veces mayor es?",
        "El experimento solo puede reunir 1 500 usuarios por grupo. ¿Es suficiente para detectar 3 pp con 80 %? ¿Qué decisión tomarías?"])
    s += bloque(5, "SRM con chi-cuadrado", "★★", "Diseño 50/50. Observado: 4 820 usuarios en control y 5 180 en tratamiento (total 10 000).", [
        "Calcula los conteos esperados y los residuos (observado − esperado).",
        "Calcula χ² y compáralo con 3.84 y con 10.83.",
        "Según la regla práctica de la clase (p < 0.001), ¿se reporta el efecto de conversión? ¿Qué harías primero?",
        "Un mes después, el SRM da p = 0.64. ¿Prueba eso que la aleatorización fue perfecta? Justifica."])
    s += bloque(6, "Diferencias en diferencias", "★★", "Tabla de medias de ventas semanales (en millones) en una cadena: tiendas con la nueva campaña (tratadas) y sin ella (control), antes y después.", [
        "Tratadas: antes 31, después 45. Control: antes 33, después 39. Calcula el cambio de cada grupo y el DiD.",
        "¿Por qué no basta comparar «después» entre tratadas y control (45 frente a 39)? ¿Y «antes y después» solo de las tratadas?",
        "Escribe el modelo de regresión y señala qué coeficiente es el DiD.",
        "Nombra el supuesto clave y una evidencia que lo respaldaría."])
    s += bloque(7, "Interpretar un p-valor", "★★", "En un A/B se obtiene una diferencia de +0.4 pp con p = 0.03 y n enorme (500 000 por grupo).", [
        "¿Verdadero o falso? «Hay un 3 % de probabilidad de que el tratamiento no tenga efecto.»",
        "¿Verdadero o falso? «Si el tratamiento no hiciera nada, ver una diferencia así o mayor ocurriría ≈ 3 % de las veces.»",
        "¿Verdadero o falso? «Como p < 0.05, el efecto es grande e importante.» Explica qué mirarías.",
        "Con 20 métricas probadas y solo una con p = 0.03, ¿qué problema hay y cómo lo corregirías?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", GREEN_C))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=GREEN_C), Spacer(1, 2)]

    s += sol("Bloque 1", [
        ["1", "Predictiva: anticipa un resultado con datos actuales; no pide el efecto de cambiar algo."],
        ["2", "Causal: pide el efecto de <i>mover</i> la página (comparar con y sin reseñas arriba)."],
        ["3", "Asociación: compara grupos que se autoseleccionaron; quienes usan la app pueden ser distintos (confusor)."],
        ["4", "No necesariamente: las tiendas pudieron diferir antes. Faltaría un control comparable y el cambio antes-después (DiD) o, mejor, asignación aleatoria de la promoción."]])
    s += sol("Bloque 2", [
        ["1", f"A: 320/4 000 = 8.00 %. B: 372/4 000 = 9.30 %. Diferencia = +{f(a['d'] * 100)} pp."],
        ["2", f"p̂ = (320 + 372)/8 000 = {f(a['pool'], 4)}. EE = √[{f(a['pool'], 4)}·{f(1 - a['pool'], 4)}·(2/4 000)] = {f(a['se_p'], 5)}. z = {f(a['d'], 4)}/{f(a['se_p'], 5)} = {f(a['z'])}."],
        ["3", f"|z| = {f(abs(a['z']))} {'>' if abs(a['z']) > 1.96 else '<'} 1.96: {'se rechaza H₀' if abs(a['z']) > 1.96 else 'no se rechaza H₀'}. p ≈ {f(a['p'], 3)}."],
        ["4", f"«El nuevo diseño subió la conversión de 8.0 % a 9.3 % (+{f(a['d'] * 100, 1)} pp) y esa diferencia es poco compatible con el puro azar (p ≈ {f(a['p'], 2)}); falta ver cuánto puede valer el efecto (su intervalo) y los guardrails.»"]])
    s += sol("Bloque 3", [
        ["1", f"A: 275/2 500 = 11.0 %. B: 325/2 500 = 13.0 %. d = +{f(b['d'] * 100)} pp. EE = √[0.11·0.89/2 500 + 0.13·0.87/2 500] = {f(b['se_w'], 5)}."],
        ["2", f"IC = {f(b['d'] * 100)} ± 1.96·{f(b['se_w'] * 100, 3)} = [{f(b['lo'] * 100)}; {f(b['hi'] * 100)}] pp. No contiene el 0."],
        ["3", f"El IC incluye valores bajo 1 pp (desde {f(b['lo'] * 100, 2)}): el efecto es positivo, pero con estos datos no se puede asegurar que alcance el 1 pp exigido. Se podría extender el experimento o decidir con el costo de equivocarse."],
        ["4", "Si repitiéramos el experimento muchas veces, el 95 % de los intervalos construidos así contendría el efecto verdadero. NO significa «95 % de probabilidad de que el efecto esté en este intervalo en particular»."]])
    n80, n90 = n_simple(0.20, 0.23), n_simple(0.20, 0.23, pot=0.90)
    n_h = n_simple(0.20, 0.215)
    pot_1500 = stats.norm.cdf((0.03 * np.sqrt(1500) - 1.96 * np.sqrt(2 * 0.215 * 0.785)) / np.sqrt(0.2 * 0.8 + 0.23 * 0.77))
    s += sol("Bloque 4", [
        ["1", f"Varianzas: 0.20·0.80 + 0.23·0.77 = {f(0.2 * 0.8 + 0.23 * 0.77, 4)}. Potencia 80 %: (1.96 + 0.842)² = 7.85 → n = 7.85·{f(0.2 * 0.8 + 0.23 * 0.77, 4)}/0.03² = {mil(np.ceil(n80))} por grupo. Potencia 90 %: (1.96 + 1.282)² = 10.51 → {mil(np.ceil(n90))}."],
        ["2", f"2 · {mil(np.ceil(n80))} = {mil(2 * np.ceil(n80))} usuarios en total."],
        ["3", f"MDE 1.5 pp: varianzas 0.20·0.80 + 0.215·0.785 = {f(0.2 * 0.8 + 0.215 * 0.785, 4)} → n = 7.85·{f(0.2 * 0.8 + 0.215 * 0.785, 4)}/0.015² = {mil(np.ceil(n_h))}: ≈ {f(n_h / n80, 1)} veces mayor (≈ 4)."],
        ["4", f"No: 1 500 < {mil(np.ceil(n80))}; la potencia sería ≈ {f(pot_1500 * 100, 0)} %. Opciones: aceptar un MDE mayor, juntar más tráfico/tiempo o no experimentar (un resultado no significativo sería poco informativo)."]])
    chi5 = stats.chisquare([4820, 5180])
    s += sol("Bloque 5", [
        ["1", "Esperados: 5 000 y 5 000. Residuos: 4 820 − 5 000 = −180 y 5 180 − 5 000 = +180."],
        ["2", f"χ² = 180²/5 000 + 180²/5 000 = 6.48 + 6.48 = {f(chi5.statistic, 2)} (1 gl). {f(chi5.statistic, 2)} &gt; 3.84 y &gt; 10.83 : sí supera ambos. p = {pv(chi5.pvalue)}."],
        ["3", f"Sí hay SRM (p ≈ {f(chi5.pvalue, 4)} &lt; 0.001): no se reporta el efecto de conversión. Primero se audita la infraestructura, la asignación y el registro de datos (¿en qué segmento se pierde el balance?)."],
        ["4", "No. Un p grande indica que no hay evidencia de desbalance en los conteos; no prueba que la aleatorización fuera perfecta (puede haber problemas que no cambian los conteos)."]])
    s += sol("Bloque 6", [
        ["1", "Tratadas: 45 − 31 = +14. Control: 39 − 33 = +6. DiD = 14 − 6 = +8 millones."],
        ["2", "Comparar «después» (45 frente a 39) mezcla el efecto con la diferencia que ya existía (31 frente a 33). Mirar solo antes-después de las tratadas mezcla el efecto con la tendencia (el control también subió +6)."],
        ["3", "Y<sub>it</sub> = β₀ + β₁Trat<sub>i</sub> + β₂Post<sub>t</sub> + β₃(Trat<sub>i</sub> × Post<sub>t</sub>) + u<sub>it</sub>; el DiD es β₃ (aquí 8)."],
        ["4", "Tendencias paralelas: sin la campaña, las tratadas habrían evolucionado como el control. Evidencia: varios períodos previos con brechas similares (gráfico, estudio de eventos, coeficiente pre-tendencia ≈ 0)."]])
    s += sol("Bloque 7", [
        ["1", "Falso. El p-valor no es la probabilidad de que H₀ sea cierta; se calcula suponiendo H₀ cierta."],
        ["2", "Verdadero: es la definición (probabilidad, bajo «sin efecto», de un resultado al menos así de extremo)."],
        ["3", "Falso. Con 500 000 por grupo, +0.4 pp puede ser significativo y a la vez pequeño. Se mira el IC y se compara con el MDE y el costo-beneficio."],
        ["4", "Comparaciones múltiples: con 20 pruebas al 5 %, P(≥ 1 falso positivo) = 1 − 0.95<super>20</super> = 64 %. Se corrige (Bonferroni: α = 0.05/20 = 0.0025; con p = 0.03 no se rechaza) o se declara una métrica primaria antes."]])
    build("apuntes/C22_Ejercicios_Cuaderno_ABTesting.pdf", s, "Ejercicios para el cuaderno · Clase 22")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
