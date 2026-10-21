"""Clase 43 · Evaluación estadística de modelos predictivos y bootstrap: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c43_numeros.json (lo escribe figuras_c43.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from math import comb, exp, log, sqrt
from pathlib import Path

import numpy as np
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c43_numeros.json").read_text(encoding="utf-8"))
C43 = colors.HexColor("#0E7490")
C43_BG = colors.HexColor("#ECFEFF")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas clase.pdf + evaluacion_bootstrap.py / .R del profesor Pedro Luiz Ramos + wisconsin.csv"
PR, IC, DF, MD, SL, CR, OF, PP, MC, DL, RE, OB, IN, RG, TM, ES = (N[k] for k in (
    "prueba", "ic", "dif", "media", "solap", "criterios", "oof", "prop", "mcnemar", "delong", "reentrenar", "oob", "incert", "regresion", "tamano", "esquema"))
PS, PA = PR["simple"], PR["amplio"]


def f(x, d=2):
    return f"{x:.{d}f}"


def ci(v, d=4):
    return f"[{v[0]:.{d}f}; {v[1]:.{d}f}]"


def mil(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.78):
    s.append(band(f"Clase 43 · {titulo}", FUENTE, C43))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C43, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C43))
    s.append(Spacer(1, 1))
    for fig, pie, *w in figuras:
        s.append(img(fig, width=CONTENT_W * (w[0] if w else fw)))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C43)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ---------------------------------------------------------------- cálculos de apoyo
def mcnemar(b, c):
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, float(stats.chi2.sf(stat, 1)), float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def p_fuera(n):
    return (1 - 1 / n) ** n


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 43 · Evaluación estadística de modelos predictivos y bootstrap",
                  "Profesor Pedro Luiz Ramos · retoma IC y valores p (Clase 22), la credibilidad bayesiana (Clase 23), SVM y Wisconsin (Clase 37) y OOF, CV y bootstrap/.632 (Clase 38)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Una métrica de prueba es una estimación, no una constante</b>: con otros casos de evaluación saldría otro valor. Por eso, después de <b>seleccionar</b> la receta con entrenamiento y validación "
                    "y <b>evaluarla</b> una sola vez en prueba, hay que <b>cuantificar la incertidumbre</b>. El <b>bootstrap</b> remuestrea con reposición los casos de prueba (con los modelos y sus predicciones fijos), "
                    "recalcula la métrica miles de veces y usa los percentiles como intervalo. Para <b>comparar dos modelos</b> se remuestrean los <b>mismos casos</b> para ambos y se mira la <b>distribución de la diferencia</b>: "
                    "si el intervalo de la diferencia contiene 0 no hay evidencia clara; si no lo contiene, la hay. Comparar dos intervalos separados no es lo mismo.", "body")], bg=C43_BG, border=C43, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c43_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>IC y valor p</b> (Clase 22) e <b>intervalo creíble</b> (Clase 23): qué significa «95 %»",
                     "<b>Regresión logística, AUC y Wisconsin</b> (Clases 31 y 37): las mismas 569 muestras y la misma partición",
                     "<b>OOF, validación cruzada y bootstrap/.632</b> (Clase 38): aquí se usa el bootstrap sobre los casos de <b>prueba</b>",
                     "<b>R ↔ Python</b>: el profesor trae los dos scripts, con generadores aleatorios distintos"], "cell", C43)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Una diferencia no basta</b> → <b>AIC, BIC y OOF</b> (seleccionar) → <b>bootstrap de la media</b> → <b>IC de AUC y log loss</b> → <b>diferencia pareada</b> → <b>alcance y límites</b>",
                      f"<b>wisconsin.csv</b>: {N['datos']['n']} muestras, {N['datos']['n_tr']} de entrenamiento (5 pliegues) y {N['datos']['n_te']} de prueba ({N['datos']['n_neg']} benignos y {N['datos']['n_pos']} malignos). Es la misma partición de la Clase 37 (con las columnas id y fold)",
                      "Pregunta guía: ¿el modelo amplio es mejor que el simple, o fue la suerte de esta prueba?"], "cell", C43)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con otras clases (sin repetirlas)</b>", "cellb"),
                  P("<b>C22 → C43</b>: allí se construía un IC y se leía el valor p de un contraste; hoy el IC es para una <b>métrica de predicción</b> y no tiene fórmula cerrada simple (se obtiene remuestreando). "
                    f"<b>C23 → C43</b>: con Beta(1 + k, 1 + n − k) el intervalo creíble de la exactitud del modelo amplio ({PP['exactitud']['k']} de {PP['exactitud']['n']}) es {ci(PP['exactitud']['metodos']['Bayes (Beta(1 + k, 1 + n − k))'], 3)}, casi igual al bootstrap {ci(PP['exactitud']['metodos']['Bootstrap percentil'], 3)}. "
                    f"<b>C37 → C43</b>: el modelo amplio es la logística de la Clase 37 (AUC {f(PA['auc'], 4)}, exactitud {f(PA['exactitud'], 4)}); ahora se le pone un intervalo. "
                    "<b>C38 → C43</b>: el bootstrap OOB y el .632 estimaban el <b>error</b> de una receta; aquí el bootstrap mide la <b>incertidumbre de la evaluación</b> de modelos ya ajustados.", "cell")], bg=SOFT, border=C43, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: ¿0.9746 es «la» AUC del modelo?</b>", "cellb"),
                  P(f"No: es la AUC en <b>estos</b> {N['datos']['n_te']} casos. Con el bootstrap, el intervalo del 95 % es {ci(IC['auc_amplio'])}; el simple, {ci(IC['auc_simple'])}. "
                    "Una sola cifra de prueba sin intervalo oculta cuánto podría cambiar con otra muestra de casos.", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Una diferencia observada no basta",
        "Dos modelos con log loss 0.30 y 0.28 difieren en 0.02, pero <b>otra muestra de evaluación podría dar otra diferencia</b>. Evaluar también es <b>cuantificar la incertidumbre</b>: el error estándar describe cuánto varía una <b>estimación</b>, no cuánto difieren los casos entre sí.",
        [("El mismo modelo, otras pruebas", f"Con acierto real 0.90, la exactitud observada tiene desviación {f(IN['sd_n100'], 4)} con n = 100 y {f(IN['sd_n1000'], 4)} con n = 1 000: el modelo no cambió, cambiaron los casos."),
         ("SE de una exactitud", f"SE ≈ √(â(1 − â)/n): {f(IN['se'][0], 4)} con 90 de 100 y {f(IN['se'][1], 4)} con 900 de 1 000. Intervalos normales aproximados {ci(IN['ic'][0], 3)} y {ci(IN['ic'][1], 3)} (diapositiva 12)."),
         ("Qué significa un IC", f"«95 %» describe el <b>procedimiento</b>: en 40 muestras simuladas, {IN['cubre']} intervalos contienen la media verdadera (la diapositiva muestra 38); no todos deben hacerlo.")],
        [("c43_incertidumbre.png", "Izquierda: distribución de la exactitud observada para n = 100 y n = 1 000 (diapositiva 11). Centro: SE e intervalo normal del mismo 90 %. Derecha: 40 intervalos al 95 % para una media (σ = 1 conocida).", 0.97)],
        ["SE(â) ≈ √(â(1 − â)/n)", "IC aprox. = â ± 1.96 · SE",
         "desviación estándar de los datos ≠ error estándar de una estimación", "dependencia entre casos → otro análisis"],
        None,
        [f"Wisconsin, modelo amplio: {PP['exactitud']['k']} de {PP['exactitud']['n']} → â = {f(PP['exactitud']['est'], 4)}, SE = {f(PP['exactitud']['se_wald'], 4)}; sensibilidad {PP['sensibilidad']['k']} de {PP['sensibilidad']['n']}: SE = {f(PP['sensibilidad']['se_wald'], 4)} (solo 64 malignos)",
         "Una mejora pequeña puede ser relevante o no según el objetivo: hace falta el intervalo y un criterio previo",
         "Con la fórmula normal, para reducir el SE a la mitad se necesitan 4 veces más casos"],
        ["La fórmula del SE supone casos independientes y un clasificador fijo; cerca de 0 o 1 el intervalo normal puede salirse de [0, 1] (use Wilson o exacto). El IC de una media no es un IC para los datos."], fw=0.97)

    topic_page(
        s, "Seleccionar y evaluar son tareas distintas: AIC, BIC y OOF",
        "Primero se <b>define</b> la unidad de separación y la métrica; se <b>selecciona</b> la receta con entrenamiento y validación; se <b>reajusta</b> con todo el entrenamiento; se <b>evalúa</b> en prueba sin volver a elegir; y recién entonces se <b>cuantifica la incertidumbre</b>. Cambiar la receta mirando la prueba la deja de hacer independiente.",
        [("Receta dentro del pliegue", "Estandarizar, elegir variables y fijar el umbral se hace <b>solo con las filas de ajuste</b> (en un Pipeline). Estandarizar antes de separar filtra información; el bootstrap no corrige una fuga anterior."),
         ("AIC y BIC", f"AIC = −2ℓ + 2p; BIC = −2ℓ + p log(n). Menor es mejor y solo se comparan <b>dentro de la columna</b>. En Wisconsin ambos eligen el modelo amplio: AIC {f(CR[2]['aic'], 2)} y BIC {f(CR[2]['bic'], 2)}, frente a {f(CR[0]['aic'], 2)} y {f(CR[0]['bic'], 2)} del simple."),
         ("Condiciones de uso", "Misma respuesta y mismas observaciones, verosimilitudes comparables y ajuste por máxima verosimilitud: no se aplican a SVM, Isolation Forest ni logísticas penalizadas como si nada cambiara.")],
        [("c43_seleccion.png", "Izquierda: la partición de Wisconsin (398 + 171; pliegues de 80, 80, 80, 79 y 79). Centro: AIC y BIC en entrenamiento. Derecha: log loss y AUC fuera de pliegue (OOF) con los cinco pliegues del profesor.", 0.97)],
        ["AIC = −2ℓ(θ̂) + 2p", "BIC = −2ℓ(θ̂) + p log(n)",
         "p cuenta el intercepto y cada coeficiente", "ΔAIC = AIC − AIC mínimo"],
        ["modelo.aic", "modelo.bic_llf   # = −2ℓ + p·ln(n), como BIC() de R"],
        [f"Ejemplo de la clase (n = 100): AIC 166, 162 y 167; BIC 173.82, 175.03 y 187.84 → AIC elige B y BIC elige A",
         f"OOF en entrenamiento: log loss {f(OF['Simple']['log_loss'], 3)}, {f(OF['Intermedio']['log_loss'], 3)} y {f(OF['Amplio']['log_loss'], 3)}; AUC {f(OF['Simple']['auc'], 3)}, {f(OF['Intermedio']['auc'], 3)} y {f(OF['Amplio']['auc'], 3)}",
         "Un AIC menor no cuantifica una mejora práctica ni garantiza buenas predicciones"],
        ["Seleccionar con OOF no vuelve independiente la cifra del ganador. Y los {0} casos de prueba ya se revisaron en clases anteriores: aquí son una <b>demostración</b>, no una validación nueva.".format(N['datos']['n_te'])])

    topic_page(
        s, "Bootstrap: remuestrear con reposición",
        "El bootstrap parte de la <b>muestra observada</b>, no de la población: se sortean n IDs <b>con reposición</b> para formar una remuestra del mismo tamaño, se calcula la estadística y se repite B veces. Las B estadísticas forman <b>una distribución bootstrap</b>; sus percentiles 2.5 y 97.5 dan el <b>IC percentil</b>.",
        [("Una remuestra", f"Con los valores 2, 4, 5, 7 y 8 (media 5.2), la remuestra 1 de la diapositiva sortea los IDs 2, 2, 1, 5 y 3 → media {f(ES['medias_ejemplo'][0], 1)}; la 2, media {f(ES['medias_ejemplo'][1], 1)}; la 3, media {f(ES['medias_ejemplo'][2], 1)}."),
         ("Error estándar e IC", f"Con B = 2 000 (semilla 20261012) en Python: SE = {f(MD['se'], 4)} e IC {ci(MD['ic'], 1)}; en R: 0.966 y [3.4; 7.0]. La distribución <b>exacta</b> (3 125 remuestras) da SE {f(MD['se_exacto'], 4)} e IC {ci(MD['ic_exacto'], 1)}."),
         ("Más remuestras ≠ más datos", f"Subir B reduce el error Monte Carlo del cálculo (SE {f(MD['por_B']['100']['se'], 3)}, {f(MD['por_B']['500']['se'], 3)} y {f(MD['por_B']['2000']['se'], 3)} con 100, 500 y 2 000), pero no la incertidumbre de tener solo 5 casos.")],
        [("c43_esquema_bootstrap.png", "Izquierda: la muestra original y las tres remuestras de la diapositiva 31; el ID 4 queda fuera de la primera. Derecha: distribución bootstrap de la media con B = 2 000 y límites percentiles.", 0.8),
         ("c43_media.png", "Izquierda: el resultado de Python. Centro: las 3 125 remuestras posibles. Derecha: el IC apenas cambia al aumentar B.", 0.8)],
        ["θ*_b = estadística de la remuestra b", "SE_boot = √(Σ(θ*_b − θ̄*)²/(B − 1))",
         "IC percentil = [Q₀.₀₂₅(θ*), Q₀.₉₇₅(θ*)]", "P(un caso queda fuera) = (1 − 1/n)ⁿ ≈ 0.368"],
        ["rng = np.random.default_rng(20261012)", "m[b] = rng.choice(v, size=5, replace=True).mean()", "np.quantile(m, [0.025, 0.975])"],
        [f"Un caso queda fuera de una remuestra con probabilidad {f(OB['p5'], 3)} si n = 5 y {f(OB['p171'], 3)} si n = 171 (límite 1/e = {f(OB['limite'], 3)})",
         "R y Python usan generadores distintos: las diferencias pequeñas entre límites son variación Monte Carlo",
         "Con cinco datos, el intervalo no está bien calibrado: sirve para entender la cuenta"],
        ["Sortear filas como si fueran independientes subestima la incertidumbre si hay varias mediciones por paciente (remuestrear pacientes) o dependencia temporal (bloques)."], fw=0.97)

    topic_page(
        s, "IC bootstrap de las métricas de un modelo fijo",
        "En la prueba <b>no se entrena de nuevo</b>: se conservan las probabilidades de cada caso y se remuestrean los casos. El <b>log loss</b> y la <b>exactitud</b> se remuestrean por filas; la <b>AUC</b>, <b>por clase</b> (107 benignos y 64 malignos), y los dos modelos usan los mismos IDs. El intervalo estudia la <b>evaluación del modelo ya ajustado</b>.",
        [("Resultados puntuales", f"Simple: AUC {f(PS['auc'], 4)}, log loss {f(PS['log_loss'], 4)}, exactitud {f(PS['exactitud'], 4)}, sensibilidad {f(PS['sensibilidad'], 4)}. Amplio: {f(PA['auc'], 4)}, {f(PA['log_loss'], 4)}, {f(PA['exactitud'], 4)} y {f(PA['sensibilidad'], 4)} (umbral 0.5 para las dos últimas)."),
         ("Intervalos percentiles", f"AUC simple {ci(IC['auc_simple'])} y amplio {ci(IC['auc_amplio'])}; log loss simple {ci(IC['ll_simple'])} y amplio {ci(IC['ll_amplio'])} (en R el profesor obtuvo [0.9004; 0.9708], [0.9503; 0.9915], [0.2124; 0.4005] y [0.1066; 0.2798])."),
         ("Otras métricas", f"Exactitud del amplio: Wilson {ci(PP['exactitud']['metodos']['Wilson'], 3)}, exacto {ci(PP['exactitud']['metodos']['Exacto (Clopper-Pearson)'], 3)}, bootstrap {ci(PP['exactitud']['metodos']['Bootstrap percentil'], 3)}. Sensibilidad (64 casos): bootstrap {ci(PP['sensibilidad']['metodos']['Bootstrap percentil'], 3)}.")],
        [("c43_auc_logloss.png", "Distribuciones bootstrap (B = 2 000) de log loss y AUC de los modelos simple y amplio; línea roja: estimación original; discontinuas: límites percentiles (diapositivas 42 y 43).", 0.72),
         ("c43_roc_banda.png", "Curva ROC con banda bootstrap del 95 % (500 remuestras por clase): la del modelo simple es más ancha.", 0.36),
         ("c43_intervalos_prop.png", "Cinco formas de intervalo para la exactitud y la sensibilidad del modelo amplio (puente con las Clases 22 y 23).", 0.5)],
        ["log loss = −media[y ln p + (1 − y) ln(1 − p)]", "AUC: se remuestrea dentro de cada clase",
         "mismos IDs para los dos modelos", "no se repite fit() ni predict() en el ciclo"],
        None,
        [f"Los errores seguros pueden inflar el log loss de una remuestra; el IC del simple ({f(IC['ll_simple'][1] - IC['ll_simple'][0], 3)} de ancho) es más ancho que el del amplio ({f(IC['ll_amplio'][1] - IC['ll_amplio'][0], 3)})",
         f"La banda ROC media mide {f(N['roc']['ancho_medio']['simple'], 3)} (simple) y {f(N['roc']['ancho_medio']['amplio'], 3)} (amplio) de ancho",
         "No aplicar log loss a puntajes de SVM o de anomalía que no sean probabilidades"],
        ["El intervalo no incluye la variación por volver a entrenar ni cambios futuros de población o prevalencia."])

    topic_page(
        s, "Comparar dos modelos: la diferencia pareada",
        "Los dos modelos ven los <b>mismos casos</b>: un caso difícil los perjudica a ambos. Por eso se remuestrean los mismos IDs, se calcula la <b>diferencia</b> en cada remuestra y se toman los percentiles de <b>esas diferencias</b>. Restar los extremos de dos intervalos separados <b>no</b> es lo mismo.",
        [("La diferencia", f"Positivo favorece al amplio: Δ log loss = simple − amplio = {f(DF['ll'], 4)}, IC {ci(DF['ic_ll'])}; ΔAUC = amplio − simple = {f(DF['auc'], 4)}, IC {ci(DF['ic_auc'])}. Ninguno contiene 0."),
         ("Emparejar angosta el intervalo", f"La AUC del simple y del amplio en cada remuestra tiene correlación {f(DF['corr_auc'], 2)}. Con remuestras independientes el ΔAUC daría {ci(SL['no_pareado'])} (ancho {f(SL['ancho_no_pareado'], 3)}, contiene 0) frente a {ci(SL['pareado'])} (ancho {f(SL['ancho_pareado'], 3)})."),
         ("Estadístico y utilidad", "Que el IC excluya 0 muestra una diferencia <b>estadística</b>; que sea <b>útil</b> depende de una mejora relevante fijada antes (0.010 no es regla universal). Si el IC contiene 0 hay «no hay evidencia clara», no «equivalencia».")],
        [("c43_pareado.png", "Izquierda y centro: diferencias pareadas de log loss y de AUC en 2 000 remuestras (rojo: 0; verde: límites). Derecha: las AUC de las dos remuestras están correlacionadas (diapositivas 47 a 50).", 0.8),
         ("c43_solapamiento.png", "Izquierda: los IC de cada AUC se solapan. Derecha: la diferencia pareada no contiene 0 y es más precisa que la no pareada.", 0.7)],
        ["Δ*_b = LL*_simple,b − LL*_amplio,b", "IC de la diferencia = percentiles de Δ*",
         "log loss: simple − amplio; AUC: amplio − simple", "positivo favorece al amplio en ambos"],
        ["d = ll_boot[:, 0] - ll_boot[:, 1]", "d_auc = auc_boot[:, 1] - auc_boot[:, 0]", "np.quantile(d, [0.025, 0.975])"],
        [f"Las dos primeras remuestras de log loss del script dan Δ = {f(DF['primeras3'][0]['d'], 4)} y {f(DF['primeras3'][1]['d'], 4)}; la diferencia original es {f(DF['ll'], 4)}",
         f"En ninguna de las 2 000 remuestras el amplio fue peor ({f(DF['frac_neg_auc'] * 100, 0)} % con ΔAUC ≤ 0)",
         "Este es un IC de diferencia, no un valor p; pero «el IC excluye 0» es equivalente a rechazar «sin diferencia» al 5 %"],
        ["Dos IC que se solapan no implican «sin diferencia significativa»: la diferencia pareada puede excluir 0 aunque los IC individuales se toquen."], fw=0.9)

    topic_page(
        s, "Alcance del intervalo: qué cubre y qué no",
        "El intervalo calculado cubre <b>la variación al remuestrear los casos de evaluación de modelos fijos</b>. No cubre la variación por <b>volver a entrenar</b> y seleccionar toda la receta, ni cambios de población, mediciones o prevalencia, ni corrige sesgos de diseño o fuga de datos.",
        [("Dos preguntas distintas", f"Repitiendo 300 particiones de las 569 muestras (reentrenando y reevaluando) el ΔAUC tiene media {f(RE['media_dauc'], 4)}, desviación {f(RE['sd_dauc'], 4)} e IC {ci(RE['ic_dauc'])}; el amplio ganó en {f(RE['frac_amplio_mejor'] * 100, 0)} % de las particiones."),
         ("McNemar y DeLong (extensión)", f"McNemar compara exactitudes con casos discordantes: b = {MC['b']} y c = {MC['c']}, χ² = {f(MC['stat'], 2)}, p = {f(MC['p_chi'], 4)} (exacto {f(MC['p_exacto'], 4)}). DeLong da ΔAUC {f(DL['diff'], 4)}, z = {f(DL['z'], 2)}, p = {f(DL['p'], 4)}. No se vieron en clase."),
         ("El tamaño de la prueba", f"El ancho del IC baja como 1/√n: con acierto 0.90, {f(TM['ancho_normal'][2], 3)} con 100 casos y {f(TM['ancho_normal'][5], 3)} con 1 000. En regresión (MAE/RMSE) con 5 casos: MAE {f(RG['mae'], 1)} [{f(RG['ic_mae'][0], 1)}; {f(RG['ic_mae'][1], 1)}], RMSE {f(RG['rmse'], 3)} {ci(RG['ic_rmse'], 3)} (ilustrativo).")],
        [("c43_mcnemar.png", "Izquierda: tabla 2 × 2 de aciertos de los dos modelos con umbral 0.5. Derecha: el mismo ΔAUC con bootstrap pareado y con DeLong.", 0.8),
         ("c43_reentrenar.png", "Izquierda: ΔAUC al repetir la partición y reentrenar. Derecha: comparación con el intervalo del bootstrap de la prueba (modelos fijos).", 0.6)],
        ["McNemar: χ² = (|b − c| − 1)²/(b + c)", "gl = 1; exacto: binomial(b + c, 0.5)",
         "se_AUC = f(estructura de los pares)", "ancho IC ∝ 1/√n"],
        ["from scipy.stats import binomtest", "binomtest(min(b, c), b + c, 0.5).pvalue"],
        [f"McNemar usa solo {MC['n_disc']} de {N['datos']['n_te']} casos (los discordantes): el amplio acierta {MC['c']} que el simple falla; el simple, {MC['b']} que el amplio falla",
         f"Repetir particiones da un IC de ancho {f(RE['ancho_rep'], 3)}; el bootstrap de la prueba, {f(RE['ancho_boot'], 3)}: parecidos aquí, pero responden preguntas distintas",
         "Con más casos de prueba el intervalo se angosta; con más remuestras solo se estabiliza el cálculo"],
        ["La prueba reutilizada es una demostración docente, no una validación externa nueva. Repetir el cálculo 2 000 veces no corrige sesgos de diseño, selección ni fuga de datos."], fw=0.97)

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase (transcripción)</b>", "cellb"),
                  bullets([f"<b>«Si los intervalos se cruzan, no hay diferencia entre los modelos»</b> (y el ejemplo de las encuestas): es <b>incorrecto en general</b>. Aquí los IC de la AUC se solapan ({f(SL['solape'][0], 3)}–{f(SL['solape'][1], 3)}) y la diferencia pareada {ci(SL['pareado'])} excluye 0; con remuestras independientes sí contendría 0. El mismo profesor muestra después que ΔAUC es significativa.",
                           f"<b>«Con log loss gana el modelo simple»</b>: el signo está al revés. Δ = simple − amplio = {f(DF['ll'], 4)} &gt; 0 y el IC {ci(DF['ic_ll'])} favorecen al <b>amplio</b> (como dice la diapositiva 49); con AUC también gana el amplio.",
                           "<b>«No se pueden usar bootstrap y validación cruzada»</b>: es imprecisión. Se pueden combinar (bootstrap de las predicciones OOF, .632, validación anidada), con cuidado; lo que no se debe es reajustar o elegir con la prueba.",
                           "<b>«Nivel de confianza de 2 puntos»</b> (encuestas): son 2 puntos de <b>margen de error</b>; el nivel de confianza es 95 %.",
                           "<b>Cifras mal transcritas</b> (ya corregidas): RMSE 3.31 (no 3.81), log loss del ejemplo 0.4723 (no 0.46), B = 100 (no 1 000) para el SE 1.029, prueba de 171 (no 161 ni 179) y 569 muestras (no 659).",
                           "<b>Fold de «80 casos»</b>: los pliegues del entrenamiento miden 80, 80, 80, 79 y 79; se ajusta con 318 o 319 casos (la diapositiva 4 dice 318)."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, scripts y verificación</b>", "cellb"),
                  bullets([f"<b>Reproducido con el script de Python</b> (semillas 20261012, 20261013, 20261017, 20261014): AIC {f(CR[2]['aic'], 4)} y BIC {f(CR[2]['bic'], 4)} del amplio; AUC {f(PS['auc'], 4)} y {f(PA['auc'], 4)}; log loss {f(PS['log_loss'], 4)} y {f(PA['log_loss'], 4)}; SE de la media {f(MD['se'], 4)}; ΔAUC {f(DF['auc'], 4)} {ci(DF['ic_auc'])}.",
                           "<b>R frente a Python</b>: las diapositivas muestran los resultados de R (SE 0.966; IC [3.4; 7.0]; [0.1066; 0.2798]…); Python con la misma semilla da otras remuestras y límites algo distintos (por ejemplo [3.2; 7.0]). Son variación Monte Carlo; el SE exacto de la media es 0.955.",
                           "<b>Slide 13</b>: «compras muy distintas entre sí» es un error de tipeo (deben ser <b>muestras</b>). <b>Script en R</b>: la ruta de lectura es absoluta (D:/Aulas_Diplomado/…): hay que cambiarla.",
                           "<b>Qué no hace el material</b>: no incluye McNemar ni DeLong, no usa pruebas con varios umbrales ni estudia la variación por reentrenar; esas piezas son extensiones de esta guía.",
                           "<b>Transcripción con ruido</b>: «Bustrap / Boostrepp» = bootstrap, «IC / ICIe / A y C» = AIC, «curva rock» = ROC, «relm» = glm, «error coláctico» = error cuadrático. Se marcó el recreo hacia 0:47:31 (el video sigue sin hueco)."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Explico por qué una métrica de prueba tiene incertidumbre y calculo el SE de una exactitud",
                           "Hago una remuestra bootstrap a mano y calculo la estadística; construyo un IC percentil desde valores ordenados",
                           "Distingo seleccionar (AIC, BIC, OOF) de evaluar, y sé qué queda dentro del pliegue",
                           "Hago bootstrap de la prueba con modelos fijos: log loss por filas, AUC por clase",
                           "Comparo dos modelos con la diferencia pareada y explico por qué solapar IC no basta",
                           "Calculo el estadístico de McNemar a mano y lo interpreto",
                           "Digo qué incertidumbre cubre el intervalo y cuál no",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Siguiente</b>: repaso final con el profesor y el Taller."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C43_Guia_EvaluacionBootstrap.pdf", s, "Clase 43 · Evaluación y bootstrap")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#0E7490">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 43 · Evaluación estadística y bootstrap", "Para leer antes de los ejercicios · lectura de ~30 min", C43))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Una cifra de prueba es una estimación", "Cambia con los casos de evaluación; el SE y el IC dicen cuánto."),
                            ("Seleccionar ≠ evaluar", "Receta con entrenamiento y validación; prueba una vez; sin fuga."),
                            ("Bootstrap = remuestrear con reposición", "Misma n; ≈ 63.2 % de los casos distintos; percentiles = IC."),
                            ("Modelos fijos, casos remuestreados", "No se reentrena: log loss por filas, AUC por clase."),
                            ("Compara la diferencia, no los dos IC", "Mismos IDs en ambos modelos; IC de Δ; si contiene 0, no hay evidencia clara."),
                            ("Conoce el alcance", "El IC no incluye reentrenar, cambios de población ni sesgos de diseño.")], C43)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C43_EvaluacionBootstrap.ipynb</font> o el cuaderno <font name='Mono'>C43_Ejercicios_Cuaderno_EvaluacionBootstrap.pdf</font>.", "small"))

    # 1 error estándar
    s.append(h("1", "Una métrica es una estimación: el error estándar"))
    s += para("Si un clasificador fijo acierta con probabilidad p en cada caso independiente, la exactitud observada en n casos tiene error estándar √(p(1 − p)/n), que se estima con â. Aproximadamente, â ± 1.96 SE contiene a p en el 95 % de los experimentos.")
    a, n1, n2 = 0.85, 200, 800
    se1, se2 = sqrt(a * (1 - a) / n1), sqrt(a * (1 - a) / n2)
    s.append(ejemplo("un filtro acierta 170 de 200 casos (0.85); otro, 680 de 800 (0.85)", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["SE con n = 200", "√(0.85 × 0.15/200)", f"{f(se1, 4)}"],
               ["IC normal", "0.85 ± 1.96 × SE", f"[{f(a - 1.96 * se1, 3)}; {f(a + 1.96 * se1, 3)}]"],
               ["SE con n = 800", "√(0.85 × 0.15/800)", f"{f(se2, 4)}"],
               ["IC normal", "0.85 ± 1.96 × SE", f"[{f(a - 1.96 * se2, 3)}; {f(a + 1.96 * se2, 3)}]"]],
              [38 * mm, 70 * mm, CONTENT_W - 108 * mm - 18]),
        P(f"Lectura: el mismo 0.85 es más preciso con 800 casos; con 4 veces más casos el SE se reduce a la mitad ({f(se1 / se2, 2)} veces). <b>Error común</b>: confundir la desviación de los datos con el error estándar de la estimación, o creer que «95 %» es la probabilidad de que este intervalo contenga a p (es la cobertura del procedimiento).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c43_incertidumbre.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(img("c43_tamano.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A1 y A2) y los Bloques 1 y 3 del cuaderno", [P("Simula exactitudes con n = 100 y n = 1 000 y compara su desviación con √(p(1 − p)/n).", "cell")]))

    # 2 bootstrap a mano
    s.append(h("2", "Bootstrap a mano: remuestra, estadística e IC percentil"))
    s += para("Se parte de la muestra observada. Se sortean n IDs con reposición (la remuestra tiene el mismo tamaño); se calcula la estadística; se repite B veces. Con los B valores ordenados, el IC percentil al 95 % usa los cuantiles 2.5 % y 97.5 %. En un ejemplo pequeño a mano se usa un nivel de 90 %: con B = 20 se descarta 1 valor en cada cola.")
    vals = np.array([3, 5, 6, 9, 12])
    ids = [4, 1, 4, 5, 2]
    rem = vals[[i - 1 for i in ids]]
    rng = np.random.default_rng(4300)
    bm = np.sort([rng.choice(vals, 5, replace=True).mean() for _ in range(20)])
    s.append(ejemplo("cinco tiempos: 3, 5, 6, 9 y 12 minutos (media 7.0)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Remuestra con IDs 4, 1, 4, 5 y 2", "9, 3, 9, 12 y 5", f"media {f(rem.mean(), 1)}"],
               ["Se repite B = 20 veces", "20 medias ordenadas", ", ".join(f(x, 1) for x in bm)],
               ["IC del 90 % (sin la 1.ª ni la 20.ª)", "2.º menor y 2.º mayor", f"[{f(bm[1], 1)}; {f(bm[-2], 1)}]"],
               ["Error estándar bootstrap", "desviación de las 20 medias", f"{f(bm.std(ddof=1), 3)}"]],
              [48 * mm, 40 * mm, CONTENT_W - 88 * mm - 18]),
        P("Lectura: la distribución de las medias remuestreadas da una idea de cuánto cambiaría la media con otra muestra. Con B = 20 y n = 5 es solo para entender la cuenta; en la práctica B = 2 000 o más. <b>Error común</b>: remuestrear sin reposición (daría siempre la misma muestra), o creer que más remuestras dan más información sobre los datos.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c43_esquema_bootstrap.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(img("c43_oob.png", width=CONTENT_W * 0.7))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A3 a A5) y los Bloques 1, 2 y 7 del cuaderno", [P("Reproduce 5.2, SE 0.9306 e IC [3.2; 7.0] con la semilla 20261012 y compáralos con las 3 125 remuestras exactas.", "cell")]))

    # 3 métricas con modelos fijos y pareado
    s.append(h("3", "Modelos fijos, diferencia pareada y por qué no basta solapar"))
    s += para("Se ajustan los modelos una vez; con sus predicciones fijas se remuestrean los casos de prueba (log loss por filas, AUC por clase) y se recalcula la métrica. Para comparar, se usa la <b>misma</b> remuestra en ambos modelos y se toman los percentiles de la <b>diferencia</b>.")
    dA = np.array([0.9, 0.2, 0.4, 1.2, 0.1, 0.6, 0.3, 0.8])
    dB = np.array([0.7, 0.2, 0.3, 0.9, 0.1, 0.5, 0.3, 0.6])
    dd = dA - dB
    m = len(dd)
    se_p = dd.std(ddof=1) / sqrt(m)
    se_u = sqrt(dA.var(ddof=1) / m + dB.var(ddof=1) / m)
    s.append(ejemplo("pérdidas de dos modelos en los mismos 8 casos (A y B)", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Diferencias A − B", "caso por caso", ", ".join(f(x, 1) for x in dd)],
               ["Media y desviación de las diferencias", "media; desviación (n − 1)", f"{f(dd.mean(), 4)} y {f(dd.std(ddof=1), 4)}"],
               ["SE pareado", "desviación/√8", f"{f(se_p, 4)}"],
               ["SE sin emparejar", "√(sA²/8 + sB²/8)", f"{f(se_u, 4)}"],
               ["Cociente", "SE no pareado / SE pareado", f"{f(se_u / se_p, 2)}"]],
              [48 * mm, 60 * mm, CONTENT_W - 108 * mm - 18]),
        P("Lectura: los dos modelos sufren los mismos casos difíciles (la pérdida del caso 4 es alta para ambos), así que las diferencias varían poco y el SE pareado es mucho menor. <b>Error común</b>: tratar las dos columnas como muestras independientes, o concluir «no hay diferencia» porque los IC de cada modelo se solapan.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c43_pareado.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(img("c43_solapamiento.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C1 a C5) y los Bloques 4 y 6 del cuaderno", [P("Remuestrea los mismos IDs para los dos modelos y compara el IC pareado con el no pareado.", "cell")]))

    # 4 McNemar
    s.append(h("4", "McNemar: comparar exactitudes con casos discordantes (extensión)"))
    s += para("Con las predicciones de dos clasificadores en los mismos casos se arma una tabla 2 × 2 de aciertos. Solo cuentan los casos <b>discordantes</b>: b (solo A acierta) y c (solo B acierta). Si no hay diferencia, b y c deberían ser parecidos: χ² = (|b − c| − 1)²/(b + c), con 1 grado de libertad.")
    b_, c_ = 12, 4
    st, pc, pe = mcnemar(b_, c_)
    s.append(ejemplo("100 casos: ambos aciertan 70; solo A, 12; solo B, 4; ambos fallan 14", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Exactitudes", "(70 + 12)/100 y (70 + 4)/100", "0.82 y 0.74"],
               ["Discordantes", "b = 12; c = 4", "b + c = 16"],
               ["Estadístico", "(|12 − 4| − 1)²/16 = 49/16", f"{f(st, 4)}"],
               ["Valor p (χ², 1 gl)", "P(χ² ≥ 3.06)", f"{f(pc, 4)}"],
               ["Valor p exacto", "2 P(X ≤ 4), X ~ Binomial(16; 0.5)", f"{f(pe, 4)}"]],
              [38 * mm, 78 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: con 0.82 frente a 0.74 no se rechaza «sin diferencia» al 5 % (p ≈ 0.08). <b>Error común</b>: usar los 100 casos en vez de los discordantes, o olvidar que McNemar compara exactitudes con un umbral fijo, no AUC ni log loss.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c43_mcnemar.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D1 y D2) y el Bloque 5 del cuaderno", [P("Calcula la tabla con los aciertos de los dos modelos de Wisconsin y compara con DeLong y con el bootstrap.", "cell")]))

    # 5 alcance
    s.append(h("5", "Qué cubre el intervalo y qué no"))
    s += para("El bootstrap de la prueba con modelos fijos responde «¿cuánto varía la evaluación de <b>este</b> modelo si cambian los casos?». No responde «¿cuánto varía el modelo si lo vuelvo a entrenar?». Tampoco corrige una fuga de datos previa ni cambios futuros de población.")
    s.append(ejemplo("tres diferencias hipotéticas de log loss con la mejora relevante fijada en 0.010", [
        pasos([["Δ̂", "IC 95 %", "Lectura"],
               ["0.010", "[−0.004; 0.024]", "no hay evidencia clara de superioridad; no prueba equivalencia"],
               ["0.002", "[0.001; 0.003]", "positiva, pero menor que la mejora relevante"],
               ["0.030", "[0.020; 0.050]", "positiva y por encima de la mejora relevante"]],
              [24 * mm, 36 * mm, CONTENT_W - 60 * mm - 18]),
        P("Lectura: excluir 0 no basta; hay que compararlo con el tamaño relevante. <b>Error común</b>: decir «equivalentes» porque el IC contiene 0.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c43_reentrenar.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D3 y D4)", [P("Repite 300 particiones (reentrenando) y compara el ancho del intervalo con el del bootstrap de la prueba.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Una métrica de prueba es una estimación; el IC dice cuánto podría variar.",
                       "Bootstrap: remuestras del mismo tamaño, con reposición; ≈ 63.2 % de casos distintos; percentiles 2.5 y 97.5.",
                       "Con modelos fijos: no se reentrena. Log loss por filas; AUC por clase; mismos IDs en ambos modelos.",
                       "Compara la diferencia pareada, no los dos IC por separado; excluir 0 no es lo mismo que una mejora relevante.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 4]))
            i += 4
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C43_Apuntes_EvaluacionBootstrap.pdf", out, "Apuntes · Clase 43 · Evaluación y bootstrap")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0E7490">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C43))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 43 · Evaluación y bootstrap", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C43))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: SE(â) = √(â(1 − â)/n) · â ± 1.96 SE · IC percentil = cuantiles 2.5 % y 97.5 % de las estadísticas bootstrap · P(caso fuera) = (1 − 1/n)ⁿ · SE pareado = desv(diferencias)/√n · McNemar: χ² = (|b − c| − 1)²/(b + c).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C43)], bg=SOFT, border=C43, pad=5))
    s.append(Spacer(1, 2))

    # datos
    y8 = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    p8 = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    ok8 = (y8 == p8).astype(int)
    r1, r2 = [3, 3, 6, 1, 8, 2, 3, 5], [7, 7, 1, 4, 2, 8, 5, 1]
    rng = np.random.default_rng(4301)
    bs40 = np.sort(np.round(rng.normal(0.88, 0.02, 40), 3))
    lA = np.array([0.6, 0.9, 0.3, 1.1, 0.4, 0.8])
    lB = np.array([0.5, 0.7, 0.3, 0.9, 0.3, 0.7])
    s += bloque(1, "Una remuestra bootstrap a mano", "★", "Un clasificador se evalúa en 8 casos. Verdadero y predicho (IDs 1 a 8): " + "; ".join(f"{i + 1}: y = {y8[i]}, pred = {p8[i]}" for i in range(8)) + ".", [
        "Calcula la exactitud original.",
        f"Remuestra A, con los IDs {', '.join(map(str, r1))}: calcula su exactitud. ¿Cuántos IDs distintos hay y cuáles no aparecen?",
        f"Remuestra B, con los IDs {', '.join(map(str, r2))}: calcula su exactitud. ¿Qué enseña sobre la variación de la métrica?"])
    s += bloque(2, "Intervalo percentil desde valores ordenados", "★★", f"Se hicieron B = 40 remuestras y se ordenaron las AUC de menor a mayor: {', '.join(f'{x:.3f}' for x in bs40)}. La AUC original fue 0.884.", [
        "Construye el IC percentil del 90 % descartando 5 % en cada cola (2 valores de cada lado).",
        "¿Contiene la AUC original? ¿Y el valor 0.85, que alguien propone como referencia?",
        "Si con B = 400 el IC casi no cambiara, ¿qué conclusión es correcta y cuál no?"])
    s += bloque(3, "Error estándar de una exactitud", "★★", "Un modelo acierta 54 de 60 casos de prueba. El bootstrap de esa prueba dio un SE de 0.040.", [
        "Calcula â y el SE con la fórmula normal. Compáralo con el SE bootstrap.",
        "Construye el IC normal del 95 %. ¿Por qué un IC exacto o de Wilson sería preferible aquí?",
        "¿Cuántos casos habría que evaluar para reducir el SE a la mitad (con la misma exactitud)?"])
    s += bloque(4, "Comparación pareada frente a no pareada", "★★★", "Pérdida (log loss por caso) de dos modelos en los mismos 6 casos. A: " + ", ".join(f"{x:.1f}" for x in lA) + ". B: " + ", ".join(f"{x:.1f}" for x in lB) + ".", [
        "Calcula las diferencias A − B, su media y su desviación (n − 1).",
        "Calcula el SE pareado y el SE como si A y B fueran muestras independientes.",
        "Explica con una frase por qué el SE pareado es menor y qué supone emparejar."])
    s += bloque(5, "McNemar a mano", "★★", "Dos clasificadores se evalúan en 120 casos: ambos aciertan 80; solo A acierta 15; solo B acierta 5; ambos fallan 20.", [
        "Calcula las exactitudes de A y B.",
        "Calcula el estadístico de McNemar con corrección de continuidad y compáralo con 3.84.",
        "Calcula el valor p aproximado (χ², 1 gl). ¿Qué concluyes?"])
    s += bloque(6, "Interpretar intervalos que se solapan", "★★★", "AUC del modelo 1: IC [0.84; 0.92]; AUC del modelo 2: IC [0.88; 0.95]. Con los mismos casos y remuestras pareadas, el IC de la diferencia (2 − 1) es [0.005; 0.070].", [
        "¿Se solapan los dos IC? ¿Se puede concluir «no hay diferencia» por eso?",
        "¿Qué dice el IC de la diferencia pareada? ¿Y si fuera [−0.010; 0.080]?",
        "Con un IC de la diferencia [0.001; 0.003] y una mejora relevante de 0.010, ¿qué se concluye?"])
    s += bloque(7, "Probabilidad de quedar fuera de la remuestra", "★★", "Una muestra de prueba tiene n = 20 casos.", [
        "Calcula la probabilidad de que un caso concreto no aparezca en una remuestra de tamaño 20.",
        "Calcula el número esperado de casos distintos en la remuestra.",
        "Compara con el límite para n grande y con n = 171 (Wisconsin)."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C43))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C43), Spacer(1, 2)]

    ia, ib = [i - 1 for i in r1], [i - 1 for i in r2]
    ea, eb = ok8[ia].mean(), ok8[ib].mean()
    dist_a = sorted(set(r1))
    falt_a = [i for i in range(1, 9) if i not in dist_a]
    s += sol("Bloque 1", [
        ["1", f"Aciertos por ID: {', '.join(f'{i + 1}→{ok8[i]}' for i in range(8))}. Exactitud = {ok8.sum()}/8 = {f(ok8.mean(), 3)}."],
        ["2", f"Remuestra A: aciertos {', '.join(str(ok8[i]) for i in ia)} → {ok8[ia].sum()}/8 = {f(ea, 3)}. IDs distintos: {len(dist_a)} ({', '.join(map(str, dist_a))}); no aparecen {', '.join(map(str, falt_a))}."],
        ["3", f"Remuestra B: aciertos {', '.join(str(ok8[i]) for i in ib)} → {ok8[ib].sum()}/8 = {f(eb, 3)}. La misma muestra produce exactitudes de {f(ea, 3)}, {f(ok8.mean(), 3)} y {f(eb, 3)}: la métrica depende de qué casos entran, y el bootstrap mide esa variación."]])
    lo2, hi2 = bs40[2], bs40[-3]
    s += sol("Bloque 2", [
        ["1", f"Se descartan los 2 menores ({bs40[0]:.3f}, {bs40[1]:.3f}) y los 2 mayores ({bs40[-2]:.3f}, {bs40[-1]:.3f}). IC del 90 % = [{lo2:.3f}; {hi2:.3f}] (el 3.º menor y el 3.º mayor; <i>np.quantile</i> interpola y puede diferir en el tercer decimal)."],
        ["2", f"0.884 {'sí' if lo2 <= 0.884 <= hi2 else 'no'} está dentro. El 0.85 {'sí' if lo2 <= 0.85 <= hi2 else 'no'} está dentro: por tanto 0.85 queda {'dentro' if lo2 <= 0.85 <= hi2 else 'fuera'} de los valores razonables según esta remuestra."],
        ["3", "Correcto: el cálculo es estable (menos error Monte Carlo). Incorrecto: pensar que se sabe más sobre la población; la incertidumbre por tener solo n casos originales no baja con B."]])
    ah = 54 / 60
    se3 = sqrt(ah * (1 - ah) / 60)
    wl = stats.beta.ppf(0.025, 54, 7), stats.beta.ppf(0.975, 55, 6)
    s += sol("Bloque 3", [
        ["1", f"â = 54/60 = {f(ah, 2)}; SE = √(0.90 × 0.10/60) = {f(se3, 4)}. El bootstrap (0.040) es casi igual: coincide porque el modelo es fijo y los casos independientes."],
        ["2", f"IC normal = 0.90 ± 1.96 × {f(se3, 4)} = [{f(ah - 1.96 * se3, 3)}; {f(ah + 1.96 * se3, 3)}]. Con 60 casos y acierto cercano a 1 la aproximación normal es mala (la cota superior se acerca a 1); el IC exacto (Clopper-Pearson) es [{f(wl[0], 3)}; {f(wl[1], 3)}]."],
        ["3", "El SE es proporcional a 1/√n: para reducirlo a la mitad hay que cuadruplicar los casos: 240."]])
    dd = lA - lB
    se_p = dd.std(ddof=1) / sqrt(6)
    se_u = sqrt(lA.var(ddof=1) / 6 + lB.var(ddof=1) / 6)
    s += sol("Bloque 4", [
        ["1", f"Diferencias: {', '.join(f(x, 1) for x in dd)}. Media = {f(dd.mean(), 4)}; desviación (n − 1) = {f(dd.std(ddof=1), 4)}."],
        ["2", f"SE pareado = {f(dd.std(ddof=1), 4)}/√6 = {f(se_p, 4)}. Sin emparejar: desviaciones {f(lA.std(ddof=1), 4)} y {f(lB.std(ddof=1), 4)} → SE = √({f(lA.var(ddof=1), 4)}/6 + {f(lB.var(ddof=1), 4)}/6) = {f(se_u, 4)}."],
        ["3", f"El SE pareado es {f(se_u / se_p, 1)} veces menor: los casos difíciles suben la pérdida de ambos modelos; al restar, esa parte se cancela. Emparejar supone que los dos modelos se evalúan en los mismos casos."]])
    b5, c5 = 15, 5
    st5, p5, pe5 = mcnemar(b5, c5)
    s += sol("Bloque 5", [
        ["1", f"A: (80 + 15)/120 = {f(95 / 120, 4)}; B: (80 + 5)/120 = {f(85 / 120, 4)}."],
        ["2", f"b = 15 y c = 5: χ² = (|15 − 5| − 1)²/20 = 81/20 = {f(st5, 2)}. Es mayor que 3.84."],
        ["3", f"p ≈ {f(p5, 4)} (exacto {f(pe5, 4)}). Se rechaza «sin diferencia» al 5 %: A acierta más que B. La diferencia absoluta es {f(10 / 120, 3)}."]])
    s += sol("Bloque 6", [
        ["1", "Sí: [0.88; 0.92] es común. No se puede concluir «no hay diferencia»: los IC marginales no usan el emparejamiento."],
        ["2", "[0.005; 0.070] no contiene 0: el modelo 2 es mejor en esta evaluación. Con [−0.010; 0.080] contendría 0: no hay evidencia clara (no prueba equivalencia)."],
        ["3", "El IC excluye 0 (diferencia estadística), pero 0.003 es menor que la mejora relevante 0.010: es positiva y chica; en la práctica no justifica el cambio si la mejora relevante fijada es 0.010."]])
    p20 = p_fuera(20)
    s += sol("Bloque 7", [
        ["1", f"(1 − 1/20)²⁰ = 0.95²⁰ = {f(p20, 4)}."],
        ["2", f"20 × (1 − {f(p20, 4)}) = {f(20 * (1 - p20), 2)} casos distintos (≈ {f((1 - p20) * 100, 1)} %)."],
        ["3", f"El límite es 1/e = {f(exp(-1), 4)} (36.8 % fuera, 63.2 % dentro); con n = 171, {f(p_fuera(171), 4)}. Con n = 20 ya es muy parecido: {f(abs(p20 - exp(-1)), 4)} de diferencia."]])
    build("apuntes/C43_Ejercicios_Cuaderno_EvaluacionBootstrap.pdf", s, "Ejercicios para el cuaderno · Clase 43")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
