"""Clase 38 · Evaluación de modelos supervisados y selección de modelos: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c38_numeros.json (lo escribe figuras_c38.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import BLUE_BG, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c38_numeros.json").read_text(encoding="utf-8"))
C38 = colors.HexColor("#65A30D")
C38_BG = colors.HexColor("#F7FEE7")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas DipDS_2026_S38 + 38_evaluacion_seleccion.py / .R.txt del profesor Eloy Alvarado"
MT, PA, CV, RP, BO, SV, CU = N["metricas"], N["particion"], N["cv"], N["repetido"], N["bootstrap"], N["sesgo_var"], N["curvas"]
HO, DU, SP, CS, RC, CA, UM, AB, AN = N["hormigon"], N["dup"], N["spam"], N["costos"], N["roc_pr"], N["calibracion"], N["umbral"], N["aic_bic"], N["anidada"]
RK = {r["modelo"]: r for r in HO["ranking"]}


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{int(round(x)):,}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.72):
    s.append(band(f"Clase 38 · {titulo}", FUENTE, C38))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C38, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C38))
    s.append(Spacer(1, 1))
    for fig, pie in figuras:
        s.append(img(fig, width=CONTENT_W * fw))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C38)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 38 · Evaluación de modelos supervisados y selección de modelos",
                  "Profesor Eloy Alvarado · sigue a SVM y clasificadores clásicos (Clase 37) y precede a Naive Bayes (Clase 40) y al bootstrap (Clase 43)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("«Mejor modelo» no significa nada hasta que se define <b>qué error importa</b> (la pérdida y la métrica), <b>en qué casos nuevos</b> (la partición) y <b>contra qué referencia</b>. "
                    "El protocolo es siempre el mismo: fijar métrica y regla <b>antes</b> de ver resultados, comparar candidatos —cada uno una <b>receta completa</b>— con <b>validación cruzada</b> sin fuga de datos, "
                    "<b>reajustar</b> lo elegido con todo el entrenamiento y evaluar <b>una sola vez</b> en la prueba reservada. En clasificación hay un paso más: una <b>probabilidad</b> no es una <b>decisión</b>; "
                    "el umbral se elige según el costo de cada error.", "body")], bg=C38_BG, border=C38, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c38_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Partición y validación</b> (Clase 30): entrenamiento, validación, prueba, K-fold y fuga de datos",
                     "<b>Sobreajuste y regularización</b> (Clases 31 y 33): ridge y lasso; sesgo y varianza",
                     "<b>Modelos</b> (Clases 34–37): regresión lineal y logística, bosques aleatorios y SVM como candidatos",
                     "<b>Matriz de confusión, precision y recall</b> (Clase 37) e inferencia (Clase 22)"], "cell", C38)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Qué es mejor</b> → <b>pérdida y métricas</b> → <b>pliegues y OOF</b> → <b>sesgo y varianza</b> → <b>elegir y reajustar</b> → <b>probabilidad, ROC/PR, calibración y umbral</b>",
                      f"<b>Hormigón</b> (regresión): {mil(PA['filas'])} mediciones, 8 predictores, resistencia en MPa; <b>spam</b> (clasificación): 4 601 correos, 57 predictores",
                      "Pregunta guía: ¿cuánto nos equivocaremos con datos que el modelo no vio, y cómo decidimos con honestidad?"], "cell", C38)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con las clases vecinas (sin repetirlas)</b>", "cellb"),
                  P("<b>C30 → C38</b>: allí se aprendió a dividir y a validar; hoy se usa para <i>decidir</i> entre candidatos y se agrega la partición por grupos. "
                    "<b>C31 → C38</b>: el sobreajuste de la Clase 31 se descompone en sesgo, varianza y ruido. <b>C34–C37 → C38</b>: árboles, bosques y SVM son solo candidatos; la evaluación es agnóstica al modelo. "
                    "<b>C22 → C38</b>: la prueba reservada es un experimento: su cifra tiene incertidumbre (intervalo bootstrap). Lo nuevo de hoy: <b>validación anidada, bootstrap/.632, ROC y PR, calibración, umbral por costo y AIC/BIC</b>.", "cell")], bg=SOFT, border=C38, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: ¿qué es una receta?</b>", "cellb"),
                  P("Un candidato no es «una regresión»: es <b>todo</b> lo que se hace desde las columnas hasta la predicción (estandarizar, expandir términos, penalizar, umbral). "
                    "Por eso se arma con un <font name='Mono'>Pipeline</font>: en cada pliegue se aprende con las filas de ajuste y se aplica a las de validación, y así no hay fuga.", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Métricas y validación sin fuga",
        "Una métrica resume el error con una <b>pérdida</b>; la partición representa los <b>casos nuevos</b>. En el hormigón se reservan <b>composiciones completas</b> (no filas), porque la misma mezcla aparece con varias edades.",
        [("MAE, RMSE y R²", "MAE promedia |error| (cada MPa pesa igual); RMSE promedia el cuadrado y luego saca raíz (castiga los errores grandes); R² = 1 − SSE/SST compara con la media y <b>puede ser negativo</b>."),
         ("Referencia primero", f"El modelo constante (media de entrenamiento) obtiene RMSE OOF {f(RK['MEDIA']['rmse'])}: es lo que aportan los predictores. Un candidato vale por cuánto mejora a su referencia."),
         ("Pliegues por grupo y OOF", f"{PA['grupos']} composiciones: {PA['grupos_tr']} para ajustar (5 pliegues) y {PA['grupos_te']} para prueba. Cada fila recibe una predicción <b>fuera de pliegue</b>; con 7 candidatos son {CV['ajustes']} ajustes.")],
        [("c38_metricas.png", f"Izquierda: un error de 4 pesa 4 con MAE y 16 con la pérdida cuadrática. Centro: el caso A (MAE {f(MT['a_mae'], 0)}, RMSE {f(MT['a_rmse'], 0)}) gana con MAE y el B (MAE {f(MT['b_mae'], 0)}, RMSE {f(MT['b_rmse'], 0)}) gana con RMSE. Derecha: correlación 1 y R² = {f(MT['r2_corr'], 0)}."),
         ("c38_oof.png", f"Izquierda: la prueba ({PA['filas_te']} filas) se aparta y el entrenamiento se divide en 5 pliegues de composiciones. Centro: RMSE por pliegue de α = 0.1 y α = 1 (RMSE OOF {f(CV['rmse_oof_01'], 3)}). Derecha: con MSE de pliegues 1 y 9, promediar RMSE da 2 y el RMSE OOF da √5 = 2.236.")],
        ["MAE = (1/n) Σ |yᵢ − ŷᵢ|",
         "RMSE = √[(1/n) Σ (yᵢ − ŷᵢ)²]",
         "R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²",
         "RMSE OOF = √[(1/806) Σ (yᵢ − ŷᵢ,OOF)²]"],
        ["rmse = np.sqrt(np.mean((y - oof) ** 2))", "mae = np.mean(np.abs(y - oof))", "# oof[valida] = modelo.fit(X[~valida], y[~valida]).predict(X[valida])"],
        [f"RMSE OOF: α = 0.1 → {f(RK['Q_A0.1']['rmse'], 3)}; α = 1 → {f(RK['Q_A1']['rmse'], 3)}; lineal → {f(RK['LINEAL']['rmse'], 3)}; media → {f(RK['MEDIA']['rmse'], 3)} MPa",
         f"Por pliegue, α = 0.1 gana en {CV['gana_01_pliegues']} de 5; promedio de RMSE {f(CV['prom_rmse_01'], 3)} frente a RMSE OOF {f(CV['rmse_oof_01'], 3)} (pliegues de 161–162 filas)",
         "Un R² ≈ 0.7 no es «bueno» ni «malo» en abstracto: depende del campo (ciencias sociales frente a ingeniería)"],
        ["No se promedian los RMSE de los pliegues como si fueran datos independientes; y si se estandariza con todas las filas, la fuga ya ocurrió antes del primer pliegue."])

    topic_page(
        s, "La validación también varía: particiones repetidas, K y bootstrap",
        "El RMSE OOF es una <b>estimación</b>: cambia si cambia la partición, y depende de cuántos datos se usan para ajustar. Repetir la partición y remuestrear (bootstrap) permite ver <b>cuánto</b>.",
        [("Particiones repetidas", f"Con {RP['R']} particiones al azar de las mismas composiciones, el RMSE OOF de α = 0.1 va de {f(RP['min']['0.1'])} a {f(RP['max']['0.1'])} (media {f(RP['media']['0.1'])}, DE {f(RP['sd']['0.1'])}). Como los candidatos comparten pliegues, la <i>diferencia</i> es más estable que cada cifra."),
         ("K, LOOCV y sesgo", f"K = 2 entrena con la mitad y es pesimista ({f(RP['sweep']['2'][0])}); K = 5 da {f(RP['sweep']['5'][0])}, K = 20 {f(RP['sweep']['20'][0])} y dejar un grupo fuera ({RP['n_grupos']} ajustes) {f(RP['logo'])}. K grande: menos sesgo, más costo."),
         ("Bootstrap y .632", f"Cada remuestra de composiciones deja ≈ 36.8 % fuera (OOB). El error OOB es pesimista y el de ajuste, optimista; .632 los mezcla: 0.368·(ajuste) + 0.632·(OOB) en MSE → RMSE {f(BO['rmse_632'])}.")],
        [("c38_cv_repetido.png", f"Izquierda: {RP['R']} particiones al azar para α = 0.1, 1 y 10 (rayas: medias). Centro: α = 0.1 gana en {f(RP['p_gana_01'] * 100, 0)} % de las particiones, pero por solo ≈ {f(abs(RP['dif_media']), 2)} MPa. Derecha: RMSE OOF según K (barras: DE de 12 particiones) y dejando un grupo fuera."),
         ("c38_bootstrap.png", f"Izquierda: fracción de composiciones que entran a la remuestra (≈ 0.632). Centro: cinco estimaciones del error de la misma receta ({f(BO['rmse_resub'])} a {f(BO['rmse_test'])} MPa). Derecha: bootstrap de las {PA['grupos_te']} composiciones de prueba: RMSE {f(BO['rmse_test'])}, IC 95 % [{f(BO['ci_test'][0])}; {f(BO['ci_test'][1])}].")],
        ["fracción OOB = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368",
         "MSE.632 = 0.368·MSE ajuste + 0.632·MSE OOB",
         "error estándar de un promedio ≈ DE / √K (con cautela: no son independientes)"],
        ["g = np.unique(grupos)", "muestra = rng.choice(g, len(g), replace=True)", "fuera = np.setdiff1d(g, muestra)   # OOB"],
        [f"Prueba: RMSE {f(BO['rmse_test'])}; la media de entrenamiento obtiene {f(BO['rmse_base_test'])}; la reducción {f(HO['mejora_rmse'] * 100, 1)} % tiene IC 95 % de la diferencia de RMSE [{f(BO['ci_dif'][0])}; {f(BO['ci_dif'][1])}] MPa",
         f"CV 5 pliegues {f(BO['rmse_cv'], 3)}, bootstrap OOB {f(BO['rmse_oob'], 3)}, .632 {f(BO['rmse_632'], 3)}, prueba {f(BO['rmse_test'], 3)}: ninguna es «la» verdad, todas son estimaciones"],
        ["Un intervalo de bootstrap con solo 86 composiciones es ancho; con casi-duplicados entre entrenamiento y prueba, además, la cifra es optimista (ver alertas)."])

    topic_page(
        s, "Sesgo, varianza y complejidad",
        "El error de predicción se descompone en <b>sesgo²</b> (qué tan lejos queda el ajuste promedio de la señal), <b>varianza</b> (cuánto cambia el ajuste con la muestra) y <b>ruido</b> (inevitable). Más flexibilidad baja el sesgo y sube la varianza.",
        [("Error de ajuste ≠ error nuevo", f"El error de ajuste siempre baja al añadir flexibilidad; el de datos nuevos hace una U. En la simulación el grado 12 ajusta mejor ({f(SV['uno']['12']['train'])}) y predice peor ({f(SV['uno']['12']['test'])}) que el grado 2 ({f(SV['uno']['2']['train'])} y {f(SV['uno']['2']['test'])})."),
         ("Qué diagnostica cada curva", "Curva de aprendizaje: si ajuste y validación quedan altos y juntos → sesgo alto (añadir flexibilidad); si hay brecha grande → varianza alta (más datos o más penalización)."),
         ("Curva de validación", f"En la cuadrícula del profesor (α ≥ 0.1) el mínimo está en el borde; bajar α hasta 10⁻⁴ solo mejora de {f(CU['rmse_oof_01'], 3)} a {f(CU['rmse_min'], 3)} MPa: ganancia práctica nula.")],
        [("c38_sesgo_varianza.png", f"Simulación (n = {SV['n']}, σ = {SV['sigma']}, {SV['reps']} muestras por grado). Izquierda: un conjunto de datos con grado 2 y 12. Centro: la descomposición. Derecha: el mejor grado esperado es {SV['mejor_grado']}; el error de ajuste sigue bajando."),
         ("c38_curvas.png", "Izquierda y centro: curvas de aprendizaje del hormigón (error según filas de ajuste): la lineal se estanca alto (sesgo), la cuadrática sigue mejorando. Derecha: ajuste y validación según α; la zona gris es la cuadrícula del profesor.")],
        ["E[(Y − f̂(x))²] = Sesgo² + Varianza + σ²",
         "Sesgo = E[f̂(x)] − f(x)",
         "Varianza = Var[f̂(x)]",
         "ruido σ² = Var(Y | X = x): no se reduce"],
        ["from sklearn.model_selection import learning_curve", "# learning_curve(modelo, X, y, groups=g, cv=GroupKFold(5), train_sizes=[.2, .5, 1.0])"],
        [f"Curva de aprendizaje de la cuadrícula: con {CU['n_entrena'][-1]} filas la cuadrática valida en {f(CU['cuad']['valid'][-1])} MPa y ajusta en {f(CU['cuad']['train'][-1])}; la lineal valida en {f(CU['lineal']['valid'][-1])} y ajusta en {f(CU['lineal']['train'][-1])} (sesgo alto)",
         f"Con α = 1000 sube el error de ajuste ({f(RK['Q_A1000']['rmse'])} RMSE OOF) porque la penalización restringe demasiado: sesgo alto otra vez"],
        ["La descomposición vale para un x fijo y ruido de media cero independiente del ajuste; con datos reales solo se estima su efecto conjunto con validación. Una U no está garantizada."])

    topic_page(
        s, "Hormigón: elegir, reajustar y evaluar una vez",
        "Siete <b>recetas completas</b> se comparan con los mismos cinco pliegues. Se elige con una regla fijada antes (menor RMSE OOF), se <b>reajusta</b> con las 806 filas y se evalúa <b>una vez</b> en las 224 reservadas.",
        [("Elegir con OOF", f"Gana la cuadrática con ridge α = 0.1 ({f(RK['Q_A0.1']['rmse'], 3)}); α = 1 queda a {f(HO['ventaja_01_1'], 3)} MPa. MAE y R² describen pero no cambian la regla."),
         ("Reajustar, no promediar", "Se vuelven a estimar escalas y coeficientes con todo el entrenamiento. No se promedian los cinco modelos de la CV: sería otro procedimiento."),
         ("Prueba y diagnóstico", f"RMSE {f(HO['prueba']['rmse'], 3)}, MAE {f(HO['prueba']['mae'], 3)}, R² {f(HO['prueba']['r2'], 3)}; la media da {f(HO['referencia']['rmse'], 3)}. Los residuos revelan dónde falla lo que el promedio esconde.")],
        [("c38_hormigon.png", f"Izquierda: ranking OOF de los siete candidatos (RMSE elige; MAE describe). Centro: diferencia por pliegue entre α = 0.1 y 1. Derecha: predicción frente a resistencia real en prueba ({HO['n_prueba']} mediciones)."),
         ("c38_diagnostico.png", f"Residuos y errores según la edad; a la derecha, el RMSE de prueba según qué tan cerca está cada composición de alguna del entrenamiento: {DU['n_cerca']} filas ({f(DU['pct_cerca'] * 100, 0)} %) son casi-duplicados y tienen RMSE {f(DU['rmse_cerca'], 1)}.")],
        ["m̂ = argmin_m  R̂_CV(m)",
         "reajustar: m̂ con las 806 filas",
         "evaluar: RMSE, MAE y R² en las 224 filas de prueba",
         "mejora = 1 − RMSE_modelo / RMSE_media"],
        ["ganador = ranking.modelo.iloc[0]", "final = clone(modelos[ganador]).fit(X_tr, y_tr)", "pred = final.predict(X_te)"],
        [f"Prueba: {f(HO['prueba']['rmse'], 3)} MPa frente a {f(HO['referencia']['rmse'], 3)} de la media: baja {f(HO['mejora_rmse'] * 100, 1)} %; R² de la media ≈ {f(HO['referencia']['r2'], 3)}",
         f"El RMSE de entrenamiento del ajuste final es {f(HO['rmse_train_final'], 2)}: más bajo que en validación y prueba, como se espera",
         f"Residuos: promedio {f(DU['media_res'], 2)} MPa y {f(DU['pos'] * 100, 0)} % positivos (subestimación); error máximo {f(DU['max_abs'], 1)} MPa"],
        ["Que gane por 0.079 MPa no prueba una superioridad estable; y si el resultado de prueba no gusta, no se vuelve a elegir con ella: se diseña una nueva selección con evaluación independiente."])

    topic_page(
        s, "Clasificación: probabilidades, costo y ranking",
        "Un clasificador entrega una <b>probabilidad</b>; la <b>decisión</b> exige un umbral. Se evalúan tres cosas distintas: ¿ordena bien? (ROC, AP), ¿estima bien la probabilidad? (log loss, Brier, calibración) y ¿decide bien? (costo).",
        [("Accuracy no basta", f"«Siempre no spam» acierta 95 % de 100 correos con 5 % de spam y detecta 0. Dos matrices con el mismo 95 % tienen costos {f(CS['A']['costo'], 0)} y {f(CS['B']['costo'], 0)} por 1 000 con FP = 5 y FN = 1."),
         ("Log loss y Brier", f"Log loss = −ln p de la clase verdadera (p = 0.9 → {f(SP['log_loss_1'], 3)}; p = 0.1 → {f(SP['log_loss_01'], 3)}). Brier = (y − p)². Ambos premian probabilidades honestas; se elige por log loss OOF."),
         ("ROC y precision–recall", "Cada punto de ambas curvas es un umbral. ROC: recall frente a tasa de FP; PR: precision frente a recall. AUC y AP resumen el ordenamiento; AP depende de la proporción de positivos.")],
        [("c38_spam.png", f"Izquierda: la referencia es la frecuencia de spam ({f(SP['prev'] * 100, 1)} %). Centro: log loss y Brier OOF de los tres candidatos. Derecha: cómo castigan una probabilidad equivocada."),
         ("c38_roc_pr.png", f"ROC con AUC {f(RC['auc']['BOSQUE_8'], 3)} (bosque) y puntos para umbrales 0.50, 0.60 y 5/6; precision–recall con AP {f(RC['ap']['BOSQUE_8'], 3)}; a la derecha, las distribuciones OOF de legítimos y spam que el umbral corta.")],
        ["precision = VP / (VP + FP);  recall = VP / (VP + FN)",
         "F1 = 2·precision·recall / (precision + recall)",
         "costo = (5·FP + 1·FN) / n  (×1 000)",
         "LogLoss = −(1/n) Σ [y ln p + (1 − y) ln(1 − p)]"],
        ["log_loss(y, p); brier_score_loss(y, p)", "roc_auc_score(y, p); average_precision_score(y, p)", "confusion_matrix(y, p >= t, labels=[0, 1])"],
        [f"OOF (3 680 correos): log loss frecuencia {f(SP['ranking']['FRECUENCIA']['log_loss'], 3)}, logística {f(SP['ranking']['LOGISTICA_C1']['log_loss'], 3)}, bosque {f(SP['ranking']['BOSQUE_8']['log_loss'], 3)}; Brier {f(SP['ranking']['FRECUENCIA']['brier'], 3)}, {f(SP['ranking']['LOGISTICA_C1']['brier'], 3)}, {f(SP['ranking']['BOSQUE_8']['brier'], 3)}",
         f"AUC bosque {f(RC['auc']['BOSQUE_8'], 3)}, AP {f(RC['ap']['BOSQUE_8'], 3)}; una AUC de 0.975 no es «97.5 % de aciertos»",
         f"F1 prefiere B ({f(CS['B']['f1'], 3)} frente a {f(CS['A']['f1'], 3)}) pero el costo prefiere A: F1 no usa la razón 5:1"],
        ["La frecuencia de spam cambia la precision: misma sensibilidad 80 % y FP 10 %, con 50 % de spam es 88.9 % y con 5 % es 29.6 %. Una precision medida en un corpus no se traslada a otro."])

    topic_page(
        s, "Calibración y umbral por costo",
        "Un buen ranking no garantiza buenas probabilidades. La <b>calibración</b> compara lo que el modelo dice con lo que ocurre, y el <b>umbral</b> convierte probabilidades en acciones minimizando un costo declarado antes.",
        [("Calibración", f"Entre correos con probabilidad ≈ 0.8 esperamos ≈ 80 % de spam. El bosque, en su 4.º quintil, predice {f(CA['q4_prob'] * 100, 1)} % y observa {f(CA['q4_frac'] * 100, 1)} %: subestima."),
         ("Umbral empírico y teórico", f"Con costos conocidos y probabilidades correctas se bloquea si p > c_FP/(c_FP + c_FN) = 5/6 ≈ 0.833. El óptimo empírico del bosque es {f(UM['t_min'], 2)} (costo OOF {f(UM['costo_min'], 1)}), lejos de 0.833 por la mala calibración."),
         ("Recalibrar es otra receta", f"Con calibración isotónica anidada, el óptimo sube a {f(UM['t_min_cal'], 2)} (cerca de la teoría), mejora el Brier ({f(CA['metricas']['BOSQUE_8']['brier'], 3)} → {f(CA['metricas']['BOSQUE_CAL']['brier'], 3)}), pero el costo OOF no baja ({f(UM['costo_min_cal'], 1)}).")],
        [("c38_calibracion.png", "Izquierda: curva de calibración del bosque en cinco grupos (sobre la diagonal, subestima). Centro: logística, bosque y bosque recalibrado en diez grupos. Derecha: log loss y Brier antes y después de recalibrar (todo anidado, sin mirar el pliegue de validación)."),
         ("c38_umbral.png", f"Izquierda: costo OOF por 1 000 correos según el umbral (mínimo del bosque en {f(UM['t_min'], 2)}). Centro: umbral óptimo según la razón de costos, con la regla teórica r/(1 + r). Derecha: de 0.50 a 0.60 hay {UM['ahorro_fp']} FP menos y {UM['mas_fn']} FN más.")],
        ["bloquear si p > c_FP / (c_FP + c_FN)",
         "costo esperado de bloquear = c_FP·(1 − p)",
         "costo esperado de dejar pasar = c_FN·p",
         "umbral teórico con 5:1 = 5/6 ≈ 0.833"],
        ["CalibratedClassifierCV(modelo, method='isotonic', cv=3)", "costo = (5 * fp + fn) / n   # por 1000: × 1000"],
        [f"Costos OOF por 1 000: t = 0.50 → {f(UM['c050'], 1)}; 0.55 → {f(UM['c055'], 1)}; 0.60 → {f(UM['c060'], 1)}; 0.65 → {f(UM['c065'], 1)}",
         f"De 0.50 a 0.60: −{UM['ahorro_fp']} FP, +{UM['mas_fn']} FN; ahorro 5×{UM['ahorro_fp']} − {UM['mas_fn']} = {UM['ahorro_unid']} unidades aunque los aciertos bajan de {f(UM['m05']['accuracy'] * 100, 1)} % a {f(UM['m06']['accuracy'] * 100, 1)} %",
         f"Prueba ({UM['n_prueba']} correos): t = 0.60 → aciertos {f(UM['prueba']['accuracy'] * 100, 1)} %, precision {f(UM['prueba']['precision'] * 100, 1)} %, recall {f(UM['prueba']['recall'] * 100, 1)} %, costo {f(UM['prueba']['costo'] * 1000, 1)}; «siempre no spam» cuesta {f(UM['referencia']['costo'] * 1000, 1)}"],
        ["La razón 5:1 es una prioridad hipotética elegida antes, no un costo medido; si cambia, cambia el umbral. Y no se reemplaza el umbral después de mirar la prueba."])

    topic_page(
        s, "AIC, BIC y validación anidada",
        "Hay dos maneras de penalizar la complejidad: con una <b>fórmula</b> (AIC, BIC) usando solo las filas de ajuste, o con <b>validación</b> (OOF). Y cuando además se <b>elige</b> con validación, el mejor puntaje obtenido es optimista: la <b>validación anidada</b> lo corrige.",
        [("AIC y BIC", f"AIC = −2 log L̂ + 2k; BIC = −2 log L̂ + k·ln n. Con n = {AB['n']} cada parámetro cuesta 2 en AIC y {f(AB['penal_bic'], 2)} en BIC. Menor es mejor y solo comparan modelos con los mismos datos."),
         ("Cuándo coinciden con CV", f"En el hormigón (OLS) el AIC elige {AB['p_aic']} términos, la CV {AB['p_cv']} y el BIC {AB['p_bic']}. No miran datos nuevos y suponen filas independientes."),
         ("Validación anidada", f"Bucle interno: elige hiperparámetros con las filas de ajuste; bucle externo: mide. Con etiquetas al azar, el mejor puntaje de CV entre {AN['candidatos']} candidatos es {f(AN['cv_mejor'], 3)}; la anidada da {f(AN['anidada'], 3)} (la verdad: 0.5).")],
        [("c38_aic_bic.png", f"Izquierda: AIC y BIC (restados de su mínimo) según los términos del modelo OLS. Centro: RMSE OOF de los mismos modelos. Derecha: penalización de AIC (2k) y BIC (k·ln n)."),
         ("c38_anidada.png", f"Izquierda: los dos bucles. Centro: {AN['reps']} simulaciones sin señal alguna. Derecha: hormigón con {AN['rejilla']} configuraciones (grado × α): el mejor RMSE interno ({f(AN['hor_interna_media'])}) es optimista frente al externo ({f(AN['hor_externa_media'])}).")],
        ["AIC = n·ln(SSE/n) + 2k + const",
         "BIC = n·ln(SSE/n) + k·ln(n) + const",
         f"ajustes: simple = pliegues × candidatos; anidada = externos × internos × candidatos + externos"],
        ["GridSearchCV(pipe, rejilla, cv=GroupKFold(5))", "cross_val_score(grid, X, y, groups=g, cv=GroupKFold(5))   # anidada"],
        [f"AIC en los extremos: nulo {f(AB['aic_nulo'], 0)}, lineal (8 términos) {f(AB['aic_lineal'], 0)}, cuadrático (16) {f(AB['aic_cuad'], 0)}, completo (44) {f(AB['aic_completo'], 0)}; BIC mínimo en {AB['p_bic']} términos ({f(AB['bic_min'], 0)})",
         f"RMSE OOF: lineal {f(AB['cv_lineal'], 2)}, cuadrático sin penalizar {f(AB['cv_cuad'], 2)}, mínimo {f(AB['cv_min'], 2)} con {AB['p_cv']} términos",
         f"Hormigón anidada: {AN['ajustes_anidados']} ajustes (simple: {AN['ajustes_simple']}); los hiperparámetros elegidos cambian entre pliegues externos"],
        ["AIC y BIC con un n ficticio (filas repetidas) parecen más seguros de lo que son. Y la anidada estima el desempeño de <b>todo el procedimiento</b> de selección; el modelo que se despliega se vuelve a elegir con todo el entrenamiento."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase</b>", "cellb"),
                  bullets([f"<b>«El R² es un número entre 0 y 1»</b>: puede ser negativo, como el propio profesor mostró (correlación 1 y R² = {f(MT['r2_corr'], 0)}); la media de entrenamiento da R² ≈ {f(HO['referencia']['r2'], 3)} en prueba.",
                           "<b>«El ajuste … lo voy a realizar con el conjunto de prueba»</b> (sección del estandarizado): debe ser con el conjunto de <b>entrenamiento</b>; con prueba habría fuga.",
                           "<b>«Voy a ocupar la pérdida cuadrática fuera de pliegue»</b> (clasificación): el script elige por <b>log loss</b> OOF; el Brier (error cuadrático de probabilidades) solo describe.",
                           f"<b>«Umbral 0.5 → costo 132; 0.6 → 128; 0.65 sube»</b>: los valores 132.3, 128.8 y 135.3 son de 0.55, 0.60 y 0.65; en 0.50 el costo es {f(UM['c050'], 1)}.",
                           f"<b>«α = 0.1 mejora en 2, 3 y 4; en el primero y el tercero gana α = 1»</b>: gana α = 1 en los pliegues <b>1 y 5</b>; α = 0.1 gana en 2, 3 y 4 ({f(CV['por_fold_01'][0], 3)} frente a {f(CV['por_fold_1'][0], 3)}, y {f(CV['por_fold_01'][4], 3)} frente a {f(CV['por_fold_1'][4], 3)}).",
                           f"<b>«342 composiciones que representan 806 mediciones de las 1 030»</b>: son las de <b>entrenamiento</b>; el total es {PA['grupos']} y {PA['grupos_te']} están en prueba. Y «10 de los 1 000 datos … 9 990 restantes» mezcla 1 000 con 10 000."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, scripts y verificación</b>", "cellb"),
                  bullets([f"<b>Grupos por igualdad exacta</b> no impiden casi-duplicados: {DU['n_cerca']} de 224 filas de prueba ({f(DU['pct_cerca'] * 100, 0)} %) tienen una composición del entrenamiento a menos de {f(DU['umbral'], 1)} desviaciones estándar y un RMSE de {f(DU['rmse_cerca'], 1)}; las demás, {f(DU['rmse_lejos'], 1)}. La cifra {f(HO['prueba']['rmse'], 2)} es válida para esta partición, no para mezclas realmente nuevas.",
                           f"<b>«Elegir no prueba superioridad estable»</b> (diap. 26 y 29) se confirma y se matiza: con {RP['R']} particiones α = 0.1 gana en {f(RP['p_gana_01'] * 100, 0)} %, pero por ≈ {f(abs(RP['dif_media']), 2)} MPa (< 1 %). El ranking es estable; la ventaja, trivial.",
                           f"<b>Umbral teórico (5/6) frente a empírico ({f(UM['t_min'], 2)})</b> (diap. 55 y 53): lo explica la calibración (4.º quintil: predice {f(CA['q4_prob'] * 100, 0)} %, observa {f(CA['q4_frac'] * 100, 0)} %). Recalibrado, el óptimo empírico es {f(UM['t_min_cal'], 2)}, pero el costo OOF no mejora ({f(UM['costo_min_cal'], 1)} frente a {f(UM['costo_min'], 1)}).",
                           f"<b>AIC y BIC</b> se nombran en clase pero no se calculan ni son OOF: aquí eligen {AB['p_aic']} y {AB['p_bic']} términos frente a {AB['p_cv']} de la CV, con n = 806 filas no independientes.",
                           "<b>R frente a Python</b> (diap. 50–57): el bosque difiere (log loss 0.228 y 0.223; costo a 0.60: 128.8 y 123.4) por implementación (<font name='Mono'>ranger</font> y scikit-learn), no por método; las cifras de este material son las de Python.",
                           "<b>Detalles</b>: la diap. 24 nombra <font name='Mono'>38_evaluacion_seleccion.R</font> y se entrega <font name='Mono'>.R.txt</font>; el script de Python escribe sus resultados en una carpeta <font name='Mono'>resultados</font> junto al material."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Transcripción con ruido</b>", "cellb"),
                  P("Whisper escribió «Breyer / Bright / Ryre» por <i>Brier</i>, «lock-lock» por <i>log loss</i>, «ROP / ROCK» por <i>ROC</i>, «reach / rich / Bridge» por <i>ridge</i>, «lazo» por <i>lasso</i>, «OUF» por <i>OOF</i>, «ORMCE / RMC» por <i>RMSE</i>, «revisión» por <i>regresión</i>, «avaliación / ligación» por <i>validación</i> y «los pies» por <i>los pliegues</i>. "
                    "El recreo (≈ 1:10:40 del video) quedó marcado en la transcripción.", "cell")], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Calculo MAE, RMSE y R² a mano y explico cuándo cada métrica cambia el ganador",
                           "Separo por grupos, armo pliegues OOF y distingo el promedio de RMSE del RMSE OOF",
                           "Diagnostico sesgo alto y varianza alta con curvas, y sé qué hacer en cada caso",
                           "Construyo puntos de una ROC a mano, calculo la AUC por trapecios y elijo un umbral por costo",
                           "Explico por qué la validación anidada y cuándo AIC/BIC y CV pueden discrepar",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Clase 40</b> (próxima del profesor): Naive Bayes y clasificación probabilística; <b>Clase 43</b>: evaluación estadística de modelos y bootstrap."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C38_Guia_EvaluacionModelos.pdf", s, "Clase 38 · Evaluación y selección de modelos")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#65A30D">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 38 · Evaluación y selección de modelos", "Para leer antes de los ejercicios · lectura de ~30 min", C38))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("La métrica es una decisión", "MAE, RMSE, R², log loss o costo miden cosas distintas; se fijan antes de ver resultados."),
                            ("Validar es predecir lo que el modelo no vio", "Pliegues por grupo, preprocesamiento dentro del Pipeline, prueba intacta hasta el final."),
                            ("OOF: una predicción honesta por fila", "RMSE OOF = raíz del promedio de todos los errores; no es el promedio de los RMSE."),
                            ("Sesgo, varianza y ruido", "Más flexibilidad: menos sesgo, más varianza; el ajuste siempre baja, lo nuevo hace una U."),
                            ("Probabilidad ≠ decisión", "ROC/PR miden el orden, log loss y calibración las probabilidades, el costo la decisión."),
                            ("Elegir también sobreajusta", "La validación anidada mide el procedimiento completo; AIC/BIC penalizan sin datos nuevos.")], C38)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C38_EvaluacionModelos.ipynb</font> o el cuaderno <font name='Mono'>C38_Ejercicios_Cuaderno_EvaluacionModelos.pdf</font>.", "small"))

    # 1 métricas
    s.append(h("1", "Pérdida y métricas: MAE, RMSE y R²"))
    s += para("El error de un caso es e = y − ŷ. Una métrica resume los errores con una función: MAE promedia |e| (cada unidad pesa igual), RMSE promedia e² y saca raíz (los errores grandes pesan más) y R² = 1 − SSE/SST compara con predecir la media. MAE y RMSE están en las unidades de y; R² no.")
    y = np.array([30, 25, 40, 35.0])
    yh = np.array([28, 29, 37, 38.0])
    e = y - yh
    sse, sst = float(np.sum(e ** 2)), float(np.sum((y - y.mean()) ** 2))
    s.append(ejemplo("cuatro mediciones: y = (30, 25, 40, 35) y ŷ = (28, 29, 37, 38)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Errores e = y − ŷ", "30−28, 25−29, 40−37, 35−38", "2, −4, 3, −3"],
               ["MAE", "(2 + 4 + 3 + 3)/4", f"{f(np.mean(abs(e)))}"],
               ["MSE y RMSE", "(4 + 16 + 9 + 9)/4 = 9.5; raíz", f"{f(float(np.sqrt(np.mean(e ** 2))), 3)}"],
               ["SSE y SST", "Σe² = 38; media de y = 32.5, Σ(y − ȳ)² = 125", "38 y 125"],
               ["R²", "1 − 38/125", f"{f(1 - sse / sst, 3)}"]],
              [44 * mm, 74 * mm, CONTENT_W - 118 * mm - 18]),
        P("Lectura: el RMSE (3.08) supera al MAE (3.00) porque hay errores desiguales; si todos fueran ±3, serían iguales. <b>Error común</b>: decir que un R² de 0.70 es «bueno» sin mirar el campo, o creer que R² no puede ser negativo (sí puede, si el modelo es peor que la media).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook y el Bloque 1 del cuaderno", [P("Calcula las tres métricas con NumPy y con <font name='Mono'>sklearn.metrics</font>, y reproduce el ejemplo de correlación 1 con R² = −149.", "cell")]))

    # 2 OOF
    s.append(h("2", "Pliegues, OOF y por qué no se promedian los RMSE"))
    s += para("Con validación cruzada, cada fila se predice con un modelo ajustado sin su pliegue (predicción <b>OOF</b>). El RMSE OOF reúne todos los errores al cuadrado y saca una sola raíz; el promedio de RMSE saca raíz en cada pliegue y luego promedia. Como la raíz no es lineal, no coinciden, y si los pliegues tienen distinto tamaño la diferencia crece. Todo el preprocesamiento va dentro del <font name='Mono'>Pipeline</font> para que en cada pliegue se aprenda solo con las filas de ajuste.")
    sizes, mses = np.array([10, 10, 20]), np.array([4.0, 9.0, 16.0])
    s.append(ejemplo("tres pliegues de 10, 10 y 20 filas con MSE 4, 9 y 16", [
        pasos([["Resumen", "Cálculo", "Resultado"],
               ["RMSE por pliegue", "√4, √9, √16", "2, 3 y 4"],
               ["Promedio de RMSE", "(2 + 3 + 4)/3", f"{f(3.0, 3)}"],
               ["Suma de cuadrados total", "10·4 + 10·9 + 20·16", f"{f(float(np.sum(sizes * mses)), 0)}"],
               ["RMSE OOF", "√(450/40)", f"{f(float(np.sqrt(np.sum(sizes * mses) / sizes.sum())), 3)}"]],
              [48 * mm, 70 * mm, CONTENT_W - 118 * mm - 18]),
        P("Lectura: el pliegue grande (20 filas, el de peor error) pesa más en el RMSE OOF, que por eso supera al promedio (3.354 frente a 3.000). <b>Error común</b>: calcular un «intervalo» con la desviación de los pliegues como si fueran réplicas independientes: comparten datos de entrenamiento.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c38_oof.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A2 y A3) y el Bloque 2 del cuaderno", [P("Calcula el RMSE OOF de los siete candidatos con tus propios pliegues y compáralo con el promedio de RMSE; después repite con particiones al azar.", "cell")]))

    # 3 sesgo-varianza
    s.append(h("3", "Sesgo, varianza y qué hacer en cada caso"))
    s += para("El error de predicción se descompone en sesgo² + varianza + ruido. Se diagnostica comparando error de ajuste y de validación: ambos altos y juntos → <b>sesgo alto</b> (modelo muy simple: añadir flexibilidad o variables); ajuste bajo y validación alta con brecha grande → <b>varianza alta</b> (más datos, más penalización o menos flexibilidad). El ruido no se reduce.")
    s.append(ejemplo("penalización ridge α con error de ajuste y de validación (RMSE)", [
        pasos([["α", "Ajuste", "Validación", "Brecha", "Diagnóstico"],
               ["0.01", "2.1", "4.8", "2.7", "varianza alta (sobreajuste)"],
               ["1", "2.9", "3.9", "1.0", "equilibrado: se elige"],
               ["100", "4.2", "4.5", "0.3", "sesgo alto (muy penalizado)"]],
              [16 * mm, 24 * mm, 26 * mm, 22 * mm, CONTENT_W - 88 * mm - 18]),
        P("Lectura: se elige α = 1 porque tiene el menor error de <b>validación</b> (3.9), no el menor de ajuste. <b>Error común</b>: elegir por error de ajuste: siempre favorece al modelo más flexible.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c38_sesgo_varianza.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C1 y C2) y el Bloque 3 del cuaderno", [P("Simula la descomposición con 400 muestras por grado y dibuja las curvas de aprendizaje del hormigón.", "cell")]))

    # 4 clasificación
    s.append(h("4", "Clasificación: matriz de confusión, costo y log loss"))
    s += para("La matriz de confusión cuenta VN, FP, FN y VP. Accuracy = (VN + VP)/n trata igual a todos los errores; el costo pondera: costo = (c_FP·FP + c_FN·FN)/n. Los costos se declaran <b>antes</b>. El log loss −ln p(clase verdadera) y el Brier (y − p)² evalúan las probabilidades, no solo la clase.")
    vn, fp, fn, vp = 150, 10, 15, 25
    n_ = vn + fp + fn + vp
    s.append(ejemplo("200 correos: VN = 150, FP = 10, FN = 15, VP = 25; costos FP = 3 y FN = 1", [
        pasos([["Métrica", "Cálculo", "Resultado"],
               ["Accuracy", "(150 + 25)/200", f"{f((vn + vp) / n_ * 100, 1)} %"],
               ["Precision y recall", "25/35 y 25/40", f"{f(vp / (vp + fp), 3)} y {f(vp / (vp + fn), 3)}"],
               ["F1", "2·25/(2·25 + 10 + 15)", f"{f(2 * vp / (2 * vp + fp + fn), 3)}"],
               ["Costo por 1 000", "(3·10 + 15)/200·1000", f"{f((3 * fp + fn) / n_ * 1000, 0)}"],
               ["Referencia «siempre no spam»", "accuracy 160/200; costo 40/200·1000", f"80 % y {f(40 / n_ * 1000, 0)}"]],
              [50 * mm, 70 * mm, CONTENT_W - 120 * mm - 18]),
        P("Lectura: el modelo acierta más (87.5 % frente a 80 %) pero cuesta <b>más</b> (225 frente a 200) porque sus 10 FP pesan 3 cada uno: con estos costos no mejora a la regla trivial. <b>Error común</b>: elegir por accuracy cuando los errores no cuestan lo mismo.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c38_costos.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B1 a B3) y el Bloque 5 del cuaderno", [P("Calcula log loss y Brier a mano para tres correos y comprueba con scikit-learn; luego calcula el costo de dos reglas.", "cell")]))

    # 5 ROC
    s.append(h("5", "ROC, AUC y umbral"))
    s += para("Para cada umbral t se marca spam si p ≥ t y se obtiene un punto (tasa de FP, recall). Al bajar t se suben ambos. La curva ROC los une; el AUC es el área y equivale a la probabilidad de que un positivo reciba mayor puntaje que un negativo. La curva precision–recall usa precision y recall y depende de la proporción de positivos. El umbral que minimiza el costo con probabilidades correctas es c_FP/(c_FP + c_FN).")
    s.append(ejemplo("tres spam con puntajes 0.9, 0.6, 0.4 y tres legítimos con 0.7, 0.3, 0.2", [
        pasos([["Umbral t", "VP", "FP", "Recall", "FPR"],
               ["0.9", "1", "0", "1/3", "0"],
               ["0.7", "1", "1", "1/3", "1/3"],
               ["0.6", "2", "1", "2/3", "1/3"],
               ["0.4", "3", "1", "1", "1/3"],
               ["0.3", "3", "2", "1", "2/3"],
               ["0.2", "3", "3", "1", "1"]],
              [30 * mm, 24 * mm, 24 * mm, 30 * mm, CONTENT_W - 108 * mm - 18]),
        P("AUC por trapecios: 1/3·(1/3) + 1·(1/3) + 1·(1/3) = 7/9 ≈ 0.778. Por pares: 0.9 supera a 3 legítimos, 0.6 a 2 y 0.4 a 2: 7 de 9. <b>Error común</b>: leer una AUC de 0.78 como «78 % de aciertos» (resume el orden, no fija ningún umbral).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c38_roc_pr.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B4 a B6) y el Bloque 4 del cuaderno", [P("Construye la ROC con <font name='Mono'>roc_curve</font>, marca los umbrales 0.5, 0.6 y 5/6 y busca el mínimo de costo en una cuadrícula.", "cell")]))

    # 6 AIC BIC y anidada
    s.append(h("6", "AIC, BIC y validación anidada"))
    s += para("Con errores normales, AIC = n·ln(SSE/n) + 2k y BIC = n·ln(SSE/n) + k·ln n (más una constante que no cambia la comparación); k cuenta los coeficientes. Gana el menor. El BIC penaliza más cuando n &gt; 7. La validación anidada usa un bucle interno para elegir hiperparámetros y uno externo para medir; así el desempeño informado no se infla por haber elegido el mejor de varios.")
    n6 = 50
    mods = [("M1", 2, 60.0), ("M2", 4, 45.0)]
    filas = [["Modelo", "k", "SSE", "AIC", "BIC"]]
    for nm_, k_, sse_ in mods:
        filas.append([nm_, str(k_), f"{sse_:.0f}", f"{n6 * np.log(sse_ / n6) + 2 * k_:.2f}", f"{n6 * np.log(sse_ / n6) + k_ * np.log(n6):.2f}"])
    s.append(ejemplo("n = 50: M1 con k = 2 y SSE = 60; M2 con k = 4 y SSE = 45", [
        pasos(filas, [28 * mm, 20 * mm, 24 * mm, 32 * mm, CONTENT_W - 104 * mm - 18]),
        P("Lectura: ΔAIC = 10.4 y ΔBIC = 6.6 a favor de M2: bajar la SSE de 60 a 45 compensa los 2 parámetros extra. <b>Anidada</b>: con 5 pliegues externos, 4 internos y 6 candidatos hay 5·4·6 + 5 = 125 ajustes, frente a 5·6 + 1 = 31 de una búsqueda simple. <b>Error común</b>: informar el mejor puntaje interno como si fuera el desempeño esperado.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c38_anidada.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook (C4 a C6) y los Bloques 6 y 7 del cuaderno", [P("Calcula AIC y BIC con <font name='Mono'>statsmodels</font>, compara con la CV y mide el optimismo con una validación anidada.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Métrica, referencia y regla de elección se fijan antes de ver resultados.",
                       "Preprocesamiento dentro del Pipeline; pliegues por grupo cuando hay filas relacionadas.",
                       "Una cifra de validación o de prueba es una estimación con incertidumbre.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    build("apuntes/C38_Apuntes_EvaluacionModelos.pdf", s, "Apuntes · Clase 38 · Evaluación y selección de modelos")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#65A30D">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C38))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 38 · Evaluación y selección de modelos", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C38))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: MAE = Σ|e|/n · RMSE = √(Σe²/n) · R² = 1 − SSE/SST · RMSE OOF = √(Σ todos los e²/n) · costo = (c_FP·FP + c_FN·FN)/n · AIC = n ln(SSE/n) + 2k · BIC = n ln(SSE/n) + k ln n.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C38)], bg=SOFT, border=C38, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "RMSE, MAE y R² a mano", "★★", "Cinco mediciones de resistencia: y = (12, 15, 9, 20, 14) y predicciones ŷ = (10, 16, 12, 17, 15).", [
        "Calcula los errores y el MAE.",
        "Calcula el RMSE. ¿Es mayor o menor que el MAE? ¿Por qué?",
        "Calcula R² con la media de y y con la media de entrenamiento 13 como referencia.",
        "Si el error del cuarto caso fuera 6 en vez de 3 (ŷ₄ = 14), ¿qué métrica sube más, MAE o RMSE? Calcúlalas."])
    s += bloque(2, "Pliegues, media y desviación", "★★", "Una validación cruzada de 4 pliegues con 25 filas cada uno entrega los RMSE: 2, 3, 4 y 5.", [
        "Calcula el promedio de RMSE y su desviación estándar muestral.",
        "Calcula el RMSE OOF reuniendo los errores. ¿Coincide con el promedio? ¿Por qué?",
        "Si el último pliegue tuviera 75 filas y los otros 25, ¿cuánto vale ahora el RMSE OOF?",
        "¿Por qué la desviación entre pliegues no es un intervalo de confianza?"])
    s += bloque(3, "Sesgo y varianza leídos en las curvas", "★★★", "Error de ajuste y de validación (RMSE) de polinomios de distinto grado: grado 1: 6.0 y 6.4; grado 3: 3.1 y 3.6; grado 6: 2.0 y 4.1; grado 10: 0.8 y 5.9; grado 15: 0.1 y 11.0.", [
        "¿Qué grado eliges y por qué?",
        "Diagnostica el grado 1 y el grado 15 (sesgo alto o varianza alta).",
        "Calcula la brecha (validación − ajuste) de cada grado. ¿Qué patrón ves?",
        "Para cada diagnóstico, propón una acción: más datos, más flexibilidad, penalización, menos variables."])
    s += bloque(4, "ROC a mano y AUC por trapecios", "★★★", "Ocho correos con puntaje de spam y etiqueta real (1 = spam): (0.95; 1), (0.85; 1), (0.80; 0), (0.70; 1), (0.55; 0), (0.45; 1), (0.30; 0), (0.10; 0).", [
        "Con el umbral bajando por cada puntaje, escribe el par (tasa de FP, recall) de cada paso.",
        "Calcula el AUC con trapecios.",
        "Cuenta los pares (positivo, negativo) bien ordenados y comprueba que da el mismo AUC.",
        "¿Qué significa un AUC de 0.8125? ¿Dice cuántos correos se clasifican bien?"])
    s += bloque(5, "Elegir el umbral por costo", "★★", "Con los mismos ocho correos, bloquear un legítimo (FP) cuesta 4 y dejar pasar un spam (FN) cuesta 1.", [
        "Calcula FP, FN y el costo para t = 0.90, 0.85, 0.75, 0.60, 0.50, 0.40 y 0.20.",
        "¿Qué umbral minimiza el costo y cuánto vale?",
        "Calcula el umbral teórico c_FP/(c_FP + c_FN) y compáralo con tu resultado.",
        "Compara con la regla «dejar pasar todo». ¿Mejora el modelo a esa referencia?"])
    s += bloque(6, "AIC y BIC", "★★", "Con n = 100 se ajustan tres modelos: M1 (k = 3, SSE = 250), M2 (k = 5, SSE = 200) y M3 (k = 9, SSE = 170).", [
        "Calcula AIC = n ln(SSE/n) + 2k de cada modelo y elige.",
        "Calcula BIC = n ln(SSE/n) + k ln n de cada modelo y elige.",
        "¿Por qué pueden discrepar? ¿Cuál es más prudente con muchos parámetros?",
        "Menciona una razón por la que ninguno reemplaza a la validación cruzada."])
    s += bloque(7, "¿Para qué la validación anidada?", "★★", "Se prueban 6 valores de α. La búsqueda simple usa 5 pliegues; la anidada, 5 externos y 4 internos.", [
        "Cuenta los ajustes de una búsqueda simple (con el reajuste final) y de la anidada.",
        "Un analista informa «el mejor RMSE de CV fue 7.9»; la anidada da 9.1. ¿Cuál estima mejor el desempeño futuro del procedimiento?",
        "Con etiquetas al azar, ¿qué exactitud esperarías de la anidada y por qué el mejor puntaje de CV da más?",
        "¿Con qué modelo final se predice en producción?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C38))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C38), Spacer(1, 2)]

    y = np.array([12, 15, 9, 20, 14.0])
    yh = np.array([10, 16, 12, 17, 15.0])
    e = y - yh
    sse, sst = float(np.sum(e ** 2)), float(np.sum((y - y.mean()) ** 2))
    yh2 = yh.copy()
    yh2[3] = 14.0
    e2 = y - yh2
    sse_b = float(np.sum((y - 13) ** 2))
    s += sol("Bloque 1", [
        ["1", f"e = y − ŷ = (2, −1, −3, 3, −1). MAE = (2 + 1 + 3 + 3 + 1)/5 = {f(np.mean(abs(e)))}."],
        ["2", f"MSE = (4 + 1 + 9 + 9 + 1)/5 = {f(sse / 5)}; RMSE = {f(float(np.sqrt(sse / 5)), 3)}. Es mayor que el MAE porque los errores no son iguales (el cuadrado pesa más los grandes)."],
        ["3", f"ȳ = 14, SST = 4 + 1 + 25 + 36 + 0 = {f(sst, 0)}; SSE = {f(sse, 0)}; R² = 1 − 24/66 = {f(1 - sse / sst, 3)}. Con referencia 13: SSE_ref = (1 + 4 + 16 + 49 + 1) = {f(sse_b, 0)}; 1 − 24/71 = {f(1 - sse / sse_b, 3)}."],
        ["4", f"Con ŷ₄ = 14 el error del cuarto es 6: errores (2, −1, −3, 6, −1). MAE = 13/5 = {f(np.mean(abs(e2)))} (sube 0.6); RMSE = √(51/5) = {f(float(np.sqrt(np.mean(e2 ** 2))), 3)} (sube {f(float(np.sqrt(np.mean(e2 ** 2)) - np.sqrt(sse / 5)), 3)}). En términos relativos, el RMSE sube más ({f(float(np.sqrt(np.mean(e2 ** 2)) / np.sqrt(sse / 5) - 1) * 100, 0)} % frente a {f(float(np.mean(abs(e2)) / np.mean(abs(e)) - 1) * 100, 0)} %)."]])
    r = np.array([2.0, 3, 4, 5])
    s += sol("Bloque 2", [
        ["1", f"Promedio = 14/4 = 3.5. Desviación muestral = √[(2.25 + 0.25 + 0.25 + 2.25)/3] = {f(float(np.std(r, ddof=1)), 3)}."],
        ["2", f"MSE por pliegue: 4, 9, 16, 25; promedio 13.5; RMSE OOF = √13.5 = {f(float(np.sqrt(13.5)), 3)}. No coincide con 3.5: la raíz no es lineal (√ de promedio ≥ promedio de √)."],
        ["3", f"Suma de cuadrados = 25·(4 + 9 + 16) + 75·25 = 725 + 1 875 = 2 600; n = 150; RMSE OOF = √(2 600/150) = {f(float(np.sqrt(2600 / 150)), 3)}. El pliegue grande pesa más."],
        ["4", "Los pliegues comparten filas de entrenamiento: sus errores no son réplicas independientes, y la desviación subestima o confunde la incertidumbre real."]])
    g = np.array([1, 3, 6, 10, 15])
    ft = np.array([6.0, 3.1, 2.0, 0.8, 0.1])
    vl = np.array([6.4, 3.6, 4.1, 5.9, 11.0])
    s += sol("Bloque 3", [
        ["1", "Grado 3: tiene el menor error de validación (3.6). Los grados 6, 10 y 15 ajustan mejor (2.0, 0.8, 0.1) pero validan peor (4.1, 5.9, 11.0)."],
        ["2", "Grado 1: ajuste 6.0 y validación 6.4, ambos altos y juntos → sesgo alto. Grado 15: ajuste 0.1 y validación 11.0 → varianza alta (sobreajuste)."],
        ["3", f"Brechas: {', '.join(f'{int(a)}: {f(float(b - c_), 1)}' for a, b, c_ in zip(g, vl, ft))}. La brecha crece con el grado; el error de ajuste solo baja."],
        ["4", "Sesgo alto: más flexibilidad (más grado o variables). Varianza alta: más datos, penalización (ridge/lasso) o menos variables/grado."]])
    sc = [(0.95, 1), (0.85, 1), (0.80, 0), (0.70, 1), (0.55, 0), (0.45, 1), (0.30, 0), (0.10, 0)]
    pts = [(0.0, 0.0)]
    tp = fp_ = 0
    for _, yy in sc:
        tp += yy
        fp_ += 1 - yy
        pts.append((fp_ / 4, tp / 4))
    auc = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1))
    s += sol("Bloque 4", [
        ["1", "(FPR, recall) por paso: (0, 0.25), (0, 0.50), (0.25, 0.50), (0.25, 0.75), (0.50, 0.75), (0.50, 1), (0.75, 1), (1, 1)."],
        ["2", f"Solo hay avance horizontal en 4 pasos de ΔFPR = 0.25, con recall 0.50, 0.75, 1 y 1: área = 0.25·0.50 + 0.25·0.75 + 0.25·1 + 0.25·1 = 0.125 + 0.1875 + 0.25 + 0.25 = {f(auc, 4)} (los tramos verticales no suman área)."],
        ["3", "Positivos: 0.95, 0.85, 0.70, 0.45; negativos: 0.80, 0.55, 0.30, 0.10. 0.95 supera a 4; 0.85 a 4; 0.70 a 3; 0.45 a 2 → 13 de 16 = 0.8125."],
        ["4", "Es la probabilidad de que un spam elegido al azar reciba mayor puntaje que un legítimo. No dice cuántos se clasifican bien: eso depende del umbral."]])
    filas5 = []
    for t_ in (0.90, 0.85, 0.75, 0.60, 0.50, 0.40, 0.20):
        fpn = sum(1 for p_, yy in sc if p_ >= t_ and yy == 0)
        fnn = sum(1 for p_, yy in sc if p_ < t_ and yy == 1)
        filas5.append(f"t = {t_:.2f}: FP {fpn}, FN {fnn}, costo {4 * fpn + fnn}")
    s += sol("Bloque 5", [
        ["1", "; ".join(filas5) + "."],
        ["2", "El mínimo es 2, con t = 0.85 (cualquier t en (0.80, 0.85]): se bloquean 0.95 y 0.85, sin FP y con 2 FN."],
        ["3", "Teórico: 4/(4 + 1) = 0.80 (bloquear si p &gt; 0.80). Coincide con la zona óptima; con probabilidades mal calibradas pueden separarse."],
        ["4", "«Dejar pasar todo»: FN = 4, FP = 0 → costo 4. El umbral 0.85 (costo 2) la mejora; con t = 0.50 (costo 9) la regla trivial sería mejor."]])
    mods6 = [("M1", 3, 250.0), ("M2", 5, 200.0), ("M3", 9, 170.0)]
    aic = {m: 100 * np.log(s_ / 100) + 2 * k_ for m, k_, s_ in mods6}
    bic = {m: 100 * np.log(s_ / 100) + k_ * np.log(100) for m, k_, s_ in mods6}
    s += sol("Bloque 6", [
        ["1", "AIC: " + "; ".join(f"{m} = {f(aic[m])}" for m in aic) + ". Gana M3 (el menor)."],
        ["2", "BIC (ln 100 = 4.605): " + "; ".join(f"{m} = {f(bic[m])}" for m in bic) + ". Gana M2."],
        ["3", "El BIC penaliza cada parámetro con 4.605 y el AIC con 2: M3 agrega 4 parámetros por una baja de SSE de 200 a 170, que el AIC acepta y el BIC no. El BIC es más prudente con muchos parámetros."],
        ["4", "Se calculan con las filas de ajuste y suponen un modelo y observaciones independientes; no miran datos nuevos ni sirven para comparar modelos de distinta familia sin verosimilitud."]])
    s += sol("Bloque 7", [
        ["1", "Simple: 5 pliegues × 6 valores + 1 reajuste = 31 ajustes. Anidada: 5 externos × 4 internos × 6 valores + 5 reajustes = 125."],
        ["2", "La anidada (9.1): el mejor RMSE de CV (7.9) se obtuvo eligiendo el mínimo entre varios y es optimista."],
        ["3", "Con etiquetas al azar la anidada da ≈ 0.50 (no hay señal que aprender); el mejor puntaje de CV entre muchos candidatos da más (en la simulación del curso, ≈ 0.68) solo porque se elige el máximo."],
        ["4", "Se vuelve a elegir con todo el entrenamiento (bucle interno sobre todo el entrenamiento) y se ajusta ese modelo con todas las filas; la anidada solo estima su desempeño."]])
    build("apuntes/C38_Ejercicios_Cuaderno_EvaluacionModelos.pdf", s, "Ejercicios para el cuaderno · Clase 38")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
