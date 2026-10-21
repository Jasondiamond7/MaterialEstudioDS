"""Clase 41 · Detección de anomalías: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c41_numeros.json (lo escribe figuras_c41.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import BLUE_BG, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c41_numeros.json").read_text(encoding="utf-8"))
C41 = colors.HexColor("#CA8A04")
C41_BG = colors.HexColor("#FEFCE8")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas deteccion_anomalias.pdf + anomalias.py / .R del profesor Pedro Luiz Ramos"
RZ, MH, KN, LF, DB, CO, PC, IFO, OC, CA, RB, TP = (N[k] for k in ("ricz", "maha", "knn", "lof", "dbscan", "coinc", "pca", "iforest", "ocsvm", "cardio", "robustez", "tipos"))


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{int(round(x)):,}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.78):
    s.append(band(f"Clase 41 · {titulo}", FUENTE, C41))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C41, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C41))
    s.append(Spacer(1, 1))
    for fig, pie in figuras:
        s.append(img(fig, width=CONTENT_W * fw))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C41)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ---------------------------------------------------------------- cálculos de apoyo (ejemplos y ejercicios)
def lof_manual(x, k):
    """LOF paso a paso en una dimensión (sin empates en las distancias a los vecinos): devuelve k-dist, vecinos, lrd y LOF."""
    x = np.asarray(x, float)
    n = len(x)
    D = np.abs(x[:, None] - x[None, :])
    np.fill_diagonal(D, np.inf)
    vec = np.argsort(D, axis=1, kind="stable")[:, :k]
    kd = np.array([D[i, vec[i, -1]] for i in range(n)])
    lrd = np.zeros(n)
    for i in range(n):
        reach = [max(D[i, j], kd[j]) for j in vec[i]]
        lrd[i] = 1 / np.mean(reach)
    lof = np.array([np.mean(lrd[vec[i]]) / lrd[i] for i in range(n)])
    return kd, vec, lrd, lof


def c_n(n):
    return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n


def reglas_1d(v):
    v = np.asarray(v, float)
    q1, q3 = np.quantile(v, [0.25, 0.75])
    ric = q3 - q1
    z = (v - v.mean()) / v.std(ddof=1)
    return dict(q1=q1, q3=q3, ric=ric, L=q1 - 1.5 * ric, U=q3 + 1.5 * ric, media=v.mean(), sd=v.std(ddof=1), z=z, zmax=(len(v) - 1) / np.sqrt(len(v)))


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 41 · Detección de anomalías",
                  "Profesor Pedro Luiz Ramos · aplica ideas de Clases 30–32 (calidad de datos, K-means, PCA), 35 (árboles) y 37 (SVM); la evaluación usa la Clase 38", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("Una <b>anomalía</b> es un caso que se aparta de lo esperado <b>bajo una referencia</b>; una alerta dice <b>qué revisar</b>, no que el dato esté mal. "
                    "Cada método define «inusual» de otra manera: un valor extremo (RIC, z), una combinación que contradice la correlación (Mahalanobis), un vecindario vacío (kNN, LOF, DBSCAN), "
                    "un patrón que el PCA no reconstruye o un punto que se <b>aísla</b> con pocos cortes al azar (Isolation Forest). Ninguna regla es universal: se usan varias, se compara dónde coinciden y "
                    "se <b>investiga antes de eliminar</b>. Con etiquetas (cardio) se evalúa como cualquier clasificador, pero con una clase muy escasa: la exactitud engaña y importan sensibilidad, precisión y el corte.", "body")], bg=C41_BG, border=C41, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c41_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Calidad de datos y atípicos</b> (Clase 30): cuartiles, boxplot, fuga de datos",
                     "<b>K-means y PCA</b> (Clases 31 y 32): centroides sensibles a extremos; componentes y reconstrucción",
                     "<b>Árboles y bosques</b> (Clases 34–35): particiones recursivas, submuestras; <b>SVM</b> (Clase 37)",
                     "<b>Matriz de confusión, precisión, sensibilidad, ROC y PR</b> (Clases 37 y 38)"], "cell", C41)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Qué es una anomalía</b> → <b>una variable</b> → <b>Mahalanobis</b> → <b>kNN, LOF, DBSCAN</b> → <b>PCA</b> → <b>Isolation Forest</b> → <b>evaluar con etiquetas</b>",
                      f"<b>puntos.csv</b> ({DB['n_nucleo'] + DB['n_borde'] + DB['n_ruido']} puntos: 200 simulados + 6 añadidos), <b>canciones.csv</b> ({PC['n']} canciones, 7 variables), <b>cardio.csv</b> ({mil(CA['n'])} registros, 21 variables, {CA['anom_tot']} anomalías)",
                      "Pregunta guía: ¿qué quiero revisar y cuántas alertas puedo atender?"], "cell", C41)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con las clases vecinas (sin repetirlas)</b>", "cellb"),
                  P("<b>C30 → C41</b>: allí se detectaron atípicos con boxplot; hoy se formalizan (RIC, z, Mahalanobis) y se agregan métodos multivariados. "
                    "<b>C31/C32 → C41</b>: K-means se distorsiona con un extremo (el profesor lo mostró con 1, 2, 3 y 30); el PCA sirve al revés: lo que no se reconstruye es anómalo. "
                    "<b>C35 → C41</b>: el Isolation Forest usa árboles de cortes aleatorios, sin variable de respuesta. <b>C37 → C41</b>: la One-Class SVM (mencionada al cierre) es una SVM sin clases. "
                    "<b>C38 → C41</b>: con una clase del 9.8 % la exactitud, la ROC y la PR cuentan historias distintas; el corte se fija con entrenamiento o validación, nunca con la prueba.", "cell")], bg=SOFT, border=C41, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: ¿atípico, error o evento real?</b>", "cellb"),
                  P("Antes de eliminar: verificar unidades, duplicados y faltantes; consultar el contexto; distinguir un error corregible (una edad de 99 que era 29) de un evento real poco frecuente (Black Friday); "
                    "comparar resultados con y sin el caso. En fraude o fallas, el caso poco frecuente <b>es</b> el objeto de estudio. No se reemplaza cada valor marcado por la media.", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "Qué es una anomalía y reglas de una variable",
        "Hay anomalías <b>globales</b>, <b>contextuales</b> y <b>conjuntas</b>. Para una sola variable hay dos reglas: el <b>RIC</b> (cuartiles, no supone normalidad) y el <b>puntaje z</b> (media y desviación, que el propio extremo distorsiona).",
        [("Tres tipos", f"Global: un 30 entre valores de 10 a 15. Contextual: un consumo normal a las 14 h es anómalo a las 3 h. Conjunta: los casos 202 y 203 tienen valores marginales comunes (|z| ≤ {f(TP['z_max_marginal'])}) pero contradicen la correlación."),
         ("RIC", f"Q1 = {f(RZ['q1'])}, Q3 = {f(RZ['q3'])}, RIC = {f(RZ['ric'], 1)}; límites {RZ['L']:g} y {RZ['U']:g}. El 30 queda fuera: alerta. La regla es de exploración, no una prueba de error."),
         ("Puntaje z", f"Media {f(RZ['media'], 3)} y desviación {f(RZ['sd'], 3)}: z(30) = {f(RZ['z30'], 3)} &lt; 3, sin alerta. No es mala suerte: con n = 8 el máximo posible es (n − 1)/√n = {f(RZ['zmax_n8'])}; hace falta n ≥ {RZ['n_min']} para poder superar 3.")],
        [("c41_tipos.png", "Izquierda: los ocho valores del profesor, con la zona del RIC. Centro: ilustración de una anomalía contextual (el mismo 78 es normal a las 14 h). Derecha: puntos.csv; dos casos añadidos con valores marginales comunes."),
         ("c41_ricz.png", f"Izquierda: boxplot con límites 8 y 18. Centro: puntajes z de los ocho valores (el 30 llega a {f(RZ['z30'], 3)}). Derecha: el |z| máximo posible crece como (n − 1)/√n; con 8 datos nunca llega a 3.")],
        ["RIC = Q3 − Q1;  L = Q1 − 1.5·RIC;  U = Q3 + 1.5·RIC",
         "alerta si x &lt; L  o  x &gt; U",
         "z = (x − media)/s;  alerta si |z| &gt; 3",
         "|z| ≤ (n − 1)/√n  (con s muestral)"],
        ["q1, q3 = np.quantile(v, [.25, .75])", "ric = q3 - q1", "z = (v - v.mean()) / v.std(ddof=1)"],
        [f"Fuera de ±2σ queda ≈ {f(RZ['p_2s'] * 100, 2)} % y fuera de ±3σ ≈ {f(RZ['p_3s'] * 100, 2)} % bajo una normal",
         f"Sin el 30 la media es {f(RZ['media_sin30'], 3)}; con él, {f(RZ['media'], 3)}: el extremo arrastra media y desviación (enmascaramiento)",
         f"Alternativa robusta: mediana {f(RZ['mediana'], 1)} y MAD {f(RZ['mad'], 1)}; z modificado = 0.6745·(x − mediana)/MAD = {f(RZ['zmod30'], 2)} &gt; 3.5"],
        ["Una alerta no demuestra error. Y si los datos son muy asimétricos, el RIC marcará valores válidos de la cola."])

    topic_page(
        s, "Distancia de Mahalanobis",
        "La distancia euclidiana ignora que las variables se mueven juntas. <b>Mahalanobis</b> mide cuántas «desviaciones» está un punto del centro <b>en la dirección en que se mueve la nube</b>: un punto contrario a la correlación queda lejos aunque sus valores sean moderados.",
        [("Idea de la elipse", f"Con correlación 0.9, (1, 1) y (1, −1) están a la misma distancia euclidiana (√2) pero D² = {f(MH['d2_hip_a'], 3)} y {f(MH['d2_hip_b'], 0)}. Las elipses de 95 % y 99 % son los contornos de D² = χ²."),
         ("Corte con χ²", f"Con p = 2 variables, χ²(2; 0.99) = {f(MH['u99'], 3)}. En puntos.csv marca {MH['n_alertas']['0.99']} casos (ids {', '.join(str(i) for i in MH['ids']['0.99'])}); con 95 % son {MH['n_alertas']['0.95']} y con 90 %, {MH['n_alertas']['0.9']}. Es una referencia aproximada bajo normalidad."),
         ("Covarianza frágil", f"Los seis casos añadidos reducen la correlación de {f(MH['corr_200'])} (200 puntos) a {f(MH['corr_todos'])}; así el caso 204 queda en D² = {f(MH['d2_adic'][3], 2)} &lt; 9.21 y «se salva». Con una covarianza robusta (MCD) su D² sube a {f(MH['d2_adic_mcd'][3], 1)}.")],
        [("c41_mahalanobis.png", f"Izquierda: dos puntos a igual distancia euclidiana (círculo) con D² muy distinto (elipse del 95 %). Centro: puntos.csv con elipses de 95 % (punteada) y 99 % y las {MH['n_alertas']['0.99']} alertas. Derecha: D² de los casos 201 a 206 con covarianza clásica y robusta.")],
        ["D² = (x − m)ᵀ S⁻¹ (x − m)",
         "alerta si D² &gt; χ²(p; 0.99)",
         "χ²(2; 0.99) = 9.210;  χ²(2; 0.95) = 5.991",
         "mahalanobis() de R devuelve D², no D"],
        ["S = np.cov(z, rowvar=False)", "d2 = np.sum((c @ np.linalg.inv(S)) * c, axis=1)", "umbral = chi2.ppf(0.99, df=z.shape[1])"],
        [f"Cuenta de la diapositiva 14: S = [[1, 0.9], [0.9, 1]], S⁻¹ = (1/0.19)·[[1, −0.9], [−0.9, 1]]; D²(1, 1) = (1 + 1 − 1.8)/0.19 = {f(MH['d2_hip_a'], 3)}; D²(1, −1) = (1 + 1 + 1.8)/0.19 = 20",
         f"Con 200 puntos normales se esperan ≈ {f(MH['esperadas_99'], 0)} alertas al 99 %: el caso 95 es una de ellas, no un error del método",
         f"Estandarizar antes no cambia Mahalanobis (diferencia máxima {MH['inv_dif']:.0e}); sí importa para kNN, LOF, DBSCAN y One-Class SVM"],
        ["S debe ser invertible (cuidado con variables redundantes) y el corte χ² es una aproximación: con parámetros estimados no garantiza un 1 % de falsas alertas."])

    topic_page(
        s, "Vecinos y densidad: kNN, LOF y DBSCAN",
        "Tres maneras de mirar el vecindario: <b>kNN</b> usa la distancia al vecino k, <b>LOF</b> compara la densidad de un punto con la de sus vecinos (sirve cuando hay zonas más y menos densas) y <b>DBSCAN</b> deja como <b>ruido</b> lo que no pertenece a ninguna región densa.",
        [("kNN", f"Distancia al vecino k = 10 (sin el propio punto); corte en el cuantil 95 % ({f(KN['umbral'], 3)}): {KN['n_alertas']} de 206. Un cuantil 95 % siempre marca ≈ 5 %."),
         ("LOF", f"LOF = prom. lrd de vecinos / lrd(i): ≈ 1 normal, mucho mayor que 1 = menos denso que su entorno. Con k = 10 y corte 1.5: {LF['n_alertas']}. Ejemplo del profesor: lrd(A) = 2, vecinos 6, 5, 7 → LOF = 3."),
         ("DBSCAN", f"ε = 0.45, minPts = 6: {DB['n_nucleo']} núcleos, {DB['n_borde']} bordes y {DB['n_ruido']} de ruido. Agrupa (no clasifica); solo el ruido se marca.")],
        [("c41_knn_lof.png", f"Izquierda: puntajes kNN ordenados y corte. Centro: dos densidades (simuladas): el punto rojo es apenas el puesto {LF['toy_rank_knn']} por kNN pero el primero por LOF ({f(LF['toy_lof'])}). Derecha: alertas con LOF &gt; 1.5 según k."),
         ("c41_dbscan.png", f"Izquierda: núcleos, bordes y ruido. Centro: el ruido cae de {DB['ruido_por_eps']['0.25']} a {DB['ruido_por_eps']['1']} al subir ε. Derecha: de los seis casos añadidos, 201, 202, 203 y 205 los marcan los cuatro métodos; el 204 no lo marca Mahalanobis; el 206 (1.9, 1.9) no lo marca ninguno.")],
        ["kNN: score(i) = d(i, vecino k)",
         "reach(i, j) = max{d(i, j), k-dist(j)}",
         "lrd(i) = 1/promedio de reach(i, j);  LOF = prom. lrd(vecinos)/lrd(i)",
         "núcleo: ≥ minPts en el radio ε (incluye al punto)"],
        ["LocalOutlierFactor(n_neighbors=10)", "DBSCAN(eps=0.45, min_samples=6)  # ruido = -1"],
        [f"Alertas: Mahalanobis {CO['totales']['Mahalanobis']}, kNN {CO['totales']['kNN']}, LOF {CO['totales']['LOF']}, DBSCAN {CO['totales']['DBSCAN']}; {CO['votos_hist']['4']} puntos los marcan los cuatro y {CO['al_menos1']} al menos uno",
         "Sugerencia del profesor: usar varios métodos y mirar dónde coinciden"],
        [f"El corte 1.5 no es universal: el caso 206 llega a LOF {f(LF['lof206_por_k'][6], 2)} con k = 30 y las alertas pasan de {LF['cuenta_por_k'][3]} (k = 10) a {LF['cuenta_por_k'][6]}. Y el ruido de DBSCAN depende de ε."])

    topic_page(
        s, "PCA: el error de reconstrucción",
        "El PCA resume las variables en pocas componentes; si un caso <b>no se reconstruye bien</b> desde ellas, su combinación de variables es inusual. El error se mide con <b>Q</b>, la suma de residuos al cuadrado en escala estandarizada.",
        [("Proyectar y volver", "T = Z·P (coordenadas en las r componentes), Ẑ = T·Pᵀ (reconstrucción) y Q = Σ(z − ẑ)². Con una componente (1, 1)/√2: A = (2, 0) tiene Q = 2 y B = (4, 4) tiene Q = 0."),
         ("Canciones", f"Siete variables, {PC['n']} canciones. Con 5 componentes se explica {f(PC['acum'][4] * 100)} % de la varianza; Q en el cuantil 95 % ({f(PC['umbral'], 3)}) marca {PC['n_alertas']} canciones. Las mayores: {', '.join(str(i) for i in PC['top_ids'][:5])} (Q = {f(PC['top_q'][0], 3)} la primera)."),
         ("Lo que Q no ve", f"B está lejos del centro y tiene Q = 0: un extremo <i>dentro</i> del subespacio. Lo cubre T² = Σ t²/λ. En canciones Q y T² casi no se parecen (correlación {f(PC['corr_q_t2'])}): detectan cosas distintas.")],
        [("c41_pca.png", f"Izquierda: A se aparta de la recta (Q = 2) y B no (Q = 0). Centro: varianza explicada; con 5 componentes {f(PC['acum'][4] * 100)} %. Derecha: Q de las {PC['n']} canciones con el corte del cuantil 95 %.")],
        ["Z = (X − media)/s;  T = Z·P_r;  Ẑ = T·P_rᵀ",
         "Q_i = Σ_j (z_ij − ẑ_ij)²",
         "T²_i = Σ_l t_il²/λ_l",
         "con r = p componentes, Q ≈ 0 para todos"],
        ["pca = PCA().fit(z)", "rec = (z @ P.T) @ P   # P = pca.components_[:r]", "q = ((z - rec) ** 2).sum(axis=1)"],
        [f"Cardio: con el 80 % de varianza se retienen {CA['r']} componentes y el corte Q (cuantil 95 % de entrenamiento) vale {f(CA['u_pca'], 3)}; AUC en prueba {f(CA['pca']['auc'], 3)}",
         f"Q depende de r: con {RB['r_lista'][0]} componentes el AUC es {f(RB['auc_q_por_r'][0], 3)}, con {RB['r_lista'][3]}, {f(RB['auc_q_por_r'][3], 3)}, con {RB['r_lista'][5]}, {f(RB['auc_q_por_r'][5], 3)}; T² (r = 8) alcanza {f(CA['auc_t2'], 3)}",
         "Para datos nuevos: misma media, escala y componentes de referencia; sin reajustar con los datos que dieron alarma"],
        ["Explicar mucha varianza no garantiza detectar todas las anomalías, y la elección de r (aquí 80 %) es una decisión que se fija antes de mirar la prueba."])

    topic_page(
        s, "Isolation Forest y One-Class SVM",
        "El <b>Isolation Forest</b> no mide distancias ni densidades: <b>corta al azar</b> y cuenta cuántos cortes hacen falta para dejar un punto solo. Los extremos se aíslan rápido; los del centro, tarde. Promediando muchos árboles se obtiene un puntaje.",
        [("Árboles de aislamiento", "Cada árbol usa una submuestra; en cada nodo elige una variable y un corte al azar y se detiene al aislar. No usa etiquetas."),
         ("Puntaje", f"s = 2^(−E[h]/c(ψ)); c(256) = {f(IFO['c256'], 2)}. E[h] = c(ψ) da 0.5; caminos más cortos dan s mayor. No es una probabilidad (la diapositiva usa c = 8 como ejemplo)."),
         ("One-Class SVM", f"SVM sin clases: encierra la región normal con un núcleo y un parámetro ν (≈ fracción de alertas). En puntos.csv: ν = 0.05 marca {OC['por_nu']['0.05']['n']}; ν = 0.10, {OC['por_nu']['0.1']['n']}.")],
        [("c41_iforest.png", f"Izquierda: una secuencia de cortes al azar que aísla al caso 201. Centro: en 2 000 árboles el 201 se aísla con {f(IFO['prof_ext_media'], 1)} cortes en promedio y un punto central con {f(IFO['prof_cen_media'], 1)}. Derecha: puntaje en función de E[h] con c(256)."),
         ("c41_ocsvm.png", "One-Class SVM (núcleo gaussiano, γ = «scale») sobre puntos.csv: la frontera amarilla encierra lo normal. A mayor ν, más alertas.")],
        ["s(x) = 2^(−E[h(x)]/c(ψ))",
         "c(ψ) = 2(ln(ψ − 1) + 0.5772) − 2(ψ − 1)/ψ",
         "sklearn: score_samples = −s (más bajo = más anómalo)",
         "contamination solo fija el corte de predict()"],
        ["m = IsolationForest(n_estimators=300, max_samples=256).fit(z_tr)", "score = -m.score_samples(z_te)   # alto = anómalo"],
        [f"Diap. 38: h = 2 y c = 8 → s = {f(IFO['s_h2_c8'], 3)}; con c(256) el mismo h = 2 da {f(IFO['s_h2_c256'], 3)}",
         f"300 árboles = caminos promediados; 256 = registros por árbol (no variables). Con «auto», predict() corta en s &gt; 0.5: {CA['auto']['alertas']} alertas en prueba, no 29",
         "No necesita estandarizar: puntajes idénticos con las variables sin escalar"],
        [f"Con 10 semillas el AUC va de {f(RB['auc_min'], 3)} a {f(RB['auc_max'], 3)}; un puntaje 0.84 no es «84 % de probabilidad»."])

    topic_page(
        s, "Cardio: evaluar con etiquetas disponibles",
        f"Con etiquetas se puede evaluar un detector, pero las anomalías son solo el <b>{f(CA['prev_te'] * 100, 1)} %</b>. Todo se ajusta con <b>entrenamiento</b> (escala, componentes, bosque y corte) y la prueba se usa una vez. La exactitud casi no supera a «todo normal»; importan sensibilidad, precisión y el corte.",
        [("Partición", f"{mil(CA['n'])} registros, {CA['anom_tot']} anomalías: entrenamiento {mil(CA['n_tr'])} ({CA['anom_tr']}), prueba {CA['n_te']} ({CA['anom_te']}). El entrenamiento no se filtra por etiqueta."),
         ("Corte del cuantil 95 %", f"IF: matriz {CA['iff']['VN']}/{CA['iff']['FP']}/{CA['iff']['FN']}/{CA['iff']['VP']}, exactitud {f(CA['iff']['acc'], 4)} (todo normal: {f(CA['todo_normal'], 4)}), sensibilidad {f(CA['iff']['recall'], 4)}, precisión {f(CA['iff']['precision'], 4)}, AUC {f(CA['iff']['auc'], 4)}. PCA: {f(CA['pca']['acc'], 4)}, {f(CA['pca']['recall'], 4)}, {f(CA['pca']['precision'], 4)}, {f(CA['pca']['auc'], 4)}."),
         ("El corte cambia todo", f"Cuantil 90 %: {CA['tabla'][2]['alertas']} alertas, sensibilidad {f(CA['tabla'][2]['recall'], 3)}; 95 %: {CA['tabla'][3]['alertas']} y {f(CA['tabla'][3]['recall'], 3)}; 99 %: {CA['tabla'][4]['alertas']} y {f(CA['tabla'][4]['recall'], 3)}. El AUC no cambia.")],
        [("c41_cardio.png", f"Izquierda: ROC (AUC {f(CA['pca']['auc'], 3)} y {f(CA['iff']['auc'], 3)}); el punto rojo es el corte del 95 %. Centro: precisión–recall con la línea de azar en {f(CA['prev_te'], 3)} (AP {f(CA['pca']['ap'], 3)} y {f(CA['iff']['ap'], 3)}). Derecha: sensibilidad, precisión y F1 según el cuantil del corte."),
         ("c41_robustez.png", f"Izquierda: AUC del bosque con 10 semillas ({f(RB['auc_min'], 3)} a {f(RB['auc_max'], 3)}); el valor de R (0.9220) cae dentro. Centro: el AUC del PCA según r. Derecha: entrenar solo con casos normales sube el AUC de {f(RB['auc_cont'], 3)} a {f(RB['auc_norm'], 3)}.")],
        ["sensibilidad = VP/(VP + FN);  precisión = VP/(VP + FP)",
         "exactitud = (VN + VP)/n",
         "alertas ≈ (1 − cuantil)·n",
         "referencia PR: precisión = prevalencia"],
        ["u = np.quantile(score_train, 0.95)", "clase = (score_test > u).astype(int)"],
        [f"Con ≈ 5 % de alertas la sensibilidad no puede pasar de {f(CA['recall_max_5'], 2)} (hay 9.8 % de anomalías)",
         f"Cuantil elegido con CV en entrenamiento (F1): {f(CA['q_elegido'], 2)}; en prueba, sensibilidad {f(CA['elegido']['recall'], 3)}, precisión {f(CA['elegido']['precision'], 3)}, {CA['elegido']['alertas']} alertas",
         "Es un benchmark, no una herramienta clínica"],
        ["Un AUC alto no implica que el cuantil 95 % detecte la mayoría; el corte no se elige con la prueba."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase (transcripción)</b>", "cellb"),
                  bullets([f"<b>«El z no marca el 30 porque el extremo elevó la media y la desviación»</b> (cierto, pero incompleto): con n = 8 marcarlo es imposible; |z| ≤ (n − 1)/√n = {f(RZ['zmax_n8'])} (y {f(RZ['zmax_n8_pob'])} con desviación poblacional). Hace falta n ≥ {RZ['n_min']}.",
                           f"<b>«Cuartil 11.65 por los empates» y «la media pasa de 12.5 a 14.62»</b>: con la interpolación del código Q1 = {f(RZ['q1'], 2)} (no 11.65) y la media sin el 30 es {f(RZ['media_sin30'], 2)}; con él, 14.625.",
                           "<b>«Seis sigmas = uno en un millón»</b>: fuera de ±6σ queda ≈ 2 por mil millones; la cifra de 3.4 por millón viene de la convención con un desplazamiento de 1.5σ.",
                           f"<b>Mahalanobis «95 %: 4 bien y 1 mal»</b>: con 95 % hay {MH['n_alertas']['0.95']} alertas (5 añadidos y 3 naturales: 25, 83, 95); con 99 %, {MH['n_alertas']['0.99']} (4 añadidos y el 95).",
                           f"<b>kNN «umbral ≈ 0.98»</b>: es {f(KN['umbral'], 3)}. Y se dice que la media sería «más razonable», pero el script usa la distancia al vecino k (el máximo de las k).",
                           "<b>«DBSCAN … método de clasificación»</b>: es agrupamiento (no supervisado); «Bayescan», «debes» y «B-scan» son DBSCAN.",
                           f"<b>Varianza de las canciones «30, 40, 60, 70, 80, 81 %»</b>: las acumuladas reales son {', '.join(f(a * 100, 1) for a in PC['acum'][:5])} %. Y «38» es la canción 138.",
                           f"<b>Cardio «166 anomalías», «sensibilidad de 16 %», «acuracia del PCA 0.9» y «sensibilidad 0.815 con 0.4994»</b>: son {CA['anom_tot']} anomalías; la sensibilidad es 16 de 54 = {f(CA['iff']['recall'], 3)} (16 es el conteo); la exactitud del PCA es {f(CA['pca']['acc'], 2)}; y con 0.4994 la sensibilidad es {f(CA['tabla'][2]['recall'], 4)} (26/54), no 0.815."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, scripts y verificación</b>", "cellb"),
                  bullets([f"<b>Estandarizar y Mahalanobis</b> (diap. 11 y 13): es invariante a escalas (diferencia {MH['inv_dif']:.0e}); la estandarización importa para kNN, LOF, DBSCAN y One-Class SVM, y no es necesaria para el Isolation Forest (puntajes idénticos).",
                           f"<b>Covarianza clásica y enmascaramiento</b> (diap. 15): con los 6 casos añadidos la correlación baja de {f(MH['corr_200'])} a {f(MH['corr_todos'])}; con MCD el caso 204 también alerta ({MH['n_mcd']} alertas).",
                           f"<b>LOF 1.5 «ilustrativo»</b> (diap. 21): la diapositiva ya lo dice; verificado: {LF['cuenta_por_k'][3]} alertas con k = 10 y {LF['cuenta_por_k'][6]} con k = 30. En R, <font name='Mono'>minPts = k + 1</font> equivale a <font name='Mono'>n_neighbors = k</font>.",
                           f"<b>Q y T²</b> (diap. 29): confirmado; en canciones los diez mayores Q y los diez mayores T² no comparten ninguno ({PC['inter_top10']}). Con r = 7, Q ≈ 0 ({PC['max_q_r7']:.0e}).",
                           f"<b>Isolation Forest R frente a Python</b> (diap. 47): 0.9220 frente a 0.9072 cabe en la variación por semilla de scikit-learn ({f(RB['auc_min'], 4)} a {f(RB['auc_max'], 4)}); no es evidencia de que un programa sea mejor.",
                           f"<b>contamination</b> (diap. 39): solo fija el corte de <font name='Mono'>predict()</font>; con 0.1 el corte es {f(CA['tabla'][2]['umbral'], 4)}, el cuantil 90 % de entrenamiento. El «auto» (−0.5) marca {CA['auto']['alertas']} casos en prueba (sensibilidad {f(CA['auto']['recall'], 3)}).",
                           f"<b>Entrenamiento contaminado</b> (diap. 41): se ajusta con {f(CA['prev_tr'] * 100, 1)} % de anomalías; con solo normales el AUC sube a {f(RB['auc_norm'], 3)} y el AP a {f(RB['ap_cont'], 3)} → {f(RB['ap_norm'], 3)}.",
                           "<b>Transcripción con ruido</b>: «HIC / RIC / ÚRIC» = RIC (IQR), «KNN / KNL» = kNN, «loft» = LOF, «Isosation / Presolation Forest» = Isolation Forest, «ESF» = IF, «camins» = K-means, «VWSS» = WSS, «ROCK» = ROC. El recreo (1:05:53–1:26:21) quedó marcado."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Calculo cuartiles, límites del RIC y z a mano, y sé por qué |z| &gt; 3 puede ser imposible",
                           "Calculo D² con una matriz de covarianza 2×2 y lo comparo con χ²",
                           "Calculo k-distancia, lrd y LOF en un conjunto pequeño y explico qué corrige LOF frente a kNN",
                           "Reconstruyo un punto con una componente (Q) y explico qué ve T² que Q no ve",
                           "Razono la longitud esperada de aislamiento y calculo s = 2^(−E[h]/c)",
                           "Elijo un corte según las alertas que puedo revisar y evalúo precisión y sensibilidad frente a la prevalencia",
                           "Resuelvo los bloques 1 a 6 del cuaderno con al menos 80 % de aciertos",
                           "<b>Siguiente</b>: el taller del curso; sin etiquetas se revisan casos con especialistas y se prueba estabilidad (vecinos, escala, semilla)."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C41_Guia_Anomalias.pdf", s, "Clase 41 · Detección de anomalías")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#CA8A04">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 41 · Detección de anomalías", "Para leer antes de los ejercicios · lectura de ~30 min", C41))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Una alerta es una pista", "Indica qué revisar; no demuestra un error. Se investiga antes de eliminar."),
                            ("Cada método define «inusual» distinto", "Extremo (RIC, z), combinación (Mahalanobis), vecindario (kNN, LOF, DBSCAN), reconstrucción (PCA), aislamiento (Isolation Forest)."),
                            ("z puede no alertar", "Con n pequeño |z| ≤ (n − 1)/√n; el extremo infla la media y la desviación."),
                            ("Escala y covarianza", "kNN, LOF y DBSCAN piden estandarizar; Mahalanobis ya corrige escala y correlación, pero su covarianza es sensible a extremos."),
                            ("El corte decide las alertas", "Un cuantil fija cuántas alertas; se elige con validación o con la capacidad de revisión, no con la prueba."),
                            ("Clase escasa: no mires la exactitud", "Compara con «todo normal» y con la prevalencia; usa sensibilidad, precisión y PR.")], C41)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C41_Anomalias.ipynb</font> o el cuaderno <font name='Mono'>C41_Ejercicios_Cuaderno_Anomalias.pdf</font>.", "small"))

    # 1 RIC y z
    s.append(h("1", "RIC y puntaje z"))
    s += para("El RIC marca los valores fuera de [Q1 − 1.5·RIC, Q3 + 1.5·RIC]; usa cuartiles, así que no depende de una normal ni de los extremos. El puntaje z = (x − media)/s marca |z| &gt; 3, pero la media y la desviación incluyen al propio extremo. Con n datos y desviación muestral, |z| nunca supera (n − 1)/√n.")
    v = [4, 5, 5, 6, 7, 8, 9, 40]
    r = reglas_1d(v)
    s.append(ejemplo("ocho valores: 4, 5, 5, 6, 7, 8, 9 y 40", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Cuartiles", "interpolación lineal, posiciones 1.75 y 5.25", f"Q1 = {f(r['q1'])}, Q3 = {f(r['q3'])}"],
               ["RIC y límites", f"RIC = {f(r['ric'])}; Q1 − 1.5·RIC; Q3 + 1.5·RIC", f"{f(r['L'])} y {f(r['U'])}"],
               ["Alerta RIC", "40 &gt; límite superior", "sí (solo el 40)"],
               ["Media y desviación", "con el 40 incluido", f"{f(r['media'], 3)} y {f(r['sd'], 3)}"],
               ["z del 40", f"({40} − {f(r['media'], 3)})/{f(r['sd'], 3)}", f"{f(r['z'][-1], 3)}"],
               ["Cota para n = 8", "(8 − 1)/√8", f"{f(r['zmax'], 3)} &lt; 3"]],
              [38 * mm, 78 * mm, CONTENT_W - 116 * mm - 18]),
        P(f"Lectura: el RIC marca el 40 y el z no (aunque 40 es más de 4 veces el segundo mayor). <b>Error común</b>: concluir «no hay atípicos» porque |z| &lt; 3; con n = 8 ese resultado es inevitable.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_ricz.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A1 y A2) y el Bloque 1 del cuaderno", [P("Calcula límites y z con NumPy, dibuja el boxplot y grafica (n − 1)/√n.", "cell")]))

    # 2 Mahalanobis
    s.append(h("2", "Distancia de Mahalanobis"))
    s += para("D² = (x − m)ᵀ S⁻¹ (x − m) con m el centro y S la covarianza. Si las variables están correlacionadas, un punto contrario a la correlación queda lejos aunque sus coordenadas sean moderadas. El corte orientativo es χ²(p; 0.99); R devuelve D², no D.")
    m_ = np.array([50.0, 30.0])
    S = np.array([[25.0, 12.0], [12.0, 16.0]])
    Si = np.linalg.inv(S)
    pa, pb = np.array([58.0, 26.0]), np.array([58.0, 34.0])
    da, db = float((pa - m_) @ Si @ (pa - m_)), float((pb - m_) @ Si @ (pb - m_))
    s.append(ejemplo("centro (50, 30), S = [[25, 12], [12, 16]]; puntos (58, 26) y (58, 34)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Determinante", "25·16 − 12²", "256"],
               ["Inversa", "(1/256)·[[16, −12], [−12, 25]]", "—"],
               ["(58, 26): d = (8, −4)", "(16·64 + 2·(−12)·8·(−4) + 25·16)/256 = 2 192/256", f"D² = {f(da, 3)}"],
               ["(58, 34): d = (8, 4)", "(16·64 + 2·(−12)·8·4 + 25·16)/256 = 656/256", f"D² = {f(db, 3)}"],
               ["Corte", "χ²(2; 0.99)", f"{f(chi2.ppf(0.99, 2), 3)}"]],
              [42 * mm, 84 * mm, CONTENT_W - 126 * mm - 18]),
        P(f"Lectura: ambos puntos están a la misma distancia euclidiana (√80 ≈ 8.94) del centro, pero (58, 26) contradice la correlación positiva (D² = {f(da, 2)}, cerca del corte) y (58, 34) la sigue (D² = {f(db, 2)}). <b>Error común</b>: tomar D² como D al compararlo con χ².", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_mahalanobis.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A3) y el Bloque 2 del cuaderno", [P("Calcula D² con NumPy y con <font name='Mono'>MinCovDet</font>, y compara cuántas alertas da cada covarianza con los seis casos añadidos.", "cell")]))

    # 3 kNN y LOF
    s.append(h("3", "kNN y LOF en un conjunto pequeño"))
    s += para("El puntaje kNN es la distancia al vecino k (sin contar el propio punto). LOF compara la densidad local con la de los vecinos: reach(i, j) = max{d(i, j), k-dist(j)}, lrd(i) = 1/promedio de reach y LOF(i) = promedio de lrd de los vecinos / lrd(i). Cerca de 1: normal; mucho mayor: candidato a revisión.")
    xs = [1.0, 2.0, 3.0, 4.0, 6.0, 15.0]
    kd, vec, lrd, lof = lof_manual(xs, 2)
    s.append(ejemplo("seis valores en una recta: 1, 2, 3, 4, 6 y 15, con k = 2", [
        pasos([["Valor", "Vecinos", "k-dist", "lrd", "LOF"]] +
              [[f"{x:g}", ", ".join(f"{xs[j]:g}" for j in vec[i]), f"{kd[i]:g}", f"{lrd[i]:.3f}", f"{lof[i]:.3f}"] for i, x in enumerate(xs)],
              [20 * mm, 38 * mm, 24 * mm, 28 * mm, CONTENT_W - 110 * mm - 18]),
        P(f"Lectura: el 15 tiene el mayor k-dist ({kd[-1]:g}) y el mayor LOF ({lof[-1]:.2f}); el 6, con k-dist {kd[4]:g}, queda solo moderado ({lof[4]:.2f}). Ejemplo de reach: reach(15, 6) = max{{9, k-dist(6) = {kd[4]:g}}} = 9. <b>Error común</b>: usar 1.5 como regla universal; el LOF depende de k.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_knn_lof.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(img("c41_dbscan.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A4 y A5) y el Bloque 3 del cuaderno", [P("Calcula k-distancias con <font name='Mono'>NearestNeighbors</font>, LOF con <font name='Mono'>LocalOutlierFactor</font> y compara ε de DBSCAN.", "cell")]))

    # 4 PCA
    s.append(h("4", "PCA: Q y T²"))
    s += para("Con las cargas P de las r componentes, la coordenada de un caso es t = z·P, su reconstrucción es ẑ = t·Pᵀ y el error Q = Σ(z − ẑ)². Q detecta combinaciones que el subespacio no representa; T² = Σ t²/λ detecta casos extremos dentro del subespacio.")
    p1 = np.array([0.8, 0.6])
    z0 = np.array([3.0, 1.0])
    t0 = float(z0 @ p1)
    zh = t0 * p1
    e0 = z0 - zh
    s.append(ejemplo("una componente p = (0.8, 0.6) y el caso z = (3, 1); λ = 4", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Coordenada", "t = 3·0.8 + 1·0.6", f"{f(t0, 2)}"],
               ["Reconstrucción", "ẑ = t·p", f"({f(zh[0], 2)}, {f(zh[1], 2)})"],
               ["Error", "e = z − ẑ", f"({f(e0[0], 2)}, {f(e0[1], 2)})"],
               ["Q", f"{f(e0[0], 2)}² + ({f(e0[1], 2)})²", f"{f(float(e0 @ e0), 2)}"],
               ["T²", "t²/λ = 3.0²/4", f"{f(t0 ** 2 / 4, 2)}"]],
              [40 * mm, 76 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: Q = 1 dice que el caso se aparta del plano de la componente; T² = 2.25 que está moderadamente lejos del centro dentro de él. <b>Error común</b>: usar todas las componentes (Q ≈ 0 para todos) o mezclar variables sin estandarizar.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_pca.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B1 y B2)", [P("Reproduce las 10 canciones con mayor Q y compara con T².", "cell")]))

    # 5 Isolation Forest
    s.append(h("5", "Isolation Forest: longitud de camino y puntaje"))
    s += para("Cada árbol corta al azar hasta aislar los puntos de su submuestra de tamaño ψ. Los casos extremos necesitan pocos cortes. El puntaje s = 2^(−E[h]/c(ψ)), con c(ψ) = 2(ln(ψ − 1) + 0.5772) − 2(ψ − 1)/ψ, vale 0.5 cuando E[h] = c(ψ) y se acerca a 1 con caminos muy cortos. No es una probabilidad. scikit-learn devuelve −s en <font name='Mono'>score_samples</font>.")
    psi = 64
    cc = c_n(psi)
    filas = [["Caso", "E[h]", "E[h]/c", "s"]]
    for nm, hh in [("extremo", 3.0), ("típico", cc), ("muy central", 12.0)]:
        filas.append([nm, f(hh, 2), f(hh / cc, 3), f(2 ** (-hh / cc), 3)])
    s.append(ejemplo("submuestra ψ = 64: c(64) = 2(ln 63 + 0.5772) − 2·63/64", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["c(64)", "2(4.1431 + 0.5772) − 1.9688", f"{f(cc, 3)}"]], [28 * mm, 80 * mm, CONTENT_W - 108 * mm - 18]),
        Spacer(1, 3),
        pasos(filas, [38 * mm, 30 * mm, 38 * mm, CONTENT_W - 106 * mm - 18]),
        P(f"Lectura: un caso aislado en 3 cortes tiene s = {f(2 ** (-3 / cc), 3)}; uno que necesita {f(cc, 2)} (el promedio) tiene 0.5. <b>Error común</b>: leer 0.76 como «76 % de probabilidad»; solo ordena.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_iforest.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(img("c41_ocsvm.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B3) y el Bloque 4 del cuaderno", [P("Verifica la fórmula con las profundidades de los árboles de scikit-learn y mide la variación por semilla.", "cell")]))

    # 6 umbral y desbalance
    s.append(h("6", "Elegir el corte y evaluar con una clase escasa"))
    s += para("Un cuantil q del puntaje de entrenamiento fija ≈ (1 − q) de alertas. Con etiquetas, cada corte da una matriz de confusión; con pocas anomalías conviene comparar con la prevalencia: un detector al azar tiene precisión igual a la prevalencia. La exactitud de «todo normal» es 1 − prevalencia.")
    vn, fp, fn, vp = 940, 20, 20, 20
    n6 = vn + fp + fn + vp
    s.append(ejemplo("1 000 registros con 40 anomalías; el detector marca 40 (VP = 20, FP = 20)", [
        pasos([["Métrica", "Cálculo", "Resultado"],
               ["Prevalencia", "40/1000", "4.0 %"],
               ["Exactitud", "(940 + 20)/1000", f"{f((vn + vp) / n6 * 100, 1)} %"],
               ["«Todo normal»", "960/1000", "96.0 %"],
               ["Sensibilidad", "20/(20 + 20)", f"{f(vp / (vp + fn), 3)}"],
               ["Precisión", "20/(20 + 20)", f"{f(vp / (vp + fp), 3)}"],
               ["Mejora sobre el azar", "precisión / prevalencia", f"{f(vp / (vp + fp) / 0.04, 1)} veces"]],
              [44 * mm, 70 * mm, CONTENT_W - 114 * mm - 18]),
        P("Lectura: la exactitud (96 %) <b>empata</b> con no hacer nada, pero el detector acierta la mitad de sus alertas, 12.5 veces mejor que el azar (4 %). <b>Error común</b>: celebrar la exactitud o descartar el modelo porque su sensibilidad es 50 %.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c41_cardio.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(img("c41_robustez.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte C del notebook y el Bloque 5 del cuaderno", [P("Calcula la tabla de cuantiles y elige el corte con validación cruzada en entrenamiento, no con la prueba.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Una alerta indica qué revisar; investiga el contexto antes de eliminar o corregir.",
                       "Usa varios métodos y mira dónde coinciden; ajusta escala, componentes, bosque y corte solo con entrenamiento.",
                       "El corte fija cuántas alertas; la prevalencia fija la referencia de precisión.",
                       "Orden sugerido: apuntes → Bloques 1–3 del cuaderno → Partes A–B del notebook → Bloques 4–6 → Parte C → ejercicio de cierre."]))
    build("apuntes/C41_Apuntes_Anomalias.pdf", s, "Apuntes · Clase 41 · Detección de anomalías")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#CA8A04">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C41))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 41 · Detección de anomalías", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C41))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: RIC = Q3 − Q1 · límites Q1 − 1.5·RIC y Q3 + 1.5·RIC · z = (x − media)/s · |z| ≤ (n − 1)/√n · D² = (x − m)ᵀS⁻¹(x − m) · reach(i, j) = max{d(i, j), k-dist(j)} · LOF = prom. lrd vecinos / lrd(i) · s = 2^(−E[h]/c(ψ)) · precisión = VP/(VP + FP) · sensibilidad = VP/(VP + FN).",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C41)], bg=SOFT, border=C41, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "RIC y z a mano", "★★", "Ocho tiempos de respuesta (ms): 50, 52, 53, 55, 56, 58, 60 y 95.", [
        "Calcula Q1, Q3, RIC y los límites del RIC (interpolación lineal). ¿Qué valor se marca?",
        "Calcula la media, la desviación muestral y el z del 95. ¿Lo marca |z| &gt; 3?",
        "Calcula (n − 1)/√n. ¿Podría algún valor de estos ocho marcarse con |z| &gt; 3?",
        "Calcula el z modificado 0.6745·(x − mediana)/MAD del 95 y compáralo con 3.5."])
    s += bloque(2, "Mahalanobis con una covarianza 2×2", "★★★", "Centro m = (10, 20) y S = [[9, 6], [6, 16]]. Puntos P1 = (13, 16), P2 = (13, 24) y P3 = (10, 30).", [
        "Calcula la correlación entre las dos variables y la inversa de S.",
        "Calcula D² de P1 y P2. ¿Qué tienen en común sus distancias euclidianas?",
        "Calcula D² de P3 y compara los tres con χ²(2; 0.99) = 9.21.",
        "¿Por qué P1 y P2 tienen D² distintos si están a la misma distancia euclidiana?"])
    s += bloque(3, "k-distancia y LOF", "★★★", "Cinco valores en una recta: 0, 1, 2, 3 y 9, con k = 2.", [
        "Escribe los dos vecinos más cercanos y la k-distancia de cada valor.",
        "¿Qué valor marcaría un corte basado en la distancia al vecino 2? Calcula su puntaje.",
        "Calcula las distancias de accesibilidad, lrd y LOF del valor 9 (usa lrd(2) = lrd(3) = 0.667).",
        "¿Qué diría LOF de un valor 9 si el resto estuviera igual de disperso (0, 3, 6, 9, 12)?"])
    s += bloque(4, "Isolation Forest: razonar el aislamiento", "★★", "Valores 0, 1, 2, 3, 4, 5, 6 y 20. Un corte se elige uniforme entre el mínimo y el máximo del nodo.", [
        "En el primer corte (uniforme entre 0 y 20), ¿qué probabilidad hay de aislar el 20? ¿Y el 0?",
        "¿Por qué el valor 3 no puede aislarse en un solo corte?",
        "Con ψ = 16, c(16) = 2(ln 15 + 0.5772) − 2·15/16. Calcula c(16) y el puntaje s para E[h] = 2 y E[h] = 8.",
        "¿Qué puntaje corresponde a E[h] = c(ψ)? ¿Y qué indica un valor 0.8?"])
    s += bloque(5, "Elegir la contaminación y el corte", "★★", "Un sistema procesa 4 000 registros y se espera ≈ 2 % de anomalías (prevalencia). El equipo solo puede revisar 60 alertas.", [
        "¿Cuántas anomalías hay? ¿Qué cuantil del puntaje de entrenamiento produce ≈ 60 alertas?",
        "Con ese corte, ¿cuál es la sensibilidad máxima posible?",
        "Si el corte se fijara con el cuantil 98 % (contamination = 0.02), ¿cuántas alertas esperas y cuál sería el máximo de VP?",
        "¿Por qué no se elige el corte mirando la prueba y qué se hace en su lugar?"])
    s += bloque(6, "Precisión y sensibilidad con clase escasa", "★★★", "5 000 casos con 100 anomalías. Corte A: 150 alertas con 40 VP. Corte B: 300 alertas con 60 VP.", [
        "Para cada corte calcula sensibilidad, precisión y exactitud. ¿Cuál es la exactitud de «todo normal»?",
        "Compara la precisión de cada corte con la prevalencia. ¿Cuántas veces mejora al azar?",
        "Calcula F1 de A y B. ¿Qué corte prefieres si cada FN cuesta 10 y cada FP cuesta 1?",
        "¿Qué falta para decidir si el modelo es útil: ROC-AUC, PR o ambos? Justifica."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C41))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C41), Spacer(1, 2)]

    v = np.array([50, 52, 53, 55, 56, 58, 60, 95.0])
    r = reglas_1d(v)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    zm = 0.6745 * (95 - med) / mad
    s += sol("Bloque 1", [
        ["1", f"Posiciones 1.75 y 5.25: Q1 = 52 + 0.75·1 = {f(r['q1'], 2)}; Q3 = 58 + 0.25·2 = {f(r['q3'], 2)}; RIC = {f(r['ric'], 2)}. Límites: {f(r['q1'], 2)} − 1.5·{f(r['ric'], 2)} = {f(r['L'], 3)} y {f(r['q3'], 2)} + {f(1.5 * r['ric'], 3)} = {f(r['U'], 3)}. Se marca el 95."],
        ["2", f"Media = {f(r['media'], 3)}; s = {f(r['sd'], 3)}; z(95) = ({95} − {f(r['media'], 3)})/{f(r['sd'], 3)} = {f(r['z'][-1], 3)}. No supera 3: no alerta."],
        ["3", f"(8 − 1)/√8 = {f(r['zmax'], 3)} &lt; 3: con 8 datos ningún valor puede tener |z| &gt; 3; el z no sirve para este tamaño."],
        ["4", f"Mediana = {f(med, 1)}; desviaciones absolutas: 5.5, 3.5, 2.5, 0.5, 0.5, 2.5, 4.5, 39.5 → MAD = {f(mad, 1)}. z modificado = 0.6745·39.5/{f(mad, 1)} = {f(zm, 2)} &gt; 3.5: alerta."]])
    m_ = np.array([10.0, 20.0])
    S = np.array([[9.0, 6.0], [6.0, 16.0]])
    Si = np.linalg.inv(S)
    d2 = lambda p: float((np.array(p) - m_) @ Si @ (np.array(p) - m_))
    s += sol("Bloque 2", [
        ["1", f"r = 6/(3·4) = {f(6 / 12, 2)}. det = 9·16 − 36 = 108; S⁻¹ = (1/108)·[[16, −6], [−6, 9]]."],
        ["2", f"P1: d = (3, −4): (16·9 + 2·(−6)·3·(−4) + 9·16)/108 = 432/108 = {f(d2((13, 16)), 3)}. P2: d = (3, 4): (144 − 144 + 144)/108 = {f(d2((13, 24)), 3)}. Ambos están a √25 = 5 del centro."],
        ["3", f"P3: d = (0, 10): 9·100/108 = {f(d2((10, 30)), 3)}. Frente a 9.21: P1 = 4.0 y P2 = 1.33 no alertan; P3 = 8.33 tampoco (cerca del corte)."],
        ["4", "Con correlación positiva (0.5), moverse hacia el mismo lado en ambas variables (P2) es «natural» y moverse en direcciones opuestas (P1) es inusual: Mahalanobis pesa por la dirección de la nube, no solo por la distancia."]])
    xs = [0.0, 1.0, 2.0, 3.0, 9.0]
    kd, vec, lrd, lof = lof_manual(xs, 2)
    s += sol("Bloque 3", [
        ["1", "0: vecinos 1 y 2, k-dist 2. 1: vecinos 0 y 2, k-dist 1. 2: vecinos 1 y 3, k-dist 1. 3: vecinos 2 y 1, k-dist 2. 9: vecinos 3 y 2, k-dist 7."],
        ["2", "El 9, con puntaje 7 (el siguiente mayor es 2). Por kNN se marcaría el 9."],
        ["3", f"reach(9, 3) = max{{6, k-dist(3) = 2}} = 6; reach(9, 2) = max{{7, 1}} = 7; promedio 6.5; lrd(9) = 1/6.5 = {f(lrd[4], 3)}. LOF(9) = ((0.667 + 0.667)/2)/{f(lrd[4], 3)} = {f(lof[4], 2)}. Alerta clara (mucho mayor que 1.5)."],
        ["4", "Con 0, 3, 6, 9, 12 los vecinos del 9 están a 3 y las densidades son parecidas: LOF = 1.25 (menor que 1.5) y no alerta, aunque la distancia al vecino sea 3 (kNN sí lo vería alto). LOF mide densidad relativa."]])
    c16 = c_n(16)
    s += sol("Bloque 4", [
        ["1", "Aislar el 20: basta un corte en (6, 20): 14/20 = 0.70. Aislar el 0: corte en (0, 1): 1/20 = 0.05."],
        ["2", "Porque cualquier corte deja al menos otro valor al mismo lado (hay valores a ambos lados del 3); hacen falta varios cortes."],
        ["3", f"c(16) = 2(2.708 + 0.5772) − 1.875 = {f(c16, 3)}. E[h] = 2: s = 2^(−2/{f(c16, 3)}) = {f(2 ** (-2 / c16), 3)}; E[h] = 8: s = 2^(−8/{f(c16, 3)}) = {f(2 ** (-8 / c16), 3)}."],
        ["4", "E[h] = c(ψ) → s = 2⁻¹ = 0.5 (caso típico). 0.8 indica aislamiento más rápido que el típico (candidato a revisar), pero no es una probabilidad."]])
    s += sol("Bloque 5", [
        ["1", "0.02·4 000 = 80 anomalías. 60/4 000 = 1.5 % de alertas → cuantil 98.5 % del puntaje de entrenamiento."],
        ["2", "Con 60 alertas se detectan como máximo 60 anomalías de 80: sensibilidad ≤ 60/80 = 0.75 (y solo si todas las alertas fueran anomalías)."],
        ["3", "Cuantil 98 % → 2 % de alertas = 80 alertas, máximo 80 VP (sensibilidad 1) solo en el caso ideal; pero supera la capacidad de revisión (60)."],
        ["4", "Porque la prueba debe medir el resultado final sin haber influido en él. El corte se elige con validación cruzada en entrenamiento o con la capacidad de revisión, y la prueba se consulta una vez."]])
    vn_a, fp_a, fn_a, vp_a = 5000 - 150 - 60 + 0 - 0, 110, 60, 40
    vn_a = 5000 - fp_a - fn_a - vp_a
    fp_b, vp_b = 240, 60
    fn_b = 40
    vn_b = 5000 - fp_b - fn_b - vp_b
    f1 = lambda vp, fp, fn: 2 * vp / (2 * vp + fp + fn)
    s += sol("Bloque 6", [
        ["1", f"A: VP 40, FP 110, FN 60, VN {vn_a}: sensibilidad 40/100 = 0.40, precisión 40/150 = {f(40 / 150, 3)}, exactitud {f((vn_a + 40) / 5000, 4)}. B: VP 60, FP 240, FN 40, VN {vn_b}: sensibilidad 0.60, precisión 60/300 = 0.20, exactitud {f((vn_b + 60) / 5000, 4)}. «Todo normal»: 4 900/5 000 = 0.98."],
        ["2", f"Prevalencia 100/5 000 = 0.02. A: {f(40 / 150 / 0.02, 1)} veces; B: {f(60 / 300 / 0.02, 1)} veces. Ambas bastante mejores que el azar, aunque las exactitudes (0.966 y 0.944) están por debajo de 0.98."],
        ["3", f"F1(A) = 80/(80 + 110 + 60) = {f(f1(40, 110, 60), 3)}; F1(B) = 120/(120 + 240 + 40) = {f(f1(60, 240, 40), 3)}. Costos: A = 10·60 + 110 = 710; B = 10·40 + 240 = 640 → se prefiere B (F1 prefiere A: F1 no usa los costos)."],
        ["4", "Ambos: la ROC-AUC resume el orden pero con 2 % de anomalías puede verse optimista; la PR (con la línea base en 0.02) muestra la precisión que se logra para cada sensibilidad. El corte final se fija con costos o capacidad de revisión."]])
    build("apuntes/C41_Ejercicios_Cuaderno_Anomalias.pdf", s, "Ejercicios para el cuaderno · Clase 41")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
