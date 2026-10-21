"""Clase 39 · Clustering: guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c39_numeros.json (lo escribe figuras_c39.py); los de los ejemplos y ejercicios se calculan aquí."""
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import BLUE_BG, ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c39_numeros.json").read_text(encoding="utf-8"))
C39 = colors.HexColor("#0E7490")
C39_BG = colors.HexColor("#ECFEFF")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas clase_39_clustering.pdf + kmeans_numero_clusters.py / .R del profesor Pedro Luiz Ramos"
IT, WB, ES, CD, SL, GP, PF, PC, VL, IN, JR, JT, KM, KP, LM = (N[k] for k in (
    "iter", "wsbs", "estandarizar", "codo", "silueta", "gap", "perfiles", "pca", "validar", "init", "jerarq", "jerarq_toy", "kmodes", "kproto", "limites"))


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ")


def topic_page(s, titulo, idea, conceptos, figuras, formulas, py, salida, cuidado, fw=0.78):
    s.append(band(f"Clase 39 · {titulo}", FUENTE, C39))
    s.append(Spacer(1, 4))
    s.append(box([P(idea, "big")], bg=SOFT, border=C39, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Conceptos clave", "h3"))
    s.append(numbered(conceptos[:3], C39))
    s.append(Spacer(1, 1))
    for fig, pie in figuras:
        s.append(img(fig, width=CONTENT_W * fw))
        s.append(P(pie, "small"))
        s.append(Spacer(1, 2))
    left = [P("Fórmulas y reglas", "h3"), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    if py:
        left += [Spacer(1, 3), box([P(x, "mono") for x in py], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida + ["<b>Cuidado.</b> " + " ".join(cuidado)], "cell", C39)]
    s.append(two_cols(left, right))
    s.append(PageBreak())


# ---------------------------------------------------------------- cálculos de apoyo (ejemplos y ejercicios)
def lloyd1d(x, c0, tope=20):
    x = np.asarray(x, float)
    c = np.array(c0, float)
    hist = []
    prev = None
    for _ in range(tope):
        d = (x[:, None] - c[None, :]) ** 2
        lab = d.argmin(1)
        cn = np.array([x[lab == g].mean() for g in range(len(c))])
        hist.append(dict(lab=lab, c_ini=c.copy(), c_nuevo=cn, d=d))
        if prev is not None and np.array_equal(lab, prev):
            break
        prev, c = lab, cn
    return hist


def sumas(x, lab):
    x = np.asarray(x, float)
    tss = float(((x - x.mean()) ** 2).sum())
    wss = float(sum(((x[lab == g] - x[lab == g].mean()) ** 2).sum() for g in np.unique(lab)))
    return tss, wss, tss - wss


def silueta_punto(x, lab, i):
    x = np.asarray(x, float)
    mi = lab == lab[i]
    a = np.abs(x[mi] - x[i]).sum() / (mi.sum() - 1)
    b = min(np.abs(x[lab == g] - x[i]).mean() for g in np.unique(lab) if g != lab[i])
    return a, b, (b - a) / max(a, b)


def ari_tabla(T):
    from math import comb
    T = np.asarray(T)
    n = int(T.sum())
    sij = sum(comb(int(v), 2) for v in T.ravel())
    sa = sum(comb(int(v), 2) for v in T.sum(1))
    sb = sum(comb(int(v), 2) for v in T.sum(0))
    esp = sa * sb / comb(n, 2)
    return (sij - esp) / (0.5 * (sa + sb) - esp)


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 39 · Clustering: K-means, K-modes y K-prototypes",
                  "Profesor Pedro Luiz Ramos · retoma K-means y PCA (Clases 31 y 32); prepara el Taller 3 (pregunta 4) y se conecta con DBSCAN y Mahalanobis (Clase 41)", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Agrupar no es clasificar</b>: no hay una Y observada que predecir, así que no existe un «grupo correcto» y el «grupo 1» no significa «mejor» ni «positivo». "
                    "<b>K-means</b> busca k centros (medias) que minimicen la suma de distancias al cuadrado de cada punto a su centro (WSS); <b>K-modes</b> hace lo mismo con categorías (modas y desacuerdos) "
                    "y <b>K-prototypes</b> mezcla ambos con un peso γ. Hay que <b>estandarizar</b>, <b>repetir con varios inicios</b> (el algoritmo cae en mínimos locales), <b>elegir k</b> con varios criterios "
                    "(codo, silueta, Gap) y <b>describir los grupos con perfiles</b>. Cuando existe una clase real, sirve solo para comprobar el método, nunca para ajustarlo.", "body")], bg=C39_BG, border=C39, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c39_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>K-means a mano</b> (Clase 31): asignar al centro más cercano y recalcular medias",
                     "<b>Estandarización y PCA</b> (Clases 31 y 32): z = (x − media)/s; componentes para dibujar en dos ejes",
                     "<b>Distancia euclidiana, media y moda</b>; matriz de contingencia (Clase 38)",
                     "<b>R ↔ Python</b>: el profesor trae el mismo flujo en los dos lenguajes"], "cell", C39)]
    right = [P("Ruta de la clase y datos", "h3"),
             bullets(["<b>Sin etiquetas</b> → <b>K-means y WSS/BSS</b> → <b>estandarizar e inicializar</b> → <b>elegir k</b> → <b>perfiles y validación</b> → <b>K-modes y K-prototypes</b>",
                      f"<b>vinos.csv</b>: 177 vinos y 13 mediciones químicas; <b>vinos_clases_reales.csv</b>: clase UCI ({PC['clases_n'][0]}, {PC['clases_n'][1]} y {PC['clases_n'][2]} vinos), solo para validar al final",
                      "Pregunta guía: ¿qué grupos útiles hay y cuánta evidencia tengo de que no son un artefacto del método?"], "cell", C39)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Puentes con otras clases (sin repetirlas)</b>", "cellb"),
                  P("<b>C31/C32 → C39</b>: allí se hizo una iteración de K-means y se vio el efecto de estandarizar; hoy se agregan el costo WSS/BSS, la elección de k y la validación. "
                    "<b>C41 → C39</b>: K-means usa distancia euclidiana (ignora la correlación) y un solo extremo lo desvía; Mahalanobis y DBSCAN, de la Clase 41, son las salidas. "
                    "<b>Taller 3, pregunta 4</b>: esta clase respalda ese ejercicio (K-means con k = 2 a 4, n_init ≥ 30, WSS y silueta media, perfiles de los grupos, y un grupo no es una clase predicha). "
                    "Aquí no se resuelve nada con los datos del taller: la plantilla está en el notebook, con los vinos.", "cell")], bg=SOFT, border=C39, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Recuadro: ¿un grupo es una clase?</b>", "cellb"),
                  P(f"No. Los grupos los fabrica el algoritmo con las variables elegidas; con otra escala, otro k u otro inicio salen otros. En los vinos, K-means con k = 3 coincide con la clase real en {PC['coincidencias']} de 177, "
                    "pero eso se sabe solo porque existe una clase de comparación. Un grupo se justifica si es estable, tiene un perfil interpretable y sirve al propósito del análisis.", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    topic_page(
        s, "K-means: asignar, promediar y el costo WSS",
        "K-means repite dos pasos: <b>asignar</b> cada punto al centro más cercano y <b>recalcular</b> cada centro como la media de su grupo. Cada vuelta <b>baja o mantiene</b> el costo WSS; al no cambiar ninguna asignación, termina.",
        [("El algoritmo", f"Fijar k y centros iniciales → asignar → promediar → repetir. Con los vinos y k = 3 converge en {PC['n_iter']} iteraciones; en el ejemplo de dos variables, en {IT['n_iter']} (WSS {f(IT['wss'][0], 1)} → {f(IT['wss'][-1], 1)})."),
         ("Qué minimiza", f"WSS = Σ de ‖z − μ_grupo‖² (distancia euclidiana <b>al cuadrado</b>). Para un grupo fijo, el centro que minimiza ese costo es la media. Un WSS menor es más compacto, no «más verdadero»."),
         ("TSS = WSS + BSS", f"Variabilidad total = dentro + entre grupos. Diapositiva 17: datos 1, 2, 3, 8, 9, 10 con centros 2 y 9: WSS = 4, BSS = 73.5, TSS = 77.5. En los vinos TSS = 2 288 y con k = 3: WSS = {f(CD['wss_k3'], 2)}, BSS = {f(WB['vinos']['bss'][2], 2)}.")],
        [("c39_iteraciones.png", f"Dos variables estandarizadas (flavonoides y color) con k = 3 y centros iniciales tomados de tres vinos. Los círculos blancos son los centros antes de actualizar y las cruces, después. El WSS final coincide con el de scikit-learn ({f(IT['wss_sklearn'], 2)})."),
         ("c39_wsbs.png", "Izquierda: la iteración 1 de la diapositiva 13. Centro: TSS = WSS + BSS en el ejemplo pequeño. Derecha: en los vinos el TSS (2 288) es fijo; al subir k el WSS baja y el BSS sube.")],
        ["WSS = Σ_g Σ_(i en g) ‖z_i − μ_g‖²",
         "μ_g = media de los z del grupo g",
         "TSS = Σ ‖z_i − z̄‖² = WSS + BSS",
         "BSS = Σ_g n_g ‖μ_g − z̄‖²"],
        ["km = KMeans(3, n_init=50, random_state=2026,", "            algorithm='lloyd').fit(z)", "km.inertia_      # WSS", "km.cluster_centers_; km.labels_ + 1"],
        ["Diap. 13: con centros 1 y 8 los puntos 1, 2, 3 van al grupo 1 y 8, 9, 10 al grupo 2; nuevos centros 2 y 9",
         "Diap. 17: 1 + 0 + 1 + 1 + 0 + 1 = 4 (cuadrados); BSS = 3(2 − 5.5)² + 3(9 − 5.5)² = 73.5",
         "R: <font name='Mono'>tot.withinss</font> = <font name='Mono'>inertia_</font>; <font name='Mono'>betweenss</font> = TSS − WSS"],
        ["El WSS usa distancias <b>al cuadrado</b> (en clase se dijo «sacamos la raíz»; con ceros y unos da igual, con otros números no). El centro es la media, no «una distancia»."])

    topic_page(
        s, "Estandarizar e inicializar",
        "K-means usa distancias, así que una variable con unidades grandes <b>domina</b> el resultado. Y como el algoritmo cae en mínimos locales, hay que <b>repetirlo desde varios inicios</b> y quedarse con el de menor WSS.",
        [("Estandarizar", f"z = (x − media)/s con s muestral (<font name='Mono'>ddof=1</font>, igual que <font name='Mono'>scale()</font>). Ejemplo de la clase: alcohol 14, media 13, desviación 0.8 → z = {f(ES['ej_z']['z'], 2)}. No elimina atípicos ni correlaciones."),
         ("Qué pasa sin estandarizar", f"La prolina (varianza {mil(ES['var']['prolina'])}) aporta el {f(ES['cuota_prolina'] * 100, 1)} % de las distancias; K-means casi solo ve prolina: ARI con la clase real {f(ES['ari_raw'], 3)} contra {f(ES['ari_z'], 3)} estandarizando."),
         ("Inicialización", f"Con inicios aleatorios y un solo ajuste, 25 corridas dieron {IN['n_unicos_random']} valores de WSS entre {f(IN['min_random'], 2)} y {f(IN['max_random'], 2)}. Conservar el mejor de n_init = 50 da {f(IN['mejor50'], 2)}.")],
        [("c39_estandarizar.png", "Izquierda: varianzas originales de las 13 variables (escala logarítmica). Centro: parte de la varianza total que aporta cada variable, con y sin estandarizar. Derecha: concordancia con la clase real (ARI) en ambos casos."),
         ("c39_inicializacion.png", "Izquierda: WSS final de 25 ajustes de un solo inicio (diapositiva 15). Centro: cada corrida termina en un mínimo local (aleatorio y k-means++). Derecha: el WSS que se conserva al aumentar n_init.")],
        ["z_ij = (x_ij − media_j)/s_j",
         "ARI = 1: grupos iguales a las clases; ≈ 0: como al azar",
         "n_init: nº de inicios; se conserva el menor WSS",
         "StandardScaler usa ddof = 0 (divisor n)"],
        ["z = (x - x.mean()) / x.std(ddof=1)", "KMeans(3, init='random', n_init=1)", "KMeans(3, n_init=50)   # conserva el mejor"],
        [f"Con <font name='Mono'>StandardScaler</font> (divisor n) el WSS es {f(ES['wss_z_ddof0'], 2)} en vez de {f(ES['wss_z_ddof1'], 2)} (factor 177/176), con los mismos grupos (ARI {f(ES['ari_ddof0_ddof1'], 1)})",
         f"De 25 inicios aleatorios, {IN['n_optimo_random']} alcanzan el óptimo {f(IN['mejor50'], 2)}; el peor llega a {f(IN['max_random'], 2)} ({f(IN['dif_pct_peor'], 1)} % más)",
         "El profesor lo resumió así: conservar el ajuste de menor WSS entre varias inicializaciones; en R es <font name='Mono'>nstart</font>"],
        ["Se conserva el de <b>menor WSS</b>, no «el que más aparece». Y 50 inicios no garantizan el óptimo global; con k grande conviene aumentar n_init."])

    topic_page(
        s, "Elegir k (1): codo y silueta",
        "No existe un k «verdadero»: el WSS siempre baja al subir k (con k = n vale cero). Por eso se usan criterios adicionales: el <b>codo</b> mira dónde la ganancia se vuelve pequeña y la <b>silueta</b> mide qué tan separados quedan los grupos.",
        [("Codo", f"WSS para k = 1 a 8: {', '.join(mil(w) for w in CD['wss'])}. La reducción cae de {f(CD['reduccion'][1], 0)} (2 → 3) a {f(CD['reduccion'][2], 0)} (3 → 4): el cambio de pendiente sugiere k = 3. No es una prueba formal."),
         ("Silueta", f"s(i) = (b − a)/máx(a, b), con a la distancia media a su grupo y b la menor distancia media a otro grupo. Cerca de 1: bien separado; cerca de 0: zona de superposición; negativo: más cerca de otro grupo. Ejemplo de la clase: a = 2, b = 6 → {f(SL['ej']['s'], 3)}."),
         ("Silueta media", f"Máximo en k = 3 ({f(SL['media_k3'], 3)}); k = 2 y k = 4 dan {f(SL['media'][0], 3)} y {f(SL['media'][2], 3)}. El grupo 3 es el menos nítido: media {f(SL['por_grupo'][2], 3)} y {SL['negativas_por_grupo'][2]} vinos con silueta negativa.")],
        [("c39_codo.png", "Izquierda: el codo en k = 3. Centro: cuánto baja el WSS en cada paso. Derecha: con k hasta 20 el WSS sigue bajando (no hay un k que lo minimice)."),
         ("c39_silueta.png", "Izquierda: gráfico de silueta con k = 3. Centro: silueta media según k. Derecha: a(i) frente a b(i); los vinos bajo la diagonal tienen s < 0.")],
        ["s(i) = (b(i) − a(i)) / máx{a(i), b(i)}",
         "a(i): distancia media a los demás de su grupo",
         "b(i): menor distancia media a otro grupo",
         "se usa la distancia euclidiana, no su cuadrado"],
        ["silhouette_score(z, km.labels_)    # media", "silhouette_samples(z, km.labels_)  # s(i)", "# k = 1: no hay silueta (falta un grupo vecino)"],
        [f"Cálculo del script con 13 variables: silueta media {f(SL['media'][0], 4)}, {f(SL['media'][1], 4)}, {f(SL['media'][2], 4)}, {f(SL['media'][3], 4)}… para k = 2, 3, 4, 5…",
         f"Calculada a mano con a(i) y b(i), la silueta coincide con scikit-learn ({'sí' if SL['manual_igual'] else 'no'}); el vino con silueta más baja es {SL['codigo_peor']} (s = {f(SL['s_peor'], 3)})",
         "Un grupo con una sola observación tiene silueta 0 por convención"],
        ["Un máximo de silueta de 0.28 es modesto: hay superposición. Y el codo y la silueta pueden discrepar; se decide con ambos y con la interpretación."])

    topic_page(
        s, "Elegir k (2): Gap y perfiles de los grupos",
        "El <b>Gap</b> compara el WSS observado con el de datos simulados <b>sin estructura de grupos</b>. Una vez elegido k, se <b>describen los grupos</b> con sus perfiles (en la escala original y estandarizada), no con su número.",
        [("Estadístico Gap", f"Gap(k) = promedio de log W* − log W, con B = 30 bases uniformes en el rango de cada columna. Regla: el primer k con Gap(k) ≥ Gap(k+1) − s(k+1). Aquí da k = {GP['k_gap']}; el máximo de Gap está en k = {GP['k_max_gap']} (la curva no tiene un máximo interior)."),
         ("Cuando discrepan", f"Codo y silueta: k = 3; Gap: k = {GP['k_gap']}. La decisión de k = 3 salió de la mayoría de criterios. La regla es una cuasi-empate: Gap(3) = {f(GP['gap_k3'], 3)} no llega a {f(GP['umbral_k3'], 3)}, que es Gap(4) − s(4)."),
         ("Perfiles", f"Grupos de {PF['n'][0]}, {PF['n'][1]} y {PF['n'][2]} vinos. Grupo 1: más prolina, flavonoides y alcohol; grupo 2: más color y ácido málico, menos matiz y flavonoides; grupo 3: menos alcohol, color y prolina.")],
        [("c39_gap.png", "Izquierda: log W observado frente al de las referencias simuladas. Centro: Gap con barras de error y la regla. Derecha: el ejemplo hipotético de la diapositiva 23 (elige k = 2 aunque Gap sigue subiendo)."),
         ("c39_perfiles.png", "Izquierda: medias estandarizadas por grupo (diapositiva 29). Centro: alcohol por grupo. Derecha: dos de las 13 variables, aunque el ajuste usó las 13.")],
        ["Gap(k) = (1/B) Σ_b log W*_kb − log W_k",
         "s_k = sd(log W*_k) · √(1 + 1/B)",
         "elegir el primer k con Gap(k) ≥ Gap(k+1) − s_{k+1}",
         "perfil = media del grupo (escala original o z)"],
        ["ref = rng.uniform(z.min(), z.max(), size=z.shape)", "gap = log_wss_ref.mean(0) - np.log(wss)", "x.groupby(grupo).mean()   # perfiles"],
        [f"Perfiles originales (alcohol, flavonoides, color): grupo 1: {f(PF['medias']['1']['alcohol'], 2)}, {f(PF['medias']['1']['flavonoides'], 2)}, {f(PF['medias']['1']['color'], 2)}; grupo 2: {f(PF['medias']['2']['alcohol'], 2)}, {f(PF['medias']['2']['flavonoides'], 2)}, {f(PF['medias']['2']['color'], 2)}; grupo 3: {f(PF['medias']['3']['alcohol'], 2)}, {f(PF['medias']['3']['flavonoides'], 2)}, {f(PF['medias']['3']['color'], 2)}",
         f"Con 4 semillas distintas y con B = 60 la regla sigue dando k = {GP['k_gap']}",
         "Los números de grupo son etiquetas: el grupo 1 de R es el 3 de Python (diapositiva 27)"],
        ["Los perfiles son medias del grupo y no describen a cada vino. Y no permiten hablar de precio, calidad ni preferencia."])

    topic_page(
        s, "Validar con las clases reales y con PCA",
        "Los vinos traen una clase real que K-means <b>no usó</b>. Compararla con los grupos sirve para <b>evaluar el método</b> (pureza, ARI), no para elegir k en un problema real, donde esa clase no existe. El PCA solo ayuda a <b>dibujar</b>.",
        [("Tabla de contingencia", f"Con k = 3: coinciden {PC['coincidencias']} de 177 vinos (pureza {f(PC['pureza'] * 100, 1)} %, ARI {f(PC['ari'], 3)}). Las {PC['discrepancias']} discrepancias son vinos de la clase 2: 3 en el grupo 1 y 3 en el grupo 2; 65 de esa clase forman el grupo 3."),
         ("PCA para dibujar", f"Las dos primeras componentes explican {f(PC['acum2'] * 100, 1)} % de la varianza. K-means se ajustó con las 13 variables; ajustarlo sobre las 2 componentes da ARI {f(PC['ari_2pc'], 3)}, casi igual aquí, pero en general se pierde información."),
         ("Qué no es", f"La pureza es descriptiva de estos vinos, no la exactitud en vinos nuevos. Con k distinto de 3 el ARI cae ({f(VL['ari'][2], 2)} con k = 4), pero sin etiquetas ese dato no existe: la silueta no lo reemplaza.")],
        [("c39_pca.png", f"Izquierda: clases reales en el plano de las dos primeras componentes. Centro: grupos de K-means coloreados por su clase mayoritaria; círculos negros, las {PC['discrepancias']} discrepancias. Derecha: tabla de contingencia."),
         ("c39_validar.png", "Izquierda: ARI y silueta media según k. Centro: los vinos que discrepan tienen silueta baja. Derecha: ARI al agrupar con las 13 variables estandarizadas, con 2 componentes o sin estandarizar.")],
        ["pureza = Σ_grupos (máx. clase en el grupo) / n",
         "ARI: 1 = igual, ≈ 0 = azar, puede ser negativo",
         "el número de grupo es arbitrario: cambia al permutar",
         "T = pd.crosstab(clase_real, grupo)"],
        ["pd.crosstab(clase_real, grupo)", "adjusted_rand_score(clase_real, grupo)", "PCA(2).fit_transform(z)   # solo para dibujar"],
        [f"Los {PC['discrepancias']} vinos que discrepan ({', '.join(PC['ids_discrepancia'])}) tienen silueta media {f(VL['sil_media_discrepan'], 3)}, contra {f(VL['sil_media_coinciden'], 3)} del resto",
         f"Sin estandarizar el ARI baja a {f(VL['ari_raw'], 3)}; con 2 componentes, {f(VL['ari_2pc'], 3)}; con las 13 estandarizadas, {f(VL['ari_13'], 3)}",
         "Recomendación del profesor: agrupar con la base original y usar el PCA solo para visualizar"],
        ["Comparar con una clase real valida el método en un caso de laboratorio. En la práctica no hay clase real: se valida con estabilidad, perfiles y propósito."])

    topic_page(
        s, "K-modes y K-prototypes",
        "Si las variables son categóricas no se puede promediar: el centro es la <b>moda</b> y la distancia cuenta <b>desacuerdos</b> (K-modes). Con variables mixtas se suman un término de medias y otro de modas ponderado por <b>γ</b> (K-prototypes).",
        [("K-modes", f"Disimilitud d(x, m) = nº de variables con valor distinto. Diap. 33: cliente (Gold, App, Sur) frente a (Gold, App, Norte): {KM['dis_cliente_A']}; frente a (Silver, Web, Sur): {KM['dis_cliente_B']}. Con alcohol, flavonoides y color categorizados, el costo con k = 4 es {f(KM['costo_k4'], 0)}."),
         ("K-prototypes", f"d = Σ_num (x − μ)² + γ Σ_cat 1(x ≠ moda). Diap. 43: observación (1, 2) y (App, Sur): costo 1 + γ frente a 2; con γ = 0.5 gana A y con γ = 3 gana B. Con ceniza y matiz estandarizados más 3 categóricas, k = 3 y γ = 1: costo {f(KP['costo_k3'], 2)}."),
         ("El peso importa", f"Con γ = 0.25, 1 y 4 los grupos son de {KP['por_gamma']['0.25']['tamanos']}, {KP['por_gamma']['1']['tamanos']} y {KP['por_gamma']['4']['tamanos']}; entre γ = 0.25 y 1 el ARI es {f(KP['ari_025_vs_1'], 2)}. Sin variable respuesta no hay ROC ni validación cruzada para elegir γ.")],
        [("c39_kmodes.png", f"Izquierda: costo de K-modes para k = 1 a 8 (sin codo claro). Centro: modas de cada grupo con k = 4. Derecha: el perfil del grupo {KM['g_color_alto']}: color alto en {KM['color_g']['alto']} de {KM['n_g']} vinos."),
         ("c39_kprototypes.png", "Izquierda: costo de K-prototypes (γ = 1). Derecha: tres particiones con k = 3 y γ = 0.25, 1 y 4, vistas en ceniza y matiz (colores alineados por el centro de matiz).")],
        ["K-modes: d(x, m) = Σ_j 1(x_j ≠ m_j)",
         "K-prototypes: Σ_num (x_j − μ_j)² + γ Σ_cat 1(x_j ≠ m_j)",
         "centro: media (numérica) o moda (categórica)",
         "costos de métodos distintos no son comparables"],
        ["KModes(n_clusters=4, init='Huang', n_init=20)", "KPrototypes(n_clusters=3, gamma=1, init='Huang')", "kp.fit(X, categorical=[2, 3, 4])"],
        [f"Costos de la diap. 52: K-means {f(CD['wss_k3'], 2)} (k = 3), K-modes {f(KM['costo_k4'], 0)} (k = 4), K-prototypes {f(KP['costo_k3'], 2)} (k = 3): <b>no se comparan</b> (unidades y funciones de costo distintas)",
         f"Diap. 41: grupo de K-modes con n = {KM['n_g']}: color alto {KM['color_g']['alto']} (83.3 %), medio {KM['color_g']['medio']} (14.6 %), bajo {KM['color_g']['bajo']} (2.1 %)",
         "Elegir método: solo numéricas → K-means; solo categóricas → K-modes; mezcla → K-prototypes"],
        ["Categorizar una variable continua pierde información (cortes docentes, no umbrales enológicos). Y γ no es la penalización de Ridge: es un peso entre dos partes de la distancia."])

    # ------------------------------------------------------------ alertas
    s.append(band("Alertas del material, repaso y criterio de dominio", "Diferencias entre materiales y qué conviene precisar (verificado con código)", NAVY))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Lo dicho en clase (transcripción)</b>", "cellb"),
                  bullets(["<b>«WSS: sacamos la raíz»</b> (ejemplo 1, 2, 3 | 8, 9, 10): el WSS suma distancias <b>al cuadrado</b>; el 4 sale igual porque los valores son 0 y 1.",
                           f"<b>«Los clústeres que más aparecieron»</b> (inicios): el criterio es el <b>menor WSS</b>. Y «50 mil iteraciones»: con los vinos convergen en {PC['n_iter']} (el script usa un máximo de 100).",
                           f"<b>Silueta «casi 0.36»</b>: es {f(SL['media_k3'], 3)} con k = 3 (y {f(SL['media'][0], 3)} con k = 2, como se dijo). Gap: la regla usa s(k+1), no el «valor anterior».",
                           f"<b>«Más de 90 %, 92 %; 8 o 9 mal clasificados»</b>: son {PC['discrepancias']} de 177 (pureza {f(PC['pureza'] * 100, 1)} %), todos de la clase 2. «Casi 56 %» de varianza en el PCA: {f(PC['acum2'] * 100, 1)} %.",
                           "<b>«Vino blanco, Merlot, Cabernet»</b>: la base (UCI Wine) tiene tres cultivares de una misma región de Italia; el CSV no trae sus nombres. La diapositiva 6 aclara que son 177 filas (la fuente tiene 178).",
                           "<b>Cifras de la transcripción</b>: «3.68» y «13.25» de alcohol son 13.68 y 12.25; «157 observaciones» son 177; «grupo 2: ceniza 0.5, matiz −0.22» es ceniza 0.225 y matiz −1.258 (diap. 49). Los números de grupo mezclan R y Python.",
                           "<b>K-modes</b>: flavonoides «alto» es ≥ 3 (se dijo «menor que 3»). «Sabemos que son 3» usa la clase real, que no existe en la práctica."], "cell", AMBER)], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Diapositivas, scripts y verificación</b>", "cellb"),
                  bullets([f"<b>Estandarizar</b> (diap. 7 y 9): <font name='Mono'>ddof=1</font> reproduce <font name='Mono'>scale()</font>; <font name='Mono'>StandardScaler</font> usa ddof = 0 y da WSS {f(ES['wss_z_ddof0'], 2)} (los mismos grupos).",
                           f"<b>Gap</b> (diap. 55 a 57): con B = 30, cuatro semillas más y B = 60 la regla da k = {GP['k_gap']}; es un resultado ajustado (Gap(3) = {f(GP['gap_k3'], 3)} frente a {f(GP['umbral_k3'], 3)}). El script de Python usa desviación con ddof = 1 y R, según su documentación, el divisor B; no se pudo ejecutar R aquí.",
                           f"<b>PCA y agrupar</b>: «agrupar con las 13 variables» es lo correcto en principio, pero aquí las 2 componentes dan ARI {f(PC['ari_2pc'], 3)} frente a {f(PC['ari'], 3)}: sin diferencia relevante en estos datos.",
                           f"<b>Verificado</b>: WSS {f(CD['wss_k3'], 2)}, BSS {f(WB['vinos']['bss'][2], 2)}, TSS 2 288, grupos de {PF['n'][0]}/{PF['n'][1]}/{PF['n'][2]}, silueta {f(SL['media_k3'], 3)}, K-modes {f(KM['costo_k4'], 0)}, K-prototypes {f(KP['costo_k3'], 2)} con grupos de {KP['tamanos_k3'][0]}/{KP['tamanos_k3'][1]}/{KP['tamanos_k3'][2]}: todo coincide con las diapositivas.",
                           f"<b>Jerárquico</b> (no se vio en clase): con enlace de Ward el ARI es {f(JR['ari']['ward'], 2)}; con single o average hay efecto cadena (grupos de 173, 3 y 1; ARI ≈ 0); K-means obtiene {f(JR['ari_kmeans'], 2)}.",
                           "<b>Transcripción con ruido</b>: «camins / camín» = K-means, «Camos / KAMMOD» = K-modes, «capot» = K-prototypes, «GEP» = Gap, «cilita / siluente» = silueta, «interacción» = iteración, «VWSS» = WSS. El recreo (hacia 1:08:40) quedó marcado y el audio termina a las 2:07:47 en la pregunta sobre γ."], "cell", RED)], bg=RED_BG, border=RED, pad=5))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Checklist, criterio de dominio y lo que viene</b>", "cellb"),
                  bullets(["Hago una iteración de K-means a mano y calculo WSS, BSS y TSS",
                           "Explico por qué se estandariza y qué pasa si no (con un ejemplo numérico)",
                           "Calculo la silueta de un punto y leo una tabla de WSS y silueta para elegir k",
                           "Explico por qué se repite con varios inicios y qué conserva n_init",
                           "Describo los grupos con perfiles y distingo grupo de clase (tabla de contingencia, ARI)",
                           "Elijo entre K-means, K-modes y K-prototypes según el tipo de variables",
                           "Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos",
                           "<b>Siguiente</b>: Clase 41 (Mahalanobis y DBSCAN) y el Taller 3, pregunta 4 (K-means con k = 2 a 4)."], "cell", GREEN, SYM)], bg=GREEN_BG, border=GREEN, pad=5))
    build("C39_Guia_Clustering.pdf", s, "Clase 39 · Clustering")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#0E7490">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 39 · Clustering", "Para leer antes de los ejercicios · lectura de ~30 min", C39))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 6 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Agrupar no es clasificar", "No hay Y observada; el número del grupo es una etiqueta arbitraria."),
                            ("K-means minimiza el WSS", "Asigna al centro más cercano y promedia; cada vuelta baja o mantiene el costo."),
                            ("Estandariza siempre", "Si no, la variable de mayor escala decide los grupos."),
                            ("Repite con varios inicios", "Hay mínimos locales: se conserva el menor WSS (n_init, nstart)."),
                            ("k no es único", "Codo, silueta y Gap pueden discrepar; decide con interpretación y estabilidad."),
                            ("Categóricas y mixtas", "K-modes usa modas y desacuerdos; K-prototypes suma medias y modas con un peso γ.")], C39)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y una <b>pista</b> para el notebook <font name='Mono'>C39_Clustering.ipynb</font> o el cuaderno <font name='Mono'>C39_Ejercicios_Cuaderno_Clustering.pdf</font>.", "small"))

    # 1 K-means a mano
    s.append(h("1", "K-means a mano: asignar, promediar y WSS"))
    s += para("Con k centros iniciales: (1) cada punto va al centro más cercano (distancia al cuadrado); (2) cada centro pasa a ser la media de su grupo; se repite hasta que no cambie ninguna asignación. El costo es WSS = Σ (x − μ_grupo)²; la variabilidad total se descompone como TSS = WSS + BSS, con BSS = Σ n_g (μ_g − media global)².")
    x = np.array([3, 5, 6, 14, 16, 22.0])
    h1 = lloyd1d(x, [5, 14])
    c_fin = h1[-1]["c_nuevo"]
    tss, wss, bss = sumas(x, h1[-1]["lab"])
    s.append(ejemplo("seis valores: 3, 5, 6, 14, 16 y 22, con k = 2 y centros iniciales 5 y 14", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Iteración 1: asignar", "(x − 5)² frente a (x − 14)²", "grupo 1: 3, 5, 6 · grupo 2: 14, 16, 22"],
               ["Actualizar", "(3 + 5 + 6)/3 y (14 + 16 + 22)/3", f"μ₁ = {f(c_fin[0], 3)}, μ₂ = {f(c_fin[1], 3)}"],
               ["Iteración 2: asignar", "con los nuevos centros", "mismas asignaciones: converge"],
               ["WSS", f"4.667 + 34.667 (por grupo)", f"{f(wss, 3)}"],
               ["TSS", "media global 11: 64 + 36 + 25 + 9 + 25 + 121", f"{f(tss, 0)}"],
               ["BSS", f"3({f(c_fin[0], 3)} − 11)² + 3({f(c_fin[1], 3)} − 11)²", f"{f(bss, 3)}  (= TSS − WSS)"]],
              [38 * mm, 74 * mm, CONTENT_W - 112 * mm - 18]),
        P("Lectura: aquí el 14 estaba justo en el centro 2 y el 5 en el centro 1, por eso converge enseguida; con otros inicios podría tardar más. <b>Error común</b>: restar sin elevar al cuadrado, o calcular el WSS con el centro inicial en vez del final.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_iteraciones.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A2 y A3) y los Bloques 1 y 2 del cuaderno", [P("Programa las dos iteraciones con NumPy y compara el WSS con <font name='Mono'>KMeans(...).inertia_</font>.", "cell")]))

    # 2 estandarizar
    s.append(h("2", "Estandarizar y por qué importa"))
    s += para("K-means compara distancias. Si una variable tiene unidades mucho mayores, casi sola decide a qué centro va cada punto. Estandarizar (z = (x − media)/s) deja todas las variables en desviaciones estándar. No elimina atípicos ni correlaciones. En Python usa <font name='Mono'>ddof=1</font> para igualar el <font name='Mono'>scale()</font> de R.")
    A, B, C = np.array([12.0, 90.0]), np.array([14.0, 95.0]), np.array([12.2, 120.0])
    mu, sd = np.array([13.0, 100.0]), np.array([0.8, 15.0])
    za, zb, zc = (A - mu) / sd, (B - mu) / sd, (C - mu) / sd
    dr = lambda u, v: float(np.linalg.norm(u - v))
    s.append(ejemplo("tres vinos (alcohol, magnesio): A = (12.0, 90), B = (14.0, 95), C = (12.2, 120); medias 13 y 100, desviaciones 0.8 y 15", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Distancias sin estandarizar", "d(A, B) = √(2² + 5²); d(A, C) = √(0.2² + 30²)", f"{f(dr(A, B), 2)} y {f(dr(A, C), 2)}"],
               ["Vecino más cercano de A", "sin estandarizar", "B (el magnesio decide)"],
               ["z de A, B, C", "(x − media)/desviación", f"A({f(za[0], 2)}; {f(za[1], 2)}) B({f(zb[0], 2)}; {f(zb[1], 2)}) C({f(zc[0], 2)}; {f(zc[1], 2)})"],
               ["Distancias estandarizadas", "d(A, B), d(A, C)", f"{f(dr(za, zb), 2)} y {f(dr(za, zc), 2)}"],
               ["Vecino más cercano de A", "estandarizando", "C (ahora pesan las dos variables)"]],
              [46 * mm, 72 * mm, CONTENT_W - 118 * mm - 18]),
        P("Lectura: el cambio de escala altera quién es «vecino» de quién, y por tanto los grupos. En los vinos reales la prolina aporta casi 100 % de las distancias. <b>Error común</b>: estandarizar con la desviación de un subconjunto distinto o olvidar guardar la media y la desviación.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_estandarizar.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A del notebook (A1 y A6) y el Bloque 6 del cuaderno", [P("Compara K-means con y sin estandarizar y mide la concordancia con la clase real (ARI).", "cell")]))

    # 3 elegir k
    s.append(h("3", "Elegir k: codo, silueta y Gap"))
    s += para("El WSS baja siempre al subir k (con k = n vale 0), así que no basta: el <b>codo</b> busca dónde la reducción se vuelve pequeña; la <b>silueta media</b> se maximiza; el <b>Gap</b> compara con datos sin estructura y toma el primer k con Gap(k) ≥ Gap(k+1) − s(k+1). Si discrepan, se razona con interpretación y estabilidad.")
    g1d = np.array([1, 2, 3, 9, 10.0])
    lab1 = np.array([0, 0, 0, 1, 1])
    a3, b3, s3 = silueta_punto(g1d, lab1, 2)
    a4, b4, s4 = silueta_punto(g1d, lab1, 3)
    gh = np.array([0.55, 0.82, 0.83, 0.84])
    sh = np.array([0.03, 0.03, 0.03, 0.03])
    s.append(ejemplo("silueta de dos puntos (grupos {1, 2, 3} y {9, 10}) y regla del Gap con una tabla hipotética", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Punto 3: a", "(|3 − 1| + |3 − 2|)/2", f"{f(a3, 2)}"],
               ["Punto 3: b", "(|3 − 9| + |3 − 10|)/2", f"{f(b3, 2)}"],
               ["Punto 3: s", "(b − a)/máx(a, b)", f"{f(s3, 3)} (bien separado)"],
               ["Punto 9: a, b", "a = |9 − 10| = 1;  b = (8 + 7 + 6)/3", f"{f(a4, 2)} y {f(b4, 2)}"],
               ["Punto 9: s", "(7 − 1)/7", f"{f(s4, 3)}"],
               ["Gap(k): 1, 2, 3, 4", "0.55, 0.82, 0.83, 0.84 (s = 0.03)", "k = 1: 0.55 ≥ 0.79? no"],
               ["Regla", "k = 2: 0.82 ≥ 0.83 − 0.03 = 0.80?", "sí → se elige k = 2"]],
              [34 * mm, 76 * mm, CONTENT_W - 110 * mm - 18]),
        P("Lectura: el Gap sigue subiendo (0.84 es el mayor) pero la regla de un error estándar elige el primer k suficiente. <b>Error común</b>: tomar el máximo del Gap o decidir k solo con el codo.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_silueta.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte B del notebook (B1 a B4) y el Bloque 4 del cuaderno", [P("Reproduce el codo, la silueta y el Gap del profesor y decide k con los tres.", "cell")]))

    # 4 iniciar y validar
    s.append(h("4", "Inicialización y validación con clases"))
    s += para("K-means puede terminar en mínimos locales según los centros iniciales: se repite (n_init, nstart) y se conserva el menor WSS. Si existe una clase real, se compara con los grupos mediante una tabla de contingencia, la pureza y el índice de Rand ajustado (ARI); los números de grupo son arbitrarios, por eso se comparan <b>particiones</b>, no etiquetas.")
    T = np.array([[18, 2], [3, 27]])
    pur = T.max(axis=0).sum() / T.sum()
    s.append(ejemplo("50 casos con clase real (A: 20, B: 30) agrupados en 2 grupos", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Tabla", "filas A y B; columnas grupo 1 y 2", "A: 18 y 2 · B: 3 y 27"],
               ["Pureza", "(18 + 27)/50", f"{f(pur, 2)}"],
               ["Pares dentro de celdas", "C(18,2) + C(2,2) + C(3,2) + C(27,2)", f"{153 + 1 + 3 + 351}"],
               ["ARI", "(pares − esperados)/(promedio − esperados)", f"{f(ari_tabla(T), 3)}"],
               ["Si se intercambian los números de grupo", "las columnas se permutan", "pureza y ARI no cambian"]],
              [46 * mm, 76 * mm, CONTENT_W - 122 * mm - 18]),
        P("Lectura: 5 casos «mezclados» bastan para bajar el ARI a 0.63 aunque la pureza sea 0.90. <b>Error común</b>: usar la pureza o el ARI para elegir k en un proyecto real: sin etiquetas no existen.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_inicializacion.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(img("c39_validar.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte A (A5) y la Parte C del notebook y el Bloque 7 del cuaderno", [P("Mide cuántos mínimos locales aparecen con un solo inicio y arma la tabla de contingencia con las clases reales.", "cell")]))

    # 5 K-modes y K-prototypes
    s.append(h("5", "K-modes y K-prototypes"))
    s += para("Con categorías no tiene sentido promediar: el centro es la moda de cada variable y la disimilitud cuenta variables con valores distintos. En K-prototypes, el costo de un punto frente a un prototipo es Σ (x_num − μ)² + γ · (nº de desacuerdos). Cambiar γ puede cambiar la asignación aunque los datos sean los mismos.")
    cl = ("Silver", "Web", "Norte")
    pA, pB = ("Gold", "App", "Norte"), ("Silver", "Web", "Sur")
    dis = lambda a, b: sum(u != v for u, v in zip(a, b))
    cA, cB = (1 - 1) ** 2 + (1 - 1) ** 2, (2 - 3) ** 2 + (1 - 2) ** 2
    s.append(ejemplo("cliente (Silver, Web, Norte) y observación (2, 1) con (Web, Norte)", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Desacuerdos con A", "(Gold, App, Norte)", f"{dis(cl, pA)}"],
               ["Desacuerdos con B", "(Silver, Web, Sur)", f"{dis(cl, pB)}"],
               ["Asignación K-modes", "menor desacuerdo", "grupo B"],
               ["Costo mixto con P1: (1, 1) y (Web, Sur)", "(2 − 1)² + (1 − 1)² + γ·1", "1 + γ"],
               ["Costo mixto con P2: (3, 2) y (Web, Norte)", "(2 − 3)² + (1 − 2)² + γ·0", "2"],
               ["Decisión", "γ = 0.5: 1.5 < 2 → P1;  γ = 3: 4 > 2 → P2", "cambia con γ"]],
              [56 * mm, 76 * mm, CONTENT_W - 132 * mm - 18]),
        P("Lectura: en γ = 1 hay empate. <b>Error común</b>: comparar el costo de K-modes (desacuerdos) con el de K-means (cuadrados) para elegir el método.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_kmodes.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(img("c39_kprototypes.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D1 y D2)", [P("Reproduce los costos de las diapositivas 40, 48 y 52 con el paquete <font name='Mono'>kmodes</font> y varía γ.", "cell")]))

    # 6 jerárquico y límites
    s.append(h("6", "Complemento: agrupamiento jerárquico y límites de K-means"))
    s += para("Este tema <b>no se vio en clase</b>: se incluye porque aparece en cursos y en el taller. El jerárquico aglomerativo parte con cada punto solo y une los dos grupos más cercanos; el <b>enlace</b> define la distancia entre grupos: <i>single</i> (mínima), <i>complete</i> (máxima), <i>average</i> y <i>Ward</i> (aumento de la suma de cuadrados). K-means, en cambio, supone grupos compactos y de tamaño parecido.")
    xs = np.array([1.0, 3.0, 8.0, 12.0])
    ls, lc = linkage(xs[:, None], "single"), linkage(xs[:, None], "complete")
    s.append(ejemplo("cuatro puntos en una recta: 1, 3, 8 y 12", [
        pasos([["Paso", "Single", "Complete"],
               ["1.º: unir {1, 3}", f"altura {f(ls[0, 2], 0)}", f"altura {f(lc[0, 2], 0)}"],
               ["2.º: unir {8, 12}", f"altura {f(ls[1, 2], 0)} (el 8 y el 12 están a 4)", f"altura {f(lc[1, 2], 0)}"],
               ["3.º: unir los dos pares", f"altura {f(ls[2, 2], 0)} (mínima entre grupos: 8 − 3)", f"altura {f(lc[2, 2], 0)} (máxima: 12 − 1)"]],
              [50 * mm, 70 * mm, CONTENT_W - 120 * mm - 18]),
        P("Lectura: el mismo árbol, alturas muy distintas; con muchos puntos <i>single</i> encadena (en los vinos produce grupos de 173, 3 y 1). <b>Error común</b>: leer la altura como si fuera comparable entre enlaces.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(img("c39_jerarquico.png", width=CONTENT_W * 0.95))
    s.append(Spacer(1, 2))
    s.append(img("c39_limites.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(pista("la Parte D del notebook (D3 y D4) y el Bloque 5 del cuaderno", [P("Compara los enlaces por ARI y silueta y reproduce las formas curvas con DBSCAN (Clase 41).", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Agrupar no es clasificar: un grupo no es una clase y su número es solo una etiqueta.",
                       "Estandariza, repite con varios inicios y elige k con más de un criterio.",
                       "Los costos de K-means, K-modes y K-prototypes no son comparables; γ cambia los grupos.",
                       "Orden sugerido: apuntes → Bloques 1–4 del cuaderno → Partes A–B del notebook → Bloques 5–7 → Partes C–D → ejercicio de cierre."]))
    build("apuntes/C39_Apuntes_Clustering.pdf", s, "Apuntes · Clase 39 · Clustering")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0E7490">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C39))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 39 · Clustering", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C39))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: WSS = Σ(x − μ_g)² · TSS = WSS + BSS · s(i) = (b − a)/máx(a, b) · z = (x − media)/s · ΔSS(Ward) = n₁n₂/(n₁ + n₂)·d²(centros) · pureza = Σ máx. por grupo/n.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C39)], bg=SOFT, border=C39, pad=5))
    s.append(Spacer(1, 2))
    x = np.array([2, 3, 7, 10, 12, 21.0])
    h1 = lloyd1d(x, [3, 12])
    lab = h1[-1]["lab"]
    cfin = h1[-1]["c_nuevo"]
    tss, wss, bss = sumas(x, lab)
    s += bloque(1, "Una iteración de K-means a mano", "★★", "Valores 2, 3, 7, 10, 12 y 21, con k = 2 y centros iniciales 3 y 12.", [
        "Asigna cada valor al centro más cercano (distancia al cuadrado) y escribe los grupos.",
        "Calcula los nuevos centros. Repite la asignación: ¿cambia algo? ¿Convergió?",
        "Calcula el WSS final."])
    s += bloque(2, "WSS, BSS y TSS", "★★", "Usa la partición final del Bloque 1.", [
        "Calcula la media global y el TSS.",
        "Calcula el BSS con n_g(μ_g − media global)² y comprueba que TSS = WSS + BSS.",
        "¿Qué fracción de la variabilidad total explica la partición (BSS/TSS)?"])
    s += bloque(3, "Silueta de un punto", "★★", "Con la partición del Bloque 1, grupos {2, 3, 7} y {10, 12, 21}.", [
        "Calcula a(i), b(i) y s(i) del valor 7 (distancia |x − y|).",
        "Calcula s(i) del valor 10. ¿Qué indica el signo?",
        "¿Por qué K-means lo dejó en el grupo 2 si su silueta es negativa?"])
    ks = np.array([1, 2, 3, 4, 5, 6])
    W = np.array([500, 300, 180, 150, 128, 112.0])
    Ssil = np.array([np.nan, 0.41, 0.55, 0.46, 0.39, 0.33])
    Gp = np.array([0.31, 0.55, 0.74, 0.75, 0.76, 0.76])
    Gs = np.array([0.03] * 6)
    s += bloque(4, "Leer una tabla para elegir k", "★★★", "WSS: 500, 300, 180, 150, 128 y 112 para k = 1 a 6. Silueta media (k = 2 a 6): 0.41, 0.55, 0.46, 0.39 y 0.33. Gap (k = 1 a 5): 0.31, 0.55, 0.74, 0.75 y 0.76, con s = 0.03.", [
        "Calcula la reducción porcentual del WSS en cada paso. ¿Dónde está el codo?",
        "¿Qué k elige la silueta? Aplica la regla Gap(k) ≥ Gap(k + 1) − s(k + 1) y di qué k elige.",
        "Si los criterios discreparan, ¿qué más revisarías antes de decidir?"])
    s += bloque(5, "Enlaces: single, complete y Ward", "★★★", "Puntos en una recta A = 0, B = 1, C = 2.5, D = 4.1 y E = 5.8 (distancias |x − y|).", [
        "Con enlace single, escribe el orden de las uniones y sus alturas.",
        "Con enlace complete, igual. ¿En qué se parecen y en qué se diferencian los dos árboles?",
        "Ward: después de unir {A, B} y {C, D}, calcula ΔSS = n₁n₂/(n₁ + n₂)·d² para unir {C, D} con E y para unir {A, B} con {C, D}. ¿Cuál se une primero?"])
    mu, sd = np.array([13.0, 750.0]), np.array([0.8, 315.0])
    PA, PB, PCc = np.array([13.0, 700.0]), np.array([14.5, 720.0]), np.array([13.1, 1100.0])
    z = lambda p: (p - mu) / sd
    s += bloque(6, "Por qué estandarizar", "★★", "Tres vinos (alcohol, prolina): A = (13.0, 700), B = (14.5, 720), C = (13.1, 1100). Medias 13 y 750; desviaciones 0.8 y 315.", [
        "Calcula d(A, B) y d(A, C) sin estandarizar. ¿Cuál es el vecino más cercano de A?",
        "Estandariza los tres vinos y recalcula d(A, B) y d(A, C). ¿Cambia el vecino?",
        "Explica con una frase por qué ocurre y qué variable dominaba."])
    Tc = np.array([[40, 5, 5], [4, 38, 8], [3, 2, 45]])
    s += bloque(7, "Grupo no es clase: leer una contingencia", "★★★", "Clase real (filas A, B, C) frente a grupos 1, 2 y 3: A = (40, 5, 5), B = (4, 38, 8), C = (3, 2, 45).", [
        "Calcula la pureza. ¿Qué clase domina en cada grupo?",
        "Si se renumeran los grupos (1 ↔ 3), ¿cambian la pureza y el ARI? ¿Y la tabla?",
        "¿Se podría usar esta pureza para elegir k en un proyecto sin clase real? Justifica."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C39))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C39), Spacer(1, 2)]

    d0 = h1[0]["d"]
    s += sol("Bloque 1", [
        ["1", f"Distancias a 3 y a 12: 2 → (1, 100); 3 → (0, 81); 7 → (16, 25); 10 → (49, 4); 12 → (81, 0); 21 → (324, 81). Grupo 1: 2, 3, 7; grupo 2: 10, 12, 21."],
        ["2", f"μ₁ = (2 + 3 + 7)/3 = {f(h1[0]['c_nuevo'][0], 2)}; μ₂ = (10 + 12 + 21)/3 = {f(h1[0]['c_nuevo'][1], 3)}. Con estos centros el 7 queda a 3 de μ₁ y a 7.33 de μ₂; el 10, a 6 y 4.33: las asignaciones no cambian. Convergió."],
        ["3", f"Grupo 1: 4 + 1 + 9 = 14. Grupo 2: ({f(10 - cfin[1], 3)})² + ({f(12 - cfin[1], 3)})² + ({f(21 - cfin[1], 3)})² = {f(((x[3:] - cfin[1]) ** 2).sum(), 3)}. WSS = {f(wss, 3)}."]])
    s += sol("Bloque 2", [
        ["1", f"Media global = 55/6 = {f(x.mean(), 3)}. TSS = Σ(x − media)² = {f(tss, 3)}."],
        ["2", f"BSS = 3({f(cfin[0], 2)} − {f(x.mean(), 3)})² + 3({f(cfin[1], 3)} − {f(x.mean(), 3)})² = {f(bss, 3)}. Comprobación: {f(wss, 3)} + {f(bss, 3)} = {f(tss, 3)}."],
        ["3", f"BSS/TSS = {f(bss / tss, 3)}: la partición explica el {f(bss / tss * 100, 1)} % de la variabilidad total."]])
    a7, b7, s7 = silueta_punto(x, lab, 2)
    a10, b10, s10 = silueta_punto(x, lab, 3)
    s += sol("Bloque 3", [
        ["1", f"a = (|7 − 2| + |7 − 3|)/2 = {f(a7, 2)}; b = (|7 − 10| + |7 − 12| + |7 − 21|)/3 = {f(b7, 3)}; s = ({f(b7, 3)} − {f(a7, 2)})/{f(b7, 3)} = {f(s7, 3)}."],
        ["2", f"a = (|10 − 12| + |10 − 21|)/2 = {f(a10, 2)}; b = (8 + 7 + 3)/3 = {f(b10, 2)}; s = ({f(b10, 2)} − {f(a10, 2)})/{f(a10, 2)} = {f(s10, 3)}. Negativo: en promedio está más cerca del otro grupo."],
        ["3", "K-means compara con el <b>centro</b>: el 10 está a 4.33 de μ₂ = 14.33 y a 6 de μ₁ = 4. La silueta compara <b>distancias medias a todos</b> los puntos: el 21 eleva la distancia media al grupo 2 (6.50) por encima de la media al grupo 1 (6.00). Criterios distintos, resultado distinto."]])
    red = [(W[i - 1] - W[i]) / W[i - 1] * 100 for i in range(1, 6)]
    s += sol("Bloque 4", [
        ["1", f"Reducciones: {', '.join(f(r, 1) + ' %' for r in red)} (k = 1→2, …, 5→6). Las caídas fuertes (40 % y 40 %) llegan hasta k = 3; después son de 16.7 %, 14.7 % y 12.5 %: codo en k = 3."],
        ["2", "La silueta es máxima en k = 3 (0.55). Gap: k = 1: 0.31 ≥ 0.55 − 0.03? no; k = 2: 0.55 ≥ 0.74 − 0.03 = 0.71? no; k = 3: 0.74 ≥ 0.75 − 0.03 = 0.72? sí → k = 3."],
        ["3", "Los tres coinciden. Aun así: tamaños y estabilidad de los grupos (otras semillas y n_init), perfiles interpretables y el propósito del análisis."]])
    xs = np.array([0, 1, 2.5, 4.1, 5.8])
    s += sol("Bloque 5", [
        ["1", "Single: {A, B} a 1.0; luego C a 1.5 (distancia B–C); luego D a 1.6 (C–D); luego E a 1.7 (D–E): efecto cadena."],
        ["2", "Complete: {A, B} a 1.0; {C, D} a 1.6; E se une a {C, D} a 3.3 (máx. con C); al final todo a 5.8 (máx. A–E). Single une todo en una cadena; complete forma dos grupos compactos y E se une a ellos."],
        ["3", f"{{C, D}} + E: centros 3.3 y 5.8, d² = 6.25, ΔSS = (2·1/3)·6.25 = {f(2 / 3 * 6.25, 3)}. {{A, B}} + {{C, D}}: centros 0.5 y 3.3, d² = 7.84, ΔSS = (2·2/4)·7.84 = {f(7.84, 2)}. Se une primero {{C, D}} con E (ΔSS menor)."]])
    s += sol("Bloque 6", [
        ["1", f"d(A, B) = √(1.5² + 20²) = {f(np.linalg.norm(PA - PB), 2)}; d(A, C) = √(0.1² + 400²) = {f(np.linalg.norm(PA - PCc), 2)}. El más cercano es B."],
        ["2", f"z(A) = (0; {f(z(PA)[1], 3)}); z(B) = ({f(z(PB)[0], 3)}; {f(z(PB)[1], 3)}); z(C) = ({f(z(PCc)[0], 3)}; {f(z(PCc)[1], 3)}). d(A, B) = {f(np.linalg.norm(z(PA) - z(PB)), 3)}; d(A, C) = {f(np.linalg.norm(z(PA) - z(PCc)), 3)}. Ahora el más cercano es C: el vecino cambió."],
        ["3", "La prolina (cientos de unidades) dominaba la distancia sin estandarizar; al pasar a desviaciones estándar el alcohol y la prolina pesan comparable."]])
    pur = Tc.max(axis=0).sum() / Tc.sum()
    s += sol("Bloque 7", [
        ["1", f"Máximos por grupo: 40 (A), 38 (B), 45 (C); pureza = 123/150 = {f(pur, 3)}. El grupo 1 está dominado por A, el 2 por B y el 3 por C."],
        ["2", f"No: la pureza y el ARI ({f(ari_tabla(Tc), 3)}) se calculan sobre la partición; permutar números de grupo solo permuta las columnas de la tabla."],
        ["3", "No: sin clase real no hay tabla ni pureza. Se elige k con criterios internos (codo, silueta, Gap), estabilidad y propósito; la clase real solo valida cuando existe."]])
    build("apuntes/C39_Ejercicios_Cuaderno_Clustering.pdf", s, "Ejercicios para el cuaderno · Clase 39")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
