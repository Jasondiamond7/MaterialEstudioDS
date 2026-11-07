"""Clase 44 · Aplicaciones de IA y panorama actual del Deep Learning (Prof. Eloy Alvarado): guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c44_numeros.json (lo escribe figuras_c44.py); los de los ejemplos y ejercicios se calculan aquí.
Los números de las diapositivas (corrida del profesor) se citan como tales: «diapositivas». Los de la reproducción propia se llaman «nuestra reproducción»."""
import json
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, pista, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c44_numeros.json").read_text(encoding="utf-8"))
C44 = colors.HexColor("#4F46E5")
C44_BG = colors.HexColor("#EEF2FF")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
FUENTE = "Fuente: transcripción + diapositivas DipDS_2026_S44.pdf del profesor Eloy Alvarado + reproducción propia con scikit-learn (digitos.csv)"
DA, LG, RD, DP, BT, PA = N["datos"], N["log"], N["red"], N["desp"], N["boot"], N["paso"]
DS = N["diapos"]
LL, EM, CO, EN, DM, BO, CJ = N["logloss"], N["emb"], N["costo"], N["envases"], N["demanda"], N["bolsa"], N["cajas"]
AZUL = colors.HexColor("#2563EB")


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{x:,}".replace(",", " ")


def pc(x, d=2):
    return f"{x * 100:.{d}f} %"


AVAIL = (297 - 16 - 18) * mm - 24


def _alto(fl):
    return sum(x.wrap(CONTENT_W, 10000)[1] for x in fl)


def pagina(s, titulo, idea, conceptos, figuras, simple, formulas, salida, cuidado, fw=0.97, extra=None):
    """Una página por tema; las figuras se reducen solo lo necesario para que todo quepa."""
    def armar(k):
        top = [band(f"Clase 44 · {titulo}", FUENTE, C44), Spacer(1, 4), box([P(idea, "big")], bg=SOFT, border=C44, pad=5), Spacer(1, 3),
               P("Conceptos clave", "h3"), numbered(conceptos[:4], C44), Spacer(1, 1)]
        figs = []
        for fig, pie, *w in figuras:
            figs += [img(fig, width=CONTENT_W * (w[0] if w else fw) * k), P(pie, "small"), Spacer(1, 2)]
        left = [P("En palabras simples", "h3"), box([P(simple, "cell")], bg=C44_BG, border=C44, width=HALF, pad=5)]
        if formulas:
            left += [Spacer(1, 3), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
        right = [P("Con los números de la clase", "h3"), bullets(salida, "cell", C44),
                 Spacer(1, 3), box([P("<b>Cuidado.</b> " + " ".join(cuidado), "cell")], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
        bottom = [two_cols(left, right)]
        if extra:
            bottom += [Spacer(1, 3)] + extra
        return top, figs, bottom

    k = 1.0
    for _ in range(12):
        top, figs, bottom = armar(k)
        total = _alto(top) + _alto(figs) + _alto(bottom)
        if total <= AVAIL:
            break
        k -= 0.04
    print(f"  página «{titulo[:34]}»: figuras a {k:.2f} · alto {total / mm:.0f} de {AVAIL / mm:.0f} mm (top {_alto(top) / mm:.0f}, figs {_alto(figs) / mm:.0f}, bottom {_alto(bottom) / mm:.0f})")
    s += top + figs + bottom
    s.append(PageBreak())


# ======================================================================= GUÍA
def guia():
    s = []
    s.append(band("Clase 44 · Aplicaciones de IA y panorama actual del Deep Learning",
                  "Profesor Eloy Alvarado · conecta clasificación (Clases 20–21 y 31), evaluación (Clases 38 y 43), distancia y clustering (Clase 39) con imágenes, texto y modelos preentrenados", NAVY))
    s.append(Spacer(1, 6))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Antes de elegir un modelo hay que definir la tarea, los datos y cómo se comprobará el resultado.</b> Un mismo dato (una foto, un texto) sirve para tareas distintas: "
                    "<b>clasificar, pronosticar, recuperar o generar</b>. Todo modelo recibe <b>números</b>, así que hay que decidir cómo <b>representar</b> la entrada (una imagen son 64 intensidades; un texto, un vector). "
                    "Después se evalúa con criterios que reflejen el <b>uso real</b>: la exactitud no basta (hay que mirar las probabilidades con el log loss), un buen resultado no garantiza resistir un cambio de cámara, "
                    "y los errores tienen <b>costos</b> y <b>carga de revisión</b>. Una red neuronal es solo una herramienta más; la elección depende de la tarea, de los datos y de la evaluación.", "body")], bg=C44_BG, border=C44, pad=6))
    s.append(Spacer(1, 4))
    s.append(P("La clase en una imagen", "h2"))
    s.append(img("c44_esquema.png", width=CONTENT_W))
    s.append(Spacer(1, 2))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Regresión logística y probabilidades</b> (Clases 20–21 y 31): cada clase recibe una probabilidad",
                     "<b>Exactitud y matriz de confusión</b> (Clase 38): aciertos, falsos positivos y falsos negativos",
                     "<b>Bootstrap</b> (Clase 43): remuestrear casos de prueba con los modelos fijos",
                     "<b>Distancia entre datos</b> (Clase 39): la idea de «cercano» en un espacio de números"], "cell", C44)]
    right = [P("Ruta de la clase", "h3"),
             bullets(["<b>Definir la tarea</b> → <b>representar con números</b> → <b>probabilidades y log loss</b> → <b>bootstrap y prueba de estrés</b>",
                      "Pausa (~15 min) y luego: <b>texto, embeddings, búsqueda, generación</b> → <b>modelos preentrenados</b> → <b>costos, revisión y diseño de la prueba</b>",
                      f"Bloque práctico: <b>dígitos escritos a mano</b> ({DA['n']} imágenes de 8 × 8; {DA['n_tr']} entrenamiento, {DA['n_va']} validación, {DA['n_te']} prueba)",
                      "La red neuronal se construye recién el 20 de octubre (Clase 45): hoy se estudian sus entradas y salidas"], "cell", C44)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Cómo leer los números de este material</b>", "cellb"),
                  P(f"Las <b>diapositivas</b> muestran la corrida del profesor. Aquí se <b>reprodujo</b> el bloque de dígitos con la misma partición (digitos.csv) y scikit-learn. "
                    f"La regresión logística coincide con las diapositivas: {LG['k']} de 360 aciertos, log loss {f(LG['ll'], 4)}, y con el desplazamiento {pc(DP['log']['acc'])} y {f(DP['log']['ll'], 4)}. "
                    f"La red neuronal sale distinta (otra inicialización, o quizá otro programa): {RD['k']} de 360 y log loss {f(RD['ll'], 4)} (diapositivas: 354 y 0.0657). "
                    "La conclusión cualitativa es la misma; cuando una cifra es de la corrida del profesor, se dice «diapositivas».", "cell")], bg=AMBER_BG, border=AMBER, pad=5))
    s.append(PageBreak())

    # 2 ------------------------------------------------------------------ tarea
    pagina(s, "Primero la tarea: IA, aprendizaje automático y profundo",
           "<b>Se parte del objetivo, no de los datos.</b> Hay que poder completar la frase: «entra X, el sistema entrega Y, y lo considero útil si Z». Si no se puede, falta precisar la tarea.",
           [("Los tres círculos", "IA es el campo más amplio (incluye reglas escritas a mano). El aprendizaje automático aprende patrones de los datos (logística, Naive Bayes). El aprendizaje profundo aprende <b>varias transformaciones sucesivas</b>. Todo deep learning es ML y es IA; no al revés."),
            ("Cuatro tareas, cuatro salidas", "<b>Clasificar</b> (una categoría), <b>pronosticar</b> (una cantidad futura), <b>recuperar</b> (fuentes relevantes) y <b>generar</b> (texto o imagen). Cada una se comprueba distinto."),
            ("Un sistema es una cadena de tareas", "Atender «me cobraron dos veces» es <b>clasificar</b> (categoría y prioridad), <b>consultar</b> (registros y procedimiento) y <b>redactar</b> (borrador). Se evalúa cada paso y el resultado completo."),
            ("Fluidez no es fidelidad", "Un texto puede leerse muy bien y afirmar algo sin respaldo. La fluidez dice si suena natural; la fidelidad, si conserva los hechos de las fuentes.")],
           [("c44_anidado.png", "Relación entre los campos (diapositiva 4): cada círculo está dentro del anterior.", 0.46),
            ("c44_cuatro_preguntas.png", "Las cuatro preguntas para estudiar una aplicación (diapositiva 3).", 0.9)],
           "Piensa en un taller. <b>Primero decides qué quieres construir</b> (la tarea), <b>luego buscas los materiales</b> (datos) y <b>al final defines cómo revisarás la obra</b> (evaluación). Comprar un martillo gigante sin saber qué construir es empezar por el modelo.",
           ["Aprendizaje profundo, dentro de aprendizaje automático, dentro de IA", "Tarea → datos → modelo → salida → comprobación → uso"],
           ["Ejemplo de clasificación de la clase: ¿este envase presenta daño? (se compara con la etiqueta correcta)",
            f"Ejemplo de pronóstico: se venderán {DM['pred']} unidades mañana; se vendieron {DM['real']} → error absoluto |{DM['pred']} − {DM['real']}| = {DM['err_modelo']}",
            "Pregunta de alumna: ¿red neuronal = deep learning? No siempre; una red con una neurona y una capa se parece a una regresión. «Profundo» pide muchas capas (se verá el 20 de octubre)"],
           ["Tener datos disponibles no es un objetivo. Empezar por «tengo estos datos, ¿qué hago?» lleva a resolver problemas que no importan."])

    # 3 ------------------------------------------------------------------ representación
    pagina(s, "Representar los datos: una imagen son números",
           "<b>Un modelo solo recibe números.</b> Una imagen de 8 × 8 píxeles se convierte en 64 intensidades (0 a 16); se leen por filas y se dividen por 16 para quedar entre 0 y 1. La etiqueta se guarda aparte.",
           [("Píxeles e intensidad", "Cada celda (píxel) tiene un número que describe su tono: 0 es blanco y 16 es negro. Con color habría tres números por celda (rojo, verde y azul)."),
            ("De matriz a vector", "La matriz de 8 × 8 se «estira» fila por fila: primera fila, luego la segunda… en total 64 números en un orden fijo. Otra lectura (por columnas) también sería válida si se usa siempre la misma."),
            ("Hay muchas representaciones", f"Un correo «{BO['texto']}» puede contarse por palabras ({', '.join(BO['palabras'])} → {BO['cuentas']}) o solo por su largo ({BO['largo']}). Contar conserva frecuencias pero pierde el orden: «el hombre mordió al perro» y «el perro mordió al hombre» tendrían los mismos conteos."),
            ("La etiqueta", "En entrenamiento se conoce el dígito correcto de cada imagen; en prueba se oculta para ver si el modelo acierta con imágenes que no usó para ajustarse.")],
           [("c44_digitos.png", f"Un ejemplo de cada dígito de la colección ({DA['n']} imágenes de 8 × 8; diapositiva 10).", 0.62),
            ("c44_imagen_numeros.png", "La imagen de la diapositiva 11 (un 8): se lee por filas y se escala. Cuanto más cerca de 16, más oscuro.", 0.97)],
           "Es como describir un cuadro por teléfono: <b>cuadrícula de 8 por 8, y te dicto el tono de cada cuadradito de izquierda a derecha y de arriba hacia abajo</b>. Quien escucha puede reconstruir el cuadro porque ambos usan el mismo orden.",
           ["intensidad ÷ 16 = valor entre 0 y 1", "8 × 8 = 64 píxeles = 64 columnas de entrada"],
           [f"La imagen de ejemplo tiene {N['ejemplo']['n_ceros']} celdas en 0 y {N['ejemplo']['n_16']} en 16; su primera fila es {N['ejemplo']['fila0']}",
            "9 ÷ 16 = 0.5625 · 14 ÷ 16 = 0.875 · 8 ÷ 16 = 0.5",
            "Todas las imágenes tienen el mismo tamaño; con más píxeles o tamaños distintos, la representación ya no es trivial"],
           ["Que los números sirvan no está garantizado: una buena representación depende de la tarea. Cambiar la representación puede cambiar el resultado del modelo."])

    # 4 ------------------------------------------------------------------ salida / log loss
    pagina(s, "La salida: probabilidades y pérdida logarítmica (log loss)",
           "<b>El clasificador entrega 10 probabilidades (una por dígito) que suman 1; se predice la clase con más probabilidad.</b> La clase elegida y la probabilidad de la clase real dan información distinta: el log loss mira esta última.",
           [("Probabilidad ≠ acierto", "Si el real es 8 y el modelo da 0.60 al 3 y 0.25 al 8, predice 3 y se equivoca, aunque haya dado algo de probabilidad al 8."),
            ("Pérdida = −log(p real)", "Se toma la probabilidad del dígito <b>real</b>, se calcula −ln(p) y se promedia sobre todas las imágenes. Menor es mejor. Una predicción confiada y equivocada cuesta muchísimo."),
            ("Dos métricas, dos preguntas", "La <b>exactitud</b> cuenta aciertos (más alto es mejor). El <b>log loss</b> castiga probabilidades mal puestas (más bajo es mejor). Dos modelos pueden acertar casi lo mismo y tener log loss muy distinto."),
            ("Umbral y regla de decisión", "«Aceptar sin revisión si p ≥ 0.90» es una regla humana, no del modelo. Una probabilidad alta no garantiza el acierto de esa imagen.")],
           [("c44_logloss.png", "Izquierda: costo −log(p) para tres valores de la diapositiva 15. Derecha: lectura rápida.", 0.97),
            ("c44_probabilidades.png", "Probabilidades de ambos modelos en nuestra reproducción para un acierto y para el error más confiado de la red (la zona amarilla marca la clase real).", 0.97)],
           "Imagina que un <b>pronosticador apuesta fichas</b> al dígito correcto. Si apostó casi todo a la respuesta verdadera, pierde poco; si apostó casi todo a una respuesta equivocada, <b>pierde muchísimo</b>. El log loss mide cuántas fichas pierde en promedio.",
           ["log loss = promedio de −ln(p del dígito real)", "exactitud = aciertos ÷ imágenes"],
           [f"−ln(0.90) = {f(LL['vals'][0], 3)} · −ln(0.60) = {f(LL['vals'][1], 3)} · −ln(0.01) = {f(LL['vals'][2], 3)} (diapositiva 15)",
            f"Imagen de la diapositiva 19 (real = 3): logística asigna {f(PA['p_log'], 6)} y pérdida {f(PA['l_log'], 4)}; nuestra red, {f(PA['p_red'], 6)} y {f(PA['l_red'], 6)} (diapositivas: 0.999775 y 0.000225)",
            "Hipótesis de la diapositiva 12: 0.60 al 3, 0.25 al 8 y 0.15 repartido entre las otras 8 clases"],
           ["−ln se calcula con el logaritmo <b>natural</b>. Si usas log base 10 los valores cambian (pero el orden entre modelos no)."])

    # 5 ------------------------------------------------------------------ datos y modelos
    pagina(s, "Dos recetas para los dígitos: regresión logística y red neuronal",
           f"<b>Los datos se separan en entrenamiento ({DA['n_tr']}), validación ({DA['n_va']}) y prueba ({DA['n_te']}).</b> Con entrenamiento se ajustan los candidatos, con validación se elige, se reajusta con las {DA['n_dev']} imágenes de desarrollo, y la prueba se mira al final, una sola vez.",
           [("Referencia: regresión logística", f"Usa los 64 píxeles para calcular 10 probabilidades. Es barata, fácil de implementar e interpretable: es el modelo a vencer. Tiene {DA['n_params_log']} parámetros."),
            ("Candidato: red neuronal", f"Aprende transformaciones de los mismos 64 píxeles. Aquí se eligió por validación entre dos tamaños (16 o 32 unidades ocultas): ganó {DA['elegida']} (log loss de validación {f(DA['cand'][f'({DA['elegida']},)'], 4)}); tiene {mil(DA['n_params_elegida'])} parámetros."),
            ("Mismas imágenes, mismos casos", "Ambos modelos se evalúan sobre las mismas 360 imágenes, así se pueden comparar. El experimento compara <b>estas dos recetas</b>: no dice qué familia gana en todos los problemas."),
            ("Lo que sí y no se puede concluir", "La separación fue por imagen y no hay identificador de escritor: no se demuestra desempeño en personas nuevas.")],
           [("c44_resultados.png", "Nuestra reproducción en las 360 imágenes de prueba: misma cantidad de errores, distinto log loss.", 0.97),
            ("c44_errores_red.png", f"Los errores de la red, ordenados por confianza (etiqueta real, predicha y probabilidad de la clase predicha). En rojo, los que la red declara con p ≥ 0.90.", 0.97)],
           "Es como dos alumnos con la <b>misma nota</b> en una prueba de 360 preguntas. Uno responde con dudas y el otro con seguridad: cuando se equivoca el seguro, el error es más grave. La nota (exactitud) no los distingue; <b>la confianza (log loss) sí</b>.",
           None,
           [f"Nuestra reproducción: logística {LG['k']} de 360 ({pc(LG['acc'])}), log loss {f(LG['ll'], 4)} · red {RD['k']} de 360 ({pc(RD['acc'])}), log loss {f(RD['ll'], 4)}",
            f"Diapositivas: logística 353 (98.06 %, 0.1353) · red 354 (98.33 %, 0.0657): un acierto más = 100/360 ≈ 0.28 puntos",
            f"De nuestra red, {RD['n_conf90']} de {RD['errores']} errores tienen p ≥ 0.90 (diapositiva 17: la red del profesor erró 6, con p entre 0.76 y 0.94)",
            f"En la red, los {RD['errores']} errores concentran {f(RD['perdida_errores'], 1)} de {f(RD['perdida_total'], 1)} de pérdida total ({f(RD['perdida_errores'] / RD['perdida_total'] * 100, 0)} %)"],
           ["Un valor distinto en la red es normal: depende de la inicialización y de las épocas. La lección de la clase no cambia: con exactitud casi igual, el log loss puede diferir bastante."])

    # 6 ------------------------------------------------------------------ bootstrap + estrés
    pagina(s, "¿La ventaja es real? Bootstrap y prueba de estrés",
           "<b>Una diferencia entre dos modelos medida en 360 imágenes podría cambiar con otras 360.</b> El bootstrap reutiliza las pérdidas de esas imágenes (modelos fijos) para estimar cuánto varía la diferencia. Después, la prueba de estrés pregunta si el resultado resiste un cambio en la entrada.",
           [("Diferencia de pérdidas", "Se resta pérdida de la logística menos pérdida de la red, imagen por imagen. Si el promedio es positivo, la red pierde menos."),
            ("Intervalo bootstrap", "Se remuestrean las 360 pérdidas con reposición B = 2 000 veces; el intervalo del 95 % usa los percentiles 2.5 y 97.5. Si <b>0 queda fuera</b>, hay evidencia de menor pérdida de la red (bajo independencia por imagen)."),
            ("Desplazar una columna", "Se corre cada imagen una columna a la derecha (la primera se rellena con 0 y la última original se descarta). Los mismos modelos, sin reajustar."),
            ("Un buen resultado no garantiza robustez", "Para una persona la imagen es «la misma»; para el modelo, los números cambiaron mucho. Otra cámara, otro trazo o otra posición pueden alterar la entrada.")],
           [("c44_bootstrap.png", f"Izquierda: distribución bootstrap de la diferencia de pérdidas medias (nuestra reproducción). Derecha: las 360 diferencias por imagen; hay {BT['n_log_gana']} imágenes donde la logística pierde menos.", 0.97),
            ("c44_estres.png", "Exactitud y log loss con las imágenes originales y desplazadas una columna a la derecha (nuestra reproducción; diapositivas 21 y 22).", 0.97)],
           "El bootstrap es como <b>preguntarte qué habría pasado si el examen hubiera tenido otras preguntas parecidas</b>: vuelves a sacar «preguntas» de las mismas 360 (con reposición) muchas veces y miras cuánto cambia la ventaja.",
           ["diferencia = pérdida media (logística) − pérdida media (red)", "IC 95 % = percentiles 2.5 y 97.5 de las B diferencias"],
           [f"Nuestra reproducción: diferencia {f(BT['dif'], 4)}, IC [{f(BT['ic'][0], 4)}; {f(BT['ic'][1], 4)}] (excluye 0). Diapositivas: 0.0696, IC [0.0478; 0.0901]",
            f"Desplazadas, logística: {pc(DP['log']['acc'])} y log loss {f(DP['log']['ll'], 4)} (diapositivas: 39.72 % y 2.3826, igual)",
            f"Desplazadas, red: {pc(DP['red']['acc'])} y {f(DP['red']['ll'], 4)} (diapositivas: 45.00 % y 3.1314). La red acierta más que la logística pero su log loss es peor",
            "El azar entre 10 clases sería 10 % de exactitud"],
           ["El intervalo no mide la utilidad práctica de la mejora ni cubre un cambio de cámara. El desplazamiento es una prueba de estrés artificial, no una cámara real."])

    # 7 ------------------------------------------------------------------ texto
    pagina(s, "Después de la pausa: texto, embeddings y búsqueda por significado",
           "<b>Un embedding representa un objeto (un texto, una imagen) con un vector para poder compararlo con otros.</b> Si dos textos se parecen, sus vectores quedan cerca. Así se busca «por significado» y no solo por palabras exactas.",
           [("Vector y embedding", "Un vector es una lista ordenada de números. Las coordenadas las aprende un modelo con ejemplos; no son probabilidades ni etiquetas de temas."),
            ("Cercanía", f"Con números inventados: Q «cobro duplicado» = {EM['q']}, A «reembolso» = {EM['a']}, B «envíos» = {EM['b']}. A está a {f(EM['d_qa'], 3)} de Q y B a {f(EM['d_qb'], 3)}: A es más cercano."),
            ("Búsqueda por palabras vs por significado", "Por palabras compara términos (falla si la fuente dice «reembolso» y tú escribes «cobro duplicado»). Por significado compara vectores, pero la cercanía puede traer un texto relacionado que no responde. Hay que comprobar que respondan, correspondan al servicio y estén vigentes."),
            ("CLIP y recomendación", "CLIP aprende con pares imagen–texto a producir vectores comparables (da una similitud, no una probabilidad de daño). En recomendación, un 1 es «hay clic» y un 0 es «no hay clic registrado», no rechazo.")],
           [("c44_embedding.png", "Los tres textos de la diapositiva 27 como puntos y vectores; distancia y similitud coseno calculadas aquí.", 0.97),
            ("c44_difusion.png", "Idea de difusión (diapositiva 32, ilustración propia): del ruido a una imagen guiada por el texto, en muchos pasos.", 0.62)],
           "Un embedding es como <b>ubicar ciudades en un mapa</b>: Santiago y Valparaíso quedan cerca; Santiago y Punta Arenas, lejos. Con textos, el «mapa» se aprende para que los de significado parecido queden juntos. Una <b>base de datos vectorial</b> solo guarda los puntos cercanos cerca unos de otros para encontrarlos rápido.",
           ["distancia(Q, A) = √((q₁ − a₁)² + (q₂ − a₂)²)", "coseno(Q, A) = (Q · A) / (‖Q‖ ‖A‖)"],
           [f"Q–A: distancia {f(EM['d_qa'], 3)}, coseno {f(EM['cos_qa'], 3)} · Q–B: distancia {f(EM['d_qb'], 3)}, coseno {f(EM['cos_qb'], 3)}",
            "Generar texto: un modelo de lenguaje aprende regularidades en el orden de las palabras y añade unidades sucesivamente usando el contexto",
            "Generar imágenes (difusión): parte de ruido visual y lo transforma gradualmente guiado por la descripción; el realismo no acredita un hecho",
            "Autosupervisado: ocultar «crédito» en «el banco aprobó el crédito» y compararlo con la palabra original; la respuesta sale de los datos"],
           ["Producir una continuación plausible no garantiza que sea verdadera. Y la diapositiva dice A = [0.35, 0.70]; en la clase se mencionó 0.3 y 0.7, una aproximación oral."])

    # 8 ------------------------------------------------------------------ preentrenados y verificación
    pagina(s, "Modelos preentrenados, panorama actual y verificación de lo generado",
           "<b>Un modelo preentrenado ya aprendió con otros datos; su utilidad local se demuestra con datos de la aplicación.</b> Y una respuesta generada se verifica afirmación por afirmación contra la fuente.",
           [("Reutilizar un aprendizaje previo", "Permite aprovechar otros datos y ajustar menos desde cero. Hay que comprobar que sea pertinente (un modelo de fauna de otro continente no sirve igual en Chile) y que haya datos locales para adaptar y evaluar."),
            ("Tres formas de adaptar", "<b>Instrucciones y contexto</b> (cambia la entrada, sin reentrenar), <b>reutilizar representaciones</b> (vectores + un clasificador propio) y <b>ajuste fino</b> (se modifica el modelo con datos de la tarea)."),
            ("Panorama y benchmarks", "Un <b>benchmark</b> es una evaluación con tareas, datos y reglas definidos; su resultado no se traslada solo a otra población. Hay que preguntar qué tarea se evaluó y si nuestros datos son comparables."),
            ("Una aplicación es más que el modelo", "Datos y contexto, modelo, verificación, decisión y seguimiento: puede fallar cualquiera. Ej.: 12 cajas de 24 unidades son 12 × 24 = 288 unidades, no 12.")],
           [("c44_adaptar.png", "Tres formas de adaptar (diapositiva 35), de menos a más cambio en el modelo.", 0.97)],
           "Un preentrenado es como <b>contratar a alguien con experiencia en otro rubro</b>: ya sabe mucho, pero hay que comprobar que lo que aprendió sirva en tu trabajo, y darle algo de práctica local. Partir desde cero tiene sentido solo si nadie ha tenido un problema parecido.",
           None,
           ["Fuente: «hasta cinco días hábiles de revisión». Borrador: «aprobada y llegará mañana»: ambas afirmaciones <b>sin respaldo</b> (la fuente no informa el resultado ni una fecha de abono)",
            "AlphaFold 3: recibe información de los componentes del complejo y propone su estructura; se comprueba contra estructuras determinadas experimentalmente. Orienta la investigación, pero no demuestra que un compuesto sea un tratamiento eficaz",
            "Respuesta corregida de la diapositiva 40: «El procedimiento indica hasta cinco días hábiles de revisión. Con esta información no puedo confirmar una aprobación ni una fecha de abono.»"],
           ["Si una afirmación no tiene respaldo en la fuente, se corrige o se declara información faltante. Basta que un paso falle para que las decisiones automatizadas salgan mal."])

    # 9 ------------------------------------------------------------------ costos y diseño de la prueba
    pagina(s, "Costos, carga de revisión y una prueba que represente el uso",
           "<b>Los errores no cuestan lo mismo y la revisión humana tiene capacidad limitada.</b> Además, la prueba debe imitar el uso previsto: las 5 fotos de un envase no son 5 envases independientes.",
           [("Costo de los errores", f"De {CO['n']} piezas, {CO['defectos']} tienen defectos. El sistema deja pasar {CO['fn']} y manda {CO['fp']} sanas a revisión. Con costos de {CO['c_fn']} y {CO['c_fp']} unidades: {CO['fn']} × {CO['c_fn']} + {CO['fp']} × {CO['c_fp']} = {CO['costo']} unidades."),
            ("Carga de revisión", f"Se detectaron {CO['detectados']} defectos. Si se revisan todas las alertas son {CO['detectados']} + {CO['fp']} = {CO['alertas']} revisiones; con capacidad de {CO['capacidad']} no alcanza (faltan {CO['faltan']})."),
            ("Separar por la unidad real", f"{mil(EN['fotos'])} fotos de {mil(EN['envases'])} envases (5 vistas cada uno). Todas las vistas de un envase van al mismo conjunto, y el bootstrap remuestrea envases completos."),
            ("Qué falta para el próximo mes", "Una cámara nueva en otra línea: hay que reunir datos de esas condiciones, reservar envases que no se usen para adaptar ni elegir, comparar una referencia simple con un candidato y medir defectos omitidos, falsas alarmas y revisiones.")],
           [("c44_costo.png", "Diapositiva 41: costo total de los errores y carga de revisión frente a la capacidad.", 0.97),
            ("c44_split_envases.png", "Caso integrador (diapositivas 46 y 47): separar por foto filtra información; separar por envase mantiene juntas las vistas (ilustración con 10 envases).", 0.97)],
           "Es como <b>corregir pruebas de varias hojas</b>: si pones la hoja 1 de un alumno en el grupo de estudio y la hoja 2 en el examen, el examen ya vio al alumno. Hay que mantener todas las hojas de la misma persona (o del mismo envase) en el mismo grupo.",
           ["costo = Σ (cantidad de error × costo unitario)", "revisiones = defectos detectados + falsas alarmas"],
           [f"Omitir un defecto cuesta {CO['c_fn']} y una falsa alarma {CO['c_fp']}: omitir cuesta más",
            f"Exactitud del sistema: {pc(CO['acc'], 1)}; «decir siempre sano» lograría {pc(CO['siempre_sano_acc'], 1)} pero omitiría los {CO['defectos']} defectos (costo {CO['siempre_sano_costo']})",
            "Otras restricciones: tiempo (latencia), costo, datos, mantenimiento; si llega 1 pieza por segundo y se tardan 5 s en procesar cada una, se acumula trabajo",
            "Primera comparación: una referencia simple (regresión, árbol, regla de historia reciente). Una alternativa simple puede resolver mejor el problema"],
           ["Cambiar el procedimiento después de mirar la prueba exige otra evaluación independiente. La conclusión se limita a las condiciones observadas."])

    # 10 ------------------------------------------------------------------ repaso
    s.append(band("Repaso, alertas del material y criterio de dominio", "Preguntas para responder sin apuntes y lo que debes poder hacer antes de la Clase 45 (redes neuronales)", NAVY))
    s.append(Spacer(1, 6))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Qué conviene saber al leer las diapositivas y la transcripción</b>", "cellb"),
                  bullets(["<b>Las cifras de la red son de la corrida del profesor.</b> La logística se reproduce exacta (353 de 360, log loss 0.1353, desplazada 39.72 % y 2.3826); la red no (ver la página inicial).",
                           "<b>La transcripción tenía errores de reconocimiento de voz</b> (red neuronal escrita como «renda oral», log loss como «LOC-LOS», 0.1353 como «0.3353»…). Se corrigieron en la versión limpia; en el original sigue el audio literal.",
                           "<b>«Deep learning» y «red neuronal» no son sinónimos:</b> una red muy pequeña se parece a una regresión. Profundo = muchas capas (Clase 45).",
                           "<b>Las diapositivas piden no sobreinterpretar:</b> los valores 0.20/0.80, 0.35/0.70 y 0.80/0.20 son inventados para ilustrar; las similitudes de CLIP (0.82, 0.39, 0.06) también.",
                           "<b>Ejemplos hipotéticos:</b> inspección de envases, demanda, cursos, cobros y 1 000 piezas son escenarios ilustrativos, no datos reales."], "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 6))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    pregs = [("¿Cuál es el orden correcto: datos, tarea o evaluación?", "Primero la tarea (qué se quiere hacer), luego los datos y cómo se evaluará."),
             ("Un clasificador da 0.60 al 3 y 0.25 al 8; el real es 8. ¿Acierta? ¿Cuánto vale la pérdida de esa imagen?", f"No (predice 3). Pérdida = −ln(0.25) = {f(-log(0.25), 3)}."),
             ("Dos modelos aciertan casi lo mismo, pero uno tiene la mitad de log loss. ¿Qué significa?", "Que sus probabilidades están mejor puestas (más seguras cuando aciertan, menos extremas al errar)."),
             ("El IC bootstrap de la diferencia de pérdidas excluye 0. ¿Qué concluyes? ¿Qué no concluyes?", "Evidencia de menor pérdida de un modelo en estos casos; no que sea mejor en otra cámara ni que la mejora sea útil."),
             ("¿Qué muestra desplazar las imágenes una columna?", "Que un buen resultado original no garantiza resistir cambios de entrada: ambos modelos caen mucho."),
             ("¿Por qué «cobro duplicado» y «reembolso» pueden estar cerca en un embedding?", "Porque el modelo aprendió a acercar textos de significado relacionado, aunque no compartan palabras."),
             ("¿Qué mide «4 × 50 + 30 × 1»?", f"El costo de los errores: 4 defectos omitidos a 50 y 30 falsas alarmas a 1 = {CO['costo']} unidades."),
             ("6 000 fotos de 1 200 envases: ¿cómo separas entrenamiento, validación y prueba?", "Por envase: las 5 vistas de cada envase en un solo conjunto.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(pregs, 1)], [7 * mm, 78 * mm, CONTENT_W - 85 * mm]))
    s.append(Spacer(1, 6))
    s.append(box([P("<b>Criterio de dominio para pasar a la Clase 45</b>", "cellb"), Spacer(1, 3),
                  bullets(["Defino una aplicación con tarea, entrada, salida y forma de comprobar",
                           "Explico cómo una imagen 8 × 8 se convierte en 64 números entre 0 y 1",
                           "Calculo −log(p) y el log loss promedio a mano, y explico por qué complementa la exactitud",
                           "Interpreto un IC bootstrap de la diferencia de pérdidas y los límites de la conclusión",
                           "Calculo el costo de errores y la carga de revisión, y diseño una separación que no filtre información"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    build("C44_Guia_AplicacionesIA_DeepLearning.pdf", s, "Clase 44 · Aplicaciones de IA y Deep Learning")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#4F46E5">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 44 · Aplicaciones de IA y panorama del Deep Learning", "Para leer antes de los ejercicios · lectura de ~35 min · pensado para quien recién parte con deep learning", C44))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 7 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Primero la tarea", "Clasificar, pronosticar, recuperar o generar; luego datos, modelo y comprobación."),
                            ("Todo entra como números", "Una imagen 8 × 8 son 64 intensidades (0–16), escaladas a 0–1."),
                            ("El modelo entrega probabilidades", "Se predice la clase más probable; la probabilidad de la clase real da el log loss."),
                            ("Exactitud y log loss no dicen lo mismo", "Mismos aciertos, distinta calidad de probabilidades."),
                            ("Un resultado no es una garantía", "Bootstrap para la incertidumbre; desplazar la imagen para ver robustez."),
                            ("Embeddings: significado como cercanía", "Textos o imágenes como vectores; distancia y coseno."),
                            ("Costos, revisión y diseño de la prueba", "Errores con precio, capacidad limitada y separación por la unidad real.")], C44)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación en palabras simples, un <b>ejemplo resuelto con números distintos</b> a los de los ejercicios y un resumen «para recordar». Ejercicios en <font name='Mono'>C44_Ejercicios_Cuaderno_AplicacionesIA_DeepLearning.pdf</font> y notebook <font name='Mono'>C44_AplicacionesIA_DeepLearning.ipynb</font>.", "small"))

    # 1 tarea
    s.append(h("1", "Antes de elegir un modelo: la tarea, los datos y la comprobación"))
    s += para("Imagina que te piden «automatizar algo». Lo primero no es escoger un algoritmo, sino responder tres preguntas: <b>¿qué entra?</b>, <b>¿qué debe salir?</b> y <b>¿cómo sabré que la salida sirve?</b> "
              "El profesor insistió en que el objetivo va <b>antes</b> que los datos: si se parte de «tengo estos datos, ¿qué hago?», se termina adaptando el método a lo que hay y no a lo que importa. "
              "Las tareas se agrupan en cuatro: <b>clasificar</b> (una categoría), <b>pronosticar</b> (una cantidad futura), <b>recuperar</b> (encontrar y ordenar fuentes) y <b>generar</b> (texto o imagen). "
              "Una aplicación real mezcla varias; a esa cadena de tareas se le llama <b>sistema</b>.")
    s.append(ejemplo("«Me cobraron dos veces y necesito el comprobante»", [
        pasos([["Paso", "Tarea", "Salida posible", "Cómo se comprueba"],
               ["1", "Clasificar", "Categoría «cobros» y prioridad", "Compararla con la categoría correcta"],
               ["2", "Consultar y recuperar", "Registros del caso y procedimiento", "Que sean pertinentes y estén vigentes"],
               ["3", "Redactar", "Borrador de respuesta", "Cada afirmación tiene respaldo; lo faltante se declara"]],
              [11 * mm, 34 * mm, 55 * mm, CONTENT_W - 100 * mm - 18]),
        P("Lectura: una sola consulta se descompone en tres tareas distintas, cada una con su evaluación. <b>Error común</b>: evaluar solo que «el texto se lea bien» (fluidez) y no que sea verdadero (fidelidad).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(P("Dentro de este mapa aparecen tres términos que suelen confundirse: la <b>inteligencia artificial</b> (todo sistema que reconoce, predice o genera, incluso con reglas escritas a mano), el <b>aprendizaje automático</b> (aprende patrones de datos: logística, Naive Bayes) y el <b>aprendizaje profundo</b> (aprende varias transformaciones sucesivas). Cada uno está dentro del anterior.", "cell"))
    s.append(Spacer(1, 2))
    s.append(img("c44_anidado.png", width=CONTENT_W * 0.3))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Tarea → datos → modelo → salida → comprobación → uso",
                       "Cuatro tareas: clasificar, pronosticar, recuperar, generar",
                       "Fluidez ≠ fidelidad: un texto bien escrito puede inventar una aprobación"]))

    # 2 representar
    s.append(h("2", "Representar los datos: de una imagen a 64 números"))
    s += para("Un modelo matemático no «ve» una foto: recibe números. Una imagen en blanco y negro es una cuadrícula de celdas (<b>píxeles</b>) y cada celda tiene una <b>intensidad</b>. En la colección de dígitos hay "
              f"{DA['n']} imágenes de 8 × 8 píxeles, con intensidades de 0 (blanco) a 16 (negro). La matriz se lee <b>fila por fila en un orden fijo</b> y se obtiene un vector de 64 números; se divide por 16 para que queden entre 0 y 1. "
              "La etiqueta (el dígito correcto) se guarda <b>aparte</b>: no es parte de la entrada.")
    fila = N["ejemplo"]["fila0"]
    s.append(ejemplo("la imagen del 8 de la diapositiva 11", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Primera fila de la matriz", "leer de izquierda a derecha", str(fila)],
               ["Tamaño de la entrada", "8 filas × 8 columnas", "64 números"],
               ["Escalar 9, 14 y 8", "dividir por 16", f"{f(9 / 16, 4)}; {f(14 / 16, 3)}; {f(8 / 16, 1)}"],
               ["Etiqueta", "se guarda separada", "8"]],
              [48 * mm, 48 * mm, CONTENT_W - 96 * mm - 18]),
        P("Lectura: el modelo recibe 64 columnas entre 0 y 1; la etiqueta solo sirve para aprender y para evaluar. <b>Error común</b>: pensar que hay una única forma «correcta» de pasar objetos a números: hay infinitas, y hay que diseñarla según la tarea.", "cell")]))
    s.append(Spacer(1, 3))
    s += para(f"Lo mismo pasa con un texto. Si se cuentan las palabras «{', '.join(BO['palabras'])}», el correo «{BO['texto']}» queda como {BO['cuentas']}: se conservan las frecuencias pero se <b>pierde el orden</b> (por eso «el hombre mordió al perro» y «el perro mordió al hombre» darían los mismos números). "
              "La representación importa porque define qué diferencias puede usar el modelo.")
    s.append(img("c44_imagen_numeros.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Imagen 8 × 8 = 64 intensidades (0–16); ÷ 16 → 0–1", "La etiqueta va aparte de la entrada", "Contar palabras conserva frecuencias y pierde el orden"]))

    # 3 probabilidades y log loss
    s.append(h("3", "La salida: probabilidades, decisión y pérdida logarítmica"))
    s += para("Para cada imagen el clasificador entrega <b>diez probabilidades</b> (una por dígito) que suman 1. La regla habitual es predecir la clase con la mayor. Pero <b>lo que se elige y la probabilidad de la clase real son dos cosas distintas</b>: "
              "si el real es 8 y el modelo da 0.25 al 8 y 0.60 al 3, predice 3 y se equivoca, aunque haya dado algo de probabilidad al 8. "
              "La <b>pérdida logarítmica</b> (log loss) mira justamente esa probabilidad de la clase real: la pérdida de una imagen es −ln(p), y el log loss es el promedio. Menor es mejor.")
    ps = [0.95, 0.70, 0.30, 0.02]
    ls = [-log(p) for p in ps]
    s.append(ejemplo("cuatro imágenes con la probabilidad que el modelo dio al dígito real", [
        pasos([["Imagen", "p del dígito real", "Pérdida −ln(p)", "Lectura"],
               ["1", "0.95", f(ls[0], 4), "muy buena"],
               ["2", "0.70", f(ls[1], 4), "buena"],
               ["3", "0.30", f(ls[2], 4), "mala (probablemente error)"],
               ["4", "0.02", f(ls[3], 4), "muy mala: error muy confiado"],
               ["Promedio", "", f(np.mean(ls), 4), "log loss de las 4 imágenes"]],
              [26 * mm, 40 * mm, 38 * mm, CONTENT_W - 104 * mm - 18]),
        P(f"Lectura: una sola imagen (la 4) aporta {f(ls[3] / sum(ls) * 100, 0)} % de la pérdida total. Por eso el log loss castiga los errores confiados. <b>Error común</b>: usar la probabilidad de la clase predicha en vez de la de la clase real (si el real es 8 y se predijo 3 con 0.90, se usa la probabilidad del 8, por ejemplo 0.01).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c44_logloss.png", width=CONTENT_W * 0.78))
    s.append(Spacer(1, 2))
    s += para("Un <b>umbral</b> es un valor de corte, por ejemplo «aceptar sin revisión si la probabilidad supera 0.90». Es una regla de decisión que se fija con validación, no algo que venga del modelo, y no garantiza el acierto: en los errores de la red hubo probabilidades de 0.76 a 0.94 (diapositiva 17).")
    s.append(recuerda(["La clase elegida ≠ la probabilidad de la clase real", "log loss = promedio de −ln(p real); más bajo es mejor",
                       "Una probabilidad alta no garantiza el acierto de esa imagen"]))

    # 4 evaluar
    s.append(h("4", "Comparar dos modelos: exactitud, log loss y bootstrap"))
    s += para(f"Se compara una regresión logística (referencia barata e interpretable) con una red neuronal. Cada una se ajusta con entrenamiento, se elige por validación y se evalúa <b>una vez</b> en las {DA['n_te']} imágenes de prueba. "
              f"En la corrida del profesor, la logística acierta 353 y la red 354: <b>un acierto más equivale a 100/360 ≈ 0.28 puntos porcentuales</b>. El log loss, en cambio, difiere bastante (0.1353 vs 0.0657). "
              f"En nuestra reproducción ambos aciertan {LG['k']} de 360 y el log loss es {f(LG['ll'], 4)} (logística) frente a {f(RD['ll'], 4)} (red).")
    a1, a2, n_ = 0.20, 0.35, 360
    s.append(ejemplo("de aciertos a exactitud y por qué un acierto pesa tan poco", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Exactitud de 353 aciertos", "353 ÷ 360", pc(353 / 360)],
               ["Exactitud de 354 aciertos", "354 ÷ 360", pc(354 / 360)],
               ["Diferencia", "(354 − 353) ÷ 360", f"{f(100 / 360, 2)} puntos"],
               ["Log loss logística − red (diapositivas)", "0.1353 − 0.0657", "0.0696"]],
              [58 * mm, 50 * mm, CONTENT_W - 108 * mm - 18]),
        P("Lectura: la exactitud casi no distingue; el log loss muestra que las probabilidades de la red son mejores. <b>Error común</b>: elegir un modelo solo por exactitud cuando los errores confiados importan.", "cell")]))
    s.append(Spacer(1, 3))
    s += para("¿Y si esa ventaja se debe a <b>cuáles</b> 360 imágenes se usaron? El <b>bootstrap</b> responde reutilizando las 360 pérdidas por imagen (con los modelos ya entrenados, sin volver a ajustarlos): "
              "se sortean 360 imágenes <b>con reposición</b>, se calcula la diferencia de pérdidas medias y se repite B = 2 000 veces. Los percentiles 2.5 y 97.5 forman un intervalo del 95 %. "
              f"En las diapositivas el intervalo es [0.0478; 0.0901]; en nuestra reproducción, [{f(BT['ic'][0], 4)}; {f(BT['ic'][1], 4)}]. En ambos <b>el 0 queda fuera</b>: hay evidencia de menor pérdida de la red, "
              "bajo independencia por imagen. No dice nada sobre la utilidad práctica de la mejora ni sobre un cambio de cámara.")
    s.append(img("c44_bootstrap.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Un acierto más son 100/360 ≈ 0.28 puntos de exactitud", "Bootstrap: remuestrear las pérdidas por imagen, modelos fijos; el IC de la diferencia excluye 0 → evidencia de menor pérdida",
                       "El IC no cubre la utilidad práctica ni el cambio de cámara"]))

    # 5 robustez
    s.append(h("5", "¿Resiste otras condiciones? Robustez y límites de una prueba"))
    s += para("Una prueba solo habla de las condiciones en que se midió: esta colección, esta cámara, estos trazos. Para ver si los modelos toleran un cambio, la clase hizo una <b>prueba de estrés</b>: "
              "desplazar cada imagen <b>una columna a la derecha</b> (la primera columna se rellena con 0 y la última original se descarta) y evaluar <b>los mismos modelos sin reajustarlos</b>. "
              f"La exactitud de la logística cae de {pc(LG['acc'])} a {pc(DP['log']['acc'])} y la de la red, de {pc(RD['acc'])} a {pc(DP['red']['acc'])} (nuestra reproducción). Con 10 clases, adivinar daría 10 %. "
              "Para una persona es «la misma imagen»; para el modelo, los números cambiaron mucho. La red desplazada acierta algo más, pero su log loss es peor porque asigna probabilidades muy pequeñas al dígito real.")
    X0 = np.array([[0, 5, 9, 2], [1, 0, 7, 3], [0, 4, 8, 0], [2, 6, 0, 1]])
    Xs = np.zeros_like(X0)
    Xs[:, 1:] = X0[:, :-1]
    s.append(ejemplo("desplazar una matriz de 4 × 4 una columna a la derecha", [
        pasos([["Fila", "Original → desplazada", "Qué pasó"],
               ["1", f"{X0[0].tolist()} → {Xs[0].tolist()}", "entra un 0 a la izquierda y sale el último (2)"],
               ["2", f"{X0[1].tolist()} → {Xs[1].tolist()}", "cada número se corre una posición"]],
              [14 * mm, 82 * mm, CONTENT_W - 96 * mm - 18]),
        P("Lectura: todos los valores se movieron de posición; para un modelo que lee «posición 3 = tal intensidad», es una entrada completamente distinta. <b>Error común</b>: creer que un buen resultado original garantiza buenos resultados con cualquier cambio de entrada.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c44_desplazadas.png", width=CONTENT_W * 0.42))
    s.append(Spacer(1, 2))
    s.append(img("c44_estres.png", width=CONTENT_W * 0.78))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Una prueba habla de sus condiciones: colección, cámara, escritores", "El desplazamiento es una prueba de estrés artificial, no una cámara real", "Un buen resultado original no garantiza robustez"]))

    # 6 embeddings
    s.append(h("6", "Embeddings: convertir significado en cercanía"))
    s += para("Un <b>vector</b> es una lista ordenada de números. Un <b>embedding</b> usa un vector para representar un objeto (un texto, una imagen) de modo que objetos parecidos queden <b>cerca</b>. "
              "Las coordenadas las aprende un modelo con ejemplos; no son probabilidades ni etiquetas de temas. Con vectores se puede medir cercanía de dos maneras: la <b>distancia</b> (qué tan lejos están los puntos) y la <b>similitud coseno</b> (qué tan parecida es su dirección; 1 = misma dirección).")
    q, a, b = np.array([0.2, 0.8]), np.array([0.35, 0.7]), np.array([0.8, 0.2])
    u, v, w = np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([4.0, 0.5])
    cosf = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    s.append(ejemplo("tres vectores inventados (distintos de la diapositiva): U = (1, 2), V = (2, 3), W = (4, 0.5)", [
        pasos([["Paso", "U–V", "U–W"],
               ["Diferencias", "(2 − 1, 3 − 2) = (1, 1)", "(4 − 1, 0.5 − 2) = (3, −1.5)"],
               ["Distancia = raíz de la suma de cuadrados", f"√(1 + 1) = {f(np.linalg.norm(u - v), 3)}", f"√(9 + 2.25) = {f(np.linalg.norm(u - w), 3)}"],
               ["Producto punto U · otro", f"1·2 + 2·3 = {f(u @ v, 0)}", f"1·4 + 2·0.5 = {f(u @ w, 0)}"],
               ["Normas", f"{f(np.linalg.norm(u), 3)} y {f(np.linalg.norm(v), 3)}", f"{f(np.linalg.norm(u), 3)} y {f(np.linalg.norm(w), 3)}"],
               ["Coseno = punto ÷ (norma × norma)", f"{f(cosf(u, v), 3)}", f"{f(cosf(u, w), 3)}"]],
              [62 * mm, 52 * mm, CONTENT_W - 114 * mm - 18]),
        P("Lectura: V está más cerca de U (distancia menor y coseno mayor). <b>Error común</b>: creer que distancia y coseno siempre coinciden; un vector duplicado, como (2, 4) respecto de (1, 2), tiene coseno 1 pero está a distancia 2.236.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c44_embedding.png", width=CONTENT_W * 0.8))
    s.append(Spacer(1, 2))
    s += para("Esta idea sostiene la <b>búsqueda por significado</b>: se convierten la consulta y los documentos en vectores y se devuelven los más cercanos. Por palabras se busca coincidencia de términos (falla si la fuente dice «reembolso» y tú escribes «cobro duplicado»); "
              "por significado se compara cercanía, pero un texto cercano puede <b>no responder</b> la pregunta, ser de otro servicio o estar desactualizado. Una <b>base de datos vectorial</b> guarda los puntos cercanos cerca unos de otros para encontrarlos rápido: es una forma de guardar, no un método de aprendizaje. "
              "CLIP aplica la idea a imagen y texto (da una similitud, no una probabilidad de daño) y la recomendación de contenidos parte de una tabla personas × cursos (1 = hay clic; 0 = no hay clic registrado, no rechazo).")
    s.append(recuerda(["Embedding = vector que representa un objeto; lo parecido queda cerca", "Distancia: raíz de la suma de cuadrados; coseno: punto ÷ (norma × norma)", "Cercano no significa «responde la pregunta»: hay que verificar"]))

    # 7 generar y preentrenados
    s.append(h("7", "Generar contenido, aprender sin etiquetas y reutilizar modelos"))
    s += para("<b>Generar texto.</b> La instrucción dice qué resultado se quiere y el contexto es el texto disponible (antecedentes y lo ya escrito). Un modelo de lenguaje aprende regularidades en el orden de las palabras y, al generar, <b>añade unidades una tras otra</b> usando el contexto. "
              "Producir una continuación plausible no garantiza que sea verdadera. <b>Generar imágenes (difusión).</b> Un modelo parte de ruido visual y lo transforma de a poco en una imagen guiada por la descripción; se evalúan objetos, cantidades y relaciones, y el realismo no acredita un hecho.")
    s += para("<b>De dónde viene la señal para aprender.</b> En el aprendizaje <b>supervisado</b> la respuesta es externa (el dígito correcto). En el <b>autosupervisado</b> la respuesta sale de los propios datos: si se oculta la palabra «crédito» en «el banco aprobó el crédito», la palabra original sirve para comparar. "
              "<b>Modelos preentrenados.</b> Alguien ya entrenó un modelo con otros datos; se puede aprovechar con tres grados de cambio: <b>instrucciones y contexto</b> (no se reentrena), <b>reutilizar representaciones</b> (vectores + un clasificador propio) o <b>ajuste fino</b> (se modifica el modelo con datos locales). "
              "Hay que comprobar que lo aprendido sea pertinente (un modelo de fauna de otro continente no sirve igual en Chile) y que se pueda evaluar con datos locales. AlphaFold 3 es un ejemplo científico: propone cómo se disponen proteínas, ADN y otras moléculas y se verifica contra estructuras determinadas experimentalmente; una estructura predicha orienta la investigación pero no demuestra por sí sola que un compuesto sea un tratamiento eficaz.")
    s.append(ejemplo("verificar un borrador contra su fuente", [
        P("Fuente: «La revisión del cobro puede tardar hasta cinco días hábiles.» · Borrador: «Su devolución está aprobada y llegará mañana.»", "cell"),
        Spacer(1, 2),
        pasos([["Paso", "Afirmación", "Respaldo en la fuente", "Decisión"],
               ["1", "Su devolución está aprobada", "No: la fuente no informa el resultado del caso", "quitarla o declarar que falta"],
               ["2", "Llegará mañana", "No: la fuente habla de revisión, no de fecha de abono", "quitarla o declarar que falta"],
               ["3", "Respuesta corregida", "«hasta cinco días hábiles de revisión»", "no confirmar aprobación ni fecha"]],
              [11 * mm, 44 * mm, 63 * mm, CONTENT_W - 118 * mm - 18]),
        P("Lectura: de un solo borrador salieron dos afirmaciones sin respaldo. <b>Error común</b>: dar por buena una redacción fluida sin verificar cada hecho.", "cell")]))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Generar = añadir unidades sucesivamente con el contexto; plausible ≠ verdadero", "Supervisado: respuesta externa; autosupervisado: la respuesta sale de los datos", "Preentrenado: punto de partida; su utilidad local se demuestra con datos locales"]))

    # 8 costos y diseño
    s.append(h("8", "Costos, carga de revisión y una prueba que represente el uso"))
    s += para("Dos errores distintos no cuestan lo mismo. En una inspección hay <b>defectos omitidos</b> (falsos negativos) y <b>falsas alarmas</b> (falsos positivos). Se les asigna un costo por caso y se suma. "
              "Además, cada alerta consume una <b>revisión</b>: aunque el sistema detecte muchos defectos, si genera más alertas de las que se pueden revisar, no sirve. Finalmente, la prueba debe parecerse al uso: "
              "si habrá <b>personas o envases nuevos</b>, todas las observaciones de una persona o envase van al mismo conjunto; si habrá <b>meses futuros</b>, se evalúa en fechas posteriores al desarrollo; si cambia la cámara, hay que observar datos de esa cámara.")
    n2, d2, fn2, fp2, cfn2, cfp2, cap2 = 500, 10, 2, 15, 40, 1, 12
    det2 = d2 - fn2
    s.append(ejemplo(f"{n2} piezas, {d2} con defecto; se omiten {fn2}, hay {fp2} falsas alarmas; costos {cfn2} y {cfp2}; capacidad {cap2} revisiones", [
        pasos([["Concepto", "Cálculo", "Resultado"],
               ["Costo de omitidos", f"{fn2} × {cfn2}", f"{fn2 * cfn2}"],
               ["Costo de falsas alarmas", f"{fp2} × {cfp2}", f"{fp2 * cfp2}"],
               ["Costo total", f"{fn2 * cfn2} + {fp2 * cfp2}", f"{fn2 * cfn2 + fp2 * cfp2} unidades"],
               ["Defectos detectados", f"{d2} − {fn2}", f"{det2}"],
               ["Revisiones si se revisan todas las alertas", f"{det2} + {fp2}", f"{det2 + fp2}"],
               ["¿Alcanza la capacidad de {0}?".format(cap2), f"{det2 + fp2} frente a {cap2}", "No: faltan {0}".format(det2 + fp2 - cap2)],
               ["Exactitud del sistema", f"({n2} − {fn2} − {fp2}) ÷ {n2}", pc((n2 - fn2 - fp2) / n2, 1)],
               ["«Decir siempre sano»", f"({n2} − {d2}) ÷ {n2}; costo {d2} × {cfn2}", f"{pc((n2 - d2) / n2, 1)}; costo {d2 * cfn2}"]],
              [62 * mm, 58 * mm, CONTENT_W - 120 * mm - 18]),
        P("Lectura: «decir siempre sano» tiene <b>mayor</b> exactitud (98.0 % frente a 96.6 %), pero omite todos los defectos y su costo es mucho más alto (10 × 40 = 400 frente a 2 × 40 + 15 × 1 = 95). <b>Error común</b>: juzgar por la exactitud sin mirar el costo y la capacidad de revisión.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c44_costo.png", width=CONTENT_W * 0.78))
    s.append(Spacer(1, 2))
    s += para(f"<b>Caso integrador.</b> Hay {mil(EN['fotos'])} fotografías de {mil(EN['envases'])} envases (cinco vistas cada uno). La decisión es por envase, así que la unidad de evaluación es el <b>envase</b>, no la foto: "
              f"si se separara por foto, vistas del mismo envase quedarían en entrenamiento y prueba y la prueba «ya conocería» el objeto. Una separación por envase 70/15/15 daría {EN['tr']}, {EN['va']} y {EN['te']} envases ({mil(EN['fotos_tr'])}, {EN['fotos_va']} y {EN['fotos_te']} fotos). "
              "Para el próximo mes (cámara y línea nuevas) hay que reunir datos de esas condiciones, reservar envases que no se usen para adaptar ni elegir, comparar una referencia simple con un candidato y medir defectos omitidos, falsas alarmas y revisiones por envase; el bootstrap remuestrea <b>envases completos</b>. "
              "Si después de mirar la prueba se cambia el procedimiento, hace falta otra evaluación independiente.")
    s += para("<b>Restricciones y automatización.</b> Además del error hay tiempo (latencia: si llega una pieza por segundo y se tardan cinco en procesar cada una, se acumula trabajo), costo, datos y mantenimiento. "
              "Conviene empezar por una <b>referencia simple</b> (regresión, árbol o una regla basada en historia reciente): si un modelo complejo no la supera, no vale la pena. Y se decide <b>qué parte automatizar</b>: proponer un borrador (alguien lo revisa), priorizar casos (se observan los que quedan atrás) o ejecutar una decisión (con evidencia, límites y seguimiento). La revisión humana también puede fallar o sobrecargarse.")
    s.append(recuerda(["Costo = Σ cantidad × costo por caso; revisiones = detectados + falsas alarmas", "La unidad de separación es la unidad de decisión (envase, persona, fecha)", "Empieza con una referencia simple y define qué automatizar"]))

    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 3]))
            i += 3
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C44_Apuntes_AplicacionesIA_DeepLearning.pdf", out, "Apuntes · Clase 44 · Aplicaciones de IA y Deep Learning")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#4F46E5">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C44))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 44 · Aplicaciones de IA y Deep Learning", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C44))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: pérdida de una imagen = −ln(p del dígito real) · log loss = promedio de las pérdidas · exactitud = aciertos ÷ imágenes · costo = Σ cantidad × costo por caso · revisiones = detectados + falsas alarmas · distancia = √(Σ diferencias²) · coseno = (A · B) ÷ (‖A‖ ‖B‖).",
                           "Valores útiles: ln(2) = 0.6931 · ln(10) = 2.3026 · ln(0.5) = −0.6931.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C44)], bg=SOFT, border=C44, pad=5))
    s.append(Spacer(1, 2))

    pA = [0.80, 0.25, 0.05]
    pcls = [0.02, 0.03, 0.05, 0.10, 0.55, 0.05, 0.04, 0.06, 0.07, 0.03]
    pa_, pb_ = [0.99, 0.98, 0.97, 0.99, 0.02], [0.70, 0.70, 0.70, 0.70, 0.30]
    s += bloque(1, "Pérdida de una predicción", "★", "Un clasificador de dígitos asigna la probabilidad indicada a la clase REAL de cada imagen.", [
        "Calcula −ln(p) para p = 0.80, 0.25 y 0.05.",
        "Para las cuatro imágenes con p real = 0.90, 0.50, 0.20 y 0.99, calcula el log loss (promedio).",
        "¿Cuántas veces mayor es la pérdida de p = 0.05 que la de p = 0.80? ¿Qué enseña eso sobre los errores confiados?"])
    s += bloque(2, "De las probabilidades a la decisión", "★★", "Para una imagen, las probabilidades de los dígitos 0 a 9 son: " + ", ".join(f"{k}: {p:.2f}" for k, p in enumerate(pcls)) + ". El dígito real es 3.", [
        "Verifica que suman 1 y di qué dígito predice el modelo. ¿Acierta?",
        "Calcula la pérdida de esa imagen. ¿Cuánto habría sido si el real fuera 4?",
        "Si la regla es «aceptar sin revisión solo si la probabilidad máxima es ≥ 0.90», ¿esta imagen se acepta? Explica por qué esa regla no garantiza el acierto."])
    s += bloque(3, "Exactitud y log loss: no dicen lo mismo", "★★", "Dos modelos, A y B, se evalúan en 5 imágenes. Probabilidad asignada al dígito REAL: A: " + ", ".join(f"{p:.2f}" for p in pa_) + ". B: " + ", ".join(f"{p:.2f}" for p in pb_) + ". Ambos aciertan en las primeras 4 y fallan en la quinta.", [
        "Calcula la exactitud de cada modelo.",
        "Calcula el log loss de A y de B.",
        "¿Cuál es mejor según la exactitud? ¿Y según el log loss? ¿Qué imagen explica la diferencia?"])
    s += bloque(4, "Un acierto más: ¿cuánto es?", "★", "En 360 imágenes de prueba, el modelo 1 acierta 353 y el modelo 2 acierta 354.", [
        "Calcula la exactitud de cada modelo (en %).",
        "¿Cuántos puntos porcentuales de diferencia hay? Compara con 100/360.",
        "Si el log loss del modelo 1 es 0.1353 y el del 2 es 0.0657, calcula la diferencia. ¿Qué indica?"])
    s += bloque(5, "Costo de errores y carga de revisión", "★★", "Una línea revisa 2 000 piezas, 50 con defecto. El sistema omite 5 defectos y manda 40 piezas sanas a revisión. Omitir cuesta 80 unidades y una falsa alarma, 2. Se pueden revisar 60 alertas.", [
        "Calcula el costo total de los errores.",
        "Calcula cuántas revisiones se necesitan y si alcanzan. ¿Cuántas faltan o sobran?",
        "Calcula la exactitud del sistema y la de «decir siempre sano». Compara también sus costos."])
    s += bloque(6, "Separar por la unidad correcta", "★★", "Hay 800 envases con 6 fotografías cada uno (4 800 fotos). Se quiere 70 % entrenamiento, 15 % validación y 15 % prueba.", [
        "Si se separa por envase, ¿cuántos envases y cuántas fotos van a cada conjunto?",
        "Si en cambio cada foto cae al azar en entrenamiento con probabilidad 0.70, ¿qué probabilidad hay de que las 6 fotos de un envase caigan todas en entrenamiento?",
        "Con esa separación por foto, ¿qué fracción aproximada de los envases tendría fotos en más de un conjunto? ¿Por qué es un problema?"])
    s += bloque(7, "Verificar una respuesta generada", "★", "Fuente: «El envío puede demorar hasta 7 días hábiles desde el despacho.» Borrador: «Su pedido llegará en 3 días y el envío es gratis.»", [
        "Separa el borrador en afirmaciones.",
        "Para cada una, indica si la fuente la respalda.",
        "Escribe una respuesta corregida que no invente datos."])
    s += bloque(8, "Distancia y coseno entre vectores", "★★★", "Tres textos tienen estos vectores: Q = (3, 4), A = (4, 3), B = (−4, 3) y C = (6, 8).", [
        "Calcula la distancia de Q a A, a B y a C.",
        "Calcula la similitud coseno de Q con A, B y C.",
        "¿Cuál es el más cercano a Q por distancia? ¿Y por coseno? Explica por qué C no coincide."])
    s += bloque(9, "Interpretar un intervalo bootstrap", "★★", "Se remuestrearon 360 imágenes 2 000 veces y se calculó, en cada remuestra, la pérdida media de la logística menos la de la red.", [
        "Si el IC del 95 % es [0.02; 0.09], ¿qué concluyes? ¿Y si fuera [−0.01; 0.06]?",
        "¿Qué NO concluyes en el primer caso respecto de una cámara distinta?",
        "¿Qué cambiaría si en vez de imágenes sueltas hubiera 5 fotos por envase?"])
    s += bloque(10, "Desplazar una imagen", "★", "Una imagen de 4 × 4 tiene filas: [0, 5, 9, 2], [1, 0, 7, 3], [0, 4, 8, 0], [2, 6, 0, 1].", [
        "Escribe la imagen desplazada una columna a la derecha (primera columna en 0, última original descartada).",
        "Si las intensidades máximas son 16, calcula el valor escalado de 9, 7 y 2.",
        "Con 10 clases, ¿qué exactitud daría adivinar al azar? ¿Por qué comparar con ese valor ayuda a leer la prueba?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C44))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [KeepTogether([P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C44), Spacer(1, 2)])]

    l1 = [-log(p) for p in pA]
    l2 = [-log(p) for p in (0.90, 0.50, 0.20, 0.99)]
    s += sol("Bloque 1", [
        ["1", f"−ln(0.80) = {f(l1[0], 4)}; −ln(0.25) = {f(l1[1], 4)}; −ln(0.05) = {f(l1[2], 4)}."],
        ["2", f"Pérdidas: {', '.join(f(x, 4) for x in l2)}. Suma = {f(sum(l2), 4)}; log loss = {f(sum(l2), 4)} ÷ 4 = {f(np.mean(l2), 4)}."],
        ["3", f"{f(l1[2], 4)} ÷ {f(l1[0], 4)} = {f(l1[2] / l1[0], 1)} veces. Un error muy confiado pesa muchísimo más que un acierto razonable: el log loss castiga las probabilidades extremas mal puestas."]])
    s += sol("Bloque 2", [
        ["1", f"Suma = {f(sum(pcls), 2)}. Predice el dígito 4 (p = 0.55). El real es 3: no acierta."],
        ["2", f"Pérdida = −ln(0.10) = {f(-log(0.10), 4)}. Si el real fuera 4: −ln(0.55) = {f(-log(0.55), 4)}."],
        ["3", "0.55 < 0.90: no se acepta sin revisión (va a revisión). La regla mira solo la confianza; una predicción con p ≥ 0.90 también puede estar equivocada (en la clase hubo errores con 0.76 a 0.94)."]])
    la = [-log(p) for p in pa_]
    lb = [-log(p) for p in pb_]
    s += sol("Bloque 3", [
        ["1", "A: 4 de 5 = 0.80; B: 4 de 5 = 0.80. Igual."],
        ["2", f"A: {', '.join(f(x, 4) for x in la)} → suma {f(sum(la), 4)} → log loss {f(np.mean(la), 4)}. B: {', '.join(f(x, 4) for x in lb)} → suma {f(sum(lb), 4)} → log loss {f(np.mean(lb), 4)}."],
        ["3", f"Exactitud: empatan. Log loss: B es mejor ({f(np.mean(lb), 4)} frente a {f(np.mean(la), 4)}). La quinta imagen explica la diferencia: A le asignó 0.02 al dígito real (pérdida {f(la[4], 3)}, el {f(la[4] / sum(la) * 100, 0)} % de su pérdida total)."]])
    s += sol("Bloque 4", [
        ["1", f"353 ÷ 360 = {pc(353 / 360)}; 354 ÷ 360 = {pc(354 / 360)}."],
        ["2", f"Diferencia = {f(100 / 360, 2)} puntos porcentuales = 100/360: un solo acierto."],
        ["3", "0.1353 − 0.0657 = 0.0696. El modelo 2 tiene bastante menos pérdida: sus probabilidades son mejores, aunque la exactitud casi no cambie."]])
    ck = 5 * 80 + 40 * 2
    s += sol("Bloque 5", [
        ["1", f"5 × 80 + 40 × 2 = 400 + 80 = {ck} unidades."],
        ["2", f"Detectados = 50 − 5 = 45; revisiones = 45 + 40 = 85. Capacidad 60: faltan {85 - 60}."],
        ["3", f"Sistema: (2000 − 5 − 40) ÷ 2000 = {pc((2000 - 45) / 2000)}. «Siempre sano»: (2000 − 50) ÷ 2000 = {pc(1950 / 2000)}, con costo 50 × 80 = {50 * 80}. Las exactitudes son parecidas, pero el costo difiere mucho ({ck} frente a {50 * 80}): la exactitud sola no cuenta cuánto cuesta cada error."]])
    s += sol("Bloque 6", [
        ["1", "Envases: 0.70 × 800 = 560, 0.15 × 800 = 120 y 120. Fotos: 560 × 6 = 3 360; 120 × 6 = 720 y 720."],
        ["2", f"0.70⁶ = {f(0.7 ** 6, 4)}."],
        ["3", f"Que las 6 fotos caigan en el mismo conjunto: 0.70⁶ + 0.15⁶ + 0.15⁶ = {f(0.7 ** 6 + 2 * 0.15 ** 6, 4)}. Entonces ≈ {f((1 - 0.7 ** 6 - 2 * 0.15 ** 6) * 100, 1)} % de los envases tiene fotos en más de un conjunto: la prueba conoce al objeto y sobreestima el desempeño con envases nuevos."]])
    s += sol("Bloque 7", [
        ["1", "(a) «Su pedido llegará en 3 días»; (b) «el envío es gratis»."],
        ["2", "(a) No: la fuente dice «hasta 7 días hábiles desde el despacho», no 3 días. (b) No: la fuente no habla del costo del envío."],
        ["3", "«El envío puede demorar hasta 7 días hábiles desde el despacho. Con esta información no puedo confirmar el plazo exacto de su pedido ni si el envío tiene costo.»"]])
    Q, A, Bv, Cv = np.array([3, 4]), np.array([4, 3]), np.array([-4, 3]), np.array([6, 8])
    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    s += sol("Bloque 8", [
        ["1", f"Q–A: √(1 + 1) = {f(np.linalg.norm(Q - A), 3)}; Q–B: √(49 + 1) = {f(np.linalg.norm(Q - Bv), 3)}; Q–C: √(9 + 16) = {f(np.linalg.norm(Q - Cv), 3)}."],
        ["2", f"‖Q‖ = 5. Q·A = 12 + 12 = 24 → coseno {f(cs(Q, A), 3)}. Q·B = −12 + 12 = 0 → coseno {f(cs(Q, Bv), 3)}. Q·C = 18 + 32 = 50, ‖C‖ = 10 → coseno {f(cs(Q, Cv), 3)}."],
        ["3", "Por distancia, el más cercano es A (1.414). Por coseno, C (1.000: misma dirección). C no coincide porque es un vector más «largo» en la misma dirección: el coseno ignora la longitud y la distancia no. B es perpendicular (coseno 0)."]])
    s += sol("Bloque 9", [
        ["1", "[0.02; 0.09] no contiene 0: evidencia de que la logística pierde más que la red en estos casos. [−0.01; 0.06] contiene 0: no hay evidencia clara de diferencia (no prueba equivalencia)."],
        ["2", "No dice nada de una cámara distinta: el intervalo solo refleja la variación de los casos de prueba con modelos fijos, bajo independencia por imagen. Tampoco mide la utilidad práctica."],
        ["3", "Las 5 fotos de un envase no son independientes: hay que remuestrear envases completos para conservar juntas sus vistas; con imágenes sueltas el intervalo sería demasiado angosto."]])
    X0 = np.array([[0, 5, 9, 2], [1, 0, 7, 3], [0, 4, 8, 0], [2, 6, 0, 1]])
    Xs = np.zeros_like(X0)
    Xs[:, 1:] = X0[:, :-1]
    s += sol("Bloque 10", [
        ["1", "Filas desplazadas: " + "; ".join(str(r) for r in Xs.tolist()) + "."],
        ["2", f"9 ÷ 16 = {f(9 / 16, 4)}; 7 ÷ 16 = {f(7 / 16, 4)}; 2 ÷ 16 = {f(2 / 16, 4)}."],
        ["3", "Adivinar entre 10 clases acierta el 10 %. Sirve como piso: una exactitud de 39.72 % o 45 % tras desplazar es bastante mejor que el azar, pero muy inferior al 98 % original."]])
    build("apuntes/C44_Ejercicios_Cuaderno_AplicacionesIA_DeepLearning.pdf", s, "Ejercicios para el cuaderno · Clase 44")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
