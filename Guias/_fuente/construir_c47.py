"""Clase 47 · Embeddings y representaciones vectoriales (Prof. Eloy Alvarado): guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c47_numeros.json (lo escribe figuras_c47.py); los de los ejemplos y ejercicios se calculan aquí.
Los números de las diapositivas (corrida del profesor) se citan como «diapositivas». Con el programa del profesor, nuestra reproducción coincide
con todas ellas; lo que se rotula «nuestro cálculo» o «ejemplo propio» no es de la clase."""
import json
from math import log, sqrt
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import CondPageBreak, KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c47_numeros.json").read_text(encoding="utf-8"))
C47 = colors.HexColor("#0E7490")
C47_BG = colors.HexColor("#ECFEFF")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
G, V1, T3, CO, H, UM, UX, MX = N["geom"], N["v01"], N["tres"], N["codigos"], N["hit"], N["umbral"], N["umbral_extra"], N["maximos"]
V11, PCA, DAT, IDF = N["v11"], N["pca"], N["datos"], N["idf"]["tabla"]


def f(x, d=3):
    return f"{x:.{d}f}"


def mn(v, d=3):
    return f"{v:.{d}f}".replace("-", "−")


def pc(x, d=1):
    return f"{x * 100:.{d}f} %"


# ======================================================================= GUÍA
def tema(s, num, titulo, idea, conceptos, figuras, simple, formulas, salida, cuidado, minimo=30):
    s.append(CondPageBreak(minimo * mm))
    s.append(P(f'<font color="#0E7490">{num}</font>  {titulo}', "h2"))
    s.append(box([P(idea, "big")], bg=SOFT, border=C47, pad=5))
    s.append(Spacer(1, 3))
    s.append(numbered(conceptos, C47))
    s.append(Spacer(1, 2))
    left = [P("En palabras simples", "h3"), box([P(simple, "cell")], bg=C47_BG, border=C47, width=HALF, pad=5)]
    if formulas:
        left += [Spacer(1, 3), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida, "cell", C47)]
    if cuidado:
        right += [Spacer(1, 3), box([P("<b>Cuidado.</b> " + cuidado, "cell")], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
    s.append(KeepTogether([two_cols(left, right)]))
    s.append(Spacer(1, 4))
    for fig, pie, *w in figuras:
        s.append(KeepTogether([img(fig, width=CONTENT_W * (w[0] if w else 0.92)), P(pie, "small"), Spacer(1, 3)]))
    s.append(Spacer(1, 4))


def guia():
    s = []
    r = N["rangos"]
    s.append(band("Clase 47 · Embeddings y representaciones vectoriales",
                  "Profesor Eloy Alvarado · cómo convertir un texto en números para que un computador encuentre la ficha de ayuda que responde una pregunta", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Para que un computador compare textos hay que convertirlos en vectores.</b> "
                    "Contar palabras es lo más simple, pero «clave» y «contraseña» quedan como vectores perpendiculares: para un conteo no se parecen en nada. "
                    "Un <b>embedding</b> es un vector de 384 números producido por una red ya entrenada (MiniLM), que deja cerca los textos con significado parecido. "
                    "Con esos vectores se <b>compara</b> cada pregunta con cada una de las 12 fichas de ayuda mediante el <b>coseno</b>, se <b>ordenan</b> las fichas por puntaje y se <b>evalúa</b> "
                    "(rango, Hit@k, MRR, preguntas sin respuesta). Un buen puntaje no garantiza una buena respuesta: hay que revisar los errores.", "body")],
                 bg=C47_BG, border=C47, pad=6))
    s.append(Spacer(1, 4))
    s.append(img("c47_etapas.png", width=CONTENT_W * 0.97))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Red neuronal</b> (Clase 45): pesos que se ajustan para reducir un error",
                     "<b>Autoencoder y representación</b> (Clase 46): un vector que resume la entrada; aquí la entrada es texto",
                     "<b>Conteo de palabras</b> (Clase 40, spam): palabras como variables",
                     "<b>PCA</b> (Clase 32): resumir muchas coordenadas en pocas",
                     "<b>Validación y prueba</b> (Clases 30 y 38): elegir con una, evaluar al final con otra"], "cell", C47)]
    right = [P("Ruta de esta guía", "h3"),
             bullets(["Texto → vector: contar palabras y TF-IDF",
                      "Embedding (MiniLM) y cómo se aprendió",
                      "Geometría: norma, producto interno, coseno",
                      "Una consulta contra 12 fichas: matriz y ranking",
                      "Evaluar: rango, Hit@k, MRR y elegir k",
                      "Errores y límites: A17/A71, información insuficiente, cortes",
                      "Mirar el espacio con PCA y reutilizar embeddings"], "cell", C47)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Cómo leer los números de este material</b>", "cellb"),
                  P("La grabación <b>empieza unos 5 minutos tarde</b> (la primera frase está a medias) y tiene un <b>recreo</b> entre 1:10:26 y 1:28:13; con las diapositivas se cubre todo. "
                    "Las cifras de las diapositivas son la corrida del profesor: se ejecutó <b>su mismo programa</b> (<font name='Mono'>embeddings.py</font>, solo numpy, con los vectores de MiniLM ya calculados, sin internet) "
                    f"y <b>las {len(N['comparacion'])} cifras comparadas coinciden exactamente</b> (tabla en la última página). "
                    f"Se trabaja con las {DAT['n_val']} consultas de <b>validación</b>; las {DAT['n_prueba']} de prueba el profesor las reserva y aquí <b>no se evalúan</b>. "
                    "Lo que se rotula «nuestro cálculo» o «ejemplo propio» no es de la clase.", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))

    # ---------------------------------------------------------------- 1
    tema(s, "1", "Texto → vector: contar palabras, y por qué los sinónimos no se parecen",
         "<b>Un computador solo compara números.</b> La forma más simple de convertir un texto en vector es contar cuántas veces aparece cada palabra de un vocabulario. Sirve para palabras exactas, pero no entiende sinónimos.",
         [("Vector", "Una lista ordenada de números donde cada posición tiene un significado fijo. Aquí cada posición es una palabra del vocabulario, así que todos los textos tienen la misma cantidad de coordenadas."),
          ("Conteo de palabras", "El texto «video clave» se convierte en (1, 0, 1) si las columnas son clave, contraseña y video (diapositiva 7). Se ignora el orden: solo importa cuántas veces sale cada palabra."),
          ("Sinónimos perpendiculares", "«clave» = (1, 0, 0) y «contraseña» = (0, 1, 0) forman 90°: su coseno es 0. Para el conteo no tienen nada en común, aunque en la ficha D01 significan lo mismo."),
          ("El problema inverso", "«Me senté en el banco de la plaza» y «Pedí un crédito en el banco» tienen la misma palabra y el mismo conteo, pero sentidos distintos (diapositiva 10). El significado depende del contexto.")],
         [("c47_conteo.png", "Diapositiva 7: tres textos como vectores de conteo (izquierda) y por qué los sinónimos quedan perpendiculares (derecha)."),
          ("c47_sin_palabras.png", "Nuestro cálculo con la tokenización del programa del profesor (minúsculas, sin tildes ni signos): V01 y D01 no comparten ninguna palabra.", 0.9)],
         "Es como buscar ropa en una tienda que ordena por la etiqueta exacta: si buscas «polera» y la tienda escribió «camiseta», el buscador por etiqueta no encuentra nada, "
         "aunque cualquier persona sabe que es lo mismo. Necesitamos un método que «sepa» que clave y contraseña significan lo mismo.",
         ["«clave» = (1, 0, 0)", "«contraseña» = (0, 1, 0)", "ángulo 90°, coseno 0"],
         ["V01 «Olvidé mi clave, ¿cómo vuelvo a entrar?» y D01 (recuperar la contraseña) comparten 0 palabras",
          f"El vocabulario de las 12 fichas tiene {DAT['vocab']} palabras y «clave» no está entre ellas: se ignora. De las siete palabras de V01 solo «a» está en el vocabulario",
          f"TF-IDF sitúa D01 en el lugar {V1['rango_tfidf']} con puntaje 0 (diapositiva 34)",
          f"Las {V1['n_tfidf_no_nulos']} fichas con puntaje distinto de 0 (D03, D09, D04, D12, D11) se activan solo por la palabra de relleno «a» (nuestro cálculo)"],
         "Contar no es «malo»: conserva las coincidencias exactas (códigos, nombres, fechas), que serán útiles en la sección 7.")

    # ---------------------------------------------------------------- 2
    wq = V11["comunes"]
    tema(s, "2", "TF-IDF: contar, pero dándole más peso a las palabras raras",
         "<b>TF-IDF multiplica cuántas veces aparece una palabra en el texto (TF) por un peso que crece cuando la palabra aparece en pocas fichas (IDF).</b> Después se normaliza para comparar con el coseno.",
         [("TF", "Term Frequency: cuántas veces aparece la palabra en el texto (diapositiva 30)."),
          ("DF e IDF", "DF (Document Frequency) es en cuántas de las N = 12 fichas aparece la palabra. IDF = ln((1 + N)/(1 + DF)) + 1: una palabra en una sola ficha pesa 2.87; una en 11 fichas, 1.08."),
          ("Peso y normalización", "Peso = TF × IDF para cada palabra del vocabulario; luego se divide el vector por su norma para que mida 1 (ver sección 4). Vocabulario e IDF se calculan con las 12 fichas; las palabras que no están ahí se ignoran (diapositiva 28)."),
          ("Qué hace y qué no", "Premia las coincidencias de palabras raras, como un código de proyecto. No entiende sinónimos. En la clase es la <b>receta de referencia</b> contra la que se compara MiniLM.")],
         [("c47_idf.png", "Nuestro cálculo con las 12 fichas: la curva del IDF según el DF (izquierda) y cinco palabras de ejemplo (derecha)."),
          ("c47_tfidf_v11.png", "Nuestro cálculo: V11 «¿Cuándo vence el proyecto A17?» frente a D11. Solo tres palabras son comunes y la suma de sus productos es el coseno que el programa asigna.")],
         "Si todas las fichas dicen «de», esa palabra no ayuda a distinguir una ficha de otra: es como gritar en una sala llena de gente. «A17», en cambio, aparece en una sola ficha: es una pista fuerte. "
         "El IDF le sube el volumen a lo raro y se lo baja a lo común.",
         ["TF-IDF = TF × IDF", "IDF = ln((1 + N) / (1 + DF)) + 1", "N = 12 fichas; ln es el logaritmo natural"],
         [f"«de»: DF = {IDF['de']['df']}, IDF = {f(IDF['de']['idf'], 2)}; «el»: DF = {IDF['el']['df']}, IDF = {f(IDF['el']['idf'], 2)}; «proyecto»: DF = {IDF['proyecto']['df']}, IDF = {f(IDF['proyecto']['idf'], 2)}; «a17»: DF = {IDF['a17']['df']}, IDF = {f(IDF['a17']['idf'], 2)}",
          f"De las {DAT['vocab']} palabras del vocabulario, {N['idf']['df_1']} están en una sola ficha",
          f"V11 con D11: «a17» aporta {f(wq[0]['prod'])}, «el» {f(wq[1]['prod'])} y «proyecto» {f(wq[2]['prod'])}; suma {f(V11['suma'])} (el puntaje de TF-IDF de V11 con D11, nuestro cálculo)"],
         "En la transcripción aparece «TF y DF»: es TF-IDF (corregido). Y TF-IDF no es otro embedding aprendido: sus pesos se calculan contando palabras en las 12 fichas.")

    # ---------------------------------------------------------------- 3
    tema(s, "3", "Embedding: un vector aprendido (MiniLM) en vez de un conteo",
         "<b>Un embedding es un vector que una red ya entrenada produce para cada texto, de modo que textos con significado parecido queden con vectores parecidos.</b> Aquí son 384 números por texto.",
         [("Embedding", "z = f_θ(x): entra el texto x, un codificador con pesos aprendidos θ lo procesa y sale un vector de d = 384 coordenadas (diapositiva 6). Es la misma idea del encoder de la Clase 46, pero con texto."),
          ("Señal de relación", "Para aprender, las parejas de entrenamiento llevan una marca: «paráfrasis», «relevante» (1) o «no relevante» (0) (diapositiva 8). Esa marca es la referencia de la pérdida, no el coseno que calcula el modelo."),
          ("El aprendizaje ocurrió antes", "Los autores entrenaron MiniLM con textos externos. Hoy solo se hace <b>inferencia</b>: los pesos no cambian. Agregar una ficha significa calcular un vector nuevo, no reentrenar la red (diapositiva 9)."),
          ("Del texto al vector", "Texto → tokens → vectores con contexto → vector de oración (MiniLM: media de las posiciones válidas, 384 valores). MPNet y LaBSE entregan 768. Los espacios no son intercambiables: consulta y fichas deben usar el mismo codificador (diapositivas 12 a 14).")],
         [],
         "Piensa en un <b>mapa gigante</b> con 384 direcciones. Un cartógrafo ya entrenado (MiniLM) puso cerca los puntos de frases parecidas y lejos las distintas. "
         "Tú no dibujas el mapa: solo lo usas. Y no puedes mezclar las coordenadas de un mapa con las de otro, aunque tengan el mismo número de ejes.",
         ["z = f_θ(x) = (z₁, z₂, …, z₃₈₄)", "destilación: error = ((0.4 − 0.2)² + (0.6 − 0.8)²) / 2 = 0.04"],
         [f"Modelo de la clase: {DAT['modelo']}; 384 coordenadas normalizadas, ya guardadas en <font name='Mono'>vectores.csv</font> (12 fichas + 32 consultas)",
          "Destilación (diapositiva 11): el maestro entrega la referencia (0.2, 0.8) y el estudiante parte en (0.4, 0.6); error medio cuadrático = 0.04. El maestro permanece fijo",
          "Hoy no se entrena nada: «hay datos locales para buscar y evaluar, no para entrenar MiniLM» (diapositiva 9)"],
         "MiniLM aquí produce vectores, no respuestas escritas (diapositiva 12). Y «ninguno tiene garantizado ser el mejor para nuestras consultas»: se valida con datos (diapositiva 13).", minimo=30)

    # ---------------------------------------------------------------- 4
    tema(s, "4", "Geometría: distancia, producto interno, norma y coseno",
         "<b>Con vectores se puede medir «parecido» de tres maneras. El coseno compara la dirección y descarta el tamaño; por eso es la medida que usa el buscador.</b>",
         [("Vector en el plano", f"Una flecha desde el origen. Su dirección es el perfil entre las coordenadas; su longitud es la escala. Ejemplo (diapositiva 15): q = (1, 1), A = (2, 0), B = (4, 4), que son conteos de «clave» y «recuperar»."),
          ("Distancia", f"‖q − a‖ = √2 ≈ {f(G['dist_a'])}; ‖q − b‖ = √18 ≈ {f(G['dist_b'])}. Como punto, A está más cerca de q (diapositiva 17)."),
          ("Producto interno", f"Multiplicar las coordenadas correspondientes y sumar: qᵀa = {G['dot_a']:g}; qᵀb = {G['dot_b']:g}. Si se duplica el documento (8, 8), el producto se duplica a {G['dot_2b']:g} aunque la dirección no cambió (diapositivas 18 y 19)."),
          ("Norma y normalización", f"‖z‖ = √(z₁² + … + z_d²) mide la longitud: ‖q‖ = √2, ‖a‖ = 2, ‖b‖ = √32. Dividir por la norma da un vector de largo 1 con la misma dirección; q y B quedan iguales, (0.7071, 0.7071) (diapositivas 20 y 21)."),
          ("Coseno", f"cos(q, z) = qᵀz / (‖q‖ ‖z‖) es el producto de los vectores normalizados: {f(G['cos_a'], 4)} para A (45°) y {G['cos_b']:.0f} para B (0°). 1 significa misma dirección, 0 perpendicular, −1 opuestos (diapositivas 22 y 23).")],
         [("c47_geometria.png", "Diapositivas 15 a 23: a la izquierda, los conteos y las distancias; a la derecha, los mismos vectores normalizados y el ángulo entre ellos.", 0.95)],
         "Imagina a dos personas apuntando con el dedo. La <b>distancia</b> mide cuán separados están sus pies; el <b>coseno</b> mide si apuntan hacia el mismo lado, sin importar cuán largo sea el brazo. "
         "Un texto repetido cuatro veces tiene la misma dirección (de qué habla) pero una longitud mayor: el coseno no se deja engañar por la longitud.",
         ["qᵀz = Σⱼ qⱼ zⱼ", "‖z‖ = √(z₁² + z₂² + … + z_d²)", "cos(q, z) = qᵀz / (‖q‖ ‖z‖)"],
         [f"Distancia: A {f(G['dist_a'])}, B {f(G['dist_b'])} (favorece a A). Producto interno: A {G['dot_a']:g}, B {G['dot_b']:g}. Coseno: A {f(G['cos_a'], 4)}, B {G['cos_b']:.0f} (favorece a B) — diapositiva 24",
          "El coseno 0.8 es un puntaje geométrico, <b>no un 80 % de probabilidad</b> de acierto (diapositiva 23)",
          "Una fila nula (norma 0) no se puede normalizar: el programa la deja en 0 en lugar de dividir"],
         "El coseno no es una solución universal: la medida se elige según el modelo y la tarea (diapositiva 24).")
    s.append(table([["Medida", "A", "B", "Qué compara"],
                    ["Distancia", f(G["dist_a"]), f(G["dist_b"]), "Separación entre los puntos finales"],
                    ["Producto interno", f"{G['dot_a']:g}", f"{G['dot_b']:g}", "Coordenadas, con el efecto de la escala"],
                    ["Coseno", f(G["cos_a"], 4), f"{G['cos_b']:.0f}", "Dirección, retirando la escala"]], [38 * mm, 24 * mm, 24 * mm, CONTENT_W - 86 * mm]))
    s.append(Spacer(1, 6))

    # ---------------------------------------------------------------- 5
    tema(s, "5", "Una consulta contra 12 fichas: del coseno en 384 dimensiones a un ranking",
         "<b>Con MiniLM los vectores vienen normalizados, así que el coseno es la suma de 384 productos. Hacerlo con cada pareja consulta-ficha da una matriz de 16 × 12; ordenar cada fila da el ranking.</b>",
         [("Una pareja", f"V01 con D01: se multiplican las 384 coordenadas y se suman. Los tres primeros productos son −0.00114, 0.00082 y −0.00218; las otras 381 coordenadas suman {f(V1['suma_resto'], 5)}; el total es {f(V1['suma'], 5)} ≈ 0.450 (diapositiva 32)."),
          ("La matriz", "Q (16 × 384) por Dᵀ (384 × 12) da S (16 × 12): una fila por consulta, una columna por ficha (diapositiva 33). Trasponer D pone las fichas en columnas."),
          ("Ordenar", "En cada fila se ordena de mayor a menor puntaje; los empates se resuelven por identificador de ficha. Siempre hay un primer lugar, aunque ninguna ficha responda."),
          ("Rango r", f"Posición de la ficha relevante en el ranking. Para V01: r = {V1['rango_tfidf']} con TF-IDF y r = {V1['rango_minilm']} con MiniLM (diapositiva 34). Para V02 con MiniLM: D06 (0.236), D02 (0.147) y D01 (0.131): r = 2 (diapositiva 35)."),
          ("Identificador", "D05 es solo el nombre de la ficha de pagos: el 05 no es un puntaje, una posición ni una coordenada (diapositiva 4).")],
         [("c47_producto384.png", "Nuestro cálculo para V01 con D01: los productos de las primeras 16 coordenadas (izquierda) y la suma acumulada de las 384 (derecha).", 0.9),
          ("c47_ranking_v01.png", "Diapositiva 34, completa: los 12 puntajes de V01 con TF-IDF y con MiniLM. Verde: la ficha correcta.", 0.93)],
         "Es como un concurso con <b>384 jueces</b>: cada uno da un poquito de puntos a la pareja (algunos positivos, otros negativos) y el total es el coseno. "
         "Se hace lo mismo con las 12 fichas, se ordenan y la de mayor puntaje queda primera. Que gane no significa que responda la pregunta.",
         ["S = Q Dᵀ  →  (16 × 384)(384 × 12) = (16 × 12)", "cos(V01, D01) = Σⱼ₌₁³⁸⁴ qⱼ dⱼ"],
         [f"De los 384 productos de V01 con D01, {V1['n_pos']} son positivos (suman {f(V1['pos_sum'])}) y {V1['n_neg']} negativos (suman {mn(V1['neg_sum'])}); total {f(V1['suma'])} (nuestro cálculo)",
          "V01 con las fichas: D01 0.450, D02 −0.150, D03 0.078, D12 −0.128 (diapositiva 33)",
          f"TF-IDF para V01: D03 {f(V1['top3_tfidf'][0][1])}, D09 {f(V1['top3_tfidf'][1][1])}, D04 {f(V1['top3_tfidf'][2][1])}; MiniLM: D01 {f(V1['top3_minilm'][0][1])}, D03 {f(V1['top3_minilm'][1][1])}, D05 {f(V1['top3_minilm'][2][1])}"],
         "Los cosenos de MiniLM de consultas distintas no son comparables como «seguridad»: V01 acierta con 0.450 y V06 acierta con 0.262 (ver los mapas).")
    s.append(img("c47_mapa_minilm.png", width=CONTENT_W * 0.9))
    s.append(P("Nuestro cálculo (coincide con el programa del profesor): la matriz S de MiniLM, 16 consultas × 12 fichas. Cuadro verde: la ficha correcta de cada consulta con respuesta; V13 a V16 no tienen ficha correcta.", "small"))
    s.append(Spacer(1, 4))
    s.append(img("c47_mapa_tfidf.png", width=CONTENT_W * 0.9))
    s.append(P("Nuestro cálculo: la misma matriz con TF-IDF. Muchos ceros: una pareja sin palabras en común no suma nada, aunque hablen de lo mismo.", "small"))
    s.append(Spacer(1, 4))

    # ---------------------------------------------------------------- 6
    h1_t, h1_m, h3_t, h3_m = H["tfidf"][0], H["minilm"][0], H["tfidf"][2], H["minilm"][2]
    tema(s, "6", "Evaluar: rango, Hit@k, MRR y elegir cuántas fichas mostrar",
         "<b>Para saber si el buscador funciona hay que medirlo con consultas cuya ficha correcta conocemos.</b> Hit@k cuenta en qué proporción de consultas la ficha correcta está entre las k primeras; MRR premia que esté más arriba.",
         [("Rango r", "La posición de la ficha relevante en el ranking de una consulta: 1 es la mejor. Es la base de las dos métricas."),
          ("Hit@k", f"Hit@k = (1/n) Σ 1[rᵢ ≤ k]. Con V01, V02 y V04 (rangos 1, 2 y 5) y k = 3, los aportes son 1, 1 y 0: Hit@3 = 2/3 ≈ {f(T3['hit3'] / 3)} (diapositiva 36). Cada fila es una consulta distinta."),
          ("MRR", f"MRR = (1/n) Σ 1/rᵢ (Mean Reciprocal Rank). Con los mismos tres rangos: (1 + 1/2 + 1/5)/3 ≈ {f(T3['mrr'])} (diapositiva 37). Distingue el primer lugar del segundo y <b>no depende de k</b>."),
          ("Elegir k", "Mostrar más fichas aumenta los aciertos, pero también lo que hay que leer. Cuando el ranking no cambia, subir k solo cambia el Hit@k; el MRR queda igual (diapositiva 44). Con 100 fichas, mostrar 3 a 5 sería mucho más útil que con 12."),
          ("Solo las respondibles", f"Hit@k y MRR se calculan con las {DAT['n_resp']} consultas que tienen ficha correcta (V01 a V12). Las {DAT['n_sin']} sin respuesta se tratan aparte (sección 7).")],
         [("c47_rangos.png", "Nuestro cálculo con las 12 consultas respondibles: rango de la ficha correcta (arriba) y rango recíproco con el MRR (abajo)."),
          ("c47_hitk.png", "Nuestro cálculo: Hit@k para todos los k de 1 a 12. La línea punteada marca k = 3, el valor de las diapositivas.", 0.92)],
         "Es como un examen: el buscador responde con una lista. Hit@k pregunta «¿estaba la respuesta correcta entre las primeras k líneas?». "
         "MRR da nota completa si estaba primera, la mitad si estaba segunda, un tercio si tercera… y luego promedia todas las consultas.",
         ["Hit@k = (1/n) Σᵢ 1[rᵢ ≤ k]", "MRR = (1/n) Σᵢ 1/rᵢ", "n = 12 consultas con respuesta"],
         [f"Hit@3: TF-IDF {h3_t}/12 = {f(h3_t / 12)}; MiniLM {h3_m}/12 = {f(h3_m / 12)}. MRR: {f(H['mrr_tfidf'])} y {f(H['mrr_minilm'])} (diapositiva 38)",
          f"Hit@1: TF-IDF {h1_t}/12; MiniLM {h1_m}/12. Pasar de k = 1 a k = 3 suma {h3_t - h1_t} consultas a TF-IDF y {h3_m - h1_m} a MiniLM (diapositiva 44)",
          f"MiniLM llega a 12/12 recién con k = {H['k_cobertura_total_minilm']} (V04 está en el lugar 5); TF-IDF, con k = {H['k_cobertura_total_tfidf']} (V07 está en el lugar 11) — nuestro cálculo",
          "V04 «¿Me conceden más días automáticamente si pido prórroga?»: TF-IDF la pone 4.ª y MiniLM 5.ª; con k = 3 ninguna la recupera"],
         "Son solo 12 consultas: una sola que cambie mueve Hit@k en 0.083 (1/12). El promedio favorable a MiniLM debe acompañarse de revisar los errores (diapositiva 38). Y k se elige con validación, no con la prueba.")

    # ---------------------------------------------------------------- 7
    c11, c12 = CO["V11"], CO["V12"]
    tema(s, "7", "Errores y límites: códigos casi iguales, información insuficiente y preguntas sin respuesta",
         "<b>Un puntaje alto no garantiza una respuesta correcta.</b> Tres situaciones lo muestran: dos fichas casi idénticas que difieren en un código, una ficha relacionada pero que no contiene el dato, y preguntas que ninguna ficha responde.",
         [("Códigos exactos", f"D11 (proyecto A17, 12 de noviembre) y D12 (A71, 19 de noviembre) comparten casi todas las palabras: solo cambian «a17» frente a «a71» y la fecha. Confundirlas entrega una fecha equivocada (diapositiva 40). En V11 y V12 ambas recetas ponen la ficha correcta primero."),
          ("Relacionada pero insuficiente", "V14 «¿Cuál es el precio total en pesos?»: ambas recetas ponen D05 primero porque habla de pagos y cuotas, pero no informa el valor de las cuotas ni el total (diapositiva 41). Un ranking siempre tiene un primer lugar."),
          ("Corte de aceptación", "Se acepta una consulta si su coseno máximo alcanza un corte c. Con c = 0.3 se aceptan 9 de 12 respondibles y 0 de 4 sin respuesta; con c = 0.5, 3 de 12 y 0 de 4 (diapositiva 42). Con 0.7 y 0.9 no se acepta ninguna (programa). Cada corte rechaza consultas que sí tenían respuesta."),
          ("Aceptar, recuperar y responder", "Son tres etapas con criterios distintos: <b>aceptar</b> (el puntaje alcanza el corte), <b>recuperar</b> (la ficha pertinente está entre las k primeras) y <b>responder</b> (la respuesta usa fielmente información suficiente). Encontrar un vecino no es contestar bien (diapositiva 43).")],
         [("c47_codigos.png", "Diapositiva 40 con nuestro cálculo: las dos fichas casi idénticas y los cosenos de V11 y V12 con la ficha correcta y con la otra.", 0.9),
          ("c47_umbral.png", "Diapositiva 42 con nuestro cálculo: el coseno máximo de cada consulta (izquierda) y cuántas consultas se aceptan con cada corte (derecha).", 0.92)],
         "Piensa en un bibliotecario que siempre te trae el libro «más cercano» al tema: pediste el <b>precio</b> y te trae el libro de <b>pagos</b>. Es el libro correcto del estante, pero la cifra no está. "
         "Y como un GPS, siempre propone una ruta, aunque la calle que buscas no exista. Por eso hace falta un corte, y mirar qué responde de verdad.",
         ["aceptar si  máxⱼ cos(q, dⱼ) ≥ c", "recuperar si  r ≤ k", "responder: requiere revisar el contenido"],
         [f"V11: ficha correcta con TF-IDF {f(c11['tfidf_rel'])} frente a {f(c11['tfidf_otro'])}; con MiniLM {f(c11['minilm_rel'])} frente a {f(c11['minilm_otro'])}. V12: TF-IDF {f(c12['tfidf_rel'])} y {f(c12['tfidf_otro'])}; MiniLM {f(c12['minilm_rel'])} y {f(c12['minilm_otro'])} (nuestro cálculo)",
          f"El margen de MiniLM en V12 es solo {f(c12['minilm_rel'] - c12['minilm_otro'])}; y el coseno entre D11 y D12 vale {f(CO['cos_D11_D12_minilm'])}, el par de fichas más parecido de las 12 (TF-IDF: {f(CO['cos_D11_D12_tfidf'])})",
          f"V14: coseno máximo {f(MX['minilm'][13])} con D05 (en clase: «0,6… 0,26, algo de ese estilo»); las consultas sin respuesta V13, V14, V15 y V16 llegan a {f(UX['max_sin'])} como máximo; V02, V06 y V10 (con respuesta) solo a {f(MX['minilm'][1])}, {f(MX['minilm'][5])} y {f(MX['minilm'][9])}",
          "Con solo 4 consultas sin respuesta es poco para generalizar (diapositiva 42)"],
         "Los cortes 0.3, 0.5, 0.7 y 0.9 son ilustrativos: el coseno no es una probabilidad. Elegir c implica aceptar el costo de rechazar consultas respondibles frente al riesgo de aceptar información insuficiente.")

    # ---------------------------------------------------------------- 8
    tema(s, "8", "Mirar el espacio con PCA y reutilizar los embeddings",
         "<b>El PCA dibuja las 384 coordenadas en un plano para explorar, pero la búsqueda sigue usando las 384.</b> Y el mismo embedding puede alimentar un clasificador, un agrupamiento o una búsqueda de vecinos.",
         [("PCA de las fichas", f"Resume las 12 fichas centradas en dos ejes: PC1 explica {pc(PCA['prop'][0])} de la variación y PC2 {pc(PCA['prop'][1])}; juntos {pc(PCA['suma'])} (diapositiva 45). Se pierde casi dos tercios de la variación."),
          ("Qué se ve", "D11 y D12 aparecen casi juntas: sus textos son muy parecidos, pero no idénticos (no están en el mismo punto). Las consultas, proyectadas con la misma base, caen cerca de su ficha correcta en varios casos y lejos en otros."),
          ("El ranking no se calcula ahí", "El ranking se calcula en el espacio original de 384 coordenadas; el plano ayuda a explorar, la calidad se juzga con consultas y relevancia (diapositiva 45)."),
          ("Otros usos", "Clasificación (aprender etiquetas desde las coordenadas, como spam), agrupamiento (revisar la coherencia y estabilidad de los grupos) y búsqueda de vecinos. Cambiar la tarea cambia qué resultado hay que evaluar (diapositiva 46)."),
          ("Qué revisar al cambiar la aplicación", "Otro codificador (recalcular fichas y consultas en el mismo espacio), otra división de las fichas, otro dominio de consultas, otro puntaje o corte: volver a medir recuperación y aceptación (diapositiva 47).")],
         [("c47_pca.png", "Diapositiva 45 con nuestro cálculo: las 12 fichas (PCA ajustado solo con ellas) y las 16 consultas de validación proyectadas con los mismos ejes.", 0.9)],
         "El PCA es una <b>foto aérea</b> de una ciudad de 384 calles: ves qué barrios quedan juntos, pero se pierde mucho (dos edificios de distinta altura parecen iguales). "
         "Para buscar la dirección exacta no usas la foto, usas el mapa completo.",
         None,
         [f"PC1 {pc(PCA['prop'][0])}, PC2 {pc(PCA['prop'][1])}, suma {pc(PCA['suma'])} (diapositiva 45; reproducido)",
          f"D11 y D12 están a la derecha del plano y muy cerca: coseno {f(CO['cos_D11_D12_minilm'])} en 384 dimensiones",
          "Resumen de la clase: el embedding convierte algo no matemático (texto) en un vector; después se aplican las mismas redes y métricas que ya conocemos"],
         "Los ejes del PCA no tienen nombre y la cercanía en el dibujo no es un ranking. Con 34 % de variación, el plano omite otras diferencias.", minimo=30)

    # ---------------------------------------------------------------- repaso
    s.append(CondPageBreak(110 * mm))
    s.append(band("Repaso, alertas del material y criterio de dominio", "Lo que debes poder explicar sin apuntes", NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Qué conviene saber al leer las diapositivas y la transcripción</b>", "cellb"),
                  bullets(["<b>La transcripción tenía errores de reconocimiento de voz</b> («reenerales» por redes neuronales; «toncodes» y «AutonCoders» por autoencoders; «Empery» por embeddings; «Mini LEM», «MIMLM» y «MiniLemme» por MiniLM; «TF y DF» por TF-IDF; "
                           "«pichas» por fichas; «octogonales» por ortogonales; «Ábula Libre» por Aula Abierta; «PSA» por PCA). Se corrigieron en 91 líneas; el original se conserva en <font name='Mono'>VideosClases/originales_sin_corregir/</font>.",
                           "<b>Dos cifras dichas en voz alta no coincidían con las diapositivas:</b> «el rango de V04 era 3» (es 5, diapositivas 36 y 39) y «con 3 candidatos pasé de 11 a 12» (es de 10 a 11, diapositiva 44). En la transcripción están corregidas.",
                           f"<b>El «0,6, si mal recuerdo, 0,26»</b> para el coseno de V14 con D05: el valor es {f(MX['minilm'][13])}.",
                           "<b>La grabación empieza unos 5 minutos tarde y tiene un recreo</b> (1:10:26 a 1:28:13): lo que falta se cubre con las diapositivas.",
                           "<b>«Las 16 consultas de evaluación»</b> que menciona el profesor son las 16 de <b>validación</b>; las 16 de prueba se reservan y aquí no se evalúan.",
                           "<b>MRR</b> se dice «Mean Reciprocal Rank» (en la grabación sale «Grant»). <b>R</b> y Python dan lo mismo en esta clase; las cifras de las diapositivas son las de Python."], "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 5))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    pregs = [("q = (1, 1) y un texto repetido cuatro veces b = (4, 4): ¿qué producto interno y qué coseno tienen? ¿Y con a = (2, 0)?", f"qᵀb = 8 y coseno 1; qᵀa = 2 y coseno {f(G['cos_a'], 4)}. El coseno quita el efecto del largo."),
             ("¿Por qué «clave» y «contraseña» son perpendiculares en un conteo?", "Cada palabra activa una columna distinta: (1, 0, 0) y (0, 1, 0). El producto interno es 0 y el coseno 0."),
             ("Si «a17» está en 1 de las 12 fichas, ¿cuál es su IDF?", f"ln(13/2) + 1 = {f(IDF['a17']['idf'], 4)}."),
             ("Con vectores normalizados, ¿cómo se obtiene el coseno?", "Sumando los productos de las coordenadas correspondientes (384 en MiniLM)."),
             ("¿Qué forma tiene S = Q Dᵀ y qué representa cada casilla?", "16 × 12: el coseno de una consulta con una ficha."),
             ("Rangos de 3 consultas: 1, 2 y 5. ¿Hit@3 y MRR?", f"Hit@3 = 2/3 ≈ {f(T3['hit3'] / 3)}; MRR = (1 + 0.5 + 0.2)/3 ≈ {f(T3['mrr'])}."),
             ("¿Qué métrica no cambia al subir k y por qué?", "El MRR: usa el ranking completo (el rango r), no solo las k primeras."),
             (f"¿Por qué «precio total en pesos» recupera D05 y aun así no se responde?", "D05 habla de pagos y cuotas, pero no informa el monto; recuperar no es responder."),
             ("¿Qué porcentaje de la variación muestra el plano PCA y por qué importa?", f"{pc(PCA['suma'])}: el dibujo omite casi dos tercios; el ranking usa las 384 coordenadas.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(pregs, 1)], [7 * mm, 88 * mm, CONTENT_W - 95 * mm]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Criterio de dominio</b>", "cellb"), Spacer(1, 3),
                  bullets(["Convierto textos cortos en vectores de conteo y explico por qué los sinónimos quedan perpendiculares",
                           "Calculo a mano un IDF, un producto interno, una norma y un coseno",
                           "Explico por qué el coseno compara direcciones y la distancia no",
                           "Ordeno 12 fichas por puntaje, encuentro el rango r y calculo Hit@k y MRR",
                           "Elijo k y un corte de aceptación explicando el costo de cada uno",
                           "Explico por qué recuperar una ficha no es responder, y por qué dos códigos casi iguales son un riesgo"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    s.append(Spacer(1, 5))
    s.append(P("<b>Cifras de las diapositivas frente a nuestra reproducción (programa del profesor)</b>", "body"))
    s.append(Spacer(1, 2))
    filas = [["Cifra", "Diapositivas", "Reproducido", "¿Igual?"]] + [[x["item"], x["diapositiva"], x["reproducido"], "sí" if x["igual"] else "NO"] for x in N["comparacion"]]
    s.append(table(filas, [66 * mm, 46 * mm, 50 * mm, CONTENT_W - 162 * mm]))
    s.append(Spacer(1, 3))
    s.append(P("No reproducible con el programa: " + "; ".join(N["no_reproducible"]) + ".", "small"))
    build("C47_Guia_Embeddings.pdf", s, "Clase 47 · Embeddings y representaciones vectoriales")


# ======================================================================= APUNTES
def h(num, title):
    return P(f'<font color="#0E7490">{num}</font>  {title}', "h2")


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 47 · Embeddings y representaciones vectoriales", "Para leer antes de los ejercicios · pensado para quien sabe Python pero nunca ha visto un embedding", C47))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 8 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Texto → vector", "un computador compara números: hay que convertir cada texto en un vector del mismo largo."),
                            ("Conteo", "cuenta palabras; sirve para coincidencias exactas; los sinónimos quedan perpendiculares."),
                            ("TF-IDF", "conteo × peso de rareza, IDF = ln((1+N)/(1+DF)) + 1."),
                            ("Embedding", "vector aprendido (MiniLM: 384 números) que deja cerca los textos de significado parecido."),
                            ("Coseno", "compara direcciones: qᵀz/(‖q‖‖z‖); con vectores normalizados es solo la suma de productos."),
                            ("Ranking", "se ordenan las 12 fichas por puntaje; el rango r es la posición de la correcta."),
                            ("Hit@k y MRR", "Hit@k: ¿está entre las k primeras?; MRR: promedio de 1/r."),
                            ("Límites", "códigos casi iguales, información insuficiente y consultas sin respuesta: recuperar no es responder.")], C47)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación en palabras simples, un <b>ejemplo resuelto con números distintos</b> a los de las diapositivas y los ejercicios, y un resumen «para recordar». "
               "Ejercicios en <font name='Mono'>C47_Ejercicios_Cuaderno_Embeddings.pdf</font> y notebook <font name='Mono'>C47_Embeddings.ipynb</font>.", "body"))
    s.append(Spacer(1, 3))

    # 1 conteo
    vocab = ["clave", "cuota", "pagar", "recuperar"]
    textos = ["pagar cuota", "pagar clave", "recuperar clave clave"]
    Mx = np.array([[t.split().count(w) for w in vocab] for t in textos], float)

    def cs(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    s.append(h("1", "Convertir un texto en un vector: contar palabras"))
    s += para("Un computador solo calcula con números. Para comparar dos textos, lo primero es <b>darle a cada texto una lista de números del mismo largo</b>. "
              "La receta más simple: elegir un <b>vocabulario</b> (por ejemplo, las palabras que aparecen en las fichas) y, para cada texto, contar cuántas veces sale cada palabra. "
              "Cada texto queda como una fila de una tabla; cada columna es una palabra. El orden de las palabras se pierde (por eso se llama «bolsa de palabras»). "
              "Y como todas las filas comparten las columnas, se pueden multiplicar, restar y comparar.")
    s.append(ejemplo("tres textos y cuatro palabras (ejemplo propio)", [
        pasos([["Texto", "clave", "cuota", "pagar", "recuperar"]] +
              [[f"«{t}»"] + [f"{int(x)}" for x in Mx[i]] for i, t in enumerate(textos)] +
              [["Producto interno de los textos 1 y 2", "", "", "", f"{int(Mx[0] @ Mx[1])}"],
               ["Producto interno de los textos 2 y 3", "", "", "", f"{int(Mx[1] @ Mx[2])}"],
               ["Producto interno de los textos 1 y 3", "", "", "", f"{int(Mx[0] @ Mx[2])}"]],
              [66 * mm, 24 * mm, 24 * mm, 24 * mm, CONTENT_W - 138 * mm - 18]),
        P(f"Lectura: los textos 1 y 2 comparten «pagar»; los textos 2 y 3, «clave»; los textos 1 y 3 no comparten ninguna palabra y su producto es 0. "
          f"El coseno confirma lo mismo sin el efecto del largo: textos 1 y 2: {cs(Mx[0], Mx[1]):.4f}; textos 2 y 3: {cs(Mx[1], Mx[2]):.4f}; textos 1 y 3: {cs(Mx[0], Mx[2]):.4f}. "
          "<b>Error común</b>: creer que el vector «entiende» el texto; solo cuenta coincidencias de palabras exactas. Si el texto 3 dijera «contraseña» en lugar de «clave», no se parecería al texto 2.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c47_conteo.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Vocabulario fijo → todos los textos tienen el mismo largo", "Conteo = coincidencias exactas; sin orden y sin significado", "Sinónimos distintos = coordenadas distintas = coseno 0"]))

    # 2 tf-idf
    docs3 = ["recuperar clave", "pagar cuota", "pagar clave"]
    q3 = "recuperar clave"
    voc3 = ["clave", "cuota", "pagar", "recuperar"]
    C3 = np.array([[t.split().count(w) for w in voc3] for t in docs3], float)
    df3 = (C3 > 0).sum(0)
    idf3 = np.log((1 + 3) / (1 + df3)) + 1
    W3 = C3 * idf3
    Wn = W3 / np.linalg.norm(W3, axis=1, keepdims=True)
    qv = np.array([q3.split().count(w) for w in voc3], float) * idf3
    qn = qv / np.linalg.norm(qv)
    sc3 = Wn @ qn
    s.append(h("2", "TF-IDF: darle más peso a lo raro, paso a paso"))
    s += para("Contar da el mismo valor a «de» que a «A17», aunque «de» aparece en casi todas las fichas y «A17» solo en una. "
              "<b>TF-IDF</b> corrige eso en tres pasos: (1) contar (TF); (2) calcular para cada palabra su <b>IDF = ln((1 + N)/(1 + DF)) + 1</b>, donde N es la cantidad de fichas y DF en cuántas aparece la palabra; "
              "(3) multiplicar y <b>normalizar</b> (dividir por la norma) para que cada vector mida 1. Después se compara con el coseno. Las palabras de la consulta que no están en el vocabulario se ignoran.")
    s.append(ejemplo("3 fichas y una consulta (ejemplo propio): «recuperar clave»", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["DF de cada palabra", "clave está en 2 fichas; cuota en 1; pagar en 2; recuperar en 1", f"DF = {', '.join(str(int(x)) for x in df3)}"],
               ["IDF (N = 3)", "ln(4/(1 + DF)) + 1: ln(4/3) + 1; ln(4/2) + 1", f"clave {idf3[0]:.4f}; cuota {idf3[1]:.4f}; pagar {idf3[2]:.4f}; recuperar {idf3[3]:.4f}"],
               ["Ficha 1 «recuperar clave»", "1·1.2877 y 1·1.6931; norma √(1.2877² + 1.6931²)", f"norma {np.linalg.norm(W3[0]):.4f}; normalizado ({Wn[0][0]:.4f}; 0; 0; {Wn[0][3]:.4f})"],
               ["Ficha 3 «pagar clave»", "1·1.2877 (clave) y 1·1.2877 (pagar); norma 1.2877·√2", f"norma {np.linalg.norm(W3[2]):.4f}; normalizado ({Wn[2][0]:.4f}; 0; {Wn[2][2]:.4f}; 0)"],
               ["Coseno de la consulta con cada ficha", "producto de los vectores normalizados", f"ficha 1: {sc3[0]:.4f}; ficha 2: {sc3[1]:.4f}; ficha 3: {sc3[2]:.4f}"]],
              [38 * mm, 76 * mm, CONTENT_W - 114 * mm - 18]),
        P(f"Lectura: la ficha 1 es idéntica a la consulta (coseno 1); la ficha 3 comparte solo «clave» ({sc3[2]:.4f}); la ficha 2 no comparte nada (0). "
          "Fíjate que «recuperar» (IDF 1.6931) pesa más que «clave» (1.2877) porque es más rara. <b>Error común</b>: olvidar que DF cuenta fichas, no repeticiones (si «clave» aparece 3 veces en la misma ficha, DF sigue valiendo 1 por esa ficha).", "cell")]))
    s.append(Spacer(1, 3))
    wq = V11["comunes"]
    s += para(f"En la clase (con las 12 fichas, N = 12): «a17» está en 1 ficha (IDF {f(IDF['a17']['idf'], 4)}), «proyecto» en 2 ({f(IDF['proyecto']['idf'], 4)}), «el» en 7 ({f(IDF['el']['idf'], 4)}) y «de» en 11 ({f(IDF['de']['idf'], 4)}). "
              f"La consulta V11 «¿Cuándo vence el proyecto A17?» comparte con D11 tres palabras: «a17» ({f(wq[0]['norm_q'])} × {f(wq[0]['norm_d'])} = {f(wq[0]['prod'])}), «el» ({f(wq[1]['norm_q'])} × {f(wq[1]['norm_d'])} = {f(wq[1]['prod'])}) y «proyecto» "
              f"({f(wq[2]['norm_q'])} × {f(wq[2]['norm_d'])} = {f(wq[2]['prod'])}). La suma es {f(V11['suma'])}: es el puntaje TF-IDF de V11 con D11 (nuestro cálculo con el programa del profesor).")
    s.append(img("c47_idf.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["TF-IDF = TF × IDF, luego normalizar", "IDF alto = palabra en pocas fichas; IDF cerca de 1 = palabra en casi todas", "Solo ve palabras exactas: no entiende sinónimos"]))

    # 3 embedding
    e = {"clave": np.array([0.9, 0.3]), "contraseña": np.array([0.8, 0.4]), "video": np.array([0.1, 0.95])}
    s.append(h("3", "Embedding: un vector aprendido donde los sinónimos quedan cerca"))
    s += para("Un <b>embedding</b> es un vector que una red neuronal ya entrenada entrega para un texto. La red se entrenó antes (con muchos textos externos y parejas marcadas como «paráfrasis» o «relevante») "
              "para que textos con el mismo sentido terminen en puntos cercanos. En la clase se usa <b>MiniLM</b>, que entrega <b>384 números</b> por texto. "
              "Hoy solo se hace <b>inferencia</b>: los pesos de la red no cambian, y los vectores de las fichas y de las consultas ya vienen calculados en <font name='Mono'>vectores.csv</font>. "
              "Para entender la idea sin 384 dimensiones, mira un caso de 2 números con valores inventados.")
    s.append(ejemplo("embedding ilustrativo de 2 coordenadas (valores inventados, no son de MiniLM)", [
        pasos([["Texto", "Vector (e₁, e₂)", "Coseno con «clave»"],
               ["«clave»", f"({e['clave'][0]}; {e['clave'][1]})", f"{cs(e['clave'], e['clave']):.4f}"],
               ["«contraseña»", f"({e['contraseña'][0]}; {e['contraseña'][1]})", f"{cs(e['clave'], e['contraseña']):.4f}"],
               ["«video»", f"({e['video'][0]}; {e['video'][1]})", f"{cs(e['clave'], e['video']):.4f}"]],
              [48 * mm, 54 * mm, CONTENT_W - 102 * mm - 18]),
        P("Lectura: con conteos, «clave» y «contraseña» tenían coseno 0; con un embedding bien entrenado quedan cerca (coseno alto) y «video», que habla de otra cosa, queda lejos. "
          "Las dos coordenadas no significan «clave» ni «video»: son números que la red aprendió. <b>Error común</b>: pensar que MiniLM se reentrena con las fichas de Aula Abierta; no se reentrena, se usa tal cual. "
          "Y no se pueden comparar vectores de dos codificadores distintos, aunque tengan el mismo largo.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(P("<b>Destilación.</b> Para no entrenar desde cero, una red «estudiante» aprende a imitar los vectores de un «maestro» ya entrenado. Si el maestro entrega (0.2, 0.8) y el estudiante parte en (0.4, 0.6), "
               "el error medio cuadrático es ((0.4 − 0.2)² + (0.6 − 0.8)²)/2 = 0.04; la retropropagación ajusta los pesos del estudiante para reducirlo (diapositiva 11). MiniLM es una red pequeña que imitó a una mayor.", "body"))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Embedding: z = f_θ(x), 384 números por texto en MiniLM", "El aprendizaje fue previo; hoy los pesos no cambian", "Consulta y fichas deben usar el mismo codificador"]))

    # 4 coseno a mano
    qa = np.array([3.0, 4.0])
    cand = {"a = (6, 8)": np.array([6.0, 8.0]), "b = (4, 3)": np.array([4.0, 3.0]), "c = (−4, 3)": np.array([-4.0, 3.0]), "d = (−3, −4)": np.array([-3.0, -4.0])}
    s.append(h("4", "Norma, producto interno y coseno, a mano"))
    s += para("Tres operaciones con vectores. <b>Producto interno</b>: multiplicar coordenada a coordenada y sumar. <b>Norma</b>: la raíz de la suma de los cuadrados (la longitud). "
              "<b>Coseno</b>: el producto interno dividido por el producto de las normas; vale 1 si apuntan igual, 0 si son perpendiculares y −1 si son opuestos. "
              "Por eso el coseno <b>no cambia si multiplicas un vector por un número positivo</b>: solo mira la dirección.")
    rows = [["Vector", "q·v", "‖v‖", "Coseno", "Distancia a q"]]
    for nm, v in cand.items():
        rows.append([nm, f"{qa @ v:.0f}".replace("-", "−"), f"{np.linalg.norm(v):.0f}", f"{cs(qa, v):.2f}".replace("-", "−"), f"{np.linalg.norm(qa - v):.3f}"])
    s.append(ejemplo("q = (3, 4), con ‖q‖ = √(9 + 16) = 5 (ejemplo propio)", [
        pasos(rows, [44 * mm, 24 * mm, 24 * mm, 30 * mm, CONTENT_W - 122 * mm - 18]),
        P("Lectura: a = 2q tiene coseno 1 aunque está a distancia 5 (lejos); b = (4, 3) está a solo 1.414 y su coseno es 24/25 = 0.96; c es perpendicular (3·(−4) + 4·3 = 0); d apunta al lado contrario (coseno −1). "
          "La distancia y el coseno responden preguntas distintas. <b>Error común</b>: creer que mayor producto interno significa más parecido: a tiene producto 50 y b solo 24, pero buena parte de ese 50 viene del tamaño de a (es 2q); "
          "el producto mezcla dirección con tamaño y el coseno las separa.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c47_geometria.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Producto interno = Σ qⱼ zⱼ; norma = √(Σ zⱼ²)", "Coseno = producto interno / (norma · norma); no depende del largo", "Un coseno de 0.8 no es 80 % de probabilidad"]))

    # 5 ranking
    row = {"F1": 0.12, "F2": 0.41, "F3": 0.07, "F4": 0.41, "F5": -0.05, "F6": 0.33}
    orden = sorted(row, key=lambda k: (-row[k], k))
    s.append(h("5", "De los puntajes a un ranking y al rango de la ficha correcta"))
    s += para("Para cada consulta hay <b>un puntaje por ficha</b> (una fila de la matriz S = Q Dᵀ). Se ordenan de mayor a menor; si dos puntajes empatan, el programa del profesor desempata por el identificador de la ficha. "
              "El <b>rango r</b> es la posición que ocupa la ficha correcta. Una fila de ceros (consulta sin ninguna palabra en el vocabulario) no entrega ranking: el programa le asigna aporte 0. "
              "Siempre hay una ficha en el primer lugar, aunque ninguna responda la pregunta.")
    s.append(ejemplo("una consulta contra 6 fichas, relevante F4 (ejemplo propio)", [
        pasos([["Posición", "Ficha", "Puntaje", "Comentario"]] +
              [[str(i + 1), k, f"{row[k]:.2f}".replace("-", "−"), "relevante" if k == "F4" else ("empate con F4: F2 va antes por identificador" if k == "F2" else "")] for i, k in enumerate(orden)],
              [22 * mm, 22 * mm, 26 * mm, CONTENT_W - 70 * mm - 18]),
        P(f"Lectura: F2 y F4 empatan en 0.41; por identificador F2 queda primera y F4 segunda: r = {orden.index('F4') + 1}, rango recíproco 1/r = {1 / (orden.index('F4') + 1):.2f}. "
          "Con k = 1 la consulta no sería un acierto; con k = 2 sí. <b>Error común</b>: contar mal la posición cuando hay negativos: −0.05 es menor que 0.07; los puntajes negativos van al final.", "cell")]))
    s.append(Spacer(1, 3))
    s += para(f"En la clase: V02 «No pude asistir, ¿puedo ver el video después?» recibe de MiniLM este orden: D06 ({f(N['v02']['scores'][0])}), D02 ({f(N['v02']['scores'][1])}), D01 ({f(N['v02']['scores'][2])}); "
              "la ficha correcta, D02, ocupa el lugar 2 (diapositiva 35).")
    s.append(img("c47_ranking_v01.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Ranking = ordenar la fila de puntajes de mayor a menor", "r = posición de la ficha relevante; empates por identificador", "El primer lugar existe siempre: no prueba que responda"]))

    # 6 hit@k y MRR
    rA = [1, 3, 2, 1, 6, 1, 4, 2]
    hit = lambda rs, k: sum(1 for r_ in rs if 0 < r_ <= k)
    mrr = lambda rs: float(np.mean([1 / r_ for r_ in rs]))
    s.append(h("6", "Hit@k y MRR: cómo medir el buscador y elegir k"))
    s += para("Con las consultas de validación, cuya ficha correcta se conoce, se calcula el rango r de cada una. <b>Hit@k</b> es la proporción de consultas con r ≤ k (¿estaba entre las k primeras?). "
              "<b>MRR</b> es el promedio de 1/r: da 1 al primer lugar, 0.5 al segundo, 0.333 al tercero… y por eso distingue un primer lugar de un segundo, cosa que Hit@k no hace. "
              "Como el MRR usa el ranking completo, <b>cambiar k no lo modifica</b>. Se calculan solo con las consultas que tienen respuesta.")
    s.append(ejemplo("8 consultas con rangos 1, 3, 2, 1, 6, 1, 4, 2 (ejemplo propio)", [
        pasos([["Métrica", "Cálculo", "Resultado"],
               ["Hit@1", "rangos ≤ 1: 1, 1, 1 → 3 de 8", f"{hit(rA, 1)}/8 = {hit(rA, 1) / 8:.3f}"],
               ["Hit@3", "rangos ≤ 3: 1, 3, 2, 1, 1, 2 → 6 de 8", f"{hit(rA, 3)}/8 = {hit(rA, 3) / 8:.3f}"],
               ["Hit@5", "rangos ≤ 5: todos menos el 6 → 7 de 8", f"{hit(rA, 5)}/8 = {hit(rA, 5) / 8:.3f}"],
               ["MRR", "(1 + 1/3 + 1/2 + 1 + 1/6 + 1 + 1/4 + 1/2) ÷ 8", f"{sum(1 / r_ for r_ in rA):.4f} ÷ 8 = {mrr(rA):.4f}"]],
              [24 * mm, 84 * mm, CONTENT_W - 108 * mm - 18]),
        P("Lectura: pasar de k = 1 a k = 3 duplica los aciertos, pero 6 y 4 siguen fuera hasta que k llegue a 4 y 6. Elegir k es un equilibrio: más candidatos significan más lectura para quien usa el buscador. "
          "<b>Error común</b>: promediar los rangos (1 + 3 + 2 + … ÷ 8) en lugar de promediar 1/r; el MRR promedia los recíprocos.", "cell")]))
    s.append(Spacer(1, 3))
    s += para(f"En la clase (12 consultas respondibles): Hit@1 = {h1_t_}/12 para TF-IDF y {h1_m_}/12 para MiniLM; Hit@3 = {h3_t_}/12 y {h3_m_}/12; MRR = {f(H['mrr_tfidf'])} y {f(H['mrr_minilm'])}. "
              f"MiniLM alcanza 12/12 con k = {H['k_cobertura_total_minilm']} y TF-IDF con k = {H['k_cobertura_total_tfidf']} (nuestro cálculo).")
    s.append(img("c47_hitk.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Hit@k = proporción con r ≤ k; MRR = promedio de 1/r", "MRR no depende de k; Hit@k sí", "Se evalúan solo las consultas con respuesta, y k se elige con validación"]))

    # 7 sin respuesta y cortes
    mxs = [("Q1", 0.62, "r = 1"), ("Q2", 0.48, "r = 2"), ("Q3", 0.35, "r = 4"), ("Q4", 0.22, "r = 1"), ("Q5", 0.41, "sin respuesta"), ("Q6", 0.30, "sin respuesta"), ("Q7", 0.18, "sin respuesta")]
    s.append(h("7", "Consultas sin respuesta, cortes de aceptación y límites"))
    s += para("Una consulta puede no tener ninguna ficha que la responda (por ejemplo «¿Hay estacionamiento gratuito?»). El ranking igual entrega un primer lugar. "
              "Una solución es aceptar una consulta solo si su <b>coseno máximo</b> alcanza un <b>corte</b> c. Tiene un costo: un corte alto rechaza también consultas que sí tenían respuesta; uno bajo acepta consultas sin respuesta. "
              "Es la misma lógica de un umbral de clasificación (Clases 38 y 41). Y aun aceptada y bien recuperada, la ficha puede ser <b>insuficiente</b>: recuperar no es responder.")
    s.append(ejemplo("7 consultas con su coseno máximo (ejemplo propio)", [
        pasos([["Consulta", "Coseno máximo", "Situación", "¿Acepta c = 0.40?", "¿Acepta c = 0.30?"]] +
              [[q_, f"{m:.2f}", sit, "sí" if m >= 0.40 else "no", "sí" if m >= 0.30 else "no"] for q_, m, sit in mxs],
              [20 * mm, 32 * mm, 40 * mm, 32 * mm, CONTENT_W - 124 * mm - 18]),
        P("Lectura: con c = 0.40 se aceptan Q1, Q2 (con respuesta) y Q5 (sin respuesta): 2 de 4 respondibles y 1 de 3 sin respuesta. Bajando a c = 0.30 se acepta además Q3 y Q6 (≥ 0.30): "
          "3 de 4 respondibles, pero 2 de 3 sin respuesta. Ningún corte separa perfecto aquí porque Q4 (con respuesta) tiene 0.22 y Q5 (sin respuesta) 0.41. "
          "<b>Error común</b>: leer el coseno como probabilidad (0.62 no es «62 % seguro»).", "cell")]))
    s.append(Spacer(1, 3))
    s += para(f"En la clase: con c = 0.3 se aceptan 9 de 12 consultas respondibles y 0 de 4 sin respuesta; con c = 0.5, 3 y 0 (diapositiva 42). Las consultas sin respuesta llegan a un coseno máximo de {f(UX['max_sin'])}; "
              f"V02 solo a {f(UX['min_resp'])}. Los cortes son ilustrativos y 4 consultas sin respuesta son pocas para generalizar. "
              f"<b>Códigos exactos</b>: D11 (A17) y D12 (A71) son casi iguales; con MiniLM su coseno es {f(CO['cos_D11_D12_minilm'])}. En V11 y V12 la ficha correcta quedó primera, pero el margen de V12 fue de solo "
              f"{f(CO['V12']['minilm_rel'] - CO['V12']['minilm_otro'])}: conviene revisar el código exacto antes de entregar una fecha.")
    s.append(img("c47_umbral.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Aceptar (corte), recuperar (r ≤ k) y responder (contenido suficiente) son tres criterios distintos", "Un corte alto pierde consultas respondibles; uno bajo acepta consultas sin respuesta", "Los códigos exactos (A17/A71) exigen revisar el dato, no solo el coseno"]))

    # 8 PCA
    s.append(h("8", "El plano PCA y para qué más sirven los embeddings"))
    s += para(f"No se pueden dibujar 384 coordenadas. El <b>PCA</b> las resume en dos ejes: en las fichas, PC1 explica {pc(PCA['prop'][0])} y PC2 {pc(PCA['prop'][1])} de la variación, juntos {pc(PCA['suma'])}. "
              "El dibujo sirve para explorar (D11 y D12 aparecen muy cerca), pero omite casi dos tercios de la variación y <b>no se usa para ordenar</b>: el ranking usa las 384 coordenadas. "
              "El mismo embedding puede ser la entrada de un clasificador (por ejemplo, spam o no spam con sus etiquetas), de un agrupamiento o de una búsqueda de vecinos: cada tarea se evalúa con su propio criterio.")
    s.append(img("c47_pca.png", width=CONTENT_W * 0.86))
    s.append(Spacer(1, 2))
    s.append(recuerda(["PCA solo dibuja: no calcula el ranking", "Cambiar de codificador exige recalcular fichas y consultas en el mismo espacio", "Cambiar la tarea cambia qué hay que evaluar"]))
    build("apuntes/C47_Apuntes_Embeddings.pdf", s, "Apuntes · Clase 47 · Embeddings")


# ======================================================================= EJERCICIOS
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#0E7490">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C47))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 47 · Embeddings", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C47))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: TF-IDF = TF × IDF · IDF = ln((1 + N)/(1 + DF)) + 1 · qᵀz = Σ qⱼ zⱼ · ‖z‖ = √(Σ zⱼ²) · cos(q, z) = qᵀz/(‖q‖‖z‖) · Hit@k = (1/n) Σ 1[rᵢ ≤ k] · MRR = (1/n) Σ 1/rᵢ.",
                           "Valores útiles: ln 2 = 0.6931 · ln 3 = 1.0986 · ln 1.5 = 0.4055 · √2 = 1.4142 · √3 = 1.7321 · √5 = 2.2361.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C47)], bg=SOFT, border=C47, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "Vectores de conteo", "★", "Vocabulario: (clase, grabación, ver, video). Textos: a = «ver video», b = «ver grabación», c = «ver clase clase».", [
        "Escribe el vector de conteo de a, b y c.",
        "Calcula el producto interno de a con b y de a con c, y el coseno de cada pareja.",
        "«video» y «grabación» son sinónimos para esta ficha. ¿Qué coseno tienen los vectores (0, 0, 0, 1) y (0, 1, 0, 0)? ¿Qué dice eso de los conteos?"])
    s += bloque(2, "TF-IDF de una palabra", "★★", "Hay N = 5 fichas. «plazo» aparece en 1 ficha, «proyecto» en 3 y «de» en las 5.", [
        "Calcula el IDF de las tres palabras con IDF = ln((1 + N)/(1 + DF)) + 1.",
        "El texto «proyecto de plazo plazo» tiene TF: proyecto 1, de 1, plazo 2. Calcula los tres pesos TF-IDF sin normalizar.",
        "Calcula la norma del vector de pesos y los pesos normalizados. ¿Qué palabra domina y por qué?"])
    s += bloque(3, "Producto interno, norma y coseno", "★★", "q = (1, 2, 2), a = (2, 1, 2), b = (2, 4, 4), c = (2, −2, 1).", [
        "Calcula ‖q‖, ‖a‖, ‖b‖ y ‖c‖.",
        "Calcula qᵀa, qᵀb, qᵀc y los tres cosenos.",
        "¿Cuál está más cerca de q por distancia euclidiana (a o b)? ¿Y por coseno? Explica la diferencia."])
    s += bloque(4, "Ranking y rango", "★★", "Una consulta tiene estos puntajes con 6 fichas: F1 0.10, F2 0.36, F3 −0.04, F4 0.36, F5 0.22, F6 0.05. La ficha relevante es F5.", [
        "Ordena las fichas de mayor a menor puntaje (los empates se resuelven por el nombre de la ficha).",
        "¿Cuál es el rango r de F5? ¿Cuánto vale 1/r?",
        "¿Es un acierto con k = 1? ¿Y con k = 3?"])
    s += bloque(5, "Hit@k y MRR", "★★", "Rangos de la ficha correcta en 8 consultas. Receta A: 1, 1, 3, 2, 1, 5, 1, 2. Receta B: 2, 1, 1, 1, 4, 1, 1, 1.", [
        "Calcula Hit@1, Hit@3 y Hit@5 de cada receta.",
        "Calcula el MRR de cada receta.",
        "¿Cuál receta prefieres si solo puedes mostrar 1 ficha? ¿Y si puedes mostrar 3? ¿Cambia el MRR con k?"])
    s += bloque(6, "Elegir k y un corte", "★★", "Ocho consultas, cada una con su coseno máximo y el rango de su ficha correcta (o «sin»): P1 0.62 (r = 1); P2 0.48 (r = 2); P3 0.35 (r = 4); P4 0.22 (r = 1); P5 0.55 (r = 1); P6 0.41 (sin); P7 0.30 (sin); P8 0.18 (sin).", [
        "Con k = 2, ¿cuántas de las 5 consultas con respuesta son aciertos? ¿Y con k = 4?",
        "Con el corte c = 0.40 (se acepta si el coseno máximo es ≥ c), ¿cuántas consultas con respuesta y cuántas sin respuesta se aceptan? Repite con c = 0.30.",
        "¿Qué corte elegirías si lo peor es entregar una ficha cuando no hay respuesta? ¿Y si lo peor es rechazar una consulta con respuesta? Justifica."])
    s += bloque(7, "El código exacto", "★★★", "La consulta «¿Cuándo vence el proyecto B28?» tiene coseno 0.58 con la ficha B28 y 0.55 con la ficha B82 (casi la misma frase con otro código); con las demás fichas, 0.20 o menos.", [
        "¿Qué rango tiene la ficha correcta? ¿Cuál es el margen entre B28 y B82?",
        "Si el buscador entregara B82, ¿qué consecuencia tendría para quien pregunta? ¿Qué cosa concreta revisarías antes de entregar la fecha?",
        "¿Por qué un conteo de palabras distingue «b28» de «b82» y, aun así, un embedding puede dejarlas muy cerca?"])
    s += bloque(8, "Pensar con evidencia", "★★★", None, [
        "«El coseno de V06 con su ficha es 0.262; entonces hay 26 % de probabilidad de que acertemos.» ¿Es correcto?",
        "«MiniLM tuvo Hit@3 de 0.917 y TF-IDF de 0.583; MiniLM es mejor en cualquier problema.» ¿Qué se puede y qué no se puede concluir? (Piensa en el número de consultas.)",
        "«D05 quedó primera para el precio total en pesos, así que el buscador respondió la pregunta.» ¿Por qué no?",
        "Subes k de 1 a 3 y el Hit@k mejora. ¿Cambia el MRR? ¿Por qué?"])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C47))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [KeepTogether([P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C47), Spacer(1, 2)])]

    def cs(a, b):
        a, b = np.array(a, float), np.array(b, float)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # 1
    voc = ["clase", "grabación", "ver", "video"]
    tx = {"a": "ver video", "b": "ver grabación", "c": "ver clase clase"}
    vv = {k: np.array([t.split().count(w) for w in voc], float) for k, t in tx.items()}
    assert list(vv["a"]) == [0, 0, 1, 1] and list(vv["b"]) == [0, 1, 1, 0] and list(vv["c"]) == [2, 0, 1, 0]
    assert vv["a"] @ vv["b"] == 1 and vv["a"] @ vv["c"] == 1
    assert abs(cs(vv["a"], vv["b"]) - 0.5) < 1e-12 and abs(cs(vv["a"], vv["c"]) - 1 / sqrt(10)) < 1e-12
    s += sol("Bloque 1", [
        ["1", "a = (0, 0, 1, 1); b = (0, 1, 1, 0); c = (2, 0, 1, 0)."],
        ["2", f"a·b = 0·0 + 0·1 + 1·1 + 1·0 = 1; ‖a‖ = ‖b‖ = √2, así que cos(a, b) = 1/2 = 0.5000. a·c = 0·2 + 0·0 + 1·1 + 1·0 = 1; ‖c‖ = √5, así que cos(a, c) = 1/(√2·√5) = 1/√10 = {1 / sqrt(10):.4f}."],
        ["3", "(0, 0, 0, 1)·(0, 1, 0, 0) = 0, y el coseno es 0: perpendiculares. Para un conteo, «video» y «grabación» no tienen nada en común aunque signifiquen lo mismo. Lo único que une a a y b es «ver»."]])
    # 2
    idf_ = {w: log(6 / (1 + d)) + 1 for w, d in (("plazo", 1), ("proyecto", 3), ("de", 5))}
    raw = {"proyecto": 1 * idf_["proyecto"], "de": 1 * idf_["de"], "plazo": 2 * idf_["plazo"]}
    nr = sqrt(sum(v ** 2 for v in raw.values()))
    assert abs(idf_["plazo"] - (log(3) + 1)) < 1e-12 and abs(idf_["proyecto"] - (log(1.5) + 1)) < 1e-12 and idf_["de"] == 1.0
    s += sol("Bloque 2", [
        ["1", f"plazo: ln(6/2) + 1 = ln 3 + 1 = {idf_['plazo']:.4f}; proyecto: ln(6/4) + 1 = ln 1.5 + 1 = {idf_['proyecto']:.4f}; de: ln(6/6) + 1 = 0 + 1 = {idf_['de']:.4f}."],
        ["2", f"proyecto: 1 · {idf_['proyecto']:.4f} = {raw['proyecto']:.4f}; de: 1 · 1 = 1.0000; plazo: 2 · {idf_['plazo']:.4f} = {raw['plazo']:.4f}."],
        ["3", f"Norma = √({raw['proyecto']:.4f}² + 1² + {raw['plazo']:.4f}²) = √({raw['proyecto'] ** 2:.4f} + 1 + {raw['plazo'] ** 2:.4f}) = √{nr ** 2:.4f} = {nr:.4f}. Normalizados: proyecto {raw['proyecto'] / nr:.4f}; de {1 / nr:.4f}; plazo {raw['plazo'] / nr:.4f}. "
              "Domina «plazo»: aparece dos veces en el texto (TF = 2) y en una sola ficha (IDF alto). «de» casi no pesa porque está en todas."]])
    # 3
    q, a, b, c = (np.array(x, float) for x in ((1, 2, 2), (2, 1, 2), (2, 4, 4), (2, -2, 1)))
    assert [np.linalg.norm(x) for x in (q, a, b, c)] == [3.0, 3.0, 6.0, 3.0]
    assert (q @ a, q @ b, q @ c) == (8.0, 18.0, 0.0)
    da, db = np.linalg.norm(q - a), np.linalg.norm(q - b)
    s += sol("Bloque 3", [
        ["1", "‖q‖ = √(1 + 4 + 4) = 3; ‖a‖ = √(4 + 1 + 4) = 3; ‖b‖ = √(4 + 16 + 16) = 6; ‖c‖ = √(4 + 4 + 1) = 3."],
        ["2", f"qᵀa = 2 + 2 + 4 = 8; qᵀb = 2 + 8 + 8 = 18; qᵀc = 2 − 4 + 2 = 0. cos(q, a) = 8/9 = {8 / 9:.4f}; cos(q, b) = 18/(3·6) = 1; cos(q, c) = 0/(3·3) = 0."],
        ["3", f"Distancias: ‖q − a‖ = ‖(−1, 1, 0)‖ = √2 = {da:.4f}; ‖q − b‖ = ‖(−1, −2, −2)‖ = 3. Por distancia, a está más cerca. Por coseno, b (coseno 1) gana a a (0.8889), porque b = 2q tiene exactamente la misma dirección: "
              "el coseno retira el efecto del largo, la distancia no."]])
    # 4
    row = {"F1": 0.10, "F2": 0.36, "F3": -0.04, "F4": 0.36, "F5": 0.22, "F6": 0.05}
    orden = sorted(row, key=lambda k: (-row[k], k))
    assert orden == ["F2", "F4", "F5", "F1", "F6", "F3"]
    s += sol("Bloque 4", [
        ["1", "F2 0.36, F4 0.36 (empate: F2 antes que F4), F5 0.22, F1 0.10, F6 0.05, F3 −0.04. Ojo: F1 (0.10) queda antes que F6 (0.05): orden F2, F4, F5, F1, F6, F3."],
        ["2", "F5 ocupa el lugar 3, así que r = 3 y 1/r = 0.3333."],
        ["3", "Con k = 1 no es un acierto (r = 3 &gt; 1). Con k = 3 sí (r = 3 ≤ 3)."]])
    # 5
    A_ = [1, 1, 3, 2, 1, 5, 1, 2]
    B_ = [2, 1, 1, 1, 4, 1, 1, 1]
    hit = lambda rs, k: sum(1 for r_ in rs if r_ <= k)
    mrr = lambda rs: sum(1 / r_ for r_ in rs) / len(rs)
    assert [hit(A_, k) for k in (1, 3, 5)] == [4, 7, 8] and [hit(B_, k) for k in (1, 3, 5)] == [6, 7, 8]
    s += sol("Bloque 5", [
        ["1", f"Receta A: Hit@1 = 4/8 = 0.500; Hit@3 = 7/8 = 0.875 (queda fuera el 5); Hit@5 = 8/8 = 1.000. Receta B: Hit@1 = 6/8 = 0.750; Hit@3 = 7/8 = 0.875 (queda fuera el 4); Hit@5 = 8/8 = 1.000."],
        ["2", f"A: 1 + 1 + 1/3 + 1/2 + 1 + 1/5 + 1 + 1/2 = {sum(1 / r_ for r_ in A_):.4f}; ÷ 8 = {mrr(A_):.4f}. B: 1/2 + 1 + 1 + 1 + 1/4 + 1 + 1 + 1 = {sum(1 / r_ for r_ in B_):.4f}; ÷ 8 = {mrr(B_):.4f}."],
        ["3", "Con k = 1 prefieres B (0.750 frente a 0.500). Con k = 3 empatan (0.875 y 0.875): para desempatar sirve el MRR, que favorece a B porque más consultas quedan en el primer lugar. El MRR no depende de k: se calcula con los rangos completos, que no cambian."]])
    # 6
    P_ = [("P1", 0.62, 1), ("P2", 0.48, 2), ("P3", 0.35, 4), ("P4", 0.22, 1), ("P5", 0.55, 1), ("P6", 0.41, None), ("P7", 0.30, None), ("P8", 0.18, None)]
    resp = [x for x in P_ if x[2]]
    sin = [x for x in P_ if not x[2]]
    k2, k4 = sum(1 for x in resp if x[2] <= 2), sum(1 for x in resp if x[2] <= 4)
    ac = lambda c_: (sum(1 for x in resp if x[1] >= c_), sum(1 for x in sin if x[1] >= c_))
    assert (len(resp), len(sin), k2, k4) == (5, 3, 4, 5) and ac(0.40) == (3, 1) and ac(0.30) == (4, 2)
    s += sol("Bloque 6", [
        ["1", "Con k = 2: P1 (r = 1), P2 (2), P4 (1) y P5 (1) son aciertos; P3 (r = 4) no: 4 de 5. Con k = 4 también entra P3: 5 de 5."],
        ["2", "c = 0.40: se aceptan P1 (0.62), P2 (0.48), P5 (0.55) con respuesta, es decir 3 de 5, y P6 (0.41) sin respuesta, 1 de 3. c = 0.30: se aceptan P1, P2, P3 (0.35), P5 con respuesta, 4 de 5, y P6 y P7 (0.30 ≥ 0.30) sin respuesta, 2 de 3."],
        ["3", "Si lo peor es entregar una ficha cuando no hay respuesta, conviene un corte alto: con c = 0.45 se aceptan solo P1, P2 y P5 (ninguna sin respuesta), a costa de rechazar P3 y P4, que sí tenían respuesta. "
              "Si lo peor es rechazar una consulta con respuesta, conviene un corte bajo: con c = 0.20 se aceptan las 5 con respuesta, pero también P6 y P7 (sin respuesta). Ningún corte separa perfecto: P4 (con respuesta, 0.22) queda bajo P6 (0.41) y P7 (0.30), que no tienen respuesta."]])
    # 7
    s += sol("Bloque 7", [
        ["1", "La ficha B28 tiene el puntaje más alto (0.58): r = 1. El margen con B82 es 0.58 − 0.55 = 0.03, muy pequeño."],
        ["2", "Quien pregunta recibiría la fecha de otro proyecto: un error con consecuencias reales. Antes de entregar convendría comprobar que el código exacto de la consulta (B28) coincide con el de la ficha recuperada, y revisar el margen entre las dos primeras fichas."],
        ["3", "Un conteo trata «b28» y «b82» como dos palabras distintas, así que las separa. Un embedding las convierte en vectores a partir de fragmentos casi iguales y puede dejar muy cerca las dos fichas (como D11 y D12 en la clase, con coseno 0.891). "
              "Una representación útil debe conservar las diferencias decisivas para la tarea."]])
    # 8
    s += sol("Bloque 8", [
        ["1", "No. El coseno es un puntaje geométrico entre −1 y 1; no es una probabilidad. Que 0.262 sea el coseno de V06 con su ficha no significa 26 % de acierto."],
        ["2", "Se puede concluir que, en estas 12 consultas de validación y con estas dos recetas, MiniLM recuperó más fichas entre las 3 primeras (11 de 12 frente a 7 de 12). No que sea mejor en cualquier problema: 12 consultas son pocas (una consulta mueve 1/12 = 0.083), "
              "el caso es construido y hay que revisar los errores y validar con otros datos."],
        ["3", "Porque recuperar no es responder: D05 habla de pagos y cuotas, pero no informa el precio total. Hay que comprobar que la ficha contenga información suficiente."],
        ["4", "No cambia: el MRR usa el rango completo de la ficha correcta (1/r), y el ranking no cambió. Solo Hit@k depende de k."]])
    build("apuntes/C47_Ejercicios_Cuaderno_Embeddings.pdf", s, "Ejercicios para el cuaderno · Clase 47")


h1_t_, h1_m_, h3_t_, h3_m_ = H["tfidf"][0], H["minilm"][0], H["tfidf"][2], H["minilm"][2]

if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
