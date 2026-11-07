"""Clase 46 · Deep Learning, autoencoders y representaciones (Prof. Eloy Alvarado): guía visual, apuntes y cuaderno de ejercicios en papel.
Los números de la guía vienen de c46_numeros.json (lo escribe figuras_c46.py); los de los ejemplos y ejercicios se calculan aquí.
Los números de las diapositivas (corrida del profesor) se citan como «diapositivas». Con el programa del profesor, nuestra reproducción coincide
con todas ellas; lo que se rotula «nuestra reproducción» o «ejemplo propio» es cálculo propio y no es de la clase."""
import json
from math import exp, sqrt, tanh
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import CondPageBreak, KeepTogether, PageBreak, Spacer

from construir_glm import ejemplo, para, pasos, recuerda
from construir_pdf import (AMBER, AMBER_BG, CONTENT_W, GREEN, GREEN_BG, NAVY, OUT, P, RED, RED_BG,
                           SOFT, band, box, build, bullets, img, numbered, table, two_cols)

N = json.loads((Path(__file__).resolve().parent / "c46_numeros.json").read_text(encoding="utf-8"))
C46 = colors.HexColor("#6D28D9")
C46_BG = colors.HexColor("#F5F3FF")
SYM = "<font name='Sym'>☐</font>"
HALF = (CONTENT_W - 6 * mm) * 0.5
M, CU, PX, CO, BI, UM, UT, PA, VE = N["mse"], N["curvas"], N["pixel"], N["codigo8"], N["bits"], N["umbral"], N["utilidad"], N["params"], N["vecinos"]
TY = N["toy"]


def f(x, d=2):
    return f"{x:.{d}f}"


def mil(x):
    return f"{x:,}".replace(",", " ")


def pc(x, d=1):
    return f"{x * 100:.{d}f} %"


def mn(v, d=4):
    """Número con signo menos tipográfico."""
    return f"{v:.{d}f}".replace("-", "−")


# ======================================================================= GUÍA
def tema(s, num, titulo, idea, conceptos, figuras, simple, formulas, salida, cuidado, minimo=50):
    """Un tema: título, idea central, conceptos, figuras (a ancho casi completo), analogía, números de la clase y advertencia."""
    s.append(CondPageBreak(minimo * mm))
    s.append(P(f'<font color="#6D28D9">{num}</font>  {titulo}', "h2"))
    s.append(box([P(idea, "big")], bg=SOFT, border=C46, pad=5))
    s.append(Spacer(1, 3))
    s.append(numbered(conceptos, C46))
    s.append(Spacer(1, 2))
    left = [P("En palabras simples", "h3"), box([P(simple, "cell")], bg=C46_BG, border=C46, width=HALF, pad=5)]
    if formulas:
        left += [Spacer(1, 3), box([P(x, "formula") for x in formulas], bg=SOFT, width=HALF, pad=5)]
    right = [P("Con los números de la clase", "h3"), bullets(salida, "cell", C46)]
    if cuidado:
        right += [Spacer(1, 3), box([P("<b>Cuidado.</b> " + cuidado, "cell")], bg=AMBER_BG, border=AMBER, width=HALF, pad=5)]
    s.append(KeepTogether([two_cols(left, right)]))
    s.append(Spacer(1, 4))
    for fig, pie, *w in figuras:
        s.append(KeepTogether([img(fig, width=CONTENT_W * (w[0] if w else 0.92)), P(pie, "small"), Spacer(1, 3)]))
    s.append(Spacer(1, 4))


def guia():
    s = []
    s.append(band("Clase 46 · Deep Learning, autoencoders y representaciones",
                  "Profesor Eloy Alvarado · la red ya no clasifica: aprende a resumir una imagen en 8 números y a volver a armarla, con los dígitos de las Clases 44 y 45", NAVY))
    s.append(Spacer(1, 5))
    s.append(box([P("La gran idea de la clase", "cellb"), Spacer(1, 3),
                  P("<b>Un autoencoder es una red neuronal que se entrena para reproducir su propia entrada pasando por un resumen pequeño.</b> "
                    "El <b>encoder</b> convierte las 64 intensidades de una imagen en un <b>vector latente</b> de 8 números; el <b>decoder</b> usa solo esos 8 números para volver a estimar las 64 intensidades. "
                    "La referencia con la que se compara es la <b>misma entrada</b>, así que no hacen falta etiquetas. El error se mide con el <b>MSE por píxel</b> y se reparte hacia las dos mitades con la misma retropropagación de la Clase 45. "
                    "Como el resumen es más pequeño que la imagen, <b>se pierde información</b>: lo que importa es qué se conserva y si eso sirve para el uso previsto (buscar parecidos, limpiar ruido, "
                    "priorizar revisiones). Un error de reconstrucción alto puede servir de <b>alerta</b>, pero es un puntaje, no una probabilidad, y exige una validación específica.", "body")],
                 bg=C46_BG, border=C46, pad=6))
    s.append(Spacer(1, 4))
    s.append(img("c46_esquema.png", width=CONTENT_W * 0.97))
    s.append(Spacer(1, 3))
    left = [P("Lo que necesitas ANTES", "h3"),
            bullets(["<b>Red neuronal</b> (Clase 45): neurona, capa, activación, pérdida, gradiente, Adam, lotes y épocas",
                     "<b>PCA</b> (Clase 32): proyectar a pocas direcciones y reconstruir",
                     "<b>Validación y prueba</b> (Clases 30 y 38): elegir con una, evaluar al final con otra",
                     "<b>Detección de anomalías</b> (Clase 41): un puntaje y un umbral"], "cell", C46)]
    right = [P("Ruta de esta guía", "h3"),
             bullets(["Representación y autoencoder → reconstruir frente a clasificar → qué se pierde",
                      "El error de reconstrucción (MSE) a mano → referencias: imagen media y PCA con 8 componentes",
                      "Profundidad, parámetros y entrenamiento → mirar el código y usarlo",
                      "El error como alerta de anomalías y sus límites"], "cell", C46)]
    s.append(two_cols(left, right))
    s.append(Spacer(1, 3))
    s.append(box([P("<b>Cómo leer los números de este material</b>", "cellb"),
                  P("La grabación disponible termina en el recreo (1:14:23): <b>cubre las diapositivas 1 a 14</b>. Las diapositivas 15 a 53 (PCA, entrenamiento, resultados, anomalías, código en R) no están en el video; "
                    "aquí se explican a partir de las diapositivas y del programa del profesor. Las cifras de las diapositivas son la corrida del profesor: se ejecutó <b>su mismo programa</b> "
                    f"(<font name='Mono'>autoencoders.py</font>, scikit-learn 1.9.1, misma semilla) y <b>las {len(N['comparacion'])} cifras comparadas coinciden exactamente</b> (tabla en la última página). "
                    "Lo que se rotula «nuestra reproducción» o «ejemplo propio» es cálculo propio, no de la clase.", "cell")],
                 bg=AMBER_BG, border=AMBER, pad=5))

    # ---------------------------------------------------------------- 1 autoencoder y representación
    h8 = CO["h"]
    tema(s, "1", "Qué es una representación y qué es un autoencoder",
         "<b>Un autoencoder resume la entrada en un vector latente y luego intenta reconstruirla con solo ese vector.</b> Se escribe h = f(x) para el encoder y x̂ = g(h) para el decoder.",
         [("Deep learning", "Redes que componen varias capas de transformación. No hay un número de capas desde el cual una red «es profunda»: más capas permiten más composiciones, pero su utilidad debe evaluarse (diapositiva 2)."),
          ("Representación", "Una forma de describir un objeto con números. La imagen se describe con 64 intensidades; el vector latente la describe con solo 8. Cada capa intermedia de una red aprende una representación."),
          ("Encoder y decoder", "El <b>encoder</b> f calcula h a partir de x; el <b>decoder</b> g calcula x̂ a partir de h. Son dos funciones con parámetros propios (θ para f y ψ para g, que en la Clase 45 se llamaban W₁ y W₂) y se ajustan juntas."),
          ("Vector latente", "Es la salida de la capa central: se llama «latente» porque está <i>dentro</i> de la red. Aquí tiene 8 coordenadas, una decisión que se toma antes de entrenar (un hiperparámetro).")],
         [("c46_codigo.png", "Diapositiva 9 (el esquema de arriba es de las diapositivas 3 y 8): las 8 coordenadas latentes de dos imágenes. Rojo = negativo, azul = positivo; las líneas punteadas marcan los límites −1 y 1 de tanh.")],
         "Imagina que tienes que describir una foto de 64 cuadraditos a un amigo, pero <b>solo puedes dictarle 8 números</b>. Tu amigo (el decoder) dibuja la foto usando solo esos 8 números. "
         "El juego consiste en elegir tan bien los 8 números (el encoder) y la forma de dibujar (el decoder) que el dibujo se parezca lo más posible a la foto original.",
         ["h = f(x)   (64 valores → 8 valores)", "x̂ = g(h)   (8 valores → 64 valores)"],
         [f"El «8» de la diapositiva 9 (id {CO['id']}, de entrenamiento) tiene h = (" + ", ".join(mn(v, 2) for v in h8) + ")",
          f"Las 8 coordenadas suman {mn(CO['suma'], 2)}: no son probabilidades y no tienen por qué sumar 1",
          f"Con tanh, todas quedan entre −1 y 1 (en las 1 797 imágenes, entre {mn(BI['coord_min'], 2)} y {mn(BI['coord_max'], 2)})",
          "Otra imagen produce otros 8 números: los pesos no cambian, lo que cambia es la entrada (diapositiva 9)"],
         "Cada coordenada latente <b>no tiene un significado propio</b> (no es «el trazo vertical»). Es solo la salida de una función con pesos aprendidos. "
         "En PCA sí se ordenan las componentes por importancia; en el vector latente de un autoencoder, no.")

    # ---------------------------------------------------------------- 2 reconstruir vs clasificar
    tema(s, "2", "Reconstruir no es clasificar: la entrada también es la referencia",
         "<b>La mecánica es la misma que en la Clase 45; cambian la salida y la referencia.</b> Un clasificador entrega 10 probabilidades y se compara con la etiqueta; un autoencoder entrega 64 intensidades y se compara con la propia imagen.",
         [("Qué es reconstruir", "Estimar de nuevo cada una de las 64 intensidades y colocarla en su mismo lugar. La salida tiene el mismo tamaño y el mismo significado que la entrada (diapositiva 5)."),
          ("La referencia es la entrada", "x entra al modelo y se conserva para evaluar la salida: la pérdida compara x y x̂ píxel a píxel (diapositiva 7). Es como un modelo de regresión donde lo «observado» es la propia entrada."),
          ("Sin etiquetas", "La etiqueta del dígito no interviene en la pérdida de reconstrucción. Por eso se puede entrenar con muchas imágenes sin etiquetar (diapositiva 6)."),
          ("Para qué sirve", "Tres usos de la diapositiva 4: buscar imágenes parecidas comparando vectores latentes; recuperar una señal limpia (se entrena con pares ruido/limpio: es otra tarea); priorizar una revisión cuando el error es alto. El uso decide qué hay que conservar y cómo comprobar que ayuda.")],
         [("c46_clasificar.png", "Diapositivas 5 y 6: la misma imagen (un 3 de prueba, id 4) con las dos tareas; el cuadro naranja es el mismo píxel en la entrada y en la reconstrucción.")],
         "Antes la red respondía <b>«¿qué dígito es?»</b> y se corregía con la etiqueta que puso una persona. Ahora responde <b>«¿cómo se ve esto?»</b> y se corrige comparando su dibujo con la imagen original. "
         "Es como copiar un dibujo de memoria: nadie tiene que decirte qué dibujo era, basta con poder comparar tu copia con el original.",
         None,
         [f"Un píxel (fila 3, columna 4) vale {f(PX['x'], 4)} en la entrada y la red lo estima en {f(PX['xhat'], 4)} (diapositiva 5)",
          "Salida de clasificación: 10 valores (uno por dígito); salida de reconstrucción: 64 valores (uno por píxel)",
          "Misma imagen de entrada, mismas capas intermedias y la misma retropropagación; cambia la tarea final (transcripción)"],
         "Reconstruir bien no garantiza que el código sirva para clasificar bien: en la diapositiva 36, dos entradas con etiquetas distintas pueden tener el mismo vector latente. Se comprueba con datos en la sección 7.")

    # ---------------------------------------------------------------- 3 pérdida de información
    t0, t1 = TY["media"]
    d0, d1 = TY["dif"]
    tema(s, "3", "Resumir es perder: el cuello de botella y los 8 números reales",
         "<b>Si dos entradas distintas producen el mismo vector latente, el decoder no puede distinguirlas.</b> Lo importante es qué información se pierde, no solo cuánta.",
         [("Ejemplo pequeño (diapositiva 11)", "Imágenes de 2 píxeles: (0.2, 0.8) y (0.8, 0.2). Un encoder fijo que solo guarda el promedio h = (x₁ + x₂)/2 da h = 0.5 para ambas, y un decoder fijo que repite h devuelve (0.5, 0.5) para las dos."),
          ("Qué se perdió", "El orden: cuál píxel era claro y cuál oscuro. El decoder es determinista: mismo h, misma salida. Ningún entrenamiento posterior puede recuperar lo que el encoder no guardó."),
          ("No todo resumen pierde lo mismo", "Con otro encoder (ejemplo propio, h = x₁ − x₂) el código también es un solo número, pero conserva el orden de estas dos imágenes. Un buen resumen es el que guarda lo que el uso necesita."),
          ("8 números reales ≠ 8 bits", "Cada coordenada es un número real entre −1 y 1, con tantas cifras como haga falta, no un 0 o un 1. Ocho bits solo distinguen 2⁸ = 256 combinaciones. Tampoco equivale a cómo se guardan los datos en el computador (diapositiva 10).")],
         [("c46_toy.png", "Diapositiva 11 (arriba) y ejemplo propio (abajo): el mismo tamaño de código puede perder o conservar el orden. Los colores van de blanco (0) a negro (1).", 0.86),
          ("c46_bits.png", "Nuestra reproducción: el decoder entrenado recibe el código real (fila central) o solo su signo ±1 (fila inferior), sin reentrenar. Primer caso de cada dígito 0 a 5 en prueba.", 0.8)],
         "Si un guardia de seguridad solo anota el <b>promedio</b> de las alturas de dos personas, nunca podrá decir cuál era la más alta. El resumen «promedio» descartó justo la diferencia. "
         "Los números reales del código son como una regla con milímetros; los bits serían una regla con solo dos marcas.",
         ["h = (x₁ + x₂)/2   y   x̂ = (h, h)", "ℓ = ((0.5 − 0.2)² + (0.5 − 0.8)²) / 2 = 0.09"],
         [f"(0.2, 0.8) → h = {t0['h']:g} → ({t0['xhat'][0]:g}, {t0['xhat'][1]:g}), MSE {f(t0['mse'])}; (0.8, 0.2) → h = {t1['h']:g} → igual, MSE {f(t1['mse'])}",
          f"Ejemplo propio: h = x₁ − x₂ da h = {mn(d0['h'], 1)} y {mn(d1['h'], 1)}: códigos distintos y MSE {f(d0['mse'])} en ambos",
          f"Nuestra reproducción: con el código real el MSE de validación es {f(BI['val_real'], 4)}; si se reemplaza por su signo (±1), sube a {f(BI['val_bits'], 4)}",
          f"Redondear el código a 1 decimal da {f(BI['val_redondeo1'], 4)} y a 2 decimales {f(BI['val_redondeo2'], 4)}: el decoder usa la parte fina de los números",
          f"Con 8 signos solo hay {BI['patrones_posibles']} códigos posibles; en las 360 imágenes de validación aparecen {BI['patrones_usados']}"],
         "El experimento de los signos no demuestra cuántos bits «vale» el código: el decoder se entrenó con valores reales y no se reentrenó para códigos binarios. Solo muestra que los números reales llevan información fina.")

    # ---------------------------------------------------------------- 4 MSE
    tema(s, "4", "El error de reconstrucción: MSE por píxel, a mano",
         "<b>El error de una imagen es el promedio, sobre sus 64 píxeles, de la diferencia al cuadrado entre la reconstrucción y la entrada.</b> El error del conjunto es el promedio de los errores de las imágenes.",
         [("Por píxel", "Se resta la entrada a la reconstrucción en cada posición j y se eleva al cuadrado: (x̂ⱼ − xⱼ)². Un píxel puede equivocarse hacia más oscuro o hacia más claro: con el cuadrado, −0.3 y +0.3 cuestan lo mismo (0.09)."),
          ("Por imagen y por conjunto", "ℓ(x, x̂) = (1/64) Σⱼ (x̂ⱼ − xⱼ)². Luego L = promedio de los ℓ de las n imágenes. Es una medida como el error cuadrático medio de una regresión, aplicada pixel a pixel."),
          ("Lo que no dice", "El MSE no dice en qué dirección se equivoca el modelo ni en qué píxel; para eso hay que mirar las imágenes y el mapa de error. El cuadrado penaliza más los errores grandes."),
          ("Cero no es el objetivo", "Una copia perfecta de la entrada (la identidad) tiene error 0 y no resume nada. Si el vector latente conservara todos los detalles, no habría aprendido nada útil. Hace falta una restricción adecuada y evaluar fuera de los datos de ajuste (diapositiva 14).")],
         [("c46_mse_toy.png", "Diapositivas 12 y 13: la diferencia al cuadrado y las tres reconstrucciones propuestas para la referencia (0.2; 0.8)."),
          ("c46_pixel.png", "Diapositiva 29: el píxel de la diapositiva 5 aporta el mayor cuadrado de la imagen; el MSE de la imagen es el promedio de los 64 cuadrados.")],
         "Piensa en una <b>multa por cada píxel mal dibujado</b>, que crece con el cuadrado del error: equivocarse por poco casi no cuesta; equivocarse mucho cuesta muy caro. "
         "La nota final de la imagen es el promedio de las 64 multas.",
         ["ℓ(x, x̂) = (1/64) Σⱼ (x̂ⱼ − xⱼ)²", "L = (1/n) Σᵢ ℓ(xᵢ, x̂ᵢ)"],
         [f"Referencia (0.2; 0.8) y reconstrucción (0.5; 0.5): ℓ = (0.09 + 0.09)/2 = {f(TY['propuestas'][2]['mse'])}; con (0.3; 0.7): {f(TY['propuestas'][1]['mse'])}; con (0.2; 0.8): 0.00",
          f"Píxel de la diapositiva 5: ({f(PX['xhat'], 4)} − {f(PX['x'], 4)})² = ({mn(PX['dif'])})² = {f(PX['cuadrado'], 4)} (diapositiva 29: ≈ 0.1629)",
          f"Los 64 cuadrados suman {f(PX['suma'], 4)}; promediados dan el MSE de esa imagen: {f(PX['mse_imagen'], 6)} (diapositiva 29: 0.017220)",
          "Ese solo píxel concentra el " + pc(PX["cuadrado"] / PX["suma"]) + " de la suma de errores de la imagen"],
         "El MSE depende de la escala: aquí las intensidades van de 0 a 1 (se dividen por 16). Un «0.017» solo se entiende al compararlo con una referencia.")

    # ---------------------------------------------------------------- 5 referencias
    tema(s, "5", "Referencias para comparar: imagen media y PCA con 8 componentes",
         f"<b>Un MSE aislado no dice si es bueno: hay que compararlo con referencias simples.</b> En validación: imagen media {f(M['media']['val'], 4)}, PCA con 8 componentes {f(M['pca8']['val'], 4)} y el autoencoder profundo {f(M['prof']['val'], 4)}.",
         [("Imagen media", "Ignora la entrada y devuelve siempre el promedio de las imágenes de entrenamiento. Es el piso de «no saber nada de esta imagen»."),
          ("PCA con 8 componentes", "Proyecta cada imagen sobre 8 direcciones (z = (x − x̄)Pₖ) y reconstruye en ese subespacio (x̂ = x̄ + zPₖᵀ). Es una reconstrucción <b>lineal</b> con un resumen también de 8 números, ajustada solo con entrenamiento."),
          ("Autoencoder", "Reconstrucción <b>no lineal</b>: las capas ocultas con tanh permiten formas que una proyección lineal no captura. Se compara en las mismas imágenes y con la misma escala (diapositiva 27)."),
          ("Cómo leer la comparación", "Menor MSE es mejor. Pero esto responde solo una pregunta (¿reconstruye?): aún falta evaluar la utilidad del código para otra tarea.")],
         [("c46_reconstrucciones.png", "Diapositiva 28: primer caso de cada dígito 0 a 5 en prueba. Solo la visualización recorta las intensidades a [0, 1]; el MSE de cada imagen usa las salidas completas.", 0.8),
          ("c46_mse_barras.png", "Diapositivas 27 y 30: MSE de las cuatro reconstrucciones en validación y prueba (izquierda) y PCA con distintos k (derecha; nuestra reproducción).", 0.86)],
         "La imagen media es el <b>retrato robot</b> que sirve para todos y no se parece a nadie. PCA es un dibujante que solo sabe usar <b>8 plantillas</b> combinadas con regla y compás. "
         "El autoencoder es un dibujante que puede combinar los 8 números de forma más flexible (curvas); por eso, si lo entrenamos bien, puede acercarse más al original.",
         ["z = (x − x̄) Pₖ", "x̂ = x̄ + z Pₖᵀ"],
         [f"Imagen media: validación {f(M['media']['val'], 4)} y prueba {f(M['media']['prueba'], 4)}; PCA 8: {f(M['pca8']['val'], 4)} y {f(M['pca8']['prueba'], 4)}; autoencoder elegido: {f(M['prof']['val'], 4)} y {f(M['prof']['prueba'], 4)} (diapositiva 27)",
          f"PCA 8 reduce el error de la media en {pc(M['reduccion_pca_vs_media'])}; el autoencoder profundo, en {pc(M['reduccion_prof_vs_media'])} y respecto de PCA 8 en {pc(M['reduccion_prof_vs_pca'])} (nuestro cálculo)",
          f"Nuestra reproducción del 64–8–64: {f(M['ae8']['val'], 4)} en validación; casi igual a PCA 8 ({f(M['pca8']['val'], 4)})",
          f"PCA con k = 1, 2, 4, 8, 16, 32 y 64: " + ", ".join(f(N['pca_k'][str(k)], 4) for k in (1, 2, 4, 8, 16, 32)) + " y 0 con k = 64 (se usa toda la información)"],
         "Una semilla y una corrida no establecen estabilidad, y la prueba ya se mostró en clase: la comparación es retrospectiva. Que gane el autoencoder en esta corrida no prueba superioridad en general (diapositiva 47).")

    # ---------------------------------------------------------------- 6 profundidad, parámetros, entrenamiento
    tema(s, "6", "Profundidad, parámetros y entrenamiento",
         f"<b>Se compararon dos arquitecturas con el mismo vector latente de 8 valores: 64–8–64 ({mil(PA['total_ae8'])} parámetros) y 64–32–8–32–64 ({mil(PA['total_prof'])} parámetros).</b> Se elige con el MSE de validación.",
         [("Receta", "Capas ocultas con tanh y salida lineal; Adam con tasa 0.002; lotes de 64 imágenes; penalización de pesos alpha = 0.0001; hasta 150 épocas por candidata; referencias: imagen media y PCA 8 (diapositiva 23)."),
          ("Latente y parámetros no son lo mismo", "Ambas redes resumen cada imagen en 8 valores, pero la que tiene capas adicionales de 32 unidades aprende más pesos y sesgos. Cada capa aporta unidades × (entradas + 1) parámetros."),
          ("Cómo aprende", "1) reconstruir x → h → x̂; 2) medir ℓ(x, x̂); 3) retropropagar los gradientes hacia las dos mitades; 4) actualizar pesos y sesgos; 5) repetir con el lote siguiente; al terminar cada época se mide la validación <i>sin</i> actualizar."),
          ("El límite de épocas", "Los mínimos de validación están en la última época (la 150): el presupuesto de entrenamiento limita la conclusión, porque la curva aún bajaba (diapositiva 26).")],
         [("c46_arquitecturas.png", "Diapositivas 17 y 18: tamaños por capa y parámetros (pesos + sesgos) de cada capa, calculados con unidades × (entradas + 1).", 0.85),
          ("c46_curvas.png", "Diapositivas 25 y 26: MSE por píxel (sin penalización) en cada época, con la mejor época marcada en rojo.")],
         "Una capa más es como añadir <b>una estación intermedia en una cadena de montaje</b>: la imagen pasa por más transformaciones antes de llegar al resumen de 8 números. "
         "Más estaciones dan más flexibilidad, pero también más piezas que ajustar y más tiempo; y que la curva siga bajando al final significa que no sabemos si ya se llegó al fondo.",
         ["parámetros de una capa = unidades × (entradas + 1)", "64–8–64: 8·(64+1) + 64·(8+1) = 520 + 576 = 1 096"],
         [f"64–8–64: 520 + 576 = {mil(PA['total_ae8'])}; 64–32–8–32–64: {' + '.join(mil(q['total']) for q in PA['prof'])} = {mil(PA['total_prof'])} (diapositiva 18)",
          f"MSE de validación al final de la época 150: {f(CU['AE8']['va_150'], 5)} (64–8–64) y {f(CU['AE32_8_32']['va_150'], 5)} (64–32–8–32–64); en la época 1: {f(CU['AE8']['va_1'], 4)} y {f(CU['AE32_8_32']['va_1'], 4)}",
          f"Entrenamiento y validación casi coinciden (época 150: {f(CU['AE8']['tr_150'], 5)} y {f(CU['AE32_8_32']['tr_150'], 5)} en entrenamiento): no se ve sobreajuste en este presupuesto",
          "Protocolo (diapositiva 22): 1 077 imágenes de entrenamiento, 360 de validación y 360 de prueba histórica; PCA y redes se ajustan con entrenamiento"],
         "Elegir la arquitectura y la época con los resultados de <i>prueba</i> también es fuga de información. La elección se hace con validación; la prueba queda para evaluar la receta ya fijada (diapositiva 21).")

    # ---------------------------------------------------------------- 7 mirar el código y usarlo
    tema(s, "7", "Mirar el código y usarlo para otra tarea",
         "<b>El vector latente puede servir para otras tareas, pero cada uso necesita su propia evaluación.</b> Un buen MSE de reconstrucción no prueba que el código ayude a clasificar, buscar o alertar.",
         [("Mirar 2 de las 8 coordenadas", "Se pueden graficar dos coordenadas y colorear por dígito; los colores son etiquetas añadidas solo para inspeccionar, el autoencoder no las usó (diapositiva 32). Dos de ocho no cuentan toda la historia."),
          ("Buscar parecidos", "Se comparan los 8 valores del vector latente y se ordenan las imágenes por cercanía (diapositiva 48). Útil si «parecido» en el código coincide con «parecido» para la tarea."),
          ("Clasificar con representaciones fijas", "Se congelan PCA y el autoencoder, se transforman las imágenes y se ajusta una regresión logística con estandarización y C = 1 sobre entrenamiento; se compara con la misma receta sobre píxeles (diapositivas 34 y 35)."),
          ("Qué evaluar según el uso", "Reconstruir: error sobre la entrada. Clasificar: métricas con etiquetas. Buscar vecinos: relevancia de lo recuperado. Alertar: errores y utilidad en condiciones reales (diapositiva 33).")],
         [("c46_codigos2d.png", "Diapositiva 32: coordenadas 1 y 2 de PCA y del autoencoder para las 360 imágenes de prueba, coloreadas por dígito.", 0.84),
          ("c46_utilidad.png", "Diapositiva 35: logística con píxeles, con PCA 8 y con el vector latente del autoencoder; prueba histórica de 360 imágenes.", 0.78)],
         "Es como resumir un libro en 8 palabras clave. Para <b>encontrar libros parecidos</b> quizá basta; para <b>decidir de qué género es</b> quizá faltan palabras; y para <b>detectar una errata</b> no sirve. "
         "Cada uso tiene su propia prueba.",
         None,
         [f"Vecinos del 8 (id {VE['consulta_id']}) en entrenamiento: distancias " + ", ".join(f(v, 3) for v in VE["dist"]) + ", los tres son 8 (diapositiva 48)",
          f"Logística sobre píxeles: log loss {f(UT['pixeles']['log_loss'], 4)} y {UT['pixeles']['errores']} errores de 360; sobre PCA 8: {f(UT['pca8']['log_loss'], 4)} y {UT['pca8']['errores']}; sobre el latente del AE: {f(UT['ae']['log_loss'], 4)} y {UT['ae']['errores']} (diapositiva 35)",
          "Los dos vectores de 8 valores dan 25 errores; los píxeles, 11: <b>la utilidad depende de la receta</b>",
          "Diapositiva 36: en un ejemplo artificial, dos entradas de etiquetas distintas se reconstruyen con MSE 0.005 y tienen el mismo vector latente"],
         "Que PCA y el autoencoder den 25 errores en esta receta no demuestra que ninguno sirva para clasificar con otro clasificador, ni que los píxeles sean siempre mejores.")

    # ---------------------------------------------------------------- 8 anomalías
    tema(s, "8", "El error de reconstrucción como alerta de anomalías, y sus límites",
         "<b>Si el modelo aprendió a reconstruir lo habitual, un error alto sugiere que la imagen no se parece a lo habitual.</b> El puntaje e(x) es el MSE de la imagen y se activa una alerta si e(x) &gt; c; no es una probabilidad de anomalía.",
         [("Puntaje y umbral", "e(x) = (1/64) Σⱼ (x̂ⱼ − xⱼ)². El umbral c se fija antes de mirar la prueba: aquí es el percentil 95 de los 360 errores de validación (diapositivas 38 y 39)."),
          ("Qué significa el percentil", "Por construcción, el 5 % de la validación queda sobre el corte. No es un 95 % de confianza sobre defectos: el percentil de un puntaje no es un nivel de confianza (diapositiva 46)."),
          ("Prueba de estrés artificial", "Se desplaza cada imagen una columna a la derecha y se compara cada salida con <i>su propia</i> entrada desplazada. Muestra sensibilidad a un cambio, no detección validada de defectos reales (diapositivas 40 y 41)."),
          ("Cuándo falla una alerta", "Variación habitual no representada (muchos avisos sobre casos aceptables); anomalías en el entrenamiento y un decoder flexible (aprende a reconstruirlas); cambio de adquisición (el puntaje refleja el cambio técnico). Se necesita una validación específica (diapositiva 42).")],
         [("c46_umbral.png", "Diapositivas 38 y 39: errores de validación ordenados (izquierda, en rojo los que superan el corte) e histograma (derecha).", 0.8),
          ("c46_peores.png", "Las seis imágenes de prueba con mayor error de reconstrucción (nuestra reproducción): todas superan el umbral; una alerta no confirma un defecto.", 0.8),
          ("c46_estres.png", "Diapositivas 40 y 41: arriba, la imagen 3 original y desplazada con sus reconstrucciones; abajo, los errores de las 360 imágenes de prueba originales y desplazadas.", 0.84)],
         "Es como el <b>detector de humo</b> de una cocina: suena cuando el ambiente se aleja de lo habitual, pero puede sonar por una tostada quemada (falsa alarma) y no suena si el humo es «habitual» para él. "
         "El sonido no te dice si hay fuego, solo que hay que ir a mirar.",
         ["e(x) = (1/64) Σⱼ (x̂ⱼ − xⱼ)²", "alerta si e(x) &gt; c"],
         [f"Umbral (percentil 95 de validación): c = {f(UM['umbral'], 6)}; {UM['val_sobre']} de {UM['n_val']} errores de validación lo superan ({pc(UM['val_pct'], 0)})",
          f"En prueba, originales: {UM['prueba_sobre']} de 360 alertas ({f(UM['prueba_pct'] * 100)} %); desplazadas: {UM['desplazadas_sobre']} de 360 ({f(UM['desplazadas_pct'] * 100)} %) (diapositiva 40)",
          f"MSE de la imagen 3 (id 4): {f(PX['mse_imagen'], 4)}; desplazada: {f(PX['mse_desplazada'], 4)} (diapositiva 41)",
          "Caso de la diapositiva 45: «rechazar el 5 % con mayor error» no equivale a «detectar cualquier defecto con 95 % de confianza»"],
         "Selección y umbral usaron la misma validación: no garantizan falsas alarmas futuras. Y el estrés artificial no mide sensibilidad ni especificidad frente a anomalías reales.")

    # ---------------------------------------------------------------- repaso
    s.append(CondPageBreak(110 * mm))
    s.append(band("Repaso, alertas del material y criterio de dominio", "Lo que debes poder explicar sin apuntes", NAVY))
    s.append(Spacer(1, 5))
    s.append(P("Alertas del material", "h2"))
    s.append(box([P("<b>Qué conviene saber al leer las diapositivas y la transcripción</b>", "cellb"),
                  bullets(["<b>La transcripción tenía errores de reconocimiento de voz</b> («DeepLens» por deep learning; «Ableton Core» y «topcoder» por autoencoder; «tarjeta hiperbólica» por tangente hiperbólica; «RER meronales» por redes neuronales; «rich o lazo» por ridge o lasso; «0208» por 0.2, 0.8; «peso y se asco» por pesos y sesgos). Se corrigieron en 49 líneas; el original se conserva en <font name='Mono'>VideosClases/originales_sin_corregir/</font>.",
                           "<b>La grabación termina en el recreo</b> (1:14:23): no incluye las diapositivas 15 a 53. Lo que esas láminas dicen se explica aquí desde las láminas y el programa, no desde lo hablado.",
                           "<b>«Pérdida logarítmica» en la clase.</b> Al describir el ciclo de entrenamiento se habló de una pérdida logarítmica; la pérdida de reconstrucción de las diapositivas y del programa es el <b>MSE</b> (log loss solo aparece al clasificar, diapositiva 35).",
                           "<b>«Clasificar de manera no supervisada».</b> Se sugirió oralmente que el autoencoder permitiría clasificar sin etiquetas; las diapositivas 35 y 36 muestran que un buen código de reconstrucción no garantiza una buena clasificación (25 errores frente a 11 con píxeles).",
                           "<b>Resumir no es reducir el archivo.</b> En clase se aclaró que pasar de 64 a 8 no busca achicar la imagen: las 8 coordenadas controlan cuántos números la describen; no es compresión de archivo ni equivale a 8 bits.",
                           "<b>El código de la diapositiva 9</b> corresponde a un «8» de entrenamiento (id 9), y el 3 de las diapositivas 5, 29 y 41 es el id 4 de prueba.",
                           "<b>Las cifras de R son otra corrida.</b> Las diapositivas usan la ejecución de Python; R entrena con RSNNS y descenso de gradiente y puede diferir."], "cell", AMBER)],
                 bg=AMBER_BG, border=AMBER, pad=5))
    s.append(Spacer(1, 5))
    s.append(P("Preguntas de repaso (intenta responder antes de leer)", "h2"))
    pregs = [("Encoder h = (x₁ + x₂)/2 y decoder (h, h). Para x = (0.2, 0.8), ¿qué h y qué reconstrucción? ¿Y el MSE?", f"h = 0.5; x̂ = (0.5, 0.5); MSE = (0.3² + 0.3²)/2 = {f(TY['propuestas'][2]['mse'])}."),
             ("¿Cuántos valores tienen x, h y x̂ en 64–8–64? ¿Cuántos parámetros?", f"64, 8 y 64; 520 + 576 = {mil(PA['total_ae8'])}."),
             ("Referencia (0.2, 0.8) y reconstrucción (0.3, 0.7). ¿Cuánto vale ℓ?", f"((0.1)² + (0.1)²)/2 = {f(TY['propuestas'][1]['mse'])}."),
             ("¿Por qué un error de reconstrucción 0 no es la meta?", "La identidad lo logra sin resumir nada; hace falta un cuello de botella y evaluar fuera del ajuste."),
             ("Un píxel vale 0.8125 y se reconstruye como 0.4089. ¿Cuánto aporta al MSE de la imagen?", f"({mn(PX['dif'])})² = {f(PX['cuadrado'], 4)}, que se promedia con los otros 63 cuadrados."),
             ("¿Con qué dos referencias se compara el autoencoder y qué valores dan en validación?", f"Imagen media {f(M['media']['val'], 4)} y PCA 8 {f(M['pca8']['val'], 4)}; el autoencoder elegido {f(M['prof']['val'], 4)}."),
             ("¿Con qué conjunto se elige entre las dos arquitecturas y qué limita la conclusión?", "Con validación (MSE); ambos mínimos están en la época 150, así que el presupuesto de épocas limita la conclusión."),
             ("Un puntaje de reconstrucción de 0.05, ¿es una probabilidad de anomalía?", "No: es un MSE; solo se compara con un umbral que debe validarse para el uso previsto."),
             ("Umbral = percentil 95 de validación: ¿qué porcentaje de validación lo supera y por qué en prueba salió 9.72 %?", f"5 % por construcción ({UM['val_sobre']} de 360); en prueba ({UM['prueba_sobre']} de 360) la proporción puede diferir porque el umbral se fijó con otras imágenes.")]
    s.append(table([["#", "Pregunta", "Respuesta corta"]] + [[str(i), q, a] for i, (q, a) in enumerate(pregs, 1)], [7 * mm, 88 * mm, CONTENT_W - 95 * mm]))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Criterio de dominio</b>", "cellb"), Spacer(1, 3),
                  bullets(["Explico con mis palabras qué es una representación, un encoder, un decoder y un vector latente",
                           "Calculo a mano la reconstrucción de un autoencoder de juguete y su MSE por píxel y por imagen",
                           "Cuento los parámetros de 64–8–64 y de 64–32–8–32–64, y explico por qué latente y parámetros no son lo mismo",
                           "Explico por qué una copia perfecta no es la meta y por qué 8 números reales no equivalen a 8 bits",
                           "Comparo un autoencoder con la imagen media y PCA, y digo qué limita esa conclusión",
                           "Explico por qué un puntaje de reconstrucción no es una probabilidad y qué validación falta para usarlo como alerta"], "cell", GREEN, SYM)],
                 bg=GREEN_BG, border=GREEN))
    s.append(Spacer(1, 5))
    s.append(P("Cifras de las diapositivas frente a nuestra reproducción (programa del profesor)", "h3"))
    filas = [["Cifra", "Diapositivas", "Reproducido", "¿Igual?"]] + [[r["item"], r["diapositiva"], r["reproducido"], "sí" if r["igual"] else "NO"] for r in N["comparacion"]]
    s.append(table(filas, [70 * mm, 46 * mm, 46 * mm, CONTENT_W - 162 * mm]))
    s.append(Spacer(1, 3))
    s.append(P("No reproducible con el programa: " + "; ".join(N["no_reproducible"]) + ".", "small"))
    build("C46_Guia_Autoencoders.pdf", s, "Clase 46 · Autoencoders y representaciones")


# ======================================================================= APUNTES
HEADS = []


def h(num, title):
    p = P(f'<font color="#6D28D9">{num}</font>  {title}', "h2")
    HEADS.append(p)
    return p


def apuntes():
    s = []
    s.append(band("Apuntes · Clase 46 · Autoencoders y representaciones", "Para leer antes de los ejercicios · pensado para quien acaba de aprender redes neuronales y nunca ha visto un autoencoder", C46))
    s.append(Spacer(1, 5))
    s.append(box([P("<b>Lo más importante, en 8 ideas</b>", "cellb"), Spacer(1, 2),
                  numbered([("Autoencoder", "red entrenada para reproducir su propia entrada pasando por un resumen pequeño."),
                            ("Encoder y decoder", "h = f(x) resume (64 → 8); x̂ = g(h) reconstruye (8 → 64)."),
                            ("Referencia = entrada", "no hacen falta etiquetas; la pérdida compara x con x̂."),
                            ("MSE por píxel", "promedio de (x̂ⱼ − xⱼ)²; ℓ por imagen, L por conjunto."),
                            ("Resumir es perder", "mismo h implica misma reconstrucción; 8 reales no son 8 bits."),
                            ("Referencias", "imagen media y PCA 8: un MSE se entiende al compararlo."),
                            ("Parámetros", "unidades × (entradas + 1) por capa; latente y parámetros no son lo mismo."),
                            ("Alerta por error", "e(x) &gt; c: es un puntaje, no una probabilidad; exige validación específica.")], C46)], bg=SOFT, border=NAVY, pad=5))
    s.append(Spacer(1, 3))
    s.append(P("Cada sección trae una explicación en palabras simples, un <b>ejemplo resuelto con números distintos</b> a los de las diapositivas y los ejercicios, y un resumen «para recordar». "
               "Ejercicios en <font name='Mono'>C46_Ejercicios_Cuaderno_Autoencoders.pdf</font> y notebook <font name='Mono'>C46_Autoencoders.ipynb</font>.", "body"))
    s.append(Spacer(1, 3))

    # 1 representación
    s.append(h("1", "Qué es una representación y qué es un autoencoder"))
    s += para("Una <b>representación</b> es una forma de describir un objeto con números. Una imagen de 8 × 8 se describe con 64 intensidades (de 0, blanco, a 1, negro, tras dividir por 16). "
              "Una red neuronal aprende representaciones en sus capas intermedias: cada capa recibe una descripción y entrega otra. Un <b>autoencoder</b> es una red con una forma especial: "
              "recibe x, la lleva a un <b>vector latente</b> h más pequeño con el <b>encoder</b> (h = f(x)) y desde h intenta estimar x otra vez con el <b>decoder</b> (x̂ = g(h)). "
              "No se pide que sea idéntico: solo que sea una aproximación buena. Y no se necesitan etiquetas: la referencia con la que se compara es la propia imagen.")
    x4 = np.array([0.0, 0.2, 0.9, 1.0])
    h4 = np.array([(x4[0] + x4[1]) / 2, (x4[2] + x4[3]) / 2])
    xr = np.array([h4[0], h4[0], h4[1], h4[1]])
    e4 = (xr - x4) ** 2
    xb = np.array([0.0, 1.0, 0.0, 1.0])
    hb = np.array([0.5, 0.5])
    s.append(ejemplo("imagen de 4 píxeles → 2 números → 4 píxeles (ejemplo propio)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Encoder (promedia de a dos)", "h₁ = (0 + 0.2)/2; h₂ = (0.9 + 1)/2", f"h = ({h4[0]:.2f}; {h4[1]:.2f})"],
               ["Decoder (repite cada h dos veces)", "x̂ = (h₁, h₁, h₂, h₂)", f"x̂ = ({xr[0]:.2f}; {xr[1]:.2f}; {xr[2]:.2f}; {xr[3]:.2f})"],
               ["Diferencias al cuadrado", "(0.1 − 0)²; (0.1 − 0.2)²; (0.95 − 0.9)²; (0.95 − 1)²", f"{e4[0]:.4f}; {e4[1]:.4f}; {e4[2]:.4f}; {e4[3]:.4f}"],
               ["MSE de la imagen", f"({e4[0]:.4f} + {e4[1]:.4f} + {e4[2]:.4f} + {e4[3]:.4f}) ÷ 4", f"{e4.mean():.5f}"],
               ["Otra imagen: (0; 1; 0; 1)", "h = (0.5; 0.5) → x̂ = (0.5; 0.5; 0.5; 0.5)", f"MSE = {((np.array([0.5] * 4) - xb) ** 2).mean():.2f}"]],
              [48 * mm, 74 * mm, CONTENT_W - 122 * mm - 18]),
        P(f"Lectura: este resumen conserva bien una imagen «suave» (MSE {e4.mean():.5f}) y destruye una con contrastes entre píxeles vecinos (MSE 0.25). Un resumen solo conserva lo que su diseño permite conservar. "
          "<b>Error común</b>: pensar que el resumen es «una versión chica de la imagen»; es un conjunto de números cuyo significado lo decide el entrenamiento.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c46_esquema.png", width=CONTENT_W * 0.92))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Autoencoder: x → h = f(x) → x̂ = g(h); h es el vector latente (aquí 8 valores)", "La referencia es la entrada: no se usan etiquetas", "Las coordenadas latentes no tienen un significado individual"]))

    # 2 forward de un autoencoder con tanh
    W1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    W2 = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, -1.0]])
    xx = np.array([1.0, 0.0, 0.5])
    a1 = xx @ W1
    hh = np.tanh(a1)
    xhat = hh @ W2
    d2 = (xhat - xx) ** 2
    s.append(h("2", "Un autoencoder pequeño, calculado a mano (tanh y salida lineal)"))
    s += para("En la clase las capas ocultas usan <b>tanh</b> y la salida es <b>lineal</b> (sin activación). La tangente hiperbólica lleva cualquier número al intervalo (−1, 1): tanh(0) = 0, tanh(1) = 0.7616, tanh(−2) = −0.9640. "
              "La salida lineal puede dar valores fuera de [0, 1]; el MSE usa esas salidas completas y solo la visualización las recorta. "
              "Con una convención fija (cada imagen es una <b>fila</b> y se multiplica x·W), un autoencoder 3 → 2 → 3 se calcula así.")
    s.append(ejemplo("red 3 → 2 → 3, sesgos en 0 (ejemplo propio)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Combinación del encoder", "a = xW₁ con x = (1; 0; 0.5), columnas (1, 0, 1) y (0, 1, −1)", f"a = ({a1[0]:.1f}; {a1[1]:.1f})".replace("-", "−")],
               ["Vector latente", "h = tanh(a)", f"h = ({hh[0]:.4f}; {hh[1]:.4f})".replace("-", "−")],
               ["Salida lineal", "x̂ = hW₂: 0.5h₁; 0.5h₂; 0.5h₁ − h₂", f"x̂ = ({xhat[0]:.4f}; {xhat[1]:.4f}; {xhat[2]:.4f})".replace("-", "−")],
               ["Cuadrados", "(x̂ − x)² píxel a píxel", f"{d2[0]:.4f}; {d2[1]:.4f}; {d2[2]:.4f}"],
               ["MSE de la imagen", f"({d2[0]:.4f} + {d2[1]:.4f} + {d2[2]:.4f}) ÷ 3", f"{np.round(d2, 4).sum() / 3:.4f}"],
               ["Parámetros", "encoder 3·2 + 2; decoder 2·3 + 3", f"{3 * 2 + 2} + {2 * 3 + 3} = {3 * 2 + 2 + 2 * 3 + 3}"]],
              [38 * mm, 78 * mm, CONTENT_W - 116 * mm - 18]),
        P("Lectura: aunque los sesgos valgan 0 aquí, se cuentan como parámetros porque se aprenden. <b>Error común</b>: aplicar tanh a la salida; en la receta de la clase la salida es lineal. "
          "Y recuerda que h tiene 2 números para una imagen de 3: ya se perdió información.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c46_tanh.png", width=CONTENT_W * 0.86))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Capas ocultas: tanh acota a (−1, 1); salida: lineal, puede salirse de [0, 1]", "Cada imagen es una fila: a = xW + b; h = tanh(a); x̂ = hW′ + b′", "Los sesgos también son parámetros"]))

    # 3 MSE
    r = np.array([0.0, 0.5, 1.0, 0.25])
    q = np.array([0.1, 0.5, 0.6, 0.35])
    dq = (q - r) ** 2
    s.append(h("3", "El error de reconstrucción (MSE): por píxel, por imagen y por conjunto"))
    s += para("Para cada píxel se resta la entrada a la reconstrucción, se eleva al cuadrado, y se promedian los 64 cuadrados para obtener el error de la <b>imagen</b>: ℓ(x, x̂) = (1/64) Σⱼ (x̂ⱼ − xⱼ)². "
              "El error del <b>conjunto</b> es el promedio de los errores de sus imágenes. Se eleva al cuadrado para que los signos no se cancelen y para penalizar más los errores grandes. "
              "El precio: el MSE solo dice «cuánto», no «hacia dónde» se equivocó. Un MSE de 0 es una copia perfecta, que no resume nada: no es la meta.")
    s.append(ejemplo("imagen de 4 píxeles y su reconstrucción (ejemplo propio)", [
        pasos([["Píxel", "Entrada x", "Reconstrucción x̂", "Diferencia", "Al cuadrado"],
               ["1", f"{r[0]:.2f}", f"{q[0]:.2f}", f"{q[0] - r[0]:.2f}", f"{dq[0]:.4f}"],
               ["2", f"{r[1]:.2f}", f"{q[1]:.2f}", f"{q[1] - r[1]:.2f}", f"{dq[1]:.4f}"],
               ["3", f"{r[2]:.2f}", f"{q[2]:.2f}", f"{q[2] - r[2]:.2f}".replace("-", "−"), f"{dq[2]:.4f}"],
               ["4", f"{r[3]:.2f}", f"{q[3]:.2f}", f"{q[3] - r[3]:.2f}", f"{dq[3]:.4f}"],
               ["MSE", "", "", "", f"{dq.sum():.4f} ÷ 4 = {dq.mean():.4f}"]],
              [18 * mm, 30 * mm, 38 * mm, 30 * mm, CONTENT_W - 116 * mm - 18]),
        P(f"Lectura: el píxel 3 se equivocó por {abs(q[2] - r[2]):.1f} y aporta {dq[2]:.2f} de los {dq.sum():.2f} totales: el cuadrado hace que un error grande domine. "
          f"En la clase: el píxel de la diapositiva 5 vale 0.8125 y se reconstruye como 0.4089: ({mn(PX['dif'])})² = {f(PX['cuadrado'], 4)}; los 64 cuadrados de esa imagen suman {f(PX['suma'], 4)} y su promedio es {f(PX['mse_imagen'], 6)}. "
          "<b>Error común</b>: promediar las diferencias sin elevarlas al cuadrado (los signos se cancelan).", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c46_pixel.png", width=CONTENT_W * 0.9))
    s.append(Spacer(1, 2))
    s.append(recuerda(["ℓ = promedio de (x̂ⱼ − xⱼ)² sobre los 64 píxeles; L = promedio sobre las imágenes", "−0.3 y +0.3 cuestan lo mismo (0.09): el MSE no dice la dirección", "Error 0 = copia perfecta: no es la meta"]))

    # 4 referencias y PCA
    s.append(h("4", "Referencias: imagen media y PCA con 8 componentes"))
    s += para("Un MSE aislado no dice si es bueno; hay que compararlo. La <b>imagen media</b> reconstruye todas las imágenes con el promedio de entrenamiento (no usa la entrada). "
              "<b>PCA</b> con k componentes guarda k números por imagen: z = (x − x̄)Pₖ, y reconstruye x̂ = x̄ + zPₖᵀ; es una reconstrucción <b>lineal</b>. "
              "Ambas se ajustan solo con entrenamiento. El autoencoder tiene la misma cantidad de números en el resumen, pero una reconstrucción <b>no lineal</b>.")
    xp = np.array([0.9, 0.7])
    xm = np.array([0.5, 0.5])
    P1 = np.array([1 / sqrt(2), 1 / sqrt(2)])
    zp = float((xp - xm) @ P1)
    xph = xm + zp * P1
    e_pca = ((xph - xp) ** 2).mean()
    e_med = ((xm - xp) ** 2).mean()
    s.append(ejemplo("PCA con 1 componente en 2 píxeles (ejemplo propio)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Centrar", "x − x̄ = (0.9 − 0.5; 0.7 − 0.5)", "(0.4; 0.2)"],
               ["Código z", "(0.4 + 0.2) ÷ √2, con P₁ = (1/√2; 1/√2)", f"z = {zp:.4f}"],
               ["Reconstruir", "x̄ + z·P₁ = (0.5 + 0.3; 0.5 + 0.3)", f"x̂ = ({xph[0]:.1f}; {xph[1]:.1f})"],
               ["MSE de PCA", "((0.8 − 0.9)² + (0.8 − 0.7)²) ÷ 2", f"{e_pca:.2f}"],
               ["MSE de la imagen media", "((0.5 − 0.9)² + (0.5 − 0.7)²) ÷ 2", f"{e_med:.2f}"]],
              [38 * mm, 78 * mm, CONTENT_W - 116 * mm - 18]),
        P(f"Lectura: con un solo número (z) PCA reduce el error de {e_med:.2f} a {e_pca:.2f} porque aprovecha que los dos píxeles tienden a variar juntos. "
          "<b>Error común</b>: creer que PCA mira etiquetas; solo usa las imágenes.", "cell")]))
    s.append(Spacer(1, 3))
    s += para(f"En la clase (validación, 360 imágenes): imagen media {f(M['media']['val'], 4)}; PCA con 8 componentes {f(M['pca8']['val'], 4)}; autoencoder 64–8–64 {f(M['ae8']['val'], 4)} (nuestra reproducción) y autoencoder profundo {f(M['prof']['val'], 4)}. "
              f"PCA 8 reduce el error de la media en {pc(M['reduccion_pca_vs_media'])} y el profundo, otro {pc(M['reduccion_prof_vs_pca'])} respecto de PCA 8. "
              "Pero una corrida, una semilla y una prueba ya conocida no establecen superioridad general.")
    s.append(img("c46_reconstrucciones.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Imagen media: ignora la entrada; PCA: reconstrucción lineal con k números; autoencoder: no lineal", "Todas las referencias se ajustan solo con entrenamiento", "Menor MSE es mejor, pero una corrida no prueba superioridad"]))

    # 5 parámetros y arquitectura
    s.append(h("5", "Arquitectura y conteo de parámetros: el latente no es el tamaño del modelo"))
    s += para("Una arquitectura 64–8–64 significa 64 entradas, una capa latente de 8 unidades y 64 salidas. Cada capa tiene (entradas + 1) parámetros por unidad (el +1 es el sesgo). "
              "En 64–32–8–32–64 hay tres capas ocultas (32, 8 y 32). Las dos redes tienen un vector latente de 8 valores, pero la segunda aprende más pesos y sesgos porque tiene capas adicionales. "
              "Más parámetros dan más flexibilidad, más tiempo de cálculo y más posibilidad de memorizar.")
    p16 = [64 * 16 + 16, 16 * 64 + 64]
    s.append(ejemplo("una red 64–16–64 (ejemplo propio) y la práctica de la diapositiva 53", [
        pasos([["Capa", "Cálculo", "Parámetros"],
               ["64 → 16", "16 · (64 + 1) = 16 · 65", f"{p16[0]}"],
               ["16 → 64", "64 · (16 + 1) = 64 · 17", f"{p16[1]}"],
               ["Total 64–16–64", f"{p16[0]} + {p16[1]}", f"{mil(sum(p16))}"],
               ["Diapositiva 53: 64–4–64", "4 · 65 + 64 · 5 = 260 + 320", f"{PA['ae4']}"],
               ["Diapositiva 53: 64–32–4–32–64", "2 080 + (32·4 + 4) + (4·32 + 32) + 2 112 = 2 080 + 132 + 160 + 2 112", f"{mil(PA['prof4'])}"]],
              [48 * mm, 80 * mm, CONTENT_W - 128 * mm - 18]),
        P("Lectura: bajar el latente de 8 a 4 reduce los parámetros de 1 096 a 580, pero reduce también lo que el decoder puede reconstruir. La práctica de la diapositiva 53 pide repetir con 4 y comparar el error de validación "
          "y los detalles de la imagen. <b>Error común</b>: creer que cuantos más parámetros, mejor reconstrucción en datos nuevos.", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c46_arquitecturas.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Parámetros por capa = unidades × (entradas + 1)", f"64–8–64: {mil(PA['total_ae8'])}; 64–32–8–32–64: {mil(PA['total_prof'])}", "Latente (8) y parámetros son cosas distintas"]))

    # 6 entrenamiento
    s.append(h("6", "Cómo se entrena y cómo se elige: la validación y el límite de épocas"))
    s += para("El entrenamiento repite cuatro pasos con cada lote de 64 imágenes: reconstruir (x → h → x̂), medir ℓ(x, x̂), <b>retropropagar</b> los gradientes hacia el decoder y el encoder, y actualizar pesos y sesgos con Adam (tasa 0.002). "
              "Al terminar cada <b>época</b> (una pasada por las 1 077 imágenes de entrenamiento) se mide el MSE de validación <i>sin</i> actualizar parámetros, y se guarda una copia de la mejor época. "
              "La validación elige entre arquitecturas y épocas; la prueba se mira al final. El error que se grafica es el MSE por píxel sin la penalización alpha = 0.0001.")
    s += para(f"En la clase los dos mínimos de validación están en la época 150, la última: 64–8–64 llega a {f(CU['AE8']['val'], 5)} y 64–32–8–32–64 a {f(CU['AE32_8_32']['val'], 5)}. "
              "Si el mínimo coincide con el límite de épocas, el presupuesto limita la conclusión: no sabemos si con más épocas seguiría mejorando. "
              "Y el programa recuerda que ese mínimo «no demuestra convergencia».")
    s.append(img("c46_curvas.png", width=CONTENT_W * 0.88))
    s.append(Spacer(1, 2))
    s.append(recuerda(["Reconstruir → medir → retropropagar → actualizar → repetir; la validación solo compara", "Se elige arquitectura y época con validación; la prueba al final", "Un mínimo en la última época no demuestra convergencia"]))

    # 7 alertas
    pts = np.array([0.011, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.027, 0.031, 0.058])
    c_ = float(np.quantile(pts, 0.8))
    al = pts > c_
    s.append(h("7", "El error como alerta: puntaje, umbral y falsas alarmas"))
    s += para("Si se entrena con lo habitual, un error de reconstrucción alto indica que una imagen no se parece a lo que el modelo aprendió: e(x) = MSE de x. "
              "Se declara una <b>alerta</b> si e(x) &gt; c. El umbral c debe fijarse antes de mirar la prueba y con datos pertinentes; aquí se usa el <b>percentil 95</b> del error de validación. "
              "Esto hace que, por construcción, el 5 % de la validación quede sobre el corte. <b>No es un 95 % de confianza sobre defectos</b>; y e(x) no es una probabilidad de anomalía.")
    s.append(ejemplo("percentil 80 de 10 puntajes (ejemplo propio)", [
        pasos([["Paso", "Cálculo", "Resultado"],
               ["Puntajes ordenados", "10 imágenes de validación", ", ".join(f"{v:.3f}" for v in pts)],
               ["Umbral c", "percentil 80 (interpolación lineal): posición 0.8 · 9 = 7.2 entre 0.027 y 0.031", f"c = 0.027 + 0.2 · (0.031 − 0.027) = {c_:.4f}"],
               ["Alertas", "puntajes mayores que c", f"{int(al.sum())} de 10: " + ", ".join(f"{v:.3f}" for v in pts[al])]],
              [34 * mm, 78 * mm, CONTENT_W - 112 * mm - 18]),
        P("Lectura: el umbral deja el 20 % superior por construcción, sin saber nada sobre defectos reales. Si una de esas alertas es una imagen perfectamente normal, es una <b>falsa alarma</b>; "
          "si un defecto real queda bajo el corte, es un <b>defecto no detectado</b>. Con el percentil 95 de la clase: c = 0.033108, 18 de 360 validaciones lo superan; en prueba, 35 de 360 originales (9.72 %) y 319 de 360 desplazadas (88.61 %). "
          "<b>Error común</b>: leer 0.058 como «5.8 % de probabilidad de defecto».", "cell")]))
    s.append(Spacer(1, 3))
    s.append(img("c46_umbral.png", width=CONTENT_W * 0.88))
    s.append(Spacer(1, 2))
    s.append(img("c46_estres.png", width=CONTENT_W * 0.84))
    s.append(Spacer(1, 2))
    s += para("<b>Cuándo puede fallar una alerta</b> (diapositiva 42): si hay variación habitual que el modelo no representó, avisará sobre casos aceptables; si hay anomalías en el entrenamiento y el decoder es muy flexible, puede aprender a reconstruirlas; "
              "si cambia la forma de adquirir las imágenes, el puntaje refleja el cambio técnico. Por eso se necesita una validación específica: definir los defectos relevantes, separar por lote o fecha, calibrar el umbral con datos pertinentes, "
              "evaluar en una reserva, examinar errores y prever revisión humana antes de automatizar (diapositiva 46).")
    s.append(recuerda(["Alerta: e(x) &gt; c; c se fija con datos pertinentes antes de mirar la prueba", "El percentil de un puntaje no es un nivel de confianza sobre defectos", "Una alerta no confirma un defecto; un puntaje no es una probabilidad"]))

    # glosario
    s.append(h("8", "Glosario rápido"))
    gl = [["Término", "Qué es"],
          ["Representación", "Descripción numérica de un objeto (64 intensidades, o 8 coordenadas latentes)."],
          ["Autoencoder", "Red entrenada para reproducir su entrada pasando por un resumen pequeño."],
          ["Encoder f, decoder g", "h = f(x) resume; x̂ = g(h) reconstruye; con parámetros θ y ψ."],
          ["Vector latente h", "Salida de la capa central; aquí 8 coordenadas entre −1 y 1 (tanh)."],
          ["Reconstrucción x̂", "Estimación de la entrada; mismo tamaño y significado que x."],
          ["MSE de reconstrucción", "Promedio de (x̂ⱼ − xⱼ)²; ℓ por imagen, L por conjunto."],
          ["Cuello de botella", "Restricción que obliga a resumir (aquí, 8 coordenadas); sin ella, la identidad copia todo."],
          ["PCA con k componentes", "Reconstrucción lineal con k números: x̂ = x̄ + (x − x̄)PₖPₖᵀ."],
          ["Capa latente y parámetros", "Latente = cuántos números resumen la imagen; parámetros = pesos y sesgos de toda la red."],
          ["Época, lote, Adam", "Pasada completa, grupo de 64 imágenes, optimizador (tasa 0.002)."],
          ["Puntaje de anomalía", "e(x) = MSE de la imagen; no es una probabilidad."],
          ["Umbral c", "Corte a partir del cual se alerta; aquí el percentil 95 de validación."],
          ["Falsa alarma", "Alerta sobre una imagen aceptable; el reverso es un defecto no detectado."]]
    s.append(table(gl, [52 * mm, CONTENT_W - 52 * mm - 18]))
    out, i = [], 0
    while i < len(s):
        if s[i] in HEADS:
            out.append(KeepTogether(s[i:i + 3]))
            i += 3
        else:
            out.append(s[i])
            i += 1
    build("apuntes/C46_Apuntes_Autoencoders.pdf", out, "Apuntes · Clase 46 · Autoencoders y representaciones")


# ======================================================================= CUADERNO
def bloque(n, title, nivel, intro, items):
    flow = [P(f'<font color="#6D28D9">Bloque {n}</font>  {title}  <font name="Sym" size="9" color="#6B7280">{nivel}</font>', "h2")]
    if intro:
        flow += [P(intro, "cell"), Spacer(1, 2)]
    flow.append(numbered([("", t) for t in items], C46))
    return [KeepTogether(flow)]


def cuaderno():
    s = []
    s.append(band("Ejercicios para el cuaderno · Clase 46 · Autoencoders", "Para resolver en papel, con calculadora · soluciones paso a paso al final", C46))
    s.append(Spacer(1, 4))
    s.append(box([P("<b>Cómo usar esta guía</b>", "cellb"),
                  bullets(["Escribe el número del bloque y del ejercicio, y tu desarrollo completo.",
                           "Fórmulas: h = f(x) · x̂ = g(h) · ℓ(x, x̂) = (1/p) Σⱼ (x̂ⱼ − xⱼ)² (p = número de píxeles) · parámetros por capa = unidades × (entradas + 1) · alerta si e(x) &gt; c · PCA: z = (x − x̄)P, x̂ = x̄ + zPᵀ.",
                           "Valores útiles: tanh(0.4) = 0.3799 · tanh(1) = 0.7616 · tanh(1.5) = 0.9051 · tanh(−0.5) = −0.4621 · tanh(−2) = −0.9640 · tanh(3) = 0.9951 · √2 = 1.4142.",
                           "Nivel: <font name='Sym'>★</font> directo · <font name='Sym'>★★</font> 2 o 3 pasos · <font name='Sym'>★★★</font> integra varios conceptos."], "cell", C46)], bg=SOFT, border=C46, pad=5))
    s.append(Spacer(1, 2))
    s += bloque(1, "Codificar y decodificar con un encoder fijo", "★", "Imágenes de 2 píxeles. El encoder fijo calcula h = (x₁ + x₂)/2 y el decoder fijo devuelve (h, h).", [
        "Para x = (0.1, 0.9), x = (0.6, 0.4) y x = (0.3, 0.3), calcula h, la reconstrucción x̂ y el MSE de cada imagen.",
        "¿Cuáles de las tres comparten vector latente? ¿Alguna se reconstruye sin error? ¿Por qué?",
        "Da dos entradas distintas, y distintas de las anteriores, que tengan el mismo h que x = (0.1, 0.9)."])
    s += bloque(2, "MSE por píxel y por imagen", "★★", "Referencia x = (0.0, 0.5, 1.0, 0.25) y reconstrucción x̂ = (0.2, 0.4, 0.7, 0.25).", [
        "Calcula la diferencia x̂ − x y su cuadrado en cada píxel.",
        "Calcula el MSE de la imagen. ¿Qué píxel domina?",
        "Una imagen de 64 píxeles tiene los 64 cuadrados con suma 1.1021 (ejemplo de la clase). ¿Cuánto vale su MSE? ¿Qué fracción de la suma aporta un píxel con cuadrado 0.1629?"])
    s += bloque(3, "tanh y salida lineal", "★", None, [
        "Calcula tanh(0), tanh(1) y tanh(−2).",
        "Una unidad latente tiene pesos (0.5, −1, 2), sesgo 0.2 y la entrada es x = (0.8, 0.4, 0.1). Calcula a y h = tanh(a).",
        "¿Puede una coordenada latente valer 1.3? ¿Y una salida lineal de la red? Una unidad de salida recibe una combinación s = 1.2: ¿qué intensidad devuelve?"])
    s += bloque(4, "Pasar una imagen por un autoencoder 3 → 2 → 3", "★★", "x = (1, 0, 0.5). Encoder: W₁ con columnas (1, 0, 1) y (0, 1, −1), sin sesgos, activación tanh. Decoder: x̂₁ = 0.5h₁, x̂₂ = 0.5h₂, x̂₃ = 0.5h₁ − h₂, salida lineal.", [
        "Calcula a = xW₁ y el vector latente h = tanh(a).",
        "Calcula la reconstrucción x̂ con cuatro decimales.",
        "Calcula el MSE de la imagen y cuenta los parámetros de la red (pesos y sesgos, aunque valgan 0)."])
    s += bloque(5, "Contar parámetros", "★★", None, [
        "Una red 64–8–64: ¿cuántos parámetros tiene? ¿Y una 64–16–64?",
        "Una red 64–32–8–32–64: ¿cuántos? ¿Qué parte del total está en las dos capas que tocan los 64 píxeles?",
        "Práctica de la diapositiva 53: con latente 4, ¿cuántos parámetros tienen 64–4–64 y 64–32–4–32–64?"])
    s += bloque(6, "Comparar con la imagen media y con PCA", "★★", "MSE de validación (diapositivas): imagen media 0.0729; PCA 8 0.0235; autoencoder 64–8–64 0.0240; autoencoder profundo 0.0193.", [
        "Calcula cuánto reduce cada método el error de la imagen media (en %).",
        "¿Cuántas veces mayor es el error de la imagen media que el del autoencoder profundo? ¿Y cuánto reduce el profundo el error de PCA 8 (en %)?",
        "PCA a mano: x̄ = (0.5, 0.5), P₁ = (1/√2, 1/√2), x = (0.3, 0.9). Calcula z, x̂ y el MSE; compáralo con el de la imagen media."])
    s += bloque(7, "Umbral y alertas", "★★", "Diez imágenes con su puntaje de reconstrucción y, entre paréntesis, si tienen un defecto real: A 0.012 (no), B 0.031 (sí), C 0.018 (no), D 0.052 (sí), E 0.024 (no), F 0.038 (no), G 0.015 (no), H 0.071 (sí), I 0.027 (no), J 0.020 (sí).", [
        "Con el umbral c = 0.030, ¿qué imágenes disparan una alerta? Cuenta alertas, falsas alarmas y defectos no detectados.",
        "Repite con c = 0.045. ¿Qué cambia? ¿Qué valor de c, tomado de la lista, deja exactamente el 20 % de las imágenes con alerta?",
        "La imagen D tiene puntaje 0.052. ¿Es 5.2 % de probabilidad de defecto? Explica."])
    s += bloque(8, "Pensar con evidencia", "★★★", "Un sistema de inspección usa un autoencoder entrenado con fotos de envases aceptados y rechaza el 5 % con mayor error. Se anuncia «detección de cualquier defecto con 95 % de confianza».", [
        "¿Qué demuestra el entrenamiento disponible y qué no?",
        "Nombra tres cosas que habría que evaluar antes de creer el anuncio.",
        "En la clase, el estrés desplazó cada imagen una columna y el 88.61 % superó el umbral. ¿Mide sensibilidad a defectos reales? Explica."])
    s.append(PageBreak())

    s.append(band("Soluciones paso a paso", "Revisa el desarrollo, no solo el número final", C46))
    s.append(Spacer(1, 3))

    def sol(title, rows):
        return [KeepTogether([P(title, "h3"), table([["#", "Solución"]] + rows, [10 * mm, CONTENT_W - 10 * mm], header_color=C46), Spacer(1, 2)])]

    def mse2(x, y):
        return float(np.mean((np.array(y) - np.array(x)) ** 2))

    # 1
    r1 = []
    for x in ((0.1, 0.9), (0.6, 0.4), (0.3, 0.3)):
        hh_ = (x[0] + x[1]) / 2
        r1.append((x, hh_, mse2(x, (hh_, hh_))))
    assert abs(r1[0][2] - 0.16) < 1e-12 and abs(r1[1][2] - 0.01) < 1e-12 and abs(r1[2][2]) < 1e-12
    s += sol("Bloque 1", [
        ["1", f"x = (0.1, 0.9): h = 0.5, x̂ = (0.5, 0.5), MSE = (0.4² + 0.4²)/2 = (0.16 + 0.16)/2 = {r1[0][2]:.2f}. x = (0.6, 0.4): h = 0.5, x̂ = (0.5, 0.5), MSE = (0.1² + 0.1²)/2 = {r1[1][2]:.2f}. x = (0.3, 0.3): h = 0.3, x̂ = (0.3, 0.3), MSE = 0."],
        ["2", "Las dos primeras comparten h = 0.5 (el decoder no puede distinguirlas). La tercera se reconstruye sin error porque sus dos píxeles son iguales: el decoder (h, h) justo repite el mismo valor."],
        ["3", "Cualquier par con suma 1 tiene h = 0.5: por ejemplo (0.2, 0.8), (0.5, 0.5) o (1.0, 0.0)."]])
    # 2
    xr_, xh_ = np.array([0.0, 0.5, 1.0, 0.25]), np.array([0.2, 0.4, 0.7, 0.25])
    d_ = xh_ - xr_
    q_ = d_ ** 2
    suma64, cuad = 1.1021, 0.1629
    s += sol("Bloque 2", [
        ["1", f"Diferencias: 0.2; −0.1; −0.3; 0.0. Cuadrados: {q_[0]:.2f}; {q_[1]:.2f}; {q_[2]:.2f}; {q_[3]:.2f}."],
        ["2", f"MSE = ({q_[0]:.2f} + {q_[1]:.2f} + {q_[2]:.2f} + {q_[3]:.2f}) ÷ 4 = {q_.sum():.2f} ÷ 4 = {q_.mean():.4f}. Domina el píxel 3 (cuadrado 0.09 de 0.14 en total)."],
        ["3", f"MSE = {suma64} ÷ 64 = {suma64 / 64:.6f} ≈ 0.01722. La fracción es {cuad} ÷ {suma64} = {cuad / suma64:.4f}, es decir, {cuad / suma64 * 100:.1f} % de la suma."]])
    # 3
    a3 = 0.5 * 0.8 - 1 * 0.4 + 2 * 0.1 + 0.2
    s += sol("Bloque 3", [
        ["1", f"tanh(0) = 0; tanh(1) = {tanh(1):.4f}; tanh(−2) = {tanh(-2):.4f}.".replace("-", "−")],
        ["2", f"a = 0.5·0.8 − 1·0.4 + 2·0.1 + 0.2 = 0.4 − 0.4 + 0.2 + 0.2 = {a3:.1f}; h = tanh({a3:.1f}) = {tanh(a3):.4f}."],
        ["3", "No para la coordenada latente: tanh siempre queda entre −1 y 1 (con tanh(3) = 0.9951 ya casi toca 1). Sí para una salida lineal: con s = 1.2 devuelve 1.2, fuera de [0, 1]; el MSE usa ese valor tal cual y solo la visualización lo recorta a 1."]])
    # 4
    a4 = np.array([1.0 * 1 + 0 * 0 + 0.5 * 1, 1.0 * 0 + 0 * 1 + 0.5 * (-1)])
    h4_ = np.tanh(a4)
    xo = np.array([0.5 * h4_[0], 0.5 * h4_[1], 0.5 * h4_[0] - h4_[1]])
    xv = np.array([1.0, 0.0, 0.5])
    d4 = (xo - xv) ** 2
    assert abs(a4[0] - 1.5) < 1e-12 and abs(a4[1] + 0.5) < 1e-12
    s += sol("Bloque 4", [
        ["1", f"a₁ = 1·1 + 0·0 + 0.5·1 = 1.5; a₂ = 1·0 + 0·1 + 0.5·(−1) = −0.5. h = (tanh 1.5; tanh(−0.5)) = ({h4_[0]:.4f}; {mn(h4_[1])})."],
        ["2", f"x̂₁ = 0.5·{h4_[0]:.4f} = {xo[0]:.4f}; x̂₂ = 0.5·({mn(h4_[1])}) = {mn(xo[1])}; x̂₃ = {xo[0]:.4f} − ({mn(h4_[1])}) = {xo[2]:.4f}."],
        ["3", f"Cuadrados: ({xo[0]:.4f} − 1)² = {d4[0]:.4f}; ({mn(xo[1])} − 0)² = {d4[1]:.4f}; ({xo[2]:.4f} − 0.5)² = {d4[2]:.4f}. MSE = {np.round(d4, 4).sum():.4f} ÷ 3 = {np.round(d4, 4).sum() / 3:.4f}. Parámetros: encoder 3·2 + 2 = 8; decoder 2·3 + 3 = 9; total 17."]])
    # 5
    p_8 = 64 * 8 + 8 + 8 * 64 + 64
    p_16 = 64 * 16 + 16 + 16 * 64 + 64
    capas = [64 * 32 + 32, 32 * 8 + 8, 8 * 32 + 32, 32 * 64 + 64]
    assert p_8 == 1096 and sum(capas) == 4744 and PA["ae4"] == 580 and PA["prof4"] == 4484
    s += sol("Bloque 5", [
        ["1", f"64–8–64: 8·65 + 64·9 = 520 + 576 = {mil(p_8)}. 64–16–64: 16·65 + 64·17 = 1 040 + 1 088 = {mil(p_16)}."],
        ["2", f"2 080 + 264 + 288 + 2 112 = {mil(sum(capas))}. Las capas 64 → 32 y 32 → 64 suman {mil(capas[0] + capas[3])}, es decir, {(capas[0] + capas[3]) / sum(capas) * 100:.1f} % del total."],
        ["3", f"64–4–64: 4·65 + 64·5 = 260 + 320 = {PA['ae4']}. 64–32–4–32–64: 2 080 + (32·4 + 4 = 132) + (4·32 + 32 = 160) + 2 112 = {mil(PA['prof4'])}."]])
    # 6
    red = [1 - 0.0235 / 0.0729, 1 - 0.0240 / 0.0729, 1 - 0.0193 / 0.0729]
    zq = ((0.3 - 0.5) + (0.9 - 0.5)) / sqrt(2)
    xq = np.array([0.5 + zq / sqrt(2), 0.5 + zq / sqrt(2)])
    eq = ((xq - np.array([0.3, 0.9])) ** 2).mean()
    em = ((np.array([0.5, 0.5]) - np.array([0.3, 0.9])) ** 2).mean()
    s += sol("Bloque 6", [
        ["1", f"PCA 8: 1 − 0.0235 ÷ 0.0729 = {red[0] * 100:.1f} %. Autoencoder 64–8–64: 1 − 0.0240 ÷ 0.0729 = {red[1] * 100:.1f} %. Profundo: 1 − 0.0193 ÷ 0.0729 = {red[2] * 100:.1f} %."],
        ["2", f"0.0729 ÷ 0.0193 = {0.0729 / 0.0193:.2f} veces. Respecto de PCA 8: 1 − 0.0193 ÷ 0.0235 = {(1 - 0.0193 / 0.0235) * 100:.1f} % (con las cifras sin redondear: {M['reduccion_prof_vs_pca'] * 100:.1f} %). Los dos autoencoders alcanzaron su mínimo en la época 150, así que no se puede afirmar que hayan convergido."],
        ["3", f"x − x̄ = (−0.2; 0.4); z = (−0.2 + 0.4) ÷ √2 = {zq:.4f}; x̂ = x̄ + z·P₁ = (0.5 + 0.1; 0.5 + 0.1) = ({xq[0]:.1f}; {xq[1]:.1f}). MSE de PCA = ((0.6 − 0.3)² + (0.6 − 0.9)²) ÷ 2 = {eq:.2f}. Imagen media: ((0.5 − 0.3)² + (0.5 − 0.9)²) ÷ 2 = (0.04 + 0.16) ÷ 2 = {em:.2f}. Aquí PCA con 1 componente apenas mejora a la media."]])
    # 7
    sc = {"A": 0.012, "B": 0.031, "C": 0.018, "D": 0.052, "E": 0.024, "F": 0.038, "G": 0.015, "H": 0.071, "I": 0.027, "J": 0.020}
    df = {"B", "D", "H", "J"}

    def cuenta(c):
        al_ = {k for k, v in sc.items() if v > c}
        return sorted(al_), len(al_), len(al_ - df), len(df - al_)
    a30, n30, f30, m30 = cuenta(0.030)
    a45, n45, f45, m45 = cuenta(0.045)
    assert (n30, f30, m30) == (4, 1, 1) and (n45, f45, m45) == (2, 0, 2)
    s += sol("Bloque 7", [
        ["1", f"Alertan {', '.join(a30)} (puntaje &gt; 0.030): {n30} alertas. Falsa alarma: F (sin defecto). Defecto no detectado: J (0.020). Aciertos: B, D y H."],
        ["2", f"Con c = 0.045 alertan {', '.join(a45)}: {n45} alertas, {f45} falsas alarmas y {m45} defectos no detectados (B y J). Subir el corte elimina la falsa alarma pero deja pasar un defecto más. Con c = 0.038 (&gt; estricto) alertan solo D y H: 2 de 10 = 20 %."],
        ["3", "No. 0.052 es el MSE de esa imagen, no una probabilidad: no está entre 0 y 1 por construcción ni se calibró contra defectos. Solo dice que la reconstrucción fue peor que el umbral."]])
    # 8
    s += sol("Bloque 8", [
        ["1", "Solo hay imágenes aceptadas por una cámara: el entrenamiento muestra qué se reconstruye bien, no que detecte defectos. El «5 % con mayor error» se fija por construcción y no mide qué fracción de defectos reales se captura."],
        ["2", "Por ejemplo: definir los defectos relevantes y el costo de rechazar; separar por lote, fecha o unidad de uso; calibrar el umbral con datos pertinentes y evaluar en una reserva; examinar los errores (falsas alarmas y defectos no detectados) y prever revisión humana (diapositiva 46)."],
        ["3", "No. Un desplazamiento artificial no es un defecto real: muestra que el puntaje responde a un cambio (el MSE se compara con la entrada desplazada), pero no entrega sensibilidad ni especificidad frente a anomalías reales (diapositiva 40). El percentil 95 tampoco es un nivel de confianza."]])
    build("apuntes/C46_Ejercicios_Cuaderno_Autoencoders.pdf", s, "Ejercicios para el cuaderno · Clase 46")


if __name__ == "__main__":
    (OUT / "apuntes").mkdir(exist_ok=True)
    guia()
    apuntes()
    cuaderno()
    print("OK")
