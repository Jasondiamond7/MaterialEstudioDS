import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, fila)

OUT = out("guion_c38_10_probabilidades.json")
v = V("Umbral, log loss, Brier y calibración")
CA = D["s_cal"]
PA = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]                                  # diapositiva 41, serie A
YA = [0, 0, 1, 0, 1, 1]
PB = [1 / (1 + math.exp(-4 * math.log(p / (1 - p)))) for p in PA]    # serie B reconstruida (reproduce log loss 0.602)

v.portada("Probabilidades", "Ordenar, estimar y decidir: tres objetivos distintos",
          "Décima parada: las probabilidades. Cómo un umbral las convierte en acciones, y cómo se mide si una probabilidad es buena.")

# ---- 1. Hook: tres preguntas
v.b("Un filtro asigna probabilidad de spam a tres correos. [[a]]Correo A: 0.90. Correo B: 0.60. Correo C: 0.20. [[b]]Con estos mismos tres números se pueden hacer tres preguntas distintas. ¿Qué reviso primero? ¿Es fiable ese 0.90? ¿Y cuáles bloqueo? [[c]]La diapositiva treinta y nueve insiste: ordenar bien, estimar bien una probabilidad, y decidir bien, son objetivos distintos.",
    CL(), T("tres preguntas para las mismas probabilidades · diapositiva 39", 960, 140, tam=46, c="tinta", ent="subir"),
    P("A: 0.90", 560, 300, "rojo", 50, en="a"), P("B: 0.60", 960, 300, "ambar", 50, en="a", t=0.3), P("C: 0.20", 1360, 300, "azul", 50, en="a", t=0.6),
    tarjetas([(400, "violeta", "¿qué reviso primero?", "ordenar: A, luego B, luego C", "b"),
              (960, "teal", "¿es fiable el 0.90?", "estimar: ¿cerca de 90 % son spam?", "b"),
              (1520, "ambar", "¿cuáles bloqueo?", "decidir: depende del umbral", "b")], y=560, w=500, h=280, tam_t=40, tam_d=34),
    P("ordenar ≠ estimar ≠ decidir", 960, 830, "violeta", 50, en="c"), diapo(39, en="a"), sec="1. Tres preguntas")

# ---- 2. Analogía: el pronóstico del tiempo
v.b("Piensa en el pronóstico del tiempo. [[a]]Si la meteoróloga ordena bien los días, los días que más llueve son los que ella marcó con más riesgo. Eso es ordenar. [[b]]Si dice «setenta por ciento de lluvia», entre todos los días en que dijo eso, debería llover más o menos siete de cada diez. Eso es estar calibrada. [[c]]Y si sacas o no el paraguas depende de cuánto te molesta mojarte. Eso es decidir, y depende de tus costos.",
    CL(), T("el pronóstico del tiempo", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(400, "violeta", "ordenar", "más riesgo → más lluvia", "a"), (960, "teal", "calibrar", "«70 %» → llueve ~7 de cada 10", "b"),
              (1520, "ambar", "decidir", "¿paraguas? depende de tus costos", "c")], y=480, w=500, h=320, tam_t=48, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el pronóstico")

# ---- 3. El umbral
PU = [0.1, 0.4, 0.6, 0.8, 0.9]
GX0, GX1, GY0, GY1 = 520, 1400, 260, 820


def bx(i):
    return GX0 + (i + 0.5) / 5 * (GX1 - GX0)


def by(p):
    return GY1 - p * (GY1 - GY0)


v.b("La diapositiva cuarenta muestra cómo se decide. [[a]]Cinco correos, con probabilidades 0.1, 0.4, 0.6, 0.8 y 0.9. [[b]]La regla: se marca como spam si la probabilidad es mayor o igual que el umbral t. Con t igual a 0.5, se marcan tres correos. [[c]]Con t igual a 0.8, se marcan dos: el de 0.8 también cuenta, porque la regla dice mayor o igual.",
    CL(), T("el umbral convierte una probabilidad en una acción · diapositiva 40", 960, 140, tam=44, c="tinta", ent="subir"),
    L(GX0, GY1, GX1, GY1, "tinta", 4), L(GX0, GY0, GX0, GY1, "tinta", 4),
    *[T(f"{p:.1f}", GX0 - 16, by(p), tam=26, c="gris", anc="rm") for p in (0, 0.5, 1.0)],
    *[x for i, p in enumerate(PU) for x in [Bx(bx(i) - 60, by(p), bx(i) + 60, GY1, "azul", None, 0, 8, grow="y", en="a", t=0.12 * i, id=f"bu{i}"),
                                            T(f"{p:.1f}", bx(i), by(p) - 26, tam=30, c="tinta", w="b", en="a", t=0.12 * i + 0.3),
                                            T(f"correo {i + 1}", bx(i), GY1 + 34, tam=26, c="gris", w="sb", en="a")]],
    L(GX0, by(0.5), GX1 + 40, by(0.5), "ambar", 4, guion=True, en="b"), T("t = 0.5 → 3 marcados", GX1 + 60, by(0.5), tam=34, c="ambar", w="sb", anc="lm", en="b", t=0.4),
    *[U(f"bu{i}", relleno="ambar", en="b", t=0.6, dur=0.4) for i in (2, 3, 4)],
    L(GX0, by(0.8), GX1 + 40, by(0.8), "rojo", 4, guion=True, en="c"), T("t = 0.8 → 2 marcados", GX1 + 60, by(0.8), tam=34, c="rojo", w="sb", anc="lm", en="c", t=0.4),
    U("bu2", relleno="azul", en="c", t=0.6, dur=0.4), *[U(f"bu{i}", relleno="rojo", en="c", t=0.6, dur=0.4) for i in (3, 4)],
    T("ŷ = 1  si  p ≥ t", 960, 940, tam=40, c="tinta", w="sb", en="b"), diapo(40, en="a"), sec="3. El umbral")
v.b("Fíjate en lo que cambió y en lo que no. [[a]]Las probabilidades son las mismas; el modelo es el mismo. [[b]]Lo que cambió es la decisión. Subir el umbral exige más evidencia para bloquear: pueden disminuir los falsos positivos, y aumentar los falsos negativos. [[c]]Por eso la diapositiva concluye: el umbral también se selecciona.",
    CL(), T("mismo modelo, otra decisión", 960, 150, tam=56, c="tinta", ent="subir"),
    P("las probabilidades no cambian", 960, 300, "gris", 48, en="a"),
    T("subir t → menos FP, más FN (posiblemente)", 960, 460, tam=46, c="tinta", w="sb", en="b"),
    P("el umbral también se selecciona", 960, 640, "violeta", 52, en="c"), diapo(40, en="a"))

# ---- 4. Ordenar igual no es estimar igual
GA = G(400, 250, 1400, 800, 0.5, 6.5, -0.05, 1.05, [dict(xs=[1, 2, 3, 4, 5, 6], ys=PA, color="azul", grosor=5, marcadores=10, revela=0.0),
                                                    dict(xs=[1, 2, 3, 4, 5, 6], ys=[round(p, 3) for p in PB], color="ambar", grosor=5, marcadores=10, revela=0.0)],
       xticks=[], yticks=[0, 0.5, 1.0], yfmt="{:.1f}", ylabel="p", id="g41", tam_txt=28)
v.b("Ahora, la diapositiva cuarenta y uno. [[a]]Seis correos, con su etiqueta real abajo. La serie A, en azul, les da probabilidades de 0.1 a 0.9. [[b]]La serie B, en naranja, los ordena exactamente igual, pero es mucho más extrema: casi cero en los primeros, casi uno en los últimos. [[c]]Mira el correo cuatro: no es spam, y B le da 0.835. B expresa más certeza, incluso cuando se equivoca.",
    CL(), T("ordenar igual no significa estimar igual · diapositiva 41", 960, 140, tam=46, c="tinta", ent="subir"),
    GA, *[T(f"y={y}", 400 + (i + 0.5) / 6 * 1000, 840, tam=28, c="rojo" if y else "gris", w="sb") for i, y in enumerate(YA)],
    U("g41", **{"series.0.revela": 1.0}, en="a", dur=1.2), T("A", 1440, 360, tam=44, c="azul", anc="lm", en="a", t=0.8),
    U("g41", **{"series.1.revela": 1.0}, en="b", dur=1.2), T("B", 1440, 270, tam=44, c="ambar", anc="lm", en="b", t=0.8),
    R(400 + 3.5 / 6 * 1000, 800 - (PB[3] + 0.05) / 1.1 * 550, 90, 90, "rojo", r=45, en="c"),
    T("correo 4: y = 0, B dice 0.835", 1430, 520, tam=28, c="rojo", w="sb", anc="lm", en="c", t=0.4),
    diapo(41, en="a"), etiqueta("serie B reconstruida: reproduce el log loss 0.602 de la diapositiva", 1010, en="b"), sec="4. Ordenar no es estimar")
v.b("Si solo miráramos el orden, A y B serían idénticas. [[a]]Pero sus probabilidades no valen lo mismo. [[b]]La diapositiva les asigna un log loss: 0.415 para A, y 0.602 para B. [[c]]Menor es mejor: A estima mejor, aunque ordene igual. La calidad de esa certeza requiere otra evaluación. Veamos qué es el log loss.",
    CL(), T("mismo orden, distinta calidad", 960, 150, tam=56, c="tinta", ent="subir"),
    P("el orden es idéntico", 960, 300, "gris", 48, en="a"),
    T("log loss A", 560, 440, tam=40, c="azul", w="sb", en="b"), N(0.415, 560, 530, "{:.3f}", 84, "azul", en="b", t=0.3),
    T("log loss B", 1360, 440, tam=40, c="ambar", w="sb", en="b", t=0.8), N(0.602, 1360, 530, "{:.3f}", 84, "ambar", en="b", t=1.1),
    P("menor es mejor: A estima mejor", 960, 730, "teal", 48, en="c"), diapo(41, en="a"))

# ---- 5. Log loss
PX = [i / 100 for i in range(2, 99, 2)]
v.b("El log loss, o pérdida logarítmica, está en la diapositiva cuarenta y dos. [[a]]Para cada correo se toma el logaritmo natural de la probabilidad que el modelo le dio a lo que realmente ocurrió, con signo menos, y se promedia. [[b]]En el gráfico: si el correo es spam, la pérdida es la curva roja, que se dispara cuando la probabilidad se acerca a cero. [[c]]Para un spam real, predecir 0.9 aporta 0.105; predecir 0.1 aporta 2.303. Asignar casi cero a lo que ocurrió recibe una penalización enorme.",
    CL(), T("log loss: penalizar probabilidades equivocadas · diapositiva 42", 960, 140, tam=46, c="tinta", ent="subir"),
    caja_formula("LogLoss = −(1/n) · Σ [ yᵢ·ln pᵢ + (1 − yᵢ)·ln(1 − pᵢ) ]", y=270, c="violeta", tam=42, w=1500, en="a"),
    G(480, 400, 1300, 900, 0, 1, 0, 4, [dict(xs=PX, ys=[-math.log(p) for p in PX], color="rojo", grosor=6, revela=0.0),
                                        dict(xs=PX, ys=[-math.log(1 - p) for p in PX], color="azul", grosor=6, revela=0.0)],
      xticks=[0, 0.5, 1], yticks=[0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="probabilidad predicha de spam", ylabel="pérdida", id="gl", tam_txt=26, en="b"),
    U("gl", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="b", t=0.3, dur=1.5),
    T("real: spam (y = 1)", 1340, 480, tam=32, c="rojo", w="sb", anc="lm", en="b", t=1.0), T("real: no spam (y = 0)", 1340, 540, tam=32, c="azul", w="sb", anc="lm", en="b", t=1.2),
    T("p = 0.9 → −ln 0.9 = 0.105", 1340, 680, tam=34, c="tinta", w="sb", anc="lm", en="c"), T("p = 0.1 → −ln 0.1 = 2.303", 1340, 750, tam=34, c="rojo", w="sb", anc="lm", en="c", t=1.0),
    diapo(42, en="a"), sec="5. El log loss")
v.b("Calculemos el log loss de la serie A, correo por correo. [[a]]Los correos uno y seis aportan 0.105 cada uno. Los correos dos y cinco, 0.223 cada uno. [[b]]Y los correos tres y cuatro, donde A dudaba, 0.4 para un spam y 0.6 para un legítimo, aportan 0.916 cada uno. [[c]]La suma es 2.488; dividida por seis, 0.415. El mismo número de la diapositiva.",
    CL(), T("el log loss de A, a mano", 960, 140, tam=54, c="tinta", ent="subir"),
    fila(270, [(f"correo {i + 1}", 330 + 252 * i) for i in range(6)], tams=[30] * 6, colores=["gris"] * 6, x0=200, x1=1720),
    fila(340, [(f"p = {p:.1f}, y = {y}", 330 + 252 * i) for i, (p, y) in enumerate(zip(PA, YA))], tams=[28] * 6, colores=["tinta"] * 6, x0=200, x1=1720),
    fila(440, [("0.105", 330), ("", 582), ("", 834), ("", 1086), ("", 1338), ("0.105", 1590)], tams=[44] * 6, colores=["verde"] * 6, en="a",
         x0=200, x1=1720),
    fila(440, [("", 330), ("0.223", 582), ("", 834), ("", 1086), ("0.223", 1338), ("", 1590)], tams=[44] * 6, colores=["teal"] * 6, en="a", t=1.2,
         x0=200, x1=1720),
    fila(440, [("", 330), ("", 582), ("0.916", 834), ("0.916", 1086), ("", 1338), ("", 1590)], tams=[44] * 6, colores=["rojo"] * 6, en="b",
         x0=200, x1=1720),
    T("(0.105 + 0.223 + 0.916 + 0.916 + 0.223 + 0.105) / 6 = 2.488 / 6 =", 1300, 600, tam=38, c="tinta", w="sb", anc="rm", en="c"),
    N(0.415, 1330, 600, "{:.3f}", 64, "azul", anc="lm", en="c", t=2.0), diapo(41, en="a"))

# ---- 6. Brier
v.b("La otra métrica para probabilidades es la de Brier, en la diapositiva cuarenta y tres. [[a]]Es el error cuadrático de una probabilidad: el promedio de la etiqueta real menos la probabilidad, al cuadrado. Está entre cero y uno, y menor es mejor. [[b]]Para un spam real: predecir 0.9 aporta 0.01; predecir 0.6, 0.16; predecir 0.1, 0.81. [[c]]Compara con el log loss: para el 0.1, el log loss daba 2.303, y Brier da 0.81. El log loss castiga mucho más la certeza equivocada.",
    CL(), T("Brier: el error cuadrático de una probabilidad · diapositiva 43", 960, 140, tam=46, c="tinta", ent="subir"),
    caja_formula("Brier = (1/n) · Σ (yᵢ − pᵢ)²", y=290, c="teal", tam=52, w=1000, en="a"),
    *[x for i, (p, b) in enumerate([(0.9, 0.01), (0.6, 0.16), (0.1, 0.81)])
      for x in [Bx(560 + 400 * i - 90, 840 - b * 400, 560 + 400 * i + 90, 840, "teal", None, 0, 10, grow="y", en="b", t=0.4 * i),
                T(f"{b:.2f}", 560 + 400 * i, 840 - b * 400 - 34, tam=42, c="teal", w="b", en="b", t=0.4 * i + 0.3),
                T(f"p = {p}", 560 + 400 * i, 880, tam=32, c="gris", w="sb", en="b", t=0.4 * i)]],
    T("siendo y = 1", 960, 930, tam=30, c="gris", w="sb", en="b"),
    T("p = 0.1:  log loss 2.303  ·  Brier 0.81", 960, 410, tam=34, c="rojo", w="sb", en="c"), diapo(43, en="a"), sec="6. Brier")
v.b("Y para la serie A completa: [[a]]los cuadrados son 0.01, 0.04, 0.36, 0.36, 0.04 y 0.01. [[b]]Suman 0.82, y divididos por seis, dan 0.137. [[c]]Una advertencia de la diapositiva: la pérdida logarítmica y la de Brier valoran probabilidades, con penalizaciones distintas, y ninguno de los dos mide solamente la calibración.",
    CL(), T("el Brier de A, a mano", 960, 150, tam=54, c="tinta", ent="subir"),
    T("0.1²  0.2²  0.6²  0.6²  0.2²  0.1²", 960, 290, tam=46, c="gris", w="sb", en="a"),
    T("0.01 + 0.04 + 0.36 + 0.36 + 0.04 + 0.01 = 0.82", 960, 390, tam=46, c="tinta", w="sb", en="a", t=1.0),
    T("Brier = 0.82 / 6 =", 1000, 520, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(0.137, 1030, 520, "{:.3f}", 66, "teal", anc="lm", en="b", t=0.6),
    P("ninguno mide SOLO calibración", 960, 700, "ambar", 48, en="c"), adicional(1050, en="a"))

# ---- 7. Calibración
v.b("Entonces, ¿qué es la calibración? Diapositiva cuarenta y cuatro. [[a]]Si tomamos todos los correos con probabilidades cercanas a 0.8, esperamos que cerca del 80 por ciento sean spam. [[b]]La calibración compara las probabilidades predichas con las frecuencias observadas, en grupos de casos parecidos. [[c]]No significa que cada grupo de diez tenga exactamente ocho spam: la muestra introduce variación. En la clase se dijo así: con probabilidad 0.8, «esperamos que cerca del 80 por ciento» sean spam, «si las probabilidades están bien calibradas».",
    CL(), T("qué debería significar una probabilidad de 0.8 · diapositiva 44", 960, 140, tam=46, c="tinta", ent="subir"),
    *[Bc(300 + 92 * i, 330, 76, 76, "rojoclaro" if i < 8 else "azulclaro", "rojo" if i < 8 else "azul", 3, 14, en="a", t=0.06 * i) for i in range(10)],
    T("p ≈ 0.8 → ~8 de cada 10 son spam", 1500, 330, tam=36, c="tinta", w="sb", en="a", t=0.8),
    P("probabilidades predichas vs. frecuencias observadas", 960, 500, "teal", 46, en="b"),
    T("no exactamente 8 de cada 10: la muestra varía", 960, 620, tam=40, c="gris", w="sb", en="c"),
    cita("esperamos que cerca del 80 % de spam, si las probabilidades están bien calibradas", 770, en="c", t=1.5, tam=38),
    etiqueta("grabación de la clase", 850, en="c", t=1.8), diapo(44, en="a"), sec="7. Calibración")
v.b("Con datos reales, adelanto lo que veremos con el bosque aleatorio de la clase. [[a]]Agrupamos sus probabilidades fuera de pliegue en cinco grupos de 736 correos. [[b]]En el cuarto grupo, la probabilidad media predicha es 0.673. [[c]]Pero la fracción observada de spam es 0.814. El modelo dice sesenta y siete por ciento, y en realidad son ochenta y uno: en ese rango, subestima.",
    CL(), T("calibración del bosque, cuarto quintil", 960, 150, tam=52, c="tinta", ent="subir"),
    T("predice (media)", 560, 340, tam=40, c="gris", w="sb", en="b"), N(CA[3]["p"], 560, 440, "{:.3f}", 90, "azul", en="b", t=0.3),
    T("observado (fracción spam)", 1360, 340, tam=40, c="gris", w="sb", en="c"), N(CA[3]["f"], 1360, 440, "{:.3f}", 90, "rojo", en="c", t=0.3),
    T("5 grupos de 736 correos", 960, 600, tam=38, c="gris", w="sb", en="a"),
    P("en ese rango, subestima", 960, 740, "ambar", 50, en="c", t=1.5), real(1050, en="a"))

# ---- 8. El error típico
v.b("El error típico: [[a]]suponer que un modelo que ordena bien, también da probabilidades confiables. [[b]]La serie B ordenaba perfecto, y aun así sus probabilidades valían menos que las de A. [[c]]Ordenar, estimar y decidir se evalúan por separado: el orden, con curvas como la ROC; la estimación, con la pérdida logarítmica, el Brier y la calibración; y la decisión, con el costo.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("ordena bien → probabilidades confiables", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("B ordenaba perfecto, y estimaba peor", 960, 420, tam=44, c="tinta", w="sb", en="b"),
    fila(580, [("ordenar", 400), ("estimar", 960), ("decidir", 1520)], tams=[44] * 3, colores=["violeta", "teal", "ambar"], en="c"),
    fila(670, [("ROC", 400), ("log loss · Brier · calibración", 960), ("costo", 1520)], tams=[36] * 3, colores=["tinta"] * 3, en="c", t=0.6),
    sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un correo es spam, y el modelo le da probabilidad 0.5: no sabe. [[b]]¿Cuánto aporta ese correo al log loss? ¿Y al Brier?",
    CL(), T("y = 1   ·   p = 0.5", 960, 300, tam=70, c="tinta", w="b", en="a"),
    P("log loss = ?", 700, 500, "violeta", 52, en="b"), P("Brier = ?", 1220, 500, "teal", 52, en="b", t=0.4), sec="9. Piensa un segundo", pausa=0.6)
v.b("Al log loss aporta menos el logaritmo natural de 0.5, [[a]]0.693. [[b]]Al Brier, uno menos 0.5, al cuadrado: [[c]]0.25. Ese es el aporte de no saber: un modelo que siempre dijera 0.5 tendría exactamente esos valores.",
    CL(), T("log loss: −ln 0.5 =", 1000, 320, tam=54, c="tinta", w="sb", anc="rm"), N(0.693, 1030, 320, "{:.3f}", 70, "violeta", anc="lm", en="a"),
    T("Brier: (1 − 0.5)² =", 1000, 470, tam=54, c="tinta", w="sb", anc="rm", en="b"), N(0.25, 1030, 470, "{:.2f}", 70, "teal", anc="lm", en="c"),
    T("el costo de «no saber»: un modelo que siempre dice 0.5", 960, 640, tam=40, c="gris", w="sb", en="c", t=1.0), ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]el umbral convierte probabilidades en acciones, y también se selecciona. [[b]]La pérdida logarítmica y el Brier valoran la probabilidad completa; el log loss castiga mucho más la certeza equivocada. [[c]]Y la calibración pregunta si un 0.8 significa, de verdad, cerca de un 80 por ciento.",
    CL(), P("umbral: probabilidad → acción (y se selecciona)", 960, 300, "ambar", 44, en="a"),
    P("log loss y Brier: la probabilidad completa", 960, 460, "violeta", 46, en="b"),
    P("calibración: ¿0.8 significa 80 %?", 960, 620, "teal", 48, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]las curvas ROC y de precisión y recall, y el protocolo completo con el que la clase elige el modelo de spam.",
    CL(), siguiente(11, "ROC, precisión y el protocolo del spam"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
