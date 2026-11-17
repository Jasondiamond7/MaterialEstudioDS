import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_04_independencia.json")
v = V("El supuesto naive")
PT = D["bayes"]["patrones"]

v.portada("El supuesto naive", "Independencia dentro de cada clase y el primer cálculo completo",
          "Cuarta parada: el supuesto que le da nombre al método. Con él vamos a calcular, de principio a fin, la probabilidad de spam del correo con oferta y sin enlace.")

# ---- 1. El supuesto
v.b("La diapositiva catorce plantea el supuesto. [[a]]Dentro de los spam, y por separado dentro de los legítimos, las señales se tratan como independientes. [[b]]Eso permite multiplicar sus probabilidades: la probabilidad de oferta y sin enlace, dado spam, es la probabilidad de oferta dado spam por la de sin enlace dado spam. [[c]]Se repite el producto dentro de los legítimos, y con más variables se agrega un factor por variable.",
    CL(), T("Naive Bayes combina señales dentro de cada clase · diapositiva 14", 960, 140, tam=46, c="tinta", ent="subir"),
    P("dentro de cada clase, las señales se tratan como independientes", 960, 290, "teal", 44, en="a"),
    caja_formula("P(oferta, sin enlace | spam) = P(oferta | spam) · P(sin enlace | spam)", y=470, c="violeta", tam=44, w=1700, en="b"),
    T("lo mismo dentro de los legítimos  ·  con más variables, un factor por variable", 960, 660, tam=38, c="tinta", w="sb", en="c"),
    diapo(14, en="a"), sec="1. El supuesto")
v.b("Fíjate en las palabras: independencia condicional, es decir, dentro de cada clase. [[a]]No dice que oferta y enlace sean independientes en general. De hecho no lo son: los dos son más frecuentes en el spam. [[b]]Dice que, una vez que sabemos la clase, saber que hay oferta no cambia la probabilidad de que haya enlace.",
    CL(), T("independencia condicional: dentro de cada clase", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "en general", "oferta y enlace van juntos: los dos abundan en el spam", "a"),
              (1300, "verde", "dentro de una clase", "conocida la clase, una señal no informa sobre la otra", "b")],
             y=480, w=600, h=340, tam_t=48, tam_d=38))

# ---- 2. Por qué ingenuo
v.b("Esta simplificación es la que le da al método el apellido de ingenuo. [[a]]Hace posible el cálculo, pero no demuestra que las señales sean independientes en los datos. [[b]]El profesor lo dijo con claridad: nada asegura que la probabilidad conjunta se pueda separar así. [[c]]Es un primer acercamiento que funciona relativamente bien, y en la parada diez veremos qué pasa cuando el supuesto falla.",
    CL(), cita("nada me asegura eso, por eso es una metodología que es, digamos, ingenua", y=280, tam=44),
    P("hace posible el cálculo", 620, 500, "verde", 48, en="a"), P("no demuestra independencia", 1300, 500, "rojo", 48, en="a"),
    P("parada 10: qué pasa cuando el supuesto falla", 960, 700, "ambar", 44, en="c"), grabacion("00:43:24"), sec="2. Por qué ingenuo")

# ---- 3. Analogía
v.b("Una analogía para el supuesto. [[a]]Imagina dos testigos de un mismo hecho. Si cada uno lo vio por su cuenta, sus testimonios se suman: dos pruebas independientes. [[b]]Pero si el segundo solo repite lo que le contó el primero, en realidad hay una sola prueba. [[c]]El clasificador bayesiano ingenuo trata a todas las señales como testigos independientes. Cuando dos señales se repiten, el modelo cuenta dos veces la misma prueba.",
    CL(), T("dos testigos", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(620, "verde", "cada uno lo vio", "dos pruebas independientes", "a"), (1300, "rojo", "uno repite al otro", "una sola prueba, contada dos veces", "b")],
             y=450, w=560, h=280, tam_t=48, tam_d=38),
    P("Naive Bayes trata a todas las señales como testigos independientes", 960, 740, "violeta", 42, en="c"), analogia(1050), sec="3. Analogía: dos testigos")

# ---- 4. El patrón observado
v.b("Calculemos el patrón que observamos, como en la diapositiva quince. [[a]]Dentro de los spam: oferta, dieciséis de veinte; sin enlace, veinte menos catorce, seis de veinte. El producto: 0.80 por 0.30, 0.24. [[b]]Dentro de los legítimos: oferta, ocho de ochenta; sin enlace, ochenta menos dieciséis, sesenta y cuatro de ochenta. El producto: 0.10 por 0.80, 0.08. [[c]]Son cálculos bajo el supuesto; no son conteos observados del patrón completo.",
    CL(), T("calcular el patrón que realmente observamos · diapositiva 15", 960, 140, tam=48, c="tinta", ent="subir"),
    T("P(oferta, sin enlace | spam) = 16/20 × (20 − 14)/20 = 0.80 × 0.30 =", 1440, 330, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(0.24, 1470, 330, "{:.2f}", 60, "rojo", anc="lm", en="a", t=2.0),
    T("P(oferta, sin enlace | legítimo) = 8/80 × (80 − 16)/80 = 0.10 × 0.80 =", 1440, 480, tam=40, c="tinta", w="sb", anc="rm", en="b"),
    N(0.08, 1470, 480, "{:.2f}", 60, "azul", anc="lm", en="b", t=2.0),
    P("cálculo bajo independencia condicional; no son conteos del patrón completo", 960, 650, "ambar", 38, en="c"),
    diapo(15, en="a"), sec="4. El patrón observado")

# ---- 5. Los factores
v.b("Para clasificar, falta incorporar la frecuencia de cada clase. La diapositiva dieciséis prepara todos los factores. [[a]]Legítimo: frecuencia de clase 0.80, oferta presente 0.10, enlace ausente 0.80. [[b]]Spam: frecuencia de clase 0.20, oferta presente 0.80, enlace ausente 0.30. [[c]]En cada columna buscamos la probabilidad de lo que efectivamente observamos en el correo: la oferta presente y el enlace ausente.",
    CL(), T("preparar los factores del correo nuevo · diapositiva 16", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["clase", "frecuencia de clase", "oferta presente", "enlace ausente"],
           ["legítimo", "80/100 = 0.80", "8/80 = 0.10", "(80 − 16)/80 = 0.80"],
           ["spam", "20/100 = 0.20", "16/20 = 0.80", "(20 − 14)/20 = 0.30"]],
          [330, 760, 1170, 1580], 300, paso=120, tam=38, x0=130, x1=1790,
          colores=[["tinta"] * 4, ["azul"] + ["tinta"] * 3, ["rojo"] + ["tinta"] * 3], ens=[None, "a", "b"]),
    P("en cada columna: la probabilidad de lo que observamos", 960, 700, "violeta", 44, en="c"), diapo(16, en="a"), sec="5. Los factores")
v.b("Mira la última columna. [[a]]La ausencia de enlace es más frecuente entre los legítimos: sesenta y cuatro de ochenta, frente a seis de veinte spam. [[b]]Es decir, no ver el enlace es una pista a favor de legítimo. Esa pista va a pesar en el resultado.",
    R(1580, 480, 430, 230, "ambar", r=20, en="a"),
    T("sin enlace: 64 de 80 legítimos  ·  6 de 20 spam", 960, 840, tam=44, c="tinta", w="b", en="a", t=0.8),
    P("no ver el enlace es una pista a favor de legítimo", 960, 950, "azul", 42, en="b"))

# ---- 6. Alerta
v.b("Aquí, una pequeña precisión sobre la grabación. [[a]]Al repasar esta tabla, cerca del minuto cincuenta, se dijo: seis correos que no tienen enlace dentro de los veinte que son legítimos. [[b]]Los seis sin enlace están entre los veinte spam. Entre los legítimos, que son ochenta, hay sesenta y cuatro sin enlace.",
    CL(), error_material("00:49:54", "6 correos que no tienen enlace dentro de los 20 que son legítimos",
                         "6 sin enlace entre los 20 spam  ·  64 sin enlace entre los 80 legítimos", en_ok="b"),
    sec="6. Una precisión")

# ---- 7. Los productos
v.b("Ahora multiplicamos, como en la diapositiva diecisiete. [[a]]El producto para legítimo, que llamamos ese cero: 0.80 por 0.10 por 0.80, igual a 0.064. [[b]]El producto para spam, ese uno: 0.20 por 0.80 por 0.30, igual a 0.048. [[c]]Y normalizamos: la probabilidad de spam dado oferta y sin enlace es 0.048 dividido por 0.064 más 0.048: aproximadamente 0.4286.",
    CL(), T("oferta presente, enlace ausente · diapositiva 17", 960, 140, tam=50, c="tinta", ent="subir"),
    T("s0 = 0.80 × 0.10 × 0.80 =", 1000, 300, tam=52, c="azul", w="sb", anc="rm", en="a"), N(0.064, 1030, 300, "{:.3f}", 66, "azul", anc="lm", en="a", t=1.5),
    T("s1 = 0.20 × 0.80 × 0.30 =", 1000, 430, tam=52, c="rojo", w="sb", anc="rm", en="b"), N(0.048, 1030, 430, "{:.3f}", 66, "rojo", anc="lm", en="b", t=1.5),
    T("P(spam | oferta, sin enlace) = 0.048 / (0.064 + 0.048) ≈", 1240, 600, tam=44, c="tinta", w="sb", anc="rm", en="c"),
    N(0.4286, 1270, 600, "{:.4f}", 70, "violeta", anc="lm", en="c", t=2.5), diapo(17, en="a"), sec="7. Los productos")
v.b("Dos advertencias de la diapositiva. [[a]]Los productos ese cero y ese uno todavía no son probabilidades de clase del correo: primero hay que normalizarlos. [[b]]Y fíjate en el resultado: aunque el correo trae oferta, la probabilidad de spam queda por debajo de un medio. La ausencia del enlace pesó a favor de legítimo.",
    P("s0 y s1 aún no son probabilidades: hay que normalizar", 960, 780, "ambar", 42, en="a"),
    P("0.4286 < 0.5: la ausencia de enlace pesó a favor de legítimo", 960, 900, "azul", 42, en="b"))

# ---- 8. Qué cambia cuando vemos el enlace
v.b("La diapositiva dieciocho compara tres situaciones. [[a]]Si solo miramos oferta: 0.16 sobre 0.08 más 0.16, que es 0.6667. [[b]]Si vemos oferta y además vemos que no hay enlace: 0.4286. [[c]]Y si vemos oferta y enlace, los dos presentes: el producto del spam es 0.20 por 0.80 por 0.70, igual a 0.112; el del legítimo, 0.80 por 0.10 por 0.20, igual a 0.016. El resultado: 0.112 sobre 0.128, que es 0.8750.",
    CL(), T("¿qué cambia cuando vemos el enlace? · diapositiva 18", 960, 140, tam=50, c="tinta", ent="subir"),
    tabla([["lo que observamos", "cálculo de la probabilidad de spam", "resultado"], ["solo oferta", "0.16 / (0.08 + 0.16)", "0.6667"],
           ["oferta, sin enlace", "0.048 / (0.064 + 0.048)", "0.4286"], ["oferta y enlace", "0.112 / (0.016 + 0.112)", "0.8750"]],
          [430, 1000, 1530], 290, paso=110, tam=42, x0=170, x1=1760,
          colores=[["tinta"] * 3, ["tinta", "tinta", "violeta"], ["azul", "tinta", "azul"], ["rojo", "tinta", "rojo"]], ens=[None, "a", "b", "c"]),
    T("spam con ambas: 0.20 × 0.80 × 0.70 = 0.112   ·   legítimo con ambas: 0.80 × 0.10 × 0.20 = 0.016", 960, 790, tam=34, c="tinta", w="sb", en="c", t=2.0),
    diapo(18, en="a"), sec="8. Ver o no ver el enlace")
v.b("La lectura de la diapositiva es fina. [[a]]No observar el enlace no es lo mismo que desconocer si existe. Cuando sabemos que no está, la probabilidad baja; cuando sabemos que está, sube. [[b]]Y con un umbral de 0.50, la ausencia del enlace cambia la decisión: con solo oferta sería spam; con oferta y sin enlace, legítimo.",
    CL(), P("saber que NO está ≠ no saber si está", 960, 320, "violeta", 54, en="a"),
    G(360, 450, 1560, 880, 0, 3, 0, 1, [dict(xs=[0.25, 2.95], ys=[0.5, 0.5], color="ambar", grosor=4, linea=True)],
      yticks=[0, 0.5, 1.0], yfmt="{:.1f}", id="g4u", tam_txt=26, en="b"),
    Bx(500, 880 - 0.6667 * 430, 740, 880, "violeta", None, 0, 10, grow="y", en="b"),
    Bx(900, 880 - 0.4286 * 430, 1140, 880, "azul", None, 0, 10, grow="y", en="b"),
    Bx(1300, 880 - 0.875 * 430, 1540, 880, "rojo", None, 0, 10, grow="y", en="b"),
    T("solo oferta", 620, 920, tam=30, c="gris", w="sb", en="b"), T("oferta, sin enlace", 1020, 920, tam=30, c="gris", w="sb", en="b"),
    T("oferta y enlace", 1420, 920, tam=30, c="gris", w="sb", en="b"), T("umbral 0.50", 1700, 665, tam=30, c="ambar", w="b", en="b"),
    diapo(18, en="a"))

# ---- 9. Los cuatro patrones
v.b("Para completar el cuadro, un cálculo adicional con los mismos factores: los cuatro patrones posibles. [[a]]Sin oferta y sin enlace: 0.0204, apenas dos por ciento. [[b]]Sin oferta y con enlace: 0.1628. [[c]]Con oferta y sin enlace, el de la diapositiva: 0.4286. Y con ambas señales: 0.8750. Cada señal presente empuja hacia spam, y cada ausencia empuja hacia legítimo.",
    CL(), T("los cuatro patrones con los mismos factores", 960, 140, tam=50, c="tinta", ent="subir"),
    barras_v([("sin oferta, sin enlace", PT["00"]["p"], "azul", "a"), ("sin oferta, con enlace", PT["01"]["p"], "teal", "b"),
              ("oferta, sin enlace", PT["10"]["p"], "violeta", "c"), ("oferta y enlace", PT["11"]["p"], "rojo", "c")],
             x0=420, paso=360, base=800, escala=520, w=220, fmt="{:.4f}", tam=38),
    adicional(1050, en="a"), sec="9. Los cuatro patrones")

# ---- 10. Error típico
v.b("Error típico: multiplicar solo las señales presentes. [[a]]Si ignoramos que el enlace está ausente, volvemos al cálculo con solo oferta: 0.6667. [[b]]El modelo de Bernoulli usa también las ausencias: el factor del enlace ausente es uno menos la probabilidad de presencia. [[c]]Y en este correo, esa ausencia baja la probabilidad hasta 0.4286.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("multiplicar solo las señales presentes → 0.6667", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 460, "check", 64, en="b"), T("Bernoulli usa también las ausencias: factor = 1 − presencia", 410, 460, tam=44, c="verde", anc="lm", en="b"),
    P("con la ausencia del enlace: 0.4286", 960, 660, "violeta", 48, en="c"), sec="10. Error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un correo no trae oferta, pero sí trae enlace. [[b]]Con los factores de la diapositiva dieciséis, ¿cuál es su probabilidad de spam?",
    CL(), P("sin oferta, con enlace", 960, 300, "teal", 56, en="a"),
    T("sin oferta: 72/80 = 0.90 y 4/20 = 0.20   ·   con enlace: 16/80 = 0.20 y 14/20 = 0.70", 960, 450, tam=36, c="tinta", w="sb", en="b"),
    P("P(spam | sin oferta, con enlace) = ?", 960, 600, "violeta", 52, en="b", t=0.5), sec="11. Piensa un segundo", pausa=0.6)
v.b("Legítimo: 0.80 por 0.90 por 0.20, [[a]]0.144. Spam: 0.20 por 0.20 por 0.70, 0.028. [[b]]Normalizando, 0.028 sobre 0.172: aproximadamente 0.1628. El enlace sube la probabilidad, pero la falta de oferta pesa más.",
    CL(), T("s0 = 0.80 × 0.90 × 0.20 = 0.144   ·   s1 = 0.20 × 0.20 × 0.70 = 0.028", 960, 330, tam=42, c="tinta", w="sb", en="a"),
    T("0.028 / (0.144 + 0.028) ≈", 1000, 480, tam=54, c="tinta", w="sb", anc="rm", en="b"), N(0.1628, 1030, 480, "{:.4f}", 72, "violeta", anc="lm", en="b", t=1.2),
    ilustrativo(1050))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]el modelo supone que, dentro de cada clase, las señales son independientes, y por eso multiplica un factor por señal. [[b]]El producto de cada clase lleva la previa, los factores de las señales presentes y los de las ausentes; luego se normaliza. [[c]]Con oferta y sin enlace: 0.064 y 0.048, y una probabilidad de spam de 0.4286.",
    CL(), P("independencia dentro de cada clase → un factor por señal", 960, 300, "violeta", 46, en="a"),
    P("previa × presentes × ausentes, y normalizar", 960, 460, "teal", 48, en="b"),
    P("oferta sin enlace: 0.048 / (0.064 + 0.048) ≈ 0.4286", 960, 620, "ambar", 46, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]¿qué pasa si una señal nunca apareció en una clase? El problema de los ceros y el suavizado de Laplace.",
    CL(), siguiente(5, "Ceros y suavizado de Laplace"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
