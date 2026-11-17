import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo)

OUT = out("guion_c40_02_bayes.json")
v = V("El teorema de Bayes")
B = D["bayes"]
A, Z = B["oferta_spam"], B["oferta_leg"]

v.portada("El teorema de Bayes", "Productos, normalización y la frecuencia de spam",
          "Segunda parada: el teorema de Bayes. Vamos a recuperar el dos tercios de la tabla con una fórmula, y vamos a ver por qué la misma palabra puede significar cosas muy distintas según cuánto spam haya.")

# ---- 1. Hook
v.b("En la parada anterior contamos directamente en la tabla: dieciséis de veinticuatro. [[a]]Pero con muchas señales nunca vamos a tener esa columna ya contada para cada combinación posible. [[b]]Lo que sí podemos estimar con facilidad es cuánto spam hay, y qué tan frecuente es cada señal dentro de cada clase. [[c]]El teorema de Bayes arma la posterior justamente con esas dos piezas.",
    CL(), T("¿y si no tenemos la columna ya contada?", 960, 150, tam=54, c="tinta", ent="subir"),
    P("lo que sí estimamos: la previa y la condicional", 960, 400, "teal", 48, en="b"),
    L(960, 470, 960, 560, "gris", 6, True, en="c"), P("Bayes arma la posterior", 960, 630, "violeta", 52, en="c"), sec="1. Por qué una fórmula")

# ---- 2. La fórmula
v.b("Esta es la fórmula de la diapositiva siete. [[a]]Arriba: la probabilidad de spam, multiplicada por la probabilidad de oferta dado spam. [[b]]Abajo: la suma de dos productos del mismo tipo, uno por cada clase: legítimo por oferta dado legítimo, más spam por oferta dado spam. [[c]]Fíjate que el producto de arriba aparece también abajo: por eso el resultado nunca pasa de uno.",
    CL(), T("Bayes recupera el mismo cálculo · diapositiva 7", 960, 140, tam=50, c="tinta", ent="subir"),
    T("P(spam | oferta) =", 960, 240, tam=46, c="violeta", w="b", en="a"),
    T("P(spam) · P(oferta | spam)", 960, 340, tam=48, c="rojo", w="sb", en="a"),
    L(250, 400, 1670, 400, "tinta", 4, en="a"),
    T("P(legítimo) · P(oferta | legítimo)  +  P(spam) · P(oferta | spam)", 960, 460, tam=44, c="tinta", w="sb", en="b"),
    R(1344, 460, 620, 90, "rojo", r=20, en="c"), R(960, 340, 660, 90, "rojo", r=20, en="c"),
    diapo(7, en="a"), sec="2. La fórmula")
v.b("Reemplazamos con las frecuencias de la tabla. [[a]]Previa de spam: veinte sobre cien, 0.20. Oferta dado spam: 0.80. Su producto: 0.16. [[b]]Previa de legítimo: 0.80. Oferta dado legítimo: 0.10. Su producto: 0.08. [[c]]Entonces, 0.16 dividido por 0.08 más 0.16, es decir, 0.16 sobre 0.24: aproximadamente 0.6667. El mismo dos tercios que contamos a mano.",
    T("0.20 × 0.80", 1160, 650, tam=46, c="rojo", w="sb", en="a"), T("= 0.16", 1480, 650, tam=46, c="rojo", w="b", en="a", t=0.6),
    L(700, 710, 1620, 710, "tinta", 4, en="b"),
    T("0.80 × 0.10  +  0.20 × 0.80   =   0.08 + 0.16", 1160, 770, tam=44, c="tinta", w="sb", en="b"),
    T("= 0.16 / 0.24 ≈", 1000, 900, tam=50, c="tinta", w="sb", anc="rm", en="c"),
    N(0.6667, 1030, 900, "{:.4f}", 70, "violeta", anc="lm", en="c", t=1.6))

# ---- 3. El denominador
v.b("¿Qué es ese denominador? [[a]]La diapositiva lo dice así: suma las dos formas de observar un correo con oferta. [[b]]Cero coma cero ocho es la fracción de los cien correos que son legítimos y traen oferta: ocho de cien. [[c]]Cero coma dieciséis es la fracción que son spam y traen oferta: dieciséis de cien. Y la suma, 0.24, son los veinticuatro de cien que traen la palabra.",
    CL(), T("el denominador: todas las formas de ver «oferta»", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(520, "azul", "0.08", "legítimo y oferta: 8 de 100", "b"), (960, "rojo", "0.16", "spam y oferta: 16 de 100", "c"),
              (1400, "violeta", "0.24", "con oferta: 24 de 100", "c")], y=480, w=400, h=300, tam_t=80, tam_d=36),
    T("0.24 = 0.08 + 0.16", 960, 760, tam=56, c="tinta", w="b", en="c", t=1.5), diapo(7, en="a"), sec="3. El denominador")

# ---- 4. Normalizar
v.b("La diapositiva ocho le pone nombre a este paso: normalizar. [[a]]Primero calculamos un producto por clase: previa por condicional. Para legítimo, 0.08; para spam, 0.16. [[b]]Esos productos todavía no son probabilidades del correo: no suman uno. [[c]]Normalizar es dividir cada producto por la suma de ambos. Cero coma cero ocho entre 0.24 da 0.3333, y cero coma dieciséis entre 0.24 da 0.6667. Ahora sí suman uno.",
    CL(), T("convertir los productos en probabilidades · diapositiva 8", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["clase", "producto: clase y oferta", "dividir por el total"], ["legítimo", "(80/100)(8/80) = 0.08", "0.08 / 0.24 ≈ 0.3333"],
           ["spam", "(20/100)(16/20) = 0.16", "0.16 / 0.24 ≈ 0.6667"]], [400, 960, 1500], 320, paso=120, tam=42, x0=180, x1=1760,
          colores=[["tinta"] * 3, ["azul", "tinta", "tinta"], ["rojo", "tinta", "violeta"]], ens=[None, "a", "c"]),
    P("los productos todavía no suman 1", 960, 720, "ambar", 44, en="b"),
    P("normalizar = dividir cada producto por la suma", 960, 860, "violeta", 46, en="c"), diapo(8, en="a"), sec="4. Normalizar")

# ---- 5. Analogía
v.b("Una analogía para normalizar. [[a]]Imagina una votación con dos candidatos. Uno saca ocho votos y el otro dieciséis. [[b]]Para pasar a porcentajes, divides cada cuenta por el total, veinticuatro: un tercio y dos tercios. [[c]]Los productos del teorema funcionan igual: son como votos de cada clase, y normalizar los convierte en probabilidades que suman uno.",
    CL(), T("como convertir votos en porcentajes", 960, 150, tam=54, c="tinta", ent="subir"),
    barras_v([("legítimo", 8, "azul", "a"), ("spam", 16, "rojo", "a")], x0=760, paso=400, base=760, escala=24, w=220, fmt="{:.0f} votos"),
    T("8/24 = 1/3     ·     16/24 = 2/3", 960, 900, tam=50, c="violeta", w="b", en="b"), analogia(1050), sec="5. Analogía: los votos")

# ---- 6. Código
v.b("En código son tres líneas, y en R y en Python se ven casi iguales. [[a]]El producto del spam, el producto del legítimo, [[b]]y la división. No hace falta ningún paquete: son multiplicaciones y una división.",
    CL(), T("la cuenta en código", 960, 150, tam=54, c="tinta", ent="subir"),
    codigo(["s1 = 0.20 * 0.80          # spam: previa × condicional", "s0 = 0.80 * 0.10          # legítimo",
            "p = s1 / (s0 + s1)        # 0.6667"], y=420, w=1500, tam=38, c="violeta", en="a"),
    P("solo multiplicar y normalizar", 960, 720, "teal", 46, en="b"), sec="6. En código")

# ---- 7. La frecuencia de spam cambia la respuesta
v.b("Ahora el experimento más importante de esta parada, de la diapositiva nueve. [[a]]Mantenemos las condicionales: oferta dado spam, 0.80; oferta dado legítimo, 0.10. [[b]]Solo cambiamos la previa: en vez de veinte por ciento de spam, suponemos un dos por ciento. Es un escenario hipotético. [[c]]Entonces la previa de legítimo es uno menos 0.02: 0.98.",
    CL(), T("la frecuencia de spam cambia la respuesta · diapositiva 9", 960, 140, tam=48, c="tinta", ent="subir"),
    P("se mantiene: P(oferta | spam) = 0.80  ·  P(oferta | legítimo) = 0.10", 960, 280, "teal", 40, en="a"),
    tabla([["probabilidad", "ejemplo inicial", "nuevo escenario"], ["P(spam)", "20/100 = 0.20", "0.02 (supuesto)"],
           ["P(legítimo)", "1 − 0.20 = 0.80", "1 − 0.02 = 0.98"]], [480, 1000, 1480], 420, paso=110, tam=42, x0=220, x1=1720,
          ens=[None, "b", "c"]), diapo(9, en="a"), sec="7. Cambia la frecuencia de spam")
v.b("Hacemos la misma cuenta. [[a]]Spam y oferta: 0.02 por 0.80, 0.016. [[b]]Legítimo y oferta: 0.98 por 0.10, 0.098. [[c]]La probabilidad de spam dado oferta es 0.016 dividido por 0.098 más 0.016: aproximadamente 0.1404. Con las mismas condicionales, pasamos de dos tercios a un catorce por ciento.",
    CL(), T("misma palabra, otra frecuencia de spam", 960, 140, tam=52, c="tinta", ent="subir"),
    tabla([["", "ejemplo inicial", "nuevo escenario"], ["spam y oferta", "0.20 × 0.80 = 0.16", "0.02 × 0.80 = 0.016"],
           ["legítimo y oferta", "0.80 × 0.10 = 0.08", "0.98 × 0.10 = 0.098"],
           ["P(spam | oferta)", "0.16 / 0.24 ≈ 0.6667", "0.016 / 0.114 ≈ 0.1404"]], [450, 1000, 1480], 300, paso=120, tam=42, x0=200, x1=1740,
          colores=[["tinta"] * 3, ["rojo", "tinta", "tinta"], ["azul", "tinta", "tinta"], ["violeta", "violeta", "violeta"]], ens=[None, "a", "b", "c"]),
    T("con «oferta»: 20 % → 66.7 %   ·   2 % → 14.0 %", 960, 860, tam=48, c="tinta", w="b", en="c", t=2.0), diapo(9, en="a"))
v.b("En la grabación, el profesor lo resume así: [[a]]pasamos de casi sesenta y siete por ciento a casi catorce, y lo único que cambiamos fue la proporción de spam. [[b]]Si el spam es raro, tenga o no la palabra oferta, es difícil que un correo sea spam.",
    CL(), cita("pasamos de tener casi 67 % de probabilidad ... casi un 14 % de que sea spam", y=330, tam=44, en="a"),
    P("yo dejé todo constante, exceptuando la proporción de spam", 960, 560, "teal", 44, en="b"), grabacion("00:26:28"))

# ---- 8. Con cuentas
v.b("Si la fórmula se siente abstracta, piensa en cuentas. Este es un cálculo adicional. [[a]]Imagina mil correos con dos por ciento de spam: veinte spam y novecientos ochenta legítimos. [[b]]De los veinte spam, el ochenta por ciento trae oferta: dieciséis. De los novecientos ochenta legítimos, el diez por ciento trae oferta: noventa y ocho. [[c]]Hay ciento catorce correos con oferta, y solo dieciséis son spam. Dieciséis sobre ciento catorce: otra vez 0.1404.",
    CL(), T("la misma cuenta con 1000 correos", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "20 spam", "80 % con oferta → 16", "a"), (1300, "azul", "980 legítimos", "10 % con oferta → 98", "a")],
             y=420, w=560, h=280, tam_t=56, tam_d=40),
    T("con oferta: 16 + 98 = 114  →  16 / 114 ≈", 1080, 720, tam=48, c="tinta", w="sb", anc="rm", en="c"),
    N(0.1404, 1110, 720, "{:.4f}", 66, "violeta", anc="lm", en="c", t=2.5),
    T("los legítimos con oferta (98) superan a los spam con oferta (16)", 960, 860, tam=38, c="azul", w="sb", en="c", t=3.5),
    adicional(1050, en="a"), sec="8. La misma cuenta con 1000 correos")

# ---- 9. La curva
GC = "gc2"
gx = [0.001] + [i / 50 for i in range(1, 50)] + [0.999]
gy = [g * A / (g * A + (1 - g) * Z) for g in gx]
v.b("La diapositiva diez dibuja la cuenta para todas las previas posibles. [[a]]En el eje horizontal, la frecuencia previa de spam; en el vertical, la probabilidad de spam al observar oferta. [[b]]La curva no es una recta: sube muy rápido al principio. [[c]]Los dos puntos son los de la tabla: dos por ciento da catorce por ciento, y veinte por ciento da sesenta y siete por ciento.",
    CL(), T("la misma palabra, dos puntos de partida · diapositiva 10", 960, 120, tam=48, c="tinta", ent="subir"),
    G(330, 220, 1300, 880, 0, 1, 0, 1, [dict(xs=gx, ys=gy, color="azul", grosor=6, revela=0.0, suave=True)],
      xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yfmt="{:.1f}", xlabel="probabilidad previa de spam",
      id=GC, tam_txt=28), T("P(spam | oferta)", 330, 190, tam=30, c="tinta", w="sb", anc="lm"),
    U(GC, **{"series.0.revela": 1.0}, en="b", dur=2.0),
    R(0.02, B["posterior"][1], 50, 50, "ambar", r=25, ref=GC, en="c"), R(0.20, B["posterior"][0], 50, 50, "ambar", r=25, ref=GC, en="c"),
    T("2 % → 14.0 %", 1560, 420, tam=46, c="ambar", w="b", en="c"), T("20 % → 66.7 %", 1560, 520, tam=46, c="ambar", w="b", en="c", t=0.3),
    diapo(10, en="a"), sec="9. La curva")
v.b("Y en el otro extremo, como comentó el profesor: [[a]]si el noventa y ocho por ciento de los correos fuera spam, [[b]]la probabilidad de spam dado oferta sería casi uno, 0.9975. Tenga o no la palabra, casi todo es spam.",
    R(0.98, B["curva"]["p"][-1], 50, 50, "rojo", r=25, ref=GC, en="a"),
    T("98 % → 0.9975", 1560, 640, tam=46, c="rojo", w="b", en="b"), grabacion("00:29:56", en="a", y=1010))

# ---- 10. La palabra ayuda pero no decide
v.b("La conclusión de la diapositiva diez es una frase para guardar: [[a]]la palabra ayuda, pero no decide por sí sola. [[b]]La misma señal, con las mismas frecuencias dentro de cada clase, lleva a probabilidades muy distintas según la previa. [[c]]Por eso, en la parada once, antes de usar el filtro en otra bandeja de correo, vamos a revisar si su frecuencia de spam se parece a la de los datos de ajuste.",
    CL(), T("la palabra ayuda, pero no decide por sí sola", 960, 300, tam=60, c="violeta", w="b", ent="subir", en="a"),
    P("misma señal + otra previa = otra probabilidad", 960, 500, "teal", 48, en="b"),
    P("otra bandeja: ¿su frecuencia de spam se parece a la del ajuste?", 960, 660, "ambar", 44, en="c"), diapo(10, en="a"), sec="10. La palabra ayuda")

# ---- 11. Error típico
v.b("Error típico: olvidar la previa. [[a]]Hay dos formas de cometerlo. Una: leer la condicional como si fuera la posterior, que es el error de la parada anterior. [[b]]Otra: entrenar con datos donde el spam es frecuente, y aplicar el filtro en una bandeja donde el spam es raro, sin ajustar nada. [[c]]El modelo seguirá creyendo que el spam es frecuente, y va a exagerar la probabilidad de spam.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("leer P(oferta | spam) como si fuera P(spam | oferta)", 410, 300, tam=42, c="rojo", anc="lm", en="a"),
    K(330, 460, "cruz", 64, en="b"), T("entrenar con spam frecuente y aplicar donde es raro", 410, 460, tam=42, c="rojo", anc="lm", en="b"),
    P("la previa del ajuste viaja dentro del modelo", 960, 680, "ambar", 46, en="c"), sec="11. Error típico")

# ---- 12. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con las mismas condicionales, 0.80 y 0.10, ¿qué probabilidad de spam dado oferta obtienes si la mitad de los correos fuera spam?",
    CL(), P("P(spam) = 0.5  ·  P(oferta | spam) = 0.80  ·  P(oferta | legítimo) = 0.10", 960, 330, "teal", 40, en="a"),
    P("P(spam | oferta) = ?", 960, 520, "violeta", 60, en="a", t=0.5), sec="12. Piensa un segundo", pausa=0.6)
v.b("El producto del spam es 0.5 por 0.80: 0.40. [[a]]El del legítimo, 0.5 por 0.10: 0.05. [[b]]Cero coma cuarenta dividido por 0.45: aproximadamente 0.8889. Con previas iguales, solo decide la razón entre las condicionales: ocho a uno.",
    CL(), T("0.5 × 0.80 = 0.40   ·   0.5 × 0.10 = 0.05", 960, 330, tam=50, c="tinta", w="sb", en="a"),
    T("0.40 / (0.05 + 0.40) ≈", 1000, 480, tam=54, c="tinta", w="sb", anc="rm", en="b"), N(0.8889, 1030, 480, "{:.4f}", 72, "violeta", anc="lm", en="b", t=1.2),
    P("con previas iguales, decide la razón 0.80 / 0.10 = 8", 960, 660, "teal", 44, en="b", t=2.0), ilustrativo(1050))

# ---- 13. Para llevar
v.b("Para llevarte: [[a]]el teorema de Bayes dice que la posterior es proporcional a la previa por la condicional. [[b]]Calculamos un producto por clase y normalizamos dividiendo por la suma. [[c]]Y la previa importa mucho: con oferta, veinte por ciento de spam da 66.7 por ciento, y dos por ciento da apenas 14.0 por ciento.",
    CL(), P("posterior proporcional a previa × condicional", 960, 300, "violeta", 52, en="a"),
    P("un producto por clase, y normalizar", 960, 460, "teal", 48, en="b"),
    P("20 % → 66.7 %   ·   2 % → 14.0 %", 960, 620, "ambar", 50, en="c"), sec="13. Para llevar")
v.b("Siguiente parada: [[a]]agregamos una segunda señal, el enlace, y vemos por qué no podemos contar todas las combinaciones.",
    CL(), siguiente(3, "Dos señales, Bernoulli y la explosión de combinaciones"), sec="14. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
