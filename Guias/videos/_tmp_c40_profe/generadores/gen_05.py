import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_05_laplace.json")
v = V("Ceros y suavizado de Laplace")
BN = D["bern100"]
AL = {a["alfa"]: a for a in BN["alfas"]}

v.portada("Ceros y suavizado de Laplace", "¿No apareció, o es imposible que aparezca?",
          "Quinta parada: el problema de los ceros. Una señal que nunca apareció en una clase puede anular todo el cálculo. El suavizado de Laplace lo evita con una idea muy simple.")

# ---- 1. El problema
v.b("La diapositiva diecinueve plantea un caso hipotético. [[a]]Agregamos la palabra premio, que no apareció en ninguno de los veinte spam del ejemplo. [[b]]Entonces la probabilidad de premio dado spam es cero sobre veinte: cero. [[c]]Si el próximo correo trae premio, multiplicar por cero anula el producto del spam, aunque todas las demás señales favorezcan esa clase.",
    CL(), T("¿no apareció, o es imposible que aparezca? · diapositiva 19", 960, 140, tam=48, c="tinta", ent="subir"),
    T("P(premio | spam) = 0/20 =", 1000, 320, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(0, 1030, 320, "{:.0f}", 72, "rojo", anc="lm", en="b", t=0.8),
    T("s1 = 0.20 × 0.80 × 0.30 × 0 = 0", 960, 500, tam=52, c="rojo", w="b", en="c"),
    P("un solo cero anula el producto del spam", 960, 650, "rojo", 46, en="c", t=1.0), diapo(19, en="a"), sec="1. El problema de los ceros")
v.b("El punto de la diapositiva es este: [[a]]observar cero casos en una muestra pequeña no demuestra que el evento sea imposible. [[b]]El profesor dio un ejemplo práctico: el nombre de una empresa conocida por estafas. Puede aparecer muy pocas veces, pero cuando aparece es una señal muy fuerte de spam. [[c]]Si su probabilidad queda en cero, perdemos justamente esa señal.",
    CL(), P("cero casos en una muestra pequeña ≠ imposible", 960, 260, "violeta", 52, en="a"),
    cita("puede que no pase mucho porque puede que no manden muchos mensajes pero cuando pasa yo sé y estoy muy seguro", y=480, tam=38, en="b"),
    P("con probabilidad 0, perdemos esa señal", 960, 700, "rojo", 46, en="c"), grabacion("01:15:57", en="b"))

# ---- 2. Analogía
v.b("Una analogía. [[a]]Un restaurante nuevo tiene tres reseñas, y ninguna habla de un plato frío. [[b]]¿Concluyes que ahí nunca sirven un plato frío? Claro que no: tres visitas son muy pocas. [[c]]Lo razonable es asignarle una probabilidad pequeña, pero no cero. Eso es exactamente lo que hace el suavizado.",
    CL(), T("tres reseñas no prueban un «nunca»", 960, 150, tam=56, c="tinta", ent="subir"),
    P("0 quejas en 3 reseñas", 620, 380, "azul", 50, en="a"), P("¿nunca hay quejas?", 1300, 380, "rojo", 50, en="b"),
    P("probabilidad pequeña, pero no cero", 960, 600, "verde", 54, en="c"), analogia(1050), sec="2. Analogía: tres reseñas")

# ---- 3. Suavizar
v.b("La diapositiva veinte muestra la solución: el suavizado de Laplace. [[a]]Agregamos uno al conteo de presente y uno al de ausente. [[b]]Para premio entre los spam: cero presencias más uno, dividido por veinte correos spam más dos. [[c]]Uno sobre veintidós: aproximadamente 0.0455. Pequeña, pero ya no es cero.",
    CL(), T("suavizar: un pequeño peso a ambas respuestas · diapositiva 20", 960, 140, tam=46, c="tinta", ent="subir"),
    P("+1 a presente   ·   +1 a ausente", 960, 290, "teal", 50, en="a"),
    T("P(premio | spam) = (0 + 1) / (20 + 2) = 1/22 ≈", 1120, 470, tam=50, c="tinta", w="sb", anc="rm", en="b"),
    N(BN["premio"], 1150, 470, "{:.4f}", 70, "verde", anc="lm", en="c", t=0.5),
    T("presencias observadas + peso agregado   /   correos spam + uno por cada respuesta", 960, 600, tam=34, c="gris", w="sb", en="c"),
    diapo(20, en="a"), sec="3. Suavizar")
v.b("¿Por qué el denominador aumenta en dos? [[a]]Porque hay dos respuestas posibles, presente y ausente, y a cada una le sumamos uno. [[b]]Así, las dos probabilidades siguen sumando uno: uno sobre veintidós para presente, y veintiuno sobre veintidós para ausente. [[c]]Ojo con lo que no se toca: el suavizado modifica las probabilidades de las señales, pero conservamos las frecuencias de clase, ochenta y veinte de cien.",
    CL(), T("el denominador suma 2: dos respuestas posibles", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(620, "verde", "presente", "(0 + 1) / 22 = 1/22", "a"), (1300, "gris", "ausente", "(20 + 1) / 22 = 21/22", "b")],
             y=430, w=560, h=280, tam_t=52, tam_d=44),
    P("las frecuencias de clase no se suavizan: 80/100 y 20/100", 960, 720, "ambar", 44, en="c"), diapo(20, en="a"))
v.b("En general, llamamos alfa al peso agregado a cada respuesta. [[a]]La probabilidad de presencia es el conteo más alfa, dividido por el total de la clase más dos alfa. [[b]]Aquí usamos alfa igual a uno; otros valores, como 0.5 o 2, pueden compararse mediante validación. [[c]]En la grabación a este parámetro se le llama también lambda; en el material, en scikit-learn y en el paquete de R es alfa, o laplace.",
    CL(), caja_formula("presencia = (conteo + α) / (n + 2α)", y=300, c="violeta", tam=60, w=1300, en="a"),
    P("α = 1 aquí  ·  otros valores (0.5, 2) se comparan con validación", 960, 500, "teal", 44, en="b"),
    T("en la grabación (01:14:43) se le dice «lambda»; en el material es α: alpha en Python, laplace en R", 960, 680, tam=36, c="gris", w="sb", mw=1500, en="c"),
    diapo(20, en="a"))

# ---- 4. Reconstruir los factores
v.b("Volvamos al correo con oferta y sin enlace, ahora con alfa igual a uno. La diapositiva veintiuno reconstruye los factores. [[a]]Legítimo, oferta presente: ocho más uno, sobre ochenta más dos: nueve sobre ochenta y dos. Enlace ausente: sesenta y cuatro más uno, sobre ochenta y dos: sesenta y cinco sobre ochenta y dos. [[b]]Spam, oferta presente: dieciséis más uno, sobre veinte más dos: diecisiete sobre veintidós. Enlace ausente: seis más uno, sobre veintidós: siete sobre veintidós.",
    CL(), T("reconstruir los factores con suavizado · diapositiva 21", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["clase", "oferta presente", "enlace ausente"], ["legítimo", "(8 + 1)/(80 + 2) = 9/82", "((80 − 16) + 1)/(80 + 2) = 65/82"],
           ["spam", "(16 + 1)/(20 + 2) = 17/22", "((20 − 14) + 1)/(20 + 2) = 7/22"]], [330, 870, 1460], 320, paso=130, tam=40, x0=130, x1=1790,
          colores=[["tinta"] * 3, ["azul", "tinta", "tinta"], ["rojo", "tinta", "tinta"]], ens=[None, "a", "b"]),
    P("spam sin enlace: 20 − 14 = 6  →  6 + 1 = 7 sobre 20 + 2 = 22", 960, 760, "teal", 42, en="b", t=3.0), diapo(21, en="a"), sec="4. Los factores suavizados")
v.b("Una precisión sobre la grabación en este punto. [[a]]Al explicar el siete sobre veintidós, cerca del minuto sesenta y siete, se dijo que los veinte spam observados son ochenta de los cien. [[b]]Son veinte de los cien. Por eso el denominador es veinte más dos, veintidós: los veinte spam, más los dos del suavizado.",
    CL(), error_material("01:07:22", "20 de los spam que son observados son 80 de los 100",
                         "son 20 de los 100: denominador 20 + 2 = 22", en_ok="b"), sec="5. Una precisión")

# ---- 5. Completar el cálculo
v.b("Completamos, como en la diapositiva veintidós. [[a]]El producto del legítimo: 0.80 por nueve sobre ochenta y dos, por sesenta y cinco sobre ochenta y dos: aproximadamente 0.0696. [[b]]El del spam: 0.20 por diecisiete sobre veintidós, por siete sobre veintidós: aproximadamente 0.0492. [[c]]Normalizando: la probabilidad de spam queda en aproximadamente 0.4140. El resultado se calcula con las fracciones completas, sin redondear los productos intermedios.",
    CL(), T("completar el cálculo con suavizado · diapositiva 22", 960, 140, tam=48, c="tinta", ent="subir"),
    T("s0 = 0.80 × 9/82 × 65/82 ≈", 1000, 300, tam=52, c="azul", w="sb", anc="rm", en="a"), N(BN["s_leg"], 1030, 300, "{:.4f}", 64, "azul", anc="lm", en="a", t=1.5),
    T("s1 = 0.20 × 17/22 × 7/22 ≈", 1000, 430, tam=52, c="rojo", w="sb", anc="rm", en="b"), N(BN["s_spam"], 1030, 430, "{:.4f}", 64, "rojo", anc="lm", en="b", t=1.5),
    T("P(spam | oferta, sin enlace) = s1 / (s0 + s1) ≈", 1150, 590, tam=46, c="tinta", w="sb", anc="rm", en="c"),
    N(BN["p_spam"], 1180, 590, "{:.4f}", 70, "violeta", anc="lm", en="c", t=1.5),
    T("con las fracciones completas, sin redondear los productos intermedios", 960, 720, tam=34, c="gris", w="sb", en="c", t=2.0),
    diapo(22, en="a"), sec="6. Completar el cálculo")
v.b("Pasamos de 0.4286 a 0.4140. [[a]]La diapositiva agrega una advertencia importante: el suavizado evita los ceros, pero no corrige la dependencia entre señales. [[b]]Si oferta y enlace estuvieran repitiendo la misma información, el suavizado no lo arreglaría.",
    P("0.4286  →  0.4140", 960, 830, "violeta", 54, en=None),
    P("suavizar evita ceros; no corrige la dependencia", 960, 950, "ambar", 44, en="a"))

# ---- 6. En código
v.b("La diapositiva veinticuatro repite esta cuenta en R y en Python, y es el primer cálculo del script. [[a]]El producto del legítimo, el del spam, y la normalización. [[b]]En los dos lenguajes, la salida es la misma: aproximadamente 0.4140.",
    CL(), T("el mismo cálculo en R y Python · diapositiva 24", 960, 140, tam=50, c="tinta", ent="subir"),
    codigo(["s_legitimo = (80/100)*(9/82)*(65/82)", "s_spam = (20/100)*(17/22)*(7/22)", "p_spam = s_spam/(s_legitimo+s_spam)", "print(p_spam)"],
           y=400, w=1400, tam=38, c="violeta", en="a"),
    P("salida en ambos lenguajes: p ≈ 0.4140", 960, 700, "teal", 48, en="b"), real(1050, en="b"), sec="7. En R y Python")
v.b("La diapositiva veinticinco ordena la salida. [[a]]Oferta entre los spam: sin suavizado, 0.8000; con alfa igual a uno, diecisiete sobre veintidós, 0.7727. [[b]]Enlace entre los spam: 0.7000 sin suavizado, y quince sobre veintidós, 0.6818, con suavizado. [[c]]Y la probabilidad de spam del correo: 0.4286 y 0.4140. Las primeras filas describen señales dentro de una clase; la última clasifica el correo.",
    CL(), T("leer la salida sin perder los conteos · diapositiva 25", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["probabilidad", "sin suavizado", "con α = 1"], ["oferta entre spam", "16/20 = 0.8000", "17/22 ≈ 0.7727"],
           ["enlace entre spam", "14/20 = 0.7000", "15/22 ≈ 0.6818"], ["spam para oferta sin enlace", "0.4286", "0.4140"]],
          [520, 1060, 1480], 300, paso=115, tam=42, x0=170, x1=1750,
          colores=[["tinta"] * 3, ["tinta"] * 3, ["tinta"] * 3, ["violeta"] * 3], ens=[None, "a", "b", "c"]),
    diapo(25, en="a"))

# ---- 7. Alerta: cuánto cambió
v.b("Sobre este cambio, una precisión de la grabación. [[a]]Cerca del minuto setenta y cinco se dijo que, a modo global, el resultado cambió casi en un uno por ciento. [[b]]La diferencia es 0.4286 menos 0.4140: 0.0146, es decir, cerca de un punto y medio porcentual. Un cambio pequeño, como se dijo, pero algo mayor que uno.",
    CL(), error_material("01:14:58", "a modo global me cambió casi en 1 %", "0.4286 − 0.4140 = 0.0146: cerca de 1.5 puntos porcentuales", en_ok="b"),
    sec="8. Cuánto cambió")

# ---- 9. El programa
v.b("La diapositiva veintiséis muestra cómo el programa aprende estas probabilidades de una vez. [[a]]Un vector con los correos por clase, ochenta y veinte; una matriz de conteos, con filas legítimo y spam, y columnas oferta y enlace. [[b]]La probabilidad de presencia es conteos más alfa, dividido por el total de la clase más dos alfa. [[c]]Para el spam sale diecisiete sobre veintidós y quince sobre veintidós. Y para una ausencia se usa uno menos ese valor.",
    CL(), T("el programa aprende las mismas probabilidades · diapositiva 26", 960, 140, tam=46, c="tinta", ent="subir"),
    codigo(["n = np.array([80., 20.])", "conteos = np.array([[8., 16.], [16., 14.]])", "alfa = 1", "prior = n / n.sum()",
            "theta = (conteos + alfa) / (n[:, None] + 2*alfa)"], y=400, w=1500, tam=36, c="violeta", en="a"),
    P("spam: (17/22, 15/22) ≈ (0.7727, 0.6818)  ·  previa (0.80, 0.20)", 960, 720, "teal", 42, en="c"),
    T("para una ausencia: 1 − theta", 960, 840, tam=40, c="tinta", w="sb", en="c", t=1.0), diapo(26, en="a"), sec="9. El programa")

# ---- 10. Cuándo importa
v.b("¿Cuándo importa el suavizado? [[a]]El profesor fue claro: sobre todo cuando hay pocos datos, o señales muy raras que queremos conservar. Con muchos datos y un alfa pequeño, las probabilidades casi no cambian. [[b]]En Spambase, un cálculo adicional lo confirma: la señal más rara entre los spam aparece en un solo correo, y la más rara entre los legítimos, en ocho. Ninguna queda en cero. [[c]]Por eso, en la parada nueve, alfa 0.5, uno y dos van a dar resultados casi iguales.",
    CL(), T("¿cuándo importa el suavizado?", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "importa", "pocos datos o señales muy raras", "a"), (1300, "gris", "casi no importa", "muchos datos y α pequeño", "a")],
             y=400, w=560, h=260, tam_t=52, tam_d=38),
    T("Spambase: la señal más rara aparece en 1 spam y en 8 legítimos; ninguna en cero", 960, 650, tam=38, c="teal", w="sb", en="b"),
    P("parada 9: α = 0.5, 1 y 2 dan resultados casi iguales", 960, 790, "violeta", 44, en="c"), adicional(1050, en="b"), sec="10. Cuándo importa")

# ---- 11. Error típico
v.b("Errores típicos. [[a]]Uno: suavizar también las frecuencias de clase. El suavizado va en las probabilidades de las señales. [[b]]Dos: creer que el suavizado arregla la dependencia entre señales. No la arregla. [[c]]Y tres: confundir suavizar con decidir. Alfa cambia el cálculo de la probabilidad; el umbral, que veremos en la parada once, cambia la acción.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("suavizar las frecuencias de clase", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("creer que el suavizado corrige la dependencia", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("confundir α (el cálculo) con el umbral (la acción)", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="11. Errores típicos")

# ---- 12. Piensa un segundo
v.b("Piensa un segundo. [[a]]Una palabra no apareció en ninguno de los ochenta legítimos. [[b]]Con alfa igual a uno, ¿cuál es su probabilidad de presencia entre los legítimos?",
    CL(), P("0 presencias entre 80 legítimos  ·  α = 1", 960, 330, "teal", 50, en="a"),
    P("presencia suavizada = ?", 960, 520, "violeta", 60, en="b"), sec="12. Piensa un segundo", pausa=0.6)
v.b("Cero más uno, sobre ochenta más dos: [[a]]uno sobre ochenta y dos, [[b]]aproximadamente 0.0122. Más pequeña que el 0.0455 de los spam, porque hay más legítimos: la evidencia de que es rara entre ellos es mayor.",
    CL(), T("(0 + 1) / (80 + 2) = 1/82 ≈", 1000, 380, tam=56, c="tinta", w="sb", anc="rm", en="a"), N(0.0122, 1030, 380, "{:.4f}", 72, "violeta", anc="lm", en="b"),
    P("más correos → más evidencia de que es rara", 960, 560, "teal", 46, en="b", t=1.0), ilustrativo(1050))

# ---- 13. Para llevar
v.b("Para llevarte: [[a]]un conteo cero haría cero todo el producto de una clase, aunque el evento sea solo raro. [[b]]Con el suavizado de Laplace, se suma alfa a presente y alfa a ausente: con alfa uno, premio pasa de cero a 0.0455. [[c]]En el ejemplo, la probabilidad de spam pasa de 0.4286 a 0.4140. Evita ceros, pero no corrige la dependencia.",
    CL(), P("un conteo 0 anula el producto de toda la clase", 960, 300, "rojo", 46, en="a"),
    P("(conteo + α) / (n + 2α): 0/20 → 1/22 ≈ 0.0455", 960, 460, "violeta", 46, en="b"),
    P("0.4286 → 0.4140 · no corrige la dependencia", 960, 620, "ambar", 46, en="c"), sec="13. Para llevar")
v.b("Siguiente parada: [[a]]separamos aprender de predecir, generalizamos a varias clases y vemos por qué en la práctica se suman logaritmos en vez de multiplicar.",
    CL(), siguiente(6, "Aprender, predecir, varias clases y logaritmos"), sec="14. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
