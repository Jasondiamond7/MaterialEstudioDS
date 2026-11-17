import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo)

OUT = out("guion_c40_06_formula.json")
v = V("Aprender, predecir y generalizar")
LG = D["log"]
l0, l1 = math.log(0.064), math.log(0.048)

v.portada("Aprender, predecir y generalizar", "K clases, un factor por variable y log-probabilidades",
          "Sexta parada. Ordenamos lo que ya sabemos hacer: qué se aprende y qué se predice. Después lo generalizamos a varias clases y a muchas variables, y vemos por qué en la práctica se suman logaritmos.")

# ---- 1. Aprender y predecir
v.b("La diapositiva veintitrés separa dos momentos. [[a]]Ajustar, con correos etiquetados: contamos cuántos correos hay en cada clase y cuántos presentan cada señal. Con esos conteos y el suavizado obtenemos las probabilidades que usará el modelo. [[b]]Predecir un correo: leemos qué señales están presentes y cuáles ausentes, sin conocer su etiqueta; elegimos los factores correspondientes, multiplicamos por la previa y normalizamos.",
    CL(), T("separar lo que aprendemos de lo que predecimos · diapositiva 23", 960, 140, tam=46, c="tinta", ent="subir"),
    tarjetas([(620, "azul", "ajustar", "contar correos por clase y presencias por señal; suavizar", "a"),
              (1300, "violeta", "predecir", "leer presentes y ausentes; factores × previa; normalizar", "b")],
             y=470, w=600, h=380, tam_t=56, tam_d=38), diapo(23, en="a"), sec="1. Aprender y predecir")
v.b("La consecuencia práctica: [[a]]las probabilidades se aprenden una sola vez, durante el ajuste. [[b]]Después sirven para clasificar todos los correos que lleguen, sin volver a contar. [[c]]Y como explicó el profesor, cuando alguien marca un correo como spam en su bandeja, está agregando un ejemplo etiquetado más para el próximo ajuste.",
    CL(), P("se aprende una vez", 620, 330, "azul", 54, en="a"), P("se usa para cada correo nuevo", 1300, 330, "violeta", 50, en="b"),
    cita("cuando usted aprieta en esto marcar como spam lo que están haciendo es clasificar una observación más", y=560, tam=40, en="c"),
    grabacion("01:11:01", en="c"))

# ---- 2. Analogía
v.b("Una analogía. [[a]]Ajustar es como escribir una tabla de consulta: para cada clase, qué tan frecuente es cada señal. [[b]]Predecir es consultar esa tabla: buscas las filas de las señales del correo, multiplicas y normalizas. [[c]]Por eso Naive Bayes es tan rápido: aprender es contar, y predecir es multiplicar.",
    CL(), T("una tabla de consulta", 960, 150, tam=56, c="tinta", ent="subir"),
    flujo([("contar: escribir la tabla", "azul", "a"), ("consultar: multiplicar y normalizar", "violeta", "b")], y=430, w=620, h=220, x0=600, paso=720, tam=42),
    P("aprender = contar  ·  predecir = multiplicar", 960, 700, "teal", 48, en="c"), analogia(1050), sec="2. Analogía: una tabla de consulta")

# ---- 3. Qué responde cada salida
v.b("La diapositiva veintisiete pone en una tabla qué responde cada cantidad, para el correo con oferta y sin enlace, con alfa igual a uno. [[a]]La entrada: las señales que observamos; oferta sí, enlace no. [[b]]La previa: la frecuencia de clase antes de leer el correo; la probabilidad de spam es 0.20. [[c]]La condicional: la frecuencia de una señal dentro de una clase; oferta en spam, diecisiete sobre veintidós. [[d]]Y la posterior: la probabilidad de clase al combinar las señales: 0.4140 de spam.",
    CL(), T("cada salida del modelo responde una pregunta · diapositiva 27", 960, 140, tam=46, c="tinta", ent="subir"),
    tabla([["cantidad", "qué significa", "en nuestro ejemplo"], ["entrada", "señales que observamos en el correo", "oferta sí, enlace no"],
           ["previa", "frecuencia de clase antes de leerlo", "P(spam) = 0.20"], ["condicional", "frecuencia de una señal dentro de clase", "oferta en spam: 17/22"],
           ["posterior", "probabilidad de clase al combinar las señales", "0.4140 de spam"]],
          [330, 900, 1500], 290, paso=110, tam=36, x0=140, x1=1790,
          colores=[["tinta"] * 3, ["gris", "tinta", "tinta"], ["azul", "tinta", "tinta"], ["teal", "tinta", "tinta"], ["violeta", "tinta", "violeta"]],
          ens=[None, "a", "b", "c", "d"]),
    diapo(27, en="a"), sec="3. Qué responde cada salida")
v.b("La frase de cierre de esa diapositiva vale oro: [[a]]el 0.7727 describe la oferta dentro del spam; [[b]]el 0.4140 describe el spam para este correo. Son preguntas diferentes, igual que en la primera parada.",
    CL(), P("0.7727: oferta dentro de spam (condicional)", 960, 360, "teal", 50, en="a"),
    P("0.4140: spam para este correo (posterior)", 960, 540, "violeta", 50, en="b"), diapo(27))

# ---- 4. K clases
v.b("Todo esto sirve también para más de dos clases. [[a]]La diapositiva veintiocho usa K clases, por ejemplo spam, personal y trabajo. [[b]]Pi sub ka es la probabilidad previa de la clase ka. Efe sub ka de equis es la probabilidad de las entradas dentro de esa clase; si las entradas son continuas, su densidad. [[c]]La posterior de la clase ka es su producto, pi por efe, dividido por la suma de los productos de todas las clases.",
    CL(), T("la misma cuenta para varias clases · diapositiva 28", 960, 140, tam=48, c="tinta", ent="subir"),
    P("K clases: por ejemplo spam, personal y trabajo", 960, 270, "teal", 44, en="a"),
    T("πₖ = P(Y = cₖ): previa de la clase  ·  fₖ(x): probabilidad (o densidad) de x en la clase", 960, 380, tam=36, c="tinta", w="sb", en="b"),
    caja_formula("P(Y = cₖ | X = x) = πₖ fₖ(x) / Σℓ πℓ fℓ(x)", y=560, c="violeta", tam=56, w=1300, en="c"),
    T("el índice ℓ recorre las clases: con tres clases, el denominador suma tres productos", 960, 740, tam=36, c="gris", w="sb", en="c", t=1.5),
    diapo(28, en="a"), sec="4. K clases")
v.b("Un ejemplo ilustrativo con tres clases. [[a]]Tomemos los productos del correo con oferta y sin enlace, 0.064 para legítimo y 0.048 para spam, y agreguemos una tercera clase, trabajo, con producto 0.016. [[b]]La suma es 0.128. [[c]]Normalizando: legítimo 0.5, spam 0.375 y trabajo 0.125. Al agregar una clase, la probabilidad de spam bajó de 0.4286 a 0.375: el denominador tiene un término más.",
    CL(), T("un ejemplo con tres clases", 960, 150, tam=54, c="tinta", ent="subir"),
    tabla([["clase", "producto", "probabilidad"], ["legítimo", "0.064", "0.064 / 0.128 = 0.5"], ["spam", "0.048", "0.048 / 0.128 = 0.375"],
           ["trabajo", "0.016", "0.016 / 0.128 = 0.125"]], [560, 960, 1380], 300, paso=110, tam=44, x0=300, x1=1660,
          colores=[["tinta"] * 3, ["azul", "tinta", "tinta"], ["rojo", "tinta", "tinta"], ["teal", "tinta", "tinta"]], ens=[None, "a", "a", "a"]),
    P("suma = 0.064 + 0.048 + 0.016 = 0.128", 960, 800, "violeta", 44, en="b"),
    T("spam: 0.4286 con dos clases → 0.375 con tres", 960, 920, tam=40, c="tinta", w="sb", en="c"), ilustrativo(1050))

# ---- 5. Un factor por variable
v.b("Ahora agregamos el supuesto ingenuo a la fórmula general, como en la diapositiva veintinueve. [[a]]Con d variables, efe sub ka de equis se reemplaza por el producto de un factor por variable: el factor del valor observado de cada variable en esa clase. [[b]]Arriba, la previa por el producto de los factores. Abajo, la suma de esos mismos productos sobre todas las clases. [[c]]La diapositiva lo resume así: el producto recorre las variables; la suma recorre las clases.",
    CL(), T("Naive Bayes: un factor por variable · diapositiva 29", 960, 140, tam=48, c="tinta", ent="subir"),
    caja_formula("fₖ(x) = Πⱼ fⱼₖ(xⱼ)", y=290, c="teal", tam=56, w=900, en="a"),
    caja_formula("P(Y = cₖ | x) = πₖ Πⱼ fⱼₖ(xⱼ) / Σℓ πℓ Πⱼ fⱼℓ(xⱼ)", y=480, c="violeta", tam=52, w=1500, en="b"),
    P("Π recorre las variables  ·  Σ recorre las clases", 960, 670, "ambar", 46, en="c"), diapo(29, en="a"), sec="5. Un factor por variable")
v.b("Con nuestro correo, d es igual a dos. [[a]]El numerador para spam es 0.20 por diecisiete sobre veintidós, por siete sobre veintidós: exactamente el producto que calculamos con suavizado en la parada anterior.",
    T("d = 2:  0.20 · (17/22) · (7/22) ≈ 0.0492", 960, 840, tam=48, c="rojo", w="b", en="a"), diapo(29))

# ---- 6. Log-probabilidades
v.b("Y aquí aparece un problema práctico, que el script de la clase resuelve sin decirlo en voz alta. Es un cálculo adicional. [[a]]Al multiplicar muchas probabilidades menores que uno, el producto se hace diminuto. [[b]]Cero coma tres elevado a seiscientos dieciocho ya es el número positivo más pequeño que el computador puede representar. Y cero coma tres elevado a seiscientos diecinueve da exactamente cero. [[c]]Con miles de palabras, todos los productos de todas las clases serían cero, y no podríamos normalizar.",
    CL(), T("multiplicar muchas probabilidades desborda", 960, 150, tam=54, c="tinta", ent="subir"),
    T("0.3⁶¹⁸ ≈ 5 × 10⁻³²⁴   (el menor número positivo representable)", 960, 330, tam=44, c="tinta", w="sb", en="b"),
    T("0.3⁶¹⁹ = 0   en el computador", 960, 450, tam=56, c="rojo", w="b", en="b", t=1.5),
    P("con miles de palabras: todos los productos serían 0", 960, 640, "ambar", 46, en="c"), adicional(1050, en="a"), sec="6. Por qué logaritmos")
v.b("La solución es trabajar con logaritmos. [[a]]El logaritmo de un producto es la suma de los logaritmos. [[b]]Seiscientos diecinueve veces el logaritmo de 0.3 es aproximadamente menos 745.26: un número perfectamente manejable. [[c]]Así, el logaritmo del producto de cada clase es el logaritmo de la previa, más la suma de los logaritmos de los factores.",
    CL(), T("producto → suma de logaritmos", 960, 150, tam=56, c="tinta", ent="subir"),
    T("log(0.3⁶¹⁹) = 619 × log(0.3) ≈", 1000, 330, tam=50, c="tinta", w="sb", anc="rm", en="b"),
    N(LG["log_619"], 1030, 330, "{:.2f}", 66, "verde", anc="lm", en="b", t=1.5),
    caja_formula("log sₖ = log πₖ + Σⱼ log fⱼₖ(xⱼ)", y=560, c="violeta", tam=56, w=1200, en="c"), adicional(1050))

# ---- 7. Normalizar en logaritmos
v.b("¿Y cómo se normaliza si tenemos logaritmos? [[a]]El script de la clase lo hace con una función llamada posterior. Primero, a cada logaritmo le resta el mayor de ellos. [[b]]Después aplica la exponencial, y finalmente divide por la suma. [[c]]Restar el máximo no cambia el resultado, porque es dividir todos los productos por el mismo número; pero evita que la exponencial dé cero para todas las clases.",
    CL(), T("normalizar con logaritmos: la función posterior del script", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["def posterior(log_conjunta):", "    z = log_conjunta - np.max(log_conjunta, axis=1, keepdims=True)", "    p = np.exp(z)",
            "    return p / p.sum(axis=1, keepdims=True)"], y=380, w=1600, tam=36, c="violeta", en="a"),
    P("restar el máximo = dividir todos los productos por el mismo número", 960, 660, "teal", 42, en="c"), real(1050, en="a", texto="Código del script de la clase (tramo 00)"),
    sec="7. Normalizar en logaritmos")
v.b("Hagámoslo a mano con el correo de oferta y sin enlace, sin suavizado. [[a]]El logaritmo de 0.064 es menos 2.7489, y el de 0.048, menos 3.0366. [[b]]Restamos el mayor, menos 2.7489: quedan cero y menos 0.2877. [[c]]La exponencial da uno y 0.75. Normalizando, 0.75 dividido por 1.75: 0.4286. El mismo resultado de la parada cuatro, sin haber multiplicado números diminutos.",
    CL(), T("la misma cuenta en logaritmos", 960, 140, tam=54, c="tinta", ent="subir"),
    tabla([["", "legítimo", "spam"], ["log del producto", f"log 0.064 = {l0:.4f}", f"log 0.048 = {l1:.4f}"],
           ["menos el máximo", "0", f"{l1 - l0:.4f}"], ["exponencial", "1", f"{math.exp(l1 - l0):.2f}"]],
          [520, 1010, 1440], 290, paso=110, tam=42, x0=240, x1=1700, ens=[None, "a", "b", "c"]),
    T("P(spam) = 0.75 / (1 + 0.75) ≈", 1000, 790, tam=52, c="tinta", w="sb", anc="rm", en="c", t=1.0),
    N(0.75 / 1.75, 1030, 790, "{:.4f}", 70, "violeta", anc="lm", en="c", t=2.0), adicional(1050, en="a"))

# ---- 8. En el script
v.b("Por eso el script de la clase trabaja con logaritmos de principio a fin. [[a]]Los modelos de scikit-learn tienen un método que entrega directamente log-probabilidades. [[b]]Y la función medir calcula el log loss con esas log-probabilidades. Su comentario lo explica: los logaritmos estables evitan restar de uno probabilidades casi unitarias. [[c]]Es un detalle técnico, pero sin él, las probabilidades de 0.99999 o más se redondearían a uno.",
    CL(), T("el script trabaja con logaritmos de principio a fin", 960, 140, tam=50, c="tinta", ent="subir"),
    codigo(["lp = modelo.predict_log_proba(xv)", "# Logaritmos estables evitan restar de uno probabilidades casi unitarias.",
            "logp = np.column_stack([np.log1p(-q), np.log(q)])"], y=380, w=1660, tam=34, c="violeta", en="a"),
    P("sin logaritmos: 0.99999… se redondearía a 1", 960, 640, "ambar", 44, en="c"), real(1050, en="a", texto="Código del script de la clase (tramos 00 y 05)"),
    sec="8. En el script")

# ---- 9. Error típico
v.b("Errores típicos. [[a]]Uno: multiplicar cientos de probabilidades directamente y obtener cero en todas las clases. [[b]]Dos: olvidar que el logaritmo de la previa también se suma; la previa es un factor más. [[c]]Y tres: tratar la suma de logaritmos como si ya fuera una probabilidad. Es solo un puntaje por clase: todavía hay que normalizar.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("multiplicar cientos de probabilidades directamente", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("olvidar sumar el logaritmo de la previa", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("leer la suma de logaritmos como una probabilidad", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="9. Errores típicos")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tres clases tienen productos 0.02, 0.06 y 0.12. [[b]]¿Cuál es la probabilidad posterior de la tercera clase?",
    CL(), P("productos: 0.02 · 0.06 · 0.12", 960, 330, "teal", 56, en="a"),
    P("posterior de la tercera clase = ?", 960, 520, "violeta", 56, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("La suma es 0.20. [[a]]Cero coma doce dividido por 0.20 [[b]]da 0.6. Y las otras dos: 0.1 y 0.3. Las tres suman uno.",
    CL(), T("0.12 / (0.02 + 0.06 + 0.12) = 0.12 / 0.20 =", 1100, 380, tam=50, c="tinta", w="sb", anc="rm", en="a"),
    N(0.6, 1130, 380, "{:.1f}", 72, "violeta", anc="lm", en="b"),
    P("las otras: 0.1 y 0.3  ·  total 1", 960, 560, "teal", 46, en="b", t=0.8), ilustrativo(1050))

# ---- 11. Puente a los datos reales
v.b("Hasta aquí, la primera mitad de la clase, y la diapositiva treinta la resume. [[a]]Con los cien correos y alfa igual a uno, el correo con oferta y sin enlace tiene probabilidad de spam 0.4140; con umbral 0.50, se clasifica como legítimo. [[b]]Ahora viene la segunda parte: aplicar la receta a cincuenta y cuatro señales reales, y comprobar las probabilidades en correos que no participaron en el ajuste.",
    CL(), T("ya construimos una probabilidad: falta ponerla a prueba · diapositiva 30", 960, 140, tam=44, c="tinta", ent="subir"),
    P("100 correos, α = 1: P(spam | x) ≈ 0.4140 → con umbral 0.50, legítimo", 960, 320, "violeta", 42, en="a"),
    P("segunda parte: 54 señales reales y correos fuera del ajuste", 960, 500, "teal", 46, en="b"), diapo(30, en="a"), sec="11. Fin de la primera parte")

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]ajustar es contar y suavizar; predecir es elegir factores, multiplicar por la previa y normalizar. [[b]]Con K clases, el denominador suma K productos; con d variables, cada producto tiene d factores. [[c]]Y en la práctica se suman logaritmos y se normaliza restando el máximo, para no desbordar.",
    CL(), P("ajustar = contar · predecir = multiplicar y normalizar", 960, 300, "violeta", 46, en="a"),
    P("Π recorre las variables · Σ recorre las clases", 960, 460, "teal", 48, en="b"),
    P("en la práctica: sumar logaritmos y restar el máximo", 960, 620, "ambar", 46, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]Spambase. Cuatro mil seiscientos un correos reales, cincuenta y cuatro señales, y la probabilidad de spam de un correo concreto.",
    CL(), siguiente(7, "Spambase: 54 señales y un correo real"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
