import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_08_variantes.json")
v = V("Variantes de Naive Bayes")
GA, MU, S = D["gauss"], D["multi"], D["spam"]

v.portada("Variantes de Naive Bayes", "Bernoulli, gaussiana y multinomial: la representación decide",
          "Octava parada. El esquema es siempre el mismo: previa por factores, y normalizar. Lo que cambia entre variantes es cómo se calcula cada factor, y eso depende de cómo representamos el correo.")

# ---- 1. La gaussiana
v.b("La diapositiva treinta y siete presenta la variante gaussiana. [[a]]Describe los valores de cada variable con una curva normal dentro de cada clase. [[b]]Qué aprende: una media, que sitúa el centro de la curva, y una dispersión, que determina su amplitud, para cada variable y cada clase. [[c]]Cómo predice: evalúa las alturas de las curvas en los valores del correo, multiplica esas alturas y la previa de clase, y luego normaliza.",
    CL(), T("la variante gaussiana describe valores con curvas · diapositiva 37", 960, 140, tam=46, c="tinta", ent="subir"),
    P("una curva normal por variable y por clase", 960, 270, "teal", 46, en="a"),
    tarjetas([(620, "azul", "qué aprende", "una media (centro) y una dispersión (amplitud)", "b"),
              (1300, "violeta", "cómo predice", "alturas en los valores del correo × previa; normalizar", "c")],
             y=530, w=600, h=320, tam_t=50, tam_d=38), diapo(37, en="a"), sec="1. La gaussiana")
v.b("Una advertencia de la misma diapositiva: [[a]]la altura de la curva es una densidad, no la probabilidad de un valor exacto. Puede incluso ser mayor que uno. [[b]]Y se conserva el supuesto ingenuo: independencia entre variables dentro de cada clase.",
    CL(), P("altura = densidad, no probabilidad de un valor exacto", 960, 360, "ambar", 48, en="a"),
    P("se mantiene la independencia dentro de cada clase", 960, 540, "teal", 48, en="b"), diapo(37, en="a"))

# ---- 2. Ejemplo del script
GG = "gg8"
zs = [i / 20 for i in range(-60, 101)]
f0 = [0.8 * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) for z in zs]
f1 = [0.2 * math.exp(-(z - 2) ** 2 / 2) / math.sqrt(2 * math.pi) for z in zs]
v.b("El script de la clase trae un ejemplo ilustrativo. [[a]]Una medición ficticia, equis igual a 1.5. La clase cero tiene media cero, desviación uno y previa 0.8. La clase uno tiene media dos, desviación uno y previa 0.2. [[b]]El gráfico muestra cada curva ya multiplicada por su previa, y la línea punteada marca el valor 1.5.",
    CL(), T("un ejemplo gaussiano del script (tramo 04)", 960, 130, tam=50, c="tinta", ent="subir"),
    G(300, 230, 1260, 860, -3, 5, 0, 0.35, [dict(xs=zs, ys=f0, color="azul", grosor=6, revela=0.0), dict(xs=zs, ys=f1, color="ambar", grosor=6, revela=0.0),
                                            dict(xs=[1.5, 1.5001], ys=[0, 0.34], color="tinta", grosor=3)],
      xticks=[-2, 0, 1.5, 2, 4], yticks=[0, 0.1, 0.2, 0.3], yfmt="{:.1f}", xlabel="x", id=GG, tam_txt=28), T("densidad ponderada", 300, 190, tam=30, c="tinta", w="sb", anc="lm"),
    U(GG, **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="b", dur=1.8),
    T("clase 0: 0.8 · f0(x)  (media 0, desv. 1)", 1580, 380, tam=34, c="azul", w="sb", en="a"),
    T("clase 1: 0.2 · f1(x)  (media 2, desv. 1)", 1580, 460, tam=34, c="ambar", w="sb", en="a"),
    P("x = 1.5", 1580, 580, "violeta", 46, en="b"), real(1050, en="a", texto="Ejemplo del script de la clase (tramo 04), recalculado"),
    sec="2. Un ejemplo gaussiano")
v.b("Hagamos la cuenta. [[a]]La altura de la curva cero en 1.5: la exponencial de menos 1.5 al cuadrado sobre dos, dividida por la raíz de dos pi: aproximadamente 0.1295. [[b]]La de la curva uno: la exponencial de menos 0.5 al cuadrado sobre dos, sobre la misma raíz: 0.3521. [[c]]Multiplicamos por las previas: 0.8 por 0.1295 da 0.1036, y 0.2 por 0.3521 da 0.0704. Normalizando, la probabilidad de la clase uno es aproximadamente 0.4046.",
    CL(), T("la cuenta gaussiana en x = 1.5", 960, 140, tam=54, c="tinta", ent="subir"),
    tabla([["", "clase 0", "clase 1"], ["densidad", f"{GA['d0']:.4f}", f"{GA['d1']:.4f}"], ["× previa", f"0.8 × {GA['d0']:.4f} = {GA['s0']:.4f}", f"0.2 × {GA['d1']:.4f} = {GA['s1']:.4f}"]],
          [500, 980, 1440], 300, paso=120, tam=44, x0=260, x1=1700, ens=[None, "a", "c"]),
    T("f(x) = exp(−(x − μ)² / 2σ²) / (σ √(2π))", 960, 660, tam=42, c="gris", w="sb", en="a"),
    T("P(clase 1 | x = 1.5) = 0.0704 / (0.1036 + 0.0704) ≈", 1220, 800, tam=44, c="tinta", w="sb", anc="rm", en="c", t=1.0),
    N(GA["post"], 1250, 800, "{:.4f}", 66, "violeta", anc="lm", en="c", t=2.0), real(1050, en="a", texto="Ejemplo del script de la clase (tramo 04), recalculado"))
v.b("Fíjate en algo interesante. [[a]]La densidad de la clase uno en 1.5 es casi tres veces la de la clase cero: el valor está más cerca de su centro. [[b]]Pero la previa de la clase cero es cuatro veces mayor. Y la previa gana: la probabilidad de la clase uno queda en 0.4046, por debajo de un medio.",
    P("densidad: clase 1 ≈ 2.7 × clase 0", 620, 940, "ambar", 40, en="a"), P("previa: clase 0 = 4 × clase 1", 1300, 940, "azul", 40, en="b"))

# ---- 3. Transformar
v.b("Para aplicar la gaussiana a las frecuencias, la diapositiva treinta y ocho las transforma primero. [[a]]Las frecuencias no son negativas, y algunas pueden ser grandes. En la receta gaussiana se usa zeta igual al logaritmo natural de uno más equis. [[b]]Para free en el correo de identificador uno: el logaritmo de 1.32, aproximadamente 0.2776. [[c]]Para money, que vale cero: el logaritmo de uno, que es cero. La transformación comprime los valores grandes y conserva los ceros.",
    CL(), T("transformar la frecuencia antes de ajustar · diapositiva 38", 960, 140, tam=48, c="tinta", ent="subir"),
    caja_formula("z = log(1 + x)   (logaritmo natural)", y=290, c="violeta", tam=54, w=1100, en="a"),
    tabla([["señal del correo 1", "valor x en el CSV", "valor transformado z"], ["free", "0.32", "log(1.32) ≈ 0.2776"], ["money", "0", "log(1) = 0"]],
          [480, 960, 1440], 460, paso=110, tam=42, x0=220, x1=1700, ens=[None, "b", "c"]),
    P("comprime los valores grandes · conserva los ceros", 960, 820, "teal", 44, en="c", t=1.0), diapo(38, en="a"), sec="3. Transformar")
v.b("¿Cuánto comprime? Un cálculo adicional con el valor más grande del archivo: [[a]]una de las palabras del vocabulario llega a 42.81 por ciento en un correo. [[b]]Transformado, el logaritmo de 43.81 es aproximadamente 3.78. [[c]]Medias y dispersiones se aprenden en esta escala. Y la diapositiva advierte: transformar no garantiza normalidad ni independencia; lo dirá la comparación fuera de pliegue. En código, R y Python usan la función log uno pe, que calcula el logaritmo de uno más equis con precisión.",
    CL(), T("el valor más grande del archivo", 960, 150, tam=54, c="tinta", ent="subir"),
    T(f"3d: x = {S['freq_max']:.2f} %   →   z = log(1 + {S['freq_max']:.2f}) ≈", 1080, 330, tam=48, c="tinta", w="sb", anc="rm", en="a"),
    N(math.log1p(S["freq_max"]), 1110, 330, "{:.2f}", 66, "violeta", anc="lm", en="b"),
    P("transformar no garantiza normalidad ni independencia", 960, 520, "ambar", 46, en="c"),
    T("en código: log1p(x) = log(1 + x) con precisión numérica", 960, 680, tam=40, c="gris", w="sb", en="c", t=1.5), adicional(1050, en="a"))

# ---- 4. Alerta: escala de las frecuencias
v.b("Aquí corresponde una precisión sobre la grabación. [[a]]Al hablar de esta transformación, cerca del minuto noventa y seis, se dijo que las frecuencias a lo más pueden ser cero y a lo más pueden ser uno. [[b]]En este archivo las frecuencias son porcentajes: pueden ir de cero a cien, y el valor más alto observado es 42.81. Justamente por eso conviene comprimirlas, como se dijo enseguida.",
    CL(), error_material("01:36:27", "a lo más pueden ser 0 y a lo más pueden ser 1",
                         f"son porcentajes (de 0 a 100); el máximo observado es {S['freq_max']:.2f} (palabra 3d)", en_ok="b"),
    sec="4. Una precisión")

# ---- 5. Multinomial
v.b("La tercera variante es la multinomial, en la diapositiva treinta y nueve. Contar repeticiones requiere el texto. [[a]]Con un vocabulario de tres términos, oferta, reunión y hoy, el mensaje oferta hoy tiene los conteos uno, cero, uno. [[b]]El mensaje oferta oferta hoy tiene dos, cero, uno. [[c]]En Bernoulli, los dos mensajes tienen las mismas presencias: uno, cero, uno. En la multinomial, cada aparición aporta un factor: la repetición sí cambia el cálculo.",
    CL(), T("contar repeticiones requiere el texto · diapositiva 39", 960, 140, tam=48, c="tinta", ent="subir"),
    tabla([["mensaje", "oferta", "reunión", "hoy"], ["oferta hoy", "1", "0", "1"], ["oferta oferta hoy", "2", "0", "1"]],
          [520, 960, 1220, 1460], 300, paso=110, tam=46, x0=260, x1=1640, ens=[None, "a", "b"]),
    P("Bernoulli: ambos (1, 0, 1)  ·  multinomial: la repetición cuenta", 960, 680, "violeta", 44, en="c"),
    T("mensajes construidos para ilustrar; no proceden de Spambase", 960, 800, tam=32, c="gris", w="sb", en="c", t=1.0), diapo(39, en="a"), sec="5. Multinomial")
v.b("El script de la clase hace la cuenta completa. [[a]]Cuenta las palabras de cada clase. Legítimo: cero oferta, cuatro reunión y dos hoy. Spam: cuatro oferta, cero reunión y dos hoy. [[b]]Suaviza sumando uno a cada término y tres al total, porque hay tres términos: en el spam, oferta queda en cinco novenos y reunión en un noveno; en el legítimo, al revés. Hoy queda en tres novenos en las dos clases. [[c]]Y la previa es un medio para cada clase.",
    CL(), T("la multinomial en el script (tramo 04)", 960, 140, tam=52, c="tinta", ent="subir"),
    tabla([["clase", "oferta", "reunión", "hoy", "total"], ["legítimo: conteos", "0", "4", "2", "6"], ["spam: conteos", "4", "0", "2", "6"],
           ["legítimo: (c + 1)/(6 + 3)", "1/9", "5/9", "3/9", ""], ["spam: (c + 1)/(6 + 3)", "5/9", "1/9", "3/9", ""]],
          [430, 860, 1100, 1320, 1530], 290, paso=105, tam=40, x0=130, x1=1720,
          colores=[["tinta"] * 5, ["azul"] + ["tinta"] * 4, ["rojo"] + ["tinta"] * 4, ["azul"] + ["tinta"] * 4, ["rojo"] + ["tinta"] * 4],
          ens=[None, "a", "a", "b", "b"]),
    P("previa: 1/2 para cada clase", 960, 860, "teal", 44, en="c"), real(1050, en="a", texto="Ejemplo del script de la clase (tramo 04), recalculado"))
v.b("Ahora los dos mensajes. [[a]]Oferta hoy: el spam multiplica cinco novenos por tres novenos; el legítimo, un noveno por tres novenos. El cociente es cinco, así que la probabilidad de spam es cinco sextos, aproximadamente 0.8333. [[b]]Oferta oferta hoy: ahora oferta entra dos veces, al cuadrado. El cociente pasa a veinticinco, y la probabilidad, a veinticinco sobre veintiséis: aproximadamente 0.9615. [[c]]En Bernoulli, los dos mensajes habrían recibido exactamente la misma probabilidad.",
    CL(), T("la repetición cambia la probabilidad", 960, 140, tam=54, c="tinta", ent="subir"),
    T("oferta hoy:  (5/9)(3/9) y (1/9)(3/9)  →  5/6 ≈", 1180, 320, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(MU["post"][0], 1210, 320, "{:.4f}", 64, "violeta", anc="lm", en="a", t=2.5),
    T("oferta oferta hoy:  (5/9)²(3/9) y (1/9)²(3/9)  →  25/26 ≈", 1180, 470, tam=40, c="tinta", w="sb", anc="rm", en="b"),
    N(MU["post"][1], 1210, 470, "{:.4f}", 64, "rojo", anc="lm", en="b", t=2.5),
    P("Bernoulli: los dos mensajes, la misma probabilidad", 960, 650, "ambar", 46, en="c"), real(1050, en="a", texto="Ejemplo del script de la clase (tramo 04), recalculado"))

# ---- 6. Analogía
v.b("Una analogía para las tres variantes. [[a]]Piensa en describir un plato por su ajo. Bernoulli pregunta: ¿tiene ajo, sí o no? [[b]]La gaussiana mide: ¿cuántos gramos de ajo tiene? Una medición continua. [[c]]Y la multinomial cuenta: ¿cuántos dientes de ajo? Para contar dientes necesitas el plato; con el porcentaje de ajo, no alcanza.",
    CL(), T("tres formas de describir el ajo de un plato", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(440, "teal", "Bernoulli", "¿tiene ajo? sí / no", "a"), (960, "azul", "gaussiana", "¿cuántos gramos?", "b"),
              (1480, "violeta", "multinomial", "¿cuántos dientes?", "c")], y=480, w=440, h=280, tam_t=48, tam_d=40),
    analogia(1050), sec="6. Analogía: el ajo")

# ---- 7. Qué comparar
v.b("La diapositiva cuarenta aterriza todo en nuestras cincuenta y cuatro frecuencias. [[a]]Bernoulli necesita presencias o ausencias: usamos frecuencia mayor que cero. [[b]]La gaussiana necesita mediciones numéricas: probamos el logaritmo de uno más la frecuencia. [[c]]La multinomial necesita conteos de términos, y eso requiere información adicional: un porcentaje no permite recuperar cuántas veces apareció una palabra sin conocer la longitud del mensaje.",
    CL(), T("la información disponible determina qué comparar · diapositiva 40", 960, 140, tam=46, c="tinta", ent="subir"),
    tabla([["variante", "entrada que necesita", "decisión para este caso"], ["Bernoulli", "presencias o ausencias", "usar frecuencia > 0"],
           ["gaussiana", "mediciones numéricas", "probar log(1 + frecuencia)"], ["multinomial", "conteos de términos", "requiere información adicional"]],
          [380, 900, 1460], 300, paso=115, tam=42, x0=120, x1=1810,
          colores=[["tinta"] * 3, ["teal", "tinta", "tinta"], ["azul", "tinta", "tinta"], ["violeta", "tinta", "rojo"]], ens=[None, "a", "b", "c"]),
    diapo(40, en="a"), sec="7. Qué comparar")
v.b("Por eso, en la clase no se tratan estas frecuencias como conteos de palabras. [[a]]Se comparan Bernoulli y la receta gaussiana, sobre los mismos correos y los mismos pliegues. [[b]]Y la pregunta siguiente es cuál produce mejores probabilidades.",
    CL(), P("Bernoulli frente a gaussiana: mismos correos, mismos pliegues", 960, 360, "violeta", 46, en="a"),
    P("¿cuál produce mejores probabilidades?", 960, 540, "teal", 50, en="b"), diapo(40, en="a"))

# ---- 8. Errores típicos
v.b("Errores típicos con las variantes. [[a]]Uno: leer la altura de la curva gaussiana como si fuera una probabilidad. [[b]]Dos: aplicar la multinomial a porcentajes, como si fueran conteos. [[c]]Y tres: creer que la transformación logarítmica vuelve normales los datos. Comprime, pero no garantiza nada: hay que comprobarlo.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("leer la densidad como probabilidad", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("aplicar la multinomial a porcentajes", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("creer que log(1 + x) vuelve normales los datos", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="8. Errores típicos")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con la misma multinomial del script, el mensaje es oferta oferta oferta hoy. [[b]]¿Cuál es la probabilidad de spam? ¿Y qué daría Bernoulli comparado con oferta hoy?",
    CL(), P("mensaje: oferta oferta oferta hoy  →  conteos (3, 0, 1)", 960, 330, "teal", 46, en="a"),
    P("multinomial: P(spam) = ?   ·   Bernoulli: ¿cambia?", 960, 520, "violeta", 48, en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("Oferta entra tres veces: [[a]]el cociente es cinco al cubo, ciento veinticinco. [[b]]La probabilidad de spam es ciento veinticinco sobre ciento veintiséis, aproximadamente 0.9921. En Bernoulli no cambia nada respecto de oferta hoy: las presencias siguen siendo uno, cero, uno.",
    CL(), T("cociente = 5³ = 125  →  125/126 ≈", 1000, 360, tam=54, c="tinta", w="sb", anc="rm", en="a"), N(125 / 126, 1030, 360, "{:.4f}", 72, "violeta", anc="lm", en="b"),
    P("Bernoulli: presencias (1, 0, 1), igual que «oferta hoy»", 960, 540, "teal", 44, en="b", t=1.0), ilustrativo(1050))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]la representación decide la variante: presencias para Bernoulli, mediciones para la gaussiana, conteos para la multinomial. [[b]]La gaussiana multiplica densidades, no probabilidades; con equis igual a 1.5, la previa gana y da 0.4046. [[c]]Y con frecuencias en porcentaje, la multinomial no se puede aplicar: se comparan Bernoulli y la gaussiana sobre el logaritmo de uno más equis.",
    CL(), P("presencias → Bernoulli · mediciones → gaussiana · conteos → multinomial", 960, 300, "violeta", 42, en="a"),
    P("gaussiana: densidades × previa; x = 1.5 → 0.4046", 960, 460, "teal", 46, en="b"),
    P("con porcentajes: Bernoulli frente a gaussiana sobre log(1 + x)", 960, 620, "ambar", 44, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]comparamos las recetas con predicciones fuera de pliegue, y descubrimos que acertar no es lo mismo que dar buenas probabilidades.",
    CL(), siguiente(9, "Comparar recetas con predicciones fuera de pliegue"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
