import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_04_codificar.json")
v = V("Codificar categóricas y construir variables")
TE = D["target_enc"]

v.portada("Codificar y construir variables", "Cómo se le explica «prepago» a un modelo, y qué columnas conviene inventar",
          "Tercera parada: codificar las variables categóricas, y construir variables nuevas antes de reducir.")

# ---- 1. Hook
v.b("Un modelo solo entiende números. [[a]]Si le das la palabra «prepago», no sabe qué hacer con ella. [[b]]Hay que traducirla. Y la forma de traducir cambia lo que el modelo entiende.",
    CL(), P("un modelo solo entiende números", 960, 250, "violeta", 56, ent="subir"),
    Bc(560, 540, 480, 200, "blanco", "gris", 5, 30, sombra=True, en="a", ent="subir"), T("«prepago»", 560, 540, tam=60, c="gris", en="a", t=0.1),
    T("?", 960, 540, tam=110, c="rojo", en="a", t=0.4), L(800, 540, 900, 540, "gris", 6, True, en="a", t=0.3),
    Bc(1360, 540, 480, 200, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("0 · 1 · …", 1360, 540, tam=60, c="verde", en="b", t=0.1),
    P("traducir cambia lo que el modelo entiende", 960, 800, "teal", 48, en="b", t=0.6), sec="1. Traducir palabras a números")

# ---- 2. Ordinal
v.b("Primera técnica: la codificación ordinal. [[a]]Se usa cuando hay un orden real. [[b]]Por ejemplo, el monto del plan, de cero, bajo, a nueve, alto. O el nivel educativo: básico, medio y superior.",
    CL(), T("ordinal encoding", 960, 200, tam=60, c="azul", ent="subir"), P("solo si hay un orden real", 960, 320, "azul", 46, en="a"),
    *[[Bc(460 + 250 * i, 560, 200, 150, "azulclaro" if i < 3 else "blanco", "azul", 4, 24, en="b", t=0.12 * i, ent="pop"), T(n, 460 + 250 * i, 530, tam=34, c="tinta", w="sb", en="b", t=0.12 * i + 0.1),
       T(str(i), 460 + 250 * i, 590, tam=44, c="azul", en="b", t=0.12 * i + 0.15)]
      for i, n in enumerate(["básico", "medio", "superior"])],
    L(380, 700, 1040, 700, "azul", 5, True, en="b", t=0.5), T("el orden de los números = el orden real", 710, 760, tam=34, c="azul", w="sb", en="b", t=0.6),
    Bc(1450, 560, 400, 260, "blanco", "violeta", 4, 24, en="b", t=0.8, ent="subir"), T("monto", 1450, 500, tam=44, c="violeta", en="b", t=0.9), T("0 (bajo) … 9 (alto)", 1450, 580, tam=36, c="tinta", w="sb", en="b", t=1.0),
    sec="2. Ordinal: cuando hay orden")
v.b("Ojo con la diapositiva: [[a]]dice «ordinal: nominal a entero». [[b]]Es al revés. La codificación ordinal es para variables ordinales, las que tienen orden.",
    CL(), P("alerta del material", 960, 230, "ambar", 48, ent="subir"), T("«Ordinal: Nominal → entero»", 960, 420, tam=64, c="gris", id="qord", en="a"),
    U("qord", tachado=1.0, en="b", dur=0.6), P("ordinal → para variables con orden", 960, 620, "verde", 54, en="b", t=0.4))

# ---- 3. El problema con nominales
REG = ["Norte", "Sur", "Este", "Oeste"]
v.b("¿Qué pasa si usas números para una variable sin orden? [[a]]Toma la región: Norte cero, Sur uno, Este dos, Oeste tres. [[b]]Ahora, para el modelo, Norte está a tres de Oeste, [[c]]y a solo uno de Sur. Sin ningún motivo.",
    CL(), T("región codificada 0, 1, 2, 3", 960, 200, tam=48, c="tinta", ent="subir"),
    L(360, 520, 1560, 520, "gris", 5, en="a"),
    *[[Bc(360 + 400 * i, 520, 70, 70, "violetaclaro", "violeta", 5, 35, en="a", t=0.12 * i, ent="pop"), T(str(i), 360 + 400 * i, 520, tam=34, c="violeta", en="a", t=0.12 * i + 0.05),
       T(n, 360 + 400 * i, 610, tam=40, c="tinta", w="sb", en="a", t=0.12 * i + 0.1)] for i, n in enumerate(REG)],
    L(360, 420, 1560, 420, "rojo", 6, True, en="b"), T("Norte ↔ Oeste = 3", 960, 370, tam=44, c="rojo", en="b", t=0.3),
    L(360, 700, 760, 700, "verde", 6, True, en="c"), T("Norte ↔ Sur = 1", 560, 760, tam=44, c="verde", en="c", t=0.3),
    P("distancias inventadas", 1360, 850, "ambar", 46, en="c", t=0.8), sec="3. Nominales: one-hot")
v.b("La solución es one-hot. [[a]]Una columna de ceros y unos por cada categoría. Norte es uno, cero, cero, cero. Oeste es cero, cero, cero, uno.",
    CL(), T("one-hot: una columna 0/1 por categoría", 960, 190, tam=48, c="verde", ent="subir"),
    *[T(n, 760 + 200 * j, 300, tam=34, c="verde", w="sb", en="a") for j, n in enumerate(REG)],
    *[[T(n, 600, 390 + 90 * i, tam=36, c="tinta", w="sb", anc="rm", en="a", t=0.1 * i)] +
      [Bc(760 + 200 * j, 390 + 90 * i, 150, 70, "verdeclaro" if i == j else "blanco", "verde" if i == j else "linea", 3, 14, en="a", t=0.1 * i + 0.05 * j, ent="pop") for j in range(4)] +
      [T("1" if i == j else "0", 760 + 200 * j, 390 + 90 * i, tam=36, c="verde" if i == j else "gris", en="a", t=0.1 * i + 0.05 * j + 0.05) for j in range(4)]
      for i, n in enumerate(REG)])
v.b("Calculemos la distancia entre Norte y Oeste. [[a]]Difieren en dos posiciones: uno al cuadrado, más uno al cuadrado, es dos. [[b]]La raíz de dos: uno coma cuarenta y uno. [[c]]Y entre cualquier otro par, también uno coma cuarenta y uno. Ninguna región queda más cerca que otra.",
    R(1060, 390, 790, 84, "ambar", 20, id="rn", ent="fade"), R(1060, 660, 790, 84, "ambar", 20, id="ro", ent="fade", t=0.2),
    T("√(1² + 1²) = √2", 1680, 440, tam=48, c="tinta", en="a"), N(1.41, 1680, 560, "= {:.2f}", 80, "verde", en="b", dur=0.6),
    P("todas a 1.41", 1680, 700, "verde", 46, en="c"))

# ---- 4. Columnas
v.b("Pero one-hot agrega columnas. Hagamos la cuenta con el ejemplo del apunte. [[a]]Región tiene cinco categorías, canal tiene cuatro, y estado es binario. [[b]]Sin quitar ninguna: cinco más cuatro más dos, once columnas.",
    CL(), T("ejemplo del apunte", 960, 190, tam=34, c="gris", w="sb"),
    *[[Bc(x, 400, 380, 200, "blanco", c, 5, 28, sombra=True, en="a", t=t_, ent="subir"), T(n, x, 360, tam=44, c=c, en="a", t=t_ + 0.1), T(d, x, 440, tam=36, c="tinta", w="sb", en="a", t=t_ + 0.2)]
      for x, c, n, d, t_ in ((480, "violeta", "región", "5 categorías", 0.0), (960, "azul", "canal", "4 categorías", 0.2), (1440, "teal", "estado", "binario (2)", 0.4))],
    T("sin drop:  5 + 4 + 2 =", 860, 640, tam=56, c="tinta", en="b"), N(11, 1300, 640, "{:.0f}", 90, "rojo", en="b", t=0.4, dur=0.6), sec="4. Cuántas columnas agrega")
v.b("Con drop igual a first, se quita una columna por variable, porque la última se deduce de las otras. [[a]]Cuatro más tres más uno: ocho columnas.",
    T("drop = 'first':  4 + 3 + 1 =", 860, 800, tam=56, c="tinta"), N(8, 1360, 800, "{:.0f}", 90, "verde", en="a", t=0.3, dur=0.6),
    T("la última categoría se deduce de las demás", 960, 920, tam=34, c="gris", w="sb", t=0.6))
v.b("En los datos de la clase, tipo de plan y estado vienen como uno o dos. [[a]]Son binarias, así que se usa drop igual a if binary: [[b]]cada una queda en una sola columna de ceros y unos. Por eso seguimos en quince columnas.",
    CL(), T("datos de la clase", 960, 190, tam=36, c="gris", w="sb"),
    P("tipo_plan: 1 prepago · 2 plan", 560, 330, "violeta", 44), P("estado: 1 activa · 2 no activa", 1360, 330, "violeta", 44, t=0.2),
    T("OneHotEncoder(drop='if_binary')", 960, 500, tam=44, c="tinta", w="m", en="a"),
    P("tipo_plan_2", 560, 670, "verde", 50, en="b"), P("estado_2", 1360, 670, "verde", 50, en="b", t=0.2), P("siguen siendo 15 columnas", 960, 840, "teal", 50, en="b", t=0.8))

# ---- 5. Target encoding
v.b("¿Y si una variable tiene trescientas categorías, como la comuna? [[a]]One-hot crearía trescientas columnas. [[b]]Aquí entran el target encoding y el frequency encoding: la categoría se reemplaza por un solo número.",
    CL(), T("comuna · 300 categorías", 960, 220, tam=60, c="tinta", ent="subir"),
    *[Bx(300 + 22 * i, 330, 316 + 22 * i, 560, "rojoclaro", "rojo", 2, 4, en="a", t=0.01 * i, ent="fade") for i in range(60)],
    T("one-hot: 300 columnas", 960, 610, tam=40, c="rojo", w="sb", en="a", t=0.6),
    P("target encoding: la tasa de fuga de la comuna", 960, 760, "violeta", 46, en="b"), P("frequency encoding: qué tan frecuente es", 960, 880, "azul", 46, en="b", t=0.5),
    sec="5. Muchas categorías: target y frequency")
v.b("Un ejemplo del apunte. [[a]]La comuna A tiene cuarenta clientes, y ocho se fugaron. La tasa cruda es ocho entre cuarenta: cero coma veinte. [[b]]Pero con pocos clientes, la tasa cruda es ruidosa. Por eso se suaviza hacia la tasa global, que es cero coma quince.",
    CL(), T("ejemplo del apunte · comuna A", 960, 190, tam=36, c="gris", w="sb"),
    T("40 clientes · 8 fugas", 960, 310, tam=56, c="tinta", en="a"), T("cruda = 8 / 40 = 0.20", 960, 430, tam=56, c="rojo", en="a", t=0.5),
    P("se suaviza hacia la tasa global: 0.15", 960, 570, "ambar", 46, en="b"))
v.b("Con un suavizado m igual a veinte: [[a]]ocho más veinte por cero coma quince, es ocho más tres: once. [[b]]Dividido en cuarenta más veinte: sesenta. [[c]]Once sesentavos: cero coma ciento ochenta y tres. Más cerca de la tasa global que el cero coma veinte crudo.",
    T("(8 + 20 · 0.15) / (40 + 20)", 960, 700, tam=54, c="violeta", t=0.1), T("= 11 / 60", 760, 820, tam=56, c="tinta", en="b"), T("= 0.183", 1180, 820, tam=70, c="verde", en="c"),
    R(1180, 820, 300, 110, "verde", 40, id="r183", en="c", t=0.3, ent="fade"))
v.b("Pero cuidado: el target encoding mira la respuesta, la fuga. [[a]]Si lo calculas con todas las filas, la prueba se filtra en la variable. [[b]]La profesora lo probó con sesenta comunas inventadas, sin ninguna relación con la fuga.",
    CL(), P("el target encoding mira «y»", 960, 260, "rojo", 56, ent="subir"), P("con todas las filas: fuga de datos", 960, 430, "rojo", 50, en="a"),
    P("experimento: 60 comunas inventadas, sin señal", 960, 620, "gris", 46, en="b"))
v.b("Con la media del target calculada sin cuidado, [[a]]el AUC en entrenamiento llega a cero coma setenta. [[b]]En la prueba: cero coma cincuenta y uno, puro azar. [[c]]Con TargetEncoder, que usa validación cruzada interna, ya en entrenamiento da cero coma cincuenta y tres: no se engaña.",
    CL(), T("60 comunas sin relación con la fuga · AUC", 960, 180, tam=40, c="gris", w="sb"),
    L(300, 860, 1620, 860, "tinta", 4),
    *[[Bx(x - 80, 860 - (val - 0.4) * 1400, x + 80, 860, col, None, 0, 10, grow="y", en=k, t=t_), N(val, x, 860 - (val - 0.4) * 1400 - 36, "{:.2f}", 46, col, en=k, t=t_ + 0.3, dur=0.6),
       T(lab, x, 900, tam=30, c="gris", w="sb", en=k, t=t_)]
      for x, val, col, lab, k, t_ in ((520, TE["media"][0], "rojo", "entren.", "a", 0.0), (720, TE["media"][1], "gris", "prueba", "b", 0.0),
                                      (1220, TE["targetenc"][0], "verde", "entren.", "c", 0.0), (1420, TE["targetenc"][1], "gris", "prueba", "c", 0.3))],
    T("media del target, sin cuidado", 620, 960, tam=34, c="rojo", w="sb", en="a"), T("TargetEncoder(cv=5)", 1320, 960, tam=34, c="verde", w="sb", en="c"),
    L(300, 860 - 0.1 * 1400, 1620, 860 - 0.1 * 1400, "tinta", 3, guion=True, en="b", t=0.4), T("azar = 0.5", 1720, 860 - 0.1 * 1400, tam=30, c="gris", w="sb", en="b", t=0.6))

# ---- 6. Construir
v.b("Último paso de la representación: construir variables. [[a]]A veces la señal no está en una columna, sino en un cociente entre dos.",
    CL(), P("construir variables (feature engineering)", 960, 260, "teal", 56, ent="subir"), P("la señal puede estar en un cociente", 960, 440, "violeta", 52, en="a"), sec="6. Construir variables")
FE = [("duracion_prom_llamada", "segundos ÷ frec_llamadas", "llamadas cortas o largas"), ("sms_por_antiguedad", "frec_sms ÷ antiguedad", "uso de SMS por mes de antigüedad"),
      ("intensidad_contacto", "n_contactos ÷ frec_llamadas", "actividad concentrada en pocos contactos")]
v.b("La clase construye tres. [[a]]Duración promedio de llamada: segundos dividido en llamadas. [[b]]SMS por antigüedad: mensajes dividido en meses de antigüedad. [[c]]E intensidad de contacto: contactos dividido en llamadas.",
    *[[Bc(960, 600 + 150 * i, 1640, 120, "blanco", "teal", 4, 24, en=k, ent="subir"), T(n, 200, 600 + 150 * i, tam=36, c="teal", w="b", anc="lm", en=k, t=0.1),
       T(f, 960, 600 + 150 * i, tam=34, c="tinta", w="m", en=k, t=0.2), T(d, 1740, 600 + 150 * i, tam=28, c="gris", w="sb", anc="rm", mw=380, en=k, t=0.3)]
      for i, ((n, f, d), k) in enumerate(zip(FE, "abc"))])
v.b("Un ejemplo del apunte: [[a]]un cliente habló siete mil doscientos segundos, en noventa llamadas. [[b]]Siete mil doscientos dividido en noventa: ochenta segundos por llamada.",
    CL(), T("ejemplo del apunte", 960, 200, tam=34, c="gris", w="sb"),
    T("7.200 s en 90 llamadas", 960, 380, tam=64, c="tinta", en="a"), T("7.200 / 90 =", 840, 560, tam=70, c="teal", en="b"), N(80, 1200, 560, "{:.0f}", 110, "teal", suf=" s", en="b", t=0.4, dur=0.7),
    T("por llamada", 960, 700, tam=40, c="gris", w="sb", en="b", t=0.8))
v.b("Un cliente con muchas llamadas cortas se comporta distinto de uno con pocas llamadas largas, [[a]]aunque los dos sumen los mismos segundos. [[b]]Y estas variables se construyen antes de reducir: después de PCA, ya no puedes calcular un cociente.",
    P("mismos segundos, distinto comportamiento", 960, 830, "violeta", 46, en="a"), P("se construyen ANTES de reducir", 960, 950, "rojo", 46, en="b"))
v.b("Con eso, la cuenta de la clase queda así: [[a]]doce variables originales, más tres construidas: quince. [[b]]Y el one-hot de tipo de plan y estado no agrega columnas. Quince columnas listas para escalar y reducir.",
    CL(), T("12 originales", 420, 450, tam=62, c="gris", en="a"), T("+ 3 construidas", 980, 450, tam=62, c="teal", en="a", t=0.3), T("= 15", 1500, 450, tam=90, c="azul", en="a", t=0.6),
    P("one-hot (if_binary): siguen 15", 960, 680, "verde", 50, en="b"))

# ---- 7. Error típico
v.b("El error típico de esta parada: [[a]]calcular el target encoding con todas las filas, antes de dividir. [[b]]Recuerda el experimento: setenta en entrenamiento, cincuenta y uno en la prueba. Es una fuga disfrazada de buena variable.",
    CL(), T("target encoding con todas las filas", 960, 360, tam=64, c="rojo", en="a"), K(960, 500, "cruz", 90, en="a", t=0.5),
    P("0.70 en entrenamiento → 0.51 en prueba", 960, 690, "ambar", 52, en="b"), sec="7. El error típico")

# ---- 8. Piensa
v.b("Piensa un segundo. ¿Qué codificación usarías? [[a]]La talla de una polera: S, M, L, XL. [[b]]El color favorito, entre seis colores. [[c]]Y un código postal, con ochocientos valores.",
    CL(), T("¿qué codificación?", 960, 200, tam=56, c="tinta", ent="subir"),
    *[[Bc(x, 520, 480, 280, "blanco", "violeta", 5, 30, sombra=True, en=k, ent="subir"), T(n, x, 470, tam=48, c="violeta", en=k, t=0.1), T(d, x, 560, tam=36, c="tinta", w="sb", en=k, t=0.2)]
      for x, n, d, k in ((400, "talla", "S · M · L · XL", "a"), (960, "color", "6 sin orden", "b"), (1520, "código postal", "800 valores", "c"))],
    ilustrativo(990, texto="ejemplos ilustrativos"), sec="8. Piensa un segundo", pausa=0.8)
v.b("[[a]]La talla tiene orden: ordinal. [[b]]El color no tiene orden: one-hot, seis columnas, o cinco con drop. [[c]]Y el código postal, con tantas categorías, target o frequency encoding, con validación interna.",
    P("ordinal", 400, 760, "azul", 50, en="a"), P("one-hot", 960, 760, "verde", 50, en="b"), P("target / frequency", 1520, 760, "ambar", 46, en="c"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]ordinal solo si hay un orden real. [[b]]One-hot para categorías sin orden. [[c]]Target encoding con muchas categorías, siempre con validación interna. [[d]]Y las variables construidas, antes de reducir.",
    CL(), P("ordinal: solo con orden real", 960, 260, "azul", 50, en="a"), P("one-hot: sin orden", 960, 420, "verde", 50, en="b"),
    P("target encoding: muchas categorías + validación interna", 960, 580, "ambar", 46, en="c"), P("construir antes de reducir", 960, 740, "teal", 50, en="d"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]PCA, la idea, y una matriz de dos por dos resuelta a mano.",
    CL(), siguiente(5, "PCA: la idea y una matriz 2×2 a mano"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
