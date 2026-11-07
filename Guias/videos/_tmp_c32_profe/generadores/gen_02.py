import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_02_dimensionalidad.json")
v = V("El problema de la dimensionalidad")

v.portada("El problema de la dimensionalidad", "Por qué demasiadas variables perjudican, en vez de ayudar",
          "Primera parada: el problema de la dimensionalidad. Por qué tener demasiadas variables puede perjudicar a un modelo.")

# ---- 1. Hook
v.b("Imagina que investigas un accidente, y hablas con seis testigos. [[a]]Parece mucha evidencia. [[b]]Pero los seis leyeron la misma noticia en el diario, y te cuentan la misma historia.",
    CL(), P("seis testigos", 960, 230, "violeta", 56, ent="subir"),
    *[[Bc(330 + 252 * i, 470, 180, 180, "blanco", "violeta", 4, 90, en="a", t=0.1 * i, ent="pop"), T("?", 330 + 252 * i, 470, tam=80, c="violeta", en="a", t=0.1 * i + 0.1)] for i in range(6)],
    Bc(960, 760, 900, 150, "ambarclaro", "ambar", 4, 30, en="b", ent="subir"), T("la misma noticia del diario", 960, 760, tam=48, c="ambar", en="b", t=0.1),
    *[L(330 + 252 * i, 570, 960, 680, "ambar", 3, en="b", t=0.4 + 0.05 * i) for i in range(6)],
    sec="1. Seis testigos, una historia")
v.b("En realidad tienes un solo testigo, repetido seis veces. [[a]]Con las variables de un modelo pasa lo mismo. Más columnas no son más información.",
    CL(), T("6 testigos", 600, 420, tam=90, c="gris", id="t6"), L(820, 420, 1020, 420, "gris", 7, True, t=0.3), T("1 historia", 1300, 420, tam=90, c="violeta", t=0.5),
    P("más columnas ≠ más información", 960, 700, "rojo", 60, en="a"))

# ---- 2. Cuatro razones
v.b("¿Por qué sobran las variables? Hay cuatro razones. [[a]]Una: el costo. Más columnas, más memoria y más tiempo, y un modelo más difícil de explicar y de poner en producción.",
    CL(), T("¿por qué sobran variables?", 960, 190, tam=52, c="tinta", ent="subir"),
    Bc(500, 430, 780, 230, "blanco", "azul", 5, 28, sombra=True, en="a", ent="subir"), T("1 · Costo", 500, 380, tam=50, c="azul", en="a", t=0.1),
    T("memoria, tiempo, producción", 500, 460, tam=34, c="tinta", w="sb", en="a", t=0.2), sec="2. Cuatro razones")
v.b("[[a]]Dos: la multicolinealidad. Variables que cuentan la misma historia, y que vuelven inestable a una regresión. [[b]]Tres: el ruido. Columnas que no traen ninguna señal.",
    Bc(1420, 430, 780, 230, "blanco", "violeta", 5, 28, sombra=True, en="a", ent="subir"), T("2 · Multicolinealidad", 1420, 380, tam=50, c="violeta", en="a", t=0.1),
    T("la misma historia, repetida", 1420, 460, tam=34, c="tinta", w="sb", en="a", t=0.2),
    Bc(500, 720, 780, 230, "blanco", "ambar", 5, 28, sombra=True, en="b", ent="subir"), T("3 · Ruido", 500, 670, tam=50, c="ambar", en="b", t=0.1),
    T("columnas sin señal", 500, 750, tam=34, c="tinta", w="sb", en="b", t=0.2))
v.b("[[a]]Y cuatro: sesgo y varianza. Con pocas observaciones, cada variable extra hace más inestables las predicciones, sin que el modelo gane nada a cambio.",
    Bc(1420, 720, 780, 230, "blanco", "rojo", 5, 28, sombra=True, en="a", ent="subir"), T("4 · Sesgo vs. varianza", 1420, 670, tam=50, c="rojo", en="a", t=0.1),
    T("pocas filas: predicciones inestables", 1420, 750, tam=34, c="tinta", w="sb", en="a", t=0.2))

# ---- 3. Multicolinealidad con datos
USO = ["segundos", "frec_llamadas", "n_contactos", "frec_sms", "monto", "clv"]
ABR = ["seg", "llam", "cont", "sms", "monto", "clv"]
CU = D["corr_uso"]


def r_(a, b):
    if a == b:
        return 1.0
    return CU.get(f"{a}|{b}", CU.get(f"{b}|{a}"))


def tono(r):
    f = max(0.0, min(1.0, (r - 0.55) / 0.45))
    base, fin = (237, 233, 254), (124, 58, 237)
    c = tuple(int(base[j] + (fin[j] - base[j]) * f) for j in range(3))
    return "#%02X%02X%02X" % c


CX, CY, CS = 560, 330, 112
celdas = []
for i, a in enumerate(USO):
    for j, b in enumerate(USO):
        r = r_(a, b)
        x, y = CX + j * CS, CY + i * CS
        celdas += [Bc(x, y, CS - 8, CS - 8, tono(r), None, 0, 12, id=f"cc{i}{j}", ent="pop", t=0.03 * (i + j)),
                   T(f"{r:.2f}", x, y, tam=30, c="blanco" if r > 0.8 else "tinta", w="b", id=f"ct{i}{j}", t=0.03 * (i + j) + 0.1)]
etq = [T(ABR[i], CX - 80, CY + i * CS, tam=30, c="gris", w="sb", anc="rm") for i in range(6)] + \
      [T(ABR[j], CX + j * CS, CY - 85, tam=30, c="gris", w="sb") for j in range(6)]
v.b("Veamos la multicolinealidad en los datos de la clase. [[a]]Esta es la correlación entre las seis variables de uso. Todas están entre cero coma sesenta y seis, y cero coma noventa y nueve.",
    CL(), T("correlación entre las 6 variables de uso", 960, 170, tam=40, c="gris", w="sb"), g("mat", celdas, etq),
    g("mat", P(f"entre {D['corr_uso_min']:.2f} y {D['corr_uso_max']:.2f}", 1560, 420, "violeta", 46, en="a"), T("datos de la clase", 1560, 520, tam=30, c="gris", w="sb", en="a", t=0.3)),
    sec="3. Multicolinealidad en los datos")
v.b("La más fuerte: segundos hablados y número de llamadas, [[a]]con cero coma noventa y nueve. [[b]]Es casi una sola variable, con dos nombres.",
    R(CX + 1 * CS, CY + 0 * CS, CS + 10, CS + 10, "rojo", 16, id="r01", en="a", ent="fade"), R(CX + 0 * CS, CY + 1 * CS, CS + 10, CS + 10, "rojo", 16, id="r10", en="a", ent="fade"),
    P("segundos ↔ llamadas: 0.99", 1560, 640, "rojo", 44, en="a", t=0.3), P("una variable, dos nombres", 1560, 760, "ambar", 44, en="b"))

# ---- 4. VIF
v.b("¿Cómo se mide? Con el VIF, el factor de inflación de la varianza. [[a]]Tomas una variable, y tratas de explicarla con todas las demás, con una regresión. [[b]]Eso te da un R cuadrado.",
    CL(), T("VIF · factor de inflación de la varianza", 960, 200, tam=50, c="violeta", ent="subir"),
    Bc(520, 520, 520, 260, "blanco", "violeta", 5, 28, sombra=True, en="a", ent="subir"), T("una variable", 520, 480, tam=44, c="violeta", en="a", t=0.1), T("explicada por las demás", 520, 560, tam=34, c="tinta", w="sb", en="a", t=0.2),
    L(800, 520, 1100, 520, "gris", 7, True, en="b"), T("R²", 1380, 520, tam=140, c="azul", en="b", t=0.2), sec="4. El VIF a mano")
v.b("Y el VIF es uno dividido en uno menos R cuadrado. [[a]]Si es mayor que diez, es una señal fuerte de que esa variable sobra.",
    CL(), caja_formula("VIF = 1 / (1 − R²)", y=330, c="violeta", tam=72), P("VIF > 10: la variable casi se calcula con las otras", 960, 600, "rojo", 46, en="a"))
v.b("Hagamos un ejemplo del apunte. [[a]]Si una variable se explica con R cuadrado cero coma noventa y seis, [[b]]uno menos cero coma noventa y seis es cero coma cero cuatro. [[c]]Y uno dividido en cero coma cero cuatro, da veinticinco.",
    CL(), T("ejemplo del apunte", 960, 190, tam=36, c="gris", w="sb"),
    T("R² = 0.96", 960, 330, tam=70, c="azul", en="a"), T("1 − 0.96 = 0.04", 960, 480, tam=64, c="tinta", en="b"),
    T("VIF = 1 / 0.04 =", 860, 650, tam=70, c="violeta", en="c"), N(25, 1240, 650, "{:.0f}", 110, "rojo", en="c", t=0.4, dur=0.8))
v.b("Veinticinco es mucho más que diez: esa variable sobra. [[a]]En cambio, con R cuadrado cero coma cincuenta, el VIF es uno dividido en cero coma cincuenta: dos. Sin problema.",
    P("25 > 10: sobra", 1560, 650, "rojo", 44), T("R² = 0.50  →  VIF = 1 / 0.50 = 2", 960, 840, tam=56, c="verde", en="a"), K(1640, 840, "check", 70, en="a", t=0.6))
VIF = D["vif"]
v.b("Ahora, los datos reales de la clase. [[a]]Segundos tiene un VIF de cuarenta coma ocho, [[b]]y el número de llamadas, cuarenta y ocho coma tres. [[c]]Contactos, doce coma dos. Las otras tres, monto, mensajes y valor del cliente, están bajo cinco.",
    CL(), T("VIF de las variables de uso · datos de la clase", 960, 190, tam=40, c="gris", w="sb"),
    L(340, 870, 1620, 870, "tinta", 4),
    *[[Bx(x - 70, 870 - VIF[c_] * 11, x + 70, 870, col, None, 0, 10, grow="y", en=k), N(VIF[c_], x, 870 - VIF[c_] * 11 - 34, "{:.1f}", 40, col, en=k, t=0.3, dur=0.6),
       T(ab, x, 905, tam=30, c="gris", w="sb", en=k)]
      for x, c_, ab, col, k in ((440, "segundos", "seg", "rojo", "a"), (640, "frec_llamadas", "llam", "rojo", "b"), (840, "n_contactos", "cont", "ambar", "c"),
                                (1040, "frec_sms", "sms", "verde", "c"), (1240, "monto", "monto", "verde", "c"), (1440, "clv", "clv", "verde", "c"))],
    L(340, 870 - 110, 1620, 870 - 110, "rojo", 3, guion=True, en="a", t=0.6), T("VIF = 10", 1720, 870 - 110, tam=30, c="rojo", w="sb", en="a", t=0.8))
v.b("Un VIF de cuarenta y ocho coma tres quiere decir que el R cuadrado es [[a]]uno menos uno dividido en cuarenta y ocho coma tres: 0.979. [[b]]Casi el noventa y ocho por ciento de las llamadas se explica con las otras variables.",
    CL(), T("VIF = 48.3", 960, 300, tam=80, c="rojo"), T("R² = 1 − 1 / 48.3", 960, 470, tam=64, c="tinta", en="a"),
    N(0.979, 960, 620, "= {:.3f}", 100, "azul", en="a", t=0.5, dur=0.8), P("las llamadas casi se calculan con las demás", 960, 820, "violeta", 48, en="b"))

# ---- 5. Ruido
RU = D["ruido"]
v.b("Ahora el ruido. La profesora hizo un experimento. [[a]]Tomó solo cien clientes, [[b]]y les fue agregando columnas de puro ruido, números al azar. ¿Qué crees que pasa con el modelo?",
    CL(), P("experimento: 100 filas + columnas de ruido", 960, 220, "ambar", 50, ent="subir"),
    Pt([(100, [300, 360, 760, 760], "#94A3B8")], 10, id="cien", visibles=100, en="a", dur=1.0), T("100 clientes", 530, 830, tam=40, c="gris", w="sb", en="a", t=0.6),
    *[Bx(900 + 30 * i, 360, 920 + 30 * i, 760, "ambarclaro", "ambar", 2, 4, en="b", t=0.04 * i, ent="fade") for i in range(24)],
    T("+ columnas al azar", 1250, 830, tam=40, c="ambar", w="sb", en="b", t=0.6), sec="5. Variables de ruido", pausa=0.3)
v.b("En entrenamiento, el modelo mejora y mejora. [[a]]Con cuatrocientas columnas de ruido, el AUC de entrenamiento llega a uno: perfecto. [[b]]Pero en la prueba, cae de 0.715, a 0.569.",
    CL(),
    G(300, 220, 1260, 840, 0, 400, 0.5, 1.0, [dict(xs=RU["ks"], ys=RU["entren"], color="gris", grosor=6, marcadores=9, revela=0.0),
                                                dict(xs=RU["ks"], ys=RU["prueba"], color="rojo", grosor=7, marcadores=10, revela=0.0)],
      xticks=[0, 100, 200, 300, 400], yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], yfmt="{:.1f}", xlabel="columnas de ruido agregadas", id="gr", t=0.1),
    T("AUC", 300, 170, tam=34, c="tinta", w="sb", anc="lm", t=0.2),
    U("gr", **{"series.0.revela": 1.0}, dur=2.0, t=0.4), P("entrenamiento → 1.00", 1600, 330, "gris", 42, en="a"),
    U("gr", **{"series.1.revela": 1.0}, en="b", dur=2.0), P(f"prueba: {RU['prueba'][0]:.3f} → {RU['prueba'][-1]:.3f}", 1600, 560, "rojo", 42, en="b", t=0.8),
    T("promedio de 25 repeticiones", 1600, 650, tam=28, c="gris", w="sb", en="b", t=1.2))
v.b("Eso es sobreajuste. [[a]]El modelo usa el ruido para memorizar a los cien clientes, [[b]]y con clientes nuevos, falla. Con pocas filas, cada columna extra es una oportunidad de memorizar.",
    P("sobreajuste", 1600, 780, "violeta", 54), R(1600, 780, 420, 120, "violeta", 40, id="rsob", en="a", ent="fade"),
    P("memoriza el ruido", 1600, 900, "ambar", 40, en="b"))
v.b("Una regla práctica: mira cuántas filas tienes por columna. [[a]]Con doscientas filas y cuarenta columnas, son doscientas entre cuarenta: cinco filas por columna. Muy pocas. [[b]]Si quitas veinticinco columnas redundantes, quedan quince: doscientas entre quince, trece coma tres.",
    CL(), T("filas por columna = n / p", 960, 200, tam=56, c="violeta", ent="subir"), T("ejemplo del apunte", 960, 280, tam=30, c="gris", w="sb"),
    Bc(560, 560, 640, 360, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("200 / 40", 560, 480, tam=64, c="tinta", en="a", t=0.1), N(5, 560, 620, "= {:.0f}", 100, "rojo", en="a", t=0.4, dur=0.6),
    Bc(1360, 560, 640, 360, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("200 / 15", 1360, 480, tam=64, c="tinta", en="b", t=0.1), N(13.3, 1360, 620, "= {:.1f}", 100, "verde", en="b", t=0.4, dur=0.6),
    T("muy pocas", 560, 800, tam=36, c="rojo", w="sb", en="a", t=0.6), T("mejor", 1360, 800, tam=36, c="verde", w="sb", en="b", t=0.6))

# ---- 6. Distancias
CO = D["conc"]
v.b("Hay un efecto más, muy curioso. [[a]]En muchas dimensiones, todas las distancias se parecen. [[b]]Medimos qué tan lejos está el punto más lejano, comparado con el más cercano.",
    CL(), P("en muchas dimensiones, las distancias se «aplanan»", 960, 220, "teal", 50, ent="subir"),
    Bc(960, 560, 120, 120, "tealclaro", "teal", 5, 60, en="a", ent="pop"), L(960, 560, 700, 450, "teal", 5, True, en="b"), T("más cercano", 640, 410, tam=32, c="teal", w="sb", en="b", t=0.3),
    L(960, 560, 1500, 760, "rojo", 5, True, en="b", t=0.4), T("más lejano", 1560, 820, tam=32, c="rojo", w="sb", en="b", t=0.6),
    T("(lejano − cercano) / cercano", 960, 920, tam=44, c="tinta", en="b", t=0.8), sec="6. Las distancias se aplanan")
v.b("Con puntos al azar en dos dimensiones, [[a]]esa razón vale cuarenta y uno: el más lejano está muchísimo más lejos que el más cercano. [[b]]En diez dimensiones, dos coma siete. [[c]]En cien, cero coma treinta y nueve. [[d]]Y en mil dimensiones, apenas cero coma uno. Todos quedan a casi la misma distancia.",
    CL(), T("(lejano − cercano) / cercano · puntos uniformes", 960, 190, tam=38, c="gris", w="sb"),
    *[[Bc(x, 480, 380, 320, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(f"d = {d}", x, 390, tam=46, c=c, en=k, t=0.1),
       N(CO[d], x, 510, fm, 96, c, en=k, t=0.2, dur=0.8)]
      for x, c, d, fm, k in ((330, "rojo", "2", "{:.0f}", "a"), (750, "ambar", "10", "{:.1f}", "b"), (1170, "teal", "100", "{:.2f}", "c"), (1590, "gris", "1000", "{:.1f}", "d"))],
    T("400 puntos al azar · 15 repeticiones", 960, 700, tam=30, c="gris", w="sb", en="d", t=0.5))
v.b("¿Por qué importa? [[a]]Porque muchos métodos, como K-Means o t-SNE, se basan en «quién está cerca de quién». [[b]]Si todos están igual de cerca, la idea de vecino pierde sentido.",
    P("¿quién está cerca de quién?", 960, 820, "violeta", 50, en="a"), P("si todos están igual de cerca, «vecino» pierde sentido", 960, 940, "rojo", 42, en="b"))

# ---- 7. Manifold y los dos caminos
v.b("¿Y cómo se sale de esto? La profesora usó una imagen bonita: [[a]]un globo terráqueo es un objeto en tres dimensiones. [[b]]Pero lo puedes dibujar en un mapa plano, sin perder las relaciones entre países.",
    CL(), R(520, 540, 380, 380, "azul", 0, g=8, pulso=False, id="globo", en="a", ent="pop"), T("3D", 520, 540, tam=80, c="azul", en="a", t=0.2),
    L(780, 540, 1060, 540, "gris", 7, True, en="b"), Bc(1380, 540, 520, 320, "azulclaro", "azul", 6, 16, en="b", ent="pop", t=0.2), T("mapa 2D", 1380, 540, tam=64, c="azul", en="b", t=0.3),
    sec="7. Manifold y dos caminos")
v.b("Esa estructura más simple, doblada dentro de un espacio grande, se llama manifold. [[a]]Reducir dimensiones es, en el fondo, encontrar y desdoblar esa estructura.",
    P("manifold: estructura simple doblada en un espacio grande", 960, 230, "violeta", 44), P("reducir = encontrar y «desdoblar» esa estructura", 960, 880, "teal", 48, en="a"))
v.b("Para reducir hay dos caminos. [[a]]Uno: la selección. Quitas columnas redundantes, y conservas las originales. Es más fácil de interpretar. [[b]]Dos: la extracción. Combinas columnas en variables nuevas, como los componentes de PCA.",
    CL(),
    tarjetas([(560, "teal", "Selección", "quitar columnas · conservar las originales", "a"), (1360, "azul", "Extracción", "combinar columnas en variables nuevas (PCA, t-SNE)", "b")], w=700, h=420),
    P("más interpretable", 560, 850, "teal", 40, en="a", t=0.6), P("lo que veremos hoy", 1360, 850, "azul", 40, en="b", t=0.6))
v.b("En el ejemplo de los testigos: [[a]]seleccionar es escuchar a uno solo. [[b]]Extraer es escribir un resumen con lo que dijeron los seis.",
    CL(), P("seleccionar = escuchar a un testigo", 960, 400, "teal", 56, en="a"), P("extraer = un resumen de los seis", 960, 600, "azul", 56, en="b"))

# ---- 8. Error típico
v.b("Un error típico: [[a]]confundir variables correlacionadas con variables ruido. Son problemas distintos.",
    CL(), T("correlacionadas  =  ruido", 960, 330, tam=80, c="tinta", id="cr", en="a"), U("cr", tachado=1.0, en="a", t=1.0, dur=0.6), sec="8. El error típico")
v.b("[[a]]Las correlacionadas sí tienen señal, pero repetida: cuentan la misma historia. [[b]]El ruido no aporta nada, aunque no se parezca a ninguna otra variable.",
    tarjetas([(560, "violeta", "correlacionadas", "señal repetida", "a"), (1360, "ambar", "ruido", "nada de señal, aunque no se parezca a otra", "b")], y=640, w=700, h=340))

# ---- 9. Piensa un segundo
v.b("Piensa un segundo: [[a]]una variable se explica con las demás con R cuadrado cero coma noventa. ¿Cuánto vale su VIF?",
    CL(), T("R² = 0.90", 960, 360, tam=100, c="azul", en="a"), T("VIF = ?", 960, 560, tam=100, c="violeta", en="a", t=0.5), sec="9. Piensa un segundo", pausa=0.8)
v.b("Uno dividido en uno menos cero coma noventa: [[a]]uno dividido en cero coma uno, [[b]]igual a diez. Justo en el límite de la regla práctica.",
    CL(), T("VIF = 1 / (1 − 0.90)", 960, 330, tam=70, c="tinta"), T("= 1 / 0.10", 960, 480, tam=70, c="tinta", en="a"),
    N(10, 960, 650, "= {:.0f}", 120, "violeta", en="b", dur=0.6), P("justo en el límite", 960, 830, "ambar", 48, en="b", t=0.6))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]las variables sobran por costo, multicolinealidad, ruido y varianza. [[b]]La multicolinealidad se mide con el VIF: más de diez es mala señal. [[c]]Y se reduce por selección o por extracción.",
    CL(), P("costo · multicolinealidad · ruido · varianza", 960, 300, "violeta", 50, en="a"), P("VIF = 1 / (1 − R²) · más de 10 es mala señal", 960, 480, "rojo", 50, en="b"),
    P("selección o extracción", 960, 660, "teal", 50, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]escalar las variables, antes de reducir.",
    CL(), siguiente(3, "Escalar: estandarizar y normalizar"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
