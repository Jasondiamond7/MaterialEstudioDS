import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_08_matriz.json"
v = V("Matriz de confusión y métricas")

v.portada("Matriz de confusión y métricas", "Cómo saber de verdad si un clasificador funciona",
          "Matriz de confusión y métricas. Cómo saber de verdad si un modelo de clasificación funciona.")

# ---------------- 1. El 95 % que engaña
v.b("Un modelo de fraude dice que acierta el noventa y cinco por ciento de las veces. [[a]]¿Es bueno? [[b]]Depende de algo que todavía no te he contado.",
    CL(), N(95, 960, 430, "{:.0f}", 260, "violeta", suf=" %", en=None, t=0.2, dur=1.0), T("de aciertos", 960, 640, tam=60, c="gris", w="sb", t=0.8),
    P("¿es bueno?", 960, 850, "ambar", 70, en="a"), sec="1. El 95 % que engaña")
v.b("Imagina un modelo que siempre responde lo mismo: [[a]]no es fraude. Si solo cinco de cada cien son fraudes, [[b]]acierta noventa y cinco veces, [[c]]y no detecta ni un solo fraude.",
    CL(),
    Pt([(95, [260, 280, 1240, 760], "#94A3B8"), (5, [1400, 280, 1640, 760], "rojo")], 13, id="p100", ent="fade", t=0.1),
    P("siempre responde: «no es fraude»", 960, 190, "violeta", 48, en="a"),
    R(750, 520, 1060, 560, "verde", 40, id="r95", en="b", ent="fade"), N(95, 750, 880, "{:.0f}", 70, "verde", suf=" aciertos", en="b", t=0.3, dur=0.8),
    R(1520, 520, 300, 560, "rojo", 40, id="r5", en="c", ent="fade"), T("0 detectados", 1520, 880, tam=48, c="rojo", en="c", t=0.2))
v.b("Por eso no basta mirar un solo número. [[a]]Hay que mirar qué tipo de errores comete el modelo. [[b]]Y para eso existe la matriz de confusión.",
    CL(), P("no basta un solo número", 960, 380, "ambar", 66, ent="subir"), T("hay que ver qué errores comete", 960, 560, tam=56, c="gris", w="sb", en="a"),
    P("matriz de confusión", 960, 780, "violeta", 74, en="b"))

# ---------------- 2. La matriz
CXm = [620, 990]
RYm = [430, 640]
CELL = {"TN": (0, 0), "FP": (0, 1), "FN": (1, 0), "TP": (1, 1)}
NOMBRE = {"TN": "verdadero negativo", "FP": "falso positivo", "FN": "falso negativo", "TP": "verdadero positivo"}
COLR = {"TN": "verde", "TP": "verde", "FP": "ambar", "FN": "rojo"}


def matriz(en=None, t0=0.0):
    out = [T("el modelo dijo:", 805, 245, tam=36, c="violeta", w="sb", id="mh0", en=en, t=t0),
           T("no fraude", CXm[0], 295, tam=36, c="tinta", w="sb", id="mh1", en=en, t=t0 + 0.1), T("fraude", CXm[1], 295, tam=36, c="tinta", w="sb", id="mh2", en=en, t=t0 + 0.15),
           T("en realidad era:", 230, 245, tam=36, c="teal", w="sb", id="mr0", en=en, t=t0),
           T("no fraude", 410, RYm[0], tam=36, c="tinta", w="sb", anc="rm", id="mr1", en=en, t=t0 + 0.2), T("fraude", 410, RYm[1], tam=36, c="tinta", w="sb", anc="rm", id="mr2", en=en, t=t0 + 0.25)]
    for k, (r, c) in CELL.items():
        out.append(Bc(CXm[c], RYm[r], 360, 200, "blanco", "linea", 4, 22, id=f"m{k}", en=en, t=t0 + 0.3 + 0.1 * (2 * r + c), ent="subir"))
    return out


def celda(k, en, nombre=True, num=None, t=0.0):
    r, c = CELL[k]
    col = COLR[k]
    out = [U(f"m{k}", relleno=col + "claro", borde=col, dur=0.5, en=en, t=t),
           T(k, CXm[c], RYm[r] - 55, tam=50, c=col, id=f"lk{k}", en=en, t=t + 0.1)]
    if nombre:
        out.append(T(NOMBRE[k], CXm[c], RYm[r] + 5, tam=34, c=col, w="sb", id=f"ln{k}", en=en, t=t + 0.2))
    if num is not None:
        out.append(N(num, CXm[c], RYm[r] + 55, "{:.0f}", 64, col, id=f"nn{k}", en=en, t=t + 0.3, dur=0.9))
    return out


PX = 1560   # panel derecho
v.b("Cuando un modelo clasifica, hay dos cosas en juego: [[a]]lo que realmente pasó, [[b]]y lo que el modelo dijo. Al cruzarlas, salen cuatro casos.",
    CL(), matriz(), g("sp", T("realidad  ×  predicción", PX, 430, tam=44, c="gris", w="sb", en="b"), T("= 4 casos", PX, 520, tam=70, c="violeta", en="b", t=0.4)),
    sec="2. La matriz de confusión")
v.b("Primero, el verdadero negativo. [[a]]Era una transacción buena, [[b]]y el modelo la dejó pasar. Es un acierto.",
    XG("sp", dur=0.25), celda("TN", "a"), g("sp", P("era una transacción buena", PX, 360, "teal", 40, en="a"), P("el modelo la dejó pasar", PX, 470, "violeta", 40, en="b"), K(PX, 600, "check", 90, "verde", en="b", t=0.5)))
v.b("Segundo, el verdadero positivo. [[a]]Era un fraude, [[b]]y el modelo lo detectó. También es un acierto.",
    XG("sp", dur=0.25), celda("TP", "a"), g("sp", P("era un fraude", PX, 360, "rojo", 40, en="a"), P("el modelo lo detectó", PX, 470, "violeta", 40, en="b"), K(PX, 600, "check", 90, "verde", en="b", t=0.5)))
v.b("Tercero, el falso positivo. [[a]]Era una transacción buena, [[b]]pero el modelo gritó fraude. Es una falsa alarma.",
    XG("sp", dur=0.25), celda("FP", "a"), g("sp", P("era una transacción buena", PX, 360, "teal", 40, en="a"), P("pero el modelo dijo fraude", PX, 470, "violeta", 40, en="b"), P("falsa alarma", PX, 600, "ambar", 54, en="b", t=0.5)))
v.b("Y cuarto, el falso negativo. [[a]]Era un fraude, [[b]]y el modelo no lo vio. Es un fraude que se escapó.",
    XG("sp", dur=0.25), celda("FN", "a"), g("sp", P("era un fraude", PX, 360, "rojo", 40, en="a"), P("el modelo dijo: no es fraude", PX, 470, "violeta", 40, en="b"), P("se escapó", PX, 600, "rojo", 54, en="b", t=0.5)))
v.b("Un truco para no confundirte: [[a]]positivo o negativo es lo que dijo el modelo. [[b]]Verdadero o falso es si acertó.",
    XG("sp", dur=0.25), g("sp", P("positivo / negativo = lo que dijo el modelo", PX, 400, "violeta", 36, en="a", mw=520), P("verdadero / falso = si acertó", PX, 560, "teal", 36, en="b", mw=520)),
    R(CXm[1], 295, 360, 60, "violeta", 20, id="hl1", en="a", ent="fade"))
v.b("Con estas cuatro cifras se calcula todo lo demás.",
    XG("sp", dur=0.25), X("hl1", dur=0.25), g("sp", P("TN  ·  FP  ·  FN  ·  TP", PX, 470, "violeta", 60), T("→ todas las métricas", PX, 580, tam=44, c="gris", w="sb", t=0.4)))

# ---------------- 3. Métricas con un ejemplo
def num(k, val, en, t=0.0):
    r, c = CELL[k]
    return N(val, CXm[c], RYm[r] + 55, "{:.0f}", 64, COLR[k], id=f"nn{k}", en=en, t=t, dur=0.9)


v.b("Un ejemplo con mil transacciones, de las cuales cincuenta son fraudes. [[a]]El modelo detecta treinta: verdaderos positivos. [[b]]Deja escapar veinte: falsos negativos. [[c]]Marca diez veces fraude sin serlo: falsos positivos. [[d]]Y deja pasar bien novecientos cuarenta: verdaderos negativos.",
    XG("sp", dur=0.25), g("sp", T("1.000 transacciones · 50 fraudes", PX, 330, tam=40, c="tinta", w="sb", mw=560, t=0.2)),
    celda("TP", "a"), celda("FN", "b"), celda("FP", "c"), celda("TN", "d"),
    num("TP", 30, "a"), num("FN", 20, "b"), num("FP", 10, "c"), num("TN", 940, "d"), sec="3. Un ejemplo con números")
v.b("La exactitud, o accuracy, es el total de aciertos. [[a]]Verdaderos positivos más verdaderos negativos, [[b]]dividido por el total: treinta más novecientos cuarenta, entre mil. Da cero coma noventa y siete.",
    XG("sp", dur=0.25), g("hl", R(CXm[1], RYm[1], 390, 230, "ambar", 30, en="a", ent="fade"), R(CXm[0], RYm[0], 390, 230, "ambar", 30, en="a", ent="fade", t=0.2)),
    g("sp", T("accuracy = (TP + TN) / n", PX, 380, tam=44, c="tinta", w="sb", en="a"), T("(30 + 940) / 1000", PX, 500, tam=56, c="tinta", en="b"), T("= 0.970", PX, 620, tam=90, c="violeta", en="b", t=0.6)))
v.b("Parece excelente: noventa y siete por ciento. [[a]]Pero fíjate en los fraudes reales: eran cincuenta, [[b]]y veinte se escaparon.",
    XG("hl", dur=0.25), g("hl", R(805, RYm[1], 780, 230, "rojo", 30, en="a", ent="fade")),
    g("sp", P("¡veinte de cincuenta se escaparon!", PX, 800, "rojo", 40, en="b", mw=600)))
v.b("La precisión responde: de todo lo que marqué como fraude, [[a]]¿cuánto realmente lo era? Marcó treinta bien y diez mal: cuarenta. [[b]]Treinta entre cuarenta: cero coma setenta y cinco.",
    XG("sp", dur=0.25), XG("hl", dur=0.25), g("hl", R(CXm[1], 535, 400, 450, "azul", 30, en="a", ent="fade")),
    g("sp", T("precisión = TP / (TP + FP)", PX, 380, tam=44, c="azul", w="sb", en="a"), T("30 / (30 + 10)  =  30 / 40", PX, 500, tam=52, c="tinta", en="b"), T("= 0.75", PX, 620, tam=90, c="azul", en="b", t=0.6)))
v.b("El recall responde: de todos los fraudes que existían, [[a]]¿cuántos detecté? Había cincuenta. Detecté treinta. [[b]]Treinta entre cincuenta: cero coma sesenta.",
    XG("sp", dur=0.25), XG("hl", dur=0.25), g("hl", R(805, RYm[1], 780, 230, "violeta", 30, en="a", ent="fade")),
    g("sp", T("recall = TP / (TP + FN)", PX, 380, tam=44, c="violeta", w="sb", en="a"), T("30 / (30 + 20)  =  30 / 50", PX, 500, tam=52, c="tinta", en="b"), T("= 0.60", PX, 620, tam=90, c="violeta", en="b", t=0.6)))
v.b("Mira lo que tienen en común: [[a]]el mismo numerador, treinta. [[b]]Solo cambia el total por el que divides.",
    XG("sp", dur=0.25), XG("hl", dur=0.25),
    g("sp", T("30", PX - 200, 440, tam=100, c="verde", en="a"), T("/ 40", PX - 90, 440, tam=70, c="azul", anc="lm", en="b"), T("30", PX - 200, 600, tam=100, c="verde", en="a"), T("/ 50", PX - 90, 600, tam=70, c="violeta", anc="lm", en="b"),
      T("precisión", PX + 190, 440, tam=34, c="azul", w="sb", en="b", t=0.3), T("recall", PX + 190, 600, tam=34, c="violeta", w="sb", en="b", t=0.3)))
v.b("Es como una red de pesca. [[a]]La precisión pregunta: de lo que saqué, ¿cuánto eran peces? [[b]]El recall pregunta: de todos los peces del mar, ¿cuántos saqué?",
    XG("sp", dur=0.25), g("sp", P("precisión: de lo que saqué, ¿cuánto eran peces?", PX, 420, "azul", 34, en="a", mw=560), P("recall: de todos los peces, ¿cuántos saqué?", PX, 600, "violeta", 34, en="b", mw=560)))
v.b("El efe uno combina las dos en un solo número, con la media armónica. [[a]]Dos por precisión por recall, dividido por su suma. [[b]]Cero coma noventa entre uno coma treinta y cinco: cero coma seiscientos sesenta y siete.",
    XG("sp", dur=0.25), g("sp", T("F1 = 2 · P · R / (P + R)", PX, 360, tam=44, c="tinta", w="sb", en="a"), T("2 · 0.75 · 0.60 / (0.75 + 0.60)", PX, 480, tam=40, c="tinta", en="a", t=0.4),
      T("= 0.90 / 1.35", PX, 590, tam=56, c="tinta", en="b"), T("= 0.667", PX, 710, tam=90, c="verde", en="b", t=0.5)))
v.b("Ahora compáralo con el modelo que siempre dice no fraude. [[a]]Acierta novecientos cincuenta de mil: cero coma noventa y cinco de exactitud. [[b]]Casi como el cero coma noventa y siete. Pero con recall cero.",
    CL(),
    Bc(560, 480, 640, 520, "violetaclaro", "violeta", 5, 32, ent="subir", t=0.2), T("nuestro modelo", 560, 290, tam=44, c="violeta", t=0.3),
    T("accuracy", 560, 400, tam=32, c="gris", w="sb", t=0.4), N(0.970, 560, 470, "{:.3f}", 80, "violeta", t=0.4, dur=0.8), T("recall", 560, 590, tam=32, c="gris", w="sb", t=0.5), N(0.60, 560, 660, "{:.2f}", 80, "violeta", t=0.5, dur=0.8),
    Bc(1360, 480, 640, 520, "rojoclaro", "rojo", 5, 32, en="a", ent="subir"), T("siempre «no fraude»", 1360, 290, tam=44, c="rojo", en="a", t=0.1),
    T("accuracy", 1360, 400, tam=32, c="gris", w="sb", en="a", t=0.2), N(0.950, 1360, 470, "{:.3f}", 80, "rojo", en="a", t=0.2, dur=0.8),
    T("recall", 1360, 590, tam=32, c="gris", w="sb", en="b"), T("0", 1360, 660, tam=80, c="rojo", en="b", t=0.1), P("¡casi igual de exacto, y no detecta nada!", 960, 900, "ambar", 48, en="b", t=0.6))
v.b("Piensa un segundo: si le pides al modelo que marque como fraude a más transacciones, [[a]]¿qué pasa con el recall, y qué pasa con las falsas alarmas?",
    CL(), T("el modelo marca más fraudes", 960, 380, tam=64, c="tinta", ent="subir", t=0.2), T("recall  →  ?", 960, 560, tam=70, c="violeta", en="a"), T("falsas alarmas  →  ?", 960, 700, tam=70, c="ambar", en="a", t=0.3), pausa=0.5)
v.b("El recall sube: detectas más fraudes. [[a]]Pero aparecen más falsas alarmas. [[b]]Y el efe uno puede mejorar o no: depende del costo de cada error.",
    P("recall sube", 960, 860, "verde", 54, t=0.1), P("más falsas alarmas", 960, 960, "ambar", 54, en="a"), T("↑", 1440, 560, tam=90, c="verde", t=0.2), T("↑", 1440, 700, tam=90, c="ambar", en="a"),
    T("F1: depende del costo", 960, 250, tam=50, c="gris", w="sb", en="b"))

# ---------------- 4. Datos de la clase
v.b("Ahora, la matriz de la clase. Es la regresión logística evaluada en la prueba de quinientos casos. [[a]]Cuatrocientos sesenta y seis verdaderos negativos, [[b]]siete falsos positivos, [[c]]diecisiete falsos negativos, [[d]]y diez verdaderos positivos.",
    CL(), matriz(), celda("TN", "a", True, None, 0.9), celda("FP", "b"), celda("FN", "c"), celda("TP", "d"),
    num("TN", 466, "a"), num("FP", 7, "b"), num("FN", 17, "c"), num("TP", 10, "d"),
    g("sp", T("prueba: 500 casos", PX, 360, tam=44, c="tinta", w="sb", mw=560, t=0.3)), sec="4. Las métricas en el fraude de la clase")
v.b("La exactitud: [[a]]diez más cuatrocientos sesenta y seis, entre quinientos. [[b]]Cero coma novecientos cincuenta y dos.",
    g("sp", T("(10 + 466) / 500", PX, 480, tam=56, c="tinta", en="a"), T("accuracy = 0.952", PX, 600, tam=64, c="violeta", en="b")))
v.b("La precisión: [[a]]diez entre diez más siete, [[b]]cero coma quinientos ochenta y ocho. Y el recall: [[c]]diez entre diez más diecisiete, cero coma trescientos setenta.",
    XG("sp", dur=0.25), g("sp", T("precisión = 10 / 17 = 0.588", PX, 440, tam=44, c="azul", w="sb", en="a"), T("recall = 10 / 27 = 0.370", PX, 560, tam=44, c="violeta", w="sb", en="c")),
    g("hl", R(CXm[1], 535, 400, 450, "azul", 30, en="a", ent="fade"), R(805, RYm[1], 780, 230, "violeta", 30, en="c", ent="fade")))
v.b("Con esas dos, el efe uno sale en [[a]]cero coma cuatrocientos cincuenta y cinco. Un modelo bastante modesto, aunque la exactitud diga cero coma noventa y cinco.",
    XG("hl", dur=0.25), g("sp", T("F1 = 0.455", PX, 700, tam=80, c="verde", en="a"), P("modesto, aunque accuracy sea 0.952", PX, 820, "ambar", 34, en="a", t=0.5, mw=600)))
v.b("Y el modelo que siempre dice no fraude: [[a]]acierta cuatrocientos setenta y tres de quinientos, [[b]]cero coma novecientos cuarenta y seis. Con recall cero. El cero coma novecientos cincuenta y dos apenas lo supera.",
    CL(),
    T("siempre «no fraude»", 960, 250, tam=54, c="rojo", ent="subir"),
    T("473 / 500", 960, 420, tam=100, c="tinta", en="a"), T("= 0.946", 960, 560, tam=100, c="rojo", en="b"), P("recall = 0", 960, 700, "rojo", 60, en="b", t=0.5),
    T("el modelo real: 0.952", 960, 860, tam=54, c="violeta", w="sb", en="b", t=1.2))
v.b("Un detalle: en clase se dijo que el efe uno era cincuenta por ciento. [[a]]La salida real del cuaderno es cero coma cuatrocientos cincuenta y cinco.",
    CL(), T("en clase: F1 ≈ 50 %", 960, 400, tam=70, c="gris", ent="subir"), T("cuaderno: 0.455", 960, 580, tam=100, c="violeta", en="a"))
v.b("Y con solo veintisiete fraudes en la prueba, [[a]]mover el umbral cambia mucho el resultado. [[b]]Por eso el umbral se elige con la validación, no con la prueba.",
    CL(), N(27, 960, 380, "{:.0f}", 200, "rojo", suf=" fraudes", dur=1.0, t=0.2), T("en la prueba de 500", 960, 560, tam=44, c="gris", w="sb", t=0.8),
    P("el umbral cambia mucho el resultado", 960, 700, "ambar", 50, en="a"), P("umbral: se elige con la validación", 960, 840, "azul", 50, en="b"))

# ---------------- 5. Qué métrica elegir
v.b("Entonces, ¿qué métrica elegir? Depende de qué error cuesta más. [[a]]Si dejar escapar un caso es muy costoso, importa el recall. [[b]]Si las falsas alarmas son lo caro, importa la precisión. [[c]]Y si los dos errores pesan, usas el efe uno.",
    CL(),
    [[Bc(x, 470, 520, 380, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 330, tam=50, c=c, en=k, t=0.1), T(d, x, 480, tam=34, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "violeta", "recall", "el falso negativo es costoso", "a"), (960, "azul", "precisión", "la falsa alarma es costosa", "b"), (1560, "verde", "F1", "ambos errores importan", "c"))],
    sec="5. Qué métrica elegir")
v.b("En el caso del banco, con dos por ciento de impago, [[a]]dejar pasar a un mal pagador es muy costoso. [[b]]Así que importa el recall, sin olvidar la precisión.",
    CL(), Pt([(98, [300, 280, 1200, 760], "#94A3B8"), (2, [1380, 280, 1500, 760], "rojo")], 13, id="pb", t=0.1),
    P("2 % de impago", 1440, 850, "rojo", 48, t=0.4), P("un mal pagador que pasa = muy caro", 700, 880, "rojo", 40, en="a"), P("recall, sin olvidar la precisión", 960, 980, "violeta", 48, en="b"))
v.b("Piensa un segundo: si lo caro fuera bloquear a clientes buenos por error, [[a]]¿qué métrica vigilarías?",
    CL(), T("lo caro: bloquear a un cliente bueno", 960, 450, tam=64, c="tinta", ent="subir"), g("qm", T("¿qué métrica?", 960, 640, tam=80, c="violeta", en="a")), pausa=0.5)
v.b("La precisión, porque mide cuántas de las alertas eran de verdad. [[a]]Una precisión alta significa pocas falsas alarmas.",
    XG("qm", dur=0.25), P("precisión", 960, 640, "azul", 80), T("pocas falsas alarmas", 960, 800, tam=54, c="azul", w="sb", en="a"))
v.b("Si en vez de una categoría predices un número, como el precio, usas otras métricas. [[a]]El MAE es el error medio, en las unidades del precio. [[b]]El RMSE castiga más los errores grandes. [[c]]Y el R cuadrado es la proporción de variabilidad explicada.",
    CL(),
    [[Bc(x, 450, 500, 330, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 340, tam=64, c=c, en=k, t=0.1), T(d, x, 470, tam=34, c="tinta", w="sb", mw=400, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "teal", "MAE", "error medio, en unidades del precio", "a"), (960, "ambar", "RMSE", "castiga más los errores grandes", "b"), (1560, "violeta", "R²", "variabilidad explicada", "c"))])
v.b("En la prueba de precios de la clase, el MAE fue trece coma nueve millones de pesos, [[a]]el RMSE, dieciséis coma dos, [[b]]y el R cuadrado, cero coma novecientos cuatro.",
    N(13.9, 360, 700, "{:.1f}", 80, "teal", suf=" M", t=0.2, dur=0.9), N(16.2, 960, 700, "{:.1f}", 80, "ambar", suf=" M", en="a", dur=0.9), N(0.904, 1560, 700, "{:.3f}", 80, "violeta", en="b", dur=0.9),
    P("pesos chilenos, prueba de precios", 960, 880, "gris", 40, t=0.5))
v.b("Aun así, un R cuadrado alto no basta. [[a]]Siempre conviene mirar también el error en las unidades del problema.",
    CL(), P("un R² alto no basta", 960, 430, "ambar", 76, ent="subir"), T("mira también el error en unidades del problema", 960, 640, tam=50, c="gris", w="sb", en="a"))
v.b("Para llevarte: [[a]]toda métrica sale de la matriz de confusión. [[b]]El accuracy engaña cuando una clase domina. [[c]]Y la métrica que eliges depende del error que más te cuesta.",
    CL(), P("todo sale de TN, FP, FN y TP", 960, 330, "violeta", 56, en="a"), P("el accuracy engaña si una clase domina", 960, 510, "rojo", 56, en="b"),
    P("elige según el error más caro", 960, 690, "teal", 56, en="c"), sec="6. Para llevar")

print(v.guardar(OUT), "beats")
