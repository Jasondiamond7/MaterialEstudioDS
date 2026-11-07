import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_07_accuracy.json"
v = V("La trampa del accuracy")

v.portada("La trampa del accuracy", "Por qué un número alto puede esconder un modelo inútil",
          "La trampa del accuracy. Por qué un porcentaje alto de aciertos puede esconder un modelo que no sirve.")

# ---------------- 1. El modelo perezoso
v.b("Un hospital quiere detectar una enfermedad rara, que tiene solo el cinco por ciento de los pacientes. [[a]]Alguien propone un modelo muy simple: [[b]]siempre responde «este paciente está sano».",
    CL(), Pt([(95, [260, 320, 1240, 760], "#94A3B8"), (5, [1400, 320, 1640, 760], "rojo")], 12, id="p100", ent="fade", t=0.2),
    P("5 % tiene la enfermedad", 1520, 220, "rojo", 40, mw=500, t=0.3), P("siempre responde: «sano»", 960, 880, "violeta", 52, en="b"), sec="1. El modelo perezoso")
v.b("Sobre mil casos, acierta novecientos cincuenta. [[a]]Su accuracy es de noventa y cinco por ciento. [[b]]Y no detecta ni un solo enfermo.",
    CL(), N(950, 960, 330, "{:.0f}", 190, "verde", suf=" de 1.000", dur=1.0, t=0.2), P("accuracy 95 %", 960, 560, "verde", 80, en="a"),
    P("enfermos detectados: 0", 960, 780, "rojo", 80, en="b"), K(1560, 780, "cruz", 90, en="b", t=0.3))
v.b("Ese es el problema. [[a]]Con clases desbalanceadas, el accuracy es engañoso. Un modelo perezoso saca nota alta, y es inútil.",
    CL(), P("accuracy alto  ≠  modelo útil", 960, 400, "ambar", 70, ent="subir", t=0.2), T("hay que mirar qué errores comete", 960, 640, tam=56, c="gris", w="sb", en="a"))

# ---------------- 2. La matriz
CXm = [620, 990]
RYm = [430, 640]
CELL = {"VN": (0, 0), "FP": (0, 1), "FN": (1, 0), "VP": (1, 1)}
NOMBRE = {"VN": "verdadero negativo", "FP": "falso positivo", "FN": "falso negativo", "VP": "verdadero positivo"}
COLR = {"VN": "verde", "VP": "verde", "FP": "ambar", "FN": "rojo"}
PX = 1560


def matriz(en=None, t0=0.0):
    out = [T("el modelo predijo:", 805, 245, tam=36, c="violeta", w="sb", id="mh0", en=en, t=t0),
           T("no cancela", CXm[0], 295, tam=36, c="tinta", w="sb", id="mh1", en=en, t=t0 + 0.1), T("cancela", CXm[1], 295, tam=36, c="tinta", w="sb", id="mh2", en=en, t=t0 + 0.15),
           T("en realidad:", 230, 245, tam=36, c="teal", w="sb", id="mr0", en=en, t=t0),
           T("no cancela", 410, RYm[0], tam=36, c="tinta", w="sb", anc="rm", id="mr1", en=en, t=t0 + 0.2), T("cancela", 410, RYm[1], tam=36, c="tinta", w="sb", anc="rm", id="mr2", en=en, t=t0 + 0.25)]
    for k, (r, c) in CELL.items():
        out.append(Bc(CXm[c], RYm[r], 360, 200, "blanco", "linea", 4, 22, id=f"m{k}", en=en, t=t0 + 0.3 + 0.1 * (2 * r + c), ent="subir"))
    return out


def celda(k, en, num=None, t=0.0):
    r, c = CELL[k]
    col = COLR[k]
    out = [U(f"m{k}", relleno=col + "claro", borde=col, dur=0.5, en=en, t=t), T(k, CXm[c], RYm[r] - 55, tam=50, c=col, id=f"lk{k}", en=en, t=t + 0.1),
           T(NOMBRE[k], CXm[c], RYm[r] + 5, tam=32, c=col, w="sb", id=f"ln{k}", en=en, t=t + 0.2)]
    if num is not None:
        out.append(N(num, CXm[c], RYm[r] + 58, "{:.0f}", 64, col, id=f"nn{k}", en=en, t=t + 0.3, dur=0.9))
    return out


v.b("La herramienta para ver los errores es la matriz de confusión. [[a]]Las filas son lo que realmente pasó. [[b]]Las columnas, lo que el modelo predijo. Así se ordena en scikit-learn.",
    CL(), matriz(), g("sp", P("filas: lo real", PX, 400, "teal", 42, en="a"), P("columnas: lo predicho", PX, 520, "violeta", 42, en="b")), sec="2. La matriz de confusión")
v.b("Arriba a la izquierda, el verdadero negativo, VN. [[a]]Era un cliente que se queda, y el modelo dijo que se queda. Un acierto.",
    XG("sp", dur=0.25), celda("VN", "a"), g("sp", P("se quedaba · dijo: se queda", PX, 420, "verde", 38, en="a", mw=560), K(PX, 560, "check", 90, en="a", t=0.4)))
v.b("Abajo a la derecha, el verdadero positivo, VP. [[a]]Era un cliente que cancela, y el modelo lo detectó. Otro acierto.",
    XG("sp", dur=0.25), celda("VP", "a"), g("sp", P("cancelaba · lo detectó", PX, 420, "verde", 38, en="a", mw=560), K(PX, 560, "check", 90, en="a", t=0.4)))
v.b("Arriba a la derecha, el falso positivo, FP. [[a]]Se quedaba, pero el modelo dijo que cancela. Es una falsa alarma.",
    XG("sp", dur=0.25), celda("FP", "a"), g("sp", P("se quedaba · dijo: cancela", PX, 420, "ambar", 38, en="a", mw=560), P("falsa alarma", PX, 560, "ambar", 52, en="a", t=0.4)))
v.b("Y abajo a la izquierda, el falso negativo, FN. [[a]]Cancelaba, pero el modelo no lo vio. Es un caso que se escapó.",
    XG("sp", dur=0.25), celda("FN", "a"), g("sp", P("cancelaba · dijo: se queda", PX, 420, "rojo", 38, en="a", mw=560), P("se escapó", PX, 560, "rojo", 52, en="a", t=0.4)))

# ---------------- 3. Club Activo
v.b("Ahora con los doscientos cuarenta clientes de prueba de Club Activo. [[a]]Ciento cincuenta verdaderos negativos. [[b]]Cuarenta falsos positivos. [[c]]Veinticuatro falsos negativos. [[d]]Y veintiséis verdaderos positivos.",
    XG("sp", dur=0.25), celda("VN", "a", 150), celda("FP", "b", 40), celda("FN", "c", 24), celda("VP", "d", 26), g("sp", T("prueba: 240 clientes", PX, 330, tam=40, c="tinta", w="sb", mw=560)),
    sec="3. Las métricas en Club Activo")
v.b("El accuracy es el total de aciertos. [[a]]Veintiséis más ciento cincuenta, entre doscientos cuarenta. [[b]]Da setenta y tres coma tres por ciento.",
    g("hl", R(CXm[0], RYm[0], 390, 230, "ambar", 30, en="a", ent="fade"), R(CXm[1], RYm[1], 390, 230, "ambar", 30, en="a", ent="fade", t=0.2)),
    XG("sp", dur=0.25), g("sp", T("accuracy = (VP + VN) / total", PX, 380, tam=40, c="tinta", w="sb", en="a"), T("(26 + 150) / 240", PX, 500, tam=54, c="tinta", en="a", t=0.3), T("= 73.3 %", PX, 620, tam=80, c="violeta", en="b")))
v.b("La precisión pregunta: de todos los que el modelo marcó como «cancela», [[a]]¿cuántos realmente cancelaban? Marcó sesenta y seis, y acertó veintiséis. [[b]]Es treinta y nueve coma cuatro por ciento.",
    XG("hl", dur=0.25), g("hl", R(CXm[1], 535, 400, 450, "azul", 30, en="a", ent="fade")), XG("sp", dur=0.25),
    g("sp", T("precisión = VP / (VP + FP)", PX, 380, tam=40, c="azul", w="sb", en="a"), T("26 / (26 + 40) = 26 / 66", PX, 500, tam=48, c="tinta", en="a", t=0.3), T("= 39.4 %", PX, 620, tam=80, c="azul", en="b")))
v.b("El recall pregunta: de todos los que realmente cancelaban, [[a]]¿cuántos detectó? Cancelaban cincuenta, y detectó veintiséis. [[b]]Es cincuenta y dos por ciento.",
    XG("hl", dur=0.25), g("hl", R(805, RYm[1], 780, 230, "violeta", 30, en="a", ent="fade")), XG("sp", dur=0.25),
    g("sp", T("recall = VP / (VP + FN)", PX, 380, tam=40, c="violeta", w="sb", en="a"), T("26 / (26 + 24) = 26 / 50", PX, 500, tam=48, c="tinta", en="a", t=0.3), T("= 52.0 %", PX, 620, tam=80, c="violeta", en="b")))
v.b("Mira lo que comparten. [[a]]El numerador es el mismo: veintiséis. [[b]]Solo cambia el total por el que divides. Es como una red de pesca: la precisión mide cuántos peces había en la red, y el recall, cuántos peces del mar sacaste.",
    XG("hl", dur=0.25), XG("sp", dur=0.25),
    g("sp", T("26", PX - 200, 440, tam=100, c="verde", en="a"), T("/ 66", PX - 90, 440, tam=70, c="azul", anc="lm", en="b"), T("26", PX - 200, 600, tam=100, c="verde", en="a"), T("/ 50", PX - 90, 600, tam=70, c="violeta", anc="lm", en="b"),
      T("precisión", PX + 190, 440, tam=34, c="azul", w="sb", en="b", t=0.3), T("recall", PX + 190, 600, tam=34, c="violeta", w="sb", en="b", t=0.3)))
v.b("El F1 combina las dos, con la media armónica. [[a]]Dos por precisión por recall, entre su suma. [[b]]Da cero coma cuarenta y cuatro, ocho.",
    XG("sp", dur=0.25), g("sp", T("F1 = 2·P·R / (P + R)", PX, 380, tam=44, c="tinta", w="sb", en="a"), T("2·0.394·0.52 / (0.394 + 0.52)", PX, 500, tam=36, c="tinta", en="a", t=0.3), T("F1 = 0.448", PX, 640, tam=80, c="verde", en="b")))
v.b("Piensa un segundo: [[a]]si hubiera mil casos más, y el modelo siguiera con el mismo patrón, ¿qué métrica te avisa que se escapan la mitad de los que cancelan?",
    XG("sp", dur=0.25), XG("hl", dur=0.25), g("sp", T("se escapa la mitad", PX, 420, tam=44, c="rojo", w="sb", en="a"), T("¿qué métrica avisa?", PX, 560, tam=52, c="violeta", en="a", t=0.4)), pausa=0.6)
v.b("El recall. Es del cincuenta y dos por ciento: casi la mitad se escapa. [[a]]El accuracy de setenta y tres por ciento no te lo diría.",
    XG("sp", dur=0.25), g("sp", P("recall 52 %", PX, 430, "violeta", 60), P("el accuracy no lo diría", PX, 580, "ambar", 44, en="a", mw=560)))

# ---------------- 4. Ejemplo resuelto del apunte
v.b("Un ejemplo del apunte, con quinientas transacciones, de las cuales veinte son fraudes. [[a]]El modelo detecta doce, deja escapar ocho, [[b]]marca treinta falsas alarmas, y acierta cuatrocientos cincuenta de las que están bien.",
    CL(), matriz(), celda("VP", "a", 12), celda("FN", "a", 8, 0.2), celda("FP", "b", 30), celda("VN", "b", 450, 0.2), g("sp", T("500 transacciones · 20 fraudes", PX, 330, tam=36, c="tinta", w="sb", mw=560)), sec="4. Un ejemplo resuelto")
v.b("El accuracy: doce más cuatrocientos cincuenta, entre quinientos. [[a]]Noventa y dos coma cuatro por ciento. [[b]]La precisión: doce entre cuarenta y dos, veintiocho coma seis. [[c]]El recall: doce entre veinte, sesenta por ciento.",
    g("sp", T("accuracy = 462 / 500 = 92.4 %", PX, 430, tam=36, c="violeta", w="sb", en="a"), T("precisión = 12 / 42 = 28.6 %", PX, 530, tam=36, c="azul", w="sb", en="b"), T("recall = 12 / 20 = 60.0 %", PX, 630, tam=36, c="violeta", w="sb", en="c")))
v.b("Y el F1: [[a]]dos por cero coma doscientos ochenta y seis por cero coma sesenta, entre su suma. [[b]]Cero coma trescientos ochenta y siete.",
    g("sp", T("F1 = 0.387", PX, 760, tam=70, c="verde", en="b")))
v.b("Ahora la línea base: un modelo que dice siempre «no hay fraude». [[a]]Acierta cuatrocientos ochenta de quinientos: noventa y seis por ciento. [[b]]Con recall de cero.",
    CL(),
    Bc(560, 500, 640, 480, "violetaclaro", "violeta", 5, 30, ent="subir", t=0.2), T("nuestro modelo", 560, 320, tam=44, c="violeta", t=0.3), T("accuracy", 560, 420, tam=32, c="gris", w="sb", t=0.4), N(92.4, 560, 490, "{:.1f}", 80, "violeta", suf=" %", t=0.4, dur=0.8),
    T("recall", 560, 600, tam=32, c="gris", w="sb", t=0.5), N(60, 560, 670, "{:.0f}", 80, "violeta", suf=" %", t=0.5, dur=0.8),
    Bc(1360, 500, 640, 480, "rojoclaro", "rojo", 5, 30, en="a", ent="subir"), T("siempre «no fraude»", 1360, 320, tam=44, c="rojo", en="a", t=0.1), T("accuracy", 1360, 420, tam=32, c="gris", w="sb", en="a", t=0.2), N(96.0, 1360, 490, "{:.1f}", 80, "rojo", suf=" %", en="a", t=0.2, dur=0.8),
    T("recall", 1360, 600, tam=32, c="gris", w="sb", en="b"), T("0 %", 1360, 670, tam=80, c="rojo", en="b", t=0.1))
v.b("Mira lo curioso: el modelo pierde en accuracy, noventa y dos contra noventa y seis. [[a]]Pero es el único que detecta fraudes. Siempre hay que comparar contra la línea base.",
    P("pierde en accuracy…", 560, 860, "ambar", 44, t=0.1), P("…pero es el único que detecta fraudes", 1160, 940, "verde", 44, en="a"))

# ---------------- 5. Línea base
v.b("En Club Activo pasa lo mismo. [[a]]Un modelo que dijera siempre «nadie cancela» acertaría el setenta y nueve coma dos por ciento. [[b]]Nuestro modelo saca setenta y tres coma tres: menos.",
    CL(), P("línea base: «nadie cancela»", 960, 260, "ambar", 54, en="a"),
    Bc(560, 560, 600, 360, "blanco", "ambar", 5, 30, sombra=True, en="a", ent="subir"), N(79.2, 560, 560, "{:.1f}", 130, "ambar", suf=" %", en="a", t=0.2, dur=1.0),
    Bc(1360, 560, 600, 360, "blanco", "violeta", 5, 30, sombra=True, en="b", ent="subir"), N(73.3, 1360, 560, "{:.1f}", 130, "violeta", suf=" %", en="b", t=0.2, dur=1.0), sec="5. La línea base")
v.b("¿Entonces el modelo es peor? No. [[a]]La línea base nunca detecta a nadie que cancele. [[b]]Nuestro modelo detecta a la mitad. Se compara con el accuracy, y también con el recall.",
    CL(), P("la línea base detecta: 0 %", 560, 400, "rojo", 48, en="a"), P("nuestro modelo detecta: 52 %", 1360, 400, "verde", 48, en="b"), P("mira accuracy y recall, juntos", 960, 760, "violeta", 56, en="b", t=0.5))
v.b("En la base real de la actividad de repaso, tres mil ciento cincuenta clientes, [[a]]un modelo que dice «nadie se fuga» ya logra ochenta y cuatro coma tres por ciento. [[b]]Así que un ochenta y ocho coma cinco no es tan impresionante como parece.",
    CL(), T("base real · prueba de 788 clientes", 960, 250, tam=44, c="gris", w="sb", ent="subir"),
    Bc(560, 560, 600, 340, "blanco", "ambar", 5, 30, sombra=True, en="a", ent="subir"), T("«nadie se fuga»", 560, 430, tam=42, c="ambar", en="a", t=0.1), N(84.3, 560, 570, "{:.1f}", 120, "ambar", suf=" %", en="a", t=0.2, dur=0.9),
    Bc(1360, 560, 600, 340, "blanco", "violeta", 5, 30, sombra=True, en="b", ent="subir"), T("logística", 1360, 430, tam=42, c="violeta", en="b", t=0.1), N(88.5, 1360, 570, "{:.1f}", 120, "violeta", suf=" %", en="b", t=0.2, dur=0.9))
v.b("Esa logística tiene precisión de setenta y uno coma cuatro, [[a]]y recall de cuarenta y cuatro coma cuatro. F1 de cero coma quinientos cuarenta y siete. [[b]]Con balanced, el accuracy baja a ochenta y dos coma uno, pero el recall sube a ochenta y siete coma nueve.",
    CL(),
    *[[Bc(x, 560, 760, 560, "blanco", c, 5, 30, sombra=True, ent="subir", en=k), T(n, x, 340, tam=46, c=c, en=k, t=0.1)] for x, c, n, k in ((520, "violeta", "logística", None), (1400, "ambar", "balanced", "b"))],
    *[T(lab, 520 - 200 + 0, 440 + 90 * i, tam=34, c="gris", w="sb", anc="lm", t=0.2) for i, lab in enumerate(("accuracy", "precisión", "recall", "F1"))],
    *[T(val, 520 + 200, 440 + 90 * i, tam=46, c="violeta", anc="rm", t=0.3, en="a") for i, val in enumerate(("88.5 %", "71.4 %", "44.4 %", "0.547"))],
    *[T(lab, 1400 - 280, 440 + 90 * i, tam=34, c="gris", w="sb", anc="lm", en="b", t=0.1) for i, lab in enumerate(("accuracy", "precisión", "recall", "F1"))],
    *[T(val, 1400 + 280, 440 + 90 * i, tam=46, c="ambar", anc="rm", en="b", t=0.3) for i, val in enumerate(("82.1 %", "46.4 %", "87.9 %", "0.607"))])
v.b("La balanceada pierde accuracy, pero gana mucho recall. [[a]]Y su F1 es mayor. Aun así, su capacidad de ordenar casos es casi la misma: el área bajo la curva es cero coma novecientos ocho contra cero coma novecientos diez. [[b]]Cambia el punto de operación, no la capacidad de ordenar.",
    CL(), P("balanced: − accuracy, + recall", 960, 280, "ambar", 52, ent="subir", t=0.2), N(0.908, 560, 560, "{:.3f}", 130, "violeta", en="a", dur=0.9), T("AUC logística", 560, 660, tam=36, c="gris", w="sb", en="a"),
    N(0.910, 1360, 560, "{:.3f}", 130, "ambar", en="a", t=0.3, dur=0.9), T("AUC balanceada", 1360, 660, tam=36, c="gris", w="sb", en="a"), P("cambia el punto de operación, no la capacidad", 960, 860, "teal", 46, en="b"))

# ---------------- 6. El costo manda
v.b("Entonces, ¿qué métrica elegir? Depende de cuál error cuesta más. [[a]]Si dejar escapar un caso cuesta mucho, prioriza el recall. [[b]]Si las falsas alarmas son lo caro, prioriza la precisión. [[c]]Y acepta menos de la otra.",
    CL(),
    *[[Bc(x, 480, 620, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 340, tam=54, c=c, en=k, t=0.1), T(d, x, 480, tam=36, c="tinta", w="sb", mw=520, en=k, t=0.25)]
      for x, c, n, d, k in ((520, "violeta", "recall", "dejar escapar un caso es muy caro", "a"), (1400, "azul", "precisión", "una falsa alarma es lo caro", "b"))],
    P("no hay un ranking único", 960, 850, "ambar", 54, en="c"), sec="6. El costo manda")
v.b("Por eso, el accuracy y el F1 no se pueden leer como un ranking único. [[a]]Un modelo puede ganar en uno y perder en otro. [[b]]Lo importante es qué error te duele más en tu negocio.",
    CL(), P("accuracy y F1: no son un ranking único", 960, 330, "violeta", 54, ent="subir"), P("un modelo gana en uno y pierde en otro", 960, 520, "gris", 48, en="a"), P("¿qué error te duele más?", 960, 720, "rojo", 66, en="b"))

# ---------------- 7. Alertas
v.b("Ahora, algunas confusiones que aparecieron en la clase. [[a]]Se dijo «esto es ciento cuarenta y nueve verdaderos positivos». [[b]]Pero la celda de arriba a la izquierda en scikit-learn son los verdaderos negativos: ciento cincuenta. Los verdaderos positivos son veintiséis.",
    CL(), P("confusión de la clase", 960, 230, "ambar", 48, ent="subir"), P("«149 verdaderos positivos»", 960, 400, "rojo", 56, en="a"), K(1600, 400, "cruz", 80, en="a", t=0.3),
    P("arriba a la izquierda = VN = 150", 960, 620, "verde", 52, en="b"), P("VP = 26", 960, 760, "verde", 56, en="b", t=0.5), sec="7. Confusiones de la clase")
v.b("Otra: se dijo que quitar las quejas desbalanceó el modelo. [[a]]No cambia el balance de clases, que sigue en veinte coma ocho por ciento. [[b]]Lo que sí cambia es el rendimiento: el accuracy baja de setenta y tres a cincuenta y dos coma cinco, la precisión de treinta y nueve a veinticuatro coma seis, [[c]]el recall sube de cincuenta y dos a sesenta y dos, y el área bajo la curva cae a cero coma quinientos setenta y cinco.",
    CL(), P("«quitar quejas desbalanceó el modelo»", 960, 230, "rojo", 46, ent="subir"), P("el balance sigue en 20.8 %", 960, 360, "verde", 46, en="a"),
    *[[T(n, 560, 520 + 90 * i, tam=40, c="gris", w="sb", anc="rm", en="b"), T(a_, 800, 520 + 90 * i, tam=48, c="violeta", en="b", t=0.1), T("→", 1000, 520 + 90 * i, tam=48, c="gris", en="b", t=0.1), T(d, 1260, 520 + 90 * i, tam=48, c=c, en="b" if i != 2 else "c", t=0.2)]
      for i, (n, a_, d, c) in enumerate((("accuracy", "73.3", "52.5", "rojo"), ("precisión", "39.4", "24.6", "rojo"), ("recall", "52.0", "62.0", "verde"), ("F1", "44.8", "35.2", "rojo"), ("AUC", "0.685", "0.575", "rojo")))])
v.b("Y otra más: se dijo que un diez por ciento de prueba corre riesgo de sobreajuste. [[a]]No: el sobreajuste depende de la complejidad del modelo. [[b]]Un diez por ciento de prueba da una estimación más ruidosa del error.",
    CL(), P("«90/10 corre riesgo de sobreajuste»", 960, 280, "rojo", 46, ent="subir"), P("el sobreajuste depende de la complejidad", 960, 520, "verde", 50, en="a"), P("poca prueba = estimación ruidosa", 960, 720, "ambar", 50, en="b"))

# ---------------- 8. Cierre
v.b("Para llevarte: [[a]]todas las métricas salen de la matriz de confusión. [[b]]El accuracy engaña cuando una clase domina. [[c]]Compara siempre con la línea base. [[d]]Y elige la métrica según el costo de cada error.",
    CL(), P("todo sale de VN, FP, FN y VP", 960, 260, "violeta", 50, en="a"), P("accuracy engaña si una clase domina", 960, 430, "rojo", 50, en="b"), P("compara con la línea base", 960, 600, "ambar", 50, en="c"), P("elige según el costo del error", 960, 770, "teal", 50, en="d"), sec="8. Para llevar")
v.b("Vamos con la séptima parada: [[a]]cómo elegir el enfoque, y cómo armar un flujo honesto.",
    CL(), Bc(960, 480, 620, 260, "tealclaro", "teal", 6, 40, id="p1", ent="pop", en="a"),
    T("7", 720, 480, tam=120, c="teal", en="a", t=0.1), T("Elegir el enfoque y el flujo honesto", 1040, 480, tam=38, c="tinta", mw=420, en="a", t=0.15), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
