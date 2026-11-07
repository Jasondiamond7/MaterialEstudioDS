import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, flujo, alerta

OUT = out("guion_c37_01_clasificar.json")
v = V("Clasificar y la línea base")
S = FG["split"]
BB = D["base_benigno"]

v.portada("Clasificar y la línea base", "Antes de cualquier modelo: los datos, la partición y la regla más simple",
          "Clase treinta y siete: máquinas de vectores de soporte y clasificadores clásicos. Empezamos por lo básico: qué significa clasificar, y contra qué regla hay que compararse.")

# ---- 1. Hook
v.b("Una biopsia llega al laboratorio. [[a]]Se miden las células: su tamaño, su textura, su forma. [[b]]Y hay que decidir: ¿benigno o maligno? [[c]]Un clasificador hace exactamente eso: recibe mediciones y devuelve una clase.",
    CL(), Bc(480, 470, 560, 380, "blanco", "teal", 5, 30, sombra=True), T("mediciones", 480, 340, tam=46, c="teal"),
    T("radio · textura", 480, 450, tam=40, c="tinta", w="sb", en="a"), T("suavidad · concavidad", 480, 530, tam=40, c="tinta", w="sb", en="a", t=0.3),
    L(790, 470, 1010, 470, "gris", 7, True, en="b"), Bc(1330, 470, 520, 380, "blanco", "violeta", 5, 30, sombra=True, en="b", t=0.2),
    T("¿benigno o maligno?", 1330, 470, tam=48, c="violeta", en="b", t=0.3),
    P("clasificador: mediciones → una clase", 960, 800, "ambar", 54, en="c"), sec="1. La pregunta")

# ---- 2. Analogía: el detector de humo
v.b("Piensa en un detector de humo. [[a]]Mide partículas en el aire y decide: alarma o silencio. [[b]]Si suena sin fuego, es una molestia: una falsa alarma. [[c]]Si se queda callado con fuego, es una tragedia. Clasificar es decidir, y los dos errores no cuestan lo mismo.",
    CL(), T("el detector de humo", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "mide y decide", "alarma o silencio", "a"), (960, "ambar", "suena sin fuego", "una molestia: falsa alarma", "b"),
              (1520, "rojo", "calla con fuego", "una tragedia", "c")], y=500, w=480, h=330, tam_t=46, tam_d=36),
    P("los dos errores NO cuestan lo mismo", 960, 820, "rojo", 50, en="c", t=1.2),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el detector")

# ---- 3. Aprender de ejemplos
v.b("La diapositiva dos lo dice así: observamos pares, x e y. [[a]]x son los predictores; y es la etiqueta conocida, cero o uno. [[b]]Con esos ejemplos etiquetados se ajusta una regla, {g(x)|ge de equis}. [[c]]Y el objetivo no es acertar en esos ejemplos, sino en casos que el modelo nunca vio.",
    CL(), T("clasificación supervisada · diapositiva 2", 960, 160, tam=50, c="tinta", ent="subir"),
    P("x = predictores (radio, textura…)", 960, 320, "azul", 48, en="a"), P("y = etiqueta conocida: 0 o 1", 960, 440, "violeta", 48, en="a", t=0.5),
    P("ejemplos etiquetados → una regla ŷ = g(x)", 960, 580, "teal", 48, en="b"),
    P("meta: acertar en casos NUEVOS", 960, 720, "verde", 52, en="c"), sec="3. Aprender de ejemplos")
v.b("Un detalle importante: [[a]]que la clase se escriba con un cero y un uno no la convierte en una variable continua. [[b]]Uno no es el doble de cero: son solo nombres para dos grupos.",
    P("0 y 1 son NOMBRES de clases, no cantidades", 960, 870, "ambar", 46, en="a"), T("«uno» no es «el doble de cero»", 960, 960, tam=38, c="gris", w="sb", en="b"))

# ---- 4. Los datos de Wisconsin
v.b("El caso de trabajo es la base de cáncer de mama de Wisconsin. [[a]]Tiene 569 muestras: [[b]]357 benignas, en azul, [[c]]y 212 malignas, en rojo. El uno significa maligno: es la clase positiva, la que queremos detectar.",
    CL(), T("la base de Wisconsin", 960, 150, tam=52, c="tinta", ent="subir"),
    Pt([(357, [240, 250, 1100, 860], "azul"), (212, [1180, 250, 1680, 860], "rojo")], radio=11, visibles=569, id="pts", en="a", dur=2.6),
    N(569, 1800, 330, "{:.0f}", 70, "tinta", en="a", dur=1.2), T("muestras", 1800, 400, tam=32, c="gris", w="sb", en="a"),
    N(357, 1800, 540, "{:.0f}", 64, "azul", en="b", dur=1.0), T("benignas (0)", 1800, 605, tam=32, c="azul", w="sb", en="b"),
    N(212, 1800, 740, "{:.0f}", 64, "rojo", en="c", dur=1.0), T("malignas (1)", 1800, 805, tam=32, c="rojo", w="sb", en="c"),
    real(1050, en="c"), sec="4. Los datos de Wisconsin")
v.b("Se usan cuatro de las treinta mediciones originales: [[a]]radio, textura, suavidad y concavidad, todas en promedio. [[b]]Y una advertencia de la diapositiva cuatro: es una base histórica para enseñar. [[c]]No representa a la población chilena, ni es una herramienta clínica validada.",
    CL(), T("4 de 30 mediciones", 960, 160, tam=52, c="tinta", ent="subir"),
    *[P(n, 300 + 440 * i, 330, c, 46, en="a", t=0.2 * i) for i, (n, c) in enumerate([("radio", "azul"), ("textura", "teal"), ("suavidad", "violeta"), ("concavidad", "ambar")])],
    T("diapositiva 4: base histórica, para enseñar", 960, 560, tam=46, c="tinta", w="sb", en="b"),
    P("no representa a la población chilena · no es una herramienta clínica", 960, 700, "rojo", 42, en="c"))

# ---- 5. Entrenamiento y prueba
RTR, RTE = [200, 280, 1040, 820], [1240, 280, 1720, 820]
v.b("La partición ya viene en el archivo, en la columna conjunto. [[a]]398 casos para entrenar: 250 benignos y 148 malignos. [[b]]Y 171 para la prueba: 107 benignos y 64 malignos. [[c]]Es estratificada: cerca de un setenta y un treinta por ciento, con la misma proporción de malignos en los dos lados.",
    CL(), T("entrenamiento y prueba · diapositiva 5", 960, 150, tam=50, c="tinta", ent="subir"),
    Pt([(357, [240, 260, 1100, 860], "azul"), (212, [1180, 260, 1680, 860], "rojo")], radio=10, id="pts2", dur=0.4),
    U("pts2", grupos=[dict(n=250, rect=RTR[:2] + [RTR[2], 640], color="azul"), dict(n=107, rect=RTE[:2] + [RTE[2], 640], color="azul"),
                      dict(n=148, rect=[RTR[0], 670, RTR[2], RTR[3]], color="rojo"), dict(n=64, rect=[RTE[0], 670, RTE[2], RTE[3]], color="rojo")],
      escalonar=0.6, en="a", dur=2.2),
    T("entrenamiento: 398", 620, 900, tam=46, c="tinta", en="a", t=1.6), T("250 + 148", 620, 960, tam=36, c="gris", w="sb", en="a", t=1.8),
    T("prueba: 171", 1480, 900, tam=46, c="tinta", en="b"), T("107 + 64", 1480, 960, tam=36, c="gris", w="sb", en="b", t=0.2),
    P("malignos: 148/398 = 37.2 %   ·   64/171 = 37.4 %", 960, 1030, "violeta", 34, en="c", t=0.8), sec="5. Entrenamiento y prueba")
v.b("La regla de oro: [[a]]todo lo que el modelo aprende sale del entrenamiento. [[b]]La prueba se guarda, y se usa una sola vez, al final, para evaluar.",
    CL(), P("todo lo que se aprende → del ENTRENAMIENTO", 960, 380, "teal", 54, en="a"), P("la PRUEBA se usa una sola vez, al final", 960, 560, "rojo", 54, en="b"))

# ---- 6. Alerta: el tamaño de la base
v.b("Una precisión sobre la grabación de la clase. [[a]]Se dijo: «con la base de 357 vamos a dividir en 70 y 30». [[b]]Pero 357 son solo los benignos. [[c]]Lo que se divide es la base completa, de 569 casos: 398 más 171.",
    CL(), alerta(170), T("«con la base de 357 vamos a dividir en 70 y 30»", 960, 350, tam=48, c="tinta", w="sb", en="a", id="cita"),
    U("cita", tachado=1.0, en="b", dur=0.8), P("357 = solo los benignos", 960, 520, "azul", 50, en="b", t=0.6),
    P("se divide la base completa: 569 = 398 + 171", 960, 690, "verde", 52, en="c"), T("grabación de la clase · diapositivas 4 y 5", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="c"),
    sec="6. Alerta: el tamaño de la base")

# ---- 7. Salida y decisión
v.b("Ahora, qué entrega cada modelo de esta clase. [[a]]La regresión logística entrega una probabilidad, {p(x)|pe de equis}, y clasifica como uno si supera un umbral, te. [[b]]La SVM entrega un puntaje, {f(x)|efe de equis}, y clasifica como uno si ese puntaje es mayor o igual que cero.",
    CL(), T("salida del modelo y decisión · diapositiva 3", 960, 160, tam=50, c="tinta", ent="subir"),
    Bc(960, 380, 1500, 150, "azulclaro", None, 0, 24, en="a"), T("logística", 380, 380, tam=48, c="azul", anc="lm", en="a", t=0.1),
    T("probabilidad p(x)", 900, 380, tam=44, c="tinta", w="sb", en="a", t=0.2), T("clase 1 si p(x) ≥ t", 1420, 380, tam=44, c="azul", w="sb", en="a", t=0.4),
    Bc(960, 570, 1500, 150, "rojoclaro", None, 0, 24, en="b"), T("SVM", 380, 570, tam=48, c="rojo", anc="lm", en="b", t=0.1),
    T("puntaje f(x)", 900, 570, tam=44, c="tinta", w="sb", en="b", t=0.2), T("clase 1 si f(x) ≥ 0", 1420, 570, tam=44, c="rojo", w="sb", en="b", t=0.4),
    sec="7. Salida y decisión")
v.b("Y la diapositiva insiste en algo que vamos a repetir muchas veces: [[a]]el puntaje de una SVM no es, por sí solo, una probabilidad. [[b]]Un puntaje de dos no significa doscientos por ciento de nada.",
    P("el puntaje de la SVM NO es una probabilidad", 960, 780, "ambar", 50, en="a"), T("f(x) = 2 no significa «200 %»", 960, 880, tam=40, c="gris", w="sb", en="b"))

# ---- 8. La regla más simple
v.b("Antes de entrenar nada, probemos la regla más simple posible: [[a]]decir siempre benigno, a todo el mundo. [[b]]En la prueba hay 107 benignos de 171. [[c]]Exactitud: 107 sobre 171, 0.626. ¡Un 62.6 por ciento sin mirar ni un solo dato!",
    CL(), T("la línea base: «siempre benigno»", 960, 160, tam=52, c="tinta", ent="subir"),
    Pt([(107, [260, 300, 1060, 620], "azul"), (64, [260, 660, 1060, 860], "rojo")], radio=11, visibles=171, id="pb", en="a", dur=1.5),
    T("107 benignos", 1110, 460, tam=40, c="azul", anc="lm", en="b"), T("64 malignos", 1110, 760, tam=40, c="rojo", anc="lm", en="b", t=0.3),
    T("exactitud = 107 / 171 =", 1580, 330, tam=40, c="tinta", w="sb", en="c"), N(62.6, 1580, 420, "{:.1f}", 80, "azul", suf=" %", en="c", t=0.4),
    real(1050, en="c"), sec="8. La regla más simple")
v.b("¿Es un buen modelo? [[a]]Mira los malignos: son 64, y la regla no detecta ni uno. [[b]]Sensibilidad: cero sobre 64, igual a cero. [[c]]Una exactitud que parece alta puede esconder que una clase nunca se identifica.",
    U("pb", atenuar=1.0), R(660, 760, 880, 260, "rojo", r=30, id="rmal", en="a"),
    T("sensibilidad = 0 / 64 =", 1580, 620, tam=40, c="tinta", w="sb", en="b"), T("0", 1580, 710, tam=80, c="rojo", en="b", t=0.4),
    X("rmal", en="c"), P("exactitud alta ≠ buen modelo", 1450, 880, "ambar", 44, en="c"))

# ---- 9. Alerta: ¿siempre uno?
v.b("Otra precisión de la clase. [[a]]Se dijo: «clasificamos todos como uno», y que así se aciertan 107 casos. [[b]]Pero con uno igual a maligno, decir siempre uno acierta solo los 64 malignos: un 37.4 por ciento. [[c]]Los 107 aciertos son de la regla contraria, siempre benigno. La diapositiva siete lo tiene bien.",
    CL(), alerta(170), T("«clasificamos todos como uno» → 107 aciertos", 960, 330, tam=48, c="tinta", w="sb", en="a", id="cita2"),
    U("cita2", tachado=1.0, en="b", dur=0.8),
    T("siempre 1 (maligno): 64 / 171 =", 1000, 490, tam=44, c="rojo", w="sb", anc="rm", en="b", t=0.4), N(37.4, 1030, 490, "{:.1f}", 56, "rojo", suf=" %", anc="lm", en="b", t=0.9),
    T("siempre 0 (benigno): 107 / 171 =", 1000, 620, tam=44, c="azul", w="sb", anc="rm", en="c"), N(62.6, 1030, 620, "{:.1f}", 56, "azul", suf=" %", anc="lm", en="c", t=0.5),
    T("grabación de la clase · diapositiva 7", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="c"), sec="9. Alerta: ¿siempre uno?")

# ---- 10. El error típico
v.b("El error típico: [[a]]celebrar una exactitud alta sin compararla con la línea base. [[b]]El profesor lo ilustró con un banco: si solo el uno por ciento de los clientes son malos pagadores, [[c]]decir siempre «buen pagador» da un 99 por ciento de exactitud, y no sirve para nada.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("celebrar la exactitud sin mirar la línea base", 410, 230, tam=46, c="rojo", anc="lm", en="a"),
    Pt([(99, [460, 360, 1460, 760], "verde"), (1, [1500, 700, 1560, 760], "rojo")], radio=13, visibles=100, en="b", dur=1.4),
    T("99 % buenos pagadores · 1 % malos", 960, 820, tam=40, c="tinta", w="sb", en="b", t=0.8),
    P("«siempre buen pagador» → 99 % de exactitud… y no detecta a nadie", 960, 930, "ambar", 42, en="c"),
    T("ejemplo de la clase", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="c"), sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si en la prueba hubiera 150 benignos y 21 malignos, ¿qué exactitud tendría la regla siempre benigno? [[b]]¿Y qué sensibilidad?",
    CL(), P("150 benignos + 21 malignos", 960, 300, "gris", 56, en="a"), P("«siempre benigno»: exactitud = ?", 960, 460, "azul", 52, en="a", t=0.6),
    P("sensibilidad = ?", 960, 600, "rojo", 52, en="b"), sec="11. Piensa un segundo", pausa=0.6)
v.b("Exactitud: 150 sobre 171, [[a]]0.877, casi un 88 por ciento. [[b]]Y la sensibilidad sigue siendo cero. [[c]]Mientras más desbalanceada es la base, más engaña la exactitud.",
    CL(), T("exactitud = 150 / 171 =", 1000, 300, tam=52, c="tinta", w="sb", anc="rm"), N(0.877, 1030, 300, "{:.3f}", 66, "azul", anc="lm", en="a"),
    T("sensibilidad = 0 / 21 =", 1000, 450, tam=52, c="tinta", w="sb", anc="rm", en="b"), T("0", 1030, 450, tam=66, c="rojo", anc="lm", en="b", t=0.4),
    P("más desbalance → más engaña la exactitud", 960, 650, "ambar", 52, en="c"), ilustrativo(1000))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]clasificar es decidir una clase, y los errores no cuestan lo mismo. [[b]]La base tiene 569 casos, con una partición fija de 398 y 171. [[c]]Y todo modelo se compara con la línea base: siempre benigno, 62.6 por ciento de exactitud y cero de sensibilidad.",
    CL(), P("clasificar = decidir; los errores no cuestan igual", 960, 300, "violeta", 48, en="a"),
    P("569 casos · entrenamiento 398 · prueba 171", 960, 460, "teal", 48, en="b"),
    P("línea base: 62.6 % de exactitud, sensibilidad 0", 960, 620, "ambar", 48, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]la matriz de confusión, y las métricas que salen de ella.",
    CL(), siguiente(2, "La matriz de confusión y las métricas"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
