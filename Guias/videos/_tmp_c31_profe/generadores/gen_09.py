import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_09_repaso.json"
v = V("Repaso final")

v.portada("Repaso final", "Las ideas de la clase, conectadas en un solo recorrido",
          "Repaso final de la clase treinta y uno. Vamos a conectar todo en un solo recorrido.")

v.b("Volvamos a las tres preguntas de la clase. [[a]]¿Qué tipo de problema es? [[b]]¿Qué datos tengo? [[c]]¿Cómo sabré que el modelo sirve? Las seis ideas que siguen las responden.",
    CL(), T("¿qué tipo de problema es?", 960, 330, tam=64, c="violeta", en="a"), T("¿qué datos tengo?", 960, 470, tam=64, c="teal", en="b"), T("¿cómo sabré que el modelo sirve?", 960, 610, tam=64, c="ambar", en="c"), sec="1. Las seis ideas")

IDEAS = [("El tipo de problema decide el modelo", "categoría: clasificación · número: regresión · sin etiqueta: clustering o reducción · recompensas: refuerzo", "violeta"),
         ("Probabilidad antes que clase", "puntaje z, sigmoide 1 / (1 + e^(−z)), y el umbral decide", "azul"),
         ("Siempre se mide el error en datos nuevos", "entrenar con un conjunto, evaluar con otro; todo lo que se aprende, solo con entrenamiento", "teal"),
         ("K-Means: asignar y promediar", "cada punto al centro más cercano; cada centro, la media de su grupo; estandarizar antes", "verde"),
         ("Complejidad y penalización", "mucha complejidad memoriza ruido; Ridge encoge y Lasso anula", "ambar"),
         ("El accuracy no basta", "matriz de confusión, recall, precisión y una línea base", "rojo")]
CXs = [500, 1420]
CYs = [330, 600, 870]


def idea(i, en, t=0.0):
    n, d, c = IDEAS[i]
    x, y = CXs[i % 2], CYs[i // 2]
    return [Bc(x, y, 860, 230, "blanco", c, 5, 30, sombra=True, id=f"id{i}", en=en, ent="subir", t=t), T(str(i + 1), x - 365, y, tam=100, c=c, id=f"in{i}", en=en, t=t + 0.1),
            T(n, x + 40, y - 56, tam=36, c=c, mw=680, id=f"it{i}", en=en, t=t + 0.15), T(d, x + 40, y + 38, tam=27, c="tinta", w="sb", mw=680, id=f"id_{i}", en=en, t=t + 0.25)]


def anillo(i, en=None):
    x, y = CXs[i % 2], CYs[i // 2]
    return [X("ri", en=en, dur=0.25), R(x, y, 900, 270, "ambar", 36, id="ri", en=en, t=0.3, ent="fade")]


v.b("Uno: [[a]]el tipo de problema decide el modelo. Categoría, clasificación. Número, regresión. Sin etiqueta, clustering o reducción. Con recompensas, refuerzo.",
    CL(), idea(0, "a"))
v.b("Dos: [[a]]la logística calcula un puntaje, la sigmoide lo vuelve probabilidad, y el umbral decide. Con puntaje cero, la probabilidad es cero coma cinco.",
    idea(1, "a"), anillo(1, "a"))
v.b("Tres: [[a]]siempre se mide el error en datos nuevos. Todo lo que se aprende, solo con el entrenamiento.",
    idea(2, "a"), anillo(2, "a"))
v.b("Cuatro: [[a]]K-Means asigna cada punto al centro más cercano, y mueve cada centro al promedio. Se estandariza antes.",
    idea(3, "a"), anillo(3, "a"))
v.b("Cinco: [[a]]demasiada complejidad memoriza ruido. Ridge encoge los coeficientes, y Lasso puede anularlos.",
    idea(4, "a"), anillo(4, "a"))
v.b("Y seis: [[a]]el accuracy no basta. Se mira la matriz de confusión, el recall, la precisión, y una línea base.",
    idea(5, "a"), anillo(5, "a"))

# ---- Números clave
NUM = [("Club Activo", "800 clientes · 20.9 % cancela", "violeta"), ("Prueba", "accuracy 73.3 · precisión 39.4 · recall 52.0 · F1 44.8 · AUC 0.685", "azul"),
       ("Regresión", "MSE 144.9 · RMSE 12.0 · R² 0.548", "teal"), ("K-Means", "segmentos de 252, 439 y 109 · el 2 cancela 7 %", "verde"),
       ("Sobreajuste", "grado 15: entrenamiento 0.90, prueba −4.79", "ambar"), ("Línea base", "«nadie cancela» 79.2 % frente a 73.3 %", "rojo")]
v.b("Y estos son los números que conviene recordar. [[a]]En Club Activo, ochocientos clientes, y veinte coma nueve por ciento cancela. [[b]]En la prueba: accuracy setenta y tres coma tres, precisión treinta y nueve coma cuatro, recall cincuenta y dos.",
    CL(), P("números clave", 960, 160, "teal", 44),
    *[[Bc(960, 300 + 120 * i, 1640, 96, "blanco", c, 4, 22, en=k, ent="subir"), T(n, 260, 300 + 120 * i, tam=34, c=c, anc="lm", en=k, t=0.1), T(d, 520, 300 + 120 * i, tam=30, c="tinta", w="sb", anc="lm", mw=1250, en=k, t=0.2)]
      for i, (n, d, c), k in ((0, NUM[0], "a"), (1, NUM[1], "b"))], sec="2. Los números a recordar")
v.b("En regresión: error cuadrático medio ciento cuarenta y cuatro coma nueve, RMSE doce, R cuadrado cero coma quinientos cuarenta y ocho. [[a]]K-Means: segmentos de doscientos cincuenta y dos, cuatrocientos treinta y nueve y ciento nueve clientes. [[b]]El segundo cancela solo siete por ciento.",
    *[[Bc(960, 300 + 120 * i, 1640, 96, "blanco", c, 4, 22, en=k, ent="subir", t=0.0 if k is None else 0.0), T(n, 260, 300 + 120 * i, tam=34, c=c, anc="lm", en=k, t=0.1), T(d, 520, 300 + 120 * i, tam=30, c="tinta", w="sb", anc="lm", mw=1250, en=k, t=0.2)]
      for i, (n, d, c), k in ((2, NUM[2], None), (3, NUM[3], "a"))])
v.b("Con grado quince, el entrenamiento llega a cero coma noventa, y la prueba cae a menos cuatro coma setenta y nueve. [[a]]Y la línea base, «nadie cancela», acierta setenta y nueve coma dos, más que nuestro modelo. Pero no detecta a nadie.",
    *[[Bc(960, 300 + 120 * i, 1640, 96, "blanco", c, 4, 22, en=k, ent="subir"), T(n, 260, 300 + 120 * i, tam=34, c=c, anc="lm", en=k, t=0.1), T(d, 520, 300 + 120 * i, tam=30, c="tinta", w="sb", anc="lm", mw=1250, en=k, t=0.2)]
      for i, (n, d, c), k in ((4, NUM[4], None), (5, NUM[5], "a"))])

# ---- Conexión
v.b("Fíjate cómo se conecta todo. [[a]]Defines el problema. [[b]]Preparas los datos y divides. [[c]]Ajustas el modelo. [[d]]Y evalúas con la métrica correcta, contra una línea base. Todo, dentro de un Pipeline.",
    CL(),
    [[Bc(x, 500, 340, 190, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 470, tam=44, c=c, en=k, t=0.1), T(d, x, 550, tam=28, c="gris", w="sb", mw=290, en=k, t=0.2)]
     for x, c, n, d, k in ((330, "violeta", "Problema", "tipo y etiqueta", "a"), (780, "teal", "Datos", "preparar y dividir", "b"), (1230, "azul", "Modelo", "ajustar con entrenamiento", "c"), (1680, "rojo", "Evaluar", "métrica y línea base", "d"))],
    [L(510 + 450 * i, 500, 600 + 450 * i, 500, "gris", 6, True, en=k, t=0.3) for i, k in enumerate("bcd")],
    R(1005, 500, 1820, 330, "teal", 50, id="rp", en="d", t=0.8, ent="fade"), P("Pipeline", 1005, 290, "teal", 56, en="d", t=0.9), sec="3. Cómo se conecta")
v.b("Piensa un segundo: [[a]]un modelo de fraude tiene noventa y ocho por ciento de accuracy, y una línea base que siempre dice «no fraude» tiene noventa y siete. ¿Qué miras antes de aplaudir?",
    CL(), T("modelo 98 %", 560, 420, tam=70, c="violeta", en="a"), T("línea base 97 %", 1360, 420, tam=70, c="ambar", en="a"), T("¿qué miras antes de aplaudir?", 960, 640, tam=64, c="tinta", en="a", t=0.6), T("cifras ilustrativas", 960, 820, tam=30, c="gris", w="sb", en="a", t=0.8), pausa=0.6)
v.b("La matriz de confusión y el recall. [[a]]La diferencia con la línea base es de un punto. Hay que ver si el modelo realmente detecta casos.",
    CL(), P("matriz de confusión · recall", 960, 400, "violeta", 70, ent="subir"), P("¿detecta casos de verdad?", 960, 640, "teal", 60, en="a"))

# ---- Alertas
ALR = [("«precisión 51 % es como lanzar una moneda»", "la comparación es con la prevalencia, 20.8 %; AUC 0.685 frente a 0.5", "rojo"),
       ("«esto es 149 verdaderos positivos»", "la celda superior izquierda son los VN (150); los VP son 26", "rojo"),
       ("«PCA separa tres clústeres»", "los formó K-Means; PCA solo cambia los ejes. Y 91.8 % es de la varianza", "ambar"),
       ("«cuántos meses le quedan antes de cancelar» es regresión lineal", "es tiempo hasta el evento con datos censurados: análisis de supervivencia", "ambar")]


def alr(i, en, t=0.0):
    f, c_, col = ALR[i]
    y = 330 + 170 * i
    return [Bc(960, y, 1700, 140, "blanco", col, 4, 24, sombra=True, en=en, ent="subir", t=t), T(f, 220, y - 28, tam=34, c=col, w="sb", anc="lm", mw=1500, en=en, t=t + 0.1), T(c_, 220, y + 26, tam=30, c="tinta", w="sb", anc="lm", mw=1500, en=en, t=t + 0.2)]


v.b("Antes de cerrar, cuatro ideas de la clase que conviene corregir. [[a]]Una precisión de cincuenta y uno por ciento no es una moneda: el azar tiene la precisión de la prevalencia, veinte coma ocho. [[b]]Y los verdaderos positivos son veintiséis, no ciento cuarenta y nueve.",
    CL(), P("confusiones de la clase", 960, 160, "ambar", 44), alr(0, "a"), alr(1, "b"), sec="4. Cuatro confusiones a evitar")
v.b("[[a]]PCA no separa clústeres: los formó K-Means, y noventa y uno coma ocho es de la varianza. [[b]]Y predecir cuántos meses le quedan a un cliente antes de cancelar no es sin más una regresión lineal: es análisis de supervivencia, porque los que aún no cancelan están censurados.",
    alr(2, "a"), alr(3, "b"))

# ---- Dominio
CRI = [("Clasifico un problema: clasificación, regresión, clustering, reducción de dimensionalidad o refuerzo", "violeta"), ("Calculo a mano una probabilidad con la sigmoide, y una iteración de K-Means", "azul"),
       ("Leo una curva de entrenamiento y prueba, y explico qué hacen Ridge y Lasso", "ambar"), ("Calculo accuracy, precisión, recall y F1, y comparo con la línea base", "rojo"),
       ("Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos", "verde")]


def cri(i, en, t=0.0):
    txt, c = CRI[i]
    y = 300 + 135 * i
    return [Bc(960, y, 1640, 108, "blanco", c, 4, 24, en=en, ent="subir", t=t), T(txt, 260, y, tam=33, c="tinta", w="sb", anc="lm", mw=1340, en=en, t=t + 0.1), Bc(1740, y, 50, 50, "blanco", c, 4, 12, en=en, ent="fade", t=t + 0.2)]


v.b("Para pasar a la siguiente clase, deberías poder hacer cinco cosas. [[a]]Clasificar un problema en su familia. [[b]]Calcular a mano una probabilidad con la sigmoide, y una iteración de K-Means. [[c]]Leer una curva de entrenamiento y prueba, y explicar Ridge y Lasso.",
    CL(), P("criterio de dominio", 960, 150, "violeta", 48, ent="subir"), cri(0, "a"), cri(1, "b"), cri(2, "c"), sec="5. Tu criterio de dominio")
v.b("[[a]]Calcular accuracy, precisión, recall y F1, y compararlos con la línea base. [[b]]Y resolver los bloques uno a siete del cuaderno, con al menos ochenta por ciento de aciertos.",
    cri(3, "a"), cri(4, "b"))
STOPS = [("apuntes", "teal"), ("cuaderno\nbloques 1–4", "violeta"), ("notebook\npartes A–C", "azul"), ("cuaderno\nbloques 5–7", "violeta"), ("notebook\nparte D", "azul"), ("ejercicio\nde cierre", "rojo")]
v.b("¿En qué orden estudiar? [[a]]Primero, los apuntes. [[b]]Luego, los bloques uno a cuatro del cuaderno, y las partes A a C del notebook. [[c]]Después, los bloques cinco a siete, y la parte D. [[d]]Y al final, el ejercicio de cierre.",
    CL(),
    [[Bc(200 + 300 * i, 480, 250, 170, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, 200 + 300 * i, 490, tam=30, c=c, w="b", mw=220, en=k, t=0.1), T(str(i + 1), 200 + 300 * i, 390, tam=44, c=c, en=k, t=0.1)]
     for i, ((n, c), k) in enumerate(zip(STOPS, "abbccd"))],
    [L(335 + 300 * i, 480, 365 + 300 * i, 480, "grisclaro", 5, True, en=k, t=0.3, dur=0.4) for i, k in enumerate("bbccd")], sec="6. Cómo estudiar")
v.b("Y si te trabas más de diez minutos, [[a]]mira la solución, [[b]]entiéndela, [[c]]y vuelve a hacerlo sin mirar.",
    CL(), P("más de 10 minutos trabado", 960, 280, "ambar", 56, ent="subir"), P("mira la solución", 500, 520, "azul", 52, en="a"), P("entiéndela", 1000, 520, "teal", 52, en="b"), P("hazlo sin mirar", 1500, 520, "violeta", 52, en="c"),
    L(740, 520, 800, 520, "gris", 5, True, en="b"), L(1200, 520, 1260, 520, "gris", 5, True, en="c"))
v.b("Eso es todo. Ya sabes elegir el tipo de problema, [[a]]construir un modelo, y medirlo con honestidad. [[b]]En la próxima clase, la treinta y dos, veremos reducción de dimensionalidad: PCA y t-SNE. Mucho éxito con tu estudio.",
    CL(), T("elegir bien el problema", 960, 360, tam=80, c="violeta", ent="subir", t=0.2), P("modelar y medir con honestidad", 960, 580, "teal", 52, en="a"),
    P("siguiente: Clase 32 · reducción de dimensionalidad", 960, 800, "azul", 48, en="b"), sec="7. Cierre")

n = v.guardar(OUT)
print(n, "beats")
