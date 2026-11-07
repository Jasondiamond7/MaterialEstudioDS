import random
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_09_fuga.json"
v = V("Fuga de datos y Pipeline")

v.portada("Fuga de datos y Pipeline", "El error silencioso que más se comete al empezar",
          "Fuga de datos y Pipeline. Es el error más común cuando empiezas con Machine Learning, y también el más silencioso.")

# ---------------- 1. Qué es la fuga
v.b("Imagina que estudias con el examen final en la mano. [[a]]Sacas un siete. [[b]]¿Eso dice que sabes? No: dice que viste las respuestas.",
    CL(), Tj(560, 520, 560, 560, "Examen final", ["Pregunta 1", "Pregunta 2", "Pregunta 3"], "rojo", dur=0.7, t=0.2),
    N(7.0, 1260, 440, "{:.1f}", 200, "verde", en="a", dur=1.0), T("la nota", 1260, 620, tam=44, c="gris", w="sb", en="a", t=0.5),
    P("viste las respuestas", 1260, 790, "rojo", 54, en="b"), sec="1. Qué es la fuga de datos")
v.b("A eso se le llama fuga de datos, o data leakage. [[a]]Es usar, directa o indirectamente, información de la prueba o de la validación durante el desarrollo.",
    CL(),
    Bc(480, 480, 480, 240, "rojoclaro", "rojo", 5, 30, ent="subir", t=0.2), T("prueba", 480, 480, tam=60, c="rojo", t=0.3),
    L(740, 480, 1000, 480, "rojo", 10, True, en="a"), T("fuga", 870, 410, tam=50, c="rojo", en="a", t=0.3),
    Bc(1380, 480, 600, 240, "violetaclaro", "violeta", 5, 30, en="a", t=0.3), T("desarrollo", 1380, 480, tam=60, c="violeta", en="a", t=0.4))
v.b("Ocurre cuando [[a]]imputas, [[b]]escalas, [[c]]seleccionas variables, [[d]]o eliges hiperparámetros, usando datos que no eran solo de entrenamiento.",
    [P(n, 260 + 470 * i, 760, c, 46, en=k) for i, (n, c, k) in enumerate([("imputar", "teal", "a"), ("escalar", "azul", "b"), ("seleccionar variables", "ambar", "c"), ("elegir hiperparámetros", "violeta", "d")])])
v.b("El resultado se ve mejor de lo que será de verdad. [[a]]Una nota inflada, que no se repite cuando el modelo llega al mundo real.",
    CL(), Bx(480, 260, 800, 860, "rojo", None, 0, 14, grow="y", dur=1.0, t=0.2), T("lo que ves", 640, 920, tam=44, c="rojo", w="sb", t=0.4),
    Bx(1120, 560, 1440, 860, "verde", None, 0, 14, grow="y", en="a", dur=1.0), T("lo que pasará", 1280, 920, tam=44, c="verde", w="sb", en="a", t=0.4),
    P("nota inflada", 640, 200, "rojo", 48, en="a", t=0.8))

# ---------------- 2. Tres verbos
v.b("Para evitarla, necesitas entender tres verbos de scikit-learn. [[a]]Fit, [[b]]transform, [[c]]y predict.",
    CL(), P("fit", 480, 480, "violeta", 100, en="a"), P("transform", 960, 480, "azul", 100, en="b"), P("predict", 1440, 480, "teal", 100, en="c"), sec="2. Los tres verbos de scikit-learn")
v.b("Fit significa aprender de los datos. [[a]]Por ejemplo, calcular una media, [[b]]o los coeficientes de un modelo.",
    Bc(480, 700, 400, 220, "violetaclaro", "violeta", 4, 26, ent="subir"), T("aprende de los datos", 480, 640, tam=36, c="violeta", w="sb", mw=340),
    T("media = 12", 480, 715, tam=44, c="tinta", w="sb", en="a"), T("coeficientes", 480, 775, tam=34, c="gris", w="sb", en="b"))
v.b("Transform significa aplicar lo aprendido. [[a]]Por ejemplo, restar esa media a otros datos.",
    Bc(960, 700, 400, 220, "azulclaro", "azul", 4, 26, ent="subir"), T("aplica lo aprendido", 960, 640, tam=36, c="azul", w="sb", mw=340),
    T("x − 12", 960, 760, tam=56, c="tinta", w="sb", en="a"))
v.b("Y predict significa predecir con un modelo que ya está ajustado.",
    Bc(1440, 700, 400, 220, "tealclaro", "teal", 4, 26, ent="subir"), T("predice con un modelo ya ajustado", 1440, 700, tam=36, c="teal", w="sb", mw=340))
v.b("Piénsalo así: [[a]]fit es estudiar. [[b]]Transform y predict son dar el examen.",
    P("fit  =  estudiar", 480, 880, "violeta", 48, en="a"), P("transform · predict  =  dar el examen", 1200, 880, "azul", 48, en="b"))
FILAS = [("fit", "solo con el entrenamiento", "violeta"), ("transform", "entrenamiento, validación y prueba", "azul"),
         ("fit_transform", "solo con el entrenamiento", "violeta"), ("predict", "validación, prueba y datos nuevos", "teal")]
v.b("¿Con qué datos se usa cada uno? [[a]]Fit, solo con el entrenamiento. [[b]]Transform, con entrenamiento, validación y prueba. [[c]]Fit transform, que hace las dos cosas seguidas, solo con el entrenamiento. [[d]]Y predict, con validación, prueba y datos nuevos.",
    CL(),
    [[Bc(960, 290 + 150 * i, 1560, 120, "blanco", c, 4, 26, en=k, ent="subir"), T(n, 400, 290 + 150 * i, tam=48, c=c, w="m", anc="lm", en=k, t=0.1), T(d, 860, 290 + 150 * i, tam=42, c="tinta", w="sb", anc="lm", en=k, t=0.2)]
     for i, ((n, d, c), k) in enumerate(zip(FILAS, "abcd"))])
v.b("La regla de oro es una sola: [[a]]fit, solo con el entrenamiento.",
    CL(), P("fit  →  solo con el entrenamiento", 960, 480, "violeta", 80, en="a"), Cand(960, 780, 120, "rojo", 0, True, en="a", t=0.4))

# ---------------- 3. Atípico escondido
v.b("Veamos cómo la fuga esconde un problema. El entrenamiento tiene tres valores: [[a]]diez, doce y catorce. [[b]]Su media es doce, [[c]]y su desviación estándar, uno coma seiscientos treinta y tres.",
    CL(),
    [[Bc(520 + 300 * i, 400, 200, 140, "violetaclaro", "violeta", 4, 24, en="a", t=0.25 * i), T(str(x), 520 + 300 * i, 400, tam=70, c="violeta", en="a", t=0.25 * i + 0.1)] for i, x in enumerate([10, 12, 14])],
    T("entrenamiento", 820, 280, tam=44, c="gris", w="sb", t=0.2), T("media = 12", 820, 580, tam=64, c="teal", en="b"), T("desviación = 1.633", 820, 700, tam=64, c="azul", en="c"),
    sec="3. Un atípico escondido")
v.b("En la prueba aparece un treinta. [[a]]Un valor muy alejado de todo lo que vimos.",
    Bc(1520, 400, 240, 160, "rojoclaro", "rojo", 5, 26, ent="pop", t=0.2), T("30", 1520, 400, tam=100, c="rojo", t=0.3), T("prueba", 1520, 290, tam=44, c="rojo", w="sb", t=0.3),
    P("¿atípico?", 1520, 560, "ambar", 48, en="a"))
v.b("Con los estadísticos del entrenamiento, que es lo correcto, [[a]]el puntaje z de treinta es treinta menos doce, entre uno coma seiscientos treinta y tres: [[b]]once.",
    CL(), P("fit solo con el entrenamiento", 960, 200, "teal", 48, ent="subir"),
    T("z  =  (30 − 12) / 1.633", 960, 450, tam=80, c="tinta", en="a"), T("= 11.0", 960, 620, tam=150, c="rojo", en="b"), P("atípico clarísimo", 960, 830, "rojo", 60, en="b", t=0.6))
v.b("Ahora, la fuga. Calculamos la media y la desviación con las cuatro filas, [[a]]incluido el treinta de la prueba. [[b]]La media sube a dieciséis coma cinco, [[c]]y la desviación a siete coma noventa y dos.",
    CL(), P("con fuga: las 4 filas", 960, 200, "rojo", 48, ent="subir"),
    [[Bc(420 + 270 * i, 400, 200, 140, "violetaclaro" if i < 3 else "rojoclaro", "violeta" if i < 3 else "rojo", 4, 24, en="a", t=0.15 * i), T(str(x), 420 + 270 * i, 400, tam=70, c="violeta" if i < 3 else "rojo", en="a", t=0.15 * i + 0.1)] for i, x in enumerate([10, 12, 14, 30])],
    T("media = 16.5", 960, 620, tam=70, c="teal", en="b"), T("desviación = 7.92", 960, 760, tam=70, c="azul", en="c"))
v.b("Con esos números, el z de treinta es treinta menos dieciséis coma cinco, entre siete coma noventa y dos. [[a]]Da uno coma setenta. [[b]]¡Parece normal!",
    CL(), T("z  =  (30 − 16.5) / 7.92", 960, 380, tam=80, c="tinta", ent="subir"), T("= 1.70", 960, 560, tam=150, c="ambar", en="a"), P("parece normal", 960, 780, "ambar", 66, en="b"))
v.b("Compáralos: [[a]]once con el método correcto, [[b]]uno coma setenta con la fuga. [[c]]La fuga escondió el atípico.",
    CL(), L(300, 840, 1620, 840, "tinta", 4, dur=0.4),
    Bx(560, 840 - 11 * 48, 760, 840, "rojo", None, 0, 12, grow="y", en="a", dur=0.9), T("11.0", 660, 840 - 11 * 48 - 40, tam=54, c="rojo", en="a", t=0.7), T("correcto", 660, 890, tam=40, c="gris", w="sb", en="a", t=0.4),
    Bx(1160, 840 - 1.7 * 48, 1360, 840, "ambar", None, 0, 12, grow="y", en="b", dur=0.9), T("1.70", 1260, 840 - 1.7 * 48 - 40, tam=54, c="ambar", en="b", t=0.7), T("con fuga", 1260, 890, tam=40, c="gris", w="sb", en="b", t=0.4),
    P("la fuga escondió el atípico", 960, 200, "rojo", 56, en="c"))
v.b("Piensa un segundo: [[a]]¿por qué, con la fuga, el treinta pareció normal?",
    CL(), T("con la fuga:  z = 1.70", 960, 400, tam=80, c="ambar", ent="subir"), T("¿por qué parece normal?", 960, 600, tam=72, c="violeta", en="a"), pausa=0.5)
v.b("Porque el propio treinta ayudó a calcular la media y la desviación, [[a]]y las infló. Es como corregir un examen con el criterio del alumno que lo hizo.",
    P("el 30 infló la media y la desviación", 960, 800, "ambar", 52, t=0.1), T("el modelo ya había visto la prueba", 960, 920, tam=46, c="rojo", w="sb", en="a"))

# ---------------- 4. Ruido
random.seed(7)
RB = [random.random() for _ in range(48)]
v.b("Un experimento lo muestra con claridad. Tomamos [[a]]mil columnas de puro ruido, [[b]]sin ninguna relación con lo que queremos predecir.",
    CL(), T("1.000 columnas de puro ruido", 960, 190, tam=54, c="gris", ent="subir"),
    [Bx(200 + 33 * i, 760 - 40 - RB[i] * 380, 200 + 33 * i + 24, 760, "#94A3B8", None, 0, 4, grow="y", en="a", t=0.02 * i, dur=0.5) for i in range(48)],
    P("ninguna relación con lo que se quiere predecir", 960, 880, "rojo", 48, en="b"), sec="4. El experimento del ruido")
v.b("Entonces ningún modelo puede acertar más de cincuenta por ciento. [[a]]Es como adivinar a cara o cruz.",
    CL(), L(300, 640, 1620, 640, "gris", 5, guion=True, dur=0.8), T("50 % = azar", 960, 580, tam=64, c="gris", w="sb", t=0.4), P("adivinar a cara o cruz", 960, 800, "violeta", 54, en="a"))
YB = 860
ys_ = lambda val: YB - val * 640
v.b("Primero, hacerlo mal: [[a]]seleccionas las diez mejores columnas usando todas las filas, y recién después validas. [[b]]La exactitud sale cerca de cero coma setenta y nueve. [[c]]Como si el modelo supiera algo.",
    CL(), L(300, YB, 1620, YB, "tinta", 4, dur=0.4), L(300, ys_(0.5), 1620, ys_(0.5), "gris", 4, guion=True, id="az", dur=0.8), T("azar 0.50", 1700, ys_(0.5), tam=36, c="gris", w="sb", id="azt"),
    P("selecciona con todas las filas, y valida", 960, 170, "rojo", 44, en="a"),
    Bx(560, ys_(0.786), 800, YB, "rojo", None, 0, 12, grow="y", id="bf", en="b", dur=1.0), T("0.79", 680, ys_(0.786) - 40, tam=60, c="rojo", id="bft", en="b", t=0.8),
    T("fuga", 680, 910, tam=40, c="rojo", w="sb", id="bfl", en="b", t=0.3), P("¡parece que sabe algo!", 880, 250, "rojo", 50, anc="lm", en="c"))
v.b("Ahora, hacerlo bien: [[a]]la selección va dentro de un Pipeline. [[b]]La exactitud sale cerca de cero coma cincuenta y uno. [[c]]Es azar, y es lo honesto.",
    P("dentro de un Pipeline", 1240, 400, "teal", 44, en="a"),
    Bx(1120, ys_(0.514), 1360, YB, "teal", None, 0, 12, grow="y", id="bp", en="b", dur=1.0), T("0.51", 1240, ys_(0.514) - 40, tam=60, c="teal", id="bpt", en="b", t=0.8), T("sin fuga", 1240, 910, tam=40, c="teal", w="sb", en="b", t=0.3),
    P("azar: lo honesto", 1610, 640, "verde", 50, en="c"))
v.b("La diferencia entre esos dos números [[a]]es toda la fuga. [[b]]Y repetimos el experimento cuarenta veces para confirmarlo.",
    L(900, ys_(0.786), 900, ys_(0.514), "ambar", 8, True, id="dif", en="a"), T("0.79 − 0.51", 1000, ys_(0.65), tam=44, c="ambar", w="sb", anc="lm", id="dift", en="a", t=0.3),
    P("= la fuga", 1000, ys_(0.65) + 70, "ambar", 48, anc="lm", en="a", t=0.6), P("40 repeticiones", 960, 970, "violeta", 46, en="b"))

# ---------------- 5. Pipeline
v.b("La solución práctica se llama Pipeline. [[a]]Encadena varios pasos, por ejemplo estandarizar y luego modelar, [[b]]y hace fit de cada paso solo con los datos que recibe.",
    CL(),
    Bc(480, 480, 480, 220, "azulclaro", "azul", 5, 30, ent="subir", en="a"), T("estandarizar", 480, 480, tam=54, c="azul", en="a", t=0.1),
    L(740, 480, 900, 480, "gris", 8, True, en="a", t=0.3),
    Bc(1160, 480, 480, 220, "violetaclaro", "violeta", 5, 30, ent="subir", en="a", t=0.3), T("modelo", 1160, 480, tam=54, c="violeta", en="a", t=0.4),
    R(820, 480, 1500, 330, "teal", 50, id="rpipe", en="b", ent="fade"), P("Pipeline", 820, 270, "teal", 56, en="b", t=0.2), sec="5. El Pipeline")
v.b("En código, es una línea: [[a]]make pipeline, el estandarizador, y el modelo. Y se ajusta con el entrenamiento.",
    CL(), T("make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X_train, y_train)", 960, 470, tam=36, c="tinta", w="m", fondo="#E8EDF4", pad=28, en="a", ent="subir", mw=1500),
    P("fit: solo con el entrenamiento", 960, 700, "teal", 52, en="a", t=0.6))
v.b("Dentro de K-Fold, el Pipeline [[a]]repite el proceso completo en cada pliegue: [[b]]aprende con los pliegues de entrenamiento, y aplica en el de validación.",
    CL(),
    [[Bx(380 + 260 * j, 330 + 80 * i, 380 + 260 * j + 240, 330 + 80 * i + 60, "azul" if i == j else "#C4B5FD", None, 0, 10, ent="fade", en="a", t=0.05 * (i * 5 + j))] for i in range(5) for j in range(5)],
    [T(f"Pliegue {i + 1}", 340, 360 + 80 * i, tam=32, c="tinta", w="sb", anc="rm", en="a", t=0.05 * i) for i in range(5)],
    P("cada pliegue re-aprende sus pasos", 960, 840, "teal", 52, en="b"))
v.b("Y se usa así: [[a]]cross val score, con el Pipeline, los datos de desarrollo, y el K-Fold.",
    CL(), T("cross_val_score(pipe, X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=42))", 960, 450, tam=34, c="tinta", w="m", fondo="#E8EDF4", pad=28, ent="subir", mw=1500, en="a"),
    P("sin fuga: la estandarización se aprende en cada pliegue", 960, 700, "teal", 44, en="a", t=0.8))
v.b("Cuando tus columnas son de distinto tipo, usas ColumnTransformer: [[a]]un tratamiento por tipo de columna. [[b]]A las numéricas se les imputa y se escalan. [[c]]A las categóricas, se les pone la moda y se les aplica one-hot.",
    CL(),
    Bc(960, 300, 520, 130, "blanco", "violeta", 4, 28, ent="subir", en="a"), T("ColumnTransformer", 960, 300, tam=44, c="violeta", en="a", t=0.1),
    L(820, 370, 560, 480, "gris", 6, True, en="b"), L(1100, 370, 1360, 480, "gris", 6, True, en="c"),
    Bc(520, 620, 760, 240, "tealclaro", "teal", 5, 28, en="b", ent="subir"), T("numéricas", 520, 530, tam=44, c="teal", en="b", t=0.1), T("imputar + escalar", 520, 650, tam=40, c="tinta", w="sb", en="b", t=0.3),
    Bc(1400, 620, 760, 240, "ambarclaro", "ambar", 5, 28, en="c", ent="subir"), T("categóricas", 1400, 530, tam=44, c="ambar", en="c", t=0.1), T("moda + one-hot", 1400, 650, tam=40, c="tinta", w="sb", en="c", t=0.3))
v.b("Al final, cuando ya decidiste todo, [[a]]ajustas con entrenamiento más validación, [[b]]y predices sobre la prueba, una sola vez.",
    CL(), T("pipe.fit(X_dev, y_dev)", 960, 380, tam=48, c="tinta", w="m", fondo="#E8EDF4", pad=26, en="a", ent="subir"),
    T("pipe.predict(X_test)", 960, 540, tam=48, c="tinta", w="m", fondo="#E8EDF4", pad=26, en="b", ent="subir"), P("la prueba: una sola vez", 960, 760, "rojo", 56, en="b", t=0.5),
    Cand(960, 900, 90, "rojo", 1, False, en="b", t=0.6))

# ---------------- 6. Checklist
CHK = [("Divido primero, antes de imputar, escalar, codificar o seleccionar", "violeta"), ("Uso un Pipeline: cada pliegue re-aprende sus pasos", "teal"),
       ("Estratifico si hay clases raras", "azul"), ("Elijo hiperparámetros con validación o K-Fold", "ambar"),
       ("Miro la prueba una sola vez, con el modelo ya decidido", "rojo"), ("Reporto varias métricas: el accuracy no basta", "verde")]


def chk(i, en, t=0.0):
    txt, c = CHK[i]
    y = 270 + 128 * i
    return [Bc(960, y, 1640, 100, "blanco", c, 4, 24, en=en, ent="subir", t=t), T(txt, 260, y, tam=38, c="tinta", w="sb", anc="lm", mw=1250, en=en, t=t + 0.1),
            K(1760, y, "check", 60, "verde", en=en, t=t + 0.5)]


v.b("Un checklist para no equivocarte. Uno: [[a]]divide primero, antes de imputar, escalar, codificar o seleccionar. Dos: [[b]]usa un Pipeline, así cada pliegue de K-Fold re-aprende sus propios pasos.",
    CL(), chk(0, "a"), chk(1, "b"), sec="6. Checklist")
v.b("Tres: [[a]]estratifica si hay clases raras. Cuatro: [[b]]elige los hiperparámetros con la validación o con K-Fold.",
    chk(2, "a"), chk(3, "b"))
v.b("Cinco: [[a]]mira la prueba una sola vez, con el modelo ya decidido. Y seis: [[b]]reporta varias métricas, porque el accuracy no basta con clases desbalanceadas.",
    chk(4, "a"), chk(5, "b"))
v.b("Piensa un segundo: imputas con la mediana calculada con todas las filas, [[a]]y recién después divides los datos. ¿Hay fuga?",
    CL(), T("mediana con todas las filas", 960, 360, tam=64, c="tinta", ent="subir"), T("después divides entrenamiento y prueba", 960, 500, tam=64, c="tinta", t=0.3),
    g("qf", T("¿hay fuga?", 960, 720, tam=90, c="violeta", en="a")), pausa=0.5)
v.b("Sí. La mediana ya incluye información de la validación y de la prueba. [[a]]Lo correcto es dividir primero, y calcular la mediana solo con el entrenamiento.",
    XG("qf", dur=0.25), P("sí, hay fuga", 960, 720, "rojo", 80), P("dividir primero  →  mediana solo con entrenamiento", 960, 880, "verde", 46, en="a"))
v.b("Para llevarte: [[a]]la fuga es usar información de la prueba o de la validación al desarrollar. [[b]]Fit, solo con el entrenamiento. [[c]]Y el Pipeline lo hace por ti.",
    CL(), P("fuga = usar información que no deberías", 960, 330, "rojo", 52, en="a"), P("fit: solo con el entrenamiento", 960, 520, "violeta", 56, en="b"),
    P("el Pipeline lo hace por ti", 960, 710, "teal", 56, en="c"), sec="7. Para llevar")

print(v.guardar(OUT), "beats")
