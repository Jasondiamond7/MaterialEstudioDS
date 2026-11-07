import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, alerta, k1

OUT = out("guion_c35_07_maxfeatures.json")
v = V("max_features en la práctica")
MF = FG["mf"]
AM = MF["ames"]
FR = ["0.010", "0.020", "0.050", "0.100", "0.200", "0.333", "0.500", "0.750", "1.000"]
ETQ = ["1 %", "2 %", "5 %", "10 %", "20 %", "1/3", "50 %", "75 %", "100 %"]
RM = [AM[f][0] / 1000 for f in FR]
HE = MF["heart"]
NA = D["nb_ames"]

v.portada("max_features en la práctica", "Ni muy pocas ni todas: el punto intermedio se valida",
          "Séptima parada: max_features con datos reales. Vas a ver dónde está el punto óptimo, cuándo no importa, y un error en el notebook de la clase.")

# ---- 1. Hook
v.b("Ya sabes que max_features controla cuántas variables compiten en cada nodo. [[a]]Pocas: árboles distintos, pero malos. Todas: árboles buenos, pero parecidos. [[b]]¿Dónde está el punto justo? Hoy lo medimos.",
    CL(), T("max_features: ¿cuántas variables compiten en cada nodo?", 960, 200, tam=46, c="tinta", ent="subir"),
    tarjetas([(560, "violeta", "pocas", "árboles distintos, pero malos", "a"), (1360, "azul", "todas", "árboles buenos, pero parecidos", "a")], y=450, w=640, h=260),
    P("¿dónde está el punto justo?", 960, 730, "ambar", 54, en="b"), sec="1. El punto justo")

# ---- 2. Analogía
v.b("Es como la sal en la comida. [[a]]Sin sal, queda insípida. Con demasiada, no se puede comer. [[b]]Hay un punto intermedio, [[c]]y depende del plato: no hay una cantidad universal. Con max_features pasa lo mismo: depende de los datos.",
    CL(), T("la sal en la comida", 960, 180, tam=56, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "sin sal", "insípida", "a"), (960, "verde", "justa", "el punto intermedio", "b"), (1520, "rojo", "demasiada", "incomible", "a")],
             y=480, w=440, h=280, tam_t=50),
    P("depende del plato → max_features depende de los datos", 960, 780, "violeta", 44, en="c"), ilustrativo(1000, texto="analogía"),
    sec="2. Analogía: la sal")

# ---- 3. Ames: el barrido
GX = dict(x0=300, y0=260, x1=1380, y1=820)
v.b("Ames: después del one-hot, cada casa tiene 341 columnas. [[a]]Entrenamos bosques de trescientos árboles, variando max_features desde el 1 % de las columnas hasta el 100 %, y medimos el error en un conjunto de validación. [[b]]Con el 1 % de las columnas, el error es 27 700 dólares.",
    CL(), T("Ames · RMSE de validación según max_features (300 árboles)", 960, 160, tam=42, c="tinta", ent="subir"),
    G(GX["x0"], GX["y0"], GX["x1"], GX["y1"], -0.4, 8.4, 21.5, 28.5,
      [dict(xs=list(range(9)), ys=RM, color="teal", grosor=7, marcadores=11, revela=0.0)], xticks=[], yticks=[22, 24, 26, 28], yfmt="{:.0f} mil", id="gm", en="a"),
    *[T(e, i, 21.05, tam=28, c="gris", w="sb", ref="gm", en="a") for i, e in enumerate(ETQ)],
    T("fracción de las 341 columnas", 840, 895, tam=32, c="tinta", w="sb", en="a"),
    U("gm", **{"series.0.revela": 0.01}, en="b", dur=0.1), T(f"1 %: {k1(AM['0.010'][0])}", 1660, 340, tam=40, c="rojo", w="b", en="b", t=0.3),
    real(1050), sec="3. Ames: el barrido")
v.b("Al subir, el error baja rápido: [[a]]con el 5 %, 23 600. [[b]]Y el mínimo está en el 10 % de las columnas: 22 500 dólares. [[c]]Después vuelve a subir: con un tercio, 23 100; y con todas las columnas, es decir con bagging, 24 300.",
    U("gm", **{"series.0.revela": 2 / 8}, en="a", dur=0.8), T(f"5 %: {k1(AM['0.050'][0])}", 1660, 430, tam=40, c="tinta", w="b", en="a", t=0.5),
    U("gm", **{"series.0.revela": 3 / 8}, en="b", dur=0.6), T(f"10 %: {k1(AM['0.100'][0])}", 1660, 520, tam=44, c="verde", w="b", en="b", t=0.4),
    R(3, AM["0.100"][0] / 1000, 60, 60, "verde", 30, 5, ref="gm", en="b", t=0.6),
    U("gm", **{"series.0.revela": 1.0}, en="c", dur=1.6), T(f"1/3: {k1(AM['0.333'][0])}", 1660, 610, tam=40, c="tinta", w="b", en="c", t=0.6),
    T(f"100 % (bagging): {k1(AM['1.000'][0])}", 1660, 700, tam=34, c="azul", w="b", en="c", t=1.4))
v.b("La forma es la que predice la teoría: [[a]]a la izquierda, árboles tan malos que el promedio no los salva. [[b]]A la derecha, árboles tan parecidos que el promedio no los descorrelaciona. [[c]]El óptimo de estos datos está en medio, cerca del 10 %.",
    P("izquierda: árboles demasiado malos", 560, 985, "rojo", 32, en="a"), P("derecha: demasiado parecidos", 1140, 985, "azul", 32, en="b"),
    P("óptimo ≈ 10 %", 1660, 820, "verde", 40, en="c"))

# ---- 4. Ejemplo: cuántas columnas son
v.b("¿Cuántas columnas son esos porcentajes? Hagamos la cuenta como la hace scikit-learn, que corta hacia abajo. [[a]]El 10 % de 341 es 34.1: treinta y cuatro candidatas por nodo. [[b]]La raíz cuadrada de 341 es 18.47: dieciocho candidatas, cerca del 5 %. [[c]]Y un tercio de 341 es 113.67: ciento trece.",
    CL(), T("¿cuántas columnas son? (scikit-learn redondea hacia abajo)", 960, 170, tam=44, c="tinta", ent="subir"),
    T("10 %:   0.10 × 341 = 34.1  →", 1080, 330, tam=50, c="tinta", w="sb", anc="rm", en="a"), T("34", 1120, 330, tam=64, c="verde", anc="lm", en="a", t=0.5),
    T("'sqrt':   √341 = 18.47  →", 1080, 460, tam=50, c="tinta", w="sb", anc="rm", en="b"), T("18", 1120, 460, tam=64, c="violeta", anc="lm", en="b", t=0.5),
    T("1/3:   341 / 3 = 113.67  →", 1080, 590, tam=50, c="tinta", w="sb", anc="rm", en="c"), T("113", 1120, 590, tam=64, c="azul", anc="lm", en="c", t=0.5),
    sec="4. ¿Cuántas columnas son?")

# ---- 5. Heart: curva plana
HM = [1, 2, 3, 4, 6, 9, 13, 18]
v.b("Ahora Heart, con sus 18 columnas. Medimos el AUC con validación cruzada repetida. [[a]]Con una sola candidata por nodo, 0.904. [[b]]Con tres, 0.902. [[c]]Con las 18, es decir bagging, 0.882.",
    CL(), T("Heart · AUC (CV 5×3) según max_features", 960, 160, tam=44, c="tinta", ent="subir"),
    G(300, 260, 1380, 760, -0.4, 7.4, 0.86, 0.92, [dict(xs=list(range(8)), ys=[HE[str(m)][1] for m in HM], color="teal", grosor=7, marcadores=11, revela=0.0)],
      xticks=[], yticks=[0.86, 0.88, 0.90, 0.92], yfmt="{:.2f}", id="gh"),
    *[T(str(m), i, 0.855, tam=30, c="gris", w="sb", ref="gh") for i, m in enumerate(HM)], T("candidatas por nodo (de 18)", 840, 860, tam=32, c="tinta", w="sb"),
    U("gh", **{"series.0.revela": 0.01}, en="a", dur=0.1), T(f"1: {HE['1'][1]:.3f}", 1660, 340, tam=40, c="tinta", w="b", en="a", t=0.3),
    U("gh", **{"series.0.revela": 2 / 7}, en="b", dur=0.6), T(f"3: {HE['3'][1]:.3f}", 1660, 430, tam=40, c="tinta", w="b", en="b", t=0.4),
    U("gh", **{"series.0.revela": 1.0}, en="c", dur=1.4), T(f"18: {HE['18'][1]:.3f}", 1660, 520, tam=40, c="azul", w="b", en="c", t=1.0),
    real(1050), sec="5. Heart: una curva casi plana")
v.b("Parece que menos es mejor, pero ojo con la escala: [[a]]toda la curva se mueve en dos centésimas, [[b]]y el desvío entre particiones de la validación cruzada es de unas tres o cuatro centésimas. [[c]]En Heart, la diferencia es indistinguible del ruido. En Ames, en cambio, era clarísima.",
    P("toda la curva: ≈ 0.02", 1660, 640, "ambar", 38, en="a"), P("desvío entre particiones: ≈ 0.03 a 0.04", 840, 940, "rojo", 38, en="b"),
    P("Heart: ruido\nAmes: diferencia clara", 1660, 790, "violeta", 32, en="c"))

# ---- 6. Error del material
v.b("Y ahora un error en el notebook de la clase. [[a]]Para la regresión de Ames, el modelo llamado Random Forest base se crea con max_features igual a uno punto cero. [[b]]Uno punto cero significa el 100 % de las columnas: todas compiten en cada nodo. [[c]]Como vimos en el video anterior, eso no es un Random Forest: es un bagging.",
    CL(), alerta(160, texto="error en el notebook de la clase"),
    codigo(["rf_reg = Pipeline([", "    (\"prep\", ames_preprocessor),", "    (\"model\", RandomForestRegressor(", "        n_estimators=120,",
            "        max_features=1.0,      # ← todas las columnas", "        ...))])"], y=440, w=1300, tam=36, c="ambar", en="a", dt=0.2),
    P("1.0 = 100 % de las columnas = bagging", 960, 760, "rojo", 50, en="c"), sec="6. Un error en el notebook")
v.b("Lo comprobamos replicando las celdas del notebook. [[a]]El BaggingRegressor da un error de prueba de 29 300 dólares. [[b]]El supuesto Random Forest, 29 100: prácticamente lo mismo. [[c]]Con un tercio de las columnas, el mismo bosque baja a 26 700 dólares.",
    CL(), T("notebook del profesor · Ames · 120 árboles · RMSE de prueba", 960, 170, tam=42, c="tinta", ent="subir"),
    *[[T(lab, 620, 340 + 140 * i, tam=40, c="tinta", w="sb", anc="rm", en=m), Bx(660, 340 + 140 * i - 40, 660 + (val - 20000) / 12, 340 + 140 * i + 40, c, None, 0, 12, grow="x", en=m, t=0.1),
       N(val / 1000, 680 + (val - 20000) / 12, 340 + 140 * i, "{:.1f}", 50, c, suf=" mil", anc="lm", en=m, t=0.4)]
      for i, (lab, val, c, m) in enumerate([("BaggingRegressor", NA["bag"]["rmse"], "azul", "a"), ("RF max_features=1.0", NA["rf10"]["rmse"], "gris", "b"),
                                            ("RF max_features=1/3", NA["rf13"]["rmse"], "verde", "c")])],
    T("eje desde 20 mil", 660, 790, tam=26, c="gris", w="sb", anc="lm"), real(1050, texto="celdas del notebook replicadas"))
v.b("Por qué pasa: [[a]]en scikit-learn, el valor por defecto de max_features para el Random Forest de regresión es justamente uno punto cero. [[b]]Si no lo cambias, tu Random Forest de regresión es un bagging. Para regresión, prueba valores entre 0.1 y 0.5, y valida.",
    P("RandomForestRegressor: max_features por defecto = 1.0", 960, 860, "ambar", 40, en="a"),
    P("regresión: probar 0.1–0.5 y validar", 960, 960, "verde", 40, en="b"))

# ---- 7. Error típico
v.b("El error típico: [[a]]recitar un valor de max_features en vez de validarlo. [[b]]La raíz cuadrada es un buen punto de partida para clasificar, no una ley. [[c]]En Ames, el óptimo fue el 10 %; en Heart, casi cualquier valor da lo mismo. Se valida con validación cruzada u OOB.",
    CL(), T("«max_features siempre es la raíz de p»", 960, 280, tam=56, c="tinta", en="a", id="e1"),
    T("'sqrt' = punto de partida, no una ley", 960, 410, tam=46, c="gris", w="sb", en="b"), U("e1", tachado=1.0, color="gris", en="b", t=0.4, dur=0.6),
    P("Ames: óptimo 10 % · Heart: casi da igual → se VALIDA", 960, 580, "verde", 46, en="c"), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]En Heart, la curva es plana. [[b]]¿Qué valor de max_features eliges, y dónde pones tu esfuerzo?",
    CL(), P("Heart: curva plana", 960, 300, "teal", 56, en="a"), T("¿qué valor eliges?", 960, 480, tam=72, c="tinta", en="b"),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("Te quedas con el valor por defecto, la raíz cuadrada, que además es rápido, [[a]]porque optimizar un parámetro que no mueve la aguja es perder tiempo. [[b]]Tu esfuerzo va a lo que sí importa: la métrica correcta, una validación honesta, y mejores datos.",
    CL(), P("el valor por defecto ('sqrt'): rápido y suficiente", 960, 300, "verde", 50, ent="subir"),
    T("optimizar lo que no mueve la aguja = perder tiempo", 960, 450, tam=46, c="gris", w="sb", en="a"),
    P("esfuerzo: métrica correcta · validación honesta · mejores datos", 960, 620, "violeta", 44, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]max_features tiene un punto intermedio: en Ames, el 10 % de las columnas bajó el error de 24 300 a 22 500 dólares. [[b]]En otros datos, como Heart, casi no importa. [[c]]Y cuidado con max_features igual a uno punto cero: eso es un bagging, como en el notebook de la clase.",
    CL(), P(f"Ames: 10 % → {k1(AM['0.100'][0])} (bagging: {k1(AM['1.000'][0])})", 960, 300, "verde", 50, en="a"),
    P("Heart: casi no importa", 960, 470, "teal", 50, en="b"), P("max_features = 1.0 → bagging (error del notebook)", 960, 640, "rojo", 46, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]cuántos árboles usar, y cómo ajustar los demás hiperparámetros sin hacer trampa.",
    CL(), siguiente(8, "Número de árboles e hiperparámetros"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
