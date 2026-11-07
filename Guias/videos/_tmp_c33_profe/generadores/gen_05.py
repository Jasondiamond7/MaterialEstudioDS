import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, alerta

OUT = out("guion_c33_05_xor.json")
v = V("El caso XOR: útil solo en conjunto")
XR, XP = D["xor"], D["xor_pts"]
DUMMY = 50.0


def serie(clase):
    xs = [x for x, c in zip(XP["x"], XP["c"]) if c == clase] + [DUMMY]
    ys = [y for y, c in zip(XP["y"], XP["c"]) if c == clase] + [DUMMY]
    return xs, ys


v.portada("El caso XOR", "Dos variables que solas no dicen nada, y juntas lo dicen todo",
          "Cuarta parada: el caso extremo de la variable útil solo en conjunto. Se llama XOR, y rompe a los filtros univariados.")

# ---- 1. Hook: caja fuerte de dos perillas
v.b("Imagina una caja fuerte con dos perillas. [[a]]Si estudias solo la primera, ninguna posición la abre. [[b]]Si estudias solo la segunda, tampoco. [[c]]Pero la combinación correcta de las dos la abre siempre.",
    CL(), Cand(960, 470, 150, "rojo", id="cf"),
    T("perilla 1 sola: nada", 450, 470, tam=46, c="gris", w="sb", en="a"), T("perilla 2 sola: nada", 1470, 470, tam=46, c="gris", w="sb", en="b"),
    U("cf", abierto=1.0, color="verde", en="c", t=0.4, dur=0.7), P("las dos juntas: se abre", 960, 760, "verde", 56, en="c", t=0.6),
    ilustrativo(1000, texto="analogía"), sec="1. Una caja fuerte de dos perillas")
v.b("Eso es un XOR. [[a]]Dos variables que, cada una por separado, no tienen ninguna relación con el target, [[b]]pero que juntas lo predicen perfecto.",
    CL(), T("XOR · «o exclusivo»", 960, 300, tam=80, c="violeta", ent="subir"),
    P("cada una sola: ninguna relación", 960, 500, "gris", 52, en="a"), P("juntas: predicción perfecta", 960, 660, "verde", 56, en="b"))

# ---- 2. Ejemplo a mano: 4 puntos
PTS = [("A", 1, 1, 0), ("B", 1, -1, 1), ("C", -1, 1, 1), ("D", -1, -1, 0)]
GX = dict(x0=300, y0=260, x1=860, y1=820)


def px(x):
    return GX["x0"] + (x + 2) / 4 * (GX["x1"] - GX["x0"])


def py(y):
    return GX["y1"] - (y + 2) / 4 * (GX["y1"] - GX["y0"])


v.b("Hagamos el ejemplo del apunte con cuatro puntos. [[a]]A está en más uno, más uno. B, en más uno, menos uno. C, en menos uno, más uno. Y D, en menos uno, menos uno. [[b]]La clase es uno cuando los signos son distintos: B y C. A y D son clase cero.",
    CL(), T("ejemplo del apunte · 4 puntos", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    L(GX["x0"], py(0), GX["x1"], py(0), "tinta", 3), L(px(0), GX["y0"], px(0), GX["y1"], "tinta", 3),
    T("x1", GX["x1"] + 30, py(0), tam=36, c="tinta", w="m"), T("x2", px(0), GX["y0"] - 30, tam=36, c="tinta", w="m"),
    *[[Bc(px(x), py(y), 70, 70, "rojo" if c else "azul", None, 0, 35, en="a", t=0.4 * i, id=f"p{nm}"), T(nm, px(x), py(y), tam=36, c="blanco", en="a", t=0.4 * i + 0.05)]
      for i, (nm, x, y, c) in enumerate(PTS)],
    T("punto   x1   x2   clase", 1340, 300, tam=38, c="gris", w="m"),
    *[T(f"  {nm}     {x:+d}   {y:+d}     {c}".replace("-", "−"), 1340, 380 + 70 * i, tam=40, c="rojo" if c else "azul", w="m", en="b", t=0.3 * i) for i, (nm, x, y, c) in enumerate(PTS)],
    T("clase 1 = signos distintos", 1340, 700, tam=36, c="rojo", w="sb", en="b", t=1.4), sec="2. Ejemplo a mano: 4 puntos")
v.b("Ahora mira solo x1. [[a]]Con más uno hay un cero y un uno: A y B. [[b]]Con menos uno, también: C y D. [[c]]x1 sola no dice nada. Y con x2 pasa exactamente lo mismo.",
    R(px(1), py(0), 120, 460, "ambar", 40, en="a"), R(px(-1), py(0), 120, 460, "ambar", 40, en="b"),
    P("x1 = +1 → un 0 y un 1", 1340, 830, "ambar", 40, en="a"), P("x1 = −1 → un 0 y un 1", 1340, 920, "ambar", 40, en="b"),
    T("x1 sola: 50 % y 50 %", 580, 930, tam=40, c="gris", w="sb", en="c"))
v.b("Pero multiplica las dos. [[a]]A: más uno por más uno, más uno. B: más uno por menos uno, menos uno. C, menos uno. Y D, menos uno por menos uno, más uno. [[b]]El producto vale más uno en las clases cero y menos uno en las clases uno: separa perfecto.",
    CL(), T("x1 · x2", 960, 200, tam=70, c="violeta", w="m", ent="subir"),
    *[[Bc(330 + 420 * i, 450, 340, 240, "blanco", "rojo" if c else "azul", 5, 26, sombra=True, en="a", t=0.4 * i, ent="subir"),
       T(f"{nm}: ({x:+d}) · ({y:+d})".replace("-", "−"), 330 + 420 * i, 400, tam=36, c="tinta", w="m", en="a", t=0.4 * i + 0.1),
       T(f"{x * y:+d}".replace("-", "−"), 330 + 420 * i, 480, tam=70, c="rojo" if c else "azul", en="a", t=0.4 * i + 0.2)] for i, (nm, x, y, c) in enumerate(PTS)],
    P("+1 → clase 0", 660, 720, "azul", 52, en="b"), P("−1 → clase 1", 1260, 720, "rojo", 52, en="b", t=0.3),
    P("juntas separan perfecto", 960, 860, "verde", 52, en="b", t=0.8))

# ---- 3. Con 5.000 puntos
x0s, y0s = serie(0)
x1s, y1s = serie(1)
v.b("El notebook de la clase lo hace en grande. [[a]]Cinco mil puntos, con x1 y x2 normales e independientes. [[b]]El target es uno cuando exactamente una de las dos es positiva. [[c]]Mira el patrón: un tablero de ajedrez.",
    CL(), T("notebook de la clase · XOR con 5.000 puntos (se ven 220)", 960, 175, tam=36, c="gris", w="sb", ent="subir"),
    Bc(700, 560, 700, 700, "blanco", "linea", 3, 24, ent="fade"),
    G(380, 240, 1020, 880, -2.6, 2.6, -2.6, 2.6, [dict(xs=x0s, ys=y0s, color="azul", linea=None, marcadores=7, alfa=0.0),
                                                  dict(xs=x1s, ys=y1s, color="rojo", linea=None, marcadores=7, alfa=0.0)],
      xticks=[], yticks=[], rejilla=False, ejes=0.0, id="gx", t=0.1),
    L(380, 560, 1020, 560, "tinta", 3, en="a"), L(700, 240, 700, 880, "tinta", 3, en="a"),
    U("gx", **{"series.0.alfa": 1.0, "series.1.alfa": 1.0}, en="a", t=0.3, dur=1.2),
    T("x1", 1050, 560, tam=34, c="tinta", w="m"), T("x2", 700, 215, tam=34, c="tinta", w="m"),
    P("clase 0", 1350, 420, "azul", 44, en="b"), P("clase 1", 1350, 520, "rojo", 44, en="b", t=0.3),
    T("1 = exactamente una positiva", 1450, 640, tam=36, c="tinta", w="sb", en="b", t=0.6),
    P("un tablero de ajedrez", 1450, 780, "violeta", 46, en="c"), sec="3. El XOR en grande")
v.b("Si proyectas los puntos sobre un solo eje, [[a]]a cada lado del cero hay mitad azules y mitad rojos. [[b]]Ninguna recta horizontal ni vertical separa los colores.",
    R(540, 560, 300, 660, "ambar", 30, en="a"), R(860, 560, 300, 660, "ambar", 30, en="a", t=0.3),
    T(f"clase 1 en el XOR: {XR['tasa'] * 100:.1f} % · casi la mitad", 1450, 900, tam=32, c="gris", w="sb", en="a", t=0.8),
    P("ningún corte de una sola variable sirve", 1450, 980, "rojo", 38, en="b"))

# ---- 4. Lo que ve el filtro
v.b("¿Qué ve un filtro univariado, como ANOVA F? [[a]]Para x1, F igual a uno coma dos uno dos, y p-valor cero coma dos siete uno. [[b]]Para x2, F igual a cuatro coma tres uno dos, y p-valor cero coma cero tres ocho. [[c]]El notebook concluye: ninguna de las dos sirve.",
    CL(), T("lo que ve un filtro univariado (ANOVA F)", 960, 200, tam=44, c="tinta", ent="subir"),
    *[[Bc(x, 470, 560, 300, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(nm, x, 370, tam=56, c=c, w="m", en=k, t=0.1),
       T(f"F = {F_:.3f}", x, 470, tam=52, c="tinta", en=k, t=0.2), T(f"p = {p_:.3f}", x, 550, tam=52, c="tinta", en=k, t=0.3)]
      for x, nm, F_, p_, c, k in ((560, "x1", XR["F1"], XR["p1"], "azul", "a"), (1360, "x2", XR["F2"], XR["p2"], "violeta", "b"))],
    T("«Veredicto del filtro: ninguna de las dos sirve»", 960, 770, tam=46, c="gris", w="sb", en="c"), sec="4. Lo que ve el filtro")
v.b("Y aquí hay una alerta del material. [[a]]El p-valor de x2 es cero coma cero tres ocho: está bajo cero coma cero cinco. [[b]]Un filtro al cinco por ciento la habría dejado pasar. [[c]]Pero por construcción, x2 es independiente del target. Es un falso positivo por azar, justo el fenómeno que mide el video nueve.",
    CL(), alerta(), T(f"x2: p = {XR['p2']:.3f}", 960, 340, tam=80, c="violeta", w="m", en="a"), T("< 0.05", 1420, 340, tam=60, c="rojo", en="a", t=0.6),
    P("un filtro al 5 % la deja pasar", 960, 520, "rojo", 50, en="b"),
    P("x2 es independiente del target → falso positivo", 960, 680, "ambar", 48, en="c"), T("α · p falsos positivos → video 9", 960, 800, tam=38, c="gris", w="sb", en="c", t=0.8))

# ---- 5. Lo que ven los modelos
MOD = [("logística con x1 y x2", XR["log"], "rojo"), ("logística + x1·x2", XR["inter"], "verde"), ("Random Forest", XR["rf"], "verde")]
v.b("Ahora, los modelos, con validación cruzada. [[a]]Una regresión logística con x1 y x2 da AUC cero coma cinco uno. No ve nada. [[b]]La misma logística, agregando el producto x1 por x2: AUC uno. [[c]]Y un Random Forest, con las dos variables originales: también uno.",
    CL(), T("AUC con validación cruzada · notebook de la clase", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    *[[Bx(x - 140, 860 - (a - 0.4) * 900, x + 140, 860, c, None, 0, 12, grow="y", en=k), N(a, x, 860 - (a - 0.4) * 900 - 45, "{:.3f}", 64, c, en=k, t=0.3, dur=0.8),
       T(lab, x, 910, tam=32, c=c, w="sb", en=k)] for (lab, a, c), x, k in zip(MOD, (480, 960, 1440), "abc")],
    L(260, 860, 1660, 860, "tinta", 4), L(260, 860 - 0.1 * 900, 1660, 860 - 0.1 * 900, "gris", 3, guion=True),
    T("azar 0.5", 1760, 860 - 0.1 * 900, tam=28, c="gris", w="sb"), sec="5. Lo que ven los modelos")
v.b("La señal siempre estuvo ahí. [[a]]Lo que faltaba era una representación que la dejara visible. [[b]]Por eso construimos interacciones en el bloque dos: si sospechas que dos variables actúan juntas, la forma más segura de que un modelo lineal lo vea es escribir el producto.",
    CL(), P("la señal siempre estuvo ahí", 960, 260, "violeta", 60, ent="subir"),
    P("faltaba una representación que la dejara visible", 960, 430, "teal", 50, en="a"),
    codigo(["d['reclamo_x_antig'] = d.reclamos * d.antiguedad"], y=640, tam=40, en="b"))
v.b("Un detalle que suele sorprender: [[a]]un árbol de decisión poco profundo también falla en el XOR. Es voraz: evalúa un corte a la vez, y ningún corte individual mejora nada. [[b]]El Random Forest lo logra porque promedia muchos árboles con cortes iniciales distintos. Lo verás en la clase treinta y cuatro.",
    CL(), tarjetas([(560, "ambar", "árbol poco profundo", "voraz: un corte a la vez · falla", "a"), (1360, "verde", "Random Forest", "promedia muchos árboles · AUC 1.000", "b")], w=700, h=360),
    P("árboles → clase 34", 960, 820, "gris", 44, en="b", t=0.8))

# ---- 6. Error típico
v.b("El error típico: [[a]]concluir que no hay señal porque el filtro, o una logística sin interacción, no la ven. [[b]]Cero coma cinco uno no significa «no hay nada». Significa «este modelo, con esta representación, no ve nada».",
    CL(), T("AUC 0.51 → «no hay señal»", 960, 330, tam=74, c="tinta", en="a", id="nhs"), U("nhs", tachado=1.0, color="gris", en="b", t=0.2, dur=0.6),
    P("= este modelo, con esta representación, no la ve", 960, 560, "rojo", 50, en="b", t=0.6), sec="6. El error típico")

# ---- 7. Piensa un segundo
v.b("Piensa un segundo. [[a]]Sospechas que reclamar pesa distinto en un cliente nuevo que en uno antiguo. [[b]]¿Qué variable le agregas a tu regresión logística para que lo pueda ver?",
    CL(), T("reclamar pesa distinto según la antigüedad", 960, 330, tam=56, c="violeta", en="a"), T("¿qué le agregas a la logística?", 960, 540, tam=64, c="tinta", en="b"),
    sec="7. Piensa un segundo", pausa=0.6)
v.b("La interacción: reclamos por antigüedad. [[a]]Justo una de las tres interacciones que construyó la clase. [[b]]Y fíjate en lo que ya vimos: reclamo por antigüedad tiene correlación cero coma nueve cinco ocho con reclamos. Ayuda, pero también trae redundancia.",
    CL(), P("reclamo_x_antig = reclamos · antigüedad", 960, 300, "verde", 54, ent="subir"),
    T("una de las 3 interacciones de la clase", 960, 460, tam=44, c="tinta", w="sb", en="a"),
    P(f"r(reclamos, reclamo_x_antig) = {D['pares90'][6][2]:.3f}", 960, 640, "ambar", 48, en="b"), T("ayuda… y trae redundancia", 960, 760, tam=40, c="gris", w="sb", en="b", t=0.6))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]en un XOR, cada variable sola no dice nada, y juntas separan perfecto. [[b]]Los filtros univariados no lo ven, y una logística tampoco, sin el producto. [[c]]Y un p-valor bajo tampoco prueba nada: x2 pasó el filtro por azar.",
    CL(), P("solas: nada · juntas: perfecto", 960, 300, "violeta", 54, en="a"), P(f"logística {XR['log']:.2f} → con x1·x2 {XR['inter']:.2f}", 960, 480, "verde", 52, en="b"),
    P(f"p = {XR['p2']:.3f} con una variable sin señal: azar", 960, 660, "ambar", 50, en="c"), sec="8. Para llevar")
v.b("Siguiente parada: [[a]]las tres familias de selección, y los métodos de filtro.",
    CL(), siguiente(6, "Taxonomía y métodos de filtro"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
