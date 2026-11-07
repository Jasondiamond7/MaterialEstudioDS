import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_07_estratificado.json"
v = V("Muestreo estratificado")

v.portada("Muestreo estratificado", "Cómo repartir bien una clase rara",
          "Muestreo estratificado. Qué hacer cuando lo que quieres detectar es raro.")

GR_ = "#94A3B8"
# ---------------- 1. El problema
v.b("Imagina un banco que revisa transacciones. [[a]]De cada cien, solo cinco son fraudes. [[b]]Y esos cinco son justo lo que el modelo tiene que aprender a detectar.",
    CL(),
    Pt([(95, [300, 250, 1300, 760], GR_), (5, [1450, 250, 1620, 760], "rojo")], 14, visibles=100, id="p100", en="a", dur=1.6),
    N(100, 960, 170, "{:.0f}", 60, "tinta", suf=" transacciones", en="a", dur=1.4),
    P("5 fraudes", 1535, 850, "rojo", 52, en="b"), sec="1. El problema de la clase rara")
v.b("Hagamos números con cuatrocientas filas. [[a]]Con cinco por ciento de fraude, [[b]]son veinte fraudes y trescientos ochenta normales.",
    CL(),
    Pt([(380, [160, 280, 1380, 760], GR_), (20, [1500, 280, 1760, 760], "rojo")], 8, id="p400", en="a"),
    N(400, 960, 170, "{:.0f}", 60, "tinta", suf=" filas", en="a", dur=1.2),
    N(20, 1630, 850, "{:.0f}", 64, "rojo", suf=" fraudes", en="b", dur=0.9), N(380, 770, 850, "{:.0f}", 64, "gris", suf=" normales", en="b", t=0.3, dur=1.1))
v.b("Ahora apartamos veinticinco por ciento para la prueba: cien filas. [[a]]Si el reparto fuera perfecto, [[b]]caerían cinco fraudes ahí: veinte por cero coma veinticinco.",
    CL(),
    Bc(960, 400, 560, 220, "rojoclaro", "rojo", 5, 30, ent="subir", t=0.1), T("prueba: 100 filas", 960, 360, tam=54, c="rojo", t=0.2), T("25 % de 400", 960, 450, tam=38, c="gris", w="sb", t=0.3),
    T("20 × 0.25  =  5", 960, 640, tam=100, c="tinta", en="b"),
    [Bc(790 + 85 * i, 780, 40, 40, "rojo", None, 0, 20, en="b", t=0.12 * i + 0.4) for i in range(5)], P("5 fraudes esperados", 960, 910, "rojo", 48, en="b", t=1.0))
BOXX = [480, 960, 1440]
v.b("Pero un corte al azar no es perfecto. [[a]]A veces caen dos fraudes, [[b]]a veces cinco, [[c]]y a veces ocho.",
    CL(),
    [[Bc(x, 480, 400, 330, "blanco", "rojo", 5, 28, sombra=True, en=k, ent="subir"), T("prueba", x, 355, tam=38, c="gris", w="sb", en=k, t=0.1)] for x, k in zip(BOXX, "abc")],
    [Bc(BOXX[0] - 60 + 120 * j, 460, 40, 40, "rojo", None, 0, 20, en="a", t=0.15 * j + 0.2) for j in range(2)],
    [Bc(BOXX[1] - 120 + 60 * j, 460, 40, 40, "rojo", None, 0, 20, en="b", t=0.1 * j + 0.2) for j in range(5)],
    [Bc(BOXX[2] - 90 + 60 * (j % 4), 440 + 60 * (j // 4), 40, 40, "rojo", None, 0, 20, en="c", t=0.08 * j + 0.2) for j in range(8)],
    [N(n_, x, 620, "{:.0f}", 100, "rojo", en=k, t=0.3, dur=0.6) for x, k, n_ in zip(BOXX, "abc", (2, 5, 8))],
    [T("fraudes", x, 710, tam=38, c="rojo", w="sb", en=k, t=0.5) for x, k in zip(BOXX, "abc")])
v.b("Con dos fraudes, el modelo casi no tiene casos con qué ser evaluado. [[a]]Con ocho, la prueba sale más difícil de lo normal. [[b]]Y todo eso depende solo de la suerte.",
    R(480, 535, 450, 440, "rojo", 40, id="rf1", ent="fade"), R(1440, 535, 450, 440, "ambar", 40, id="rf2", en="a", ent="fade"), P("depende de la suerte", 960, 900, "violeta", 56, en="b"))
v.b("Piensa un segundo: [[a]]¿cómo podrías asegurar que la prueba tenga siempre cinco fraudes?",
    CL(), T("20 fraudes  ·  380 normales", 960, 360, tam=60, c="tinta", ent="subir", t=0.2), T("prueba de 100 filas", 960, 520, tam=60, c="rojo", t=0.4),
    g("pq", T("¿cómo asegurar 5 fraudes siempre?", 960, 700, tam=64, c="violeta", en="a")), pausa=0.5)
v.b("Repartiendo cada clase por separado, en la misma proporción. [[a]]Eso es estratificar.",
    XG("pq", dur=0.3), P("cada clase por separado", 960, 750, "teal", 60, t=0.1), P("estratificar", 960, 900, "violeta", 80, en="a"))

# ---------------- 2. Estratificar
KID = [330, 780, 1230, 1680]
v.b("Es como repartir caramelos de dos colores entre cuatro niños. [[a]]Si los repartes a ciegas, un niño puede quedarse con casi todos los rojos. [[b]]Si repartes por color, cada niño recibe la misma mezcla.",
    CL(),
    [[Bc(x, 520, 380, 420, "blanco", "grisclaro", 4, 26, ent="subir", t=0.1 * i), T(f"niño {i + 1}", x, 340, tam=38, c="gris", w="sb", t=0.1 * i + 0.1)] for i, x in enumerate(KID)],
    g("c1", [T(str(n_), x, 520, tam=120, c="rojo", en="a", t=0.2 + 0.1 * i) for i, (x, n_) in enumerate(zip(KID, (14, 0, 3, 3)))],
      [T("caramelos rojos", x, 640, tam=30, c="rojo", w="sb", en="a", t=0.8) for x in KID], P("a ciegas: reparto desigual", 960, 860, "rojo", 52, en="a", t=0.8)),
    XG("c1", en="b", dur=0.3),
    [T("5", x, 520, tam=120, c="verde", en="b", t=0.5 + 0.1 * i) for i, x in enumerate(KID)],
    [T("caramelos rojos", x, 640, tam=30, c="verde", w="sb", en="b", t=0.9) for x in KID], P("por color: todos reciben 5", 960, 860, "verde", 52, en="b", t=0.9),
    sec="2. Cómo funciona estratificar")
v.b("Estratificar significa dividir cada clase por separado, y después juntar. [[a]]Los veinte fraudes se reparten en quince y cinco. [[b]]Los trescientos ochenta normales, en doscientos ochenta y cinco y noventa y cinco.",
    CL(),
    Pt([(380, [160, 300, 900, 560], GR_), (20, [960, 300, 1200, 560], "rojo")], 8, id="est", en=None, t=0.1),
    T("antes de dividir", 700, 230, tam=40, c="gris", w="sb", t=0.2),
    U("est", grupos=[dict(n=285, rect=[160, 620, 900, 900], color=GR_), dict(n=15, rect=[960, 620, 1200, 900], color="rojo"),
                      dict(n=95, rect=[1300, 620, 1560, 900], color=GR_), dict(n=5, rect=[1620, 620, 1760, 900], color="rojo")], escalonar=0.6, dur=2.0, en="a"),
    T("entrenamiento", 580, 950, tam=40, c="violeta", w="sb", en="a", t=1.0), T("prueba", 1530, 950, tam=40, c="rojo", w="sb", en="a", t=1.0),
    N(15, 1080, 590, "{:.0f}", 44, "rojo", en="a", t=0.5, dur=0.8), N(5, 1690, 590, "{:.0f}", 44, "rojo", en="a", t=0.5, dur=0.8),
    N(285, 530, 590, "{:.0f}", 44, "gris", en="b", t=0.3, dur=1.0), N(95, 1430, 590, "{:.0f}", 44, "gris", en="b", t=0.3, dur=1.0))
v.b("El resultado: [[a]]la prueba siempre tiene cinco fraudes, [[b]]el cinco por ciento de sus cien filas.",
    CL(), Bc(960, 450, 700, 300, "rojoclaro", "rojo", 5, 30, ent="subir", t=0.1), T("prueba", 960, 340, tam=44, c="gris", w="sb", t=0.2),
    N(5, 960, 450, "{:.0f}", 130, "rojo", suf=" fraudes", en="a", dur=0.7), T("de 100 filas  =  5 %", 960, 560, tam=48, c="rojo", w="sb", en="b"),
    P("siempre 5, no 2 ni 8", 960, 800, "verde", 60, en="b", t=0.5))
v.b("Ojo con esto: estratificar no agrega fraudes. [[a]]Solo los reparte bien.",
    CL(), P("no agrega fraudes", 960, 430, "ambar", 70, ent="subir"), P("solo los reparte bien", 960, 640, "verde", 70, en="a"))
v.b("En Python, se hace con un argumento: [[a]]stratify igual a y. [[b]]Le dices que respete las proporciones de la columna objetivo.",
    CL(), T("train_test_split(X, y, stratify=y)", 960, 400, tam=60, c="tinta", w="m", fondo="#E8EDF4", pad=30, en="a", ent="subir"),
    P("respeta las proporciones de y", 960, 640, "teal", 56, en="b"))

# ---------------- 3. Cuaderno
v.b("En el cuaderno de la clase hay dos mil casos. [[a]]Ciento nueve son fraudes: [[b]]cinco coma cuarenta y cinco por ciento.",
    CL(),
    Pt([(1891, [160, 280, 1520, 760], GR_), (109, [1620, 280, 1760, 760], "rojo")], 5, id="p2000", en="a"),
    T("2.000 casos", 960, 170, tam=60, c="tinta", en="a"), N(109, 1690, 850, "{:.0f}", 60, "rojo", suf=" fraudes", en="b", dur=0.9),
    P("5.45 %", 960, 850, "rojo", 60, en="b", t=0.6), sec="3. En el cuaderno de la clase")
v.b("Al estratificar, [[a]]el entrenamiento queda con cinco coma cuatrocientos sesenta y siete por ciento, [[b]]y la prueba con cinco coma cuatrocientos. Casi idénticos.",
    CL(),
    Bx(200, 700, 260 + 5.467 * 130, 820, "violeta", None, 0, 14, grow="x", en="a", dur=0.9), T("entrenamiento", 960, 640, tam=44, c="violeta", w="sb", en="a"),
    N(5.467, 1280, 760, "{:.3f}", 70, "violeta", suf=" %", en="a", dur=0.9, anc="lm"), T("82 de 1.500", 1280, 830, tam=36, c="gris", w="sb", anc="lm", en="a", t=0.7),
    Bx(200, 320, 260 + 5.4 * 130, 440, "rojo", None, 0, 14, grow="x", en="b", dur=0.9), T("prueba", 960, 260, tam=44, c="rojo", w="sb", en="b"),
    N(5.4, 1280, 380, "{:.3f}", 70, "rojo", suf=" %", en="b", dur=0.9, anc="lm"), T("27 de 500", 1280, 450, tam=36, c="gris", w="sb", anc="lm", en="b", t=0.7))
H2 = [0, 0, 6, 2, 5, 9, 21, 18, 36, 45, 48, 68, 66, 93, 110, 81, 88, 77, 57, 43, 39, 27, 30, 11, 12, 2, 1, 4, 0, 1, 0]
hx2 = lambda val: 280 + (val - 2.4) / 6.2 * 1360
HB2 = 800
v.b("Sin estratificar, el porcentaje de fraude en la prueba cambia con cada corte. [[a]]Repetimos mil veces el reparto. [[b]]Y salió desde dos coma ocho, [[c]]hasta ocho coma cuatro por ciento.",
    CL(),
    T("% de fraude en la prueba · 1.000 repartos al azar", 960, 190, tam=44, c="gris", w="sb", ent="subir"),
    L(260, HB2, 1680, HB2, "tinta", 4, dur=0.4),
    [Bx(hx2(2.4 + 0.2 * i) + 2, HB2 - c * 3, hx2(2.6 + 0.2 * i) - 2, HB2, GR_, None, 0, 3, id=f"h2{i}", grow="y", en="a", t=0.02 * i, dur=0.5) for i, c in enumerate(H2) if c],
    [T(f"{tv} %", hx2(tv), HB2 + 36, tam=32, c="gris", w="sb", en="a", t=0.3) for tv in (3, 4, 5, 6, 7, 8)],
    L(hx2(2.8), 400, hx2(2.8), HB2, "rojo", 5, guion=True, id="m28", en="b"), P("2.8 %", hx2(2.8), 360, "rojo", 40, en="b", t=0.3),
    L(hx2(8.4), 400, hx2(8.4), HB2, "rojo", 5, guion=True, id="m84", en="c"), P("8.4 %", hx2(8.4), 360, "rojo", 40, en="c", t=0.3))
v.b("Con estratificación, en cambio, [[a]]siempre es el mismo: cinco coma cuatro por ciento. Una sola línea.",
    L(hx2(5.4), 300, hx2(5.4), HB2, "violeta", 9, id="vs", en="a"), P("estratificado: siempre 5.4 %", hx2(5.4), 260, "violeta", 44, en="a", t=0.3))
v.b("Con pocos datos es peor. [[a]]Tomamos doscientos casos, de los que doce son fraudes. [[b]]La prueba tiene cincuenta filas.",
    CL(), N(200, 960, 200, "{:.0f}", 70, "tinta", suf=" casos", ent="fade", en="a", dur=0.9),
    [Bc(430 + 96 * (i % 12), 400 + 96 * (i // 12), 70, 70, "rojo" if i < 12 else "#CBD5E1", None, 0, 14, en="a", ent="fade", t=0.04 * i) for i in range(12)],
    N(12, 960, 600, "{:.0f}", 120, "rojo", suf=" fraudes", en="a", t=0.3, dur=0.8), P("prueba de 50 filas", 960, 820, "rojo", 52, en="b"))
C2 = [22, 115, 245, 290, 190, 89, 36, 9, 4]
cx = lambda k: 360 + 135 * k
v.b("Mira cuántos fraudes cayeron en la prueba, en mil repartos al azar. [[a]]En dos coma dos por ciento de las veces, no cayó ninguno.",
    CL(),
    T("fraudes en la prueba (de 12)", 960, 190, tam=44, c="gris", w="sb", ent="subir"), L(260, 800, 1620, 800, "tinta", 4, dur=0.4),
    [[Bx(cx(k) - 50, 800 - c * 1.4, cx(k) + 50, 800, "#9CA3AF", None, 0, 6, id=f"c2{k}", grow="y", en=None, t=0.1 * k, dur=0.5), T(str(k), cx(k), 845, tam=36, c="gris", w="sb", t=0.1 * k)] for k, c in enumerate(C2)],
    U("c20", relleno="rojo", dur=0.4, en="a"), P("2.2 % de las veces: cero fraudes", 760, 330, "rojo", 46, en="a", t=0.3))
v.b("Un modelo evaluado sin ningún fraude no puede decirte nada sobre fraudes. [[a]]Es como probar un detector de incendios sin fuego.",
    CL(), P("prueba sin fraudes", 960, 400, "rojo", 66, ent="subir"), T("no dice nada sobre fraudes", 960, 560, tam=54, c="gris", w="sb", en="a"),
    P("detector de incendios, sin fuego", 960, 740, "ambar", 54, en="a", t=0.6))
v.b("Con estratificación, en esos mil repartos la prueba siempre recibió [[a]]exactamente tres fraudes.",
    CL(), T("1.000 repartos", 960, 330, tam=90, c="tinta", t=0.1), T("→", 960, 480, tam=90, c="gris", t=0.3), N(3, 960, 640, "{:.0f}", 190, "verde", suf=" fraudes", en="a", dur=0.8),
    T("siempre", 960, 800, tam=54, c="verde", w="sb", en="a", t=0.3))

# ---------------- 4. Otros
v.b("Hay otros tipos de muestreo. [[a]]El aleatorio simple, cuando no hay estructura ni desbalance. [[b]]El estratificado, para las clases raras. [[c]]Y el consciente del tiempo, cuando los datos tienen orden cronológico.",
    CL(),
    [[Bc(x, 480, 520, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 380, tam=48, c=c, en=k, t=0.1), T(d, x, 520, tam=34, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "gris", "aleatorio simple", "sin estructura ni desbalance", "a"), (960, "violeta", "estratificado", "clases raras", "b"), (1560, "azul", "consciente del tiempo", "orden cronológico", "c"))] if False else
    [[Bc(x, 480, 520, 360, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 380, tam=44, c=c, mw=440, en=k, t=0.1), T(d, x, 540, tam=34, c="tinta", w="sb", mw=420, en=k, t=0.25)]
     for x, c, n, d, k in ((360, "teal", "aleatorio simple", "sin estructura ni desbalance", "a"), (960, "violeta", "estratificado", "clases raras", "b"), (1560, "azul", "consciente del tiempo", "orden cronológico", "c"))],
    sec="4. Otros muestreos y K-Fold")
v.b("Y si usas K-Fold con clasificación desbalanceada, [[a]]usa Estratificado K-Fold: cada pliegue conserva la proporción de las clases.",
    CL(), T("StratifiedKFold", 960, 400, tam=80, c="violeta", w="m", fondo="#E8EDF4", pad=30, en="a", ent="subir"),
    P("cada pliegue conserva la proporción", 960, 640, "teal", 56, en="a", t=0.5))
v.b("Piensa un segundo: tienes mil filas y dos por ciento de impago. [[a]]Separas una prueba de veinte por ciento, con estratificación. ¿Cuántos impagos caen en ella?",
    CL(), T("1.000 filas  ·  2 % de impago", 960, 360, tam=60, c="tinta", ent="subir", t=0.2), T("prueba: 20 %  =  200 filas", 960, 520, tam=60, c="rojo", t=0.5),
    g("q3", T("¿cuántos impagos?", 960, 720, tam=72, c="violeta", en="a")), pausa=0.5)
v.b("Mil por dos por ciento son veinte impagos. [[a]]Y el veinte por ciento de veinte es cuatro.",
    XG("q3", dur=0.3), T("1.000 × 0.02  =  20", 960, 640, tam=64, c="tinta", w="sb", t=0.2), T("20 × 0.20  =  4", 960, 760, tam=64, c="tinta", w="sb", en="a"),
    P("4 impagos", 960, 920, "verde", 70, en="a", t=0.5))
v.b("Para llevarte: [[a]]con clases raras, estratifica. [[b]]Así cada partición conserva la proporción de cada clase. [[c]]No agrega casos raros: solo los reparte bien.",
    CL(), P("clase rara  →  estratifica", 960, 330, "violeta", 62, en="a"), P("cada partición conserva la proporción", 960, 520, "teal", 56, en="b"),
    P("no agrega fraudes: los reparte bien", 960, 710, "ambar", 56, en="c"), sec="5. Para llevar")

print(v.guardar(OUT), "beats")
