import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_06_kfold.json"
v = V("Validación cruzada K-Fold")

v.portada("Validación cruzada K-Fold", "Un resultado estable, en lugar de uno de suerte",
          "Validación cruzada, K-Fold. Cómo evaluar un modelo sin depender de la suerte de una sola división.")

# ---------------- 1. El problema
v.b("Imagina que tu profesor te pone un solo examen. [[a]]Si justo salen las preguntas que dominas, sacas un siete. [[b]]Si salen las que no estudiaste, sacas un cuatro.",
    CL(),
    Tj(480, 560, 560, 520, "Examen A", ["Pregunta 1", "Pregunta 2", "Pregunta 3"], "verde", en="a", dur=0.7),
    N(7.0, 480, 920, "{:.1f}", 90, "verde", en="a", t=0.4, dur=0.9),
    Tj(1440, 560, 560, 520, "Examen B", ["Pregunta 1", "Pregunta 2", "Pregunta 3"], "rojo", en="b", dur=0.7),
    N(4.0, 1440, 920, "{:.1f}", 90, "rojo", en="b", t=0.4, dur=0.9),
    sec="1. El problema de una sola partición")
v.b("¿Cuál de las dos notas dice cuánto sabes de verdad? [[a]]Ninguna: depende de la suerte del examen.",
    P("¿cuál dice cuánto sabes?", 960, 980, "violeta", 54, t=0.2), R(480, 920, 300, 130, "ambar", 40, id="rn1", en=None, t=0.5), R(1440, 920, 300, 130, "ambar", 40, id="rn2", en=None, t=0.9),
    T("depende de la suerte", 960, 780, tam=52, c="rojo", en="a"))
v.b("Con los datos pasa igual. Si divides en entrenamiento y validación una sola vez, [[a]]esa partición puede salir fácil, [[b]]o difícil, solo por azar.",
    CL(),
    T("20 filas de ejemplo", 960, 200, tam=44, c="gris", w="sb", ent="subir"),
    [Bx(280 + 66 * i, 300, 280 + 66 * i + 58, 370, "#C4B5FD", None, 0, 8, id=f"s1{i}", ent="fade", t=0.03 * i) for i in range(20)],
    [Bx(280 + 66 * i, 560, 280 + 66 * i + 58, 630, "#C4B5FD", None, 0, 8, id=f"s2{i}", ent="fade", t=0.03 * i + 0.1) for i in range(20)],
    [U(f"s1{i}", relleno="azul", dur=0.4, en="a", t=0.05 * k) for k, i in enumerate([1, 6, 8, 13, 17])],
    [U(f"s2{i}", relleno="azul", dur=0.4, en="b", t=0.05 * k) for k, i in enumerate([2, 3, 9, 14, 18])],
    T("partición A", 160, 335, tam=36, c="gris", w="sb", en="a"), N(0.96, 1740, 335, "{:.2f}", 64, "verde", en="a", t=0.5, dur=0.8),
    T("partición B", 160, 595, tam=36, c="gris", w="sb", en="b"), N(0.82, 1740, 595, "{:.2f}", 64, "rojo", en="b", t=0.5, dur=0.8))
v.b("[[a]]Mismo modelo, mismos datos. [[b]]Solo cambió el corte, y la nota cambió mucho.",
    P("mismo modelo, mismos datos", 960, 820, "violeta", 54, en="a"), P("solo cambió el corte", 960, 930, "ambar", 54, en="b"))

HC_ = [1, 0, 0, 0, 1, 2, 3, 5, 5, 2, 5, 20, 14, 13, 22, 32, 34, 39, 32, 29, 19, 11, 6, 5]
HE_ = [0.819, 0.825, 0.831, 0.837, 0.843, 0.849, 0.855, 0.861, 0.867, 0.873, 0.879, 0.885, 0.891, 0.897, 0.904, 0.91, 0.916, 0.922, 0.928, 0.934, 0.94, 0.946, 0.952, 0.958, 0.964]
HB_ = 800
hx = lambda val: 300 + (val - 0.80) / 0.18 * 1300


def hist(en, t0=0.0):
    out = [L(280, HB_, 1640, HB_, "tinta", 4, id="hbase", dur=0.4, en=en, t=t0)]
    out += [Bx(hx(HE_[i]) + 2, HB_ - c * 8, hx(HE_[i + 1]) - 2, HB_, "#94A3B8", None, 0, 4, id=f"hb{i}", grow="y", en=en, t=t0 + 0.05 * i, dur=0.5) for i, c in enumerate(HC_)]
    out += [T(f"{tv:.2f}", hx(tv), HB_ + 36, tam=32, c="gris", w="sb", id=f"ht{k}", en=en, t=t0 + 0.3) for k, tv in enumerate([0.80, 0.85, 0.90, 0.95])]
    return out


v.b("En las viviendas de la clase lo comprobamos. Repetimos trescientas particiones simples y medimos el R cuadrado de validación. [[a]]Mira cómo se reparten los resultados.",
    CL(), T("300 particiones simples  ·  R² de validación", 960, 200, tam=48, c="gris", w="sb", ent="subir"), hist("a"))
v.b("Algunas particiones dan cerca de [[a]]cero coma ochenta y dos. [[b]]Otras, cerca de cero coma noventa y seis.",
    L(hx(0.82), 300, hx(0.82), HB_, "rojo", 5, guion=True, id="vmin", en="a"), P("0.82", hx(0.82), 250, "rojo", 44, en="a", t=0.3),
    L(hx(0.96), 300, hx(0.96), HB_, "verde", 5, guion=True, id="vmax", en="b"), P("0.96", hx(0.96), 250, "verde", 44, en="b", t=0.3))
v.b("Con mala suerte, habrías creído que el modelo es regular. [[a]]Con buena suerte, que es excelente. [[b]]Y es el mismo modelo.",
    P("mala suerte: parece regular", 560, 450, "rojo", 40, t=0.3), P("buena suerte: parece excelente", 1500, 450, "verde", 40, en="a"),
    T("el mismo modelo", 960, 960, tam=56, c="violeta", en="b"))

# ---------------- 2. La idea K-Fold
v.b("La solución se llama validación cruzada, o K-Fold. [[a]]La idea: en vez de un solo corte, [[b]]hacemos varios y promediamos.",
    CL(),
    P("K-Fold", 960, 330, "violeta", 130, ent="pop", t=0.2), T("validación cruzada", 960, 500, tam=60, c="gris", w="sb", en="a"),
    P("varios cortes  →  promedio", 960, 720, "teal", 64, en="b"), sec="2. Cómo funciona K-Fold")
RY = [390, 480, 570, 660, 750]
bx = lambda j: 380 + 260 * j


def bloque(i, j, c, en, t, ent="fade"):
    return Bx(bx(j), RY[i] - 38, bx(j) + 240, RY[i] + 38, c, None, 0, 10, id=f"fb{i}{j}", en=en, t=t, ent=ent)


def fila(i, en, t0=0.0, val=None):
    out = []
    for j in range(5):
        out.append(bloque(i, j, "#C4B5FD", en, t0 + 0.06 * j))
    out.append(T(f"Pliegue {i + 1}", 340, RY[i], tam=34, c="tinta", w="sb", anc="rm", id=f"fl{i}", en=en, t=t0))
    return out


v.b("Paso uno: partimos los datos en K partes del mismo tamaño, que llamamos pliegues. [[a]]K lo eliges tú. Por ejemplo, cinco.",
    CL(), fila(0, "a"), T("datos partidos en K = 5 pliegues", 960, 170, tam=48, c="gris", w="sb", ent="subir", t=0.2),
    [T(str(j + 1), bx(j) + 120, RY[0], tam=44, c="violeta", id=f"fn{j}", en="a", t=0.06 * j + 0.2) for j in range(5)])
v.b("Paso dos: [[a]]el primer pliegue hace de validación. [[b]]Los otros cuatro entrenan. Y se calcula la nota.",
    U("fb00", relleno="azul", dur=0.5, en="a"),
    P("valida", bx(0) + 120, 300, "azul", 36, en="a"), P("entrenan", bx(2) + 120, 300, "violeta", 36, en="b"),
    T("nota 1", 1800, RY[0], tam=44, c="azul", w="b", en="b", t=0.9))
v.b("Paso tres: repetimos rotando. [[a]]Ahora valida el segundo pliegue. [[b]]Luego el tercero, [[c]]el cuarto, [[d]]y el quinto.",
    fila(1, "a", 0.0), U("fb11", relleno="azul", dur=0.4, en="a", t=0.6), T("nota 2", 1800, RY[1], tam=44, c="azul", en="a", t=0.9),
    fila(2, "b", 0.0), U("fb22", relleno="azul", dur=0.4, en="b", t=0.6), T("nota 3", 1800, RY[2], tam=44, c="azul", en="b", t=0.9),
    fila(3, "c", 0.0), U("fb33", relleno="azul", dur=0.4, en="c", t=0.6), T("nota 4", 1800, RY[3], tam=44, c="azul", en="c", t=0.9),
    fila(4, "d", 0.0), U("fb44", relleno="azul", dur=0.4, en="d", t=0.6), T("nota 5", 1800, RY[4], tam=44, c="azul", en="d", t=0.9))
v.b("Cada pliegue validó una vez, y entrenó las otras cuatro. [[a]]Así todos los datos tienen su turno de ser examen.",
    R(1020, 570, 1380, 490, "ambar", 40, id="rg", en="a", ent="fade"), g("tn", P("todos tienen su turno de validar", 960, 900, "teal", 52, en="a", t=0.3)))
v.b("Paso cuatro: [[a]]promediamos las cinco notas. [[b]]Y miramos también cuánto varían entre sí: la desviación.",
    X("rg", dur=0.2), XG("tn", dur=0.2), P("promedio de las 5 notas", 700, 900, "azul", 48, en="a"), P("desviación entre pliegues", 1380, 900, "violeta", 48, en="b"),
    T("el resultado es un promedio y una dispersión", 960, 1000, tam=40, c="gris", w="sb", en="b", t=0.6))

# ejemplo a mano
v.b("Hagamos un ejemplo a mano. Tenemos cien filas y cuatro pliegues. [[a]]Cada pliegue tiene cien entre cuatro: veinticinco filas de validación. [[b]]Y entrenamos con las otras setenta y cinco.",
    CL(),
    T("100 filas  ·  K = 4", 960, 190, tam=54, c="tinta", ent="subir"),
    Pt([(25, [260, 300, 560, 720], "azul"), (75, [660, 300, 1660, 720], "#C4B5FD")], 12, visibles=100, id="pk", en="a", dur=1.6),
    N(25, 410, 800, "{:.0f}", 90, "azul", suf=" valida", en="a", t=0.5, dur=0.8),
    N(75, 1160, 800, "{:.0f}", 90, "violeta", suf=" entrenan", en="b", dur=1.0), sec="3. Un ejemplo a mano")
S4 = [0.80, 0.84, 0.78, 0.86]
BS = 860
v.b("Las cuatro notas fueron cero coma ochenta, cero coma ochenta y cuatro, cero coma setenta y ocho y cero coma ochenta y seis. [[a]]El promedio es la suma, tres coma veintiocho, entre cuatro: [[b]]cero coma ochenta y dos.",
    CL(),
    L(300, BS, 1500, BS, "tinta", 4, dur=0.4),
    [[Bx(450 + 280 * i, BS - s_ * 600, 450 + 280 * i + 150, BS, "azul", None, 0, 8, id=f"kb{i}", grow="y", en=None, t=0.3 + 0.35 * i, dur=0.5),
      T(f"{s_:.2f}", 525 + 280 * i, BS - s_ * 600 - 40, tam=44, c="tinta", t=0.7 + 0.35 * i)] for i, s_ in enumerate(S4)],
    T("(0.80 + 0.84 + 0.78 + 0.86) / 4  =  3.28 / 4", 960, 200, tam=50, c="tinta", w="sb", en="a"),
    L(300, BS - 0.82 * 600, 1500, BS - 0.82 * 600, "violeta", 6, guion=True, id="kmed", en="b"), T("promedio 0.82", 1560, BS - 0.82 * 600, tam=44, c="violeta", anc="lm", en="b", t=0.3))
v.b("La desviación mide cuánto se alejan las notas del promedio. [[a]]Las diferencias son cero coma cero dos, cero coma cero dos, cero coma cero cuatro y cero coma cero cuatro. [[b]]Las elevamos al cuadrado, las promediamos y sacamos la raíz.",
    CL(),
    T("diferencias con el promedio", 960, 200, tam=46, c="gris", w="sb", ent="subir"),
    [[Bc(430 + 350 * i, 400, 260, 150, "blanco", "linea", 3, 20, en="a", t=0.25 * i), T(f"{d_:.2f}", 430 + 350 * i, 400, tam=64, c="tinta", en="a", t=0.25 * i + 0.05)] for i, d_ in enumerate([0.02, 0.02, 0.04, 0.04])],
    T("√( (0.02² + 0.02² + 0.04² + 0.04²) / 4 )", 960, 620, tam=56, c="tinta", w="sb", en="b"),
    T("= √0.001", 960, 740, tam=64, c="violeta", en="b", t=0.9))
v.b("Da cero coma cero treinta y dos. [[a]]Una desviación pequeña: las notas se parecen entre sí.",
    T("= 0.032", 960, 900, tam=100, c="verde", en=None), P("desviación pequeña = resultado estable", 960, 1000, "verde", 44, en="a"))
v.b("Piensa un segundo: [[a]]si la desviación entre pliegues fuera muy grande, ¿qué te diría del modelo?",
    CL(), T("desviación muy grande", 960, 400, tam=80, c="rojo", t=0.2), T("¿qué significa?", 960, 580, tam=64, c="violeta", en="a"), pausa=0.5)
v.b("Que el resultado depende mucho de la muestra: [[a]]el modelo no es estable, y no te puedes fiar de un solo número.",
    P("depende mucho de la muestra", 960, 760, "rojo", 60), P("poco estable", 960, 880, "ambar", 52, en="a"))

# ---------------- 3. Viviendas
v.b("Volvamos a las viviendas de la clase. [[a]]De las trescientas, cuarenta y cinco quedan reservadas para la prueba. [[b]]K-Fold se hace con las doscientas cincuenta y cinco restantes.",
    CL(),
    Bx(160, 380, 1450, 520, "violeta", None, 0, 20, id="dv", en="a", ent="subir"), Bx(1470, 380, 1760, 520, "rojo", None, 0, 20, id="dt", en="a", ent="subir"),
    T("255 viviendas de desarrollo", 805, 450, tam=50, c="blanco", en="b"), T("prueba: 45", 1615, 450, tam=44, c="blanco", en="a", t=0.3),
    Cand(1615, 640, 90, "rojo", 0, False, en="a", t=0.4), T("fuera de todo el K-Fold", 1615, 760, tam=34, c="rojo", w="sb", en="a", t=0.7),
    T("300 viviendas", 960, 250, tam=54, c="tinta", ent="subir"), sec="4. K-Fold en las viviendas de la clase")
v.b("Con cinco pliegues, cada uno tiene cincuenta y uno viviendas. [[a]]Cinco veces cincuenta y uno, doscientas cincuenta y cinco.",
    [[Bx(160 + 258 * j + 4, 560, 160 + 258 * (j + 1) - 4, 660, "azul" if j == 0 else "#C4B5FD", None, 0, 10, en="a", t=0.1 * j, ent="fade"),
      T("51", 160 + 258 * j + 129, 610, tam=48, c="blanco", en="a", t=0.1 * j + 0.1)] for j in range(5)],
    T("5 × 51 = 255", 805, 780, tam=64, c="violeta", en="a", t=0.7))
FK = [0.941, 0.931, 0.918, 0.895, 0.933]
KB = 860
ky = lambda val: KB - (val - 0.80) * 3200
v.b("Estos fueron los R cuadrado de validación en cada pliegue: [[a]]cero coma novecientos cuarenta y uno, [[b]]cero coma novecientos treinta y uno, [[c]]cero coma novecientos dieciocho, [[d]]cero coma ochocientos noventa y cinco, [[e]]y cero coma novecientos treinta y tres.",
    CL(),
    T("R² por pliegue (eje desde 0.80)", 960, 200, tam=46, c="gris", w="sb", ent="subir"), L(300, KB, 1560, KB, "tinta", 4, dur=0.4),
    [[Bx(420 + 230 * i, ky(s_), 420 + 230 * i + 150, KB, "azul", None, 0, 8, id=f"fk{i}", grow="y", en="abcde"[i], dur=0.6),
      T(f"{s_:.3f}", 495 + 230 * i, ky(s_) - 36, tam=40, c="tinta", id=f"fkl{i}", en="abcde"[i], t=0.5),
      T(f"pliegue {i + 1}", 495 + 230 * i, KB + 44, tam=32, c="gris", w="sb", en="abcde"[i], t=0.2)] for i, s_ in enumerate(FK)])
v.b("El promedio es cero coma novecientos veinticuatro, [[a]]y la desviación, cero coma cero dieciséis. [[b]]Un resultado alto, y estable.",
    L(300, ky(0.9237), 1560, ky(0.9237), "violeta", 6, guion=True, id="fmed", en="a"), T("media 0.924", 1620, ky(0.9237), tam=44, c="violeta", anc="lm", en="a", t=0.3),
    Bx(300, ky(0.9237 + 0.0159), 1560, ky(0.9237 - 0.0159), "violetaclaro", None, 0, 0, id="fband", z=0.2, en="a", t=0.5, ent="fade"),
    T("± 0.016", 1620, ky(0.9237) + 60, tam=40, c="violeta", w="sb", anc="lm", en="b"))
v.b("Compáralo con una sola partición. [[a]]Allí el resultado podía ir de cero coma ochenta y dos a cero coma noventa y seis. [[b]]K-Fold te da un número estable, con su dispersión.",
    CL(), T("una partición  vs  K-Fold", 960, 200, tam=48, c="gris", w="sb", ent="subir"), hist("a"),
    L(hx(0.9237), 300, hx(0.9237), HB_, "violeta", 7, id="hk", en="b"), P("K-Fold: 0.924", hx(0.9237), 260, "violeta", 44, en="b", t=0.3))
v.b("Un detalle de la clase: [[a]]el cuaderno hizo el K-Fold con las trescientas filas, [[b]]y eso incluye la prueba. Dio cero coma novecientos veintiuno. Lo correcto es usar solo entrenamiento más validación.",
    CL(),
    Bx(160, 380, 1450, 520, "violeta", None, 0, 20, ent="subir"), Bx(1470, 380, 1760, 520, "rojo", None, 0, 20, ent="subir", t=0.1),
    T("300 filas", 960, 450, tam=50, c="blanco", en="a"), R(960, 450, 1660, 190, "ambar", 40, en="b", id="r300"),
    P("incluye la prueba", 1615, 640, "rojo", 44, en="b", t=0.3), K(1615, 760, "cruz", 80, "rojo", en="b", t=0.6),
    N(0.921, 960, 880, "{:.3f}", 80, "gris", en="a", t=0.4, dur=0.8), T("incorrecto: 0.921", 960, 980, tam=40, c="rojo", w="sb", en="b", t=0.3))
v.b("K-Fold se usa durante el desarrollo, para comparar y elegir modelos. [[a]]La prueba sigue reservada hasta el final, y K-Fold no la reemplaza.",
    CL(), P("K-Fold: durante el desarrollo", 960, 380, "azul", 58, ent="subir"), P("la prueba: al final, una sola vez", 960, 560, "rojo", 58, en="a"),
    Cand(960, 780, 100, "rojo", 0, True, en="a", t=0.4))
v.b("Y si trabajas con un Pipeline, [[a]]K-Fold repite todos los pasos dentro de cada pliegue, sin filtrar información.",
    CL(),
    T("cross_val_score(pipe, X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=42))", 960, 440, tam=34, c="tinta", w="m", fondo="#E8EDF4", pad=26, ent="subir", mw=1500),
    P("cada pliegue re-aprende sus pasos", 960, 700, "teal", 56, en="a"))
v.b("Piensa un segundo: con doscientas cincuenta y cinco filas y cinco pliegues, [[a]]¿cuántas filas entrenan en cada vuelta?",
    CL(), T("255 filas  ·  5 pliegues", 960, 400, tam=64, c="tinta", ent="subir"), T("¿cuántas entrenan?", 960, 600, tam=72, c="violeta", en="a"), pausa=0.5)
v.b("Doscientas cuatro: [[a]]doscientas cincuenta y cinco menos las cincuenta y una que validan.",
    P("204", 960, 760, "verde", 130), T("255 − 51 = 204", 960, 930, tam=60, c="tinta", w="sb", en="a"))
v.b("Para llevarte: [[a]]K-Fold te da un promedio y una dispersión, [[b]]en lugar de un solo número de suerte. [[c]]Y se hace con entrenamiento y validación: la prueba queda fuera.",
    CL(),
    P("promedio + dispersión", 960, 330, "violeta", 60, en="a"), P("no un solo número de suerte", 960, 510, "ambar", 56, en="b"),
    P("la prueba queda fuera", 960, 690, "rojo", 56, en="c"), sec="5. Para llevar")

print(v.guardar(OUT), "beats")
