import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, nube, serie_curva, fila)

OUT = out("guion_c37_08_gamma.json")
v = V("C y gamma: subajuste y sobreajuste")
Z = D["z2_train"]
RB = D["rbf2"]
G4 = FG["gamma4d"]
RJ = FG["rejilla_prof"]
CAJA = (-2.1, 4.0, -1.5, 4.4)

v.portada("C y gamma", "Subajuste, sobreajuste y el punto justo",
          "Octava parada: C y gamma juntos, con los datos reales. Vas a ver un modelo que memoriza, uno que no aprende, y cómo encontrar el punto justo.")

# ---- 1. Hook
g50, g1 = RB["50"], RB["1"]
v.b("Una SVM radial con gamma igual a cincuenta acierta el 96 por ciento del entrenamiento. [[a]]Con gamma igual a uno, solo el 93.2. [[b]]¿Gana la de cincuenta? Mira la prueba: 88.9 contra 90.6. [[c]]La que más acierta en entrenamiento es la peor en casos nuevos. Memorizar no es aprender.",
    CL(), T("¿qué modelo es mejor?", 960, 150, tam=54, c="tinta", ent="subir"),
    fila(330, [("", 520), ("entrenamiento", 960), ("prueba", 1400)], tams=[40, 40, 40], colores=["gris"] * 3),
    fila(440, [("γ = 50", 520), (f"{100 * g50['acc_train']:.1f} %", 960), (f"{100 * g50['acc_test']:.1f} %", 1400)], tams=[48, 56, 56],
         colores=["rojo", "verde", "rojo"], fondo="rojoclaro", x0=340, x1=1580, h=100),
    fila(570, [("γ = 1", 520), (f"{100 * g1['acc_train']:.1f} %", 960), (f"{100 * g1['acc_test']:.1f} %", 1400)], tams=[48, 56, 56],
         colores=["azul", "tinta", "verde"], en="a", fondo="azulclaro", x0=340, x1=1580, h=100),
    R(1400, 505, 300, 270, "ambar", r=26, en="b"), P("memorizar no es aprender", 960, 780, "ambar", 52, en="c"),
    T("SVM radial, C = 1, 2 variables", 1860, 1010, tam=26, c="gris", w="sb", anc="rm"), real(1050), sec="1. Memorizar no es aprender")

# ---- 2. Analogía: estudiar para la prueba
v.b("Piensa en tres estudiantes antes de una prueba. [[a]]El primero solo leyó los títulos: no entiende casi nada, y le va mal en todo. Eso es subajuste. [[b]]El segundo memorizó las respuestas exactas del ensayo: en el ensayo saca siete, pero en la prueba real, con preguntas nuevas, se cae. Eso es sobreajuste. [[c]]El tercero entendió las ideas: le va bien en el ensayo, y también en la prueba.",
    CL(), T("tres estudiantes", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "leyó títulos", "mal en todo: SUBAJUSTE", "a"), (960, "rojo", "memorizó el ensayo", "bien en el ensayo, mal en la prueba: SOBREAJUSTE", "b"),
              (1520, "verde", "entendió", "bien en los dos", "c")], y=520, w=500, h=360, tam_t=44, tam_d=34),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: tres estudiantes")

# ---- 3. Gamma con datos reales
SEGS = [("0.01", "verde"), ("1", "azul"), ("50", "rojo")]
series = nube((Z["x"], Z["y"]), Z["c"], mk=6)
idx = {}
for k, col in SEGS:
    idx[k] = list(range(len(series), len(series) + len(RB[k]["segs"])))
    series += [serie_curva(s["x"], s["y"], col, 6) for s in RB[k]["segs"]]


def ver(k, a):
    return {f"series.{i}.alfa": a for i in idx[k]}


v.b("Ahora lo vemos en el plano, con las dos variables y C igual a uno. [[a]]Con gamma igual a 0.01, la frontera es casi una recta suave. Entrenamiento, 92.2; prueba, 90.6. [[b]]Con gamma igual a uno, la frontera se curva un poco más: 93.2 y 90.6. [[c]]Y con gamma igual a cincuenta, aparecen islas: pequeñas fronteras cerradas alrededor de casos sueltos. Entrenamiento, 96.0; prueba, 88.9.",
    CL(), T("la frontera radial para tres valores de γ (C = 1)", 960, 140, tam=46, c="tinta", ent="subir"),
    G(420, 230, 1360, 920, *CAJA, series, xticks=[-2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="radio (estandarizado)",
      ylabel="concav.", id="gg", tam_txt=28),
    U("gg", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.3, dur=1.2),
    U("gg", **ver("0.01", 1.0), en="a", dur=0.6),
    T(f"γ = 0.01: {100 * RB['0.01']['acc_train']:.1f} % / {100 * RB['0.01']['acc_test']:.1f} %", 1640, 380, tam=34, c="verde", w="sb", en="a", t=0.3),
    U("gg", **ver("0.01", 0.0), **ver("1", 1.0), en="b", dur=0.8),
    T(f"γ = 1: {100 * RB['1']['acc_train']:.1f} % / {100 * RB['1']['acc_test']:.1f} %", 1640, 450, tam=34, c="azul", w="sb", en="b", t=0.3),
    U("gg", **ver("1", 0.0), **ver("50", 1.0), en="c", dur=0.8),
    T(f"γ = 50: {100 * RB['50']['acc_train']:.1f} % / {100 * RB['50']['acc_test']:.1f} %", 1640, 520, tam=34, c="rojo", w="sb", en="c", t=0.3),
    T("entrenamiento / prueba", 1640, 310, tam=28, c="gris", w="sb", t=0.6), real(1050), sec="3. γ con datos reales")
v.b("Fíjate en cuántos vectores de soporte necesita cada uno. [[a]]Con gamma 0.01, 155. Con gamma uno, 92. [[b]]Con gamma cincuenta, 279 de los 398 casos. El modelo se apoya en casi todos los puntos: los está memorizando.",
    T(f"vectores de soporte: {RB['0.01']['n_sv']} · {RB['1']['n_sv']} · {RB['50']['n_sv']}", 1640, 650, tam=32, c="tinta", w="sb", en="a"),
    P("γ = 50: 279 de 398 → memoriza", 1640, 760, "rojo", 34, en="b"))

# ---- 4. La curva de validación
LG = [math.log10(g) for g in G4["gamma"]]
v.b("Ahora con las cuatro variables, y dieciséis valores de gamma, de 0.001 a cien. [[a]]En gris, la exactitud de entrenamiento: sube y sube, hasta llegar al cien por ciento. [[b]]En azul, la validación cruzada: sube, llega a un máximo, y después se desploma. [[c]]El máximo está en gamma igual a 0.046: 94.5 por ciento.",
    CL(), T("validación cruzada en entrenamiento · SVM radial, C = 1, 4 variables", 960, 140, tam=44, c="tinta", ent="subir"),
    G(540, 240, 1400, 860, -3, 2, 0.5, 1.0, [dict(xs=LG, ys=G4["acc_train"], color="gris", grosor=4, marcadores=7, revela=0.0),
                                             dict(xs=LG, ys=G4["acc_cv"], color="azul", grosor=6, marcadores=8, revela=0.0)],
      xticks=[-3, -2, -1, 0, 1, 2], yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], yfmt="{:.1f}", xlabel="log₁₀ γ", ylabel="exact.", id="gv", en="a"),
    U("gv", **{"series.0.revela": 1.0}, en="a", t=0.2, dur=2.0), U("gv", **{"series.1.revela": 1.0}, en="b", t=0.2, dur=2.2),
    T("entrenamiento", 1620, 300, tam=34, c="gris", w="sb", en="a"), T("validación cruzada", 1620, 360, tam=34, c="azul", w="sb", en="b"),
    R(math.log10(G4["mejor_gamma_cv"]), G4["mejor_acc_cv"], 60, 60, "verde", ref="gv", en="c"),
    T(f"mejor γ ≈ {G4['mejor_gamma_cv']:.3f}: {100 * G4['mejor_acc_cv']:.1f} %", 1620, 470, tam=34, c="verde", w="sb", en="c"),
    real(1050, en="a"), sec="4. La curva de validación")
v.b("Las dos puntas cuentan la historia. [[a]]A la izquierda, con gamma igual a 0.001, la validación da 79.2 por ciento: el modelo es demasiado rígido. Subajuste. [[b]]A la derecha, con gamma igual a cien, el entrenamiento es perfecto, y la validación cae a 63.3. [[c]]En la prueba, ese modelo acierta 62.6 por ciento: exactamente lo mismo que decir siempre benigno.",
    R(-3, G4["acc_cv"][0], 60, 60, "gris", ref="gv", en="a"), T(f"γ = 0.001: {100 * G4['acc_cv'][0]:.1f} % → subajuste", 1620, 560, tam=32, c="gris", w="sb", en="a"),
    R(2, G4["acc_cv"][-1], 60, 60, "rojo", ref="gv", en="b"), T(f"γ = 100: entr. 100 % · CV {100 * G4['acc_cv'][-1]:.1f} %", 1620, 640, tam=32, c="rojo", w="sb", en="b"),
    P(f"prueba (γ = 100): {100 * G4['acc_test'][-1]:.1f} %", 1620, 740, "ambar", 32, en="c"),
    T("= «siempre benigno»", 1620, 810, tam=32, c="ambar", w="sb", en="c", t=0.4))
v.b("¿Por qué pasa eso? [[a]]Con gamma enorme, cada vector de soporte solo influye en un radio minúsculo. [[b]]Un caso nuevo queda lejos de todos, todas sus similitudes valen casi cero, y el puntaje se reduce a la constante be. [[c]]El modelo termina diciendo lo mismo para todos: la clase más frecuente.",
    CL(), P("γ enorme → cada vector influye en un radio minúsculo", 960, 300, "rojo", 46, en="a"),
    P("caso nuevo: todas las K ≈ 0 → f(x) ≈ b", 960, 460, "violeta", 48, en="b"),
    P("→ predice la clase más frecuente para todos", 960, 620, "ambar", 48, en="c"))

# ---- 5. C y gamma juntos
RJT = {(r["C"], r["gamma"]): r["media"] for r in RJ["tabla"]}
CS, GS = [0.1, 1.0, 10.0], [0.01, 0.1, 1.0]


def color_celda(m):
    return "verde" if m >= 0.935 else ("azul" if m >= 0.92 else ("ambar" if m >= 0.9 else "rojo"))


celdas = []
for i, Cv in enumerate(CS[::-1]):
    for j, gv in enumerate(GS):
        m = RJT[(Cv, gv)]
        cc = color_celda(m)
        celdas += [Bc(760 + 220 * j, 360 + 170 * i, 200, 150, cc + "claro", cc, 4, 18, en="a", t=0.06 * (3 * i + j)),
                   T(f"{m:.3f}", 760 + 220 * j, 360 + 170 * i, tam=46, c=cc, en="a", t=0.06 * (3 * i + j) + 0.05)]
v.b("C y gamma actúan juntos. Esta es la rejilla del bloque opcional de la clase: tres valores de C por tres de gamma, con validación cruzada de cinco folds. [[a]]Cada casilla es la exactitud media de validación. [[b]]Abajo a la izquierda, C igual a 0.1 y gamma 0.01: los dos chicos, el modelo más rígido: 79.4 por ciento. [[c]]El máximo está en C igual a diez y gamma 0.1, la configuración de la clase: 94.0.",
    CL(), T("la rejilla de la clase: exactitud media (5 folds, entrenamiento)", 960, 140, tam=44, c="tinta", ent="subir"),
    *[T(f"C = {c:g}", 600, 360 + 170 * i, tam=38, c="gris", w="sb", anc="rm") for i, c in enumerate(CS[::-1])],
    *[T(f"γ = {g:g}", 760 + 220 * j, 250, tam=38, c="gris", w="sb") for j, g in enumerate(GS)],
    *celdas, R(760, 700, 220, 170, "rojo", r=20, en="b"), R(980, 360, 220, 170, "verde", r=20, en="c"),
    P("C y γ chicos → rígido (subajuste)", 1600, 700, "rojo", 30, en="b"), P("C = 10, γ = 0.1 → la mejor", 1600, 360, "verde", 30, en="c"),
    real(1050, en="a"), sec="5. C y γ juntos")
v.b("Y otra cosa que muestra la rejilla: [[a]]subir {C|, ce,} de uno a diez con gamma igual a uno empeora: de 92.7 a 91.5. [[b]]C grande con gamma grande es la combinación más propensa a sobreajustar: castiga cada error y además mira cada punto de cerca.",
    R(1200, 360, 220, 170, "ambar", r=20, en="a"), T("C = 10, γ = 1 < C = 1, γ = 1", 1600, 500, tam=30, c="ambar", w="sb", en="a"),
    P("C y γ grandes → riesgo de sobreajuste", 960, 990, "ambar", 40, en="b"))

# ---- 6. El error típico
v.b("El error típico: [[a]]elegir gamma mirando la exactitud de entrenamiento. [[b]]Con ese criterio, cualquier gamma de 21 en adelante sería perfecta: cien por ciento. [[c]]Y en validación esos modelos aciertan un 63 por ciento. La exactitud de entrenamiento siempre premia la memoria.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("elegir γ por la exactitud de entrenamiento", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("γ ≥ 21.5:  entrenamiento = 100 %", 960, 420, tam=50, c="tinta", w="sb", en="b"),
    T(f"…y validación ≈ {100 * G4['acc_cv'][-3]:.0f} %", 960, 540, tam=50, c="rojo", w="sb", en="c"),
    P("la exactitud de entrenamiento premia la memoria", 960, 700, "ambar", 48, en="c", t=1.0), real(1050, en="b"), sec="6. El error típico")

# ---- 7. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un modelo da 99 por ciento en entrenamiento y 70 en validación. Otro da 80 y 79. [[b]]¿Cuál sobreajusta? ¿Y el otro, qué problema podría tener?",
    CL(), P("modelo A: 99 % entrenamiento · 70 % validación", 960, 330, "rojo", 48, en="a"),
    P("modelo B: 80 % entrenamiento · 79 % validación", 960, 470, "gris", 48, en="a", t=0.6), P("¿cuál sobreajusta? ¿y el otro?", 960, 640, "violeta", 48, en="b"),
    ilustrativo(1000), sec="7. Piensa un segundo", pausa=0.6)
v.b("El modelo A sobreajusta: [[a]]una brecha de veintinueve puntos entre entrenamiento y validación. [[b]]El modelo B no sobreajusta, pero puede estar subajustando: los dos números son bajos y casi iguales. [[c]]Lo que buscamos es lo mejor de ambos: validación alta, y cerca del entrenamiento.",
    CL(), P("A: brecha de 29 puntos → sobreajusta", 960, 320, "rojo", 50, en="a"),
    P("B: bajos y parejos → puede subajustar", 960, 480, "gris", 50, en="b"), P("ideal: validación alta y cerca del entrenamiento", 960, 640, "verde", 48, en="c"),
    ilustrativo(1000))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]gamma chica da fronteras suaves y puede subajustar; gamma grande da islas y sobreajusta. [[b]]C y gamma actúan juntos: grandes los dos, más riesgo de memorizar. [[c]]Y se eligen con validación cruzada, nunca por la exactitud de entrenamiento. En la clase, la mejor combinación fue {C|, ce,} igual a diez y gamma 0.1.",
    CL(), P("γ chica → suave (subajuste) · γ grande → islas (sobreajuste)", 960, 300, "violeta", 44, en="a"),
    P("C y γ grandes → más riesgo de memorizar", 960, 460, "rojo", 46, en="b"),
    P("se eligen con validación cruzada (clase: C = 10, γ = 0.1)", 960, 620, "teal", 44, en="c"), sec="8. Para llevar")
v.b("Siguiente parada: [[a]]por qué hay que estandarizar antes de una SVM, y un clasificador clásico que depende todavía más de las distancias: los {k|k,} vecinos más cercanos.",
    CL(), siguiente(9, "Estandarizar: SVM y kNN"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
