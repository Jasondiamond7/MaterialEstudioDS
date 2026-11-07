import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, nube, serie_curva, recta_caja, fila)

OUT = out("guion_c37_05_margen.json")
v = V("SVM: hiperplano, margen y vectores de soporte")
DU = D["duro"]
P_ = D["duro_pts"]
YF = D["duro_yf"]
W, B = DU["w"], DU["b"]
CAJA = (-1.6, 3.2, -1.3, 3.2)
G0 = (400, 210, 1260, 900)


def lineas_margen():
    out_ = []
    for niv, col, g in [(0.0, "tinta", 6), (1.0, "rojo", 3), (-1.0, "azul", 3)]:
        xs, ys = recta_caja(W, B, niv, *CAJA)
        out_.append(serie_curva(xs, ys, col, g))
    return out_


def graf(id_, en=None, t=None, extra=None):
    series = nube((P_["x"], P_["y"]), P_["c"], mk=9) + (extra or [])
    return G(*G0, *CAJA, series, xticks=[-1, 0, 1, 2, 3], yticks=[-1, 0, 1, 2, 3], yfmt="{:.0f}", xlabel="radio (estandarizado)",
             ylabel="concav.", id=id_, en=en, t=t, tam_txt=28)


v.portada("SVM: el margen", "Una frontera con la calle más ancha posible",
          "Quinta parada: las máquinas de vectores de soporte. Vas a calcular un puntaje, una distancia y el ancho del margen, y a decidir qué puntos son vectores de soporte.")

# ---- 1. Hook: infinitas rectas
CAND = [((1.0, 0.25), 0.1), ((0.6, 1.0), 0.2), ((1.0, 1.6), -0.6)]
v.b("Mira estos dos grupos: treinta y dos casos reales, benignos en azul y malignos en rojo. [[a]]Hay infinitas rectas que los separan sin errores. Esta, [[b]]esta otra, [[c]]o esta. ¿Cuál elegir?",
    CL(), T("¿cuál recta elegir?", 960, 140, tam=54, c="tinta", ent="subir"),
    graf("g1", extra=[serie_curva(*recta_caja(w_, b_, 0.0, *CAJA), "grisclaro", 5) for w_, b_ in CAND]),
    U("g1", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.3, dur=1.4),
    U("g1", **{"series.2.alfa": 1.0}, en="a", dur=0.5), U("g1", **{"series.3.alfa": 1.0}, en="b", dur=0.5), U("g1", **{"series.4.alfa": 1.0}, en="c", dur=0.5),
    T("subconjunto separable de 32 casos de la base", 1860, 1050, tam=26, c="teal", w="sb", anc="rm", t=0.6), sec="1. Infinitas rectas")
v.b("La SVM tiene una respuesta precisa: [[a]]la recta que deja la calle más ancha posible entre los dos grupos.",
    U("g1", **{"series.2.alfa": 0.25, "series.3.alfa": 0.25, "series.4.alfa": 0.25}, dur=0.5),
    P("la calle MÁS ANCHA", 1580, 450, "ambar", 40, en="a"))

# ---- 2. Analogía: la calle entre dos barrios
v.b("Piensa en una calle nueva entre dos barrios. [[a]]Quieres que sea lo más ancha posible, sin tocar ninguna casa. [[b]]¿Qué casas deciden dónde va la calle? Solo las que quedan en la vereda, las más cercanas al otro barrio. [[c]]Las casas del fondo podrían moverse, y la calle no cambiaría.",
    CL(), T("la calle entre dos barrios", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "la calle", "lo más ancha posible, sin tocar casas", "a"), (960, "ambar", "la vereda", "solo esas casas deciden dónde va", "b"),
              (1520, "gris", "el fondo", "moverlas no cambia nada", "c")], y=500, w=480, h=330, tam_t=50, tam_d=36),
    P("casas de la vereda = vectores de soporte", 960, 820, "violeta", 48, en="c", t=1.2), ilustrativo(1000, texto="analogía"), sec="2. Analogía: la calle")

# ---- 3. El hiperplano
v.b("Primero, la frontera. La diapositiva treinta y dos la escribe así: [[a]]{f(x)|efe de equis} es igual a w por x más be, y la frontera es donde {f(x)|efe de equis} vale cero. [[b]]El vector w fija la orientación; be desplaza la frontera. [[c]]Y el signo de {f(x)|efe de equis} decide la clase: positivo, más uno; negativo, menos uno.",
    CL(), T("hiperplano y función de decisión · diapositiva 32", 960, 150, tam=48, c="tinta", ent="subir"),
    caja_formula("f(x) = w·x + b        frontera: f(x) = 0", y=310, c="violeta", tam=54, w=1400, en="a"),
    P("w: orientación", 700, 480, "azul", 48, en="b"), P("b: desplazamiento", 1220, 480, "teal", 48, en="b", t=0.3),
    P("f(x) ≥ 0 → +1      f(x) < 0 → −1", 960, 640, "rojo", 50, en="c"), sec="3. El hiperplano")
v.b("Un detalle de notación: [[a]]para explicar la geometría usamos las etiquetas menos uno y más uno. En los scripts se mantienen cero y uno. [[b]]Y la distancia de un punto a la frontera es el valor absoluto de {f(x)|efe de equis}, dividido por la norma de w.",
    T("geometría: −1 / +1  ·  scripts: 0 / 1 (diapositiva 31)", 960, 800, tam=38, c="gris", w="sb", en="a"),
    P("distancia a la frontera = |f(x)| / ‖w‖", 960, 910, "ambar", 50, en="b"))

# ---- 4. Cuenta resuelta (diapositiva 35)
v.b("Hagamos la cuenta de la diapositiva treinta y cinco. [[a]]Supón que w es uno, menos uno, y be vale menos 0.5. [[b]]Tomamos el punto x igual a dos, uno.",
    CL(), T("cuenta resuelta · diapositiva 35", 960, 150, tam=50, c="tinta", ent="subir"),
    P("w = (1, −1)      b = −0.5", 760, 290, "violeta", 50, en="a"), P("x = (2, 1)", 1360, 290, "azul", 50, en="b"),
    ilustrativo(1000, texto="ejemplo de la diapositiva 35"), sec="4. Cuenta resuelta")
v.b("Paso uno, el puntaje: [[a]]uno por dos, menos uno por uno, menos 0.5. Dos menos uno menos 0.5: [[b]]0.5. [[c]]Paso dos, la clase: como {f(x)|efe de equis} es positivo, predecimos más uno.",
    T("f(x) = 1·2 + (−1)·1 − 0.5 = 2 − 1 − 0.5 =", 1180, 430, tam=46, c="tinta", w="sb", anc="rm", en="a"), N(0.5, 1210, 430, "{:.1f}", 58, "violeta", anc="lm", en="b"),
    T("f(x) > 0  →  clase +1", 960, 540, tam=46, c="rojo", w="sb", en="c"))
v.b("Paso tres, la norma de w: [[a]]la raíz de uno al cuadrado más menos uno al cuadrado: raíz de dos, 1.414. [[b]]Paso cuatro, la distancia: 0.5 sobre 1.414, [[c]]0.354. Fíjate que la distancia no es 0.5: el puntaje hay que dividirlo por la norma.",
    T("‖w‖ = √(1² + (−1)²) = √2 =", 1180, 650, tam=46, c="tinta", w="sb", anc="rm", en="a"), N(1.414, 1210, 650, "{:.3f}", 58, "azul", anc="lm", en="a", t=0.8),
    T("distancia = 0.5 / 1.414 =", 1180, 770, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(0.354, 1210, 770, "{:.3f}", 58, "ambar", anc="lm", en="c"),
    P("la distancia NO es f(x): se divide por ‖w‖", 960, 900, "ambar", 44, en="c", t=1.2))

# ---- 5. El margen
v.b("Ahora, el margen. [[a]]La SVM fija la escala de w y be para que los puntos cumplan {y·f(x)|ye por efe de equis} mayor o igual que uno. [[b]]Así aparecen dos bordes: efe igual a más uno, y efe igual a menos uno. [[c]]Cada borde está a uno sobre la norma de w de la frontera. El ancho total de la calle es dos sobre la norma de w.",
    CL(), T("margen máximo: caso separable · diapositiva 34", 960, 150, tam=48, c="tinta", ent="subir"),
    P("se exige  yᵢ·f(xᵢ) ≥ 1", 960, 290, "violeta", 50, en="a"),
    P("bordes: f(x) = +1  y  f(x) = −1", 960, 410, "azul", 48, en="b"),
    caja_formula("ancho del margen = 2 / ‖w‖", y=570, c="ambar", tam=60, w=1000, en="c"), sec="5. El margen")
v.b("Por eso la SVM minimiza un medio de la norma de w al cuadrado: [[a]]achicar la norma es ensanchar la calle. [[b]]En el ejemplo anterior, con norma raíz de dos, el ancho sería dos sobre 1.414: 1.414.",
    P("mín ½‖w‖²  →  calle más ancha", 960, 740, "violeta", 50, en="a"),
    T("ejemplo: 2 / √2 = 2 / 1.414 = 1.414", 960, 860, tam=44, c="tinta", w="sb", en="b"))

# ---- 6. El margen con datos reales
v.b("Volvamos a los treinta y dos casos reales. [[a]]La SVM de margen duro encuentra esta frontera, la línea negra. [[b]]Y estos son sus dos bordes: efe igual a más uno, en rojo, y menos uno, en azul. [[c]]Ningún caso queda dentro de la calle.",
    CL(), T("margen duro con 32 casos reales", 960, 140, tam=50, c="tinta", ent="subir"),
    graf("g2", extra=lineas_margen()), U("g2", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.2, dur=1.0),
    U("g2", **{"series.2.alfa": 1.0}, en="a", dur=0.6), U("g2", **{"series.3.alfa": 1.0, "series.4.alfa": 1.0}, en="b", dur=0.6),
    T("f = 0", 1420, 330, tam=36, c="tinta", w="sb", anc="lm", en="a"), T("f = +1", 1420, 390, tam=36, c="rojo", w="sb", anc="lm", en="b"),
    T("f = −1", 1420, 450, tam=36, c="azul", w="sb", anc="lm", en="b", t=0.2),
    T("subconjunto separable de 32 casos de la base", 1860, 1050, tam=26, c="teal", w="sb", anc="rm"), sec="6. El margen con datos reales")
v.b("Los números del ajuste: [[a]]w vale 1.316 y 1.071, y be, 0.082. [[b]]La norma de w es la raíz de 1.316 al cuadrado más 1.071 al cuadrado: 1.697. [[c]]Y el ancho del margen: dos sobre 1.697, 1.179 desviaciones estándar.",
    T("w = (1.316, 1.071)", 1420, 560, tam=38, c="violeta", w="sb", anc="lm", en="a"), T("b = 0.082", 1420, 615, tam=38, c="violeta", w="sb", anc="lm", en="a", t=0.3),
    T("‖w‖ = 1.697", 1420, 700, tam=38, c="azul", w="sb", anc="lm", en="b"),
    T("ancho = 2 / 1.697 =", 1420, 790, tam=38, c="tinta", w="sb", anc="lm", en="c"), N(DU["ancho"], 1420, 860, "{:.3f}", 60, "ambar", anc="lm", en="c", t=0.5))

# ---- 7. ¿Quiénes son los vectores de soporte?
SV = [YF[0], YF[1], YF[3]]
v.b("¿Quiénes son los vectores de soporte? [[a]]Los puntos que quedan justo sobre los bordes: los que cumplen {y·f(x)|ye por efe de equis} igual a uno. [[b]]Comprobémoslo con un maligno, en zeta igual a 0.54 y 0.194. [[c]]Efe vale 1.316 por 0.54, más 1.071 por 0.194, más 0.082: 0.711 más 0.208 más 0.082, igual a 1.000. Está sobre el borde: es vector de soporte.",
    CL(), T("¿quiénes sostienen la calle?", 960, 140, tam=50, c="tinta", ent="subir"),
    graf("g3", extra=lineas_margen()), U("g3", **{"series.0.revela": 1.0, "series.1.revela": 1.0, "series.2.alfa": 1.0, "series.3.alfa": 1.0, "series.4.alfa": 1.0}, dur=0.6),
    *[R(P_["x"][i], P_["y"][i], 46, 46, "ambar", g=5, ref="g3", en="a", t=0.2 * k) for k, i in enumerate(P_["sv"])],
    P("vectores de soporte: y·f(x) = 1", 1590, 300, "ambar", 36, en="a"),
    T("maligno z = (0.54, 0.194):", 1590, 430, tam=34, c="tinta", w="sb", en="b"),
    T("f = 1.316·0.54 + 1.071·0.194 + 0.082", 1590, 500, tam=30, c="tinta", w="sb", en="c"),
    T("= 0.711 + 0.208 + 0.082 = 1.000", 1590, 560, tam=34, c="rojo", w="sb", en="c", t=0.8), sec="7. ¿Quiénes son los vectores de soporte?")
v.b("Ahora un benigno cercano, en menos 0.678 y menos 0.545. [[a]]Efe vale menos 1.394, y como su etiqueta es menos uno, {y·f|ye por efe} da 1.394. [[b]]Es mayor que uno: está más allá del borde, y no es vector de soporte. [[c]]De los treinta y dos casos, solo tres quedan exactamente en el borde. Si borras cualquiera de los otros veintinueve, la frontera no se mueve.",
    R(-0.678, -0.545, 46, 46, "gris", g=4, pulso=False, ref="g3", en="a"),
    T("benigno z = (−0.678, −0.545):", 1590, 660, tam=34, c="tinta", w="sb"),
    T("f = −1.394  →  y·f = 1.394 > 1", 1590, 720, tam=34, c="azul", w="sb", en="a"),
    T("no es vector de soporte", 1590, 780, tam=34, c="gris", w="sb", en="b"),
    P("3 de 32 sostienen la frontera", 1590, 880, "violeta", 40, en="c"),
    T("subconjunto separable de 32 casos de la base", 1860, 1050, tam=26, c="teal", w="sb", anc="rm"))

# ---- 8. El error típico
v.b("El error típico: [[a]]creer que un margen ancho garantiza un buen modelo. [[b]]La diapositiva treinta y cuatro lo advierte: un margen amplio no garantiza, por sí solo, buen rendimiento en datos nuevos. [[c]]Y estos treinta y dos casos se eligieron separables a propósito. Con los 398 casos reales, las clases se mezclan, y el margen duro ni siquiera existe. Eso lo arregla el margen suave.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("«margen ancho = buen modelo»", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("diapositiva 34: un margen amplio no garantiza buen rendimiento en datos nuevos", 960, 400, tam=38, c="tinta", w="sb", en="b"),
    P("con los 398 casos las clases se mezclan: no hay margen duro", 960, 560, "ambar", 44, en="c"),
    T("→ margen suave (siguiente video)", 960, 680, tam=40, c="gris", w="sb", en="c", t=1.0), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si w es tres, cuatro, be vale menos cinco, y el punto es x igual a dos, uno, [[b]]¿cuánto vale {f(x)|efe de equis}? ¿A qué distancia de la frontera está? ¿Y cuál es el ancho del margen?",
    CL(), P("w = (3, 4)   ·   b = −5   ·   x = (2, 1)", 960, 300, "violeta", 52, en="a"),
    P("f(x) = ?", 560, 470, "azul", 50, en="b"), P("distancia = ?", 960, 470, "ambar", 50, en="b", t=0.3), P("ancho = ?", 1360, 470, "teal", 50, en="b", t=0.6),
    ilustrativo(1000), sec="9. Piensa un segundo", pausa=0.6)
v.b("Efe de x: tres por dos más cuatro por uno menos cinco, [[a]]cinco. [[b]]La norma de w: raíz de nueve más dieciséis, raíz de veinticinco, cinco. La distancia: cinco sobre cinco, uno. [[c]]Y el ancho del margen: dos sobre cinco, 0.4.",
    CL(), T("f(x) = 3·2 + 4·1 − 5 =", 1000, 300, tam=50, c="tinta", w="sb", anc="rm"), N(5, 1030, 300, "{:.0f}", 62, "azul", anc="lm", en="a"),
    T("‖w‖ = √(9 + 16) = 5   →   distancia = 5 / 5 =", 1000, 430, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(1, 1030, 430, "{:.0f}", 62, "ambar", anc="lm", en="b", t=1.0),
    T("ancho = 2 / 5 =", 1000, 560, tam=50, c="tinta", w="sb", anc="rm", en="c"), N(0.4, 1030, 560, "{:.1f}", 62, "teal", anc="lm", en="c", t=0.4),
    ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]la SVM usa la frontera {f(x)|efe de equis} igual a cero, y la distancia es el valor absoluto de efe sobre la norma de w. [[b]]El ancho del margen es dos sobre la norma de w, y la SVM lo maximiza. [[c]]Y solo los vectores de soporte, los puntos con {y·f|ye por efe} igual a uno, deciden dónde va la frontera.",
    CL(), P("distancia = |f(x)| / ‖w‖", 960, 300, "ambar", 50, en="a"), P("ancho = 2 / ‖w‖ → se maximiza", 960, 460, "violeta", 50, en="b"),
    P("vectores de soporte: y·f(x) = 1 → deciden la frontera", 960, 620, "teal", 46, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]el margen suave. Qué pasa cuando las clases se mezclan, la holgura, y el parámetro C.",
    CL(), siguiente(6, "Margen suave, holgura y el parámetro C"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
