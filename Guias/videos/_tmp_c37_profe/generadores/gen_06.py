import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, nube, serie_curva, recta_caja, fila, barras_h)

OUT = out("guion_c37_06_suave.json")
v = V("Margen suave, holgura y el parámetro C")
Z = D["z2_train"]
L2 = D["lin2"]
HC = FG["holgura_clase"]
C4 = FG["C4d"]
PE = D["perdidas"]
CAJA = (-2.1, 4.0, -1.5, 4.4)
G0 = (400, 220, 1300, 920)


def margen_series(w, b, alfa=0.0):
    out_ = []
    for niv, col, g in [(0.0, "tinta", 6), (1.0, "rojo", 3), (-1.0, "azul", 3)]:
        xs, ys = recta_caja(w, b, niv, *CAJA)
        out_.append(serie_curva(xs, ys, col, g, alfa=alfa))
    return out_


def nube398(id_, extra=None, en=None):
    return G(*G0, *CAJA, nube((Z["x"], Z["y"]), Z["c"], mk=6) + (extra or []), xticks=[-2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3, 4],
             yfmt="{:.0f}", xlabel="radio (estandarizado)", ylabel="concav.", id=id_, en=en, tam_txt=28)


v.portada("Margen suave y el parámetro C", "Cuando las clases se mezclan: holgura, hinge loss y cuánto castigar",
          "Sexta parada: el margen suave. Vas a calcular holguras a mano, a ver la pérdida hinge, y a mirar qué hace el parámetro C con los datos reales.")

# ---- 1. Hook
v.b("Estos son los 398 casos de entrenamiento, con radio y concavidad estandarizados. [[a]]Mira la zona del medio: benignos y malignos se mezclan. [[b]]No existe ninguna recta que los separe sin errores. [[c]]¿La SVM se rinde? No. Deja que algunos puntos crucen, pero les cobra por cruzar.",
    CL(), T("los 398 casos reales se mezclan", 960, 140, tam=50, c="tinta", ent="subir"),
    nube398("g1"), U("g1", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.3, dur=1.6),
    R(0.3, 0.0, 300, 240, "ambar", r=30, ref="g1", en="a", id="zona"),
    P("ninguna recta separa sin errores", 1610, 420, "rojo", 36, en="b"), P("cruzar se permite… pero se cobra", 1610, 560, "ambar", 36, en="c"),
    real(1050, en="a"), sec="1. Las clases se mezclan")

# ---- 2. Analogía: las multas
v.b("Piensa en un estacionamiento con líneas pintadas. [[a]]Puedes pasarte un poco de la línea, pero te ponen una multa proporcional a cuánto te pasaste. [[b]]Si la multa es barata, mucha gente se pasa. [[c]]Si es carísima, todos se esfuerzan por no tocar la línea. En la SVM, el valor de la multa se llama C.",
    CL(), T("el estacionamiento con multas", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "pasarse un poco", "multa según cuánto te pasaste", "a"), (960, "verde", "multa barata", "muchos se pasan", "b"),
              (1520, "rojo", "multa carísima", "nadie toca la línea", "c")], y=500, w=480, h=330, tam_t=46, tam_d=36),
    P("el valor de la multa = C", 960, 820, "ambar", 54, en="c", t=1.2), ilustrativo(1000, texto="analogía"), sec="2. Analogía: las multas")

# ---- 3. La holgura
v.b("La diapositiva treinta y seis lo escribe así. [[a]]Se minimiza un medio de la norma de w al cuadrado, más C por la suma de las holguras: la letra ksi. [[b]]La condición se relaja: {y·f(x)|ye por efe de equis} ya no tiene que ser mayor que uno, sino mayor que uno menos la holgura ksi. [[c]]Y en la solución, cada holgura vale el máximo entre cero y uno menos {y·f|ye por efe}.",
    CL(), T("margen suave · diapositiva 36", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("mín  ½‖w‖² + C · Σ ξᵢ", y=310, c="violeta", tam=58, w=1000, en="a", wf="sb"),
    P("yᵢ·f(xᵢ) ≥ 1 − ξᵢ ,   ξᵢ ≥ 0", 960, 470, "azul", 50, en="b"),
    caja_formula("ξᵢ = máx(0, 1 − yᵢ·f(xᵢ))", y=640, c="ambar", tam=56, w=1000, en="c", wf="sb"), sec="3. La holgura")
v.b("En inglés esto se llama soft margin, margen suave. [[a]]La holgura mide cuánto se pasó cada punto del borde de su lado.",
    T("soft margin = margen suave", 960, 820, tam=44, c="gris", w="sb", en="a"), T("ξ = cuánto se pasó del borde de su lado", 960, 900, tam=40, c="tinta", w="sb", en="a", t=0.5))

# ---- 4. Holgura a mano (tabla de la diapositiva 36)
FIL = [("1.4", "0", "correcto y fuera del margen", "verde"), ("0.5", "0.5", "correcto, pero dentro del margen", "ambar"),
       ("−0.3", "1.3", "mal clasificado", "rojo")]
v.b("Calculemos tres holguras, con la tabla de la diapositiva. [[a]]Un punto con {y·f|ye por efe} igual a 1.4: uno menos 1.4 es negativo, así que la holgura ksi es cero. Correcto, y fuera del margen. [[b]]Con {y·f|ye por efe} igual a 0.5: ksi es 0.5. Está bien clasificado, pero dentro de la calle. [[c]]Y con menos 0.3: la holgura ksi es uno más 0.3, igual a 1.3. Está mal clasificado.",
    CL(), T("holgura a mano · diapositiva 36", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(270, [("y·f(x)", 500), ("ξ = máx(0, 1 − y·f)", 900), ("lectura", 1400)], tams=[38, 38, 38], colores=["gris"] * 3),
    *[fila(380 + 120 * i, [(a, 500), (b, 900), (c, 1400)], tams=[48, 52, 38], colores=["tinta", col, col], en=k, fondo=col + "claro", x0=300, x1=1720, h=96)
      for i, ((a, b, c, col), k) in enumerate(zip(FIL, ["a", "b", "c"]))],
    ilustrativo(1000, texto="ejemplo de la diapositiva 36"), sec="4. Holgura a mano")
v.b("Fíjate en el segundo caso. [[a]]Está bien clasificado, y aun así paga holgura. [[b]]La holgura penaliza las violaciones del margen, no solo los errores de clasificación.",
    R(960, 500, 1440, 110, "ambar", r=24, en="a"), P("la holgura castiga violar el MARGEN, no solo equivocarse", 960, 850, "ambar", 44, en="b"))

# ---- 5. La pérdida hinge
YF = [-2 + 0.05 * i for i in range(101)]
v.b("Esa función, el máximo entre cero y uno menos {y·f|ye por efe}, tiene nombre propio: [[a]]es la pérdida hinge, la pérdida bisagra, de la diapositiva sesenta y seis. [[b]]Vale cero cuando el caso queda más allá del margen correcto, y crece en línea recta cuando se acerca o cruza. [[c]]A su lado, la pérdida logística: logaritmo de uno más e a la menos {y·f|ye por efe}.",
    CL(), T("pérdida hinge frente a pérdida logística · diapositiva 66", 960, 140, tam=46, c="tinta", ent="subir"),
    G(380, 240, 1380, 880, -2, 3, 0, 3, [dict(xs=[-2, 1, 3], ys=[3, 0, 0], color="violeta", grosor=7, revela=0.0),
                                         dict(xs=YF, ys=[math.log(1 + math.exp(-x)) for x in YF], color="teal", grosor=6, revela=0.0)],
      xticks=[-2, -1, 0, 1, 2, 3], yticks=[0, 1, 2, 3], yfmt="{:.0f}", xlabel="y·f(x)", ylabel="pérdida", id="gh", en="a"),
    U("gh", **{"series.0.revela": 1.0}, en="a", t=0.3, dur=1.6), U("gh", **{"series.1.revela": 1.0}, en="c", t=0.2, dur=1.6),
    P("hinge = máx(0, 1 − y·f)", 1640, 380, "violeta", 34, en="a"), P("logística = log(1 + e^(−y·f))", 1640, 500, "teal", 32, en="c"),
    sec="5. La pérdida hinge")
v.b("Comparemos en los tres puntos de la tabla. [[a]]Con 1.4: hinge cero, logística 0.220. [[b]]Con 0.5: hinge 0.5, logística 0.474. [[c]]Con menos 0.3: hinge 1.3, logística 0.854. La diferencia clave: la hinge se apaga del todo; la logística nunca llega a cero.",
    T(f"y·f = 1.4:  hinge 0  ·  log {PE['1.4']['log']:.3f}", 1640, 640, tam=30, c="tinta", w="sb", en="a"),
    T(f"y·f = 0.5:  hinge 0.5  ·  log {PE['0.5']['log']:.3f}", 1640, 700, tam=30, c="tinta", w="sb", en="b"),
    T(f"y·f = −0.3:  hinge 1.3  ·  log {PE['-0.3']['log']:.3f}", 1640, 760, tam=30, c="tinta", w="sb", en="c"),
    P("hinge se apaga · logística nunca es 0", 960, 990, "ambar", 40, en="c", t=1.5))
v.b("Y de ahí sale lo más importante de la SVM. [[a]]Los puntos lejos del margen, del lado correcto, tienen pérdida cero: no influyen en nada. [[b]]Solo los que están en el borde, dentro de la calle o del lado equivocado, mueven la frontera. Esos son los vectores de soporte. [[c]]En la logística, en cambio, todos los puntos aportan algo.",
    CL(), P("lejos y del lado correcto → pérdida 0 → no influyen", 960, 330, "verde", 46, en="a"),
    P("borde, dentro o del lado equivocado → vectores de soporte", 960, 490, "violeta", 44, en="b"),
    P("logística: todos los puntos aportan", 960, 650, "teal", 46, en="c"))

# ---- 6. El parámetro C
v.b("Ahora, el parámetro C. Según la diapositiva treinta y siete: [[a]]un C pequeño le da poco peso a las violaciones: el modelo se regulariza más, y la calle se ensancha. [[b]]Un C grande castiga mucho las violaciones del entrenamiento: la calle se angosta. [[c]]Un ejemplo: si las holguras suman cuatro, con C igual a 0.1 la multa total es 0.4; con C igual a diez, es cuarenta.",
    CL(), T("el parámetro C · diapositiva 37", 960, 150, tam=50, c="tinta", ent="subir"),
    P("C pequeño → poco peso a las violaciones → calle ancha", 960, 300, "verde", 44, en="a"),
    P("C grande → castiga las violaciones → calle angosta", 960, 430, "rojo", 44, en="b"),
    T("Σ ξ = 4:   C = 0.1 → 0.1 × 4 =", 960, 590, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(0.4, 990, 590, "{:.1f}", 58, "verde", anc="lm", en="c", t=0.6),
    T("C = 10 → 10 × 4 =", 960, 700, tam=46, c="tinta", w="sb", anc="rm", en="c", t=1.6), N(40, 990, 700, "{:.0f}", 58, "rojo", anc="lm", en="c", t=2.2),
    ilustrativo(1000, texto="ejemplo de la diapositiva 37"), sec="6. El parámetro C")

# ---- 7. C con datos reales
EX = margen_series(L2["0.01"]["w"], L2["0.01"]["b"]) + margen_series(L2["1"]["w"], L2["1"]["b"]) + margen_series(L2["100"]["w"], L2["100"]["b"])
v.b("Veámoslo con los datos reales, una SVM lineal con dos variables. [[a]]Con C igual a 0.01, la calle es ancha: 2.24 desviaciones estándar, y 188 vectores de soporte. [[b]]Con C igual a uno, se angosta a 0.86, con 83 vectores de soporte. [[c]]Y con C igual a cien, apenas cambia: 0.81 de ancho y 78 vectores de soporte.",
    CL(), T("SVM lineal con 2 variables: tres valores de C", 960, 140, tam=48, c="tinta", ent="subir"),
    nube398("g2", extra=EX), U("g2", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.2, dur=1.0),
    U("g2", **{"series.2.alfa": 1.0, "series.3.alfa": 1.0, "series.4.alfa": 1.0}, en="a", dur=0.6),
    T("C = 0.01:  ancho 2.24 · 188 SV", 1600, 380, tam=34, c="verde", w="sb", en="a", t=0.3),
    U("g2", **{"series.2.alfa": 0.0, "series.3.alfa": 0.0, "series.4.alfa": 0.0, "series.5.alfa": 1.0, "series.6.alfa": 1.0, "series.7.alfa": 1.0}, en="b", dur=0.8),
    T("C = 1:  ancho 0.86 · 83 SV", 1600, 450, tam=34, c="azul", w="sb", en="b", t=0.3),
    U("g2", **{"series.5.alfa": 0.0, "series.6.alfa": 0.0, "series.7.alfa": 0.0, "series.8.alfa": 1.0, "series.9.alfa": 1.0, "series.10.alfa": 1.0}, en="c", dur=0.8),
    T("C = 100:  ancho 0.81 · 78 SV", 1600, 520, tam=34, c="rojo", w="sb", en="c", t=0.3),
    real(1050, en="a"), sec="7. C con datos reales")
v.b("¿Y en la prueba? [[a]]Los tres modelos aciertan exactamente lo mismo: 88.3 por ciento. [[b]]Cambiar {C|, ce,} movió la calle, pero no mejoró nada. Como dice la diapositiva: un C mayor no siempre mejora el modelo; lo decide la validación.",
    P("prueba: 88.3 % con los tres", 1600, 660, "teal", 40, en="a"), T("un C mayor no siempre mejora", 1600, 750, tam=34, c="gris", w="sb", en="b"),
    T("lo decide la validación", 1600, 810, tam=34, c="gris", w="sb", en="b", t=0.4))

# ---- 8. El modelo de la clase
v.b("Ahora, el modelo de la clase: SVM lineal con las cuatro variables y C igual a 0.1. [[a]]De los 398 casos de entrenamiento, 308 quedan bien clasificados y fuera del margen: holgura cero. [[b]]70 quedan dentro de la calle, [[c]]y 20 quedan del lado equivocado.",
    CL(), T("el modelo de la clase: lineal, 4 variables, C = 0.1", 960, 150, tam=48, c="tinta", ent="subir"),
    barras_h([("fuera del margen (ξ = 0)", HC["fuera"], "verde", "a"), ("dentro del margen", HC["dentro"], "ambar", "b"),
              ("mal clasificados", HC["mal"], "rojo", "c")], x0=760, y0=330, paso=150, escala=2.6, fmt="{:.0f}", tam=40, h=80, xl=720),
    real(1050, en="a"), sec="8. El modelo de la clase")
v.b("¿Cuántos vectores de soporte tiene? [[a]]93: los que quedan en el borde, dentro de la calle o del lado equivocado. 47 benignos y 46 malignos. [[b]]Los otros 305 casos podrían desaparecer, y la frontera sería la misma.",
    P("93 vectores de soporte  (47 + 46)", 960, 820, "violeta", 50, en="a"), T("los otros 305 casos no mueven la frontera", 960, 920, tam=40, c="gris", w="sb", en="b"))

# ---- 9. ¿Qué C elegir?
CX = [math.log10(c) for c in C4["C"]]
v.b("¿Y cómo se elige C? Con validación cruzada, solo en el entrenamiento. [[a]]Esta curva prueba trece valores de C, de 0.001 a mil, con las cuatro variables. [[b]]Con C muy chico, el modelo subajusta: 66 por ciento. [[c]]El máximo de la validación cruzada está justo en C igual a 0.1: 94.0 por ciento. Después la curva se aplana.",
    CL(), T("validación cruzada (5 folds) en entrenamiento: SVM lineal, 4 variables", 960, 140, tam=44, c="tinta", ent="subir"),
    G(540, 240, 1400, 860, -3, 3, 0.6, 1.0, [dict(xs=CX, ys=C4["acc_train"], color="gris", grosor=4, marcadores=7, revela=0.0),
                                             dict(xs=CX, ys=C4["acc_cv"], color="azul", grosor=6, marcadores=8, revela=0.0)],
      xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[0.6, 0.7, 0.8, 0.9, 1.0], yfmt="{:.1f}", xlabel="log₁₀ C", ylabel="exact.", id="gc", en="a"),
    U("gc", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="a", t=0.3, dur=2.0),
    T("entrenamiento", 1620, 300, tam=34, c="gris", w="sb", en="a"), T("validación cruzada", 1620, 360, tam=34, c="azul", w="sb", en="a", t=0.3),
    R(-3, C4["acc_cv"][0], 60, 60, "rojo", ref="gc", en="b"), T(f"C = 0.001: {100 * C4['acc_cv'][0]:.1f} %", 1620, 500, tam=34, c="rojo", w="sb", en="b"),
    R(-1, C4["acc_cv"][4], 60, 60, "verde", ref="gc", en="c"), T(f"mejor: C = 0.1 → {100 * C4['mejor_acc_cv']:.1f} %", 1620, 600, tam=34, c="verde", w="sb", en="c"),
    real(1050, en="a"), sec="9. ¿Qué C elegir?")

# ---- 10. El error típico
v.b("El error típico: [[a]]leer el parámetro C como si fuera la fuerza de la regularización, al estilo de la lambda de Lasso. [[b]]Es al revés: un C grande significa menos regularización, porque las violaciones pesan más. [[c]]Y en R, el mismo parámetro se llama cost.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("«C grande = más regularización»", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    P("al revés: C grande → MENOS regularización", 960, 420, "verde", 50, en="b"),
    P("en R se llama cost: svm(…, cost = 0.1)", 960, 580, "teal", 46, en="c"), sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tres puntos tienen {y·f|ye por efe} igual a dos, cero, y menos 1.5. [[b]]¿Cuánto vale cada holgura? ¿Y cuánto suma la multa con C igual a dos?",
    CL(), P("y·f = 2   ·   y·f = 0   ·   y·f = −1.5", 960, 330, "azul", 52, en="a"), P("ξ = ?", 760, 500, "ambar", 52, en="b"),
    P("C · Σξ con C = 2 = ?", 1200, 500, "rojo", 48, en="b", t=0.4), ilustrativo(1000), sec="11. Piensa un segundo", pausa=0.6)
v.b("Con dos: cero, porque está más allá del margen. [[a]]Con cero: uno, justo en la frontera. [[b]]Con menos 1.5: 2.5. [[c]]La suma es 3.5, y con C igual a dos la multa es siete.",
    CL(), T("y·f = 2  →  ξ = máx(0, −1) = 0", 960, 280, tam=46, c="verde", w="sb"),
    T("y·f = 0  →  ξ = 1", 960, 390, tam=46, c="ambar", w="sb", en="a"), T("y·f = −1.5  →  ξ = 2.5", 960, 500, tam=46, c="rojo", w="sb", en="b"),
    T("Σξ = 3.5   →   C · Σξ = 2 × 3.5 =", 940, 640, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(7, 970, 640, "{:.0f}", 62, "rojo", anc="lm", en="c", t=0.8),
    ilustrativo(1000))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]la holgura es el máximo entre cero y uno menos {y·f|ye por efe}, la pérdida hinge, y castiga violar el margen. [[b]]C es el precio de cada violación: chico, calle ancha; grande, calle angosta. [[c]]Y C se elige con validación cruzada en entrenamiento: con el kernel lineal de la clase, el mejor es 0.1.",
    CL(), P("ξ = máx(0, 1 − y·f) = pérdida hinge", 960, 300, "ambar", 48, en="a"),
    P("C chico → calle ancha · C grande → calle angosta", 960, 460, "violeta", 46, en="b"),
    P("C se elige con validación cruzada (lineal: C = 0.1)", 960, 620, "teal", 46, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]los kernels. Cómo una SVM puede trazar fronteras curvas sin dejar de ser lineal por dentro.",
    CL(), siguiente(7, "Kernels y el kernel radial a mano"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
