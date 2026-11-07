import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, nube, serie_curva, serie_puntos, recta_caja, fila, codigo)

OUT = out("guion_c37_07_kernels.json")
v = V("Kernels y el kernel radial a mano")
Z = D["z2_train"]
K2 = FG["kernels2d"]
KD = FG["k_dist2"]
CAJA = (-2.1, 4.0, -1.5, 4.4)

v.portada("Kernels", "Fronteras curvas con un truco: medir la similitud",
          "Séptima parada: los kernels. Vas a ver por qué un problema que no se separa con una recta se separa al subirlo de dimensión, y vas a calcular el kernel radial a mano.")

# ---- 1. Hook (1D -> 2D)
AZ = [-2.0, -1.6, 1.6, 2.0]
RO = [-0.6, -0.2, 0.3, 0.7]
v.b("Mira estos puntos sobre una recta: rojos en el centro, azules en los extremos. [[a]]¿Puedes separarlos con un solo corte? [[b]]No: cualquier umbral deja azules de los dos lados. [[c]]Una frontera lineal, aquí, no sirve.",
    CL(), T("¿un solo corte?", 960, 150, tam=54, c="tinta", ent="subir"),
    L(260, 520, 1660, 520, "gris", 4), *[Bc(960 + 330 * x, 520, 40, 40, "azul", None, 0, 20, t=0.1 * i) for i, x in enumerate(AZ)],
    *[Bc(960 + 330 * x, 520, 40, 40, "rojo", None, 0, 20, t=0.4 + 0.1 * i) for i, x in enumerate(RO)],
    L(960 + 330 * 1.15, 380, 960 + 330 * 1.15, 660, "ambar", 6, en="a"), T("?", 960 + 330 * 1.15, 340, tam=60, c="ambar", en="a"),
    P("cualquier corte deja azules a ambos lados", 960, 760, "rojo", 46, en="b"), ilustrativo(1000), sec="1. Un corte no alcanza")

# ---- 2. Analogía: levantar las monedas
v.b("Piensa en monedas sobre una mesa: las rojas al centro, rodeadas de azules. [[a]]Sobre la mesa, ninguna línea recta las separa. [[b]]Pero si levantas las monedas del centro, si les agregas una altura, [[c]]una hoja horizontal pasa entre las dos alturas y las separa. Subir a otra dimensión vuelve lineal lo que no lo era.",
    CL(), T("las monedas sobre la mesa", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "sobre la mesa", "ninguna recta separa", "a"), (960, "ambar", "levantar", "agregar una altura", "b"),
              (1520, "verde", "una hoja plana", "separa las dos alturas", "c")], y=500, w=480, h=330, tam_t=48, tam_d=36),
    P("otra dimensión → lo curvo se vuelve lineal", 960, 820, "violeta", 48, en="c", t=1.2), ilustrativo(1000, texto="analogía"), sec="2. Analogía: levantar monedas")

# ---- 3. Subir a otra dimensión
SQ_AZ, SQ_RO = [x * x for x in AZ], [x * x for x in RO]
v.b("Hagámoslo con números. [[a]]A cada punto le agregamos una segunda coordenada: su valor al cuadrado. [[b]]Los azules, en menos dos y 1.6, pasan a cuatro y 2.56. Los rojos, cerca de cero, quedan abajo: el mayor es 0.7 al cuadrado, 0.49. [[c]]Ahora una recta horizontal, a la altura uno, los separa perfectamente.",
    CL(), T("agregar x² como segunda coordenada", 960, 140, tam=50, c="tinta", ent="subir"),
    G(460, 240, 1460, 860, -2.5, 2.5, -0.3, 4.5, [serie_puntos(AZ, SQ_AZ, "azul", mk=14), serie_puntos(RO, SQ_RO, "rojo", mk=14),
                                                 serie_curva([-2.5, 2.5], [1.0, 1.0], "ambar", 6)],
      xticks=[-2, -1, 0, 1, 2], yticks=[0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="x", ylabel="x²", id="gq", en="a"),
    U("gq", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="b", dur=1.4),
    U("gq", **{"series.2.alfa": 1.0}, en="c", dur=0.6), T("x² = 1", 1560, 600, tam=40, c="ambar", w="sb", en="c", t=0.3),
    ilustrativo(1000), sec="3. Subir a otra dimensión")
v.b("Y si volvemos a la recta original, esa frontera horizontal se ve como dos cortes: en menos uno y en uno. [[a]]Una frontera lineal en el espacio nuevo es una frontera curva en el original.",
    P("lineal en el espacio nuevo = curva en el original", 960, 980, "violeta", 44, en="a"))

# ---- 4. El truco del kernel
v.b("El problema es que subir de dimensión puede ser carísimo: muchas coordenadas nuevas. [[a]]Aquí aparece el truco, de la diapositiva treinta y nueve. Un kernel calcula directamente el producto interno entre los puntos ya transformados, sin construirlos. [[b]]Los tres kernels de la clase: el lineal, equis por zeta; el polinomial, equis por zeta más erre, elevado a de; [[c]]y el radial, e elevado a menos gamma por la distancia al cuadrado.",
    CL(), T("funciones kernel · diapositiva 39", 960, 140, tam=50, c="tinta", ent="subir"),
    caja_formula("K(x, z) = φ(x) · φ(z)", y=290, c="violeta", tam=54, w=900, en="a", wf="sb"),
    fila(450, [("lineal", 560), ("K(x, z) = x·z", 1200)], tams=[44, 44], colores=["azul", "tinta"], en="b", fondo="azulclaro", x0=360, x1=1560, h=90),
    fila(560, [("polinomial", 560), ("K(x, z) = (x·z + r)^d", 1200)], tams=[44, 44], colores=["teal", "tinta"], en="b", t=0.5, fondo="tealclaro", x0=360, x1=1560, h=90),
    fila(670, [("radial (RBF)", 560), ("K(x, z) = exp(−γ‖x − z‖²)", 1200)], tams=[44, 44], colores=["rojo", "tinta"], en="c", fondo="rojoclaro", x0=360, x1=1560, h=90),
    sec="4. El truco del kernel")
v.b("Sobre los hiperparámetros: [[a]]el lineal no tiene gamma. [[b]]El polinomial tiene el grado, el desplazamiento erre, y en scikit-learn también una gamma que multiplica a equis por zeta. [[c]]Y el radial tiene gamma.",
    T("lineal: sin γ", 560, 820, tam=38, c="azul", w="sb", en="a"), T("polinomial: grado d, r y γ", 960, 820, tam=38, c="teal", w="sb", en="b"),
    T("radial: γ", 1360, 820, tam=38, c="rojo", w="sb", en="c"), T("(más C en todos)", 960, 900, tam=32, c="gris", w="sb", en="c", t=0.5))

# ---- 5. El truco con números
v.b("Comprobemos el truco con números. [[a]]Tomemos x igual a uno, dos, y zeta igual a tres, uno, con el kernel polinomial de grado dos y erre igual a uno. [[b]]Equis por zeta es tres más dos, cinco. Más uno, seis. Al cuadrado: [[c]]treinta y seis. Una sola cuenta.",
    CL(), T("el truco con números (kernel polinomial, d = 2, r = 1)", 960, 140, tam=46, c="tinta", ent="subir"),
    P("x = (1, 2)      z = (3, 1)", 960, 280, "azul", 50, en="a"),
    T("K(x, z) = (x·z + 1)² = (1·3 + 2·1 + 1)² = (5 + 1)² =", 1260, 420, tam=42, c="tinta", w="sb", anc="rm", en="b"), N(36, 1290, 420, "{:.0f}", 60, "violeta", anc="lm", en="c"),
    ilustrativo(1000), sec="5. El truco con números")
v.b("Ahora, por el camino largo. [[a]]Ese kernel equivale a llevar cada punto a seis coordenadas: uno, raíz de dos por cada coordenada, los cuadrados, y raíz de dos por el producto. [[b]]El producto interno de esos dos vectores de seis coordenadas es uno, más seis, más cuatro, más nueve, más cuatro, más doce: [[c]]otra vez treinta y seis. El kernel llegó al mismo número sin construir las seis coordenadas.",
    T("φ(x) = (1, √2·x₁, √2·x₂, x₁², x₂², √2·x₁x₂)", 960, 560, tam=40, c="teal", w="sb", en="a"),
    T("φ(x)·φ(z) = 1 + 2·(1·3) + 2·(2·1) + 1·9 + 4·1 + 2·(1·2)(3·1)", 960, 670, tam=36, c="tinta", w="sb", en="b"),
    T("= 1 + 6 + 4 + 9 + 4 + 12 =", 1100, 770, tam=42, c="tinta", w="sb", anc="rm", en="b", t=1.5), N(36, 1130, 770, "{:.0f}", 60, "violeta", anc="lm", en="c"),
    P("mismo número, sin construir φ", 960, 900, "verde", 44, en="c", t=0.8))

# ---- 6. El kernel radial a mano
v.b("Ahora el kernel radial, con la diapositiva cuarenta. [[a]]Dos puntos a distancia dos, en el espacio estandarizado: la distancia al cuadrado es cuatro. [[b]]Con gamma igual a 0.1: e elevado a menos 0.1 por cuatro, e a la menos 0.4: 0.6703. Todavía bastante parecidos. [[c]]Con gamma igual a uno: e a la menos cuatro, 0.0183. Casi nada.",
    CL(), T("el kernel radial a mano · diapositiva 40", 960, 140, tam=50, c="tinta", ent="subir"),
    P("distancia 2  →  ‖x − z‖² = 4", 960, 280, "azul", 50, en="a"),
    T("γ = 0.1:  K = e^(−0.1·4) = e^(−0.4) =", 1120, 420, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(KD["g01"], 1150, 420, "{:.4f}", 58, "verde", anc="lm", en="b", t=1.2),
    T("γ = 1:  K = e^(−1·4) = e^(−4) =", 1120, 540, tam=44, c="tinta", w="sb", anc="rm", en="c"), N(KD["g1"], 1150, 540, "{:.4f}", 58, "rojo", anc="lm", en="c", t=0.9),
    ilustrativo(1000, texto="ejemplo de la diapositiva 40"), sec="6. El kernel radial a mano")
v.b("Otro, con coordenadas. [[a]]x igual a uno, dos, y zeta igual a dos, cero. La diferencia es menos uno, dos; su norma al cuadrado, uno más cuatro, cinco. [[b]]Con gamma igual a 0.1, e a la menos 0.5: 0.607. [[c]]Con gamma igual a uno, e a la menos cinco: 0.0067.",
    T("x = (1, 2), z = (2, 0):  ‖x − z‖² = (−1)² + 2² = 5", 960, 680, tam=42, c="tinta", w="sb", en="a"),
    T("γ = 0.1 → e^(−0.5) = 0.607", 660, 800, tam=42, c="verde", w="sb", en="b"), T("γ = 1 → e^(−5) = 0.0067", 1300, 800, tam=42, c="rojo", w="sb", en="c"),
    ilustrativo(1040, texto="segundo cálculo: ejemplo ilustrativo"))

# ---- 7. Gamma y el alcance
DS = [0.04 * i for i in range(101)]
v.b("Mira las curvas completas: similitud contra distancia, para cuatro valores de gamma. [[a]]Con gamma pequeña, en azul, la similitud decae lento: un punto influye lejos. [[b]]Con gamma grande, en rojo, se desploma enseguida: cada punto influye solo en su vecindario. [[c]]Por eso gamma controla el alcance de la similitud.",
    CL(), T("γ controla el alcance", 960, 140, tam=52, c="tinta", ent="subir"),
    G(480, 240, 1400, 860, 0, 4, 0, 1, [dict(xs=DS, ys=[math.exp(-g * d * d) for d in DS], color=c, grosor=6, revela=0.0)
                                        for g, c in [(0.1, "azul"), (0.5, "teal"), (1, "ambar"), (5, "rojo")]],
      xticks=[0, 1, 2, 3, 4], yticks=[0, 0.25, 0.5, 0.75, 1], yfmt="{:.2f}", xlabel="distancia ‖x − z‖", ylabel="K", id="gk", en="a"),
    U("gk", **{"series.0.revela": 1.0}, en="a", dur=1.2), U("gk", **{"series.1.revela": 1.0, "series.2.revela": 1.0}, en="a", t=1.0, dur=1.2),
    U("gk", **{"series.3.revela": 1.0}, en="b", dur=1.0),
    T("γ = 0.1", 1600, 330, tam=36, c="azul", w="sb", en="a"), T("γ = 0.5", 1600, 390, tam=36, c="teal", w="sb", en="a", t=1.0),
    T("γ = 1", 1600, 450, tam=36, c="ambar", w="sb", en="a", t=1.2), T("γ = 5", 1600, 510, tam=36, c="rojo", w="sb", en="b"),
    P("γ chica: influencia lejana", 1600, 660, "azul", 32, en="a", t=2.0), P("γ grande: influencia local", 1600, 760, "rojo", 32, en="b", t=0.6),
    sec="7. γ y el alcance")
v.b("Y así decide la SVM radial: [[a]]el puntaje de un caso nuevo es una suma de similitudes con los vectores de soporte, cada una con su peso. [[b]]Cada vector de soporte pone una pequeña colina alrededor de sí. La frontera es donde la suma de colinas cruza el cero.",
    CL(), caja_formula("f(x) = Σ αᵢ·yᵢ·K(xᵢ, x) + b", y=360, c="violeta", tam=56, w=1100, en="a", wf="sb"),
    P("cada vector de soporte = una colina alrededor de sí", 960, 560, "ambar", 46, en="b"),
    T("frontera: donde la suma cruza el 0", 960, 680, tam=42, c="gris", w="sb", en="b", t=0.8))

# ---- 8. Con datos reales
RP = D["rbf2_prof"]
L2 = D["lin2"]["1"]
xs_l, ys_l = recta_caja(L2["w"], L2["b"], 0.0, *CAJA)
v.b("Con los datos reales, en dos variables. [[a]]Una SVM lineal, con C igual a uno, traza esta recta: acierta el 88.3 por ciento de la prueba. [[b]]La SVM radial, con la configuración de la clase, C igual a diez y gamma igual a 0.1, traza esta curva: acierta el 90.1 por ciento. [[c]]La curva se dobla donde las clases se mezclan.",
    CL(), T("lineal frente a radial · 2 variables · datos reales", 960, 140, tam=46, c="tinta", ent="subir"),
    G(420, 230, 1360, 920, *CAJA, nube((Z["x"], Z["y"]), Z["c"], mk=6) + [serie_curva(xs_l, ys_l, "gris", 6)] +
      [serie_curva(s["x"], s["y"], "violeta", 7) for s in RP["segs"]],
      xticks=[-2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="radio (estandarizado)", ylabel="concav.", id="gr", tam_txt=28),
    U("gr", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, t=0.3, dur=1.2),
    U("gr", **{"series.2.alfa": 1.0}, en="a", dur=0.6), T(f"lineal (C = 1): {100 * L2['acc_test']:.1f} %", 1620, 400, tam=36, c="gris", w="sb", en="a", t=0.3),
    U("gr", **{f"series.{3 + i}.alfa": 1.0 for i in range(len(RP["segs"]))}, en="b", dur=0.8),
    T(f"radial (C = 10, γ = 0.1): {100 * RP['acc_test']:.1f} %", 1620, 480, tam=34, c="violeta", w="sb", en="b", t=0.3),
    real(1050, en="a"), sec="8. Con datos reales")
v.b("En la figura de la guía, con C igual a uno, el kernel polinomial de grado tres acierta el 88.9 por ciento, [[a]]y el radial con gamma 0.5, el 90.1. Con solo dos variables, las diferencias son de uno o dos casos.",
    T(f"guía, C = 1: polinomial (grado 3) {100 * K2['Polinomial (grado 3)']['acc_test']:.1f} %", 1640, 580, tam=28, c="tinta", w="sb"),
    T(f"RBF (γ = 0.5) {100 * K2['RBF (γ = 0.5)']['acc_test']:.1f} %", 1640, 625, tam=28, c="tinta", w="sb", t=0.6),
    P("diferencias de 1 o 2 casos", 1640, 720, "ambar", 34, en="a"))

# ---- 9. En R y en Python
v.b("En el código. [[a]]En R, la función svm con kernel igual a radial, cost igual a diez y gamma igual a 0.1. [[b]]En Python, la clase {SVC|ese uve ce}, con kernel igual a RBF, C igual a diez y gamma igual a 0.1. Python llama RBF al kernel radial. [[c]]Las dos usan la misma fórmula: e a la menos gamma por la distancia al cuadrado.",
    CL(), T("el kernel radial en el código de la clase", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["svm(x = x_entrenamiento, y = y_entrenamiento,", "    kernel = \"radial\", cost = 10, gamma = 0.1, scale = TRUE)"], y=330, w=1700, tam=32, c="teal", en="a"),
    codigo(["SVC(kernel=\"rbf\", C=10, gamma=0.1)"], y=560, w=1700, tam=34, c="azul", en="b"),
    P("misma fórmula: exp(−γ‖x − z‖²)", 960, 760, "violeta", 46, en="c"), sec="9. En R y en Python")

# ---- 10. El error típico
v.b("El error típico al calcular a mano: [[a]]olvidar el cuadrado de la distancia. [[b]]Con distancia dos y gamma 0.1, si alguien hace e a la menos 0.1 por dos, obtiene 0.819. [[c]]Lo correcto es e a la menos 0.1 por cuatro: 0.670. La fórmula usa la distancia al cuadrado.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("olvidar el cuadrado de la distancia", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("mal:  e^(−0.1·2) = 0.819", 960, 420, tam=50, c="rojo", w="sb", en="b", id="mal"), U("mal", tachado=1.0, en="c", dur=0.6),
    T("bien:  e^(−0.1·2²) = e^(−0.4) = 0.670", 960, 560, tam=50, c="verde", w="sb", en="c", t=0.4), sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Dos puntos están a distancia tres. [[b]]¿Cuánto vale el kernel radial con gamma 0.1? ¿Y con gamma uno?",
    CL(), P("distancia = 3", 960, 330, "azul", 56, en="a"), P("γ = 0.1 → K = ?", 700, 500, "verde", 50, en="b"), P("γ = 1 → K = ?", 1220, 500, "rojo", 50, en="b", t=0.4),
    sec="11. Piensa un segundo", pausa=0.6)
v.b("La distancia al cuadrado es nueve. [[a]]Con gamma 0.1: e a la menos 0.9, 0.407. [[b]]Con gamma uno: e a la menos nueve, 0.0001. [[c]]Con gamma grande, a distancia tres ya no hay ninguna similitud.",
    CL(), T("‖x − z‖² = 9", 960, 280, tam=54, c="tinta", w="sb"),
    T("γ = 0.1:  e^(−0.9) =", 960, 420, tam=50, c="tinta", w="sb", anc="rm", en="a"), N(0.407, 990, 420, "{:.3f}", 62, "verde", anc="lm", en="a", t=0.5),
    T("γ = 1:  e^(−9) =", 960, 540, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(0.0001, 990, 540, "{:.4f}", 62, "rojo", anc="lm", en="b", t=0.5),
    P("γ grande: a distancia 3 ya no hay similitud", 960, 700, "ambar", 46, en="c"), ilustrativo(1000))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]un kernel calcula la similitud entre puntos como si estuvieran en un espacio más grande, sin construirlo. [[b]]El radial es e a la menos gamma por la distancia al cuadrado. [[c]]Y gamma controla el alcance: chica, influencia lejana y fronteras suaves; grande, influencia local y fronteras irregulares.",
    CL(), P("kernel = producto interno en otro espacio, sin construirlo", 960, 300, "violeta", 46, en="a"),
    P("radial: K = exp(−γ‖x − z‖²)", 960, 460, "rojo", 50, en="b"), P("γ chica → suave · γ grande → local e irregular", 960, 620, "ambar", 46, en="c"),
    sec="12. Para llevar")
v.b("Siguiente parada: [[a]]C y gamma juntos, con datos reales. Cuándo el modelo subajusta, y cuándo sobreajusta.",
    CL(), siguiente(8, "C y gamma: subajuste y sobreajuste"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
