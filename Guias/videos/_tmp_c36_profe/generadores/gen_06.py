import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, alerta, cien

OUT = out("guion_c36_06_perdida.json")
v = V("La pérdida y el gradiente")
NA = D["nb_ames"]
GB_HUB = NA["Gradient Boosting"]["rmse_test"]
GB_CUA = D["nb_ames_gb_cuadratica_rmse"]

FS = [2 + 0.2 * i for i in range(101)]
Y0, DEL = 12, 2.0


def hub(r):
    return 0.5 * r * r if abs(r) <= DEL else DEL * (abs(r) - 0.5 * DEL)


v.portada("La pérdida y el gradiente", "Por qué se llama «gradient», y qué aprende cada árbol",
          "Sexta parada: la pérdida y el gradiente. Por qué el algoritmo se llama Gradient Boosting, y qué cambia cuando cambias la pérdida.")

# ---- 1. Hook
v.b("En la parada anterior, cada árbol aprendía los residuos. [[a]]Pero el algoritmo no se llama residual boosting. Se llama Gradient Boosting. [[b]]¿Qué tiene que ver un gradiente con todo esto?",
    CL(), T("cada árbol aprendía los residuos…", 960, 280, tam=54, c="tinta", ent="subir"),
    T("…pero se llama GRADIENT Boosting", 960, 440, tam=60, c="ambar", en="a"), T("¿por qué «gradient»?", 960, 620, tam=80, c="tinta", en="b"),
    sec="1. ¿Por qué «gradient»?")

# ---- 2. Analogía: el cerro con niebla
v.b("Imagina que estás en un cerro, con una niebla tan densa que no ves el valle. [[a]]Pero puedes sentir la pendiente bajo tus pies. [[b]]Entonces das un paso hacia donde el suelo baja más. [[c]]Y repites. Paso a paso, llegas abajo, sin haber visto nunca el valle completo.",
    CL(), T("bajar un cerro con niebla", 960, 160, tam=52, c="tinta", ent="subir"),
    G(360, 260, 1560, 820, -4, 4, 0, 17, [dict(xs=[-4 + 0.1 * i for i in range(81)], ys=[(-4 + 0.1 * i) ** 2 for i in range(81)], color="gris", grosor=6,
                                             suave=True)], xticks=[], yticks=[], rejilla=False, id="cerro"),
    *[Bc(360 + (x + 4) / 8 * 1200, 820 - (x * x) / 17 * 560 - 18, 34, 34, "ambar", "blanco", 3, 17, en=e, t=tt, id=f"pz{k}")
      for k, (x, e, tt) in enumerate([(-3.6, "a", 0.0), (-2.6, "b", 0.0), (-1.6, "b", 0.8), (-0.8, "c", 0.0), (-0.2, "c", 0.6)])],
    P("pendiente = gradiente  ·  largo del paso = η", 960, 920, "ambar", 44, en="c", t=1.0),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el cerro")
v.b("En boosting, el cerro es la pérdida: [[a]]una función que mide qué tan mal predices. [[b]]El gradiente dice hacia dónde sube la pérdida. Su negativo, el gradiente negativo, dice hacia dónde baja. [[c]]Y cada árbol nuevo aprende a predecir esa dirección.",
    CL(), T("el cerro = la pérdida L(y, F)", 960, 200, tam=56, c="tinta", en="a"),
    T("gradiente: hacia dónde SUBE la pérdida", 960, 360, tam=46, c="rojo", w="sb", en="b"),
    T("gradiente negativo −∂L/∂F: hacia dónde BAJA", 960, 460, tam=46, c="verde", w="sb", en="b", t=0.8),
    P("cada árbol nuevo aprende esa dirección", 960, 640, "azul", 52, en="c"))

# ---- 3. La pérdida cuadrática
v.b("Empecemos con la pérdida cuadrática: un medio de y menos efe, al cuadrado. [[a]]Su derivada respecto de efe es menos, y menos efe. [[b]]Y el gradiente negativo es y menos efe: exactamente el residuo. [[c]]Por eso, con pérdida cuadrática, ajustar el gradiente es lo mismo que ajustar los residuos. Para eso está el un medio: para que la derivada quede limpia.",
    CL(), T("la pérdida cuadrática", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("L = ½ · (y − F)²", y=290, c="ambar", tam=60, w=800),
    T("∂L/∂F = −(y − F)", 960, 440, tam=54, c="tinta", w="sb", en="a"), T("−∂L/∂F = y − F = residuo", 960, 560, tam=60, c="verde", en="b"),
    P("cuadrática: gradiente negativo = residuo", 960, 720, "verde", 48, en="c"), T("el ½ deja la derivada limpia", 960, 820, tam=38, c="gris", w="sb", en="c", t=1.5),
    sec="3. La pérdida cuadrática")

# ---- 4. Otras pérdidas
v.b("Pero hay otras pérdidas. La diapositiva once lo dice: la pérdida define qué significa equivocarse. [[a]]La absoluta, el valor absoluto de y menos efe. [[b]]Y la de Huber, que es cuadrática para errores chicos y absoluta para errores grandes. Aquí, con y igual a 12.",
    CL(), T("tres pérdidas, con y = 12 · ¿qué significa equivocarse?", 960, 140, tam=44, c="tinta", ent="subir"),
    G(300, 240, 1300, 830, 2, 22, 0, 50, [dict(xs=FS, ys=[0.5 * (Y0 - f) ** 2 for f in FS], color="ambar", grosor=6, revela=0.0),
                                          dict(xs=FS, ys=[abs(Y0 - f) for f in FS], color="azul", grosor=6, revela=0.0),
                                          dict(xs=FS, ys=[hub(Y0 - f) for f in FS], color="verde", grosor=6, revela=0.0)],
      xticks=[2, 7, 12, 17, 22], yticks=[0, 10, 20, 30, 40, 50], yfmt="{:.0f}", xlabel="predicción F", ylabel="L(y, F)", id="gl"),
    U("gl", **{"series.0.revela": 1.0}, dur=1.5), P("cuadrática ½(y − F)²", 1600, 330, "ambar", 38),
    U("gl", **{"series.1.revela": 1.0}, en="a", dur=1.5), P("absoluta |y − F|", 1600, 450, "azul", 38, en="a"),
    U("gl", **{"series.2.revela": 1.0}, en="b", dur=1.5), P("Huber (δ = 2)", 1600, 570, "verde", 38, en="b"),
    ilustrativo(1000), sec="4. Otras pérdidas")
v.b("Mira las colas. [[a]]La cuadrática castiga muchísimo los errores grandes: crece como una parábola. [[b]]La absoluta y la de Huber crecen en línea recta: un error enorme no domina todo.",
    P("cuadrática: castiga muchísimo lo grande", 1610, 700, "ambar", 28, en="a"),
    P("absoluta y Huber: crecen en línea recta", 1610, 790, "verde", 28, en="b"))

# ---- 5. Ejemplo resuelto: qué aprende el siguiente árbol (apunte)
v.b("¿Qué aprende el siguiente árbol con cada pérdida? Con el ejemplo del apunte: [[a]]y vale 12, y la predicción actual es 9. [[b]]Con la cuadrática, el gradiente negativo es el residuo: 12 menos 9, más 3. [[c]]Con la absoluta, es solo el signo del residuo: más 1. Y con Huber, con delta igual a 2, el residuo recortado en 2: más 2.",
    CL(), T("qué aprende el siguiente árbol · ejemplo del apunte", 960, 150, tam=46, c="tinta", ent="subir"),
    P("y = 12 · F = 9", 960, 270, "teal", 52, en="a"),
    T("cuadrática:  y − F = 12 − 9 =", 1060, 410, tam=46, c="ambar", w="sb", anc="rm", en="b"), T("+3", 1090, 410, tam=60, c="ambar", anc="lm", en="b", t=0.8),
    T("absoluta:  signo(y − F) =", 1060, 530, tam=46, c="azul", w="sb", anc="rm", en="c"), T("+1", 1090, 530, tam=60, c="azul", anc="lm", en="c", t=0.5),
    T("Huber (δ = 2):  recortar +3 en ±2 →", 1060, 650, tam=46, c="verde", w="sb", anc="rm", en="c", t=1.2), T("+2", 1090, 650, tam=60, c="verde", anc="lm", en="c", t=2.0),
    ilustrativo(1000, texto="ejemplo del apunte"), sec="5. Qué aprende el árbol")
v.b("Ahora un valor atípico: supón que y vale 40, y la predicción sigue en 9. [[a]]Con la cuadrática, el árbol tiene que aprender más 31: ese solo caso arrastra al árbol. [[b]]Con la absoluta, más 1. Con Huber, más 2. [[c]]Por eso la absoluta y la de Huber se llaman robustas: un caso extremo no domina al árbol.",
    CL(), P("y = 40 (atípico) · F = 9", 960, 230, "rojo", 52),
    T("cuadrática:", 900, 380, tam=48, c="ambar", w="sb", anc="rm", en="a"), T("+31", 930, 380, tam=64, c="ambar", anc="lm", en="a", t=0.4),
    T("absoluta:", 900, 500, tam=48, c="azul", w="sb", anc="rm", en="b"), T("+1", 930, 500, tam=64, c="azul", anc="lm", en="b", t=0.3),
    T("Huber:", 900, 620, tam=48, c="verde", w="sb", anc="rm", en="b", t=0.8), T("+2", 930, 620, tam=64, c="verde", anc="lm", en="b", t=1.0),
    P("absoluta y Huber = pérdidas ROBUSTAS", 960, 790, "verde", 50, en="c"), ilustrativo(1000))

# ---- 6. Clasificación: log-loss
v.b("¿Y en clasificación? Ahí la pérdida es la logística, la log-loss. [[a]]El modelo suma puntajes, efe, y los convierte en probabilidad con la función logística. [[b]]El gradiente negativo es y menos pe: la clase real, cero o uno, menos la probabilidad que da el modelo.",
    CL(), T("clasificación: la pérdida logística (log-loss)", 960, 150, tam=48, c="tinta", ent="subir"),
    caja_formula("p = 1 / (1 + e^(−F))", y=310, c="violeta", tam=56, w=900, en="a"),
    T("gradiente negativo = y − p", 960, 480, tam=60, c="verde", en="b"), T("y = 0 o 1  ·  p = probabilidad del modelo", 960, 580, tam=40, c="gris", w="sb", en="b", t=0.6),
    sec="6. Clasificación")
v.b("Con el apunte: un paciente enfermo, y igual a uno. [[a]]Si el puntaje es cero, pe es 0.5, y el árbol aprende 0.5. [[b]]Si el puntaje ya es dos, pe es 0.881, y el árbol aprende apenas 0.119. [[c]]A medida que el modelo acierta, el gradiente se achica solo.",
    CL(), P("y = 1 (paciente enfermo)", 960, 230, "rojo", 50),
    T("F = 0  →  p = 0.5  →  y − p =", 1060, 380, tam=46, c="tinta", w="sb", anc="rm", en="a"), N(0.5, 1090, 380, "{:.1f}", 60, "verde", anc="lm", en="a", t=0.8),
    T("F = 2  →  p = 0.881  →  y − p =", 1060, 510, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(0.119, 1090, 510, "{:.3f}", 60, "verde", anc="lm", en="b", t=1.2),
    P("cuanto más acierta, más chico el gradiente", 960, 690, "verde", 48, en="c"), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 7. En Python
v.b("En scikit-learn, la pérdida es un parámetro. [[a]]En regresión, la cuadrática es la de defecto, y también están la absoluta y la de Huber. [[b]]En clasificación, la de defecto es la log-loss.",
    CL(), T("la pérdida en scikit-learn", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["GradientBoostingRegressor(loss='squared_error')   # por defecto", "GradientBoostingRegressor(loss='absolute_error')",
            "GradientBoostingRegressor(loss='huber')", "GradientBoostingClassifier(loss='log_loss')       # por defecto"],
           y=440, w=1560, tam=36, en="a", dt=0.4), sec="7. En Python")

# ---- 8. Alerta del notebook
v.b("Y aquí, una alerta del notebook de la clase. [[a]]En Ames, el Gradient Boosting se entrena con la pérdida de Huber. [[b]]Pero después se evalúa, y se optimiza, con el RMSE, que corresponde a la pérdida cuadrática. [[c]]El objetivo con que se entrena y la métrica con que se juzga no coinciden.",
    CL(), alerta(170, texto="alerta del notebook"),
    T("Ames:  GradientBoostingRegressor(loss='huber')", 960, 330, tam=44, c="tinta", w="m", en="a"),
    T("se evalúa y se optimiza con RMSE (pérdida cuadrática)", 960, 450, tam=44, c="tinta", w="sb", en="b"),
    P("objetivo ≠ métrica", 960, 600, "rojo", 56, en="c"), sec="8. Alerta del notebook")
v.b("Lo medimos con la misma configuración del notebook. [[a]]Con Huber, el RMSE en test es 29 200 dólares. [[b]]Con la pérdida cuadrática, 28 600. Unos quinientos dólares mejor, solo por alinear el objetivo con la métrica.",
    T(f"Huber: RMSE de test {cien(GB_HUB).replace(' ', '.')} USD", 960, 740, tam=44, c="tinta", w="sb", en="a"),
    T(f"cuadrática: RMSE de test {cien(GB_CUA).replace(' ', '.')} USD", 960, 830, tam=44, c="verde", w="sb", en="b"),
    real(1050, texto="réplica del notebook (150 árboles, η = 0.05, profundidad 2)"))

# ---- 9. Error típico
v.b("El error típico: [[a]]decir que boosting siempre ajusta los residuos, y menos efe. [[b]]Eso es cierto solo con la pérdida cuadrática. [[c]]Con la absoluta, el árbol aprende signos; con Huber, residuos recortados; con la logística, y menos pe. Lo general es el gradiente negativo.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("«boosting siempre ajusta y − F»", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("solo con la pérdida cuadrática", 960, 400, tam=46, c="tinta", w="sb", en="b"),
    T("absoluta: signos · Huber: recortados · logística: y − p", 960, 520, tam=42, c="gris", w="sb", en="c"),
    P("lo general: el GRADIENTE NEGATIVO", 960, 680, "verde", 52, en="c", t=1.5), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con la pérdida absoluta, dos casas tienen residuos de más 1 y de más 100. [[b]]¿Qué valores tiene que aprender el siguiente árbol?",
    CL(), P("pérdida absoluta · residuos +1 y +100", 960, 300, "azul", 54, en="a"), T("¿qué aprende el árbol?", 960, 470, tam=72, c="tinta", en="b"),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("Más 1 y más 1. [[a]]Con la pérdida absoluta, el árbol solo ve el signo: las dos casas le piden lo mismo, subir. [[b]]El tamaño del error no importa para la dirección, y por eso un caso extremo no arrastra al modelo.",
    CL(), T("+1 y +1", 960, 320, tam=96, c="azul"), P("solo cuenta el signo: las dos piden «subir»", 960, 500, "azul", 48, en="a"),
    T("un caso extremo no arrastra al modelo", 960, 620, tam=42, c="gris", w="sb", en="b"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]la pérdida define qué significa equivocarse. [[b]]Cada árbol aprende el gradiente negativo de esa pérdida, que con la cuadrática coincide con el residuo. [[c]]Y entrena con la misma pérdida que vas a usar para juzgar al modelo.",
    CL(), P("la pérdida define qué es equivocarse", 960, 300, "ambar", 50, en="a"),
    P("cada árbol aprende −∂L/∂F (cuadrática: el residuo)", 960, 460, "verde", 46, en="b"),
    P("entrena con la pérdida con la que vas a juzgar", 960, 620, "azul", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la tasa de aprendizaje y el número de árboles. Dos perillas que siempre se mueven juntas.",
    CL(), siguiente(7, "Tasa de aprendizaje y número de árboles"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
