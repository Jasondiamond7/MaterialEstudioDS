import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, k1, cien

OUT = out("guion_c36_07_tasa.json")
v = V("Tasa de aprendizaje y número de árboles")
LR = {d["lr"]: d for d in FG["lr"]}
CU = D["curvas"]
YMAX = 45


def serie(key, c, g=6):
    c_ = CU[key]
    xs = list(range(1, 801, 2))
    return dict(xs=xs, ys=[min(c_[x - 1] / 1000, YMAX) for x in xs], color=c, grosor=g, revela=0.0)


v.portada("Tasa y número de árboles", "Pasos chicos y muchos árboles, o pasos grandes y pocos",
          "Séptima parada: la tasa de aprendizaje y el número de árboles. Dos perillas que siempre se mueven juntas.")

# ---- 1. Hook
v.b("Si cada árbol corrige errores, ¿por qué no dejar que cada uno corrija todo lo que pueda? [[a]]Eta igual a uno, pasos completos, y listo. [[b]]Con datos reales vas a ver que eso sale peor. Y vas a ver cuántos árboles más hacen falta cuando achicas el paso.",
    CL(), T("¿por qué no pasos completos?", 960, 300, tam=66, c="tinta", ent="subir"), P("η = 1 y listo", 960, 470, "rojo", 60, en="a"),
    T("spoiler: sale peor", 960, 640, tam=54, c="gris", w="sb", en="b"), sec="1. ¿Por qué no η = 1?")

# ---- 2. Analogía
v.b("Piensa en dos formas de estudiar para un examen. [[a]]Una: todo de golpe, la noche anterior. [[b]]Otra: un poco cada día, durante semanas. [[c]]La segunda toma más días, pero deja un aprendizaje más estable. Eta es el tamaño de cada sesión, y los árboles son los días.",
    CL(), T("dos formas de estudiar", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "todo de golpe", "la noche anterior: η grande, pocos árboles", "a"),
              (1360, "verde", "un poco cada día", "más días, más estable: η chico, muchos árboles", "b")], y=470, w=640, h=360, tam_t=56),
    P("η = tamaño de la sesión · árboles = días", 960, 790, "ambar", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía")

# ---- 3. Las tres perillas (diapositiva 11)
v.b("La diapositiva once habla de tres perillas fundamentales. [[a]]La pérdida, que define qué significa equivocarse. [[b]]La tasa eta, que controla el tamaño de cada paso: menor eta suele requerir más árboles. [[c]]Y la complejidad, que limita lo que aprende cada árbol: su profundidad, sus hojas, sus muestras mínimas.",
    CL(), T("las tres perillas · diapositiva 11", 960, 160, tam=50, c="tinta", ent="subir"),
    tarjetas([(400, "rojo", "PÉRDIDA", "qué significa equivocarse", "a"), (960, "ambar", "TASA η", "tamaño de cada paso", "b"),
              (1520, "azul", "COMPLEJIDAD", "profundidad, hojas, muestras mínimas", "c")], y=480, w=480, h=330, tam_t=46, tam_d=34),
    sec="3. Las tres perillas")
v.b("En boosting, los árboles suelen ser chicos: [[a]]profundidad dos a seis, o incluso stumps. [[b]]Achicar cada paso tiene un nombre en inglés: shrinkage, encogimiento.",
    P("árboles chicos: profundidad 2 a 6, o stumps", 960, 780, "azul", 46, en="a"), P("η = shrinkage (encogimiento)", 960, 890, "ambar", 46, en="b"))

# ---- 4. Modelo idealizado (apunte)
v.b("Un modelo idealizado ayuda a ver el efecto. [[a]]Supón que cada árbol corrige exactamente una fracción eta del residuo que queda. [[b]]Entonces, tras eme árboles, queda el residuo inicial por uno menos eta, elevado a eme.",
    CL(), T("un modelo idealizado", 960, 160, tam=52, c="tinta", ent="subir"),
    T("cada árbol corrige exactamente la fracción η del residuo", 960, 300, tam=42, c="tinta", w="sb", en="a"),
    caja_formula("residuo tras m árboles = r₀ · (1 − η)ᵐ", y=460, c="ambar", tam=56, w=1200, en="b"), sec="4. Un modelo idealizado")
v.b("Con el ejemplo del apunte: el residuo parte en 10, y queremos bajarlo a menos de 1. [[a]]Con eta 0.5: queda 5, luego 2.5, luego 1.25, y con el cuarto árbol, 0.625. Bastan cuatro árboles. [[b]]Con eta 0.1: 9, 8.1, 7.29… y recién con el árbol 22 baja de 1.",
    CL(), T("r₀ = 10 · ¿cuántos árboles para bajar de 1?", 960, 150, tam=46, c="tinta", ent="subir"),
    T("η = 0.5:", 330, 300, tam=46, c="verde", anc="rm", en="a"),
    *[[Bx(370 + 120 * k, 300 - 36, 370 + 120 * k + 10 * v_, 300 + 36, "verde", None, 0, 8, grow="x", en="a", t=0.4 * k) if False else
       T(f"{v_:g}", 420 + 160 * k, 300, tam=44, c="verde", w="b", en="a", t=0.5 * k)] for k, v_ in enumerate([5, 2.5, 1.25, 0.625])],
    T("→ 4 árboles", 1180, 300, tam=46, c="verde", w="b", anc="lm", en="a", t=2.2),
    T("η = 0.1:", 330, 440, tam=46, c="azul", anc="rm", en="b"),
    *[T(f"{v_:g}", 420 + 160 * k, 440, tam=44, c="azul", w="b", en="b", t=0.5 * k) for k, v_ in enumerate([9, 8.1, 7.29])],
    T("…", 900, 440, tam=44, c="azul", w="b", en="b", t=1.6), T("→ 22 árboles", 1180, 440, tam=46, c="azul", w="b", anc="lm", en="b", t=2.2),
    ilustrativo(1000, texto="ejemplo del apunte"))
v.b("La cuenta exacta: [[a]]0.5 elevado a cuatro es 0.0625: menos de una décima. [[b]]0.9 elevado a 21 todavía es 0.109; elevado a 22, 0.098. [[c]]Dividir eta por cinco multiplicó por cinco y medio los árboles necesarios.",
    T("0.5⁴ = 0.0625 < 0.1", 960, 600, tam=48, c="verde", w="sb", en="a"),
    T("0.9²¹ = 0.109  ·  0.9²² = 0.098 < 0.1", 960, 700, tam=48, c="azul", w="sb", en="b"),
    P("η ÷ 5  →  árboles × 5.5", 960, 840, "ambar", 54, en="c"))

# ---- 5. Datos reales: cuatro curvas en Ames
v.b("Ahora con datos reales. [[a]]Gradient Boosting en Ames, con árboles de profundidad tres, hasta ochocientos árboles, y cuatro valores de eta. [[b]]Con eta igual a uno, el error baja rapidísimo, y su mejor punto llega con solo nueve árboles: 30 300 dólares. [[c]]Después empeora.",
    CL(), T("Ames · RMSE de validación según árboles y η", 960, 140, tam=46, c="tinta", ent="subir"),
    G(260, 240, 1240, 840, 0, 800, 17, YMAX, [serie("lr1.0", "rojo"), serie("lr0.3", "ambar"), serie("lr0.1", "violeta"), serie("lr0.03", "verde")],
      xticks=[0, 200, 400, 600, 800], yticks=[20, 25, 30, 35, 40, 45], yfmt="{:.0f} mil", xlabel="número de árboles", id="gl", en="a"),
    U("gl", **{"series.0.revela": 1.0}, en="b", dur=2.0), P(f"η = 1: mejor con {LR[1.0]['n']} árboles · {k1(LR[1.0]['best'])}", 1590, 300, "rojo", 28, en="b", t=0.8),
    T(f"con 800: {k1(LR[1.0]['last'])}", 1590, 360, tam=28, c="rojo", w="sb", en="c"), real(1050), sec="5. Cuatro curvas reales")
v.b("Con eta 0.3, [[a]]su mejor punto llega con 462 árboles: 22 500. [[b]]Con eta 0.1, con 587 árboles: 21 500. [[c]]Y con eta 0.03, con 792 árboles, casi el tope de ochocientos: también 21 500.",
    U("gl", **{"series.1.revela": 1.0}, en="a", dur=1.6), P(f"η = 0.3: {LR[0.3]['n']} árboles · {k1(LR[0.3]['best'])}", 1590, 450, "ambar", 28, en="a", t=0.6),
    U("gl", **{"series.2.revela": 1.0}, en="b", dur=1.6), P(f"η = 0.1: {LR[0.1]['n']} árboles · {k1(LR[0.1]['best'])}", 1590, 540, "violeta", 28, en="b", t=0.6),
    U("gl", **{"series.3.revela": 1.0}, en="c", dur=1.6), P(f"η = 0.03: {LR[0.03]['n']} árboles · {k1(LR[0.03]['best'])}", 1590, 630, "verde", 28, en="c", t=0.6))
v.b("Tres lecciones. [[a]]Eta grande aprende rápido, pero se queda corto y luego empeora. [[b]]Eta chico necesita muchos más árboles, pero llega más abajo. [[c]]Y entre 0.1 y 0.03 ya casi no hay diferencia de error: solo de costo.",
    P("η grande: rápido, pero se queda corto", 1590, 740, "rojo", 28, en="a"), P("η chico: más árboles, llega más abajo", 1590, 810, "verde", 28, en="b"),
    T("el inicio de las curvas queda fuera de escala", 750, 985, tam=28, c="gris", w="sb", en="c"),
    P("0.1 vs 0.03: mismo error, distinto costo", 1590, 880, "violeta", 28, en="c"))

# ---- 6. Error típico
v.b("El error típico: [[a]]bajar eta, sin subir el número de árboles. [[b]]Mira el corte en cien árboles. Con eta 0.1, el error ya va en 22 700. [[c]]Con eta 0.03, todavía va en 25 900: el modelo quedó subajustado, a medio camino.",
    CL(), K(330, 220, "cruz", 70, en="a"), T("bajar η sin subir los árboles", 410, 220, tam=50, c="rojo", anc="lm", en="a"),
    T("con 100 árboles:", 960, 360, tam=46, c="tinta", w="sb", en="b"),
    T("η = 0.1:", 900, 480, tam=48, c="violeta", w="sb", anc="rm", en="b", t=0.5), T(k1(CU["lr0.1"][99]), 930, 480, tam=56, c="violeta", anc="lm", en="b", t=1.0),
    T("η = 0.03:", 900, 600, tam=48, c="verde", w="sb", anc="rm", en="c"), T(k1(CU["lr0.03"][99]), 930, 600, tam=56, c="rojo", anc="lm", en="c", t=0.5),
    P("η bajo + pocos árboles = SUBAJUSTE", 960, 770, "rojo", 52, en="c", t=1.5), real(1050), sec="6. El error típico")
v.b("La regla de la diapositiva once: [[a]]más árboles con menor tasa suele ser una combinación estable, pero aumenta el costo. [[b]]Eta y el número de árboles se ajustan en pareja, nunca por separado.",
    CL(), P("más árboles + menor η: estable, pero más caro", 960, 380, "ambar", 50, en="a"),
    P("η y n_estimators se ajustan EN PAREJA", 960, 560, "azul", 54, en="b"), T("diapositiva 11", 1860, 1030, tam=26, c="gris", w="sb", anc="rm"))

# ---- 7. En Python
v.b("En Python, las dos perillas son parámetros del mismo modelo. [[a]]n_estimators es el número de árboles, y learning_rate es eta. [[b]]Y la complejidad, con max_depth, o con el mínimo de muestras por hoja.",
    CL(), T("las perillas en scikit-learn", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["GradientBoostingRegressor(", "    n_estimators=600,     # número de árboles", "    learning_rate=0.1,    # η",
            "    max_depth=3,          # complejidad", "    min_samples_leaf=10)"], y=460, w=1300, tam=38, en="a", dt=0.3),
    P("n_estimators ↔ learning_rate: siempre juntos", 960, 800, "ambar", 44, en="b"), sec="7. En Python")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Pasar de eta 0.1 a eta 0.03, ¿vale la pena en Ames?",
    CL(), P("η = 0.1  →  η = 0.03", 960, 320, "violeta", 60, en="a"), T("¿vale la pena?", 960, 500, tam=76, c="tinta", en="a", t=0.6),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("Con eta 0.1, el mínimo es 21 512 dólares, con 587 árboles. [[a]]Con eta 0.03, 21 462 dólares, con 792. [[b]]Cincuenta dólares de diferencia, por un treinta y cinco por ciento más de árboles. [[c]]En este caso, no vale la pena. Por eso, en la práctica, se fija un eta moderado y se deja que la validación elija los árboles.",
    CL(), T(f"η = 0.1:  {LR[0.1]['best']:,} USD con {LR[0.1]['n']} árboles".replace(",", "."), 960, 280, tam=48, c="violeta", w="sb"),
    T(f"η = 0.03:  {LR[0.03]['best']:,} USD con {LR[0.03]['n']} árboles".replace(",", "."), 960, 390, tam=48, c="verde", w="sb", en="a"),
    P("50 USD menos · 35 % más árboles", 960, 540, "ambar", 52, en="b"),
    P("η moderado + la validación elige los árboles", 960, 700, "azul", 48, en="c"), real(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]eta es el tamaño del paso, y achicarlo exige más árboles. [[b]]Eta grande aprende rápido pero se queda corto; eta chico con pocos árboles subajusta. [[c]]Se ajustan en pareja, y el número de árboles conviene que lo elija la validación.",
    CL(), P("η chico → más árboles", 960, 300, "ambar", 52, en="a"), P("η grande: corto · η chico con pocos árboles: subajuste", 960, 460, "rojo", 44, en="b"),
    P("en pareja · la validación elige los árboles", 960, 620, "verde", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]XGBoost, LightGBM y CatBoost. Gradient Boosting, con ingeniería.",
    CL(), siguiente(8, "XGBoost, LightGBM y CatBoost"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
