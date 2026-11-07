import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, nube, recta, serie_curva

OUT = out("guion_c37_03_logistica.json")
v = V("Regresión logística: del puntaje a la probabilidad")
LD = D["logistica_defecto"]
LG = D["script"]["Logistica"]


def sig(x):
    return 1 / (1 + math.exp(-x))


v.portada("Regresión logística", "Un puntaje lineal, una curva en S y una probabilidad",
          "Tercera parada: la regresión logística. Vas a calcular a mano una probabilidad, y a ver por qué el umbral decide, pero no cambia la probabilidad.")

# ---- 1. Hook
v.b("¿Por qué no usar una regresión lineal común para predecir el cero y el uno? [[a]]Porque una recta no sabe detenerse. [[b]]Con valores grandes predice más que uno, y con valores chicos, menos que cero. [[c]]Con los datos de Wisconsin, una regresión lineal predice desde menos 0.32 hasta 1.58. Eso no es una probabilidad.",
    CL(), T("¿y si usamos una regresión lineal?", 960, 160, tam=54, c="tinta", ent="subir"),
    G(420, 290, 1500, 800, -4, 4, -0.5, 1.5, [dict(xs=[-4, 4], ys=[1.0, 1.0], color="gris", grosor=3, revela=1.0, lapiz=False),
                                             dict(xs=[-4, 4], ys=[0.0, 0.0], color="gris", grosor=3, revela=1.0, lapiz=False),
                                             dict(xs=[-4, 4], ys=[-0.45, 1.45], color="ambar", grosor=6, revela=0.0)],
      xticks=[-4, -2, 0, 2, 4], yticks=[-0.5, 0, 0.5, 1, 1.5], yfmt="{:.1f}", xlabel="puntaje", ylabel="ŷ", id="gl", en="a"),
    U("gl", **{"series.2.revela": 1.0}, en="a", t=0.3, dur=1.5),
    P("> 1", 1640, 330, "rojo", 44, en="b"), P("< 0", 1640, 760, "rojo", 44, en="b", t=0.4),
    P(f"en prueba: de {D['lineal01']['min']:.2f} a {D['lineal01']['max']:.2f}", 960, 985, "ambar", 44, en="c"),
    real(1050, en="c"), sec="1. Una recta no se detiene")

# ---- 2. Analogía: el dimmer
v.b("Piensa en un regulador de luz, un dimmer. [[a]]Al principio, girar la perilla cambia mucho la luz. [[b]]Pero cerca del máximo, por más que gires, la luz ya no sube: se queda en el cien por ciento. [[c]]La regresión logística hace eso con un puntaje: lo convierte en un número que nunca baja de cero ni pasa de uno.",
    CL(), T("el regulador de luz", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "en el medio", "girar cambia mucho la luz", "a"), (960, "ambar", "cerca del máximo", "la luz ya no sube", "b"),
              (1520, "violeta", "la logística", "puntaje → número entre 0 y 1", "c")], y=500, w=480, h=330, tam_t=46, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el regulador de luz")

# ---- 3. El modelo
v.b("La diapositiva quince escribe el modelo. [[a]]Lo que se modela es la probabilidad de la clase positiva: pe de x, la probabilidad de que y sea uno. [[b]]Su logaritmo de los odds, pe sobre uno menos pe, es una combinación lineal: beta cero más beta por x. [[c]]Al despejar, pe es uno sobre uno más e elevado a menos ese puntaje.",
    CL(), T("el modelo · diapositiva 15", 960, 150, tam=50, c="tinta", ent="subir"),
    P("p(x) = P(Y = 1 | X = x)", 960, 300, "azul", 52, en="a"),
    caja_formula("log( p / (1 − p) ) = β₀ + βᵀx = η", y=470, c="violeta", tam=54, w=1200, en="b", wf="sb"),
    caja_formula("p = 1 / (1 + e^(−η))", y=680, c="teal", tam=60, w=900, en="c", wf="sb"), sec="3. El modelo")
v.b("A ese puntaje lineal lo llamamos eta. [[a]]El modelo es lineal en los coeficientes, como una regresión. [[b]]Pero la probabilidad que sale queda siempre entre cero y uno.",
    P("η = puntaje lineal", 960, 860, "violeta", 46, en="a"), T("lineal en los coeficientes · probabilidad entre 0 y 1", 960, 960, tam=38, c="gris", w="sb", en="b"))

# ---- 4. La curva en S
ETA = [-6 + 0.1 * i for i in range(121)]
v.b("Ahora, la curva. [[a]]En el eje horizontal, el puntaje eta; en el vertical, la probabilidad. [[b]]Cuando eta vale cero, la probabilidad es exactamente 0.5. [[c]]Y cuando eta vale uno, como marca la diapositiva dieciséis, la probabilidad es 0.731.",
    CL(), T("la función logística · diapositiva 16", 960, 140, tam=50, c="tinta", ent="subir"),
    G(360, 250, 1560, 850, -6, 6, 0, 1, [dict(xs=ETA, ys=[sig(e) for e in ETA], color="teal", grosor=7, revela=0.0)],
      xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[0, 0.25, 0.5, 0.75, 1], yfmt="{:.2f}", xlabel="puntaje lineal η", ylabel="p", id="gs", en="a"),
    U("gs", **{"series.0.revela": 1.0}, en="a", t=0.3, dur=2.0),
    Bc(0, 0.5, 26, 26, "violeta", None, 0, 13, ref="gs", en="b"), T("η = 0 → p = 0.5", 0.3, 0.42, tam=36, c="violeta", w="sb", anc="lm", ref="gs", en="b", t=0.2),
    Bc(1, sig(1), 26, 26, "ambar", None, 0, 13, ref="gs", en="c"), T("η = 1 → p = 0.731", 1.3, 0.66, tam=36, c="ambar", w="sb", anc="lm", ref="gs", en="c", t=0.2),
    sec="4. La curva en S")
v.b("Y aquí está la clave de la curva: [[a]]subir eta de cero a uno aumenta la probabilidad en 0.23. [[b]]Pero subir eta de cuatro a cinco la aumenta apenas en 0.01. [[c]]Un aumento constante del puntaje no produce un aumento constante de la probabilidad.",
    T("0 → 1:  +0.23", 1740, 560, tam=40, c="teal", w="sb", en="a"), Bc(4, sig(4), 22, 22, "rojo", None, 0, 11, ref="gs", en="b"),
    Bc(5, sig(5), 22, 22, "rojo", None, 0, 11, ref="gs", en="b", t=0.1), T("4 → 5:  +0.01", 1740, 330, tam=40, c="rojo", w="sb", en="b", t=0.3),
    P("mismo paso en η ≠ mismo paso en p", 960, 1000, "ambar", 42, en="c"))

# ---- 5. Paso a paso (diapositiva 17)
v.b("Hagamos una predicción paso a paso, con el ejemplo de la diapositiva diecisiete. [[a]]Los coeficientes son hipotéticos: eta es menos uno, más 0.8 por zeta uno, más 0.4 por zeta dos. [[b]]Y el caso tiene zeta uno igual a 1.5 y zeta dos igual a dos, en variables estandarizadas.",
    CL(), T("una predicción a mano · diapositiva 17", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("η = −1 + 0.8·z₁ + 0.4·z₂", y=310, c="violeta", tam=56, w=1000, en="a", wf="sb"),
    P("z₁ = 1.5", 760, 470, "azul", 50, en="b"), P("z₂ = 2", 1160, 470, "azul", 50, en="b", t=0.3),
    ilustrativo(1000, texto="ejemplo de la diapositiva 17"), sec="5. Paso a paso")
v.b("Paso uno, el puntaje: [[a]]menos uno, más 0.8 por 1.5, que es 1.2, más 0.4 por dos, que es 0.8. [[b]]Menos uno más 1.2 más 0.8: eta vale uno. [[c]]Paso dos, la probabilidad: uno sobre uno más e elevado a menos uno. e a la menos uno es 0.368; uno sobre 1.368 da 0.731.",
    T("η = −1 + 0.8·1.5 + 0.4·2 = −1 + 1.2 + 0.8 =", 1240, 610, tam=44, c="tinta", w="sb", anc="rm", en="a"), N(1, 1270, 610, "{:.0f}", 58, "violeta", anc="lm", en="b", t=0.2),
    T("p = 1 / (1 + e^(−1)) = 1 / (1 + 0.368) = 1 / 1.368 =", 1240, 740, tam=42, c="tinta", w="sb", anc="rm", en="c"), N(0.731, 1270, 740, "{:.3f}", 58, "teal", anc="lm", en="c", t=1.5))
v.b("Paso tres, la decisión. [[a]]Con el umbral habitual, te igual a 0.5: 0.731 es mayor que 0.5, y la clase es uno. [[b]]Con un umbral más exigente, te igual a 0.8: 0.731 es menor que 0.8, y la clase es cero. [[c]]Fíjate: la probabilidad no cambió. Lo que cambió fue la decisión.",
    CL(), T("p = 0.731", 960, 200, tam=80, c="teal"),
    Bc(560, 470, 640, 240, "blanco", "verde", 5, 30, sombra=True, en="a"), T("t = 0.5", 560, 420, tam=48, c="verde", en="a", t=0.1),
    T("0.731 ≥ 0.5 → clase 1", 560, 510, tam=40, c="tinta", w="sb", en="a", t=0.3),
    Bc(1360, 470, 640, 240, "blanco", "rojo", 5, 30, sombra=True, en="b"), T("t = 0.8", 1360, 420, tam=48, c="rojo", en="b", t=0.1),
    T("0.731 < 0.8 → clase 0", 1360, 510, tam=40, c="tinta", w="sb", en="b", t=0.3),
    P("la probabilidad no cambia; cambia la DECISIÓN", 960, 760, "ambar", 52, en="c"), ilustrativo(1000, texto="ejemplo de la diapositiva 17"))

# ---- 6. La pérdida logística
v.b("¿Y de dónde salen los coeficientes? [[a]]Se eligen minimizando la pérdida logística, de la diapositiva diecinueve. [[b]]Para cada caso, si es positivo, se suma menos el logaritmo de pe; si es negativo, menos el logaritmo de uno menos pe.",
    CL(), T("la pérdida logística · diapositiva 19", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("pérdida = − Σ [ yᵢ·log(pᵢ) + (1 − yᵢ)·log(1 − pᵢ) ]", y=320, c="rojo", tam=50, w=1500, en="a", wf="sb"),
    P("y = 1 → −log(p)", 680, 500, "azul", 46, en="b"), P("y = 0 → −log(1 − p)", 1260, 500, "azul", 46, en="b", t=0.4),
    sec="6. La pérdida logística")
v.b("Veamos cuánto cuesta un caso positivo, según la probabilidad que le dio el modelo. [[a]]Si le dio 0.9: menos el logaritmo de 0.9, 0.105. Poco. [[b]]Si le dio 0.5: 0.693. [[c]]Y si le dio 0.01, muy seguro y equivocado: 4.605. Una predicción muy segura y equivocada se paga caro.",
    T("p = 0.9  →  −log 0.9 =", 960, 650, tam=44, c="tinta", w="sb", anc="rm", en="a"), N(0.105, 990, 650, "{:.3f}", 54, "verde", anc="lm", en="a", t=0.5),
    T("p = 0.5  →  −log 0.5 =", 960, 750, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(0.693, 990, 750, "{:.3f}", 54, "ambar", anc="lm", en="b", t=0.5),
    T("p = 0.01  →  −log 0.01 =", 960, 850, tam=44, c="tinta", w="sb", anc="rm", en="c"), N(4.605, 990, 850, "{:.3f}", 54, "rojo", anc="lm", en="c", t=0.5),
    P("seguro y equivocado → se paga caro", 1500, 960, "rojo", 40, en="c", t=1.5))

# ---- 7. La frontera
L2 = D["log2"]
Z = D["z2_train"]
xs_r, ys_r = recta(L2["b"], L2["b0"], 0.0, -1.8, 3.4)
v.b("Con el umbral 0.5, la regla se puede escribir sin probabilidades: [[a]]pe es mayor o igual que 0.5 exactamente cuando eta es mayor o igual que cero. [[b]]Con dos predictores, eso es una recta. Aquí están los 398 casos de entrenamiento, con radio y concavidad estandarizados. [[c]]Y esta es la frontera de una logística ajustada solo con esas dos variables.",
    CL(), T("la frontera: η = 0", 960, 140, tam=50, c="tinta", ent="subir"),
    P("p ≥ 0.5  equivale a  η ≥ 0", 1640, 300, "violeta", 36, en="a"),
    G(400, 230, 1380, 900, -2.1, 4.6, -1.5, 4.4, nube((Z["x"], Z["y"]), Z["c"], mk=6) + [serie_curva(xs_r, ys_r, "tinta", 6)],
      xticks=[-2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="radio (estandarizado)", ylabel="concav.", id="gf", en="b", tam_txt=28),
    U("gf", **{"series.0.revela": 1.0, "series.1.revela": 1.0}, en="b", t=0.3, dur=1.6),
    U("gf", **{"series.2.alfa": 1.0}, en="c", dur=0.8),
    T("benignos", 1620, 460, tam=36, c="azul", w="sb", en="b", t=0.5), T("malignos", 1620, 530, tam=36, c="rojo", w="sb", en="b", t=0.7),
    T("recta: η = 0", 1620, 640, tam=36, c="tinta", w="sb", en="c"), T(f"prueba: {100 * L2['acc_test']:.1f} %", 1620, 720, tam=36, c="teal", w="sb", en="c", t=0.5),
    real(1050, en="b"), sec="7. La frontera")
v.b("Con más predictores, la recta pasa a ser un hiperplano. [[a]]Y si se agregan interacciones o transformaciones, la frontera puede curvarse en las variables originales. [[b]]En la clase se usan solo efectos lineales, para que la comparación sea fácil de seguir.",
    P("más predictores → hiperplano", 1620, 830, "azul", 34, en="a"), T("interacciones → fronteras curvas", 1620, 900, tam=30, c="gris", w="sb", en="a", t=0.5),
    T("la clase: solo efectos lineales", 1620, 960, tam=30, c="gris", w="sb", en="b"))

# ---- 8. En R y en Python
v.b("En el código de la clase, con las cuatro variables. [[a]]En R: glm, con la fórmula maligno contra las cuatro variables y la familia binomial. Y predict con type igual a response entrega la probabilidad de maligno. [[b]]En Python: Logistic Regression con penalty igual a None, y predict proba, quedándose con la segunda columna.",
    CL(), T("la logística en el código de la clase", 960, 140, tam=50, c="tinta", ent="subir"),
    codigo(["glm(maligno ~ radio + textura + suavidad + concavidad,", "    data = entrenamiento, family = binomial())",
            "predict(logistica, newdata = prueba, type = \"response\")"], y=330, w=1700, tam=32, c="teal", en="a"),
    codigo(["LogisticRegression(penalty=None, max_iter=2000)", "logistica.predict_proba(x_prueba_escalado)[:, 1]"], y=630, w=1700, tam=32, c="azul", en="b"),
    sec="8. En R y en Python")
PR = D["primeras_prob"]
v.b("Las primeras seis probabilidades de la prueba son estas. [[a]]0.9997, 0.9949, 0.9, 0.9915, 0.7447 y prácticamente uno. [[b]]Con el umbral 0.5, los seis casos se clasifican como malignos.",
    P("   ".join(f"{p:.4f}" for p in PR), 960, 850, "violeta", 40, en="a"), T("con t = 0.5 → los seis, clase 1", 960, 950, tam=38, c="gris", w="sb", en="b"),
    real(1050, en="a"))

# ---- 9. El error típico
v.b("El error típico: [[a]]escribir Logistic Regression con los paréntesis vacíos, creyendo que es la logística clásica. [[b]]Por defecto, scikit-learn agrega una penalización a los coeficientes. La diapositiva diecinueve lo advierte. [[c]]En la prueba se nota: sin penalización, 94.2 por ciento de exactitud; con la penalización por defecto, 93.0. No reproduce el glm de R.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("LogisticRegression() «a secas»", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("por defecto penaliza los coeficientes (diapositiva 19)", 960, 380, tam=40, c="tinta", w="sb", en="b"),
    T("penalty=None:", 1000, 530, tam=44, c="verde", w="sb", anc="rm", en="c"), N(100 * LG["acc"], 1030, 530, "{:.1f}", 56, "verde", suf=" %", anc="lm", en="c", t=0.3),
    T(f"por defecto:", 1000, 640, tam=44, c="rojo", w="sb", anc="rm", en="c", t=1.2), N(100 * LD["acc"], 1030, 640, "{:.1f}", 56, "rojo", suf=" %", anc="lm", en="c", t=1.5),
    P("con penalización no reproduce glm(…, family = binomial())", 960, 800, "ambar", 42, en="c", t=2.5), real(1050, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo, con el ejemplo del apunte. [[a]]Si eta es menos dos, más 0.9 por zeta uno, más 0.5 por zeta dos, y el caso tiene zeta uno igual a dos y zeta dos igual a uno, [[b]]¿cuál es su probabilidad? ¿Y su clase con umbral 0.7?",
    CL(), caja_formula("η = −2 + 0.9·z₁ + 0.5·z₂     z₁ = 2, z₂ = 1", y=320, c="violeta", tam=48, w=1400, en="a", wf="sb"),
    P("p = ?", 760, 520, "teal", 56, en="b"), P("clase con t = 0.7 = ?", 1260, 520, "ambar", 50, en="b", t=0.4),
    ilustrativo(1000, texto="ejemplo del apunte"), sec="10. Piensa un segundo", pausa=0.6)
v.b("El puntaje: menos dos más 1.8 más 0.5, [[a]]0.3. [[b]]La probabilidad: uno sobre uno más e a la menos 0.3, que es 0.741: uno sobre 1.741, [[c]]0.574. Con umbral 0.5 sería clase uno; con 0.7, clase cero.",
    CL(), T("η = −2 + 0.9·2 + 0.5·1 = −2 + 1.8 + 0.5 =", 1240, 300, tam=44, c="tinta", w="sb", anc="rm"), N(0.3, 1270, 300, "{:.1f}", 58, "violeta", anc="lm", en="a"),
    T("p = 1 / (1 + e^(−0.3)) = 1 / (1 + 0.741) =", 1240, 440, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(0.574, 1270, 440, "{:.3f}", 58, "teal", anc="lm", en="c"),
    P("t = 0.5 → clase 1   ·   t = 0.7 → clase 0", 960, 620, "ambar", 48, en="c", t=1.0), ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]la logística calcula un puntaje lineal, eta, y lo convierte en probabilidad con uno sobre uno más e a la menos eta. [[b]]El umbral decide la clase, sin cambiar la probabilidad. [[c]]Y en Python, para la logística clásica, hay que poner penalty igual a None.",
    CL(), P("η lineal → p = 1 / (1 + e^(−η))", 960, 300, "teal", 50, en="a"), P("el umbral decide; la probabilidad no cambia", 960, 460, "ambar", 48, en="b"),
    P("logística clásica en Python: penalty=None", 960, 620, "azul", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]el umbral con datos reales, los odds, y cómo leer los coeficientes sin caer en trampas.",
    CL(), siguiente(4, "Umbral, odds y coeficientes"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
