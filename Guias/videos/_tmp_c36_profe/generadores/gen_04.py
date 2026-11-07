import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo

OUT = out("guion_c36_04_adavoto.json")
v = V("AdaBoost: el voto y los datos reales")
A = D["ada70"]
AD = FG["ada"]
R1, R2, R3 = A["rondas"]
XP, YP = A["X"], A["y"]

# lienzo del grafico de dispersion (px) : edad 30-80, frecuencia cardiaca max 60-200
GX0, GY0, GX1, GY1 = 230, 230, 1110, 870


def px(age):
    return GX0 + (age - 30) / 50 * (GX1 - GX0)


def py(hr):
    return GY1 - (hr - 60) / 140 * (GY1 - GY0)


def dia(w):
    return round(20 * (w * 70) ** 0.5, 1)


v.portada("AdaBoost: el voto", "Tres rondas con los pacientes de Heart, y un voto que no es por mayoría",
          "Cuarta parada: el voto de AdaBoost. Y tres rondas reales con setenta pacientes del dataset Heart.")

# ---- 1. Hook
v.b("Tres árboles votan. [[a]]El primero dice menos uno; el segundo, más uno; el tercero, más uno. [[b]]Por mayoría, ganaría más uno. [[c]]Pero AdaBoost puede decir menos uno. ¿Cómo?",
    CL(), *[[Bc(560 + 400 * i, 380, 300, 180, "blanco", c, 5, 26, sombra=True, en="a", t=0.3 * i),
             T(f"árbol {i + 1}", 560 + 400 * i, 345, tam=40, c="gris", en="a", t=0.3 * i + 0.1),
             T(vt, 560 + 400 * i, 410, tam=60, c=c, en="a", t=0.3 * i + 0.2)] for i, (vt, c) in enumerate([("−1", "azul"), ("+1", "rojo"), ("+1", "rojo")])],
    T("por mayoría: +1", 960, 620, tam=56, c="rojo", w="sb", en="b"), P("AdaBoost: puede decir −1", 960, 760, "azul", 56, en="c"), sec="1. Un voto raro")

# ---- 2. Analogía
v.b("Piensa en tres meteorólogos. [[a]]Uno acierta casi siempre; los otros dos, apenas más que tirar una moneda. [[b]]Si el bueno dice que llueve y los otros dos dicen que no, ¿a quién le crees? [[c]]Al que tiene mejor historial. Su voto pesa más.",
    CL(), T("tres meteorólogos", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "verde", "acierta casi siempre", "dice: llueve", "a"), (960, "gris", "casi una moneda", "dice: no llueve", "a"),
              (1520, "gris", "casi una moneda", "dice: no llueve", "a")], y=470, w=480, h=300, tam_t=40, tam_d=38),
    T("¿a quién le crees?", 960, 730, tam=60, c="tinta", en="b"), P("al de mejor historial: su voto pesa más", 960, 860, "verde", 48, en="c"),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía")

# ---- 3. La fórmula del voto
v.b("El voto final de AdaBoost hace exactamente eso. [[a]]Cada árbol vota más uno o menos uno, y su voto se multiplica por su alfa. [[b]]Se suma todo, y la clase final es el signo de esa suma: positiva, más uno; negativa, menos uno.",
    CL(), T("el voto ponderado", 960, 160, tam=52, c="tinta", ent="subir"),
    caja_formula("H(x) = signo( α₁·h₁(x) + α₂·h₂(x) + … )", y=360, c="violeta", tam=56, w=1300, en="a"),
    T("hₘ(x) = +1 o −1  ·  αₘ = peso del árbol", 960, 520, tam=42, c="tinta", w="sb", en="a", t=0.6),
    T("suma > 0 → +1   ·   suma < 0 → −1", 960, 640, tam=46, c="azul", w="sb", en="b"), sec="3. El voto ponderado")

# ---- 4. Ejemplo resuelto (apunte)
v.b("Volvamos a los tres árboles, con el ejemplo del apunte. [[a]]Sus alfas son 0.9, 0.4 y 0.3. [[b]]El primero vota menos uno: aporta menos 0.9. [[c]]Los otros dos votan más uno: aportan 0.4 más 0.3, o sea 0.7.",
    CL(), T("tres árboles · ejemplo del apunte", 960, 150, tam=48, c="tinta", ent="subir"),
    *[[P(f"α = {a}", 560 + 400 * i, 280, c, 46, en="a", t=0.2 * i), T(f"vota {vt}", 560 + 400 * i, 360, tam=40, c=c, w="sb", en="a", t=0.2 * i + 0.1)]
      for i, (a, vt, c) in enumerate([("0.9", "−1", "azul"), ("0.4", "+1", "rojo"), ("0.3", "+1", "rojo")])],
    T("0.9 × (−1) =", 900, 500, tam=48, c="tinta", w="sb", anc="rm", en="b"), T("−0.9", 930, 500, tam=56, c="azul", anc="lm", en="b", t=0.4),
    T("0.4 × (+1) + 0.3 × (+1) =", 900, 600, tam=48, c="tinta", w="sb", anc="rm", en="c"), T("+0.7", 930, 600, tam=56, c="rojo", anc="lm", en="c", t=0.5),
    ilustrativo(1000, texto="ejemplo del apunte"), sec="4. Ejemplo: el voto")
v.b("La suma: menos 0.9 más 0.7, [[a]]igual a menos 0.2. [[b]]Es negativa, así que gana menos uno, aunque haya más votos a favor de más uno. [[c]]Un árbol confiable puede ganarle a dos mediocres.",
    T("suma = −0.9 + 0.7 =", 900, 720, tam=52, c="tinta", w="sb", anc="rm"), T("−0.2", 930, 720, tam=64, c="azul", anc="lm", en="a"),
    P("negativa → gana −1", 960, 850, "azul", 50, en="b"), T("un árbol confiable le gana a dos mediocres", 960, 950, tam=38, c="gris", w="sb", en="c"))

# ---- 5. Datos reales: tres rondas con 70 pacientes
puntos = []
for i, ((age, hr), yy) in enumerate(zip(XP, YP)):
    c = "rojo" if yy == 1 else "azul"
    d0 = dia(R1["w"][i])
    puntos.append(Bc(px(age), py(hr), d0, d0, c, "blanco", 2, d0 / 2, id=f"p{i}", en="a", t=0.01 * i, ent="pop"))
v.b("Ahora con datos reales. [[a]]Setenta pacientes del dataset Heart, con dos variables: la edad y la frecuencia cardíaca máxima. [[b]]En rojo, los que tienen enfermedad cardíaca; en azul, los que no. Todos empiezan con el mismo peso: uno sobre setenta.",
    CL(), T("Heart · 70 pacientes · AdaBoost con stumps", 960, 140, tam=46, c="tinta", ent="subir"),
    G(GX0, GY0, GX1, GY1, 30, 80, 60, 200, [], xticks=[30, 40, 50, 60, 70, 80], yticks=[60, 100, 140, 180], yfmt="{:.0f}",
      xlabel="edad", ylabel="frec. máx.", id="gs"),
    *puntos, P("rojo: enfermedad · azul: no", 1520, 300, "gris", 38, en="b"), T("tamaño del punto = peso", 1520, 380, tam=36, c="gris", w="sb", en="b", t=0.4),
    real(1050), sec="5. Tres rondas reales")


def corte(r, en, t=0.0, id="corte"):
    if r["feature"] == 0:
        return L(px(r["thr"]), GY0, px(r["thr"]), GY1, "ambar", 7, en=en, t=t, id=id)
    return L(GX0, py(r["thr"]), GX1, py(r["thr"]), "ambar", 7, en=en, t=t, id=id)


v.b("Ronda uno. [[a]]El stump elige un corte: edad menor que 54.5. [[b]]Se equivoca en 19 de los 70 pacientes. Como todos pesan lo mismo, épsilon es 19 sobre 70: 0.271.",
    corte(R1, "a", id="c1"), T("edad < 54.5", px(R1["thr"]) + 10, GY0 + 30, tam=32, c="ambar", w="b", anc="lm", en="a", t=0.3),
    *[U(f"p{i}", borde="ambar", grosor=5, en="b", t=0.01 * k) for k, i in enumerate(R1["mal"])],
    T("ronda 1", 1520, 470, tam=44, c="tinta", en="b"), T("ε = 19 / 70 =", 1560, 550, tam=42, c="tinta", w="sb", anc="rm", en="b", t=0.5),
    N(R1["err"], 1580, 550, "{:.3f}", 50, "rojo", anc="lm", en="b", t=0.9))
v.b("Su alfa: uno menos épsilon, sobre épsilon, es 51 sobre 19: 2.684. [[a]]El logaritmo natural es 0.987, y la mitad: alrededor de 0.49.",
    T("(1 − ε)/ε = 51/19 = 2.684", 1520, 630, tam=34, c="tinta", w="sb"),
    T("ln 2.684 = 0.987 → α = 0.987 / 2 ≈", 1600, 700, tam=32, c="tinta", w="sb", anc="rm", en="a"), N(0.49, 1620, 700, "{:.2f}", 46, "violeta", anc="lm", en="a", t=0.6))
v.b("Ronda dos. [[a]]Los 19 errores crecen, y los aciertos se achican. [[b]]El nuevo stump corta por la frecuencia cardíaca: menor que 147.5. [[c]]Falla en 21 pacientes. Contando, serían 21 de 70, 0.3. Pero el error ponderado es 0.338: falló en casos que pesaban más.",
    X("c1"), *[U(f"p{i}", w=dia(R2["w"][i]), h=dia(R2["w"][i]), radio=dia(R2["w"][i]) / 2, borde="blanco", grosor=2, en="a", dur=1.0) for i in range(70)],
    corte(R2, "b", id="c2"), T("frec. < 147.5", GX1 - 10, py(R2["thr"]) - 28, tam=32, c="ambar", w="b", anc="rm", en="b", t=0.3),
    *[U(f"p{i}", borde="ambar", grosor=5, en="c", t=0.01 * k) for k, i in enumerate(R2["mal"])],
    T("ronda 2", 1520, 800, tam=44, c="tinta", en="c"), T("21 / 70 = 0.3, pero ε =", 1580, 870, tam=34, c="tinta", w="sb", anc="rm", en="c", t=1.5),
    N(R2["err"], 1600, 870, "{:.3f}", 46, "rojo", anc="lm", en="c", t=2.5))
v.b("Su alfa baja a 0.34. [[a]]Ronda tres: los pesos se reacomodan otra vez, y el stump corta en una frecuencia de 177.5. [[b]]Su error ya es 0.453, muy cerca de 0.5, y su alfa apenas 0.09.",
    T("α = 0.34", 1520, 940, tam=40, c="violeta", w="b"),
    X("c2"), *[U(f"p{i}", w=dia(R3["w"][i]), h=dia(R3["w"][i]), radio=dia(R3["w"][i]) / 2, borde="blanco", grosor=2, en="a", dur=1.0) for i in range(70)],
    corte(R3, "a", t=1.0, id="c3"), *[U(f"p{i}", borde="ambar", grosor=5, en="b", t=0.01 * k) for k, i in enumerate(R3["mal"])],
    Bc(1520, 640, 660, 780, "fondo", None, 0, 0, en="a", ent="fade"),
    T("ronda 3", 1520, 470, tam=44, c="tinta", en="a", t=0.2), T("ε =", 1500, 560, tam=42, c="tinta", w="sb", anc="rm", en="b"),
    N(R3["err"], 1520, 560, "{:.3f}", 50, "rojo", anc="lm", en="b", t=0.3), T("α =", 1500, 650, tam=42, c="tinta", w="sb", anc="rm", en="b", t=0.8),
    N(R3["alpha"], 1520, 650, "{:.2f}", 50, "violeta", anc="lm", en="b", t=1.1))
v.b("El patrón es típico. [[a]]Error de 0.271, 0.338, 0.453. Alfa de 0.49, 0.34, 0.09. [[b]]A medida que solo quedan casos difíciles, el error se acerca a 0.5, y cada árbol nuevo aporta menos.",
    CL(), T("las tres rondas", 960, 160, tam=52, c="tinta", ent="subir"),
    *[[T(f"ronda {k + 1}", 560 + 400 * k, 300, tam=44, c="tinta", en="a", t=0.3 * k),
       P(f"ε = {AD['err'][k]:.3f}", 560 + 400 * k, 400, "rojo", 46, en="a", t=0.3 * k + 0.1),
       P(f"α = {AD['alpha'][k]:.2f}", 560 + 400 * k, 500, "violeta", 46, en="a", t=0.3 * k + 0.2)] for k in range(3)],
    P("quedan los casos difíciles: ε → 0.5 y α → 0", 960, 680, "ambar", 48, en="b"), real(1050))
v.b("Con treinta stumps, el voto ponderado clasifica bien a 52 de los 70 pacientes: 0.74. [[a]]Y la frontera que dibuja ese voto es escalonada: cada stump aporta un corte recto, y la suma de muchos cortes forma una escalera.",
    P("30 stumps: 52 de 70 bien clasificados = 0.74", 960, 830, "verde", 46),
    T("frontera escalonada: la suma de muchos cortes rectos", 960, 930, tam=38, c="gris", w="sb", en="a"))

# ---- 6. Ventajas y limitaciones (diapositiva 8)
v.b("La diapositiva ocho hace el balance. [[a]]Ventajas: intuición clara y pocos hiperparámetros, buen desempeño con aprendices muy simples, no exige escalar variables, y sirve para clasificación y regresión. [[b]]Limitaciones: es sensible al ruido y a las etiquetas incorrectas, el entrenamiento es secuencial, y es menos flexible que las variantes modernas.",
    CL(), T("AdaBoost · diapositiva 8", 960, 150, tam=50, c="tinta", ent="subir"),
    Tj(540, 540, 760, 560, "VENTAJAS", ["intuición clara, pocos hiperparámetros", "bien con aprendices muy simples", "no exige escalar variables",
                                         "clasificación y regresión"], c="verde", en="a", tam=34),
    Tj(1380, 540, 760, 560, "LIMITACIONES", ["sensible a ruido y etiquetas malas", "entrenamiento secuencial", "menos flexible que las variantes modernas"],
       c="rojo", en="b", tam=34), sec="6. Ventajas y limitaciones")
v.b("Y su consejo práctico: [[a]]úsalo como línea base fuerte, cuando los datos están relativamente limpios. [[b]]Ojo con el nombre: Ada viene de adaptive, adaptativo, porque los pesos se adaptan en cada ronda.",
    P("línea base fuerte con datos limpios", 960, 900, "verde", 46, en="a"), T("AdaBoost = Adaptive Boosting", 960, 990, tam=38, c="gris", w="sb", en="b"))

# ---- 7. Por qué el ruido le hace daño
v.b("¿Por qué el ruido le hace tanto daño? [[a]]Imagina un paciente con la etiqueta mal puesta. Ningún árbol razonable lo clasifica bien, así que falla en todas las rondas. [[b]]Con alfa cerca de 0.49, cada ronda multiplica su peso por e elevado a 0.49, alrededor de 1.63. [[c]]Tras cinco rondas, 1.63 elevado a cinco: su peso se multiplicó por 11.5, sin contar la normalización.",
    CL(), T("una etiqueta mal puesta", 960, 160, tam=52, c="tinta", ent="subir"),
    T("falla en TODAS las rondas", 960, 280, tam=46, c="rojo", w="sb", en="a"),
    T("cada ronda: × e^0.49 ≈ × 1.63", 960, 390, tam=46, c="tinta", w="sb", en="b"),
    *[Bx(420 + 230 * k, 860 - 1.63 ** k * 34, 560 + 230 * k, 860, "rojo", None, 0, 10, grow="y", en="c", t=0.25 * k) for k in range(6)],
    *[T(f"{1.63 ** k:.1f}" if k else "1", 490 + 230 * k, 860 - 1.63 ** k * 34 - 30, tam=32, c="tinta", w="b", en="c", t=0.25 * k + 0.3) for k in range(6)],
    *[T(f"ronda {k}", 490 + 230 * k, 900, tam=28, c="gris", w="sb", en="c") for k in range(6)], ilustrativo(1000), sec="7. El daño del ruido")
v.b("El modelo se obsesiona con un caso que nunca va a poder acertar. [[a]]Por eso, antes de usar boosting, revisa las etiquetas.",
    P("se obsesiona con un caso imposible → revisa las etiquetas", 960, 990, "ambar", 40, en="a"))

# ---- 8. En Python
v.b("En Python, AdaBoost es una línea. [[a]]El notebook usa cien árboles con un learning_rate de 0.05. [[b]]Por defecto son cincuenta árboles, y el modelo base es un stump. [[c]]El learning_rate achica el alfa de cada árbol: igual que eta en el resto de la clase.",
    CL(), T("AdaBoost en scikit-learn", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["from sklearn.ensemble import AdaBoostClassifier", "ada = AdaBoostClassifier(n_estimators=100,", "                         learning_rate=0.05,",
            "                         random_state=42)"], y=420, w=1400, tam=38, en="a", dt=0.3),
    T("por defecto: n_estimators = 50 y un árbol de profundidad 1", 960, 660, tam=38, c="gris", w="sb", en="b"),
    P("learning_rate achica el α de cada árbol", 960, 770, "ambar", 46, en="c"), sec="8. En Python")

# ---- 9. Error típico
v.b("El error típico: [[a]]codificar las clases como cero y uno, y aplicar la fórmula del voto. [[b]]Con cero y uno, un árbol que vota cero no aporta nada, y el signo de la suma siempre sale positivo. [[c]]La fórmula del voto necesita clases menos uno y más uno. scikit-learn se encarga de eso por dentro; a mano, te toca a ti.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("clases 0 y 1 en la fórmula del voto", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("un voto 0 no aporta · la suma nunca es negativa", 960, 420, tam=44, c="tinta", w="sb", en="b"),
    P("la fórmula usa clases −1 y +1", 960, 580, "verde", 50, en="c"), T("scikit-learn lo hace por dentro; a mano, tú", 960, 690, tam=38, c="gris", w="sb", en="c", t=0.6),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo, con un ejercicio del cuaderno. [[a]]Cuatro stumps, con alfas 0.8, 0.5, 0.45 y 0.2, votan más uno, menos uno, menos uno y más uno. [[b]]Hay empate dos a dos. ¿Quién gana?",
    CL(), *[[P(f"α = {a}", 400 + 373 * i, 300, c, 44, en="a", t=0.2 * i), T(f"vota {vt}", 400 + 373 * i, 380, tam=40, c=c, w="sb", en="a", t=0.2 * i + 0.1)]
            for i, (a, vt, c) in enumerate([("0.8", "+1", "rojo"), ("0.5", "−1", "azul"), ("0.45", "−1", "azul"), ("0.2", "+1", "rojo")])],
    T("2 votos contra 2: ¿quién gana?", 960, 560, tam=64, c="tinta", en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Suma: 0.8 menos 0.5, menos 0.45, más 0.2, [[a]]igual a más 0.05. [[b]]Gana más uno, por muy poco: el primer stump, el más confiable, desempata.",
    CL(), T("0.8 − 0.5 − 0.45 + 0.2 =", 960, 330, tam=56, c="tinta", w="sb", anc="rm"), T("+0.05", 990, 330, tam=68, c="rojo", anc="lm", en="a"),
    P("gana +1: desempata el stump más confiable (α = 0.8)", 960, 520, "rojo", 46, en="b"), ilustrativo(1000, texto="ejercicio del cuaderno"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el voto de AdaBoost es una suma ponderada por alfa, no una votación por mayoría. [[b]]Con datos reales, el error de cada ronda sube hacia 0.5 y alfa baja. [[c]]Y su punto débil es el ruido: una etiqueta mala acumula peso ronda tras ronda.",
    CL(), P("voto = signo de Σ α·h: no es por mayoría", 960, 300, "violeta", 48, en="a"), P("ronda a ronda: ε → 0.5, α → 0", 960, 460, "ambar", 48, en="b"),
    P("punto débil: el ruido acumula peso", 960, 620, "rojo", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]Gradient Boosting. En vez de pesos, cada árbol recibe directamente el error: los residuos.",
    CL(), siguiente(5, "Gradient Boosting: corregir residuos"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
