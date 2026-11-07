import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, flujo, casillas

OUT = out("guion_c36_03_adapesos.json")
v = V("AdaBoost: pesos, error y alfa")

v.portada("AdaBoost: pesos y error", "Cómo un árbol le señala al siguiente los casos difíciles",
          "Tercera parada: AdaBoost. Vas a hacer a mano una ronda completa: el error, el peso del árbol y los nuevos pesos.")

# ---- 1. Hook
v.b("En boosting, cada árbol tiene que decirle algo al siguiente. [[a]]En AdaBoost, ese mensaje es muy concreto: [[b]]fíjate en estos casos, que yo clasifiqué mal. [[c]]¿Cómo se transmite ese mensaje? Con pesos.",
    CL(), T("¿qué le dice un árbol al siguiente?", 960, 280, tam=58, c="tinta", ent="subir"),
    P("«fíjate en ESTOS casos: yo los clasifiqué mal»", 960, 470, "rojo", 54, en="b"), T("con PESOS", 960, 660, tam=90, c="ambar", en="c"),
    sec="1. El mensaje")

# ---- 2. Analogía
v.b("Piensa en un profesor que prepara ensayos para una prueba. [[a]]En el primer ensayo, todas las preguntas valen lo mismo. [[b]]Corrige, y ve en qué temas fallaste. [[c]]El siguiente ensayo trae más preguntas justo de esos temas. No te hace estudiar todo de nuevo: te hace estudiar lo que te costó.",
    CL(), T("los ensayos del profesor", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "ensayo 1", "todas las preguntas valen igual", "a"), (960, "rojo", "corrige", "¿en qué temas fallaste?", "b"),
              (1520, "ambar", "ensayo 2", "más preguntas de esos temas", "c")], y=490, w=480, h=330, tam_t=50, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: los ensayos")

# ---- 3. Diapositiva 6
v.b("La diapositiva seis muestra el ciclo de AdaBoost. [[a]]Al inicio, todos los casos tienen el mismo peso. [[b]]El árbol uno falla, digamos, en tres casos. [[c]]Se reponderan: esos casos pesan más. El árbol dos se concentra en ellos. Y al final, un voto, donde cada árbol pesa según su calidad.",
    CL(), T("el ciclo de AdaBoost · diapositiva 6", 960, 160, tam=50, c="tinta", ent="subir"),
    flujo([("pesos iguales", "teal", "a"), ("árbol 1 falla en 3", "rojo", "b"), ("reponderar", "ambar", "c"), ("árbol 2 se concentra", "azul", "c"),
           ("voto ponderado", "verde", "c")], y=460, w=300, h=190, x0=230, paso=365, tam=36),
    P("mayor error del árbol → menor influencia", 960, 700, "violeta", 44, en="c", t=1.5),
    P("caso mal clasificado → más peso en la siguiente ronda", 960, 800, "ambar", 44, en="c", t=2.5),
    ilustrativo(1000, texto="ejemplo de la diapositiva 6"), sec="3. El ciclo")

# ---- 4. El stump
v.b("¿Qué árboles usa AdaBoost? Normalmente, árboles de un solo corte, que se llaman stumps, tocones. [[a]]Por ejemplo: ¿la edad es menor que 55? Sí, a un lado; no, al otro. [[b]]Un stump solo es un aprendiz muy débil. [[c]]Y en scikit-learn, el modelo base por defecto de AdaBoost es justamente eso: un árbol de profundidad uno.",
    CL(), T("un stump: un árbol de un solo corte", 960, 160, tam=52, c="tinta", ent="subir"),
    Bc(960, 340, 420, 120, "blanco", "violeta", 5, 26, sombra=True, en="a"), T("¿edad < 55?", 960, 340, tam=46, c="violeta", en="a", t=0.1),
    L(860, 400, 640, 540, "gris", 5, en="a", t=0.3), L(1060, 400, 1280, 540, "gris", 5, en="a", t=0.3),
    Bc(640, 590, 260, 100, "azulclaro", "azul", 4, 20, en="a", t=0.5), T("sí: clase −1", 640, 590, tam=38, c="azul", en="a", t=0.6),
    Bc(1280, 590, 260, 100, "rojoclaro", "rojo", 4, 20, en="a", t=0.5), T("no: clase +1", 1280, 590, tam=38, c="rojo", en="a", t=0.6),
    P("un aprendiz muy débil", 960, 760, "ambar", 46, en="b"),
    T("scikit-learn: AdaBoostClassifier usa por defecto un árbol de profundidad 1", 960, 880, tam=34, c="gris", w="sb", en="c"),
    ilustrativo(1000), sec="4. Los stumps")

# ---- 5. El error ponderado
v.b("Primer ingrediente: el error del árbol, la letra épsilon. [[a]]No es la fracción de casos mal clasificados: es la suma de los pesos de esos casos, dividida por la suma de todos los pesos. [[b]]Es un número entre cero y uno: cero si no se equivoca nunca, uno si se equivoca en todo.",
    CL(), T("ingrediente 1: el error ponderado ε", 960, 160, tam=52, c="tinta", ent="subir"),
    caja_formula("ε = (suma de pesos de los errores) / (suma de todos los pesos)", y=360, c="rojo", tam=46, w=1600, en="a"),
    T("0 ≤ ε ≤ 1", 960, 540, tam=60, c="tinta", en="b"), T("0: nunca se equivoca  ·  1: se equivoca en todo", 960, 630, tam=40, c="gris", w="sb", en="b", t=0.4),
    sec="5. El error ε")

# ---- 6. Alfa
eps = [0.02 + 0.01 * i for i in range(97)]
alf = [0.5 * math.log((1 - e) / e) for e in eps]
v.b("Segundo ingrediente: el peso del árbol en el voto final, la letra alfa. [[a]]Se calcula como un medio del logaritmo natural de uno menos épsilon, sobre épsilon. [[b]]Mira la curva: con error chico, alfa es grande. Con error 0.5, el azar, alfa vale cero. Y con error mayor que 0.5, alfa es negativo.",
    CL(), T("ingrediente 2: el peso del árbol α", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("α = ½ · ln((1 − ε) / ε)", y=290, c="violeta", tam=56, w=900, en="a"),
    G(560, 420, 1400, 900, 0, 1, -2, 2, [dict(xs=eps, ys=alf, color="violeta", grosor=6, revela=0.0), dict(xs=[0, 1], ys=[0, 0], color="gris", grosor=3, revela=0.0, lapiz=False)],
      xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[-2, -1, 0, 1, 2], yfmt="{:.0f}", xlabel="error ε", ylabel="α", id="ga", en="b"),
    U("ga", **{"series.1.revela": 1.0}, en="b", t=0.3, dur=0.4), U("ga", **{"series.0.revela": 1.0}, en="b", t=0.5, dur=2.5),
    P("ε chico → α grande", 1650, 500, "verde", 36, en="b", t=0.8), P("ε = 0.5 → α = 0", 1650, 640, "gris", 36, en="b", t=1.8),
    P("ε > 0.5 → α < 0", 1650, 780, "rojo", 36, en="b", t=2.6), sec="6. El peso α")
v.b("Algunos valores, para tener la escala: [[a]]con error 0.1, alfa vale 1.10. [[b]]Con 0.3, vale 0.42. [[c]]Con 0.5, cero. La diapositiva lo resume en una frase: mayor error del árbol, menor influencia.",
    CL(), T("valores de α", 960, 160, tam=52, c="tinta", ent="subir"),
    T("ε = 0.1  →  α = ½ · ln 9 =", 1060, 330, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(1.10, 1090, 330, "{:.2f}", 60, "verde", anc="lm", en="a", t=0.5),
    T("ε = 0.3  →  α = ½ · ln(0.7/0.3) =", 1060, 460, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(0.42, 1090, 460, "{:.2f}", 60, "azul", anc="lm", en="b", t=0.5),
    T("ε = 0.5  →  α = ½ · ln 1 =", 1060, 590, tam=48, c="tinta", w="sb", anc="rm", en="c"), T("0", 1090, 590, tam=60, c="gris", anc="lm", en="c", t=0.4),
    P("mayor error del árbol → menor influencia", 960, 780, "violeta", 48, en="c", t=1.2))

# ---- 7. Actualizar pesos
v.b("Tercer ingrediente: actualizar los pesos. [[a]]Los casos mal clasificados multiplican su peso por e elevado a alfa: suben. [[b]]Los bien clasificados, por e elevado a menos alfa: bajan. [[c]]Y al final se normaliza, para que todos los pesos vuelvan a sumar uno.",
    CL(), T("ingrediente 3: actualizar los pesos", 960, 160, tam=52, c="tinta", ent="subir"),
    P("errores:  w × e^α   (suben)", 960, 340, "rojo", 52, en="a"), P("aciertos:  w × e^(−α)   (bajan)", 960, 480, "verde", 52, en="b"),
    P("normalizar: que sumen 1", 960, 620, "azul", 52, en="c"), sec="7. Actualizar los pesos")

# ---- 8-10. Ejemplo resuelto (apunte)
ERR = [2, 6]
XS = [330 + 140 * i for i in range(10)]
v.b("Ahora, una ronda completa a mano, con el ejemplo del apunte. [[a]]Diez observaciones, todas con peso 0.1. [[b]]El stump falla en dos de ellas.",
    CL(), T("una ronda de AdaBoost · 10 observaciones", 960, 150, tam=48, c="tinta", ent="subir"),
    *[Bx(x - 50, 780 - 0.1 * 1400, x + 50, 780, "azul", None, 0, 10, grow="y", en="a", t=0.05 * i, id=f"bw{i}") for i, x in enumerate(XS)],
    *[T("0.1", x, 780 - 0.1 * 1400 - 30, tam=30, c="tinta", w="b", en="a", t=0.05 * i + 0.4, id=f"tw{i}") for i, x in enumerate(XS)],
    *[T(str(i + 1), x, 820, tam=30, c="gris", w="sb", en="a") for i, x in enumerate(XS)],
    *[U(f"bw{i}", relleno="rojo", en="b", t=0.3) for i in ERR], *[K(XS[i], 860, "cruz", 44, en="b", t=0.5) for i in ERR],
    ilustrativo(1000, texto="ejemplo del apunte"), sec="8. Una ronda a mano")
v.b("Paso uno, el error: [[a]]los dos errores pesan 0.1 cada uno, así que épsilon es 0.2. [[b]]Paso dos, alfa: un medio del logaritmo de 0.8 sobre 0.2, o sea, un medio del logaritmo de cuatro. [[c]]El logaritmo de cuatro es 1.386, y la mitad: 0.693.",
    T("ε = 0.1 + 0.1 =", 640, 300, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(0.2, 670, 300, "{:.1f}", 58, "rojo", anc="lm", en="a", t=0.4),
    T("α = ½ · ln(0.8/0.2) = ½ · ln 4 = ½ · 1.386 =", 1420, 400, tam=44, c="tinta", w="sb", anc="rm", en="b"),
    N(0.693, 1450, 400, "{:.3f}", 58, "violeta", anc="lm", en="c", t=0.5))
v.b("Paso tres, los factores. [[a]]e elevado a 0.693 es 2: los errores duplican su peso, de 0.1 a 0.2. [[b]]e elevado a menos 0.693 es 0.5: los aciertos lo reducen a la mitad, de 0.1 a 0.05.",
    CL(), T("paso 3: multiplicar", 960, 150, tam=48, c="tinta", ent="subir"),
    T("errores:  0.1 × e^0.693 = 0.1 × 2 =", 1060, 290, tam=46, c="rojo", w="sb", anc="rm", en="a"), N(0.2, 1090, 290, "{:.1f}", 56, "rojo", anc="lm", en="a", t=0.6),
    T("aciertos:  0.1 × e^(−0.693) = 0.1 × 0.5 =", 1060, 400, tam=46, c="verde", w="sb", anc="rm", en="b"), N(0.05, 1090, 400, "{:.2f}", 56, "verde", anc="lm", en="b", t=0.6),
    *[Bx(x - 50, 900 - (0.2 if i in ERR else 0.05) * 1400, x + 50, 900, "rojo" if i in ERR else "verde", None, 0, 10, grow="y", en=("a" if i in ERR else "b"), t=0.3)
      for i, x in enumerate(XS)])
v.b("Pero ahora los pesos ya no suman uno. [[a]]Suman dos por 0.2, más ocho por 0.05: 0.4 más 0.4, igual a 0.8. [[b]]Paso cuatro, normalizar: dividimos todo por 0.8. [[c]]Los errores quedan en 0.2 sobre 0.8, igual a 0.25. Y los aciertos, en 0.05 sobre 0.8, igual a 0.0625.",
    CL(), T("paso 4: normalizar", 960, 150, tam=48, c="tinta", ent="subir"),
    T("suma = 2 × 0.2 + 8 × 0.05 = 0.4 + 0.4 =", 1060, 270, tam=44, c="tinta", w="sb", anc="rm", en="a"), N(0.8, 1090, 270, "{:.1f}", 54, "azul", anc="lm", en="a", t=0.6),
    T("errores: 0.2 / 0.8 =", 1060, 370, tam=44, c="rojo", w="sb", anc="rm", en="c"), N(0.25, 1090, 370, "{:.2f}", 54, "rojo", anc="lm", en="c", t=0.5),
    T("aciertos: 0.05 / 0.8 =", 1060, 460, tam=44, c="verde", w="sb", anc="rm", en="c", t=1.5), N(0.0625, 1090, 460, "{:.4f}", 54, "verde", anc="lm", en="c", t=2.0),
    *[Bx(x - 50, 900 - (0.25 if i in ERR else 0.0625) * 1400, x + 50, 900, "rojo" if i in ERR else "verde", None, 0, 10, grow="y", en="b", t=0.05 * i)
      for i, x in enumerate(XS)],
    *[T(str(i + 1), x, 940, tam=30, c="gris", w="sb", en="b") for i, x in enumerate(XS)], ilustrativo(1000, texto="ejemplo del apunte"))

# ---- 11. El detalle elegante: 50 %
v.b("Y ahora, el detalle más elegante. [[a]]Suma el peso de los dos errores: 0.25 más 0.25, igual a 0.5. [[b]]La mitad de todo el peso quedó en los casos que el árbol falló. [[c]]Esto pasa siempre, con cualquier épsilon: con los nuevos pesos, el árbol anterior tiene exactamente un error de 0.5. Es decir, es como el azar.",
    CL(), T("2 × 0.25 =", 900, 300, tam=60, c="tinta", w="sb", anc="rm", en="a"), N(0.5, 930, 300, "{:.1f}", 76, "rojo", anc="lm", en="a", t=0.5),
    P("la MITAD del peso queda en los errores", 960, 470, "rojo", 52, en="b"),
    T("siempre: con los nuevos pesos, el árbol anterior tiene ε = 0.5", 960, 640, tam=44, c="tinta", w="sb", en="c"),
    sec="9. Siempre la mitad")
v.b("¿Para qué sirve eso? [[a]]Obliga al siguiente árbol a aprender algo distinto. Si repitiera el mismo corte, no ganaría nada. [[b]]Por eso cada stump mira el problema desde otro ángulo.",
    P("obliga al siguiente árbol a aprender algo DISTINTO", 960, 800, "verde", 46, en="a"),
    T("repetir el mismo corte no gana nada", 960, 900, tam=38, c="gris", w="sb", en="b"))

# ---- 12. Error típico
v.b("El error típico: [[a]]calcular épsilon contando casos, en vez de sumar pesos. [[b]]En la segunda ronda se nota. Supón que el nuevo stump falla en uno de los dos casos difíciles, que pesa 0.25, y en dos casos fáciles, de 0.0625 cada uno. [[c]]Contando casos, serían tres de diez: 0.3. Pero el error ponderado es 0.25 más 0.125: 0.375.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("contar casos en vez de sumar pesos", 410, 230, tam=46, c="rojo", anc="lm", en="a"),
    T("ronda 2: falla en 1 difícil (0.25) y en 2 fáciles (0.0625 c/u)", 960, 380, tam=42, c="tinta", w="sb", en="b"),
    T("contando: 3 / 10 = 0.3", 600, 520, tam=48, c="gris", w="sb", en="c"), K(870, 520, "cruz", 50, en="c", t=0.4),
    T("ponderado: 0.25 + 2 × 0.0625 =", 1300, 620, tam=46, c="tinta", w="sb", anc="rm", en="c", t=1.0), N(0.375, 1330, 620, "{:.3f}", 58, "rojo", anc="lm", en="c", t=1.5),
    ilustrativo(1000), sec="10. El error típico")
v.b("Y con ese épsilon, alfa es un medio del logaritmo de 0.625 sobre 0.375, [[a]]un medio del logaritmo de 1.667, un medio de 0.511: [[b]]aproximadamente 0.26. Un árbol mucho menos influyente que el primero.",
    T("α = ½ · ln(0.625/0.375) = ½ · ln 1.667 = ½ · 0.511 ≈", 1360, 760, tam=42, c="tinta", w="sb", anc="rm", en="a"),
    N(0.26, 1390, 760, "{:.2f}", 56, "violeta", anc="lm", en="b"), P("mucho menos influyente que el primero (0.693)", 960, 880, "ambar", 42, en="b", t=0.6))

# ---- 13. Piensa un segundo
v.b("Piensa un segundo. [[a]]¿Qué alfa recibe un stump con error 0.5? [[b]]¿Y uno con error 0.8?",
    CL(), P("ε = 0.5  →  α = ?", 960, 300, "gris", 60, en="a"), P("ε = 0.8  →  α = ?", 960, 460, "rojo", 60, en="b"),
    sec="11. Piensa un segundo", pausa=0.6)
v.b("Con error 0.5: un medio del logaritmo de uno, [[a]]cero. El árbol no aporta nada al voto. [[b]]Con error 0.8: un medio del logaritmo de 0.2 sobre 0.8, un medio del logaritmo de 0.25, [[c]]menos 0.693. Es peor que el azar, y su voto se invierte: equivale a predecir lo contrario.",
    CL(), T("ε = 0.5:  α = ½ · ln 1 =", 1000, 290, tam=50, c="tinta", w="sb", anc="rm"), T("0", 1030, 290, tam=62, c="gris", anc="lm", en="a"),
    T("ε = 0.8:  α = ½ · ln(0.2/0.8) = ½ · ln 0.25 =", 1180, 430, tam=46, c="tinta", w="sb", anc="rm", en="b"),
    T("−0.693", 1210, 430, tam=62, c="rojo", anc="lm", en="c"), P("peor que el azar → su voto se invierte", 960, 620, "rojo", 48, en="c", t=0.6))

# ---- 14. Para llevar
v.b("Para llevarte: [[a]]épsilon es la suma de los pesos de los errores. [[b]]Alfa, un medio del logaritmo de uno menos épsilon sobre épsilon: menos error, más voz en el voto. [[c]]Y los errores multiplican su peso por e elevado a alfa, hasta quedar con la mitad del peso total.",
    CL(), P("ε = suma de pesos de los errores", 960, 300, "rojo", 50, en="a"), P("α = ½ · ln((1 − ε)/ε): menos error, más voz", 960, 460, "violeta", 48, en="b"),
    P("errores × e^α → quedan con la mitad del peso", 960, 620, "ambar", 48, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]el voto final de AdaBoost, y tres rondas con los pacientes de Heart.",
    CL(), siguiente(4, "AdaBoost: el voto y los datos reales"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
