import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, fila, k1

OUT = out("guion_c35_06_randomforest.json")
v = V("Random Forest: azar en las variables")
RA = FG["boot"]["raices"]
VB, VR, VR10 = FG["var"]["Bagging (todas)"], FG["var"]["RF (1/3)"], FG["var"]["RF (10 %)"]
VARS = ["Edad", "Ingreso", "Zona", "Canal", "Saldo", "Uso", "Antig.", "Mora"]

v.portada("Random Forest", "Bagging de árboles + azar en las variables de cada corte",
          "Sexta parada: el Random Forest. Es un bagging con una sola idea extra, y esa idea es la que baja la correlación entre los árboles.")

# ---- 1. Hook
v.b("Recuerda lo que vimos en Heart: [[a]]en casi la mitad de los árboles bootstrap, la primera pregunta era Thal. [[b]]Cuando una variable es tan fuerte, casi todos los árboles del bagging empiezan igual, [[c]]se parecen, y su correlación ro sube. Justo lo que no queremos.",
    CL(), T("en Heart, la raíz de 200 árboles bootstrap:", 960, 200, tam=46, c="tinta", ent="subir"),
    P(f"Thal en el {RA['Thal'] * 100:.1f} %", 960, 340, "azul", 60, en="a"),
    T("una variable fuerte → árboles que empiezan igual", 960, 500, tam=46, c="tinta", w="sb", en="b"),
    P("se parecen → ρ sube", 960, 650, "rojo", 54, en="c"), real(1050), sec="1. Una variable que domina")

# ---- 2. Analogía
v.b("Piensa en un equipo de fútbol con una estrella. [[a]]Si la estrella juega siempre, todas las jugadas pasan por ella, y el equipo es predecible. [[b]]Ahora el técnico, en cada jugada, sienta al azar a varios jugadores, a veces a la estrella. [[c]]Los demás tienen que inventar otras jugadas, y el equipo aprende caminos distintos.",
    CL(), T("un equipo con una estrella", 960, 170, tam=50, c="tinta", ent="subir"),
    Pt([(10, [300, 360, 800, 520], "gris"), (1, [520, 560, 580, 620], "ambar")], 22, en="a"),
    T("todo pasa por la estrella", 550, 700, tam=36, c="gris", w="sb", en="a", t=0.3),
    Pt([(4, [1100, 360, 1600, 440], "gris"), (6, [1100, 470, 1600, 560], "verde")], 22, en="b"),
    T("en cada jugada, juegan solo algunos", 1350, 640, tam=36, c="verde", w="sb", en="b", t=0.3),
    P("caminos distintos", 1350, 760, "violeta", 46, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: sentar a la estrella")

# ---- 3. La definición
v.b("La definición de la diapositiva diez: [[a]]Random Forest es igual a bagging de árboles, más selección aleatoria de variables. [[b]]Filas: cada árbol recibe una muestra bootstrap, igual que en bagging. [[c]]Variables: cada división evalúa solo un subconjunto aleatorio. Resultado: los árboles se parecen menos entre sí.",
    CL(), caja_formula("Random Forest = bagging de árboles + variables al azar", y=230, c="violeta", tam=52, w=1500, en="a"),
    tarjetas([(400, "azul", "filas", "cada árbol: una muestra bootstrap", "b"), (960, "violeta", "variables", "cada división: un subconjunto al azar", "c"),
              (1520, "verde", "resultado", "árboles menos parecidos", "c")], y=560, w=480, h=330, tam_t=50, tam_d=34),
    T("diapositiva 10", 960, 860, tam=30, c="gris", w="sb", en="a", t=0.5), sec="3. La definición")

# ---- 4. Un nodo, paso a paso (diapositiva 11)
v.b("Veámoslo en un nodo, con la diapositiva once. [[a]]Hay ocho variables disponibles: edad, ingreso, zona, canal, saldo, uso, antigüedad y mora. [[b]]Con mtry igual a tres, en este nodo se sortean solo tres: ingreso, saldo y antigüedad.",
    CL(), T("diapositiva 11 · un nodo, p = 8 variables", 960, 160, tam=46, c="tinta", ent="subir"),
    *[[Bc(330 + 180 * i, 320, 160, 90, "blanco", "gris", 4, 16, en="a", t=0.06 * i, ent="pop", id=f"vb{i}"),
       T(VARS[i], 330 + 180 * i, 320, tam=32, c="gris", w="b", en="a", t=0.06 * i + 0.05, id=f"vt{i}")] for i in range(8)],
    *[[U(f"vb{i}", relleno="violetaclaro", borde="violeta", en="b", t=0.25 * k), U(f"vt{i}", color="violeta", en="b", t=0.25 * k)] for k, i in enumerate([1, 4, 6])],
    P("mtry = 3: Ingreso · Saldo · Antig.", 960, 480, "violeta", 48, en="b", t=0.9), ilustrativo(1000, texto="ejemplo de la diapositiva 11"),
    sec="4. Un nodo, paso a paso")
v.b("El mejor corte se busca solo dentro de esas tres: [[a]]y resulta saldo menor que 420. [[b]]Aunque otra variable excluida, digamos edad, hubiera dado un corte mejor en este nodo, aquí no puede competir. [[c]]Y en el siguiente nodo se sortean otras tres, de nuevo.",
    Bc(960, 650, 720, 120, "verdeclaro", "verde", 5, 26, en="a", ent="pop"), T("mejor división: Saldo < 420", 960, 650, tam=44, c="verde", en="a", t=0.1),
    U("vb0", relleno="rojoclaro", borde="rojo", en="b"), T("Edad no compite aquí", 330, 410, tam=28, c="rojo", w="sb", en="b", t=0.2),
    P("siguiente nodo: otro sorteo de 3", 960, 820, "azul", 44, en="c"))

# ---- 5. Por nodo, no por árbol
v.b("Ese detalle es clave, y la nota de la diapositiva diez lo subraya: [[a]]la selección ocurre en cada nodo, no una sola vez por árbol. [[b]]Si sortearas las variables una vez por árbol, un árbol al que le tocaron variables malas sería malo en todo su recorrido. Eso se llama random subspace. [[c]]En Ames, con el 10 % de las columnas, sortear por árbol da un error de 27 100 dólares; sortear por nodo, 22 500.",
    CL(), P("por NODO, no por árbol (nota de la diapositiva 10)", 960, 200, "violeta", 48, en="a"),
    tarjetas([(560, "rojo", "por árbol", "random subspace: un sorteo para todo el árbol", "b"), (1360, "verde", "por nodo", "Random Forest: un sorteo en cada corte", "b")], y=470, w=640, h=280),
    T("Ames · 10 % de las columnas · RMSE de validación", 960, 690, tam=36, c="gris", w="sb", en="c"),
    T(k1(D["extra_por_arbol_10"]), 560, 790, tam=64, c="rojo", en="c", t=0.3), T(k1(D["extra_por_nodo_10"]), 1360, 790, tam=64, c="verde", en="c", t=0.6),
    real(1050), sec="5. Por nodo, no por árbol")

# ---- 6. Ejemplo resuelto: la variable dominante
v.b("Un cálculo a mano para sentir el efecto, del apunte. [[a]]Diez variables, y cuatro candidatas por nodo. [[b]]La probabilidad de que la variable dominante esté entre las candidatas de un nodo es cuatro sobre diez: 0.40. [[c]]O sea, en el 60 % de los nodos ni siquiera puede competir.",
    CL(), T("ejemplo del apunte · p = 10 variables, m = 4 candidatas", 960, 160, tam=44, c="tinta", ent="subir", en="a"),
    T("P(la dominante es candidata en un nodo) = m/p = 4/10 =", 1240, 320, tam=42, c="tinta", w="sb", anc="rm", en="b"), N(0.40, 1270, 320, "{:.2f}", 60, "azul", anc="lm", en="b", t=0.6),
    T("P(NO es candidata en un nodo) = 1 − 0.40 =", 1240, 430, tam=42, c="tinta", w="sb", anc="rm", en="c"), N(0.60, 1270, 430, "{:.2f}", 60, "rojo", anc="lm", en="c", t=0.6),
    ilustrativo(1000, texto="ejemplo del apunte"), sec="6. La variable dominante, a mano")
v.b("¿Y que falte en la raíz de varios árboles? [[a]]En tres árboles seguidos: 0.6 al cubo, 0.216. [[b]]En diez árboles seguidos: 0.6 elevado a diez, 0.006. [[c]]La dominante sigue mandando en muchos árboles, pero ya no en todos: los demás aprenden de otras variables, y se vuelven distintos.",
    T("ausente en la raíz de 3 árboles: 0.6³ =", 1240, 560, tam=42, c="tinta", w="sb", anc="rm", en="a"), N(0.6 ** 3, 1270, 560, "{:.3f}", 60, "violeta", anc="lm", en="a", t=0.6),
    T("ausente en la raíz de 10 árboles: 0.6¹⁰ =", 1240, 670, tam=42, c="tinta", w="sb", anc="rm", en="b"), N(0.6 ** 10, 1270, 670, "{:.3f}", 60, "violeta", anc="lm", en="b", t=0.6),
    P("ya no manda en TODOS los árboles → árboles distintos", 960, 830, "verde", 46, en="c"))

# ---- 7. Cuántas candidatas por defecto
v.b("¿Cuántas candidatas se usan? En scikit-learn se controla con max_features. [[a]]Para clasificar, el valor de partida es la raíz cuadrada del número de variables. [[b]]Ojo con un detalle: en Heart hay 13 variables, pero después del one-hot quedan 18 columnas. [[c]]La raíz de 18 es 4.24, y scikit-learn redondea hacia abajo: cuatro candidatas por nodo, no tres.",
    CL(), T("¿cuántas candidatas? → max_features", 960, 170, tam=50, c="tinta", ent="subir"),
    P("clasificación: 'sqrt' → √p", 960, 300, "violeta", 50, en="a"),
    T("Heart: 13 variables → 18 columnas tras el one-hot", 960, 450, tam=44, c="tinta", w="sb", en="b"),
    T(f"√18 = {18 ** 0.5:.2f}  →  4 candidatas por nodo (no 3)", 960, 570, tam=52, c="azul", w="b", en="c"), sec="7. Cuántas candidatas")
v.b("Y si max_features vale el total de variables, [[a]]todas las variables compiten en todos los nodos: [[b]]ese Random Forest es, exactamente, un bagging de árboles. Guarda este dato para el próximo video.",
    P("max_features = p  →  Random Forest = bagging", 960, 760, "rojo", 52, en="a"), T("guárdalo para el próximo video", 960, 880, tam=40, c="gris", w="sb", en="b"))

# ---- 8. La tabla y el costo
v.b("La diapositiva doce lo resume en una tabla. [[a]]Muestreo de filas: bootstrap en los dos. Agregación: voto o promedio en los dos. OOB: disponible en ambos. [[b]]La única diferencia está en las variables por nodo: todas en bagging, un subconjunto al azar en Random Forest. [[c]]Y por eso la correlación entre árboles: puede mantenerse alta en bagging, tiende a bajar en Random Forest.",
    CL(), T("diapositiva 12 · bagging con árboles vs Random Forest", 960, 150, tam=44, c="tinta", ent="subir"),
    fila(250, [("aspecto", 480), ("bagging", 1010), ("Random Forest", 1460)], colores=["gris", "azul", "verde"], pesos=["b", "b", "b"]),
    fila(350, [("muestreo de filas", 480), ("bootstrap", 1010), ("bootstrap", 1460)], en="a", fondo="linea"),
    fila(450, [("agregación", 480), ("voto o promedio", 1010), ("voto o promedio", 1460)], en="a"),
    fila(550, [("uso de OOB", 480), ("disponible", 1010), ("disponible", 1460)], en="a", fondo="linea"),
    fila(650, [("variables por nodo", 480), ("todas", 1010), ("subconjunto al azar", 1460)], en="b", colores=["tinta", "azul", "verde"], pesos=["b", "b", "b"]),
    fila(750, [("correlación entre árboles", 480), ("puede seguir alta", 1010), ("tiende a bajar", 1460)], en="c", fondo="linea", colores=["tinta", "rojo", "verde"]),
    sec="8. La tabla de la clase")
v.b("¿Tiene costo? [[a]]Sí: cada árbol es un poco peor, porque a veces no puede usar la mejor variable. En Ames, sigma sube de 36 600 a 37 900 dólares. [[b]]Pero ro baja de 0.049 a 0.024, y como vimos, el promedio gana. Un poco más de sesgo por árbol, a cambio de mucha menos varianza.",
    CL(), T("el costo y la ganancia · Ames", 960, 170, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "costo", f"cada árbol algo peor: σ {k1(VB['sigma'])} → {k1(VR['sigma'])}", "a"),
              (1360, "verde", "ganancia", f"menos parecidos: ρ {VB['rho']:.3f} → {VR['rho']:.3f}", "b")], y=470, w=660, h=330),
    P("un poco más de sesgo por árbol · mucha menos varianza del promedio", 960, 790, "violeta", 42, en="b", t=0.8), real(1050))

# ---- 9. Error típico
v.b("El error típico: [[a]]imaginar que cada árbol recibe un grupo fijo de variables, por ejemplo el árbol uno las variables uno, cinco y ocho, y el árbol dos otras. [[b]]Eso es por árbol. El Random Forest sortea en cada nodo, [[c]]y por eso un mismo árbol termina usando muchas variables distintas a lo largo de su recorrido.",
    CL(), T("«cada árbol recibe sus variables fijas»", 960, 280, tam=56, c="tinta", en="a", id="e1"),
    T("árbol 1: variables 1, 5, 8 · árbol 2: otras…", 960, 400, tam=42, c="gris", w="sb", en="a", t=0.5),
    U("e1", tachado=1.0, color="gris", en="b", dur=0.6), P("Random Forest: un sorteo en CADA nodo", 960, 560, "verde", 50, en="b", t=0.4),
    T("un mismo árbol usa muchas variables distintas", 960, 690, tam=42, c="tinta", w="sb", en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si con menos candidatas los árboles se parecen menos, ¿por qué no usar siempre una sola variable candidata por nodo?",
    CL(), T("¿por qué no usar siempre max_features = 1?", 960, 380, tam=64, c="tinta", en="a"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Porque cada árbol se vuelve demasiado malo: [[a]]con una sola candidata, el corte se hace sobre una variable al azar, sea buena o no. [[b]]Sigma sube tanto que se come la ganancia en ro. [[c]]Hay un punto intermedio, y no se adivina: se valida. Eso es justo el próximo video.",
    CL(), T("una sola candidata = cortar con una variable al azar", 960, 300, tam=50, c="tinta", ent="subir"),
    P("σ sube tanto que se come la ganancia en ρ", 960, 450, "rojo", 48, en="a"), P("hay un punto intermedio: se VALIDA", 960, 620, "verde", 52, en="b"),
    T("→ próximo video", 960, 760, tam=40, c="gris", w="sb", en="c"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]Random Forest es bagging de árboles más un sorteo de variables candidatas en cada nodo. [[b]]Eso baja la correlación entre árboles, a cambio de árboles un poco peores. [[c]]Y con max_features igual al total de variables, vuelves al bagging.",
    CL(), P("Random Forest = bagging + sorteo de variables en CADA nodo", 960, 300, "violeta", 46, en="a"),
    P("ρ baja · cada árbol algo peor · el promedio gana", 960, 470, "verde", 48, en="b"), P("max_features = p  →  bagging", 960, 640, "azul", 50, en="c"),
    sec="11. Para llevar")
v.b("Siguiente parada: [[a]]max_features en la práctica, con datos reales, y un error en el notebook de la clase.",
    CL(), siguiente(7, "max_features en la práctica"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
