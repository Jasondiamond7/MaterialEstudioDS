import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, flujo, k1

OUT = out("guion_c36_05_gradient.json")
v = V("Gradient Boosting: corregir residuos")
CU = D["curvas"]["gb"]

# ejemplo de la diapositiva 10 + stump ilustrativo {B} vs {A, C, D}
OBS = ["A", "B", "C", "D"]
YV = [12, 7, 11, 10]
F0 = sum(YV) / 4                                  # 10
R1 = [y - F0 for y in YV]                         # +2, -3, +1, 0
H1 = [1.0, -3.0, 1.0, 1.0]                        # media de r en cada hoja
ETA = 0.1
F1 = [round(F0 + ETA * h, 2) for h in H1]         # 10.1, 9.7, 10.1, 10.1
R2 = [round(y - f, 2) for y, f in zip(YV, F1)]    # 1.9, -2.7, 0.9, -0.1


def sgn(x):
    s = f"{x:+.1f}" if abs(x - round(x)) > 1e-9 else f"{x:+.0f}"
    return s.replace("-", "−").replace("+0", "0") if x == 0 else s.replace("-", "−")


v.portada("Gradient Boosting", "Cada árbol aprende lo que todavía falta: los residuos",
          "Quinta parada: Gradient Boosting. Vas a hacer una ronda completa a mano con la tabla de la diapositiva diez.")

# ---- 1. Hook
v.b("AdaBoost le avisa al siguiente árbol con pesos: fíjate en estos casos. [[a]]Gradient Boosting es más directo. [[b]]Le pasa al siguiente árbol el error mismo, con su signo y su tamaño: [[c]]a esta casa le faltan dos mil dólares; a esta otra le sobran tres mil.",
    CL(), T("AdaBoost: pesos («fíjate en estos casos»)", 960, 260, tam=50, c="gris", w="sb", ent="subir"),
    T("Gradient Boosting: el error mismo", 960, 420, tam=60, c="ambar", en="b"),
    P("a esta casa le faltan 2 mil · a esta le sobran 3 mil", 960, 600, "azul", 48, en="c"), sec="1. El error mismo")

# ---- 2. Analogía: el sastre
v.b("Piensa en un sastre. [[a]]En la primera prueba, el traje queda más o menos. [[b]]El sastre mide cuánto falta en cada costura: dos centímetros aquí, tres de más allá. [[c]]El siguiente ajuste no rehace el traje: corrige justo esas diferencias.",
    CL(), T("el sastre", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "gris", "prueba 1", "el traje queda más o menos", "a"), (960, "rojo", "medir", "+2 cm aquí · −3 cm allá", "b"),
              (1520, "verde", "ajuste", "corrige justo esas diferencias", "c")], y=490, w=480, h=330, tam_t=50, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el sastre")
v.b("Esas diferencias, lo que falta para llegar al valor real, se llaman residuos. [[a]]Residuo igual a y menos efe: el valor real menos la predicción actual.",
    P("residuo  r = y − F", 960, 800, "ambar", 60, en="a"), T("valor real − predicción actual", 960, 910, tam=40, c="gris", w="sb", en="a", t=0.5))

# ---- 3. La regla de actualización (diapositiva 9)
v.b("La diapositiva nueve escribe la regla. [[a]]El modelo nuevo, efe eme, es el modelo anterior más eta por hache eme, el árbol nuevo. [[b]]Y el ciclo: una pérdida, su gradiente negativo, un árbol que lo aproxima, y el modelo actualizado.",
    CL(), T("la regla · diapositiva 9", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)", y=320, c="ambar", tam=60, w=1100, en="a"),
    flujo([("pérdida", "rojo", "b"), ("gradiente negativo", "violeta", "b"), ("nuevo árbol hₘ", "azul", "b"), ("modelo actualizado", "verde", "b")],
          y=600, w=330, h=170, x0=330, paso=420, tam=38), sec="3. La regla")
v.b("Y abajo, una frase clave: [[a]]con error cuadrático, el gradiente negativo coincide con los residuos. En la próxima parada veremos por qué. Por ahora, trabajemos con residuos.",
    P("error cuadrático: gradiente negativo = residuos", 960, 830, "violeta", 48, en="a"), T("diapositiva 9", 1860, 1030, tam=26, c="gris", w="sb", anc="rm"))

# ---- 4-8. Ejemplo resuelto: la tabla de la diapositiva 10
COLX = [260, 470, 680, 930, 1200, 1460, 1720]
HEAD = ["obs.", "y", "F₀", "r₁ = y − F₀", "h₁", "F₁", "r₂ = y − F₁"]
FY = [360 + 110 * i for i in range(4)]


def celda(txt, j, i, c="tinta", en=None, t=0.0, tam=44):
    return T(txt, COLX[j], FY[i], tam=tam, c=c, w="b", en=en, t=t)


v.b("Ahora a mano, con la diapositiva diez. [[a]]Cuatro observaciones: A, B, C y D, con valores 12, 7, 11 y 10. [[b]]El primer paso es la predicción base, efe cero: el promedio. 12 más 7 más 11 más 10 son 40, y 40 dividido por cuatro: 10.",
    CL(), T("una ronda a mano · diapositiva 10 (pérdida cuadrática)", 960, 140, tam=44, c="tinta", ent="subir"),
    *[T(h, COLX[j], 250, tam=34, c="gris", w="sb", en=("a" if j < 2 else ("b" if j == 2 else "zz"))) for j, h in enumerate(HEAD[:3])],
    L(180, 295, 1820, 295, "linea", 3, en="a"),
    *[[celda(o, 0, i, en="a", t=0.15 * i), celda(str(y), 1, i, c="azul", en="a", t=0.15 * i + 0.1)] for i, (o, y) in enumerate(zip(OBS, YV))],
    T("F₀ = (12 + 7 + 11 + 10) / 4 = 40 / 4 = 10", 960, 860, tam=44, c="teal", w="sb", en="b", id="f0txt"),
    *[celda("10", 2, i, c="teal", en="b", t=1.5 + 0.1 * i) for i in range(4)],
    ilustrativo(1000, texto="ejemplo de la diapositiva 10"), sec="4. La tabla de la diapositiva 10")
v.b("Segundo paso: los residuos, y menos efe cero. [[a]]A: 12 menos 10, más 2. B: 7 menos 10, menos 3. [[b]]C: más 1. Y D: cero, porque D ya vale justo el promedio. [[c]]Los residuos son lo que el primer árbol tiene que aprender.",
    T(HEAD[3], COLX[3], 250, tam=34, c="gris", w="sb"),
    celda(sgn(R1[0]), 3, 0, c="ambar", en="a"), celda(sgn(R1[1]), 3, 1, c="ambar", en="a", t=1.2),
    celda(sgn(R1[2]), 3, 2, c="ambar", en="b"), celda("0", 3, 3, c="ambar", en="b", t=0.6),
    P("r₁ = lo que el árbol 1 tiene que aprender", 960, 940, "ambar", 40, en="c", id="r1p"))
v.b("Tercer paso: un árbol pequeño que predice esos residuos. [[a]]La diapositiva no dice dónde corta, así que usemos un stump de ejemplo: pone a B en una hoja, y a A, C y D en la otra. [[b]]Cada hoja predice el promedio de sus residuos: para B, menos 3. [[c]]Para A, C y D: 2 más 1 más 0, dividido por tres, igual a 1.",
    X("f0txt", "r1p"), T(HEAD[4], COLX[4], 250, tam=34, c="gris", w="sb"),
    T("stump de ejemplo: {B} | {A, C, D}", 960, 860, tam=40, c="azul", w="sb", en="a", id="stx"),
    celda("−3", 4, 1, c="azul", en="b"), T("(2 + 1 + 0) / 3 = 1", 960, 940, tam=36, c="azul", w="sb", en="c", id="ptx"),
    *[celda("+1", 4, i, c="azul", en="c", t=0.8 + 0.1 * k) for k, i in enumerate([0, 2, 3])])
v.b("Cuarto paso: actualizar con eta. Usemos 0.1. [[a]]Para A, C y D: 10 más 0.1 por 1, igual a 10.1. [[b]]Para B: 10 más 0.1 por menos 3, igual a 9.7. Solo avanzamos un diez por ciento de la corrección.",
    X("stx", "ptx"), T(HEAD[5], COLX[5], 250, tam=34, c="gris", w="sb"), T("η = 0.1", 960, 860, tam=40, c="ambar", w="b", id="etx"),
    *[celda("10.1", 5, i, c="verde", en="a", t=0.3 + 0.1 * k) for k, i in enumerate([0, 2, 3])],
    T("10 + 0.1 · 1 = 10.1  ·  10 + 0.1 · (−3) = 9.7", 960, 940, tam=38, c="verde", w="sb", en="a", id="utx"), celda("9.7", 5, 1, c="verde", en="b", t=0.5))
v.b("Quinto paso: los nuevos residuos. [[a]]A, 1.9. B, menos 2.7. C, 0.9. D, menos 0.1. [[b]]Todos se achicaron un poco, salvo D, que estaba perfecto y ahora se pasa por una décima.",
    X("etx", "utx"), T(HEAD[6], COLX[6], 250, tam=34, c="gris", w="sb"),
    *[celda(f"{r:+.1f}".replace("-", "−"), 6, i, c="rojo", en="a", t=0.7 * i) for i, r in enumerate(R2)],
    P("todos se achican, salvo D (se pasa por 0.1)", 960, 900, "rojo", 40, en="b"))
v.b("¿Mejoró el modelo? Midamos el error cuadrático medio. [[a]]Antes: 4 más 9 más 1 más 0, dividido por cuatro: 3.5. [[b]]Después: 3.61 más 7.29 más 0.81 más 0.01, igual a 11.72, dividido por cuatro: 2.93. [[c]]Bajó. Y el siguiente árbol vuelve a estudiar lo que aún falta: los residuos nuevos.",
    CL(), T("¿mejoró? error cuadrático medio", 960, 160, tam=50, c="tinta", ent="subir"),
    T("antes:  (4 + 9 + 1 + 0) / 4 =", 1060, 330, tam=46, c="tinta", w="sb", anc="rm", en="a"), N(3.5, 1090, 330, "{:.1f}", 60, "ambar", anc="lm", en="a", t=0.8),
    T("después:  (3.61 + 7.29 + 0.81 + 0.01) / 4 = 11.72 / 4 =", 1260, 460, tam=40, c="tinta", w="sb", anc="rm", en="b"),
    N(2.93, 1290, 460, "{:.2f}", 60, "verde", anc="lm", en="b", t=1.4),
    P("el siguiente árbol estudia lo que aún falta: r₂", 960, 640, "azul", 48, en="c"), ilustrativo(1000, texto="diapositiva 10 + stump de ejemplo"))

# ---- 9. Datos reales
v.b("Con datos reales pasa lo mismo, a gran escala. [[a]]En Ames, Gradient Boosting con árboles de profundidad tres y eta 0.1. [[b]]Con un árbol, el error de validación es 69 000 dólares. Con diez, 40 700. [[c]]Con cincuenta, 24 300. Con cien, 22 700. Y con trescientos, 21 800.",
    CL(), T("Ames · RMSE de validación de Gradient Boosting", 960, 150, tam=46, c="tinta", ent="subir"),
    T("profundidad 3 · η = 0.1", 960, 220, tam=34, c="gris", w="sb", en="a"),
    *[[Bx(330 + 300 * k, 880 - CU[m - 1] / 1000 * 8, 530 + 300 * k, 880, c, None, 0, 12, grow="y", en=e, t=0.2 * (k % 2)),
       T(k1(CU[m - 1]), 430 + 300 * k, 880 - CU[m - 1] / 1000 * 8 - 34, tam=34, c="tinta", w="b", en=e, t=0.2 * (k % 2) + 0.4),
       T(f"{m} árbol" + ("es" if m > 1 else ""), 430 + 300 * k, 920, tam=30, c="gris", w="sb", en=e)]
      for k, (m, c, e) in enumerate([(1, "gris", "b"), (10, "azul", "b"), (50, "violeta", "c"), (100, "teal", "c"), (300, "verde", "c")])],
    real(1050), sec="5. Con datos reales")

# ---- 10. En Python
v.b("En Python: [[a]]GradientBoostingRegressor, con el número de árboles, el learning_rate, que es eta, y la profundidad. [[b]]Con la pérdida cuadrática, efe cero es el promedio, igual que a mano. [[c]]Y staged_predict te entrega efe uno, efe dos, efe tres: la predicción después de cada árbol.",
    CL(), T("Gradient Boosting en scikit-learn", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["from sklearn.ensemble import GradientBoostingRegressor", "gb = GradientBoostingRegressor(n_estimators=300,",
            "         learning_rate=0.1, max_depth=3)", "gb.fit(X_fit, y_fit)", "for F in gb.staged_predict(X_val):   # F1, F2, F3, ...",
            "    ..."], y=450, w=1500, tam=36, en="a", dt=0.3),
    T("pérdida cuadrática: F₀ = promedio de y", 960, 760, tam=40, c="teal", w="sb", en="b"),
    P("staged_predict: la predicción tras cada árbol", 960, 870, "azul", 44, en="c"), sec="6. En Python")

# ---- 11. Error típico
v.b("El error típico al programarlo a mano: [[a]]entrenar cada árbol nuevo sobre y, en vez de sobre los residuos. [[b]]Así, todos los árboles aprenderían lo mismo que el primero, y sumarlos no corregiría nada. [[c]]El otro error: sumar el árbol completo, sin multiplicar por eta.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("árbol nuevo entrenado sobre y", 410, 260, tam=48, c="rojo", anc="lm", en="a"),
    T("todos aprenden lo mismo · no corrige nada", 960, 400, tam=42, c="tinta", w="sb", en="b"),
    K(330, 560, "check", 70, en="b", t=0.6), T("árbol nuevo entrenado sobre r = y − F", 410, 560, tam=48, c="verde", anc="lm", en="b", t=0.6),
    P("y no olvides multiplicar por η", 960, 740, "ambar", 48, en="c"), sec="7. El error típico")

# ---- 12. Piensa un segundo
v.b("Piensa un segundo. [[a]]En la tabla, ¿qué pasaría con eta igual a uno? [[b]]¿Los residuos bajan más o menos?",
    CL(), P("misma tabla, con η = 1", 960, 300, "ambar", 56, en="a"), T("¿bajan más o menos los residuos?", 960, 470, tam=64, c="tinta", en="b"),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("Con eta igual a uno, A, C y D pasan a 11, y B a 7. [[a]]Los residuos quedan en 1, 0, 0 y menos 1, y el error cuadrático medio cae a 0.5. [[b]]Mucho mejor en esta tabla. [[c]]Pero ojo: cada árbol también aprende el ruido de sus datos, y con eta uno lo copia completo. Por eso, en datos reales, los pasos chicos suelen generalizar mejor.",
    CL(), T("η = 1:  F₁ = 11, 7, 11, 11", 960, 280, tam=52, c="tinta", w="sb"),
    T("residuos: +1, 0, 0, −1  →  error cuadrático medio =", 1260, 400, tam=44, c="tinta", w="sb", anc="rm", en="a"),
    N(0.5, 1290, 400, "{:.1f}", 60, "verde", anc="lm", en="a", t=1.0),
    P("mejor en esta tabla…", 960, 540, "verde", 46, en="b"),
    P("…pero con η = 1 cada árbol copia también su ruido", 960, 660, "rojo", 46, en="c"), ilustrativo(1000, texto="diapositiva 10 + stump de ejemplo"))

# ---- 13. Para llevar
v.b("Para llevarte: [[a]]Gradient Boosting parte del promedio y entrena cada árbol sobre los residuos. [[b]]Cada árbol se suma multiplicado por eta. [[c]]Y en cada ronda, el siguiente árbol estudia lo que todavía falta por explicar.",
    CL(), P("parte del promedio · cada árbol aprende los residuos", 960, 300, "ambar", 46, en="a"),
    P("Fₘ = Fₘ₋₁ + η · hₘ", 960, 460, "azul", 52, en="b"), P("cada ronda estudia lo que aún falta", 960, 620, "verde", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]por qué se llama gradient. La pérdida y su gradiente.",
    CL(), siguiente(6, "La pérdida y el gradiente"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
