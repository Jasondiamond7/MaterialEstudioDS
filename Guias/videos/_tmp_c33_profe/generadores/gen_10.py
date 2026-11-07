import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, caja_formula, alerta

OUT = out("guion_c33_10_repaso.json")
v = V("Repaso final y checklist")
FA, XR, T4, M4, RU, RS = D["fallas"], D["xor"], D["top4"], D["manual4"], D["ruido"], D["resumen"]

v.portada("Repaso final", "Las seis ideas, el checklist del proyecto y tu criterio de dominio",
          "Repaso final de la clase treinta y tres. Las seis ideas, los números que conviene recordar, el checklist del proyecto, y cómo saber si ya dominas el tema.")

# ---- 1. Seis ideas
IDEAS = [("1", "Limpiar primero", "reglas de negocio · imposible → NaN → mediana · sin borrar filas", "teal"),
         ("2", "Construir con sentido", "ratios con épsilon · logs · binarias · cuartiles", "violeta"),
         ("3", "Tres razones para sobrar", "irrelevante · redundante · útil solo en conjunto", "rojo"),
         ("4", "Tres familias, un embudo", "filter rápido · wrapper caro · embedded (Lasso)", "azul"),
         ("5", "La trampa", "todo lo que mira el target, dentro de cada fold", "ambar"),
         ("6", "Desconfía", "un AUC demasiado bueno suele ser leakage", "verde")]


def idea(i, en, t=0.0):
    n, a, b, c = IDEAS[i]
    x, y = (520 if i % 2 == 0 else 1400), 330 + 220 * (i // 2)
    return [Bc(x, y, 820, 180, "blanco", c, 5, 28, sombra=True, en=en, ent="subir", t=t), T(n, x - 350, y, tam=72, c=c, en=en, t=t + 0.05),
            T(a, x + 40, y - 30, tam=38, c=c, w="b", mw=660, en=en, t=t + 0.1), T(b, x + 40, y + 34, tam=27, c="tinta", w="sb", mw=660, en=en, t=t + 0.2)]


v.b("Repasemos la clase en seis ideas. [[a]]Una: limpiar primero, con reglas de negocio, sin borrar filas. [[b]]Dos: construir variables con sentido, y con épsilon en los denominadores.",
    CL(), T("la clase en 6 ideas", 960, 180, tam=48, c="tinta", ent="subir"), idea(0, "a"), idea(1, "b"), sec="1. Seis ideas")
v.b("[[a]]Tres: una variable sobra por tres razones distintas, y cada una pide una decisión distinta. [[b]]Cuatro: filter, wrapper y embedded son etapas de un embudo, no rivales.",
    idea(2, "a"), idea(3, "b"))
v.b("[[a]]Cinco: todo lo que mira el target va dentro de cada fold. [[b]]Y seis: desconfía de un AUC demasiado bueno.",
    idea(4, "a"), idea(5, "b"))

# ---- 2. Los números de la clase
NUM = [("limpieza", f"{D['imposibles']} imposibles · {D['edad']['menores']} clientes de 0 a 14 años", "teal"),
       ("construir", f"12 → 50 · AUC {RS[0][2]:.3f} → {RS[1][2]:.3f}", "violeta"),
       ("numero_fallas", f"sola {FA['sola']:.3f} · base {FA['base']:.3f} → {FA['base_f']:.3f}", "rojo"),
       ("XOR", f"logística {XR['log']:.3f} · con x1·x2 {XR['inter']:.3f}", "rojo"),
       ("cuatro contra cuatro", f"top-4 {T4['auc']:.3f} · sin redundancia {M4['auc']:.3f}", "azul"),
       ("Lasso C = 0.05", f"{D['lasso_n']} de 50 · AUC {RS[2][2]:.4f} · {D['estab']['siempre']} estables", "verde"),
       ("ruido puro", f"afuera {RU['mal']:.3f} · adentro {RU['bien']:.3f} · verdad 0.500", "ambar")]
v.b("Los números que conviene recordar, todos de la base real o del notebook. [[a]]Cuarenta y tres imposibles. De doce a cincuenta variables, con el AUC de cero coma nueve dos a cero coma nueve cuatro uno. [[b]]Número de fallas: azar sola, útil acompañada. El XOR: cero coma cinco uno sin el producto, uno con él. [[c]]Cuatro contra cuatro: cero coma ocho cero cuatro contra cero coma ocho nueve tres. El Lasso: veintisiete de cincuenta. Y el ruido puro: cero coma ochocientos ochenta y dos afuera, cero coma quinientos catorce adentro.",
    CL(), T("los números de la clase · verificados", 960, 170, tam=44, c="tinta", ent="subir"),
    *[[Bc(960, 270 + 98 * i, 1560, 84, "blanco", c, 4, 20, en=k, ent="subir", t=t_), T(a, 400, 270 + 98 * i, tam=34, c=c, w="b", en=k, t=t_ + 0.05),
       T(b, 1150, 270 + 98 * i, tam=34, c="tinta", w="sb", en=k, t=t_ + 0.15)]
      for i, ((a, b, c), k, t_) in enumerate(zip(NUM, "aabbccc", (0, 0.9, 0, 1.6, 0, 1.6, 2.6)))],
    sec="2. Los números de la clase")

# ---- 3. Checklist
CHK = ["Limpiar con criterio de negocio, no estadístico", "Separar test antes de tocar cualquier otra cosa", "Construir candidatas de forma sistemática",
       "Filtro base: varianza casi nula, cardinalidad, señal mínima", "Detectar redundancia y desempatar con criterio de negocio",
       "No descartar por p-valor sin probar en conjunto", "Método multivariado: RFE o Lasso sobre lo que quedó", "Todo lo anterior DENTRO de un pipeline validado por CV",
       "Medir estabilidad ante cambios de partición y de semilla", "Documentar por qué quedó cada variable y qué cuesta mantenerla"]


def chk(i, en, t):
    x, y = (520 if i < 5 else 1400), 300 + 120 * (i % 5)
    c = "ambar" if i in (0, 1, 9) else "teal"
    return [Bc(x, y, 840, 100, "blanco", c, 4, 22, en=en, ent="subir", t=t), T(str(i + 1), x - 375, y, tam=48, c=c, en=en, t=t + 0.05),
            T(CHK[i], x + 30, y, tam=28, c="tinta", w="sb", mw=700, en=en, t=t + 0.1)]


v.b("El entregable conceptual de la clase es este checklist, en diez pasos. [[a]]Limpiar con criterio de negocio. Separar el test antes de todo. Construir candidatas. [[b]]Un filtro base. Y detectar la redundancia, desempatando con criterio.",
    CL(), T("checklist metodológico · el entregable de la clase", 960, 180, tam=42, c="tinta", ent="subir"),
    [chk(i, "a", 0.5 * i) for i in range(3)], [chk(i, "b", 0.6 * (i - 3)) for i in range(3, 5)], sec="3. El checklist del proyecto")
v.b("[[a]]No descartar por p-valor sin probar en conjunto. Un método multivariado, RFE o Lasso. [[b]]Todo dentro de un pipeline validado. Medir la estabilidad. Y documentar por qué quedó cada variable.",
    [chk(i, "a", 0.7 * (i - 5)) for i in range(5, 7)], [chk(i, "b", 0.5 * (i - 7)) for i in range(7, 10)])
v.b("Fíjate en los que marqué en ámbar: [[a]]los pasos uno, dos y diez no son estadísticos. [[b]]Según el profesor, son los que más modelos salvan.",
    R(520, 300, 870, 130, "ambar", 26, en="a"), R(520, 420, 870, 130, "ambar", 26, en="a", t=0.3), R(1400, 780, 870, 130, "ambar", 26, en="a", t=0.6),
    P("los que más modelos salvan", 960, 950, "ambar", 44, en="b"))

# ---- 4. Las alertas del material
AL = [("numero_fallas «irrelevante»", "sube el modelo acompañada: no lo es"), ("«25 de 50» y «0.938»", f"reproducido: {D['lasso_n']} de 50 y {RS[2][2]:.4f}"),
      ("«6 contra 6»: 9 puntos", "depende de un empate en la MI: con el top-6 del notebook, < 1 punto"),
      ("XOR: «ninguna sirve»", f"x2 tiene p = {XR['p2']:.3f}: un falso positivo por azar"),
      ("«11 pares sobre 0.90»", "la tabla incluye un par de 0.728"), ("cuartiles antes del CV", "usan estadísticas del conjunto: en rigor, dentro del fold")]
v.b("Y las alertas del material que vimos en la serie. [[a]]Número de fallas no es irrelevante. El Lasso deja veintisiete, no veinticinco. [[b]]El seis contra seis depende de un empate. Y en el XOR, x2 pasa el filtro por azar. [[c]]La tabla de pares mete uno de cero coma setenta y dos, y los cuartiles, en rigor, van dentro del fold.",
    CL(), alerta(y=170),
    *[[Bc(960, 280 + 112 * i, 1640, 96, "ambarclaro", None, 0, 20, en=k, ent="subir", t=t_), T(a, 160, 280 + 112 * i, tam=32, c="ambar", w="b", anc="lm", en=k, t=t_ + 0.05),
       T(b, 1760, 280 + 112 * i, tam=30, c="tinta", w="sb", anc="rm", en=k, t=t_ + 0.15)]
      for i, ((a, b), k, t_) in enumerate(zip(AL, "aabbcc", (0, 1.4, 0, 2.0, 0, 1.6)))],
    sec="4. Las alertas del material")

# ---- 5. Adelanto: WOE e IV, a mano
v.b("Un adelanto que el profesor pidió anotar, y que no se calcula en esta clase: [[a]]el Weight of Evidence, que se usa en banca para discretizar variables antes de una logística. [[b]]Se parte la variable en tramos, y en cada tramo se compara la proporción de buenos con la de malos.",
    CL(), T("adelanto · WOE e Information Value", 960, 190, tam=48, c="violeta", ent="subir"),
    caja_formula("WOE = ln(% buenos / % malos)", y=370, c="violeta", tam=60, w=1100, en="a"),
    P("se mencionó en clase · no se calcula en el notebook", 960, 560, "gris", 44, en="b"), sec="5. Adelanto: WOE a mano")
v.b("El ejemplo del apunte: ochocientos buenos y doscientos malos, en dos tramos. [[a]]Tramo A: quinientos de ochocientos buenos, sesenta y dos coma cinco por ciento; y sesenta de doscientos malos, treinta por ciento. [[b]]WOE: logaritmo de cero coma seis dos cinco entre cero coma treinta, cero coma siete tres cuatro. [[c]]Aporte al IV: cero coma tres dos cinco por cero coma siete tres cuatro, cero coma dos tres nueve.",
    CL(), T("ejemplo del apunte · 800 buenos · 200 malos", 960, 180, tam=38, c="gris", w="sb", ent="subir"),
    T("tramo", 260, 280, tam=30, c="gris", w="sb"), T("% buenos", 560, 280, tam=30, c="gris", w="sb"), T("% malos", 880, 280, tam=30, c="gris", w="sb"),
    T("WOE", 1200, 280, tam=30, c="gris", w="sb"), T("aporte al IV", 1560, 280, tam=30, c="gris", w="sb"),
    Bc(960, 400, 1640, 130, "blanco", "azul", 4, 22, en="a", ent="subir"), T("A", 260, 400, tam=56, c="azul", en="a"),
    T("500/800 = 62.5 %", 560, 400, tam=36, c="tinta", w="sb", en="a", t=0.2), T("60/200 = 30 %", 880, 400, tam=36, c="tinta", w="sb", en="a", t=1.6),
    T("ln(0.625 / 0.30)", 1200, 380, tam=30, c="tinta", w="sb", en="b"), N(0.734, 1200, 425, "{:.3f}", 40, "azul", en="b", t=0.4),
    T("0.325 · 0.734", 1560, 380, tam=30, c="tinta", w="sb", en="c"), N(0.239, 1560, 425, "{:.3f}", 40, "azul", en="c", t=0.4))
v.b("Tramo B: [[a]]treinta y siete coma cinco por ciento de los buenos, y setenta por ciento de los malos. [[b]]WOE: menos cero coma seis dos cuatro. Aporte: cero coma dos cero tres. [[c]]El IV suma los aportes: cero coma cuarenta y cuatro. Poder predictivo fuerte.",
    Bc(960, 560, 1640, 130, "blanco", "rojo", 4, 22, en="a", ent="subir"), T("B", 260, 560, tam=56, c="rojo", en="a"),
    T("300/800 = 37.5 %", 560, 560, tam=36, c="tinta", w="sb", en="a", t=0.2), T("140/200 = 70 %", 880, 560, tam=36, c="tinta", w="sb", en="a", t=0.6),
    T("ln(0.375 / 0.70)", 1200, 540, tam=30, c="tinta", w="sb", en="b"), T("−0.624", 1200, 585, tam=40, c="rojo", en="b", t=0.3),
    T("(−0.325) · (−0.624)", 1560, 540, tam=30, c="tinta", w="sb", en="b", t=0.6), N(0.203, 1560, 585, "{:.3f}", 40, "rojo", en="b", t=0.9),
    T("IV = 0.239 + 0.203 =", 1180, 740, tam=50, c="tinta", anc="rm", en="c"), N(0.44, 1210, 740, "{:.2f}", 70, "violeta", en="c", t=0.4, anc="lm"),
    T("< 0.02 nulo · 0.02–0.1 débil · 0.1–0.3 medio · 0.3–0.5 fuerte · > 0.5 sospechoso (leakage)", 960, 870, tam=32, c="gris", w="sb", en="c", t=1.0))

# ---- 6. Piensa un segundo
v.b("Piensa un segundo: dos preguntas rápidas. [[a]]Una variable tiene p-valor cero coma sesenta. ¿La descartas? [[b]]Y en sklearn, ¿un C chico en el Lasso deja más o menos variables?",
    CL(), T("1 · p = 0.60: ¿la descartas?", 960, 330, tam=64, c="violeta", en="a"), T("2 · Lasso con C chico: ¿más o menos variables?", 960, 530, tam=56, c="violeta", en="b"),
    sec="6. Piensa un segundo", pausa=0.6)
v.b("Uno: no, sin antes probarla en conjunto. Número de fallas tenía cero coma sesenta, y subía el modelo. [[a]]Dos: menos. C es el inverso de la penalización: con C igual a cero coma cero uno, solo trece variables sobreviven.",
    P(f"1 · no sin probarla acompañada: {FA['base']:.3f} → {FA['base_f']:.3f}", 960, 720, "verde", 44),
    P(f"2 · menos: C = 0.01 → {D['lasso_C']['0.01']} variables", 960, 860, "verde", 44, en="a"))

# ---- 7. Criterio de dominio
DOM = ["Detecto valores imposibles y justifico la regla de negocio", "Construyo ratios con épsilon y conozco las familias de variables",
       "Distingo irrelevante, redundante y útil en conjunto, con un ejemplo", "Explico filter, wrapper y embedded, y cuándo usar cada uno",
       "Armo un Pipeline donde toda la selección vive dentro del fold", "Resuelvo los bloques 1 a 8 del cuaderno con 80 % de aciertos"]
v.b("¿Cómo sabes si ya dominas el tema? [[a]]Detectas valores imposibles y justificas tu regla. Construyes ratios con épsilon. Distingues los tres motivos, con un ejemplo de cada uno. [[b]]Explicas las tres familias. Armas un Pipeline con la selección adentro. Y resuelves el cuaderno con ochenta por ciento de aciertos.",
    CL(), T("criterio de dominio para pasar a la clase 34", 960, 180, tam=44, c="tinta", ent="subir"),
    *[[Bc(960, 290 + 105 * i, 1400, 88, "blanco", "verde", 4, 20, en=("a" if i < 3 else "b"), ent="subir", t=0.9 * (i % 3)),
       T(tx, 340, 290 + 105 * i, tam=32, c="tinta", w="sb", anc="lm", en=("a" if i < 3 else "b"), t=0.9 * (i % 3) + 0.1),
       K(1580, 290 + 105 * i, "check", 44, en=("a" if i < 3 else "b"), t=0.9 * (i % 3) + 0.4)] for i, tx in enumerate(DOM)],
    sec="7. ¿Ya lo dominas?")

# ---- 8. Hacia la clase 34
v.b("¿Qué viene? La clase treinta y cuatro: árboles de decisión. [[a]]Un modelo que hace su propia selección de variables mientras se entrena, [[b]]y que elige sus cortes con la ganancia de información: la misma información mutua que usamos hoy como filtro. [[c]]Y vas a ver por qué reclamos siempre queda arriba de todo.",
    CL(), P("clase 34 · árboles de decisión", 960, 220, "verde", 60, ent="subir"),
    T("selecciona variables mientras se entrena", 960, 380, tam=48, c="tinta", w="sb", en="a"),
    T("ganancia de información = la información mutua de hoy", 960, 500, tam=48, c="tinta", w="sb", en="b"),
    P("¿por qué reclamos queda siempre arriba?", 960, 680, "ambar", 50, en="c"), sec="8. Hacia la clase 34")
v.b("Y una pregunta para pensar antes de esa clase: [[a]]si tuvieras que explicarle a un gerente comercial por qué se fuga un cliente, ¿le mostrarías doce coeficientes, o preferirías otra cosa? [[b]]Nos vemos en la clase treinta y cuatro.",
    CL(), T("para pensar", 960, 260, tam=44, c="gris", w="sb", ent="subir"),
    T("¿12 coeficientes, u otra cosa, para explicarle la fuga a un gerente?", 960, 450, tam=54, c="violeta", mw=1500, en="a"),
    P("nos vemos en la clase 34", 960, 700, "teal", 56, en="b"))

n = v.guardar(OUT)
print(n, "beats")
