import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, NODOS

OUT = out("guion_c34_11_repaso.json")
v = V("Repaso final y checklist")
CU, FI, RC = D["curva"], D["final"], D["reclamos"]
best = max(CU, key=lambda c: c["va"])
SL = CU[-1]
H11 = NODOS[11]

v.portada("Repaso final", "Las ideas clave, los números de la clase, el checklist y tu criterio de dominio",
          "Repaso final de la clase treinta y cuatro. Las ideas clave, los números que conviene recordar, el checklist para entrenar un árbol, y cómo saber si ya dominas el tema.")

IDEAS = [("1", "Rectángulos", "preguntas sí/no · cada hoja predice la tasa de su grupo", "violeta"),
         ("2", "Impureza y ganancia", "Gini o entropía · gana el corte con más ganancia", "azul"),
         ("3", "Voraz", "el mejor corte ahora, sin mirar adelante", "rojo"),
         ("4", "Sobreajuste y poda", "curva de profundidad · ccp_alpha validado", "ambar"),
         ("5", "Importancia", "permutación sobre el test, no impureza", "verde"),
         ("6", "Al bosque", "explicable e inestable → promediar muchos", "teal")]


def idea(i, en, t=0.0):
    n, a, b, c = IDEAS[i]
    x, y = (520 if i % 2 == 0 else 1400), 330 + 220 * (i // 2)
    return [Bc(x, y, 820, 180, "blanco", c, 5, 28, sombra=True, en=en, ent="subir", t=t), T(n, x - 350, y, tam=72, c=c, en=en, t=t + 0.05),
            T(a, x + 40, y - 30, tam=38, c=c, w="b", mw=660, en=en, t=t + 0.1), T(b, x + 40, y + 34, tam=27, c="tinta", w="sb", mw=660, en=en, t=t + 0.2)]


# ---- 1. Seis ideas
v.b("Repasemos la clase en seis ideas. [[a]]Uno: un árbol parte el espacio con preguntas de sí o no, y cada hoja predice la tasa de fuga de su grupo. [[b]]Dos: elige cada corte con la impureza, Gini o entropía, y se queda con la mayor ganancia.",
    CL(), T("la clase en 6 ideas", 960, 180, tam=48, c="tinta", ent="subir"), idea(0, "a"), idea(1, "b"), sec="1. Seis ideas")
v.b("[[a]]Tres: es voraz, elige lo mejor ahora sin mirar adelante. [[b]]Cuatro: sin límites memoriza; se controla con la curva de profundidad y con la poda validada.",
    idea(2, "a"), idea(3, "b"))
v.b("[[a]]Cinco: la importancia por impureza premia la cardinalidad; usa permutación sobre el test. [[b]]Y seis: el árbol es explicable pero inestable; promediar muchos, en un bosque, predice mucho mejor.",
    idea(4, "a"), idea(5, "b"))

# ---- 2. Los números
NUM = [("el árbol de la clase", f"7 hojas · hoja más roja: {H11['n']} clientes, {pct(H11['fugas'] / H11['n'], 0)} de fuga, lift {H11['fugas'] / H11['n'] / D['tasa']:.1f}", "violeta"),
       ("raíz", f"Gini {D['gini_raiz']:.3f} · reclamos gana {RC['gan']:.3f} · edad ≈ 0", "azul"),
       ("sobreajuste", f"mejor profundidad {best['d']}: {best['va']:.3f} · sin límite {SL['hojas']} hojas: {SL['tr']:.3f} → {SL['va']:.3f}", "rojo"),
       ("poda", f"α = {D['grid_alpha']:.4f} · {D['grid_hojas']} hojas · {D['grid_auc']:.3f}", "ambar"),
       ("modelos", f"árbol podado {FI['podado']:.3f} · logística {FI['logistica']:.3f} · bosque {FI['rf300']:.3f}", "verde")]
v.b("Los números de la clase que conviene recordar. [[a]]El árbol de profundidad tres tiene siete hojas, y la más roja, ciento cuarenta y nueve clientes con noventa y cuatro por ciento de fuga: un lift de seis. [[b]]En la raíz, el Gini es cero coma dos seis cinco, y reclamos gana con cero coma cero seis nueve, mientras edad no gana nada.",
    CL(), T("los números de la clase", 960, 170, tam=48, c="tinta", ent="subir"),
    *[[Bc(960, 290 + 130 * i, 1600, 110, "blanco", c, 4, 22, sombra=True, en=m, ent="subir"), T(a, 220, 290 + 130 * i, tam=36, c=c, w="b", anc="lm", en=m, t=0.1),
       T(b, 650, 290 + 130 * i, tam=32, c="tinta", w="sb", anc="lm", en=m, t=0.2)] for i, ((a, b, c), m) in enumerate(zip(NUM[:2], "ab"))],
    real(1050), sec="2. Los números de la clase")
v.b("[[a]]La mejor profundidad es seis, con cero coma nueve uno uno. Sin límite, doscientas veinticuatro hojas: {1.000|uno} en entrenamiento, y cero coma ocho dos siete en validación. [[b]]La poda validada elige alfa cero coma cero cero uno tres, con treinta y seis hojas. [[c]]Y al final: árbol podado cero coma nueve cero nueve, logística cero coma nueve dos cero, y bosque cero coma nueve siete cuatro.",
    *[[Bc(960, 290 + 130 * i, 1600, 110, "blanco", c, 4, 22, sombra=True, en=m, ent="subir"), T(a, 220, 290 + 130 * i, tam=36, c=c, w="b", anc="lm", en=m, t=0.1),
       T(b, 650, 290 + 130 * i, tam=32, c="tinta", w="sb", anc="lm", en=m, t=0.2)] for i, ((a, b, c), m) in enumerate(zip(NUM, "xxabc")) if i >= 2])

# ---- 3. Checklist
CHK = [("separa train y test al azar, estratificado", "nunca por fecha sin querer"), ("fija min_samples_leaf con criterio", "hojas chicas = tasas de ruido"),
       ("traza la curva de profundidad", "elige el máximo de validación"), ("poda con ccp_alpha validado", "GridSearchCV, no a ojo"),
       ("revisa el desvío entre folds", "buen promedio + mucho desvío = inestable"), ("importancia por permutación", "sobre el test"),
       ("traduce las hojas a reglas", "n, tasa y lift de cada segmento"), ("compara con una logística y un bosque", "y decide según para qué es")]
v.b("El checklist para entrenar un árbol, en ocho pasos. [[a]]Separa entrenamiento y prueba al azar, y estratificado. [[b]]Fija el mínimo de clientes por hoja con criterio. [[c]]Traza la curva de profundidad. [[d]]Y poda con un alfa validado.",
    CL(), T("checklist para entrenar un árbol", 960, 160, tam=46, c="tinta", ent="subir"),
    *[[Bc(960, 250 + 95 * i, 1640, 80, "blanco", "verde", 3, 18, en=m, ent="subir"), K(200, 250 + 95 * i, "check", 36, en=m, t=0.1),
       T(a, 250, 250 + 95 * i, tam=32, c="tinta", w="sb", anc="lm", en=m, t=0.15), T(b, 1750, 250 + 95 * i, tam=28, c="gris", w="sb", anc="rm", en=m, t=0.2)]
      for i, ((a, b), m) in enumerate(zip(CHK[:4], "abcd"))], sec="3. El checklist")
v.b("[[a]]Revisa el desvío entre folds, no solo el promedio. [[b]]Mide la importancia por permutación, sobre el test. [[c]]Traduce las hojas a reglas, con tamaño, tasa y lift. [[d]]Y compara con una logística y un bosque, y decide según para qué es el modelo.",
    *[[Bc(960, 250 + 95 * i, 1640, 80, "blanco", "verde", 3, 18, en=m, ent="subir"), K(200, 250 + 95 * i, "check", 36, en=m, t=0.1),
       T(a, 250, 250 + 95 * i, tam=32, c="tinta", w="sb", anc="lm", en=m, t=0.15), T(b, 1750, 250 + 95 * i, tam=28, c="gris", w="sb", anc="rm", en=m, t=0.2)]
      for i, ((a, b), m) in enumerate(zip(CHK, "xxxxabcd")) if i >= 4])

# ---- 4. Alertas del material
AL = ["diap. 13: la nota «~0.168 y ~0.097» está mal: es 0.196 y 0.069 · conteos 211/284 (real 210/285)",
      "diap. 13: los conteos de edad > 40 no salen de la base (real 970 y 2.180) · la conclusión se mantiene",
      "diap. 19: «0.001 · 0.8996 · 42 hojas» no sale de su código (da 0.0013 · 36 hojas)",
      "diap. 7 y clase: el 94 % exige 3 preguntas (con 2: 216 clientes, 85.6 %)",
      "notebook: Gini y entropía «casi idénticos» → aquí 0.906 vs 0.920",
      "diap. 11: 0.269 es redondeo; la raíz exacta es 0.265"]
v.b("Las alertas del material que viste en la serie. [[a]]La nota de la diapositiva trece y sus conteos. [[b]]El resultado de la diapositiva diecinueve. [[c]]La regla del noventa y cuatro por ciento con dos preguntas en vez de tres. [[d]]Gini contra entropía, y el redondeo de la diapositiva once.",
    CL(), alerta(150, texto="alertas del material"),
    *[[Bc(960, 260 + 110 * i, 1700, 92, "blanco", "ambar", 3, 18, en=m, ent="subir", t=0.4 * (i % 2)),
       T(a, 960, 260 + 110 * i, tam=29, c="tinta", w="sb", mw=1640, en=m, t=0.4 * (i % 2) + 0.1)] for i, (a, m) in enumerate(zip(AL, "aabcdd"))],
    sec="4. Alertas del material")

# ---- 5. Preguntas rápidas
v.b("Tres preguntas rápidas para comprobarte. [[a]]Uno: ¿por qué se dice que el árbol es voraz? [[b]]Dos: ¿cuánto vale el Gini de un nodo con quince de una clase y cinco de otra? [[c]]Tres: ¿por qué no hace falta escalar las variables para un árbol?",
    CL(), T("tres preguntas rápidas", 960, 170, tam=48, c="tinta", ent="subir"),
    *[P(q, 960, 330 + 170 * i, "violeta", 44, en=m) for i, (q, m) in enumerate(zip(["1. ¿por qué «voraz»?", "2. Gini de 15 contra 5", "3. ¿por qué no escalar?"], "abc"))],
    sec="5. Tres preguntas rápidas", pausa=0.6)
v.b("Respuestas. [[a]]Uno: porque elige el mejor corte en cada nodo sin mirar los siguientes, así que no garantiza el árbol óptimo. [[b]]Dos: pe igual a cero coma setenta y cinco, y Gini uno menos cero coma cinco seis dos cinco menos cero coma cero seis dos cinco: cero coma tres siete cinco. [[c]]Tres: porque solo pregunta si x está bajo un umbral, y una transformación que respeta el orden mueve el umbral, no la partición.",
    CL(), T("respuestas", 960, 170, tam=48, c="verde", ent="subir"),
    *[[Bc(960, 330 + 170 * i, 1600, 140, "blanco", "verde", 4, 22, sombra=True, en=m, ent="subir"), T(r, 960, 330 + 170 * i, tam=34, c="tinta", w="sb", mw=1520, en=m, t=0.1)]
      for i, (r, m) in enumerate(zip(["1. elige el mejor corte ahora, sin mirar adelante: no garantiza el óptimo",
                                       "2. p = 0.75 → 1 − 0.5625 − 0.0625 = 0.375",
                                       "3. x ≤ umbral: una transformación monótona mueve el umbral, no la partición"], "abc"))])

# ---- 6. Criterio de dominio
DOM = ["calculo a mano Gini, entropía y la ganancia de un corte", "leo un árbol y traduzco sus hojas a reglas con n, tasa y lift",
       "diagnostico sobreajuste con la curva de profundidad", "podo con ccp_alpha elegido por validación cruzada", "explico por qué uso importancia por permutación"]
v.b("¿Cómo sabes si ya dominas la clase? [[a]]Si calculas a mano Gini, entropía y la ganancia de un corte. [[b]]Si lees un árbol y lo traduces a reglas. [[c]]Si diagnosticas el sobreajuste con la curva de profundidad, y podas con un alfa validado. [[d]]Y si puedes explicar por qué usas importancia por permutación.",
    CL(), T("criterio de dominio", 960, 170, tam=48, c="tinta", ent="subir"),
    *[[Bc(960, 290 + 120 * i, 1500, 100, "blanco", "teal", 3, 20, en=m, ent="subir"), K(300, 290 + 120 * i, "check", 40, en=m, t=0.1),
       T(d, 360, 290 + 120 * i, tam=34, c="tinta", w="sb", anc="lm", en=m, t=0.15)] for i, (d, m) in enumerate(zip(DOM, "abccd"))],
    T("y resuelves los bloques 1 a 8 del cuaderno con al menos 80 % de aciertos", 960, 920, tam=32, c="gris", w="sb", en="d", t=0.5), sec="6. Criterio de dominio")

# ---- 7. El consejo del profesor
v.b("Y el consejo con que cerró el profesor. [[a]]Hay cosas que se hacen siempre: estadística descriptiva, y validación cruzada, o al menos separar bien entrenamiento y prueba. [[b]]Lo demás depende del problema. Parte siempre por el modelo más simple, y sofistícalo con lo que tengas. [[c]]Y usa los asistentes de inteligencia artificial, pero con mucho contexto, guiándolos, no pidiéndoles que hagan la pega por ti.",
    CL(), T("el consejo del profesor", 960, 170, tam=48, c="tinta", ent="subir"),
    tarjetas([(400, "verde", "siempre", "descriptiva · validación cruzada", "a"), (960, "violeta", "según el problema", "parte por lo más simple", "b"),
              (1520, "azul", "asistentes de IA", "con contexto, guiándolos", "c")], y=520, w=500, h=340),
    P("cada problema es diferente", 960, 820, "teal", 48, en="c", t=0.8), sec="7. El consejo del profesor")

# ---- 8. Cierre
v.b("Para pensar antes de la próxima clase: [[a]]¿en qué problema de tu trabajo preferirías un árbol de ocho reglas, antes que un modelo con mejor AUC? [[b]]Gracias por acompañar la serie de la clase treinta y cuatro.",
    CL(), T("para pensar", 960, 260, tam=44, c="gris", w="sb", ent="subir"),
    T("¿en qué problema preferirías un árbol de 8 reglas antes que un modelo con mejor AUC?", 960, 430, tam=50, c="violeta", w="b", mw=1500, en="a"),
    P("fin de la serie · Clase 34", 960, 700, "teal", 52, en="b"), sec="8. Cierre")

n = v.guardar(OUT)
print(n, "beats")
