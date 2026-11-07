import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, codigo

OUT = out("guion_c34_07_poda.json")
v = V("Poda: antes o después de crecer")
PO = D["poda"]
GA, GU, GH, D19 = D["grid_alpha"], D["grid_auc"], D["grid_hojas"], D["diap19_cv5"]
c_ = lambda e, a, h: e + a * h  # noqa: E731

v.portada("Poda: antes o después", "Pre-poda con límites, post-poda con ccp_alpha, y cómo elegir alfa sin hacer trampa",
          "Séptima parada. Ya sabes que el árbol sin límites memoriza. Hoy aprendes las dos formas de frenarlo: antes de crecer, o después, podando.")

# ---- 1. Hook + analogía
v.b("Piensa en un jardinero con un arbusto. [[a]]Puede ponerle un tutor y limitar cuánto crece desde el principio. [[b]]O puede dejarlo crecer libre, y después cortar las ramas que no aportan. [[c]]Con los árboles de decisión pasa exactamente lo mismo: pre-poda, o post-poda.",
    CL(), T("dos formas de frenar el crecimiento", 960, 200, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "antes", "limitar el crecimiento desde el principio", "a"), (1360, "verde", "después", "dejar crecer y cortar lo que no aporta", "b")], y=500, w=660, h=320),
    P("pre-poda  ·  post-poda", 960, 800, "violeta", 56, en="c"), ilustrativo(1000, texto="analogía"), sec="1. Dos formas de frenar")

# ---- 2. Pre-poda
HP = [("max_depth", "profundidad máxima", "3 – 8"), ("min_samples_leaf", "mínimo de clientes por hoja", "5 – 60"),
      ("min_samples_split", "mínimo para intentar partir", "10 – 50"), ("max_leaf_nodes", "cantidad total de hojas", "10 – 50")]
v.b("La pre-poda pone límites antes de entrenar. [[a]]Profundidad máxima. [[b]]Mínimo de clientes por hoja. [[c]]Mínimo de clientes para intentar partir un nodo. [[d]]Y cantidad máxima de hojas.",
    CL(), T("pre-poda: límites antes de entrenar", 960, 180, tam=48, c="azul", ent="subir"),
    *[[Bc(960, 320 + 130 * i, 1500, 110, "blanco", "azul", 3, 20, en=m, ent="subir"), T(a, 300, 320 + 130 * i, tam=40, c="azul", w="m", anc="lm", en=m, t=0.1),
       T(b, 1000, 320 + 130 * i, tam=34, c="tinta", w="sb", anc="lm", en=m, t=0.2), T(r, 1680, 320 + 130 * i, tam=30, c="gris", w="sb", anc="rm", en=m, t=0.3)]
      for i, ((a, b, r), m) in enumerate(zip(HP, "abcd"))],
    T("rangos típicos de la presentación genérica (Arbol de decision.pptx)", 960, 900, tam=28, c="gris", w="sb", en="d", t=0.5), sec="2. Pre-poda")
v.b("El profesor dio un consejo práctico: [[a]]no pongas profundidad uno, porque queda un árbol pelado. Parte con tres o cuatro. [[b]]Y el mínimo de casos por hoja depende de lo que investigas: si las hojas son campañas, cada hoja necesita clientes suficientes. [[c]]La pre-poda es rápida y directa. Su riesgo: un corte pobre arriba puede esconder uno muy bueno debajo.",
    CL(), P("parte con profundidad 3 o 4, no 1", 960, 260, "azul", 52, en="a"),
    P("hojas mínimas según el uso: una campaña necesita clientes", 960, 420, "teal", 46, en="b"),
    tarjetas([(560, "verde", "rápida", "y directa", "c"), (1360, "rojo", "riesgo", "un corte pobre esconde uno bueno debajo", "c")], y=680, w=640, h=260))

# ---- 3. Post-poda: la fórmula
v.b("La post-poda deja crecer el árbol completo, y después recorta. [[a]]El criterio es el costo-complejidad: el costo de un árbol es su error, más alfa por el número de hojas. [[b]]Alfa es cuánto cobra cada hoja. En sklearn se llama ccp_alpha.",
    CL(), T("post-poda: dejar crecer y recortar", 960, 190, tam=48, c="verde", ent="subir"),
    caja_formula("costo(T) = error(T) + α · nº de hojas", y=400, c="violeta", tam=60, en="a", w=1300),
    P("α = el precio de cada hoja  ·  ccp_alpha en sklearn  ·  cp en rpart", 960, 600, "teal", 42, en="b"), sec="3. Post-poda: costo-complejidad")
v.b("Con alfa igual a cero, ninguna hoja cuesta nada, y sobrevive el árbol completo. [[a]]Al subir alfa, cada hoja tiene que justificar su existencia, reduciendo el error más de lo que cuesta. [[b]]Para cada alfa hay un único subárbol que minimiza el costo. Y la validación cruzada elige cuál alfa usar.",
    CL(), P("α = 0 → sobrevive el árbol completo", 960, 280, "gris", 52, ent="subir"),
    P("α sube → cada hoja debe pagar su precio", 960, 440, "violeta", 52, en="a"),
    P("un subárbol por α · la validación cruzada elige α", 960, 600, "verde", 50, en="b"))

# ---- 4. Ejemplo del apunte
v.b("Hagámoslo a mano, con el ejemplo del apunte. [[a]]Un árbol grande, T uno: error cero coma diez, con veinte hojas. [[b]]Un árbol chico, T dos: error cero coma trece, con cinco hojas. ¿Cuál gana?",
    CL(), P("ejemplo del apunte", 960, 160, "gris", 34, ent="subir"),
    tarjetas([(560, "azul", "T1 · grande", "error 0.10 · 20 hojas", "a"), (1360, "verde", "T2 · chico", "error 0.13 · 5 hojas", "b")], y=420, w=640, h=280),
    sec="4. Ejemplo: ¿qué árbol gana?")
v.b("Con alfa igual a cero coma cero cero uno: [[a]]T uno cuesta cero coma diez más cero coma cero dos, igual a cero coma ciento veinte. T dos, cero coma trece más cero coma cero cero cinco: cero coma ciento treinta y cinco. Gana el grande. [[b]]Con alfa igual a cero coma cero cero cinco: [[c]]T uno cuesta cero coma dos. T dos, cero coma ciento cincuenta y cinco. Ahora gana el chico.",
    T("α = 0.001", 300, 650, tam=44, c="violeta", w="b"),
    T(f"T1: 0.10 + 20 × 0.001 = {c_(0.10, 0.001, 20):.3f}", 760, 620, tam=36, c="azul", w="sb", anc="lm", en="a"),
    T(f"T2: 0.13 + 5 × 0.001 = {c_(0.13, 0.001, 5):.3f}", 760, 680, tam=36, c="verde", w="sb", anc="lm", en="a", t=0.4), K(1700, 620, "check", 50, en="a", t=0.9),
    T("α = 0.005", 300, 820, tam=44, c="violeta", w="b", en="b"),
    T(f"T1: 0.10 + 20 × 0.005 = {c_(0.10, 0.005, 20):.3f}", 760, 790, tam=36, c="azul", w="sb", anc="lm", en="c"),
    T(f"T2: 0.13 + 5 × 0.005 = {c_(0.13, 0.005, 5):.3f}", 760, 850, tam=36, c="verde", w="sb", anc="lm", en="c", t=0.4), K(1700, 850, "check", 50, en="c", t=0.9))
v.b("¿Dónde empatan? [[a]]Cuando cero coma diez más veinte alfa es igual a cero coma trece más cinco alfa. [[b]]Quince alfa es igual a cero coma cero tres: alfa igual a cero coma cero cero dos. Por encima de eso, conviene el árbol podado.",
    CL(), P("ejemplo del apunte", 960, 160, "gris", 34),
    T("0.10 + 20α = 0.13 + 5α", 960, 330, tam=64, c="tinta", w="sb", en="a"),
    T("15α = 0.03", 960, 460, tam=60, c="gris", w="sb", en="b"), T("α =", 900, 590, tam=72, c="tinta", w="sb", anc="rm", en="b", t=0.5),
    N(0.03 / 15, 930, 590, "{:.3f}", 96, "violeta", anc="lm", en="b", t=0.7), P("α mayor → gana el árbol podado", 960, 760, "verde", 48, en="b", t=1.2))

# ---- 5. Con los datos de la clase
GX = dict(x0=300, y0=260, x1=1200, y1=760)
v.b("Ahora con los datos de la clase. [[a]]Con alfa cero, el árbol completo: doscientas veinticuatro hojas, y AUC de validación cero coma ocho dos siete. [[b]]Con alfa cero coma cero cero cero cinco quedan ciento dieciséis hojas, y el AUC sube a cero coma ocho cinco uno. [[c]]Con alfa cero coma cero cero uno, cuarenta y dos hojas, y cero coma nueve cero cero: el punto dulce.",
    CL(), T("cada α, su árbol · AUC con validación cruzada", 960, 170, tam=44, c="tinta", ent="subir"),
    G(GX["x0"], GX["y0"], GX["x1"], GX["y1"], -0.5, 5.5, 0.78, 0.92,
      [dict(xs=list(range(6)), ys=[p["auc"] for p in PO], color="verde", grosor=7, marcadores=11, revela=0.0)],
      xticks=[], yticks=[0.80, 0.84, 0.88, 0.92], yfmt="{:.2f}", id="gpo"),
    *[[T(f"{p['alpha']:g}", i, 0.765, tam=28, c="violeta", w="b", ref="gpo"), T(f"{p['hojas']} hojas", i, 0.755, tam=24, c="gris", w="sb", ref="gpo", dy=20)] for i, p in enumerate(PO)],
    T("ccp_alpha", 1260, 806, tam=28, c="violeta", w="b", anc="lm"),
    U("gpo", **{"series.0.revela": 0.01}, en="a", dur=0.1), T(f"α = 0 → {PO[0]['hojas']} hojas · {PO[0]['auc']:.3f}", 1560, 360, tam=36, c="gris", w="b", en="a", t=0.3),
    U("gpo", **{"series.0.revela": 0.2}, en="b", dur=0.8), T(f"α = 0.0005 → {PO[1]['hojas']} · {PO[1]['auc']:.3f}", 1560, 440, tam=36, c="tinta", w="b", en="b", t=0.5),
    U("gpo", **{"series.0.revela": 0.4}, en="c", dur=0.8), T(f"α = 0.001 → {PO[2]['hojas']} · {PO[2]['auc']:.3f}", 1560, 520, tam=36, c="verde", w="b", en="c", t=0.5),
    real(1050), sec="5. Con los datos de la clase")
v.b("Y si sigues subiendo alfa, el árbol se queda corto. [[a]]Con cero coma cero cero dos, quince hojas y cero coma ocho ocho tres. [[b]]Con cero coma cero cero cinco, cinco hojas. [[c]]Con cero coma cero uno, apenas tres hojas, y el AUC cae a cero coma ocho cero cinco. Podar de más también es un error.",
    U("gpo", **{"series.0.revela": 0.6}, en="a", dur=0.8), T(f"α = 0.002 → {PO[3]['hojas']} · {PO[3]['auc']:.3f}", 1560, 600, tam=36, c="tinta", w="b", en="a", t=0.5),
    U("gpo", **{"series.0.revela": 0.8}, en="b", dur=0.8), T(f"α = 0.005 → {PO[4]['hojas']} · {PO[4]['auc']:.3f}", 1560, 680, tam=36, c="tinta", w="b", en="b", t=0.5),
    U("gpo", **{"series.0.revela": 1.0}, en="c", dur=0.8), T(f"α = 0.01 → {PO[5]['hojas']} · {PO[5]['auc']:.3f}", 1560, 760, tam=36, c="rojo", w="b", en="c", t=0.5),
    Bc(2, PO[2]["auc"], 34, 34, None, "verde", 5, 17, ref="gpo", en="c", t=0.8, ent="pop"),
    P("ni de menos ni de más", 1560, 880, "violeta", 40, en="c", t=1.2))
v.b("El profesor lo planteó como una decisión de negocio. [[a]]Con quince hojas, un AUC de cero coma ocho ocho, fácil de explicar. [[b]]Con cuarenta y dos hojas, más AUC, pero quizás demasiadas reglas para explicar. [[c]]Y con cinco hojas, poco, pero quizás suficiente para una encuesta pequeña. Según el problema, tú decides.",
    CL(), T("¿cuántas hojas quieres explicar?", 960, 200, tam=50, c="tinta", ent="subir"),
    tarjetas([(400, "gris", f"{PO[4]['hojas']} hojas", f"AUC {PO[4]['auc']:.3f} · una encuesta pequeña", "c"), (960, "verde", f"{PO[3]['hojas']} hojas", f"AUC {PO[3]['auc']:.3f} · fácil de explicar", "a"),
              (1520, "violeta", f"{PO[2]['hojas']} hojas", f"AUC {PO[2]['auc']:.3f} · muchas reglas", "b")], y=520, w=480, h=320),
    P("según el problema, tú decides", 960, 820, "teal", 48, en="c", t=0.8))

# ---- 6. Elegir alfa con validación cruzada
v.b("Pero alfa no se elige a ojo. [[a]]Primero, sklearn calcula los alfas donde el árbol óptimo cambia: con estos datos, ciento diez alfas. [[b]]El notebook toma uno de cada cinco, veintidós candidatos, y GridSearchCV los compara con validación cruzada.",
    CL(), codigo(["path = arbol.cost_complexity_pruning_path(X, y)", "alphas = path.ccp_alphas[:-1]",
                  "grid = GridSearchCV(DecisionTreeClassifier(random_state=0),", "         {'ccp_alpha': alphas[::5]}, cv=cv, scoring='roc_auc')"],
                 y=360, tam=34, dt=0.4),
    P(f"{D['n_alphas']} alfas relevantes", 560, 640, "violeta", 46, en="a"), P(f"uno de cada 5 → {D['grid_n_cand']} candidatos", 1360, 640, "teal", 46, en="b"),
    sec="6. Elegir alfa con validación cruzada")
v.b("¿El resultado? [[a]]El mejor alfa es cero coma cero cero uno tres. [[b]]Deja un árbol de treinta y seis hojas, [[c]]con AUC de validación cruzada de cero coma nueve cero nueve.",
    CL(), T("resultado de GridSearchCV (notebook de la clase)", 960, 200, tam=42, c="tinta", ent="subir"),
    *[[Bc(x, 480, 460, 280, "blanco", c, 5, 30, sombra=True, en=m, ent="subir"), T(lab, x, 400, tam=36, c="gris", w="sb", en=m, t=0.1),
       N(val, x, 500, fmt, 84, c, en=m, t=0.2)] for x, lab, val, fmt, c, m in
      ((460, "ccp_alpha", GA, "{:.4f}", "violeta", "a"), (960, "hojas", GH, "{:.0f}", "verde", "b"), (1460, "AUC (CV)", GU, "{:.3f}", "azul", "c"))],
    real(1050))

# ---- 7. Alerta del material
v.b("Ahora, una alerta del material. [[a]]La diapositiva diecinueve muestra como resultado de ese código: alfa cero coma cero cero uno, AUC cero coma ocho nueve nueve seis, y cuarenta y dos hojas. [[b]]Corrí ese mismo código, tal cual está en la diapositiva: elige alfa cero coma cero cero uno tres, con treinta y seis hojas, y AUC cero coma ocho ocho uno. [[c]]El cero coma ocho nueve nueve seis no sale de ese código: es la fila de alfa cero coma cero cero uno de la tabla de la diapositiva dieciocho.",
    CL(), alerta(),
    T("diapositiva 19: {'ccp_alpha': 0.001} 0.8996 · hojas: 42", 960, 320, tam=40, c="tinta", w="sb", en="a", id="d19"),
    U("d19", tachado=1.0, color="gris", en="b", dur=0.6),
    Bc(560, 560, 640, 220, "blanco", "verde", 5, 28, sombra=True, en="b", ent="subir", t=0.4), T("su propio código (cv=5)", 560, 500, tam=32, c="gris", w="sb", en="b", t=0.5),
    T(f"α {D19['alpha']:.4f} · {D19['hojas']} hojas · {D19['auc']:.3f}", 560, 580, tam=38, c="verde", w="b", en="b", t=0.6),
    Bc(1360, 560, 640, 220, "blanco", "verde", 5, 28, sombra=True, en="b", ent="subir", t=0.8), T("notebook (CV con shuffle)", 1360, 500, tam=32, c="gris", w="sb", en="b", t=0.9),
    T(f"α {GA:.4f} · {GH} hojas · {GU:.3f}", 1360, 580, tam=38, c="verde", w="b", en="b", t=1.0),
    P(f"0.8996 = la fila α = 0.001 de la diapositiva 18", 960, 800, "ambar", 44, en="c"), sec="7. Alerta: la diapositiva 19")

# ---- 8. Error típico
v.b("El error típico: [[a]]elegir alfa a ojo, o mirando el AUC del test. [[b]]Eso es ajustar al test, y el número que reportas queda optimista. [[c]]Alfa se elige con validación cruzada sobre el entrenamiento, como cualquier hiperparámetro.",
    CL(), T("elegir α a ojo… o mirando el test", 960, 280, tam=60, c="tinta", en="a", id="e1"),
    P("ajustar al test = AUC optimista", 960, 450, "rojo", 52, en="b"), U("e1", tachado=1.0, color="gris", en="c", dur=0.6),
    P("α con validación cruzada (GridSearchCV)", 960, 620, "verde", 52, en="c", t=0.4), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo, con el ejemplo del apunte. [[a]]Si alfa vale cero coma cero cero tres, ¿qué árbol gana: T uno, con error cero coma diez y veinte hojas, o T dos, con cero coma trece y cinco hojas?",
    CL(), P("α = 0.003", 960, 280, "violeta", 70, en="a"), T("T1: 0.10 y 20 hojas   ·   T2: 0.13 y 5 hojas", 960, 460, tam=48, c="tinta", w="sb", en="a", t=0.4),
    T("¿cuál gana?", 960, 620, tam=70, c="tinta", en="a", t=0.8), sec="9. Piensa un segundo", pausa=0.6)
v.b("T uno cuesta cero coma diez más cero coma cero seis: cero coma dieciséis. [[a]]T dos cuesta cero coma trece más cero coma cero uno cinco: cero coma ciento cuarenta y cinco. [[b]]Gana el chico. Tenía que ser así: tres milésimas está por encima del empate en dos milésimas.",
    CL(), T(f"T1: 0.10 + 20 × 0.003 = {c_(0.10, 0.003, 20):.3f}", 960, 300, tam=52, c="azul", w="sb"),
    T(f"T2: 0.13 + 5 × 0.003 = {c_(0.13, 0.003, 5):.3f}", 960, 420, tam=52, c="verde", w="sb", en="a"), K(1600, 420, "check", 60, en="b"),
    P("0.003 > 0.002 (el empate) → gana el podado", 960, 620, "verde", 50, en="b", t=0.4))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]pre-poda es poner límites antes de crecer; post-poda es dejar crecer y recortar. [[b]]ccp_alpha es el precio de cada hoja: más alto, menos hojas. [[c]]Y alfa se elige con validación cruzada. Aquí, cero coma cero cero uno tres, treinta y seis hojas, y cero coma nueve cero nueve.",
    CL(), P("pre-poda: límites · post-poda: recortar", 960, 300, "violeta", 52, en="a"),
    P("ccp_alpha = precio por hoja", 960, 480, "teal", 54, en="b"), P(f"α por CV: {GA:.4f} · {GH} hojas · {GU:.3f}", 960, 660, "verde", 52, en="c"),
    sec="10. Para llevar")
v.b("Siguiente parada: [[a]]¿qué variables importan en un árbol? Dos maneras de medirlo, y una de ellas engaña.",
    CL(), siguiente(8, "Importancia: impureza contra permutación"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
