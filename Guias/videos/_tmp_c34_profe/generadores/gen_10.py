import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula, codigo

OUT = out("guion_c34_10_bosque.json")
v = V("El salto al bosque")
RN, CF, FI = D["rf_n"], D["rf_cfg"], D["final"]
CU = D["curva"]
best = max(CU, key=lambda c: c["va"])
var = lambda r, b: r + (1 - r) / b  # noqa: E731

v.portada("El salto al bosque", "Random Forest: promediar muchos árboles para vencer la inestabilidad",
          "Décima parada. Si el problema del árbol es la inestabilidad, la solución es promediar muchos. Hoy entiendes por qué el bosque predice tanto mejor, y qué se pierde en el camino.")

# ---- 1. Hook
v.b("El profesor lo dijo sin rodeos: [[a]]hoy es lo mismo hacer un árbol que un bosque. Técnicamente no demora nada, es una línea más de Python. [[b]]«Yo prefiero que hagan un bosque». [[c]]Porque mirar un solo árbol es como mirar una sola arista del problema.",
    CL(), T("«hoy es lo mismo hacer un árbol que un bosque»", 960, 240, tam=54, c="tinta", en="a"),
    T("una línea más de Python", 960, 340, tam=40, c="gris", w="sb", en="a", t=0.5),
    P("«yo prefiero que hagan un bosque»", 960, 490, "verde", 54, en="b"),
    P("un árbol = una sola arista del problema", 960, 650, "violeta", 48, en="c"), sec="1. Un árbol o un bosque")

# ---- 2. Analogía
v.b("Piensa en un diagnóstico difícil. [[a]]Un solo médico, que vio un grupo particular de pacientes, puede equivocarse por sus manías. [[b]]Ahora pregúntale a trescientos médicos, cada uno formado con pacientes distintos, y promedia sus opiniones. [[c]]Las manías de cada uno se cancelan, y queda lo que todos ven.",
    CL(), Bc(480, 470, 520, 340, "blanco", "gris", 5, 30, sombra=True, en="a", ent="subir"), T("1 médico", 480, 350, tam=44, c="gris", en="a", t=0.1),
    Pt([(1, [460, 440, 500, 480], "gris")], 26, en="a", t=0.2), T("sus manías pesan", 480, 560, tam=34, c="gris", w="sb", en="a", t=0.3),
    Bc(1360, 470, 760, 340, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), T("300 médicos", 1360, 350, tam=44, c="verde", en="b", t=0.1),
    Pt([(60, [1060, 400, 1660, 540], "verde")], 10, visibles=60, en="b", t=0.2, dur=1.2), T("se promedian", 1360, 590, tam=34, c="verde", w="sb", en="b", t=0.5),
    P("las manías se cancelan · queda lo común", 960, 800, "violeta", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: muchos médicos")

# ---- 3. La receta
v.b("La receta de un Random Forest tiene tres pasos. [[a]]Uno: para cada árbol, se sortean tres mil ciento cincuenta clientes de los tres mil ciento cincuenta, con reemplazo. Algunos entran varias veces, y otros no entran. Eso se llama bootstrap.",
    CL(), T("la receta en 3 pasos", 960, 170, tam=50, c="tinta", ent="subir"),
    Bc(960, 450, 1500, 360, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("1. bootstrap", 960, 320, tam=50, c="azul", en="a", t=0.1),
    T(f"{mil(D['n'])} sorteados de {mil(D['n'])}, con reemplazo", 960, 410, tam=40, c="tinta", w="sb", en="a", t=0.2),
    Pt([(12, [520, 480, 1400, 560], "azul")], 20, en="a", t=0.4), T("algunos repetidos · otros afuera", 960, 600, tam=34, c="gris", w="sb", en="a", t=0.6),
    sec="3. La receta")
v.b("¿Cuántos quedan afuera? [[a]]La probabilidad de que un cliente no salga en ninguno de los sorteos es uno menos uno sobre ene, elevado a ene. [[b]]Con tres mil ciento cincuenta clientes da cero coma tres seis ocho: cerca de un tercio. [[c]]Ese tercio se llama out-of-bag, y sirve para estimar el error sin separar un test aparte.",
    CL(), caja_formula("P(quedar afuera) = (1 − 1/n)ⁿ", y=280, c="azul", tam=60, en="a", w=1100),
    T(f"(1 − 1/{mil(D['n'])})^{mil(D['n'])} =", 900, 470, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(D["oob_teorico"], 930, 470, "{:.3f}", 84, "azul", anc="lm", en="b", t=0.5),
    T("≈ un tercio (tiende a 1/e)", 960, 580, tam=40, c="gris", w="sb", en="b", t=1.0), P("out-of-bag: estimar el error sin un test aparte", 960, 730, "teal", 46, en="c"))
v.b("Dos: en cada corte, el árbol solo puede mirar algunas variables, sorteadas de nuevo cada vez. [[a]]Por defecto, la raíz cuadrada del total: con doce variables, tres. [[b]]Así ningún árbol puede apoyarse siempre en reclamos. [[c]]Y tres: se entrenan cientos de árboles así, y la predicción final es el promedio de sus probabilidades.",
    CL(), Bc(960, 330, 1500, 260, "blanco", "violeta", 5, 30, sombra=True, ent="subir"), T("2. variables al azar en cada corte", 960, 260, tam=46, c="violeta", t=0.1),
    T(f"max_features = 'sqrt' → √12 ≈ {D['sqrt_p']} de {D['p']}", 960, 340, tam=40, c="tinta", w="sb", en="a"),
    T("ningún árbol puede apoyarse siempre en reclamos", 960, 410, tam=34, c="rojo", w="sb", en="b"),
    Bc(960, 650, 1500, 200, "blanco", "verde", 5, 30, sombra=True, en="c", ent="subir"), T("3. repetir y promediar", 960, 610, tam=46, c="verde", en="c", t=0.1),
    T("predicción = promedio de las probabilidades de todos los árboles", 960, 680, tam=34, c="tinta", w="sb", en="c", t=0.2))

# ---- 4. Por qué funciona
v.b("¿Por qué funciona? [[a]]Cada árbol sobreajusta, pero cada uno sobreajusta a un ruido distinto, porque vio otra muestra y otras variables. [[b]]Hay una fórmula: si promedias B árboles con varianza sigma al cuadrado, y correlación ro entre ellos, la varianza del promedio es ro sigma al cuadrado, más uno menos ro, sigma al cuadrado, sobre B.",
    CL(), P("cada árbol sobreajusta a un ruido DISTINTO", 960, 230, "violeta", 50, en="a"),
    caja_formula("Var(promedio) = ρσ² + (1 − ρ)σ² / B", y=460, c="azul", tam=60, en="b", w=1200),
    T("B = número de árboles · ρ = correlación entre árboles", 960, 600, tam=36, c="gris", w="sb", en="b", t=0.6), sec="4. Por qué funciona")
v.b("Ejemplo del apunte, con sigma al cuadrado igual a uno. [[a]]Con ro cero coma tres y diez árboles: cero coma tres más cero coma siete sobre diez, igual a cero coma treinta y siete. [[b]]Con cien árboles: cero coma trescientos siete. Casi no baja más: el piso es ro. [[c]]Pero si los árboles no estuvieran correlacionados, con cien árboles la varianza sería cero coma cero uno. Por eso el paso dos, que descorrelaciona los árboles, es esencial.",
    CL(), P("ejemplo del apunte · σ² = 1", 960, 170, "gris", 34, ent="subir"),
    T(f"ρ = 0.3 · B = 10:   0.3 + 0.7/10 = {var(0.3, 10):.2f}", 960, 320, tam=46, c="tinta", w="sb", en="a"),
    T(f"ρ = 0.3 · B = 100:  0.3 + 0.7/100 = {var(0.3, 100):.3f}", 960, 430, tam=46, c="tinta", w="sb", en="b"),
    P("el piso es ρσ²: más árboles ya no ayudan", 960, 540, "ambar", 44, en="b", t=0.6),
    T(f"ρ = 0 · B = 100:   1/100 = {var(0.0, 100):.2f}", 960, 660, tam=46, c="verde", w="sb", en="c"),
    P("descorrelacionar (paso 2) es esencial", 960, 790, "verde", 48, en="c", t=0.6))

# ---- 5. Cuántos árboles
GX = dict(x0=300, y0=250, x1=1200, y1=820)
v.b("Con los datos de la clase: ¿cuántos árboles hacen falta? [[a]]Un solo árbol del bosque da cero coma siete nueve siete: peor que el árbol podado. [[b]]Con cinco árboles, cero coma nueve tres uno. [[c]]Con veinticinco, cero coma nueve seis ocho. Y con cien y con trescientos, cero coma nueve siete tres y cero coma nueve siete cuatro.",
    CL(), T("AUC con validación cruzada según el número de árboles", 960, 170, tam=42, c="tinta", ent="subir"),
    G(GX["x0"], GX["y0"], GX["x1"], GX["y1"], -0.4, 4.4, 0.76, 1.0,
      [dict(xs=list(range(5)), ys=[r[1] for r in RN], color="verde", grosor=7, marcadores=11, revela=0.0)],
      xticks=[], yticks=[0.8, 0.85, 0.9, 0.95, 1.0], yfmt="{:.2f}", id="gb"),
    *[T(str(r[0]), i, 0.745, tam=30, c="gris", w="sb", ref="gb") for i, r in enumerate(RN)], T("número de árboles", 750, 900, tam=32, c="tinta", w="sb"),
    U("gb", **{"series.0.revela": 0.01}, en="a", dur=0.1), T(f"1 árbol: {RN[0][1]:.3f}", 1560, 360, tam=40, c="rojo", w="b", en="a", t=0.3),
    U("gb", **{"series.0.revela": 0.25}, en="b", dur=0.8), T(f"5: {RN[1][1]:.3f}", 1560, 450, tam=40, c="tinta", w="b", en="b", t=0.5),
    U("gb", **{"series.0.revela": 1.0}, en="c", dur=2.0), T(f"25: {RN[2][1]:.3f}", 1560, 540, tam=40, c="tinta", w="b", en="c", t=0.5),
    T(f"100: {RN[3][1]:.3f}  ·  300: {RN[4][1]:.3f}", 1560, 630, tam=40, c="verde", w="b", en="c", t=1.6),
    real(1050), sec="5. ¿Cuántos árboles?")
v.b("Dos lecciones. [[a]]La ganancia no está en los árboles, que solos son peores: está en promediarlos. [[b]]Y se estabiliza cerca de cien árboles. Agregar más no empeora, solo cuesta tiempo.",
    P("la ganancia está en PROMEDIAR", 1560, 740, "violeta", 36, en="a"), P("se estabiliza cerca de 100", 1560, 830, "teal", 36, en="b"))
v.b("¿Qué aporta cada fuente de azar? [[a]]Solo bootstrap, con todas las variables en cada corte: cero coma nueve seis nueve. [[b]]Bootstrap más el subconjunto de variables: cero coma nueve siete cuatro, la mejor. [[c]]Y solo el subconjunto de variables, sin bootstrap: cero coma nueve seis ocho. Las dos fuentes juntas ganan.",
    CL(), T("cada fuente de azar · bosque de 200 árboles", 960, 170, tam=44, c="tinta", ent="subir"),
    *[[Bx(x - 150, 840 - (val - 0.95) * 12000, x + 150, 840, c, None, 0, 12, grow="y", en=m),
       N(val, x, 840 - (val - 0.95) * 12000 - 44, "{:.4f}", 52, c, en=m, t=0.3), T(lab, x, 890, tam=32, c=c, w="sb", mw=420, en=m)]
      for x, val, lab, c, m in ((460, CF["solo_bootstrap"], "solo bootstrap", "azul", "a"), (960, CF["bootstrap_sqrt"], "bootstrap + variables al azar", "verde", "b"),
                                (1460, CF["sin_bootstrap_sqrt"], "solo variables al azar", "violeta", "c"))],
    L(260, 840, 1660, 840, "tinta", 4), T("eje desde 0.95", 1780, 840, tam=24, c="gris", w="sb"), real(1050))

# ---- 6. Comparación final
MODS = [("árbol prof. 3", FI["arbol3"], "gris"), ("árbol prof. 5", FI["arbol5"], "gris"), ("árbol podado (CV)", FI["podado"], "verde"),
        ("regresión logística", FI["logistica"], "azul"), ("Random Forest (300)", FI["rf300"], "violeta")]
v.b("La comparación final de la clase, todos con la misma validación cruzada. [[a]]Árbol de profundidad tres: cero coma ocho cinco cero. Profundidad cinco: cero coma nueve cero seis. [[b]]Árbol podado: cero coma nueve cero nueve. Logística: cero coma nueve dos cero. [[c]]Y el bosque: cero coma nueve siete cuatro.",
    CL(), T("comparación final · AUC con validación cruzada", 960, 170, tam=44, c="tinta", ent="subir"),
    *[[T(lab, 620, 290 + 110 * i, tam=36, c="tinta", w="sb", anc="rm", en=m),
       Bx(640, 290 + 110 * i - 30, 640 + (val - 0.8) * 5500, 290 + 110 * i + 30, c, None, 0, 10, grow="x", en=m, t=0.2 * (i % 2)),
       N(val, 660 + (val - 0.8) * 5500, 290 + 110 * i, "{:.3f}", 44, c, anc="lm", en=m, t=0.2 * (i % 2) + 0.3)]
      for i, ((lab, val, c), m) in enumerate(zip(MODS, ["a", "a", "b", "b", "c"]))],
    T("eje desde 0.80", 640, 860, tam=26, c="gris", w="sb", anc="lm"), real(1050), sec="6. La comparación final")
v.b("Ahora, una aclaración del material. [[a]]La diapositiva veinticinco dice que el mejor árbol llega a cero coma nueve uno uno, y la veintisiete dice que solo llega a cero coma nueve cero seis. [[b]]No se contradicen: cero coma nueve uno uno es el árbol de profundidad seis, y cero coma nueve cero seis el de profundidad cinco. [[c]]Contra el mejor árbol, el bosque gana seis puntos de AUC.",
    CL(), alerta(170, texto="aclaración del material"),
    tarjetas([(560, "ambar", "diapositiva 25", f"«el mejor árbol solo {best['va']:.3f}»", "a"), (1360, "ambar", "diapositiva 27", f"«solo llega a {FI['arbol5']:.3f}»", "a")], y=400, w=640, h=240),
    T(f"{best['va']:.3f} = profundidad {best['d']}   ·   {FI['arbol5']:.3f} = profundidad 5", 960, 620, tam=44, c="verde", w="b", en="b"),
    P(f"bosque − mejor árbol = {FI['rf300']:.3f} − {best['va']:.3f} = {(FI['rf300'] - best['va']) * 100:.1f} puntos", 960, 760, "violeta", 44, en="c"))

# ---- 7. El precio
v.b("Pero el bosque tiene un precio. [[a]]El árbol de profundidad tres cabe en una lámina, y se convierte en siete reglas para el call center. [[b]]Un bosque de trescientos árboles no se puede dibujar. [[c]]Ganamos seis puntos de AUC, y perdimos la explicación. Es el mismo trade-off de la clase treinta y dos con PCA.",
    CL(), tarjetas([(560, "verde", "árbol prof. 3", f"cabe en una lámina · 7 reglas · AUC {FI['arbol3']:.3f}", "a"),
                    (1360, "violeta", "bosque de 300", f"no se puede dibujar · AUC {FI['rf300']:.3f}", "b")], y=420, w=660, h=320),
    P("+6 puntos de AUC  ·  −la explicación", 960, 700, "rojo", 52, en="c"), T("el mismo trade-off de la clase 32 (PCA)", 960, 820, tam=38, c="gris", w="sb", en="c", t=0.6),
    sec="7. El precio del bosque")
v.b("Y una tensión que ya viste en el video seis: [[a]]el profesor dijo que un AUC sobre cero coma nueve en datos reales es sospechoso. El bosque da cero coma nueve siete cuatro, [[b]]con una base que él mismo llamó de libro. [[c]]Disfruta el número para aprender, pero en tu trabajo, un cero coma nueve siete se investiga antes de presentarlo.",
    CL(), P("«sobre 0.90, algo malo está pasando»", 960, 260, "rojo", 50, en="a"),
    T(f"bosque: {FI['rf300']:.3f}", 960, 400, tam=72, c="violeta", w="b", en="a", t=0.5),
    P("base «de libro»", 960, 540, "ambar", 48, en="b"), P("en datos reales: se investiga antes de presentarlo", 960, 690, "violeta", 46, en="c"))

# ---- 8. Para qué es el modelo
v.b("Entonces, ¿cuál gana? [[a]]La pregunta correcta no es cuál número se ve más alto, sino para qué es el modelo. [[b]]Si hay que explicarle la decisión a un cliente, a un regulador o a marketing, un árbol podado gana. [[c]]Si solo importa predecir, el bosque. Y el profesor confesó su favorito entre los modelos clásicos: el Gradient Boosting, árboles que se corrigen uno tras otro, que vas a ver más adelante.",
    CL(), T("¿para qué es el modelo?", 960, 200, tam=60, c="tinta", ent="subir", en="a"),
    tarjetas([(560, "verde", "explicar", "cliente · regulador · marketing → árbol podado", "b"), (1360, "violeta", "predecir", "solo importa el AUC → bosque", "c")], y=480, w=660, h=300),
    P("favorito del profesor: Gradient Boosting (árboles secuenciales)", 960, 780, "ambar", 42, en="c", t=1.2), sec="8. ¿Para qué es el modelo?")

# ---- 9. Error típico
v.b("El error típico: [[a]]elegir el modelo solo por el número más alto. [[b]]O al revés: usar un bosque cuando el negocio necesita reglas que pueda auditar. [[c]]Primero define para qué es el modelo; después compara números.",
    CL(), T("elegir solo por el AUC más alto", 960, 280, tam=58, c="tinta", en="a", id="e1"),
    T("… o un bosque cuando piden reglas auditables", 960, 400, tam=46, c="gris", w="sb", en="b"), U("e1", tachado=1.0, color="gris", en="c", dur=0.6),
    P("primero: ¿para qué es? · después: los números", 960, 580, "verde", 50, en="c", t=0.4), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un solo árbol del bosque da cero coma siete nueve siete, peor que el árbol podado. [[b]]¿Cómo puede ser que trescientos de esos árboles juntos den cero coma nueve siete cuatro?",
    CL(), P(f"1 árbol del bosque: {RN[0][1]:.3f}", 960, 300, "rojo", 54, en="a"), P(f"300 juntos: {RN[4][1]:.3f}", 960, 450, "verde", 54, en="b"),
    T("¿cómo puede ser?", 960, 640, tam=70, c="tinta", en="b", t=0.5), sec="10. Piensa un segundo", pausa=0.6)
v.b("Cada árbol del bosque crece profundo y sobreajusta, por eso solo rinde poco. [[a]]Pero como cada uno vio otra muestra y otras variables, sus errores son distintos. [[b]]Al promediar, los errores se cancelan, y la señal común se suma.",
    CL(), P("cada árbol: profundo y sobreajustado", 960, 300, "rojo", 52, ent="subir"),
    T("errores distintos (otra muestra, otras variables)", 960, 460, tam=46, c="tinta", w="sb", en="a"),
    P("promediar: los errores se cancelan · la señal se suma", 960, 640, "verde", 48, en="b"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]un Random Forest promedia cientos de árboles, cada uno con bootstrap y variables al azar. [[b]]Con trescientos árboles llega a cero coma nueve siete cuatro, contra cero coma nueve uno uno del mejor árbol. [[c]]Y pierde lo que hacía especial al árbol: la explicación. La decisión depende de para qué es el modelo.",
    CL(), P("bosque = bootstrap + variables al azar + promedio", 960, 300, "violeta", 48, en="a"),
    P(f"{FI['rf300']:.3f} vs {best['va']:.3f} del mejor árbol", 960, 480, "verde", 52, en="b"), P("pierde la explicación: ¿para qué es el modelo?", 960, 660, "rojo", 48, en="c"),
    sec="11. Para llevar")
v.b("Última parada: [[a]]el repaso final de la clase, con el checklist para entrenar un árbol.",
    CL(), siguiente(11, "Repaso final y checklist"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
