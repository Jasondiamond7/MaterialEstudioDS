import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, grupos_series, revelar, GCOL)

OUT = out("guion_c39_11_kprototypes.json")
v = V("K-prototypes para variables mixtas")
KP = D["kproto"]
CO, GP, SE = KP["costo"], KP["grupos"], KP["sens"]
CEN, MAT = KP["ceniza"], KP["matiz"]


def colores_por_matiz(gm):
    """Colores alineados por el centro de matiz (como la diapositiva 50): mayor matiz azul, medio teal, menor ambar."""
    mc = SE[gm]["matiz_centros"]
    orden = sorted(range(3), key=lambda g: -mc[g])
    col = [None] * 3
    for pos, g in enumerate(orden):
        col[g] = ["azul", "teal", "ambar"][pos]
    return col


v.portada("K-prototypes", "Números y categorías juntos, con un peso que los equilibra",
          "Undécima parada: K-prototypes, para cuando la base mezcla variables numéricas y categóricas, que es lo más común.")

# ---- 1. Hook
v.b("En la práctica, casi ninguna base es puramente numérica ni puramente categórica. [[a]]Un banco tiene edad y saldo, que son números; y plan y región, que son categorías. [[b]]Si usas K-means, tienes que inventar números para las categorías. Si usas K-modes, tienes que cortar los números en categorías y perder información. [[c]]K-prototypes evita las dos cosas: usa cada variable en su propio idioma.",
    CL(), T("lo más común: datos mixtos", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "numéricas", "edad, saldo", "a"), (1360, "violeta", "categóricas", "plan, región", "a")], y=380, w=560, h=220, tam_t=50, tam_d=40),
    T("K-means: inventa números · K-modes: corta y pierde información", 960, 600, tam=42, c="rojo", w="sb", en="b"),
    P("K-prototypes: cada variable en su propio idioma", 960, 760, "teal", 48, en="c"), sec="1. Datos mixtos")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Comparas departamentos: metros cuadrados y precio, que son números; y comuna y orientación, que son categorías. [[b]]Para decir qué tan distintos son dos departamentos, sumas la diferencia en números más una penalización por cada categoría distinta. [[c]]La gran pregunta es cuánto vale esa penalización: ¿estar en otra comuna equivale a diez metros cuadrados de diferencia, o a cien? Ese precio es el peso gamma.",
    CL(), T("comparar departamentos", 960, 150, tam=56, c="tinta", ent="subir"),
    T("diferencia = (diferencia numérica) + penalización × (categorías distintas)", 960, 330, tam=42, c="tinta", w="sb", en="b"),
    P("¿otra comuna = 10 m² o 100 m²?", 960, 520, "ambar", 52, en="c"),
    P("ese precio es el peso γ", 960, 680, "violeta", 52, en="c", t=2.5), analogia(1050), sec="2. Analogía: los departamentos")

# ---- 3. La fórmula
v.b("La fórmula de la diapositiva cuarenta y dos. [[a]]Para las variables numéricas, estandarizadas, se suman las diferencias al cuadrado con la media del prototipo, como en K-means. [[b]]Para las categóricas, se cuentan los desacuerdos con la moda del prototipo, como en K-modes, y se multiplican por gamma. [[c]]El prototipo de cada grupo tiene medias en las numéricas y modas en las categóricas. Y gamma, mayor que cero, es el peso de los desacuerdos.",
    CL(), T("K-prototypes para variables mixtas · diapositiva 42", 960, 140, tam=48, c="tinta", ent="subir"),
    caja_formula("d(x, q) = Σ numéricas (x − media)²  +  γ · Σ categóricas 1(x ≠ moda)", y=330, c="violeta", tam=44, w=1700, en="a"),
    T("parte numérica: como K-means", 560, 500, tam=40, c="azul", w="sb", en="a", t=2.0),
    T("parte categórica: como K-modes, por γ", 1360, 500, tam=40, c="violeta", w="sb", en="b"),
    P("prototipo = medias + modas · γ > 0 = peso de los desacuerdos", 960, 680, "teal", 42, en="c"), diapo(42, en="a"), sec="3. La fórmula")
v.b("Un detalle que la diapositiva subraya: [[a]]el equilibrio depende del peso, pero también del número de variables de cada tipo. Con diez numéricas y una categórica, la parte numérica domina aunque gamma sea uno.",
    T("el equilibrio depende de γ y de cuántas variables hay de cada tipo", 960, 860, tam=38, c="ambar", w="sb", en="a"))

# ---- 4. Cuenta resuelta
v.b("La cuenta resuelta de la diapositiva cuarenta y tres. [[a]]Una observación tiene valores numéricos uno y dos, y categorías: la app, y la región sur. [[b]]Prototipo A: números cero y dos, categorías: la app, y la región norte. Parte numérica: uno menos cero al cuadrado, más dos menos dos al cuadrado: uno. Y un desacuerdo, en región. Costo: uno más gamma. [[c]]Prototipo B: números dos y tres, categorías: la app, y la región sur. Parte numérica: uno más uno, dos. Cero desacuerdos. Costo: dos.",
    CL(), T("costo mixto: una cuenta resuelta · diapositiva 43", 960, 130, tam=48, c="tinta", ent="subir"),
    fila(250, [("", 380), ("numérica", 720), ("categorías", 1100), ("costo", 1500)], tams=[34] * 4, colores=["gris"] * 4),
    fila(350, [("observación", 380), ("(1, 2)", 720), ("(App, Sur)", 1100), ("", 1500)], tams=[44] * 4, en="a", fondo="linea", x0=240, x1=1700),
    fila(460, [("prototipo A", 380), ("(0, 2)", 720), ("(App, Norte)", 1100), ("1 + γ", 1500)], tams=[44] * 4, colores=["azul", "tinta", "tinta", "azul"],
         en="b", fondo="azulclaro", x0=240, x1=1700),
    fila(570, [("prototipo B", 380), ("(2, 3)", 720), ("(App, Sur)", 1100), ("2", 1500)], tams=[44] * 4, colores=["ambar", "tinta", "tinta", "ambar"],
         en="c", fondo="ambarclaro", x0=240, x1=1700),
    T("A: (1 − 0)² + (2 − 2)² = 1, y un desacuerdo", 960, 700, tam=38, c="azul", w="sb", en="b", t=3.0),
    T("B: (1 − 2)² + (2 − 3)² = 2, y cero desacuerdos", 960, 780, tam=38, c="ambar", w="sb", en="c", t=2.0),
    diapo(43, en="a"), sec="4. Una cuenta resuelta")
v.b("Y ahora el peso decide. [[a]]Con gamma igual a 0.5, el costo del prototipo A es 1.5, menor que dos: la observación se asigna al prototipo A. [[b]]Con gamma igual a tres, el costo de A sube a cuatro, mayor que dos: se asigna al prototipo B. [[c]]Mismos datos, distinta asignación, solo por cambiar el peso. En un cálculo adicional: con gamma igual a uno hay empate exacto, dos contra dos.",
    CL(), T("γ decide la asignación", 960, 150, tam=56, c="tinta", ent="subir"),
    T("γ = 0.5:   A = 1 + 0.5 = 1.5  <  B = 2   →  A", 960, 330, tam=48, c="azul", w="sb", en="a"),
    T("γ = 3:   A = 1 + 3 = 4  >  B = 2   →  B", 960, 460, tam=48, c="ambar", w="sb", en="b"),
    P("mismos datos, distinta asignación", 960, 620, "violeta", 48, en="c"),
    T("γ = 1:  A = 2 = B  (empate)", 960, 760, tam=40, c="gris", w="sb", en="c", t=2.5), diapo(43, en="a"), adicional(1010, en="c",
    texto="Empate con γ = 1: cálculo adicional"))

# ---- 5. Datos mixtos de los vinos
v.b("Con los vinos, la clase arma una base mixta. [[a]]Dos variables numéricas estandarizadas: ceniza y matiz. [[b]]Y tres categóricas: el alcohol, los flavonoides y el color, con los cortes de K-modes. [[c]]Fíjate en una decisión de diseño: alcohol, flavonoides y color entran solo como categorías, no también como números. Incluir la misma medición dos veces le daría peso duplicado a la misma información.",
    CL(), T("datos mixtos · diapositivas 44 y 45", 960, 140, tam=50, c="tinta", ent="subir"),
    fila(260, [("índice", 560), ("variable", 960), ("tipo", 1360)], tams=[34] * 3, colores=["gris"] * 3),
    [fila(350 + 85 * i, [(str(i), 560), (nm, 960), (tp, 1360)], tams=[40] * 3, colores=["tinta", "tinta", "azul" if i < 2 else "violeta"],
          en=("a" if i < 2 else "b"), t=0.2 * i, fondo=("azulclaro" if i < 2 else "violetaclaro"), x0=420, x1=1500, h=74)
     for i, (nm, tp) in enumerate([("ceniza", "numérica (z)"), ("matiz", "numérica (z)"), ("alcohol", "categórica"), ("flavonoides", "categórica"),
                                   ("color", "categórica")])],
    P("sin duplicar: cada medición entra una sola vez", 960, 860, "ambar", 44, en="c"), diapo(45, en="a"), sec="5. Los vinos mixtos")
v.b("En el código de Python, [[a]]hay que decirle a K-prototypes qué columnas son categóricas: categorical igual a dos, tres y cuatro, porque Python cuenta desde cero. [[b]]El peso se llama gamma en Python, y lambda en la función de R. Y la diapositiva cuarenta y seis aclara: ese lambda es el peso de las categorías, no la penalización de Ridge.",
    CL(), codigo(["KPrototypes(n_clusters=k, init=\"Huang\", n_init=20, gamma=peso,", "            max_iter=100, random_state=2026)",
                  "modelo.fit(mixtos.to_numpy(), categorical=[2, 3, 4])"], y=330, w=1700, tam=34, en=None),
    R(968, 330 + 52.7, 460, 56, "ambar", r=12, en="a"),
    P("Python: gamma · R (kproto): lambda", 960, 580, "violeta", 46, en="b"),
    T("lambda aquí es el peso de las categorías, no la penalización de Ridge", 960, 700, tam=38, c="rojo", w="sb", en="b", t=2.5),
    diapo(47, en=None), grabacion("1:54:58", en="b", y=1010))

# ---- 6. Costo
GK = "gk11"
v.b("El costo de K-prototypes para k de uno a ocho, con peso uno. [[a]]Con un grupo vale 635. Un cálculo adicional lo explica: la parte numérica es dos variables por ciento setenta y seis, 352; y la parte categórica son 283 desacuerdos, el mismo costo de K-modes con un grupo. Trescientos cincuenta y dos más doscientos ochenta y tres, seiscientos treinta y cinco. [[b]]Después baja a 439.61, a 326.64 con tres grupos, y a 285.73 con cuatro. [[c]]La clase usa {k|, ka,} igual a tres como solución de ejemplo. Y advierte: que el costo baje al aumentar k no basta para justificar más grupos.",
    CL(), T("costo de K-prototypes (peso = 1) · diapositiva 48", 960, 130, tam=48, c="tinta", ent="subir"),
    G(520, 240, 1220, 860, 0.5, 8.5, 150, 680, [dict(xs=list(range(1, 9)), ys=CO, color="teal", grosor=6, marcadores=11, revela=0.0)],
      xticks=list(range(1, 9)), yticks=[200, 400, 600], yfmt="{:.0f}", xlabel="número de grupos k", ylabel="costo", id=GK, tam_txt=26),
    U(GK, **{"series.0.revela": 0.02}, en="a", dur=0.3), T("k = 1: 2 × 176 + 283 = 635", 1500, 360, tam=38, c="violeta", w="sb", en="a", t=2.0),
    U(GK, **{"series.0.revela": 1.0}, en="b", dur=2.5), R(3, CO[2], 66, 66, "ambar", r=33, ref=GK, en="c"),
    T("k = 3: 326.64", 1500, 480, tam=44, c="ambar", w="b", en="c"),
    T("un costo menor no justifica más grupos", 1500, 580, tam=34, c="rojo", w="sb", en="c", t=2.5), real(1050, en="b",
    texto="Recalculado con el código de las diapositivas (kmodes 0.12)"), adicional(1010, en="a", texto="Descomposición de 635: cálculo adicional"),
    sec="6. El costo")

# ---- 7. Los prototipos
v.b("Los prototipos con tres grupos, de la diapositiva cuarenta y nueve. [[a]]Grupo uno, sesenta y nueve vinos: ceniza 0.625 y matiz 0.506 desviaciones sobre la media; alcohol alto, flavonoides medios, color medio. [[b]]Grupo dos, cuarenta y siete vinos: ceniza 0.225, matiz menos 1.258; alcohol alto, flavonoides bajos, color alto. [[c]]Grupo tres, sesenta y uno: ceniza menos 0.881, matiz 0.396; alcohol bajo, flavonoides medios, color bajo.",
    CL(), T("un prototipo combina medias y modas · diapositiva 49", 960, 130, tam=48, c="tinta", ent="subir"),
    fila(250, [("grupo", 300), ("n", 480), ("ceniza z", 700), ("matiz z", 940), ("alcohol", 1180), ("flavon.", 1420), ("color", 1640)],
         tams=[32] * 7, colores=["gris"] * 7, x0=200, x1=1760),
    [fila(350 + 100 * i, [(str(i + 1), 300), (str(GP[str(i + 1)]["n"]), 480), (f"{GP[str(i + 1)]['ceniza']:.3f}".replace("-", "−"), 700),
                           (f"{GP[str(i + 1)]['matiz']:.3f}".replace("-", "−"), 940), (GP[str(i + 1)]["alcohol"], 1180),
                           (GP[str(i + 1)]["flavonoides"], 1420), (GP[str(i + 1)]["color"], 1640)],
          tams=[40] * 7, colores=[GCOL[[0, 1, 2][i]]] + ["tinta"] * 6, en="abc"[i], fondo=["azulclaro", "ambarclaro", "tealclaro"][i], x0=200, x1=1760)
     for i in range(3)],
    diapo(49, en="a"), sec="7. Los prototipos")
v.b("Dos lecturas. [[a]]En el grupo dos, matiz está 1.258 desviaciones bajo la media general. Y color alto es la moda, pero solo treinta de cuarenta y siete vinos la tienen: un 63.8 por ciento. [[b]]Y un prototipo puede no corresponder a ninguna observación real: es un resumen, como el hincha típico, que quizás nadie es exactamente.",
    T("grupo 2: color alto en 30 / 47 = 63.8 %", 960, 730, tam=44, c="ambar", w="b", en="a"),
    P("un prototipo puede no corresponder a ningún vino real", 960, 860, "violeta", 42, en="b"))

# ---- 8. Sensibilidad al peso
panel = []
for j, gm in enumerate(["0.25", "1", "4"]):
    x0 = 160 + 590 * j
    col = colores_por_matiz(gm)
    gid = f"gs11{j}"
    panel += [G(x0 + 60, 260, x0 + 520, 760, -4, 3.5, -2.5, 3.6, grupos_series(CEN, MAT, KP["labels"][gm], colores=col, mk=6),
                xticks=[-2, 0, 2], yticks=[-2, 0, 2], yfmt="{:.0f}", xlabel="ceniza (z)", ylabel=("matiz (z)" if j == 0 else ""), id=gid, tam_txt=22,
                en="abc"[j]),
              revelar(gid, 3, en="abc"[j], dur=1.0),
              T(f"peso = {gm}", x0 + 290, 220, tam=38, c="tinta", w="b", en="abc"[j]),
              T(" · ".join(str(t) for t in SE[gm]["tamanos"]), x0 + 290, 880, tam=34, c="gris", w="sb", en="abc"[j], t=0.6)]
v.b("¿Qué pasa si cambiamos el peso, manteniendo tres grupos? La diapositiva cincuenta lo dibuja en el plano de ceniza y matiz. [[a]]Con peso 0.25 mandan las variables numéricas: los grupos se ordenan casi en franjas de matiz, con tamaños sesenta, sesenta y cincuenta y siete. [[b]]Con peso uno, la solución que vimos: sesenta y nueve, cuarenta y siete y sesenta y uno. [[c]]Con peso cuatro mandan las categorías, y en este plano los grupos se ven mezclados: cincuenta y cinco, sesenta y ocho y cincuenta y cuatro.",
    CL(), T("sensibilidad al peso de las categorías · diapositiva 50", 960, 130, tam=46, c="tinta", ent="subir"), panel,
    etiqueta("Recalculado con el código de las diapositivas · colores alineados por el centro de matiz", 1050, c="teal"), sec="8. Sensibilidad al peso")
v.b("Un matiz sobre lo dicho en clase: [[a]]con peso 0.25 se dijo que se logra una buena clasificación. Se ve más ordenado en este dibujo porque el dibujo usa justamente las dos variables numéricas, que son las que mandan con peso bajo. [[b]]Eso no prueba que esos grupos sean mejores: solo que el peso cambia qué variables deciden. [[c]]Y como no hay variable respuesta, el peso no se puede elegir por validación cruzada ni por área bajo la curva. Hay que probar varios y juzgar los perfiles.",
    CL(), alerta(160, texto="matiz sobre la grabación"),
    T("peso bajo → mandan ceniza y matiz → el dibujo de ceniza y matiz se ve ordenado", 960, 320, tam=38, c="tinta", w="sb", en="a"),
    P("ordenado en el dibujo ≠ mejores grupos", 960, 470, "ambar", 46, en="b"),
    T("sin respuesta: no hay validación cruzada ni AUC para elegir γ", 960, 620, tam=40, c="rojo", w="sb", en="c"),
    grabacion("1:58:23 y 1:59:18", en="a"))

# ---- 9. Error típico
v.b("El error típico con K-prototypes: [[a]]no estandarizar la parte numérica. Si la prolina entra en sus unidades, con varianza cercana a cien mil, cualquier gamma razonable queda aplastado, y las categorías no cuentan nada. [[b]]El segundo error: meter la misma medición como número y como categoría. [[c]]Lo correcto: numéricas estandarizadas, cada medición una sola vez, y varios valores de gamma comparados por sus perfiles.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("parte numérica sin estandarizar", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("la misma medición como número y como categoría", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 620, "check", 64, en="c"), T("estandarizar, no duplicar, probar varios γ", 410, 620, tam=44, c="verde", anc="lm", en="c"),
    sec="9. El error típico")

# ---- 10. Piensa
v.b("Piensa un segundo. [[a]]Una observación tiene números cero y cero, y categorías Web y Norte. El prototipo P tiene uno y uno, Web y Norte. El prototipo Q tiene cero y uno, la app, y la región sur. [[b]]Con gamma igual a uno, ¿a cuál va? ¿Y desde qué peso cambiaría de prototipo?",
    CL(), T("observación: (0, 0) · Web, Norte", 960, 260, tam=50, c="tinta", w="b", en="a"),
    T("P: (1, 1) · Web, Norte      Q: (0, 1) · App, Sur", 960, 370, tam=46, c="gris", w="sb", en="a", t=1.5),
    P("con γ = 1, ¿P o Q? ¿desde qué γ cambia?", 960, 560, "violeta", 52, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("P: [[a]]uno más uno, dos, y cero desacuerdos: costo dos. Q: cero más uno, uno, y dos desacuerdos: uno más dos gamma. [[b]]Con gamma igual a uno, Q cuesta tres: va a P. Q gana solo si uno más dos gamma es menor que dos, es decir, con gamma menor que 0.5.",
    CL(), T("P = 1 + 1 + 0 = 2", 700, 320, tam=50, c="azul", w="b", en="a"), T("Q = 0 + 1 + 2γ = 1 + 2γ", 1260, 320, tam=50, c="ambar", w="b", en="a", t=2.0),
    T("γ = 1: Q = 3 > 2 → P", 960, 470, tam=50, c="verde", w="sb", en="b"),
    P("Q gana solo si γ < 0.5", 960, 620, "violeta", 50, en="b", t=2.5), ilustrativo(1050))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el algoritmo K-prototypes suma la parte de K-means, distancias al cuadrado en las numéricas, y la parte de K-modes, desacuerdos en las categóricas, multiplicada por gamma. [[b]]El prototipo combina medias y modas, y puede no parecerse a ninguna observación. [[c]]Y el peso gamma cambia las particiones: no se puede elegir por validación cruzada, hay que explorarlo.",
    CL(), P("costo = cuadrados (numéricas) + γ · desacuerdos (categóricas)", 960, 300, "violeta", 42, en="a"),
    P("prototipo = medias + modas", 960, 460, "teal", 48, en="b"),
    P("γ cambia los grupos: explorarlo", 960, 620, "ambar", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada, la última: [[a]]cómo comparar los tres métodos, cómo evaluar un agrupamiento sin respuesta, y un repaso de toda la clase.",
    CL(), siguiente(12, "Comparar métodos, evaluar y repaso"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
