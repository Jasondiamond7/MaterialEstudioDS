import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, arbol, mil, pct, NODOS, POS, alerta, color_tasa

OUT = out("guion_c34_02_leer.json")
v = V("Leer el árbol: reglas y segmentos")
ND = NODOS
tz = lambda i: ND[i]["fugas"] / ND[i]["n"]  # noqa: E731
H11, H10, H12, H6, H4, H3, H7 = (ND[i] for i in (11, 10, 12, 6, 4, 3, 7))


def anillo(i, c="violeta", en=None, t=None, id=None):
    x, y = POS[i]
    w = 235 if ND[i]["var"] is None else 400
    return R(x, y, w, 140, c, 22, en=en, t=t, id=id)


v.portada("Leer el árbol", "De las cajas a las reglas de negocio: segmentos, tasas y lift",
          "Segunda parada. Hoy aprendes a leer un árbol en voz alta, caja por caja, y a convertir sus hojas en reglas que el negocio entiende.")

# ---- 1. Hook
v.b("Este es el árbol de la clase, entrenado con los tres mil ciento cincuenta clientes. [[a]]Profundidad tres: como máximo tres preguntas encadenadas. [[b]]Y ninguna hoja con menos de sesenta clientes. [[c]]Cabe en una lámina, y eso es su gran virtud.",
    CL(), arbol(pre="ar1", en={0: None, 1: None, 2: None, 3: None}),
    P("max_depth = 3", 300, 140, "violeta", 38, en="a"), P("min_samples_leaf = 60", 960, 140, "violeta", 38, en="b"),
    P("cabe en una lámina", 1600, 140, "verde", 38, en="c"), real(1050), sec="1. Un árbol que cabe en una lámina")
v.b("Fíjate en algo: había doce variables disponibles. [[a]]El árbol usó solo seis: reclamos, segundos, antigüedad, ce ele ve, frecuencia de llamadas y número de fallas. [[b]]Las otras seis no aparecen en ningún corte. El árbol las descartó solo: eso es la selección embedded de la clase treinta y tres.",
    *[anillo(i, "azul", en="a", t=0.25 * k) for k, i in enumerate((0, 1, 2, 5, 8, 9))],
    Bc(1640, 870, 440, 170, "blanco", "gris", 4, 22, sombra=True, en="b", ent="pop"),
    T("no usadas:", 1640, 815, tam=28, c="gris", w="sb", en="b", t=0.1),
    T("estado · monto · frec_sms", 1640, 860, tam=26, c="tinta", w="sb", en="b", t=0.2),
    T("n_contactos · tipo_plan · edad", 1640, 900, tam=26, c="tinta", w="sb", en="b", t=0.3))

# ---- 2. Cómo se lee una caja
v.b("¿Cómo se lee una caja? [[a]]Arriba va la pregunta. [[b]]Abajo, cuántos clientes llegaron ahí y qué proporción se fugó. [[c]]Y una convención de sklearn: el «sí» siempre va a la izquierda. Si se cumple la condición, bajas por la izquierda.",
    CL(), Bc(960, 420, 760, 240, "blanco", "gris", 6, 30, sombra=True, ent="pop"),
    T("reclamos ≤ 0.5", 960, 370, tam=64, c="tinta", en="a"), T(f"n {mil(D['n'])} · fuga {pct(D['tasa'])}", 960, 470, tam=48, c="gris", w="sb", en="b"),
    L(760, 545, 520, 720, "grisclaro", 7, en="c"), L(1160, 545, 1400, 720, "grisclaro", 7, en="c"),
    P("sí → izquierda", 520, 790, "verde", 44, en="c", t=0.3), P("no → derecha", 1400, 790, "rojo", 44, en="c", t=0.5), sec="2. Cómo se lee una caja")
v.b("Ojo con la raíz. [[a]]Reclamos vale cero o uno, así que «reclamos menor o igual a cero coma cinco» significa: no reclamó. [[b]]La izquierda son los que no reclamaron, y la derecha, los que sí.",
    CL(), T("reclamos solo vale 0 o 1", 960, 260, tam=52, c="tinta", ent="subir"),
    P("reclamos ≤ 0.5  =  no reclamó", 960, 440, "violeta", 60, en="a"),
    tarjetas([(560, "verde", "izquierda", f"no reclamó · {mil(ND[1]['n'])} clientes", "b"), (1360, "rojo", "derecha", f"sí reclamó · {mil(ND[8]['n'])} clientes", "b")], y=720, w=640, h=260))

# ---- 3. Rama izquierda
v.b("Recorramos la rama izquierda. [[a]]No reclamaron dos mil ochocientos setenta y dos clientes: el noventa y uno por ciento de la base. [[b]]Y de ellos se fuga solo el nueve coma nueve por ciento.",
    CL(), arbol(pre="ar2"), anillo(1, "verde", en="a"),
    Bc(1500, 160, 700, 90, "blanco", "verde", 4, 24, en="b", ent="subir"),
    T(f"{mil(ND[1]['n'])} = {100 * ND[1]['n'] / D['n']:.0f} % de la base · fuga {pct(tz(1))}", 1500, 160, tam=34, c="verde", en="b", t=0.1),
    real(1050), sec="3. La rama de los que no reclamaron")
v.b("Siguiente pregunta: [[a]]¿usa novecientos treinta y cuatro segundos o menos? [[b]]Los que usan poco son cuatrocientos cuarenta y cuatro, con treinta y cinco por ciento de fuga. [[c]]Los que usan más, dos mil cuatrocientos veintiocho, con apenas cinco coma tres.",
    anillo(2, "ambar", en="b"), anillo(5, "verde", en="c"), R(510, 450, 400, 140, "violeta", 22, en="a"))
v.b("Y el tercer nivel. [[a]]Entre los que usan poco, los de antigüedad mayor a veintiocho y medio se fugan un cuarenta y tres por ciento: [[b]]uso bajo pese a la antigüedad. [[c]]Y a la derecha está el núcleo sano de la cartera: dos mil trescientos cincuenta y nueve clientes con cuatro coma cinco por ciento.",
    anillo(4, "ambar", en="a"), P("uso bajo pese a la antigüedad", 397, 990, "ambar", 32, en="b"),
    anillo(7, "verde", en="c"), P("núcleo sano de la cartera", 900, 990, "verde", 32, en="c", t=0.3))

# ---- 4. Rama derecha
v.b("Ahora la rama derecha: los que reclamaron. [[a]]Son doscientos setenta y ocho, y se fuga el setenta y cinco coma cinco por ciento. [[b]]Debajo, ¿hace cincuenta y siete llamadas al mes o menos? [[c]]Los que reclamaron y usan poco: doscientos dieciséis, con ochenta y cinco coma seis por ciento.",
    CL(), arbol(pre="ar3"), anillo(8, "rojo", en="a"), R(1410, 450, 400, 140, "violeta", 22, en="b"), anillo(9, "rojo", en="c"),
    real(1050), sec="4. La rama de los que reclamaron")
v.b("Y la última pregunta: [[a]]¿tuvo fallas técnicas? [[b]]Si reclamó, usa poco y tuvo fallas: ciento cuarenta y nueve clientes, y se fuga el noventa y cuatro por ciento. [[c]]El profesor lo dijo así: esto es una locura.",
    R(1185, 640, 400, 140, "violeta", 22, en="a"), anillo(11, "rojo", en="b"),
    Bc(1640, 870, 440, 200, "blanco", "rojo", 5, 26, sombra=True, en="b", ent="pop", t=0.4),
    T(f"{H11['n']} clientes", 1640, 830, tam=36, c="tinta", w="sb", en="b", t=0.5), N(100 * tz(11), 1640, 900, "{:.0f}", 70, "rojo", suf=" %", en="b", t=0.6),
    P("«esto es una locura»", 1640, 990, "ambar", 34, en="c"))

# ---- 5. De hojas a reglas
HOJAS = [(11, "reclamó · llamadas ≤ 57.5 · con fallas", "reclamó, usa poco y tuvo fallas"),
         (10, "reclamó · llamadas ≤ 57.5 · sin fallas", "reclamó y usa poco"),
         (4, "no reclamó · seg ≤ 934 · antig > 28.5", "uso bajo pese a la antigüedad"),
         (12, "reclamó · llamadas > 57.5", "reclamó pero sigue usando"),
         (6, "no reclamó · seg > 934 · clv ≤ 61.35", "usa, pero vale poco"),
         (3, "no reclamó · seg ≤ 934 · antig ≤ 28.5", "nuevo y de poco uso"),
         (7, "no reclamó · seg > 934 · clv > 61.35", "núcleo sano de la cartera")]
v.b("Ahora traduce cada hoja a una regla de negocio. [[a]]Ordenadas de mayor a menor fuga, [[b]]las siete hojas cubren a los tres mil ciento cincuenta clientes, sin repetir a ninguno.",
    CL(), T("las 7 hojas como reglas", 960, 140, tam=46, c="tinta", ent="subir"),
    *[T(h, x, 215, tam=26, c="gris", w="sb", anc=a) for h, x, a in (("regla", 120, "lm"), ("clientes", 1080, "mm"), ("fuga", 1260, "mm"), ("lectura", 1400, "lm"))],
    *[[Bc(960, 290 + 96 * k, 1760, 80, "blanco", color_tasa(tz(i)), 3, 16, en="a", t=0.35 * k, ent="subir"),
       T(r, 120, 290 + 96 * k, tam=29, c="tinta", w="sb", anc="lm", en="a", t=0.35 * k + 0.05),
       T(mil(ND[i]["n"]), 1080, 290 + 96 * k, tam=34, c="tinta", w="b", en="a", t=0.35 * k + 0.1),
       T(pct(tz(i)), 1260, 290 + 96 * k, tam=34, c=color_tasa(tz(i)), w="b", en="a", t=0.35 * k + 0.15),
       T(lec, 1400, 290 + 96 * k, tam=26, c="gris", w="sb", anc="lm", en="a", t=0.35 * k + 0.2)] for k, (i, r, lec) in enumerate(HOJAS)],
    P(f"suma: {' + '.join(str(ND[i]['n']) for i, _r, _l in HOJAS)} = {mil(D['hojas3_suma_n'])}", 960, 990, "teal", 30, en="b"),
    sec="5. De hojas a reglas")
v.b("Esto es lo que da un árbol y una logística no. [[a]]«Cliente que reclamó, usa poco y tuvo fallas» no es un coeficiente. [[b]]Es un segmento de ciento cuarenta y nueve personas, al que se le puede mandar una campaña mañana.",
    CL(), T("la logística da", 560, 300, tam=40, c="gris", w="sb", ent="subir"), T("f(β₀ + β₁·reclamos + β₂·segundos + …)", 560, 400, tam=40, c="azul", w="sb", en="a"),
    T("coeficientes, iguales en todo el espacio", 560, 490, tam=34, c="azul", w="sb", en="a", t=0.2),
    T("el árbol da", 1360, 300, tam=40, c="gris", w="sb", ent="subir"), P("reclamó · usa poco · tuvo fallas", 1360, 400, "rojo", 40, en="b"),
    T(f"{H11['n']} personas · {pct(tz(11), 0)} de fuga", 1360, 490, tam=40, c="rojo", w="b", en="b", t=0.3),
    P("una campaña, mañana", 1360, 640, "verde", 46, en="b", t=0.8))

# ---- 6. Ejemplo resuelto: el lift y la campaña
v.b("Midamos qué tan bueno es ese segmento. [[a]]El lift es la tasa de la hoja dividida por la tasa general. [[b]]Cero coma nueve cuatro cero, dividido en cero coma uno cinco siete: [[c]]casi seis. Ese grupo se fuga seis veces más que el cliente promedio.",
    CL(), T("lift = tasa de la hoja / tasa general", 960, 230, tam=54, c="violeta", en="a"),
    T(f"{tz(11):.3f} / {D['tasa']:.3f} =", 900, 450, tam=72, c="tinta", en="b", anc="rm"),
    N(tz(11) / D["tasa"], 940, 450, "{:.2f}", 100, "rojo", en="c", anc="lm", dur=1.0),
    P("6 veces el riesgo promedio", 960, 640, "rojo", 52, en="c", t=0.8), real(1010), sec="6. Ejemplo: el lift y la campaña")
v.b("Y ahora el cálculo de campaña. [[a]]Esa hoja tiene ciento cuarenta y nueve clientes: el cuatro coma siete por ciento de la base. [[b]]Adentro hay ciento cuarenta fugados, de los cuatrocientos noventa y cinco que hay en total: [[c]]el veintiocho por ciento de todas las fugas. [[d]]Llamando a menos del cinco por ciento de la cartera, llegas a más de un cuarto de los que se van.",
    CL(), T("si llamas solo a esa hoja…", 960, 200, tam=48, c="tinta", ent="subir"),
    Bc(560, 470, 640, 280, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("contactas", 560, 380, tam=36, c="gris", w="sb", en="a", t=0.1),
    T(f"{H11['n']} / {mil(D['n'])}", 560, 460, tam=48, c="tinta", w="sb", en="a", t=0.2), N(100 * H11["n"] / D["n"], 560, 545, "{:.1f}", 64, "violeta", suf=" %", en="a", t=0.4),
    Bc(1360, 470, 640, 280, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("capturas fugas", 1360, 380, tam=36, c="gris", w="sb", en="b", t=0.1),
    T(f"{H11['fugas']} / {D['n_fuga']}", 1360, 460, tam=48, c="tinta", w="sb", en="b", t=0.2), N(100 * H11["fugas"] / D["n_fuga"], 1360, 545, "{:.1f}", 64, "rojo", suf=" %", en="c"),
    P("menos del 5 % de la cartera → más de un cuarto de las fugas", 960, 770, "teal", 42, en="d"))
v.b("El profesor también mostró una hoja más chica. [[a]]No reclamó, usa más de novecientos treinta y cuatro segundos, pero vale poco: ce ele ve menor a sesenta y uno. [[b]]Sesenta y nueve clientes, el dos coma dos por ciento de la base, con casi treinta y cinco por ciento de fuga. [[c]]Sirve para una campaña de bajo presupuesto, más precisa que tirar una moneda.",
    CL(), arbol(pre="ar4"), anillo(6, "ambar", en="a"),
    Bc(1640, 870, 440, 200, "blanco", "ambar", 5, 26, sombra=True, en="b", ent="pop"),
    T(f"{H6['n']} clientes · {100 * H6['n'] / D['n']:.1f} %", 1640, 830, tam=32, c="tinta", w="sb", en="b", t=0.1),
    N(100 * tz(6), 1640, 900, "{:.1f}", 64, "ambar", suf=" %", en="b", t=0.3), P("campaña de bajo presupuesto", 1640, 990, "ambar", 30, en="c"))
v.b("Y aquí entra el ce ele ve, el valor del cliente en el tiempo. [[a]]Antes de gastar en retener, pregunta si ese cliente vale la pena. [[b]]A veces conviene un regalo para que se quede contento. [[c]]Y a veces es un cliente que ni siquiera conviene tener. Son decisiones de negocio, no del árbol.",
    CL(), P("CLV · valor del cliente en el tiempo", 960, 240, "violeta", 52, ent="subir"),
    tarjetas([(560, "verde", "vale la pena", "un regalo para retenerlo", "b"), (1360, "gris", "no vale la pena", "dejarlo ir también es decisión", "c")], y=600, w=640, h=320),
    T("¿cuánto vale este cliente?", 960, 370, tam=44, c="tinta", w="sb", en="a"))

# ---- 7. Alerta del material
v.b("Ahora, tres alertas del material. [[a]]Primera: la diapositiva siete, y la explicación en clase, resumen la regla del noventa y cuatro por ciento con dos preguntas: reclamó, y llamadas menor a cincuenta y siete coma cinco. [[b]]Con solo esas dos, el grupo es de doscientos dieciséis clientes, y la fuga es ochenta y cinco coma seis. [[c]]El noventa y cuatro exige la tercera pregunta: que haya tenido fallas.",
    CL(), alerta(),
    T("diapositiva 7:  ¿reclamos > 0? → ¿frec_llamadas ≤ 57.5? → 94 %", 960, 330, tam=40, c="tinta", w="sb", en="a"),
    Bc(560, 560, 640, 240, "blanco", "ambar", 5, 28, sombra=True, en="b", ent="subir"), T("con 2 preguntas", 560, 490, tam=34, c="gris", w="sb", en="b", t=0.1),
    T(f"{ND[9]['n']} clientes · {pct(tz(9))}", 560, 580, tam=48, c="ambar", w="b", en="b", t=0.2),
    Bc(1360, 560, 640, 240, "blanco", "rojo", 5, 28, sombra=True, en="c", ent="subir"), T("+ tuvo fallas", 1360, 490, tam=34, c="gris", w="sb", en="c", t=0.1),
    T(f"{H11['n']} clientes · {pct(tz(11))}", 1360, 580, tam=48, c="rojo", w="b", en="c", t=0.2), sec="7. Alertas del material")
v.b("[[a]]Segunda: al leer en voz alta la hoja de sesenta y nueve clientes, se dijo «segundos menor a novecientos treinta y cuatro». [[b]]Pero esa hoja está en la rama contraria: segundos mayor a novecientos treinta y cuatro, y ce ele ve menor a sesenta y uno coma tres. [[c]]Tercera: la tabla de la diapositiva nueve muestra solo cinco de las siete hojas. Faltan la de sesenta y nueve, y la de ciento cuatro.",
    CL(), alerta(),
    T("en clase: «segundos menor a 934»", 960, 330, tam=46, c="tinta", w="sb", en="a", id="a2"), U("a2", tachado=1.0, color="gris", en="b", dur=0.6),
    P("correcto: segundos > 934 · clv ≤ 61.3", 960, 450, "verde", 46, en="b", t=0.4),
    Bc(960, 680, 1300, 200, "blanco", "ambar", 4, 24, sombra=True, en="c", ent="subir"),
    T("diapositiva 9: 5 de 7 hojas", 960, 640, tam=42, c="ambar", en="c", t=0.1),
    T(f"faltan: {H6['n']} clientes ({pct(tz(6))})  y  {H3['n']} clientes ({pct(tz(3))})", 960, 715, tam=36, c="tinta", w="sb", en="c", t=0.3))

# ---- 8. Error típico
v.b("El error típico al leer hojas: [[a]]confiar en una tasa calculada con muy pocos clientes. [[b]]Con una hoja de cinco clientes, cada cliente mueve la tasa veinte puntos: cinco de cinco es cien por ciento, cuatro de cinco es ochenta. [[c]]Con sesenta, cada cliente mueve menos de dos puntos. Por eso la clase fija sesenta clientes como mínimo por hoja.",
    CL(), T("hojas chicas = tasas de ruido", 960, 200, tam=56, c="tinta", ent="subir"),
    Bc(560, 520, 640, 340, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("hoja de 5", 560, 400, tam=44, c="rojo", en="a", t=0.1),
    Pt([(5, [420, 450, 700, 500], "rojo")], 20, en="b"), T("5 de 5 = 100 %", 560, 560, tam=40, c="tinta", w="sb", en="b", t=0.3),
    T("1 cliente = 20 puntos", 560, 630, tam=36, c="rojo", w="b", en="b", t=0.6),
    Bc(1360, 520, 640, 340, "blanco", "verde", 5, 30, sombra=True, en="c", ent="subir"), T("hoja de 60", 1360, 400, tam=44, c="verde", en="c", t=0.1),
    T(f"1 cliente = {100 / 60:.1f} puntos", 1360, 560, tam=40, c="verde", w="b", en="c", t=0.3),
    P("min_samples_leaf = 60", 960, 830, "violeta", 48, en="c", t=0.8), ilustrativo(1000, texto="ejemplo ilustrativo (hoja de 5)"), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un cliente que no reclamó, usa mil doscientos segundos, y tiene un ce ele ve de cincuenta. [[b]]¿En qué hoja cae, y cuál es su probabilidad de fuga?",
    CL(), P("no reclamó · 1.200 segundos · clv = 50", 960, 320, "violeta", 52, en="a"),
    T("¿en qué hoja cae?", 960, 520, tam=66, c="tinta", en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("No reclamó: izquierda. [[a]]Mil doscientos es más que novecientos treinta y cuatro: derecha. [[b]]Cincuenta es menor que sesenta y uno coma tres: izquierda. [[c]]Cae en la hoja de sesenta y nueve clientes, con treinta y cuatro coma ocho por ciento de fuga.",
    CL(), arbol(pre="ar5"), R(960, 270, 400, 140, "violeta", 22), R(510, 450, 400, 140, "violeta", 22, en="a"),
    R(735, 640, 400, 140, "violeta", 22, en="b"), anillo(6, "ambar", en="c"),
    P(f"hoja de {H6['n']} · {pct(tz(6))}", 1640, 900, "ambar", 44, en="c", t=0.3))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]«sí» va a la izquierda, y cada caja dice regla, clientes y tasa. [[b]]Cada hoja es un segmento: regla, tamaño, tasa y lift. [[c]]Y las hojas chicas mienten: por eso hay un mínimo de clientes por hoja.",
    CL(), P("sí → izquierda · regla, n y tasa", 960, 300, "violeta", 54, en="a"),
    P("hoja = segmento: regla, n, tasa, lift", 960, 480, "teal", 54, en="b"), P("hojas chicas mienten", 960, 660, "rojo", 54, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]¿por qué el árbol eligió reclamos como primera pregunta? Para responder, hay que medir el desorden: Gini y entropía.",
    CL(), siguiente(3, "Impureza: Gini y entropía"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
