import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util34 import D, out, tarjetas, siguiente, ilustrativo, real, mil, pct, alerta, caja_formula

OUT = out("guion_c34_04_corte.json")
v = V("El primer corte, a mano")
RC, ED, N_ = D["reclamos"], D["edad40"], D["n"]
G0 = D["gini_raiz"]


def g2(f, n):
    p = f / n
    return 1 - p * p - (1 - p) ** 2


# valores del calculo a mano (3 decimales en pantalla; se verifican en verificar_numeros.py)
gA1, gA0 = g2(RC["f_si"], RC["n_si"]), g2(RC["f_no"], RC["n_no"])
wA1, wA0 = RC["n_si"] / N_, RC["n_no"] / N_
gpA = wA1 * gA1 + wA0 * gA0
gB1, gB0 = g2(ED["f_si"], ED["n_si"]), g2(ED["f_no"], ED["n_no"])
wB1, wB0 = ED["n_si"] / N_, ED["n_no"] / N_
gpB = wB1 * gB1 + wB0 * gB0

v.portada("El primer corte, a mano", "La ganancia: cuánto baja la impureza al partir. Reclamos contra edad",
          "Cuarta parada. Hoy haces a mano el cálculo que el árbol hace miles de veces: la ganancia de un corte. Y descubres por qué reclamos gana por paliza.")

# ---- 1. Hook
v.b("Dos candidatos para la primera pregunta del árbol. [[a]]Candidato A: ¿el cliente reclamó? [[b]]Candidato B: ¿tiene más de cuarenta años? [[c]]El árbol elige el que deja a los dos grupos hijos menos mezclados. Hoy lo calculas tú.",
    CL(), T("¿cuál pregunta va primero?", 960, 210, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "A", "¿reclamos > 0?", "a"), (1360, "azul", "B", "¿edad > 40?", "b")], y=520, w=640, h=320, tam_t=90, tam_d=48),
    P("gana el que deja hijos menos mezclados", 960, 820, "violeta", 48, en="c"), sec="1. Dos candidatos")

# ---- 2. La fórmula
v.b("La regla es la ganancia. [[a]]Es la impureza del padre, menos la impureza de los hijos. [[b]]Pero cada hijo se pondera por su tamaño: ene sub jota sobre ene.",
    CL(), caja_formula("ganancia = imp(padre) − Σ (nⱼ / n) · imp(hijoⱼ)", y=300, c="violeta", tam=54, w=1500),
    T("impureza del padre", 560, 470, tam=40, c="gris", w="sb", en="a"), T("menos la de los hijos", 1360, 470, tam=40, c="gris", w="sb", en="a", t=0.3),
    P("cada hijo pesa según su tamaño: nⱼ / n", 960, 620, "teal", 50, en="b"), sec="2. La regla: la ganancia")
v.b("¿Por qué ponderar? [[a]]Porque un hijo puro con dos clientes no vale lo mismo que un hijo puro con dos mil. [[b]]Sin ponderar, el algoritmo aislaría a tres clientes purísimos y creería haber encontrado algo.",
    CL(), Bc(560, 450, 640, 300, "blanco", "gris", 5, 30, sombra=True, en="a", ent="subir"), T("hijo puro de 2", 560, 360, tam=44, c="gris", en="a", t=0.1),
    Pt([(2, [500, 440, 620, 500], "verde")], 24, en="a", t=0.2),
    Bc(1360, 450, 640, 300, "blanco", "verde", 5, 30, sombra=True, en="a", ent="subir", t=0.4), T("hijo puro de 2.000", 1360, 360, tam=44, c="verde", en="a", t=0.5),
    Pt([(120, [1100, 410, 1620, 560], "verde")], 8, en="a", t=0.6),
    P("sin ponderar: aislaría 3 clientes «purísimos»", 960, 760, "rojo", 46, en="b"))

# ---- 3. Ejemplo A: reclamos
v.b("Empecemos por el candidato A, con los datos reales. [[a]]El padre es la raíz: Gini cero coma dos seis cinco. [[b]]Los que reclamaron son doscientos setenta y ocho, y se fugan doscientos diez. [[c]]Los que no reclamaron, dos mil ochocientos setenta y dos, y se fugan doscientos ochenta y cinco.",
    CL(), T("candidato A: ¿reclamos > 0?", 960, 150, tam=48, c="rojo", ent="subir"),
    Bc(960, 290, 560, 110, "blanco", "gris", 4, 24, en="a", ent="pop"), T(f"padre: {mil(N_)} clientes · Gini {G0:.3f}", 960, 290, tam=36, c="tinta", w="sb", en="a", t=0.1),
    L(820, 345, 560, 450, "grisclaro", 6, en="b"), L(1100, 345, 1360, 450, "grisclaro", 6, en="c"),
    Bc(560, 560, 640, 220, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("sí reclamó", 560, 485, tam=36, c="rojo", w="sb", en="b", t=0.1),
    T(f"{RC['n_si']} clientes · {RC['f_si']} se fugan", 560, 560, tam=40, c="tinta", w="sb", en="b", t=0.2),
    Bc(1360, 560, 640, 220, "blanco", "verde", 5, 30, sombra=True, en="c", ent="subir"), T("no reclamó", 1360, 485, tam=36, c="verde", w="sb", en="c", t=0.1),
    T(f"{mil(RC['n_no'])} clientes · {RC['f_no']} se fugan", 1360, 560, tam=40, c="tinta", w="sb", en="c", t=0.2),
    real(1050), sec="3. Ejemplo A: reclamos")
v.b("Paso uno: la proporción de fuga de cada hijo. [[a]]Doscientos diez sobre doscientos setenta y ocho: cero coma siete cinco cinco. [[b]]Doscientos ochenta y cinco sobre dos mil ochocientos setenta y dos: cero coma cero nueve nueve dos. [[c]]Paso dos: el Gini de cada hijo. Cero coma tres siete cero, y cero coma uno siete nueve.",
    T(f"p = {RC['f_si']}/{RC['n_si']} = {RC['f_si'] / RC['n_si']:.3f}", 560, 625, tam=36, c="rojo", w="b", en="a"),
    T(f"p = {RC['f_no']}/{mil(RC['n_no'])} = {RC['f_no'] / RC['n_no']:.4f}", 1360, 625, tam=36, c="verde", w="b", en="b"),
    T(f"Gini = 1 − {RC['f_si'] / RC['n_si']:.3f}² − {1 - RC['f_si'] / RC['n_si']:.3f}² = {gA1:.3f}", 560, 760, tam=32, c="rojo", w="sb", en="c"),
    T(f"Gini = 1 − {RC['f_no'] / RC['n_no']:.4f}² − {1 - RC['f_no'] / RC['n_no']:.4f}² = {gA0:.3f}", 1360, 760, tam=32, c="verde", w="sb", en="c", t=0.6))
v.b("Paso tres: ponderar. [[a]]El hijo de los que reclamaron pesa doscientos setenta y ocho sobre tres mil ciento cincuenta: cero coma cero ocho ocho tres. [[b]]El otro pesa cero coma nueve uno uno siete. [[c]]El Gini ponderado es cero coma uno nueve seis.",
    CL(), T("paso 3: ponderar por tamaño", 960, 170, tam=46, c="tinta", ent="subir"),
    T(f"{RC['n_si']} / {mil(N_)} = {wA1:.4f}", 560, 320, tam=48, c="rojo", w="sb", en="a"),
    T(f"{mil(RC['n_no'])} / {mil(N_)} = {wA0:.4f}", 1360, 320, tam=48, c="verde", w="sb", en="b"),
    T(f"{wA1:.4f} × {gA1:.3f}  +  {wA0:.4f} × {gA0:.3f}", 960, 480, tam=50, c="tinta", w="sb", en="c"),
    T(f"= {round(wA1, 4) * round(gA1, 3):.4f} + {round(wA0, 4) * round(gA0, 3):.4f} =", 900, 590, tam=50, c="gris", w="sb", anc="rm", en="c", t=0.6),
    N(gpA, 930, 590, "{:.3f}", 80, "violeta", anc="lm", en="c", t=1.0), real(1050))
v.b("Paso cuatro: restar al padre. [[a]]Cero coma dos seis cinco menos cero coma uno nueve seis: [[b]]la ganancia de reclamos es cero coma cero seis nueve.",
    CL(), T("paso 4: ganancia = padre − ponderado", 960, 200, tam=46, c="tinta", ent="subir"),
    T(f"{G0:.3f} − {gpA:.3f} =", 900, 420, tam=80, c="tinta", w="sb", anc="rm", en="a"),
    N(G0 - gpA, 930, 420, "{:.3f}", 120, "rojo", anc="lm", en="b", dur=1.0), P("ganancia de reclamos", 960, 600, "rojo", 52, en="b", t=0.6), real(1050))

# ---- 4. Ejemplo B: edad
v.b("Ahora el candidato B, edad mayor que cuarenta, también con los datos reales. [[a]]Arriba de cuarenta hay novecientos setenta clientes, y se fugan ciento cuarenta y siete: quince coma dos por ciento. [[b]]Hasta cuarenta, dos mil ciento ochenta, y se fugan trescientos cuarenta y ocho: dieciséis por ciento. [[c]]Las dos tasas se parecen muchísimo a la del padre, quince coma siete.",
    CL(), T("candidato B: ¿edad > 40?", 960, 150, tam=48, c="azul", ent="subir"),
    Bc(560, 400, 640, 240, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("más de 40", 560, 320, tam=36, c="azul", w="sb", en="a", t=0.1),
    T(f"{ED['n_si']} · {ED['f_si']} se fugan", 560, 395, tam=40, c="tinta", w="sb", en="a", t=0.2), T(pct(ED["f_si"] / ED["n_si"]), 560, 465, tam=48, c="azul", w="b", en="a", t=0.3),
    Bc(1360, 400, 640, 240, "blanco", "azul", 5, 30, sombra=True, en="b", ent="subir"), T("40 o menos", 1360, 320, tam=36, c="azul", w="sb", en="b", t=0.1),
    T(f"{mil(ED['n_no'])} · {ED['f_no']} se fugan", 1360, 395, tam=40, c="tinta", w="sb", en="b", t=0.2), T(pct(ED["f_no"] / ED["n_no"]), 1360, 465, tam=48, c="azul", w="b", en="b", t=0.3),
    P(f"padre: {pct(D['tasa'])}  ·  los hijos se parecen al padre", 960, 640, "ambar", 46, en="c"), real(1050), sec="4. Ejemplo B: edad")
v.b("Hagamos la cuenta rápida. [[a]]Gini de los hijos: cero coma dos cinco siete, y cero coma dos seis ocho. [[b]]Ponderados por su tamaño: cero coma dos seis cinco. [[c]]Exactamente el Gini del padre. La ganancia es cero.",
    CL(), T("candidato B, la cuenta", 960, 170, tam=46, c="azul", ent="subir"),
    T(f"Gini hijos: {gB1:.3f}  y  {gB0:.3f}", 960, 320, tam=52, c="tinta", w="sb", en="a"),
    T(f"{wB1:.4f} × {gB1:.3f} + {wB0:.4f} × {gB0:.3f} = {gpB:.3f}", 960, 450, tam=46, c="gris", w="sb", en="b"),
    T(f"{G0:.3f} − {gpB:.3f} =", 900, 600, tam=72, c="tinta", w="sb", anc="rm", en="c"), N(abs(G0 - gpB), 930, 600, "{:.3f}", 100, "azul", anc="lm", en="c", t=0.5),
    P("hijos iguales al padre → ganancia cero", 960, 780, "azul", 48, en="c", t=1.0), real(1050))
v.b("El veredicto. [[a]]Reclamos baja la impureza en cero coma cero seis nueve. [[b]]Edad, en cero. [[c]]El árbol elige reclamos. Y lo mismo pasa con la entropía: [[d]]la ganancia de reclamos es cero coma uno tres uno, y la de edad, cero. Mismo ganador.",
    CL(), T("ganancia con Gini", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    Bx(560, 600 - RC["gan"] * 4000, 860, 600, "rojo", None, 0, 12, grow="y", en="a"), T("reclamos", 710, 640, tam=36, c="rojo", w="sb", en="a"),
    N(RC["gan"], 710, 600 - RC["gan"] * 4000 - 40, "{:.3f}", 56, "rojo", en="a", t=0.3),
    Bx(1060, 594, 1360, 600, "azul", None, 0, 3, grow="y", en="b"), T("edad > 40", 1210, 640, tam=36, c="azul", w="sb", en="b"), N(0.0, 1210, 560, "{:.3f}", 56, "azul", en="b", t=0.2),
    L(480, 600, 1440, 600, "tinta", 4), K(710, 760, "check", 70, en="c"),
    P(f"entropía: reclamos {RC['gan_e']:.3f} · edad {abs(ED['gan_e']):.3f}", 960, 870, "violeta", 44, en="d"), real(1050),
    sec="5. El veredicto")

# ---- 6. Alerta del material: diapositiva 13
v.b("Ahora, una alerta del material. [[a]]La diapositiva trece propone este mismo ejercicio, y su nota de respuesta dice que el candidato A da un Gini ponderado de cero coma uno seis ocho, y ganancia de cero coma cero nueve siete. [[b]]Eso está mal. [[c]]Lo acabas de calcular: con los datos reales da cero coma uno nueve seis, y cero coma cero seis nueve. Lo mismo que dice la diapositiva doce.",
    CL(), alerta(),
    T("nota de la diapositiva 13:  ponderado ~0.168 · ganancia ~0.097", 960, 330, tam=42, c="tinta", w="sb", en="a", id="n13"),
    U("n13", tachado=1.0, color="gris", en="b", dur=0.6), K(1700, 330, "cruz", 60, en="b", t=0.3),
    P(f"real: ponderado {gpA:.3f} · ganancia {G0 - gpA:.3f}", 960, 480, "verde", 50, en="c"), T("(igual que la diapositiva 12)", 960, 570, tam=34, c="gris", w="sb", en="c", t=0.5),
    sec="6. Alerta: la diapositiva 13")
v.b("Y hay dos detalles más en esa diapositiva. [[a]]Usa doscientos once y doscientos ochenta y cuatro fugas, y los datos reales son doscientos diez y doscientos ochenta y cinco. Con sus números, la ganancia daría cero coma cero siete cero: casi igual. [[b]]Y los conteos del candidato edad, mil quinientos cincuenta y nueve y mil quinientos noventa y uno, no salen de la base. [[c]]Con la base real son novecientos setenta y dos mil ciento ochenta. La conclusión no cambia: edad no separa nada.",
    CL(), alerta(),
    Bc(960, 350, 1500, 150, "blanco", "ambar", 4, 24, sombra=True, en="a", ent="subir"),
    T("diapositiva: 211 y 284 fugas  ·  real: 210 y 285", 960, 325, tam=40, c="tinta", w="sb", en="a", t=0.1),
    T(f"con sus números: ganancia {D['diap13']['gan']:.3f}", 960, 385, tam=34, c="gris", w="sb", en="a", t=0.4),
    Bc(960, 600, 1500, 170, "blanco", "ambar", 4, 24, sombra=True, en="b", ent="subir"),
    T("diapositiva: edad > 40 → 1.559 y 1.591 clientes", 960, 570, tam=40, c="tinta", w="sb", en="b", t=0.1),
    T(f"base real: {ED['n_si']} y {mil(ED['n_no'])}  ·  ganancia ≈ 0 en ambos casos", 960, 635, tam=36, c="verde", w="b", en="c"))

# ---- 7. Error típico
v.b("El error típico: [[a]]olvidar la ponderación y promediar los dos Gini como si pesaran lo mismo. [[b]]Con reclamos daría cero coma tres siete cero más cero coma uno siete nueve, dividido en dos: cero coma dos siete cuatro. [[c]]¡Más que el padre! Una «ganancia» negativa, que no tiene sentido. El grupo chico pesaría igual que el grande.",
    CL(), T("promediar sin ponderar", 960, 220, tam=58, c="tinta", en="a", id="e1"),
    T(f"({gA1:.3f} + {gA0:.3f}) / 2 =", 900, 400, tam=60, c="gris", w="sb", anc="rm", en="b"), N((gA1 + gA0) / 2, 930, 400, "{:.3f}", 84, "rojo", anc="lm", en="b", t=0.5),
    P(f"¡mayor que el padre ({G0:.3f})!", 960, 560, "rojo", 52, en="c"), U("e1", tachado=1.0, color="gris", en="c", t=0.6, dur=0.6),
    P("siempre ponderar por nⱼ / n", 960, 720, "verde", 52, en="c", t=1.2), sec="7. El error típico")

# ---- 8. Piensa un segundo (ejemplo del apunte)
v.b("Piensa un segundo, con el ejemplo del apunte. [[a]]Un padre de cien clientes con veinte fugas: Gini cero coma tres dos. [[b]]El corte parte en cuarenta clientes con dieciséis fugas, y sesenta con cuatro. [[c]]¿Cuál es la ganancia?",
    CL(), P("ejemplo del apunte", 960, 160, "gris", 34, ent="subir"),
    T("padre: 100 clientes · 20 fugas · Gini 0.32", 960, 280, tam=48, c="tinta", w="sb", en="a"),
    tarjetas([(560, "ambar", "izquierda", "40 clientes · 16 fugas", "b"), (1360, "verde", "derecha", "60 clientes · 4 fugas", "b")], y=520, w=640, h=260),
    T("¿ganancia?", 960, 790, tam=72, c="violeta", en="c"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Izquierda: pe igual a cero coma cuatro, Gini cero coma cuatro ocho. Derecha: pe cero coma cero seis seis siete, Gini cero coma uno dos cuatro. [[a]]Ponderado: cero coma cuatro por cero coma cuatro ocho, más cero coma seis por cero coma uno dos cuatro cuatro: cero coma dos seis siete. [[b]]Ganancia: cero coma tres dos menos cero coma dos seis siete, igual a cero coma cero cinco tres.",
    CL(), P("ejemplo del apunte", 960, 160, "gris", 34),
    T(f"izq: p = 0.4 → Gini {g2(16, 40):.2f}     der: p = 0.0667 → Gini {g2(4, 60):.3f}", 960, 300, tam=42, c="tinta", w="sb", ent="subir"),
    T(f"0.4 × {g2(16, 40):.2f} + 0.6 × {g2(4, 60):.4f} = {0.4 * g2(16, 40) + 0.6 * g2(4, 60):.3f}", 960, 440, tam=48, c="gris", w="sb", en="a"),
    T(f"0.32 − {0.4 * g2(16, 40) + 0.6 * g2(4, 60):.3f} =", 900, 600, tam=64, c="tinta", w="sb", anc="rm", en="b"),
    N(0.32 - (0.4 * g2(16, 40) + 0.6 * g2(4, 60)), 930, 600, "{:.3f}", 90, "violeta", anc="lm", en="b", t=0.5))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]ganancia es impureza del padre menos la de los hijos, ponderada por tamaño. [[b]]Reclamos gana con cero coma cero seis nueve; edad no gana nada, porque sus hijos se parecen al padre. [[c]]Y siempre, en cuatro pasos: proporción, Gini, ponderar, restar.",
    CL(), P("ganancia = padre − hijos ponderados", 960, 300, "violeta", 54, en="a"),
    P(f"reclamos {RC['gan']:.3f} · edad ≈ 0", 960, 480, "rojo", 54, en="b"), P("proporción → Gini → ponderar → restar", 960, 660, "teal", 52, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el árbol no prueba dos candidatos, prueba cientos. Veamos el barrido completo, y por qué se le llama voraz.",
    CL(), siguiente(5, "El barrido voraz"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
