import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, caja_formula, alerta

OUT = out("guion_c33_06_filtros.json")
v = V("Taxonomía y métodos de filtro")
RK, XR = D["rank12"], D["xor"]

v.portada("Filter, wrapper y embedded", "Tres formas de decidir, y los filtros: rápidos, escalables y con un punto ciego",
          "Quinta parada: las tres familias de métodos de selección de variables, y la primera de ellas, los filtros.")

# ---- 1. Hook: embudo de contratación
v.b("Imagina que una empresa recibe cinco mil postulaciones. [[a]]Primero filtra por currículum: rápido y barato, sin entrevistar a nadie. [[b]]Después entrevista a doscientos. [[c]]Y solo a los últimos treinta los pone a prueba en el trabajo real.",
    CL(), P("5.000 postulaciones", 960, 210, "violeta", 54, ent="subir"),
    Bx(360, 320, 1560, 440, "violetaclaro", "violeta", 3, 20, grow="x", en="a"), T("currículum · 5.000 → 200", 960, 380, tam=44, c="violeta", en="a", t=0.4),
    Bx(560, 480, 1360, 600, "azulclaro", "azul", 3, 20, grow="x", en="b"), T("entrevista · 200 → 30", 960, 540, tam=44, c="azul", en="b", t=0.4),
    Bx(760, 640, 1160, 760, "verdeclaro", "verde", 3, 20, grow="x", en="c"), T("prueba real · 30", 960, 700, tam=44, c="verde", en="c", t=0.4),
    ilustrativo(1000, texto="analogía"), sec="1. Un embudo de contratación")
v.b("La selección de variables funciona igual: es un embudo. [[a]]Filtro para bajar de cinco mil a doscientas, embedded para bajar a treinta, [[b]]y wrapper solo si sobra cómputo. No compiten: son etapas.",
    CL(), T("la regla práctica de la clase", 960, 200, tam=44, c="gris", w="sb", ent="subir"),
    *[[Bc(330 + 420 * i, 470, 340, 220, "blanco", c, 5, 28, sombra=True, en=k, ent="subir", t=t_), T(n, 330 + 420 * i, 430, tam=70, c=c, en=k, t=t_ + 0.1),
       T(lab, 330 + 420 * i, 520, tam=32, c=c, w="sb", en=k, t=t_ + 0.2)]
      for i, (n, lab, c, k, t_) in enumerate((("5.000", "candidatas", "gris", "a", 0.0), ("200", "tras el filtro", "violeta", "a", 0.6),
                                              ("30", "tras embedded", "azul", "a", 1.4), ("12", "tras wrapper", "verde", "b", 0.0)))],
    *[L(500 + 420 * i, 470, 580 + 420 * i, 470, "gris", 6, True, en=("a" if i < 2 else "b"), t=0.4 + 0.8 * i if i < 2 else 0) for i in range(3)],
    P("no compiten: son etapas del mismo embudo", 960, 760, "ambar", 48, en="b", t=0.6))

# ---- 2. Las tres familias
FAMS = [("Filter", "mide cada variable con una métrica, sin entrenar el modelo", "costo bajo · no ve el conjunto", "varianza · correlación · MI · ANOVA F", "violeta"),
        ("Wrapper", "entrena el modelo con distintos subconjuntos y compara", "costo alto · ve el conjunto", "RFE · forward · backward", "azul"),
        ("Embedded", "la selección ocurre dentro del entrenamiento", "costo medio · ve el conjunto", "Lasso · importancia de árboles", "verde")]
v.b("Las tres familias. [[a]]Filter mide cada variable con una métrica estadística, sin entrenar el modelo. Es barato, pero si es univariado, no ve el conjunto. [[b]]Wrapper entrena el modelo con distintos subconjuntos y compara. Ve el conjunto, pero es caro. [[c]]Y embedded selecciona mientras entrena, como el Lasso. Costo medio.",
    CL(), *[[Bc(360 + 600 * i, 520, 560, 560, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(nm, 360 + 600 * i, 300, tam=60, c=c, en=k, t=0.1),
             T(d1, 360 + 600 * i, 430, tam=34, c="tinta", w="sb", mw=480, en=k, t=0.2), P(d2, 360 + 600 * i, 580, c, 30, en=k, t=0.4),
             T(d3, 360 + 600 * i, 700, tam=30, c="gris", w="sb", mw=480, en=k, t=0.5)] for i, ((nm, d1, d2, d3, c), k) in enumerate(zip(FAMS, "abc"))],
    sec="2. Las tres familias")
v.b("¿Cuánto cuesta cada una, en modelos entrenados? [[a]]Un filtro, cero. Un embedded, uno. [[b]]Un RFE que elimina de a una, p menos k: con cincuenta variables, hasta doce, treinta y ocho por fold. [[c]]Y probar todos los subconjuntos, dos elevado a p, menos uno. Eso lo calculamos en el video ocho.",
    P("filter: 0 modelos", 560, 860, "violeta", 36, en="a"), P("embedded: 1", 960, 860, "verde", 36, en="a", t=0.3),
    P("RFE: p − k = 50 − 12 = 38", 1400, 860, "azul", 36, en="b"), P("búsqueda exhaustiva: 2^p − 1 → video 8", 960, 960, "rojo", 38, en="c"))
v.b("Un matiz sobre la tabla de la clase. [[a]]Dice que wrapper y embedded «sí ven interacciones». Más preciso: ven el conjunto, como número de fallas, que solo aporta acompañada. [[b]]Pero si el modelo de adentro es una logística, tampoco ven un XOR sin el producto: [[c]]en el video anterior, la logística con x1 y x2 dio cero coma cinco uno.",
    CL(), P("un matiz", 960, 200, "ambar", 48, ent="subir"),
    T("«wrapper y embedded: ¿ven interacciones? sí»", 960, 340, tam=50, c="gris", w="sb", en="a"),
    P("ven el conjunto: numero_fallas acompañada", 960, 480, "verde", 46, en="a", t=1.2),
    P("con una logística adentro, un XOR sin producto no se ve", 960, 640, "rojo", 44, en="b"),
    N(XR["log"], 960, 790, "AUC {:.3f}", 70, "rojo", en="c"))

# ---- 3. Filtro 1: varianza
v.b("Ahora, los filtros, uno por uno. El primero: la varianza. [[a]]Una variable casi constante no puede explicar nada, porque vale lo mismo para casi todos los clientes. [[b]]Para una binaria, con una proporción pi de unos, la varianza es pi por uno menos pi.",
    CL(), T("filtro 1 · varianza casi nula", 960, 190, tam=48, c="violeta", ent="subir"),
    Pt([(97, [500, 300, 1420, 560], "#94A3B8"), (3, [900, 590, 1020, 620], "rojo")], 12, visibles=100, en="a", dur=1.2),
    T("97 % ceros, 3 % unos: casi constante", 960, 680, tam=38, c="gris", w="sb", en="a", t=1.0),
    caja_formula("Var(binaria) = π (1 − π)", y=840, c="violeta", tam=56, w=1000, en="b"), ilustrativo(1000), sec="3. Filtro 1: la varianza")
VAR = [("0.5 %", "0.005 · 0.995 = 0.005", "no: casi constante", "rojo"), ("1 %", "0.01 · 0.99 = 0.0099", "no, por muy poco", "ambar"), ("10 %", "0.1 · 0.9 = 0.09", "sí", "verde")]
v.b("Con un umbral de cero coma cero uno, hagamos tres casos del apunte. [[a]]Medio por ciento de unos: cero coma cero cero cinco. No pasa. [[b]]Uno por ciento: cero coma cero cero noventa y nueve. No pasa, por muy poco. [[c]]Diez por ciento: cero coma cero nueve. Pasa.",
    CL(), T("VarianceThreshold(threshold=0.01) · ejemplo del apunte", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    T("% de unos (π)", 400, 300, tam=32, c="gris", w="sb"), T("π (1 − π)", 960, 300, tam=32, c="gris", w="sb"), T("¿pasa 0.01?", 1500, 300, tam=32, c="gris", w="sb"),
    *[[Bc(960, 410 + 140 * i, 1500, 110, "blanco", c, 4, 22, en=k, ent="subir"), T(a, 400, 410 + 140 * i, tam=48, c=c, en=k, t=0.1),
       T(b, 960, 410 + 140 * i, tam=42, c="tinta", w="m", en=k, t=0.2), T(r, 1500, 410 + 140 * i, tam=42, c=c, w="b", en=k, t=0.5)]
      for i, ((a, b, r, c), k) in enumerate(zip(VAR, "abc"))],
    T("en palabras: se descartan binarias con ~1 % o menos en su categoría chica", 960, 860, tam=36, c="tinta", w="sb", en="c", t=1.0))
v.b("¿Y en la base de la clase? [[a]]El filtro de varianza, sobre las cincuenta candidatas, no descarta ninguna. [[b]]Es un filtro de higiene: barato, y casi nunca decide nada importante.",
    CL(), Bc(960, 430, 700, 320, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), N(len(D["vt_fuera"]), 960, 400, "{:.0f}", 140, "violeta", en="a", t=0.1),
    T("de 50 descartadas", 960, 530, tam=40, c="violeta", w="sb", en="a", t=0.3), P("un filtro de higiene", 960, 720, "gris", 50, en="b"))

# ---- 4. Filtro 2: correlación vs información mutua
UX = [-2, -1, 0, 1, 2]
v.b("El segundo filtro: la correlación de Pearson. Su punto ciego: [[a]]solo mide relación lineal. [[b]]Mira esta U perfecta: y igual a x al cuadrado. Los puntos menos dos, menos uno, cero, uno y dos dan cuatro, uno, cero, uno, cuatro. [[c]]La dependencia es total, y la correlación da exactamente cero.",
    CL(), T("filtro 2 · correlación de Pearson", 960, 190, tam=48, c="violeta", ent="subir"),
    G(300, 280, 1000, 860, -2.5, 2.5, -0.5, 4.8, [dict(xs=UX, ys=[x * x for x in UX], color="rojo", grosor=5, marcadores=12, suave=True, revela=0.0)],
      xticks=UX, yticks=[0, 1, 2, 3, 4], yfmt="{:.0f}", id="gu", en="a"),
    U("gu", **{"series.0.revela": 1.0}, en="b", dur=1.8), T("y = x²", 1000, 300, tam=44, c="rojo", w="m", en="b"),
    N(0.0, 1380, 520, "r = {:.2f}", 90, "rojo", en="c"), P("dependencia total · r = 0", 1380, 680, "ambar", 44, en="c", t=0.5),
    ilustrativo(1000), sec="4. Filtro 2: correlación e información mutua")
v.b("Por eso existe la información mutua. [[a]]Mide cuánta incertidumbre sobre la fuga desaparece al conocer la variable. [[b]]Es mayor o igual que cero, y vale cero solo si son independientes: [[c]]detecta cualquier dependencia, también la U.",
    CL(), T("información mutua", 960, 190, tam=50, c="violeta", ent="subir"),
    caja_formula("I(X; Y) = H(Y) − H(Y | X)", y=360, c="violeta", tam=60, w=1100, en="a"),
    P("I ≥ 0  ·  I = 0 solo si son independientes", 960, 560, "teal", 48, en="b"), P("ve relaciones no lineales, como la U", 960, 710, "verde", 48, en="c"))
v.b("Dos cuidados prácticos. [[a]]No tiene techo: sirve para ordenar variables dentro de un dataset, no para comparar entre datasets. [[b]]Y sklearn la estima con vecinos más cercanos, con algo de azar. Sin fijar random state, cambia en cada corrida.",
    P("sin techo → solo para ordenar dentro de un dataset", 960, 840, "ambar", 40, en="a"),
    T("mutual_info_classif(X, y, random_state=0)", 960, 950, tam=40, c="tinta", w="m", en="b"))

# ---- 5. Filtro 3: ANOVA F, a mano
GA, GB = [2, 4, 6], [8, 10, 12]
v.b("El tercer filtro: ANOVA F. [[a]]Compara cuánto se separan las medias de los grupos, contra cuánto varían los datos dentro de cada grupo. [[b]]Aquí los grupos son dos: los que se fugan, y los que no.",
    CL(), T("filtro 3 · ANOVA F", 960, 190, tam=48, c="violeta", ent="subir"),
    caja_formula("F = CM entre / CM dentro", y=360, c="violeta", tam=60, w=1000, en="a"),
    P("grupo 1 = se fuga · grupo 0 = se queda", 960, 560, "teal", 48, en="b"), sec="5. Filtro 3: ANOVA F a mano")
v.b("Un ejemplo chico. [[a]]Una variable vale dos, cuatro y seis en los que se fugan, media cuatro, [[b]]y ocho, diez y doce en los que se quedan, media diez. La media global es siete.",
    CL(), T("ejemplo ilustrativo · 6 clientes", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    L(260, 400, 1660, 400, "tinta", 4),
    *[[Bc(260 + (x / 14) * 1400, 400, 46, 46, "rojo", None, 0, 23, en="a", t=0.2 * i), T(str(x), 260 + (x / 14) * 1400, 460, tam=34, c="rojo", en="a", t=0.2 * i)] for i, x in enumerate(GA)],
    *[[Bc(260 + (x / 14) * 1400, 400, 46, 46, "azul", None, 0, 23, en="b", t=0.2 * i), T(str(x), 260 + (x / 14) * 1400, 460, tam=34, c="azul", en="b", t=0.2 * i)] for i, x in enumerate(GB)],
    T("media 4", 260 + (4 / 14) * 1400, 320, tam=36, c="rojo", w="sb", en="a", t=0.8), T("media 10", 260 + (10 / 14) * 1400, 320, tam=36, c="azul", w="sb", en="b", t=0.8),
    L(260 + 0.5 * 1400, 360, 260 + 0.5 * 1400, 440, "gris", 4, guion=True, en="b", t=1.2), T("global 7", 260 + 0.5 * 1400, 540, tam=34, c="gris", w="sb", en="b", t=1.2))
v.b("Cuadrado medio entre: [[a]]tres por cuatro menos siete al cuadrado, más tres por diez menos siete al cuadrado: veintisiete más veintisiete, cincuenta y cuatro, dividido en uno. [[b]]Cuadrado medio dentro: las desviaciones al cuadrado dentro de cada grupo suman dieciséis, dividido en seis menos dos: cuatro. [[c]]F igual a cincuenta y cuatro entre cuatro: trece coma cinco.",
    T("CM entre = [3·(4 − 7)² + 3·(10 − 7)²] / 1 =", 1080, 640, tam=40, c="tinta", w="sb", anc="rm", en="a"), N(54, 1110, 640, "{:.0f}", 56, "violeta", en="a", t=0.6, anc="lm"),
    T("CM dentro = (4 + 0 + 4 + 4 + 0 + 4) / (6 − 2) =", 1080, 740, tam=40, c="tinta", w="sb", anc="rm", en="b"), N(4, 1110, 740, "{:.0f}", 56, "violeta", en="b", t=0.6, anc="lm"),
    T("F = 54 / 4 =", 1080, 860, tam=48, c="tinta", anc="rm", en="c"), N(13.5, 1110, 860, "{:.1f}", 70, "verde", en="c", t=0.4, anc="lm"))
v.b("Su punto ciego es el mismo de Pearson: [[a]]solo mira diferencias de medias. Con una U, las medias de los grupos pueden coincidir y F da casi cero. [[b]]Y una alerta de la presentación complementaria de la carpeta: dice que ANOVA compara medias del target entre categorías de la predictora. [[c]]Eso es en regresión. En clasificación, como aquí, es al revés: compara la media de la variable entre fugados y no fugados.",
    CL(), P("punto ciego: solo medias → una U da F ≈ 0", 960, 230, "rojo", 48, ent="subir", en="a"),
    alerta(y=400, en="b"), T("«compara medias del target continuo entre categorías de la predictora»", 960, 520, tam=40, c="gris", w="sb", mw=1600, en="b", t=0.3, id="qan"),
    U("qan", tachado=1.0, en="c", dur=0.6), P("aquí: media de la variable en fugados vs no fugados · f_classif(X, y)", 960, 690, "verde", 42, en="c", t=0.4))

# ---- 6. El ranking real
v.b("Ahora sí, los filtros sobre las cincuenta candidatas. Este es el top doce, ordenado por información mutua. [[a]]Primero, segundos, con cero coma uno tres uno. [[b]]Segundo, el logaritmo de segundos, con cero coma uno dos nueve. [[c]]Y siguen contactos por mes, logaritmo del valor del cliente, valor por antigüedad, y tiene reclamo.",
    CL(), T("top 12 de 50 por información mutua · base real (random_state = 0)", 960, 170, tam=34, c="gris", w="sb", ent="subir"),
    T("#", 300, 240, tam=28, c="gris", w="sb"), T("variable", 640, 240, tam=28, c="gris", w="sb"), T("MI", 1180, 240, tam=28, c="gris", w="sb"), T("F de ANOVA", 1500, 240, tam=28, c="gris", w="sb"),
    *[[T(str(i + 1), 300, 300 + 56 * i, tam=32, c="gris", en=k, t=t_), T(nm, 400, 300 + 56 * i, tam=32, c="tinta", w="m", anc="lm", en=k, t=t_ + 0.05),
       Bx(1020, 300 + 56 * i - 16, 1020 + mi * 2200, 300 + 56 * i + 16, "violeta", None, 0, 6, grow="x", en=k, t=t_ + 0.1),
       T(f"{mi:.4f}", 1020 + mi * 2200 + 12, 300 + 56 * i, tam=28, c="violeta", w="sb", anc="lm", en=k, t=t_ + 0.3),
       T(f"{F_:.2f}", 1560, 300 + 56 * i, tam=30, c="tinta", w="sb", anc="rm", en=k, t=t_ + 0.3)]
      for i, ((nm, mi, F_, _p), k, t_) in enumerate(zip(RK, ["a", "b"] + ["c"] * 10, [0, 0] + [0.25 * j for j in range(10)]))],
    sec="6. El ranking de la clase")
v.b("Fíjate en dos cosas. [[a]]Información mutua y F no dan el mismo orden. Intensidad de contacto está décima en información mutua, pero tiene F cinco coma ocho ocho. [[b]]Cuando difieren tanto, la relación no es lineal, y eso ya es información útil. [[c]]Y segundos y su logaritmo ocupan los dos primeros lugares: son la misma información.",
    R(960, 300 + 56 * 9, 1360, 54, "ambar", 18, en="a"), P("MI alta, F bajo → no lineal", 1400, 1000, "ambar", 36, en="b"),
    R(960, 300 + 56 * 0.5, 1360, 112, "rojo", 18, en="c"), P("¿la misma variable dos veces?", 520, 1000, "rojo", 36, en="c", t=0.4))
v.b("Una nota del material. [[a]]El ranking de la diapositiva difiere un poco del notebook: por ejemplo, ahí llamadas por mes aparece con cero coma uno cero dos tres, y aquí con cero coma cero nueve ocho siete. [[b]]No es un error: la información mutua se estima con azar, y cambia con la semilla y con el orden de las columnas. [[c]]En el video siete verás que ese detalle cambia una conclusión de la clase.",
    CL(), P("nota del material", 960, 220, "ambar", 48, ent="subir"),
    *[[Bc(x, 440, 560, 220, "blanco", c, 5, 26, sombra=True, en="a", ent="subir", t=t_), T(lab, x, 390, tam=34, c=c, w="sb", en="a", t=t_ + 0.1),
       T(val, x, 470, tam=60, c=c, en="a", t=t_ + 0.2)] for x, lab, val, c, t_ in ((620, "diapositiva 22", "0.1023", "gris", 0.0), (1300, "notebook", f"{RK[7][1]:.4f}", "violeta", 0.4))],
    T("llamadas_por_mes · información mutua", 960, 590, tam=34, c="tinta", w="sb", en="a", t=0.6),
    P("no es un error: la MI cambia con la semilla y el orden de columnas", 960, 720, "teal", 42, en="b"), P("→ video 7", 960, 850, "violeta", 44, en="c"))

# ---- 7. Error típico
v.b("El error típico con los filtros: [[a]]leer el ranking como si fuera exacto, y quedarse con los primeros. [[b]]Un filtro ordena bien las variables de a una, pero no sabe nada del conjunto. [[c]]Ya lo viste: número de fallas y el XOR se le escapan.",
    CL(), T("«me quedo con el top del ranking»", 960, 300, tam=66, c="tinta", en="a", id="tr"), U("tr", tachado=1.0, color="gris", en="b", t=0.2, dur=0.6),
    P("ordena de a una · no ve el conjunto", 960, 500, "rojo", 54, en="b", t=0.5), P("numero_fallas · XOR", 960, 680, "ambar", 50, en="c"), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tiene reclamo tiene F de mil ciento quince, el más alto de la lista. Pero en información mutua aparece sexta. [[b]]Y reclamos, su gemela idéntica, ni siquiera entra al top doce. ¿Qué te dice eso de un ranking?",
    CL(), P("tiene_reclamo: F = 1115 (el más alto) · MI 6.º lugar", 960, 330, "violeta", 46, en="a"),
    P(f"reclamos: MI {D['sospechosas']['reclamos'][0]:.4f} · lugar {D['rank_mi_pos']['reclamos']}", 960, 480, "azul", 46, en="b"),
    T("¿qué te dice eso de un ranking?", 960, 660, tam=60, c="tinta", en="b", t=0.8), sec="8. Piensa un segundo", pausa=0.6)
v.b("Que depende de la métrica, y de la estimación. [[a]]Dos métricas razonables ordenan distinto, y dos columnas idénticas pueden quedar en lugares distintos. [[b]]Un ranking univariado es un punto de partida, nunca la última palabra.",
    CL(), P("depende de la métrica y de la estimación", 960, 300, "teal", 56, ent="subir"),
    T("dos columnas idénticas, dos lugares distintos", 960, 470, tam=48, c="tinta", w="sb", en="a"),
    P("ranking univariado = punto de partida, no la última palabra", 960, 660, "violeta", 48, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]filter, wrapper y embedded son etapas de un embudo, no rivales. [[b]]Los filtros miden de a una: varianza, correlación, información mutua y ANOVA F. [[c]]Y su punto ciego es el conjunto: lo que solo aporta acompañado, se les escapa.",
    CL(), P("filter → embedded → wrapper: un embudo", 960, 300, "violeta", 52, en="a"), P("varianza · Pearson · MI · ANOVA F", 960, 480, "teal", 52, en="b"),
    P("punto ciego: el conjunto", 960, 660, "rojo", 54, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el clímax de la clase. Cuatro variables contra cuatro variables.",
    CL(), siguiente(7, "Cuatro contra cuatro: el ranking ingenuo"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
