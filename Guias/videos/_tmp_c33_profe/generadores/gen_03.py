import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, caja_formula

OUT = out("guion_c33_03_construir.json")
v = V("Construir variables: de 12 a 50")
CS, RS = D["clv_stats"], D["resumen"]

v.portada("De 12 a 50 variables", "Fabricar candidatas con sentido de negocio, y la redundancia que eso trae",
          "Segunda parada: construir variables. Con doce columnas no hay nada que seleccionar: primero hay que fabricar candidatas.")

# ---- 1. Hook: dos clientes con 60 llamadas
v.b("Mira a estos dos clientes. [[a]]Los dos hicieron sesenta llamadas. [[b]]Pero uno lleva veinte meses con la empresa, [[c]]y el otro, solo dos.",
    CL(), T("dos clientes", 960, 200, tam=48, c="gris", w="sb", ent="subir"),
    *[[Bc(x, 500, 640, 380, "blanco", c, 5, 30, sombra=True, ent="subir", t=0.2 * j), T(nm, x, 360, tam=46, c=c, t=0.2 * j + 0.1),
       T("60 llamadas", x, 470, tam=60, c="tinta", en="a", t=0.2 * j)] for j, (x, c, nm) in enumerate(((560, "azul", "cliente A"), (1360, "violeta", "cliente B")))],
    T("20 meses", 560, 580, tam=50, c="azul", w="sb", en="b"), T("2 meses", 1360, 580, tam=50, c="violeta", w="sb", en="c"),
    ilustrativo(1000), sec="1. Dos clientes iguales… o no")
v.b("¿Son iguales? No. [[a]]Sesenta entre veinte: el cliente A llama tres veces al mes. [[b]]Sesenta entre dos: el cliente B llama treinta veces al mes. [[c]]Lo que distingue a estos clientes no es la variable cruda: es el ratio.",
    T("60 / 20 =", 480, 780, tam=48, c="tinta", w="sb", en="a"), N(3, 680, 780, "{:.0f}", 70, "azul", suf=" / mes", en="a", t=0.3, anc="lm"),
    T("60 / 2 =", 1270, 780, tam=48, c="tinta", w="sb", en="b"), N(30, 1460, 780, "{:.0f}", 70, "violeta", suf=" / mes", en="b", t=0.3, anc="lm"),
    P("lo que distingue es el ratio", 960, 920, "ambar", 50, en="c"))
v.b("Un ratio normaliza una magnitud por su exposición. [[a]]Llamadas por mes, valor del cliente por mes de antigüedad, segundos por llamada. [[b]]En la experiencia del profesor, los ratios son algo bien potente: discriminan mejor que la variable cruda.",
    CL(), caja_formula("ratio = magnitud / exposición", y=280, c="ambar", tam=60, w=1200),
    *[P(tx, x, 500, "violeta", 40, en="a", t=0.4 * j) for j, (tx, x) in enumerate((("llamadas_por_mes", 450), ("clv_por_antiguedad", 960), ("duracion_prom_llamada", 1480)))],
    T("«algo bien potente»", 960, 700, tam=64, c="ambar", en="b"), T("profesor Eduardo Rodríguez", 960, 790, tam=32, c="gris", w="sb", en="b", t=0.5))

# ---- 2. Con 12 no hay nada que seleccionar
v.b("¿Por qué construir? [[a]]Porque seleccionar entre doce variables es un ejercicio de juguete. [[b]]El problema real aparece con cincuenta, doscientas o dos mil candidatas, [[c]]cuando hay que justificar cada descarte.",
    CL(), T("12 columnas", 560, 360, tam=80, c="gris", en="a"), T("= ejercicio de juguete", 560, 470, tam=40, c="gris", w="sb", en="a", t=0.4),
    T("50 · 200 · 2.000", 1360, 360, tam=80, c="violeta", en="b"), T("= el problema real", 1360, 470, tam=40, c="violeta", w="sb", en="b", t=0.4),
    P("hay que justificar cada descarte", 960, 700, "ambar", 54, en="c"), sec="2. Con 12 no hay nada que elegir")

# ---- 3. Las familias
FAM = [("originales", "reclamos, segundos…", 10, "gris"), ("ratios", "clv_por_antiguedad", 13, "ambar"), ("interacciones", "reclamo_x_antig", 3, "rojo"),
       ("logaritmos", "log_segundos", 5, "azul"), ("binarias", "tiene_reclamo", 5, "teal"), ("cuartiles", "edad_q1 … edad_q4", 12, "violeta"),
       ("one-hot", "plan_2, estado_2", 2, "verde")]
ESC = 26
acum = [sum(f[2] for f in FAM[:i]) for i in range(len(FAM) + 1)]


def fam(i, en, t=0.0):
    nm, ej, n, c = FAM[i]
    y = 290 + 88 * i
    x0 = 1040 + acum[i] * ESC * 0.62
    return [T(nm, 300, y, tam=38, c=c, w="b", anc="lm", en=en, t=t), T(ej, 600, y, tam=28, c="tinta", w="m", anc="lm", en=en, t=t + 0.1),
            T(str(n), 1010, y, tam=40, c=c, anc="rm", en=en, t=t + 0.15),
            Bx(x0, y - 26, x0 + n * ESC * 0.62, y + 26, c, None, 0, 6, grow="x", en=en, t=t + 0.2)]


v.b("Hoy aplicamos la misma lógica de la clase treinta y dos, pero en serio. [[a]]Se conservan diez variables originales. [[b]]Se agregan trece ratios [[c]]y tres interacciones, efectos que solo existen combinados, como reclamo por antigüedad.",
    CL(), T("las familias de candidatas", 960, 190, tam=44, c="tinta", ent="subir"), fam(0, "a"), fam(1, "b"), fam(2, "c"),
    sec="3. Las siete familias")
v.b("[[a]]Cinco logaritmos, para aplastar colas largas. [[b]]Cinco indicadores binarios, como tiene reclamo o cliente nuevo. [[c]]Doce columnas de cuartiles, de edad, antigüedad y valor del cliente. [[d]]Y dos one-hot: plan dos y estado dos.",
    fam(3, "a"), fam(4, "b"), fam(5, "c"), fam(6, "d"))
v.b("Sumemos: [[a]]diez, más trece, más tres, más cinco, más cinco, más doce, más dos. [[b]]Cincuenta candidatas.",
    L(260, 905, 1660, 905, "tinta", 3, en="a"), T("10 + 13 + 3 + 5 + 5 + 12 + 2 =", 860, 960, tam=44, c="tinta", w="sb", en="a"),
    N(D["p"], 1260, 960, "{:.0f}", 72, "ambar", en="b", anc="lm"))
v.b("¿Y por qué diez originales, y no doce? [[a]]Porque tipo de plan y estado se reemplazan por sus indicadores, plan dos y estado dos. [[b]]Las dos originales se van. Por eso quedan cincuenta, y no cincuenta y dos.",
    CL(), T("12 originales", 560, 300, tam=64, c="gris", ent="subir"),
    T("tipo_plan", 420, 480, tam=48, c="rojo", w="m", en="a", id="tp"), T("estado", 760, 480, tam=48, c="rojo", w="m", en="a", id="es"),
    U("tp", tachado=1.0, en="a", t=0.8, dur=0.5), U("es", tachado=1.0, en="a", t=1.0, dur=0.5),
    L(980, 480, 1110, 480, "gris", 7, True, en="a", t=1.2), P("plan_2 · estado_2", 1400, 480, "verde", 52, en="a", t=1.4),
    T("10 + 2 indicadores, dentro de las 50", 960, 680, tam=48, c="tinta", w="sb", en="b"), P("50, no 52", 960, 820, "ambar", 56, en="b", t=0.6))

# ---- 4. Ejemplo: ratios con épsilon
v.b("Ahora, un detalle técnico que salva pipelines. [[a]]Todo ratio se calcula con un épsilon en el denominador: [[b]]x dividido en y más un millonésimo.",
    CL(), T("ratios con épsilon", 960, 200, tam=48, c="tinta", ent="subir"),
    caja_formula("ratio = x / (y + 1e-6)", y=420, c="violeta", tam=70, w=1100, en="a"),
    T("ε = 0.000001", 960, 600, tam=56, c="violeta", w="m", en="b"), sec="4. Ejemplo: el épsilon")
EJ = [("20 meses, 60 llamadas", "60 / (20 + 0.000001)", "3.0 llamadas por mes", "azul"), ("60 llamadas, 4.500 segundos", "4.500 / (60 + 0.000001)", "75 segundos por llamada", "azul"),
      ("0 meses (cliente nuevo), 60 llamadas", "60 / (0 + 0.000001)", "60.000.000", "rojo")]
v.b("Tres casos del apunte. [[a]]Veinte meses y sesenta llamadas: tres llamadas por mes. [[b]]Sesenta llamadas y cuatro mil quinientos segundos: setenta y cinco segundos por llamada. [[c]]Y un cliente nuevo, con cero meses: sesenta millones.",
    CL(), T("ejemplo del apunte", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    *[[Bc(960, 340 + 170 * i, 1640, 140, "blanco", c, 4, 24, sombra=True, en=k, ent="subir"), T(a, 180, 315 + 170 * i, tam=32, c="gris", w="sb", anc="lm", en=k, t=0.1),
       T(b, 180, 365 + 170 * i, tam=38, c="tinta", w="m", anc="lm", en=k, t=0.2), T(r, 1740, 340 + 170 * i, tam=46, c=c, anc="rm", en=k, t=0.5)]
      for i, ((a, b, r, c), k) in enumerate(zip(EJ, "abc"))])
v.b("Sin el épsilon, esa tercera fila daría infinito. [[a]]Y un solo infinito rompe el escalado, el modelo, y todo el pipeline de selección que viene después. [[b]]Ojo también con que el resultado es gigante: esos casos conviene revisarlos, por ejemplo con un log o un tope.",
    T("60 / 0 = ∞", 960, 860, tam=64, c="rojo", w="m"),
    P("un infinito rompe todo el pipeline", 560, 970, "rojo", 40, en="a"), P("valores gigantes → log o tope", 1400, 970, "ambar", 40, en="b"))
v.b("¿Y en nuestra base? [[a]]Después de limpiar, ningún cliente tiene antigüedad cero, y el máximo de llamadas por mes es veintidós coma uno. [[b]]Se verificó: cero infinitos. [[c]]Pero el código tiene que estar listo para el cliente nuevo que llegue mañana.",
    CL(), T("en la base real", 960, 200, tam=44, c="gris", w="sb", ent="subir"),
    Bc(560, 470, 640, 300, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), N(D["max_llamadas_por_mes"], 560, 440, "{:.1f}", 110, "azul", en="a", t=0.1),
    T("máximo de llamadas por mes", 560, 560, tam=32, c="azul", w="sb", en="a", t=0.2),
    Bc(1360, 470, 640, 300, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), N(D["inf"], 1360, 440, "{:.0f}", 110, "verde", en="b", t=0.1),
    T("np.isinf(X).sum().sum()", 1360, 560, tam=30, c="verde", w="m", en="b", t=0.2),
    P("el código debe estar listo para el cliente nuevo", 960, 780, "ambar", 46, en="c"))

# ---- 5. Logs, binarias, cuartiles
v.b("Las otras familias, con números de la base. [[a]]El log de uno más x aplasta las colas largas: el valor del cliente va de cero a dos mil ciento sesenta y cinco, [[b]]y su logaritmo, de cero a siete coma sesenta y ocho. La mediana, doscientos veintiocho, queda en cinco coma cuarenta y cuatro.",
    CL(), T("log(1 + clv)", 960, 200, tam=56, c="azul", w="m", ent="subir"),
    L(300, 420, 1620, 420, "tinta", 4, en="a"), T("clv", 230, 420, tam=36, c="gris", w="sb", anc="rm", en="a"),
    *[[Bc(300 + (val / CS["max"]) * 1320, 420, 22, 22, c, None, 0, 11, en="a", t=0.2 * j), T(lab, 300 + (val / CS["max"]) * 1320, 370, tam=32, c=c, en="a", t=0.2 * j + 0.1)]
      for j, (val, lab, c) in enumerate(((0, "0", "gris"), (CS["med"], "228", "violeta"), (CS["max"], "2.165", "rojo")))],
    L(300, 680, 1620, 680, "tinta", 4, en="b"), T("log", 230, 680, tam=36, c="gris", w="sb", anc="rm", en="b"),
    *[[Bc(300 + (val / CS["log_max"]) * 1320, 680, 22, 22, c, None, 0, 11, en="b", t=0.2 * j), T(lab, 300 + (val / CS["log_max"]) * 1320, 630, tam=32, c=c, en="b", t=0.2 * j + 0.1)]
      for j, (val, lab, c) in enumerate(((0, "0", "gris"), (CS["log_med"], f"{CS['log_med']:.2f}", "violeta"), (CS["log_max"], f"{CS['log_max']:.2f}", "rojo")))],
    T("la mediana deja de estar «aplastada» contra el cero", 960, 840, tam=38, c="violeta", w="sb", en="b", t=1.2), sec="5. Logs, binarias y cuartiles")
v.b("[[a]]Las binarias marcan una condición de negocio: tiene reclamo, o cliente nuevo, con menos de doce meses. [[b]]Los cuartiles parten una variable en cuatro tramos, para capturar relaciones no lineales. [[c]]Y las interacciones multiplican dos variables que actúan juntas.",
    CL(), tarjetas([(360, "teal", "binarias", "cliente_nuevo = antigüedad < 12", "a"), (960, "violeta", "cuartiles", "clv_q1 … clv_q4 con pd.qcut", "b"),
                    (1560, "rojo", "interacciones", "reclamo_x_antig = reclamos · antigüedad", "c")], y=500, h=400, tam_d=34))
v.b("En código se ve así. [[a]]Un ratio con épsilon. [[b]]El logaritmo con log uno pe. [[c]]Y los cuartiles con qcut.",
    CL(), codigo(["eps = 1e-6", "d['llamadas_por_mes'] = d.frec_llamadas / (d.antiguedad + eps)", "d['log_clv'] = np.log1p(d.clv)",
                  "d['clv_q'] = pd.qcut(d.clv, 4, labels=False)"], y=460, tam=36, dt=0.0),
    R(960, 404, 1600, 110, "ambar", 16, en="a"), R(960, 488, 1600, 56, "azul", 16, en="b"), R(960, 544, 1600, 56, "violeta", 16, en="c"))

# ---- 6. ¿Sirvió?
v.b("¿Y sirvió construir? [[a]]Con las doce originales, la regresión logística llega a un AUC de cero coma novecientos veinte. [[b]]Con las cincuenta candidatas, a cero coma novecientos cuarenta y uno. Dos puntos más.",
    CL(), T("AUC con validación cruzada de 5 folds · base real", 960, 190, tam=38, c="gris", w="sb", ent="subir"),
    *[[Bx(x - 140, 860 - (a - 0.8) * 3500, x + 140, 860, c, None, 0, 12, grow="y", en=k), N(a, x, 860 - (a - 0.8) * 3500 - 45, "{:.3f}", 66, c, en=k, t=0.3, dur=0.8),
       T(lab, x, 910, tam=34, c=c, w="sb", en=k)]
      for x, a, c, lab, k in ((660, RS[0][2], "gris", "12 originales", "a"), (1260, RS[1][2], "violeta", "50 candidatas", "b"))],
    L(420, 860, 1500, 860, "tinta", 4), T("eje desde 0.80", 1640, 860, tam=26, c="gris", w="sb"), sec="6. ¿Sirvió construir?")

# ---- 7. Error típico
v.b("Pero aquí viene el error típico: [[a]]creer que basta con crear variables. [[b]]Construir genera redundancia por construcción. Tiene reclamo y reclamos tienen correlación uno: [[c]]como reclamos ya vale cero o uno, es la misma información con otro nombre.",
    CL(), T("crear variables = mejor modelo", 960, 260, tam=70, c="tinta", ent="subir", id="cvm"), U("cvm", tachado=1.0, color="gris", en="a", t=0.8, dur=0.6),
    P("reclamos  ↔  tiene_reclamo", 760, 480, "rojo", 54, en="b"), N(D["r_reclamos_tiene"], 1380, 480, "r = {:.3f}", 66, "rojo", en="b", t=0.4, anc="lm"),
    T("reclamos vale 0 o 1  →  (reclamos > 0) es lo mismo que reclamos", 960, 640, tam=42, c="tinta", w="sb", en="c"), sec="7. El error típico")
v.b("Entre las cincuenta candidatas hay once pares con correlación sobre cero coma noventa. [[a]]Crear variables sin seleccionar después no mejora el modelo: lo infla. [[b]]Por eso el siguiente paso de la clase es justamente decidir cuáles sobran.",
    CL(), N(D["n_pares90"], 960, 330, "{:.0f}", 160, "rojo"), T("pares con |r| > 0.90 entre las 50", 960, 460, tam=44, c="rojo", w="sb", t=0.3),
    P("crear sin seleccionar → infla el modelo", 960, 640, "ambar", 54, en="a"), P("siguiente: ¿cuáles sobran?", 960, 800, "violeta", 50, en="b"))

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Construir un ratio o un logaritmo, ¿mira la columna fuga? [[b]]¿Puede hacerse antes de separar entrenamiento y prueba?",
    CL(), T("ratio · logaritmo · one-hot", 960, 320, tam=66, c="violeta", en="a"), T("¿miran la fuga?", 960, 520, tam=60, c="tinta", en="a", t=0.6),
    T("¿pueden ir antes del split?", 960, 660, tam=60, c="tinta", en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("No miran la fuga: son fórmulas fijas, fila por fila. [[a]]Por eso pueden ir antes del split. [[b]]Pero ojo con la ventana temporal: una variable construida con información posterior al momento de predecir es leakage, aunque no use la columna fuga.",
    CL(), P("fórmula fija, fila por fila", 960, 260, "verde", 56, ent="subir"), K(1420, 260, "check", 60, t=0.4),
    P("puede ir antes del split", 960, 420, "verde", 56, en="a"),
    Bc(960, 680, 1500, 220, "ambarclaro", "ambar", 4, 30, en="b"), T("ventana temporal: información del futuro = leakage", 960, 650, tam=44, c="ambar", w="b", en="b", t=0.1),
    T("aunque no uses la columna fuga", 960, 720, tam=36, c="tinta", w="sb", en="b", t=0.3))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]los ratios normalizan por exposición, y siempre llevan épsilon. [[b]]Construir llevó la base de doce a cincuenta candidatas, y el AUC de cero coma novecientos veinte a cero coma novecientos cuarenta y uno. [[c]]Pero también fabricó redundancia: ahora hay que seleccionar.",
    CL(), P("ratio = x / (y + 1e-6)", 960, 300, "ambar", 54, en="a"), P("12 → 50 candidatas · AUC 0.920 → 0.941", 960, 480, "violeta", 52, en="b"),
    P("11 pares redundantes → hay que seleccionar", 960, 660, "rojo", 50, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]por qué una variable sobra. Irrelevante, redundante, o útil solo en conjunto.",
    CL(), siguiente(4, "Irrelevante, redundante o útil en conjunto"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
