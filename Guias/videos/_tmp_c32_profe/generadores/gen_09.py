import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_09_caso.json")
v = V("Pipeline sin fuga y el caso Churn real")
FE, RE = D["fuga_exp"], D["real"]
PA = RE["pares"]

v.portada("Pipeline sin fuga y caso real", "Ajustar todo con entrenamiento, y el caso con 3.150 clientes reales",
          "Octava parada: cómo armar todo esto sin hacer trampa, y el caso aplicado con la base real de churn.")

# ---- 1. Hook
v.b("Imagina que un profesor arma la prueba, [[a]]y antes de tomarla, deja que los alumnos estudien con las mismas preguntas. [[b]]Todos sacan siete. Pero la nota ya no mide nada.",
    CL(), Bc(560, 520, 460, 520, "blanco", "violeta", 5, 26, sombra=True, ent="subir"), T("prueba", 560, 330, tam=50, c="violeta", t=0.1),
    *[L(380, 420 + 70 * i, 740, 420 + 70 * i, "linea", 5, t=0.15 + 0.05 * i) for i in range(5)],
    L(800, 520, 1000, 520, "rojo", 7, True, en="a"), P("¡las mismas preguntas!", 1380, 420, "rojo", 46, en="a", t=0.2),
    T("nota: 7.0", 1380, 600, tam=80, c="verde", en="b"), P("pero no mide nada", 1380, 760, "ambar", 46, en="b", t=0.6), ilustrativo(990, texto="analogía"),
    sec="1. Estudiar con la prueba")
v.b("Eso es la fuga de datos. [[a]]Y la regla para evitarla es una sola: [[b]]todo lo que aprende de los datos se ajusta solo con el entrenamiento.",
    CL(), P("fuga de datos", 960, 300, "rojo", 66, ent="subir"),
    Bc(960, 580, 1500, 200, "verdeclaro", "verde", 6, 40, en="b", ent="pop"), T("todo lo que aprende de los datos se ajusta solo con entrenamiento", 960, 580, tam=46, c="verde", mw=1380, en="b", t=0.1))

# ---- 2. Qué aprende cada paso
PASOS = [("Imputador", "la mediana o la moda", "Pipeline · fit con entrenamiento", "teal"),
         ("StandardScaler", "media y desviación de cada columna", "Pipeline · fit con entrenamiento", "teal"),
         ("OneHot / Ordinal", "la lista de categorías", "ColumnTransformer dentro del Pipeline", "violeta"),
         ("TargetEncoder", "la tasa de fuga por categoría (mira «y»)", "Pipeline · validación cruzada interna", "ambar"),
         ("PCA", "los componentes", "Pipeline · después del escalador", "azul"),
         ("t-SNE", "no tiene transform", "solo para explorar, fuera del modelo", "rojo")]


def fila(i, en, t=0.0):
    n, a, d, c = PASOS[i]
    y = 300 + 115 * i
    return [Bc(960, y, 1720, 96, "blanco", c, 4, 22, en=en, ent="subir", t=t), T(n, 140, y, tam=36, c=c, w="b", anc="lm", en=en, t=t + 0.1),
            T(a, 820, y, tam=32, c="tinta", w="sb", en=en, t=t + 0.2), T(d, 1800, y, tam=28, c="gris", w="sb", anc="rm", en=en, t=t + 0.3)]


v.b("Veamos qué aprende cada paso. [[a]]El imputador aprende la mediana o la moda. [[b]]El StandardScaler, la media y la desviación. [[c]]Los codificadores, la lista de categorías.",
    CL(), T("paso", 140, 200, tam=32, c="gris", w="sb", anc="lm"), T("qué aprende de los datos", 820, 200, tam=32, c="gris", w="sb"), T("dónde va", 1800, 200, tam=32, c="gris", w="sb", anc="rm"),
    fila(0, "a"), fila(1, "b"), fila(2, "c"), sec="2. Qué aprende cada paso")
v.b("[[a]]El TargetEncoder aprende la tasa de fuga por categoría: mira la respuesta, así que necesita validación interna. [[b]]PCA aprende los componentes, y va después del escalador. [[c]]Y t-SNE no tiene transform: se queda afuera del modelo.",
    fila(3, "a"), fila(4, "b"), fila(5, "c"))

# ---- 3. Experimento
v.b("¿Qué tan grave es saltarse la regla? La profesora hizo un experimento con datos de puro ruido. [[a]]Cien filas, mil columnas al azar, y etiquetas también al azar. [[b]]No hay nada que aprender: lo honesto es acertar la mitad, cincuenta por ciento.",
    CL(), P("datos de puro ruido · 40 repeticiones", 960, 220, "gris", 46, ent="subir"),
    *[[Bc(x, 480, 420, 220, "blanco", "gris", 5, 28, sombra=True, en="a", ent="subir", t=t_), T(n, x, 450, tam=60, c="tinta", en="a", t=t_ + 0.1), T(d, x, 520, tam=32, c="gris", w="sb", en="a", t=t_ + 0.2)]
      for x, n, d, t_ in ((480, "100", "filas", 0.0), (960, "1.000", "columnas al azar", 0.2), (1440, "y", "al azar", 0.4))],
    P("lo honesto: 50 % de exactitud", 960, 760, "verde", 52, en="b"), sec="3. Un experimento con ruido")
v.b("Primero, la trampa: elegir las diez «mejores» columnas mirando todas las filas, y después validar. [[a]]La exactitud sale cero coma setenta y nueve. [[b]]Dentro de un Pipeline, donde la selección se rehace en cada pliegue: cero coma cincuenta y uno. La verdad.",
    CL(), T("exactitud en validación cruzada (promedio de 40)", 960, 190, tam=38, c="gris", w="sb"),
    L(300, 860, 1620, 860, "tinta", 4),
    Bx(440, 860 - FE["sel_fuera"] * 700, 640, 860, "rojo", None, 0, 10, grow="y", en="a"), N(FE["sel_fuera"], 540, 860 - FE["sel_fuera"] * 700 - 40, "{:.2f}", 56, "rojo", en="a", t=0.3, dur=0.6),
    T("selección fuera", 540, 900, tam=30, c="rojo", w="sb", en="a"),
    Bx(720, 860 - FE["sel_pipe"] * 700, 920, 860, "verde", None, 0, 10, grow="y", en="b"), N(FE["sel_pipe"], 820, 860 - FE["sel_pipe"] * 700 - 40, "{:.2f}", 56, "verde", en="b", t=0.3, dur=0.6),
    T("selección en Pipeline", 820, 900, tam=30, c="verde", w="sb", en="b"),
    L(300, 860 - 0.5 * 700, 1620, 860 - 0.5 * 700, "tinta", 3, guion=True, t=0.2), T("azar = 0.5", 1720, 860 - 0.5 * 700, tam=30, c="gris", w="sb", t=0.3))
v.b("¿Y con PCA? [[a]]PCA ajustado afuera da cero coma cincuenta y uno, y adentro, cero coma cincuenta y dos. [[b]]PCA no mira la etiqueta, así que no la filtra. Pero igual es un mal hábito: la prueba influyó en los componentes.",
    Bx(1080, 860 - FE["pca_fuera"] * 700, 1280, 860, "ambar", None, 0, 10, grow="y", en="a"), N(FE["pca_fuera"], 1180, 860 - FE["pca_fuera"] * 700 - 40, "{:.2f}", 56, "ambar", en="a", t=0.3, dur=0.6),
    T("PCA fuera", 1180, 900, tam=30, c="ambar", w="sb", en="a"),
    Bx(1360, 860 - FE["pca_pipe"] * 700, 1560, 860, "verde", None, 0, 10, grow="y", en="a", t=0.3), N(FE["pca_pipe"], 1460, 860 - FE["pca_pipe"] * 700 - 40, "{:.2f}", 56, "verde", en="a", t=0.6, dur=0.6),
    T("PCA en Pipeline", 1460, 900, tam=30, c="verde", w="sb", en="a", t=0.3),
    P("PCA no ve «y»: no la filtra, pero el hábito es peligroso", 960, 990, "ambar", 36, en="b"))

# ---- 4. El Pipeline honesto
v.b("Así se ve el Pipeline honesto de la clase. [[a]]Escalador, PCA con ocho componentes, y regresión logística, ajustados con fit sobre el entrenamiento. [[b]]Y si usas GridSearchCV o cross val score, el escalador y PCA se vuelven a ajustar en cada pliegue. Así eliges k sin mirar la prueba.",
    CL(), T("el Pipeline de la clase", 960, 200, tam=44, c="gris", w="sb", ent="subir"),
    Bc(960, 400, 1700, 200, "blanco", "violeta", 4, 26, sombra=True, en="a", ent="subir"),
    T("make_pipeline(StandardScaler(), PCA(n_components=8),", 960, 370, tam=40, c="tinta", w="m", en="a", t=0.1),
    T("LogisticRegression()).fit(X_train, y_train)", 960, 440, tam=40, c="tinta", w="m", en="a", t=0.3),
    P("GridSearchCV: escalador y PCA se reajustan en cada pliegue", 960, 640, "teal", 44, en="b"), P("k se elige sin mirar la prueba", 960, 770, "verde", 48, en="b", t=0.8),
    sec="4. El Pipeline honesto")
CHK = [("construyo variables antes de reducir", "teal"), ("divido antes de escalar", "verde"), ("estandarizo antes de PCA", "azul"),
       ("elijo k con criterio y con validación", "violeta"), ("interpreto con las cargas, no con la diapositiva", "ambar"), ("uso t-SNE solo para explorar", "rojo")]
v.b("El checklist del flujo. [[a]]Construyo variables antes de reducir. Divido antes de escalar. Estandarizo antes de PCA. [[b]]Elijo k con criterio y con validación. Interpreto con las cargas. Y uso t-SNE solo para explorar.",
    CL(), T("checklist del flujo", 960, 190, tam=48, c="tinta", ent="subir"),
    *[[Bc(960, 310 + 110 * i, 1300, 90, "blanco", c, 4, 22, en="a" if i < 3 else "b", ent="subir", t=0.35 * (i % 3)),
       T(tx, 400, 310 + 110 * i, tam=36, c="tinta", w="sb", anc="lm", en="a" if i < 3 else "b", t=0.35 * (i % 3) + 0.1),
       K(1540, 310 + 110 * i, "check", 50, en="a" if i < 3 else "b", t=0.35 * (i % 3) + 0.3)] for i, (tx, c) in enumerate(CHK)])

# ---- 5. Caso real
v.b("Ahora el caso aplicado de las sesiones treinta y uno y treinta y dos. [[a]]La base real: tres mil ciento cincuenta clientes de una telco, doce variables, [[b]]y un quince coma siete por ciento de fuga.",
    CL(), P("caso aplicado · base real de churn", 960, 220, "violeta", 54, ent="subir"),
    *[[Bc(x, 520, 440, 280, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(val, x, 490, tam=90, c=c, en=k, t=0.1), T(lab, x, 590, tam=34, c="gris", w="sb", en=k, t=0.2)]
      for x, c, val, lab, k in ((440, "violeta", "3.150", "clientes", "a"), (960, "azul", "12", "variables", "a"), (1480, "rojo", f"{RE['fuga']:.1f} %", "se fuga", "b"))],
    sec="5. El caso real: desafíos 1 y 2")
v.b("Desafío uno. Retención quiere anticipar quién cancela, y tiene el historial con la fuga, cero o uno. [[a]]Eso es supervisado: clasificación, con y igual a fuga. [[b]]Marketing quiere descubrir tipos de clientes, sin etiqueta: no supervisado.",
    CL(), tarjetas([(560, "violeta", "Retención", "supervisado · clasificación · y = fuga", "a"), (1360, "teal", "Marketing", "no supervisado · tipos de clientes", "b")], w=720, h=360))
v.b("Una precisión que ya vimos en la clase treinta y uno: [[a]]el caso dice que predecir cuántos meses le quedan a un cliente sería regresión. [[b]]En rigor es análisis de supervivencia, porque de los que aún no cancelan no sabes cuándo lo harán.",
    P("«meses que le quedan» → el caso dice: regresión", 960, 790, "gris", 40, en="a"), P("en rigor: análisis de supervivencia (datos censurados)", 960, 900, "ambar", 40, en="b"))
v.b("Desafío dos: las correlaciones de la base real. [[a]]Segundos y llamadas, cero coma noventa. Mensajes y valor del cliente, cero coma noventa y dos. Contactos y llamadas, cero coma setenta y tres. [[b]]Fallas y monto, cero coma cincuenta y nueve. Y estado con segundos, menos cero coma treinta.",
    CL(), T("correlaciones · base real (verificadas)", 960, 190, tam=40, c="gris", w="sb"),
    *[[Bc(960, 300 + 110 * i, 1200, 90, "blanco", c, 4, 22, en=k, ent="subir", t=0.2 * (i % 3)), T(n, 420, 300 + 110 * i, tam=36, c="tinta", w="sb", anc="lm", en=k, t=0.2 * (i % 3) + 0.1),
       N(PA[key], 1440, 300 + 110 * i, "{:+.2f}", 44, c, en=k, t=0.2 * (i % 3) + 0.2, dur=0.5)]
      for i, (n, key, c, k) in enumerate((("segundos ↔ frec_llamadas", "segundos|frec_llamadas", "rojo", "a"), ("frec_sms ↔ clv", "frec_sms|clv", "rojo", "a"),
                                          ("n_contactos ↔ frec_llamadas", "n_contactos|frec_llamadas", "ambar", "a"), ("numero_fallas ↔ monto", "numero_fallas|monto", "ambar", "b"),
                                          ("estado ↔ segundos", "estado|segundos", "azul", "b")))])
v.b("¿Qué problema es? [[a]]Multicolinealidad. [[b]]¿Y qué hay que hacer antes de PCA? Dos pasos: estandarizar, porque segundos llega a miles; y codificar con one-hot el tipo de plan y el estado, que vienen como uno y dos.",
    P("multicolinealidad", 960, 880, "violeta", 46, en="a"), P("antes de PCA: estandarizar + one-hot", 960, 980, "teal", 42, en="b"))

# ---- 6. Desafío 3
PR = RE["prueba"]
v.b("Desafío tres. Con quince columnas, PCA en la base real. [[a]]PC1 explica veintiséis coma tres por ciento, [[b]]y se necesitan siete componentes para llegar al ochenta coma cuatro.",
    CL(),
    G(260, 240, 1180, 860, 0.4, 15.6, 0, 100, [dict(xs=list(range(1, 16)), ys=RE["acum"], color="ambar", grosor=6, marcadores=8, revela=0.0)],
      xticks=list(range(1, 16)), yticks=[0, 20, 40, 60, 80, 100], yfmt="{:.0f}", xlabel="componente", id="gr", t=0.0),
    T("% de varianza · base real", 260, 190, tam=32, c="tinta", w="sb", anc="lm"),
    *[Bx(k - 0.34, RE["var"][k - 1], k + 0.34, 0, "violeta", None, 0, 6, ref="gr", grow="y", t=0.05 * k) for k in range(1, 16)],
    N(RE["var"][0], 1500, 380, "PC1 = {:.1f}", 64, "violeta", suf=" %", en="a", dur=0.7),
    U("gr", **{"series.0.revela": 6 / 14}, en="b", dur=1.4), Bc(7, RE["acum"][6], 26, 26, "blanco", "verde", 5, 13, ref="gr", ent="pop", en="b", t=1.4),
    N(RE["acum"][6], 1500, 520, "7 → {:.1f}", 64, "verde", suf=" %", en="b", t=1.2, dur=0.7), sec="6. El caso real: desafío 3")
v.b("Y la logística en la prueba: [[a]]con las quince variables, F1 cero coma quinientos setenta y cuatro, y MCC cero coma quinientos cuarenta y seis. [[b]]Con siete componentes, cero coma quinientos cuarenta y ocho, y cero coma quinientos treinta y cuatro.",
    CL(), T("base real · en la prueba", 960, 200, tam=40, c="gris", w="sb"),
    *[[Bc(960, y, 1300, 150, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, 400, y, tam=44, c=c, anc="lm", en=k, t=0.1),
       T(f"F1 {PR[key]['F1']:.3f}", 1080, y, tam=48, c="tinta", en=k, t=0.2), T(f"MCC {PR[key]['MCC']:.3f}", 1430, y, tam=48, c="tinta", en=k, t=0.3)]
      for y, c, n, key, k in ((380, "gris", "15 variables", "15", "a"), (580, "violeta", "7 componentes", "pca7", "b"))])
v.b("Aquí sí hay señal real: el F1 es más del doble que en los datos simulados. [[a]]Y el recorte cuesta algo: el F1 baja cero coma cero dos seis. [[b]]¿Conviene? Depende del negocio: un modelo más simple, contra un poco de desempeño. Y la decisión se confirma con validación cruzada, no con una sola partición.",
    P("F1 baja 0.574 − 0.548 = 0.026", 960, 760, "ambar", 46, en="a"), P("simplicidad frente a desempeño: decide el negocio, con validación cruzada", 960, 880, "violeta", 38, en="b"))

# ---- 7. Pregunta 3.2
CI = RE["circ"]
v.b("La pregunta tres punto dos del caso dice: [[a]]«fallas y reclamos apuntan en dirección casi opuesta al uso». [[b]]Y propone: más fallas, menos uso, fuga silenciosa. Comprobémoslo con los datos.",
    CL(), P("pregunta 3.2 del caso", 960, 200, "ambar", 46, ent="subir"),
    T("«fallas y reclamos apuntan opuestos al uso»", 960, 360, tam=54, c="tinta", id="q32", en="a"), T("→ más fallas, menos uso, «fuga silenciosa»", 960, 470, tam=44, c="gris", w="sb", en="b"),
    sec="7. Pregunta 3.2: comprobar antes de creer")
v.b("En la base real, las fallas se correlacionan positivamente con el uso. [[a]]Con segundos, más cero coma cuarenta y nueve. Con llamadas, más cero coma cincuenta y siete. Con monto, más cero coma cincuenta y nueve: el propio caso lo reporta. [[b]]Y su correlación con PC1 es más cero coma sesenta y uno, la misma dirección que segundos, con cero coma ochenta y cuatro.",
    *[[P(f"fallas ↔ {n}: {PA[key]:+.2f}", x, 620, "rojo", 40, en="a", t=0.3 * j)] for j, (n, key, x) in enumerate((("segundos", "numero_fallas|segundos", 440), ("llamadas", "numero_fallas|frec_llamadas", 960), ("monto", "numero_fallas|monto", 1480)))],
    T(f"correlación con PC1:  fallas {CI['numero_fallas'][0]:+.2f}  ·  segundos {CI['segundos'][0]:+.2f}", 960, 760, tam=44, c="violeta", en="b"))
v.b("Solo los reclamos son levemente negativos: menos cero coma doce con segundos. [[a]]La interpretación «más fallas, menos uso» no se sostiene. [[b]]La lección: una interpretación bonita se comprueba con components_ y con las correlaciones, antes de llevarla al negocio.",
    T(f"reclamos ↔ segundos: {PA['reclamos|segundos']:+.2f}", 960, 860, tam=40, c="gris", w="sb"), U("q32", tachado=1.0, en="a", dur=0.6),
    P("comprobar antes de contar la historia", 960, 970, "verde", 44, en="b"))
v.b("Y otra alerta sobre la limpieza: [[a]]el código de la clase solo corrige la edad y los faltantes. [[b]]Quedan valores imposibles: segundos hasta menos catorce mil seiscientos cuarenta y cinco, y llamadas hasta menos ciento setenta y seis. Hay que limpiarlos antes de PCA.",
    CL(), P("alerta: limpieza incompleta", 960, 220, "ambar", 48, ent="subir"), P("la clase corrige solo edad y faltantes", 960, 380, "gris", 44, en="a"),
    P(f"segundos mínimo: −14.645", 640, 560, "rojo", 46, en="b"), P(f"frec_llamadas mínimo: −176", 1300, 560, "rojo", 46, en="b", t=0.3),
    P("limpiar negativos imposibles antes de PCA", 960, 740, "verde", 48, en="b", t=0.8))

# ---- 8. Error típico y piensa
v.b("El error típico de esta parada: [[a]]ajustar el escalador, o el PCA, con todas las filas, y recién después dividir. [[b]]Primero se divide. Después, todo se ajusta con el entrenamiento, dentro de un Pipeline.",
    CL(), T("fit con todas las filas  →  dividir", 960, 330, tam=64, c="rojo", en="a"), K(1600, 330, "cruz", 70, en="a", t=0.5),
    T("dividir  →  fit con entrenamiento (Pipeline)", 960, 560, tam=60, c="verde", en="b"), K(1660, 560, "check", 70, en="b", t=0.5), sec="8. Error típico y piensa un segundo")
v.b("Piensa un segundo: ¿fuga o no? [[a]]Uno: imputar la mediana de la edad con todas las filas, antes de dividir. [[b]]Dos: calcular SMS por antigüedad antes de dividir.",
    CL(), T("¿fuga o no?", 960, 220, tam=60, c="tinta", ent="subir"),
    Bc(960, 420, 1500, 130, "blanco", "violeta", 4, 24, en="a", ent="subir"), T("1 · imputar la mediana con todas las filas", 960, 420, tam=44, c="tinta", w="sb", en="a", t=0.1),
    Bc(960, 600, 1500, 130, "blanco", "violeta", 4, 24, en="b", ent="subir"), T("2 · calcular sms_por_antiguedad antes de dividir", 960, 600, tam=44, c="tinta", w="sb", en="b", t=0.1),
    pausa=0.8)
v.b("[[a]]La primera es fuga: la mediana se aprendió mirando también la prueba. [[b]]La segunda no: un cociente se calcula fila por fila, y no aprende nada del conjunto.",
    P("fuga: aprende la mediana con la prueba", 960, 780, "rojo", 44, en="a"), K(1760, 420, "cruz", 60, en="a"),
    P("no: es fila a fila, no aprende nada", 960, 900, "verde", 44, en="b"), K(1760, 600, "check", 60, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]todo lo que aprende de los datos se ajusta con entrenamiento, dentro de un Pipeline. [[b]]En la base real, siete componentes conservan el ochenta coma cuatro por ciento, y cuestan un poco de F1. [[c]]Y toda interpretación se comprueba con los datos.",
    CL(), P("todo lo que aprende → fit con entrenamiento, en un Pipeline", 960, 300, "verde", 46, en="a"), P("base real: 7 componentes = 80.4 % · F1 0.574 → 0.548", 960, 480, "violeta", 46, en="b"),
    P("comprobar cada interpretación con los datos", 960, 660, "ambar", 48, en="c"), sec="9. Para llevar")
v.b("Y ahora sí: [[a]]el repaso final de la clase.",
    CL(), siguiente(10, "Repaso final"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
