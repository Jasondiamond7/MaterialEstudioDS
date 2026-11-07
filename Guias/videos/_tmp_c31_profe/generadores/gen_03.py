import sys, json, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c31_03_clasificacion.json"
D = json.load(open(r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe\datos31.json", encoding="utf-8"))
v = V("Clasificación: la regresión logística")

v.portada("Clasificación: la regresión logística", "Del puntaje a la probabilidad, y de la probabilidad a la decisión",
          "Clasificación y regresión logística. Cómo un modelo decide si un cliente cancela o se queda.")

# ---- 1. Hook
v.b("Imagina que tienes que adivinar quién va a cancelar su suscripción. [[a]]No quieres un «tal vez». [[b]]Quieres un sí o un no. Pero antes de decidir, el modelo calcula algo más fino.",
    CL(), P("¿cancela?", 960, 330, "violeta", 90, en="a"), P("sí", 700, 600, "rojo", 80, en="b"), P("no", 1220, 600, "verde", 80, en="b", t=0.3),
    T("antes de decidir, calcula algo más fino", 960, 850, tam=48, c="gris", w="sb", en="b", t=0.8), sec="1. Un puntaje que se vuelve probabilidad")
v.b("Calcula una probabilidad. [[a]]Por ejemplo, este cliente tiene setenta por ciento de probabilidad de cancelar. [[b]]Y luego una regla simple, el umbral, convierte esa probabilidad en una decisión.",
    CL(), Bc(560, 480, 620, 360, "blanco", "violeta", 5, 30, sombra=True, ent="subir", t=0.2), T("probabilidad de cancelar", 560, 340, tam=40, c="gris", w="sb", t=0.3),
    N(70, 560, 480, "{:.0f}", 150, "violeta", suf=" %", en="a", dur=1.0),
    L(900, 480, 1080, 480, "gris", 8, True, en="b"), Bc(1400, 480, 520, 360, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"),
    T("umbral 0.5", 1400, 380, tam=48, c="teal", en="b", t=0.2), P("decisión: cancela", 1400, 540, "rojo", 44, en="b", t=0.5))
v.b("Como en las clases diecisiete a veintiuno, aquí la regresión logística aparece como un modelo de clasificación. [[a]]La pregunta es cómo pasa de las variables de un cliente a esa probabilidad.",
    CL(), P("variables del cliente", 480, 480, "teal", 48, ent="subir", t=0.2), L(760, 480, 1000, 480, "gris", 8, True, en="a"), P("probabilidad", 1380, 480, "violeta", 48, en="a", t=0.3), T("?", 1000, 400, tam=90, c="ambar", en="a", t=0.2))

# ---- 2. Puntaje
v.b("Primero calcula un puntaje, que se llama logit. [[a]]Es una suma ponderada: un punto de partida, [[b]]más cada variable multiplicada por su coeficiente.",
    CL(),
    T("z  =  b₀  +  b₁·x₁  +  b₂·x₂  +  …", 960, 330, tam=70, c="tinta", id="fz", t=0.3),
    P("punto de partida", 420, 520, "gris", 40, en="a"), L(420, 475, 520, 375, "gris", 4, True, en="a", t=0.2),
    P("cada variable × su coeficiente", 1300, 520, "violeta", 40, en="b"), L(1300, 475, 1260, 375, "violeta", 4, True, en="b", t=0.2), sec=None)
v.b("Es como un termómetro de riesgo. [[a]]Cada queja de soporte sube la temperatura. [[b]]Cada mes de antigüedad la baja.",
    CL(), Bc(960, 520, 300, 560, "blanco", "grisclaro", 5, 40, ent="subir", t=0.2),
    Bx(910, 620, 1010, 780, "rojoclaro", "rojo", 3, 20, id="term", t=0.4), T("riesgo", 960, 195, tam=44, c="gris", w="sb", t=0.3),
    P("queja de soporte", 1500, 400, "rojo", 42, en="a"), T("↑", 1250, 400, tam=80, c="rojo", en="a", t=0.2), U("term", y0=470, en="a", dur=1.0),
    P("mes de antigüedad", 1500, 640, "verde", 42, en="b"), T("↓", 1250, 640, tam=80, c="verde", en="b", t=0.2), U("term", y0=660, en="b", dur=1.0))
v.b("Pero un puntaje puede valer cualquier cosa: [[a]]menos cinco, cero, o diez. [[b]]Una probabilidad, en cambio, tiene que estar siempre entre cero y uno. Necesitamos un traductor.",
    CL(), T("puntaje:  −5   0   10", 960, 350, tam=80, c="tinta", en="a"), T("probabilidad: entre 0 y 1", 960, 520, tam=80, c="violeta", en="b"),
    P("necesitamos un traductor", 960, 800, "ambar", 56, en="b", t=0.8))

# ---- 3. Sigmoide
zz = [x / 10 for x in range(-60, 61, 5)]
pp = [1 / (1 + math.exp(-z)) for z in zz]
GX0, GY0, GX1, GY1 = 240, 200, 1280, 820


def fx(z):
    return GX0 + (z + 6) / 12 * (GX1 - GX0)


def fy(p):
    return GY1 - p * (GY1 - GY0)


def sig_graf(en=None, rev=0.0):
    return G(GX0, GY0, GX1, GY1, -6, 6, 0, 1, [dict(xs=zz, ys=pp, color="violeta", grosor=8, revela=rev, lapiz=True)],
             xticks=[-4, -2, 0, 2, 4], yticks=[0, 0.5, 1], yfmt="{:g}", xlabel="puntaje z (logit)", id="sg", en=en)


v.b("Ese traductor es la función sigmoide. [[a]]Toma cualquier puntaje, y lo aprieta entre cero y uno. [[b]]La fórmula es uno, dividido por uno más e elevado a menos z.",
    CL(), sig_graf("a"), U("sg", **{"series.0.revela": 1.0}, en="a", dur=2.5),
    g("fm", T("p = 1 / (1 + e^(−z))", 1600, 420, tam=64, c="violeta", w="b", en="b")), sec="2. La sigmoide")
v.b("Mira lo que hace. [[a]]Con puntaje cero, la probabilidad es justo cero coma cinco. [[b]]Con un puntaje muy alto, se acerca a uno. [[c]]Y con uno muy negativo, se acerca a cero.",
    L(fx(0), fy(0), fx(0), fy(0.5), "ambar", 5, guion=True, en="a", id="v0"), L(fx(0), fy(0.5), fx(-6), fy(0.5), "ambar", 5, guion=True, en="a", t=0.4, id="h0"),
    g("sp", P("z = 0  →  p = 0.5", 1600, 560, "ambar", 40, en="a", mw=560)),
    R(fx(5), fy(0.99), 150, 110, "verde", 30, id="ra", en="b", ent="fade"), g("sp", P("z alto  →  p cerca de 1", 1600, 660, "verde", 38, en="b", mw=560)),
    X("ra", en="c", dur=0.3), R(fx(-5), fy(0.01), 150, 110, "rojo", 30, id="rb", en="c", ent="fade"), g("sp", P("z muy negativo  →  p cerca de 0", 1600, 760, "rojo", 38, en="c", mw=560)))
v.b("Y esto conecta con el umbral. [[a]]Con umbral cero coma cinco, el modelo dice «cancela» justo cuando el puntaje es mayor que cero.",
    X("rb", dur=0.3), R(fx(0), fy(0.5), 120, 120, "violeta", 30, id="rz", en="a", ent="fade"), XG("sp", dur=0.3), XG("fm", dur=0.3),
    P("p > 0.5  ⇔  z > 0", 1600, 480, "violeta", 54, en="a"), P("cancela", 1000, 330, "rojo", 40, en="a", t=0.3), P("se queda", 480, 700, "verde", 40, en="a", t=0.3))

# ---- 4. Ejemplo resuelto
v.b("Hagamos una cuenta completa, con la que usa el apunte. [[a]]El puntaje es menos uno, más uno coma cinco por las quejas, menos cero coma dos por el uso.",
    CL(),
    T("z = −1.0 + 1.5·quejas − 0.2·uso", 960, 250, tam=64, c="tinta", ent="subir", t=0.2, id="fe"), sec="3. Un ejemplo resuelto a mano")
v.b("Cliente uno: dos quejas y seis usos al mes. [[a]]Menos uno, más uno coma cinco por dos, menos cero coma dos por seis. [[b]]Menos uno, más tres, menos uno coma dos: da cero coma ocho.",
    Bc(560, 560, 760, 480, "blanco", "violeta", 5, 30, sombra=True, id="c1", ent="subir", t=0.2), T("Cliente 1", 560, 380, tam=46, c="violeta", t=0.3), T("2 quejas · 6 usos", 560, 450, tam=36, c="gris", w="sb", t=0.4),
    T("−1.0 + 1.5·2 − 0.2·6", 560, 560, tam=48, c="tinta", en="a"), T("−1.0 + 3.0 − 1.2", 560, 650, tam=48, c="tinta", en="b"), T("z = 0.8", 560, 750, tam=70, c="violeta", en="b", t=0.5))
v.b("Ahora la sigmoide. [[a]]Uno, dividido por uno más e elevado a menos cero coma ocho. [[b]]La exponencial vale 0.449, así que queda uno entre 1.449. [[c]]La probabilidad de cancelar es cero coma sesenta y nueve.",
    T("p = 1 / (1 + e^(−0.8))", 1400, 400, tam=48, c="tinta", en="a"), T("= 1 / (1 + 0.449)", 1400, 500, tam=48, c="tinta", en="b"), T("p = 0.690", 1400, 620, tam=80, c="violeta", en="c"),
    P("69 % de probabilidad", 1400, 760, "violeta", 40, en="c", t=0.5))
v.b("Cliente dos: cero quejas y ocho usos. [[a]]Menos uno, más cero, menos uno coma seis: el puntaje es menos dos coma seis. [[b]]Y la probabilidad baja a cero coma cero sesenta y nueve.",
    CL(mant=["fe"]),
    Bc(560, 580, 760, 480, "blanco", "teal", 5, 30, sombra=True, ent="subir", t=0.2), T("Cliente 2", 560, 400, tam=46, c="teal", t=0.3), T("0 quejas · 8 usos", 560, 470, tam=36, c="gris", w="sb", t=0.4),
    T("−1.0 + 0 − 0.2·8", 560, 580, tam=48, c="tinta", en="a"), T("z = −2.6", 560, 690, tam=70, c="teal", en="a", t=0.4),
    T("p = 1 / (1 + e^2.6)", 1400, 430, tam=48, c="tinta", en="b"), T("p = 0.069", 1400, 570, tam=90, c="teal", en="b", t=0.4), P("casi no cancela", 1400, 720, "teal", 40, en="b", t=0.8))
v.b("¿Y qué pasa si el cliente tiene una queja más? [[a]]Cada queja multiplica las odds, que son la probabilidad dividida por su complemento, [[b]]por e elevado a uno coma cinco. Eso es cuatro coma cuarenta y ocho veces.",
    CL(), T("odds = p / (1 − p)", 960, 300, tam=60, c="tinta", en="a"), T("+1 queja", 560, 560, tam=64, c="rojo", en="b"), L(760, 560, 1000, 560, "gris", 8, True, en="b", t=0.2),
    T("odds × e^1.5", 1360, 500, tam=60, c="violeta", en="b", t=0.3), N(4.48, 1360, 640, "{:.2f}", 110, "violeta", pre="× ", en="b", t=0.5, dur=0.9))
v.b("Piensa un segundo: [[a]]el cliente uno tiene probabilidad cero coma sesenta y nueve. ¿Cancela con umbral cero coma cinco? ¿Y con umbral cero coma siete?",
    CL(), N(0.69, 960, 330, "{:.2f}", 150, "violeta", en="a", dur=0.8), T("umbral 0.5  →  ?", 700, 640, tam=60, c="teal", en="a", t=0.5), T("umbral 0.7  →  ?", 1280, 640, tam=60, c="ambar", en="a", t=0.8), pausa=0.6)
v.b("Con cero coma cinco, sí cancela: [[a]]cero coma sesenta y nueve pasa el umbral. [[b]]Con cero coma siete, ya no: se queda por muy poco.",
    P("0.5  →  cancela", 700, 800, "rojo", 54, en="a"), K(700, 900, "check", 60, c="rojo", en="a", t=0.2), P("0.7  →  se queda", 1280, 800, "verde", 54, en="b"), K(1280, 900, "cruz", 60, c="verde", en="b", t=0.2))

# ---- 5. Los datos de la clase
v.b("Ahora, el caso de la clase: Club Activo, con ochocientos clientes. [[a]]Se dividen en entrenamiento y prueba, de forma estratificada, como en la clase treinta. [[b]]Quinientos sesenta para aprender, y doscientos cuarenta para probar.",
    CL(),
    Pt([(560, [160, 380, 1180, 700], "azul"), (240, [1260, 380, 1760, 700], "ambar")], 7, id="pdiv", en="a", dur=0.8),
    T("entrenamiento", 670, 780, tam=46, c="azul", en="a", t=0.3), N(560, 670, 860, "{:.0f}", 80, "azul", en="b", dur=0.8),
    T("prueba", 1510, 780, tam=46, c="ambar", en="a", t=0.3), N(240, 1510, 860, "{:.0f}", 80, "ambar", en="b", dur=0.8), sec="4. El caso Club Activo")
v.b("Todo va dentro de un Pipeline: [[a]]primero se imputan los faltantes, [[b]]luego se estandarizan las variables, [[c]]y al final se ajusta la regresión logística. [[d]]Y el fit se hace solo con el entrenamiento.",
    CL(),
    *[[Bc(x, 480, 380, 200, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 480, tam=44, c=c, mw=320, en=k, t=0.1)]
      for x, c, n, k in ((320, "teal", "imputar", "a"), (800, "azul", "estandarizar", "b"), (1280, "violeta", "regresión logística", "c"))],
    [L(520, 480, 600, 480, "gris", 6, True, en="b", t=0.1), L(1000, 480, 1080, 480, "gris", 6, True, en="c", t=0.1)],
    R(800, 480, 1400, 290, "teal", 40, id="rp", en="c", t=0.5, ent="fade"), P("Pipeline", 800, 290, "teal", 50, en="c", t=0.5), P("fit solo con entrenamiento", 1640, 480, "ambar", 40, en="d", mw=420))
v.b("Con las variables ya estandarizadas, los coeficientes se pueden comparar. [[a]]Las quejas empujan el riesgo hacia arriba: más uno coma cero ocho. [[b]]La antigüedad lo baja: menos cero coma ochenta. [[c]]El uso lo baja un poco: menos cero coma veintidós. [[d]]Y la edad casi no pesa.",
    CL(),
    *[[T(n, 740, y, tam=46, c="tinta", w="sb", anc="rm", en=k),
       (Bx(1100, y - 27, 1100 + val * 400, y + 27, c, None, 0, 8, en=k, grow="x", t=0.1) if val > 0 else Bx(1100 + val * 400, y - 27, 1100, y + 27, c, None, 0, 8, en=k, ent="fade", t=0.1)),
       T(f"{val:+.2f}".replace("-", "−"), 1100 + val * 400 + (90 if val > 0 else -90), y, tam=40, c=c, en=k, t=0.5)]
      for n, y, val, c, k in (("quejas", 380, 1.08, "rojo", "a"), ("antigüedad", 500, -0.80, "verde", "b"), ("uso", 620, -0.22, "verde", "c"), ("edad", 740, 0.08, "gris", "d"))],
    L(1100, 320, 1100, 800, "tinta", 4, en="a"), T("sube el riesgo →", 1600, 320, tam=34, c="gris", w="sb", en="a"), T("← baja el riesgo", 600, 320, tam=34, c="gris", w="sb", en="a"), T("intercepto: −0.39", 960, 900, tam=36, c="gris", w="sb", en="d"))
v.b("Con ese modelo, evaluamos en la prueba, que el modelo nunca vio. [[a]]Y ahí viene una decisión importante: el peso de las clases.",
    CL(), P("evaluar en datos nunca vistos", 960, 400, "azul", 58, ent="subir", t=0.2), P("el peso de las clases", 960, 640, "ambar", 66, en="a"))

# ---- 6. class_weight
v.b("Solo uno de cada cinco clientes cancela. [[a]]Si el modelo cuenta todos los errores igual, aprende a decir que casi nadie cancela, [[b]]porque así se equivoca poco.",
    CL(), Pt([(79, [300, 360, 1100, 640], "#94A3B8"), (21, [1180, 360, 1500, 640], "rojo")], 12, id="p100", ent="fade", t=0.2), P("1 de cada 5 cancela", 960, 760, "rojo", 52, en="a", t=0.0), P("aprende a decir «nadie cancela»", 960, 900, "ambar", 48, en="b"), sec="5. El peso de las clases")
v.b("La opción balanced le da más peso a la clase rara. [[a]]Pondera cada clase al revés de su frecuencia: cero coma sesenta y tres para quien se queda, [[b]]y dos coma treinta y nueve para quien cancela.",
    CL(), Bc(560, 480, 640, 400, "blanco", "verde", 5, 30, sombra=True, ent="subir", t=0.2), T("se queda", 560, 360, tam=48, c="verde", t=0.3), N(0.63, 560, 500, "{:.2f}", 140, "verde", en="a", dur=0.9),
    Bc(1360, 480, 640, 400, "blanco", "rojo", 5, 30, sombra=True, en="b", ent="subir"), T("cancela", 1360, 360, tam=48, c="rojo", en="b", t=0.1), N(2.39, 1360, 500, "{:.2f}", 140, "rojo", en="b", dur=0.9, t=0.2),
    P("class_weight = 'balanced'", 960, 860, "violeta", 46, t=0.5))
v.b("Cada cliente que cancela y se pasa de largo pesa casi cuatro veces más que uno que se queda. [[a]]El modelo se vuelve alarmista. [[b]]Gana recall, pero pierde precisión.",
    P("modelo alarmista", 960, 820, "ambar", 56, en="a"), P("gana recall · pierde precisión", 960, 940, "violeta", 50, en="b"))
v.b("Mira la diferencia. [[a]]Sin ponderar, la precisión es sesenta y nueve por ciento, pero el recall solo dieciocho. [[b]]Con balanced, la precisión baja a treinta y nueve, y el recall sube a cincuenta y dos.",
    CL(),
    Bc(560, 500, 700, 520, "blanco", "grisclaro", 5, 30, sombra=True, ent="subir", t=0.2), T("sin ponderar", 560, 290, tam=46, c="gris"), T("precisión", 560, 380, tam=34, c="gris", w="sb"), N(69.2, 560, 450, "{:.1f}", 90, "azul", suf=" %", en="a", dur=0.9),
    T("recall", 560, 560, tam=34, c="gris", w="sb"), N(18.0, 560, 630, "{:.1f}", 90, "violeta", suf=" %", en="a", t=0.3, dur=0.9),
    Bc(1360, 500, 700, 520, "violetaclaro", "violeta", 5, 30, sombra=True, en="b", ent="subir"), T("balanced", 1360, 290, tam=46, c="violeta", en="b", t=0.1), T("precisión", 1360, 380, tam=34, c="gris", w="sb", en="b", t=0.2),
    N(39.4, 1360, 450, "{:.1f}", 90, "azul", suf=" %", en="b", t=0.3, dur=0.9), T("recall", 1360, 560, tam=34, c="gris", w="sb", en="b", t=0.4), N(52.0, 1360, 630, "{:.1f}", 90, "violeta", suf=" %", en="b", t=0.5, dur=0.9),
    P("una decisión de modelado, no un arreglo gratuito", 960, 960, "ambar", 38))

# ---- 7. El umbral
def pts_serie(y, color):
    par = sorted((z, p) for z, p, c in zip(D["sig"]["z"], D["sig"]["p"], D["sig"]["y"]) if c == y)
    xs = [a for a, _ in par]
    ys = [b for _, b in par]
    return dict(xs=xs, ys=ys, color=color, linea=False, marcadores=9)


zt = lambda t: math.log(t / (1 - t))  # noqa: E731
v.b("Hay otra decisión de negocio: el umbral. [[a]]Estos puntos son una muestra de los clientes de prueba, ubicados sobre la sigmoide. [[b]]Los grises se quedaron, los ámbar cancelaron. El color no se usó para ubicarlos.",
    CL(),
    G(GX0, GY0, GX1, GY1, -6, 6, 0, 1, [dict(xs=zz, ys=pp, color="violeta", grosor=6), pts_serie(0, "#94A3B8"), pts_serie(1, "ambar")],
      xticks=[-4, -2, 0, 2, 4], yticks=[0, 0.5, 1], yfmt="{:g}", xlabel="puntaje z (logit)", id="sg2", t=0.2), T("probabilidad de cancelar", GX0, 150, tam=34, c="tinta", w="sb", anc="lm", t=0.3),
    P("se quedaron", 1620, 360, "gris", 38, en="a"), P("cancelaron", 1620, 460, "ambar", 38, en="b"), sec="6. El umbral es una decisión de negocio")
UMB = D["umbral"]
v.b("Con umbral cero coma cinco, el modelo marca a sesenta y seis clientes como «cancela». [[a]]Detecta a la mitad, el cincuenta y dos por ciento. [[b]]Pero solo cuatro de cada diez alertas son aciertos: precisión treinta y nueve.",
    L(fx(0), GY0, fx(0), GY1, "tinta", 5, guion=True, id="um", t=0.2), P("umbral 0.5", 1620, 600, "teal", 40, id="ut", t=0.2), N(66, 1620, 700, "{:.0f}", 90, "tinta", suf=" alertas", t=0.3, dur=0.8, id="nal"),
    P("recall 52 %", 1620, 820, "violeta", 40, id="pr1", en="a"), P("precisión 39 %", 1620, 910, "azul", 40, id="pr2", en="b"))
v.b("Si bajas el umbral a cero coma tres, el modelo avisa más seguido. [[a]]Detecta al ochenta y dos por ciento, [[b]]pero la mayoría de las alertas son falsas: precisión veintisiete por ciento.",
    U("um", x0=fx(zt(0.3)), x1=fx(zt(0.3)), dur=1.0), U("nal", valor=153, dur=0.9), U("ut", texto="umbral 0.3"), U("pr1", texto="recall 82 %", en="a"), U("pr2", texto="precisión 27 %", en="b"))
v.b("Y si lo subes a cero coma siete, el modelo avisa poco. [[a]]La precisión sube a cincuenta por ciento, [[b]]pero el recall cae a veinte. Se escapan cuatro de cada cinco.",
    U("um", x0=fx(zt(0.7)), x1=fx(zt(0.7)), dur=1.0), U("nal", valor=20, dur=0.9), U("ut", texto="umbral 0.7"), U("pr2", texto="precisión 50 %", en="a"), U("pr1", texto="recall 20 %", en="b"))
v.b("Fíjate: el umbral no cambia el modelo. [[a]]Cambia dónde pones el corte. Y dónde cortar depende de cuánto cuesta cada error.",
    P("sube el umbral: más precisión, menos recall", 960, 960, "ambar", 40, en="a", mw=1400))
v.b("Piensa un segundo: si perder un cliente cuesta mucho más que molestar con una oferta de más, [[a]]¿subirías o bajarías el umbral?",
    CL(), T("perder un cliente  >>  molestar con una oferta", 960, 400, tam=60, c="tinta", en="a"), T("¿umbral más alto o más bajo?", 960, 600, tam=72, c="violeta", en="a", t=0.5), pausa=0.6)
v.b("Lo bajarías. [[a]]Prefieres algunas falsas alarmas antes que dejar escapar clientes. Ganas recall, y aceptas menos precisión.",
    CL(), P("bajar el umbral", 960, 400, "verde", 76, ent="subir"), T("↑ recall    ↓ precisión", 960, 620, tam=64, c="tinta", w="sb", en="a"))

# ---- 8. Resultado y error típico
v.b("Resumamos el resultado, con el umbral de cero coma cinco. [[a]]Accuracy, setenta y tres coma tres por ciento. [[b]]Precisión, treinta y nueve coma cuatro. [[c]]Recall, cincuenta y dos. [[d]]Y F1, cuarenta y cuatro coma ocho.",
    CL(),
    *[[Bc(x, 480, 380, 290, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 380, tam=40, c=c, en=k, t=0.1), N(val, x, 520, "{:.1f}", 90, c, suf=" %", en=k, t=0.2, dur=0.9)]
      for x, c, n, val, k in ((260, "violeta", "accuracy", 73.3, "a"), (740, "azul", "precisión", 39.4, "b"), (1220, "ambar", "recall", 52.0, "c"), (1700, "verde", "F1", 44.8, "d"))],
    sec="7. Resultado, error típico y para llevar")
v.b("Y el área bajo la curva, el A U C, es cero coma seiscientos ochenta y cinco. [[a]]Un modelo al azar da cero coma cinco. [[b]]Así que es un modelo modesto: mejor que el azar, pero lejos de excelente.",
    CL(), T("AUC", 960, 330, tam=60, c="gris", w="sb"), N(0.685, 960, 470, "{:.3f}", 190, "violeta", dur=1.0, t=0.2), P("azar = 0.5", 560, 760, "gris", 48, en="a"), P("este modelo = 0.685", 1360, 760, "violeta", 48, en="b"),
    P("modesto: ni moneda ni excelente", 960, 900, "ambar", 46, en="b", t=0.6))
v.b("Este es el error típico. En clase se dijo que una precisión de cincuenta y uno por ciento es como lanzar una moneda. [[a]]Pero la moneda no es el cincuenta por ciento. [[b]]Un clasificador al azar tiene una precisión igual a la prevalencia, que aquí es veinte coma ocho por ciento. [[c]]Y este logra treinta y nueve.",
    CL(), P("«es como lanzar una moneda»", 960, 280, "rojo", 54, ent="subir", t=0.2), K(1560, 280, "cruz", 80, en="a"),
    Bc(560, 620, 640, 340, "blanco", "gris", 5, 30, sombra=True, en="b", ent="subir"), T("al azar", 560, 500, tam=46, c="gris", en="b", t=0.1), N(20.8, 560, 630, "{:.1f}", 110, "gris", suf=" %", en="b", t=0.2, dur=0.9),
    Bc(1360, 620, 640, 340, "blanco", "azul", 5, 30, sombra=True, en="c", ent="subir"), T("este modelo", 1360, 500, tam=46, c="azul", en="c", t=0.1), N(39.4, 1360, 630, "{:.1f}", 110, "azul", suf=" %", en="c", t=0.2, dur=0.9))
v.b("Y un detalle de la frontera: con solo dos variables, [[a]]antigüedad y quejas, la frontera de decisión es una recta, [[b]]porque el puntaje es una suma lineal.",
    CL(), Bc(960, 520, 900, 540, "blanco", "grisclaro", 5, 24, ent="subir", t=0.2), Pt([(40, [560, 330, 920, 640], "#94A3B8"), (14, [1040, 330, 1300, 640], "rojo")], 9, id="pf", en="a", t=0.1),
    L(1000, 290, 965, 760, "violeta", 8, en="b"), T("antigüedad  →", 960, 790, tam=34, c="gris", w="sb", en="a"), P("frontera: una recta", 1500, 520, "violeta", 44, en="b", mw=400))
v.b("Para llevarte: [[a]]la logística calcula un puntaje, la sigmoide lo vuelve probabilidad, y el umbral decide. [[b]]Balanced gana recall y pierde precisión. [[c]]Y el umbral se elige según el costo de cada error.",
    CL(), P("puntaje  →  probabilidad  →  decisión", 960, 330, "violeta", 54, en="a"), P("balanced: ↑ recall, ↓ precisión", 960, 540, "ambar", 54, en="b"), P("el umbral lo decide el costo del error", 960, 750, "teal", 54, en="c"))
v.b("Vamos con la tercera parada: [[a]]la regresión, para predecir un número.",
    CL(), Bc(960, 480, 620, 260, "azulclaro", "azul", 6, 40, id="p1", ent="pop", en="a"),
    T("3", 720, 480, tam=120, c="azul", en="a", t=0.1), T("Regresión: predecir un número", 1040, 480, tam=44, c="tinta", mw=420, en="a", t=0.15), sec="8. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
