import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_05_pca.json")
v = V("PCA: la idea y una matriz 2×2 a mano")
GEO = D["geo"]
R_ = GEO["r"]

v.portada("PCA: la idea", "La dirección donde los datos varían más, y una matriz de 2×2 resuelta a mano",
          "Cuarta parada: el análisis de componentes principales, PCA. Primero la intuición, y después un ejemplo resuelto a mano.")

# ---- 1. Hook
v.b("Imagina que tienes que sacarle una sola foto a una banca de plaza. [[a]]Si la fotografías de frente, de lado, se ve como una línea. [[b]]Si la fotografías en diagonal, se ve su largo, su forma. Eliges el ángulo que muestra más.",
    CL(), P("una sola foto", 960, 220, "violeta", 56, ent="subir"),
    Bc(560, 540, 120, 120, "blanco", "gris", 5, 20, en="a", ent="pop"), T("de lado", 560, 680, tam=40, c="gris", w="sb", en="a", t=0.2), K(560, 780, "cruz", 60, en="a", t=0.5),
    Bc(1360, 540, 560, 120, "violetaclaro", "violeta", 5, 20, en="b", ent="pop"), T("en diagonal", 1360, 680, tam=40, c="violeta", w="sb", en="b", t=0.2), K(1360, 780, "check", 60, en="b", t=0.5),
    ilustrativo(990, texto="analogía"), sec="1. El mejor ángulo")
v.b("PCA hace eso con tus datos. [[a]]Busca la dirección en que los datos varían más. [[b]]Esa dirección es el primer componente, PC1. Y como la profesora dijo: cada componente es una receta, una mezcla de las variables originales.",
    CL(), P("PCA busca la dirección de máxima varianza", 960, 300, "azul", 54, en="a"), P("esa dirección = PC1", 960, 460, "violeta", 54, en="b"),
    P("cada componente es una «receta» de las variables", 960, 620, "teal", 48, en="b", t=0.8))

# ---- 2. Paso a paso con datos
G0 = dict(x0=300, y0=200, x1=980, y1=880, xmin=-3.5, xmax=3.5, ymin=-3.5, ymax=3.5)


def gr(id_, t=0.1, **kw):
    return G(G0["x0"], G0["y0"], G0["x1"], G0["y1"], G0["xmin"], G0["xmax"], G0["ymin"], G0["ymax"],
             [dict(xs=GEO["xs"], ys=GEO["ys"], color="#94A3B8", linea=False, marcadores=7)],
             xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-3, -2, -1, 0, 1, 2, 3], yfmt="{:.0f}", xlabel="segundos (estandarizado)", id=id_, t=t, **kw)


v.b("Veamos con datos de la clase. [[a]]Dos variables estandarizadas: los segundos, y el valor del cliente. [[b]]Su correlación es cero coma ochenta y cuatro: la nube de puntos sube en diagonal.",
    CL(), gr("gp", t=0.0), T("clv (estandarizado)", 300, 150, tam=32, c="tinta", w="sb", anc="lm", id="ylab"),
    g("der", P("2 variables estandarizadas", 1440, 360, "gris", 42, en="a"), T(f"r = {R_:.2f}", 1440, 520, tam=90, c="violeta", en="b"),
      T("muestra de 220 de los 1.500 clientes", 1440, 640, tam=30, c="gris", w="sb", en="b", t=0.4)), sec="2. La dirección de máxima varianza")


def lin(th):
    c_, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
    return dict(x0=-3.3 * c_, y0=-3.3 * s_, x1=3.3 * c_, y1=3.3 * s_)


def var(th):
    return 1 + R_ * math.sin(math.radians(2 * th))


ANG = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
v.b("Ahora probamos direcciones. [[a]]Una línea horizontal: si proyectamos los puntos sobre ella, la varianza es uno. [[b]]La vamos girando, [[c]]y la varianza crece, hasta un máximo en la diagonal. [[d]]Si seguimos girando, baja hasta casi cero.",
    XG("der", dur=0.3), L(**lin(0), c="ambar", g=6, id="dir", ref="gp", en="a"),
    g("der", T("varianza proyectada", 1440, 330, tam=36, c="gris", w="sb", en="a")), N(var(0), 1440, 450, "{:.2f}", 110, "ambar", id="nv", en="a", dur=0.4),
    *[U("dir", **lin(th), en="b", t=0.3 * i, dur=0.3, ease="lineal") for i, th in enumerate(ANG[1:4])],
    *[U("nv", valor=var(th), en="b", t=0.3 * i, dur=0.3, ease="lineal") for i, th in enumerate(ANG[1:4])],
    g("der", P("máximo: 1.84", 1440, 600, "violeta", 46, en="c")),
    *[U("dir", **lin(th), en="d", t=0.3 * i, dur=0.3, ease="lineal") for i, th in enumerate(ANG[4:])],
    *[U("nv", valor=var(th), en="d", t=0.3 * i, dur=0.3, ease="lineal") for i, th in enumerate(ANG[4:])],
    g("der", P("mínimo: 0.16", 1440, 720, "teal", 46, en="d", t=1.8)))
v.b("[[a]]La dirección de máxima varianza es PC1: la diagonal, uno coma ochenta y cuatro. [[b]]Y la siguiente, perpendicular a la primera, es PC2: cero coma dieciséis.",
    X("dir", "nv", dur=0.3), XG("der", dur=0.3),
    L(**lin(45), c="violeta", g=8, punta=True, id="pc1", ref="gp", en="a"), T("PC1", 3.0, 3.25, tam=40, c="violeta", ref="gp", en="a", t=0.3),
    L(**lin(135), c="teal", g=7, punta=True, id="pc2", ref="gp", en="b"), T("PC2", -2.9, 2.95, tam=40, c="teal", ref="gp", en="b", t=0.3))
v.b("Esos dos números se llaman valores propios, o lambda. [[a]]Lambda uno, uno coma ochenta y cuatro, es la varianza de PC1. [[b]]Lambda dos, cero coma dieciséis, la de PC2. [[c]]Suman dos, que es el número de variables estandarizadas.",
    Bc(1440, 400, 600, 150, "blanco", "violeta", 5, 26, sombra=True, en="a", ent="subir"), T(f"λ₁ = {GEO['eig'][0]:.2f}", 1440, 400, tam=60, c="violeta", en="a", t=0.1),
    Bc(1440, 580, 600, 150, "blanco", "teal", 5, 26, sombra=True, en="b", ent="subir"), T(f"λ₂ = {GEO['eig'][1]:.2f}", 1440, 580, tam=60, c="teal", en="b", t=0.1),
    P("λ₁ + λ₂ = 2 = número de variables", 1440, 760, "gris", 38, en="c"))
v.b("¿Cuánto conserva PC1? [[a]]Uno coma ochenta y cuatro, dividido en dos: noventa y uno coma ocho por ciento de la varianza. [[b]]Con una sola columna nueva, guardamos casi todo lo que tenían las dos.",
    CL(mant=["gp", "pc1", "ylab"]),
    T("1.84 / 2 =", 1300, 450, tam=60, c="tinta", en="a"), N(GEO["var"][0], 1600, 450, "{:.1f}", 90, "violeta", suf=" %", en="a", t=0.3, dur=0.8),
    P("una columna conserva casi todo", 1440, 640, "violeta", 44, en="b"))

# proyecciones de unos puntos sobre PC1
SEL = [i for i in range(0, 220, 22)][:9]
proy = []
for i in SEL:
    x, y = GEO["xs"][i], GEO["ys"][i]
    s = (x + y) / 2
    proy += [L(x, y, s, s, "ambar", 3, guion=True, ref="gp", en="a", t=0.08 * k) for k in [SEL.index(i)]]
    proy += [Bc(s, s, 18, 18, "violeta", None, 0, 9, ref="gp", en="b", t=0.05 * SEL.index(i), ent="pop")]
v.b("Proyectar es dejar caer cada punto sobre la línea, en ángulo recto. [[a]]Cada cliente queda con un solo número: su posición sobre PC1. [[b]]Ese número se llama puntaje del componente.",
    proy, P("cada cliente → 1 número sobre PC1", 1440, 820, "ambar", 40, en="a", t=0.6), P("puntaje del componente", 1440, 920, "violeta", 40, en="b"))
v.b("Fíjate en lo que la profesora destacó: [[a]]PCA no usa la variable fuga para nada. [[b]]Es no supervisado: solo mira cómo varían las variables entre sí.",
    CL(), P("PCA no usa la fuga", 960, 360, "rojo", 66, en="a"), P("no supervisado: solo mira cómo varían las variables", 960, 560, "teal", 50, en="b"))

# ---- 3. A mano 2x2
def matriz(x, y, filas, c="tinta", tam=56, en=None, t=0.0, id_=None):
    nc = len(filas[0])
    w = 150 if nc == 2 else 75
    acts = [L(x - w, y - 90, x - w, y + 90, c, 5, en=en, t=t), L(x + w, y - 90, x + w, y + 90, c, 5, en=en, t=t),
            L(x - w, y - 90, x - w + 25, y - 90, c, 5, en=en, t=t), L(x - w, y + 90, x - w + 25, y + 90, c, 5, en=en, t=t),
            L(x + w, y - 90, x + w - 25, y - 90, c, 5, en=en, t=t), L(x + w, y + 90, x + w - 25, y + 90, c, 5, en=en, t=t)]
    for i, fila in enumerate(filas):
        for j, val in enumerate(fila):
            xx = x - 65 + 130 * j if nc == 2 else x
            acts.append(T(val, xx, y - 42 + 84 * i, tam=tam, c=c, en=en, t=t + 0.1))
    return acts


v.b("Ahora, resolvamos un PCA a mano, con el ejemplo del apunte. [[a]]Dos variables estandarizadas, con correlación cero coma seis. [[b]]Con datos estandarizados, la matriz de covarianza es la matriz de correlaciones: unos en la diagonal, y cero coma seis fuera.",
    CL(), T("PCA a mano · ejemplo del apunte", 960, 190, tam=44, c="gris", w="sb", ent="subir"), P("r = 0.6", 960, 310, "violeta", 54, en="a"),
    matriz(960, 560, [["1", "0.6"], ["0.6", "1"]], "violeta", 60, en="b"), T("matriz de covarianza (= correlaciones)", 960, 760, tam=36, c="gris", w="sb", en="b", t=0.4),
    sec="3. PCA a mano con una 2×2")
v.b("Probemos la dirección diagonal, uno, uno. [[a]]Multiplicamos la matriz por ella: uno más cero coma seis, es uno coma seis, en las dos filas. [[b]]Da uno coma seis, uno coma seis: la misma dirección, estirada uno coma seis veces. Ese uno coma seis es lambda uno.",
    CL(), matriz(420, 480, [["1", "0.6"], ["0.6", "1"]], "violeta", 50), T("×", 640, 480, tam=60, c="gris"),
    matriz(780, 480, [["1"], ["1"]], "tinta", 50), T("=", 940, 480, tam=60, c="gris", en="a"),
    T("1 + 0.6 = 1.6", 1400, 420, tam=50, c="tinta", en="a", t=0.2), T("0.6 + 1 = 1.6", 1400, 540, tam=50, c="tinta", en="a", t=0.4),
    T("= 1.6 × (1, 1)", 1180, 720, tam=64, c="violeta", en="b"), P("λ₁ = 1 + r = 1.6", 1180, 860, "violeta", 50, en="b", t=0.6))
v.b("Con la otra dirección, uno, menos uno: [[a]]uno menos cero coma seis da cero coma cuatro, y cero coma seis menos uno da menos cero coma cuatro. [[b]]Es cero coma cuatro por uno, menos uno. Lambda dos vale cero coma cuatro. La regla: uno más r, y uno menos r.",
    CL(), matriz(420, 480, [["1", "0.6"], ["0.6", "1"]], "violeta", 50), T("×", 640, 480, tam=60, c="gris"),
    matriz(780, 480, [["1"], ["−1"]], "tinta", 50), T("=", 940, 480, tam=60, c="gris", en="a"),
    T("1 − 0.6 = 0.4", 1400, 420, tam=50, c="tinta", en="a", t=0.2), T("0.6 − 1 = −0.4", 1400, 540, tam=50, c="tinta", en="a", t=0.4),
    T("= 0.4 × (1, −1)", 1180, 720, tam=64, c="teal", en="b"), P("λ₂ = 1 − r = 0.4", 1180, 860, "teal", 50, en="b", t=0.6))
v.b("Varianza explicada: [[a]]uno coma seis, dividido en dos, ochenta por ciento. [[b]]Cero coma cuatro dividido en dos, veinte por ciento. Suman cien.",
    CL(), T("varianza explicada = λ / suma de λ", 960, 220, tam=48, c="gris", w="sb", ent="subir"),
    Bc(560, 520, 620, 340, "blanco", "violeta", 5, 30, sombra=True, en="a", ent="subir"), T("1.6 / 2", 560, 440, tam=60, c="tinta", en="a", t=0.1), N(80, 560, 580, "{:.0f}", 110, "violeta", suf=" %", en="a", t=0.3, dur=0.7),
    Bc(1360, 520, 620, 340, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"), T("0.4 / 2", 1360, 440, tam=60, c="tinta", en="b", t=0.1), N(20, 1360, 580, "{:.0f}", 110, "teal", suf=" %", en="b", t=0.3, dur=0.7))
v.b("Las direcciones se normalizan para que midan uno. [[a]]Uno, uno, dividido en raíz de dos: cero coma setecientos siete, cero coma setecientos siete. [[b]]Y comprueba que son perpendiculares: [[c]]cero coma setecientos siete por cero coma setecientos siete, más cero coma setecientos siete por menos cero coma setecientos siete, da cero.",
    CL(), T("vectores de largo 1", 960, 200, tam=48, c="gris", w="sb", ent="subir"),
    T("v₁ = (1, 1) / √2 = (0.707, 0.707)", 960, 340, tam=54, c="violeta", en="a"), T("v₂ = (1, −1) / √2 = (0.707, −0.707)", 960, 450, tam=54, c="teal", en="a", t=0.4),
    T("¿perpendiculares?", 960, 590, tam=44, c="gris", w="sb", en="b"),
    T("0.707 · 0.707 + 0.707 · (−0.707) = 0", 960, 700, tam=54, c="tinta", en="c"), K(1600, 700, "check", 70, en="c", t=1.2))

# proyección de un punto
v.b("Ahora proyectamos un cliente, con z igual a uno coma cero, y cero coma dos. [[a]]Su puntaje en PC1 es el producto punto con v uno: uno coma cero más cero coma dos, dividido en raíz de dos. [[b]]Uno coma dos dividido en uno coma cuatro uno cuatro: cero coma ochocientos cuarenta y nueve.",
    CL(), P("cliente z = (1.0 , 0.2)", 960, 200, "azul", 52, ent="subir"),
    T("PC1 = v₁ · z = (1.0 + 0.2) / √2", 960, 360, tam=54, c="violeta", en="a"), T("= 1.2 / 1.414", 860, 490, tam=54, c="tinta", en="b"),
    N(0.849, 1260, 490, "= {:.3f}", 70, "violeta", en="b", t=0.4, dur=0.7), sec="4. Proyectar y reconstruir")
v.b("Su puntaje en PC2: [[a]]uno coma cero menos cero coma dos, dividido en raíz de dos: [[b]]cero coma ocho entre uno coma cuatro uno cuatro, cero coma quinientos sesenta y seis.",
    T("PC2 = v₂ · z = (1.0 − 0.2) / √2", 960, 650, tam=54, c="teal", en="a"), N(0.566, 960, 780, "= 0.8 / 1.414 = {:.3f}", 60, "teal", en="b", dur=0.7))
v.b("¿Y si nos quedamos solo con PC1? Reconstruimos el punto: [[a]]cero coma ochocientos cuarenta y nueve, por cero coma setecientos siete, da cero coma seis, en las dos coordenadas. [[b]]El punto original era uno coma cero, cero coma dos. Lo reconstruido es cero coma seis, cero coma seis.",
    CL(), T("reconstruir con solo PC1", 960, 200, tam=48, c="gris", w="sb", ent="subir"),
    T("0.849 · (0.707 , 0.707) = (0.6 , 0.6)", 960, 340, tam=56, c="violeta", en="a"),
    Bc(620, 560, 560, 200, "blanco", "azul", 5, 28, sombra=True, en="b", ent="subir"), T("original (1.0 , 0.2)", 620, 560, tam=44, c="azul", en="b", t=0.1),
    Bc(1300, 560, 560, 200, "blanco", "violeta", 5, 28, sombra=True, en="b", ent="subir", t=0.3), T("reconstruido (0.6 , 0.6)", 1300, 560, tam=44, c="violeta", en="b", t=0.4))
v.b("El error de reconstrucción: [[a]]uno coma cero menos cero coma seis, al cuadrado, más cero coma dos menos cero coma seis, al cuadrado. [[b]]Cero coma dieciséis más cero coma dieciséis: cero coma treinta y dos. [[c]]Y fíjate: es justo el puntaje de PC2 al cuadrado. Lo que se pierde es lo que estaba en PC2.",
    T("(1.0 − 0.6)² + (0.2 − 0.6)²", 960, 750, tam=50, c="tinta", en="a"), T("= 0.16 + 0.16 = 0.32", 960, 850, tam=56, c="rojo", en="b"),
    P("= 0.566²: lo que había en PC2", 960, 960, "teal", 42, en="c"))

# volver a los datos
v.b("Ahora vuelve a los datos de la clase. [[a]]Segundos y valor del cliente tenían r igual a cero coma ochenta y cuatro. [[b]]Con la regla, lambda uno es uno más cero coma ochenta y cuatro: uno coma ochenta y cuatro. Exactamente lo que vimos en el gráfico.",
    CL(), T("con los datos de la clase", 960, 220, tam=44, c="gris", w="sb", ent="subir"), T(f"r = {R_:.3f}", 960, 380, tam=70, c="violeta", en="a"),
    T(f"λ₁ = 1 + {R_:.3f} = {1 + R_:.3f}", 960, 540, tam=70, c="violeta", en="b"), T(f"{1 + R_:.3f} / 2 = {GEO['var'][0]:.1f} %", 960, 690, tam=60, c="tinta", en="b", t=0.8),
    P("la misma cuenta, con datos reales", 960, 860, "teal", 46, en="b", t=1.4), sec="5. De vuelta a los datos")

# ---- 6. Errores
v.b("Dos errores típicos. [[a]]El primero se dijo en la clase: «PCA es un clustering». [[b]]No. PCA no forma grupos: solo cambia los ejes con que miras los datos.",
    CL(), T("«PCA es un clustering»", 960, 330, tam=76, c="tinta", id="pcl", en="a"), U("pcl", tachado=1.0, en="b", dur=0.6),
    P("PCA no forma grupos: cambia los ejes", 960, 560, "verde", 54, en="b", t=0.4), sec="6. Errores típicos")
v.b("El segundo está en las diapositivas: [[a]]el ejemplo de valores propios usa una matriz que no es simétrica, diez, menos dieciocho, seis, menos once. [[b]]Sus vectores no son perpendiculares. [[c]]Una matriz de covarianza siempre es simétrica, y por eso sus componentes sí son perpendiculares. En Python se usa np.linalg.eigh.",
    CL(), P("alerta del material", 960, 200, "ambar", 46, ent="subir"),
    matriz(560, 470, [["10", "−18"], ["6", "−11"]], "gris", 50, en="a"), T("no simétrica", 560, 640, tam=36, c="gris", w="sb", en="a", t=0.3),
    P("vectores no perpendiculares", 560, 760, "rojo", 40, en="b"),
    matriz(1360, 470, [["1", "0.6"], ["0.6", "1"]], "violeta", 50, en="c"), T("covarianza: simétrica", 1360, 640, tam=36, c="violeta", w="sb", en="c", t=0.3),
    P("componentes perpendiculares", 1360, 760, "verde", 40, en="c", t=0.5), T("np.linalg.eigh", 1360, 870, tam=36, c="tinta", w="m", en="c", t=0.8))

# ---- 7. Piensa
v.b("Piensa un segundo: [[a]]dos variables estandarizadas, con correlación cero coma nueve. ¿Qué porcentaje de la varianza explica PC1?",
    CL(), T("r = 0.9", 960, 360, tam=110, c="violeta", en="a"), T("PC1 explica ¿?", 960, 560, tam=80, c="tinta", en="a", t=0.5),
    sec="7. Piensa un segundo", pausa=0.8)
v.b("Lambda uno es uno más cero coma nueve: [[a]]uno coma nueve. [[b]]Dividido en dos, noventa y cinco por ciento. Mientras más correlacionadas, más resume PC1.",
    CL(), T("λ₁ = 1 + 0.9 = 1.9", 960, 330, tam=76, c="tinta", en="a"), T("1.9 / 2 =", 820, 500, tam=76, c="tinta", en="b"),
    N(95, 1200, 500, "{:.0f}", 110, "violeta", suf=" %", en="b", t=0.3, dur=0.6), P("más correlación → PC1 resume más", 960, 700, "teal", 52, en="b", t=0.8))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]PC1 es la dirección de máxima varianza, y cada componente siguiente es perpendicular al anterior. [[b]]Los valores propios, lambda, son las varianzas de cada componente. [[c]]Y en una matriz de correlaciones de dos por dos, lambda es uno más r, y uno menos r.",
    CL(), P("PC1: máxima varianza · los siguientes, perpendiculares", 960, 300, "violeta", 46, en="a"), P("λ = varianza de cada componente", 960, 480, "azul", 50, en="b"),
    P("2×2: λ = 1 + r  y  1 − r", 960, 660, "teal", 54, en="c"), sec="8. Para llevar")
v.b("Siguiente parada: [[a]]con quince variables, ¿cuántos componentes nos quedamos?",
    CL(), siguiente(6, "Varianza explicada y cuántos componentes"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
