import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v)

OUT = out("guion_c39_06_silueta.json")
v = V("La silueta")
KM, SI = D["km"], D["sil"]
S = KM["silueta"][1:]

v.portada("La silueta", "¿Cada punto está más cerca de su grupo que del vecino?",
          "Sexta parada: la silueta. Un criterio que no solo mira si los grupos son compactos, sino también si están separados.")

# ---- 1. Hook
v.b("El codo solo mira una cosa: qué tan compactos son los grupos. [[a]]Pero un grupo puede ser compacto y estar pegado a otro grupo. [[b]]La silueta hace una pregunta más completa, punto por punto: [[c]]¿está este punto claramente más cerca de su propio grupo que del grupo vecino? Si la respuesta es sí para casi todos, los grupos están bien separados.",
    CL(), T("compacto no basta: también separado", 960, 150, tam=54, c="tinta", ent="subir"),
    P("el codo solo mide compacidad", 960, 330, "azul", 46, en="a"),
    P("¿más cerca de su grupo que del grupo vecino?", 960, 520, "teal", 50, en="c"), sec="1. Compacto y separado")

# ---- 2. Analogía
v.b("Una analogía. [[a]]En una fiesta, cada invitado está sentado en una mesa. [[b]]Para cada persona, mides la distancia promedio a la gente de su mesa, y la distancia promedio a la mesa vecina más cercana. [[c]]Si está mucho más cerca de los suyos, está bien ubicado. Si está igual de cerca de ambos, está en la frontera. Y si está más cerca de la otra mesa, debería cambiarse.",
    CL(), T("las mesas de una fiesta", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "verde", "bien ubicado", "mucho más cerca de su mesa", "c"), (960, "ambar", "en la frontera", "igual de cerca de ambas", "c"),
              (1440, "rojo", "mal ubicado", "más cerca de la otra mesa", "c")], y=520, w=420, h=280, tam_t=46, tam_d=36),
    T("distancia media a su mesa  ·  distancia media a la mesa vecina", 960, 300, tam=40, c="tinta", w="sb", en="b"),
    analogia(1050), sec="2. Analogía: las mesas")

# ---- 3. La fórmula
v.b("La diapositiva veinte lo formaliza. [[a]]Para cada observación se calculan dos números. El primero: la distancia media a los otros puntos de su propio grupo. [[b]]El segundo: la menor distancia media a los puntos de otro grupo; es decir, el grupo vecino más cercano. [[c]]La silueta es el segundo menos el primero, dividido por el mayor de los dos.",
    CL(), T("silueta: interpretación y ejemplo · diapositiva 20", 960, 140, tam=48, c="tinta", ent="subir"),
    T("a(i): distancia media a los otros puntos de su grupo", 960, 290, tam=42, c="azul", w="sb", en="a"),
    T("b(i): menor distancia media a los puntos de otro grupo", 960, 390, tam=42, c="ambar", w="sb", en="b"),
    caja_formula("s(i) = [ b(i) − a(i) ] / máx{ a(i), b(i) }", y=550, c="violeta", tam=54, w=1200, en="c"), diapo(20, en="a"), sec="3. La fórmula")
v.b("El ejemplo de la diapositiva: [[a]]distancia media a su grupo, dos; distancia media al vecino, seis. [[b]]Seis menos dos es cuatro; dividido por seis, el mayor, da 0.667. [[c]]Una aclaración importante: aquí se usa la distancia euclidiana, no su cuadrado. Y por convención, un punto que está solo en su grupo tiene silueta cero.",
    T("a(i) = 2   ·   b(i) = 6   →   s(i) = (6 − 2) / 6 =", 1120, 720, tam=44, c="tinta", w="sb", anc="rm", en="a"),
    N(0.667, 1150, 720, "{:.3f}", 64, "violeta", anc="lm", en="b", t=1.5),
    T("distancia euclidiana (sin cuadrado) · un punto solo en su grupo: s = 0", 960, 860, tam=36, c="rojo", w="sb", en="c"))

# ---- 4. Lectura
v.b("¿Cómo se lee? La silueta va de menos uno a uno. [[a]]Cerca de uno: el punto está mucho más cerca de su grupo que del vecino; hay buena separación. [[b]]Cerca de cero: está en una zona de superposición, en la frontera entre dos grupos. [[c]]Negativa: en promedio, está más cerca de otro grupo que del suyo.",
    CL(), T("de −1 a 1", 960, 150, tam=56, c="tinta", ent="subir"),
    L(300, 420, 1620, 420, "gris", 8, en=None),
    T("−1", 300, 490, tam=40, c="rojo", w="b"), T("0", 960, 490, tam=40, c="ambar", w="b"), T("1", 1620, 490, tam=40, c="verde", w="b"),
    P("cerca de 1: bien separado", 1500, 330, "verde", 40, en="a"), P("cerca de 0: superposición", 960, 620, "ambar", 40, en="b"),
    P("negativa: más cerca de otro grupo", 420, 330, "rojo", 40, en="c"), diapo(20, en="a"), sec="4. Cómo se lee")

# ---- 5. Cuenta adicional
v.b("Hagamos una cuenta completa con los datos de siempre: uno, dos y tres en un grupo; ocho, nueve y diez en el otro. [[a]]Tomemos el punto tres. Su distancia media a su grupo: dos hasta el uno, uno hasta el dos; promedio, 1.5. [[b]]Su distancia media al otro grupo: cinco, seis y siete; promedio, seis. [[c]]Silueta: seis menos uno coma cinco, cuatro coma cinco, dividido por seis: 0.75. Un punto bien ubicado, aunque sea el del borde.",
    CL(), T("la silueta del punto 3", 960, 150, tam=54, c="tinta", ent="subir"),
    casillas([1, 2, 3], 290, x0=520, paso=140, c="azul", en=None, pre="sa"), casillas([8, 9, 10], 290, x0=1120, paso=140, c="ambar", en=None, pre="sb"),
    R(800, 290, 140, 130, "violeta", r=20, en="a"),
    T("a = (2 + 1) / 2 =", 1000, 450, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(1.5, 1030, 450, "{:.1f}", 64, "azul", anc="lm", en="a", t=1.5),
    T("b = (5 + 6 + 7) / 3 =", 1000, 570, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(6, 1030, 570, "{:.0f}", 64, "ambar", anc="lm", en="b", t=1.5),
    T("s = (6 − 1.5) / 6 =", 1000, 690, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(0.75, 1030, 690, "{:.2f}", 64, "violeta", anc="lm", en="c", t=2.5),
    adicional(1050, en="a"), sec="5. Una cuenta completa")
v.b("Y el punto uno, en el extremo izquierdo: [[a]]su distancia media a su grupo también es uno coma cinco, pero al otro grupo es siete, ocho y nueve: ocho. [[b]]Su silueta es ocho menos uno coma cinco, sobre ocho: 0.813. [[c]]Para tener la silueta media, se calcula esto para todos los puntos y se promedia.",
    T("punto 1: a = 1.5 · b = 8 · s = 6.5 / 8 =", 1000, 830, tam=42, c="tinta", w="sb", anc="rm", en="a"), N(0.813, 1030, 830, "{:.3f}", 58, "violeta", anc="lm", en="b"),
    P("silueta media = promedio sobre todos los puntos", 960, 950, "teal", 40, en="c"))

# ---- 6. Los vinos
GS = "gs6"
v.b("Con los vinos y las trece variables. [[a]]La silueta media para {k|, ka,} igual a dos es 0.258. Para tres, 0.284. Para cuatro, 0.258. [[b]]Y después baja: 0.235, 0.192, 0.199 y 0.136. [[c]]El máximo está en {k|, ka,} igual a tres: la silueta coincide con el codo. No hay silueta para {k|, ka,} igual a uno, porque no existe un grupo vecino.",
    CL(), T("silueta media en los vinos · diapositiva 21", 960, 130, tam=50, c="tinta", ent="subir"),
    G(360, 230, 1240, 860, 1.5, 8.5, 0, 0.32, [dict(xs=list(range(2, 9)), ys=S, color="ambar", grosor=6, marcadores=11, revela=0.0)],
      xticks=list(range(2, 9)), yticks=[0, 0.1, 0.2, 0.3], yfmt="{:.1f}", xlabel="número de grupos k", ylabel="silueta", id=GS, tam_txt=28),
    U(GS, **{"series.0.revela": 0.34}, en="a", dur=1.5), U(GS, **{"series.0.revela": 1.0}, en="b", dur=1.8),
    R(3, S[1], 70, 70, "verde", r=35, ref=GS, en="c"), T("k = 3: 0.284", 1480, 380, tam=48, c="verde", w="b", en="c", t=0.4),
    T("k = 1: sin grupo vecino", 1480, 480, tam=36, c="gris", w="sb", en="c", t=2.5), real(1050, en="a"), sec="6. Los vinos")
v.b("Pero ojo con el tamaño del número. [[a]]Una silueta de 0.284 es modesta: indica que los grupos se separan, pero que persiste superposición. [[b]]Mirando por dentro, en un cálculo adicional: los grupos uno y dos tienen silueta media cerca de 0.34 y 0.35; el grupo tres, solo 0.177, y ahí hay siete vinos con silueta negativa. [[c]]Es el mejor de estos candidatos, no una separación perfecta.",
    CL(), T("0.284: el mejor de estos candidatos, no una separación perfecta", 960, 150, tam=44, c="tinta", ent="subir"),
    barras_v([("grupo 1", SI["por_grupo"][0], "azul", "b"), ("grupo 2", SI["por_grupo"][1], "ambar", "b"), ("grupo 3", SI["por_grupo"][2], "teal", "b")],
             x0=560, paso=400, base=760, escala=1200, w=200, fmt="{:.3f}"),
    P("persiste superposición", 1620, 420, "ambar", 40, en="a"),
    T("grupo 3: 7 vinos con silueta negativa", 960, 900, tam=40, c="rojo", w="sb", en="b", t=2.5),
    adicional(1050, en="b"), diapo(21, en="a", y=1010))

v.b("En R, la silueta se calcula con la función silhouette del paquete cluster. [[a]]Recibe los grupos del ajuste y la matriz de distancias entre todos los vinos, que se calcula una sola vez con dist. [[b]]La tercera columna del resultado es la silueta de cada vino, y su promedio es la silueta media. En Python, todo eso lo hace una sola función, que ya entrega el promedio.",
    CL(), T("la silueta en R y en Python · diapositivas 25 y 26", 960, 130, tam=48, c="tinta", ent="subir"),
    codigo(["distancias <- dist(z)", "silueta[k] <- mean(silhouette(modelos[[k]]$cluster, distancias)[, 3])"], y=330, w=1700, tam=34, c="azul", en="a"),
    codigo(["silueta.append(silhouette_score(z, modelo.labels_))"], y=560, w=1700, tam=34, c="violeta", en="b"),
    P("columna 3 = silueta de cada vino · promedio = silueta media", 960, 740, "teal", 42, en="b", t=2.0), diapo(25, en="a"))

# ---- 7. Error típico
v.b("Errores típicos con la silueta. [[a]]Uno: leer 0.284 como un veintiocho por ciento de aciertos. No es un porcentaje; es una medida de separación relativa. [[b]]Dos: calcularla con distancias al cuadrado, como el WSS. La silueta usa la distancia euclidiana, sin cuadrado. [[c]]Y tres: pedirla con un solo grupo. Con {k|, ka,} igual a uno no hay vecino, por eso el script guarda un valor faltante.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("0.284 ≠ 28.4 % de aciertos", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("usar distancias al cuadrado", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("pedir silueta con k = 1", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    codigo(["if k > 1: silueta.append(silhouette_score(z, modelo.labels_))", "else:     silueta.append(np.nan)"], y=800, w=1500, tam=32, c="teal",
           en="c", t=1.5), sec="7. Errores típicos")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un vino tiene distancia media tres a su propio grupo, y dos al grupo vecino. [[b]]¿Cuánto vale su silueta, y qué significa?",
    CL(), T("distancia media a su grupo = 3   ·   al grupo vecino = 2", 960, 320, tam=50, c="tinta", w="b", en="a"),
    P("s = ?   ¿qué significa?", 960, 520, "violeta", 56, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Dos menos tres es menos uno; [[a]]dividido por tres, el mayor, da [[b]]menos 0.333. Es negativa: ese vino está, en promedio, más cerca del grupo vecino que del suyo. Probablemente está mal asignado.",
    CL(), T("s = (2 − 3) / 3 =", 1000, 340, tam=54, c="tinta", w="sb", anc="rm", en="a"), N(-0.333, 1030, 340, "{:.3f}", 72, "rojo", anc="lm", en="b"),
    P("negativa: más cerca del grupo vecino", 960, 540, "rojo", 48, en="b", t=1.0), ilustrativo(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]la silueta compara, para cada punto, la distancia media a su grupo con la distancia media al grupo vecino. [[b]]Va de menos uno a uno: cerca de uno es buena separación, cerca de cero es frontera, negativa es mala asignación. [[c]]En los vinos, el máximo está en {k|, ka,} igual a tres, con 0.284: el mejor candidato, con superposición.",
    CL(), P("s = (b − a) / máx(a, b) · distancia sin cuadrado", 960, 300, "violeta", 46, en="a"),
    P("1 separado · 0 frontera · negativa mal asignado", 960, 460, "teal", 46, en="b"),
    P("vinos: máximo en k = 3 (0.284)", 960, 620, "verde", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el estadístico Gap, que compara nuestros grupos con datos sin ninguna estructura. Y qué hacer cuando los criterios no se ponen de acuerdo.",
    CL(), siguiente(7, "El estadístico Gap y los criterios en desacuerdo"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
