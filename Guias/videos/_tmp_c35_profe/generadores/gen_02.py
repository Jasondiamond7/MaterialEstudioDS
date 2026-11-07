import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, casillas, codigo, caja_formula, mils

OUT = out("guion_c35_02_bootstrap.json")
v = V("Bootstrap a mano")
BO = FG["boot"]
OBS = BO["obs"]

# cadena (7/8)^k con intermedios redondeados a 3 decimales (lo que se ve en pantalla)
cad = [0.875]
for _ in range(7):
    cad.append(round(cad[-1] * 0.875, 3))

v.portada("Bootstrap a mano", "Muestras distintas sin recolectar un solo dato nuevo",
          "Segunda parada: el bootstrap. Vas a hacer una muestra a mano, y vas a descubrir de dónde sale ese famoso 63.2 %.")

# ---- 1. Hook
v.b("Para promediar árboles distintos necesitas muestras distintas. [[a]]Pero tienes un solo conjunto de datos. [[b]]¿Cómo consigues doscientos conjuntos distintos sin recolectar ni un solo dato nuevo?",
    CL(), T("necesitas muestras distintas…", 960, 260, tam=56, c="tinta", ent="subir"),
    Bc(960, 470, 520, 160, "blanco", "teal", 5, 26, sombra=True, en="a", ent="pop"), T("1 conjunto de datos", 960, 470, tam=44, c="teal", en="a", t=0.1),
    T("¿200 conjuntos distintos, sin datos nuevos?", 960, 680, tam=56, c="violeta", en="b"), sec="1. El problema")

# ---- 2. Analogía
v.b("Piensa en una tómbola con ocho bolitas numeradas. [[a]]Sacas una, anotas su número, [[b]]y la devuelves a la tómbola. [[c]]Repites ocho veces. Como la bolita vuelve, puede salir de nuevo, y otras pueden no salir nunca.",
    CL(), T("una tómbola con 8 bolitas", 960, 170, tam=54, c="tinta", ent="subir"),
    Bc(500, 520, 420, 420, "blanco", "gris", 5, 210, sombra=True), Pt([(8, [380, 420, 620, 620], "azul")], 26),
    T("1. sacas y anotas", 1200, 400, tam=46, c="azul", anc="lm", en="a"), T("2. la DEVUELVES", 1200, 510, tam=46, c="verde", anc="lm", en="b"),
    T("3. repites 8 veces", 1200, 620, tam=46, c="violeta", anc="lm", en="c"),
    P("unas salen repetidas · otras no salen", 960, 870, "ambar", 46, en="c", t=0.8), ilustrativo(1000, texto="analogía"),
    sec="2. Analogía: la tómbola")
v.b("Eso es un muestreo con reemplazo. [[a]]Y la clave está en esas dos palabras: con reemplazo. [[b]]Cada extracción tiene siempre las mismas ocho bolitas disponibles.",
    P("muestreo CON REEMPLAZO", 1350, 760, "verde", 54, en="a"), T("cada extracción: siempre las mismas 8 disponibles", 960, 960, tam=38, c="gris", w="sb", en="b"))

# ---- 3. A mano: diapositiva 5
v.b("Hagámoslo con la diapositiva cinco de la clase. [[a]]Los datos originales son ocho filas, numeradas del uno al ocho. [[b]]La primera muestra bootstrap salió así: uno, dos, dos, cuatro, seis, seis, siete y ocho.",
    CL(), T("diapositiva 5 · muestras ilustrativas", 960, 150, tam=40, c="gris", w="sb", ent="subir"),
    T("datos originales", 960, 250, tam=40, c="teal", en="a"), casillas(list(range(1, 9)), 340, c="teal", en="a", pre="or"),
    T("bootstrap 1", 960, 490, tam=40, c="azul", en="b"), casillas([1, 2, 2, 4, 6, 6, 7, 8], 580, c="azul", en="b", pre="b1", dt=0.18),
    ilustrativo(1000, texto="ejemplo de la diapositiva 5"), sec="3. Una muestra a mano")
v.b("Mira lo que pasó. [[a]]El dos salió dos veces, y el seis también. [[b]]Y el tres y el cinco no salieron nunca. [[c]]Esas filas que quedaron afuera tienen nombre: son las filas fuera de bolsa, en inglés out-of-bag.",
    U("b1b1", relleno="ambarclaro", borde="ambar", en="a"), U("b1b2", relleno="ambarclaro", borde="ambar", en="a"),
    U("b1b4", relleno="ambarclaro", borde="ambar", en="a", t=0.3), U("b1b5", relleno="ambarclaro", borde="ambar", en="a", t=0.3),
    T("repetidas: 2 y 6", 960, 700, tam=44, c="ambar", en="a", t=0.5),
    U("orb2", relleno="rojoclaro", borde="rojo", en="b"), U("orb4", relleno="rojoclaro", borde="rojo", en="b"),
    T("nunca salieron: 3 y 5", 960, 790, tam=44, c="rojo", en="b", t=0.3),
    P("fuera de bolsa = out-of-bag (OOB)", 960, 900, "teal", 46, en="c"))
v.b("Las otras dos muestras de la diapositiva: [[a]]la segunda tiene cinco filas distintas y deja afuera el dos, el seis y el ocho. [[b]]La tercera tiene seis distintas y deja afuera el uno y el siete. [[c]]Tres muestras distintas, fabricadas con las mismas ocho filas.",
    CL(), T("las tres muestras de la diapositiva 5", 960, 150, tam=44, c="tinta", ent="subir"),
    T("bootstrap 1", 230, 300, tam=34, c="azul", anc="lm"), casillas([1, 2, 2, 4, 6, 6, 7, 8], 300, x0=580, paso=128, w=104, h=92, tam=46, c="azul", pre="m1", dt=0.05),
    T("6 distintas", 1660, 300, tam=34, c="azul", w="sb", anc="lm"),
    T("bootstrap 2", 230, 470, tam=34, c="violeta", anc="lm", en="a"), casillas([1, 1, 3, 3, 4, 5, 7, 7], 470, x0=580, paso=128, w=104, h=92, tam=46, c="violeta", en="a", pre="m2", dt=0.08),
    T("5 distintas", 1660, 470, tam=34, c="violeta", w="sb", anc="lm", en="a", t=0.8), T("afuera: 2, 6, 8", 1060, 560, tam=32, c="rojo", w="sb", en="a", t=1.0),
    T("bootstrap 3", 230, 680, tam=34, c="teal", anc="lm", en="b"), casillas([2, 3, 4, 4, 5, 6, 6, 8], 680, x0=580, paso=128, w=104, h=92, tam=46, c="teal", en="b", pre="m3", dt=0.08),
    T("6 distintas", 1660, 680, tam=34, c="teal", w="sb", anc="lm", en="b", t=0.8), T("afuera: 1, 7", 1060, 770, tam=32, c="rojo", w="sb", en="b", t=1.0),
    P("3 conjuntos distintos con las mismas 8 filas", 960, 900, "verde", 44, en="c"), ilustrativo(1000, texto="ejemplo de la diapositiva 5"))

# ---- 4. ¿Cuántas quedan afuera? (ejemplo resuelto)
v.b("Ahora la pregunta importante: ¿qué fracción de filas queda afuera, en promedio? [[a]]Sigue a una fila cualquiera, digamos la tres. En una extracción, la probabilidad de que NO salga es siete octavos: 0.875. [[b]]Y como las extracciones son independientes, para que no salga en ninguna de las ocho, multiplicamos 0.875 ocho veces.",
    CL(), T("¿qué fracción queda afuera?", 960, 160, tam=54, c="tinta", ent="subir"),
    T("una extracción:  P(la fila 3 no sale) = 7/8 =", 1080, 300, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(0.875, 1110, 300, "{:.3f}", 64, "azul", anc="lm", en="a", t=0.6),
    T("8 extracciones:  0.875 × 0.875 × … (8 veces)", 960, 420, tam=48, c="tinta", w="sb", en="b"), sec="4. ¿Cuántas quedan afuera?")
v.b("Hagamos la cuenta paso a paso, redondeando a tres decimales. [[a]]Después de dos extracciones, 0.766. [[b]]Después de cuatro, 0.586. [[c]]Y después de las ocho, 0.344. Una de cada tres filas queda afuera.",
    *[[Bx(250 + 180 * i, 900 - cad[i] * 380, 390 + 180 * i, 900, "azul" if i < 7 else "rojo", None, 0, 10, grow="y", en=("a" if i < 2 else ("b" if i < 4 else "c")), t=0.15 * (i % 2)),
       T(f"{cad[i]:.3f}", 320 + 180 * i, 900 - cad[i] * 380 - 34, tam=32, c="tinta" if i < 7 else "rojo", w="b",
         en=("a" if i < 2 else ("b" if i < 4 else "c")), t=0.15 * (i % 2) + 0.3), T(str(i + 1), 320 + 180 * i, 935, tam=28, c="gris", w="sb", en="a")]
      for i in range(8)],
    T("extracciones", 960, 985, tam=30, c="gris", w="sb", en="a", t=0.4))
v.b("En general, con ene filas, la probabilidad de quedar afuera es uno menos uno sobre ene, elevado a ene. [[a]]Con ocho filas, 0.344. [[b]]Con los 227 pacientes de entrenamiento de Heart, 0.367. [[c]]Con las 2 344 casas de entrenamiento de Ames, 0.368. Y ese número tiene nombre: uno sobre e.",
    CL(), caja_formula("P(fila afuera) = (1 − 1/n)ⁿ", y=230, c="azul", tam=60, w=1000),
    *[[T(lab, 860, 400 + 120 * i, tam=46, c="tinta", w="sb", anc="rm", en=m), N(val, 900, 400 + 120 * i, "{:.3f}", 60, "azul", anc="lm", en=m, t=0.4)]
      for i, (lab, val, m) in enumerate([("n = 8:", BO["oob8"], "a"), ("n = 227 (Heart):", BO["oob227"], "b"), ("n = 2.344 (Ames):", BO["oob2344"], "c")])],
    P("→ e⁻¹ ≈ 0.368", 960, 800, "violeta", 56, en="c", t=1.0))
v.b("De ahí salen los dos números de la diapositiva cinco y seis. [[a]]Un 36.8 % de las filas queda fuera de cada muestra bootstrap. [[b]]Y el resto, uno menos 0.368, es decir un 63.2 %, son las filas únicas que sí entran.",
    CL(), T("los dos números de la clase", 960, 180, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "≈ 36.8 %", "fuera de la muestra: filas OOB", "a"), (1360, "verde", "≈ 63.2 %", "filas únicas dentro: 1 − e⁻¹", "b")], y=500, w=640, h=360, tam_t=80))

# ---- 5. Cuántas veces entra una fila (datos reales)
v.b("¿Y las filas que sí entran, cuántas veces entran? [[a]]Lo simulamos con Heart: dos mil muestras bootstrap de 227 pacientes. [[b]]Un 36.8 % de las veces, una fila no entra. Un 36.7 %, entra exactamente una vez. [[c]]Un 18.4 %, dos veces. Y tres o más veces, el resto: cerca del 8 %.",
    CL(), T("¿cuántas veces entra una fila? · 2.000 muestras de Heart", 960, 160, tam=44, c="tinta", ent="subir"),
    *[[Bx(330 + 230 * k, 860 - OBS[k] * 1500, 490 + 230 * k, 860, ("rojo" if k == 0 else "azul"), None, 0, 12, grow="y", en=("b" if k < 2 else "c"), t=0.1 * k),
       T(f"{OBS[k] * 100:.1f} %", 410 + 230 * k, 860 - OBS[k] * 1500 - 36, tam=34, c="tinta", w="b", en=("b" if k < 2 else "c"), t=0.1 * k + 0.4),
       T(["0", "1", "2", "3", "4", "5+"][k], 410 + 230 * k, 900, tam=36, c="gris", w="sb", en="a")] for k in range(6)],
    T("veces que entra", 960, 960, tam=32, c="gris", w="sb", en="a"), L(300, 860, 1700, 860, "tinta", 4, en="a"), real(1050),
    sec="5. Cuántas veces entra una fila")
v.b("Y la fracción afuera varía poco de muestra en muestra: [[a]]en promedio 36.8 %, con un desvío de apenas 2 puntos. [[b]]El bootstrap es una máquina muy predecible de fabricar diversidad.",
    P(f"afuera: {BO['oob_media'] * 100:.1f} % en promedio · desvío {BO['oob_sd'] * 100:.1f} puntos", 960, 980 - 0, "teal", 40, en="a", ent="subir"))

# ---- 6. En Python
v.b("En Python, una muestra bootstrap es una línea. [[a]]Sacas ene números enteros al azar entre cero y ene, con reemplazo. [[b]]Y para contar cuántas veces salió cada fila, usas bincount. Las filas con cuenta cero son el OOB de esa muestra.",
    CL(), T("bootstrap en Python", 960, 170, tam=50, c="tinta", ent="subir"),
    codigo(["rng = np.random.default_rng(42)", "idx = rng.integers(0, n, n)        # n filas CON reemplazo", "cuenta = np.bincount(idx, minlength=n)",
            "oob = np.where(cuenta == 0)[0]     # filas fuera de bolsa"], y=470, w=1500, tam=38, en="a", dt=0.4),
    P("cuenta = 0 → OOB", 960, 780, "teal", 46, en="b"), sec="6. En Python")

# ---- 7. Error típico
v.b("El error típico: [[a]]sacar la muestra SIN reemplazo. Si sacas ocho de ocho sin devolver, obtienes exactamente las mismas ocho filas, en otro orden. [[b]]Cero diversidad: todos los árboles serían iguales. [[c]]Y el segundo error: creer que la muestra bootstrap es una fracción de los datos. Por defecto tiene ene extracciones, tantas como filas.",
    CL(), T("8 de 8 SIN reemplazo", 960, 220, tam=56, c="tinta", en="a"), casillas([3, 7, 1, 5, 8, 2, 6, 4], 360, c="gris", en="a", pre="sr", dt=0.08),
    P("= las mismas 8 filas · cero diversidad", 960, 510, "rojo", 46, en="b"),
    T("la muestra bootstrap tiene n extracciones (no una fracción)", 960, 680, tam=42, c="tinta", w="sb", en="c"),
    T("max_samples lo cambia, pero el valor por defecto es n", 960, 770, tam=36, c="gris", w="sb", en="c", t=0.6), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Las tres muestras de la diapositiva tienen seis, cinco y seis filas distintas, de ocho. [[b]]¿Eso calza con la fórmula?",
    CL(), P("6, 5 y 6 filas distintas de 8", 960, 300, "azul", 56, en="a"), T("¿calza con la fórmula?", 960, 480, tam=72, c="tinta", en="b"),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("La fórmula dice que, con ocho filas, queda afuera un 34.4 %, o sea entra un 65.6 %: [[a]]en promedio, 5.25 filas distintas de ocho. [[b]]Las muestras de la diapositiva promedian 5.67. [[c]]Calza: con solo tres muestras de ocho filas, la diferencia es puro azar.",
    CL(), T("esperado: 8 × (1 − 0.344) = 8 × 0.656 =", 1020, 300, tam=48, c="tinta", w="sb", anc="rm"), N(8 * (1 - 0.344), 1050, 300, "{:.2f}", 64, "azul", anc="lm", en="a"),
    T("diapositiva: (6 + 5 + 6) / 3 =", 1020, 440, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(17 / 3, 1050, 440, "{:.2f}", 64, "violeta", anc="lm", en="b", t=0.4),
    P("calza: con 3 muestras de 8 filas, la diferencia es azar", 960, 620, "verde", 46, en="c"), ilustrativo(1000, texto="ejemplo de la diapositiva 5"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]bootstrap es sacar ene filas con reemplazo. [[b]]Cada fila queda afuera con probabilidad uno menos uno sobre ene, elevado a ene, que tiende a 0.368. [[c]]Así fabricas muestras distintas con los mismos datos, y de regalo, un 36.8 % de filas fuera de bolsa en cada una.",
    CL(), P("bootstrap = n filas CON reemplazo", 960, 300, "azul", 52, en="a"),
    P("(1 − 1/n)ⁿ → e⁻¹ ≈ 0.368", 960, 470, "violeta", 52, en="b"), P("muestras distintas + 36.8 % fuera de bolsa", 960, 640, "teal", 50, en="c"),
    sec="9. Para llevar")
v.b("Siguiente parada: [[a]]qué hacer con ese 36.8 % que quedó afuera. Una validación gratis.",
    CL(), siguiente(3, "Fuera de bolsa: el error OOB"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
