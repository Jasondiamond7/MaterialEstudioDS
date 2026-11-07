import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_03_escalar.json")
v = V("Escalar: estandarizar y normalizar")
SD, ME = D["sd"], D["mean"]

v.portada("Escalar las variables", "Estandarizar y normalizar: que ninguna variable grite más fuerte",
          "Segunda parada: escalar las variables. Por qué es obligatorio antes de PCA, y cómo se hace sin hacer trampa.")

# ---- 1. Hook
v.b("Imagina una reunión donde todos opinan. [[a]]Pero una persona habla con un megáfono. [[b]]No importa si tiene razón: es la única que se escucha.",
    CL(), P("una reunión", 960, 230, "violeta", 56, ent="subir"),
    *[Bc(400 + 160 * i, 520, 110, 110, "blanco", "gris", 4, 55, ent="pop", t=0.08 * i) for i in range(5)],
    Bc(1400, 520, 150, 150, "rojoclaro", "rojo", 6, 75, en="a", ent="pop"), T("megáfono", 1400, 660, tam=40, c="rojo", w="b", en="a", t=0.2),
    P("la única que se escucha", 960, 820, "rojo", 52, en="b"), sec="1. El megáfono")
v.b("En los datos de la clase, [[a]]la variable segundos llega a miles. [[b]]Y reclamos es solo cero o uno. Sin escalar, segundos es la persona del megáfono.",
    CL(), Bc(560, 480, 640, 300, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("segundos", 560, 420, tam=56, c="rojo", en="a", t=0.1), T("llega a miles", 560, 520, tam=40, c="tinta", w="sb", en="a", t=0.2),
    Bc(1360, 480, 640, 300, "blanco", "teal", 5, 30, sombra=True, en="b", ent="subir"), T("reclamos", 1360, 420, tam=56, c="teal", en="b", t=0.1), T("0 o 1", 1360, 520, tam=40, c="tinta", w="sb", en="b", t=0.2),
    P("sin escalar, segundos tiene el megáfono", 960, 790, "rojo", 50, en="b", t=0.8))

# ---- 2. La escala manda
def es(v_):
    return f"{v_:,.0f}".replace(",", ".") if v_ >= 1000 else (f"{v_:.0f}" if v_ >= 100 else (f"{v_:.1f}" if v_ >= 10 else f"{v_:.2f}"))


VARS = [("segundos", "rojo"), ("clv", "ambar"), ("frec_sms", "violeta"), ("frec_llamadas", "azul"), ("monto", "teal"), ("reclamos", "verde")]


def lx(v_):                       # eje logaritmico 0.1 .. 10.000 -> 560 .. 1640 px
    return 560 + (math.log10(v_) + 1) / 5 * 1080


v.b("Mira la desviación estándar de seis variables, en el entrenamiento. [[a]]Segundos: dos mil novecientos sesenta y nueve. [[b]]El valor del cliente, doscientos treinta y uno. [[c]]Mensajes, setenta y cinco. Llamadas, cuarenta y cinco coma siete. [[d]]Monto, dos coma cuatro. Y reclamos, cero coma veintisiete.",
    CL(), T("desviación estándar · datos de la clase (entrenamiento)", 960, 180, tam=38, c="gris", w="sb"),
    *[[T(nm, 520, 290 + 100 * i, tam=34, c="tinta", w="sb", anc="rm", en=k), Bx(560, 260 + 100 * i, lx(SD[nm]), 320 + 100 * i, col, None, 0, 10, grow="x", en=k, t=0.1),
       T(es(SD[nm]), lx(SD[nm]) + 20, 290 + 100 * i, tam=36, c=col, anc="lm", en=k, t=0.5)]
      for i, ((nm, col), k) in enumerate(zip(VARS, ["a", "b", "c", "c", "d", "d"]))],
    *[T(lab, lx(val), 900, tam=28, c="gris", w="sb", en="a") for lab, val in (("0.1", 0.1), ("1", 1), ("10", 10), ("100", 100), ("1.000", 1000), ("10.000", 10000))],
    T("eje en escala logarítmica", 1100, 950, tam=28, c="gris", w="sb", en="a"), sec="2. La escala manda")
v.b("Las barras están en escala logarítmica: cada marca es diez veces la anterior. [[a]]Entre segundos y reclamos hay más de cuatro órdenes de magnitud.",
    R(1000, 740, 1420, 170, "ambar", 30, id="rr1", ent="fade", t=0.2), P("más de 4 órdenes de magnitud", 1100, 1010, "ambar", 40, en="a"))
v.b("¿Y la varianza? Es la desviación al cuadrado. [[a]]Con los números redondos del apunte: tres mil al cuadrado son nueve millones. [[b]]Cero coma tres al cuadrado es cero coma cero nueve. [[c]]Nueve millones dividido en cero coma cero nueve: cien millones de veces más varianza.",
    CL(), T("ejemplo del apunte", 960, 190, tam=34, c="gris", w="sb"),
    T("3.000² = 9.000.000", 960, 340, tam=64, c="rojo", en="a"), T("0.3² = 0.09", 960, 480, tam=64, c="teal", en="b"),
    T("9.000.000 / 0.09 =", 820, 650, tam=60, c="tinta", en="c"), T("100.000.000", 1380, 650, tam=76, c="violeta", en="c", t=0.4),
    P("veces más varianza", 960, 820, "violeta", 46, en="c", t=0.8))
v.b("Y PCA busca justamente la dirección de mayor varianza. [[a]]Sin escalar, el primer componente sería casi solo segundos, solo por su unidad de medida.",
    CL(), P("PCA busca la máxima varianza", 960, 380, "azul", 60, ent="subir"), P("sin escalar: PC1 ≈ segundos, por la unidad", 960, 600, "rojo", 54, en="a"))

# ---- 3. Estandarizar
v.b("La solución más común es estandarizar. [[a]]A cada valor le restas la media, [[b]]y lo divides por la desviación estándar. Eso se llama z-score.",
    CL(), T("estandarizar (z-score)", 960, 200, tam=56, c="violeta", ent="subir"),
    caja_formula("z = (x − media) / desviación", y=420, c="violeta", tam=64, w=1300),
    P("1 · restar la media", 640, 640, "azul", 46, en="a"), P("2 · dividir por la desviación", 1300, 640, "teal", 46, en="b"), sec="3. Estandarizar: z-score")
v.b("El resultado se lee como: [[a]]¿a cuántas desviaciones está este valor del promedio? [[b]]Después de estandarizar, todas las variables quedan con media cero y desviación uno.",
    P("¿a cuántas desviaciones del promedio?", 960, 790, "violeta", 48, en="a"), P("media 0 · desviación 1", 960, 910, "verde", 48, en="b"))
v.b("Un ejemplo del apunte. [[a]]Segundos tiene media cuatro mil, y desviación tres mil. Un cliente habló diez mil segundos. [[b]]Diez mil menos cuatro mil, son seis mil. [[c]]Seis mil dividido en tres mil: z igual a dos.",
    CL(), T("ejemplo del apunte", 960, 180, tam=34, c="gris", w="sb"),
    Bc(560, 520, 720, 560, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), T("segundos", 560, 300, tam=48, c="rojo", en="a", t=0.1),
    T("media 4.000 · desv. 3.000", 560, 380, tam=34, c="gris", w="sb", en="a", t=0.2), T("x = 10.000", 560, 460, tam=44, c="tinta", en="a", t=0.3),
    T("(10.000 − 4.000) / 3.000", 560, 570, tam=40, c="tinta", w="sb", en="b"), T("= 6.000 / 3.000", 560, 640, tam=40, c="tinta", w="sb", en="b", t=0.4),
    N(2, 560, 730, "z = {:.0f}", 80, "rojo", en="c", dur=0.6), sec="4. Un ejemplo a mano")
v.b("Ahora reclamos: media cero coma uno, y desviación cero coma tres. Un cliente con un reclamo: [[a]]uno menos cero coma uno, es cero coma nueve. [[b]]Dividido en cero coma tres: z igual a tres.",
    Bc(1360, 520, 720, 560, "blanco", "teal", 5, 30, sombra=True, ent="subir"), T("reclamos", 1360, 300, tam=48, c="teal", t=0.1),
    T("media 0.1 · desv. 0.3", 1360, 380, tam=34, c="gris", w="sb", t=0.2), T("x = 1", 1360, 460, tam=44, c="tinta", t=0.3),
    T("(1 − 0.1) / 0.3", 1360, 570, tam=40, c="tinta", w="sb", en="a"), T("= 0.9 / 0.3", 1360, 640, tam=40, c="tinta", w="sb", en="a", t=0.4),
    N(3, 1360, 730, "z = {:.0f}", 80, "teal", en="b", dur=0.6))
v.b("¿Te das cuenta? [[a]]Antes, diez mil segundos parecían enormes al lado de un reclamo. [[b]]Estandarizados, el reclamo es más inusual: está a tres desviaciones, y los segundos solo a dos.",
    P("antes: 10.000 frente a 1", 960, 880, "gris", 44, en="a"), P("después: el reclamo (z = 3) es más inusual que los segundos (z = 2)", 960, 980, "violeta", 38, en="b"))
zs = (10000 - ME["segundos"]) / SD["segundos"]
zr = (1 - 0.0782) / 0.2685          # media y desviación exactas del StandardScaler (ver datos.py / figuras_c32)
v.b("Con los datos reales de la clase, los números son parecidos. [[a]]Segundos tiene media cuatro mil doscientos dos y desviación dos mil novecientos sesenta y nueve: diez mil segundos dan z uno coma noventa y cinco. [[b]]Y un reclamo, con media 0.078 y desviación 0.269, da z 3.43.",
    CL(), T("con los datos de la clase (entrenamiento)", 960, 200, tam=40, c="gris", w="sb"),
    T(f"(10.000 − 4.202) / 2.969 = {zs:.2f}", 960, 420, tam=56, c="rojo", en="a"),
    T(f"(1 − 0.078) / 0.269 = {zr:.2f}", 960, 600, tam=56, c="teal", en="b"))

# ---- 4. Min-max
v.b("La otra opción es normalizar, con min-max. [[a]]Le restas el mínimo, y divides por el rango: máximo menos mínimo. [[b]]Todo queda comprimido entre cero y uno, y se conserva el orden.",
    CL(), T("normalizar (min-max)", 960, 200, tam=56, c="azul", ent="subir"),
    caja_formula("(x − mín) / (máx − mín)", y=420, c="azul", tam=64, w=1200), P("todo queda entre 0 y 1", 960, 660, "teal", 50, en="b"), sec="5. Normalizar: min-max")
v.b("Con el ejemplo del apunte: mínimo cincuenta, máximo diecisiete mil. [[a]]Diez mil menos cincuenta, son nueve mil novecientos cincuenta. [[b]]Diecisiete mil menos cincuenta, dieciséis mil novecientos cincuenta. [[c]]El cociente: cero coma quinientos ochenta y siete.",
    CL(), T("ejemplo del apunte · mín 50 · máx 17.000 · x = 10.000", 960, 200, tam=38, c="gris", w="sb"),
    T("10.000 − 50 = 9.950", 960, 360, tam=60, c="tinta", en="a"), T("17.000 − 50 = 16.950", 960, 500, tam=60, c="tinta", en="b"),
    T("9.950 / 16.950 =", 820, 680, tam=62, c="azul", en="c"), N(0.587, 1290, 680, "{:.3f}", 90, "azul", en="c", t=0.4, dur=0.8),
    L(560, 860, 1360, 860, "gris", 5, en="c", t=0.6), T("0", 560, 910, tam=34, c="gris", w="sb", en="c", t=0.6), T("1", 1360, 910, tam=34, c="gris", w="sb", en="c", t=0.6),
    Bc(560 + 800 * 0.587, 860, 34, 34, "azul", None, 0, 17, en="c", t=0.9, ent="pop"))
v.b("¿Cuál uso? [[a]]Para PCA y t-SNE, lo habitual es estandarizar. [[b]]{Min-max|Min max} sirve cuando necesitas límites fijos, por ejemplo en redes neuronales. [[c]]Y ojo: ninguna de las dos exige que la variable sea normal.",
    CL(), tarjetas([(560, "violeta", "z-score", "lo habitual para PCA y t-SNE", "a"), (1360, "azul", "min-max", "límites fijos (redes neuronales)", "b")], y=460, w=700, h=340),
    P("ninguna exige que la variable sea normal", 960, 800, "ambar", 48, en="c"))

# ---- 5. Sin trampa
v.b("Ahora, lo más importante: ¿con qué datos se calculan la media y la desviación? [[a]]Solo con el entrenamiento. [[b]]Esa es la lección de la clase treinta sobre la fuga de datos.",
    CL(), T("¿con qué filas se calcula la media y la desviación?", 960, 230, tam=48, c="tinta", ent="subir"),
    P("solo con el entrenamiento", 960, 430, "verde", 60, en="a"), P("Clase 30: fuga de datos", 960, 600, "gris", 46, en="b"), sec="6. Sin hacer trampa")
v.b("Los mil quinientos clientes se dividen en mil ciento veinticinco para entrenar, y trescientos setenta y cinco para la prueba. [[a]]El StandardScaler aprende la media y la desviación con fit, solo sobre el entrenamiento. [[b]]Después, con transform, aplica esos mismos números a la prueba.",
    CL(), Bx(260, 300, 260 + 1050, 460, "verdeclaro", "verde", 4, 20, ent="fade"), T("entrenamiento · 1.125", 260 + 525, 380, tam=44, c="verde"),
    Bx(1340, 300, 1340 + 350, 460, "ambarclaro", "ambar", 4, 20, ent="fade", t=0.2), T("prueba · 375", 1515, 380, tam=44, c="ambar", t=0.2),
    L(785, 470, 785, 600, "verde", 6, True, en="a"), Bc(785, 680, 620, 150, "blanco", "violeta", 5, 26, sombra=True, en="a", ent="subir", t=0.2),
    T("fit: aprende media y desv.", 785, 680, tam=38, c="violeta", en="a", t=0.3),
    L(1100, 700, 1500, 480, "violeta", 6, True, en="b"), T("transform", 1400, 640, tam=38, c="violeta", w="sb", en="b", t=0.3),
    T("StandardScaler().fit(X_train)", 960, 900, tam=40, c="tinta", w="m", en="a", t=0.6))
v.b("Si en cambio ajustas el escalador con todas las filas, [[a]]la prueba ya influyó en la media y en la desviación. [[b]]Es fuga: el modelo vio algo de los datos con que lo vas a evaluar.",
    CL(), T("fit con todas las filas", 960, 330, tam=70, c="rojo", ent="subir"), K(1460, 330, "cruz", 80, t=0.5),
    P("la prueba influyó en media y desviación", 960, 540, "ambar", 50, en="a"), P("fuga de datos", 960, 720, "rojo", 60, en="b"))

# ---- 6. Error típico
HIS = [2, 6, 9, 7, 5, 3, 2, 1, 1]
v.b("Un error típico: [[a]]creer que estandarizar convierte la variable en una normal, en una campana.",
    CL(), T("«estandarizar la vuelve normal»", 960, 300, tam=70, c="tinta", id="mito", en="a"), sec="7. El error típico")
v.b("No es así. [[a]]Mira esta variable con cola hacia la derecha. [[b]]Al estandarizar, cambian los números del eje, pero la forma es exactamente la misma.",
    U("mito", tachado=1.0, dur=0.6, t=0.1),
    *[Bx(300 + 70 * i, 860 - h * 40, 360 + 70 * i, 860, "violeta", None, 0, 6, grow="y", en="a", t=0.05 * i) for i, h in enumerate(HIS)],
    T("original", 600, 910, tam=34, c="violeta", w="sb", en="a"),
    L(980, 700, 1080, 700, "gris", 6, True, en="b"),
    *[Bx(1200 + 70 * i, 860 - h * 40, 1260 + 70 * i, 860, "teal", None, 0, 6, grow="y", en="b", t=0.05 * i) for i, h in enumerate(HIS)],
    T("estandarizada: misma forma", 1500, 910, tam=34, c="teal", w="sb", en="b"), ilustrativo(990, en="a"))

# ---- 7. Piensa
v.b("Piensa un segundo: [[a]]si aplicas PCA sin escalar, a segundos, con desviación dos mil novecientos sesenta y nueve, y a reclamos, con cero coma veintisiete, ¿qué variable se lleva el primer componente?",
    CL(), T("PCA sin escalar", 960, 260, tam=60, c="tinta", en="a"),
    P("segundos · desv. 2.969", 560, 460, "rojo", 50, en="a", t=0.3), P("reclamos · desv. 0.27", 1360, 460, "teal", 50, en="a", t=0.5),
    T("¿quién se lleva PC1?", 960, 680, tam=76, c="violeta", en="a", t=0.9), sec="8. Piensa un segundo", pausa=0.6)
v.b("Segundos, por lejos. [[a]]En el video siete lo veremos con los datos: sin escalar, segundos concentra el noventa y nueve coma dos por ciento de toda la varianza.",
    CL(), P("segundos, por lejos", 960, 380, "rojo", 70, ent="subir"), N(99.2, 960, 600, "{:.1f}", 140, "rojo", suf=" %", en="a", dur=0.9),
    T("de la varianza total, sin escalar (video 7)", 960, 760, tam=40, c="gris", w="sb", en="a", t=0.5))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]la variable de mayor magnitud domina varianzas y distancias. [[b]]Estandariza: z igual a x menos la media, dividido por la desviación. [[c]]Y ajusta el escalador solo con el entrenamiento.",
    CL(), P("la mayor magnitud domina varianzas y distancias", 960, 300, "rojo", 50, en="a"), P("z = (x − media) / desviación", 960, 480, "violeta", 54, en="b"),
    P("fit solo con entrenamiento", 960, 660, "verde", 54, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]codificar las variables categóricas, y construir variables nuevas.",
    CL(), siguiente(4, "Codificar categóricas y construir variables"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
