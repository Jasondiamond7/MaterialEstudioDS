import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, flujo, grupos_series, serie_puntos, revelar, cruz, GCOL)

OUT = out("guion_c39_04_iteraciones.json")
v = V("Las iteraciones de K-means y los inicios")
IT = D["iter2d"]
IN = D["init"]
KM = D["km"]
X, Y = IT["x"], IT["y"]
H = IT["hist"]


def asignar(cen):
    return [min(range(3), key=lambda g: (x - cen[g][0]) ** 2 + (y - cen[g][1]) ** 2) for x, y in zip(X, Y)]


v.portada("Las iteraciones de K-means", "Asignar, recalcular, repetir… y probar muchos inicios",
          "Cuarta parada: cómo encuentra K-means los grupos. Un procedimiento de dos pasos que se repite, y un detalle que lo cambia todo: el punto de partida.")

# ---- 1. Hook
v.b("No existe una fórmula que te entregue directamente los mejores grupos. [[a]]Con ciento setenta y siete vinos y tres grupos hay una cantidad astronómica de particiones posibles: no se pueden revisar todas. [[b]]Lo que hace K-means es más modesto, y muy ingenioso: parte de un intento, lo corrige, y repite hasta que ya no hay nada que corregir.",
    CL(), T("no hay fórmula directa", 960, 150, tam=56, c="tinta", ent="subir"),
    P("particiones posibles: demasiadas para revisarlas todas", 960, 380, "rojo", 46, en="a"),
    P("intentar → corregir → repetir", 960, 560, "teal", 56, en="b"), sec="1. Sin fórmula directa")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Tienes tres carritos de helados para una playa llena de gente. Los pones en tres lugares cualquiera. [[b]]Cada bañista camina al carrito más cercano. [[c]]Después mueves cada carrito al centro de la gente que lo eligió. Con los carritos en su nuevo lugar, algunos bañistas cambian de carrito. Repites, hasta que nadie cambia. Ese es K-means.",
    CL(), T("tres carritos en la playa", 960, 150, tam=56, c="tinta", ent="subir"),
    flujo([("poner 3 carritos al azar", "ambar", "a"), ("cada bañista va al más cercano", "azul", "b"), ("mover cada carrito al centro de su gente", "teal", "c"),
           ("repetir hasta que nadie cambie", "violeta", "c")], y=480, w=360, h=230, x0=300, paso=440, tam=36),
    analogia(1050), sec="2. Analogía: los carritos")

# ---- 3. El algoritmo
v.b("Formalmente, la diapositiva doce lo escribe en cuatro pasos. [[a]]Uno: fijar k y elegir los centros iniciales. [[b]]Dos: asignar cada observación al centro más cercano. Tres: recalcular cada centro como la media de su grupo. [[c]]Cuatro: repetir la asignación y la actualización hasta que el ajuste se estabilice.",
    CL(), T("iteraciones de K-means · diapositiva 12", 960, 150, tam=52, c="tinta", ent="subir"),
    T("1. fijar k y elegir centros iniciales", 360, 310, tam=46, c="tinta", w="sb", anc="lm", en="a"),
    T("2. asignar cada observación al centro más cercano", 360, 420, tam=46, c="azul", w="sb", anc="lm", en="b"),
    T("3. recalcular cada centro como la media de su grupo", 360, 530, tam=46, c="teal", w="sb", anc="lm", en="b", t=2.5),
    T("4. repetir hasta estabilizar", 360, 640, tam=46, c="violeta", w="sb", anc="lm", en="c"), diapo(12, en="a"), sec="3. Los cuatro pasos")
v.b("Hay una garantía y una trampa. [[a]]La garantía: cada paso reduce el WSS o lo deja igual; nunca lo empeora. Por eso el algoritmo siempre se detiene. [[b]]La trampa: se detiene en una solución local, la mejor alrededor del punto de partida, que no tiene por qué ser la mejor de todas.",
    P("cada paso baja o mantiene el WSS", 960, 790, "verde", 44, en="a"),
    P("pero la solución puede ser local", 960, 900, "rojo", 44, en="b"))

# ---- 4. Una iteración resuelta
v.b("Una iteración resuelta, de la diapositiva trece. [[a]]Datos: uno, dos, tres, ocho, nueve y diez. Fijamos {k|, ka,} igual a dos, con centros iniciales en uno y en ocho. [[b]]Distancias al cuadrado al centro uno: cero, uno, cuatro, cuarenta y nueve, sesenta y cuatro y ochenta y uno. Al centro ocho: cuarenta y nueve, treinta y seis, veinticinco, cero, uno y cuatro. [[c]]Cada dato va al centro con la distancia menor: uno, dos y tres al grupo uno; ocho, nueve y diez al grupo dos.",
    CL(), T("una iteración resuelta · diapositiva 13", 960, 130, tam=50, c="tinta", ent="subir"),
    fila(240, [("x", 560), ("(x − 1)²", 820), ("(x − 8)²", 1080), ("grupo", 1340)], tams=[34] * 4, colores=["gris"] * 4, en="a"),
    [fila(240 + 90 * (i + 1), [(str(x), 560), (str((x - 1) ** 2), 820), (str((x - 8) ** 2), 1080), ("1" if x < 5 else "2", 1340)],
          tams=[40] * 4, colores=["tinta", "azul" if x < 5 else "gris", "gris" if x < 5 else "ambar", "azul" if x < 5 else "ambar"],
          en="b" if True else None, t=0.25 * i, fondo=("azulclaro" if x < 5 else "ambarclaro"), x0=440, x1=1460, h=76)
     for i, x in enumerate([1, 2, 3, 8, 9, 10])],
    diapo(13, en="a"), sec="4. Una iteración a mano")
v.b("Ahora recalculamos los centros. [[a]]El nuevo centro uno es la media de uno, dos y tres: dos. [[b]]El nuevo centro dos, la media de ocho, nueve y diez: nueve. [[c]]Si repetimos la asignación con los centros dos y nueve, ningún dato cambia de grupo. El algoritmo convergió en la segunda iteración.",
    P("nuevo centro 1 = (1 + 2 + 3) / 3 = 2", 960, 880, "azul", 42, en="a"),
    P("nuevo centro 2 = (8 + 9 + 10) / 3 = 9", 960, 970, "ambar", 42, en="b"))

# ---- 5. En dos dimensiones con vinos reales
xr = (min(X) - 0.3, max(X) + 0.3)
yr = (min(Y) - 0.3, max(Y) + 0.3)
c0, c1, cf = H[0]["centros"], H[1]["centros"], H[-1]["centros"]
lab1 = asignar(c0)
labf = IT["lab_final"]
GI = "gi4"
v.b("Ahora con vinos de verdad, en solo dos variables para poder verlo: flavonoides y color, estandarizados. [[a]]Estos son los ciento setenta y siete vinos. [[b]]Elegimos tres vinos cualquiera como centros iniciales: las tres cruces. [[c]]Primera asignación: cada vino toma el color del centro más cercano.",
    CL(), T("K-means paso a paso con dos mediciones reales", 960, 130, tam=48, c="tinta", ent="subir"),
    G(300, 210, 1180, 900, xr[0], xr[1], yr[0], yr[1], [serie_puntos(X, Y, "gris", 7)] + grupos_series(X, Y, lab1, mk=7),
      xticks=[-1, 0, 1, 2, 3], yticks=[-1, 0, 1, 2, 3], yfmt="{:.0f}", xlabel="flavonoides (z)", ylabel="color (z)", id=GI, tam_txt=26),
    revelar(GI, 1, en="a", dur=1.2),
    [cruz(c[0], c[1], GCOL[g], 54, ref=GI, en="b", id=f"cz{g}") for g, c in enumerate(c0)],
    U(GI, **{"series.0.alfa": 0.0}, en="c", dur=0.5), revelar(GI, 4, en="c", desde=1, dur=1.4),
    T("iteración 1", 1500, 300, tam=44, c="tinta", w="b", en="c"),
    T(f"WSS = {H[0]['wss_asig']:.2f}", 1500, 380, tam=44, c="violeta", w="b", en="c", t=0.6, id="w1"),
    adicional(1050, en="a"), sec="5. Paso a paso con vinos reales")
v.b("Segundo paso: [[a]]cada cruz se mueve a la media de los vinos de su color. [[b]]Con los centros en su nuevo lugar, se reasigna: algunos vinos de la frontera cambian de color. [[c]]Y se repite. El WSS baja de 165.83 en la primera asignación a 105.50 en la segunda, y después a 103.24, 102.90, 102.78 y 102.58.",
    [U(f"cz{g}", x=c1[g][0], y=c1[g][1], en="a", dur=1.0) for g in range(3)],
    T("iteración 2", 1500, 480, tam=40, c="tinta", w="b", en="b"),
    T(f"WSS = {H[1]['wss_asig']:.2f}", 1500, 550, tam=40, c="violeta", w="b", en="b", t=0.5),
    T(" → ".join(f"{h['wss_asig']:.2f}" for h in H[2:6]), 1500, 660, tam=34, c="gris", w="sb", en="c", mw=640))
v.b("En la iteración siete, ningún vino cambia de grupo: [[a]]el algoritmo convergió, con WSS 102.54 y grupos de cuarenta y dos, setenta y cuatro y sesenta y uno vinos. [[b]]Fíjate en la curva: la mayor parte de la mejora ocurre en las dos primeras iteraciones; después son ajustes finos.",
    [U(f"cz{g}", x=cf[g][0], y=cf[g][1], en="a", dur=1.0) for g in range(3)],
    G(300, 210, 1180, 900, xr[0], xr[1], yr[0], yr[1], grupos_series(X, Y, labf, mk=7, revela=1.0), xticks=[], yticks=[], id="gi4b", ejes=0.0,
      en="a", dur=1.0),
    T(f"iteración 7: WSS = {H[-1]['wss_asig']:.2f}", 1500, 760, tam=40, c="verde", w="b", en="a", t=0.4),
    T("tamaños: 42 · 74 · 61", 1500, 830, tam=36, c="tinta", w="sb", en="a", t=0.8),
    P("casi toda la mejora: en las 2 primeras iteraciones", 1500, 960, "teal", 30, en="b"))

# ---- 6. Convergencia y alerta
v.b("¿Cuántas iteraciones hacen falta? Hasta que nada cambie. [[a]]El script de la clase fija un máximo de cien iteraciones. [[b]]Con las trece variables y tres grupos, el ajuste elegido convergió en seis. [[c]]En la grabación se dijo que los algoritmos corren cincuenta mil o sesenta mil veces. Ojo: el cincuenta del código no son iteraciones, son cincuenta puntos de partida distintos, y cada uno converge en pocas iteraciones.",
    CL(), alerta(160, texto="precisión sobre la grabación"),
    T("máximo del script: max_iter = 100", 960, 300, tam=46, c="tinta", w="sb", en="a"),
    T("con k = 3 convergió en", 900, 420, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(KM["n_iter"], 930, 420, "{:.0f}", 70, "verde", anc="lm", en="b", suf=" iteraciones"),
    cita("varios algoritmos van a correr ahí 50 mil veces, 60 mil veces", y=580, en="c"),
    P("n_init = 50: cincuenta inicios, no cincuenta mil iteraciones", 960, 740, "ambar", 42, en="c", t=3.0),
    grabacion("0:43:41", en="c"), real(1010, en="b"), sec="6. ¿Cuántas iteraciones?")

# ---- 7. La inicialización importa
v.b("Ahora la trampa del óptimo local, con un ejemplo mínimo. [[a]]Cuatro puntos en las esquinas de un rectángulo largo: cero, cero; cero, uno; cuatro, cero; y cuatro, uno. Queremos dos grupos. [[b]]La partición natural es izquierda contra derecha: centros en cero, cero coma cinco y cuatro, cero coma cinco. Cada punto está a cero coma cinco de su centro: WSS igual a cuatro por cero coma veinticinco, uno. [[c]]Pero si partimos con un centro abajo y otro arriba, el algoritmo encuentra arriba contra abajo, con centros en dos, cero y dos, uno: WSS igual a cuatro por cuatro, dieciséis. Y se queda ahí: cada punto ya está más cerca de su propio centro.",
    CL(), T("un óptimo local con cuatro puntos", 960, 130, tam=52, c="tinta", ent="subir"),
    G(200, 260, 860, 700, -0.5, 4.5, -0.5, 1.5, [dict(xs=[0, 0, 4, 4.0001], ys=[0, 1, 0, 1], color="gris", linea=False, marcadores=14, revela=1.0, lapiz=False)],
      xticks=[0, 2, 4], yticks=[0, 1], yfmt="{:.0f}", id="gr4a", tam_txt=26, en="a"),
    T("izquierda / derecha", 530, 790, tam=38, c="verde", w="b", en="b"),
    [cruz(x, y, "verde", 40, ref="gr4a", en="b", t=0.3) for x, y in ((0, 0.5), (4, 0.5))],
    T("WSS = 4 × 0.25 = 1", 530, 860, tam=40, c="verde", w="sb", en="b", t=0.6),
    G(1060, 260, 1720, 700, -0.5, 4.5, -0.5, 1.5, [dict(xs=[0, 0, 4, 4.0001], ys=[0, 1, 0, 1], color="gris", linea=False, marcadores=14, revela=1.0, lapiz=False)],
      xticks=[0, 2, 4], yticks=[0, 1], yfmt="{:.0f}", id="gr4b", tam_txt=26, en="c"),
    T("abajo / arriba", 1390, 790, tam=38, c="rojo", w="b", en="c"),
    [cruz(x, y, "rojo", 40, ref="gr4b", en="c", t=0.3) for x, y in ((2, 0), (2, 1))],
    T("WSS = 4 × 4 = 16 (y no se mueve)", 1390, 860, tam=40, c="rojo", w="sb", en="c", t=0.6),
    ilustrativo(1050, en="a"), sec="7. Un óptimo local")

# ---- 8. 50 inicios con los vinos
WS = IN["wss"]
v.b("Con los vinos reales pasa lo mismo. [[a]]Corrimos K-means con tres grupos cincuenta veces, cada vez desde un único punto de partida al azar. [[b]]El WSS final va desde 1267.37, el mejor, hasta 1604.15, el peor: un veintisiete por ciento más. [[c]]Por eso la regla de la diapositiva quince: se prueban varios inicios, y se conserva el ajuste de menor WSS. Con cincuenta inicios, el script obtiene 1267.37.",
    CL(), T("la inicialización importa · diapositiva 15", 960, 130, tam=50, c="tinta", ent="subir"),
    G(330, 230, 1330, 840, 0, 51, 1240, 1640, [dict(xs=list(range(1, 51)), ys=WS, color="violeta", linea=False, marcadores=9, revela=0.0, lapiz=False),
                                             dict(xs=[0, 51], ys=[IN["min"], IN["min"]], color="verde", grosor=3, revela=0.0, lapiz=False)],
      xticks=[1, 10, 20, 30, 40, 50], yticks=[1300, 1400, 1500, 1600], yfmt="{:.0f}", xlabel="inicio", ylabel="WSS", id="gw4", tam_txt=26),
    U("gw4", **{"series.0.revela": 1.0}, en="a", dur=2.0),
    T("mejor:", 1620, 420, tam=40, c="gris", w="sb", en="b"), N(IN["min"], 1620, 490, "{:.2f}", 60, "verde", en="b"),
    T("peor:", 1620, 600, tam=40, c="gris", w="sb", en="b", t=1.0), N(IN["max"], 1620, 670, "{:.2f}", 60, "rojo", en="b", t=1.0),
    U("gw4", **{"series.1.revela": 1.0}, en="c", dur=0.8),
    P("se conserva el de menor WSS", 960, 960, "teal", 44, en="c"), adicional(1050, en="a",
    texto="Cálculo adicional: 50 inicios aleatorios sueltos (init = random, n_init = 1)"), sec="8. Cincuenta inicios")

# ---- 9. Alerta
v.b("Una precisión sobre lo dicho en clase. [[a]]Se dijo: después evaluamos cuáles fueron los clústeres que más aparecieron, y definimos esos como los finales. [[b]]El criterio no es el más frecuente: es el de menor WSS, como dice la diapositiva quince, y como el profesor agregó enseguida. [[c]]En nuestros cincuenta inicios coinciden: el mejor valor aparece dieciocho veces, más que cualquier otro. Pero no tiene por qué ser así: con k grande, el mejor ajuste puede aparecer muy pocas veces.",
    CL(), alerta(160),
    cita("evaluamos cuáles fueron los clústeres que más aparecieron, que más ocurrieron", y=300, en="a"),
    P("criterio: el menor WSS, no el más frecuente", 960, 460, "verde", 46, en="b"),
    T("aquí coinciden: 1267.37 aparece 18 de 50 veces", 960, 600, tam=42, c="tinta", w="sb", en="c"),
    T("con k grande, el mejor puede aparecer pocas veces", 960, 700, tam=42, c="rojo", w="sb", en="c", t=3.0),
    grabacion("0:41:13", en="a"), etiqueta("Diapositiva 15: «Conservamos el ajuste de menor WSS entre varias inicializaciones»", 1010, en="b"),
    sec="9. Alerta: el más frecuente")

# ---- 10. Error típico
v.b("El error típico: [[a]]correr K-means una sola vez, con una sola semilla, y reportar esos grupos. [[b]]Si tuviste mala suerte, puedes estar en un óptimo local mucho peor. [[c]]Lo correcto: muchos inicios, quedarse con el menor WSS, y fijar la semilla, el random_state, para que otra persona pueda repetir el cálculo.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("un solo inicio, una sola semilla", 410, 300, tam=46, c="rojo", anc="lm", en="a"),
    T("puede quedar en un óptimo local mucho peor", 960, 450, tam=42, c="tinta", w="sb", en="b"),
    K(330, 620, "check", 70, en="c"), T("muchos inicios + menor WSS + semilla fija", 410, 620, tam=46, c="verde", anc="lm", en="c"),
    sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Datos: cero, uno, cinco y seis. Dos grupos, con centros iniciales en cero y en uno. [[b]]¿Dónde terminan los centros, y cuánto vale el WSS final?",
    CL(), T("datos 0, 1, 5, 6   ·   centros iniciales 0 y 1", 960, 300, tam=54, c="tinta", w="b", en="a"),
    P("¿centros finales? ¿WSS final?", 960, 500, "violeta", 56, en="b"), sec="11. Piensa un segundo", pausa=0.6)
v.b("Iteración uno: [[a]]el cero va al centro cero; el uno, el cinco y el seis, al centro uno. Nuevos centros: cero, y cuatro. [[b]]Iteración dos: el uno ahora está más cerca del cero que del cuatro. Grupos: cero y uno; cinco y seis. Centros: cero coma cinco y cinco coma cinco. [[c]]Iteración tres: nada cambia. WSS: cuatro veces cero coma veinticinco, uno.",
    CL(), T("iteración 1: {0} y {1, 5, 6} → centros 0 y 4", 960, 300, tam=46, c="tinta", w="sb", en="a"),
    T("iteración 2: {0, 1} y {5, 6} → centros 0.5 y 5.5", 960, 430, tam=46, c="tinta", w="sb", en="b"),
    T("iteración 3: sin cambios · WSS = 4 × 0.25 =", 1100, 560, tam=46, c="tinta", w="sb", anc="rm", en="c"),
    N(1, 1130, 560, "{:.0f}", 66, "verde", anc="lm", en="c", t=2.5), ilustrativo(1050))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]K-means alterna dos pasos, asignar al centro más cercano y recalcular las medias, hasta que nada cambia. [[b]]Cada paso baja o mantiene el WSS, pero el resultado puede ser un óptimo local. [[c]]Por eso se prueban muchos inicios, cincuenta en la clase, y se conserva el de menor WSS.",
    CL(), P("asignar → recalcular medias → repetir", 960, 300, "azul", 50, en="a"),
    P("siempre converge, pero puede ser local", 960, 460, "rojo", 48, en="b"),
    P("muchos inicios: gana el menor WSS", 960, 620, "teal", 48, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]elegir el número de grupos. Empezamos por el método del codo.",
    CL(), siguiente(5, "Elegir k: el codo"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
