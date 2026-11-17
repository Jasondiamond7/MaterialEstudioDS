import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, barras_v)

OUT = out("guion_c39_05_codo.json")
v = V("Elegir k: el codo")
KM, CO = D["km"], D["codo"]
W = KM["wss"]
RED = CO["red"]

v.portada("Elegir k: el codo", "El WSS siempre baja: ¿dónde deja de valer la pena?",
          "Quinta parada: cuántos grupos. El primer criterio es el más visual de todos: el codo.")

# ---- 1. Hook
v.b("Si K-means minimiza el WSS, ¿por qué no elegimos el número de grupos que da el menor WSS? [[a]]Porque el WSS siempre baja cuando agregas grupos. [[b]]En el extremo, con un grupo por cada vino, cada vino es su propio centro, y el WSS vale exactamente cero. [[c]]Cero error, y ninguna información: eso no es agrupar. El costo de ajuste necesita un criterio adicional para elegir k.",
    CL(), T("¿elegir el k con menor WSS?", 960, 150, tam=56, c="tinta", ent="subir"),
    P("el WSS siempre baja al agregar grupos", 960, 330, "azul", 48, en="a"),
    T("177 grupos para 177 vinos → WSS =", 1040, 500, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(0, 1070, 500, "{:.0f}", 80, "rojo", anc="lm", en="b", t=1.5),
    P("hace falta un criterio adicional para elegir k", 960, 700, "violeta", 46, en="c"), diapo(18, en="a"), sec="1. El WSS siempre baja")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Un restaurante lleno contrata meseros. El primero cambia todo: de nadie atendiendo a alguien atendiendo. [[b]]El segundo también ayuda mucho. El tercero, bastante. [[c]]Pero el décimo mesero casi no reduce la espera. Hay un punto en que agregar uno más ya no compensa. El método del codo busca ese punto en la curva del WSS.",
    CL(), T("los meseros del restaurante", 960, 150, tam=56, c="tinta", ent="subir"),
    barras_v([("1", 40, "azul", "a"), ("2", 24, "azul", "b"), ("3", 16, "azul", "b"), ("4", 13, "gris", "c"), ("5", 11.5, "gris", "c"),
              ("10", 10.5, "gris", "c")], x0=480, paso=190, base=820, escala=12, w=130, fmt="{:.1f}"),
    T("espera (minutos) según el número de meseros", 960, 900, tam=32, c="gris", w="sb", en="a"),
    P("rendimientos decrecientes", 1500, 360, "violeta", 40, en="c"), analogia(1050), sec="2. Analogía: los meseros")

# ---- 3. Los criterios
v.b("La diapositiva dieciocho lista los criterios de la clase. [[a]]El codo: buscar dónde la disminución del WSS se vuelve cada vez menor. [[b]]La silueta: comparar la cercanía al propio grupo con la cercanía al grupo vecino. El Gap: contrastar la compacidad con una referencia simulada sin grupos. [[c]]Y siempre, revisar tamaños, estabilidad e interpretación. Ningún número reemplaza esa revisión.",
    CL(), T("elección del número de grupos · diapositiva 18", 960, 150, tam=50, c="tinta", ent="subir"),
    T("• codo: dónde la baja del WSS se hace pequeña", 300, 310, tam=44, c="azul", w="sb", anc="lm", en="a"),
    T("• silueta: propio grupo contra grupo vecino", 300, 420, tam=44, c="teal", w="sb", anc="lm", en="b"),
    T("• Gap: compacidad contra una referencia simulada", 300, 530, tam=44, c="violeta", w="sb", anc="lm", en="b", t=2.5),
    T("• revisar tamaños, estabilidad e interpretación", 300, 640, tam=44, c="ambar", w="sb", anc="lm", en="c"),
    diapo(18, en="a"), sec="3. Los criterios")

# ---- 4. El codo en los vinos
GC = "gc5"
v.b("Este es el codo de los vinos, recalculado con el script de Python. [[a]]Con un grupo, el WSS es igual al TSS: 2288. [[b]]Con dos grupos baja a 1643.90. Con tres, a 1267.37. [[c]]Y desde ahí la curva sigue bajando, pero mucho más despacio: 1164.91 con cuatro, 1092.18 con cinco, hasta 931.19 con ocho.",
    CL(), T("codo en los vinos · diapositiva 19", 960, 130, tam=50, c="tinta", ent="subir"),
    G(360, 230, 1240, 860, 0.5, 8.5, 800, 2400, [dict(xs=KM["k"], ys=W, color="azul", grosor=6, marcadores=11, revela=0.0)],
      xticks=list(range(1, 9)), yticks=[1000, 1500, 2000], yfmt="{:.0f}", xlabel="número de grupos k", ylabel="WSS", id=GC, tam_txt=28),
    U(GC, **{"series.0.revela": 0.02}, en="a", dur=0.4), U(GC, **{"series.0.revela": 0.29}, en="b", dur=1.2),
    U(GC, **{"series.0.revela": 1.0}, en="c", dur=2.0),
    R(3, W[2], 70, 70, "ambar", r=35, ref=GC, en="b", t=1.5),
    T("k = 3: 1267.37", 1480, 400, tam=46, c="ambar", w="b", en="b", t=1.5), real(1050, en="a"), sec="4. El codo de los vinos")
v.b("Para ver el codo con números, miremos cuánto baja el WSS en cada paso. [[a]]De uno a dos grupos, baja 644.1. De dos a tres, 376.5. [[b]]De tres a cuatro, solo 102.5. Y después, 72.7, 58.8, 62.9 y 39.2. [[c]]El quiebre está en tres: hasta ahí, cada grupo nuevo aporta mucho; después, poco. Por eso la diapositiva dice que {k|, ka,} igual a tres es un candidato interpretable, aunque la curva siga disminuyendo.",
    CL(), T("cuánto baja el WSS en cada paso", 960, 130, tam=52, c="tinta", ent="subir"),
    barras_v([(f"{i + 1}→{i + 2}", RED[i], "azul" if i < 2 else "gris", "a" if i < 2 else "b") for i in range(7)],
             x0=420, paso=180, base=820, escala=0.85, w=120, fmt="{:.1f}"),
    P("el quiebre está en k = 3", 1500, 330, "ambar", 44, en="c"), adicional(1050, en="a"))

# ---- 5. Lo que el codo no es
v.b("Dos advertencias de la misma diapositiva. [[a]]El codo no es una prueba formal: es una lectura visual, y dos personas pueden ver el codo en lugares distintos. [[b]]Y no garantiza un único candidato. Aquí, por ejemplo, también se podría defender cuatro. [[c]]Por eso no se usa solo: se contrasta con la silueta y con el Gap, que veremos ahora.",
    CL(), K(330, 320, "cruz", 70, en="a"), T("el codo no es una prueba formal", 410, 320, tam=46, c="rojo", anc="lm", en="a"),
    K(330, 470, "cruz", 70, en="b"), T("no garantiza un único candidato", 410, 470, tam=46, c="rojo", anc="lm", en="b"),
    K(330, 640, "check", 70, en="c"), T("se contrasta con la silueta y el Gap", 410, 640, tam=46, c="verde", anc="lm", en="c"),
    diapo(19, en="a"), sec="5. Lo que el codo no es")

# ---- 6. En código
v.b("El código de Python de la diapositiva veintiséis. [[a]]Un ciclo for recorre los valores de k. [[b]]En cada vuelta se ajusta K-means con cincuenta inicios, un máximo de cien iteraciones, la semilla 2026 y el algoritmo de Lloyd, el clásico de asignar y actualizar. [[c]]Se guarda el modelo, y su atributo inertia, que es justamente el WSS.",
    CL(), T("K-means para cada k en Python · diapositiva 26", 960, 130, tam=48, c="tinta", ent="subir"),
    codigo(["for k in range(1, 9):", "    modelo = KMeans(n_clusters=k, n_init=50, max_iter=100,", '                    random_state=2026, algorithm="lloyd", tol=1e-8)',
            "    modelo.fit(z)", "    modelos.append(modelo)", "    wss.append(modelo.inertia_)"], y=420, w=1680, tam=34, ids=[f"c5{i}" for i in range(6)]),
    R(960, 420 - 52.7 * 2.5, 1640, 56, "ambar", r=12, en="a", id="r51"), X("r51", en="b"),
    R(960, 420 - 52.7 * 1.5, 1640, 56, "ambar", r=12, en="b", id="r52"), X("r52", en="c"),
    R(960, 420 + 52.7 * 2.5, 1640, 56, "ambar", r=12, en="c"),
    P("inertia_ = WSS · labels_ = grupos, numerados desde cero", 960, 800, "teal", 40, en="c", t=1.5), diapo(26, en="a"), sec="6. En código")

v.b("El mismo ciclo en R, de la diapositiva veinticinco. [[a]]La función kmeans recibe los datos estandarizados, el número de centros, cincuenta inicios con {nstart|en start}, cien iteraciones como máximo, y el algoritmo de Lloyd. [[b]]En cada vuelta se guarda el WSS total del ajuste. [[c]]Y la semilla se fija una sola vez, al comienzo, con el valor 2026. Mismo procedimiento, distinto lenguaje: por eso R y Python llegan a la misma partición.",
    CL(), T("K-means para cada k en R · diapositiva 25", 960, 130, tam=48, c="tinta", ent="subir"),
    codigo(["set.seed(2026)", "for (k in 1:8) {", "  modelos[[k]] <- kmeans(z, centers = k, nstart = 50,", '                         iter.max = 100, algorithm = "Lloyd")',
            "  wss[k] <- modelos[[k]]$tot.withinss", "}"], y=430, w=1600, tam=34, c="azul"),
    R(960, 430 - 52.7 * 0.5, 1560, 110, "ambar", r=12, en="a", id="r5r"), X("r5r", en="b"), R(960, 430 + 52.7 * 1.5, 1560, 56, "ambar", r=12, en="b", id="r5s"),
    X("r5s", en="c"), R(960, 430 - 52.7 * 2.5, 1560, 56, "ambar", r=12, en="c"),
    P("misma partición en R y en Python", 960, 820, "teal", 44, en="c", t=2.0), diapo(25, en="a"))
v.b("Otra forma de leer el codo es en porcentaje. [[a]]Con dos grupos, el WSS es un 28.2 por ciento menor que el TSS. Con tres, un 44.6 por ciento menor. [[b]]Con cuatro, apenas sube a 49.1, y con ocho grupos llega solo a 59.3. [[c]]Es decir: el tercer grupo agrega más de dieciséis puntos porcentuales; los cinco grupos siguientes, juntos, menos de quince.",
    CL(), T("cuánto del TSS se explica con k grupos", 960, 150, tam=52, c="tinta", ent="subir"),
    barras_v([(str(k + 1), CO["pct_tss"][k], "azul" if k < 3 else "gris", "a" if k < 3 else "b") for k in range(1, 8)],
             x0=420, paso=180, base=820, escala=6, w=120, fmt="{:.1f}"),
    T("% de reducción del WSS respecto de un solo grupo", 960, 900, tam=32, c="gris", w="sb", en="a"),
    P("el tercer grupo: más de 16 puntos · los 5 siguientes: menos de 15", 1250, 300, "ambar", 32, en="c"), adicional(1050, en="a"))

# ---- 7. Alerta: range
v.b("Un detalle de Python que vale la pena fijar. [[a]]En la grabación, al mostrar este código, se dijo que el ciclo va a variar de uno hasta nueve clústeres. [[b]]Pero range de uno a nueve se detiene antes del nueve: entrega uno, dos, tres, y así hasta ocho. [[c]]Son ocho valores de k, igual que el uno a ocho del script de R. Por eso la tabla y el gráfico llegan hasta {k|, ka,} igual a ocho.",
    CL(), alerta(160, texto="precisión sobre la grabación"),
    cita("un K-means, que va a variar, por ejemplo, de uno hasta nueve clusters", y=300, en="a"),
    codigo(["list(range(1, 9))   # [1, 2, 3, 4, 5, 6, 7, 8]"], y=470, w=1300, tam=40, c="verde", en="b"),
    P("ocho valores de k: de 1 a 8 (en R: 1:8)", 960, 640, "teal", 46, en="c"),
    grabacion("1:08:54", en="a"), sec="7. Alerta: range(1, 9)")

# ---- 8. Error típico
v.b("El error típico: [[a]]elegir k mirando solo el valor más bajo del WSS. [[b]]Siempre ganará el k más grande que hayas probado: con ocho probados, ocho; con veinte, veinte. [[c]]El WSS se usa para comparar ganancias entre valores de k vecinos, no para elegir su mínimo.",
    CL(), K(330, 320, "cruz", 70, en="a"), T("elegir el k con menor WSS", 410, 320, tam=46, c="rojo", anc="lm", en="a"),
    T("siempre gana el k más grande que probaste", 960, 470, tam=44, c="tinta", w="sb", en="b"),
    K(330, 640, "check", 70, en="c"), T("comparar cuánto se gana al pasar de k a k + 1", 410, 640, tam=46, c="verde", anc="lm", en="c"),
    sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un análisis entrega estos WSS para k de uno a cinco: cien, sesenta, treinta y cinco, treinta y uno y veintiocho. [[b]]¿Dónde está el codo?",
    CL(), T("k:      1      2      3      4      5", 960, 300, tam=52, c="gris", w="sb", en="a"),
    T("WSS:  100    60    35    31    28", 960, 400, tam=56, c="tinta", w="b", en="a"),
    P("¿dónde está el codo?", 960, 580, "violeta", 56, en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("Las bajadas son: [[a]]cuarenta, veinticinco, cuatro y tres. [[b]]Hasta {k|, ka,} igual a tres cada grupo aporta mucho; el cuarto grupo apenas reduce cuatro. El codo está en {k|, ka,} igual a tres.",
    CL(), T("bajadas: 40 · 25 · 4 · 3", 960, 330, tam=60, c="tinta", w="b", en="a"),
    P("codo en k = 3", 960, 520, "ambar", 60, en="b"), ilustrativo(1050))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]el WSS siempre baja al agregar grupos, y con un grupo por fila vale cero. [[b]]El codo busca el punto donde la ganancia se vuelve pequeña; en los vinos, k igual a tres. [[c]]Es una lectura visual, no una prueba: hay que contrastarla con otros criterios.",
    CL(), P("el WSS siempre baja", 960, 300, "azul", 50, en="a"),
    P("codo: donde la ganancia se hace pequeña (vinos: k = 3)", 960, 460, "ambar", 44, en="b"),
    P("lectura visual, no prueba formal", 960, 620, "violeta", 48, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]la silueta, que mide si cada vino está más cerca de su grupo que del grupo vecino.",
    CL(), siguiente(6, "La silueta"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
