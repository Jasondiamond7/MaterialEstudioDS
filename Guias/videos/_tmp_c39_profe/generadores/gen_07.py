import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, serie_puntos)

OUT = out("guion_c39_07_gap.json")
v = V("El estadístico Gap y los criterios en desacuerdo")
GP = D["gap"]
GA, ER = GP["gap"], GP["err"]

v.portada("El estadístico Gap", "Comparar con datos sin grupos… y decidir cuando los criterios discrepan",
          "Séptima parada: el estadístico Gap. Y una situación muy real: qué hacer cuando dos criterios recomiendan números distintos.")

# ---- 1. Hook
v.b("Aquí va una verdad incómoda: [[a]]K-means siempre entrega grupos. Aunque los datos no tengan ninguna estructura, aunque sean puro ruido, si le pides tres grupos, te da tres grupos. [[b]]Entonces, ¿cómo sabemos si los grupos que encontramos son más compactos de lo que saldría por azar? [[c]]Esa es la idea del Gap: comparar con datos simulados que no tienen grupos.",
    CL(), T("K-means siempre entrega grupos", 960, 150, tam=56, c="tinta", ent="subir"),
    P("incluso con datos sin estructura", 960, 320, "rojo", 48, en="a"),
    P("¿más compactos que por azar?", 960, 480, "ambar", 50, en="b"),
    P("Gap: comparar con datos simulados sin grupos", 960, 660, "teal", 48, en="c"), sec="1. Siempre hay grupos")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Sacaste un seis en una prueba de alternativas. ¿Es bueno? Depende. [[b]]Si alguien que responde al azar saca un dos, tu seis es muy bueno. Si el azar saca un cinco y medio, tu seis casi no dice nada. [[c]]El Gap hace eso: compara tu WSS con el WSS que saldría en datos al azar, y mira la distancia, la brecha, entre ambos. Gap significa brecha.",
    CL(), T("tu nota contra la nota del azar", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "azar saca 2", "tu 6 es muy bueno", "b"), (1360, "rojo", "azar saca 5.5", "tu 6 casi no dice nada", "b")],
             y=420, w=620, h=260, tam_t=50, tam_d=40),
    T("tu nota: 6", 960, 240, tam=44, c="tinta", w="b", en="a"),
    P("Gap = brecha entre tu WSS y el del azar", 960, 720, "teal", 46, en="c"), analogia(1050), sec="2. Analogía: la nota del azar")

# ---- 3. La definición
v.b("La definición de la diapositiva veintidós, de Tibshirani, Walther y Hastie, de dos mil uno. [[a]]Primero, se ajusta K-means a los datos reales para cada k, y se guarda su WSS. [[b]]Segundo, se simulan B bases de referencia, sin grupos, y se calcula su WSS para los mismos valores de k. [[c]]Tercero, se comparan en escala logarítmica: el Gap es el promedio del logaritmo del WSS de referencia, menos el logaritmo del WSS observado. Un Gap grande significa: mucho más compacto que el azar.",
    CL(), T("Gap: una referencia sin estructura de grupos · diapositiva 22", 960, 130, tam=44, c="tinta", ent="subir"),
    T("1. K-means a los datos reales para cada k → WSS observado", 300, 270, tam=40, c="azul", w="sb", anc="lm", en="a"),
    T("2. simular B bases sin grupos → WSS de referencia", 300, 360, tam=40, c="ambar", w="sb", anc="lm", en="b"),
    T("3. comparar en escala logarítmica", 300, 450, tam=40, c="violeta", w="sb", anc="lm", en="c"),
    caja_formula("Gap(k) = promedio de log WSS* (referencia) − log WSS (observado)", y=620, c="violeta", tam=42, w=1640, en="c", t=1.0),
    T("aquí: B = 30 bases uniformes en el rango de cada columna estandarizada", 960, 800, tam=36, c="gris", w="sb", en="c", t=3.0),
    diapo(22, en="a"), etiqueta("Tibshirani, Walther y Hastie (2001)", 1010, en="a"), sec="3. La definición")

# ---- 4. La regla
v.b("¿Y cómo se elige k con el Gap? No se toma el máximo. [[a]]La diapositiva veintitrés usa la regla de un error estándar: elegimos el primer k cuyo Gap sea mayor o igual que el Gap del siguiente k, menos el error estándar de ese siguiente k. [[b]]La idea: si dar un paso más mejora el Gap menos que el ruido de la simulación, no vale la pena darlo.",
    CL(), T("Gap: regla de un error estándar · diapositiva 23", 960, 140, tam=48, c="tinta", ent="subir"),
    caja_formula("primer k con  Gap(k) ≥ Gap(k + 1) − s(k + 1)", y=320, c="violeta", tam=54, w=1400, en="a"),
    P("si el paso siguiente mejora menos que el ruido, no se da", 960, 520, "teal", 44, en="b"), diapo(23, en="a"), sec="4. La regla de un error estándar")
v.b("El ejemplo hipotético de la diapositiva. [[a]]Gap de 0.40, 0.62 y 0.64 para {k|, ka,} igual a uno, dos y tres, con errores de 0.04, 0.05 y 0.04. [[b]]Para {k|, ka,} igual a uno: ¿0.40 es mayor o igual que 0.62 menos 0.05, que es 0.57? No: no cumple. [[c]]Para {k|, ka,} igual a dos: ¿0.62 es mayor o igual que 0.64 menos 0.04, que es 0.60? Sí. Elegimos dos, aunque el Gap más alto esté en tres.",
    CL(), T("ejemplo hipotético · diapositiva 23", 960, 140, tam=50, c="tinta", ent="subir"),
    fila(260, [("k", 600), ("Gap", 960), ("s", 1320)], tams=[36] * 3, colores=["gris"] * 3, en="a"),
    fila(350, [("1", 600), ("0.40", 960), ("0.04", 1320)], tams=[44] * 3, en="a", t=0.3, fondo="linea", x0=420, x1=1500, h=80),
    fila(440, [("2", 600), ("0.62", 960), ("0.05", 1320)], tams=[44] * 3, en="a", t=0.6, fondo="linea", x0=420, x1=1500, h=80),
    fila(530, [("3", 600), ("0.64", 960), ("0.04", 1320)], tams=[44] * 3, en="a", t=0.9, fondo="linea", x0=420, x1=1500, h=80),
    K(400, 680, "cruz", 54, en="b"), T("k = 1:  0.40 < 0.62 − 0.05 = 0.57  → no cumple", 460, 680, tam=42, c="rojo", w="sb", anc="lm", en="b"),
    K(400, 800, "check", 54, en="c"), T("k = 2:  0.62 ≥ 0.64 − 0.04 = 0.60  → elegimos 2", 460, 800, tam=42, c="verde", w="sb", anc="lm", en="c"),
    diapo(23, en="a"))

# ---- 5. Alerta
v.b("Aquí conviene una precisión sobre la grabación. [[a]]Después de resolver este ejemplo, se dijo que el algoritmo sigue corriendo hasta parar donde k es igual a cuatro. [[b]]En el ejemplo hipotético la regla se detiene en {k|, ka,} igual a dos; el cuatro es el resultado de los vinos, que vemos enseguida. [[c]]Y otra precisión: se dijo que se mira el próximo Gap menos el error estándar del k actual. El error que se resta es el del siguiente valor de k, como dice la fórmula de la diapositiva.",
    CL(), alerta(160),
    cita("vamos corriendo este algoritmo hasta que aquella situación para exactamente en una regla acá donde k es igual a 4", y=310, en="a", tam=38),
    P("ejemplo hipotético: la regla se detiene en k = 2", 960, 480, "verde", 44, en="b"),
    T("k = 4 es el resultado de los vinos", 960, 580, tam=40, c="tinta", w="sb", en="b", t=2.5),
    cita("él va a mirar exactamente el próximo gap menos un s_k", y=720, en="c", tam=38),
    T("se resta s(k + 1), el error del siguiente k", 960, 820, tam=40, c="verde", w="sb", en="c", t=2.5),
    grabacion("1:02:02 y 1:00:47", en="a"), sec="5. Alerta: el ejemplo y s(k + 1)")

# ---- 6. Los vinos
GG = "gg7"
v.b("Ahora los vinos, con treinta simulaciones. [[a]]El Gap crece desde 0.800 con un grupo, a 1.015 con dos, 1.199 con tres, y 1.227 con cuatro. [[b]]Para {k|, ka,} igual a tres: ¿1.199 es mayor o igual que 1.227 menos 0.022, que es 1.205? No, por muy poco. [[c]]Para {k|, ka,} igual a cuatro: ¿1.227 es mayor o igual que 1.245 menos 0.021, que es 1.224? Sí. El Gap recomienda {k|, ka,} igual a cuatro, tanto en R como en Python.",
    CL(), T("cuando los criterios discrepan · diapositiva 24", 960, 130, tam=48, c="tinta", ent="subir"),
    G(330, 230, 1110, 860, 0.5, 8.5, 0.75, 1.35, [dict(xs=list(range(1, 9)), ys=GA, color="azul", grosor=6, marcadores=11, revela=0.0)],
      xticks=list(range(1, 9)), yticks=[0.8, 1.0, 1.2], yfmt="{:.1f}", xlabel="número de grupos k", ylabel="Gap", id=GG, tam_txt=28),
    U(GG, **{"series.0.revela": 1.0}, en="a", dur=2.0),
    T(f"k = 3:  {GA[2]:.3f} < {GA[3]:.3f} − {ER[3]:.3f} = {GA[3] - ER[3]:.3f}", 1180, 420, tam=36, c="rojo", w="sb", anc="lm", en="b"),
    T("no cumple (por poco)", 1180, 480, tam=34, c="rojo", w="sb", anc="lm", en="b", t=2.5),
    T(f"k = 4:  {GA[3]:.3f} ≥ {GA[4]:.3f} − {ER[4]:.3f} = {GA[4] - ER[4]:.3f}", 1180, 600, tam=36, c="verde", w="sb", anc="lm", en="c"),
    T("cumple → k = 4", 1180, 660, tam=40, c="verde", w="b", anc="lm", en="c", t=2.5),
    R(4, GA[3], 66, 66, "verde", r=33, ref=GG, en="c", t=2.5), real(1050, en="a"), sec="6. El Gap en los vinos")
v.b("Fíjate en la forma de la curva: [[a]]el Gap sigue subiendo hasta {k|, ka,} igual a ocho. Si eligiéramos el máximo, tendríamos ocho grupos. [[b]]La regla de un error estándar frena antes, donde la mejora deja de superar el ruido de la simulación. [[c]]Y el profesor lo notó: de tres a cuatro el Gap cambia poco. Con otra semilla, más simulaciones o otra referencia, la recomendación podría moverse.",
    P("el máximo estaría en k = 8", 1480, 780, "rojo", 38, en="a"),
    P("la regla frena donde la mejora ≈ ruido", 1480, 870, "teal", 34, en="b"),
    T("otra semilla, B o referencia podría cambiarlo", 1480, 950, tam=30, c="gris", w="sb", en="c"))

# ---- 7. Criterios en desacuerdo
v.b("Entonces tenemos un desacuerdo. [[a]]El codo sugiere tres. La silueta, tres. El Gap, con esta referencia y treinta simulaciones, cuatro. [[b]]¿Qué se hace? La diapositiva veinticuatro decide: se trabaja con {k|, ka,} igual a tres para describir perfiles, y se reconoce {k|, ka,} igual a cuatro como alternativa. [[c]]No hay un k verdadero garantizado. Los criterios son evidencia, y la decisión final también considera la interpretación y el propósito.",
    CL(), T("tres criterios, dos respuestas", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "codo", "k = 3", "a"), (960, "ambar", "silueta", "k = 3", "a"), (1440, "violeta", "Gap", "k = 4", "a")],
             y=400, w=400, h=240, tam_t=48, tam_d=56),
    P("se trabaja con k = 3; k = 4 queda como alternativa", 960, 650, "teal", 46, en="b"),
    P("no hay un k verdadero garantizado", 960, 790, "rojo", 46, en="c"), diapo(24, en="b"), sec="7. Criterios en desacuerdo")

# ---- 8. En código
v.b("En el código de la clase, el Gap son pocas líneas. [[a]]Se generan treinta bases uniformes entre el mínimo y el máximo de cada columna estandarizada, y para cada una se ajusta K-means con veinte inicios. [[b]]Se guarda el logaritmo del WSS de cada simulación. [[c]]El Gap es el promedio de esos logaritmos menos el logaritmo del WSS real. Y el error estándar es la desviación de las simulaciones, multiplicada por la raíz de uno más uno sobre B.",
    CL(), T("complemento: Gap en Python · diapositivas 56 y 57", 960, 130, tam=48, c="tinta", ent="subir"),
    codigo(["referencia = rng.uniform(z.min(), z.max(), size=z.shape)", "modelo = KMeans(n_clusters=k, n_init=20, ...).fit(referencia)",
            "log_wss_ref[b, k - 1] = np.log(modelo.inertia_)", "gap = log_wss_ref.mean(axis=0) - np.log(wss)",
            "error_gap = log_wss_ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)"], y=440, w=1700, tam=34, ids=[f"c7{i}" for i in range(5)]),
    R(960, 440 - 52.7 * 2, 1660, 56, "ambar", r=12, en="a", id="r71"), X("r71", en="b"), R(960, 440, 1660, 56, "ambar", r=12, en="b", id="r72"),
    X("r72", en="c"), R(960, 440 + 52.7 * 1.5, 1660, 110, "ambar", r=12, en="c"), diapo(56, en="a"), sec="8. En código")

# ---- 9. Error típico
v.b("El error típico con el Gap: [[a]]elegir el k con el Gap más alto. En los vinos daría ocho. [[b]]El otro error es olvidar que la referencia y el número de simulaciones forman parte del criterio: el Gap no es una propiedad fija de los datos, depende de con qué los comparas.",
    CL(), K(330, 320, "cruz", 70, en="a"), T("elegir el máximo del Gap (aquí: k = 8)", 410, 320, tam=46, c="rojo", anc="lm", en="a"),
    K(330, 480, "cruz", 70, en="b"), T("olvidar que la referencia es parte del criterio", 410, 480, tam=46, c="rojo", anc="lm", en="b"),
    K(330, 640, "check", 70, en="b", t=3.0), T("regla de un error estándar + reportar B y referencia", 410, 640, tam=44, c="verde", anc="lm", en="b", t=3.0),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Gap de 0.50, 0.70, 0.75 y 0.76 para k de uno a cuatro, con errores de 0.03, 0.04, 0.06 y 0.05. [[b]]¿Qué k elige la regla de un error estándar?",
    CL(), T("Gap: 0.50 · 0.70 · 0.75 · 0.76", 960, 300, tam=54, c="tinta", w="b", en="a"),
    T("s: 0.03 · 0.04 · 0.06 · 0.05", 960, 400, tam=50, c="gris", w="sb", en="a"),
    P("¿qué k elige la regla?", 960, 580, "violeta", 56, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Para {k|, ka,} igual a uno: [[a]]¿0.50 es mayor o igual que 0.70 menos 0.04, o sea 0.66? No. [[b]]Para {k|, ka,} igual a dos: ¿0.70 es mayor o igual que 0.75 menos 0.06, o sea 0.69? Sí. La regla elige dos, aunque el Gap siga subiendo hasta cuatro.",
    CL(), K(400, 330, "cruz", 54, en="a"), T("k = 1:  0.50 < 0.70 − 0.04 = 0.66", 460, 330, tam=46, c="rojo", w="sb", anc="lm", en="a"),
    K(400, 470, "check", 54, en="b"), T("k = 2:  0.70 ≥ 0.75 − 0.06 = 0.69  → k = 2", 460, 470, tam=46, c="verde", w="sb", anc="lm", en="b"),
    ilustrativo(1050))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el Gap compara el logaritmo del WSS con el de datos simulados sin grupos. [[b]]Se elige con la regla de un error estándar, no con el máximo; en los vinos, k igual a cuatro. [[c]]Y cuando los criterios discrepan, se elige con criterio, y se reporta la alternativa: aquí, tres para describir, y cuatro como opción.",
    CL(), P("Gap: WSS real contra WSS de datos sin grupos", 960, 300, "violeta", 46, en="a"),
    P("regla de un error estándar (vinos: k = 4)", 960, 460, "teal", 46, en="b"),
    P("criterios en desacuerdo: decidir y reportar la alternativa", 960, 620, "ambar", 44, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]ya tenemos tres grupos. ¿Qué vinos hay en cada uno? Los perfiles, y por qué el número de grupo es solo una etiqueta.",
    CL(), siguiente(8, "Etiquetas y perfiles de los grupos"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
