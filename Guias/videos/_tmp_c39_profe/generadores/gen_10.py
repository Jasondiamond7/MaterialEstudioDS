import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, barras_v)

OUT = out("guion_c39_10_kmodes.json")
v = V("K-modes para variables categóricas")
KO = D["kmodes"]
CO = KO["costo"]
G2 = KO["grupos"]["2"]
CA = KO["cats"]

v.portada("K-modes", "Agrupar categorías: modas en vez de medias, desacuerdos en vez de distancias",
          "Décima parada: K-modes. Qué cambia cuando las variables no son números, sino categorías.")

# ---- 1. Hook
v.b("Una pregunta tramposa: [[a]]¿cuál es el promedio entre App, Web y Sucursal? [[b]]No existe. Puedes codificarlas como uno, dos y tres, y calcular que el promedio es dos, la Web. Pero eso no significa nada. [[c]]Con variables categóricas, la media y la distancia euclidiana dejan de tener sentido. Necesitamos otro representante y otra forma de medir diferencias.",
    CL(), T("¿el promedio entre App, Web y Sucursal?", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "App", "código 1", "a"), (960, "violeta", "Web", "código 2", "a"), (1440, "teal", "Sucursal", "código 3", "a")],
             y=400, w=400, h=220, tam_t=50, tam_d=36),
    P("promedio de los códigos = 2 = «Web»: no significa nada", 960, 650, "rojo", 44, en="b"),
    P("hace falta otro representante y otra diferencia", 960, 800, "teal", 46, en="c"), diapo(32, en="a"), sec="1. No hay promedio de categorías")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Quieres describir el uniforme típico de los hinchas de un equipo en el estadio. [[b]]No promedias colores: miras la camiseta más común, el gorro más común, la bufanda más común. [[c]]Y para ver si una persona se parece al hincha típico, cuentas en cuántas prendas no coincide. Eso es K-modes: la moda como centro, y los desacuerdos como distancia.",
    CL(), T("el hincha típico", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "rojo", "camiseta", "la más común", "b"), (960, "rojo", "gorro", "el más común", "b"), (1440, "rojo", "bufanda", "la más común", "b")],
             y=420, w=400, h=220, tam_t=48, tam_d=36),
    P("centro = moda · distancia = número de desacuerdos", 960, 700, "violeta", 46, en="c"), analogia(1050), sec="2. Analogía: el hincha típico")

# ---- 3. El método
v.b("La diapositiva treinta y dos define K-modes en tres ideas. [[a]]El centro de cada grupo contiene la categoría más frecuente de cada variable: su moda. [[b]]La disimilitud entre una observación y un centro cuenta en cuántas variables tienen valores diferentes. [[c]]Y el algoritmo alterna, igual que K-means, entre asignar observaciones y actualizar modas.",
    CL(), T("K-modes para variables categóricas · diapositiva 32", 960, 140, tam=48, c="tinta", ent="subir"),
    T("• centro: la categoría más frecuente de cada variable", 300, 290, tam=44, c="azul", w="sb", anc="lm", en="a"),
    T("• disimilitud: número de variables con valores distintos", 300, 400, tam=44, c="violeta", w="sb", anc="lm", en="b"),
    T("• alternar: asignar observaciones y actualizar modas", 300, 510, tam=44, c="teal", w="sb", anc="lm", en="c"),
    caja_formula("d(x, m) = Σ 1( x_j ≠ m_j )   (cuántas variables difieren)", y=700, c="violeta", tam=44, w=1400, en="b", t=2.0),
    diapo(32, en="a"), sec="3. El método")
v.b("Una consecuencia elegante: [[a]]si cambias los códigos uno, dos y tres por nombres, la disimilitud no cambia. Solo importa si dos valores son iguales o distintos. [[b]]No se interpretan las diferencias numéricas entre códigos, y las categorías no necesitan ningún orden.",
    T("cambiar códigos por nombres no cambia la disimilitud", 960, 860, tam=40, c="teal", w="sb", en="a"),
    T("solo importa igual o distinto: sin orden", 960, 940, tam=36, c="gris", w="sb", en="b"))

# ---- 4. Asignación resuelta
v.b("Una asignación resuelta, de la diapositiva treinta y tres. [[a]]Un cliente tiene plan Gold, usa la App y vive en el Sur. [[b]]El centro A es Gold, App, Norte: coincide en plan y canal, y difiere en región. Cero más cero más uno: un desacuerdo. [[c]]El centro B es Silver, Web, Sur: difiere en plan y en canal, y coincide en región. Uno más uno más cero: dos desacuerdos. El cliente va al grupo A, porque uno es menor que dos.",
    CL(), T("desacuerdos: una asignación resuelta · diapositiva 33", 960, 130, tam=46, c="tinta", ent="subir"),
    fila(250, [("", 380), ("plan", 680), ("canal", 960), ("región", 1240), ("desacuerdos", 1560)], tams=[34] * 5, colores=["gris"] * 5),
    fila(350, [("cliente", 380), ("Gold", 680), ("App", 960), ("Sur", 1240), ("", 1560)], tams=[44] * 5, colores=["tinta"] * 5, en="a",
         fondo="linea", x0=260, x1=1720),
    fila(460, [("centro A", 380), ("Gold", 680), ("App", 960), ("Norte", 1240), ("0 + 0 + 1 = 1", 1560)], tams=[44] * 5,
         colores=["azul", "verde", "verde", "rojo", "azul"], en="b", fondo="azulclaro", x0=260, x1=1720),
    fila(570, [("centro B", 380), ("Silver", 680), ("Web", 960), ("Sur", 1240), ("1 + 1 + 0 = 2", 1560)], tams=[44] * 5,
         colores=["ambar", "rojo", "rojo", "verde", "ambar"], en="c", fondo="ambarclaro", x0=260, x1=1720),
    P("el cliente va al grupo A: 1 < 2", 960, 730, "azul", 48, en="c", t=4.0), diapo(33, en="a"), sec="4. Una asignación resuelta")

# ---- 5. Actualizar la moda
v.b("Ahora la actualización, de la diapositiva treinta y cuatro. [[a]]Un grupo tiene tres clientes: Gold, App, Norte; Gold, Web, Norte; y Silver, App, Sur. [[b]]La moda de cada columna: en plan, Gold, que aparece dos veces; en canal, App, dos veces; en región, Norte, dos veces. El centro es Gold, App, Norte. [[c]]El costo del grupo suma los desacuerdos de cada cliente con ese centro: cero, uno y dos. Total: tres desacuerdos.",
    CL(), T("actualización de una moda · diapositiva 34", 960, 130, tam=48, c="tinta", ent="subir"),
    fila(250, [("plan", 600), ("canal", 900), ("región", 1200), ("desacuerdos", 1520)], tams=[34] * 4, colores=["gris"] * 4),
    fila(340, [("Gold", 600), ("App", 900), ("Norte", 1200), ("0", 1520)], tams=[44] * 4, en="a", fondo="linea", x0=440, x1=1700),
    fila(440, [("Gold", 600), ("Web", 900), ("Norte", 1200), ("1", 1520)], tams=[44] * 4, en="a", t=0.4, fondo="linea", x0=440, x1=1700),
    fila(540, [("Silver", 600), ("App", 900), ("Sur", 1200), ("2", 1520)], tams=[44] * 4, en="a", t=0.8, fondo="linea", x0=440, x1=1700),
    fila(660, [("Gold", 600), ("App", 900), ("Norte", 1200)], tams=[48] * 3, colores=["violeta"] * 3, pesos=["b"] * 3, en="b", fondo="violetaclaro",
         x0=440, x1=1360),
    T("moda →", 420, 660, tam=36, c="violeta", w="b", anc="rm", en="b"),
    P("costo del grupo = 0 + 1 + 2 = 3", 960, 800, "teal", 48, en="c"), diapo(34, en="a"), sec="5. Actualizar la moda")
v.b("Dos detalles de la misma diapositiva: [[a]]la moda puede tener empates, y no siempre supera el cincuenta por ciento. Si tres clientes usan App, Web y Sucursal, cualquiera de las tres es moda, y ninguna pasa de un tercio.",
    T("empates posibles · la moda no siempre supera el 50 %", 960, 920, tam=40, c="ambar", w="sb", en="a"))

# ---- 6. Categorías de los vinos
v.b("Para practicar con datos de la clase, el profesor convirtió tres mediciones de los vinos en categorías. [[a]]Alcohol: bajo si es menor que trece, alto si es trece o más; no hay medio. [[b]]Flavonoides: bajo si es menor que 1.5, medio entre 1.5 y 3, alto desde 3. [[c]]Color: bajo si es menor que 4, medio entre 4 y 6.5, alto desde 6.5. Son cortes docentes, no umbrales de calidad enológica.",
    CL(), T("categorías para el ejemplo de vinos · diapositiva 35", 960, 130, tam=48, c="tinta", ent="subir"),
    fila(250, [("variable", 420), ("bajo", 800), ("medio", 1180), ("alto", 1560)], tams=[34] * 4, colores=["gris"] * 4),
    fila(350, [("alcohol", 420), ("x < 13", 800), ("no se usa", 1180), ("x ≥ 13", 1560)], tams=[42] * 4, en="a", fondo="linea", x0=240, x1=1740),
    fila(460, [("flavonoides", 420), ("x < 1.5", 800), ("1.5 ≤ x < 3", 1180), ("x ≥ 3", 1560)], tams=[42] * 4, en="b", fondo="linea", x0=240, x1=1740),
    fila(570, [("color", 420), ("x < 4", 800), ("4 ≤ x < 6.5", 1180), ("x ≥ 6.5", 1560)], tams=[42] * 4, en="c", fondo="linea", x0=240, x1=1740),
    P("cortes docentes, no umbrales de calidad", 960, 720, "ambar", 46, en="c", t=3.0), diapo(35, en="a"), sec="6. Los vinos en categorías")
v.b("Con esos cortes, [[a]]hay noventa y dos vinos de alcohol alto y ochenta y cinco de alcohol bajo; en color, setenta bajos, sesenta y siete medios y cuarenta altos. [[b]]Un detalle fino de la diapositiva treinta y seis: los intervalos son cerrados por la izquierda, así que un valor de flavonoides igual a 1.5 cae en medio. En estos datos hay exactamente un vino con flavonoides 1.5, y uno con color 4: los dos van a medio. [[c]]Y dos advertencias: categorizar pierde información de las mediciones originales, y aquí las categorías se comparan solo por igualdad, sin usar su orden.",
    CL(), T(f"alcohol: alto {CA['alcohol']['alto']} · bajo {CA['alcohol']['bajo']}   ·   color: bajo {CA['color']['bajo']} · medio {CA['color']['medio']} · alto {CA['color']['alto']}",
            960, 250, tam=40, c="tinta", w="sb", en="a"),
    T("un vino con flavonoides = 1.5 y uno con color = 4 → ambos «medio»", 960, 400, tam=40, c="violeta", w="sb", en="b"),
    P("categorizar pierde información", 960, 560, "rojo", 46, en="c"), P("se compara por igualdad, sin orden", 960, 690, "teal", 46, en="c", t=1.5),
    adicional(1050, en="a", texto="Conteos: cálculo adicional con los cortes de la clase"), diapo(36, en="b", y=1010))

# ---- 7. El código y el costo
GK = "gk10"
v.b("En Python, con el paquete K-modes de la diapositiva treinta y nueve, [[a]]se ajusta K-modes para k de uno a ocho, con veinte inicios cada uno, y se guarda el costo total. [[b]]El costo baja de 283 desacuerdos con un grupo, a 148, 102, 75 con cuatro grupos, y sigue: 58, 48, 38 y 26. [[c]]No hay un codo claro. La clase usa {k|, ka,} igual a cuatro solo para ilustrar perfiles, sin afirmar que sea el único valor adecuado. Y recuerda: este costo cuenta desacuerdos; no es una suma de cuadrados.",
    CL(), T("costo de K-modes · diapositivas 39 y 40", 960, 130, tam=48, c="tinta", ent="subir"),
    codigo(['KModes(n_clusters=k, init="Huang", n_init=20, max_iter=100, random_state=2026)'], y=215, w=1700, tam=32, en="a"),
    G(500, 390, 1200, 900, 0.5, 8.5, 0, 300, [dict(xs=list(range(1, 9)), ys=CO, color="violeta", grosor=6, marcadores=11, revela=0.0)],
      xticks=list(range(1, 9)), yticks=[0, 100, 200, 300], yfmt="{:.0f}", xlabel="número de grupos k", ylabel="costo", id=GK, tam_txt=26),
    U(GK, **{"series.0.revela": 1.0}, en="b", dur=2.5), R(4, CO[3], 66, 66, "ambar", r=33, ref=GK, en="c"),
    T("k = 4: 75 desacuerdos", 1500, 480, tam=44, c="ambar", w="b", en="c"),
    T("sin codo claro", 1500, 570, tam=40, c="rojo", w="sb", en="c", t=1.0),
    T("costo = desacuerdos, no cuadrados", 1500, 660, tam=36, c="gris", w="sb", en="c", t=3.5), real(1050, en="b",
    texto="Recalculado con el código de las diapositivas (kmodes 0.12)"), sec="7. El código y el costo")

# ---- 8. Un perfil categórico
v.b("Leamos un perfil, el de la diapositiva cuarenta y uno: el grupo dos de K-modes, con cuarenta y ocho vinos. [[a]]Su centro modal es alcohol alto, flavonoides bajos y color alto. [[b]]Pero mira el color por dentro: cuarenta vinos tienen color alto, el 83.3 por ciento; siete tienen color medio, el 14.6; y uno tiene color bajo, el 2.1. [[c]]El perfil resume frecuencias. No significa que todos los vinos del grupo tengan color alto.",
    CL(), T("lectura de un perfil categórico · diapositiva 41", 960, 130, tam=48, c="tinta", ent="subir"),
    P("centro modal: alcohol alto · flavonoides bajos · color alto", 960, 250, "violeta", 42, en="a"),
    barras_v([("alto", G2["color"]["alto"], "rojo", "b"), ("medio", G2["color"]["medio"], "ambar", "b"), ("bajo", G2["color"]["bajo"], "azul", "b")],
             x0=560, paso=400, base=780, escala=9, w=200, fmt="{:.0f}"),
    T("40 / 48 = 83.3 %", 560, 860, tam=36, c="rojo", w="sb", en="b", t=0.5), T("7 / 48 = 14.6 %", 960, 860, tam=36, c="ambar", w="sb", en="b", t=0.8),
    T("1 / 48 = 2.1 %", 1360, 860, tam=36, c="azul", w="sb", en="b", t=1.1),
    P("el perfil resume frecuencias", 960, 960, "teal", 40, en="c"), diapo(41, en="a"), sec="8. Un perfil categórico")

# ---- 9. Alerta
v.b("Una precisión sobre la grabación. [[a]]Al ver la curva de costo, se dijo: me parece que el método debería haber elegido en torno a cuatro, pero sabemos que son tres. [[b]]Ese sabemos que son tres viene de las variedades reales, una etiqueta que en un problema de clustering real no existe. [[c]]Y además, K-modes aquí solo ve tres mediciones cortadas en dos o tres categorías: apenas dieciocho perfiles posibles. No hay razón para esperar que recupere las tres variedades.",
    CL(), alerta(160),
    cita("me parece que el método debería haber elegido en torno de 4, pero sabemos que son 3", y=300, en="a"),
    T("«sabemos que son 3» usa la variedad real: en la práctica no existe", 960, 460, tam=42, c="rojo", w="sb", en="b"),
    T("2 × 3 × 3 = 18 perfiles posibles con estas categorías", 960, 600, tam=42, c="tinta", w="sb", en="c"),
    grabacion("1:48:00", en="a"), sec="9. Alerta: «sabemos que son 3»")

# ---- 10. Error típico
v.b("El error típico: [[a]]codificar las categorías con números y correr K-means como si nada. [[b]]Si App es uno, Web es dos y Sucursal es tres, K-means cree que App está más cerca de Web que de Sucursal, y que el promedio de App y Sucursal es Web. [[c]]Esas distancias son un invento de la codificación. La diapositiva treinta y siete lo dice: K-modes admite las etiquetas directamente; no hace falta codificarlas.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("codificar 1, 2, 3 y usar K-means", 410, 300, tam=46, c="rojo", anc="lm", en="a"),
    T("«App está más cerca de Web que de Sucursal»: inventado", 960, 450, tam=42, c="tinta", w="sb", en="b"),
    K(330, 620, "check", 70, en="c"), T("K-modes usa las etiquetas tal cual", 410, 620, tam=46, c="verde", anc="lm", en="c"),
    diapo(37, en="c"), sec="10. El error típico")

# ---- 11. Piensa
v.b("Piensa un segundo. [[a]]Un cliente es Silver, Web, Norte. Los centros son los mismos de antes: A, Gold, App, Norte; y B, Silver, Web, Sur. [[b]]¿A qué grupo va?",
    CL(), T("cliente: Silver · Web · Norte", 960, 280, tam=54, c="tinta", w="b", en="a"),
    T("A: Gold · App · Norte      B: Silver · Web · Sur", 960, 400, tam=46, c="gris", w="sb", en="a", t=1.0),
    P("¿grupo A o grupo B?", 960, 580, "violeta", 56, en="b"), sec="11. Piensa un segundo", pausa=0.6)
v.b("Con A: [[a]]difiere en plan y en canal, y coincide en región: uno más uno más cero, dos desacuerdos. [[b]]Con B: coincide en plan y canal, y difiere en región: un desacuerdo. Va al grupo B.",
    CL(), T("A: 1 + 1 + 0 = 2", 760, 340, tam=56, c="ambar", w="b", en="a"), T("B: 0 + 0 + 1 = 1", 1260, 340, tam=56, c="verde", w="b", en="b"),
    P("va al grupo B", 960, 540, "verde", 56, en="b", t=1.0), ilustrativo(1050))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]K-modes es K-means para categorías: el centro es la moda de cada variable, y la distancia cuenta desacuerdos. [[b]]Su costo es un número de desacuerdos, no una suma de cuadrados. [[c]]Y un perfil modal resume frecuencias: siempre conviene mirar las proporciones dentro de cada grupo.",
    CL(), P("centro = moda · distancia = desacuerdos", 960, 300, "violeta", 48, en="a"),
    P("costo = desacuerdos (no cuadrados)", 960, 460, "teal", 48, en="b"),
    P("mirar las proporciones dentro del grupo", 960, 620, "ambar", 48, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]la mayoría de los datos reales mezclan números y categorías. K-prototypes combina las dos ideas.",
    CL(), siguiente(11, "K-prototypes para variables mixtas"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
