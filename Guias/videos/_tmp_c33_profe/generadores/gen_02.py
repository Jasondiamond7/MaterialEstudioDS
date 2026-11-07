import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, siguiente, ilustrativo, codigo, barras, alerta

OUT = out("guion_c33_02_limpiar.json")
v = V("Limpiar con criterio de negocio")
MM, NG, ED, NU, MI_ = D["minmax"], D["neg"], D["edad"], D["nulos"], D["medianas_imp"]

v.portada("Limpiar con criterio", "Ninguna técnica de selección arregla un dato imposible",
          "Primera parada: limpiar con criterio de negocio. Antes de elegir variables, tienes que poder confiar en ellas.")

# ---- 1. Hook: el termómetro
LEC = [180, 182, 179, -40]
v.b("Imagina un termómetro de horno. [[a]]Marca ciento ochenta, ciento ochenta y dos, ciento setenta y nueve, [[b]]y de pronto, menos cuarenta grados. [[c]]El horno no se congeló: el termómetro falló.",
    CL(), P("un termómetro de horno", 960, 220, "violeta", 54, ent="subir"),
    *[[Bc(390 + 380 * i, 470, 300, 200, "blanco", "rojo" if l < 0 else "teal", 5, 26, sombra=True, en=("a" if l > 0 else "b"), t=(0.35 * i if l > 0 else 0), ent="pop"),
       T(f"{l} °C".replace("-", "−"), 390 + 380 * i, 470, tam=66, c="rojo" if l < 0 else "teal", en=("a" if l > 0 else "b"), t=(0.35 * i if l > 0 else 0) + 0.1)] for i, l in enumerate(LEC)],
    P("no es un horno frío: es un sensor roto", 960, 720, "rojo", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="1. Un termómetro roto")
v.b("Si promedias esa lectura, [[a]]la media cae a ciento veinticinco coma veinticinco grados, y tu receta sale cruda. [[b]]La mediana, en cambio, queda en ciento setenta y nueve coma cinco: casi no se mueve.",
    T("(180 + 182 + 179 − 40) / 4 =", 760, 860, tam=44, c="tinta", w="sb", en="a"), N(125.25, 1330, 860, "{:.2f}", 60, "rojo", suf=" °C", en="a", t=0.4),
    T("mediana =", 1500, 950, tam=40, c="tinta", w="sb", anc="rm", en="b"), N(179.5, 1530, 950, "{:.1f}", 56, "verde", suf=" °C", en="b", t=0.3, anc="lm"))
v.b("Con los datos pasa lo mismo. [[a]]Si una variable tiene registros corruptos, su media, su varianza y su correlación con la fuga quedan contaminadas. [[b]]Y cualquier ranking que construyas sobre ella va a estar mal.",
    CL(), P("ninguna técnica de selección arregla un dato imposible", 960, 240, "violeta", 46, ent="subir"),
    *[[P(tx, x, 470, "rojo", 48, en="a", t=0.3 * j)] for j, (tx, x) in enumerate((("media", 560), ("varianza", 960), ("correlación", 1360)))],
    T("contaminadas", 960, 570, tam=40, c="rojo", w="sb", en="a", t=1.0),
    L(960, 620, 960, 700, "gris", 6, True, en="b"), P("ranking equivocado", 960, 770, "rojo", 56, en="b", t=0.3))

# ---- 2. Primer diagnóstico
FILAS = [("segundos", MM["segundos"], NG["segundos"][0]), ("frec_llamadas", MM["frec_llamadas"], NG["frec_llamadas"][0]),
         ("antiguedad", MM["antiguedad"], NG["antiguedad"][0]), ("frec_sms", MM["frec_sms"], NG["frec_sms"][0])]
v.b("El primer diagnóstico es describe: mínimos y máximos de cada columna. [[a]]Segundos tiene un mínimo de menos catorce mil seiscientos cuarenta y cinco. [[b]]Frecuencia de llamadas, menos ciento setenta y seis. [[c]]Antigüedad, menos cuarenta meses, y los mensajes, menos nueve.",
    CL(), T("d.describe().T[['min', 'max']]", 960, 190, tam=40, c="violeta", w="m", ent="subir"),
    T("variable", 420, 300, tam=32, c="gris", w="sb"), T("mínimo", 900, 300, tam=32, c="gris", w="sb"), T("máximo", 1300, 300, tam=32, c="gris", w="sb"),
    *[[Bc(960, 400 + 120 * i, 1300, 96, "blanco", "linea", 3, 20, en=k, ent="subir"), T(nm, 420, 400 + 120 * i, tam=38, c="tinta", w="m", en=k, t=0.1),
       T(f"{mm[0]:,.0f}".replace(",", ".").replace("-", "−"), 900, 400 + 120 * i, tam=44, c="rojo", en=k, t=0.2),
       T(f"{mm[1]:,.0f}".replace(",", "."), 1300, 400 + 120 * i, tam=44, c="tinta", en=k, t=0.3)]
      for i, ((nm, mm, _n), k) in enumerate(zip(FILAS, ["a", "b", "c", "c"]))],
    T("base cruda · 3.150 filas", 960, 920, tam=30, c="gris", w="sb", t=0.4), sec="2. El primer diagnóstico")
v.b("Una llamada negativa no existe. [[a]]¿Cuántas son? Dieciocho en segundos, doce en llamadas, nueve en antigüedad, una en mensajes, [[b]]y tres edades: una negativa y dos sobre cien. [[c]]En total, cuarenta y tres valores imposibles.",
    CL(), T("valores imposibles · base real", 960, 190, tam=44, c="tinta", ent="subir"),
    barras([NG["segundos"][0], NG["frec_llamadas"][0], NG["antiguedad"][0], NG["frec_sms"][0]], 420, 780, 170, 22, "rojo",
           ["segundos < 0", "llamadas < 0", "antigüedad < 0", "sms < 0"], fmt="{:.0f}", suf="", en="a", t0=0.0, dt=0.35, sep=270),
    barras([ED["neg"] + ED["mas100"]], 1500, 780, 170, 22, "ambar", ["edad < 0 o > 100"], fmt="{:.0f}", suf="", en="b"),
    L(300, 780, 1640, 780, "tinta", 4),
    P(f"total: {D['imposibles']}", 1500, 340, "rojo", 60, en="c"))
v.b("Y hay más que revisar. [[a]]Ochenta y tres nulos repartidos en seis columnas: cuarenta solo en segundos. [[b]]Y diecinueve filas duplicadas. Todo eso se mira antes de pensar en seleccionar.",
    CL(), T("lo que también se revisa", 960, 200, tam=44, c="gris", w="sb", ent="subir"),
    Bc(560, 500, 640, 340, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), N(D["nulos_tot"], 560, 460, "{:.0f}", 120, "azul", en="a", t=0.1),
    T(f"nulos · {NU['segundos']} en segundos", 560, 590, tam=36, c="azul", w="sb", en="a", t=0.3),
    Bc(1360, 500, 640, 340, "blanco", "violeta", 5, 30, sombra=True, en="b", ent="subir"), N(D["dup"], 1360, 460, "{:.0f}", 120, "violeta", en="b", t=0.1),
    T("filas duplicadas", 1360, 590, tam=36, c="violeta", w="sb", en="b", t=0.3),
    T("d.isna().sum()      d.duplicated().sum()", 960, 800, tam=36, c="tinta", w="m", en="b", t=0.6))

# ---- 3. La regla la pone el negocio
v.b("Ahora, la edad. [[a]]La regla de la clase es: edad válida entre quince y cien años. [[b]]Ese umbral no sale de los datos. Sale de saber que una telco no contrata a un cliente de doscientos treinta y tres años.",
    CL(), P("edad válida: 15 ≤ edad ≤ 100", 960, 260, "teal", 60, en="a"),
    T("«no sale de los datos: sale del negocio»", 960, 460, tam=54, c="tinta", en="b"),
    T("233 años", 960, 640, tam=100, c="rojo", en="b", t=0.6, id="e233"), sec="3. La regla la pone el negocio")
v.b("Pero mira el desglose. [[a]]Los casos claramente imposibles son solo tres: una edad negativa y dos sobre cien. [[b]]El rango, en cambio, deja fuera a ciento ocho clientes de cero a catorce años: el tres coma cuatro por ciento de la cartera.",
    CL(), T("fuera del rango [15, 100]", 960, 190, tam=44, c="gris", w="sb", ent="subir"),
    *[[Bc(x, 480, 420, 280, "blanco", c, 5, 30, sombra=True, en=k, ent="subir", t=t_), N(val, x, 450, "{:.0f}", 110, c, en=k, t=t_ + 0.1),
       T(lab, x, 560, tam=34, c=c, w="sb", en=k, t=t_ + 0.2)]
      for x, c, val, lab, k, t_ in ((400, "rojo", ED["neg"], f"negativa · mín {ED['min']:.0f}".replace("-", "−"), "a", 0.0), (880, "rojo", ED["mas100"], f"sobre 100 · máx {ED['max']:.0f}", "a", 0.4),
                                    (1440, "ambar", ED["menores"], "entre 0 y 14 años", "b", 0.0))],
    P(f"{ED['menores']} = {D['pct_menores']:.1f} % de la cartera", 1440, 720, "ambar", 44, en="b", t=0.8))
v.b("¿Son errores de carga, o planes contratados a nombre de un menor? [[a]]El dato no lo dice. [[b]]Esta decisión no se resuelve con estadística: se le pregunta al dueño del proceso, y mientras tanto se documenta.",
    CL(), T("¿error de carga?", 560, 330, tam=60, c="rojo", ent="subir"), T("¿plan a nombre de un menor?", 1360, 330, tam=60, c="verde", ent="subir", t=0.3),
    P("el dato no lo dice", 960, 500, "gris", 54, en="a"),
    P("pregunta al negocio  ·  documenta la decisión", 960, 680, "teal", 52, en="b"))

# ---- 4. NaN y no borrar filas
v.b("¿Y qué haces con un valor imposible? [[a]]Lo marcas como NaN: no sabes el valor real, y poner un cero sería inventar. [[b]]Después lo imputas.",
    CL(), T("antiguedad = −40", 960, 280, tam=70, c="rojo", w="m", ent="subir"),
    L(960, 360, 960, 450, "gris", 6, True, en="a"), P("NaN · «no lo sé»", 960, 520, "azul", 60, en="a", t=0.2),
    T("poner 0 = inventar un dato", 960, 650, tam=44, c="gris", w="sb", en="a", t=0.8, id="cero"), U("cero", tachado=1.0, en="a", t=1.6, dur=0.5),
    L(960, 720, 960, 800, "gris", 6, True, en="b"), P("imputar", 960, 870, "verde", 56, en="b", t=0.2), sec="4. NaN, nunca borrar la fila")
CEL = ["fallas", "reclamos", "antig.", "monto", "seg.", "llam.", "sms", "contac.", "plan", "estado", "edad", "clv"]
v.b("Y lo más importante, la advertencia que más repite el profesor: [[a]]no borres la fila. [[b]]Si el problema está en una sola columna, borrar la fila elimina la información de las otras once.",
    CL(), T("un cliente = una fila de 12 variables", 960, 220, tam=44, c="gris", w="sb", ent="subir"),
    *[[Bc(255 + 128 * j, 470, 116, 116, "rojoclaro" if j == 2 else "blanco", "rojo" if j == 2 else "teal", 4, 14, id=f"c{j}", t=0.05 * j),
       T(CEL[j], 255 + 128 * j, 470, tam=24, c="tinta", w="sb", id=f"ct{j}", t=0.05 * j + 0.05)] for j in range(12)],
    T("−40", 511, 560, tam=34, c="rojo", t=0.8),
    *[U(f"c{j}", en="b", t=0.6, dur=0.6, relleno="rojoclaro", borde="rojo") for j in range(12)],
    K(1800, 470, "cruz", 80, en="b", t=1.2), P("borras 11 variables buenas", 960, 700, "rojo", 52, en="b", t=1.4))

# ---- 5. Ejemplo: media o mediana
SEG = ["3.000", "3.200", "2.900", "3.100", "−14.645"]
v.b("Hagamos un ejemplo del apunte. [[a]]Cinco clientes con sus segundos: tres mil, tres mil doscientos, dos mil novecientos, tres mil cien, [[b]]y uno imposible: menos catorce mil seiscientos cuarenta y cinco.",
    CL(), T("ejemplo del apunte · segundos de 5 clientes", 960, 190, tam=40, c="gris", w="sb", ent="subir"),
    *[[Bc(330 + 315 * i, 360, 270, 140, "blanco", "rojo" if i == 4 else "azul", 5, 24, sombra=True, en=("a" if i < 4 else "b"), t=(0.3 * i if i < 4 else 0), ent="pop", id=f"s{i}"),
       T(s, 330 + 315 * i, 360, tam=50, c="rojo" if i == 4 else "azul", en=("a" if i < 4 else "b"), t=(0.3 * i if i < 4 else 0) + 0.1, id=f"st{i}")] for i, s in enumerate(SEG)],
    sec="5. Ejemplo: ¿media o mediana?")
v.b("Paso uno, la media con el error incluido. [[a]]Los cuatro buenos suman doce mil doscientos. [[b]]Menos catorce mil seiscientos cuarenta y cinco, da menos dos mil cuatrocientos cuarenta y cinco. [[c]]Dividido en cinco: menos cuatrocientos ochenta y nueve segundos. Una media negativa.",
    T("1 · media", 300, 560, tam=44, c="rojo", anc="lm"),
    T("3.000 + 3.200 + 2.900 + 3.100 = 12.200", 560, 560, tam=42, c="tinta", w="sb", anc="lm", en="a"),
    T("(12.200 − 14.645) / 5 =", 560, 660, tam=42, c="tinta", w="sb", anc="lm", en="b"),
    N(-489, 1180, 660, "{:.0f}", 64, "rojo", en="c", anc="lm"), K(1440, 660, "cruz", 54, en="c", t=0.5))
v.b("Paso dos, la mediana con el error incluido. [[a]]Ordenamos: el imposible queda al principio, y el del medio es tres mil. [[b]]Casi no se movió.",
    T("2 · mediana", 300, 770, tam=44, c="verde", anc="lm"),
    T("−14.645 · 2.900 · [3.000] · 3.100 · 3.200", 680, 770, tam=42, c="tinta", w="sb", anc="lm", en="a"),
    K(1580, 770, "check", 54, en="b"))
v.b("Paso tres: marcamos el imposible como NaN, y calculamos la mediana de los cuatro que quedan. [[a]]Tres mil más tres mil cien, dividido en dos: tres mil cincuenta. [[b]]Con ese valor se imputa el NaN.",
    U("s4", relleno="azulclaro", borde="azul", dur=0.5), U("st4", texto="NaN", color="azul", dur=0.3, t=0.2),
    T("3 · imputar", 300, 900, tam=44, c="azul", anc="lm"), T("(3.000 + 3.100) / 2 =", 600, 900, tam=42, c="tinta", w="sb", anc="lm", en="a"),
    T("3.050", 1080, 900, tam=60, c="azul", en="a", t=0.5, anc="lm"), U("st4", texto="3.050", en="b", dur=0.3), R(330 + 315 * 4, 360, 300, 170, "verde", 26, en="b", t=0.2))
v.b("En la base real pasa lo mismo, a otra escala. [[a]]Con los negativos y los nulos marcados, la mediana de segundos es tres mil treinta y uno coma cinco. [[b]]Ese es el valor que reciben los cincuenta y ocho casos marcados.",
    CL(), T("segundos · base real", 960, 220, tam=48, c="tinta", ent="subir"),
    Bc(560, 500, 640, 300, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), N(D["nan_tras_marcar"]["segundos"], 560, 470, "{:.0f}", 110, "rojo", en="a", t=0.1),
    T(f"marcados · {NU['segundos']} nulos + {NG['segundos'][0]} negativos", 560, 590, tam=32, c="rojo", w="sb", en="a", t=0.2),
    L(900, 500, 1020, 500, "gris", 7, True, en="b"),
    Bc(1360, 500, 640, 300, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), N(MI_["segundos"], 1360, 470, "{:.1f}", 100, "verde", en="b", t=0.1),
    T("mediana con que se imputa", 1360, 590, tam=32, c="verde", w="sb", en="b", t=0.2))
v.b("Y así se ve en código, como en la diapositiva de la clase. [[a]]Primero, los negativos a NaN. [[b]]Después, la edad fuera de rango, también a NaN. [[c]]Y al final, imputar cada columna con su mediana.",
    CL(), codigo(["for c in ['segundos', 'frec_llamadas', 'frec_sms', 'antiguedad', 'n_contactos']:",
                  "    df.loc[df[c] < 0, c] = np.nan",
                  "df.loc[(df.edad < 15) | (df.edad > 100), 'edad'] = np.nan",
                  "for c in df.columns:",
                  "    df[c] = df[c].fillna(df[c].median())"], y=470, tam=30, dt=0.0, en="a"),
    R(960, 400, 1600, 100, "rojo", 16, en="a", t=0.3), R(960, 470, 1600, 56, "ambar", 16, en="b"), R(960, 540, 1600, 100, "verde", 16, en="c"),
    T("resultado: (3150, 13) · tasa de fuga 0.157 · 0 nulos", 960, 760, tam=36, c="gris", w="sb", en="c", t=0.6))

# ---- 6. Error típico
v.b("El error típico: [[a]]ver un valor raro y borrarlo, o reemplazarlo sin preguntar de dónde viene. [[b]]El profesor lo ilustró con una anécdota: un pájaro posado sobre un sensor genera un outlier falso. [[c]]Un fraude, en cambio, genera un outlier real, y es justo lo que quieres detectar.",
    CL(), T("ver un valor raro → borrarlo", 960, 260, tam=64, c="tinta", ent="subir", id="brr"), U("brr", tachado=1.0, color="gris", en="a", t=1.2, dur=0.6),
    tarjetas([(560, "gris", "pájaro en el sensor", "outlier falso · error de medición", "b"), (1360, "rojo", "fraude", "outlier real · la señal que buscas", "c")], y=600, w=640, h=320),
    sec="6. El error típico")
v.b("Por eso la pregunta antes de tocar un valor es siempre la misma: [[a]]¿de dónde viene este dato?",
    CL(), P("¿de dónde viene este dato?", 960, 480, "violeta", 80, en="a"))

# ---- 7. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un cliente tiene antigüedad menos cuarenta meses. ¿Lo pones en cero, borras la fila, o qué haces?",
    CL(), T("antiguedad = −40", 960, 330, tam=80, c="rojo", w="m", en="a"), T("¿cero? ¿borrar la fila? ¿otra cosa?", 960, 560, tam=60, c="tinta", en="a", t=0.8),
    sec="7. Piensa un segundo", pausa=0.6)
v.b("Ni cero ni borrar. [[a]]Lo marcas como NaN y lo imputas con la mediana, que en esta base es treinta y cinco meses. [[b]]Un detalle para más adelante: la mediana usa todo el conjunto, así que en el modelo final la imputación va dentro de cada fold. Lo vemos en el video nueve.",
    CL(), P("ni cero ni borrar", 960, 230, "rojo", 60, ent="subir"),
    T("−40", 640, 450, tam=90, c="rojo", en="a"), L(780, 450, 900, 450, "gris", 7, True, en="a", t=0.3), T("NaN", 1050, 450, tam=90, c="azul", en="a", t=0.5),
    L(1190, 450, 1310, 450, "gris", 7, True, en="a", t=0.9), N(MI_["antiguedad"], 1420, 450, "{:.0f}", 90, "verde", en="a", t=1.1), T("meses (mediana)", 1420, 540, tam=30, c="verde", w="sb", en="a", t=1.3),
    P("imputar dentro de cada fold → video 9", 960, 760, "ambar", 46, en="b"))

# ---- 8. Para llevar
v.b("Para llevarte: [[a]]la limpieza va antes de la selección. [[b]]El umbral lo pone el negocio, no la estadística. [[c]]Imposible pasa a NaN y se imputa con la mediana, sin borrar la fila.",
    CL(), P("limpiar va antes de seleccionar", 960, 300, "teal", 54, en="a"), P("el umbral lo pone el negocio", 960, 480, "violeta", 54, en="b"),
    P("imposible → NaN → mediana · sin borrar filas", 960, 660, "azul", 50, en="c"), sec="8. Para llevar")
v.b("Siguiente parada: [[a]]construir variables, de doce a cincuenta.",
    CL(), siguiente(3, "Construir variables: de 12 a 50"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
