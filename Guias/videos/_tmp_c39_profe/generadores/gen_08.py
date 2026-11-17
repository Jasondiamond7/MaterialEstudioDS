import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, GCOL)

OUT = out("guion_c39_08_perfiles.json")
v = V("Etiquetas y perfiles de los grupos")
KM, DT = D["km"], D["datos"]
CE, PF, ME, SD = KM["centros"], KM["perfiles"], DT["media"], DT["sd"]
NV = KM["tamanos"]
VARS = DT["variables"]
CORTO = {"alcohol": "alcohol", "acido_malico": "ác. málico", "ceniza": "ceniza", "alcalinidad": "alcalin.", "magnesio": "magnesio",
         "fenoles": "fenoles", "flavonoides": "flavon.", "fenoles_no_flav": "fen. no fl.", "proantocianidinas": "proantoc.",
         "color": "color", "matiz": "matiz", "od280": "OD280", "prolina": "prolina"}


def tono(z):
    """Azul para valores positivos, rojo para negativos; intensidad segun |z| (hasta 1.3)."""
    a = min(abs(z) / 1.3, 1.0)
    base = (37, 99, 235) if z >= 0 else (220, 38, 38)
    c = tuple(int(255 - (255 - b) * a * 0.75) for b in base)
    return "#%02X%02X%02X" % c


v.portada("Perfiles de los grupos", "El número es una etiqueta; el perfil es el significado",
          "Octava parada: ya tenemos tres grupos de vinos. Ahora hay que darles significado, con sus perfiles.")

# ---- 1. Hook
v.b("K-means terminó, y entregó tres grupos de sesenta y dos, cincuenta y sesenta y cinco vinos. [[a]]¿Qué sabemos de ellos? Por ahora, nada: solo que el algoritmo los numeró uno, dos y tres. [[b]]El trabajo de verdad empieza ahora: describir en qué se diferencian. [[c]]Porque un grupo sin perfil no sirve para tomar ninguna decisión.",
    CL(), T("tres grupos… ¿y ahora qué?", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, GCOL[0], "grupo 1", "62 vinos", "a"), (960, GCOL[1], "grupo 2", "50 vinos", "a"), (1440, GCOL[2], "grupo 3", "65 vinos", "a")],
             y=420, w=400, h=240, tam_t=50, tam_d=42),
    P("hay que describir en qué se diferencian", 960, 680, "violeta", 46, en="b"), real(1050, en="a"), sec="1. Tres grupos sin nombre")

# ---- 2. Analogía
v.b("Una analogía. [[a]]En un matrimonio, las mesas están numeradas del uno al diez. [[b]]Nadie piensa que la mesa uno es mejor que la mesa siete: el número solo sirve para encontrarla. [[c]]Para saber quiénes se sientan en cada mesa, miras la lista de invitados. En clustering, esa lista es el perfil del grupo.",
    CL(), T("las mesas de un matrimonio", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "ambar", "mesa 1", "¿mejor que la 7? no", "b"), (1360, "ambar", "mesa 7", "solo un número", "b")], y=420, w=560, h=240, tam_t=50, tam_d=40),
    P("el perfil = la lista de quién está en cada grupo", 960, 700, "violeta", 46, en="c"), analogia(1050), sec="2. Analogía: las mesas")

# ---- 3. R y Python
v.b("La diapositiva veintisiete lo muestra con un caso concreto. [[a]]Con {k|, ka,} igual a tres, el script de R y el de Python encuentran exactamente la misma partición. [[b]]Pero la numeran distinto: el grupo uno de R, de sesenta y cinco vinos, es el grupo tres de Python. El dos coincide. Y el tres de R, de sesenta y dos vinos, es el uno de Python. [[c]]Mismos vinos, mismo WSS de 1267.37, otros nombres.",
    CL(), T("los números de grupo son etiquetas · diapositiva 27", 960, 140, tam=48, c="tinta", ent="subir"),
    fila(270, [("grupo R", 420), ("grupo Python", 760), ("tamaño", 1060), ("rasgo descriptivo", 1440)], tams=[34] * 4, colores=["gris"] * 4, en="a"),
    fila(370, [("1", 420), ("3", 760), ("65", 1060), ("menor alcohol y color", 1440)], tams=[42, 42, 42, 36], colores=["tinta", GCOL[2], "tinta", "tinta"],
         en="b", fondo="tealclaro"),
    fila(470, [("2", 420), ("2", 760), ("50", 1060), ("mayor color y menor matiz", 1440)], tams=[42, 42, 42, 36], colores=["tinta", GCOL[1], "tinta", "tinta"],
         en="b", t=1.5, fondo="ambarclaro"),
    fila(570, [("3", 420), ("1", 760), ("62", 1060), ("más flavonoides y prolina", 1440)], tams=[42, 42, 42, 36], colores=["tinta", GCOL[0], "tinta", "tinta"],
         en="b", t=3.0, fondo="azulclaro"),
    P("WSS = 1267.37 · BSS = 1020.63 · TSS = 2288", 960, 730, "violeta", 44, en="c"), diapo(27, en="a"), sec="3. Mismos grupos, otros números")
v.b("Desde aquí usamos siempre la numeración de Python, la del script. [[a]]Y recuerda la advertencia de la diapositiva: con otros valores de k, no solo cambian los números; también pueden cambiar las particiones.",
    T("desde aquí: numeración de Python", 960, 870, tam=42, c="tinta", w="b", en=None),
    T("con otro k pueden cambiar también las particiones", 960, 950, tam=36, c="gris", w="sb", en="a"))

# ---- 4. Perfiles en la escala original
v.b("Primero, los perfiles en la escala original, de la diapositiva veintiocho. [[a]]El grupo uno, de sesenta y dos vinos: alcohol 13.68, flavonoides 3.00, color 5.45. [[b]]El grupo dos, de cincuenta: alcohol 13.14, flavonoides 0.82, color 7.18. [[c]]El grupo tres, de sesenta y cinco: alcohol 12.25, flavonoides 2.05, color 2.97. Como referencia, las medias generales son 13.00, 2.04 y 5.03.",
    CL(), T("perfiles en la escala original · diapositiva 28", 960, 140, tam=48, c="tinta", ent="subir"),
    fila(270, [("grupo", 400), ("n", 640), ("alcohol", 900), ("flavonoides", 1220), ("color", 1520)], tams=[34] * 5, colores=["gris"] * 5),
    [fila(370 + 100 * g, [(str(g + 1), 400), (str(NV[g]), 640), (f"{PF[str(g + 1)]['alcohol']:.2f}", 900),
                           (f"{PF[str(g + 1)]['flavonoides']:.2f}", 1220), (f"{PF[str(g + 1)]['color']:.2f}", 1520)],
          tams=[44] * 5, colores=[GCOL[g]] + ["tinta"] * 4, en="abc"[g], fondo=["azulclaro", "ambarclaro", "tealclaro"][g]) for g in range(3)],
    fila(680, [("media general", 520), (f"{ME['alcohol']:.2f}", 900), (f"{ME['flavonoides']:.2f}", 1220), (f"{ME['color']:.2f}", 1520)],
         tams=[36, 40, 40, 40], colores=["gris"] * 4, en="c", t=3.5), diapo(28, en="a"), sec="4. Medias en unidades originales")
v.b("Con eso ya se puede describir: [[a]]el grupo uno tiene más alcohol y más flavonoides. El dos, muy pocos flavonoides y mucho color. El tres, menos alcohol y poco color. [[b]]Pero cuidado: son promedios del grupo. No describen exactamente a cada vino. [[c]]Conviene mirarlos junto con las distribuciones y los tamaños, y no sacar conclusiones que los datos no permiten: estas mediciones no dicen nada sobre precio, calidad ni preferencia.",
    T("1: más alcohol y flavonoides · 2: pocos flavonoides, mucho color · 3: menos alcohol, poco color", 960, 790, tam=32, c="tinta", w="sb", en="a"),
    P("son promedios: no describen a cada vino", 960, 880, "ambar", 40, en="b"),
    T("ni precio, ni calidad, ni preferencia", 960, 960, tam=34, c="rojo", w="sb", en="c"))

# ---- 5. Perfiles estandarizados
X0, DX, Y0, DY = 470, 108, 360, 130
celdas = []
for j, var in enumerate(VARS):
    celdas.append(T(CORTO[var], X0 + DX * j, Y0 - 90, tam=22, c="gris", w="sb"))
for g in range(3):
    k = "abc"[g]
    celdas.append(T(f"grupo {g + 1} ({NV[g]})", X0 - 70, Y0 + DY * g, tam=30, c=GCOL[g], w="b", anc="rm", en=k))
    for j, var in enumerate(VARS):
        zv = CE[str(g + 1)][var]
        celdas += [Bc(X0 + DX * j, Y0 + DY * g, DX - 8, DY - 10, tono(zv), None, 0, 10, en=k, t=0.04 * j),
                   T(f"{zv:.1f}".replace("-", "−"), X0 + DX * j, Y0 + DY * g, tam=28, c="tinta", w="b", en=k, t=0.04 * j + 0.05)]
v.b("Ahora, la vista de las trece variables a la vez, en la escala estandarizada: los centros de K-means. [[a]]El grupo uno está sobre la media en casi todo: prolina 1.1 desviaciones, flavonoides 1.0, alcohol 0.8. [[b]]El grupo dos está bajo la media en flavonoides y matiz, alrededor de menos 1.2, y sobre la media en color y ácido málico. [[c]]El grupo tres está bajo la media en alcohol, color y prolina. Un valor positivo indica una media de grupo superior a la media general.",
    CL(), T("perfiles estandarizados · diapositiva 29", 960, 130, tam=48, c="tinta", ent="subir"), celdas,
    T("azul: sobre la media general · rojo: bajo la media (en desviaciones estándar)", 960, 800, tam=32, c="gris", w="sb", en="c", t=2.0),
    diapo(29, en="a"), sec="5. Perfiles estandarizados")

# ---- 6. Alerta: numeración y unidades
v.b("Aquí hay dos precisiones sobre la grabación, y las dos importan. [[a]]Primera: se dijo que el grupo uno tiene menor alcohol y un color específico, y que el grupo tres tiene más flavonoides y prolina. Esa es la numeración de R. [[b]]Un minuto después, que el grupo uno tiene 13.68 de alcohol: esa es la numeración de Python. [[c]]El mismo nombre, grupo uno, se usó para dos grupos distintos. Justamente lo que advierte la diapositiva veintisiete.",
    CL(), alerta(160),
    cita("el grupo 1 muestra que tiene un menor alcohol y un color específico", y=300, en="a", tam=38),
    T("→ numeración de R (= grupo 3 de Python)", 960, 380, tam=36, c="ambar", w="sb", en="a", t=3.0),
    cita("el grupo 1 tiene 13.68 de alcohol, el grupo 2 13.14, el grupo 3 12.25", y=520, en="b", tam=38),
    T("→ numeración de Python", 960, 600, tam=36, c="azul", w="sb", en="b", t=3.0),
    P("«grupo 1» se usó para dos grupos distintos", 960, 740, "rojo", 44, en="c"),
    grabacion("1:26:25 y 1:27:02", en="a"), sec="6. Alerta: numeración y unidades")
v.b("Segunda precisión: [[a]]se dijo que el grupo uno tiene cero coma ocho más de alcohol, y el grupo tres, menos cero coma ochenta y nueve. [[b]]Esos números son desviaciones estándar, no grados de alcohol. El grupo uno está 0.83 desviaciones sobre la media; en unidades, son 13.68 menos 13.00: 0.68 de alcohol. [[c]]El grupo tres está a menos 0.92 desviaciones, no menos 0.89; en unidades, 12.25 menos 13.00: menos 0.75.",
    CL(), alerta(160),
    cita("el grupo 1 tiene 0.8 más de alcohol. El grupo 2, 0.2; en cuanto al grupo 3, tiene menos 0.89", y=300, en="a", tam=38),
    fila(450, [("", 360), ("en desviaciones (z)", 860), ("en unidades de alcohol", 1400)], tams=[34] * 3, colores=["gris"] * 3, en="b"),
    fila(550, [("grupo 1", 360), (f"{CE['1']['alcohol']:.2f}", 860), ("13.68 − 13.00 = 0.68", 1400)], tams=[40] * 3,
         colores=[GCOL[0], "tinta", "tinta"], en="b", t=1.0, fondo="azulclaro"),
    fila(660, [("grupo 3", 360), (f"{CE['3']['alcohol']:.2f}".replace("-", "−"), 860), ("12.25 − 13.00 = −0.75", 1400)], tams=[40] * 3,
         colores=[GCOL[2], "tinta", "tinta"], en="c", fondo="tealclaro"),
    grabacion("1:27:59", en="a"), etiqueta("Diapositiva 29: −0.9 para el grupo 3 (Python)", 1010, en="c"))

# ---- 7. Error típico
v.b("El error típico al leer perfiles: [[a]]comparar diferencias en unidades originales entre variables distintas. Una diferencia de cien en prolina parece enorme, y una de 0.5 en matiz, mínima. [[b]]Pero cien de prolina es apenas un tercio de desviación, y 0.5 de matiz son más de dos desviaciones. [[c]]Para comparar variables entre sí, se usan los perfiles estandarizados. Para describir un grupo con palabras, se vuelve a las unidades originales.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("comparar variables en unidades distintas", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    T(f"prolina: 100 / {SD['prolina']:.0f} = 0.32 desviaciones   ·   matiz: 0.5 / {SD['matiz']:.3f} = 2.20 desviaciones", 960, 450, tam=36, c="tinta", w="sb", en="b"),
    K(330, 620, "check", 64, en="c"), T("comparar con z; describir con unidades originales", 410, 620, tam=44, c="verde", anc="lm", en="c"),
    adicional(1050, en="b", texto="Cálculo adicional con las desviaciones reales"), sec="7. El error típico")

# ---- 8. Piensa
v.b("Piensa un segundo. [[a]]El grupo dos tiene color medio 7.18. La media general del color es 5.03, y su desviación, 2.30. [[b]]¿Cuánto vale ese color en desviaciones estándar?",
    CL(), T("color del grupo 2 = 7.18 · media = 5.03 · desviación = 2.30", 960, 320, tam=48, c="tinta", w="b", en="a"),
    P("z = ?", 960, 520, "violeta", 60, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Siete coma dieciocho menos cinco coma cero tres es 2.15. [[a]]Dividido por 2.30, da [[b]]0.935. Casi una desviación sobre la media: en la diapositiva veintinueve aparece como 0.9.",
    CL(), T("z = (7.18 − 5.03) / 2.30 = 2.15 / 2.30 =", 1120, 360, tam=48, c="tinta", w="sb", anc="rm", en="a"),
    N(0.935, 1150, 360, "{:.3f}", 72, "azul", anc="lm", en="b"), P("en la diapositiva 29: 0.9", 960, 560, "teal", 46, en="b", t=1.0),
    ilustrativo(1050, texto="Ejemplo ilustrativo con valores reales"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]el número de un grupo es solo una etiqueta; R y Python numeran distinto la misma partición. [[b]]El significado sale del perfil: medias originales para describir, perfiles estandarizados para comparar variables. [[c]]Y son promedios: no describen a cada vino, ni permiten hablar de calidad o precio.",
    CL(), P("el número es una etiqueta (R ≠ Python)", 960, 300, "ambar", 48, en="a"),
    P("describir con unidades · comparar con z", 960, 460, "teal", 48, en="b"),
    P("son promedios: ni calidad ni precio", 960, 620, "rojo", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]dibujar los grupos con una proyección, compararlos con las variedades reales, y ver los límites de K-means.",
    CL(), siguiente(9, "Proyección, clases reales y límites de K-means"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
