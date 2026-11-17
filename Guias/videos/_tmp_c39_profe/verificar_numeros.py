"""Comprueba los numeros de la serie C39.
(1) Cada cifra real dicha o mostrada se compara con datos39.json (recalculado ejecutando el script de la clase y el codigo de
    las diapositivas: datos.py) y con las diapositivas (slides.txt, extraido con pdftotext -enc UTF-8).
(2) Cada ejemplo resuelto se recalcula aqui a mano, con los mismos redondeos intermedios que se ven en pantalla.
(3) Cada literal se busca en el guion del video donde se dice (narracion + texto en pantalla).
(4) Las alertas del material se verifican contra la grabacion (transcripcion), las diapositivas y el script del profesor.
Ejecutar con cualquier Python 3 (solo usa json/math/re)."""
import json
import math
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
RAIZ = AQUI.parents[2]
D = json.loads((AQUI / "datos39.json").read_text(encoding="utf-8"))
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c39_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)
    return re.sub(r"\[\[\w+\]\]", "", txt).replace(" ", " ")


S = {n: say(n) for n in range(1, 13)}
SL = re.sub(r"\s+", " ", (AQUI / "slides.txt").read_text(encoding="utf-8") + " " + (AQUI / "slides_pag.txt").read_text(encoding="utf-8"))


def en(n, *lits):
    global ok
    for l in lits:
        if l in S[n]:
            ok += 1
        else:
            mal.append(f"video {n}: no aparece «{l}»")


def diap(*lits):
    global ok
    for l in lits:
        if l in SL:
            ok += 1
        else:
            mal.append(f"diapositivas: no aparece «{l}»")


r1 = lambda x: round(x, 1)  # noqa: E731
r2 = lambda x: round(x, 2)  # noqa: E731
r3 = lambda x: round(x, 3)  # noqa: E731
DT, KM, CO, SI, GP, PC = D["datos"], D["km"], D["codo"], D["sil"], D["gap"], D["pca"]
KO, KP, IT, IN = D["kmodes"], D["kproto"], D["iter2d"], D["init"]

# ================================================================ video 1 (diapositivas 2-5)
chk("v1 blobs n", len(D["sim_blobs"]["x"]), 90, 0)
diap("Logística, SVM, Naive Bayes.", "K-means Numéricas Media por variable", "K-modes Categóricas Moda por variable", "K-prototypes Mixtas Medias y modas",
     "El grupo 1 no significa “positivo”, “mejor” ni “mayor riesgo”.")
en(1, "Noventa observaciones", "K-prototypes")
# ================================================================ video 2 (diapositivas 6-9)
chk("v2 n", DT["n"], 177, 0); chk("v2 p", DT["p"], 13, 0)
chk("v2 media ceniza", r2(DT["media"]["ceniza"]), 2.37); chk("v2 media prolina", round(DT["media"]["prolina"]), 748, 0)
chk("v2 ej 14", (14 - 13) / 0.8, 1.25)
chk("v2 vino1 alcohol", DT["primer"]["alcohol"], 14.23); chk("v2 media alcohol", r2(DT["media"]["alcohol"]), 13.00)
chk("v2 sd alcohol", r3(DT["sd"]["alcohol"]), 0.814); chk("v2 z a mano", r3(round(14.23 - 13.00, 2) / 0.814), 1.511)
chk("v2 z real", r3(DT["primer"]["z_alcohol"]), 1.509)
chk("v2 var prolina", round(DT["var"]["prolina"]), 99292, 0); chk("v2 var magnesio", round(DT["var"]["magnesio"]), 204, 0)
chk("v2 var alcalinidad", r1(DT["var"]["alcalinidad"]), 11.2); chk("v2 var color", r2(DT["var"]["color"]), 5.27)
chk("v2 cuota prolina", r1(100 * DT["cuota_var"]["prolina"]), 99.8)
chk("v2 raw coinc", DT["raw_coinc"], 125, 0); chk("v2 z coinc", PC["coincidencias"], 171, 0)
chk("v2 ddof0", r2(DT["ddof0_wss"]), 1274.57); chk("v2 ddof0 a mano", r2(1267.373961 * 177 / 176), 1274.57); chk("v2 ddof0 ari", DT["ddof0_ari"], 1.0)
chk("v2 piensa", r3(round(7.6 - 5.03, 2) / 2.30), 1.117); chk("v2 media color", r2(DT["media"]["color"]), 5.03); chk("v2 sd color", r2(DT["sd"]["color"]), 2.30)
diap("Datos: 177 vinos y 13 mediciones", "con 178 filas en la fuente original. En el archivo recibido hay 177", "z = 14 − 13 = 1.25", "0.8",
     "Estandarizar iguala las escalas. No elimina atípicos ni correlaciones.", "z = (x - x.mean()) / x.std(ddof=1)")
en(2, "2.37", "748", "177", "1.25", "14.23", "13.00", "0.814", "1.511", "1.509", "1.51", "99.8", "1274.57", "1267.37", "125", "171", "1.117", "5.03", "2.30",
   "204", "11.2", "5.27")
# ================================================================ video 3 (diapositivas 10, 11, 16, 17)
chk("v3 dist2", (1 - 3) ** 2 + (2 - 1) ** 2, 5, 0); chk("v3 dist", r3(math.sqrt(5)), 2.236)
chk("v3 media 1,2,6", (1 + 2 + 6) / 3, 3); chk("v3 ss media", 4 + 1 + 9, 14, 0); chk("v3 ss 2", 1 + 0 + 16, 17, 0); chk("v3 ss 4", 9 + 4 + 4, 17, 0)
chk("v3 wss toy", 1 + 0 + 1 + 1 + 0 + 1, 4, 0); chk("v3 bss parte", 3 * (2 - 5.5) ** 2, 36.75); chk("v3 bss", 2 * 36.75, 73.5)
chk("v3 tss", sum((x - 5.5) ** 2 for x in (1, 2, 3, 8, 9, 10)), 77.5); chk("v3 cuadra", 4 + 73.5, 77.5)
chk("v3 wss vinos", r2(KM["wss_k3"]), 1267.37); chk("v3 bss vinos", r2(KM["bss"]), 1020.63); chk("v3 tss vinos", KM["tss"], 2288, 1e-6)
chk("v3 tss 13x176", 13 * 176, 2288, 0)
chk("v3 piensa wss", 4 + 0 + 4 + 0, 8, 0); chk("v3 piensa bss", 3 * (2 - 4) ** 2 + 1 * (10 - 4) ** 2, 48, 0)
chk("v3 piensa tss", sum((x - 4) ** 2 for x in (0, 2, 4, 10)), 56, 0); chk("v3 piensa media", (0 + 2 + 4 + 10) / 4, 4)
diap("∥x − c∥2 = (1 − 3)2 + (2 − 1)2 = 4 + 1 = 5", "5 ≈ 2.236", "WSS = 1 + 0 + 1 + 1 + 0 + 1 = 4", "36.75 + 36.75 = 73.5",
     "TSS = 20.25 + 12.25 + 6.25 + 6.25 + 12.25 + 20.25 = 77.5", "4 + 73.5 = 77.5", "WSS = 1267.37, BSS = 1020.63, TSS = 2288")
en(3, "2.236", "14", "17", "36.75", "73.5", "77.5", "20.25", "1267.37", "1020.63", "2288", "1274.57", "48", "56")
# ================================================================ video 4 (diapositivas 12-15)
chk("v4 toy centros", ((1 + 2 + 3) / 3, (8 + 9 + 10) / 3) == (2, 9), 1, 0)
chk("v4 toy dist", [(x - 1) ** 2 for x in (1, 2, 3, 8, 9, 10)] == [0, 1, 4, 49, 64, 81], 1, 0)
chk("v4 toy dist2", [(x - 8) ** 2 for x in (1, 2, 3, 8, 9, 10)] == [49, 36, 25, 0, 1, 4], 1, 0)
H = IT["hist"]
for i, val in enumerate([165.83, 105.50, 103.24, 102.90, 102.78, 102.58]):
    chk(f"v4 iter {i + 1}", r2(H[i]["wss_asig"]), val)
chk("v4 n iter 2d", len(H), 7, 0); chk("v4 wss final", r2(H[-1]["wss_asig"]), 102.54); chk("v4 tamaños", H[-1]["tamanos"] == [42, 74, 61], 1, 0)
chk("v4 n_iter k3", KM["n_iter"], 6, 0)
chk("v4 rect bueno", 4 * 0.5 ** 2, 1); chk("v4 rect malo", 4 * 2 ** 2, 16)
chk("v4 rect estable", (0 - 2) ** 2 + 0 < (0 - 2) ** 2 + 1, 1, 0)
chk("v4 init min", r2(IN["min"]), 1267.37); chk("v4 init max", r2(IN["max"]), 1604.15); chk("v4 27 %", round(100 * (IN["max"] / IN["min"] - 1)), 27, 0)
chk("v4 n min", IN["n_min"], 18, 0); chk("v4 init 50", r2(IN["mejor_n_init50"]), 1267.37)
chk("v4 piensa", (0.5 ** 2) * 4, 1)
diap("Nuevos centros: µ1 = (1 + 2 + 3)/3 = 2 y µ2 = (8 + 9 + 10)/3 = 9", "Conservamos el ajuste de menor WSS entre varias inicializaciones.",
     "nstart=50 en R y n_init=50 en Python")
en(4, "165.83", "105.50", "103.24", "102.90", "102.78", "102.58", "102.54", "cuarenta y dos, setenta y cuatro y sesenta y uno", "1267.37", "1604.15",
   "veintisiete por ciento", "dieciocho veces", "convergió en seis", "0.5 y 5.5")
# ================================================================ video 5 (diapositivas 18, 19, 26)
for i, val in enumerate([2288, 1643.90, 1267.37, 1164.91, 1092.18, 1033.37, 970.43, 931.19]):
    chk(f"v5 wss k{i + 1}", r2(KM["wss"][i]), val)
for i, val in enumerate([644.1, 376.5, 102.5, 72.7, 58.8, 62.9, 39.2]):
    chk(f"v5 baja {i + 1}", r1(CO["red"][i]), val)
chk("v5 piensa", [100 - 60, 60 - 35, 35 - 31, 31 - 28] == [40, 25, 4, 3], 1, 0)
diap("k=3: 1267,37", "k = 3 es un candidato interpretable", "inertia_ es WSS. labels_ contiene los grupos, numerados desde cero.")
en(5, "2288", "1643.90", "1267.37", "1164.91", "1092.18", "931.19", "644.1", "376.5", "102.5", "72.7", "58.8", "62.9", "39.2")
# ================================================================ video 6 (diapositivas 20-21)
chk("v6 ej diap", r3((6 - 2) / 6), 0.667)
chk("v6 a3", (2 + 1) / 2, 1.5); chk("v6 b3", (5 + 6 + 7) / 3, 6); chk("v6 s3", (6 - 1.5) / 6, 0.75)
chk("v6 b1", (7 + 8 + 9) / 3, 8); chk("v6 s1 (0.8125, mitad hacia arriba)", float(__import__("decimal").Decimal("6.5") / 8), 0.8125, 0); chk("v6 s1 redondeo", 0.813, 0.8125, 0.0005)
for i, val in enumerate([0.258, 0.284, 0.258, 0.235, 0.192, 0.199, 0.136]):
    chk(f"v6 sil k{i + 2}", r3(KM["silueta"][i + 1]), val)
chk("v6 sil grupos", (r2(SI["por_grupo"][0]), r2(SI["por_grupo"][1]), r3(SI["por_grupo"][2])) == (0.34, 0.35, 0.177), 1, 0)
chk("v6 negativas", SI["negativas"] == [0, 0, 7], 1, 0); chk("v6 piensa", r3((2 - 3) / 3), -0.333)
diap("s(i) = (6 − 2)/6 = 0.667", "k=3: 0,284", "Por convención, una observación sola en su grupo tiene silueta cero")
en(6, "0.667", "1.5", "0.75", "0.813", "0.258", "0.284", "0.235", "0.192", "0.199", "0.136", "0.177", "siete vinos", "0.333")
# ================================================================ video 7 (diapositivas 22-24, 56-57)
chk("v7 hip k1", r2(0.62 - 0.05), 0.57); chk("v7 hip k2", r2(0.64 - 0.04), 0.60); chk("v7 hip elige", 0.62 >= 0.60 and not 0.40 >= 0.57, 1, 0)
for i, val in enumerate([0.800, 1.015, 1.199, 1.227, 1.245]):
    chk(f"v7 gap k{i + 1}", r3(GP["gap"][i]), val)
chk("v7 err k4", r3(GP["err"][3]), 0.022); chk("v7 err k5", r3(GP["err"][4]), 0.021)
chk("v7 umbral k3", r3(1.227 - 0.022), 1.205); chk("v7 umbral k4", r3(1.245 - 0.021), 1.224)
chk("v7 umbral k3 real", r3(GP["umbral"][2]), 1.205); chk("v7 umbral k4 real", r3(GP["umbral"][3]), 1.224)
chk("v7 k_gap", GP["k_gap"], 4, 0); chk("v7 cumple", GP["cumple"][:4] == [False, False, False, True], 1, 0); chk("v7 max", GP["k_max_gap"], 8, 0)
chk("v7 piensa k1", r2(0.70 - 0.04), 0.66); chk("v7 piensa k2", r2(0.75 - 0.06), 0.69); chk("v7 piensa elige", 0.70 >= 0.69 and not 0.50 >= 0.66, 1, 0)
diap("Para k = 1: 0.40 < 0.62 − 0.05 = 0.57. No cumple.", "Para k = 2: 0.62 ≥ 0.64 − 0.04 = 0.60. Elegimos 2.", "Regla de un error estandar: k=4",
     "Silueta: k = 3. Gap con esta referencia y 30 simulaciones: k = 4 en R y Python.", "error_gap = log_wss_ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)")
en(7, "0.57", "0.60", "0.800", "1.015", "1.199", "1.227", "1.205", "1.224", "1.245", "0.022", "0.021", "0.66", "0.69")
# ================================================================ video 8 (diapositivas 27-29)
chk("v8 tamaños", KM["tamanos"] == [62, 50, 65], 1, 0)
PF, CE, ME = KM["perfiles"], KM["centros"], DT["media"]
for g, (a, f, c) in {"1": (13.68, 3.00, 5.45), "2": (13.14, 0.82, 7.18), "3": (12.25, 2.05, 2.97)}.items():
    chk(f"v8 alcohol {g}", r2(PF[g]["alcohol"]), a); chk(f"v8 flav {g}", r2(PF[g]["flavonoides"]), f, 0.0051); chk(f"v8 color {g}", r2(PF[g]["color"]), c)
chk("v8 medias", (r2(ME["alcohol"]), r2(ME["flavonoides"]), r2(ME["color"])) == (13.00, 2.04, 5.03), 1, 0)
chk("v8 z g1 alcohol", r2(CE["1"]["alcohol"]), 0.83); chk("v8 z g3 alcohol", r2(CE["3"]["alcohol"]), -0.92)
chk("v8 unid g1", r2(13.68 - 13.00), 0.68); chk("v8 unid g3", r2(12.25 - 13.00), -0.75)
chk("v8 z g1 prolina", r1(CE["1"]["prolina"]), 1.1); chk("v8 z g1 flav", r1(CE["1"]["flavonoides"]), 1.0)
chk("v8 z g2 flav/matiz", (r1(CE["2"]["flavonoides"]), r1(CE["2"]["matiz"])) == (-1.2, -1.2), 1, 0)
chk("v8 prolina 100", r2(100 / round(DT["sd"]["prolina"])), 0.32); chk("v8 matiz 0.5", r2(0.5 / r3(DT["sd"]["matiz"])), 2.20)
chk("v8 piensa", r3(round(7.18 - 5.03, 2) / 2.30), 0.935)
diap("1 3 65 Menor alcohol y color", "2 2 50 Mayor color y menor matiz", "3 1 62 Más flavonoides y prolina", "1 62 13.68 3.00 5.45", "2 50 13.14 0.82 7.18",
     "3 65 12.25 2.05 2.97")
en(8, "13.68", "3.00", "5.45", "13.14", "0.82", "7.18", "12.25", "2.05", "2.97", "13.00", "2.04", "5.03", "0.83", "0.68", "0.92", "0.75", "0.935",
   "1267.37", "1020.63", "2288", "0.32", "2.20")
# ================================================================ video 9 (diapositiva 30-31, script seccion 7)
chk("v9 pc1", r1(PC["var"][0]), 36.0); chk("v9 pc2", r1(PC["var"][1]), 19.4); chk("v9 pc", r1(PC["var2"]), 55.3)
chk("v9 tabla", PC["tabla"] == [[59, 0, 0], [3, 3, 65], [0, 47, 0]], 1, 0)
chk("v9 coinc", PC["coincidencias"], 171, 0); chk("v9 pureza", r1(100 * PC["pureza"]), 96.6); chk("v9 disc", PC["discrepa"], 6, 0)
chk("v9 disc clase 2", set(PC["disc_clase"]) == {2}, 1, 0); chk("v9 a mano", r1(100 * 171 / 177), 96.6)
chk("v9 lunas", round(100 * D["sim_lunas"]["acuerdo"]), 75, 0); chk("v9 atipico", D["sim_atipico"]["tamanos"] == [1, 80], 1, 0)
chk("v9 piensa", (40 + 45) / 100 * 100, 85)
diap("PCA permite dibujar una aproximación en dos dimensiones.", "El agrupamiento se ajustó con las 13 variables, no con estas dos coordenadas.")
en(9, "36.0", "19.4", "55.3", "96.6", "171", "seis discrepancias", "setenta y cinco por ciento", "ochenta puntos", "85")
# ================================================================ video 10 (diapositivas 32-41)
chk("v10 A", 0 + 0 + 1, 1, 0); chk("v10 B", 1 + 1 + 0, 2, 0); chk("v10 moda costo", 0 + 1 + 2, 3, 0)
chk("v10 cats", (KO["cats"]["alcohol"]["alto"], KO["cats"]["alcohol"]["bajo"], KO["cats"]["color"]["bajo"], KO["cats"]["color"]["medio"],
                 KO["cats"]["color"]["alto"]) == (92, 85, 70, 67, 40), 1, 0)
chk("v10 empates", (KO["empates_flav_15"], KO["empates_color_4"]) == (1, 1), 1, 0)
for i, val in enumerate([283, 148, 102, 75, 58, 48, 38, 26]):
    chk(f"v10 costo k{i + 1}", KO["costo"][i], val, 0)
G2 = KO["grupos"]["2"]
chk("v10 g2 n", G2["n"], 48, 0); chk("v10 g2 centro", G2["centro"] == ["alto", "bajo", "alto"], 1, 0)
chk("v10 g2 color", (G2["color"]["alto"], G2["color"]["medio"], G2["color"]["bajo"]) == (40, 7, 1), 1, 0)
chk("v10 83.3", r1(100 * 40 / 48), 83.3); chk("v10 14.6", r1(100 * 7 / 48), 14.6); chk("v10 2.1", r1(100 * 1 / 48), 2.1)
chk("v10 18 perfiles", 2 * 3 * 3, 18, 0); chk("v10 piensa A", 1 + 1 + 0, 2, 0); chk("v10 piensa B", 0 + 0 + 1, 1, 0)
diap("0+0+1=1", "1+1+0=2", "Costo del grupo = 0 + 1 + 2 = 3 desacuerdos.", "Alto 40 40/48 = 83.3 %", "1/48 = 2.1 %", "7/48 = 14.6 %",
     "Su centro modal es (alcohol alto, flavonoides bajos, color alto).", "Usaremos k = 4 para ilustrar perfiles", "un valor de flavonoides igual a 1.5 pertenece a “medio”")
en(10, "283", "148", "102", "75", "58", "48", "38", "26", "83.3", "14.6", "2.1", "noventa y dos", "ochenta y cinco", "dieciocho perfiles")
# ================================================================ video 11 (diapositivas 42-50)
chk("v11 A num", (1 - 0) ** 2 + (2 - 2) ** 2, 1, 0); chk("v11 B num", (1 - 2) ** 2 + (2 - 3) ** 2, 2, 0)
chk("v11 g05", 1 + 0.5, 1.5); chk("v11 g3", 1 + 3, 4); chk("v11 empate", 1 + 1, 2)
for i, val in enumerate([635, 439.61, 326.64, 285.73]):
    chk(f"v11 costo k{i + 1}", r2(KP["costo"][i]), val)
chk("v11 635 = 352 + 283", 2 * 176 + KO["costo"][0], KP["costo"][0], 1e-6)
GPk = KP["grupos"]
for g, (n_, c_, m_, a_, f_, co_) in {"1": (69, 0.625, 0.506, "alto", "medio", "medio"), "2": (47, 0.225, -1.258, "alto", "bajo", "alto"),
                                     "3": (61, -0.881, 0.396, "bajo", "medio", "bajo")}.items():
    chk(f"v11 g{g} n", GPk[g]["n"], n_, 0); chk(f"v11 g{g} ceniza", r3(GPk[g]["ceniza"]), c_); chk(f"v11 g{g} matiz", r3(GPk[g]["matiz"]), m_)
    chk(f"v11 g{g} modas", (GPk[g]["alcohol"], GPk[g]["flavonoides"], GPk[g]["color"]) == (a_, f_, co_), 1, 0)
chk("v11 30/47", GPk["2"]["color_n"]["alto"], 30, 0); chk("v11 63.8", r1(100 * 30 / 47), 63.8)
chk("v11 sens", (KP["sens"]["0.25"]["tamanos"], KP["sens"]["1"]["tamanos"], KP["sens"]["4"]["tamanos"]) == ([60, 60, 57], [69, 47, 61], [55, 68, 54]), 1, 0)
chk("v11 piensa P", 1 + 1 + 0, 2, 0); chk("v11 piensa Q g1", 0 + 1 + 2 * 1, 3, 0); chk("v11 piensa umbral", (2 - 1) / 2, 0.5)
diap("γ = 0.5 : 1.5 < 2 ⇒ A; γ = 3 : 4 > 2 ⇒ B.", "El ajuste usa categorical=[2, 3, 4] porque Python cuenta desde cero.",
     "Aquí lambda es el peso de las categorías. No es la penalización de Ridge.", "solo 30/47 ≈ 63.8 % tiene esa categoría", "K-prototypes 3 326.64")
en(11, "1.5", "635", "352", "283", "439.61", "326.64", "285.73", "0.625", "0.506", "0.225", "1.258", "0.881", "0.396", "63.8", "sesenta, sesenta y cincuenta y siete",
   "sesenta y nueve, cuarenta y siete y sesenta y uno", "cincuenta y cinco, sesenta y ocho y cincuenta y cuatro", "0.5")
# ================================================================ video 12 (diapositivas 51-53)
diap("K-means 3 1267.37", "K-modes 4 75", "No elegimos K-modes porque 75 sea el menor número de esta tabla.", "No requiere automáticamente una división 70/30.",
     "No calculamos exactitud ni AUC: no hay una clase observada que predecir.")
en(12, "1267.37", "75", "326.64", "2288", "0.284", "171")
# ================================================================ (4) alertas: grabacion, diapositivas y script
tr = re.sub(r"\s+", " ", (RAIZ / "VideosClases" / "Clase 39 Clustering Profesor pedro luiz ramos.txt").read_text(encoding="utf-8"))
for frase in ("1 menos 2 al cuadrado, después sacamos la raíz, va a ser 1",                       # video 3
              "evaluamos cuáles fueron los clústeres que más aparecieron, que más ocurrieron",     # video 4
              "Conservamos el ajuste del menor WSS entre varias inicializaciones",                 # video 4: dicho bien despues
              "varios algoritmos van a correr ahí 50 mil veces, 60 mil veces",                     # video 4
              "un K-means, que va a variar, por ejemplo, de uno hasta nueve clusters",            # video 5
              "él va a mirar exactamente el próximo gap menos un s_k",                             # video 7
              "donde k es igual a 4",                                                              # video 7
              "no me acuerdo si es Merlot", "un vino blanco y otro tipo de vino, Cabernet",       # video 2
              "yo tengo ahí casi 56% de la variabilidad explicada por estos componentes",          # video 9
              "debe tener alguna cosa como 8, 9 puntos de 177 mal clasificados",                   # video 9
              "más de 90%, 92%", "1, 2, 3, 4, 5, 6",
              "el grupo 1 muestra que tiene un menor alcohol y un color específico",               # video 8
              "el grupo 1 tiene 13.68 de alcohol, el grupo 2 13.14, el grupo 3 12.25",
              "el grupo 1 tiene 0.8 más de alcohol. El grupo 2, 0.2; en cuanto al grupo 3, tiene menos 0.89",
              "me parece que el método debería haber elegido en torno de 4, pero sabemos que son 3",  # video 10
              "tuvimos un cambio en el costo, o en la varianza, básicamente",                      # video 12
              "no es una penalización como, por ejemplo, la de Ridge", "no elegimos K-modes porque 75 sea el menor número",
              "metí las 13 covariables", "el ejemplo sencillo acá es, voy a estandarizar la variable alcohol",
              "casi 0.28", "con dos grupos tenemos una silueta media de 0.26"):
    if frase in tr:
        ok += 1
    else:
        mal.append(f"grabacion: no aparece «{frase}»")
py = (next(p for p in RAIZ.glob("Clase 39*") if p.is_dir()) / "kmeans_numero_clusters.py").read_text(encoding="utf-8")
for frase in ("valores_k = range(1, 9)", "n_init=50, max_iter=100", "random_state=2026", "z = (x - x.mean()) / x.std(ddof=1)",
              "if gap[k - 1] >= gap[k] - error_gap[k]:", "error_gap = log_wss_ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)", "B = 30",
              "pureza = np.mean(~discrepa)"):
    if frase in py:
        ok += 1
    else:
        mal.append(f"script: no aparece «{frase}»")
chk("script range(1,9) = 1..8", list(range(1, 9)) == [1, 2, 3, 4, 5, 6, 7, 8], 1, 0)
chk("script k=3 iteraciones <= 100", KM["n_iter"] <= 100, 1, 0)

print(f"comprobaciones OK: {ok}   fallas: {len(mal)}")
for m in mal:
    print("  FALLA", m)
