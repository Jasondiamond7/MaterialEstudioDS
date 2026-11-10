"""Comprueba los numeros de la serie C38.
(1) Cada cifra real dicha o mostrada se compara con datos38.json (recalculado ejecutando el script de la clase: datos.py)
    y con las diapositivas (slides.txt, extraido con pdftotext -enc UTF-8).
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
D = json.loads((AQUI / "datos38.json").read_text(encoding="utf-8"))
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c38_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)
    return re.sub(r"\[\[\w+\]\]", "", txt).replace(" ", " ")


S = {n: say(n) for n in range(1, 13)}
SL = (AQUI / "slides.txt").read_text(encoding="utf-8")


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


HP, HF, HR = D["h_part"], D["h_final"], D["h_resid"]
RK = {r["modelo"]: r for r in D["h_ranking"]}
SR = {r["modelo"]: r for r in D["s_ranking"]}
SP, SF = D["s_part"], D["s_final"]
CT = {f"{c['t']:.2f}": c for c in D["s_costos"]}
r3 = lambda x: round(x, 3)  # noqa: E731

# ================================================================ particiones (diapositivas 5-7, 13, 33, 47)
for k, val in dict(filas=1030, grupos=428, filas_tr=806, grupos_tr=342, filas_te=224, grupos_te=86).items():
    chk(f"hormigon {k}", HP[k], val, 0)
for k, val in dict(n=4601, tr=3680, te=921, n0=2230, n1=1450).items():
    chk(f"spam {k}", SP[k], val, 0)
diap("806 mediciones / 342 composiciones", "224 mediciones / 86 composiciones", "428 composiciones en total", "1030 mediciones",
     "2230 (60.6%)", "1450 (39.4%)", "4601 correos; 57 entradas", "Reservar 921 correos para prueba; usar 3680")
# ================================================================ video 1
EJ, EJ4 = HR["ej"][0], HR["ej"][3]
chk("v1 real", EJ["real"], 61.9, 0.05); chk("v1 pred", EJ["pred"], 59.6, 0.05); chk("v1 media", EJ["base"], 35.5, 0.05)
chk("v1 err modelo", round(61.9 - 59.6, 1), 2.3); chk("v1 err media", round(61.9 - 35.5, 1), 26.4)
chk("v1 real4", EJ4["real"], 42.6, 0.05); chk("v1 pred4", EJ4["pred"], 54.0, 0.05); chk("v1 err4", round(42.6 - 54.0, 1), -11.4)
chk("v1 media tr", round(HP["media_tr"], 1), 35.5)
diap("y = 35 MPa", "p = 0.70")
en(1, "1030", "61.9", "59.6", "35.5", "2.3", "26.4", "42.6", "54.0", "11.4", "224", "0.70", "0.85")
# ================================================================ video 2 (diapositivas 8-12)
A, B = [0, 0, 0, 8], [3, 3, 3, 3]
mae = lambda e: sum(abs(x) for x in e) / len(e)  # noqa: E731
rmse = lambda e: math.sqrt(sum(x * x for x in e) / len(e))  # noqa: E731
chk("v2 MAE A", mae(A), 2); chk("v2 MAE B", mae(B), 3); chk("v2 RMSE A", rmse(A), 4); chk("v2 RMSE B", rmse(B), 3)
y, yh = [1, 2, 3], [11, 12, 13]
sse = sum((a - b) ** 2 for a, b in zip(y, yh)); sst = sum((a - 2) ** 2 for a in y)
chk("v2 sse", sse, 300); chk("v2 sst", sst, 2); chk("v2 R2", 1 - sse / sst, -149); chk("v2 rmse 10", rmse([a - b for a, b in zip(y, yh)]), 10)
chk("v2 R2 media OOF", round(RK["MEDIA"]["r2"], 3), -0.002); chk("v2 R2 media prueba", round(HF["ref_r2"], 3), -0.007)
chk("v2 rmse ganador", r3(RK["Q_A0.1"]["rmse"]), 8.229); chk("v2 mae ganador", r3(RK["Q_A0.1"]["mae"]), 6.234); chk("v2 r2 ganador", r3(RK["Q_A0.1"]["r2"]), 0.758)
E = [4, 0, -2, 2]
chk("v2 piensa MAE", mae(E), 2); chk("v2 piensa RMSE", r3(rmse(E)), 2.449)
diap("A tiene MAE = 2 y RMSE = 4; B tiene MAE = 3 y RMSE = 3", "R2 = 1 − 300", "−149", "RMSE = 10")
en(2, "8.229", "6.234", "0.758", "0.002", "0.007", "149", "2.449", "es un número que está entre 0 y 1")
# ================================================================ video 3 (diapositivas 7 y 13)
GE = D["h_grupo_ej"]
chk("v3 edades grupo", GE["n_edades"], 8, 0)
for e, r in ((1, 12.6), (28, 44.1), (270, 55.2)):
    chk(f"v3 grupo edad {e}", next(f["res"] for f in GE["filas"] if f["edad"] == e), r, 0.05)
chk("v3 grupos varias", GE["grupos_con_varias"], 182, 0); chk("v3 filas varias", GE["filas_en_grupos_varias"], 784, 0)
chk("v3 % grupos", round(100 * 86 / 428, 1), 20.1); chk("v3 % filas", round(100 * 224 / 1030, 1), 21.7)
chk("v3 pliegues grupos", sum(HP["fold_grupos"]), 342, 0); chk("v3 pliegues filas", sum(HP["fold_filas"]), 806, 0)
chk("v3 pliegue min/max", (min(HP["fold_grupos"]), max(HP["fold_grupos"]), min(HP["fold_filas"]), max(HP["fold_filas"])) == (68, 69, 161, 162), 1, 0)
chk("v3 342+86", 342 + 86, 428, 0); chk("v3 piensa", (100 * 10 / 50, 100 * 60 / 250) == (20, 24), 1, 0)
diap("Se reservó 20 % de los grupos")
en(3, "12.6", "44.1", "55.2", "182", "784", "20.1", "21.7", "428", "342", "86", "68", "69", "161", "162")
# ================================================================ video 4 (diapositivas 14-17, 23)
chk("v4 terminos", 8 + 8 + 8 * 7 // 2, 44, 0); chk("v4 ajustes", 7 * 5, 35, 0)
chk("v4 ridge a=0", 20 / 10, 2); chk("v4 ridge a=10", 20 / 20, 1); chk("v4 ridge a=1000", round(20 / 1010, 2), 0.02)
chk("v4 piensa", 4 + 4 + 4 * 3 // 2, 14, 0)
diap("Elegimos el modelo 5: 8.7", "8 lineales", "28 interacciones", "35 ajustes de CV", "= 0.1, 1, 10, 100, 1000")
en(4, "8.7", "35.5", "44", "28", "35", "0.02", "14")
# ================================================================ video 5 (diapositivas 18-19)
diap("RMSE de ajuste: 0.67", "RMSE en datos nuevos: 0.64", "RMSE de ajuste: 0.58", "RMSE en datos nuevos: 0.73")
chk("v5 descomp", (12 - 10) ** 2 + 1 + 0.25, 5.25)
chk("v5 lineal", r3(RK["LINEAL"]["rmse"]), 10.560); chk("v5 a1000", r3(RK["Q_A1000"]["rmse"]), 13.051)
chk("v5 ajuste a0.1", r3(RK["Q_A0.1"]["rmse_ajuste"]), 6.958)
chk("v5 piensa A", 1 + 3 + 0.5, 4.5); chk("v5 piensa B", 2.5 + 0.5 + 0.5, 3.5)
en(5, "0.67", "0.64", "0.58", "0.73", "5.25", "10.560", "13.051", "8.229", "6.958", "4.5", "3.5")
# ================================================================ video 6 (diapositivas 20-25)
chk("v6 prom RMSE", (math.sqrt(1) + math.sqrt(9)) / 2, 2); chk("v6 RMSE OOF", r3(math.sqrt((1 + 9) / 2)), 2.236)
FQ = [f["rmse"] for f in D["h_fold"]["Q_A0.1"]]
for a, b in zip(FQ, [7.110, 7.670, 8.806, 9.320, 8.058]):
    chk("v6 pliegue a0.1", r3(a), b)
chk("v6 prom pliegues", r3(sum(FQ) / 5), 8.193); chk("v6 prom datos", r3(D["h_prom_vs_oof"]["Q_A0.1"]["prom"]), 8.193)
chk("v6 media ok", (10 + 20 + 30) / 3, 20); chk("v6 media fuga", (10 + 20 + 30 + 60) / 4, 30)
chk("v6 piensa prom", r3((2 + 4 + 5) / 3), 3.667); chk("v6 piensa oof", r3(math.sqrt((4 + 16 + 25) / 3)), 3.873)
diap("1 + 3 = 2", "5 ≈ 2.236", "35 ajustes de CV")
en(6, "806", "2.236", "7.110", "7.670", "8.806", "9.320", "8.058", "8.193", "8.229", "3.667", "3.873")
# ================================================================ video 7 (diapositivas 26-29)
TAB = {"Q_A0.1": (8.229, 6.234), "Q_A1": (8.308, 6.260), "Q_A10": (8.489, 6.390), "Q_A100": (9.640, 7.638), "LINEAL": (10.560, 8.426),
       "Q_A1000": (13.051, 10.595), "MEDIA": (16.746, 13.631)}
for m, (r, a) in TAB.items():
    chk(f"v7 rmse {m}", r3(RK[m]["rmse"]), r); chk(f"v7 mae {m}", r3(RK[m]["mae"]), a)
diap("8.229 8.308 8.489 9.640 10.560 13.051 16.746", "6.234 6.260 6.390 7.638 8.426 10.595 13.631")
chk("v7 orden MAE = orden RMSE", [m for m, _ in sorted(TAB.items(), key=lambda kv: kv[1][1])] == list(TAB), 1, 0)
chk("v7 dif", round(RK["Q_A1"]["rmse"] - RK["Q_A0.1"]["rmse"], 3), 0.079); chk("v7 50.9", round(100 * (1 - 8.229 / 16.746), 1), 50.9)
for a, (aj, oo) in {"0.1": (6.958, 8.229), "10": (7.130, 8.489), "1000": (12.511, 13.051)}.items():
    chk(f"v7 curva {a} ajuste", r3(RK[f"Q_A{a}"]["rmse_ajuste"]), aj); chk(f"v7 curva {a} oof", r3(RK[f"Q_A{a}"]["rmse"]), oo)
diap("6.958 7.130 12.511", "8.229 8.489 13.051")
F1_ = [r3(f["rmse"]) for f in D["h_fold"]["Q_A1"]]
for a, b in zip(F1_, [7.099, 8.069, 8.836, 9.401, 7.957]):
    chk("v7 pliegue a1", a, b)
DIF = [r3(r3(a) - b) for a, b in zip(FQ, F1_)]
chk("v7 difs", DIF == [0.011, -0.399, -0.030, -0.081, 0.101], 1, 0)
chk("v7 gana 0.1 en 2,3,4", [k + 1 for k, d in enumerate(DIF) if d < 0] == [2, 3, 4], 1, 0)
chk("v7 gana 1 en 1,5", [k + 1 for k, d in enumerate(DIF) if d > 0] == [1, 5], 1, 0)
chk("v7 piensa", (sum(1 for d in (-0.2, 0.1, -0.3, 0.2, -0.1) if d < 0), round(sum((-0.2, 0.1, -0.3, 0.2, -0.1)) / 5, 2)) == (3, -0.06), 1, 0)
diap("7.110 7.099", "7.670 8.069", "8.806 8.836", "9.320 9.401", "8.058 7.957")
en(7, "8.229", "8.308", "0.079", "10.560", "13.051", "16.746", "50.9", "6.958", "7.130", "8.489", "12.511", "0.011", "0.399", "0.030",
   "0.081", "0.101", "7.099", "7.957", "8.836", "0.06")
# ================================================================ video 8 (diapositivas 30-32)
for k, val in dict(rmse=8.823, mae=6.494, r2=0.716, ref_rmse=16.605, ref_mae=12.826, ref_r2=-0.007).items():
    chk(f"v8 {k}", r3(HF[k]), val)
chk("v8 baja", round(100 * HF["baja"], 1), 46.9); chk("v8 baja a mano", round(100 * (1 - 8.823 / 16.605), 1), 46.9)
diap("8.823 16.605", "6.494 12.826", "0.716 -0.007", "El RMSE baja 46.9 %")
chk("v8 max", HR["max"], 24.39, 0.005); chk("v8 min", HR["min"], -32.81, 0.005); chk("v8 >15", HR["sobre15"], 25, 0); chk("v8 >20", HR["sobre20"], 9, 0)
for k, (n_, r) in {"1-7": (57, 10.247), "14-28": (104, 7.330), "56-365": (63, 9.649)}.items():
    chk(f"v8 edad {k} n", HR["rmse_edad"][k]["n"], n_, 0); chk(f"v8 edad {k} rmse", r3(HR["rmse_edad"][k]["rmse"]), r)
chk("v8 piensa", 100 * (1 - 6 / 15), 60)
en(8, "806", "224", "8.823", "6.494", "0.716", "16.605", "12.826", "0.007", "46.9", "8.229", "24.39", "32.81", "25", "10.247", "7.330", "9.649",
   "57", "104", "63", "60")
# ================================================================ video 9 (diapositivas 33-38)
chk("v9 % legit", round(100 * 2230 / 3680, 1), 60.6); chk("v9 % spam", round(100 * 1450 / 3680, 1), 39.4)
chk("v9 95 %", 95 / 100, 0.95); chk("v9 990", 1000 - 10, 990, 0)
chk("v9 acc A", (890 + 60) / 1000, 0.95); chk("v9 acc B", (860 + 90) / 1000, 0.95)
chk("v9 costo A", 5 * 10 + 40, 90, 0); chk("v9 costo B", 5 * 40 + 10, 210, 0)
chk("v9 F1 A", r3(120 / 170), 0.706); chk("v9 F1 B", r3(180 / 230), 0.783)
chk("v9 F1 A prec-rec", r3(2 * (60 / 70) * 0.6 / ((60 / 70) + 0.6)), 0.706)
chk("v9 piensa acc", (895 + 50) / 10, 94.5); chk("v9 piensa costo", 5 * 5 + 50, 75, 0)
diap("acierta 95 de 100 correos, pero detecta 0 de los 5 spam", "A: 5(10) + 40 = 90.", "B: 5(40) + 10 = 210.", "0.706 0.783")
en(9, "4601", "57", "921", "3680", "2230", "1450", "60.6", "39.4", "95", "990", "890", "860", "90", "210", "0.706", "0.783", "94.5", "75")
# ================================================================ video 10 (diapositivas 39-44)
PU = [0.1, 0.4, 0.6, 0.8, 0.9]
chk("v10 t=0.5", sum(p >= 0.5 for p in PU), 3, 0); chk("v10 t=0.8", sum(p >= 0.8 for p in PU), 2, 0)
diap("Con t = 0.5 se marcan 3 correos; con t = 0.8, se marcan 2")
PA, YA = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], [0, 0, 1, 0, 1, 1]
ll = lambda ps, ys: -sum(math.log(p if y else 1 - p) for p, y in zip(ps, ys)) / len(ps)  # noqa: E731
chk("v10 logloss A", r3(ll(PA, YA)), 0.415)
PB = [1 / (1 + math.exp(-4 * math.log(p / (1 - p)))) for p in PA]
chk("v10 logloss B (reconstruida)", r3(ll(PB, YA)), 0.602); chk("v10 B correo 4", r3(PB[3]), 0.835)
terms = [r3(-math.log(p if y else 1 - p)) for p, y in zip(PA, YA)]
chk("v10 terminos", terms == [0.105, 0.223, 0.916, 0.916, 0.223, 0.105], 1, 0)
chk("v10 suma", r3(sum(terms)), 2.488); chk("v10 suma/6", r3(2.488 / 6), 0.415)
chk("v10 -ln0.9", r3(-math.log(0.9)), 0.105); chk("v10 -ln0.1", r3(-math.log(0.1)), 2.303)
for p, b in ((0.9, 0.01), (0.6, 0.16), (0.1, 0.81)):
    chk(f"v10 brier {p}", round((1 - p) ** 2, 2), b)
chk("v10 brier A", r3(sum((y - p) ** 2 for p, y in zip(PA, YA)) / 6), 0.137); chk("v10 brier suma", round(sum((y - p) ** 2 for p, y in zip(PA, YA)), 2), 0.82)
CA = D["s_cal"]
chk("v10 q4 p", r3(CA[3]["p"]), 0.673); chk("v10 q4 f", r3(CA[3]["f"]), 0.814); chk("v10 736", CA[3]["n"], 736, 0)
chk("v10 piensa ll", r3(math.log(2)), 0.693); chk("v10 piensa brier", 0.5 ** 2, 0.25)
diap("A: log loss 0.415", "B: log loss 0.602", "predecir 0.9 aporta 0.105; predecir 0.1 aporta 2.303", "predecir 0.9 aporta 0.01; predecir 0.1 aporta 0.81",
     "0.673", "0.814")
en(10, "0.90", "0.60", "0.20", "0.415", "0.602", "0.835", "0.105", "0.223", "0.916", "2.488", "2.303", "0.81", "0.16", "0.137", "0.673", "0.814",
   "736", "0.693", "0.25")
# ================================================================ video 11 (diapositivas 45-52)
BO = SR["BOSQUE_8"]
chk("v11 auc", r3(BO["auc"]), 0.975); chk("v11 ap", r3(BO["ap"]), 0.966); chk("v11 acc 0.5", round(100 * CT["0.50"]["acc"], 1), 93.1)
P5, P6 = D["s_puntos"]["0.50"], D["s_puntos"]["0.60"]
chk("v11 fpr 0.5", r3(P5["fpr"]), 0.033); chk("v11 rec 0.5", r3(P5["rec"]), 0.875); chk("v11 fpr 0.6", r3(P6["fpr"]), 0.020); chk("v11 rec 0.6", r3(P6["rec"]), 0.825)
chk("v11 prec 50%", round(100 * 400 / 450, 1), 88.9); chk("v11 prec 5%", round(100 * 40 / 135, 1), 29.6)
chk("v11 VP/FP 5%", (0.8 * 50, 0.1 * 950) == (40, 95), 1, 0)
for m, (l, b) in {"FRECUENCIA": (0.671, 0.239), "LOGISTICA_C1": (0.275, 0.075), "BOSQUE_8": (0.228, 0.061)}.items():
    chk(f"v11 logloss {m}", r3(SR[m]["log_loss"]), l); chk(f"v11 brier {m}", r3(SR[m]["brier"]), b)
chk("v11 ganador", D["s_ganador"] == "BOSQUE_8", 1, 0); chk("v11 prev", r3(SP["prev"]), 0.394)
chk("v11 -ln .394", r3(-math.log(0.394)), 0.931); chk("v11 -ln .606", r3(-math.log(0.606)), 0.501)
chk("v11 t1", r3(0.394 * 0.931), 0.367); chk("v11 t2", r3(0.606 * 0.501), 0.304); chk("v11 suma", r3(0.367 + 0.304), 0.671)
chk("v11 brier ref", r3(0.394 * 0.606), 0.239)
chk("v11 piensa", round(100 * 160 / 240, 1), 66.7)
diap("ROC AUC", "0.975 0.976", "0.966 0.967", "88.9%", "29.6%", "0.671 0.275 0.228", "0.239 0.075 0.061", "AUC de 0.975 no significa 97.5 % de aciertos",
     "Promedio de 150 árboles; profundidad máxima 8; mínimo 5 casos por hoja")
en(11, "0.975", "93.1", "0.033", "0.875", "0.020", "0.825", "0.966", "88.9", "29.6", "400", "450", "135", "921", "3680", "0.394", "150",
   "0.671", "0.239", "0.275", "0.075", "0.228", "0.061", "0.931", "0.501", "0.367", "0.304", "0.673", "0.814", "97.5", "66.7")
# ================================================================ video 12 (diapositivas 53-61)
for t, c in {"0.50": 149.7, "0.55": 132.3, "0.60": 128.8, "0.65": 135.3}.items():
    chk(f"v12 costo {t}", round(CT[t]["c1000"], 1), c)
chk("v12 umbral", D["s_umbral"], 0.6); chk("v12 19", len(CT), 19, 0); chk("v12 t=0.05", round(CT["0.05"]["c1000"], 1), 2286.4)
chk("v12 min", min(CT.values(), key=lambda c: (c["c1000"], -c["t"]))["t"], 0.6)
c5, c6 = CT["0.50"], CT["0.60"]
chk("v12 FP/FN", (c5["FP"], c5["FN"], c6["FP"], c6["FN"]) == (74, 181, 44, 254), 1, 0)
chk("v12 acc", (round(100 * c5["acc"], 1), round(100 * c6["acc"], 1)) == (93.1, 91.9), 1, 0)
chk("v12 ahorro", 5 * (74 - 44) - (254 - 181), 77, 0); chk("v12 ahorro/1000", round(77 / 3680 * 1000, 1), 20.9)
chk("v12 dif costos", round(149.7 - 128.8, 1), 20.9)
chk("v12 5/6", r3(5 / 6), 0.833); chk("v12 p=.9", round(5 * (1 - 0.9), 2), 0.5)
chk("v12 p=.7", round(5 * (1 - 0.7), 2), 1.5)
for k, val in dict(VN=553, FP=5, FN=53, VP=310).items():
    chk(f"v12 prueba {k}", SF[k], val, 0)
chk("v12 acc", round(100 * SF["acc"], 1), 93.7); chk("v12 prec", round(100 * SF["prec"], 1), 98.4); chk("v12 rec", round(100 * SF["rec"], 1), 85.4)
chk("v12 logloss", r3(SF["log_loss"]), 0.194); chk("v12 brier", round(SF["brier"], 4), 0.0502)
chk("v12 costo", round(SF["c1000"], 1), 84.7); chk("v12 costo mano", round((5 * 5 + 53) / 921 * 1000, 1), 84.7)
chk("v12 ref", round(SF["ref_c1000"], 1), 394.1); chk("v12 ref mano", round(363 / 921 * 1000, 1), 394.1)
chk("v12 R costo", round((5 * 7 + 51) / 921 * 1000, 1), 93.4); chk("v12 R acc", round(100 * (551 + 312) / 921, 1), 93.7)
chk("v12 piensa", 3 / 4, 0.75)
diap("132.3 133.7", "128.8 123.4", "135.3 132.9", "Entre los 19 umbrales ensayados, ambos eligen t = 0.60", "0.50 74 181", "0.60 44 254",
     "5 × 30 − 73 = 77", "0.833", "No spam 553 5", "Spam 53 310", "84.7 93.4", "cuesta 394.1 por 1000", "93.7 93.7", "98.4 97.8", "85.4 86.0",
     "No spam 551 7", "Spam 51 312")
en(12, "149.7", "128.8", "132.3", "135.3", "0.60", "19", "74", "181", "93.1", "44", "254", "91.9", "30", "73", "77", "20.9", "0.833", "0.9",
   "1.5", "3680", "921", "553", "310", "93.7", "98.4", "85.4", "0.194", "0.0502", "84.7", "394.1", "93.4", "0.75")
# ================================================================ (4) alertas del material
tr = re.sub(r"\s+", " ", (RAIZ / "VideosClases" / "Clase 38 - Evaluacion de modelos supervisados y seleccion de modelos Profesora eloy alvarado.txt")
            .read_text(encoding="utf-8"))
for frase in ("es un número que está entre 0 y 1",                                       # video 2: R² entre 0 y 1
              "al tener un número negativo significa que aquí hay problemas",            # video 2: lo dicho bien despues
              "voy a quedar con 342 composiciones distintas y que hacen o que representan 806 mediciones de las 1.030",   # video 3
              "tener 428 composiciones distintas y voy a reservar el 20% de los grupos",  # video 3: lo dicho bien
              "voy a realizarlo digamos con el conjunto de prueba",                       # video 4
              "en el primero y en el tercero gana alfa igual a 1",                        # video 7
              "en el segundo, en el tercero y en el cuarto",                              # video 7: primera parte correcta
              "son 10 de los 1000 datos",                                                 # video 9
              "los 9.990 restantes",                                                      # video 9
              "voy a ocupar la pérdida cuadrática fuera de pliegue y voy a escoger el ganador",   # video 11
              "umbral 0,55 el costo asociado va a estar en 132",                          # video 12 (dicho bien)
              "el costo va a ser 128",
              "este concepto de mejor modelo es un poco etéreo", "es una receta completa", "es como hacer trampa",
              "no estoy modelando el fenómeno, el problema, sino que estoy modelando mis datos", "va a funcionar como oráculo",
              "de 100 95 le achuntó", "si las probabilidades están bien calibradas"):
    if frase in tr:
        ok += 1
    else:
        mal.append(f"grabacion: no aparece «{frase}»")
diap("usando las filas autorizadas", "Log loss selecciona; Brier describe", "Bajo cero favorece 0.1; sobre cero, 1")
py = (next(p for p in RAIZ.glob("Clase 38*") if p.is_dir()) / "38_evaluacion_seleccion.py").read_text(encoding="utf-8")
for frase in ("ajuste=clone(modelo).fit(x.loc[~valida],y[~valida])", "sort_values(['log_loss','orden'])", "sort_values(['RMSE','orden'])",
              "'costo':(5*fp+fn)/n", "assert set(entrenamiento.grupo).isdisjoint(prueba.grupo)"):
    if frase in py:
        ok += 1
    else:
        mal.append(f"script: no aparece «{frase}»")

print(f"comprobaciones OK: {ok}   fallas: {len(mal)}")
for m in mal:
    print("  FALLA", m)
