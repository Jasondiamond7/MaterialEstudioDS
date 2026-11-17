"""Comprueba los numeros de la serie C40 (Naive Bayes).
(1) Cada cifra real dicha o mostrada se compara con datos40.json (recalculado ejecutando el script de la clase tal cual: datos.py)
    y con las diapositivas (slides_layout.txt, extraido con pdftotext -layout -enc UTF-8).
(2) Cada ejemplo resuelto se recalcula aqui a mano, con los mismos redondeos que se ven en pantalla.
(3) Cada literal se busca en el guion del video donde se dice (narracion + texto en pantalla).
(4) Las alertas del material y las citas se verifican contra la grabacion (transcripcion y .srt con sus marcas de tiempo),
    las diapositivas y el script del profesor.
Ejecutar con cualquier Python 3 (solo usa json/math/re/unicodedata)."""
import json
import math
import re
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
RAIZ = AQUI.parents[2]
D = json.loads((AQUI / "datos40.json").read_text(encoding="utf-8"))
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c40_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)
    return re.sub(r"\[\[\w+\]\]", "", txt).replace("\u202f", " ")


S = {n: say(n) for n in range(1, 13)}
SL = re.sub(r"\s+", " ", (AQUI / "slides_layout.txt").read_text(encoding="utf-8") + " " + (AQUI / "slides.txt").read_text(encoding="utf-8"))


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
r4 = lambda x: round(x, 4)  # noqa: E731
B, BN, DU, GA, MU, SP, O, I1, UM, PR = (D["bayes"], D["bern100"], D["dup"], D["gauss"], D["multi"], D["spam"], D["oof"], D["id1"],
                                        D["umbral"], D["prueba"])
TB = {round(r["t"], 2): r for r in UM["tabla"]}

# ================================================================ video 1 (diapositivas 2-6)
chk("v1 P(oferta|spam)", 16 / 20, 0.80); chk("v1 P(oferta|leg)", 8 / 80, 0.10); chk("v1 complemento", 72 / 80, 0.90)
chk("v1 posterior", r4(16 / 24), 0.6667); chk("v1 posterior real", r4(B["p_oferta"]), 0.6667); chk("v1 leg|oferta", r4(8 / 24), 0.3333)
chk("v1 piensa", r3(4 / 76), 0.053); chk("v1 tabla", 8 + 72 + 16 + 4, 100, 0); chk("v1 total oferta", 8 + 16, 24, 0); chk("v1 sin oferta", 72 + 4, 76, 0)
diap("Legítimo (0) 8 72 80", "16 4 20", "Total 24 76 100", "La palabra aparece en 16 de los 20 spam y en 8 de los 80 correos legítimos.",
     "El 80 % de los spam contiene “oferta”, pero sólo el 66.7 % de los correos con “oferta” es spam.", "P(oferta | spam) = 16 = 0.80")
en(1, "16/20", "0.80", "8/80", "0.10", "0.6667", "66.7", "0.3333", "0.053", "4/76", "setenta y dos", "veinticuatro", "setenta y seis")
# ================================================================ video 2 (diapositivas 7-10)
chk("v2 0.20x0.80", 0.20 * 0.80, 0.16); chk("v2 0.80x0.10", 0.80 * 0.10, 0.08); chk("v2 0.16/0.24", r4(0.16 / 0.24), 0.6667)
chk("v2 prior 0.02 spam", 0.02 * 0.80, 0.016); chk("v2 prior 0.02 leg", 0.98 * 0.10, 0.098); chk("v2 0.1404", r4(0.016 / 0.114), 0.1404)
chk("v2 0.1404 real", r4(B["posterior"][1]), 0.1404); chk("v2 0.6667 real", r4(B["posterior"][0]), 0.6667)
chk("v2 1000 correos", r4(16 / 114), 0.1404); chk("v2 98 leg", 980 * 0.10, 98); chk("v2 16 spam", 20 * 0.80, 16)
chk("v2 98%", r4(B["curva"]["p"][-1]), 0.9975); chk("v2 piensa", r4(0.40 / 0.45), 0.8889); chk("v2 razón", 0.80 / 0.10, 8)
diap("0.20 × 0.80 = 0.16", "0.02 × 0.80 = 0.016", "0.98 × 0.10 = 0.098", "1 − 0.02 = 0.98", "(80/100)(8/80) = 0.08", "0.16/0.24 ≈ 0.6667",
     "Con “oferta”: 20 % → 66.7 % en el primer caso; 2 % → 14.0 % en el segundo.", "La palabra ayuda, pero no decide por sí sola.")
en(2, "0.16", "0.24", "0.6667", "0.3333", "0.016", "0.098", "0.1404", "0.114", "14.0", "66.7", "0.9975", "0.8889", "ciento catorce", "noventa y ocho")
# ================================================================ video 3 (diapositivas 11-13)
chk("v3 0.70", 14 / 20, 0.70); chk("v3 0.30", 6 / 20, 0.30); chk("v3 2^10", 2 ** 10, 1024, 0); chk("v3 2^54", 2 ** 54 / 1e16, 1.8, 0.05)
chk("v3 parametros", 2 ** 10 - 1, 1023, 0); chk("v3 piensa", 2 ** 3, 8, 0); chk("v3 3680", SP["n_ent"], 3680, 0)
diap("Legítimo 80 8 16", "Spam 20 16 14", "14 14/20 = 0.70", "6/20 = 0.30", "1024", "Un cero significa ausencia observada de la señal; no significa dato faltante.")
en(3, "14/20 = 0.70", "6/20 = 0.30", "1024", "1023", "3680", "1.8 × 10¹⁶", "mil veinticuatro", "dieciocho mil billones")
# ================================================================ video 4 (diapositivas 14-18)
chk("v4 0.24", 0.80 * 0.30, 0.24); chk("v4 0.08", 0.10 * 0.80, 0.08); chk("v4 s0", B["patrones"]["10"]["s0"], 0.064); chk("v4 s1", B["patrones"]["10"]["s1"], 0.048)
chk("v4 0.4286", r4(B["patrones"]["10"]["p"]), 0.4286); chk("v4 0.112", 0.20 * 0.80 * 0.70, 0.112); chk("v4 0.016", 0.80 * 0.10 * 0.20, 0.016)
chk("v4 0.8750", r4(B["patrones"]["11"]["p"]), 0.875); chk("v4 00", r4(B["patrones"]["00"]["p"]), 0.0204); chk("v4 01", r4(B["patrones"]["01"]["p"]), 0.1628)
chk("v4 piensa s0", 0.80 * 0.90 * 0.20, 0.144); chk("v4 piensa s1", 0.20 * 0.20 * 0.70, 0.028); chk("v4 piensa", r4(0.028 / 0.172), 0.1628)
chk("v4 64 de 80", 80 - 16, 64, 0)
diap("s0 = 0.80 × 0.10 × 0.80 = 0.064", "s1 = 0.20 × 0.80 × 0.30 = 0.048", "0.048/(0.064 + 0.048) 0.4286", "0.112/(0.016 + 0.112) 0.8750",
     "Con umbral 0.50, su ausencia cambia la decisión a legítimo.", "0.80 × 0.30 = 0.24", "0.10 × 0.80 = 0.08")
en(4, "0.064", "0.048", "0.4286", "0.112", "0.016", "0.8750", "0.0204", "0.1628", "0.144", "0.028", "0.6667", "sesenta y cuatro de ochenta")
# ================================================================ video 5 (diapositivas 19-26)
chk("v5 premio", r4(1 / 22), 0.0455); chk("v5 premio real", r4(BN["premio"]), 0.0455); chk("v5 s0", r4(BN["s_leg"]), 0.0696)
chk("v5 s0 mano", r4(0.80 * 9 / 82 * 65 / 82), 0.0696); chk("v5 s1", r4(BN["s_spam"]), 0.0492); chk("v5 s1 mano", r4(0.20 * 17 / 22 * 7 / 22), 0.0492)
chk("v5 p", r4(BN["p_spam"]), 0.4140); chk("v5 0.7727", r4(17 / 22), 0.7727); chk("v5 0.6818", r4(15 / 22), 0.6818)
chk("v5 diferencia", r4(BN["tabla"][0]["posterior"] - BN["p_spam"]), 0.0146); chk("v5 1/82", r4(1 / 82), 0.0122)
chk("v5 min pres", SP["min_pres"] == [8, 1], 1, 0)
diap("1 ≈ 0.0455.", "Pasamos de 0.4286 a 0.4140.", "16/20 = 0.8000 17/22 ≈ 0.7727", "14/20 = 0.7000 15/22 ≈ 0.6818", "Salida en ambos lenguajes: pspam ≈ 0.4140.",
     "Spam sin enlace: 20 − 14 = 6. Con suavizado: 6 + 1 = 7 sobre 20 + 2 = 22.", "(0.7727, 0.6818). La previa es (0.80, 0.20).")
en(5, "1/22", "0.0455", "9/82", "65/82", "17/22", "7/22", "0.0696", "0.0492", "0.4140", "0.4286", "0.7727", "0.6818", "0.0146", "0.0122", "1/82")
# ================================================================ video 6 (diapositivas 23, 27-30 + script)
chk("v6 3 clases", 0.064 + 0.048 + 0.016, 0.128); chk("v6 leg", 0.064 / 0.128, 0.5); chk("v6 spam", 0.048 / 0.128, 0.375); chk("v6 trabajo", 0.016 / 0.128, 0.125)
chk("v6 log 0.064", r4(math.log(0.064)), -2.7489); chk("v6 log 0.048", r4(math.log(0.048)), -3.0366); chk("v6 dif", r4(math.log(0.048 / 0.064)), -0.2877)
chk("v6 0.75", math.exp(math.log(0.048) - math.log(0.064)), 0.75); chk("v6 0.4286 log", r4(0.75 / 1.75), 0.4286)
chk("v6 0.3^619", D["log"]["p03_619"], 0, 0); chk("v6 0.3^618", D["log"]["p03_618"] > 0, 1, 0); chk("v6 n desborde", D["log"]["n_desborde"], 619, 0)
chk("v6 745.26", r2(D["log"]["log_619"]), -745.26); chk("v6 piensa", 0.12 / 0.20, 0.6); chk("v6 0.0492", r4(0.20 * 17 / 22 * 7 / 22), 0.0492)
diap("Las probabilidades se aprenden una vez durante el ajuste.", "El 0.7727 describe oferta dentro de spam. El 0.4140 describe spam para este",
     "El producto recorre las variables; la suma recorre las clases.", "Con oferta y enlace, d = 2: el numerador para spam es 0.20(17/22)(7/22).")
en(6, "0.4140", "0.7727", "0.20", "0.128", "0.375", "0.125", "2.7489", "3.0366", "0.2877", "0.75", "745.26", "0.4286", "0.6", "seiscientos diecinueve")
# ================================================================ video 7 (diapositivas 31-36)
chk("v7 n", SP["n"], 4601, 0); chk("v7 ent", SP["n_ent"], 3680, 0); chk("v7 prueba", SP["n_prueba"], 921, 0); chk("v7 p", SP["p"], 54, 0)
chk("v7 particion", (SP["particion"]["entrenamiento_0"], SP["particion"]["entrenamiento_1"], SP["particion"]["prueba_0"], SP["particion"]["prueba_1"]) == (2230, 1450, 558, 363), 1, 0)
chk("v7 39.4", r1(100 * SP["prior_spam"]), 39.4); chk("v7 grupos", SP["n_grupos"], 4207, 0); chk("v7 presentes", SP["id1"]["presentes"], 9, 0)
chk("v7 free", SP["id1"]["free"], 0.32); chk("v7 money", SP["id1"]["money"], 0); chk("v7 !", SP["id1"]["excl"], 0.778); chk("v7 id", SP["id1"]["id"], 1, 0)
chk("v7 p id1", r4(I1["posterior"]), 0.9964); chk("v7 piensa", 0.015 * 200, 3); chk("v7 folds", sum(SP["folds"]), 3680, 0)
diap("48 frecuencias de palabras Las 48 columnas", "6 frecuencias de caracteres Las 6 columnas", "Palabra free 0.32 % de las palabras",
     "X contiene 3680 correos y 54 frecuencias", "Desarrollo 3680", "Prueba histórica 921", "Ambos devuelven ID 1 y probabilidad de spam p ≈ 0.9964.",
     "Hopkins et al. (1999), Spambase, UCI", "0.778 % de los caracteres", "Presencia 1, porque 0.32 > 0")
en(7, "4601", "3680", "921", "2230", "1450", "558", "363", "39.4", "4207", "0.32", "0.778", "0.9964", "0.996442", "1.5", "nueve están presentes")
# ================================================================ video 8 (diapositivas 37-40 + script 04)
chk("v8 d0", r4(GA["d0"]), 0.1295); chk("v8 d1", r4(GA["d1"]), 0.3521); chk("v8 s0", r4(GA["s0"]), 0.1036); chk("v8 s1", r4(GA["s1"]), 0.0704)
chk("v8 post", r4(GA["post"]), 0.4046); chk("v8 d0 mano", r4(math.exp(-1.5 ** 2 / 2) / math.sqrt(2 * math.pi)), 0.1295)
chk("v8 d1 mano", r4(math.exp(-0.5 ** 2 / 2) / math.sqrt(2 * math.pi)), 0.3521); chk("v8 2.7", r1(GA["d1"] / GA["d0"]), 2.7)
chk("v8 log1.32", r4(math.log(1.32)), 0.2776); chk("v8 z free real", r4(SP["id1"]["z_free"]), 0.2776); chk("v8 max", SP["freq_max"], 42.81)
chk("v8 log 43.81", r2(math.log1p(42.81)), 3.78); chk("v8 theta", MU["theta"] == [[0.111111, 0.555556, 0.333333], [0.555556, 0.111111, 0.333333]], 1, 0)
chk("v8 multi 1", r4(MU["post"][0]), 0.8333); chk("v8 multi 2", r4(MU["post"][1]), 0.9615); chk("v8 5/6", r4(5 / 6), 0.8333); chk("v8 25/26", r4(25 / 26), 0.9615)
chk("v8 piensa", r4(125 / 126), 0.9921)
diap("log(1.32) ≈ 0.2776", "La altura es una densidad, no la probabilidad de un valor exacto.", "Multinomial conserva las repeticiones; Bernoulli conserva sólo las presencias.",
     "Bernoulli Presencias o ausencias Usar frecuencia > 0", "Gaussiana Mediciones numéricas Probar log(1 + frecuencia)")
en(8, "0.1295", "0.3521", "0.1036", "0.0704", "0.4046", "0.2776", "42.81", "3.78", "0.8333", "0.9615", "5/9", "1/9", "3/9", "0.9921", "125/126", "25/26")
# ================================================================ video 9 (diapositivas 41-45)
for k, (ll, br, au) in {"FRECUENCIA": (0.671, 0.239, 0.499), "BERN_05": (0.556, 0.099, 0.952), "BERN_1": (0.557, 0.099, 0.951),
                        "BERN_2": (0.558, 0.100, 0.951), "GAUSS_LOG": (4.931, 0.190, 0.940)}.items():
    chk(f"v9 {k} ll", r3(O[k]["log_loss"]), ll); chk(f"v9 {k} brier", r3(O[k]["brier"]), br); chk(f"v9 {k} auc", r3(O[k]["auc"]), au)
chk("v9 ganador", D["oof_ganador"] == "BERN_05", 1, 0); chk("v9 acc gauss", r1(100 * O["GAUSS_LOG"]["accuracy"]), 80.8)
chk("v9 25 ajustes", 5 * 5, 25, 0); chk("v9 -log0.9", r3(-math.log(0.9)), 0.105); chk("v9 -log0.1", r3(-math.log(0.1)), 2.303); chk("v9 -log0.001", r3(-math.log(0.001)), 6.908)
chk("v9 fuera gauss", r1(100 * D["oof_extremos"]["gauss_fuera_001"]), 98.6); chk("v9 fuera bern", r1(100 * D["oof_extremos"]["bern05_fuera_001"]), 80.9)
chk("v9 leg 999 g", D["oof_extremos"]["gauss_leg_999"], 592, 0); chk("v9 leg 999 b", D["oof_extremos"]["bern_leg_999"], 27, 0)
chk("v9 ref entropia", r3(-(0.394022 * math.log(0.394022) + 0.605978 * math.log(0.605978))), 0.671)
chk("v9 piensa", r3((-math.log(0.8) - math.log(0.6)) / 2), 0.367); chk("v9 0.223", r3(-math.log(0.8)), 0.223); chk("v9 0.511", r3(-math.log(0.6)), 0.511)
diap("Frecuencia de spam 0.671 0.239 0.499", "Bernoulli, α = 0.5 0.556 0.099 0.952", "Bernoulli, α = 1 0.557 0.099 0.951", "Bernoulli, α = 2 0.558 0.100 0.951",
     "Gaussiana, log(1 + x) 4.931 0.190 0.940", "La gaussiana acierta el 80.8 % con umbral 0.5, pero su log loss es 4.931", "5 × 5 = 25 ajustes",
     "Elegimos Bernoulli con α = 0.5: log loss 0.556.")
en(9, "0.671", "0.239", "0.499", "0.556", "0.099", "0.952", "0.557", "0.558", "4.931", "0.190", "0.940", "80.8", "0.105", "2.303", "6.908", "98.6", "80.9",
   "quinientos noventa y dos", "veintisiete", "0.367", "0.223", "0.511", "veinticinco ajustes")
# ================================================================ video 10 (diapositivas 46-50)
chk("v10 0.8000", DU["post"][0], 0.8); chk("v10 0.9412", r4(DU["post"][1]), 0.9412); chk("v10 0.9961", r4(DU["post"][2]), 0.9961)
chk("v10 16/17", r4(16 / 17), 0.9412); chk("v10 256/257", r4(256 / 257), 0.9961); chk("v10 64/65", r4(64 / 65), 0.9846)
DR = {d["k"]: d for d in D["dup_real"]}
chk("v10 dup 1", r3(DR[1]["log_loss"]), 0.557); chk("v10 dup 2", r3(DR[2]["log_loss"]), 1.042); chk("v10 dup 4", r3(DR[4]["log_loss"]), 1.876)
chk("v10 dup auc", r3(DR[4]["auc"]), 0.951); chk("v10 auc igual", DR[1]["auc"] == DR[2]["auc"] == DR[4]["auc"], 1, 0)
C9 = D["cal9"]
chk("v10 cal n", C9["n"], 1157, 0); chk("v10 cal pred", r2(100 * C9["prob"]), 99.44); chk("v10 cal obs", r2(100 * C9["frac"]), 92.13); chk("v10 cal spam", C9["spam"], 1066, 0)
chk("v10 address 0", r4(227 / 2232), 0.1017); chk("v10 address 1", r4(499 / 1452), 0.3437); chk("v10 address real", (I1["address"]["c0"], I1["address"]["c1"]) == (226, 498), 1, 0)
chk("v10 free p0", r4(I1["free"]["p0"]), 0.0959); chk("v10 free p1", r4(I1["free"]["p1"]), 0.5558)
chk("v10 prior log", r3(I1["prior_log"]), -0.430); chk("v10 suma", r3(I1["suma_aportes"]), 6.065); chk("v10 total", r3(I1["log_odds"]), 5.635)
chk("v10 sigmoide", r4(1 / (1 + math.exp(-I1["log_odds"]))), 0.9964); chk("v10 lados", (I1["n_spam_lado"], I1["n_leg_lado"]) == (34, 20), 1, 0)
diap("= 0.8000.", "≈ 0.9412.", "≈ 0.9961.", "1157 correos.", "Predicción media: 99.44 %.", "1066 ≈ 92.13 %.", "226 + 1 ≈ 0.1017", "498 + 1 ≈ 0.3437",
     "pspam = 0.9964.", "Address presente favorece spam: 0.3437 > 0.1017.")
en(10, "0.8000", "0.9412", "0.9961", "4 : 1", "16 : 1", "256 : 1", "1.042", "1.876", "0.951", "99.44", "92.13", "1066/1157", "0.1017", "0.3437", "0.0959",
   "0.5558", "6.065", "5.635", "0.9964", "0.9846", "64/65", "mil ciento cincuenta y siete")
# ================================================================ video 11 (diapositivas 51-54)
chk("v11 p0.90", 5 * (1 - 0.90), 0.50); chk("v11 p0.70", 5 * (1 - 0.70), 1.50); chk("v11 5/6", r3(5 / 6), 0.833); chk("v11 2/3", r3(2 / 3), 0.667)
chk("v11 t", UM["t"], 0.95); chk("v11 FP", TB[0.95]["FP"], 78, 0); chk("v11 FN", TB[0.95]["FN"], 416, 0); chk("v11 costo", r1(TB[0.95]["costo"]), 219.0)
chk("v11 costo mano", r1(1000 * (5 * 78 + 416) / 3680), 219.0); chk("v11 806", 5 * 78 + 416, 806, 0)
for t, (fp, fn, c) in {0.5: (152, 263, 278.0), 0.8: (111, 328, 239.9), 0.9: (92, 383, 229.1)}.items():
    chk(f"v11 {t} FP", TB[t]["FP"], fp, 0); chk(f"v11 {t} FN", TB[t]["FN"], fn, 0); chk(f"v11 {t} costo", r1(TB[t]["costo"]), c)
    chk(f"v11 {t} mano", r1(1000 * (5 * fp + fn) / 3680), c)
chk("v11 0.05", r1(TB[0.05]["costo"]), 448.9); chk("v11 base OOF", r1(O["FRECUENCIA"]["costo"]), 394.0); chk("v11 base mano", r1(1000 * 1450 / 3680), 394.0)
chk("v11 acc 0.5", r1(100 * TB[0.5]["acc"]), 88.7); chk("v11 acc 0.95", r1(100 * TB[0.95]["acc"]), 86.6)
chk("v11 matriz", (PR["VN"], PR["FP"], PR["FN"], PR["VP"]) == (541, 17, 89, 274), 1, 0); chk("v11 prueba costo", r1(PR["costo"]), 188.9)
chk("v11 prueba mano", r1(1000 * (5 * 17 + 89) / 921), 188.9); chk("v11 base prueba", r1(PR["base_costo"]), 394.1); chk("v11 base mano2", r1(1000 * 363 / 921), 394.1)
chk("v11 ll prueba", r3(PR["log_loss"]), 0.533); chk("v11 brier prueba", r3(PR["brier"]), 0.094); chk("v11 363", 89 + 274, 363, 0)
chk("v11 min en rejilla", min(UM["tabla"], key=lambda r: (r["costo"], -r["t"]))["t"], 0.95)
diap("5(1 − 0.90) = 0.50 1(0.90) = 0.90 Bloquear", "5(1 − 0.70) = 1.50 1(0.70) = 0.70 Dejar pasar", "Entre 19 umbrales, de 0.05 a 0.95, elegimos 0.95.",
     "Con t = 0.95: 78 legítimos bloqueados y 416 spam que pasan, entre 3680 correos.", "= 219.0.", "Legítimo 541 17", "89 274", "= 188.9.", "= 394.1.",
     "Log loss: 0.533. Brier: 0.094.", "Los 89 + 274 = 363 spam pasan:")
en(11, "0.50", "1.50", "0.833", "219.0", "806", "278.0", "239.9", "229.1", "448.9", "394.0", "88.7", "86.6", "188.9", "394.1", "0.533", "0.094", "0.667",
   "setenta y ocho", "cuatrocientos dieciséis", "quinientos cuarenta y uno", "diecisiete", "ochenta y nueve", "doscientos setenta y cuatro")
# ================================================================ video 12 (diapositivas 55-61)
chk("v12 rebalance", r4(0.5 * 0.8 / (0.5 * 0.8 + 0.5 * 0.1)), 0.8889)
diap("Con oferta: 66.7 % → 14.0 % al", "Bloquear cuando la probabilidad de spam sea al menos 0.95", "Frecuencias por clase con suavizado α = 0.5",
     "Aún no: que el 99.64 % de correos similares sea realmente spam", "bernoulli_naive_bayes() BernoulliNB()", "Suavizado: laplace en R; alpha en Python.",
     "Suavizado y umbral cumplen tareas distintas")
en(12, "0.6667", "0.8889", "0.95", "0.9964", "99.64", "0.4140", "0.556", "0.1404", "0.4286", "80.8", "4.931", "99.44", "92.13", "219.0", "188.9", "394.1",
   "42.81")

# ================================================================ (4) citas y alertas: grabacion (.txt y .srt con marca de tiempo)
VC = RAIZ / "VideosClases"
tr_raw = next(VC.glob("Clase 40*.txt")).read_text(encoding="utf-8")


def nz(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.replace("%", " % ").replace(",", " ").replace(".", " . ")
    t = re.sub(r"[^\w%. ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


TR = nz(tr_raw)
srt = next(VC.glob("Clase 40*.srt")).read_text(encoding="utf-8").strip().split("\n\n")
BL = []
for b in srt:
    ls = b.strip().split("\n")
    if len(ls) >= 3:
        h, m, s = ls[1].split(" --> ")[0].split(":")
        BL.append((int(h) * 3600 + int(m) * 60 + float(s.replace(",", ".")), nz(" ".join(ls[2:]))))


def t_hms(h):
    a, b, c = h.split(":")
    return int(a) * 3600 + int(b) * 60 + int(c)


def cita_ok(hms, frase, tol=60):
    """La frase (o sus partes separadas por ' ... ') aparece en la transcripcion, y empieza en un bloque .srt a menos de tol s de hms."""
    global ok
    partes = [nz(p) for p in frase.split("...")]
    if not all(p in TR for p in partes):
        mal.append(f"grabacion: no aparece «{frase}»")
        return
    ini = " ".join(partes[0].split()[:5])
    ts = [BL[i][0] for i in range(len(BL)) if ini in " ".join(x[1] for x in BL[i:i + 4]) and ini not in " ".join(x[1] for x in BL[i + 1:i + 4])]
    if any(abs(t - t_hms(hms)) <= tol for t in ts):
        ok += 1
    else:
        mal.append(f"grabacion: «{frase[:50]}» no cerca de {hms} (bloques en {[round(t) for t in ts]})")


CITAS = [("00:07:10", "el acto de predecir y el acto de tomar una decisión son cosas distintas", 1),
         ("00:26:28", "pasamos de tener casi 67 % de probabilidad ... casi un 14 % de que sea spam", 2),
         ("00:26:28", "yo deje todo constante exceptuando la proporción de spam", 2),
         ("00:29:56", "qué pasa si 98% de los correos son spam", None),
         ("00:32:47", "yo no sé de esta tabla descriptiva si es que estos ocho de oferta pertenecen a los ocho con el enlace", 3),
         ("00:32:22", "dentro de spam 14 no tienen enlace", 3),
         ("00:34:55", "la categoría que yo clasifique como uno, que es cuando tiene un enlace presente, son 14", None),
         ("00:39:11", "si no pasó en mi datos entonces la frecuencia es cero, y por consecuencia la probabilidad es cero", 3),
         ("00:43:24", "nada me asegura eso, por eso es una metodología que es, digamos, ingenua", 4),
         ("00:49:54", "6 correos que no tienen enlace dentro de los 20 que son legítimos", 4),
         ("01:15:57", "puede que no pase mucho porque puede que no manden muchos mensajes pero cuando pasa yo sé y estoy muy seguro", 5),
         ("01:07:22", "20 de los spam que son observados son 80 de los 100", 5),
         ("01:14:43", "lambda igual a 1", None),
         ("01:14:58", "a modo global me cambió casi en 1 %", 5),
         ("01:11:01", "cuando usted aprieta en esto marcar como spam lo que están haciendo es clasificar una observación más", 6),
         ("01:32:34", "para el ID1 por ejemplo vuelve con una probabilidad de 0.9954", 7),
         ("01:36:27", "a lo más pueden ser 0 y a lo más pueden ser 1", 8),
         ("01:36:48", "entonces pueden ser muy grandes", None),
         ("01:43:43", "por qué me voy a complicar la vida si simplemente puedo sacar hacer dos divisiones", 9),
         ("01:48:57", "cuando lo evalúe estas metodologías en prueba, no hay mucha discrepancia", 9),
         ("01:44:57", "esto lo voy a evaluar en el conjunto de prueba", None),
         ("01:52:16", "tengo 200 correos en mi conjunto de datos, de los cuales 80 son spam", 10),
         ("01:54:10", "tengo dos predictores y son lo mismo porque me entregan la misma información", None),
         ("01:57:37", "Bernoulli está bastante más cercano que el caso gausiano", None),
         ("02:01:47", "sí es un sobre ajuste porque estoy contando dos veces lo mismo", 10),
         ("02:03:32", "cincuenta y tanto por ciento de dentro de spam y casi 0,1 dentro de legítimo", None),
         ("02:18:52", "si el costo de equivocarse es muy grande voy a querer estar muy muy seguro", 11),
         ("02:09:50", "elijo un umbral muy bajo", None),
         ("02:00:33", "después tengo que compensar ese balanceo que yo hice ya porque si no voy a sobreestimar cosas", 12)]
for hms, frase, vid in CITAS:
    cita_ok(hms, frase)
    if vid:                                   # la cita, tal como se muestra, y su marca de tiempo estan en el guion de ese video
        en(vid, hms)
# las marcas de tiempo usadas en pantalla (grabacion(...) y alertas) son exactamente las de la lista
usadas = sorted({m for n in S for m in re.findall(r"\b\d\d:\d\d:\d\d\b", S[n])})
lista = {h for h, _f, _v in CITAS}
for m in usadas:
    if m in lista:
        ok += 1
    else:
        mal.append(f"marca de tiempo {m} usada en un guion sin cita verificada")
# alertas: la diapositiva o el script contradicen lo dicho
diap("Spam 20 16 14")                                                     # 00:32:22 y 00:49:54: 14 spam con enlace
chk("alerta 1.5 puntos", r2(100 * (0.428571 - 0.414006)), 1.46)           # 01:14:58
chk("alerta 0.9964 (no 0.9954)", r4(I1["posterior"]) != 0.9954, 1, 0)     # 01:32:34
chk("alerta porcentajes > 1", SP["freq_max"] > 1, 1, 0)                   # 01:36:27
diap("Experimento controlado: 100 spam y 100 legítimos; una señal aparece en 80 spam y", "La selección usa desarrollo; los 921 correos de prueba histórica se mantienen aparte.",
     "El correo y su información son los mismos. El modelo contó dos veces una")
# ================================================================ script de la clase
py = (next(p for p in RAIZ.glob("Clase 40*") if p.is_dir()) / "40_naive_bayes.py").read_text(encoding="utf-8")
for frase in ("s_legitimo = (80/100)*(9/82)*(65/82)", "s_spam = (20/100)*(17/22)*(7/22)", "theta = (conteos+alfa)/(n[:,None]+2*alfa)",
              "def posterior(log_conjunta):", "z = log_conjunta - np.max(log_conjunta, axis=1, keepdims=True)", "lp=modelo.predict_log_proba(xv)",
              "# Logaritmos estables evitan restar de uno probabilidades casi unitarias.", "BernoulliNB(alpha=valor,binarize=0.0)",
              "GaussianNB(var_smoothing=valor).fit(np.log1p(xa),ya)", "umbrales=np.arange(1,20)/20", "costo_1000=1000*(5*fp+fn)/len(y)",
              "cuentas=np.array([[0.,4.,2.],[4.,0.,2.]]) # oferta, reunion, hoy", "theta_m=(cuentas+1)/(cuentas.sum(axis=1,keepdims=True)+3)",
              "mensajes=np.array([[1.,0.,1.],[2.,0.,1.]])", "densidad_0 = np.exp(-1.5**2/2)/np.sqrt(2*np.pi)", "s0_gauss, s1_gauss = .8*densidad_0, .2*densidad_1",
              "('BERN_05','bern',.5),('BERN_1','bern',1.),('BERN_2','bern',2.),('GAUSS_LOG','gauss',1e-9)", "copias = np.array([1,2,4])",
              "senal_spam, senal_legitimo = 80/100, 20/100", "ascending=[True,False]"):
    if frase in py:
        ok += 1
    else:
        mal.append(f"script: no aparece «{frase}»")

print(f"comprobaciones OK: {ok}   fallas: {len(mal)}")
for m in mal:
    print("  FALLA", m)
