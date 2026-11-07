"""Comprueba los numeros de la serie C35.
(1) Cada cifra real dicha o mostrada se compara con datos35.json (recalculado con la base real: datos.py).
(2) Cada ejemplo resuelto se recalcula aqui a mano, con los mismos redondeos intermedios que se ven en pantalla.
(3) Cada literal se busca en el guion del video donde se dice (subtitulos = texto de pantalla de la narracion).
(4) Las afirmaciones sobre errores del material se verifican contra el notebook y la API de scikit-learn.
Ejecutar con el venv de Clase 35 (tiene scikit-learn)."""
import json
import math
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
D = json.loads((AQUI / "datos35.json").read_text(encoding="utf-8"))
F = D["fig"]
ok, mal = 0, []
NB = " "


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c35_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)     # texto en pantalla
    return re.sub(r"\[\[\w+\]\]", "", txt).replace(NB, " ")


S = {n: say(n) for n in range(1, 12)}


def en(n, *lits):
    global ok
    for l in lits:
        if l in S[n]:
            ok += 1
        else:
            mal.append(f"video {n}: no aparece «{l}»")


def r3(x):
    return f"{x:.3f}"


def cien(x):
    """dolares -> texto redondeado a centenas como se narra: 26004.9 -> '26 000'."""
    v = int(round(x / 100.0)) * 100
    return f"{v:,}".replace(",", " ")


# ================================================================ video 1
chk("Heart n", D["heart_n"], 303, 0); chk("Heart p", D["heart_p"], 13, 0); chk("Heart si", D["heart_si"], 139, 0)
chk("46 %", 139 / 303 * 100, 46, 0.5); chk("train 227", D["heart_tr"], 227, 0); chk("test 76", D["heart_te"], 76, 0)
RA = F["boot"]["raices"]
chk("Thal 48.5", RA["Thal"] * 100, 48.5, 0.05); chk("ChestPain 30", RA["ChestPain"] * 100, 30, 0.05); chk("Ca 20", RA["Ca"] * 100, 20, 0.05)
chk("desacuerdo 26 %", F["boot"]["desacuerdo"] * 100, 26, 0.5)
AP = D["prop_auc_promedio"]
for lit, val in (("0.744", D["prop_auc_un_arbol_media"]), ("0.586", D["prop_auc_un_arbol_min"]), ("0.817", D["prop_auc_un_arbol_max"]),
                 ("0.878", AP["5"]), ("0.894", AP["25"]), ("0.905", AP["100"])):
    chk(f"v1 {lit}", r3(val), lit, 0); en(1, lit)
en(1, "303 pacientes", "139 de 303", "48.5 %", "26 %", "76 pacientes")

# ================================================================ video 2 (bootstrap)
cad = [0.875]
for _ in range(7):
    cad.append(round(cad[-1] * 0.875, 3))
chk("(7/8)^2 redondeado", cad[1], 0.766); chk("(7/8)^4 redondeado", cad[3], 0.586); chk("(7/8)^8 redondeado", cad[7], 0.344)
chk("(7/8)^8 exacto", round((7 / 8) ** 8, 3), 0.344)
chk("oob n=8", F["boot"]["oob8"], 0.344); chk("oob n=227", F["boot"]["oob227"], 0.367); chk("oob n=2344", F["boot"]["oob2344"], 0.368)
chk("e^-1", math.exp(-1), 0.368); chk("1-e^-1 = 63.2 %", (1 - math.exp(-1)) * 100, 63.2, 0.05)
chk("Ames train 2344", D["ames_tr"], 2344, 0)
OB = F["boot"]["obs"]
chk("0 veces 36.8", OB[0] * 100, 36.8, 0.05); chk("1 vez 36.7", OB[1] * 100, 36.7, 0.05); chk("2 veces 18.4", OB[2] * 100, 18.4, 0.05)
chk("3+ ~8 %", sum(OB[3:]) * 100, 8.1, 0.15)
chk("oob media 36.8", F["boot"]["oob_media"] * 100, 36.8, 0.05); chk("oob sd 2.0", F["boot"]["oob_sd"] * 100, 2.0, 0.05)
b1, b2, b3 = [1, 2, 2, 4, 6, 6, 7, 8], [1, 1, 3, 3, 4, 5, 7, 7], [2, 3, 4, 4, 5, 6, 6, 8]
chk("b1 distintas 6", len(set(b1)), 6, 0); chk("b2 distintas 5", len(set(b2)), 5, 0); chk("b3 distintas 6", len(set(b3)), 6, 0)
chk("b1 afuera {3,5}", sorted(set(range(1, 9)) - set(b1)) == [3, 5], 1, 0)
chk("b2 afuera {2,6,8}", sorted(set(range(1, 9)) - set(b2)) == [2, 6, 8], 1, 0)
chk("b3 afuera {1,7}", sorted(set(range(1, 9)) - set(b3)) == [1, 7], 1, 0)
chk("esperado 8*(1-0.344)", 8 * (1 - 0.344), 5.25, 0.005); chk("promedio diapositiva 17/3", 17 / 3, 5.67, 0.005)
en(2, "0.875", "0.766", "0.586", "0.344", "0.367", "0.368", "63.2 %", "36.8 %", "36.7 %", "18.4 %", "5.25", "5.67", "2 344 casas")

# ================================================================ video 3 (OOB)
chk("0.368*120=44", round(0.368 * 120), 44, 0); chk("0.368*500=184", round(0.368 * 500), 184, 0); chk("0.368*300=110", round(0.368 * 300), 110, 0)
# ejemplo a mano: votos OOB
VOT = {"P1": ["Sí", "Sí"], "P2": ["No", "No"], "P3": ["No", "No", "Sí"], "P4": ["Sí", "No", "No"]}
REAL = {"P1": "Sí", "P2": "No", "P3": "Sí", "P4": "No"}
pred = {k: max(set(v), key=v.count) for k, v in VOT.items()}
err = sum(pred[k] != REAL[k] for k in REAL) / 4
chk("error OOB a mano 1/4", err, 0.25, 0); chk("exactitud OOB 0.75", 1 - err, 0.75, 0)
chk("0.632^10 = 1.0 %", 0.632 ** 10 * 100, 1.0, 0.05); chk("0.632^30 ≈ 0.0001 %", 0.632 ** 30 * 100, 0.0001, 0.00005)
H, A = F["oobcv"]["heart"], F["oobcv"]["ames"]
for lit, val in (("0.811", H["oob"]), ("0.798", H["cv"]), ("0.855", H["test"])):
    chk(f"v3 {lit}", r3(val), lit, 0); en(3, lit)
chk("cv sd ≈ 0.05", H["cv_sd"], 0.05, 0.005)
for lit, val in (("26 000", A["oob"]), ("26 300", A["cv"]), ("27 000", A["test"]), ("27 300", F["curvas"]["oob_rf"]["300"]), ("23 100", F["curvas"]["rf"]["300"])):
    chk(f"v3 {lit}", cien(val) == lit, 1, 0); en(3, lit)

# ================================================================ video 4 (bagging)
chk("frasco promedio", sum([310, 520, 380, 700, 450, 260]) / 6, 436.7, 0.05)
chk("3/5", 3 / 5, 0.60); PR = [0.80, 0.30, 0.70, 0.45, 0.20]
chk("suma 2.45", sum(PR), 2.45, 1e-9); chk("promedio 0.49", sum(PR) / 5, 0.49, 1e-9); chk("votos 2/5", sum(p > 0.5 for p in PR) / 5, 0.40, 0)
chk("umbral 0.45: promedio Sí", sum(PR) / 5 >= 0.45, 1, 0); chk("umbral 0.45: votos No", 0.40 >= 0.45, 0, 0)
chk("precios suman 1000", sum([180, 210, 195, 240, 175]), 1000, 0); chk("mediana 195", sorted([180, 210, 195, 240, 175])[2], 195, 0)
CH, CA = F["cmp_h"], F["cmp_a"]
for lit, val in (("0.753", CH["Árbol"][0]), ("0.893", CH["Bagging de árboles"][0]), ("0.913", CH["Reg. logística"][0]), ("0.912", CH["Bagging de logísticas"][0])):
    chk(f"v4 {lit}", r3(val), lit, 0); en(4, lit)
for lit, val in (("45 800", CA["Árbol"][0]), ("28 900", CA["Bagging"][0])):
    chk(f"v4 {lit}", cien(val) == lit, 1, 0); en(4, lit)
chk("+0.14 AUC", CH["Bagging de árboles"][0] - CH["Árbol"][0], 0.14, 0.005); chk("casi 17 000", CA["Árbol"][0] - CA["Bagging"][0], 17000, 100)

# ================================================================ video 5 (varianza)
var = lambda r, s2, b: r * s2 + (1 - r) * s2 / b  # noqa: E731
for b, e in ((1, 64), (4, 40), (16, 34)):
    chk(f"rho .5 B={b}", var(0.5, 64, b), e, 1e-9)
for b, e in ((1, 64), (4, 20.8), (16, 10)):
    chk(f"rho .1 B={b}", var(0.1, 64, b), e, 1e-9)
chk("piso .5", 0.5 * 64, 32, 0); chk("piso .1", 0.1 * 64, 6.4, 1e-9); chk("cinco veces", 32 / 6.4, 5, 1e-9)
chk("opcion 1 = 32.03", var(0.5, 64, 1000), 32.03, 0.005); chk("opcion 2 = 10", var(0.1, 64, 16), 10, 1e-9)
VB, VR, V10 = F["var"]["Bagging (todas)"], F["var"]["RF (1/3)"], F["var"]["RF (10 %)"]
for lit, val in (("36 600", VB["sigma"]), ("13 900", VB["formula"]["10"]), ("13 900", VB["medido"]["10"]), ("8 800", VB["formula"]["100"]),
                 ("8 800", VB["medido"]["100"]), ("8 100", VB["piso"]), ("5 800", VR["piso"]), ("4 400", V10["piso"])):
    chk(f"v5 {lit}", cien(val) == lit, 1, 0); en(5, lit)
chk("rho bagging .049", VB["rho"], 0.049, 0); chk("rho RF1/3 .024", VR["rho"], 0.024, 0); chk("rho RF10 .011", V10["rho"], 0.011, 0)
en(5, "0.049", "0.024", "0.011", "32.03")
# la formula reproduce lo medido (comprobacion independiente con sigma y rho redondeados)
chk("formula B=10 con sigma, rho", math.sqrt(var(VB["rho"], VB["sigma"] ** 2, 10)), 13886, 60)

# ================================================================ video 6 (Random Forest)
chk("por arbol 27 100", cien(D["extra_por_arbol_10"]) == "27 100", 1, 0); chk("por nodo 22 500", cien(D["extra_por_nodo_10"]) == "22 500", 1, 0)
en(6, "27 100", "22 500")
chk("4/10", 4 / 10, 0.40); chk("1-0.4", 1 - 0.4, 0.60); chk("0.6^3", 0.6 ** 3, 0.216); chk("0.6^10", round(0.6 ** 10, 3), 0.006)
chk("sqrt 18 = 4.24", math.sqrt(18), 4.24, 0.005); chk("int(sqrt 18) = 4", int(math.sqrt(18)), 4, 0); chk("Heart 18 columnas", D["heart_cols_oh"], 18, 0)
en(6, "0.40", "0.216", "0.006", "4.24", "36 600", "37 900", "0.049", "0.024")
chk("sigma RF1/3 37 900", cien(VR["sigma"]) == "37 900", 1, 0)

# ================================================================ video 7 (max_features)
AM = F["mf"]["ames"]
chk("341 columnas", F["mf"]["ncols_ames"], 341, 0)
for lit, k in (("27 700", "0.010"), ("23 600", "0.050"), ("22 500", "0.100"), ("23 100", "0.333"), ("24 300", "1.000")):
    chk(f"v7 {lit}", cien(AM[k][0]) == lit, 1, 0); en(7, lit)
chk("minimo en 10 %", float(min(AM, key=lambda k: AM[k][0])), 0.1, 1e-9)
chk("0.10*341", 0.10 * 341, 34.1, 1e-9); chk("int 34", int(0.10 * 341), 34, 0); chk("sqrt 341", math.sqrt(341), 18.47, 0.005)
chk("int sqrt 341", int(math.sqrt(341)), 18, 0); chk("341/3", 341 / 3, 113.67, 0.005); chk("int 341/3", int(341 / 3), 113, 0)
HE = F["mf"]["heart"]
for lit, m in (("0.904", "1"), ("0.902", "3"), ("0.882", "18")):
    chk(f"v7 heart {lit}", f"{HE[m][1]:.3f}", lit, 0); en(7, lit)
chk("rango ≈ 0.02", HE["1"][1] - HE["18"][1], 0.02, 0.005)
NA = D["nb_ames"]
for lit, k in (("29 300", "bag"), ("29 100", "rf10"), ("26 700", "rf13")):
    chk(f"v7 nb {lit}", cien(NA[k]["rmse"]) == lit, 1, 0); en(7, lit)

# ================================================================ video 8 (hiperparametros)
CU = F["curvas"]
for lit, val in (("36 100", CU["tree"]), ("37 300", CU["rf"]["1"]), ("24 400", CU["rf"]["10"]), ("23 300", CU["rf"]["50"]), ("23 100", CU["rf"]["100"]),
                 ("23 100", CU["rf"]["300"])):
    chk(f"v8 {lit}", cien(val) == lit, 1, 0); en(8, lit)
chk("B 2 % = 34", CU["b95_rf"], 34, 0); chk("3x2x2x2", 3 * 2 * 2 * 2, 24, 0); chk("24x5", 24 * 5, 120, 0)
GR = D["nb_heart_grid_rf"]
chk("grilla 24 combinaciones", GR["n_comb"], 24, 0)
chk("base 0.897", r3(GR["base_cv"]), "0.897", 0); chk("mejor 0.908", r3(GR["best"]), "0.908", 0); chk("+0.011", GR["best"] - GR["base_cv"], 0.011, 0.0006)
chk("mejor: 600 arboles", GR["params"]["model__n_estimators"], 600, 0); chk("mejor: hoja 3", GR["params"]["model__min_samples_leaf"], 3, 0)
chk("46 % positivos", D["nb_heart_pos_tr"] * 100, 46, 0.5)
chk("train acc 1.000", D["prop_rf_train_acc"], 1.0, 0); chk("test 0.855", r3(D["prop_rf_test_acc"]), "0.855", 0); chk("oob 0.811", r3(D["prop_rf_oob_acc"]), "0.811", 0)
en(8, "0.897", "0.908", "1.000", "0.855", "0.811", "24 combinaciones", "24 por 5")

# ================================================================ video 9 (importancia y limites)
gA = round(1 - 0.9 ** 2 - 0.1 ** 2, 4); gD = round(1 - 0.26 ** 2 - 0.74 ** 2, 4)
chk("27/3 = 0.9", 27 / 30, 0.9, 1e-12); chk("13/50 = 0.26", 13 / 50, 0.26, 1e-12)
chk("Gini 0.18", gA, 0.18, 1e-9); chk("Gini 0.3848", gD, 0.3848, 1e-9)
pondA = round((30 * gA + 50 * gD) / 80, 3); chk("pond A 0.308", pondA, 0.308, 1e-9); dA = round(0.5 - pondA, 3); chk("dA 0.192", dA, 0.192, 1e-9)
gB1 = round(1 - 0.55 ** 2 - 0.45 ** 2, 4); gB2 = round(1 - (2 / 30) ** 2 - (28 / 30) ** 2, 4)
chk("Gini 0.495", gB1, 0.495, 1e-9); chk("Gini 0.1244", gB2, 0.1244, 1e-9)
pondB = round((20 * gB1 + 30 * gB2) / 50, 4); chk("pond B 0.2726", pondB, 0.2726, 1e-9)
dB = round(gD - pondB, 4); chk("dB 0.1122", dB, 0.1122, 1e-9); aB = round(dB * 50 / 80, 4); chk("aporte B 0.0701", aB, 0.0701, 1e-9)
tot = round(dA + aB, 4); chk("total 0.2621", tot, 0.2621, 1e-9)
chk("A 0.733", round(dA / tot, 3), 0.733, 1e-9); chk("B 0.267", round(aB / tot, 3), 0.267, 1e-9)
en(9, "0.18", "0.3848", "0.308", "0.192", "0.495", "0.1244", "0.2726", "0.1122", "0.0701", "0.2621", "0.733", "0.267")
IM = F["imp"]
chk("ruido puesto 7", IM["rank_ruido_cont"], 7, 0); chk("ruido 6.9 %", IM["mdi_ruido"]["Ruido_continuo"] * 100, 6.9, 0.05)
chk("ruido perm 0.001", round(IM["perm_ruido"]["Ruido_continuo"], 3), 0.001, 1e-9)
chk("Gr_Liv_Area perm 22.5", IM["ames_perm"]["Gr_Liv_Area"] * 100, 22.5, 0.05); chk("+11 900", cien(IM["ames_perm_rmse"]["Gr_Liv_Area"]) == "11 900", 1, 0)
chk("Gr_Liv_Area MDI 12.2", IM["ames_mdi"]["Gr_Liv_Area"] * 100, 12.2, 0.05)
chk("Garage_Cars primero MDI", max(IM["ames_mdi"], key=IM["ames_mdi"].get) == "Garage_Cars", 1, 0)
EX = D["prop_extrap"]
chk("max entrenado 392 000", EX["max_y_entrenado"], 392000, 0); chk("133 casas grandes", EX["n_grandes"], 133, 0)
for lit, val in (("316 700", EX["media_real_grandes"]), ("222 900", EX["media_pred_grandes"]), ("337 500", EX["max_pred_grandes"])):
    chk(f"v9 {lit}", cien(val) == lit, 1, 0); en(9, lit)
chk("94 000 menos", EX["media_real_grandes"] - EX["media_pred_grandes"], 94000, 500)
chk("max pred < max visto", EX["max_pred_grandes"] < EX["max_y_entrenado"], 1, 0)
IMG = F["img"]
chk("digitos test 96.9", IMG["test"] * 100, 96.9, 0.05); chk("desordenados 97.3", IMG["shuffled"] * 100, 97.3, 0.05); chk("corridos 50.4", IMG["shifted"] * 100, 50.4, 0.05)
en(9, "6.9 %", "22.5 %", "12.2 %", "11 900", "96.9 %", "97.3 %", "50.4 %", "94 000")

# ================================================================ video 10 (comparacion)
for lit, k in (("0.753", "Árbol"), ("0.893", "Bagging de árboles"), ("0.906", "Random Forest"), ("0.913", "Reg. logística")):
    chk(f"v10 {lit}", r3(CH[k][0]), lit, 0); en(10, lit)
chk("diferencia log-RF < 0.01", CH["Reg. logística"][0] - CH["Random Forest"][0], 0.0063, 0.0005)
chk("desvios ≈ 0.035", CH["Reg. logística"][1], 0.035, 0.002)
for lit, val in (("45 800", CA["Árbol"][0]), ("28 900", CA["Bagging"][0]), ("27 000", CA["Random Forest (1/3)"][0]), ("9 700", CA["Random Forest (1/3)"][1])):
    chk(f"v10 {lit}", cien(val) == lit, 1, 0); en(10, lit)
chk("casi tres veces", CA["Random Forest (1/3)"][0] / CA["Random Forest (1/3)"][1], 2.8, 0.05)
NH = D["nb_heart"]
for lit, val in (("0.905", NH["bag"]["auc"]), ("0.928", NH["rf"]["auc"]), ("0.941", GR["test_auc"])):
    chk(f"v10 nb {lit}", r3(val), lit, 0); en(10, lit)
chk("Ames 2930", D["ames_n"], 2930, 0); chk("Ames 80 variables", D["ames_p"], 80, 0); chk("Ames test 586", D["ames_te"], 586, 0)

# ================================================================ video 11 (repaso)
en(11, "63.2 %", "36.8 %", "0.811", "0.798", "29 100", "29 300", "96.9 %", "50.4 %")

# ================================================================ (4) errores del material: contra el notebook y la API
nb = json.loads(next((AQUI.parents[2] ).glob("Clase 35*/Ejemplo Random Forest y Bagging.ipynb")).read_text(encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
chk("notebook: RandomForestRegressor(max_features=1.0)", bool(re.search(r"RandomForestRegressor\(\s*n_estimators=120,\s*max_features=1\.0", src)), 1, 0)
chk("notebook: class_weight solo en RF", src.count('class_weight="balanced"'), 1, 0)
chk("notebook: bagging sortea estimator__max_features", '"model__estimator__max_features": [None, "sqrt", 0.7]' in src, 1, 0)
chk("notebook: RF10 ≈ bagging (|Δ| < 1 %)", abs(NA["rf10"]["rmse"] - NA["bag"]["rmse"]) / NA["bag"]["rmse"] < 0.01, 1, 0)
try:
    import inspect
    from sklearn.ensemble import RandomForestRegressor
    chk("sklearn: max_features por defecto del RF de regresion = 1.0", inspect.signature(RandomForestRegressor).parameters["max_features"].default, 1.0, 0)
except Exception as e:  # noqa: BLE001
    mal.append(f"sin sklearn para verificar el valor por defecto: {e}")

print(f"OK {ok} comprobaciones" + ("" if not mal else f" · {len(mal)} FALLAS:"))
for m in mal:
    print("  -", m)
