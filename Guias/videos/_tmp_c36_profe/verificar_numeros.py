"""Comprueba los numeros de la serie C36.
(1) Cada cifra real dicha o mostrada se compara con datos36.json (recalculado con la base real: datos.py).
(2) Cada ejemplo resuelto se recalcula aqui a mano, con los mismos redondeos intermedios que se ven en pantalla.
(3) Cada literal se busca en el guion del video donde se dice (narracion + texto en pantalla).
(4) Las alertas del material se verifican contra las diapositivas, el notebook y las API (scikit-learn, XGBoost, LightGBM).
Ejecutar con el venv de Clase 35 (tiene scikit-learn, xgboost y lightgbm)."""
import inspect
import json
import math
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
RAIZ = AQUI.parents[2]
D = json.loads((AQUI / "datos36.json").read_text(encoding="utf-8"))
F = D["fig"]
CU = D["curvas"]
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c36_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)     # texto en pantalla
    return re.sub(r"\[\[\w+\]\]", "", txt).replace(" ", " ")


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
    v = int(round(x / 100.0)) * 100
    return f"{v:,}".replace(",", " ")


def k1(v):
    return f"{v / 1000:.1f} mil"


def a(eps):
    return 0.5 * math.log((1 - eps) / eps)


# ================================================================ base
chk("Heart n", D["heart_n"], 303, 0); chk("Heart test", D["heart_te"], 76, 0); chk("Heart train", D["heart_tr"], 227, 0)
chk("Ames n", D["ames_n"], 2930, 0)

# ================================================================ video 1
SQ = F["secuencial_rmse"]
for lit, val in (("78 500", SQ[0]), ("68 400", SQ[1]), ("55 400", SQ[2]), ("50 900", SQ[3])):
    chk(f"v1 {lit}", cien(val * 1000) == lit, 1, 0); en(1, lit)
en(1, "78.5", "68.4", "55.4", "50.9")
# ejemplo del apunte: 160 -> 175 -> 181 -> 178 ; eta = 1 -> 220
f = 160
for h, esp in ((50, 175), (20, 181), (-10, 178)):
    f = f + 0.3 * h
    chk(f"v1 etapa {esp}", f, esp, 1e-9)
chk("v1 eta=1", 160 + 50 + 20 - 10, 220, 0); en(1, "160 + 0.3 · 50 = 160 + 15 =", "181 + 0.3 · (−10) = 181 − 3 =")

# ================================================================ video 2
BB = F["bb"]
for lit, val in (("41 200", BB["tree3"]), ("39 300", BB["rf1"]), ("23 100", BB["rf300"]), ("69 000", BB["gb1"]), ("21 800", BB["gb300"])):
    chk(f"v2 {lit}", cien(val) == lit, 1, 0); en(2, lit)
chk("v2 bosque con 100 arboles = 23 100", cien(CU["rf"][99]) == "23 100", 1, 0)
chk("v2 curvas = figura (rf1, gb300)", CU["rf"][0], BB["rf1"], 1e-6); chk("v2 curva gb 300", CU["gb"][-1], BB["gb300"], 1e-6)
chk("v2 mejor GB en 250", BB["gb_best"], 250, 0); chk("v2 min GB", int(min(range(300), key=CU["gb"].__getitem__)) + 1, 250, 0)
chk("v2 GB sube tras el minimo", CU["gb"][-1] > BB["gb_min"], 1, 0); en(2, "mínimo en 250 árboles: 21.786 USD", "con 300: 21.847 USD"); chk("v2 21 786 y 21 847", (round(BB["gb_min"]), round(BB["gb300"])) == (21786, 21847), 1, 0)
chk("v2 primer arbol con eta=1 = arbol suelto", CU["lr1.0"][0], BB["tree3"], 1e-6)

# ================================================================ video 3 (ejemplos a mano)
chk("v3 alfa(0.1) = 1.10", round(a(0.1), 2), 1.10, 1e-9); chk("v3 alfa(0.3) = 0.42", round(a(0.3), 2), 0.42, 1e-9)
chk("v3 ln 4 = 1.386", round(math.log(4), 3), 1.386, 1e-9); chk("v3 alfa(0.2) = 1.386/2 = 0.693", 1.386 / 2, 0.693, 1e-9)
chk("v3 e^0.693 = 2", round(math.exp(0.693), 3), 2.0, 0.001); chk("v3 e^-0.693 = 0.5", round(math.exp(-0.693), 3), 0.5, 0.001)
chk("v3 suma 0.8", 2 * 0.2 + 8 * 0.05, 0.8, 1e-9); chk("v3 0.2/0.8", 0.2 / 0.8, 0.25, 1e-9); chk("v3 0.05/0.8", 0.05 / 0.8, 0.0625, 1e-9)
chk("v3 la mitad", 2 * 0.25, 0.5, 1e-9)
for e_ in (0.05, 0.2, 0.37, 0.6):                        # la mitad del peso queda en los errores, para cualquier epsilon
    ae = a(e_)
    chk(f"v3 mitad con eps={e_}", e_ * math.exp(ae) / (e_ * math.exp(ae) + (1 - e_) * math.exp(-ae)), 0.5, 1e-9)
chk("v3 ronda 2 ponderado", 0.25 + 2 * 0.0625, 0.375, 1e-9); chk("v3 ronda 2 contando", 3 / 10, 0.3, 1e-9)
chk("v3 0.625/0.375 = 1.667", round(0.625 / 0.375, 3), 1.667, 1e-9); chk("v3 ln 1.667 = 0.511", round(math.log(1.667), 3), 0.511, 1e-9)
chk("v3 ½·0.511 ≈ 0.26", round(0.511 / 2, 2), 0.26, 0.0051); chk("v3 alfa exacto ronda 2", round(a(0.375), 2), 0.26, 1e-9)
chk("v3 alfa(0.8) = -0.693", round(a(0.8), 3), -0.693, 1e-9); chk("v3 ln 0.25", round(math.log(0.25), 3), -1.386, 1e-9)
en(3, "½ · ln(0.8/0.2) = ½ · ln 4 = ½ · 1.386 =", "0.25 + 2 × 0.0625 =", "½ · ln(0.625/0.375) = ½ · ln 1.667 = ½ · 0.511 ≈")
chk("v3 API: AdaBoost usa un arbol de profundidad 1", D["api_ada_estimador_base_depth"], 1, 0)

# ================================================================ video 4
A70, AD = D["ada70"], F["ada"]
chk("v4 19 errores", len(A70["rondas"][0]["mal"]), 19, 0); chk("v4 19/70", round(19 / 70, 3), 0.271, 1e-9)
chk("v4 eps1 = figura", round(A70["rondas"][0]["err"], 3), AD["err"][0], 1e-9)
chk("v4 51/19 = 2.684", round(51 / 19, 3), 2.684, 1e-9); chk("v4 ln 2.684 = 0.987", round(math.log(2.684), 3), 0.987, 1e-9)
chk("v4 0.987/2 ≈ 0.49", round(0.987 / 2, 2), 0.49, 1e-9); chk("v4 alfa1 = figura", round(A70["rondas"][0]["alpha"], 2), AD["alpha"][0], 1e-9)
chk("v4 21 errores en ronda 2", len(A70["rondas"][1]["mal"]), 21, 0); chk("v4 eps2 = 0.338", round(A70["rondas"][1]["err"], 3), 0.338, 1e-9)
chk("v4 eps3 = 0.453", round(A70["rondas"][2]["err"], 3), 0.453, 1e-9)
for k, (e_, al) in enumerate(zip(AD["err"], AD["alpha"])):
    en(4, f"ε = {e_:.3f}", f"α = {al:.2f}")
for k, thr in enumerate((54.5, 147.5, 177.5)):
    chk(f"v4 corte ronda {k + 1}", A70["rondas"][k]["thr"], thr, 1e-9)
en(4, "edad < 54.5", "frec. < 147.5", "52 de 70")
chk("v4 52 de 70", A70["aciertos30"], 52, 0); chk("v4 52/70 = 0.74", round(52 / 70, 2), 0.74, 1e-9); chk("v4 acc30 figura", round(AD["acc30"], 2), 0.74, 1e-9)
chk("v4 voto apunte", -0.9 + 0.4 + 0.3, -0.2, 1e-9); chk("v4 voto cuaderno", 0.8 - 0.5 - 0.45 + 0.2, 0.05, 1e-9)
chk("v4 e^0.49 = 1.63", round(math.exp(0.49), 2), 1.63, 1e-9); chk("v4 1.63^5 = 11.5", round(1.63 ** 5, 1), 11.5, 1e-9)
chk("v4 API: n_estimators por defecto 50", D["api_ada_n_estimators_defecto"], 50, 0)

# ================================================================ video 5 (diapositiva 10 + stump de ejemplo)
YV = [12, 7, 11, 10]
F0 = sum(YV) / 4; chk("v5 F0", F0, 10, 0)
R1 = [y - F0 for y in YV]; chk("v5 residuos", R1 == [2, -3, 1, 0], 1, 0)
h = [(2 + 1 + 0) / 3, -3, (2 + 1 + 0) / 3, (2 + 1 + 0) / 3]; chk("v5 hoja ACD", h[0], 1, 1e-12)
F1 = [F0 + 0.1 * x for x in h]; R2 = [y - f for y, f in zip(YV, F1)]
for got, esp in zip(F1, [10.1, 9.7, 10.1, 10.1]):
    chk("v5 F1", got, esp, 1e-9)
for got, esp in zip(R2, [1.9, -2.7, 0.9, -0.1]):
    chk("v5 r2", got, esp, 1e-9)
chk("v5 MSE antes", sum(r * r for r in R1) / 4, 3.5, 1e-9)
chk("v5 suma cuadrados", 3.61 + 7.29 + 0.81 + 0.01, 11.72, 1e-9); chk("v5 MSE despues", round(11.72 / 4, 2), 2.93, 1e-9)
chk("v5 MSE despues exacto", round(sum(r * r for r in R2) / 4, 2), 2.93, 1e-9)
R2b = [y - (F0 + x) for y, x in zip(YV, h)]; chk("v5 eta=1 residuos", [round(r, 9) for r in R2b] == [1, 0, 0, -1], 1, 0)
chk("v5 eta=1 MSE", sum(r * r for r in R2b) / 4, 0.5, 1e-9)
for m, lit in ((1, "69.0 mil"), (10, "40.7 mil"), (50, "24.3 mil"), (100, "22.7 mil"), (300, "21.8 mil")):
    chk(f"v5 GB {m} arboles", k1(CU["gb"][m - 1]) == lit, 1, 0); en(5, lit)
en(5, "69 000", "40 700", "24 300", "22 700", "21 800")

# ================================================================ video 6
chk("v6 cuadratica +3", 12 - 9, 3, 0); chk("v6 huber recorta", max(-2, min(2, 3)), 2, 0)
chk("v6 atipico +31", 40 - 9, 31, 0)
chk("v6 p(0)", 1 / (1 + math.exp(0)), 0.5, 1e-12); chk("v6 p(2) = 0.881", round(1 / (1 + math.exp(-2)), 3), 0.881, 1e-9)
chk("v6 1 - 0.881", 1 - 0.881, 0.119, 1e-9)
GBH, GBC = D["nb_ames"]["Gradient Boosting"]["rmse_test"], D["nb_ames_gb_cuadratica_rmse"]
en(6, "29 200", "28 600", "29.200", "28.600"); chk("v6 huber 29 200", cien(GBH) == "29 200", 1, 0); chk("v6 cuadratica 28 600", cien(GBC) == "28 600", 1, 0)
chk("v6 ~500 USD mejor", GBH - GBC, 535, 40)
try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    chk("v6 API: GBClassifier loss por defecto log_loss", inspect.signature(GradientBoostingClassifier).parameters["loss"].default == "log_loss", 1, 0)
    chk("v6 API: GBRegressor loss por defecto squared_error", inspect.signature(GradientBoostingRegressor).parameters["loss"].default == "squared_error", 1, 0)
except Exception as e:  # noqa: BLE001
    mal.append(f"sin sklearn: {e}")

# ================================================================ video 7
chk("v7 0.5 serie", [10 * 0.5 ** m for m in (1, 2, 3, 4)] == [5, 2.5, 1.25, 0.625], 1, 0)
for got, esp in zip([10 * 0.9 ** m for m in (1, 2, 3)], [9, 8.1, 7.29]):
    chk("v7 0.9 serie", got, esp, 1e-9)
chk("v7 0.5^4", 0.5 ** 4, 0.0625, 1e-12); chk("v7 0.9^21", round(0.9 ** 21, 3), 0.109, 1e-9); chk("v7 0.9^22", round(0.9 ** 22, 3), 0.098, 1e-9)
chk("v7 22 arboles", min(m for m in range(1, 100) if 0.9 ** m < 0.1), 22, 0); chk("v7 4 arboles", min(m for m in range(1, 100) if 0.5 ** m < 0.1), 4, 0)
chk("v7 x5.5", 22 / 4, 5.5, 0)
LR = {d["lr"]: d for d in F["lr"]}
for lr, n_, best in ((1.0, 9, "30.3 mil"), (0.3, 462, "22.5 mil"), (0.1, 587, "21.5 mil"), (0.03, 792, "21.5 mil")):
    chk(f"v7 eta {lr} n", LR[lr]["n"], n_, 0); chk(f"v7 eta {lr} mejor", k1(LR[lr]["best"]) == best, 1, 0)
    chk(f"v7 eta {lr} curva", int(min(range(800), key=CU[f"lr{lr}"].__getitem__)) + 1, n_, 0)
en(7, "30 300", "22 500", "21 500", "31.8 mil", "21 512", "21 462", "22.7 mil", "25.9 mil")
chk("v7 eta 1 con 800", k1(LR[1.0]["last"]) == "31.8 mil", 1, 0)
chk("v7 100 arboles eta 0.1", k1(CU["lr0.1"][99]) == "22.7 mil", 1, 0); chk("v7 100 arboles eta 0.03", k1(CU["lr0.03"][99]) == "25.9 mil", 1, 0)
chk("v7 50 USD", LR[0.1]["best"] - LR[0.03]["best"], 50, 0); chk("v7 35 % mas arboles", round(100 * (792 / 587 - 1)), 35, 0)

# ================================================================ video 8
CA = F["cmp_a"]
chk("v8 ~6 veces (XGB)", CA["Gradient Boosting"][2] / CA["XGBoost"][2], 6.5, 0.6); chk("v8 ~6 veces (LGBM)", CA["Gradient Boosting"][2] / CA["LightGBM"][2], 6.2, 0.6)
en(8, f"{CA['Gradient Boosting'][2]:.2f} s", f"{CA['XGBoost'][2]:.2f} s", f"{CA['LightGBM'][2]:.2f} s")
chk("v8 lambda 0", 12 / 3, 4, 0); chk("v8 lambda 1", 12 / 4, 3, 0); chk("v8 lambda 3", 12 / 6, 2, 0)
chk("v8 2^4", 2 ** 4, 16, 0); chk("v8 2^3", 2 ** 3, 8, 0); chk("v8 31 ≈ 4 × 8", round(31 / 8), 4, 0); chk("v8 2^5 = 32", 2 ** 5, 32, 0)
chk("v8 LGBM train 8 000", cien(CA["LightGBM"][1]) == "8 000", 1, 0); chk("v8 LGBM test 27 000", cien(CA["LightGBM"][0]) == "27 000", 1, 0)
chk("v8 mas de 3 veces", CA["LightGBM"][0] / CA["LightGBM"][1] > 3, 1, 0); en(8, "8 000", "27 000")
try:
    import pandas as pd
    import lightgbm as lgb
    ames = pd.read_csv(next(RAIZ.glob("Clase*35*")) / "ames_housing.csv")
    nu = int(ames.Gr_Liv_Area.nunique())
    nb = lgb.Dataset(ames[["Gr_Liv_Area"]].values.astype(float), ames.Sale_Price.values, params={"verbosity": -1}).construct().feature_num_bin(0)
    chk("v8 1 292 valores distintos", nu, 1292, 0); chk("v8 255 bins", nb, 255, 0); chk("v8 ≈5 veces menos", round(nu / nb), 5, 0)
    en(8, "1 292", "1.292", "255 bins")
except Exception as e:  # noqa: BLE001
    mal.append(f"sin lightgbm para bins: {e}")
chk("v8 subsample sin freq: identico", D["api_lgbm_subsample_sin_freq_identico"], 1, 0)
chk("v8 subsample con freq: cambia", D["api_lgbm_subsample_con_freq_cambia"], 1, 0); chk("v8 subsample_freq por defecto 0", D["api_lgbm_subsample_freq_defecto"], 0, 0)

# ================================================================ video 9
ES = F["es"]
chk("v9 mejor 754", ES["best"], 754, 0); chk("v9 parada 805", ES["parada"], 805, 0); chk("v9 754 + 50 + 1 = 805", ES["best"] + 50 + 1, 805, 0)
chk("v9 val 21 000", cien(ES["val"]) == "21 000", 1, 0); chk("v9 train 9 400", cien(ES["train"]) == "9 400", 1, 0); en(9, "21 000", "9 400", "754", "805")
chk("v9 ahorro ~700", 1500 - 805, 695, 0)
SER = [30, 25, 22, 21, 21.4, 21.9, 22.5]
best, sin, par = SER[0], 0, None
for k, x in enumerate(SER[1:], 2):
    if x < best:
        best, sin = x, 0
    else:
        sin += 1
    if sin == 2:
        par = k
        break
chk("v9 apunte para tras la ronda 6", par, 6, 0); chk("v9 mejor ronda 4", SER.index(21) + 1, 4, 0)
SER2 = [30, 25, 22, 21, 21.4, 20.9]; chk("v9 piensa: 20.9 es nuevo minimo", SER2[5] < min(SER2[:5]), 1, 0)
NH, NHT, NA, NAT = D["nb_heart"], D["nb_heart_tuned"], D["nb_ames"], D["nb_ames_tuned"]
for lit, val in (("0.930", NH["AdaBoost"]["auc"]), ("0.929", NHT["ada"]["auc"]), ("0.937", NH["XGBoost"]["auc"]), ("0.929", NHT["xgb"]["auc"])):
    chk(f"v9 {lit}", r3(val) == lit, 1, 0); en(9, lit)
for lit, val in (("29.2 mil", NA["Gradient Boosting"]["rmse_test"]), ("23.9 mil", NAT["gb"]["rmse_test"]), ("25.8 mil", NA["LightGBM"]["rmse_test"]),
                 ("25.9 mil", NAT["lgbm"]["rmse_test"])):
    chk(f"v9 {lit}", k1(val) == lit, 1, 0); en(9, lit)
chk("v9 1 728 combinaciones", 4 * 4 * 4 * 3 * 3 * 3, 1728, 0); chk("v9 grilla 2x2", 2 * 2, 4, 0); en(9, "1 728", "1.728")
chk("v9 API: n_iter_no_change por defecto None", D["api_gb_n_iter_no_change"] is None, 1, 0)
chk("v9 API: GB entrena todos los arboles", D["api_gb_arboles_entrenados"], 300, 0)

# ================================================================ video 10
CH = F["cmp_h"]
for m in ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "Random Forest"]:
    en(10, r3(CH[m][0]))
    chk(f"v10 desvio {m} entre 0.035 y 0.044", 0.0345 <= CH[m][1] < 0.0445, 1, 0)
chk("v10 diferencia 1.º - último < desvio", CH["AdaBoost"][0] - CH["Gradient Boosting"][0] < min(v[1] for v in CH.values()), 1, 0)
en(10, r3(CH["AdaBoost"][0] - CH["Gradient Boosting"][0]), "de 0.035 a 0.044")
for lit, m in (("24 400", "XGBoost"), ("26 800", "Gradient Boosting"), ("27 000", "LightGBM"), ("27 500", "Random Forest"), ("36 200", "AdaBoost")):
    chk(f"v10 {lit}", cien(CA[m][0]) == lit, 1, 0); en(10, lit)
for m, ac, auc in (("XGBoost", 65, "0.937"), ("LightGBM", 65, "0.932"), ("AdaBoost", 64, "0.930"), ("Gradient Boosting", 64, "0.928")):
    chk(f"v10 nb aciertos {m}", NH[m]["aciertos"], ac, 0); chk(f"v10 nb AUC {m}", r3(NH[m]["auc"]) == auc, 1, 0)
    en(10, f"{ac} de 76 · AUC {auc}")
chk("v10 1/76 = 1.3 puntos", round(100 / 76, 1), 1.3, 1e-9)
chk("v10 diferencia 2 400", cien(CA["Gradient Boosting"][0] - CA["XGBoost"][0]) == "2 400", 1, 0)
chk("v10 ~9 %", round(100 * (CA["Gradient Boosting"][0] - CA["XGBoost"][0]) / CA["Gradient Boosting"][0]), 9, 0)
chk("v10 API: importancia LightGBM = split", D["api_lgbm_importance_defecto"] == "split", 1, 0)
chk("v10 API: importancia XGBoost = gain", D["api_xgb_importance_es_gain"], 1, 0)

# ================================================================ video 11 (repaso)
en(11, "23 100", "21 800", "30 300", "21 500", "805", "21 000", "24 400", "36 200", "0.693")

# ================================================================ (4) alertas del material: diapositivas, notebook y API
sl = (AQUI / "slides.txt").read_text(encoding="utf-8")
bloque16 = sl.split("=== slide 16")[1].split("=== slide 17")[0]
chk("diapositiva 16: titulo dice cinco", "Comparación de los cinco modelos" in bloque16, 1, 0)
chk("diapositiva 16: la tabla trae 4 modelos", sum(1 for m in ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
                                                    if re.search(rf"^{m}$", bloque16, re.M)), 4, 0)
chk("diapositiva 16: falta CatBoost", "CatBoost" not in bloque16, 1, 0); chk("diapositiva 18: CatBoost", "CatBoost" in sl.split("=== slide 18")[1], 1, 0)
nums = re.findall(r"^(\d+)/20$", sl, re.M)
chk("numeracion: salta 13 -> 15", "14" not in nums and "13" in nums and "15" in nums, 1, 0); chk("19 laminas", sl.count("=== slide "), 19, 0)
nb = json.loads(next(RAIZ.glob("Clase*36*/Ejemplo Boosting.ipynb")).read_text(encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
lg = re.findall(r"LGBM(?:Classifier|Regressor)\((.*?)\)", src, re.S)
chk("notebook: LGBM con subsample=0.8", sum("subsample=0.8" in x for x in lg), 2, 0)
chk("notebook: ningun subsample_freq", "subsample_freq" not in src, 1, 0)
chk("notebook: GB de Ames con loss='huber'", "loss='huber'" in src, 1, 0)
chk("notebook: optimiza con RMSE", "scoring='neg_root_mean_squared_error'" in src, 1, 0)
chk("notebook: importancia de XGBoost y de LightGBM", "fitted_classifiers['XGBoost']" in src and "fitted_regressors['LightGBM']" in src, 1, 0)
chk("notebook: FAST_MODE = True", "FAST_MODE = True" in src, 1, 0)

print(f"OK {ok} comprobaciones" + ("" if not mal else f" · {len(mal)} FALLAS:"))
for m in mal:
    print("  -", m)
