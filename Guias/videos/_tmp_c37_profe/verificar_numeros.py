"""Comprueba los numeros de la serie C37.
(1) Cada cifra real dicha o mostrada se compara con datos37.json (recalculado con la base real: datos.py) y con las diapositivas.
(2) Cada ejemplo resuelto se recalcula aqui a mano, con los mismos redondeos intermedios que se ven en pantalla.
(3) Cada literal se busca en el guion del video donde se dice (narracion + texto en pantalla).
(4) Las alertas del material se verifican contra la grabacion, las diapositivas y los scripts del profesor.
Ejecutar con cualquier Python 3 (solo usa json/math/re)."""
import json
import math
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
RAIZ = AQUI.parents[2]
D = json.loads((AQUI / "datos37.json").read_text(encoding="utf-8"))
F = D["fig"]
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


def say(n):
    g = next(FUENTE.glob(f"guion_c37_{n:02}_*.json"))
    js = json.loads(g.read_text(encoding="utf-8"))
    txt = " ".join(b["say"] for b in js["beats"])
    txt += " " + json.dumps([b["acciones"] for b in js["beats"]], ensure_ascii=False)
    return re.sub(r"\[\[\w+\]\]", "", txt).replace(" ", " ")


S = {n: say(n) for n in range(1, 13)}


def en(n, *lits):
    global ok
    for l in lits:
        if l in S[n]:
            ok += 1
        else:
            mal.append(f"video {n}: no aparece «{l}»")


def sig(x):
    return 1 / (1 + math.exp(-x))


SC = D["script"]
LG, LI, RA = SC["Logistica"], SC["SVM lineal"], SC["SVM radial"]
# ================================================================ base y particion (diapositivas 4-5)
chk("n", D["n"], 569, 0); chk("benignos", D["conteo"]["0"], 357, 0); chk("malignos", D["conteo"]["1"], 212, 0)
SP = F["split"]
for k, v in dict(train=398, test=171, train_ben=250, train_mal=148, test_ben=107, test_mal=64).items():
    chk(f"split {k}", SP[k], v, 0)
# ================================================================ video 1
chk("v1 base 107/171", D["base_benigno"]["acc"], 107 / 171); chk("v1 62.6", round(100 * 107 / 171, 1), 62.6)
chk("v1 siempre 1", round(100 * 64 / 171, 1), 37.4); chk("v1 148/398", round(100 * 148 / 398, 1), 37.2)
chk("v1 piensa 150/171", round(150 / 171, 3), 0.877)
en(1, "569", "357", "212", "398", "171", "250", "148", "107", "64", "62.6", "37.4", "0.877")
# ================================================================ video 2 (diapositiva 9 y logistica real)
VN, FP, FN, VP = 56, 14, 6, 24
chk("v2 exactitud", (VP + VN) / 100, 0.80); chk("v2 sens", VP / (VP + FN), 0.80); chk("v2 espe", VN / (VN + FP), 0.80)
chk("v2 prec", round(VP / (VP + FP), 3), 0.632); chk("v2 F1", round(2 * VP / (2 * VP + FP + FN), 3), 0.706)
chk("v2 log tn", LG["tn"], 103, 0); chk("v2 log fp", LG["fp"], 4, 0); chk("v2 log fn", LG["fn"], 6, 0); chk("v2 log tp", LG["tp"], 58, 0)
chk("v2 sens 58/64", round(58 / 64, 3), 0.906); chk("v2 prec 58/62", round(58 / 62, 3), 0.935); chk("v2 103/107", round(103 / 107, 3), 0.963)
chk("v2 piensa sens", 80 / 100, 0.80); chk("v2 piensa prec", round(80 / 90, 3), 0.889)
en(2, "0.632", "0.706", "0.906", "0.935", "0.963", "0.889")
# ================================================================ video 3
chk("v3 lineal min", D["lineal01"]["min"], -0.32); chk("v3 lineal max", D["lineal01"]["max"], 1.58)
chk("v3 eta", -1 + 0.8 * 1.5 + 0.4 * 2, 1.0); chk("v3 e^-1", round(math.exp(-1), 3), 0.368); chk("v3 1/1.368", round(1 / 1.368, 3), 0.731)
chk("v3 sig(1)", round(sig(1), 3), 0.731); chk("v3 0->1", round(sig(1) - sig(0), 2), 0.23); chk("v3 4->5", round(sig(5) - sig(4), 2), 0.01)
for p, l in ((0.9, 0.105), (0.5, 0.693), (0.01, 4.605)):
    chk(f"v3 -log {p}", round(-math.log(p), 3), l)
chk("v3 2D logistica", D["log2"]["acc_test"], 0.889)
for a, b in zip(D["primeras_prob"], [0.9997, 0.9949, 0.9, 0.9915, 0.7447, 1.0]):
    chk("v3 primeras prob", a, b)
chk("v3 defecto acc", round(100 * D["logistica_defecto"]["acc"], 1), 93.0); chk("v3 none acc", round(100 * LG["acc"], 1), 94.2)
chk("v3 piensa eta", -2 + 0.9 * 2 + 0.5 * 1, 0.3); chk("v3 e^-0.3", round(math.exp(-0.3), 3), 0.741); chk("v3 1/1.741", round(1 / 1.741, 3), 0.574)
en(3, "0.32", "1.58", "0.368", "0.731", "0.105", "0.693", "4.605", "88.9", "0.9997", "94.2", "93.0", "0.574")
# ================================================================ video 4
UP = F["umbral_prueba"]
for t, v in {"0.3": (94, 13, 5, 59), "0.5": (103, 4, 6, 58), "0.8": (105, 2, 12, 52)}.items():
    for k, x in zip(("tn", "fp", "fn", "tp"), v):
        chk(f"v4 t={t} {k}", UP[t][k], x, 0)
chk("v4 espe 0.3", round(94 / 107, 3), 0.879); chk("v4 sens 0.8 (52/64 = 0.8125)", 52 / 64, 0.8125)
chk("v4 cruce", F["umbral_cv"]["cruce"]["t"], 0.34)
chk("v4 odds 0.8", 0.8 / 0.2, 4); chk("v4 e^0.8", round(math.exp(0.8), 2), 2.23); chk("v4 e", round(math.e, 3), 2.718)
chk("v4 2.718*2.23", round(2.718 * 2.23, 2), 6.06, 0.011); chk("v4 sig(1.8)", round(sig(1.8), 3), 0.858); chk("v4 0.731*2.23", round(0.731 * 2.23, 2), 1.63)
GR = F["glm_como_R"]
for k, v in (("intercepto", -42.2511),):
    chk("v4 glm intercepto", GR["intercepto"], v)
for k, v in dict(radio=1.3399, textura=0.4421, suavidad=124.9794, concavidad=21.4077).items():
    chk(f"v4 glm {k}", GR["coef"][k], v)
chk("v4 e^0.4421", round(math.exp(0.4421), 2), 1.56); chk("v4 e^1.25", round(D["glm_suavidad_001"], 2), 3.49)
chk("v4 cifras e^124.98", len(str(int(math.exp(124.9794)))), 55, 0); chk("v4 p nuevo", GR["p_nuevo"], 0.3919)
chk("v4 radio std", F["logistica_std"]["coef"]["radio"], 4.9247)
for b_, v in (("1", -0.978), ("2", -0.295), ("5", -0.014)):
    chk(f"v4 loglik {b_}", round(D["separacion"][b_], 3), v)
chk("v4 e^-0.7", round(math.exp(-0.7), 3), 0.497)
en(4, "0.879", "0.813", "0.34", "2.23", "6.05", "0.858", "1.63", "124.98", "55 cifras", "3.49", "0.3919", "0.978", "0.497")
# ================================================================ video 5
f = 1 * 2 + (-1) * 1 - 0.5
chk("v5 f", f, 0.5); chk("v5 norma", round(math.sqrt(2), 3), 1.414); chk("v5 dist", round(0.5 / 1.414, 3), 0.354)
DU = D["duro"]
chk("v5 n_sv", DU["n_sv"], 3, 0); chk("v5 norma real", round(math.hypot(1.316, 1.071), 3), 1.697); chk("v5 ancho", round(2 / 1.697, 3), 1.179)
chk("v5 ancho datos", DU["ancho"], 1.179)
fsv = 1.316 * 0.54 + 1.071 * 0.194 + 0.082
chk("v5 f SV", round(fsv, 3), 1.000, 0.0006); chk("v5 terminos", round(1.316 * 0.54, 3) + round(1.071 * 0.194, 3), 0.919)
fnb = 1.316 * -0.678 + 1.071 * -0.545 + 0.082
chk("v5 f no-SV", round(fnb, 3), -1.394); chk("v5 yf datos", D["duro_yf"][3]["yf"], 1.394)
chk("v5 piensa f", 3 * 2 + 4 * 1 - 5, 5); chk("v5 piensa ancho", 2 / 5, 0.4)
en(5, "0.354", "1.414", "1.697", "1.179", "1.394", "0.711", "0.208")
# ================================================================ video 6
for yf, xi in ((1.4, 0), (0.5, 0.5), (-0.3, 1.3)):
    chk(f"v6 xi {yf}", max(0, 1 - yf), xi)
for k, v in (("1.4", 0.220), ("0.5", 0.474), ("-0.3", 0.854)):
    chk(f"v6 log {k}", round(D["perdidas"][k]["log"], 3), v)
chk("v6 0.1*4", 0.1 * 4, 0.4); chk("v6 10*4", 10 * 4, 40)
L2 = D["lin2"]
for c, (sv, an) in {"0.01": (188, 2.24), "1": (83, 0.86), "100": (78, 0.81)}.items():
    chk(f"v6 C={c} sv", L2[c]["n_sv"], sv, 0); chk(f"v6 C={c} ancho", round(2 / math.hypot(*L2[c]["w"]), 2), an); chk(f"v6 C={c} test", L2[c]["acc_test"], 0.883)
HC = F["holgura_clase"]
chk("v6 fuera", HC["fuera"], 308, 0); chk("v6 dentro", HC["dentro"], 70, 0); chk("v6 mal", HC["mal"], 20, 0)
chk("v6 sv 93", sum(D["n_support"]["lineal"]), 93, 0); chk("v6 47", D["n_support"]["lineal"][0], 47, 0); chk("v6 305", 398 - 93, 305, 0)
chk("v6 mejor C", F["C4d"]["mejor_C_cv"], 0.1); chk("v6 mejor acc", round(100 * F["C4d"]["mejor_acc_cv"], 1), 94.0)
chk("v6 piensa", 2 * (0 + 1 + 2.5), 7)
en(6, "0.220", "0.474", "0.854", "2.24", "188", "0.86", "83", "0.81", "78", "88.3", "308", "70", "20", "93", "94.0", "3.5")
# ================================================================ video 7
chk("v7 x2 0.7", 0.7 ** 2, 0.49); chk("v7 1.6^2", 1.6 ** 2, 2.56)
chk("v7 poly", (1 * 3 + 2 * 1 + 1) ** 2, 36)
phi_x = [1, math.sqrt(2) * 1, math.sqrt(2) * 2, 1, 4, math.sqrt(2) * 2]
phi_z = [1, math.sqrt(2) * 3, math.sqrt(2) * 1, 9, 1, math.sqrt(2) * 3]
chk("v7 phi", sum(a * b for a, b in zip(phi_x, phi_z)), 36, 1e-9)
chk("v7 e^-0.4", F["k_dist2"]["g01"], 0.6703); chk("v7 e^-4", F["k_dist2"]["g1"], 0.0183)
chk("v7 e^-0.5", round(math.exp(-0.5), 3), 0.607); chk("v7 e^-5", round(math.exp(-5), 4), 0.0067)
chk("v7 lineal 2D", L2["1"]["acc_test"], 0.883); chk("v7 radial 2D", D["rbf2_prof"]["acc_test"], 0.901)
chk("v7 poly guia", F["kernels2d"]["Polinomial (grado 3)"]["acc_test"], 0.889)
chk("v7 mal e^-0.2", round(math.exp(-0.2), 3), 0.819); chk("v7 e^-0.9", round(math.exp(-0.9), 3), 0.407); chk("v7 e^-9", round(math.exp(-9), 4), 0.0001)
en(7, "36", "0.6703", "0.0183", "0.607", "0.0067", "88.3", "90.1", "88.9", "0.819", "0.407")
# ================================================================ video 8
RB = D["rbf2"]
for g, (tr_, te_, sv) in {"0.01": (92.2, 90.6, 155), "1": (93.2, 90.6, 92), "50": (96.0, 88.9, 279)}.items():
    chk(f"v8 g={g} train", round(100 * RB[g]["acc_train"], 1), tr_); chk(f"v8 g={g} test", round(100 * RB[g]["acc_test"], 1), te_)
    chk(f"v8 g={g} sv", RB[g]["n_sv"], sv, 0)
G4 = F["gamma4d"]
chk("v8 mejor g", G4["mejor_gamma_cv"], 0.0464); chk("v8 mejor acc", round(100 * G4["mejor_acc_cv"], 1), 94.5)
chk("v8 g=0.001", round(100 * G4["acc_cv"][0], 1), 79.2); chk("v8 g=100 cv", round(100 * G4["acc_cv"][-1], 1), 63.3)
chk("v8 g=100 test = base", G4["acc_test"][-1], D["base_benigno"]["acc"])
TB = {(r["C"], r["gamma"]): r for r in D["grid_prof"]["tabla"]}
chk("v8 0.1/0.01", round(TB[(0.1, 0.01)]["media"], 3), 0.794); chk("v8 10/0.1", round(TB[(10.0, 0.1)]["media"], 3), 0.940)
chk("v8 1/1", round(TB[(1.0, 1.0)]["media"], 3), 0.927); chk("v8 10/1", round(TB[(10.0, 1.0)]["media"], 3), 0.915, 0.0011)
en(8, "96.0", "88.9", "93.2", "90.6", "155", "92", "279", "94.5", "79.2", "63.3", "62.6", "79.4", "94.0", "92.7", "91.5")
# ================================================================ video 9
SD = D["sd"]
chk("v9 sd radio", round(SD["radio"], 2), 3.67); chk("v9 sd textura", round(SD["textura"], 2), 4.22)
chk("v9 sd concav", round(SD["concavidad"], 3), 0.080); chk("v9 sd suav", round(SD["suavidad"], 3), 0.014)
chk("v9 ratio", round(SD["textura"] / SD["suavidad"]), 293, 0)
chk("v9 0.26/3.67", round(0.26 / 3.67, 2), 0.07); chk("v9 0.22/0.080", round(0.22 / 0.080, 2), 2.75)
chk("v9 z radio", round((14 - 14.179) / 3.669, 3), -0.049); chk("v9 z suav", round((0.10 - 0.09618) / 0.0144, 3), 0.265)
chk("v9 z datos", D["nuevo"]["z"][2], 0.2652)
E4 = F["escala4d"]
chk("v9 lineal sin", round(100 * E4["Lineal C = 0.1"]["sin"], 1), 86.0); chk("v9 rbf con", round(100 * E4["RBF C = 10, γ = 0.1"]["con"], 1), 97.1)
chk("v9 rbf sin", round(100 * E4["RBF C = 10, γ = 0.1"]["sin"], 1), 90.1)
chk("v9 votos esc", sum(x["y"] for x in D["vecinos"]), 3, 0); chk("v9 votos crudo", sum(x["y"] for x in D["vecinos_sin_escalar"]), 2, 0)
chk("v9 knn p", D["knn_nuevo"]["p"], 0.6); chk("v9 2º vecino concav", D["vecinos_sin_escalar"][1]["x"][3], 0.3003)
OT = D["otros"]
chk("v9 knn acc", round(100 * OT["kNN (k = 5)"]["acc"], 1), 92.4); chk("v9 knn crudo", round(100 * OT["kNN sin escalar"]["acc"], 1), 91.2)
chk("v9 auc", round(OT["kNN (k = 5)"]["auc"], 3), 0.978); chk("v9 auc crudo", round(OT["kNN sin escalar"]["auc"], 3), 0.910)
chk("v9 piensa", round(5.71 / 4.21, 2), 1.36); chk("v9 25-19.29", round(25 - 19.29, 2), 5.71)
en(9, "3.67", "4.22", "293", "0.07", "2.75", "0.049", "0.265", "86.0", "97.1", "90.1", "0.223", "92.4", "91.2", "0.978", "0.910", "1.36")
# ================================================================ video 10 (diapositivas 52-54)
for m, (a, s, e) in ((LG, (0.9415, 0.9062, 0.9626)), (LI, (0.9181, 0.8438, 0.9626)), (RA, (0.9708, 0.9375, 0.9907))):
    chk("v10 acc", m["acc"], a); chk("v10 sens", m["sens"], s); chk("v10 espe", m["espe"], e)
chk("v10 prec radial", RA["prec"], 0.9836); chk("v10 errores", (RA["fn"] + RA["fp"], LG["fn"] + LG["fp"], LI["fn"] + LI["fp"]) == (5, 10, 14), 1, 0)
chk("v10 detectados", (LG["tp"], LI["tp"], RA["tp"]) == (58, 54, 60), 1, 0); chk("v10 1 caso", round(100 / 171, 2), 0.58)
for nom, v in {"LDA": (105, 2, 13, 51), "QDA": (103, 4, 9, 55), "kNN (k = 5)": (103, 4, 9, 55)}.items():
    chk(f"v10 {nom}", tuple(OT[nom][k] for k in ("tn", "fp", "fn", "tp")) == v, 1, 0)
chk("v10 LDA espe", round(OT["LDA"]["espe"], 3), 0.981)
chk("v10 costos", (10 * RA["fn"] + RA["fp"], 10 * LG["fn"] + LG["fp"], 10 * LI["fn"] + LI["fp"]) == (41, 64, 104), 1, 0)
chk("v10 piensa", (10 * 4 + 10, 10 * 8 + 2) == (50, 82), 1, 0)
en(10, "0.9415", "0.9181", "0.9708", "0.9062", "0.8438", "0.9375", "0.9626", "0.9907", "0.9836", "92.4", "91.2", "0.981", "41", "104")
# ================================================================ video 11
GP = D["grid_prof"]
chk("v11 best", (GP["best"]["C"], GP["best"]["gamma"]) == (10.0, 0.1), 1, 0); chk("v11 best score", round(GP["best_score"], 3), 0.940)
chk("v11 sd best", round(TB[(10.0, 0.1)]["sd"], 3), 0.020); chk("v11 2º", round(TB[(1.0, 0.1)]["media"], 3), 0.935)
chk("v11 sd 2º", round(TB[(1.0, 0.1)]["sd"], 3), 0.044); chk("v11 casos", round((0.9396 - 0.9347) * 398), 2, 0)
chk("v11 45", 9 * 5, 45, 0)
RAm = F["rejilla_amplia"]
chk("v11 amplia", RAm["mejor"]["svc__C"], 100, 0); chk("v11 amplia g", RAm["mejor"]["svc__gamma"], 0.01)
chk("v11 amplia cv", round(RAm["acc_cv"], 3), 0.945); chk("v11 amplia test", round(100 * RAm["prueba_mejor"]["acc"], 1), 94.7)
pares = [m > b for m in (0.9, 0.7, 0.4) for b in (0.5, 0.2)]
chk("v11 auc mano", round(sum(pares) / 6, 3), 0.833)
AU = D["script_auc"]
chk("v11 auc log", AU["Logistica"], 0.9746); chk("v11 auc lin", AU["SVM lineal"], 0.9746); chk("v11 auc rad", AU["SVM radial"], 0.9834)
chk("v11 auc clases rad", round(D["auc_con_clases"]["SVM radial"], 3), 0.964); chk("v11 auc clases log", round(D["auc_con_clases"]["Logistica"], 3), 0.934)
PL = F["platt"]
chk("v11 platt disc", PL["discrepancias"], 3, 0); chk("v11 platt acc", round(100 * PL["acc_p05"], 1), 95.3)
chk("v11 piensa", sum(m > b for m in (0.8, 0.6) for b in (0.7, 0.1)) / 4, 0.75)
en(11, "0.940", "0.020", "0.935", "0.044", "0.945", "94.7", "0.833", "0.9746", "0.9834", "0.964", "0.934", "95.3", "0.75")
# ================================================================ video 12
en(12, "569", "62.6", "0.632", "0.731", "3.49", "0.670", "0.018", "97.1", "90.1", "0.9746")
# ================================================================ (4) alertas del material
tr = (RAIZ / "VideosClases" / "Clase 37 - SVM y clasificadores clasicos.txt").read_text(encoding="utf-8")
for frase in ("con la base de 357 vamos a dividir en 70 y 30", "clasificamos todos como uno", "Voy a clasificar 107 personas correctamente",
              "tiene garantizado que tiene una solución única", "el kernel radial solamente tiene cinco errores"):
    if frase in tr:
        ok += 1
    else:
        mal.append(f"grabacion: no aparece «{frase}»")
sl = (AQUI / "slides.txt").read_bytes().decode("cp1252", errors="ignore")
for frase in ("Los scripts de esta clase no calculan AUC", "0.9415", "0.9708", "106 + 60", "p = 0.3919", "K = e-0.4 = 0.6703"):
    if frase in sl:
        ok += 1
    else:
        mal.append(f"diapositivas: no aparece «{frase}»")
pyc = (RAIZ / "Clase 37 SVM y clasificadores clasicos Pedro luiz ramos" / "clasificadores.py").read_text(encoding="utf-8")
rc = (RAIZ / "Clase 37 SVM y clasificadores clasicos Pedro luiz ramos" / "clasificadores.R.txt").read_text(encoding="utf-8")
for frase, txt in (("roc_auc_score(y_prueba, probabilidad)", pyc), ("pROC::auc(y_prueba, probabilidad", rc), ("SVC(kernel=\"rbf\", C=10, gamma=0.1)", pyc)):
    if frase in txt:
        ok += 1
    else:
        mal.append(f"scripts: no aparece «{frase}»")

print(f"comprobaciones OK: {ok}   fallas: {len(mal)}")
for m in mal:
    print("  FALLA", m)
