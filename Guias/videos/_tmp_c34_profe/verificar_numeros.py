"""Comprueba los numeros de la serie C34: (1) cifras fijas escritas en narracion/pantalla contra datos34.json (recalculado
con la base real), (2) cada ejemplo resuelto recalculado aqui a mano, con los mismos redondeos que se ven en pantalla,
(3) las cifras que el material afirma y que los videos marcan como error."""
import json
import math
from pathlib import Path

D = json.loads((Path(__file__).resolve().parent / "datos34.json").read_text(encoding="utf-8"))
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


g = lambda f, n: 1 - (f / n) ** 2 - (1 - f / n) ** 2  # noqa: E731
H = lambda p: 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))  # noqa: E731
ND = {n["id"]: n for n in D["arbol3_nodos"]}

# ---- (1) base y arbol (videos 1, 2)
chk("n", D["n"], 3150, 0); chk("fugas", D["n_fuga"], 495, 0); chk("tasa", D["tasa"], 0.157, 0.0006); chk("12 var", D["p"], 12, 0)
chk("hojas", D["arbol3_hojas"], 7, 0); chk("suma hojas", D["hojas3_suma_n"], 3150, 0)
for i, n, f in ((11, 149, 140), (10, 67, 45), (4, 340, 147), (12, 62, 25), (6, 69, 24), (3, 104, 9), (7, 2359, 105), (1, 2872, 285), (8, 278, 210), (9, 216, 185)):
    chk(f"nodo {i} n", ND[i]["n"], n, 0); chk(f"nodo {i} fugas", ND[i]["fugas"], f, 0)
chk("umbral segundos", ND[1]["umbral"], 934, 0.01); chk("umbral frec", ND[8]["umbral"], 57.5, 0.01); chk("umbral clv", ND[5]["umbral"], 61.35, 0.01)
chk("umbral antig", ND[2]["umbral"], 28.5, 0.01); chk("umbral fallas", ND[9]["umbral"], 0.5, 0.01)
chk("91 % base", 2872 / 3150 * 100, 91.2, 0.05); chk("tasa 94.0", 140 / 149 * 100, 94.0, 0.05); chk("tasa 67.2", 45 / 67 * 100, 67.2, 0.05)
chk("tasa 85.6", 185 / 216 * 100, 85.6, 0.05); chk("tasa 34.8", 24 / 69 * 100, 34.8, 0.05); chk("pct 69", 69 / 3150 * 100, 2.2, 0.05)
chk("tasa 8.7", 9 / 104 * 100, 8.7, 0.05)
# ejemplo: cliente que reclamo, 40 llamadas, 0 fallas -> hoja 10 (40 <= 57.5, 0 <= 0.5); lift
chk("cliente -> hoja 10", (40 <= 57.5) and (0 <= 0.5), 1, 0); chk("lift hoja 10", 0.672 / 0.157, 4.3, 0.05)
# ejemplo: lift hoja 149 y campana
chk("lift 149 (pantalla 0.940/0.157)", 0.940 / 0.157, 5.98, 0.01); chk("contacta %", 149 / 3150 * 100, 4.7, 0.05); chk("captura %", 140 / 495 * 100, 28.3, 0.05)
chk("1 cliente en hoja de 60", 100 / 60, 1.7, 0.05)
# piensa: no reclamo, 1200 s, clv 50 -> hoja 6
chk("piensa video 2 -> hoja 6", (1200 > 934) and (50 <= 61.35), 1, 0)

# ---- (2) video 3: impureza
chk("p_queda", 2655 / 3150, 0.8429); chk("p_fuga", 495 / 3150, 0.1571); chk("0.8429^2", 0.842857 ** 2, 0.7104); chk("0.1571^2", 0.157143 ** 2, 0.0247)
chk("1-0.7104-0.0247 (pantalla)", 1 - 0.7104 - 0.0247, 0.2649, 0.00005); chk("Gini raiz", D["gini_raiz"], 0.2649, 0.00005)
chk("log2 0.8429", math.log2(2655 / 3150), -0.2466); chk("log2 0.1571", math.log2(495 / 3150), -2.670)
chk("0.8429*0.2466", 0.8429 * 0.2466, 0.2079); chk("0.1571*2.670", 0.1571 * 2.670, 0.4195); chk("suma entropia (pantalla)", 0.2079 + 0.4195, 0.6274, 0.00005)
chk("entropia raiz", D["ent_raiz"], 0.6274, 0.00005); chk("Gini 80/20 (frascos)", g(2, 10), 0.32)
for k, (a, b) in {"100/0": (0, 0), "84/16": (0.269, 0.634), "50/50": (0.5, 1.0), "6/94": (0.113, 0.327)}.items():
    chk(f"tabla {k} gini", D["tabla_imp"][k][0], a); chk(f"tabla {k} ent", D["tabla_imp"][k][1], b)
chk("Gini 15/5", g(15, 20), 0.375); chk("0.75^2+0.25^2", 0.5625 + 0.0625, 0.625); chk("H 15/5", H(0.75), 0.811)
chk("0.75*0.415+0.25*2", 0.75 * 0.415 + 0.5, 0.811, 0.0006); chk("Gini nodo reclamo", g(210, 278), 0.370)

# ---- (2) video 4: el primer corte a mano
R, E = D["reclamos"], D["edad40"]
chk("reclamo si n/f", R["n_si"] * 1000 + R["f_si"], 278210, 0); chk("reclamo no n/f", R["n_no"] * 1000 + R["f_no"], 2872285, 0)
chk("p si", 210 / 278, 0.755); chk("p no", 285 / 2872, 0.0992, 0.00005)
chk("1-0.755^2-0.245^2 (pantalla)", 1 - 0.755 ** 2 - 0.245 ** 2, 0.370); chk("1-0.0992^2-0.9008^2 (pantalla)", 1 - 0.0992 ** 2 - 0.9008 ** 2, 0.179)
chk("w si", 278 / 3150, 0.0883, 0.00005); chk("w no", 2872 / 3150, 0.9117, 0.00005)
chk("0.0883*0.370", 0.0883 * 0.370, 0.0327, 0.00005); chk("0.9117*0.179", 0.9117 * 0.179, 0.1632, 0.00005)
chk("0.0327+0.1632 -> 0.196", 0.0327 + 0.1632, 0.196); chk("gp real", R["gp"], 0.196); chk("ganancia 0.265-0.196", 0.265 - 0.196, 0.069, 1e-9)
chk("ganancia real", R["gan"], 0.0693, 0.00005); chk("ganancia entropia", R["gan_e"], 0.131)
chk("edad>40 n/f", E["n_si"] * 1000 + E["f_si"], 970147, 0); chk("edad<=40 n/f", E["n_no"] * 1000 + E["f_no"], 2180348, 0)
chk("tasa 15.2", 147 / 970 * 100, 15.2, 0.05); chk("tasa 16.0", 348 / 2180 * 100, 16.0, 0.05)
chk("Gini 0.257", g(147, 970), 0.257); chk("Gini 0.268", g(348, 2180), 0.268)
chk("0.3079*0.257+0.6921*0.268 (pantalla)", 0.3079 * 0.257 + 0.6921 * 0.268, 0.265); chk("ganancia edad ~0", D["gini_raiz"] - E["gp"], 0.0, 0.0001)
chk("promedio sin ponderar", (g(210, 278) + g(285, 2872)) / 2, 0.274)
chk("diap13 con sus numeros: ponderado", D["diap13"]["gp"], 0.195); chk("diap13 con sus numeros: ganancia", D["diap13"]["gan"], 0.070)
chk("apunte: Gini(16/40)", g(16, 40), 0.48); chk("apunte: Gini(4/60)", g(4, 60), 0.1244, 0.00005)
chk("apunte: ponderado", 0.4 * 0.48 + 0.6 * 0.1244, 0.267); chk("apunte: ganancia", 0.32 - 0.267, 0.053, 1e-9)

# ---- (2) video 5: barrido
B = {b["var"]: b for b in D["barrido"]}
chk("combinaciones", D["combos"], 378, 0); chk("estado", B["estado"]["gan"], 0.0270); chk("edad", B["edad"]["gan"], 0.0005, 0.00005)
chk("fallas raiz", B["numero_fallas"]["gan"], 0.0011, 0.00005); chk("ratio 0.0693/0.0270", 0.0693 / 0.0270, 2.6, 0.05)
chk("umbrales 22,25,31,40", (22 + 25) / 2 + (25 + 31) / 2 + (31 + 40) / 2, 23.5 + 28 + 35.5, 0)
N2R, N2N = D["nivel2_reclamo"][0], D["nivel2_sin_reclamo"][0]
chk("nivel2 reclamo frec", N2R[0][2], 0.0712); chk("nivel2 no reclamo seg", N2N[0][2], 0.0232); chk("empate", N2N[0][2] - N2N[1][2], 0.0006, 0.00005)
chk("Gini nodo 9", g(185, 216), 0.246); chk("Gini 67", g(45, 67), 0.441); chk("Gini 149", g(140, 149), 0.114)
gp9 = 67 / 216 * g(45, 67) + 149 / 216 * g(140, 149)
chk("ponderado nodo 9", gp9, 0.215); chk("ganancia fallas nivel 3", g(185, 216) - gp9, 0.031); chk("28 veces", (g(185, 216) - gp9) / 0.0011, 28, 0.5)
chk("gini cv", D["crit"]["gini"], 0.9058, 0.00005); chk("entropy cv", D["crit"]["entropy"], 0.9202, 0.00005)

# ---- (2) video 6: validacion cruzada y curva
chk("bloque", D["fold_n"], 630, 0); chk("entrena", D["fold_train"], 2520, 0)
chk("suma rondas", sum(D["rondas"]), 4.5290, 0.00005); chk("promedio", sum(D["rondas"]) / 5, 0.9058, 0.00005); chk("desvio", D["rondas_std"], 0.0146, 0.00005)
chk("fugados por bloque", round(D["fold_tasas"][0] * 630), 99, 0)
C = {c["d"]: c for c in D["curva"]}
chk("mejor prof 6", C[6]["va"], 0.9114, 0.00005); chk("sin limite val", C[0]["va"], 0.8267, 0.00005); chk("sin limite train", C[0]["tr"], 1.0, 0)
chk("224 hojas", C[0]["hojas"], 224, 0); chk("21 niveles", C[0]["prof_real"], 21, 0); chk("perdida", (C[6]["va"] - C[0]["va"]) * 100, 8.5, 0.05)
chk("brecha 3", (C[3]["tr"] - C[3]["va"]) * 100, 0.6, 0.05); chk("brecha 6", (C[6]["tr"] - C[6]["va"]) * 100, 2.9, 0.05)
chk("brecha 5", (C[5]["tr"] - C[5]["va"]) * 100, 2.2, 0.05); chk("brecha sin limite", (C[0]["tr"] - C[0]["va"]) * 100, 17.3, 0.05)

# ---- (2) video 7: poda
P_ = {p["alpha"]: p for p in D["poda"]}
for a, h, u in ((0.0, 224, 0.827), (0.0005, 116, 0.851), (0.001, 42, 0.900), (0.002, 15, 0.883), (0.005, 5, 0.835), (0.01, 3, 0.805)):
    chk(f"poda {a} hojas", P_[a]["hojas"], h, 0); chk(f"poda {a} auc", P_[a]["auc"], u)
chk("110 alfas", D["n_alphas"], 110, 0); chk("22 candidatos", D["grid_n_cand"], 22, 0); chk("alfa grid", D["grid_alpha"], 0.0013, 0.00005)
chk("hojas grid", D["grid_hojas"], 36, 0); chk("auc grid", D["grid_auc"], 0.909); chk("diap19 su codigo auc", D["diap19_cv5"]["auc"], 0.881)
chk("diap19 su codigo hojas", D["diap19_cv5"]["hojas"], 36, 0)
for a, t1, t2 in ((0.001, 0.120, 0.135), (0.005, 0.200, 0.155), (0.003, 0.160, 0.145)):
    chk(f"apunte T1 a={a}", 0.10 + 20 * a, t1, 1e-9); chk(f"apunte T2 a={a}", 0.13 + 5 * a, t2, 1e-9)
chk("empate alfa", 0.03 / 15, 0.002, 1e-12)

# ---- (2) video 8: importancia
I = {r["var"]: r for r in D["imp"]}
chk("clv imp", I["clv"]["imp"], 0.109); chk("clv perm", I["clv"]["perm"], 0.0140, 0.00005); chk("estado imp", I["estado"]["imp"], 0.045, 0.0006)
chk("estado perm", I["estado"]["perm"], 0.0185, 0.00005); chk("antig imp", I["antiguedad"]["imp"], 0.113); chk("clv unicos", I["clv"]["u"], 2654, 0)
chk("ratio imp", I["clv"]["imp"] / I["estado"]["imp"], 2.4, 0.05); chk("ratio perm", I["estado"]["perm"] / I["clv"]["perm"], 1.3, 0.05)
rk_i = [r["var"] for r in D["imp"]]; rk_p = [r["var"] for r in sorted(D["imp"], key=lambda r: -r["perm"])]
chk("clv 5 -> 7", rk_i.index("clv") * 10 + rk_p.index("clv"), 46, 0); chk("estado 10 -> 6", rk_i.index("estado") * 10 + rk_p.index("estado"), 95, 0)
chk("fallas 8 -> 5", rk_i.index("numero_fallas") * 10 + rk_p.index("numero_fallas"), 74, 0)
chk("test 788", D["n_test"], 788, 0); chk("apunte perm", 0.900 - 0.865, 0.035, 1e-9)

# ---- (2) video 9: limites
chk("arbol identico", D["mono"]["arbol_dif"], 0.0, 0); chk("hojas iguales", D["mono_hojas_iguales"], 1, 0)
chk("log 0.9199", D["mono"]["log"][0], 0.9199, 0.00005); chk("log 0.9231", D["mono"]["log"][1], 0.9231, 0.00005)
chk("log(1+934)", math.log1p(934), 6.841); chk("sqrt 61.35", math.sqrt(61.35), 7.833)
for r, (izq, u) in zip(D["inest"], (("clv", 73.5), ("segundos", 629.0), ("clv", 73.5), ("frec_llamadas", 13.5), ("segundos", 935.5), ("segundos", 574.0))):
    chk(f"inest sem {r['sem']}", (r["izq"] == izq) and (r["raiz"] == "reclamos") and abs(r["umbral"] - u) < 0.05, 1, 0)
chk("reclamos sola", D["auc_reclamos_sola"], 0.699); chk("sin reclamos", D["sin_reclamos"]["arbol5"], 0.864)

# ---- (2) video 10: bosque
chk("oob", (1 - 1 / 3150) ** 3150, 0.368); chk("sqrt p", D["sqrt_p"], 3, 0)
chk("var 0.3,10", 0.3 + 0.7 / 10, 0.37, 1e-9); chk("var 0.3,100", 0.3 + 0.7 / 100, 0.307, 1e-9); chk("var 0,100", 1 / 100, 0.01, 1e-12)
for (k, a), b in zip(D["rf_n"], (0.797, 0.931, 0.968, 0.973, 0.974)):
    chk(f"rf {k}", a, b)
chk("solo bootstrap", D["rf_cfg"]["solo_bootstrap"], 0.969); chk("bootstrap+sqrt", D["rf_cfg"]["bootstrap_sqrt"], 0.974)
chk("sin bootstrap", D["rf_cfg"]["sin_bootstrap_sqrt"], 0.968)
for k, b in (("arbol3", 0.850), ("arbol5", 0.906), ("podado", 0.909), ("logistica", 0.920), ("rf300", 0.974)):
    chk(f"final {k}", D["final"][k], b)
chk("bosque - mejor arbol", (D["final"]["rf300"] - C[6]["va"]) * 100, 6.2, 0.05)

print(f"OK {ok} comprobaciones" + ("" if not mal else f" · FALLAN {len(mal)}:"))
for m in mal:
    print("  ", m)
