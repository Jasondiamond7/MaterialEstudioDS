"""Comprueba los numeros de la serie C33: (1) cifras escritas a mano en los generadores contra datos33.json,
(2) cada ejemplo resuelto recalculado aqui, (3) cifras que solo existen en el notebook impreso (salidas de celdas)."""
import json
import math
from pathlib import Path

D = json.loads((Path(__file__).resolve().parent / "datos33.json").read_text(encoding="utf-8"))
ok, mal = 0, []


def chk(nombre, a, b, tol=0.0005):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        mal.append(f"{nombre}: {a} != {b}")


# ---- (1) cifras de la base real usadas como texto fijo en guiones/narracion
chk("filas", D["n"], 3150, 0); chk("fugas", D["n_fuga"], 495, 0); chk("tasa %", D["tasa"], 15.71, 0.005)
chk("exactitud 'nadie se va'", D["acc_nadie"], 84.3, 0.05); chk("imposibles", D["imposibles"], 43, 0)
chk("seg<0", D["neg"]["segundos"][0], 18, 0); chk("llam<0", D["neg"]["frec_llamadas"][0], 12, 0); chk("antig<0", D["neg"]["antiguedad"][0], 9, 0)
chk("sms<0", D["neg"]["frec_sms"][0], 1, 0); chk("min seg", D["minmax"]["segundos"][0], -14645, 0); chk("min llam", D["minmax"]["frec_llamadas"][0], -176, 0)
chk("min antig", D["minmax"]["antiguedad"][0], -40, 0); chk("min sms", D["minmax"]["frec_sms"][0], -9, 0)
chk("edad neg", D["edad"]["neg"], 1, 0); chk("edad >100", D["edad"]["mas100"], 2, 0); chk("edad 0-14", D["edad"]["menores"], 108, 0)
chk("edad max", D["edad"]["max"], 233, 0); chk("pct menores", D["pct_menores"], 3.4, 0.05)
chk("nulos", D["nulos_tot"], 83, 0); chk("nulos seg", D["nulos"]["segundos"], 40, 0); chk("duplicados", D["dup"], 19, 0)
chk("seg marcados", D["nan_tras_marcar"]["segundos"], 58, 0); chk("mediana seg", D["medianas_imp"]["segundos"], 3031.5, 0)
chk("mediana antig", D["medianas_imp"]["antiguedad"], 35, 0)
chk("candidatas", D["p"], 50, 0); chk("suma familias", sum(D["familias"].values()), 50, 0); chk("infinitos", D["inf"], 0, 0)
chk("max llam/mes", D["max_llamadas_por_mes"], 22.1, 0.05); chk("clv max", D["clv_stats"]["max"], 2165.28, 0.005)
chk("log clv max", D["clv_stats"]["log_max"], 7.68, 0.005); chk("log clv med", D["clv_stats"]["log_med"], 5.44, 0.005)
chk("r reclamos-tiene", D["r_reclamos_tiene"], 1.0); chk("pares>0.90", D["n_pares90"], 11, 0)
chk("corr fuga reclamos", D["corr_fuga"]["reclamos"], 0.511); chk("corr fuga estado_2", D["corr_fuga"]["estado_2"], 0.319)
chk("corr fuga fallas", D["corr_fuga"]["numero_fallas"], -0.009)
chk("fallas sola", D["fallas"]["sola"], 0.5261); chk("base5", D["fallas"]["base"], 0.9069); chk("base+f", D["fallas"]["base_f"], 0.9102)
chk("base+f+m", D["fallas"]["base_f_m"], 0.9171); chk("p fallas", D["sospechosas"]["numero_fallas"][2], 0.602)
chk("p edad", D["sospechosas"]["edad"][2], 0.297, 0.001)
chk("XOR F1", D["xor"]["F1"], 1.212); chk("XOR p1", D["xor"]["p1"], 0.271); chk("XOR F2", D["xor"]["F2"], 4.312); chk("XOR p2", D["xor"]["p2"], 0.038)
chk("XOR log", D["xor"]["log"], 0.510); chk("XOR inter", D["xor"]["inter"], 1.0); chk("XOR rf", D["xor"]["rf"], 1.0)
chk("VT descartadas", len(D["vt_fuera"]), 0, 0)
chk("MI segundos", D["rank12"][0][1], 0.1315); chk("MI intensidad", D["rank12"][9][1], 0.0946); chk("F intensidad", D["rank12"][9][2], 5.88, 0.005)
chk("MI llamadas_por_mes", D["rank12"][7][1], 0.0987); chk("pos reclamos MI", D["rank_mi_pos"]["reclamos"], 13, 0)
chk("MI clv", D["mi_par"]["clv"], 0.1018); chk("MI frec_sms", D["mi_par"]["frec_sms"], 0.0576)
chk("corr cruda seg-llam", D["corr_raw"]["segundos|frec_llamadas"], 0.907); chk("corr limpia seg-llam", D["corr_limpio"]["segundos|frec_llamadas"], 0.935)
chk("corr cruda llam-cont", D["corr_raw"]["frec_llamadas|n_contactos"], 0.728); chk("corr sms-clv", D["corr_limpio"]["frec_sms|clv"], 0.925)
chk("RFECV", D["rfe_n"], 42, 0); chk("lasso n", D["lasso_n"], 27, 0); chk("lasso fallas coef", D["lasso_top"][3][1], 0.517)
for c_, n_ in (("0.01", 13), ("0.05", 27), ("0.1", 33), ("0.5", 42), ("1.0", 45)):
    chk(f"lasso C={c_}", D["lasso_C"][c_], n_, 0)
chk("top4 auc", D["top4"]["auc"], 0.8040); chk("manual4 auc", D["manual4"]["auc"], 0.8934); chk("top4 cm", D["top4"]["cm"], 0.55); chk("manual4 cm", D["manual4"]["cm"], 0.22)
chk("k4 sin red", D["k_tabla"][0][2], 0.8754); chk("k6 top", D["k_tabla"][1][1], 0.8972); chk("k12 top", D["k_tabla"][4][1], 0.9013); chk("k12 sin red", D["k_tabla"][4][2], 0.8856)
chk("slide top6", D["slide_top6"]["auc"], 0.8105); chk("slide man6", D["slide_man6"]["auc"], 0.9041); chk("slide top6 cm", D["slide_top6"]["cm"], 0.50)
chk("slide man6 cm", D["slide_man6"]["cm"], 0.33); chk("nb top6", D["top6_nb"]["auc"], 0.8972)
chk("12 orig", D["resumen"][0][2], 0.9199); chk("50 cand", D["resumen"][1][2], 0.9406); chk("lasso auc", D["resumen"][2][2], 0.9396)
chk("ruido fuera", D["ruido"]["mal"], 0.882); chk("ruido dentro", D["ruido"]["bien"], 0.514); chk("p<0.05", D["ruido"]["signif05"], 57, 0)
chk("p<0.01", D["ruido"]["signif01"], 10, 0); chk("honesto media", D["honesto"]["media"], 0.9015, 0.0005)
chk("real fuera", D["real_fuera_dentro"]["fuera"], 0.9013); chk("real dentro", D["real_fuera_dentro"]["dentro"], 0.8996)
chk("estables", D["estab"]["siempre"], 16, 0); chk("a veces", D["estab"]["a_veces"], 19, 0); chk("nunca", D["estab"]["nunca"], 15, 0)
chk("años 2^50", D["anios50"], 35702, 0)
chk("par top4 seg-logseg", D["pares_top4"][0][2], 0.634); chk("suma pares top4", sum(r for *_, r in D["pares_top4"]), 3.314, 0.0015)
chk("suma pares manual4", sum(r for *_, r in D["pares_manual4"]), 1.324, 0.0015)

# ---- (2) ejemplos resueltos (ilustrativos o del apunte), recalculados
L = [180, 182, 179, -40]
chk("termometro media", sum(L) / 4, 125.25); chk("termometro mediana", sorted(L)[1] / 2 + sorted(L)[2] / 2, 179.5)
S = [3000, 3200, 2900, 3100, -14645]
chk("suma 4 buenos", sum(S[:4]), 12200, 0); chk("media con error", sum(S) / 5, -489, 0); chk("mediana con error", sorted(S)[2], 3000, 0)
chk("mediana sin error", (3000 + 3100) / 2, 3050, 0)
chk("ratio 60/20", 60 / (20 + 1e-6), 3.0, 1e-5); chk("ratio 4500/60", 4500 / (60 + 1e-6), 75, 1e-5); chk("ratio 60/0", 60 / (0 + 1e-6), 60e6, 1)
chk("ratio A 60/20", 60 / 20, 3, 0); chk("ratio B 60/2", 60 / 2, 30, 0)
chk("pares 50", 50 * 49 / 2, 1225, 0)
xs = [-2, -1, 0, 1, 2]
ys = [x * x for x in xs]
mx, my = sum(xs) / 5, sum(ys) / 5
chk("r de la U", sum((a - mx) * (b - my) for a, b in zip(xs, ys)), 0, 1e-12)
chk("var 0.5%", 0.005 * 0.995, 0.005, 0.00005); chk("var 1%", 0.01 * 0.99, 0.0099, 1e-9); chk("var 10%", 0.1 * 0.9, 0.09, 1e-9)
chk("umbral pi", (1 - math.sqrt(1 - 0.04)) / 2, 0.0101, 0.00005)
GA, GB = [2, 4, 6], [8, 10, 12]
m, ma, mb = 7, 4, 10
cme = (3 * (ma - m) ** 2 + 3 * (mb - m) ** 2) / 1
cmd = (sum((x - ma) ** 2 for x in GA) + sum((x - mb) ** 2 for x in GB)) / (6 - 2)
chk("CM entre", cme, 54, 0); chk("CM dentro", cmd, 4, 0); chk("F", cme / cmd, 13.5, 0)
for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    chk(f"XOR producto {x},{y}", x * y, 1 if (x > 0) == (y > 0) else -1, 0)
chk("2^10-1", 2 ** 10 - 1, 1023, 0); chk("2^20-1", 2 ** 20 - 1, 1048575, 0); chk("2^20 min", (2 ** 20 - 1) / 1000 / 60, 17.5, 0.05)
chk("2^30 dias", (2 ** 30 - 1) / 1000 / 86400, 12.4, 0.05); chk("2^30 ~1.07e9", (2 ** 30 - 1) / 1e9, 1.07, 0.005); chk("2^50 ~1.13e15", (2 ** 50 - 1) / 1e15, 1.13, 0.005)
chk("RFE p-k", 50 - 12, 38, 0)
chk("alfa*p 1", 1000 * 0.05, 50, 0); chk("alfa*p 2", 200 * 0.05, 10, 0); chk("alfa*p 3", 1000 * 0.01, 10, 0)
chk("inventado", D["ruido"]["mal"] - 0.5, 0.382, 1e-9)
woeA, woeB = math.log(0.625 / 0.30), math.log(0.375 / 0.70)
ivA, ivB = (0.625 - 0.30) * woeA, (0.375 - 0.70) * woeB
chk("WOE A", woeA, 0.734); chk("WOE B", woeB, -0.624); chk("IV A", ivA, 0.239); chk("IV B", ivB, 0.203); chk("IV", ivA + ivB, 0.44, 0.005)
chk("diferencia 4v4 (puntos)", (D["manual4"]["auc"] - D["top4"]["auc"]) * 100, 8.9, 0.05)
chk("diferencia 6v6 nb (puntos)", (D["slide_man6"]["auc"] - D["top6_nb"]["auc"]) * 100, 0.7, 0.05)
chk("inflacion real (puntos)", (D["real_fuera_dentro"]["fuera"] - D["real_fuera_dentro"]["dentro"]) * 100, 0.17, 0.005)

# ---- (3) cifras que el video toma del notebook impreso de la clase (no reproducibles exactamente: MI sin semilla)
NB = [0.8948, 0.9138, 0.8897, 0.9229, 0.8866]
chk("media folds notebook", sum(NB) / 5, 0.9015, 0.0001)
chk("rerun honesto vs notebook", D["honesto"]["media"], 0.9015, 0.0005)

print(f"{ok} comprobaciones OK, {len(mal)} fallidas")
for x in mal:
    print("  FALLA", x)
