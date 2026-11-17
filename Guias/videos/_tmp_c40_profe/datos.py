"""Numeros reales para los videos de la Clase 40 (venv de Clase 35: pandas 3 + sklearn 1.9).

1) Ejecuta 40_naive_bayes.py del profesor TAL CUAL (runpy, sin modificar el archivo). Solo se redirige:
   - la carpeta de salida (variable de entorno DIPDS_RESULTADOS que el propio script lee) -> _tmp_c40_profe/resultados_clase;
   - las figuras: el script no llama plt.show() salvo en modo interactivo (MOSTRAR = 0 al ejecutarlo como archivo);
     igual se fija DIPDS_MOSTRAR_FIGURAS=0 y backend Agg; los PDF quedan en resultados_clase/python;
   - la salida impresa -> resultados_clase/salida_python.txt.
   Despues lee las variables finales de esa ejecucion (comparacion, cal, costos, resultado, explicacion, ...).
2) Calculos adicionales para los videos (marcados como tales en pantalla): alfa 0.5 y 2 en el ejemplo de 100 correos,
   log-probabilidades del correo ID 1, desbordamiento de un producto de probabilidades, conteos minimos por senal, etc.
Escribe datos40.json (al menos 6 decimales) junto a este archivo.
"""
import contextlib
import io
import json
import os
import runpy
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CLASE = next(p for p in RAIZ.glob("Clase 40*") if p.is_dir())
RES = AQUI / "resultados_clase"
RES.mkdir(exist_ok=True)
os.environ["DIPDS_RESULTADOS"] = str(RES)
os.environ["DIPDS_MOSTRAR_FIGURAS"] = "0"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.naive_bayes import BernoulliNB  # noqa: E402

R6 = lambda v: round(float(v), 6)  # noqa: E731
L6 = lambda xs: [round(float(v), 6) for v in xs]  # noqa: E731

# ------------------------------------------------------------ 1. el script de la clase, tal cual
buf = io.StringIO()
cwd0 = os.getcwd()
os.chdir(RES)
try:
    with contextlib.redirect_stdout(buf):
        G = runpy.run_path(str(CLASE / "40_naive_bayes.py"), run_name="__main__")
finally:
    os.chdir(cwd0)
(RES / "salida_python.txt").write_text(buf.getvalue(), encoding="utf-8")

D = {"versiones": dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)}

# ---- 01 Bayes con una senal
D["bayes"] = dict(prior=L6(G["pi"]), oferta_spam=R6(G["oferta_spam"]), oferta_leg=R6(G["oferta_legitimo"]),
                  posterior=L6(G["p"]))
g = np.array([0.001, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98])
D["bayes"]["curva"] = dict(g=L6(g), p=L6(g * 0.8 / (g * 0.8 + (1 - g) * 0.1)))
# tabla de 100 correos y combinaciones (diapositivas 5-18)
D["bayes"]["s_leg_oferta"] = R6(0.8 * 0.1)
D["bayes"]["s_spam_oferta"] = R6(0.2 * 0.8)
D["bayes"]["p_oferta"] = R6(16 / 24)
pats = {}
for of in (0, 1):
    for en in (0, 1):
        s0 = 0.8 * (0.1 if of else 0.9) * (16 / 80 if en else 64 / 80)
        s1 = 0.2 * (0.8 if of else 0.2) * (14 / 20 if en else 6 / 20)
        pats[f"{of}{en}"] = dict(s0=R6(s0), s1=R6(s1), p=R6(s1 / (s0 + s1)))
D["bayes"]["patrones"] = pats

# ---- 02 Bernoulli con y sin suavizado (del script) + alfa 0.5 y 2 (adicional)
D["bern100"] = dict(p_spam=R6(G["p_spam"]), s_leg=R6(G["s_legitimo"]), s_spam=R6(G["s_spam"]),
                    tabla=[{k: R6(v) for k, v in f.items()} for f in G["filas"]])
extra = []
n, conteos, x_nuevo = G["n"], G["conteos"], G["x_nuevo"]
for alfa in (0.5, 1.0, 2.0):
    theta = (conteos + alfa) / (n[:, None] + 2 * alfa)
    lc = np.log(n / n.sum()) + (x_nuevo * np.log(theta) + (1 - x_nuevo) * np.log1p(-theta)).sum(axis=1)
    extra.append(dict(alfa=alfa, posterior=R6(G["posterior"](lc[None, :])[0, 1]), theta=[L6(r) for r in theta]))
D["bern100"]["alfas"] = extra
D["bern100"]["premio"] = R6(1 / 22)

# ---- 03 duplicacion
D["dup"] = dict(copias=[int(c) for c in G["copias"]], post=L6(G["prob_copias"]))

# ---- 04 gaussiana y multinomial
D["gauss"] = dict(d0=R6(G["densidad_0"]), d1=R6(G["densidad_1"]), s0=R6(G["s0_gauss"]), s1=R6(G["s1_gauss"]),
                  post=R6(G["s1_gauss"] / (G["s0_gauss"] + G["s1_gauss"])))
D["multi"] = dict(theta=[L6(r) for r in G["theta_m"]], post=L6(G["pm"]))

# ---- 05 Spambase
datos, ent, prueba, X, y = G["datos"], G["ent"], G["prueba"], G["X"], G["y"]
part = G["datos"].groupby(["conjunto", "spam"]).size()
D["spam"] = dict(n=int(len(datos)), n_ent=int(len(ent)), n_prueba=int(len(prueba)), p=int(len(G["variables"])),
                 particion={f"{c}_{s}": int(v) for (c, s), v in part.items()},
                 n0=int((y == 0).sum()), n1=int((y == 1).sum()), prior_spam=R6(y.mean()),
                 folds=[int(v) for v in np.bincount(G["folds"])[1:]], n_grupos=int(datos.grupo.nunique()),
                 freq_max=R6(datos[G["variables"]].to_numpy().max()),
                 freq_max_col=str(datos[G["variables"]].max().idxmax()))
id1 = ent.loc[ent.id.idxmin()]
D["spam"]["id1"] = dict(id=int(id1.id), free=R6(id1.word_freq_free), money=R6(id1.word_freq_money), excl=R6(id1["char_freq_!"]),
                        z_free=R6(np.log1p(id1.word_freq_free)), spam=int(id1.spam), presentes=int((X[G["indice"]] > 0).sum()))

# ---- 06 comparacion OOF
cmp_ = G["comparacion"]
D["oof"] = {r.modelo: dict(log_loss=R6(r.log_loss), brier=R6(r.brier), auc=R6(r.auc), accuracy=R6(r.accuracy),
                           VN=int(r.VN), FP=int(r.FP), FN=int(r.FN), VP=int(r.VP), costo=R6(r.costo_1000)) for r in cmp_.itertuples()}
D["oof_ganador"] = str(G["ganador"])
pg = G["oof"]["GAUSS_LOG"]
pb = G["oof"]["BERN_05"]
D["oof_extremos"] = dict(gauss_fuera_001=R6(np.mean((pg < 0.01) | (pg > 0.99))), bern05_fuera_001=R6(np.mean((pb < 0.01) | (pb > 0.99))))
# error mas caro de la gaussiana (adicional): legitimos con p >= 0.999 y spam con p <= 0.001
D["oof_extremos"]["gauss_leg_999"] = int(np.sum((y == 0) & (pg >= 0.999)))
D["oof_extremos"]["gauss_spam_001"] = int(np.sum((y == 1) & (pg <= 0.001)))
D["oof_extremos"]["bern_leg_999"] = int(np.sum((y == 0) & (pb >= 0.999)))

# ---- 07 calibracion
cal = G["cal"]
D["cal"] = [dict(modelo=r.modelo, bin=int(r.bin), n=int(r.n), prob_media=R6(r.prob_media), fraccion=R6(r.fraccion_spam))
            for r in cal.itertuples()]
c9 = cal[(cal.modelo == "BERN_1") & (cal.bin == 9)].iloc[0]
D["cal9"] = dict(n=int(c9.n), prob=R6(c9.prob_media), frac=R6(c9.fraccion_spam), spam=int(round(c9.fraccion_spam * c9.n)))

# ---- 08 correo ID 1
ex = G["explicacion"]
D["id1"] = dict(posterior=R6(G["modelo_bern"].predict_proba(X[G["indice"]:G["indice"] + 1])[0, 1]),
                prior_log=R6(G["prior_log"]), suma_aportes=R6(G["aportes"].sum()),
                n_spam_lado=int((G["aportes"] > 0).sum()), n_leg_lado=int((G["aportes"] < 0).sum()))
top = G["top"]
D["id1"]["top"] = [dict(var=str(r.variable).replace("word_freq_", "").replace("char_freq_", ""), presente=int(r.presente),
                        c0=int(r.conteo_legitimo), c1=int(r.conteo_spam), p0=R6(r.prob_legitimo), p1=R6(r.prob_spam),
                        aporte=R6(r.aporte_log_odds)) for r in top.itertuples()]
row = ex[ex.variable == "word_freq_address"].iloc[0]
D["id1"]["address"] = dict(c0=int(row.conteo_legitimo), c1=int(row.conteo_spam), p0=R6(row.prob_legitimo), p1=R6(row.prob_spam),
                           aporte=R6(row.aporte_log_odds))
for nom in ("word_freq_free", "word_freq_money", "char_freq_!"):
    r = ex[ex.variable == nom].iloc[0]
    D["id1"][nom.split("_")[-1]] = dict(presente=int(r.presente), c0=int(r.conteo_legitimo), c1=int(r.conteo_spam),
                                         p0=R6(r.prob_legitimo), p1=R6(r.prob_spam), aporte=R6(r.aporte_log_odds))
# log-probabilidades conjuntas del correo ID 1 (adicional): log prior + suma de log factores, por clase
mb = G["modelo_bern"]
jl = mb.predict_joint_log_proba(X[G["indice"]:G["indice"] + 1])[0]
D["id1"]["log_conj"] = L6(jl)
D["id1"]["log_odds"] = R6(jl[1] - jl[0])
D["id1"]["prod"] = [float(f"{v:.6e}") for v in np.exp(jl)]
# conteos minimos (adicional): ninguna senal queda con cero casos en el desarrollo
Xb = (X > 0)
pres0, pres1 = Xb[y == 0].sum(axis=0), Xb[y == 1].sum(axis=0)
D["spam"]["min_pres"] = [int(pres0.min()), int(pres1.min())]
D["spam"]["min_aus"] = [int((~Xb[y == 0]).sum(axis=0).min()), int((~Xb[y == 1]).sum(axis=0).min())]
D["spam"]["min_pres_var"] = [str(np.array(G["variables"])[pres0.argmin()]), str(np.array(G["variables"])[pres1.argmin()])]

# ---- 09 umbrales y comprobacion historica
co = G["costos"]
D["umbral"] = dict(t=R6(G["t"]), tabla=[dict(t=R6(r.umbral), costo=R6(r.costo_1000), FP=int(r.FP), FN=int(r.FN), acc=R6(r.accuracy))
                                       for r in co.itertuples()])
res = G["resultado"]
r0, r1 = res.iloc[0], res.iloc[1]
D["prueba"] = dict(modelo=str(r0.modelo), umbral=R6(r0.umbral), VN=int(r0.VN), FP=int(r0.FP), FN=int(r0.FN), VP=int(r0.VP),
                   costo=R6(r0.costo_1000), log_loss=R6(r0.log_loss), brier=R6(r0.brier), auc=R6(r0.auc), acc=R6(r0.accuracy),
                   base_costo=R6(r1.costo_1000), base_FN=int(r1.FN), base_logloss=R6(r1.log_loss))

# ---- adicional: desbordamiento de un producto de probabilidades
D["log"] = dict(p03_618=float(f"{0.3 ** 618:.6e}"), p03_619=float(0.3 ** 619), log_619=R6(619 * np.log(0.3)),
                n_desborde=int(next(k for k in range(1, 2000) if 0.3 ** k == 0.0)))
# probabilidades de presencia de 'free' (para la cita de la clase)
rf = ex[ex.variable == "word_freq_free"].iloc[0]
D["free"] = dict(p_spam_pres=R6(pres1[G["variables"].index("word_freq_free")] / D["spam"]["n1"]),
                 p_leg_pres=R6(pres0[G["variables"].index("word_freq_free")] / D["spam"]["n0"]))

# ---- adicional: repetir las 54 senales k veces (dependencia extrema) con Bernoulli alfa = 1, mismos pliegues OOF
folds = G["folds"]
dup = []
for k in (1, 2, 4):
    Xk = np.hstack([X] * k)
    pred, lp = np.empty(len(y)), np.empty((len(y), 2))
    for f in sorted(np.unique(folds)):
        val = folds == f
        pred[val], _, lp[val] = G["ajustar_predecir"](Xk[~val], y[~val], Xk[val], "bern", 1.0)
    m = G["medir"](y, pred, logp=lp)
    dup.append(dict(k=k, log_loss=R6(m["log_loss"]), auc=R6(m["auc"]), accuracy=R6(m["accuracy"])))
D["dup_real"] = dup

(AQUI / "datos40.json").write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(D, ensure_ascii=False)[:6000])
