"""Numeros reales para los videos de la Clase 38 (venv de Clase 35: pandas 3 + sklearn 1.9).

1) Ejecuta 38_evaluacion_seleccion.py del profesor TAL CUAL (runpy, sin modificarlo) con DIPDS_RESULTADOS apuntando a
   _tmp_c38_profe/resultados_clase (asi no escribe nada en la carpeta de la clase) y sin mostrar figuras.
   Lee los CSV que el propio script guarda (hormigon/ y spam/) y algunas variables finales de su ejecucion.
2) Calculos adicionales para los videos (marcados como tales en pantalla): promedio de RMSE por pliegue frente al RMSE OOF,
   residuos de prueba por edad, puntos de la ROC, etc.
Escribe datos38.json junto a este archivo.
"""
import json
import os
import runpy
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CLASE = next(p for p in RAIZ.glob("Clase 38*") if p.is_dir())
RES = AQUI / "resultados_clase"
os.environ["DIPDS_RESULTADOS"] = str(RES)
os.environ["DIPDS_MOSTRAR_FIGURAS"] = "0"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

# ------------------------------------------------------------ 1. el script de la clase, tal cual
G = runpy.run_path(str(CLASE / "38_evaluacion_seleccion.py"), run_name="__main__")
H = RES / "python" / "hormigon"
S = RES / "python" / "spam"
D = {"versiones": dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)}

# ---- hormigon
datos = pd.read_csv(CLASE / "hormigon_s38.csv")
tr = datos[datos.conjunto == "entrenamiento"]
te = datos[datos.conjunto == "prueba"]
D["h_part"] = dict(filas=len(datos), grupos=int(datos.grupo.nunique()), filas_tr=len(tr), grupos_tr=int(tr.grupo.nunique()),
                   filas_te=len(te), grupos_te=int(te.grupo.nunique()),
                   fold_filas=[int((tr.fold == k).sum()) for k in range(1, 6)],
                   fold_grupos=[int(tr[tr.fold == k].grupo.nunique()) for k in range(1, 6)],
                   media_tr=round(float(tr.resistencia.mean()), 3), media_te=round(float(te.resistencia.mean()), 3),
                   min_tr=round(float(tr.resistencia.min()), 2), max_tr=round(float(tr.resistencia.max()), 2),
                   edades=sorted(int(e) for e in datos.edad.unique()))
# grupo con mas edades (para el ejemplo de composicion repetida)
gmax = datos.groupby("grupo").edad.nunique().sort_values(ascending=False, kind="stable")
g0 = int(gmax.index[0])
D["h_grupo_ej"] = dict(grupo=g0, n_edades=int(gmax.iloc[0]),
                       filas=[dict(edad=int(r.edad), res=round(float(r.resistencia), 1), conjunto=r.conjunto)
                              for r in datos[datos.grupo == g0].sort_values("edad").itertuples()],
                       grupos_con_varias=int((datos.groupby("grupo").size() > 1).sum()),
                       filas_en_grupos_varias=int(datos.groupby("grupo").size()[lambda s: s > 1].sum()))
rk = pd.read_csv(H / "comparacion_cv.csv")
D["h_ranking"] = [dict(modelo=r.modelo, rmse=round(float(r.RMSE), 6), mae=round(float(r.MAE), 6), r2=round(float(r.R2), 6),
                       rmse_ajuste=round(float(r.RMSE_ajuste), 6)) for r in rk.itertuples()]
pf = pd.read_csv(H / "metricas_por_fold.csv")
D["h_fold"] = {m: [dict(fold=int(r.fold), n=int(r.n), rmse=round(float(r.RMSE), 6)) for r in pf[pf.modelo == m].itertuples()]
               for m in pf.modelo.unique()}
ev = pd.read_csv(H / "evaluacion_final.csv").iloc[0]
rf = pd.read_csv(H / "referencia.csv").iloc[0]
D["h_final"] = dict(modelo=ev.modelo, rmse=round(float(ev.RMSE), 6), mae=round(float(ev.MAE), 6), r2=round(float(ev.R2), 6),
                    ref_rmse=round(float(rf.RMSE), 6), ref_mae=round(float(rf.MAE), 6), ref_r2=round(float(rf.R2), 6),
                    baja=round(float(1 - ev.RMSE / rf.RMSE), 6))
pt = pd.read_csv(H / "predicciones_test.csv")
res = pt.residuo.to_numpy()
pte = pt.merge(te[["id", "edad"]], on="id")
D["h_resid"] = dict(n=len(pt), media=round(float(res.mean()), 3), max=round(float(res.max()), 2), min=round(float(res.min()), 2),
                    abs_max=round(float(np.abs(res).max()), 2), pos=int((res > 0).sum()),
                    sobre15=int((np.abs(res) > 15).sum()), sobre20=int((np.abs(res) > 20).sum()),
                    rmse_edad={str(lbl): dict(n=int(len(g)), rmse=round(float(np.sqrt(np.mean(g.residuo ** 2))), 3))
                               for lbl, g in pte.groupby(pd.cut(pte.edad, [0, 7, 28, 400], labels=["1-7", "14-28", "56-365"]),
                                                         observed=True)},
                    ej=[dict(id=int(r.id), real=round(float(r.real), 1), pred=round(float(r.predicha), 1),
                             base=round(float(r.referencia), 1)) for r in pt.head(4).itertuples()],
                    pred=[round(float(v), 2) for v in pt.predicha], real=[round(float(v), 2) for v in pt.real],
                    edad=[int(v) for v in pte.edad])
# adicional: promedio de RMSE por pliegue frente a RMSE OOF (mismo candidato)
D["h_prom_vs_oof"] = {m: dict(prom=round(float(np.mean([f["rmse"] for f in D["h_fold"][m]])), 6),
                              oof=round(float(next(r["rmse"] for r in D["h_ranking"] if r["modelo"] == m)), 6))
                      for m in ("Q_A0.1", "Q_A1", "LINEAL")}

# ---- spam
sp = pd.read_csv(CLASE / "spam_s38.csv")
str_ = sp[sp.conjunto == "entrenamiento"]
ste = sp[sp.conjunto == "prueba"]
D["s_part"] = dict(n=len(sp), vars=57, tr=len(str_), te=len(ste), n0=int((str_.spam == 0).sum()), n1=int((str_.spam == 1).sum()),
                   prev=round(float(str_.spam.mean()), 6), te_spam=int(ste.spam.sum()), te_no=int((ste.spam == 0).sum()),
                   grupos=int(sp.grupo.nunique()), n_total_spam=int(sp.spam.sum()))
rks = pd.read_csv(S / "comparacion_cv.csv")
D["s_ranking"] = [dict(modelo=r.modelo, log_loss=round(float(r.log_loss), 6), brier=round(float(r.Brier), 6),
                       auc=round(float(r.ROC_AUC), 6), ap=round(float(r.AP), 6)) for r in rks.itertuples()]
cal = pd.read_csv(S / "calibracion.csv")
D["s_cal"] = [dict(p=round(float(r.media_probabilidad), 6), f=round(float(r.fraccion_spam), 6), n=int(r.n)) for r in cal.itertuples()]
co = pd.read_csv(S / "costos_umbrales.csv")
D["s_costos"] = [dict(t=round(float(r.umbral), 2), VN=int(r.VN), FP=int(r.FP), FN=int(r.FN), VP=int(r.VP), acc=round(float(r.accuracy), 6),
                      prec=round(float(r.precision), 6), rec=round(float(r.recall), 6), f1=round(float(r.F1), 6),
                      c1000=round(float(1000 * r.costo), 2)) for r in co.itertuples()]
D["s_umbral"] = float(G["umbral"])
fin = pd.read_csv(S / "evaluacion_final.csv").iloc[0]
ref = pd.read_csv(S / "referencia.csv").iloc[0]
D["s_final"] = dict(modelo=fin.modelo, log_loss=round(float(fin.log_loss), 6), brier=round(float(fin.Brier), 6),
                    auc=round(float(fin.ROC_AUC), 6), ap=round(float(fin.AP), 6), VN=int(fin.VN), FP=int(fin.FP), FN=int(fin.FN),
                    VP=int(fin.VP), acc=round(float(fin.accuracy), 6), prec=round(float(fin.precision), 6),
                    rec=round(float(fin.recall), 6), f1=round(float(fin.F1), 6), c1000=round(float(1000 * fin.costo), 2),
                    ref_c1000=round(float(1000 * ref.costo), 2), ref_FN=int(ref.FN), ref_acc=round(float(ref.accuracy), 6))
# adicional: puntos de la ROC OOF del ganador y la ROC/PR en umbrales concretos
oofs = pd.read_csv(S / "predicciones_oof.csv")
g = oofs[oofs.modelo == G["ganador"]]
y, p = g.real.to_numpy(), g.probabilidad.to_numpy()
fpr, tpr, thr = roc_curve(y, p)
idx = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
D["s_roc"] = dict(fpr=[round(float(v), 6) for v in fpr[idx]], tpr=[round(float(v), 6) for v in tpr[idx]])
pts = {}
for t in (0.5, 0.6, 0.8):
    vn, fp, fn, vp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    pts[f"{t:.2f}"] = dict(fpr=round(float(fp / (fp + vn)), 6), rec=round(float(vp / (vp + fn)), 6), prec=round(float(vp / (vp + fp)), 6))
D["s_puntos"] = pts
D["s_hist"] = [int(v) for v in np.histogram(p, bins=np.linspace(0, 1, 11))[0]]
D["s_ganador"] = G["ganador"]
(AQUI / "datos38.json").write_text(json.dumps(D, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in D.items() if k not in ("s_roc",)}, ensure_ascii=False, default=float))
