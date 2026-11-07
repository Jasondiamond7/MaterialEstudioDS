"""Numeros reales para los videos de la Clase 37 (venv de Clase 35: pandas 3 + sklearn 1.9).

1) Importa figuras_c37.py (misma base y particion que la guia) con `save` anulado: no se sobrescribe ninguna figura.
   Ejecuta todos sus calculos (margen, C, kernels, gamma, rejilla, comparacion, umbral, escala) y numeros_base().
2) Replica clasificadores.py del profesor tal cual (bloques 1-9 y el opcional de GridSearchCV) sobre wisconsin_clase.csv.
3) Calculos adicionales para los videos (marcados como tales en pantalla): kNN, LDA y QDA con la misma particion,
   vecinos del caso nuevo, perdidas hinge y logistica, separacion perfecta (MLE), puntos de soporte del margen duro.
Escribe datos37.json junto a este archivo.
"""
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
sys.path.insert(0, str(FUENTE))
import matplotlib.pyplot as plt  # noqa: E402
import figuras_c37 as F  # noqa: E402

F.save = lambda fig, name: plt.close(fig)          # no tocar Guias/_fuente/img
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

D = {"versiones": dict(sklearn=sklearn.__version__, pandas=pd.__version__)}

# ------------------------------------------------------------ 1. figuras de la guia
F.numeros_base()
for k in ["margen_fig", "c_fig", "kernels_fig", "gamma_fig", "rejilla_fig", "comparacion_fig", "umbral_fig", "escala_fig"]:
    getattr(F, k)()
D["fig"] = json.loads(json.dumps(F.NUM, default=float))

# ------------------------------------------------------------ 2. clasificadores.py tal cual
datos = pd.read_csv(F.CARPETA / "wisconsin_clase.csv")
D["n"] = int(len(datos))
D["conteo"] = {str(k): int(v) for k, v in datos["maligno"].value_counts().items()}
entrenamiento = datos[datos["conjunto"] == "entrenamiento"]
prueba = datos[datos["conjunto"] == "prueba"]
variables = ["radio", "textura", "suavidad", "concavidad"]
x_entrenamiento, x_prueba = entrenamiento[variables], prueba[variables]
y_entrenamiento, y_prueba = entrenamiento["maligno"], prueba["maligno"]
escala = StandardScaler()
x_entrenamiento_escalado = escala.fit_transform(x_entrenamiento)
x_prueba_escalado = escala.transform(x_prueba)
D["escala"] = dict(media={v: round(float(m), 5) for v, m in zip(variables, escala.mean_)},
                   sd_poblacional={v: round(float(s), 5) for v, s in zip(variables, escala.scale_)})

logistica = LogisticRegression(penalty=None, max_iter=2000).fit(x_entrenamiento_escalado, y_entrenamiento)
probabilidad = logistica.predict_proba(x_prueba_escalado)[:, 1]
clase_logistica = (probabilidad >= 0.5).astype(int)
svm_lineal = SVC(kernel="linear", C=0.1).fit(x_entrenamiento_escalado, y_entrenamiento)
clase_lineal = svm_lineal.predict(x_prueba_escalado)
svm_radial = SVC(kernel="rbf", C=10, gamma=0.1).fit(x_entrenamiento_escalado, y_entrenamiento)
clase_radial = svm_radial.predict(x_prueba_escalado)


def metr(y, c):
    tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), acc=round(float(accuracy_score(y, c)), 4),
                sens=round(float(recall_score(y, c, pos_label=1)), 4), espe=round(float(recall_score(y, c, pos_label=0)), 4),
                prec=round(float(tp / (tp + fp)) if tp + fp else 0.0, 4))


D["script"] = {"Logistica": metr(y_prueba, clase_logistica), "SVM lineal": metr(y_prueba, clase_lineal),
               "SVM radial": metr(y_prueba, clase_radial)}
D["script_auc"] = {"Logistica": round(float(roc_auc_score(y_prueba, probabilidad)), 4),
                   "SVM lineal": round(float(roc_auc_score(y_prueba, svm_lineal.decision_function(x_prueba_escalado))), 4),
                   "SVM radial": round(float(roc_auc_score(y_prueba, svm_radial.decision_function(x_prueba_escalado))), 4)}
# AUC mal calculada con clases 0/1 (error frecuente de la diapositiva 58)
D["auc_con_clases"] = {"SVM radial": round(float(roc_auc_score(y_prueba, clase_radial)), 4),
                       "Logistica": round(float(roc_auc_score(y_prueba, clase_logistica)), 4)}
D["primeras_prob"] = [round(float(p), 4) for p in probabilidad[:6]]
nuevo = pd.DataFrame({"radio": [14], "textura": [20], "suavidad": [0.10], "concavidad": [0.08]})
nuevo_escalado = escala.transform(nuevo)
D["nuevo"] = dict(z=[round(float(v), 4) for v in nuevo_escalado[0]], p_log=round(float(logistica.predict_proba(nuevo_escalado)[0, 1]), 4),
                  radial=int(svm_radial.predict(nuevo_escalado)[0]), lineal=int(svm_lineal.predict(nuevo_escalado)[0]),
                  f_radial=round(float(svm_radial.decision_function(nuevo_escalado)[0]), 4),
                  f_lineal=round(float(svm_lineal.decision_function(nuevo_escalado)[0]), 4))
D["n_support"] = dict(lineal=[int(v) for v in svm_lineal.n_support_], radial=[int(v) for v in svm_radial.n_support_])
D["logistica_coef_std"] = dict(b0=round(float(logistica.intercept_[0]), 4), b=[round(float(v), 4) for v in logistica.coef_[0]])
# linea base: siempre benigno / siempre maligno
D["base_benigno"] = metr(y_prueba, np.zeros(len(y_prueba), dtype=int))
D["base_maligno"] = metr(y_prueba, np.ones(len(y_prueba), dtype=int))

# bloque opcional: GridSearchCV del profesor (comentado en el script)
modelo = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
busqueda = GridSearchCV(modelo, {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}, cv=5, scoring="accuracy")
busqueda.fit(x_entrenamiento, y_entrenamiento)
res = pd.DataFrame(busqueda.cv_results_)
D["grid_prof"] = dict(best=dict(C=float(busqueda.best_params_["svc__C"]), gamma=float(busqueda.best_params_["svc__gamma"])),
                      best_score=round(float(busqueda.best_score_), 4),
                      tabla=[dict(C=float(r.param_svc__C), gamma=float(r.param_svc__gamma), media=round(float(r.mean_test_score), 4),
                                  sd=round(float(r.std_test_score), 4), rank=int(r.rank_test_score)) for r in res.itertuples()])

# ------------------------------------------------------------ 3. calculos adicionales (no estan en la clase)
Xtr, Xte, ytr, yte = x_entrenamiento_escalado, x_prueba_escalado, y_entrenamiento.to_numpy(), y_prueba.to_numpy()
otros = {}
for nombre, m, esc in [("kNN (k = 5)", KNeighborsClassifier(5), True), ("kNN sin escalar", KNeighborsClassifier(5), False),
                       ("LDA", LinearDiscriminantAnalysis(), True), ("QDA", QuadraticDiscriminantAnalysis(), True)]:
    if esc:
        m.fit(Xtr, ytr)
        c = m.predict(Xte)
        s = m.predict_proba(Xte)[:, 1]
    else:
        m.fit(x_entrenamiento.to_numpy(), ytr)
        c = m.predict(x_prueba.to_numpy())
        s = m.predict_proba(x_prueba.to_numpy())[:, 1]
    otros[nombre] = dict(**metr(yte, c), auc=round(float(roc_auc_score(yte, s)), 4))
D["otros"] = otros
# los 5 vecinos del caso nuevo (escalado con el entrenamiento) y sin escalar
for clave, Xref, q in [("vecinos", Xtr, nuevo_escalado[0]), ("vecinos_sin_escalar", x_entrenamiento.to_numpy(), nuevo.to_numpy()[0])]:
    dist = np.sqrt(((Xref - q) ** 2).sum(axis=1))
    idx = np.argsort(dist, kind="stable")[:5]
    D[clave] = [dict(d=round(float(dist[i]), 4), y=int(ytr[i]), x=[round(float(v), 4) for v in x_entrenamiento.to_numpy()[i]]) for i in idx]
knn = KNeighborsClassifier(5).fit(Xtr, ytr)
D["knn_nuevo"] = dict(p=round(float(knn.predict_proba(nuevo_escalado)[0, 1]), 4), clase=int(knn.predict(nuevo_escalado)[0]))
# sd muestrales (como en figuras) y ratio
sd = x_entrenamiento.std()
D["sd"] = {k: round(float(v), 4) for k, v in sd.items()}
# perdidas para y*f de la diapositiva 36
D["perdidas"] = {str(v): dict(hinge=round(max(0.0, 1 - v), 4), log=round(math.log(1 + math.exp(-v)), 4)) for v in (1.4, 0.5, -0.3, 2.0, 0.0, 3.0)}
# separacion perfecta (notebook E2): log-verosimilitud creciente
x_sep = np.array([-3, -2, -1, 1, 2, 3.0])
y_sep = np.array([0, 0, 0, 1, 1, 1])
D["separacion"] = {str(b): float(-np.sum(np.logaddexp(0, -(2 * y_sep - 1) * b * x_sep))) for b in (1, 2, 5, 10, 20)}
# margen duro (32 casos de la figura): y*f de cada punto, para decidir los vectores de soporte
rng_sub = pd.concat([F.tr[(F.tr["maligno"] == 0) & (F.tr["radio"] < 13) & (F.tr["concavidad"] < 0.06)].sample(18, random_state=3),
                     F.tr[(F.tr["maligno"] == 1) & (F.tr["radio"] > 15.5) & (F.tr["concavidad"] > 0.08)].sample(14, random_state=3)])
Zs = F.esc2.transform(rng_sub[F.V2])
ys = rng_sub["maligno"].to_numpy()
duro = SVC(kernel="linear", C=1e6).fit(Zs, ys)
yf = (2 * ys - 1) * duro.decision_function(Zs)
orden = np.argsort(yf)
D["duro_yf"] = [dict(z=[round(float(v), 3) for v in Zs[i]], y=int(ys[i]), yf=round(float(yf[i]), 3), sv=bool(i in set(duro.support_))) for i in orden[:8]]
D["duro_yf_max"] = round(float(yf.max()), 3)
D["duro"] = dict(w=[round(float(v), 4) for v in duro.coef_[0]], b=round(float(duro.intercept_[0]), 4),
                 norma=round(float(np.linalg.norm(duro.coef_[0])), 4), ancho=round(float(2 / np.linalg.norm(duro.coef_[0])), 4),
                 n_sv=int(len(duro.support_)))
# lineal C=0.1 del profesor: cuantas filas cumplen y*f <= 1 (soporte) frente a n_support_
fy4 = (2 * ytr - 1) * svm_lineal.decision_function(Xtr)
D["lineal_yf"] = dict(menor1=int((fy4 < 1 - 1e-3).sum()), igual1=int((np.abs(fy4 - 1) <= 1e-3).sum()), n_sv=int(len(svm_lineal.support_)))
# logistica: odds y probabilidad del ejemplo de la diapositiva 17
eta = -1 + 0.8 * 1.5 + 0.4 * 2
D["ej_logistica"] = dict(eta=eta, p=round(1 / (1 + math.exp(-eta)), 4), odds08=round(math.exp(0.8), 4))
D["glm_suavidad_001"] = round(math.exp(124.9794 * 0.01), 4)
D["glm_textura_odds"] = round(math.exp(0.4421), 4)
# defecto de LogisticRegression (C=1, L2) frente a penalty=None
d = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
D["logistica_defecto"] = dict(**metr(yte, d.predict(Xte)), coef=[round(float(v), 4) for v in d.coef_[0]])

# ------------------------------------------------------------ 4. geometria 2D para dibujar (radio y concavidad estandarizados)
Z2tr, Z2te = F.Z2tr, F.Z2te
D["z2_train"] = dict(x=[round(float(v), 3) for v in Z2tr[:, 0]], y=[round(float(v), 3) for v in Z2tr[:, 1]], c=[int(v) for v in F.ytr])
D["z2_test"] = dict(x=[round(float(v), 3) for v in Z2te[:, 0]], y=[round(float(v), 3) for v in Z2te[:, 1]], c=[int(v) for v in F.yte])
D["duro_pts"] = dict(x=[round(float(v), 3) for v in Zs[:, 0]], y=[round(float(v), 3) for v in Zs[:, 1]], c=[int(v) for v in ys],
                     sv=[int(i) for i in duro.support_])


def contornos(modelo, nivel=0.0, n=300, paso=3):
    gx, gy = np.meshgrid(np.linspace(F.LIM[0], F.LIM[1], n), np.linspace(F.LIM[2], F.LIM[3], n))
    f = modelo.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    cs = plt.contour(gx, gy, f, levels=[nivel])
    segs = []
    for s in cs.allsegs[0]:
        if len(s) < 8:
            continue
        s = s[::paso] if len(s) > 3 * paso else s
        if s[-1, 0] < s[0, 0]:
            s = s[::-1]
        segs.append(dict(x=[round(float(v), 3) for v in s[:, 0]], y=[round(float(v), 3) for v in s[:, 1]]))
    plt.close("all")
    return segs


lin2 = {}
for Cv in (0.01, 1, 100):
    m = SVC(kernel="linear", C=Cv).fit(Z2tr, F.ytr)
    lin2[str(Cv)] = dict(w=[round(float(v), 4) for v in m.coef_[0]], b=round(float(m.intercept_[0]), 4), n_sv=int(len(m.support_)),
                         ancho=round(float(2 / np.linalg.norm(m.coef_[0])), 3), acc_test=round(float(m.score(Z2te, F.yte)), 3),
                         acc_train=round(float(m.score(Z2tr, F.ytr)), 3))
D["lin2"] = lin2
rbf2 = {}
for gam in (0.01, 1, 50):
    m = SVC(kernel="rbf", C=1, gamma=gam).fit(Z2tr, F.ytr)
    rbf2[str(gam)] = dict(segs=contornos(m), acc_train=round(float(m.score(Z2tr, F.ytr)), 3), acc_test=round(float(m.score(Z2te, F.yte)), 3),
                          n_sv=int(len(m.support_)))
D["rbf2"] = rbf2
m = SVC(kernel="rbf", C=10, gamma=0.1).fit(Z2tr, F.ytr)
D["rbf2_prof"] = dict(segs=contornos(m), acc_test=round(float(m.score(Z2te, F.yte)), 3))
m = LogisticRegression(penalty=None, max_iter=2000).fit(Z2tr, F.ytr)
D["log2"] = dict(b0=round(float(m.intercept_[0]), 4), b=[round(float(v), 4) for v in m.coef_[0]], acc_test=round(float(m.score(Z2te, F.yte)), 3))
# regresion lineal con 0/1 (no da probabilidades): rango de la prediccion en prueba
from sklearn.linear_model import LinearRegression  # noqa: E402
yh = LinearRegression().fit(Xtr, ytr).predict(Xte)
D["lineal01"] = dict(min=round(float(yh.min()), 2), max=round(float(yh.max()), 2), bajo0=int((yh < 0).sum()), sobre1=int((yh > 1).sum()))

(AQUI / "datos37.json").write_text(json.dumps(D, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in D.items() if k != "fig"}, ensure_ascii=False, indent=1, default=float))
