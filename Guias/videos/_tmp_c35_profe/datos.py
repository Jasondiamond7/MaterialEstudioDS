"""Numeros reales para los videos de la Clase 35 (venv de Clase 35: pandas 3 + sklearn 1.9 + openpyxl).

1) Importa figuras_c35.py (misma base y particiones que la guia) con `save` anulado: no se sobrescribe ninguna figura.
   Ejecuta sus calculos (bootstrap, varianza, max_features, curvas, importancia, OOB/CV, imagenes, comparacion).
2) Replica el notebook del profesor (Ejemplo Random Forest y Bagging.ipynb): modelos base de Heart y Ames con OOB,
   y GridSearchCV de clasificacion (Heart). La busqueda de Ames no se ejecuta (costosa y no se usa en los videos).
3) Calculos propios para los ejemplos del video (arbol suelto vs promedio en Heart, bosque con hojas de 1 en train).
Escribe datos35.json junto a este archivo.
"""
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
sys.path.insert(0, str(FUENTE))
import matplotlib.pyplot as plt  # noqa: E402
import figuras_c35 as F  # noqa: E402

F.save = lambda fig, name: plt.close(fig)          # no tocar Guias/_fuente/img
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

RS = 42
D = {}
pasos = sys.argv[1:] or ["figuras", "notebook", "propios"]

# ------------------------------------------------------------ 0. base
D["heart_n"], D["heart_p"] = int(len(F.Xh)), int(F.Xh.shape[1])
D["heart_si"] = int(F.yh.sum())
D["heart_tr"], D["heart_te"] = int(len(F.Xh_tr)), int(len(F.Xh_te))
D["heart_cols_oh"] = int(F.H_tr.shape[1])
D["ames_n"], D["ames_p"] = int(len(F.Xa)), int(F.Xa.shape[1])
D["ames_tr"], D["ames_te"] = int(len(F.Xa_tr)), int(len(F.Xa_te))
D["ames_fit"], D["ames_val"] = int(len(F.Xa_fit)), int(len(F.Xa_val))
D["ames_mediana"] = float(F.ya.median())

if "figuras" in pasos:
    for k in ["bootstrap", "varianza", "max_features", "curvas", "importancia", "oob_cv", "imagenes", "comparacion"]:
        t0 = perf_counter()
        getattr(F, k + "_fig")()
        print(k, round(perf_counter() - t0, 1), "s", flush=True)
    D["fig"] = json.loads(json.dumps(F.NUM, default=float))


# ------------------------------------------------------------ 2. notebook del profesor (replica de las celdas 13-15, 18, 33-35)
def prep_prof(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)], remainder="drop")


if "notebook" in pasos:
    t0 = perf_counter()
    cat_h = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Thal"]
    num_h = [c for c in F.Xh.columns if c not in cat_h]
    ph = prep_prof(num_h, cat_h)
    bag = Pipeline([("prep", ph), ("model", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), n_estimators=120,
                                                              bootstrap=True, oob_score=True, random_state=RS, n_jobs=6))])
    rf = Pipeline([("prep", ph), ("model", RandomForestClassifier(n_estimators=120, max_features="sqrt", bootstrap=True, oob_score=True,
                                                                  class_weight="balanced", random_state=RS, n_jobs=6))])
    rfnw = Pipeline([("prep", ph), ("model", RandomForestClassifier(n_estimators=120, max_features="sqrt", bootstrap=True, oob_score=True,
                                                                    random_state=RS, n_jobs=6))])
    nb = {}
    for nm, m in [("bag", bag), ("rf", rf), ("rf_sin_peso", rfnw)]:
        m.fit(F.Xh_tr, F.yh_tr)
        pr = m.predict(F.Xh_te)
        nb[nm] = dict(acc=accuracy_score(F.yh_te, pr), auc=roc_auc_score(F.yh_te, m.predict_proba(F.Xh_te)[:, 1]),
                      oob=float(m.named_steps["model"].oob_score_))
    D["nb_heart"] = nb
    D["nb_heart_pos_tr"] = round(float(F.yh_tr.mean()), 3)
    D["nb_heart_pos_te"] = round(float(F.yh_te.mean()), 3)
    # GridSearchCV de clasificacion, igual que la celda 18 (FAST_MODE = False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    rf_grid = {"model__n_estimators": [100, 300, 600], "model__max_features": ["sqrt", 0.7], "model__max_depth": [None, 6],
               "model__min_samples_leaf": [1, 3]}
    g = GridSearchCV(clone(rf).set_params(model__n_jobs=1), rf_grid, scoring="roc_auc", cv=cv, n_jobs=12, refit=True).fit(F.Xh_tr, F.yh_tr)
    D["nb_heart_grid_rf"] = dict(best=float(g.best_score_), params={k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                                                                   for k, v in g.best_params_.items()},
                                 n_comb=int(len(g.cv_results_["params"])),
                                 base_cv=float(g.cv_results_["mean_test_score"][[i for i, p in enumerate(g.cv_results_["params"])
                                                                                if p == {"model__n_estimators": 100, "model__max_features": "sqrt",
                                                                                         "model__max_depth": None, "model__min_samples_leaf": 1}][0]]),
                                 test_auc=float(roc_auc_score(F.yh_te, g.best_estimator_.predict_proba(F.Xh_te)[:, 1])))
    print("notebook heart", round(perf_counter() - t0, 1), "s", flush=True)
    # Ames: modelos base del notebook (BaggingRegressor vs RandomForestRegressor(max_features=1.0)) + el mismo RF con 1/3
    t0 = perf_counter()
    cat_a = F.Xa.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    num_a = F.Xa.select_dtypes(include=np.number).columns.tolist()
    pa = prep_prof(num_a, cat_a)
    na = {}
    for nm, mod in [("bag", BaggingRegressor(estimator=DecisionTreeRegressor(random_state=RS), n_estimators=120, bootstrap=True,
                                             oob_score=True, random_state=RS, n_jobs=6)),
                    ("rf10", RandomForestRegressor(n_estimators=120, max_features=1.0, bootstrap=True, oob_score=True, random_state=RS, n_jobs=6)),
                    ("rf13", RandomForestRegressor(n_estimators=120, max_features=1 / 3, bootstrap=True, oob_score=True, random_state=RS, n_jobs=6))]:
        m = Pipeline([("prep", pa), ("model", mod)]).fit(F.Xa_tr, F.ya_tr)
        p = m.predict(F.Xa_te)
        na[nm] = dict(rmse=float(np.sqrt(mean_squared_error(F.ya_te, p))), mae=float(mean_absolute_error(F.ya_te, p)),
                      r2=float(r2_score(F.ya_te, p)), oob_r2=float(m.named_steps["model"].oob_score_))
    D["nb_ames"] = na
    print("notebook ames", round(perf_counter() - t0, 1), "s", flush=True)

# ------------------------------------------------------------ 3. calculos propios para los ejemplos del video
if "propios" in pasos:
    t0 = perf_counter()
    # (a) un arbol suelto vs el promedio de B arboles bootstrap (Heart, test de 76)
    rng = np.random.default_rng(RS)
    n = len(F.H_tr)
    probs = []
    for b in range(100):
        idx = rng.integers(0, n, n)
        t = DecisionTreeClassifier(random_state=b).fit(F.H_tr[idx], F.yh_tr.values[idx])
        probs.append(t.predict_proba(F.H_te)[:, 1])
    P = np.array(probs)
    auc1 = [roc_auc_score(F.yh_te, P[b]) for b in range(100)]
    D["prop_auc_un_arbol_media"] = float(np.mean(auc1))
    D["prop_auc_un_arbol_min"] = float(np.min(auc1))
    D["prop_auc_un_arbol_max"] = float(np.max(auc1))
    D["prop_auc_promedio"] = {B: float(roc_auc_score(F.yh_te, P[:B].mean(axis=0))) for B in (1, 5, 10, 25, 50, 100)}
    # (b) bosque con hojas de 1: exactitud en train vs test (Heart)
    rf = RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1, oob_score=True).fit(F.H_tr, F.yh_tr)
    D["prop_rf_train_acc"] = float(accuracy_score(F.yh_tr, rf.predict(F.H_tr)))
    D["prop_rf_test_acc"] = float(accuracy_score(F.yh_te, rf.predict(F.H_te)))
    D["prop_rf_oob_acc"] = float(rf.oob_score_)
    # (c) no extrapola: Ames entrenado con casas de Gr_Liv_Area < 1800 pies2, prediccion en las > 2400
    Xa_t = F.prep(F.Xa_tr).fit(F.Xa_tr)
    chica = (F.Xa_tr["Gr_Liv_Area"] < 1800).values
    grande = (F.Xa_tr["Gr_Liv_Area"] > 2400).values
    At = Xa_t.transform(F.Xa_tr)
    rfx = RandomForestRegressor(n_estimators=200, max_features=1 / 3, random_state=RS, n_jobs=-1).fit(At[chica], F.ya_tr.values[chica])
    pg = rfx.predict(At[grande])
    D["prop_extrap"] = dict(max_y_entrenado=float(F.ya_tr.values[chica].max()), max_pred_grandes=float(pg.max()),
                            media_real_grandes=float(F.ya_tr.values[grande].mean()), media_pred_grandes=float(pg.mean()),
                            n_grandes=int(grande.sum()), n_chicas=int(chica.sum()))
    print("propios", round(perf_counter() - t0, 1), "s", flush=True)

if "extra" in pasos:
    # por arbol (random subspace) vs por nodo (Random Forest), misma fraccion 10 % de las columnas, Ames validacion
    t0 = perf_counter()
    sub = BaggingRegressor(DecisionTreeRegressor(), n_estimators=300, max_features=0.10, n_jobs=-1, random_state=RS).fit(F.A_fit, F.yf)
    nodo = RandomForestRegressor(n_estimators=300, max_features=0.10, n_jobs=-1, random_state=RS).fit(F.A_fit, F.yf)
    D["extra_por_arbol_10"] = F.rmse(F.yv, sub.predict(F.A_val))
    D["extra_por_nodo_10"] = F.rmse(F.yv, nodo.predict(F.A_val))
    print("extra", round(perf_counter() - t0, 1), "s", flush=True)

out = AQUI / "datos35.json"
if out.exists() and len(pasos) < 3:
    viejo = json.loads(out.read_text(encoding="utf-8"))
    viejo.update(D)
    D = viejo
out.write_text(json.dumps(D, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("OK", out)
