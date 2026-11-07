"""Numeros reales para los videos de la Clase 36 (venv de Clase 35: pandas 3 + sklearn 1.9 + xgboost + lightgbm + openpyxl).

1) Importa figuras_c36.py (misma base y particiones que la guia) con `save` anulado: no se sobrescribe ninguna figura.
   Ejecuta sus calculos (residuos secuenciales, bagging vs boosting, AdaBoost a mano, learning rate, early stopping, comparacion).
2) Replica el notebook del profesor (Ejemplo Boosting.ipynb) tal como se entrego (FAST_MODE = True): modelos base de Heart y
   Ames, GridSearchCV de AdaBoost, RandomizedSearchCV de XGBoost, GridSearchCV de Gradient Boosting y RandomizedSearchCV de LightGBM.
3) Comprobaciones de API para las alertas del material (LightGBM subsample sin subsample_freq, tipo de importancia, tol de GB).
Escribe datos36.json junto a este archivo.
"""
import inspect
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
import figuras_c36 as F  # noqa: E402

F.save = lambda fig, name: plt.close(fig)          # no tocar Guias/_fuente/img
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,  # noqa: E402
                              GradientBoostingRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402
import lightgbm  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402

RS = 42
D = {"versiones": dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, lightgbm=lightgbm.__version__,
                       pandas=pd.__version__)}
pasos = sys.argv[1:] or ["figuras", "notebook", "api"]

D["heart_n"], D["heart_p"] = int(len(F.heart)), int(F.heart.shape[1] - 1)
D["heart_si"] = int((F.heart["AHD"] == "Yes").sum())
D["heart_tr"], D["heart_te"] = int(len(F.Xh_tr)), int(len(F.Xh_te))
D["ames_n"] = int(len(F.ames))
D["ames_tr"], D["ames_te"] = int(len(F.Xa_tr)), int(len(F.Xa_te))
D["ames_fit"], D["ames_val"] = int(len(F.Xa_fit)), int(len(F.Xa_val))

if "figuras" in pasos:
    for k in ["secuencial", "bagging_boosting", "adaboost_fig", "lr_nest", "xgb_lgbm", "comparacion"]:
        t0 = perf_counter()
        getattr(F, k)()
        print(k, round(perf_counter() - t0, 1), "s", flush=True)
    D["fig"] = json.loads(json.dumps(F.NUM, default=float))


# ------------------------------------------------------------ 2. notebook del profesor, FAST_MODE = True (como se entrego)
def make_preprocessor(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)], remainder="drop")


if "notebook" in pasos:
    t0 = perf_counter()
    heart = pd.read_excel(F.CLASE35 / "Heart.xlsx")
    Xh, yh = heart.drop(columns="AHD"), heart["AHD"].map({"No": 0, "Yes": 1})
    Xtr, Xte, ytr, yte = train_test_split(Xh, yh, test_size=0.25, stratify=yh, random_state=RS)
    D["nb_heart_pos_tr"], D["nb_heart_pos_te"] = round(float(ytr.mean()), 3), round(float(yte.mean()), 3)
    hp = make_preprocessor(Xtr)
    est = {
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", n_jobs=-1, random_state=RS),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, max_depth=-1, subsample=0.8,
                                   colsample_bytree=0.8, verbosity=-1, n_jobs=-1, random_state=RS),
    }
    pipes = {k: Pipeline([("preprocessor", clone(hp)), ("model", m)]) for k, m in est.items()}
    nbh = {}
    for k, p in pipes.items():
        m = clone(p).fit(Xtr, ytr)
        pr, prt = m.predict(Xte), m.predict(Xtr)
        nbh[k] = dict(acc_train=accuracy_score(ytr, prt), acc_test=accuracy_score(yte, pr), recall=recall_score(yte, pr),
                      f1=f1_score(yte, pr), auc=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                      aciertos=int((pr == yte.values).sum()))
    D["nb_heart"] = nbh
    cvc = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
    ada = GridSearchCV(pipes["AdaBoost"], {"model__n_estimators": [50, 150], "model__learning_rate": [0.05, 0.20]},
                       scoring="roc_auc", cv=cvc, n_jobs=-1, return_train_score=True).fit(Xtr, ytr)
    xgbp = Pipeline([("preprocessor", clone(hp)), ("model", XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=RS))])
    xgr = RandomizedSearchCV(xgbp, {"model__n_estimators": [80, 150, 250, 400], "model__learning_rate": [0.02, 0.05, 0.10, 0.20],
                                    "model__max_depth": [1, 2, 3, 4], "model__subsample": [0.65, 0.80, 1.0],
                                    "model__colsample_bytree": [0.65, 0.80, 1.0], "model__min_child_weight": [1, 3, 5]},
                             n_iter=6, scoring="roc_auc", cv=cvc, random_state=RS, n_jobs=-1, return_train_score=True).fit(Xtr, ytr)

    def fin_c(m):
        pr = m.predict(Xte)
        return dict(acc_test=accuracy_score(yte, pr), recall=recall_score(yte, pr), f1=f1_score(yte, pr),
                    auc=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    # la configuracion base de AdaBoost (100 arboles, 0.05), evaluada con la MISMA validacion cruzada que la busqueda
    from sklearn.model_selection import cross_val_score
    base_cv = float(cross_val_score(pipes["AdaBoost"], Xtr, ytr, cv=cvc, scoring="roc_auc").mean())
    D["nb_heart_tuned"] = dict(ada=dict(params={k: v for k, v in ada.best_params_.items()}, cv=float(ada.best_score_), **fin_c(ada.best_estimator_),
                                        base_cv=base_cv),
                               xgb=dict(params={k: v for k, v in xgr.best_params_.items()}, cv=float(xgr.best_score_), **fin_c(xgr.best_estimator_)))
    print("notebook heart", round(perf_counter() - t0, 1), "s", flush=True)

    t0 = perf_counter()
    ames = pd.read_csv(F.CLASE35 / "ames_housing.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    D["nb_ames_cols"] = int(ames.shape[1])
    D["nb_ames_tiene_latlon"] = bool({"Longitude", "Latitude"} <= set(ames.columns))
    Xa, ya = ames.drop(columns="Sale_Price"), ames["Sale_Price"]
    Atr, Ate, yatr, yate = train_test_split(Xa, ya, test_size=0.20, random_state=RS)
    D["nb_ames_mediana_tr"], D["nb_ames_mediana_te"] = float(yatr.median()), float(yate.median())
    ap = make_preprocessor(Atr)
    rest = {
        "AdaBoost": AdaBoostRegressor(n_estimators=150, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, loss="huber", random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                objective="reg:squarederror", n_jobs=-1, random_state=RS),
        "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                  verbosity=-1, n_jobs=-1, random_state=RS),
    }
    rp = {k: Pipeline([("preprocessor", clone(ap)), ("model", m)]) for k, m in rest.items()}
    nba = {}
    for k, p in rp.items():
        t = perf_counter()
        m = clone(p).fit(Atr, yatr)
        dt = perf_counter() - t
        pr, prt = m.predict(Ate), m.predict(Atr)
        nba[k] = dict(rmse_train=float(mean_squared_error(yatr, prt) ** 0.5), rmse_test=float(mean_squared_error(yate, pr) ** 0.5),
                      mae=float(mean_absolute_error(yate, pr)), r2=float(r2_score(yate, pr)), seg=dt)
    D["nb_ames"] = nba
    # Huber vs cuadratica con la misma configuracion (objetivo vs metrica)
    m2 = Pipeline([("preprocessor", clone(ap)), ("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                                                     random_state=RS))]).fit(Atr, yatr)
    D["nb_ames_gb_cuadratica_rmse"] = float(mean_squared_error(yate, m2.predict(Ate)) ** 0.5)
    cvr = KFold(n_splits=3, shuffle=True, random_state=RS)
    gbr = GridSearchCV(rp["Gradient Boosting"], {"model__n_estimators": [120, 220], "model__learning_rate": [0.05, 0.12],
                                                 "model__max_depth": [2, 3], "model__min_samples_leaf": [5, 10]},
                       scoring="neg_root_mean_squared_error", cv=cvr, n_jobs=-1, return_train_score=True).fit(Atr, yatr)
    lgp = Pipeline([("preprocessor", clone(ap)), ("model", LGBMRegressor(verbosity=-1, n_jobs=1, random_state=RS))])
    lgr = RandomizedSearchCV(lgp, {"model__n_estimators": [120, 220, 350, 500], "model__learning_rate": [0.02, 0.04, 0.07, 0.12],
                                   "model__num_leaves": [12, 20, 31, 48], "model__max_depth": [-1, 3, 5, 7],
                                   "model__min_child_samples": [10, 20, 35, 50], "model__subsample": [0.70, 0.85, 1.0],
                                   "model__colsample_bytree": [0.70, 0.85, 1.0]},
                             n_iter=6, scoring="neg_root_mean_squared_error", cv=cvr, random_state=RS, n_jobs=-1,
                             return_train_score=True).fit(Atr, yatr)

    def fin_r(m):
        pr = m.predict(Ate)
        return dict(rmse_test=float(mean_squared_error(yate, pr) ** 0.5), mae=float(mean_absolute_error(yate, pr)), r2=float(r2_score(yate, pr)))
    D["nb_ames_tuned"] = dict(gb=dict(params={k: v for k, v in gbr.best_params_.items()}, cv=float(-gbr.best_score_), **fin_r(gbr.best_estimator_)),
                              lgbm=dict(params={k: v for k, v in lgr.best_params_.items()}, cv=float(-lgr.best_score_), **fin_r(lgr.best_estimator_)))
    print("notebook ames", round(perf_counter() - t0, 1), "s", flush=True)

# ------------------------------------------------------------ 3. API: alertas del material
if "api" in pasos:
    H_tr = F.prep(F.Xh_tr).fit(F.Xh_tr)
    A = H_tr.transform(F.Xh_tr)
    B = H_tr.transform(F.Xh_te)
    kw = dict(n_estimators=100, learning_rate=0.05, num_leaves=15, colsample_bytree=0.8, verbosity=-1, n_jobs=1, random_state=RS)
    p08 = LGBMClassifier(subsample=0.8, **kw).fit(A, F.yh_tr).predict_proba(B)[:, 1]
    p10 = LGBMClassifier(subsample=1.0, **kw).fit(A, F.yh_tr).predict_proba(B)[:, 1]
    p08f = LGBMClassifier(subsample=0.8, subsample_freq=1, **kw).fit(A, F.yh_tr).predict_proba(B)[:, 1]
    D["api_lgbm_subsample_sin_freq_identico"] = bool(np.allclose(p08, p10))
    D["api_lgbm_subsample_con_freq_cambia"] = bool(not np.allclose(p08f, p10))
    D["api_lgbm_subsample_freq_defecto"] = inspect.signature(LGBMClassifier).parameters["subsample_freq"].default
    D["api_lgbm_importance_defecto"] = inspect.signature(LGBMClassifier).parameters["importance_type"].default
    xm = XGBClassifier(n_estimators=50, max_depth=3, random_state=RS).fit(A, F.yh_tr)
    g = xm.get_booster().get_score(importance_type="gain")
    vec = np.array([g.get(f"f{i}", 0.0) for i in range(A.shape[1])])
    D["api_xgb_importance_es_gain"] = bool(np.allclose(xm.feature_importances_, vec / vec.sum(), atol=1e-6))
    pg = inspect.signature(GradientBoostingRegressor).parameters
    D["api_gb_n_iter_no_change"] = pg["n_iter_no_change"].default
    D["api_gb_tol"] = pg["tol"].default
    gbt = GradientBoostingRegressor(n_estimators=300, random_state=RS).fit(F.A_fit[:, :5], F.ya_fit)
    D["api_gb_arboles_entrenados"] = int(gbt.n_estimators_)
    ab = AdaBoostClassifier(random_state=RS).fit(A, F.yh_tr)
    D["api_ada_estimador_base_depth"] = int(ab.estimator_.max_depth)
    D["api_ada_n_estimators_defecto"] = inspect.signature(AdaBoostClassifier).parameters["n_estimators"].default

# ------------------------------------------------------------ 4. curvas completas (mismo codigo y semillas que figuras_c36)
if "curvas" in pasos:
    t0 = perf_counter()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    rf = RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS).fit(F.A_fit, F.ya_fit)
    cum = np.cumsum([e.predict(F.A_val) for e in rf.estimators_], axis=0) / np.arange(1, 301).reshape(-1, 1)
    rf_curve = [F.rmse(F.ya_val, c) for c in cum]
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RS).fit(F.A_fit, F.ya_fit)
    gb_curve = [F.rmse(F.ya_val, p) for p in gb.staged_predict(F.A_val)]
    C = dict(rf=rf_curve, gb=gb_curve)
    for lr in [1.0, 0.3, 0.1, 0.03]:
        g = GradientBoostingRegressor(n_estimators=800, max_depth=3, learning_rate=lr, random_state=RS).fit(F.A_fit, F.ya_fit)
        C[f"lr{lr}"] = [F.rmse(F.ya_val, p) for p in g.staged_predict(F.A_val)]
    mod = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                       early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=RS)
    mod.fit(F.A_fit, F.ya_fit, eval_set=[(F.A_fit, F.ya_fit), (F.A_val, F.ya_val)], verbose=False)
    ev = mod.evals_result()
    C["xgb_tr"], C["xgb_va"] = ev["validation_0"]["rmse"], ev["validation_1"]["rmse"]
    D["curvas"] = {k: [float(v) for v in vs] for k, vs in C.items()}
    print("curvas", round(perf_counter() - t0, 1), "s", flush=True)

# ------------------------------------------------------------ 5. AdaBoost a mano con 70 pacientes (mismo codigo que adaboost_fig)
if "ada70" in pasos:
    d = F.heart[["Age", "MaxHR", "AHD"]].sample(70, random_state=3).reset_index(drop=True)
    X = d[["Age", "MaxHR"]].values.astype(float)
    y = np.where(d.AHD == "Yes", 1, -1)
    hist = F.adaboost_mano(X, y, 30)
    rondas = []
    for h in hist[:3]:
        st = h["stump"].tree_
        izq = int(np.sign(st.value[1][0][1] - st.value[1][0][0]) or 1)       # clase que predice la hoja izquierda (<= umbral)
        rondas.append(dict(w=[float(x) for x in h["w"]], err=h["err"], alpha=float(h["alpha"]), feature=int(st.feature[0]),
                           thr=float(st.threshold[0]), mal=[int(i) for i in np.where(h["pred"] != y)[0]],
                           izq=int(h["stump"].predict([[0, 0] if st.feature[0] == 0 else [0, 0]])[0])))
    Ftr = sum(h["alpha"] * h["stump"].predict(X) for h in hist)
    D["ada70"] = dict(X=X.tolist(), y=y.tolist(), rondas=rondas, aciertos30=int((np.sign(Ftr) == y).sum()))
    print("ada70", [(r["feature"], round(r["thr"], 1), len(r["mal"]), round(r["err"], 4)) for r in rondas])

out = AQUI / "datos36.json"
if out.exists() and len(pasos) < 3:
    viejo = json.loads(out.read_text(encoding="utf-8"))
    viejo.update(D)
    D = viejo
out.write_text(json.dumps(D, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("OK", out)
