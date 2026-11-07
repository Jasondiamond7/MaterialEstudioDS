"""Numeros reales para los videos de la Clase 34 (venv de Clase 35: pandas 3 + sklearn 1.9 + openpyxl).

Reutiliza la base limpia que arma figuras_c34.py (Base de datos Churn.xlsx de la Clase 33, limpiada igual que en clase)
y replica, celda por celda, el notebook de la clase (clase34_explicado.ipynb, que no trae salidas porque carga un
churn_telco.csv que no existe). Escribe datos34.json junto a este archivo.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
sys.path.insert(0, str(FUENTE))
import figuras_c34 as F  # noqa: E402  (solo se usan X e y; no se llaman funciones de figuras)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

X, y = F.X.copy(), F.y.copy()
D = {}
r4 = lambda v: round(float(v), 4)  # noqa: E731


def gini(v):
    if len(v) == 0:
        return 0.0
    p = v.mean()
    return 1 - p ** 2 - (1 - p) ** 2


def ent(p):
    return 0.0 if p in (0, 1) else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


# ============================================================ 1. base
yv = y.values
D["n"], D["p"] = len(X), X.shape[1]
D["cols"] = list(X.columns)
D["n_fuga"] = int(y.sum())
D["n_queda"] = D["n"] - D["n_fuga"]
D["tasa"] = r4(y.mean())
D["gini_raiz"] = r4(gini(yv))
D["ent_raiz"] = r4(ent(y.mean()))
D["nunique"] = {c: int(v) for c, v in X.nunique().items()}

# ============================================================ 2. arbol de profundidad 3 (diapositivas 8-9)
arb = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=60, random_state=0).fit(X, y)
D["arbol3_hojas"], D["arbol3_prof"] = int(arb.get_n_leaves()), int(arb.get_depth())
D["arbol3_texto"] = export_text(arb, feature_names=list(X.columns), decimals=1)
t = arb.tree_
nodos = []
for i in range(t.node_count):
    v = t.value[i][0]
    nodos.append(dict(id=i, n=int(t.n_node_samples[i]), izq=int(t.children_left[i]), der=int(t.children_right[i]),
                      var=(X.columns[t.feature[i]] if t.children_left[i] != -1 else None),
                      umbral=(round(float(t.threshold[i]), 2) if t.children_left[i] != -1 else None)))
hoja = arb.apply(X)
for nd in nodos:                                   # tasa real (value de sklearn 1.9 son fracciones)
    pass
# tasa y fugas por nodo: recorrer con mascaras
def mascara(i, m, out):
    out[i] = m
    nd = nodos[i]
    if nd["izq"] != -1:
        xv = X[nd["var"]].values
        mascara(nd["izq"], m & (xv <= t.threshold[i]), out)
        mascara(nd["der"], m & (xv > t.threshold[i]), out)


MS = {}
mascara(0, np.ones(len(X), bool), MS)
for nd in nodos:
    m = MS[nd["id"]]
    nd["fugas"] = int(yv[m].sum())
    nd["tasa"] = r4(yv[m].mean())
    nd["gini"] = r4(gini(yv[m]))
    assert int(m.sum()) == nd["n"]
D["arbol3_nodos"] = nodos


def ruta(i, acc, out):
    nd = nodos[i]
    if nd["izq"] == -1:
        out.append(dict(regla=acc, n=nd["n"], fugas=nd["fugas"], tasa=nd["tasa"], pct=round(nd["n"] / D["n"] * 100, 1),
                        lift=round(nd["tasa"] / D["tasa"], 2), id=i))
        return
    ruta(nd["izq"], acc + [f"{nd['var']} <= {nd['umbral']}"], out)
    ruta(nd["der"], acc + [f"{nd['var']} > {nd['umbral']}"], out)


hojas = []
ruta(0, [], hojas)
D["hojas3"] = sorted(hojas, key=lambda h: -h["tasa"])
D["hojas3_suma_n"] = sum(h["n"] for h in hojas)

# ============================================================ 3. barrido de la raiz (celda 9)
G0 = gini(yv)
n = len(yv)
filas, combos = [], 0
for c in X.columns:
    v = X[c].values
    cand = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 60)))
    combos += len(cand)
    mejor = None
    for u in cand:
        izq = v <= u
        der = ~izq
        if izq.sum() < 60 or der.sum() < 60:
            continue
        gp = izq.mean() * gini(yv[izq]) + der.mean() * gini(yv[der])
        g = G0 - gp
        if mejor is None or g > mejor[2]:
            mejor = (float(u), gp, g, int(izq.sum()), int(der.sum()))
    if mejor:
        filas.append(dict(var=c, umbral=round(mejor[0], 2), gp=r4(mejor[1]), gan=r4(mejor[2]), n_izq=mejor[3], n_der=mejor[4]))
D["barrido"] = sorted(filas, key=lambda f: -f["gan"])
D["combos"] = combos

# ============================================================ 4. el corte de reclamos a mano (celdas 10-11, diap. 13)
def corte(mask):
    a, b = yv[mask], yv[~mask]
    pa, pb = a.mean(), b.mean()
    wa, wb = mask.mean(), 1 - mask.mean()
    gp = wa * gini(a) + wb * gini(b)
    ep = wa * ent(pa) + wb * ent(pb)
    return dict(n_si=int(mask.sum()), f_si=int(a.sum()), p_si=r4(pa), g_si=r4(gini(a)), e_si=r4(ent(pa)),
                n_no=int((~mask).sum()), f_no=int(b.sum()), p_no=r4(pb), g_no=r4(gini(b)), e_no=r4(ent(pb)),
                w_si=r4(wa), w_no=r4(wb), gp=r4(gp), gan=r4(G0 - gp), ep=r4(ep), gan_e=r4(ent(y.mean()) - ep))


D["reclamos"] = corte((X.reclamos > 0).values)
D["edad40"] = corte((X.edad > 40).values)
# lo que diria la diapositiva 13 con SUS conteos (211 y 284)
pa, pb = 211 / 278, 284 / 2872
gpa = 278 / 3150 * (1 - pa ** 2 - (1 - pa) ** 2) + 2872 / 3150 * (1 - pb ** 2 - (1 - pb) ** 2)
D["diap13"] = dict(g_si=r4(1 - pa ** 2 - (1 - pa) ** 2), g_no=r4(1 - pb ** 2 - (1 - pb) ** 2), gp=r4(gpa), gan=r4(0.265 - gpa),
                   gan_exacta=r4(G0 - gpa))
pa, pb = 240 / 1559, 255 / 1591
gpb = 1559 / 3150 * (1 - pa ** 2 - (1 - pa) ** 2) + 1591 / 3150 * (1 - pb ** 2 - (1 - pb) ** 2)
D["diap13_B"] = dict(gp=r4(gpb), gan=r4(G0 - gpb))

# segundo nivel: el barrido completo dentro de cada hijo de la raiz (con min 60 por lado)
def mejor_en(mask):
    sub, ys = X[mask], yv[mask]
    g0 = gini(ys)
    res = []
    for c in X.columns:
        v = sub[c].values
        vals = np.unique(v)
        cand = (vals[:-1] + vals[1:]) / 2
        best = None
        for u in cand:
            iz = v <= u
            if iz.sum() < 60 or (~iz).sum() < 60:
                continue
            gp = iz.mean() * gini(ys[iz]) + (~iz).mean() * gini(ys[~iz])
            if best is None or g0 - gp > best[1]:
                best = (float(u), g0 - gp)
        if best:
            res.append((c, round(best[0], 2), r4(best[1])))
    return sorted(res, key=lambda r: -r[2])[:4], r4(g0)


D["nivel2_reclamo"] = mejor_en((X.reclamos > 0).values)
D["nivel2_sin_reclamo"] = mejor_en((X.reclamos <= 0).values)

# ============================================================ 5. gini vs entropia (celda 12)
cv = StratifiedKFold(5, shuffle=True, random_state=1)
D["crit"] = {c: r4(cross_val_score(DecisionTreeClassifier(criterion=c, max_depth=5, random_state=0), X, y, cv=cv,
                                   scoring="roc_auc").mean()) for c in ["gini", "entropy"]}

# ============================================================ 6. CV por ronda (celda 14)
s = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=0), X, y, cv=cv, scoring="roc_auc")
D["rondas"] = [r4(a) for a in s]
D["rondas_media"], D["rondas_std"] = r4(s.mean()), r4(s.std())
D["fold_n"], D["fold_train"] = len(X) // 5, len(X) - len(X) // 5
D["fold_tasas"] = [r4(yv[te].mean()) for _tr, te in cv.split(X, y)]

# ============================================================ 7. curva de profundidad (celda 16)
curva = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    acv = cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()
    m.fit(X, y)
    atr = roc_auc_score(y, m.predict_proba(X)[:, 1])
    curva.append(dict(d=(d or 0), tr=r4(atr), va=r4(acv), hojas=int(m.get_n_leaves()), prof_real=int(m.get_depth())))
D["curva"] = curva

# ============================================================ 8. poda (celdas 18-19)
base = DecisionTreeClassifier(random_state=0)
alphas = base.cost_complexity_pruning_path(X, y).ccp_alphas[:-1]
D["n_alphas"], D["alpha_min"], D["alpha_max"] = len(alphas), float(alphas.min()), float(alphas.max())
grid = GridSearchCV(DecisionTreeClassifier(random_state=0), {"ccp_alpha": alphas[::5]}, cv=cv, scoring="roc_auc",
                    n_jobs=-1).fit(X, y)
D["grid_alpha"], D["grid_auc"] = float(grid.best_params_["ccp_alpha"]), r4(grid.best_score_)
D["grid_hojas"] = int(grid.best_estimator_.get_n_leaves())
D["grid_n_cand"] = len(alphas[::5])
tabla = []
for a in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
    tt = DecisionTreeClassifier(random_state=0, ccp_alpha=a)
    au = cross_val_score(tt, X, y, cv=cv, scoring="roc_auc").mean()
    tt.fit(X, y)
    tabla.append(dict(alpha=a, hojas=int(tt.get_n_leaves()), auc=r4(au)))
D["poda"] = tabla

# ============================================================ 9. importancia (celda 21)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0, scoring="roc_auc")
D["auc_test_rf"] = r4(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
imp = pd.DataFrame({"imp": rf.feature_importances_, "perm": pi.importances_mean, "u": X.nunique()}, index=X.columns)
imp = imp.sort_values("imp", ascending=False)
D["imp"] = [dict(var=i, imp=r4(r.imp), perm=r4(r.perm), u=int(r.u)) for i, r in imp.iterrows()]
D["n_test"] = len(Xte)

# ============================================================ 10. transformaciones monotonas (celda 24)
X2 = X.copy()
X2["segundos"] = np.log1p(X2["segundos"].clip(lower=0))
X2["clv"] = np.sqrt(X2["clv"])
arb5 = DecisionTreeClassifier(max_depth=5, random_state=0)
logi = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
auc = lambda m, d: float(cross_val_score(m, d, y, cv=cv, scoring="roc_auc").mean())  # noqa: E731
D["mono"] = dict(arbol=[r4(auc(arb5, X)), r4(auc(arb5, X2))], log=[r4(auc(logi, X)), r4(auc(logi, X2))])
D["mono"]["arbol_dif"] = auc(arb5, X2) - auc(arb5, X)
# umbral de la raiz en escala original vs log (mismo corte, otro numero): segundos en el nivel 2
t1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0).fit(X, y)
t2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0).fit(X2, y)
D["mono_umbrales"] = [[X.columns[f] if f >= 0 else None, round(float(a), 3), round(float(b), 3)]
                      for f, a, b in zip(t1.tree_.feature, t1.tree_.threshold, t2.tree_.threshold) if f >= 0]
D["mono_hojas_iguales"] = bool((t1.apply(X) == t2.apply(X2)).all())

# ============================================================ 11. inestabilidad (celda 26)
inest = []
for sem in range(6):
    Xs, _, ys, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sem)
    tt = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xs, ys)
    e = tt.tree_
    iz = e.children_left[0]
    de = e.children_right[0]
    inest.append(dict(sem=sem, raiz=X.columns[e.feature[0]], izq=X.columns[e.feature[iz]], umbral=round(float(e.threshold[iz]), 1),
                      der=X.columns[e.feature[de]], umbral_der=round(float(e.threshold[de]), 1)))
D["inest"] = inest

# ============================================================ 12. bosque (celdas 28, 30, 31)
D["rf_n"] = [[nt, r4(cross_val_score(RandomForestClassifier(n_estimators=nt, random_state=0, n_jobs=-1), X, y, cv=cv,
                                     scoring="roc_auc").mean())] for nt in [1, 5, 25, 100, 300]]
cfg = [("solo_bootstrap", dict(max_features=None)), ("bootstrap_sqrt", dict(max_features="sqrt")),
       ("sin_bootstrap_sqrt", dict(bootstrap=False, max_features="sqrt"))]
D["rf_cfg"] = {k: r4(cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, **kw), X, y, cv=cv,
                                     scoring="roc_auc").mean()) for k, kw in cfg}
D["sqrt_p"] = int(np.sqrt(X.shape[1]))
mods = {"arbol3": DecisionTreeClassifier(max_depth=3, random_state=0), "arbol5": DecisionTreeClassifier(max_depth=5, random_state=0),
        "podado": grid.best_estimator_, "logistica": logi,
        "rf300": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)}
D["final"] = {k: r4(cross_val_score(v, X, y, cv=cv, scoring="roc_auc").mean()) for k, v in mods.items()}
D["oob_teorico"] = r4((1 - 1 / D["n"]) ** D["n"])

# ============================================================ 13. extras para los videos
D["auc_reclamos_sola"] = r4(roc_auc_score(y, X.reclamos))
Xs_ = X.drop(columns="reclamos")
D["sin_reclamos"] = dict(arbol5=r4(cross_val_score(arb5, Xs_, y, cv=cv, scoring="roc_auc").mean()),
                         logistica=r4(cross_val_score(logi, Xs_, y, cv=cv, scoring="roc_auc").mean()))
g5 = GridSearchCV(DecisionTreeClassifier(random_state=0), {"ccp_alpha": alphas[::5]}, cv=5, scoring="roc_auc", n_jobs=-1).fit(X, y)
D["diap19_cv5"] = dict(alpha=float(g5.best_params_["ccp_alpha"]), auc=r4(g5.best_score_), hojas=int(g5.best_estimator_.get_n_leaves()))
D["tabla_imp"] = {f"{a}/{b}": [r4(1 - (a / 100) ** 2 - (b / 100) ** 2), r4(ent(b / 100))] for a, b in [(100, 0), (84, 16), (50, 50), (6, 94)]}

(AQUI / "datos34.json").write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in D.items() if k not in ("arbol3_texto", "arbol3_nodos")}, ensure_ascii=False)[:6000])
print(D["arbol3_texto"])
