"""Numeros reales para los videos de la Clase 33 (venv de Clase 35: pandas + sklearn + openpyxl).

Reutiliza la base cruda que carga figuras_c33.py (Base de datos Churn.xlsx) y replica, celda por celda, el notebook
de la clase (clase 33- codigo python.ipynb). Escribe datos33.json junto a este archivo.
"""
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
sys.path.insert(0, str(FUENTE))
import figuras_c33 as F  # noqa: E402  (solo se usa su DataFrame crudo F.D; no se llaman funciones de figuras)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold, f_classif, mutual_info_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

D = {}
r4 = lambda v: round(float(v), 4)  # noqa: E731

# ============================================================ 1. crudo
raw = F.D.copy()
D["n"], D["cols"] = len(raw), raw.shape[1]
D["n_fuga"] = int(raw.fuga.sum())
D["tasa"] = round(float(raw.fuga.mean()) * 100, 2)
D["acc_nadie"] = round(100 - D["tasa"], 1)
neg = {}
for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
    k = int((raw[c] < 0).sum())
    neg[c] = [k, float(raw[c].min())]
D["neg"] = neg
D["edad"] = dict(neg=int((raw.edad < 0).sum()), min=float(raw.edad.min()), menores=int(((raw.edad >= 0) & (raw.edad < 15)).sum()),
                 mas100=int((raw.edad > 100).sum()), max=float(raw.edad.max()), total=int(((raw.edad < 15) | (raw.edad > 100)).sum()))
D["minmax"] = {c: [float(raw[c].min()), float(raw[c].max())] for c in raw.columns}
D["imposibles"] = sum(v[0] for v in neg.values()) + D["edad"]["neg"] + D["edad"]["mas100"]
D["nulos"] = {c: int(v) for c, v in raw.isna().sum().items() if v}
D["nulos_tot"] = int(raw.isna().sum().sum())
D["dup"] = int(raw.duplicated().sum())
D["seg_media_raw"], D["seg_mediana_raw"] = round(float(raw.segundos.mean()), 2), round(float(raw.segundos.median()), 1)
D["pct_menores"] = round(D["edad"]["menores"] / D["n"] * 100, 1)
D["corr_raw"] = {f"{a}|{b}": round(float(raw[a].corr(raw[b])), 3)
                 for a, b in [("segundos", "frec_llamadas"), ("frec_sms", "clv"), ("frec_llamadas", "n_contactos")]}

# ============================================================ 2. limpieza (celda 10)
df = raw.copy()
for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
    df.loc[df[c] < 0, c] = np.nan
df.loc[(df.edad < 15) | (df.edad > 100), "edad"] = np.nan
D["nan_tras_marcar"] = {c: int(v) for c, v in df.isna().sum().items() if v}
D["medianas_imp"] = {c: float(df[c].median()) for c in df.columns if df[c].isna().any()}
for c in df.columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
D["seg_mediana_limpia"] = float(df.segundos.median())
D["corr_limpio"] = {f"{a}|{b}": round(float(df[a].corr(df[b])), 3)
                    for a, b in [("segundos", "frec_llamadas"), ("frec_sms", "clv"), ("frec_llamadas", "n_contactos")]}

# ============================================================ 3. construccion (celda 12)
e = 1e-6
d = df.copy()
pares = [("duracion_prom_llamada", "segundos", "frec_llamadas"), ("sms_por_llamada", "frec_sms", "frec_llamadas"),
         ("intensidad_contacto", "n_contactos", "frec_llamadas"), ("segundos_por_contacto", "segundos", "n_contactos"),
         ("monto_por_llamada", "monto", "frec_llamadas"), ("clv_por_llamada", "clv", "frec_llamadas"),
         ("clv_por_segundo", "clv", "segundos"), ("sms_por_antiguedad", "frec_sms", "antiguedad"),
         ("clv_por_antiguedad", "clv", "antiguedad"), ("llamadas_por_mes", "frec_llamadas", "antiguedad"),
         ("contactos_por_mes", "n_contactos", "antiguedad"), ("fallas_por_antiguedad", "numero_fallas", "antiguedad"),
         ("antiguedad_rel_edad", "antiguedad", "edad")]
for nom, a, b in pares:
    d[nom] = d[a] / (d[b] + e)
d["reclamo_x_fallas"] = d.reclamos * d.numero_fallas
d["reclamo_x_antig"] = d.reclamos * d.antiguedad
d["fallas_x_contactos"] = d.numero_fallas * d.n_contactos
for c in ["segundos", "clv", "frec_sms", "frec_llamadas", "n_contactos"]:
    d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))
med, p75 = d.segundos.median(), d.clv.quantile(0.75)
d["tiene_reclamo"] = (d.reclamos > 0).astype(int)
d["sin_sms"] = (d.frec_sms == 0).astype(int)
d["uso_alto"] = (d.segundos > med).astype(int)
d["clv_alto"] = (d.clv > p75).astype(int)
d["cliente_nuevo"] = (d.antiguedad < 12).astype(int)
for col in ["edad", "antiguedad", "clv"]:
    q = pd.qcut(d[col], 4, labels=False, duplicates="drop")
    for k in sorted(q.dropna().unique()):
        d[f"{col}_q{int(k) + 1}"] = (q == k).astype(int)
d["plan_2"] = (d.tipo_plan == 2).astype(int)
d["estado_2"] = (d.estado == 2).astype(int)
d = d.drop(columns=["tipo_plan", "estado"])
X, y = d.drop(columns="fuga"), d["fuga"]
D["clv_stats"] = dict(max=float(d.clv.max()), med=float(d.clv.median()), log_max=round(float(np.log1p(d.clv.max())), 3),
                     log_med=round(float(np.log1p(d.clv.median())), 3), p75=round(float(p75), 2), seg_med=float(med))
D["r_reclamos_tiene"] = round(float(d.reclamos.corr(d.tiene_reclamo)), 3)
D["reclamos_valores"] = sorted(float(v) for v in d.reclamos.unique())
D["p"] = X.shape[1]
D["familias"] = dict(originales=10, ratios=len(pares), interacciones=3, logs=5, binarias=5,
                     cuartiles=sum(c.endswith(("_q1", "_q2", "_q3", "_q4")) for c in X.columns), onehot=2)
D["inf"] = int(np.isinf(X.values).sum())
D["max_llamadas_por_mes"] = float(d.llamadas_por_mes.max())
D["antig_cero"] = int((df.antiguedad == 0).sum())

originales = ["numero_fallas", "reclamos", "antiguedad", "monto", "segundos", "frec_llamadas", "frec_sms", "n_contactos",
              "edad", "clv", "plan_2", "estado_2"]
ct = d[originales + ["fuga"]].corr()["fuga"].drop("fuga")
D["corr_fuga"] = {k: round(float(v), 3) for k, v in ct.sort_values(key=abs, ascending=False).items()}

# ============================================================ 4. numero_fallas en conjunto (celda 20)
cv = StratifiedKFold(5, shuffle=True, random_state=1)


def auc(cols):
    m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
    return float(cross_val_score(m, X[cols], y, cv=cv, scoring="roc_auc").mean())


base = ["reclamos", "segundos", "frec_llamadas", "estado_2", "clv"]
D["fallas"] = dict(sola=r4(auc(["numero_fallas"])), base=r4(auc(base)), base_f=r4(auc(base + ["numero_fallas"])),
                   base_f_m=r4(auc(base + ["numero_fallas", "monto"])))

# ============================================================ 5. XOR (celda 22)
rng = np.random.default_rng(7)
x1, x2 = rng.normal(size=5000), rng.normal(size=5000)
t = ((x1 > 0) ^ (x2 > 0)).astype(int)
Z, Zi = np.c_[x1, x2], np.c_[x1, x2, x1 * x2]
Fx, px = f_classif(Z, t)
D["xor"] = dict(F1=round(float(Fx[0]), 3), p1=round(float(px[0]), 3), F2=round(float(Fx[1]), 3), p2=round(float(px[1]), 3),
                log=round(float(cross_val_score(LogisticRegression(max_iter=2000), Z, t, cv=5, scoring="roc_auc").mean()), 3),
                inter=round(float(cross_val_score(LogisticRegression(max_iter=2000), Zi, t, cv=5, scoring="roc_auc").mean()), 3),
                rf=round(float(cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0), Z, t, cv=5,
                                               scoring="roc_auc").mean()), 3),
                tasa=round(float(t.mean()), 3), n=5000)
# nube de puntos para el video (primeros 220 puntos del mismo XOR, recortados a [-2.6, 2.6])
k_ = np.abs(Z[:400]).max(axis=1) < 2.6
D["xor_pts"] = dict(x=[round(float(v), 2) for v in x1[:400][k_][:220]], y=[round(float(v), 2) for v in x2[:400][k_][:220]],
                    c=[int(v) for v in t[:400][k_][:220]])

# ============================================================ 6. filtros (celdas 27-28)
vt = VarianceThreshold(threshold=0.01).fit(X)
D["vt_fuera"] = list(X.columns[~vt.get_support()])
mi = pd.Series(mutual_info_classif(X, y, random_state=0), index=X.columns)
Fv, pv = f_classif(X, y)
rank = pd.DataFrame({"MI": mi, "F": Fv, "p": pv})
top12 = rank.sort_values("MI", ascending=False).head(12)
D["rank12"] = [[i, r4(r.MI), round(float(r.F), 2), r4(r.p)] for i, r in top12.iterrows()]
D["sospechosas"] = {c: [r4(rank.loc[c, "MI"]), round(float(rank.loc[c, "F"]), 2), r4(rank.loc[c, "p"])]
                    for c in ["numero_fallas", "edad", "antiguedad", "reclamos"]}
D["rank_mi_pos"] = {c: int(list(rank.sort_values("MI", ascending=False).index).index(c)) + 1 for c in ["reclamos", "tiene_reclamo", "llamadas_por_mes"]}

# redundancia (celdas 30-31)
corr = X.corr().abs()
sup = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pr = sup.stack().sort_values(ascending=False)
D["n_pares90"] = int((pr > 0.90).sum())
D["pares90"] = [[a, b, round(float(v), 3), (a if mi[a] >= mi[b] else b)] for (a, b), v in pr[pr > 0.90].items()]
D["mi_par"] = {c: r4(mi[c]) for c in ["reclamos", "tiene_reclamo", "segundos", "frec_llamadas", "frec_sms", "clv"]}

# ============================================================ 7. wrapper y embedded (celdas 25, 33-35)
D["subconj"] = {str(p): [2 ** p - 1, (2 ** p - 1) / 1000] for p in [10, 20, 30, 50]}
D["anios50"] = round((2 ** 50 - 1) / 1000 / 60 / 60 / 24 / 365)
est = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
rfe = RFECV(estimator=est, step=1, cv=5, scoring="roc_auc", min_features_to_select=4,
            importance_getter="named_steps.clf.coef_").fit(X, y)
D["rfe_n"] = int(rfe.n_features_)
D["rfe_fuera"] = [c for c in X.columns if c not in set(X.columns[rfe.support_])]


def lasso_fit(Cval, Xa=X, ya=y):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=Cval, max_iter=3000,
                                                                          random_state=0))]).fit(Xa, ya)


coef = pd.Series(lasso_fit(0.05)[-1].coef_[0], index=X.columns)
D["lasso_n"] = int((coef != 0).sum())
D["lasso_top"] = [[k, round(float(coef[k]), 3)] for k in coef.abs().sort_values(ascending=False).index[:8]]
D["lasso_C"] = {str(Cv): int((lasso_fit(Cv)[-1].coef_[0] != 0).sum()) for Cv in [0.01, 0.05, 0.1, 0.5, 1.0]}

# ============================================================ 8. top-k (celdas 37-44)
top4 = mi.sort_values(ascending=False).head(4).index.tolist()
manual = ["reclamos", "segundos", "estado_2", "numero_fallas"]


def corr_media(cols):
    m = X[cols].corr().abs().values
    return float(m[np.triu_indices(len(cols), 1)].mean())


def pares_r(cols):
    m = X[cols].corr().abs()
    return [[cols[i], cols[j], round(float(m.iloc[i, j]), 3)] for i in range(len(cols)) for j in range(i + 1, len(cols))]


D["pares_top4"], D["pares_manual4"] = pares_r(top4), pares_r(manual)
D["top4"] = dict(cols=top4, auc=r4(auc(top4)), cm=round(corr_media(top4), 2))
D["manual4"] = dict(cols=manual, auc=r4(auc(manual)), cm=round(corr_media(manual), 2))
sem = []
for s_ in range(5):
    mi_s = pd.Series(mutual_info_classif(X, y, random_state=s_), index=X.columns)
    t4 = mi_s.sort_values(ascending=False).head(4).index.tolist()
    sem.append([s_, r4(auc(t4)), t4])
D["semillas"] = sem


def sin_redundancia(k):
    sel = [mi.idxmax()]
    while len(sel) < k:
        resto = [c for c in X.columns if c not in sel]
        redun = X[resto].corrwith(X[sel[0]]).abs()
        for s_ in sel[1:]:
            redun = np.maximum(redun, X[resto].corrwith(X[s_]).abs())
        sel.append((mi[resto] - redun).idxmax())
    return sel


D["k_tabla"] = []
for k in [4, 6, 8, 10, 12]:
    tk = mi.sort_values(ascending=False).head(k).index.tolist()
    a1, a2 = auc(tk), auc(sin_redundancia(k))
    D["k_tabla"].append([k, r4(a1), r4(a2), r4(a2 - a1)])
top6 = mi.sort_values(ascending=False).head(6).index.tolist()
D["top6_nb"] = dict(cols=top6, auc=r4(auc(top6)), cm=round(corr_media(top6), 2))
slide_top6 = ["segundos", "log_segundos", "contactos_por_mes", "log_clv", "clv_por_antiguedad", "llamadas_por_mes"]
slide_man6 = ["reclamos", "segundos", "frec_llamadas", "estado_2", "numero_fallas", "monto"]
D["slide_top6"] = dict(cols=slide_top6, auc=r4(auc(slide_top6)), cm=round(corr_media(slide_top6), 2))
D["slide_man6"] = dict(cols=slide_man6, auc=r4(auc(slide_man6)), cm=round(corr_media(slide_man6), 2))
sel_lasso = coef[coef != 0].index.tolist()
D["resumen"] = [["12 originales", 12, r4(auc(originales))], ["50 candidatas", D["p"], r4(auc(list(X.columns)))],
                [f"Lasso ({len(sel_lasso)})", len(sel_lasso), r4(auc(sel_lasso))], ["Top-4 por MI", 4, D["top4"]["auc"]],
                ["4 sin redundancia", 4, D["manual4"]["auc"]]]

# ============================================================ 9. la trampa (celdas 46, 48)
rng = np.random.default_rng(42)
Xr, yr = rng.normal(size=(100, 1000)), rng.integers(0, 2, size=100)
Xs = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)
mal = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yr, cv=5, scoring="roc_auc").mean())
ok = float(cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=2000))]), Xr, yr,
                           cv=5, scoring="roc_auc").mean())
_, p_r = f_classif(Xr, yr)
D["ruido"] = dict(mal=round(mal, 3), bien=round(ok, 3), inventado=round(mal - 0.5, 3), signif05=int((p_r < 0.05).sum()),
                  signif01=int((p_r < 0.01).sum()), unos=int(yr.sum()))
pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                 ("sel", SelectKBest(mutual_info_classif, k=12)), ("clf", LogisticRegression(max_iter=3000))])
ah = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
D["honesto"] = dict(folds=[r4(v) for v in ah], media=r4(ah.mean()))

# misma seleccion MI k=12 fuera vs dentro del CV, con la base real (semilla fija para comparar)
from functools import partial  # noqa: E402
mi0 = partial(mutual_info_classif, random_state=0)
cols12 = list(X.columns[SelectKBest(mi0, k=12).fit(X, y).get_support()])
pipe0 = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(mi0, k=12)),
                  ("clf", LogisticRegression(max_iter=3000))])
D["real_fuera_dentro"] = dict(fuera=r4(auc(cols12)), dentro=r4(cross_val_score(pipe0, X, y, cv=cv, scoring="roc_auc").mean()))

# estabilidad del Lasso (celda 51)
cuenta = Counter()
for semilla in range(5):
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=semilla)
    c_ = pd.Series(lasso_fit(0.05, Xtr, ytr)[-1].coef_[0], index=X.columns)
    cuenta.update(c_[c_ != 0].index.tolist())
D["estab"] = dict(siempre=sum(1 for v in cuenta.values() if v == 5), a_veces=sum(1 for v in cuenta.values() if v < 5),
                  nunca=D["p"] - len(cuenta))

(AQUI / "datos33.json").write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(D, ensure_ascii=False)[:200], "...")
print("OK datos33.json")
