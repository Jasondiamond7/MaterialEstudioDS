"""Numeros reales para los videos de la Clase 32 (se ejecuta con el venv de Clase 35, que tiene pandas + sklearn + openpyxl).

Reutiliza los datos de figuras_c32.py (dataset sintetico de la profesora, default_rng(32), y base real del xlsx), sin
generar figuras. Escribe datos32.json junto a este archivo.
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTE = AQUI.parents[1] / "_fuente"
sys.path.insert(0, str(FUENTE))
import figuras_c32 as F  # noqa: E402  (solo se usan sus datos de nivel de modulo; no se llaman funciones *_fig)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE, trustworthiness  # noqa: E402
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder  # noqa: E402

RS = F.RS
D = {}
r3 = lambda v: round(float(v), 3)  # noqa: E731

# ------------------------------------------------ sintetico: tamanos
D["n"] = len(F.df_s)
D["fuga"] = round(float(F.y.mean()) * 100, 2)
D["n_tr"], D["n_te"] = len(F.ytr), len(F.yte)
D["fugas_te"] = int(F.yte.sum())
D["cols_raw"] = list(F.X_raw.columns)
D["cols"] = F.COLS

# ------------------------------------------------ 1. dimensionalidad
corr = F.X_raw.corr()
uso = ["segundos", "frec_llamadas", "n_contactos", "frec_sms", "monto", "clv"]
cu = corr.loc[uso, uso].values[np.triu_indices(6, 1)]
D["corr_seg_llam"] = round(float(corr.loc["segundos", "frec_llamadas"]), 2)
D["corr_uso_min"], D["corr_uso_max"] = round(float(cu.min()), 2), round(float(cu.max()), 2)
D["corr_uso"] = {f"{a}|{b}": round(float(corr.loc[a, b]), 2) for i, a in enumerate(uso) for b in uso[i + 1:]}
Z = (F.X_raw - F.X_raw.mean()) / F.X_raw.std()
vif = {}
for c in uso:
    otros = [k for k in F.X_raw.columns if k != c]
    r2 = LinearRegression().fit(Z[otros], Z[c]).score(Z[otros], Z[c])
    vif[c] = round(float(1 / (1 - r2)), 1)
D["vif"] = vif

rng = np.random.default_rng(1)
conc = {}
for d in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    P = rng.uniform(size=(400, d))
    r = []
    for q in range(15):
        dist = np.linalg.norm(np.delete(P, q, axis=0) - P[q], axis=1)
        r.append((dist.max() - dist.min()) / dist.min())
    conc[str(d)] = round(float(np.mean(r)), 2)
D["conc"] = conc

ks = [0, 10, 25, 50, 100, 200, 400]
te_m, tr_m = [], []
for k in ks:
    te, tr = [], []
    for s in range(25):
        r = np.random.default_rng(100 + s)
        idx = r.choice(len(F.Xtr_raw), 100, replace=False)
        while F.ytr.values[idx].sum() < 8:
            idx = r.choice(len(F.Xtr_raw), 100, replace=False)
        A, T = F.Xtr_raw.values[idx], F.Xte_raw.values
        if k:
            A = np.hstack([A, r.normal(size=(100, k))])
            T = np.hstack([T, r.normal(size=(len(T), k))])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(A, F.ytr.values[idx])
        te.append(roc_auc_score(F.yte, m.predict_proba(T)[:, 1]))
        tr.append(roc_auc_score(F.ytr.values[idx], m.predict_proba(A)[:, 1]))
    te_m.append(r3(np.mean(te)))
    tr_m.append(r3(np.mean(tr)))
D["ruido"] = dict(ks=ks, prueba=te_m, entren=tr_m)
print("dimensionalidad ok", D["corr_seg_llam"], D["vif"], D["conc"]["2"], D["conc"]["1000"], D["ruido"])

# ------------------------------------------------ 2. representacion
D["sd"] = {k: round(float(v), 2) for k, v in F.Xtr_raw[["segundos", "clv", "frec_sms", "frec_llamadas", "monto", "reclamos"]].std().items()}
D["mean"] = {k: round(float(v), 2) for k, v in F.Xtr_raw[["segundos", "reclamos"]].mean().items()}
rng = np.random.default_rng(7)
ctr = rng.choice([f"c{i}" for i in range(60)], len(F.Xtr_raw))
cte = rng.choice([f"c{i}" for i in range(60)], len(F.Xte_raw))
ya, yb = F.ytr.values, F.yte.values
med = pd.Series(ya).groupby(ctr).mean()
ntr, nte = pd.Series(ctr).map(med).values, pd.Series(cte).map(med).fillna(ya.mean()).values
tenc = TargetEncoder(smooth="auto", target_type="binary", cv=5, random_state=RS)
cvtr = tenc.fit_transform(ctr.reshape(-1, 1), ya).ravel()
cvte = tenc.transform(cte.reshape(-1, 1)).ravel()
oh = OneHotEncoder(handle_unknown="ignore")
Otr, Ote = oh.fit_transform(ctr.reshape(-1, 1)), oh.transform(cte.reshape(-1, 1))
lr = LogisticRegression(max_iter=2000).fit(Otr, ya)
D["target_enc"] = {"media": [r3(roc_auc_score(ya, ntr)), r3(roc_auc_score(yb, nte))],
                   "onehot": [r3(roc_auc_score(ya, lr.predict_proba(Otr)[:, 1])), r3(roc_auc_score(yb, lr.predict_proba(Ote)[:, 1]))],
                   "targetenc": [r3(roc_auc_score(ya, cvtr)), r3(roc_auc_score(yb, cvte))]}
print("representacion ok", D["sd"], D["target_enc"])

# ------------------------------------------------ 3. geometria 2D (segundos, clv)
par = ["segundos", "clv"]
Z2 = ((F.X_raw[par] - F.X_raw[par].mean()) / F.X_raw[par].std()).values
p2 = PCA(random_state=RS).fit(Z2)
u = p2.components_[0] * np.sign(p2.components_[0][0])
D["geo"] = dict(r=round(float(np.corrcoef(Z2.T)[0, 1]), 3), eig=[r3(v) for v in p2.explained_variance_],
                var=[round(float(v) * 100, 1) for v in p2.explained_variance_ratio_], u=[r3(v) for v in u])
idx = np.random.default_rng(5).choice(len(Z2), 220, replace=False)
D["geo"]["xs"] = [round(float(v), 2) for v in Z2[idx, 0]]
D["geo"]["ys"] = [round(float(v), 2) for v in Z2[idx, 1]]
print("geo ok", {k: v for k, v in D["geo"].items() if k not in ("xs", "ys")})

# ------------------------------------------------ 4. scree, CV, prueba
ve, va = F.var_exp, F.var_acum
D["scree_s"] = dict(var=[round(float(v) * 100, 1) for v in ve], acum=[round(float(v) * 100, 1) for v in va],
                    eig=[r3(v) for v in F.pca_full.explained_variance_],
                    k80=int(np.argmax(va >= 0.80) + 1), k85=int(np.argmax(va >= 0.85) + 1), k90=int(np.argmax(va >= 0.90) + 1))
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
med_, des_ = [], []
for k in range(1, 16):
    pipe = Pipeline([("esc", StandardScaler()), ("pca", PCA(n_components=k, random_state=RS)), ("lr", LogisticRegression(max_iter=2000))])
    sc = cross_val_score(pipe, F.Xtr_raw, F.ytr, cv=cv, scoring="roc_auc")
    med_.append(r3(sc.mean()))
    des_.append(r3(sc.std()))
sc15 = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), F.Xtr_raw, F.ytr, cv=cv, scoring="roc_auc")
D["cv_auc"] = dict(medias=med_, desv=des_, vars15=r3(sc15.mean()), vars15_sd=r3(sc15.std()))


def evaluar(modelo, Xa, Xb, ya_, yb_):
    modelo.fit(Xa, ya_)
    p = modelo.predict(Xb)
    return dict(F1=r3(f1_score(yb_, p)), MCC=r3(matthews_corrcoef(yb_, p)), AUC=r3(roc_auc_score(yb_, modelo.predict_proba(Xb)[:, 1])))


D["prueba"] = {"15": evaluar(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), F.Xtr_raw, F.Xte_raw, F.ytr, F.yte)}
for k in (3, 8):
    D["prueba"][f"pca{k}"] = evaluar(make_pipeline(StandardScaler(), PCA(k, random_state=RS), LogisticRegression(max_iter=2000)),
                                     F.Xtr_raw, F.Xte_raw, F.ytr, F.yte)
print("scree ok", D["scree_s"]["acum"][:10], D["cv_auc"], D["prueba"])

# ------------------------------------------------ 5. cargas
comp = F.pca_full.components_
D["cargas"] = {f"PC{j + 1}": {c: r3(comp[j][i]) for i, c in enumerate(F.COLS)} for j in range(4)}
sc = F.pca_full.transform(F.Xtr)
D["pc_fuga_corr"] = [r3(np.corrcoef(sc[:, j], F.ytr.values)[0, 1]) for j in range(5)]
corr_c = comp[:2].T * np.sqrt(F.pca_full.explained_variance_[:2])
D["circ"] = {c: [round(float(corr_c[i, 0]), 2), round(float(corr_c[i, 1]), 2)] for i, c in enumerate(F.COLS)}
idx = np.random.default_rng(9).choice(len(sc), 420, replace=False)
D["proy"] = dict(x=[round(float(v), 2) for v in sc[idx, 0]], y=[round(float(v), 2) for v in sc[idx, 1]],
                 fuga=[int(v) for v in F.ytr.values[idx]])
p_raw = PCA(random_state=RS).fit(F.Xtr_raw.values)
vr = F.Xtr_raw.var()
cuota = (vr / vr.sum()).sort_values(ascending=False)
D["escala"] = dict(cuota={k: round(float(v) * 100, 2) for k, v in cuota.head(4).items()},
                   pc1_raw=round(float(p_raw.explained_variance_ratio_[0]) * 100, 2),
                   r_pc1_seg=round(float(abs(np.corrcoef(F.Xtr_raw["segundos"], p_raw.transform(F.Xtr_raw.values)[:, 0])[0, 1])), 4),
                   carga_seg=r3(abs(p_raw.components_[0][F.COLS.index("segundos")])))
print("cargas ok", D["pc_fuga_corr"], D["escala"])

# ------------------------------------------------ 6. t-SNE
idx = np.random.default_rng(11).choice(len(F.Xtr), 500, replace=False)
D["tsne"] = dict(idx_fuga=[int(v) for v in F.ytr.values[idx]], emb={}, tw={})
e = PCA(n_components=2, random_state=RS).fit_transform(F.Xtr)
D["tsne"]["tw"]["PCA"] = r3(trustworthiness(F.Xtr, e, n_neighbors=10))
D["tsne"]["emb"]["PCA"] = [[round(float(a), 2), round(float(b), 2)] for a, b in e[idx]]
for perp in (5, 30, 80):
    e = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=RS).fit_transform(F.Xtr)
    D["tsne"]["tw"][f"p{perp}"] = r3(trustworthiness(F.Xtr, e, n_neighbors=10))
    D["tsne"]["emb"][f"p{perp}"] = [[round(float(a), 2), round(float(b), 2)] for a, b in e[idx]]
for sem in (0, 1):
    e = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto", random_state=sem).fit_transform(F.Xtr)
    D["tsne"]["emb"][f"s{sem}"] = [[round(float(a), 2), round(float(b), 2)] for a, b in e[idx]]
print("tsne ok", D["tsne"]["tw"])

# ------------------------------------------------ 7. fuga con seleccion / PCA fuera del Pipeline
res = {"sel_fuera": [], "sel_pipe": [], "pca_fuera": [], "pca_pipe": []}
for s in range(40):
    rng = np.random.default_rng(s)
    Xn, yn = rng.normal(size=(100, 1000)), rng.integers(0, 2, 100)
    lr = LogisticRegression(max_iter=1000)
    res["sel_fuera"].append(cross_val_score(lr, SelectKBest(f_classif, k=10).fit_transform(Xn, yn), yn, cv=5).mean())
    res["sel_pipe"].append(cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=10)), ("m", lr)]), Xn, yn, cv=5).mean())
    Zn = PCA(n_components=10, random_state=RS).fit_transform(StandardScaler().fit_transform(Xn))
    res["pca_fuera"].append(cross_val_score(lr, Zn, yn, cv=5).mean())
    res["pca_pipe"].append(cross_val_score(Pipeline([("esc", StandardScaler()), ("pca", PCA(n_components=10, random_state=RS)), ("m", lr)]), Xn, yn, cv=5).mean())
D["fuga_exp"] = {k: r3(np.mean(v)) for k, v in res.items()}
print("fuga ok", D["fuga_exp"])

# ------------------------------------------------ 8. base real
df_r = F.df_r
D["real"] = dict(n=len(df_r), fuga=round(float(F.yr.mean()) * 100, 2),
                 var=[round(float(v) * 100, 1) for v in F.varR], acum=[round(float(v) * 100, 1) for v in F.acumR],
                 k80=int(np.argmax(F.acumR >= 0.80) + 1))
cr = df_r.drop(columns=["fuga"]).corr()
D["real"]["pares"] = {f"{a}|{b}": round(float(cr.loc[a, b]), 2) for a, b in
                      [("segundos", "frec_llamadas"), ("frec_sms", "clv"), ("n_contactos", "frec_llamadas"),
                       ("numero_fallas", "monto"), ("estado", "segundos"), ("numero_fallas", "segundos"),
                       ("numero_fallas", "frec_llamadas"), ("reclamos", "segundos")]}
D["real"]["prueba"] = {"15": evaluar(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), F.XRtr_raw, F.XRte_raw, F.yrtr, F.yrte),
                       "pca7": evaluar(make_pipeline(StandardScaler(), PCA(7, random_state=RS), LogisticRegression(max_iter=2000)),
                                       F.XRtr_raw, F.XRte_raw, F.yrtr, F.yrte)}
corr_r = F.pca_R.components_[:2].T * np.sqrt(F.pca_R.explained_variance_[:2])
COLR = list(F.XR_enc.columns)
D["real"]["circ"] = {c: [round(float(corr_r[i, 0]), 2), round(float(corr_r[i, 1]), 2)] for i, c in enumerate(COLR)}
D["real"]["min_seg"] = float(df_r["segundos"].min())
D["real"]["min_llam"] = float(df_r["frec_llamadas"].min())
print("real ok", {k: v for k, v in D["real"].items() if k not in ("circ",)})

(AQUI / "datos32.json").write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
print("OK datos32.json")
