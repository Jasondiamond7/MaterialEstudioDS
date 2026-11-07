import sys, json
sys.path.insert(0, r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\videos\_tmp_c31_profe")
from club import *
D = {}
# --- sigmoide
lr = clf.named_steps["lr"]
Zte = clf[:-1].transform(Xte); z = lr.decision_function(Zte); p = 1/(1+np.exp(-z))
print("check p vs proba", float(np.abs(p-proba).max()))
rng = np.random.default_rng(3)
idx = np.sort(rng.choice(len(z), 70, replace=False))
D["sig"] = dict(z=[round(float(a),2) for a in z[idx]], p=[round(float(a),3) for a in p[idx]], y=[int(a) for a in yte.values[idx]])
D["sig"]["n_p05"] = int((proba>=0.5).sum())
# umbrales
th = {}
for t in (0.3,0.5,0.7):
    pr_ = (proba>=t).astype(int)
    th[t] = dict(prec=round(float(precision_score(yte,pr_)),3), rec=round(float(recall_score(yte,pr_)),3), n=int(pr_.sum()))
D["umbral"]=th; print(th)
cw = len(ytr)/(2*np.bincount(ytr)); print("cw", cw, np.bincount(ytr))
print("coef", lr.coef_[0], lr.intercept_)
# --- regresion
Xr, yr = df[FEATS_REG], df["gasto_mensual"]
Xrt, Xre, yrt, yre = train_test_split(Xr, yr, test_size=0.30, random_state=RS)
reg = Pipeline([("imp", SimpleImputer(strategy="median")), ("lr", LinearRegression())]).fit(Xrt, yrt)
pr = reg.predict(Xre); l2 = reg.named_steps["lr"]
print("reg", mean_squared_error(yre,pr), np.sqrt(mean_squared_error(yre,pr)), r2_score(yre,pr), r2_score(yrt,reg.predict(Xrt)), l2.coef_, l2.intercept_, yr.mean(), yr.std())
idx = np.sort(rng.choice(len(pr), 90, replace=False))
D.setdefault("reg",{}).update(dict(real=[round(float(a),1) for a in yre.values[idx]], pred=[round(float(a),1) for a in pr[idx]]))
res = yre.values-pr; print("res sd", res.std(), res.mean())
h,_ = np.histogram(res, bins=np.arange(-42,43,6)); D["reg"]["hist"]=[int(a) for a in h]; print("hist", h.tolist(), len(res))
# --- kmeans
pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
Z = pre.fit_transform(df[FEATS_CLU])
km = KMeans(3, random_state=RS, n_init=10).fit(Z); lab = km.labels_
sizes = [int((lab==i).sum()) for i in range(3)]; print("sizes", sizes, km.cluster_centers_.round(2).tolist())
for k in (2,3,4):
    kk = KMeans(k, random_state=RS, n_init=10).fit(Z); print("sil", k, round(silhouette_score(Z, kk.labels_),3))
print("canc por seg", [round(float(df.cancelo[lab==i].mean()),3) for i in range(3)])
# subsample para dibujar
idx = np.sort(rng.choice(len(Z), 160, replace=False))
D["km"] = dict(x=[round(float(a),2) for a in Z[idx,0]], y=[round(float(a),2) for a in Z[idx,1]], lab=[int(a) for a in lab[idx]],
               cent=[[round(float(a),2) for a in c[:2]] for c in km.cluster_centers_], sizes=sizes)
Zr = pre.fit_transform(df[FEATS_CLU]); print("PCA", PCA(2,random_state=RS).fit(Zr).explained_variance_ratio_)
Z2 = StandardScaler  # unused
raw = KMeans(3, random_state=RS, n_init=10).fit(df[FEATS_CLU].fillna(df[FEATS_CLU].median()))
from sklearn.metrics import adjusted_rand_score
print("ARI", adjusted_rand_score(lab, raw.labels_))
# --- ajuste
xt, xe, yt, ye = train_test_split(X_DATO.reshape(-1,1), Y_DATO, test_size=0.3, random_state=7)
def poly(g, esc=True):
    return make_pipeline(*([StandardScaler()] if esc else []), PolynomialFeatures(g, include_bias=False), LinearRegression())
xl = np.linspace(0,10,60).reshape(-1,1)
A = dict(xt=[round(float(a),2) for a in xt[:,0]], yt=[round(float(a),2) for a in yt], xe=[round(float(a),2) for a in xe[:,0]], ye=[round(float(a),2) for a in ye], xl=[round(float(a),2) for a in xl[:,0]])
for g in (1,2,15):
    m = poly(g).fit(xt,yt); A[f"c{g}"] = [round(float(a),2) for a in m.predict(xl)]
    print("grado", g, round(r2_score(yt,m.predict(xt)),2), round(r2_score(ye,m.predict(xe)),2))
m = poly(15,False).fit(xt,yt); print("g15 sin", r2_score(yt,m.predict(xt)), r2_score(ye,m.predict(xe)))
tr=[];te=[]
for g in range(1,16):
    m=poly(g).fit(xt,yt); tr.append(round(float(r2_score(yt,m.predict(xt))),3)); te.append(round(float(r2_score(ye,m.predict(xe))),3))
A["tr"]=tr; A["te"]=te; D["ajuste"]=A; print(len(xt), len(xe), tr, te)
# --- regularizacion
d = df.assign(ruido_1=RUIDO1, ruido_2=RUIDO2)
cols = ["antiguedad_meses","uso_mensual","edad","quejas_soporte","ruido_1","ruido_2"]
Xs = StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
A1,A2,b1,b2 = train_test_split(Xs, d["gasto_mensual"], test_size=0.3, random_state=42)
for nm,m in [("OLS",LinearRegression()),("Ridge5",Ridge(alpha=5.0)),("Lasso0.5",Lasso(alpha=0.5))]:
    m.fit(A1,b1); print(nm, dict(zip(cols,[round(float(v),2) for v in m.coef_])), "r2", round(r2_score(b2,m.predict(A2)),3))
D["reg_coef"] = {nm: [round(float(v),2) for v in m.fit(A1,b1).coef_] for nm,m in [("OLS",LinearRegression()),("Ridge5",Ridge(alpha=5.0)),("Lasso05",Lasso(alpha=0.5))]}
json.dump(D, open("datos31.json","w"), ensure_ascii=False)
# --- apuntes a mano
import math
print("sig apuntes", 1/(1+math.exp(-0.8)), 1/(1+math.exp(2.6)), math.exp(1.5))
print("sig ej", 1/(1+math.exp(-0.7)), 1/(1+math.exp(2.2)), math.exp(0.8))

# --- extras K-Means/PCA
D = json.load(open("datos31.json", encoding="utf-8"))
ks = list(range(2, 9))
D["km"]["inercia"] = {k: round(float(KMeans(k, random_state=RS, n_init=10).fit(Z).inertia_), 1) for k in ks}
D["km"]["silueta"] = {k: round(float(silhouette_score(Z, KMeans(k, random_state=RS, n_init=10).fit(Z).labels_)), 3) for k in ks}
pca = PCA(3, random_state=RS).fit(Z); E = pca.transform(Z)
D["km"]["pca_var"] = [round(float(a), 4) for a in pca.explained_variance_ratio_]
rng2 = np.random.default_rng(3); rng2.choice(800, 70, replace=False)  # (no usado)
sub_idx = np.sort(np.random.default_rng(3).choice(800, 160, replace=False))
print("idx igual a la anterior?", False)
D["km"]["pcx"] = [round(float(a), 2) for a in E[sub_idx, 0]]; D["km"]["pcy"] = [round(float(a), 2) for a in E[sub_idx, 1]]
# la submuestra de km se hizo con rng secuencial; recalculamos coherente con la misma submuestra
D["km"]["x"] = [round(float(a), 2) for a in Z[sub_idx, 0]]; D["km"]["y"] = [round(float(a), 2) for a in Z[sub_idx, 1]]
D["km"]["lab"] = [int(a) for a in lab[sub_idx]]
D["km"]["perfil"] = {int(i): [round(float(a), 2) for a in km.cluster_centers_[i]] for i in range(3)}
print(D["km"]["inercia"], D["km"]["silueta"], D["km"]["pca_var"], D["km"]["perfil"])
json.dump(D, open("datos31.json", "w"), ensure_ascii=False)
