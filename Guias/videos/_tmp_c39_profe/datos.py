"""Numeros reales para los videos de la Clase 39 (venv de Clase 35: pandas 3 + sklearn 1.9 + kmodes 0.12).

1) Ejecuta kmeans_numero_clusters.py del profesor TAL CUAL (runpy, sin modificar el archivo). Solo se redirige:
   - os.chdir(r"D:\\codigos\\clustering") -> _tmp_c39_profe/resultados_clase (copia de vinos.csv y vinos_clases_reales.csv);
   - plt.show() -> guarda cada figura en resultados_clase/fig_NN.png (backend Agg);
   - la salida impresa -> resultados_clase/salida_python.txt.
   Despues lee las variables finales de esa ejecucion (wss, silueta, kmedias, tabla, gap, ...).
2) K-modes y K-prototypes con el codigo de las diapositivas 37, 39, 45 y 47 (el .py de la clase no los trae).
3) Calculos adicionales para los videos (marcados como tales en pantalla): iteraciones en dos variables, inicios
   aleatorios sueltos, silueta por grupo, ejemplos simulados (lunas, atipico), etc.
Escribe datos39.json (al menos 6 decimales) junto a este archivo.
"""
import builtins
import contextlib
import io
import json
import os
import runpy
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CLASE = next(p for p in RAIZ.glob("Clase 39*") if p.is_dir())
RES = AQUI / "resultados_clase"
RES.mkdir(exist_ok=True)
for f in ("vinos.csv", "vinos_clases_reales.csv"):
    shutil.copy2(CLASE / f, RES / f)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import kmodes  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_samples, adjusted_rand_score  # noqa: E402

R6 = lambda v: round(float(v), 6)  # noqa: E731
L6 = lambda xs: [round(float(v), 6) for v in xs]  # noqa: E731

# ------------------------------------------------------------ 1. el script de la clase, tal cual
_chdir = os.chdir
_nfig = [0]


def _show(*a, **k):
    for num in plt.get_fignums():
        _nfig[0] += 1
        plt.figure(num).savefig(RES / f"fig_{_nfig[0]:02}.png", dpi=90)
    plt.close("all")


os.chdir = lambda p: _chdir(RES)          # D:\codigos\clustering -> resultados_clase
plt.show = _show
buf = io.StringIO()
cwd0 = os.getcwd()
try:
    with contextlib.redirect_stdout(buf):
        G = runpy.run_path(str(CLASE / "kmeans_numero_clusters.py"), run_name="__main__")
finally:
    os.chdir = _chdir
    _chdir(cwd0)
(RES / "salida_python.txt").write_text(buf.getvalue(), encoding="utf-8")

vinos, x, z = G["vinos"], G["x"], G["z"]
D = {"versiones": dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, kmodes=kmodes.__version__)}
D["datos"] = dict(n=int(vinos.shape[0]), cols=int(vinos.shape[1]), p=int(x.shape[1]), variables=list(x.columns),
                  media={c: R6(x[c].mean()) for c in x.columns}, sd={c: R6(x[c].std(ddof=1)) for c in x.columns},
                  var={c: R6(x[c].var(ddof=1)) for c in x.columns}, primer=dict(codigo=str(vinos.codigo.iloc[0]),
                  alcohol=R6(vinos.alcohol.iloc[0]), z_alcohol=R6(z.alcohol.iloc[0])))
sv = x.var(ddof=1)
D["datos"]["cuota_var"] = {c: R6(sv[c] / sv.sum()) for c in x.columns}
D["datos"]["z_media_max"] = R6(np.abs(z.mean()).max())
D["datos"]["z_sd"] = L6(z.std(ddof=1))
# K-means sin estandarizar (adicional): la prolina domina
km_raw = KMeans(n_clusters=3, n_init=50, max_iter=100, random_state=2026, algorithm="lloyd", tol=1e-8).fit(x)
grupo = np.asarray(G["grupo"])
D["datos"]["raw_tamanos"] = sorted(int(v) for v in np.bincount(km_raw.labels_))
D["datos"]["raw_ari_vs_z"] = R6(adjusted_rand_score(grupo, km_raw.labels_))
clase_real = np.asarray(G["clase_real"])
D["datos"]["raw_ari_clase"] = R6(adjusted_rand_score(clase_real, km_raw.labels_))
D["datos"]["z_ari_clase"] = R6(adjusted_rand_score(clase_real, grupo))
# coincidencia con la clase mayoritaria sin estandarizar
tab_raw = pd.crosstab(clase_real, km_raw.labels_)
D["datos"]["raw_coinc"] = int(tab_raw.max(axis=0).sum())
# estandarizar con ddof=0 (numpy) en vez de ddof=1 (R, pandas): mismos grupos, WSS multiplicado por 177/176
z0 = (x - x.mean()) / x.std(ddof=0)
km0 = KMeans(n_clusters=3, n_init=50, max_iter=100, random_state=2026, algorithm="lloyd", tol=1e-8).fit(z0)
D["datos"]["ddof0_wss"] = R6(km0.inertia_)
D["datos"]["ddof0_ari"] = R6(adjusted_rand_score(grupo, km0.labels_))

# ---- K-means k = 1..8 (del script)
D["km"] = dict(k=list(G["valores_k"]), wss=L6(G["wss"]), silueta=[None if np.isnan(s) else R6(s) for s in G["silueta"]],
               k_silueta=int(G["k_silueta"]), tamanos=[int((grupo == g).sum()) for g in (1, 2, 3)],
               centros={str(g): {c: R6(v) for c, v in G["centros"].loc[g].items()} for g in (1, 2, 3)},
               wss_k3=R6(G["kmedias"].inertia_), tss=R6(G["tss"]), bss=R6(G["tss"] - G["kmedias"].inertia_),
               n_iter=int(G["kmedias"].n_iter_),
               perfiles={str(g): {c: R6(v) for c, v in G["perfiles"].loc[g].items()} for g in (1, 2, 3)},
               n_iter_por_k=[int(m.n_iter_) for m in G["modelos"]])
w = np.array(G["wss"])
D["codo"] = dict(red=L6(w[:-1] - w[1:]), pct=L6(100 * (w[:-1] - w[1:]) / w[:-1]), pct_tss=L6(100 * (1 - w / w[0])))
# ---- silueta por observacion (adicional)
s_i = silhouette_samples(z, G["kmedias"].labels_)
D["sil"] = dict(por_grupo=[R6(s_i[grupo == g].mean()) for g in (1, 2, 3)], negativas=[int((s_i[grupo == g] < 0).sum()) for g in (1, 2, 3)],
                min=R6(s_i.min()), max=R6(s_i.max()), media=R6(s_i.mean()))
# ---- Gap (del script)
gap, err = np.array(G["gap"]), np.array(G["error_gap"])
D["gap"] = dict(gap=L6(gap), err=L6(err), k_gap=int(G["k_gap"]), B=int(G["B"]),
                umbral=L6(gap[1:] - err[1:]), cumple=[bool(gap[k - 1] >= gap[k] - err[k]) for k in range(1, 8)],
                logw=L6(np.log(G["wss"])), logw_ref=L6(G["log_wss_ref"].mean(axis=0)), k_max_gap=int(np.argmax(gap) + 1))
# ---- PCA y clases reales (del script)
tabla = G["tabla"]
D["pca"] = dict(var=L6(G["varianza"]), var2=R6(sum(G["varianza"])),
                tabla=[[int(tabla.loc[c, g]) for g in (1, 2, 3)] for c in (1, 2, 3)],
                clase_del_grupo={str(g): int(c) for g, c in G["clase_del_grupo"].items()},
                coincidencias=int(G["coincidencias"]), pureza=R6(G["pureza"]), discrepa=int(np.sum(G["discrepa"])),
                clases_n=[int((clase_real == c).sum()) for c in (1, 2, 3)],
                disc_clase=[int(c) for c in clase_real[G["discrepa"]]], disc_grupo=[int(g) for g in grupo[G["discrepa"]]],
                pc1=L6(G["coordenadas"][:, 0]), pc2=L6(G["coordenadas"][:, 1]), grupo=[int(g) for g in grupo],
                clase=[int(c) for c in clase_real])
D["figuras_clase"] = _nfig[0]

# ------------------------------------------------------------ 2. K-modes y K-prototypes (codigo de las diapositivas)
from kmodes.kmodes import KModes  # noqa: E402
from kmodes.kprototypes import KPrototypes  # noqa: E402

categorias = pd.DataFrame({
    "alcohol": np.where(vinos.alcohol < 13, "bajo", "alto"),
    "flavonoides": np.where(vinos.flavonoides < 1.5, "bajo",
                            np.where(vinos.flavonoides < 3, "medio", "alto")),
    "color": np.where(vinos.color < 4, "bajo",
                      np.where(vinos.color < 6.5, "medio", "alto"))
})
modelos_modos, costo_modos = [], []
for k in range(1, 9):
    modelo = KModes(n_clusters=k, init="Huang", n_init=20, max_iter=100, random_state=2026)
    modelo.fit(categorias)
    modelos_modos.append(modelo)
    costo_modos.append(modelo.cost_)
kmodas = modelos_modos[3]
grupo_modos = kmodas.labels_ + 1
cm = {}
for g in range(1, 5):
    sub = categorias[grupo_modos == g]
    cm[str(g)] = dict(n=int(len(sub)), centro=list(kmodas.cluster_centroids_[g - 1]),
                      color={c: int((sub.color == c).sum()) for c in ("alto", "medio", "bajo")},
                      alcohol={c: int((sub.alcohol == c).sum()) for c in ("alto", "bajo")},
                      flavonoides={c: int((sub.flavonoides == c).sum()) for c in ("alto", "medio", "bajo")},
                      costo=int(sum((sub.values != kmodas.cluster_centroids_[g - 1]).sum(axis=1))))
D["kmodes"] = dict(costo=L6(costo_modos), grupos=cm,
                   cats={c: {v: int((categorias[c] == v).sum()) for v in sorted(categorias[c].unique())} for c in categorias},
                   tabla_clase=pd.crosstab(clase_real, grupo_modos).values.tolist(),
                   empates_flav_15=int((vinos.flavonoides == 1.5).sum()), empates_color_4=int((vinos.color == 4).sum()))

mixtos = pd.concat([z[["ceniza", "matiz"]], categorias], axis=1)
peso = 1
modelos_proto, costo_proto = [], []
for k in range(1, 9):
    modelo = KPrototypes(n_clusters=k, init="Huang", n_init=20, gamma=peso, max_iter=100, random_state=2026)
    modelo.fit(mixtos.to_numpy(), categorical=[2, 3, 4])
    modelos_proto.append(modelo)
    costo_proto.append(modelo.cost_)
kprototipos = modelos_proto[2]
grupo_proto = kprototipos.labels_ + 1
cp = {}
for g in range(1, 4):
    sub = mixtos[grupo_proto == g]
    cen = kprototipos.cluster_centroids_[g - 1]
    cp[str(g)] = dict(n=int(len(sub)), ceniza=R6(float(cen[0])), matiz=R6(float(cen[1])), alcohol=str(cen[2]), flavonoides=str(cen[3]),
                      color=str(cen[4]), color_n={c: int((sub.color == c).sum()) for c in ("alto", "medio", "bajo")},
                      media_ceniza=R6(sub.ceniza.mean()), media_matiz=R6(sub.matiz.mean()))
sens = {}
for gm in (0.25, 1, 4):
    m_ = KPrototypes(n_clusters=3, init="Huang", n_init=20, gamma=gm, max_iter=100, random_state=2026)
    m_.fit(mixtos.to_numpy(), categorical=[2, 3, 4])
    sens[str(gm)] = dict(costo=R6(m_.cost_), tamanos=[int(v) for v in np.bincount(m_.labels_)], labels=[int(v) for v in m_.labels_],
                         matiz_centros=[R6(float(c[1])) for c in m_.cluster_centroids_])
D["kproto"] = dict(costo=L6(costo_proto), grupos=cp, sens={k_: {kk: vv for kk, vv in v.items() if kk != "labels"} for k_, v in sens.items()},
                   ari_025_1=R6(adjusted_rand_score(sens["0.25"]["labels"], sens["1"]["labels"])),
                   ari_1_4=R6(adjusted_rand_score(sens["1"]["labels"], sens["4"]["labels"])),
                   iguales_1=bool(adjusted_rand_score(sens["1"]["labels"], kprototipos.labels_) > 0.999999),
                   ceniza=L6(mixtos.ceniza), matiz=L6(mixtos.matiz), labels={k_: v["labels"] for k_, v in sens.items()})

# ------------------------------------------------------------ 3. calculos adicionales
# 3a. iteraciones de Lloyd en dos variables estandarizadas (flavonoides, color), k = 3, desde tres vinos fijos
Z2 = z[["flavonoides", "color"]].to_numpy()
ini = [141, 117, 4]
cen = Z2[ini].copy()
hist = []
for it in range(1, 30):
    d2 = ((Z2[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
    lab = d2.argmin(axis=1)
    wss_asig = float(d2[np.arange(len(Z2)), lab].sum())
    nuevo = np.array([Z2[lab == g].mean(axis=0) for g in range(3)])
    wss_nuevo = float(((Z2 - nuevo[lab]) ** 2).sum())
    hist.append(dict(it=it, centros=[L6(c) for c in cen], tamanos=[int((lab == g).sum()) for g in range(3)], wss_asig=R6(wss_asig),
                     wss_tras=R6(wss_nuevo)))
    if np.allclose(nuevo, cen):
        break
    cen = nuevo
D["iter2d"] = dict(ini=ini, codigos=[str(vinos.codigo.iloc[i]) for i in ini], hist=hist, x=L6(Z2[:, 0]), y=L6(Z2[:, 1]),
                   lab_final=[int(v) for v in lab])
# 3b. inicios aleatorios sueltos sobre las 13 variables, k = 3 (diapositiva 15: 50 inicios)
ws = [KMeans(n_clusters=3, init="random", n_init=1, max_iter=100, random_state=s, algorithm="lloyd", tol=1e-8).fit(z).inertia_
      for s in range(50)]
D["init"] = dict(wss=L6(ws), min=R6(min(ws)), max=R6(max(ws)), n_min=int(sum(abs(v - min(ws)) < 1e-6 for v in ws)),
                 n_distintos=int(len({round(v, 4) for v in ws})), mejor_n_init50=R6(G["kmedias"].inertia_))
from collections import Counter  # noqa: E402
cnt = Counter(round(v, 2) for v in ws)
D["init"]["frecuencias"] = sorted([[k_, v] for k_, v in cnt.items()], key=lambda t: -t[1])
# 3c. ejemplos simulados
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
Xb, yb = make_blobs(n_samples=90, centers=[[-2.5, 2.0], [2.5, 2.5], [0.0, -2.0]], cluster_std=0.85, random_state=2026)
kb = KMeans(n_clusters=3, n_init=20, random_state=2026).fit(Xb)
D["sim_blobs"] = dict(x=L6(Xb[:, 0]), y=L6(Xb[:, 1]), lab=[int(v) for v in kb.labels_], cx=L6(kb.cluster_centers_[:, 0]),
                      cy=L6(kb.cluster_centers_[:, 1]))
Xm, ym = make_moons(n_samples=200, noise=0.07, random_state=2026)
km_ = KMeans(n_clusters=2, n_init=20, random_state=2026).fit(Xm)
acc = max(np.mean(km_.labels_ == ym), np.mean(km_.labels_ != ym))
D["sim_lunas"] = dict(x=L6(Xm[:, 0]), y=L6(Xm[:, 1]), real=[int(v) for v in ym], lab=[int(v) for v in km_.labels_],
                      acuerdo=R6(acc), cx=L6(km_.cluster_centers_[:, 0]), cy=L6(km_.cluster_centers_[:, 1]))
# atipico: dos nubes y un punto lejano, k = 2 -> el atipico se queda con un grupo para el solo
Xa, ya = make_blobs(n_samples=80, centers=[[-2.0, 0.0], [2.0, 0.0]], cluster_std=0.6, random_state=7)
Xa = np.vstack([Xa, [[25.0, 0.0]]])
ka = KMeans(n_clusters=2, n_init=20, random_state=2026).fit(Xa)
D["sim_atipico"] = dict(x=L6(Xa[:, 0]), y=L6(Xa[:, 1]), lab=[int(v) for v in ka.labels_], tamanos=sorted(int(v) for v in np.bincount(ka.labels_)),
                        cx=L6(ka.cluster_centers_[:, 0]), cy=L6(ka.cluster_centers_[:, 1]))
(AQUI / "datos39.json").write_text(json.dumps(D, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
brev = {k_: v for k_, v in D.items() if k_ not in ("pca", "iter2d", "sim_blobs", "sim_lunas", "sim_atipico")}
brev["kproto"] = {k_: v for k_, v in D["kproto"].items() if k_ not in ("ceniza", "matiz", "labels")}
print(json.dumps(brev, ensure_ascii=False, default=float)[:6000])
