# Clase 39: K-means y numero de grupos
# Pedro Luiz Ramos - 29 de septiembre de 2026
# Paquetes: numpy, pandas, matplotlib y scikit-learn.

# %% 1. Leer los datos
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.chdir(r"D:\codigos\clustering")
vinos = pd.read_csv("vinos.csv")
print(vinos.head())
print(vinos.shape)

# %% 2. Estandarizar
# Codigo identifica al vino; no es una variable para formar los grupos.
x = vinos.drop(columns="codigo")
# ddof=1 usa la misma desviacion estandar muestral que scale() de R.
z = (x - x.mean()) / x.std(ddof=1)
print(z.head().round(2))

# %% 3. Ajustar K-means para distintos numeros de grupos
# n_init=50 prueba 50 inicios y conserva el de menor WSS para cada k.
valores_k = range(1, 9)
modelos = []
wss = []
silueta = []

for k in valores_k:
    modelo = KMeans(n_clusters=k, n_init=50, max_iter=100,
                    random_state=2026, algorithm="lloyd", tol=1e-8)
    modelo.fit(z)
    modelos.append(modelo)
    wss.append(modelo.inertia_)

    # La silueta compara grupos, por eso no se calcula con k = 1.
    if k > 1:
        silueta.append(silhouette_score(z, modelo.labels_))
    else:
        silueta.append(np.nan)

comparacion = pd.DataFrame({"k": valores_k, "WSS": wss, "Silueta": silueta})
print(comparacion.round(4).to_string(index=False))

# %% 4. Elegir k: codo y silueta
# En el codo buscamos donde la reduccion del WSS comienza a ser menor.
# Para elegir automaticamente, usamos la mayor silueta entre k = 2 y k = 8.
fila_mejor = comparacion["Silueta"].idxmax()
k_silueta = int(comparacion.loc[fila_mejor, "k"])
print("k recomendado por silueta:", k_silueta)

fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
ejes[0].plot(valores_k, wss, "o-", color="steelblue")
ejes[0].set(xlabel="Numero de grupos (k)", ylabel="WSS", title="Codo")
ejes[1].plot(range(2, 9), silueta[1:], "o-", color="darkorange")
ejes[1].set(xlabel="Numero de grupos (k)", ylabel="Silueta media", title="Silueta")
ejes[1].axvline(k_silueta, linestyle="--", color="grey")
ejes[1].scatter(k_silueta, silueta[k_silueta - 1], color="darkgreen", s=80, zorder=3)
plt.tight_layout()
plt.show()

# %% 5. Revisar la solucion elegida
# Recuperamos el ajuste ya calculado. Las listas de Python parten en cero.
k_elegido = k_silueta
kmedias = modelos[k_elegido - 1]
grupo = kmedias.labels_ + 1

print("\nCantidad de vinos por grupo:")
print(pd.Series(grupo).value_counts().sort_index())

# Los centroides estan en escala estandarizada; 1 significa una desviacion
# estandar sobre la media general de esa variable.
centros = pd.DataFrame(kmedias.cluster_centers_, columns=z.columns,
                       index=range(1, k_elegido + 1))
centros.index.name = "Grupo"
print("\nCentroides estandarizados:")
print(centros.round(2))
tss = np.sum((z.to_numpy() - z.to_numpy().mean(axis=0)) ** 2)
print("WSS:", round(kmedias.inertia_, 3))
print("BSS:", round(tss - kmedias.inertia_, 3))
print("TSS:", round(tss, 3))

# Las medias en unidades originales ayudan a describir cada grupo.
perfiles = x.groupby(grupo).mean()
perfiles.index.name = "Grupo"
print("\nMedias en unidades originales:")
print(perfiles.round(2))

# %% 6. Graficar los grupos
# Mostramos dos variables, pero K-means utilizo las 13 variables estandarizadas.
# Los numeros de grupo son etiquetas y pueden cambiar entre R y Python.
colores = plt.get_cmap("tab10")
plt.figure(figsize=(7, 5))
for g in range(1, k_elegido + 1):
    filas = grupo == g
    plt.scatter(vinos.loc[filas, "alcohol"], vinos.loc[filas, "flavonoides"],
                color=colores(g - 1), label=f"Grupo {g}", s=25)
plt.xlabel("Alcohol")
plt.ylabel("Flavonoides")
plt.title("Grupos de K-means")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.boxplot([vinos.loc[grupo == g, "alcohol"] for g in range(1, k_elegido + 1)])
plt.xlabel("Grupo")
plt.ylabel("Alcohol")
plt.title("Alcohol por grupo")
plt.tight_layout()
plt.show()

# %% 7. Comparar los grupos con las clases reales
# Recuperamos las clases de UCI en el mismo orden de los vinos.
# No se usaron para ajustar K-means ni para elegir k.
etiquetas = pd.read_csv("vinos_clases_reales.csv").set_index("codigo")
clase_real = vinos["codigo"].map(etiquetas["clase_real"]).to_numpy()
tabla = pd.crosstab(clase_real, grupo, rownames=["Clase real"], colnames=["Grupo"])
print(tabla)

# Los numeros de grupo son arbitrarios. Coloreamos cada grupo segun
# su clase mayoritaria para poder comparar los dos graficos.
clase_del_grupo = tabla.idxmax(axis=0)
clase_asignada = pd.Series(grupo).map(clase_del_grupo).to_numpy()
discrepa = clase_asignada != clase_real
coincidencias = np.sum(~discrepa)
pureza = np.mean(~discrepa)
print("Coincidencias con la clase mayoritaria:", coincidencias, "de", len(vinos))
print("Pureza de los grupos:", round(100 * pureza, 1), "%")
# Es una comparacion descriptiva de estos vinos, no exactitud en datos nuevos.

# PCA resume las 13 variables en dos ejes para dibujar los mismos vinos.
# No volvemos a ajustar K-means sobre estos dos componentes.
# Los signos de los ejes pueden invertirse entre R y Python sin cambiar los grupos.
pca = PCA(n_components=2)
coordenadas = pca.fit_transform(z)
varianza = 100 * pca.explained_variance_ratio_
paleta = ["#4067A0", "#C66A25", "#247D72"]

fig, ejes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
for c in range(1, 4):
    filas = clase_real == c
    ejes[0].scatter(coordenadas[filas, 0], coordenadas[filas, 1],
                    color=paleta[c - 1], label=f"Clase {c}", s=30)
for g in range(1, k_elegido + 1):
    filas = grupo == g
    c = int(clase_del_grupo[g])
    ejes[1].scatter(coordenadas[filas, 0], coordenadas[filas, 1],
                    color=paleta[c - 1], label=f"Grupo {g} (clase {c})", s=30)
ejes[1].scatter(coordenadas[discrepa, 0], coordenadas[discrepa, 1],
                facecolors="none", edgecolors="black", s=110, linewidths=1.5)
ejes[0].set_title("Clases reales de los vinos", pad=36)
ejes[1].set_title("Grupos encontrados por K-means", pad=36)
for eje in ejes:
    eje.set_xlabel(f"PC1 ({varianza[0]:.1f}% de varianza)")
    eje.set_ylabel(f"PC2 ({varianza[1]:.1f}% de varianza)")
    eje.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, fontsize=8.5, frameon=False)
fig.suptitle(f"Coincidencia con clase mayoritaria: {coincidencias}/{len(vinos)} vinos "
             f"({100 * pureza:.1f}%).")
fig.text(0.5, 0.02, "Colores por clase mayoritaria; circulos negros: discrepancias.",
         ha="center", fontsize=10)
plt.tight_layout(rect=(0, 0.05, 1, 0.94))
plt.show()

# %% 8. Complemento: elegir k con Gap
# Comparamos con 30 bases simuladas sin grupos, uniformes en el rango
# de cada columna de z. Este bloque puede demorar mas que los anteriores.
rng = np.random.default_rng(2026)
B = 30
log_wss_ref = np.zeros((B, len(valores_k)))

for b in range(B):
    referencia = rng.uniform(z.min().to_numpy(), z.max().to_numpy(), size=z.shape)
    for k in valores_k:
        modelo = KMeans(n_clusters=k, n_init=20, max_iter=100,
                        random_state=2026 + b, algorithm="lloyd", tol=1e-8)
        modelo.fit(referencia)
        log_wss_ref[b, k - 1] = np.log(modelo.inertia_)

gap = log_wss_ref.mean(axis=0) - np.log(wss)
error_gap = log_wss_ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)
resultado_gap = pd.DataFrame({"k": valores_k, "Gap": gap, "Error": error_gap})
print(resultado_gap.round(3).to_string(index=False))

# Elegimos el primer k con Gap(k) >= Gap(k+1) - Error(k+1).
# Si ninguno cumple antes, conservamos el mayor k evaluado.
k_gap = max(valores_k)
for k in range(1, max(valores_k)):
    if gap[k - 1] >= gap[k] - error_gap[k]:
        k_gap = k
        break
print("k recomendado por Gap:", k_gap)

plt.figure(figsize=(7, 5))
plt.errorbar(valores_k, gap, yerr=error_gap, fmt="o-", capsize=4, color="steelblue")
plt.axvline(k_gap, linestyle="--", color="darkorange")
plt.xlabel("Numero de grupos (k)")
plt.ylabel("Gap")
plt.title("Estadistico Gap")
plt.tight_layout()
plt.show()

print(pd.DataFrame({"Criterio": ["Silueta", "Gap"],
                    "k": [k_silueta, k_gap]}).to_string(index=False))
# Los criterios pueden discrepar: no hay un k verdadero garantizado.
# Para los perfiles y graficos anteriores usamos la solucion de silueta.
