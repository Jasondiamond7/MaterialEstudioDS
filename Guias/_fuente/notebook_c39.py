"""Genera el notebook guiado de clustering (Clase 39).

Uso normal:      python notebook_c39.py             -> Guias/notebooks/C39_Clustering.ipynb
Modo de prueba:  C39_TEST=1 python notebook_c39.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c39_numeros.json (lo escribe figuras_c39.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib, scipy, scikit-learn y kmodes.
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C39_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C39_Clustering_test.ipynb" if TEST else "C39_Clustering.ipynb")
N = json.loads((HERE / "c39_numeros.json").read_text(encoding="utf-8"))
cells = []


def _get(obj, parts):
    if not parts:
        return obj
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if isinstance(obj, list):
            if i == 1 and parts[0].isdigit():
                return _get(obj[int(parts[0])], parts[1:])
        elif k in obj:
            return _get(obj[k], parts[i:])
    raise KeyError(parts)


def val(m):
    obj = _get(N, m.group(1).split("."))
    d = m.group(2)
    return str(obj) if d is None else f"{float(obj):.{int(d)}f}"


def fill(text):
    return re.sub(r"\[\[([\w.]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- encabezado
md("""
# Clase 39 · Clustering: K-means, K-modes y K-prototypes
**Diplomado Data Science** · Prof. Pedro Luiz Ramos · retoma K-means y PCA (Clases 31 y 32), prepara la pregunta 4 del Taller 3 y se conecta con DBSCAN y Mahalanobis (Clase 41)

Material de apoyo:
- `Guias/C39_Guia_Clustering.pdf` — guía visual de la clase.
- `Guias/apuntes/C39_Apuntes_Clustering.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C39_Ejercicios_Cuaderno_Clustering.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta «Clase 39 …»): `clase_39_clustering.pdf` (diapositivas), `kmeans_numero_clusters.py` y `.R.txt`, `vinos.csv` y `vinos_clases_reales.csv`.
- Transcripción: `VideosClases/Clase 39 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` y `kmodes` (para K-modes y K-prototypes). Si no los tienes:
```
pip install numpy pandas matplotlib scipy scikit-learn kmodes
```

**Datos**: `vinos.csv` (177 vinos, columna `codigo` y 13 mediciones químicas) y `vinos_clases_reales.csv` (la clase UCI de cada vino: **solo para validar al final**, nunca para ajustar). El notebook busca los archivos en la carpeta de la Clase 39 **sin modificarla** (los scripts del profesor usan `os.chdir("D:/codigos/clustering")`; aquí no hace falta).

### Objetivos
1. Entender que **agrupar no es clasificar**: no hay Y y el número del grupo es solo una etiqueta.
2. Hacer a mano y en código el ciclo de K-means (asignar, promediar) y el costo **WSS = dentro**, con **TSS = WSS + BSS**.
3. **Reproducir** los resultados del profesor: WSS 1267.37 con k = 3, silueta 0.284, Gap con k = 4, perfiles y comparación con las clases reales.
4. Saber **por qué estandarizar** y **por qué repetir con varios inicios**.
5. Elegir k con **codo, silueta y Gap** y decidir cuando discrepan.
6. Extender a **K-modes** y **K-prototypes**, y conocer el jerárquico y los límites de K-means.
7. Traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 10 | A1 · Datos y estandarización |
| A | 15 | A2 · Una iteración de K-means a mano |
| A | 10 | A3 · WSS, BSS y TSS |
| A | 15 | A4 · K-means con las 13 variables |
| A | 15 | A5 · La inicialización importa |
| A | 10 | A6 · ¿Qué pasa sin estandarizar? |
| B | 10 | B1 · Codo |
| B | 20 | B2 · Silueta |
| B | 20 | B3 · Gap |
| B | 10 | B4 · Decidir k cuando los criterios discrepan |
| C | 15 | C1 · Perfiles de los grupos |
| C | 20 | C2 · Contingencia, pureza y ARI con la clase real |
| C | 10 | C3 · ¿Agrupar con el PCA? |
| C | 20 | C4 · Plantilla: K-means con k = 2 a 4 (puente con el Taller 3) |
| D | 15 | D1 · K-modes |
| D | 20 | D2 · K-prototypes y el peso γ |
| D | 15 | D3 · Complemento: agrupamiento jerárquico |
| D | 10 | D4 · Límites de K-means |
| D | 10 | D5 · Alertas del material, verificadas con código |
| D | 10 | D6 · Tabla de correspondencia R ↔ Python |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Lo aleatorio usa semilla fija; con otras versiones de las librerías puede variar la última cifra.
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import sklearn
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
    HAY_KMODES = True
except ImportError:
    HAY_KMODES = False
    print("Falta el paquete kmodes: instálalo con  pip install kmodes  para la Parte D (D1 y D2).")

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 2026   # semilla para todo lo aleatorio
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso (Clases 31, 32 y 38)
- **K-means** (Clase 31): elegir k centros, asignar cada punto al más cercano, recalcular cada centro como la **media** de su grupo y repetir.
- **Estandarizar** = restar la media y dividir por la desviación **muestral** (`ddof=1`, como `scale()` de R y `std()` de pandas). `StandardScaler` usa `ddof=0`: cambia el WSS en la tercera cifra, no los grupos.
- **PCA** (Clase 32): sirve aquí para **dibujar** en dos ejes; el agrupamiento se ajusta con las 13 variables.
- **Matriz de contingencia** (Clase 38): `pd.crosstab` entre la clase real y el grupo.

### Funciones comunes
""")
code("""
def buscar_clase():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 39*/vinos.csv"):
            return p.parent
        if (base / "vinos.csv").exists():
            return base
    raise FileNotFoundError("No encuentro vinos.csv: abre el notebook dentro de la carpeta del curso.")


CARPETA = buscar_clase()


def kmeans_prof(Z, k, n_init=50, seed=RS):
    # los mismos parámetros del script del profesor (diapositiva 26)
    return KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=seed, algorithm="lloyd", tol=1e-8).fit(Z)


def tss_wss_bss(Z, etiquetas):
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    tss = float(((Z - Z.mean(axis=0)) ** 2).sum())
    wss = float(sum(((Z[etiquetas == g] - Z[etiquetas == g].mean(axis=0)) ** 2).sum() for g in np.unique(etiquetas)))
    return tss, wss, tss - wss


def purezas(clase, grupo):
    t = pd.crosstab(clase, grupo)
    return t, float(t.max(axis=0).sum() / len(clase))


print("Material:", CARPETA.name)
""")

# ================================================================== PARTE A
md("""
## Parte A · K-means: algoritmo, costo e inicialización

### A1 · Datos y estandarización (10 min)
`vinos.csv` tiene `codigo` (el identificador **no** participa) y 13 mediciones. Las clases reales viven en otro archivo y se unen por `codigo`, pero **no se usan** para ajustar. La diapositiva 7 da un ejemplo: alcohol de 14, media 13 y desviación 0.8 → `z = (14 − 13)/0.8 = 1.25`.
""")
code("""
vinos = pd.read_csv(CARPETA / "vinos.csv")
reales = pd.read_csv(CARPETA / "vinos_clases_reales.csv").set_index("codigo")
clase_real = vinos["codigo"].map(reales["clase_real"]).to_numpy()
x = vinos.drop(columns="codigo")
z = (x - x.mean()) / x.std(ddof=1)          # igual que scale() de R
print(vinos.shape, "| clases reales (1, 2, 3):", np.bincount(clase_real)[1:])
print("media ≈ 0:", bool(np.allclose(z.mean(), 0)), "| desviación = 1:", bool(np.allclose(z.std(ddof=1), 1)))
print("z de la diapositiva 7:", (14 - 13) / 0.8)
display(z.head().round(2))
z_ss = StandardScaler().fit_transform(x)     # StandardScaler divide por n, no por n - 1
print("razón entre las dos escalas:", round(float(z.iloc[0, 0] / z_ss[0, 0]), 5), "= sqrt(177/176) =", round(float(np.sqrt(177 / 176)), 5))
""")
md("""
Los cuatro puntos a recordar: (1) `codigo` no es una variable; (2) el z usa la desviación **muestral**; (3) con `StandardScaler` los números cambian ligeramente (razón √(177/176)) pero no los grupos; (4) estandarizar **no elimina atípicos ni correlaciones**.

### ✍️ Ejercicio A1
Estandariza a mano la columna `prolina` (media y desviación con `ddof=1`) y comprueba que coincide con `z["prolina"]`. ¿Cuánto vale el z del primer vino?
""")
code("""
# tu código aquí
""")
solution("""
pr = x["prolina"]
z_pr = (pr - pr.mean()) / pr.std(ddof=1)
print(np.allclose(z_pr, z["prolina"]), "| primer vino:", round(float(z_pr.iloc[0]), 3), "| prolina original:", pr.iloc[0])
""", "El primer vino (prolina 1065) tiene z = 1.005: está algo más de una desviación estándar sobre la media.")

md("""
### A2 · Una iteración de K-means a mano (15 min)
Diapositiva 13: datos `1, 2, 3, 8, 9, 10`, `k = 2` y centros iniciales `1` y `8`. Primero **asignar** (distancia al cuadrado), luego **recalcular** (media del grupo). El algoritmo termina cuando ninguna asignación cambia.
""")
code("""
def lloyd(datos, centros, tope=20):
    datos = np.asarray(datos, float)
    c = np.array(centros, float)
    previa = None
    for it in range(1, tope + 1):
        d2 = (datos[:, None] - c[None, :]) ** 2          # distancia al cuadrado a cada centro
        grupo = d2.argmin(axis=1)
        nuevos = np.array([datos[grupo == g].mean() for g in range(len(c))])
        print(f"iteración {it}: grupos {grupo + 1} | centros {np.round(c, 3)} -> {np.round(nuevos, 3)}")
        if previa is not None and np.array_equal(grupo, previa):
            print("sin cambios: convergió")
            return grupo, nuevos
        previa, c = grupo, nuevos
    return grupo, nuevos


dato_toy = np.array([1, 2, 3, 8, 9, 10])
grupo_toy, centros_toy = lloyd(dato_toy, [1, 8])
print("distancias al cuadrado al centro 1:", ((dato_toy - 1) ** 2).tolist())
print("distancias al cuadrado al centro 8:", ((dato_toy - 8) ** 2).tolist())
""")
md("""
Coincide con la diapositiva 13: distancias `0, 1, 4, 49, 64, 81` al centro 1 y `49, 36, 25, 0, 1, 4` al centro 8; nuevos centros **2 y 9**. En la segunda vuelta nada cambia. En un caso real los puntos de partida se eligen al azar y el algoritmo puede tardar más.

### ✍️ Ejercicio A2
Aplica `lloyd` a `[4, 6, 7, 15, 17, 25]` con centros iniciales `6` y `17`. ¿Cuántas iteraciones hacen falta y cuáles son los centros finales?
""")
code("""
# tu código aquí
""")
solution("""
g_a2, c_a2 = lloyd([4, 6, 7, 15, 17, 25], [6, 17])
print("centros finales:", np.round(c_a2, 3))
""", "Dos iteraciones (la segunda confirma que no cambia nada): grupos {4, 6, 7} y {15, 17, 25}; centros 5.667 y 19.")

md("""
### A3 · WSS, BSS y TSS (10 min)
`WSS = Σ (x − μ_grupo)²` (dentro), `BSS = Σ n_g (μ_g − media global)²` (entre grupos) y `TSS = Σ (x − media global)²`. Se cumple `TSS = WSS + BSS`. **Ojo**: son distancias **al cuadrado**; en clase se dijo «sacamos la raíz», pero el WSS no la lleva (con ceros y unos sale igual).
""")
code("""
tss_t, wss_t, bss_t = tss_wss_bss(dato_toy, grupo_toy)
print(f"TSS = {tss_t}, WSS = {wss_t}, BSS = {bss_t}  ->  WSS + BSS = {wss_t + bss_t}")
print("BSS a mano:", 3 * (2 - 5.5) ** 2 + 3 * (9 - 5.5) ** 2)
print("TSS por partes:", [(v - dato_toy.mean()) ** 2 for v in dato_toy])
""")
md("""
Diapositiva 17: `WSS = 4`, `BSS = 36.75 + 36.75 = 73.5` y `TSS = 77.5`. Al fijar los datos y la escala, **reducir el WSS equivale a aumentar el BSS**: por eso k grande siempre mejora el WSS (con `k = n` vale 0) y hace falta otro criterio para elegir k.

### ✍️ Ejercicio A3
Con los datos `[4, 6, 7, 15, 17, 25]` y la partición del ejercicio A2, calcula TSS, WSS y BSS y la fracción BSS/TSS.
""")
code("""
# tu código aquí
""")
solution("""
dat_a3 = np.array([4, 6, 7, 15, 17, 25])
tss_a, wss_a, bss_a = tss_wss_bss(dat_a3, g_a2)
print(round(tss_a, 3), round(wss_a, 3), round(bss_a, 3), "| BSS/TSS =", round(bss_a / tss_a, 3))
""", "TSS = 327.333 (alrededor de la media 12.33), WSS = 60.667 y BSS = 266.667: la partición explica el 81.5 % de la variabilidad.")

md("""
### A4 · K-means con las 13 variables (15 min)
Es el ajuste del script del profesor: `KMeans(n_clusters=k, n_init=50, max_iter=100, random_state=2026, algorithm="lloyd", tol=1e-8)`. Con `k = 3` se reproducen las diapositivas 27 y 28. `inertia_` es el WSS; `labels_` va de 0 a 2, por eso `+ 1`.
""")
code("""
km3 = kmeans_prof(z, 3)
grupo = km3.labels_ + 1
tss, wss, bss = tss_wss_bss(z.to_numpy(), grupo)
print(f"WSS = {km3.inertia_:.3f} (a mano {wss:.3f}) | BSS = {bss:.3f} | TSS = {tss:.1f} | iteraciones: {km3.n_iter_}")
print("tamaños:", np.bincount(grupo)[1:])
centros = pd.DataFrame(km3.cluster_centers_, columns=z.columns, index=[1, 2, 3]).round(2)
centros.index.name = "Grupo"
display(centros)
perfiles = x.groupby(grupo).mean().round(2)
perfiles.index.name = "Grupo"
display(perfiles[["alcohol", "flavonoides", "color"]])
""")
md("""
Resultados de las diapositivas 27 y 28: **WSS = [[codo.wss_k3|2]]**, **BSS = [[wsbs.vinos.bss.2|2]]**, **TSS = 2 288**; grupos de **[[perfiles.n.0]], [[perfiles.n.1]] y [[perfiles.n.2]]** vinos; alcohol 13.68, 13.14 y 12.25. Observa que K-means converge en pocas iteraciones ([[pca.n_iter]] aquí): en clase se habló de «50 mil iteraciones», pero con estos datos bastan unas cuantas.

**Numeración**: los números de grupo son etiquetas. En R, el grupo 1 es el grupo 3 de Python (65 vinos), el 2 es el 2 y el 3 es el 1 (diapositiva 27). La transcripción mezcla las dos numeraciones.

### ✍️ Ejercicio A4
Ajusta `k = 4` con la misma función. ¿Cuánto vale el WSS y cuántas iteraciones hizo? ¿Cuánto bajó respecto de `k = 3`?
""")
code("""
# tu código aquí
""")
solution("""
km4 = kmeans_prof(z, 4)
print(f"WSS(k=4) = {km4.inertia_:.2f}, iteraciones = {km4.n_iter_}, baja {km3.inertia_ - km4.inertia_:.2f} ({(1 - km4.inertia_ / km3.inertia_) * 100:.1f} %)")
""", "WSS(k = 4) = 1164.91: baja 102.47, un 8.1 % respecto de k = 3.")

md("""
### A5 · La inicialización importa (15 min)
K-means puede caer en **mínimos locales**: el resultado depende de los centros iniciales. Diapositiva 15: mismos datos y `k = 3`, distintos puntos de partida. La solución práctica es **repetir y conservar el menor WSS** (`n_init` en Python, `nstart` en R). Ojo: el criterio es el **menor WSS**, no «el clúster que más aparece».
""")
code("""
wss_uno = np.array([KMeans(3, init="random", n_init=1, max_iter=100, algorithm="lloyd", tol=1e-8, random_state=s).fit(z).inertia_ for s in range(25)])
print("WSS de 25 ajustes de un solo inicio (redondeado):", sorted(set(np.round(wss_uno, 2))))
print(f"mínimo {wss_uno.min():.2f} | máximo {wss_uno.max():.2f} | óptimo de n_init = 50: {km3.inertia_:.2f} | ajustes que lo alcanzan: {int((np.round(wss_uno, 2) == round(km3.inertia_, 2)).sum())} de 25")
plt.scatter(range(1, 26), wss_uno, color="steelblue"); plt.axhline(km3.inertia_, color="firebrick", ls="--")
plt.xlabel("inicialización"); plt.ylabel("WSS final"); plt.title("Un solo inicio: mínimos locales"); plt.show()
""")
md("""
Con inicios aleatorios hay [[init.n_unicos_random]] valores distintos de WSS, entre [[init.min_random|2]] y [[init.max_random|2]] (el peor, un [[init.dif_pct_peor|1]] % mayor); solo [[init.n_optimo_random]] de 25 alcanzan el óptimo. Con `k-means++` (el valor por defecto de scikit-learn) los inicios son mejores, pero **tampoco garantizan** el óptimo con un solo ajuste.

### ✍️ Ejercicio A5
Para `n_init = 1, 5 y 20` repite el ajuste (`init="random"`) con 20 semillas distintas y calcula la mediana del WSS que se conserva. ¿Desde qué `n_init` casi nunca se cae en un mínimo local malo?
""")
code("""
# tu código aquí
""")
solution("""
for ni in (1, 5, 20):
    w_ni = [KMeans(3, init="random", n_init=ni, max_iter=100, algorithm="lloyd", tol=1e-8, random_state=s).fit(z).inertia_ for s in range(20)]
    print(f"n_init = {ni:2d}: mediana {np.median(w_ni):.2f} | máximo {np.max(w_ni):.2f}")
""", "Con 20 semillas: n_init = 1 da mediana 1268.33 y máximo 1271.70; n_init = 5, mediana 1267.37 y máximo 1269.19; n_init = 20, siempre el óptimo 1267.37. (En las 25 corridas de arriba hubo un ajuste de 1 604: con un solo inicio puede ocurrir.) Con k grande conviene subir n_init.")

md("""
### A6 · ¿Qué pasa sin estandarizar? (10 min)
Si una variable tiene unidades mucho mayores, domina las distancias. Aquí la **prolina** tiene varianza ~99 000 frente a ~0.05–200 de las demás. Comparamos el agrupamiento con la clase real mediante el **índice de Rand ajustado (ARI)**: 1 = particiones iguales, ≈ 0 = como al azar.
""")
code("""
varianzas = x.var(ddof=1).sort_values(ascending=False)
print((varianzas / varianzas.sum() * 100).round(2).head(4).to_string())
km_raw = kmeans_prof(x.to_numpy(), 3)
print(f"ARI con la clase real | sin estandarizar: {adjusted_rand_score(clase_real, km_raw.labels_):.3f} | estandarizando: {adjusted_rand_score(clase_real, km3.labels_):.3f}")
print("tamaños sin estandarizar:", np.bincount(km_raw.labels_))
""")
md("""
La prolina aporta el **[[estandarizar.cuota_prolina|4]]** de la varianza total (≈ [[estandarizar.var.prolina|0]] frente a 204 del magnesio): sin estandarizar, K-means casi solo ve prolina y el ARI cae de [[estandarizar.ari_z|3]] a [[estandarizar.ari_raw|3]]. Estandarizar **no garantiza** buenos grupos, pero evita este problema.

### ✍️ Ejercicio A6
Estandariza solo `prolina` y `magnesio` (deja las demás sin estandarizar) y mide el ARI. ¿Mejora? ¿Qué variable pasa a dominar?
""")
code("""
# tu código aquí
""")
solution("""
parcial = x.copy()
for c_ in ("prolina", "magnesio"):
    parcial[c_] = (x[c_] - x[c_].mean()) / x[c_].std(ddof=1)
print("ARI:", round(adjusted_rand_score(clase_real, kmeans_prof(parcial.to_numpy(), 3).labels_), 3))
print("varianzas:", parcial.var(ddof=1).sort_values(ascending=False).round(2).head(3).to_dict())
""", "Ahora dominan la alcalinidad (varianza 11.2) y el color (5.3); el ARI sube de 0.375 a 0.456, pero queda lejos de 0.897: hay que estandarizar **todas** las variables.")

# ================================================================== PARTE B
md("""
## Parte B · Elegir el número de grupos

El WSS **siempre baja** al subir k. Hacen falta criterios adicionales: **codo** (WSS), **silueta** (separación) y **Gap** (comparación con datos sin estructura). Ningún criterio es una prueba formal.

### B1 · Codo (10 min)
Ajustamos k = 1 a 8 (como el script) y miramos dónde la reducción del WSS empieza a ser pequeña.
""")
code("""
valores_k = range(1, 9)
modelos = [kmeans_prof(z, k) for k in valores_k]
wss_k = np.array([m.inertia_ for m in modelos])
tabla_codo = pd.DataFrame({"k": list(valores_k), "WSS": wss_k.round(2), "baja": np.r_[np.nan, -np.diff(wss_k)].round(1), "baja_%": np.r_[np.nan, -np.diff(wss_k) / wss_k[:-1] * 100].round(1)})
display(tabla_codo)
plt.plot(valores_k, wss_k, "o-", color="teal"); plt.axvline(3, color="firebrick", ls="--")
plt.xlabel("número de grupos k"); plt.ylabel("WSS"); plt.title("Codo"); plt.show()
""")
md("""
La reducción cae de **[[codo.reduccion.1|0]]** (de 2 a 3 grupos, [[codo.pct.1|1]] %) a **[[codo.reduccion.2|0]]** (de 3 a 4, [[codo.pct.2|1]] %): el codo está en **k = 3**. Con k hasta 20 el WSS sigue bajando sin un mínimo: el codo es una lectura, no un resultado automático. El profesor comentó que ajustar 8 valores de k tarda unos 3 segundos (algoritmo muy optimizado).

### ✍️ Ejercicio B1
Calcula el WSS de `k = n = 177` (cada vino su propio grupo) con la definición. ¿Sirve como solución?
""")
code("""
# tu código aquí
""")
solution("""
etiquetas_n = np.arange(len(z))
print("WSS con k = n:", tss_wss_bss(z.to_numpy(), etiquetas_n)[1])
""", "Vale 0: cada punto es su propio centro. Es una «partición» perfecta en WSS pero no agrupa nada: por eso el WSS por sí solo no basta para elegir k.")

md("""
### B2 · Silueta (20 min)
Para cada vino: `a(i)` = distancia media a los **otros** vinos de su grupo; `b(i)` = menor distancia media a los vinos de **otro** grupo; `s(i) = (b − a)/máx(a, b)`. Cerca de 1: bien separado; cerca de 0: zona de superposición; negativo: más cerca de otro grupo. Se usa la distancia euclidiana (no su cuadrado). Diapositiva 20: a = 2 y b = 6 → s = 0.667.
""")
code("""
D = np.sqrt(((z.to_numpy()[:, None, :] - z.to_numpy()[None, :, :]) ** 2).sum(axis=2))     # distancias entre vinos
et = km3.labels_
a = np.array([D[i, et == et[i]].sum() / ((et == et[i]).sum() - 1) for i in range(len(z))])
b = np.array([min(D[i, et == g].mean() for g in range(3) if g != et[i]) for i in range(len(z))])
s_mano = (b - a) / np.maximum(a, b)
s_sk = silhouette_samples(z, et)
print("a mano = scikit-learn:", bool(np.allclose(s_mano, s_sk)), "| ejemplo de la clase:", round((6 - 2) / 6, 3))
print(f"silueta media: {s_sk.mean():.4f} | por grupo: {[round(float(s_sk[et == g].mean()), 3) for g in range(3)]} | vinos con s < 0: {int((s_sk < 0).sum())}")
medias_s = [silhouette_score(z, modelos[k - 1].labels_) for k in range(2, 9)]
print("silueta media para k = 2..8:", np.round(medias_s, 4))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
y0 = 0
for g in range(3):
    v_ = np.sort(s_sk[et == g]); ax[0].fill_betweenx(np.arange(y0, y0 + len(v_)), 0, v_); y0 += len(v_) + 5
ax[0].axvline(s_sk.mean(), color="firebrick", ls="--"); ax[0].set(title="Gráfico de silueta (k = 3)", xlabel="s(i)", yticks=[])
ax[1].plot(range(2, 9), medias_s, "o-", color="darkorange"); ax[1].set(title="Silueta media según k", xlabel="k"); plt.tight_layout(); plt.show()
""")
md("""
La silueta media es **[[silueta.media_k3|4]]** con k = 3 y es el **máximo** entre k = 2 y 8 (la diapositiva 21 dice 0.284). En clase se leyó «casi 0.36»: no es ese valor. Es un máximo **modesto**: hay superposición, sobre todo en el grupo 3 (media [[silueta.por_grupo.2|3]] y [[silueta.n_negativas]] vinos con s negativa). No hay silueta para k = 1 (falta un grupo vecino).

### ✍️ Ejercicio B2
¿Qué vino tiene la silueta más baja y cuánto valen su `a` y su `b`? ¿Cuántos vinos con s < 0 hay en cada grupo?
""")
code("""
# tu código aquí
""")
solution("""
i_peor = int(s_sk.argmin())
print(vinos.codigo[i_peor], f"| a = {a[i_peor]:.3f}, b = {b[i_peor]:.3f}, s = {s_sk[i_peor]:.4f}")
print({g + 1: int(((s_sk < 0) & (et == g)).sum()) for g in range(3)})
""", "El vino VIN79 tiene a = 4.233, b = 4.130 y s = −0.024; los 7 vinos con silueta negativa están todos en el grupo 3.")

md("""
### B3 · Gap (20 min)
El estadístico Gap compara el WSS observado con el de **datos simulados sin estructura** (uniformes en el rango de cada columna estandarizada): `Gap(k) = promedio log W*_k − log W_k`. La regla de la diapositiva 23: el **primer k** con `Gap(k) ≥ Gap(k+1) − s(k+1)`. Este bloque tarda unos segundos (30 bases × 8 valores de k). Es el código del profesor.
""")
code("""
def gap_estadistico(Z, B=30, semilla=RS, n_init=20, kmax=8):
    Z = np.asarray(Z, float)
    wss_ = np.array([kmeans_prof(Z, k).inertia_ for k in range(1, kmax + 1)])
    rng = np.random.default_rng(semilla)
    ref = np.zeros((B, kmax))
    for bb in range(B):
        R = rng.uniform(Z.min(axis=0), Z.max(axis=0), size=Z.shape)
        for k in range(1, kmax + 1):
            ref[bb, k - 1] = np.log(KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=semilla + bb, algorithm="lloyd", tol=1e-8).fit(R).inertia_)
    gap_ = ref.mean(axis=0) - np.log(wss_)
    err_ = ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)
    k_elegido = kmax
    for k in range(1, kmax):
        if gap_[k - 1] >= gap_[k] - err_[k]:
            k_elegido = k
            break
    return gap_, err_, k_elegido


gap, error_gap, k_gap = gap_estadistico(z)
display(pd.DataFrame({"k": range(1, 9), "Gap": gap.round(3), "Error": error_gap.round(3)}))
print("k según la regla de un error estándar:", k_gap, "| máximo del Gap en k =", int(gap.argmax()) + 1)
print(f"Gap(3) = {gap[2]:.4f} frente a Gap(4) - s(4) = {gap[3] - error_gap[3]:.4f}")
plt.errorbar(range(1, 9), gap, yerr=error_gap, fmt="o-", capsize=4); plt.axvline(k_gap, color="darkorange", ls="--")
plt.xlabel("k"); plt.ylabel("Gap"); plt.title("Estadístico Gap"); plt.show()
""")
md("""
La regla da **k = [[gap.k_gap]]** (como en las diapositivas 24 y 57, en R y en Python), mientras que codo y silueta dan 3. Es un resultado **ajustado**: Gap(3) = [[gap.gap_k3|4]] no alcanza Gap(4) − s(4) = [[gap.umbral_k3|4]]. Con otras 4 semillas y con B = 60 la regla sigue dando 4. El Gap sigue subiendo hasta k = 8: la regla de un error estándar existe justamente para no elegir siempre el máximo. En el ejemplo hipotético de la diapositiva 23 (Gap 0.40, 0.62, 0.64; s 0.04, 0.05, 0.04), 0.62 ≥ 0.64 − 0.04 → se elige k = 2.

### ✍️ Ejercicio B3
Repite `gap_estadistico` con `semilla = 1` y `B = 20`. ¿Cambia el k elegido? (Tarda unos 15 segundos.)
""")
code("""
# tu código aquí
""")
solution("""
g_b3, e_b3, k_b3 = gap_estadistico(z, B=20, semilla=1)
print("k elegido:", k_b3, "| Gap(3) - [Gap(4) - s(4)] =", round(float(g_b3[2] - (g_b3[3] - e_b3[3])), 4))
""", "Sigue dando k = 4: la regla es estable frente a la semilla y a B, aunque la diferencia en k = 3 es pequeña (≈ −0.006). Una referencia distinta (por ejemplo alineada con componentes principales) podría dar otro k.")

md("""
### B4 · Decidir k cuando los criterios discrepan (10 min)
Resumen de lo calculado: codo → 3; silueta → 3; Gap → 4. El profesor trabaja con **k = 3**: la mayoría de criterios, más interpretación; **k = 4 queda como alternativa**. Además de los tres criterios conviene revisar tamaños, estabilidad frente a la semilla y si los grupos se pueden describir.
""")
code("""
res_k = []
for k in (3, 4):
    m_k = modelos[k - 1]
    res_k.append(dict(k=k, WSS=round(m_k.inertia_, 2), silueta=round(silhouette_score(z, m_k.labels_), 4), tamaños=sorted(np.bincount(m_k.labels_).tolist(), reverse=True),
                      ARI_entre_semillas=round(adjusted_rand_score(m_k.labels_, kmeans_prof(z, k, seed=7).labels_), 3)))
display(pd.DataFrame(res_k))
""")
md("""
Con k = 3 los grupos son más nítidos (silueta 0.284 frente a 0.258) y coinciden con la estructura de tres clases conocida; con k = 4 hay un grupo extra. **Un k no se «descubre»**: se elige con criterios, estabilidad e interpretación. Si no hubiera una clase real para contrastar, la decisión sería más incierta.

### ✍️ Ejercicio B4
Escribe en un comentario tu regla de decisión para elegir k **antes** de mirar la clase real, y aplícala con los números anteriores.
""")
code("""
# tu código aquí
""")
solution("""
# Regla: elegir el k que maximiza la silueta media si el codo y el Gap no la contradicen de forma clara;
# si discrepan, preferir el k con más criterios a favor, grupos estables entre semillas y perfiles interpretables.
votos = {"codo": 3, "silueta": int(np.argmax(medias_s)) + 2, "gap": k_gap}
print(votos, "-> k =", max(set(votos.values()), key=list(votos.values()).count))
""", "Votos: codo 3, silueta 3, Gap 4 → k = 3 (mayoría), con k = 4 como alternativa. Documenta la regla antes de comparar con la clase real.")

# ================================================================== PARTE C
md("""
## Parte C · Interpretar y validar los grupos

### C1 · Perfiles de los grupos (15 min)
Un grupo se describe con su **perfil**: la media de cada variable, en la escala original (más fácil de leer) y en la estandarizada (comparable entre variables). Un valor positivo en la escala estandarizada indica una media del grupo sobre la media general. Los perfiles son medias: **no describen a cada vino**.
""")
code("""
cent_z = pd.DataFrame(km3.cluster_centers_, columns=z.columns, index=[1, 2, 3])
display(cent_z.T.round(2))
for g in (1, 2, 3):
    print(f"Grupo {g} (n = {int((grupo == g).sum())}): más alto en", list(cent_z.loc[g].sort_values(ascending=False).index[:3]), "| más bajo en", list(cent_z.loc[g].sort_values().index[:2]))
plt.figure(figsize=(8, 4)); plt.imshow(cent_z.T.to_numpy(), cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto"); plt.colorbar(label="media estandarizada")
plt.yticks(range(13), cent_z.columns); plt.xticks(range(3), ["Grupo 1", "Grupo 2", "Grupo 3"]); plt.grid(False); plt.title("Perfiles estandarizados"); plt.show()
""")
md("""
Lectura (diapositiva 29, etiquetas de Python): el **grupo 1** tiene más prolina ([[perfiles.centros.1.prolina|2]]), flavonoides ([[perfiles.centros.1.flavonoides|2]]) y alcohol; el **grupo 2**, más color y ácido málico y menos matiz y flavonoides; el **grupo 3**, menos alcohol, color y prolina. En la escala original el alcohol medio es 13.68, 13.14 y 12.25. Las mediciones **no permiten** hablar de precio, calidad ni preferencia.

### ✍️ Ejercicio C1
¿Cuál es el color medio (escala original) de cada grupo y cuál grupo lo tiene más alto? Calcula también cuántas desviaciones estándar sobre la media general está el grupo 2 en `matiz` (negativo = bajo).
""")
code("""
# tu código aquí
""")
solution("""
print(x.groupby(grupo)["color"].mean().round(2).to_dict())
print("matiz estandarizado del grupo 2:", round(float(cent_z.loc[2, "matiz"]), 3))
""", "Color medio: 5.45, 7.18 y 2.97; el más alto es el grupo 2. En matiz, el grupo 2 está 1.167 desviaciones estándar bajo la media general.")

md("""
### C2 · Contingencia, pureza y ARI con la clase real (20 min)
Ahora sí usamos `vinos_clases_reales.csv`: **no se usó para ajustar ni para elegir k**. La tabla de contingencia cruza clase real y grupo; la **pureza** (suma del máximo de cada grupo / n) y el **ARI** resumen la coincidencia. Es una comparación **descriptiva de estos vinos**, no la exactitud en vinos nuevos. Un grupo no es una clase: los números pueden estar permutados.
""")
code("""
tabla, pureza = purezas(clase_real, grupo)
display(tabla)
clase_del_grupo = tabla.idxmax(axis=0)
asignada = pd.Series(grupo).map(clase_del_grupo).to_numpy()
discrepa = asignada != clase_real
print(f"coinciden {int((~discrepa).sum())} de {len(z)} | pureza {pureza:.3%} | ARI {adjusted_rand_score(clase_real, grupo):.4f}")
print("vinos que discrepan:", list(vinos.codigo[discrepa]))
co = PCA(n_components=2).fit(z)
plano = co.transform(z); varp = 100 * co.explained_variance_ratio_
paleta = ["#4067A0", "#C66A25", "#247D72"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for c in (1, 2, 3):
    ax[0].scatter(*plano[clase_real == c].T, color=paleta[c - 1], s=20, label=f"Clase {c}")
    ax[1].scatter(*plano[asignada == c].T, color=paleta[c - 1], s=20, label=f"Grupo(s) de clase {c}")
ax[1].scatter(*plano[discrepa].T, facecolors="none", edgecolors="black", s=110, lw=1.4)
ax[0].set(title="Clases reales", xlabel=f"PC1 ({varp[0]:.1f} %)", ylabel=f"PC2 ({varp[1]:.1f} %)"); ax[1].set(title="Grupos de K-means (círculos: discrepancias)", xlabel=f"PC1 ({varp[0]:.1f} %)")
ax[0].legend(); plt.tight_layout(); plt.show()
print(f"las dos primeras componentes explican {varp.sum():.1f} % de la varianza")
""")
md("""
Resultado: **171 de 177** coinciden, pureza **[[pca.pureza|3]]** (96.6 %) y ARI **[[pca.ari|3]]**. En clase se dijo «8 o 9 mal clasificados, 92 %»: son **6** (todos de la clase 2: 3 en el grupo 1 y 3 en el grupo 2) y 96.6 %. Las dos componentes explican **[[pca.acum2|3]]** (el profesor dijo «casi 56 %»: es 55.3 %). Los 6 vinos que discrepan tienen silueta baja (media [[validar.sil_media_discrepan|3]] frente a [[validar.sil_media_coinciden|3]]); a los demás casos de la clase 2 los agrupa bien. Sin etiquetas, esta tabla **no existe**: la validación real es la estabilidad, los perfiles y el propósito.

### ✍️ Ejercicio C2
Calcula el ARI entre la clase real y K-means con `k = 4` y con `k = 2`. ¿Qué te dice sobre «el k correcto»? ¿Se puede calcular en un proyecto real?
""")
code("""
# tu código aquí
""")
solution("""
for k_ in (2, 3, 4):
    print(k_, round(adjusted_rand_score(clase_real, modelos[k_ - 1].labels_), 3))
""", "ARI: 0.373 (k = 2), 0.897 (k = 3) y 0.762 (k = 4). Con tres clases conocidas, k = 3 es lo mejor, pero el ARI exige la clase real, que en un proyecto real no existe: no sirve para elegir k.")

md("""
### C3 · ¿Agrupar con el PCA? (10 min)
Pregunta de un alumno: ¿conviene agrupar con las componentes principales? El profesor recomienda **agrupar con la base original** y usar el PCA solo para visualizar (pierde información, y el PCA solo trabaja con variables numéricas). Comprobamos qué ocurre aquí con dos componentes.
""")
code("""
km_pc = kmeans_prof(plano, 3)
t_pc, pur_pc = purezas(clase_real, km_pc.labels_)
print(f"K-means con las 2 componentes: pureza {pur_pc:.3%} | ARI con la clase {adjusted_rand_score(clase_real, km_pc.labels_):.3f} | ARI con la partición de 13 variables {adjusted_rand_score(grupo, km_pc.labels_):.3f}")
print("K-means con 13 variables:     pureza", f"{pureza:.3%}", "| ARI", round(adjusted_rand_score(clase_real, grupo), 3))
""")
md("""
Con dos componentes el ARI es **[[pca.ari_2pc|3]]** (frente a [[pca.ari|3]]): **prácticamente igual en estos datos**. La recomendación del profesor es correcta en general (las distancias del espacio original son las relevantes y con dos componentes se descarta el 44.7 % de la varianza), pero esta base tiene grupos tan marcados que la pérdida casi no se nota. No lo generalices.

### ✍️ Ejercicio C3
Repite con las **5 primeras componentes** (≈ 80 % de la varianza). ¿Cambia el ARI?
""")
code("""
# tu código aquí
""")
solution("""
pc5 = PCA(n_components=5).fit_transform(z)
print(round(float(PCA().fit(z).explained_variance_ratio_[:5].sum()), 3), "| ARI:", round(adjusted_rand_score(clase_real, kmeans_prof(pc5, 3).labels_), 3))
""", "Las 5 componentes explican 80.1 % y el ARI con 3 grupos es 0.897, igual al de las 13 variables: la reducción no cambió el resultado aquí.")

md("""
### C4 · Plantilla: K-means con k = 2 a 4 (puente con el Taller 3, pregunta 4) (20 min)
La pregunta 4 del Taller 3 pide algo de esta forma: ajustar K-means con **k = 2, 3 y 4** y `n_init ≥ 30`, resumir **WSS y silueta media**, describir **los perfiles** y recordar que **un grupo no es una clase predicha**. Esta plantilla lo hace con los vinos; para el taller sustituirías los datos (aquí **no** se abre ni se usa ninguna base del taller).
""")
code("""
def plantilla_kmeans(datos, ks=(2, 3, 4), n_init=30, semilla=RS):
    # datos: DataFrame ya estandarizado (media 0, desviación 1)
    filas, ajustes = [], {}
    for k in ks:
        m = KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=semilla, algorithm="lloyd", tol=1e-8).fit(datos)
        ajustes[k] = m
        filas.append(dict(k=k, WSS=round(m.inertia_, 2), silueta_media=round(silhouette_score(datos, m.labels_), 4), tamanos=np.bincount(m.labels_).tolist()))
    return pd.DataFrame(filas), ajustes


res_plantilla, ajustes_pl = plantilla_kmeans(z[["alcohol", "flavonoides", "color", "prolina"]].pipe(lambda d: (d - d.mean()) / d.std(ddof=1)))
display(res_plantilla)
k_elegido = int(res_plantilla.loc[res_plantilla.silueta_media.idxmax(), "k"])
grupos_pl = ajustes_pl[k_elegido].labels_ + 1
print("k elegido por silueta:", k_elegido)
display(x.groupby(grupos_pl)[["alcohol", "flavonoides", "color", "prolina"]].mean().round(2))
display(pd.crosstab(clase_real, grupos_pl, rownames=["clase real"], colnames=["grupo"]))
""")
md("""
Ideas para el taller: (1) **estandariza** y guarda cómo; (2) **n_init ≥ 30** para no quedarte con un mínimo local; (3) resume **WSS y silueta** para cada k y elige con criterio (no solo con el máximo); (4) interpreta los **perfiles** en la escala original; (5) **no confundas grupo con clase**: aunque exista una clase predicha por un modelo, el grupo no es esa clase; la tabla de contingencia muestra cuánto se parecen, no que sean lo mismo.

### ✍️ Ejercicio C4
Ejecuta la plantilla con las 13 variables estandarizadas y `n_init = 30` para k = 2, 3 y 4: ¿coincide el WSS con el de la parte A? ¿Qué k tiene la mayor silueta?
""")
code("""
# tu código aquí
""")
solution("""
res_13, _ = plantilla_kmeans(z)
display(res_13)
""", "WSS 1643.90, 1267.37 y 1164.91 (los mismos de la Parte A, porque 30 inicios bastan) y silueta máxima en k = 3 (0.284).")

# ================================================================== PARTE D
md("""
## Parte D · K-modes, K-prototypes, jerárquico y alertas

### D1 · K-modes (15 min)
Con **variables categóricas** no se puede promediar: el centro es la **moda** de cada variable y la **disimilitud** cuenta cuántas variables tienen valores distintos. Diapositiva 33: cliente `(Gold, App, Sur)`; centro A `(Gold, App, Norte)` → 1 desacuerdo; centro B `(Silver, Web, Sur)` → 2. Se asigna al grupo A. Para los vinos, la clase categoriza tres variables con cortes **docentes** (no umbrales enológicos): alcohol < 13 → bajo, ≥ 13 → alto; flavonoides < 1.5 → bajo, 1.5 a < 3 → medio, ≥ 3 → alto; color < 4 → bajo, 4 a < 6.5 → medio, ≥ 6.5 → alto. **Categorizar pierde información**.
""")
code("""
categorias = pd.DataFrame({
    "alcohol": np.where(vinos.alcohol < 13, "bajo", "alto"),
    "flavonoides": np.where(vinos.flavonoides < 1.5, "bajo", np.where(vinos.flavonoides < 3, "medio", "alto")),
    "color": np.where(vinos.color < 4, "bajo", np.where(vinos.color < 6.5, "medio", "alto")),
})
disim = lambda u, v: sum(p != q for p, q in zip(u, v))
print("diapositiva 33:", disim(("Gold", "App", "Sur"), ("Gold", "App", "Norte")), "y", disim(("Gold", "App", "Sur"), ("Silver", "Web", "Sur")))
if HAY_KMODES:
    costo_modos = [KModes(n_clusters=k, init="Huang", n_init=20, max_iter=100, random_state=RS).fit(categorias).cost_ for k in range(1, 9)]
    print("costo (total de desacuerdos) para k = 1..8:", [int(c) for c in costo_modos])
    kmodas = KModes(n_clusters=4, init="Huang", n_init=20, max_iter=100, random_state=RS).fit(categorias)
    grupo_modos = kmodas.labels_ + 1
    print("tamaños (k = 4):", np.bincount(grupo_modos)[1:].tolist())
    display(pd.DataFrame(kmodas.cluster_centroids_, columns=categorias.columns, index=range(1, 5)))
    g2 = int(np.argmax([((grupo_modos == g) & (categorias.color == "alto")).sum() for g in range(1, 5)])) + 1
    print(f"grupo {g2}: color", categorias.color[grupo_modos == g2].value_counts().to_dict())
""")
md("""
Reproduce las diapositivas 40 y 41: costo **[[kmodes.costo_k4|0]]** con k = 4 (la curva **no tiene un codo claro**), grupos de 72, 48, 42 y 15 y, en el grupo de 48 vinos con centro (alto, bajo, alto), color alto en **40 de 48 (83.3 %)**, medio en 7 (14.6 %) y bajo en 1 (2.1 %). El perfil resume **frecuencias**, no que todos los vinos tengan ese color. Lo de «sabemos que son 3» usa la clase real: **no** está disponible en la práctica. El costo cuenta **desacuerdos**, no sumas de cuadrados.

### ✍️ Ejercicio D1
Calcula el costo del grupo de la diapositiva 34: clientes `(Gold, App, Norte)`, `(Gold, Web, Norte)` y `(Silver, App, Sur)`. ¿Cuál es la moda por columna y cuántos desacuerdos suma el grupo?
""")
code("""
# tu código aquí
""")
solution("""
filas_d1 = [("Gold", "App", "Norte"), ("Gold", "Web", "Norte"), ("Silver", "App", "Sur")]
moda_d1 = tuple(max(set(c), key=list(c).count) for c in zip(*filas_d1))
print("moda:", moda_d1, "| desacuerdos por cliente:", [disim(f_, moda_d1) for f_ in filas_d1], "| costo del grupo:", sum(disim(f_, moda_d1) for f_ in filas_d1))
""", "Moda (Gold, App, Norte); desacuerdos 0, 1 y 2; costo del grupo = 3. Una moda puede tener empates y no siempre supera el 50 %.")

md("""
### D2 · K-prototypes y el peso γ (20 min)
Para variables **mixtas**: `d = Σ_num (x − μ)² + γ · Σ_cat 1(x ≠ moda)`. Diapositiva 43: observación `(1, 2)` con `(App, Sur)`; prototipo A `(0, 2)` con `(App, Norte)` → costo `1 + γ`; prototipo B `(2, 3)` con `(App, Sur)` → costo `2`. Con γ = 0.5 gana A (1.5 < 2); con γ = 3 gana B (4 > 2): **cambiar γ cambia la asignación con los mismos datos**. En la diapositiva 44, las variables numéricas son `ceniza` y `matiz` (estandarizadas) y las categóricas, las tres de D1. γ **no** es la penalización de Ridge: es un peso entre las dos partes de la distancia (en R se llama `lambda`).
""")
code("""
mixtos = pd.concat([z[["ceniza", "matiz"]], categorias], axis=1)
print("costos de la diapositiva 43:", {g_: (1 + g_) for g_ in (0.5, 3)}, "frente a 2")
if HAY_KMODES:
    costo_proto = [KPrototypes(n_clusters=k, init="Huang", n_init=20, gamma=1, max_iter=100, random_state=RS).fit(mixtos.to_numpy(), categorical=[2, 3, 4]).cost_ for k in range(1, 9)]
    print("costo mixto (γ = 1) para k = 1..8:", np.round(costo_proto, 2))
    kproto = KPrototypes(n_clusters=3, init="Huang", n_init=20, gamma=1, max_iter=100, random_state=RS).fit(mixtos.to_numpy(), categorical=[2, 3, 4])
    print("tamaños (k = 3):", np.bincount(kproto.labels_).tolist(), "| costo:", round(kproto.cost_, 2))
    display(pd.DataFrame(kproto.cluster_centroids_, columns=mixtos.columns, index=[1, 2, 3]))
    etiq = {}
    for g_ in (0.25, 1, 4):
        m_g = KPrototypes(n_clusters=3, init="Huang", n_init=20, gamma=g_, max_iter=100, random_state=RS).fit(mixtos.to_numpy(), categorical=[2, 3, 4])
        etiq[g_] = m_g.labels_
        print(f"γ = {g_}: grupos de {np.bincount(m_g.labels_).tolist()}, costo {m_g.cost_:.2f}")
    print("ARI entre γ = 0.25 y 1:", round(adjusted_rand_score(etiq[0.25], etiq[1]), 3), "| entre γ = 1 y 4:", round(adjusted_rand_score(etiq[1], etiq[4]), 3))
""")
md("""
Reproduce las diapositivas 48 a 50: costo **[[kproto.costo_k3|2]]** con k = 3 y grupos de **69, 47 y 61** (el grupo de 47 tiene matiz medio −1.258 y «color alto» como moda, aunque solo 30 de 47 = 63.8 % tienen esa categoría: **un prototipo puede no corresponder a ningún vino real**). Con γ = 0.25, 1 y 4 los grupos son [[kproto.por_gamma.0.25.tamanos]], [[kproto.por_gamma.1.tamanos]] y [[kproto.por_gamma.4.tamanos]]; entre γ = 0.25 y 1 el ARI es [[kproto.ari_025_vs_1|2]]. **Sin variable respuesta** no hay exactitud, ROC ni validación cruzada para elegir γ; se miran criterios internos, estabilidad y utilidad.

**Comparar costos entre métodos no tiene sentido** (diapositiva 52): K-means 1267.37 (cuadrados), K-modes 75 (desacuerdos) y K-prototypes 326.64 (mixto) tienen unidades y datos de entrada distintos. Se elige el método por el tipo de variables: numéricas → K-means; categóricas → K-modes; mixtas → K-prototypes.

### ✍️ Ejercicio D2
Calcula a mano los costos de la observación `(2, 1)` con `(Web, Norte)` frente a `P1 = (1, 1), (Web, Sur)` y `P2 = (3, 2), (Web, Norte)` para γ = 0.5, 1 y 2. ¿Con qué γ gana cada prototipo?
""")
code("""
# tu código aquí
""")
solution("""
for g_ in (0.5, 1, 2):
    c1 = (2 - 1) ** 2 + (1 - 1) ** 2 + g_ * 1       # P1: un desacuerdo (Norte frente a Sur)
    c2 = (2 - 3) ** 2 + (1 - 2) ** 2 + g_ * 0       # P2: ningún desacuerdo
    print(f"γ = {g_}: P1 = {c1}, P2 = {c2} -> gana", "P1" if c1 < c2 else ("P2" if c2 < c1 else "empate"))
""", "γ = 0.5 → P1 (1.5 < 2); γ = 1 → empate (2 = 2); γ = 2 → P2 (3 > 2). El umbral es γ = 1.")

md("""
### D3 · Complemento: agrupamiento jerárquico (15 min)
**No se vio en clase**; se incluye porque aparece en otros cursos y como alternativa a K-means. El método aglomerativo parte con cada vino solo y une los dos grupos más cercanos; el **enlace** define la distancia entre grupos: `single` (mínima), `complete` (máxima), `average` y `ward` (menor aumento de la suma de cuadrados). El resultado es un **dendrograma**; se corta a la altura que da k grupos.
""")
code("""
resultado_j = {}
for enlace in ("single", "complete", "average", "ward"):
    L = linkage(z.to_numpy(), method=enlace)
    lab_j = fcluster(L, 3, criterion="maxclust")
    resultado_j[enlace] = dict(tamaños=sorted(np.bincount(lab_j)[1:].tolist(), reverse=True), ARI=round(adjusted_rand_score(clase_real, lab_j), 3), silueta=round(silhouette_score(z, lab_j), 3))
display(pd.DataFrame(resultado_j).T)
Lw = linkage(z.to_numpy(), method="ward")
plt.figure(figsize=(9, 3.6)); dendrogram(Lw, truncate_mode="lastp", p=30, no_labels=True, color_threshold=(Lw[-3, 2] + Lw[-2, 2]) / 2)
plt.title("Dendrograma con enlace de Ward (30 ramas finales)"); plt.ylabel("distancia de fusión"); plt.show()
# ejemplo pequeño de la guía: cinco puntos en una recta
xs = np.array([0.0, 1.0, 2.5, 4.1, 5.8])
for enlace in ("single", "complete", "ward"):
    print(enlace, [(int(a_), int(b_), round(float(h_), 3)) for a_, b_, h_, _ in linkage(xs[:, None], enlace)])
""")
md("""
Con k = 3: **Ward** logra un ARI de [[jerarq.ari.ward|2]] (grupos de 64, 58 y 55), **complete** [[jerarq.ari.complete|2]], mientras que **single** y **average** producen el *efecto cadena*: un grupo de 173 vinos y dos casi vacíos (ARI ≈ 0). K-means obtiene [[jerarq.ari_kmeans|2]]. Las alturas **no son comparables entre enlaces**. El jerárquico no necesita fijar k de antemano, pero es costoso en datos grandes (necesita la matriz de distancias).

### ✍️ Ejercicio D3
Corta el dendrograma de Ward en 4 grupos y calcula el ARI con la clase real. ¿Qué ves en los tamaños?
""")
code("""
# tu código aquí
""")
solution("""
lab4 = fcluster(Lw, 4, criterion="maxclust")
print(sorted(np.bincount(lab4)[1:].tolist(), reverse=True), "| ARI:", round(adjusted_rand_score(clase_real, lab4), 3))
""", "Grupos de 64, 58, 32 y 23: se parte uno de los tres grupos y el ARI baja de 0.789 (k = 3) a 0.726; igual que con K-means, un k mayor refina pero no mejora la coincidencia con la clase.")

md("""
### D4 · Límites de K-means (10 min)
K-means supone grupos **compactos y redondos**, de tamaños y dispersiones parecidas, y es **sensible a atípicos** porque el centro es una media. El profesor mostró formas curvas que K-means divide mal y mencionó DBSCAN, que agrupa por **densidad** (Clase 41). Tres ejemplos simulados:
""")
code("""
Xm, ym = make_moons(300, noise=0.06, random_state=RS)
print("lunas | K-means ARI:", round(adjusted_rand_score(ym, KMeans(2, n_init=10, random_state=RS).fit(Xm).labels_), 3),
      "| DBSCAN ARI:", round(adjusted_rand_score(ym, DBSCAN(eps=0.2, min_samples=5).fit(Xm).labels_), 3))
rng = np.random.default_rng(RS)
base = np.vstack([rng.normal(c, 0.8, (80, 2)) for c in ([0, 0], [4, 0], [2, 3.4])])
yb = np.repeat([0, 1, 2], 80)
con_extremo = np.vstack([base, [[30.0, 30.0]]])
print("un solo extremo | ARI sin él:", round(adjusted_rand_score(yb, KMeans(3, n_init=10, random_state=RS).fit(base).labels_), 3),
      "| con él:", round(adjusted_rand_score(yb, KMeans(3, n_init=10, random_state=RS).fit(con_extremo).labels_[:-1]), 3))
print("tamaños de los grupos con el extremo:", np.bincount(KMeans(3, n_init=10, random_state=RS).fit(con_extremo).labels_).tolist())
""")
md("""
Con formas curvas K-means obtiene un ARI de ≈ [[limites.ari_moons_kmeans|2]] y DBSCAN, 1.00. **Un solo extremo** hace que K-means dedique un grupo a ese punto y mezcle los otros dos (en esta simulación el ARI baja de 0.95 a ≈ 0.54): por eso hay que revisar atípicos (Clase 41: Mahalanobis, LOF, Isolation Forest) y estandarizar con criterio. También se complican los grupos de tamaños y dispersiones muy distintos.

### ✍️ Ejercicio D4
Con la base simulada `base`, calcula el ARI de K-means con `k = 3` y `k = 2`. ¿Qué te enseña sobre elegir un k menor que el real?
""")
code("""
# tu código aquí
""")
solution("""
for k_ in (2, 3):
    print(k_, round(adjusted_rand_score(yb, KMeans(k_, n_init=10, random_state=RS).fit(base).labels_), 3))
""", "Con k = 2 se fusionan dos grupos y el ARI baja de 0.95 (k = 3) a 0.54: subestimar k oculta estructura; sobreestimarlo la parte.")

md("""
### D5 · Alertas del material, verificadas con código (10 min)

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Clase (ejemplo 1, 2, 3, 8, 9, 10) | «sacamos la raíz» del WSS | El WSS suma cuadrados; sale igual por ser 0 y 1 (A3). |
| Clase | «los clústeres que más aparecieron» | El criterio es el **menor WSS** (A5). |
| Clase | «50 mil iteraciones» | Converge en 6 (A4); `max_iter = 100` sobra. |
| Clase | silueta «casi 0.36» | Es 0.284 con k = 3 (B2). |
| Clase | Gap «valor residual anterior» | Se usa s(k+1) del paso siguiente (B3). |
| Clase | «8 o 9 mal clasificados, 92 %» | 6 de 177, 96.6 % (C2). |
| Clase | PCA «casi 56 %» | 55.3 % (C2). |
| Clase | «vino blanco, Merlot, Cabernet» | UCI Wine: 3 cultivares italianos; el CSV no trae nombres (C2). |
| Clase | «grupo 3: 13.25 de alcohol», «157 observaciones» | 12.25 y 177 (A4). |
| Clase | K-modes: «menor que 3» para flavonoides alto | Alto es ≥ 3 (D1). |
| Diap. 7 y 9 | `ddof = 1` reproduce `scale()` | Cierto; `StandardScaler` usa `ddof = 0`: WSS 1274.57 (A1). |
| Diap. 55–57 | Gap con k = 4 | Verificado (B3): un resultado ajustado; 4 semillas y B = 60 dan 4. R divide la desviación entre B (según su documentación) y el script de Python entre B − 1. No se ejecutó R. |
| Clase | «agrupar con la base original» frente a las componentes | Correcto en principio; aquí 2 componentes dan ARI 0.894 frente a 0.897 (C3). |
| Diap. 52 | los costos no se comparan | Confirmado: 1267.37, 75 y 326.64 miden cosas distintas (D1, D2). |

**Transcripción con ruido**: «camins / camín» = K-means, «Camos / KAMMOD» = K-modes, «capot» = K-prototypes, «GEP» = Gap, «cilita / siluente» = silueta, «interacción» = iteración, «VWSS / UWSS» = WSS, «Rolsocytes» = Royal Statistical Society. El recreo (hacia 1:08:40) quedó marcado y el audio termina a las 2:07:47 en la pregunta sobre γ.
""")
code("""
# comprobaciones rápidas de la tabla
print("WSS con StandardScaler:", round(KMeans(3, n_init=50, max_iter=100, random_state=RS, algorithm="lloyd", tol=1e-8).fit(z_ss).inertia_, 2))
print("iteraciones de K-means (k = 3):", km3.n_iter_, "| silueta k = 3:", round(float(medias_s[1]), 3), "| discrepancias:", int(discrepa.sum()))
print("varianza de las 2 primeras componentes:", round(float(varp.sum()), 1), "%")
print("ARI de 2 componentes:", round(adjusted_rand_score(clase_real, km_pc.labels_), 3), "| ARI de 13 variables:", round(adjusted_rand_score(clase_real, grupo), 3))
""")
md("""
### D6 · Tabla de correspondencia R ↔ Python (10 min)
Los scripts del profesor (`kmeans_numero_clusters.R.txt` y `.py`) siguen el mismo flujo:

| Qué se hace | R | Python |
|---|---|---|
| Leer datos | `read.csv("vinos.csv")` | `pd.read_csv("vinos.csv")` |
| Quitar el identificador | `vinos[, names(vinos) != "codigo"]` | `vinos.drop(columns="codigo")` |
| Estandarizar | `scale(x)` | `(x - x.mean()) / x.std(ddof=1)` |
| K-means | `kmeans(z, centers = k, nstart = 50, iter.max = 100, algorithm = "Lloyd")` | `KMeans(n_clusters=k, n_init=50, max_iter=100, algorithm="lloyd")` |
| Semilla | `set.seed(2026)` | `random_state=2026` |
| WSS | `modelo$tot.withinss` | `modelo.inertia_` |
| BSS y TSS | `modelo$betweenss`, `modelo$totss` | `tss - inertia_` con `tss = ((z - z.mean())**2).sum()` |
| Grupos | `modelo$cluster` (desde 1) | `modelo.labels_` (desde 0; se suma 1) |
| Centros | `modelo$centers` | `modelo.cluster_centers_` |
| Distancias | `dist(z)` | `scipy.spatial.distance.pdist` o `silhouette_score` directo |
| Silueta media | `mean(silhouette(cluster, dist(z))[, 3])` | `silhouette_score(z, labels_)` |
| Estadístico Gap | `cluster::clusGap(...)` y `maxSE(..., "Tibs2001SEmax")` | cálculo propio con simulación (B3) |
| Contingencia | `table(Clase_real = y, Grupo = g)` | `pd.crosstab(y, g)` |
| PCA para dibujar | `prcomp(z, center = FALSE, scale. = FALSE)` | `PCA(n_components=2).fit_transform(z)` |
| K-modes | `klaR::kmodes(cat, modes = k, weighted = FALSE)` | `kmodes.kmodes.KModes(n_clusters=k, init="Huang")` |
| K-prototypes | `clustMixType::kproto(x, k, lambda = peso)` | `kmodes.kprototypes.KPrototypes(n_clusters=k, gamma=peso)` |
| Costo | `withindiff` (K-modes), `tot.withinss` (kproto) | `cost_` |

*Diferencias a recordar*: los **números de grupo** pueden cambiar entre programas (diapositiva 27); las **semillas** no generan las mismas secuencias, así que Gap y los inicios difieren un poco; el **signo de las componentes** del PCA puede invertirse; `lambda` (R) y `gamma` (Python) son el mismo peso γ.

### ✍️ Ejercicio D6
Traduce a Python: `silueta <- mean(silhouette(modelo$cluster, dist(z))[, 3])`.
""")
code("""
# tu código aquí
""")
solution("""
print(round(silhouette_score(z, km3.labels_), 4))
""", "`silhouette_score(z, modelo.labels_)` calcula la media directamente (las distancias se calculan por dentro).")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes **segmentar los vinos** para un catálogo y justificarlo. Usa `x` (sin estandarizar) y responde con código:

1. **Plan previo**: escribe en un comentario cómo elegirás k y cómo juzgarás la estabilidad, **antes** de calcular nada.
2. **Preparación**: estandariza (`ddof=1`) y ajusta K-means con `k = 2 a 6`, `n_init ≥ 30` y una semilla fija. Resume WSS y silueta media.
3. **Elección**: propón un k aplicando tu regla y revisa **estabilidad**: ARI entre ajustes con 5 semillas distintas.
4. **Perfiles**: describe cada grupo con tres variables en la escala original.
5. **Validación externa**: solo al final compara con la clase real (tabla y ARI) y explica por qué esa clase no se usó antes.
6. **Sin código**: dos advertencias para el catálogo (por ejemplo, que un grupo no es una clase o una calidad).
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1) plan: elegir el k con mayor silueta media si es estable (ARI > 0.9 entre semillas); si hay empate, el k más pequeño interpretable
res_c = []
for k_ in range(2, 7):
    m_c = KMeans(n_clusters=k_, n_init=30, max_iter=100, random_state=RS, algorithm="lloyd", tol=1e-8).fit(z)
    otras = [KMeans(n_clusters=k_, n_init=30, max_iter=100, random_state=s_, algorithm="lloyd", tol=1e-8).fit(z).labels_ for s_ in range(1, 6)]
    res_c.append(dict(k=k_, WSS=round(m_c.inertia_, 2), silueta=round(silhouette_score(z, m_c.labels_), 4), ARI_min_entre_semillas=round(min(adjusted_rand_score(m_c.labels_, o) for o in otras), 3)))
res_c = pd.DataFrame(res_c)
display(res_c)
k_final = int(res_c.loc[res_c.silueta.idxmax(), "k"])
m_final = KMeans(n_clusters=k_final, n_init=30, max_iter=100, random_state=RS, algorithm="lloyd", tol=1e-8).fit(z)
gf = m_final.labels_ + 1
display(x.groupby(gf)[["alcohol", "flavonoides", "color"]].mean().round(2))
display(pd.crosstab(clase_real, gf, rownames=["clase real"], colnames=["grupo"]))
print("k =", k_final, "| ARI con la clase real:", round(adjusted_rand_score(clase_real, gf), 3))
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: silueta máxima en **k = 3** ([[silueta.media_k3|3]]); la partición es **estable** (ARI = 1.000 entre semillas con `n_init ≥ 30`, porque casi siempre se alcanza el mismo óptimo). Perfiles: grupo de 62 vinos (alcohol 13.68, flavonoides 3.00, color 5.45), grupo de 50 (13.14, 0.82, 7.18) y grupo de 65 (12.25, 2.05, 2.97). Validación externa: pureza 96.6 % y ARI [[pca.ari|3]]. **Advertencias**: (1) los grupos son un **resumen químico**: no demuestran precio, calidad ni preferencia; (2) la silueta (0.28) indica **superposición** y 6 vinos quedan en un grupo distinto del de su clase: usar los grupos como guía, no como etiqueta definitiva. (3) Un k distinto (por ejemplo 4, por el Gap) es defendible; documenta la decisión.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué agrupar no es clasificar? ¿Qué significa que el «grupo 1» no sea «mejor»?
2. ¿Qué dos pasos repite K-means y qué minimiza? ¿Por qué el centro es la media?
3. ¿Qué miden WSS, BSS y TSS y cómo se relacionan?
4. ¿Por qué hay que estandarizar? ¿Qué pasó con la prolina?
5. ¿Qué es un mínimo local y qué hace `n_init`? ¿Qué criterio decide el ajuste que se conserva?
6. ¿Por qué el WSS no sirve por sí solo para elegir k? ¿Qué mide la silueta y cómo se lee?
7. ¿Qué compara el Gap y qué regla se aplica? ¿Por qué da 4 si el codo y la silueta dan 3?
8. ¿Qué es un perfil de grupo y qué NO se puede concluir de él?
9. ¿Qué mide el ARI y por qué no sirve para elegir k en un proyecto real?
10. ¿Cuándo usar K-modes y cuándo K-prototypes? ¿Qué hace γ? ¿Por qué no se comparan los costos?
11. Nombra dos límites de K-means y una alternativa para cada uno.

<details><summary><b>Respuestas</b></summary>

1. Porque no hay una Y observada: los grupos los construye el algoritmo con las variables elegidas y su número es una etiqueta arbitraria; el «grupo 1» no significa «positivo», «mejor» ni «mayor riesgo».
2. Asignar cada punto al centro más cercano y recalcular cada centro como la media de su grupo. Minimiza la suma de distancias euclidianas **al cuadrado** (WSS); para un grupo fijo, la media minimiza esa suma.
3. WSS = variabilidad dentro de los grupos, BSS = entre grupos (centros frente a la media global, ponderados por tamaño) y TSS = total; TSS = WSS + BSS.
4. Porque K-means usa distancias y la variable de mayor escala domina; la prolina aportaba casi el 100 % de la varianza y el ARI caía de 0.897 a 0.375.
5. Un mínimo local es un resultado que no se puede mejorar con pasos locales pero no es el mejor; `n_init` repite desde distintos inicios y se conserva el de **menor WSS**.
6. Porque siempre baja al subir k (con k = n vale 0). La silueta, s = (b − a)/máx(a, b), mide la separación: cerca de 1 bien separado, cerca de 0 superposición, negativo más cerca de otro grupo.
7. El WSS observado frente al de datos simulados sin estructura; se elige el primer k con Gap(k) ≥ Gap(k+1) − s(k+1). El resultado es ajustado y los criterios pueden discrepar; se decide con interpretación y estabilidad.
8. La media de cada variable por grupo (escala original o estandarizada). No describe a cada vino ni permite hablar de precio, calidad o preferencia.
9. La concordancia entre dos particiones (1 = iguales, ≈ 0 = azar). Exige la clase real, que en la práctica no existe.
10. K-modes con variables categóricas (moda y desacuerdos) y K-prototypes con mixtas (medias, modas y peso γ). γ cambia las asignaciones aunque los datos sean los mismos. Los costos tienen unidades y datos de entrada distintos.
11. Formas curvas (DBSCAN) y atípicos o grupos de tamaños muy distintos (revisar atípicos con Mahalanobis, LOF o Isolation Forest; modelos de mezclas).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** Clase 41 (Mahalanobis, DBSCAN) y el Taller 3, pregunta 4 (K-means con k = 2 a 4). Revisa `Guias/apuntes/C39_Ejercicios_Cuaderno_Clustering.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
