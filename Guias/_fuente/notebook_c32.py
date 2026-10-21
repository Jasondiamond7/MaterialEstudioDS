"""Genera el notebook guiado de Reducción de dimensionalidad y representación de variables (Clase 32).

Uso normal:      python notebook_c32.py             -> Guias/notebooks/C32_ReduccionDimensionalidad.ipynb
Modo de prueba:  C32_TEST=1 python notebook_c32.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C32_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C32_ReduccionDimensionalidad_test.ipynb" if TEST else "C32_ReduccionDimensionalidad.ipynb")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{note}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n"
       f"{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- encabezado
md("""
# Clase 32 · Reducción de dimensionalidad y representación de variables
**Diplomado Data Science** · Profa. Laura Sánchez

Material de apoyo:
- `Guias/C32_Guia_ReduccionDimensionalidad.pdf` — guía visual de la clase.
- `Guias/apuntes/C32_Apuntes_ReduccionDimensionalidad.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C32_Ejercicios_Cuaderno_ReduccionDimensionalidad.pdf` — ejercicios en papel (antes o después de este notebook).
- Material de la profesora: `Sesion_32_Reducción_dimensionalidad.ipynb`, las diapositivas, el apunte, `Base de datos Churn-Actividad repaso.xlsx` y las respuestas del caso de repaso (Sesiones 31 y 32).
- Transcripción: `VideosClases/Clase 32 Reduccion de dimensionalidad y representacion de variables.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `pandas`, `numpy`, `matplotlib` y `openpyxl` (para leer el .xlsx). Si no los tienes:
```
pip install scikit-learn pandas numpy matplotlib openpyxl
```
**Datos**: (1) el churn **simulado** de la profesora (1.500 clientes), que se genera con código y semilla fija: obtienes sus mismos números; (2) la **base real** de 3.150 clientes de la actividad de repaso (archivo en la carpeta de la clase; se usa en C5 y en el ejercicio de cierre).

### Objetivos
1. Diagnosticar por qué sobran variables: **multicolinealidad (VIF)**, **ruido** y **distancias en alta dimensión**.
2. **Representar** las variables: estandarizar sin fuga, codificar (ordinal, one-hot, target) y construir cocientes.
3. Entender **PCA**: valores y vectores propios, varianza explicada, elección de k con **validación cruzada** y lectura de **cargas**.
4. Armar un `Pipeline` con `ColumnTransformer`, `StandardScaler` y `PCA` **ajustado solo con entrenamiento**.
5. Usar **t-SNE** (perplexity, semillas) para explorar y decidir entre PCA, t-SNE, selección y feature engineering.
6. Reconocer lo que el material original afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y datos |
| A | 10 | A1 · Panorama y correlaciones |
| A | 10 | A2 · Multicolinealidad: el VIF |
| A | 15 | A3 · Ruido y distancias en alta dimensión |
| B | 15 | B1 · Construir, codificar, dividir y estandarizar |
| B | 10 | B2 · Qué pasa si no se estandariza |
| B | 20 | B3 · Codificar categóricas: ordinal, one-hot y target |
| B | 10 | B4 · Un ColumnTransformer completo |
| C | 15 | C1 · Valores y vectores propios a mano |
| C | 15 | C2 · PCA y varianza explicada |
| C | 20 | C3 · Elegir k con validación cruzada |
| C | 15 | C4 · Cargas y círculo de correlaciones |
| C | 15 | C5 · La base real de la actividad de repaso |
| D | 20 | D1 · t-SNE: perplexity |
| D | 10 | D2 · Semillas y límites de t-SNE |
| D | 10 | D3 · PCA vs t-SNE: ¿cuál uso? |
| — | 10 | Alertas del material |
| — | 35 | Ejercicio de cierre **sin ayuda** (base real) |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook con `scikit-learn 1.9`. Con otras versiones pueden variar un poco (sobre todo t-SNE).
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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 32   # semilla de la clase
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Los datos de la clase (simulados)
La profesora genera un churn de telco con `np.random.default_rng(32)`: dos factores ocultos (**uso** e **insatisfacción**) producen las variables, y la fuga depende de ellos. Por eso hay seis variables de uso casi duplicadas. La función es **idéntica** a la de su notebook.
""")
code("""
def generar_churn_sintetico(n=1500, seed=RS):
    rng = np.random.default_rng(seed)
    insatisfaccion = rng.normal(0, 1, n)
    uso = rng.normal(0, 1, n)
    antiguedad = np.clip(rng.normal(24, 12, n), 1, 72).round(0)
    tipo_plan = rng.choice([1, 2], size=n, p=[0.55, 0.45])
    estado = rng.choice([1, 2], size=n, p=[0.85, 0.15])
    edad = np.clip(rng.normal(38, 12, n), 18, 80).round(0)
    numero_fallas = np.clip(6 + 4 * insatisfaccion + rng.normal(0, 2, n), 0, 35).round(0)
    reclamos = (rng.uniform(0, 1, n) < (0.05 + 0.12 * (insatisfaccion > 0.5))).astype(int)
    monto = np.clip((4 + 2 * uso + rng.normal(0, 1.5, n)).round(0), 0, 9)
    segundos = np.clip(4000 + 3000 * uso - 800 * insatisfaccion + rng.normal(0, 900, n), 50, None).round(0)
    frec_llamadas = np.clip((segundos / 65) + rng.normal(0, 8, n), 1, None).round(0)
    frec_sms = np.clip(60 + 90 * uso + rng.normal(0, 40, n), 0, None).round(0)
    n_contactos = np.clip((frec_llamadas * 0.35) + rng.normal(0, 5, n), 0, None).round(0)
    clv = np.clip(300 + 220 * uso - 90 * insatisfaccion + rng.normal(0, 120, n), 10, None).round(0)
    logit = (-2.2 + 0.9 * insatisfaccion + 0.05 * numero_fallas + 1.1 * reclamos - 0.02 * antiguedad - 0.15 * uso)
    fuga = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "numero_fallas": numero_fallas, "reclamos": reclamos, "antiguedad": antiguedad, "monto": monto,
        "segundos": segundos, "frec_llamadas": frec_llamadas, "frec_sms": frec_sms, "n_contactos": n_contactos,
        "tipo_plan": tipo_plan, "estado": estado, "edad": edad, "clv": clv, "fuga": fuga})


df = generar_churn_sintetico()
y = df["fuga"]
X_raw = df.drop(columns=["fuga"])
print("Observaciones:", df.shape[0], "| variables (sin 'fuga'):", X_raw.shape[1])
print(f"Tasa de fuga: {y.mean():.2%}  (≈ 85 % no se fuga: desbalanceado, hay que estratificar)")
df.head()
""")

# ================================================================== PARTE A
md("""
## Parte A · El problema de la dimensionalidad

### A1 · Panorama y correlaciones (10 min)
Antes de reducir, mira qué tan redundantes son las variables. La matriz de correlaciones es la primera pista.
""")
code("""
corr = X_raw.corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        if i != j and abs(corr.values[i, j]) >= 0.6:
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
ax.grid(False); fig.colorbar(im, fraction=0.046, pad=0.04)
ax.set_title("Correlación entre las 12 variables originales")
plt.tight_layout(); plt.show()

uso = ["segundos", "frec_llamadas", "n_contactos", "frec_sms", "monto", "clv"]
r_uso = corr.loc[uso, uso].values[np.triu_indices(6, 1)]
print(f"Las 6 variables de uso: correlación mínima {r_uso.min():.2f}, media {r_uso.mean():.2f}")
""")
md("""
**Lectura:** seis variables de uso (`segundos`, `frec_llamadas`, `n_contactos`, `frec_sms`, `monto`, `clv`) están correlacionadas entre 0.66 y 0.99 (media 0.80). Cuentan casi la misma historia: *cuánto usa el cliente el servicio*. Ese es el diagnóstico que justifica reducir.

### ✍️ Ejercicio A1
Lista los pares de variables con |correlación| ≥ 0.90. ¿Cuántos pares son y cuáles?
""")
code("""
# tu código aquí
""")
solution("""
pares = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.90]
print(len(pares), "pares:", pares)
""", "Son 3 pares: `segundos`–`frec_llamadas` (0.99), `frec_llamadas`–`n_contactos` (0.96) y `segundos`–`n_contactos` (0.94).")

md("""
### A2 · Multicolinealidad: el VIF (10 min)
El **factor de inflación de la varianza** de una variable es `VIF = 1 / (1 − R²)`, donde R² viene de explicar esa variable con **todas las demás**. Regla práctica: VIF > 10 es una señal fuerte de redundancia.
""")
code("""
def vif(frame):
    Z = (frame - frame.mean()) / frame.std()
    salida = {}
    for c in frame.columns:
        otros = [k for k in frame.columns if k != c]
        r2 = LinearRegression().fit(Z[otros], Z[c]).score(Z[otros], Z[c])
        salida[c] = 1 / (1 - r2)
    return pd.Series(salida).round(1).sort_values(ascending=False)

print(vif(X_raw))
""")
md("""
`frec_llamadas` (48.3), `segundos` (40.8) y `n_contactos` (12.2) superan 10; `frec_sms` (3.6), `clv` (3.7) y `monto` (2.3) no. El VIF mide redundancia **con el resto**, no solo por pares.

### ✍️ Ejercicio A2
Quita `frec_llamadas` y vuelve a calcular el VIF. ¿Qué pasa con `segundos` y `n_contactos`? ¿Qué te dice sobre «seleccionar» una variable de cada grupo?
""")
code("""
# tu código aquí
""")
solution("""
print(vif(X_raw.drop(columns=["frec_llamadas"])))
""", "Sin `frec_llamadas`, el VIF de `segundos` baja de 40.8 a 13.5 y el de `n_contactos` de 12.2 a 9.1: eran redundantes **entre sí**. Seleccionar una variable por grupo reduce la multicolinealidad pero descarta información; PCA, en cambio, la resume.")

md("""
### A3 · Ruido y distancias en alta dimensión (15 min)
Dos efectos de tener demasiadas columnas: (1) las **distancias se aplanan** (todos quedan igual de lejos) y (2) con **pocas filas** las columnas de ruido hacen que el modelo memorice (sobreajuste).
""")
code("""
rng = np.random.default_rng(1)
print("d | (más lejano − más cercano) / más cercano   (400 puntos uniformes)")
for d in (2, 10, 100, 1000):
    P = rng.uniform(size=(400, d))
    r = []
    for q in range(15):
        dist = np.linalg.norm(np.delete(P, q, axis=0) - P[q], axis=1)
        r.append((dist.max() - dist.min()) / dist.min())
    print(f"{d:5d} | {np.mean(r):8.2f}")
""")
md("""
Con 2 dimensiones el más lejano está ~38 veces más lejos que el más cercano; con 1.000, solo 0.1 veces: **todos los puntos quedan casi a la misma distancia**, y cualquier método basado en distancias (t-SNE, vecinos más cercanos) pierde sentido. Ahora el ruido, con solo **100 filas de entrenamiento**:
""")
code("""
# Para el experimento usamos las 15 columnas de la clase (se construyen en B1); aquí las preparamos de forma mínima
def construir(frame):
    f = frame.copy()
    f["duracion_prom_llamada"] = f["segundos"] / f["frec_llamadas"].replace(0, np.nan)
    f["sms_por_antiguedad"] = f["frec_sms"] / f["antiguedad"].replace(0, np.nan)
    f["intensidad_contacto"] = f["n_contactos"] / f["frec_llamadas"].replace(0, np.nan)
    f = f.fillna(0)
    f["tipo_plan_2"] = (f.pop("tipo_plan") == 2).astype(int)     # one-hot de una variable binaria
    f["estado_2"] = (f.pop("estado") == 2).astype(int)
    return f

X_all = construir(X_raw)
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_all, y, test_size=0.25, random_state=RS, stratify=y)
print("Columnas:", X_all.shape[1], "| train:", Xa_tr.shape, "| test:", Xa_te.shape)

ruido = {}
for k in (0, 25, 100, 400):
    aucs = []
    for s in range(10):
        r = np.random.default_rng(100 + s)
        idx = r.choice(len(Xa_tr), 100, replace=False)
        while ya_tr.values[idx].sum() < 8:
            idx = r.choice(len(Xa_tr), 100, replace=False)
        A, T = Xa_tr.values[idx], Xa_te.values
        if k:
            A = np.hstack([A, r.normal(size=(100, k))])
            T = np.hstack([T, r.normal(size=(len(T), k))])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(A, ya_tr.values[idx])
        aucs.append(roc_auc_score(ya_te, m.predict_proba(T)[:, 1]))
    ruido[k] = round(float(np.mean(aucs)), 3)
print("AUC en prueba según columnas de ruido agregadas:", ruido)
""")
md("""
Con 100 filas, agregar columnas de ruido baja el AUC de prueba (≈ 0.74 sin ruido → ≈ 0.57 con 400 columnas; 0.65 con 25 y 0.63 con 100) mientras el del entrenamiento sube hacia 1: **más varianza sin compensar el sesgo**. (Con 10 repeticiones los valores varían un poco respecto de los 25 de la guía.)

### ✍️ Ejercicio A3
Repite el experimento con **400 filas** de entrenamiento y 100 columnas de ruido. ¿Se atenúa el efecto? ¿Qué te dice sobre la relación filas/columnas?
""")
code("""
# tu código aquí
""")
solution("""
def auc_con_ruido(n_filas, k_ruido, reps=10):
    aucs = []
    for sem in range(reps):
        r = np.random.default_rng(200 + sem)
        idx = r.choice(len(Xa_tr), n_filas, replace=False)
        A, T = Xa_tr.values[idx], Xa_te.values
        if k_ruido:
            A = np.hstack([A, r.normal(size=(n_filas, k_ruido))])
            T = np.hstack([T, r.normal(size=(len(T), k_ruido))])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(A, ya_tr.values[idx])
        aucs.append(roc_auc_score(ya_te, m.predict_proba(T)[:, 1]))
    return round(float(np.mean(aucs)), 3)

for n_filas in (100, 400):
    print(f"{n_filas} filas: sin ruido {auc_con_ruido(n_filas, 0)} | con 100 columnas de ruido {auc_con_ruido(n_filas, 100)}")
""", "Más filas ayudan (sin ruido el AUC pasa de 0.709 a 0.767), pero **no neutralizan el ruido**: 100 columnas de ruido siguen restando ≈ 0.09 con 100 filas (0.709 → 0.624) y ≈ 0.12 con 400 (0.767 → 0.647). Quitar o resumir columnas sin señal sigue valiendo la pena.")

# ================================================================== PARTE B
md("""
## Parte B · Representación de variables

### B1 · Construir, codificar, dividir y estandarizar (15 min)
Orden de la clase: **construir** variables nuevas (cocientes) → **codificar** categóricas → **dividir** → **estandarizar** (`fit` solo con entrenamiento). Los cocientes y el one-hot de una variable binaria no aprenden nada del conjunto, por eso pueden hacerse antes de dividir.
""")
code("""
X_enc = construir(X_raw)
print("Columnas:", X_raw.shape[1], "originales →", X_enc.shape[1], "tras construir 3 variables y codificar tipo_plan y estado")
display(X_enc[["duracion_prom_llamada", "sms_por_antiguedad", "intensidad_contacto"]].describe().T.round(2))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=RS, stratify=y)
escalador = StandardScaler().fit(X_train_raw)                  # FIT SOLO con entrenamiento
X_train = escalador.transform(X_train_raw)
X_test = escalador.transform(X_test_raw)
print("train:", X_train.shape, "| test:", X_test.shape, "| fuga train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))
print("Media de las columnas escaladas (train):", X_train.mean(axis=0).round(2).max(), "| desviación:", X_train.std(axis=0).round(2).min(), "–", X_train.std(axis=0).round(2).max())
""")
md("""
Las 12 variables originales pasan a 15: tres cocientes nuevos y, como `tipo_plan` y `estado` son binarias, un one-hot con `drop` deja **una** columna cada una (recodificar no crea variables). Tras estandarizar, cada columna tiene media 0 y desviación 1 **en el entrenamiento**; en la prueba no serán exactamente 0 y 1, y eso es lo correcto.

### ✍️ Ejercicio B1
¿Cuántas columnas tendría el one-hot de `tipo_plan` y `estado` **sin** `drop` (dos columnas por variable)? ¿Por qué se descarta una en el caso binario?
""")
code("""
# tu código aquí
""")
solution("""
oh = OneHotEncoder(sparse_output=False).fit(X_raw[["tipo_plan", "estado"]].astype(str))
print(len(oh.get_feature_names_out()), oh.get_feature_names_out())
""", "Sin `drop` serían 4 columnas (17 en total, en vez de 15). En una variable binaria, una columna es 1 menos la otra: redundancia perfecta (multicolinealidad exacta).")

md("""
### B2 · Qué pasa si no se estandariza (10 min)
PCA maximiza **varianza**; una variable medida en miles tendrá una varianza gigantesca solo por su unidad.
""")
code("""
var_cruda = X_train_raw.var()
print("Parte de la varianza total de cada variable (sin estandarizar):")
print((var_cruda / var_cruda.sum()).sort_values(ascending=False).head(3).map("{:.1%}".format).to_string())

p_raw = PCA(random_state=RS).fit(X_train_raw)
p_std = PCA(random_state=RS).fit(X_train)
print(f"\\nPC1 sin estandarizar: {p_raw.explained_variance_ratio_[0]:.2%} | estandarizado: {p_std.explained_variance_ratio_[0]:.2%}")
print("Carga de 'segundos' en PC1 sin estandarizar:", round(abs(p_raw.components_[0][list(X_train_raw.columns).index('segundos')]), 3))
""")
md("""
Sin estandarizar, `segundos` concentra el **99.2 %** de la varianza y PC1 explica **99.71 %**: ¡un solo componente «resume» todo! Es un **artefacto de las unidades**, no una estructura de los datos. Estandarizado, PC1 explica 34.9 %.

### ✍️ Ejercicio B2
Repite con `MinMaxScaler` (normalización min-max) en vez de `StandardScaler`. ¿Cuánto explica PC1? ¿El resultado coincide con el estandarizado?
""")
code("""
# tu código aquí
""")
solution("""
mm = MinMaxScaler().fit(X_train_raw)
p_mm = PCA(random_state=RS).fit(mm.transform(X_train_raw))
print("PC1 con min-max:", round(p_mm.explained_variance_ratio_[0], 3), "| con StandardScaler:", round(p_std.explained_variance_ratio_[0], 3))
print("k para 80 %: min-max", int(np.argmax(np.cumsum(p_mm.explained_variance_ratio_) >= 0.8) + 1),
      "| estandarizado", int(np.argmax(np.cumsum(p_std.explained_variance_ratio_) >= 0.8) + 1))
""", "Con min-max PC1 explica 32.5 % (34.9 % estandarizado) y bastan 4 componentes para el 80 % (8 con StandardScaler): no coincide. Las variables con colas largas (como los cocientes) quedan comprimidas en un rango pequeño y pesan menos. Que una variable quede en una escala «comparable» no significa que el resultado sea el mismo: la estandarización es la habitual para PCA.")

md("""
### B3 · Codificar categóricas: ordinal, one-hot y target (20 min)
La diapositiva presenta tres codificaciones:
- **Ordinal**: enteros **cuando hay un orden real** (bajo < medio < alto). *(La diapositiva dice «Nominal → entero»: es un error de rotulación.)*
- **One-hot**: una columna 0/1 por categoría; no impone orden (todas las categorías quedan a la misma distancia, √2).
- **Target / frequency**: reemplaza la categoría por una estadística (tasa de fuga, frecuencia). Útil con **muchas** categorías, pero el target encoding **mira `y`**: hay que calcularlo con validación interna.

Experimento: inventamos 60 comunas **sin relación con la fuga** y las codificamos de tres formas.
""")
code("""
rng = np.random.default_rng(7)
niveles = [f"c{i}" for i in range(60)]
com_tr = rng.choice(niveles, len(X_train_raw))
com_te = rng.choice(niveles, len(X_test_raw))

# (1) media del target calculada "a mano" con todo el entrenamiento
medias = pd.Series(y_train.values).groupby(com_tr).mean()
enc1_tr = pd.Series(com_tr).map(medias).values
enc1_te = pd.Series(com_te).map(medias).fillna(y_train.mean()).values
# (2) one-hot
oh = OneHotEncoder(handle_unknown="ignore").fit(com_tr.reshape(-1, 1))
lr = LogisticRegression(max_iter=2000).fit(oh.transform(com_tr.reshape(-1, 1)), y_train)
# (3) TargetEncoder de scikit-learn (validación cruzada interna en fit_transform)
te = TargetEncoder(smooth="auto", target_type="binary", cv=5, random_state=RS)
enc3_tr = te.fit_transform(com_tr.reshape(-1, 1), y_train).ravel()
enc3_te = te.transform(com_te.reshape(-1, 1)).ravel()

res = pd.DataFrame({
    "media del target (a mano)": [roc_auc_score(y_train, enc1_tr), roc_auc_score(y_test, enc1_te)],
    "one-hot (60 columnas)": [roc_auc_score(y_train, lr.predict_proba(oh.transform(com_tr.reshape(-1, 1)))[:, 1]),
                              roc_auc_score(y_test, lr.predict_proba(oh.transform(com_te.reshape(-1, 1)))[:, 1])],
    "TargetEncoder (cv=5)": [roc_auc_score(y_train, enc3_tr), roc_auc_score(y_test, enc3_te)]},
    index=["AUC entrenamiento", "AUC prueba"]).round(3)
res
""")
md("""
La variable **no tiene señal** (el AUC honesto es 0.5). La media del target «a mano» y el one-hot logran **0.70** en entrenamiento y **0.51** en prueba: memorizaron las 60 comunas. El `TargetEncoder` de scikit-learn calcula las tasas con validación cruzada interna dentro de `fit_transform` y no se engaña (0.53 en entrenamiento, 0.51 en prueba). Moraleja: el target encoding calculado con todas las filas es **fuga de datos**.

### ✍️ Ejercicio B3
Crea `nivel_plan` a partir de `monto` (0–2 «bajo», 3–5 «medio», 6–9 «alto») y codifícalo con `OrdinalEncoder` respetando el orden. ¿Qué valores recibe «alto»? ¿Qué pasaría con el orden alfabético por defecto?
""")
code("""
# tu código aquí
""")
solution("""
nivel = pd.cut(X_raw["monto"], bins=[-1, 2, 5, 9], labels=["bajo", "medio", "alto"]).astype(str).to_frame("nivel_plan")
orden = OrdinalEncoder(categories=[["bajo", "medio", "alto"]]).fit(nivel)
defecto = OrdinalEncoder().fit(nivel)
print("con orden explícito:", dict(zip(["bajo", "medio", "alto"], orden.transform(pd.DataFrame({"nivel_plan": ["bajo", "medio", "alto"]})).ravel())))
print("orden alfabético   :", dict(zip(defecto.categories_[0], range(3))))
""", "Con `categories=[[\"bajo\",\"medio\",\"alto\"]]`: bajo = 0, medio = 1, alto = 2. Por defecto el orden es alfabético (alto = 0, bajo = 1, medio = 2): impone un orden falso. Por eso el orden debe indicarse.")

md("""
### B4 · Un ColumnTransformer completo (10 min)
Para variables categóricas **no binarias** (aquí inventamos `region`) conviene dejar el one-hot **dentro** del `ColumnTransformer`, para que se aprenda con entrenamiento. Mismo patrón que la Clase 30:
""")
code("""
dfc = X_raw.copy()
dfc["region"] = np.random.default_rng(5).choice(["Norte", "Sur", "Este", "Oeste", "Centro"], len(dfc))   # SIMULADA, sin relación con la fuga
num_cols = [c for c in X_raw.columns if c not in ("tipo_plan", "estado")]
prep = ColumnTransformer([("num", StandardScaler(), num_cols),
                          ("cat", OneHotEncoder(drop="first", sparse_output=False), ["tipo_plan", "estado", "region"])])
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(dfc, y, test_size=0.25, random_state=RS, stratify=y)
Z_tr = prep.fit_transform(Xc_tr)          # aprende media, desviación y categorías SOLO con entrenamiento
Z_te = prep.transform(Xc_te)
print("Columnas de salida:", Z_tr.shape[1], "=", len(num_cols), "numéricas +", Z_tr.shape[1] - len(num_cols), "binarias (drop='first')")
print(prep.get_feature_names_out()[-6:])
""")
md("""
10 numéricas + (1 + 1 + 4) one-hot = 16 columnas. Con `drop="first"` cada categórica pierde una columna redundante. Si llega un cliente de una región nunca vista, fallará salvo que uses `handle_unknown="ignore"` (sin `drop`).

### ✍️ Ejercicio B4
¿Cuántas columnas saldrían con `drop=None`? Compruébalo.
""")
code("""
# tu código aquí
""")
solution("""
prep2 = ColumnTransformer([("num", StandardScaler(), num_cols),
                           ("cat", OneHotEncoder(sparse_output=False), ["tipo_plan", "estado", "region"])])
print(prep2.fit_transform(Xc_tr).shape[1])
""", "10 + (2 + 2 + 5) = 19 columnas: con `drop=\"first\"` son 16.")

# ================================================================== PARTE C
md("""
## Parte C · Extracción de variables: PCA

### C1 · Valores y vectores propios a mano (15 min)
PCA diagonaliza la **matriz de covarianza** de los datos estandarizados. Sus **vectores propios** son las direcciones (componentes) y sus **valores propios** λ, las varianzas.
""")
code("""
# Primero el ejemplo de la profesora: una matriz NO simétrica
A = np.array([[10, -18], [6, -11]])
w, V = np.linalg.eig(A)
print("Valores propios:", np.real(w), "| ¿A v = λ v?", np.allclose(A @ V[:, 0], w[0] * V[:, 0]))
print("Producto punto de los dos vectores propios:", round(float(np.real(V[:, 0] @ V[:, 1])), 3), " <- no son perpendiculares")
""")
md("""
Cumple `A v = λ v`, pero sus vectores **no son ortogonales** (producto 0.99). Las covarianzas forman una matriz **simétrica**, y para esas los vectores propios sí son ortogonales (y los λ son reales). Con los datos de la clase:
""")
code("""
C = np.cov(X_train, rowvar=False)                       # covarianza de los datos estandarizados (= correlación)
lam, vec = np.linalg.eigh(C)                            # eigh: para matrices simétricas
orden = np.argsort(lam)[::-1]
lam, vec = lam[orden], vec[:, orden]

pca = PCA(random_state=RS).fit(X_train)
print("Σλ =", lam.sum().round(3), "(= nº de variables, porque están estandarizadas)")
print("λ a mano       :", lam[:5].round(3))
print("λ de PCA       :", pca.explained_variance_[:5].round(3))
print("¿Mismos vectores (salvo el signo)?", np.allclose(np.abs(vec.T), np.abs(pca.components_), atol=1e-6))
print("Ortogonales (V'V = I):", np.allclose(vec.T @ vec, np.eye(15)))
print("Varianza explicada = λ / Σλ:", (lam / lam.sum())[:3].round(3), "| de PCA:", pca.explained_variance_ratio_[:3].round(3))
""")
md("""
A mano y con `PCA` salen los mismos λ y los mismos vectores; **solo el signo es arbitrario**. La varianza explicada es λ/Σλ, con Σλ = 15.

### ✍️ Ejercicio C1
Para la matriz de correlación `[[1, 0.8], [0.8, 1]]` calcula λ y vectores propios con `np.linalg.eigh` y la varianza explicada de cada componente. Proyecta z = (1.2, 0.8).
""")
code("""
# tu código aquí
""")
solution("""
R = np.array([[1, 0.8], [0.8, 1]])
l, v = np.linalg.eigh(R)
l, v = l[::-1], v[:, ::-1]
print("λ:", l, "| varianza explicada:", l / l.sum())
print("vectores (columnas):\\n", v.round(3))
z = np.array([1.2, 0.8])
print("puntajes (v'z):", (v.T @ z).round(3))
""", "λ = 1.8 y 0.2 (1 ± r); varianza explicada 90 % y 10 %. Vectores (0.707, 0.707) y (0.707, −0.707) (el signo puede salir invertido). Puntajes: PC1 = 1.414, PC2 = ±0.283.")

md("""
### C2 · PCA y varianza explicada (15 min)
Ajustamos PCA con el **entrenamiento** y miramos el scree plot.
""")
code("""
var_exp = pca.explained_variance_ratio_
var_acum = np.cumsum(var_exp)
comps = np.arange(1, 16)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(comps, var_exp * 100, color="#1B4B66", label="% varianza individual")
ax.plot(comps, var_acum * 100, color="#E8873A", marker="o", label="% acumulada")
ax.axhline(80, ls=":", color="gray"); ax.axhline(90, ls=":", color="gray")
ax.set_xlabel("componente"); ax.set_ylabel("% de varianza"); ax.legend(); ax.set_title("Scree plot — datos de la clase")
plt.tight_layout(); plt.show()

print("Varianza por componente (%):", (var_exp * 100).round(1)[:9])
for u in (0.80, 0.85, 0.90):
    print(f"k para ≥ {u:.0%}: {int(np.argmax(var_acum >= u) + 1)}  (acumulada {var_acum[int(np.argmax(var_acum >= u))]:.1%})")
print(f"Con 7 componentes: {var_acum[6]:.1%}")
""")
md("""
PC1 explica 34.9 %; las barras bajan lentamente (8.9, 7.8, 7.4…): **no hay un codo claro**. Para ≥ 80 % hacen falta **8** componentes (85.0 %); con 7 solo se llega a **78.7 %** (la diapositiva «7 componentes ≈ 80 %» no cuadra con su tabla de 8). Con 9 se supera el 90 % (90.5 %). Es lo que dijo la profesora: «es un poco un arte».

### ✍️ Ejercicio C2
`PCA(n_components=0.90)` elige k automáticamente para una varianza dada. ¿Cuántos componentes selecciona y cuánta varianza conserva realmente?
""")
code("""
# tu código aquí
""")
solution("""
p90 = PCA(n_components=0.90, random_state=RS).fit(X_train)
print(p90.n_components_, round(float(p90.explained_variance_ratio_.sum()), 3))
""", "Selecciona 9 componentes y conserva 90.5 %: es el primer k cuya acumulada supera el umbral.")

md("""
### C3 · Elegir k con validación cruzada (20 min)
La varianza explicada es una pista; lo que importa es el **desempeño**. Hacemos que `n_components` sea un hiperparámetro de un `Pipeline` (**escalador → PCA → regresión logística**) y lo elegimos con `GridSearchCV` **solo con el entrenamiento**: en cada pliegue, el escalador y el PCA se reajustan sin ver el pliegue de validación (sin fuga).
""")
code("""
pipe = Pipeline([("esc", StandardScaler()), ("pca", PCA(random_state=RS)), ("lr", LogisticRegression(max_iter=2000))])
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
gs = GridSearchCV(pipe, {"pca__n_components": list(range(1, 16))}, cv=cv, scoring="roc_auc").fit(X_train_raw, y_train)

tabla = pd.DataFrame({"k": gs.cv_results_["param_pca__n_components"].astype(int),
                      "AUC medio": gs.cv_results_["mean_test_score"].round(3),
                      "desv.": gs.cv_results_["std_test_score"].round(3)}).set_index("k")
print(tabla.T.to_string())
print("\\nMejor k por AUC:", gs.best_params_["pca__n_components"], "| AUC =", round(gs.best_score_, 3))
""")
md("""
El AUC sube de 0.549 (k = 1) a **0.773 con k = 3** y luego se estabiliza, bajando un poco hasta 0.745 con 8 y con las 15 variables (0.746). No hace falta llegar al 80–90 % de varianza: con 3 componentes el modelo ya rinde igual o mejor **en validación cruzada**. La razón: PC3 (fallas y reclamos) es el componente con mayor relación con la fuga y los componentes 4 a 15 agregan sobre todo ruido. Ahora la comparación final **en la prueba** (¡una sola vez!):
""")
code("""
def evaluar(modelo, Xtr, Xte):
    modelo.fit(Xtr, y_train)
    p = modelo.predict(Xte)
    return {"F1": round(f1_score(y_test, p), 3), "MCC": round(matthews_corrcoef(y_test, p), 3),
            "AUC": round(roc_auc_score(y_test, modelo.predict_proba(Xte)[:, 1]), 3)}

filas = {"15 variables": evaluar(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_train_raw, X_test_raw)}
for k in (3, 8):
    filas[f"PCA {k} componentes"] = evaluar(make_pipeline(StandardScaler(), PCA(k, random_state=RS), LogisticRegression(max_iter=2000)), X_train_raw, X_test_raw)
pd.DataFrame(filas).T
""")
md("""
Con el **umbral 0.5** y 57 fugas en la prueba: 15 variables F1 0.270 / MCC 0.265; PCA 8, F1 0.267 / MCC 0.252 (los números de la diapositiva). La diferencia **no es significativa**: con tan pocas fugas, 1 o 2 aciertos cambian el F1 en el segundo decimal. La diapositiva concluye «se cede algo de desempeño»; aquí es **ruido**: en la prueba el AUC es 0.791 (15 variables), 0.786 (k = 3) y 0.776 (k = 8): un empate técnico. Y F1 ≈ 0.27 indica un modelo débil: no está usando `class_weight` ni ajustando el umbral (Clase 30).

### ✍️ Ejercicio C3
Aplica la **regla de una desviación estándar**: elige el **menor** k cuyo AUC medio esté dentro de una desviación del mejor. ¿Qué k sale?
""")
code("""
# tu código aquí
""")
solution("""
mejor = tabla["AUC medio"].idxmax()
umbral = tabla.loc[mejor, "AUC medio"] - tabla.loc[mejor, "desv."]
k_elegido = int(tabla[tabla["AUC medio"] >= umbral].index.min())
print("mejor k:", mejor, "| umbral:", round(float(umbral), 3), "| k elegido (regla 1-SE):", k_elegido)
""", "El mejor es k = 3 (0.773, desv. 0.019): umbral 0.754; el menor k que lo alcanza es 3. Es el modelo más simple que rinde igual que el mejor.")

md("""
### C4 · Cargas y círculo de correlaciones (15 min)
Cada fila de `components_` es un componente y sus números son las **cargas**. Con datos estandarizados, `carga × √λ` es la **correlación** de la variable con el componente.
""")
code("""
cols = list(X_train_raw.columns)
cargas = pd.DataFrame(pca.components_[:4].T, index=cols, columns=[f"PC{i}" for i in range(1, 5)]).round(2)
display(cargas)
for j in range(3):
    top = cargas.iloc[:, j].abs().sort_values(ascending=False).head(3).index.tolist()
    print(f"PC{j + 1} ({var_exp[j]:.1%}) se define sobre todo por:", top)

pcs = pca.transform(X_train)
print("\\nCorrelación de cada componente con la fuga (que PCA no usó):", [round(float(np.corrcoef(pcs[:, j], y_train)[0, 1]), 3) for j in range(5)])
""")
code("""
def circulo(pca_, columnas, figsize=(6.5, 6.5)):
    c = pca_.components_[:2].T * np.sqrt(pca_.explained_variance_[:2])          # correlaciones reales (sin reescalar)
    fig, ax = plt.subplots(figsize=figsize)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color="lightgray")
    for i, col in enumerate(columnas):
        ax.arrow(0, 0, c[i, 0], c[i, 1], color="#1B4B66", head_width=0.025, length_includes_head=True)
        if np.hypot(*c[i]) > 0.35:
            ax.text(c[i, 0] * 1.1, c[i, 1] * 1.1, col, fontsize=7, ha="center")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pca_.explained_variance_ratio_[0]:.1%})"); ax.set_ylabel(f"PC2 ({pca_.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Círculo de correlaciones"); plt.tight_layout(); plt.show()

circulo(pca, cols)
""")
md("""
- **PC1 (34.9 %)**: `segundos`, `frec_llamadas`, `n_contactos`, `clv`, `frec_sms`, `monto` con cargas 0.35–0.42: «**uso del servicio**».
- **PC2 (8.9 %)**: `antiguedad` (0.71) y `sms_por_antiguedad` (−0.60): «antigüedad vs. intensidad de SMS». La profesora dijo «tipo de plan y antigüedad».
- **PC3 (7.8 %)**: `numero_fallas` (0.72) y `reclamos` (0.50): «fricción con el servicio». **La diapositiva atribuye esto a PC2**; en este notebook es PC3.
- La fuga se correlaciona 0.33 con PC3 y −0.06 con PC1: PCA no usa la etiqueta, y el componente que más varianza explica no es el que mejor la predice.

### ✍️ Ejercicio C4
Calcula la correlación entre `segundos` y PC1 a partir de las cargas (`carga × √λ`) y compárala con `np.corrcoef` directo. ¿Coinciden?
""")
code("""
# tu código aquí
""")
solution("""
i = cols.index("segundos")
via_cargas = pca.components_[0][i] * np.sqrt(pca.explained_variance_[0])
directa = np.corrcoef(X_train[:, i], pcs[:, 0])[0, 1]
print(round(float(via_cargas), 3), round(float(directa), 3))
""", "Coinciden (≈ 0.97): con variables estandarizadas, carga × √λ es la correlación entre la variable y el componente.")

md("""
### C5 · La base real de la actividad de repaso (15 min)
Reproducimos con la **base real** (3.150 clientes) las cifras del caso de repaso. La limpieza es la del notebook de la profesora (edad imposible → mediana; faltantes con mediana o moda).
""")
code("""
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase 32*")))
carpeta = next(raiz.glob("Clase 32*"))
real = pd.read_excel(carpeta / "Base de datos Churn-Actividad repaso.xlsx")
print("Filas y columnas:", real.shape, "| tasa de fuga:", round(real["fuga"].mean(), 4), "| nulos declarados:", int(real.isna().sum().sum()))

def limpiar_clase(d):
    d = d.copy()
    d.loc[(d["edad"] < 0) | (d["edad"] > 100), "edad"] = np.nan
    for c in ["segundos", "frec_llamadas", "frec_sms", "n_contactos", "antiguedad", "edad"]:
        d[c] = d[c].fillna(d[c].median())
    d["reclamos"] = d["reclamos"].fillna(d["reclamos"].mode()[0])
    return d

rc = limpiar_clase(real)
yr = rc["fuga"]
Xr = construir(rc.drop(columns=["fuga"]))
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=RS, stratify=yr)
esc_r = StandardScaler().fit(Xr_tr)
pca_r = PCA(random_state=RS).fit(esc_r.transform(Xr_tr))
acum_r = np.cumsum(pca_r.explained_variance_ratio_)
print(f"PC1: {pca_r.explained_variance_ratio_[0]:.1%} | k para ≥ 80 %: {int(np.argmax(acum_r >= 0.8) + 1)} ({acum_r[int(np.argmax(acum_r >= 0.8))]:.1%}) | k para ≥ 90 %: {int(np.argmax(acum_r >= 0.9) + 1)}")

for nombre, modelo in {"15 variables": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                       "PCA 7 componentes": make_pipeline(StandardScaler(), PCA(7, random_state=RS), LogisticRegression(max_iter=2000))}.items():
    modelo.fit(Xr_tr, yr_tr)
    p = modelo.predict(Xr_te)
    print(f"{nombre:18s} F1 {f1_score(yr_te, p):.3f} | MCC {matthews_corrcoef(yr_te, p):.3f}")
""")
md("""
Coinciden con el caso de repaso: PC1 = **26.3 %**, **7 componentes = 80.4 %**, F1/MCC **0.574 / 0.546** con las 15 variables y **0.548 / 0.534** con PCA de 7. Aquí sí hay señal real (F1 más alto) y el recorte de componentes cuesta algo más. Ahora comprobamos la **pregunta 3.2** del caso: «numero_fallas y reclamos apuntan en dirección casi opuesta al grupo de uso».
""")
code("""
cr = rc.drop(columns=["fuga"]).corr()
print("Correlación de numero_fallas con segundos / frec_llamadas / monto:", cr.loc["numero_fallas", ["segundos", "frec_llamadas", "monto"]].round(2).tolist())
print("Correlación de reclamos con segundos / frec_llamadas / monto     :", cr.loc["reclamos", ["segundos", "frec_llamadas", "monto"]].round(2).tolist())
corr_r = pd.DataFrame(pca_r.components_[:2].T * np.sqrt(pca_r.explained_variance_[:2]), index=Xr.columns, columns=["PC1", "PC2"]).round(2)
corr_r.loc[["numero_fallas", "reclamos", "segundos", "frec_llamadas", "clv"]]
""")
md("""
**La premisa no se sostiene con los datos**: `numero_fallas` se correlaciona **positivamente** con `segundos` (+0.49), `frec_llamadas` (+0.57) y `monto` (+0.59, el propio caso lo reporta en la tabla de la pregunta 2.1), y su carga en PC1 es +0.61 (frente a 0.84 de `segundos`): apunta en la **misma** dirección que el uso. Solo `reclamos` es ligeramente negativo (−0.12 con `segundos`, correlación con PC1 de −0.17). La interpretación «más fallas, menos uso → fuga silenciosa» no está respaldada.

### ✍️ Ejercicio C5
¿Qué variables definen PC2 y PC3 en la **base real**? Compáralas con las del churn simulado y con lo que dice la diapositiva.
""")
code("""
# tu código aquí
""")
solution("""
cargas_r = pd.DataFrame(pca_r.components_[:3].T, index=Xr.columns, columns=["PC1", "PC2", "PC3"]).round(2)
for c in cargas_r.columns:
    print(c, cargas_r[c].abs().sort_values(ascending=False).head(4).index.tolist())
""", "En la base real, PC2 (16.3 %) está formado por `frec_sms` (0.52), `sms_por_antiguedad` (0.51) y `clv` (0.42); PC3 (9.4 %) por `intensidad_contacto` (0.65) y `duracion_prom_llamada` (0.56). Fallas y reclamos no definen PC2 en ninguno de los dos datasets: la diapositiva no coincide con ninguno.")

# ================================================================== PARTE D
md("""
## Parte D · t-SNE

t-SNE (*t-distributed stochastic neighbor embedding*, van der Maaten y Hinton, 2008) busca un mapa 2D donde los **vecinos** del espacio original queden cerca. Es no lineal, **no determinista** y no tiene `transform`.

### D1 · Perplexity (20 min)
`perplexity` ≈ a cuántos vecinos presta atención el algoritmo (se prueba entre 5 y 50 y debe ser menor que n). Comparamos con PCA en 2D y medimos qué tan bien conserva cada mapa a los **10 vecinos más cercanos** (`trustworthiness`).
""")
code("""
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
embs = {"PCA 2D": PCA(2, random_state=RS).fit_transform(X_train)}
for perp in (5, 30, 80):
    embs[f"t-SNE p={perp}"] = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=RS).fit_transform(X_train)

for ax, (nombre, e) in zip(axes, embs.items()):
    for val, color, lab in [(0, "#B9C4CC", "activo"), (1, "#E8873A", "fuga")]:
        m = y_train.values == val
        ax.scatter(e[m, 0], e[m, 1], s=6, alpha=0.6, color=color, label=lab)
    ax.set_title(nombre); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()

tw = {n: round(float(trustworthiness(X_train, e, n_neighbors=10)), 3) for n, e in embs.items()}
print("Trustworthiness (10 vecinos):", tw)
""")
md("""
La trustworthiness sube de **0.799** (PCA 2D) a **0.977–0.982** con t-SNE (perplexity 5, 30 y 80: 0.977, 0.982, 0.979): t-SNE conserva mucho mejor las **vecindades** porque es lo único que busca. Con perplexity 5 los grupos se parten en piezas pequeñas; con 80 hay menos islas. Los puntos de fuga **no forman un grupo propio**: t-SNE no usa la etiqueta, solo ve a los vecinos en las variables.

### ✍️ Ejercicio D1
Prueba `perplexity=1500` (mayor que n). ¿Qué ocurre? ¿Y con perplexity = 2?
""")
code("""
# tu código aquí
""")
solution("""
try:
    TSNE(perplexity=1500, init="pca", random_state=RS).fit_transform(X_train)
except Exception as e:
    print("perplexity=1500 ->", type(e).__name__, ":", str(e)[:90])
e2 = TSNE(perplexity=2, init="pca", learning_rate="auto", random_state=RS).fit_transform(X_train)
print("perplexity=2: trustworthiness", round(float(trustworthiness(X_train, e2, n_neighbors=10)), 3))
""", "Con 1.500 > n = 1.125 scikit-learn lanza un `ValueError`: la perplexity debe ser menor que el número de observaciones. Con 2 se obtiene un mapa muy fragmentado (trustworthiness 0.963, menor que con 30).")

md("""
### D2 · Semillas y límites de t-SNE (10 min)
t-SNE **no es determinista** (otra semilla, otro dibujo) y **no tiene `transform`**: no puede proyectar clientes nuevos.
""")
code("""
print("¿TSNE tiene transform?", hasattr(TSNE(), "transform"), "| ¿PCA?", hasattr(PCA(), "transform"))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, sem in zip(axes, (0, 1, 2)):
    e = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto", random_state=sem).fit_transform(X_train)
    for val, color in [(0, "#B9C4CC"), (1, "#E8873A")]:
        m = y_train.values == val
        ax.scatter(e[m, 0], e[m, 1], s=6, alpha=0.6, color=color)
    ax.set_title(f"perplexity 30 · semilla {sem}"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")
md("""
Los tres mapas muestran islas parecidas pero **en posiciones distintas**: la posición absoluta, el tamaño de las islas y la distancia entre ellas **no significan nada**. Ver islas tampoco prueba que existan clústeres (t-SNE no es K-means: no hay centroides).

### ✍️ Ejercicio D2
Para producción, un colega propone reemplazar PCA por t-SNE como paso previo al modelo. Da **dos** razones en contra.
""")
code("""
# tu código aquí
""")
solution("""
# Solo se puede ajustar y transformar a la vez: no hay forma de proyectar clientes nuevos
try:
    TSNE(perplexity=30).fit(X_train).transform(X_test)
except AttributeError as e:
    print("AttributeError:", e)
""", "(1) No tiene `transform`: cada cliente nuevo exigiría recalcular todo el mapa. (2) No es determinista y es costoso (compara pares de observaciones). Por eso t-SNE se usa para **explorar**, no dentro del Pipeline de producción.")

md("""
### D3 · PCA vs t-SNE: ¿cuál uso? (10 min)

| | PCA | t-SNE |
|---|---|---|
| Naturaleza | Lineal | No lineal |
| Qué conserva | Varianza global | Vecindades locales |
| Determinístico | Sí | No |
| `transform` de datos nuevos | Sí | No |
| Costo | Bajo | Alto |
| Uso típico | Reducir antes de modelar / producción | Explorar y visualizar |

### ✍️ Ejercicio D3
Elige **selección, feature engineering, PCA o t-SNE**: (a) seis variables de uso correlacionadas y un modelo liviano para producción; (b) ver si hay estructuras locales no lineales entre clientes; (c) distinguir llamadas largas de cortas con `segundos` y `frec_llamadas`; (d) una columna no aporta y complica la explicación.
""")
code("""
# tu respuesta aquí
""")
solution("""
respuestas = {"a": "PCA (o seleccionar una por grupo)", "b": "t-SNE (explorar)", "c": "feature engineering: segundos / frec_llamadas", "d": "selección (quitar la columna)"}
print(respuestas)
""", "Coincide con la actividad de repaso del apunte: (a) PCA, (b) t-SNE, (c) feature engineering, (d) selección.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene corregir

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Diapositiva 21 | «7 componentes (≈ 80 %)», pero la tabla dice 8 | En el churn simulado, 7 componentes = 78.7 % (no llega a 80) y 8 = 85.0 %. En la base real, 7 = 80.4 % (C2, C5). |
| Diapositivas 19 y apunte | «PC2 mezcla fallas y reclamos» | En el notebook son **PC3** (PC2 = antigüedad y sms_por_antiguedad); en la base real, PC2 = SMS y CLV (C4, C5). |
| Diapositiva 16 | «dataset real… 1.500 clientes» | Los 1.500 son **simulados**; la base real tiene 3.150. |
| Notebook, `USE_REAL_DATA = True` | lee `RUTA_CHURN` con `read_csv(sep=";")` | `RUTA_CHURN` no está definida en ninguna celda y el archivo es un .xlsx: falla. |
| Notebook, valores propios | matriz `[[10, −18], [6, −11]]` | No es simétrica: sus vectores no son ortogonales (producto 0.99). PCA diagonaliza una matriz **simétrica** de covarianzas; sus vectores son ortogonales (C1). |
| Diapositiva 21 y notebook | «con PCA se cede algo de desempeño» | F1 0.270 vs 0.267 con 57 fugas es ruido; en AUC, 3 componentes (0.773) superan a las 15 variables (0.746) (C3). |
| Diapositiva «Ordinal encoding» | «Nominal → entero cuando existe un orden real» | Debe decir **ordinal**. «Target / frequency encoding» son dos técnicas, y el target encoding calculado con todas las filas filtra la etiqueta (0.70 → 0.51, B3). |
| Clase | «PCA es un clustering» | No: PCA es una proyección lineal; no forma grupos. |
| Clase | «el 85 % de fuga… con 8 componentes cubrimos a los que se fugan» | Confunde el 85 % de **varianza** con el 85 % de clientes que **no** se fugan (la fuga es 15 %). No tienen relación. |
| Clase | «con train/test sale la misma descomposición que con tres conjuntos» | PCA depende de las filas con que se ajusta: otra división, otros componentes. |
| Clase | varianza = «promedio de distancias a la media»; desbalance = «sesgo» | Son distancias **al cuadrado**; y el desbalance de clases no es el «sesgo» del par sesgo–varianza. |
| Caso de repaso, pregunta 3.2 | fallas y reclamos apuntan «opuestos» al uso | En la base real `numero_fallas` correlaciona **positivamente** con el uso (+0.49 a +0.59) y carga +0.61 en PC1; solo `reclamos` es ligeramente negativo (C5). |
| Limpieza de la base real | solo corrige edad y faltantes | Quedan negativos imposibles (`segundos` hasta −14.645, `frec_llamadas` −176) y edades de 0 a 14; `monto` llega a 10 en 7 filas, y la diapositiva dice 0 a 9. |
| Diapositiva t-SNE | «Hinton & van der Maaten» | El orden de autores es van der Maaten y Hinton (2008). |

**Transcripción con ruido**: «TSN/TSM/TESEC/DCNE» = t-SNE, «pcea/pcl/PFA» = PCA, «laso/lazo» = Lasso, «reach» = Ridge, «Future» = Feature, «camins» = K-means, «Mathieu's» = Matthews, «yu-nyua» = Ñuñoa, «agent values» = eigenvalues. El recreo (≈ 1:03:29) no está en el video; hubo bucles de repetición, ya omitidos.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista de la telco y recibes la **base real de la actividad de repaso** (`real`, 3.150 clientes). Responde con código (usa `limpiar_clase`, `construir` y el mismo `train_test_split` de C5):

1. Separa train/test **estratificado** (25 %, `random_state=32`) y arma un `Pipeline` **escalador → PCA → regresión logística** (`class_weight="balanced"`).
2. Con `GridSearchCV` (5-Fold estratificado, `scoring="roc_auc"`) **solo con el entrenamiento**, elige `n_components` entre 1 y 15. Aplica la regla de una desviación estándar.
3. Compara en la **prueba** (una sola vez) el AUC del modelo elegido con el de las 15 variables y con PCA de 7 componentes.
4. Lee las cargas de PC1 y PC2 del PCA elegido (ajustado sobre entrenamiento escalado) y escribe una frase de interpretación **sin inventar**.
5. Con 1.500 clientes del entrenamiento, calcula un t-SNE con perplexity 30 y mide la trustworthiness. ¿Qué técnica usarías para producción y cuál para explorar?
6. Responde sin código: ¿qué corregirías del caso de repaso antes de dárselo a otra persona?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
Xr_tr2, Xr_te2, yr_tr2, yr_te2 = train_test_split(Xr, yr, test_size=0.25, random_state=RS, stratify=yr)
pipe_r = Pipeline([("esc", StandardScaler()), ("pca", PCA(random_state=RS)),
                   ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
cv_r = StratifiedKFold(5, shuffle=True, random_state=RS)
gs_r = GridSearchCV(pipe_r, {"pca__n_components": list(range(1, 16))}, cv=cv_r, scoring="roc_auc").fit(Xr_tr2, yr_tr2)
t_r = pd.DataFrame({"k": gs_r.cv_results_["param_pca__n_components"].astype(int), "m": gs_r.cv_results_["mean_test_score"], "s": gs_r.cv_results_["std_test_score"]}).set_index("k")
mejor_r = int(t_r["m"].idxmax())
k_r = int(t_r[t_r["m"] >= t_r.loc[mejor_r, "m"] - t_r.loc[mejor_r, "s"]].index.min())
print("mejor k:", mejor_r, "AUC", round(t_r.loc[mejor_r, "m"], 3), "| k (regla 1-SE):", k_r)

def auc_prueba(modelo):
    modelo.fit(Xr_tr2, yr_tr2)
    return round(roc_auc_score(yr_te2, modelo.predict_proba(Xr_te2)[:, 1]), 3)

print("AUC prueba | 15 variables:", auc_prueba(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))))
print("AUC prueba | PCA 7        :", auc_prueba(make_pipeline(StandardScaler(), PCA(7, random_state=RS), LogisticRegression(max_iter=2000, class_weight="balanced"))))
print(f"AUC prueba | PCA {k_r}        :", auc_prueba(make_pipeline(StandardScaler(), PCA(k_r, random_state=RS), LogisticRegression(max_iter=2000, class_weight="balanced"))))

Zr = StandardScaler().fit_transform(Xr_tr2)
pk = PCA(k_r, random_state=RS).fit(Zr)
cg = pd.DataFrame(pk.components_[:2].T, index=Xr.columns, columns=["PC1", "PC2"]).round(2)
print(cg.reindex(cg["PC1"].abs().sort_values(ascending=False).index).head(5))
print(cg["PC2"].abs().sort_values(ascending=False).head(4).round(2).to_dict())

muestra = np.random.default_rng(RS).choice(len(Zr), 1500, replace=False)
e_r = TSNE(perplexity=30, init="pca", learning_rate="auto", random_state=RS).fit_transform(Zr[muestra])
print("trustworthiness t-SNE:", round(float(trustworthiness(Zr[muestra], e_r, n_neighbors=10)), 3))
""", "Ver «Resultado de referencia» en la celda siguiente. Correcciones al material: (1) la pregunta 3.2 no se sostiene (fallas correlaciona positivamente con el uso); (2) «7 componentes ≈ 80 %» con tabla de 8; (3) PC2 no es fallas y reclamos; (4) «PCA es un clustering» y el 85 % de fuga/varianza; (5) elegir k con validación cruzada dentro de un Pipeline y no con F1 de una sola partición; (6) limpiar los negativos imposibles antes de PCA; (7) el target encoding exige validación interna.")
md("""
**Resultado de referencia** (base real, `random_state=32`): el AUC en validación cruzada sigue subiendo con k (mejor k = 15, AUC 0.921) y la regla de una desviación estándar elige **k = 11**; en la prueba el AUC es **0.923** con las 15 variables, **0.915** con PCA de 7 y **0.922** con PCA de 11: aquí PCA no mejora el desempeño, solo reduce columnas, a diferencia del churn simulado. Cargas de PC1 (26.3 % de varianza): `frec_llamadas` 0.43, `segundos` 0.42, `n_contactos` 0.36, `clv` 0.34 y `numero_fallas` 0.31 (mismo signo: las fallas acompañan al uso, no se oponen); PC2: `frec_sms` 0.52, `sms_por_antiguedad` 0.51, `clv` 0.42. t-SNE con perplexity 30 y 1.500 clientes: trustworthiness 0.994. Para **producción**, PCA (o las 15 variables); para **explorar**, t-SNE.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. Da tres razones por las que demasiadas variables perjudican a un modelo.
2. ¿Por qué se estandariza antes de PCA y con qué datos se hace el `fit`?
3. ¿Qué son los valores y vectores propios en PCA y por qué los vectores son ortogonales?
4. ¿Cómo eliges k? ¿Qué ventaja tiene la validación cruzada sobre el umbral de varianza?
5. ¿Qué diferencia hay entre ordinal, one-hot y target encoding, y cuál puede filtrar la etiqueta?
6. ¿Cuándo usarías PCA y cuándo t-SNE? ¿Por qué t-SNE no va en producción?

<details><summary><b>Respuestas</b></summary>

1. Costo computacional, multicolinealidad (inestabilidad; VIF alto), ruido (sobreajuste con pocas filas) y distancias que se aplanan.
2. PCA maximiza varianza: sin estandarizar domina la variable de mayor unidad (aquí `segundos` = 99.2 %). El escalador se ajusta **solo con entrenamiento**, dentro del Pipeline.
3. Los vectores propios de la covarianza son las direcciones de los componentes; λ es la varianza de cada uno. Como la covarianza es simétrica, sus vectores son ortogonales.
4. Con un umbral de varianza (80–90 %), el codo, o mejor con el desempeño (AUC) en validación cruzada de un Pipeline. La CV mide lo que importa (predecir), no solo la varianza.
5. Ordinal: enteros con orden real. One-hot: una columna por categoría, sin orden. Target: tasa de la etiqueta por categoría; puede filtrar la etiqueta si se calcula con todas las filas.
6. PCA: reducir antes de modelar y producción (determinista, con `transform`). t-SNE: explorar la estructura de vecindad; no tiene `transform`, no es determinista y es costoso.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 33 · Feature engineering y selección de variables. Revisa `Guias/apuntes/C32_Ejercicios_Cuaderno_ReduccionDimensionalidad.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
