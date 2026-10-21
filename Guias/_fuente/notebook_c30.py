"""Genera el notebook guiado de Preparación para Machine Learning (Clase 30).

Uso normal:      python notebook_c30.py             -> Guias/notebooks/C30_PreparacionML.ipynb
Modo de prueba:  C30_TEST=1 python notebook_c30.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C30_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C30_PreparacionML_test.ipynb" if TEST else "C30_PreparacionML.ipynb")
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
# Clase 30 · Preparación para Machine Learning: variables, datos y validación
**Diplomado Data Science** · Profa. Laura Sánchez

Material de apoyo:
- `Guias/C30_Guia_PreparacionML.pdf` — guía visual de la clase.
- `Guias/apuntes/C30_Apuntes_PreparacionML.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C30_Ejercicios_Cuaderno_PreparacionML.pdf` — ejercicios en papel (antes o después de este notebook).
- Material de la profesora: `Sesion_30_Preparacion_ML.ipynb` y `.pdf`, `Apunte de Estudio Preparación ML.pdf`, `Caso de aplicación para repaso.pdf` y sus respuestas.
- Transcripción: `VideosClases/Clase 30  Preparacion ML variables, datos y validacion.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `pandas`, `numpy` y `matplotlib`. Si no los tienes:
```
pip install scikit-learn pandas numpy matplotlib
```
**No hay archivos de datos que descargar**: los datos de la profesora son **simulados** y se generan con código (300 viviendas con `np.random.seed(42)` y 2.000 casos de fraude con `make_classification`). Aquí se generan igual y en el mismo orden, así que obtienes sus mismos números.

### Objetivos
1. Reconocer los **tipos de variables**, imputar datos faltantes con criterio y diagnosticar la **calidad** de un dato.
2. Dividir en **entrenamiento / validación / prueba**, usar la validación para decidir y reconocer el **sobreajuste**.
3. Entender y evitar la **fuga de datos** (*data leakage*) usando `Pipeline` y `ColumnTransformer`.
4. Ver por qué se **estandariza** antes de Ridge y Lasso, y evaluar con **K-Fold** y **estratificación**.
5. Calcular **accuracy, precision, recall y F1** (a mano y con scikit-learn), mover el umbral y detectar lo que el material original afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y mini-guía de la API de scikit-learn |
| A | 10 | A1 · Tipos de variables y one-hot |
| A | 15 | A2 · Datos faltantes e imputación |
| A | 10 | A3 · Dimensiones de calidad (coherencia) |
| B | 15 | B1 · Entrenamiento, validación y prueba |
| B | 10 | B2 · Sobreajuste: la brecha |
| B | 20 | B3 · Fuga de datos: dos experimentos |
| B | 15 | B4 · Pipeline y ColumnTransformer |
| C | 20 | C1 · Ridge y Lasso: antes y después de estandarizar |
| C | 15 | C2 · Elegir alpha con la validación |
| C | 15 | C3 · K-Fold |
| D | 15 | D1 · Muestreo estratificado |
| D | 20 | D2 · La trampa del accuracy y la matriz de confusión |
| D | 15 | D3 · El umbral y el equilibrio precision / recall |
| D | 10 | D4 · Métricas de regresión |
| — | 10 | Alertas del material |
| — | 30 | Ejercicio de cierre **sin ayuda** (scoring de crédito) |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook con `scikit-learn 1.9`. Con otras versiones pueden variar un poco.
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 42   # semilla
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Los datos de la clase (simulados)
Se generan **exactamente igual que en el notebook de la profesora**, con la misma semilla y en el mismo orden de llamadas aleatorias (por eso conviene no cambiar el orden de esta celda). El precio depende de la superficie, las habitaciones y la antigüedad, más ruido.
""")
code("""
np.random.seed(42)   # semilla global, como en la clase
n = 300
viviendas = pd.DataFrame({
    "superficie_m2": np.random.normal(90, 25, n).round(1),
    "n_habitaciones": np.random.choice([1, 2, 3, 4, 5], n),
    "barrio": np.random.choice(["Centro", "Norte", "Sur", "Este"], n),
    "antiguedad_años": np.random.randint(0, 50, n),
    "tiene_ascensor": np.random.choice(["Sí", "No"], n),
})
viviendas["precio_clp"] = (
    2_200_000 * viviendas["superficie_m2"]
    + 8_000_000 * viviendas["n_habitaciones"]
    - 900_000 * viviendas["antiguedad_años"]
    + np.random.normal(0, 15_000_000, n)
).round(0)

# Bloque 2 de la clase: a 20 viviendas les falta la antigüedad
viviendas_sucias = viviendas.copy()
faltantes = np.random.choice(viviendas_sucias.index, size=20, replace=False)
viviendas_sucias.loc[faltantes, "antiguedad_años"] = np.nan

# Fraude (Bloque 6): 2.000 casos, 5 % de positivos
X_fraude, y_fraude = make_classification(n_samples=2000, n_features=5, weights=[0.95, 0.05], random_state=42)

print("viviendas:", viviendas.shape, "| faltantes de antigüedad:", int(viviendas_sucias["antiguedad_años"].isna().sum()))
print("fraude   :", X_fraude.shape, "| positivos:", int(y_fraude.sum()), f"({y_fraude.mean():.3%})")
viviendas.head()
""")

md("""
## Mini-guía de la API de scikit-learn (léela antes de seguir)
Es tu primera clase de Machine Learning y scikit-learn será tu herramienta. Casi todo el código de hoy es una combinación de **tres verbos**:

| Verbo | Qué hace | ¿Con qué datos? |
|---|---|---|
| `fit(X, y)` | **Aprende** de los datos (la media y desviación de un escalador, los coeficientes de un modelo) | **solo entrenamiento** |
| `transform(X)` | **Aplica** lo aprendido (escala, imputa, codifica) | entrenamiento, validación, prueba |
| `predict(X)` | **Predice** con un modelo ya ajustado | validación, prueba, datos nuevos |

Reglas de oro: (1) **`fit` solo con entrenamiento**; (2) `transform` y `predict` se pueden usar con cualquier dato; (3) un `Pipeline` encadena pasos y hace el `fit` de cada uno solo con los datos que recibe; (4) un `ColumnTransformer` aplica un tratamiento distinto a cada tipo de columna. Lo veremos paso a paso.
""")
code("""
# El verbo fit / transform en miniatura
entrenamiento = np.array([[10.0], [12.0], [14.0]])
prueba = np.array([[30.0]])

escalador = StandardScaler()
escalador.fit(entrenamiento)                      # APRENDE: media y desviación del entrenamiento
print("media aprendida:", escalador.mean_, "| desviación:", escalador.scale_.round(3))
print("entrenamiento escalado:", escalador.transform(entrenamiento).ravel().round(3))
print("prueba escalada       :", escalador.transform(prueba).ravel().round(3), " <- el 30 se ve como el atípico que es")
""")
md("""
Con las estadísticas del **entrenamiento** (media 12, desviación 1.633) el valor 30 queda en z = 11.0: un atípico claro. Guarda esa idea: en B3 veremos qué pasa si el escalador se ajusta con **todo** el dataset.
""")

# ================================================================== PARTE A
md("""
## Parte A · Variables y calidad de los datos

### A1 · Tipos de variables y one-hot (10 min)
Los modelos solo entienden números. Primero hay que saber **qué tipo** es cada columna.
""")
code("""
print(viviendas.dtypes)
print()
print("barrio (nominal)  :", viviendas["barrio"].unique())
print("n_habitaciones    :", sorted(viviendas["n_habitaciones"].unique()), "(numérica discreta)")

# One-hot con pandas: una columna 0/1 por categoría
dummies = pd.get_dummies(viviendas[["barrio", "tiene_ascensor"]], dtype=int)
print("\\nColumnas tras el one-hot:", list(dummies.columns))
dummies.head(3)
""")
md("""
- `superficie_m2`, `antiguedad_años`, `precio_clp` son numéricas; `n_habitaciones` es **numérica discreta** (en el código toma valores de 1 a 5; el texto de la clase decía «entre 1 y 6»).
- `barrio` y `tiene_ascensor` son **categóricas**. `barrio` es **nominal** (no hay orden); «básico/medio/alto» sería **ordinal**.
- El **one-hot** convierte cada categoría en una columna 0/1. `barrio` pasa de 1 columna a 4.

### ✍️ Ejercicio A1
(a) Con `OneHotEncoder(drop="first", sparse_output=False)` aplicado a `barrio`, ¿cuántas columnas obtienes? ¿Por qué se elimina una? (b) Clasifica `antiguedad_años` y `tiene_ascensor`.
""")
code("""
# tu código aquí
""")
solution("""
enc = OneHotEncoder(drop="first", sparse_output=False)
cols = enc.fit_transform(viviendas[["barrio"]])
print("Columnas:", cols.shape[1], enc.get_feature_names_out())
""", "(a) 3 columnas: con 4 barrios basta con 3 columnas 0/1 (si las tres valen 0, es el barrio eliminado); evita columnas redundantes. (b) `antiguedad_años`: numérica discreta; `tiene_ascensor`: categórica binaria (una columna 0/1).")

md("""
### A2 · Datos faltantes e imputación (15 min)
En el Bloque 2 a 20 viviendas les falta la antigüedad (6.7 %). Reglas de la clase: **media** (numérica simétrica, sin atípicos), **mediana** (con atípicos o asimetría), **moda** (categórica), **regresión** (si se relaciona con otras columnas). Y **no imputar** si falta un tercio de los datos: ahí hay un problema de recolección.
""")
code("""
col = viviendas_sucias["antiguedad_años"]
print(f"Faltantes: {col.isna().sum()} de {len(col)} ({col.isna().mean():.1%})")
print("media observada:", round(col.mean(), 2), "| mediana observada:", col.median())

relleno = col.fillna(col.mean())
print("\\nDesviación estándar antes de imputar:", round(col.std(), 2))
print("Desviación estándar tras imputar con la media:", round(relleno.std(), 2), " <- baja: 20 filas con el mismo valor")
print("Desviación estándar de la columna completa (real):", round(viviendas["antiguedad_años"].std(), 2))
""")
md("""
Imputar con la media **deja la desviación más pequeña**: las 20 filas rellenas caen en el mismo valor (14.19 → 13.71; la real es 14.28). Con la media de una variable simétrica el sesgo es pequeño, pero es una pista de que imputar siempre pierde información.

La forma correcta en scikit-learn es `SimpleImputer`, con el verbo `fit` **solo en entrenamiento**:
""")
code("""
X_ant = viviendas_sucias[["antiguedad_años"]]
Xa_tr, Xa_te = train_test_split(X_ant, test_size=0.25, random_state=RS)

imp = SimpleImputer(strategy="median")
imp.fit(Xa_tr)                                   # la mediana se aprende SOLO con el entrenamiento
print("mediana aprendida:", imp.statistics_)
print("faltantes en prueba antes :", int(Xa_te.isna().sum().iloc[0]))
print("faltantes en prueba después:", int(np.isnan(imp.transform(Xa_te)).sum()))

# Caso del banco (diapositivas): ¿qué estadístico?
print("\\nIngreso con cola larga -> mediana | Edad simétrica -> media | Comuna -> moda ('most_frequent')")
""")
md("""
### ✍️ Ejercicio A2
Ingresos observados (millones de CLP): `[1.2, 1.4, 1.5, 1.6, 9.0]` y un valor faltante. Calcula la media y la mediana de los observados y decide con cuál imputarías y por qué.
""")
code("""
# tu código aquí
""")
solution("""
ingresos = pd.Series([1.2, 1.4, 1.5, 1.6, 9.0])
print("media:", round(ingresos.mean(), 2), "| mediana:", ingresos.median())
""", "Media 2.94, mediana 1.5. El 9.0 arrastra la media; se imputa con la mediana, que representa mejor al cliente típico.")

md("""
### A3 · Dimensiones de calidad: la coherencia (10 min)
Las cuatro dimensiones: **exactitud**, **completitud**, **coherencia**, **frescura**. Simulemos el caso del banco: el `monto_solicitado` en la web no siempre coincide con el del contrato firmado.
""")
code("""
rng = np.random.default_rng(0)
monto_contrato = rng.integers(2, 20, 10) * 1_000_000
monto_web = monto_contrato.copy()
monto_web[[2, 7]] += 500_000                       # dos registros no coinciden (SIMULADO)

banco = pd.DataFrame({"contrato": monto_contrato, "web": monto_web})
banco["coincide"] = banco["contrato"] == banco["web"]
print(banco)
print(f"\\nRegistros con fuentes distintas: {(~banco['coincide']).mean():.0%}")
""")
md("""
Dos fuentes del mismo dato **no concuerdan**: por la definición de la propia clase («¿las distintas fuentes concuerdan entre sí?»), el problema es de **coherencia**. La **exactitud** aparece después, al decidir cuál valor es la verdad (el contrato firmado es el *ground truth*). La respuesta oficial del caso dice «Exactitud»; en el examen conviene mencionar ambas.
""")

# ================================================================== PARTE B
md("""
## Parte B · División de datos, sobreajuste y fuga

### B1 · Entrenamiento, validación y prueba (15 min)
La división 70/15/15 de la profesora se hace con **dos** llamadas a `train_test_split`: primero 70 % / 30 %, luego ese 30 % por la mitad.
""")
code("""
X = viviendas[["superficie_m2", "n_habitaciones", "antiguedad_años"]]
y = viviendas["precio_clp"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Entrenamiento: {len(X_train)} ({len(X_train) / len(X):.0%}) | Validación: {len(X_val)} ({len(X_val) / len(X):.0%}) | Prueba: {len(X_test)} ({len(X_test) / len(X):.0%})")
""")
md("""
La profesora crea `X_val` pero **no la usa**. Para qué sirve: **elegir** entre alternativas sin mirar la prueba. Probemos tres modelos con la validación y recién después reportemos la prueba **del modelo elegido**:
""")
code("""
def rmse(y_real, y_pred):
    return float(np.sqrt(mean_squared_error(y_real, y_pred)))

candidatos = {
    "Regresión Ridge (alpha=10)": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Ridge (alpha=1000)": make_pipeline(StandardScaler(), Ridge(alpha=1000)),
    "Árbol (profundidad 3)": DecisionTreeRegressor(max_depth=3, random_state=0),
}
for nombre, modelo in candidatos.items():
    modelo.fit(X_train, y_train)
    print(f"{nombre:28s} RMSE validación: {rmse(y_val, modelo.predict(X_val)):>14,.0f}")

mejor = candidatos["Regresión Ridge (alpha=10)"]
print(f"\\nModelo elegido por validación -> RMSE en PRUEBA (una sola vez): {rmse(y_test, mejor.predict(X_test)):,.0f}")
""")
md("""
La elección salió de la validación; la prueba solo **informa** el resultado final. Si eligieras mirando la prueba, dejaría de ser una medición independiente: eso también es fuga.

### ✍️ Ejercicio B1
Con 5.000 registros y una división 80/10/10: (a) ¿cuántos hay en cada conjunto? (b) escribe las dos llamadas a `train_test_split` con los `test_size` correctos.
""")
code("""
# tu código aquí
""")
solution("""
N = 5000
idx = np.arange(N)
tr, temp = train_test_split(idx, test_size=0.20, random_state=RS)
va, te = train_test_split(temp, test_size=0.50, random_state=RS)
print(len(tr), len(va), len(te))
""", "(a) 4.000 / 500 / 500. (b) primera llamada `test_size=0.20`, segunda `test_size=0.50` sobre ese 20 %.")

md("""
### B2 · Sobreajuste: la brecha (10 min)
El modelo **memoriza** el entrenamiento y falla con datos nuevos. La señal: la brecha entre entrenamiento y validación. Un árbol de decisión permite subir la complejidad a voluntad (profundidad).
""")
code("""
prof = list(range(1, 13))
tr, va = [], []
for d in prof:
    arbol = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    tr.append(arbol.score(X_train, y_train))
    va.append(arbol.score(X_val, y_val))

mejor_d = prof[int(np.argmax(va))]
print("profundidad | R² entrenamiento | R² validación | brecha")
for d, a, b in zip(prof, tr, va):
    print(f"{d:11d} | {a:16.3f} | {b:13.3f} | {a - b:6.3f}")
print("\\nMejor validación: profundidad", mejor_d)
plt.plot(prof, tr, "-o", label="entrenamiento")
plt.plot(prof, va, "-o", label="validación")
plt.axvline(mejor_d, color="g", ls="--")
plt.xlabel("profundidad del árbol")
plt.ylabel("R²")
plt.legend()
plt.show()
""")
md("""
El entrenamiento llega a 0.999 y la validación se queda en ≈ 0.80 (su máximo, 0.847, es en profundidad 7). **No elegimos el modelo que mejor ajusta el entrenamiento**, sino el que mejor generaliza. Dos aclaraciones:
- En clase se dijo que una validación **mejor** que el entrenamiento «es raro». Con solo 45 filas ocurre con frecuencia (ver Alertas): es una señal para revisar, no un error.
- Las cifras de la diapositiva (98 % vs 61 %) son ilustrativas; aquí tienes una curva real.

### ✍️ Ejercicio B2
Repite con `max_depth` de 1 a 5 usando `min_samples_leaf=10` en lugar de la profundidad libre. ¿Se reduce la brecha? ¿A qué costo?
""")
code("""
# tu código aquí
""")
solution("""
for d in (1, 3, 5, None):
    a = DecisionTreeRegressor(max_depth=d, min_samples_leaf=10, random_state=0).fit(X_train, y_train)
    print(d, round(a.score(X_train, y_train), 3), round(a.score(X_val, y_val), 3))
""", "Con hojas de al menos 10 filas el árbol sin límite de profundidad ya no memoriza (entrenamiento ≈ 0.9), así que la brecha se acorta. El costo es perder detalle: sube el sesgo.")

md("""
### B3 · Fuga de datos (*data leakage*): dos experimentos (20 min)
**Fuga** = usar información de validación o prueba durante el desarrollo (imputar, escalar, seleccionar variables, elegir hiperparámetros). El resultado se ve **mejor de lo real**. El error clásico es hacer el preprocesamiento con **todo** el dataset y dividir después.

**Experimento 1 · el escalador.** Con el ejemplo de A (entrenamiento 10, 12, 14; prueba 30):
""")
code("""
datos = np.array([[10.0], [12.0], [14.0], [30.0]])
z_ok = (30 - datos[:3].mean()) / datos[:3].std()            # estadísticos SOLO del entrenamiento
z_fuga = (30 - datos.mean()) / datos.std()                  # estadísticos con las 4 filas (fuga)
print(f"z correcto (fit solo con entrenamiento): {z_ok:.2f}  -> atípico claro")
print(f"z con fuga (fit con todo)              : {z_fuga:.2f}  -> parece normal")
""")
md("""
Con la fuga, el atípico **desaparece**: el escalador ya «vio» la prueba. Es un efecto pequeño con datos grandes y normales, pero conceptualmente incorrecto. El segundo experimento muestra una fuga con consecuencias enormes.

**Experimento 2 · selección de variables sobre ruido.** 100 filas, 1.000 columnas de **ruido puro** y etiquetas al azar: ningún modelo honesto puede pasar de ≈ 50 % de exactitud. Seleccionamos las 10 «mejores» columnas y validamos con K-Fold, de dos formas:
""")
code("""
fuga, honesto = [], []
for s in range(40):
    rng = np.random.default_rng(s)
    Xn = rng.normal(size=(100, 1000))
    yn = rng.integers(0, 2, 100)
    # MAL: selecciona con TODAS las filas y luego valida (fuga)
    Xsel = SelectKBest(f_classif, k=10).fit_transform(Xn, yn)
    fuga.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=5).mean())
    # BIEN: la selección está dentro del Pipeline y se re-aprende en cada pliegue
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("m", LogisticRegression(max_iter=1000))])
    honesto.append(cross_val_score(pipe, Xn, yn, cv=5).mean())

print(f"Con fuga       : exactitud media {np.mean(fuga):.3f} (de {np.min(fuga):.2f} a {np.max(fuga):.2f})")
print(f"Con Pipeline   : exactitud media {np.mean(honesto):.3f} (de {np.min(honesto):.2f} a {np.max(honesto):.2f})")
""")
md("""
Con fuga el modelo «acierta» ≈ 79 % con datos que **no contienen ninguna señal**; con el Pipeline da ≈ 51 %, el azar. La selección vio las etiquetas de **todas** las filas, incluidas las de validación. Un `Pipeline` repite el `fit` de cada paso **solo con el entrenamiento de cada pliegue**.

### ✍️ Ejercicio B3
Repite el experimento 1 con entrenamiento `[50, 55, 60]` y prueba `[90]`. Calcula el z correcto y el z con fuga y explica en una frase por qué difieren.
""")
code("""
# tu código aquí
""")
solution("""
datos = np.array([50.0, 55.0, 60.0, 90.0])
z_ok = (90 - datos[:3].mean()) / datos[:3].std()
z_fuga = (90 - datos.mean()) / datos.std()
print(round(z_ok, 2), round(z_fuga, 2))
""", "z correcto = 8.57 y z con fuga = 1.69. Con fuga la media y la desviación incluyen al propio 90, que se «esconde» y deja de verse atípico.")

md("""
### B4 · Pipeline y ColumnTransformer (15 min)
Juntamos todo: las 300 viviendas tienen columnas numéricas (con faltantes) y categóricas. Un **`ColumnTransformer`** aplica un tratamiento por tipo de columna y un **`Pipeline`** lo encadena con el modelo. Se llama a `.fit` **una sola vez**, con el entrenamiento.
""")
code("""
num = ["superficie_m2", "n_habitaciones", "antiguedad_años"]
cat = ["barrio", "tiene_ascensor"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
modelo = Pipeline([("prep", prep), ("ridge", Ridge(alpha=10))])

Xs = viviendas_sucias[num + cat]               # con los 20 faltantes
ys = viviendas_sucias["precio_clp"]
Xs_tr, Xs_tmp, ys_tr, ys_tmp = train_test_split(Xs, ys, test_size=0.30, random_state=42)
Xs_va, Xs_te, ys_va, ys_te = train_test_split(Xs_tmp, ys_tmp, test_size=0.50, random_state=42)

modelo.fit(Xs_tr, ys_tr)                       # imputa, escala, codifica y ajusta: TODO con el entrenamiento
print("R² validación:", round(modelo.score(Xs_va, ys_va), 3))
print("Columnas tras preparar:", modelo.named_steps["prep"].get_feature_names_out().tolist())
""")
md("""
Fíjate en lo que **no** escribiste: ningún `fit_transform` suelto sobre el dataset completo. Al llamar `predict`, el pipeline aplica **con lo aprendido en entrenamiento** el mismo preprocesamiento a los datos nuevos (incluso a los que traen faltantes).

### ✍️ Ejercicio B4
Cambia el imputador numérico por `strategy="mean"` y el modelo por `Lasso(alpha=2)`. ¿Cambia el R² de validación? ¿Qué coeficientes quedan en cero?
""")
code("""
# tu código aquí
""")
solution("""
prep2 = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("esc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
m2 = Pipeline([("prep", prep2), ("lasso", Lasso(alpha=2, max_iter=10000))]).fit(Xs_tr, ys_tr)
nombres = m2.named_steps["prep"].get_feature_names_out()
coef = pd.Series(m2.named_steps["lasso"].coef_, index=nombres).round(0)
print("R² validación:", round(m2.score(Xs_va, ys_va), 3))
print(coef[coef == 0])
""", "Como el precio está en CLP (decenas de millones), alpha = 2 casi no castiga: Lasso se parece a la regresión sin regularizar: solo anula una dummy (`tiene_ascensor_Sí`) y el R² de validación queda en 0.906. Otra razón para fijarte en la **escala del objetivo** al elegir alpha.")

# ================================================================== PARTE C
md("""
## Parte C · Regularización y validación cruzada

### C1 · Ridge y Lasso: antes y después de estandarizar (20 min)
**Regularizar** = castigar la complejidad. **Ridge (L2)** suma `α·Σβ²` al error: achica los coeficientes. **Lasso (L1)** suma `α·Σ|β|`: puede llevarlos **exactamente a cero**. Agregamos `ruido_aleatorio`, una variable que no participa en el precio.
""")
code("""
X_reg = X.copy()
X_reg["ruido_aleatorio"] = np.random.default_rng(42).normal(0, 1, len(X_reg))
y_reg = y / 1_000_000                                      # precio en millones de CLP

Xr_tr, Xr_tmp, yr_tr, yr_tmp = train_test_split(X_reg, y_reg, test_size=0.30, random_state=42)
Xr_val, Xr_te, yr_val, yr_te = train_test_split(Xr_tmp, yr_tmp, test_size=0.50, random_state=42)

def tabla_coef(con_escala):
    modelos = {"Sin regularizar": LinearRegression(),
               "Ridge (alpha=10)": Ridge(alpha=10), "Lasso (alpha=2)": Lasso(alpha=2, max_iter=10000)}
    salida = pd.DataFrame(index=X_reg.columns)
    for nombre, m in modelos.items():
        pipe = make_pipeline(StandardScaler(), m) if con_escala else make_pipeline(m)
        pipe.fit(Xr_tr, yr_tr)
        salida[nombre] = pipe[-1].coef_.round(3)
    return salida

print("4A. SIN estandarizar:")
display(tabla_coef(False))
print("4B. CON estandarización:")
display(tabla_coef(True))
""")
md("""
Lo que muestran las tablas (idénticas a las de la clase):
- **Escala**: sin estandarizar, la superficie vale 2.18 y las habitaciones 8.19; estandarizadas, 56.4 y 11.9. Las dos tablas **no son comparables entre sí**: cambian las unidades.
- **Ridge** achica (56.40 → 53.80) y deja el ruido en −0.44. **Lasso** lo lleva a **0**.
- ⚠️ En clase se dijo que sin estandarizar «es difícil que Lasso lleve a cero». En esta demo **lo lleva a cero en ambos casos** (4A: −0.000). El daño de no estandarizar es otro: el castigo **depende de las unidades**. Comprobémoslo.
""")
code("""
# Misma regresión Ridge; solo cambia la unidad de la superficie: m² -> centenas de m²
Xr_tr2, Xr_val2 = Xr_tr.copy(), Xr_val.copy()
Xr_tr2["superficie_m2"] /= 100
Xr_val2["superficie_m2"] /= 100

for etiqueta, (A, B) in {"superficie en m²": (Xr_tr, Xr_val), "superficie en centenas de m²": (Xr_tr2, Xr_val2)}.items():
    sin = Ridge(alpha=30).fit(A, yr_tr)
    con = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(A, yr_tr)
    print(f"{etiqueta:30s} R² validación: sin estandarizar {sin.score(B, yr_val):.3f} | estandarizado {con.score(B, yr_val):.3f}")
""")
md("""
Sin estandarizar, **cambiar las unidades cambia el modelo** (0.916 → 0.532): la variable en «centenas de m²» necesita un coeficiente 100 veces mayor y el castigo lo golpea. Estandarizando, el modelo es **idéntico** (0.887) sea cual sea la unidad. Esa es la razón de fondo para estandarizar antes de Ridge o Lasso.

> «Estandarizar no significa arreglar un modelo malo»: solo pone las variables en una escala comparable.

### ✍️ Ejercicio C1
Haz el **camino de Lasso**: con las variables estandarizadas, calcula los coeficientes para `alpha` en `[0.1, 0.4, 1, 5, 15, 70]`. ¿Qué variable se anula primero? ¿Cuál resiste más?
""")
code("""
# tu código aquí
""")
solution("""
filas = {}
for al in (0.1, 0.4, 1, 5, 15, 70):
    m = make_pipeline(StandardScaler(), Lasso(alpha=al, max_iter=50000)).fit(Xr_tr, yr_tr)
    filas[al] = m[-1].coef_.round(2)
print(pd.DataFrame(filas, index=X_reg.columns).T)
""", "El ruido se anula primero (alpha ≈ 0.4). Después habitaciones y antigüedad (alpha ≈ 12.6) y la superficie, la más importante, resiste hasta alpha ≈ 63.")

md("""
### C2 · Elegir alpha con la validación (15 min)
`alpha` es un **hiperparámetro**: no lo aprende el modelo, lo eliges tú, y se elige con la **validación**.
""")
code("""
resultados = []
for al in (0.01, 0.1, 1, 10, 30, 100, 300):
    m = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(Xr_tr, yr_tr)
    resultados.append((al, m.score(Xr_tr, yr_tr), m.score(Xr_val, yr_val)))
res = pd.DataFrame(resultados, columns=["alpha", "R² entrenamiento", "R² validación"]).round(3)
display(res)
mejor_alpha = float(res.loc[res["R² validación"].idxmax(), "alpha"])
print("alpha elegido (máxima validación):", mejor_alpha)
""")
md("""
Con alphas pequeños casi no hay castigo; con alphas enormes el modelo se aplana (subajuste). Elegimos con la **validación** y recién entonces miramos la prueba:
""")
code("""
final = make_pipeline(StandardScaler(), Ridge(alpha=mejor_alpha)).fit(Xr_tr, yr_tr)
print("R² en PRUEBA con el alpha elegido:", round(final.score(Xr_te, yr_te), 3))
""")
md("""
### ✍️ Ejercicio C2
Repite la selección de alpha para **Lasso** con `alphas = (0.01, 0.1, 0.4, 1, 2, 5, 15)`. ¿Con qué alpha se maximiza la validación? ¿Cuántos coeficientes quedan en cero?
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for al in (0.01, 0.1, 0.4, 1, 2, 5, 15):
    m = make_pipeline(StandardScaler(), Lasso(alpha=al, max_iter=50000)).fit(Xr_tr, yr_tr)
    filas.append((al, round(m.score(Xr_val, yr_val), 3), int((m[-1].coef_ == 0).sum())))
print(pd.DataFrame(filas, columns=["alpha", "R² validación", "coef. en cero"]))
""", "La validación es casi plana entre alpha 0.01 y 2; desde alpha ≈ 5 empieza a caer. Con alphas medios el ruido queda en cero sin perder desempeño: es el uso típico de Lasso.")

md("""
### C3 · K-Fold (15 min)
Una sola partición de validación puede ser «fácil» o «difícil» por azar. **K-Fold** parte los datos en K pliegues; cada uno valida una vez y los demás entrenan; se promedian los resultados. Mira la **desviación** además del promedio. Primero a mano, para ver el mecanismo:
""")
code("""
X_dev = pd.concat([X_train, X_val])         # desarrollo = entrenamiento + validación (255 filas)
y_dev = pd.concat([y_train, y_val])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
puntajes_manual = []
for k, (idx_tr, idx_va) in enumerate(kf.split(X_dev), 1):
    m = make_pipeline(StandardScaler(), Ridge(alpha=10))
    m.fit(X_dev.iloc[idx_tr], y_dev.iloc[idx_tr])                 # fit solo con los 4 pliegues de entrenamiento
    puntajes_manual.append(m.score(X_dev.iloc[idx_va], y_dev.iloc[idx_va]))
    print(f"Pliegue {k}: entrena {len(idx_tr)} | valida {len(idx_va)} | R² {puntajes_manual[-1]:.3f}")
print(f"Promedio {np.mean(puntajes_manual):.3f} | desviación {np.std(puntajes_manual):.3f}")
""")
md("""
`cross_val_score` hace exactamente eso en una línea. Con un `Pipeline`, el escalador se re-aprende en cada pliegue (sin fuga):
""")
code("""
modelo_cv = make_pipeline(StandardScaler(), Ridge(alpha=10))
p_dev = cross_val_score(modelo_cv, X_dev, y_dev, cv=kf, scoring="r2")
print("Entrenamiento + validación (255 filas):", p_dev.round(3), "| media", round(p_dev.mean(), 3), "| desv.", round(p_dev.std(), 3))

p_300 = cross_val_score(modelo_cv, X, y, cv=kf, scoring="r2")     # lo que hace el notebook de la clase
print("Las 300 filas (incluye la prueba)     :", p_300.round(3), "| media", round(p_300.mean(), 3), "| desv.", round(p_300.std(), 3))
""")
md("""
El notebook de la clase hace la segunda versión (las 300 filas), que **incluye la prueba** aunque diga que sigue reservada (0.921 ± 0.010). Lo correcto es usar solo entrenamiento (+ validación): 0.924 ± 0.016. Y la prueba (45 filas) da 0.906 al final.

Y ¿cuánto oscila **una sola** partición? Hagamos 300 particiones simples:
""")
code("""
simples = []
for s in range(300):
    a, b, c, d = train_test_split(X_dev, y_dev, test_size=0.18, random_state=s)
    simples.append(make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(a, c).score(b, d))
print(f"Una sola partición: R² de {min(simples):.3f} a {max(simples):.3f} (desv. {np.std(simples):.3f})")
print(f"K-Fold: {p_dev.mean():.3f} ± {p_dev.std():.3f}")
print("Con ddof=1 (desviación muestral):", round(p_dev.std(ddof=1), 3), "<- np.std usa ddof=0 por defecto")
""")
md("""
Una partición puede dar desde 0.82 hasta 0.96: **no te quedes con un solo número**. Los pliegues no son independientes, así que la desviación es una guía, no un intervalo de confianza. Y recuerda: K-Fold sirve **durante el desarrollo**; la prueba queda para el final.

### ✍️ Ejercicio C3
Calcula el K-Fold con `n_splits=10` sobre `X_dev`. ¿Cambia la media? ¿Y la desviación? ¿Cuántas filas valida cada pliegue?
""")
code("""
# tu código aquí
""")
solution("""
kf10 = KFold(n_splits=10, shuffle=True, random_state=RS)
p10 = cross_val_score(modelo_cv, X_dev, y_dev, cv=kf10, scoring="r2")
print(p10.round(3), "| media", round(p10.mean(), 3), "| desv.", round(p10.std(), 3))
print("filas por pliegue de validación ≈", len(X_dev) // 10)
""", "La media se mantiene cerca de 0.92; la desviación sube porque cada pliegue valida menos filas (≈ 25). Cada pliegue entrena con el 90 % de los datos (más K = más ajustes y más tiempo).")

# ================================================================== PARTE D
md("""
## Parte D · Clases desbalanceadas y métricas

### D1 · Muestreo estratificado (15 min)
Si solo el 5 % son fraudes, una partición al azar puede dejar casi sin ellos a la prueba. `stratify=y` conserva la proporción. Primero, lo que hizo la profesora:
""")
code("""
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_fraude, y_fraude, test_size=0.25, stratify=y_fraude, random_state=42)
print(f"Dataset completo: {y_fraude.mean():.3%} ({int(y_fraude.sum())} fraudes)")
print(f"Entrenamiento   : {yf_tr.mean():.3%} ({int(yf_tr.sum())} fraudes de {len(yf_tr)})")
print(f"Prueba          : {yf_te.mean():.3%} ({int(yf_te.sum())} fraudes de {len(yf_te)})")
""")
md("""
Ahora **sin** estratificar, repetido 1.000 veces, y también con un dataset pequeño (200 filas con 12 fraudes):
""")
code("""
prop_n = [train_test_split(X_fraude, y_fraude, test_size=0.25, random_state=s)[3].mean() for s in range(1000)]
print(f"n = 2.000, sin estratificar: la prueba tiene de {min(prop_n):.1%} a {max(prop_n):.1%} de fraude (desv. {np.std(prop_n):.2%}); con stratify siempre 5.4 %")

idx = np.random.default_rng(0).choice(2000, 200, replace=False)
Xs_, ys_ = X_fraude[idx], y_fraude[idx]
cuenta = np.array([train_test_split(Xs_, ys_, test_size=0.25, random_state=s)[3].sum() for s in range(1000)])
print(f"n = 200 ({int(ys_.sum())} fraudes), prueba de 50 filas, SIN estratificar: de {cuenta.min()} a {cuenta.max()} fraudes; cero en {np.mean(cuenta == 0):.1%} de las particiones")
cuenta_s = np.array([train_test_split(Xs_, ys_, test_size=0.25, stratify=ys_, random_state=s)[3].sum() for s in range(1000)])
print("Con stratify siempre:", np.unique(cuenta_s))
""")
md("""
Estratificar **no crea fraudes**: solo asegura que cada parte los reciba en proporción. Y en el K-Fold de una clasificación desbalanceada se usa `StratifiedKFold`.

### ✍️ Ejercicio D1
800 casos con 4 % de fraude y `test_size=0.25`: ¿cuántos fraudes esperas en total, en la prueba y en el entrenamiento si estratificas? Compruébalo con `make_classification(n_samples=800, weights=[0.96, 0.04])` (las proporciones pueden diferir un poco por el ruido de etiquetas del generador).
""")
code("""
# tu código aquí
""")
solution("""
Xd, yd = make_classification(n_samples=800, n_features=5, weights=[0.96, 0.04], random_state=RS)
a, b, c, d = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=RS)
print("total:", int(yd.sum()), "| entrenamiento:", int(c.sum()), "de", len(c), "| prueba:", int(d.sum()), "de", len(d))
""", "Esperado: 32 fraudes en total, 8 en la prueba (200 filas) y 24 en el entrenamiento (600). En la corrida de referencia salen 38 (4.75 %), 9 y 29: el generador agrega ruido a las etiquetas, pero la proporción se conserva en cada parte (≈ 4.5 %).")

md("""
### D2 · La trampa del accuracy y la matriz de confusión (20 min)
Entrenamos una regresión logística y miramos la **matriz de confusión**; todas las métricas salen de sus cuatro números.
""")
code("""
modelo_fraude = LogisticRegression(max_iter=1000).fit(Xf_tr, yf_tr)
pred = modelo_fraude.predict(Xf_te)
tn, fp, fn, tp = confusion_matrix(yf_te, pred).ravel()
print(f"TN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}")

# A mano
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"A mano       : accuracy {accuracy:.3f} | precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
print(f"scikit-learn : accuracy {accuracy_score(yf_te, pred):.3f} | precision {precision_score(yf_te, pred):.3f} | "
      f"recall {recall_score(yf_te, pred):.3f} | F1 {f1_score(yf_te, pred):.3f}")
""")
md("""
- **Accuracy 0.952**: parece excelente.
- **Precision 0.588** = 10 / (10 + 7): cuando alerta, acierta el 59 %. **Recall 0.370** = 10 / (10 + 17): detecta solo 10 de los 27 fraudes. **F1 0.455** (en clase se dijo «50 %»; la salida real es 0.455).
- Precision y recall tienen el **mismo numerador (TP)** pero **otro total**: como una probabilidad condicional, cambia «dado qué».

Ahora la trampa: un modelo **tonto** que siempre dice «no fraude».
""")
code("""
tonto = DummyClassifier(strategy="most_frequent").fit(Xf_tr, yf_tr)
p_t = tonto.predict(Xf_te)
print(f"Siempre 'no fraude': accuracy {accuracy_score(yf_te, p_t):.3f} | recall {recall_score(yf_te, p_t):.3f} | precision {precision_score(yf_te, p_t, zero_division=0):.3f}")
print(f"Regresión logística: accuracy {accuracy_score(yf_te, pred):.3f} | recall {recall_score(yf_te, pred):.3f}")
""")
md("""
El modelo tonto logra 0.946 de accuracy con recall 0: **el accuracy casi no distingue** al modelo que no sirve del que detecta algo. Por eso se miran varias métricas, según el costo del error: si dejar escapar un fraude es caro, importa el **recall**; si las falsas alarmas son caras, la **precision**; si importan ambos, el **F1**.

### ✍️ Ejercicio D2
Con TP = 30, FP = 10, FN = 20, TN = 940 (1.000 transacciones) calcula a mano accuracy, precision, recall y F1 y el accuracy del modelo que siempre dice «no fraude».
""")
code("""
# tu código aquí
""")
solution("""
TP, FP, FN, TN = 30, 10, 20, 940
acc = (TP + TN) / 1000
p = TP / (TP + FP)
r = TP / (TP + FN)
print(acc, p, r, round(2 * p * r / (p + r), 3), "| siempre 'no':", (TN + FP) / 1000)
""", "Accuracy 0.970, precision 0.750, recall 0.600, F1 0.667. Siempre «no fraude»: 950/1.000 = 0.950. El 0.970 parece genial, pero deja escapar 20 de 50 fraudes.")

md("""
### D3 · El umbral y el equilibrio precision / recall (15 min)
`predict` usa el umbral 0.5 sobre la probabilidad. Con `predict_proba` puedes moverlo. **El umbral es un hiperparámetro: se elige en validación, no en la prueba.** Sacamos una validación del entrenamiento:
""")
code("""
Xa, Xv, ya, yv = train_test_split(Xf_tr, yf_tr, test_size=0.25, stratify=yf_tr, random_state=42)
m = LogisticRegression(max_iter=1000).fit(Xa, ya)
prob_v = m.predict_proba(Xv)[:, 1]

print("umbral | precision | recall |   F1   (validación)")
for u in (0.5, 0.4, 0.3, 0.2, 0.1):
    p_ = prob_v >= u
    print(f"{u:6.1f} | {precision_score(yv, p_, zero_division=0):9.3f} | {recall_score(yv, p_):6.3f} | {f1_score(yv, p_, zero_division=0):6.3f}")

umbrales = np.linspace(0.02, 0.8, 40)
mejor_u = umbrales[int(np.argmax([f1_score(yv, prob_v >= u, zero_division=0) for u in umbrales]))]
print("\\nMejor F1 en validación con umbral:", round(float(mejor_u), 2), f"(la validación tiene solo {int(yv.sum())} fraudes)")
prueba_u = m.predict_proba(Xf_te)[:, 1] >= mejor_u
print(f"En la PRUEBA con ese umbral: precision {precision_score(yf_te, prueba_u):.3f} | recall {recall_score(yf_te, prueba_u):.3f} | F1 {f1_score(yf_te, prueba_u):.3f}")
""")
md("""
Bajar el umbral **sube el recall y baja la precision**: se detectan más fraudes a costa de más falsas alarmas. El mejor F1 en validación (≈ 0.72 con umbral 0.44) **no se repite** en la prueba (≈ 0.44): con solo 20 fraudes en la validación y 27 en la prueba, estas métricas son muy ruidosas. Otra palanca es `class_weight="balanced"`:
""")
code("""
bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xf_tr, yf_tr)
pb = bal.predict(Xf_te)
print("TN, FP, FN, TP:", confusion_matrix(yf_te, pb).ravel())
print(f"accuracy {accuracy_score(yf_te, pb):.3f} | precision {precision_score(yf_te, pb):.3f} | recall {recall_score(yf_te, pb):.3f} | F1 {f1_score(yf_te, pb):.3f}")
""")
md("""
El modelo balanceado detecta 22 de 27 fraudes (recall 0.815) pero baja el accuracy a 0.896 y la precision a 0.319: ¡empeora el accuracy y mejora el negocio, si el fraude es caro! En el caso del banco (2 % de impago, falso negativo costoso) el criterio es **recall**, sin olvidar la precision.

### ✍️ Ejercicio D3
Con el modelo `m` y la validación, ¿qué umbral necesitas para lograr un **recall ≥ 0.80** en validación? ¿Qué precision tiene ese umbral?
""")
code("""
# tu código aquí
""")
solution("""
for u in np.linspace(0.02, 0.8, 40)[::-1]:
    p_ = prob_v >= u
    if recall_score(yv, p_) >= 0.80:
        print("umbral", round(float(u), 2), "| recall", round(recall_score(yv, p_), 3), "| precision", round(precision_score(yv, p_, zero_division=0), 3))
        break
""", "Se recorre el umbral desde el más exigente hacia abajo y se queda con el primero que cumple el recall. Con umbral ≈ 0.24 el recall llega a 0.80 y la precision baja a ≈ 0.52 (más falsas alarmas): es la decisión de negocio.")

md("""
### D4 · Métricas de regresión (10 min)
Para el precio usamos MAE, RMSE y R². Recién aquí se usa la prueba, con el modelo ya elegido:
""")
code("""
modelo_precio = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X_train, y_train)
pred_p = modelo_precio.predict(X_test)
mae = mean_absolute_error(y_test, pred_p)
rmse_ = rmse(y_test, pred_p)
r2 = r2_score(y_test, pred_p)
print(f"MAE  : {mae:,.0f} CLP | RMSE : {rmse_:,.0f} CLP | R² : {r2:.3f}")
print(f"Error típico relativo (MAE / precio medio): {mae / y_test.mean():.1%}")
""")
md("""
- **MAE** 13.9 M de CLP: error medio en unidades de precio. **RMSE** 16.2 M: castiga más los errores grandes (es mayor que el MAE). **R²** 0.904: variabilidad explicada.
- Un R² alto **no** significa un modelo perfecto: ¿es aceptable un error de ≈ 14 M para una vivienda de ≈ 190 M? Eso lo decide el negocio.

### ✍️ Ejercicio D4
Calcula MAE, RMSE y R² del modelo **sobre el entrenamiento** y compáralos con los de la prueba. ¿Qué te dice la diferencia?
""")
code("""
# tu código aquí
""")
solution("""
p_tr = modelo_precio.predict(X_train)
print(round(mean_absolute_error(y_train, p_tr)), round(rmse(y_train, p_tr)), round(r2_score(y_train, p_tr), 3))
""", "En entrenamiento el R² es ≈ 0.93 frente a 0.904 en prueba: una brecha pequeña (modelo lineal de pocos parámetros). Una brecha grande habría sido señal de sobreajuste.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene corregir

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Notebook, Bloque 5 | K-Fold con `X, y` y «la prueba sigue reservada» | `cross_val_score(modelo, X, y)` usa las **300 filas**, incluida la prueba: 0.921 ± 0.010. Con entrenamiento + validación: 0.924 ± 0.016 (C3). |
| Notebook, Bloques 3 y 7 | `X_val` se crea; el clasificador se evalúa directo en prueba | La validación **nunca se usa**; la elección de alpha y umbral debe hacerse en validación (C2, D3). |
| Clase y Bloque 4 | «Sin estandarizar es difícil que Lasso lleve a cero» | Con alpha = 2 Lasso anula el ruido en ambos casos. El daño es de **unidades**: Ridge (alpha 30) R² 0.916 → 0.532 al cambiar a centenas de m² (C1). Además las dos tablas de coeficientes no son comparables. |
| Diapositiva «Prueba (61 %)» | brecha 98 % → 61 % | Es una curva ilustrativa y es de **validación**; la profesora lo aclara en clase. Una curva real en B2: 0.999 vs 0.801. |
| Diapositiva «Lección del COVID» | «el conjunto de prueba simula ese futuro» | Una prueba al azar del mismo periodo **no** detecta un cambio de contexto; se necesita una división **temporal**. El apunte lo matiza. |
| Clase | «Validación mejor que entrenamiento = algo falla» | Con 45 filas ocurre en el 40 % de las particiones (120 de 300). Es una señal para revisar, no un error. |
| Bloque 4, texto | `n_habitaciones` entre 1 y 6 | El código genera 1 a 5. |
| Clase, Bloque 7 | F1 «50 %» | La salida real es **0.455**. |
| K-Fold | `.std()` | NumPy usa **ddof = 0** (0.016; con ddof = 1, 0.018); los pliegues no son independientes. |
| Imputación (slides) | media / mediana / moda | Se calcula **solo con entrenamiento** (fuga si no). La media además reduce la desviación (A2). |
| Caso, pregunta 1.2 | «Exactitud» | Con las definiciones del apunte, que web y contrato no coincidan es primero **coherencia**; la exactitud entra al decidir la verdad (A3). |
| Caso, pregunta 3.3 | Recall | Correcto, pero un banco real fija un umbral en validación, mira la precision y el costo de rechazar a un buen pagador (D3). |

**Transcripción con ruido**: «Rich/reach/range/pitch» = Ridge, «laso/lazo/lasco» = Lasso, «overpicking» = overfitting, «exagitu/actitudes» = exactitud, «exageridad» = exhaustividad (recall), «casco holding/fast folding» = K-Fold, «dale» = DALL-E. Tramo hacia 0:28–0:55 casi irrecuperable, recreo hacia 1:12–1:24.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (30 min, SIN mirar lo anterior)

Eres el analista de un banco que modela el **impago** de créditos de consumo. Los datos son **simulados** (no hay archivo): 20.000 clientes, 2 % de impago, con faltantes y una columna categórica.
""")
code("""
rng = np.random.default_rng(7)
Xc, yc = make_classification(n_samples=20000, n_features=8, n_informative=4, weights=[0.98, 0.02], flip_y=0.0, random_state=7)
clientes = pd.DataFrame(Xc, columns=[f"v{i}" for i in range(8)])
clientes["comuna"] = rng.choice(["A", "B", "C", "D", "E"], len(clientes))
clientes.loc[rng.choice(len(clientes), 1600, replace=False), "v0"] = np.nan          # 8 % de faltantes
clientes.loc[rng.choice(len(clientes), 1000, replace=False), "comuna"] = np.nan      # 5 % de faltantes
default = pd.Series(yc, name="default")
print(clientes.shape, "| impago:", f"{default.mean():.2%}", "| faltantes:", clientes.isna().sum()[lambda s: s > 0].to_dict())
""")
md("""
1. Separa **primero** una prueba del 20 % **estratificada**.
2. Arma un `Pipeline` con `ColumnTransformer` (numéricas: mediana + estandarizar; `comuna`: moda + one-hot) y una `LogisticRegression(penalty="l1", solver="liblinear", C=0.1)` (el equivalente de Lasso).
3. Valida con `StratifiedKFold` (5) usando `scoring="recall"` y reporta media y desviación **solo con el 80 %**.
4. Con una validación sacada de ese 80 %, elige el umbral que logre recall ≥ 0.6 con la mayor precision posible.
5. Mira la prueba **una sola vez** y reporta la matriz de confusión, precision, recall y F1.
6. Responde sin código: ¿por qué el accuracy no sirve aquí? ¿Qué corregirías del material antes de dárselo a un cliente?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
X_dev_c, X_te_c, y_dev_c, y_te_c = train_test_split(clientes, default, test_size=0.20, stratify=default, random_state=RS)

num_c = [f"v{i}" for i in range(8)]
prep_c = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num_c),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["comuna"]),
])
pipe_c = Pipeline([("prep", prep_c), ("modelo", LogisticRegression(penalty="l1", solver="liblinear", C=0.1))])

cv = StratifiedKFold(5, shuffle=True, random_state=RS)
rec_cv = cross_val_score(pipe_c, X_dev_c, y_dev_c, cv=cv, scoring="recall")
print("Recall K-Fold:", rec_cv.round(3), "| media", round(rec_cv.mean(), 3), "| desv.", round(rec_cv.std(), 3))

# Umbral con una validación sacada del 80 %
X_a, X_v, y_a, y_v = train_test_split(X_dev_c, y_dev_c, test_size=0.25, stratify=y_dev_c, random_state=RS)
pipe_c.fit(X_a, y_a)
prob = pipe_c.predict_proba(X_v)[:, 1]
mejor = None
for u in np.linspace(0.01, 0.5, 100):
    p_ = prob >= u
    if recall_score(y_v, p_) >= 0.6:
        pr = precision_score(y_v, p_, zero_division=0)
        if mejor is None or pr > mejor[1]:
            mejor = (u, pr)
print("Umbral elegido en validación:", None if mejor is None else round(float(mejor[0]), 3))

# Prueba UNA vez, con el modelo reentrenado en todo el 80 %
pipe_c.fit(X_dev_c, y_dev_c)
u = 0.5 if mejor is None else mejor[0]
pred_c = pipe_c.predict_proba(X_te_c)[:, 1] >= u
print("TN, FP, FN, TP:", confusion_matrix(y_te_c, pred_c).ravel())
print(f"precision {precision_score(y_te_c, pred_c):.3f} | recall {recall_score(y_te_c, pred_c):.3f} | F1 {f1_score(y_te_c, pred_c):.3f}")
""", "Resultado de referencia: con umbral 0.5 el recall de CV es bajo (≈ 0.34 ± 0.03); con el umbral elegido en validación (≈ 0.13) la prueba da precision ≈ 0.37, recall ≈ 0.64 y F1 ≈ 0.47. El accuracy no sirve: ya es ≈ 98 % con un modelo que siempre dice «no impago». Correcciones al material: (1) hacer el K-Fold solo con entrenamiento (+ validación), (2) usar la validación para alpha y umbral, (3) corregir «Lasso no llega a cero sin estandarizar» (es un problema de unidades), (4) la pregunta 1.2 del caso es coherencia antes que exactitud, (5) imputar y escalar dentro del Pipeline.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué hace cada verbo `fit`, `transform` y `predict`, y con qué datos se usa cada uno?
2. ¿Para qué sirve cada uno de los tres conjuntos y por qué la prueba se mira una sola vez?
3. ¿Qué es la fuga de datos? Da dos ejemplos y cómo los evita un `Pipeline`.
4. ¿Por qué se estandariza antes de Ridge o Lasso? ¿Qué cambia si mides la superficie en centenas de m²?
5. ¿Qué agrega K-Fold frente a una sola partición y qué conjunto no usa?
6. ¿Por qué el accuracy engaña con 95 % de no fraude y qué miras en su lugar?

<details><summary><b>Respuestas</b></summary>

1. `fit` aprende (media, coeficientes) y se usa **solo con entrenamiento**; `transform` aplica lo aprendido (cualquier conjunto); `predict` predice con un modelo ya ajustado.
2. Entrenamiento: el modelo aprende parámetros. Validación: eliges alpha, umbral o modelo. Prueba: veredicto final; si decides mirándola deja de ser independiente.
3. Usar información de validación o prueba al desarrollar (por ejemplo, escalar o seleccionar variables con todas las filas, o elegir hiperparámetros con la prueba). El `Pipeline` hace el `fit` de cada paso solo con el entrenamiento (también dentro de cada pliegue).
4. Porque el castigo depende del tamaño de los coeficientes, y este depende de las unidades. En centenas de m² el coeficiente es 100 veces mayor y se castiga más (R² 0.916 → 0.532 con Ridge, alpha 30). Estandarizado, el resultado es el mismo (0.887).
5. Un promedio y una dispersión en vez de un solo número; usa entrenamiento (+ validación) y deja fuera la prueba.
6. Un modelo que siempre dice «no fraude» acierta 95 % con recall 0. Se miran matriz de confusión, precision, recall y F1, según el costo del error.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 31 · Introducción al Machine Learning. Revisa `Guias/apuntes/C30_Ejercicios_Cuaderno_PreparacionML.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
