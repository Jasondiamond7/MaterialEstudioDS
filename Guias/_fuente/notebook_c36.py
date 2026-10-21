"""Genera el notebook guiado de Boosting (Clase 36).

Uso normal:      python notebook_c36.py             -> Guias/notebooks/C36_Boosting.ipynb
Modo de prueba:  C36_TEST=1 python notebook_c36.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C36_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C36_Boosting_test.ipynb" if TEST else "C36_Boosting.ipynb")
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
# Clase 36 · Boosting: AdaBoost, Gradient Boosting, XGBoost y LightGBM
**Diplomado Data Science** · 2 sesiones · Prof. Bastian Galasso

Material de apoyo:
- `Guias/C36_Guia_Boosting.pdf` — guía visual de la clase.
- `Guias/apuntes/C36_Apuntes_Boosting.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C36_Ejercicios_Cuaderno_Boosting.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor: `Clase 36 - Boosting/Clase Boosting.pptx` y `Ejemplo Boosting.ipynb`.
- Transcripción: `VideosClases/Clase 36 Boosting Profesor Bastian Galasso.txt`.

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `xgboost`, `lightgbm`, `openpyxl` (para leer `Heart.xlsx`) y `seaborn` (opcional). Instala una vez en tu entorno:
```
pip install scikit-learn xgboost lightgbm openpyxl
```
Los datos son los mismos de la Clase 35 (`Clase 35-BastianGalasso/Heart.xlsx` y `ames_housing.csv`); el notebook los busca solo.

### Objetivos
1. Entender por qué boosting entrena **en secuencia** y qué lo distingue de bagging y Random Forest.
2. Programar **AdaBoost** y **Gradient Boosting** desde cero, y comprobar que reproducen a scikit-learn.
3. Razonar sobre la **pérdida**, la **tasa de aprendizaje** η y el **número de árboles**, y por qué se ajustan juntos.
4. Usar **XGBoost** y **LightGBM** con sus hiperparámetros clave y con **early stopping**.
5. Comparar los modelos bajo el mismo pipeline, con validación honesta, y detectar lo que el material original afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y carga de datos |
| A | 10 | A1 · Residuos a mano (el ejemplo de la clase) |
| A | 10 | A2 · Boosting paso a paso sobre Ames |
| A | 10 | A3 · Bagging vs boosting: curvas de error |
| B | 15 | B1 · AdaBoost desde cero |
| B | 10 | B2 · AdaBoost en scikit-learn: estimador base y η |
| B | 10 | B3 · ¿Es AdaBoost sensible al ruido? Un experimento |
| C | 15 | C1 · Gradient Boosting desde cero |
| C | 10 | C2 · La pérdida importa: cuadrática, absoluta, Huber |
| C | 15 | C3 · Tasa de aprendizaje y número de árboles |
| C | 5 | C4 · El gradiente de la pérdida logística |
| D | 10 | D1 · XGBoost con early stopping |
| D | 10 | D2 · LightGBM: hojas, datos pequeños y sobreajuste |
| D | 10 | D3 · Dos trampas: `subsample_freq` y la importancia |
| D | 15 | D4 · Comparación de los cinco modelos (con Random Forest) |
| D | 15 | D5 · Optimización con búsqueda aleatoria (ejercicios del profesor) |
| — | 10 | Alertas del material |
| — | 20 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook con `scikit-learn 1.9`, `xgboost 3.4` y `lightgbm 4.7`. Con otras versiones pueden variar un poco (sobre todo los tiempos).
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import sklearn, xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, KFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 42   # semilla

# Busca la carpeta con Heart.xlsx y ames_housing.csv subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase*35*")))
carpeta = next(raiz.glob("Clase*35*"))

print("scikit-learn:", sklearn.__version__, "| xgboost:", xgboost.__version__, "| lightgbm:", lgb.__version__)
print("Datos en:", carpeta.name)
""")
md("""
### Los datos
- **Heart** (303 pacientes, 13 variables): clasificación. `AHD` = `Yes` si hay enfermedad cardíaca (46% de los casos).
- **Ames Housing** (2.930 casas, 81 variables): regresión del precio de venta `Sale_Price`.

Seguimos el notebook del profesor: separación 75/25 estratificada en Heart y 80/20 en Ames, y el mismo preprocesamiento **dentro de un pipeline** (mediana para numéricas; moda y one-hot para categóricas). En la clase se dijo que se eliminan `Latitude` y `Longitude` de Ames («no entregan información tal cual»); el notebook que entregó el profesor no lo hace, así que lo hacemos aquí.
""")
code("""
def make_preprocessor(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


# --- Heart (clasificación)
heart = pd.read_excel(carpeta / "Heart.xlsx")
X_h, y_h = heart.drop(columns="AHD"), heart["AHD"].map({"No": 0, "Yes": 1})
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_h, y_h, test_size=0.25, stratify=y_h, random_state=RS)

# --- Ames (regresión). Quitamos Latitude/Longitude como se dijo en clase.
ames = pd.read_csv(carpeta / "ames_housing.csv").drop(columns=["Unnamed: 0", "Latitude", "Longitude"])
X_a, y_a = ames.drop(columns="Sale_Price"), ames["Sale_Price"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_a, y_a, test_size=0.20, random_state=RS)

# Para las curvas de error: una validación DENTRO del entrenamiento de Ames (el test queda intacto)
Xa_fit, Xa_val, ya_fit, ya_val = train_test_split(Xa_tr, ya_tr, test_size=0.25, random_state=RS)
pre_a = make_preprocessor(Xa_fit).fit(Xa_fit)
A_fit, A_val = pre_a.transform(Xa_fit), pre_a.transform(Xa_val)

print("Heart  :", heart.shape, "| train", Xh_tr.shape, "| test", Xh_te.shape, "| % AHD:", round(y_h.mean(), 3))
print("Ames   :", ames.shape, "| train", Xa_tr.shape, "| test", Xa_te.shape, "| precio mediano:", f"${y_a.median():,.0f}")
print("Ames, matriz tras preprocesar (fit/val):", A_fit.shape, A_val.shape)
""")

# ================================================================== PARTE A
md("""
## Parte A · La intuición: aprender de los errores

Los modelos de la Clase 35 (bagging, Random Forest) entrenan árboles **independientes**, cada uno con su muestra, y promedian. Boosting hace otra cosa: el árbol 2 recibe información de **dónde se equivocó el árbol 1** para no repetir los errores. Por eso se entrena en secuencia, sobre el dataset completo, y el modelo final es una **suma ponderada**:

$$F(x) = F_0(x) + \\eta\\,h_1(x) + \\eta\\,h_2(x) + \\dots + \\eta\\,h_M(x)$$

### A1 · Residuos a mano: el ejemplo de la clase (10 min)
La diapositiva 10 usa cuatro observaciones A, B, C, D con y = (12, 7, 11, 10) y una predicción base F₀ = 10. Los residuos son r₁ = y − F₀ = (+2, −3, +1, 0). Un árbol pequeño aprende a predecir esos residuos, y la nueva predicción es F₁ = F₀ + η·h₁.

Aquí el «árbol» es un *stump* que separa {A, C} de {B, D}: cada hoja predice la media de los residuos de sus observaciones.
""")
code("""
y_toy = np.array([12., 7., 11., 10.])
F0 = 10.0
r1 = y_toy - F0
grupo = np.array([[0], [1], [0], [1]])        # A, C -> hoja 0 ; B, D -> hoja 1
h1 = DecisionTreeRegressor(max_depth=1).fit(grupo, r1)

filas = []
for eta in (1.0, 0.5):
    F1 = F0 + eta * h1.predict(grupo)
    filas.append(pd.DataFrame({"obs": list("ABCD"), "y": y_toy, "F0": F0, "r1 = y - F0": r1,
                               "h1 (media por hoja)": h1.predict(grupo), "eta": eta,
                               "F1": F1, "y - F1": y_toy - F1}))
    print(f"eta = {eta}:  RMSE antes = {rmse(y_toy, np.full(4, F0)):.3f}   RMSE después = {rmse(y_toy, F1):.3f}")
pd.concat(filas).round(3).reset_index(drop=True)
""")
md("""
Fíjate en dos cosas: (1) con η = 1 baja más rápido, pero la observación D, que estaba perfecta (residuo 0), **empeora** porque comparte hoja con B; (2) con η = 0.5 se avanza menos por paso y queda más trabajo para el siguiente árbol, que estudiará los nuevos residuos «lo que aún falta explicar».

> En clase el profesor usó un árbol imaginario que predecía (+1, −4, +1, 0) para estos residuos, con lo que F₁ = (11, 6, 11, 10) y los nuevos residuos son (1, 1, 0, 0). Es una ilustración: un árbol que ajusta medias por hoja no predice −4 cuando el residuo es −3 (se pasaría), pero el mecanismo es el mismo.

### ✍️ Ejercicio A1
Comprueba con código el Bloque 5 del cuaderno en papel: y = (10, 4, 9, 17), η = 0.5. Árbol 1 separa {A, B} de {C, D}; árbol 2 separa {B} del resto. Calcula F₀, F₁, F₂ y el RMSE de cada etapa. ¿Cuál observación sigue peor explicada?
""")
code("""
# tu código aquí
""")
solution("""
y2 = np.array([10., 4., 9., 17.])
eta = 0.5
F = np.full(4, y2.mean())
print("F0:", F, "  RMSE:", round(rmse(y2, F), 3))

grupos = [np.array([[0], [0], [1], [1]]),      # árbol 1: {A,B} | {C,D}
          np.array([[0], [1], [0], [0]])]      # árbol 2: {B} | {A,C,D}
for k, g in enumerate(grupos, start=1):
    r = y2 - F
    arbol = DecisionTreeRegressor(max_depth=1).fit(g, r)
    F = F + eta * arbol.predict(g)
    print(f"Árbol {k}: h = {arbol.predict(g)}  ->  F{k} = {F}  residuos = {y2 - F}  RMSE = {rmse(y2, F):.3f}")
""", "Debes obtener RMSE 4.637 → 3.841 → 3.112, y la observación D (residuo 4.75 tras el árbol 2) es la peor explicada.")

md("""
### A2 · Boosting paso a paso sobre Ames (10 min)
Ahora lo mismo con datos reales y varios pasos: predecir el precio (en miles de USD) solo con la superficie habitable `Gr_Liv_Area`. Partimos de F₀ = promedio y, en cada etapa, ajustamos un árbol de profundidad 2 a los residuos y sumamos η = 0.3 de su predicción.
""")
code("""
d = ames.loc[ames.Gr_Liv_Area < 4000, ["Gr_Liv_Area", "Sale_Price"]]
x1 = d[["Gr_Liv_Area"]].values
y1 = d["Sale_Price"].values / 1000

eta = 0.3
F = np.full(len(y1), y1.mean())
curva = [rmse(y1, F)]
for m in range(50):
    r = y1 - F                                          # lo que el modelo aún no explica
    arbol = DecisionTreeRegressor(max_depth=2).fit(x1, r)
    F = F + eta * arbol.predict(x1)
    curva.append(rmse(y1, F))

print({m: round(curva[m], 1) for m in (0, 1, 5, 20, 50)})

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(curva, color="#EA580C", lw=2)
axes[0].set(xlabel="Árboles sumados", ylabel="RMSE en entrenamiento (miles de USD)", title="Cada árbol reduce el error de entrenamiento")
axes[1].scatter(x1, y1, s=4, alpha=0.3, color="gray")
orden = np.argsort(x1[:, 0])
axes[1].plot(x1[orden], F[orden], color="#EA580C", lw=2)
axes[1].set(xlabel="Superficie habitable (pies²)", ylabel="Precio (miles de USD)", title="El modelo tras 50 árboles")
plt.tight_layout(); plt.show()
""")
md("""
El error de **entrenamiento** baja con cada árbol (de ≈ 78 a ≈ 51). Eso siempre ocurre; lo que importa es el error en datos que el modelo no vio, y por eso la Parte C introduce la validación.

### ✍️ Ejercicio A2
Repite con η = 1.0 y con η = 0.05 (50 árboles en ambos casos). ¿Cuál tiene el menor RMSE de entrenamiento a los 50 árboles? ¿Es por eso el mejor modelo? Calcula también el RMSE en la validación `Xa_val` solo con la variable `Gr_Liv_Area` para decidir.
""")
code("""
# tu código aquí
""")
solution("""
x_fit, y_fit = Xa_fit[["Gr_Liv_Area"]].values, ya_fit.values / 1000
x_val, y_val = Xa_val[["Gr_Liv_Area"]].values, ya_val.values / 1000

def boosting_1d(eta, M=50):
    F_tr = np.full(len(y_fit), y_fit.mean())
    F_va = np.full(len(y_val), y_fit.mean())
    for _ in range(M):
        t = DecisionTreeRegressor(max_depth=2).fit(x_fit, y_fit - F_tr)
        F_tr += eta * t.predict(x_fit)
        F_va += eta * t.predict(x_val)
    return rmse(y_fit, F_tr), rmse(y_val, F_va)

for eta in (1.0, 0.3, 0.05):
    tr, va = boosting_1d(eta)
    print(f"eta = {eta:<5} RMSE train = {tr:6.1f}   RMSE validación = {va:6.1f}")
""", "Con η = 1 el entrenamiento baja más (RMSE 44.3) pero su validación es la peor (59.6): memoriza antes. Con η = 0.3 queda en 49.1 / 53.2 y con η = 0.05 en 54.0 / 52.2: en este problema de una sola variable, el modelo de pasos más chicos (η = 0.05) tiene la mejor validación, aunque su entrenamiento sea el peor. Menor error de entrenamiento no significa mejor modelo.")

md("""
### A3 · Bagging vs boosting: curvas de error (10 min)
Comparemos el efecto de **agregar árboles** en un Random Forest y en Gradient Boosting, ambos sobre todas las variables de Ames y evaluados en la validación (no en el test).

- Random Forest: el promedio de los primeros *k* árboles (calculamos las predicciones acumuladas).
- Gradient Boosting: `staged_predict` entrega F₁, F₂, F₃… sin reentrenar.
""")
code("""
rf = RandomForestRegressor(n_estimators=300, max_features=1/3, n_jobs=-1, random_state=RS).fit(A_fit, ya_fit)
acumulado = np.cumsum([e.predict(A_val) for e in rf.estimators_], axis=0) / np.arange(1, 301)[:, None]
curva_rf = [rmse(ya_val, p) for p in acumulado]

gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RS).fit(A_fit, ya_fit)
curva_gb = [rmse(ya_val, p) for p in gb.staged_predict(A_val)]

arbol_solo = DecisionTreeRegressor(max_depth=3, random_state=RS).fit(A_fit, ya_fit)
print(f"Un árbol de profundidad 3        : {rmse(ya_val, arbol_solo.predict(A_val)):,.0f}")
print(f"Bosque, 1 / 100 / 300 árboles    : {curva_rf[0]:,.0f} / {curva_rf[99]:,.0f} / {curva_rf[-1]:,.0f}")
print(f"Boosting, 1 / 100 / 300 árboles  : {curva_gb[0]:,.0f} / {curva_gb[99]:,.0f} / {curva_gb[-1]:,.0f}")
print(f"Mejor ronda de boosting: {int(np.argmin(curva_gb)) + 1}  (RMSE {min(curva_gb):,.0f})")

plt.figure(figsize=(8.5, 3.8))
plt.plot(range(1, 301), curva_rf, label="Random Forest", color="#2563EB", lw=2)
plt.plot(range(1, 301), curva_gb, label="Gradient Boosting (η = 0.1)", color="#EA580C", lw=2)
plt.ylim(15000, 50000); plt.xlabel("Número de árboles"); plt.ylabel("RMSE en validación (USD)")
plt.legend(); plt.title("Más árboles: el bosque se estanca, boosting sigue mejorando"); plt.show()
""")
md("""
Lectura: el bosque baja rápido y se aplana (agregar árboles solo reduce la varianza del promedio; su sesgo no cambia). Boosting empieza lento a propósito (η = 0.1: cada árbol aporta poco) pero sigue corrigiendo sesgo y supera al bosque. **Cuidado**: esto no quiere decir que boosting siempre gane (en la Parte D el bosque queda muy cerca).

### ✍️ Ejercicio A3
(a) ¿A partir de cuántos árboles el RMSE de validación del bosque queda a menos de un 2% de su valor final? ¿Y el de boosting? (b) ¿Por qué se puede paralelizar el entrenamiento del bosque con `n_jobs=-1` pero no el de los árboles de boosting?
""")
code("""
# tu código aquí
""")
solution("""
def arboles_para_estancar(curva, tolerancia=0.02):
    final = curva[-1]
    return next(i + 1 for i, v in enumerate(curva) if v <= final * (1 + tolerancia))

print("Bosque   :", arboles_para_estancar(curva_rf), "árboles")
print("Boosting :", arboles_para_estancar(curva_gb), "árboles")
""", "(a) En la corrida de referencia: el bosque ≈ 44 árboles y boosting ≈ 138 (boosting sigue mejorando mucho más tiempo). (b) Los árboles del bosque son independientes (cada uno usa su propia muestra), así que se reparten entre procesadores. En boosting el árbol *m* necesita el error acumulado de los m−1 anteriores. XGBoost y LightGBM sí paralelizan **dentro** de cada árbol (la búsqueda de cortes), no entre árboles.")

# ================================================================== PARTE B
md("""
## Parte B · AdaBoost: enfocar los casos difíciles

AdaBoost (*Adaptive Boosting*, Freund y Schapire, 1997) trabaja con clases −1/+1 y pesos por observación:

1. Pesos iguales: wᵢ = 1/n.
2. Entrena un árbol (normalmente un *stump*, de profundidad 1) con esos pesos.
3. Error ponderado: ε = Σ wᵢ·1[hₘ(xᵢ) ≠ yᵢ] / Σ wᵢ.
4. Peso del modelo: αₘ = ½ ln((1 − ε)/ε).
5. Actualiza: wᵢ ← wᵢ·exp(−αₘ·yᵢ·hₘ(xᵢ)) y normaliza (los errores suben, los aciertos bajan).
6. Voto final: H(x) = signo(Σ αₘ hₘ(x)).

### B1 · AdaBoost desde cero (15 min)
Lo programamos con las 5 variables numéricas de Heart (imputadas con la mediana) y lo comparamos con `AdaBoostClassifier` de scikit-learn.
""")
code("""
num_h = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]
imp = SimpleImputer(strategy="median").fit(Xh_tr[num_h])
Xn_tr, Xn_te = imp.transform(Xh_tr[num_h]), imp.transform(Xh_te[num_h])
ys_tr, ys_te = np.where(yh_tr == 1, 1, -1), np.where(yh_te == 1, 1, -1)       # clases -1 / +1


def adaboost_scratch(X, y, M):
    n = len(y)
    w = np.full(n, 1 / n)
    modelos = []
    for m in range(M):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        eps = w[pred != y].sum() / w.sum()                 # error ponderado
        alpha = 0.5 * np.log((1 - eps) / eps)              # peso del modelo
        w = w * np.exp(-alpha * y * pred)                  # sube el peso de los errores
        w = w / w.sum()
        modelos.append((stump, eps, alpha))
    return modelos


def adaboost_decision(modelos, X):
    return sum(alpha * stump.predict(X) for stump, _, alpha in modelos)


ada_mano = adaboost_scratch(Xn_tr, ys_tr, 50)
print("Primeras rondas (ε, α):", [(round(float(e), 3), round(float(a), 3)) for _, e, a in ada_mano[:5]])

pred_mano = np.sign(adaboost_decision(ada_mano, Xn_te))
ada_sk = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50,
                            learning_rate=1.0, random_state=0).fit(Xn_tr, ys_tr)
print("Exactitud test  | a mano:", round((pred_mano == ys_te).mean(), 3), "| sklearn:", round(ada_sk.score(Xn_te, ys_te), 3))
print("Coinciden las predicciones:", (pred_mano == ada_sk.predict(Xn_te)).mean())
print("ε de sklearn (5 primeras):", ada_sk.estimator_errors_[:5].round(3))
print("α de sklearn (5 primeras):", ada_sk.estimator_weights_[:5].round(3), "= 2 × α a mano")
""")
md("""
Coinciden en todas las predicciones y en todos los errores ε. Los pesos αₘ de scikit-learn son **el doble** de los tuyos porque su versión (SAMME) no incluye el factor ½: multiplica por 2 tanto los votos como el exponente de actualización, y el signo de la suma no cambia. Es una diferencia de convención, no de resultado.

Observa también que ε sube ronda a ronda (0.30, 0.39, 0.39…) y α baja: a medida que solo quedan casos difíciles, cada stump aporta menos.

### ✍️ Ejercicio B1
Replica en código el Bloque 2 del cuaderno en papel: 10 observaciones con pesos 0.1, el primer stump falla en 3 (las observaciones 0, 1 y 2). Calcula ε, α, los pesos nuevos normalizados y cuánto pesan en total las 3 observaciones mal clasificadas.
""")
code("""
# tu código aquí
""")
solution("""
w = np.full(10, 0.1)
error = np.array([True, True, True] + [False] * 7)

eps = w[error].sum() / w.sum()
alpha = 0.5 * np.log((1 - eps) / eps)
w_nuevo = w * np.exp(np.where(error, alpha, -alpha))        # errores: e^alpha · aciertos: e^-alpha
w_nuevo = w_nuevo / w_nuevo.sum()

print("ε =", round(eps, 3), " α =", round(alpha, 3))
print("pesos nuevos:", w_nuevo.round(4))
print("peso total de los errores:", round(w_nuevo[error].sum(), 3))
""", "ε = 0.3, α = 0.424, cada error pasa a 0.1667 y cada acierto a 0.0714. Las 3 observaciones mal clasificadas pesan 0.5 en total: tras normalizar, los errores del árbol anterior concentran la mitad del peso. Esto ocurre siempre, para cualquier ε.")

md("""
### B2 · AdaBoost en scikit-learn: estimador base y η (10 min)
En el notebook del profesor, `AdaBoostClassifier` no muestra parámetros de árbol (`max_depth`, `min_samples_leaf`…) porque su **estimador base** es configurable: por defecto es un *stump* (un `DecisionTreeClassifier` de profundidad 1). Si quieres árboles más profundos, los pasas con `estimator=`. Evaluamos con validación cruzada sobre el train de Heart (pipeline completo, con preprocesamiento dentro).
""")
code("""
cv_h = StratifiedKFold(5, shuffle=True, random_state=RS)
configs = {
    "stump · η = 1.0":        dict(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=250, learning_rate=1.0),
    "stump · η = 0.05 (prof.)": dict(n_estimators=250, learning_rate=0.05),            # base por defecto = stump
    "prof. 2 · η = 0.05":     dict(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=250, learning_rate=0.05),
    "prof. 3 · η = 0.05":     dict(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=250, learning_rate=0.05),
}
filas = []
for nombre, kw in configs.items():
    pipe = Pipeline([("pre", make_preprocessor(Xh_tr)), ("m", AdaBoostClassifier(random_state=RS, **kw))])
    r = cross_validate(pipe, Xh_tr, yh_tr, cv=cv_h, scoring="roc_auc", return_train_score=True)
    filas.append({"Configuración": nombre, "AUC train": r["train_score"].mean(), "AUC validación": r["test_score"].mean(),
                  "desvío": r["test_score"].std()})
pd.DataFrame(filas).round(3)
""")
md("""
La validación es prácticamente igual en las cuatro configuraciones (0.87–0.88, con desvíos de 0.03 a 0.05), pero la **brecha** cambia: con árboles de profundidad 3 el AUC de entrenamiento llega a 0.999 (memoriza) y con stumps y η = 0.05 queda en 0.936, la configuración más sobria. Más complejidad en el aprendiz base no compró nada en validación: AdaBoost fue pensado para aprendices **débiles**.

### B3 · ¿Es AdaBoost sensible al ruido? Un experimento (10 min)
La diapositiva 8 y la clase dicen que AdaBoost es sensible al ruido y a las etiquetas incorrectas: como pone énfasis en los errores, una observación mal etiquetada se vuelve «imposible» y acapara el peso. Pongámoslo a prueba: invertimos al azar una fracción del `AHD` del **train** (el test queda limpio) y medimos el AUC en el test, promediando 8 repeticiones.
""")
code("""
pre_h = make_preprocessor(Xh_tr).fit(Xh_tr)
Ph_tr, Ph_te = pre_h.transform(Xh_tr), pre_h.transform(Xh_te)

modelos_ruido = {
    "AdaBoost η = 1":      lambda: AdaBoostClassifier(n_estimators=250, learning_rate=1.0, random_state=RS),
    "AdaBoost η = 0.05":   lambda: AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=RS),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=2, random_state=RS),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=300, random_state=RS),
}
niveles = [0.0, 0.1, 0.2, 0.3]
res = {nombre: [] for nombre in modelos_ruido}
for ruido in niveles:
    for nombre, fabrica in modelos_ruido.items():
        aucs = []
        for semilla in range(8):
            rng = np.random.default_rng(semilla)
            y_sucio = yh_tr.values.copy()
            idx = rng.choice(len(y_sucio), int(ruido * len(y_sucio)), replace=False)
            y_sucio[idx] = 1 - y_sucio[idx]
            modelo = fabrica().fit(Ph_tr, y_sucio)
            aucs.append(roc_auc_score(yh_te, modelo.predict_proba(Ph_te)[:, 1]))
        res[nombre].append(np.mean(aucs))

tabla_ruido = pd.DataFrame(res, index=[f"{int(r * 100)}% de etiquetas invertidas" for r in niveles]).T
display(tabla_ruido.round(3))

for nombre, vals in res.items():
    plt.plot([int(r * 100) for r in niveles], vals, marker="o", label=nombre)
plt.xlabel("% de etiquetas invertidas en el train"); plt.ylabel("AUC en test"); plt.legend()
plt.title("Efecto del ruido en las etiquetas"); plt.show()
""")
md("""
Resultado: **no es una propiedad exclusiva de AdaBoost.** AdaBoost con η = 1 (el AdaBoost «clásico») y Gradient Boosting con árboles de profundidad 2 caen mucho con ruido, más que el bosque. Pero AdaBoost con η = 0.05 (la configuración del profesor), que está muy regularizado, resiste el ruido mejor que el propio bosque. Lo que vuelve frágil a un modelo de boosting es **su flexibilidad** (η alto, árboles profundos, muchas rondas que persiguen los casos difíciles), no el nombre del algoritmo.

### ✍️ Ejercicio B2
Agrega al experimento un `GradientBoostingClassifier` con árboles de profundidad 1 (stumps), η = 0.05 y 250 árboles. ¿Cae más o menos que el de profundidad 2? ¿Qué te dice sobre «AdaBoost es sensible al ruido»?
""")
code("""
# tu código aquí
""")
solution("""
def auc_con_ruido(fabrica, ruido, semillas=range(8)):
    aucs = []
    for semilla in semillas:
        rng = np.random.default_rng(semilla)
        y_sucio = yh_tr.values.copy()
        idx = rng.choice(len(y_sucio), int(ruido * len(y_sucio)), replace=False)
        y_sucio[idx] = 1 - y_sucio[idx]
        modelo = fabrica().fit(Ph_tr, y_sucio)
        aucs.append(roc_auc_score(yh_te, modelo.predict_proba(Ph_te)[:, 1]))
    return np.mean(aucs)

gb_stump = lambda: GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=1, random_state=RS)
print({f"{int(r * 100)}%": round(auc_con_ruido(gb_stump, r), 3) for r in niveles})
""", "El GB con stumps cae bastante menos que el de profundidad 2 (≈ 0.94 → 0.84 al 30%, contra ≈ 0.94 → 0.76). A igual η, árboles más simples son más robustos al ruido. La conclusión matizada: la sensibilidad al ruido depende de la flexibilidad (η, profundidad, número de rondas), no solo del algoritmo.")

# ================================================================== PARTE C
md("""
## Parte C · Gradient Boosting: ajustar el gradiente de la pérdida

En vez de reponderar casos, Gradient Boosting define una **pérdida** L(y, F) y ajusta cada árbol al **gradiente negativo** −∂L/∂F evaluado en las predicciones actuales:

$$F_m(x) = F_{m-1}(x) + \\eta\\, h_m(x), \\qquad h_m \\approx -\\frac{\\partial L}{\\partial F}\\Big|_{F_{m-1}}$$

Con la pérdida cuadrática L = ½(y − F)², el gradiente negativo es **y − F: el residuo**. Para otras pérdidas deja de serlo.

### C1 · Gradient Boosting desde cero (15 min)
Lo programamos con pérdida cuadrática: F₀ = promedio de y; en cada ronda, un árbol de profundidad 2 sobre los residuos. Usamos 5 variables numéricas de Ames para que los empates entre columnas no distorsionen la comparación con scikit-learn.
""")
code("""
cols_a = ["Gr_Liv_Area", "Lot_Area", "Year_Built", "Garage_Area", "Total_Bsmt_SF"]
Xn_a_tr = Xa_tr[cols_a].fillna(0).values
Xn_a_te = Xa_te[cols_a].fillna(0).values


def gb_scratch(X, y, M=100, eta=0.1, profundidad=2):
    F0 = y.mean()
    F = np.full(len(y), F0)
    arboles = []
    for m in range(M):
        r = y - F                                                   # gradiente negativo (pérdida cuadrática)
        arbol = DecisionTreeRegressor(max_depth=profundidad, random_state=0).fit(X, r)
        F = F + eta * arbol.predict(X)
        arboles.append(arbol)
    return F0, arboles


def gb_predict(F0, arboles, X, eta=0.1):
    return F0 + eta * sum(a.predict(X) for a in arboles)


F0_a, arboles_a = gb_scratch(Xn_a_tr, ya_tr.values, M=100, eta=0.1, profundidad=2)
pred_mano = gb_predict(F0_a, arboles_a, Xn_a_te, eta=0.1)

gb_sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2,
                                  criterion="squared_error", random_state=0).fit(Xn_a_tr, ya_tr)
pred_sk = gb_sk.predict(Xn_a_te)

print("RMSE test a mano :", f"{rmse(ya_te, pred_mano):,.1f}")
print("RMSE test sklearn:", f"{rmse(ya_te, pred_sk):,.1f}")
print("Diferencia máxima entre predicciones:", f"{np.abs(pred_mano - pred_sk).max():.2e}")
""")
md("""
Las predicciones coinciden hasta el redondeo numérico: scikit-learn hace exactamente esto (con `criterion="squared_error"`, ya que su valor por defecto, `friedman_mse`, es una variante del mismo criterio). Todo lo demás de `GradientBoostingRegressor` (submuestreo, parada temprana, otras pérdidas) son opciones sobre este bucle.

### ✍️ Ejercicio C1
Con tu `gb_scratch`, entrena 300 árboles con η = 0.1 y calcula el RMSE de validación después de cada ronda (usa un conjunto de validación separado del test: parte `Xn_a_tr` en 75/25). ¿En qué ronda está el mínimo? Esa es la ronda que elegiría el *early stopping*.
""")
code("""
# tu código aquí
""")
solution("""
X1, X2, y1_, y2_ = train_test_split(Xn_a_tr, ya_tr.values, test_size=0.25, random_state=RS)
F0_v, arboles_v = gb_scratch(X1, y1_, M=300, eta=0.1, profundidad=2)

pred_val = np.full(len(y2_), F0_v)
curva = []
for arbol in arboles_v:
    pred_val = pred_val + 0.1 * arbol.predict(X2)
    curva.append(rmse(y2_, pred_val))

mejor = int(np.argmin(curva)) + 1
print("Mejor ronda:", mejor, " RMSE de validación:", f"{min(curva):,.0f}", " | ronda 300:", f"{curva[-1]:,.0f}")
plt.plot(curva); plt.axvline(mejor, color="green", ls="--"); plt.xlabel("Ronda"); plt.ylabel("RMSE validación"); plt.show()
""", "El mínimo está en la ronda 189 (RMSE 30,540); a la ronda 300 el error ya subió un poco (30,696). Con solo 5 variables el modelo se queda sin estructura que aprender y empieza a ajustar ruido: ese es el punto donde el early stopping detendría el entrenamiento.")

md("""
### C2 · La pérdida importa: cuadrática, absoluta y Huber (10 min)
La pérdida define qué es «equivocarse». Con la cuadrática, un error de 100 pesa 100 veces más que uno de 10 (porque se eleva al cuadrado); con la absoluta pesa 10 veces más; Huber es cuadrática para errores chicos y lineal para los grandes. En Ames hay casas caras que son valores extremos, así que la pérdida influye. Entrenamos tres Gradient Boosting y los medimos con RMSE **y** MAE en validación.
""")
code("""
filas = []
for perdida in ("squared_error", "absolute_error", "huber"):
    t0 = perf_counter()
    m = GradientBoostingRegressor(loss=perdida, n_estimators=200, learning_rate=0.1, max_depth=3,
                                  random_state=RS).fit(A_fit, ya_fit)
    p = m.predict(A_val)
    filas.append({"Pérdida": perdida, "RMSE validación": rmse(ya_val, p),
                  "MAE validación": mean_absolute_error(ya_val, p), "Tiempo (s)": perf_counter() - t0})
pd.DataFrame(filas).round(1)
""")
md("""
Lo importante es la **coherencia entre pérdida y métrica**: la cuadrática optimiza directamente el RMSE y la absoluta, el MAE. Aquí, sin embargo, la pérdida absoluta quedó mejor en **ambas** métricas (RMSE ≈ 21.2 mil y MAE ≈ 14.4 mil, contra ≈ 22.0 mil y 14.8 mil de la cuadrática): una pérdida más robusta a los valores extremos puede generalizar mejor aunque no optimice directamente la métrica con que mides. En el notebook del profesor el `GradientBoostingRegressor` de Ames usa `loss='huber'`, pero luego se compara y se optimiza con RMSE: es una discrepancia entre lo que el modelo minimiza al entrenar y lo que se mide al evaluar (ver «Alertas del material»).

### ✍️ Ejercicio C2
Calcula, para el modelo con pérdida cuadrática y el de Huber, el error absoluto de la casa **peor predicha** de la validación y cuántas casas tienen un error superior a 50.000 USD. ¿Qué pérdida es más sensible a los valores extremos?
""")
code("""
# tu código aquí
""")
solution("""
for perdida in ("squared_error", "huber"):
    m = GradientBoostingRegressor(loss=perdida, n_estimators=200, learning_rate=0.1, max_depth=3,
                                  random_state=RS).fit(A_fit, ya_fit)
    error = np.abs(ya_val - m.predict(A_val))
    print(f"{perdida:<14} peor error = {error.max():>9,.0f}   casas con error > 50.000: {int((error > 50000).sum())}")
""", "Los resultados son casi idénticos (peor error ≈ 156 mil en ambos; 19 y 20 casas con error mayor a 50 mil): en este dataset, con δ por defecto y estos árboles, Huber se comporta casi como la cuadrática. La teoría dice que la cuadrática castiga el cuadrado del error y por eso persigue más a las casas extremas, y que la absoluta es la más robusta, pero aquí el efecto es pequeño; lo importante es saber qué penaliza tu pérdida. Qué conviene depende de si, para tu negocio, un error grande es mucho peor que varios chicos.")

md("""
### C3 · Tasa de aprendizaje y número de árboles (15 min)
Según el profesor: «a menor tasa de aprendizaje, mejor desempeño, pero se necesitan más árboles», como estudiar un poco todos los días en lugar de todo la noche anterior. Verifiquémoslo. Para cada η entrenamos 800 árboles y usamos `staged_predict` para ver el RMSE de validación en **cada** número de árboles.
""")
code("""
resultados = []
plt.figure(figsize=(9, 3.8))
for eta in (1.0, 0.3, 0.1, 0.03):
    m = GradientBoostingRegressor(n_estimators=800, learning_rate=eta, max_depth=3, random_state=RS).fit(A_fit, ya_fit)
    curva = [rmse(ya_val, p) for p in m.staged_predict(A_val)]
    k = int(np.argmin(curva))
    resultados.append({"η": eta, "mejor n de árboles": k + 1, "RMSE en el mejor n": curva[k], "RMSE a los 100 árboles": curva[99],
                       "RMSE a los 800": curva[-1]})
    plt.plot(range(1, 801), curva, label=f"η = {eta}")
plt.ylim(17000, 40000); plt.xlabel("Número de árboles"); plt.ylabel("RMSE validación (USD)"); plt.legend()
plt.title("η y número de árboles se compensan"); plt.show()
pd.DataFrame(resultados).round({"η": 2, "RMSE en el mejor n": 0, "RMSE a los 100 árboles": 0, "RMSE a los 800": 0})
""")
md("""
Tres lecturas:
1. **η = 1** llega a su mejor punto con muy pocos árboles y después **empeora**: cada árbol corrige todo su residuo, incluido el ruido.
2. **Menor η → más árboles para el mismo error**: η = 0.3 llega a su mínimo con ≈ 460 árboles (RMSE ≈ 22.5 mil), η = 0.1 con ≈ 590 (21.5 mil) y η = 0.03 con ≈ 790, casi el tope de 800 (21.5 mil).
3. Con **pocos árboles**, η bajo es peor: a los 100 árboles η = 0.03 está muy por encima de η = 0.1. «A menor tasa, mejor» solo es cierto si le das árboles suficientes, y el mejor RMSE de η = 0.1 y η = 0.03 es prácticamente el mismo (21.5 mil): bajar de 0.1 a 0.03 no ganó nada y pidió un 35% más de árboles.

**Early stopping dentro de scikit-learn.** `n_iter_no_change` reserva una fracción de validación y detiene el entrenamiento cuando no mejora en k rondas; así no tienes que adivinar `n_estimators`.
""")
code("""
m_es = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, n_iter_no_change=20,
                                 validation_fraction=0.2, tol=1e-4, random_state=RS).fit(A_fit, ya_fit)
print("Árboles realmente entrenados:", m_es.n_estimators_, "de 3000 posibles")
print("RMSE de validación:", f"{rmse(ya_val, m_es.predict(A_val)):,.0f}")
""")
md("""
En la corrida de referencia entrena 352 de los 3.000 árboles posibles y llega a un RMSE de validación ≈ 21.6 mil: la misma calidad que el mejor punto de la curva, sin adivinar `n_estimators`.

Nota: **por defecto** `n_iter_no_change=None`, es decir, scikit-learn entrena exactamente `n_estimators` árboles. En clase se mencionó una «tolerancia de ≈ 0.001» para detener la búsqueda; esa es la idea general de los optimizadores numéricos, pero en estos modelos la parada hay que **activarla** (y su `tol` por defecto es 1e-4).

### ✍️ Ejercicio C3
Con `max_depth=3` y η = 0.1, ¿qué pasa con el mejor número de árboles y con el RMSE de validación si cambias la profundidad a 1 y a 6? Haz una tabla con profundidad, mejor n y RMSE. ¿Por qué un árbol más profundo necesita menos rondas?
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for prof in (1, 3, 6):
    m = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1, max_depth=prof, random_state=RS).fit(A_fit, ya_fit)
    curva = [rmse(ya_val, p) for p in m.staged_predict(A_val)]
    k = int(np.argmin(curva))
    filas.append({"max_depth": prof, "mejor n": k + 1, "RMSE validación": round(curva[k])})
pd.DataFrame(filas)
""", "Resultado: profundidad 1 → el mínimo está en el tope de 800 árboles (RMSE ≈ 22.2 mil, ¡aún no termina!), profundidad 3 → 587 árboles (21.5 mil), profundidad 6 → 438 árboles (23.2 mil). Un árbol más profundo captura más estructura (interacciones) por ronda, corrige más del residuo en cada paso y el mínimo llega antes; un stump aprende muy poco por ronda y necesita muchos más árboles. Pero uno demasiado profundo (6) sobreajusta antes y termina peor. Por eso los árboles de boosting suelen ser chicos.")

md("""
### C4 · El gradiente de la pérdida logística (5 min)
En clasificación se usa la pérdida logística con p = 1/(1 + e⁻ᶠ). Su gradiente negativo es **y − p**: la diferencia entre la clase real (0 o 1) y la probabilidad actual. A medida que el modelo acierta con seguridad, el gradiente se achica solo.
""")
code("""
def sigmoide(f):
    return 1 / (1 + np.exp(-f))

for y_real, f in [(1, 0.0), (1, 2.0), (1, -2.0), (0, 2.0)]:
    p = sigmoide(f)
    print(f"y = {y_real}  F = {f:>4}  p = {p:.3f}  gradiente negativo y - p = {y_real - p:+.3f}")
""")
md("""
### ✍️ Ejercicio C4
(a) Calcula y − p para (y = 0, p = 0.8) y para (y = 1, p = 0.2) (Bloque 6 del cuaderno). ¿A qué valor de F corresponde p = 0.8? (b) Heart tiene 46% de enfermos: ¿cuánto vale F₀ = ln(p/(1 − p)) para la predicción inicial de un Gradient Boosting de clasificación?
""")
code("""
# tu código aquí
""")
solution("""
print("y - p:", 0 - 0.8, 1 - 0.2)
print("F para p = 0.8:", round(np.log(0.8 / 0.2), 3))
p0 = y_h.mean()
print("F0 de Heart:", round(np.log(p0 / (1 - p0)), 3), "(p0 =", round(p0, 3), ")")
""", "(a) −0.8 y +0.8: modelo muy seguro y equivocado ⇒ gradiente grande. F = ln(4) = 1.386. (b) F₀ = ln(0.459/0.541) = −0.165 (la proporción es 139/303 = 0.459): un valor casi neutro, porque la clase está casi balanceada.")

# ================================================================== PARTE D
md("""
## Parte D · XGBoost y LightGBM

**XGBoost** (*Extreme Gradient Boosting*) es Gradient Boosting con ingeniería: objetivo **regularizado** (pérdida + penalización por complejidad), gradientes de primer y segundo orden, submuestreo de filas y columnas, manejo de nulos y de dispersión, y un núcleo en C++. **LightGBM** busca velocidad con **histogramas** y crece el árbol **por hoja** (siempre expande la hoja con mayor ganancia). No están en scikit-learn: se instalan aparte, aunque ofrecen una interfaz compatible (`fit`, `predict`, `Pipeline`, `GridSearchCV`).

### D1 · XGBoost con early stopping (10 min)
Con η bajo y muchos árboles posibles, ¿cuántos hacen falta? El *early stopping* lo decide: reserva una validación y detiene el entrenamiento cuando no mejora durante `early_stopping_rounds` rondas, quedándose con la mejor.
""")
code("""
xgb_es = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                      early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=RS)
xgb_es.fit(A_fit, ya_fit, eval_set=[(A_fit, ya_fit), (A_val, ya_val)], verbose=False)

curvas = xgb_es.evals_result()
tr, va = curvas["validation_0"]["rmse"], curvas["validation_1"]["rmse"]
print("Rondas ejecutadas:", len(va), "de 1500 | mejor ronda:", xgb_es.best_iteration)
print(f"En la mejor ronda: RMSE train = {tr[xgb_es.best_iteration]:,.0f} | RMSE validación = {va[xgb_es.best_iteration]:,.0f}")

plt.figure(figsize=(8.5, 3.8))
plt.plot(tr, label="entrenamiento", color="gray"); plt.plot(va, label="validación", color="#EA580C")
plt.axvline(xgb_es.best_iteration, color="green", ls="--", label="mejor ronda")
plt.yscale("log"); plt.xlabel("Ronda"); plt.ylabel("RMSE (escala log)"); plt.legend()
plt.title("XGBoost: el entrenamiento sigue bajando, la validación se aplana"); plt.show()
""")
md("""
La brecha entre entrenamiento (≈ 9 mil) y validación (≈ 21 mil) es grande, pero la validación **ya no mejora**: ese es el punto en que seguir entrenando solo memoriza. Reglas de uso: (1) el conjunto de `eval_set` **no puede ser el test** (se usa para elegir la ronda, así que dejaría de ser independiente); (2) si después reentrenas con todo el train, usa `best_iteration + 1` árboles.

### ✍️ Ejercicio D1
Repite el early stopping con **LightGBM** (`LGBMRegressor`, 15 hojas, η = 0.05, `subsample=0.8` con `subsample_freq=1`, `colsample_bytree=0.8`). En LightGBM se usa el *callback* `lgb.early_stopping(50)`. Reporta la mejor ronda y el RMSE de validación, y compáralo con XGBoost.
""")
code("""
# tu código aquí
""")
solution("""
lgb_es = LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, verbosity=-1, n_jobs=-1, random_state=RS)
lgb_es.fit(A_fit, ya_fit, eval_set=[(A_val, ya_val)], eval_metric="rmse",
           callbacks=[lgb.early_stopping(50, verbose=False)])

print("Mejor ronda LightGBM:", lgb_es.best_iteration_, "| RMSE validación:", f"{rmse(ya_val, lgb_es.predict(A_val)):,.0f}")
print("Mejor ronda XGBoost :", xgb_es.best_iteration, "| RMSE validación:", f"{va[xgb_es.best_iteration]:,.0f}")
""", "En la corrida de referencia LightGBM se detiene con la mejor ronda en 279 (RMSE de validación ≈ 21.6 mil) y XGBoost en 754 (≈ 21.0 mil): calidad parecida, pero LightGBM necesita muchas menos rondas con 15 hojas por árbol (más trabajo por árbol que los árboles de profundidad 3 de XGBoost). La ronda óptima depende de la configuración, no solo de la librería.")

md("""
### D2 · LightGBM: hojas, datos pequeños y sobreajuste (10 min)
El crecimiento **por hoja** de LightGBM es eficiente, pero «puede sobreajustar datos pequeños si no se limita» (diapositiva 13). El parámetro que lo limita es `num_leaves` (junto con `min_child_samples`, el mínimo de observaciones por hoja, 20 por defecto). Probemos en Heart (303 filas, solo 227 de entrenamiento) con validación cruzada y observando la brecha entre el AUC de entrenamiento y el de validación.
""")
code("""
filas = []
for hojas in (3, 7, 15, 31, 63):
    for min_hoja in (20, 5):
        pipe = Pipeline([("pre", make_preprocessor(Xh_tr)),
                         ("m", LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=hojas, min_child_samples=min_hoja,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbosity=-1,
                                              n_jobs=1, random_state=RS))])
        r = cross_validate(pipe, Xh_tr, yh_tr, cv=cv_h, scoring="roc_auc", return_train_score=True)
        filas.append({"num_leaves": hojas, "min_child_samples": min_hoja, "AUC train": r["train_score"].mean(),
                      "AUC validación": r["test_score"].mean(), "brecha": r["train_score"].mean() - r["test_score"].mean()})
pd.DataFrame(filas).round(3)
""")
md("""
Lo que muestra la tabla: (1) **el mejor modelo en validación es el más simple**: 3 hojas, AUC de validación ≈ 0.89 y la brecha más chica (≈ 0.08); (2) desde 7 hojas el AUC de entrenamiento llega a ≈ 1.0 mientras la validación **no mejora** (≈ 0.88): toda la flexibilidad extra memoriza; (3) más allá de 7 hojas los resultados son prácticamente idénticos, porque con 227 filas y `min_child_samples` de 20 (o 5) el árbol no puede crear más hojas útiles. Es el riesgo que advierte el profesor: con pocos datos, un LightGBM «por defecto» (31 hojas) ya es más flexible de lo que los datos pueden sostener.

### ✍️ Ejercicio D2
Haz el mismo barrido de `num_leaves` en **Ames** (usa `A_fit`/`A_val`, 300 árboles, η = 0.05, `min_child_samples=20`). ¿Cambia la conclusión sobre el sobreajuste con un dataset de 1.700 filas? Compara la brecha train-validación.
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for hojas in (7, 15, 31, 63, 127):
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=hojas, min_child_samples=20, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, verbosity=-1, n_jobs=-1, random_state=RS).fit(A_fit, ya_fit)
    filas.append({"num_leaves": hojas, "RMSE train": rmse(ya_fit, m.predict(A_fit)), "RMSE validación": rmse(ya_val, m.predict(A_val))})
pd.DataFrame(filas).round(0)
""", "Con 1.758 filas el óptimo está en torno a 31 hojas (validación ≈ 21.3 mil), pero la brecha crece de forma sostenida al subir num_leaves: el RMSE de entrenamiento baja de 16.2 mil (7 hojas) a 8.4 mil (63 hojas) mientras la validación no mejora. Con 63 y 127 hojas el resultado es idéntico: `min_child_samples=20` impide más hojas. El modelo del profesor (31 hojas) tiene en D4 un RMSE de entrenamiento ≈ 8 mil contra ≈ 27 mil en test: por eso lo calificó como sobreajustado.")

md("""
### D3 · Dos trampas: `subsample_freq` y la importancia (10 min)
**Trampa 1: el submuestreo de LightGBM.** En el notebook del profesor, `LGBMClassifier(..., subsample=0.8, ...)` parece usar el 80% de las filas por árbol. Pero en LightGBM `subsample` solo se aplica si `subsample_freq > 0` (por defecto es 0). Comprobémoslo comparando predicciones:
""")
code("""
pre_h2 = make_preprocessor(Xh_tr).fit(Xh_tr)
P_tr, P_te = pre_h2.transform(Xh_tr), pre_h2.transform(Xh_te)

def pred_lgbm(**kw):
    return LGBMClassifier(n_estimators=100, verbosity=-1, random_state=1, **kw).fit(P_tr, yh_tr).predict_proba(P_te)[:, 1]

p_sin = pred_lgbm(subsample=0.5)                      # como en el notebook del profesor
p_uno = pred_lgbm(subsample=1.0)
p_con = pred_lgbm(subsample=0.5, subsample_freq=1)    # submuestreo realmente activo
print("subsample=0.5 sin subsample_freq == subsample=1.0 :", np.allclose(p_sin, p_uno))
print("subsample=0.5 con subsample_freq=1 es distinto    :", not np.allclose(p_sin, p_con))

def pred_xgb(**kw):
    return XGBClassifier(n_estimators=100, random_state=1, **kw).fit(P_tr, yh_tr).predict_proba(P_te)[:, 1]
print("XGBoost: subsample=0.5 cambia el resultado        :", not np.allclose(pred_xgb(subsample=0.5), pred_xgb(subsample=1.0)))
""")
md("""
En el notebook del profesor, entonces, el submuestreo de filas de LightGBM **no estaba activo** (el de columnas, `colsample_bytree`, sí). Es un error silencioso: nada falla, solo no hace lo que parece.

**Trampa 2: la «importancia» no significa lo mismo en cada librería.** El notebook usa `feature_importances_` para ambos modelos con el título «importancia por ganancia o impureza». Pero:
""")
code("""
xgb_h = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RS).fit(P_tr, yh_tr)
lgb_h = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, verbosity=-1, random_state=RS).fit(P_tr, yh_tr)
nombres = pre_h2.get_feature_names_out()

print("XGBoost  importance_type:", xgb_h.importance_type or "gain (ganancia, por defecto)")
print("LightGBM importance_type:", lgb_h.importance_type, "(número de veces que se usó la variable en un corte)")

perm = permutation_importance(xgb_h, P_te, yh_te, scoring="roc_auc", n_repeats=20, random_state=RS)
tabla = pd.DataFrame({
    "XGBoost (ganancia)": pd.Series(xgb_h.feature_importances_, nombres),
    "LightGBM (nº de cortes)": pd.Series(lgb_h.feature_importances_ / lgb_h.feature_importances_.sum(), nombres),
    "Permutación XGBoost (test)": pd.Series(perm.importances_mean, nombres),
})
tabla.sort_values("Permutación XGBoost (test)", ascending=False).head(8).round(3)
""")
md("""
Las dos primeras columnas miden cosas distintas: una suma de **ganancia** y otra cuenta **cortes** (una variable continua, con muchos umbrales posibles, se usa muchas veces aunque aporte poco). La tercera, la **permutación** sobre datos no vistos, es la comparable entre modelos (se vio en la Clase 34): mide cuánto cae el AUC al desordenar una variable. Y recuerda: ninguna importancia es un efecto causal.

### ✍️ Ejercicio D3 (Ejercicio 5 del profesor)
Calcula la importancia por **permutación** sobre el test para el LightGBM `lgb_h`, con `scoring="roc_auc"`, y compárala con su `feature_importances_`. (a) ¿Cambian las variables más importantes? (b) ¿Qué métrica usarías en clasificación? (c) ¿Por qué la importancia no es causalidad?
""")
code("""
# tu código aquí
""")
solution("""
perm_l = permutation_importance(lgb_h, P_te, yh_te, scoring="roc_auc", n_repeats=20, random_state=RS)
comp = pd.DataFrame({"LightGBM (nº de cortes)": pd.Series(lgb_h.feature_importances_, nombres),
                     "Permutación (test)": pd.Series(perm_l.importances_mean, nombres)})
print("Top 5 por nº de cortes:", comp["LightGBM (nº de cortes)"].nlargest(5).index.tolist())
print("Top 5 por permutación :", comp["Permutación (test)"].nlargest(5).index.tolist())
comp.sort_values("Permutación (test)", ascending=False).head(8).round(3)
""", "(a) Sí cambian: por número de cortes dominan las variables continuas (Age, MaxHR, Chol, Oldpeak, RestBP); por permutación en el test destacan `Ca` (vasos coloreados), el dolor de pecho asintomático (`ChestPain_asymptomatic`), `Sex` y `Slope`. (b) AUC (o recall/F1 si el costo de un falso negativo es alto), la misma métrica con que evalúas el modelo. (c) Porque mide cuánto usa el modelo una variable para predecir, no qué ocurriría si se interviniera sobre ella; una variable puede ser útil solo por estar correlacionada con la causa real.")

md("""
### D4 · Comparación de los cinco modelos (15 min)
Ahora bajo el **mismo pipeline** y la misma partición, los cuatro modelos del profesor más un **Random Forest** como referencia. Configuraciones: las del notebook del profesor (η = 0.05; 250 árboles en Heart y 350 en Ames), salvo que LightGBM lleva `subsample_freq=1` para que el submuestreo funcione.

Para Heart, como solo hay 76 pacientes en el test (un paciente = 1.3 puntos de exactitud), usamos además **validación cruzada repetida 5×3** sobre los 303 pacientes: es una comparación mucho más estable.
""")
code("""
def modelos_clasificacion():
    return {
        "AdaBoost": AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=2, random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", n_jobs=1, random_state=RS),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbosity=-1, n_jobs=1, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RS),
    }

# 1) Partición del profesor (75/25): un solo test de 76 pacientes
filas = []
for nombre, modelo in modelos_clasificacion().items():
    pipe = Pipeline([("pre", make_preprocessor(Xh_tr)), ("m", modelo)])
    t0 = perf_counter(); pipe.fit(Xh_tr, yh_tr); dt = perf_counter() - t0
    p = pipe.predict(Xh_te)
    filas.append({"Modelo": nombre, "Accuracy train": accuracy_score(yh_tr, pipe.predict(Xh_tr)),
                  "Accuracy test": accuracy_score(yh_te, p), "Recall": recall_score(yh_te, p), "F1": f1_score(yh_te, p),
                  "ROC AUC test": roc_auc_score(yh_te, pipe.predict_proba(Xh_te)[:, 1]), "Tiempo (s)": dt})
display(pd.DataFrame(filas).sort_values("ROC AUC test", ascending=False).round(3).reset_index(drop=True))

# 2) Validación cruzada repetida 5x3 con los 303 pacientes
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
filas = []
for nombre, modelo in modelos_clasificacion().items():
    pipe = Pipeline([("pre", make_preprocessor(X_h)), ("m", modelo)])
    r = cross_validate(pipe, X_h, y_h, cv=cv_rep, scoring="roc_auc", n_jobs=-1)
    filas.append({"Modelo": nombre, "AUC (CV 5×3) media": r["test_score"].mean(), "desvío": r["test_score"].std()})
display(pd.DataFrame(filas).sort_values("AUC (CV 5×3) media", ascending=False).round(3).reset_index(drop=True))
""")
md("""
En la partición única, el orden de los modelos depende de uno o dos pacientes: en la primera tabla AdaBoost lidera el AUC (0.937) y es el único sin sobreajuste evidente (exactitud 0.86 en train y 0.84 en test), mientras que el bosque y LightGBM casi memorizan el entrenamiento (≈ 1.0). En la validación cruzada repetida **los cinco modelos quedan entre 0.89 y 0.91 de AUC, con desvíos de ≈ 0.04**: AdaBoost sigue primero (0.912), pero a solo 0.02 del último, **menos que un desvío**. El mensaje del profesor es correcto: «a problemas simples, modelos simples; no es que uno sea el mejor de todos». AdaBoost no «ganó» porque sea superior, sino porque con 303 filas no se distingue de los demás.

Con **Ames** (2.930 casas) la situación es otra: hay más datos y más estructura.
""")
code("""
def modelos_regresion():
    return {
        "AdaBoost": AdaBoostRegressor(n_estimators=350, learning_rate=0.05, random_state=RS),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=0.05, max_depth=2, random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, random_state=RS),
        "LightGBM": LGBMRegressor(n_estimators=350, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbosity=-1, n_jobs=-1, random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_features=1/3, n_jobs=-1, random_state=RS),
    }

filas = []
for nombre, modelo in modelos_regresion().items():
    pipe = Pipeline([("pre", make_preprocessor(Xa_tr)), ("m", modelo)])
    t0 = perf_counter(); pipe.fit(Xa_tr, ya_tr); dt = perf_counter() - t0
    filas.append({"Modelo": nombre, "RMSE train": rmse(ya_tr, pipe.predict(Xa_tr)), "RMSE test": rmse(ya_te, pipe.predict(Xa_te)),
                  "MAE test": mean_absolute_error(ya_te, pipe.predict(Xa_te)), "R² test": r2_score(ya_te, pipe.predict(Xa_te)),
                  "Tiempo (s)": dt})
tabla_ames = pd.DataFrame(filas).sort_values("RMSE test").reset_index(drop=True)
display(tabla_ames.style.format({"RMSE train": "${:,.0f}", "RMSE test": "${:,.0f}", "MAE test": "${:,.0f}", "R² test": "{:.3f}", "Tiempo (s)": "{:.2f}"}))
""")
md("""
Aquí sí hay un ordenamiento claro: **XGBoost lidera** (RMSE ≈ 24 mil), seguido de Gradient Boosting, LightGBM y el bosque (≈ 27 mil), y **AdaBoost queda muy atrás** (≈ 36 mil). ¿Por qué falla AdaBoost en regresión? Su estimador base por defecto es un árbol de profundidad 3 con el algoritmo AdaBoost.R2: es mucho menos flexible que los métodos de gradiente para una variable continua. Observa además que **LightGBM y el bosque tienen una brecha enorme** entre el RMSE de entrenamiento (≈ 8–10 mil) y el de test (≈ 27 mil): sobreajuste, tal como lo señaló el profesor.

También cambia el costo: el AdaBoost de scikit-learn, el más «simple», fue el más lento, y XGBoost, el más sofisticado, uno de los más rápidos (núcleo en C++, paralelismo dentro de cada árbol).

### ✍️ Ejercicio D4 (Ejercicios 1 y 3 del profesor)
(a) En la tabla de Heart de la partición única, ¿cuál modelo tiene la mayor brecha entre `Accuracy train` y `Accuracy test`? ¿Sugiere sobreajuste? (b) Si el objetivo fuera detectar la mayor cantidad posible de pacientes con enfermedad, ¿qué métrica priorizarías? (c) En Ames, calcula la brecha relativa (RMSE test / RMSE train) de cada modelo: ¿cuál generaliza peor?
""")
code("""
# tu código aquí
""")
solution("""
tabla_ames["RMSE test / RMSE train"] = tabla_ames["RMSE test"] / tabla_ames["RMSE train"]
display(tabla_ames[["Modelo", "RMSE train", "RMSE test", "RMSE test / RMSE train"]].round(2))
""", "(a) Las mayores brechas son las del bosque (accuracy 1.000 en train vs 0.855 en test) y LightGBM (0.996 vs 0.855), seguidos de XGBoost (0.991 vs 0.868) y Gradient Boosting (0.969 vs 0.868): casi memorizan el entrenamiento, lo que sugiere sobreajuste (más complejos: más profundidad u hojas). AdaBoost con stumps casi no tiene brecha (0.863 vs 0.842). (b) El recall (sensibilidad): cada falso negativo es un paciente enfermo no detectado; después revisa la precisión para no saturar de falsos positivos. (c) LightGBM (3.4 veces) y el bosque (2.8 veces); AdaBoost es el que menos brecha tiene (1.2), pero porque ya ajusta mal en entrenamiento (subajuste). Una brecha pequeña no es mérito si el error es alto en ambos lados.")

md("""
### D5 · Optimización con búsqueda aleatoria (15 min)
El profesor optimizó AdaBoost con `GridSearchCV` y XGBoost con `RandomizedSearchCV`, siempre **solo con el train** y con el test intacto, y dejó como ejercicio optimizar Gradient Boosting y LightGBM (clasificación), y XGBoost (regresión). Dos lecciones suyas que vale la pena recordar: (1) **incluye en la grilla tu configuración base**: si no, la búsqueda puede elegir algo peor que lo que ya tenías; (2) «no es que el modelo más avanzado gane siempre»: tras optimizar, las diferencias siguen siendo pequeñas.

Aquí un ejemplo con **LightGBM en Heart** (ejercicio 2 del profesor), con pocas iteraciones para ir rápido.
""")
code("""
pipe_lgbm = Pipeline([("pre", make_preprocessor(Xh_tr)),
                      ("m", LGBMClassifier(verbosity=-1, n_jobs=1, subsample_freq=1, random_state=RS))])
busqueda_lgbm = RandomizedSearchCV(
    pipe_lgbm,
    param_distributions={"m__n_estimators": [80, 150, 250, 400], "m__learning_rate": [0.02, 0.05, 0.1],
                         "m__num_leaves": [4, 7, 15], "m__min_child_samples": [5, 10, 20],
                         "m__subsample": [0.65, 0.8, 1.0], "m__colsample_bytree": [0.65, 0.8, 1.0]},
    n_iter=15, scoring="roc_auc", cv=cv_h, random_state=RS, n_jobs=-1, return_train_score=True)
busqueda_lgbm.fit(Xh_tr, yh_tr)

print("Mejores parámetros:", {k.replace("m__", ""): v for k, v in busqueda_lgbm.best_params_.items()})
print("AUC de validación cruzada:", round(busqueda_lgbm.best_score_, 3))
print("AUC en test (se mira UNA vez):", round(roc_auc_score(yh_te, busqueda_lgbm.predict_proba(Xh_te)[:, 1]), 3))
""")
md("""
### ✍️ Ejercicio D5 (Ejercicio 4 del profesor)
Optimiza **XGBoost en Ames** con `RandomizedSearchCV` (10 combinaciones, `KFold(3)`, `scoring="neg_root_mean_squared_error"`; parámetros: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`) usando solo `Xa_tr`. Compara su RMSE de test con el XGBoost base de D4 y con el LightGBM. No uses estratificación: en regresión no tiene sentido.
""")
code("""
# tu código aquí
""")
solution("""
cv_reg = KFold(3, shuffle=True, random_state=RS)
pipe_xgb = Pipeline([("pre", make_preprocessor(Xa_tr)), ("m", XGBRegressor(n_jobs=-1, random_state=RS))])
busqueda_xgb = RandomizedSearchCV(
    pipe_xgb,
    param_distributions={"m__n_estimators": [200, 350, 500], "m__learning_rate": [0.03, 0.05, 0.1],
                         "m__max_depth": [2, 3, 4, 5], "m__min_child_weight": [1, 3, 5],
                         "m__subsample": [0.7, 0.85, 1.0], "m__colsample_bytree": [0.6, 0.8, 1.0]},
    n_iter=10, scoring="neg_root_mean_squared_error", cv=cv_reg, random_state=RS, n_jobs=1)
busqueda_xgb.fit(Xa_tr, ya_tr)

print("Mejores parámetros:", {k.replace("m__", ""): v for k, v in busqueda_xgb.best_params_.items()})
print("RMSE de validación cruzada:", f"{-busqueda_xgb.best_score_:,.0f}")
print("RMSE test (optimizado)   :", f"{rmse(ya_te, busqueda_xgb.predict(Xa_te)):,.0f}")
base = tabla_ames.set_index("Modelo")["RMSE test"]
print("RMSE test XGBoost base   :", f"{base['XGBoost']:,.0f}", "| LightGBM base:", f"{base['LightGBM']:,.0f}")
""", "En la corrida de referencia, la búsqueda con 10 combinaciones quedó en un RMSE de test de ≈ 24.8 mil, algo **peor** que el XGBoost base (24.4 mil): con pocas combinaciones y una validación cruzada de 3 bloques, la búsqueda puede no superar a una buena configuración inicial. Es justo la lección 1 del profesor: incluye tu configuración base en el espacio de búsqueda. Lo importante es el método (solo train, CV, test una sola vez), no el número exacto.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene corregir

En las diapositivas, la clase y el notebook del profesor hay imprecisiones que verificamos (algunas, con código, en este mismo notebook):

| Dónde | Qué dice | Qué es correcto |
|---|---|---|
| Clase | «AdaBoost significa *additive* boosting» | *Adaptive Boosting* (Freund y Schapire, 1997) |
| Diapositiva 16 | Título: «Comparación de los **cinco** modelos» | La tabla solo trae cuatro filas; falta CatBoost (sí aparece en la diapositiva 18). La numeración impresa salta del 13 al 15: falta la lámina «14/20» |
| Clase | «Gradient Boosting se detiene con tolerancia ≈ 0.001» | En scikit-learn no hay parada por defecto (`n_iter_no_change=None`): entrena `n_estimators` árboles; su `tol` por defecto es 1e-4 y solo actúa si activas la parada (C3) |
| Clase | «Boosting trabaja sobre y − F» | Solo con pérdida cuadrática; en general es el gradiente negativo (C2, C4) |
| Clase | «A menor tasa de aprendizaje, mejor desempeño» | Solo si aumentas los árboles; con pocos árboles, η bajo subajusta (C3) |
| Diapositiva 8 y clase | «AdaBoost es sensible al ruido» | Depende de η, profundidad y rondas: un AdaBoost muy regularizado resistió mejor que el bosque (B3) |
| Notebook | `LGBM…(subsample=0.8)` | Sin `subsample_freq > 0` el submuestreo de filas **no se aplica** (D3) |
| Notebook | Importancia «por ganancia o impureza» para XGBoost y LightGBM | XGBoost usa ganancia; LightGBM, por defecto, **número de cortes** (D3) |
| Notebook | `GradientBoostingRegressor(loss='huber')` evaluado y optimizado con RMSE | La pérdida de entrenamiento y la métrica de selección no coinciden (C2) |
| Clase y notebook | «Se eliminan Latitude y Longitude» | El notebook entregado no las elimina (aquí sí) |
| Notebook | `sns.boxplot(..., palette=...)` sin `hue` | Está obsoleto en seaborn reciente (da advertencia); usa `color=` o `hue=` |
| Notebook | `Gradient Boosting` y `AdaBoost` ganan o pierden por 1 paciente | Con 76 pacientes de test, una sola partición no es evidencia: usa CV repetida (D4) |
| Diapositiva 3 vs 9 | `F(x) = f₁ + ηf₂ + …` vs. `Fₘ = Fₘ₋₁ + η·hₘ` | Son consistentes si se interpreta f₁ como F₀ (la predicción base no se multiplica por η) |

**Transcripción con ruido**: «bugging/Bindin» = bagging, «Roundoff Forest/Random Offers» = Random Forest, «Caldus» = CatBoost, «XzBus/Xtbus» = XGBoost, «Light DBM» = LightGBM, «KDNN» = KNN, «Cycle Learn» = scikit-learn, «Nameless Housing» = Ames Housing, «KERX» = Heart. Hay bucles de Whisper (líneas 216–244 y 1337–1341) y un comentario ajeno a la clase (líneas 508–512).
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (20 min, SIN mirar lo anterior)

Eres el analista y debes entregar un modelo de precio para Ames **sin tocar el test hasta el final**:

1. Parte `Xa_tr` en entrenamiento y validación (75/25, `random_state=0`) y preprocesa con `make_preprocessor` (ajustado solo con el entrenamiento).
2. Entrena un **XGBoost con early stopping** (η = 0.05, `max_depth=3`, hasta 2.000 árboles) sobre ese entrenamiento y anota la mejor ronda.
3. Con `n_estimators = mejor ronda + 1`, **reentrena con todo `Xa_tr`** (dentro de un `Pipeline`) y calcula el RMSE de test **una sola vez**.
4. Compáralo con un `RandomForestRegressor` (300 árboles) entrenado igual. ¿Cuál elegirías y por qué (error, tiempo, brecha train-test)?
5. Responde sin código: ¿qué dos cosas del material del profesor corregirías antes de dárselo a un cliente (dato: una es de LightGBM y otra de la importancia de variables)?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
Xc_fit, Xc_val, yc_fit, yc_val = train_test_split(Xa_tr, ya_tr, test_size=0.25, random_state=0)
pre_c = make_preprocessor(Xc_fit).fit(Xc_fit)
xgb_c = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                     early_stopping_rounds=50, n_jobs=-1, random_state=RS)
xgb_c.fit(pre_c.transform(Xc_fit), yc_fit, eval_set=[(pre_c.transform(Xc_val), yc_val)], verbose=False)
mejor = xgb_c.best_iteration + 1
print("Mejor número de árboles:", mejor)

final = Pipeline([("pre", make_preprocessor(Xa_tr)),
                  ("m", XGBRegressor(n_estimators=mejor, learning_rate=0.05, max_depth=3, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=-1, random_state=RS))])
t0 = perf_counter(); final.fit(Xa_tr, ya_tr); t_xgb = perf_counter() - t0

bosque = Pipeline([("pre", make_preprocessor(Xa_tr)),
                   ("m", RandomForestRegressor(n_estimators=300, max_features=1/3, n_jobs=-1, random_state=RS))])
t0 = perf_counter(); bosque.fit(Xa_tr, ya_tr); t_rf = perf_counter() - t0

for nombre, modelo, t in (("XGBoost (early stopping)", final, t_xgb), ("Random Forest", bosque, t_rf)):
    print(f"{nombre:<26} RMSE train = {rmse(ya_tr, modelo.predict(Xa_tr)):>7,.0f}  RMSE test = {rmse(ya_te, modelo.predict(Xa_te)):>7,.0f}  tiempo = {t:.1f}s")
""", "En la corrida de referencia: mejor ronda 754; XGBoost con early stopping queda en un RMSE de test de ≈ 23.9 mil (train 10.3 mil) y el bosque en ≈ 27.5 mil (train 9.9 mil), con el bosque mostrando la mayor brecha train-test. Recomendación: XGBoost (menor error, entrenamiento rápido y número de árboles elegido con datos, no a ojo). Correcciones antes de entregar: (1) en LightGBM, `subsample` no hace nada sin `subsample_freq=1`; (2) la importancia de XGBoost (ganancia) y de LightGBM (nº de cortes) no se pueden comparar: usar permutación sobre el test. (Otras válidas: loss='huber' evaluada con RMSE; AdaBoost «sensible al ruido» depende de η.)")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué diferencia fundamental hay entre cómo usan los árboles el bagging y el boosting? ¿Por qué boosting no se puede paralelizar entre árboles?
2. En AdaBoost, ¿qué ocurre con el peso de una observación mal clasificada y con el peso (α) de un árbol con error ε = 0.5? ¿Y con ε > 0.5?
3. ¿Qué predice cada árbol en Gradient Boosting y cuándo eso coincide con los residuos?
4. ¿Por qué η y `n_estimators` se ajustan juntos? ¿Qué le pasaría a un modelo con η = 0.01 y 100 árboles?
5. ¿Qué es el early stopping, qué conjunto necesita y por qué ese conjunto no puede ser el test?
6. Nombra dos diferencias entre XGBoost y LightGBM, y una situación en que preferirías cada uno.

<details><summary><b>Respuestas</b></summary>

1. En bagging los árboles son **independientes** (cada uno con su muestra) y se promedian: reduce varianza y se paraleliza. En boosting se entrenan **en secuencia** sobre el dataset completo y cada uno usa el error del anterior (pesos en AdaBoost, gradiente en Gradient Boosting): reduce sesgo y error. No se paraleliza entre árboles porque el árbol *m* necesita el resultado del *m−1*.
2. Su peso **sube** (se multiplica por e^α, luego se normaliza). Con ε = 0.5, α = ½ ln(1) = 0: el árbol no aporta al voto; con ε > 0.5, α < 0: el árbol es peor que el azar y su voto se invierte.
3. Cada árbol predice el **gradiente negativo de la pérdida** en las predicciones actuales. Con pérdida cuadrática (½(y − F)²) eso es exactamente el residuo y − F; con la absoluta es el signo del residuo; con la logística es y − p.
4. Porque η es el tamaño del paso: con η bajo cada árbol aporta poco y hacen falta más árboles para llegar al mismo punto. Con η = 0.01 y 100 árboles el residuo sigue en 0.99¹⁰⁰ ≈ 37% del inicial: el modelo queda subajustado. Hay que subir los árboles o usar early stopping.
5. Es detener el entrenamiento cuando el error en un conjunto de **validación** no mejora durante k rondas, eligiendo así `n_estimators` automáticamente. No puede ser el test porque se usa para tomar una decisión (cuándo parar), y entonces el test dejaría de dar una estimación honesta del error.
6. XGBoost: crecimiento por nivel, objetivo regularizado, muy robusto y buen candidato por defecto. LightGBM: crecimiento por hoja con histogramas, mucho más rápido con muchas filas, pero riesgo de sobreajuste en datos pequeños. XGBoost: rendimiento general o producción con pocos/medianos datos; LightGBM: gran escala (millones de filas) o cómputo limitado.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** repaso con el profesor Galasso (ejemplos y ejercicios de las dos sesiones). Revisa `Guias/apuntes/C36_Ejercicios_Cuaderno_Boosting.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
