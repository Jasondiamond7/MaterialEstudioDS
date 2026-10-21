"""Genera el notebook guiado de Bagging y Random Forest (Clase 35).

Uso normal:      python notebook_c35.py             -> Guias/notebooks/C35_RandomForest.ipynb
Modo de prueba:  C35_TEST=1 python notebook_c35.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C35_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C35_RandomForest_test.ipynb" if TEST else "C35_RandomForest.ipynb")
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
# Clase 35 · Bagging y Random Forest
**Diplomado Data Science** · Prof. Bastian Galasso

Material de apoyo:
- `Guias/C35_Guia_RandomForest.pdf` — guía visual de la clase.
- `Guias/apuntes/C35_Apuntes_RandomForest.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C35_Ejercicios_Cuaderno_RandomForest.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor: `Clase 35-BastianGalasso/Clase Random Forest.pptx`, `Ejemplo Random Forest y Bagging.ipynb` y `Random Forest en imágenes.ipynb`.
- Transcripción: `VideosClases/clase35-RandomForest Y Bagging.txt`.

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `scipy`, `openpyxl` (para leer `Heart.xlsx`) y `matplotlib`. Instala una vez en tu entorno:
```
pip install scikit-learn scipy openpyxl matplotlib pandas
```
Los datos (`Heart.xlsx`, `ames_housing.csv`) están en la carpeta de la clase; el notebook la busca solo.

### Objetivos
1. Entender por qué un árbol es **inestable** y cómo el **bootstrap** crea diversidad sin datos nuevos.
2. Programar **bagging desde cero** (bootstrap + árboles + voto + error OOB) y comprobar que reproduce a scikit-learn.
3. Verificar con datos la fórmula de la **varianza de un promedio** ρσ² + (1 − ρ)σ²/B y ver cómo `max_features` reduce ρ.
4. Usar el **error OOB**, ajustar hiperparámetros (`GridSearchCV`, `RandomizedSearchCV`) y evaluar sin mirar el test.
5. Interpretar el modelo (importancia por impureza vs permutación), conocer sus límites (extrapolación, imágenes) y detectar lo que el material original afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y carga de datos |
| A | 10 | A1 · Bootstrap y fracción OOB a mano |
| A | 10 | A2 · La inestabilidad de un árbol |
| A | 20 | A3 · Bagging desde cero (con OOB) |
| B | 15 | B1 · Los modelos base del profesor: Bagging vs Random Forest |
| B | 15 | B2 · Verificar la fórmula ρσ² + (1 − ρ)σ²/B |
| B | 15 | B3 · `max_features`: por nodo, no por árbol |
| B | 10 | B4 · ¿Cuántos árboles? Curvas de error |
| C | 15 | C1 · OOB vs validación cruzada vs test |
| C | 10 | C2 · Complejidad del árbol, `class_weight` y `max_samples` |
| C | 15 | C3 · `GridSearchCV` vs `RandomizedSearchCV` (preguntas del profesor) |
| D | 10 | D1 · Importancia: impureza vs permutación |
| D | 10 | D2 · Random Forest no extrapola |
| D | 15 | D3 · Imágenes (el ejemplo del profesor, con otro conjunto) |
| D | 10 | D4 · Bagging de modelos estables y comparación final |
| — | 10 | Alertas del material |
| — | 20 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook con `scikit-learn 1.9`. Con otras versiones o procesadores pueden variar un poco (sobre todo los tiempos).
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

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 42   # semilla

# Busca la carpeta con Heart.xlsx y ames_housing.csv subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase*35*")))
carpeta = next(raiz.glob("Clase*35*"))

print("scikit-learn:", sklearn.__version__)
print("Datos en:", carpeta.name)
""")
md("""
### Los datos
- **Heart** (303 pacientes, 13 variables): clasificación. `AHD` = `Yes` si hay enfermedad cardíaca (46 % de los casos: 139 de 303).
- **Ames Housing** (2.930 casas, 81 variables): regresión del precio de venta `Sale_Price`.

Seguimos el notebook del profesor: separación 75/25 estratificada en Heart y 80/20 en Ames, y el mismo preprocesamiento **dentro de un pipeline** (mediana para numéricas; moda y one-hot para categóricas). A diferencia de la Clase 36, aquí **no** se eliminan `Latitude` y `Longitude`: el notebook de esta clase las conserva.
""")
code("""
def make_preprocessor(X, escalar=False):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    pasos_num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if escalar else [])
    return ColumnTransformer([
        ("num", Pipeline(pasos_num), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


# --- Heart (clasificación)
heart = pd.read_excel(carpeta / "Heart.xlsx")
X_h, y_h = heart.drop(columns="AHD"), heart["AHD"].map({"No": 0, "Yes": 1})
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_h, y_h, test_size=0.25, stratify=y_h, random_state=RS)

# --- Ames (regresión)
ames = pd.read_csv(carpeta / "ames_housing.csv").drop(columns=["Unnamed: 0"])
X_a, y_a = ames.drop(columns="Sale_Price"), ames["Sale_Price"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_a, y_a, test_size=0.20, random_state=RS)

# Una validación DENTRO del entrenamiento de Ames para las curvas (el test queda intacto)
Xa_fit, Xa_val, ya_fit, ya_val = train_test_split(Xa_tr, ya_tr, test_size=0.25, random_state=RS)
pre_a = make_preprocessor(Xa_tr).fit(Xa_fit)
A_fit, A_val = pre_a.transform(Xa_fit), pre_a.transform(Xa_val)
yf, yv = ya_fit.values, ya_val.values

# Heart ya preprocesado (para los experimentos con matrices)
pre_h = make_preprocessor(Xh_tr).fit(Xh_tr)
H_tr, H_te = pre_h.transform(Xh_tr), pre_h.transform(Xh_te)
nombres_h = pre_h.get_feature_names_out()

print("Heart  :", heart.shape, "| train", Xh_tr.shape, "| test", Xh_te.shape, "| % AHD:", round(y_h.mean(), 3))
print("Ames   :", ames.shape, "| train", Xa_tr.shape, "| test", Xa_te.shape, "| precio mediano:", f"${y_a.median():,.0f}")
print("Heart tras preprocesar:", H_tr.shape, "| Ames (fit/val):", A_fit.shape, A_val.shape)
""")

# ================================================================== PARTE A
md("""
## Parte A · Bootstrap, OOB y el árbol inestable

Un árbol profundo tiene **bajo sesgo y alta varianza**: pequeños cambios en los datos cambian sus preguntas. La idea de la clase es entrenar muchos árboles, cada uno con una muestra distinta, y **agregar** sus predicciones. Para tener muestras distintas sin recolectar datos nuevos se usa el **bootstrap**: n extracciones **con reemplazo** de las n filas originales.

### A1 · Bootstrap y fracción OOB a mano (10 min)
Es el ejemplo de la diapositiva 5, con los números 1 a 8. Las filas que no salen en una muestra son su **OOB** (*out of bag*, «fuera de la bolsa»).
""")
code("""
rng = np.random.default_rng(1)
datos = np.arange(1, 9)
for i in range(1, 4):
    muestra = np.sort(rng.choice(datos, size=8, replace=True))
    fuera = np.setdiff1d(datos, muestra)
    print(f"Bootstrap {i}: {muestra}  únicos = {len(set(muestra))}  OOB = {fuera}")

print()
print("  n     P(fila fuera) teórica (1-1/n)^n   simulada (2.000 muestras)")
for n in (8, 50, 227, 2344):
    teorica = (1 - 1 / n) ** n
    simulada = np.mean([1 - len(np.unique(rng.integers(0, n, n))) / n for _ in range(2000)])
    print(f"{n:5d}   {teorica:8.4f}                           {simulada:8.4f}")
print("e^-1 =", round(np.exp(-1), 4))
""")
md("""
Con n pequeño la fracción OOB es un poco menor que 36.8 % (para n = 8 es 34.4 %), pero converge muy rápido a e⁻¹ ≈ 0.368. Por eso **cada muestra bootstrap contiene ≈ 63.2 % de filas únicas** y deja ≈ 36.8 % fuera. Dos aclaraciones sobre lo que se dijo en clase:
- La muestra bootstrap **por defecto tiene el mismo tamaño n** que los datos (scikit-learn: `max_samples=None`). No «500 de 1.000»: son 1.000 extracciones con repetidos, de las que ≈ 632 son filas distintas. Con `max_samples` puedes pedir otro tamaño.
- Las muestras bootstrap **no son independientes entre sí** (todas salen de los mismos datos): lo que es independiente es cada extracción *dentro* de una muestra.

### ✍️ Ejercicio A1
(a) Calcula la probabilidad de que una fila quede fuera de una muestra bootstrap para n = 5, 20 y 1.000. (b) ¿Cuántas filas **distintas** se esperan en una muestra bootstrap con n = 100? (c) Simula 5.000 muestras con n = 20 y comprueba (a).
""")
code("""
# tu código aquí
""")
solution("""
for n in (5, 20, 1000):
    print(f"n = {n:4d}: P(fuera) = {(1 - 1 / n) ** n:.4f}")

n = 100
print("Filas distintas esperadas con n = 100:", round(n * (1 - (1 - 1 / n) ** n), 1))

rng = np.random.default_rng(0)
sim = np.mean([1 - len(np.unique(rng.integers(0, 20, 20))) / 20 for _ in range(5000)])
print("Simulado con n = 20:", round(sim, 4), "(teórico", round((1 - 1 / 20) ** 20, 4), ")")
""", "(a) n = 5: 0.328; n = 20: 0.358; n = 1.000: 0.368. (b) 100·(1 − 0.99¹⁰⁰) ≈ 63.4 filas distintas. (c) La simulación debe quedar cerca de 0.358.")

md("""
### A2 · La inestabilidad de un árbol (10 min)
Entrenamos **200 árboles sin podar**, cada uno con una muestra bootstrap de Heart, y miramos (1) con qué variable parte cada uno (la raíz) y (2) cuántas veces dos árboles discrepan al clasificar a un mismo paciente del test.
""")
code("""
rng = np.random.default_rng(RS)
n = len(H_tr)
cat_h = Xh_tr.select_dtypes(exclude=np.number).columns.tolist()

def variable_original(nombre, cols_cat):
    # 'cat__ChestPain_asymptomatic' -> 'ChestPain'; 'num__Age' -> 'Age'
    if nombre.startswith("num__"):
        return nombre[5:]
    resto = nombre[5:]
    candidatas = [c for c in cols_cat if resto.startswith(c + "_")]
    return max(candidatas, key=len) if candidatas else resto

raices, pred_test = [], []
for b in range(200):
    idx = rng.integers(0, n, n)
    arbol = DecisionTreeClassifier(random_state=b).fit(H_tr[idx], yh_tr.values[idx])
    raices.append(variable_original(nombres_h[arbol.tree_.feature[0]], cat_h))
    pred_test.append(arbol.predict(H_te))

print("Variable de la raíz en 200 árboles bootstrap:")
print(pd.Series(raices).value_counts(normalize=True).round(3).to_string())

P = np.array(pred_test)
pares = [(i, j) for i in range(0, 200, 2) for j in range(1, 200, 2) if i != j]
print("\\nProporción de pacientes del test en que dos árboles discrepan:", round(np.mean([np.mean(P[i] != P[j]) for i, j in pares]), 3))
""")
md("""
Los árboles **no se ponen de acuerdo ni en la primera pregunta**: la raíz es `Thal` en ≈ la mitad de los casos (48.5 %), pero también `ChestPain` (30 %) o `Ca` (20 %), y dos árboles cualquiera se contradicen en ≈ 1 de cada 4 pacientes (26 %). Ese es el problema (alta varianza) que bagging ataca: si cada árbol se equivoca «en lugares distintos», promediar los errores los cancela en parte.

### ✍️ Ejercicio A2
Repite el experimento con árboles de profundidad máxima 3. ¿Se estabiliza la variable de la raíz? ¿Disminuye el desacuerdo entre árboles? ¿Cuál es el costo de limitar la profundidad?
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(RS)
raices3, pred3 = [], []
for b in range(200):
    idx = rng.integers(0, n, n)
    arbol = DecisionTreeClassifier(max_depth=3, random_state=b).fit(H_tr[idx], yh_tr.values[idx])
    raices3.append(variable_original(nombres_h[arbol.tree_.feature[0]], cat_h))
    pred3.append(arbol.predict(H_te))
print(pd.Series(raices3).value_counts(normalize=True).round(3).to_string())
P3 = np.array(pred3)
print("Desacuerdo medio:", round(np.mean([np.mean(P3[i] != P3[j]) for i, j in pares]), 3))
""", "La raíz es **exactamente la misma** (con la misma muestra, la primera pregunta no depende de la profundidad máxima: Thal 48.5 %, ChestPain 30 %, Ca 20 %), pero el desacuerdo entre árboles baja de ≈ 26 % a ≈ 22 %: un árbol más simple es algo más estable. El costo es **sesgo**: un árbol de profundidad 3 no puede capturar relaciones complejas. Bagging permite usar árboles profundos (poco sesgo) y reducir la varianza con el promedio.")

md("""
### A3 · Bagging desde cero, con OOB (20 min)
Algoritmo de la diapositiva 8: (1) muestrear B muestras bootstrap, (2) ajustar un árbol en cada una, (3) obtener B respuestas para x, (4) agregar. Además registramos qué filas quedaron **fuera** de cada muestra para calcular el error OOB: cada fila solo se predice con los árboles que **no** la vieron.
""")
code("""
def bagging_scratch(X, y, B=100, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(y)
    arboles, fuera = [], []
    for b in range(B):
        idx = rng.integers(0, n, n)                       # bootstrap: n extracciones con reemplazo
        arboles.append(DecisionTreeClassifier(random_state=b).fit(X[idx], y[idx]))
        mask = np.ones(n, dtype=bool)
        mask[idx] = False                                 # True = la fila quedó fuera (OOB) de este árbol
        fuera.append(mask)
    return arboles, np.array(fuera)


def proba_bagging(arboles, X):
    return np.mean([a.predict_proba(X)[:, 1] for a in arboles], axis=0)     # promedio de probabilidades


def proba_oob(arboles, fuera, X):
    P = np.array([a.predict_proba(X)[:, 1] for a in arboles])               # (B, n)
    return (P * fuera).sum(axis=0) / fuera.sum(axis=0)                      # solo árboles que NO vieron la fila


y_tr = yh_tr.values
arboles, fuera = bagging_scratch(H_tr, y_tr, B=100)
p_test = proba_bagging(arboles, H_te)
p_oob = proba_oob(arboles, fuera, H_tr)

print("Árbol solo (el primero)   | exactitud test:", round(accuracy_score(yh_te, arboles[0].predict(H_te)), 3),
      "| AUC:", round(roc_auc_score(yh_te, arboles[0].predict_proba(H_te)[:, 1]), 3))
print("Bagging a mano (B = 100)  | exactitud test:", round(accuracy_score(yh_te, p_test > 0.5), 3),
      "| AUC:", round(roc_auc_score(yh_te, p_test), 3))
print("Exactitud OOB a mano      :", round(accuracy_score(y_tr, p_oob > 0.5), 3))
print("Cada fila quedó fuera de ≈", round(fuera.sum(axis=0).mean(), 1), "de 100 árboles (≈ 37)")

bag_sk = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True, random_state=0).fit(H_tr, y_tr)
print("\\nBaggingClassifier sklearn | exactitud test:", round(bag_sk.score(H_te, yh_te), 3),
      "| AUC:", round(roc_auc_score(yh_te, bag_sk.predict_proba(H_te)[:, 1]), 3), "| OOB:", round(bag_sk.oob_score_, 3))
""")
md("""
Nuestro bagging «de juguete» y el de scikit-learn dan prácticamente lo mismo (las diferencias vienen de las semillas, es decir, de qué muestras salieron). Observa también que **un solo árbol** queda muy por debajo del promedio de 100 (AUC ≈ 0.73 contra ≈ 0.92 en esta corrida), y que el OOB (≈ 0.80, calculado sin tocar el test) es una estimación realista del desempeño.

Una nota sobre la votación. La diapositiva 13 cuenta votos (3 de 5 = 0.60); scikit-learn **promedia las probabilidades de cada árbol**. Con hojas puras (`min_samples_leaf=1`) ambas cosas coinciden, con hojas más grandes ya no:
""")
code("""
for hoja in (1, 5):
    rf_v = RandomForestClassifier(n_estimators=100, min_samples_leaf=hoja, random_state=0).fit(H_tr, y_tr)
    votos = np.mean([e.predict(H_te) for e in rf_v.estimators_], axis=0)          # fracción de árboles que votan «Yes»
    prob = rf_v.predict_proba(H_te)[:, 1]
    print(f"min_samples_leaf = {hoja}: máxima diferencia |votos − predict_proba| = {np.abs(votos - prob).max():.3f}")
""")
md("""
### ✍️ Ejercicio A3
Extiende `bagging_scratch` a **regresión** (usa `DecisionTreeRegressor` y promedia las predicciones) y aplícalo a Ames (`A_fit`, `yf`) con B = 50. Compara el RMSE en `A_val` de un solo árbol sin podar y del bagging. ¿Cuánto baja el error?
""")
code("""
# tu código aquí
""")
solution("""
def bagging_reg_scratch(X, y, B=50, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(y)
    arboles = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        arboles.append(DecisionTreeRegressor(random_state=b).fit(X[idx], y[idx]))
    return arboles

arboles_r = bagging_reg_scratch(A_fit, yf, B=50)
pred_arbol = arboles_r[0].predict(A_val)
pred_bag = np.mean([a.predict(A_val) for a in arboles_r], axis=0)
solo = DecisionTreeRegressor(random_state=0).fit(A_fit, yf)
print("Árbol sin podar (muestra completa):", f"{rmse(yv, solo.predict(A_val)):,.0f}")
print("Un árbol bootstrap                :", f"{rmse(yv, pred_arbol):,.0f}")
print("Bagging de 50 árboles             :", f"{rmse(yv, pred_bag):,.0f}")
""", "Un árbol sin podar queda en ≈ 35 mil USD de RMSE y un árbol entrenado con una muestra bootstrap es peor (≈ 44 mil: ve solo ≈ 63 % de las filas únicas). El promedio de 50 árboles baja a ≈ 24.5 mil: una reducción de ≈ 30 % respecto del árbol completo (y de ≈ 44 % respecto de un árbol bootstrap) **sin cambiar el tipo de modelo**, solo agregando.")

# ================================================================== PARTE B
md("""
## Parte B · Bagging vs Random Forest, y por qué promediar funciona

### B1 · Los modelos base del profesor (15 min)
El notebook del profesor entrena un `BaggingClassifier` y un `RandomForestClassifier` (120 árboles, `oob_score=True`) dentro de un pipeline con el preprocesamiento anterior, y los compara en el test. Reproducimos su configuración (con `class_weight='balanced'` en el bosque, como él) y añadimos un bosque **sin** `class_weight` para comparar en igualdad de condiciones.
""")
code("""
def modelos_heart():
    return {
        "Bagging (prof.)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), n_estimators=120,
                                             bootstrap=True, oob_score=True, random_state=RS, n_jobs=-1),
        "Random Forest (prof., balanced)": RandomForestClassifier(n_estimators=120, max_features="sqrt", bootstrap=True,
                                                                  oob_score=True, class_weight="balanced", random_state=RS, n_jobs=-1),
        "Random Forest sin class_weight": RandomForestClassifier(n_estimators=120, max_features="sqrt", oob_score=True,
                                                                 random_state=RS, n_jobs=-1),
    }

filas = []
for nombre, modelo in modelos_heart().items():
    pipe = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", modelo)])
    t0 = perf_counter(); pipe.fit(Xh_tr, yh_tr); dt = perf_counter() - t0
    p = pipe.predict(Xh_te)
    filas.append({"Modelo": nombre, "Accuracy": accuracy_score(yh_te, p), "Recall": recall_score(yh_te, p), "F1": f1_score(yh_te, p),
                  "ROC AUC": roc_auc_score(yh_te, pipe.predict_proba(Xh_te)[:, 1]), "OOB (exactitud)": modelo.oob_score_,
                  "Tiempo (s)": dt})
display(pd.DataFrame(filas).set_index("Modelo").round(3))
""")
md("""
En el test de 76 pacientes el bosque supera al bagging (en nuestra corrida: accuracy 0.816 vs 0.842 y AUC 0.901 vs 0.924; en la clase, 0.81 vs 0.86 y 0.90 vs 0.92). Dos cosas para leer con cuidado: (1) **un paciente = 1.3 puntos de exactitud**, así que diferencias de uno o dos pacientes no son evidencia (ver D4 con validación cruzada repetida); (2) el bosque del profesor usa `class_weight="balanced"` y el bagging no: la comparación no es en igualdad de condiciones. La fila sin `class_weight` lo separa: el bosque conserva su ventaja (AUC 0.917), así que `class_weight` no era lo decisivo en un conjunto tan equilibrado. Sobre el tiempo: la primera fila siempre incluye costos de «puesta en marcha» (en clase: 1.2 s vs 0.1 s); si ejecutas la celda de nuevo los tiempos se igualan.

En **regresión** el notebook compara el bagging con un `RandomForestRegressor(max_features=1.0)`. Pero con `max_features=1.0` (todas las variables en cada nodo) el bosque **es** un bagging: no hay selección de variables. Lo comprobamos en Ames:
""")
code("""
bag_r = Pipeline([("prep", make_preprocessor(Xa_tr)), ("model", BaggingRegressor(DecisionTreeRegressor(random_state=RS), n_estimators=120,
                                                                                  oob_score=True, random_state=RS, n_jobs=-1))])
rf_r_todas = Pipeline([("prep", make_preprocessor(Xa_tr)), ("model", RandomForestRegressor(n_estimators=120, max_features=1.0,
                                                                                            oob_score=True, random_state=RS, n_jobs=-1))])
rf_r_tercio = Pipeline([("prep", make_preprocessor(Xa_tr)), ("model", RandomForestRegressor(n_estimators=120, max_features=1 / 3,
                                                                                             oob_score=True, random_state=RS, n_jobs=-1))])
filas = []
for nombre, pipe in {"Bagging": bag_r, "RF max_features=1.0 (prof.)": rf_r_todas, "RF max_features=1/3": rf_r_tercio}.items():
    t0 = perf_counter(); pipe.fit(Xa_tr, ya_tr); dt = perf_counter() - t0
    pred = pipe.predict(Xa_te)
    filas.append({"Modelo": nombre, "RMSE": rmse(ya_te, pred), "MAE": mean_absolute_error(ya_te, pred), "R2": r2_score(ya_te, pred),
                  "OOB R2": pipe.named_steps["model"].oob_score_, "Tiempo (s)": dt})
display(pd.DataFrame(filas).set_index("Modelo").style.format({"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "R2": "{:.3f}", "OOB R2": "{:.3f}", "Tiempo (s)": "{:.1f}"}))
""")
md("""
El «bosque» del profesor (`max_features=1.0`) y el bagging son el mismo algoritmo y dan resultados casi idénticos (RMSE de test ≈ 29.1 mil y ≈ 29.3 mil USD); la ganancia real (≈ 26.7 mil) del Random Forest aparece al **reducir** `max_features` (fila 3). Esto es justo lo que miramos a continuación.

### ✍️ Ejercicio B1
Con los pipelines de arriba, calcula el RMSE de test del bagging **por tipo de casa**: ¿en qué rango de precio (≤ 150 mil, 150–300 mil, > 300 mil USD) se equivoca más el bosque `max_features=1/3`? ¿Qué te dice sobre dónde está el error?
""")
code("""
# tu código aquí
""")
solution("""
pred = rf_r_tercio.predict(Xa_te)
tramo = pd.cut(ya_te, [0, 150_000, 300_000, np.inf], labels=["≤ 150 mil", "150–300 mil", "> 300 mil"])
res = pd.DataFrame({"y": ya_te.values, "pred": pred, "tramo": tramo.values})
tabla = res.groupby("tramo").apply(lambda d: pd.Series({"n": len(d), "RMSE": rmse(d.y, d.pred), "error relativo medio": np.mean(np.abs(d.y - d.pred) / d.y)}))
display(tabla.round(2))
""", "El RMSE en dólares se dispara en las casas caras (≈ 59 mil en las > 300 mil, que son solo 63 y heterogéneas; contra ≈ 16 mil y ≈ 22 mil en los otros tramos); en error **relativo** las baratas y las caras son parecidas (≈ 11 %) y las medias son las mejor predichas (≈ 7 %). La métrica en dólares está dominada por la cola de precios altos. (Los árboles, además, no extrapolan: ver D2.)")

md("""
### B2 · Verificar la fórmula ρσ² + (1 − ρ)σ²/B (15 min)
La diapositiva 9 dice que la varianza del promedio de B árboles con varianza σ² y correlación ρ es

$$\\mathrm{Var}(\\bar T_B) = \\rho\\,\\sigma^2 + \\frac{1-\\rho}{B}\\,\\sigma^2$$

Con B → ∞ el segundo término desaparece, pero **ρσ² se queda**: «aumentar B no elimina ese componente». En clase se dijo que ρ «no es algo que uno mida». Sí se puede estimar. Lo hacemos así (como en el libro *Elements of Statistical Learning*, cap. 15): repetimos K = 30 veces «entrenar un bosque de 50 árboles con una submuestra distinta del train» y miramos, en cada casa de la validación, cuánto varía la predicción de un árbol (σ²), cuánto se parecen dos árboles distintos (ρ) y cuánto varía el promedio de B árboles.
""")
code("""
def estimar_rho(max_features, K=30, B=50, frac=0.5, semilla=0):
    rng = np.random.default_rng(semilla)
    m = int(len(A_fit) * frac)
    P = np.empty((K, B, len(A_val)))
    for k in range(K):
        idx = rng.choice(len(A_fit), m, replace=False)                      # otra muestra de entrenamiento
        rf = RandomForestRegressor(n_estimators=B, max_features=max_features, n_jobs=-1,
                                   random_state=int(rng.integers(1e9))).fit(A_fit[idx], yf[idx])
        P[k] = np.array([e.predict(A_val) for e in rf.estimators_])
    C = P - P.mean(axis=0, keepdims=True)
    cov = np.einsum("kbx,kcx->bcx", C, C) / (K - 1)                         # covarianza entre árboles b y c, por casa
    diag = np.einsum("bbx->bx", cov)
    sigma2 = diag.mean()
    off = (cov.sum(axis=(0, 1)) - diag.sum(axis=0)) / (B * (B - 1))
    rho = off.mean() / sigma2
    medido = {b: P[:, :b, :].mean(axis=1).var(axis=0, ddof=1).mean() for b in (1, 5, 10, 50)}
    return sigma2, rho, medido

t0 = perf_counter()
filas = []
for nombre, mf in [("Bagging (todas las variables)", 1.0), ("Random Forest (1/3)", 1 / 3)]:
    s2, rho, medido = estimar_rho(mf)
    for b, v in medido.items():
        filas.append({"Modelo": nombre, "ρ̂": rho, "σ̂ (un árbol, USD)": np.sqrt(s2), "B": b,
                      "desvío medido": np.sqrt(v), "desvío según la fórmula": np.sqrt(rho * s2 + (1 - rho) * s2 / b),
                      "piso √(ρσ²)": np.sqrt(rho * s2)})
display(pd.DataFrame(filas).round({"ρ̂": 3, "σ̂ (un árbol, USD)": 0, "desvío medido": 0, "desvío según la fórmula": 0, "piso √(ρσ²)": 0}))
print("Tiempo:", round(perf_counter() - t0, 1), "s")
""")
md("""
Tres lecturas:
1. **La fórmula funciona**: el desvío medido del promedio coincide con el de la fórmula casi al dólar.
2. **ρ es pequeño pero decide el piso**: con B = 1 el desvío es ≈ 37–39 mil USD; con B = 50 baja a ≈ 9.4 mil (bagging) y ≈ 7.8 mil (bosque), pero el piso √(ρσ²) (≈ 7.9 mil y ≈ 5.7 mil) no se puede cruzar con más árboles.
3. **El Random Forest tiene menor ρ** (≈ 0.022 contra ≈ 0.046 en esta corrida; la guía, con K = 60, da 0.024 y 0.049) aunque cada árbol individual sea algo más variable (σ mayor): eso es lo que logra `max_features`. Por eso el bosque alcanza un piso más bajo.

Un detalle de precisión: la ρ de la fórmula es la correlación entre las **predicciones** de dos árboles (al repetir la muestra y el azar del algoritmo), no entre sus «errores» a secas; en clase se usó la palabra «errores».

### ✍️ Ejercicio B2
Usa la fórmula con σ = 36 mil y ρ = 0.05: ¿cuál es el desvío con B = 1, 10, 100 y B → ∞? Luego, ¿cuántos árboles hacen falta para que el término (1 − ρ)σ²/B sea la décima parte de ρσ²? (Responde con lápiz y luego verifica con código.)
""")
code("""
# tu código aquí
""")
solution("""
sigma, rho = 36_000, 0.05
for B in (1, 10, 100):
    print(f"B = {B:3d}: desvío = {np.sqrt(rho * sigma**2 + (1 - rho) * sigma**2 / B):,.0f}")
print(f"B → ∞  : desvío = {np.sqrt(rho) * sigma:,.0f}")
# (1-ρ)σ²/B = 0.1·ρσ²  ->  B = (1-ρ)/(0.1·ρ)
print("Árboles para que el 2.º término sea 10 % del 1.º:", (1 - rho) / (0.1 * rho))
""", "Desvíos ≈ 36.0 mil, 13.7 mil, 8.8 mil y 8.05 mil (piso). Hacen falta B = (1 − ρ)/(0.1ρ) = 190 árboles. Con ρ más bajo, el piso baja, pero hacen falta *más* árboles para acercarse a él.")

md("""
### B3 · `max_features`: por nodo, no por árbol (15 min)
En clase se explicó el Random Forest así: «cada árbol va a tener una muestra de las variables… uno selecciona la 1, 5 y 8, el otro la 7, 15 y 20». Las diapositivas (nota de la diapositiva 10) dicen lo correcto: **la selección ocurre en cada nodo, no una sola vez por árbol**. Son dos algoritmos distintos:
- *Random subspace* (por árbol): `BaggingRegressor(max_features=0.1)` — cada árbol solo ve el 10 % de las columnas durante toda su vida.
- *Random Forest* (por nodo): `RandomForestRegressor(max_features=0.1)` — en cada nodo se sortean candidatas nuevas.

Comparemos en Ames con la misma fracción:
""")
code("""
por_arbol = BaggingRegressor(DecisionTreeRegressor(), n_estimators=300, max_features=0.1, n_jobs=-1, random_state=RS).fit(A_fit, yf)
por_nodo = RandomForestRegressor(n_estimators=300, max_features=0.1, n_jobs=-1, random_state=RS).fit(A_fit, yf)
print(f"Por árbol (random subspace, 10 %): RMSE validación = {rmse(yv, por_arbol.predict(A_val)):,.0f}")
print(f"Por nodo  (Random Forest, 10 %)  : RMSE validación = {rmse(yv, por_nodo.predict(A_val)):,.0f}")
""")
md("""
Con la misma fracción de variables, el bosque (por nodo) es mucho mejor: con la selección por árbol, un árbol al que le tocaron columnas malas es malo **para siempre**; por nodo, cada pregunta tiene otra oportunidad.

Ahora el barrido de `max_features` en Ames y en Heart. El profesor mencionó que «hay una recomendación de que `max_features` sea igual a 1, y yo lo he probado y en muchos casos entrega modelos muchísimo mejores». Pongámoslo a prueba:
""")
code("""
fracciones = [0.01, 0.02, 0.05, 0.10, 0.20, 1 / 3, 0.50, 1.0]
filas = []
for f in fracciones:
    rf = RandomForestRegressor(n_estimators=300, max_features=f, oob_score=True, n_jobs=-1, random_state=RS).fit(A_fit, yf)
    filas.append({"max_features (fracción)": round(f, 3), "columnas por nodo": max(1, int(f * A_fit.shape[1])),
                  "RMSE validación": rmse(yv, rf.predict(A_val)), "RMSE OOB": rmse(yf, rf.oob_prediction_)})
tabla_mf = pd.DataFrame(filas)
display(tabla_mf.round({"RMSE validación": 0, "RMSE OOB": 0, "max_features (fracción)": 3}))

p_h = H_tr.shape[1]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
filas = []
for m in [1, 2, 3, 4, 6, 9, 13, p_h]:
    pipe = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", RandomForestClassifier(n_estimators=300, max_features=m, n_jobs=-1, random_state=RS))])
    s = cross_val_score(pipe, Xh_tr, yh_tr, cv=cv, scoring="roc_auc")
    filas.append({"max_features (columnas)": m, "AUC CV 5×3": s.mean(), "desvío": s.std()})
display(pd.DataFrame(filas).round(3))
""")
md("""
Resultado: **el consejo depende de los datos**. En Heart (18 columnas tras el one-hot) `max_features=1` tiene la mejor AUC (≈ 0.905), pero la diferencia con `max_features=3` (≈ 0.902) está muy dentro del desvío (≈ 0.03): la curva es casi plana. En Ames (341 columnas) `max_features=1` (o el 1 %, ≈ 3 columnas) es lo **peor** que probamos (RMSE ≈ 27.7 mil frente a ≈ 22.5 mil con el 10 %). La regla útil: `max_features` es un hiperparámetro que se **valida**, no un valor mágico. El óptimo aquí está en torno a 10–20 % de las columnas, no en `1.0` (bagging) ni en `1`.

Atención al conteo: con el one-hot, «13 variables» del profesor son **18 columnas** para el árbol, y `max_features="sqrt"` calcula √18 ≈ 4 (no √13 ≈ 3 como se dijo en clase), porque el bosque ve las columnas ya codificadas.

### ✍️ Ejercicio B3
(a) Calcula cuántas columnas por nodo usa `max_features="sqrt"` y `"log2"` en Ames (341 columnas). (b) Entrena un bosque con cada uno y compara su RMSE de validación con el de `max_features=1/3` (el valor por defecto antiguo de sklearn para regresión). ¿Cuál queda mejor?
""")
code("""
# tu código aquí
""")
solution("""
p = A_fit.shape[1]
print("sqrt:", int(np.sqrt(p)), "columnas | log2:", int(np.log2(p)), "columnas | 1/3:", p // 3, "columnas")
for mf in ("sqrt", "log2", 1 / 3):
    rf = RandomForestRegressor(n_estimators=300, max_features=mf, n_jobs=-1, random_state=RS).fit(A_fit, yf)
    print(f"max_features = {mf!s:>6}: RMSE validación = {rmse(yv, rf.predict(A_val)):,.0f}")
""", "Con p = 341: √p ≈ 18 columnas, log₂p ≈ 8 columnas y p/3 ≈ 113. `sqrt` (18 columnas, ≈ 5 %) da ≈ 23.3 mil, casi igual a `1/3` (23.1 mil); `log2` (8 columnas, ≈ 2 %) da ≈ 25.6 mil, claramente peor: demasiado pocas candidatas por nodo. El óptimo está entre el 5 y el 20 % de las columnas.")

md("""
### B4 · ¿Cuántos árboles? Curvas de error (10 min)
El profesor lo resumió así: más árboles estabilizan el resultado, pero cuestan más. Veamos en Ames cuánto ayuda cada árbol extra: calculamos el RMSE de validación del promedio de los primeros *k* árboles (sin reentrenar).
""")
code("""
B = 300
arbol_solo = DecisionTreeRegressor(random_state=RS).fit(A_fit, yf)
bag = BaggingRegressor(DecisionTreeRegressor(), n_estimators=B, n_jobs=-1, random_state=RS).fit(A_fit, yf)
rf = RandomForestRegressor(n_estimators=B, max_features=1 / 3, n_jobs=-1, random_state=RS).fit(A_fit, yf)

k = np.arange(1, B + 1)[:, None]
p_bag = np.cumsum([e.predict(A_val[:, f]) for e, f in zip(bag.estimators_, bag.estimators_features_)], axis=0) / k
p_rf = np.cumsum([e.predict(A_val) for e in rf.estimators_], axis=0) / k
curva_bag = [rmse(yv, p) for p in p_bag]
curva_rf = [rmse(yv, p) for p in p_rf]

print(f"Un árbol sin podar              : {rmse(yv, arbol_solo.predict(A_val)):,.0f}")
for kk in (1, 10, 50, 100, 300):
    print(f"{kk:3d} árboles | bagging {curva_bag[kk - 1]:>7,.0f} | Random Forest {curva_rf[kk - 1]:>7,.0f}")

plt.figure(figsize=(8.5, 3.8))
plt.plot(range(1, B + 1), curva_bag, label="Bagging", color="#2563EB", lw=2)
plt.plot(range(1, B + 1), curva_rf, label="Random Forest (1/3)", color="#0D9488", lw=2)
plt.axhline(rmse(yv, arbol_solo.predict(A_val)), color="gray", ls=":", label="un árbol")
plt.ylim(15000, 50000); plt.xlabel("Número de árboles"); plt.ylabel("RMSE en validación (USD)")
plt.legend(); plt.title("Más árboles: casi todo el beneficio llega pronto"); plt.show()
""")
md("""
Lectura: el error cae muy rápido en los primeros 10–30 árboles y luego se **aplana** (ambos modelos quedan a menos de un 2 % de su valor final antes de los 40 árboles). Más árboles no producen sobreajuste, pero sí cuestan tiempo y memoria: pasar de 100 a 300 árboles casi no cambia el error. Además la curva del Random Forest queda **por debajo** de la del bagging: no por tener más árboles, sino por su menor ρ (B2).

### ✍️ Ejercicio B4
(a) ¿A partir de cuántos árboles el RMSE de cada modelo queda a menos de un 1 % de su valor con 300 árboles? (b) Mide el tiempo de ajuste del bosque con `n_jobs=1` y con `n_jobs=-1`. En clase se dijo que con 10 procesadores el tiempo baja «10 veces»; ¿qué aceleración obtienes realmente?
""")
code("""
# tu código aquí
""")
solution("""
def arboles_para(curva, tol=0.01):
    return next(i + 1 for i, v in enumerate(curva) if v <= curva[-1] * (1 + tol))

print("Árboles para quedar a < 1 % del final → bagging:", arboles_para(curva_bag), "| bosque:", arboles_para(curva_rf))

tiempos = {}
for nj in (1, -1):
    t0 = perf_counter()
    RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=nj, random_state=0).fit(A_fit, yf)
    tiempos[nj] = perf_counter() - t0
print({k: round(v, 2) for k, v in tiempos.items()}, "| aceleración:", round(tiempos[1] / tiempos[-1], 1), "x")
""", "(a) En la corrida de referencia, ≈ 40 árboles para el 1 % (42 el bagging, 40 el bosque) y ≈ 33 para el 2 %. (b) La aceleración real es **menor que el número de núcleos** (≈ 6.6× con 16 núcleos lógicos en nuestra máquina): hay costos de reparto del trabajo y de memoria, y no todo el proceso se paraleliza. «Entrenar en paralelo reduce el tiempo en N veces» es el límite ideal, no el resultado típico.")

# ================================================================== PARTE C
md("""
## Parte C · OOB, hiperparámetros y búsqueda

### C1 · OOB vs validación cruzada vs test (15 min)
El OOB es una **validación interna gratuita**: cada fila se predice solo con los árboles que no la vieron, así que se obtiene una estimación honesta con **un único ajuste**. La diapositiva 6 y el profesor insisten en que «no reemplaza siempre» a un test independiente. Comparemos las tres estimaciones en Heart (exactitud) y Ames (RMSE), con su costo.
""")
code("""
# Heart
t0 = perf_counter()
rf_h = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=RS).fit(H_tr, yh_tr)
t_oob = perf_counter() - t0
t0 = perf_counter()
pipe_h = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RS))])
cv_h = cross_validate(pipe_h, Xh_tr, yh_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), scoring="accuracy")
t_cv = perf_counter() - t0
print(f"Heart  | OOB {rf_h.oob_score_:.3f} ({t_oob:.1f} s)  | CV 5 bloques {cv_h['test_score'].mean():.3f} ± {cv_h['test_score'].std():.3f} ({t_cv:.1f} s)  | test {rf_h.score(H_te, yh_te):.3f}")

# Ames
t0 = perf_counter()
Xa_t, Xa_s = make_preprocessor(Xa_tr).fit(Xa_tr).transform(Xa_tr), None
pre_full = make_preprocessor(Xa_tr).fit(Xa_tr)
Xa_t, Xa_s = pre_full.transform(Xa_tr), pre_full.transform(Xa_te)
rf_a = RandomForestRegressor(n_estimators=300, max_features=1 / 3, oob_score=True, n_jobs=-1, random_state=RS).fit(Xa_t, ya_tr)
t_oob = perf_counter() - t0
t0 = perf_counter()
pipe_a = Pipeline([("prep", make_preprocessor(Xa_tr)), ("model", RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS))])
cv_a = cross_validate(pipe_a, Xa_tr, ya_tr, cv=KFold(5, shuffle=True, random_state=RS), scoring="neg_root_mean_squared_error")
t_cv = perf_counter() - t0
print(f"Ames   | OOB {rmse(ya_tr, rf_a.oob_prediction_):,.0f} ({t_oob:.1f} s)  | CV 5 bloques {-cv_a['test_score'].mean():,.0f} ({t_cv:.1f} s)  | test {rmse(ya_te, rf_a.predict(Xa_s)):,.0f}")
""")
md("""
Las tres estimaciones están en el mismo rango, y el OOB cuesta **un ajuste** en vez de cinco. En Heart la exactitud OOB (≈ 0.81) y la de CV (≈ 0.80) coinciden, y el test (76 pacientes, ≈ 0.86) cae dentro del ruido de una muestra tan chica (el desvío entre bloques de CV es ≈ 0.05). Dos cautelas: el OOB usa solo ≈ 37 % de los árboles para cada fila (es levemente pesimista con pocos árboles), y **sirve para comparar configuraciones**, pero el informe final sigue necesitando un test que no se haya usado para elegir nada. Y una precisión sobre la clase: el OOB lo tienen los modelos con bootstrap; en boosting, el `GradientBoosting*` de scikit-learn con `subsample < 1` también entrega `oob_improvement_` (Clase 36).

### ✍️ Ejercicio C1
Calcula **a mano** el OOB del bosque `rf_h` usando `rf_h.estimators_samples_` (los índices de cada muestra bootstrap) y `predict_proba` de cada árbol, y comprueba que coincide con `rf_h.oob_score_`.
""")
code("""
# tu código aquí
""")
solution("""
n = len(yh_tr)
suma = np.zeros(n)
cuenta = np.zeros(n)
for arbol, muestra in zip(rf_h.estimators_, rf_h.estimators_samples_):
    fuera = np.ones(n, dtype=bool)
    fuera[muestra] = False
    suma[fuera] += arbol.predict_proba(H_tr[fuera])[:, 1]
    cuenta[fuera] += 1
p_oob = suma / cuenta
print("Exactitud OOB a mano:", round(accuracy_score(yh_tr, p_oob > 0.5), 4), "| rf_h.oob_score_:", round(rf_h.oob_score_, 4))
print("Máxima diferencia con oob_decision_function_:", np.abs(p_oob - rf_h.oob_decision_function_[:, 1]).max())
""", "Deben coincidir (diferencia ≈ 0). El OOB no es magia: es un promedio de probabilidades usando, para cada paciente, solo los árboles cuya muestra bootstrap no lo incluyó.")

md("""
### C2 · Complejidad del árbol, `class_weight` y `max_samples` (10 min)
En clase se dijo que, al promediar, «tus probabilidades de sobreajuste son menores… casi que lo eliminas del todo». Veamos qué es cierto. En Heart: exactitud en el **train**, OOB y test para distintos `min_samples_leaf`.
""")
code("""
filas = []
for hoja in (1, 3, 5, 10, 20):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=hoja, oob_score=True, n_jobs=-1, random_state=RS).fit(H_tr, yh_tr)
    filas.append({"min_samples_leaf": hoja, "train": rf.score(H_tr, yh_tr), "OOB": rf.oob_score_, "test": rf.score(H_te, yh_te),
                  "AUC test": roc_auc_score(yh_te, rf.predict_proba(H_te)[:, 1])})
display(pd.DataFrame(filas).round(3))

# max_samples: cada árbol ve menos filas -> más rápido y más diversidad
for ms in (None, 0.5):
    t0 = perf_counter()
    rf = RandomForestRegressor(n_estimators=300, max_features=1 / 3, max_samples=ms, n_jobs=-1, random_state=RS).fit(A_fit, yf)
    print(f"Ames, max_samples = {ms}: RMSE validación {rmse(yv, rf.predict(A_val)):,.0f} | tiempo {perf_counter() - t0:.1f} s")
""")
md("""
Con `min_samples_leaf=1` el bosque **memoriza el entrenamiento** (exactitud 1.000 en el train) y aun así generaliza razonablemente (≈ 0.85 en test): lo que promediar controla es la **varianza de la predicción**, no el ajuste al train. Así que «casi elimina el sobreajuste» es una exageración: el bosque sobreajusta el train igual que un árbol profundo; lo que no hace es que ese sobreajuste dañe tanto al test. Subir `min_samples_leaf` baja la exactitud de train, pero casi no cambia el test: la regla práctica es validar ambos extremos.

`max_samples=0.5` hace que cada árbol use la mitad de las filas: ajusta más rápido y, en este caso, casi sin costo en error.

### ✍️ Ejercicio C2
Heart: compara `class_weight=None`, `"balanced"` y `"balanced_subsample"` con el **recall** y la AUC en validación cruzada repetida (Heart está casi balanceado: 46 % positivos). ¿Cambia algo? ¿En qué situación sí importaría `class_weight`?
""")
code("""
# tu código aquí
""")
solution("""
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
for cw in (None, "balanced", "balanced_subsample"):
    pipe = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=RS))])
    r = cross_validate(pipe, Xh_tr, yh_tr, cv=cv, scoring=["recall", "roc_auc"])
    print(f"class_weight = {cw!s:>18}: recall {r['test_recall'].mean():.3f} | AUC {r['test_roc_auc'].mean():.3f}")
""", "Con 46 % de casos positivos casi no cambia nada (diferencias dentro del ruido). `class_weight` importa cuando la clase de interés es rara (p. ej. 2 % de fraudes): sin él, el bosque casi no aprende de ella. En ese caso conviene medir recall, precisión, F1 o PR-AUC, no exactitud.")

md("""
### C3 · `GridSearchCV` vs `RandomizedSearchCV` (15 min)
El profesor usó ambos para Bagging y Random Forest, y su conclusión fue: **el ganador debe justificarse con validación cruzada y test, no por el nombre del algoritmo**. Una grilla de 24 combinaciones con 5 bloques son 120 ajustes por modelo (así lo calculó en clase, y escala rápido). Aquí hacemos una versión corta para Random Forest en Heart.
""")
code("""
cv_cls = StratifiedKFold(5, shuffle=True, random_state=RS)
pipe_rf = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", RandomForestClassifier(random_state=RS, n_jobs=1))])

grilla = {"model__n_estimators": [100, 300], "model__max_features": ["sqrt", 0.7],
          "model__max_depth": [None, 6], "model__min_samples_leaf": [1, 3]}
t0 = perf_counter()
grid = GridSearchCV(pipe_rf, grilla, scoring="roc_auc", cv=cv_cls, n_jobs=-1, refit=True).fit(Xh_tr, yh_tr)
t_grid = perf_counter() - t0

distribuciones = {"model__n_estimators": [100, 200, 300, 500], "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
                  "model__max_depth": [None, 4, 6, 10], "model__min_samples_leaf": [1, 2, 3, 5], "model__min_samples_split": [2, 5, 10]}
t0 = perf_counter()
rand = RandomizedSearchCV(pipe_rf, distribuciones, n_iter=12, scoring="roc_auc", cv=cv_cls, random_state=RS, n_jobs=-1, refit=True).fit(Xh_tr, yh_tr)
t_rand = perf_counter() - t0

print(f"Grid ({len(grid.cv_results_['params'])} combinaciones, {t_grid:.1f} s)  : AUC CV = {grid.best_score_:.4f}")
print("   ", {k.replace("model__", ""): v for k, v in grid.best_params_.items()})
print(f"Random (12 combinaciones, {t_rand:.1f} s): AUC CV = {rand.best_score_:.4f}")
print("   ", {k.replace("model__", ""): v for k, v in rand.best_params_.items()})

base = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))])
print("RF base (sin buscar), AUC CV:", round(cross_val_score(base, Xh_tr, yh_tr, cv=cv_cls, scoring="roc_auc").mean(), 4))

mejor = grid if grid.best_score_ >= rand.best_score_ else rand            # elegir SOLO con la CV, como el profesor
print("\\nSeleccionado por CV:", "Grid" if mejor is grid else "Random", "| AUC en test (se mira UNA vez):",
      round(roc_auc_score(yh_te, mejor.predict_proba(Xh_te)[:, 1]), 3))
""")
md("""
Observa cuánto ayuda la búsqueda: poco. El RF base queda en AUC ≈ 0.899 y los «optimizados» en ≈ 0.911 (Grid) y ≈ 0.910 (Random): una mejora de ≈ 0.01, bastante menor que el desvío entre bloques (≈ 0.03), y Grid y Random llegan a lo mismo con configuraciones distintas. En el test, el AUC del mejor es ≈ 0.93. El profesor lo dijo en clase: la optimización «puede subir unos puntos» pero «jamás en la vida» convierte un 0.70 en 0.95. Y de nuevo: **incluye tu configuración base en la grilla**, si no la búsqueda puede elegir algo peor que lo que ya tenías.

Una trampa del notebook del profesor: en su búsqueda aleatoria, el **Bagging** incluye `estimator__max_features ∈ {None, 'sqrt', 0.7}`. Con `'sqrt'` o `0.7` dentro del *árbol*, se está sorteando variables **por nodo**: es decir, esa configuración de «Bagging» es en realidad un Random Forest. La comparación Bagging vs Random Forest se vuelve borrosa.

### ✍️ Ejercicio C3 (preguntas finales del profesor)
Responde con tus resultados: (1) ¿Cambió el ganador después de optimizar? (2) ¿Grid y Random encontraron configuraciones similares? (3) ¿Qué hiperparámetros explican mejor la diferencia entre Bagging y Random Forest? (4) ¿Qué costo adicional tuvo la mejora? Para (3), optimiza también un `BaggingClassifier` y compara con tu mejor bosque.
""")
code("""
# tu código aquí
""")
solution("""
pipe_bag = Pipeline([("prep", make_preprocessor(Xh_tr)), ("model", BaggingClassifier(DecisionTreeClassifier(), random_state=RS, n_jobs=1))])
grilla_bag = {"model__n_estimators": [100, 300], "model__max_samples": [0.7, 1.0],
              "model__estimator__max_depth": [None, 6], "model__estimator__min_samples_leaf": [1, 3]}
grid_bag = GridSearchCV(pipe_bag, grilla_bag, scoring="roc_auc", cv=cv_cls, n_jobs=-1).fit(Xh_tr, yh_tr)
print("Bagging optimizado : AUC CV =", round(grid_bag.best_score_, 4), {k.replace("model__", ""): v for k, v in grid_bag.best_params_.items()})
print("Random Forest      : AUC CV =", round(grid.best_score_, 4))
print("AUC test | Bagging:", round(roc_auc_score(yh_te, grid_bag.predict_proba(Xh_te)[:, 1]), 3),
      "| RF:", round(roc_auc_score(yh_te, grid.predict_proba(Xh_te)[:, 1]), 3))
""", "Las respuestas dependen de tu corrida, pero la lección es la del profesor: las diferencias son pequeñas y el ganador cambia con la métrica y la partición. Lo que más separa a los dos modelos no es una perilla del árbol, sino `max_features` (selección de variables por nodo), que el bagging no tiene. El costo de la mejora: decenas de ajustes extra para ganar, a lo sumo, unas centésimas de AUC.")

# ================================================================== PARTE D
md("""
## Parte D · Interpretar, límites y aplicaciones

### D1 · Importancia: impureza vs permutación (10 min)
La diapositiva 15 advierte que «la importancia por impureza puede inducir a error» y que, para interpretar, conviene la **permutación**. En clase se explicó que el problema es que los árboles «están con menos variables». La causa real es otra: la importancia por impureza (MDI, `feature_importances_`) se calcula **sobre el train** y favorece a las variables con muchos cortes posibles (continuas, muchas categorías), aunque no tengan relación con el objetivo. Lo demostramos agregando a Heart dos columnas de **puro ruido**:
""")
code("""
rng = np.random.default_rng(7)
Xh_r = X_h.copy()
Xh_r["Ruido_continuo"] = rng.normal(size=len(Xh_r))
Xh_r["Ruido_binario"] = rng.integers(0, 2, size=len(Xh_r))
r_tr, r_te, s_tr, s_te = train_test_split(Xh_r, y_h, test_size=0.25, stratify=y_h, random_state=RS)

pipe_r = Pipeline([("prep", make_preprocessor(r_tr)), ("model", RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1))]).fit(r_tr, s_tr)
cat_r = r_tr.select_dtypes(exclude=np.number).columns.tolist()
nombres = pipe_r.named_steps["prep"].get_feature_names_out()
mdi = pd.Series(pipe_r.named_steps["model"].feature_importances_, nombres)
mdi = mdi.groupby([variable_original(n, cat_r) for n in mdi.index]).sum()            # suma las dummies de cada variable
perm = permutation_importance(pipe_r, r_te, s_te, scoring="roc_auc", n_repeats=30, random_state=RS)
perm = pd.Series(perm.importances_mean, r_te.columns)

tabla = pd.DataFrame({"MDI (impureza, train)": mdi, "Permutación (caída de AUC, test)": perm})
tabla["puesto MDI"] = tabla.iloc[:, 0].rank(ascending=False).astype(int)
tabla["puesto permutación"] = tabla.iloc[:, 1].rank(ascending=False).astype(int)
display(tabla.sort_values("MDI (impureza, train)", ascending=False).round(3))
""")
md("""
La columna de **ruido continuo** queda en el puesto 7 de 15 por impureza (≈ 7 % de la importancia total): **por encima de variables reales** como `Chol`, `RestBP`, `ExAng`, `Slope` y `Sex`. En la **permutación** cae al puesto 10 y su efecto es ≈ 0 (caída de AUC ≈ 0.001): al desordenarla el modelo no pierde nada, porque no aportaba. Y por si faltaba: ninguna importancia es causalidad. La impureza mide cómo el modelo eligió sus preguntas; la permutación mide cuánto empeora la predicción si esa variable se desordena.

(Para las variables con *one-hot*, `permutation_importance` sobre el pipeline completo desordena la columna original, como hizo el profesor. Con 76 pacientes de test y pocas repeticiones esas cifras son ruidosas: aquí usamos 30 repeticiones.)

### ✍️ Ejercicio D1
Calcula la permutación sobre el **train** en vez del test. ¿Cambia la importancia del ruido continuo? ¿Por qué se recomienda calcularla sobre datos no vistos?
""")
code("""
# tu código aquí
""")
solution("""
perm_tr = permutation_importance(pipe_r, r_tr, s_tr, scoring="roc_auc", n_repeats=10, random_state=RS)
perm_tr = pd.Series(perm_tr.importances_mean, r_tr.columns)
print("Permutación en el TRAIN  :", perm_tr[["Ruido_continuo", "Ruido_binario"]].round(4).to_dict())
print("Permutación en el TEST   :", perm[["Ruido_continuo", "Ruido_binario"]].round(4).to_dict())
print("AUC del bosque en train vs test:", round(roc_auc_score(s_tr, pipe_r.predict_proba(r_tr)[:, 1]), 3), "vs", round(roc_auc_score(s_te, pipe_r.predict_proba(r_te)[:, 1]), 3))
""", "El ruido sigue sin importar (≈ 0.0005 en el train), pero el bosque tiene AUC = 1.0 en el train (contra ≈ 0.93 en el test): ha memorizado, y con el train **cualquier variable parece irrelevante o relevante por memorización**. Se calcula sobre datos no vistos porque lo que se quiere saber es cuánto usa el modelo cada variable para *generalizar*, no para recordar el entrenamiento.")

md("""
### D2 · Random Forest no extrapola (10 min)
Otra limitación de las diapositivas (y del profesor): en regresión el bosque **no predice fuera del rango observado**, porque promedia valores de hojas. Entrenamos con casas **pequeñas** (`Gr_Liv_Area` < 1.800 pies²) y predecimos casas **grandes** (> 2.400), usando solo esa variable. Comparamos con una regresión lineal.
""")
code("""
d = ames[["Gr_Liv_Area", "Sale_Price"]]
chica, grande = d[d.Gr_Liv_Area < 1800], d[d.Gr_Liv_Area > 2400]
rf_e = RandomForestRegressor(n_estimators=200, random_state=RS, n_jobs=-1).fit(chica[["Gr_Liv_Area"]], chica.Sale_Price)
lr_e = LinearRegression().fit(chica[["Gr_Liv_Area"]], chica.Sale_Price)

print("Precio máximo visto en el entrenamiento:", f"${chica.Sale_Price.max():,.0f}")
print("Mayor predicción del bosque en casas grandes:", f"${rf_e.predict(grande[['Gr_Liv_Area']]).max():,.0f}")
print("Mayor predicción de la regresión lineal     :", f"${lr_e.predict(grande[['Gr_Liv_Area']]).max():,.0f}")
print(f"RMSE en casas grandes | bosque: {rmse(grande.Sale_Price, rf_e.predict(grande[['Gr_Liv_Area']])):,.0f}"
      f" | lineal: {rmse(grande.Sale_Price, lr_e.predict(grande[['Gr_Liv_Area']])):,.0f}")

xs = np.linspace(500, 4000, 300).reshape(-1, 1)
plt.figure(figsize=(7, 3.8))
plt.scatter(d.Gr_Liv_Area, d.Sale_Price / 1000, s=4, color="lightgray", label="todas las casas")
plt.scatter(chica.Gr_Liv_Area, chica.Sale_Price / 1000, s=4, color="#0D9488", label="entrenamiento")
plt.plot(xs, rf_e.predict(xs) / 1000, color="#0D9488", lw=2.2, label="Random Forest")
plt.plot(xs, lr_e.predict(xs) / 1000, color="#DC2626", lw=2, label="regresión lineal")
plt.xlabel("Superficie habitable (pies²)"); plt.ylabel("Precio (miles USD)"); plt.legend(); plt.title("El bosque se queda «plano» fuera del rango"); plt.show()
""")
md("""
El bosque **se aplana**: tras el borde de los datos de entrenamiento repite el último valor aprendido y nunca supera el máximo visto. La regresión lineal sí continúa la tendencia (en las casas más grandes llega a predecir ≈ 647 mil, pasándose, pero su RMSE en ese tramo es menor: ≈ 120 mil contra ≈ 208 mil del bosque, que nunca pasa de ≈ 158 mil). No es un defecto de «mal ajuste»: es estructural. Si tu problema exige predecir valores nuevos fuera del rango (series con tendencia, precios en inflación…), un bosque necesita ayuda (diferenciar, modelar la tendencia aparte) o un modelo distinto.

### ✍️ Ejercicio D2
Repite con un `GradientBoostingRegressor` (Clase 36). ¿Extrapola mejor? ¿Qué te dice sobre «todos los modelos de árboles»?
""")
code("""
# tu código aquí
""")
solution("""
gb_e = GradientBoostingRegressor(n_estimators=200, random_state=RS).fit(chica[["Gr_Liv_Area"]], chica.Sale_Price)
print("Mayor predicción del boosting en casas grandes:", f"${gb_e.predict(grande[['Gr_Liv_Area']]).max():,.0f}",
      "| máximo de entrenamiento:", f"${chica.Sale_Price.max():,.0f}")
""", "Tampoco extrapola: el boosting es una **suma de árboles**, y cada árbol es constante por tramos, así que fuera del rango también queda plano. La limitación es de los árboles, no del bosque.")

md("""
### D3 · Imágenes: el ejemplo del profesor, con otro conjunto (15 min)
El tercer notebook del profesor (`Random Forest en imágenes.ipynb`) usa **CIFAR-10** (60.000 imágenes a color de 32×32) convertidas en vectores de 3.072 valores RGB, sin extracción de características ni optimización, para mostrar «la versatilidad» del modelo. **No se alcanzó a ver en clase** (el profesor la anunció «si nos alcanza el tiempo»). Ese notebook descarga ~170 MB de internet, así que aquí usamos un equivalente que viene con scikit-learn: **los dígitos manuscritos** (1.797 imágenes en escala de grises de 8×8 = 64 variables). La idea es idéntica: cada píxel es una variable. Si quieres replicar el original, ejecuta su notebook con conexión a internet.
""")
code("""
digitos = load_digits()
X_d, y_d = digitos.data / 16.0, digitos.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X_d, y_d, test_size=0.25, stratify=y_d, random_state=RS)

rf_d = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=RS).fit(Xd_tr, yd_tr)
print(f"OOB: {rf_d.oob_score_:.3f} | test: {rf_d.score(Xd_te, yd_te):.3f}  (entrenamiento {len(Xd_tr)}, test {len(Xd_te)})")

def desplazar(X, dx=1):
    im = X.reshape(-1, 8, 8)
    out = np.zeros_like(im)
    out[:, :, dx:] = im[:, :, :-dx]          # mueve cada imagen un píxel a la derecha
    return out.reshape(len(X), -1)

perm_px = np.random.default_rng(RS).permutation(64)           # reordenar los píxeles de forma fija
rf_mezcla = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS).fit(Xd_tr[:, perm_px], yd_tr)
print(f"Test con las imágenes desplazadas 1 píxel : {rf_d.score(desplazar(Xd_te), yd_te):.3f}")
print(f"Test con los píxeles reordenados al azar : {rf_mezcla.score(Xd_te[:, perm_px], yd_te):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].imshow(rf_d.feature_importances_.reshape(8, 8), cmap="Greens"); axes[0].set_title("Importancia (MDI) de cada píxel"); axes[0].grid(False)
mal = np.where(rf_d.predict(Xd_te) != yd_te)[0][:8]
axes[1].imshow(np.hstack([Xd_te[i].reshape(8, 8) for i in mal]), cmap="gray_r"); axes[1].grid(False); axes[1].axis("off")
axes[1].set_title("Dígitos mal clasificados: real → predicho: " + ", ".join(f"{yd_te[i]}→{rf_d.predict(Xd_te[[i]])[0]}" for i in mal), fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""
Resultado: el bosque clasifica bien estos dígitos (≈ 0.97 de exactitud) **tratando cada píxel como una columna independiente**. Pero la segunda prueba delata el límite: si **reordenamos los píxeles al azar** (destruyendo toda la estructura espacial) el bosque rinde **igual** (≈ 0.97), porque nunca usó la vecindad de los píxeles; y si **desplazamos las imágenes un solo píxel**, la exactitud se **cae a ≈ 0.50**, porque cada columna ahora significa otra cosa. Esto es lo que dice el cierre del notebook del profesor: el modelo «no entiende explícitamente la posición ni la estructura espacial». Las redes convolucionales sí lo hacen, y por eso son el estándar en imágenes (diapositiva 16: «en imágenes, texto o series muy largas, modelos especializados pueden resultar más adecuados»).

### ✍️ Ejercicio D3
Para el notebook original con CIFAR-10 el profesor no optimiza hiperparámetros. En los dígitos, prueba `max_features` ∈ {2, 8, 32, 64} (con 64 = bagging). ¿Cuál es mejor en OOB? ¿Confirma lo que viste en B3?
""")
code("""
# tu código aquí
""")
solution("""
for m in (2, 8, 32, 64):
    rf = RandomForestClassifier(n_estimators=300, max_features=m, oob_score=True, n_jobs=-1, random_state=RS).fit(Xd_tr, yd_tr)
    print(f"max_features = {m:2d}: OOB = {rf.oob_score_:.3f} | test = {rf.score(Xd_te, yd_te):.3f}")
""", "Con 8 columnas (= √64, el valor por defecto) el OOB es el mejor (0.976); 2 y 32 quedan cerca (0.970 y 0.975) y con 64 (bagging) baja a 0.943, con test 0.929. De nuevo: el valor intermedio le gana al máximo.")

md("""
### D4 · Bagging de modelos estables y comparación final (10 min)
El profesor señaló que el bagging «puede aplicarse a cualquier modelo», no solo árboles. Pero funciona donde el modelo base es **inestable**. Una regresión logística es estable (cambia poco si cambian algunas filas): hacer su bagging casi no ayuda. Cerramos con la comparación de todo lo visto, en validación cruzada repetida 5×3 sobre los 303 pacientes (la comparación honesta, a diferencia de un test de 76 pacientes) y en el test de Ames.
""")
code("""
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
modelos_h = {
    "Árbol": Pipeline([("prep", make_preprocessor(X_h)), ("model", DecisionTreeClassifier(random_state=RS))]),
    "Regresión logística": Pipeline([("prep", make_preprocessor(X_h, True)), ("model", LogisticRegression(max_iter=2000))]),
    "Bagging de logísticas": Pipeline([("prep", make_preprocessor(X_h, True)), ("model", BaggingClassifier(LogisticRegression(max_iter=2000), n_estimators=100, random_state=RS))]),
    "Bagging de árboles": Pipeline([("prep", make_preprocessor(X_h)), ("model", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, n_jobs=-1, random_state=RS))]),
    "Random Forest": Pipeline([("prep", make_preprocessor(X_h)), ("model", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS))]),
}
filas = []
for nombre, pipe in modelos_h.items():
    s = cross_val_score(pipe, X_h, y_h, cv=cv_rep, scoring="roc_auc")
    filas.append({"Modelo": nombre, "AUC media (CV 5×3)": s.mean(), "desvío": s.std()})
display(pd.DataFrame(filas).round(3))

filas = []
for nombre, modelo in {"Árbol": DecisionTreeRegressor(random_state=RS),
                       "Bagging": BaggingRegressor(DecisionTreeRegressor(), n_estimators=300, n_jobs=-1, random_state=RS),
                       "Random Forest (1/3)": RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS)}.items():
    t0 = perf_counter(); modelo.fit(Xa_t, ya_tr); dt = perf_counter() - t0
    filas.append({"Modelo": nombre, "RMSE train": rmse(ya_tr, modelo.predict(Xa_t)), "RMSE test": rmse(ya_te, modelo.predict(Xa_s)), "Tiempo (s)": dt})
display(pd.DataFrame(filas).style.format({"RMSE train": "${:,.0f}", "RMSE test": "${:,.0f}", "Tiempo (s)": "{:.1f}"}))
""")
md("""
Lectura: (1) pasar de **un árbol** (AUC ≈ 0.75) a un bagging o un bosque (≈ 0.89–0.91) es un salto enorme: eso es reducir varianza. (2) La **regresión logística (≈ 0.913) iguala o supera al bosque** en Heart, y su bagging no cambia nada (≈ 0.912): con 303 pacientes y relaciones casi lineales, «a problemas simples, modelos simples». (3) Entre Bagging de árboles (≈ 0.893) y Random Forest (≈ 0.906) la diferencia cabe dentro del desvío (≈ 0.04): no hay un ganador claro en Heart. (4) En Ames, con más datos y más estructura, sí se ordenan: árbol ≈ 46 mil, bagging ≈ 29 mil, Random Forest ≈ 27 mil USD.

Puente a las otras clases: el **árbol** (Clase 34) es el ladrillo; aquí lo promediamos en paralelo para bajar la varianza; en la **Clase 36 (Boosting)** los árboles se entrenan en secuencia para bajar el sesgo y el error.

### ✍️ Ejercicio D4
Calcula, en Ames, la **brecha** RMSE test / RMSE train de cada modelo. ¿Quién sobreajusta más el train? ¿Qué quiere decir esto para el árbol y para el bosque (¿son lo mismo «brecha grande» y «mal modelo»?)?
""")
code("""
# tu código aquí
""")
solution("""
res_a = {}
for nombre, modelo in {"Árbol": DecisionTreeRegressor(random_state=RS),
                       "Bagging": BaggingRegressor(DecisionTreeRegressor(), n_estimators=300, n_jobs=-1, random_state=RS),
                       "Random Forest (1/3)": RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=RS)}.items():
    modelo.fit(Xa_t, ya_tr)
    tr, te = rmse(ya_tr, modelo.predict(Xa_t)), rmse(ya_te, modelo.predict(Xa_s))
    res_a[nombre] = (tr, te, te / max(tr, 1))
print(pd.DataFrame(res_a, index=["RMSE train", "RMSE test", "test / train"]).T.round(1))
""", "El árbol sin podar tiene RMSE de train = 0 (memoriza) y el peor test; bagging y bosque tienen un train ≈ 10 mil y un test ≈ 27–29 mil: brecha de ≈ 3× pero, con diferencia, mejor test. Una brecha grande **no** significa mal modelo si el error de test es el menor: lo que decide es el test (o la CV), no la brecha.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene corregir

En las diapositivas, la clase y los notebooks del profesor hay imprecisiones que verificamos (la mayoría, con código, en este mismo notebook):

| Dónde | Qué dice | Qué es correcto |
|---|---|---|
| Clase | «Cada árbol del Random Forest tiene una muestra de las variables (la 1, 5 y 8; el otro la 7, 15 y 20)» | La selección es **por nodo**, no por árbol (nota de la diapositiva 10). Por árbol sería *random subspace* (`BaggingRegressor(max_features=…)`), claramente peor (B3) |
| Clase | «De 1.000 registros se toman 500 por muestra bootstrap» | Por defecto la muestra tiene **n** extracciones (≈ 63 % de filas únicas); el tamaño se cambia con `max_samples` (A1) |
| Clase | «Las muestras bootstrap son independientes» | Se parecen entre sí (comparten los datos); lo independiente son las extracciones dentro de cada muestra (A1) |
| Clase | «`max_features` = 1 entrega modelos muchísimo mejores» | Depende: en Heart es indistinguible (la curva es plana); en Ames es lo peor del barrido (B3) |
| Clase | «El conjunto casi elimina el sobreajuste» | El bosque sin límites alcanza exactitud 1.000 en train; lo que reduce es la varianza de la predicción, no el ajuste al train (C2) |
| Clase | «En paralelo con 10 procesadores se reduce el tiempo 10 veces (380 procesadores, 380 veces)» | Es el límite ideal; la aceleración real es menor (B4) |
| Clase | «Con max_features="sqrt" y 13 variables escoge 3» | Con el one-hot son 18 columnas: √18 ≈ 4 (B3) |
| Clase | «Los Random Forest aceptan categóricas sin problema» | En scikit-learn hay que codificarlas (one-hot): `RandomForestClassifier` sobre `Heart` sin procesar falla con `could not convert string to float` |
| Clase | «El OOB se pierde en boosting porque no usa bootstrap» | Gradient Boosting de scikit-learn con `subsample < 1` sí entrega `oob_improvement_` (C1) |
| Clase | «Importancia por impureza puede inducir a error porque hay menos variables» | La causa es el sesgo hacia variables con muchos cortes y que se calcula en el train; con ruido puro se ve (D1) |
| Clase | «El AUC siempre es mayor que 0.5» | Es 1 − AUC si inviertes las clases; un modelo que ordena al revés tiene AUC < 0.5 |
| Clase | «Hay un 54 % de no y un 56 % de sí» | 164/303 = 54.1 % y 139/303 = 45.9 %: es **46 %** |
| Diapositiva 9 / clase | «ρ = correlación entre errores»; «no se mide» | ρ es la correlación entre las **predicciones** de dos árboles bajo el azar de datos y algoritmo; sí se estima (B2). La fórmula supone varianzas iguales y ρ común (lo dice la nota) |
| Diapositiva 13 | Voto simple: 3 de 5 → 0.60 | scikit-learn promedia **probabilidades**; coincide con el voto solo con hojas puras (A3) |
| Notebook | RF de regresión con `max_features=1.0` como «Random Forest» | Con 1.0 es un bagging; el RF real exige `max_features < 1` (B1) |
| Notebook | `class_weight="balanced"` solo en el bosque | La comparación con Bagging no es en igualdad de condiciones (B1) |
| Notebook | `estimator__max_features ∈ {None, 'sqrt', 0.7}` en el Bagging aleatorio | Con `'sqrt'` o `0.7` el «Bagging» sortea variables por nodo: es un Random Forest (C3) |
| Notebook | Importancia por permutación sobre 76 pacientes con 5 repeticiones | Resultado ruidoso; usar más repeticiones o CV (D1) |
| Notebook | `DATA_DIR` fijo (`C:\\Users\\...\\Clase35-BastianGalasso`), `N_JOBS = 6`, `FAST_MODE = False` | Rutas y paralelismo propios de su equipo; en clase se recomendó `FAST_MODE=True` para equipos modestos |
| Notebook imágenes | CIFAR-10 (descarga ≈ 170 MB; `tar.extractall(filter="data")` necesita Python ≥ 3.12) | Se puede sustituir por los dígitos de scikit-learn (D3); no se vio en clase |
| Notebook imágenes | Titula «versatilidad» | Cierto en lo técnico, pero RF ignora la estructura espacial (D3) |

**Transcripción con ruido**: «busstrap/Bustrap/boost trap» = bootstrap, «Vallin/Bail/Bind/Baguí/Vagin» = bagging, «random forks/Rando ofertes/Randolph Farris» = Random Forest, «obb/OVE/auto bag/o o b» = OOB, «rock aoc/Rock AUC» = ROC AUC, «aquira» = accuracy, «ricó» = recall, «min sample lift/min sample left» = `min_samples_leaf`, «Mtry» = `mtry`, «her.xlsx» = `Heart.xlsx`, «Nameless/Ames housing» = Ames Housing. La clase termina antes de la parte de regresión y del ejemplo de imágenes.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (20 min, SIN mirar lo anterior)

Eres el analista y debes entregar un modelo de precio para Ames **sin tocar el test hasta el final**:

1. Con `Xa_tr` y `ya_tr` define un `Pipeline` (`make_preprocessor` + `RandomForestRegressor(n_estimators=200, oob_score=True)`).
2. Busca `max_features` entre `[0.05, 0.1, 0.2, 0.33, 0.5]` con `GridSearchCV` (3 bloques, `neg_root_mean_squared_error`). Incluye el valor por defecto de `RandomForestRegressor` (`1.0`).
3. Reporta el RMSE de CV, el **RMSE OOB** del mejor modelo y el RMSE de test **una sola vez**.
4. Compara con un `BaggingRegressor` (mismo número de árboles): ¿cuánto vale `max_features` en dólares?
5. Responde sin código: ¿qué dos cosas del material del profesor corregirías antes de dárselo a un cliente?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
pipe = Pipeline([("prep", make_preprocessor(Xa_tr)),
                 ("model", RandomForestRegressor(n_estimators=200, oob_score=True, n_jobs=1, random_state=RS))])
grid_a = GridSearchCV(pipe, {"model__max_features": [0.05, 0.1, 0.2, 1 / 3, 0.5, 1.0]},
                      scoring="neg_root_mean_squared_error", cv=KFold(3, shuffle=True, random_state=RS), n_jobs=-1).fit(Xa_tr, ya_tr)
print("Mejor max_features:", grid_a.best_params_["model__max_features"], "| RMSE CV:", f"{-grid_a.best_score_:,.0f}")
mejor_rf = grid_a.best_estimator_
print("RMSE OOB (train):", f"{rmse(ya_tr, mejor_rf.named_steps['model'].oob_prediction_):,.0f}")
print("RMSE test       :", f"{rmse(ya_te, mejor_rf.predict(Xa_te)):,.0f}")

bag_a = Pipeline([("prep", make_preprocessor(Xa_tr)), ("model", BaggingRegressor(DecisionTreeRegressor(), n_estimators=200, n_jobs=-1, random_state=RS))]).fit(Xa_tr, ya_tr)
print("RMSE test Bagging:", f"{rmse(ya_te, bag_a.predict(Xa_te)):,.0f}")
""", "En la corrida de referencia la CV de 3 bloques elige `max_features = 1/3` (RMSE de CV ≈ 26.9 mil, OOB ≈ 26.1 mil, test ≈ 26.6 mil) y el bagging queda en ≈ 29.2 mil: ≈ 2.7 mil dólares menos de error (en la validación de B3 el óptimo era ≈ 0.1; con otra partición cambia). El hiperparámetro que distingue a un Random Forest de un bagging vale unos miles de dólares por casa. Correcciones sugeridas antes de entregar: (1) el «Random Forest» de regresión del profesor con `max_features=1.0` es un bagging, y el «Bagging» de su búsqueda aleatoria puede convertirse en RF si sortea `max_features` dentro del árbol; (2) la importancia por impureza se calcula en el train y favorece variables con muchos cortes: usar permutación sobre datos no vistos (con más de 5 repeticiones). Otras válidas: `class_weight` solo en un modelo, comparar con un test de 76 pacientes.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué un bootstrap deja fuera ≈ 36.8 % de las filas? ¿Cómo se usan esas filas?
2. ¿Qué dice la fórmula ρσ² + (1 − ρ)σ²/B y por qué «más árboles» no basta?
3. ¿Qué agrega Random Forest a bagging, y dónde ocurre esa selección (por árbol, por nodo)?
4. ¿Cómo se entrenan en paralelo los árboles de un bosque y por qué no se puede en boosting?
5. ¿Qué se pierde al pasar de un árbol a un bosque y qué dos limitaciones aparecen en regresión y en interpretación?
6. ¿Por qué la importancia por impureza puede engañar y qué alternativa recomienda la diapositiva 15?

<details><summary><b>Respuestas</b></summary>

1. Cada fila no sale en una extracción con probabilidad 1 − 1/n; tras n extracciones queda fuera con (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368. Las filas OOB de cada árbol sirven de validación interna: cada fila se predice solo con los árboles que no la vieron y de ahí sale el error OOB (con un único ajuste).
2. La varianza del promedio de B árboles con varianza σ² y correlación ρ es ρσ² + (1 − ρ)σ²/B. Más árboles hacen desaparecer el segundo término, pero el primero (ρσ²) queda: si los árboles se parecen (ρ alto), agregar más no ayuda. Hay que bajar ρ.
3. Selección aleatoria de variables **en cada nodo** (`max_features`): los árboles se parecen menos (ρ menor) a costa de que cada uno sea algo peor (σ² algo mayor y un poco más de sesgo). Por árbol sería otra cosa (random subspace).
4. Cada árbol se entrena con su muestra, sin depender de los demás: se reparte entre procesadores (`n_jobs`). En boosting el árbol *m* necesita el resultado del *m − 1*: es secuencial.
5. Se pierde interpretabilidad (un árbol se lee; un bosque no) y costo de memoria/predicción. En regresión, el bosque no extrapola fuera del rango visto.
6. La impureza (MDI) se calcula con el train y favorece variables con muchos cortes posibles (ruido continuo incluido). Se prefiere la importancia por permutación sobre datos no vistos (o dependencia parcial / explicaciones locales); y ninguna es causalidad.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 36 · Boosting (los árboles ya no se promedian en paralelo: se entrenan en secuencia). Revisa `Guias/apuntes/C35_Ejercicios_Cuaderno_RandomForest.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
