"""Genera el notebook de práctica «Taller espejo» (Clase 42): réplica 1:1 de la estructura del Taller #3
sobre OTRA base (churn de telecomunicaciones, Clase 33). Está RESUELTO, con interpretación tras cada resultado.

Uso:  python -X utf8 notebook_c42.py   -> Guias/notebooks/C42_Practica_TallerEspejo.ipynb  (sin salidas)
Luego se ejecuta con nbclient (ver verificar_c42.py / instrucciones del informe) para dejarlo con salidas.
No lee ni usa los archivos del taller real (bank.csv / clientes_*.csv).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "C42_Practica_TallerEspejo.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def transfer(text):
    md("> **En el Taller 3 esto corresponde a…** " + text.strip())


from texto_c42 import T  # interpretaciones (se escriben tras ver los resultados)

# ====================================================================== ENCABEZADO
md(r"""
# Clase 42 · Práctica «Taller espejo»
**Diplomado Data Science** · Preparación para el Taller #3 (Prof. Eloy Alvarado) · 100 % resuelto, con interpretación

> **Qué es esto.** El Taller #3 (campaña bancaria, 60 puntos, entrega 21-oct-2026) es una evaluación **grupal y calificada**. Este notebook **no contiene nada del taller**: no usa `clientes_entrenamiento.csv` ni `clientes_prueba.csv` y no entrega resultados de esa base. Resuelve **la misma estructura, pregunta por pregunta**, sobre **otra base** (la de fuga de clientes de telefonía de la Clase 33), para que veas cada técnica funcionando de punta a punta, con la interpretación que se espera. Después de cada paso hay un recuadro **«En el Taller 3 esto corresponde a…»** que dice cómo trasladar el paso a `bank` (nombres de columnas y qué cambia). **Los números, tablas y conclusiones del taller los obtiene tu grupo.**

**Material de apoyo:** `Guias/C42_Guia_Taller3.pdf` (qué se pide, dónde estudiarlo, trampas, checklist) y los notebooks de las clases 30–41 en esta misma carpeta.

### Correspondencia con el enunciado
| Taller #3 (banco) | Esta práctica (churn) | Pts |
|---|---|---|
| 1(a) tarea, unidad, respuesta, variable que no se puede usar (`duracion`) | Parte 1(a); análogo: `clv` | 3 |
| 1(b) tipos, faltantes, `unknown`, duplicados, balance; negativos y código −1 | Parte 1(b) | 4 |
| 1(c) `contactado_antes`, `log_contactos_previos`, one-hot, escalamiento | Parte 1(c): `sin_sms`, `log_frec_sms`, one-hot, escalamiento | 5 |
| 2(a) logística todas vs. 10 mejores por F de ANOVA | Parte 2(a) | 5 |
| 2(b) seis modelos con log loss, Brier, AUC OOF | Parte 2(b) | 11 |
| 2(c) interpretación | Parte 2(c) | 8 |
| 3(a) umbral por costo FP + 5·FN | Parte 3(a) | 4 |
| 3(b) prueba: métricas, ROC, matrices en t\* y 0.50 | Parte 3(b) | 6 |
| 4(a) PCA; 4(b) K-means k = 2, 3, 4 | Parte 4 | 3 + 5 |
| 5(a) Mahalanobis² y corte χ²; 5(b) interpretación | Parte 5 | 3 + 3 |

### Qué es igual y qué cambia respecto del taller
- **Igual**: semilla 2026, clase positiva = 1, validación cruzada de 5 pliegues **con una columna `pliegue` dada**, toda transformación y selección aprendida **solo con las filas de ajuste** de cada pliegue (`Pipeline` + `ColumnTransformer`), mismas configuraciones de modelos, mismas métricas, mismo costo FP + 5·FN, mismas preguntas.
- **Cambia**: la base (3 150 clientes, no 4 521), la respuesta (`fuga`, no `contrata`), las columnas y la separación entrenamiento/prueba (aquí la **creamos** 80/20 estratificada con semilla 2026; en el taller viene hecha, igual que la columna `pliegue`).

⚠️ **Antes de empezar**: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` y `openpyxl` (para leer el `.xlsx`). Los datos están en la carpeta de la Clase 33; el notebook la busca solo.
""")

code(r"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

SEMILLA = 2026                       # igual que el taller
np.random.seed(SEMILLA)

# Busca la carpeta de la Clase 33 (donde está el Excel) subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase*33*")))
carpeta = next(raiz.glob("Clase*33*"))

print("Versiones -> python-pandas:", pd.__version__, "| numpy:", np.__version__, "| scipy:", scipy.__version__,
      "| scikit-learn:", sklearn.__version__)
print("Datos en:", carpeta.name)
""")
transfer(r"""
Con `bank` leerás `pd.read_csv("clientes_entrenamiento.csv")` y `pd.read_csv("clientes_prueba.csv")` con **rutas relativas** (los dos `.csv` junto al notebook, como pide el enunciado) y no necesitas buscar ninguna carpeta. Deja impresas las **versiones de los paquetes** como arriba: el enunciado lo exige. La semilla 2026 se usa en `random_state` de todo lo aleatorio (bosque, boosting, K-means).
""")

# ====================================================================== PARTE 1
md(r"""
---
## Parte 1 · Problema, preparación y representación (12 pts en el taller)

### 1(a) Tarea, unidad de observación, respuesta, y la variable que «no existe» al predecir
La base de churn (Clase 33) tiene 3 150 clientes de una telefónica y 12 variables; `fuga = 1` si el cliente se fue. Para que la práctica sea idéntica a la del taller necesitamos tres cosas que ahí vienen dadas y aquí **construimos**:

1. `id_cliente` (identificador del registro; no es predictor).
2. Una separación **entrenamiento / prueba** 80/20 estratificada por la respuesta, con semilla 2026.
3. Una columna **`pliegue`** (1 a 5) en el entrenamiento: pliegues estratificados fijos. Desde aquí **no se vuelven a generar**: igual que en el taller, todos los modelos usan los mismos pliegues.
""")
code(r"""
def limpiar(df):
    # Valores imposibles -> NaN (regla de negocio de la Clase 33; no aprende nada de los datos)
    df = df.copy()
    for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
        df.loc[df[c] < 0, c] = np.nan
    df.loc[(df["edad"] < 15) | (df["edad"] > 100), "edad"] = np.nan
    return df


crudo = pd.read_excel(carpeta / "Base de datos Churn.xlsx")
crudo.insert(0, "id_cliente", np.arange(1, len(crudo) + 1))      # identificador del registro
print("Filas y columnas de la base completa:", crudo.shape)

# Separación 80/20 estratificada (semilla 2026). Conservamos las filas: no se elimina nada.
ent, pru = train_test_split(crudo, test_size=0.20, stratify=crudo["fuga"], random_state=SEMILLA)
ent = ent.sort_values("id_cliente").reset_index(drop=True)
pru = pru.sort_values("id_cliente").reset_index(drop=True)

# Columna `pliegue` (1..5): 5 pliegues estratificados, fijos desde ahora
ent["pliegue"] = 0
for k, (_, idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(ent, ent["fuga"]), start=1):
    ent.loc[idx, "pliegue"] = k

print("Entrenamiento:", ent.shape, "| Prueba:", pru.shape)
print("Tasa de fuga -> entrenamiento: %.4f | prueba: %.4f" % (ent.fuga.mean(), pru.fuga.mean()))
print(ent.groupby("pliegue")["fuga"].agg(filas="size", fugas="sum", tasa="mean").to_string())
""")
transfer(r"""
Todo este bloque **ya viene hecho por el profesor**: `clientes_entrenamiento.csv` (3 617 filas) y `clientes_prueba.csv` (904 filas) están separados, `id_cliente` existe y `pliegue` está en el entrenamiento. **No crees tu propia separación ni tus propios pliegues**: usa la columna dada (`entrenamiento["pliegue"]`) con sus valores tal como vienen (mira `.unique()` para saber si van de 0 a 4 o de 1 a 5). La respuesta es `contrata` y la clase positiva es `contrata = 1`.
""")
md(T["p1a"])
code(r"""
# Variable análoga a `duracion`: columna que NO existiría al momento de predecir.
print(ent.groupby("fuga")[["clv", "segundos", "frec_llamadas", "n_contactos", "antiguedad"]].mean().round(2).to_string())
LEAK = ["clv"]
NO_PRED = ["id_cliente", "pliegue", "fuga"] + LEAK
print("\nColumnas excluidas de los predictores:", NO_PRED)
""")
transfer(r"""
La tarea es **clasificación binaria supervisada**, la unidad de observación es **un registro (cliente y su contacto de campaña)** y la respuesta es `contrata`. La variable prohibida es **`duracion`**: se mide cuando la llamada termina y la llamada registrada es justamente el evento que se quiere anticipar («la predicción se realiza antes de la llamada»). Se excluyen `duracion`, `id_cliente`, `pliegue` y `contrata`. Argumenta con los tres ingredientes de arriba (cuándo se mide, qué contiene, qué pasaría en producción). Ojo: otras variables de la campaña (`contactos_campana` «incluida la llamada registrada») merecen al menos una mención crítica, pero el enunciado solo pide excluir `duracion`.
""")

md(r"""
### 1(b) Tipos de variables, faltantes, categorías raras, duplicados y balance de clases
Se revisa **la base de entrenamiento** (la prueba se reserva). Dos cosas distintas pueden verse mal: un valor **imposible** (error de captura) y un valor **inusual pero posible**. En el taller, `saldo` negativo es posible (cuenta sobregirada) y `dias_previos = -1` es un **código** («no hubo contacto previo»), no un número de días. Aquí el equivalente son los valores negativos imposibles en `segundos`, `antiguedad`, etc. y la ausencia de categorías `unknown` (aquí los faltantes son `NaN` reales).
""")
code(r"""
print("Tipos de variables:")
print(ent.drop(columns=["id_cliente", "pliegue"]).dtypes.to_string())

neg = (ent[["antiguedad", "segundos", "frec_llamadas", "frec_sms", "n_contactos", "edad"]] < 0).sum()
print("\nValores NEGATIVOS por variable (imposibles aquí):\n", neg.to_string())
print("Edades fuera de 15-100:", int(((ent.edad < 15) | (ent.edad > 100)).sum()))

ent_l = limpiar(ent)
fal = ent_l.drop(columns=["id_cliente", "pliegue"]).isna().sum()
print("\nFaltantes tras convertir imposibles en NaN:\n", fal[fal > 0].to_string())
print("Filas con algún faltante:", int(ent_l.drop(columns=["id_cliente", "pliegue"]).isna().any(axis=1).sum()))

cols_sin_id = [c for c in ent.columns if c not in ("id_cliente", "pliegue")]
print("\nFilas duplicadas (ignorando id y pliegue):", int(ent.duplicated(subset=cols_sin_id).sum()),
      "-> se informan y se CONSERVAN (el enunciado pide conservar las filas)")

print("\nBalance de clases (entrenamiento):")
print(ent.fuga.value_counts().rename("n").to_frame().assign(pct=lambda d: d.n / d.n.sum()).to_string())
base = max(ent.fuga.mean(), 1 - ent.fuga.mean())
print("\nExactitud de predecir SIEMPRE la clase mayoritaria (0):", round(base, 4))
print("Sensibilidad de esa regla:", 0.0)
""")
transfer(r"""
Haz el mismo inventario sobre `clientes_entrenamiento.csv`: `dtypes`, faltantes (`isna().sum()`), **cuántas filas tienen `unknown`** en cada categórica (`ocupacion`, `educacion`, `canal`, `resultado_previo`…; `unknown` es un valor, no un `NaN`, y **se mantiene como categoría**), duplicados (`duplicated()` ignorando `id_cliente` y `pliegue`), y `value_counts(normalize=True)` de `contrata`. Interpreta: (i) `saldo < 0` es plausible (sobregiro/deuda), no un error; (ii) `dias_previos = -1` es un **código de «sin contacto previo»**: tratarlo como −1 día distorsionaría cualquier promedio y cualquier escala; (iii) la exactitud de «nunca contrata» es la proporción de la clase mayoritaria y **no detecta ningún contrato** (sensibilidad 0), así que no basta; requiere sensibilidad, AUC, log loss y una referencia.
""")
md(T["p1b"])

md(r"""
### 1(c) Variables nuevas, one-hot y escalamiento «necesario para cada método»
El taller pide `contactado_antes` (indicador de que `dias_previos ≠ -1`) y `log_contactos_previos = log(1 + contactos_previos)`, reemplazar el −1 por 0 conservando el indicador, mantener `unknown` y codificar con one-hot. Los análogos en churn:

| Taller (banco) | Práctica (churn) | Idea |
|---|---|---|
| `contactado_antes = 1[dias_previos ≠ -1]` | `sin_sms = 1[frec_sms = 0]` | indicador de un **estado especial** (ausencia) que no se debe mezclar con la cantidad |
| `log_contactos_previos = log(1 + contactos_previos)` | `log_frec_sms = log(1 + frec_sms)` | conteo muy asimétrico: el log reduce la cola; `1 +` permite el cero |
| `-1 → 0` en `dias_previos` | los negativos imposibles → `NaN` → mediana (dentro del pliegue) | el código no puede entrar como número |
| `unknown` como categoría + one-hot | `tipo_plan`, `estado`, `nivel_monto` como categorías + one-hot | categorías sin orden |

Ambas variables nuevas son transformaciones **fila a fila sin parámetros aprendidos** (no usan la respuesta ni otras filas), así que se pueden calcular antes de separar sin causar fuga. En cambio **imputar medianas, escalar, one-hot (qué categorías existen) y seleccionar columnas** sí aprenden de los datos: eso va **dentro de cada pliegue**.
""")
code(r"""
def preparar(df):
    d = limpiar(df)
    d["sin_sms"] = (d["frec_sms"] == 0).astype(int)                 # indicador (análogo a contactado_antes)
    d["log_frec_sms"] = np.log1p(d["frec_sms"])                      # log(1 + conteo) (análogo a log_contactos_previos)
    d["nivel_monto"] = pd.cut(d["monto"], [-1, 0, 1, 2, 99], labels=["m0", "m1", "m2", "m3mas"]).astype(str)
    d["tipo_plan"] = "plan" + d["tipo_plan"].astype(str)
    d["estado"] = "est" + d["estado"].astype(str)
    return d


tr = preparar(ent)
te = preparar(pru)
ytr = tr["fuga"].to_numpy()
yte = te["fuga"].to_numpy()

NUM = ["numero_fallas", "antiguedad", "segundos", "frec_llamadas", "frec_sms", "n_contactos", "edad", "log_frec_sms"]
BIN = ["reclamos", "sin_sms"]
CAT = ["tipo_plan", "estado", "nivel_monto"]
PRED = NUM + BIN + CAT
assert not set(PRED) & set(NO_PRED)
print("Predictores:", len(PRED), "->", PRED)
print("\nsin_sms: %.3f de los clientes | media de log_frec_sms: %.3f" % (tr.sin_sms.mean(), tr.log_frec_sms.mean()))
print(tr.groupby("sin_sms")["fuga"].agg(n="size", tasa_fuga="mean").to_string())
""")
md(r"""
**Escalamiento «necesario para cada método».** El escalamiento solo importa cuando el método usa **distancias o magnitudes**:

| Método | ¿Escalar numéricas? | Por qué |
|---|---|---|
| Regresión logística sin penalización | recomendable (estandarizar) | el ajuste es invariante a la escala, pero la optimización numérica converge mejor y los coeficientes quedan comparables |
| Logística + selección por F de ANOVA | indiferente para F; se estandariza igual por la logística | F de ANOVA compara medias entre clases dividido por varianza: no cambia con la escala |
| Árbol, Random Forest, Boosting | **no** | solo comparan «x ≤ umbral»: invariantes a transformaciones monótonas |
| SVM radial | **sí, obligatorio** | usa distancias euclidianas en el kernel: una variable en miles domina a una en unidades |
| Naive Bayes gaussiano | no | modela cada variable por separado (su propia media y varianza) |
| PCA y K-means (partes 4–5) | **sí** | varianza y distancias dependen de las unidades |

Todo el preprocesamiento se arma como un `ColumnTransformer` **dentro del `Pipeline`**: se ajusta con las filas de ajuste de cada pliegue y se aplica sin reajustar a las filas del pliegue de validación (o a la prueba).
""")
code(r"""
def prep(escalar):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("esc", StandardScaler())] if escalar else [])
    ct = ColumnTransformer([
        ("num", Pipeline(num), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CAT)],
        verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


# Vista de las columnas preparadas (ajustando con TODO el entrenamiento solo para listarlas)
ct = prep(True).fit(tr[PRED])
Xprep = ct.transform(tr[PRED])
P_COLS = list(Xprep.columns)
p = len(P_COLS)
print("Columnas preparadas p =", p)
print(P_COLS)
print("Con p =", p, "el gamma del SVM del enunciado es 1/p =", round(1 / p, 5))
""")
transfer(r"""
`contactado_antes = (dias_previos != -1).astype(int)`; `dias_previos = dias_previos.replace(-1, 0)` **después** de crear el indicador (si no, pierdes la información); `log_contactos_previos = np.log1p(contactos_previos)`. Las binarias `mora`, `hipoteca`, `prestamo` son `yes/no`: pásalas a 1/0 (o déjalas en el one-hot). Las categóricas son `ocupacion`, `estado_civil`, `educacion`, `canal`, `mes`, `resultado_previo` (y decide qué haces con `dia` y `mes`: documenta tu decisión). **Mantén `unknown` como una categoría más** (no lo imputes). `OneHotEncoder(handle_unknown="ignore")` dentro del `ColumnTransformer`; escala solo las numéricas continuas (`edad`, `saldo`, `contactos_campana`, `dias_previos`, `log_contactos_previos`, …) y **solo para los métodos que lo necesitan** (tabla de arriba). Y recuerda: `p` en `gamma = 1/p` es el número de columnas **ya preparadas** (después del one-hot), no el de columnas originales.
""")

# ====================================================================== PARTE 2
md(r"""
---
## Parte 2 · Selección de variables y modelos supervisados (24 pts)

### Un motor de validación cruzada con la columna `pliegue` dada
Para cada pliegue `k`: las filas con `pliegue ≠ k` son las **filas de ajuste** (todo se aprende ahí: imputación, escala, one-hot, selección de columnas, modelo) y las filas con `pliegue = k` son las de **validación**, de donde salen las predicciones fuera de pliegue (OOF): probabilidades para los modelos probabilísticos y **puntajes de decisión** para la SVM. Se mide además el log loss **de entrenamiento** de cada ajuste (necesario para los árboles).

El **modelo de referencia** predice, en cada pliegue, la proporción positiva de las filas de ajuste (`DummyClassifier(strategy="prior")`).
""")
code(r"""
PLIEGUES = sorted(tr["pliegue"].unique())


def oof(modelo, svm=False):
    # Predicciones fuera de pliegue con la columna `pliegue` dada. Devuelve (pred, ajustes, log loss de entrenamiento por ajuste).
    pred = np.zeros(len(tr))
    ajustes, ll_ent = [], []
    for k in PLIEGUES:
        aj = (tr["pliegue"] != k).to_numpy()
        va = ~aj
        m = clone(modelo).fit(tr.loc[aj, PRED], ytr[aj])
        if svm:
            pred[va] = m.decision_function(tr.loc[va, PRED])
        else:
            pred[va] = m.predict_proba(tr.loc[va, PRED])[:, 1]
            ll_ent.append(log_loss(ytr[aj], m.predict_proba(tr.loc[aj, PRED])[:, 1]))
        ajustes.append(m)
    return pred, ajustes, ll_ent


def metricas(pred, svm=False):
    por_pliegue = [roc_auc_score(ytr[tr["pliegue"].to_numpy() == k], pred[tr["pliegue"].to_numpy() == k]) for k in PLIEGUES]
    out = {"AUC": roc_auc_score(ytr, pred), "AUC medio por pliegue": float(np.mean(por_pliegue))}
    if not svm:
        out = {"log loss": log_loss(ytr, pred), "Brier": brier_score_loss(ytr, pred), **out}
    return out


# --- Las siete configuraciones de la tabla del enunciado (más la referencia)
LR = dict(penalty=None, max_iter=5000)               # sin penalización
MODELOS = {
    "Referencia (tasa positiva)": Pipeline([("prep", prep(False)), ("m", DummyClassifier(strategy="prior"))]),
    "Logística (todas las columnas)": Pipeline([("prep", prep(True)), ("m", LogisticRegression(**LR))]),
    "Logística (10 mejores por F)": Pipeline([("prep", prep(True)), ("sel", SelectKBest(f_classif, k=10)), ("m", LogisticRegression(**LR))]),
    "Árbol prof. 3": Pipeline([("prep", prep(False)), ("m", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEMILLA))]),
    "Árbol prof. 6": Pipeline([("prep", prep(False)), ("m", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEMILLA))]),
    "Random Forest": Pipeline([("prep", prep(False)), ("m", RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=10,
                                                                                    random_state=SEMILLA, n_jobs=-1))]),
    "Boosting (GradientBoosting)": Pipeline([("prep", prep(False)), ("m", GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, n_iter_no_change=20, validation_fraction=0.1, random_state=SEMILLA))]),
    "SVM radial": Pipeline([("prep", prep(True)), ("m", SVC(kernel="rbf", C=1.0, gamma="auto"))]),     # gamma='auto' = 1/p
    "Naive Bayes gaussiano": Pipeline([("prep", prep(False)), ("m", GaussianNB())]),
}
SVM = {"SVM radial"}

RES, AJ, LLENT, OOF = {}, {}, {}, {}
for nombre, mod in MODELOS.items():
    pred, ajs, ll = oof(mod, svm=nombre in SVM)
    OOF[nombre], AJ[nombre], LLENT[nombre] = pred, ajs, ll
    RES[nombre] = metricas(pred, svm=nombre in SVM)
print("Modelos ajustados con", len(PLIEGUES), "pliegues cada uno:", len(RES), "modelos")
""")
transfer(r"""
Usa **la columna `pliegue` que viene en `clientes_entrenamiento.csv`**: `for k in sorted(entrenamiento["pliegue"].unique())`. **No** uses `cross_val_score(cv=5)`, `StratifiedKFold` ni `GridSearchCV` con tus propios pliegues: el enunciado pide «los cinco pliegues entregados» (es un punto que se pierde). Para cada pliegue: separar `ajuste = pliegue != k` y `valid = pliegue == k`, ajustar el `Pipeline` solo con `ajuste` y guardar la predicción de `valid` en un vector OOF alineado con las filas. Respuesta `contrata`; clase positiva 1. El **modelo de referencia** en cada pliegue es `ajuste["contrata"].mean()` aplicado a todas las filas de `valid` (y, en la evaluación final, la tasa de **todo** el entrenamiento). Boosting: elige **una** variante vista en clase (p. ej. `GradientBoostingClassifier`, `AdaBoost`, `HistGradientBoosting`, `XGBoost`), con ≤ 200 árboles, `max_depth = 3`, e **informa la variante y la tasa de aprendizaje**.
""")

md(r"""
### 2(a) Logística con todas las columnas vs. con las 10 de mayor F de ANOVA (5 pts)
`SelectKBest(f_classif, k=10)` está **dentro** del `Pipeline`, de modo que en cada pliegue el estadístico F se calcula solo con las filas de ajuste. La lista de columnas elegidas puede cambiar de un pliegue a otro: se informa pliegue por pliegue.
""")
code(r"""
sel_tab = {}
for k, m in zip(PLIEGUES, AJ["Logística (10 mejores por F)"]):
    sel_tab[f"pliegue {k}"] = np.array(P_COLS)[m.named_steps["sel"].get_support()]
sel_df = pd.DataFrame(sel_tab)
sel_df.index = [f"#{i}" for i in range(1, 11)]
print("Columnas seleccionadas (top 10 por F) en cada pliegue:")
print(sel_df.to_string())

frec = pd.Series(np.concatenate(list(sel_tab.values()))).value_counts()
print("\nEn cuántos de los 5 pliegues fue elegida cada columna:")
print(frec.to_string())
print("Columnas que NUNCA se eligen:", sorted(set(P_COLS) - set(frec.index)))
""")
code(r"""
tab_a = pd.DataFrame({n: RES[n] for n in ["Logística (todas las columnas)", "Logística (10 mejores por F)"]}).T
print(tab_a.to_string())

# Limitación del filtro univariado: no mira redundancia entre columnas (F se calcula una columna a la vez)
m1 = AJ["Logística (10 mejores por F)"][0]
Xaj = prep(True).fit_transform(tr.loc[tr.pliegue != PLIEGUES[0], PRED])
elegidas = list(sel_df["pliegue 1"])
cor = Xaj[elegidas].corr().abs().where(lambda d: np.triu(np.ones(d.shape), 1).astype(bool)).stack().sort_values(ascending=False)
print("\nPares más correlacionados ENTRE las 10 columnas elegidas (pliegue 1):")
print(cor.head(5).to_string())
""")
transfer(r"""
Mismo experimento con tus columnas preparadas (después del one-hot): `SelectKBest(f_classif, k=10)` entre el preprocesamiento y la logística, **dentro del Pipeline**. Para informar las columnas de cada pliegue: `pipe.named_steps["sel"].get_support()` sobre los nombres de `pipe.named_steps["prep"].get_feature_names_out()`. Verifica que **p > 10** (si no, no hay nada que seleccionar). Tabla pedida: columnas por pliegue + log loss/Brier/AUC OOF de las dos logísticas. «Sin penalización»: `LogisticRegression(penalty=None)` en scikit-learn, `glm(family = binomial)` en R. Respuestas conceptuales: (i) F de ANOVA es un **filtro supervisado** que ordena columnas **una a una** según cuánto separan las medias de `contrata`; PCA es **no supervisado** y **crea** combinaciones nuevas que maximizan varianza sin mirar la respuesta; (ii) el filtro univariado ignora la **redundancia** y las **interacciones**: puede elegir dos columnas casi idénticas y descartar una que solo es útil junto a otra.
""")
md(T["p2a"])

md(r"""
### 2(b) Comparación de las configuraciones (11 pts)
Métricas OOF: **log loss**, **Brier** y **AUC** para los modelos probabilísticos; solo **AUC** para la SVM (el puntaje de decisión no es una probabilidad). Para los **dos árboles** se agrega el log loss **medio de entrenamiento** de los cinco ajustes. Las métricas globales se calculan sobre el vector OOF completo (las 2 520 predicciones); se añade el AUC medio por pliegue como chequeo.
""")
code(r"""
tabla = pd.DataFrame(RES).T
tabla["log loss entren. (medio 5 ajustes)"] = [np.mean(LLENT[n]) if n.startswith("Árbol") else np.nan for n in tabla.index]
cols = ["log loss", "Brier", "AUC", "AUC medio por pliegue", "log loss entren. (medio 5 ajustes)"]
print(tabla[cols].to_string(na_rep="—"))

prob = tabla.drop(index=["Referencia (tasa positiva)", "SVM radial"])
ELEGIDO = prob["log loss"].idxmin()
print("\nModelo probabilístico con MENOR log loss OOF (sin contar la referencia):", ELEGIDO)
print("Boosting: variante = GradientBoostingClassifier, tasa de aprendizaje = 0.05, árboles por pliegue =",
      [m.named_steps["m"].n_estimators_ for m in AJ["Boosting (GradientBoosting)"]])
print("Árboles por pliegue: hojas prof.3 =", [int(m.named_steps["m"].get_n_leaves()) for m in AJ["Árbol prof. 3"]],
      "| prof.6 =", [int(m.named_steps["m"].get_n_leaves()) for m in AJ["Árbol prof. 6"]])
""")
transfer(r"""
La tabla final del taller tiene **las mismas filas** (referencia, logística todas, logística top-10, árbol 3, árbol 6, Random Forest, Boosting, SVM, Naive Bayes) y las mismas columnas. Trampas: **SVM** → `SVC(kernel="rbf", C=1, gamma=1/p)` (`gamma="auto"` hace exactamente `1/p` en scikit-learn; en R `e1071::svm(gamma = 1/p)`) con **`decision_function`** y **solo AUC** (no calcules log loss ni Brier con puntajes sin calibrar); **árboles** → `max_depth` 3 y 6 con `min_samples_leaf = 20`, más el log loss medio de entrenamiento **de los cinco ajustes** (calcula el log loss sobre `ajuste` en cada pliegue y promedia); **Random Forest** → `n_estimators = 200`, `max_features = "sqrt"`, `min_samples_leaf = 10`; **Naive Bayes** → variante gaussiana (`GaussianNB`) sobre las columnas numéricas y binarias preparadas. La selección del modelo se hace **por menor log loss OOF entre los probabilísticos** (no por AUC). Escribe la regla de selección en el notebook: el enunciado acepta ganadores distintos «cuando la regla esté justificada».
""")
md(T["p2b"])

md(r"""
### 2(c) Interpretación (8 pts)
Cinco evidencias numéricas respaldan cada punto antes de escribir la interpretación: (1) brecha entrenamiento–OOF de los árboles; (2) tamaño de los bosques vs. árbol único; (3) calibración y extremos de Naive Bayes; (4) efecto de C, gamma y escalamiento en la SVM; (5) ranking vs. probabilidad.
""")
code(r"""
# (1) Árboles: profundidad y sobreajuste (entrenamiento vs. OOF)
arb = pd.DataFrame({n: {"log loss entrenamiento": np.mean(LLENT[n]), "log loss OOF": RES[n]["log loss"],
                        "brecha": RES[n]["log loss"] - np.mean(LLENT[n]), "AUC OOF": RES[n]["AUC"]}
                    for n in ["Árbol prof. 3", "Árbol prof. 6", "Random Forest", "Boosting (GradientBoosting)"]}).T
arb["log loss entrenamiento"] = [np.mean(LLENT[n]) for n in arb.index]
arb["brecha"] = arb["log loss OOF"] - arb["log loss entrenamiento"]
print(arb.to_string())

# (3) Naive Bayes: cuán extremas son sus probabilidades, y calibración por tramos de probabilidad
def extremos(v):
    return {"p<0.01": np.mean(v < 0.01), "p>0.99": np.mean(v > 0.99), "min": v.min(), "max": v.max()}
print("\nExtremos de las probabilidades OOF:")
print(pd.DataFrame({n: extremos(OOF[n]) for n in ["Naive Bayes gaussiano", "Logística (todas las columnas)", "Random Forest", ELEGIDO]}).T.to_string())

def calibracion(v, y, tramos=(0, .05, .10, .20, .40, .60, 1.0)):
    g = pd.cut(v, list(tramos), include_lowest=True)
    d = pd.DataFrame({"p": v, "y": y, "g": g}).groupby("g", observed=True).agg(n=("y", "size"), p_media=("p", "mean"), tasa_obs=("y", "mean"))
    return d
print("\nCalibración OOF de Naive Bayes (p media vs. tasa observada):")
print(calibracion(OOF["Naive Bayes gaussiano"], ytr).to_string())
""")
code(r"""
# (4) SVM: efecto de C, gamma y escalamiento (AUC OOF con las mismas pliegues)
p_ = p
variantes = {
    "Enunciado: C=1, gamma=1/p, ESCALADA": (prep(True), dict(C=1, gamma=1 / p_)),
    "C=0.1, gamma=1/p":                    (prep(True), dict(C=0.1, gamma=1 / p_)),
    "C=10, gamma=1/p":                     (prep(True), dict(C=10, gamma=1 / p_)),
    "C=1, gamma=0.1/p (más suave)":        (prep(True), dict(C=1, gamma=0.1 / p_)),
    "C=1, gamma=10/p (más local)":         (prep(True), dict(C=1, gamma=10 / p_)),
    "C=1, gamma=1/p, SIN escalar":         (prep(False), dict(C=1, gamma=1 / p_)),
}
filas = {}
for nom, (pp, kw) in variantes.items():
    pr, _, _ = oof(Pipeline([("prep", pp), ("m", SVC(kernel="rbf", **kw))]), svm=True)
    filas[nom] = {"AUC OOF": roc_auc_score(ytr, pr)}
print(pd.DataFrame(filas).T.to_string())

# (5) ¿Mejor orden implica mejores probabilidades?  El AUC solo depende del ORDEN; el log loss, de los valores.
pe = OOF[ELEGIDO]
print("\nTransformación monótona de las probabilidades del modelo elegido (%s):" % ELEGIDO)
for etiqueta, v in [("original", pe), ("p^3 (mismo orden, probabilidades deformadas)", pe ** 3), ("sqrt(p)", np.sqrt(pe))]:
    v = np.clip(v, 1e-9, 1 - 1e-9)
    print(f"  {etiqueta:48s} AUC = {roc_auc_score(ytr, v):.4f} | log loss = {log_loss(ytr, v):.4f} | Brier = {brier_score_loss(ytr, v):.4f}")
print("\nRanking por AUC vs. ranking por log loss (modelos probabilísticos):")
rk = prob[["AUC", "log loss"]].copy()
rk["rango AUC"] = rk["AUC"].rank(ascending=False).astype(int)
rk["rango log loss"] = rk["log loss"].rank().astype(int)
print(rk.sort_values("rango log loss").to_string())
""")
transfer(r"""
Estas cinco evidencias se calculan igual con los OOF de **tus** modelos y apoyan cada línea de la pregunta 2(c): *profundidad y sobreajuste* (log loss de entrenamiento muy inferior al OOF en el árbol profundo); *bagging vs. Random Forest vs. boosting* (varianza vs. sesgo, decorrelación de árboles por `max_features`, corrección secuencial de errores con tasa de aprendizaje); *independencia condicional* (Naive Bayes asume que, dado `contrata`, las columnas son independientes; con muchas columnas correlacionadas o dummies de la misma variable esa suposición se rompe y las probabilidades se vuelven extremas); *gaussiana sobre columnas binarias* (una dummy 0/1 no es normal: su «campana» es un artefacto; las probabilidades salen mal calibradas aunque el orden sea aceptable); *C, gamma y escala* (C controla cuánto se castiga el error vs. margen; gamma el alcance del kernel; sin escalar domina la variable de mayor magnitud). Cierre obligatorio: **un mejor ordenamiento (AUC) no implica mejores probabilidades (log loss/Brier)**; demuéstralo con tu propia tabla o con una transformación monótona como arriba.
""")
md(T["p2c"])

# ====================================================================== PARTE 3
md(r"""
---
## Parte 3 · Evaluación y decisión (10 pts)

### 3(a) Umbral que minimiza el costo hipotético (4 pts)
Con las probabilidades OOF del modelo elegido se evalúan los umbrales 0.05, 0.10, …, 0.95. Se predice «fuga» si `p ≥ t`. El costo es

$$C(t) = FP(t) + 5\,FN(t)$$

donde **FP** es una alerta de fuga que no ocurre (contactar a quien no se iba) y **FN** una fuga no detectada (perder a quien sí se iba). El umbral elegido es el de menor costo (si hay empate, el menor `t`).
""")
code(r"""
UMBRALES = np.round(np.arange(0.05, 0.951, 0.05), 2)


def matriz(y, pred_pos):
    tn, fp, fn, tp = confusion_matrix(y, pred_pos, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


filas = []
for t in UMBRALES:
    tn, fp, fn, tp = matriz(ytr, pe >= t)
    filas.append({"t": t, "VN": tn, "FP": fp, "FN": fn, "VP": tp, "costo C(t)": fp + 5 * fn,
                  "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp)})
curva = pd.DataFrame(filas).set_index("t")
T_STAR = float(curva["costo C(t)"].idxmin())
print("Modelo elegido:", ELEGIDO)
print(curva.to_string())
print("\nUmbral elegido t* =", T_STAR, "| costo mínimo =", int(curva.loc[T_STAR, "costo C(t)"]),
      "| costo en t = 0.50:", int(curva.loc[0.50, "costo C(t)"]))
print("Referencias: no alertar a nadie cuesta 5·(fugas) =", 5 * int(ytr.sum()), "| alertar a todos cuesta (no fugas) =", int((1 - ytr).sum()))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(curva.index, curva["costo C(t)"], "o-", color="#CA8A04")
ax[0].axvline(T_STAR, ls="--", color="k")
ax[0].set(xlabel="umbral t", ylabel="C(t) = FP + 5·FN", title="Costo OOF por umbral")
ax[1].plot(curva.index, curva["FP"], "o-", label="FP (alerta sin fuga)")
ax[1].plot(curva.index, curva["FN"], "s-", label="FN (fuga no detectada)")
ax[1].axvline(T_STAR, ls="--", color="k")
ax[1].set(xlabel="umbral t", ylabel="cantidad", title="Los dos errores"); ax[1].legend()
plt.tight_layout(); plt.show()
""")
transfer(r"""
Usa **las probabilidades OOF de tu modelo elegido** (menor log loss OOF) y el costo `FP + 5·FN` con la definición del enunciado: FP = alerta de contratación que no ocurre, FN = contratación no detectada. Umbrales exactos `0.05, 0.10, …, 0.95` con regla **`p ≥ t`** (no `>`; con `>` cambian los casos en el borde). Presenta la tabla (o la curva) y **interpreta los dos errores**: qué significa en términos del banco un FP (un contacto telefónico poco productivo) y un FN (un cliente que habría contratado y no se llamó) y por qué un FN pesa 5 veces más empuja el umbral hacia valores bajos. Reporta también el costo de `t = 0.50` y las dos estrategias triviales (nadie / todos).
""")
md(T["p3a"])

md(r"""
### 3(b) Reajuste con todo el entrenamiento y evaluación en la prueba (6 pts)
Se reajusta el modelo elegido con **todo el entrenamiento** (el `Pipeline` aprende imputación, escala, one-hot y —si corresponde— selección con las 2 520 filas) y se evalúa en la prueba **una sola vez**. La referencia usa la tasa positiva de **todo** el entrenamiento. Se informa log loss, Brier, AUC y la curva ROC, y se compara `t*` con 0.50 mediante matriz de confusión, sensibilidad, especificidad y **C/n** (costo medio por cliente).
""")
code(r"""
final = clone(MODELOS[ELEGIDO]).fit(tr[PRED], ytr)
p_te = final.predict_proba(te[PRED])[:, 1]
ref = clone(MODELOS["Referencia (tasa positiva)"]).fit(tr[PRED], ytr)
p_ref = ref.predict_proba(te[PRED])[:, 1]
print("Tasa positiva del entrenamiento completo (referencia):", round(ytr.mean(), 4), "| tasa de fuga en la prueba:", round(yte.mean(), 4))

met = pd.DataFrame({
    ELEGIDO: {"log loss": log_loss(yte, p_te), "Brier": brier_score_loss(yte, p_te), "AUC": roc_auc_score(yte, p_te)},
    "Referencia": {"log loss": log_loss(yte, p_ref), "Brier": brier_score_loss(yte, p_ref), "AUC": roc_auc_score(yte, p_ref)},
}).T
print(met.to_string())

filas = {}
for nombre, pr in [(ELEGIDO, p_te), ("Referencia", p_ref)]:
    for t in (T_STAR, 0.50):
        tn, fp, fn, tp = matriz(yte, pr >= t)
        filas[(nombre, f"t={t:.2f}")] = {"VN": tn, "FP": fp, "FN": fn, "VP": tp, "sensibilidad": tp / (tp + fn),
                                          "especificidad": tn / (tn + fp), "C = FP+5FN": fp + 5 * fn, "C/n": (fp + 5 * fn) / len(yte)}
print()
tab_cm = pd.DataFrame(filas).T
for c_ in ["VN", "FP", "FN", "VP", "C = FP+5FN"]:
    tab_cm[c_] = tab_cm[c_].astype(int)
print(tab_cm.to_string())

fpr, tpr, _ = roc_curve(yte, p_te)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fpr, tpr, color="#CA8A04", label=f"{ELEGIDO} (AUC = {roc_auc_score(yte, p_te):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", label="Referencia (AUC = 0.5)")
for t, mk in [(T_STAR, "o"), (0.50, "s")]:
    tn, fp, fn, tp = matriz(yte, p_te >= t)
    ax[0].plot(fp / (fp + tn), tp / (tp + fn), mk, ms=9, label=f"umbral {t:.2f}")
ax[0].set(xlabel="1 − especificidad", ylabel="sensibilidad", title="Curva ROC en la prueba"); ax[0].legend()
ax[1].hist([p_te[yte == 0], p_te[yte == 1]], bins=25, stacked=True, label=["no se fuga", "se fuga"], color=["#9CA3AF", "#CA8A04"])
ax[1].axvline(T_STAR, color="k", ls="--"); ax[1].axvline(0.5, color="r", ls=":")
ax[1].set(xlabel="probabilidad predicha", ylabel="clientes", title="Distribución en prueba"); ax[1].legend()
plt.tight_layout(); plt.show()
""")
transfer(r"""
Reajusta **el mismo `Pipeline` elegido** con todo `clientes_entrenamiento.csv` (nunca con la prueba) y evalúa en `clientes_prueba.csv` **una sola vez**. Informa log loss, Brier, AUC, **la curva ROC** (figura) y, para `t*` (el de 3(a)) y para 0.50: matriz de confusión, sensibilidad, especificidad y `C/n` con `C = FP + 5·FN`. Incluye el **modelo de referencia** con la tasa del entrenamiento completo. Luego responde las tres preguntas finales con argumentos: *qué recomendarían al banco* (a quién llamar y con qué umbral, en términos del costo), *qué permiten concluir los resultados* (calidad de ordenamiento, calibración, costo frente a la referencia y a 0.50) y *qué falta verificar para una campaña futura* (estabilidad temporal, que `contactos_campana` y otras variables estén disponibles **antes** de la llamada, costos reales FP/FN, incertidumbre por tamaño de muestra, equidad entre grupos, monitoreo). El enunciado pide expresamente extenderse en esto.
""")
md(T["p3b"])

# ====================================================================== PARTE 4
md(r"""
---
## Parte 4 · Componentes principales y agrupamiento (8 pts)

### Variables y estandarización
El enunciado fija **cuatro variables** del entrenamiento, **estandarizadas con media y desviación muestral** (divisor n − 1). Aquí elegimos cuatro numéricas de la base de churn que mantienen la mezcla del taller (conteos asimétricos y una medida continua): `numero_fallas`, `frec_llamadas`, `frec_sms` y `n_contactos`. Cuidado: `StandardScaler` usa divisor n (no n − 1); para seguir la instrucción se estandariza a mano con `std(ddof=1)`.
""")
code(r"""
V4 = ["numero_fallas", "frec_llamadas", "frec_sms", "n_contactos"]
X4_raw = tr[V4].copy()
print("Faltantes en las 4 variables (entrenamiento):", X4_raw.isna().sum().to_dict())
med = X4_raw.median()
X4 = X4_raw.fillna(med)                          # imputación con la mediana del entrenamiento (afecta a muy pocas filas)
mu, sd = X4.mean(), X4.std(ddof=1)              # media y desviación MUESTRAL
Z = (X4 - mu) / sd
print("\nMedias de Z:", Z.mean().round(10).to_dict())
print("Desv. muestrales de Z:", Z.std(ddof=1).round(10).to_dict())
print("\nDescriptivos originales:")
print(X4.describe().T[["mean", "std", "min", "50%", "max"]].to_string())
print("\nCorrelaciones:")
print(X4.corr().round(3).to_string())
""")
transfer(r"""
Variables del enunciado: **`edad`, `saldo`, `contactos_campana`, `contactos_previos`**, tomadas del **conjunto de entrenamiento**, estandarizadas con `media` y `std(ddof=1)` (la desviación **muestral**). Con `bank` no hay faltantes que imputar; no incluyas `log_contactos_previos` ni `dias_previos`: son cuatro variables exactamente las nombradas. Estandariza una vez y reutiliza la misma matriz `Z` en PCA, K-means y (la distancia de Mahalanobis no la necesita, pero da lo mismo con `Z` o con los datos originales). Las puntuaciones de la prueba **no** se usan en las partes 4 y 5.
""")

md(r"""
### 4(a) PCA: varianza explicada, cargas y plano (3 pts)
""")
code(r"""
pca = PCA(n_components=4).fit(Z)
var = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i}" for i in range(1, 5)])
print("Varianza explicada:\n", pd.DataFrame({"autovalor": pca.explained_variance_, "% varianza": var, "% acumulado": var.cumsum()}).to_string())
print("\nPC1 + PC2 =", round(var["PC1"] + var["PC2"], 4))
cargas = pd.DataFrame(pca.components_.T, index=V4, columns=var.index)
print("\nCargas (autovectores; filas = variables):")
print(cargas.round(3).to_string())
print("\nCargas escaladas (autovector·√autovalor = correlación variable–PC):")
print((cargas * np.sqrt(pca.explained_variance_)).round(3).to_string())

scores = pd.DataFrame(pca.transform(Z), columns=var.index, index=tr.index)
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(scores["PC1"], scores["PC2"], s=8, alpha=0.4, color="#6B7280")
ax.set(xlabel=f"PC1 ({var['PC1']:.1%})", ylabel=f"PC2 ({var['PC2']:.1%})", title="Observaciones en el plano PC1–PC2")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
""")
transfer(r"""
Mismo código con `Z` de `edad, saldo, contactos_campana, contactos_previos`. Informa **varianza explicada por PC1 y PC2, su suma, las cargas** (tabla variables × componentes), y el **gráfico PC1–PC2**. Para la pregunta «¿qué información resume la proyección y cuál omite?» razona con tus números: porcentaje de varianza retenida y omitida (PC3–PC4), qué variables dominan cada componente (signos y magnitudes de las cargas), que PCA **no usa `contrata`** y que dos observaciones cercanas en el plano pueden diferir en las dimensiones omitidas. Los **signos** de los componentes son arbitrarios (cambian entre Python y R): interpreta magnitudes y relaciones, no el signo absoluto.
""")
md(T["p4a"])

md(r"""
### 4(b) K-means con k = 2, 3 y 4 (5 pts)
`KMeans(n_clusters=k, n_init=30, random_state=2026)` sobre `Z`. Se compara **WSS** (suma de cuadrados intra-grupo, `inertia_`) y **silueta media**, y se elige el k de mayor silueta. Luego, tamaños, perfiles (en unidades originales), grupos en el plano PCA y tasa de fuga dentro de cada grupo.
""")
code(r"""
res_k = {}
for k in (2, 3, 4):
    km = KMeans(n_clusters=k, n_init=30, random_state=SEMILLA).fit(Z)
    res_k[k] = dict(modelo=km, WSS=km.inertia_, silueta=silhouette_score(Z, km.labels_))
print(pd.DataFrame({k: {"WSS": v["WSS"], "silueta media": v["silueta"]} for k, v in res_k.items()}).T.to_string())
K_STAR = max(res_k, key=lambda k: res_k[k]["silueta"])
print("\nk elegido (mayor silueta) =", K_STAR)

km = res_k[K_STAR]["modelo"]
lab = pd.Series(km.labels_, index=tr.index, name="grupo")
perf = X4.assign(grupo=lab).groupby("grupo").agg(["mean"]).droplevel(1, axis=1)
perf.insert(0, "n", lab.value_counts().sort_index())
perf["% del total"] = perf["n"] / len(lab)
perf["tasa de fuga"] = pd.Series(ytr, index=tr.index).groupby(lab).mean()
print("\nPerfiles (medias en unidades originales), tamaño y tasa de fuga por grupo:")
print(perf.to_string())
print("\nPerfiles en unidades estandarizadas (media de Z):")
print(Z.assign(grupo=lab).groupby("grupo").mean().round(2).to_string())
print("\nTasa de fuga global:", round(ytr.mean(), 4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
colores = ["#CA8A04", "#2563EB", "#059669", "#DC2626"]
for g in sorted(lab.unique()):
    m = lab == g
    ax[0].scatter(scores.loc[m, "PC1"], scores.loc[m, "PC2"], s=8, alpha=0.5, color=colores[g], label=f"grupo {g} (n={m.sum()})")
ax[0].set(xlabel="PC1", ylabel="PC2", title=f"K-means (k={K_STAR}) en el plano PCA"); ax[0].legend(markerscale=2)
ks = list(res_k)
ax[1].plot(ks, [res_k[k]["WSS"] for k in ks], "o-", color="#CA8A04"); ax[1].set(xlabel="k", ylabel="WSS", title="WSS y silueta")
ax2 = ax[1].twinx(); ax2.plot(ks, [res_k[k]["silueta"] for k in ks], "s--", color="#2563EB"); ax2.set_ylabel("silueta media"); ax2.grid(False)
plt.tight_layout(); plt.show()
""")
transfer(r"""
`KMeans(n_clusters=k, n_init=30, random_state=2026)` para k = 2, 3 y 4 **sobre las cuatro variables estandarizadas** (nunca sobre las originales ni sobre los dos componentes). En R: `kmeans(Z, centers = k, nstart = 30)` con `set.seed(2026)`. Tabla con `tot.withinss` (WSS) y silueta media por k (`cluster::silhouette`); **elige el k de mayor silueta** (no el del «codo»). Informa tamaños, perfiles (medias por grupo en unidades originales: describe con palabras, p. ej. «clientes con saldo alto y pocos contactos»), el gráfico en el plano PCA coloreado por grupo y la **proporción de `contrata` dentro de cada grupo**. Explicación pedida: los grupos se forman **sin mirar `contrata`** con solo 4 variables y dependen de k y de la escala; una alta proporción de contrata en un grupo no convierte al grupo en una clase «predicha»: no hay umbral, no hay modelo evaluado, y los grupos no se construyeron para separar la respuesta.
""")
md(T["p4b"])

# ====================================================================== PARTE 5
md(r"""
---
## Parte 5 · Detección de anomalías (6 pts)

### 5(a) Distancia de Mahalanobis al cuadrado y corte χ² (3 pts)
Con media muestral $m$ y covarianza muestral $S$ de las cuatro variables:
$$D^2_i = (x_i - m)^\top S^{-1} (x_i - m).$$
Si los datos fueran normales multivariados, $D^2 \sim \chi^2_4$ y el corte natural sería su cuantil 0.99.
""")
code(r"""
S = np.cov(X4.to_numpy(), rowvar=False)                   # covarianza muestral (divisor n - 1)
dif = X4.to_numpy() - X4.mean().to_numpy()
d2 = np.einsum("ij,jk,ik->i", dif, np.linalg.inv(S), dif)
CORTE = chi2.ppf(0.99, df=4)
marc = d2 > CORTE
print("Corte (chi-cuadrado 0.99, 4 g.l.):", round(CORTE, 4))
print("Registros marcados:", int(marc.sum()), "de", len(d2), "->", f"{marc.mean():.2%}", "(esperado si fuera normal: 1.00%)")
print("Mediana de D^2:", round(np.median(d2), 3), "| mediana teórica chi2(4):", round(chi2.ppf(0.5, 4), 3))
print("Control: Mahalanobis^2 con datos estandarizados es idéntico:", np.allclose(d2, np.einsum("ij,jk,ik->i", Z.to_numpy(), np.linalg.inv(np.cov(Z.to_numpy(), rowvar=False)), Z.to_numpy())))

top = pd.Series(d2, index=tr.index).nlargest(2)
dd = tr.loc[top.index, ["id_cliente"] + V4 + ["fuga"]].copy()
dd.insert(1, "D2", top.values)
print("\nLos dos registros con mayor D^2 (valores originales):")
print(dd.to_string())
print("\nSus puntajes z (cuántas desv. muestrales de la media):")
print(Z.loc[top.index].round(2).to_string())
print("\nPercentiles de cada variable (0-100) de esos registros:")
print(X4.rank(pct=True).loc[top.index].mul(100).round(1).to_string())
print("\nResumen de las variables (media y desv. muestral):")
print(pd.DataFrame({"media": X4.mean(), "sd": X4.std(ddof=1), "máx": X4.max()}).T.round(2).to_string())
""")
transfer(r"""
Idéntico con `edad, saldo, contactos_campana, contactos_previos` del entrenamiento: `D²` con la media y la covarianza **muestrales** (R: `mahalanobis(x, colMeans(x), cov(x))`, ya devuelve D² al cuadrado), corte `qchisq(0.99, df = 4)` / `chi2.ppf(0.99, 4)`, regla `D² > corte`. Informa **el corte, la cantidad y el porcentaje de marcados**, y los **dos registros de mayor D²** con sus **valores originales** (no estandarizados), incluyendo `id_cliente`. En la tabla de esos dos registros añade sus puntajes z para poder explicar qué variable (o qué combinación) los hace inusuales.
""")
md(T["p5a"])

md(r"""
### 5(b) ¿Por qué son inusuales? ¿Son errores? ¿Garantiza el corte un 1 %? (3 pts)
""")
code(r"""
# Evidencia para 5(b): ¿hay variables muy asimétricas? ¿qué pasa al marcar con otro criterio? ¿cambia con una covarianza robusta?
print("Asimetría de cada variable:", X4.skew().round(2).to_dict())
print("Mayor |z| univariado de cada uno de los 2 registros:", Z.loc[top.index].abs().max(axis=1).round(2).to_dict())
for q in (0.95, 0.99, 0.999):
    c = chi2.ppf(q, 4)
    print(f"  cuantil {q}: corte {c:.2f} -> marcados {int((d2 > c).sum())} ({(d2 > c).mean():.2%}) | esperado {1 - q:.2%}")

# Mahalanobis sin los 2 registros más extremos (¿cuánto influyen en S?)
resto = X4.drop(index=top.index)
S2 = np.cov(resto.to_numpy(), rowvar=False)
d2b = np.einsum("ij,jk,ik->i", X4.to_numpy() - resto.mean().to_numpy(), np.linalg.inv(S2), X4.to_numpy() - resto.mean().to_numpy())
print("\nD^2 de los dos registros recalculando m y S SIN ellos:", np.round(d2b[[X4.index.get_loc(i) for i in top.index]], 1),
      "| antes:", np.round(top.values, 1))
print("Registros marcados con m y S sin ellos:", int((d2b > CORTE).sum()))
""")
transfer(r"""
Responde en tres partes, siempre con **tus números**: (i) *por qué son inusuales*: qué variable(s) toman valores extremos o qué combinación contradice la correlación entre las variables (D² mide distancia **en la métrica de la covarianza**: un valor moderado en una variable puede ser raro dado el valor de otra); (ii) *¿demuestra errores?* No: una distancia grande es una **señal de revisión**, no una prueba; en `bank`, un `saldo` muy alto o muchos `contactos_campana` pueden ser clientes reales y valiosos (o una captura errónea: hay que verificarlo); (iii) *¿el corte garantiza marcar al 1 %?* No: 1 % es lo esperado **solo si** los datos son normales multivariados y m y S se conocen; con datos asimétricos y de cola pesada (como `saldo`) se marcan bastantes más, y m y S estimados con los mismos datos (inflados por los extremos) cambian el resultado. Antes de eliminar un registro: verificar unidades y captura, buscar duplicados, revisar el resto de sus columnas y el contexto de negocio, comparar el modelo con y sin el registro, decidir con un criterio escrito (conservar, corregir o eliminar) y no eliminar si el caso es una población real de interés.
""")
md(T["p5b"])

# ====================================================================== CIERRE
md(r"""
---
## Resumen de decisiones de esta práctica y checklist de transferencia
| Paso | Decisión en la práctica | Qué hacer con `bank` |
|---|---|---|
| Variable prohibida | `clv` (supuesto de la práctica) | `duracion` (+ `id_cliente`, `pliegue`, `contrata`) |
| Pliegues | creados aquí con semilla 2026 | **usar la columna `pliegue` dada** |
| Indicador + log | `sin_sms`, `log_frec_sms` | `contactado_antes`, `log_contactos_previos` |
| One-hot | `tipo_plan`, `estado`, `nivel_monto` | todas las categóricas, `unknown` incluida |
| Selección | F de ANOVA k = 10 dentro del `Pipeline` | igual, informando columnas por pliegue |
| Modelos | 9 filas con la misma configuración | idéntico |
| Elegido | menor log loss OOF probabilístico | su propio ganador (puede ser otro) |
| Umbral | min FP + 5·FN con `p ≥ t` | idéntico, con **tus** OOF |
| Cuatro variables | `numero_fallas`, `frec_llamadas`, `frec_sms`, `n_contactos` | `edad`, `saldo`, `contactos_campana`, `contactos_previos` |
| Mahalanobis | corte χ² 0.99, 4 g.l. | idéntico |

### Lista de verificación antes de entregar (la guía en PDF la amplía)
- [ ] Un solo lenguaje (R **o** Python) en todo el taller.
- [ ] `duracion`, `id_cliente`, `pliegue` y `contrata` fuera de los predictores.
- [ ] Imputación, escala, one-hot y selección **dentro** de cada pliegue (`Pipeline`).
- [ ] Se usó la columna `pliegue` entregada; semilla 2026; clase positiva = 1.
- [ ] SVM: puntaje de decisión y solo AUC. Árboles: log loss medio de entrenamiento de los cinco ajustes.
- [ ] Modelo elegido = menor log loss OOF probabilístico; regla escrita.
- [ ] Prueba usada una sola vez, en 3(b); referencia con la tasa del entrenamiento completo.
- [ ] PCA/K-means/Mahalanobis con las **cuatro** variables estandarizadas (media y desv. muestral).
- [ ] Notebook ejecutado de punta a punta, rutas relativas, versiones de paquetes, integrantes identificados.
""")

md(r"""
## Nota para grupos que elijan **R**: correspondencia con esta práctica
| Paso | Python (esta práctica) | R (equivalente) | Ojo con… |
|---|---|---|---|
| Lectura | `pd.read_csv` | `read.csv()` / `readr::read_csv()` | rutas relativas; `stringsAsFactors` |
| Semilla | `random_state=2026` | `set.seed(2026)` antes de cada paso aleatorio | R no tiene un `random_state` por modelo |
| Pipeline por pliegue | `Pipeline` + `ColumnTransformer` | `tidymodels` (`recipes` + `workflows`) o un bucle sobre `pliegue` con `prep()`/`bake()` | la receta se ajusta solo con las filas de ajuste |
| One-hot | `OneHotEncoder` | `step_dummy()` / `model.matrix()` | niveles ausentes en el pliegue de validación |
| Logística sin penalización | `LogisticRegression(penalty=None)` | `glm(y ~ ., family = binomial)` | separación perfecta: avisos de convergencia |
| Selección F de ANOVA | `SelectKBest(f_classif, k=10)` | `anova`/`oneway.test` por columna o `caret::sbf` | calcular en las filas de ajuste |
| Árbol | `DecisionTreeClassifier(max_depth, min_samples_leaf=20)` | `rpart(..., control = rpart.control(maxdepth, minbucket = 20, cp = 0))` | `cp = 0` evita la poda; `minbucket` ≈ mínimo por hoja |
| Random Forest | `n_estimators=200, max_features="sqrt", min_samples_leaf=10` | `ranger(num.trees = 200, mtry = floor(sqrt(p)), min.node.size = 10, probability = TRUE)` | `min.node.size` no es idéntico a `min_samples_leaf` |
| Boosting | `GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=…)` | `xgboost` / `gbm` (`n.trees`, `interaction.depth`, `shrinkage`) | informar variante y tasa |
| SVM radial | `SVC(C=1, gamma=1/p)` + `decision_function` | `e1071::svm(kernel = "radial", cost = 1, gamma = 1/p, decision.values = TRUE)` | el signo del puntaje depende del orden de las clases |
| Naive Bayes | `GaussianNB` | `e1071::naiveBayes()` (gaussiano para numéricas) | una dummy 0/1 no es gaussiana |
| Métricas | `log_loss`, `brier_score_loss`, `roc_auc_score` | `MLmetrics::LogLoss`, `mean((p - y)^2)`, `pROC::auc` | clase positiva = 1 |
| PCA | `PCA` sobre `Z` | `prcomp(Z)` (con `Z` ya estandarizada con `sd()`, que es muestral) | signos distintos |
| K-means | `KMeans(n_init=30)` | `kmeans(Z, centers = k, nstart = 30)` | algoritmo y semilla distintos: etiquetas permutadas |
| Silueta | `silhouette_score` | `cluster::silhouette(km$cluster, dist(Z))` | usar la media de `sil_width` |
| Mahalanobis | `einsum` con `inv(S)` | `mahalanobis(x, colMeans(x), cov(x))` | R ya devuelve D² |
| Corte | `chi2.ppf(0.99, 4)` | `qchisq(0.99, df = 4)` | — |

El profesor aceptó expresamente **diferencias numéricas** entre lenguajes (algoritmos, semillas, signos de componentes) siempre que la implementación y la regla de selección estén justificadas.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"c42-{i:03d}"
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
