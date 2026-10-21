"""Genera el notebook guiado de Árboles de decisión (Clase 34).

Uso normal:      python notebook_c34.py             -> Guias/notebooks/C34_ArbolesDecision.ipynb
Modo de prueba:  C34_TEST=1 python notebook_c34.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C34_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C34_ArbolesDecision_test.ipynb" if TEST else "C34_ArbolesDecision.ipynb")
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
# Clase 34 · Árboles de decisión
**Diplomado Data Science** · Prof. Eduardo Rodríguez · misma base de fuga (churn) de la Clase 33

Material de apoyo:
- `Guias/C34_Guia_ArbolesDecision.pdf` — guía visual de la clase.
- `Guias/apuntes/C34_Apuntes_ArbolesDecision.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C34_Ejercicios_Cuaderno_ArbolesDecision.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor: `Clase 34 Árboles de decisión (Profesor Eduardo Rodríguez)/clase34_explicado.ipynb`, `S34_arboles_de_decision.pptx` y `Arbol de decision.pptx` (presentación genérica).
- Transcripción: `Clase 34 Árboles de decisión.txt`.

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `openpyxl` (para leer el `.xlsx`) y `matplotlib`. Instala una vez en tu entorno:
```
pip install scikit-learn openpyxl matplotlib pandas
```
Los datos (`Base de datos Churn.xlsx`) están en la carpeta de la Clase 33; el notebook la busca solo.

### Objetivos
1. Explicar cómo un árbol **parte el espacio** en rectángulos y leer un árbol entrenado como **reglas de negocio** (`export_text`, `plot_tree`, hojas con n, tasa y lift).
2. Calcular a mano **Gini, entropía y ganancia**, elegir un corte y comprobar con código que scikit-learn hace lo mismo (barrido de la raíz).
3. Ver la **ganancia de información** de la Clase 33 desde dentro del modelo: es la misma cantidad que usa el árbol para cortar.
4. Diagnosticar **sobreajuste** con la curva de profundidad y controlarlo con pre-poda (`max_depth`, `min_samples_leaf`) y post-poda (`ccp_alpha` con la ruta de costo-complejidad).
5. Distinguir la importancia **por impureza** (sesgada) de la **por permutación**, y conocer los límites de ambas.
6. Comprobar que el árbol es invariante a transformaciones monótonas, que es inestable, y entender por qué promediar árboles (Random Forest, Clase 35) ayuda.
7. Detectar lo que el material original afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y carga de datos |
| A | 15 | A1 · Entrenar y leer un árbol: hojas, reglas y lift |
| A | 10 | A2 · Rectángulos frente a un plano (logística) |
| A | 10 | A3 · Lo que un árbol hace mal: la diagonal |
| B | 15 | B1 · Impureza: Gini y entropía a mano |
| B | 15 | B2 · Un corte a mano y su verificación contra scikit-learn |
| B | 20 | B3 · El barrido completo de la raíz (y de un nodo interno) |
| B | 10 | B4 · ¿Gini o entropía? |
| B | 10 | B5 · La ganancia es la información mutua (puente con la Clase 33) |
| C | 10 | C1 · Validación cruzada, ronda por ronda |
| C | 15 | C2 · La curva de profundidad: sobreajuste a la vista |
| C | 25 | C3 · Post-poda: ruta de costo-complejidad y `ccp_alpha` |
| D | 20 | D1 · Importancia: impureza frente a permutación |
| D | 10 | D2 · Escalar o transformar: el árbol es invariante |
| D | 10 | D3 · Inestabilidad del árbol |
| D | 15 | D4 · Puente al bosque (sin repetir la Clase 35) |
| D | 10 | D5 · Comparación final: ¿para qué es el modelo? |
| — | 10 | Alertas del material |
| — | 25 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Todo el preprocesamiento (imputación) vive **dentro de un `Pipeline`**: en validación cruzada se ajusta solo con el bloque de entrenamiento de cada ronda. Por eso algunos números difieren en el tercer decimal de los de las guías en PDF (allí la mediana se calculó con toda la base antes de validar); el texto de este notebook cita la salida real de las celdas.
- Los números del texto fueron verificados contra la salida real de este notebook con `scikit-learn 1.9`. Con otras versiones pueden variar un poco.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

# Busca la carpeta de la Clase 33 (donde está el Excel) subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase*33*")))
carpeta = next(raiz.glob("Clase*33*"))

print("scikit-learn:", sklearn.__version__)
print("Datos en:", carpeta.name)
""")
md("""
### Los datos
La **misma base de fuga** de la Clase 33: 3.150 clientes de una telefónica y 12 variables. `fuga = 1` si el cliente se fue (15.7 %).

La Clase 33 limpió con criterio de negocio: un valor negativo en `segundos`, `frec_llamadas`, `frec_sms`, `antiguedad` o `n_contactos` es imposible (error de captura) y una `edad` fuera de 15–100 también. Esos valores pasan a `NaN` en una función (`limpiar`, que **no aprende nada** de los datos) y la **mediana se imputa dentro del `Pipeline`**. Como dijo el profesor, esta clase **no construye las 50 variables** de la Clase 33: el árbol llega a lo mismo partiendo el espacio y con 12 variables el resultado se puede leer.

Un detalle de método: en la Parte A y B solo **describimos** la base (qué corte elige el árbol, cuánta impureza hay). Ahí no hay nada que evaluar, así que se ajusta con todos los datos, igual que en clase. Desde la Parte C, donde medimos desempeño, cada número sale de validación cruzada con la imputación dentro del `Pipeline`.
""")
code("""
def limpiar(df):
    \"\"\"Valores imposibles -> NaN (regla de negocio; no aprende nada de los datos).\"\"\"
    df = df.copy()
    for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
        df.loc[df[c] < 0, c] = np.nan
    df.loc[(df["edad"] < 15) | (df["edad"] > 100), "edad"] = np.nan
    return df


crudo = pd.read_excel(carpeta / "Base de datos Churn.xlsx")
datos = limpiar(crudo)
y = datos.pop("fuga")
X = datos
VARS = list(X.columns)

print("Clientes:", X.shape[0], "| variables:", X.shape[1], "| tasa de fuga:", round(y.mean(), 4), f"({y.sum()} de {len(y)})")
print("Valores faltantes tras limpiar (por variable):")
print(X.isna().sum()[X.isna().sum() > 0].to_string())
print("Total de celdas a imputar:", int(X.isna().sum().sum()))
""")
code("""
# --- Piezas que usaremos en todo el notebook
def imputador():
    return SimpleImputer(strategy="median").set_output(transform="pandas")


def pipe_arbol(**kw):
    return Pipeline([("imp", imputador()), ("arbol", DecisionTreeClassifier(random_state=0, **kw))])


def pipe_bosque(**kw):
    return Pipeline([("imp", imputador()), ("bosque", RandomForestClassifier(random_state=0, n_jobs=-1, **kw))])


def pipe_logistica():
    return Pipeline([("imp", imputador()), ("esc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])


CV = StratifiedKFold(5, shuffle=True, random_state=1)     # la validación cruzada de la clase


def auc_cv(modelo, datos=None):
    return cross_val_score(modelo, X if datos is None else datos, y, cv=CV, scoring="roc_auc")


# Un test reservado (75/25 estratificado), como en la clase: sirve para la importancia y el ejercicio de cierre
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Base imputada con la mediana de TODA la base: SOLO para describir (Partes A y B), nunca para evaluar
Xi = imputador().fit_transform(X)
yv = y.values
print("Train:", Xtr.shape, "| test:", Xte.shape, "| imputada (descriptiva):", Xi.shape)
""")

# ================================================================== PARTE A
md("""
## Parte A · Cómo parte el espacio un árbol

Una **regresión logística** ajusta un único plano para todo el espacio: cada variable pesa lo mismo en todas partes. Un **árbol** no ajusta ninguna fórmula: hace preguntas de sí/no encadenadas («¿reclamó?», «¿usa menos de 934 segundos?») y corta el espacio en **rectángulos**. En cada uno predice una constante: la tasa de fuga de los clientes de entrenamiento que cayeron ahí.

$$\\hat p(x) = \\sum_m c_m \\cdot \\mathbb{1}\\{x \\in R_m\\}$$

### A1 · Entrenar y leer un árbol (15 min)
`max_depth=3` limita las preguntas encadenadas; `min_samples_leaf=60` prohíbe hojas con menos de 60 clientes (con hojas de 5 clientes, una tasa de 100 % puede ser puro ruido de muestra chica); `random_state=0` hace reproducible el desempate entre cortes iguales.
""")
code("""
arbol = pipe_arbol(max_depth=3, min_samples_leaf=60).fit(X, y)
n_arbol = arbol[-1]
print("Hojas:", n_arbol.get_n_leaves(), "| profundidad:", n_arbol.get_depth())
print()
print(export_text(n_arbol, feature_names=VARS, decimals=1))
""")
code("""
fig, ax = plt.subplots(figsize=(17, 6.5))
plot_tree(n_arbol, feature_names=VARS, class_names=["se queda", "se fuga"], filled=True, rounded=True,
          impurity=False, fontsize=9, ax=ax)
plt.show()
""")
md("""
`plot_tree` muestra `value` como conteos `[se queda, se fuga]`, que cuestan leer. Lo útil para el negocio es la **tabla de hojas**: cuántos clientes hay en cada región, su tasa de fuga y el **lift** (tasa de la hoja / tasa general). `apply` devuelve la hoja de cada cliente y `reglas_hojas` reconstruye la regla que lleva a cada una.
""")
code("""
def reglas_hojas(modelo, nombres):
    \"\"\"Devuelve {id_de_hoja: regla en texto} recorriendo el árbol.\"\"\"
    t = modelo.tree_
    salida = {}

    def rec(i, condiciones):
        if t.children_left[i] == -1:
            salida[i] = " y ".join(condiciones)
            return
        v, u = nombres[t.feature[i]], t.threshold[i]
        rec(t.children_left[i], condiciones + [f"{v} ≤ {u:.1f}"])
        rec(t.children_right[i], condiciones + [f"{v} > {u:.1f}"])

    rec(0, [])
    return salida


def tabla_hojas(pipe, Xd, yd):
    Xt = pipe[:-1].transform(Xd)
    modelo = pipe[-1]
    hoja = modelo.apply(Xt)
    t = (pd.DataFrame({"hoja": hoja, "fuga": np.asarray(yd)}).groupby("hoja")
         .agg(clientes=("fuga", "size"), fugas=("fuga", "sum"), tasa=("fuga", "mean")))
    t["% de la base"] = 100 * t["clientes"] / len(yd)
    t["lift"] = t["tasa"] / np.mean(yd)
    t["regla"] = pd.Series(reglas_hojas(modelo, list(Xt.columns)))
    return t.sort_values("tasa", ascending=False).reset_index(drop=True)


hojas = tabla_hojas(arbol, X, y)
display(hojas.style.format({"tasa": "{:.1%}", "% de la base": "{:.1f}", "lift": "{:.1f}"}))
""")
md("""
Las 7 hojas, de mayor a menor riesgo: 149 clientes con **94 %** de fuga (lift ≈ 6), 67 con 67 %, 340 con 43 %, 62 con 40 %, 69 con 35 %, 104 con 9 % y el «núcleo sano» de 2.359 clientes con 4.5 %. La hoja más roja se lee como *reclamó, usa poco el servicio (57 llamadas o menos) y tuvo fallas técnicas*: no es un coeficiente, es un **segmento** al que se le puede mandar una campaña mañana. Eso es lo que un árbol da y una logística no.

### ✍️ Ejercicio A1
La campaña de retención solo alcanza a **unos 220 clientes**. (a) Con la tabla de hojas, ¿qué hojas contactarías y qué fracción de los 495 fugados capturarías? (b) Compara con contactar a 216 clientes al azar. (c) Escribe en una frase de negocio la regla de la segunda hoja.
""")
code("""
# tu código aquí
""")
solution("""
h = hojas.copy()
h["fugas acum."] = h["fugas"].cumsum()
h["clientes acum."] = h["clientes"].cumsum()
h["% de los fugados capturados"] = 100 * h["fugas acum."] / y.sum()
display(h[["clientes", "tasa", "clientes acum.", "fugas acum.", "% de los fugados capturados"]].head(4))

n_c = int(h.loc[1, "clientes acum."])
print(f"Contactando las 2 primeras hojas ({n_c} clientes): {h.loc[1, '% de los fugados capturados']:.1f} % de los fugados")
print(f"Al azar, {n_c} clientes capturarían ≈ {100 * n_c / len(y):.1f} % de los fugados")
""", "(a) Las dos primeras hojas suman 216 clientes y capturan 185 de los 495 fugados (≈ 37 %). (b) Al azar, 216 clientes son el 6.9 % de la base y capturarían ≈ 6.9 % de los fugados: el árbol concentra la fuga **5.4 veces** mejor. (c) «Clientes que reclamaron, hacen 57 llamadas o menos y no han tenido fallas técnicas: 67 % se fuga» (lift ≈ 4.3).")

md("""
### A2 · Rectángulos frente a un plano (10 min)
Para ver la diferencia dibujamos las dos fronteras en un plano de **dos variables** (`frec_llamadas` y `segundos`). La logística traza una sola frontera suave; el árbol, una escalera de rectángulos. Solo es una imagen ilustrativa: el modelo real usa las 12 variables.
""")
code("""
par = ["frec_llamadas", "segundos"]
modelos_2d = {
    "Árbol (profundidad 3, hojas ≥ 60)": pipe_arbol(max_depth=3, min_samples_leaf=60).fit(X[par], y),
    "Regresión logística": pipe_logistica().fit(X[par], y),
}
muestra = Xi[par].sample(700, random_state=0)
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, (nombre, m) in zip(ejes, modelos_2d.items()):
    DecisionBoundaryDisplay.from_estimator(m, X[par], response_method="predict_proba", plot_method="pcolormesh",
                                           cmap="RdYlGn_r", alpha=0.55, vmin=0, vmax=1, grid_resolution=160, ax=ax)
    ax.scatter(muestra[par[0]], muestra[par[1]], c=y.loc[muestra.index], cmap="RdYlGn_r", s=8, edgecolor="none", alpha=0.7)
    ax.set_title(nombre, loc="left", fontsize=10)
    ax.set_xlabel(par[0]); ax.set_ylabel(par[1])
plt.tight_layout(); plt.show()
""")
md("""
El árbol parte el plano con cortes **perpendiculares a los ejes** y predice una constante por región (el color es la probabilidad de fuga). Dos consecuencias que se verifican en el resto del notebook: (1) captura **no linealidades e interacciones solas**, porque cada rama puede usar variables distintas; (2) solo corta perpendicular a los ejes, y eso tiene un costo.

### A3 · Lo que un árbol hace mal: la diagonal (10 min)
Si la frontera real es una recta oblicua, el árbol necesita una **escalera** de muchos cortes para aproximarla. Con datos sintéticos donde la clase es `x1 > x2` lo vemos:
""")
code("""
rng = np.random.default_rng(0)
P = rng.uniform(0, 1, size=(4000, 2))
d = (P[:, 0] > P[:, 1]).astype(int)                       # la frontera real es la diagonal
Pa, Pb, da, db = train_test_split(P, d, test_size=0.5, random_state=0)

print("profundidad | exactitud en datos nuevos")
for prof in (1, 2, 3, 5, 8):
    acc = DecisionTreeClassifier(max_depth=prof, random_state=0).fit(Pa, da).score(Pb, db)
    print(f"{prof:11d} | {acc:.3f}")
print("logística   |", round(LogisticRegression().fit(Pa, da).score(Pb, db), 3))
""")
md("""
### ✍️ Ejercicio A3
(a) ¿Cuál es la **menor profundidad** con la que el árbol llega a una exactitud de al menos 0.97 en los datos nuevos? ¿Cuántas hojas tiene? (b) Rota las variables: agrega la columna `x1 − x2` y vuelve a ajustar un árbol de profundidad 1. ¿Qué pasa y qué lección deja sobre la ingeniería de variables?
""")
code("""
# tu código aquí
""")
solution("""
for prof in range(1, 13):
    arb = DecisionTreeClassifier(max_depth=prof, random_state=0).fit(Pa, da)
    acc = arb.score(Pb, db)
    if acc >= 0.97:
        print(f"(a) Profundidad {prof}: exactitud {acc:.3f} con {arb.get_n_leaves()} hojas")
        break

Pa2, Pb2 = np.c_[Pa, Pa[:, 0] - Pa[:, 1]], np.c_[Pb, Pb[:, 0] - Pb[:, 1]]
tocon = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Pa2, da)
print("(b) Con x1 - x2 y profundidad 1: exactitud", round(tocon.score(Pb2, db), 3), "| corte en la variable", tocon.tree_.feature[0], "(2 = x1 - x2)")
""", "(a) Hacen falta 5 niveles y 27 hojas para pasar de 0.97 (la logística, con un solo plano, llega a 0.998). (b) Con la columna `x1 − x2` un único corte (profundidad 1) acierta ≈ 100 %: el árbol no \"descubre\" combinaciones lineales de variables, hay que dárselas. Es un buen caso donde **sí** conviene construir una variable (contrasta con la afirmación de que el árbol \"llega a lo mismo\" que las 50 variables de la Clase 33: para interacciones no lineales sí, para combinaciones lineales oblicuas cuesta).")

# ================================================================== PARTE B
md("""
## Parte B · Impureza, ganancia y el corte

En cada nodo el árbol prueba **todas las variables y todos los umbrales** y se queda con el corte que más reduce la mezcla de clases. Es un algoritmo **voraz**: elige lo mejor *ahora*, sin mirar dos pasos adelante, por eso no garantiza el árbol óptimo, solo uno bueno y rápido.

### B1 · Impureza: Gini y entropía a mano (15 min)
Con p = proporción de fuga **dentro del nodo** (se recalcula en cada nodo con los clientes que caen ahí):

$$\\text{Gini} = 1 - p^2 - (1-p)^2 \\qquad \\text{Entropía} = -p\\log_2 p - (1-p)\\log_2(1-p)$$

Las dos valen 0 si el nodo es puro y alcanzan su máximo con 50/50 (Gini 0.5, entropía 1).
""")
code("""
def gini_p(p):
    p = np.asarray(p, dtype=float)
    g = 1 - p ** 2 - (1 - p) ** 2
    return g if g.ndim else float(g)


def entropia_p(p):
    p = np.asarray(p, dtype=float)
    q = np.clip(p, 1e-12, 1 - 1e-12)
    h = np.where((p > 0) & (p < 1), -(q * np.log2(q) + (1 - q) * np.log2(1 - q)), 0.0)
    return h if h.ndim else float(h)


IMP = {"gini": gini_p, "entropia": entropia_p}

print("Nodo           p      Gini   Entropía")
for nombre, p in [("puro", 0.0), ("20 % de fuga", 0.2), ("mitad y mitad", 0.5), ("raíz de la clase", y.mean())]:
    print(f"{nombre:15s} {p:5.3f}  {gini_p(p):.3f}   {entropia_p(p):.3f}")

fig, ax = plt.subplots(figsize=(7, 3.2))
pp = np.linspace(0.001, 0.999, 300)
ax.plot(pp, gini_p(pp), label="Gini (máx 0.5)")
ax.plot(pp, entropia_p(pp), label="Entropía (máx 1)")
ax.axvline(y.mean(), color="red", ls="--", lw=1)
ax.set_xlabel("Proporción de fuga en el nodo (p)"); ax.set_ylabel("Impureza"); ax.legend()
plt.tight_layout(); plt.show()
""")
md("""
En la raíz hay 3.150 clientes y 495 se fugan: p = 0.157 y **Gini = 1 − 0.843² − 0.157² = 0.265**. La entropía de la raíz es 0.627 bits (la diapositiva 11 muestra 0.634 porque redondea a 84/16).

### ✍️ Ejercicio B1
(a) Gini y entropía de un nodo de 20 clientes con 12 fugas (ejemplo de la presentación genérica). (b) Lo mismo para un nodo de 300 clientes con 45 fugas. (c) ¿Para qué proporción p la entropía vale exactamente 0.5? ¿Cuánto vale el Gini ahí? (d) ¿Por qué un nodo con 99 % de fuga tiene baja impureza aunque sea \"de fuga\"?
""")
code("""
# tu código aquí
""")
solution("""
for n, k in [(20, 12), (300, 45)]:
    p = k / n
    print(f"n = {n}, fugas = {k}: p = {p:.3f} | Gini = {gini_p(p):.3f} | entropía = {entropia_p(p):.3f}")

malla = np.linspace(0.001, 0.5, 500000)
p_05 = malla[np.argmin(np.abs(entropia_p(malla) - 0.5))]
print(f"(c) entropía = 0.5 en p ≈ {p_05:.3f} | Gini ahí = {gini_p(p_05):.3f}")
print(f"(d) Gini con p = 0.99: {gini_p(0.99):.4f} | entropía: {entropia_p(0.99):.3f}")
""", "(a) p = 0.6 → Gini 0.480, entropía 0.971. (b) p = 0.15 → Gini 0.255, entropía 0.610. (c) p ≈ 0.110: con solo 11 % de fuga la entropía ya es la mitad de su máximo, mientras el Gini vale 0.196 (39 % de su máximo): la entropía crece más rápido cerca de 0. (d) La impureza mide **mezcla**, no cuál clase domina: un nodo casi puro (cualquiera de las dos clases) está \"ordenado\" y no necesita más preguntas.")

md("""
### B2 · Un corte a mano y su verificación contra scikit-learn (15 min)
La **ganancia** de un corte es la impureza del padre menos la impureza **ponderada** de los hijos:

$$\\text{ganancia} = \\text{Imp(padre)} - \\Big[\\tfrac{n_{izq}}{n}\\text{Imp(izq)} + \\tfrac{n_{der}}{n}\\text{Imp(der)}\\Big]$$

La ponderación es esencial: un hijo puro con 2 clientes no vale lo mismo que uno puro con 2.000. Probamos el corte `reclamos > 0` (¿el cliente reclamó?):
""")
code("""
def ganancia_corte(izq, yd, impureza="gini"):
    \"\"\"izq: máscara booleana del hijo izquierdo. Devuelve (imp. padre, imp. ponderada de los hijos, ganancia).\"\"\"
    f = IMP[impureza]
    padre = f(yd.mean())
    pond = izq.mean() * f(yd[izq].mean()) + (~izq).mean() * f(yd[~izq].mean())
    return padre, pond, padre - pond


reclamo = (Xi["reclamos"] > 0).values
for nombre, m in [("Sí reclamó", reclamo), ("No reclamó", ~reclamo)]:
    print(f"{nombre:11s} n = {m.sum():5d} | fugas = {int(yv[m].sum()):4d} | p = {yv[m].mean():.3f} | Gini = {gini_p(yv[m].mean()):.3f}")

padre, pond, gan = ganancia_corte(~reclamo, yv)
print(f"\\nGini del padre: {padre:.4f} | Gini ponderado de los hijos: {pond:.4f} | ganancia: {gan:.4f}")

# ¿Lo mismo que calcula scikit-learn?  impurity[i] y n_node_samples[i] están en tree_
t = n_arbol.tree_
izq, der = t.children_left[0], t.children_right[0]
pond_sk = (t.n_node_samples[izq] * t.impurity[izq] + t.n_node_samples[der] * t.impurity[der]) / t.n_node_samples[0]
print(f"scikit-learn: raíz = {VARS[t.feature[0]]} ≤ {t.threshold[0]:.1f} | Gini raíz {t.impurity[0]:.4f} | ganancia {t.impurity[0] - pond_sk:.4f}")
""")
md("""
A mano: reclamó = 278 clientes (210 fugan, 75.5 %, Gini 0.370); no reclamó = 2.872 (285 fugan, 9.9 %, Gini 0.179); ponderado 0.196 y **ganancia 0.069**. scikit-learn calcula exactamente lo mismo y su raíz es `reclamos ≤ 0.5` (el umbral es el punto medio entre dos valores observados, 0 y 1).

### ✍️ Ejercicio B2
La diapositiva 13 pide elegir entre dos candidatos: **A** `reclamos > 0` y **B** `edad > 40`. (a) Calcula con los datos la ganancia de **B** y compara sus conteos con los de la diapositiva (1.559 / 1.591 clientes y 240 / 255 fugas). (b) Calcula la ganancia de A con los conteos de la diapositiva (211 y 284 fugas) y con los reales (210 y 285). (c) La nota de respuesta de la diapositiva dice «Gini ponderado ~0,168 y ganancia ~0,097» para A. ¿Es correcto? ¿Qué le pasa a la ganancia cuando los dos hijos se parecen al padre?
""")
code("""
# tu código aquí
""")
solution("""
edad40 = (Xi["edad"] > 40).values
for nombre, m in [("edad > 40", edad40), ("edad ≤ 40", ~edad40)]:
    print(f"(a) {nombre}: n = {m.sum()}, fugas = {int(yv[m].sum())}")
print("    ganancia real de B:", round(ganancia_corte(~edad40, yv)[2], 5))


def gan_conteos(n1, f1, n2, f2):
    n = n1 + n2
    p = (f1 + f2) / n
    pond = (n1 * gini_p(f1 / n1) + n2 * gini_p(f2 / n2)) / n
    return gini_p(p), pond, gini_p(p) - pond


for nombre, args in [("A con 211/284 (diapositiva)", (278, 211, 2872, 284)), ("A con 210/285 (reales)", (278, 210, 2872, 285)),
                     ("B con conteos de la diapositiva", (1559, 240, 1591, 255))]:
    pad, pon, ga = gan_conteos(*args)
    print(f"(b) {nombre}: Gini ponderado {pon:.4f} | ganancia {ga:.4f}")
""", "(a) Con los datos reales `edad > 40` deja 970 clientes (147 fugas) a un lado y 2.180 (348 fugas) al otro, y su ganancia es ≈ 0.00003: no es lo que muestra la diapositiva (1.559 / 1.591; 240 / 255). La conclusión (B no separa nada) se mantiene, pero **los conteos de B no salen de estos datos**. (b) A con 211/284 da ganancia 0.0700 y con 210/285 da 0.0693 (Gini ponderado 0.195 y 0.196). (c) No: la nota dice ~0.168 y ~0.097, pero con los datos de la propia diapositiva da ≈ 0.195 y ≈ 0.070 (la diapositiva 12 muestra 0.0693). Cuando los hijos se parecen al padre la ganancia tiende a 0: el corte no separa.")

md("""
### B3 · El barrido completo de la raíz (20 min)
¿Por qué el árbol empieza por `reclamos` y no por otra variable? No hay magia: prueba **cada variable y cada umbral** y calcula la ganancia. El profesor lo reproduce con un doble bucle y 60 umbrales por variable (percentiles 2 a 98) para ir más rápido. Aquí lo hacemos de forma **vectorizada** y comparamos el barrido por percentiles con el **exacto** (todos los puntos medios entre valores consecutivos distintos, que es lo que hace scikit-learn).
""")
code("""
def mejor_corte(v, yd, min_hoja=60, impureza="gini", candidatos=None):
    \"\"\"Mejor umbral de una variable: devuelve (umbral, ganancia, n de candidatos).
    candidatos=None -> exacto (puntos medios entre valores distintos); min_hoja descarta hijos con < min_hoja clientes.\"\"\"
    f = IMP[impureza]
    v = np.asarray(v, float)
    yd = np.asarray(yd, float)
    orden = np.argsort(v, kind="stable")
    vs, ys = v[orden], yd[orden]
    acum = np.cumsum(ys)
    if candidatos is None:
        vals = np.unique(vs)
        candidatos = (vals[:-1] + vals[1:]) / 2
    n = len(vs)
    n_i = np.searchsorted(vs, candidatos, side="right")          # cuántos clientes quedan a la izquierda (v <= u)
    ok = (n_i >= min_hoja) & (n - n_i >= min_hoja)
    cand, n_i = candidatos[ok], n_i[ok]
    if len(cand) == 0:
        return np.nan, np.nan, len(candidatos)
    f_i = acum[n_i - 1]
    f_d = acum[-1] - f_i
    pond = (n_i * f(f_i / n_i) + (n - n_i) * f(f_d / (n - n_i))) / n
    g = f(yd.mean()) - pond
    k = int(np.argmax(g))
    return cand[k], g[k], len(candidatos)


def barrido(Xd, yd, impureza="gini", min_hoja=60):
    filas = []
    for c in Xd.columns:
        v = Xd[c].values
        pct = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 60)))
        u_p, g_p, n_c = mejor_corte(v, yd, min_hoja, impureza, candidatos=pct)
        u_e, g_e, _ = mejor_corte(v, yd, min_hoja, impureza)
        filas.append({"variable": c, "umbral (60 percentiles)": u_p, "ganancia (60 percentiles)": g_p,
                      "umbral exacto": u_e, "ganancia exacta": g_e, "candidatos": n_c})
    return pd.DataFrame(filas).sort_values("ganancia exacta", ascending=False).set_index("variable")


tabla = barrido(Xi, yv)
display(tabla.round(4))
print("Combinaciones (variable, umbral) evaluadas con 60 percentiles:", int(tabla["candidatos"].sum()))
print("Primera pregunta de scikit-learn:", VARS[t.feature[0]], "≤", t.threshold[0])
""")
md("""
Resultados (Gini): **reclamos gana con 0.0693**, más del doble que la segunda (`estado`, 0.0270); en el otro extremo `edad` aporta 0.0005 (sus hijos se parecen al padre, y por eso en la Clase 33 tenía información mutua casi nula). Se evaluaron **378 combinaciones** para elegir *una* sola pregunta, y el barrido completo se repite en **cada nodo**. Dos matices: (1) los umbrales por percentiles dan lo mismo en las variables con pocos valores, pero en `clv`, `segundos` y `frec_sms` el barrido exacto encuentra una ganancia algo mayor, así que «el resultado es el mismo» es cierto para el ganador, no para todas las filas; (2) la tabla de la diapositiva 12 muestra solo 9 de las 12 variables: faltan `frec_sms`, `monto` y `tipo_plan`, y las dos primeras tienen más ganancia que `antiguedad`.

### ✍️ Ejercicio B3
El barrido se repite dentro de cada hijo. (a) Aísla a los clientes que **no reclamaron** (el hijo izquierdo de la raíz) y ejecuta el barrido exacto con `min_hoja=60`: ¿qué variable y umbral elige? (b) Compara con el segundo nivel del árbol de la sección A1. (c) ¿Cambia la respuesta si usas la entropía?
""")
code("""
# tu código aquí
""")
solution("""
no_rec = ~reclamo
sub = barrido(Xi[no_rec], yv[no_rec])
display(sub.head(4)[["umbral exacto", "ganancia exacta"]].round(4))
print("(b) Árbol de scikit-learn, hijo izquierdo de la raíz:", VARS[t.feature[izq]], "≤", t.threshold[izq])
sub_e = barrido(Xi[no_rec], yv[no_rec], impureza="entropia")
print("(c) Con entropía el ganador es:", sub_e.index[0], "con umbral", sub_e["umbral exacto"].iloc[0])
""", "(a)-(b) Dentro de los que no reclamaron, el mejor corte es `segundos ≤ 934`, el mismo que muestra el árbol: el algoritmo repite el barrido en cada nodo, sin mirar a los demás. (c) Con la entropía el ganador de **este nodo** es otro: `clv` con umbral ≈ 157. En la raíz ambos criterios coinciden (`reclamos`), pero por debajo pueden elegir variables distintas: «el ranking de cortes casi siempre coincide» (diapositiva 7 de la presentación genérica) vale para la raíz, no necesariamente para todos los nodos.")

md("""
### B4 · ¿Gini o entropía? (10 min)
El comentario del notebook del profesor (y la presentación genérica) dice que dan árboles «casi idénticos». Lo medimos con la misma validación cruzada de la clase:
""")
code("""
for crit in ("gini", "entropy"):
    s = auc_cv(pipe_arbol(max_depth=5, criterion=crit))
    print(f"criterion = {crit:8s} | AUC por ronda: {np.round(s, 4)} | promedio {s.mean():.4f} | desvío {s.std():.4f}")

print()
print("Ganador en la raíz según el criterio:")
for imp in ("gini", "entropia"):
    b = barrido(Xi, yv, impureza=imp)
    print(f"  {imp:9s} -> {b.index[0]}, ganancia {b['ganancia exacta'].iloc[0]:.4f}; segunda: {b.index[1]} ({b['ganancia exacta'].iloc[1]:.4f})")
""")
md("""
La raíz es la misma (y la segunda candidata ya difiere: `estado` con Gini, `clv` con entropía), pero en profundidad 5 la **entropía da 1.4 puntos más de AUC** (0.920 contra 0.906) y las cinco rondas la favorecen, una diferencia mayor que el desvío entre rondas (≈ 0.014). «Casi idénticos» es una descripción de la **raíz**, no del desempeño: el criterio sí importa a veces, y se elige con validación cruzada como cualquier hiperparámetro.

### ✍️ Ejercicio B4
Repite la comparación Gini / entropía para `max_depth` = 3, 5, 8 y 12. ¿En qué profundidades gana la entropía? ¿Qué significa una diferencia menor que el desvío?
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for prof in (3, 5, 8, 12):
    sg = auc_cv(pipe_arbol(max_depth=prof, criterion="gini"))
    se = auc_cv(pipe_arbol(max_depth=prof, criterion="entropy"))
    filas.append({"profundidad": prof, "AUC gini": sg.mean(), "AUC entropía": se.mean(), "diferencia": se.mean() - sg.mean(),
                  "desvío gini": sg.std(), "rondas donde gana entropía": int((se > sg).sum())})
display(pd.DataFrame(filas).round(4))
""", "La entropía gana en promedio en las cuatro profundidades (+0.037, +0.015, +0.009 y +0.020 de AUC) y en las 5 rondas con profundidad 3 y 5. En profundidad 8 solo gana en 2 de 5 rondas y la diferencia (0.009) es menor que el desvío (0.017): ahí no se puede concluir. Una diferencia menor que el desvío entre rondas no permite declarar mejor a un criterio.")

md("""
### B5 · La ganancia es la información mutua (puente con la Clase 33) (10 min)
La **ganancia de información** que usamos como filtro en la Clase 33 es el mismo cálculo que el árbol aplica una vez por variable en lugar de recursivamente. Con la entropía como impureza y una variable binaria, la ganancia del corte **es** la información mutua entre la variable y `fuga` (medida en bits):
""")
code("""
sale = (Xi["reclamos"] > 0).values
ganancia_bits = ganancia_corte(sale, yv, "entropia")[2]
mi_bits = mutual_info_score(yv, sale) / np.log(2)         # mutual_info_score usa logaritmo natural (nats)
print(f"Ganancia por entropía del corte reclamos > 0 : {ganancia_bits:.5f} bits")
print(f"Información mutua (mutual_info_score / ln 2)  : {mi_bits:.5f} bits")
""")
md("""
Coinciden. Por eso en la Clase 33 `reclamos` encabezaba el ranking de información mutua y hoy es la raíz del árbol, y por eso `edad` (información mutua casi nula) no ayuda a cortar. El filtro de la Clase 33 es la primera pregunta de un árbol de profundidad 1.

### ✍️ Ejercicio B5
Comprueba la igualdad ganancia por entropía = información mutua para otras dos variables binarias: `estado > 1` y `numero_fallas > 0`. ¿Cuál de las tres variables binarias (`reclamos`, `estado`, `numero_fallas`) aporta más información?
""")
code("""
# tu código aquí
""")
solution("""
candidatos = {"reclamos > 0": (Xi["reclamos"] > 0), "estado > 1": (Xi["estado"] > 1), "numero_fallas > 0": (Xi["numero_fallas"] > 0)}
for nombre, m in candidatos.items():
    m = m.values
    g = ganancia_corte(m, yv, "entropia")[2]
    mi = mutual_info_score(yv, m) / np.log(2)
    print(f"{nombre:18s} ganancia (entropía) = {g:.5f} bits | información mutua = {mi:.5f} bits | iguales: {np.isclose(g, mi)}")
""", "Las dos cantidades coinciden en las tres. `reclamos` aporta el doble de información que `estado` (0.131 contra 0.065 bits) y unas 47 veces más que `numero_fallas > 0` (0.003 bits), que casi no separa a los clientes por sí sola (aunque dentro de la rama de quienes reclamaron y usan poco, `numero_fallas` pasa de 67 % a 94 %: una variable débil sola puede ser fuerte en conjunto).")

# ================================================================== PARTE C
md("""
## Parte C · Sobreajuste, validación cruzada y poda

Un árbol sin límite crece hasta que cada hoja es pura: acierta todo el entrenamiento y **memoriza**. Para medir cuánto generaliza hay que evaluar en datos que no vio. La imputación va **dentro** del `Pipeline`, así que en cada ronda se ajusta solo con los bloques de entrenamiento.

### C1 · Validación cruzada, ronda por ronda (10 min)
Partimos los 3.150 clientes en 5 bloques de 630; entrenamos con 4 (2.520) y medimos el AUC en el restante; rotamos. Importan el **promedio y el desvío**: si las rondas fueran 0.70 y 0.95, un promedio de 0.82 no significaría nada.
""")
code("""
modelo = pipe_arbol(max_depth=5)
res = cross_validate(modelo, X, y, cv=CV, scoring="roc_auc")
for i, ((tr, te), auc) in enumerate(zip(CV.split(X, y), res["test_score"]), 1):
    print(f"ronda {i}: entrena con {len(tr)}, mide en {len(te)} | fuga en el bloque: {y.iloc[te].mean():.3f} | AUC {auc:.4f}")
print(f"\\npromedio {res['test_score'].mean():.4f} | desvío {res['test_score'].std():.4f}")

# ¿Es tan grave no estratificar? Mínimo y máximo de la tasa de fuga en el bloque, con 300 particiones al azar
rng = np.random.default_rng(0)
tasas = [y.iloc[te].mean() for s in range(300) for _, te in KFold(5, shuffle=True, random_state=s).split(X)]
print(f"Sin estratificar (300 particiones): tasa de fuga por bloque entre {min(tasas):.3f} y {max(tasas):.3f}")
""")
md("""
Con `StratifiedKFold` cada bloque conserva el 15.7 % de fuga. Las rondas dan 0.901, 0.920, 0.911, 0.918 y 0.880: **promedio 0.906, desvío 0.015**. Estratificar es una buena práctica, pero la justificación de la clase («al azar, un bloque podría quedar casi sin casos positivos») exagera: con bloques de 630 clientes y 15.7 % de fuga, una partición al azar deja bloques con ≈ 12–20 % de fuga, no «casi sin positivos». Se vuelve crítico con clases mucho más raras o muestras pequeñas.

### ✍️ Ejercicio C1
Repite la validación con `RepeatedStratifiedKFold(5, n_repeats=6)`: ¿cuánto cambia el promedio y el desvío respecto de la partición única de la clase? ¿Qué te dice sobre confiar en una sola partición?
""")
code("""
# tu código aquí
""")
solution("""
rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
s = cross_val_score(pipe_arbol(max_depth=5), X, y, cv=rep, scoring="roc_auc").reshape(6, 5)
print("Promedio de cada repetición:", np.round(s.mean(axis=1), 4))
print(f"Promedio global {s.mean():.4f} | desvío entre las 30 rondas {s.std():.4f}")
""", "El promedio de cada repetición va de ≈ 0.897 a 0.909 y el global queda en ≈ 0.904 (desvío entre las 30 rondas ≈ 0.017): la partición única de la clase (0.906) es representativa, pero según la partición el promedio se mueve ±0.006. Reporta siempre promedio y desvío, y si dos modelos difieren menos que el desvío, no hay un ganador claro.")

md("""
### C2 · La curva de profundidad: sobreajuste a la vista (15 min)
Comparamos el AUC en los **mismos datos de entrenamiento** (ajustando con toda la base, como en clase) con el AUC honesto de validación cruzada, a medida que dejamos crecer el árbol (`max_depth=None` es «sin límite»). La brecha entre ambos es el sobreajuste.
""")
code("""
filas = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    m = pipe_arbol(max_depth=d)
    cvs = auc_cv(m)
    m.fit(X, y)
    filas.append({"profundidad": "sin límite" if d is None else str(d),
                  "AUC train": roc_auc_score(y, m.predict_proba(X)[:, 1]),
                  "AUC validación": cvs.mean(), "desvío": cvs.std(), "hojas": m[-1].get_n_leaves()})
curva = pd.DataFrame(filas)
curva["brecha"] = curva["AUC train"] - curva["AUC validación"]
display(curva.round(4))

mejor = curva["AUC validación"].idxmax()
print(f"Mejor profundidad: {curva.profundidad[mejor]} con AUC {curva['AUC validación'][mejor]:.4f}")
print(f"Sin límite: {curva.hojas.iloc[-1]} hojas, AUC train {curva['AUC train'].iloc[-1]:.3f} y validación {curva['AUC validación'].iloc[-1]:.4f}")

fig, ax = plt.subplots(figsize=(9, 4.2))
xs = range(len(curva))
ax.plot(xs, curva["AUC train"], "o-", color="#C0392B", lw=2, label="AUC en entrenamiento")
ax.plot(xs, curva["AUC validación"], "o-", color="#16A34A", lw=2, label="AUC en validación cruzada")
ax.fill_between(xs, curva["AUC validación"], curva["AUC train"], color="#C0392B", alpha=0.08)
ax.axvline(mejor, color="grey", ls=":")
ax.set_xticks(list(xs)); ax.set_xticklabels(curva["profundidad"])
ax.set_xlabel("Profundidad máxima del árbol"); ax.set_ylabel("AUC"); ax.legend()
plt.tight_layout(); plt.show()
""")
md("""
El AUC de entrenamiento **sube siempre** hasta 1.000; el de validación sube hasta la **profundidad 6 (0.911)** y después baja. Sin límite el árbol tiene 224 hojas (casi una por cada 14 clientes), AUC de entrenamiento 1.000 y de validación 0.827: perdió ≈ 8 puntos por memorizar.

Sobre el criterio del profesor («como mucho unos 2 puntos de diferencia entre entrenamiento y validación»): la profundidad 3 cumple (brecha ≈ 0.006); la 6, que maximiza la validación, tiene una brecha de ≈ 0.03. Hay que decidir si el punto extra de AUC justifica el sobreajuste, o podar. Además, un AUC de 0.90+ «demasiado bueno» en datos reales suele ser **leakage**, no un gran modelo (acá no lo es: `reclamos` es una variable legítima, anterior a la fuga).

### ✍️ Ejercicio C2
La pre-poda también se controla con `min_samples_leaf`. Sin limitar la profundidad, evalúa `min_samples_leaf` ∈ {1, 5, 10, 20, 40, 60, 100, 200}. ¿Cuál maximiza el AUC de validación? ¿Cuántas hojas tiene? Compara con el 60 elegido en clase (que fue para **leer** el árbol, no para optimizar).
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for k in (1, 5, 10, 20, 40, 60, 100, 200):
    m = pipe_arbol(min_samples_leaf=k)
    cvs = auc_cv(m)
    filas.append({"min_samples_leaf": k, "AUC validación": cvs.mean(), "desvío": cvs.std(), "hojas": m.fit(X, y)[-1].get_n_leaves()})
res_k = pd.DataFrame(filas)
display(res_k.round(4))
print("Mejor:", res_k.loc[res_k["AUC validación"].idxmax(), "min_samples_leaf"])
""", "El AUC de validación sube desde 0.827 con hojas de 1 cliente, llega a su máximo con hojas de 20 clientes (AUC 0.919, 59 hojas; con 40 casi igual, 0.9185 y 33 hojas) y vuelve a bajar con hojas muy grandes (200: 0.892, 9 hojas). Es incluso mejor que el mejor `max_depth` (0.911) y que el `ccp_alpha` de C3 (0.909). Con 60 se pierde poco (0.916) y el árbol es más corto y legible (25 hojas): la regla de la diapositiva 12 de la presentación genérica («1–5 % de las observaciones», es decir 32–157 clientes aquí) contradice su propia tabla («5–20»).")

md("""
### C3 · Post-poda: ruta de costo-complejidad y `ccp_alpha` (25 min)
La **pre-poda** frena el crecimiento (el riesgo: un corte pobre arriba puede esconder uno muy bueno debajo). La **post-poda** deja crecer el árbol completo y recorta las ramas que no compensan su costo. scikit-learn usa la poda por **costo-complejidad**: cada hoja «cuesta» α.

$$\\text{costo}_\\alpha(T) = \\text{error}(T) + \\alpha\\cdot|\\text{hojas}(T)|$$

Con α = 0 sobrevive todo el árbol; al subir α, cada hoja debe justificar su existencia. `cost_complexity_pruning_path` entrega los α **donde el árbol óptimo cambia**; después α se elige por validación cruzada, no a ojo. (La ruta se calcula con la base imputada solo para obtener una grilla de candidatos; el α se decide con el `Pipeline` y la validación cruzada.)
""")
code("""
camino = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xi, y)
alphas = camino.ccp_alphas[:-1]               # el último α poda hasta dejar solo la raíz
print(f"α relevantes: {len(alphas)} (de {alphas.min():.5f} a {alphas.max():.5f})")

print()
print("  α        hojas   AUC validación")
tabla_alpha = []
for a in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01):
    m = pipe_arbol(ccp_alpha=a)
    cvs = auc_cv(m)
    hojas_a = m.fit(X, y)[-1].get_n_leaves()
    tabla_alpha.append((a, hojas_a, cvs.mean()))
    print(f"  {a:<7}  {hojas_a:5d}   {cvs.mean():.4f}")
""")
code("""
grilla = GridSearchCV(pipe_arbol(), {"arbol__ccp_alpha": alphas[::5]}, cv=CV, scoring="roc_auc", n_jobs=-1).fit(X, y)
alfa_opt = grilla.best_params_["arbol__ccp_alpha"]
podado = grilla.best_estimator_
print(f"Mejor ccp_alpha: {alfa_opt:.5f} | AUC en CV: {grilla.best_score_:.4f} | hojas: {podado[-1].get_n_leaves()}")

r = pd.DataFrame(grilla.cv_results_)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.errorbar(r["param_arbol__ccp_alpha"].astype(float), r["mean_test_score"], yerr=r["std_test_score"], fmt="o-", ms=3, lw=1, capsize=2)
ax.axvline(alfa_opt, color="red", ls="--", lw=1)
ax.set_xscale("symlog", linthresh=1e-4)
ax.set_xlabel("ccp_alpha (escala simétrica-log)"); ax.set_ylabel("AUC validación (± desvío)")
plt.tight_layout(); plt.show()
""")
md("""
La búsqueda elige **α = 0.0013** con **36 hojas** y AUC de validación **0.909**. En la tabla de la clase el «punto dulce» es α = 0.001: de 224 hojas a 42 y el AUC sube de 0.827 a ≈ 0.90. Más poda (α = 0.005 o 0.01: 5 o 3 hojas) deja un árbol demasiado simple y el AUC cae a 0.83–0.81. Y fíjate en la barra de error: varias α vecinas del óptimo son estadísticamente indistinguibles.

La diapositiva 19 muestra como resultado `{'ccp_alpha': 0.001} 0.8996, hojas: 42`, pero el código que muestra (`alphas[::5]`) no produce ese valor: el 0.8996 corresponde a la fila α = 0.001 de la tabla de la diapositiva 18 (aquí 0.898 por la imputación dentro del pipeline).

### ✍️ Ejercicio C3
La **regla del error estándar** elige el árbol más simple cuya validación esté a menos de 1 error estándar (desvío / √5) del mejor. Con `grilla.cv_results_`, ¿qué α elegiría esa regla y cuántas hojas tiene? ¿Cuánto AUC cedes frente al óptimo? ¿Cuál presentarías a gerencia?
""")
code("""
# tu código aquí
""")
solution("""
r = pd.DataFrame(grilla.cv_results_)
r["alpha"] = r["param_arbol__ccp_alpha"].astype(float)
mejor_fila = r.loc[r["mean_test_score"].idxmax()]
se = mejor_fila["std_test_score"] / np.sqrt(5)
umbral = mejor_fila["mean_test_score"] - se
elegible = r[r["mean_test_score"] >= umbral]
fila_1se = elegible.loc[elegible["alpha"].idxmax()]             # α más grande = árbol más simple
hojas_1se = pipe_arbol(ccp_alpha=fila_1se["alpha"]).fit(X, y)[-1].get_n_leaves()
print(f"Óptimo: α = {mejor_fila['alpha']:.5f}, AUC {mejor_fila['mean_test_score']:.4f}, {podado[-1].get_n_leaves()} hojas")
print(f"1-SE:   α = {fila_1se['alpha']:.5f}, AUC {fila_1se['mean_test_score']:.4f}, {hojas_1se} hojas (umbral {umbral:.4f})")
""", "La regla del error estándar elige α ≈ 0.0014 y un árbol de **25 hojas** con AUC 0.906: cede solo 0.004 de AUC (0.909 contra 0.906) y quita 11 hojas. Sigue siendo grande para una lámina: para gerencia puede convenir un árbol aún más pequeño (el de 15 hojas de α = 0.002 tiene 0.883; el de profundidad 3, 7 hojas, 0.850); si el objetivo es solo predecir, el de 36 hojas (o directamente un bosque).")

# ================================================================== PARTE D
md("""
## Parte D · Interpretar, límites y el salto al bosque

### D1 · Importancia: impureza frente a permutación (20 min)
`feature_importances_` suma, para cada variable, la reducción de impureza de todos los cortes donde participó. Tiene un sesgo: una variable con **muchos valores distintos** ofrece más umbrales y, por azar, alguno reduce la impureza. La **importancia por permutación** desordena una columna **en datos que el modelo no vio** y mide cuánto cae el AUC. Usamos un bosque (como en clase) porque su importancia es más estable que la de un solo árbol.
""")
code("""
bosque = pipe_bosque(n_estimators=300).fit(Xtr, ytr)
imp_imp = pd.Series(bosque[-1].feature_importances_, index=VARS)

pi_te = permutation_importance(bosque, Xte, yte, n_repeats=20, random_state=0, scoring="roc_auc")
pi_tr = permutation_importance(bosque, Xtr, ytr, n_repeats=5, random_state=0, scoring="roc_auc")

comp = pd.DataFrame({"impureza": imp_imp, "permutación (test)": pi_te.importances_mean,
                     "permutación (train)": pi_tr.importances_mean, "valores únicos": X.nunique()})
comp = comp.sort_values("impureza", ascending=False)
comp["puesto impureza"] = np.arange(1, len(comp) + 1)
comp["puesto permutación"] = comp["permutación (test)"].rank(ascending=False).astype(int)
display(comp.round(4))
print("AUC del bosque: train", round(roc_auc_score(ytr, bosque.predict_proba(Xtr)[:, 1]), 4),
      "| test", round(roc_auc_score(yte, bosque.predict_proba(Xte)[:, 1]), 4))
""")
md("""
Lo que se ve: `reclamos` encabeza ambos rankings. `clv` (2.654 valores distintos) está **5.º por impureza** (0.109) pero **7.º por permutación** (0.014), y `estado` (binaria) pasa del puesto 10 al 6 (impureza 0.045, permutación 0.018): un patrón consistente con el sesgo por cardinalidad, aunque con variables reales es modesto (`antiguedad`, con solo 47 valores, es 4.ª por impureza y también importante por permutación). Dos precisiones sobre lo que dice el material: (1) la diapositiva 21 omite `antiguedad` de su tabla y el profesor dice que `clv` es «la cuarta más alta» por impureza, cuando con los datos es la quinta; (2) «clv y edad no importan» es exagerado: `clv` cuesta 0.014 de AUC al desordenarla, más que `n_contactos` o `frec_sms`.

La prueba limpia del sesgo es agregar una **variable de ruido puro** con muchos valores (nada que ver con la fuga):
""")
code("""
rng = np.random.default_rng(0)
Xtr_r, Xte_r = Xtr.copy(), Xte.copy()
Xtr_r["ruido"] = rng.normal(size=len(Xtr_r))
Xte_r["ruido"] = rng.normal(size=len(Xte_r))

bosque_r = pipe_bosque(n_estimators=300).fit(Xtr_r, ytr)
imp_r = pd.Series(bosque_r[-1].feature_importances_, index=Xtr_r.columns)
per_r = permutation_importance(bosque_r, Xte_r, yte, n_repeats=20, random_state=0, scoring="roc_auc")
per_r = pd.Series(per_r.importances_mean, index=Xtr_r.columns)

cmp_r = pd.DataFrame({"impureza": imp_r, "permutación (test)": per_r}).sort_values("impureza", ascending=False)
cmp_r["puesto impureza"] = np.arange(1, len(cmp_r) + 1)
display(cmp_r.round(4))
print("Ruido puro -> impureza:", round(imp_r["ruido"], 4), "(más que estado:", round(imp_r["estado"], 4), ") | permutación:", round(per_r["ruido"], 4))
""")
md("""
La variable de **ruido puro** recibe importancia por impureza ≈ 0.065 (puesto 8 de 13, **por encima de `estado`**, `frec_sms` y `monto`, que sí son señal) y por permutación ≈ 0 (incluso ligeramente negativa). Ese es el sesgo, medido.

Otra imprecisión del material: la diapositiva 22 etiqueta «SESGADA: mide sobre el entrenamiento / CORRECTA: mide sobre datos no vistos», mezclando **dos problemas distintos**. El sesgo de la impureza es por **cardinalidad** (no desaparece evaluando en el test: se calcula con el árbol ya construido). Y calcular la permutación sobre el train es otro error: en la tabla, con el bosque memorizado (AUC de train ≈ 1.000), el ranking cambia (`segundos` cae al 5.º puesto y `estado` sube al 2.º).

### ✍️ Ejercicio D1
«Usa **siempre** importancia por permutación» (diapositiva 21) tiene una excepción: las variables **correlacionadas**. Agrega a los datos de entrenamiento y de test una copia exacta de `segundos` (`segundos_copia = 2·segundos + 1`), reentrena el bosque y calcula la permutación. ¿Qué pasa con la importancia de `segundos`? ¿Qué recomendarías para interpretar?
""")
code("""
# tu código aquí
""")
solution("""
Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
for d in (Xtr_c, Xte_c):
    d["segundos_copia"] = 2 * d["segundos"] + 1

bosque_c = pipe_bosque(n_estimators=300).fit(Xtr_c, ytr)
pc = permutation_importance(bosque_c, Xte_c, yte, n_repeats=20, random_state=0, scoring="roc_auc")
pc = pd.Series(pc.importances_mean, index=Xtr_c.columns)
print("Antes   -> segundos:", round(comp.loc["segundos", "permutación (test)"], 4))
print("Después -> segundos:", round(pc["segundos"], 4), "| segundos_copia:", round(pc["segundos_copia"], 4), "| suma:", round(pc["segundos"] + pc["segundos_copia"], 4))
""", "La importancia de `segundos` (≈ 0.033) se reparte con su copia: ≈ 0.012 y ≈ 0.014. Cada una parece «poco importante» porque, al desordenar una, el modelo se apoya en la otra. Con variables correlacionadas conviene agruparlas (desordenar el grupo completo), eliminar la redundancia antes de interpretar, o complementar con otras herramientas; y recordar que ninguna importancia es causalidad ni dice si el efecto es positivo o negativo (para eso, las reglas o las dependencias parciales).")

md("""
### D2 · Escalar o transformar: el árbol es invariante (10 min)
Un árbol solo pregunta «¿x ≤ umbral?». Cualquier transformación **monótona creciente** g mueve el umbral pero no cambia qué clientes caen a cada lado: $x_i \\le u \\iff g(x_i) \\le g(u)$. Por eso el resultado es **idéntico**, no parecido. Lo comprobamos aplicando log a `segundos` y raíz a `clv`; la logística, en cambio, sí cambia.
""")
code("""
X2 = X.copy()
X2["segundos"] = np.log1p(X2["segundos"].clip(lower=0))
X2["clv"] = np.sqrt(X2["clv"])

print(f"{'':26s} {'original':>9} {'log y raíz':>11} {'diferencia':>11}")
for nombre, m in [("Árbol (profundidad 5)", pipe_arbol(max_depth=5)), ("Regresión logística", pipe_logistica())]:
    a1, a2 = auc_cv(m).mean(), auc_cv(m, X2).mean()
    print(f"{nombre:26s} {a1:9.4f} {a2:11.4f} {a2 - a1:+11.4f}")

p1 = pipe_arbol(max_depth=5).fit(X, y).predict_proba(X)
p2 = pipe_arbol(max_depth=5).fit(X2, y).predict_proba(X2)
print("Predicciones del árbol idénticas en todos los clientes:", np.array_equal(p1, p2))
""")
md("""
El árbol da **exactamente 0.9058** en los dos casos (y las probabilidades por cliente coinciden); la logística pasa de 0.9199 a 0.9231. Por eso con árboles no hace falta `StandardScaler` (ni sirve de nada ponerlo) y los valores atípicos en las variables predictoras molestan poco. El umbral de `segundos` ≤ 934 se vuelve ≈ 6.84 = log(1 + 934) y ningún cliente cambia de lado.

### ✍️ Ejercicio D2
La invariancia vale solo para transformaciones **monótonas**. Aplica a `segundos` la transformación no monótona `(segundos − 3000)²` y evalúa el árbol de profundidad 5 con la misma validación cruzada. ¿Se mantiene el 0.9058? ¿Por qué cambia?
""")
code("""
# tu código aquí
""")
solution("""
X3 = X.copy()
X3["segundos"] = (X3["segundos"] - 3000) ** 2
print("AUC original      :", round(auc_cv(pipe_arbol(max_depth=5)).mean(), 4))
print("AUC (seg - 3000)² :", round(auc_cv(pipe_arbol(max_depth=5), X3).mean(), 4))
""", "Cambia (0.9058 → 0.8956): (x − 3000)² no respeta el orden, así que clientes con 1.000 y con 5.000 segundos quedan igual de «lejos» de 3.000 y se mezclan en el mismo lado del umbral. Si la transformación no preserva el orden, la partición es otra.")

md("""
### D3 · Inestabilidad del árbol (10 min)
Un árbol tiene **alta varianza**: cambiar unas pocas observaciones cambia los cortes. Entrenamos seis árboles con seis muestras distintas del 70 % de los datos (estratificadas) y miramos la raíz y el corte de más abajo a la izquierda.
""")
code("""
filas = []
for semilla in range(6):
    Xs, _, ys, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semilla)
    m = pipe_arbol(max_depth=4).fit(Xs, ys)
    e = m[-1].tree_
    nodo = e.children_left[0]
    filas.append({"semilla": semilla, "raíz": VARS[e.feature[0]], "nivel 2 (izq.)": VARS[e.feature[nodo]], "umbral": round(e.threshold[nodo], 1)})
display(pd.DataFrame(filas))
""")
md("""
La raíz es **siempre `reclamos`** (su ventaja es demasiado grande), pero el nivel siguiente cambia con la muestra: `clv`, `segundos`, `frec_llamadas`… Con reentrenamientos trimestrales, eso es explicarle al negocio un árbol distinto cada vez. Es el problema que el bosque viene a resolver.

### ✍️ Ejercicio D3
Con 30 semillas en lugar de 6, ¿con qué frecuencia aparece cada variable en el nivel 2 izquierdo? ¿Y con árboles más acotados (`min_samples_leaf=60`), el panorama es más estable?
""")
code("""
# tu código aquí
""")
solution("""
def nivel2(**kw):
    out = []
    for semilla in range(30):
        Xs, _, ys, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semilla)
        e = pipe_arbol(max_depth=4, **kw).fit(Xs, ys)[-1].tree_
        out.append(VARS[e.feature[e.children_left[0]]])
    return pd.Series(out).value_counts()


print("Sin min_samples_leaf:"); print(nivel2().to_string())
print("\\nCon min_samples_leaf = 60:"); print(nivel2(min_samples_leaf=60).to_string())
""", "En las 30 muestras el nivel 2 izquierdo es `segundos` en 17, `frec_llamadas` en 9 y `clv` en 4: ninguna variable gana siempre. Con `min_samples_leaf=60` los conteos son exactamente los mismos: el límite estabiliza las tasas de las hojas, pero no el corte de segundo nivel (esos nodos son grandes y el límite no los afecta).")

md("""
### D4 · Puente al bosque, sin repetir la Clase 35 (15 min)
Si el problema del árbol es la varianza, la solución es **promediar muchos árboles**. La receta de un Random Forest tiene tres ingredientes: (1) una **muestra bootstrap** por árbol (n extracciones con reemplazo; ≈ un tercio de los clientes queda fuera de cada una: el *out-of-bag*); (2) en **cada nodo** solo se consideran √p variables (con p = 12, unas 3); (3) se promedian las probabilidades. La varianza del promedio de B árboles con varianza σ² y correlación ρ es $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$: más árboles ayudan, pero el piso es ρσ², por eso el paso (2) busca bajar ρ. La Clase 35 verifica todo esto desde cero; aquí solo medimos lo que importa para decidir.
""")
code("""
print("Árboles | AUC (CV)")
for nt in (1, 5, 25, 100, 300):
    print(f"{nt:7d} | {auc_cv(pipe_bosque(n_estimators=nt)).mean():.4f}")

print("\\nFuente de azar                                   | AUC (CV)")
for nom, kw in [("solo bootstrap (todas las variables por corte)", dict(max_features=None)),
                ("bootstrap + subconjunto de variables (default)", dict(max_features="sqrt")),
                ("sin bootstrap, solo subconjunto de variables", dict(bootstrap=False, max_features="sqrt"))]:
    print(f"{nom:49s}| {auc_cv(pipe_bosque(n_estimators=200, **kw)).mean():.4f}")

print("\\nVariables por corte con 'sqrt':", int(np.sqrt(len(VARS))), "de", len(VARS))
""")
md("""
Un solo árbol del bosque rinde **0.797**, peor que un árbol podado; con 5 árboles ya sube a 0.93, con 25 a 0.97 y se estabiliza cerca de 100 (0.973 → 0.974 con 300): **la ganancia no está en los árboles, está en promediarlos**. Cada fuente de azar aporta, y juntas dan lo mejor (0.974).

Una precisión sobre la presentación genérica (diapositiva 17), que dice que promediar árboles descorrelacionados «reduce la varianza **sin aumentar el sesgo**»: cada árbol del bosque es individualmente **peor** que un árbol completo (0.797 contra 0.827), porque ve una muestra y variables restringidas; el promedio lo compensa con creces, pero no es gratis.

### ✍️ Ejercicio D4
Con `n_estimators` ∈ {10, 25, 50, 100, 150, 200, 300}, ¿cuál es el menor número de árboles que queda a menos de 0.001 de AUC del valor con 300? Y la afirmación del profesor «hoy es lo mismo hacer un árbol que un bosque, no demora nada»: mide el tiempo de ajuste de un árbol y de un bosque de 300 árboles (con `n_jobs=-1`) y su predicción.
""")
code("""
# tu código aquí
""")
solution("""
aucs = {nt: auc_cv(pipe_bosque(n_estimators=nt)).mean() for nt in (10, 25, 50, 100, 150, 200, 300)}
print({k: round(v, 4) for k, v in aucs.items()})
print("Menor número de árboles a menos de 0.001 del de 300:", min(k for k, v in aucs.items() if aucs[300] - v < 0.001))

for nombre, m in [("árbol de profundidad 6", pipe_arbol(max_depth=6)), ("bosque de 300 árboles", pipe_bosque(n_estimators=300))]:
    t0 = perf_counter(); m.fit(Xtr, ytr); t_fit = perf_counter() - t0
    t0 = perf_counter(); m.predict_proba(Xte); t_pred = perf_counter() - t0
    print(f"{nombre:24s} ajuste {t_fit:.3f} s | predicción {t_pred:.3f} s")
""", "Con ≈ 100 árboles el AUC ya está a menos de 0.001 del de 300 (el profesor decía «se estabiliza cerca de 100»). Sobre los tiempos: el bosque es **decenas de veces más lento** que un árbol, tanto en ajuste como en predicción (las cifras exactas dependen de tu equipo). Con 3.150 filas sigue siendo cuestión de segundos, así que el costo computacional rara vez decide; lo que sí se pierde es la **interpretabilidad** (un árbol se dibuja, 300 no). «Es lo mismo» es falso: lo que cambió es que el costo ya no es el argumento principal.")

md("""
### D5 · Comparación final: ¿para qué es el modelo? (10 min)
Todos los modelos de la clase con la misma validación cruzada (imputación dentro del `Pipeline` en todos):
""")
code("""
modelos = {
    "Árbol profundidad 3": pipe_arbol(max_depth=3),
    "Árbol profundidad 5": pipe_arbol(max_depth=5),
    "Árbol podado (ccp_alpha por CV)": pipe_arbol(ccp_alpha=alfa_opt),
    "Regresión logística": pipe_logistica(),
    "Random Forest (300)": pipe_bosque(n_estimators=300),
}
filas = []
for nombre, m in modelos.items():
    s = auc_cv(m)
    filas.append({"modelo": nombre, "AUC (CV)": s.mean(), "desvío": s.std()})
display(pd.DataFrame(filas).sort_values("AUC (CV)", ascending=False).round(4))
""")
md("""
Lectura: la **logística (0.920) supera al mejor árbol (0.909)** y el bosque (0.974) supera a ambos por mucho: **un árbol solo suele predecir peor que una logística**, y los ensambles ganan en desempeño. Entonces, ¿cuándo un árbol? *Cuando la explicación vale más que el punto de AUC*: el árbol de profundidad 3 cabe en una lámina y se traduce a 7 reglas con n, tasa y lift; el bosque de 300 árboles no. Es el mismo trade-off de la Clase 32 (PCA): se cede interpretabilidad a cambio de desempeño. (Nota de método: el α del árbol podado se eligió con esta misma validación cruzada, así que su 0.909 es algo optimista; el ejercicio de cierre lo hace con un test reservado.) El camino natural es el que sugiere la presentación genérica: árbol podado como línea base → Random Forest (**Clase 35**) → boosting (**Clase 36**) si el desempeño lo justifica.

### ✍️ Ejercicio D5
Con los datos de train/test reservados (`Xtr`, `Xte`), ajusta el árbol podado (con el mismo α), la logística y el bosque de 300 árboles. Si el equipo de retención solo puede contactar al **10 % de los clientes del test** con mayor probabilidad de fuga, ¿qué fracción de los fugados captura cada modelo? ¿Qué problema tiene el árbol para ordenar a los clientes?
""")
code("""
# tu código aquí
""")
solution("""
def captura_top(m, frac=0.10):
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    k = int(round(frac * len(yte)))
    top = np.argsort(-p, kind="stable")[:k]
    return yte.values[top].sum() / yte.sum(), len(np.unique(p))


for nombre, m in [("Árbol podado", pipe_arbol(ccp_alpha=alfa_opt)), ("Regresión logística", pipe_logistica()),
                  ("Random Forest (300)", pipe_bosque(n_estimators=300))]:
    cap, n_prob = captura_top(m)
    print(f"{nombre:20s} captura {100 * cap:5.1f} % de los fugados en el 10 % superior | probabilidades distintas: {n_prob}")
""", "El bosque captura ≈ 54 % de los fugados en el 10 % superior; el árbol podado ≈ 46 % y la logística ≈ 45 %. El árbol solo produce **tantas probabilidades distintas como hojas** (19 para 788 clientes): dentro de una hoja todos empatan y el orden entre ellos es arbitrario, por lo que rankea peor, y un 10 % exacto puede cortar una hoja por la mitad.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene corregir

En las diapositivas, la clase y el notebook del profesor hay imprecisiones que verificamos (la mayoría con código en este mismo notebook):

| Dónde | Qué dice | Qué es correcto |
|---|---|---|
| S34, diapositiva 13 (nota) | A da «Gini ponderado ~0,168 y ganancia ~0,097» | Con los datos de la diapositiva da **0.195 y 0.070**, igual que la diapositiva 12 (0.0693). Además usa 211 y 284 fugas; los datos reales son 210 y 285 (B2) |
| S34, diapositiva 13 | Candidato B: `edad > 40` con 1.559 / 1.591 clientes y 240 / 255 fugas | Con los datos, `edad > 40` deja 970 / 2.180 clientes con 147 / 348 fugas; la ganancia sigue siendo ≈ 0, pero los conteos no salen de la base (B2) |
| S34, diapositiva 19 | El código muestra `{'ccp_alpha': 0.001}`, AUC 0.8996, 42 hojas | La búsqueda con `alphas[::5]` elige **α = 0.0013, 36 hojas, AUC 0.909**. El 0.8996 es la fila α = 0.001 de la diapositiva 18 (C3) |
| S34, diapositivas 25 y 27 | «El mejor árbol solo 0,911» y «solo llega a 0,906» | No se contradicen: 0.911 es la profundidad 6 y 0.906 la profundidad 5 (C2) |
| S34, diapositiva 11 | La fila 84 % / 16 % muestra Gini 0.269 | Por redondeo; el valor exacto de la raíz es **0.265** (B1). En la transcripción el profesor dice «0,2669» |
| S34, diapositiva 12 y notebook | El barrido con 60 percentiles da «el mismo resultado»; la tabla muestra las 12 variables | El ganador es el mismo, pero con el barrido exacto `clv`, `segundos` y `frec_sms` obtienen más ganancia. La tabla omite `frec_sms`, `monto` y `tipo_plan` (B3) |
| Notebook y presentación genérica | Gini y entropía «dan árboles casi idénticos» | En profundidad 5 la entropía da +1.4 puntos de AUC (0.920 contra 0.906) y gana en las 5 rondas (B4) |
| Clase (transcripción) | «Entre más Gini, más voy a tener mayor ganancia» | La ganancia es una **reducción** de impureza: los hijos con *más* Gini dejan *menos* ganancia (`edad > 40`: hijos con el Gini del padre, ganancia ≈ 0.00003; `reclamos`: hijos con 0.37 y 0.18, ganancia 0.069) (B2) |
| S34, diapositiva 16 | Sin estratificar, un bloque «podría quedar casi sin casos positivos» | Con bloques de 630 y 15.7 % de fuga, al azar el bloque queda entre ≈ 12 % y 20 %; estratificar es buena práctica, pero la justificación exagera (C1) |
| Presentación genérica, diap. 12 | `min_samples_leaf` «5–20» en la tabla y «1–5 % de las observaciones» en la nota | Con 3.150 clientes, 1–5 % son 32–157: las dos reglas no coinciden (C2) |
| S34 diap. 21 y notebook | Tabla sin `antiguedad`; «clv es la cuarta más alta por impureza»; «clv y edad no importan» | Con los datos `antiguedad` (0.113) está sobre `clv` (0.109) y `clv` es 5.º; por permutación `clv` pierde 0.014 de AUC, no es irrelevante (D1) |
| S34 diap. 22 y notebook | «SESGADA: mide sobre el entrenamiento / CORRECTA: datos no vistos»; «usá siempre permutación» | Son dos problemas distintos: el sesgo por cardinalidad (impureza) y calcular la permutación en el train. Con variables correlacionadas la permutación también reparte la importancia (D1) |
| Presentación genérica, diap. 17 | Promediar árboles «reduce la varianza sin aumentar el sesgo» | Cada árbol del bosque es individualmente peor que uno completo (0.797 contra 0.827): el promedio lo compensa, pero no es gratis (D4) |
| Clase | «Hoy es lo mismo hacer un árbol que un bosque, no demora nada» | El bosque es decenas de veces más lento y pierde la lectura de reglas; el costo ya no decide, pero no es lo mismo (D4) |
| Presentación genérica, diap. 8 | CART: faltantes con «sustitutos (surrogates)»; «scikit-learn» como uso actual | scikit-learn **no** implementa surrogates; desde la 1.3 los árboles aceptan `NaN` mandándolos al lado que más reduce la impureza (verificación abajo) |
| Notebook | Carga `churn_telco.csv` y monta Google Drive | Ese archivo no existe en la carpeta; usa el Excel de la Clase 33 |

**Transcripción con ruido**: bucles de Whisper (líneas 855–857 y 893–901), un fragmento ajeno a la clase sobre cómo pausar OBS Studio (líneas 1085–1101) y términos mal reconocidos: «Spirman» (Spearman), «la nova» (ANOVA), «vernugis» (Bernoulli), «araso» (Lasso), «Scorchart» (scorecard), «Attention y Solve Unit» (el paper «Attention Is All You Need», 2017), «rock/ROC» (AUC-ROC) y «Gini KS/Leaf/Regini» (Gini, KS, lift).
""")
code("""
# Verificación rápida de las alertas que no tienen su propia celda arriba
print("Gini del 84/16 redondeado:", round(gini_p(0.16), 4), "| de la raíz real:", round(gini_p(y.mean()), 4))

# Un árbol de scikit-learn acepta NaN directamente (sin imputar); no usa 'surrogates'
arbol_nan = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xtr, ytr)
print("Árbol ajustado con NaN sin imputar -> hojas:", arbol_nan.get_n_leaves(),
      "| AUC test:", round(roc_auc_score(yte, arbol_nan.predict_proba(Xte)[:, 1]), 4),
      "| NaN en Xtr:", int(Xtr.isna().sum().sum()))

# La ganancia es una reducción: hijos con la misma mezcla del padre -> ganancia 0
al_azar = np.random.default_rng(0).random(len(yv)) < 0.5
print("Ganancia de un corte al azar (hijos parecidos al padre):", round(ganancia_corte(al_azar, yv)[2], 5))
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (25 min, SIN mirar lo anterior)

Eres el analista y debes entregar un árbol de fuga **sin tocar el test hasta el final**:

1. Con `Xtr` y `ytr` define un `Pipeline` (imputación por mediana + `DecisionTreeClassifier(random_state=0)`).
2. Busca con `GridSearchCV` (5 bloques estratificados, `roc_auc`) la combinación de `min_samples_leaf` ∈ {1, 5, 20, 60} y `ccp_alpha` ∈ {0, 0.0005, 0.001, 0.002, 0.005}. Reporta los mejores parámetros, el AUC de CV y el número de hojas.
3. Evalúa **una sola vez** en `Xte`: el AUC del árbol, el de una regresión logística y el de un Random Forest de 300 árboles.
4. Traduce a reglas de negocio las **dos hojas de mayor tasa de fuga** del árbol elegido (n, tasa y lift calculados en el entrenamiento).
5. Calcula la importancia por permutación del árbol elegido **en el test** (20 repeticiones) y nombra las 3 variables más importantes.
6. Responde sin código: ¿qué dos cosas del material del profesor corregirías antes de dárselo al área de retención?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
busq = GridSearchCV(pipe_arbol(), {"arbol__min_samples_leaf": [1, 5, 20, 60],
                                   "arbol__ccp_alpha": [0, 0.0005, 0.001, 0.002, 0.005]},
                    cv=cv5, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
elegido = busq.best_estimator_
print("Mejores parámetros:", busq.best_params_, "| AUC CV:", round(busq.best_score_, 4), "| hojas:", elegido[-1].get_n_leaves())

print("\\nAUC en el test (una sola vez):")
print("  Árbol elegido      :", round(roc_auc_score(yte, elegido.predict_proba(Xte)[:, 1]), 4))
print("  Regresión logística:", round(roc_auc_score(yte, pipe_logistica().fit(Xtr, ytr).predict_proba(Xte)[:, 1]), 4))
print("  Random Forest (300):", round(roc_auc_score(yte, pipe_bosque(n_estimators=300).fit(Xtr, ytr).predict_proba(Xte)[:, 1]), 4))

print("\\nDos hojas de mayor fuga (entrenamiento):")
display(tabla_hojas(elegido, Xtr, ytr).head(2)[["clientes", "tasa", "lift", "regla"]].style.format({"tasa": "{:.1%}", "lift": "{:.1f}"}))

pi_arbol = permutation_importance(elegido, Xte, yte, n_repeats=20, random_state=0, scoring="roc_auc")
print("Importancia por permutación (test), top 3:")
print(pd.Series(pi_arbol.importances_mean, index=VARS).sort_values(ascending=False).head(3).round(4).to_string())
""", "La búsqueda elige `min_samples_leaf=20` y `ccp_alpha=0` (46 hojas, AUC de CV 0.909). En el test: árbol 0.898, logística 0.906 y Random Forest 0.972. Las dos hojas de mayor fuga son segmentos **pequeños** (58 y 24 clientes, 100 % y 96 % de fuga, lift ≈ 6), con reglas largas del tipo «reclamó, usa 5.778 segundos o menos, tiene más de 7 fallas y antigüedad de 37 meses o menos» (la regla cruda repite `numero_fallas > 0.5` y `> 7.5`: se simplifica a la más estricta); con tan pocos clientes la tasa es frágil, la razón para un `min_samples_leaf` mayor. Por permutación en el test las tres variables más importantes son `segundos` (0.094), `reclamos` (0.088) y `n_contactos` (0.052): en un árbol solo la importancia es más concentrada e inestable que en un bosque. Correcciones sugeridas antes de entregar: (1) la nota de la diapositiva 13 (ganancia ~0.097) y los conteos de `edad > 40` no salen de los datos, y la diapositiva 19 muestra un resultado (0.001 / 0.8996 / 42 hojas) que su código no produce; (2) «usa siempre permutación» y «SESGADA = entrenamiento» mezclan dos problemas y fallan con variables correlacionadas. Otras válidas: «Gini y entropía casi idénticos» (en profundidad 5 hay 1.4 puntos), «es lo mismo un árbol que un bosque» o la tabla de la diapositiva 12 con 9 de las 12 variables.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué diferencia hay entre lo que predice un árbol y lo que predice una regresión logística? ¿Qué es una hoja?
2. Calcula Gini y entropía de un nodo con 15 clientes de una clase y 5 de la otra, y explica por qué se pondera por el tamaño de los hijos al calcular la ganancia.
3. ¿Por qué se dice que el árbol es «voraz» y qué consecuencia tiene?
4. ¿Qué muestra la curva de profundidad, y cómo eliges el punto adecuado? ¿Qué es la brecha?
5. ¿Qué diferencia hay entre pre-poda y post-poda? ¿Qué hace `ccp_alpha` y cómo se elige?
6. ¿Por qué engaña la importancia por impureza, y qué problema tiene la importancia por permutación con variables correlacionadas?
7. ¿Por qué un árbol no necesita escalar, y qué transformaciones lo cambian?
8. Tu árbol saca 0.99 en entrenamiento y 0.71 en test. ¿Qué haces y en qué orden?
9. ¿Cuándo eliges un árbol en lugar de un Random Forest o una logística?

<details><summary><b>Respuestas</b></summary>

1. La logística ajusta un plano con coeficientes; el árbol parte el espacio en rectángulos y en cada **hoja** (región) predice la tasa de fuga de los clientes de entrenamiento que cayeron allí.
2. p = 0.75 → Gini = 1 − 0.5625 − 0.0625 = 0.375; entropía = 0.311 + 0.5 = 0.811. Se pondera porque un hijo puro con 2 clientes no equivale a uno puro con 2.000: sin ponderar, el árbol premiaría cortes que aíslan unas pocas filas.
3. Elige el mejor corte en cada nodo sin mirar los siguientes: es rápido pero no garantiza el árbol óptimo; un corte malo arriba condiciona todo lo de abajo.
4. Que el AUC de entrenamiento sube siempre y el de validación sube hasta un punto (aquí profundidad 6) y luego baja. Se elige el máximo de validación (idealmente con una brecha tolerable, ~2 puntos según el profesor, o podando). La brecha es AUC de train − AUC de validación: mide el sobreajuste.
5. La pre-poda limita el crecimiento (`max_depth`, `min_samples_leaf`…); la post-poda deja crecer y recorta con `ccp_alpha`, el «precio» por hoja del costo error + α·hojas. Se elige con `cost_complexity_pruning_path` y validación cruzada (o la regla de 1 error estándar), no a ojo.
6. La impureza premia variables con muchos valores distintos (un ruido puro obtuvo más importancia que `estado`), y la permutación se debe calcular en datos no vistos. Con variables correlacionadas, la permutación reparte la importancia entre ellas (cada una parece menos importante) y ninguna importancia es causalidad.
7. Solo pregunta «¿x ≤ umbral?»: una transformación **monótona creciente** mueve el umbral pero no la partición (AUC idéntico). Una transformación no monótona sí la cambia.
8. Sobreajuste. Primero regularizar (`max_depth`, `min_samples_leaf`) y trazar la curva de validación con validación cruzada; después podar con `ccp_alpha` validado; solo si sigue siendo insuficiente, pasar a un ensamble.
9. Cuando la explicación vale más que el punto de AUC (reglas que el negocio entiende y audita, segmentos accionables). Si solo importa predecir, un bosque o boosting; una logística bien especificada suele superar a un árbol solo.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 35 · Bagging y Random Forest (el bosque que hoy solo vimos de lejos: bootstrap, OOB, `max_features` y la varianza del promedio, desde cero). Revisa `Guias/apuntes/C34_Ejercicios_Cuaderno_ArbolesDecision.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
