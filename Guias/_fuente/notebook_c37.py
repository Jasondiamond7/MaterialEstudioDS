"""Genera el notebook guiado de SVM y clasificadores clásicos (Clase 37).

Uso normal:      python notebook_c37.py             -> Guias/notebooks/C37_SVM.ipynb
Modo de prueba:  C37_TEST=1 python notebook_c37.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
"""
import json
import os
from pathlib import Path

TEST = os.environ.get("C37_TEST") == "1"
DIR = Path(__file__).resolve().parent.parent / "notebooks"
OUT = DIR / ("_C37_SVM_test.ipynb" if TEST else "C37_SVM.ipynb")
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
# Clase 37 · SVM y clasificadores clásicos
**Diplomado Data Science** · Prof. Pedro Luiz Ramos

Material de apoyo:
- `Guias/C37_Guia_SVM.pdf` — guía visual de la clase.
- `Guias/apuntes/C37_Apuntes_SVM.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C37_Ejercicios_Cuaderno_SVM.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta de la Clase 37): `clase_svm.pdf` (diapositivas), `clasificadores.py`, `clasificadores.R.txt` y `wisconsin_clase.csv`.
- Transcripción: `VideosClases/Clase 37 - SVM y clasificadores clasicos.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `scikit-learn`, `pandas`, `numpy` y `matplotlib`. Si no los tienes:
```
pip install scikit-learn pandas numpy matplotlib
```
**Datos**: `wisconsin_clase.csv`, 569 muestras del *Breast Cancer Wisconsin (Diagnostic)* con cuatro predictores (`radio`, `textura`, `suavidad`, `concavidad`), la respuesta `maligno` (1 = maligno, 0 = benigno) y la columna `conjunto`, que **ya trae la partición del profesor**: 398 casos de entrenamiento y 171 de prueba. El notebook busca el archivo en la carpeta de la Clase 37. Es una base histórica de una región de Estados Unidos: sirve para enseñar, no es una herramienta clínica.

### Objetivos
1. Reproducir los resultados del profesor (regresión logística, SVM lineal y SVM radial) con la partición dada, estandarizando **solo con el entrenamiento**.
2. Pasar de la **probabilidad** al **umbral** y calcular exactitud, sensibilidad, especificidad, precisión y AUC.
3. Entender la **geometría de la SVM**: hiperplano, margen 2/‖w‖, holguras y vectores de soporte.
4. Ver el efecto de **C**, del **kernel** y de **gamma**, y de **estandarizar**.
5. Elegir hiperparámetros con **validación cruzada solo en entrenamiento** (`GridSearchCV` dentro de un `Pipeline`) y evaluar **una sola vez** en prueba.
6. Distinguir **puntaje** (`decision_function`) de **probabilidad** (`probability=True`) y traducir entre **R y Python**.
7. Reconocer lo que el material afirma de forma imprecisa.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y datos |
| A | 10 | A1 · Línea base y partición |
| A | 15 | A2 · Regresión logística: `penalty=None` y la correspondencia con `glm` |
| A | 15 | A3 · De la probabilidad a la clase: el umbral |
| A | 10 | A4 · Métricas desde la matriz de confusión |
| B | 20 | B1 · SVM lineal: hiperplano, margen y vectores de soporte |
| B | 15 | B2 · El parámetro C |
| B | 20 | B3 · Kernels y gamma |
| B | 10 | B4 · Estandarizar |
| C | 20 | C1 · `GridSearchCV` con C y gamma (solo entrenamiento) |
| C | 10 | C2 · Fuga de información con la escala |
| C | 15 | C3 · Puntaje frente a probabilidad (y calibración) |
| D | 15 | D1 · Comparar los tres modelos de la clase |
| D | 10 | D2 · Tabla de correspondencia R ↔ Python |
| — | 15 | Alertas del material, verificadas con código |
| — | 35 | Ejercicio de cierre **sin ayuda** |

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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 42   # semilla para lo que sea aleatorio (folds mezclados, Platt)
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso de la API (Clases 30 y 31)
Los verbos valen igual para escaladores y modelos: **`fit`** aprende (solo con entrenamiento), **`transform`** aplica lo aprendido y **`predict`** predice con un modelo ya ajustado. Un **`Pipeline`** encadena pasos y hace el `fit` de cada uno solo con los datos que recibe; dentro de una validación cruzada eso significa que la escala se aprende **en cada fold**, solo con los folds de ajuste. Hoy hay tres cosas nuevas: **`decision_function`** (el puntaje f(x) de una SVM), **`GridSearchCV`** (probar combinaciones de hiperparámetros con validación cruzada) y los atributos que dejan los modelos ya ajustados (`coef_`, `intercept_`, `support_vectors_`).

### Los datos de la clase: Wisconsin
El archivo ya trae la columna `conjunto`: **no hay que dividir nada**. El profesor lo explica así: dentro de las 398 filas de entrenamiento se eligen los hiperparámetros con validación cruzada, y las 171 de prueba se miran una sola vez al final.
""")
code("""
def buscar_datos():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in list(base.glob("Clase 37*/wisconsin_clase.csv")) + list(base.glob("wisconsin_clase.csv")):
            return p
    raise FileNotFoundError("No encuentro wisconsin_clase.csv: abre el notebook dentro de la carpeta del curso.")


datos = pd.read_csv(buscar_datos())
V4 = ["radio", "textura", "suavidad", "concavidad"]
entrenamiento = datos[datos["conjunto"] == "entrenamiento"]
prueba = datos[datos["conjunto"] == "prueba"]
X_tr, y_tr = entrenamiento[V4], entrenamiento["maligno"].to_numpy()
X_te, y_te = prueba[V4], prueba["maligno"].to_numpy()
print(datos.shape, "| faltantes:", int(datos[V4].isna().sum().sum()))
display(pd.crosstab(datos["conjunto"], datos["maligno"], margins=True))
display(X_tr.describe().round(3).loc[["mean", "std", "min", "max"]])
""")
md("""
Entrenamiento 398 (250 benignos y 148 malignos) y prueba 171 (107 y 64): partición estratificada ≈ 70/30. Fíjate en las **escalas**: el `radio` va de 7 a 28 y la `suavidad` de 0.05 a 0.16. Eso importará en B4.

*Dos precisiones sobre lo dicho en clase*: la base tiene **569** casos (357 son solo los benignos; se dijo «con la base de 357 vamos a dividir en 70 y 30») y la clase positiva es **maligno (1)**.
""")

# ================================================================== PARTE A
md("""
## Parte A · Regresión logística y evaluación

### A1 · Línea base y partición (10 min)
Antes de ajustar cualquier modelo hay que saber qué hace una regla trivial. La **línea base** «siempre benigno» acierta 107 de 171 en la prueba (exactitud 0.626) y **no detecta ningún maligno** (sensibilidad 0). Una exactitud «alta» puede ocultar que una clase nunca se identifica.
""")
code("""
base = np.zeros(len(y_te), dtype=int)                                   # «siempre benigno»
print(f"exactitud {accuracy_score(y_te, base):.4f} | sensibilidad {recall_score(y_te, base, zero_division=0):.1f} | especificidad {recall_score(y_te, base, pos_label=0):.1f}")
""")
md("""
### ✍️ Ejercicio A1
En clase se dijo que la regla «siempre uno» acierta 107 casos. Calcula exactitud, sensibilidad y especificidad de la regla «siempre **maligno**» (todo 1). ¿Cuál regla acierta 107?
""")
code("""
# tu código aquí
""")
solution("""
siempre_uno = np.ones(len(y_te), dtype=int)
print("siempre 1 -> aciertos:", int((siempre_uno == y_te).sum()), "| exactitud", round(accuracy_score(y_te, siempre_uno), 4),
      "| sensibilidad", recall_score(y_te, siempre_uno), "| especificidad", recall_score(y_te, siempre_uno, pos_label=0))
print("siempre 0 -> aciertos:", int((base == y_te).sum()))
""", "La regla «siempre maligno» acierta 64 de 171 (exactitud 0.374), con sensibilidad 1 y especificidad 0. La que acierta 107 es «siempre **benigno** (0)»; la diapositiva 7 lo dice bien, la explicación hablada se equivocó de clase.")

md("""
### A2 · Regresión logística: `penalty=None` y la correspondencia con `glm` (15 min)
La regresión logística modela la probabilidad `p(x) = 1/(1 + e^(−η))`, con `η = β₀ + β·x`. En R, `glm(maligno ~ ..., family = binomial())` la ajusta **sin penalización**. En Python, `LogisticRegression()` **sí penaliza por defecto** (L2, `C = 1`); para reproducir `glm` hay que escribir `penalty=None`. Escalamos con `StandardScaler` ajustado **solo con el entrenamiento**, como en el script del profesor.
""")
code("""
escala = StandardScaler()
Xtr_e = escala.fit_transform(X_tr)       # media y sd del ENTRENAMIENTO
Xte_e = escala.transform(X_te)           # la misma transformación para la prueba

logistica = LogisticRegression(penalty=None, max_iter=2000).fit(Xtr_e, y_tr)
p_log = logistica.predict_proba(Xte_e)[:, 1]        # [:, 1] = P(clase 1 = maligno)
clase_log = (p_log >= 0.5).astype(int)
cm_log = confusion_matrix(y_te, clase_log, labels=[0, 1])
print("Matriz (filas = real, columnas = predicho, orden 0, 1):\\n", cm_log)
print("coeficientes estandarizados:", dict(zip(V4, logistica.coef_[0].round(3))), "| intercepto:", round(float(logistica.intercept_[0]), 3))

# la misma regresión en unidades originales: reproduce el glm de R de la diapositiva 28
glm_python = LogisticRegression(penalty=None, max_iter=100000, tol=1e-10).fit(X_tr, y_tr)
print("\\nunidades originales:", dict(zip(V4, glm_python.coef_[0].round(4))), "| intercepto:", round(float(glm_python.intercept_[0]), 4))
""")
md("""
Matriz **103, 4, 6, 58**: exactitud 0.9415, sensibilidad 0.9062, especificidad 0.9626 (diapositiva 52). Los coeficientes en unidades originales **coinciden con el `glm` de R** (−42.2511, 1.3399, 0.4421, 124.9794, 21.4077); los estandarizados son otros números, pero las **probabilidades son las mismas**. En unidades originales, `exp(0.4421) = 1.56` es el factor de los *odds* por cada unidad de `textura`; en cambio «una unidad de `suavidad`» no tiene sentido (el rango completo es 0.11): usa «+0.01», que multiplica los odds por `exp(1.2498) = 3.49`.

### ✍️ Ejercicio A2
Ajusta `LogisticRegression()` **sin** `penalty=None` (la opción por defecto). ¿Cambia la matriz de confusión? ¿Por qué no reproduce `glm`?
""")
code("""
# tu código aquí
""")
solution("""
defecto = LogisticRegression(max_iter=2000).fit(Xtr_e, y_tr)
print(confusion_matrix(y_te, defecto.predict(Xte_e), labels=[0, 1]))
print("coef. por defecto:", defecto.coef_[0].round(3), "| penalizado (L2, C = 1)")
print("coef. sin penalizar:", logistica.coef_[0].round(3))
""", "Con la penalización por defecto la matriz es [[102, 5], [7, 57]] (exactitud 0.930, un error más que sin penalizar) y los coeficientes se encogen. Por eso la diapositiva 19 insiste: `LogisticRegression()` no reproduce automáticamente `glm(family = binomial())`; hay que usar `penalty=None`.")

md("""
### A3 · De la probabilidad a la clase: el umbral (15 min)
Dos cosas distintas: la **probabilidad** `p` (la calcula el modelo) y la **clase** (la decidimos nosotros con un umbral `t`: clase 1 si `p ≥ t`). Si cambiamos `t`, **la probabilidad no cambia**; solo cambia la decisión. Primero, a mano, con el ejemplo de la diapositiva 17 (`η = −1 + 0.8·z₁ + 0.4·z₂`, con `z₁ = 1.5`, `z₂ = 2`):
""")
code("""
eta = -1 + 0.8 * 1.5 + 0.4 * 2
p = 1 / (1 + np.exp(-eta))
print(f"η = {eta:.2f} | p = {p:.3f} | t = 0.5 -> {int(p >= 0.5)} | t = 0.8 -> {int(p >= 0.8)} | odds = {p / (1 - p):.2f} = e^η = {np.exp(eta):.2f}")

# y ahora con el modelo real: matriz, sensibilidad y especificidad para varios umbrales
filas = []
for t in (0.3, 0.5, 0.8):
    tn, fp, fn, tp = confusion_matrix(y_te, (p_log >= t).astype(int), labels=[0, 1]).ravel()
    filas.append(dict(t=t, VN=tn, FP=fp, FN=fn, VP=tp, errores=fp + fn, sensibilidad=round(tp / (tp + fn), 4), especificidad=round(tn / (tn + fp), 4)))
display(pd.DataFrame(filas).set_index("t"))
""")
md("""
Con **`t = 0.5`** hay 10 errores (4 + 6); con **`t = 0.8`** hay 14 (2 + 12), como se vio en clase. Subir el umbral **gana especificidad** (0.963 → 0.981) y **pierde sensibilidad** (0.906 → 0.813): se clasifican como malignos solo los casos más seguros. En un diagnóstico, dejar escapar un maligno suele costar más que una falsa alarma; por eso el umbral es una **decisión** y no se elige mirando la prueba.

### ✍️ Ejercicio A3
Elige el umbral con **validación cruzada solo en entrenamiento** (`cross_val_predict` con `method="predict_proba"`, 5 folds mezclados y `random_state=RS`), maximizando la **exactitud balanceada** `(sensibilidad + especificidad)/2`, dentro de un `Pipeline` (escala + logística). Evalúalo **una vez** en la prueba. ¿Mejora frente a 0.5?
""")
code("""
# tu código aquí
""")
solution("""
pipe_log = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=2000))
p_cv = cross_val_predict(pipe_log, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), method="predict_proba")[:, 1]
umbrales = np.linspace(0.05, 0.95, 91)
bal = [(recall_score(y_tr, (p_cv >= u).astype(int)) + recall_score(y_tr, (p_cv >= u).astype(int), pos_label=0)) / 2 for u in umbrales]
u_mejor = float(umbrales[int(np.argmax(bal))])
pipe_log.fit(X_tr, y_tr)
p_prueba = pipe_log.predict_proba(X_te)[:, 1]
for u in (0.5, u_mejor):
    c = (p_prueba >= u).astype(int)
    print(f"t = {u:.2f}: errores {int((c != y_te).sum())} | sensibilidad {recall_score(y_te, c):.4f} | especificidad {recall_score(y_te, c, pos_label=0):.4f}")
""", "El umbral que maximiza la exactitud balanceada en validación cruzada es **0.44** y, en la prueba, **no mejora** frente a 0.5 (11 errores frente a 10; la sensibilidad es la misma, 0.906, y la especificidad baja de 0.963 a 0.953): una diferencia de un caso entre 171 es ruido. Lo importante es el **método**: el umbral se elige **sin mirar la prueba**.")

md("""
### A4 · Métricas desde la matriz de confusión (10 min)
Con la clase positiva declarada (maligno = 1), filas = real y columnas = predicho: `[[VN, FP], [FN, VP]]`. Con el ejemplo de la diapositiva 9 (VN 56, FP 14, FN 6, VP 24): exactitud 0.80, sensibilidad 0.80, especificidad 0.80, precisión 24/38 = 0.632, F1 = 48/(48 + 14 + 6) = 0.706.
""")
code("""
def metricas(vn, fp, fn, vp):
    n = vn + fp + fn + vp
    sens, espe = vp / (vp + fn), vn / (vn + fp)
    return dict(exactitud=(vn + vp) / n, sensibilidad=sens, especificidad=espe, precision=vp / (vp + fp),
                F1=2 * vp / (2 * vp + fp + fn), balanceada=(sens + espe) / 2)

print({k: round(v, 3) for k, v in metricas(56, 14, 6, 24).items()})
""")
md("""
### ✍️ Ejercicio A4
Con la matriz de la regresión logística en la prueba (`cm_log`), calcula las seis métricas con `metricas`. ¿Por qué la exactitud balanceada (0.934) es menor que la exactitud (0.942)? ¿Con qué función de scikit-learn sacas la especificidad?
""")
code("""
# tu código aquí
""")
solution("""
vn, fp, fn, vp = cm_log.ravel()
print({k: round(v, 4) for k, v in metricas(vn, fp, fn, vp).items()})
print("especificidad con recall_score(pos_label=0):", round(recall_score(y_te, clase_log, pos_label=0), 4))
""", "Exactitud 0.9415, sensibilidad 0.9062, especificidad 0.9626, precisión 0.9355, F1 0.9206 y exactitud balanceada 0.9344. La balanceada pesa igual a las dos clases, y la clase maligna (la menos frecuente) se detecta algo peor (0.906); la exactitud pesa por frecuencia (más benignos). La especificidad es `recall_score(..., pos_label=0)`.")

# ================================================================== PARTE B
md("""
## Parte B · La SVM por dentro

### B1 · SVM lineal: hiperplano, margen y vectores de soporte (20 min)
La SVM lineal aprende `f(x) = w·x + b`; la clase es el **signo** de `f(x)`. La **distancia** de un punto a la frontera es `|f(x)|/‖w‖` y el **ancho total del margen** es `2/‖w‖` (bordes en `f = ±1`). Para un caso con `y = ±1` se mira `y·f(x)`: `≥ 1` está bien y fuera del margen; entre 0 y 1, dentro del margen; `< 0`, mal clasificado. La **holgura** es `ξ = máx(0, 1 − y·f(x))`. Usamos el modelo del profesor (`kernel="linear"`, `C = 0.1`, 4 variables estandarizadas).
""")
code("""
svm_lineal = SVC(kernel="linear", C=0.1).fit(Xtr_e, y_tr)
w, b = svm_lineal.coef_[0], float(svm_lineal.intercept_[0])
print("w =", w.round(3), "| b =", round(b, 3), "| ‖w‖ =", round(float(np.linalg.norm(w)), 3), "| ancho del margen 2/‖w‖ =", round(float(2 / np.linalg.norm(w)), 3))

# f(x) a mano para un caso de prueba y comparación con decision_function
x0 = Xte_e[0]
f_mano = float(w @ x0 + b)
print(f"caso 0 de prueba: f a mano = {f_mano:.4f} | decision_function = {svm_lineal.decision_function([x0])[0]:.4f} | distancia = {abs(f_mano) / np.linalg.norm(w):.3f}")

# holguras en el entrenamiento
y_pm = 2 * y_tr - 1                                     # etiquetas −1 / +1 solo para esta cuenta
yf = y_pm * svm_lineal.decision_function(Xtr_e)
xi = np.maximum(0, 1 - yf)
print("mal clasificados (y·f < 0):", int((yf < 0).sum()), "| dentro del margen (0 ≤ y·f < 1):", int(((yf >= 0) & (yf < 1)).sum()), "| correctos y fuera (y·f ≥ 1):", int((yf >= 1).sum()))
print("vectores de soporte:", int(svm_lineal.n_support_.sum()), svm_lineal.n_support_, "| Σξ =", round(float(xi.sum()), 2))
print("matriz en prueba:", confusion_matrix(y_te, svm_lineal.predict(Xte_e), labels=[0, 1]).ravel())
""")
md("""
Con `C = 0.1` hay **93 vectores de soporte** (47 + 46); en el entrenamiento, 20 casos mal clasificados, 70 dentro del margen y 308 correctos y fuera. (Los soportes son los casos con `y·f ≤ 1`; el conteo por `y·f` puede diferir en uno o dos casos del de scikit-learn por la tolerancia numérica.) La matriz de prueba es **103, 4, 10, 54**: exactitud 0.918, sensibilidad 0.844. `f(x)` **no es una probabilidad**: sus valores no están entre 0 y 1.

*Etiquetas −1 y +1*: el profesor las usa para explicar la geometría («hay que ser menos uno y uno»), pero en `SVC` y en `svm()` de R las clases pueden seguir siendo 0 y 1 (la diapositiva 31 lo aclara); lo comprobamos en las alertas.

### ✍️ Ejercicio B1
Para los **tres primeros** casos de prueba, calcula a mano `f(x) = w·x + b` y la clase predicha, y compáralas con `svm_lineal.predict`. ¿Cuántos casos de **entrenamiento** tienen `y·f(x) ≤ 1` (soportes)?
""")
code("""
# tu código aquí
""")
solution("""
for i in range(3):
    f_i = float(w @ Xte_e[i] + b)
    print(f"caso {i}: f = {f_i:+.3f} -> clase {int(f_i >= 0)} | predict = {int(svm_lineal.predict([Xte_e[i]])[0])} | real = {int(y_te[i])}")
print("casos con y·f ≤ 1 (soportes):", int((yf <= 1 + 1e-6).sum()), "| support_ de sklearn:", len(svm_lineal.support_))
""", "La clase es el signo de f(x), y coincide con `predict`. Los casos con `y·f ≤ 1` son los vectores de soporte: 90 por este conteo y 93 según sklearn (la diferencia viene de la tolerancia numérica del optimizador en casos que quedan justo en el borde).")

md("""
### B2 · El parámetro C (15 min)
`C` pondera las holguras en `mín ½‖w‖² + C·Σξ`. **C pequeño**: se toleran más violaciones (margen ancho, regularización más fuerte). **C grande**: se castigan casi todas (margen estrecho, más pegado al entrenamiento). El efecto se ve en el **ancho del margen** y en la **cantidad de soportes**; y la **validación cruzada** (dentro de un `Pipeline`, solo en entrenamiento) dice cuánto conviene. Con `Σξ = 4`, la contribución es `0.1·4 = 0.4` con `C = 0.1` y `10·4 = 40` con `C = 10`.
""")
code("""
filas = []
for C in (0.001, 0.01, 0.1, 1, 10, 1000):
    pipe = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5).mean()            # 5 folds estratificados, SOLO entrenamiento
    pipe.fit(X_tr, y_tr)
    s = pipe[-1]
    filas.append(dict(C=C, soportes=len(s.support_), ancho_margen=round(float(2 / np.linalg.norm(s.coef_[0])), 3),
                      exactitud_entrenamiento=round(pipe.score(X_tr, y_tr), 4), exactitud_cv=round(float(cv), 4)))
display(pd.DataFrame(filas).set_index("C"))
""")
md("""
Al subir `C` el margen se estrecha y bajan los soportes (296 → 52 entre `C = 0.001` y `C = 1000`). La exactitud de validación cruzada sube hasta **`C = 0.1` (0.940)** y luego se mantiene o baja un poco (0.930 con `C = 1000`): «mientras más `C`, mejor» no es cierto. (No mostramos aquí la exactitud de prueba para no tentarnos a elegir con ella.)

### ✍️ Ejercicio B2
En clase alguien propuso: «siempre conviene un C pequeño, porque así no deja que acepte mal los valores». ¿Qué tiene de equivocado? Compara en una tabla los **soportes** y la **exactitud de validación cruzada** de `C = 0.01` y `C = 100`.
""")
code("""
# tu código aquí
""")
solution("""
for C in (0.01, 100):
    pipe = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
    print(f"C = {C}: exactitud CV {cross_val_score(pipe, X_tr, y_tr, cv=5).mean():.4f} | soportes {len(pipe.fit(X_tr, y_tr)[-1].support_)}")
""", "Es al revés: un C **pequeño** da menos peso a las violaciones, así que **acepta más errores** (más soportes: 174 con C = 0.01 frente a 52 con C = 100) y puede subajustar (CV 0.917 frente a 0.930). Un C grande castiga más las violaciones y se pega al entrenamiento. Ninguno es mejor sin validar.")

md("""
### B3 · Kernels y gamma (20 min)
Un **kernel** `K(x, z) = ⟨φ(x), φ(z)⟩` calcula un producto interno en un espacio de características sin construirlo (*truco del kernel*): la frontera es lineal allí y **curva** en el espacio original. Lineal `x·z`; polinomial `(γ x·z + r)^d` (hiperparámetros `γ`, `r` y el grado `d`); radial (RBF) `exp(−γ‖x − z‖²)`. En el RBF, `γ` fija el **alcance**: con `‖x − z‖² = 4` la similitud es `e^(−0.4) = 0.6703` si `γ = 0.1` y `e^(−4) = 0.0183` si `γ = 1`.
""")
code("""
print("similitud RBF a distancia² = 4:", {g: round(float(np.exp(-4 * g)), 4) for g in (0.1, 1)})

filas = []
for gamma in (0.001, 0.01, 0.1, 1, 10, 100):
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma=gamma))
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5).mean()
    pipe.fit(X_tr, y_tr)
    filas.append(dict(gamma=gamma, soportes=len(pipe[-1].support_), exactitud_entrenamiento=round(pipe.score(X_tr, y_tr), 4), exactitud_cv=round(float(cv), 4)))
display(pd.DataFrame(filas).set_index("gamma"))
""")
md("""
Con `γ` pequeño el RBF es casi lineal (subajuste); con `γ` grande la exactitud de **entrenamiento** sube hasta 1.0 pero la de **validación** se desploma (0.633 con `γ = 100`): sobreajuste. El valor intermedio gana (`γ ≈ 0.01`–`0.1`). Esto es lo que dice la diapositiva 41: «una frontera muy detallada puede ajustarse a particularidades del entrenamiento que no se repiten».

*Una precisión*: en clase se dijo que «el polinomial tiene un hiperparámetro, el radial gamma y el lineal ninguno». En realidad el polinomial tiene tres (`γ`, `r` y el grado) y solo el **lineal** carece de `γ`.

### ✍️ Ejercicio B3
Ajusta con `StandardScaler` un `SVC(kernel="poly", degree=3, coef0=1, C=1)` y un `SVC(kernel="rbf", C=1, gamma=0.5)` y compara su exactitud de **validación cruzada** (5 folds, solo entrenamiento) con la del lineal con `C = 1`. ¿Hay un kernel claramente mejor?
""")
code("""
# tu código aquí
""")
solution("""
for nombre, modelo in [("lineal", SVC(kernel="linear", C=1)), ("polinomial (3)", SVC(kernel="poly", degree=3, coef0=1, C=1)), ("RBF (γ = 0.5)", SVC(kernel="rbf", C=1, gamma=0.5))]:
    pipe = make_pipeline(StandardScaler(), modelo)
    s = cross_val_score(pipe, X_tr, y_tr, cv=5)
    print(f"{nombre:16s} CV {s.mean():.4f} ± {s.std():.4f}")
""", "Los tres quedan en torno a 0.93–0.94 con desviaciones de 2 a 5 puntos entre folds: **ninguno es claramente mejor**; lo que cuenta es elegir con validación, no por el nombre del kernel.")

md("""
### B4 · Estandarizar (10 min)
La SVM usa **distancias**: si una variable tiene una escala mucho mayor, domina. Con estas cuatro variables la desviación estándar va de 0.014 (`suavidad`) a 4.2 (`textura`): un factor de ~290. Comparemos los modelos del profesor **con** y **sin** estandarizar (misma partición).
""")
code("""
filas = []
for nombre, kw in [("lineal C=0.1", dict(kernel="linear", C=0.1)), ("RBF C=10 γ=0.1", dict(kernel="rbf", C=10, gamma=0.1))]:
    sin = SVC(**kw).fit(X_tr.to_numpy(), y_tr)
    con = make_pipeline(StandardScaler(), SVC(**kw)).fit(X_tr, y_tr)
    filas.append(dict(modelo=nombre, sin_estandarizar=round(sin.score(X_te.to_numpy(), y_te), 4), estandarizado=round(con.score(X_te, y_te), 4)))
display(pd.DataFrame(filas).set_index("modelo"))
print("desviaciones estándar:", X_tr.std().round(4).to_dict())
""")
md("""
Sin estandarizar, la exactitud de prueba baja de 0.918 a **0.860** (lineal) y de 0.971 a **0.901** (RBF). *Estandarizar no elimina valores atípicos ni garantiza normalidad*: solo pone las variables en una escala comparable (diapositiva 14). Y la media y la sd deben salir **del entrenamiento**.

### ✍️ Ejercicio B4
`SVC(gamma="scale")` (el valor por defecto de scikit-learn) usa `1/(n_variables · X.var())`. Calcula ese valor con las variables **estandarizadas** y con las **sin estandarizar**. ¿Qué ocurre con la segunda?
""")
code("""
# tu código aquí
""")
solution("""
g_est = 1 / (X_tr.shape[1] * Xtr_e.var())
g_crudo = 1 / (X_tr.shape[1] * X_tr.to_numpy().var())
print("gamma='scale' estandarizado:", round(float(g_est), 4), "| sin estandarizar:", round(float(g_crudo), 5))
""", "Con las variables estandarizadas la varianza es 1 y `gamma='scale'` = 1/4 = 0.25 (igual al valor por defecto de `e1071`, 1/p). Sin estandarizar, la varianza del conjunto está dominada por `radio` y `textura` (de 10 a 40) y `gamma` resulta mucho menor (0.003): la elección automática ya no es comparable. Otro motivo para estandarizar.")

# ================================================================== PARTE C
md("""
## Parte C · Validación y elección de hiperparámetros

### C1 · `GridSearchCV` con C y gamma, solo en entrenamiento (20 min)
La diapositiva 62 busca la mejor combinación con cinco folds. El `Pipeline` es esencial: así `StandardScaler` se vuelve a ajustar **dentro de cada fold**, solo con los folds de ajuste. Se prueban 3 × 3 combinaciones × 5 folds = **45 ajustes** (y al final reajusta la mejor con todo el entrenamiento).
""")
code("""
modelo = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
rejilla = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}
busqueda = GridSearchCV(modelo, rejilla, cv=5, scoring="accuracy").fit(X_tr, y_tr)        # SOLO entrenamiento
print("mejor:", busqueda.best_params_, "| exactitud media en validación:", round(busqueda.best_score_, 4))
res = pd.DataFrame(busqueda.cv_results_)[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score", "rank_test_score"]]
display(res.sort_values("rank_test_score").round(4))
# la prueba se mira UNA vez, con la configuración ya elegida
mejor = busqueda.best_estimator_
print("matriz en prueba:", confusion_matrix(y_te, mejor.predict(X_te), labels=[0, 1]).ravel(), "| exactitud:", round(mejor.score(X_te, y_te), 4))
""")
md("""
Gana **C = 10, γ = 0.1** (0.9396), la misma configuración que fijó el profesor; la segunda es `C = 1, γ = 0.1` (0.9347), dentro de una desviación estándar. En R la búsqueda eligió `C = 1, γ = 0.1`: **no es una contradicción**; los folds internos no son los mismos (diapositiva 63) y las diferencias están dentro del ruido. En la prueba, 106, 1, 4, 60 (0.9708).

### ✍️ Ejercicio C1
Amplía la rejilla a `C ∈ {0.01, 0.1, 1, 10, 100, 1000}` y `γ ∈ {0.001, 0.01, 0.1, 1, 10}` (150 ajustes). ¿Qué combinación gana? ¿Cuánta exactitud de **prueba** tiene frente a la del profesor (0.9708)? ¿Qué concluyes?
""")
code("""
# tu código aquí
""")
solution("""
amplia = GridSearchCV(modelo, {"svc__C": [0.01, 0.1, 1, 10, 100, 1000], "svc__gamma": [0.001, 0.01, 0.1, 1, 10]}, cv=5).fit(X_tr, y_tr)
print("mejor:", amplia.best_params_, "| CV", round(amplia.best_score_, 4), "| prueba", round(amplia.best_estimator_.score(X_te, y_te), 4))
tabla = pd.DataFrame(amplia.cv_results_).sort_values("rank_test_score").head(4)[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score"]]
display(tabla.round(4))
""", "Gana `C = 100, γ = 0.01` (CV 0.9447, apenas por encima de otras combinaciones) y en la prueba da 0.9474, **menos** que el 0.9708 de la configuración del profesor. Conclusión: con 171 casos (cada caso vale 0.58 puntos) y desviaciones de 2–4 puntos entre folds, esas diferencias no demuestran que una configuración sea mejor; **no se elige con la prueba** ni se interpreta como superioridad.")

md("""
### C2 · Fuga de información con la escala (10 min)
La diapositiva 13 lo dice: calcular medias y desviaciones con **toda la base** antes de validar es un error. Veamos la diferencia: estandarizar **antes** de la validación cruzada (la escala «ve» los folds de validación) frente al `Pipeline` correcto. En esta base el efecto numérico es pequeño, pero el procedimiento es incorrecto y en otros casos (selección de variables, imputación) puede inflar mucho los resultados.
""")
code("""
# incorrecto: la media y la sd se calculan con TODO el entrenamiento, incluidos los folds de validación
X_fuga = StandardScaler().fit_transform(X_tr)
fuga = cross_val_score(SVC(kernel="rbf", C=10, gamma=0.1), X_fuga, y_tr, cv=5).mean()
# correcto: dentro de cada fold la escala se aprende solo con los folds de ajuste
ok = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.1)), X_tr, y_tr, cv=5).mean()
print(f"estandarizar antes: {fuga:.4f} | Pipeline (correcto): {ok:.4f} | diferencia: {fuga - ok:+.4f}")
""")
md("""
Las dos cifras son casi iguales aquí, pero solo la segunda estima honestamente el desempeño. **La misma regla vale para** imputar, seleccionar variables y re-muestrear: todo lo que aprende de los datos va **dentro** del `Pipeline`.

### ✍️ Ejercicio C2
¿Por qué es **aún más importante** el `Pipeline` en `GridSearchCV` que en una sola validación cruzada? (Piensa en cuántas veces se mira el mismo fold de validación.) Escríbelo como comentario.
""")
code("""
# tu respuesta aquí
""")
solution("""
respuesta = (
    "GridSearchCV evalua decenas de combinaciones sobre los mismos folds: si la escala (o cualquier paso aprendido) se calculo antes, "
    "cada combinacion se favorece con informacion de los folds de validacion y el 'mejor' se elige con un criterio contaminado. "
    "Con el Pipeline, cada ajuste recalcula la escala solo con sus folds de ajuste."
)
print(respuesta)
""", "El mismo fold de validación se usa para comparar 9 (o 30) combinaciones: cualquier filtración se repite y se acumula en la selección del «mejor» (sesgo de selección). Por eso la regla de la diapositiva 13 y el `Pipeline`.")

md("""
### C3 · Puntaje frente a probabilidad, y calibración (15 min)
Una SVM entrega un **puntaje** `f(x)` (`decision_function`), no una probabilidad: la diapositiva 3 lo dice. Con `SVC(probability=True)` scikit-learn calibra una probabilidad con el **ajuste de Platt** (una sigmoide aplicada al puntaje, con una validación cruzada interna de 5 folds), y **`predict` sigue usando el signo de `f(x)`**: por eso `predict` y «`p ≥ 0.5`» pueden **discrepar** en algunos casos. La **AUC** usa puntajes (o probabilidades), nunca las clases 0/1.
""")
code("""
svm_radial = SVC(kernel="rbf", C=10, gamma=0.1).fit(Xtr_e, y_tr)
f_rad = svm_radial.decision_function(Xte_e)
platt = SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=0).fit(Xtr_e, y_tr)
p_platt = platt.predict_proba(Xte_e)[:, 1]
pred = platt.predict(Xte_e)
disc = (p_platt >= 0.5) != (pred == 1)
print("AUC con el puntaje f(x):", round(roc_auc_score(y_te, f_rad), 4), "| AUC con la probabilidad de Platt:", round(roc_auc_score(y_te, p_platt), 4))
print("predict vs (p ≥ 0.5): discrepan", int(disc.sum()), "casos | exactitud predict:", round(accuracy_score(y_te, pred), 4), "| exactitud p ≥ 0.5:", round(accuracy_score(y_te, (p_platt >= 0.5).astype(int)), 4))
print("Brier (menor = mejor calibrada): SVM Platt", round(brier_score_loss(y_te, p_platt), 4), "| logística", round(brier_score_loss(y_te, p_log), 4))
""")
md("""
La **AUC es idéntica** (0.9834) porque la sigmoide de Platt es monótona: conserva el orden de los puntajes. Pero en **3 casos** `predict` y `p ≥ 0.5` discrepan y la exactitud baja de 0.9708 a 0.9532. Conclusión práctica: si necesitas **probabilidades**, la logística las da de forma natural; en la SVM, `probability=True` encarece el ajuste (validación cruzada interna) y hay que verificar la calibración; si solo necesitas clases u ordenar casos, usa `predict` o `decision_function`.

### ✍️ Ejercicio C3
Calcula la AUC de la **SVM lineal** (`C = 0.1`) con `decision_function` y compárala con la de la logística. ¿Qué observas? ¿Por qué dos modelos con distinta exactitud pueden tener la misma AUC (diapositiva 64)?
""")
code("""
# tu código aquí
""")
solution("""
f_lin = svm_lineal.decision_function(Xte_e)
print("AUC SVM lineal:", round(roc_auc_score(y_te, f_lin), 4), "| AUC logística:", round(roc_auc_score(y_te, p_log), 4))
print("exactitud: lineal", round(accuracy_score(y_te, svm_lineal.predict(Xte_e)), 4), "| logística", round(accuracy_score(y_te, clase_log), 4))
""", "Ambas dan **AUC 0.9746**, aunque la exactitud es 0.918 y 0.942. La AUC mide el **orden** de los puntajes entre positivos y negativos a través de todos los umbrales; la exactitud depende de **un** corte (el de 0.5 en la logística y el de f = 0 en la SVM). Dos modelos pueden ordenar igual y operar distinto en su corte habitual.")

# ================================================================== PARTE D
md("""
## Parte D · Comparar y cerrar

### D1 · Los tres modelos de la clase (15 min)
Reproducimos la tabla de la diapositiva 52 (y la del script `clasificadores.py`) y agregamos la AUC. Los tres usan la **misma partición**, la **misma escala** (del entrenamiento) y configuraciones **fijadas antes de ver la prueba**: logística sin penalización con `t = 0.5`, SVM lineal con `C = 0.1` y SVM radial con `C = 10, γ = 0.1`.
""")
code("""
modelos = {"Logística": (clase_log, p_log), "SVM lineal": (svm_lineal.predict(Xte_e), svm_lineal.decision_function(Xte_e)),
           "SVM radial": (svm_radial.predict(Xte_e), svm_radial.decision_function(Xte_e))}
filas = []
for nombre, (clase, puntaje) in modelos.items():
    tn, fp, fn, tp = confusion_matrix(y_te, clase, labels=[0, 1]).ravel()
    filas.append(dict(Modelo=nombre, VN=tn, FP=fp, FN=fn, VP=tp, Exactitud=accuracy_score(y_te, clase), Sensibilidad=recall_score(y_te, clase, pos_label=1),
                      Especificidad=recall_score(y_te, clase, pos_label=0), Precision=precision_score(y_te, clase), AUC=roc_auc_score(y_te, puntaje)))
comparacion = pd.DataFrame(filas).set_index("Modelo")
display(comparacion.round(4))

nuevo = pd.DataFrame({"radio": [14], "textura": [20], "suavidad": [0.10], "concavidad": [0.08]})
nuevo_e = escala.transform(nuevo)
print("caso nuevo -> logística p =", round(float(logistica.predict_proba(nuevo_e)[0, 1]), 4), "| SVM radial clase:", int(svm_radial.predict(nuevo_e)[0]),
      "| SVM lineal clase:", int(svm_lineal.predict(nuevo_e)[0]))
""")
md("""
| Modelo | VN · FP · FN · VP | Exactitud | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| Logística | 103 · 4 · 6 · 58 | 0.9415 | 0.9062 | 0.9626 | 0.9746 |
| SVM lineal | 103 · 4 · 10 · 54 | 0.9181 | 0.8438 | 0.9626 | 0.9746 |
| SVM radial | 106 · 1 · 4 · 60 | 0.9708 | 0.9375 | 0.9907 | 0.9834 |

Los números **coinciden con las diapositivas 52 y 53**. Para el caso nuevo (14, 20, 0.10, 0.08) la logística da `p = 0.3919` (clase 0) y la SVM radial **clase 1**: **los modelos pueden discrepar** porque sus fronteras son distintas. Es una **comparación docente de una partición**: un caso de prueba vale 0.58 puntos, así que «gana el radial» no es una superioridad universal ni una validación clínica.

### ✍️ Ejercicio D1
Dibuja las curvas ROC de los tres modelos en una sola figura y marca el punto de la logística con `t = 0.5` y con `t = 0.8`. ¿Qué punto está «sobre» la curva de la logística y por qué?
""")
code("""
# tu código aquí
""")
solution("""
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for nombre, (clase, puntaje) in modelos.items():
    fpr, tpr, _ = roc_curve(y_te, puntaje)
    ax.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC {roc_auc_score(y_te, puntaje):.4f})")
for t, marca in [(0.5, "o"), (0.8, "s")]:
    tn, fp, fn, tp = confusion_matrix(y_te, (p_log >= t).astype(int), labels=[0, 1]).ravel()
    ax.scatter([fp / (fp + tn)], [tp / (tp + fn)], marker=marca, s=70, edgecolor="black", zorder=5, label=f"logística t = {t}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("1 − especificidad"); ax.set_ylabel("sensibilidad"); ax.legend(fontsize=8)
plt.show()
""", "Cada umbral `t` es **un punto** de la curva ROC de la logística: `t = 0.5` (sensibilidad 0.906, 1 − especificidad 0.037) y `t = 0.8` (0.813, 0.019). La curva resume todos los umbrales; la AUC no depende de elegir uno.")

md("""
### D2 · Tabla de correspondencia R ↔ Python (10 min)
Los scripts del profesor (`clasificadores.R.txt` y `clasificadores.py`) hacen lo mismo con las mismas filas y obtienen las **mismas matrices de confusión**. Esta tabla te ayuda a leer uno desde el otro:

| Qué se hace | R | Python (scikit-learn) |
|---|---|---|
| Leer los datos | `read.csv("wisconsin_clase.csv")` | `pd.read_csv("wisconsin_clase.csv")` |
| Filas de entrenamiento | `subset(datos, conjunto == "entrenamiento")` | `datos[datos["conjunto"] == "entrenamiento"]` |
| Respuesta como clases | `factor(maligno, levels = c(0, 1))` | enteros 0/1 (no hace falta factor) |
| Estandarizar | `svm(..., scale = TRUE)` (internamente) | `StandardScaler()`: `fit_transform` en entrenamiento, `transform` en prueba |
| Regresión logística | `glm(maligno ~ ..., family = binomial())` | `LogisticRegression(penalty=None, max_iter=2000)` |
| Probabilidad de maligno | `predict(modelo, newdata, type = "response")` | `modelo.predict_proba(X)[:, 1]` |
| Clase con umbral 0.5 | `ifelse(p >= 0.5, 1, 0)` | `(p >= 0.5).astype(int)` |
| SVM lineal | `svm(x, y, kernel = "linear", cost = 0.1)` | `SVC(kernel="linear", C=0.1)` |
| SVM radial | `svm(..., kernel = "radial", cost = 10, gamma = 0.1)` | `SVC(kernel="rbf", C=10, gamma=0.1)` |
| Parámetro de penalización de violaciones | `cost` | `C` (mismo parámetro de libsvm) |
| Puntaje de la SVM | `attr(predict(m, ..., decision.values = TRUE), "decision.values")` | `m.decision_function(X)` |
| Matriz de confusión | `table(Real = y, Prediccion = clase)` | `confusion_matrix(y, clase, labels=[0, 1])` |
| Sensibilidad / especificidad | `m[2, 2] / sum(m[2, ])` · `m[1, 1] / sum(m[1, ])` | `recall_score(y, clase, pos_label=1)` · `pos_label=0` |
| AUC | `pROC::auc(y, puntaje, levels = c("0", "1"), direction = "<")` | `roc_auc_score(y, puntaje)` |
| Búsqueda de hiperparámetros | `tune.svm(..., cross = 5)` (error medio) | `GridSearchCV(pipe, rejilla, cv=5, scoring="accuracy")` (exactitud media) |

*Dos diferencias a recordar*: en R los coeficientes de `glm` están en unidades **originales**; en el script de Python están **estandarizados** (otra escala, mismas probabilidades). Y como `e1071` y `SVC` usan la misma biblioteca (libsvm), con los mismos datos, escala y parámetros las predicciones coinciden.

### ✍️ Ejercicio D2
Traduce a Python esta línea de R: `tune.svm(x, y, kernel = "radial", scale = TRUE, cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1), tunecontrol = tune.control(cross = 5))` y explica por qué en Python la escala va **dentro** del `Pipeline`.
""")
code("""
# tu código aquí
""")
solution("""
traduccion = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                          {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}, cv=5, scoring="accuracy")
print(traduccion.fit(X_tr, y_tr).best_params_)
""", "`scale = TRUE` de `tune.svm` ya estima la escala **dentro de cada fold**; en Python eso se logra poniendo `StandardScaler` dentro del `Pipeline` (los parámetros se nombran `svc__C` y `svc__gamma`). Si se estandarizara antes, la escala «vería» los folds de validación (C2).")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene precisar (verificado con código)

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Clase | «con la base de 357 vamos a dividir en 70 y 30»; la grabación dice «559 muestras» | La base tiene **569** casos (357 son los benignos); partición 398 / 171 (A1). |
| Clase | la regla «siempre uno» acierta 107 casos | La regla que acierta 107 es «siempre **benigno (0)**»; «siempre 1» acierta 64 (A1). |
| Clase | «para SVM hay que usar −1 y 1» | Solo la convención geométrica: `SVC` acepta 0/1 y da las mismas predicciones (E1). |
| Clase | el estimador de máxima verosimilitud «tiene garantizado que tiene una solución única» | Con **separación perfecta** no existe máximo finito: la verosimilitud sigue creciendo (E2). |
| Clase | la regresión lineal con 0/1 «no da una probabilidad» | Cierto: en la prueba da valores de −0.32 a 1.58 (E3). |
| Clase | falso negativo: «va a ser mal pagador y predecimos buen pagador» | Con 1 = buen pagador eso es un falso **positivo**. |
| Clase | «el polinomial tiene un hiperparámetro, el radial gamma, el lineal ninguno» | El polinomial tiene tres (γ, r, grado d); solo el **lineal** no tiene γ (B3). |
| Clase | «cuando voy a hacer inferencia, siempre me voy con regularización» | La penalización **sesga** los coeficientes y no da los intervalos clásicos; es para predicción/selección. |
| Clase | Python: «C igual a 1 … los dos valores iguales a 10» | El script fija C = 10 y γ = 0.1 (diapositiva 43). |
| Diapositiva 28 | «una unidad de suavidad multiplica los odds por e^124.98» | Sin sentido práctico (rango 0.05–0.16): usar «+0.01» (×3.49) o variables estandarizadas (A2). |
| Diapositiva 63 | R elige C = 1, γ = 0.1; Python C = 10, γ = 0.1 | No es contradicción: folds distintos y diferencias dentro del ruido (C1). |
| Script / clase | «cada modelo tendrá una AUC distinta» | Logística y SVM lineal empatan en 0.9746 con distinta exactitud (C3). |
| Python | `SVC(probability=True)` como «la probabilidad de la SVM» | Es una calibración de Platt con CV interna; puede discrepar de `predict` (3 casos, C3). |

**Transcripción con ruido**: «Luis Colzin» = Wisconsin, «RLM / relíem net» = glm / glmnet, «e de 71» = e1071, «Timi Chirani» = Tibshirani, «Breeman» = Breiman, «CAFOD» = k-folds, «crânio / k-neu» = kernel, «rock / roca» = ROC, «iance / yance» = chance, «chate PT» = ChatGPT. El volumen cayó ≈ 12 dB desde ≈ 65 min (se transcribió una copia amplificada); el recreo (≈ 1:26:14) no está en el video.
""")
code("""
# E1 · etiquetas 0/1 frente a −1/+1: mismas predicciones
a = SVC(kernel="linear", C=0.1).fit(Xtr_e, y_tr)
b_ = SVC(kernel="linear", C=0.1).fit(Xtr_e, 2 * y_tr - 1)
print("E1 · predicciones iguales con etiquetas 0/1 y −1/+1:", bool(np.array_equal(a.predict(Xte_e), (b_.predict(Xte_e) + 1) // 2)))
""")
code("""
# E2 · separación perfecta: la log-verosimilitud sigue subiendo con el coeficiente, no hay máximo
x_sep = np.array([-3, -2, -1, 1, 2, 3.0]); y_sep = np.array([0, 0, 0, 1, 1, 1])
def loglik(beta):
    s = 2 * y_sep - 1                                     # −1 / +1: log p = −log(1 + e^(−s·beta·x))
    return float(-np.sum(np.logaddexp(0, -s * beta * x_sep)))
print("E2 · log-verosimilitud:", {beta: float(f"{loglik(beta):.3g}") for beta in (1, 2, 5, 10, 20)}, "(siempre creciente hacia 0)")
""")
code("""
# E3 · regresión lineal con 0/1: «probabilidades» fuera de [0, 1]
lineal = LinearRegression().fit(Xtr_e, y_tr)
yh = lineal.predict(Xte_e)
print(f"E3 · rango de la predicción: {yh.min():.2f} a {yh.max():.2f} | por debajo de 0: {int((yh < 0).sum())} | por encima de 1: {int((yh > 1).sum())}")
""")
md("""
Resultados: **E1** `True` (mismas predicciones); **E2** la log-verosimilitud sube de −0.978 (β = 1) a −0.295, −0.0135, −9.1e-05 y −4.1e-09 (β = 20) y nunca alcanza un máximo (con datos separables el MLE «no existe»; scikit-learn se detiene cuando el gradiente es pequeño, y por eso devuelve un número finito arbitrario); **E3** rango −0.32 a 1.58, con 17 casos bajo 0 y 9 sobre 1.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes entregar una recomendación sobre la base `wisconsin_clase.csv` (partición dada: usa `X_tr`, `y_tr`, `X_te`, `y_te`). Responde con código:

1. Arma un `Pipeline` (`StandardScaler` + SVM) y elige con `GridSearchCV` (5 folds, **solo entrenamiento**) entre: kernel **lineal** con `C ∈ {0.01, 0.1, 1, 10}` y kernel **RBF** con `C ∈ {1, 10, 100}` y `γ ∈ {0.01, 0.1, 1}`. Para esto usa una lista de dos diccionarios en la rejilla (`"svc__kernel"`).
2. Reporta la matriz de confusión, la exactitud, la sensibilidad y la especificidad de la configuración elegida **en la prueba (una sola vez)** y compárala con la logística sin penalización (`t = 0.5`) y con la línea base «siempre benigno».
3. Cada **falso negativo** (maligno no detectado) cuesta 10 veces un falso positivo. Calcula el costo `10·FN + FP` del modelo del punto 2 y de la logística; busca con `cross_val_predict` (solo entrenamiento) el umbral de la logística que minimiza ese costo y evalúalo en la prueba.
4. Responde sin código: ¿qué dirías de la validez clínica de estos resultados? ¿Qué corregirías en el material de clase antes de dárselo a otra persona?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis correcciones:**

""")
solution("""
# 1. búsqueda solo en entrenamiento (dos rejillas: lineal y radial)
pipe = Pipeline([("esc", StandardScaler()), ("svc", SVC())])
rejillas = [{"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]},
            {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": [0.01, 0.1, 1]}]
gs = GridSearchCV(pipe, rejillas, cv=5, scoring="accuracy").fit(X_tr, y_tr)
print("mejor:", gs.best_params_, "| CV:", round(gs.best_score_, 4))

# 2. prueba, una sola vez
c_svm = gs.predict(X_te)
tn, fp, fn, tp = confusion_matrix(y_te, c_svm, labels=[0, 1]).ravel()
print(f"SVM elegida: matriz {tn, fp, fn, tp} | exactitud {accuracy_score(y_te, c_svm):.4f} | sensibilidad {tp / (tp + fn):.4f} | especificidad {tn / (tn + fp):.4f}")
print("logística (0.5): exactitud", round(accuracy_score(y_te, clase_log), 4), "| línea base:", round(accuracy_score(y_te, base), 4))

# 3. costo 10·FN + FP
def costo(clase):
    tn_, fp_, fn_, tp_ = confusion_matrix(y_te, clase, labels=[0, 1]).ravel()
    return 10 * fn_ + fp_
pipe_log = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=2000))
p_cv = cross_val_predict(pipe_log, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), method="predict_proba")[:, 1]
umbrales = np.linspace(0.02, 0.98, 97)
costos_cv = [10 * int(((p_cv < u) & (y_tr == 1)).sum()) + int(((p_cv >= u) & (y_tr == 0)).sum()) for u in umbrales]
u_costo = float(umbrales[int(np.argmin(costos_cv))])
p_prueba = pipe_log.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print("costo en prueba | SVM elegida:", costo(c_svm), "| logística 0.5:", costo((p_prueba >= 0.5).astype(int)), f"| logística con t = {u_costo:.2f} (elegido en CV):", costo((p_prueba >= u_costo).astype(int)))
tn, fp, fn, tp = confusion_matrix(y_te, (p_prueba >= u_costo).astype(int), labels=[0, 1]).ravel()
print(f"con t = {u_costo:.2f}: FN {fn}, FP {fp} | sensibilidad {tp / (tp + fn):.4f} | especificidad {tn / (tn + fp):.4f}")
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia** (partición del profesor): la búsqueda elige `C = 100, γ = 0.01` con kernel RBF (exactitud de validación cruzada 0.9447; empata con otras combinaciones). En la prueba da 104, 3, 6, 58: exactitud 0.9474, sensibilidad 0.9062 y especificidad 0.9720, **menos** que el radial del profesor (0.9708 y 0.9375): una diferencia de 1 a 3 casos que no prueba nada. La logística con `t = 0.5` comete 6 FN y 4 FP (costo 10·6 + 4 = 64) y la SVM elegida 6 FN y 3 FP (costo 63). Con un costo 10:1 el umbral que minimiza el costo en validación cruzada es muy bajo (0.03 en esta corrida): en la prueba da 1 FN y 30 FP (costo 40; sensibilidad 0.984, especificidad 0.720). El costo baja, pero a cambio de muchas falsas alarmas y con una curva de costo ruidosa; el umbral óptimo depende del costo de cada error, no solo de la exactitud. **Correcciones**: (1) la base tiene 569 casos, no 357 ni 559; (2) «siempre uno» no acierta 107 (es «siempre benigno»); (3) −1/+1 es solo una convención; (4) el MLE no siempre es único (separación perfecta); (5) el polinomial tiene γ, r y d; (6) la interpretación de la suavidad en la diapositiva 28 y la escala de los coeficientes de R frente a Python; (7) elegir C, γ y t con validación cruzada y no con la prueba, y no sobreinterpretar diferencias de 1–3 puntos con 171 casos de prueba; (8) `probability=True` calibra, pero puede discrepar de `predict`. **Validez clínica**: es una base histórica de una región (no representa a la población chilena), con 4 de las 30 variables y una sola partición: sirve para aprender, no como herramienta clínica.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué diferencia hay entre la probabilidad estimada y la clase predicha en la regresión logística? ¿Qué pasa con la probabilidad al cambiar el umbral?
2. Define exactitud, sensibilidad, especificidad y precisión desde la matriz de confusión. ¿Cuál es la línea base «siempre benigno» y por qué importa?
3. En una SVM, ¿qué miden `f(x)`, la distancia `|f(x)|/‖w‖` y el ancho del margen? ¿Qué son los vectores de soporte?
4. ¿Qué hace C? ¿Qué pasa con el margen y los soportes al subirlo? ¿Por qué no basta con «mientras más, mejor»?
5. ¿Para qué sirve el kernel radial y qué controla gamma? ¿Qué ocurre con gamma muy grande?
6. ¿Por qué se estandariza con la media y la sd del entrenamiento y por qué va dentro de un `Pipeline`? ¿Qué es una fuga de información?
7. ¿Cómo se elige C y gamma sin usar la prueba? ¿Qué diferencia hay entre `decision_function` y `predict_proba` de una SVM?

<details><summary><b>Respuestas</b></summary>

1. La probabilidad la calcula el modelo (p = 1/(1 + e^(−η))); la clase la decide un umbral (clase 1 si p ≥ t). Cambiar el umbral cambia la decisión, **no** la probabilidad.
2. Exactitud (VP + VN)/n; sensibilidad VP/(VP + FN); especificidad VN/(VN + FP); precisión VP/(VP + FP). Siempre benigno acierta 107/171 = 0.626 y detecta 0 malignos: es la referencia mínima que el modelo debe superar.
3. `f(x) = w·x + b` es un puntaje (su signo da la clase); `|f(x)|/‖w‖` es la distancia geométrica a la frontera; el margen mide 2/‖w‖. Los vectores de soporte son los casos sobre el margen o dentro de él (`y·f ≤ 1`): solo ellos determinan la solución.
4. C pondera las violaciones del margen: C pequeño las tolera (margen ancho, regularización fuerte, riesgo de subajuste); C grande las castiga (margen estrecho, menos soportes, riesgo de sobreajuste). Se elige con validación.
5. Permite fronteras curvas sin construir φ(x) (truco del kernel). γ fija el alcance de la similitud exp(−γ‖x − z‖²): γ grande da fronteras muy locales que memorizan (sobreajuste); γ pequeño, casi lineales.
6. Porque la SVM usa distancias y las escalas distintas se dominan; la media y la sd se aprenden solo con el entrenamiento para que nada de la prueba (o de los folds de validación) influya. Un `Pipeline` repite eso en cada fold. Fuga: usar información de datos de validación o prueba para ajustar un paso del modelo.
7. Con validación cruzada solo en entrenamiento (`GridSearchCV` con un `Pipeline`) y se evalúa una sola vez en prueba. `decision_function` es un puntaje (no es probabilidad); `predict_proba` (con `probability=True`) es una calibración de Platt que puede discrepar de `predict`.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 38 · Evaluación de modelos supervisados y selección de modelos (Prof. Eloy Alvarado). Revisa `Guias/apuntes/C37_Ejercicios_Cuaderno_SVM.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
