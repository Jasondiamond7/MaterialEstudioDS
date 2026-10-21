"""Genera el notebook guiado de evaluación y selección de modelos (Clase 38).

Uso normal:      python notebook_c38.py             -> Guias/notebooks/C38_EvaluacionModelos.ipynb
Modo de prueba:  C38_TEST=1 python notebook_c38.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c38_numeros.json (lo escribe figuras_c38.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib, scipy, scikit-learn y statsmodels (solo en C5).
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C38_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C38_EvaluacionModelos_test.ipynb" if TEST else "C38_EvaluacionModelos.ipynb")
N = json.loads((HERE / "c38_numeros.json").read_text(encoding="utf-8"))
cells = []


def val(m):
    obj = N
    for k in m.group(1).split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    d = m.group(2)
    return str(obj) if d is None else f"{float(obj):.{int(d)}f}"


def fill(text):
    return re.sub(r"\[\[([\w.]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- encabezado
md("""
# Clase 38 · Evaluación de modelos supervisados y selección de modelos
**Diplomado Data Science** · Prof. Eloy Alvarado · consolida las Clases 30 (partición y validación), 31 (sobreajuste), 34–37 (árboles, bosques, SVM) y 22 (inferencia)

Material de apoyo:
- `Guias/C38_Guia_EvaluacionModelos.pdf` — guía visual de la clase.
- `Guias/apuntes/C38_Apuntes_EvaluacionModelos.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C38_Ejercicios_Cuaderno_EvaluacionModelos.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta «Clase 38 …»): `DipDS_2026_S38.pdf` (diapositivas), `38_evaluacion_seleccion.py` y `.R.txt`, `hormigon_s38.csv` y `spam_s38.csv`.
- Transcripción: `VideosClases/Clase 38 - … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` y `statsmodels` (solo en C5). Si no los tienes:
```
pip install numpy pandas matplotlib scipy scikit-learn statsmodels
```

**Datos**: `hormigon_s38.csv` (1 030 mediciones de resistencia del hormigón en MPa, 8 predictores) y `spam_s38.csv` (4 601 correos, 57 predictores). Ambos traen la columna `conjunto` (entrenamiento o prueba), `grupo` (filas que deben ir siempre juntas) y `fold` (pliegue de validación cruzada): **la partición del profesor ya está hecha** y este notebook la usa tal cual. El notebook busca los archivos en la carpeta de la Clase 38 sin modificarla.

### Objetivos
1. Definir **qué es «mejor»**: pérdida, métricas (MAE, RMSE, R², log loss, Brier, costo) y referencia, **antes** de mirar resultados.
2. Armar **candidatos como recetas** (`Pipeline`) y compararlos con **predicciones fuera de pliegue (OOF)** sin fuga de datos.
3. Reproducir los resultados del profesor (regresión y clasificación) y la regla de elegir, **reajustar** y **evaluar una vez**.
4. Entender **sesgo, varianza y ruido**, curvas de aprendizaje y de validación.
5. Conocer **variantes de validación**: K-fold, dejar un grupo fuera, bootstrap y estimador .632, y la **validación anidada**.
6. Leer **ROC, precision–recall, calibración** y elegir un **umbral por costo**; comparar **AIC/BIC** con la validación cruzada.
7. Traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Datos y partición por composición |
| A | 10 | A2 · Pérdida y métricas a mano |
| A | 20 | A3 · Siete candidatos y predicciones OOF |
| A | 15 | A4 · Curva de validación y diferencias por pliegue |
| A | 15 | A5 · Reajustar, evaluar una vez y diagnosticar |
| B | 10 | B1 · Spam: datos y referencia |
| B | 20 | B2 · Probabilidades OOF: log loss y Brier |
| B | 15 | B3 · ROC, AUC y precision–recall |
| B | 10 | B4 · Calibración |
| B | 15 | B5 · Costo y umbral |
| B | 10 | B6 · Reajustar y evaluar en prueba |
| C | 15 | C1 · Sesgo, varianza y ruido |
| C | 15 | C2 · Curvas de aprendizaje y de validación |
| C | 15 | C3 · Particiones repetidas, K y dejar un grupo fuera |
| C | 15 | C4 · Bootstrap y estimador .632 |
| C | 15 | C5 · AIC y BIC frente a validación cruzada |
| C | 20 | C6 · Validación anidada |
| C | 15 | C7 · Recalibrar y umbral teórico |
| C | 10 | C8 · Casi-duplicados entre entrenamiento y prueba |
| D | 15 | D1 · Alertas del material, verificadas con código |
| D | 10 | D2 · Tabla de correspondencia R ↔ Python |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Las simulaciones usan semilla fija; con otras versiones de las librerías pueden variar en la última cifra.
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
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 2026   # semilla para todo lo aleatorio
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso de la API (Clases 30 y 31)
- **`Pipeline`** encadena pasos: en cada `fit` aprende solo con las filas que recibe, así que dentro de una validación cruzada la escala se aprende **en cada pliegue**, sin ver el de validación (sin fuga de datos).
- **OOF** (*out-of-fold*): la predicción de cada fila viene de un modelo ajustado **sin su pliegue**.
- **Grupo**: filas que no pueden separarse entre ajuste y validación (aquí, una misma composición de hormigón con distintas edades).
- Todo lo aleatorio usa `np.random.default_rng(semilla)`.

### Funciones comunes
Las usaremos en todo el notebook. Son las mismas ideas del script del profesor, escritas con nombres propios.
""")
code("""
def buscar_clase():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 38*/hormigon_s38.csv"):
            return p.parent
        if (base / "hormigon_s38.csv").exists():
            return base
    raise FileNotFoundError("No encuentro hormigon_s38.csv: abre el notebook dentro de la carpeta del curso.")


CARPETA = buscar_clase()
V8 = ["cemento", "escoria", "ceniza", "agua", "superplastificante", "agregado_grueso", "agregado_fino", "edad"]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def ridge_cuad(alpha, grado=2):
    # receta del profesor: estandarizar, expandir términos, estandarizar de nuevo, ridge
    return make_pipeline(StandardScaler(), PolynomialFeatures(grado, include_bias=False), StandardScaler(), Ridge(alpha=alpha, solver="svd"))


def candidatos_reg():
    m = {"MEDIA": DummyRegressor(strategy="mean"), "LINEAL": make_pipeline(StandardScaler(), LinearRegression())}
    for a in (0.1, 1, 10, 100, 1000):
        m[f"Q_A{a:g}"] = ridge_cuad(a)
    return m


def oof_reg(modelo, x, y, fold):
    # predicción fuera de pliegue: cada fila la predice un modelo que no vio su pliegue
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        v = fold == k
        oof[v] = clone(modelo).fit(x[~v], y[~v]).predict(x[v])
    return oof


def folds_por_grupo(grupos, K, rng):
    # reparte GRUPOS completos en K pliegues al azar
    ug = np.unique(grupos)
    asign = dict(zip(rng.permutation(ug), np.arange(len(ug)) % K))
    return np.array([asign[g] for g in grupos])


def decision(y, p, t, cfp=5.0, cfn=1.0):
    vn, fp, fn, vp = confusion_matrix(y, np.asarray(p) >= t, labels=[0, 1]).ravel()
    n = vn + fp + fn + vp
    return dict(umbral=float(t), VN=int(vn), FP=int(fp), FN=int(fn), VP=int(vp), accuracy=(vn + vp) / n,
                precision=vp / (vp + fp) if vp + fp else 0.0, recall=vp / (vp + fn), F1=2 * vp / (2 * vp + fp + fn), costo=(cfp * fp + cfn * fn) / n)


def probs(y, p):
    return dict(log_loss=log_loss(y, p, labels=[0, 1]), Brier=brier_score_loss(y, p), AUC=roc_auc_score(y, p), AP=average_precision_score(y, p))


print("Material:", CARPETA.name)
""")

# ================================================================== PARTE A
md("""
## Parte A · Regresión: la resistencia del hormigón

### A1 · Datos y partición por composición (15 min)
Pregunta: **¿cuánto nos equivocaremos en una mezcla que quedó fuera del ajuste?** El archivo trae 1 030 mediciones; varias tienen **los mismos siete ingredientes** y solo cambia la edad. Si esas filas quedaran repartidas entre ajuste y validación, el modelo vería la misma mezcla en los dos lados. Por eso el profesor las agrupa por igualdad exacta (`grupo`) y reserva **grupos completos**: 342 composiciones para ajustar (806 filas, en 5 pliegues) y 86 para la prueba (224 filas).
""")
code("""
h = pd.read_csv(CARPETA / "hormigon_s38.csv")
htr = h[h.conjunto == "entrenamiento"].reset_index(drop=True)
hte = h[h.conjunto == "prueba"].reset_index(drop=True)
assert len(htr) == 806 and len(hte) == 224
assert set(htr.grupo).isdisjoint(hte.grupo)                 # ninguna composición a ambos lados
assert htr.groupby("grupo").fold.nunique().eq(1).all()      # cada composición vive en un solo pliegue
print(f"Entrenamiento: {len(htr)} filas, {htr.grupo.nunique()} composiciones | Prueba: {len(hte)} filas, {hte.grupo.nunique()} composiciones")
print("Filas por pliegue:", htr.fold.value_counts().sort_index().to_dict())
x_tr, y_tr, g_tr, f_tr = htr[V8].to_numpy(), htr.resistencia.to_numpy(), htr.grupo.to_numpy(), htr.fold.to_numpy()
x_te, y_te = hte[V8].to_numpy(), hte.resistencia.to_numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].scatter(htr.edad, htr.resistencia, s=12, alpha=0.5)
axs[0].set(xscale="log", xlabel="edad (días, escala log)", ylabel="resistencia (MPa)", title="Solo entrenamiento")
axs[1].hist(htr.resistencia, bins=20, alpha=0.85)
axs[1].set(xlabel="resistencia (MPa)", ylabel="mediciones", title="La respuesta es continua")
plt.tight_layout(); plt.show()
""")
md("""
Las columnas `id`, `grupo`, `conjunto` y `fold` **nunca** son predictores: solo describen la partición. La resistencia va de unos 5 a unos 80 MPa y se concentra entre 30 y 40; si el entrenamiento no cubriera las colas, el modelo no podría predecirlas.

### ✍️ Ejercicio A1
Comprueba con código que **cada pliegue** tiene entre 161 y 162 filas, que el número de composiciones por pliegue es parecido y que ninguna composición del entrenamiento aparece en dos pliegues.
""")
code("""
# tu código aquí
""")
solution("""
print(htr.groupby("fold").agg(filas=("id", "size"), composiciones=("grupo", "nunique")))
print("composiciones en más de un pliegue:", int((htr.groupby("grupo").fold.nunique() > 1).sum()))
""", "Cada pliegue tiene 161–162 filas y unas 68 composiciones; ninguna composición está en dos pliegues (0).")

md("""
### A2 · Pérdida y métricas a mano (10 min)
La **pérdida** mide el error de un caso: `L(y, ŷ) = (y − ŷ)²`. El **riesgo** es su promedio sobre la población de interés. Las métricas resumen los errores de una muestra: **MAE** (promedio de |error|, pesa linealmente), **RMSE** (raíz del promedio del cuadrado, castiga los errores grandes) y **R²** (compara con predecir la media). Ejemplos de la clase:
""")
code("""
# 1) cambiar la métrica puede cambiar el ganador: A tiene MAE 2 y RMSE 4; B tiene MAE 3 y RMSE 3
A = np.array([0, 0, 0, 8.0]); B = np.array([3, 3, 3, 3.0])
for nombre, e in [("A", A), ("B", B)]:
    print(f"{nombre}: MAE = {np.mean(np.abs(e)):.0f} | RMSE = {np.sqrt(np.mean(e ** 2)):.0f}")

# 2) una correlación alta puede ocultar errores grandes
y = np.array([1.0, 2.0, 3.0]); yh = y + 10
print(f"correlación = {np.corrcoef(y, yh)[0, 1]:.0f} | RMSE = {rmse(y, yh):.0f} | R² = {r2(y, yh):.0f}   (SSE = {np.sum((y - yh) ** 2):.0f}, SST = {np.sum((y - y.mean()) ** 2):.0f})")
""")
md("""
Con **MAE** gana A (2 < 3) y con **RMSE** gana B (3 < 4): el criterio se define **antes** de ver resultados. La predicción `ŷ = y + 10` tiene correlación 1 pero RMSE 10 y **R² = −149**: «R² entre 0 y 1» no es una regla (puede ser negativo cuando el modelo es peor que predecir la media del conjunto evaluado).

### ✍️ Ejercicio A2
Con `y = (12, 15, 9, 20, 14)` y `ŷ = (10, 16, 12, 17, 15)` calcula MAE, RMSE y R² a mano (con NumPy) y comprueba con `sklearn.metrics`.
""")
code("""
# tu código aquí
""")
solution("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y2 = np.array([12, 15, 9, 20, 14.0]); p2 = np.array([10, 16, 12, 17, 15.0])
print(f"a mano: MAE {mae(y2, p2):.3f} | RMSE {rmse(y2, p2):.3f} | R² {r2(y2, p2):.3f}")
print(f"sklearn: MAE {mean_absolute_error(y2, p2):.3f} | RMSE {mean_squared_error(y2, p2) ** 0.5:.3f} | R² {r2_score(y2, p2):.3f}")
""", "MAE 2.000, RMSE 2.191 y R² 0.636 (SSE 24 y SST 66). El RMSE supera al MAE porque los errores no son todos iguales.")

md("""
### A3 · Siete candidatos y predicciones OOF (20 min)
Cada candidato es una **receta completa**: (1) el modelo constante (media de entrenamiento), (2) regresión lineal múltiple y (3) regresión cuadrática con penalización **ridge** (8 lineales + 8 cuadrados + 28 interacciones = 44 términos) con α = 0.1, 1, 10, 100 y 1000. Para cada uno se hacen **5 ajustes** (uno por pliegue) y cada fila recibe una predicción OOF: 35 ajustes en total. La regla se fija **antes**: gana el menor **RMSE OOF**.
""")
code("""
modelos = candidatos_reg()
filas = []
oof_reg_todos = {}
for orden, (nombre, m) in enumerate(modelos.items()):
    o = oof_reg(m, x_tr, y_tr, f_tr)
    oof_reg_todos[nombre] = o
    ajuste = np.sqrt(np.mean(np.concatenate([(y_tr[f_tr != k] - clone(m).fit(x_tr[f_tr != k], y_tr[f_tr != k]).predict(x_tr[f_tr != k])) ** 2 for k in range(1, 6)])))
    filas.append(dict(modelo=nombre, orden=orden, RMSE=rmse(y_tr, o), MAE=mae(y_tr, o), R2=r2(y_tr, o), RMSE_ajuste=float(ajuste)))
ranking = pd.DataFrame(filas).sort_values(["RMSE", "orden"]).reset_index(drop=True)
display(ranking[["modelo", "RMSE", "MAE", "R2", "RMSE_ajuste"]].round(3))
ganador = ranking.modelo.iloc[0]
print("Elegido por CV:", ganador, "| ventaja sobre el segundo:", round(ranking.RMSE.iloc[1] - ranking.RMSE.iloc[0], 3), "MPa")
""")
md("""
Reproduce la diapositiva 26 del profesor: gana la cuadrática con **α = 0.1 (RMSE OOF [[cv.rmse_oof_01|3]])**, seguida de α = 1 ([[cv.rmse_oof_1|3]]); la media de entrenamiento obtiene 16.746. La ventaja de α = 0.1 sobre α = 1 es solo ≈ 0.079 MPa: **elegir no prueba superioridad estable**. Como en el script, MAE y R² describen pero no cambian la regla.

#### El RMSE OOF no es el promedio de los RMSE por pliegue
El RMSE OOF reúne **todos** los errores al cuadrado y saca **una** raíz; el promedio de RMSE saca raíz en cada pliegue y luego promedia. La raíz no es lineal, así que no coinciden.
""")
code("""
o = oof_reg_todos["Q_A0.1"]
por_pliegue = [rmse(y_tr[f_tr == k], o[f_tr == k]) for k in range(1, 6)]
print("RMSE por pliegue:", np.round(por_pliegue, 3))
print(f"promedio de RMSE = {np.mean(por_pliegue):.3f} | RMSE OOF = {rmse(y_tr, o):.3f}")
# el ejemplo de la diapositiva 21: dos pliegues de igual tamaño con MSE 1 y 9
print(f"ejemplo: promedio de RMSE = {(np.sqrt(1) + np.sqrt(9)) / 2:.3f} | RMSE OOF = {np.sqrt((1 + 9) / 2):.3f}")
""")
md("""
Ambos son válidos pero **no intercambiables**; el profesor selecciona con el **RMSE OOF** (cada *medición* pesa igual; los grupos solo controlan la separación).

### ✍️ Ejercicio A3
Verifica la afirmación «el preprocesamiento aprende en cada pliegue»: para el candidato `Q_A0.1` y el pliegue 1, comprueba que la media que aprendió el primer `StandardScaler` es la media de las filas de **ajuste** (no la de todas las filas).
""")
code("""
# tu código aquí
""")
solution("""
v = f_tr == 1
m1 = clone(modelos["Q_A0.1"]).fit(x_tr[~v], y_tr[~v])
media_aprendida = m1.steps[0][1].mean_
print("coincide con las filas de ajuste:", np.allclose(media_aprendida, x_tr[~v].mean(0)))
print("coincide con TODAS las filas:    ", np.allclose(media_aprendida, x_tr.mean(0)))
""", "La primera es `True` y la segunda `False`: la escala se aprende solo con las filas de ajuste del pliegue. Eso es lo que evita la fuga de datos.")

md("""
### A4 · Curva de validación y diferencias por pliegue (15 min)
La figura compara el error de **ajuste** (filas usadas para estimar) con el de **validación** (OOF) según α. Luego se desagrega la diferencia entre los dos mejores por pliegue.
""")
code("""
q = ranking[ranking.modelo.str.startswith("Q_")].copy()
q["alpha"] = q.modelo.str.replace("Q_A", "").astype(float)
q = q.sort_values("alpha")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(q.alpha, q.RMSE_ajuste, "o-", label="ajuste")
axs[0].plot(q.alpha, q.RMSE, "o-", label="validación (OOF)")
axs[0].set(xscale="log", xlabel="alpha (más penalización a la derecha)", ylabel="RMSE (MPa)", title="Error de ajuste y de validación")
axs[0].legend()
o1, o2 = oof_reg_todos["Q_A0.1"], oof_reg_todos["Q_A1"]
delta = np.array([rmse(y_tr[f_tr == k], o1[f_tr == k]) - rmse(y_tr[f_tr == k], o2[f_tr == k]) for k in range(1, 6)])
axs[1].bar(range(1, 6), delta, color=np.where(delta < 0, "tab:green", "tab:purple")); axs[1].axhline(0, c="k", lw=1)
axs[1].set(xlabel="pliegue", ylabel="RMSE(α = 0.1) − RMSE(α = 1)", title="Bajo 0 favorece α = 0.1")
plt.tight_layout(); plt.show()
print("diferencias por pliegue:", np.round(delta, 3), "| α = 0.1 gana en", int((delta < 0).sum()), "de 5")
""")
md("""
Con α = 1000 suben **ambos** errores: la penalización restringe demasiado (sesgo alto). A la izquierda el error de validación se estabiliza. **α = 0.1 gana en 3 de 5 pliegues** (2, 3 y 4) y **α = 1 gana en el 1 y el 5**: la regla elige por el resumen, no porque gane en cada pliegue. Los ajustes comparten datos: cinco diferencias no son cinco réplicas independientes ni un intervalo de confianza. (En la clase se dijo «en el primero y el tercero gana α = 1»; los datos muestran que son el primero y el quinto.)

### ✍️ Ejercicio A4
El mejor α está en el **borde** de la cuadrícula del profesor. Repite la validación con α = 0.0001, 0.001 y 0.01 y comprueba si vale la pena bajar más.
""")
code("""
# tu código aquí
""")
solution("""
for a in (1e-4, 1e-3, 1e-2, 0.1):
    print(f"alpha = {a:g}: RMSE OOF = {rmse(y_tr, oof_reg(ridge_cuad(a), x_tr, y_tr, f_tr)):.3f}")
""", "Baja de [[cv.rmse_oof_01|3]] (α = 0.1) a ≈ 8.219 con α = 10⁻⁴: una ganancia de ≈ 0.01 MPa, sin valor práctico. El ganador está en el borde de la cuadrícula, pero la mejora posible es mínima.")

md("""
### A5 · Reajustar, evaluar una vez y diagnosticar (15 min)
Elegida la receta, se **reajusta** (se vuelven a estimar escalas y coeficientes) con las 806 filas de entrenamiento y se evalúa **una sola vez** en las 224 filas reservadas. No se promedian los cinco modelos de la validación cruzada: sería otro procedimiento. La **referencia** se fija desde el inicio: la media de **entrenamiento** (no la de prueba).
""")
code("""
final = clone(modelos[ganador]).fit(x_tr, y_tr)
p_te = final.predict(x_te)
base = DummyRegressor(strategy="mean").fit(x_tr, y_tr)
p_base = base.predict(x_te)
tabla = pd.DataFrame({"RMSE": [rmse(y_te, p_te), rmse(y_te, p_base)], "MAE": [mae(y_te, p_te), mae(y_te, p_base)], "R2": [r2(y_te, p_te), r2(y_te, p_base)]},
                     index=[f"{ganador} (elegido)", "media de entrenamiento"])
display(tabla.round(3))
print(f"El RMSE baja {100 * (1 - rmse(y_te, p_te) / rmse(y_te, p_base)):.1f} % frente a la media | RMSE en entrenamiento del ajuste final: {rmse(y_tr, final.predict(x_tr)):.2f}")

res = y_te - p_te
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].scatter(y_te, p_te, s=14, alpha=0.6); lim = [y_te.min() - 2, y_te.max() + 2]; axs[0].plot(lim, lim, "--", c="tab:orange")
axs[0].set(xlabel="real (MPa)", ylabel="predicha (MPa)", title="Predicciones en prueba")
axs[1].scatter(p_te, res, s=14, alpha=0.6); axs[1].axhline(0, c="tab:orange", ls="--"); axs[1].set(xlabel="predicha", ylabel="real − predicha", title="Residuos (positivo = subestima)")
axs[2].scatter(hte.edad, np.abs(res), s=14, alpha=0.6); axs[2].set(xscale="log", xlabel="edad (días)", ylabel="error absoluto (MPa)", title="Errores según edad")
plt.tight_layout(); plt.show()
""")
md("""
Reproduce la diapositiva 31: **RMSE [[hormigon.prueba.rmse|3]], MAE [[hormigon.prueba.mae|3]] y R² [[hormigon.prueba.r2|3]]**; la media de entrenamiento obtiene RMSE [[hormigon.referencia.rmse|3]] y R² ≈ −0.007: usar los predictores baja el RMSE 46.9 %. El RMSE de entrenamiento (7.06) es menor que el de prueba, como se espera. Los residuos positivos son **subestimaciones** (observado mayor que predicho). Si el resultado de prueba no fuera satisfactorio, **no se vuelve a elegir con ella**: se diseña una nueva selección con evaluación independiente.

### ✍️ Ejercicio A5
¿Cuántas de las 224 predicciones de prueba **subestiman** (residuo positivo) y cuál es el error absoluto máximo? ¿Son los errores más grandes de edades pequeñas o grandes?
""")
code("""
# tu código aquí
""")
solution("""
print("subestiman:", int((res > 0).sum()), "de", len(res), f"({(res > 0).mean():.0%})")
i = int(np.argmax(np.abs(res)))
print(f"error absoluto máximo: {np.abs(res).max():.1f} MPa (edad {hte.edad.iloc[i]} días, real {y_te[i]:.1f}, predicha {p_te[i]:.1f})")
print(hte.assign(abs_err=np.abs(res)).groupby(pd.cut(hte.edad, [0, 7, 28, 90, 400])).abs_err.mean().round(2))
""", "97 de 224 subestiman (43 %); el promedio del residuo es ≈ −1.1 MPa (la media global sobreestima). El error máximo es ≈ 32.8 MPa. Mira la tabla por edad: los errores medios cambian con la edad, así que el RMSE global esconde un patrón.")

# ================================================================== PARTE B
md("""
## Parte B · Clasificación: detectar spam

### B1 · Spam: datos y referencia (10 min)
Se trabaja con 4 601 correos y 57 entradas numéricas (frecuencias de palabras y caracteres y medidas de mayúsculas). **spam = 1**, legítimo = 0. Se reservan 921 correos para la prueba y 3 680 para ajustar y seleccionar. El modelo no devuelve «sí/no»: devuelve una **probabilidad**, y después hace falta un **umbral** para decidir. La referencia más simple es la **frecuencia** de spam (una probabilidad constante).
""")
code("""
s = pd.read_csv(CARPETA / "spam_s38.csv")
vs = [c for c in s if c.startswith(("word_freq_", "char_freq_", "capital_run_"))]
str_ = s[s.conjunto == "entrenamiento"].reset_index(drop=True)
ste = s[s.conjunto == "prueba"].reset_index(drop=True)
assert len(vs) == 57 and len(str_) == 3680 and len(ste) == 921 and set(str_.grupo).isdisjoint(ste.grupo)
xs_tr, ys_tr, fs_tr, gs_tr = str_[vs].to_numpy(), str_.spam.to_numpy(), str_.fold.to_numpy(), str_.grupo.to_numpy()
xs_te, ys_te = ste[vs].to_numpy(), ste.spam.to_numpy()
print("correos de entrenamiento por clase:", np.bincount(ys_tr), f"-> spam = {ys_tr.mean():.1%}")
print("Costos hipotéticos (se declaran ANTES de comparar): FP = 5, FN = 1")
""")
md("""
Hay un 39.4 % de spam en el entrenamiento: algo desbalanceado, pero no extremo. Con eventos muy raros (fraude: 10 en 10 000) el modelo apenas ve ejemplos y el *accuracy* engaña. **Costos**: un correo legítimo bloqueado (**FP**) pesa 5 y un spam que pasa (**FN**) pesa 1. Es una razón **arbitraria** pensada para el ejemplo; en un caso real se estimaría con el negocio.

### B2 · Probabilidades OOF: log loss y Brier (20 min)
Tres candidatos: la **frecuencia** (referencia), la **regresión logística** (C = 1, con escala aprendida dentro del ajuste) y un **bosque aleatorio** (150 árboles, profundidad 8, mínimo 5 casos por hoja). Se comparan por **log loss OOF**; el **Brier** describe. Antes, a mano, qué castiga cada pérdida:
""")
code("""
# un spam real (y = 1): predecir 0.9 o 0.1
for p in (0.9, 0.1):
    print(f"p = {p}: log loss = {-np.log(p):.3f} | Brier = {(1 - p) ** 2:.2f}")
""")
md("Predecir 0.9 para un spam real cuesta 0.105 (log loss) y 0.01 (Brier); predecir 0.1 cuesta 2.303 y 0.81: el log loss castiga mucho más la certeza equivocada. Ahora los tres candidatos:")
code("""
def modelos_spam():
    return {"FRECUENCIA": DummyClassifier(strategy="prior"),
            "LOGISTICA_C1": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, solver="lbfgs")),
            "BOSQUE_8": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, max_features="sqrt", random_state=20260924, n_jobs=1)}


oof_p = {}
filas = []
for orden, (nombre, m) in enumerate(modelos_spam().items()):
    p = np.zeros(len(ys_tr))
    for k in range(1, 6):
        v = fs_tr == k
        assert set(gs_tr[v]).isdisjoint(gs_tr[~v])
        p[v] = clone(m).fit(xs_tr[~v], ys_tr[~v]).predict_proba(xs_tr[v])[:, 1]
    oof_p[nombre] = p
    filas.append(dict(modelo=nombre, orden=orden, **probs(ys_tr, p)))
rank_s = pd.DataFrame(filas).sort_values(["log_loss", "orden"]).reset_index(drop=True)
display(rank_s.round(3))
ganador_s = rank_s.modelo.iloc[0]
p_oof = oof_p[ganador_s]
print("Elegido por log loss OOF:", ganador_s)
""")
md("""
Reproduce la diapositiva 50 (columnas de Python): log loss **0.671** (frecuencia), **0.275** (logística) y **[[spam.ranking.BOSQUE_8.log_loss|3]]** (bosque); Brier 0.239, 0.075 y 0.061. Gana el bosque. En R (`ranger`) el bosque da 0.223 y 0.060: la diferencia es de **implementación**, no de método; por eso no se mezclan cifras de los dos lenguajes. La frecuencia, que no usa predictores, es el piso para valorar cuánto aportan los 57 predictores.

### ✍️ Ejercicio B2
Calcula a mano (NumPy) el log loss y el Brier de la logística OOF y comprueba con `log_loss` y `brier_score_loss`.
""")
code("""
# tu código aquí
""")
solution("""
p = np.clip(oof_p["LOGISTICA_C1"], 1e-15, 1 - 1e-15)
ll = -np.mean(ys_tr * np.log(p) + (1 - ys_tr) * np.log(1 - p))
br = np.mean((ys_tr - p) ** 2)
print(f"a mano: log loss {ll:.4f} | Brier {br:.4f}")
print(f"sklearn: log loss {log_loss(ys_tr, oof_p['LOGISTICA_C1']):.4f} | Brier {brier_score_loss(ys_tr, oof_p['LOGISTICA_C1']):.4f}")
""", "Log loss 0.2747 y Brier 0.0752 con las dos formas.")

md("""
### B3 · ROC, AUC y precision–recall (15 min)
Cada punto de la curva **ROC** (recall frente a tasa de falsos positivos) y de la curva **precision–recall** corresponde a un **umbral**. El **AUC** y el **AP** (*average precision*) resumen el **ordenamiento**: no fijan un umbral ni certifican calibración. Primero a mano, con tres spam y tres legítimos:
""")
code("""
sc = np.array([0.9, 0.6, 0.4, 0.7, 0.3, 0.2]); yy = np.array([1, 1, 1, 0, 0, 0])
orden = np.argsort(-sc)
tp = np.cumsum(yy[orden]) / yy.sum(); fp = np.cumsum(1 - yy[orden]) / (1 - yy).sum()
fpr = np.r_[0, fp]; tpr = np.r_[0, tp]
print("puntos (FPR, recall):", [(round(float(a), 3), round(float(b), 3)) for a, b in zip(fpr, tpr)])
auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
pares = np.mean([(a > b) + 0.5 * (a == b) for a in sc[yy == 1] for b in sc[yy == 0]])
print(f"AUC por trapecios = {auc:.4f} | por pares = {pares:.4f} | roc_auc_score = {roc_auc_score(yy, sc):.4f}")
""")
md("El AUC es la probabilidad de que un positivo reciba mayor puntaje que un negativo: aquí 7 de 9 pares (0.778). Ahora el bosque con los correos OOF, marcando tres umbrales (0.50, 0.60 y 5/6):")
code("""
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for nombre in ("LOGISTICA_C1", "BOSQUE_8"):
    fp_, tp_, _ = roc_curve(ys_tr, oof_p[nombre]); axs[0].plot(fp_, tp_, label=f"{nombre} AUC {roc_auc_score(ys_tr, oof_p[nombre]):.3f}")
    pr_, rc_, _ = precision_recall_curve(ys_tr, oof_p[nombre]); axs[1].plot(rc_, pr_, label=f"{nombre} AP {average_precision_score(ys_tr, oof_p[nombre]):.3f}")
for t in (0.5, 0.6, 5 / 6):
    d = decision(ys_tr, p_oof, t)
    axs[0].scatter(d["FP"] / (d["FP"] + d["VN"]), d["recall"], s=50, zorder=5, label=f"umbral {t:.2f}")
    axs[1].scatter(d["recall"], d["precision"], s=50, zorder=5)
    axs[2].axvline(t, ls="--")
axs[0].plot([0, 1], [0, 1], "--", c="gray"); axs[0].set(xlabel="tasa de falsos positivos", ylabel="recall", title="ROC"); axs[0].legend(fontsize=7)
axs[1].axhline(ys_tr.mean(), ls="--", c="gray"); axs[1].set(xlabel="recall", ylabel="precision", title="Precision–recall"); axs[1].legend(fontsize=7)
axs[2].hist(p_oof[ys_tr == 0], bins=20, alpha=0.7, label="legítimos"); axs[2].hist(p_oof[ys_tr == 1], bins=20, alpha=0.7, label="spam")
axs[2].set(xlabel="probabilidad OOF de spam (bosque)", title="El umbral corta dos distribuciones"); axs[2].legend()
plt.tight_layout(); plt.show()
""")
md("""
Reproduce la diapositiva 51 (Python): **AUC [[roc_pr.auc.BOSQUE_8|3]] y AP [[roc_pr.ap.BOSQUE_8|3]]**. Una AUC de 0.975 **no** significa «97.5 % de aciertos»: resume el orden sobre todos los umbrales. Y el AP depende de la proporción de positivos (la línea punteada de la derecha es la frecuencia de spam).

### ✍️ Ejercicio B3
Con misma sensibilidad 80 % y misma tasa de FP 10 %, calcula la **precision** en 1 000 correos con 50 % de spam y con 5 % de spam. ¿Qué te dice sobre trasladar una precision de un corpus a otro?
""")
code("""
# tu código aquí
""")
solution("""
for prev in (0.5, 0.05):
    pos = 1000 * prev; neg = 1000 - pos
    vp, fp = 0.8 * pos, 0.1 * neg
    print(f"spam {prev:.0%}: VP = {vp:.0f}, FP = {fp:.0f} -> precision = {vp / (vp + fp):.1%}")
""", "88.9 % con 50 % de spam y 29.6 % con 5 %. La precision depende de la **prevalencia**: una precision medida en un corpus no se traslada sin más a otro.")

md("""
### B4 · Calibración (10 min)
Si el modelo dice 0.8, esperamos que ≈ 80 % de esos correos sean spam. La **calibración** compara probabilidad predicha con frecuencia observada por grupos. Un buen AUC **no garantiza** buena calibración.
""")
code("""
bins = pd.qcut(p_oof, q=5, duplicates="drop")
cal = pd.DataFrame({"p": p_oof, "y": ys_tr, "b": bins}).groupby("b", observed=True).agg(prob_media=("p", "mean"), fraccion_spam=("y", "mean"), n=("y", "size")).reset_index(drop=True)
display(cal.round(3))
plt.plot([0, 1], [0, 1], "--", c="gray", label="ideal"); plt.plot(cal.prob_media, cal.fraccion_spam, "o-", label="bosque OOF, 5 grupos")
plt.xlabel("probabilidad media predicha"); plt.ylabel("fracción observada de spam"); plt.legend(); plt.show()
""")
md("""
En el cuarto quintil el bosque predice **[[calibracion.q4_prob|3]]** (67 %) y se observa **[[calibracion.q4_frac|3]]** (81 %) de spam: **subestima** (bajo la diagonal sobreestimaría). Es la razón por la que, más abajo, el umbral óptimo empírico (0.60) queda lejos del teórico (0.833).

### B5 · Costo y umbral (15 min)
La **matriz de confusión** conserva los conteos; la **métrica de costo** decide cuánto pesa cada error: `costo = (5·FP + 1·FN) / n`, que se multiplica por 1 000 para comparar. Las probabilidades no cambian: solo cambia **desde qué valor** se dice que es spam. Se prueban 19 umbrales (0.05, 0.10, …, 0.95) y se elige el de menor costo OOF (empates: el mayor umbral).
""")
code("""
umbrales = np.round(np.arange(0.05, 0.96, 0.05), 2)
costos = pd.DataFrame([decision(ys_tr, p_oof, t) for t in umbrales])
elec = costos.sort_values(["costo", "umbral"], ascending=[True, False]).iloc[0]
umbral = float(elec.umbral)
print(f"Umbral elegido: {umbral:.2f} | costo OOF por 1000: {1000 * elec.costo:.1f}")
display((costos.set_index("umbral").loc[[0.5, 0.55, 0.6, 0.65], ["FP", "FN", "accuracy", "costo"]].assign(costo_por_1000=lambda d: 1000 * d.costo)).round(3))
plt.plot(costos.umbral, 1000 * costos.costo, "o-"); plt.axvline(0.5, ls="--", c="gray", label="0.50"); plt.axvline(umbral, ls="--", c="tab:orange", label=f"elegido {umbral:.2f}")
plt.axvline(5 / 6, ls=":", c="tab:purple", label="teórico 5/6"); plt.ylim(0, 600); plt.xlabel("umbral"); plt.ylabel("costo OOF por 1000"); plt.legend(); plt.show()
""")
md("""
Reproduce la diapositiva 53 (Python): costo **149.7** con 0.50, **132.3** con 0.55, **128.8** con 0.60 (mínimo) y **135.3** con 0.65. De 0.50 a 0.60: **30 FP menos y 73 FN más**; ahorro 5×30 − 73 = **77 unidades** aunque los aciertos bajan de 93.1 % a 91.9 %: se evitan los errores que pesan más. (En clase se leyeron «132 con 0.5»: el 132.3 es de 0.55.)

El **umbral teórico**: bloquear cuesta `5(1 − p)` en valor esperado y dejar pasar cuesta `p`; conviene bloquear si `5(1 − p) < p`, o sea `p > 5/6 ≈ 0.833`. Supone **probabilidades correctas**, costos conocidos y aciertos sin costo; el empírico (0.60) difiere por la mala calibración, la muestra finita y la cuadrícula finita.

### ✍️ Ejercicio B5
Repite la elección con costos FP = 2 y FN = 1 (razón 2:1). ¿Qué umbral resulta? ¿Se parece a 2/3?
""")
code("""
# tu código aquí
""")
solution("""
fino = np.round(np.arange(0.02, 0.99, 0.01), 2)
c2 = np.array([decision(ys_tr, p_oof, t, 2.0, 1.0)["costo"] for t in fino])
print("umbral óptimo con 2:1 =", fino[np.where(c2 <= c2.min() + 1e-12)[0][-1]], "| teórico 2/3 =", round(2 / 3, 3))
""", "El óptimo empírico con 2:1 es ≈ 0.53, por debajo de 2/3: el bosque subestima las probabilidades de spam, así que el umbral que minimiza el costo queda más bajo que el teórico.")

md("""
### B6 · Reajustar y evaluar en prueba (10 min)
La regla completa queda **cerrada antes** de la prueba: bosque de profundidad 8, umbral 0.60, costos 5:1. Se reajusta con los 3 680 correos y se evalúa en los 921 reservados **una vez**. Como en regresión, aquí se amplía el abanico de métricas (no solo la que eligió).
""")
code("""
fin = clone(modelos_spam()[ganador_s]).fit(xs_tr, ys_tr)
pt = fin.predict_proba(xs_te)[:, 1]
final_s = {**probs(ys_te, pt), **decision(ys_te, pt, umbral)}
ref_s = decision(ys_te, np.zeros(len(ys_te)), 0.5)
display(pd.DataFrame([final_s]).round(3))
print("Matriz de la regla elegida (filas reales, columnas predichas):")
print(np.array([[final_s["VN"], final_s["FP"]], [final_s["FN"], final_s["VP"]]]))
print(f"costo por 1000: regla elegida {1000 * final_s['costo']:.1f} | «siempre no spam» {1000 * ref_s['costo']:.1f}")
""")
md("""
Reproduce la diapositiva 57 (Python): matriz **553 / 5 / 53 / 310**, **93.7 %** de aciertos, precision 98.4 %, recall 85.4 % y **costo 84.7** por 1 000 (la regla «siempre no spam» cuesta 394.1). Es la evaluación del **procedimiento completo** (preprocesamiento, modelo, umbral y costos declarados) en este corpus histórico, no la validación de un filtro actual.

### ✍️ Ejercicio B6
Calcula el costo de prueba de la regla con umbral 0.50 en vez de 0.60 **solo para ver la diferencia** y explica por qué eso **no** autoriza a cambiar el umbral ahora.
""")
code("""
# tu código aquí
""")
solution("""
d50 = decision(ys_te, pt, 0.5)
print(f"t = 0.50: FP {d50['FP']}, FN {d50['FN']}, costo por 1000 = {1000 * d50['costo']:.1f} | t = 0.60: {1000 * final_s['costo']:.1f}")
""", "Si el 0.50 saliera mejor o peor en prueba, cambiar el umbral con esa información convertiría la prueba en parte de la selección y ya no sería evaluación independiente: habría que hacer una nueva selección y una nueva evaluación.")

# ================================================================== PARTE C
md("""
## Parte C · Profundizar: lo que la clase menciona y conviene entender

### C1 · Sesgo, varianza y ruido (15 min)
El error cuadrático esperado de un modelo en un punto `x` es `sesgo² + varianza + ruido`. Con una señal conocida y muchos conjuntos de datos simulados se pueden **medir** las tres partes. Aquí: señal `f(x) = 1.5·sen(2.2x) + 0.8x − 0.5`, ruido σ = 0.6, n = 30 y polinomios de grado 1 a 8.
""")
code("""
def senal(x):
    return 1.5 * np.sin(2.2 * x) + 0.8 * x - 0.5


rng = np.random.default_rng(RS)
n, sig, reps = 30, 0.6, 400
grados = np.arange(1, 9)
xg = np.linspace(-0.9, 0.9, 101)
pred = np.zeros((reps, len(grados), len(xg))); tr_err = np.zeros((reps, len(grados)))
for r in range(reps):
    xs = np.sort(rng.uniform(-1, 1, n)); ys = senal(xs) + rng.normal(0, sig, n)
    for j, gd in enumerate(grados):
        p = np.polynomial.Polynomial.fit(xs, ys, gd)
        pred[r, j] = p(xg); tr_err[r, j] = np.mean((ys - p(xs)) ** 2)
f0 = senal(xg)
sesgo2 = ((pred.mean(0) - f0) ** 2).mean(1); varia = pred.var(0, ddof=1).mean(1); total = sesgo2 + varia + sig ** 2
display(pd.DataFrame({"grado": grados, "sesgo²": sesgo2, "varianza": varia, "ruido": sig ** 2, "error esperado": total, "RMSE ajuste": np.sqrt(tr_err.mean(0)), "RMSE nuevo": np.sqrt(total)}).round(3))
plt.plot(grados, np.sqrt(tr_err.mean(0)), "o-", label="error de ajuste"); plt.plot(grados, np.sqrt(total), "o-", label="error en datos nuevos")
plt.ylim(0.4, 1.7); plt.xlabel("grado (complejidad)"); plt.ylabel("RMSE"); plt.legend(); plt.show()
print("grado con menor error esperado:", int(grados[np.argmin(total)]))
""")
md("""
El error de **ajuste** baja siempre; el de **datos nuevos** hace una U con mínimo en el **grado [[sesgo_var.mejor_grado]]**. A la derecha del mínimo la **varianza** explota (el ajuste sigue el ruido de cada muestra); el **ruido σ² = 0.36 no se elimina**. En un conjunto concreto (como la diapositiva 18) el grado 12 ajusta mejor ([[sesgo_var.uno.12.train|2]] frente a [[sesgo_var.uno.2.train|2]]) pero predice peor en datos nuevos ([[sesgo_var.uno.12.test|2]] frente a [[sesgo_var.uno.2.test|2]]). Las cifras del profesor (0.67/0.64 y 0.58/0.73) vienen de **su** simulación; la historia es la misma.

### ✍️ Ejercicio C1
Repite la simulación con n = 120 en vez de 30 (grados 1 a 8, 100 réplicas). ¿Cambia el grado con menor error esperado? ¿Qué pasa con la varianza del grado 8?
""")
code("""
# tu código aquí
""")
solution("""
rng2 = np.random.default_rng(1)
n2, reps2 = 120, 100
pred2 = np.zeros((reps2, len(grados), len(xg)))
for r in range(reps2):
    xs = np.sort(rng2.uniform(-1, 1, n2)); ys = senal(xs) + rng2.normal(0, sig, n2)
    for j, gd in enumerate(grados):
        pred2[r, j] = np.polynomial.Polynomial.fit(xs, ys, gd)(xg)
v2 = pred2.var(0, ddof=1).mean(1); s2 = ((pred2.mean(0) - f0) ** 2).mean(1)
print("error esperado por grado:", np.round(s2 + v2 + sig ** 2, 3))
print("varianza del grado 8 con n = 30:", round(float(varia[-1]), 3), "| con n = 120:", round(float(v2[-1]), 3))
""", "Con más datos la varianza del grado 8 cae muchísimo y el error esperado de los grados altos deja de explotar: **más datos permiten más complejidad**. El error de los grados 4 a 8 queda en una meseta (≈ 0.37–0.38) en vez de explotar; el mínimo sigue en el grado 3.")

md("""
### C2 · Curvas de aprendizaje y de validación (15 min)
La **curva de aprendizaje** muestra el error de ajuste y de validación según cuántas filas se usan para ajustar: si ambos quedan altos y juntos hay **sesgo alto**; si hay brecha grande hay **varianza alta**. Aquí con el hormigón: regresión lineal frente a la cuadrática con α = 0.1 (se sortean composiciones del conjunto de ajuste de cada pliegue).
""")
code("""
rng = np.random.default_rng(RS)
fr = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
mods = {"lineal": make_pipeline(StandardScaler(), LinearRegression()), "cuadrática α = 0.1": ridge_cuad(0.1)}
curvas = {k: {"train": [], "valid": []} for k in mods}
for fr_ in fr:
    acum = {k: ([], []) for k in mods}
    for k in range(1, 6):
        v = f_tr == k; pool = np.unique(g_tr[~v])
        for _ in range(5 if fr_ < 1 else 1):
            sel = rng.choice(pool, max(3, int(round(fr_ * len(pool)))), replace=False)
            filas = np.where(~v & np.isin(g_tr, sel))[0]
            for nm, m in mods.items():
                ft = clone(m).fit(x_tr[filas], y_tr[filas])
                acum[nm][0].append(np.mean((y_tr[filas] - ft.predict(x_tr[filas])) ** 2)); acum[nm][1].append(np.mean((y_tr[v] - ft.predict(x_tr[v])) ** 2))
    for nm in mods:
        curvas[nm]["train"].append(float(np.sqrt(np.mean(acum[nm][0])))); curvas[nm]["valid"].append(float(np.sqrt(np.mean(acum[nm][1]))))
ntr = [int(round(f_ * 0.8 * len(htr))) for f_ in fr]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, nm in zip(axs, mods):
    ax.plot(ntr, curvas[nm]["train"], "o-", label="ajuste"); ax.plot(ntr, curvas[nm]["valid"], "o-", label="validación"); ax.set(title=nm, xlabel="filas de ajuste", ylim=(3, 20)); ax.legend()
axs[0].set_ylabel("RMSE (MPa)"); plt.tight_layout(); plt.show()
print("con todas las filas de ajuste -> lineal:", round(curvas["lineal"]["train"][-1], 2), round(curvas["lineal"]["valid"][-1], 2), "| cuadrática:", round(curvas["cuadrática α = 0.1"]["train"][-1], 2), round(curvas["cuadrática α = 0.1"]["valid"][-1], 2))
""")
md("""
La **lineal** se estanca cerca de 10 MPa tanto en ajuste como en validación: **sesgo alto** (le falta flexibilidad; más datos no ayudan, más términos sí). La **cuadrática** tiene brecha grande con pocos datos (varianza alta) y **sigue mejorando** al añadir filas: con 645 filas valida en [[curvas.cuad.valid.6|2]] MPa. Y la curva de validación según α (A4) se aplana por debajo de 0.1.

### ✍️ Ejercicio C2
Con las curvas de arriba, ¿conseguir 1 000 mediciones más ayudaría a la lineal? ¿Y a la cuadrática? Justifica con la forma de las curvas.
""")
code("""
# tu código aquí (puedes calcular la pendiente de los últimos tramos de la curva de validación)
""")
solution("""
for nm in mods:
    v = curvas[nm]["valid"]
    print(nm, "| mejora de validación en los últimos dos tramos:", round(v[-3] - v[-2], 3), "y", round(v[-2] - v[-1], 3), "MPa")
""", "A la **lineal** no le sirve: sus curvas ya son planas (sesgo alto). A la **cuadrática** le sirve todavía algo: la validación sigue bajando, aunque cada vez menos.")

md("""
### C3 · Particiones repetidas, K y dejar un grupo fuera (15 min)
La partición de 5 pliegues del profesor es **una** de muchas posibles: ¿cuánto cambia el RMSE OOF si se re-sortean los pliegues? Y ¿qué pasa con K? Con K pequeño se entrena con menos datos (estimación pesimista); con K grande, más ajustes. **Dejar un grupo fuera** (*leave-one-group-out*, la versión por composición de LOOCV) usa las 342 composiciones como pliegues.
""")
code("""
rng = np.random.default_rng(RS)
R_ = 40
res_rep = {a: [] for a in (0.1, 1, 10)}
for _ in range(R_):
    f_ = folds_por_grupo(g_tr, 5, rng)
    for a in res_rep:
        res_rep[a].append(rmse(y_tr, oof_reg(ridge_cuad(a), x_tr, y_tr, f_)))
res_rep = {a: np.array(v) for a, v in res_rep.items()}
dif = res_rep[0.1] - res_rep[1]
tab = pd.DataFrame({f"α = {a}": [v.mean(), v.std(ddof=1), v.min(), v.max()] for a, v in res_rep.items()}, index=["media", "DE", "mín", "máx"])
display(tab.round(3))
print(f"α = 0.1 gana a α = 1 en {np.mean(dif < 0):.1%} de las particiones | diferencia media {dif.mean():.3f} MPa (DE {dif.std(ddof=1):.3f})")
""")
code("""
barrido = {}
for K in (2, 3, 5, 10, 20):
    v = [rmse(y_tr, oof_reg(ridge_cuad(0.1), x_tr, y_tr, folds_por_grupo(g_tr, K, rng))) for _ in range(12)]
    barrido[K] = (float(np.mean(v)), float(np.std(v, ddof=1)))
ug = np.unique(g_tr)
logo = rmse(y_tr, oof_reg(ridge_cuad(0.1), x_tr, y_tr, np.searchsorted(ug, g_tr)))     # 342 ajustes
print({k: tuple(round(x, 3) for x in v) for k, v in barrido.items()}, "| un grupo fuera:", round(logo, 3))
""")
md("""
Con 40 particiones al azar, el RMSE OOF de α = 0.1 va de 7.73 a 8.43 —una **dispersión de ≈ 0.16 MPa** que no se ve con una sola partición—. Como los candidatos comparten pliegues, la **diferencia** es más estable que cada cifra: α = 0.1 gana en ≈ 98 % de las particiones, pero por ≈ **0.07 MPa** (menos de 1 %). El ranking es estable; la ventaja, trivial. Con **K = 2** el RMSE sube a ≈ 8.74 (se entrena con la mitad: pesimista); con K = 5, 10 y 20 baja a ≈ 8.04, 7.89 y 7.88, y con **un grupo fuera** a 7.83. K grande reduce el sesgo de la estimación a cambio de más ajustes.

### ✍️ Ejercicio C3
Cuenta cuántos ajustes hacen K = 5 y *un grupo fuera* para los 7 candidatos, y explica por qué LOOCV por **filas** sería una mala idea con este archivo.
""")
code("""
# tu código aquí
""")
solution("""
print("K = 5, 7 candidatos:", 5 * 7, "ajustes | un grupo fuera:", len(ug) * 7)
print("filas por composición (media):", round(len(htr) / htr.grupo.nunique(), 2), "-> con LOOCV por filas, la misma mezcla (con otra edad) quedaría en el ajuste")
""", "35 frente a 2 394 ajustes. LOOCV por filas dejaría la misma composición (otra edad) en el ajuste y la validación: fuga y estimación optimista. Por eso la unidad que se deja fuera es el **grupo**.")

md("""
### C4 · Bootstrap y estimador .632 (15 min)
El **bootstrap** remuestrea **composiciones** con reemplazo: cada remuestra incluye ≈ 63.2 % de ellas (`1 − 1/e`) y deja ≈ 36.8 % **fuera** (OOB, *out-of-bag*), que sirve de validación. El error OOB es **pesimista** (entrena con ≈ 63 % de las composiciones) y el de ajuste es **optimista**; el estimador **.632** los mezcla: `0.368·(ajuste) + 0.632·(OOB)` (en MSE). Además, remuestrear la **prueba** da un intervalo para su RMSE.
""")
code("""
por_g = {gg: np.where(g_tr == gg)[0] for gg in ug}
rng = np.random.default_rng(RS)
B = 200
mse_oob, dentro = [], []
for _ in range(B):
    samp = rng.choice(ug, len(ug), replace=True)
    filas = np.concatenate([por_g[s] for s in samp])
    fuera = np.setdiff1d(ug, np.unique(samp)); ofilas = np.concatenate([por_g[s] for s in fuera])
    m = ridge_cuad(0.1).fit(x_tr[filas], y_tr[filas])
    mse_oob.append(float(np.mean((y_tr[ofilas] - m.predict(x_tr[ofilas])) ** 2))); dentro.append(1 - len(fuera) / len(ug))
full = ridge_cuad(0.1).fit(x_tr, y_tr)
mse_res = float(np.mean((y_tr - full.predict(x_tr)) ** 2)); mse_b = float(np.mean(mse_oob)); mse632 = 0.368 * mse_res + 0.632 * mse_b
print(f"fracción dentro: {np.mean(dentro):.3f} (teoría 1 - 1/e = {1 - np.exp(-1):.3f})")
print(f"RMSE resustitución {np.sqrt(mse_res):.3f} | bootstrap OOB {np.sqrt(mse_b):.3f} | .632 {np.sqrt(mse632):.3f} | CV 5 pliegues {rmse(y_tr, oof_reg_todos['Q_A0.1']):.3f}")

# intervalo bootstrap para el RMSE de prueba (se remuestrean las 86 composiciones de prueba)
gt = hte.grupo.to_numpy(); ugt = np.unique(gt); por_gt = {gg: np.where(gt == gg)[0] for gg in ugt}
ra, rb = [], []
for _ in range(2000):
    f_ = np.concatenate([por_gt[s] for s in rng.choice(ugt, len(ugt), replace=True)])
    ra.append(rmse(y_te[f_], p_te[f_])); rb.append(rmse(y_te[f_], p_base[f_]))
print(f"RMSE de prueba {rmse(y_te, p_te):.2f} | IC bootstrap 95 % [{np.percentile(ra, 2.5):.2f}; {np.percentile(ra, 97.5):.2f}] | media de entrenamiento {rmse(y_te, p_base):.2f}")
""")
md("""
Las cuatro estimaciones del error de la **misma receta** ordenan así: resustitución [[bootstrap.rmse_resub|2]] (optimista) < CV 5 pliegues [[bootstrap.rmse_cv|2]] < .632 [[bootstrap.rmse_632|2]] < bootstrap OOB [[bootstrap.rmse_oob|2]] (pesimista); la prueba da [[bootstrap.rmse_test|2]]. Ninguna es «la» verdad: todas son estimaciones. El **IC bootstrap del RMSE de prueba** es ancho —[[bootstrap.ci_test.0|2]] a [[bootstrap.ci_test.1|2]] MPa— porque solo hay 86 composiciones: la cifra 8.82 tiene incertidumbre propia (la Clase 43 profundiza en el bootstrap).

### ✍️ Ejercicio C4
Calcula la fracción teórica de composiciones que **no** entran a una remuestra con n = 342 y compárala con el 36.8 % del límite. ¿Cuántas composiciones quedan fuera en promedio?
""")
code("""
# tu código aquí
""")
solution("""
n_g = len(ug)
fuera_teo = (1 - 1 / n_g) ** n_g
print(f"fuera (exacta) = {fuera_teo:.4f} | límite e^-1 = {np.exp(-1):.4f} | composiciones fuera ≈ {n_g * fuera_teo:.1f} de {n_g}")
""", "(1 − 1/342)³⁴² ≈ 0.3673 frente al límite 0.3679: ≈ 125.6 composiciones fuera de 342 en cada remuestra.")

md("""
### C5 · AIC y BIC frente a validación cruzada (15 min)
Con errores normales, `AIC = n·ln(SSE/n) + 2k + const` y `BIC = n·ln(SSE/n) + k·ln(n) + const` (k = número de coeficientes). Se calculan con **todas las filas de ajuste** y penalizan la complejidad con una fórmula, no con datos nuevos. Gana el **menor**. Aquí se comparan modelos de mínimos cuadrados que añaden términos (8 lineales, 8 cuadrados y 28 interacciones) con `statsmodels`, y se contrasta con el RMSE OOF.
""")
code("""
import statsmodels.api as sm

Xs = StandardScaler().fit_transform(htr[V8])
pf = PolynomialFeatures(2, include_bias=False).fit(Xs); Z = pf.transform(Xs); nom = pf.get_feature_names_out([f"x{i}" for i in range(8)])
corr = np.array([abs(np.corrcoef(Z[:, j], y_tr)[0, 1]) for j in range(Z.shape[1])])
lin = [j for j in np.argsort(-corr) if "^" not in nom[j] and " " not in nom[j]]
cua = [j for j in np.argsort(-corr) if "^2" in nom[j]]
inter = [j for j in np.argsort(-corr) if " " in nom[j]]
orden_t = lin + cua + inter                 # primero los 8 lineales, luego cuadrados e interacciones
n_ = len(y_tr)
aic, bic, cvr = [], [], []
for p in range(0, 45):
    X = sm.add_constant(Z[:, orden_t[:p]], has_constant="add") if p else np.ones((n_, 1))
    r = sm.OLS(y_tr, X).fit(); aic.append(r.aic); bic.append(r.bic)
    oof = np.zeros(n_)
    for k in range(1, 6):
        v = f_tr == k; oof[v] = X[v] @ np.linalg.lstsq(X[~v], y_tr[~v], rcond=None)[0]
    cvr.append(rmse(y_tr, oof))
aic, bic, cvr = map(np.array, (aic, bic, cvr))
print(f"AIC elige {aic.argmin()} términos | BIC elige {bic.argmin()} | CV elige {cvr.argmin()} (RMSE OOF {cvr.min():.2f})")
# comprobación a mano para el modelo completo (k = 45 coeficientes)
sse = float(sm.OLS(y_tr, sm.add_constant(Z[:, orden_t], has_constant="add")).fit().ssr)
print(f"AIC a mano {n_ * np.log(sse / n_) + n_ * (1 + np.log(2 * np.pi)) + 2 * 45:.2f} | statsmodels {aic[44]:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(aic - aic.min(), label="AIC"); axs[0].plot(bic - bic.min(), label="BIC"); axs[0].set(ylim=(-30, 600), xlabel="términos", ylabel="Δ criterio"); axs[0].legend()
axs[1].plot(cvr); axs[1].set(ylim=(7.5, 12.5), xlabel="términos", ylabel="RMSE OOF (MPa)"); plt.tight_layout(); plt.show()
""")
md("""
El **AIC elige [[aic_bic.p_aic]]** términos y la **CV [[aic_bic.p_cv]]**, mientras el **BIC elige [[aic_bic.p_bic]]**: con n = 806 el BIC cobra `ln 806 = 6.69` por parámetro frente a 2 del AIC. La fórmula a mano coincide con `statsmodels` (5 527.92). Dos cautelas: (1) AIC y BIC **no miran datos nuevos** ni son OOF; (2) suponen filas **independientes** y aquí hay composiciones repetidas, así que el `n` efectivo es menor y los criterios parecen más seguros de lo que son. En clase se mencionan como métricas posibles, pero el script **no** los calcula: la elección se hace con validación cruzada.

### ✍️ Ejercicio C5
Con n = 100: M1 (k = 3, SSE = 250), M2 (k = 5, SSE = 200) y M3 (k = 9, SSE = 170). Calcula AIC y BIC (sin la constante) y decide con cada uno.
""")
code("""
# tu código aquí
""")
solution("""
for nm, k, sse_ in [("M1", 3, 250), ("M2", 5, 200), ("M3", 9, 170)]:
    print(nm, f"AIC = {100 * np.log(sse_ / 100) + 2 * k:.2f} | BIC = {100 * np.log(sse_ / 100) + k * np.log(100):.2f}")
""", "AIC: 97.63, 79.31 y 71.06 → elige **M3**. BIC: 105.44, 92.34 y 94.51 → elige **M2**. El BIC (4.605 por parámetro) es más prudente con muchos parámetros que el AIC (2).")

md("""
### C6 · Validación anidada (20 min)
Cuando se **elige** con validación cruzada (por ejemplo el mejor α o el mejor grado), el mejor puntaje obtenido es **optimista**: ya se usó para elegir. La **validación anidada** separa los papeles: un bucle **interno** elige hiperparámetros con las filas de ajuste y un bucle **externo** mide el desempeño con pliegues que no participaron en la elección. Primero una demostración sin señal alguna (etiquetas al azar) y luego el hormigón con `GridSearchCV`.
""")
code("""
rng = np.random.default_rng(RS)
reps_, n_r, M = 300, 60, 100


def nc_acc(Xt, yt, Xv, yv):                       # centroide más cercano, una variable a la vez (vectorizado)
    m0, m1 = Xt[yt == 0].mean(0), Xt[yt == 1].mean(0)
    thr, sg = (m0 + m1) / 2, np.where(m1 > m0, 1, -1)
    return (((Xv - thr) * sg > 0).astype(int) == yv[:, None]).mean(0)


def cv_feat(X, yv, K, rg):
    idx = rg.permutation(len(yv)) % K
    acc = np.zeros(X.shape[1])
    for k in range(K):
        v = idx == k
        acc += nc_acc(X[~v], yv[~v], X[v], yv[v]) * v.sum()
    return acc / len(yv)


mejor, anid, nuevo = [], [], []
for _ in range(reps_):
    X = rng.normal(size=(n_r, M)); yv = (np.arange(n_r) % 2).astype(int); rng.shuffle(yv)
    a = cv_feat(X, yv, 5, rng); j = int(np.argmax(a)); mejor.append(float(a[j]))
    Xn = rng.normal(size=(2000, M)); yn = rng.integers(0, 2, 2000); nuevo.append(float(nc_acc(X, yv, Xn, yn)[j]))
    idx = rng.permutation(n_r) % 5; tot = 0.0
    for k in range(5):
        v = idx == k
        ai = cv_feat(X[~v], yv[~v], 4, rng)
        tot += nc_acc(X[~v], yv[~v], X[v], yv[v])[int(np.argmax(ai))] * v.sum()
    anid.append(tot / n_r)
print(f"etiquetas al azar, {M} candidatos, n = {n_r}:")
print(f"  mejor puntaje de CV = {np.mean(mejor):.3f} | validación anidada = {np.mean(anid):.3f} | en datos nuevos = {np.mean(nuevo):.3f} (la verdad: 0.5)")
""")
md("""
Sin ninguna señal, el **mejor puntaje de CV** entre 100 candidatos es **[[anidada.cv_mejor|3]]** (se eligió el máximo del azar), mientras la **anidada** da **[[anidada.anidada|3]]** y los datos nuevos **[[anidada.nuevo|3]]**: la verdad es 0.5. Ahora el hormigón, con una rejilla de grado {1, 2, 3} × α {0.001 … 1000} (21 configuraciones). Cada pliegue externo (los de la clase) ejecuta un `GridSearchCV` interno con `GroupKFold(5)` sobre composiciones:
""")
code("""
pipe = Pipeline([("esc1", StandardScaler()), ("poli", PolynomialFeatures(include_bias=False)), ("esc2", StandardScaler()), ("ridge", Ridge(solver="svd"))])
rejilla = {"poli__degree": [1, 2, 3], "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
interno, externo, elegidos, tam = [], [], [], []
for k in range(1, 6):
    v = f_tr == k
    gs = GridSearchCV(pipe, rejilla, cv=GroupKFold(5), scoring="neg_root_mean_squared_error")
    gs.fit(x_tr[~v], y_tr[~v], groups=g_tr[~v])                     # bucle interno: elige
    interno.append(-gs.best_score_)
    externo.append(rmse(y_tr[v], gs.predict(x_tr[v])))              # bucle externo: mide
    elegidos.append((gs.best_params_["poli__degree"], gs.best_params_["ridge__alpha"])); tam.append(int(v.sum()))
print("mejor RMSE interno por pliegue externo:", np.round(interno, 2))
print("RMSE externo:                          ", np.round(externo, 2))
print("configuraciones elegidas (grado, α):", elegidos)
print(f"promedio interno {np.mean(interno):.2f} | externo (combinado) {np.sqrt(np.average(np.square(externo), weights=tam)):.2f}")
print("ajustes de la anidada:", 5 * 5 * 21 + 5, "| de una búsqueda simple:", 5 * 21 + 1)
""")
md("""
El mejor RMSE **interno** (≈ 8.1) es **optimista** frente al **externo** (≈ 8.5): informar el primero como desempeño esperado sería engañoso (con los pliegues internos al azar de la figura de la guía, la brecha fue mayor: ≈ 8.0 frente a 9.1; el tamaño del optimismo depende de la partición). Además, la **configuración elegida cambia** de un pliegue externo a otro (distintos grados y α): la selección es inestable cuando varias configuraciones rinden casi igual. La anidada estima el desempeño de **todo el procedimiento de selección**; el modelo que se despliega se vuelve a elegir con **todo** el entrenamiento. Cuesta 530 ajustes frente a 106.

### ✍️ Ejercicio C6
Si el bucle externo tuviera 10 pliegues, el interno 5 y la rejilla 21 configuraciones, ¿cuántos ajustes serían? ¿Y por qué la validación anidada **no** es necesaria cuando se compara un único modelo sin elegir hiperparámetros?
""")
code("""
# tu código aquí
""")
solution("""
print("ajustes:", 10 * 5 * 21 + 10)
""", "1 060 ajustes. Con un solo modelo fijado de antemano no hay elección que contamine la estimación: la validación cruzada simple (o la prueba) ya es una estimación honesta; la anidada corrige el optimismo **de elegir**.")

md("""
### C7 · Recalibrar y umbral teórico (15 min)
El bosque **subestima** las probabilidades de spam (B4) y por eso el umbral empírico (0.60) quedó lejos del teórico (0.833). ¿Qué pasa si se **recalibra**? La regresión isotónica ajusta una función monótona de la probabilidad predicha. Pero **recalibrar cambia la receta**: hay que validarla **anidada** (la calibración se aprende solo con las filas de ajuste de cada pliegue).
""")
code("""
rng = np.random.default_rng(RS)
pc = np.zeros(len(ys_tr))
for k in range(1, 6):
    v = fs_tr == k
    ids = folds_por_grupo(gs_tr[~v], 3, rng)
    splits = [(np.where(ids != j)[0], np.where(ids == j)[0]) for j in range(3)]
    cal_m = CalibratedClassifierCV(clone(modelos_spam()["BOSQUE_8"]), method="isotonic", cv=splits).fit(xs_tr[~v], ys_tr[~v])
    pc[v] = cal_m.predict_proba(xs_tr[v])[:, 1]
print("bosque:", {k: round(x, 3) for k, x in probs(ys_tr, p_oof).items()})
print("bosque recalibrado:", {k: round(x, 3) for k, x in probs(ys_tr, pc).items()})
c_raw = np.array([1000 * decision(ys_tr, p_oof, t)["costo"] for t in umbrales]); c_cal = np.array([1000 * decision(ys_tr, pc, t)["costo"] for t in umbrales])
print(f"umbral óptimo: bosque {umbrales[c_raw.argmin()]:.2f} (costo {c_raw.min():.1f}) | recalibrado {umbrales[c_cal.argmin()]:.2f} (costo {c_cal.min():.1f})")
plt.plot(umbrales, c_raw, "o-", label="bosque"); plt.plot(umbrales, c_cal, "o-", label="recalibrado"); plt.axvline(5 / 6, ls=":", c="tab:purple", label="teórico 5/6")
plt.ylim(0, 600); plt.xlabel("umbral"); plt.ylabel("costo OOF por 1000"); plt.legend(); plt.show()
""")
md("""
Recalibrado, el óptimo empírico sube a **0.80** (cerca del teórico 0.833) y el **Brier mejora** (0.061 → [[calibracion.metricas.BOSQUE_CAL.brier|3]]), pero el **costo OOF no baja** (132.9 frente a 128.8) y el log loss queda igual. Moraleja: la teoría se cumple cuando las probabilidades están calibradas; en la práctica el umbral se elige empíricamente y **una recalibración también se valida**.

### ✍️ Ejercicio C7
Con probabilidades calibradas, ¿qué umbral teórico corresponde a costos FP = 10 y FN = 1? ¿Y si FP = 1 y FN = 5? Interprétalos.
""")
code("""
# tu código aquí
""")
solution("""
for cfp, cfn in [(10, 1), (1, 5)]:
    print(f"FP = {cfp}, FN = {cfn}: bloquear si p > {cfp / (cfp + cfn):.3f}")
""", "10/11 ≈ 0.909 (casi hay que estar seguro: el FP es carísimo) y 1/6 ≈ 0.167 (basta una sospecha leve: el FN es el caro).")

md("""
### C8 · Casi-duplicados entre entrenamiento y prueba (10 min)
El profesor agrupa por **igualdad exacta** de los siete ingredientes. Pero hay mezclas que difieren en un solo ingrediente (por ejemplo, 1040 frente a 1055 kg/m³ de agregado grueso) y quedan en grupos distintos: la validación y la prueba pueden contener composiciones **casi idénticas** a las del ajuste. Se mide la distancia (en desviaciones estándar) de cada fila de prueba a la composición más parecida del entrenamiento.
""")
code("""
I7 = V8[:7]
sc7 = StandardScaler().fit(htr[I7])
nn = NearestNeighbors(n_neighbors=1).fit(sc7.transform(htr[I7]))
dist = nn.kneighbors(sc7.transform(hte[I7]))[0][:, 0]
cerca = dist < 0.1
print(f"filas de prueba con una composición casi idéntica (< 0.1 DE) en el entrenamiento: {cerca.sum()} de {len(dist)} ({cerca.mean():.0%})")
print(f"RMSE en prueba: cerca {rmse(y_te[cerca], p_te[cerca]):.2f} | lejos {rmse(y_te[~cerca], p_te[~cerca]):.2f} | total {rmse(y_te, p_te):.2f}")
cuart = pd.qcut(dist, 4, labels=["Q1", "Q2", "Q3", "Q4"])
display(pd.DataFrame({"dist_media": pd.Series(dist).groupby(cuart).mean(), "RMSE": pd.Series(res ** 2).groupby(cuart).mean() ** 0.5}).round(2))
""")
md("""
**49 de 224 filas** de prueba (22 %) tienen una composición del entrenamiento a menos de 0.1 desviaciones estándar y un **RMSE de ≈ 4.4**; las demás, **≈ 9.7**. Y el RMSE crece con la distancia (≈ 4.3 en el cuartil más cercano, ≈ 13.0 en el más lejano). La cifra **8.82 es válida para esta partición**, pero para mezclas **realmente nuevas** el error esperado es mayor. Es una limitación que el profesor anticipa («no acredita generalización a otra planta o laboratorio») y que conviene tener presente al comunicar el resultado.

### ✍️ Ejercicio C8
Calcula qué porcentaje de las filas de **entrenamiento** tiene, en *otro grupo* del entrenamiento, una composición a menos de 0.1 DE. ¿Qué implica para la validación cruzada?
""")
code("""
# tu código aquí
""")
solution("""
A = sc7.transform(htr[I7]); nn2 = NearestNeighbors(n_neighbors=40).fit(A)
dd, ii = nn2.kneighbors(A)
cerca_otro = []
for r in range(len(A)):
    d_otro = next((dj for dj, j in zip(dd[r], ii[r]) if g_tr[j] != g_tr[r]), np.inf)
    cerca_otro.append(d_otro < 0.1)
print(f"filas con una composición casi idéntica en OTRO grupo: {np.mean(cerca_otro):.1%}")
""", "Casi una de cada cuatro filas (≈ 24 %) de entrenamiento tiene una vecina casi idéntica en otro grupo: **dentro de la CV también hay casi-duplicados** entre pliegues, así que el RMSE OOF es algo optimista para composiciones realmente nuevas.")

# ================================================================== ALERTAS Y D2
md("""
## Parte D · Alertas y traducción

### D1 · Alertas del material: qué conviene precisar (verificado con código) (15 min)

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Clase | «R² es un número entre 0 y 1» | Puede ser **negativo**: correlación 1 con R² = −149 (A2); la media de entrenamiento da R² ≈ −0.007 en prueba (A5). |
| Clase | «el ajuste … lo voy a realizar con el conjunto de prueba» | Debe ser con el **entrenamiento** (con prueba habría fuga) (A3). |
| Clase | «pérdida cuadrática fuera de pliegue» en clasificación | El script elige por **log loss** OOF; el Brier solo describe (B2). |
| Clase | «umbral 0.5 → costo 132; 0.6 → 128» | 132.3 es de **0.55**; en 0.50 el costo es 149.7 (B5). |
| Clase | «en el primero y el tercero gana α = 1» | Gana α = 1 en los pliegues **1 y 5** (A4). |
| Clase | «342 composiciones … 806 mediciones de las 1 030» | Son las de **entrenamiento**; el total es 428 composiciones (A1). |
| Clase | «10 de los 1 000 datos … 9 990 restantes» | Mezcla 1 000 con 10 000 (ejemplo de clases desbalanceadas). |
| Diap. 26 y 29 | «elegir no prueba superioridad estable» | Se confirma y se matiza: α = 0.1 gana en ≈ 98 % de 40 particiones, por ≈ 0.07 MPa (C3). |
| Diap. 13 y 7 | grupos por igualdad exacta | No impiden **casi-duplicados**: 22 % de la prueba tiene una vecina casi idéntica en el entrenamiento, con RMSE ≈ 4.4 frente a ≈ 9.7 (C8). |
| Diap. 55 y 53 | umbral teórico 5/6 frente a empírico 0.60 | Lo explica la **calibración**; recalibrado el óptimo sube a 0.80, pero el costo OOF no mejora (C7). |
| Clase | AIC y BIC como métricas posibles | No se calculan en el script ni son OOF; aquí eligen 43 y 22 términos frente a 43 de la CV (C5). |
| Diap. 50–57 | R y Python dan cifras algo distintas en el bosque | Es **implementación** (`ranger` frente a scikit-learn): log loss 0.228 y 0.223; costo a 0.60: 128.8 y 123.4. |
| Diap. 24 | `38_evaluacion_seleccion.R` | Se entrega como `.R.txt`; y el script de Python escribe resultados en una carpeta `resultados` junto al material. |

**Transcripción con ruido**: «Breyer / Bright / Ryre» = Brier, «lock-lock» = log loss, «ROP / ROCK» = ROC, «reach / rich / Bridge» = ridge, «lazo» = lasso, «OUF» = OOF, «ORMCE / RMC» = RMSE, «revisión» = regresión, «avaliación / ligación» = validación, «los pies» = los pliegues. El recreo (≈ 1:10:40 del video) quedó marcado en la transcripción.
""")
code("""
# comprobaciones rápidas de la tabla
print("R² con y = (1,2,3), ŷ = y + 10:", round(r2(np.array([1.0, 2, 3]), np.array([11.0, 12, 13])), 1))
print("costo por 1000 con t = 0.50, 0.55, 0.60:", [round(float(1000 * decision(ys_tr, p_oof, t)["costo"]), 1) for t in (0.5, 0.55, 0.6)])
print("pliegues donde α = 0.1 gana a α = 1:", [k for k in range(1, 6) if rmse(y_tr[f_tr == k], oof_reg_todos['Q_A0.1'][f_tr == k]) < rmse(y_tr[f_tr == k], oof_reg_todos['Q_A1'][f_tr == k])])
print("composiciones: entrenamiento", htr.grupo.nunique(), "| prueba", hte.grupo.nunique(), "| total", h.grupo.nunique())
""")
md("""
### D2 · Tabla de correspondencia R ↔ Python (10 min)
Los scripts del profesor (`38_evaluacion_seleccion.R.txt` y `.py`) siguen el mismo protocolo con las mismas particiones. Esta tabla ayuda a leer uno desde el otro:

| Qué se hace | R | Python (scikit-learn) |
|---|---|---|
| Leer los datos | `read.csv("hormigon_s38.csv")` | `pd.read_csv("hormigon_s38.csv")` |
| Filas de entrenamiento | `subset(datos, conjunto == "entrenamiento")` | `datos[datos["conjunto"] == "entrenamiento"]` |
| Modelo constante | `mean(y_entrenamiento)` | `DummyRegressor(strategy="mean")` |
| Regresión lineal | `lm(y ~ ., data = ...)` | `make_pipeline(StandardScaler(), LinearRegression())` |
| Términos cuadráticos e interacciones | `poly(x, 2)` o fórmula `(.)^2 + I(x^2)` | `PolynomialFeatures(2, include_bias=False)` |
| Ridge con penalización α | `glmnet(..., alpha = 0, lambda = a)` (o el código propio del script) | `Ridge(alpha=a, solver="svd")` |
| Receta completa (sin fuga) | funciones que reajustan en cada pliegue | `Pipeline(...)` y `clone(modelo).fit(X[~valida], y[~valida])` |
| Pliegues por composición | columna `fold` y `grupo` | columna `fold` o `GroupKFold` |
| RMSE OOF | `sqrt(mean((y - oof)^2))` | `np.sqrt(np.mean((y - oof) ** 2))` |
| Regresión logística | `glm(spam ~ ., family = binomial())` | `LogisticRegression(C=1)` |
| Bosque aleatorio | `ranger::ranger(...)` | `RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5)` |
| Probabilidad de spam | `predict(modelo, x)$predictions[, "1"]` o `type = "response"` | `modelo.predict_proba(X)[:, 1]` |
| Log loss y Brier | `-mean(y*log(p) + (1-y)*log(1-p))` · `mean((y - p)^2)` | `log_loss(y, p)` · `brier_score_loss(y, p)` |
| ROC y AUC | `pROC::roc(y, p)` y `auc()` | `roc_curve(y, p)` y `roc_auc_score(y, p)` |
| Matriz de confusión | `table(Real = y, Predicha = p >= t)` | `confusion_matrix(y, p >= t, labels=[0, 1])` |
| Costo | `(5 * fp + fn) / n` | `(5 * fp + fn) / n` |
| Búsqueda de hiperparámetros | `caret::train(..., trControl = ...)` | `GridSearchCV(pipe, rejilla, cv=GroupKFold(5))` |
| AIC / BIC | `AIC(modelo)` · `BIC(modelo)` | `modelo.aic` · `modelo.bic` (statsmodels) |
| Semilla | `set.seed(123)` | `random_state=...` / `np.random.default_rng(123)` |

*Diferencias a recordar*: en R los bosques usan `ranger`; en Python, scikit-learn: con la misma receta producen probabilidades algo distintas (no son errores). `AIC()` de R y `statsmodels` suman constantes distintas en algunos modelos: solo se comparan entre sí AIC del **mismo** programa y los **mismos** datos.

### ✍️ Ejercicio D2
Traduce a Python: `oof <- numeric(n); for (k in 1:5) { valida <- fold == k; oof[valida] <- predict(ajustar(x[!valida, ], y[!valida]), x[valida, ]) }` y explica por qué el ajuste usa `x[!valida, ]`.
""")
code("""
# tu código aquí
""")
solution("""
oof_d2 = np.zeros(len(y_tr))
for k in range(1, 6):
    valida = f_tr == k
    oof_d2[valida] = clone(modelos["Q_A0.1"]).fit(x_tr[~valida], y_tr[~valida]).predict(x_tr[valida])
print("RMSE OOF:", round(rmse(y_tr, oof_d2), 3))
""", "`~valida` (negación del indicador) selecciona las filas de **ajuste**: ningún pliegue se usa para predecirse a sí mismo. El resultado es el mismo RMSE OOF de A3.")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes recomendar un procedimiento de evaluación para una **nueva** tarea de clasificación con la base de spam, pero ahora con costos distintos: **FP = 3 y FN = 1**. Usa la partición del profesor (`xs_tr`, `ys_tr`, `fs_tr`, `gs_tr`, `xs_te`, `ys_te`) y responde con código:

1. **Plan previo**: escribe en un comentario la métrica de selección de probabilidades, la regla de elección del umbral y la referencia de decisión, **antes** de calcular nada.
2. **Candidatos**: frecuencia, logística (C = 1) y bosque (profundidad 8) con probabilidades OOF. ¿Cuál gana por log loss?
3. **Umbral**: costo OOF para t = 0.05, …, 0.95 con FP = 3. ¿Qué umbral elige? Compara con el teórico 3/(3 + 1).
4. **Reajuste y prueba**: ajusta con todo el entrenamiento y evalúa **una vez** en la prueba: matriz, aciertos, costo por 1 000 y costo de «siempre no spam».
5. **Incertidumbre**: intervalo bootstrap del costo de prueba (remuestrea correos de prueba, 1 000 veces).
6. **Sin código**: una frase para la dirección. ¿Qué dos advertencias agregarías (costos hipotéticos, corpus histórico, calibración, nueva evaluación si se cambia la regla)?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1) plan previo: log loss OOF para probabilidades; umbral por menor costo OOF (FP = 3, FN = 1; empates: mayor umbral); referencia: «siempre no spam»
# 2) candidatos
for nombre in ("FRECUENCIA", "LOGISTICA_C1", "BOSQUE_8"):
    print(nombre, round(log_loss(ys_tr, oof_p[nombre]), 3))
# 3) umbral por costo OOF con FP = 3
c3 = pd.DataFrame([decision(ys_tr, p_oof, t, 3.0, 1.0) for t in umbrales])
e3 = c3.sort_values(["costo", "umbral"], ascending=[True, False]).iloc[0]
t3 = float(e3.umbral)
print(f"umbral elegido {t3:.2f} | costo OOF por 1000 {1000 * e3.costo:.1f} | teórico 3/4 = 0.75")
# 4) prueba, una vez
d3 = decision(ys_te, pt, t3, 3.0, 1.0); r3 = decision(ys_te, np.zeros(len(ys_te)), 0.5, 3.0, 1.0)
print(f"matriz: VN {d3['VN']}, FP {d3['FP']}, FN {d3['FN']}, VP {d3['VP']} | aciertos {d3['accuracy']:.1%} | costo {1000 * d3['costo']:.1f} | siempre no spam {1000 * r3['costo']:.1f}")
# 5) bootstrap de los correos de prueba
rng = np.random.default_rng(RS); cb = []
for _ in range(1000):
    i = rng.integers(0, len(ys_te), len(ys_te)); cb.append(1000 * decision(ys_te[i], pt[i], t3, 3.0, 1.0)["costo"])
print(f"IC bootstrap 95 % del costo de prueba: [{np.percentile(cb, 2.5):.1f}; {np.percentile(cb, 97.5):.1f}]")
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: (2) el bosque gana por log loss OOF (0.228 frente a 0.275 y 0.671). (3) Con FP = 3 el umbral óptimo baja respecto de 0.60 (FP = 5): el FP es menos caro, así que se bloquea con menos evidencia; el teórico 0.75 queda por encima del empírico por la subestimación del bosque. (4) En prueba, con t = 0.55: matriz 550 / 8 / 47 / 316, aciertos 94.0 % y costo 77.1 por 1 000 frente a 394.1 de «siempre no spam». (5) El intervalo bootstrap del costo es [55.4; 102.1]: unos 45 puntos de ancho. (6) **Recomendación**: «el bosque con el umbral elegido reduce el costo muy por debajo de no filtrar nada, con la razón de costos 3:1 declarada». Advertencias: los costos son **hipotéticos** (si cambian, cambia el umbral) y el corpus es **histórico** (no valida un filtro actual); además, si se cambia el modelo, el umbral o la recalibración **después** de ver la prueba, hay que hacer una **nueva selección y una nueva evaluación independiente**.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué diferencia hay entre pérdida, riesgo y métrica? ¿Por qué MAE y RMSE pueden elegir candidatos distintos?
2. ¿Por qué el R² puede ser negativo y qué referencia usa?
3. ¿Qué es una predicción OOF y por qué el preprocesamiento va dentro del `Pipeline`?
4. ¿Por qué el RMSE OOF no es el promedio de los RMSE por pliegue?
5. ¿Cómo se descompone el error de predicción? ¿Qué mide cada parte y cuál no se puede reducir?
6. ¿Qué indican una curva de aprendizaje con ajuste y validación altos y juntos, y otra con brecha grande?
7. ¿Qué ventaja y qué costo tiene K grande o dejar un grupo fuera? ¿Qué es el estimador .632?
8. ¿Qué mide la ROC, qué la AUC y qué el AP? ¿Por qué una AUC alta no garantiza calibración?
9. ¿Cómo se elige un umbral por costo y cuál es el umbral teórico con probabilidades correctas?
10. ¿Qué corrige la validación anidada y por qué no sustituye a la prueba reservada?
11. ¿En qué se parecen y en qué se distinguen AIC/BIC de la validación cruzada?

<details><summary><b>Respuestas</b></summary>

1. La pérdida mide el error de un caso, el riesgo es su promedio en la población de interés y la métrica es el resumen en una muestra. MAE pesa linealmente y RMSE castiga los errores grandes, así que pueden dar rankings distintos; se fija la métrica antes de ver resultados.
2. R² = 1 − SSE/SST compara con predecir la media del conjunto evaluado; si el modelo es peor que esa referencia, es negativo (por ejemplo, con un desplazamiento constante).
3. La predicción de cada fila viene de un modelo ajustado sin su pliegue. El `Pipeline` aprende escalas y coeficientes solo con las filas de ajuste de cada pliegue: evita la fuga de datos.
4. La raíz no es lineal: el RMSE OOF reúne todos los errores al cuadrado y saca una sola raíz (cada observación pesa igual); el promedio de RMSE saca raíz por pliegue (cada pliegue pesa igual). Con pliegues de distinto tamaño difieren más.
5. Sesgo² (distancia entre la señal y el ajuste promedio), varianza (cuánto cambia el ajuste con la muestra) y ruido (variación de Y dado X). El ruido no se reduce.
6. Ajuste y validación altos y juntos: sesgo alto (falta flexibilidad). Brecha grande con ajuste bajo: varianza alta (más datos, penalización o menos complejidad).
7. Menos sesgo de estimación (se entrena con más datos) a cambio de más ajustes y, a veces, más varianza. El .632 mezcla el error de ajuste (optimista) y el OOB (pesimista): 0.368·ajuste + 0.632·OOB.
8. La ROC grafica recall frente a tasa de FP para cada umbral; la AUC es el área (probabilidad de ordenar bien un par positivo–negativo); el AP resume la curva precision–recall. Resumen el ordenamiento, no la calibración ni el umbral.
9. Se evalúa el costo OOF en una cuadrícula de umbrales y se elige el menor (regla fijada antes). Con probabilidades correctas, se bloquea si p > c_FP/(c_FP + c_FN).
10. Corrige el optimismo de haber elegido con el mismo puntaje que se informa: el bucle interno elige y el externo mide. No sustituye a la prueba reservada, que evalúa la regla cerrada una sola vez.
11. AIC/BIC penalizan la complejidad con una fórmula usando solo las filas de ajuste (y suponen independencia); la validación cruzada mide error en datos no vistos. Pueden elegir modelos distintos (el BIC es más estricto).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** Clase 40 · Naive Bayes y clasificación probabilística (Prof. Alvarado) y Clase 43 · evaluación estadística de modelos predictivos y bootstrap. Revisa `Guias/apuntes/C38_Ejercicios_Cuaderno_EvaluacionModelos.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
