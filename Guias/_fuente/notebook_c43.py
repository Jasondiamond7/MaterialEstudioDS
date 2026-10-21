"""Genera el notebook guiado de evaluación estadística y bootstrap (Clase 43).

Uso normal:      python notebook_c43.py             -> Guias/notebooks/C43_EvaluacionBootstrap.ipynb
Modo de prueba:  C43_TEST=1 python notebook_c43.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c43_numeros.json (lo escribe figuras_c43.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib, scipy, statsmodels y scikit-learn.
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C43_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C43_EvaluacionBootstrap_test.ipynb" if TEST else "C43_EvaluacionBootstrap.ipynb")
N = json.loads((HERE / "c43_numeros.json").read_text(encoding="utf-8"))
N["wil"] = N["prop"]["exactitud"]["metodos"]["Wilson"]
N["exa"] = N["prop"]["exactitud"]["metodos"]["Exacto (Clopper-Pearson)"]
N["bay"] = N["prop"]["exactitud"]["metodos"]["Bayes (Beta(1 + k, 1 + n − k))"]
N["btp"] = N["prop"]["exactitud"]["metodos"]["Bootstrap percentil"]
N["btp_s"] = N["prop"]["sensibilidad"]["metodos"]["Bootstrap percentil"]
N["wal"] = N["prop"]["exactitud"]["metodos"]["Normal (Wald)"]
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
# Clase 43 · Evaluación estadística de modelos predictivos y bootstrap
**Diplomado Data Science** · Prof. Pedro Luiz Ramos · retoma intervalos y valores p (Clase 22), la credibilidad bayesiana (Clase 23), SVM y los datos de Wisconsin (Clase 37) y OOF, validación cruzada y bootstrap/.632 (Clase 38)

Material de apoyo:
- `Guias/C43_Guia_EvaluacionBootstrap.pdf` — guía visual de la clase.
- `Guias/apuntes/C43_Apuntes_EvaluacionBootstrap.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C43_Ejercicios_Cuaderno_EvaluacionBootstrap.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta «Clase 43 …»): `clase.pdf` (diapositivas), `evaluacion_bootstrap.py` y `.R.txt`, y `wisconsin.csv`.
- Transcripción: `VideosClases/Clase 43 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels` y `scikit-learn`. Si no los tienes:
```
pip install numpy pandas matplotlib scipy statsmodels scikit-learn
```

**Datos**: `wisconsin.csv` (569 muestras; predictores `radio`, `textura`, `suavidad` y `concavidad`; `maligno` = 1 si es maligno) con las columnas `conjunto` (398 de entrenamiento y 171 de prueba) y `fold` (pliegue de validación cruzada del entrenamiento): **la partición del profesor ya está hecha** y se usa tal cual. Es la misma partición de `wisconsin_clase.csv` de la Clase 37 (con las columnas extra `id` y `fold`). El notebook busca el archivo en la carpeta de la Clase 43 **sin modificarla**. **Aviso del profesor**: la prueba ya fue revisada en clases anteriores; aquí se reutiliza como demostración, no como validación externa nueva.

### Objetivos
1. Entender por qué una métrica de prueba es una **estimación con incertidumbre** y calcular el error estándar de una exactitud.
2. Hacer un **bootstrap a mano y con código**: remuestrear con reposición, error estándar e intervalo percentil.
3. **Reproducir** los resultados del profesor con sus semillas: AIC y BIC; AUC 0.9394 y 0.9746; log loss 0.2997 y 0.1833; SE 0.9306 e IC [3.2; 7.0] de la media.
4. Construir **IC bootstrap** de log loss, AUC, exactitud y sensibilidad con **modelos fijos** (sin reentrenar), y compararlos con otros intervalos (normal, Wilson, exacto, bayesiano).
5. Comparar dos modelos con la **diferencia pareada**, entender por qué **solapar IC no basta**, y conocer **McNemar** y **DeLong** (extensiones).
6. Saber qué cubre el intervalo y qué no (reentrenar, tamaño de la prueba) y construir todo con **Pipelines** ajustados solo con el entrenamiento.
7. Traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Una diferencia observada no basta |
| A | 10 | A2 · ¿Qué significa un IC del 95 %? |
| A | 20 | A3 · Bootstrap de una media (semilla del profesor) |
| A | 10 | A4 · ¿Qué fracción queda fuera de una remuestra? |
| A | 10 | A5 · Bootstrap de MAE y RMSE |
| B | 10 | B1 · Los datos y la partición |
| B | 15 | B2 · AIC y BIC en entrenamiento |
| B | 20 | B3 · Pipelines y predicciones fuera de pliegue |
| B | 15 | B4 · Evaluar una sola vez en prueba |
| C | 15 | C1 · Log loss: remuestrear filas |
| C | 15 | C2 · AUC: remuestrear por clase |
| C | 15 | C3 · Exactitud y sensibilidad: cinco intervalos |
| C | 10 | C4 · Curva ROC con banda bootstrap |
| C | 25 | C5 · La diferencia pareada |
| D | 15 | D1 · McNemar (extensión) |
| D | 15 | D2 · DeLong (extensión) |
| D | 20 | D3 · Reentrenar: otra pregunta |
| D | 10 | D4 · El tamaño de la prueba |
| D | 10 | D5 · Alertas del material, verificadas con código |
| D | 10 | D6 · Tabla de correspondencia R ↔ Python |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Lo aleatorio usa semilla fija (las del profesor en lo que reproduce su script); con otras versiones de las librerías puede variar la última cifra.
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats

import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
B = 2000    # número de remuestras (el del profesor)
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso (Clases 22, 23, 37 y 38)
- **IC y valor p** (Clase 22): un IC del 95 % es un **procedimiento** que en repetición contiene el parámetro el 95 % de las veces. Un contraste «sin diferencia» se rechaza al 5 % cuando el IC de la diferencia no contiene 0.
- **Intervalo creíble** (Clase 23): con una previa Beta(1, 1) y k aciertos en n casos, la posterior es Beta(1 + k, 1 + n − k).
- **Regresión logística, AUC y Wisconsin** (Clase 37): las mismas 569 muestras; el modelo «amplio» (cuatro predictores) tiene AUC 0.9746 en la prueba.
- **OOF, validación cruzada y bootstrap/.632** (Clase 38): una predicción *fuera de pliegue* no participó en el ajuste que la produjo; un caso queda fuera de una remuestra bootstrap con probabilidad (1 − 1/n)ⁿ ≈ 0.368.
- **Pipeline**: todo el preprocesamiento (escalar) vive dentro del modelo y se ajusta **solo con los datos de entrenamiento** de cada pliegue.

### Funciones comunes
""")
code("""
def buscar_clase():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 43*/wisconsin.csv"):
            return p.parent
        if (base / "wisconsin.csv").exists():
            return base
    raise FileNotFoundError("No encuentro wisconsin.csv: abre el notebook dentro de la carpeta del curso.")


CARPETA = buscar_clase()


def q(x, axis=None):
    # intervalo percentil del 95 %: cuantiles 2.5 % y 97.5 %
    return np.quantile(x, [0.025, 0.975], axis=axis)


def hist_ic(ax, datos, estimacion, titulo, color="steelblue"):
    ax.hist(datos, bins=25, color=color, alpha=0.85)
    ax.axvline(estimacion, color="firebrick", lw=2)
    for x in q(datos):
        ax.axvline(x, color="black", ls="--")
    ax.set_title(titulo, fontsize=10)


print("Material:", CARPETA.name)
""")

# ================================================================== PARTE A
md("""
## Parte A · Incertidumbre y bootstrap con números pequeños

### A1 · Una diferencia observada no basta (15 min)
Diapositivas 2 y 11: dos modelos tienen log loss 0.30 y 0.28 (diferencia 0.02). Con **otra muestra de evaluación** la diferencia podría ser otra. Primero se ve cuánto cambia una sola exactitud: un clasificador con probabilidad de acierto **0.90** se evalúa en muchas muestras de n = 100 y de n = 1 000.
""")
code("""
rng = np.random.default_rng(2026)
p0 = 0.9
for n in (100, 1000):
    obs = rng.binomial(n, p0, size=20000) / n            # exactitud observada en 20 000 pruebas distintas
    print(f"n = {n:>4}: media {obs.mean():.4f} | desviación {obs.std():.4f} | fórmula sqrt(p(1-p)/n) = {np.sqrt(p0 * (1 - p0) / n):.4f}")

# el mismo 90 % con distinta precisión (diapositiva 12): intervalo normal aproximado
for k, n in ((90, 100), (900, 1000)):
    ah = k / n
    se = np.sqrt(ah * (1 - ah) / n)
    print(f"{k} de {n}: SE = {se:.4f} | IC normal = [{ah - 1.96 * se:.3f}, {ah + 1.96 * se:.3f}]")
""")
md("""
La desviación de la exactitud observada coincide con √(p(1 − p)/n): **0.030** con 100 casos y **0.0095** con 1 000. El modelo no cambió; cambiaron los casos con que se evalúa. Hay que distinguir la **desviación estándar de los datos** (cuánto difieren los casos entre sí) del **error estándar de una estimación** (cuánto varía la exactitud, el log loss, etc. al cambiar la muestra).

### ✍️ Ejercicio A1
Un clasificador acierta con probabilidad 0.80. Simula 20 000 pruebas con n = 50 y con n = 500, calcula la desviación de la exactitud observada y compárala con la fórmula. ¿Cuántas veces menor es con 500 casos?
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(1)
p0 = 0.8
sd = {}
for n in (50, 500):
    obs = rng.binomial(n, p0, size=20000) / n
    sd[n] = obs.std()
    print(f"n = {n}: desviación {sd[n]:.4f} | fórmula {np.sqrt(p0 * (1 - p0) / n):.4f}")
print("cociente:", round(sd[50] / sd[500], 2), "(teórico sqrt(10) =", round(np.sqrt(10), 2), ")")
""", "Con n = 50 sale ≈ 0.057 y con n = 500 ≈ 0.018: el cociente es ≈ √10 ≈ 3.16. Para reducir el SE a la mitad hacen falta **4 veces** más casos.")

md("""
### A2 · ¿Qué significa un IC del 95 %? (10 min)
Diapositiva 14: se generan 40 muestras normales de tamaño 30 (media 0, desviación 1 conocida) y se construye un IC para la media en cada una. El **95 %** describe el procedimiento: en repetición, ≈ 95 % de los intervalos contiene la media verdadera; **no todos deben hacerlo**.
""")
code("""
rng = np.random.default_rng(43002)
cubre = 0
for e in range(40):
    x = rng.normal(0, 1, 30)
    h = 1.96 / np.sqrt(30)
    cubre += (x.mean() - h <= 0 <= x.mean() + h)
print("intervalos que contienen la media verdadera:", cubre, "de 40")
""")
md("""
Con esta semilla **[[incert.cubre]] de 40** intervalos contienen la media verdadera (en la diapositiva son 38). Con más repeticiones la fracción se acerca a 0.95.

### ✍️ Ejercicio A2
Repite el experimento con **4 000** muestras (en vez de 40) y calcula la fracción de intervalos que contienen la media verdadera.
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(5)
h = 1.96 / np.sqrt(30)
medias = rng.normal(0, 1, (4000, 30)).mean(axis=1)
print("cobertura:", np.mean((medias - h <= 0) & (0 <= medias + h)).round(4))
""", "Sale ≈ 0.95: la cobertura describe lo que ocurre **en repetición**, no cada intervalo.")

md("""
### A3 · Bootstrap de una media (20 min)
Diapositivas 29 a 36: cinco valores (2, 4, 5, 7, 8; media 5.2). El **bootstrap** sortea 5 IDs **con reposición** (la remuestra tiene el mismo tamaño), calcula la media y repite B = 2 000 veces. Las B medias forman la **distribución bootstrap**; su desviación es el **error estándar bootstrap** y sus percentiles 2.5 y 97.5 dan el **IC percentil**. Se usa la semilla del profesor (20261012) y las mismas llamadas que su script.
""")
code("""
valores = np.array([2, 4, 5, 7, 8])
rng = np.random.default_rng(20261012)
medias_boot = np.empty(B)
for b in range(B):
    muestra = rng.choice(valores, size=len(valores), replace=True)
    medias_boot[b] = np.mean(muestra)
print("Media:", np.mean(valores), "| Error estándar bootstrap:", round(np.std(medias_boot, ddof=1), 4))
print("IC percentil:", q(medias_boot))

# tres remuestras como las de la diapositiva 31 (IDs 1 a 5): se ve el emparejamiento ID -> valor
for ids in ([2, 2, 1, 5, 3], [5, 5, 4, 3, 5], [1, 1, 2, 3, 4]):
    vals = valores[[i - 1 for i in ids]]
    print("IDs", ids, "-> valores", vals, "-> media", vals.mean())
fig, ax = plt.subplots(figsize=(6, 3))
hist_ic(ax, medias_boot, valores.mean(), "Bootstrap de la media (B = 2 000)")
plt.show()
""")
md("""
En Python (semilla 20261012): SE = **[[media.se|4]]** e IC percentil **[[media.ic.0|1]]; [[media.ic.1|1]]**. Las diapositivas muestran los resultados de **R** (SE 0.966 e IC [3.4; 7.0]): R y Python tienen generadores aleatorios distintos y producen remuestras distintas; la diferencia es **variación Monte Carlo**.

¿Cuál es el valor «verdadero» del bootstrap? Con 5 valores hay solo 5⁵ = 3 125 remuestras posibles: se puede enumerar la distribución **exacta**.
""")
code("""
todas = np.array([np.mean(t) for t in product(valores, repeat=5)])      # las 3 125 remuestras posibles (con orden)
print("remuestras posibles:", len(todas))
print("SE exacto:", round(todas.std(), 4), "| IC exacto:", q(todas))
print("fórmula plug-in sqrt(sum((x - media)^2)/n)/sqrt(n):", round(np.sqrt(np.sum((valores - valores.mean()) ** 2) / 5) / np.sqrt(5), 4))

# más remuestras no es más datos (diapositiva 36): las primeras 100, 500 y 2 000 remuestras
for bb in (100, 500, 2000):
    print(f"B = {bb:>4}: SE {np.std(medias_boot[:bb], ddof=1):.3f} | IC {q(medias_boot[:bb])}")
""")
md("""
El SE exacto es **[[media.se_exacto|4]]** (y la fórmula directa da lo mismo); el de R (0.966) y el de Python (0.9306) son aproximaciones Monte Carlo de ese valor. Subir B **estabiliza el cálculo**, pero con solo cinco casos originales la incertidumbre de fondo sigue ahí. Una semilla permite repetir la ejecución; no garantiza validez estadística.

### ✍️ Ejercicio A3
Con los tiempos 3, 5, 6, 9 y 12 (media 7.0), calcula con B = 2 000 y `default_rng(7)` el SE bootstrap y el IC percentil de la media. Compara el SE con el exacto (enumerando las 3 125 remuestras).
""")
code("""
# tu código aquí
""")
solution("""
x = np.array([3, 5, 6, 9, 12])
rng = np.random.default_rng(7)
m = np.array([rng.choice(x, size=5, replace=True).mean() for _ in range(B)])
exactas = np.array([np.mean(t) for t in product(x, repeat=5)])
print("SE bootstrap:", round(m.std(ddof=1), 4), "| IC:", q(m))
print("SE exacto   :", round(exactas.std(), 4), "| IC exacto:", q(exactas))
""", "El SE exacto es √(Σ(x − x̄)²/5)/√5 = √(50/5)/√5 ≈ **1.41**; el bootstrap de 2 000 remuestras se acerca a ese valor con pequeña variación Monte Carlo.")

md("""
### A4 · ¿Qué fracción queda fuera de una remuestra? (10 min)
Bridge con la Clase 38 (.632). Una remuestra de tamaño n deja fuera cada caso con probabilidad (1 − 1/n)ⁿ → 1/e ≈ 0.368; entran ≈ 63.2 % de los casos distintos. Es la idea del **fuera de bolsa** (OOB).
""")
code("""
for n in (5, 20, 171, 10000):
    print(f"n = {n:>5}: P(queda fuera) = {(1 - 1 / n) ** n:.4f}")
print("límite 1/e:", round(np.exp(-1), 4))

rng = np.random.default_rng(43003)
n = 171
dentro = [len(np.unique(rng.integers(0, n, n))) / n for _ in range(2000)]
print(f"fracción de casos distintos en una remuestra de 171: media {np.mean(dentro):.4f} (teórica {1 - (1 - 1 / n) ** n:.4f})")
""")
md("""
Con n = 5 queda fuera el **[[oob.p5|3]]** (≈ 1 de cada 3 casos); con n = 171, **[[oob.p171|3]]**, casi el límite 0.368. En las remuestras de la diapositiva 31 el ID 4 no aparece en la primera.

### ✍️ Ejercicio A4
Calcula la probabilidad de que un caso concreto **aparezca al menos una vez** en una remuestra de n = 30 y el número esperado de casos distintos. Verifícalo por simulación.
""")
code("""
# tu código aquí
""")
solution("""
n = 30
p_fuera = (1 - 1 / n) ** n
print("P(aparece) =", round(1 - p_fuera, 4), "| distintos esperados =", round(n * (1 - p_fuera), 2))
rng = np.random.default_rng(3)
print("simulación:", round(np.mean([len(np.unique(rng.integers(0, n, n))) for _ in range(20000)]), 2))
""", "P(aparece) ≈ 0.638 y el número esperado de casos distintos ≈ **19.1**; la simulación coincide.")

md("""
### A5 · Bootstrap de MAE y RMSE (10 min)
Diapositivas 8 y 41: cinco casos con respuesta continua; observado 10, 20, 30, 40, 50 y predicho 9, 22, 27, 44, 45 (errores 1, −2, 3, −4, 5). MAE = 3; RMSE = √11 ≈ 3.317. El bootstrap remuestrea los **pares** (observado, predicho) y recalcula la raíz en cada remuestra. Con cinco casos el IC es **ilustrativo**.
""")
code("""
observado = np.array([10, 20, 30, 40, 50])
predicho = np.array([9, 22, 27, 44, 45])
errores = observado - predicho
print("MAE:", np.mean(np.abs(errores)), "| RMSE:", round(np.sqrt(np.mean(errores ** 2)), 4))
rng = np.random.default_rng(20261014)
reg_boot = np.empty((B, 2))
for b in range(B):
    i = rng.choice(len(observado), size=len(observado), replace=True)
    e = observado[i] - predicho[i]
    reg_boot[b] = np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))
print("IC MAE y RMSE (columnas):\\n", q(reg_boot, axis=0))
""")
md("""
Python da MAE **[[regresion.ic_mae.0|1]]; [[regresion.ic_mae.1|1]]** y RMSE **[[regresion.ic_rmse.0|3]]; [[regresion.ic_rmse.1|3]]**; R (diapositiva 41) da [1.800; 4.200] y [1.949; 4.313]. MAE y RMSE no son comparables entre sí (RMSE pesa más los errores grandes): se elige la métrica **antes** de mirar los resultados.

### ✍️ Ejercicio A5
Con predicciones 11, 19, 33, 38 y 52 (mismos observados), calcula MAE, RMSE y un IC percentil del RMSE con B = 2 000 y `default_rng(11)`.
""")
code("""
# tu código aquí
""")
solution("""
obs = np.array([10, 20, 30, 40, 50])
pre = np.array([11, 19, 33, 38, 52])
e = obs - pre
print("MAE:", np.mean(np.abs(e)), "| RMSE:", round(np.sqrt(np.mean(e ** 2)), 4))
rng = np.random.default_rng(11)
r = np.empty(B)
for b in range(B):
    i = rng.choice(5, size=5, replace=True)
    r[b] = np.sqrt(np.mean((obs[i] - pre[i]) ** 2))
print("IC RMSE:", q(r))
""", "Errores −1, 1, −3, 2, −2: MAE = **1.8** y RMSE = √(19/5) ≈ **1.949**. El IC es amplio porque solo hay 5 casos.")

# ================================================================== PARTE B
md("""
## Parte B · Wisconsin: seleccionar con entrenamiento, evaluar en prueba

### B1 · Los datos y la partición (10 min)
Diapositivas 3 a 6: **seleccionar y evaluar son tareas distintas**: se define la unidad de separación y la métrica; se selecciona la receta con entrenamiento y validación; se reajusta con todo el entrenamiento; se evalúa en prueba sin volver a elegir; y recién entonces se cuantifica la incertidumbre. `id`, `conjunto` y `fold` **no** son predictores.
""")
code("""
datos = pd.read_csv(CARPETA / "wisconsin.csv")
entrenamiento = datos[datos["conjunto"] == "entrenamiento"].reset_index(drop=True)
prueba = datos[datos["conjunto"] == "prueba"].reset_index(drop=True)
print(pd.crosstab(datos["conjunto"], datos["maligno"]))
VARS = ["radio", "textura", "suavidad", "concavidad"]
assert len(datos) == 569 and len(entrenamiento) == 398 and len(prueba) == 171
assert set(entrenamiento.fold) == {1, 2, 3, 4, 5} and (prueba.fold == 0).all() and datos.id.is_unique
print("tamaño de los pliegues:", entrenamiento.fold.value_counts().sort_index().tolist())

# ¿es la misma base de la Clase 37?
c37 = next(Path(CARPETA).parent.glob("Clase 37*/wisconsin_clase.csv"), None)
if c37 is not None:
    d37 = pd.read_csv(c37)
    print("igual que wisconsin_clase.csv (Clase 37):", np.allclose(d37[VARS].to_numpy(), datos[VARS].to_numpy()) and (d37.conjunto.to_numpy() == datos.conjunto.to_numpy()).all())
""")
md("""
Hay **250 benignos y 148 malignos** en el entrenamiento y **107 benignos y 64 malignos** en la prueba. Los pliegues miden 80, 80, 80, 79 y 79: «uno de 80 casos valida y los otros 318 ajustan» (diapositiva 4) es exacto solo para los tres primeros; en los otros dos se ajusta con 319. La base coincide con la de la Clase 37.

### ✍️ Ejercicio B1
Calcula la proporción de malignos en entrenamiento y en prueba, y el tamaño de ajuste de cada pliegue (casos de entrenamiento fuera del pliegue).
""")
code("""
# tu código aquí
""")
solution("""
print("malignos: entrenamiento", round(entrenamiento.maligno.mean(), 4), "| prueba", round(prueba.maligno.mean(), 4))
print("casos de ajuste por pliegue:", [int((entrenamiento.fold != k).sum()) for k in range(1, 6)])
""", "Entrenamiento 148/398 = **0.3719**; prueba 64/171 = **0.3743** (partición estratificada). Casos de ajuste: 318, 318, 318, 319 y 319.")

md("""
### B2 · AIC y BIC en entrenamiento (15 min)
Diapositivas 18 a 28. **AIC = −2ℓ + 2p** y **BIC = −2ℓ + p·ln(n)**, con p = parámetros estimados **incluido el intercepto**. Menor es mejor; solo se comparan dentro de la misma columna y entre modelos comparables (misma respuesta, mismas observaciones, máxima verosimilitud). Se ajustan tres logísticas **sin penalización** con `statsmodels`, como el profesor.
""")
code("""
fs = {"Simple": "maligno ~ radio + textura",
      "Intermedio": "maligno ~ radio + textura + suavidad",
      "Amplio": "maligno ~ radio + textura + suavidad + concavidad"}
modelos = {k: smf.glm(f, data=entrenamiento, family=sm.families.Binomial()).fit() for k, f in fs.items()}
criterios = pd.DataFrame({"Modelo": list(modelos), "Parametros": [len(m.params) for m in modelos.values()],
                          "logLik": [m.llf for m in modelos.values()],
                          "AIC": [m.aic for m in modelos.values()], "BIC": [m.bic_llf for m in modelos.values()]})
display(criterios.round(4))

# comprobación a mano de la fila del modelo amplio (diapositiva 27)
m = modelos["Amplio"]
n_tr, p_ = len(entrenamiento), len(m.params)
print("AIC a mano:", round(-2 * m.llf + 2 * p_, 4), "| BIC a mano:", round(-2 * m.llf + p_ * np.log(n_tr), 4))
""")
md("""
Resultados (diapositiva 27): AIC **[[criterios.0.aic|2]]**, **[[criterios.1.aic|2]]** y **[[criterios.2.aic|2]]**; BIC **[[criterios.0.bic|2]]**, **[[criterios.1.bic|2]]** y **[[criterios.2.bic|2]]**. Ambos criterios favorecen al **amplio**; los 171 casos de prueba no intervienen. En `statsmodels` se usa `bic_llf` (definición basada en la log-verosimilitud, igual que `BIC()` de R); `bic_deviance` es otra definición (basada en la desviación) y **no** coincide, y en versiones antiguas `.bic` era esa.

Ejemplo hipotético de la clase (n = 100): modelos A, B y C con ℓ = −80, −76 y −75.5 y p = 3, 5 y 8 dan AIC 166, 162 y 167 (gana B) y BIC 173.82, 175.03 y 187.84 (gana A): **AIC y BIC pueden elegir distinto**.

### ✍️ Ejercicio B2
Calcula los AIC y BIC del ejemplo hipotético (n = 100) y los ΔAIC respecto del mínimo. ¿Qué modelo elige cada criterio?
""")
code("""
# tu código aquí
""")
solution("""
ll = np.array([-80, -76, -75.5])
p_ = np.array([3, 5, 8])
aic = -2 * ll + 2 * p_
bic = -2 * ll + p_ * np.log(100)
print("AIC:", aic, "| ΔAIC:", aic - aic.min())
print("BIC:", bic.round(2), "| ΔBIC:", (bic - bic.min()).round(2))
""", "AIC = 166, 162 y 167 (ΔAIC 4, 0 y 5) → elige **B**; BIC = 173.82, 175.03 y 187.84 → elige **A**. No hay un AIC o BIC absoluto: todos los candidatos podrían predecir mal.")

md("""
### B3 · Pipelines y predicciones fuera de pliegue (20 min)
Diapositivas 3 a 6: la receta completa se ajusta **dentro del pliegue**: estandarizar con la media y la escala de las filas de ajuste; no estandarizar toda la base antes de separar. En `sklearn` eso significa un `Pipeline(StandardScaler, LogisticRegression)` que se reajusta en cada pliegue. La logística **sin penalización** (`C=np.inf`, sin regularización) es el mismo modelo del GLM; el escalado no cambia sus predicciones, pero lo ponemos dentro del Pipeline para que la receta sea correcta cuando importe.
""")
code("""
def receta():
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, solver="newton-cg", tol=1e-10, max_iter=1000))


COLS = {"Simple": [0, 1], "Intermedio": [0, 1, 2], "Amplio": [0, 1, 2, 3]}
Xe, ye, fe = entrenamiento[VARS].to_numpy(), entrenamiento.maligno.to_numpy(), entrenamiento.fold.to_numpy()


def oof(cols):
    # cada caso se predice con un modelo (y un escalado) que NO lo vio
    p = np.zeros(len(ye))
    for k in np.unique(fe):
        v = fe == k
        m = clone(receta()).fit(Xe[~v][:, cols], ye[~v])
        p[v] = m.predict_proba(Xe[v][:, cols])[:, 1]
    return p


filas = []
for nom, cols in COLS.items():
    p = oof(cols)
    filas.append(dict(modelo=nom, log_loss=log_loss(ye, p), AUC=roc_auc_score(ye, p), exactitud=np.mean((p >= 0.5) == ye)))
oof_tab = pd.DataFrame(filas)
display(oof_tab.round(4))
""")
md("""
Con la validación cruzada de cinco pliegues el log loss OOF es **[[oof.Simple.log_loss|3]]**, **[[oof.Intermedio.log_loss|3]]** y **[[oof.Amplio.log_loss|3]]**; el AUC OOF, **[[oof.Simple.auc|3]]**, **[[oof.Intermedio.auc|3]]** y **[[oof.Amplio.auc|3]]**. El amplio también es mejor aquí. Pero **seleccionar con OOF no convierte el resultado del ganador en una evaluación independiente** (diapositiva 5): sirve para elegir, no para informar.

### ✍️ Ejercicio B3
Agrega una cuarta receta con `radio`, `textura` y `concavidad` (sin `suavidad`) y calcula su log loss y AUC OOF. ¿Supera al amplio?
""")
code("""
# tu código aquí
""")
solution("""
p = oof([0, 1, 3])
print("radio + textura + concavidad -> log loss", round(log_loss(ye, p), 4), "| AUC", round(roc_auc_score(ye, p), 4))
print(oof_tab.round(4).to_string(index=False))
""", "Compara con la tabla: si su log loss OOF es mayor que el del amplio (0.154), no lo supera; la decisión se toma con OOF y **antes** de mirar la prueba.")

md("""
### B4 · Evaluar una sola vez en prueba (15 min)
Diapositivas 38 y 39. Se reajustan los modelos simple y amplio con **los 398 casos de entrenamiento**, se obtienen las probabilidades de prueba y se **conservan** (no se vuelve a llamar a `fit` ni a `predict` en lo que sigue). Exactitud y sensibilidad usan el umbral fijo 0.5; AUC usa el orden de las predicciones y log loss, las probabilidades.
""")
code("""
ms, ma = modelos["Simple"], modelos["Amplio"]
y = prueba["maligno"].to_numpy()
p_simple = ms.predict(prueba).to_numpy()
p_amplio = ma.predict(prueba).to_numpy()
umbral = 0.5
comparacion = pd.DataFrame({
    "Modelo": ["Simple", "Amplio"],
    "AUC": [roc_auc_score(y, p_simple), roc_auc_score(y, p_amplio)],
    "LogLoss": [log_loss(y, p_simple), log_loss(y, p_amplio)],
    "Exactitud": [np.mean((p_simple >= umbral) == y), np.mean((p_amplio >= umbral) == y)],
    "Sensibilidad": [np.mean(p_simple[y == 1] >= umbral), np.mean(p_amplio[y == 1] >= umbral)]})
display(comparacion.round(4))
assert comparacion.round(4).AUC.tolist() == [0.9394, 0.9746] and comparacion.round(4).LogLoss.tolist() == [0.2997, 0.1833]
print("casos:", len(y), "| benignos:", int((y == 0).sum()), "| malignos:", int((y == 1).sum()))

# la misma receta con Pipelines de sklearn (ajustada con los 398 casos) da las mismas probabilidades
pp = {nom: clone(receta()).fit(Xe[:, c], ye).predict_proba(prueba[VARS].to_numpy()[:, c])[:, 1] for nom, c in COLS.items()}
print("máx. diferencia GLM vs Pipeline (amplio):", float(np.abs(pp["Amplio"] - p_amplio).max()))
""")
md("""
Resultados de la diapositiva 39: AUC **[[prueba.simple.auc|4]]** y **[[prueba.amplio.auc|4]]**; log loss **[[prueba.simple.log_loss|4]]** y **[[prueba.amplio.log_loss|4]]**; exactitud **[[prueba.simple.exactitud|4]]** y **[[prueba.amplio.exactitud|4]]**; sensibilidad **[[prueba.simple.sensibilidad|4]]** y **[[prueba.amplio.sensibilidad|4]]**. El `Pipeline` reproduce las mismas probabilidades (la diferencia máxima es del orden de 1e−8). **La prueba ya se revisó antes**: estos resultados no son una validación independiente nueva.

### ✍️ Ejercicio B4
Con las probabilidades del modelo amplio, arma la matriz de confusión con umbral 0.5 y calcula sensibilidad y especificidad.
""")
code("""
# tu código aquí
""")
solution("""
pred = p_amplio >= 0.5
vn, fp = int(np.sum((y == 0) & ~pred)), int(np.sum((y == 0) & pred))
fn, vp = int(np.sum((y == 1) & ~pred)), int(np.sum((y == 1) & pred))
print("VN", vn, "FP", fp, "FN", fn, "VP", vp)
print("sensibilidad", round(vp / (vp + fn), 4), "| especificidad", round(vn / (vn + fp), 4))
""", "VN 103, FP 4, FN 6, VP 58: sensibilidad 58/64 = **0.9062**; especificidad 103/107 = **0.9626**. Con solo 64 malignos la sensibilidad es la métrica más incierta (ver C3).")

# ================================================================== PARTE C
md("""
## Parte C · Bootstrap en la prueba con modelos fijos

Diapositiva 38: se ajustan los modelos una vez, se conservan las probabilidades de prueba, se **remuestrean los casos de prueba** con esas probabilidades fijas y se recalculan AUC y log loss en las remuestras (**no los coeficientes**). El intervalo estudia «¿cuánto varía la evaluación de **este** modelo si cambian los casos?».

### C1 · Log loss: remuestrear filas (15 min)
Se sortean n = 171 índices con reposición; el **mismo** vector de índices se aplica a las etiquetas y a los dos modelos. Semilla del profesor: 20261013.
""")
code("""
n = len(y)
rng = np.random.default_rng(20261013)
logloss_boot = np.empty((B, 2))
for b in range(B):
    indices = rng.choice(n, size=n, replace=True)
    logloss_boot[b, 0] = log_loss(y[indices], p_simple[indices])
    logloss_boot[b, 1] = log_loss(y[indices], p_amplio[indices])
ic_logloss = q(logloss_boot, axis=0)
print("IC log loss; columnas Simple, Amplio:\\n", ic_logloss)
print("SE bootstrap:", logloss_boot.std(axis=0, ddof=1).round(4))
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
hist_ic(axs[0], logloss_boot[:, 0], comparacion.LogLoss[0], "Log loss: simple")
hist_ic(axs[1], logloss_boot[:, 1], comparacion.LogLoss[1], "Log loss: amplio", "teal")
plt.show()
""")
md("""
Python: simple **[[ic.ll_simple.0|4]]; [[ic.ll_simple.1|4]]** y amplio **[[ic.ll_amplio.0|4]]; [[ic.ll_amplio.1|4]]**. R (diapositiva 53): [0.2124; 0.4005] y [0.1066; 0.2798]. Las diferencias pequeñas son variación Monte Carlo. Los errores seguros pueden elevar mucho la pérdida de una remuestra. **No** se aplica log loss a puntajes de SVM o de anomalía que no sean probabilidades.

### ✍️ Ejercicio C1
Calcula con `default_rng(5)` y B = 1 000 el IC percentil del log loss del modelo amplio y su ancho. ¿Cambia mucho con respecto al de B = 2 000?
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(5)
v = np.empty(1000)
for b in range(1000):
    i = rng.choice(n, size=n, replace=True)
    v[b] = log_loss(y[i], p_amplio[i])
print("IC (B = 1 000):", q(v), "| ancho", round(float(np.diff(q(v))[0]), 4))
print("IC (B = 2 000):", ic_logloss[:, 1], "| ancho", round(float(np.diff(ic_logloss[:, 1])[0]), 4))
""", "Los dos intervalos son casi iguales (ancho ≈ 0.17): subir B no elimina la incertidumbre de tener 171 casos.")

md("""
### C2 · AUC: remuestrear por clase (15 min)
Para la AUC se sortean **107 benignos y 64 malignos** con reposición (se conservan los tamaños de cada clase); ambos modelos usan los mismos IDs. Semilla del profesor: 20261017.
""")
code("""
negativos, positivos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
rng = np.random.default_rng(20261017)
auc_boot = np.empty((B, 2))
for b in range(B):
    i0 = rng.choice(negativos, size=len(negativos), replace=True)
    i1 = rng.choice(positivos, size=len(positivos), replace=True)
    i = np.concatenate([i0, i1])
    auc_boot[b, 0] = roc_auc_score(y[i], p_simple[i])
    auc_boot[b, 1] = roc_auc_score(y[i], p_amplio[i])
ic_auc = q(auc_boot, axis=0)
print("IC AUC; columnas Simple, Amplio:\\n", ic_auc)
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
hist_ic(axs[0], auc_boot[:, 0], comparacion.AUC[0], "AUC: simple")
hist_ic(axs[1], auc_boot[:, 1], comparacion.AUC[1], "AUC: amplio", "teal")
plt.show()
""")
md("""
AUC simple **[[prueba.simple.auc|4]]** con IC **[[ic.auc_simple.0|4]]; [[ic.auc_simple.1|4]]** y amplia **[[prueba.amplio.auc|4]]** con IC **[[ic.auc_amplio.0|4]]; [[ic.auc_amplio.1|4]]** (R: [0.9004; 0.9708] y [0.9503; 0.9915]). Una advertencia: este intervalo **no incluye** la variación por volver a entrenar (ver D3).

### ✍️ Ejercicio C2
¿Qué cambia si en vez de remuestrear por clase se remuestrean las 171 filas sin estratificar? Calcula el IC de la AUC del modelo amplio con `default_rng(8)` y compara el ancho.
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(8)
a = []
while len(a) < B:
    i = rng.choice(n, size=n, replace=True)
    if len(np.unique(y[i])) == 2:           # la AUC exige las dos clases
        a.append(roc_auc_score(y[i], p_amplio[i]))
a = np.array(a)
print("filas   :", q(a), "ancho", round(float(np.diff(q(a))[0]), 4))
print("por clase:", ic_auc[:, 1], "ancho", round(float(np.diff(ic_auc[:, 1])[0]), 4))
""", "Los anchos son parecidos; al remuestrear filas el número de malignos varía entre remuestras (en la estratificada se fija en 64 y 107).")

md("""
### C3 · Exactitud y sensibilidad: cinco intervalos (15 min)
Puente con las Clases 22 y 23. Para la exactitud del modelo amplio (161 aciertos de 171) y su sensibilidad (58 de 64) se comparan: **normal (Wald)**, **Wilson**, **exacto (Clopper-Pearson)**, **bayesiano** (Beta(1 + k, 1 + n − k), previa uniforme) y **bootstrap percentil**.
""")
code("""
pred_a = p_amplio >= 0.5
rng = np.random.default_rng(43004)
acc_b, sen_b = np.empty(B), np.empty(B)
for b in range(B):
    i = rng.choice(n, size=n, replace=True)
    acc_b[b] = np.mean(pred_a[i] == y[i])
    sen_b[b] = np.mean(pred_a[i][y[i] == 1])

for nombre, k, nn, bt in (("exactitud", int(np.sum(pred_a == y)), n, acc_b), ("sensibilidad", int(np.sum(pred_a[y == 1])), int((y == 1).sum()), sen_b)):
    ah = k / nn
    filas = {"Wald": proportion_confint(k, nn, method="normal"), "Wilson": proportion_confint(k, nn, method="wilson"),
             "Exacto": proportion_confint(k, nn, method="beta"), "Bayes": tuple(stats.beta.ppf([0.025, 0.975], 1 + k, 1 + nn - k)),
             "Bootstrap": tuple(q(bt))}
    print(f"{nombre}: {k}/{nn} = {ah:.4f} | SE Wald {np.sqrt(ah * (1 - ah) / nn):.4f} | SE bootstrap {bt.std(ddof=1):.4f}")
    display(pd.DataFrame(filas, index=["inferior", "superior"]).T.round(4))
""")
md("""
Para la exactitud (161/171 = 0.9415), Wald da **[[wal.0|3]]; [[wal.1|3]]**, Wilson **[[wil.0|3]]; [[wil.1|3]]**, exacto **[[exa.0|3]]; [[exa.1|3]]**, bayesiano **[[bay.0|3]]; [[bay.1|3]]** y bootstrap **[[btp.0|3]]; [[btp.1|3]]**. El Wald es el que más se acerca a 1 (poco fiable con exactitudes altas); los otros cuatro coinciden. Para la sensibilidad, con solo 64 malignos, el bootstrap da **[[btp_s.0|3]]; [[btp_s.1|3]]**: mucho más ancho que el de la exactitud.

### ✍️ Ejercicio C3
Calcula la especificidad del modelo amplio (103 de 107) y su IC de Wilson y su IC bootstrap percentil (con `default_rng(9)`, B = 2 000, remuestreando solo los benignos).
""")
code("""
# tu código aquí
""")
solution("""
benignos = np.flatnonzero(y == 0)
k = int(np.sum(~pred_a[benignos]))
print("especificidad:", k, "/", len(benignos), "=", round(k / len(benignos), 4))
print("Wilson:", np.round(proportion_confint(k, len(benignos), method="wilson"), 4))
rng = np.random.default_rng(9)
esp = [np.mean(~pred_a[rng.choice(benignos, len(benignos), replace=True)]) for _ in range(B)]
print("bootstrap:", np.round(q(esp), 4))
""", "103/107 = **0.9626**; Wilson ≈ [0.908; 0.985] y bootstrap ≈ [0.925; 0.991]: el bootstrap percentil es algo más optimista cerca de 1.")

md("""
### C4 · Curva ROC con banda bootstrap (10 min)
Para cada remuestra por clase se calcula la curva ROC y se interpola la sensibilidad en una rejilla de 1 − especificidad; los percentiles 2.5 y 97.5 dan una **banda**. Es una vista gráfica de la incertidumbre de la AUC.
""")
code("""
grid = np.linspace(0, 1, 101)
rng = np.random.default_rng(43005)
Bb = 500
tpr = {"simple": np.empty((Bb, 101)), "amplio": np.empty((Bb, 101))}
for b in range(Bb):
    i = np.concatenate([rng.choice(negativos, len(negativos), replace=True), rng.choice(positivos, len(positivos), replace=True)])
    for nom, p in (("simple", p_simple), ("amplio", p_amplio)):
        f_, t_, _ = roc_curve(y[i], p[i])
        tpr[nom][b] = np.interp(grid, f_, t_)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for nom, p, col in (("simple", p_simple, "tab:blue"), ("amplio", p_amplio, "teal")):
    lo, hi = q(tpr[nom], axis=0)
    ax.fill_between(grid, lo, hi, color=col, alpha=0.2)
    f_, t_, _ = roc_curve(y, p)
    ax.plot(f_, t_, color=col, lw=2, label=f"{nom}: AUC {roc_auc_score(y, p):.4f}")
ax.plot([0, 1], [0, 1], ":", color="gray")
ax.set(xlabel="1 − especificidad", ylabel="sensibilidad", title="ROC y banda bootstrap del 95 %")
ax.legend()
plt.show()
print("ancho medio de la banda:", {k: round(float(np.mean(q(v, axis=0)[1] - q(v, axis=0)[0])), 3) for k, v in tpr.items()})
""")
md("""
La banda del modelo simple es más ancha (**[[roc.ancho_medio.simple|3]]** frente a **[[roc.ancho_medio.amplio|3]]** de ancho medio). Las bandas se solapan en parte del rango, pero eso **no** responde si los modelos difieren: para eso hace falta la diferencia pareada (C5).

### ✍️ Ejercicio C4
Calcula con las mismas remuestras la sensibilidad en 1 − especificidad = 0.1 del modelo amplio (punto de la rejilla `grid[10]`) y su IC percentil.
""")
code("""
# tu código aquí
""")
solution("""
j = 10
print("1 - esp =", grid[j], "| IC de la sensibilidad (amplio):", np.round(q(tpr["amplio"][:, j]), 4), "| simple:", np.round(q(tpr["simple"][:, j]), 4))
""", "En ese punto el IC del amplio es bastante más alto que el del simple: las bandas se separan en la zona de pocos falsos positivos.")

md("""
### C5 · La diferencia pareada (25 min)
Diapositivas 15 y 47 a 50. Los dos modelos ven los **mismos casos**: un caso difícil perjudica a ambos. Por eso se usan **los mismos índices** en las dos columnas y la estadística pasa a ser una **diferencia**: Δ log loss = simple − amplio y ΔAUC = amplio − simple (**positivo favorece al amplio** en ambos). Se toman los **percentiles de las diferencias**, no se restan los extremos de dos intervalos separados.
""")
code("""
delta_logloss = logloss_boot[:, 0] - logloss_boot[:, 1]
delta_auc = auc_boot[:, 1] - auc_boot[:, 0]
diferencias = pd.DataFrame({
    "Metrica": ["Log loss simple - amplio", "AUC amplio - simple"],
    "Diferencia": [comparacion.LogLoss[0] - comparacion.LogLoss[1], comparacion.AUC[1] - comparacion.AUC[0]],
    "Inferior": [np.quantile(delta_logloss, 0.025), np.quantile(delta_auc, 0.025)],
    "Superior": [np.quantile(delta_logloss, 0.975), np.quantile(delta_auc, 0.975)]})
display(diferencias.round(4))
print("fracción de remuestras con Δ <= 0:", np.mean(delta_logloss <= 0), np.mean(delta_auc <= 0))
print("correlación de las AUC en las remuestras:", round(float(np.corrcoef(auc_boot[:, 0], auc_boot[:, 1])[0, 1]), 3))
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
for ax, d, t in ((axs[0], delta_logloss, "Δ log loss (simple − amplio)"), (axs[1], delta_auc, "ΔAUC (amplio − simple)")):
    ax.hist(d, bins=25, color="steelblue", alpha=0.85)
    ax.axvline(0, color="firebrick", lw=2)
    for x in q(d):
        ax.axvline(x, color="darkgreen", ls="--")
    ax.set_title(t, fontsize=10)
plt.show()
""")
md("""
Resultados: Δ log loss = **[[dif.ll|4]]** con IC **[[dif.ic_ll.0|4]]; [[dif.ic_ll.1|4]]** y ΔAUC = **[[dif.auc|4]]** con IC **[[dif.ic_auc.0|4]]; [[dif.ic_auc.1|4]]** (R: [0.0515; 0.1841] y [0.0123; 0.0624]). Ningún intervalo contiene 0: **el amplio es mejor en las dos métricas**. Ya no hace falta mirar si los IC de cada modelo se tocan.

Ahora, ¿qué pasa si se ignora el emparejamiento? Se remuestrea un vector de índices distinto para cada modelo y se compara el IC del ΔAUC.
""")
code("""
rng = np.random.default_rng(43006)
d_no = np.empty(B)
for b in range(B):
    i1 = np.concatenate([rng.choice(negativos, len(negativos), replace=True), rng.choice(positivos, len(positivos), replace=True)])
    i2 = np.concatenate([rng.choice(negativos, len(negativos), replace=True), rng.choice(positivos, len(positivos), replace=True)])
    d_no[b] = roc_auc_score(y[i2], p_amplio[i2]) - roc_auc_score(y[i1], p_simple[i1])
for nom, d in (("pareado", delta_auc), ("no pareado", d_no)):
    lo, hi = q(d)
    print(f"{nom:>10}: IC [{lo:.4f}, {hi:.4f}] | ancho {hi - lo:.4f} | contiene 0: {lo <= 0 <= hi}")

# ¿se solapan los IC de cada AUC?
print("IC AUC simple :", ic_auc[:, 0].round(4), "| IC AUC amplio:", ic_auc[:, 1].round(4), "| se solapan:", bool(ic_auc[0, 1] < ic_auc[1, 0]))
""")
md("""
Pareado: **[[solap.pareado.0|4]]; [[solap.pareado.1|4]]** (ancho **[[solap.ancho_pareado|3]]**). No pareado: **[[solap.no_pareado.0|4]]; [[solap.no_pareado.1|4]]** (ancho **[[solap.ancho_no_pareado|3]]**) y **sí contiene 0**. Emparejar angosta el intervalo porque las AUC de las remuestras están correlacionadas (≈ [[dif.corr_auc|2]]). Además, los IC de cada AUC **se solapan** ([[solap.solape.0|3]]–[[solap.solape.1|3]]) y la diferencia pareada excluye 0: **solapar no es «no hay diferencia»**.

Un IC que excluye 0 muestra una diferencia **estadística**; que sea **útil** depende de una mejora relevante fijada **antes** (diapositiva 51: 0.010 no es una regla universal). Si el IC contiene 0, la lectura es «no hay evidencia clara», no «los modelos son equivalentes».

### ✍️ Ejercicio C5
Calcula la diferencia pareada de **exactitud** (amplio − simple, umbral 0.5) con los **mismos** índices por fila de C1 (semilla 20261013): estimación original, IC percentil y si contiene 0.
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(20261013)
ok_s, ok_a = (p_simple >= 0.5) == y, (p_amplio >= 0.5) == y
dacc = np.empty(B)
for b in range(B):
    i = rng.choice(n, size=n, replace=True)
    dacc[b] = ok_a[i].mean() - ok_s[i].mean()
print("Δ exactitud original:", round(ok_a.mean() - ok_s.mean(), 4), "| IC:", q(dacc).round(4), "| contiene 0:", bool(q(dacc)[0] <= 0 <= q(dacc)[1]))
""", "La diferencia original es 0.0819 y el IC pareado ≈ [0.035; 0.129]: no contiene 0. Cuidado: la exactitud depende del umbral 0.5; las otras dos métricas no.")

# ================================================================== PARTE D
md("""
## Parte D · Alcance del intervalo y extensiones

### D1 · McNemar (extensión, 15 min)
**No se vio en clase.** McNemar compara las **exactitudes** de dos clasificadores en los mismos casos con un umbral fijo. Se arma una tabla 2 × 2 de aciertos; solo cuentan los **discordantes**: b (solo el simple acierta) y c (solo el amplio acierta). Estadístico: χ² = (|b − c| − 1)²/(b + c), con 1 grado de libertad; la versión exacta usa la binomial con p = 0.5.
""")
code("""
ok_s, ok_a = (p_simple >= 0.5) == y, (p_amplio >= 0.5) == y
a_ = int(np.sum(ok_s & ok_a)); b_ = int(np.sum(ok_s & ~ok_a)); c_ = int(np.sum(~ok_s & ok_a)); d_ = int(np.sum(~ok_s & ~ok_a))
print(pd.DataFrame([[a_, b_], [c_, d_]], index=["simple acierta", "simple falla"], columns=["amplio acierta", "amplio falla"]))
stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
print(f"χ² = {stat:.4f} | p (χ², 1 gl) = {stats.chi2.sf(stat, 1):.5f} | p exacto = {stats.binomtest(min(b_, c_), b_ + c_, 0.5).pvalue:.5f}")
""")
md("""
Tabla: ambos aciertan **[[mcnemar.a]]**; solo el simple, **[[mcnemar.b]]**; solo el amplio, **[[mcnemar.c]]**; ambos fallan **[[mcnemar.d]]**. χ² = **[[mcnemar.stat|2]]**, p = **[[mcnemar.p_chi|5]]** (exacto [[mcnemar.p_exacto|5]]). Solo se usan los [[mcnemar.n_disc]] casos discordantes de 171. McNemar responde por la **exactitud con el umbral 0.5**; no compara AUC ni log loss.

### ✍️ Ejercicio D1
Con 200 casos, ambos aciertan 150; solo A acierta 20; solo B acierta 10; ambos fallan 20. Calcula el estadístico y el p exacto.
""")
code("""
# tu código aquí
""")
solution("""
b, c = 20, 10
stat = (abs(b - c) - 1) ** 2 / (b + c)
print("χ² =", round(stat, 4), "| p χ² =", round(float(stats.chi2.sf(stat, 1)), 4), "| p exacto =", round(float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), 4))
""", "χ² = 81/30 = **2.70**, p ≈ 0.100 (exacto ≈ 0.099): no se rechaza «sin diferencia» al 5 %, aunque A acierta 85 % y B 80 %.")

md("""
### D2 · DeLong (extensión, 15 min)
**Tampoco se vio en clase.** La prueba de **DeLong** compara dos AUC **correlacionadas** (mismos casos) con una fórmula basada en la estructura de los pares (positivo, negativo). Es la alternativa analítica al bootstrap pareado para el ΔAUC.
""")
code("""
def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    m, n_ = pos.sum(), neg.sum()
    v10, v01, auc = [], [], []
    for p in (p1, p2):
        a, b = p[pos][:, None], p[neg][None, :]
        psi = (a > b) + 0.5 * (a == b)
        v10.append(psi.mean(1)); v01.append(psi.mean(0)); auc.append(psi.mean())
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_
    z = (auc[1] - auc[0]) / np.sqrt(var)
    return dict(diff=auc[1] - auc[0], se=np.sqrt(var), z=z, p=2 * stats.norm.sf(abs(z)))


dl = delong(y, p_simple, p_amplio)
print({k: round(float(v), 4) for k, v in dl.items()})
print("IC 95 % DeLong:", np.round([dl["diff"] - 1.96 * dl["se"], dl["diff"] + 1.96 * dl["se"]], 4), "| bootstrap pareado:", q(delta_auc).round(4), "| SE bootstrap:", round(float(delta_auc.std(ddof=1)), 4))
""")
md("""
DeLong da ΔAUC = **[[delong.diff|4]]**, SE **[[delong.se|4]]**, z = **[[delong.z|2]]** y p = **[[delong.p|4]]**; el bootstrap pareado da SE **[[solap.se_pareado|4]]**: los dos caminos coinciden en lo esencial.

### ✍️ Ejercicio D2
Aplica `delong` al modelo **intermedio** frente al **simple** con las probabilidades de prueba de los GLM (`modelos["Intermedio"].predict(prueba)`). ¿Es significativa la diferencia de AUC?
""")
code("""
# tu código aquí
""")
solution("""
p_int = modelos["Intermedio"].predict(prueba).to_numpy()
d = delong(y, p_simple, p_int)
print({k: round(float(v), 4) for k, v in d.items()})
""", "Sale ΔAUC ≈ 0.030 (SE ≈ 0.011, p ≈ 0.004): el intermedio es significativamente mejor que el simple; compárala con el intermedio frente al amplio en el ejercicio de cierre (ΔAUC ≈ 0.005).")

md("""
### D3 · Reentrenar: otra pregunta (20 min)
Diapositivas 17 y 52: el bootstrap de la prueba estudia «¿cuánto varía la evaluación de **este** modelo fijo?». Otra pregunta distinta es «¿cuánto varía el modelo si lo **vuelvo a entrenar**?». Para estudiarla se repite la **partición** (reentrenando y reevaluando). Aquí: 300 particiones estratificadas de las 569 muestras con 171 en prueba (se ignora la partición del profesor), con Pipelines.
""")
code("""
Xt, yt = datos[VARS].to_numpy(), datos.maligno.to_numpy()
sss = StratifiedShuffleSplit(n_splits=300, test_size=171, random_state=43007)
dauc, dll, auc_a = [], [], []
for itr, ite in sss.split(Xt, yt):
    r = []
    for cols in ([0, 1], [0, 1, 2, 3]):
        m = clone(receta()).fit(Xt[itr][:, cols], yt[itr])
        p = m.predict_proba(Xt[ite][:, cols])[:, 1]
        r.append((roc_auc_score(yt[ite], p), log_loss(yt[ite], p)))
    dauc.append(r[1][0] - r[0][0]); dll.append(r[0][1] - r[1][1]); auc_a.append(r[1][0])
dauc, dll, auc_a = map(np.array, (dauc, dll, auc_a))
print("ΔAUC: media", round(dauc.mean(), 4), "| desviación", round(dauc.std(ddof=1), 4), "| IC", q(dauc).round(4), "| amplio mejor en", np.mean(dauc > 0))
print("AUC del amplio: desviación", round(auc_a.std(ddof=1), 4), "| rango", auc_a.min().round(4), auc_a.max().round(4))
print("ancho del IC del ΔAUC: repetir partición", round(float(np.diff(q(dauc))[0]), 4), "| bootstrap de la prueba (modelos fijos)", round(float(np.diff(q(delta_auc))[0]), 4))
""")
md("""
Con 300 particiones el ΔAUC tiene media **[[reentrenar.media_dauc|4]]**, desviación **[[reentrenar.sd_dauc|4]]** e IC **[[reentrenar.ic_dauc.0|4]]; [[reentrenar.ic_dauc.1|4]]**; el amplio gana en el **100 %** de las particiones. Los anchos son parecidos (**[[reentrenar.ancho_rep|3]]** y **[[reentrenar.ancho_boot|3]]**), pero **responden preguntas distintas**: un intervalo puede ser correcto para una y no responder la otra. Ninguno corrige sesgos de diseño ni la fuga de datos, ni cambios futuros de población o prevalencia.

### ✍️ Ejercicio D3
Con los resultados de D3 calcula la fracción de particiones donde el **log loss** del amplio fue menor que el del simple y el IC de Δ log loss.
""")
code("""
# tu código aquí
""")
solution("""
print("amplio con menor log loss en", np.mean(dll > 0), "de las particiones | IC de Δ log loss:", q(dll).round(4))
""", "Sale ≈ 99.7 % y un IC positivo (≈ [0.04; 0.17]): coherente con el bootstrap de la prueba.")

md("""
### D4 · El tamaño de la prueba (10 min)
El ancho del IC baja como **1/√n**: para reducir el intervalo a la mitad se necesitan 4 veces más casos. Se compara el ancho normal 2·1.96·√(p(1 − p)/n) con el del bootstrap para un clasificador con acierto real 0.90.
""")
code("""
rng = np.random.default_rng(43008)
p0 = 0.9
filas = []
for nn in (25, 100, 171, 1000):
    wb = []
    for _ in range(10):
        x = rng.binomial(1, p0, nn)
        bm = np.array([rng.choice(x, nn, replace=True).mean() for _ in range(300)])
        wb.append(np.diff(q(bm))[0])
    filas.append(dict(n=nn, ancho_normal=2 * 1.96 * np.sqrt(p0 * (1 - p0) / nn), ancho_bootstrap=np.mean(wb)))
display(pd.DataFrame(filas).round(4))
""")
md("""
Con 100 casos el ancho normal es **[[tamano.ancho_normal.2|3]]** y con 1 000, **[[tamano.ancho_normal.5|3]]**: una reducción de ≈ √10. El bootstrap sigue la misma tendencia. Más **remuestras** (B) solo estabilizan el cálculo; más **casos de prueba** reducen la incertidumbre.

### ✍️ Ejercicio D4
¿Cuántos casos de prueba hacen falta para que el IC normal de una exactitud de 0.90 tenga ancho 0.05 (aproximadamente)?
""")
code("""
# tu código aquí
""")
solution("""
ancho = 0.05
n_nec = (2 * 1.96) ** 2 * 0.9 * 0.1 / ancho ** 2
print("n necesario ≈", int(np.ceil(n_nec)))
""", "n ≈ (2·1.96)² · 0.09 / 0.05² ≈ **554** casos.")

md("""
### D5 · Alertas del material, verificadas con código (10 min)
Se revisan afirmaciones de la clase y las diapositivas.
""")
code("""
# 1) «si los intervalos se cruzan no hay diferencia»: los IC de cada AUC se solapan, pero el IC de la diferencia pareada no contiene 0
print("1) solapan:", bool(ic_auc[0, 1] < ic_auc[1, 0]), "| IC ΔAUC pareado:", q(delta_auc).round(4), "| contiene 0:", bool(q(delta_auc)[0] <= 0 <= q(delta_auc)[1]))
# 2) «con log loss gana el simple»: el signo de Δ log loss (simple − amplio) es positivo, favorece al amplio
print("2) Δ log loss (simple − amplio):", round(comparacion.LogLoss[0] - comparacion.LogLoss[1], 4), "| IC:", q(delta_logloss).round(4))
# 3) cifras de la transcripción: RMSE, log loss del ejemplo y tamaño de la prueba
print("3) RMSE:", round(float(np.sqrt(np.mean(errores ** 2))), 4), "| log loss del ejemplo:", round(float(-np.mean(np.log([0.9, 0.6, 0.35, 0.8]))), 4), "| casos de prueba:", len(prueba))
# 4) los pliegues no tienen todos 80 casos
print("4) tamaño de los pliegues:", entrenamiento.fold.value_counts().sort_index().tolist())
# 5) SE exacto del bootstrap de la media frente al de R (0.966) y al de Python
print("5) SE exacto:", round(float(todas.std()), 4), "| Python:", round(float(np.std(medias_boot, ddof=1)), 4), "| R: 0.966")
# 6) bic de statsmodels: bic_llf coincide con BIC() de R; bic_deviance no
print("6) bic_llf:", round(modelos["Amplio"].bic_llf, 2), "| bic_deviance:", round(modelos["Amplio"].bic_deviance, 2))
""")
md("""
**Lo dicho en clase (transcripción)**
- **«Si los intervalos se cruzan, no hay diferencia entre los modelos»** (y el ejemplo de las encuestas): es **incorrecto en general**. Aquí los IC de la AUC se solapan y la diferencia pareada excluye 0; sin emparejar, el intervalo sí contendría 0. El profesor muestra después que ΔAUC es significativa.
- **«Con log loss gana el modelo simple»**: el signo está al revés. Δ = simple − amplio es **positivo** y su IC excluye 0, a favor del **amplio** (diapositiva 49); con AUC también gana el amplio.
- **«No se pueden usar bootstrap y validación cruzada»**: es una imprecisión; se pueden combinar (bootstrap de predicciones OOF, .632, validación anidada) con cuidado. Lo que no debe hacerse es reajustar o elegir con la prueba.
- **«Nivel de confianza de 2 puntos»** (encuestas): son 2 puntos de **margen de error**; el nivel de confianza es 95 %.
- **Cifras mal transcritas (ya corregidas)**: RMSE 3.31 (no 3.81); log loss del ejemplo 0.4723 (no 0.46); B = 100 (no 1 000) para el SE 1.029; prueba de 171 (no 161 ni 179); 569 muestras (no 659).

**Diapositivas y scripts**: la diapositiva 13 dice «compras muy distintas entre sí» (error de tipeo: **muestras**); las diapositivas muestran los resultados de **R**; el script en R lee un archivo con ruta absoluta (`D:/Aulas_Diplomado/…`), hay que cambiarla; el script de Python no escribe archivos (usa `plt.show()`). El material **no** cubre McNemar ni DeLong (extensiones de esta guía) ni estudia la variación por reentrenar.

### D6 · Tabla de correspondencia R ↔ Python (10 min)
| Qué | R | Python |
|---|---|---|
| Leer el CSV | `read.csv()` | `pd.read_csv()` |
| Filtrar entrenamiento y prueba | `subset(datos, conjunto == "entrenamiento")` | `datos[datos["conjunto"] == "entrenamiento"]` |
| Regresión logística | `glm(y ~ x, data, family = binomial)` | `smf.glm("y ~ x", data, family=sm.families.Binomial()).fit()` |
| Probabilidades de prueba | `predict(modelo, newdata, type = "response")` | `modelo.predict(prueba)` |
| AIC | `AIC(m1, m2, m3)` | `modelo.aic` |
| BIC | `BIC(m1, m2, m3)` | `modelo.bic_llf` |
| Sortear con reposición | `sample(x, size, replace = TRUE)` | `rng.choice(x, size=…, replace=True)` |
| Semilla | `set.seed(20261012)` | `rng = np.random.default_rng(20261012)` |
| Cuantiles | `quantile(x, c(0.025, 0.975))` | `np.quantile(x, [0.025, 0.975])` |
| Desviación (n − 1) | `sd(x)` | `np.std(x, ddof=1)` |
| AUC | `pROC::roc()` y `auc()` | `roc_auc_score(y, p)` |
| Log loss | función propia con recorte | `log_loss(y, p)` |
| Por columnas | `apply(M, 2, quantile, probs = …)` | `np.quantile(M, [...], axis=0)` |
| Histograma | `hist()`, `abline()` | `plt.hist()`, `plt.axvline()` |

**Diferencias que importan**: (1) R y Python tienen generadores aleatorios distintos: con la misma semilla las remuestras difieren y los límites cambian algo (variación Monte Carlo); (2) en `statsmodels` el BIC comparable con R es `bic_llf`; (3) `np.std` usa `ddof=0` por defecto; el profesor usa `ddof=1` como `sd()` de R.

## Resumen operativo
1. Define la métrica y la unidad de remuestreo **antes**; selecciona con entrenamiento y validación (AIC, BIC, OOF) y evalúa **una vez** en prueba.
2. Con **modelos fijos**, remuestrea los casos de prueba (log loss por filas, AUC por clase) y usa percentiles (B = 2 000).
3. Para **comparar**, usa los mismos índices en ambos modelos y mira la **diferencia**; solapar IC no basta; excluir 0 no es lo mismo que una mejora relevante.
4. El intervalo **no** cubre el reentrenamiento, la fuga de datos ni los cambios de población; con más casos baja como 1/√n.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes decidir si **vale la pena pasar del modelo intermedio** (`radio + textura + suavidad`) **al amplio** (los cuatro predictores). Los dos se ajustan con los 398 casos de entrenamiento; la prueba (171) se usa **una sola vez**.

1. **Plan previo**: escribe en un comentario qué métrica principal usarás, cómo medirás la incertidumbre y qué mejora consideras relevante, **antes** de calcular nada.
2. **Estimaciones puntuales** en prueba: AUC, log loss y exactitud (umbral 0.5) de los dos modelos.
3. **Bootstrap pareado** (B = 2 000, `default_rng(43020)`): en cada iteración remuestrea por clase (benignos y malignos) para ΔAUC = amplio − intermedio y por filas (un segundo vector de índices) para Δ log loss = intermedio − amplio. Entrega los IC percentiles.
4. **McNemar** con la tabla de aciertos a umbral 0.5.
5. **Conclusión sin código**: ¿hay evidencia clara de superioridad? ¿Las métricas coinciden? ¿Qué incertidumbre **no** cubre el intervalo?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1) plan: métrica principal = AUC (decidida antes); IC percentil de la diferencia pareada; mejora relevante fijada = 0.01 de AUC
p_int = modelos["Intermedio"].predict(prueba).to_numpy()
print("AUC  :", round(roc_auc_score(y, p_int), 4), round(roc_auc_score(y, p_amplio), 4))
print("logL :", round(log_loss(y, p_int), 4), round(log_loss(y, p_amplio), 4))
print("exact:", round(np.mean((p_int >= 0.5) == y), 4), round(np.mean((p_amplio >= 0.5) == y), 4))
rng = np.random.default_rng(43020)
dauc_c, dll_c = np.empty(B), np.empty(B)
for b in range(B):
    i = np.concatenate([rng.choice(negativos, len(negativos), replace=True), rng.choice(positivos, len(positivos), replace=True)])
    dauc_c[b] = roc_auc_score(y[i], p_amplio[i]) - roc_auc_score(y[i], p_int[i])
    j = rng.choice(n, size=n, replace=True)
    dll_c[b] = log_loss(y[j], p_int[j]) - log_loss(y[j], p_amplio[j])
print("ΔAUC :", round(roc_auc_score(y, p_amplio) - roc_auc_score(y, p_int), 4), "IC", q(dauc_c).round(4))
print("Δ logL:", round(log_loss(y, p_int) - log_loss(y, p_amplio), 4), "IC", q(dll_c).round(4))
ok_i, ok_a = (p_int >= 0.5) == y, (p_amplio >= 0.5) == y
b, c = int(np.sum(ok_i & ~ok_a)), int(np.sum(~ok_i & ok_a))
print("McNemar: b =", b, "c =", c, "| p exacto =", round(float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), 4))
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: AUC intermedio [[cierre.auc_i|4]] y amplio [[cierre.auc_a|4]] (ΔAUC = [[cierre.dauc|4]], IC [[cierre.ic_dauc.0|4]]; [[cierre.ic_dauc.1|4]], **contiene 0**); log loss [[cierre.ll_i|4]] y [[cierre.ll_a|4]] (Δ = [[cierre.dll|4]], IC [[cierre.ic_dll.0|4]]; [[cierre.ic_dll.1|4]], que roza 0); exactitud [[cierre.acc_i|4]] y [[cierre.acc_a|4]]. McNemar: b = [[cierre.b]] y c = [[cierre.c]], p exacto = [[cierre.p_exacto|4]]. **Lectura**: con la métrica definida antes (AUC) **no hay evidencia clara** de superioridad (el IC contiene 0; tampoco prueba equivalencia). El log loss y la exactitud sí apuntan al amplio (el IC de Δ log loss apenas incluye 0 y McNemar da p ≈ 0.03), pero **cambiar de métrica después de ver los resultados** no es válido: la decisión depende de cuál se fijó antes y de si una mejora de 0.005 de AUC es relevante. Advertencias: (1) el intervalo cubre solo la variación de los casos de prueba con modelos fijos, no el reentrenamiento ni cambios de población; (2) la prueba ya se revisó antes: es una demostración, no una validación externa nueva.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué una exactitud de 0.90 en 100 casos no es «la» exactitud del modelo? ¿Cómo se calcula su error estándar aproximado?
2. ¿Qué diferencia hay entre la desviación estándar de los datos y el error estándar de una estimación?
3. ¿Qué significa un intervalo de confianza del 95 %? ¿Deben contener el parámetro todos los intervalos?
4. Describe una remuestra bootstrap: ¿de qué tamaño, con o sin reposición, y qué fracción de casos distintos contiene?
5. ¿Cómo se construye el IC percentil? ¿Qué cambia si aumentas B? ¿Y si aumentas el número de casos de prueba?
6. ¿Qué se remuestrea y qué se mantiene fijo al evaluar dos modelos ya ajustados? ¿Por qué no se repite `fit` en el ciclo?
7. ¿Por qué la AUC se remuestrea por clase y el log loss por filas en el script del profesor?
8. ¿Qué son AIC y BIC, qué cuenta p, y cuándo no se pueden comparar?
9. ¿Por qué la receta (estandarizar, elegir variables, fijar el umbral) va dentro del pliegue?
10. ¿Por qué se comparan los modelos con la diferencia pareada? ¿Por qué solapar los IC de cada modelo no basta?
11. Si el IC de la diferencia es [−0.004; 0.024], ¿qué concluyes? ¿Y si es [0.001; 0.003] y la mejora relevante es 0.010?
12. ¿Qué incertidumbre cubre el IC de la prueba y cuál no? ¿Qué es McNemar y cuándo se usa?

<details><summary><b>Respuestas</b></summary>

1. Porque es una estimación con otros 100 casos podría salir otro valor; SE ≈ √(â(1 − â)/n) = √(0.09/100) = 0.030.
2. La primera describe cuánto difieren los casos entre sí; la segunda, cuánto varía la estimación (media, exactitud, log loss) al cambiar la muestra.
3. Es la cobertura del procedimiento: en repetición, ≈ 95 % de los intervalos contiene el parámetro. No todos deben hacerlo (en la diapositiva, 38 de 40).
4. Del mismo tamaño que la muestra, con reposición; contiene ≈ 63.2 % de casos distintos y deja ≈ 36.8 % fuera.
5. Con los percentiles 2.5 y 97.5 de las B estadísticas. Más B estabiliza el cálculo (menos error Monte Carlo); más casos de prueba reducen la incertidumbre real.
6. Se remuestrean los casos de prueba con sus probabilidades; los modelos (coeficientes) quedan fijos. Repetir `fit` mediría otra incertidumbre (la del entrenamiento).
7. Para que la AUC siempre tenga las dos clases y conserve 107 y 64; el log loss se puede remuestrear por filas porque no necesita las dos clases.
8. AIC = −2ℓ + 2p y BIC = −2ℓ + p ln n (p incluye el intercepto). Requieren misma respuesta, mismas observaciones, verosimilitudes comparables y máxima verosimilitud; no sirven para SVM, Isolation Forest ni logísticas penalizadas contando coeficientes.
9. Para que la media, la escala, las variables y el umbral se calculen solo con las filas de ajuste; si no, se filtra información de validación.
10. Porque ambos modelos ven los mismos casos y el emparejamiento conserva la dificultad compartida; la diferencia tiene su propio IC y suele ser más angosto. Los IC separados pueden solaparse aunque la diferencia excluya 0.
11. [−0.004; 0.024]: no hay evidencia clara de superioridad (no prueba equivalencia). [0.001; 0.003]: positiva pero menor que la mejora relevante.
12. Cubre la variación al remuestrear casos de evaluación con modelos fijos; no cubre reentrenar, cambios de población, sesgos de diseño ni fuga. McNemar compara exactitudes de dos clasificadores en los mismos casos usando los discordantes.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** repaso final con el profesor. Revisa `Guias/apuntes/C43_Ejercicios_Cuaderno_EvaluacionBootstrap.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
