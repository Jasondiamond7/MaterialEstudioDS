"""Genera el notebook guiado de Naive Bayes y clasificación probabilística (Clase 40).

Uso normal:      python notebook_c40.py             -> Guias/notebooks/C40_NaiveBayes.ipynb
Modo de prueba:  C40_TEST=1 python notebook_c40.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c40_numeros.json (lo escribe figuras_c40.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib y scikit-learn (no necesita paquetes adicionales).
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C40_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C40_NaiveBayes_test.ipynb" if TEST else "C40_NaiveBayes.ipynb")
N = json.loads((HERE / "c40_numeros.json").read_text(encoding="utf-8"))
_fr = N["fronteras"]["resultados"]
N["fr"] = {"nb_acc": _fr["Naive Bayes gaussiano"]["accuracy"], "nb_ll": _fr["Naive Bayes gaussiano"]["log_loss"], "lda_acc": _fr["LDA (covarianza común)"]["accuracy"],
           "qda_acc": _fr["QDA (covarianza por clase)"]["accuracy"], "lg_acc": _fr["Regresión logística"]["accuracy"], "lg_ll": _fr["Regresión logística"]["log_loss"]}
_oh = N["onehot"]["prueba"]
N["oh"] = {"cat_auc": _oh["CategoricalNB"]["auc"], "cat_ll": _oh["CategoricalNB"]["log_loss"], "ber_ll": _oh["BernoulliNB (dummies)"]["log_loss"], "gau_ll": _oh["GaussianNB (dummies)"]["log_loss"]}
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
# Clase 40 · Naive Bayes y clasificación probabilística
**Diplomado Data Science** · Prof. Eloy Alvarado · usa Bayes y la previa (Clase 23), retoma la regresión logística y los clasificadores (Clases 31 y 37) y la evaluación probabilística de la Clase 38 (OOF, log loss, Brier, calibración, umbral por costo, el mismo spam) y apoya la pregunta 2 del Taller 3

Material de apoyo:
- `Guias/C40_Guia_NaiveBayes.pdf` — guía visual de la clase.
- `Guias/apuntes/C40_Apuntes_NaiveBayes.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C40_Ejercicios_Cuaderno_NaiveBayes.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta «Clase 40 …»): `DipDS_2026_S40.pdf` (diapositivas), `40_naive_bayes.py` y `.R.txt`, y `spam_s40.csv`.
- Transcripción: `VideosClases/Clase 40 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib` y `scikit-learn` (no hace falta nada más). Si no los tienes:
```
pip install numpy pandas matplotlib scikit-learn
```

**Datos**: `spam_s40.csv` (4 601 correos de Spambase; 54 frecuencias de palabras y caracteres, más 3 medidas de mayúsculas que **no** se usan) con las columnas `conjunto` (entrenamiento o prueba), `grupo` (filas que deben ir juntas) y `fold` (pliegue de validación cruzada): **la partición del profesor ya está hecha** y se usa tal cual. Es el mismo spam de la Clase 38. El notebook busca el archivo en la carpeta de la Clase 40 **sin modificarla**. Nada de lo que se hace aquí usa los datos del Taller 3.

### Objetivos
1. Distinguir **previa, condicional y posterior** y calcular P(spam | señales) a mano con Bayes.
2. Entender el supuesto de **independencia condicional**, el **suavizado de Laplace** y por qué se **suman logaritmos**.
3. **Reproducir** los resultados del profesor: log loss OOF 0.671, 0.556, 0.557, 0.558 y 4.931; umbral 0.95 con costo 219.0 por 1000; prueba histórica con 17 FP, 89 FN y costo 188.9.
4. Comparar Naive Bayes con la **regresión logística**: ordenan parecido, pero las probabilidades de Naive Bayes son menos fiables (calibración).
5. Ver qué ocurre cuando se **repiten señales** y cuando se aplica un **gaussiano a columnas 0/1**.
6. Construir el flujo completo con **Pipelines** ajustados solo con los datos de entrenamiento y decidir con un **umbral por costo**.
7. Traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Previa, condicional y posterior (100 correos) |
| A | 15 | A2 · Dos señales e independencia |
| A | 10 | A3 · Suavizado de Laplace |
| A | 10 | A4 · Duplicar una señal |
| A | 10 | A5 · Multinomial: un ejemplo con conteos |
| B | 10 | B1 · Los datos y la partición |
| B | 10 | B2 · Presencia o frecuencia |
| B | 20 | B3 · Pipelines y predicciones fuera de pliegue |
| B | 15 | B4 · Reproducir la comparación del profesor |
| C | 15 | C1 · Explicar un correo con log-odds |
| C | 10 | C2 · ¿Por qué logaritmos? |
| C | 15 | C3 · Señales repetidas con los correos reales |
| C | 20 | C4 · Calibración: Naive Bayes frente a la logística |
| C | 15 | C5 · Fronteras: Naive Bayes, LDA, QDA y logística |
| D | 15 | D1 · Gaussiana y log(1 + x) |
| D | 25 | D2 · Gaussiano sobre columnas 0/1 (binarias y one-hot) |
| D | 20 | D3 · Umbral por costo y prueba histórica |
| D | 10 | D4 · Alertas del material, verificadas con código |
| D | 10 | D5 · Tabla de correspondencia R ↔ Python |
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 2026   # semilla para todo lo aleatorio
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso (Clases 23, 31, 37 y 38)
- **Bayes** (Clase 23): P(A | B) = P(B | A) P(A) / P(B). La **previa** es la frecuencia de la clase; la **condicional**, la frecuencia de una señal *dentro* de una clase; la **posterior**, la probabilidad de la clase al ver las señales.
- **Regresión logística** (Clases 31 y 37): modela P(clase 1 | x) directamente; es el clasificador de referencia.
- **OOF, log loss, Brier, calibración y umbral por costo** (Clase 38): una predicción *fuera de pliegue* no participó en el ajuste que la produjo; el **log loss** castiga la certeza equivocada; la **calibración** compara probabilidad predicha con frecuencia observada; el **umbral** sale de los costos de FP y FN.
- **Pipeline**: todo el preprocesamiento (escalar, transformar) vive dentro del modelo y se ajusta **solo con los datos de entrenamiento** de cada pliegue; así no se filtra información de validación.

### Funciones comunes
""")
code("""
def buscar_clase():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 40*/spam_s40.csv"):
            return p.parent
        if (base / "spam_s40.csv").exists():
            return base
    raise FileNotFoundError("No encuentro spam_s40.csv: abre el notebook dentro de la carpeta del curso.")


CARPETA = buscar_clase()


def log_probs(m, x):
    # log-probabilidades estables: desde decision_function (logística) o predict_log_proba (Naive Bayes)
    if hasattr(m, "decision_function"):
        lo = m.decision_function(x)
        return np.column_stack([-np.logaddexp(0, lo), -np.logaddexp(0, -lo)])
    if hasattr(m, "predict_log_proba"):
        return m.predict_log_proba(x)
    q = np.clip(m.predict_proba(x), 1e-15, 1)       # modelos sin predict_log_proba (p. ej. CalibratedClassifierCV)
    return np.log(q)


def oof(modelo, X, y, folds):
    # predicciones fuera de pliegue: cada correo se predice con un modelo que no lo vio (el Pipeline se reajusta en cada pliegue)
    p = np.zeros(len(y))
    lp = np.zeros((len(y), 2))
    for k in np.unique(folds):
        v = folds == k
        m = clone(modelo).fit(X[~v], y[~v])
        lp[v] = log_probs(m, X[v])
        p[v] = np.exp(lp[v, 1])
    return p, lp


def medir(y, p, lp=None, t=0.5):
    # las mismas medidas que `medir` del script del profesor (log loss y AUC desde las log-probabilidades)
    q = np.clip(p, 1e-15, 1 - 1e-15)
    if lp is None:
        lp = np.column_stack([np.log1p(-q), np.log(q)])
    perdida = -np.maximum(lp[np.arange(len(y)), y], np.log(1e-15)).mean()
    pred = p >= t
    fp, fn = int(np.sum((y == 0) & pred)), int(np.sum((y == 1) & ~pred))
    return dict(log_loss=perdida, brier=np.mean((y - p) ** 2), auc=roc_auc_score(y, lp[:, 1] - lp[:, 0]), accuracy=np.mean(pred == y),
                VN=int(np.sum((y == 0) & ~pred)), FP=fp, FN=fn, VP=int(np.sum((y == 1) & pred)), costo_1000=1000 * (5 * fp + fn) / len(y))


print("Material:", CARPETA.name)
""")

# ================================================================== PARTE A
md("""
## Parte A · Bayes y Naive Bayes a mano

### A1 · Previa, condicional y posterior (15 min)
Diapositivas 4 a 10: **100 correos construidos** (no son datos de Spambase): 80 legítimos y 20 spam; la palabra «oferta» aparece en 8 legítimos y en 16 spam. Hay tres preguntas distintas: la **previa** P(spam), la **condicional** P(oferta | spam) y la **posterior** P(spam | oferta).
""")
code("""
n_leg, n_spam = 80, 20
of_leg, of_spam = 8, 16
previa = n_spam / (n_leg + n_spam)
p_of_spam, p_of_leg = of_spam / n_spam, of_leg / n_leg
print("previa P(spam)       :", previa)
print("P(oferta | spam)     :", p_of_spam, "  <- se mira la FILA de spam")
print("P(oferta | legítimo) :", p_of_leg)
print("P(spam | oferta)     :", of_spam / (of_spam + of_leg), "  <- se mira la COLUMNA de oferta")

def posterior_una_senal(prev):
    s1, s0 = prev * p_of_spam, (1 - prev) * p_of_leg      # producto de cada clase
    return s1 / (s0 + s1)                                  # se normaliza

for prev in (0.20, 0.02):
    print(f"previa {prev:.2f} -> P(spam | oferta) = {posterior_una_senal(prev):.4f}")
""")
md("""
Coincide con las diapositivas 6 a 9: con 20 % de spam, P(spam | oferta) = **[[bayes.post_oferta|4]]**; con 2 %, **[[bayes.post_prior_002|4]]**. La condicional (0.80) no cambia, pero la previa sí: **la misma palabra no decide por sí sola**. Y P(oferta | spam) = 0.80 **no** es P(spam | oferta) = 0.667.

### ✍️ Ejercicio A1
En 500 correos hay 400 legítimos y 100 spam. La palabra «gratis» está en 20 legítimos y en 60 spam. Calcula P(spam | gratis) y repite con una previa de spam del 5 % (mismas condicionales).
""")
code("""
# tu código aquí
""")
solution("""
n0, n1, c0, c1 = 400, 100, 20, 60
t1, t0 = c1 / n1, c0 / n0
for prev in (n1 / (n0 + n1), 0.05):
    s1, s0 = prev * t1, (1 - prev) * t0
    print(f"previa {prev:.2f}: P(spam | gratis) = {s1 / (s0 + s1):.4f}")
""", "Con 20 % de spam: 0.12/(0.12 + 0.04) = **0.75**. Con 5 %: 0.03/(0.03 + 0.0475) = **0.387**. La condicional es la misma (0.60 y 0.05); cambia la previa.")

md("""
### A2 · Dos señales e independencia (15 min)
Se agrega «enlace»: 16 de los 80 legítimos y 14 de los 20 spam lo traen. Queremos clasificar un correo con **oferta presente y enlace ausente**. No conocemos cómo se combinan las dos señales dentro de cada clase (la tabla conjunta): **Naive Bayes supone que son independientes dentro de la clase** y multiplica un factor por señal.
""")
code("""
# factores dentro de cada clase para x = (oferta presente, enlace ausente)
f_leg = (8 / 80) * (1 - 16 / 80)
f_spam = (16 / 20) * (1 - 14 / 20)
s0 = (80 / 100) * f_leg        # producto del legítimo
s1 = (20 / 100) * f_spam       # producto del spam
print(f"P(oferta, sin enlace | spam) = {f_spam:.2f} ; | legítimo = {f_leg:.2f}")
print(f"s0 = {s0:.3f}, s1 = {s1:.3f}  ->  P(spam | oferta, sin enlace) = {s1 / (s0 + s1):.4f}")

# los otros dos casos de la diapositiva 18
for nombre, xo, xe in [("oferta y enlace", 1, 1), ("solo oferta (sin mirar el enlace)", 1, None)]:
    a1 = 0.2 * (16 / 20) * ((14 / 20) if xe == 1 else 1)
    a0 = 0.8 * (8 / 80) * ((16 / 80) if xe == 1 else 1)
    print(f"{nombre}: {a1 / (a0 + a1):.4f}")
""")
md("""
Resultados de las diapositivas 17 y 18: **[[bayes.post_oferta_sin_enlace|4]]** (oferta sin enlace), **[[bayes.post_oferta_enlace|4]]** (oferta y enlace) y 0.6667 (solo oferta). Mirar una segunda señal cambia la probabilidad. Con 2 señales hay 4 patrones; con 10, **1 024**; con 54, 2^54: sin independencia habría que contar cada patrón y muchos no aparecerían nunca.

Comprobación con `BernoulliNB`: como solo usa las **frecuencias marginales**, basta construir cualquier tabla de 100 correos con esos conteos.
""")
code("""
# 100 filas (oferta, enlace) cuyas marginales coinciden con la tabla; la combinación exacta no importa para BernoulliNB
y_toy = np.array([0] * 80 + [1] * 20)
oferta = np.array([1] * 8 + [0] * 72 + [1] * 16 + [0] * 4)
enlace = np.array([1] * 16 + [0] * 64 + [1] * 14 + [0] * 6)
X_toy = np.column_stack([oferta, enlace])
for alfa in (1e-10, 1.0):                  # sklearn sustituye alpha = 0 por 1e-10
    nb = BernoulliNB(alpha=alfa).fit(X_toy, y_toy)
    print(f"alpha = {alfa:g}: P(spam | oferta, sin enlace) =", round(float(nb.predict_proba(np.array([[1, 0]]))[0, 1]), 4))
""")
md("""
Con `alpha → 0` da 0.4286 y con `alpha = 1` da 0.4140 (diapositivas 17 y 22). Nota que `BernoulliNB` también multiplica **las ausencias**, no solo las presencias.

### ✍️ Ejercicio A2
Con las mismas tablas, calcula la probabilidad de spam de un correo **sin oferta y con enlace** (sin suavizar). ¿Es más o menos probable que spam que el caso «oferta sin enlace»?
""")
code("""
# tu código aquí
""")
solution("""
a1 = 0.2 * (1 - 16 / 20) * (14 / 20)
a0 = 0.8 * (1 - 8 / 80) * (16 / 80)
print(round(a1 / (a0 + a1), 4), "| con BernoulliNB:", round(float(BernoulliNB(alpha=1e-10).fit(X_toy, y_toy).predict_proba(np.array([[0, 1]]))[0, 1]), 4))
""", "0.2 × 0.2 × 0.7 = 0.028 (spam) y 0.8 × 0.9 × 0.2 = 0.144 (legítimo): P(spam) = **0.163**. Es mucho menos probable que spam que el caso «oferta sin enlace» (0.429).")

md("""
### A3 · Suavizado de Laplace (10 min)
Si una señal nunca aparece en una clase, su frecuencia es 0 y **anula todo el producto**, aunque las demás señales apunten a esa clase. Observar cero casos no demuestra que sea imposible. El **suavizado de Laplace** suma `α` a «presente» y `α` a «ausente»: `θ̂ = (conteo + α)/(n + 2α)`.
""")
code("""
# «premio»: 0 de los 20 spam
for alfa in (0, 1, 0.5):
    print(f"alpha = {alfa}: P(premio | spam) = {(0 + alfa) / (20 + 2 * alfa):.4f}")

# el ejemplo de 100 correos con alpha = 1 (diapositivas 21 y 22)
s0 = 0.8 * (9 / 82) * (65 / 82)
s1 = 0.2 * (17 / 22) * (7 / 22)
print(f"s0 = {s0:.4f}, s1 = {s1:.4f}  ->  P(spam) = {s1 / (s0 + s1):.4f}")
""")
md("""
«premio» pasa de 0 a 1/22 = **[[laplace.premio|4]]** y la probabilidad del ejemplo, de 0.4286 a **[[laplace.post_alpha1|4]]**. Observa dos cosas: (1) el suavizado **no corrige la dependencia entre señales**; (2) con 3 680 correos y 54 señales **ninguna celda tiene cero casos** ([[laplace.celdas_cero]] celdas en cero), así que el log loss casi no cambia con α (lo verás en B4).

### ✍️ Ejercicio A3
En los 20 spam del ejemplo, «urgente» aparece 0 veces y en los 80 legítimos, 2 veces. Calcula P(urgente | spam) y P(urgente | legítimo) con α = 1 y α = 0.5. ¿En cuál de las dos clases cambia más la estimación?
""")
code("""
# tu código aquí
""")
solution("""
for alfa in (1, 0.5):
    print(alfa, "spam:", round((0 + alfa) / (20 + 2 * alfa), 4), "| legítimo:", round((2 + alfa) / (80 + 2 * alfa), 4))
""", "α = 1: 0.0455 en spam (desde 0) y 0.0366 en legítimos (desde 0.025). α = 0.5: 0.0238 y 0.0309. Cambia más en la clase con el cero y con menos correos.")

md("""
### A4 · Duplicar una señal (10 min)
Diapositivas 46 y 47: 100 spam y 100 legítimos; una señal aparece en 80 spam y 20 legítimos. Con **una** copia P(spam | señal) = 0.80. Si la **misma** columna se copia, Naive Bayes la multiplica de nuevo y la certeza crece, aunque no hay información nueva.
""")
code("""
copias = np.array([1, 2, 4])
s1 = 0.5 * 0.8 ** copias
s0 = 0.5 * 0.2 ** copias
print("copias:", copias, "-> P(clase 1) =", np.round(s1 / (s1 + s0), 4), "| información disponible: 0.8")
""")
md("""
La probabilidad sube de 0.8000 a **[[dup.post.1|4]]** (2 copias) y **[[dup.post.2|4]]** (4 copias). En una regresión, dos predictores idénticos no agregan información; aquí **sí aumentan la certeza**. Es **sobreconfianza por dependencia**, no «sobreajuste» (que sería ajustar el ruido de la muestra).

### ✍️ Ejercicio A4
Una señal aparece en 60 de 100 spam y en 15 de 100 legítimos (previas iguales). Calcula la probabilidad con 1, 2 y 3 copias.
""")
code("""
# tu código aquí
""")
solution("""
for k in (1, 2, 3):
    s1, s0 = 0.5 * 0.6 ** k, 0.5 * 0.15 ** k
    print(k, round(s1 / (s0 + s1), 4))
""", "0.8000, 0.9412 y 0.9846: la información real es siempre la misma señal.")

md("""
### A5 · Multinomial: un ejemplo con conteos (10 min)
Diapositiva 39: con un vocabulario fijo (oferta, reunión, hoy), el mensaje «oferta hoy» se codifica (1, 0, 1) y «oferta oferta hoy», (2, 0, 1). **Bernoulli** ve el mismo vector (1, 0, 1) en los dos; **multinomial** cuenta cada aparición y las repeticiones cambian el cálculo. En los datos de spam las frecuencias vienen en porcentaje, así que **no** hay conteos y no se usa la multinomial.
""")
code("""
vocabulario = ["oferta", "reunión", "hoy"]
cuentas = np.array([[0., 4., 2.],     # palabras contadas en la clase 0
                    [4., 0., 2.]])    # palabras contadas en la clase 1
mensajes = np.array([[1., 0., 1.], [2., 0., 1.]])
mn = MultinomialNB(alpha=1.0, class_prior=[0.5, 0.5]).fit(cuentas, [0, 1])
print("P(clase 1) con MultinomialNB:", np.round(mn.predict_proba(mensajes)[:, 1], 4))
theta = (cuentas + 1) / (cuentas.sum(axis=1, keepdims=True) + 3)       # el cálculo a mano del script del profesor
log_conjunta = np.log(0.5) + mensajes @ np.log(theta).T
print("a mano:", np.round(np.exp(log_conjunta[:, 1]) / np.exp(log_conjunta).sum(axis=1), 4))
""")
md("""
Resultados del script del profesor: **[[variantes.posterior_multinomial.0|4]]** y **[[variantes.posterior_multinomial.1|4]]**. Repetir «oferta» sube la probabilidad de la clase 1 solo en la variante multinomial.

**Qué variante usar** (diapositiva 40): Bernoulli → presencias 0/1; gaussiana → mediciones numéricas (por ejemplo log(1 + x)); multinomial → conteos de términos (cuántas veces aparece cada palabra).

## Parte B · Spam real: Pipelines, OOF y la comparación del profesor

### B1 · Los datos y la partición (10 min)
`spam_s40.csv` trae `conjunto` (3 680 de desarrollo y 921 de prueba histórica), `grupo` (las filas con las mismas 57 entradas originales van juntas) y `fold` (los 5 pliegues de validación cruzada del desarrollo). **No** son entradas del modelo: `id`, `grupo`, `fold`, `conjunto`, `spam` ni las tres medidas de mayúsculas.
""")
code("""
datos = pd.read_csv(CARPETA / "spam_s40.csv")
vs = [c for c in datos if c.startswith(("word_freq_", "char_freq_"))]
tr = datos[datos.conjunto == "entrenamiento"].reset_index(drop=True)
te = datos[datos.conjunto == "prueba"].reset_index(drop=True)
assert len(vs) == 54 and len(tr) == 3680 and len(te) == 921 and set(tr.grupo).isdisjoint(te.grupo)
assert tr.groupby("grupo").fold.nunique().max() == 1
X, y, folds = tr[vs].to_numpy(), tr.spam.to_numpy(), tr.fold.to_numpy()
Xte, yte = te[vs].to_numpy(), te.spam.to_numpy()
print("entrenamiento por clase:", np.bincount(y), f"-> spam = {y.mean():.1%}")
print("prueba por clase:       ", np.bincount(yte), f"-> spam = {yte.mean():.1%}")
print("correos por pliegue:", np.bincount(folds)[1:])
display(datos.groupby(["conjunto", "spam"]).size().rename("n").reset_index())
""")
md("""
Hay **39.4 % de spam** en el desarrollo (1 450 de 3 680): algo desbalanceado, pero no extremo. Los costos (FP = 5 y FN = 1) son una razón **arbitraria** pensada para el ejemplo; en un caso real se estimaría con el negocio.

### B2 · Presencia o frecuencia (10 min)
Diapositivas 32 y 36 a 38. El correo con menor `id` (ID 1) tiene `free` = 0.32 (%), `money` = 0 y `!` = 0.778. **Bernoulli** solo conserva si la frecuencia es mayor que 0; la **gaussiana** usa `log(1 + x)`, que comprime los valores grandes y conserva los ceros.
""")
code("""
i1 = int(np.argmin(tr.id.to_numpy()))
for nom in ("word_freq_free", "word_freq_money", "char_freq_!"):
    v = tr.loc[i1, nom]
    print(f"{nom:16s} frecuencia {v:6.3f}  presencia {int(v > 0)}  log(1 + x) = {np.log1p(v):.4f}")
print("ID del correo:", tr.id.iloc[i1], "| spam =", tr.spam.iloc[i1])
""")
md("""
### ✍️ Ejercicio B2
¿En qué porcentaje de los correos de desarrollo aparece (frecuencia > 0) la palabra «free»? ¿Y en cada clase por separado?
""")
code("""
# tu código aquí
""")
solution("""
j = vs.index("word_freq_free")
pres = X[:, j] > 0
print("global:", round(pres.mean(), 4), "| legítimos:", round(pres[y == 0].mean(), 4), "| spam:", round(pres[y == 1].mean(), 4))
""", "Aparece en el 27.7 % de los correos: en el 9.6 % de los legítimos y el 55.6 % de los spam (la barra de «free» en la diapositiva 50). Es una señal informativa porque difiere mucho entre clases.")

md("""
### B3 · Pipelines y predicciones fuera de pliegue (20 min)
Se comparan **cinco recetas** (diapositiva 41): una referencia sin señales (la frecuencia de spam), Bernoulli con `α = 0.5, 1, 2` y la gaussiana sobre `log(1 + x)`. Cada receta es un **Pipeline**: la transformación va **dentro** del modelo y se ajusta solo con los pliegues de entrenamiento (`clone(modelo).fit(X[~v], y[~v])`). Así no se filtra información de validación; con la gaussiana, la media y la desviación de cada columna se aprenden únicamente en los pliegues de ajuste.
""")
code("""
from sklearn.dummy import DummyClassifier

def recetas():
    return {
        "FRECUENCIA": DummyClassifier(strategy="prior"),
        "BERN_05": BernoulliNB(alpha=0.5, binarize=0.0),
        "BERN_1": BernoulliNB(alpha=1.0, binarize=0.0),
        "BERN_2": BernoulliNB(alpha=2.0, binarize=0.0),
        "GAUSS_LOG": make_pipeline(FunctionTransformer(np.log1p), GaussianNB(var_smoothing=1e-9)),
    }

oof_p, oof_lp = {}, {}
for nombre, modelo in recetas().items():
    oof_p[nombre], oof_lp[nombre] = oof(modelo, X, y, folds)
print({k: v.shape for k, v in oof_p.items()})
""")
md("""
**Por qué un Pipeline y no `StandardScaler().fit_transform(X)` antes de dividir**: si el escalado se ajusta con todos los correos, la media y la desviación de los pliegues de validación ya «vieron» esas filas y la estimación del error sale optimista. Con la logística de la Clase 38 (que sí escala) se usa `make_pipeline(StandardScaler(), LogisticRegression())`: dentro de cada pliegue, el escalador se ajusta solo con 4 de los 5 pliegues.

### B4 · Reproducir la comparación del profesor (15 min)
Diapositiva 44: log loss, Brier y AUC OOF de las cinco recetas, en los mismos 3 680 correos.
""")
code("""
comparacion = pd.DataFrame([dict(modelo=k, **medir(y, oof_p[k], oof_lp[k])) for k in oof_p])
display(comparacion[["modelo", "log_loss", "brier", "auc", "accuracy", "costo_1000"]].round(4))
ganador = comparacion.loc[comparacion.log_loss.idxmin(), "modelo"]
print("Selección por log loss OOF:", ganador)
""")
md("""
Reproduce la diapositiva 44: log loss **0.671** (frecuencia), **[[comparacion.oof.BERN_05.log_loss|3]]** (Bernoulli α = 0.5), **[[comparacion.oof.BERN_1.log_loss|3]]**, **0.558** y **[[comparacion.oof.GAUSS_LOG.log_loss|3]]** (gaussiana); Brier 0.239, 0.099, 0.099, 0.100 y 0.190; AUC 0.499, [[comparacion.oof.BERN_05.auc|3]], 0.951, 0.951 y [[comparacion.oof.GAUSS_LOG.auc|3]]. **Gana Bernoulli con α = 0.5**, pero los tres suavizados dan casi lo mismo. La gaussiana acierta **80.8 %** con umbral 0.5, y aun así su log loss es nueve veces el del Bernoulli: **la exactitud no ve la certeza equivocada**.

La referencia es un **piso**: AUC 0.5 y log loss 0.671; los modelos con señales deben superarla con claridad. El profesor aclara que el mínimo OOF «intervino en la selección»: no es una evaluación independiente del ganador.

### ✍️ Ejercicio B4
Agrega las recetas `BERN_025` (α = 0.25) y `BERN_5` (α = 5) y calcula su log loss OOF. ¿Cambia el ganador? ¿Qué te dice sobre la importancia del suavizado con 3 680 correos?
""")
code("""
# tu código aquí
""")
solution("""
extra = {}
for nombre, a in [("BERN_025", 0.25), ("BERN_5", 5.0)]:
    p, lp = oof(BernoulliNB(alpha=a, binarize=0.0), X, y, folds)
    extra[nombre] = round(float(medir(y, p, lp)["log_loss"]), 4)
print(extra, "| BERN_05:", round(float(comparacion.set_index("modelo").loc["BERN_05", "log_loss"]), 4))
""", "El log loss pasa de 0.5563 (α = 0.5) a ≈ 0.5618 con α = 5 y a ≈ 0.5559 con α = 0.25: el ganador no cambia. Con muchos correos y señales no tan raras el suavizado casi no importa; sí importa con pocos datos o señales muy raras (como «premio»).")

md("""
## Parte C · Probabilidades, independencia y calibración

### C1 · Explicar un correo con log-odds (15 min)
Diapositivas 49 y 50. Con Bernoulli, el aporte de cada señal es `log(θ₁/θ₀)` si está presente y `log((1 − θ₁)/(1 − θ₀))` si está ausente. La previa y los 54 aportes **se suman** y la probabilidad es la función logística del total.
""")
code("""
mod = BernoulliNB(alpha=1.0, binarize=0.0).fit(X, y)
b = (X[i1] > 0).astype(float)
theta = np.exp(mod.feature_log_prob_)                      # fila 0: legítimo, fila 1: spam
aportes = b * np.log(theta[1] / theta[0]) + (1 - b) * np.log((1 - theta[1]) / (1 - theta[0]))
prior_lo = np.log(y.mean() / (1 - y.mean()))
total = prior_lo + aportes.sum()
print(f"log-odds previa = {prior_lo:.3f} | suma de aportes = {aportes.sum():.3f} | total = {total:.3f}")
print(f"P(spam) = 1/(1 + e^(-total)) = {1 / (1 + np.exp(-total)):.4f} | predict_proba = {mod.predict_proba(X[i1:i1 + 1])[0, 1]:.4f}")
j = vs.index("word_freq_address")
c0, c1 = ((X[:, j] > 0) == bool(b[j]))[y == 0].sum(), ((X[:, j] > 0) == bool(b[j]))[y == 1].sum()
print(f"address presente: legítimo {(c0 + 1) / (2230 + 2):.4f}, spam {(c1 + 1) / (1450 + 2):.4f} (diapositiva 49)")
top = pd.Series(aportes, index=vs).sort_values(key=np.abs, ascending=False).head(8)
display(top.round(3))
""")
md("""
La previa de log-odds es **[[bern.log_odds_prior|3]]**, los aportes suman **[[bern.suma_aportes|3]]** y el total [[bern.log_odds_total|3]] da **[[bern.posterior|4]]** (diapositiva 50: 0.9964). De las 54 señales, [[bern.n_aportes_positivos]] empujan hacia spam y [[bern.n_aportes_negativos]] hacia legítimo. Las barras de la diapositiva 50 son **factores**: no se suman ni representan efectos causales; lo que se suma son los logaritmos.

### ✍️ Ejercicio C1
¿Cuál es el correo de desarrollo con la **menor** probabilidad de spam según `mod`? Calcula su log-odds total y la señal que más lo empuja hacia legítimo.
""")
code("""
# tu código aquí
""")
solution("""
pr = mod.predict_proba(X)[:, 1]
i_min = int(np.argmin(pr))
bb = (X[i_min] > 0).astype(float)
ap = bb * np.log(theta[1] / theta[0]) + (1 - bb) * np.log((1 - theta[1]) / (1 - theta[0]))
print("id:", tr.id.iloc[i_min], "| spam real:", tr.spam.iloc[i_min], "| P(spam) =", pr[i_min], "| log-odds =", round(float(prior_lo + ap.sum()), 2))
print("señal más negativa:", vs[int(np.argmin(ap))], round(float(ap.min()), 3))
""", "Es el correo con id 3255 (legítimo): log-odds ≈ −47.5, es decir P(spam) ≈ 2e−21, y la señal que más lo empuja hacia legítimo es «telnet» (aporte ≈ −3.9). El modelo aprende frecuencias del **corpus** (palabras como «telnet» o «hp» son típicas de pocos remitentes), no de todos los correos.")

md("""
### C2 · ¿Por qué logaritmos? (10 min)
Un producto de muchos factores pequeños **desborda hacia 0** en doble precisión (el mínimo representable es ≈ 1e−308). Con 54 señales no hay problema, pero con un vocabulario de miles de palabras sí. Sumar logaritmos no pierde precisión, y la diferencia `ln s1 − ln s0` es el log-odds.
""")
code("""
d = np.arange(1, 1201)
prod = np.cumprod(np.full(len(d), 0.3))
d_cero = int(d[np.argmax(prod == 0)])
print("el producto de factores 0.3 se vuelve exactamente 0 en d =", d_cero, "| el logaritmo en ese punto:", round(d_cero * np.log10(0.3), 1), "(log10)")
print("suma de logaritmos con d = 1200:", round(1200 * np.log(0.3), 1), "(log natural) -> se conserva")
""")
md("""
El producto vale 0 desde d = **[[logprob.d_desborde]]** factores; la suma de logaritmos sigue bien definida. Por eso `predict_log_proba` calcula `log π + Σ log f` y normaliza con `logsumexp`.

### ✍️ Ejercicio C2
Dos clases con `ln s1 = −1 203.4` y `ln s0 = −1 207.9`. Calcula la probabilidad de la clase 1 **sin** calcular `exp(ln s1)` ni `exp(ln s0)` (que darían 0 en doble precisión).
""")
code("""
# tu código aquí
""")
solution("""
l1, l0 = -1203.4, -1207.9
lo = l1 - l0
print("exp(l1) =", np.exp(l1), "(0 en doble precisión) | log-odds =", lo, "| P =", round(float(1 / (1 + np.exp(-lo))), 4))
""", "El log-odds es 4.5 y P(clase 1) = 1/(1 + e^(−4.5)) = **0.989**. El truco: `exp` se aplica solo al log-odds, que ya es moderado.")

md("""
### C3 · Señales repetidas con los correos reales (15 min)
Se repiten **todas** las columnas binarias 1, 2, 4 y 8 veces y se vuelve a ajustar Naive Bayes Bernoulli. No hay información nueva, solo copias.
""")
code("""
Xb = (X > 0).astype(float)
filas = []
for k in (1, 2, 4, 8):
    p, lp = oof(BernoulliNB(alpha=1.0), np.hstack([Xb] * k), y, folds)
    m = medir(y, p, lp)
    filas.append(dict(copias=k, log_loss=m["log_loss"], brier=m["brier"], auc=m["auc"], accuracy=m["accuracy"], fraccion_extrema=np.mean((p < 0.01) | (p > 0.99))))
dup = pd.DataFrame(filas)
display(dup.round(4))
""")
md("""
El **AUC no cambia** ([[dup.real.0.auc|3]] con 1 copia y [[dup.real.2.auc|3]] con 4) y la exactitud tampoco, pero el **log loss crece de [[dup.real.0.log_loss|3]] a [[dup.real.3.log_loss|3]]** con 8 copias: el orden se conserva y las probabilidades se vuelven cada vez más extremas. Con los datos reales hay dependencias parecidas (por ejemplo, varias palabras que se usan juntas), así que **algo de este efecto ya está presente en el modelo de 54 señales**: por eso Naive Bayes suele ordenar bien y calibrar mal.

### ✍️ Ejercicio C3
Calcula la fracción de correos con probabilidad OOF fuera de (0.01, 0.99) para Bernoulli α = 1 (una copia). ¿Y para la regresión logística con las mismas 54 señales, escalada dentro de un Pipeline?
""")
code("""
# tu código aquí
""")
solution("""
p_nb = oof_p["BERN_1"]
logit54 = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
p_lg, lp_lg = oof(logit54, X, y, folds)
f = lambda p: round(float(np.mean((p < 0.01) | (p > 0.99))), 3)
print("Naive Bayes:", f(p_nb), "| logística:", f(p_lg), "| log loss logística:", round(float(medir(y, p_lg, lp_lg)["log_loss"]), 3))
""", "Naive Bayes deja ≈ 81 % de las probabilidades fuera de (0.01, 0.99) y la logística, ≈ 40 %; el log loss de la logística con las 54 señales es ≈ 0.318.")

md("""
### C4 · Calibración: Naive Bayes frente a la logística (20 min)
Una probabilidad está **calibrada** si, entre los correos a los que se asigna 0.9, ≈ 90 % son spam. Se compara el Bernoulli α = 1 (diapositiva 48), la gaussiana, la regresión logística con las **57 entradas** de la Clase 38 (escalada dentro de un Pipeline) y el Bernoulli **recalibrado** con una sigmoide (también dentro de cada pliegue).
""")
code("""
v57 = vs + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
X57 = tr[v57].to_numpy()
logit57 = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
oof_p["LOGIT57"], oof_lp["LOGIT57"] = oof(logit57, X57, y, folds)
cal_nb = CalibratedClassifierCV(BernoulliNB(alpha=1.0, binarize=0.0), method="sigmoid", cv=5)
oof_p["BERN_CAL"], _ = oof(cal_nb, X, y, folds)

def tabla_calibracion(p, nombre):
    bins = np.minimum((p * 10).astype(int), 9)
    filas = [dict(modelo=nombre, intervalo=f"{b / 10:.1f}-{(b + 1) / 10:.1f}", n=int((bins == b).sum()), prob_media=p[bins == b].mean(), fraccion_spam=y[bins == b].mean()) for b in range(10) if (bins == b).any()]
    return pd.DataFrame(filas)

cal1 = tabla_calibracion(oof_p["BERN_1"], "BERN_1")
display(cal1.round(3))
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 1], [0, 1], ":", c="gray")
for k, col in [("BERN_1", "tab:red"), ("BERN_CAL", "tab:green"), ("LOGIT57", "tab:blue")]:
    t = tabla_calibracion(oof_p[k], k)
    ax.plot(t.prob_media, t.fraccion_spam, "o-", c=col, label=k)
ax.set(xlabel="probabilidad media predicha", ylabel="fracción observada de spam"); ax.legend(); plt.show()
resumen = pd.DataFrame([dict(modelo=k, **medir(y, oof_p[k], oof_lp.get(k))) for k in ("BERN_05", "GAUSS_LOG", "LOGIT57", "BERN_CAL")])
display(resumen[["modelo", "log_loss", "brier", "auc"]].round(4))
""")
md("""
En el intervalo 0.9–1 hay **[[comparacion.bern1_top.n]] correos**: la predicción media es [[comparacion.bern1_top.pred_media|4]] y el spam observado, [[comparacion.bern1_top.spam_obs|4]] (1 066 de 1 157): **99.44 % predicho frente a 92.13 % observado**, como en la diapositiva 48. Naive Bayes es **sobreconfiado**. Con el mismo orden (AUC [[comparacion.oof.BERN_05.auc|3]] frente a [[comparacion.oof.LOGIT57.auc|3]] de la logística de 57 entradas) la logística tiene log loss [[comparacion.oof.LOGIT57.log_loss|3]] y Naive Bayes [[comparacion.oof.BERN_05.log_loss|3]]. Recalibrar con una sigmoide lo baja a [[comparacion.oof.BERN_CAL.log_loss|3]]: casi no cambia el orden, mejoran las probabilidades.

**Una probabilidad alta no es certeza** (diapositiva 58): expresa una preferencia del modelo bajo su ajuste; su fiabilidad se comprueba con la calibración, y la rareza del caso, con otras herramientas.

### ✍️ Ejercicio C4
Calibra con `method="isotonic"` en vez de `"sigmoid"` y compara el log loss OOF. ¿Qué método elegirías con pocos datos?
""")
code("""
# tu código aquí
""")
solution("""
cal_iso = CalibratedClassifierCV(BernoulliNB(alpha=1.0, binarize=0.0), method="isotonic", cv=5)
p_iso, lp_iso = oof(cal_iso, X, y, folds)
print("sigmoide:", round(float(medir(y, oof_p["BERN_CAL"])["log_loss"]), 4), "| isotónica:", round(float(medir(y, p_iso)["log_loss"]), 4))
""", "Sigmoide: 0.329; isotónica: 0.299; los dos mejoran mucho el log loss de Naive Bayes (≈ 0.557). Aquí, con 3 680 correos, la isotónica gana; pero tiene más parámetros y necesita más datos, y con pocos datos la **sigmoide** (dos parámetros) es más estable.")

md("""
### C5 · Fronteras: Naive Bayes, LDA, QDA y logística (15 min)
Naive Bayes gaussiano es un modelo **generativo**: aprende cómo se distribuyen las señales en cada clase. **LDA** y **QDA** también, pero con covarianza completa (común o por clase); Naive Bayes la supone **diagonal**. La **logística** es discriminativa: modela directamente P(clase | x). Con dos señales (`log(1 + «!»)` y `log(1 + «your»)`) se ven las fronteras P = 0.5.
""")
code("""
cols = [vs.index("char_freq_!"), vs.index("word_freq_your")]
Z, Zte = np.log1p(X[:, cols]), np.log1p(Xte[:, cols])
modelos = {"NB gaussiano": GaussianNB(), "LDA": LinearDiscriminantAnalysis(), "QDA": QuadraticDiscriminantAnalysis(), "Logística": LogisticRegression(C=1e4, max_iter=3000)}
g1 = np.linspace(-0.05, 1.9, 200); g2 = np.linspace(-0.05, 2.35, 200)
G1, G2 = np.meshgrid(g1, g2)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
filas = []
for ax, (nom, m) in zip(axs, modelos.items()):
    m.fit(Z, y)
    P = m.predict_proba(np.c_[G1.ravel(), G2.ravel()])[:, 1].reshape(G1.shape)
    ax.contourf(G1, G2, P, levels=[0, 0.5, 1], colors=["tab:blue", "tab:red"], alpha=0.15)
    ax.contour(G1, G2, P, levels=[0.5], colors="k")
    sub = np.random.default_rng(RS).choice(len(y), 600, replace=False)
    ax.scatter(Z[sub, 0], Z[sub, 1], c=np.where(y[sub] == 1, "tab:red", "tab:blue"), s=6, alpha=0.5)
    ax.set(title=nom, xlabel="log(1 + «!»)")
    pt = m.predict_proba(Zte)[:, 1]
    mt = medir(yte, pt)
    filas.append(dict(modelo=nom, exactitud=mt["accuracy"], log_loss=mt["log_loss"], auc=mt["auc"]))
axs[0].set_ylabel("log(1 + «your»)"); plt.show()
display(pd.DataFrame(filas).round(4))
""")
md("""
En los 921 correos de prueba: Naive Bayes [[fr.nb_acc|3]] de exactitud y log loss [[fr.nb_ll|3]]; LDA [[fr.lda_acc|3]]; QDA [[fr.qda_acc|3]]; logística [[fr.lg_acc|3]] con log loss [[fr.lg_ll|3]]. Con solo dos señales y poca correlación, las diferencias son pequeñas; LDA y logística dan **fronteras rectas**, Naive Bayes y QDA, **curvas** (cada clase tiene su propia varianza). Naive Bayes gaussiano **es QDA con covarianza diagonal**.

## Parte D · Gaussiana, columnas 0/1 y decisión

### D1 · Gaussiana y log(1 + x) (15 min)
Diapositivas 37 y 38. Ejemplo controlado del script: `x = 1.5`, medias 0 y 2, desviaciones 1 y previas 0.8 y 0.2. La **altura** de la curva es una densidad, no una probabilidad.
""")
code("""
z = 1.5
densidad = lambda x, mu, sd: np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
d0, d1 = densidad(z, 0, 1), densidad(z, 2, 1)
q0, q1 = 0.8 * d0, 0.2 * d1
print(f"alturas: {d0:.4f} y {d1:.4f} | productos: {q0:.4f} y {q1:.4f} | P(clase 1) = {q1 / (q0 + q1):.4f}")

# con los correos: ¿se parecen a una normal las frecuencias log(1 + x)?
for nom in ("word_freq_you", "char_freq_!"):
    zz = np.log1p(X[:, vs.index(nom)])
    print(f"{nom}: legítimos {np.mean(zz[y == 0] == 0):.0%} de ceros (media {zz[y == 0].mean():.3f}) | spam {np.mean(zz[y == 1] == 0):.0%} de ceros (media {zz[y == 1].mean():.3f})")
""")
md("""
Resultado: densidades [[gauss.densidad_0|4]] y [[gauss.densidad_1|4]], productos [[gauss.producto_0|4]] y [[gauss.producto_1|4]] y **P(clase 1) = [[gauss.post|4]]**. Con los correos, `log(1 + x)` conserva los ceros: en «you» el 40 % de los legítimos tiene 0 y en «!» el 73 %; una curva normal no describe bien esa masa en 0 (**transformar no garantiza normalidad**). Por eso, aun con AUC aceptable ([[comparacion.oof.GAUSS_LOG.auc|3]]), el log loss es [[comparacion.oof.GAUSS_LOG.log_loss|3]]: muchas probabilidades quedan clavadas en 0 o 1.

### ✍️ Ejercicio D1
Con la receta `GAUSS_LOG` (ya calculada), ¿qué fracción de las probabilidades OOF vale prácticamente 1 (> 0.999)? ¿Qué fracción de los correos **legítimos** recibe esa probabilidad?
""")
code("""
# tu código aquí
""")
solution("""
pg = oof_p["GAUSS_LOG"]
print("fracción > 0.999:", round(float(np.mean(pg > 0.999)), 3), "| entre legítimos:", round(float(np.mean(pg[y == 0] > 0.999)), 3))
""", "El 53.2 % de los correos queda con P > 0.999 y el 26.5 % de los legítimos recibe esa certeza equivocada: de ahí el log loss de 4.93, aunque el AUC sea 0.94.")

md("""
### D2 · Gaussiano sobre columnas 0/1: binarias y one-hot (25 min)
En el Taller 3 aparece un Naive Bayes gaussiano sobre columnas 0/1 y se pregunta por sus límites. Aquí se estudia **solo con los correos de spam**. Una columna 0/1 con proporción `p` en una clase se describe con **media p y varianza p(1 − p)**: una «campana» sobre dos valores. Problemas: (1) la densidad **no** es una probabilidad (puede ser > 1); (2) si `p` es casi 0, la varianza es casi cero y el aporte de «presente» se **exagera**; (3) un conjunto de dummies **one-hot** no son independientes (exactamente una vale 1).
""")
code("""
Xb = (X > 0).astype(float)
modelos_b = {
    "Bernoulli α=1": (BernoulliNB(alpha=1.0), Xb),
    "Gaussiano (0/1), var_smoothing 1e-9": (GaussianNB(var_smoothing=1e-9), Xb),
    "Gaussiano (0/1), var_smoothing 1e-2": (GaussianNB(var_smoothing=1e-2), Xb),
    "Gaussiano log(1 + x)": (GaussianNB(var_smoothing=1e-9), np.log1p(X)),
}
filas = []
for nom, (m, xx) in modelos_b.items():
    p, lp = oof(m, xx, y, folds)
    mt = medir(y, p, lp)
    filas.append(dict(modelo=nom, log_loss=mt["log_loss"], brier=mt["brier"], auc=mt["auc"], accuracy=mt["accuracy"]))
display(pd.DataFrame(filas).round(4))

# una columna muy rara: la palabra «3d»
j = vs.index("word_freq_3d")
p0, p1 = Xb[y == 0, j].mean(), Xb[y == 1, j].mean()
print(f"«3d» presente: legítimos {p0:.4f} (varianza {p0 * (1 - p0):.4f}) | spam {p1:.4f} (varianza {p1 * (1 - p1):.4f})")
ll_normal = lambda x, m, v: -0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v)
print("aporte de «presente» a favor de spam: Bernoulli", round(float(np.log(p1 / p0)), 2), "| normal", round(float(ll_normal(1, p1, p1 * (1 - p1)) - ll_normal(1, p0, p0 * (1 - p0))), 1))
""")
md("""
Con los 54 indicadores 0/1, el log loss OOF es **[[gaussbin.res.0.log_loss|2]]** con Bernoulli y **[[gaussbin.res.1.log_loss|2]]** con el gaussiano ([[gaussbin.res.2.log_loss|2]] con `var_smoothing = 1e-2`); el AUC casi no cambia ([[gaussbin.res.0.auc|3]] frente a [[gaussbin.res.1.auc|3]]). Es decir: el gaussiano **ordena bien y calibra mal**. En «3d» (presente en [[gaussbin.n_presente.0]] legítimos y [[gaussbin.n_presente.1]] spam), el aporte de «presente» es **[[gaussbin.lr_bern_3d|2]]** con Bernoulli y **[[gaussbin.lr_gauss_3d|0]]** con la normal: la varianza minúscula del grupo legítimo exagera la evidencia. Para señales casi siempre ausentes llega a **[[gaussbin.lr_gauss_min|0]]** en contra. Una sola columna puede dominar el producto.

Ahora el caso **one-hot** con una variable del spam: se categoriza el carácter «!» en tres tramos (ninguno; hasta 0.3; más de 0.3) y se compara la probabilidad empírica con tres modelos.
""")
code("""
from sklearn.naive_bayes import CategoricalNB
j = vs.index("char_freq_!")
cat = np.digitize(X[:, j], [1e-4, 0.3])          # 0: ninguno, 1: hasta 0.3, 2: más de 0.3
cat_te = np.digitize(Xte[:, j], [1e-4, 0.3])
oh, oh_te = np.eye(3)[cat], np.eye(3)[cat_te]
emp = [y[cat == k].mean() for k in range(3)]
tabla = pd.DataFrame({
    "tramo": ["ninguno", "hasta 0.3", "más de 0.3"], "n": [int((cat == k).sum()) for k in range(3)], "empírica": emp,
    "CategoricalNB": CategoricalNB(alpha=1.0).fit(cat[:, None], y).predict_proba(np.arange(3)[:, None])[:, 1],
    "BernoulliNB (dummies)": BernoulliNB(alpha=1.0).fit(oh, y).predict_proba(np.eye(3))[:, 1],
    "GaussianNB (dummies)": GaussianNB().fit(oh, y).predict_proba(np.eye(3))[:, 1]})
display(tabla.round(4))
for nom, m, a, b in [("CategoricalNB", CategoricalNB(alpha=1.0), cat[:, None], cat_te[:, None]), ("BernoulliNB (dummies)", BernoulliNB(alpha=1.0), oh, oh_te), ("GaussianNB (dummies)", GaussianNB(), oh, oh_te)]:
    mt = medir(yte, m.fit(a, y).predict_proba(b)[:, 1])
    print(f"{nom:24s} prueba: log loss {mt['log_loss']:.4f}  AUC {mt['auc']:.4f}")
""")
md("""
Con una **sola** variable categórica, la mejor estimación de P(spam | tramo) es la frecuencia de cada tramo ([[onehot.empirica.0|3]], [[onehot.empirica.1|3]], [[onehot.empirica.2|3]]), y `CategoricalNB` la reproduce ([[onehot.categorical.0|3]], [[onehot.categorical.1|3]], [[onehot.categorical.2|3]]). `GaussianNB` sobre las tres dummies da [[onehot.gauss.0|3]], [[onehot.gauss.1|3]] y [[onehot.gauss.2|3]]: **exagera** las probabilidades hacia 0 y 1, porque trata las tres dummies como independientes y con varianza pequeña. El **AUC no cambia** ([[oh.cat_auc|3]]: el orden de los tramos es el mismo) y el log loss de prueba sí: [[oh.cat_ll|3]] (Categorical), [[oh.ber_ll|3]] (Bernoulli) y [[oh.gau_ll|3]] (gaussiano).

**Conclusión conceptual**: Naive Bayes gaussiano sobre 0/1 **funciona como aproximación** (ordena), pero (a) una dummy no es normal, (b) la independencia condicional es especialmente falsa en dummies de una misma variable o en columnas redundantes, y (c) por eso las probabilidades no son fiables. La variante natural para 0/1 es Bernoulli; para categorías, `CategoricalNB`.

### ✍️ Ejercicio D2
Repite la tabla con `var_smoothing = 1e-2` en el `GaussianNB` sobre las dummies. ¿Se acerca a la probabilidad empírica? ¿Por qué un «arreglo» así no resuelve el problema de fondo?
""")
code("""
# tu código aquí
""")
solution("""
g2 = GaussianNB(var_smoothing=1e-2).fit(oh, y)
print("GaussianNB var_smoothing=1e-2:", np.round(g2.predict_proba(np.eye(3))[:, 1], 4), "| empírica:", np.round(emp, 4))
""", "Casi no cambia: 0.023, 0.694 y 0.990, frente a las empíricas 0.134, 0.522 y 0.792. `var_smoothing` es un parámetro de ajuste, no un modelo correcto: sigue tratando 0/1 como normales y las dummies como independientes. Es mejor cambiar de variante (Bernoulli o Categorical).")

md("""
### D3 · Umbral por costo y prueba histórica (20 min)
Diapositivas 51 a 54. Un FP (bloquear un legítimo) cuesta 5; un FN (dejar pasar spam), 1. Con una probabilidad `p` **calibrada**, bloquear cuesta `5(1 − p)` y dejar pasar, `p`: conviene bloquear si `p > 5/6 ≈ 0.833`. Como la calibración es imperfecta, se barren 19 umbrales (0.05 a 0.95) sobre las probabilidades **OOF del ganador** y se elige el de menor costo (empates: el mayor umbral). Solo después se prueba **una vez** en los 921 correos.
""")
code("""
umbrales = np.arange(1, 20) / 20
costos = pd.DataFrame([dict(umbral=t, **medir(y, oof_p[ganador], oof_lp[ganador], t)) for t in umbrales])
elegido = costos.sort_values(["costo_1000", "umbral"], ascending=[True, False]).iloc[0]
t = float(elegido.umbral)
display(costos[["umbral", "FP", "FN", "accuracy", "costo_1000"]].round(3).tail(8))
print(f"umbral elegido: {t:.2f} | FP = {int(elegido.FP)}, FN = {int(elegido.FN)} | costo OOF por 1000 = {elegido.costo_1000:.1f}")

modelo_final = recetas()[ganador].fit(X, y)                    # se reajusta con todo el desarrollo
lpt = log_probs(modelo_final, Xte)
pt = np.exp(lpt[:, 1])
mt = medir(yte, pt, lpt, t)
base = medir(yte, np.full(len(yte), y.mean()), None, 1.0)
print(f"prueba histórica: VN {mt['VN']}, FP {mt['FP']}, FN {mt['FN']}, VP {mt['VP']} | costo {mt['costo_1000']:.1f} | log loss {mt['log_loss']:.4f} | Brier {mt['brier']:.4f}")
print(f"si se deja pasar todo: costo {base['costo_1000']:.1f}")
""")
md("""
Reproduce las diapositivas 52 a 54: umbral **[[umbral.t_nb|2]]**, **78 FP y 416 FN** en OOF (costo [[umbral.c_nb_min|1]] por 1000) y en la prueba **541 / 17 / 89 / 274**, costo **[[umbral.prueba_nb.costo_1000|1]]** (frente a **[[umbral.base_prueba|1]]** si todo pasa), log loss [[umbral.prueba_nb.log_loss|3]] y Brier [[umbral.prueba_nb.brier|3]]. Dos precauciones: (1) el 0.95 está **en el borde de la rejilla**, así que estar allí no demuestra que sea el mejor umbral posible; (2) los 921 correos ya se usaron al comparar clasificadores: es una **comprobación retrospectiva**, no una prueba independiente.

La regla completa (diapositiva 56): representar → aprender con α → predecir → bloquear si P(spam) ≥ 0.95 → comprobar costo y calibración. **Suavizado y umbral cumplen tareas distintas**: uno modifica el cálculo de la probabilidad; el otro, la acción.

### ✍️ Ejercicio D3
Elige el umbral con costos FP = 3 y FN = 1 (umbral teórico 3/4). ¿Qué umbral resulta con el ganador y cuánto cuesta por 1000 en OOF? ¿Coincide con el teórico?
""")
code("""
# tu código aquí
""")
solution("""
c3 = [1000 * (3 * medir(y, oof_p[ganador], oof_lp[ganador], u)["FP"] + medir(y, oof_p[ganador], oof_lp[ganador], u)["FN"]) / len(y) for u in umbrales]
j3 = len(c3) - 1 - int(np.argmin(c3[::-1]))
print("umbral con FP = 3:", umbrales[j3], "| costo OOF por 1000:", round(c3[j3], 1), "| teórico:", 3 / 4)
""", "Sale **0.85** (costo OOF 176.1 por 1000), por encima del teórico 0.75: como las probabilidades de Naive Bayes están **sobreconfiadas**, hay legítimos con P cercana a 1 y conviene exigir más. Más cerca del teórico solo se llegaría tras **recalibrar**.")

md("""
### D4 · Alertas del material, verificadas con código (10 min)
Se revisan afirmaciones de la clase y las diapositivas.
""")
code("""
# 1) tabla de 100 correos: enlace en legítimos y en spam (diapositiva 11)
print("legítimos con enlace: 16/80 =", 16 / 80, "| spam con enlace: 14/20 =", 14 / 20, "| spam SIN enlace:", 20 - 14)
# 2) suavizado: el denominador suma 2*alpha (un alpha por cada respuesta)
print("spam sin enlace con alpha=1:", (6 + 1) / (20 + 2))
# 3) costo en OOF con umbral 0.95 (el profesor habló de «2 16»; el código da otro número)
o = medir(y, oof_p["BERN_05"], oof_lp["BERN_05"], 0.95)
print("FP =", o["FP"], "FN =", o["FN"], "| costo por 1000 =", round(o["costo_1000"], 1))
# 4) log loss de la frecuencia (se dijo 0.61)
print("log loss de la frecuencia:", round(float(medir(y, oof_p["FRECUENCIA"], oof_lp["FRECUENCIA"])["log_loss"]), 3), "| correos:", len(datos))
# 5) con probabilidades calibradas, el umbral teórico es 5/6
print("5/6 =", round(5 / 6, 4), "| umbral empírico en OOF:", t)
""")
md("""
**Lo dicho en clase (transcripción)**
- **Tabla de enlaces**: se dijo «16 correos legítimos tienen enlace y 14 spam no tienen enlace»; la diapositiva 11 dice 16 legítimos con enlace y **14 spam con enlace** (6 sin enlace). Después se usa bien. También «6 correos sin enlace dentro de los 20 legítimos» son los 20 **spam**.
- **Costo**: la transcripción mencionó «2 16»; el código y la diapositiva 53 dan **219.0** por 1000 (78 FP y 416 FN) con umbral 0.95. El log loss de la frecuencia es **0.671** (no 0.61) y los correos son **4 601**, no 6 000.
- **«Es sobreajuste»** (duplicar una columna): es **sobreconfianza por dependencia**; en regresión, columnas idénticas no suman información, en Naive Bayes sí aumentan la certeza.
- **Umbral y calibración**: con probabilidades calibradas el umbral teórico es 5/6; el empírico depende de la rejilla y de la calibración.
- **Balanceo de clases** (pregunta de Romina): remuestrear ayuda a aprender la clase rara, pero **cambia la previa**; después hay que reponer la prevalencia real o las probabilidades quedan sobreestimadas.
- **Suavizar no corrige la dependencia entre señales**: es correcto.

**Diapositivas y scripts**: el script no compara con la logística, LDA ni QDA; esas comparaciones son de esta guía (la logística usa la receta de la Clase 38). Las diapositivas explican que **la multinomial exige conteos**, por eso no se aplica a los porcentajes de Spambase. La transcripción tiene ruido: «Nahid Valles», «naivayes», «NUGI» = Naive Bayes y Bernoulli; «suavizamiento de la plaza» = de Laplace; «el láser» = el enlace.

### D5 · Tabla de correspondencia R ↔ Python (10 min)
Los scripts del profesor hacen el cálculo **a mano** en ambos lenguajes; las librerías lo automatizan (diapositiva 61).

| Qué | R | Python |
|---|---|---|
| Naive Bayes Bernoulli | `naivebayes::bernoulli_naive_bayes()` | `BernoulliNB(alpha=…)` |
| Naive Bayes gaussiano | `naivebayes::gaussian_naive_bayes()` | `GaussianNB(var_smoothing=…)` |
| Naive Bayes multinomial | `naivebayes::multinomial_naive_bayes()` | `MultinomialNB(alpha=…)` |
| Suavizado de Laplace | `laplace = 1` | `alpha = 1` |
| Probabilidades | `predict(modelo, nuevos, type = "prob")` | `modelo.predict_proba(nuevos)` |
| Log-probabilidades | `predict(..., type = "prob")` y `log()` | `modelo.predict_log_proba(nuevos)` |
| Dos o más categorías | `factor` + `naive_bayes()` | `CategoricalNB()` |
| Regresión logística | `glm(y ~ ., family = binomial)` | `LogisticRegression()` |
| LDA / QDA | `MASS::lda()` / `MASS::qda()` | `LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis` |
| Log(1 + x) | `log1p(x)` | `np.log1p(x)` |
| Pliegues OOF | bucle con `folds == k` | bucle con `folds == k` o `cross_val_predict` |
| Curva ROC y AUC | `pROC::roc()` | `roc_curve`, `roc_auc_score` |
| Calibración | `calibration()` o tabla manual | `CalibratedClassifierCV`, tablas por intervalos |
| Lectura del CSV | `read.csv(..., check.names = FALSE)` | `pd.read_csv()` |
| Seleccionar columnas por nombre | `names(datos)[grepl("_freq_", names(datos))]` | `[c for c in datos if c.startswith(("word_freq_", "char_freq_"))]` |

**Diferencias que importan**: (1) los dos scripts producen los mismos números en los ejemplos pequeños; en las tablas de 3 680 correos, la logística y los árboles pueden diferir por implementación; (2) en Python, `alpha = 0` se sustituye por 1e−10 y `predict_proba` puede dar exactamente 0 o 1; (3) R usa `check.names = FALSE` porque las columnas traen caracteres como `!` y `$`.

## Resumen operativo
1. Representa las señales (presencia o `log(1 + x)`), aprende por clase con α y combina **sumando logaritmos**.
2. Compara con predicciones **OOF** usando log loss, Brier y AUC; revisa la **calibración**.
3. Elige el **umbral** con los costos, en desarrollo; prueba **una vez**.
4. Recuerda: Naive Bayes ordena bien y calibra peor; señales repetidas o dependientes inflan la certeza; un gaussiano sobre columnas 0/1 es una aproximación débil.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes **construir un filtro de spam** con `spam_s40.csv` y justificarlo ante el equipo. Los costos son FP = 5 y FN = 1.

1. **Plan previo**: escribe en un comentario cómo compararás modelos, cómo elegirás el umbral y qué probarás **una sola vez**, **antes** de calcular nada.
2. **Modelos**: compara en **OOF** (con Pipelines) Naive Bayes Bernoulli (α = 0.5 y 1), la gaussiana sobre `log(1 + x)` y la regresión logística de las 54 señales escalada dentro del Pipeline. Resume log loss, Brier y AUC.
3. **Calibración**: haz la tabla por intervalos del mejor Naive Bayes y comenta si es sobreconfiado.
4. **Umbral**: busca el umbral de menor costo en OOF del modelo elegido por log loss.
5. **Prueba única**: evalúa el modelo y el umbral en los 921 correos históricos y compara con «dejar pasar todo».
6. **Sin código**: dos advertencias sobre el uso del filtro (por ejemplo, que una probabilidad alta no es certeza y que el corpus es histórico).
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1) plan: modelo con menor log loss OOF; umbral con menor costo OOF (19 umbrales, empates al mayor); probar una sola vez en prueba
cands = {"BERN_05": BernoulliNB(alpha=0.5), "BERN_1": BernoulliNB(alpha=1.0),
         "GAUSS_LOG": make_pipeline(FunctionTransformer(np.log1p), GaussianNB()),
         "LOGISTICA": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))}
res_c, op, olp = [], {}, {}
for nom, m in cands.items():
    op[nom], olp[nom] = oof(m, X, y, folds)
    mt = medir(y, op[nom], olp[nom])
    res_c.append(dict(modelo=nom, log_loss=mt["log_loss"], brier=mt["brier"], auc=mt["auc"]))
res_c = pd.DataFrame(res_c)
display(res_c.round(4))
mejor = res_c.loc[res_c.log_loss.idxmin(), "modelo"]
cos = [medir(y, op[mejor], olp[mejor], u)["costo_1000"] for u in umbrales]
jm = len(cos) - 1 - int(np.argmin(cos[::-1]))
u_ok = float(umbrales[jm])
fin = clone(cands[mejor]).fit(X, y)
lp_f = log_probs(fin, Xte)
pt_f = np.exp(lp_f[:, 1])
mt_f = medir(yte, pt_f, lp_f, u_ok)
print("modelo:", mejor, "| umbral OOF:", u_ok, "| costo OOF:", round(cos[jm], 1))
print("prueba: FP", mt_f["FP"], "FN", mt_f["FN"], "| costo", round(mt_f["costo_1000"], 1), "| dejar pasar todo:", round(base["costo_1000"], 1))
display(tabla_calibracion(op["BERN_1"], "BERN_1").round(3))
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: gana la **regresión logística** (log loss OOF 0.318 frente a 0.556 de Bernoulli α = 0.5 y 4.931 de la gaussiana); el AUC de Naive Bayes Bernoulli (0.952) es casi el de la logística (0.950). Naive Bayes es **sobreconfiado** (intervalo 0.9–1: predice 99.4 % y observa 92.1 %). Con la logística, el umbral OOF es 0.75 (costo 260.6 por 1000) y en la prueba única salen 9 FP y 73 FN, costo 128.1 frente a 394.1 si todo pasa (con Naive Bayes Bernoulli α = 0.5 y umbral 0.95 fueron 188.9). La comprobación en prueba se hace **una sola vez**. Advertencias: (1) una probabilidad alta no es certeza ni dice que el correo sea «típico»; (2) Spambase es un corpus **histórico**: el filtro debe reevaluarse en la bandeja actual con datos separados y su prevalencia de spam.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué diferencia hay entre la previa, la condicional y la posterior? ¿Por qué P(oferta | spam) ≠ P(spam | oferta)?
2. ¿Por qué cambia P(spam | oferta) cuando cambia la frecuencia de spam si las condicionales son iguales?
3. ¿Qué supone Naive Bayes? ¿Por qué evita contar 2^d combinaciones y qué pasa si el supuesto falla?
4. ¿Qué hace el suavizado de Laplace y cuándo importa? ¿Corrige la dependencia entre señales?
5. ¿Por qué se suman logaritmos? ¿Qué es el log-odds y cómo se convierte en probabilidad?
6. ¿Cuándo usar Bernoulli, gaussiana o multinomial? ¿Por qué no se usó la multinomial con Spambase?
7. ¿Por qué un gaussiano sobre una columna 0/1 es una aproximación débil? ¿Qué cambia con dummies one-hot?
8. ¿Qué es una predicción OOF y por qué el escalado va dentro de un Pipeline?
9. ¿Por qué Naive Bayes y la logística pueden tener el mismo AUC y distinto log loss?
10. ¿Cómo se elige el umbral con costos FP = 5 y FN = 1? ¿Por qué 5/6 no coincide con el 0.95 empírico?
11. ¿Una probabilidad de 99.6 % significa que el 99.6 % de los correos parecidos es spam?

<details><summary><b>Respuestas</b></summary>

1. La previa es la frecuencia de la clase; la condicional, la frecuencia de la señal **dentro** de la clase (se mira la fila); la posterior, la clase **dado** que se vio la señal (se mira la columna). Son preguntas distintas con el mismo vocabulario.
2. Porque la posterior mezcla previa y condicionales: con 2 % de spam, 0.02 × 0.8 = 0.016 frente a 0.98 × 0.1 = 0.098; la misma palabra pasa de 66.7 % a 14.0 %.
3. Que, dentro de cada clase, las señales son independientes: el factor conjunto es un producto de factores por señal. Si falla (señales repetidas o correlacionadas), el modelo cuenta dos veces la misma evidencia y se sobreconfía.
4. Suma α a «presente» y a «ausente» para que un conteo cero no anule el producto. Importa con pocos datos o señales muy raras; no corrige la dependencia.
5. Para evitar el desborde hacia 0 y porque el log-odds suma la evidencia; P = 1/(1 + e^(−log-odds)).
6. Bernoulli con presencias 0/1; gaussiana con mediciones (por ejemplo log(1 + x)); multinomial con conteos de términos. Spambase trae porcentajes, no conteos.
7. Una dummy no es normal; la media p y la varianza p(1 − p) producen densidades que no son probabilidades y exageran señales raras. En one-hot, las dummies no son independientes (solo una vale 1). Mejor Bernoulli o `CategoricalNB`.
8. Es la predicción de un correo con un modelo que no lo vio. El escalado dentro del Pipeline se ajusta solo con los pliegues de ajuste; así no se filtra información.
9. El AUC mide el **orden**; el log loss y la calibración miden las **probabilidades**. Naive Bayes ordena bien pero tiende a probabilidades extremas.
10. Bloquear si 5(1 − p) < p, es decir p > 5/6; en la práctica se barre una rejilla con OOF y se elige el menor costo. El 0.95 refleja probabilidades sobreconfiadas y el borde de la rejilla; se recalibra o se afina la búsqueda.
11. No: es una preferencia del modelo bajo su ajuste; la fiabilidad se comprueba con la calibración (en el intervalo 0.9–1, 99.4 % predicho y 92.1 % observado).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** Clase 41 (anomalías) y el Taller 3. Revisa `Guias/apuntes/C40_Ejercicios_Cuaderno_NaiveBayes.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
