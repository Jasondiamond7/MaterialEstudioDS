"""Genera el notebook guiado de la Clase 44 (Aplicaciones de IA y panorama actual del Deep Learning).

Uso normal:      python notebook_c44.py             -> Guias/notebooks/C44_AplicacionesIA_DeepLearning.ipynb
Modo de prueba:  C44_TEST=1 python notebook_c44.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c44_numeros.json (lo escribe figuras_c44.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib y scikit-learn.
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C44_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C44_AplicacionesIA_DeepLearning_test.ipynb" if TEST else "C44_AplicacionesIA_DeepLearning.ipynb")
N = json.loads((HERE / "c44_numeros.json").read_text(encoding="utf-8"))
# textos ya formateados para las marcas [[s.xxx]]
pc = lambda x: f"{x * 100:.2f} %"
N["s"] = dict(log_acc=pc(N["log"]["acc"]), red_acc=pc(N["red"]["acc"]), dlog_acc=pc(N["desp"]["log"]["acc"]), dred_acc=pc(N["desp"]["red"]["acc"]),
              d2log_acc=pc(N["desp_k"]["2"]["log"]["acc"]), d2red_acc=pc(N["desp_k"]["2"]["red"]["acc"]),
              dm1log_acc=pc(N["desp_k"]["-1"]["log"]["acc"]), dm1red_acc=pc(N["desp_k"]["-1"]["red"]["acc"]),
              cand16=f"{N['datos']['cand']['(16,)']:.4f}", cand32=f"{N['datos']['cand']['(32,)']:.4f}", cap=N["costo"]["capacidad"], costo=N["costo"]["costo"], acc_costo=pc(N["costo"]["acc"]), siempre=pc(N["costo"]["siempre_sano_acc"]))
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
    return re.sub(r"\[\[([\w.\-]+)(?:\|(\d+))?\]\]", val, text)


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
# Clase 44 · Aplicaciones de IA y panorama actual del Deep Learning
**Diplomado Data Science** · Prof. Eloy Alvarado · conecta clasificación (Clases 20–21 y 31), evaluación y bootstrap (Clases 38 y 43) y distancia entre datos (Clase 39) con imágenes, texto y modelos preentrenados

Material de apoyo:
- `Guias/C44_Guia_AplicacionesIA_DeepLearning.pdf` — guía visual de la clase.
- `Guias/apuntes/C44_Apuntes_AplicacionesIA_DeepLearning.pdf` — apuntes explicados paso a paso, con ejemplos resueltos.
- `Guias/apuntes/C44_Ejercicios_Cuaderno_AplicacionesIA_DeepLearning.pdf` — ejercicios en papel (antes o después de este notebook).
- Diapositivas del profesor (carpeta «Clase 44 …»): `DipDS_2026_S44.pdf`.
- Transcripción: `VideosClases/Clase 44 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib` y `scikit-learn`. Si no los tienes:
```
pip install numpy pandas matplotlib scikit-learn
```

**Datos**: la colección de dígitos escritos a mano de scikit-learn (`load_digits()`, UCI «Optical Recognition of Handwritten Digits»): **1 797 imágenes de 8 × 8 píxeles** con intensidades de 0 a 16 y una etiqueta entre 0 y 9. La clase usa una partición **1 077 entrenamiento / 360 validación / 360 prueba**; esa partición del profesor está en la columna `conjunto` de `digitos.csv` (carpeta de la Clase 45). El notebook la busca ahí; si no la encuentra, **arma una partición propia** (y las cifras cambian un poco).

**Cómo leer los números de este notebook**: la regresión logística reproduce *exactamente* las cifras de las diapositivas. La red neuronal (que el profesor entrenó con su propio código) **no** sale idéntica: aquí es una `MLPClassifier` con semilla fija. Cada vez que hay una diferencia, el texto lo dice (*diapositivas* = corrida del profesor; *nuestra reproducción* = este notebook).

### Objetivos
1. Ver una **imagen como números** (8 × 8 = 64 intensidades, escaladas entre 0 y 1) y entender qué recibe un modelo.
2. Obtener **probabilidades por clase** (`predict_proba`) y leer la diferencia entre la clase elegida y la probabilidad de la clase real.
3. Calcular a mano y con scikit-learn la **exactitud** y el **log loss**, y ver por qué no dicen lo mismo.
4. Inspeccionar los **errores confiados** y probar un **umbral** de aceptación automática.
5. Estimar la incertidumbre de la **diferencia de pérdidas** con bootstrap.
6. Hacer una **prueba de estrés** (desplazar las imágenes) y entender los límites de una evaluación.
7. Trabajar con **embeddings de juguete**: distancia y similitud coseno.
8. Calcular **costos de errores** y **carga de revisión**, y **separar por la unidad correcta** (envases, no fotos).

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Una imagen son números |
| A | 10 | A2 · La partición y las etiquetas |
| B | 15 | B1 · Regresión logística y `predict_proba` |
| B | 20 | B2 · Red neuronal: elegir por validación |
| B | 20 | B3 · Exactitud y log loss, a mano y con sklearn |
| C | 20 | C1 · Los errores confiados |
| C | 15 | C2 · Un umbral de aceptación automática |
| D | 20 | D1 · Bootstrap de la diferencia de pérdidas |
| E | 20 | E1 · Prueba de estrés: desplazar una columna |
| F | 20 | F1 · Embeddings de juguete |
| G | 20 | G1 · Costo de errores y carga de revisión |
| — | 25 | Ejercicio de cierre **sin ayuda** |

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
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso
- **Probabilidades y regresión logística** (Clases 20–21 y 31): para cada observación el modelo da una probabilidad por clase; la clase predicha es la de mayor probabilidad.
- **Exactitud y matriz de confusión** (Clase 38): proporción de aciertos; falsos positivos y falsos negativos.
- **Bootstrap** (Clase 43): remuestrear con reposición los casos de prueba con los modelos ya fijos, y usar los percentiles como intervalo.
- **Distancia** (Clase 39): qué tan lejos están dos puntos.
- **Tres círculos** (diapositiva 4): el aprendizaje profundo está dentro del aprendizaje automático, que está dentro de la inteligencia artificial.
""")

# ================================================================== PARTE A
md("""
## Parte A · Una imagen son números

### A1 · Una imagen son números (15 min)
Un modelo no «ve» una foto: recibe **números**. Cada imagen de la colección es una cuadrícula de 8 × 8 celdas (píxeles) y cada celda tiene una **intensidad** entre 0 (blanco) y 16 (negro). Se lee la matriz **fila por fila** y queda un vector de 64 números; se divide por 16 para que estén entre 0 y 1. La etiqueta (el dígito correcto) se guarda **aparte**.
""")
code("""
digitos = load_digits()
X_int = digitos.data.astype(int)       # intensidades 0-16, una fila por imagen (64 columnas)
y = digitos.target                     # etiqueta: el dígito correcto
print("Entrada:", X_int.shape, "| etiquetas:", y.shape, "| intensidad mínima y máxima:", X_int.min(), X_int.max())

i = 8                                  # la imagen de la diapositiva 11 (un 8)
imagen = X_int[i].reshape(8, 8)
print("Etiqueta:", y[i])
print(imagen)
print("Primera fila:", imagen[0].tolist())
print("Primera fila ÷ 16:", (imagen[0] / 16).round(4).tolist())

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(imagen, cmap="gray_r", vmin=0, vmax=16)
axes[0].set_title(f"etiqueta {y[i]}")
axes[0].grid(False)
im = axes[1].imshow(imagen / 16, cmap="gray_r", vmin=0, vmax=1)
axes[1].set_title("misma imagen, escala 0 a 1")
axes[1].grid(False)
plt.tight_layout()
plt.show()
""")
md("""
La primera fila es `[0, 0, 9, 14, 8, 1, 0, 0]` y, escalada, `[0, 0, 0.5625, 0.875, 0.5, 0.0625, 0, 0]`. Esa imagen tiene [[ejemplo.n_ceros]] celdas en 0 y [[ejemplo.n_16]] en 16. Recuerda: **no hay una única forma de pasar un objeto a números** (un correo se puede contar por palabras o por largo); aquí se usan las 64 intensidades.

### ✍️ Ejercicio A1
Toma la imagen número **100**. Muestra su etiqueta, su matriz de 8 × 8, cuántas celdas tienen intensidad mayor que 8 y su primera fila escalada entre 0 y 1.
""")
code("""
# tu código aquí
""")
solution("""
i = 100
im = X_int[i].reshape(8, 8)
print("Etiqueta:", y[i])
print(im)
print("Celdas con intensidad > 8:", int((im > 8).sum()))
print("Primera fila ÷ 16:", (im[0] / 16).round(4).tolist())
""", "Ver la matriz: las celdas con intensidad alta forman el trazo del dígito.")

md("""
### A2 · La partición y las etiquetas (10 min)
Se separan los datos en **entrenamiento** (ajustar los modelos candidatos), **validación** (elegir entre candidatos) y **prueba** (evaluar al final, una sola vez). La separación es por **imagen**: la colección no trae el escritor de cada dígito, así que no se puede asegurar que sirva para escritores nuevos.
""")
code("""
def buscar_particion():
    # Devuelve la columna «conjunto» del profesor (digitos.csv) o None si no está.
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 45*/digitos.csv"):
            return pd.read_csv(p)["conjunto"].to_numpy()
        if (base / "digitos.csv").exists():
            return pd.read_csv(base / "digitos.csv")["conjunto"].to_numpy()
    return None


conjunto = buscar_particion()
if conjunto is None:
    # partición propia (estratificada) si no está digitos.csv
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    i_dev, i_te = train_test_split(idx, test_size=360, stratify=y, random_state=44)
    i_tr, i_va = train_test_split(i_dev, test_size=360, stratify=y[i_dev], random_state=44)
    conjunto = np.empty(len(y), dtype=object)
    conjunto[i_tr], conjunto[i_va], conjunto[i_te] = "entrenamiento", "validacion", "prueba"
    print("⚠️ No encontré digitos.csv: uso una partición propia; las cifras serán algo distintas a las del texto.")
else:
    print("Uso la partición del profesor (columna «conjunto» de digitos.csv).")

tr, va, te = (conjunto == k for k in ("entrenamiento", "validacion", "prueba"))
dev = tr | va                          # desarrollo: entrenamiento + validación
X = X_int / 16                         # la entrada de los modelos: 64 números entre 0 y 1
print("entrenamiento:", tr.sum(), "| validación:", va.sum(), "| prueba:", te.sum(), "| desarrollo:", dev.sum())
print("dígitos en prueba:", np.bincount(y[te]).tolist())
""")
md("""
Con la partición del profesor hay **[[datos.n_tr]]** imágenes de entrenamiento, **[[datos.n_va]]** de validación y **[[datos.n_te]]** de prueba; el *desarrollo* son **[[datos.n_dev]]** (entrenamiento + validación). Después de elegir el modelo se reajusta con las [[datos.n_dev]] y la prueba **no participa** en ninguna decisión.
""")

# ================================================================== PARTE B
md("""
## Parte B · Dos modelos para la misma tarea

### B1 · Regresión logística y `predict_proba` (15 min)
La regresión logística (la referencia barata e interpretable) usa los 64 valores para calcular **10 probabilidades**, una por dígito, que suman 1. Se ajusta con las imágenes de desarrollo.
""")
code("""
lg = LogisticRegression(C=1, max_iter=3000, tol=1e-8).fit(X[dev], y[dev])
P_lg = lg.predict_proba(X[te])           # una fila por imagen de prueba, 10 columnas (dígitos 0 a 9)
print("forma de P_lg:", P_lg.shape, "| suma de la primera fila:", P_lg[0].sum().round(6))

j = 0                                    # primera imagen de prueba
print("dígito real:", y[te][j], "| clase predicha:", P_lg[j].argmax())
print(pd.Series(P_lg[j].round(4), name="probabilidad").to_frame().T)
""")
md("""
La clase predicha es la de **mayor probabilidad** (`argmax`). Pero lo que se predice y la **probabilidad del dígito real** son cosas distintas: el log loss (más abajo) usa la segunda.

### ✍️ Ejercicio B1
Para la imagen de prueba número **5**: muestra el dígito real, la clase predicha por la logística, la probabilidad que le asignó a la clase **real** y la probabilidad máxima. ¿Acertó?
""")
code("""
# tu código aquí
""")
solution("""
j = 5
real = y[te][j]
pred = P_lg[j].argmax()
print("real:", real, "| predicha:", pred)
print("probabilidad de la clase real:", P_lg[j, real].round(4))
print("probabilidad máxima:", P_lg[j].max().round(4))
print("¿Acertó?", pred == real)
""", "Si la clase real coincide con la predicha, la probabilidad real es la máxima; si no, es menor.")

md("""
### B2 · Red neuronal: elegir por validación (20 min)
La red aprende **transformaciones sucesivas** de los 64 valores para producir las 10 probabilidades (su construcción se estudia el 20 de octubre; hoy solo importan sus entradas y salidas). Como en la clase, se comparan **dos candidatas** (16 o 32 unidades en la capa oculta): se ajustan con *entrenamiento*, se elige la de menor **log loss de validación** y se **reajusta con el desarrollo**.
""")
code("""
def entrenar_red(capas, Xa, ya):
    return MLPClassifier(hidden_layer_sizes=capas, activation="relu", solver="adam", learning_rate_init=0.002, batch_size=64, alpha=1e-4,
                         max_iter=120, tol=0, n_iter_no_change=1000, random_state=20261020).fit(Xa, ya)


val_ll = {}
for capas in [(16,), (32,)]:
    red_c = entrenar_red(capas, X[tr], y[tr])
    val_ll[capas] = log_loss(y[va], red_c.predict_proba(X[va]), labels=np.arange(10))
    print(f"candidata {capas}: log loss de validación = {val_ll[capas]:.4f}")

elegida = min(val_ll, key=val_ll.get)
print("elegida:", elegida)
red = entrenar_red(elegida, X[dev], y[dev])      # se reajusta con las 1 437 imágenes de desarrollo
P_red = red.predict_proba(X[te])
print("parámetros de la red elegida:", sum(w.size + b.size for w, b in zip(red.coefs_, red.intercepts_)), "| de la logística:", lg.coef_.size + lg.intercept_.size)
""")
md("""
En nuestra reproducción, gana la candidata de **[[datos.elegida]]** unidades (log loss de validación [[s.cand16]] con 16 y [[s.cand32]] con 32) y tiene [[datos.n_params_elegida]] parámetros frente a [[datos.n_params_log]] de la logística. La elección usó solo validación: la prueba aún no se ha mirado.

### ✍️ Ejercicio B2
¿Qué pasaría si eligieras la red mirando el log loss de **prueba** en vez del de validación? Calcula el log loss de prueba de las dos candidatas (ajustadas con entrenamiento) y explica en un comentario por qué eso **no** es una evaluación limpia.
""")
code("""
# tu código aquí
""")
solution("""
for capas in [(16,), (32,)]:
    r = entrenar_red(capas, X[tr], y[tr])
    print(capas, "log loss de prueba:", round(log_loss(y[te], r.predict_proba(X[te]), labels=np.arange(10)), 4))
# Comentario: si la elección usa la prueba, la prueba deja de ser independiente: el resultado final
# queda optimista porque ya participó en una decisión. Por eso se elige con validación.
""", "Compara con la elección por validación: si coinciden, no hay problema práctico, pero el procedimiento igual es incorrecto.")

md("""
### B3 · Exactitud y log loss, a mano y con sklearn (20 min)
- **Exactitud** = aciertos ÷ imágenes (más alto es mejor).
- **Pérdida de una imagen** = −ln(probabilidad asignada al dígito **real**). **Log loss** = promedio de esas pérdidas (más bajo es mejor).

Ejemplos de la diapositiva 15: −ln(0.90) ≈ 0.105, −ln(0.60) ≈ 0.511 y −ln(0.01) ≈ 4.605. Una predicción confiada y equivocada cuesta mucho.
""")
code("""
y_te = y[te]


def perdidas(y_real, P):
    # −ln de la probabilidad del dígito real, imagen por imagen
    return -np.log(np.clip(P[np.arange(len(y_real)), y_real], 1e-15, 1))


print("−ln de 0.90, 0.60 y 0.01:", [round(float(-np.log(p)), 3) for p in (0.90, 0.60, 0.01)])

filas = []
for nombre, P in [("Logística", P_lg), ("Red neuronal", P_red)]:
    L = perdidas(y_te, P)
    aciertos = int((P.argmax(1) == y_te).sum())
    filas.append({"modelo": nombre, "aciertos": aciertos, "exactitud (%)": round(100 * aciertos / len(y_te), 2),
                  "log loss (a mano)": round(float(L.mean()), 4), "log loss (sklearn)": round(log_loss(y_te, P, labels=np.arange(10)), 4)})
display(pd.DataFrame(filas))
""")
md("""
**Nuestra reproducción**: logística [[log.k]] de 360 ([[s.log_acc]]) con log loss [[log.ll|4]]; red [[red.k]] de 360 ([[s.red_acc]]) con log loss [[red.ll|4]]. **Diapositivas**: logística 353 (98.06 %, 0.1353) y red 354 (98.33 %, 0.0657). La logística coincide; la red no (otra inicialización o programa), pero la lección es la misma: **con exactitudes casi iguales, el log loss puede diferir bastante**. Un acierto más son 100/360 ≈ 0.28 puntos de exactitud.

### ✍️ Ejercicio B3
Para la **red**, calcula la pérdida de cada imagen de prueba y muestra las **5 imágenes con mayor pérdida**: índice, dígito real, clase predicha y probabilidad del dígito real. ¿Qué parte de la pérdida total aportan esas 5?
""")
code("""
# tu código aquí
""")
solution("""
L_red = perdidas(y_te, P_red)
top = np.argsort(L_red)[::-1][:5]
tabla = pd.DataFrame({"imagen": top, "real": y_te[top], "predicha": P_red[top].argmax(1), "p real": P_red[top, y_te[top]].round(4), "pérdida": L_red[top].round(3)})
display(tabla)
print("fracción de la pérdida total que aportan estas 5:", round(float(L_red[top].sum() / L_red.sum()), 3))
""", "Pocas imágenes concentran gran parte de la pérdida: son los errores (y los casi-errores) confiados. En nuestra red, los [[red.errores]] errores aportan [[red.perdida_errores|1]] de [[red.perdida_total|1]] de pérdida total.")

# ================================================================== PARTE C
md("""
## Parte C · Errores confiados y umbrales

### C1 · Los errores confiados (20 min)
Una probabilidad alta **no garantiza** el acierto de esa imagen. Se miran los errores de la red ordenados por la probabilidad que le dio a la clase predicha (diapositiva 17: la red del profesor erró 6 imágenes, con p entre 0.76 y 0.94).
""")
code("""
cl_red = P_red.argmax(1)
mal = np.where(cl_red != y_te)[0]
orden = mal[np.argsort(-P_red[mal].max(1))]        # de más a menos confiado
print("errores de la red:", len(mal), "| de la logística:", int((P_lg.argmax(1) != y_te).sum()))

fig, axes = plt.subplots(1, min(6, len(orden)), figsize=(10, 2.4))
for ax, k in zip(np.atleast_1d(axes), orden[:6]):
    ax.imshow(X[te][k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"Real {y_te[k]} · Pred. {cl_red[k]}\\np = {P_red[k].max():.2f}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")
md("""
En nuestra red, **[[red.n_conf90]] de [[red.errores]]** errores tienen probabilidad ≥ 0.90 (la logística: [[log.n_conf90]]). La red es más «segura» en general (probabilidad máxima media [[red.pmax_media|3]] frente a [[log.pmax_media|3]]), y por eso sus errores pesan tanto en el log loss.

### ✍️ Ejercicio C1
Para la **logística**, muestra los errores en una tabla con: índice, dígito real, clase predicha, probabilidad de la clase predicha y probabilidad del dígito real, ordenados de más a menos confiado.
""")
code("""
# tu código aquí
""")
solution("""
cl_lg = P_lg.argmax(1)
mal_lg = np.where(cl_lg != y_te)[0]
t = pd.DataFrame({"imagen": mal_lg, "real": y_te[mal_lg], "predicha": cl_lg[mal_lg],
                  "p predicha": P_lg[mal_lg].max(1).round(4), "p real": P_lg[mal_lg, y_te[mal_lg]].round(4)})
display(t.sort_values("p predicha", ascending=False))
""", "La logística tiene [[log.errores]] errores y ninguno con p ≥ 0.90: es menos «segura» que la red.")

md("""
### C2 · Un umbral de aceptación automática (15 min)
Un **umbral** es un valor de corte: «aceptar sin revisión si la probabilidad máxima es ≥ t; el resto va a revisión». Es una regla humana que se elige con validación y costos, **no** algo que venga del modelo.
""")
code("""
def umbral(P, t):
    auto = P.max(1) >= t
    mal_ = P.argmax(1) != y_te
    return {"t": t, "aceptadas sin revisión": int(auto.sum()), "errores aceptados": int((auto & mal_).sum()), "van a revisión": int((~auto).sum()),
            "errores en revisión": int((~auto & mal_).sum())}


display(pd.DataFrame([{"modelo": m, **umbral(P, t)} for m, P in (("Logística", P_lg), ("Red neuronal", P_red)) for t in (0.90, 0.99)]))
""")
md("""
Con **t = 0.90**, la red acepta [[umbral.red.0.9.auto]] imágenes sin revisión y deja pasar [[umbral.red.0.9.err_auto]] errores (solo [[umbral.red.0.9.revision]] van a revisión); la logística acepta [[umbral.log.0.9.auto]] y [[umbral.log.0.9.revision]] van a revisión, sin errores aceptados. Con **t = 0.99**, la red acepta [[umbral.red.0.99.auto]] y deja pasar [[umbral.red.0.99.err_auto]]; la logística acepta [[umbral.log.0.99.auto]]. Subir el umbral baja los errores aceptados pero sube la carga de revisión.

### ✍️ Ejercicio C2
Para la **red**, recorre los umbrales `[0.5, 0.7, 0.9, 0.95, 0.99, 0.999]` y arma una tabla con revisiones y errores aceptados. Supón que cada error aceptado cuesta 50 unidades y cada revisión 1. ¿Qué umbral minimiza el costo total?
""")
code("""
# tu código aquí
""")
solution("""
filas = []
for t in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    u = umbral(P_red, t)
    filas.append({**u, "costo": 50 * u["errores aceptados"] + 1 * u["van a revisión"]})
tabla = pd.DataFrame(filas)
display(tabla)
print("umbral de menor costo:", tabla.loc[tabla["costo"].idxmin(), "t"])
""", "El costo depende de los precios que supongas: si omitir un error es muy caro, conviene un umbral alto, pero la revisión tiene capacidad limitada. La regla se elige con validación, no con prueba.")

# ================================================================== PARTE D
md("""
## Parte D · ¿La ventaja de la red es real?

### D1 · Bootstrap de la diferencia de pérdidas (20 min)
La red tuvo menor log loss en estas 360 imágenes. ¿Cuánto podría variar esa ventaja con **otras** 360? El bootstrap reutiliza las 360 pérdidas por imagen (los dos modelos quedan **fijos**) y se remuestrean **con reposición**: B = 2 000 veces se calcula `pérdida media de la logística − pérdida media de la red`. Los percentiles 2.5 y 97.5 forman el intervalo del 95 %.
""")
code("""
dl = perdidas(y_te, P_lg) - perdidas(y_te, P_red)       # positivo = la red pierde menos en esa imagen
B = 2000
rng = np.random.default_rng(44001)
n = len(dl)
boot = np.array([dl[rng.choice(n, size=n, replace=True)].mean() for _ in range(B)])
ic = np.quantile(boot, [0.025, 0.975])
print(f"diferencia observada: {dl.mean():.4f} | IC bootstrap 95 %: [{ic[0]:.4f}; {ic[1]:.4f}]")
print("imágenes donde la red pierde menos:", int((dl > 0).sum()), "| donde pierde menos la logística:", int((dl < 0).sum()))

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(boot, bins=40, color="steelblue", alpha=0.85)
ax.axvline(dl.mean(), color="firebrick", lw=2)
for x in ic:
    ax.axvline(x, color="black", ls="--")
ax.axvline(0, color="green", lw=2)
ax.set_title("Diferencia de pérdidas medias (logística − red)")
plt.show()
""")
md("""
**Nuestra reproducción**: diferencia [[boot.dif|4]], IC del 95 % de [[boot.ic.0|4]] a [[boot.ic.1|4]] (el 0 queda fuera). **Diapositivas**: 0.0696, IC [0.0478; 0.0901]. En ambos casos el 0 no está en el intervalo: hay evidencia de menor pérdida de la red **en estas condiciones**, bajo independencia por imagen. Hay [[boot.n_log_gana]] imágenes donde la logística pierde menos, y la red gana en [[boot.n_red_gana]]. El intervalo **no** mide la utilidad práctica de la mejora ni cubre un cambio de cámara.

### ✍️ Ejercicio D1
Construye un IC bootstrap del 95 % para la **exactitud de la red** (B = 2 000, semilla 44002) y compáralo con ± 1.96·√(â(1 − â)/n).
""")
code("""
# tu código aquí
""")
solution("""
ok = (P_red.argmax(1) == y_te).astype(float)
rng2 = np.random.default_rng(44002)
ba = np.array([ok[rng2.choice(len(ok), size=len(ok), replace=True)].mean() for _ in range(2000)])
ah = ok.mean()
se = np.sqrt(ah * (1 - ah) / len(ok))
print("exactitud:", round(ah, 4), "| IC bootstrap:", np.quantile(ba, [0.025, 0.975]).round(4), "| IC normal:", [round(ah - 1.96 * se, 4), round(ah + 1.96 * se, 4)])
""", "Nuestra reproducción: exactitud [[boot_acc_red.acc|4]], IC bootstrap de [[boot_acc_red.ic.0|4]] a [[boot_acc_red.ic.1|4]], con SE bootstrap [[boot_acc_red.se|4]] frente a [[boot_acc_red.se_formula|4]] de la fórmula: coinciden porque los modelos son fijos y las imágenes se tratan como independientes.")

# ================================================================== PARTE E
md("""
## Parte E · ¿Resiste otras condiciones?

### E1 · Prueba de estrés: desplazar una columna (20 min)
Se desplaza cada imagen de **prueba** una columna a la derecha: la primera columna se rellena con 0 y la última original se descarta (diapositiva 21). Los modelos **no se reajustan**: se comprueba si toleran esta transformación. Es una prueba de estrés artificial; no representa todos los cambios de una cámara real.
""")
code("""
def desplazar(Xv, k=1):
    # corre cada imagen k columnas a la derecha (k > 0) o a la izquierda (k < 0); rellena con 0
    im = Xv.reshape(-1, 8, 8)
    out = np.zeros_like(im)
    if k > 0:
        out[:, :, k:] = im[:, :, :-k]
    else:
        out[:, :, :k] = im[:, :, -k:]
    return out.reshape(-1, 64)


Xs = desplazar(X[te], 1)
filas = []
for nombre, modelo, Po in (("Logística", lg, P_lg), ("Red neuronal", red, P_red)):
    Ps = modelo.predict_proba(Xs)
    for cond, PP in (("original", Po), ("desplazada 1 columna", Ps)):
        filas.append({"modelo": nombre, "condición": cond, "exactitud (%)": round(100 * (PP.argmax(1) == y_te).mean(), 2), "log loss": round(log_loss(y_te, PP, labels=np.arange(10)), 4)})
display(pd.DataFrame(filas))

fig, axes = plt.subplots(2, 4, figsize=(6, 3.2))
for c_, k in enumerate([int(np.where(y_te == d)[0][0]) for d in (3, 4, 8, 5)]):
    axes[0, c_].imshow(X[te][k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[1, c_].imshow(Xs[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[0, c_].set_title(f"original {y_te[k]}", fontsize=8)
    axes[1, c_].set_title(f"desplazada {y_te[k]}", fontsize=8)
for a in axes.flat:
    a.axis("off")
plt.tight_layout()
plt.show()
""")
md("""
**Nuestra reproducción**: la logística cae de [[s.log_acc]] a **[[s.dlog_acc]]** (log loss [[desp.log.ll|4]]); la red, de [[s.red_acc]] a **[[s.dred_acc]]** (log loss [[desp.red.ll|4]]). **Diapositivas**: logística 39.72 % y 2.3826 (igual a la nuestra); red 45.00 % y 3.1314. En ambos casos la red desplazada acierta más que la logística, pero su log loss es **peor**: asigna probabilidades muy pequeñas al dígito real (de sus [[desp.red.errores]] errores, [[desp.red.n_conf90_errores]] tienen p ≥ 0.90). Con 10 clases, adivinar daría 10 %. Un buen resultado original **no garantiza** resistir un cambio de entrada.

### ✍️ Ejercicio E1
Evalúa los dos modelos con las imágenes desplazadas **2 columnas a la derecha** (`desplazar(X[te], 2)`) y **1 columna a la izquierda** (`desplazar(X[te], -1)`). ¿Qué pasa con la exactitud? ¿Un desplazamiento hacia la izquierda es igual de dañino?
""")
code("""
# tu código aquí
""")
solution("""
for k in (2, -1):
    Xk = desplazar(X[te], k)
    for nombre, modelo in (("Logística", lg), ("Red neuronal", red)):
        Pk = modelo.predict_proba(Xk)
        print(f"desplazar {k:+d} | {nombre}: exactitud {100 * (Pk.argmax(1) == y_te).mean():.2f} % | log loss {log_loss(y_te, Pk, labels=np.arange(10)):.4f}")
""", "Resultados de nuestra reproducción: 2 columnas a la derecha → logística [[s.d2log_acc]] y red [[s.d2red_acc]]; 1 columna a la izquierda → logística [[s.dm1log_acc]] y red [[s.dm1red_acc]]. Con 2 columnas a la derecha el daño es mayor y la exactitud queda cerca del azar (10 %); con 1 a la izquierda la caída es parecida a la de 1 a la derecha. La lección: la prueba habla de las condiciones en que se midió.")

# ================================================================== PARTE F
md("""
## Parte F · Embeddings: significado como cercanía

### F1 · Embeddings de juguete (20 min)
Un **vector** es una lista ordenada de números. Un **embedding** representa un objeto (un texto, una imagen) con un vector para poder compararlo con otros. Los textos de la diapositiva 27, con **números inventados**: Q «cobro duplicado» = (0.20, 0.80), A «reembolso» = (0.35, 0.70), B «envíos» = (0.80, 0.20). Se compara con la **distancia** y con la **similitud coseno** (el coseno del ángulo entre vectores: 1 = misma dirección).
""")
code("""
Q, A, Bv = np.array([0.20, 0.80]), np.array([0.35, 0.70]), np.array([0.80, 0.20])


def distancia(u, v):
    return float(np.sqrt(((u - v) ** 2).sum()))


def coseno(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


for nombre, v in (("A (reembolso)", A), ("B (envíos)", Bv)):
    print(f"Q–{nombre}: distancia = {distancia(Q, v):.3f} | coseno = {coseno(Q, v):.3f}")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
for v, nm, col in ((Q, "Q", "firebrick"), (A, "A", "green"), (Bv, "B", "royalblue")):
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col))
    ax.text(v[0] + 0.02, v[1] + 0.02, nm, color=col)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Textos como vectores")
plt.show()
""")
md("""
A está a [[emb.d_qa|3]] de Q y B a [[emb.d_qb|3]]; los cosenos son [[emb.cos_qa|3]] y [[emb.cos_qb|3]] (en ese orden). **A está más cerca de Q que B** por distancia y por ángulo. Dato de la clase: de dónde salen esos números es lo importante: un modelo **aprende** una regla consistente que convierte cada texto en un vector. Una **base de datos vectorial** solo guarda los puntos cercanos cerca entre sí para encontrarlos rápido. Y la cercanía no garantiza que el texto **responda** la consulta.

### ✍️ Ejercicio F1
Vectores nuevos: Q = (3, 4), A = (4, 3), B = (−4, 3), C = (6, 8). Calcula con tus funciones la distancia y el coseno de Q con cada uno. ¿Cuál es el más cercano por distancia? ¿Y por coseno? ¿Por qué no coinciden?
""")
code("""
# tu código aquí
""")
solution("""
Q2 = np.array([3, 4])
docs = {"A": np.array([4, 3]), "B": np.array([-4, 3]), "C": np.array([6, 8])}
for nombre, v in docs.items():
    print(f"Q–{nombre}: distancia = {distancia(Q2, v):.3f} | coseno = {coseno(Q2, v):.3f}")
""", "Distancias: A = 1.414, B = 7.071, C = 5.000. Cosenos: A = 0.960, B = 0.000, C = 1.000. Por distancia gana A; por coseno, C (misma dirección que Q, solo más «largo»). El coseno ignora la longitud; la distancia no. B es perpendicular a Q (coseno 0).")

# ================================================================== PARTE G
md("""
## Parte G · Costos, carga de revisión y la unidad correcta

### G1 · Costo de errores y carga de revisión (20 min)
Diapositiva 41 (ejemplo hipotético): de 1 000 piezas, 20 tienen defectos. El sistema deja pasar **4** defectos y manda **30** piezas sanas a revisión. Omitir un defecto cuesta **50** unidades y una falsa alarma, **1**.
""")
code("""
n_piezas, defectos, omitidos, falsas = 1000, 20, 4, 30
c_omitido, c_falsa, capacidad = 50, 1, 20

costo = omitidos * c_omitido + falsas * c_falsa
detectados = defectos - omitidos
revisiones = detectados + falsas
print("costo de los errores:", omitidos, "×", c_omitido, "+", falsas, "×", c_falsa, "=", costo)
print("defectos detectados:", detectados, "| revisiones si se revisan todas las alertas:", revisiones, "| capacidad:", capacidad, "| ¿alcanza?", revisiones <= capacidad)
print("exactitud del sistema:", (n_piezas - omitidos - falsas) / n_piezas, "| «decir siempre sano»:", (n_piezas - defectos) / n_piezas, "con costo", defectos * c_omitido)
""")
md("""
El costo es **[[s.costo]]** unidades; hay [[costo.alertas]] revisiones y la capacidad es [[s.cap]]: **no alcanza** (faltan [[costo.faltan]]). La exactitud del sistema es [[s.acc_costo]], y «decir siempre sano» tendría [[s.siempre]]: **mejor exactitud, pero omitiría los 20 defectos** (costo [[costo.siempre_sano_costo]] frente a [[s.costo]]). La exactitud sola no cuenta cuánto cuesta cada error.

### ✍️ Ejercicio G1
Una línea revisa 2 000 piezas, 50 con defecto. Se omiten 5 defectos y hay 40 falsas alarmas. Omitir cuesta 80 y una falsa alarma 2; se pueden revisar 60 alertas. Calcula el costo, las revisiones necesarias y si alcanzan, y compáralo con «decir siempre sano».
""")
code("""
# tu código aquí
""")
solution("""
n2, d2, om2, fa2, co2, cf2, cap2 = 2000, 50, 5, 40, 80, 2, 60
costo2 = om2 * co2 + fa2 * cf2
rev2 = (d2 - om2) + fa2
print("costo:", costo2, "| revisiones:", rev2, "| faltan:", max(0, rev2 - cap2))
print("exactitud del sistema:", (n2 - om2 - fa2) / n2, "| «siempre sano»:", (n2 - d2) / n2, "con costo", d2 * co2)
""", "Costo 480; 85 revisiones (faltan 25); exactitud 0.9775 frente a 0.975 de «siempre sano», cuyo costo sería 4 000.")

md("""
### Cierre del bloque: la unidad correcta (caso integrador)
Diapositiva 46: una empresa tiene **6 000 fotografías de 1 200 envases** (cinco vistas cada uno). La decisión es **por envase**, así que si se separan las fotos al azar, vistas del mismo envase quedan en entrenamiento y en prueba y la prueba «ya conoce» el objeto. **Seis mil fotografías no equivalen a seis mil envases independientes.**
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre **sin ayuda** (25 min)

Sin mirar las soluciones anteriores:
1. Crea un arreglo `envase_id` con los identificadores de 6 000 fotos de 1 200 envases (cinco vistas por envase).
2. Separa **por envase** en entrenamiento / validación / prueba con **840 / 180 / 180** envases (usa `GroupShuffleSplit` de scikit-learn dos veces, con semilla).
3. Verifica con código que **ningún envase** aparece en dos conjuntos, y cuenta fotos y envases por conjunto.
4. Remuestrea con reposición **envases completos** (no fotos) dentro del conjunto de prueba y calcula, con una pérdida inventada por foto (`np.random.default_rng(1).exponential(0.3, size=900)`), un IC bootstrap de la pérdida media.
""")
code("""
# tu código aquí
""")
solution("""
from sklearn.model_selection import GroupShuffleSplit

envase_id = np.repeat(np.arange(1200), 5)                  # 6 000 fotos
idx = np.arange(len(envase_id))
g1 = GroupShuffleSplit(n_splits=1, test_size=180, random_state=44).split(idx, groups=envase_id)
i_dev2, i_te2 = next(g1)
g2 = GroupShuffleSplit(n_splits=1, test_size=180, random_state=45).split(idx[i_dev2], groups=envase_id[i_dev2])
a, b = next(g2)
i_tr2, i_va2 = i_dev2[a], i_dev2[b]

conjuntos = {"entrenamiento": i_tr2, "validación": i_va2, "prueba": i_te2}
for nombre, ii in conjuntos.items():
    print(nombre, "| fotos:", len(ii), "| envases:", len(np.unique(envase_id[ii])))
e_tr, e_va, e_te = (set(envase_id[ii]) for ii in conjuntos.values())
print("envases compartidos entre conjuntos:", len(e_tr & e_va), len(e_tr & e_te), len(e_va & e_te))

# bootstrap por envase completo en prueba
perd = np.random.default_rng(1).exponential(0.3, size=len(i_te2))
env_te = envase_id[i_te2]
envs = np.unique(env_te)
rng3 = np.random.default_rng(2)
medias = []
for _ in range(2000):
    elegidos = rng3.choice(envs, size=len(envs), replace=True)
    medias.append(np.concatenate([perd[env_te == e] for e in elegidos]).mean())
print("pérdida media:", perd.mean().round(4), "| IC bootstrap por envase:", np.quantile(medias, [0.025, 0.975]).round(4))
""", "Debes ver [[envases.tr]], [[envases.va]] y [[envases.te]] envases ([[envases.fotos_tr]], [[envases.fotos_va]] y [[envases.fotos_te]] fotos) y **0 envases compartidos**. El bootstrap sortea **envases completos** para conservar juntas sus cinco vistas.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué se define primero la tarea y luego los datos? Nombra las cuatro tareas de la clase.
2. ¿Cómo se convierte una imagen de 8 × 8 en la entrada de un modelo? ¿Qué papel tiene dividir por 16?
3. Un modelo da 0.60 al 3 y 0.25 al 8; el real es 8. ¿Acierta? ¿Cuál es la pérdida de esa imagen?
4. ¿Qué mide el log loss y por qué complementa a la exactitud?
5. ¿Por qué se elige con validación y se evalúa una sola vez en prueba? ¿Por qué se reajusta con el desarrollo?
6. ¿Qué es un umbral de aceptación automática y por qué una probabilidad alta no garantiza el acierto?
7. ¿Qué se remuestrea en el bootstrap de la diferencia de pérdidas y qué se mantiene fijo?
8. El IC bootstrap de la diferencia no contiene 0. ¿Qué concluyes y qué no?
9. ¿Qué muestra desplazar las imágenes una columna?
10. ¿Qué es un embedding? ¿Cómo se comparan dos? ¿Por qué cercano no significa «responde»?
11. ¿Cómo se calculan el costo de los errores y la carga de revisión? ¿Por qué la exactitud sola engaña?
12. ¿Por qué hay que separar por envase (o por persona) y no por foto?

<details><summary><b>Respuestas</b></summary>

1. Porque el objetivo fija qué datos se necesitan y cómo evaluar; si se parte de los datos se adapta el método a lo que hay. Clasificar, pronosticar, recuperar y generar.
2. Se leen las 64 intensidades (0 a 16) fila por fila y se dividen por 16 para dejarlas entre 0 y 1; la etiqueta va aparte.
3. No: predice 3. La pérdida es −ln(0.25) = 1.386.
4. La pérdida media de −ln(p del dígito real). Castiga los errores muy confiados; dos modelos con exactitud casi igual pueden tener log loss muy distinto.
5. Elegir con validación mantiene la prueba independiente; reajustar con desarrollo aprovecha más datos una vez decidido el modelo; la prueba se mira al final y una sola vez.
6. Una regla de corte (p ≥ t) para aceptar sin revisión; la probabilidad alta no es garantía individual (en la clase hubo errores con p entre 0.76 y 0.94).
7. Se remuestrean con reposición las pérdidas por imagen de prueba; los modelos (y sus predicciones) quedan fijos.
8. Evidencia de menor pérdida de un modelo en estas condiciones (bajo independencia por imagen); no que sea mejor con otra cámara ni que la mejora sea útil en la práctica.
9. Que un buen resultado original no garantiza robustez: ambos modelos caen mucho.
10. Un vector que representa un objeto; se comparan con distancia o coseno; porque un texto relacionado puede no responder la consulta.
11. Σ cantidad × costo por caso; revisiones = defectos detectados + falsas alarmas. La exactitud no incorpora los costos distintos ni la capacidad.
12. Porque las vistas de un mismo objeto no son independientes: si quedan en conjuntos distintos, la prueba ya conoce el objeto y sobreestima el desempeño.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** el 20 de octubre (Clase 45) se construye una red neuronal: capas, neuronas, funciones de activación y cómo aprende. Revisa `Guias/apuntes/C44_Ejercicios_Cuaderno_AplicacionesIA_DeepLearning.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
