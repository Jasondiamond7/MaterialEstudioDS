"""Genera el notebook guiado de la Clase 46 (Deep Learning, autoencoders y representaciones).

Uso normal:      python notebook_c46.py             -> Guias/notebooks/C46_Autoencoders.ipynb
Modo de prueba:  C46_TEST=1 python notebook_c46.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c46_numeros.json (lo escribe figuras_c46.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib y scikit-learn.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

TEST = os.environ.get("C46_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C46_Autoencoders_test.ipynb" if TEST else "C46_Autoencoders.ipynb")
N = json.loads((HERE / "c46_numeros.json").read_text(encoding="utf-8"))
pc = lambda x: f"{x * 100:.2f} %"
EX = N["extra"]
_pd = N["umbral"]["por_digito"]
N["s"] = dict(pix_pct=pc(N["pixel"]["cuadrado"] / N["pixel"]["suma"]), red_media=pc(N["mse"]["reduccion_pca_vs_media"]), red_prof=pc(N["mse"]["reduccion_prof_vs_pca"]), tp=pc(N["umbral"]["prueba_pct"]), td=pc(N["umbral"]["desplazadas_pct"]),
              cociente=f"{N['mse']['media']['val'] / N['mse']['prof']['val']:.2f}")
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
# Clase 46 · Deep Learning, autoencoders y representaciones
**Diplomado Data Science** · Prof. Eloy Alvarado · la red ya no clasifica: aprende a resumir una imagen en 8 números y a volver a armarla, con los dígitos escritos a mano de las Clases 44 y 45

Material de apoyo:
- `Guias/C46_Guia_Autoencoders.pdf` — guía visual de la clase.
- `Guias/apuntes/C46_Apuntes_Autoencoders.pdf` — apuntes explicados paso a paso, con ejemplos resueltos.
- `Guias/apuntes/C46_Ejercicios_Cuaderno_Autoencoders.pdf` — ejercicios en papel (antes o después de este notebook).
- Diapositivas y programa del profesor (carpeta «Clase 46 …»): `autoencoders.pdf`, `autoencoders.py`, `digitos.csv`.
- Transcripción: `VideosClases/Clase 46 … .txt` (versión corregida; cubre las diapositivas 1 a 14, el video termina en el recreo).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scikit-learn` y `threadpoolctl`. Si no los tienes:
```
pip install numpy pandas matplotlib scikit-learn threadpoolctl
```

**Datos**: `digitos.csv` de la carpeta de la Clase 46 (1 797 imágenes de 8 × 8 píxeles, columnas `px00`…`px63`, `digito` y `conjunto`: **1 077 entrenamiento / 360 validación / 360 prueba**). Si el notebook no lo encuentra, usa `load_digits()` y arma una partición propia (las cifras cambian un poco).

**Cómo leer los números**: la receta de entrenamiento es **la misma del profesor** (`autoencoders.py`): 64–8–64 y 64–32–8–32–64, tanh en las capas ocultas y salida lineal, Adam con tasa 0.002, lote 64, alpha 0.0001, 150 épocas, semilla 20261022, elección por MSE de validación. Con scikit-learn 1.9.1 y `digitos.csv` las cifras coinciden **exactamente** con las diapositivas. Lo marcado como *nuestra reproducción* (juguetes, código de signos, práctica con latente 4) es cálculo propio.

### Objetivos
1. Calcular **a mano, con numpy**, un autoencoder de juguete: encoder, vector latente, decoder y MSE.
2. Entender por qué **resumir es perder** y por qué 8 números reales no son 8 bits.
3. Calcular el **MSE por píxel, por imagen y por conjunto**, y compararlo con la **imagen media** y con **PCA de 8 componentes**.
4. Entrenar con `MLPRegressor` (entrada = salida) la receta del profesor y **elegir la arquitectura con validación**.
5. **Abrir el autoencoder**: extraer el vector latente de 8 valores con `coefs_` e `intercepts_` y contar parámetros.
6. Usar el error de reconstrucción como **puntaje de alerta**, fijar un **umbral** y ver sus límites.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 25 | A1 · Autoencoder de juguete 2 → 1 → 2, A2 · Una red 3 → 2 → 3 con tanh |
| B | 20 | B1 · Datos y partición, B2 · MSE a mano y referencias (media y PCA) |
| C | 30 | C1 · Entrenar con `MLPRegressor`, C2 · Curvas, C3 · Comparar con las referencias |
| D | 30 | D1 · Abrir la red: vector latente y parámetros, D2 · Un píxel y una imagen, D3 · Imágenes reconstruidas |
| E | 25 | E1 · Puntaje y umbral, E2 · Prueba de estrés y límites |
| — | 25 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Lo aleatorio usa semilla fija; con otras versiones de las librerías puede variar la última cifra.
- El entrenamiento completo (dos redes × 150 épocas) tarda unos segundos en una CPU normal.
""")

md("## Preparación")
code("""
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
np.set_printoptions(suppress=True, precision=4)
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso
- **Red neuronal** (Clase 45): una neurona calcula a = Σ w·x + b y aplica una activación; una capa son varias neuronas, con una fila por observación (X·W + b).
- **tanh** lleva cualquier número a (−1, 1); una **salida lineal** no cambia el valor.
- **Parámetros de una capa** = unidades × (entradas + 1).
- **Validación y prueba** (Clases 30 y 38): se elige con validación; la prueba se mira al final.
- **PCA** (Clase 32): proyecta a k direcciones y se puede reconstruir: x̂ = x̄ + (x − x̄)·P·Pᵀ.
""")

# ================================================================== PARTE A
md("""
## Parte A · Un autoencoder de juguete (con numpy)

### A1 · 2 píxeles → 1 número → 2 píxeles (diapositivas 11, 12 y 13)
Un **autoencoder** tiene un **encoder** h = f(x) que resume la entrada y un **decoder** x̂ = g(h) que intenta reconstruirla. En el ejemplo de la clase, las imágenes tienen **2 píxeles**; el encoder fijo guarda solo el **promedio** h = (x₁ + x₂)/2 y el decoder fijo devuelve (h, h). El error de una imagen es el promedio de las diferencias al cuadrado: **ℓ = mean((x̂ − x)²)**.
""")
code("""
def encoder_promedio(x):
    return (x[0] + x[1]) / 2


def decoder_repite(h):
    return np.array([h, h])


def mse(a, b):
    # promedio de (b - a)^2: ℓ de una imagen, o L de varias
    return float(np.mean((np.asarray(b) - np.asarray(a)) ** 2))


for x in (np.array([0.2, 0.8]), np.array([0.8, 0.2])):
    h = encoder_promedio(x)
    x_hat = decoder_repite(h)
    print(f"x = {x} -> h = {h:.2f} -> x̂ = {x_hat} | MSE a mano = {mse(x, x_hat):.4f} | sklearn = {mean_squared_error(x, x_hat):.4f}")

print("\\nTres reconstrucciones propuestas para la referencia (0.2, 0.8):")
ref = np.array([0.2, 0.8])
for prop in ([0.2, 0.8], [0.3, 0.7], [0.5, 0.5]):
    print(prop, "-> MSE =", round(mse(ref, prop), 4))
""")
md("""
Dos entradas **distintas** producen el **mismo** código h = 0.5, así que el decoder no puede distinguirlas: se **perdió el orden**. El MSE de ambas es **0.09**, y la propuesta (0.3, 0.7) tendría solo **0.01**. La copia perfecta (0.2, 0.8) da 0, pero **no resume nada**: no es la meta.

Otro encoder del mismo tamaño puede conservar lo que este perdió. **Ejemplo propio (no es de la clase)**: h = x₁ − x₂ con decoder (0.5 + h/2, 0.5 − h/2) devuelve exactamente (0.2, 0.8) y (0.8, 0.2) (solo sirve para pares que suman 1).
""")
code("""
def encoder_dif(x):
    return x[0] - x[1]


def decoder_dif(h):
    return np.array([0.5 + h / 2, 0.5 - h / 2])


for x in (np.array([0.2, 0.8]), np.array([0.8, 0.2])):
    h = encoder_dif(x)
    x_hat = decoder_dif(h)
    print(f"x = {x} -> h = {h:+.2f} -> x̂ = {np.round(x_hat, 4)} | MSE = {mse(x, x_hat):.4f}")
""")
md("""
### ✍️ Ejercicio A1
Con el encoder h = (x₁ + x₂)/2 y el decoder (h, h), calcula h, x̂ y el MSE para x = (0.1, 0.9), x = (0.6, 0.4) y x = (0.3, 0.3). ¿Cuáles comparten vector latente? ¿Alguna se reconstruye sin error?
""")
code("""
# tu código aquí
""")
solution("""
for x in (np.array([0.1, 0.9]), np.array([0.6, 0.4]), np.array([0.3, 0.3])):
    h = encoder_promedio(x)
    x_hat = decoder_repite(h)
    print(x, "-> h =", round(h, 2), "-> x̂ =", x_hat, "-> MSE =", round(mse(x, x_hat), 4))
""", "Debes ver MSE 0.16, 0.01 y 0.0. Las dos primeras comparten h = 0.5 (el decoder no puede distinguirlas). La tercera no tiene error porque sus dos píxeles son iguales y el decoder (h, h) los repite.")

md("""
### A2 · Una red 3 → 2 → 3 con tanh (diapositivas 8, 9 y 16)
En la clase el encoder y el decoder son **capas de una red neuronal**: capas ocultas con **tanh** y salida **lineal**. Con la convención de siempre (una **fila** por imagen): **a = x·W₁ + b₁**, **h = tanh(a)**, **x̂ = h·W₂ + b₂**. Aquí los pesos son números simples elegidos a mano (ejemplo propio).
""")
code("""
x = np.array([[1.0, 0.0, 0.5]])                 # una imagen de 3 píxeles (1 fila)
W1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])   # columnas = unidades del encoder
b1 = np.zeros(2)
W2 = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, -1.0]])     # 2 unidades -> 3 píxeles
b2 = np.zeros(3)

a = x @ W1 + b1
h = np.tanh(a)                  # vector latente (2 números)
x_hat = h @ W2 + b2             # salida lineal
print("a =", a, "\\nh =", h, "\\nx̂ =", x_hat)
print("MSE =", round(mse(x, x_hat), 4))
print("parámetros:", W1.size + b1.size + W2.size + b2.size, "= (3·2 + 2) + (2·3 + 3)")
print("tanh(0), tanh(1), tanh(-2):", np.tanh([0.0, 1.0, -2.0]))
""")
md("""
Con 3 píxeles pasamos a 2 números: ya se **perdió información**. La salida lineal puede salirse de [0, 1]: el MSE usa esas salidas completas y solo la visualización las recorta.

### ✍️ Ejercicio A2
Una imagen de 3 píxeles x = (0.8, 0.2, 0.4) pasa por una red 3 → 2 → 3 con W₁ = [[0.5, −1], [1, 0.5], [−0.5, 1]], b₁ = (0.1, −0.1), W₂ = [[1, 0, −1], [0.5, 1, 0.5]] y b₂ = 0. Calcula a, h = tanh(a), x̂ y el MSE. ¿Cuántos parámetros tiene la red?
""")
code("""
# tu código aquí
""")
_x = np.array([[0.8, 0.2, 0.4]])
_W1 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
_b1 = np.array([0.1, -0.1])
_W2 = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]])
_a = _x @ _W1 + _b1
_h = np.tanh(_a)
_xh = _h @ _W2
_m = float(np.mean((_xh - _x) ** 2))
solution("""
x = np.array([[0.8, 0.2, 0.4]])
W1 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
b1 = np.array([0.1, -0.1])
W2 = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]])
a = x @ W1 + b1
h = np.tanh(a)
x_hat = h @ W2
print("a =", a, "h =", h, "x̂ =", x_hat, "MSE =", round(mse(x, x_hat), 4))
print("parámetros:", W1.size + b1.size + W2.size + 3)
""", f"Debes ver a = ({_a[0, 0]:.1f}, {_a[0, 1]:.1f}), h = ({_h[0, 0]:.4f}, {_h[0, 1]:.4f}), x̂ = ({_xh[0, 0]:.4f}, {_xh[0, 1]:.4f}, {_xh[0, 2]:.4f}) y MSE = {_m:.4f}. Parámetros: 3·2 + 2 + 2·3 + 3 = 17 (los tres sesgos de la salida cuentan aunque valgan 0).".replace("-", "−"))

# ================================================================== PARTE B
md("""
## Parte B · Los datos y el error de reconstrucción

### B1 · Datos y partición (diapositivas 5 y 22)
Cada imagen son 64 intensidades (0 a 16) que se dividen por 16 para quedar entre 0 y 1 (**0 = blanco, 1 = negro**). La etiqueta del dígito **no se usa** para entrenar el autoencoder. La partición del profesor está en la columna `conjunto` de `digitos.csv`.
""")
code("""
def buscar_csv():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 46*/digitos.csv"):
            return p
        if (base / "digitos.csv").exists():
            return base / "digitos.csv"
    return None


ruta = buscar_csv()
if ruta is not None:
    datos = pd.read_csv(ruta)
    X = datos.filter(regex="^px").to_numpy(dtype=float) / 16
    y = datos["digito"].to_numpy(dtype=int)
    ids = datos["id"].to_numpy(dtype=int)
    tr, va, te = (datos["conjunto"].eq(k).to_numpy() for k in ("entrenamiento", "validacion", "prueba"))
    print("Uso digitos.csv y la partición del profesor.")
else:
    from sklearn.model_selection import train_test_split
    dig = load_digits()
    X, y = dig.data / 16, dig.target
    ids = np.arange(1, len(y) + 1)
    idx = np.arange(len(y))
    i_dev, i_te = train_test_split(idx, test_size=360, stratify=y, random_state=46)
    i_tr, i_va = train_test_split(i_dev, test_size=360, stratify=y[i_dev], random_state=46)
    tr, va, te = (np.isin(idx, i) for i in (i_tr, i_va, i_te))
    print("⚠️ No encontré digitos.csv: uso una partición propia; las cifras serán algo distintas a las del texto.")

print("entrada:", X.shape, "| entrenamiento:", tr.sum(), "| validación:", va.sum(), "| prueba:", te.sum())
print("valores entre", X.min(), "y", X.max())
""")
code("""
fig, axes = plt.subplots(1, 6, figsize=(9, 1.9))
for ax, k in zip(axes, range(6)):
    j = np.where(te & (y == k))[0][0]
    ax.imshow(X[j].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"dígito {k}", fontsize=9)
    ax.axis("off")
fig.suptitle("Primer caso de cada dígito 0 a 5 en prueba", fontsize=10)
plt.show()
""")
md("""
### B2 · El MSE a mano y dos referencias: la imagen media y PCA con 8 componentes (diapositivas 12, 15, 27 y 30)
Para una imagen, **ℓ = (1/64)·Σ (x̂ⱼ − xⱼ)²**; para un conjunto, el promedio de los ℓ de sus imágenes (es lo que calcula `mean_squared_error` sobre dos matrices). Un MSE aislado no dice si es bueno: se compara con dos referencias que se **ajustan solo con entrenamiento**:
- **Imagen media**: ignora la entrada y devuelve siempre el promedio de las imágenes de entrenamiento.
- **PCA con 8 componentes**: z = (x − x̄)·Pₖ y x̂ = x̄ + z·Pₖᵀ (reconstrucción **lineal** con 8 números).
""")
code("""
X_tr, X_va, X_te = X[tr], X[va], X[te]

# 1) Imagen media (aprendida con entrenamiento)
media = X_tr.mean(axis=0)
rec_media = np.tile(media, (len(X_va), 1))

# 2) PCA con 8 componentes (aprendido con entrenamiento)
pca = PCA(n_components=8, svd_solver="full").fit(X_tr)
rec_pca = pca.inverse_transform(pca.transform(X_va))

mse_media = np.mean((X_va - rec_media) ** 2)
mse_pca = np.mean((X_va - rec_pca) ** 2)
print("MSE de validación a mano   -> media:", round(mse_media, 4), "| PCA 8:", round(mse_pca, 4))
print("MSE de validación sklearn  -> media:", round(mean_squared_error(X_va, rec_media), 4), "| PCA 8:", round(mean_squared_error(X_va, rec_pca), 4))

# MSE por imagen: un número por fila
err_pca = np.mean((X_va - rec_pca) ** 2, axis=1)
print("MSE por imagen (PCA 8): forma", err_pca.shape, "| promedio", round(err_pca.mean(), 4), "= el MSE del conjunto")
""")
md("""
En la clase (diapositiva 27): imagen media **[[mse.media.val|4]]** y PCA 8 **[[mse.pca8.val|4]]** en validación; **[[mse.media.prueba|4]]** y **[[mse.pca8.prueba|4]]** en prueba. PCA 8 reduce el error de la imagen media en **[[s.red_media]]**. Menor MSE es mejor.

### ✍️ Ejercicio B2
Calcula el MSE de validación de PCA con k = 1, 2, 4, 16 y 32 componentes (ajustando con entrenamiento). ¿Qué pasa con el error al aumentar k? ¿Cuánto vale con k = 64?
""")
code("""
# tu código aquí
""")
solution("""
for k in (1, 2, 4, 16, 32, 64):
    p = PCA(n_components=k, svd_solver="full").fit(X_tr)
    r = p.inverse_transform(p.transform(X_va))
    print(f"k = {k:2d} -> MSE validación = {np.mean((X_va - r) ** 2):.5f}")
""", "Resultados de nuestra corrida: k = 1 → [[pca_k.1|4]]; k = 2 → [[pca_k.2|4]]; k = 4 → [[pca_k.4|4]]; k = 16 → [[pca_k.16|4]]; k = 32 → [[pca_k.32|4]]; con k = 64 el error es prácticamente 0 (se conserva toda la información). Más dimensiones, menos error de reconstrucción.")

# ================================================================== PARTE C
md("""
## Parte C · El autoencoder con `MLPRegressor` (entrada = salida)

### C1 · La receta del profesor (diapositivas 17, 19, 23 y 26)
Un autoencoder es una red neuronal normal: `MLPRegressor(...)` con `fit(X, X)`, es decir, **la misma matriz X como entrada y como objetivo**. `hidden_layer_sizes=(8,)` significa 64 → 8 → 64 y `(32, 8, 32)`, 64 → 32 → 8 → 32 → 64: solo se indican las capas ocultas. Receta: tanh en las ocultas, salida lineal (la que usa `MLPRegressor`), Adam con tasa 0.002, lotes de 64, alpha = 0.0001 y hasta 150 épocas. Cada época se hace con `partial_fit`; tras cada una se mide la **validación sin actualizar pesos** y se guarda una copia (`deepcopy`) de la mejor época.
""")
code("""
RECETA = dict(activation="tanh", solver="adam", learning_rate_init=0.002, batch_size=64, alpha=0.0001, early_stopping=False)
SEMILLA = 20261022
EPOCAS = 150


def entrenar_ae(capas, X_ajuste, X_val, epocas=EPOCAS, semilla=SEMILLA):
    red = MLPRegressor(hidden_layer_sizes=capas, random_state=semilla, **RECETA)
    mejor, mejor_epoca, copia, curva_tr, curva_va = np.inf, 0, None, [], []
    with threadpool_limits(limits=1):
        for epoca in range(1, epocas + 1):
            red.partial_fit(X_ajuste, X_ajuste)            # X entra y X es el objetivo
            mse_tr = mean_squared_error(X_ajuste, red.predict(X_ajuste))
            mse_va = mean_squared_error(X_val, red.predict(X_val))
            curva_tr.append(mse_tr)
            curva_va.append(mse_va)
            if mse_va < mejor:                             # la validación solo compara
                mejor, mejor_epoca, copia = mse_va, epoca, deepcopy(red)
    return dict(modelo=copia, epoca=mejor_epoca, mejor=mejor, tr=curva_tr, va=curva_va)


def n_parametros(m):
    return int(sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_)))


R = {nombre: entrenar_ae(capas, X_tr, X_va) for nombre, capas in (("64-8-64", (8,)), ("64-32-8-32-64", (32, 8, 32)))}
for nombre, r in R.items():
    print(f"{nombre:14s} | mejor época {r['epoca']:3d} | MSE validación {r['mejor']:.5f} | parámetros {n_parametros(r['modelo']):5d}")

elegida = min(R, key=lambda k: R[k]["mejor"])
modelo = R[elegida]["modelo"]
print("\\nElegida con validación:", elegida)
if R[elegida]["epoca"] == EPOCAS:
    print("El mínimo coincide con el límite de épocas: el presupuesto limita la conclusión (no demuestra convergencia).")
""")
md("""
Con scikit-learn 1.9.1 y la semilla de la clase, los mínimos de validación son **[[curvas.AE8.val|5]]** (64–8–64) y **[[curvas.AE32_8_32.val|5]]** (64–32–8–32–64), ambos en la época **150**: igual que las diapositivas 25 y 26. Parámetros: 520 + 576 = **1 096** y 2 080 + 264 + 288 + 2 112 = **4 744** (diapositiva 18).

### C2 · Curvas de aprendizaje (diapositiva 25)
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (nombre, r) in zip(axes, R.items()):
    ax.plot(range(1, EPOCAS + 1), r["tr"], label="Entrenamiento")
    ax.plot(range(1, EPOCAS + 1), r["va"], label="Validación")
    ax.scatter([r["epoca"]], [r["mejor"]], color="red", zorder=3)
    ax.set(title=nombre, xlabel="época", ylim=(0.015, 0.05))
axes[0].set_ylabel("MSE por píxel (sin penalización)")
axes[0].legend()
plt.tight_layout()
plt.show()
print("Época 1 -> MSE de validación:", {k: round(r["va"][0], 4) for k, r in R.items()})
""")
md("""
Entrenamiento y validación casi coinciden y ambas curvas **seguían bajando** al terminar: no se ve sobreajuste en este presupuesto, pero tampoco hay prueba de convergencia.

### C3 · Comparar con las referencias (diapositiva 27)
""")
code("""
def mse_conjunto(f_rec, Xc):
    return float(np.mean((Xc - f_rec(Xc)) ** 2))


metodos = {
    "Imagen media": lambda Z: np.tile(media, (len(Z), 1)),
    "PCA 8": lambda Z: pca.inverse_transform(pca.transform(Z)),
    "AE 64-8-64": R["64-8-64"]["modelo"].predict,
    "AE 64-32-8-32-64": R["64-32-8-32-64"]["modelo"].predict,
}
tabla = pd.DataFrame({"validación": {k: mse_conjunto(f, X_va) for k, f in metodos.items()},
                      "prueba histórica": {k: mse_conjunto(f, X_te) for k, f in metodos.items()}})
print(tabla.round(4))
print("\\nEl AE profundo reduce el error de PCA 8 en", round(100 * (1 - tabla.loc["AE 64-32-8-32-64", "validación"] / tabla.loc["PCA 8", "validación"]), 1), "% (validación)")
""")
md("""
**Lectura**: imagen media **[[mse.media.val|4]]** > PCA 8 **[[mse.pca8.val|4]]** ≈ AE 64–8–64 **[[mse.ae8.val|4]]** > AE profundo **[[mse.prof.val|4]]** en validación; el AE profundo reduce el error de PCA 8 en **[[s.red_prof]]**. El 64–8–64 queda casi igual que PCA 8 y la profundidad mejora un poco. Pero: una corrida y una semilla no establecen superioridad general, y la prueba ya se mostró en clase (comparación retrospectiva). La diapositiva 27 solo responde «¿reconstruye?»; aún falta evaluar la utilidad del código.

### ✍️ Ejercicio C
Entrena la misma receta con un vector latente de **16** (`(16,)`, es decir, 64 → 16 → 64) y compáralo con PCA de 16 componentes en validación. ¿Cuántos parámetros tiene la red?
""")
code("""
# tu código aquí
""")
solution("""
r16 = entrenar_ae((16,), X_tr, X_va)
p16 = PCA(n_components=16, svd_solver="full").fit(X_tr)
mse_p16 = np.mean((X_va - p16.inverse_transform(p16.transform(X_va))) ** 2)
print("AE 64-16-64:", round(r16["mejor"], 5), "(época", r16["epoca"], ") | PCA 16:", round(mse_p16, 5), "| parámetros:", n_parametros(r16["modelo"]))
""", "Resultados de nuestra corrida: AE 64–16–64 llega a [[extra.ae16.val|5]] (época [[extra.ae16.epoca]]) y PCA 16 a [[extra.pca16_val|5]]; parámetros: 16·65 + 64·17 = [[extra.ae16.params]]. Con 16 coordenadas el autoencoder de una capa sigue casi igual que PCA (mínimo en la última época otra vez).")

# ================================================================== PARTE D
md("""
## Parte D · Abrir el autoencoder

### D1 · El vector latente y los parámetros (diapositivas 8, 9, 18 y 24)
`MLPRegressor.predict` devuelve la **reconstrucción** (la salida de la red). Para ver el **vector latente** hay que recorrer solo las capas del encoder con los pesos aprendidos: `coefs_[j]` es la matriz de la capa j (entradas × unidades) e `intercepts_[j]` sus sesgos. Con tanh, **h = tanh(…tanh(x·W₀ + b₀)… )** hasta la capa central; el decoder continúa con las capas siguientes y una **salida lineal**.
""")
code("""
def codificar(m, Xc):
    H = Xc.copy()
    for j in range(len(m.hidden_layer_sizes) // 2 + 1):      # capas del encoder hasta la central
        H = np.tanh(H @ m.coefs_[j] + m.intercepts_[j])
    return H


def decodificar(m, H):
    n = len(m.coefs_)
    for j in range(len(m.hidden_layer_sizes) // 2 + 1, n):   # capas posteriores a la central
        H = H @ m.coefs_[j] + m.intercepts_[j]
        if j < n - 1:
            H = np.tanh(H)                                   # ocultas: tanh; la salida queda lineal
    return H


H = codificar(modelo, X)
print("vectores latentes: una fila por imagen y una columna por coordenada ->", H.shape)
print("decodificar(codificar(x)) == predict(x):", np.allclose(decodificar(modelo, H), modelo.predict(X)))
print("coordenadas entre", H.min().round(2), "y", H.max().round(2), "(tanh acota a (-1, 1))")

print("\\nCapas de la red elegida (entradas x unidades) y parámetros:")
for j, (w, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_)):
    print(f"  capa {j}: W {w.shape} + b {b.shape} = {w.size + b.size}")
print("  total =", n_parametros(modelo))

i9 = int(np.where(ids == 9)[0][0])
print("\\nVector latente del 8 de la diapositiva 9 (id 9):", H[i9].round(2), "| suma:", H[i9].sum().round(2))
""")
md("""
Debes ver las mismas 8 coordenadas de la diapositiva 9 (−0.52, −0.01, 0.04, −0.14, −0.56, −0.27, −0.07, −0.34). **No son probabilidades** ni tienen que sumar 1; son salidas de tanh y no tienen un significado individual (a diferencia de las componentes de PCA, que se ordenan por importancia). Cada imagen produce otras 8 coordenadas: los pesos no cambian, cambia la entrada.

### ✍️ Ejercicio D1
Para el 3 de la diapositiva 5 (id 4) y el 8 de la diapositiva 9 (id 9), imprime sus vectores latentes y calcula **cuántas coordenadas son negativas** en cada uno. ¿Se parecen? ¿Qué distancia euclídea hay entre ambos?
""")
code("""
# tu código aquí
""")
solution("""
i4 = int(np.where(ids == 4)[0][0])
for nombre, i in (("3 (id 4)", i4), ("8 (id 9)", i9)):
    print(nombre, H[i].round(2), "| coordenadas negativas:", int((H[i] < 0).sum()))
print("distancia euclídea:", round(float(np.linalg.norm(H[i4] - H[i9])), 3))
""", "El vector del 3 tiene 6 coordenadas negativas y el del 8, 7 (solo la coordenada 3 es positiva en el 8): ninguna coordenada, sola, «dice» el dígito. La distancia entre ambos depende de las 8 a la vez; comparar vectores latentes es lo que usa la búsqueda de imágenes parecidas (diapositiva 48).")

md("""
### D2 · Un píxel, una imagen: el MSE de la diapositiva 29
La diapositiva 5 marca el píxel de la **fila 3, columna 4** del 3 con id 4: vale **0.8125** y se reconstruye como **0.4089**. Su aporte al error es (0.4089 − 0.8125)² ≈ **0.1629**; el MSE de la imagen es el **promedio de los 64 cuadrados**: **0.017220**.
""")
code("""
i4 = int(np.where(ids == 4)[0][0])
x4 = X[i4]
r4 = modelo.predict(x4[None])[0]
cuadrados = (r4 - x4) ** 2                               # 64 contribuciones
print("píxel (fila 3, columna 4):", x4[2 * 8 + 3], "->", round(r4[2 * 8 + 3], 4))
print("(0.4089 - 0.8125)^2 =", round((r4[2 * 8 + 3] - x4[2 * 8 + 3]) ** 2, 4))
print("suma de los 64 cuadrados:", round(cuadrados.sum(), 4), "| MSE de la imagen:", round(cuadrados.mean(), 6))
print("el píxel más costoso es el", int(cuadrados.argmax()), "(posición", divmod(int(cuadrados.argmax()), 8), "en fila, columna desde 0)")
print("peso de ese píxel en la suma:", round(100 * cuadrados.max() / cuadrados.sum(), 1), "%")

fig, axes = plt.subplots(1, 3, figsize=(8.5, 3))
for ax, im, t, kw in ((axes[0], x4, "original", dict(cmap="gray_r", vmin=0, vmax=1)),
                      (axes[1], np.clip(r4, 0, 1), "reconstrucción", dict(cmap="gray_r", vmin=0, vmax=1)),
                      (axes[2], cuadrados, "error² por píxel", dict(cmap="magma_r"))):
    h_ = ax.imshow(im.reshape(8, 8), **kw)
    ax.set_title(t)
    ax.axis("off")
plt.colorbar(h_, ax=axes[2], fraction=0.046)
plt.show()
""")
md("""
**Lectura**: un solo píxel concentra el **[[s.pix_pct]]** de la suma de errores de la imagen ([[pixel.cuadrado|4]] de [[pixel.suma|4]]). El MSE solo dice «cuánto»: para ver «dónde» hay que mirar el mapa de error.

### ✍️ Ejercicio D2
Calcula el MSE por imagen de las 360 imágenes de validación con el autoencoder elegido (un número por imagen). ¿Cuál es el MSE medio, la mediana y la imagen con peor reconstrucción (su id y su dígito)?
""")
code("""
# tu código aquí
""")
solution("""
err_va = np.mean((X_va - modelo.predict(X_va)) ** 2, axis=1)
print("MSE medio (= MSE del conjunto):", round(err_va.mean(), 5), "| mediana:", round(np.median(err_va), 5))
peor = int(err_va.argmax())
print("peor imagen de validación: id", ids[va][peor], "| dígito", y[va][peor], "| MSE", round(err_va[peor], 4))
""", "Debes ver un MSE medio de [[mse.prof.val|5]] (el de la diapositiva 27), una mediana de [[umbral.mediana_val|5]] y un máximo de [[umbral.max_val|4]]: la distribución tiene cola derecha (pocas imágenes mal reconstruidas).")

md("""
### D3 · Imágenes reconstruidas (diapositiva 28)
""")
code("""
casos = [int(np.where(te & (y == k))[0][0]) for k in range(6)]
filas = [("Original", X[casos]), ("Imagen media", np.tile(media, (6, 1))), ("PCA 8", pca.inverse_transform(pca.transform(X[casos]))),
         ("AE 64-8-64", R["64-8-64"]["modelo"].predict(X[casos])), ("AE profundo", modelo.predict(X[casos]))]
fig, axes = plt.subplots(len(filas), 6, figsize=(8.5, 7.2))
for r_, (nombre, im) in enumerate(filas):
    for c_ in range(6):
        ax = axes[r_, c_]
        ax.imshow(np.clip(im[c_].reshape(8, 8), 0, 1), cmap="gray_r", vmin=0, vmax=1)   # solo la visualización recorta a [0, 1]
        ax.set_xticks([])
        ax.set_yticks([])
        if c_ == 0:
            ax.set_ylabel(nombre, rotation=0, ha="right", va="center", fontsize=9)
        if r_ > 0:
            ax.set_xlabel(f"MSE {np.mean((X[casos][c_] - im[c_]) ** 2):.3f}", fontsize=8)
plt.tight_layout()
plt.show()
""")
md("""
### ¿Cuánto es «8 números»? Reales frente a bits (nuestra reproducción)
Cada coordenada es un número real entre −1 y 1; **ocho coordenadas reales no equivalen a ocho bits** (diapositiva 10). Un experimento sencillo: reemplazar el código por su **signo** (±1, es decir, 8 bits por imagen) y decodificar **sin reentrenar**.
""")
code("""
H_va = codificar(modelo, X_va)
rec_real = decodificar(modelo, H_va)
rec_bits = decodificar(modelo, np.where(H_va >= 0, 1.0, -1.0))
rec_1dec = decodificar(modelo, np.round(H_va, 1))
print("MSE de validación con el código real      :", round(mean_squared_error(X_va, rec_real), 4))
print("MSE de validación con el código a 1 decimal:", round(mean_squared_error(X_va, rec_1dec), 4))
print("MSE de validación con solo los signos (±1) :", round(mean_squared_error(X_va, rec_bits), 4))
print("códigos de signos posibles: 2^8 =", 2 ** 8, "| distintos en validación:", len({tuple(f) for f in np.where(H_va >= 0, 1, -1)}))
""")
md("""
Con el código real el MSE es **[[bits.val_real|4]]**; a 1 decimal, **[[bits.val_redondeo1|4]]**; con solo los signos sube a **[[bits.val_bits|4]]**, peor que la imagen media (**[[mse.media.val|4]]**), y hay solo **[[bits.patrones_usados]]** patrones distintos. Esto **no** mide cuántos bits «vale» el código (el decoder se entrenó con reales y no se reentrenó para códigos binarios): solo muestra que los números reales llevan información fina.

### ✍️ Ejercicio D3
Toma el código del 3 (id 4) y súmale 0.05 a su primera coordenada. Decodifícalo y calcula (a) cuánto cambia la reconstrucción (diferencia máxima en un píxel) y (b) el MSE de esa nueva reconstrucción frente a la imagen original.
""")
code("""
# tu código aquí
""")
solution("""
h = H[i4].copy()
h_nuevo = h.copy()
h_nuevo[0] += 0.05
r_a = decodificar(modelo, h[None])[0]
r_b = decodificar(modelo, h_nuevo[None])[0]
print("cambio máximo en un píxel:", round(float(np.abs(r_b - r_a).max()), 4))
print("MSE original :", round(float(np.mean((r_a - X[i4]) ** 2)), 6))
print("MSE modificado:", round(float(np.mean((r_b - X[i4]) ** 2)), 6))
""", "Resultados de nuestra corrida: el cambio máximo en un píxel es [[bits.perturbar.cambio_max|4]] y el MSE original [[bits.perturbar.mse_original|6]]. Mover una sola coordenada 0.05 cambia la reconstrucción de forma suave (el decoder es una función continua): el código real guarda información fina.")

# ================================================================== PARTE E
md("""
## Parte E · El error de reconstrucción como alerta

### E1 · Puntaje y umbral (diapositivas 38 y 39)
El **puntaje** de una imagen es su MSE de reconstrucción: **e(x) = mean((x̂ − x)²)**. Se declara una **alerta** si **e(x) > c**. El umbral c se fija **con validación, antes de mirar la prueba**: aquí, el **percentil 95** de los 360 errores de validación. Por construcción, el 5 % de la validación queda sobre el corte. **El puntaje no es una probabilidad de anomalía y el percentil 95 no es un 95 % de confianza.**
""")
code("""
def puntaje(m, Xc):
    return np.mean((Xc - m.predict(Xc)) ** 2, axis=1)


e_va, e_te = puntaje(modelo, X_va), puntaje(modelo, X_te)
umbral = np.quantile(e_va, 0.95)
print(f"umbral c = percentil 95 de validación = {umbral:.6f}")
print("validación sobre c:", int((e_va > umbral).sum()), "de", len(e_va))
print("prueba     sobre c:", int((e_te > umbral).sum()), "de", len(e_te), f"({100 * (e_te > umbral).mean():.2f} %)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].bar(range(len(e_va)), np.sort(e_va), color=["tab:red" if v > umbral else "tab:purple" for v in np.sort(e_va)], width=1.0)
axes[0].axhline(umbral, color="green", ls="--")
axes[0].set(title="Errores de validación ordenados", xlabel="posición", ylabel="MSE por imagen")
axes[1].hist(e_va, bins=24, color="tab:purple", alpha=0.85, edgecolor="white")
axes[1].axvline(umbral, color="green", ls="--", label="umbral")
axes[1].set(title="Histograma (validación)", xlabel="MSE por imagen", ylabel="imágenes")
axes[1].legend()
plt.tight_layout()
plt.show()
""")
md("""
El umbral es **[[umbral.umbral|6]]** y **[[umbral.val_sobre]] de 360** errores de validación lo superan (5 %). En prueba son **[[umbral.prueba_sobre]] de 360** ([[s.tp]]): la proporción puede diferir de 5 % porque el umbral se fijó con otras imágenes. Las alertas **no confirman defectos**: son imágenes que el modelo reconstruyó peor.

### ✍️ Ejercicio E1
Calcula la matriz «alerta / sin alerta» por dígito en prueba: ¿qué dígitos concentran más alertas con este umbral? ¿Por qué podría pasar eso si no hay defectos?
""")
code("""
# tu código aquí
""")
solution("""
alerta = e_te > umbral
por_digito = pd.DataFrame({"alertas": pd.Series(alerta).groupby(y[te]).sum(), "imágenes": pd.Series(alerta).groupby(y[te]).size()})
por_digito["%"] = (100 * por_digito["alertas"] / por_digito["imágenes"]).round(1)
print(por_digito.T)
""", f"En nuestra corrida el dígito 8 concentra {_pd[8]['alertas']} de las {sum(d['alertas'] for d in _pd)} alertas ({_pd[8]['alertas']} de {_pd[8]['imagenes']} ochos de prueba), seguido del 7 ({_pd[7]['alertas']}) y el 2 ({_pd[2]['alertas']}); el 0 no tiene ninguna. Son los dígitos que el modelo reconstruye peor, no imágenes «defectuosas»: hay variación habitual que el modelo no representa bien. Eso es justo el límite de la diapositiva 42: muchos avisos sobre casos aceptables.")

md("""
### E2 · Prueba de estrés y límites (diapositivas 40, 41 y 42)
Se **desplaza cada imagen una columna a la derecha** (rellenando con ceros) y se compara cada salida con **su propia entrada desplazada**. Esto muestra sensibilidad a un cambio, **no** detección validada de defectos reales.
""")
code("""
desplazadas = np.zeros((len(X_te), 8, 8))
desplazadas[:, :, 1:] = X_te.reshape(-1, 8, 8)[:, :, :-1]
desplazadas = desplazadas.reshape(-1, 64)
e_des = puntaje(modelo, desplazadas)
print(f"alertas en originales: {int((e_te > umbral).sum())} de 360 ({100 * (e_te > umbral).mean():.2f} %) | desplazadas: {int((e_des > umbral).sum())} de 360 ({100 * (e_des > umbral).mean():.2f} %)")

x4d = np.zeros((8, 8))
x4d[:, 1:] = x4.reshape(8, 8)[:, :-1]
print("MSE del 3 (id 4): original", round(float(puntaje(modelo, x4[None])[0]), 4), "| desplazado", round(float(puntaje(modelo, x4d.reshape(1, 64))[0]), 4))

plt.figure(figsize=(8, 3.2))
bins = np.linspace(0, 0.2, 41)
plt.hist(np.clip(e_te, 0, 0.2), bins=bins, alpha=0.8, label="originales", color="tab:purple")
plt.hist(np.clip(e_des, 0, 0.2), bins=bins, alpha=0.7, label="desplazadas", color="tab:orange")
plt.axvline(umbral, color="green", ls="--", label="umbral de validación")
plt.xlabel("MSE por imagen (prueba)")
plt.ylabel("imágenes")
plt.legend()
plt.show()
""")
md("""
Alertas en prueba: **[[umbral.prueba_sobre]] de 360** ([[s.tp]]) en originales y **[[umbral.desplazadas_sobre]] de 360** ([[s.td]]) en desplazadas; el 3 de la diapositiva 41 pasa de MSE **[[pixel.mse_imagen|4]]** a **[[pixel.mse_desplazada|4]]**. Una red densa no es invariante a desplazamientos: correr la imagen una columna cambia por completo las coordenadas que recibe.

**Límites** (diapositivas 42 a 46):
- Variación habitual que el modelo no representó → muchos avisos sobre casos aceptables.
- Anomalías en el entrenamiento y decoder flexible → el modelo puede aprender a reconstruirlas.
- Cambio de adquisición → el puntaje refleja el cambio técnico.
- Selección y umbral usaron **la misma validación**: no garantizan falsas alarmas futuras.
- El estrés artificial no mide sensibilidad ni especificidad frente a anomalías reales.
- «Rechazar el 5 % con mayor error» **no** equivale a «detectar cualquier defecto con 95 % de confianza». Se necesita una **validación específica**: definir defectos relevantes, separar por lote o fecha, calibrar el umbral con datos pertinentes, evaluar en una reserva y prever revisión humana.

### ✍️ Ejercicio E2
Una lista de 10 puntajes de reconstrucción (con su etiqueta real de defecto): A 0.012 (no), B 0.031 (sí), C 0.018 (no), D 0.052 (sí), E 0.024 (no), F 0.038 (no), G 0.015 (no), H 0.071 (sí), I 0.027 (no), J 0.020 (sí). Con el umbral c = 0.030, cuenta las alertas, las **falsas alarmas** y los **defectos no detectados**; repite con c = 0.045.
""")
code("""
# tu código aquí
""")
solution("""
puntajes = dict(A=0.012, B=0.031, C=0.018, D=0.052, E=0.024, F=0.038, G=0.015, H=0.071, I=0.027, J=0.020)
defectos = {"B", "D", "H", "J"}
for c in (0.030, 0.045):
    alertas = {k for k, v in puntajes.items() if v > c}
    print(f"c = {c}: alertas {sorted(alertas)} | falsas alarmas {sorted(alertas - defectos)} | defectos no detectados {sorted(defectos - alertas)}")
""", "Con c = 0.030: alertas B, D, F y H (4); falsa alarma F; defecto no detectado J. Con c = 0.045: alertas D y H (2); sin falsas alarmas; defectos no detectados B y J. Subir el corte elimina la falsa alarma pero deja pasar más defectos: el umbral es un compromiso que hay que validar para el uso previsto.")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (sin ayuda · 25 min)
Es la **práctica de la diapositiva 53**: repite el experimento con un vector latente de **4** en lugar de 8 (`DIM_LATENTE = 4`, arquitecturas `(4,)` y `(32, 4, 32)`), con la misma receta.

1. Entrena ambas redes con `entrenar_ae` y elige con la validación. ¿Cuál gana y en qué época está su mínimo?
2. Compara con PCA de 4 componentes en validación y con las redes de 8 coordenadas.
3. Cuenta los parámetros de cada red a mano y verifica con `n_parametros`.
4. Mira las imágenes reconstruidas de los 6 primeros dígitos de prueba con latente 4: ¿qué detalles se pierden?
5. ¿Mirarías la prueba para decidir entre 4 y 8 coordenadas? ¿Por qué no?
""")
code("""
# tu código aquí
""")
solution("""
R4 = {nombre: entrenar_ae(capas, X_tr, X_va) for nombre, capas in (("64-4-64", (4,)), ("64-32-4-32-64", (32, 4, 32)))}
for nombre, r in R4.items():
    print(f"{nombre:14s} | mejor época {r['epoca']:3d} | MSE validación {r['mejor']:.5f} | parámetros {n_parametros(r['modelo'])}")
p4 = PCA(n_components=4, svd_solver="full").fit(X_tr)
print("PCA 4:", round(float(np.mean((X_va - p4.inverse_transform(p4.transform(X_va))) ** 2)), 5))
print("a mano: 64-4-64 =", 64 * 4 + 4 + 4 * 64 + 64, "| 64-32-4-32-64 =", (64 * 32 + 32) + (32 * 4 + 4) + (4 * 32 + 32) + (32 * 64 + 64))
""", "Resultados de nuestra corrida: 64–4–64 llega a [[extra.ae4.val|5]] (época [[extra.ae4.epoca]]) y 64–32–4–32–64 a [[extra.prof4.val|5]] (época [[extra.prof4.epoca]]); PCA 4 da [[extra.pca4_val|5]]. Gana la red profunda; la de una capa queda **peor** que PCA 4. Parámetros: 580 y 4 484. Con latente 4 el error sube frente a latente 8 ([[mse.ae8.val|4]] y [[mse.prof.val|4]]): se pierden más detalles. Y no: la decisión entre 4 y 8 se toma con validación; la prueba queda para evaluar la receta ya fijada.")

md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué es un autoencoder? ¿Qué son el encoder, el decoder y el vector latente?
2. ¿Qué referencia usa la pérdida de reconstrucción? ¿Hacen falta etiquetas?
3. Con el encoder h = (x₁ + x₂)/2 y el decoder (h, h): ¿qué reconstrucción y qué MSE da (0.2, 0.8)? ¿Qué se perdió?
4. ¿Qué es el MSE por píxel, por imagen y por conjunto? ¿Por qué se eleva al cuadrado?
5. ¿Por qué un error de reconstrucción de 0 no es la meta?
6. ¿Por qué 8 coordenadas reales no equivalen a 8 bits?
7. ¿Con qué dos referencias se compara el autoencoder y qué MSE dan en validación?
8. ¿Cuántos parámetros tienen 64–8–64 y 64–32–8–32–64? ¿Por qué el latente y los parámetros no son lo mismo?
9. ¿Con qué conjunto se eligió entre las arquitecturas y por qué el mínimo en la época 150 limita la conclusión?
10. ¿Cómo se obtiene el vector latente de un `MLPRegressor` entrenado? ¿Qué significa que sean 8 coordenadas entre −1 y 1?
11. ¿Cómo se fija el umbral de una alerta por reconstrucción? ¿Qué porcentaje de validación lo supera por construcción?
12. ¿Es el puntaje una probabilidad de anomalía? ¿Qué evidencia falta para usarlo en un sistema de inspección real?

<details><summary><b>Respuestas</b></summary>

1. Una red entrenada para reproducir su propia entrada pasando por un resumen pequeño. El encoder calcula h = f(x), el decoder x̂ = g(h) y h es el vector latente (aquí 8 valores).
2. La propia entrada: se compara x̂ con x píxel a píxel. No hacen falta etiquetas.
3. h = 0.5 y x̂ = (0.5, 0.5): MSE = (0.3² + 0.3²)/2 = 0.09. Se perdió el orden: (0.8, 0.2) da el mismo código.
4. ℓ = promedio de (x̂ⱼ − xⱼ)² sobre los 64 píxeles; L = promedio sobre las imágenes. Se eleva al cuadrado para que los signos no se cancelen y para penalizar más los errores grandes.
5. Una copia perfecta (la identidad) tiene error 0 y no resume nada; hace falta un cuello de botella y evaluar fuera del ajuste.
6. Cada coordenada es un real entre −1 y 1 con infinitos valores posibles; 8 bits solo dan 2⁸ = 256 combinaciones. En el experimento, reemplazar el código por sus signos sube el MSE de 0.0193 a 0.1277.
7. Imagen media (0.0729) y PCA de 8 componentes (0.0235); el autoencoder elegido llega a 0.0193.
8. 1 096 (520 + 576) y 4 744 (2 080 + 264 + 288 + 2 112). El latente es cuántos números resumen la imagen (8 en ambas); los parámetros son todos los pesos y sesgos.
9. Con validación (menor MSE). El mínimo en la última época significa que la curva aún bajaba: el presupuesto de épocas limita la conclusión.
10. Recorriendo solo las capas del encoder con `coefs_` e `intercepts_` y tanh. Son las salidas de tanh: acotadas, sin significado individual y no probabilidades.
11. Con el percentil 95 de los errores de validación, antes de mirar la prueba; el 5 % de la validación lo supera por construcción.
12. No: es un MSE. Falta una validación específica (definir defectos, separar por lote o fecha, calibrar el umbral con datos pertinentes, evaluar en una reserva y revisar los errores).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Para seguir**: la práctica de la diapositiva 53 (latente 4) y las secciones 07–09 del programa `autoencoders.py` (prueba, utilidad para clasificar y alerta con desplazamiento) amplían lo visto aquí.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
