"""Genera el notebook guiado de la Clase 45 (Redes neuronales).

Uso normal:      python notebook_c45.py             -> Guias/notebooks/C45_RedesNeuronales.ipynb
Modo de prueba:  C45_TEST=1 python notebook_c45.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c45_numeros.json (lo escribe figuras_c45.py) con marcas [[ruta|decimales]].
Usa numpy, pandas, matplotlib y scikit-learn.
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C45_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C45_RedesNeuronales_test.ipynb" if TEST else "C45_RedesNeuronales.ipynb")
N = json.loads((HERE / "c45_numeros.json").read_text(encoding="utf-8"))
pc = lambda x: f"{x * 100:.2f} %"
AL = N["alphas"]
N["s"] = dict(log_acc=pc(N["eval"]["log"]["prueba"]["acc"]), red_acc=pc(N["eval"]["red"]["prueba"]["acc"]),
              a0=f"{AL['0.0']['val_120']:.4f}", a4=f"{AL['0.0001']['val_120']:.4f}", a2=f"{AL['0.01']['val_120']:.4f}", a1=f"{AL['0.1']['val_120']:.4f}",
              a10=f"{AL['1.0']['val_120']:.4f}", a30=f"{AL['3.0']['val_120']:.4f}",
              w2=f"{N['anchos']['2']['val']:.3f}", w8=f"{N['anchos']['8']['val']:.3f}", w32=f"{N['anchos']['32']['val']:.3f}", w64=f"{N['anchos']['64']['val']:.3f}")
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
# Clase 45 · Redes neuronales
**Diplomado Data Science** · Prof. Eloy Alvarado · primera mirada al interior de una red neuronal, con los dígitos escritos a mano de la Clase 44

Material de apoyo:
- `Guias/C45_Guia_RedesNeuronales.pdf` — guía visual de la clase.
- `Guias/apuntes/C45_Apuntes_RedesNeuronales.pdf` — apuntes explicados paso a paso, con ejemplos resueltos.
- `Guias/apuntes/C45_Ejercicios_Cuaderno_RedesNeuronales.pdf` — ejercicios en papel (antes o después de este notebook).
- Diapositivas y programa del profesor (carpeta «Clase 45 …»): `redes_neuronales.pdf`, `redes_neuronales.py`, `digitos.csv`.
- Transcripción: `VideosClases/Clase 45 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib` y `scikit-learn`. Si no los tienes:
```
pip install numpy pandas matplotlib scikit-learn
```

**Datos**: `digitos.csv` de la carpeta de la Clase 45 (1 797 imágenes de 8 × 8 píxeles, columnas `px00`…`px63`, `digito` y `conjunto`: **1 077 entrenamiento / 360 validación / 360 prueba**). Si el notebook no lo encuentra, usa `load_digits()` y arma una partición propia (las cifras cambian un poco).

**Cómo leer los números**: la receta de entrenamiento es **la misma del profesor** (`redes_neuronales.py`): 64–16–10 y 64–32–10, ReLU + softmax, Adam con tasa 0.002, lote 64, alpha 0.0001, 120 épocas, semilla 20261020. Con scikit-learn 1.9.1 y `digitos.csv` las cifras coinciden **exactamente** con las diapositivas. Lo marcado como *nuestra reproducción* (otros anchos, otros alphas) es cálculo propio.

### Objetivos
1. Calcular **a mano, con numpy**, una neurona, una capa y una red pequeña (ReLU, sigmoide, softmax).
2. Entender la **pérdida** −log p y el **descenso de gradiente** en una función de una variable.
3. Seguir la **retropropagación** de un ejemplo pequeño y verificar el gradiente numéricamente.
4. Entrenar con `MLPClassifier` la receta del profesor y **elegir el ancho con la validación**.
5. Leer **curvas** de entrenamiento y validación, y ver el efecto de **alpha** y del **ancho**.
6. **Abrir la red**: pesos, parámetros y propagación manual.
7. Comparar con la **regresión logística** y revisar los **errores**.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 20 | A1 · La neurona, A2 · Una capa y una red pequeña |
| B | 20 | B1 · Softmax y pérdida |
| C | 25 | C1 · Descenso de gradiente, C2 · Retropropagación |
| D | 15 | D1 · Datos y partición |
| E | 30 | E1 · Entrenar la receta del profesor, E2 · Curvas |
| F | 20 | F1 · Abrir la red, F2 · Efecto de alpha y del ancho |
| G | 20 | G1 · Logística frente a red y errores |
| — | 25 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Lo aleatorio usa semilla fija; con otras versiones de las librerías puede variar la última cifra.
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
np.set_printoptions(suppress=True, precision=4)
print("scikit-learn:", sklearn.__version__)
""")
md("""
### Mini-repaso
- **Regresión logística y probabilidades** (Clases 20–21 y 31): una probabilidad por clase.
- **Log loss** (Clases 38 y 44): promedio de −log(probabilidad de la clase real).
- **Validación y prueba** (Clases 30 y 38): se elige con validación; la prueba se mira al final.
- **Producto de matrices**: (n × p) · (p × q) = (n × q). Cada columna de la segunda matriz produce una columna del resultado.
- **Derivada**: la pendiente de una curva; si es negativa, aumentar la variable baja la función.
""")

# ================================================================== PARTE A
md("""
## Parte A · Neurona, capa y red pequeña (con numpy)

### A1 · La neurona (20 min)
Una neurona calcula **a = w₁x₁ + … + wₚxₚ + b** y luego aplica una función de activación: **h = φ(a)**. Los pesos **w** y el sesgo **b** son parámetros (se aprenden); **x** son los datos. Con ReLU, φ(a) = máx(0, a). Es el ejemplo de la diapositiva 6: x = (0.5, 1), w = (0.8, −0.4), b = 0.2.
""")
code("""
def relu(a):
    return np.maximum(0, a)


def sigmoide(a):
    return 1 / (1 + np.exp(-a))


w = np.array([0.8, -0.4])
b = 0.2
for nombre, x, sesgo in [("base", np.array([0.5, 1.0]), b), ("x2 = 2", np.array([0.5, 2.0]), b), ("b = 0.5", np.array([0.5, 1.0]), 0.5)]:
    productos = w * x
    a = productos.sum() + sesgo
    print(f"{nombre:8s} productos {productos} + sesgo {sesgo} -> a = {a:.2f} -> ReLU = {relu(a):.2f}")

a = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
print("\\na        :", a)
print("ReLU     :", relu(a))
print("sigmoide :", sigmoide(a).round(4))
print("tanh     :", np.tanh(a).round(4))
""")
md("""
Con los mismos pesos, cambiar la entrada o el sesgo cambia la activación: base **0.2**, con x₂ = 2 el resultado es **0** (la suma es −0.2 y ReLU la lleva a 0), y con b = 0.5 es **0.5**. Una activación 0 no es un error: es un valor válido.

### ✍️ Ejercicio A1
Una neurona tiene pesos w = (1.5, −2, 0.5), sesgo b = 0.25 y usa ReLU. Calcula a y la activación para x = (2, 1, 1) y para x = (0, 1, 1). ¿Cuál de las dos entradas «apaga» la neurona?
""")
code("""
# tu código aquí
""")
solution("""
w = np.array([1.5, -2.0, 0.5])
b = 0.25
for x in (np.array([2.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0])):
    a = w @ x + b
    print(x, "-> a =", a, "-> h =", relu(a))
""", "Debes ver a = 1.75 (h = 1.75) y a = −1.25 (h = 0): la segunda entrada apaga la neurona.")

md("""
### A2 · Una capa y una red pequeña (diapositivas 8, 12, 14 y 15)
Una **capa** son varias neuronas con la misma entrada. Los pesos se ordenan en una matriz **W** con **una columna por neurona**, y cada **fila de X es una observación**: A = X·W + b, H = ReLU(A). Después, una neurona de salida con sigmoide entrega una probabilidad (ejemplo binario).
""")
code("""
W1 = np.array([[1.0, -1.0], [0.5, 0.5]])
b1 = np.array([0.0, 0.0])
x = np.array([[1.0, 2.0]])
A1 = x @ W1 + b1
H = relu(A1)
print("a =", A1, " h =", H)

W2 = np.array([[0.5], [-1.0]])
b2 = -0.5
s = (H @ W2 + b2).item()
print("puntaje s =", s, "| probabilidad = sigmoide(s) =", round(float(sigmoide(s)), 4))

# tres observaciones a la vez: cada fila usa los mismos pesos
X3 = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 1.0]])
A3 = X3 @ W1 + b1
print("\\nA =\\n", A3)
print("H =\\n", relu(A3))
print("formas: X", X3.shape, "· W1", W1.shape, "= A", A3.shape)
""")
md("""
Resultados: h = **(2, 0)**, s = **0.5** y probabilidad **0.6225** (diapositivas 14 y 15). Con tres observaciones, el −1.5 de la segunda fila pasa a 0. Si **no** hubiera activación entre capas, dos capas se reducirían a una: (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂).

### ✍️ Ejercicio A2
Para la red de arriba, comprueba que **sin ReLU** la salida de dos capas coincide con la de una sola capa con W* = W₁W₂ y b* = b₁W₂ + b₂ (usa x = (3, −1) y los pesos de abajo). ¿Qué enseña esto sobre la activación?
""")
code("""
Wa = np.array([[1.0, 2.0], [0.0, 1.0]])
ba = np.array([1.0, 0.0])
Wb = np.array([[2.0], [1.0]])
bb = np.array([0.5])
x = np.array([3.0, -1.0])
# tu código aquí
""")
solution("""
dos_capas = (x @ Wa + ba) @ Wb + bb
W_c = Wa @ Wb
b_c = ba @ Wb + bb
una_capa = x @ W_c + b_c
print("dos capas:", dos_capas, "| una capa equivalente:", una_capa, "| W* =", W_c.ravel(), "b* =", b_c)
""", "Ambas dan [[compone.dos_capas|1]]: sin una activación no lineal, apilar capas no aporta capacidad nueva.")

# ================================================================== PARTE B
md("""
## Parte B · Salida y pérdida

### B1 · Softmax y −log p (diapositivas 16 a 18)
Para clases excluyentes, la capa de salida entrega un **puntaje por clase** y **softmax** los convierte en probabilidades: se exponencia cada puntaje y se divide por la suma. Para evitar números enormes se resta el máximo. La **pérdida de una imagen** es −log de la probabilidad de la clase **real**.
""")
code("""
def softmax(s):
    s = np.asarray(s, dtype=float)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


s = np.array([0.0, np.log(2), 0.0])
p = softmax(s)
print("p =", p, "| suma =", p.sum())
print("sumar 100 a todos los puntajes:", softmax(s + 100))

for pr in (0.8, 0.5, 0.1):
    print(f"p de la clase real = {pr}: pérdida = {-np.log(pr):.3f}")
""")
md("""
Las probabilidades son **(0.25, 0.50, 0.25)** y no cambian al sumar 100 (diapositiva 17). Las pérdidas son **0.223**, **0.693** y **2.303**: cuanto menos probabilidad se da a la clase real, más sube la pérdida (diapositiva 18). Ojo: se usa la probabilidad de la clase **real**, aunque otra clase tenga mayor probabilidad.

### ✍️ Ejercicio B1
Los puntajes de una red para una imagen son s = (2, 1, 0) y la clase real es la tercera (índice 2). Calcula las probabilidades, la clase predicha y la pérdida. ¿Acierta el modelo?
""")
code("""
# tu código aquí
""")
solution("""
s = np.array([2.0, 1.0, 0.0])
p = softmax(s)
print("probabilidades:", p, "| predice la clase", p.argmax())
print("pérdida de la clase real (2):", -np.log(p[2]))
""", "Predice la clase 0 (p = 0.665) y no acierta; la pérdida es −ln(0.090) = 2.408.")

# ================================================================== PARTE C
md("""
## Parte C · Aprender: gradiente y retropropagación

### C1 · Descenso de gradiente en una variable (diapositivas 19, 20 y 27)
**Aprender** es buscar los parámetros que minimizan una función objetivo J. El **gradiente** (en una variable, la derivada) dice hacia dónde sube J; se avanza en sentido contrario: **w⁺ = w − η·J′(w)**, con η la tasa de aprendizaje. Con el juguete J(w) = (w − 3)², J′(w) = 2(w − 3).
""")
code("""
J = lambda w: (w - 3) ** 2
dJ = lambda w: 2 * (w - 3)

w, eta = 1.0, 0.1
print(f"w = {w}: J = {J(w)}, J' = {dJ(w)}, nuevo w = {w - eta * dJ(w)}, J(nuevo) = {J(w - eta * dJ(w)):.2f}")

w = 1.0
for t in range(8):
    print(f"paso {t}: w = {w:.4f}  J = {J(w):.4f}")
    w = w - eta * dJ(w)

print("\\nPérdida tras 10 actualizaciones desde w = 0, según la tasa:")
for eta in (0.05, 0.1, 0.4, 1.05):
    w = 0.0
    for _ in range(10):
        w -= eta * dJ(w)
    print(f"  eta = {eta:<5}: J = {J(w):.3g}")
""")
md("""
Desde w = 1: J = **4**, J′ = **−4**, nuevo w = **1.4** y J = **2.56** (diapositiva 20). El error |w − 3| se multiplica por (1 − 2η) en cada paso: con η = 0.4 baja muy rápido, con η = 0.05 lento y con η = **1.05** el factor es −1.1: el error **crece** y el método se aleja del mínimo (diapositiva 27).

### ✍️ Ejercicio C1
Minimiza J(w) = (w − 4)² con η = 0.25 desde w = 0: haz tres actualizaciones e imprime w y J en cada una. ¿Por qué factor se multiplica el error |w − 4| en cada paso? Repite con η = 1.2.
""")
code("""
# tu código aquí
""")
solution("""
for eta in (0.25, 1.2):
    w = 0.0
    print("eta =", eta)
    for t in range(1, 4):
        w = w - eta * 2 * (w - 4)
        print(f"  w{t} = {w:.4f}  J = {(w - 4) ** 2:.4f}  |w - 4| = {abs(w - 4):.4f}")
    print("  factor por paso:", abs(1 - 2 * eta))
""", "Con η = 0.25 el factor es 0.5 (w = 2, 3, 3.5); con η = 1.2 es 1.4 > 1: el error crece (w = 9.6, −9.12, …).")

md("""
### C2 · Retropropagación de un ejemplo pequeño (diapositivas 21 a 24)
Una entrada x = 0.5 con etiqueta y = 1, una unidad oculta ReLU y una salida sigmoide, con parámetros θ = (w₁, b₁, w₂, b₂) = (0.4, 0.1, 0.8, −0.2). La **retropropagación** aplica la regla de la cadena hacia atrás: δ₂ = p − y; δ₁ = δ₂·w₂·ReLU′(a₁); y el gradiente de cada peso es el δ de su unidad por la entrada de ese peso. Luego un paso de descenso con η = 0.1. Se verifica el gradiente con diferencias finitas.
""")
code("""
def adelante(theta, x=0.5, y=1):
    w1, b1, w2, b2 = theta
    a1 = w1 * x + b1
    h = max(a1, 0.0)
    a2 = w2 * h + b2
    p = sigmoide(a2)
    perdida = -np.log(p if y == 1 else 1 - p)
    return a1, h, a2, p, perdida


theta = np.array([0.4, 0.1, 0.8, -0.2])
x, y = 0.5, 1
a1, h, a2, p, perdida = adelante(theta)
print(f"a1 = {a1:.2f}, h = {h:.2f}, a2 = {a2:.2f}, p = {p:.6f}, pérdida = {perdida:.6f}")

delta2 = p - y
delta1 = delta2 * theta[2] * float(a1 > 0)
grad = np.array([delta1 * x, delta1, delta2 * h, delta2])
print("gradiente (w1, b1, w2, b2):", grad.round(4))

# comprobación numérica
eps = 1e-6
num = np.array([(adelante(theta + eps * v)[4] - adelante(theta - eps * v)[4]) / (2 * eps) for v in np.eye(4)])
print("diferencia máxima con el gradiente numérico:", np.abs(grad - num).max())

nuevo = theta - 0.1 * grad
*_, p2, perdida2 = adelante(nuevo)
print("\\nnuevos parámetros:", nuevo.round(4))
print(f"p: {p:.6f} -> {p2:.6f} | pérdida: {perdida:.6f} -> {perdida2:.6f}")
""")
md("""
Debes ver el gradiente **(−0.1960, −0.3920, −0.1470, −0.4900)**, nuevos parámetros **(0.4196, 0.1392, 0.8147, −0.1510)** y la pérdida bajando de **0.673347** a **0.628702** (diapositivas 23 y 24). El gradiente analítico coincide con el numérico. Un solo paso que baja la pérdida de un ejemplo **no demuestra** que la red generalice: solo muestra el mecanismo.

### ✍️ Ejercicio C2
Con x = 2, y = 0 y θ = (0.5, −0.5, 1, 0), calcula el gradiente, da un paso con η = 0.1 y comprueba que la pérdida baja. (Pista: con y = 0, δ₂ = p − 0 y la pérdida es −ln(1 − p).)
""")
code("""
# tu código aquí
""")
solution("""
theta = np.array([0.5, -0.5, 1.0, 0.0])
x, y = 2.0, 0
a1, h, a2, p, perdida = adelante(theta, x, y)
d2 = p - y
d1 = d2 * theta[2] * float(a1 > 0)
grad = np.array([d1 * x, d1, d2 * h, d2])
nuevo = theta - 0.1 * grad
*_, p2, perdida2 = adelante(nuevo, x, y)
print("gradiente:", grad.round(4))
print("pérdida antes/después:", round(perdida, 4), round(perdida2, 4))
""", "Gradiente ≈ (1.2449, 0.6224, 0.3112, 0.6225); la pérdida baja de 0.9741 a 0.7553.")

# ================================================================== PARTE D
md("""
## Parte D · Los datos

### D1 · Datos y partición (diapositivas 3, 4 y 31)
Cada imagen son 64 intensidades (0 a 16) que se dividen por 16 para quedar entre 0 y 1; la etiqueta va aparte. La partición del profesor está en la columna `conjunto` de `digitos.csv`.
""")
code("""
def buscar_csv():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in base.glob("Clase 45*/digitos.csv"):
            return p
        if (base / "digitos.csv").exists():
            return base / "digitos.csv"
    return None


ruta = buscar_csv()
if ruta is not None:
    datos = pd.read_csv(ruta)
    X = datos.filter(regex="^px").to_numpy(dtype=float) / 16
    y = datos["digito"].to_numpy(dtype=int)
    tr, va, te = (datos["conjunto"].eq(k).to_numpy() for k in ("entrenamiento", "validacion", "prueba"))
    print("Uso digitos.csv y la partición del profesor.")
else:
    from sklearn.model_selection import train_test_split
    dig = load_digits()
    X, y = dig.data / 16, dig.target
    idx = np.arange(len(y))
    i_dev, i_te = train_test_split(idx, test_size=360, stratify=y, random_state=45)
    i_tr, i_va = train_test_split(i_dev, test_size=360, stratify=y[i_dev], random_state=45)
    tr, va, te = (np.isin(idx, i) for i in (i_tr, i_va, i_te))
    print("⚠️ No encontré digitos.csv: uso una partición propia; las cifras serán algo distintas a las del texto.")

print("entrada:", X.shape, "| entrenamiento:", tr.sum(), "| validación:", va.sum(), "| prueba:", te.sum())
print("valores entre", X.min(), "y", X.max())


def log_loss(reales, P):
    # promedio de -log(probabilidad asignada a la clase real)
    return float(-np.log(np.clip(P[np.arange(len(reales)), reales], 1e-15, 1)).mean())
""")
md("""
Con la partición del profesor hay **[[datos.n_tr]]** imágenes de entrenamiento, **[[datos.n_va]]** de validación y **[[datos.n_te]]** de prueba. La prueba **no participa** en ninguna decisión de esta parte; solo se mira al final (y, como ya se mostró en clase, esa comparación es retrospectiva).

### ✍️ Ejercicio D1
¿Cuántos lotes completos de 64 hay en el entrenamiento, cuántas imágenes tiene el último y cuántas actualizaciones hay por época y en 120 épocas?
""")
code("""
# tu código aquí
""")
solution("""
n, B = int(tr.sum()), 64
completos, resto = divmod(n, B)
lotes = completos + (1 if resto else 0)
print("lotes completos:", completos, "| último lote:", resto, "| por época:", lotes, "| en 120 épocas:", 120 * lotes)
""", "Debes ver 16 lotes completos, un último lote de [[lotes.resto]] imágenes, [[lotes.lotes]] actualizaciones por época y [[lotes.actualiz]] en 120 épocas (diapositiva 26).")

# ================================================================== PARTE E
md("""
## Parte E · Entrenar y elegir

### E1 · La receta del profesor (diapositivas 25, 33 y 35)
Dos candidatas, **64–16–10** y **64–32–10**: ReLU en la capa oculta y softmax en la salida; optimizador **Adam** con tasa 0.002, **lote 64**, **alpha** 0.0001 (penalización de pesos), **120 épocas** y semilla 20261020. Después de **cada época** se mide el log loss (sin penalización) en entrenamiento y validación, y se **conserva una copia de la red de la época con menor log loss de validación**. Se usa `partial_fit` sobre la misma red, que ejecuta una época completa por llamada.
""")
code("""
def entrenar(capas, alpha=1e-4, epocas=120, semilla=20261020):
    red = MLPClassifier(hidden_layer_sizes=capas, activation="relu", solver="adam", learning_rate_init=0.002,
                        batch_size=64, alpha=alpha, random_state=semilla, shuffle=True, early_stopping=False)
    mejor, ep, copia, curva_tr, curva_va = np.inf, 0, None, [], []
    with threadpool_limits(limits=1):
        for e in range(1, epocas + 1):
            red.partial_fit(X[tr], y[tr], classes=np.arange(10))
            ltr = log_loss(y[tr], red.predict_proba(X[tr]))
            lva = log_loss(y[va], red.predict_proba(X[va]))
            curva_tr.append(ltr)
            curva_va.append(lva)
            if lva < mejor:
                mejor, ep, copia = lva, e, deepcopy(red)
    return dict(modelo=copia, final=red, epoca=ep, val=mejor, tr=curva_tr, va=curva_va)


def n_parametros(m):
    return sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_))


R = {16: entrenar((16,)), 32: entrenar((32,))}
for k, r in R.items():
    print(f"Red{k}: parámetros {n_parametros(r['modelo'])} | mejor época {r['epoca']} | log loss de validación {r['val']:.4f}")
elegida = min(R, key=lambda k: R[k]["val"])
print("Se conserva Red" + str(elegida), "de la época", R[elegida]["epoca"])
""")
md("""
Con la partición del profesor: **Red16** llega a [[curvas.Red16.val|4]] en la época [[curvas.Red16.epoca]] y **Red32** a [[curvas.Red32.val|4]] en la época [[curvas.Red32.epoca]]; se conserva **Red32** (diapositiva 35). Los parámetros son **1 210** y **2 410**. Los mínimos están cerca del final del presupuesto de 120 épocas, así que **no afirmamos convergencia**. Elegir con validación significa que la prueba no se usó para decidir, y que **no se reajusta** con entrenamiento + validación.

### ✍️ Ejercicio E1
Calcula a mano el número de parámetros de la 64–16–10 y la 64–32–10 y compáralo con `n_parametros`. ¿Cuántos parámetros añadiría una segunda capa oculta de 32 a la 64–32–10?
""")
code("""
# tu código aquí
""")
solution("""
p16 = 16 * (64 + 1) + 10 * (16 + 1)
p32 = 32 * (64 + 1) + 10 * (32 + 1)
p_doble = 32 * (64 + 1) + 32 * (32 + 1) + 10 * (32 + 1)
print(p16, n_parametros(R[16]["modelo"]), "|", p32, n_parametros(R[32]["modelo"]))
print("con dos capas de 32:", p_doble, "→ añade", p_doble - p32)
""", "Debes ver 1 210 y 2 410; la segunda capa añade 32·33 = 1 056 (total 3 466): no «duplica» los parámetros.")

md("""
### E2 · Leer las curvas (diapositivas 30 y 36)
La curva de entrenamiento sigue bajando; la de validación se aplana. Si entrenamiento baja pero validación no, la red empieza a memorizar (sobreajuste). Se usa escala logarítmica para ver el detalle.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, k in zip(axes, (16, 32)):
    r = R[k]
    e = np.arange(1, 121)
    ax.semilogy(e, r["tr"], label="entrenamiento")
    ax.semilogy(e, r["va"], label="validación")
    ax.axvline(r["epoca"], color="gray", ls="--", label=f"época elegida {r['epoca']}")
    ax.set(title=f"Red{k}", xlabel="época")
    ax.legend()
axes[0].set_ylabel("log loss sin penalización")
plt.tight_layout()
plt.show()

for k in (16, 32):
    r = R[k]
    print(f"Red{k}: época 1 -> entr. {r['tr'][0]:.3f}, valid. {r['va'][0]:.3f} | época 120 -> entr. {r['tr'][-1]:.4f}, valid. {r['va'][-1]:.4f}")
""")
md("""
En la época 120, Red32 tiene entrenamiento **[[curvas.Red32.tr_120|4]]** y validación **[[curvas.Red32.va_120|4]]**: la brecha entre ambas es la señal de que la red ajusta mejor lo que ya vio que lo nuevo.

### ✍️ Ejercicio E2
Para Red32, ¿en qué época el entrenamiento cruza por debajo de la validación por última vez? ¿Qué cambia en la validación entre las épocas 100 y 120?
""")
code("""
# tu código aquí
""")
solution("""
tr_, va_ = np.array(R[32]["tr"]), np.array(R[32]["va"])
cruces = np.where(tr_ > va_)[0]
print("última época con entrenamiento > validación:", cruces.max() + 1)
print("validación en la época 100 y 120:", va_[99].round(4), va_[119].round(4))
""", "Pasada esa época la validación casi no cambia entre la 100 y la 120, mientras el entrenamiento sigue cayendo.")

# ================================================================== PARTE F
md("""
## Parte F · Abrir la red y ajustar hiperparámetros

### F1 · Abrir la red (diapositivas 8, 13 y 34)
`coefs_` guarda las matrices de pesos W y `intercepts_` los sesgos. Con ellas se repite la propagación a mano: H = ReLU(X·W₁ + b₁), P = softmax(H·W₂ + b₂), y debe coincidir con `predict_proba`.
""")
code("""
red = R[elegida]["modelo"]
for j, (W, b) in enumerate(zip(red.coefs_, red.intercepts_), 1):
    print(f"W{j}: {W.shape} | b{j}: {b.shape}")


def propagar(X_, m):
    H = relu(X_ @ m.coefs_[0] + m.intercepts_[0])
    return softmax(H @ m.coefs_[1] + m.intercepts_[1]), H


P_manual, H = propagar(X[va], red)
print("H:", H.shape, "| P:", P_manual.shape)
print("diferencia máxima con predict_proba:", np.abs(P_manual - red.predict_proba(X[va])).max())
print("parámetros:", n_parametros(red))
print("filas de P suman 1:", np.allclose(P_manual.sum(1), 1))
""")
md("""
Las dimensiones son **(360, 64) → (360, 32) → (360, 10)** y la propagación manual coincide con `predict_proba` (diferencia ~0). Es exactamente lo que se estudió con números pequeños, solo que más grande. No hay magia: **cada capa es un producto de matrices más un sesgo y una activación**.

### ✍️ Ejercicio F1
Toma la primera imagen de validación: ¿cuántas de las 32 activaciones de la capa oculta son exactamente 0 (ReLU apagadas)? ¿Qué clase predice y con qué probabilidad? ¿Coincide con su etiqueta?
""")
code("""
# tu código aquí
""")
solution("""
i = 0
h_i = H[i]
print("activaciones en 0:", int((h_i == 0).sum()), "de", len(h_i))
p_i = P_manual[i]
print("predice", p_i.argmax(), "con p =", p_i.max().round(4), "| etiqueta real:", y[va][i])
""", "Verás varias activaciones en 0: ReLU apaga las unidades cuya suma es negativa; la salida usa solo las demás.")

md("""
### F2 · El efecto de alpha y del ancho (nuestra reproducción)
**alpha** penaliza la suma de los pesos al cuadrado (no los sesgos); no es la tasa de aprendizaje. Un valor moderado reduce los pesos; uno excesivo impide aprender. Se entrena la red 64–32–10 con la misma receta y solo cambia alpha. (En scikit-learn el término se suma como α/(2·tamaño del lote)·ΣW².)
""")
code("""
filas = []
for a in (0.0, 1e-4, 1e-2, 1.0):
    r = entrenar((32,)) if a == 1e-4 else entrenar((32,), alpha=a)
    suma_w2 = sum((w ** 2).sum() for w in r["final"].coefs_)
    filas.append({"alpha": a, "entrenamiento_120": r["tr"][-1], "validacion_120": r["va"][-1], "suma_W2": suma_w2})
tabla_alpha = pd.DataFrame(filas)
print(tabla_alpha.round(4).to_string(index=False))
""")
md("""
Con la red 64–32–10, la validación en la época 120 es **[[s.a0]]** (alpha = 0), **[[s.a4]]** (0.0001, receta), **[[s.a2]]** (0.01) y **[[s.a10]]** (alpha = 1; la red casi no aprende, con entrenamiento 0.29). La suma de pesos al cuadrado baja al subir alpha. Aquí 0.01 mejora un poco la validación; no es una conclusión general: con otra semilla o más épocas podría cambiar.

### ✍️ Ejercicio F2
Repite la receta con 8 y 64 unidades ocultas (alpha = 0.0001). ¿Qué ancho tiene el menor log loss de validación entre 8, 16, 32 y 64? (Reusa las redes ya entrenadas para 16 y 32.)
""")
code("""
# tu código aquí
""")
solution("""
res = {16: R[16]["val"], 32: R[32]["val"]}
for ancho in (8, 64):
    res[ancho] = entrenar((ancho,))["val"]
for ancho in sorted(res):
    print(ancho, round(res[ancho], 4))
print("mejor ancho:", min(res, key=res.get))
""", "Esperado (nuestra reproducción): 8 → [[s.w8]], 16 → 0.0894, 32 → [[s.w32]], 64 → [[s.w64]]. El mínimo está en 32: más capacidad no siempre mejora la validación.")

# ================================================================== PARTE G
md("""
## Parte G · Comparar con la logística y revisar los errores

### G1 · Logística frente a red (diapositivas 37 a 41)
Se compara la red elegida con la **regresión logística** (referencia) en validación y en prueba. La prueba se mira **una vez, al final**; en la clase ya se había mostrado, así que esta comparación es **retrospectiva**. Se reportan log loss, exactitud y errores.
""")
code("""
with threadpool_limits(limits=1):
    logistica = LogisticRegression(C=1, max_iter=3000, tol=1e-8).fit(X[tr], y[tr])

filas = []
for nombre, m in (("logística", logistica), ("red", red)):
    for conj, mask in (("validación", va), ("prueba", te)):
        P = m.predict_proba(X[mask])
        cl = P.argmax(1)
        filas.append({"modelo": nombre, "conjunto": conj, "log_loss": log_loss(y[mask], P), "exactitud": (cl == y[mask]).mean(), "errores": int((cl != y[mask]).sum())})
print(pd.DataFrame(filas).round(4).to_string(index=False))
""")
md("""
En prueba, la red tiene log loss **[[eval.red.prueba.ll|4]]** frente a **[[eval.log.prueba.ll|4]]** de la logística, pero comete **[[eval.red.prueba.errores]]** errores frente a **[[eval.log.prueba.errores]]**: exactitud **[[s.red_acc]]** y **[[s.log_acc]]**. No es contradictorio: el log loss mira las **probabilidades** de la clase real y la exactitud solo si la clase elegida acertó (diapositiva 38).

### ✍️ Ejercicio G1
Con dos modelos, A y B, y dos casos con y = 1: A da 0.51 y 0.51; B da 0.99 y 0.49. Calcula errores (umbral 0.5) y log loss de cada uno. ¿Cuál tiene menos errores y cuál menos pérdida?
""")
code("""
# tu código aquí
""")
solution("""
for nombre, ps in (("A", [0.51, 0.51]), ("B", [0.99, 0.49])):
    ps = np.array(ps)
    print(nombre, "errores:", int((ps < 0.5).sum()), "| log loss:", (-np.log(ps)).mean().round(4))
""", "A: 0 errores y pérdida [[metricas.A.perdida|4]]; B: 1 error y pérdida [[metricas.B.perdida|4]] (diapositiva 38).")

md("""
### G2 · Matriz de confusión y errores seguros (diapositivas 39 a 41)
La matriz de confusión muestra qué se confunde con qué (fila = real, columna = predicho). Los **errores seguros** (probabilidad máxima alta) pesan mucho en el log loss.
""")
code("""
P = red.predict_proba(X[te])
cl = P.argmax(1)
yt = y[te]
conf = np.zeros((10, 10), dtype=int)
np.add.at(conf, (yt, cl), 1)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(conf, cmap="Blues")
for i in range(10):
    for j in range(10):
        if conf[i, j]:
            ax.text(j, i, conf[i, j], ha="center", va="center", color="white" if conf[i, j] > conf.max() / 2 else "black")
ax.set(xticks=range(10), yticks=range(10), xlabel="predicho", ylabel="real")
ax.grid(False)
plt.tight_layout()
plt.show()

malos = np.where(cl != yt)[0]
tabla_err = pd.DataFrame({"pos": malos, "real": yt[malos], "pred": cl[malos], "p_max": P[malos].max(1).round(2),
                          "perdida": (-np.log(P[malos, yt[malos]])).round(2)})
print(tabla_err.to_string(index=False))
print("errores con p máx ≥ 0.90:", int((tabla_err["p_max"] >= 0.9).sum()))
print("pérdida total de los errores:", tabla_err["perdida"].sum().round(2), "de", (-np.log(P[np.arange(len(yt)), yt])).sum().round(2))

fig, axes = plt.subplots(1, 6, figsize=(10, 2.2))
for ax, i in zip(axes, malos[:6]):
    ax.imshow(X[te][i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"real {yt[i]} → {cl[i]}\\np {P[i].max():.2f}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")
md("""
Hay **[[errores_resumen.n]]** errores en 360 imágenes y **[[errores_resumen.n_conf90]]** con probabilidad máxima ≥ 0.90: el 6 → 1 (p ≈ 0.99, pérdida ≈ 5.75). Los diez errores suman unas **[[errores_resumen.perdida_errores|1]]** unidades de pérdida de **[[errores_resumen.perdida_total|1]]** en total. Los primeros seis (9→8, 1→6, 1→6, 7→8, 5→6, 6→1) son los de la diapositiva 40.

### ✍️ Ejercicio G2
¿Cuántas imágenes de la clase «9» hay en prueba y a cuántas las predice mal la red? ¿A qué dígitos las confunde?
""")
code("""
# tu código aquí
""")
solution("""
mask9 = yt == 9
print("imágenes con etiqueta 9:", int(mask9.sum()), "| mal clasificadas:", int((cl[mask9] != 9).sum()))
print("confundidas con:", sorted(cl[mask9 & (cl != 9)].tolist()))
""", "La fila del 9 de la matriz muestra 33 aciertos, 2 predichos como 3 y 1 como 8.")

# ---------------------------------------------------------------- cierre
md("""
## Ejercicio de cierre · SIN ayuda (25 min)
Usando lo aprendido, **cambia UNA decisión de la receta** (por ejemplo, la tasa de aprendizaje de 0.002 a 0.0005 o a 0.01, o un ancho de 24 en vez de 32) y entrena con `entrenar`. Responde con tu propio código:
1. ¿Cómo cambia el log loss mínimo de validación y la época elegida respecto de Red32?
2. ¿Cambian las curvas de entrenamiento y validación? Descríbelo en una frase.
3. ¿Mirarías la prueba para decidir si tu cambio es mejor? ¿Por qué no?
""")
code("""
# tu código aquí
""")
solution("""
def entrenar_tasa(capas, tasa, epocas=120, semilla=20261020):
    red_ = MLPClassifier(hidden_layer_sizes=capas, activation="relu", solver="adam", learning_rate_init=tasa,
                         batch_size=64, alpha=1e-4, random_state=semilla, shuffle=True, early_stopping=False)
    mejor, ep = np.inf, 0
    with threadpool_limits(limits=1):
        for e in range(1, epocas + 1):
            red_.partial_fit(X[tr], y[tr], classes=np.arange(10))
            lva = log_loss(y[va], red_.predict_proba(X[va]))
            if lva < mejor:
                mejor, ep = lva, e
    return mejor, ep


for tasa in (0.0005, 0.002, 0.01):
    print("tasa", tasa, "->", entrenar_tasa((32,), tasa))
""", "Resultados de nuestra corrida: tasa 0.0005 → 0.1195 (época 120, aún bajando: aprende más lento); 0.002 → 0.0633 (época 118); 0.01 → 0.0728 (mínimo en la época 31: llega antes pero queda peor). La prueba no se mira: la decisión se toma con validación.")

md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué calcula una neurona? ¿Qué son los pesos, el sesgo y la activación?
2. ¿Por qué hace falta una activación no lineal entre capas?
3. ¿Qué dimensiones tienen X, W₁ y H en la red 64–32–10 con 360 imágenes?
4. ¿Cuántos parámetros tiene la 64–32–10? ¿Y la 64–16–10?
5. ¿Qué hace softmax? ¿Cómo se calcula la pérdida de una imagen?
6. ¿Qué indica el gradiente y qué hace η en w⁺ = w − η·J′(w)?
7. ¿Qué calcula la retropropagación y qué hace el optimizador?
8. 1 077 imágenes y lote 64: ¿cuántas actualizaciones por época y en 120 épocas?
9. ¿Con qué se elige entre Red16 y Red32? ¿Por qué no se mira la prueba para elegir?
10. ¿Qué es alpha? ¿Es la tasa de aprendizaje?
11. La red tiene menor log loss pero más errores que la logística: ¿por qué?
12. ¿Por qué un error con probabilidad 0.99 pesa tanto?

<details><summary><b>Respuestas</b></summary>

1. a = Σ wⱼxⱼ + b y h = φ(a). Pesos y sesgo son parámetros que se aprenden; la activación es la salida h de la unidad para esa entrada.
2. Sin ella, dos capas lineales equivalen a una: (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂).
3. X: 360×64; W₁: 64×32; H: 360×32 (y W₂: 32×10; P: 360×10).
4. 2 410 (32·65 + 10·33) y 1 210 (16·65 + 10·17).
5. Convierte puntajes en probabilidades (exponencia y divide por la suma). La pérdida es −log de la probabilidad de la clase real.
6. El gradiente indica hacia dónde sube J; η es el tamaño del paso: muy pequeño es lento y muy grande puede hacer crecer J.
7. Calcula los gradientes de todos los parámetros con la regla de la cadena; el optimizador (aquí Adam) decide cómo cambiarlos.
8. 17 (16 lotes de 64 + 1 de 53) y 120 × 17 = 2 040.
9. Con la validación, por menor log loss sin penalización; la prueba debe quedar independiente para evaluar al final.
10. La penalización L2 de los pesos; no es la tasa de aprendizaje (`learning_rate_init`).
11. El log loss mira las probabilidades de la clase real; la exactitud solo cuenta si la clase elegida acertó; pueden discrepar.
12. Porque −log(0.003) ≈ 5.75: un error muy seguro cuesta muchísimo más que uno dudoso.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 46 (Deep Learning, autoencoders y representaciones): se conserva la mecánica de capas, activaciones y gradientes, pero la salida y la pérdida cambian para reconstruir la entrada. Lectura sugerida por el profesor: Goodfellow, Bengio y Courville, capítulo 14.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
