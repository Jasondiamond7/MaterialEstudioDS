"""Redes neuronales: calcular, entrenar, elegir y evaluar.

Ejecutar las secciones en orden. Solo se necesita digitos.csv como entrada.
La configuración editable está en 03; la evaluación de prueba se habilita en 07.
Requiere Python 3.11 o posterior. Instala una vez desde la terminal:
python -m pip install numpy==2.3.5 pandas==3.0.1 matplotlib==3.11.2 scikit-learn==1.9.1 threadpoolctl==3.7.0
Extrae el ZIP completo y ejecuta: python redes_neuronales.py
Para evaluar la receta elegida: python redes_neuronales.py --evaluar-prueba
R y Python entrenan por separado; las cifras proyectadas usan esta receta base.
"""
# %% 00. Datos y particiones — diapositivas 3, 4 y 31
from pathlib import Path
from copy import deepcopy
import argparse
import json
import math
import re
import numpy as np
import pandas as pd
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")  # En terminal guardamos las figuras; en el cuaderno se muestran.
elif "get_ipython" in globals():
    get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (ROOT / "digitos.csv").exists():
    raise FileNotFoundError("Conserva este programa y digitos.csv en la misma carpeta.")
datos = pd.read_csv(ROOT / "digitos.csv")
# Un píxel con intensidad 16 pasa a valer 1. La etiqueta no forma parte de X.
X = datos.filter(regex="^px").to_numpy(dtype=float) / 16
y = datos.digito.to_numpy(dtype=int)
tr = datos.conjunto.eq("entrenamiento").to_numpy()
va = datos.conjunto.eq("validacion").to_numpy()
te = datos.conjunto.eq("prueba").to_numpy()
assert X.shape == (1797, 64)
assert (tr.sum(), va.sum(), te.sum()) == (1077, 360, 360)
assert np.all(tr.astype(int) + va.astype(int) + te.astype(int) == 1)
print("Entrada:", X.shape, "| entrenamiento:", tr.sum(), "| validación:", va.sum(), "| prueba:", te.sum())


def log_loss(reales, probabilidades):
    """Promedio de -log(probabilidad asignada a la clase observada)."""
    p_real = probabilidades[np.arange(len(reales)), reales]
    return float(-np.log(np.clip(p_real, 1e-15, 1)).mean())


def metricas(reales, probabilidades):
    clases = probabilidades.argmax(axis=1)
    return {"log_loss": log_loss(reales, probabilidades),
            "exactitud": float(np.mean(clases == reales)),
            "errores": int(np.sum(clases != reales))}


# %% 01. De una entrada a una probabilidad — diapositivas 8 y 14–15
entrada = np.array([[1., 2.]])
W1 = np.array([[1., -1.], [.5, .5]])
H = np.maximum(entrada @ W1, 0)
puntaje = (H @ np.array([[.5], [-1.]]) - .5).item()
probabilidad = 1 / (1 + np.exp(-puntaje))
print("Activaciones:", H, "| puntaje:", puntaje, "| probabilidad:", probabilidad)
# Esperamos [2, 0], 0.5 y 0.622459. Todavía no hemos entrenado la red de dígitos.


# %% 02. Seguir un gradiente — diapositivas 19–24
# Ejemplo binario con x=0.5, y=1; theta contiene (w1, b1, w2, b2).
theta = np.array([.4, .1, .8, -.2])
tasa_ejemplo = .1


def perdida_pequena(t):
    h = max(t[0] * .5 + t[1], 0)
    return np.logaddexp(0, -(t[2] * h + t[3]))


a1 = theta[0] * .5 + theta[1]
h = max(a1, 0)
p = 1 / (1 + np.exp(-(theta[2] * h + theta[3])))
delta_salida = p - 1
delta_oculta = delta_salida * theta[2] * float(a1 > 0)
gradiente = np.array([delta_oculta * .5, delta_oculta, delta_salida * h, delta_salida])
epsilon = 1e-6
numerico = np.array([
    (perdida_pequena(theta + epsilon * v) - perdida_pequena(theta - epsilon * v)) / (2 * epsilon)
    for v in np.eye(4)
])
assert np.max(abs(gradiente - numerico)) < 1e-8
actualizados = theta - tasa_ejemplo * gradiente
print("Gradiente:", gradiente, "| parámetros después del paso:", actualizados)
print("Pérdida antes/después:", perdida_pequena(theta), perdida_pequena(actualizados))
# El paso manual es descenso de gradiente. La red de dígitos usará Adam.
# Cambiar tasa_ejemplo permite explorar este ejemplo, sin cambiar la tasa de Adam.


# %% 03. Configurar nuestra red — diapositivas 9, 13, 25–29 y 33
# MODIFICA AQUÍ. Cada tupla describe una candidata: (32,) tiene una capa oculta;
# (32, 16) tendría dos. Primero ejecuta la receta base; luego cambia una decisión.
CAPAS_CANDIDATAS = [(16,), (32,)]
ACTIVACION = "relu"           # "relu" o "tanh"; la salida sigue siendo softmax.
TASA = .002                   # Tasa inicial de Adam.
LOTE = 64                     # Imágenes por actualización.
ALPHA = .0001                 # Penalización L2 de los pesos en scikit-learn.
EPOCAS = 120                  # Pasadas completas por entrenamiento, por candidata.
SEMILLA = 20261020            # Inicialización y orden de las observaciones.
EXPERIMENTO = "base"          # Cambia el nombre para conservar cada comparación.

# La terminal también permite cambiar estos parámetros mediante argumentos.
parser = argparse.ArgumentParser(description="Entrenar una red de dígitos desde cero.")
parser.add_argument("--capas", nargs="+", default=[",".join(map(str, c)) for c in CAPAS_CANDIDATAS])
parser.add_argument("--activacion", choices=["relu", "tanh"], default=ACTIVACION)
parser.add_argument("--tasa", type=float, default=TASA)
parser.add_argument("--lote", type=int, default=LOTE)
parser.add_argument("--alpha", type=float, default=ALPHA)
parser.add_argument("--epocas", type=int, default=EPOCAS)
parser.add_argument("--semilla", type=int, default=SEMILLA)
parser.add_argument("--experimento", default=EXPERIMENTO)
parser.add_argument("--evaluar-prueba", action="store_true")
# En Jupyter no usamos los argumentos del proceso que inició el kernel.
args = parser.parse_args() if "__file__" in globals() else parser.parse_args([])
capas = [tuple(int(v) for v in item.split(",")) for item in args.capas]
if not all(c and all(w > 0 for w in c) for c in capas):
    raise ValueError("Cada candidata debe tener al menos una capa de ancho positivo.")
if len(set(capas)) != len(capas):
    raise ValueError("Incluye cada arquitectura una sola vez.")
if args.epocas < 1 or not 1 <= args.lote <= int(tr.sum()):
    raise ValueError("Usa épocas positivas y un lote entre 1 y 1077.")
if not np.isfinite(args.tasa) or args.tasa <= 0 or not np.isfinite(args.alpha) or args.alpha < 0:
    raise ValueError("La tasa debe ser finita y positiva; alpha, finito y no negativo.")
if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experimento):
    raise ValueError("Nombra el experimento con letras sin tildes, números, guion o guion bajo.")
OUT = ROOT / "mis_resultados" / args.experimento
OUT.mkdir(parents=True, exist_ok=True)
# Una nueva ejecución con el mismo nombre reemplaza esa comparación.
# Retirar solo salidas de prueba previas evita confundirlas con la nueva receta.
for nombre_archivo in ["metricas.csv", "confusion.png", "errores.csv"]:
    (OUT / nombre_archivo).unlink(missing_ok=True)
receta = {"capas": capas, "activacion": args.activacion, "tasa": args.tasa,
          "lote": args.lote, "alpha": args.alpha, "epocas": args.epocas, "semilla": args.semilla}
print("Receta:", receta)
print("Lotes por época:", math.ceil(tr.sum() / args.lote),
      "| actualizaciones por candidata:", args.epocas * math.ceil(tr.sum() / args.lote))
# La penalización afecta al ajuste; la selección usa log loss sin penalización.


# %% 04. Entrenar desde cero y conservar la mejor época — diapositivas 25–26, 30–31 y 35–36
historial = []
seleccion = []
modelos_elegidos = {}
mejor_global = np.inf
with threadpool_limits(limits=1):
    for arquitectura in capas:
        nombre = "Red" + "_".join(map(str, arquitectura))
        # Este constructor crea un modelo nuevo: no leemos pesos guardados.
        red = MLPClassifier(hidden_layer_sizes=arquitectura, activation=args.activacion,
                            solver="adam", learning_rate_init=args.tasa, batch_size=args.lote,
                            alpha=args.alpha, random_state=args.semilla,
                            shuffle=True, early_stopping=False)
        mejor_validacion = np.inf
        for epoca in range(1, args.epocas + 1):
            # partial_fit recorre una época; dentro calcula gradientes por lotes y aplica Adam.
            # Se llama sobre la MISMA red para conservar pesos y estado del optimizador.
            red.partial_fit(X[tr], y[tr], classes=np.arange(10))
            perdida_tr = log_loss(y[tr], red.predict_proba(X[tr]))
            perdida_va = log_loss(y[va], red.predict_proba(X[va]))
            historial.append({"modelo": nombre, "epoca": epoca,
                              "entrenamiento": perdida_tr, "validacion": perdida_va})
            if perdida_va < mejor_validacion:
                mejor_validacion = perdida_va
                mejor_epoca = epoca
                # Copiar: si guardáramos solo una referencia, cambiaría en la próxima época.
                copia_elegida = deepcopy(red)
            if epoca == 1 or epoca % 20 == 0 or epoca == args.epocas:
                print(f"{nombre} | época {epoca:3d} | entrenamiento {perdida_tr:.5f} | validación {perdida_va:.5f}")
        seleccion.append({"modelo": nombre, "epoca": mejor_epoca, "log_loss": mejor_validacion,
                          "parametros": sum(w.size + b.size for w, b in zip(red.coefs_, red.intercepts_))})
        modelos_elegidos[nombre] = copia_elegida
        if mejor_validacion < mejor_global:
            mejor_global = mejor_validacion
            modelo_final = copia_elegida
            nombre_final = nombre
            epoca_final = mejor_epoca
# La prueba no aparece en este bucle. No reajustamos con entrenamiento + validación.
curvas = pd.DataFrame(historial)
eleccion = pd.DataFrame(seleccion)
curvas.to_csv(OUT / "curvas.csv", index=False)
eleccion.to_csv(OUT / "seleccion.csv", index=False)
print("\nElección por validación")
print(eleccion.to_string(index=False))
print("Modelo elegido:", nombre_final, "| época:", epoca_final)


# %% 05. Leer las curvas y comparar — diapositivas 30, 35 y 36
fig, axes = plt.subplots(1, len(capas), figsize=(5 * len(capas), 3.6), squeeze=False, sharey=True)
for ax, fila in zip(axes.flat, seleccion):
    g = curvas[curvas.modelo.eq(fila["modelo"])]
    ax.plot(g.epoca, g.entrenamiento, label="Entrenamiento")
    ax.plot(g.epoca, g.validacion, label="Validación")
    ax.axvline(fila["epoca"], linestyle="--", color="gray", label="Época elegida")
    ax.set(title=fila["modelo"], xlabel="Época", ylabel="Log loss sin penalización")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "curvas.png", dpi=160)
if "__file__" not in globals():
    plt.show()
plt.close(fig)
# Antes de usar prueba, cambia UNA decisión en 03 y vuelve a ejecutar 03–06.
# Usa otro nombre de experimento. Compara el mínimo de validación y ambas curvas.
# Las cifras de las diapositivas corresponden a la receta base, no a toda modificación.


# %% 06. Abrir la red que acabamos de entrenar — diapositivas 8, 14, 17 y 34
def propagar_con_pesos(entrada, modelo):
    """Repetir las operaciones matriciales con los parámetros recién aprendidos."""
    a = entrada.copy()
    dimensiones = [a.shape]
    for j, (w, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_)):
        a = a @ w + b
        if j < len(modelo.coefs_) - 1:
            a = np.maximum(a, 0) if modelo.activation == "relu" else np.tanh(a)
            dimensiones.append(a.shape)
    exponenciales = np.exp(a - a.max(axis=1, keepdims=True))
    p = exponenciales / exponenciales.sum(axis=1, keepdims=True)
    return p, dimensiones + [p.shape]


# Comprobamos con validación: no necesitamos abrir prueba para revisar las matrices.
calculadas, dimensiones = propagar_con_pesos(X[va], modelo_final)
referencia = modelo_final.predict_proba(X[va])
diferencia = float(np.max(abs(calculadas - referencia)))
assert diferencia < 1e-10
print("Dimensiones en validación:", dimensiones)
print("Diferencia entre propagación explícita y predict_proba:", diferencia)
pesos_exportados = {}
for j, (w, b) in enumerate(zip(modelo_final.coefs_, modelo_final.intercepts_), 1):
    pesos_exportados[f"W{j}"] = w
    pesos_exportados[f"b{j}"] = b
np.savez(OUT / "pesos_aprendidos.npz", **pesos_exportados)
resumen = {"receta": receta, "red": nombre_final, "epoca": epoca_final,
           "log_loss_validacion": mejor_global, "seleccion": seleccion,
           "prueba_evaluada": False, "diferencia_propagacion": diferencia,
           "versiones": {"numpy": np.__version__, "sklearn": sklearn.__version__}}
(OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")


# %% 07. Evaluar la decisión fijada — diapositivas 37–41 y 43–46
# En el cuaderno, cambia a True solo después de cerrar las decisiones con validación.
# Desde terminal puedes añadir --evaluar-prueba a la ejecución de la receta fijada.
EVALUAR_PRUEBA = False
if EVALUAR_PRUEBA or args.evaluar_prueba:
    filas = []
    with threadpool_limits(limits=1):
        logistica = LogisticRegression(C=1, max_iter=3000, tol=1e-8).fit(X[tr], y[tr])
        for nombre, modelo in [("Logistica", logistica), ("Red", modelo_final)]:
            for conjunto, mascara in [("validacion", va), ("prueba", te)]:
                filas.append({"modelo": nombre, "conjunto": conjunto,
                              **metricas(y[mascara], modelo.predict_proba(X[mascara]))})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "metricas.csv", index=False)
    print("\nEvaluación de la receta fijada:\n", tabla.to_string(index=False))
    probabilidades = modelo_final.predict_proba(X[te])
    clases = probabilidades.argmax(axis=1)
    matriz = np.zeros((10, 10), dtype=int)
    np.add.at(matriz, (y[te], clases), 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matriz, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if matriz[i, j]:
                ax.text(j, i, matriz[i, j], ha="center", va="center",
                        color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicho", ylabel="Real")
    fig.tight_layout(); fig.savefig(OUT / "confusion.png", dpi=160)
    if "__file__" not in globals():
        plt.show()
    plt.close(fig)
    errores = datos.loc[te, ["id", "digito"]].copy()
    errores["predicho"] = clases
    errores["p_max"] = probabilidades.max(axis=1)
    errores = errores[errores.digito.ne(errores.predicho)]
    errores.to_csv(OUT / "errores.csv", index=False)
    print("\nPrimeros seis errores:\n", errores.head(6).to_string(index=False))
    resumen["prueba_evaluada"] = True
    resumen["metricas"] = filas
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("La prueba de este caso ya se mostró en clase: esta comparación es retrospectiva.")
else:
    print("Entrenamiento terminado. Prueba pendiente: al fijar tu receta, habilita EVALUAR_PRUEBA en 07.")
print("Tus resultados están en mis_resultados/" + args.experimento)
