"""Autoencoders: leer datos, entrenar, reconstruir y evaluar.

Extrae el ZIP y ejecuta las secciones en orden, desde la carpeta del material.
Instala una vez: python -m pip install numpy==2.3.5 pandas==3.0.1 matplotlib==3.11.2 scikit-learn==1.9.1 threadpoolctl==3.7.0
Para ejecutar todo: python autoencoders.py
Modifica los parámetros de 02 y vuelve a ejecutar desde allí.
Las secciones 07–09 amplían la práctica; prueba queda desactivada en 07.
"""

# %% 00. Bibliotecas y carpeta de trabajo
from pathlib import Path
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")  # En terminal, las figuras se guardan como PNG.
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

CARPETA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# Un hilo basta para estas matrices pequeñas y facilita reproducir la práctica.
limites = threadpool_limits(limits=1)

# %% 01. Lectura de datos y separación de conjuntos
datos = pd.read_csv(CARPETA / "digitos.csv")
X = datos.filter(regex="^px").to_numpy() / 16
y = datos["digito"].to_numpy()
tr = datos["conjunto"] == "entrenamiento"
va = datos["conjunto"] == "validacion"
te = datos["conjunto"] == "prueba"
X_tr = X[tr]
X_va = X[va]
X_te = X[te]
print("Entrenamiento, validación y prueba:", X_tr.shape, X_va.shape, X_te.shape)
# Cada fila es una imagen; las 64 columnas son intensidades entre 0 y 1.
# La etiqueta y se reserva para la clasificación de 08; no entrena el autoencoder.

# %% 02. Parámetros que puedes modificar
DIM_LATENTE = 8
ARQUITECTURAS = [(DIM_LATENTE,), (32, DIM_LATENTE, 32)]
ACTIVACION = "tanh"          # También puedes comparar "relu".
TASA = 0.002                 # Tamaño inicial de los pasos de Adam.
LOTE = 64                    # Imágenes utilizadas por actualización.
ALPHA = 0.0001               # Penalización L2 de los pesos.
EPOCAS = 150                 # Pasadas completas por entrenamiento.
SEMILLA = 20261022
EXPERIMENTO = "base"         # Usa otro nombre para conservar tus resultados.
CUANTIL = 0.95               # Regla fijada antes de mirar prueba.

# Solo indicamos las capas ocultas: entrada y salida tienen 64 valores.
# (8,) representa 64 -> 8 -> 64; (32, 8, 32), 64 -> 32 -> 8 -> 32 -> 64.
# Mantén DIM_LATENTE en el centro al editar estas dos arquitecturas.
SALIDA = CARPETA / "mis_resultados" / EXPERIMENTO
SALIDA.mkdir(parents=True, exist_ok=True)

# %% 03. Crear las redes, entrenar una época y medir validación
historial = []
seleccion = []
mejor_global = np.inf
for capas in ARQUITECTURAS:
    nombre = "AE" + "_".join(map(str, capas))
    # Crear un objeto nuevo inicializa una candidata desde cero.
    red = MLPRegressor(
        hidden_layer_sizes=capas,
        activation=ACTIVACION,
        solver="adam",
        learning_rate_init=TASA,
        batch_size=LOTE,
        alpha=ALPHA,
        random_state=SEMILLA,
        early_stopping=False,
    )
    mejor_mse = np.inf
    for epoca in range(1, EPOCAS + 1):
        # X_tr aparece dos veces: entrada y objetivo de reconstrucción.
        # partial_fit recorre una época y conserva pesos y estado del optimizador.
        red.partial_fit(X_tr, X_tr)
        rec_tr = red.predict(X_tr)
        rec_va = red.predict(X_va)
        mse_tr = mean_squared_error(X_tr, rec_tr)
        mse_va = mean_squared_error(X_va, rec_va)
        historial.append([nombre, epoca, mse_tr, mse_va])
        # Validación solo compara; no modifica los pesos de red.
        if mse_va < mejor_mse:
            mejor_mse = mse_va
            mejor_epoca = epoca
            red_elegida = deepcopy(red)
        if epoca == 1 or epoca % 25 == 0 or epoca == EPOCAS:
            print(nombre, "| época", epoca, "| MSE validación", round(mse_va, 6))
    seleccion.append([nombre, mejor_epoca, mejor_mse])
    if mejor_mse < mejor_global:
        mejor_global = mejor_mse
        modelo = red_elegida
        nombre_elegido = nombre
        epoca_elegida = mejor_epoca

# deepcopy guarda una copia independiente de la mejor época: entrenar después
# no debe cambiar esa copia. No se cargan pesos ni modelos desde archivos.
curvas = pd.DataFrame(historial, columns=["modelo", "epoca", "entrenamiento", "validacion"])
eleccion = pd.DataFrame(seleccion, columns=["modelo", "epoca", "validacion"])
curvas.to_csv(SALIDA / "curvas.csv", index=False)
eleccion.to_csv(SALIDA / "seleccion.csv", index=False)
print(eleccion)
print("Red elegida:", nombre_elegido, "| época:", epoca_elegida)
if epoca_elegida == EPOCAS:
    print("El mínimo coincide con el límite de épocas; no demuestra convergencia.")

# %% 04. Gráfico del aprendizaje
fig, axes = plt.subplots(1, len(ARQUITECTURAS), figsize=(10, 4), squeeze=False, sharey=True)
for ax, nombre in zip(axes.flat, eleccion["modelo"]):
    curva = curvas[curvas["modelo"] == nombre]
    ax.plot(curva["epoca"], curva["entrenamiento"], label="Entrenamiento")
    ax.plot(curva["epoca"], curva["validacion"], label="Validación")
    ax.set(title=nombre, xlabel="Época", ylabel="MSE por píxel")
    ax.legend()
fig.tight_layout()
fig.savefig(SALIDA / "curvas.png", dpi=160)
if "__file__" not in globals():
    plt.show()
plt.close(fig)

# %% 05. Reconstrucción de validación y comparación con PCA
reconstruidas = modelo.predict(X_va)
mse_ae = mean_squared_error(X_va, reconstruidas)

# PCA aprende sus direcciones solo con entrenamiento.
pca = PCA(n_components=DIM_LATENTE, svd_solver="full")
pca.fit(X_tr)
Z_va = pca.transform(X_va)
rec_pca = pca.inverse_transform(Z_va)
mse_pca = mean_squared_error(X_va, rec_pca)
media = X_tr.mean(axis=0)
rec_media = np.tile(media, (len(X_va), 1))
mse_media = mean_squared_error(X_va, rec_media)
comparacion = pd.DataFrame({"modelo": ["Media", "PCA", "Autoencoder"],
                            "mse_validacion": [mse_media, mse_pca, mse_ae]})
print(comparacion)
comparacion.to_csv(SALIDA / "comparacion_validacion.csv", index=False)

# Mostrar las primeras seis imágenes: las columnas comparan el mismo caso.
fig, axes = plt.subplots(3, 6, figsize=(10, 5))
for fila, (titulo, imagenes) in enumerate([
    ("Original", X_va), ("PCA", rec_pca), ("Autoencoder", reconstruidas)
]):
    for columna in range(6):
        ax = axes[fila, columna]
        ax.imshow(imagenes[columna].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if columna == 0:
            ax.set_ylabel(titulo)
fig.tight_layout()
fig.savefig(SALIDA / "reconstrucciones_validacion.png", dpi=160)
if "__file__" not in globals():
    plt.show()
plt.close(fig)
# El gráfico limita los tonos a [0,1]; el MSE usa las salidas completas.

# %% 06. Obtener el vector latente con el encoder ya entrenado
# MLPRegressor.predict devuelve la reconstrucción. Para observar la capa central,
# recorremos solamente las capas del encoder con los pesos aprendidos.
H = X.copy()
capa_latente = len(modelo.hidden_layer_sizes) // 2
for j in range(capa_latente + 1):
    H = H @ modelo.coefs_[j] + modelo.intercepts_[j]
    if ACTIVACION == "tanh":
        H = np.tanh(H)
    else:
        H = np.maximum(H, 0)
print("Una fila por imagen y una columna por coordenada latente:", H.shape)
pd.DataFrame(H[va]).to_csv(SALIDA / "vectores_validacion.csv", index=False)
# Cambia DIM_LATENTE o TASA, cambia EXPERIMENTO y repite 02–06.
# Compara con validación tanto el MSE como las imágenes reconstruidas.

# %% 07. Prueba: ejecutar solo después de fijar la receta
EVALUAR_PRUEBA = False
# En terminal también puedes usar: python autoencoders.py --evaluar-prueba
prueba_habilitada = EVALUAR_PRUEBA or "--evaluar-prueba" in sys.argv
if prueba_habilitada:
    rec_te = modelo.predict(X_te)
    pca_te = pca.inverse_transform(pca.transform(X_te))
    media_te = np.tile(media, (len(X_te), 1))
    prueba = pd.DataFrame({"modelo": ["Media", "PCA", "Autoencoder"], "mse_prueba": [
        mean_squared_error(X_te, media_te), mean_squared_error(X_te, pca_te),
        mean_squared_error(X_te, rec_te)]})
    print(prueba)
    prueba.to_csv(SALIDA / "comparacion_prueba.csv", index=False)
else:
    print("Prueba pendiente. Fija la receta, activa EVALUAR_PRUEBA y ejecuta 07–09.")

# %% 08. Ampliación: ¿sirve la representación para clasificar?
if prueba_habilitada:
    Z = pca.transform(X)
    utilidad = []
    for nombre, representacion in [("Pixeles", X), ("PCA", Z), ("Autoencoder", H)]:
        # Las etiquetas aparecen aquí. PCA y el encoder permanecen fijos.
        clasificador = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=3000, tol=1e-8)
        )
        clasificador.fit(representacion[tr], y[tr])
        probabilidades = clasificador.predict_proba(representacion[te])
        predichas = clasificador.predict(representacion[te])
        utilidad.append([nombre, log_loss(y[te], probabilidades),
                         accuracy_score(y[te], predichas), int(np.sum(predichas != y[te]))])
    utilidad = pd.DataFrame(utilidad, columns=["representacion", "log_loss", "exactitud", "errores"])
    print(utilidad)
    utilidad.to_csv(SALIDA / "utilidad.csv", index=False)

# %% 09. Ampliación: alerta por error grande y desplazamiento artificial
if prueba_habilitada:
    error_va = np.mean((X_va - modelo.predict(X_va)) ** 2, axis=1)
    umbral = np.quantile(error_va, CUANTIL)
    # Mover cada imagen una columna a la derecha; rellenar con ceros la izquierda.
    desplazadas = np.zeros((len(X_te), 8, 8))
    desplazadas[:, :, 1:] = X_te.reshape(-1, 8, 8)[:, :, :-1]
    desplazadas = desplazadas.reshape(-1, 64)
    error_original = np.mean((X_te - modelo.predict(X_te)) ** 2, axis=1)
    error_desplazada = np.mean((desplazadas - modelo.predict(desplazadas)) ** 2, axis=1)
    # Cada salida se compara con su propia entrada: aquí no se entrena a quitar ruido.
    estres = pd.DataFrame({"error_original": error_original, "error_desplazada": error_desplazada})
    estres.to_csv(SALIDA / "estres.csv", index=False)
    alertas = pd.DataFrame({"umbral": [umbral], "total": [len(X_te)],
                           "originales": [int(np.sum(error_original > umbral))],
                           "desplazadas": [int(np.sum(error_desplazada > umbral))]})
    print(alertas)
    alertas.to_csv(SALIDA / "alertas.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(error_original, bins=22, alpha=0.65, label="Original")
    ax.hist(error_desplazada, bins=22, alpha=0.6, label="Desplazada")
    ax.axvline(umbral, color="green", linestyle="--", label="Umbral de validación")
    ax.set(xlabel="MSE por imagen", ylabel="Imágenes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "estres.png", dpi=160)
    if "__file__" not in globals():
        plt.show()
    plt.close(fig)
    print("Estrés artificial: no mide detección de defectos reales.")
    print("Selección y umbral usan la misma validación; no garantizan falsas alarmas futuras.")
print("La prueba ya se mostró en clase: esta comparación es retrospectiva.")
print("Resultados en", SALIDA)
