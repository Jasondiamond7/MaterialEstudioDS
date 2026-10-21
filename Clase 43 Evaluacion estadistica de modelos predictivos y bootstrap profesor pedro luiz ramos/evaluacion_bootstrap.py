# Evaluacion estadistica y bootstrap - Pedro Luiz Ramos
# Ejecutar con esta carpeta como directorio de trabajo.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss, roc_auc_score

# 1. Datos de las clases anteriores
datos = pd.read_csv("wisconsin.csv")
entrenamiento = datos[datos["conjunto"] == "entrenamiento"].copy()
prueba = datos[datos["conjunto"] == "prueba"].copy()
print(pd.crosstab(datos["conjunto"], datos["maligno"]))

# La prueba ya fue revisada en clase. Este ejercicio es una demostracion.
# id, conjunto y fold no son variables del modelo.

# 2. Tres logisticas sin penalizacion: comparar AIC y BIC
modelo_simple = smf.glm(
    "maligno ~ radio + textura", data=entrenamiento,
    family=sm.families.Binomial()).fit()
modelo_intermedio = smf.glm(
    "maligno ~ radio + textura + suavidad", data=entrenamiento,
    family=sm.families.Binomial()).fit()
modelo_amplio = smf.glm(
    "maligno ~ radio + textura + suavidad + concavidad", data=entrenamiento,
    family=sm.families.Binomial()).fit()
criterios = pd.DataFrame({
    "Modelo": ["Simple", "Intermedio", "Amplio"],
    "Parametros": [len(modelo_simple.params), len(modelo_intermedio.params),
                   len(modelo_amplio.params)],
    "AIC": [modelo_simple.aic, modelo_intermedio.aic, modelo_amplio.aic],
    "BIC": [modelo_simple.bic_llf, modelo_intermedio.bic_llf,
            modelo_amplio.bic_llf]})
print(criterios.round(4).to_string(index=False))

# bic_llf usa -2*logLik + log(n)*p, la misma definicion que BIC() en R.
# Ambos criterios usan solo los 398 casos de entrenamiento; menor es mejor.

# 3. Predicciones de los modelos simple y amplio
p_simple = modelo_simple.predict(prueba).to_numpy()
p_amplio = modelo_amplio.predict(prueba).to_numpy()
y = prueba["maligno"].to_numpy()
umbral = 0.5
comparacion = pd.DataFrame({
    "Modelo": ["Simple", "Amplio"],
    "AUC": [roc_auc_score(y, p_simple), roc_auc_score(y, p_amplio)],
    "LogLoss": [log_loss(y, p_simple), log_loss(y, p_amplio)],
    "Exactitud": [np.mean((p_simple >= umbral) == y),
                  np.mean((p_amplio >= umbral) == y)],
    "Sensibilidad": [np.mean(p_simple[y == 1] >= umbral),
                     np.mean(p_amplio[y == 1] >= umbral)]})
print(comparacion.round(4).to_string(index=False))

# 4. Bootstrap de una media: muestrear con reposicion
valores = np.array([2, 4, 5, 7, 8])
B = 2000
rng = np.random.default_rng(20261012)
medias_boot = np.empty(B)
for b in range(B):
    muestra = rng.choice(valores, size=len(valores), replace=True)
    medias_boot[b] = np.mean(muestra)
print("Media:", np.mean(valores), "Error estandar:", np.std(medias_boot, ddof=1))
print("IC percentil:", np.quantile(medias_boot, [0.025, 0.975]))
plt.hist(medias_boot, bins=20, color="steelblue")
plt.axvline(np.mean(valores), color="firebrick")
plt.title("Bootstrap de la media")
plt.xlabel("Media de la remuestra")
plt.show()

# 5. Log loss: remuestrear filas de prueba, sin ajustar de nuevo
# Las probabilidades de prueba quedan fijas. No repetimos fit() ni predict().
n = len(y)
rng = np.random.default_rng(20261013)
logloss_boot = np.empty((B, 2))
for b in range(B):
    indices = rng.choice(n, size=n, replace=True)
    logloss_boot[b, 0] = log_loss(y[indices], p_simple[indices])
    logloss_boot[b, 1] = log_loss(y[indices], p_amplio[indices])
ic_logloss = np.quantile(logloss_boot, [0.025, 0.975], axis=0)
print("IC log loss; columnas Simple, Amplio:\n", ic_logloss)
plt.hist(logloss_boot[:, 1], bins=25, color="steelblue")
for corte in ic_logloss[:, 1]:
    plt.axvline(corte, color="firebrick", linestyle="--")
plt.title("Log loss: modelo amplio")
plt.xlabel("Log loss de la remuestra")
plt.show()

# 6. AUC: bootstrap de prueba dentro de cada clase
# Sorteamos 107 benignos y 64 malignos con reposicion; ambos modelos usan los mismos IDs.
negativos = np.flatnonzero(y == 0)
positivos = np.flatnonzero(y == 1)
rng = np.random.default_rng(20261017)
auc_boot = np.empty((B, 2))
for b in range(B):
    i0 = rng.choice(negativos, size=len(negativos), replace=True)
    i1 = rng.choice(positivos, size=len(positivos), replace=True)
    i = np.concatenate([i0, i1])
    auc_boot[b, 0] = roc_auc_score(y[i], p_simple[i])
    auc_boot[b, 1] = roc_auc_score(y[i], p_amplio[i])
ic_auc = np.quantile(auc_boot, [0.025, 0.975], axis=0)
print("IC AUC; columnas Simple, Amplio:\n", ic_auc)
plt.hist(auc_boot[:, 1], bins=25, color="steelblue")
for corte in ic_auc[:, 1]:
    plt.axvline(corte, color="firebrick", linestyle="--")
plt.title("AUC: modelo amplio")
plt.xlabel("AUC de la remuestra de prueba")
plt.show()

# No entrenamos nuevamente. Este IC estudia la evaluacion de los modelos ya ajustados.

# 7. Comparacion pareada: los dos modelos ven los mismos casos
delta_logloss = logloss_boot[:, 0] - logloss_boot[:, 1]
delta_auc = auc_boot[:, 1] - auc_boot[:, 0]

# Positivo favorece al amplio: menor log loss o mayor AUC.
diferencias = pd.DataFrame({
    "Metrica": ["Log loss simple - amplio", "AUC amplio - simple"],
    "Diferencia": [comparacion.loc[0, "LogLoss"] - comparacion.loc[1, "LogLoss"],
                   comparacion.loc[1, "AUC"] - comparacion.loc[0, "AUC"]],
    "Inferior": [np.quantile(delta_logloss, 0.025), np.quantile(delta_auc, 0.025)],
    "Superior": [np.quantile(delta_logloss, 0.975), np.quantile(delta_auc, 0.975)]})
print(diferencias.round(4).to_string(index=False))
plt.hist(delta_logloss, bins=25, color="steelblue")
plt.axvline(0, color="firebrick")
for corte in np.quantile(delta_logloss, [0.025, 0.975]):
    plt.axvline(corte, color="darkgreen", linestyle="--")
plt.title("Diferencia pareada de log loss")
plt.xlabel("Log loss simple - amplio")
plt.show()
plt.hist(delta_auc, bins=25, color="steelblue")
plt.axvline(0, color="firebrick")
for corte in np.quantile(delta_auc, [0.025, 0.975]):
    plt.axvline(corte, color="darkgreen", linestyle="--")
plt.title("Diferencia pareada de AUC")
plt.xlabel("AUC amplio - simple")
plt.show()

# Los intervalos usan modelos ya ajustados. No incluyen la variacion del entrenamiento.
# R y Python ajustan los mismos GLM, pero generan remuestras diferentes.

# 8. Ejemplo numerico de regresion: MAE y RMSE
# Predicciones hipoteticas y fijas para cinco casos; no ajustamos otro modelo.
observado = np.array([10, 20, 30, 40, 50])
predicho = np.array([9, 22, 27, 44, 45])
errores = observado - predicho
print("MAE:", np.mean(np.abs(errores)), "RMSE:", np.sqrt(np.mean(errores**2)))

rng = np.random.default_rng(20261014)
regresion_boot = np.empty((B, 2))
for b in range(B):
    indices = rng.choice(len(observado), size=len(observado), replace=True)
    errores_b = observado[indices] - predicho[indices]
    regresion_boot[b, 0] = np.mean(np.abs(errores_b))
    regresion_boot[b, 1] = np.sqrt(np.mean(errores_b**2))
print("IC MAE, RMSE:\n", np.quantile(regresion_boot, [0.025, 0.975], axis=0))
# Recalculamos la raiz en cada remuestra. Con cinco casos, el IC es ilustrativo.
