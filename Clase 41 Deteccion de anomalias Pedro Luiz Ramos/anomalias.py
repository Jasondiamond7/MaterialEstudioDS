# Deteccion de anomalias - Pedro Luiz Ramos

# %% 1. Preparar la sesion
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, roc_auc_score, RocCurveDisplay

os.chdir("D:/Anomalias")

# %% 2. Una variable: RIC y puntaje z
valores = np.array([10, 11, 12, 12, 13, 14, 15, 30])
q1, q3 = np.quantile(valores, [0.25, 0.75])
ric = q3 - q1
limite_inferior = q1 - 1.5 * ric
limite_superior = q3 + 1.5 * ric
alerta_ric = (valores < limite_inferior) | (valores > limite_superior)
puntaje_z = (valores - valores.mean()) / valores.std(ddof=1)
print(pd.DataFrame({"Valor": valores, "Alerta_RIC": alerta_ric,
                    "Z": puntaje_z, "Alerta_Z": abs(puntaje_z) > 3}).round(3))
plt.boxplot(valores, vert=False)
plt.title("Valores para revisar")
plt.xlabel("Valor")
plt.tight_layout()
plt.show()

# %% 3. Dos variables: preparar los mismos puntos para los cuatro metodos
puntos = pd.read_csv("puntos.csv")
x = puntos[["x1", "x2"]]
z = ((x - x.mean()) / x.std(ddof=1)).to_numpy()
plt.scatter(z[:, 0], z[:, 1], s=20)
plt.xlabel("x1 estandarizada")
plt.ylabel("x2 estandarizada")
plt.title("Datos simulados")
plt.tight_layout()
plt.show()

# %% 4. Mahalanobis: calcular la distancia al cuadrado
centro = z.mean(axis=0)
covarianza = np.cov(z, rowvar=False)
inversa = np.linalg.inv(covarianza)
diferencia = z - centro
distancia_m = np.sum((diferencia @ inversa) * diferencia, axis=1)
umbral_m = chi2.ppf(0.99, df=z.shape[1])
alerta_m = distancia_m > umbral_m
print("Umbral de Mahalanobis al cuadrado:", umbral_m)
print(puntos.loc[alerta_m])
# El corte chi-cuadrado es una referencia aproximada bajo normalidad.
plt.scatter(z[:, 0], z[:, 1], c=np.where(alerta_m, "firebrick", "steelblue"), s=20)
plt.xlabel("x1 estandarizada")
plt.ylabel("x2 estandarizada")
plt.title("Mahalanobis: puntos marcados en rojo")
plt.tight_layout()
plt.show()

# %% 5. kNN: distancia al decimo vecino, excluyendo el propio punto
k = 10
vecinos = NearestNeighbors(n_neighbors=k).fit(z)
# Sin pasar X, kneighbors excluye al propio punto de entrenamiento.
distancias, indices = vecinos.kneighbors()
puntaje_knn = distancias[:, -1]
umbral_knn = np.quantile(puntaje_knn, 0.95)
alerta_knn = puntaje_knn > umbral_knn
print(puntos.loc[alerta_knn])
plt.plot(np.sort(puntaje_knn), ".", color="steelblue")
plt.axhline(umbral_knn, color="firebrick", linestyle="--")
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia al vecino 10")
plt.tight_layout()
plt.show()

# %% 6. LOF: comparar la densidad de cada punto con la de sus vecinos
modelo_lof = LocalOutlierFactor(n_neighbors=k)
modelo_lof.fit_predict(z)
puntaje_lof = (
    -modelo_lof.negative_outlier_factor_)
umbral_lof = 1.5
alerta_lof = puntaje_lof > umbral_lof
print(puntos.loc[alerta_lof])
plt.scatter(z[:, 0], z[:, 1], c=np.where(alerta_lof, "firebrick", "steelblue"), s=20)
plt.xlabel("x1 estandarizada")
plt.ylabel("x2 estandarizada")
plt.title("LOF: corte ilustrativo en 1.5")
plt.tight_layout()
plt.show()
# 1.5 es un criterio docente, no un limite universal.

# %% 7. DBSCAN: ruido no significa automaticamente error
epsilon = 0.45
min_puntos = 6
vecinos_db = NearestNeighbors(n_neighbors=min_puntos - 1).fit(z)
distancias_db, _ = vecinos_db.kneighbors()
plt.plot(np.sort(distancias_db[:, -1]), color="steelblue")
plt.axhline(epsilon, color="firebrick", linestyle="--")
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia al vecino 5")
plt.tight_layout()
plt.show()
modelo_db = DBSCAN(eps=epsilon, min_samples=min_puntos).fit(z)
alerta_db = modelo_db.labels_ == -1
print(pd.Series(modelo_db.labels_).value_counts().sort_index())
plt.scatter(z[:, 0], z[:, 1], c=np.where(alerta_db, "firebrick", "steelblue"), s=20)
plt.xlabel("x1 estandarizada")
plt.ylabel("x2 estandarizada")
plt.title("DBSCAN: ruido en rojo")
plt.tight_layout()
plt.show()

revision = pd.DataFrame({"ID": puntos["id"], "Mahalanobis": alerta_m,
                         "kNN": alerta_knn, "LOF": alerta_lof, "DBSCAN": alerta_db})
print(revision.drop(columns="ID").sum())
print(revision.tail(6).to_string(index=False))

# %% 8. PCA con canciones: reconstruir usando las primeras cinco componentes
canciones = pd.read_csv("canciones.csv")
x_canciones = canciones.drop(columns="id")
z_canciones = (x_canciones - x_canciones.mean()) / x_canciones.std(ddof=1)
pca_canciones = PCA().fit(z_canciones)
varianza = pca_canciones.explained_variance_ratio_
print(pd.DataFrame({"Componente": range(1, 8), "Varianza": varianza,
                    "Acumulada": np.cumsum(varianza)}))
r = 5
proyeccion = pca_canciones.transform(z_canciones)[:, :r]
reconstruccion = proyeccion @ pca_canciones.components_[:r] + pca_canciones.mean_
# Q suma residuos al cuadrado en escala estandarizada, no en unidades mixtas.
q_canciones = np.sum((z_canciones.to_numpy() - reconstruccion) ** 2, axis=1)
umbral_canciones = np.quantile(q_canciones, 0.95)
alerta_canciones = q_canciones > umbral_canciones
ranking = pd.DataFrame({"ID": canciones["id"], "Q": q_canciones,
                        "Alerta": alerta_canciones}).sort_values("Q", ascending=False)
print(ranking.head(10).to_string(index=False))
plt.scatter(canciones["id"], q_canciones,
            c=np.where(alerta_canciones, "firebrick", "steelblue"), s=20)
plt.axhline(umbral_canciones, color="firebrick", linestyle="--")
plt.xlabel("Cancion")
plt.ylabel("Q: suma de residuos al cuadrado")
plt.tight_layout()
plt.show()
# Esta revision es exploratoria: no hay etiquetas para calcular acuracia.

# %% 9. Cardio: entrenar y evaluar en conjuntos distintos
cardio = pd.read_csv("cardio.csv")
entrenamiento = cardio[cardio["conjunto"] == "entrenamiento"]
prueba = cardio[cardio["conjunto"] == "prueba"]
variables = [f"v{j}" for j in range(1, 22)]
x_train = entrenamiento[variables]
x_test = prueba[variables]
print("Entrenamiento:", len(x_train), "Prueba:", len(x_test))
# id, conjunto y anomalia no son predictores. No usamos etiquetas para ajustar.
centro = x_train.mean()
escala = x_train.std(ddof=1)
z_train = (x_train - centro) / escala
z_test = (x_test - centro) / escala

# %% 10. PCA: elegir componentes y corte solamente con entrenamiento
pca = PCA().fit(z_train)
acumulada = np.cumsum(pca.explained_variance_ratio_)
r_cardio = int(np.argmax(acumulada >= 0.80) + 1)
cargas = pca.components_[:r_cardio]
recon_train = (z_train.to_numpy() - pca.mean_) @ cargas.T @ cargas + pca.mean_
recon_test = (z_test.to_numpy() - pca.mean_) @ cargas.T @ cargas + pca.mean_
q_train = np.sum((z_train.to_numpy() - recon_train) ** 2, axis=1)
q_test = np.sum((z_test.to_numpy() - recon_test) ** 2, axis=1)
umbral_pca = np.quantile(q_train, 0.95)
clase_pca = (q_test > umbral_pca).astype(int)
print("Componentes de cardio:", r_cardio, "Umbral Q:", umbral_pca)

# %% 11. Isolation Forest clasico: cortes sobre una variable por nodo
modelo_if = IsolationForest(n_estimators=300, max_samples=256,
                             contamination="auto", random_state=2026, n_jobs=1)
modelo_if.fit(z_train)
# sklearn entrega mayor normalidad; cambiamos el signo para ordenar anomalidad.
score_train = -modelo_if.score_samples(z_train)
score_test = -modelo_if.score_samples(z_test)
umbral_if = np.quantile(score_train, 0.95)
clase_if = (score_test > umbral_if).astype(int)
print("Umbral de Isolation Forest:", umbral_if)
# Usamos nuestro cuantil de entrenamiento, no el corte automatico de predict().

# %% 12. Evaluacion: 1 significa anomalia, 0 significa normal
y_test = prueba["anomalia"].to_numpy()
matriz_pca = confusion_matrix(y_test, clase_pca, labels=[0, 1])
matriz_if = confusion_matrix(y_test, clase_if, labels=[0, 1])
print("PCA, filas reales y columnas predichas:\n", matriz_pca)
print("Isolation Forest, filas reales y columnas predichas:\n", matriz_if)
comparacion = pd.DataFrame({
    "Modelo": ["PCA", "Isolation Forest"],
    "Acuracia": [accuracy_score(y_test, clase_pca), accuracy_score(y_test, clase_if)],
    "Sensibilidad": [recall_score(y_test, clase_pca), recall_score(y_test, clase_if)],
    "Precision": [precision_score(y_test, clase_pca), precision_score(y_test, clase_if)],
    "AUC": [roc_auc_score(y_test, q_test), roc_auc_score(y_test, score_test)]})
print(comparacion.round(4).to_string(index=False))
print("Acuracia si marcamos todo normal:", np.mean(y_test == 0))
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, q_test, name="PCA", ax=ax)
RocCurveDisplay.from_predictions(y_test, score_test, name="Isolation Forest", ax=ax)
ax.set_title("Cardio: curvas ROC en prueba")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Sensibilidad")
plt.tight_layout()
plt.show()

# %% 13. Un corte mas alto genera menos alertas; el ranking no cambia
umbrales = np.quantile(score_train, [0.90, 0.95, 0.99])
alertas = []
sensibilidad = []
for umbral in umbrales:
    clase = (score_test > umbral).astype(int)
    alertas.append(int(clase.sum()))
    sensibilidad.append(recall_score(y_test, clase))
print(pd.DataFrame({"Cuantil": [0.90, 0.95, 0.99], "Umbral": umbrales,
                    "Alertas": alertas, "Sensibilidad": sensibilidad}))
# No elegimos el corte con prueba; esta tabla solo ilustra su efecto.
# Los puntajes no son probabilidades: no calculamos log loss ni Brier con ellos.
