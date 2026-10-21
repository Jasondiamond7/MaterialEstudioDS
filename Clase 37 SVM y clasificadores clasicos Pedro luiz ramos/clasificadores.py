# CLASIFICADORES: REGRESION LOGISTICA Y SVM
# Ejecutar de arriba hacia abajo. Cada # %% inicia un bloque.
# En toda la clase: 0 = benigno; 1 = maligno.

# %% 1. Preparar la sesion
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

os.chdir(r"D:\codigos")

# %% 2. Leer y mirar los datos
datos = pd.read_csv("wisconsin_clase.csv")
print(datos.head())
print(datos["maligno"].value_counts())

# El color indica la clase real. Estos son solo dos de los cuatro predictores.
benignos = datos[datos["maligno"] == 0]
malignos = datos[datos["maligno"] == 1]
plt.figure()
plt.scatter(benignos["radio"], benignos["textura"], color="steelblue", label="Benigno")
plt.scatter(malignos["radio"], malignos["textura"], color="tomato", label="Maligno")
plt.xlabel("Radio")
plt.ylabel("Textura")
plt.title("Datos: clases reales")
plt.legend()
plt.tight_layout()
plt.show()

# %% 3. Separar entrenamiento y prueba
# La particion ya viene en el CSV: 398 casos para aprender y 171 para evaluar.
# No usamos los casos de prueba para ajustar los modelos.
entrenamiento = datos[datos["conjunto"] == "entrenamiento"]
prueba = datos[datos["conjunto"] == "prueba"]

variables = ["radio", "textura", "suavidad", "concavidad"]
x_entrenamiento = entrenamiento[variables]
x_prueba = prueba[variables]
y_entrenamiento = entrenamiento["maligno"]
y_prueba = prueba["maligno"]

# fit_transform aprende la media y la escala SOLO con entrenamiento.
# transform aplica esa misma transformacion a los casos de prueba.
escala = StandardScaler()
x_entrenamiento_escalado = escala.fit_transform(x_entrenamiento)
x_prueba_escalado = escala.transform(x_prueba)

# %% 4. Regresion logistica
# penalty=None evita agregar la penalizacion que Python usa por defecto.
# Asi ajustamos una logistica sin penalizacion, como glm de R.
logistica = LogisticRegression(penalty=None, max_iter=2000)
logistica.fit(x_entrenamiento_escalado, y_entrenamiento)

# predict_proba entrega dos columnas: P(clase = 0) y P(clase = 1).
probabilidad = logistica.predict_proba(x_prueba_escalado)[:, 1]
print(probabilidad[:6])

# Fijamos el umbral en 0.5: una probabilidad de 0.7 se clasifica como 1.
clase_logistica = (probabilidad >= 0.5).astype(int)
matriz_logistica = confusion_matrix(y_prueba, clase_logistica, labels=[0, 1])
print("Logistica: filas = real; columnas = prediccion; orden = 0, 1")
print(matriz_logistica)

# Python cuenta desde 0: [0,0] = VN; [0,1] = FP; [1,0] = FN; [1,1] = VP.
# Exactitud = (VN + VP) / total.
# Sensibilidad = VP / (VP + FN): proporcion de malignos detectados.
# Especificidad = VN / (VN + FP): proporcion de benignos bien clasificados.

# %% 5. SVM lineal
# C de Python equivale a cost de R. SVC entrega clases con predict.
svm_lineal = SVC(kernel="linear", C=0.1)
svm_lineal.fit(x_entrenamiento_escalado, y_entrenamiento)
clase_lineal = svm_lineal.predict(x_prueba_escalado)
matriz_lineal = confusion_matrix(y_prueba, clase_lineal, labels=[0, 1])
print("SVM lineal: filas = real; columnas = prediccion; orden = 0, 1")
print(matriz_lineal)

# %% 6. SVM radial
# C penaliza las violaciones del margen; gamma controla el alcance de la similitud.
# Usamos las configuraciones de la clase, sin elegirlas mirando prueba.
svm_radial = SVC(kernel="rbf", C=10, gamma=0.1)
svm_radial.fit(x_entrenamiento_escalado, y_entrenamiento)
clase_radial = svm_radial.predict(x_prueba_escalado)
matriz_radial = confusion_matrix(y_prueba, clase_radial, labels=[0, 1])
print("SVM radial: filas = real; columnas = prediccion; orden = 0, 1")
print(matriz_radial)

# %% 7. Comparar los resultados
# recall_score con pos_label=1 calcula sensibilidad; con 0, especificidad.
comparacion = pd.DataFrame({
    "Modelo": ["Logistica", "SVM lineal", "SVM radial"],
    "Exactitud": [accuracy_score(y_prueba, clase_logistica),
                  accuracy_score(y_prueba, clase_lineal),
                  accuracy_score(y_prueba, clase_radial)],
    "Sensibilidad": [recall_score(y_prueba, clase_logistica, pos_label=1),
                     recall_score(y_prueba, clase_lineal, pos_label=1),
                     recall_score(y_prueba, clase_radial, pos_label=1)],
    "Especificidad": [recall_score(y_prueba, clase_logistica, pos_label=0),
                      recall_score(y_prueba, clase_lineal, pos_label=0),
                      recall_score(y_prueba, clase_radial, pos_label=0)]
})
print(comparacion.round(3).to_string(index=False))

# Mirar sensibilidad ademas de exactitud: los errores no tienen el mismo costo.
plt.figure()
plt.bar(comparacion["Modelo"], comparacion["Sensibilidad"],
        color=["steelblue", "darkorange", "seagreen"])
plt.ylim(0, 1)
plt.ylabel("Sensibilidad")
plt.title("Malignos detectados en prueba")
plt.tight_layout()
plt.show()

# %% 8. Predecir un caso nuevo
nuevo = pd.DataFrame({"radio": [14], "textura": [20],
                      "suavidad": [0.10], "concavidad": [0.08]})
nuevo_escalado = escala.transform(nuevo)
print(logistica.predict_proba(nuevo_escalado)[:, 1])  # Probabilidad de maligno.
print(svm_radial.predict(nuevo_escalado))            # Clase: 0 o 1.
# Los modelos pueden discrepar. Este ejemplo es docente, no un diagnostico.

# %% 9. AUC de los tres modelos
# La AUC usa probabilidades o puntajes continuos, NO las clases 0/1.
# Compara el ordenamiento entre benignos y malignos al variar el umbral.
from sklearn.metrics import roc_auc_score

auc_logistica = roc_auc_score(y_prueba, probabilidad)

# decision_function entrega un puntaje: valores mayores favorecen la clase 1.
# Estos puntajes de SVM no son probabilidades.
puntaje_lineal = svm_lineal.decision_function(x_prueba_escalado)
puntaje_radial = svm_radial.decision_function(x_prueba_escalado)
auc_lineal = roc_auc_score(y_prueba, puntaje_lineal)
auc_radial = roc_auc_score(y_prueba, puntaje_radial)

comparacion["AUC"] = [auc_logistica, auc_lineal, auc_radial]
print(comparacion.round(4).to_string(index=False))
# AUC = 0.5: ordenamiento al azar como referencia; AUC = 1: ordenamiento perfecto.
# Una AUC de 0.98 NO significa 98% de clasificaciones correctas.

# %% OPCIONAL, SI QUEDA TIEMPO: validacion cruzada
# Quitar # a las siguientes lineas para probar combinaciones con 5 folds.
# Esta busqueda usa SOLO entrenamiento y maximiza la exactitud (1 - error).
# Es una demostracion breve; no modifica los modelos del ejemplo principal.
# El pipeline vuelve a estimar la escala dentro de cada fold: recibe datos sin escalar.
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import GridSearchCV
# modelo = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
# busqueda = GridSearchCV(modelo,
#                         {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]},
#                         cv=5, scoring="accuracy")
# busqueda.fit(x_entrenamiento, y_entrenamiento)
# print(busqueda.best_params_)

# OPCIONAL: guardar solo esta tabla en la carpeta indicada al inicio.
# comparacion.to_csv("comparacion_Python.csv", index=False)
