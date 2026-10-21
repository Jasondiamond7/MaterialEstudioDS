# Deteccion de anomalias - Pedro Luiz Ramos
# Instalar una vez: install.packages(c("dbscan", "isotree", "pROC"))

# 1. Preparar la sesion ---------------------------------------------------
setwd("D:/Anomalias")
library(dbscan)
library(isotree)
library(pROC)

# 2. Una variable: RIC y puntaje z ----------------------------------------
valores <- c(10, 11, 12, 12, 13, 14, 15, 30)
cuartiles <- quantile(valores, c(0.25, 0.75))
ric <- diff(cuartiles)
limite_inferior <- cuartiles[1] - 1.5 * ric
limite_superior <- cuartiles[2] + 1.5 * ric
alerta_ric <- valores < limite_inferior | valores > limite_superior
puntaje_z <- (valores - mean(valores)) / sd(valores)
print(data.frame(Valor = valores, Alerta_RIC = alerta_ric,
                 Z = round(puntaje_z, 3), Alerta_Z = abs(puntaje_z) > 3))
boxplot(valores, horizontal = TRUE, main = "Valores para revisar",
        xlab = "Valor", col = "lightblue")

# 3. Dos variables: preparar los mismos puntos para los cuatro metodos -----
puntos <- read.csv("puntos.csv")
x <- as.matrix(puntos[, c("x1", "x2")])
z <- scale(x)
plot(z, pch = 19, col = "steelblue", xlab = "x1 estandarizada",
     ylab = "x2 estandarizada", main = "Datos simulados")

# 4. Mahalanobis: la funcion de R entrega la distancia al cuadrado ---------
distancia_m <- mahalanobis(z, center = colMeans(z), cov = cov(z))
umbral_m <- qchisq(0.99, df = ncol(z))
alerta_m <- distancia_m > umbral_m
cat("Umbral de Mahalanobis al cuadrado:", umbral_m, "\n")
print(puntos[alerta_m, ])
# El corte chi-cuadrado es una referencia aproximada bajo normalidad.
plot(z, col = ifelse(alerta_m, "firebrick", "steelblue"), pch = 19,
     xlab = "x1 estandarizada", ylab = "x2 estandarizada",
     main = "Mahalanobis: puntos marcados en rojo")

# 5. kNN: distancia al decimo vecino, excluyendo el propio punto ------------
k <- 10
puntaje_knn <- kNNdist(z, k = k)
umbral_knn <- quantile(puntaje_knn, 0.95)
alerta_knn <- puntaje_knn > umbral_knn
print(puntos[alerta_knn, ])
plot(sort(puntaje_knn), pch = 19, col = "steelblue",
     xlab = "Observaciones ordenadas", ylab = "Distancia al vecino 10")
abline(h = umbral_knn, col = "firebrick", lty = 2)

# 6. LOF: comparar la densidad de cada punto con la de sus vecinos ----------
# minPts incluye al propio punto: 11 significa 10 vecinos externos.
puntaje_lof <- lof(z, minPts = k + 1)
umbral_lof <- 1.5
alerta_lof <- puntaje_lof > umbral_lof
print(puntos[alerta_lof, ])
plot(z, col = ifelse(alerta_lof, "firebrick", "steelblue"), pch = 19,
     xlab = "x1 estandarizada", ylab = "x2 estandarizada",
     main = "LOF: corte ilustrativo en 1.5")
# 1.5 es un criterio docente, no un limite universal.

# 7. DBSCAN: ruido no significa automaticamente error ----------------------
epsilon <- 0.45
min_puntos <- 6
kNNdistplot(z, k = min_puntos - 1)
abline(h = epsilon, col = "firebrick", lty = 2)
modelo_db <- dbscan(z, eps = epsilon, minPts = min_puntos)
alerta_db <- modelo_db$cluster == 0
print(table(modelo_db$cluster))
plot(z, col = ifelse(alerta_db, "firebrick", "steelblue"), pch = 19,
     xlab = "x1 estandarizada", ylab = "x2 estandarizada",
     main = "DBSCAN: ruido en rojo")

revision <- data.frame(ID = puntos$id, Mahalanobis = alerta_m,
                       kNN = alerta_knn, LOF = alerta_lof, DBSCAN = alerta_db)
print(colSums(revision[, -1]))
print(tail(revision, 6))

# 8. PCA con canciones: reconstruir usando las primeras cinco componentes --
canciones <- read.csv("canciones.csv")
x_canciones <- as.matrix(canciones[, -1])
z_canciones <- scale(x_canciones)
pca_canciones <- prcomp(z_canciones, center = FALSE, scale. = FALSE)
varianza <- pca_canciones$sdev^2 / sum(pca_canciones$sdev^2)
print(data.frame(Componente = 1:7, Varianza = varianza,
                 Acumulada = cumsum(varianza)))
r <- 5
proyeccion <- pca_canciones$x[, 1:r]
reconstruccion <- proyeccion %*% t(pca_canciones$rotation[, 1:r])
# Q suma residuos al cuadrado en escala estandarizada, no en unidades mixtas.
q_canciones <- rowSums((z_canciones - reconstruccion)^2)
umbral_canciones <- quantile(q_canciones, 0.95)
alerta_canciones <- q_canciones > umbral_canciones
orden <- order(q_canciones, decreasing = TRUE)
print(head(data.frame(ID = canciones$id[orden], Q = q_canciones[orden],
                      Alerta = alerta_canciones[orden]), 10))
plot(q_canciones, pch = 19, col = ifelse(alerta_canciones, "firebrick", "steelblue"),
     xlab = "Cancion", ylab = "Q: suma de residuos al cuadrado")
abline(h = umbral_canciones, col = "firebrick", lty = 2)
# Esta revision es exploratoria: no hay etiquetas para calcular acuracia.

# 9. Cardio: entrenar y evaluar en conjuntos distintos ---------------------
cardio <- read.csv("cardio.csv")
entrenamiento <- subset(cardio, conjunto == "entrenamiento")
prueba <- subset(cardio, conjunto == "prueba")
variables <- paste0("v", 1:21)
x_train <- as.matrix(entrenamiento[, variables])
x_test <- as.matrix(prueba[, variables])
cat("Entrenamiento:", nrow(x_train), "Prueba:", nrow(x_test), "\n")
# id, conjunto y anomalia no son predictores. No usamos etiquetas para ajustar.
centro <- colMeans(x_train)
escala <- apply(x_train, 2, sd)
z_train <- scale(x_train, center = centro, scale = escala)
z_test <- scale(x_test, center = centro, scale = escala)

# 10. PCA: elegir componentes y corte solamente con entrenamiento ----------
pca <- prcomp(z_train, center = FALSE, scale. = FALSE)
acumulada <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
r_cardio <- which(acumulada >= 0.80)[1]
cargas <- pca$rotation[, 1:r_cardio]
recon_train <- (z_train %*% cargas) %*% t(cargas)
recon_test <- (z_test %*% cargas) %*% t(cargas)
q_train <- rowSums((z_train - recon_train)^2)
q_test <- rowSums((z_test - recon_test)^2)
umbral_pca <- quantile(q_train, 0.95)
clase_pca <- as.integer(q_test > umbral_pca)
cat("Componentes de cardio:", r_cardio, "Umbral Q:", umbral_pca, "\n")

# 11. Isolation Forest clasico: cortes sobre una variable por nodo ---------
modelo_if <- isolation.forest(z_train, ndim = 1, ntrees = 300,
                              sample_size = 256, seed = 2026, nthreads = 1)
score_train <- predict(modelo_if, z_train, type = "score")
score_test <- predict(modelo_if, z_test, type = "score")
# En isotree un puntaje mayor significa mayor evidencia de anomalia.
umbral_if <- quantile(score_train, 0.95)
clase_if <- as.integer(score_test > umbral_if)
cat("Umbral de Isolation Forest:", umbral_if, "\n")

# 12. Evaluacion: 1 significa anomalia, 0 significa normal ------------------
y_test <- prueba$anomalia
matriz_pca <- table(Real = factor(y_test, levels = 0:1),
                    Predicha = factor(clase_pca, levels = 0:1))
matriz_if <- table(Real = factor(y_test, levels = 0:1),
                   Predicha = factor(clase_if, levels = 0:1))
print(matriz_pca)
print(matriz_if)
roc_pca <- roc(y_test, q_test, levels = c(0, 1), direction = "<", quiet = TRUE)
roc_if <- roc(y_test, score_test, levels = c(0, 1), direction = "<", quiet = TRUE)
comparacion <- data.frame(
  Modelo = c("PCA", "Isolation Forest"),
  Acuracia = c(mean(clase_pca == y_test), mean(clase_if == y_test)),
  Sensibilidad = c(mean(clase_pca[y_test == 1] == 1),
                   mean(clase_if[y_test == 1] == 1)),
  Precision = c(matriz_pca[2, 2] / sum(matriz_pca[, 2]),
                matriz_if[2, 2] / sum(matriz_if[, 2])),
  AUC = c(as.numeric(auc(roc_pca)), as.numeric(auc(roc_if))))
print(comparacion, digits = 4, row.names = FALSE)
cat("Acuracia si marcamos todo normal:", mean(y_test == 0), "\n")
plot(roc_pca, col = "steelblue", legacy.axes = TRUE,
     main = "Cardio: curvas ROC en prueba")
plot(roc_if, col = "firebrick", add = TRUE)
legend("bottomright", c("PCA", "Isolation Forest"),
       col = c("steelblue", "firebrick"), lwd = 2, bty = "n")

# 13. Un corte mas alto genera menos alertas; el ranking no cambia ----------
umbrales <- quantile(score_train, c(0.90, 0.95, 0.99))
sensibilidad <- numeric(3)
alertas <- numeric(3)
for (j in 1:3) {
  clase <- as.integer(score_test > umbrales[j])
  alertas[j] <- sum(clase)
  sensibilidad[j] <- mean(clase[y_test == 1] == 1)
}
print(data.frame(Cuantil = c(0.90, 0.95, 0.99), Umbral = as.numeric(umbrales),
                 Alertas = alertas, Sensibilidad = sensibilidad))
# No elegimos el corte con prueba; esta tabla solo ilustra su efecto.
# Los puntajes no son probabilidades: no calculamos log loss ni Brier con ellos.
