# Roadmap de estudio — Diplomado Data Science

> Versión 1 · 2026-10-05 · Se ajusta cada vez que se agrega material nuevo.

## Punto de partida

| Dimensión | Estado |
|---|---|
| Avance del diplomado | Clase 40 (Naive Bayes) |
| Punto de inicio | Clase 17, en orden |
| Tiempo | 8 h/semana · lun, mar, mié, vie, dom (~95 min/sesión) |
| Python | Intermedio (pandas OK) · scikit-learn por reforzar |
| Estadística | Base débil → nivelación previa obligatoria |
| Álgebra lineal | Se asume básica → refuerzo puntual en Clase 32 (PCA) |
| Fuente principal | Transcripciones de clase (no videos) |
| Entrega próxima | Proyecto N°2 (~2026-11-05) · se integrará cuando haya enunciado |

## Decisión de lenguaje: R vs Python

Los notebooks de las **Clases 17–27 están en R**; desde la **Clase 30 son Python**.

- **Decisión:** entender los conceptos con la clase y practicar en **Python con `statsmodels`**.
  Su API de fórmulas (`smf.ols("y ~ x1 + x2", data=df)`) es casi idéntica a `lm(y ~ x1 + x2)` de R.
- El código R de la clase se **lee** para seguir el razonamiento, no se reescribe desde cero.
- **Por qué:** el bloque de ML y el proyecto son Python; aprender R a fondo ahora costaría horas que no tenemos.

## Estructura semanal

| Día | Uso |
|---|---|
| Lun, Mar, Mié, Vie | Contenido nuevo |
| Dom | Repaso espaciado + mini evaluación (5–8 preguntas) |

**Estructura de cada sesión (~95 min):**
1. 10 min — repaso rápido de la sesión anterior (3 preguntas sin mirar apuntes)
2. 25 min — concepto nuevo (transcripción + resumen)
3. 45 min — práctica guiada en notebook
4. 15 min — ejercicio de cierre sin ayuda + notas propias

## Niveles de profundidad

- **A — A fondo:** base de lo que viene después. Teoría + práctica + criterio de dominio estricto.
- **B — Práctico:** saber aplicarlo e interpretarlo; la derivación teórica es opcional.

## Etapas

### Etapa 0 — Nivelación estadística · Semanas 1–2 (6–18 oct) · Nivel A

Sin carpeta propia: se construye con material generado y datasets existentes (`salarios.xlsx`, `ENS.csv`).

| Sesión | Tema |
|---|---|
| 0.1 | Estadística descriptiva y tipos de variables |
| 0.2 | Distribuciones de probabilidad · la normal y el puntaje z |
| 0.3 | Muestra vs población · Teorema Central del Límite · error estándar |
| 0.4 | Intervalos de confianza |
| 0.5 | Lógica de la prueba de hipótesis · H0/H1 |
| 0.6 | p-valor · errores tipo I y II · potencia |
| 0.7 | Pruebas t (una y dos muestras) · proporciones · chi-cuadrado |

> Correlación se movió a la Clase 18, que la enseña completa. Guía: `Guias/01_Etapa0_Estadistica.pdf`.

**Criterio de dominio:** explicar sin apuntes qué es un p-valor y un IC 95%, y en un dataset nuevo
plantear H0/H1, elegir la prueba correcta, ejecutarla en Python e interpretar el resultado en una frase de negocio.

### Etapa 1 — Análisis de regresión (Clases 17–19) · Semanas 3–5 (19 oct–8 nov) · Nivel A

ANOVA · correlación · regresión lineal simple · regresión lineal múltiple · supuestos y diagnóstico.

**Criterio de dominio:** ajustar una regresión múltiple, interpretar coeficientes, p-valores y R²,
y verificar supuestos (residuos, normalidad, homocedasticidad, multicolinealidad).

### Etapa 2 — Modelos lineales generalizados (Clases 20–21) · Semanas 6–7 (9–22 nov)

Familia exponencial y función de enlace (A) · regresión logística (A) · Poisson (B) · multinomial (B).

**Criterio de dominio:** elegir el GLM según el tipo de variable respuesta e interpretar un odds ratio.

### Etapa 3 — A/B testing y causalidad (Clase 22) · Semana 8 (23–29 nov) · Nivel A

Diseño de experimentos · tamaño muestral · SRM · diferencias en diferencias.

**Criterio de dominio:** analizar un A/B test de punta a punta y detectar un SRM.

### Etapa 4 — Estadística bayesiana (Clases 23–24) · Semanas 9–10 (30 nov–13 dic) · Nivel A en teorema de Bayes

Teorema de Bayes · prior/likelihood/posterior · comparación frecuentista vs bayesiano.
**Base directa de Naive Bayes (Clase 40).**

**Criterio de dominio:** resolver a mano un problema de Bayes y explicar cómo cambia el posterior con más datos.

### Etapa 5 — Series de tiempo (Clases 25–27) · Semanas 11–12 (14–27 dic) · Nivel B

Componentes · estacionariedad · autocorrelación · modelos ARIMA · pronóstico y validación temporal.

**Criterio de dominio:** descomponer una serie, ajustar un modelo y evaluar el pronóstico con partición temporal.

### Etapa 6 — Fundamentos de ML (Clases 30–32) · Semanas 13–15 (28 dic–17 ene) · Nivel A

Incluye **nivelación de scikit-learn**: `fit/predict/transform`, `Pipeline`, `ColumnTransformer`, validación cruzada.
Preparación de datos · train/test · métricas · sesgo-varianza · PCA (con refuerzo de álgebra lineal).

**Criterio de dominio:** construir un `Pipeline` completo con validación cruzada y justificar la métrica elegida.

### Etapa 7 — Modelos (Clases 33–35) · Semanas 16–18 (18 ene–7 feb) · Nivel A

Selección de variables · árboles de decisión · bagging · Random Forest.

**Criterio de dominio:** entrenar y ajustar un árbol y un Random Forest, comparar con validación cruzada
y explicar la importancia de variables.

### Etapa 8 — Clases 28–29 y 36–40+ · Por definir

Se planifica cuando exista el material.

## Repaso espaciado

| Cuándo | Qué |
|---|---|
| Cada domingo | Repaso de la semana + mini evaluación |
| +1 semana tras cerrar una etapa | Repaso de esa etapa (20 min dentro del domingo) |
| +3 semanas tras cerrar una etapa | Segundo repaso con ejercicio nuevo |
| Fin de cada etapa | Mini proyecto integrador con un dataset de la carpeta |

## Material pendiente

| Necesario antes de | Falta |
|---|---|
| Semana 9 | Transcripción Clase 24 |
| Semana 11 | Transcripciones Clases 25–27 · el PDF "Clase 3" de series de tiempo no está en la carpeta |
| Semana 13 | Transcripciones Clases 30, 31, 32 |
| Semana 16 | Transcripción Clase 33 (hay `.mp4` en `VideosClases`) |
| Etapa 8 | Carpetas Clases 28, 29, 36–40 |

## Registro de avance

| Fecha | Sesión | Estado | Notas |
|---|---|---|---|
| | | | |
