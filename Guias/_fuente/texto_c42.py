"""Interpretaciones de la práctica espejo (Clase 42). Los números provienen de la ejecución con semilla 2026
(se verifican al ejecutar el notebook; con otras versiones de paquetes pueden variar en el último decimal)."""
T = {}

T["p1a"] = """
**Interpretación.**

- **Tarea**: clasificación binaria supervisada (anticipar si un cliente se fuga). **Unidad de observación**: un cliente (una fila). **Respuesta**: `fuga` (1 = se fue; es la clase positiva). La tasa es 15.7 % tanto en entrenamiento (2 520) como en prueba (630): la estratificación las deja iguales, y los cinco pliegues quedan con 504 filas y ≈ 79 fugas cada uno.
- **La variable que no se puede usar.** En el taller, `duracion` se mide **al terminar** la llamada registrada, que es justamente el evento que se quiere anticipar: no existe cuando hay que decidir a quién llamar, y además está muy asociada a la respuesta (una llamada larga suele ser una oferta aceptada). Usarla produce una métrica de validación espectacular y un modelo inservible en producción. La base de churn **no tiene** una columna idéntica, así que para ejercitar el razonamiento adoptamos un **supuesto explícito**: `clv` (valor de vida del cliente) se calcula al cierre del periodo e incorpora la permanencia realmente observada, por lo que parte de su valor es **consecuencia** de la fuga. La tabla es coherente con la sospecha: media 541 en quienes se quedan y 124 en quienes se fueron (más de 4 veces), una brecha mucho mayor que la de `antiguedad` (32.5 vs 31.6).
- **Honestidad sobre el supuesto**: en las Clases 33 y 34 `clv` se usó como predictor legítimo porque la base no documenta cuándo se calcula. Aquí la tratamos como filtrada **para practicar la regla**: «¿esta columna existiría, con este valor, en el momento en que debo predecir?». Si la respuesta es no (o no lo sé), se excluye.
- Se excluyen de los predictores `clv`, `id_cliente` (identificador: memorizarlo no generaliza), `pliegue` (partición) y `fuga` (la respuesta). Quedan 13 predictores originales.
"""

T["p1b"] = """
**Interpretación.**

- **Tipos**: todas las columnas son numéricas en el archivo, pero `tipo_plan` y `estado` son **códigos de categorías** (1/2), `reclamos` es binaria (0/1) y `monto` toma valores enteros 0–10; se tratan como categóricas/binaria según corresponda (ver 1(c)).
- **Valores imposibles**: hay negativos en `antiguedad` (9), `segundos` (17), `frec_llamadas` (8) y `edad` (1), y 85 edades fuera de 15–100. Una antigüedad, un tiempo de llamada o una frecuencia **no pueden ser negativos**: son errores de captura y se convierten en `NaN`. Contraste con el taller: un **`saldo` negativo sí es posible** (sobregiro, deuda) y es información, no error; y `dias_previos = -1` **no es un número de días** sino un **código** («sin contacto previo») que no debe promediarse ni escalarse como cantidad.
- **Faltantes**: tras la limpieza hay 166 filas (6.6 %) con algún faltante (sobre todo `edad`: 85 y `segundos`: 51). No se eliminan filas: se imputa con la **mediana del conjunto de ajuste de cada pliegue** (dentro del `Pipeline`). En el banco no hay `NaN`, pero sí la categoría `unknown`, que **se mantiene como categoría**.
- **Duplicados**: 10 filas repetidas en todas las columnas (sin `id_cliente`). Con variables mayormente discretas pueden ser clientes distintos con el mismo patrón; no hay un identificador natural para decidir, así que se informan y se **conservan**.
- **Balance**: 84.3 % no se fuga / 15.7 % se fuga. Predecir siempre «no se fuga» acierta el **84.3 %**, pero detecta **0 de 396** fugas (sensibilidad 0): una exactitud alta que no sirve para el negocio. Por eso se usan log loss, Brier, AUC, sensibilidad/especificidad y el costo, y se incluye siempre el modelo de referencia (que además fija un piso de log loss: 0.435).
"""

T["p2a"] = """
**Interpretación.**

| Modelo | log loss OOF | Brier OOF | AUC OOF |
|---|---|---|---|
| Logística, 16 columnas | 0.2461 | 0.0771 | 0.9168 |
| Logística, 10 mejores por F | 0.2555 | 0.0792 | 0.9079 |

- **Columnas seleccionadas**: en los cinco pliegues se eligen **las mismas 10** (`segundos`, `frec_llamadas`, `frec_sms`, `n_contactos`, `log_frec_sms`, `reclamos`, `estado_est2` y tres dummies de `nivel_monto`). Nunca se eligen `antiguedad`, `edad`, `nivel_monto_m2`, `numero_fallas`, `sin_sms` ni `tipo_plan_plan2`. La estabilidad no es una regla general: aquí cada ajuste usa ≈ 2 016 filas y las diferencias de F entre columnas fuertes y débiles son grandes; con menos filas o columnas de F parecido la lista cambiaría de un pliegue a otro, y eso también se informa.
- **Comparación**: la versión con 10 columnas es **algo peor en las tres métricas** (log loss +0.009, AUC −0.009). Con 16 columnas y ≈ 2 000 filas de ajuste la logística casi no sobreajusta, así que quitar columnas no ayuda a generalizar y se pierde algo de señal (una columna con F bajo **por sí sola** puede aportar al combinarse con otras).
- **Diferencia con PCA**: F de ANOVA es un **filtro supervisado**: puntúa cada columna original según cuánto separan sus medias las dos clases (usa la respuesta) y **conserva columnas interpretables**. PCA es **no supervisado**: **crea** combinaciones lineales nuevas que maximizan la varianza, sin mirar la respuesta, y las columnas originales dejan de verse.
- **Limitación del filtro univariado**: puntúa **una columna a la vez**, ignorando redundancia e interacciones. Entre las 10 elegidas, `segundos` y `frec_llamadas` tienen |r| = 0.93 y `frec_sms` con `log_frec_sms` 0.77: el filtro se queda con varias versiones de la misma información y puede dejar fuera columnas que solo ayudan en conjunto. Además, las dummies de una misma variable (`nivel_monto`) se evalúan por separado.
"""

T["p2b"] = """
**Interpretación.**

- **Selección**: se define de antemano la regla «menor log loss OOF entre los modelos probabilísticos (sin contar la referencia)». Gana **Boosting** (0.1789; Brier 0.0545; AUC 0.9618), seguido de Random Forest (0.2016). La SVM (solo AUC: 0.9356) queda tercera en orden, pero **no compite en probabilidades** porque su puntaje no lo es.
- **Boosting informado**: `GradientBoostingClassifier`, hasta 200 árboles de profundidad 3, tasa de aprendizaje 0.05, con detención temprana (10 % de las filas **de ajuste** como validación interna, de modo que tampoco hay fuga); usó 173, 200, 133, 200 y 200 árboles en los cinco pliegues.
- **Referencia**: log loss 0.4349, Brier 0.1324 y AUC ≈ 0.5 (0.4988 en el OOF conjunto porque la constante cambia levemente entre pliegues; 0.5000 como media por pliegue). Todo modelo útil debe ganarle por amplio margen.
- **Naive Bayes** tiene un AUC decente (0.890) pero un log loss de **1.46, más de tres veces el de la referencia**: ordena bien y asigna probabilidades pésimas (ver 2(c)).
- **Árboles**: profundidad 3 → log loss de entrenamiento 0.2635 y OOF 0.2951; profundidad 6 → entrenamiento 0.2013 pero OOF **0.4008**. El AUC sí mejora con la profundidad (0.856 → 0.903): ordenar y dar buenas probabilidades son cosas distintas.
- Un ganador distinto en el banco es perfectamente aceptable si la regla de selección está escrita y justificada.
"""

T["p2c"] = """
**Interpretación** (con las evidencias de las celdas anteriores).

1. **Profundidad y sobreajuste de los árboles.** Al pasar de profundidad 3 (≈ 7 hojas) a 6 (≈ 23 hojas) el log loss de **entrenamiento baja** (0.2635 → 0.2013) pero el **OOF sube** (0.2951 → 0.4008): la brecha crece de 0.03 a 0.20. Hojas con pocas observaciones (mínimo 20) producen probabilidades casi 0 o 1 que fallan fuera de la muestra, y el log loss castiga mucho esos errores confiados. La brecha entrenamiento–validación es la señal de sobreajuste.
2. **Bagging, Random Forest y boosting.** Un árbol profundo tiene **varianza alta**. *Bagging* promedia árboles entrenados sobre remuestras bootstrap y baja la varianza; *Random Forest* añade un sorteo de columnas en cada división (aquí √16 = 4) para **decorrelacionar** los árboles, de modo que el promedio mejore más. *Boosting* construye árboles pequeños **en secuencia**, cada uno corrigiendo los errores del anterior, y reduce el **sesgo**; la tasa de aprendizaje (0.05) y la detención temprana controlan el sobreajuste. Resultado: RF (0.2016) y boosting (0.1789) superan por mucho al mejor árbol individual (0.2951), y boosting gana en las tres métricas.
3. **Naive Bayes.** Supone **independencia condicional**: dada la respuesta, las columnas serían independientes. Aquí es falsa (|r| = 0.93 entre `segundos` y `frec_llamadas`; `log_frec_sms` es función de `frec_sms`; las dummies de `nivel_monto` son mutuamente excluyentes), así que la misma información se multiplica varias veces y las probabilidades se vuelven **extremas**: 51 % de las predicciones OOF es < 0.01 y 26 % es > 0.99. En la calibración, donde predice ≈ 0.97 (982 clientes) la tasa observada es solo 36 %. La variante **gaussiana** además modela columnas **binarias** (`reclamos`, `sin_sms`, dummies) como normales con media y varianza propias de cada clase: una columna 0/1 no es normal y su varianza pequeña genera razones de verosimilitud desproporcionadas. Orden aceptable (AUC 0.890), probabilidades inservibles (log loss 1.46).
4. **SVM: C, gamma y escalamiento.** Con la configuración del enunciado (C = 1, gamma = 1/p = 0.0625, variables escaladas) el AUC OOF es 0.9356. **C** controla el castigo al error frente al margen: con C = 0.1 baja a 0.9121 (modelo demasiado rígido) y con C = 10 sube a 0.9535 (más flexible; no se usa porque el enunciado fija C = 1, pero muestra que el valor importa). **Gamma** fija el alcance del kernel: un gamma 10 veces menor da 0.9092 (frontera casi lineal) y 10 veces mayor 0.9327 (más local). El **escalamiento** es decisivo: sin escalar el AUC cae a **0.623**, porque `segundos` y las frecuencias (miles) dominan las distancias del kernel. Además, el puntaje de decisión no está acotado ni calibrado: solo sirve para ordenar.
5. **¿Mejor ordenamiento implica mejores probabilidades? No.** El AUC depende **solo del orden**: al elevar las probabilidades del modelo elegido al cubo, el AUC se mantiene en 0.9618, pero el log loss pasa de 0.1789 a 0.3278 (y con la raíz, a 0.2699). En la tabla real: el árbol de profundidad 6 tiene mejor AUC que el de profundidad 3 (0.903 vs 0.856) y peor log loss (0.401 vs 0.295); Naive Bayes ordena bien y calibra pésimo. Un buen ordenamiento es **necesario pero no suficiente** cuando la decisión usa las probabilidades (por ejemplo, el umbral por costo de la Parte 3).
"""

T["p3a"] = """
**Interpretación.**

- **Umbral elegido: t\\* = 0.15**, con costo mínimo C = 391 (FP = 236, FN = 31). Con t = 0.50 el costo sería **755** (FP = 70, FN = 137). Como referencias, «no alertar a nadie» cuesta 5 × 396 = 1 980 y «alertar a todos» 2 124.
- **Lectura de los dos errores.** Al subir el umbral hay **menos FP** (menos contactos de retención a clientes que no se iban: de 571 en 0.05 a 0 en 0.95) pero **más FN** (clientes que se fueron sin que se les ofreciera nada: de 7 a 333). Como un FN cuesta 5 veces un FP, el mínimo cae en un umbral **bajo**: se acepta alertar a bastantes clientes de más para perder pocos. Con t\\* = 0.15 se detecta el **92 %** de las fugas (sensibilidad 0.922) a cambio de 236 alertas innecesarias (especificidad 0.889); con 0.50 solo se detecta el 65 %.
- **Coherencia teórica.** Si las probabilidades estuvieran bien calibradas, el umbral óptimo sería 1/(1 + 5) = 0.167: el 0.15 observado es consistente. La curva es **plana cerca del mínimo** (0.10: 416, 0.20: 409), por lo que 0.15 no es más «verdadero» que sus vecinos; conviene elegirlo con un criterio explícito.
- **Cuidado**: el umbral se eligió con los mismos OOF con que se mide su costo (optimismo leve); la confirmación honesta es la prueba de 3(b). Y los costos 1 y 5 son hipotéticos: con costos reales cambiaría el umbral.
"""

T["p3b"] = """
**Interpretación.**

| Prueba (630 clientes) | log loss | Brier | AUC |
|---|---|---|---|
| Boosting | 0.2073 | 0.0637 | 0.9456 |
| Referencia (tasa 0.157) | 0.4349 | 0.1324 | 0.5000 |

| Regla | VN | FP | FN | VP | Sensibilidad | Especificidad | C | C/n |
|---|---|---|---|---|---|---|---|---|
| Boosting, t\\* = 0.15 | 479 | 52 | 17 | 82 | 0.828 | 0.902 | 137 | 0.218 |
| Boosting, t = 0.50 | 516 | 15 | 45 | 54 | 0.545 | 0.972 | 240 | 0.381 |
| Referencia, t = 0.15 (alerta a todos) | 0 | 531 | 0 | 99 | 1.000 | 0.000 | 531 | 0.843 |
| Referencia, t = 0.50 (no alerta a nadie) | 531 | 0 | 99 | 0 | 0.000 | 1.000 | 495 | 0.786 |

- **Qué se concluye.** El modelo conserva su calidad en datos no vistos: AUC 0.946 (0.962 en OOF) y log loss 0.207 (0.179 en OOF), una degradación pequeña y esperable (con solo 99 fugas en la prueba el error de estimación es apreciable). Supera a la referencia por mucho (log loss 0.207 vs 0.435).
- **Qué recomendar.** Usar el boosting con **t\\* = 0.15**: reduce el costo medio a 0.218 por cliente, **43 % menos** que con 0.50 (0.381) y **72 % menos** que la mejor regla trivial (0.786, «no alertar a nadie»). A cambio, hay 52 contactos de retención a clientes que no se iban (10 % de quienes se quedan) y se pierden 17 de las 99 fugas. Hay que comunicar esa tasa de falsas alarmas y verificar que el costo unitario de contactar sea efectivamente 1/5 del de perder un cliente.
- **Qué falta verificar** antes de una campaña futura: (i) la incertidumbre de las métricas (intervalos por *bootstrap*; la prueba es una sola muestra pequeña); (ii) los costos reales de FP y FN; (iii) la calibración en la prueba y la estabilidad en el tiempo (deriva de datos); (iv) que **todas las variables estén disponibles antes de decidir** (`clv` se excluyó por el supuesto; `n_contactos` y similares deben medirse hasta el momento de la decisión); (v) que contactar a un cliente de riesgo **efectivamente reduzca** la fuga (predecir no es causar; haría falta un experimento A/B); (vi) la equidad entre grupos de clientes y el monitoreo continuo.
"""

T["p4a"] = """
**Interpretación.**

- **Varianza explicada**: PC1 = 56.3 %, PC2 = 25.2 %, **suma = 81.5 %**; PC3 = 12.7 % y PC4 = 5.9 % quedan fuera del plano (18.5 %).
- **Cargas**: PC1 tiene signos positivos en `numero_fallas` (0.52), `frec_llamadas` (0.61) y `n_contactos` (0.59) y casi cero en `frec_sms` (0.09): es un componente de **«nivel general de actividad»** (clientes con muchas fallas, llamadas y contactos frente a los de poca actividad). PC2 es prácticamente `frec_sms` (0.98): el eje del **uso de SMS**, que casi no se relaciona con las otras tres (correlaciones de alrededor de 0.1 o menos).
- **Qué resume y qué omite el plano**: resume el nivel conjunto de actividad y el uso de SMS, 81 % de la variación de las cuatro variables estandarizadas. **Omite** el 18.5 % restante, principalmente el contraste entre `numero_fallas` y `n_contactos` (PC3: 0.82 vs −0.45), y por construcción **no usa `fuga`**: dos clientes cercanos en el plano pueden diferir en lo omitido, y la dirección de máxima varianza no es necesariamente la que mejor separa la respuesta. Los signos de los componentes son arbitrarios.
"""

T["p4b"] = """
**Interpretación.**

| k | WSS | silueta media |
|---|---|---|
| 2 | 6 436 | 0.371 |
| 3 | 4 628 | **0.400** |
| 4 | 3 716 | 0.334 |

- **Elección**: el WSS **siempre baja** al aumentar k (no sirve solo para elegir); la silueta es máxima en **k = 3** (0.400), que es la regla del enunciado.
- **Grupos (k = 3)**: **grupo 0** (437; 17.3 %): usuarios intensivos de SMS (`frec_sms` ≈ 287, z = +1.9) con actividad promedio en lo demás; fuga **0.9 %**. **Grupo 1** (1 382; 54.8 %): baja actividad en todo (z entre −0.4 y −0.6: ≈ 39 llamadas, 16 SMS, 14 contactos); fuga **24.7 %**. **Grupo 2** (701; 27.8 %): mucha actividad en fallas, llamadas y contactos (z ≈ +1.0 a +1.1); fuga **7.3 %**. Tasa global: 15.7 %.
- **Por qué los grupos no son clases predichas**: se formaron **sin mirar `fuga`** y con solo cuatro variables; no hay umbral ni modelo evaluado. En el grupo 1, el de mayor fuga, el **75 % de los clientes no se fuga**: rotularlo «se fuga» daría un 75 % de falsas alarmas (aunque contiene unas 341 de las 396 fugas, 86 %, en el 55 % de los clientes). Sirve para **describir perfiles**, no para decidir: los grupos dependen de k y de la escala, y la tasa de fuga por grupo es un resultado descriptivo, no una predicción.
"""

T["p5a"] = """
**Interpretación.**

- **Corte**: χ²(4; 0.99) = **13.277**. **153 de 2 520** registros (**6.07 %**) lo superan.
- **Dos mayores distancias**: `id_cliente` 490 con D² = 34.50 (`numero_fallas` 8, `frec_llamadas` 49, `frec_sms` 51, `n_contactos` 83) y `id_cliente` 1240 con D² = 27.73 (`numero_fallas` 0, `frec_llamadas` 35, `frec_sms` 33, `n_contactos` 70). Se informan con valores **originales**, junto con sus puntajes z para poder explicarlos.
- Comprobación: usar los datos estandarizados da exactamente los mismos D² (Mahalanobis es invariante a cambios de escala lineales).
"""

T["p5b"] = """
**Interpretación.**

- **¿Por qué son inusuales?** En ambos, `n_contactos` es muy alto (83 y 70; z = 3.47 y 2.71; percentiles 99.5 y 97.4), pero `frec_llamadas` es **menor que el promedio** (49 y 35 vs 71; z = −0.39 y −0.63), aunque ambas variables están correlacionadas (r = 0.76): lo esperable es que muchos contactos vengan con muchas llamadas. Su rareza es una **combinación que contradice la correlación**, más que un valor extremo en una sola variable: ningún z univariado pasa de 3.5. Eso es lo que mide Mahalanobis y lo que no ve una regla variable por variable.
- **¿Demuestra errores?** No. Un D² alto señala un registro **estadísticamente inusual**, no incorrecto: puede ser un cliente real contactado por otros canales, o un error de captura; la distancia no distingue entre ambos.
- **¿Garantiza marcar el 1 %?** No: marca el **6.07 %**. El 1 % solo se cumple si los datos son normales multivariados y los parámetros se conocen; aquí las variables son **asimétricas** (asimetrías 1.0 a 1.9) y de cola pesada, y la media y la covarianza se estiman con los mismos datos. Con cuantiles más exigentes también se pasa de lo esperado: 0.95 → 10.9 % (esperado 5 %); 0.999 → 1.9 % (esperado 0.1 %). Los dos registros extremos no distorsionan la covarianza (sin ellos, D² = 35.4 y 28.4; marcados 155).
- **Antes de eliminar un registro**: verificar unidades y captura contra la fuente; revisar el resto de sus columnas y el contexto de negocio; buscar duplicados; comparar resultados con y sin el registro; y dejar por escrito la decisión (conservar, corregir o eliminar). Un registro inusual puede ser justamente el caso de interés.
"""
