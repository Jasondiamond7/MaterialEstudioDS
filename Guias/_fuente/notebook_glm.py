"""Genera el notebook guiado de GLM (Clases 20-21)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "C20-21_GLM.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    body = f"{note}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n"
       f"{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- header
md("""
# Clases 20-21 · Modelos Lineales Generalizados (GLM)
**Diplomado Data Science** · 2 sesiones de ~95 min · guía: `Guias/03_GLM_C20-21.pdf`

Material de apoyo:
- `Guias/apuntes/03_Apuntes_GLM_C20-21.pdf` — apunte con pistas conceptuales.
- `Guias/apuntes/03_Ejercicios_Cuaderno_GLM.pdf` — ejercicios en papel (antes o después de este notebook).

⚠️ **Antes de empezar**: este notebook necesita `statsmodels` y `scikit-learn`, que tu entorno todavía no tiene. Instala en tu venv:
```
pip install statsmodels scikit-learn
```

### Objetivos de estas sesiones
1. Ajustar e interpretar una **regresión logística** (odds, odds ratio, probabilidad predicha).
2. Evaluar un clasificador binario más allá del accuracy: matriz de confusión, sensibilidad, especificidad, ROC y AUC.
3. Ajustar una **regresión de Poisson con offset** para tasas, y detectar sobre-dispersión.
4. Ajustar un **modelo multinomial** y usarlo para predecir y para calcular probabilidades por clase.

### Plan de las sesiones
| Sesión | Min | Bloque |
|---|---|---|
| 1 | 5 | Preparación y carga de datos |
| 1 | 5 | A1 · Prevalencia |
| 1 | 10 | A2 · Ajustar `chd01 ~ cigs` |
| 1 | 10 | A3 · Odds ratio |
| 1 | 10 | A4 · Probabilidad predicha |
| 1 | 10 | A5 · Agregar `age` (test de razón de verosimilitud) |
| 1 | 5 | A6 · Logit vs. probit |
| 1 | 15 | B1 · Train/test y la trampa del accuracy |
| 1 | 10 | B2 · Sensibilidad, especificidad, precisión a mano |
| 1 | 15 | B3 · Curva ROC, AUC y umbral de Youden |
| 2 | 10 | C1 · Tasas por 1000 habitantes |
| 2 | 10 | C2 · Poisson con offset |
| 2 | 15 | C3 · Sobre-dispersión y quasi-Poisson |
| 2 | 10 | C4 · ¿Aporta la ciudad? |
| 2 | 5 | D1 · Construir PID3 |
| 2 | 10 | D2 · Modelo multinomial simple |
| 2 | 10 | D3 · Probabilidades predichas |
| 2 | 5 | D4 · Accuracy y la clase fantasma |
| 2 | 15 | Ejercicio de cierre **sin ayuda** |
| 2 | 5 | Autoevaluación y notas |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números que aparecen en el texto (coeficientes, p-valores, AIC) están verificados contra la salida real de este notebook. Si a ti te salen ligeramente distintos revisa primero la versión de `statsmodels`.
""")

# ---------------------------------------------------------------- setup
md("## Preparación")
code("""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Busca la carpeta del diplomado subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Guias" / "datos").exists())
datos = raiz / "Guias" / "datos"

wcgs = pd.read_csv(datos / "wcgs.csv")
wcgs["chd01"] = (wcgs["chd"] == "yes").astype(int)

cancer = pd.read_csv(datos / "cancer_pulmon.csv")
nes = pd.read_csv(datos / "nes96.csv")

print("wcgs:", wcgs.shape, "| cancer_pulmon:", cancer.shape, "| nes96:", nes.shape)
wcgs.head()
""")
md("""
**wcgs.csv** (n=3154): cohorte del *Western Collaborative Group Study*. Cada fila es un trabajador varón, con edad, presión arterial (`sdp`, `dbp`), colesterol (`chol`, 12 valores faltantes), cigarrillos diarios (`cigs`) y si tuvo un evento coronario en el seguimiento (`chd`: "yes"/"no"). Creamos `chd01` como 0/1 porque statsmodels necesita la respuesta numérica.

**cancer_pulmon.csv** (24 filas): casos de cáncer de pulmón y población en riesgo, cruzando 6 grupos de edad y 4 ciudades danesas. Es un dataset *agregado*: cada fila no es una persona, sino un grupo edad × ciudad con un conteo de casos y su población expuesta.

**nes96.csv** (n=944): encuesta electoral de EE.UU. 1996. Usaremos `PID` (afiliación partidaria detallada), `age`, `educ` e `income` para predecir la afiliación en 3 categorías.
""")

# ================================================================== PARTE A
md("""
## Parte A · Regresión logística (`wcgs`)

### A1 · Prevalencia (5 min)
Antes de modelar nada, mira qué tan frecuente es el evento. Va a importar mucho en la Parte B.
""")
code("""
prevalencia = wcgs["chd01"].mean()
print(f"Prevalencia de CHD: {prevalencia:.2%}  ({wcgs['chd01'].sum()} de {len(wcgs)})")
""")
md("""
Con **8.15%** de eventos, el dataset está desbalanceado: por cada persona con el evento hay ~11 sin él. Un modelo que **siempre** prediga "no" ya acierta el 92% de las veces sin aprender absolutamente nada. Guarda esta idea para la Parte B.
""")

md("""
### A2 · Ajustar `chd01 ~ cigs` (10 min)
En statsmodels, un GLM se ajusta así: `smf.glm(formula, datos, family=...).fit()`. Para una respuesta binaria, la familia es `sm.families.Binomial()` (link logit por defecto).
""")
code("""
modelo_cigs = smf.glm("chd01 ~ cigs", wcgs, family=sm.families.Binomial()).fit()
print(modelo_cigs.summary())
""")
md("""
Columnas de la tabla de coeficientes:
- **coef**: el coeficiente en la escala del *link* (log-odds para logit, no en probabilidad).
- **std err**: error estándar del coeficiente.
- **z, P>|z|**: estadístico de Wald y su p-valor (¿el coeficiente es distinguible de 0?).
- **[0.025, 0.975]**: intervalo de confianza del 95% para el coeficiente, **en log-odds**.

Más abajo en el resumen: **Deviance** (mide el desajuste del modelo: mientras menor, mejor, análoga a la suma de cuadrados residual) y **AIC** (para comparar modelos, pague una penalización por cada parámetro).

### ✍️ Ejercicio A2
Sin mirar el resumen de nuevo: ¿el coeficiente de `cigs` es significativo? ¿Cuál es la deviance nula, la deviance residual y el AIC del modelo?
""")
code("""
# tu código aquí
""")
solution("""
print(f"Intercept = {modelo_cigs.params['Intercept']:.4f}  (SE {modelo_cigs.bse['Intercept']:.4f})")
print(f"cigs      = {modelo_cigs.params['cigs']:.4f}  (SE {modelo_cigs.bse['cigs']:.4f})")
print(f"Deviance nula:     {modelo_cigs.null_deviance:.1f}")
print(f"Deviance residual: {modelo_cigs.deviance:.1f}")
print(f"AIC: {modelo_cigs.aic:.2f}")
""", "Intercept = -2.7422 (SE 0.0921), cigs = 0.0232 (SE 0.0040, z = 5.74, p < 0.001): **sí es significativo**. "
     "Deviance nula 1781.2, deviance residual 1750.0 (bajó solo 31.2 puntos con una variable: `cigs` solo explica una parte chica). AIC = 1753.97.")

md("""
### A3 · Odds ratio (10 min)
El coeficiente está en log-odds, que nadie interpreta directamente. Para pasar a **odds ratio (OR)** se exponencia: $OR = e^{\\beta}$.

- $OR = 1$: la variable no cambia las odds.
- $OR > 1$: las multiplica (aumenta el riesgo).
- $OR < 1$: las divide (lo reduce).

### ✍️ Ejercicio A3
Calcula `exp(params)` y su intervalo de confianza del 95% (recuerda: hay que exponenciar los límites del IC de los coeficientes, no calcular un IC nuevo). Luego calcula el OR para 10 y para 20 cigarrillos adicionales ($OR_{10} = e^{10\\beta}$). Escribe una frase interpretando el OR de `cigs`.
""")
code("""
# tu código aquí
""")
solution("""
or_tabla = np.exp(modelo_cigs.params)
ci_or = np.exp(modelo_cigs.conf_int())
print(or_tabla, "\\n")
print(ci_or, "\\n")

or_10 = np.exp(10 * modelo_cigs.params["cigs"])
or_20 = np.exp(20 * modelo_cigs.params["cigs"])
print(f"OR por 10 cigarrillos extra: {or_10:.4f}")
print(f"OR por 20 cigarrillos extra: {or_20:.4f}")
""", "`exp(cigs)` = 1.0235 con IC95% [1.015, 1.032]: **cada cigarrillo diario adicional multiplica las odds de CHD por 1.0235, es decir +2.35%**. "
     "Para 10 cigarrillos extra: $e^{10 \\times 0.0232} = 1.2614$ → **+26% en las odds**. "
     "⚠️ **Alerta**: la guía en PDF dice '+35%' para este mismo cálculo — es un error de la guía, no tuyo; con los coeficientes exactos del modelo el valor correcto es **+26%**. "
     "Para 20 cigarrillos: ×1.591 → +59% (no es el doble de +26%, porque las odds se multiplican, no se suman).")

md("""
### A4 · Probabilidad predicha (10 min)
El OR describe **odds**, no probabilidad, y mucha gente las confunde. La relación completa es:

$$\\text{odds} = \\frac{p}{1-p} \\qquad p = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x)}}$$

### ✍️ Ejercicio A4
Para `cigs` = 0, 20 y 40: calcula la probabilidad predicha **a mano**, con la fórmula de arriba usando `modelo_cigs.params`. Después compárala con `modelo_cigs.predict(...)`. ¿Coinciden?
""")
code("""
# tu código aquí
""")
solution("""
b0, b1 = modelo_cigs.params["Intercept"], modelo_cigs.params["cigs"]
for cig in [0, 20, 40]:
    p_formula = 1 / (1 + np.exp(-(b0 + b1 * cig)))
    p_predict = modelo_cigs.predict(pd.DataFrame({"cigs": [cig]})).iloc[0]
    print(f"cigs={cig:>2}: fórmula={p_formula:.4f}  predict()={p_predict:.4f}")
""", "p(0) = 6.05%, p(20) = 9.30%, p(40) = 14.02%: la fórmula a mano y `.predict()` coinciden siempre (predict solo automatiza la misma cuenta). "
     "Nota la **no linealidad**: de 20 a 40 cigarrillos la probabilidad no se duplica (9.3% → 14.0%, no 18.6%), porque lo que se multiplica linealmente son las **odds**, no la probabilidad.")

md("""
### A5 · ¿Ayuda agregar `age`? (10 min)
Para comparar dos modelos **anidados** (uno es caso particular del otro) se usa el **test de razón de verosimilitud (LR test)**:

$$LR = \\text{deviance}_{\\text{reducido}} - \\text{deviance}_{\\text{completo}} \\sim \\chi^2_{df}$$

donde $df$ es la cantidad de parámetros nuevos. Se calcula el p-valor con `scipy.stats.chi2.sf(LR, df)`.

### ✍️ Ejercicio A5
Ajusta `chd01 ~ cigs + age`, calcula el LR contra `modelo_cigs` y su p-valor (df=1). Compara también los AIC de ambos modelos.
""")
code("""
# tu código aquí
""")
solution("""
modelo_cigs_edad = smf.glm("chd01 ~ cigs + age", wcgs, family=sm.families.Binomial()).fit()

LR = modelo_cigs.deviance - modelo_cigs_edad.deviance
p_valor = stats.chi2.sf(LR, df=1)
print(f"LR = {LR:.3f}, p = {p_valor:.2e}")
print(f"AIC solo cigs:      {modelo_cigs.aic:.2f}")
print(f"AIC cigs + age:     {modelo_cigs_edad.aic:.2f}")
""", "LR = 44.109, p = 3.1×10⁻¹¹: extremadamente significativo, `age` sí aporta. El AIC baja de 1753.97 a 1711.86 (mientras menor, mejor): confirma lo mismo que el LR test, "
     "pero el LR test además te da un p-valor formal mientras que el AIC solo ordena modelos sin decir \"significativo\" o no.")

md("""
### A6 · Logit vs. probit (5 min)
El link logit no es la única opción para una respuesta binaria. El link **probit** usa la función de distribución acumulada normal en vez de la logística. En la práctica casi siempre se usa logit porque sus coeficientes se interpretan como odds ratios; probit no tiene una interpretación tan directa.
""")
code("""
modelo_probit = smf.glm("chd01 ~ cigs", wcgs, family=sm.families.Binomial(link=sm.families.links.Probit())).fit()
print(f"AIC logit:  {modelo_cigs.aic:.2f}")
print(f"AIC probit: {modelo_probit.aic:.2f}")
""")
md("""
AIC logit = 1753.97 vs. AIC probit = 1753.49: la diferencia es mínima (0.48 puntos). Con una sola variable continua, ambos links casi siempre dan un ajuste casi idéntico; la elección rara vez cambia las conclusiones.
""")

# ================================================================== PARTE B
md("""
## Parte B · Evaluar un clasificador

Ajustar un modelo es la mitad del trabajo: también hay que medir **qué tan bien clasifica**, y el accuracy solo no basta cuando las clases están desbalanceadas (recuerda el 8.15% de A1).

### B1 · Train/test y la trampa del accuracy (15 min)
Dividimos los datos en entrenamiento (70%) y prueba (30%) con `train_test_split`. `stratify=y` asegura que ambos conjuntos mantengan la misma proporción de eventos. Ajustamos en *train*, predecimos en *test*.

### ✍️ Ejercicio B1
1. Elimina filas con NaN en `cigs`, `age` o `chd01`.
2. Divide 70/30 con `train_test_split(..., test_size=0.3, random_state=42, stratify=y)`.
3. Ajusta `chd01 ~ cigs + age` **solo en train**.
4. Predice probabilidades en *test* y clasifica con umbral 0.5.
5. Calcula la matriz de confusión (`sklearn.metrics.confusion_matrix`) y el accuracy.
""")
code("""
# tu código aquí
""")
solution("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

wcgs_b = wcgs.dropna(subset=["cigs", "age", "chd01"])
train, test = train_test_split(wcgs_b, test_size=0.3, random_state=42, stratify=wcgs_b["chd01"])

modelo_b = smf.glm("chd01 ~ cigs + age", train, family=sm.families.Binomial()).fit()
probs = modelo_b.predict(test)
pred_05 = (probs >= 0.5).astype(int)

cm = confusion_matrix(test["chd01"], pred_05)
print("Matriz de confusión (umbral 0.5):\\n", cm)
print(f"Accuracy: {accuracy_score(test['chd01'], pred_05):.4f}")
""", "Matriz: `[[869, 1], [77, 0]]` (filas = real no/yes, columnas = predicho no/yes). **Accuracy = 91.76%**, que suena excelente... "
     "pero el modelo predijo \"yes\" una sola vez en los 947 casos de test, y esa única predicción fue incorrecta. "
     "**Nunca** detectó un caso real de CHD (los 77 \"yes\" reales quedaron todos en la celda inferior izquierda). Es la misma trampa que viste en clase con 93%/0%: "
     "con clases desbalanceadas, un modelo que casi siempre dice \"no\" tiene accuracy alto sin servir para nada.")

md("""
### B2 · Sensibilidad, especificidad y precisión a mano (10 min)
De la matriz de confusión (`TN, FP, FN, TP` con `.ravel()`):

$$\\text{sensibilidad} = \\frac{TP}{TP+FN} \\qquad \\text{especificidad} = \\frac{TN}{TN+FP} \\qquad \\text{precisión} = \\frac{TP}{TP+FP}$$

### ✍️ Ejercicio B2
Usando la matriz `cm` de B1, calcula las tres métricas a mano (sin funciones de sklearn).
""")
code("""
# tu código aquí
""")
solution("""
tn, fp, fn, tp = cm.ravel()
sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Sensibilidad: {sensibilidad:.4f}")
print(f"Especificidad: {especificidad:.4f}")
print(f"Precisión: {precision:.4f}")
""", "TN=869, FP=1, FN=77, TP=0. **Sensibilidad = 0%** (no detecta ningún caso real), **especificidad = 99.89%** (casi perfecto en los sanos, porque casi nunca predice \"yes\"), "
     "**precisión = 0%** (la única predicción positiva fue un falso positivo). Con umbral 0.5 el modelo es inútil para detectar CHD, aunque su accuracy global sea alto.")

md("""
### B3 · Curva ROC, AUC y umbral de Youden (15 min)
El umbral 0.5 es arbitrario. La **curva ROC** grafica *sensibilidad (TPR)* vs. *1 - especificidad (FPR)* para **todos** los umbrales posibles. El **AUC** (área bajo la curva) resume qué tan bien el modelo *ordena* los casos, sin depender de un umbral.

El **índice de Youden** $J = TPR - FPR$ elige el umbral que maximiza esa diferencia: un compromiso razonable entre detectar casos y no generar demasiados falsos positivos.

### ✍️ Ejercicio B3
1. Calcula `roc_curve` y `auc` sobre las probabilidades de test.
2. Encuentra el umbral que maximiza `tpr - fpr`.
3. Reclasifica con ese umbral y recalcula la matriz de confusión y las métricas de B2.
4. Compara con el umbral 0.5: ¿qué se gana y qué se pierde?
""")
code("""
# tu código aquí
""")
solution("""
from sklearn.metrics import roc_curve, auc

fpr, tpr, umbrales = roc_curve(test["chd01"], probs)
auc_valor = auc(fpr, tpr)
print(f"AUC: {auc_valor:.4f}")

youden = tpr - fpr
idx = np.argmax(youden)
umbral_youden = umbrales[idx]
print(f"Umbral de Youden: {umbral_youden:.4f} (TPR={tpr[idx]:.4f}, FPR={fpr[idx]:.4f})")

pred_youden = (probs >= umbral_youden).astype(int)
cm_youden = confusion_matrix(test["chd01"], pred_youden)
tn2, fp2, fn2, tp2 = cm_youden.ravel()
print("\\nMatriz con umbral de Youden:\\n", cm_youden)
print(f"Accuracy:     {accuracy_score(test['chd01'], pred_youden):.4f}")
print(f"Sensibilidad: {tp2/(tp2+fn2):.4f}")
print(f"Especificidad:{tn2/(tn2+fp2):.4f}")
print(f"Precisión:    {tp2/(tp2+fp2):.4f}")
""", "AUC = 0.6370 (un modelo sin ninguna capacidad de discriminación tendría AUC = 0.5; este es modesto pero mejor que el azar). "
     "El umbral de Youden es ≈0.105 (mucho más bajo que 0.5, razonable porque la prevalencia es de solo 8%). Con ese umbral: matriz `[[681, 189], [42, 35]]`, "
     "**accuracy = 75.61%** (bajó mucho), pero **sensibilidad = 45.45%** (de detectar 0 casos pasa a detectar el 45%) y especificidad = 78.28%, precisión = 15.62%. "
     "**El tradeoff**: bajar el umbral sacrifica accuracy global y dispara los falsos positivos (de 1 a 189), pero multiplica la capacidad de detectar casos reales. "
     "Cuál es mejor depende del costo de un falso negativo (no detectar una CHD real) frente al de un falso positivo (una alarma de más): en salud, casi siempre el falso negativo es más caro. "
     "**Nota**: estos números van a diferir un poco de los que viste en clase, porque la partición train/test (y por lo tanto el modelo ajustado) es distinta cada vez que cambia `random_state` o las variables usadas.")

# ================================================================== PARTE C
md("""
## Parte C · Poisson con offset (`cancer_pulmon`)

### C1 · Tasas por 1000 habitantes (10 min)
`cancer_pulmon` tiene conteos (`casos`) y su población expuesta (`poblacion`) por grupo edad × ciudad. Comparar conteos crudos entre grupos es engañoso si los grupos tienen poblaciones distintas: hay que mirar la **tasa**.

### ✍️ Ejercicio C1
1. Crea `tasa_mil = casos / poblacion * 1000`.
2. Grafica la tasa por grupo de edad, una línea por ciudad.
""")
code("""
# tu código aquí
""")
solution("""
cancer["tasa_mil"] = cancer["casos"] / cancer["poblacion"] * 1000
orden_edad = ["40-54", "55-59", "60-64", "65-69", "70-74", "Over 74"]

fig, ax = plt.subplots(figsize=(7, 4))
for ciudad in sorted(cancer["ciudad"].unique()):
    sub = cancer[cancer["ciudad"] == ciudad].set_index("edad").loc[orden_edad]
    ax.plot(orden_edad, sub["tasa_mil"], marker="o", label=ciudad)
ax.set_ylabel("Casos por 1000 habitantes")
ax.set_xlabel("Grupo de edad")
ax.legend()
plt.tight_layout()
plt.show()
""", "La tasa sube con la edad en las 4 ciudades, con un máximo alrededor de 70-74 años. Las curvas de las 4 ciudades son parecidas en forma, aunque Fredericia se ve algo más alta en casi todos los grupos: eso es justamente lo que vamos a probar formalmente en C2-C4.")

md("""
### C2 · Poisson con offset (10 min)
Para modelar una **tasa** (no un conteo crudo) con Poisson, se usa un *offset*: $\\log(\\mu) = \\log(\\text{población}) + X\\beta$. El offset fuerza el coeficiente de $\\log(\\text{población})$ a ser exactamente 1, así el modelo estima efectos sobre la **tasa**, no sobre el conteo total (que obviamente es mayor donde hay más gente).

Los coeficientes exponenciados son **razones de tasas (rate ratios, RR)**, con la misma lógica que el odds ratio: $RR=1$ sin efecto, $RR>1$ más riesgo, $RR<1$ menos riesgo. La referencia es la categoría alfabéticamente primera: ciudad "Fredericia" y edad "40-54".

### ✍️ Ejercicio C2
Ajusta `casos ~ C(ciudad) + C(edad)` con `family=sm.families.Poisson()` y `offset=np.log(poblacion)`. Calcula los RR (`exp(params)`) e interpreta el de Horsens y el del grupo 70-74.
""")
code("""
# tu código aquí
""")
solution("""
modelo_pois = smf.glm("casos ~ C(ciudad) + C(edad)", cancer,
                       family=sm.families.Poisson(), offset=np.log(cancer["poblacion"])).fit()
print(modelo_pois.summary())

rr = np.exp(modelo_pois.params)
print(rr)
""", "RR Horsens = 0.719: ajustando por edad, Horsens tiene una tasa de cáncer de pulmón **28% menor** que Fredericia (la referencia). "
     "RR del grupo 70-74 = 6.404: ajustando por ciudad, las personas de 70-74 tienen una tasa **6.4 veces** la del grupo de referencia (40-54). "
     "Deviance residual = 23.45 (df=15), AIC = 137.84.")

md("""
### C3 · Sobre-dispersión y quasi-Poisson (15 min)
Poisson asume que **varianza = media**. En la práctica casi nunca se cumple exactamente; cuando la varianza real es mayor, hay **sobre-dispersión**, y los errores estándar de Poisson quedan subestimados (p-valores artificialmente chicos).

Se mide con $\\phi = \\dfrac{\\chi^2_{\\text{Pearson}}}{df_{\\text{residual}}}$. Si $\\phi \\gg 1$, conviene un **quasi-Poisson** (`fit(scale="X2")`): mantiene los mismos coeficientes puntuales, pero infla los errores estándar por $\\sqrt{\\phi}$.

### ✍️ Ejercicio C3
1. Calcula $\\phi$ para `modelo_pois`.
2. Reajusta con `fit(scale="X2")` y compara los SE y p-valores de Horsens entre ambos ajustes.
""")
code("""
# tu código aquí
""")
solution("""
phi = modelo_pois.pearson_chi2 / modelo_pois.df_resid
print(f"phi = {phi:.4f}")

modelo_quasi = smf.glm("casos ~ C(ciudad) + C(edad)", cancer,
                        family=sm.families.Poisson(), offset=np.log(cancer["poblacion"])).fit(scale="X2")

comparacion = pd.DataFrame({
    "coef": modelo_pois.params,
    "SE_poisson": modelo_pois.bse,
    "SE_quasi": modelo_quasi.bse,
    "p_poisson": modelo_pois.pvalues,
    "p_quasi": modelo_quasi.pvalues,
})
print(comparacion.round(4))
""", "$\\phi = 1.504$: hay sobre-dispersión moderada (50% más varianza de la que Poisson esperaría). Los coeficientes puntuales **no cambian** entre Poisson y quasi-Poisson (misma columna `coef`), "
     "pero el SE de Horsens pasa de 0.1826 a 0.2226 (factor $\\sqrt{1.504} \\approx 1.226$), y su p-valor sube de 0.069 a 0.138: lo que con Poisson parecía *casi* significativo, con la corrección deja de serlo. "
     "Ignorar la sobre-dispersión te hace ver significados donde no los hay.")

md("""
### C4 · ¿Aporta la ciudad una vez controlada la edad? (10 min)
En C2 viste diferencias entre ciudades. Pero, ¿son reales o solo reflejan que las ciudades tienen distinta composición de edad? Se responde con un test de razón de verosimilitud entre el modelo con solo edad y el modelo completo.

### ✍️ Ejercicio C4
Ajusta `casos ~ C(edad)` (sin ciudad), calcula el LR contra `modelo_pois` (df = 3, porque son 4 ciudades − 1) y su p-valor.
""")
code("""
# tu código aquí
""")
solution("""
modelo_edad = smf.glm("casos ~ C(edad)", cancer,
                       family=sm.families.Poisson(), offset=np.log(cancer["poblacion"])).fit()

LR = modelo_edad.deviance - modelo_pois.deviance
p_valor = stats.chi2.sf(LR, df=3)
print(f"LR = {LR:.3f}, df=3, p = {p_valor:.3f}")
""", "LR = 4.859, p = 0.182: **no hay evidencia de que la ciudad aporte** una vez que la edad ya está en el modelo. "
     "Las diferencias que viste en C1/C2 probablemente reflejan que las ciudades tienen distinta composición de edad (confusión), no un efecto propio de vivir en una u otra ciudad.")

# ================================================================== PARTE D
md("""
## Parte D · Multinomial (`nes96`)

### D1 · Construir PID3 (5 min)
`PID` tiene 7 categorías detalladas. Las agrupamos en 3: Demócrata, Republicano e Independiente.

### ✍️ Ejercicio D1
Crea `PID3` agrupando `strDem`/`weakDem`/`indDem` → "Democrat", `strRep`/`weakRep`/`indRep` → "Republican", `indind` → "Independent". Verifica los conteos.
""")
code("""
# tu código aquí
""")
solution("""
def construir_pid3(pid):
    if pid in ("strDem", "weakDem", "indDem"):
        return "Democrat"
    if pid in ("strRep", "weakRep", "indRep"):
        return "Republican"
    return "Independent"

nes["PID3"] = nes["PID"].apply(construir_pid3)
print(nes["PID3"].value_counts())
print(nes["PID3"].value_counts(normalize=True).round(4))
""", "Democrat 488 (51.7%), Republican 419 (44.4%), Independent 37 (3.9%). Independent es una clase muy minoritaria: ya anticipa el problema de D4.")

md("""
### D2 · Modelo multinomial simple (10 min)
En un modelo multinomial se elige una categoría de referencia (aquí, Democrat) y se estima **un set de coeficientes por cada otra categoría**, comparándola contra la referencia. `exp(coef)` se llama **razón de riesgo relativo (RRR)**: cuánto se multiplica la probabilidad relativa de esa categoría frente a la referencia por unidad de X.

Para simplificar, usamos `educ_num` (posición de `educ` en su orden natural: MS=0 ... MAdeg=6) en vez de todas las categorías de `educ` como dummies.

### ✍️ Ejercicio D2
1. Crea `y` = 0 (Democrat), 1 (Republican), 2 (Independent) y `educ_num` según el orden `['MS','HSdrop','HS','Coll','CCdeg','BAdeg','MAdeg']`.
2. Ajusta `smf.mnlogit("y ~ age + educ_num", nes)`.
3. Calcula `exp(params)` e interpreta el coeficiente de `educ_num` para Republicano y el de `age` para Independiente.
""")
code("""
# tu código aquí
""")
solution("""
orden_educ = ["MS", "HSdrop", "HS", "Coll", "CCdeg", "BAdeg", "MAdeg"]
nes["educ_num"] = nes["educ"].apply(lambda e: orden_educ.index(e))

mapa_y = {"Democrat": 0, "Republican": 1, "Independent": 2}
nes["y"] = nes["PID3"].map(mapa_y)

modelo_simple = smf.mnlogit("y ~ age + educ_num", nes).fit(disp=False)
print(modelo_simple.summary())
print(np.exp(modelo_simple.params))
""", "Para **Republicano** (vs. Democrat, la referencia): `educ_num` = 0.1095 (p = 0.010, significativo), RRR = $e^{0.1095}$ = 1.1157 → cada escalón educativo aumenta ~11.6% la probabilidad relativa de ser republicano. "
     "`age` = 0.0068 (p = 0.102, **no** significativo). "
     "Para **Independiente**: `educ_num` = -0.0845 (p = 0.445) y `age` = 0.0019 (p = 0.856), ninguno significativo: ni la edad ni la educación distinguen bien a los independientes con este modelo, lo cual va a reaparecer en D4.")

md("""
### D3 · Probabilidades predichas (10 min)
`.predict()` en un modelo multinomial devuelve **una probabilidad por categoría**, y las tres siempre suman 1.

### ✍️ Ejercicio D3
Con `educ_num` fijo en "BAdeg", predice las probabilidades para una persona de 30 años y una de 70 años. Verifica que cada fila suma 1 e interpreta la diferencia.
""")
code("""
# tu código aquí
""")
solution("""
educ_num_badeg = orden_educ.index("BAdeg")
nuevos = pd.DataFrame({"age": [30, 70], "educ_num": [educ_num_badeg, educ_num_badeg]})

probs_pred = modelo_simple.predict(nuevos)
probs_pred.columns = ["Democrat", "Republican", "Independent"]
probs_pred.index = ["30 años", "70 años"]

print(probs_pred.round(4))
print("\\nSuma por fila:\\n", probs_pred.sum(axis=1))
""", "30 años (BAdeg): Democrat 51.10%, Republican 45.63%, Independent 3.27%. "
     "70 años (BAdeg): Democrat 44.66%, Republican 52.26%, Independent 3.08%. Ambas filas suman 1.0000. "
     "Con la misma educación, envejecer desplaza probabilidad de Demócrata a Republicano (consistente con el coeficiente positivo de `age` en D2), mientras que la probabilidad de Independiente se mantiene baja (~3%) casi sin cambios con la edad.")

md("""
### D4 · Accuracy y la clase fantasma (5 min)
Para clasificar, se elige la categoría con mayor probabilidad predicha (`.idxmax(axis=1)`) y se compara contra la real.

### ✍️ Ejercicio D4
Ajusta el modelo completo `y ~ age + C(educ) + C(income)`, calcula el accuracy de clasificación y cuenta cuántas veces se predijo cada clase. ¿Qué pasa con Independent?
""")
code("""
# tu código aquí
""")
solution("""
modelo_completo = smf.mnlogit("y ~ age + C(educ) + C(income)", nes).fit(disp=False)
print(f"Deviance: {-2*modelo_completo.llf:.1f}  AIC: {modelo_completo.aic:.1f}")

pred_clase = modelo_completo.predict(nes).idxmax(axis=1)
accuracy = (pred_clase.values == nes["y"].values).mean()
print(f"Accuracy: {accuracy:.4f}")
print(pred_clase.value_counts())
""", "Deviance = 1458.5, AIC = 1582.5 (test LR contra el modelo nulo: LR=105.85, p=0.00024: el modelo completo sí mejora sobre predecir solo con la proporción base). "
     "Accuracy = 57.63%. Pero al contar las clases predichas: Democrat (0) se predijo 536 veces, Republican (1) 408 veces, e **Independent (2) nunca**, ni una sola vez en 944 observaciones. "
     "Es la misma trampa de la Parte B pero con 3 clases: Independent es solo el 3.9% de los datos y, con estos predictores, el modelo nunca le asigna la probabilidad más alta a nadie — exactamente lo que viste en D3, donde su probabilidad predicha nunca pasó de ~3-5%. "
     "(Nota: `income` tiene 24 categorías; con pocas observaciones en algunos tramos aparecen coeficientes enormes y una advertencia de convergencia — es una señal de casi-separación perfecta, no un error tuyo.)")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (15 min, SIN mirar lo anterior)

Con `wcgs`, ajusta un modelo logístico para `chd01` usando `cigs`, `age`, `chol` y `sdp` (elimina filas con NaN primero).

1. Ajusta el modelo.
2. Construye una tabla con el OR, su IC 95% y el p-valor de cada variable.
3. Escribe una conclusión de **3 líneas**: ¿qué variable tiene el mayor efecto relativo? ¿todas son significativas? ¿qué le dirías a alguien que fuma y tiene presión alta?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi conclusión:**

1.
2.
3.
""")
solution("""
wcgs_cierre = wcgs.dropna(subset=["cigs", "age", "chol", "sdp", "chd01"])
modelo_cierre = smf.glm("chd01 ~ cigs + age + chol + sdp", wcgs_cierre, family=sm.families.Binomial()).fit()

orr = np.exp(modelo_cierre.params)
ci = np.exp(modelo_cierre.conf_int())
tabla = pd.DataFrame({"OR": orr, "IC_2.5%": ci[0], "IC_97.5%": ci[1], "p": modelo_cierre.pvalues})
print(tabla.round(4))
""", "Con n=3142 (tras quitar 12 NaN de `chol`): OR cigs=1.0224 [1.0141, 1.0309], age=1.0668 [1.0421, 1.0920], chol=1.0111 [1.0082, 1.0141], sdp=1.0215 [1.0136, 1.0293]. "
     "**Las 4 son significativas (p < 0.001)**. En términos relativos por unidad, `age` tiene el mayor OR (6.68% más odds por año), pero recuerda que \"por unidad\" no es comparable entre escalas distintas (un año no es lo mismo que un mg/dL de colesterol). "
     "**Conclusión esperada**: fumar, la edad, el colesterol y la presión sistólica aumentan de forma independiente y significativa las odds de un evento coronario; alguien que fuma y tiene presión alta acumula ambos efectos multiplicativamente (no se suman, se multiplican), por lo que el riesgo conjunto es mayor que cualquiera de los dos factores por separado.")

# ---------------------------------------------------------------- self-eval
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué hay que exponenciar el coeficiente de una regresión logística para interpretarlo como "cuánto cambia el riesgo"?
2. En un clasificador con clases desbalanceadas (como el 8% de CHD), ¿por qué el accuracy puede ser engañoso y qué otras dos métricas conviene mirar?
3. En Poisson con offset, ¿qué significa que $\\phi$ (la razón de sobre-dispersión) sea mucho mayor que 1, y qué corrección se aplica?
4. En un modelo multinomial, ¿contra qué categoría se interpreta cada coeficiente?

<details><summary><b>Respuestas</b></summary>

1. El coeficiente está en la escala de **log-odds**, que no tiene una interpretación intuitiva directa. Exponenciarlo lo convierte en un **odds ratio**: cuánto se multiplican las odds por cada unidad de la variable.
2. El accuracy solo mide "cuántas veces acertó en total", y con clases desbalanceadas un modelo que casi siempre predice la clase mayoritaria tiene accuracy alto sin detectar nunca la clase minoritaria. Conviene mirar **sensibilidad** (detecta los positivos reales) y **especificidad** (o AUC, que resume el desempeño en todos los umbrales).
3. Significa que la varianza real de los datos es mayor que la que Poisson predice (varianza = media). Se corrige con un **quasi-Poisson** (`fit(scale="X2")`), que mantiene los coeficientes puntuales pero infla los errores estándar, evitando p-valores artificialmente pequeños.
4. Contra la **categoría de referencia** (la que el modelo deja fuera, por defecto la codificada como 0). Cada set de coeficientes compara esa categoría contra la de referencia, nunca entre dos categorías no-referencia directamente.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** modelos mixtos / series de tiempo (según el plan del diplomado). Revisa `Guias/apuntes/03_Ejercicios_Cuaderno_GLM.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT)
