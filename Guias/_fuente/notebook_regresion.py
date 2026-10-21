"""Genera el notebook guiado de ANOVA y Regresión (Clases 17-19)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "C17-19_Regresion.ipynb"
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
# Clases 17-19 · ANOVA y Análisis de Regresión
**Diplomado Data Science** · 2 sesiones de ~95 min · Prof. Ana María Alvarado

Material de apoyo:
- `Guias/02_Regresion_C17-19.pdf` — guía de la clase (en R).
- `Guias/apuntes/02_Apuntes_Regresion_C17-19.pdf` — apunte con pistas conceptuales.
- `Guias/apuntes/02_Ejercicios_Cuaderno_Regresion.pdf` — ejercicios en papel (antes o después de este notebook).

Las clases 17 a 19 se dieron en **R**. Este notebook practica los mismos conceptos en **Python con `statsmodels`**, que es la librería que vas a usar en el resto del diplomado y en el mundo real.

⚠️ **Antes de empezar**: este notebook necesita `statsmodels` y `openpyxl` (para leer el `.xlsx`). Instala en tu venv:
```
pip install statsmodels openpyxl
```

### Objetivos de estas sesiones
1. Ajustar e interpretar un **ANOVA de una y dos vías**, y saber cuándo usar Tukey.
2. Distinguir correlación de regresión, ajustar una **regresión lineal simple** y revisar sus supuestos (normalidad, independencia, homoscedasticidad).
3. Ajustar una **regresión lineal múltiple**, interpretar cada coeficiente, entender la trampa de las variables dummy y comparar modelos con R² ajustado, AIC y VIF.
4. Usar selección de variables (*forward*/*backward* por AIC) y aplicar un modelo a **datos nuevos**, incluyendo detectar cuándo esos datos nuevos están corrompidos.

### Plan de las sesiones
| Sesión | Min | Bloque |
|---|---|---|
| 1 | 5 | Preparación y carga de datos |
| 1 | 10 | A1 · Explorar `salarios` por industria |
| 1 | 15 | A2 · ANOVA de una vía |
| 1 | 10 | A3 · Tukey: ¿qué pares difieren? |
| 1 | 15 | A4 · ANOVA de dos vías e interacción |
| 1 | 10 | A5 · Normalidad de los residuos |
| 1 | 10 | B1 · Correlación: Pearson vs. Spearman |
| 1 | 10 | B2 · Regresión simple `bio ~ ph` |
| 2 | 15 | B3 · Diagnóstico de la regresión simple |
| 2 | 10 | B4 · Puntos influyentes (distancia de Cook) |
| 2 | 10 | C1 · Regresión múltiple de salarios |
| 2 | 10 | C2 · La trampa de las variables dummy |
| 2 | 10 | C3 · R² vs. R² ajustado y VIF |
| 2 | 15 | C4 · Selección de variables (forward/backward) |
| 2 | 10 | C5 · Diagnóstico del modelo seleccionado |
| 2 | 5 | D1 · Predecir con `bio_new` (y que explote) |
| 2 | 10 | D2 · Detectar el error en los datos nuevos |
| 2 | 10 | D3 · Predecir con un modelo confiable |
| 2 | 15 | Ejercicio de cierre **sin ayuda** |
| 2 | 5 | Autoevaluación y notas |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla: equivocarse primero es lo que fija el aprendizaje.
- Los números que aparecen en el texto (coeficientes, p-valores, R², AIC) están verificados contra la salida real de este notebook. Si a ti te salen ligeramente distintos revisa primero la versión de `statsmodels`.
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

# Busca la carpeta de la clase subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase 17*")))
carpeta = next(raiz.glob("Clase 17*"))

salarios = pd.read_excel(carpeta / "salarios.xlsx")

biomasa = pd.read_csv(carpeta / "biomass.txt", sep="\\t")
biomasa.columns = biomasa.columns.str.lower()

bio_nuevo = pd.read_csv(carpeta / "bio_new.csv", sep=";", encoding="utf-8-sig")
bio_nuevo.columns = bio_nuevo.columns.str.lower()

print("salarios:", salarios.shape, "| biomasa:", biomasa.shape, "| bio_nuevo:", bio_nuevo.shape)
salarios.head()
""")
md("""
**salarios.xlsx** (n=40): `Salario` (en millones), `Industria` (Banca, Minería, Retail), `Sexo` ("H" = hombre, "M" = mujer) y años de `Experiencia`.

**biomass.txt** (n=45): biomasa de marismas (`bio`) y 14 variables de suelo (acidez, salinidad, nutrientes, metales) en 3 tipos de vegetación (`type`: DVEG, SHRT, TALL) y 3 localidades (`loc`). En la clase de R este archivo se procesó con `janitor::clean_names()`; aquí lo replicamos bajando todos los nombres de columna a minúscula.

**bio_new.csv** (9 filas): mismas columnas que `biomass.txt`, pero sin la columna `bio` (es justamente lo que vamos a predecir en la Parte D). El archivo trae un `;` como separador y una marca BOM al inicio, por eso se lee con `sep=";"` y `encoding="utf-8-sig"`.
""")

# ================================================================== PARTE A
md("""
## Parte A · ANOVA (`salarios`)

### A1 · Explorar `salarios` por industria (10 min)
Antes de correr cualquier test, siempre conviene mirar los datos. Un ANOVA compara **medias de grupos**; un boxplot y una tabla de medias ya adelantan si va a haber diferencias.

### ✍️ Ejercicio A1
Calcula la media y el tamaño de cada grupo de `Industria`, y haz un boxplot de `Salario` por `Industria`.
""")
code("""
# tu código aquí
""")
solution("""
print(salarios.groupby("Industria")["Salario"].agg(["mean", "std", "count"]))

fig, ax = plt.subplots(figsize=(6, 4))
salarios.boxplot(column="Salario", by="Industria", ax=ax)
ax.set_title("")
plt.suptitle("")
plt.tight_layout()
plt.show()
""", "Medias: Banca 3.358, Minería 4.477, Retail 2.903 (n=40 en total, los tres grupos no son necesariamente del mismo tamaño). "
     "A simple vista Minería paga más y Retail menos, pero \"a simple vista\" no es una prueba estadística: para eso está el ANOVA de A2.")

md("""
### A2 · ANOVA de una vía (15 min)
El ANOVA de una vía prueba $H_0$: todas las medias de grupo son iguales, contra $H_1$: al menos una difiere. En statsmodels se ajusta como una regresión con una variable categórica y se extrae la tabla de ANOVA con `sm.stats.anova_lm`:

```python
modelo = smf.ols("Salario ~ C(Industria)", salarios).fit()
tabla = sm.stats.anova_lm(modelo, typ=1)
```

Columnas de la tabla:
- **sum_sq**: suma de cuadrados (variabilidad explicada por el grupo, y la que queda sin explicar en "Residual").
- **df**: grados de libertad (g-1 para el factor, n-g para el residual).
- **mean_sq** = sum_sq / df.
- **F** = mean_sq(grupo) / mean_sq(residual): mientras mayor, más evidencia de que los grupos difieren.
- **PR(>F)**: el p-valor del test F.

### ✍️ Ejercicio A2
Ajusta `Salario ~ C(Industria)` y muestra la tabla de ANOVA. ¿Rechazas $H_0$ al 5%? Mira también `modelo.params`: ¿cuál es la categoría de referencia y qué significa cada coeficiente?
""")
code("""
# tu código aquí
""")
solution("""
modelo_industria = smf.ols("Salario ~ C(Industria)", salarios).fit()
tabla_anova = sm.stats.anova_lm(modelo_industria, typ=1)
print(tabla_anova)
print()
print(modelo_industria.params)
""", "F = 18.33, p ≈ 3.0 × 10⁻⁶ (df 2 y 37): se rechaza $H_0$ con amplio margen, hay diferencias reales entre industrias. "
     "SS(Industria) = 17.88, SS(Residual) = 18.04. "
     "`Industria` queda codificada con **Banca como referencia** (es la primera alfabéticamente): Intercept = 3.358 (la media de Banca), `C(Industria)[T.Minería]` = +1.119 (Minería gana 1.119 más que Banca en promedio) y `C(Industria)[T.Retail]` = -0.455 (Retail gana 0.455 menos que Banca). "
     "El ANOVA te dice que *hay* diferencia entre grupos, pero no *cuáles* pares difieren: para eso es Tukey en A3.")

md("""
### A3 · Tukey: ¿qué pares difieren? (10 min)
El test F del ANOVA es global: si rechazas $H_0$, sabes que no todas las medias son iguales, pero no cuáles. Comparar cada par con un t-test individual infla el error tipo I (más comparaciones, más chance de un falso positivo). Tukey HSD corrige por esto.

### ✍️ Ejercicio A3
Usa `pairwise_tukeyhsd` de `statsmodels.stats.multicomp` sobre `Salario` agrupado por `Industria`. ¿Qué pares rechazan $H_0$ al 5%?
""")
code("""
# tu código aquí
""")
solution("""
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(salarios["Salario"], salarios["Industria"])
print(tukey)
""", "Banca-Minería: diff = 1.119, p = 0.0008 → **se rechaza** (difieren). "
     "Banca-Retail: diff = -0.455, p = 0.225 → no se rechaza (no hay evidencia suficiente de diferencia). "
     "Minería-Retail: diff = -1.574, p ≈ 0 → **se rechaza** (difieren, y es la diferencia más grande de las tres). "
     "Conclusión: Minería paga distinto a los otros dos, pero Banca y Retail no son distinguibles entre sí con estos datos.")

md("""
### A4 · ANOVA de dos vías e interacción (15 min)
¿El efecto de la industria sobre el salario es el mismo para hombres y mujeres? Eso es justamente lo que prueba el término de **interacción** `C(Industria):C(Sexo)` en un ANOVA de dos vías. Si la interacción es significativa, no tiene sentido hablar de "el efecto de la industria" sin especificar el sexo (y viceversa).

### ✍️ Ejercicio A4
1. Ajusta `Salario ~ C(Industria) * C(Sexo)` (el `*` agrega automáticamente los efectos simples y la interacción) y muestra la tabla de ANOVA.
2. Calcula la tabla de medias `Industria × Sexo` con `groupby(...).mean().unstack()` y grafica un gráfico de interacción (una línea por sexo, industria en el eje x).
""")
code("""
# tu código aquí
""")
solution("""
modelo_dosvias = smf.ols("Salario ~ C(Industria) * C(Sexo)", salarios).fit()
print(sm.stats.anova_lm(modelo_dosvias, typ=1))

medias = salarios.groupby(["Industria", "Sexo"])["Salario"].mean().unstack()
print()
print(medias)

fig, ax = plt.subplots(figsize=(6, 4))
for sexo in medias.columns:
    ax.plot(medias.index, medias[sexo], marker="o", label=sexo)
ax.set_ylabel("Salario promedio")
ax.set_xlabel("Industria")
ax.legend(title="Sexo")
plt.tight_layout()
plt.show()
""", "Industria: F = 23.29, p = 4.3×10⁻⁷. Sexo: F = 5.46, p = 0.025. **Interacción: F = 3.77, p = 0.033** → significativa al 5%: "
     "**sí hay interacción**. Se ve en las medias: en Banca (H 3.633 vs. M 3.083) y en Retail (H 3.525 vs. M 2.489) los hombres ganan más, "
     "pero en Minería es al revés (H 4.357 vs. M 4.617, las mujeres ganan más). Las líneas del gráfico se cruzan en vez de ser paralelas: esa es la señal visual de interacción. "
     "Con interacción significativa, decir \"las mujeres ganan menos\" sin aclarar la industria es, con estos datos, incorrecto.")

md("""
### A5 · Normalidad de los residuos (10 min)
El ANOVA asume residuos (aproximadamente) normales. Se revisa con el test de Shapiro-Wilk sobre `modelo.resid`.

⚠️ **Importante para todo este notebook**: en los tests de supuestos (normalidad, homoscedasticidad, independencia), $H_0$ = "el supuesto se cumple". Por eso en estos tests **queremos un p-valor grande** (no rechazar), al revés que en el ANOVA o en un t-test de coeficientes, donde buscamos un p-valor chico.

### ✍️ Ejercicio A5
Aplica `stats.shapiro` a los residuos de `modelo_industria` (el ANOVA de una vía de A2). ¿Se cumple el supuesto de normalidad?
""")
code("""
# tu código aquí
""")
solution("""
shapiro_industria = stats.shapiro(modelo_industria.resid)
print(shapiro_industria)
""", "p = 0.931 (bien lejos de 0.05): **no se rechaza la normalidad**, el supuesto se cumple sin problema. "
     "El ANOVA de A2 queda validado: la conclusión de que las industrias difieren no está comprometida por residuos no-normales.")

# ================================================================== PARTE B
md("""
## Parte B · Correlación y regresión simple (`biomasa` y `salarios`)

### B1 · Correlación: Pearson vs. Spearman (10 min)
**Pearson** mide asociación **lineal**. **Spearman** mide asociación **monótona** (basada en rangos, no en los valores) y es más robusto a outliers y a relaciones no exactamente lineales. Si ambas dan resultados parecidos, la relación probablemente es aproximadamente lineal.

### ✍️ Ejercicio B1
Calcula la correlación de Pearson entre `bio` y cada una de las 14 variables numéricas de `biomasa` (todas menos `loc`, `type`, `bio`), ordena por valor absoluto y muestra las 5 más fuertes. Repite con Spearman (`.corr(method="spearman")`).
""")
code("""
# tu código aquí
""")
solution("""
numericas = ["h2s", "sal", "eh7", "ph", "buf", "p", "k", "ca", "mg", "na", "mn", "zn", "cu", "nh4"]

pearson = biomasa[numericas].corrwith(biomasa["bio"]).sort_values(key=abs, ascending=False)
print("Pearson:\\n", pearson.head(5), "\\n")

spearman = biomasa[numericas].corrwith(biomasa["bio"], method="spearman").sort_values(key=abs, ascending=False)
print("Spearman:\\n", spearman.head(5))
""", "Pearson: `ph` 0.774, `ca` 0.643, `buf` -0.734, `nh4` -0.626, `zn` -0.624. "
     "Spearman: `ph` 0.792, `buf` -0.725, `nh4` -0.644, `p` -0.584 (entre las más fuertes). "
     "`ph` es, por lejos, la variable más asociada a la biomasa en ambos métodos, y los valores de Pearson y Spearman son parecidos entre sí: consistente con una relación aproximadamente lineal (la vamos a confirmar en B2).")

md("""
### B2 · Regresión simple `bio ~ ph` (10 min)
Con una sola variable explicativa, el R² de la regresión es exactamente $r^2$ (el cuadrado de la correlación de Pearson). Es una buena forma de verificar que entendiste ambos conceptos.

### ✍️ Ejercicio B2
1. Ajusta `bio ~ ph` con `smf.ols` e imprime el resumen.
2. Verifica a mano que `modelo.rsquared` ≈ `r**2`, con `r` la correlación de Pearson de B1.
3. Interpreta la pendiente. ¿Tiene sentido interpretar el intercepto?
""")
code("""
# tu código aquí
""")
solution("""
modelo_ph = smf.ols("bio ~ ph", biomasa).fit()
print(modelo_ph.summary())

r_ph = biomasa["bio"].corr(biomasa["ph"])
print(f"\\nr = {r_ph:.4f}, r^2 = {r_ph**2:.4f}, R^2 del modelo = {modelo_ph.rsquared:.4f}")
""", "bio = -885.21 + 409.80 · ph, R² = 0.5994, F = 64.33 (p = 4.4×10⁻¹⁰). "
     "r = 0.7742, r² = 0.5994: **coincide exactamente** con el R² del modelo (así tiene que ser en regresión simple). "
     "La pendiente dice que, en este rango de pH observado, cada unidad de pH se asocia a 409.8 unidades más de biomasa. "
     "El intercepto (-885.21, la biomasa \"predicha\" en pH = 0) **no tiene interpretación práctica**: el pH del suelo nunca es 0 en estos datos (el rango observado es bien distinto), así que es una extrapolación sin sentido, no un valor real.")

md("""
### B3 · Diagnóstico de la regresión simple (15 min)
Toda regresión lineal por mínimos cuadrados asume: (1) normalidad de los residuos, (2) independencia de los residuos, (3) homoscedasticidad (varianza constante). Se revisan con:

| Supuesto | Test | $H_0$ | ¿Qué p-valor queremos? |
|---|---|---|---|
| Normalidad | Shapiro-Wilk | residuos normales | **grande** |
| Independencia | Durbin-Watson | sin autocorrelación (estadístico ≈ 2) | no es un p-valor; DW entre 1.5 y 2.5 es razonable |
| Homoscedasticidad | Breusch-Pagan | varianza constante | **grande** |

### ✍️ Ejercicio B3
Para `modelo_ph` (de B2): calcula Shapiro, Durbin-Watson (`sm.stats.durbin_watson` o `statsmodels.stats.stattools.durbin_watson`) y Breusch-Pagan (`statsmodels.stats.diagnostic.het_breuschpagan`, necesita la matriz de diseño `modelo.model.exog`). Además haz un QQ-plot (`sm.qqplot`) y un gráfico de residuos vs. valores ajustados.
""")
code("""
# tu código aquí
""")
solution("""
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

shapiro_ph = stats.shapiro(modelo_ph.resid)
dw_ph = durbin_watson(modelo_ph.resid)
bp_ph = het_breuschpagan(modelo_ph.resid, modelo_ph.model.exog)

print(f"Shapiro: estadístico={shapiro_ph.statistic:.4f}, p={shapiro_ph.pvalue:.4f}")
print(f"Durbin-Watson: {dw_ph:.4f}")
print(f"Breusch-Pagan: LM={bp_ph[0]:.4f}, p={bp_ph[1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sm.qqplot(modelo_ph.resid, line="45", ax=axes[0])
axes[0].set_title("QQ-plot de residuos")
axes[1].scatter(modelo_ph.fittedvalues, modelo_ph.resid)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Valores ajustados")
axes[1].set_ylabel("Residuos")
axes[1].set_title("Residuos vs. ajustados")
plt.tight_layout()
plt.show()
""", "Shapiro: p = 0.032 → **se rechaza la normalidad** al 5% (justo por debajo del umbral). "
     "Durbin-Watson = 0.85, muy por debajo de 2 → **autocorrelación positiva fuerte**: los residuos consecutivos se parecen entre sí. Esto no es raro acá: las filas de `biomasa` están ordenadas por `loc` y `type`, no son observaciones independientes entre sí tomadas al azar. "
     "Breusch-Pagan: p = 0.983 → **no se rechaza homoscedasticidad**, la varianza de los residuos es razonablemente constante. "
     "Resumen: 2 de 3 supuestos presentan problemas (normalidad al límite, independencia claramente violada). El modelo `bio ~ ph` sirve para describir la asociación, pero sus intervalos de confianza y p-valores hay que tomarlos con cautela.")

md("""
### B4 · Puntos influyentes: distancia de Cook (10 min)
Un punto puede tener un residuo chico y, aun así, cambiar mucho el modelo si está en una posición extrema del eje X (alto *leverage*). La **distancia de Cook** combina ambas cosas en un solo número: mientras más grande, más cambia el modelo si se elimina esa observación. No hay un umbral universal, pero valores muy por encima del resto del grupo son candidatos a revisar.

### ✍️ Ejercicio B4
Obtén la distancia de Cook de `modelo_ph` con `modelo_ph.get_influence().cooks_distance[0]` y muestra las 3 observaciones con mayor distancia (con sus columnas `loc`, `type`, `bio`, `ph`).
""")
code("""
# tu código aquí
""")
solution("""
influencia = modelo_ph.get_influence()
cooks = influencia.cooks_distance[0]

top3 = np.argsort(cooks)[::-1][:3]
tabla_influyentes = biomasa.iloc[top3][["loc", "type", "bio", "ph"]].copy()
tabla_influyentes["cooks_d"] = cooks[top3]
print(tabla_influyentes)
""", "Las 3 observaciones más influyentes son todas de tipo `TALL`: fila con `bio`=1660, `ph`=7.45 (Cook's D = 0.138, la más influyente), "
     "y dos filas con `bio`=1400 y `bio`=1276, ambas con `ph`=3.45 (Cook's D ≈ 0.096 y 0.071). "
     "Tiene sentido: `TALL` tiene el rango de pH más amplio de los tres tipos de vegetación (los valores más altos y más bajos del dataset), así que esas filas \"tiran\" más de la recta. Ninguna distancia es dramáticamente alta (no hay un salto brusco tipo 10x respecto a las demás), así que no hay evidencia de un outlier que haya que eliminar, pero vale la pena tenerlas en el radar.")

# ================================================================== PARTE C
md("""
## Parte C · Regresión múltiple

### C1 · Regresión múltiple de salarios (10 min)
Con más de una variable explicativa, cada coeficiente se interpreta **"manteniendo constante lo demás"**: el efecto de esa variable, aislado del resto.

### ✍️ Ejercicio C1
1. Ajusta `Salario ~ Experiencia + C(Sexo) + C(Industria)`.
2. Interpreta el coeficiente de `Experiencia` y el de `C(Sexo)[T.M]` en una frase cada uno.
3. Predice el salario de una mujer en Minería con 10 años de experiencia, a mano con los coeficientes, y verifica con `.predict()`.
""")
code("""
# tu código aquí
""")
solution("""
modelo_multiple = smf.ols("Salario ~ Experiencia + C(Sexo) + C(Industria)", salarios).fit()
print(modelo_multiple.summary())

b = modelo_multiple.params
pred_mano = b["Intercept"] + b["Experiencia"] * 10 + b["C(Sexo)[T.M]"] + b["C(Industria)[T.Minería]"]
pred_predict = modelo_multiple.predict(pd.DataFrame({"Experiencia": [10], "Sexo": ["M"], "Industria": ["Minería"]}))
print(f"\\nPredicción a mano: {pred_mano:.4f}")
print(f"Predicción con .predict(): {pred_predict.iloc[0]:.4f}")
""", "Intercept = 2.004 (hombre, Banca, 0 años de experiencia), Experiencia = 0.125 (p = 0.0001): **manteniendo fijos sexo e industria, cada año extra de experiencia suma 0.125 al salario**. "
     "`C(Sexo)[T.M]` = -0.417 (p = 0.020): **manteniendo fijos experiencia e industria, una mujer gana en promedio 0.417 menos que un hombre**. "
     "Industria: Minería +0.625 (p = 0.013, significativo), Retail -0.322 (p = 0.133, no significativo) respecto a Banca. R² ajustado = 0.687. "
     "Predicción para mujer, Minería, 10 años: 2.004 + 1.251 - 0.417 + 0.625 = **3.463** (la cuenta a mano y `.predict()` coinciden, como debe ser). "
     "⚠️ Nota que este coeficiente de sexo es el efecto *promedio, controlando por industria*, y en A4 viste que el efecto real de ser mujer **depende de la industria** (interacción): este modelo sin interacción es una simplificación.")

md("""
### C2 · La trampa de las variables dummy (10 min)
`C(Industria)` en la fórmula crea automáticamente dummies y deja una fuera (la de referencia). Si en cambio creas **todas** las dummies con `pd.get_dummies()` y las metes junto con el intercepto, el modelo queda **sobre-parametrizado**: la suma de todas las dummies es siempre 1, exactamente igual a la columna del intercepto, así que las columnas son linealmente dependientes (la matriz de diseño no tiene rango completo).

### ✍️ Ejercicio C2
1. Crea `pd.get_dummies(salarios["Industria"])` (las 3 dummies, sin `drop_first`) y agrégale una columna de 1's (intercepto) con `sm.add_constant`.
2. Calcula `np.linalg.matrix_rank(...)` de esa matriz y compárala con su número de columnas.
3. Repite con `pd.get_dummies(salarios["Industria"], drop_first=True)` + intercepto. ¿Cambia el rango?
""")
code("""
# tu código aquí
""")
solution("""
dummies_todas = pd.get_dummies(salarios["Industria"])
X_todas = sm.add_constant(dummies_todas).astype(float)
print("Todas las dummies + intercepto:", X_todas.shape, "-> rango =", np.linalg.matrix_rank(X_todas.values))

dummies_drop = pd.get_dummies(salarios["Industria"], drop_first=True)
X_drop = sm.add_constant(dummies_drop).astype(float)
print("drop_first + intercepto:      ", X_drop.shape, "-> rango =", np.linalg.matrix_rank(X_drop.values))
""", "Con las 3 dummies + intercepto: la matriz tiene **4 columnas pero rango 3** (deficiente en 1). Es la trampa de las dummies: esas 4 columnas no son toda la información independiente que parecen ser, "
     "porque `Banca + Minería + Retail = 1` para toda fila, lo mismo que la columna de intercepto. Un modelo así es irresoluble de forma única (hay infinitas combinaciones de coeficientes que ajustan igual). "
     "Con `drop_first=True`: 3 columnas y rango 3 → **rango completo**, se puede estimar sin ambigüedad. Por eso `C(...)` en la fórmula de statsmodels (y `factor()` en R) dejan una categoría fuera automáticamente: no es una convención arbitraria, es lo que hace que el modelo sea estimable.")

md("""
### C3 · R² vs. R² ajustado y VIF (10 min)
El **R²** nunca baja al agregar una variable, aunque sea pura basura (ruido sin relación real con `y`): matemáticamente, mínimos cuadrados siempre puede usar esa variable para reducir algo la suma de cuadrados residual. El **R² ajustado** penaliza por cada parámetro agregado, y por eso puede **bajar** si la variable nueva no aporta lo suficiente para justificar el costo.

El **VIF** (factor de inflación de varianza) mide **multicolinealidad**: cuánto se infla la varianza del coeficiente de una variable por estar correlacionada con las demás predictoras. VIF > 10 (o ya sobre 5, según el criterio) es señal de alerta.

### ✍️ Ejercicio C3
1. Ajusta `bio ~ ph + buf` y luego `bio ~ ph + buf + sal`. Compara R² y R² ajustado entre ambos.
2. Calcula el VIF de `ph`, `buf` y `sal` en el segundo modelo con `variance_inflation_factor` (de `statsmodels.stats.outliers_influence`), y la correlación entre `ph` y `buf`.
""")
code("""
# tu código aquí
""")
solution("""
from statsmodels.stats.outliers_influence import variance_inflation_factor

modelo_2v = smf.ols("bio ~ ph + buf", biomasa).fit()
modelo_3v = smf.ols("bio ~ ph + buf + sal", biomasa).fit()
print(f"ph + buf:       R2={modelo_2v.rsquared:.4f}  R2_adj={modelo_2v.rsquared_adj:.4f}")
print(f"ph + buf + sal: R2={modelo_3v.rsquared:.4f}  R2_adj={modelo_3v.rsquared_adj:.4f}")

X = sm.add_constant(biomasa[["ph", "buf", "sal"]]).astype(float)
for i, col in enumerate(X.columns):
    print(f"VIF {col}: {variance_inflation_factor(X.values, i):.3f}")

print(f"\\ncorr(ph, buf) = {biomasa['ph'].corr(biomasa['buf']):.4f}")
""", "`ph + buf`: R² = 0.5994, R² ajustado = 0.5803. `ph + buf + sal`: R² = 0.6037 (**sube** un poco, como siempre que agregas una variable), pero R² ajustado = 0.5747 (**baja**). "
     "Ese es exactamente el punto: `sal` casi no mejora el ajuste real, solo infla el R² por construcción; el ajustado lo delata. "
     "VIF: `ph` = 9.96, `buf` = 9.94 (**ambos altos**, cerca del umbral de 10), `sal` = 1.04 (bajo, no colineal). "
     "corr(ph, buf) = -0.946: `ph` y `buf` están **fuertemente correlacionados entre sí**, y eso es justamente lo que produce los VIF altos: el modelo no puede separar bien cuánto del efecto es de `ph` y cuánto de `buf`, porque varían casi siempre juntos (en direcciones opuestas).")

md("""
### C4 · Selección de variables: forward y backward por AIC (15 min)
Con 16 candidatas (14 numéricas + `loc` + `type`) hay $2^{16}$ = 65,536 subconjuntos posibles: probarlos todos no es viable a mano. El algoritmo **forward** parte del modelo nulo y, en cada paso, agrega la variable que más reduce el AIC, hasta que ninguna lo reduce más. El **backward** hace lo opuesto: parte del modelo completo y va *quitando* la variable cuya eliminación más reduce el AIC.

Para que el AIC sea comparable con el que calcula `step()` en R, hay que usar la fórmula de R (no la de statsmodels, que usa otra constante):

$$AIC_R = n \\cdot \\log(RSS/n) + 2 \\cdot (k+1)$$

donde $k$ es el número de parámetros del modelo **incluyendo el intercepto**, y el `+1` extra es por el parámetro de varianza $\\sigma^2$ que R también cuenta.

### ✍️ Ejercicio C4
Completa la función `forward_aic` (el esqueleto de abajo) y ejecútala sobre `biomasa` con `respuesta="bio"`. Después (en la solución) compara con el resultado de hacer el proceso *backward* desde el modelo completo.
""")
code("""
candidatas_numericas = ["h2s", "sal", "eh7", "ph", "buf", "p", "k", "ca", "mg", "na", "mn", "zn", "cu", "nh4"]
candidatas_categoricas = ["loc", "type"]


def aic_r(formula, datos):
    \"\"\"AIC al estilo de R para un modelo OLS ajustado con `formula`.\"\"\"
    modelo = smf.ols(formula, datos).fit()
    n = len(datos)
    rss = np.sum(modelo.resid ** 2)
    k = modelo.df_model + 1  # parámetros, incluyendo el intercepto
    aic = n * np.log(rss / n) + 2 * (k + 1)  # +1 por sigma, como hace R
    return aic, modelo


def forward_aic(datos, respuesta, candidatas_num, candidatas_cat):
    \"\"\"
    Selección hacia adelante por AIC (replica step(..., direction='forward') de R).

    Completa:
    1. Arranca del modelo nulo (`respuesta ~ 1`) y su AIC.
    2. En cada paso, para cada variable candidata que todavía no está en el modelo,
       prueba agregarla (usa `C(variable)` si es categórica) y calcula su AIC con `aic_r`.
    3. Si la mejor variable nueva reduce el AIC actual, agrégala al modelo y repite.
    4. Si ninguna variable reduce el AIC, detente.
    5. Devuelve (modelo_final, lista_de_variables_en_orden_de_entrada).
    \"\"\"
    # tu código aquí
    pass


# una vez completada la función, descomenta estas líneas para probarla:
# modelo_forward, orden_forward = forward_aic(biomasa, "bio", candidatas_numericas, candidatas_categoricas)
# print("Variables seleccionadas, en orden:", orden_forward)
# print(modelo_forward.summary())
""")
solution("""
def forward_aic(datos, respuesta, candidatas_num, candidatas_cat):
    todas = candidatas_num + candidatas_cat
    terminos_en_modelo = []
    restantes = todas.copy()

    formula_actual = f"{respuesta} ~ 1"
    aic_actual, _ = aic_r(formula_actual, datos)

    orden = []
    while True:
        mejor_aic = aic_actual
        mejor_variable = None
        mejor_formula = None
        for variable in restantes:
            termino = f"C({variable})" if variable in candidatas_cat else variable
            if terminos_en_modelo:
                formula_candidata = formula_actual + " + " + termino
            else:
                formula_candidata = f"{respuesta} ~ " + termino
            aic_candidato, _ = aic_r(formula_candidata, datos)
            if aic_candidato < mejor_aic:
                mejor_aic = aic_candidato
                mejor_variable = variable
                mejor_formula = formula_candidata
        if mejor_variable is None:
            break
        terminos_en_modelo.append(mejor_variable)
        formula_actual = mejor_formula
        aic_actual = mejor_aic
        restantes.remove(mejor_variable)
        orden.append(mejor_variable)

    modelo_final = smf.ols(formula_actual, datos).fit()
    return modelo_final, orden


modelo_forward, orden_forward = forward_aic(biomasa, "bio", candidatas_numericas, candidatas_categoricas)
print("Forward, orden de entrada:", orden_forward)
print(f"R2={modelo_forward.rsquared:.4f}  R2_adj={modelo_forward.rsquared_adj:.4f}  AIC(statsmodels)={modelo_forward.aic:.2f}")


def backward_aic(datos, respuesta, candidatas_num, candidatas_cat):
    todas = candidatas_num + candidatas_cat

    def construir_formula(terminos):
        if not terminos:
            return f"{respuesta} ~ 1"
        piezas = [f"C({t})" if t in candidatas_cat else t for t in terminos]
        return f"{respuesta} ~ " + " + ".join(piezas)

    terminos_en_modelo = todas.copy()
    formula_actual = construir_formula(terminos_en_modelo)
    aic_actual, _ = aic_r(formula_actual, datos)
    eliminadas = []

    while True:
        mejor_aic = aic_actual
        peor_variable = None
        mejor_formula = formula_actual
        for variable in terminos_en_modelo:
            candidatos = [t for t in terminos_en_modelo if t != variable]
            formula_candidata = construir_formula(candidatos)
            aic_candidato, _ = aic_r(formula_candidata, datos)
            if aic_candidato < mejor_aic:
                mejor_aic = aic_candidato
                peor_variable = variable
                mejor_formula = formula_candidata
        if peor_variable is None:
            break
        terminos_en_modelo.remove(peor_variable)
        formula_actual = mejor_formula
        aic_actual = mejor_aic
        eliminadas.append(peor_variable)

    modelo_final = smf.ols(formula_actual, datos).fit()
    return modelo_final, terminos_en_modelo, eliminadas


modelo_backward, variables_backward, orden_eliminadas = backward_aic(biomasa, "bio", candidatas_numericas, candidatas_categoricas)
print("\\nBackward, variables eliminadas en orden:", orden_eliminadas)
print("Backward, variables que quedaron:", variables_backward)
print(f"R2={modelo_backward.rsquared:.4f}  R2_adj={modelo_backward.rsquared_adj:.4f}  AIC(statsmodels)={modelo_backward.aic:.2f}")
""", "**Forward** entra en este orden: `ph`, `type`, `buf`, `cu`, `mn`, `mg` → R² = 0.849, R² ajustado = 0.820, AIC (statsmodels) = 642.07. "
     "**Backward** (desde el modelo completo) elimina en orden `p`, `eh7`, `ph`, `na`, `sal`, `zn`, `buf`, `h2s`, `k`, y termina con `loc`, `type`, `ca`, `mg`, `mn`, `cu`, `nh4` → R² = 0.870, R² ajustado = 0.836, AIC = 639.31. "
     "**Forward y backward llegan a modelos distintos**: forward se queda con `ph` y `buf` (y nunca llega a ver `loc`, `ca` ni `nh4`, porque una vez que entró `ph` esas variables ya no mejoraban lo suficiente el AIC), mientras que backward hace el camino inverso y termina sin `ph` ni `buf` pero con `loc`, `ca` y `nh4`. "
     "Esto no es un error de programación: es una propiedad conocida de la selección por pasos (*stepwise*): el camino que se recorre (agregar de a una vs. quitar de a una) determina a qué óptimo local se llega, y no hay garantía de que ambos coincidan ni de que alguno sea el mejor subconjunto posible entre los 65,536.")

md("""
### C5 · Diagnóstico del modelo seleccionado (10 min)
Un R² alto no sirve de nada si los supuestos del modelo no se cumplen: los p-valores y los intervalos de confianza de C4 dependen de ellos.

### ✍️ Ejercicio C5
Aplica a `modelo_forward` (de C4) los mismos tres diagnósticos de B3: Shapiro, Durbin-Watson y Breusch-Pagan. ¿El modelo es confiable?
""")
code("""
# tu código aquí
""")
solution("""
shapiro_fwd = stats.shapiro(modelo_forward.resid)
dw_fwd = durbin_watson(modelo_forward.resid)
bp_fwd = het_breuschpagan(modelo_forward.resid, modelo_forward.model.exog)

print(f"Shapiro:       p = {shapiro_fwd.pvalue:.4f}")
print(f"Durbin-Watson: {dw_fwd:.4f}")
print(f"Breusch-Pagan: p = {bp_fwd[1]:.4f}")
""", "Shapiro: p = 0.138 (**normalidad OK**). Durbin-Watson = 1.57 (en el límite razonable, pero igual que en B3 recuerda que las filas de `biomasa` están ordenadas por `loc`/`type`, no son independientes al azar: en la clase en R, el `dwtest` de la profesora rechazó formalmente la independencia para un modelo de esta familia). "
     "Breusch-Pagan: p = 0.088 (**homoscedasticidad OK**, aunque no con mucho margen). "
     "**Conclusión igual que en clase**: normalidad y homoscedasticidad se sostienen razonablemente, pero la independencia queda en duda. Un R² ajustado de 0.82 es alto, pero con residuos autocorrelacionados los errores estándar (y por lo tanto los p-valores de cada coeficiente) no son del todo confiables: el modelo describe bien los datos que tiene, pero hay que ser cauteloso extrapolando su inferencia.")

# ================================================================== PARTE D
md("""
## Parte D · Predecir con datos nuevos (`bio_nuevo`)

### D1 · Predecir con el modelo forward (5 min)
Ya tenemos un modelo entrenado (`modelo_forward`, de C4) y datos nuevos sin `bio` (`bio_nuevo`). El siguiente paso natural sería predecir con `.predict()`.

### ✍️ Ejercicio D1
Predice `bio` para las 9 filas de `bio_nuevo` usando `modelo_forward`. Compara el rango de las predicciones con el rango real de `bio` en `biomasa` (`biomasa["bio"].describe()`).
""")
code("""
# tu código aquí
""")
solution("""
pred_forward = modelo_forward.predict(bio_nuevo)
print(pred_forward)

print(f"\\nRango de predicciones: [{pred_forward.min():.0f}, {pred_forward.max():.0f}]")
print(f"Rango real de bio en biomasa: [{biomasa['bio'].min():.0f}, {biomasa['bio'].max():.0f}]")
""", "Las predicciones van de **~36,513 a ~108,064**. La biomasa real en el dataset de entrenamiento va de 200 a 2,500 aproximadamente: las predicciones son **de 15 a 40 veces más grandes que cualquier biomasa jamás observada**. "
     "Algo está mal, y no es el modelo (que en C5 tenía diagnósticos razonables): hay que revisar los datos nuevos antes de confiar en una sola predicción. Eso es exactamente lo que hacemos en D2.")

md("""
### D2 · Detectar el error en los datos nuevos (10 min)
Cuando una predicción es absurda, el primer sospechoso no es el modelo: es si los datos nuevos están en la **misma escala** que los datos de entrenamiento. `modelo_forward` usa, entre otras, `mn` y `cu`: justo las variables que más pesan en una predicción tan desproporcionada.

### ✍️ Ejercicio D2
1. Compara `biomasa[["mn", "zn", "cu"]].describe()` contra `bio_nuevo[["mn", "zn", "cu"]].describe()`.
2. Para las primeras 6 filas, calcula la razón fila a fila `bio_nuevo[col] / biomasa[col]` para esas tres columnas. ¿Es el mismo factor en todas las filas?
""")
code("""
# tu código aquí
""")
solution("""
print("biomasa:\\n", biomasa[["mn", "zn", "cu"]].describe().round(2), "\\n")
print("bio_nuevo:\\n", bio_nuevo[["mn", "zn", "cu"]].describe().round(2), "\\n")

razon = (bio_nuevo[["mn", "zn", "cu"]].iloc[:6] / biomasa[["mn", "zn", "cu"]].iloc[:6].values).round(2)
print("Razón bio_nuevo / biomasa (primeras 6 filas):\\n", razon)
""", "`biomasa`: `mn` va de ~7 a ~89, `zn` de ~0.2 a ~31, `cu` de ~1.7 a ~6.1. `bio_nuevo`: `mn` va de ~77 a ~644, `zn` de ~9.9 a ~178, `cu` de ~175 a ~511: **uno o dos órdenes de magnitud más grandes**, eso solo ya es una señal de alarma. "
     "La razón fila a fila lo confirma: en las primeras 3 filas `mn` y `zn` están exactamente **×10** y `cu` está exactamente **×100** (razones 10.07, 10.06, 100.00 — un **punto decimal corrido**). "
     "Pero la fila 4 (`mn` razón 1.99) y la fila 5 (`mn` razón 8.43) **no siguen el mismo factor**: el error no es un ×10 o ×100 uniforme que se pueda simplemente dividir de vuelta, es **inconsistente fila por fila**. "
     "**Esto es un error en el archivo `bio_new.csv`** (probablemente del punto decimal al capturar los datos), no algo que tu código deba \"corregir\" con una fórmula: reportarías el problema y pedirías el archivo corregido antes de confiar en cualquier predicción que use `mn`, `zn` o `cu` de este archivo.")

md("""
### D3 · Predecir con un modelo que sí sea confiable (10 min)
Mientras no se corrija `bio_new.csv`, la jugada correcta es predecir con un modelo que **no dependa** de las columnas corrompidas. `modelo_ph_type` (`bio ~ ph + C(type)`, de C3 y parecido) solo usa `ph` y `type`, que no muestran el mismo problema de escala.

### ✍️ Ejercicio D3
1. Ajusta (o reusa) `bio ~ ph + C(type)` sobre `biomasa` y predice sobre `bio_nuevo`. Verifica que las predicciones caigan dentro del rango observado de `bio`.
2. Calcula los intervalos de predicción al 95% con `modelo.get_prediction(bio_nuevo).summary_frame(alpha=0.05)`.
""")
code("""
# tu código aquí
""")
solution("""
modelo_ph_type = smf.ols("bio ~ ph + C(type)", biomasa).fit()

pred_confiable = modelo_ph_type.predict(bio_nuevo)
print("Predicciones (bio ~ ph + C(type)):")
print(pred_confiable.round(1))
print(f"\\nRango de predicciones: [{pred_confiable.min():.0f}, {pred_confiable.max():.0f}]")
print(f"Rango real de bio en biomasa: [{biomasa['bio'].min():.0f}, {biomasa['bio'].max():.0f}]")

intervalos = modelo_ph_type.get_prediction(bio_nuevo).summary_frame(alpha=0.05)
print("\\nIntervalos de predicción al 95%:")
print(intervalos.round(1))
""", "Las predicciones van de ~529 a ~1,249: **todas dentro del rango observado de `bio` (200-2,500 aprox.)**, a diferencia del desastre de D1. `R2 = 0.765, R2_adj = 0.748` para este modelo (de C3), que además solo usa `ph` y `type`, libres del problema de escala detectado en D2. "
     "Los intervalos de predicción del 95% (`obs_ci_lower`/`obs_ci_upper`) son bastante anchos (de ±400 a ±700 unidades), reflejando que una sola fila nueva tiene bastante incertidumbre; para dos filas (tipo `TALL`) el límite inferior del intervalo es **negativo**, lo que no tiene sentido físico (`bio` no puede ser negativo) — es una limitación conocida de los intervalos de un modelo lineal, no un error de cálculo: el modelo no sabe que `bio` tiene que ser positivo. "
     "**Conclusión de la Parte D**: no se debe predecir con `modelo_forward` sobre `bio_nuevo` tal como está. Lo correcto es (1) reportar el error de las columnas `mn`/`zn`/`cu` para que se corrija el archivo, y (2) mientras tanto, usar un modelo que no dependa de esas columnas, como `bio ~ ph + C(type)`, dejando claro que es una solución provisional y con menos variables explicativas que el modelo forward completo.")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (15 min, SIN mirar lo anterior)

Con `salarios`, ajusta y compara estos 4 modelos anidados:
1. `Salario ~ Experiencia`
2. `Salario ~ Experiencia + C(Sexo)`
3. `Salario ~ Experiencia + C(Sexo) + C(Industria)`
4. `Salario ~ Experiencia + C(Sexo) * C(Industria)` (agrega la interacción)

1. Para cada uno, calcula R² ajustado y AIC.
2. Elige el mejor según esos dos criterios y corre sus diagnósticos (Shapiro, Durbin-Watson, Breusch-Pagan).
3. Escribe una conclusión de **3 líneas**: ¿cuál modelo elegirías y por qué? ¿todos los coeficientes de ese modelo son individualmente significativos? ¿qué le dirías a alguien que te pregunta "¿importa el sexo para el salario en esta empresa?"
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
m1 = smf.ols("Salario ~ Experiencia", salarios).fit()
m2 = smf.ols("Salario ~ Experiencia + C(Sexo)", salarios).fit()
m3 = smf.ols("Salario ~ Experiencia + C(Sexo) + C(Industria)", salarios).fit()
m4 = smf.ols("Salario ~ Experiencia + C(Sexo) * C(Industria)", salarios).fit()

comparacion = pd.DataFrame({
    "modelo": ["Experiencia", "+Sexo", "+Industria", "+Industria:Sexo"],
    "R2_adj": [m1.rsquared_adj, m2.rsquared_adj, m3.rsquared_adj, m4.rsquared_adj],
    "AIC": [m1.aic, m2.aic, m3.aic, m4.aic],
})
print(comparacion.round(4))

print("\\n", m4.summary())

print(f"\\nShapiro:       p = {stats.shapiro(m4.resid).pvalue:.4f}")
print(f"Durbin-Watson: {durbin_watson(m4.resid):.4f}")
print(f"Breusch-Pagan: p = {het_breuschpagan(m4.resid, m4.model.exog)[1]:.4f}")
""", "R² ajustado: 0.523 → 0.572 → 0.687 → **0.730** (sube en cada paso). AIC: 82.54 → 79.16 → 68.38 → **64.17** (baja en cada paso, también mejora en cada paso). "
     "**El modelo con interacción (modelo 4) es el mejor por ambos criterios.** Sin embargo, al mirar sus p-valores individuales, varios coeficientes de la interacción no son significativos por separado (p. ej. `Sexo[M]:Industria[Minería]` con p ≈ 0.065, apenas sobre el umbral): con solo 40 observaciones repartidas en 6 combinaciones de industria×sexo, cada coeficiente de interacción tiene poca información para estimarse con precisión, aunque el modelo completo ajuste mejor en conjunto. "
     "Diagnósticos del modelo 4: Shapiro p = 0.535 (normalidad OK), Durbin-Watson = 1.75 (cercano a 2, razonable), Breusch-Pagan p = 0.817 (homoscedasticidad OK): los tres supuestos se sostienen bien. "
     "**Conclusión esperada (3 líneas)**: El modelo con interacción Industria×Sexo es el que mejor explica el salario según R² ajustado y AIC, y sus residuos cumplen los supuestos de normalidad, independencia y homoscedasticidad razonablemente bien. "
     "Aun así, con solo 40 observaciones varios coeficientes de la interacción no alcanzan significancia individual, así que hay que ser prudente al generalizar el tamaño exacto de cada diferencia. "
     "A la pregunta \"¿importa el sexo?\": sí, pero **no de forma uniforme** — el ANOVA de dos vías (Parte A) y este modelo coinciden en que el efecto del sexo sobre el salario depende de la industria (favorece a los hombres en Banca y Retail, pero a las mujeres en Minería), así que no existe una única cifra de \"brecha salarial por sexo\" que aplique a toda la empresa por igual.")

# ---------------------------------------------------------------- self-eval
md("""
## Autoevaluación (responde sin mirar)

1. En un ANOVA, ¿qué diferencia hay entre el p-valor del test F global y el p-valor de un test de Tukey? ¿Por qué no basta con hacer t-tests de a pares en vez de Tukey?
2. En los tests de supuestos de una regresión (normalidad, homoscedasticidad), ¿qué es $H_0$, y por qué en estos tests se busca un p-valor **grande** y no chico?
3. ¿Por qué el R² nunca baja al agregar una variable a un modelo, aunque sea ruido puro, y qué soluciona el R² ajustado?
4. Si `forward` y `backward` llegan a modelos de variables distintos sobre los mismos datos, ¿cuál de los dos es "el correcto"?

<details><summary><b>Respuestas</b></summary>

1. El F global solo dice que **al menos un par** de medias difiere, sin decir cuál. Tukey compara **todos los pares** y corrige el umbral de significancia para que la probabilidad de al menos un falso positivo en el conjunto de comparaciones se mantenga controlada (al 5%, no más). Hacer t-tests de a pares sin esa corrección infla el error tipo I: con más comparaciones, aumenta la chance de encontrar "diferencias" que son solo ruido.
2. En estos tests, $H_0$ = "el supuesto se cumple" (residuos normales, varianza constante, etc.). Rechazar $H_0$ significa tener evidencia de que el supuesto **no** se cumple, así que un p-valor grande (no rechazar) es la noticia buena: el supuesto parece sostenerse con los datos que tienes. Es al revés que en un ANOVA o en el test de un coeficiente, donde un p-valor chico es lo que buscas.
3. Mínimos cuadrados elige los coeficientes que minimizan la suma de cuadrados residual; agregar una variable más nunca puede empeorar ese mínimo (en el peor caso, su coeficiente sale ≈0 y el ajuste queda igual). El R² ajustado resta una penalización por cada parámetro agregado, así que si la variable nueva no reduce lo suficiente el error como para justificar ese costo, el R² ajustado **baja**, delatando que no vale la pena incluirla.
4. **Ninguno de los dos es "el correcto" por definición.** Ambos son heurísticas *greedy* (miran un paso a la vez) y pueden quedar atrapados en óptimos locales distintos; ninguno garantiza encontrar el mejor subconjunto entre todos los posibles. Lo razonable es compararlos con criterios adicionales (R² ajustado, AIC, validación cruzada, diagnósticos de residuos, sentido de negocio de las variables) en vez de asumir que uno de los dos es automáticamente superior.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Carpeta 2 · Modelos Lineales Generalizados (`C20-21_GLM.ipynb`). Revisa `Guias/apuntes/02_Ejercicios_Cuaderno_Regresion.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT)
