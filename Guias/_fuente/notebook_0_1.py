"""Genera el notebook guiado de la sesión 0.1 (Describir los datos)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "0.1_Describir_los_datos.ipynb"
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
# Sesión 0.1 · Describir los datos
**Etapa 0 · Nivelación estadística** · martes 6 de octubre · ~95 min · guía: `Guias/01_Etapa0_Estadistica.pdf` (página 2)

### Objetivos de hoy
1. Clasificar cada variable según su tipo y saber qué análisis permite.
2. Describir el centro y la dispersión de una variable, y explicar cuándo usar media o mediana.
3. Leer un boxplot y detectar outliers con la regla de 1.5·IQR.

### Plan de la sesión
| Min | Bloque |
|---|---|
| 5 | Preparación y carga de datos |
| 10 | 1 · Tipos de variables |
| 5 | 2 · Datos faltantes |
| 15 | 3 · Centro: media y mediana |
| 15 | 4 · Dispersión |
| 15 | 5 · Boxplot y outliers |
| 10 | 6 · Comparar grupos |
| 15 | 7 · Ejercicio de cierre **sin ayuda** |
| 5 | Autoevaluación y notas |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código o la respuesta.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla: equivocarse primero es lo que fija el aprendizaje.
""")

# ---------------------------------------------------------------- setup
md("## Preparación")
code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Busca la carpeta del diplomado subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase 17*")))
carpeta = next(raiz.glob("Clase 17*"))

ens = pd.read_csv(carpeta / "ENS.csv")
salarios = pd.read_excel(carpeta / "salarios.xlsx")

print("ENS:", ens.shape, "| salarios:", salarios.shape)
ens.head()
""")
md("""
**ENS.csv**: 350 personas de una encuesta de salud.
- `COLESTEROL`: colesterol total en mg/dL.
- `EDAD`: grupo etario.
- `FUMA` y `DEPORTE`: Sí o No.

**salarios.xlsx**: 40 personas, con `Salario`, `Industria`, `Sexo` y años de `Experiencia`. Se usa en las Clases 17 a 19.
""")

# ---------------------------------------------------------------- 1
md("""
## 1 · Tipos de variables (10 min)

El tipo de variable **decide qué puedes calcular**. No tiene sentido sacar el promedio de "Industria", y en un gráfico de barras no se aprecia la forma del colesterol.

| Tipo | Subtipo | Ejemplo | Qué se calcula |
|---|---|---|---|
| **Numérica** | continua | colesterol, salario | media, mediana, desviación estándar, histograma |
| | discreta | n° de hijos | lo mismo, pero solo toma valores enteros |
| **Categórica** | nominal | industria, fuma | frecuencias, %, gráfico de barras |
| | ordinal | nivel educacional | frecuencias, mediana; tiene orden pero no distancias |

⚠️ pandas solo distingue números (`float`, `int`) y texto (`object` o `str`, según la versión). **El tipo estadístico lo decides tú**, no `dtypes`. Una región codificada como 1, 2 o 3 es categórica aunque pandas la lea como `int`. Ese es exactamente el error que advierte la profesora en la Clase 17.
""")
code("""
print(ens.dtypes, "\\n")
print(salarios.dtypes)
""")
md("""
### ✍️ Ejercicio 1
Clasifica las 8 columnas (4 de ENS y 4 de salarios) en numérica continua o discreta, o categórica nominal u ordinal. Completa el diccionario.
""")
code("""
tipos = {
    "COLESTEROL": "",
    "EDAD": "",
    "FUMA": "",
    "DEPORTE": "",
    "Salario": "",
    "Industria": "",
    "Sexo": "",
    "Experiencia": "",
}
""")
solution("""
tipos = {
    "COLESTEROL": "numérica continua",
    "EDAD": "categórica ordinal",      # Joven < Adulto < Adulto Mayor
    "FUMA": "categórica nominal",      # binaria (Sí/No)
    "DEPORTE": "categórica nominal",   # binaria
    "Salario": "numérica continua",
    "Industria": "categórica nominal",
    "Sexo": "categórica nominal",      # H / M
    "Experiencia": "numérica continua",
}
""", "EDAD tiene orden natural, por eso es **ordinal**. Industria no tiene orden: Banca no es «mayor» que Retail.")

# ---------------------------------------------------------------- 2
md("""
## 2 · Datos faltantes (5 min)

Antes de calcular cualquier cosa, **cuenta los vacíos**. Si no lo haces, pandas los ignora en silencio y puedes sacar conclusiones sobre un grupo distinto al que crees.
""")
code("""
ens.isna().sum()
""")
md("""
### ✍️ Ejercicio 2
1. ¿Qué **porcentaje** de las filas tiene `EDAD` vacía?
2. ¿Te preocupa ese porcentaje? Responde en una línea.
""")
code("""
# tu código aquí
""")
solution("""
pct = ens["EDAD"].isna().mean() * 100
print(f"{pct:.1f}% de EDAD está vacía")   # 1.1%
""", "Con 4 de 350 (1.1%), excluirlas casi no cambia nada. Si fuera el 30%, habría que investigar **por qué** faltan antes de seguir.")

# ---------------------------------------------------------------- 3
md("""
## 3 · Centro: media y mediana (15 min)

- **Media**: suma todo y divide por n. Usa todos los valores, así que **un valor extremo la arrastra**.
- **Mediana**: el valor del medio al ordenar. **No le importan los extremos**.

Regla práctica:
- Si media ≈ mediana, la distribución es más o menos simétrica.
- Si la media es mucho mayor que la mediana, hay una cola de valores altos (asimetría a la derecha).
""")
code("""
col = ens["COLESTEROL"]
print(f"Media:   {col.mean():.1f}")
print(f"Mediana: {col.median():.1f}")
""")
md("""
### ✍️ Ejercicio 3 · predice antes de ejecutar
El salario promedio en `salarios` es 3.55. Supón que agregamos a una persona con salario **50**, por ejemplo un gerente general.

**Antes de ejecutar**, escribe tu predicción:
- ¿La media sube mucho, poco o nada?
- ¿Y la mediana?

Después ejecuta la celda y compara.
""")
code("""
# Mi predicción: media → ...   mediana → ...

s = salarios["Salario"]
s_con_gerente = pd.concat([s, pd.Series([50])])

print(f"Media:   {s.mean():.2f} → {s_con_gerente.mean():.2f}")
print(f"Mediana: {s.median():.2f} → {s_con_gerente.median():.2f}")
""")
md("""
<details><summary><b>Ver explicación</b></summary>

La media salta de 3.55 a 4.68: **una sola persona** la sube un 32%. La mediana apenas pasa de 3.62 a 3.65.

**Lección**: con ingresos, precios o tiempos de espera, que suelen tener valores extremos, la mediana describe mejor el caso típico. Por eso las estadísticas oficiales de ingresos suelen reportar la mediana.
</details>
""")

# ---------------------------------------------------------------- 4
md("""
## 4 · Dispersión (15 min)

Dos grupos pueden tener la misma media y ser muy distintos. La dispersión mide **cuánto se alejan los datos del centro**.

| Medida | Fórmula | Interpretación |
|---|---|---|
| Varianza | $s^2 = \\frac{\\sum (x - \\bar{x})^2}{n - 1}$ | promedio de distancias al cuadrado (en unidades²) |
| Desviación estándar | $s = \\sqrt{s^2}$ | distancia típica a la media, **en las unidades originales** |
| IQR | $Q_3 - Q_1$ | ancho del 50% central; robusto a extremos |

¿Por qué se divide por **n − 1** y no por n? Con una muestra, la media $\\bar{x}$ queda "pegada" a los datos, y eso subestima la dispersión real. Dividir por n − 1 corrige ese sesgo. Lo verás con más detalle en la sesión 0.3.
""")
md("""
### ✍️ Ejercicio 4 · a mano y con la librería
Con `x = [2, 4, 4, 4, 5, 5, 7, 9]`:
1. Calcula la desviación estándar **paso a paso**: media → desvíos → cuadrados → suma → dividir por n − 1 → raíz.
2. Compárala con `np.std(x)`, `np.std(x, ddof=1)` y `pd.Series(x).std()`. ¿Cuáles coinciden con la tuya y por qué?
""")
code("""
x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
# tu código aquí
""")
solution("""
x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
media = x.mean()                       # 5.0
desvios = x - media
suma_cuadrados = (desvios ** 2).sum()  # 32
s = np.sqrt(suma_cuadrados / (len(x) - 1))
print(f"A mano (n-1):        {s:.4f}")                  # 2.1381
print(f"np.std(x):           {np.std(x):.4f}")          # 2.0000  ← divide por n
print(f"np.std(x, ddof=1):   {np.std(x, ddof=1):.4f}")  # 2.1381
print(f"pd.Series(x).std():  {pd.Series(x).std():.4f}") # 2.1381
""", "**numpy divide por n por defecto; pandas divide por n − 1.** Con 8 datos la diferencia es de 7%. Con muestras, usa siempre n − 1.")

# ---------------------------------------------------------------- 5
md("""
## 5 · Visualizar: histograma y boxplot (15 min)

- **Histograma**: muestra la *forma* completa, es decir, dónde se concentran los datos, si hay asimetría y si hay más de un pico.
- **Boxplot**: resume 5 números (mín, Q1, mediana, Q3 y máx) y marca los **outliers**, que son los valores a más de 1.5·IQR fuera de la caja.
""")
code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist(col, bins=25, color="#475569", edgecolor="white")
ax1.axvline(col.mean(), color="red", label=f"media {col.mean():.1f}")
ax1.axvline(col.median(), color="orange", ls="--", label=f"mediana {col.median():.1f}")
ax1.set_title("Colesterol")
ax1.legend()

orden = ["Joven", "Adulto", "Adulto Mayor"]
ax2.boxplot([ens.loc[ens.EDAD == g, "COLESTEROL"] for g in orden])
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(orden)
ax2.set_title("Colesterol por grupo de edad")
plt.tight_layout()
plt.show()
""")
md("""
### ✍️ Ejercicio 5
Escribe una función `outliers(x)` que devuelva los valores de una Serie que están fuera de $[Q_1 - 1.5 \\cdot IQR,\\; Q_3 + 1.5 \\cdot IQR]$. Luego aplícala al colesterol **dentro de cada grupo de edad**.

¿Cuántos outliers encontraste? ¿Coinciden con los puntos del boxplot?

*Pista*: `x.quantile([0.25, 0.75])` y `ens.groupby("EDAD")["COLESTEROL"].apply(...)`.
""")
code("""
# tu código aquí
""")
solution("""
def outliers(x):
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return x[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)]

ens.groupby("EDAD")["COLESTEROL"].apply(outliers)
""", "Deberían coincidir con los puntos del boxplot. Recuerda: un outlier **no es un error**. Antes de decidir qué hacer con él, hay que entender por qué existe (Clase 18).")

# ---------------------------------------------------------------- 6
md("""
## 6 · Comparar grupos (10 min)

`groupby` + `agg` es la herramienta más usada en análisis descriptivo. Para variables categóricas se usan `value_counts` y `crosstab`.
""")
code("""
print(ens["FUMA"].value_counts(normalize=True).round(3), "\\n")
pd.crosstab(ens["FUMA"], ens["DEPORTE"], margins=True)
""")
md("""
### ✍️ Ejercicio 6
1. Calcula media, mediana, desviación estándar y n del colesterol según `FUMA`, y luego según `DEPORTE`.
2. ¿Cuál de las dos diferencias de medias te parece más grande?
3. **Pregunta clave**: ¿basta con ver que dos medias son distintas para afirmar que la diferencia es real?
""")
code("""
# tu código aquí
""")
solution("""
for grupo in ["FUMA", "DEPORTE"]:
    print(ens.groupby(grupo)["COLESTEROL"].agg(["mean", "median", "std", "count"]).round(1), "\\n")
""", "Fumar: 196.3 vs 190.9 (5 mg/dL de diferencia). Deporte: 177.9 vs 196.4 (18 mg/dL).\n\n"
     "Sobre la pregunta 3: **no basta**. Con muestras, siempre hay diferencias por azar. Para saber si una diferencia es real "
     "necesitas una **prueba de hipótesis** (sesiones 0.5 a 0.7). Spoiler: una de estas dos diferencias es real y la otra no.")

# ---------------------------------------------------------------- 7
md("""
## 7 · ✍️ Ejercicio de cierre (15 min, SIN mirar lo anterior)

Con `salarios`:
1. Clasifica sus 4 variables.
2. Calcula media, mediana, desviación estándar e IQR del salario **por industria**.
3. Haz un boxplot del salario por industria.
4. Escribe una conclusión de **3 líneas**: ¿qué industria paga más? ¿cuál es la más dispersa? ¿hay outliers?
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
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

resumen = salarios.groupby("Industria")["Salario"].agg(["mean", "median", "std", iqr, "count"]).round(2)
print(resumen)

orden = resumen.sort_values("mean").index
plt.boxplot([salarios.loc[salarios.Industria == i, "Salario"] for i in orden])
plt.xticks([1, 2, 3], orden)
plt.title("Salario por industria")
plt.show()
""", "**Conclusión esperada**: Minería paga más (media 4.48) y es la menos dispersa (s = 0.52). Retail paga menos (2.90) y es la más dispersa (s = 0.86). "
     "Media y mediana son parecidas en las tres industrias, así que no hay asimetrías fuertes.")

# ---------------------------------------------------------------- self-eval
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué "región = 1, 2, 3" no debería promediarse?
2. Si en un dataset de precios la media es mucho mayor que la mediana, ¿qué te dice eso?
3. ¿Qué diferencia hay entre `np.std(x)` y `pd.Series(x).std()`?

<details><summary><b>Respuestas</b></summary>

1. Es categórica nominal: los números son etiquetas, no cantidades. «Región 2» no es el doble de «Región 1».
2. Hay una cola de valores altos (asimetría a la derecha). La mediana describe mejor el caso típico.
3. numpy divide por n y pandas por n − 1. Con muestras, usa n − 1.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión (0.2, miércoles 7):** distribuciones y la normal. ¿Se parece el colesterol a una campana?
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT)
