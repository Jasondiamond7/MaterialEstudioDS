"""Genera el notebook guiado de Feature Engineering y selección de variables (Clase 33)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "C33_FeatureEngineering.ipynb"
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
# Clase 33 · Feature Engineering y selección de variables
**Diplomado Data Science** · 2 sesiones de ~95 min · Prof. Eduardo Rodríguez

Material de apoyo:
- `Guias/C33_Guia_FeatureEngineering.pdf` — guía de la clase.
- `Guias/apuntes/C33_Apuntes_FeatureEngineering.pdf` — apunte con pistas conceptuales.
- `Guias/apuntes/C33_Ejercicios_Cuaderno_FeatureEngineering.pdf` — ejercicios en papel (antes o después de este notebook).
- Transcripción de la clase: `VideosClases/Clase 33 - Feature engineering metodologia de seleccion de variables.txt`.

⚠️ **Antes de empezar**: este notebook necesita `scikit-learn`, `openpyxl` (para leer el `.xlsx`) y `seaborn`. Instala en tu venv:
```
pip install scikit-learn openpyxl seaborn
```

### Objetivos de estas sesiones
1. Limpiar un dataset con **criterio de negocio**, no estadístico: distinguir un valor imposible de un outlier legítimo.
2. Construir sistemáticamente candidatas a partir de variables originales (ratios, interacciones, logaritmos, indicadores, cuartiles, one-hot) y entender que **crear variables sin seleccionar después solo las infla**.
3. Distinguir tres motivos por los que una variable puede sobrar — irrelevante, redundante, relevante solo en conjunto — y por qué cada uno pide una decisión distinta.
4. Usar filtros (varianza, información mutua, ANOVA F), wrapper (RFE) y embedded (Lasso), y saber cuándo cada uno falla.
5. Detectar y evitar **data leakage** en la selección de variables: por qué la selección tiene que vivir dentro del pipeline validado por CV.

### Plan de las sesiones
| Sesión | Min | Bloque |
|---|---|---|
| 1 | 5 | Preparación y carga de datos |
| 1 | 10 | A1 · Primer diagnóstico: valores imposibles |
| 1 | 10 | A2 · Cuantificar lo imposible (y los duplicados) |
| 1 | 15 | A3 · Limpieza con criterio de negocio |
| 1 | 5 | A4 · Por qué no se borran filas |
| 1 | 10 | B1 · Ratios: normalizar por exposición |
| 1 | 15 | B2 · Interacciones, logaritmos, indicadores, cuartiles y one-hot |
| 1 | 10 | B3 · Verificación de las 50 candidatas |
| 2 | 8 | C1 · Correlación de las originales con `fuga` |
| 2 | 12 | C2 · Información mutua y ANOVA F |
| 2 | 8 | C3 · Irrelevante sola, útil en conjunto |
| 2 | 8 | C4 · El contraejemplo XOR |
| 2 | 8 | C5 · Redundancia entre candidatas |
| 2 | 5 | D1 · El costo de la búsqueda exhaustiva |
| 2 | 12 | D2 · Wrapper: RFECV dentro de un pipeline |
| 2 | 12 | D3 · Embedded: Lasso y el efecto de `C` |
| 2 | 10 | D4 · Cuatro contra cuatro |
| 2 | 8 | E1 · El experimento del ruido puro |
| 2 | 5 | E2 · Por qué pasa: falsos positivos esperados |
| 2 | 8 | E3 · El pipeline honesto |
| 2 | 8 | E4 · Antes del split o dentro del fold |
| 2 | 15 | Ejercicio de cierre **sin ayuda** |
| 2 | 5 | Autoevaluación y notas |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla: equivocarse primero es lo que fija el aprendizaje.
- Los números que aparecen en el texto (tasas, correlaciones, AUC) están verificados contra la salida real de este notebook. Si a ti te salen ligeramente distintos revisa primero la versión de `scikit-learn` (la información mutua depende de `random_state`, y eso se discute explícitamente en C2).
""")

# ---------------------------------------------------------------- setup
md("## Preparación")
code("""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Busca la carpeta de la clase subiendo desde la ubicación del notebook
raiz = next(p for p in [Path.cwd(), *Path.cwd().parents] if any(p.glob("Clase 33*")))
carpeta = next(raiz.glob("Clase 33*"))

df = pd.read_excel(carpeta / "Base de datos Churn.xlsx")

print("Filas y columnas:", df.shape)
print("Tasa de fuga:", round(df.fuga.mean(), 4))
df.head()
""")
md("""
**Base de datos Churn.xlsx** (n=3.150): 12 variables de una telco más la variable objetivo `fuga` (1 = el cliente se fue).

| Variable | Qué es |
|---|---|
| `numero_fallas` | fallas técnicas reportadas |
| `reclamos` | reclamos formales presentados |
| `antiguedad` | meses como cliente |
| `monto` | monto de facturación |
| `segundos` | segundos totales de llamadas |
| `frec_llamadas` | cantidad de llamadas |
| `frec_sms` | cantidad de SMS |
| `n_contactos` | contactos con servicio al cliente |
| `tipo_plan` | plan contratado (categórica) |
| `estado` | región o estado (categórica) |
| `edad` | edad del cliente |
| `clv` | *customer lifetime value* estimado |
| `fuga` | **target**: 1 si el cliente se fue |
""")

# ================================================================== PARTE A
md("""
## Parte A · Limpieza con criterio de negocio

Ninguna técnica de selección arregla un dato imposible. Si una variable tiene registros corruptos, su media, su varianza y su correlación con el target están contaminadas, y cualquier ranking que construyamos sobre ellas va a estar mal. **El orden importa**: primero se limpia, después se selecciona.

### A1 · Primer diagnóstico: valores imposibles (10 min)
`describe()` entrega los momentos y los cuartiles de cada variable. No se trata todavía de entender el negocio, sino de **detectar valores imposibles**: mínimos negativos donde no puede haberlos, máximos absurdos, medias muy separadas de la mediana (señal de cola larga o de basura).

### ✍️ Ejercicio A1
Muestra `df.describe().T` (redondeado) y `df.isnull().sum()`. Mirando los mínimos y máximos, **lista a mano** qué variables tienen valores que no pueden existir en la realidad (por ejemplo, ¿puede `segundos` ser negativo? ¿puede alguien tener 233 años?).
""")
code("""
# tu código aquí
""")
solution("""
display(df.describe().T.round(2))
display(df.isnull().sum().to_frame("nulos"))
""", "Mínimos negativos (imposibles) en `segundos`, `frec_llamadas`, `frec_sms` y `antiguedad` — nadie puede hablar \"−14.645\" segundos. "
     "`edad` tiene un mínimo negativo y un máximo de 233: ninguno de los dos es una persona real. "
     "Nulos declarados: `reclamos` 13, `antiguedad` 5, `segundos` 40, `frec_llamadas` 19, `frec_sms` 4, `n_contactos` 2 (el resto, 0). "
     "Importante: que una columna tenga 0 nulos declarados **no significa que no falten datos** — los negativos imposibles de `segundos` no son `NaN`, son basura disfrazada de número válido, y recién los vamos a contar en A2.")

md("""
### A2 · Cuantificar lo imposible (y los duplicados) (10 min)
Antes de decidir qué hacer, hay que saber **cuánto** hay de cada problema: cuántos negativos imposibles por variable, cuántos casos de `edad` fuera de rango, y si hay filas repetidas.

### ✍️ Ejercicio A2
1. Para `segundos`, `frec_llamadas`, `frec_sms`, `antiguedad` y `n_contactos`: cuenta cuántos valores negativos tiene cada una y su mínimo.
2. Para `edad`: cuenta negativos, cuántos valores caen entre 0 y 14 años, y cuántos superan los 100.
3. Cuenta filas duplicadas con `df.duplicated().sum()`.
""")
code("""
# tu código aquí
""")
solution("""
print("Valores negativos donde no puede haberlos:")
for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
    n = int((df[c] < 0).sum())
    if n:
        print(f"  {c:15s} {n:3d} casos   mínimo: {df[c].min():.0f}")

print("\\nEdad — desglose de lo que queda fuera del rango [15, 100]:")
print(f"  negativas          : {int((df.edad < 0).sum()):3d}   mínimo: {df.edad.min():.0f}")
print(f"  entre 0 y 14 años  : {int(((df.edad >= 0) & (df.edad < 15)).sum()):3d}")
print(f"  mayores de 100     : {int((df.edad > 100).sum()):3d}   máximo: {df.edad.max():.0f}")

print(f"\\nFilas duplicadas: {int(df.duplicated().sum())}")
""", "`segundos` 18 casos negativos (mínimo −14.645), `frec_llamadas` 12 (mínimo −176), `frec_sms` 1 (mínimo −9), `antiguedad` 9 (mínimo −40), `n_contactos` 0. "
     "`edad`: 1 negativa, 108 entre 0 y 14 años (3,4 % de la cartera), 2 sobre 100 (máximo 233). "
     "19 filas duplicadas. "
     "Nota el matiz en `edad`: los casos **claramente** imposibles son solo 3 (la negativa y las 2 sobre 100). Las 108 entre 0 y 14 años no son imposibles por sí mismas — podrían ser cuentas contratadas a nombre de un menor —, pero sí quedan fuera del rango de negocio `[15, 100]` que vamos a aplicar en A3. Esa es exactamente la clase de decisión que se discute en A4.")

md("""
### A3 · Limpieza con criterio de negocio (15 min)
Dos decisiones, y ninguna de las dos es puramente estadística:

1. **Qué es imposible.** Un negativo en `segundos` o en `antiguedad` no existe: es un error de captura o de carga. Se convierte en `NaN` porque *no sabemos* el valor real — poner un cero sería inventar un dato.
2. **Qué rango es admisible.** El umbral de edad `[15, 100]` no sale de los datos: sale de saber que una telco no contrata a un cliente de 233 años.

Después se imputa por **mediana** y no por media, porque la mediana es robusta a las colas que viste en A1-A2.

### ✍️ Ejercicio A3
1. Antes de tocar nada, calcula la media y la mediana de `segundos` tal como está (negativos incluidos como si fueran números válidos).
2. Convierte a `NaN` los negativos de `segundos`, `frec_llamadas`, `frec_sms`, `antiguedad`, `n_contactos`.
3. Convierte a `NaN` los `edad` fuera de `[15, 100]`.
4. Vuelve a calcular media y mediana de `segundos` (ya sin los negativos, pero todavía sin imputar). ¿Cuánto cambió cada una? ¿Sigue habiendo una diferencia grande entre media y mediana, y a qué la atribuyes ahora?
5. Imputa cada columna con `NaN` por su propia mediana.
""")
code("""
# tu código aquí
""")
solution("""
# 1. Media y mediana ANTES de limpiar (negativos todavía como números "válidos")
media_antes = df["segundos"].mean()
mediana_antes = df["segundos"].median()

# 2. Un negativo acá es imposible por definición
for c in ["segundos", "frec_llamadas", "frec_sms", "antiguedad", "n_contactos"]:
    df.loc[df[c] < 0, c] = np.nan

# 3. Rango admisible por criterio de negocio
df.loc[(df.edad < 15) | (df.edad > 100), "edad"] = np.nan

# 4. Media y mediana DESPUÉS de sacar los negativos (todavía sin imputar)
print(f"Antes  (negativos incluidos) -> media: {media_antes:.1f}  mediana: {mediana_antes:.1f}")
print(f"Después (negativos -> NaN)   -> media: {df['segundos'].mean():.1f}  mediana: {df['segundos'].median():.1f}")

# 5. Imputación por mediana (robusta a las colas)
for c in df.columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

print("\\nDespués de limpiar:", df.shape, "| tasa de fuga:", round(df.fuga.mean(), 4))
print("Nulos restantes:", int(df.isna().sum().sum()))
""", "Antes: media 4.453,9, mediana 2.996,5. Después de sacar los 18 negativos: media 4.506,4, mediana 3.031,5. "
     "El cambio es chico en los dos casos (apenas 1-2 %): con solo 18 valores imposibles entre más de 3.100 válidos, ninguno de los dos estadísticos se mueve dramáticamente al sacarlos — así que **no** es cierto que \"la media se descontrola y la mediana no\" en este caso puntual. "
     "Lo que sí persiste, antes y después, es que la media queda **muy por encima** de la mediana (4.506 vs. 3.031, casi 50 % más alta): eso ya no es por los negativos (que se fueron), es la cola larga natural de `segundos` que viste en los box plots de A1 — muchas llamadas cortas y algunas pocas muy largas (hasta 17.090). "
     "Esa es la razón de fondo, más general que el caso de los negativos, para imputar con mediana: en una variable con cola larga la mediana sigue representando \"al cliente típico\", mientras que la media queda arrastrada hacia los usuarios de uso extremo. "
     "Tasa de fuga después de limpiar: 0,1571 (495 de 3.150) — no cambia respecto a la original porque la limpieza no tocó ninguna fila, solo valores puntuales. Nulos restantes: 0.")

md("""
### A4 · Por qué no se borran filas (5 min)
Una alternativa más simple a imputar sería `df.dropna()` y `df.drop_duplicates()`. El profesor advierte explícitamente contra esto en casos como el de `edad`: las 108 filas con 0-14 años no son necesariamente basura, podrían ser un segmento real (cuentas de menores), y **borrarlas en lugar de imputarlas destruye información que quizás el negocio sí necesita**. Lo mismo aplica, en menor medida, a las filas duplicadas: 19 filas iguales en un dataset de 3.150 podrían ser clientes legítimamente idénticos en estas 13 columnas, no necesariamente un error de carga.

La regla general: **borrar filas es la opción más cómoda para el analista y la más riesgosa para el negocio.** Imputar (o, mejor aún, preguntarle al dueño del proceso qué significa el dato) conserva el tamaño de muestra y obliga a documentar la decisión en vez de esconderla.

✍️ **Tu respuesta** (sin código, una o dos frases): si en A2 hubieras encontrado que las 108 filas de 0-14 años tienen, además, `tipo_plan` siempre igual a un valor específico, ¿cambiaría tu decisión de imputarlas en vez de borrarlas? ¿Por qué?

-
""")

# ================================================================== PARTE B
md("""
## Parte B · Construcción de variables: de 12 a 50

Con 12 columnas limpias no hay mucho que seleccionar. El problema real de selección aparece cuando hay 50, 200 o 2.000 candidatas y hay que **justificar cada descarte**. Esta parte aplica, de forma sistemática, seis familias de transformaciones sobre las 12 variables originales:

| Familia | Cuántas | Para qué sirve |
|---|---|---|
| Originales que se conservan | 10 | Punto de partida |
| Ratios | 13 | Normalizar por exposición |
| Interacciones | 3 | Efectos que solo existen combinados |
| Logaritmos | 5 | Comprimir colas largas |
| Indicadores binarios | 5 | Capturar umbrales |
| Discretización en cuartiles | 12 | Capturar no linealidad |
| One-hot | 2 | Codificar categóricas |
| **Total** | **50** | |

Son 12 originales, pero `tipo_plan` y `estado` desaparecen al reemplazarlas por sus indicadores one-hot; por eso quedan 50 y no 52.

### B1 · Ratios: normalizar por exposición (10 min)
Un cliente con 5.000 segundos de llamadas puede ser alguien que hace muchas llamadas cortas o pocas llamadas largas: el total no distingue esos dos comportamientos, el ratio sí.

$$\\text{duración promedio} = \\frac{\\text{segundos}}{\\text{frec\\_llamadas} + \\varepsilon}$$

El $\\varepsilon = 10^{-6}$ evita la división por cero: un solo infinito rompe todo el pipeline que viene después.

### ✍️ Ejercicio B1
Sobre `d = df.copy()`, construye estos 3 ratios con el patrón `d[nombre] = d[numerador] / (d[denominador] + e)`:
1. `llamadas_por_mes` = `frec_llamadas` / `antiguedad`
2. `contactos_por_mes` = `n_contactos` / `antiguedad`
3. `clv_por_antiguedad` = `clv` / `antiguedad`

(En la solución vas a ver las 13 que construye el profesor — son la misma idea repetida.)
""")
code("""
e = 1e-6
d = df.copy()

# tu código aquí (los 3 ratios de arriba)
""")
solution("""
e = 1e-6
d = df.copy()

pares = [
  ("duracion_prom_llamada", "segundos",      "frec_llamadas"),
  ("sms_por_llamada",       "frec_sms",      "frec_llamadas"),
  ("intensidad_contacto",   "n_contactos",   "frec_llamadas"),
  ("segundos_por_contacto", "segundos",      "n_contactos"),
  ("monto_por_llamada",     "monto",         "frec_llamadas"),
  ("clv_por_llamada",       "clv",           "frec_llamadas"),
  ("clv_por_segundo",       "clv",           "segundos"),
  ("sms_por_antiguedad",    "frec_sms",      "antiguedad"),
  ("clv_por_antiguedad",    "clv",           "antiguedad"),
  ("llamadas_por_mes",      "frec_llamadas", "antiguedad"),
  ("contactos_por_mes",     "n_contactos",   "antiguedad"),
  ("fallas_por_antiguedad", "numero_fallas", "antiguedad"),
  ("antiguedad_rel_edad",   "antiguedad",    "edad"),
]
for nom, a, b in pares:
    d[nom] = d[a] / (d[b] + e)

print("Columnas después de los ratios:", d.shape[1])
""", "13 ratios nuevos, uno por cada par de la lista `pares`. Fíjate que todos siguen el mismo patrón (`numerador / (denominador + e)`): una vez que entiendes la idea, el código es mecánico. "
     "`d.shape[1]` pasa de 13 (las 12 originales + `fuga`) a 26.")

md("""
### B2 · Interacciones, logaritmos, indicadores, cuartiles y one-hot (15 min)
- **Interacciones** ($X_j \\cdot X_k$): permiten que el efecto de una variable dependa del valor de otra. Sin ellas, un modelo lineal impone que el efecto de `numero_fallas` sea el mismo para quien reclamó y para quien no.
- **Logaritmos** ($\\log(1+x)$, no $\\log(x)$, porque hay ceros): comprimen colas largas. La diferencia entre 10 y 100 pasa a pesar lo mismo que entre 100 y 1.000.
- **Indicadores y cuartiles**: un modelo lineal asume que el efecto de una variable es constante en todo su rango. Discretizar rompe ese supuesto: cada tramo recibe su propio coeficiente.

### ✍️ Ejercicio B2
Sobre el mismo `d` de B1, agrega:
1. 3 interacciones: `reclamo_x_fallas` = `reclamos * numero_fallas`, `reclamo_x_antig` = `reclamos * antiguedad`, `fallas_x_contactos` = `numero_fallas * n_contactos`.
2. 5 logaritmos `log1p` sobre `segundos`, `clv`, `frec_sms`, `frec_llamadas`, `n_contactos` (usa `.clip(lower=0)` antes del log, por si quedó algún residuo numérico negativo).
3. 5 indicadores binarios: `tiene_reclamo` (`reclamos > 0`), `sin_sms` (`frec_sms == 0`), `uso_alto` (`segundos` > su mediana), `clv_alto` (`clv` > su percentil 75), `cliente_nuevo` (`antiguedad < 12`).
4. Cuartiles (dummies) de `edad`, `antiguedad` y `clv` con `pd.qcut(..., 4, labels=False, duplicates="drop")`.
5. One-hot de `tipo_plan` (`plan_2` = `tipo_plan == 2`) y `estado` (`estado_2` = `estado == 2`), y elimina las columnas originales `tipo_plan` y `estado`.

Al final, separa `X = d.drop(columns="fuga")` y `y = d["fuga"]`.
""")
code("""
# tu código aquí
""")
solution("""
# --- 2. INTERACCIONES: efectos que solo existen combinados -----------
d["reclamo_x_fallas"]   = d.reclamos * d.numero_fallas
d["reclamo_x_antig"]    = d.reclamos * d.antiguedad
d["fallas_x_contactos"] = d.numero_fallas * d.n_contactos

# --- 3. LOGARITMOS: comprimen colas largas ---------------------------
for c in ["segundos", "clv", "frec_sms", "frec_llamadas", "n_contactos"]:
    d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))

# --- 4. INDICADORES BINARIOS: capturan umbrales ----------------------
med = d.segundos.median()
p75 = d.clv.quantile(0.75)
d["tiene_reclamo"] = (d.reclamos > 0).astype(int)
d["sin_sms"]       = (d.frec_sms == 0).astype(int)
d["uso_alto"]      = (d.segundos > med).astype(int)
d["clv_alto"]      = (d.clv > p75).astype(int)
d["cliente_nuevo"] = (d.antiguedad < 12).astype(int)

# --- 5. DISCRETIZACIÓN EN CUARTILES: captura no linealidad -----------
for col in ["edad", "antiguedad", "clv"]:
    q = pd.qcut(d[col], 4, labels=False, duplicates="drop")
    for k in sorted(q.dropna().unique()):
        d[f"{col}_q{int(k)+1}"] = (q == k).astype(int)

# --- 6. ONE-HOT: las categóricas originales se van -------------------
d["plan_2"]   = (d.tipo_plan == 2).astype(int)
d["estado_2"] = (d.estado == 2).astype(int)
d = d.drop(columns=["tipo_plan", "estado"])

X = d.drop(columns="fuga")
y = d["fuga"]
print("Candidatas:", X.shape[1], "  observaciones:", X.shape[0])
""", "50 candidatas, 3.150 observaciones: exactamente la tabla de la introducción de la Parte B. "
     "Nota algo que vas a explotar en C1 y C5: varias de estas variables son **la misma información con otra forma** por construcción — `tiene_reclamo` es casi lo mismo que `reclamos`, `log_segundos` es una transformación monótona de `segundos`. Eso no es un error: es el costo esperado de construir candidatas de forma sistemática, y es justamente lo que los filtros de redundancia de la Parte C están para detectar.")

md("""
### B3 · Verificación de las 50 candidatas (10 min)
Antes de seguir, conviene confirmar que la construcción no introdujo basura numérica (`inf` por una división que `e` no alcanzó a evitar, o `NaN` por un cuartil mal definido).

### ✍️ Ejercicio B3
1. Confirma que `X.shape[1] == 50`.
2. Cuenta cuántos `inf`/`-inf` y cuántos `NaN` hay en `X` (pista: `np.isinf(X.select_dtypes(include=np.number)).sum().sum()`).
3. Verifica a mano la afirmación del cierre de B2: calcula la correlación entre `reclamos` y `tiene_reclamo`. ¿Es realmente redundancia casi perfecta?
""")
code("""
# tu código aquí
""")
solution("""
print("Columnas en X:", X.shape[1], "-> ¿son 50?", X.shape[1] == 50)

n_inf = int(np.isinf(X.select_dtypes(include=np.number)).sum().sum())
n_nan = int(X.isna().sum().sum())
print(f"Valores infinitos: {n_inf}  |  Valores NaN: {n_nan}")

print(f"\\ncorr(reclamos, tiene_reclamo) = {X['reclamos'].corr(X['tiene_reclamo']):.3f}")
""", "50 columnas, 0 infinitos y 0 NaN: la construcción quedó limpia (el `e = 1e-6` hizo su trabajo en los ratios). "
     "`corr(reclamos, tiene_reclamo) = 1.000`: es redundancia perfecta, no solo \"casi\" — `tiene_reclamo` es literalmente una versión binarizada de `reclamos`, así que llevan exactamente la misma información de orden. Guarda este par en la cabeza para C5.")

# ================================================================== PARTE C
md("""
## Parte C · Filtros y redundancia

Una variable puede sobrar por tres motivos distintos, y **cada uno pide una decisión distinta**:

| Tipo | Qué pasa | Qué hacer |
|---|---|---|
| **Irrelevante** | No aporta información sobre el target, ni sola ni acompañada | Se descarta |
| **Redundante** | Aporta, pero su información ya está en otra variable | Se conserva una de las dos |
| **Relevante solo en conjunto** | Sola parece ruido; combinada con otra, aporta | **Los filtros univariados la matan** |

### C1 · Correlación de las originales con `fuga` (8 min)
$$r_{XY} = \\frac{\\operatorname{Cov}(X,Y)}{\\sigma_X \\, \\sigma_Y}$$
Mide solo relación **lineal**: si $Y$ depende de $X$ en forma de U, la correlación puede dar cero aunque la dependencia sea perfecta (volvemos a esto en C4).

### ✍️ Ejercicio C1
Calcula la correlación de Pearson de cada una de las 12 variables originales (incluyendo `plan_2` y `estado_2`, que son el resultado del one-hot) con `fuga`, y ordénalas por valor absoluto.
""")
code("""
originales = ["numero_fallas", "reclamos", "antiguedad", "monto", "segundos",
              "frec_llamadas", "frec_sms", "n_contactos", "edad", "clv",
              "plan_2", "estado_2"]

# tu código aquí
""")
solution("""
corr_target = d[originales + ["fuga"]].corr()["fuga"].drop("fuga")
display(corr_target.sort_values(key=abs, ascending=False).to_frame("corr con fuga").round(3))
""", "De mayor a menor en valor absoluto: `reclamos` 0,511 (por lejos la más fuerte), `estado_2` 0,319, `frec_llamadas` −0,296, `segundos` −0,294, `clv` −0,289, `n_contactos` −0,279, `frec_sms` −0,221, `monto` −0,202, `plan_2` −0,106, `antiguedad` −0,032, `edad` −0,019 y `numero_fallas` −0,009 (las tres últimas, prácticamente nulas). "
     "Guarda la sospecha sobre `numero_fallas`, `edad` y `antiguedad`: parecen irrelevantes por este ranking, pero todavía no probamos si lo son **en conjunto** (C3) ni con una medida que capture dependencias no lineales (C2).")

md("""
### C2 · Información mutua y ANOVA F (12 min)
La **información mutua** $I(X;Y)$ mide cuánta incertidumbre sobre $Y$ desaparece al conocer $X$. A diferencia de Pearson, **detecta cualquier tipo de dependencia**, no solo lineal. Pero para variables continuas se estima con vecinos más cercanos, lo que introduce aleatoriedad: **sin fijar `random_state` el valor cambia en cada corrida**.

El **ANOVA F** compara la dispersión *entre* los dos grupos de `fuga` contra la dispersión *dentro* de cada grupo — es, otra vez, una comparación de **medias**, con el mismo punto ciego que la correlación ante relaciones no lineales.

### ✍️ Ejercicio C2
1. Calcula `mutual_info_classif(X, y, random_state=0)` y `f_classif(X, y)` para las 50 candidatas. Junta ambos en una tabla y muestra las 10 variables con mayor información mutua.
2. Repite la información mutua con `random_state` de 0 a 4 y mira cómo cambia el top-4. ¿El orden es estable?
""")
code("""
from sklearn.feature_selection import mutual_info_classif, f_classif

# tu código aquí
""")
solution("""
mi = pd.Series(mutual_info_classif(X, y, random_state=0), index=X.columns)
F, p_val = f_classif(X, y)
rank = pd.DataFrame({"MI": mi, "F": F, "p": p_val})
display(rank.sort_values("MI", ascending=False).head(10).round(4))

print("\\nTop-4 según la semilla de mutual_info_classif:\\n")
for s in range(5):
    mi_s = pd.Series(mutual_info_classif(X, y, random_state=s), index=X.columns)
    print(f"  seed {s} -> {mi_s.sort_values(ascending=False).head(4).index.tolist()}")
""", "Top por información mutua: `segundos`, `log_segundos`, `contactos_por_mes`, `log_clv` ocupan los primeros lugares, con valores entre 0,10 y 0,13 (el valor exacto cambia un poco con la semilla, como se ve abajo). "
     "Y al cambiar `random_state` de 0 a 4, el **top-4 cambia de composición** entre corridas — no es un error, es la varianza esperada de un estimador por vecinos más cercanos sobre 3.150 filas. "
     "Dos variables por sospechar están, otra vez, con información mutua baja: confirmamos en C3 si eso significa que no sirven.")

md("""
### C3 · Irrelevante sola, útil en conjunto (8 min)
En C1 y C2, `numero_fallas` quedó con correlación e información mutua prácticamente nulas. Antes de descartarla, probemos lo correcto: medir cuánto aporta **acompañada**, no sola. Un filtro univariado ordena por $s(X_j, Y)$ e ignora el resto; existen variables donde $s(X_j, Y) \\approx 0$ pero $s((X_j, X_k), Y)$ es grande.

### ✍️ Ejercicio C3
Define una función `auc(cols)` que entrene `StandardScaler + LogisticRegression` dentro de un `Pipeline` y devuelva el AUC promedio de `cross_val_score` con `StratifiedKFold(5, shuffle=True, random_state=1)`. Compara:
1. `numero_fallas` sola.
2. Una base de 5 variables: `reclamos`, `segundos`, `frec_llamadas`, `estado_2`, `clv`.
3. La base + `numero_fallas`.
4. La base + `numero_fallas` + `monto`.
""")
code("""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(5, shuffle=True, random_state=1)

def auc(cols):
    # tu código aquí
    pass

# una vez completada auc(), descomenta:
# base = ["reclamos", "segundos", "frec_llamadas", "estado_2", "clv"]
# print(f"Solo numero_fallas            : AUC {auc(['numero_fallas']):.4f}")
# print(f"Base de 5 variables           : AUC {auc(base):.4f}")
# print(f"Base + numero_fallas          : AUC {auc(base + ['numero_fallas']):.4f}")
# print(f"Base + numero_fallas + monto  : AUC {auc(base + ['numero_fallas', 'monto']):.4f}")
""")
solution("""
def auc(cols):
    modelo = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=3000))])
    return cross_val_score(modelo, X[cols], y, cv=cv, scoring="roc_auc").mean()

base = ["reclamos", "segundos", "frec_llamadas", "estado_2", "clv"]

print(f"Solo numero_fallas            : AUC {auc(['numero_fallas']):.4f}")
print(f"Base de 5 variables           : AUC {auc(base):.4f}")
print(f"Base + numero_fallas          : AUC {auc(base + ['numero_fallas']):.4f}")
print(f"Base + numero_fallas + monto  : AUC {auc(base + ['numero_fallas', 'monto']):.4f}")
""", "Sola, `numero_fallas` da AUC ≈ 0,526 (prácticamente azar, consistente con su correlación e información mutua casi nulas). "
     "La base de 5 variables ya da AUC ≈ 0,907. Agregando `numero_fallas` sube a ≈ 0,910, y agregando además `monto` sube a ≈ 0,917: **mejora el conjunto aunque sola no sirva para nada**. "
     "La conclusión que hay que llevarse: un ranking univariado bajo no significa «esta variable no sirve», significa **«esta variable no sirve sola»**. Es el argumento central a favor de los métodos multivariados (Parte D).")

md("""
### C4 · El contraejemplo XOR (8 min)
El caso extremo de C3: dos variables **independientes del target por separado** que juntas lo predicen perfecto.

### ✍️ Ejercicio C4
1. Genera `x1`, `x2` normales independientes (`rng = np.random.default_rng(7)`, 5.000 filas) y un target `t = (x1 > 0) XOR (x2 > 0)`.
2. Calcula el ANOVA F univariado de `x1` y `x2` contra `t` con `f_classif`. ¿Qué dice el filtro?
3. Compara el AUC (CV) de una logística con `[x1, x2]` contra una logística con `[x1, x2, x1*x2]` (agregando la interacción).
4. Opcional: compara también con un `RandomForestClassifier(n_estimators=200, random_state=0)` sobre `[x1, x2]` sin la interacción explícita.
""")
code("""
from sklearn.feature_selection import f_classif

rng = np.random.default_rng(7)
x1 = rng.normal(size=5000)
x2 = rng.normal(size=5000)
t  = ((x1 > 0) ^ (x2 > 0)).astype(int)

# tu código aquí
""")
solution("""
Z  = np.c_[x1, x2]
Zi = np.c_[x1, x2, x1 * x2]

F_xor, p_xor = f_classif(Z, t)
print("Lo que ve un filtro univariado (ANOVA F):")
print(f"  x1 -> F = {F_xor[0]:6.3f},  p = {p_xor[0]:.3f}")
print(f"  x2 -> F = {F_xor[1]:6.3f},  p = {p_xor[1]:.3f}")

for nombre, datos in [("Logística con x1 y x2", Z),
                      ("Logística + interacción x1*x2", Zi)]:
    a = cross_val_score(LogisticRegression(max_iter=2000), datos, t,
                        cv=5, scoring="roc_auc").mean()
    print(f"  {nombre:32s} AUC {a:.3f}")

from sklearn.ensemble import RandomForestClassifier
a_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0),
                       Z, t, cv=5, scoring="roc_auc").mean()
print(f"  {'Random Forest (200 árboles)':32s} AUC {a_rf:.3f}")
""", "El ANOVA F da valores débiles para ambas: `x1` con F=1,212 y p=0,271 (lejos de ser significativa) y `x2` con F=4,312 y p=0,038 (apenas por debajo de 0,05, pero con un F minúsculo). Ninguna de las dos muestra algo parecido a la señal perfecta que sabemos que existe: el filtro univariado, en el mejor de los casos, encuentra una pista débil e inconsistente entre semillas, nunca la estructura real. "
     "La logística con `x1, x2` sin más da AUC ≈ 0,51 (no ve nada); con el término de interacción agregado, AUC = 1,000 — la señal siempre estuvo ahí, faltaba una representación que la dejara visible. "
     "El Random Forest sobre `[x1, x2]`, sin interacción explícita, también llega a AUC = 1,000: promediando muchos árboles con distintos cortes termina encontrando la estructura que un solo corte lineal no puede ver. "
     "Moraleja para la Parte B: si sospechás que dos variables actúan en conjunto, la forma más segura de que un modelo lineal lo vea es escribir el producto explícitamente — por eso construimos interacciones.")

md("""
### C5 · Redundancia entre candidatas (8 min)
La matriz de correlación es simétrica y tiene 1 en la diagonal: dos de cada tres celdas sobran. El truco para recorrer cada par **una sola vez** es quedarse con el triángulo superior (`np.triu(..., k=1)`).

### ✍️ Ejercicio C5
1. Calcula la matriz de correlación absoluta de las 50 candidatas (`X.corr().abs()`), quédate con el triángulo superior y cuenta cuántos pares superan 0,90.
2. Para cada par por encima de 0,90, aplica esta regla de desempate: **se queda la variable con mayor información mutua** (la `mi` de C2) — es la más simple de justificar ante un comité, porque no exige explicar dos variables casi idénticas.
""")
code("""
# tu código aquí
""")
solution("""
corr = X.corr().abs()
mascara = np.triu(np.ones(corr.shape), k=1).astype(bool)
sup = corr.where(mascara)
pares = sup.stack().sort_values(ascending=False)

print(f"Pares con correlación sobre 0,90: {int((pares > 0.90).sum())}\\n")
display(pares.head(10).to_frame("|correlación|").round(3))

print("\\nDesempate por información mutua:\\n")
for a, b in pares[pares > 0.90].index:
    gana = a if mi[a] >= mi[b] else b
    pierde = b if gana == a else a
    print(f"  {a:24s} vs {b:24s} -> queda {gana}")
""", "11 pares con correlación sobre 0,90. Los dos más extremos son redundancia perfecta por construcción: `reclamos` vs. `tiene_reclamo` (1,000, el par de B3) y `clv_alto` vs. `clv_q4` (1,000, un indicador binario que coincide exactamente con el cuartil superior). "
     "Detectar el par **no es decidir**: el desempate (quedarse con la de mayor información mutua) es un criterio del analista, no una verdad matemática — otro criterio razonable sería quedarse con la variable continua en vez de la binaria, porque conserva más información. Lo importante es aplicar **un** criterio explícito y documentarlo, no improvisar par por par.")

# ================================================================== PARTE D
md("""
## Parte D · Wrapper y embedded

### D1 · El costo de la búsqueda exhaustiva (5 min)
Con $p$ variables, probar todos los subconjuntos posibles cuesta $2^p - 1$ modelos.

### ✍️ Ejercicio D1
Con `p = X.shape[1]` (50), calcula $2^p - 1$ y cuántos años tomaría entrenar esa cantidad de modelos a razón de 1.000 modelos por segundo.
""")
code("""
# tu código aquí
""")
solution("""
p = X.shape[1]
subconjuntos = 2**p - 1
segundos_totales = subconjuntos / 1000
print(f"Variables: {p}")
print(f"Subconjuntos posibles: {subconjuntos:,}")
print(f"A 1.000 modelos/segundo: {segundos_totales/60/60/24/365:,.0f} años")
""", "$2^{50}-1 \\approx 1,13 \\times 10^{15}$ subconjuntos, unos **35.700 años** a 1.000 modelos por segundo. Ninguna búsqueda exhaustiva es viable con 50 candidatas: todos los métodos prácticos (RFE, Lasso) son heurísticas que recorren solo una fracción diminuta de ese espacio.")

md("""
### D2 · Wrapper: RFECV dentro de un pipeline (12 min)
*Recursive Feature Elimination con validación cruzada*: entrena, ordena por $|\\beta_j|$, elimina la más débil, repite — todo **dentro de cada fold**. RFE ordena por tamaño de coeficiente, así que si una variable como `clv` está en una escala muy distinta de un indicador 0/1, su coeficiente va a ser chico por **unidad de medida**, no por importancia real. Por eso el escalador va **dentro** del pipeline que le pasas a `RFECV`, nunca aplicado antes sobre todo `X`.

### ✍️ Ejercicio D2
Ajusta un `RFECV` con `estimator = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])`, `step=1`, `cv=5`, `scoring="roc_auc"`, `min_features_to_select=4`, y `importance_getter="named_steps.clf.coef_"`. ¿Cuántas variables selecciona?
""")
code("""
from sklearn.feature_selection import RFECV

# tu código aquí
""")
solution("""
estimador = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000)),
])

rfe = RFECV(
    estimator = estimador,
    step = 1,
    cv = 5,
    scoring = "roc_auc",
    min_features_to_select = 4,
    importance_getter = "named_steps.clf.coef_",
).fit(X, y)

print(f"Variables elegidas por RFECV: {rfe.n_features_}")
print(list(X.columns[rfe.support_]))
""", "RFECV elige 42 de las 50 candidatas: elimina muy pocas. Con un dataset donde la mayoría de las variables aportan algo (aunque sea redundante) y el `scoring` es AUC con validación cruzada, el algoritmo no encuentra mucha ganancia en seguir podando más allá de ese punto. "
     "Esto no significa que las 42 sean todas necesarias — significa que, para este criterio (maximizar AUC promedio de CV), la penalización por tener más variables no existe explícitamente: RFECV optimiza desempeño, no parsimonia. Si quisieras un modelo más chico para explicar a un comité, necesitas otro criterio (como Lasso en D3, donde `C` controla directamente cuántas sobreviven).")

md("""
### D3 · Embedded: Lasso y el efecto de `C` (12 min)
Lasso penaliza la suma de valores absolutos de los coeficientes ($\\|\\beta\\|_1$) y por eso produce **ceros exactos** (a diferencia de Ridge, que achica pero casi nunca anula). En la parametrización de `scikit-learn`, $C$ es el **inverso** de la fuerza de penalización: $C$ chico → penalización relativa fuerte → menos variables sobreviven. `solver="liblinear"` es obligatorio con `penalty="l1"` (el solver por defecto no la soporta).

### ✍️ Ejercicio D3
1. Ajusta `Pipeline([StandardScaler, LogisticRegression(penalty="l1", solver="liblinear", C=0.05, random_state=0)])` sobre `X, y`. ¿Cuántos coeficientes quedan distintos de cero?
2. Repite para `C` en `[0.01, 0.05, 0.1, 0.5, 1.0]` y grafica (o solo imprime) cuántas variables sobreviven en cada caso.
""")
code("""
# tu código aquí
""")
solution("""
lasso = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                               C=0.05, max_iter=3000, random_state=0)),
]).fit(X, y)

coef = pd.Series(lasso[-1].coef_[0], index=X.columns)
sel_lasso = coef[coef != 0].index.tolist()
print(f"{len(sel_lasso)} de {len(coef)} variables sobreviven con C=0.05\\n")

print(f"AUC con las 50 candidatas completas : {auc(list(X.columns)):.4f}")
print(f"AUC con las {len(sel_lasso)} de Lasso (C=0.05)         : {auc(sel_lasso):.4f}")

resultados = {}
for Cval in [0.01, 0.05, 0.1, 0.5, 1.0]:
    m = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                             C=Cval, max_iter=3000,
                                             random_state=0))]).fit(X, y)
    resultados[Cval] = int((m[-1].coef_[0] != 0).sum())
    print(f"  C = {Cval:<5} -> {resultados[Cval]:2d} variables vivas")

plt.figure(figsize=(5, 3.5))
plt.plot(list(resultados.keys()), list(resultados.values()), marker="o")
plt.xlabel("C (inverso de la penalización)")
plt.ylabel("Variables con coeficiente != 0")
plt.title("Efecto de C en Lasso")
plt.tight_layout()
plt.show()
""", "Con `C=0.05`, sobreviven 27 de 50, y el AUC con esas 27 (0,9396) es prácticamente idéntico al AUC con las 50 candidatas completas (0,9406): perder 23 variables cuesta apenas una milésima de AUC. La curva completa de `C`: `0.01` → 13, `0.05` → 27, `0.1` → 33, `0.5` → 42, `1.0` → 45 variables vivas. "
     "El patrón es monótono y confirma la teoría: a medida que `C` crece (penalización relativa más débil), sobreviven más variables, acercándose al total de 50 sin nunca necesariamente llegarlo. "
     "No existe un `C` \"correcto\" universal — depende de cuántas variables puede sostener el negocio en producción. `C=0.05` (27 variables, casi el mismo AUC que las 50) es un punto intermedio razonable para este ejercicio, no una regla.")

md("""
### D4 · Cuatro contra cuatro (10 min)
Comparemos dos subconjuntos del **mismo tamaño**: las 4 variables con mayor información mutua (de C2) contra 4 elegidas a mano evitando redundancia (`reclamos`, `segundos`, `estado_2`, `numero_fallas`).

### ✍️ Ejercicio D4
1. Calcula el AUC (con la función `auc()` de C3) y la correlación media entre pares (`corr_media`, función a escribir: promedio de `|corr|` en el triángulo superior de esas 4 columnas) para ambos conjuntos.
2. Repite el experimento para `k` en `[4, 6, 8, 10, 12]`, comparando el top-k por información mutua contra una selección golosa que penaliza redundancia (mRMR simplificado: en cada paso agrega la variable con mayor `mi - máxima_correlación_con_lo_ya_elegido`). ¿La ventaja de evitar redundancia se mantiene al crecer `k`?
""")
code("""
# tu código aquí
""")
solution("""
top4   = mi.sort_values(ascending=False).head(4).index.tolist()
manual = ["reclamos", "segundos", "estado_2", "numero_fallas"]

def corr_media(cols):
    m = X[cols].corr().abs().values
    return m[np.triu_indices(len(cols), 1)].mean()

print(f"Top-4 por información mutua : AUC {auc(top4):.4f}   | correlación media: {corr_media(top4):.2f}")
print(f"  {top4}")
print(f"4 elegidas sin redundancia  : AUC {auc(manual):.4f}   | correlación media: {corr_media(manual):.2f}")
print(f"  {manual}")

def sin_redundancia(k):
    sel = [mi.idxmax()]
    while len(sel) < k:
        resto = [c for c in X.columns if c not in sel]
        redun = X[resto].corrwith(X[sel[0]]).abs()
        for s_ in sel[1:]:
            redun = np.maximum(redun, X[resto].corrwith(X[s_]).abs())
        sel.append((mi[resto] - redun).idxmax())
    return sel

print(f"\\n{'k':>3} {'top-k por MI':>14} {'evitando redundancia':>22} {'ventaja':>10}")
for k in [4, 6, 8, 10, 12]:
    t = mi.sort_values(ascending=False).head(k).index.tolist()
    m = sin_redundancia(k)
    print(f"{k:>3} {auc(t):>14.4f} {auc(m):>22.4f} {auc(m)-auc(t):>+10.4f}")
""", "Top-4 por información mutua: AUC ≈ 0,804, correlación media ≈ 0,55 (gasta 2 de sus 4 cupos entre `segundos` y `log_segundos`, que son casi la misma variable, y no incluye ningún `reclamos`). "
     "4 a mano: AUC ≈ 0,893, correlación media ≈ 0,22 — **casi 9 puntos de AUC de diferencia con la misma cantidad de variables.** "
     "Pero la tabla por `k` creciente muestra que **la ventaja se invierte**: con `k=4` evitar redundancia gana claramente, con `k=6` el ranking univariado ya iguala o supera a la selección golosa. La lección no es \"los filtros univariados son malos\": es que el costo de ignorar la redundancia es **inversamente proporcional al presupuesto de variables**. Con presupuesto grande, el ranking incorpora solo igual a las variables fuertes; con presupuesto chico (el caso típico de un modelo que va a un comité), la redundancia duele mucho.")

# ================================================================== PARTE E
md("""
## Parte E · La trampa del leakage

Todo lo que seleccionamos en las Partes C y D se hizo mirando las 3.150 filas completas, y después se midió el desempeño sobre esas mismas filas. **Todos los AUC calculados hasta ahora están, en algún grado, inflados.** Esta parte cuantifica cuánto, con un caso donde conocemos la respuesta de antemano.

### E1 · El experimento del ruido puro (8 min)
100 observaciones, 1.000 variables generadas al azar, target lanzado a cara o sello. Por construcción **no hay ninguna señal**: cualquier medición honesta tiene que dar AUC ≈ 0,50.

### ✍️ Ejercicio E1
1. Genera `Xr` (100×1000, normal) y `yr` (100, binario al azar) con `rng = np.random.default_rng(42)`.
2. **Mal**: selecciona las 20 mejores con `SelectKBest(f_classif, k=20)` sobre **todo** `Xr, yr`, y luego mide el AUC de una logística con `cross_val_score` sobre esas 20 columnas ya seleccionadas.
3. **Bien**: pon `SelectKBest` **dentro** del mismo `Pipeline` que la logística, y mide el AUC con `cross_val_score` sobre el pipeline completo (sin seleccionar nada antes).
4. Compara ambos números contra la verdad (0,50).
""")
code("""
from sklearn.feature_selection import SelectKBest

rng = np.random.default_rng(42)
Xr = rng.normal(size=(100, 1000))
yr = rng.integers(0, 2, size=100)

# tu código aquí
""")
solution("""
Xs = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)
auc_mal = cross_val_score(LogisticRegression(max_iter=2000), Xs, yr,
                          cv=5, scoring="roc_auc").mean()

pipe_ok = Pipeline([
    ("sel", SelectKBest(f_classif, k=20)),
    ("clf", LogisticRegression(max_iter=2000)),
])
auc_bien = cross_val_score(pipe_ok, Xr, yr, cv=5, scoring="roc_auc").mean()

print(f"Selección FUERA del CV       : AUC {auc_mal:.3f}")
print(f"Selección DENTRO del pipeline: AUC {auc_bien:.3f}")
print(f"\\nLa verdad                    : AUC 0.500")
print(f"Puntos de AUC inventados     : {auc_mal - 0.5:.3f}")
""", "Seleccionar fuera del CV da AUC = 0,882: un número completamente inventado para un problema sin ninguna señal real. "
     "Seleccionar dentro del pipeline da AUC = 0,514, muy cerca del 0,50 verdadero (la diferencia restante es el ruido normal de un dataset de solo 100 observaciones). "
     "La brecha, 0,882 − 0,500 = 0,382 puntos de AUC, es literalmente la cantidad de desempeño que te \"regalaste\" a vos mismo por seleccionar mirando los datos de validación.")

md("""
### E2 · Por qué pasa: falsos positivos esperados (5 min)
Con $p$ variables independientes del target y un test al nivel $\\alpha = 0{,}05$, el número esperado de falsos positivos es $\\mathbb{E}[\\text{FP}] = \\alpha \\cdot p$.

### ✍️ Ejercicio E2
Calcula $\\alpha \\cdot p$ para $p=1.000$ y $\\alpha=0{,}05$. Compáralo con las 20 que `SelectKBest` efectivamente eligió en E1. ¿Qué te dice ese número sobre por qué el AUC "malo" salió tan alto?
""")
code("""
# tu código aquí
""")
solution("""
alpha = 0.05
p_ruido = 1000
fp_esperados = alpha * p_ruido
print(f"Falsos positivos esperados: {fp_esperados:.0f} de {p_ruido} variables de ruido puro")
""", "50 falsos positivos esperados de las 1.000 variables de ruido, muy por encima de las 20 que `SelectKBest` necesitaba elegir. Elegiste las 20 que **mejor coincidieron por pura casualidad** con el target en esas 100 observaciones concretas, y después validaste sobre esas mismas 100 observaciones: el fold de validación contiene exactamente los casos que hicieron que esas variables parecieran buenas. El modelo no generaliza, **memoriza una coincidencia**, y la validación cruzada no lo detecta porque la trampa ocurrió *antes* de partir los datos.")

md("""
### E3 · El pipeline honesto (8 min)
Con la base de churn real (no el experimento de ruido), armemos un pipeline donde **todo lo que mira los datos** —imputación, escalado y selección— vive dentro de cada fold.

### ✍️ Ejercicio E3
Construye `Pipeline([SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(mutual_info_classif, k=12), LogisticRegression(max_iter=3000)])` y mide su AUC con `StratifiedKFold(5, shuffle=True, random_state=1)` sobre `X, y` (las 50 candidatas de la Parte B).

⚠️ `mutual_info_classif` tiene su propia aleatoriedad interna (viste en C2 que cambia con la semilla). Para que el resultado sea reproducible, fija esa semilla con `functools.partial(mutual_info_classif, random_state=0)` en vez de pasar la función tal cual.
""")
code("""
from functools import partial
from sklearn.impute import SimpleImputer

# tu código aquí
""")
solution("""
mi_fijo = partial(mutual_info_classif, random_state=0)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("sel", SelectKBest(mi_fijo, k=12)),
    ("clf", LogisticRegression(max_iter=3000)),
])

cv_e3 = StratifiedKFold(5, shuffle=True, random_state=1)
auc_honesto = cross_val_score(pipe, X, y, cv=cv_e3, scoring="roc_auc")

print("AUC por fold:", auc_honesto.round(4))
print("AUC medio   :", auc_honesto.mean().round(4))
""", "El AUC medio de este pipeline honesto da 0,8996 (compáralo con el 0,9406 de usar las 50 candidatas sin seleccionar, calculado en D3: la selección a 12 variables pierde apenas 4 puntos de AUC frente a usar las 50). "
     "La diferencia con E1 es que aquí **sí hay señal real** en los datos (no es ruido puro), así que no esperamos que el AUC caiga a 0,5 al hacerlo bien — esperamos que sea un número un poco más conservador (y honesto) que si hubiéramos seleccionado por fuera del CV con las 50 candidatas completas. "
     "Si no fijas la semilla de `mutual_info_classif` (el `⚠️` de arriba), vas a obtener un número parecido pero no idéntico cada vez que corras esta celda — el mismo fenómeno de C2, aplicado ahora dentro de un pipeline.")

md("""
### E4 · Antes del split o dentro del fold (8 min)
No todos los pasos de un pipeline son iguales: algunos no miran el target y pueden ir antes de partir los datos; otros sí lo miran (o usan un estadístico calculado con todo el conjunto) y tienen que ir dentro de cada fold.

### ✍️ Ejercicio C33-E4
Clasifica cada paso como **antes del split** o **dentro del fold**, y justifica brevemente los dos que más dudas generan:
1. Convertir a `NaN` los negativos imposibles de `segundos`.
2. Construir los 50 ratios/logaritmos/interacciones de la Parte B.
3. Imputar los `NaN` por la mediana.
4. Escalar con `StandardScaler`.
5. Filtrar por información mutua (`SelectKBest`).
6. Elegir qué variable de un par redundante se conserva.
7. Ajustar RFE o Lasso.
""")
code("""
# tu código aquí (escribe tu clasificación como comentarios o un dict)
""")
solution("""
clasificacion = {
    "Convertir negativos a NaN":            "antes del split  (no mira el target, es una regla fija de negocio)",
    "Construir ratios/logs/interacciones":  "antes del split  (son funciones deterministas de cada fila, no usan estadísticos del conjunto)",
    "Imputar por mediana":                  "DENTRO del fold  (la mediana se calcula con el conjunto, y calcularla con datos de test filtra información)",
    "Escalar (StandardScaler)":             "DENTRO del fold  (usa la media y el desvío del conjunto completo)",
    "Filtrar por información mutua":        "DENTRO del fold  (mira el target directamente)",
    "Elegir el par redundante a conservar": "DENTRO del fold  (el desempate se basó en información mutua, que mira el target)",
    "RFE o Lasso":                          "DENTRO del fold  (ambos miran el target para decidir qué variable sobrevive)",
}
for paso, cuando in clasificacion.items():
    print(f"{paso:42s} -> {cuando}")
""", "Los dos que más dudas suelen generar: **imputar por mediana** parece inocente porque no mira el target, pero sí usa un estadístico (la mediana) calculado sobre todas las filas — si lo calculas antes de partir, el fold de entrenamiento \"sabe\" algo sobre la mediana del fold de test. Y **elegir el par redundante a conservar**: parece una decisión de negocio (como la limpieza), pero en E3-E4 el criterio de desempate fue la información mutua con el target, así que sí mira `y` y por lo tanto tiene que recalcularse dentro de cada fold, no decidirse una sola vez con todos los datos. "
     "Regla general para no errar: **si el paso necesita `y`, o necesita un número (media, mediana, desvío) calculado sobre el conjunto completo, va dentro del fold.** Si es una transformación fila por fila que no necesita mirar nada del conjunto ni del target, puede ir antes.")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (15 min, SIN mirar lo anterior)

Construye **tu propio** pipeline honesto de selección de variables sobre `X, y` (las 50 candidatas):

1. Elige un método (Lasso con el `C` que prefieras, o `SelectKBest` con el `k` que prefieras) y arma un `Pipeline` que incluya imputación, escalado, selección y el clasificador — todo dentro del mismo pipeline.
2. Mide su AUC con `StratifiedKFold(5, shuffle=True, random_state=1)` y `cross_val_score`.
3. Ajusta el pipeline completo una vez más sobre todo `X, y` (fuera de la validación cruzada, solo para *inspeccionar* qué quedó, no para reportar el AUC) y lista las variables que sobrevivieron.
4. De esas variables, elige 3 y justifica en una frase **de negocio** (no estadística) por qué tiene sentido que le importen a un modelo de fuga.
5. Responde sin código: en una telco real, las variables de este dataset se calculan con información hasta cierto mes (por ejemplo, junio). Si quisieras predecir qué clientes se van a fugar en julio, ¿por qué las variables de entrada tienen que venir de **junio hacia atrás** y no de julio o después? ¿Qué pasaría con tu AUC de validación si, por error, alguna variable usara información de julio?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mis variables elegidas y su justificación de negocio:**

1.
2.
3.

✍️ **Mi respuesta sobre la ventana temporal:**

""")
solution("""
pipe_cierre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("sel", LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                               max_iter=3000, random_state=0)),
])
# Usamos Lasso embebido como selector: ajustamos aparte para poder leer coef_.
cv_cierre = StratifiedKFold(5, shuffle=True, random_state=1)
pipe_eval = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                               max_iter=3000, random_state=0)),
])
auc_cierre = cross_val_score(pipe_eval, X, y, cv=cv_cierre, scoring="roc_auc")
print("AUC por fold:", auc_cierre.round(4))
print("AUC medio   :", auc_cierre.mean().round(4))

pipe_eval.fit(X, y)
coef_cierre = pd.Series(pipe_eval["clf"].coef_[0], index=X.columns)
sobrevivientes = coef_cierre[coef_cierre != 0].reindex(
    coef_cierre.abs().sort_values(ascending=False).index).dropna()
print(f"\\n{len(sobrevivientes)} variables sobreviven:")
print(sobrevivientes.round(3))
""", "Con Lasso `C=0.05` dentro de un pipeline con imputación y escalado, el AUC medio de CV da 0,9339 (parecido al 0,9396 de D3, que usaba Lasso-27 ajustado directamente sobre todos los datos sin CV ni imputación dentro del fold: la pequeña diferencia es, justamente, el costo honesto de hacer todo dentro del pipeline en vez de seleccionar una sola vez con todos los datos). "
     "Entre las 27 que sobreviven, las de mayor coeficiente absoluto son `frec_llamadas` (−0,592), `monto` (−0,524), `cliente_nuevo` (+0,520), `numero_fallas` (+0,517) y `reclamos` (+0,441) — una mezcla distinta a la de C1 (ahí ganaba `reclamos` sola; acá, en conjunto y con Lasso, otras variables toman el primer lugar). "
     "Justificación de negocio esperada (ejemplo con 3 de ellas): `frec_llamadas` (coeficiente negativo) — un cliente que sigue llamando activamente está usando el servicio, y el uso activo suele anticipar permanencia, no fuga. `cliente_nuevo` (coeficiente positivo) — los clientes con menos de 12 meses de antigüedad todavía no desarrollaron lealtad ni costos de cambio, es razonable que se fuguen más que la cartera madura. `numero_fallas` (coeficiente positivo) — más fallas técnicas reportadas es más fricción con el servicio, un motivo directo para irse (y recuerda que en C3 esta misma variable parecía irrelevante sola: acá, en conjunto, sí aporta). "
     "**Ventana temporal**: si el modelo se va a usar para predecir fugas de julio, toda variable de entrada tiene que calcularse con datos hasta junio inclusive, porque en el momento real de predecir (antes de julio) julio todavía no existe. Si por error alguna variable usara información de julio (por ejemplo, `n_contactos` de julio, que podría incluir la llamada en la que el cliente pide cancelar), el modelo estaría usando **información del futuro respecto al momento de la predicción**: es una forma de data leakage distinta a la de la Parte E (leakage temporal, no leakage de selección), y el AUC de validación saldría artificialmente alto — tan alto como inútil, porque en producción esa información todavía no existiría al momento de predecir.")

# ---------------------------------------------------------------- self-eval
md("""
## Autoevaluación (responde sin mirar)

1. ¿Cuál es la diferencia entre una variable **irrelevante**, una **redundante** y una **relevante solo en conjunto**? ¿Por qué un filtro univariado puede fallar con la tercera y no con las otras dos?
2. ¿Por qué se usa la mediana y no la media para imputar `segundos`, y por qué se imputa en vez de simplemente borrar las filas con valores imposibles?
3. En Lasso, ¿qué significa que `C` sea chico? ¿Por qué Ridge (L2) no sirve para seleccionar variables aunque también penalice los coeficientes?
4. ¿Qué parte exacta del pipeline de selección tiene que vivir "dentro del fold" en la validación cruzada, y qué pasa con el AUC si en cambio se hace antes de partir los datos?
5. Si `forward`/RFE y Lasso llegan a conjuntos de variables distintos sobre los mismos datos, ¿cuál de los dos es "el correcto"?

<details><summary><b>Respuestas</b></summary>

1. **Irrelevante**: no aporta información sobre el target ni sola ni acompañada (se descarta). **Redundante**: aporta, pero esa información ya está en otra variable (se conserva solo una de las dos). **Relevante solo en conjunto**: sola parece ruido, pero combinada con otra variable sí predice (como `numero_fallas` en C3, o el XOR en C4). Un filtro univariado evalúa cada variable *aislada* del resto, así que no tiene forma de ver que una variable aporta únicamente en combinación — por definición, mide exactamente lo que esa variable no tiene: señal sola.
2. La mediana es robusta a las colas (no se mueve con un outlier), mientras que la media sí — y `segundos` tenía valores negativos extremos (hasta −14.645) que, de promediarse, distorsionarían cualquier imputación. Se imputa en vez de borrar porque borrar filas reduce el tamaño de muestra y, en casos como `edad` (108 filas entre 0-14 años), puede eliminar un segmento de negocio real en vez de solo basura.
3. En la parametrización de `scikit-learn`, $C$ es el inverso de la fuerza de penalización: $C$ chico implica que el error de predicción pesa poco frente a la penalización $\\|\\beta\\|_1$, así que más coeficientes se van exactamente a cero. Ridge penaliza $\\|\\beta\\|_2^2$, cuya forma geométrica (una esfera, sin esquinas) hace que el óptimo casi nunca caiga justo en un eje — los coeficientes se achican pero rara vez llegan a cero exacto, así que Ridge regulariza pero no selecciona.
4. Todo paso que mire el target (filtros, RFE, Lasso) o que use un estadístico calculado sobre el conjunto completo (la media y el desvío del escalador, la mediana del imputador) tiene que ir dentro de cada fold. Si se hace antes de partir los datos, el fold de validación queda "contaminado" con información que ayudó a elegir esas variables, y el AUC medido queda inflado — en el experimento de ruido puro de E1, la inflación llevó un problema sin ninguna señal real (AUC verdadero 0,50) a un AUC medido de alrededor de 0,85-0,90.
5. **Ninguno de los dos es "el correcto" por definición.** RFE y Lasso son heurísticas distintas (una wrapper, voraz y basada en reentrenar; otra embedded, basada en una penalización) que no tienen por qué coincidir, y ninguna garantiza encontrar el mejor subconjunto entre los $2^{50}-1$ posibles (D1). Lo razonable es compararlos con criterios adicionales — AUC de CV, estabilidad ante cambios de semilla o de partición, sentido de negocio de las variables que sobreviven — en vez de asumir que uno es automáticamente superior.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 34 · Árboles de decisión. Revisa `Guias/apuntes/C33_Ejercicios_Cuaderno_FeatureEngineering.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT)
