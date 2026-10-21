"""Genera el notebook guiado de A/B testing y causalidad aplicada (Clase 22).

Uso normal:      python notebook_c22.py             -> Guias/notebooks/C22_ABTesting.ipynb
Modo de prueba:  C22_TEST=1 python notebook_c22.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c22_numeros.json (lo escribe figuras_c22.py) con marcas [[ruta|decimales]].
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C22_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C22_ABTesting_test.ipynb" if TEST else "C22_ABTesting.ipynb")
N = json.loads((HERE / "c22_numeros.json").read_text(encoding="utf-8"))
cells = []


def val(m):
    obj = N
    for k in m.group(1).split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    d = m.group(2)
    if d is None:
        return str(obj)
    s = f"{float(obj):.{int(d)}f}"
    return s


def fill(text):
    return re.sub(r"\[\[([\w.]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>")


# ---------------------------------------------------------------- encabezado
md("""
# Clase 22 · Experimentación, A/B testing y causalidad aplicada
**Diplomado Data Science** · Prof. Maritza Márquez

Material de apoyo:
- `Guias/C22_Guia_ABTesting.pdf` — guía visual de la clase.
- `Guias/apuntes/C22_Apuntes_ABTesting.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C22_Ejercicios_Cuaderno_ABTesting.pdf` — ejercicios en papel (antes o después de este notebook).
- Material de la profesora (carpeta de la Clase 22): `Clase22.pdf` (diapositivas), `Clase22_ABtesting.R` y `.ipynb` (práctica en R), `ab_checkout.csv`, `ab_srm.csv` y `did_app_regiones.csv`.
- Transcripción: `VideosClases/Clase 22 Experimentacion A-B testing y causalidad aplicada.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy` y `statsmodels`. Si no los tienes:
```
pip install numpy pandas matplotlib scipy statsmodels
```
**Datos** (los tres archivos de la profesora):
- `ab_checkout.csv`: experimento **limpio** con 22 000 usuarios (control y tratamiento; checkout nuevo). Columnas: `grupo`, `dispositivo`, `region`, `gasto_pre`, `sesiones_pre` (previas al experimento), `convierte` (0/1) y `monto_compra`.
- `ab_srm.csv`: experimento **roto** (11 476 usuarios) para diagnosticar un *Sample Ratio Mismatch*.
- `did_app_regiones.csv`: panel de 160 comunas × 6 meses (`uso_app`) con un lanzamiento en la Región Metropolitana (RM) para **diferencias en diferencias**.

### Objetivos
1. Distinguir una pregunta **predictiva** de una **causal** y ver por qué **aleatorizar** hace comparables los grupos.
2. **Diseñar**: calcular el tamaño de muestra (α, potencia, MDE) y entender de dónde sale cada número.
3. **Chequear** el experimento: SRM (chi-cuadrado), balance de covariables, *peeking* y comparaciones múltiples.
4. **Analizar**: diferencia de tasas, intervalo de confianza y p-valor (con una explicación intuitiva), test de proporciones, modelo lineal y logístico.
5. Ganar **precisión** con covariables y CUPED, y mirar la **heterogeneidad** con cautela.
6. Estimar un efecto **sin aleatorizar** con diferencias en diferencias y defender las tendencias paralelas.
7. Reproducir los resultados de la profesora, traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación y datos |
| A | 15 | A1 · Resultados potenciales y por qué aleatorizar |
| A | 10 | A2 · Balance de covariables |
| A | 20 | A3 · Tamaño de muestra, MDE y potencia |
| B | 10 | B1 · SRM en el experimento limpio |
| B | 20 | B2 · Diferencia de tasas, IC y p-valor |
| B | 15 | B3 · El p-valor con tus propias manos: permutación |
| B | 15 | B4 · Modelo lineal y logístico |
| C | 20 | C1 · Covariables y CUPED |
| C | 15 | C2 · Heterogeneidad por dispositivo |
| C | 15 | C3 · El experimento roto (`ab_srm.csv`) |
| C | 20 | C4 · Peeking y comparaciones múltiples (simulación) |
| D | 25 | D1 · Diferencias en diferencias |
| D | 10 | D2 · Tabla de correspondencia R ↔ Python |
| — | 15 | Alertas del material, verificadas con código |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook (`scipy 1.18`, `statsmodels 0.15`). Con otras versiones pueden variar un poco; las simulaciones usan semilla fija.
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import scipy
import statsmodels
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 2026   # semilla para todo lo aleatorio
print("scipy:", scipy.__version__, "| statsmodels:", statsmodels.__version__)
""")
md("""
### Mini-repaso de lo que se usa hoy
- **`proportions_ztest`**, **`confint_proportions_2indep`** y **`NormalIndPower`** (de `statsmodels`) hacen lo que en R hacen `prop.test` y `power.prop.test`; **`scipy.stats.chisquare`** es el `chisq.test` del SRM.
- **`smf.ols`** y **`smf.logit`** son `lm` y `glm(family = binomial)`: se escriben con fórmulas igual que en R (`convierte ~ trat`).
- Todo lo aleatorio (barajar grupos, simular) usa `np.random.default_rng(RS)`: así los resultados se repiten.

### Los datos de la clase
Se cargan los tres archivos; `trat` vale 1 para tratamiento (variante B, checkout nuevo) y 0 para control (variante A, checkout actual).
""")
code("""
def buscar(nombre):
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in list(base.glob(f"Clase 22*/{nombre}")) + list(base.glob(nombre)):
            return p
    raise FileNotFoundError(f"No encuentro {nombre}: abre el notebook dentro de la carpeta del curso.")


ab = pd.read_csv(buscar("ab_checkout.csv"))
ab["trat"] = (ab["grupo"] == "tratamiento").astype(int)
srm = pd.read_csv(buscar("ab_srm.csv"))
did = pd.read_csv(buscar("did_app_regiones.csv"))
did["trat"] = (did["zona"] == "RM").astype(int)
print("ab_checkout:", ab.shape, "| ab_srm:", srm.shape, "| did_app_regiones:", did.shape)
display(ab.head(3))
display(ab.groupby("grupo").agg(usuarios=("convierte", "size"), conversiones=("convierte", "sum"), tasa=("convierte", "mean")).round(4))
""")
md("""
Control **10 965** usuarios y 974 conversiones (**8.88 %**); tratamiento **11 035** y 1 194 (**10.82 %**): son los números de la diapositiva 16. Más abajo veremos si esa diferencia es creíble.
""")

# ================================================================== PARTE A
md("""
## Parte A · Causalidad y diseño

### A1 · Resultados potenciales y por qué aleatorizar (15 min)
Cada usuario tendría **dos resultados**: `Y(1)` si recibe el cambio y `Y(0)` si no. El efecto individual `τ = Y(1) − Y(0)` **nunca se observa** (de cada persona vemos solo uno). En la vida real no podemos ver ambos, pero **en una simulación sí**: así vemos qué estima cada método.

Simulamos 20 000 usuarios con un «interés» latente `z` (un **confusor**): quienes tienen más interés compran más **aunque no se les cambie nada**. El cambio sube la probabilidad de comprar en **+2 puntos porcentuales** para todos (el ATE verdadero).
""")
code("""
rng = np.random.default_rng(RS)
n = 20000
z = rng.normal(size=n)                                   # interés latente (confusor)
p0 = 1 / (1 + np.exp(-(-2.3 + 0.9 * z)))                 # prob. de comprar SIN el cambio
p1 = np.clip(p0 + 0.02, 0, 1)                            # prob. CON el cambio: +2 pp para todos
u = rng.random(n)
y0, y1 = (u < p0).astype(int), (u < p1).astype(int)      # los dos resultados potenciales (los conocemos solo aquí)
print(f"ATE verdadero (promedio de Y(1) − Y(0)): {100 * (y1 - y0).mean():.2f} pp")

# (a) Autoselección: quien tiene más interés adopta el cambio con más probabilidad
adopta = (rng.random(n) < 1 / (1 + np.exp(-1.2 * z))).astype(int)
y_obs_auto = np.where(adopta == 1, y1, y0)
d_auto = y_obs_auto[adopta == 1].mean() - y_obs_auto[adopta == 0].mean()
# (b) Aleatorización: una moneda decide
azar = rng.integers(0, 2, n)
y_obs_azar = np.where(azar == 1, y1, y0)
d_azar = y_obs_azar[azar == 1].mean() - y_obs_azar[azar == 0].mean()
print(f"diferencia con autoselección: {100 * d_auto:.2f} pp   <- sesgada por el confusor")
print(f"diferencia con aleatorización: {100 * d_azar:.2f} pp  <- cerca del ATE verdadero")
print(f"interés medio de quienes adoptan vs no adoptan: {z[adopta == 1].mean():.2f} vs {z[adopta == 0].mean():.2f} | con azar: {z[azar == 1].mean():.2f} vs {z[azar == 0].mean():.2f}")
""")
md("""
Con **autoselección** la diferencia es mucho mayor que 2 pp porque quienes adoptan **ya eran** más propensos a comprar (el grupo tratado tiene mayor interés medio). Con **aleatorización** el interés queda equilibrado y la diferencia se acerca al efecto verdadero (con algo de ruido muestral). Esa es la frase de la diapositiva 11: *«la asignación por azar rompe la asociación sistemática entre tratamiento y características observadas y no observadas»*.

### ✍️ Ejercicio A1
Repite la aleatorización 500 veces (semilla fija) y calcula el promedio y la desviación estándar de la diferencia estimada. ¿El método es **insesgado**? ¿Qué mide la desviación estándar (en términos de un experimento real)?
""")
code("""
# tu código aquí
""")
solution("""
rng2 = np.random.default_rng(7)
difs = []
for _ in range(500):
    a_ = rng2.integers(0, 2, n)
    y_ = np.where(a_ == 1, y1, y0)
    difs.append(y_[a_ == 1].mean() - y_[a_ == 0].mean())
difs = np.array(difs)
print(f"promedio de las 500 diferencias: {100 * difs.mean():.2f} pp | sd: {100 * difs.std():.2f} pp | ATE verdadero: {100 * (y1 - y0).mean():.2f} pp")
""", "El promedio de las 500 estimaciones queda muy cerca del ATE verdadero (≈ 2 pp): el estimador es **insesgado** bajo aleatorización. La desviación estándar (≈ 0.5 pp) es el **error estándar**: cuánto cambiaría la diferencia de un experimento a otro solo por el azar de a quién le toca qué. Ese ruido es el que el IC y el p-valor cuantifican.")

md("""
### A2 · Balance de covariables (10 min)
En un experimento aleatorizado, las covariables **previas** deberían quedar parecidas entre grupos. Se mide con la **diferencia estandarizada** `d = (media_T − media_C)/sd` (regla práctica: `|d| < 0.1` está bien).
""")
code("""
def smd(df, col):
    t, c = df.loc[df["trat"] == 1, col], df.loc[df["trat"] == 0, col]
    return (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)


filas = {v: smd(ab, v) for v in ["gasto_pre", "sesiones_pre"]}
for col in ["dispositivo", "region"]:
    for lev in sorted(ab[col].unique()):
        ind = (ab[col] == lev).astype(int)
        t, c = ind[ab["trat"] == 1], ind[ab["trat"] == 0]
        filas[f"{col} = {lev}"] = (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
balance = pd.Series(filas).round(4)
display(balance.to_frame("d"))
print("máximo |d|:", round(float(balance.abs().max()), 4))
print(ab.groupby("grupo")[["gasto_pre", "sesiones_pre"]].mean().round(3))
""")
md("""
Todas las diferencias son minúsculas (máximo |d| = [[balance.max_smd_aleatorio|3]]): los grupos son comparables en lo **observado**. Importante: el azar equilibra también lo **no observado**, y eso es lo que no podemos medir.

### ✍️ Ejercicio A2
La columna `monto_compra` ¿sirve para chequear el balance? Calcula `smd(ab, "monto_compra")` y explica por qué **no** se debe usar (ni ajustar por ella).
""")
code("""
# tu código aquí
""")
solution("""
print("d de monto_compra:", round(float(smd(ab, "monto_compra")), 4))
print(ab.groupby("grupo")["monto_compra"].mean().round(1))
""", "`monto_compra` se mide **después** del tratamiento: es un resultado (y además depende de `convierte`). Su diferencia entre grupos no indica un mal balance sino, probablemente, el efecto del tratamiento. Ajustar por una variable afectada por el tratamiento distorsiona el efecto (slide 23: «nunca se ajusta por variables afectadas por el tratamiento»). Solo se chequean y se usan covariables **previas**.")

md("""
### A3 · Tamaño de muestra, MDE y potencia (20 min)
Antes de lanzar se fija: el **MDE** (efecto mínimo que importa detectar), el nivel de significancia **α** (falso positivo tolerado) y la **potencia 1 − β** (probabilidad de detectar un efecto del tamaño del MDE). Con la **conversión histórica de 9 %** y un MDE de **1.5 pp** (10.5 %), `power.prop.test` de R entrega el tamaño por grupo. Esta es la misma fórmula en Python:

`n = [ z₁₋α/₂·√(2·p̄·(1−p̄)) + z₁₋β·√(p₁(1−p₁) + p₂(1−p₂)) ]² / (p₂ − p₁)²`, con `p̄ = (p₁ + p₂)/2`.
""")
code("""
def n_power_prop_test(p1, p2, alfa=0.05, potencia=0.80):
    za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(potencia)
    pbar = (p1 + p2) / 2
    return ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / abs(p1 - p2)) ** 2


n_r = n_power_prop_test(0.09, 0.105)
print(f"fórmula de power.prop.test: n = {n_r:.3f} por grupo -> {int(np.ceil(n_r)):,} por grupo, {2 * int(np.ceil(n_r)):,} en total")

# la versión de statsmodels usa el tamaño de efecto h de Cohen (arcoseno)
h = proportion_effectsize(0.105, 0.09)
n_sm = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.80, ratio=1)
print(f"statsmodels (arcoseno, h = {h:.4f}): n = {n_sm:.2f} -> {int(np.ceil(n_sm)):,} por grupo, {2 * int(np.ceil(n_sm)):,} en total")
""")
md("""
`power.prop.test` da **6 137.95 → 6 138** por grupo (12 276 en total). La diapositiva 18 dice «aproximadamente **6 131** (12 262 en total)» como salida de `power.prop.test`: **6 131 es el valor de la aproximación del arcoseno** (`statsmodels`, h de Cohen), no el de R. La diferencia es de 7 usuarios por grupo (0.1 %) y no cambia la decisión, pero conviene saber de dónde sale cada número (ver «Alertas»).

Dos propiedades que hay que *sentir*: (1) si el MDE se **divide por 2**, n se **multiplica por ≈ 4** (n ∝ 1/MDE²); (2) subir la potencia de 80 % a 90 % cuesta ≈ 34 % más usuarios.
""")
code("""
filas = []
for mde in (0.0075, 0.01, 0.015, 0.02, 0.03):
    filas.append(dict(MDE_pp=mde * 100, p2=0.09 + mde, n_por_grupo_80=int(np.ceil(n_power_prop_test(0.09, 0.09 + mde))),
                      n_por_grupo_90=int(np.ceil(n_power_prop_test(0.09, 0.09 + mde, potencia=0.90)))))
display(pd.DataFrame(filas).set_index("MDE_pp").round(3))

# ¿qué MDE alcanzaba el experimento real (11 000 por grupo)? se despeja el efecto con la misma función
from scipy.optimize import brentq
n_real = len(ab) / 2
mde_real = brentq(lambda d: n_power_prop_test(0.09, 0.09 + d) - n_real, 1e-4, 0.1)
print(f"MDE alcanzado con {n_real:,.0f} por grupo y potencia 80 %: {100 * mde_real:.2f} pp")
""")
md("""
Con 11 000 por grupo el experimento real detectaba efectos de ≈ **[[poder.mde_real_80|2]] pp** con 80 % de potencia: tenía **mucho más** tamaño que el mínimo (6 138). Una pregunta de la profesora: *«¿el experimento real tiene al menos ese n por grupo?»* Sí (≈ 11 000 > 6 138). *«¿Es el MDE de 1.5 pp una diferencia de negocio relevante?»* Eso lo decide el negocio, no la estadística.

### ✍️ Ejercicio A3
Con línea base 12 % y un MDE de +2 pp, calcula `n` por grupo para potencia 80 % y 90 %. Luego calcula **qué potencia** tendría un experimento con solo 1 500 usuarios por grupo (usa `NormalIndPower().power(...)`). ¿Qué decisión tomarías?
""")
code("""
# tu código aquí
""")
solution("""
print("n por grupo, 80 %:", int(np.ceil(n_power_prop_test(0.12, 0.14))), "| 90 %:", int(np.ceil(n_power_prop_test(0.12, 0.14, potencia=0.90))))
h2 = proportion_effectsize(0.14, 0.12)
print("potencia con 1 500 por grupo:", round(float(NormalIndPower().power(effect_size=h2, nobs1=1500, alpha=0.05, ratio=1)), 3))
""", "Hacen falta ≈ **4 440 por grupo** con 80 % de potencia y ≈ **5 940** con 90 %. Con 1 500 por grupo la potencia es solo ≈ **0.37**: hay más probabilidad de no detectar un efecto real de 2 pp que de detectarlo. Decisión razonable: aceptar un MDE mayor, reunir más tráfico/tiempo o no hacer el experimento (un resultado «no significativo» sería casi no informativo).")

# ================================================================== PARTE B
md("""
## Parte B · El efecto principal

### B1 · SRM en el experimento limpio (10 min)
El **Sample Ratio Mismatch** es una alarma: si el diseño asigna 50/50 y los conteos observados difieren más de lo que el azar permite, algo falló (asignación, registro, bots…). Se prueba con un **chi-cuadrado**: `χ² = Σ (observado − esperado)²/esperado`, con (grupos − 1) grados de libertad. Es la función `srm_test` del script de R.
""")
code("""
def srm_test(grupo, ratio=(0.5, 0.5)):
    tab = grupo.value_counts().reindex(["control", "tratamiento"])
    esperado = tab.sum() * np.array(ratio)
    chi = stats.chisquare(tab.values, f_exp=esperado)
    return pd.DataFrame({"grupo": tab.index, "usuarios": tab.values, "esperado": esperado, "residuo": tab.values - esperado,
                         "chi2": chi.statistic, "p_valor": chi.pvalue})


srm_checkout = srm_test(ab["grupo"])
display(srm_checkout.round(4))
""")
md("""
Observado 10 965 y 11 035 frente a 11 000 esperados: `χ² = (35²/11 000)·2 = 0.2227`, **p = [[srm.checkout.p|3]]**. Con la regla práctica (p < 0.001 → auditar) no hay alarma. **Precisión importante**: en clase se dijo que un p grande indica que el experimento «está completamente aleatorizado». Un p grande solo dice que **no hay evidencia de desbalance en los conteos**; no *prueba* que la aleatorización fue perfecta.

### ✍️ Ejercicio B1
Calcula a mano `χ²` y el p-valor para 10 000 usuarios con conteos 5 100 / 4 900 (esperado 50/50) y para 5 300 / 4 700. ¿En cuál se declara SRM con p < 0.001?
""")
code("""
# tu código aquí
""")
solution("""
for obs in ([5100, 4900], [5300, 4700]):
    chi = sum((o - 5000) ** 2 / 5000 for o in obs)
    print(obs, "chi2 =", round(chi, 2), "| p =", float(f"{stats.chi2.sf(chi, 1):.3g}"))
""", "5 100 / 4 900: χ² = 4.0 (p ≈ 0.046): bajo 0.05 pero no bajo 0.001, así que con la regla práctica no se declara SRM (sí merece revisar). 5 300 / 4 700: χ² = 36 (p ≈ 2 × 10⁻⁹): SRM claro.")

md("""
### B2 · Diferencia de tasas, intervalo de confianza y p-valor (20 min)
**La idea, sin fórmulas.** Cada tasa observada es una estimación con ruido (otro grupo de usuarios habría dado otra tasa). El **error estándar (EE)** mide ese ruido. La diferencia dividida por su EE, `z`, cuenta «cuántos ruidos» se aleja del 0.
- **p-valor**: si el cambio *no hiciera nada* (H₀), ¿qué tan raro sería ver una diferencia al menos así de grande solo por azar? Un p-valor pequeño dice que el azar no alcanza para explicar lo visto. **No** es la probabilidad de que H₀ sea cierta, y **no** mide qué tan grande o importante es el efecto.
- **Intervalo de confianza 95 %**: el rango de efectos que los datos «toleran». Dice **cuánto**, no solo si hay efecto.

En fórmulas: `EE = √[p̂ᴀ(1−p̂ᴀ)/nᴀ + p̂ʙ(1−p̂ʙ)/nʙ]` (para el IC), `IC = (p̂ʙ − p̂ᴀ) ± 1.96·EE` y `z = (p̂ʙ − p̂ᴀ)/√[p̂(1−p̂)(1/nᴀ + 1/nʙ)]` con la tasa conjunta `p̂` (para el test, porque bajo H₀ las dos tasas son iguales).
""")
code("""
c, t = ab[ab["trat"] == 0]["convierte"], ab[ab["trat"] == 1]["convierte"]
xA, nA, xB, nB = int(c.sum()), len(c), int(t.sum()), len(t)
pA, pB = xA / nA, xB / nB
d = pB - pA
ee_wald = np.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
p_conj = (xA + xB) / (nA + nB)
ee_conj = np.sqrt(p_conj * (1 - p_conj) * (1 / nA + 1 / nB))
z_obs = d / ee_conj
p_val = 2 * stats.norm.sf(abs(z_obs))
lo, hi = d - 1.959964 * ee_wald, d + 1.959964 * ee_wald
print(f"pA = {pA:.4f} | pB = {pB:.4f} | diferencia = {100 * d:.3f} pp | relativo = {100 * d / pA:.1f} %")
print(f"EE (Wald) = {100 * ee_wald:.3f} pp | IC 95 % = [{100 * lo:.2f}; {100 * hi:.2f}] pp")
print(f"EE (tasa conjunta) = {100 * ee_conj:.3f} pp | z = {z_obs:.3f} | chi2 = {z_obs ** 2:.2f} | p = {p_val:.3g}")

# lo mismo con librerías (equivale a prop.test(correct = FALSE) de R)
z_sm, p_sm = proportions_ztest([xB, xA], [nB, nA])
ic_sm = confint_proportions_2indep(xB, nB, xA, nA, method="wald")
print(f"statsmodels: z = {z_sm:.3f} | p = {p_sm:.3g} | IC wald = [{100 * ic_sm[0]:.2f}; {100 * ic_sm[1]:.2f}] pp")
""")
md("""
Resultado (coincide con la diapositiva 16): diferencia **[[efecto.dif_pp|2]] pp**, IC 95 % **[[efecto.ic_pp.0|2]] a [[efecto.ic_pp.1|2]] pp**, `z = [[efecto.z|2]]` y **p ≈ 1.43 × 10⁻⁶** (`χ² = [[efecto.chi2|2]]`, el valor que reporta `prop.test`). Lectura: *el efecto parece positivo y su IC no contiene el 0 ni el MDE de 1.5 pp en el extremo bajo (llega a 1.15)*; se puede decir que el aumento es de **entre ≈ 1.2 y ≈ 2.7 pp**. Ojo: lo de la clase «p-valor muy bajito → efecto **real y relevante**» mezcla dos ideas: el p-valor dice que el azar no lo explica; la **relevancia** se juzga con el IC frente al MDE y los costos.

Un detalle técnico: el **test** usa el EE con la tasa conjunta ([[efecto.se_pooled|5]]) y el **IC** usa el EE de Wald ([[efecto.se_wald|5]]); por eso `z` y los extremos del IC no son exactamente «equivalentes», aunque aquí casi coinciden.

### ✍️ Ejercicio B2
Con la **diferencia relativa** (`pB/pA − 1`) y sus dos extremos de IC (divide los extremos del IC en pp por `pA`), ¿cuánto «sube» la conversión en términos relativos? Además, calcula el IC con el método de **Newcombe** (`method="newcomb"`) y compáralo con el de Wald. ¿Cambia la conclusión?
""")
code("""
# tu código aquí
""")
solution("""
print(f"relativo: {100 * d / pA:.1f} % (IC aprox. [{100 * lo / pA:.1f} %; {100 * hi / pA:.1f} %])")
ic_nw = confint_proportions_2indep(xB, nB, xA, nA, method="newcomb")
print(f"IC Newcombe: [{100 * ic_nw[0]:.2f}; {100 * ic_nw[1]:.2f}] pp | IC Wald: [{100 * lo:.2f}; {100 * hi:.2f}] pp")
""", "La conversión sube ≈ 22 % en términos relativos (IC aproximado de 13 % a 31 %). El IC de Newcombe es casi idéntico al de Wald porque n es grande y las tasas no son extremas; con muestras pequeñas o tasas cercanas a 0 o 1 es preferible Newcombe o Wilson. La conclusión no cambia.")

md("""
### B3 · El p-valor con tus propias manos: permutación (15 min)
Si el tratamiento no hiciera nada, **quién quedó en tratamiento sería puro azar**. Podemos simularlo: barajamos las etiquetas de grupo muchas veces y vemos qué diferencias aparecen «solo por azar». El p-valor es la fracción de barajados con una diferencia **al menos tan extrema** como la observada.
""")
code("""
rng = np.random.default_rng(RS)
y = ab["convierte"].to_numpy()
g = ab["trat"].to_numpy()
B = 5000
difs_perm = np.empty(B)
for b in range(B):
    gp = rng.permutation(g)
    difs_perm[b] = y[gp == 1].mean() - y[gp == 0].mean()
extremos = int((np.abs(difs_perm) >= abs(d)).sum())
print(f"diferencia observada: {100 * d:.2f} pp | sd de las diferencias barajadas: {100 * difs_perm.std():.3f} pp (EE teórico: {100 * ee_conj:.3f} pp)")
print(f"barajados tan extremos como lo observado: {extremos} de {B} -> p < {1 / B:.4f}")
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(100 * difs_perm, bins=50, color="gray", alpha=0.8)
ax.axvline(100 * d, color="red", lw=2)
ax.set_xlabel("diferencia al barajar los grupos (pp)")
plt.show()
""")
md("""
La desviación estándar de las diferencias barajadas (≈ 0.40 pp) **es el error estándar**, y la observada (1.94 pp) queda a casi 5 desviaciones: **ninguno** de los barajados la alcanza (p < 0.0002). Esa es exactamente la historia que cuenta el z-test, sin usar la normal.

### ✍️ Ejercicio B3
Los p-valores **bajo H₀** son uniformes: el 5 % cae bajo 0.05. Compruébalo: toma 1 500 usuarios al azar del **control** como «A» y otros 1 500 del control como «B» (un test **A/A**), repite 1 000 veces y calcula la fracción de p-valores < 0.05 con `proportions_ztest`. Luego repite con A del control y B del tratamiento. ¿Qué fracción da p < 0.05 en cada caso y cómo se llama la segunda?
""")
code("""
# tu código aquí
""")
solution("""
rng3 = np.random.default_rng(11)
yc, yt = c.to_numpy(), t.to_numpy()
pa, pb = [], []
for _ in range(1000):
    a_, b_ = rng3.choice(yc, 1500), rng3.choice(yc, 1500)
    pa.append(proportions_ztest([b_.sum(), a_.sum()], [1500, 1500])[1])
    b2 = rng3.choice(yt, 1500)
    pb.append(proportions_ztest([b2.sum(), a_.sum()], [1500, 1500])[1])
print("A/A: fracción p < 0.05 =", round(float((np.array(pa) < 0.05).mean()), 3), "| A/B con el efecto real: fracción p < 0.05 =", round(float((np.array(pb) < 0.05).mean()), 3))
""", "En el A/A ≈ 5 % de los p-valores cae bajo 0.05 (los falsos positivos que aceptamos con α = 0.05). En el A/B con el efecto real (1.94 pp) y 1 500 por grupo, ≈ 0.4 de los experimentos detecta el efecto: esa fracción es la **potencia** con ese n (coincide con la curva de la parte A3).")

md("""
### B4 · Modelo lineal y logístico (15 min)
Con `Y = 1` si convierte y `T = 1` si es tratamiento, el **modelo lineal de probabilidad** `Y = β₀ + β₁·T + ε` da `β̂₀ = tasa del control` y `β̂₁ = diferencia de tasas` (la misma del test). La **regresión logística** `log(p/(1−p)) = β₀ + β₁·T` da `exp(β̂₁)` = **odds ratio**. En R: `lm(convierte ~ trat)` y `glm(convierte ~ trat, family = binomial)`.
""")
code("""
mpl = smf.ols("convierte ~ trat", ab).fit()
logit = smf.logit("convierte ~ trat", ab).fit(disp=0)
print("MODELO LINEAL (probabilidad)")
print(pd.concat([mpl.params, mpl.bse, mpl.tvalues, mpl.pvalues, mpl.conf_int()], axis=1).set_axis(["coef", "EE", "t", "p", "IC 2.5%", "IC 97.5%"], axis=1).round(5))
print("\\nLOGIT")
print(pd.concat([logit.params, logit.bse, logit.tvalues, logit.pvalues], axis=1).set_axis(["coef", "EE", "z", "p"], axis=1).round(5))
oratio = np.exp(logit.params["trat"]); ic_or = np.exp(logit.conf_int().loc["trat"])
print(f"\\nodds ratio = {oratio:.4f} | IC 95 % = [{ic_or[0]:.4f}; {ic_or[1]:.4f}] | odds del control = {np.exp(logit.params['Intercept']):.5f} (= {pA:.4f}/{1 - pA:.4f} = {pA / (1 - pA):.5f})")
print(f"errores robustos (HC1) del modelo lineal: EE = {mpl.get_robustcov_results('HC1').bse[1]:.6f} (clásico: {mpl.bse['trat']:.6f})")
""")
md("""
- **Lineal**: `β̂₀ = [[modelos.lpm.b0|4]]` (la tasa del control) y `β̂₁ = [[modelos.lpm.b1|5]]` (**la misma diferencia** del test), `p = 1.43 × 10⁻⁶`.
- **Logit**: `β̂₁ = [[modelos.logit.b1|4]]`, **odds ratio [[modelos.logit.odds_ratio|3]]** (IC [[modelos.logit.ic_or.0|3]] a [[modelos.logit.ic_or.1|3]]): los odds de convertir son ≈ 24 % mayores con el tratamiento. Es otra **escala**; no es «24 puntos».
- Las tres rutas (test, lineal, logit) llegan a la **misma conclusión** (p ≈ 10⁻⁶, IC sin el 0), pero **no** a los mismos números: la profesora lo recalcó («no necesariamente muestran los mismos resultados, sino que llegan a las mismas conclusiones»). Con respuesta binaria los errores del modelo lineal no tienen varianza constante; aquí los errores robustos (HC1) son idénticos a 4 decimales.

### ✍️ Ejercicio B4
El **efecto marginal promedio** del logit es el cambio promedio de probabilidad al pasar T de 0 a 1. Calcúlalo con `logit.get_margeff(dummy=True)` (o a mano con `predict`) y compáralo con la diferencia de tasas. ¿Por qué coinciden?
""")
code("""
# tu código aquí
""")
solution("""
print(logit.get_margeff(dummy=True).summary_frame().round(5))
p1_ = logit.predict(pd.DataFrame({"trat": [1]}))[0]; p0_ = logit.predict(pd.DataFrame({"trat": [0]}))[0]
print("predict(T=1) − predict(T=0) =", round(float(p1_ - p0_), 5), "| diferencia de tasas =", round(float(d), 5))
""", "Coinciden (0.0194; con `dummy=True` el efecto marginal de una variable binaria es el cambio discreto 0 → 1, y no la derivada) porque el modelo solo tiene un predictor binario: el logit reproduce exactamente las dos tasas observadas (es un modelo **saturado**). Con covariables, el efecto marginal promedio y la diferencia de tasas ya no serían idénticos.")

# ================================================================== PARTE C
md("""
## Parte C · Precisión, heterogeneidad y errores frecuentes

### C1 · Covariables y CUPED (20 min)
En un experimento bien aleatorizado, ajustar por covariables **previas** no corrige sesgo: **reduce el ruido** (error estándar) y da un IC más estrecho. **CUPED** hace algo equivalente: `Y_cuped = Y − θ(X − X̄)` con `θ = Cov(Y, X)/Var(X)`, usando una variable pre-experimento `X` (aquí `gasto_pre`). La reducción de varianza es ≈ **ρ²**, el cuadrado de la correlación entre `Y` y `X`.
""")
code("""
mpl_adj = smf.ols("convierte ~ trat + gasto_pre + sesiones_pre + C(dispositivo)", ab).fit()
theta = np.cov(ab["convierte"], ab["gasto_pre"])[0, 1] / ab["gasto_pre"].var()
ab["y_cuped"] = ab["convierte"] - theta * (ab["gasto_pre"] - ab["gasto_pre"].mean())
mpl_cuped = smf.ols("y_cuped ~ trat", ab).fit()
tabla = pd.DataFrame({m: [r.params["trat"], r.bse["trat"], r.pvalues["trat"], *r.conf_int().loc["trat"]] for m, r in
                      {"sin ajuste": mpl, "con covariables": mpl_adj, "CUPED (gasto_pre)": mpl_cuped}.items()},
                     index=["efecto", "EE", "p", "IC 2.5%", "IC 97.5%"]).T
display(tabla.round(5))
red = 1 - (mpl_cuped.bse["trat"] / mpl.bse["trat"]) ** 2
rho = np.corrcoef(ab["convierte"], ab["gasto_pre"])[0, 1]
print(f"theta = {theta:.5f} | reducción de varianza con CUPED = {100 * red:.1f} % | rho = {rho:.4f}, rho² = {100 * rho ** 2:.1f} %")
""")
md("""
Resultados (como en la clase y en la diapositiva 25): el efecto sin ajuste es **[[modelos.lpm.b1|5]]** y con covariables **[[modelos.ajustado.b1|5]]**; el EE baja de [[modelos.lpm.se|5]] a [[modelos.ajustado.se|5]] (con covariables) y [[modelos.cuped.se|5]] (CUPED). La **reducción de varianza con CUPED es [[modelos.cuped.reduccion_var|4]] ≈ 8.9 %**, prácticamente **ρ² = [[modelos.cuped.rho2|4]]** (ρ = [[modelos.cuped.rho|3]]). Es una ganancia modesta: la profesora comentó que en una empresa quizás no valdría la complejidad; en otros contextos (métricas con mucha variabilidad y covariables muy correlacionadas) CUPED puede reducir la varianza a la mitad.

**Dos precisiones.** (1) CUPED y la regresión con covariables **no corrigen** un SRM ni una mala aleatorización; solo ganan precisión. (2) El `θ` de este cálculo se estima con los mismos datos; para inferencia estricta se ajusta como covariable en la regresión (`convierte ~ trat + gasto_pre`) y el EE se calcula con esa regresión.

### ✍️ Ejercicio C1
Aplica CUPED usando **`sesiones_pre`** en lugar de `gasto_pre` y también con **ambas** covariables (regresión). ¿Cuál reduce más la varianza? ¿Por qué `sesiones_pre` ayuda menos?
""")
code("""
# tu código aquí
""")
solution("""
th2 = np.cov(ab["convierte"], ab["sesiones_pre"])[0, 1] / ab["sesiones_pre"].var()
ab["y_cuped2"] = ab["convierte"] - th2 * (ab["sesiones_pre"] - ab["sesiones_pre"].mean())
m2 = smf.ols("y_cuped2 ~ trat", ab).fit()
m3 = smf.ols("convierte ~ trat + gasto_pre + sesiones_pre", ab).fit()
for nombre, r in [("CUPED sesiones_pre", m2), ("regresión gasto + sesiones", m3)]:
    print(f"{nombre}: efecto = {r.params['trat']:.5f} | EE = {r.bse['trat']:.6f} | reducción de varianza = {100 * (1 - (r.bse['trat'] / mpl.bse['trat']) ** 2):.1f} %")
print("correlación Y-sesiones:", round(float(np.corrcoef(ab['convierte'], ab['sesiones_pre'])[0, 1]), 4))
""", "`sesiones_pre` casi no reduce nada (≈ 0.0 %): su correlación con la conversión es casi nula (≈ 0.02) y la reducción es ≈ ρ². Con ambas covariables en la regresión la reducción es ≈ la de `gasto_pre` sola (≈ 9 %). Moraleja: la covariable ayuda solo si **predice** la respuesta.")

md("""
### C2 · Heterogeneidad por dispositivo (15 min)
«No todo efecto promedio cuenta la historia completa»: ¿el tratamiento funciona distinto por dispositivo? El modelo `convierte ~ trat * dispositivo` incluye las **interacciones**; en R se usa `trat * dispositivo`. La referencia es *desktop* (la profesora: «si el PC no me aparece es porque lo considero la base»). **Debe estar preespecificado** si se usará para decidir.
""")
code("""
het = smf.ols("convierte ~ trat * C(dispositivo)", ab).fit()
display(pd.concat([het.params, het.bse, het.pvalues, het.conf_int()], axis=1).set_axis(["coef", "EE", "p", "IC 2.5%", "IC 97.5%"], axis=1).round(4))
nulo = smf.ols("convierte ~ trat + C(dispositivo)", ab).fit()
F, p_F, _ = het.compare_f_test(nulo)
print(f"prueba conjunta de las 2 interacciones: F = {F:.3f}, p = {p_F:.3f}")

filas = []
for col in ["dispositivo", "region"]:
    for lev in sorted(ab[col].unique()):
        sub = ab[ab[col] == lev]
        ca, tb = sub[sub["trat"] == 0]["convierte"], sub[sub["trat"] == 1]["convierte"]
        dd = tb.mean() - ca.mean(); ee = np.sqrt(ca.var() / len(ca) + tb.var() / len(tb))
        filas.append(dict(corte=f"{col}: {lev}", n=len(sub), efecto_pp=100 * dd, IC_inf=100 * (dd - 1.96 * ee), IC_sup=100 * (dd + 1.96 * ee), p=2 * stats.norm.sf(abs(dd / ee))))
display(pd.DataFrame(filas).set_index("corte").round(3))
""")
md("""
Efecto en *desktop* (referencia): **[[hetero.modelo.b_trat|2]] pp**; las interacciones son `+[[hetero.modelo.int_movil|2]]` (móvil, p = [[hetero.modelo.p_movil|2]]) y `[[hetero.modelo.int_tablet|2]]` (tablet, p = [[hetero.modelo.p_tablet|2]]), con **intervalos que contienen el 0** y una prueba conjunta de p = [[hetero.modelo.p_conjunto|2]]: *no hay evidencia estadística suficiente para afirmar que el efecto cambie según el dispositivo* (la conclusión de la profesora). Cuidado con el exceso de lectura: «sin evidencia» **no** es «efecto igual»; los IC de las interacciones son anchos (p. ej. tablet, con solo 1 288 usuarios, va de −4.5 a +2.6 pp).

Fíjate en la tabla de subgrupos: de los 7 cortes, **cuatro** salen «significativos» y tres no, aunque los efectos son casi iguales (≈ 0.6 a 2.4 pp). Lo que cambia es **n**: un subgrupo con pocos usuarios tiene IC ancho. Compara **efectos con sus IC**, no p-valores entre sí.

### ✍️ Ejercicio C2
Haz la misma prueba conjunta de interacción para **`region`**. ¿Qué concluyes? Y si ahora probaras la interacción de **cada** región por separado (7 cortes en total), ¿cuál sería el umbral de Bonferroni para mantener α = 0.05 en toda la familia?
""")
code("""
# tu código aquí
""")
solution("""
hr = smf.ols("convierte ~ trat * C(region)", ab).fit()
F_r, p_r, _ = hr.compare_f_test(smf.ols("convierte ~ trat + C(region)", ab).fit())
print(f"interacción con región: F = {F_r:.3f}, p = {p_r:.3f}")
print("umbral de Bonferroni con 7 cortes:", round(0.05 / 7, 4))
""", "p ≈ 0.90: no hay evidencia de que el efecto varíe por región. Con 7 cortes el umbral de Bonferroni es 0.05/7 = 0.0071; la probabilidad de al menos un falso positivo sin corregir sería 1 − 0.95⁷ ≈ 30 %.")

md("""
### C3 · El experimento roto: `ab_srm.csv` (15 min)
Mismo diseño (50/50), pero ahora el SRM **sí** salta. Hay que **diagnosticar** antes de mirar la conversión.
""")
code("""
srm_roto = srm_test(srm["grupo"])
display(srm_roto.round(4))
ct = pd.crosstab(srm["grupo"], srm["dispositivo"])
display(ct)
display(pd.crosstab(srm["grupo"], srm["dispositivo"], normalize="index").round(4))
for dsp in ct.columns:
    r = stats.chisquare(ct[dsp].values)
    print(f"{dsp}: control {ct.loc['control', dsp]} frente a tratamiento {ct.loc['tratamiento', dsp]} -> chi2 = {r.statistic:.2f}, p = {r.pvalue:.3g}")
print("conversión ingenua por grupo:", srm.groupby("grupo")["convierte"].mean().round(4).to_dict())
""")
md("""
En `ab_srm` hay **6 041** usuarios en control y **5 435** en tratamiento (esperado 5 738 cada uno): `χ² = 32.0`, **p = 1.5 × 10⁻⁸ < 0.001** → se audita y **no se reporta el efecto**. El cruce por dispositivo localiza el problema: en *desktop* los grupos están equilibrados (2 462 frente a 2 382, p = 0.25), pero en **móvil** faltan usuarios de tratamiento (3 579 frente a 3 053, p ≈ 10⁻¹⁰): probablemente el registro o la asignación fallan en el flujo móvil del tratamiento (por ejemplo, el checkout nuevo se cae o no registra en móvil).

¿Por qué **no** reportar la conversión (12.5 % frente a 12.7 %) aunque parezca razonable? Porque el desbalance **no es aleatorio**: los usuarios que se «perdieron» en móvil no son una muestra al azar, y la composición de ambos grupos ya no es comparable (el tratamiento tiene más desktop: 43.8 % frente a 40.8 %).

### ✍️ Ejercicio C3
Mira solo el segmento **desktop** de `ab_srm`, donde el SRM no salta. ¿Se puede reportar ahí la diferencia de conversión? Calcula el efecto y su IC. ¿Qué advertencia agregarías?
""")
code("""
# tu código aquí
""")
solution("""
dk = srm[srm["dispositivo"] == "desktop"]
a_, b_ = dk[dk["grupo"] == "control"]["convierte"], dk[dk["grupo"] == "tratamiento"]["convierte"]
dif_ = b_.mean() - a_.mean(); ee_ = np.sqrt(a_.var() / len(a_) + b_.var() / len(b_))
print(f"desktop: control {a_.mean():.4f} (n = {len(a_)}) | tratamiento {b_.mean():.4f} (n = {len(b_)}) | dif = {100 * dif_:.2f} pp | IC = [{100 * (dif_ - 1.96 * ee_):.2f}; {100 * (dif_ + 1.96 * ee_):.2f}]")
""", "En desktop el SRM no salta, pero **no se puede dar por bueno**: si la asignación falla en móvil, puede fallar de formas que los conteos de desktop no muestran, y el segmento es un análisis **post hoc** no preespecificado. Además, aquí ni el signo coincide con el experimento limpio (≈ −1 pp, IC de −2.9 a +0.9). Como máximo, se reporta como **exploratorio** y con la advertencia de que el experimento debe repetirse tras corregir el problema.")

md("""
### C4 · Peeking y comparaciones múltiples (simulación, 20 min)
**Peeking**: mirar el p-valor varias veces mientras el experimento corre y **detenerlo en cuanto baja de 0.05**. Cada mirada es otra oportunidad de falso positivo. Simulamos experimentos **sin efecto** (A/A con la tasa del control, 8.88 %, 12 000 usuarios por grupo) y miramos el p-valor en `k` momentos equiespaciados.
""")
code("""
rng = np.random.default_rng(RS)
sims, n_final, bloque = 3000, 12000, 100
nb = n_final // bloque
ca = rng.binomial(bloque, pA, size=(sims, nb)).cumsum(axis=1)
cb = rng.binomial(bloque, pA, size=(sims, nb)).cumsum(axis=1)
nn = bloque * np.arange(1, nb + 1)
pp = (ca + cb) / (2 * nn)
zz = (cb / nn - ca / nn) / np.sqrt(2 * pp * (1 - pp) / nn)
rechaza = np.abs(zz) > 1.96
filas = []
for k in (1, 2, 5, 10, 20):
    idx = [int(round(nb * i / k)) - 1 for i in range(1, k + 1)]
    filas.append(dict(miradas=k, falso_positivo_simulado=rechaza[:, idx].any(axis=1).mean(), si_fueran_independientes=1 - 0.95 ** k))
filas.append(dict(miradas="cada 100 usuarios (120)", falso_positivo_simulado=rechaza.any(axis=1).mean(), si_fueran_independientes=1 - 0.95 ** 120))
display(pd.DataFrame(filas).set_index("miradas").round(3))
""")
md("""
Con **1 mirada** el falso positivo es ≈ 5 % (lo correcto). Con **5 miradas** ≈ 14 %, con **10** ≈ 19–20 % (19.1 % en las figuras de la guía y 19.9 % aquí: cambia la semilla) y mirando **cada 100 usuarios** ≈ 39 %. La diapositiva 21 dice «con 10 miradas independientes al 5 %, el falso positivo puede acercarse a 40 %»: 1 − 0.95¹⁰ = 40.1 % **solo si las 10 miradas fueran independientes**, pero las miradas sobre datos **que se acumulan** están muy correlacionadas (cada una incluye los datos de la anterior); por eso el valor real con 10 miradas es ≈ 20 %. Los ≈ 40 % corresponden a mirar de forma casi continua. La conclusión práctica de la clase es correcta (el peeking infla el error); lo impreciso es la cifra para 10 miradas.

**Comparaciones múltiples**: con `m` pruebas independientes al 5 %, `P(≥ 1 falso positivo) = 1 − 0.95^m`: 7 cortes → 30 %; 20 métricas → **64 %**. Se corrige (Bonferroni: usar `α/m`, conservador) o, mejor, se **preespecifica** una métrica primaria.

### ✍️ Ejercicio C4
Con las mismas simulaciones, calcula el falso positivo con **5 miradas** si en cada mirada usas el umbral corregido de Bonferroni `α/5` (|z| > 2.576). ¿Se controla el error? ¿Qué se pierde?
""")
code("""
# tu código aquí
""")
solution("""
z_b = stats.norm.ppf(1 - 0.05 / 5 / 2)
idx5 = [int(round(nb * i / 5)) - 1 for i in range(1, 6)]
fp_b = (np.abs(zz[:, idx5]) > z_b).any(axis=1).mean()
print(f"umbral |z| > {z_b:.3f} en 5 miradas: falso positivo = {fp_b:.3f}")
""", "El falso positivo baja a ≈ 3 % (≤ 5 %): se controla, de forma conservadora porque las miradas están correlacionadas. Lo que se pierde es **potencia**: hace falta más evidencia en cada mirada para declarar un efecto. Métodos secuenciales (O'Brien-Fleming, alpha spending) hacen un mejor compromiso; la regla práctica de la clase es más simple: **fijar la fecha de término y no mirar antes**.")

# ================================================================== PARTE D
md("""
## Parte D · Cuando no se puede aleatorizar: diferencias en diferencias

### D1 · DiD con `did_app_regiones.csv` (25 min)
Una campaña se lanzó solo en la **Región Metropolitana** (64 comunas tratadas; 96 de control) entre marzo y abril de 2026. Se mide `uso_app` mensual. El **DiD** compara el cambio del grupo tratado con el cambio del control: `DiD = (Ȳ_T,post − Ȳ_T,pre) − (Ȳ_C,post − Ȳ_C,pre)`. Su supuesto clave: **tendencias paralelas** (sin el lanzamiento, ambos habrían evolucionado igual).
""")
code("""
tend = did.groupby(["zona", "mes"])["uso_app"].mean().unstack(0).round(3)
tend["brecha RM − otras"] = (tend["RM"] - tend["otras"]).round(3)
display(tend)
fig, ax = plt.subplots(figsize=(6.5, 3.3))
for zona, col in [("RM", "tab:green"), ("otras", "gray")]:
    ax.plot(range(6), tend[zona].values, "o-", color=col, label=zona)
ax.axvline(2.5, color="k", ls=":")
ax.set_xticks(range(6)); ax.set_xticklabels(tend.index, rotation=30, fontsize=8)
ax.set_ylabel("uso promedio de la app"); ax.legend()
plt.show()

cm = did.groupby(["trat", "post"])["uso_app"].mean()
did_manual = (cm[1, 1] - cm[1, 0]) - (cm[0, 1] - cm[0, 0])
print(f"DiD con la tabla 2 × 2: ({cm[1, 1]:.2f} − {cm[1, 0]:.2f}) − ({cm[0, 1]:.2f} − {cm[0, 0]:.2f}) = {did_manual:.4f}")
""")
md("""
La **brecha** RM − otras es casi constante antes (2.93, 2.84, 2.75) y salta a ≈ 8.2–8.8 después: las líneas son paralelas antes del lanzamiento (el gráfico de la profesora). El DiD de la tabla 2 × 2 es **(63.37 − 52.48) − (54.78 − 49.64) = 5.76**. Ahora lo estimamos con regresión: el efecto es el coeficiente de la interacción `trat:post`.
""")
code("""
m_did = smf.ols("uso_app ~ trat * post", did).fit()
grupos = pd.factorize(did["comuna"])[0]
m_did_cl = smf.ols("uso_app ~ trat * post", did).fit(cov_type="cluster", cov_kwds={"groups": grupos})
m_fe = smf.ols("uso_app ~ trat:post + C(comuna) + C(mes)", did).fit()
m_fe_cl = smf.ols("uso_app ~ trat:post + C(comuna) + C(mes)", did).fit(cov_type="cluster", cov_kwds={"groups": grupos})
filas = []
for nombre, r in [("trat * post (como el script)", m_did), ("trat * post, errores por comuna", m_did_cl), ("efectos fijos de comuna y mes", m_fe), ("efectos fijos, errores por comuna", m_fe_cl)]:
    ic = r.conf_int().loc["trat:post"]
    filas.append(dict(modelo=nombre, efecto=r.params["trat:post"], EE=r.bse["trat:post"], IC_inf=ic[0], IC_sup=ic[1], p=r.pvalues["trat:post"]))
display(pd.DataFrame(filas).set_index("modelo").round(4))
""")
md("""
El estimador es el mismo (**5.76**) en todas las versiones y coincide con la diapositiva 30 (IC [4.58; 6.94] con el modelo simple). Lo que cambia es el **error estándar**: el script `lm(uso_app ~ trat * post)` trata las 960 observaciones como independientes, pero **cada comuna aparece 6 veces** (observaciones correlacionadas). Lo correcto es **agrupar los errores por comuna** (cluster); aquí el EE baja de 0.60 a ≈ 0.33 y el IC pasa a [5.10; 6.41]: el modelo simple era, en este caso, **conservador** (la comuna explica mucha variación, que no pertenece al efecto). En otros paneles el sesgo puede ir al revés; la regla es agrupar siempre.

**Tendencias paralelas: ¿cómo se respalda?** (1) Gráfico previo; (2) **coeficiente de pre-tendencia**: con solo los meses previos, la interacción `trat × tiempo` debería ser ≈ 0; (3) **estudio de eventos** (la brecha mes a mes respecto de un mes de referencia).
""")
code("""
meses = sorted(did["mes"].unique())
did["t"] = did["mes"].map({m: i for i, m in enumerate(meses)})
pre = did[did["post"] == 0]
pt = smf.ols("uso_app ~ trat * t", pre).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pre["comuna"])[0]})
print(f"pre-tendencia (trat × t) = {pt.params['trat:t']:.3f} por mes, EE = {pt.bse['trat:t']:.3f}, p = {pt.pvalues['trat:t']:.3f}")

# estudio de eventos: brecha RM − otras por mes, respecto de marzo (el último mes previo)
ref = "2026-03"
for m in meses:
    if m != ref:
        did["ev_" + m[-2:]] = ((did["mes"] == m) & (did["trat"] == 1)).astype(int)
f_ev = "uso_app ~ " + " + ".join("ev_" + m[-2:] for m in meses if m != ref) + " + C(comuna) + C(mes)"
ev = smf.ols(f_ev, did).fit(cov_type="cluster", cov_kwds={"groups": grupos})
filas = [dict(mes=m, efecto=(0 if m == ref else ev.params["ev_" + m[-2:]]), IC_inf=(0 if m == ref else ev.conf_int().loc["ev_" + m[-2:], 0]), IC_sup=(0 if m == ref else ev.conf_int().loc["ev_" + m[-2:], 1])) for m in meses]
display(pd.DataFrame(filas).set_index("mes").round(3))
""")
md("""
La pre-tendencia es ≈ **−0.09 por mes (p = 0.76)**: no hay evidencia de pendientes distintas antes. En el estudio de eventos, enero y febrero quedan en ≈ 0 respecto de marzo y abril–junio saltan a ≈ 5.4–6.1. **Esto respalda, pero no prueba**, las tendencias paralelas (el supuesto trata de lo que *habría* pasado sin el lanzamiento, que no se observa). Además, el DiD supone que **nada más cambió solo en la RM** al mismo tiempo (otras campañas, feriados regionales).

### ✍️ Ejercicio D1
Haz un **placebo**: usa solo los meses previos (enero a marzo) y finge que el lanzamiento fue en **febrero** (`post_falso = 1` para febrero y marzo). El DiD placebo debería ser ≈ 0. ¿Lo es? ¿Qué te dice sobre las tendencias paralelas?
""")
code("""
# tu código aquí
""")
solution("""
pre2 = did[did["post"] == 0].copy()
pre2["post_falso"] = (pre2["mes"] >= "2026-02").astype(int)
pl = smf.ols("uso_app ~ trat * post_falso", pre2).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pre2["comuna"])[0]})
print(f"DiD placebo = {pl.params['trat:post_falso']:.3f} | EE = {pl.bse['trat:post_falso']:.3f} | IC = [{pl.conf_int().loc['trat:post_falso', 0]:.2f}; {pl.conf_int().loc['trat:post_falso', 1]:.2f}] | p = {pl.pvalues['trat:post_falso']:.3f}")
""", "El DiD placebo es ≈ −0.1 (IC que contiene el 0, p grande): sin lanzamiento real no aparece «efecto». Es coherente con tendencias paralelas previas. Un placebo significativo habría sido una señal de alerta.")

md("""
### D2 · Tabla de correspondencia R ↔ Python (10 min)
El script de la profesora (`Clase22_ABtesting.R`) y este notebook hacen lo mismo. Esta tabla ayuda a leer uno desde el otro:

| Qué se hace | R | Python |
|---|---|---|
| Leer datos | `read.csv("ab_checkout.csv")` | `pd.read_csv("ab_checkout.csv")` |
| Variable 0/1 de tratamiento | `as.integer(grupo == "tratamiento")` | `(ab["grupo"] == "tratamiento").astype(int)` |
| Resumen por grupo | `group_by(grupo) %>% summarise(n(), sum(convierte), mean(convierte))` | `ab.groupby("grupo").agg(...)` |
| Tamaño de muestra | `power.prop.test(p1 = 0.09, p2 = 0.105, sig.level = 0.05, power = 0.8)` | fórmula propia (`n_power_prop_test`) o `NormalIndPower().solve_power(h, alpha, power)` (arcoseno) |
| SRM (chi-cuadrado) | `chisq.test(table(grupo), p = c(0.5, 0.5))` | `stats.chisquare(conteos, f_exp=esperados)` |
| Test de dos proporciones | `prop.test(x, n, correct = FALSE)` | `proportions_ztest([xB, xA], [nB, nA])` + `confint_proportions_2indep(..., method="wald")` |
| Modelo lineal | `lm(convierte ~ trat)` | `smf.ols("convierte ~ trat", ab).fit()` |
| Regresión logística | `glm(convierte ~ trat, family = binomial)` | `smf.logit("convierte ~ trat", ab).fit()` |
| Odds ratio con IC | `tidy(logit, conf.int = TRUE, exponentiate = TRUE)` | `np.exp(logit.params)`, `np.exp(logit.conf_int())` |
| Tabla de coeficientes | `broom::tidy(modelo, conf.int = TRUE)` | `modelo.summary()`, `.params`, `.bse`, `.conf_int()` |
| Covarianza y varianza (CUPED) | `cov(y, x) / var(x)` | `np.cov(y, x)[0, 1] / x.var()` |
| Interacción | `lm(convierte ~ trat * dispositivo)` | `smf.ols("convierte ~ trat * C(dispositivo)", ab)` |
| Cruce de tablas | `table(grupo, dispositivo)`, `prop.table(..., margin = 1)` | `pd.crosstab(...)`, `normalize="index"` |
| DiD | `lm(uso_app ~ trat * post)` | `smf.ols("uso_app ~ trat * post", did).fit()` |
| Efectos fijos | `lm(uso_app ~ trat:post + comuna + mes)` | `smf.ols("uso_app ~ trat:post + C(comuna) + C(mes)", did)` |
| Errores agrupados | `sandwich::vcovCL(m, cluster = ~comuna)` | `.fit(cov_type="cluster", cov_kwds={"groups": ...})` |

*Diferencias a recordar*: en R `lm(y ~ x1 * x2)` y en Python `smf.ols("y ~ x1 * x2")` coinciden; las variables categóricas se envuelven en `C(...)` (en R basta con que sean `factor`). `prop.test` reporta χ²; `proportions_ztest` reporta z (χ² = z²).

### ✍️ Ejercicio D2
Traduce a Python: `res <- ab %>% group_by(dispositivo, grupo) %>% summarise(tasa = mean(convierte), usuarios = n())` y arma la figura de barras agrupadas de la parte 4 del script de R.
""")
code("""
# tu código aquí
""")
solution("""
res = ab.groupby(["dispositivo", "grupo"]).agg(tasa=("convierte", "mean"), usuarios=("convierte", "size")).reset_index()
tabla = res.pivot(index="dispositivo", columns="grupo", values="tasa")
display(tabla.round(4))
tabla.plot(kind="bar", figsize=(5, 3))
plt.ylabel("conversión"); plt.show()
""", "`groupby` + `agg` equivale a `group_by` + `summarise`; `pivot` arma la tabla ancha para dibujar barras agrupadas (`geom_col(position = 'dodge')`).")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene precisar (verificado con código)

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Diapositivas 17–18 | `power.prop.test` entrega ≈ 6 131 por grupo (12 262 en total) | `power.prop.test` da **6 137.95 → 6 138** (12 276). 6 131 es la aproximación del arcoseno / h de Cohen (E1). |
| Clase (parámetros) | «P2 = 0.15» | El código usa `p2 = 0.105` (MDE de 1.5 pp); con 0.15 harían falta ≈ 459 por grupo (E1). |
| Diapositiva 21 | con 10 miradas independientes al 5 %, el falso positivo «puede acercarse a 40 %» | 1 − 0.95¹⁰ = 40.1 % exige miradas independientes; con datos acumulados es ≈ 20 % (C4); ≈ 40 % si se mira casi continuamente. |
| Clase | «p-valor bajito → el efecto parece real y relevante» | El p-valor dice que el azar no lo explica; la relevancia sale del IC frente al MDE (B2). |
| Clase | «p-valor del SRM grande → completamente aleatorizado» | Solo «sin evidencia de desbalance en los conteos»; no es prueba de aleatorización (B1). |
| Clase | en el contraste «la idea es que sean distintas» | El test es neutro; buscar rechazar H₀ es la raíz del peeking y de las comparaciones múltiples. |
| Diapositiva 15 | «el test de proporciones y la regresión dicen lo mismo» | Misma conclusión y mismo efecto (lineal), pero distintos EE, y el logit está en otra escala (B4). |
| Diapositivas 29–30 y script | `lm(uso_app ~ trat * post)` sin agrupar | Cada comuna se repite 6 veces: hay que agrupar errores por comuna (D1: IC [5.10; 6.41] frente a [4.58; 6.94]). |
| Diapositiva 29 | «si antes ya corrían en direcciones distintas, el DiD queda a medio camino» | El DiD queda **sesgado** en la diferencia de tendencias; no hay «medio camino» definido (E4). |
| Diapositivas 5 y 6 | «Estima lo ocurrido si ¿cambiamos…?»; «27 %:,» | Erratas de redacción. Las cifras históricas (Lind 1747, estreptomicina 1948, Obama 2008, Bing 12 %) no se verificaron aquí. |

**Transcripción con ruido**: «Wardrails» = guardrails, «picking» = peeking, «Coupet / Cuper» = CUPED, «DIT» = DiD, «Yesu-i» = Yᵢ, «Tao Tzu Y» = τᵢ, «ATG» = ATE, «VIM» = Bing, «SAS» = SaaS, «cita» = `srm_test`, «Valleciana» = Bayesiana. El recreo (≈ 1:21:15 del video) no está en el video.
""")
code("""
# E1 · tamaño de muestra: de dónde salen 6 131 y 6 138, y qué pasa con p2 = 0.15
print("E1 · power.prop.test:", round(n_power_prop_test(0.09, 0.105), 2), "| arcoseno (statsmodels):", round(float(NormalIndPower().solve_power(proportion_effectsize(0.105, 0.09), alpha=0.05, power=0.8)), 2),
      "| con p2 = 0.15:", round(n_power_prop_test(0.09, 0.15), 1))
""")
code("""
# E2 · peeking: 10 miradas independientes frente a 10 miradas sobre datos acumulados (de C4)
idx10 = [int(round(nb * i / 10)) - 1 for i in range(1, 11)]
print(f"E2 · si fueran independientes: {1 - 0.95 ** 10:.3f} | simulado con datos acumulados: {rechaza[:, idx10].any(axis=1).mean():.3f} | mirando cada 100 usuarios: {rechaza.any(axis=1).mean():.3f}")
""")
code("""
# E3 · DiD: errores simples frente a agrupados por comuna
print(f"E3 · EE trat:post  simple = {m_did.bse['trat:post']:.3f} | agrupado por comuna = {m_did_cl.bse['trat:post']:.3f} | IC simple [{m_did.conf_int().loc['trat:post', 0]:.2f}; {m_did.conf_int().loc['trat:post', 1]:.2f}] | IC agrupado [{m_did_cl.conf_int().loc['trat:post', 0]:.2f}; {m_did_cl.conf_int().loc['trat:post', 1]:.2f}]")
""")
code("""
# E4 · si las tendencias NO fueran paralelas, el DiD se sesga: se crea una tendencia extra de +1.5 por mes solo en la RM
trucado = did.copy()
trucado["uso_app"] = trucado["uso_app"] + 1.5 * trucado["t"] * trucado["trat"]
mt = smf.ols("uso_app ~ trat * post", trucado).fit()
print(f"E4 · DiD real = {m_did.params['trat:post']:.2f} | con una tendencia diferencial de +1.5/mes en la RM = {mt.params['trat:post']:.2f} (sesgo = {mt.params['trat:post'] - m_did.params['trat:post']:+.2f}, aunque el «efecto» del lanzamiento no cambió)")
""")
md("""
Resultados: **E1** `6137.95`, `6130.59` y `459.3`; **E2** 0.401 (independientes) frente a ≈ 0.20 (simulado) y ≈ 0.39 (cada 100 usuarios); **E3** EE simple 0.601 frente a 0.333 agrupado; **E4** el DiD sube de 5.76 a ≈ 10.3: un sesgo de ≈ +4.5 cuando la RM ya tenía una tendencia propia distinta de la del control; el DiD **atribuye** al lanzamiento esa diferencia de tendencia.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista y debes entregar una **recomendación de lanzamiento** del nuevo checkout (`ab_checkout.csv`). Responde con código:

1. **Diseño**: ¿cuántos usuarios por grupo hacían falta para detectar +1.5 pp sobre 9 % (α = 0.05, potencia 80 %)? ¿Tuvo el experimento suficiente tamaño?
2. **Chequeo**: SRM (¿hay alarma?) y balance de `gasto_pre` y `sesiones_pre` (|d|).
3. **Efecto principal** en conversión: diferencia, IC 95 % (Wald), p-valor y odds ratio. ¿Supera el MDE de 1.5 pp en el extremo bajo del IC?
4. **Métrica secundaria / guardrail**: compara `monto_compra` **por usuario** (incluye los ceros) entre grupos con un **test t de Welch** y con un **bootstrap** de la diferencia de medias (2 000 remuestreos, semilla fija). ¿El ingreso por usuario sube? Con **dos** métricas (conversión y monto), ¿qué corrección de α usarías?
5. **Precisión**: ajusta el efecto por `gasto_pre`, `sesiones_pre` y `dispositivo`. ¿Cambia el efecto? ¿Y el IC?
6. **Sin código**: con una frase para un gerente, ¿lanzas? ¿Qué dos advertencias agregarías (heterogeneidad, peeking, duración, validez externa)?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1. diseño
n_nec = int(np.ceil(n_power_prop_test(0.09, 0.105)))
print(f"1. n necesario por grupo = {n_nec:,} | n real = {nA:,} y {nB:,} -> {'suficiente' if min(nA, nB) >= n_nec else 'insuficiente'}")

# 2. chequeos
print("2. SRM p =", round(float(srm_checkout['p_valor'].iloc[0]), 3), "| máx |d| =", round(float(max(abs(smd(ab, 'gasto_pre')), abs(smd(ab, 'sesiones_pre')))), 4))

# 3. efecto principal
print(f"3. dif = {100 * d:.2f} pp | IC = [{100 * lo:.2f}; {100 * hi:.2f}] | p = {p_val:.2g} | OR = {oratio:.3f} | extremo bajo > 1.5 pp: {100 * lo > 1.5}")

# 4. monto por usuario: Welch + bootstrap, y corrección de Bonferroni para 2 métricas
mc_, mt_ = ab.loc[ab["trat"] == 0, "monto_compra"].to_numpy(), ab.loc[ab["trat"] == 1, "monto_compra"].to_numpy()
welch = stats.ttest_ind(mt_, mc_, equal_var=False)
rng4 = np.random.default_rng(RS)
boot = np.array([rng4.choice(mt_, len(mt_)).mean() - rng4.choice(mc_, len(mc_)).mean() for _ in range(2000)])
print(f"4. monto medio por usuario: control {mc_.mean():,.0f} | tratamiento {mt_.mean():,.0f} | dif = {mt_.mean() - mc_.mean():,.0f} | Welch p = {welch.pvalue:.3g} | IC bootstrap 95 % = [{np.percentile(boot, 2.5):,.0f}; {np.percentile(boot, 97.5):,.0f}] | Bonferroni: α = 0.05/2 = 0.025")

# 5. ajuste
ma_ = smf.ols("convierte ~ trat + gasto_pre + sesiones_pre + C(dispositivo)", ab).fit()
print(f"5. efecto ajustado = {100 * ma_.params['trat']:.2f} pp | IC = [{100 * ma_.conf_int().loc['trat', 0]:.2f}; {100 * ma_.conf_int().loc['trat', 1]:.2f}] (ancho {100 * (ma_.conf_int().loc['trat', 1] - ma_.conf_int().loc['trat', 0]):.2f} frente a {100 * (hi - lo):.2f})")
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: (1) hacían falta **6 138** por grupo y el experimento tuvo ≈ 11 000: sobrado. (2) SRM p = 0.64 y |d| máximo ≈ 0.01: sin alarma. (3) +1.94 pp, IC [1.15; 2.72], p = 1.4 × 10⁻⁶, OR 1.24; el extremo bajo del IC (1.15) **no** supera 1.5 pp: el efecto es positivo con certeza, pero no se puede asegurar que alcance el MDE. (4) El monto medio por usuario sube (de ≈ 4 771 a ≈ 5 946, con un IC bootstrap que no incluye el 0); con dos métricas se usa α = 0.025 o, mejor, se declara una primaria (conversión) y la otra como secundaria. (5) Con covariables el efecto es 1.99 pp y el IC se estrecha de 1.57 a 1.50 pp de ancho. (6) **Recomendación**: lanzar, con la advertencia de que el efecto podría ser de solo 1.2 pp (menor que el MDE) y que se debe vigilar `guardrails` que aquí no existen en los datos (latencia, errores, reclamos): el experimento no los mide. Otras cautelas: la heterogeneidad por dispositivo y región es exploratoria (no hay evidencia de diferencias); la duración del experimento y la estacionalidad no se conocen; y los resultados valen para la población y el período del experimento.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Por qué una correlación entre «adoptar el cambio» y «mejorar» no prueba causalidad? ¿Qué hace la aleatorización?
2. ¿Qué es el contrafactual y por qué el efecto individual no se observa? ¿Qué es el ATE y cómo se estima en un A/B?
3. Define p-valor, intervalo de confianza 95 %, α y potencia en una frase cada uno. ¿Qué **no** es un p-valor?
4. ¿Qué pasa con n si el MDE se divide por 2? ¿Y si la potencia sube de 80 % a 90 %? ¿De dónde sale 6 138?
5. ¿Qué es un SRM y cómo se prueba? ¿Qué haces si p < 0.001? ¿Un p = 0.64 prueba que la aleatorización fue perfecta?
6. ¿Qué es el peeking y por qué infla el falso positivo? ¿Por qué 1 − 0.95¹⁰ sobreestima el error con 10 miradas?
7. ¿Para qué sirven las covariables y CUPED en un experimento aleatorizado? ¿Qué **no** corrigen?
8. ¿Cuándo se usa DiD, cuál es su supuesto clave, cómo lo respaldas y qué errores estándar usas?

<details><summary><b>Respuestas</b></summary>

1. Porque los expuestos pueden ser distintos de los no expuestos (confusores): la diferencia mezcla el efecto con esas diferencias previas. La aleatorización hace la asignación independiente de cualquier característica (observada o no) y deja grupos comparables en promedio.
2. El contrafactual es el resultado que habría tenido la misma unidad con la otra condición; de cada unidad se observa solo una. ATE = E[Y(1) − Y(0)]; en un A/B se estima con la diferencia de promedios entre tratamiento y control.
3. p-valor: probabilidad, si H₀ fuera cierta, de un resultado al menos así de extremo. IC 95 %: rango de efectos compatibles con los datos (en repeticiones, el 95 % de estos intervalos contiene el valor verdadero). α: falso positivo tolerado. Potencia: probabilidad de detectar un efecto real del tamaño del MDE. El p-valor **no** es la probabilidad de que H₀ sea cierta ni mide el tamaño del efecto.
4. n se multiplica por ≈ 4 (n ∝ 1/MDE²); pasar de 80 % a 90 % cuesta ≈ 34 % más. 6 138 sale de la fórmula de `power.prop.test` con p₁ = 0.09, p₂ = 0.105, α = 0.05 y potencia 80 % (6 131 es la aproximación del arcoseno).
5. Sample Ratio Mismatch: los conteos por grupo no coinciden con la proporción planeada. Se prueba con chi-cuadrado sobre los conteos. Si p < 0.001 se audita la instrumentación y no se reporta el efecto. Un p = 0.64 solo indica que no hay evidencia de desbalance en los conteos.
6. Mirar el p-valor repetidamente y detenerse cuando baja de 0.05. Cada mirada es otra oportunidad de falso positivo. La fórmula 1 − 0.95¹⁰ supone miradas independientes; con datos acumulados están correlacionadas, y el valor real con 10 miradas es ≈ 20 %.
7. Ganan precisión (menor error estándar, IC más estrecho). No corrigen un SRM, una mala aleatorización ni el sesgo; y solo se ajusta por variables medidas antes del tratamiento.
8. Cuando no se puede aleatorizar. Supuesto: tendencias paralelas (sin el cambio, tratados y controles habrían evolucionado igual). Se respalda con el gráfico previo, un estudio de eventos y un placebo; los errores estándar se agrupan por la unidad que se repite (comuna).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clases 23 y 24 · Estadística bayesiana (Prof. Maritza Márquez). Revisa `Guias/apuntes/C22_Ejercicios_Cuaderno_ABTesting.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
