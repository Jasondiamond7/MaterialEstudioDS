"""Genera el notebook guiado de estadística bayesiana (Clase 23, primera sesión).

Uso normal:      python notebook_c23.py             -> Guias/notebooks/C23_Bayesiana.ipynb
Modo de prueba:  C23_TEST=1 python notebook_c23.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c23_numeros.json (lo escribe figuras_c23.py) con marcas [[ruta|decimales]].
Solo usa numpy, pandas, matplotlib y scipy (no hace falta PyMC: la clase no usa MCMC; eso es de la Clase 24).
"""
import json
import os
import re
from pathlib import Path

TEST = os.environ.get("C23_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C23_Bayesiana_test.ipynb" if TEST else "C23_Bayesiana.ipynb")
N = json.loads((HERE / "c23_numeros.json").read_text(encoding="utf-8"))
cells = []


def val(m):
    obj = N
    for k in m.group(1).split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    d = m.group(2)
    if d is None:
        return str(obj)
    return f"{float(obj):.{int(d)}f}"


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
# Clase 23 · Estadística bayesiana: fundamentos
**Diplomado Data Science** · Prof. Maritza Márquez · primera de dos sesiones (la Clase 24 trae predicción, Monte Carlo y MCMC)

Material de apoyo:
- `Guias/C23_Guia_Bayesiana.pdf` — guía visual de la clase.
- `Guias/apuntes/C23_Apuntes_Bayesiana.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C23_Ejercicios_Cuaderno_Bayesiana.pdf` — ejercicios en papel (antes o después de este notebook).
- Material de la profesora (carpeta «Clase 23, 24 …»): `Clase 23.pdf` (diapositivas), `Clase23.ipynb` (práctica en R) y `Clase 23 - Práctica de Estadística Bayesiana en R.pdf`.
- Transcripción: `VideosClases/Clase 23  estadisticas bayesianas.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib` y `scipy`. Si no los tienes:
```
pip install numpy pandas matplotlib scipy
```
No hace falta **PyMC** ni otra librería bayesiana: en esta sesión todo tiene fórmula cerrada (Beta–Binomial) o se resuelve con una grilla o con simulación simple. El MCMC es de la Clase 24.

**Datos**: la clase no usa archivos. Trabajamos con una **prueba diagnóstica** (prevalencia 1 %, sensibilidad 95 %, especificidad 90 %) y con **14 aprobados de 20** estudiantes y priori Beta(2, 2). Solo la parte D1 (puente con la Clase 22) lee `ab_checkout.csv`, el archivo de la Clase 22.

### Objetivos
1. Distinguir el enfoque **frecuentista** del **bayesiano** e interpretar la probabilidad como incertidumbre.
2. Usar el **teorema de Bayes** (con frecuencias naturales) y entender que P(enfermo | positivo) ≠ sensibilidad.
3. Construir un modelo **priori × verosimilitud → posteriori** y resolverlo con el modelo conjugado **Beta–Binomial**.
4. Resumir la posteriori: **media, MAP, intervalo creíble** y probabilidades como P(θ > 0.6 | y).
5. Ver el efecto de la **priori** y del **tamaño de muestra**, y la **actualización secuencial**.
6. Comparar **intervalo creíble** con **intervalo de confianza** (y medir su cobertura).
7. Reproducir los resultados de la profesora, traducir **R ↔ Python** y reconocer lo impreciso del material.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Frecuencias naturales y Bayes |
| A | 15 | A2 · La posteriori como nueva priori (dos tests) |
| A | 10 | A3 · Prevalencia y valor de un test positivo |
| B | 20 | B1 · Beta–Binomial: la práctica de la profesora |
| B | 15 | B2 · Posteriori ∝ priori × verosimilitud, con una grilla |
| B | 15 | B3 · P(θ > 0.6 | y): exacta y por simulación |
| B | 15 | B4 · El ejercicio de la diapositiva 16 |
| C | 15 | C1 · Sensibilidad a la priori |
| C | 15 | C2 · Más datos y actualización secuencial |
| C | 20 | C3 · Intervalo creíble frente a intervalo de confianza |
| C | 10 | C4 · MAP, media, mediana y HPD |
| D | 20 | D1 · Puente con la Clase 22: A/B bayesiano |
| D | 15 | D2 · Otras conjugadas |
| D | 10 | D3 · Distribución predictiva (adelanto de la Clase 24) |
| D | 10 | D4 · Tabla de correspondencia R ↔ Python |
| — | 15 | Alertas del material, verificadas con código |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`).
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook. Las simulaciones usan semilla fija; con otras versiones de las librerías pueden variar en la última cifra.
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
from scipy import integrate, optimize, stats

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
RS = 2026   # semilla para todo lo aleatorio
print("numpy:", np.__version__, "| scipy:", scipy.__version__)
""")
md("""
### Mini-repaso de lo que se usa hoy
- **`stats.beta(a, b)`** es la distribución Beta: `.pdf` = `dbeta`, `.cdf` = `pbeta`, `.ppf` = `qbeta`, `.rvs` = `rbeta`, `.sf` = `1 - pbeta`, `.mean()`, `.std()`, `.interval(0.95)`.
- **`stats.binom`** es la binomial (`.pmf` = `dbinom`) y **`stats.betabinom`** la beta-binomial (distribución predictiva).
- **Verosimilitud**: la misma fórmula de la binomial, pero mirada como función de θ con los datos fijos.
- Todo lo aleatorio usa `np.random.default_rng(semilla)`: así los resultados se repiten (los números aleatorios de R y de NumPy **no** coinciden aunque la semilla sea la misma).
""")

# ================================================================== PARTE A
md("""
## Parte A · Teorema de Bayes

### A1 · Frecuencias naturales y Bayes (15 min)
Pregunta: una enfermedad tiene prevalencia 1 %; la prueba tiene sensibilidad 95 % y especificidad 90 %. Si una persona da positivo, ¿cuál es la probabilidad de que esté enferma? Es la **probabilidad condicional** P(E | +). Se resuelve **contando personas**: se reparte una población de 10 000 y se mira, entre quienes dan positivo, cuántos están enfermos. Luego se comprueba con la fórmula `P(E | +) = P(+ | E)·P(E) / P(+)`.
""")
code("""
prev, sens, esp = 0.01, 0.95, 0.90
n = 10_000

enfermos, sanos = round(n * prev), round(n * (1 - prev))
vp = round(enfermos * sens)                  # verdaderos positivos
fp = round(sanos * (1 - esp))                # falsos positivos
tabla = pd.DataFrame({"da +": [vp, fp], "da −": [enfermos - vp, sanos - fp]}, index=["enfermos", "sanos"])
tabla.loc["total"] = tabla.sum()
display(tabla)
print(f"Entre los {vp + fp} positivos, {vp} están enfermos -> P(E | +) = {vp / (vp + fp):.4f}")

# con la fórmula (es el código de la profesora, traducido de R)
p_pos = sens * prev + (1 - esp) * (1 - prev)          # evidencia P(+)
p_enf_pos = sens * prev / p_pos
print(f"P(+) = {p_pos:.4f} | P(E | +) = {p_enf_pos:.7f}   (R: 0.0875576)")
""")
md("""
Solo **8.8 %** de quienes dan positivo están enfermos, aunque la prueba detecta al 95 % de los enfermos. La razón: hay muchísimos más sanos que enfermos, y el 10 % de falsos positivos sobre 9 900 sanos (990) supera a los 95 verdaderos positivos. La **prevalencia es la información previa** y el test positivo la **actualiza**, pero no la borra. Moraleja de la clase: *nunca se interpreta un test sin considerar la prevalencia*. Y se ve por qué **P(E | +) ≠ P(+ | E)**: la sensibilidad es la segunda; lo que quiere saber el paciente es la primera.

### ✍️ Ejercicio A1
Una prueba de detección tiene sensibilidad 92 % y especificidad 97 %; la prevalencia es 3 %. Con 20 000 personas, completa el reparto y calcula P(enfermo | positivo) de las dos maneras (contando y con la fórmula). ¿Cuántos falsos positivos hay por cada verdadero positivo?
""")
code("""
# tu código aquí
""")
solution("""
prev2, sens2, esp2, n2 = 0.03, 0.92, 0.97, 20_000
enf2, san2 = round(n2 * prev2), round(n2 * (1 - prev2))
vp2, fp2 = round(enf2 * sens2), round(san2 * (1 - esp2))
print(f"enfermos {enf2}, sanos {san2}, verdaderos + {vp2}, falsos + {fp2}")
print(f"contando: {vp2 / (vp2 + fp2):.4f}")
print(f"fórmula: {sens2 * prev2 / (sens2 * prev2 + (1 - esp2) * (1 - prev2)):.4f}")
print(f"falsos positivos por cada verdadero positivo: {fp2 / vp2:.2f}")
""", "Verdaderos positivos 552, falsos positivos 582: P(E | +) ≈ **0.487**. Hay ≈ 1.05 falsos positivos por cada verdadero positivo: incluso con 97 % de especificidad y 3 % de prevalencia, un positivo equivale casi a «cara o cruz».")

md("""
### A2 · La posteriori como nueva priori: dos tests (15 min)
El teorema de Bayes se puede **encadenar**: la posteriori de hoy es la priori de mañana. Si la misma persona se hace un segundo test **independiente** (condicionado a su estado de salud) y vuelve a dar positivo, se usa 0.0876 como nueva prevalencia. Lo comprobamos con una simulación de un millón de personas.
""")
code("""
def bayes_pos(prior, sens=0.95, esp=0.90):
    return sens * prior / (sens * prior + (1 - esp) * (1 - prior))


p1 = bayes_pos(0.01)
p2 = bayes_pos(p1)
p3 = bayes_pos(p2)
print(f"tras 1 positivo: {p1:.4f} | tras 2: {p2:.4f} | tras 3: {p3:.4f}")

# simulación: un millón de personas, dos tests independientes dado el estado de salud
rng = np.random.default_rng(RS)
m = 1_000_000
enf = rng.random(m) < 0.01
t1 = np.where(enf, rng.random(m) < 0.95, rng.random(m) < 0.10)
t2 = np.where(enf, rng.random(m) < 0.95, rng.random(m) < 0.10)
print(f"simulado: P(E | +) = {enf[t1].mean():.4f} | P(E | + y +) = {enf[t1 & t2].mean():.4f}")
""")
md("""
La simulación reproduce 0.088 y ≈ 0.47 (el valor exacto es 0.4769; la diferencia es ruido de simulación). Con tres positivos seguidos la probabilidad llega a ≈ 0.90. Es la misma idea que usaremos con la Beta: **actualizar la creencia con cada dato nuevo**.

### ✍️ Ejercicio A2
¿Cuántos tests positivos **seguidos** (independientes) hacen falta para que la probabilidad de estar enfermo supere 0.99 con la prevalencia de 1 %? Usa un ciclo con `bayes_pos`.
""")
code("""
# tu código aquí
""")
solution("""
p, k = 0.01, 0
while p <= 0.99:
    p = bayes_pos(p)
    k += 1
    print(f"tras {k} positivo(s): {p:.5f}")
""", "Se necesitan **5** positivos seguidos: tras 3 la probabilidad es 0.896, tras 4 es 0.988 (aún no supera 0.99) y tras 5 es 0.9987.")

md("""
### A3 · Prevalencia y valor de un test positivo (10 min)
¿Cómo cambia P(E | +) con la prevalencia y con la especificidad? Es la curva de la figura de la guía.
""")
code("""
prevs = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
tab = pd.DataFrame({f"esp = {e:.2f}": bayes_pos(prevs, 0.95, e) for e in (0.90, 0.95, 0.99)}, index=[f"{p:.1%}" for p in prevs])
display(tab.round(3))

x = np.logspace(-3, np.log10(0.6), 200)
for e in (0.90, 0.95, 0.99):
    plt.plot(x * 100, bayes_pos(x, 0.95, e) * 100, label=f"especificidad {e:.0%}")
plt.xscale("log"); plt.xlabel("prevalencia (%)"); plt.ylabel("P(enfermo | positivo) (%)"); plt.legend(); plt.show()
""")
md("""
### ✍️ Ejercicio A3
Con sensibilidad 95 % y especificidad 90 %, ¿qué prevalencia hace que un positivo signifique 50 % de probabilidad de estar enfermo? Resuélvelo numéricamente con `optimize.brentq` y comprueba con una fórmula cerrada (pista: la probabilidad es 0.5 cuando `sens·prev = (1 − esp)(1 − prev)`).
""")
code("""
# tu código aquí
""")
solution("""
num = optimize.brentq(lambda p: bayes_pos(p) - 0.5, 1e-6, 0.99)
cerrada = (1 - 0.90) / (0.95 + (1 - 0.90))
print(f"numérica: {num:.4f} | cerrada: {cerrada:.4f}")
""", "La prevalencia crítica es (1 − esp)/(sens + 1 − esp) = 0.10/1.05 ≈ **0.0952** (9.5 %). Con prevalencias menores, un positivo es más probablemente un falso positivo que un verdadero.")

# ================================================================== PARTE B
md("""
## Parte B · El modelo Beta–Binomial

### B1 · La práctica de la profesora, en Python (20 min)
Un modelo bayesiano tiene tres ingredientes: **priori** p(θ), **verosimilitud** p(y | θ) y **posteriori** p(θ | y) ∝ p(y | θ)·p(θ). Queremos estimar la proporción θ de estudiantes que aprueban un módulo. Datos: **y = 14 aprobados de n = 20**. Priori: **θ ~ Beta(2, 2)** (media 0.5). Como los datos son binomiales y la priori es Beta, la posteriori es **Beta(α + y, β + n − y)** (modelo conjugado).
""")
code("""
y, n = 14, 20
alpha, beta = 2, 2

alpha_post = alpha + y
beta_post = beta + n - y
print("alpha_post =", alpha_post, "| beta_post =", beta_post)       # R: 16 y 8

post = stats.beta(alpha_post, beta_post)
media_post = alpha_post / (alpha_post + beta_post)
ic_95 = post.ppf([0.025, 0.975])                                      # qbeta(c(0.025, 0.975), ...)
print(f"media posteriori = {media_post:.7f}   (R: 0.6666667)")
print(f"IC creíble 95 % = [{ic_95[0]:.7f}, {ic_95[1]:.7f}]   (R: 0.4708083 0.8362364)")
""")
md("""
Reproducimos los números de la profesora: Beta(16, 8), media 0.6667 e IC creíble [0.4708; 0.8362]. La lectura bayesiana es directa: *dados los datos y la priori elegida, la probabilidad posteriori de que θ esté entre 0.47 y 0.84 es 0.95* (en clase se leyó «0.83»: el extremo superior es 0.8362, que se redondea a 0.84). **No** significa que el 95 % de los datos esté dentro del intervalo: el intervalo se refiere al **parámetro**.

La media posteriori (0.667) es un **promedio ponderado** de la media de la priori (0.5) y de la proporción observada (0.7), con pesos α + β = 4 y n = 20:
""")
code("""
w = (alpha + beta) / (alpha + beta + n)
print(f"peso de la priori = {w:.4f} | 0.5·w + 0.7·(1 − w) = {0.5 * w + (y / n) * (1 - w):.4f}")
print(f"ancho del IC: priori {np.diff(stats.beta(alpha, beta).ppf([0.025, 0.975]))[0]:.3f} -> posteriori {np.diff(ic_95)[0]:.3f}")
""")
md("""
Observa que **la incertidumbre bajó** (el intervalo se estrechó de 0.81 a 0.37) y que la estimación se acerca a 0.70 pero «suavizada» por la priori.

### ✍️ Ejercicio B1
Adherencia a un tratamiento: priori Beta(4, 6) y 27 de 50 pacientes adhieren. Calcula los parámetros posteriori, la media, la moda (MAP) `(α − 1)/(α + β − 2)`, la desviación y el IC creíble del 95 %.
""")
code("""
# tu código aquí
""")
solution("""
a1, b1 = 4 + 27, 6 + 50 - 27
d1 = stats.beta(a1, b1)
print(f"Beta({a1}, {b1}) | media {d1.mean():.4f} | MAP {(a1 - 1) / (a1 + b1 - 2):.4f} | sd {d1.std():.4f} | IC {np.round(d1.ppf([0.025, 0.975]), 4)}")
""", "Posteriori Beta(31, 29): media 0.5167, MAP 0.5172, desviación 0.0640, IC creíble [0.3912; 0.6411]. Es la misma estructura que la del ejemplo de clase.")

md("""
### B2 · Posteriori ∝ priori × verosimilitud, con una grilla (15 min)
La profesora explicó el símbolo **∝ («proporcional a»)**: se ignora temporalmente la constante de normalización p(y) = ∫ p(y | θ)p(θ)dθ, que solo sirve para que la posteriori integre 1. Podemos **verlo**: multiplicamos priori × verosimilitud en una grilla de θ, normalizamos dividiendo por el área y comparamos con la fórmula conjugada.
""")
code("""
theta = np.linspace(0.0005, 0.9995, 4000)
priori = stats.beta(alpha, beta).pdf(theta)
verosim = stats.binom.pmf(y, n, theta)                  # verosimilitud: la binomial como función de θ
no_norm = priori * verosim                              # posteriori no normalizada
evidencia = integrate.trapezoid(no_norm, theta)         # p(y) = área bajo la curva
grilla = no_norm / evidencia
cerrada = stats.beta(alpha_post, beta_post).pdf(theta)
print(f"evidencia p(y) por grilla = {evidencia:.5f}")
from scipy.special import betaln, comb
print(f"evidencia p(y) por fórmula = {comb(n, y) * np.exp(betaln(alpha_post, beta_post) - betaln(alpha, beta)):.5f}   [C(n, y)·B(α+y, β+n−y)/B(α, β)]")
print(f"diferencia máxima grilla vs Beta(16, 8) = {np.abs(grilla - cerrada).max():.1e}")
print(f"área bajo la verosimilitud (no es 1, no es una densidad en θ) = {integrate.trapezoid(verosim, theta):.4f}")

fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
ax[0].plot(theta, priori, color="tab:blue"); ax[0].set_title("priori Beta(2, 2)")
ax[1].plot(theta, verosim, color="tab:orange"); ax[1].set_title("verosimilitud (14 de 20)")
ax[2].plot(theta, priori, "--", color="tab:blue", label="priori"); ax[2].plot(theta, grilla, lw=5, alpha=0.35, color="tab:green", label="grilla normalizada")
ax[2].plot(theta, cerrada, color="k", label="Beta(16, 8)"); ax[2].legend(); ax[2].set_title("posteriori")
for a_ in ax: a_.set_xlabel("θ")
plt.tight_layout(); plt.show()
""")
md("""
La grilla y la fórmula coinciden (diferencia ≈ 10⁻¹⁴): **normalizar el producto da la Beta(16, 8)**. La verosimilitud **no** integra 1 sobre θ (su área es ≈ 0.048): mide *compatibilidad* entre los datos y cada θ, no es una distribución de θ (diapositiva 20).

### ✍️ Ejercicio B2
Repite la grilla con una priori **uniforme** Beta(1, 1) y los mismos datos. ¿Qué distribución es la posteriori? Comprueba que la grilla coincide con ella y calcula la evidencia p(y). (Pista: con priori uniforme p(y) = 1/(n + 1).)
""")
code("""
# tu código aquí
""")
solution("""
no_norm_u = stats.beta(1, 1).pdf(theta) * verosim
ev_u = integrate.trapezoid(no_norm_u, theta)
print(f"evidencia = {ev_u:.5f} | 1/(n + 1) = {1 / (n + 1):.5f}")
print(f"max |grilla − Beta(15, 7)| = {np.abs(no_norm_u / ev_u - stats.beta(15, 7).pdf(theta)).max():.1e}")
""", "La posteriori es **Beta(15, 7)** (1 + 14, 1 + 6) y la evidencia vale 1/21 ≈ 0.0476: con priori uniforme, todos los resultados posibles (0, 1, …, 20) son igualmente probables antes de ver los datos.")

md("""
### B3 · P(θ > 0.6 | y): exacta y por simulación (15 min)
La parte 4 de la práctica calcula `P(θ > 0.6 | y)` con `rbeta` y `mean(theta_sim > 0.6)`: **simulación de Monte Carlo** (se saca una muestra de la posteriori y se cuenta la fracción que cumple la condición). También se puede calcular **exacta** con la cola de la Beta: `1 − pbeta(0.6, 16, 8)`.
""")
code("""
exacta = post.sf(0.6)                                   # 1 - pbeta(0.6, 16, 8)
rng = np.random.default_rng(123)
S = 10_000
theta_sim = post.rvs(S, random_state=rng)               # rbeta(S, 16, 8)
mc = (theta_sim > 0.6).mean()
se = np.sqrt(exacta * (1 - exacta) / S)                 # error estándar de Monte Carlo
print(f"exacta = {exacta:.7f} | simulada (S = {S:,}) = {mc:.4f} | error estándar ≈ {se:.4f}")
print(f"el 0.7581 de la práctica en R está a {abs(0.7581 - exacta) / se:.2f} errores estándar del valor exacto")
""")
md("""
La práctica en R entrega **0.7581** y la tabla de la práctica (con `pbeta`) 0.7627. No hay contradicción: la simulación tiene un **error de Monte Carlo** ≈ 0.0043 con 10 000 valores (el 0.7581 de R está a ≈ 1.1 errores estándar del valor exacto). Los números aleatorios de R y de NumPy no coinciden aunque la semilla sea la misma; por eso aquí se compara con el valor **exacto**.

### ✍️ Ejercicio B3
¿Cómo baja el error de Monte Carlo al aumentar S? Calcula la estimación para S = 100, 1 000, 10 000, 100 000 y 1 000 000 y su error absoluto frente al valor exacto. ¿Cuánto hay que multiplicar S para dividir el error por 10?
""")
code("""
# tu código aquí
""")
solution("""
rng3 = np.random.default_rng(RS)
filas = []
for S_ in (100, 1_000, 10_000, 100_000, 1_000_000):
    est = (post.rvs(S_, random_state=rng3) > 0.6).mean()
    filas.append(dict(S=S_, estimacion=est, error_abs=abs(est - exacta), error_estandar=np.sqrt(exacta * (1 - exacta) / S_)))
display(pd.DataFrame(filas).round(5))
""", "El error estándar de Monte Carlo es √(p(1 − p)/S): baja con √S. Para **dividir el error por 10 hay que multiplicar S por 100**. Con 10 000 simulaciones, el tercer decimal ya es ruido.")

md("""
### B4 · El ejercicio de la diapositiva 16 (15 min)
> Una plataforma educativa tiene una tasa histórica de finalización de cursos de 40 %. En una nueva cohorte de 20 estudiantes, 12 finalizan. 1) ¿Interpretación frecuentista directa? 2) ¿Qué información previa razonable podríamos incorporar? 3) ¿La evidencia de esta cohorte parece mejorar la tasa histórica? 4) ¿Qué incertidumbre queda?

La profesora dijo que lo harían en R, pero no se resolvió en el video. Lo resolvemos aquí. **Priori razonable**: la tasa histórica 40 % con el peso de 20 cursos previos: **Beta(8, 12)** (media 0.40, α + β = 20).
""")
code("""
y16, n16 = 12, 20
# 1) frecuentista
p_hat = y16 / n16
se_w = np.sqrt(p_hat * (1 - p_hat) / n16)
prueba = stats.binomtest(y16, n16, 0.40, alternative="greater")
ci_cp = stats.binomtest(y16, n16, 0.40).proportion_ci(confidence_level=0.95, method="exact")   # IC bilateral exacto
print(f"frecuentista: p̂ = {p_hat:.2f} (+{(p_hat - 0.40) * 100:.0f} pp) | Wald [{p_hat - 1.96 * se_w:.3f}, {p_hat + 1.96 * se_w:.3f}] | exacto [{ci_cp.low:.3f}, {ci_cp.high:.3f}]")
print(f"prueba binomial unilateral H0: θ = 0.40 -> p = {prueba.pvalue:.4f}")

# 2-4) bayesiano con dos prioris
filas = []
for nom, a_, b_ in [("histórica Beta(8, 12)", 8, 12), ("uniforme Beta(1, 1)", 1, 1)]:
    d_ = stats.beta(a_ + y16, b_ + n16 - y16)
    filas.append(dict(priori=nom, posteriori=f"Beta({a_ + y16}, {b_ + n16 - y16})", media=d_.mean(), sd=d_.std(), li=d_.ppf(0.025), ls=d_.ppf(0.975), p_mayor_040=d_.sf(0.40)))
display(pd.DataFrame(filas).round(3))
""")
md("""
Con la **priori histórica** (que pesa como 20 cursos) la posteriori es Beta(20, 20): media 0.50 y P(θ > 0.40 | y) ≈ 0.90, con un intervalo todavía amplio [0.35; 0.65]. Con la **uniforme**, la media sube a 0.59 y P(θ > 0.40 | y) ≈ 0.96. El frecuentista obtiene p̂ = 0.60 y un p-valor unilateral ≈ 0.057 (poca evidencia frente a 0.05). **Las tres lecturas dicen que la tasa probablemente mejoró, pero que 20 estudiantes dejan mucha incertidumbre**, y la conclusión depende de cuánto pese la priori.

### ✍️ Ejercicio B4
Con la priori histórica, ¿cuántos estudiantes más (con la misma proporción 60 %) harían falta para que P(θ > 0.40 | y) supere 0.99? Busca con un ciclo sobre n.
""")
code("""
# tu código aquí
""")
solution("""
for n_ in range(20, 400, 10):
    y_ = round(0.6 * n_)
    p = stats.beta(8 + y_, 12 + n_ - y_).sf(0.40)
    if p > 0.99:
        print(f"n = {n_} (y = {y_}) -> P(θ > 0.40 | y) = {p:.4f}")
        break
""", "Con ≈ 50 estudiantes (30 finalizan) la probabilidad ya supera 0.99 (0.992). Más datos reducen la dependencia de la priori y concentran la posteriori alrededor de 0.60.")

# ================================================================== PARTE C
md("""
## Parte C · Priori, tamaño de muestra, intervalos y resúmenes

### C1 · Sensibilidad a la priori: la tabla de la profesora (15 min)
La parte 5 de la práctica compara tres prioris: **Uniforme** Beta(1, 1), **Moderada centrada en 0.5** Beta(2, 2) y **Optimista** Beta(8, 2). Reproducimos su tabla.
""")
code("""
def tabla_prioris(y, n, prioris):
    filas = []
    for nombre, a_, b_ in prioris:
        d_ = stats.beta(a_ + y, b_ + n - y)
        filas.append(dict(previa=nombre, alpha_post=a_ + y, beta_post=b_ + n - y, media_posteriori=d_.mean(),
                          LI_95=d_.ppf(0.025), LS_95=d_.ppf(0.975), prob_mayor_06=d_.sf(0.6), peso_priori=(a_ + b_) / (a_ + b_ + n)))
    return pd.DataFrame(filas)


prioris = [("Uniforme", 1, 1), ("Moderada centrada en 0.5", 2, 2), ("Optimista", 8, 2)]
res14 = tabla_prioris(14, 20, prioris)
display(res14.round(7))
""")
md("""
Coincide con la tabla de R (medias 0.6818, 0.6667 y 0.7333; `prob_mayor_06` 0.7998, 0.7627 y 0.9430). La priori **Optimista** Beta(8, 2) equivale a **10 observaciones previas** (80 % de éxitos): con n = 20 pesa un **tercio** de la información, y por eso empuja tanto la media y la probabilidad. «Optimista» no es «poco informativa».

### ✍️ Ejercicio C1 (es el «Ejercicio para los estudiantes» de la profesora)
Cambia los datos a **8 aprobados de 20** y repite el análisis. Preguntas de la profesora: 1) ¿la posterior cambia mucho respecto al caso anterior? 2) ¿qué previa tiene mayor influencia? 3) ¿cómo comunicarías el resultado a una persona no estadística?
""")
code("""
# tu código aquí
""")
solution("""
res8 = tabla_prioris(8, 20, prioris)
display(res8.round(4))
print("cambio de la media (Moderada):", round(res14.loc[1, "media_posteriori"], 3), "->", round(res8.loc[1, "media_posteriori"], 3))
""", "1) **Sí**: con 8 de 20 la media posteriori con Beta(2, 2) baja de 0.667 a 0.417 y P(θ > 0.6 | y) cae de 0.76 a 0.035. 2) La que más influye es la **Optimista** Beta(8, 2): su media sube a 0.533 (frente a 0.409 y 0.417), y P(θ > 0.6 | y) es 0.23 en lugar de 0.035: cuando los datos contradicen la priori, la priori informativa pesa más. 3) «Con lo observado, lo más razonable es que entre 22 % y 62 % de los estudiantes apruebe; es poco probable (< 4 %) que supere el 60 %; con una creencia previa muy optimista subiría a 23 %, pero esa creencia habría que justificarla.»")

md("""
### C2 · Más datos y actualización secuencial (15 min)
Dos propiedades clave: (1) con **más datos las prioris distintas convergen**; (2) la actualización **por lotes** da lo mismo que procesar todo junto: *la posteriori de un lote es la priori del siguiente*. Es la ventaja con datos que llegan «por goteo».
""")
code("""
# (1) convergencia: 70 % de éxitos y n creciente
filas = []
for n_ in (10, 20, 100, 1000):
    y_ = round(0.7 * n_)
    filas.append(dict(n=n_, media_uniforme=(1 + y_) / (2 + n_), media_optimista=(8 + y_) / (10 + n_), peso_optimista=10 / (10 + n_)))
display(pd.DataFrame(filas).round(4))

# (2) por lotes frente a todo junto (priori Beta(2, 2))
lotes = [(4, 3), (6, 4), (5, 3), (5, 4)]           # (n, éxitos) de cada llegada
a_, b_ = 2, 2
for i, (nn, yy) in enumerate(lotes, 1):
    a_, b_ = a_ + yy, b_ + nn - yy
    d_ = stats.beta(a_, b_)
    print(f"lote {i}: n = {nn}, y = {yy} -> Beta({a_}, {b_}) | media {d_.mean():.4f} | IC [{d_.ppf(0.025):.3f}, {d_.ppf(0.975):.3f}]")
print("todo junto:", (2 + sum(l[1] for l in lotes), 2 + sum(l[0] - l[1] for l in lotes)), "= Beta(16, 8)")
""")
md("""
Con 70 % de éxitos las medias de Beta(1, 1) y Beta(8, 2) pasan de 0.667 y 0.750 (n = 10) a 0.700 y 0.701 (n = 1 000): **la priori se «olvida»** con datos. Y por lotes se llega a la misma Beta(16, 8) que con los 20 datos juntos.

### ✍️ Ejercicio C2
Simula 60 observaciones Bernoulli con θ = 0.3 (semilla fija) y actualiza **una por una** desde la priori Beta(2, 2). Dibuja la media posteriori y su IC creíble 95 % en función del número de observaciones y confirma que el resultado final coincide con el de actualizar todo de una vez.
""")
code("""
# tu código aquí
""")
solution("""
datos = np.random.default_rng(RS).random(60) < 0.3
a_, b_ = 2, 2
medias, los, his = [], [], []
for x in datos:
    a_, b_ = a_ + x, b_ + (1 - x)
    d_ = stats.beta(a_, b_)
    medias.append(d_.mean()); los.append(d_.ppf(0.025)); his.append(d_.ppf(0.975))
plt.plot(range(1, 61), medias, color="tab:green"); plt.fill_between(range(1, 61), los, his, alpha=0.2, color="tab:green")
plt.axhline(0.3, ls=":", color="k"); plt.xlabel("observaciones"); plt.ylabel("media posteriori e IC 95 %"); plt.show()
junto = stats.beta(2 + datos.sum(), 2 + 60 - datos.sum())
print(f"una por una: Beta({a_}, {b_}) | todo junto: Beta({junto.args[0]}, {junto.args[1]})")
""", "Las dos formas dan la misma posteriori. El intervalo se estrecha rápido al principio y más lento después (≈ 1/√n); la media oscila y se acerca a la proporción observada.")

md("""
### C3 · Intervalo creíble frente a intervalo de confianza (20 min)
Diapositivas 27–28: el **creíble** es una probabilidad posteriori sobre el parámetro; el de **confianza** es un procedimiento de largo plazo. Para verlo, medimos la **cobertura frecuentista exacta** (la fracción de muestras cuyo intervalo contiene a θ, sumando sobre todos los resultados posibles de la binomial) para varios intervalos con n = 20.
""")
code("""
def ic_wald(y, n, z=1.959964):
    p = y / n; s = np.sqrt(p * (1 - p) / n)
    return max(0, p - z * s), min(1, p + z * s)

def ic_wilson(y, n, z=1.959964):
    p = y / n; den = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

def ic_clopper(y, n):
    return (0.0 if y == 0 else stats.beta.ppf(0.025, y, n - y + 1)), (1.0 if y == n else stats.beta.ppf(0.975, y + 1, n - y))

def ic_creible(y, n, a=1, b=1):
    return stats.beta.ppf(0.025, a + y, b + n - y), stats.beta.ppf(0.975, a + y, b + n - y)

def cobertura(metodo, theta, n):
    ys = np.arange(n + 1)
    pm = stats.binom.pmf(ys, n, theta)
    return float(sum(p for y_, p in zip(ys, pm) if metodo(y_, n)[0] <= theta <= metodo(y_, n)[1]))


metodos = {"Wald": ic_wald, "Wilson": ic_wilson, "Clopper–Pearson": ic_clopper, "creíble Beta(1, 1)": ic_creible}
print("Los mismos datos (14 de 20):")
for nom, m_ in metodos.items():
    lo, hi = m_(14, 20)
    print(f"  {nom:20s} [{lo:.4f}, {hi:.4f}]")
print("  creíble Beta(2, 2)   ", np.round(ic_creible(14, 20, 2, 2), 4), "<- la de la profesora")

ths = np.linspace(0.04, 0.96, 231)
cob = pd.DataFrame({nom: [cobertura(m_, t, 20) for t in ths] for nom, m_ in metodos.items()}, index=ths)
print("\\ncobertura media sobre θ (nominal 0.95):")
display((cob.mean() * 100).round(2).to_frame("cobertura media (%)").join((cob.min() * 100).round(2).to_frame("mínima (%)")))
print("cobertura en θ = 0.65:", {k: round(cobertura(m_, 0.65, 20) * 100, 2) for k, m_ in metodos.items()})
cob.plot(figsize=(7, 3)); plt.axhline(0.95, color="k", ls=":"); plt.ylim(0.5, 1.0); plt.xlabel("θ verdadero"); plt.ylabel("cobertura"); plt.show()
""")
md("""
Con 14 de 20 los cinco intervalos se **parecen** (por ejemplo, [0.478; 0.854] el creíble uniforme y [0.481; 0.855] el de Wilson), pero **se interpretan distinto**. En cobertura: el de **Wald** (el de «estimador ± 1.96 EE») cubre en promedio solo ≈ 89 % con n = 20 y llega a 56 % cerca de los extremos; Wilson y el creíble con priori uniforme cubren ≈ 95.4 %; Clopper–Pearson es conservador (≈ 97.6 %). Que un creíble tenga buena cobertura **no** es su definición (su definición es la probabilidad posteriori), pero es una buena propiedad práctica con priori poco informativa.

### ✍️ Ejercicio C3
Simula 20 000 muestras de n = 20 con θ = 0.10 y mide la cobertura de Wald y del creíble Beta(1, 1). Compara con la cobertura exacta de `cobertura(...)`.
""")
code("""
# tu código aquí
""")
solution("""
rng_c = np.random.default_rng(RS)
ys_ = rng_c.binomial(20, 0.10, 20_000)
sim = {nom: np.mean([m_(y_, 20)[0] <= 0.10 <= m_(y_, 20)[1] for y_ in ys_]) for nom, m_ in [("Wald", ic_wald), ("creíble", ic_creible)]}
print("simulada:", {k: round(float(v), 4) for k, v in sim.items()})
print("exacta:  ", {"Wald": round(cobertura(ic_wald, 0.10, 20), 4), "creíble": round(cobertura(ic_creible, 0.10, 20), 4)})
""", "Con θ = 0.10 y n = 20, Wald cubre ≈ 88 % (muy por debajo de 95 %) y el creíble ≈ 96 %: con proporciones pequeñas y n modesto Wald falla. La simulación coincide con la cobertura exacta salvo ruido de Monte Carlo.")

md("""
### C4 · MAP, media, mediana y HPD (10 min)
La posteriori completa es el resultado; un solo número la resume pero **oculta la incertidumbre** (pregunta de la diapositiva 30). Comparamos resúmenes y dos tipos de intervalo creíble de 95 %: **por colas** (el de `qbeta`) y **HPD** (el más corto).
""")
code("""
def hpd(a, b, masa=0.95):
    d = stats.beta(a, b)
    f = lambda l: d.ppf(l + masa) - d.ppf(l)
    l = optimize.minimize_scalar(f, bounds=(0, 1 - masa), method="bounded", options={"xatol": 1e-10}).x
    lo, hi = d.ppf(l), d.ppf(l + masa)
    if f(0) <= f(l) + 1e-12:                       # moda en el borde: el intervalo parte en 0
        lo, hi = d.ppf(0), d.ppf(masa)
    return lo, hi


def resumen(a, b):
    d = stats.beta(a, b)
    moda = (a - 1) / (a + b - 2) if a > 1 and b > 1 else (0.0 if a <= 1 else 1.0)
    return dict(posteriori=f"Beta({a}, {b})", MAP=moda, mediana=d.median(), media=d.mean(), colas=np.round(d.interval(0.95), 3), HPD=np.round(hpd(a, b), 3))


display(pd.DataFrame([resumen(16, 8), resumen(2, 10), resumen(1, 11)], index=["14 de 20, Beta(2, 2)", "1 de 10, uniforme", "0 de 10, uniforme"]).round(4))
""")
md("""
Con Beta(16, 8) (casi simétrica) media, mediana y MAP casi coinciden (0.667, 0.671, 0.682) y los dos intervalos son parecidos. Con la asimétrica Beta(2, 10) se separan: MAP 0.100 < mediana 0.148 < media 0.167, y el HPD [0.006; 0.368] es más corto que el de colas [0.023; 0.413]. Con 0 éxitos de 10 y priori uniforme, el **MAP es 0** (= máxima verosimilitud) mientras que la media es 0.083: decir «θ = 0» sería arriesgado.

### ✍️ Ejercicio C4
Con la posteriori Beta(3, 14) (2 éxitos de 15 con priori uniforme), calcula MAP, media, mediana, IC por colas y HPD, y la probabilidad P(θ > 0.25 | y). ¿Cuál resumen elegirías para informar un valor típico y por qué?
""")
code("""
# tu código aquí
""")
solution("""
display(pd.DataFrame([resumen(3, 14)]).round(4))
print("P(θ > 0.25 | y) =", round(stats.beta(3, 14).sf(0.25), 4))
""", "MAP 0.133, mediana 0.164, media 0.176; IC por colas [0.040; 0.383] y HPD [0.025; 0.352]; P(θ > 0.25 | y) ≈ 0.197. Con una posteriori asimétrica no hay «el» estimador correcto: se informa la mediana (o la media) **junto con el intervalo**, aclarando cuál se usó.")

# ================================================================== PARTE D
md("""
## Parte D · Puentes

### D1 · Puente con la Clase 22: A/B bayesiano (20 min)
En la Clase 22 el experimento del checkout se analizó con la diferencia de tasas, un IC y un p-valor. La estadística bayesiana lo contesta con probabilidades directas: **P(B > A | datos)** y **P(mejora > MDE | datos)**. *(La clase 23 no hizo este ejemplo: es una extensión de este material que usa los datos de la Clase 22.)* Con priori uniforme en cada grupo, la posteriori de cada tasa es Beta(1 + x, 1 + n − x).
""")
code("""
def buscar(nombre):
    for base in [Path.cwd(), *Path.cwd().parents]:
        for p in list(base.glob(f"Clase 22*/{nombre}")) + list(base.glob(nombre)):
            return p
    raise FileNotFoundError(f"No encuentro {nombre}: abre el notebook dentro de la carpeta del curso (necesita la carpeta de la Clase 22).")


ab = pd.read_csv(buscar("ab_checkout.csv"))
ab["trat"] = (ab["grupo"] == "tratamiento").astype(int)
nA, nB = int((ab["trat"] == 0).sum()), int((ab["trat"] == 1).sum())
xA, xB = int(ab.loc[ab["trat"] == 0, "convierte"].sum()), int(ab.loc[ab["trat"] == 1, "convierte"].sum())
print(f"control {xA}/{nA} = {xA / nA:.4%} | tratamiento {xB}/{nB} = {xB / nB:.4%} | diferencia = {(xB / nB - xA / nA) * 100:.3f} pp")

rng = np.random.default_rng(RS)
S = 200_000
tA = stats.beta(1 + xA, 1 + nA - xA).rvs(S, random_state=rng)
tB = stats.beta(1 + xB, 1 + nB - xB).rvs(S, random_state=rng)
mejora = tB - tA
print(f"P(B > A) = {(tB > tA).mean():.5f}  (todas las simulaciones)")
print(f"mejora: media {mejora.mean() * 100:.3f} pp | IC creíble 95 % = [{np.percentile(mejora, 2.5) * 100:.2f}, {np.percentile(mejora, 97.5) * 100:.2f}] pp")
print(f"P(mejora > 1.0 pp) = {(mejora > 0.01).mean():.4f} | P(mejora > 1.5 pp) = {(mejora > 0.015).mean():.4f}   (MDE de la Clase 22: 1.5 pp)")

# comparación con el IC de Wald de la Clase 22
pA, pB = xA / nA, xB / nB
se = np.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
print(f"IC de Wald (Clase 22) = [{(pB - pA - 1.96 * se) * 100:.2f}, {(pB - pA + 1.96 * se) * 100:.2f}] pp")
""")
md("""
Con 22 000 usuarios y priori plana, el **IC creíble [1.15; 2.72] pp casi coincide con el de Wald [1.15; 2.72]** de la Clase 22, y P(B > A) es prácticamente 1 (el p-valor era 1.4 × 10⁻⁶). La diferencia está en lo que se puede decir: la Clase 22 solo podía observar que el extremo bajo del IC (1.15) no superaba el MDE de 1.5 pp; aquí se calcula directamente **P(mejora > 1.5 pp | datos) ≈ 0.86**. La decisión sigue necesitando el costo, los guardrails y el criterio de negocio: una probabilidad alta de ser mejor no es lo mismo que una mejora grande.

Con **pocos datos** la relación con el p-valor se ve mejor. Con priori plana, **P(B > A) ≈ 1 − p unilateral**, y ambos se leen distinto:
""")
code("""
rs = np.random.default_rng(RS + 1)
cy = ab.loc[ab["trat"] == 0, "convierte"].to_numpy()[rs.permutation(nA)]
ty = ab.loc[ab["trat"] == 1, "convierte"].to_numpy()[rs.permutation(nB)]
filas = []
for m_ in (200, 500, 1000, 2000, 5000):
    a_, b_ = int(cy[:m_].sum()), int(ty[:m_].sum())
    pp = (a_ + b_) / (2 * m_)
    z = (b_ / m_ - a_ / m_) / np.sqrt(pp * (1 - pp) * 2 / m_)
    da = stats.beta(1 + a_, 1 + m_ - a_).rvs(60_000, random_state=rs)
    db = stats.beta(1 + b_, 1 + m_ - b_).rvs(60_000, random_state=rs)
    filas.append(dict(por_grupo=m_, dif_pp=(b_ - a_) / m_ * 100, P_B_mayor_A=(db > da).mean(), uno_menos_p_unilateral=1 - stats.norm.sf(z)))
display(pd.DataFrame(filas).round(4))
""")
md("""
### ✍️ Ejercicio D1
En un A/B pequeño, A convirtió 48 de 400 y B convirtió 66 de 400. Calcula P(B > A), el IC creíble de la mejora y P(mejora > 2 pp) con priori uniforme. Compáralo con el p-valor unilateral de la prueba z de dos proporciones. ¿Por qué 0.97 de probabilidad de que B sea mejor **no** basta para lanzarlo?
""")
code("""
# tu código aquí
""")
solution("""
rg = np.random.default_rng(1)
da = stats.beta(49, 353).rvs(400_000, random_state=rg); db = stats.beta(67, 335).rvs(400_000, random_state=rg)
d_ = db - da
pp = 114 / 800; z = (66 / 400 - 48 / 400) / np.sqrt(pp * (1 - pp) * 2 / 400)
print(f"P(B > A) = {(db > da).mean():.3f} | IC creíble = [{np.percentile(d_, 2.5) * 100:.2f}, {np.percentile(d_, 97.5) * 100:.2f}] pp | P(mejora > 2 pp) = {(d_ > 0.02).mean():.3f}")
print(f"z = {z:.2f} | p unilateral = {stats.norm.sf(z):.4f} | 1 − p = {1 - stats.norm.sf(z):.4f}")
""", "P(B > A) ≈ 0.965 (casi 1 − p unilateral = 0.966, z = 1.82). El IC creíble de la mejora es [−0.35; 9.34] pp: **incluye el 0**, o sea que con tan pocos datos la mejora podría ser nula o muy grande. P(mejora > 2 pp) ≈ 0.84. Una probabilidad alta de que B sea *algo* mejor no dice cuánto mejor es, ni cuál es el costo de equivocarse ni cómo van los guardrails.")

md("""
### D2 · Otras conjugadas (15 min)
La diapositiva 29 reúne cuatro pares conjugados: **Binomial–Beta**, **Bernoulli–Beta**, **Poisson–Gamma** y **Exponencial–Gamma** (Gamma en forma–tasa). «No hace falta deducirlas: se buscan en una tabla». Pero conviene **verificar** una: normalizamos priori × verosimilitud en una grilla y comparamos con la fórmula de la tabla. Poisson–Gamma: λ | y ~ Gamma(a + S, b + m), con S = Σyᵢ y m = número de observaciones.
""")
code("""
cts = np.array([3, 1, 4, 2, 5])                      # conteos
a0, b0 = 2, 1                                        # priori Gamma(forma 2, tasa 1)
S_, m_ = cts.sum(), len(cts)
lam = np.linspace(0.01, 8, 4000)
verosim = np.prod([stats.poisson.pmf(c, lam) for c in cts], axis=0)
no_norm = stats.gamma(a0, scale=1 / b0).pdf(lam) * verosim
grilla = no_norm / integrate.trapezoid(no_norm, lam)
cerrada = stats.gamma(a0 + S_, scale=1 / (b0 + m_)).pdf(lam)        # scipy usa escala = 1 / tasa
print(f"Poisson–Gamma: Gamma({a0 + S_}, {b0 + m_}) | media {(a0 + S_) / (b0 + m_):.4f} | máx |grilla − fórmula| = {np.abs(grilla - cerrada).max():.1e}")

tiempos = np.array([0.8, 1.9, 0.4, 1.2])
xe = np.linspace(0.01, 4, 4000)
ver_e = np.prod([stats.expon.pdf(t, scale=1 / xe) for t in tiempos], axis=0)
nn_e = stats.gamma(2, scale=1).pdf(xe) * ver_e
g_e = nn_e / integrate.trapezoid(nn_e, xe)
c_e = stats.gamma(2 + len(tiempos), scale=1 / (1 + tiempos.sum())).pdf(xe)
print(f"Exponencial–Gamma: Gamma({2 + len(tiempos)}, {1 + tiempos.sum():.1f}) | máx |grilla − fórmula| = {np.abs(g_e - c_e).max():.1e}")
""")
md("""
Las dos fórmulas se verifican (diferencias ≤ 10⁻⁵, error de la grilla). **Cuidado con la parametrización**: en R, `dgamma(x, shape, rate)`; en scipy `stats.gamma(a, scale=1/tasa)`. Con la tasa mal escrita, la «Gamma» sale con otra media.

### ✍️ Ejercicio D2
Fallas por semana en una línea de producción: 2, 0, 3, 1, 4, 2. Priori Gamma(3, 2) (media 1.5). Calcula la posteriori, su media, el IC creíble del 95 % y P(λ > 2) exacta y por simulación (100 000 valores).
""")
code("""
# tu código aquí
""")
solution("""
fallas = np.array([2, 0, 3, 1, 4, 2])
ap, bp = 3 + fallas.sum(), 2 + len(fallas)
dg = stats.gamma(ap, scale=1 / bp)
sim = dg.rvs(100_000, random_state=np.random.default_rng(5))
print(f"Gamma({ap}, {bp}) | media {dg.mean():.4f} | IC {np.round(dg.interval(0.95), 3)} | P(λ > 2): exacta {dg.sf(2):.4f}, simulada {(sim > 2).mean():.4f}")
""", "Posteriori Gamma(15, 8), media 1.875 fallas por semana, IC creíble [1.049; 2.936] y P(λ > 2) ≈ 0.3675 exacta (0.3673 simulada). Los datos (media 2.0) y la priori (media 1.5) se combinan en 1.875.")

md("""
### D3 · Distribución predictiva (adelanto de la Clase 24) (10 min)
La Clase 24 estudia la **predicción bayesiana**; aquí solo un adelanto. Si la posteriori de θ es Beta(a, b), el número de éxitos en m observaciones **nuevas** sigue una **beta-binomial(m, a, b)**: mezcla la variabilidad del azar con la incertidumbre sobre θ. Con la posteriori Beta(16, 8), ¿cuántos de los próximos 20 estudiantes aprobarán?
""")
code("""
pred = stats.betabinom(20, 16, 8)
ks = np.arange(21)
print(f"esperado = {pred.mean():.3f} (= 20·16/24) | sd predictiva = {pred.std():.3f} | sd si θ fuera exactamente 2/3 = {stats.binom(20, 16 / 24).std():.3f}")
print(f"P(al menos 15 de 20) = {pred.sf(14):.4f}")

# comprobación por simulación en dos pasos: θ ~ posteriori y luego y ~ Binomial(20, θ)
rng = np.random.default_rng(RS)
th = stats.beta(16, 8).rvs(200_000, random_state=rng)
y_nuevo = rng.binomial(20, th)
print(f"simulado: media {y_nuevo.mean():.3f} | P(≥ 15) = {(y_nuevo >= 15).mean():.4f}")
plt.bar(ks - 0.2, pred.pmf(ks), width=0.4, label="beta-binomial (predictiva)")
plt.bar(ks + 0.2, stats.binom(20, 16 / 24).pmf(ks), width=0.4, label="Binomial(20, 2/3) (θ fijo)")
plt.xlabel("aprobados entre 20 nuevos"); plt.legend(); plt.show()
""")
md("""
La predictiva es **más dispersa** que la Binomial con θ fijo (desviación 2.80 frente a 2.11 con θ fijo en 2/3, para 20 nuevos) porque arrastra la incertidumbre sobre θ. Es la razón por la que la Clase 24 insiste en predecir con la **distribución posterior completa** y no con una estimación puntual.

### ✍️ Ejercicio D3
Con la posteriori Beta(31, 29) (adherencia a un tratamiento), llegan 10 pacientes nuevos. Calcula el esperado y P(al menos 6 adhieren), y compara con la Binomial(10, 31/60).
""")
code("""
# tu código aquí
""")
solution("""
p_ = stats.betabinom(10, 31, 29)
print(f"esperado {p_.mean():.3f} | P(≥ 6) predictiva = {p_.sf(5):.4f} | con θ fijo = {stats.binom(10, 31 / 60).sf(5):.4f}")
""", "Esperado 5.167 y P(al menos 6) ≈ 0.425 (predictiva) frente a 0.419 con θ fijo: parecidas en el centro; la predictiva tiene colas algo más pesadas.")

md("""
### D4 · Tabla de correspondencia R ↔ Python (10 min)
| Qué | R (práctica de la profesora) | Python |
|---|---|---|
| Densidad Beta | `dbeta(theta, a, b)` | `stats.beta.pdf(theta, a, b)` |
| Probabilidad acumulada | `pbeta(0.6, a, b)` | `stats.beta.cdf(0.6, a, b)` |
| Cola `P(θ > 0.6)` | `1 - pbeta(0.6, a, b)` | `stats.beta.sf(0.6, a, b)` |
| Cuantiles (IC creíble) | `qbeta(c(0.025, 0.975), a, b)` | `stats.beta.ppf([0.025, 0.975], a, b)` o `.interval(0.95)` |
| Simulación | `rbeta(S, a, b)` | `stats.beta.rvs(a, b, size=S, random_state=rng)` |
| Semilla | `set.seed(123)` | `rng = np.random.default_rng(123)` (**no** da los mismos números que R) |
| Binomial | `dbinom(y, n, p)` | `stats.binom.pmf(y, n, p)` |
| Beta-binomial | (paquete extra) | `stats.betabinom(m, a, b)` |
| Media posteriori | `a / (a + b)` | `a / (a + b)` o `stats.beta(a, b).mean()` |
| Gamma (forma, tasa) | `dgamma(x, shape, rate)` | `stats.gamma.pdf(x, shape, scale=1/rate)` |
| Probabilidad por simulación | `mean(theta_sim > 0.6)` | `(theta_sim > 0.6).mean()` |
| Data frame de resultados | `data.frame(...)`, `rbind(...)` | `pd.DataFrame(...)`, `pd.concat(...)` |
| Gráfico | `plot`, `lines`, `abline`, `legend` | `plt.plot`, `plt.axvline`, `plt.legend` |

*Diferencias a recordar*: en R, `for (i in 1:nrow(df))` recorre filas; en Python se usa `for ... in zip(...)` o una lista de tuplas. Las funciones `d`/`p`/`q`/`r` de R (densidad, acumulada, cuantil, aleatorio) son los métodos `.pdf`/`.cdf`/`.ppf`/`.rvs` de scipy.

### ✍️ Ejercicio D4
Traduce a Python `prioris <- data.frame(nombre = c("A", "B"), alpha = c(1, 3), beta = c(1, 7))` y, con 18 éxitos de 40, calcula para cada fila `media_posteriori`, `LI_95`, `LS_95` y `prob_mayor_04` (= `1 - pbeta(0.4, ...)`).
""")
code("""
# tu código aquí
""")
solution("""
prioris_d4 = pd.DataFrame({"nombre": ["A", "B"], "alpha": [1, 3], "beta": [1, 7]})
prioris_d4["a_post"] = prioris_d4["alpha"] + 18
prioris_d4["b_post"] = prioris_d4["beta"] + 40 - 18
prioris_d4["media_posteriori"] = prioris_d4["a_post"] / (prioris_d4["a_post"] + prioris_d4["b_post"])
prioris_d4["LI_95"] = stats.beta.ppf(0.025, prioris_d4["a_post"], prioris_d4["b_post"])
prioris_d4["LS_95"] = stats.beta.ppf(0.975, prioris_d4["a_post"], prioris_d4["b_post"])
prioris_d4["prob_mayor_04"] = stats.beta.sf(0.4, prioris_d4["a_post"], prioris_d4["b_post"])
display(prioris_d4.round(4))
""", "Los métodos de scipy aceptan vectores (pandas Series), así que no hace falta el ciclo `for` de R. Posterioris Beta(19, 23) y Beta(21, 29); medias 0.4524 y 0.4200.")

# ================================================================== ALERTAS
md("""
## Alertas del material: qué conviene precisar (verificado con código)

| Dónde | Qué dice | Qué pasa realmente (verificado) |
|---|---|---|
| Clase (densidad Beta) | «gamma de alfa **más** gamma de beta» en el denominador | Es el **producto** Γ(α)Γ(β); la diapositiva 23 lo escribe bien (E1). |
| Clase y diapositiva 23 | «α y β son éxitos y fracasos previos» frente a «no son dos aprobados y dos reprobados» | Como pseudo-conteos funcionan: Beta(2, 2) + datos = Beta(1, 1) + datos con 1 éxito y 1 fracaso más (E2). |
| Práctica en R | «la posteriori = sensibilidad · priori **sobre la posteriori**» | El denominador es la **evidencia** P(+) = 0.1085 (E3). |
| Clase | «los valores 0.025 y 0.975 son cuartiles» | Son **cuantiles** (percentiles 2.5 y 97.5), no cuartiles (E3). |
| Práctica (R) | `mean(theta_sim > 0.6)` = 0.7581 | El valor exacto es 0.7627; la diferencia es error de Monte Carlo ≈ 0.0043 (B3 y E4). |
| Clase y práctica | IC «entre 0.47 y 0.83 / 0.84» | El extremo exacto es 0.8362 (se redondea a 0.84). Hay otros IC creíbles válidos, como el HPD [0.481; 0.845] (C4, E3). |
| Clase | «95 % de las veces ese parámetro quede dentro del intervalo» (frecuentista) | El parámetro es fijo; lo que varía entre muestras es el **intervalo** (C3). |
| Clase | «función de densidad de la binomial» | Es una función de **masa**; como función de θ se llama verosimilitud y no integra 1 (B2). |
| Práctica (R) | priori «Optimista» Beta(8, 2) | Es informativa: equivale a 10 observaciones (33 % del peso con n = 20) (C1, E4). |
| Diapositiva 8 | «P(θ > 0.6 | datos) = 0.82» | Es un ejemplo ilustrativo; el valor de la práctica es 0.76 (E4). |
| Diapositiva 32 | práctica «en R y Python» | Solo se entrega el archivo en R; este notebook añade la versión en Python. |

**Transcripción con ruido**: «vallenciana / valleciana / vallesiana / valleseano / vayaciana» = bayesiana, «Valles» = Bayes, «cobi» = COVID, «cuartilas / cubeta» = cuantiles / qbeta, «Colap» = Colab. El recreo (≈ 1:24:30 del video) no está en el video.
""")
code("""
# E1 · la constante de la densidad Beta: producto Γ(α)Γ(β), no suma
from scipy.special import gamma as G
a_, b_, x_ = 16, 8, 0.6
f_prod = G(a_ + b_) / (G(a_) * G(b_)) * x_ ** (a_ - 1) * (1 - x_) ** (b_ - 1)
f_suma = G(a_ + b_) / (G(a_) + G(b_)) * x_ ** (a_ - 1) * (1 - x_) ** (b_ - 1)
print(f"E1 · stats.beta.pdf = {stats.beta.pdf(x_, a_, b_):.4f} | con producto = {f_prod:.4f} | con suma = {f_suma:.3e}")
""")
code("""
# E2 · α y β como pseudo-conteos: Beta(2, 2) + 14/20 = Beta(1, 1) + 15 éxitos y 7 fracasos
print("E2 · Beta(2, 2) + 14 de 20:", (2 + 14, 2 + 6), "| Beta(1, 1) + 15 éxitos y 7 fracasos:", (1 + 15, 1 + 7), "| mismo resultado:", (2 + 14, 2 + 6) == (1 + 15, 1 + 7))
""")
code("""
# E3 · evidencia como denominador, cuantiles frente a cuartiles, IC por colas frente a HPD
print(f"E3 · evidencia P(+) = {0.95 * 0.01 + 0.10 * 0.99:.4f} | 0.95·0.01/P(+) = {0.95 * 0.01 / (0.95 * 0.01 + 0.10 * 0.99):.7f} (la posteriori; el denominador es la evidencia)")
print("     qbeta(0.025), qbeta(0.975) =", np.round(stats.beta.ppf([0.025, 0.975], 16, 8), 4), "| cuartiles (25, 50, 75 %) =", np.round(stats.beta.ppf([0.25, 0.5, 0.75], 16, 8), 4))
print("     IC por colas", np.round(stats.beta(16, 8).interval(0.95), 4), "| HPD", np.round(hpd(16, 8), 4))
""")
code("""
# E4 · Monte Carlo frente a exacto; peso de la priori Optimista; valor de la diapositiva 8
print(f"E4 · exacta {stats.beta(16, 8).sf(0.6):.4f} | R con 10 000 simulaciones 0.7581 (|dif| = {abs(0.7581 - stats.beta(16, 8).sf(0.6)):.4f} ≈ {abs(0.7581 - stats.beta(16, 8).sf(0.6)) / np.sqrt(0.76 * 0.24 / 10000):.1f} EE)")
print(f"     peso de Beta(8, 2) con n = 20: {10 / 30:.3f} | equivale a {8 + 2} observaciones previas")
""")
md("""
Resultados: **E1** `stats.beta.pdf` y la fórmula con producto dan 3.0217; con la suma sale ≈ 15 230, que ni siquiera es una densidad razonable. **E2** idénticos: (16, 8). **E3** P(+) = 0.1085 y 0.0095/0.1085 = 0.0876; los cuartiles de la Beta(16, 8) son 0.604, 0.671 y 0.734 (distintos de los extremos 0.4708 y 0.8362 del intervalo); IC por colas [0.4708; 0.8362] y HPD [0.4811; 0.8446]. **E4** exacta 0.7627; el 0.7581 de R está a 0.0046 ≈ 1.1 errores estándar; peso 0.333.
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Eres analista de una plataforma educativa. Se tiene la **historia de 5 cohortes** de 20 estudiantes con 12, 13, 11, 14 y 12 aprobados (62 de 100). La nueva cohorte tiene **n = 50 con 33 aprobados**. Responde con código:

1. **Priori**: construye (a) una priori con **todo el peso** de la historia, Beta(1 + 62, 1 + 38), y (b) una priori **descontada** (25 % del peso) que conserve la proporción histórica: Beta(1 + 0.25·62, 1 + 0.25·38). ¿Qué peso (α + β) tiene cada una?
2. **Posteriori** con los datos de la nueva cohorte para cada priori: parámetros, media, desviación y IC creíble del 95 %.
3. **Decisión**: P(θ > 0.60 | y) con cada priori. ¿Cambia la conclusión?
4. **Frecuentista**: proporción observada, IC de Wilson al 95 % y p-valor unilateral de H₀: θ = 0.60. Compara con el bayesiano.
5. **Predicción** (adelanto de la Clase 24): para la **próxima cohorte de 20**, el número esperado de aprobados y P(al menos 15 aprueben) con cada posteriori.
6. **Sin código**: una frase para la dirección de la plataforma. ¿Qué dos advertencias agregarías (justificación de la priori, cohortes comparables, independencia de los estudiantes)?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
hist_y, hist_n = 62, 100
y_c, n_c = 33, 50
prioris_c = {"completa": (1 + hist_y, 1 + hist_n - hist_y), "descontada 25 %": (1 + 0.25 * hist_y, 1 + 0.25 * (hist_n - hist_y))}
for nom, (a_, b_) in prioris_c.items():
    d_ = stats.beta(a_ + y_c, b_ + n_c - y_c)
    pr_ = stats.betabinom(20, a_ + y_c, b_ + n_c - y_c)
    print(f"{nom}: priori Beta({a_:.2f}, {b_:.2f}) peso {a_ + b_:.1f} | posteriori Beta({a_ + y_c:.2f}, {b_ + n_c - y_c:.2f}) media {d_.mean():.4f} sd {d_.std():.4f} IC {np.round(d_.interval(0.95), 4)}")
    print(f"   P(θ > 0.60 | y) = {d_.sf(0.60):.4f} | próxima cohorte de 20: esperado {pr_.mean():.2f}, P(≥ 15) = {pr_.sf(14):.4f}")
p_obs = y_c / n_c
wl = ic_wilson(y_c, n_c)
pb_ = stats.binomtest(y_c, n_c, 0.60, alternative="greater")
print(f"frecuentista: p̂ = {p_obs:.2f} | Wilson [{wl[0]:.4f}, {wl[1]:.4f}] | p unilateral = {pb_.pvalue:.4f}")
""", "Ver «Resultado de referencia» en la celda siguiente.")
md("""
**Resultado de referencia**: (1) la priori completa Beta(63, 39) pesa **102** observaciones (media 0.618); la descontada Beta(16.5, 10.5) pesa **27** (media 0.611; la completa tiene media 0.618). (2) Con la completa, la posteriori es Beta(96, 56): media 0.632, sd 0.039, IC [0.554; 0.707]; con la descontada, Beta(49.5, 27.5): media 0.643, sd 0.054, IC [0.535; 0.745]. (3) P(θ > 0.60 | y) ≈ 0.79 con las dos prioris (0.792 y 0.786): la conclusión «probablemente supera 0.60, pero no es seguro» no cambia. (4) p̂ = 0.66, Wilson [0.522; 0.776], p unilateral ≈ 0.24: poca evidencia frente a 0.60 (la lectura frecuentista es más cauta que «79 %», pero responden preguntas distintas). (5) Esperado ≈ 12.6 y 12.9 aprobados de 20; P(≥ 15) ≈ 0.21 y 0.25. (6) **Recomendación**: «es probable (≈ 79 %) que la tasa de aprobación esté sobre 60 %, pero con un intervalo de ± 8 puntos no es seguro; lo más razonable es mantener el programa y seguir acumulando cohortes». Advertencias: la priori debe justificarse (¿las cohortes pasadas son comparables a la actual?) y los estudiantes de una misma cohorte pueden no ser independientes (profesor, horario), lo que haría más estrecha de lo real la incertidumbre.
""")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué es fijo y qué es aleatorio en el enfoque frecuentista? ¿Y en el bayesiano?
2. ¿Por qué P(enfermo | positivo) ≠ sensibilidad? ¿Qué es la prevalencia en términos de Bayes?
3. Define priori, verosimilitud, posteriori y evidencia. ¿Qué significa el símbolo ∝?
4. En el modelo Beta–Binomial, ¿cómo se actualizan los parámetros? ¿Qué es conjugación?
5. ¿Cómo se calculan la media, el MAP y un intervalo creíble de 95 % de una Beta(α, β)? ¿Son iguales?
6. ¿Qué diferencia hay entre «95 % de probabilidad de que θ esté en el intervalo» y «el procedimiento cubre en el 95 % de las muestras»?
7. ¿Cómo influye el tamaño de la priori (α + β) y el de la muestra? ¿Qué es la actualización secuencial?
8. ¿Qué dice P(B > A | datos) = 0.97 y qué **no** dice? ¿En qué se parece y en qué se distingue de 1 − p?

<details><summary><b>Respuestas</b></summary>

1. Frecuentista: el parámetro es fijo pero desconocido y los datos varían de muestra en muestra. Bayesiano: los datos observados son fijos y la incertidumbre está en el parámetro, descrita con una distribución.
2. Porque miran grupos distintos: P(+ | E) mira a los enfermos y P(E | +) mira a los positivos; cuando hay muchos más sanos que enfermos, los falsos positivos dominan. La prevalencia es la **priori**.
3. Priori: lo que se sabe de θ antes de los datos. Verosimilitud: compatibilidad de los datos con cada θ (no es una distribución de θ). Posteriori: lo que se sabe después. Evidencia p(y): constante que normaliza. ∝ significa «proporcional a»: se ignora la constante de normalización.
4. θ | y ~ Beta(α + y, β + n − y): éxitos previos + observados y fracasos previos + observados. Conjugación: priori y posteriori pertenecen a la misma familia.
5. Media α/(α + β); MAP (α − 1)/(α + β − 2) si α, β > 1; intervalo con `qbeta(0.025)` y `qbeta(0.975)` (por colas) o HPD. Son iguales solo si la posteriori es simétrica.
6. La primera es una probabilidad posteriori sobre θ (intervalo creíble); la segunda es una propiedad del método de largo plazo (intervalo de confianza): una vez calculado, el intervalo contiene o no a θ.
7. Cuanto mayor α + β, más pesa la priori; cuanto mayor n, menos importa. Actualización secuencial: la posteriori de un lote es la priori del siguiente y el resultado es el mismo que con todos los datos juntos.
8. Dice que, dados el modelo, la priori y los datos, hay 97 % de probabilidad de que B supere a A; **no** dice cuánto mejor es, ni que sea una mejora relevante ni que sea segura. Con priori plana se parece a 1 − p unilateral, pero la primera es una probabilidad sobre el parámetro y 1 − p no.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próxima sesión:** Clase 24 · modelo Normal con varianza conocida, distribución predictiva, simulación de Monte Carlo e introducción a MCMC (Metropolis–Hastings). Revisa `Guias/apuntes/C23_Ejercicios_Cuaderno_Bayesiana.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
