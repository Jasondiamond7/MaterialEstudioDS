"""Genera el notebook guiado de estadística bayesiana, parte 2 (Clase 24).

Uso normal:      python notebook_c24.py             -> Guias/notebooks/C24_Bayesiana2.ipynb
Modo de prueba:  C24_TEST=1 python notebook_c24.py  -> las soluciones se emiten como celdas de código
                                                       (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c24_numeros.json (lo escribe figuras_c24.py) con marcas [[ruta|decimales]].
Solo usa numpy, pandas, matplotlib, scipy y statsmodels: el muestreador MCMC se escribe a mano (no hace falta Stan ni PyMC).
"""
import inspect
import json
import os
import re
from pathlib import Path

import figuras_c24 as F

TEST = os.environ.get("C24_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C24_Bayesiana2_test.ipynb" if TEST else "C24_Bayesiana2.ipynb")
N = json.loads((HERE / "c24_numeros.json").read_text(encoding="utf-8"))
N["I"] = N["pima"]["res"]["(Intercept)"]
N["G"] = N["pima"]["res"]["glu_z"]
N["BM"] = N["pima"]["res"]["bmi_z"]
N["AG"] = N["pima"]["res"]["age_z"]
cells = []


def _get(obj, parts):
    if not parts:
        return obj
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if isinstance(obj, list):
            if i == 1 and parts[0].isdigit():
                return _get(obj[int(parts[0])], parts[1:])
        elif k in obj:
            return _get(obj[k], parts[i:])
    raise KeyError(parts)


def val(m):
    obj = _get(N, m.group(1).split("."))
    d = m.group(2)
    return str(obj) if d is None else f"{float(obj):.{int(d)}f}"


def fill(text):
    return re.sub(r"\[\[([\w.]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    md(f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>")


NUCLEO = "\n\n\n".join(inspect.getsource(fn).rstrip() for fn in (
    F.normal_normal, F.log_beta_nucleo, F.metropolis, F.autocorr, F.ess_rhat, F.log_posterior_logistica, F.metropolis_bloque))

# ---------------------------------------------------------------- encabezado
md("""
# Clase 24 · Estadística bayesiana, parte 2: posteriori, predicción, Monte Carlo y MCMC
**Diplomado Data Science** · Prof. Maritza Márquez · continúa la Clase 23 (priori, verosimilitud, Beta–Binomial) y se conecta con regresión y GLM (Clases 17–21), A/B testing (Clase 22) y bootstrap (Clase 43)

Material de apoyo:
- `Guias/C24_Guia_Bayesiana2.pdf` — guía visual de la clase.
- `Guias/apuntes/C24_Apuntes_Bayesiana2.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/C24_Ejercicios_Cuaderno_Bayesiana2.pdf` — ejercicios en papel (antes o después de este notebook).
- Material de la profesora (carpeta «Clase 23, 24 …»): `Clase 24.pdf` (diapositivas) y `Clase24.ipynb` (R: regresión logística bayesiana con `rstanarm` sobre `MASS::Pima.tr`).
- Transcripción: `VideosClases/Clase 24 … .txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy` y `statsmodels`. Si no los tienes:
```
pip install numpy pandas matplotlib scipy statsmodels
```
**No hace falta Stan, `rstanarm` ni PyMC.** El cuaderno de la profesora usa `stan_glm` (algoritmo HMC/NUTS); aquí el muestreador es un **Metropolis escrito a mano con numpy**, que es justo el algoritmo que la clase explicó. Reproduce los resultados de la profesora (con el error de Monte Carlo esperable) pero es menos eficiente que NUTS: lo verás en el ESS.

**Datos**: `Guias/datos/pima_tr.csv` = `Pima.tr` del paquete `MASS` de R ([[pima.n]] mujeres de origen Pima; `type` = diagnóstico de diabetes, `glu` glucosa, `bmi` IMC, `age` edad).

### Objetivos
1. Calcular la posteriori **Normal–Normal** (media = promedio ponderado por precisiones) y la **distribución predictiva** (Beta–Binomial y Normal).
2. Aproximar probabilidades y esperanzas posteriores con **Monte Carlo** y medir su error (también en una comparación A/B).
3. Entender **por qué** hace falta MCMC y programar **Metropolis** paso a paso.
4. Diagnosticar una cadena: **traza, tamaño de paso, burn-in, autocorrelación, ESS, varias cadenas y R-hat**.
5. Ajustar una **regresión logística bayesiana** con los datos Pima y **reproducir los resultados de la profesora**: medianas −0.921, 0.966, 0.548 y 0.573; R-hat ≈ 1.
6. Comparar con el **glm clásico** (C17–21) y el **bootstrap** (C43), y reconocer lo impreciso del material.
7. Traducir **R ↔ Python**.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Normal–Normal a mano y en grilla |
| A | 10 | A2 · Más datos, menos peso de la priori |
| A | 15 | A3 · Predictiva Beta–Binomial |
| A | 10 | A4 · Predictiva Normal |
| B | 15 | B1 · Probabilidades por simulación |
| B | 15 | B2 · El error baja como 1/√M |
| B | 15 | B3 · Comparación A/B por Monte Carlo |
| C | 10 | C1 · Por qué MCMC |
| C | 20 | C2 · Metropolis paso a paso |
| C | 15 | C3 · Una cadena completa frente a la Beta exacta |
| C | 15 | C4 · Tamaño de paso y traza |
| C | 10 | C5 · Burn-in |
| C | 15 | C6 · Autocorrelación, ESS y thinning |
| C | 15 | C7 · Varias cadenas y R-hat |
| C | 10 | C8 · Propuesta asimétrica y corrección de Hastings |
| D | 15 | D1 · Los datos Pima |
| D | 15 | D2 · Estandarizar y elegir prioris |
| D | 20 | D3 · Ajuste: 4 cadenas, 1 000 de calentamiento |
| D | 15 | D4 · Diagnósticos |
| D | 20 | D5 · Resúmenes, odds ratios y predicciones |
| D | 15 | D6 · pp_check y exactitud aparente |
| D | 20 | D7 · Frente al glm y al bootstrap |
| D | 15 | D8 · Corrida larga frente a rstanarm |
| D | 10 | D9 · Alertas verificadas y tabla R ↔ Python |
| — | 35 | Ejercicio de cierre **sin ayuda** |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`). Todo lo aleatorio usa semilla fija.
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real de este notebook; con otras versiones de las librerías puede variar la última cifra. El notebook completo tarda unos minutos (el ajuste largo de D8 y el bootstrap de D7 son lo más lento).
""")

# ---------------------------------------------------------------- preparación
md("## Preparación")
code("""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import display
from scipy import stats
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
""")
md("""
### Mini-repaso (Clases 17–23 y 43)
- **Posteriori** (C23): p(θ | y) ∝ p(y | θ)·p(θ); con Beta–Binomial, θ | y ~ Beta(α + y, β + n − y). Hoy: Beta(16, 8) = Beta(2, 2) + 14 aprobados de 20.
- **Intervalo creíble**: P(θ ∈ [a, b] | y) = 0.95; **IC de confianza**: propiedad del procedimiento (C22).
- **Regresión logística** (C20–21): logit(p) = β₀ + β₁x₁ + …, odds ratio = exp(β); aquí se ajusta con el enfoque bayesiano.
- **Bootstrap** (C43): remuestrear los datos para medir la variabilidad de un estimador.

### Funciones comunes
La celda siguiente define **todo el motor** del cuaderno: `normal_normal` (posteriori Normal–Normal), `log_beta_nucleo` y `metropolis` (Metropolis de una dimensión), `autocorr` y `ess_rhat` (diagnósticos), `log_posterior_logistica` y `metropolis_bloque` (regresión logística bayesiana). Léelas con calma: son las piezas de la clase.
""")
code(NUCLEO)

# ================================================================== PARTE A
md("""
## Parte A · Posteriori Normal–Normal y distribución predictiva

### A1 · Normal–Normal a mano y en grilla (15 min)
Ejemplo de la clase: puntaje promedio μ en una prueba de diagnóstico. Priori **μ ~ N(60, 15²)**; nueva cohorte de **n = 25** con **ȳ = 66**; **σ = 10** conocida. La posteriori es N(mₙ, sₙ²) con
**1/sₙ² = 1/s₀² + n/σ²** y **mₙ = sₙ²·(m₀/s₀² + n·ȳ/σ²)** (promedio de m₀ y ȳ ponderado por **precisiones**).
""")
code("""
m0, s0, sigma, n, ybar = 60, 15, 10, 25, 66
mn, vn = normal_normal(m0, s0, sigma, n, ybar)
w_priori = (1 / s0**2) / (1 / vn)
print(f"precisión priori = {1/s0**2:.4f} | precisión datos = {n/sigma**2:.4f} | peso de la priori = {w_priori:.4f}")
print(f"media posteriori = {mn:.4f} | varianza = {vn:.4f} | sd = {np.sqrt(vn):.4f}")
post = stats.norm(mn, np.sqrt(vn))
print(f"IC creíble 95 % = {post.interval(0.95)} | P(mu > 65 | y) = {post.sf(65):.4f}")
""")
md("""
La posteriori N([[normal.media|2]], [[normal.sd|2]]²) queda casi sobre ȳ = 66 porque la priori pesa solo [[normal.w_priori|3]] (1.7 %). La pregunta práctica de la clase es **P(μ > 65 | y) = [[normal.p65|3]]**.

✍️ **Ejercicio A1.** Verifica **sin la fórmula** que la posteriori es esa normal: multiplica la priori N(60, 15²) por la verosimilitud de ȳ (normal con media μ y sd σ/√n) en una grilla de μ entre 40 y 90, normaliza con la regla del trapecio (`trapezoid`) y compara la media y la sd de la grilla con las de la fórmula.
""")
code("""
# tu código aquí
""")
solution("""
mu = np.linspace(40, 90, 5001)
g = stats.norm.pdf(mu, m0, s0) * stats.norm.pdf(ybar, mu, sigma / np.sqrt(n))     # priori x verosimilitud de ybar
g = g / trapezoid(g, mu)                                                           # normaliza
media_g = trapezoid(mu * g, mu)
sd_g = np.sqrt(trapezoid((mu - media_g) ** 2 * g, mu))
print(f"grilla: media = {media_g:.4f}, sd = {sd_g:.4f} | fórmula: media = {mn:.4f}, sd = {np.sqrt(vn):.4f}")
print("P(mu > 65) en la grilla:", round(float(trapezoid(g[mu > 65], mu[mu > 65])), 4))
""", "La media y la sd coinciden hasta 4 decimales (la probabilidad por trapecio difiere en el 3.er decimal por el borde de la grilla): la grilla es la definición (priori × verosimilitud ÷ evidencia) y la fórmula es su resultado en forma cerrada.")

md("""
### A2 · Más datos, menos peso de la priori (10 min)
La profesora preguntó: *si aumenta el tamaño muestral, ¿la priori desaparece o pierde peso relativo?* Calcula la posteriori para n = 1, 5, 25, 100 y 400 con el mismo ȳ = 66.
""")
code("""
filas = []
for k in [1, 5, 25, 100, 400]:
    m_, v_ = normal_normal(m0, s0, sigma, k, ybar)
    filas.append({"n": k, "media": m_, "sd": np.sqrt(v_), "peso priori": (1 / s0**2) / (1 / v_)})
display(pd.DataFrame(filas).round(4))
""")
md("""
Con n = 1 la media es [[normal.por_n.1.0|2]] (la priori pesa 31 %); con n = 400, [[normal.por_n.400.0|2]] y sd [[normal.por_n.400.1|2]]. **La priori no desaparece, pierde peso relativo**: su precisión es fija (1/225) y la de los datos crece con n.

✍️ **Ejercicio A2.** ¿Desde qué n el peso de la priori es menor que 1 %? ¿Y si la priori fuera mucho más vaga, s₀ = 100? (Resuelve con la fórmula del peso w = (1/s₀²)/(1/s₀² + n/σ²).)
""")
code("""
# tu código aquí
""")
solution("""
for s0_ in (15, 100):
    ns = np.arange(1, 5000)
    w = (1 / s0_**2) / (1 / s0_**2 + ns / sigma**2)
    print(f"s0 = {s0_:>3}: el peso baja de 1 % desde n = {ns[w < 0.01][0]}")
""", "Con s₀ = 15 hace falta n ≈ 45; con una priori muy vaga (s₀ = 100), n ≈ 1: más vaga = menos peso desde el inicio. Resolviendo w < 0.01: n > 99·σ²/s₀².")

md("""
### A3 · Predictiva Beta–Binomial (15 min)
Con la posteriori **Beta(16, 8)** de la Clase 23, ¿cuántos de los **m = 10** estudiantes de la próxima cohorte aprobarán? La predictiva promedia sobre θ: se simula **θ⁽ˢ⁾ ~ Beta(16, 8)** y luego **Ỹ⁽ˢ⁾ ~ Binomial(10, θ⁽ˢ⁾)**; su forma exacta es la **beta-binomial**. La comparamos con la predicción «plug-in» que usa solo la media posterior 0.667.
""")
code("""
a, b, m = 16, 8, 10
rng = np.random.default_rng(24201)
S = 20000
th = stats.beta(a, b).rvs(S, random_state=rng)
ysim = rng.binomial(m, th)
bb = stats.betabinom(m, a, b)
plug = stats.binom(m, a / (a + b))
k = np.arange(m + 1)
print(f"media: predictiva {bb.mean():.3f} (simulada {ysim.mean():.3f}) | varianza: predictiva {bb.var():.3f} vs plug-in {plug.var():.3f}")
print(f"P(Y < 5): exacta {bb.cdf(4):.4f} | simulada {np.mean(ysim < 5):.4f} | plug-in {plug.cdf(4):.4f}")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(k - 0.2, bb.pmf(k), width=0.4, label="predictiva (beta-binomial)")
ax.bar(k + 0.2, plug.pmf(k), width=0.4, label="plug-in Binomial(10, 0.667)")
ax.set(xlabel="aprobados de 10", ylabel="probabilidad"); ax.legend(); plt.show()
""")
md("""
La media es la misma ([[pred.media|3]]), pero la **varianza** pasa de [[pred.var_plug|3]] (θ fijo) a [[pred.var|3]]: la predictiva suma el azar de los 10 estudiantes **y** la incertidumbre sobre θ. Por eso **P(menos de 5)** es [[pred.p_menos5|3]] (simulada [[pred.p_menos5_sim|3]]) y no [[pred.p_menos5_plug|3]]: «no basta decir que aprobarán 7; conviene decir qué tan probable es que aprueben menos de 5».

✍️ **Ejercicio A3.** Con la misma posteriori, calcula (exacto y simulado) la probabilidad de que **aprueben al menos 9** de los 10 y compárala con el plug-in. Después comprueba la fórmula de la varianza: Var(Ỹ) = m·αβ(α + β + m)/((α + β)²(α + β + 1)).
""")
code("""
# tu código aquí
""")
solution("""
print("P(Y >= 9): exacta", round(float(bb.sf(8)), 4), "| simulada", round(float(np.mean(ysim >= 9)), 4), "| plug-in", round(float(plug.sf(8)), 4))
var_f = m * a * b * (a + b + m) / ((a + b) ** 2 * (a + b + 1))
print("varianza por fórmula:", round(var_f, 4), "| scipy:", round(float(bb.var()), 4))
""", "La predictiva da más probabilidad a los extremos (colas más pesadas): P(Y ≥ 9) ≈ 0.148 frente a 0.104 del plug-in. La fórmula coincide con `scipy` (3.0222).")

md("""
### A4 · Predictiva Normal (10 min)
En el modelo Normal–Normal la predictiva de un estudiante nuevo es **N(mₙ, σ² + sₙ²)**: se suman la varianza del dato y la de la posteriori.
""")
code("""
sd_pred = np.sqrt(sigma**2 + vn)
print(f"sd predictiva = {sd_pred:.4f} (frente a sigma = {sigma})")
print("P(nuevo > 65): predictiva", round(float(stats.norm.sf(65, mn, sd_pred)), 4), "| plug-in N(mn, sigma^2)", round(float(stats.norm.sf(65, mn, sigma)), 4))
rng = np.random.default_rng(24202)
mu_s = rng.normal(mn, np.sqrt(vn), 100000)                 # mu^(s) ~ posteriori
y_nuevo = rng.normal(mu_s, sigma)                          # dato nuevo | mu^(s)
print("simulación: media", round(float(y_nuevo.mean()), 3), "sd", round(float(y_nuevo.std()), 3))
""")
md("""
Con n = 25 la incertidumbre sobre μ (sₙ ≈ 2) es chica frente a σ = 10: la sd pasa de 10 a [[normal.pred_sd|2]]. Si n fuera 1, sₙ sería 8.3 y la predictiva mucho más ancha que el plug-in.

✍️ **Ejercicio A4.** Repite con **n = 1** (ȳ = 66): calcula mₙ, sₙ, la sd de la predictiva y P(nuevo > 65) con la predictiva y con el plug-in.
""")
code("""
# tu código aquí
""")
solution("""
m1, v1 = normal_normal(m0, s0, sigma, 1, ybar)
sd1 = np.sqrt(sigma**2 + v1)
print(f"n = 1: mn = {m1:.3f}, sn = {np.sqrt(v1):.3f}, sd predictiva = {sd1:.3f}")
print("P(nuevo > 65): predictiva", round(float(stats.norm.sf(65, m1, sd1)), 4), "| plug-in", round(float(stats.norm.sf(65, m1, sigma)), 4))
""", "Con n = 1: mₙ = 64.154, sₙ = 8.321 y sd predictiva 13.0 (frente a 10). Las probabilidades son 0.4741 (predictiva) y 0.4663 (plug-in): la diferencia crece cuando n es pequeño.")

# ================================================================== PARTE B
md("""
## Parte B · Simulación de Monte Carlo

### B1 · Probabilidades por simulación (15 min)
Cualquier integral posterior se reemplaza por un promedio: **P(θ > 0.6 | y) ≈ (1/M) Σ 1{θ⁽ᵐ⁾ > 0.6}**. La profesora lo ilustró con «780 de 1 000 son mayores a 0.6».
""")
code("""
post_b = stats.beta(16, 8)
exacta = post_b.sf(0.6)
rng = np.random.default_rng(24301)
M = 1000
th = post_b.rvs(M, random_state=rng)
est = np.mean(th > 0.6)
se = np.sqrt(exacta * (1 - exacta) / M)
print(f"{int((th > 0.6).sum())} de {M} superan 0.6 -> estimación {est:.3f} | exacta {exacta:.4f} | error estándar de Monte Carlo = {se:.4f}")
print("E[theta] ~", round(float(th.mean()), 4), "(exacta", round(float(post_b.mean()), 4), ") | E[theta^2] ~", round(float((th**2).mean()), 4), "(exacta", round(float(post_b.var() + post_b.mean()**2), 4), ")")
""")
md("""
Con M = 1 000 salen [[mc.cuenta]] de 1 000 (**[[mc.estimado|3]]**) y la respuesta exacta es [[mc.exacta|4]]; el error estándar de Monte Carlo es ≈ [[mc.se|4]]. En la clase de hoy el valor «780/1 000» era solo un ejemplo hipotético.

✍️ **Ejercicio B1.** Estima por simulación (M = 20 000) la probabilidad de que θ esté **entre 0.5 y 0.8**, y la media de la **razón de odds** θ/(1 − θ). Compara la primera con la cdf exacta.
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(1)
th2 = post_b.rvs(20000, random_state=rng)
print("P(0.5 < theta < 0.8): simulada", round(float(np.mean((th2 > 0.5) & (th2 < 0.8))), 4), "| exacta", round(float(post_b.cdf(0.8) - post_b.cdf(0.5)), 4))
print("E[odds] simulada:", round(float((th2 / (1 - th2)).mean()), 3))
""", "La probabilidad exacta es 0.8819. La media de los odds θ/(1 − θ) es 16/7 ≈ 2.29 (exacta para una Beta(16, 8)): es la esperanza de una transformación no lineal, distinta de 0.667/0.333 = 2.")

md("""
### B2 · El error baja como 1/√M (15 min)
Mide el error absoluto medio de la estimación de P(θ > 0.6 | y) para M = 10, 100, 1 000 y 10 000 (200 réplicas cada una) y compáralo con 0.80·√(p(1 − p)/M).
""")
code("""
rng = np.random.default_rng(24302)
Ms = np.array([10, 100, 1000, 10000])
err = np.array([np.mean([abs(np.mean(post_b.rvs(M_, random_state=rng) > 0.6) - exacta) for _ in range(200)]) for M_ in Ms])
teoria = np.sqrt(2 / np.pi) * np.sqrt(exacta * (1 - exacta) / Ms)
display(pd.DataFrame({"M": Ms, "error medio": err, "teoría": teoria}).round(4))
""")
md("""
Cada 100 veces más simulaciones, el error cae ≈ 10 veces (error ≈ 0.011 con M = 1 000 y ≈ 0.004 con M = 10 000). Es error de **cálculo**: no se confunde con la incertidumbre sobre θ, que la da la posteriori.

✍️ **Ejercicio B2.** ¿Cuántas simulaciones hacen falta para que el error estándar de P̂ sea ≤ 0.002? ¿Y si las muestras vinieran de una cadena con ESS = 10 % de M?
""")
code("""
# tu código aquí
""")
solution("""
p = exacta
M_iid = p * (1 - p) / 0.002 ** 2
print("M (independientes):", int(np.ceil(M_iid)), "| M (cadena con ESS = 10 % de M):", int(np.ceil(M_iid / 0.10)))
""", "Con muestras independientes M ≈ 45 000; con una cadena cuyo ESS es 10 % de M hace falta 10 veces más (≈ 450 000): el error depende del ESS, no de M.")

md("""
### B3 · Comparación A/B por Monte Carlo (15 min)
La diapositiva 28 anunciaba «Comparación A/B mediante simulación Monte Carlo». Con números nuevos: control **120 conversiones de 1 000** y tratamiento **145 de 1 000**, prioris Beta(1, 1). La diferencia de las dos posterioris no se calcula a mano: se **simula**.
""")
code("""
rng = np.random.default_rng(24303)
pA = stats.beta(1 + 120, 1 + 880).rvs(100000, random_state=rng)
pB = stats.beta(1 + 145, 1 + 855).rvs(100000, random_state=rng)
dif = pB - pA
print(f"P(B > A | datos) = {np.mean(dif > 0):.4f}")
print(f"diferencia: media {dif.mean():.4f}, IC creíble 95 % = {np.quantile(dif, [0.025, 0.975]).round(4)}")
print(f"P(mejora > 0.01) = {np.mean(dif > 0.01):.4f}")
print("elevación relativa mediana:", round(float(np.median(dif / pA)), 3))
""")
md("""
Es el análisis bayesiano del A/B testing de la Clase 22: en lugar de un p-valor se obtiene **P(B > A | datos)** y la probabilidad de que la mejora supere un mínimo relevante.

✍️ **Ejercicio B3.** Con los mismos datos, ¿cuál es la probabilidad de que la **elevación relativa** (θ_B/θ_A − 1) supere 10 %? ¿Cambia mucho si se usa una priori Beta(12, 88) (tasa esperada 12 %, equivalente a 100 observaciones previas)?
""")
code("""
# tu código aquí
""")
solution("""
rng = np.random.default_rng(5)
for (a0, b0), nom in (((1, 1), "Beta(1, 1)"), ((12, 88), "Beta(12, 88)")):
    pa = stats.beta(a0 + 120, b0 + 880).rvs(100000, random_state=rng)
    pb = stats.beta(a0 + 145, b0 + 855).rvs(100000, random_state=rng)
    print(nom, "-> P(elevación > 10 %):", round(float(np.mean(pb / pa - 1 > 0.10)), 3), "| P(B > A):", round(float(np.mean(pb > pa)), 3))
""", "Con la priori plana P(elevación > 10 %) ≈ 0.79; con la informativa Beta(12, 88) baja a ≈ 0.76: la priori pesa como 100 observaciones frente a 1 000 por grupo y tira ambas tasas hacia 12 %, así que la conclusión casi no cambia.")

# ================================================================== PARTE C
md("""
## Parte C · MCMC con un Metropolis escrito a mano

### C1 · Por qué MCMC (10 min)
En Beta–Binomial y Normal–Normal la posteriori **tiene nombre** y se simula directo (`rvs`). En la regresión logística no: solo se conoce el **núcleo** p(y | β)·p(β). Una grilla necesitaría 100^d evaluaciones (d = 4 coeficientes → 10⁸). Veamos el núcleo de la posteriori de (β₀, β_glu) de los datos Pima (los cargamos completos en la Parte D).
""")
code("""
carpeta = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Guias" / "datos" / "pima_tr.csv").exists()) / "Guias" / "datos"
pima = pd.read_csv(carpeta / "pima_tr.csv")
pima["diabetes"] = (pima["type"] == "Yes").astype(int)
cols = ["glu", "bmi", "age"]
medias, sds = pima[cols].mean(), pima[cols].std(ddof=1)
Z = (pima[cols] - medias) / sds
X = np.column_stack([np.ones(len(pima)), Z.to_numpy()])
y = pima["diabetes"].to_numpy()
logp = lambda beta: log_posterior_logistica(beta, X, y)       # núcleo de la posteriori (log), sin constante

g0, g1 = np.linspace(-1.6, -0.3, 80), np.linspace(0.3, 1.8, 80)
LP = np.array([[logp(np.array([u, v, 0.55, 0.57])) for u in g0] for v in g1])
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.contourf(g0, g1, np.exp(LP - LP.max()), levels=12, cmap="Greens")
ax.set(xlabel="β0 (intercepto)", ylabel="β_glu"); plt.show()
print("evaluaciones de una grilla de 100 por eje con d = 4:", 100 ** 4, "(1e8)")
""")
md("""
La posteriori existe como función (se puede **evaluar** salvo una constante), pero no tiene una forma cerrada conocida ni es fácil de simular directamente. **MCMC** construye una cadena cuyo comportamiento a largo plazo reproduce esa distribución.

### C2 · Metropolis paso a paso (20 min)
Objetivo de prueba: la Beta(16, 8), cuyo núcleo es θ¹⁵(1 − θ)⁷ (así podemos comparar con la respuesta exacta). Algoritmo desde θ: **(1)** proponer θ* = θ + τ·N(0, 1); **(2)** calcular **r = núcleo(θ*)/núcleo(θ)**; **(3)** aceptar con probabilidad **α = min(1, r)** (si u ~ U(0,1) es menor que α); **(4)** si no, la cadena **repite** θ. Seis pasos reales (semilla 24500, desde 0.55, τ = 0.10):
""")
code("""
logp_beta = log_beta_nucleo(16, 8)
nucleo = lambda t: np.exp(logp_beta(t))
rng = np.random.default_rng(24500)
x, tau, filas = 0.55, 0.10, []
for paso in range(1, 7):
    prop = x + tau * rng.normal()
    u = rng.uniform()
    r = nucleo(prop) / nucleo(x)
    acepta = u < min(1, r)
    filas.append({"paso": paso, "actual": x, "propuesta": prop, "r": r, "alfa": min(1, r), "u": u, "acepta": acepta})
    if acepta:
        x = prop
display(pd.DataFrame(filas).round(3))
""")
md("""
Los pasos 1, 2, 3 y 5 se aceptan (el 3 es una **bajada** con r = [[pasos.2.razon|2]] < 1 que se acepta porque u = [[pasos.2.u|2]] es menor que α); los pasos 4 y 6 se rechazan y la cadena **se queda** donde estaba. Rechazar no es un error: es parte del algoritmo.

✍️ **Ejercicio C2.** Hazlo **a mano** para el núcleo θ⁵(1 − θ)² (una Beta(6, 3)): actual 0.50, propuesta 0.65. Calcula r y α; ¿se acepta con u = 0.80? Después propone 1.05: ¿qué ocurre?
""")
code("""
# tu código aquí
""")
solution("""
k6 = lambda t: t ** 5 * (1 - t) ** 2 if 0 < t < 1 else 0.0
r = k6(0.65) / k6(0.50)
print("r =", round(r, 4), "| alfa =", min(1, round(r, 4)), "| acepta con u = 0.80:", 0.80 < min(1, r))
print("propuesta 1.05: núcleo =", k6(1.05), "-> r = 0 -> se rechaza y la cadena repite 0.65")
""", "r = 1.819 > 1, así que α = 1 y siempre se acepta; fuera de (0, 1) el núcleo vale 0 y la propuesta siempre se rechaza.")

md("""
### C3 · Una cadena completa frente a la Beta exacta (15 min)
Ahora la función `metropolis` hace lo mismo muchas veces. 5 000 iteraciones desde 0.5 con τ = 0.12:
""")
code("""
cadena, acep = metropolis(logp_beta, 0.5, 0.12, 5000, np.random.default_rng(24310))
exacta_b = stats.beta(16, 8)
print(f"aceptación = {acep:.3f} | media de la cadena = {cadena.mean():.4f} (exacta {exacta_b.mean():.4f}) | sd = {cadena.std():.4f} (exacta {exacta_b.std():.4f})")
print("P(theta > 0.6): cadena", round(float(np.mean(cadena > 0.6)), 4), "| exacta", round(float(exacta_b.sf(0.6)), 4))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
axs[0].plot(cadena, lw=0.6); axs[0].set(title="traza", xlabel="iteración", ylabel="theta")
axs[1].hist(cadena, bins=40, density=True, alpha=0.6); tt = np.linspace(0.2, 1, 200); axs[1].plot(tt, exacta_b.pdf(tt), "k"); axs[1].set(title="frente a la Beta(16, 8)")
plt.show()
""")
md("""
La cadena **no es una muestra independiente**, pero sus frecuencias reproducen la posteriori. Observa cuánto se parece el histograma a la curva exacta y que la media de la cadena se acerca a 0.667.

✍️ **Ejercicio C3.** Usa la cadena para estimar la mediana y el intervalo creíble de 95 % de θ y compáralos con `exacta_b.ppf`. ¿Qué error esperas, sabiendo que las muestras están autocorrelacionadas?
""")
code("""
# tu código aquí
""")
solution("""
print("cadena:", np.quantile(cadena, [0.025, 0.5, 0.975]).round(4))
print("exacta:", exacta_b.ppf([0.025, 0.5, 0.975]).round(4))
ess_c, _ = ess_rhat(cadena[None, :])
print("ESS =", round(ess_c), "de", len(cadena), "-> el error de la media es sd/sqrt(ESS) =", round(float(cadena.std() / np.sqrt(ess_c)), 4))
""", "Los cuantiles de la cadena se parecen a los exactos (0.471, 0.671, 0.836) con error del orden de sd/√ESS: el ESS, no las 5 000 iteraciones, manda.")

md("""
### C4 · Tamaño de paso y traza (15 min)
El tamaño de paso τ decide la calidad de la cadena. Tres valores sobre la misma posteriori (2 000 iteraciones desde 0.5): **τ = 0.005**, **0.12** y **1.5**.
""")
code("""
res_paso = []
fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
for i, tau_ in enumerate([0.005, 0.12, 1.5]):
    ch, ac = metropolis(logp_beta, 0.5, tau_, 2000, np.random.default_rng(24601 + i))
    ess_, _ = ess_rhat(ch[None, :])
    res_paso.append((tau_, ac, ess_, ch.mean()))
    axs[i].plot(ch, lw=0.7); axs[i].axhline(exacta_b.mean(), color="k", ls=":"); axs[i].set(ylim=(0.2, 1), ylabel=f"tau = {tau_}")
axs[2].set_xlabel("iteración"); plt.show()
display(pd.DataFrame(res_paso, columns=["tau", "aceptación", "ESS", "media"]).round(3))
""")
md("""
- **τ = 0.005**: aceptación [[traza.0.aceptacion|2]] (98 %) y ESS ≈ [[traza.0.ess|0]]: casi todo se acepta pero la cadena **avanza como tortuga**; su media ([[traza.0.media|3]]) está lejos de 0.667.
- **τ = 0.12**: aceptación [[traza.1.aceptacion|2]] y ESS ≈ [[traza.1.ess|0]]: «oruga peluda», el punto medio.
- **τ = 1.5**: aceptación [[traza.2.aceptacion|2]] y ESS ≈ [[traza.2.ess|0]]: casi todo se rechaza y la cadena queda **pegada** por tramos.

✍️ **Ejercicio C4.** Barre τ en {0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1, 2} con 3 000 iteraciones y grafica el ESS frente a τ. ¿Dónde está el máximo y cuál es la aceptación ahí? (Regla práctica: 25–50 %.)
""")
code("""
# tu código aquí
""")
solution("""
taus = [0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1, 2]
out = []
for t_ in taus:
    ch, ac = metropolis(logp_beta, 0.5, t_, 3000, np.random.default_rng(7))
    out.append((t_, ac, ess_rhat(ch[None, :])[0]))
tabla = pd.DataFrame(out, columns=["tau", "aceptación", "ESS"]).round(3)
display(tabla)
plt.semilogx(tabla["tau"], tabla["ESS"], "o-"); plt.xlabel("tau"); plt.ylabel("ESS"); plt.show()
""", "El ESS tiene un máximo en un paso intermedio (aquí τ = 0.25: ESS ≈ 530 y aceptación ≈ 42 %) y cae por los dos lados.")

md("""
### C5 · Burn-in (10 min)
Una cadena que parte lejos de la zona de la posteriori necesita unas iteraciones para llegar: el **burn-in** (en Stan, «warmup», que además **adapta** el paso). Dos cadenas desde 0.03 y 0.97 (τ = 0.05):
""")
code("""
c1, _ = metropolis(logp_beta, 0.03, 0.05, 1500, np.random.default_rng(24701))
c2, _ = metropolis(logp_beta, 0.97, 0.05, 1500, np.random.default_rng(24702))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
axs[0].plot(c1[:400], label="desde 0.03"); axs[0].plot(c2[:400], label="desde 0.97"); axs[0].axvspan(0, 150, color="red", alpha=0.1); axs[0].legend(); axs[0].set(title="primeras 400 iteraciones")
axs[1].hist(c1, bins=np.linspace(0, 1, 50), density=True, alpha=0.5, label="con inicio"); axs[1].hist(c1[150:], bins=np.linspace(0, 1, 50), density=True, alpha=0.5, label="sin burn-in")
tt = np.linspace(0.01, 0.99, 200); axs[1].plot(tt, exacta_b.pdf(tt), "k"); axs[1].legend(); plt.show()
""")
md("""
Para medir el efecto sin depender de una sola cadena, se promedia el **sesgo** de la media de la cadena según las primeras iteraciones descartadas:
""")
code("""
nb = [0, 25, 50, 100, 150, 250, 400]
sesgo = np.zeros(len(nb))
rng = np.random.default_rng(24703)
R_rep = 400
for _ in range(R_rep):
    c_, _a = metropolis(logp_beta, 0.03, 0.05, 700, rng)
    for i, b_ in enumerate(nb):
        sesgo[i] += c_[b_:].mean() - exacta_b.mean()
display(pd.DataFrame({"descartadas": nb, "sesgo medio": sesgo / R_rep}).round(4))
""")
md("""
Sin descartar nada, el sesgo medio es [[burnin.sesgo.0|4]] (la cadena parte lejos y arrastra el promedio); con 50–100 iteraciones descartadas es ≈ 0. **El burn-in elimina el efecto del punto de partida; no arregla un modelo mal especificado ni una priori inadecuada.**

✍️ **Ejercicio C5.** ¿Cuántas iteraciones habría que descartar si el punto de partida fuera 0.001 con τ = 0.02? Repite el experimento de sesgo (100 réplicas de 1 500 iteraciones) y decide un burn-in razonable (sesgo absoluto < 0.002).
""")
code("""
# tu código aquí
""")
solution("""
nb2 = [0, 100, 250, 500, 750, 1000]
ses = np.zeros(len(nb2))
rng = np.random.default_rng(11)
for _ in range(100):
    c_, _a = metropolis(logp_beta, 0.001, 0.02, 1500, rng)
    for i, b_ in enumerate(nb2):
        ses[i] += c_[b_:].mean() - exacta_b.mean()
display(pd.DataFrame({"descartadas": nb2, "sesgo medio": ses / 100}).round(4))
""", "Con un paso chico y un punto de partida extremo hace falta un burn-in mucho mayor: el sesgo baja de 0.002 recién hacia las 500 iteraciones: la regla «descartar la mitad» es conservadora pero segura; lo correcto es mirar trazas de varias cadenas.")

md("""
### C6 · Autocorrelación, ESS y thinning (15 min)
Los valores de una cadena son dependientes. La **autocorrelación** ρₖ mide cuánto se parece θ⁽ᵗ⁾ a θ⁽ᵗ⁺ᵏ⁾, y el **tamaño efectivo** ESS ≈ M/(1 + 2Σρₖ) cuántas muestras independientes «vale» la cadena. 5 000 iteraciones por paso:
""")
code("""
res_acf = []
fig, ax = plt.subplots(figsize=(7, 3.4))
cad_acf = []
for i, tau_ in enumerate([0.005, 0.12, 1.5]):
    ch, ac = metropolis(logp_beta, 0.68, tau_, 5000, np.random.default_rng(24801 + i))
    cad_acf.append(ch)
    ax.plot(autocorr(ch, 60), label=f"tau = {tau_}")
    res_acf.append((tau_, ac, ess_rhat(ch[None, :])[0], autocorr(ch, 10)[10]))
ax.axhline(0, color="k", lw=0.8); ax.set(xlabel="rezago", ylabel="autocorrelación"); ax.legend(); plt.show()
display(pd.DataFrame(res_acf, columns=["tau", "aceptación", "ESS", "ρ10"]).round(3))
""")
md("""
Con τ = 0.12 el ESS es ≈ [[acf.res.1.ess|0]] de 5 000 ([[acf.res.1.ac10|2]] de autocorrelación a 10 rezagos); con τ = 0.005 es ≈ [[acf.res.0.ess|0]] (¡5 000 iteraciones equivalen a 2 muestras!). **Thinning** (guardar 1 de cada k) no crea información: con la cadena de τ = 0.12 el ESS baja:
""")
code("""
ch_t = cad_acf[1]
for k_ in (1, 10, 50, 100):
    sub = ch_t[::k_]
    print(f"thin = {k_:>3}: {len(sub):>5} valores, ESS = {ess_rhat(sub[None, :])[0]:.0f}")
""")
md("""
Adelgazar solo ahorra memoria: ESS pasa de [[acf.thin.0.ess|0]] a [[acf.thin.1.ess|0]] con thin = 10 y a [[acf.thin.3.ess|0]] con thin = 100.

✍️ **Ejercicio C6.** El error de Monte Carlo de la media de la cadena τ = 0.12 es sd/√ESS. Calcúlalo y compáralo con el que tendrías con 5 000 muestras independientes. ¿Cuánto mayor es?
""")
code("""
# tu código aquí
""")
solution("""
ch = cad_acf[1]
ess_ = ess_rhat(ch[None, :])[0]
print("error con ESS:", round(float(ch.std() / np.sqrt(ess_)), 4), "| con 5000 independientes:", round(float(ch.std() / np.sqrt(5000)), 4), "| razón:", round(float(np.sqrt(5000 / ess_)), 2))
""", "La razón es √(5 000/ESS) ≈ 2.4: la cadena autocorrelada tiene un error ≈ 2.4 veces mayor que el de una muestra independiente del mismo tamaño.")

md("""
### C7 · Varias cadenas y R-hat (15 min)
Se parten **cuatro cadenas desde puntos distintos** (0.05, 0.3, 0.7, 0.95). **R-hat** compara la varianza **entre** cadenas con la de **dentro**: ≈ 1 si todas exploran lo mismo; > 1.01 sugiere falta de convergencia.
""")
code("""
starts = [0.05, 0.3, 0.7, 0.95]
cad4 = np.array([metropolis(logp_beta, s_, 0.05, 3000, np.random.default_rng(24901 + i))[0] for i, s_ in enumerate(starts)])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for k in range(4):
    axs[0].plot(cad4[k, :120], lw=0.8); axs[1].plot(cad4[k], lw=0.5)
axs[0].set(title="120 iteraciones"); axs[1].set(title="3 000 iteraciones"); plt.show()
for L in (20, 80, 320, 1280, 3000):
    e_, r_ = ess_rhat(cad4[:, :L])
    print(f"{L:>5} iteraciones por cadena: R-hat = {r_:.3f}, ESS = {e_:.0f}")
""")
md("""
Con 120 iteraciones R-hat = [[rhat.rhat_120|3]] (las cadenas aún no se mezclan) y con 3 000 baja a **[[rhat.rhat_3000|3]]** (ESS ≈ [[rhat.ess_3000|0]]). El cuaderno de la profesora muestra R-hat de 1.000, 1.000, 1.002 y 1.001: «cercano a 1 implica buena convergencia; por encima de 1.01 puede indicar problemas». **Es necesario, no suficiente**: si todas las cadenas se quedan en la misma región equivocada, R-hat sale ≈ 1.

✍️ **Ejercicio C7.** Inicia las cuatro cadenas con τ = 1.5 (paso grande) y 3 000 iteraciones: ¿qué R-hat y ESS obtienes? ¿Qué te dice sobre confiar solo en R-hat?
""")
code("""
# tu código aquí
""")
solution("""
cad_g = np.array([metropolis(logp_beta, s_, 1.5, 3000, np.random.default_rng(5 + i))[0] for i, s_ in enumerate(starts)])
e_, r_ = ess_rhat(cad_g)
print(f"tau = 1.5: R-hat = {r_:.3f}, ESS = {e_:.0f}, aceptación media = {np.mean(cad_g[:, 1:] != cad_g[:, :-1]):.3f}")
""", "Con paso grande el R-hat sale 1.006 (parece aceptable) aunque el ESS es ≈ 625 de 12 000 (5 %) y la aceptación 8 %: las cadenas son compatibles entre sí pero aportan poca información. Siempre se mira R-hat **y** ESS **y** la traza.")

md("""
### C8 · Propuesta asimétrica y corrección de Hastings (10 min)
La diapositiva 24 describe el algoritmo con una propuesta «cercana» y acepta según la razón de posterioris. Eso es válido si la propuesta es **simétrica** (p. ej. normal centrada en el valor actual). Con una propuesta **asimétrica** hay que usar r = [p(θ*)·q(θ | θ*)]/[p(θ)·q(θ* | θ)] (**Hastings**); si no, la cadena converge a **otra** distribución. Propuesta con deriva: θ* = θ + 0.08 + 0.10·N(0, 1).
""")
code("""
def mh_deriva(corregir, rng, n=60000, mu=0.08, s=0.1):
    x, cad = 0.6, np.empty(n)
    lq = lambda a, centro: -0.5 * ((a - centro) / s) ** 2          # log densidad normal (sin constante, se cancela)
    for t in range(n):
        prop = x + mu + s * rng.normal()
        logr = logp_beta(prop) - logp_beta(x)
        if corregir:
            logr += lq(x, prop + mu) - lq(prop, x + mu)           # log q(x | prop) - log q(prop | x)
        if np.log(rng.uniform()) < logr:
            x = prop
        cad[t] = x
    return cad

rng = np.random.default_rng(24990)
sin_corr = mh_deriva(False, rng)
con_corr = mh_deriva(True, rng)
print(f"media sin corrección = {sin_corr[2000:].mean():.3f} | con corrección = {con_corr[2000:].mean():.3f} | exacta = {exacta_b.mean():.3f}")
""")
md("""
Sin corrección la media sale **[[verif.asimetrica.media_sin_correccion|3]]** (sesgada hacia arriba, por la deriva) y con corrección **[[verif.asimetrica.media_con_correccion|3]]**, como la exacta ([[verif.asimetrica.exacta|3]]). Con la propuesta normal simétrica de la clase no hace falta la corrección: se cancela.

✍️ **Ejercicio C8.** Demuestra con código que con propuesta simétrica los dos términos q se cancelan: calcula `lq(x, prop) - lq(prop, x)` para varios pares (x, prop) con centro en el valor actual.
""")
code("""
# tu código aquí
""")
solution("""
s_ = 0.1
lq = lambda a, centro: -0.5 * ((a - centro) / s_) ** 2
pares = [(0.3, 0.45), (0.7, 0.62), (0.5, 0.9)]
print([round(lq(x_, p_) - lq(p_, x_), 12) for x_, p_ in pares])
""", "Con q(a | b) = N(b, s²), la densidad de ir de x a prop es igual a la de ir de prop a x (depende solo de |x − prop|): el cociente vale 1 y la corrección es 0 en logaritmos.")

# ================================================================== PARTE D
md("""
## Parte D · Regresión logística bayesiana con los datos Pima

### D1 · Los datos (15 min)
`Pima.tr`: 200 mujeres de origen Pima (cerca de Phoenix, Arizona). Respuesta `type` = diabetes (Yes/No, criterio de la OMS); predictores: glucosa (`glu`), índice de masa corporal (`bmi`) y edad (`age`).
""")
code("""
print(pima.shape, "| valores faltantes:", int(pima[["type", "glu", "bmi", "age"]].isna().sum().sum()))
display(pima[["type", "glu", "bmi", "age"]].head(6))
print(pima["type"].value_counts().to_dict(), "|", pima["type"].value_counts(normalize=True).round(2).to_dict())
fig, ax = plt.subplots(figsize=(4.5, 3.6))
ax.boxplot([pima.loc[pima.diabetes == 0, "glu"], pima.loc[pima.diabetes == 1, "glu"]], tick_labels=["No", "Yes"])
ax.set(ylabel="glucosa plasmática", title="Glucosa según diagnóstico"); plt.show()
""")
md("""
Hay [[pima.n_no]] «No» y [[pima.n_si]] «Yes» (66 % y 34 %, como en la clase) y ningún faltante. Las medianas de glucosa son [[pima.med_glu.0|0]] y [[pima.med_glu.1|0]]: la glucosa parece importante **antes** de ajustar nada (la profesora lo destacó).

✍️ **Ejercicio D1.** Cuenta cuántos valores atípicos (fuera de 1.5·RIC) hay en la glucosa del grupo «No». ¿Cambia tu lectura del boxplot?
""")
code("""
# tu código aquí
""")
solution("""
g = pima.loc[pima.diabetes == 0, "glu"]
q1, q3 = g.quantile([0.25, 0.75])
lim = q3 + 1.5 * (q3 - q1)
print("valores atípicos por arriba:", int((g > lim).sum()), "| límite:", round(float(lim), 1))
""", "Hay unos pocos valores atípicos por arriba en el grupo «No» (las puntas del boxplot): son valores posibles, no errores; no cambian la lectura de que el grupo «Yes» tiene mediana más alta.")

md("""
### D2 · Estandarizar y elegir prioris (15 min)
Se **estandarizan** los predictores (z = (x − media)/sd) para que **una misma priori** sea razonable para todos los coeficientes; la profesora usa **N(0, 1)** para los coeficientes y N(0, 2.5²) para el intercepto (`autoscale = FALSE` en `rstanarm`). Las columnas ya se estandarizaron en la celda de C1:
""")
code("""
print(Z.describe().loc[["mean", "std", "min", "max"]].round(4))
print("media sin estandarizar:", medias.round(2).to_dict(), "| sd:", sds.round(2).to_dict())
b_ejemplo = np.array([-0.9, 1.0, 0.5, 0.6])
print("log posteriori en un punto de ejemplo:", round(float(logp(b_ejemplo)), 3))
""")
md("""
Los mínimos y máximos quedan en rangos parecidos (glu_z de [[pima.rango_z.glu.0|2]] a [[pima.rango_z.glu.1|2]]); las sd son 1. Un **coeficiente estandarizado** mide el efecto de **una desviación estándar** del predictor sobre el logit, y por eso N(0, 1) (efecto de ±2 en log-odds con 95 % de probabilidad) es una priori débilmente informativa. El intercepto es el logit de una persona **promedio**: con probabilidad 0.34 en la muestra, ≈ −0.66, de ahí la N(0, 2.5²).

✍️ **Ejercicio D2.** Calcula el logit del intercepto esperado si todos los predictores valen su media y la proporción de diabetes es 0.34. ¿Está dentro de la priori N(0, 2.5²) del intercepto? ¿Y la posteriori −0.92 (la de la profesora)?
""")
code("""
# tu código aquí
""")
solution("""
p0 = pima.diabetes.mean()
print("logit(0.34) =", round(float(np.log(p0 / (1 - p0))), 3), "| z de la priori N(0, 2.5):", round(float(np.log(p0 / (1 - p0)) / 2.5), 2))
""", "logit(0.34) ≈ −0.66; está a 0.26 desviaciones de la media de la priori: perfectamente compatible. La posteriori (−0.92) es menor porque el efecto de las otras variables cambia la referencia.")

md("""
### D3 · Ajuste: 4 cadenas, 2 000 iteraciones, 1 000 de calentamiento (20 min)
Misma configuración que `stan_glm(..., chains = 4, iter = 2000, warmup = 1000)`: cuatro cadenas desde **puntos de partida dispersos**, las primeras 1 000 iteraciones son calentamiento (aquí se usan para adaptar la covarianza de la propuesta: 2.38²/d veces la covarianza reciente) y quedan **4 × 1 000 = 4 000** simulaciones. Es un Metropolis por bloques con propuesta normal multivariada.
""")
code("""
inicios = np.random.default_rng(24100).normal(0, 1.5, size=(4, 4))
cadenas, aceptacion = [], []
for k in range(4):
    c_, a_ = metropolis_bloque(logp, inicios[k], 1000, 2000, np.random.default_rng(24110 + k))
    cadenas.append(c_)
    aceptacion.append(a_)
cadenas = np.array(cadenas)                      # (4 cadenas, 1000 iteraciones, 4 parámetros)
print("forma:", cadenas.shape, "| aceptación por cadena:", np.round(aceptacion, 3))
""")
md("""
La aceptación media es [[pima.aceptacion_media|2]] (≈ 31–36 %), cerca de lo óptimo para 4 dimensiones.

### D4 · Diagnósticos (15 min)
Trazas de las 4 cadenas y, por parámetro, **R-hat** y **ESS** (en R: `summary(modelo)` columnas `Rhat` y `n_eff`).
""")
code("""
nombres = ["(Intercept)", "glu_z", "bmi_z", "age_z"]
fig, axs = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
for j, ax in enumerate(axs.ravel()):
    for k in range(4):
        ax.plot(cadenas[k, :, j], lw=0.5)
    ax.set_ylabel(nombres[j])
plt.show()
diag = pd.DataFrame([(nm, *ess_rhat(cadenas[:, :, j])[::-1]) for j, nm in enumerate(nombres)], columns=["parámetro", "R-hat", "ESS"]).set_index("parámetro")
display(diag.round(3))
""")
md("""
Las trazas se superponen y no hay tendencias ni tramos pegados (lo que la profesora quería ver). **R-hat** va de [[AG.rhat|3]] a [[I.rhat|3]] (el del intercepto es 1.013, apenas sobre 1.01) y el **ESS** de ≈ 250 a 330 de 4 000. Compara con Stan: n_eff entre 3 668 y 4 139. **Un Metropolis de paseo aleatorio es ≈ 12 veces menos eficiente que NUTS** en este problema: con la misma longitud, mucha menos información. Remedio: correr más iteraciones (D8).

✍️ **Ejercicio D4.** ¿Qué fracción de las 4 000 simulaciones «vale» cada parámetro (ESS/4 000)? ¿Cuál es el error de Monte Carlo de la **media** de glu_z (sd posterior/√ESS) y cuántos decimales de la media son confiables?
""")
code("""
# tu código aquí
""")
solution("""
muestras = cadenas.reshape(-1, 4)
print((diag["ESS"] / 4000).round(3).to_dict())
sd_g = muestras[:, 1].std(ddof=1)
print("error MC de la media de glu_z:", round(float(sd_g / np.sqrt(diag.loc["glu_z", "ESS"])), 4))
""", "ESS/4 000 ≈ 0.06–0.08 (6–8 %); error de Monte Carlo ≈ 0.012: solo el primer decimal de la media es seguro. Con 100 000+ iteraciones se obtiene el segundo.")

md("""
### D5 · Resúmenes, odds ratios y predicciones (20 min)
Todo sale de las muestras: media, mediana, intervalo creíble (cuantiles 2.5 % y 97.5 %), **P(β > 0 | y)** y **odds ratio** = exp(β). Como los predictores están estandarizados, el OR compara pacientes que difieren en **una desviación estándar** del predictor.
""")
code("""
muestras = cadenas.reshape(-1, 4)
resumen = pd.DataFrame({
    "media": muestras.mean(0), "mediana": np.median(muestras, 0),
    "lím. inf. 95 %": np.quantile(muestras, 0.025, axis=0), "lím. sup. 95 %": np.quantile(muestras, 0.975, axis=0),
    "P(β > 0)": (muestras > 0).mean(0),
    "OR mediana": np.exp(np.median(muestras, 0)), "OR inf.": np.exp(np.quantile(muestras, 0.025, axis=0)), "OR sup.": np.exp(np.quantile(muestras, 0.975, axis=0)),
}, index=nombres)
display(resumen.round(3))
""")
md("""
Mis medianas ([[I.mediana|3]], [[G.mediana|3]], [[BM.mediana|3]], [[AG.mediana|3]]) están cerca de las de la profesora (−0.921, 0.966, 0.548 y 0.573); las diferencias vienen del **error de Monte Carlo** (ESS ≈ 300) y de que son generadores y algoritmos distintos (D8 lo muestra con una corrida larga). **Interpretación (glucosa)**: mediana [[G.mediana|3]], intervalo creíble 95 % [[[G.q025|3]]; [[G.q975|3]]]; el OR mediano [[G.or_med|2]] [[[G.or_lo|2]]; [[G.or_hi|2]]] significa que +1 desviación estándar de glucosa (≈ 31 mg/dl) multiplica los odds de diabetes por ≈ 2.6, manteniendo IMC y edad constantes. Los tres intervalos están **a la derecha de cero**.

Predicciones: **probabilidad de diabetes** para dos perfiles nuevos (cuidado: se estandarizan con la **media y sd de los datos de ajuste**):
""")
code("""
nuevos = pd.DataFrame({"perfil": ["Perfil A", "Perfil B"], "glu": [100, 160], "bmi": [25, 35], "age": [30, 55]})
Zn = np.column_stack([np.ones(2)] + [(nuevos[c_] - medias[c_]) / sds[c_] for c_ in cols])
prob = 1 / (1 + np.exp(-(muestras @ Zn.T)))                 # (4000, 2): una probabilidad por muestra y perfil
display(pd.DataFrame({"perfil": nuevos.perfil, "media": prob.mean(0), "mediana": np.median(prob, 0),
                      "lím. inf. 95 %": np.quantile(prob, 0.025, axis=0), "lím. sup. 95 %": np.quantile(prob, 0.975, axis=0)}).round(3))
""")
md("""
El Perfil A (glu 100, IMC 25, 30 años) tiene probabilidad [[pred_pima.perfiles.0.media|3]] ([[pred_pima.perfiles.0.lo|3]] a [[pred_pima.perfiles.0.hi|3]]) y el B (160, 35, 55) [[pred_pima.perfiles.1.media|3]] ([[pred_pima.perfiles.1.lo|3]] a [[pred_pima.perfiles.1.hi|3]]); la profesora obtuvo 0.085 y 0.827. Notar que el intervalo es de la **probabilidad** (incertidumbre de los coeficientes), no de un caso individual.

✍️ **Ejercicio D5.** Calcula P(OR de glucosa > 2 | y) y P(OR de IMC > OR de edad | y) con las muestras. ¿Qué responde cada una?
""")
code("""
# tu código aquí
""")
solution("""
or_glu = np.exp(muestras[:, 1])
print("P(OR_glu > 2) =", round(float(np.mean(or_glu > 2)), 3))
print("P(OR_bmi > OR_age) =", round(float(np.mean(muestras[:, 2] > muestras[:, 3])), 3))
""", "La primera (≈ 0.9) dice que es muy probable que +1 sd de glucosa más que duplique los odds. La segunda (≈ 0.42) compara los efectos de IMC y edad: no hay evidencia de que uno supere al otro. Son preguntas imposibles de contestar con un solo valor p.")

md("""
### D6 · pp_check y exactitud aparente (15 min)
**Chequeo predictivo posterior**: se simulan 200 mujeres con cada muestra posterior y se compara, por ejemplo, el **número de diabéticas** con el observado ([[pred_pima.obs]]). También la **matriz de confusión** con la probabilidad media por paciente.
""")
code("""
p_i = 1 / (1 + np.exp(-(muestras @ X.T)))                   # (4000, 200)
rng = np.random.default_rng(24951)
idx = rng.choice(muestras.shape[0], 1000, replace=False)
rep = np.array([rng.binomial(1, p_i[i]).sum() for i in idx])
print(f"diabéticas simuladas: media {rep.mean():.1f}, 2.5 %-97.5 % = {np.quantile(rep, [0.025, 0.975])} | observadas: {int(y.sum())}")
pred = (p_i.mean(0) >= 0.5).astype(int)
conf = pd.crosstab(pd.Series(y, name="Observado"), pd.Series(pred, name="Predicho"))
display(conf)
print("exactitud aparente:", np.mean(pred == y))
""")
md("""
Lo observado ([[pred_pima.obs]]) cae en el centro de lo simulado ([[pred_pima.rep_lo|0]]–[[pred_pima.rep_hi|0]]): el modelo reproduce esa característica. La exactitud **0.76** (113, 19, 29 y 39) es **aparente**: se mide con los mismos datos del ajuste y es optimista (alerta D9); para evaluar de verdad hace falta validación (C38, C43).

✍️ **Ejercicio D6.** Repite el pp_check con otro estadístico: la **proporción de diabéticas entre las mujeres de más de 45 años** (observada vs simulada). ¿El modelo la reproduce?
""")
code("""
# tu código aquí
""")
solution("""
viejas = (pima["age"] > 45).to_numpy()
obs = y[viejas].mean()
rng = np.random.default_rng(3)
sim = np.array([rng.binomial(1, p_i[i][viejas]).mean() for i in idx])
print(f"mayores de 45: n = {viejas.sum()}, observada = {obs:.3f}, simulada media = {sim.mean():.3f}, IC = {np.quantile(sim, [0.025, 0.975]).round(3)}")
""", "El valor observado (≈ 0.63, con 27 mujeres) queda dentro del intervalo simulado (≈ 0.44–0.85; media 0.65): el modelo lo reproduce sin problemas.")

md("""
### D7 · Frente al glm y al bootstrap (20 min)
Conexión con **C17–21** (glm clásico) y **C43** (bootstrap): tres formas de medir la incertidumbre del coeficiente de glucosa.
""")
code("""
glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(pd.DataFrame({"estimación": glm.params, "error estándar": glm.bse, "z": glm.tvalues, "p": glm.pvalues}, index=nombres).round(4))
rng = np.random.default_rng(24952)
B = 2000
bs = np.empty(B)
for b in range(B):
    i = rng.choice(len(y), len(y), replace=True)
    bs[b] = sm.GLM(y[i], X[i], family=sm.families.Binomial()).fit().params[1]
print(f"glu_z: Wald {glm.params[1]:.3f} ± {1.96 * glm.bse[1]:.3f} | bootstrap sd {bs.std(ddof=1):.3f}, IC percentil {np.quantile(bs, [0.025, 0.975]).round(3)} | posteriori sd {muestras[:, 1].std(ddof=1):.3f}, IC creíble {np.quantile(muestras[:, 1], [0.025, 0.975]).round(3)}")
""")
md("""
El glm da exactamente los números de la profesora (−0.9243, 0.9769, 0.5632, 0.5770; AIC 196.39). Para glu_z: **bootstrap** sd [[pred_pima.boot.sd|3]], IC [[[pred_pima.boot.lo|3]]; [[pred_pima.boot.hi|3]]]; **posteriori** sd [[pred_pima.post.sd|3]], IC creíble [[[pred_pima.post.lo|3]]; [[pred_pima.post.hi|3]]]; **Wald** 0.977 [0.577; 1.377]. Casi coinciden porque hay 200 datos y una priori débil. Cambia **la lectura**: el valor p dice cuán raro sería el dato si β = 0; la posteriori dice cuán probable es que β sea positivo (P ≈ 1.000) y permite P(OR > 2 | y), predicciones con incertidumbre, etc. El bootstrap mide la variabilidad de un **estimador** al remuestrear datos; la posteriori, la incertidumbre **sobre el parámetro** dados el modelo, la priori y los datos.

✍️ **Ejercicio D7.** Sensibilidad a la priori: ajusta con coeficientes N(0, 0.25²) (mucho más informativa) y compara medias posteriores con las de N(0, 1). ¿Qué coeficiente se encoge más? ¿Por qué?
""")
code("""
# tu código aquí
""")
solution("""
logp_fuerte = lambda b: log_posterior_logistica(b, X, y, sd_coef=0.25)
ch_f = np.array([metropolis_bloque(logp_fuerte, inicios[k], 1000, 3000, np.random.default_rng(900 + k))[0] for k in range(4)])
m_f = ch_f.reshape(-1, 4).mean(0)
display(pd.DataFrame({"N(0, 1)": muestras.mean(0), "N(0, 0.25²)": m_f, "glm": glm.params}, index=nombres).round(3))
""", "Con una priori estrecha los coeficientes se encogen hacia 0 (glu_z: de ≈ 0.94 a ≈ 0.67) porque la priori pesa más: el que tiene mayor efecto estimado pierde más en términos absolutos. Con n = 200 y N(0, 1) la priori casi no se nota; con N(0, 0.25²) sí.")

md("""
### D8 · Corrida larga frente a rstanarm (15 min)
Para saber si el muestreador a mano **reproduce** a `stan_glm`, se corren 4 cadenas de 30 000 iteraciones (5 000 de calentamiento): el error de Monte Carlo baja ≈ 8 veces.
""")
code("""
larga = np.array([metropolis_bloque(logp, inicios[k], 5000, 30000, np.random.default_rng(24120 + k))[0] for k in range(4)])
ml = larga.reshape(-1, 4)
rstan = {"mediana": [-0.921, 0.966, 0.548, 0.573], "sd": [0.194, 0.204, 0.194, 0.186], "q025": [-1.321, 0.580, 0.189, 0.210], "q975": [-0.555, 1.386, 0.951, 0.926]}
tabla = pd.DataFrame({
    "mediana (mío)": np.median(ml, 0), "mediana (rstanarm)": rstan["mediana"],
    "sd (mío)": ml.std(0, ddof=1), "sd (rstanarm)": rstan["sd"],
    "2.5 % (mío)": np.quantile(ml, 0.025, axis=0), "2.5 % (rstanarm)": rstan["q025"],
    "97.5 % (mío)": np.quantile(ml, 0.975, axis=0), "97.5 % (rstanarm)": rstan["q975"],
    "R-hat": [ess_rhat(larga[:, :, j])[1] for j in range(4)], "ESS": [ess_rhat(larga[:, :, j])[0] for j in range(4)]}, index=nombres)
display(tabla.round(3))
""")
md("""
Las medianas coinciden con las de la profesora a ±0.01 ([[I.larga_mediana|3]] frente a −0.921; [[G.larga_mediana|3]] a 0.966; [[BM.larga_mediana|3]] a 0.548; [[AG.larga_mediana|3]] a 0.573), también las sd y los intervalos, y ahora R-hat ≈ 1.001 y ESS de 4 600 a 8 000: **la posteriori es la misma**; cambia solo el algoritmo (Stan lo logra con 4 000 iteraciones por NUTS). Con 4 × 25 000 iteraciones se obtiene la precisión que NUTS alcanza con 4 000.

### D9 · Alertas verificadas y tabla R ↔ Python (10 min)
**Imprecisiones del material, verificadas con lo que acabas de ejecutar:**
1. **rstanarm no usa Metropolis–Hastings**: `stan_glm` usa **HMC/NUTS**; el algoritmo explicado en clase (y el de este notebook) es un Metropolis. «Warmup» en Stan **adapta** el paso: no es solo burn-in. (C2–C8, D3: ESS ≈ 300 aquí frente a ≈ 4 000 de Stan.)
2. **Propuesta asimétrica**: la diapositiva 24 omite el cociente de propuestas (Hastings); sin él la media sale 0.778 en lugar de 0.667 (C8).
3. **«Más simulaciones o cadenas pueden empeorar la convergencia»**: impreciso. El error de Monte Carlo baja siempre como 1/√M (B2); lo que no se arregla con más iteraciones es un **modelo mal especificado** (diapositiva 26).
4. **«Monte Carlo cuando no puedo calcular la integral; MCMC cuando ni siquiera normalizada puedo tener la expresión»**: el Monte Carlo ordinario requiere **saber simular** de la posteriori; MCMC solo requiere evaluarla salvo una constante.
5. **Diapositiva 20 («mientras mayor sea M, menor el error»)** solo vale con muestras independientes; con una cadena manda el ESS (C6: 5 000 iteraciones, ESS 2).
6. **Diapositiva 28** anuncia `Clase24_Bayes_R.Rmd` (Normal–Normal, predictiva, A/B, Metropolis; «R y Python»); lo entregado es `Clase24.ipynb` (solo R, rstanarm). Este notebook cubre todo en Python.
7. **Cuaderno de la profesora**: `posterior_linpred(transform = TRUE)` está obsoleto (usar `posterior_epred`); la exactitud 0.76 es aparente; «datos real» es un error de tipeo.
8. En clase se llamó «intervalo de confianza» a un intervalo para los odds ratios: es **creíble** (la profesora lo corrigió).
9. **«Python es una caja negra, R está bien documentado»** es una opinión: Stan, PyMC y brms están igual de documentados; lo importante es saber qué cadenas, iteraciones y diagnósticos se usaron.

**Tabla de correspondencia R ↔ Python**

| Tarea | R (profesora) | Python (este notebook) |
|---|---|---|
| Datos | `data("Pima.tr", package = "MASS")` | `pd.read_csv("Guias/datos/pima_tr.csv")` |
| Respuesta binaria | `ifelse(type == "Yes", 1, 0)` | `(pima["type"] == "Yes").astype(int)` |
| Estandarizar | `(x - mean(x)) / sd(x)` | `(x - x.mean()) / x.std(ddof=1)` |
| Prioris | `normal(0, 1, autoscale = FALSE)`; `normal(0, 2.5, ...)` | `log_posterior_logistica(sd_coef=1, sd_int=2.5)` |
| Ajuste bayesiano | `stan_glm(..., chains = 4, iter = 2000, warmup = 1000, seed = 2026)` | `metropolis_bloque(...)` por cadena |
| Resumen con Rhat / n_eff | `summary(modelo)` | `ess_rhat(cadenas[:, :, j])` |
| Trazas | `mcmc_trace(as.array(modelo))` | `plt.plot(cadenas[k, :, j])` |
| Muestras posteriores | `as.matrix(modelo)` | `cadenas.reshape(-1, 4)` |
| Intervalo creíble | `quantile(x, c(0.025, 0.975))` | `np.quantile(x, [0.025, 0.975])` |
| Odds ratio | `exp(median(x))` | `np.exp(np.median(x))` |
| Probabilidad por perfil | `posterior_epred(modelo, newdata)` | `1 / (1 + np.exp(-(muestras @ Zn.T)))` |
| pp_check | `pp_check(modelo, plotfun = "bars")` | simular con `rng.binomial(1, p_i)` |
| glm clásico | `glm(..., family = binomial)` | `sm.GLM(y, X, family=sm.families.Binomial()).fit()` |
| Beta / normal | `rbeta`, `pbeta`, `qbeta` / `pnorm` | `stats.beta.rvs/cdf/ppf` / `stats.norm.cdf` |
| Semilla | `set.seed(2026)` | `np.random.default_rng(2026)` (**otro generador**: no da los mismos números) |
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre (35 min, SIN mirar lo anterior)

Un equipo de producto quiere saber si una **nueva pantalla de pago** sube la tasa de conversión. Datos: **control 90 de 600** y **tratamiento 117 de 700**. Prioris Beta(1, 1). Además, el tiempo de carga (ms) de 16 sesiones tiene media 520 ms; se sabe σ = 80 ms y la priori es μ ~ N(500, 40²).

1. **Plan previo**: en un comentario, escribe qué probabilidad vas a calcular, qué mejora mínima consideras relevante (usa 1 punto porcentual) y cómo medirás el error de simulación.
2. **Monte Carlo** (M = 100 000, `default_rng(2400)`): P(B > A), intervalo creíble 95 % de la diferencia y P(mejora > 0.01) con su error estándar de simulación.
3. **Normal–Normal** del tiempo de carga: posteriori (media, sd), P(μ > 510 | y) y la **predictiva** de la próxima sesión (sd).
4. **Metropolis**: con el núcleo θ⁹⁰(1 − θ)⁵¹⁰ de la posteriori de la tasa del control (Beta(91, 511)), 4 cadenas de 4 000 iteraciones desde 0.05, 0.15, 0.30 y 0.60 con τ = 0.015 y semillas 1 a 4. Descarta 1 000 de burn-in; reporta aceptación, R-hat, ESS y compara la media con la exacta.
5. **Conclusión sin código**: ¿hay evidencia clara de mejora? ¿Qué incertidumbre cubre cada cifra y cuál no (modelo, priori, sesgos de diseño)?
""")
code("""
# tu código aquí
""")
md("""
✍️ **Mi recomendación y mis advertencias:**

""")
solution("""
# 1) plan: P(B > A) y P(mejora > 0.01); error de simulación = sqrt(p(1-p)/M)
rng = np.random.default_rng(2400)
pA = stats.beta(1 + 90, 1 + 510).rvs(100000, random_state=rng)
pB = stats.beta(1 + 117, 1 + 583).rvs(100000, random_state=rng)
d = pB - pA
p_mej = np.mean(d > 0.01)
print(f"P(B>A) = {np.mean(d > 0):.4f} | dif. media = {d.mean():.4f} | IC 95 % = {np.quantile(d, [0.025, 0.975]).round(4)}")
print(f"P(mejora > 0.01) = {p_mej:.4f} (error de simulación {np.sqrt(p_mej * (1 - p_mej) / 100000):.4f})")
# 3) Normal-Normal del tiempo de carga
mc_, vc_ = normal_normal(500, 40, 80, 16, 520)
print(f"tiempo: media posteriori = {mc_:.2f}, sd = {np.sqrt(vc_):.3f}, P(mu > 510) = {stats.norm.sf(510, mc_, np.sqrt(vc_)):.4f}, sd predictiva = {np.sqrt(80**2 + vc_):.3f}")
# 4) Metropolis, 4 cadenas
lp_c = log_beta_nucleo(91, 511)
cs, ac = [], []
for s_, st_ in zip([1, 2, 3, 4], [0.05, 0.15, 0.30, 0.60]):
    c_, a_ = metropolis(lp_c, st_, 0.015, 4000, np.random.default_rng(s_))
    cs.append(c_[1000:]); ac.append(a_)
cs = np.array(cs)
e_, r_ = ess_rhat(cs)
print(f"aceptación = {np.round(ac, 3)} | R-hat = {r_:.3f} | ESS = {e_:.0f} | media cadenas = {cs.mean():.4f} (exacta {91 / 602:.4f})")
""", "Referencia: P(B > A) ≈ 0.80; la diferencia ≈ 0.017 con IC ≈ [−0.023; 0.056]; P(mejora > 0.01) ≈ 0.63 (error ≈ 0.0015). Tiempo: media ≈ 516.0, sd ≈ 17.9, P(μ > 510) ≈ 0.63, sd predictiva ≈ 82.0. Metropolis: aceptación ≈ 70 %, R-hat ≈ 1.003, ESS ≈ 1 400 y media ≈ 0.1516 (exacta 0.1512). **Conclusión**: hay ≈ 80 % de probabilidad de que B sea mejor y ≈ 63 % de que la mejora supere 1 punto; el intervalo incluye 0, así que no hay evidencia clara. Los intervalos no cubren sesgos de diseño (asignación no aleatoria, efecto novedad) y dependen de la priori y del modelo.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. En el modelo Normal–Normal, ¿qué es una precisión y cómo se combinan las de la priori y los datos? ¿Qué pasa con la media posterior al crecer n?
2. ¿Por qué la distribución predictiva es más dispersa que usar un θ fijo? Escribe la varianza predictiva del caso Normal.
3. ¿Cómo se aproxima P(θ > c | y) por Monte Carlo? ¿Cómo cambia el error estándar al multiplicar M por 100?
4. ¿Por qué no basta una grilla ni una fórmula en la regresión logística bayesiana? ¿Qué se necesita saber de la posteriori para usar Metropolis?
5. Describe un paso de Metropolis: propuesta, razón r, probabilidad de aceptación y qué se guarda si se rechaza.
6. ¿Qué pasa si el tamaño de paso es demasiado pequeño? ¿Y demasiado grande? ¿Cómo se ve en la traza, la aceptación y el ESS?
7. ¿Qué es el burn-in? ¿Qué no arregla?
8. ¿Qué mide la autocorrelación y qué es el ESS? ¿Por qué el thinning no mejora el ESS?
9. ¿Qué compara el R-hat y por qué se parten las cadenas de puntos dispersos? ¿Basta con R-hat ≈ 1?
10. ¿Cómo se obtiene un intervalo creíble y P(β > 0 | y) a partir de las muestras de la cadena? ¿Qué es un odds ratio con un predictor estandarizado?
11. ¿Qué diferencia hay entre el bootstrap del glm y la posteriori? ¿Y entre un IC de confianza y uno creíble?
12. ¿Qué algoritmo usa realmente `stan_glm` y qué hace «warmup»? ¿Qué le falta a la descripción de la diapositiva 24 para propuestas asimétricas?

<details><summary><b>Respuestas</b></summary>

1. Precisión = 1/varianza. La posterior suma las precisiones de la priori (1/s₀²) y de los datos (n/σ²); la media posterior es el promedio de m₀ y ȳ ponderado por ellas. Con n grande los datos dominan: la media se acerca a ȳ y la sd baja ≈ σ/√n.
2. Porque promedia sobre todos los θ posibles: suma el azar del dato nuevo y la incertidumbre sobre θ. En el caso Normal, Var(Ỹ) = σ² + sₙ².
3. Con la fracción de simulaciones que cumplen θ > c, (1/M) Σ 1{θ⁽ᵐ⁾ > c}. El error estándar √(p(1 − p)/M) se divide por 10 al multiplicar M por 100.
4. Porque la posteriori no tiene familia conocida y una grilla cuesta 100^d evaluaciones. Metropolis solo necesita evaluar el **núcleo** p(y | θ)p(θ) (la constante se cancela en r).
5. Propone θ* = θ + τ·N(0,1); calcula r = núcleo(θ*)/núcleo(θ); acepta con probabilidad min(1, r) (si u < α). Si rechaza, la cadena repite θ.
6. Paso pequeño: aceptación altísima pero la cadena avanza lento, traza suave, autocorrelación alta, ESS bajo. Paso grande: aceptación bajísima, traza con tramos pegados, ESS bajo. El punto medio (aceptación 25–50 %) maximiza el ESS.
7. Descartar las primeras iteraciones, que dependen del punto de partida. No arregla un modelo mal especificado ni una priori inadecuada.
8. La autocorrelación mide la dependencia entre θ⁽ᵗ⁾ y θ⁽ᵗ⁺ᵏ⁾; el ESS ≈ M/(1 + 2Σρₖ) es cuántas muestras independientes vale la cadena. Guardar 1 de cada k tira información, así que el ESS no sube.
9. Compara la varianza entre cadenas con la de dentro; se parten de puntos dispersos para que, si no han convergido, se note. No basta: puede ser ≈ 1 con ESS bajo o si todas las cadenas quedan en la misma región equivocada.
10. Cuantiles 2.5 % y 97.5 % de las muestras; P(β > 0 | y) = fracción de muestras positivas. El OR = exp(β): efecto multiplicativo en los odds de +1 desviación estándar del predictor, con los demás constantes.
11. El bootstrap mide la variabilidad del estimador al remuestrear datos (sin priori); la posteriori, la incertidumbre del parámetro dados modelo, priori y datos. El IC de confianza es propiedad del procedimiento; el creíble es una probabilidad sobre θ.
12. Usa HMC/NUTS. «Warmup» adapta el paso (y la matriz de masas), no es solo burn-in. A la diapositiva le falta el cociente de las densidades de propuesta q (corrección de Hastings).
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** series de tiempo (Clases 25–27) y, más adelante, evaluación estadística de modelos con bootstrap (Clase 43). Revisa `Guias/apuntes/C24_Ejercicios_Cuaderno_Bayesiana2.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
