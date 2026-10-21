"""Genera el notebook guiado de series de tiempo (Clases 25-27, con el cierre de la Clase 28).

Uso normal:      python notebook_c25_27.py              -> Guias/notebooks/C25-27_SeriesTiempo.ipynb
Modo de prueba:  C25_27_TEST=1 python notebook_c25_27.py -> las soluciones se emiten como celdas de código
                                                          (para ejecutar el notebook completo con nbclient).
Los números del texto se leen de c25_27_numeros.json (lo escribe figuras_c25_27.py) con marcas [[ruta|decimales]].
Solo usa numpy, pandas, matplotlib, scipy y statsmodels (sin pmdarima ni R)."""
import inspect
import json
import os
import re
from pathlib import Path

import figuras_c25_27 as F

TEST = os.environ.get("C25_27_TEST") == "1"
HERE = Path(__file__).resolve().parent
DIR = HERE.parent / "notebooks"
OUT = DIR / ("_C25-27_SeriesTiempo_test.ipynb" if TEST else "C25-27_SeriesTiempo.ipynb")
N = json.loads((HERE / "c25_27_numeros.json").read_text(encoding="utf-8"))
N["M03"], N["M02"], N["M10"], N["M20"], N["M11"] = (N["modelos"][k] for k in ("MA(3)", "MA(2)", "AR(1)", "AR(2)", "ARMA(1,1)"))
N["MAE"] = N["pron"]["mae"]
N["CBM"] = N["citybike"]["mae"]
N["VM"] = N["valid"]["mae"]
N["RES"] = N["adf"]["res"]
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
    return re.sub(r"\[\[([^\[\]|]+)(?:\|(\d+))?\]\]", val, text)


def md(text):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {}, "source": fill(text).strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


def solution(code_text, note=""):
    if TEST:
        code(code_text)
        return
    body = f"{fill(note)}\n\n" if note else ""
    src = f"<details><summary><b>Ver solución</b> (inténtalo al menos 5 minutos antes)</summary>\n\n{body}```python\n{code_text.strip()}\n```\n\n</details>"
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {}, "source": src})      # el código no pasa por fill() (puede contener [[...]])


NUCLEO = "\n\n\n".join(inspect.getsource(fn).rstrip() for fn in (
    F.acf_manual, F.pacf_yw, F.simular_arma, F.descomponer_aditiva, F.arma_css, F.ljung_box, F.adf_regresion, F.particion_temporal, F.ingenuo, F.mae, F.rmse, F.ar1_pronostico))

# ---------------------------------------------------------------- encabezado
md("""
# Clases 25–27 · Modelos predictivos de series de tiempo: de la descomposición a ARIMA y SARIMA
**Diplomado Data Science** · Prof. Cristian Vásquez (PUC) · se conecta con regresión y GLM (Clases 17–21), validación (Clases 30 y 38) y bootstrap (Clase 43) · incluye el cuaderno de cierre de la Clase 28

Material de apoyo:
- `Guias/04_SeriesTiempo_C25-27.pdf` — guía visual (12 páginas).
- `Guias/apuntes/04_Apuntes_SeriesTiempo_C25-27.pdf` — apuntes con ejemplos resueltos y pistas.
- `Guias/apuntes/04_Ejercicios_Cuaderno_SeriesTiempo.pdf` — ejercicios en papel (antes o después de este notebook).
- Material del profesor (carpeta «Clase 25, 26, 27 …»): presentaciones `01. Clase 1`, `01. Clase 2` y `01. Clase 4 Series de Tiempo.pdf` (portadas rotuladas «Clase 25», «Clase 25» y «Clase 27»), cuadernos **en R** `01. Código Clase 1` y `02. Código Clase 2`, y el cuaderno **en Python** del cierre `DS26_28_Cierre_de_Módulo.ipynb` (Clase 28). La sesión de repaso del sábado («Clase 3») no tiene archivos y el cuaderno en R de la Clase 27 no se entregó.
- Transcripciones: `VideosClases/Clase 25 … 27 …_cristian_vasquez.txt` (versión corregida).

⚠️ **Antes de empezar**: necesitas `numpy`, `pandas`, `matplotlib`, `scipy` y `statsmodels`. Si no los tienes:
```
pip install numpy pandas matplotlib scipy statsmodels
```
**No hace falta R ni `pmdarima`.** El profesor trabajó con R (`tseries`, `forecast`); aquí todo se reproduce en Python con `statsmodels` y con unas pocas funciones escritas a mano (ACF, PACF, simulación ARMA, ajuste por mínimos cuadrados condicionales como `tseries::arma`, Ljung-Box, Dickey-Fuller). La tabla R ↔ Python está al final.

**Datos** (en `Guias/datos/series_tiempo/`, copiados de la carpeta de la clase): pasajeros de EE. UU. (`Passengers.xlsx`), anillos de árbol de Idaho (`tree.xlsx`), IPC (`ipc_var.txt`), IPSA (`ipsa.txt`), PM2.5 (`pm25.xlsx`), desocupación (`desocupacion.xlsx`), el IAC del INE (`IAC_INE.xlsx`, que no se usó en clase y sirve para el ejercicio de cierre), la serie diaria de CityBike del cuaderno de cierre y la serie `USAccDeaths` (muertes accidentales de EE. UU. 1973–1978: es un dato de R que el profesor usa en la Clase 27 y **no** venía en la carpeta).

### Objetivos
1. Reconocer **ruido blanco**, estacionariedad, tendencia y estacionalidad, y **descomponer** una serie (Clase 25).
2. Calcular e interpretar la **ACF y la PACF**, y simular e identificar modelos **AR, MA y ARMA** (Clase 26).
3. **Estimar** (CSS y máxima verosimilitud), **elegir por AIC**, **diagnosticar** residuos (Ljung-Box) y **pronosticar con intervalos** (Clase 26).
4. Aplicar **modelos estacionales** (SARMA/SARIMA), **diferenciar** y usar el test de **Dickey-Fuller** (Clase 27).
5. **Validar en el tiempo** (corte y origen móvil) y comparar contra referencias ingenuas, nunca al azar.
6. Reproducir los resultados del profesor y reconocer lo impreciso del material.
7. Traducir **R ↔ Python**.

### Plan
| Parte | Min | Bloque |
|---|---|---|
| — | 10 | Preparación |
| A | 15 | A1 · Las series de la clase (y un error en el código del IPC) |
| A | 15 | A2 · Ruido blanco y bandas de la ACF |
| A | 15 | A3 · ACF a mano |
| A | 20 | A4 · Descomposición aditiva |
| A | 20 | A5 · Regresión con dummies y residuos autocorrelacionados |
| A | 15 | A6 · Armónicas y Box-Cox |
| A | 10 | A7 · Primera validación temporal |
| B | 20 | B1 · AR, MA y ARMA: ACF y PACF teóricas y muestrales |
| B | 10 | B2 · Estacionariedad e invertibilidad |
| B | 10 | B3 · Anillos de árbol: la serie |
| B | 25 | B4 · Cinco modelos: CSS frente a máxima verosimilitud |
| B | 20 | B5 · Diagnóstico: Ljung-Box y normalidad |
| B | 20 | B6 · Pronóstico, intervalos y validación a ciegas |
| C | 15 | C1 · Operador de rezago y diferencias |
| C | 15 | C2 · AR estacional simulado |
| C | 25 | C3 · SARMA con las muertes accidentales |
| C | 15 | C4 · Tendencia determinística y estocástica |
| C | 20 | C5 · Dickey-Fuller a mano y con statsmodels |
| C | 15 | C6 · Retornos del IPSA y MA con parámetros fijados |
| C | 25 | C7 · SARIMA de pasajeros con origen móvil |
| C | 20 | C8 · Cierre del módulo (Clase 28): CityBike |
| D | 15 | D1 · Puentes con C17–21 (HAC) y C43 (bootstrap) |
| D | 10 | D2 · Alertas verificadas y tabla R ↔ Python |
| — | 40 | Ejercicio de cierre **sin ayuda** (IAC) |

### Cómo usar este notebook
- Ejecuta las celdas en orden (`Shift + Enter`). Todo lo aleatorio usa semilla fija.
- Las celdas marcadas con ✍️ son **tuyas**: escribe el código antes de abrir la solución.
- Cada ejercicio tiene una solución desplegable. Intenta **al menos 5 minutos** antes de abrirla.
- Los números del texto fueron verificados contra la salida real; con otras versiones de las librerías puede variar la última cifra. El notebook completo tarda unos minutos.
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
from scipy import optimize, stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

carpeta = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Guias" / "datos" / "series_tiempo").exists()) / "Guias" / "datos" / "series_tiempo"


def leer(nombre, **kw):
    return pd.read_csv(carpeta / nombre, parse_dates=["fecha"], **kw)


def ts(df, col):
    \"\"\"Serie mensual con índice de fechas (inicio de mes).\"\"\"
    return pd.Series(df[col].values.astype(float), index=pd.DatetimeIndex(df["fecha"], freq="MS"))


pas_todo = leer("passengers.csv")                       # octubre 2002 - septiembre 2019
pas = pas_todo[pas_todo["fecha"] >= "2010-01-01"].reset_index(drop=True)      # los 117 meses que usa el profesor
tree = pd.read_csv(carpeta / "tree_idaho.csv")
ipc, ipsa, pm25, des, iac, usa = (leer(f"{n}.csv") for n in ("ipc_var", "ipsa", "pm25", "desocupacion", "iac_ine", "usaccdeaths"))
print({k: len(v) for k, v in dict(pasajeros=pas, arbol=tree, ipc=ipc, ipsa=ipsa, pm25=pm25, desocupacion=des, iac=iac, usacc=usa).items()})
""")
md("""
### Mini-repaso (Clases 17–21, 30, 38 y 43)
- **Regresión** (C17–19): `y = β₀ + β₁x + ε`; los errores deben ser independientes. En una serie de tiempo **no lo son**, y de ahí nace la clase.
- **AIC** (C20–21): −2 log L + 2k, menor es mejor.
- **Validación** (C30, C38): se evalúa con datos que el modelo no vio; en series, **hacia adelante en el tiempo**.
- **Bootstrap** (C43): remuestrear para medir la incertidumbre; aquí se compara con el intervalo de predicción.

### Funciones comunes
La celda siguiente define el **motor** del cuaderno: `acf_manual` y `pacf_yw` (ACF y PACF), `simular_arma` (como `arima.sim` de R), `descomponer_aditiva` (como `decompose`), `arma_css` (réplica de `tseries::arma`, la función del profesor), `ljung_box`, `adf_regresion` (Dickey-Fuller como regresión), `particion_temporal`, `ingenuo`, `mae`, `rmse` y `ar1_pronostico`. Léelas con calma: son las piezas de las tres clases.
""")
code(NUCLEO)

# ================================================================== PARTE A
md("""
## Parte A · Clase 25: ruido blanco, ACF, descomposición y regresión

### A1 · Las series de la clase (15 min)
El profesor mostró cuatro series: variación mensual del **IPC** (1988–2009), el **IPSA** (1990–2000), el **PM2.5** diario y la **desocupación** (2010–2026). Antes de modelar, siempre se mira el gráfico: ¿hay tendencia?, ¿varianza constante?, ¿estacionalidad?, ¿saltos?
""")
code("""
fig, axs = plt.subplots(2, 2, figsize=(11, 6))
axs[0, 0].plot(ipc["fecha"], ipc["var_ipc"], lw=0.9); axs[0, 0].set_title("IPC: variación mensual (%)")
i9 = ipsa[ipsa["fecha"] >= "1990-01-01"]
axs[0, 1].plot(i9["fecha"], i9["ipsa"], color="C4"); axs[0, 1].set_title("IPSA 1990-2000")
axs[1, 0].plot(pm25["fecha"], pm25["pm25"], lw=0.4, color="C2"); axs[1, 0].set_title("PM2.5 diario")
axs[1, 1].plot(des["fecha"], des["tasa"], color="C1"); axs[1, 1].set_title("Tasa de desocupación (%)")
plt.tight_layout(); plt.show()
print("PM2.5:", pm25["fecha"].min().date(), "->", pm25["fecha"].max().date(), "| desocupación máxima:", round(des["tasa"].max(), 1), "en", des.loc[des["tasa"].idxmax(), "fecha"].strftime("%Y-%m"))
""")
md("""
El PM2.5 de los datos va de **[[ejemplos.pm_ini]] a [[ejemplos.pm_fin]]** ([[ejemplos.pm_n]] días) aunque la diapositiva dice «1990–2000»; la desocupación llega a [[ejemplos.des_max|1]] % en [[ejemplos.des_max_fecha]] (la pandemia: una **intervención**, no un ciclo).

⚠️ **Un error en el código del IPC.** El cuaderno en R lee `ipc_var.txt` (una tabla de 12 filas = meses × 22 columnas = años) **por filas**, así que los valores quedan mezclados (enero de todos los años, luego febrero…). Leída por columnas, la serie tiene su orden temporal.

✍️ **Ejercicio A1.** Reproduce el error: arma la serie leyendo la tabla por filas y compara su ACF(1) con la de la serie ordenada. ¿Cuál es mayor y por qué?
""")
code("""
# tu código aquí
""")
solution("""
tabla = ipc["var_ipc"].values.reshape(22, 12)            # 22 años x 12 meses (orden correcto)
mal = tabla.T.reshape(-1)                                # lo que hace el código de R: lee por filas la tabla meses x años
print("ACF(1) serie bien ordenada:", round(acf_manual(ipc["var_ipc"].values, 1)[1], 3))
print("ACF(1) serie mezclada     :", round(acf_manual(mal, 1)[1], 3))
""", "Ordenada: [[ejemplos.ipc_acf1_ok|3]]; mezclada: [[ejemplos.ipc_acf1_codigo_R|3]]. Mezclar meses de años distintos rompe la dependencia temporal y **baja** la autocorrelación: un error de orden se esconde fácilmente porque el gráfico «se ve» como una serie.")

md("""
### A2 · Ruido blanco y las bandas de la ACF (15 min)
Ruido blanco: **media 0, varianza constante y covarianza 0**. Se simulan tres series de n = 144 (tendencia, varianza creciente y ruido blanco) y se dibuja su ACF con las bandas ±1.96/√n.
""")
code("""
rng = np.random.default_rng(250101)
n = 144
t = np.arange(1, n + 1)
sA = 2 + 4 * t / 24 + rng.normal(0, 2, n)           # tendencia
sB = rng.normal(0, 0.5 + 5 * t / n)                 # varianza creciente
sC = rng.normal(0, 2, n)                            # ruido blanco
banda = 1.96 / np.sqrt(n)
fig, axs = plt.subplots(2, 3, figsize=(12, 5))
for j, (s_, nm) in enumerate(zip((sA, sB, sC), ("a) tendencia", "b) varianza crece", "c) ruido blanco"))):
    axs[0, j].plot(s_, lw=1); axs[0, j].axhline(0, color="k", lw=0.6); axs[0, j].set_title(nm)
    r = acf_manual(s_, 20)
    axs[1, j].vlines(range(1, 21), 0, r[1:], lw=3); axs[1, j].axhspan(-banda, banda, color="orange", alpha=0.25)
    axs[1, j].set_ylim(-0.3, 1); axs[1, j].set_title(f"ACF · fuera de bandas: {int(np.sum(np.abs(r[1:]) > banda))} de 20")
plt.tight_layout(); plt.show()
""")
md("""
La serie con tendencia tiene ACF(1) = [[ruido.acf1_tend|2]] (decae lentamente); la de **varianza creciente** tiene ACF casi 0 ([[ruido.acf1_het|2]]) pero **no** es ruido blanco: la ACF solo revisa el supuesto de covarianza. La tercera tiene [[ruido.fuera_c]] de [[ruido.de_c]] autocorrelaciones fuera de la banda (±[[ruido.banda|3]]).

✍️ **Ejercicio A2.** Con ruido blanco de n = 144, ¿qué fracción de las 20 autocorrelaciones queda fuera de las bandas **en promedio**? Simula 2 000 series y compruébalo. ¿Qué te dice sobre «una autocorrelación fuera de la banda»?
""")
code("""
# tu código aquí
""")
solution("""
rng2 = np.random.default_rng(1)
fuera = [np.mean(np.abs(acf_manual(rng2.normal(size=144), 20)[1:]) > banda) for _ in range(2000)]
print("fracción media fuera de la banda:", round(float(np.mean(fuera)), 4))
print("series con al menos 1 de 20 fuera:", round(float(np.mean(np.array(fuera) > 0)), 3))
""", "Cerca de 4–5 % de las barras (≈ 1 de 20 por azar) y más de la mitad de las series de ruido blanco (≈ 56 % aquí; 64 % si las 20 fueran independientes) tiene **al menos una** barra fuera. Una sola barra aislada, sin patrón, no descarta el ruido blanco.")

md("""
### A3 · ACF a mano (15 min)
La diapositiva 21 define la ACF muestral: la suma de productos de desviaciones a distancia k sobre la suma de cuadrados, con la media de **toda** la serie. Verifica que `acf_manual` coincide con `statsmodels` y calcula a mano r₁ y r₂ de 8 valores (el ejemplo resuelto del apunte).
""")
code("""
y8 = np.array([12, 15, 14, 18, 17, 21, 20, 24], float)
d = y8 - y8.mean()
r1 = np.sum(d[:-1] * d[1:]) / np.sum(d ** 2)
r2 = np.sum(d[:-2] * d[2:]) / np.sum(d ** 2)
print(f"a mano: r1 = {r1:.4f}, r2 = {r2:.4f} | acf_manual: {acf_manual(y8, 2)[1:].round(4)} | statsmodels: {sm_acf(y8, nlags=2, fft=False)[1:].round(4)}")
y_pas = ts(pas, "INTERNATIONAL").values
print("máxima diferencia con statsmodels (pasajeros, 20 rezagos):", float(np.max(np.abs(acf_manual(y_pas, 20) - sm_acf(y_pas, nlags=20, fft=False)))))
print("PACF de los pasajeros, máx. diferencia:", float(np.max(np.abs(pacf_yw(y_pas, 20) - sm_pacf(y_pas, nlags=20, method="ywm")))))
""")
md("""
✍️ **Ejercicio A3.** ¿Qué pasaría si, en vez de la media global, se usara la media de cada tramo (`y[:-k]` y `y[k:]`) como si fuera `np.corrcoef`? Calcula ambas versiones para k = 1 y 2 y compáralas.
""")
code("""
# tu código aquí
""")
solution("""
for k in (1, 2):
    print(k, "ACF muestral:", round(acf_manual(y8, k)[k], 4), "| np.corrcoef por tramos:", round(float(np.corrcoef(y8[:-k], y8[k:])[0, 1]), 4))
""", "Difieren: la ACF muestral usa la media global y divide por la suma de cuadrados **total** (garantiza una función semidefinida positiva). La correlación por tramos no.")

md("""
### A4 · Descomposición aditiva (20 min)
Serie: pasajeros de vuelos internacionales de EE. UU., enero 2010–septiembre 2019 (n = [[descomp.n]]). Yₜ = Tₜ + Sₜ + εₜ. `decompose()` de R estima la tendencia con una media móvil **centrada 2×12** (se pierden 6 meses por extremo) y la estacionalidad con el promedio por mes.
""")
code("""
y = ts(pas, "INTERNATIONAL")
tend, est, ruido = descomponer_aditiva(y.values, 12)
ref = seasonal_decompose(y, model="additive", period=12)
print("diferencia máx. con statsmodels: tendencia", np.nanmax(np.abs(tend - ref.trend.values)), "| estacional", np.max(np.abs(est - ref.seasonal.values)))
print("NA de tendencia:", int(np.isnan(tend).sum()), "| mes de mayor efecto:", int(np.argmax(est[:12]) + 1), "(julio) =", round(est[:12].max() / 1e6, 2), "millones")
fig, axs = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
for ax, v, nm in zip(axs, (y.values, tend, est, ruido), ("observada", "tendencia", "estacional", "ruido")):
    ax.plot(y.index, v / 1e6); ax.set_ylabel(nm)
plt.tight_layout(); plt.show()
rr = ruido[~np.isnan(ruido)]
print("ACF(1) y ACF(12) del ruido:", acf_manual(rr, 12)[[1, 12]].round(3), "| banda:", round(1.96 / np.sqrt(len(rr)), 3))
""")
md("""
Después de descomponer, el ruido **no** es ruido blanco: ACF(1) = [[descomp.acf1_ruido|2]] y ACF(12) = [[descomp.acf12_ruido|2]] (banda ±[[descomp.banda|2]]). La descomposición no es el final: queda dependencia que modelar (el profesor lo mostró con `ggAcf(Ruido)`).

✍️ **Ejercicio A4.** El efecto estacional de cada mes se estima como el promedio de (y − tendencia) en ese mes y se recentra. Verifica a mano para julio: calcula el promedio de (y − tendencia) en los julios disponibles y réstale la media de los 12 promedios mensuales.
""")
code("""
# tu código aquí
""")
solution("""
desv = y.values - tend
prom = np.array([np.nanmean(desv[k::12]) for k in range(12)])
print("julio sin recentrar:", round(prom[6] / 1e6, 3), "| recentrado:", round((prom[6] - prom.mean()) / 1e6, 3), "| est[6]:", round(est[6] / 1e6, 3))
print("los 12 efectos suman:", round(float((prom - prom.mean()).sum()), 6))
""", "El recentrado hace que los 12 efectos sumen 0 (para que no se confundan con el nivel).")

md("""
### A5 · Regresión con tendencia y dummies; residuos autocorrelacionados (20 min)
El profesor ajusta `lm(International ~ Tiempo + Month)` con enero como referencia. Es la regresión de C17–19 con una tendencia lineal y 11 dummies mensuales. Los coeficientes de la diapositiva 33: intercepto 11 498 890, tendencia 68 688, julio 4 126 027.
""")
code("""
df = pas.copy()
df["t"] = np.arange(1, len(df) + 1)
df["mes"] = df["fecha"].dt.month
mod = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df).fit()
print(mod.params[["Intercept", "t", "C(mes)[T.7]", "C(mes)[T.8]", "C(mes)[T.2]"]].round(0).to_string())
print(f"R2 = {mod.rsquared:.4f} | R2 ajustado = {mod.rsquared_adj:.4f} | significativos (5 %): {int((mod.pvalues.drop('Intercept') < 0.05).sum())} de {len(mod.params) - 1}")
res = mod.resid.values
print(f"Durbin-Watson = {sm.stats.durbin_watson(res):.2f} | ACF(1) = {acf_manual(res, 1)[1]:.2f} | ACF(12) = {acf_manual(res, 12)[12]:.2f}")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
axs[0].plot(df["fecha"], df["INTERNATIONAL"] / 1e6); axs[0].plot(df["fecha"], mod.fittedvalues / 1e6); axs[0].set_title("observada y ajustada")
axs[1].plot(df["fecha"], res / 1e3); axs[1].axhline(0, color="k", lw=0.6); axs[1].set_title("residuos")
axs[2].vlines(range(1, 25), 0, acf_manual(res, 24)[1:], lw=3); axs[2].axhspan(-1.96 / np.sqrt(len(res)), 1.96 / np.sqrt(len(res)), color="orange", alpha=0.25); axs[2].set_title("ACF de los residuos")
plt.tight_layout(); plt.show()
""")
md("""
Se reproducen los coeficientes de la diapositiva (intercepto [[regresion.intercepto|0]], tendencia [[regresion.tiempo|0]] por mes, julio +[[regresion.m7|0]]); R² ajustado **[[regresion.r2aj|3]]** (el profesor dijo «98 %»; más adelante «97 %»). Pero el Durbin-Watson es **[[regresion.dw|2]]** (≈ 2 indicaría residuos no autocorrelacionados) y la ACF(1) de los residuos [[regresion.acf1|2]]: el supuesto de independencia de C17–19 **se rompe**. Los coeficientes siguen siendo consistentes, pero los errores estándar y los valores p del `summary()` **no son confiables**.

✍️ **Ejercicio A5.** Recalcula el error estándar del coeficiente de tendencia con errores robustos a autocorrelación (HAC, `cov_type="HAC"`, 12 rezagos) y compáralo con el de OLS.
""")
code("""
# tu código aquí
""")
solution("""
hac = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print("error estándar de la tendencia: OLS", round(float(mod.bse["t"]), 0), "| HAC", round(float(hac.bse["t"]), 0), "| razón", round(float(hac.bse["t"] / mod.bse["t"]), 2))
""", "OLS: [[puentes.bse_ols|0]]; HAC: [[puentes.bse_hac|0]] (×[[puentes.razon|2]]). Con residuos positivamente autocorrelacionados el error estándar de OLS está **subestimado**.")

md("""
### A6 · Armónicas y Box-Cox (15 min)
- **Regresión armónica** (diapositivas 37–40): un ciclo suave se modela con `sen` y `cos`; con 1, 2 y 3 frecuencias se arman patrones más ricos. La estacionalidad con «saltos» (diciembre) se modela mejor con dummies.
- **Box-Cox**: (Y^α − 1)/α estabiliza una varianza que crece con el nivel; α = 0 es el logaritmo.
""")
code("""
t = np.arange(1, len(y) + 1)
def armonicas(t, K, s=12):
    cols = [np.ones_like(t, float), t.astype(float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / s), np.cos(2 * np.pi * k * t / s)]
    return np.column_stack(cols)
for K in (1, 2, 3, 6):
    r = sm.OLS(y.values, armonicas(t, K)).fit()
    print(f"K = {K}: R2 ajustado = {r.rsquared_adj:.4f}")
print(f"dummies (A5): {mod.rsquared_adj:.4f}")
i90 = ts(ipsa[ipsa["fecha"] >= "1990-01-01"], "ipsa")
alpha = stats.boxcox_normmax(i90.values, method="mle")
print(f"alpha de Box-Cox del IPSA (máx. verosimilitud): {alpha:.2f}")
m = len(i90) // 2
print("sd de los saltos mensuales, 1.ª mitad vs 2.ª:", np.diff(i90.values[:m]).std().round(0), np.diff(i90.values[m:]).std().round(0))
print("sd de los log-retornos, 1.ª mitad vs 2.ª:", np.diff(np.log(i90.values[:m])).std().round(3), np.diff(np.log(i90.values[m:])).std().round(3))
""")
md("""
Con 6 armónicas se obtiene **exactamente** el mismo ajuste que con las 11 dummies (R² ajustado 0.9777: ambas generan el mismo espacio de funciones con periodo 12); con 1 a 3, menos, pero con muchos menos parámetros. El IPSA muestra el problema: la desviación de los saltos mensuales sube de [[boxcox.sd_dif_mitad1|0]] a [[boxcox.sd_dif_mitad2|0]] puntos, pero la del **log-retorno** se mantiene ([[boxcox.sd_dlog_mitad1|3]] y [[boxcox.sd_dlog_mitad2|3]]). La máxima verosimilitud de Box-Cox da α̂ = [[boxcox.alpha_ipsa|2]], cercano a 1: **Box-Cox no siempre pide transformar**.

✍️ **Ejercicio A6.** Calcula α̂ de Box-Cox para los pasajeros internacionales completos (2002–2019) y explica qué transformación sugiere.
""")
code("""
# tu código aquí
""")
solution("""
yy = ts(pas_todo, "INTERNATIONAL")
print("alpha Box-Cox pasajeros 2002-2019:", round(float(stats.boxcox_normmax(yy.values, method="mle")), 2))
""", "α̂ ≈ [[boxcox.alpha_pasajeros|2]]: cercano a 0, es decir, el **logaritmo** estabiliza la varianza (la estacionalidad crece con el nivel: serie multiplicativa).")

md("""
### A7 · Primera validación temporal (10 min)
Un modelo de regresión con dummies se evalúa **hacia adelante**: entrenar con el pasado y probar con los últimos 12 meses. Barajar al azar mezclaría el futuro con el entrenamiento.
""")
code("""
tr, te = df.iloc[:-12], df.iloc[-12:]
m_tr = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", tr).fit()
pred = m_tr.predict(te)
print(f"MAE últimos 12 meses = {mae(te['INTERNATIONAL'], pred):,.0f} pasajeros ({np.mean(np.abs(te['INTERNATIONAL'] - pred) / te['INTERNATIONAL']) * 100:.1f} %)")
print(f"MAE ingenuo estacional = {mae(te['INTERNATIONAL'], ingenuo(tr['INTERNATIONAL'].values, 12, 12)):,.0f}")
""")
md("""
✍️ **Ejercicio A7.** Compara con una partición **al azar** (12 meses elegidos al azar como prueba, el resto como entrenamiento). ¿Por qué ese MAE parece mejor y qué tiene de engañoso?
""")
code("""
# tu código aquí
""")
solution("""
rng3 = np.random.default_rng(7)
idx = rng3.choice(len(df), 12, replace=False)
mask = np.ones(len(df), bool); mask[idx] = False
m_az = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df[mask]).fit()
print(f"MAE con partición al azar = {mae(df.loc[~mask, 'INTERNATIONAL'], m_az.predict(df[~mask])):,.0f}")
""", "El MAE al azar (≈ 397 000) sale algo **menor** que el temporal (≈ 428 000): el modelo ya «vio» meses vecinos de cada mes de prueba (interpolación), no el futuro. Con una **serie** lo que importa es extrapolar, y el error honesto es el temporal; con modelos más flexibles la diferencia es mucho mayor.")

# ================================================================== PARTE B
md("""
## Parte B · Clase 26: AR, MA, ARMA, estimación, AIC, diagnóstico y pronóstico

### B1 · AR, MA y ARMA: ACF y PACF (20 min)
Reproduce los gráficos de la clase: AR(1) con φ = 0.8 y −0.75, AR(2) con (0.5, 0.3), MA(1) con θ = −0.9 (en R: `ma = +0.9`), MA(2) y ARMA(1,1) con φ = 0.8 y θ = 0.5.

⚠️ **Convención de signos**: las diapositivas escriben Yₜ = εₜ − θεₜ₋₁; R (`arima.sim`, `arima`) y `statsmodels` usan **+ θεₜ₋₁**. `simular_arma(ar, ma, …)` sigue la de R: `ma = −θ_diapositiva`.
""")
code("""
from statsmodels.tsa.arima_process import ArmaProcess

def teorica(ar, ma, nl=20):
    ap = ArmaProcess(np.r_[1, -np.atleast_1d(ar)] if len(np.atleast_1d(ar)) else np.array([1.0]), np.r_[1, np.atleast_1d(ma)] if len(np.atleast_1d(ma)) else np.array([1.0]))
    return ap.acf(nl + 1)[1:nl + 1], ap.pacf(nl + 1)[1:nl + 1]

casos = {"AR(1) φ=0.8": ([0.8], []), "AR(1) φ=-0.75": ([-0.75], []), "AR(2) (0.5, 0.3)": ([0.5, 0.3], []), "MA(1) θ=-0.9": ([], [0.9]), "MA(2) θ=(-0.7,-0.2)": ([], [0.7, 0.2]), "ARMA(1,1)": ([0.8], [-0.5])}
fig, axs = plt.subplots(2, 6, figsize=(17, 5))
for j, (nm, (ar, ma)) in enumerate(casos.items()):
    y_ = simular_arma(ar, ma, 600, np.random.default_rng(2600 + j))
    ta, tp = teorica(ar, ma)
    for i, (m_, tt, nn) in enumerate(((acf_manual(y_, 20)[1:], ta, "ACF"), (pacf_yw(y_, 20)[1:], tp, "PACF"))):
        axs[i, j].vlines(range(1, 21), 0, m_, lw=3); axs[i, j].plot(range(1, 21), tt, "r.", ms=7); axs[i, j].axhspan(-1.96 / np.sqrt(600), 1.96 / np.sqrt(600), color="orange", alpha=0.25)
        axs[i, j].set_ylim(-1, 1); axs[i, j].set_title(f"{nn} {nm}", fontsize=8)
plt.tight_layout(); plt.show()
print("ρ(1) del MA(1) con θ = -0.9:", round(-(-0.9) / (1 + 0.81), 4), "| teórica:", teorica([], [0.9])[0][:2].round(4))
""")
md("""
Lectura (puntos rojos = teoría): **AR**: ACF decae y PACF se corta en p; **MA**: ACF se corta en q y PACF decae; **ARMA**: ambas decaen. ρ(1) del MA(1) con θ = −0.9 vale [[ma_teo.rho1_theta09|3]] (la ACF teórica a distancia ≥ 2 es 0).

✍️ **Ejercicio B1.** Simula un AR(2) con φ = (0.4, 0.2) y calcula a mano ρ(1) = φ₁/(1 − φ₂) y ρ(2) = φ₂ + φ₁²/(1 − φ₂); compáralos con `ArmaProcess` y con la ACF muestral (n = 2 000).
""")
code("""
# tu código aquí
""")
solution("""
p1, p2 = 0.4, 0.2
print("a mano: rho1 =", round(p1 / (1 - p2), 4), "rho2 =", round(p2 + p1 ** 2 / (1 - p2), 4))
print("teórica:", teorica([p1, p2], [], 2)[0].round(4))
print("muestral:", acf_manual(simular_arma([p1, p2], [], 2000, np.random.default_rng(5)), 2)[1:].round(4))
""", "ρ(1) = 0.5 y ρ(2) = 0.4 (los tres coinciden, la muestral salvo el error de simulación).")

md("""
### B2 · Estacionariedad e invertibilidad (10 min)
Un AR es estacionario si las **raíces** del polinomio φ(z) = 1 − φ₁z − … − φₚzᵖ están **fuera** del círculo unitario. Para AR(1): |φ| < 1; para AR(2): φ₂ + φ₁ < 1, φ₂ − φ₁ < 1 y |φ₂| < 1. Un MA es siempre estacionario; se pide que las raíces de θ(z) estén fuera del círculo (invertibilidad).
""")
code("""
def es_estacionario(phis):
    raices = np.roots(np.r_[-np.array(phis[::-1]), 1.0])      # raíces de 1 - phi1 z - ... - phip z^p
    return bool(np.all(np.abs(raices) > 1)), np.abs(raices).round(3)

for phis in ([0.5], [1.0], [1.2], [0.5, 0.3], [1.2, 0.0], [0.4, 0.7]):
    print(phis, es_estacionario(phis))
rng = np.random.default_rng(2611)
for phi in (0.5, 0.95, 1.0):
    e = rng.normal(size=250); x = np.zeros(250)
    for i in range(1, 250):
        x[i] = phi * x[i - 1] + e[i]
    print(f"φ = {phi}: sd de la serie = {x.std():.2f} | ACF(1) = {acf_manual(x, 1)[1]:.2f}")
""")
md("""
✍️ **Ejercicio B2.** ¿Es estacionario el AR(2) con φ = (0.9, −0.2)? Compruébalo con las tres desigualdades y con las raíces. ¿Y el MA(1) con θ = −1.5 (en la notación de las diapositivas), es invertible?
""")
code("""
# tu código aquí
""")
solution("""
p1, p2 = 0.9, -0.2
print("desigualdades:", p2 + p1 < 1, p2 - p1 < 1, abs(p2) < 1, "| raíces:", es_estacionario([p1, p2]))
th = -1.5; ma_R = -th                              # convención de R: ma = -theta
print("MA(1) θ=-1.5: raíz de 1 + ma z =", round(-1 / ma_R, 3), "-> invertible:", abs(1 / ma_R) > 1)
""", "El AR(2) es estacionario (0.7 < 1, −1.1 < 1, 0.2 < 1; raíces fuera del círculo). El MA(1) con |θ| = 1.5 > 1 **no es invertible** (la raíz vale 0.667 < 1); siempre es estacionario pero no se puede pasar a un AR para pronosticar.")

md("""
### B3 · Anillos de árbol: la serie (10 min)
`tree.xlsx`: ancho anual del anillo de un árbol de Idaho (dendrocronología), [[arbol.n]] años desde 1492. Estas series reconstruyen el clima del pasado. Un anillo ancho = año lluvioso; delgado = seco.
""")
code("""
tr_y = tree["Idaho"].values
print(f"media {tr_y.mean():.4f} | mediana {np.median(tr_y):.3f} | sd {tr_y.std(ddof=1):.4f} | varianza {tr_y.var(ddof=1):.5f} | RIC {np.subtract(*np.percentile(tr_y, [75, 25])):.3f}")
print(pd.Series(tr_y).describe().round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
axs[0].plot(tree["Year"], tr_y, lw=0.8); axs[0].axhline(tr_y.mean(), color="r"); axs[0].set_title("ancho del anillo (mm)")
b = 1.96 / np.sqrt(len(tr_y))
for ax, v, nm in zip(axs[1:], (acf_manual(tr_y, 15), pacf_yw(tr_y, 15)), ("ACF", "PACF")):
    ax.vlines(range(1, 16), 0, v[1:], lw=3); ax.axhspan(-b, b, color="orange", alpha=0.25); ax.set_title(nm); ax.set_ylim(-0.2, 0.8)
plt.tight_layout(); plt.show()
print("ACF 1-5:", acf_manual(tr_y, 5)[1:].round(3), "| PACF 1-5:", pacf_yw(tr_y, 5)[1:].round(3), "| banda:", round(b, 3))
""")
md("""
Coincide con las diapositivas 40–42: media [[arbol.media|3]], mediana [[arbol.mediana|3]], sd [[arbol.sd|4]], RIC [[arbol.iqr|3]]. La serie oscila alrededor de una constante sin tendencia y con varianza estable: **candidata a un ARMA**. La ACF decae (ρ̂₁ = [[arbol.acf.0|3]]) y la PACF sobresale en los rezagos 1 y 2 (banda ±[[arbol.banda|3]]). Los datos tienen **autocorrelación débil**: ρ̂₁ ≈ 0.27.

✍️ **Ejercicio B3.** ¿Qué modelos candidatos propondrías mirando solo la ACF y la PACF? Justifica y lista al menos tres.
""")
code("""
# tu código aquí (puedes responder en un comentario)
""")
solution("""
# ACF: decaimiento; PACF: significativa en 1 y 2 (y 3 apenas fuera de la banda en la ACF).
candidatos = ["AR(2)", "ARMA(1,1)", "AR(1)", "MA(2)", "MA(3)"]
print(candidatos)
""", "Se proponen AR(2) (PACF con 2 valores), ARMA(1,1) (ambas decaen), AR(1), MA(2) y MA(3) (3 valores en la ACF): justamente los cinco modelos de la clase. ACF y PACF **sugieren** candidatos; no deciden.")

md("""
### B4 · Cinco modelos: CSS frente a máxima verosimilitud (25 min)
El profesor ajustó MA(3), MA(2), AR(1), AR(2) y ARMA(1,1) con `tseries::arma` (**mínimos cuadrados condicionales**, CSS) y mostró dos tablas de AIC: las de `arma` (diapositivas 44–48: 96.37, 98.21, 100.22, 94.64, 97.41) y las de `arima` (**máxima verosimilitud**, diapositiva 49: 102.77, 104.77, 105.98, 100.48, 100.11). `arma_css` replica el primero; `statsmodels.ARIMA` el segundo.
""")
code("""
filas = []
for nm, (p, q) in {"MA(3)": (0, 3), "MA(2)": (0, 2), "AR(1)": (1, 0), "AR(2)": (2, 0), "ARMA(1,1)": (1, 1)}.items():
    c = arma_css(tr_y, p, q)
    m = ARIMA(tr_y, order=(p, 0, q), trend="c").fit()
    filas.append({"modelo": nm, "AIC (CSS, tseries)": c["aic"], "AIC (MLE, arima)": m.aic, "BIC (MLE)": m.bic, "σ² (CSS)": c["sigma2"],
                  "φ (CSS)": ", ".join(f"{v:.4f}" for v in c["phi"]), "θ (CSS)": ", ".join(f"{v:.4f}" for v in c["theta"]), "intercepto tseries": c["intercepto"]})
tabla = pd.DataFrame(filas).set_index("modelo")
display(tabla.round(4))
print("menor AIC por CSS:", tabla["AIC (CSS, tseries)"].idxmin(), "| menor AIC por MLE:", tabla["AIC (MLE, arima)"].idxmin(), "| menor BIC:", tabla["BIC (MLE)"].idxmin())
""")
md("""
**Los dos métodos reproducen las diapositivas**: CSS da MA(3) con ma = [[M03.theta.0|5]], [[M03.theta.1|5]], [[M03.theta.2|5]] y AIC [[M03.css_aic|2]]; ARMA(1,1) con φ = [[M11.phi.0|4]], θ = [[M11.theta.0|4]] (los de `arma`). Con máxima verosimilitud el ARMA(1,1) es φ = [[M11.mle_params.0|4]], ma = [[M11.mle_params.1|4]] y el AIC [[M11.mle_aic|2]]. **Cuidado**:
- El menor AIC es **AR(2)** con CSS y **ARMA(1,1)** con MLE: los AIC de métodos distintos no se comparan (el profesor lo notó en clase: «el AIC está calculado diferente»).
- El MA(3) tiene un coeficiente borderline (p = 0.058): se descarta aunque su AIC sea bajo.
- En las diapositivas 50, el «modelo ajustado» escribe `Yₜ = 1.0 − 0.652Yₜ₋₁ + εₜ + 0.415εₜ₋₁`: tiene los **signos al revés** y 1.0 es la media, no el intercepto. Lo correcto: Yₜ − 1 = 0.652(Yₜ₋₁ − 1) + εₜ − 0.415εₜ₋₁ (en la notación de la diapositiva 25, θ = +0.415).

✍️ **Ejercicio B4.** Calcula los estadísticos t de cada coeficiente del ARMA(1,1) por CSS (`c["phi"] / c["se"][1]`, …) y decide si ambos son significativos. Luego compara el BIC de los cinco modelos: ¿cambia el ganador?
""")
code("""
# tu código aquí
""")
solution("""
c11 = arma_css(tr_y, 1, 1)
print("φ:", round(float(c11["phi"][0]), 4), "t =", round(float(c11["phi"][0] / c11["se"][1]), 2), "| θ:", round(float(c11["theta"][0]), 4), "t =", round(float(c11["theta"][0] / c11["se"][2]), 2))
print(tabla["BIC (MLE)"].round(2).to_string())
""", "t = [[M11.t.0|2]] y [[M11.t.1|2]] (diapositiva: 4.88 y −2.42): ambos significativos al 5 %. Por BIC el ganador es ARMA(1,1) ([[M11.mle_bic|2]]), igual que por AIC-MLE.")

md("""
### B5 · Diagnóstico: Ljung-Box y normalidad (20 min)
Un modelo se acepta si sus **residuos son ruido blanco**. El test de Ljung-Box tiene H₀: ρ₁ = … = ρₘ = 0 y Q(m) = n(n + 2) Σ r̂ᵣ²/(n − r) ~ χ²(**m − p − q**) bajo H₀ (la diapositiva omite los grados de libertad). Para la normalidad, el profesor usó Kolmogorov-Smirnov con μ y σ **estimados**.
""")
code("""
m11 = ARIMA(tr_y, order=(1, 1 - 1, 1), trend="c").fit()
e = np.asarray(m11.resid)
Q, p = ljung_box(e, 10, ddof=2)
ref = acorr_ljungbox(e, lags=[10], model_df=2)
print(f"ARMA(1,1): Q(10) = {Q:.3f}, p = {p:.4f} | statsmodels: {ref['lb_stat'].iloc[0]:.3f}, {ref['lb_pvalue'].iloc[0]:.4f}")
print("sin restar parámetros (gl = m), p =", round(ljung_box(e, 10, ddof=0)[1], 4))
m10 = ARIMA(tr_y, order=(1, 0, 0), trend="c").fit()
print("AR(1): p a m = 10:", round(ljung_box(np.asarray(m10.resid), 10, ddof=1)[1], 4), "| ARMA(1,1) a m = 20:", round(ljung_box(e, 20, 2)[1], 4))
z = e / np.sqrt(m11.params[-1])
ks = stats.kstest(z, stats.norm(z.mean(), z.std(ddof=1)).cdf)
print(f"KS con μ,σ estimados: D = {ks.statistic:.4f}, p = {ks.pvalue:.3f} | Lilliefors: p = {lilliefors(z, dist='norm')[1]:.3f} | Shapiro-Wilk: p = {stats.shapiro(z).pvalue:.3f}")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
axs[0].plot(z, lw=0.7); axs[0].axhspan(-2, 2, color="orange", alpha=0.2); axs[0].set_title("residuos estandarizados")
axs[1].vlines(range(1, 21), 0, acf_manual(e, 20)[1:], lw=3); axs[1].axhspan(-1.96 / np.sqrt(len(e)), 1.96 / np.sqrt(len(e)), color="orange", alpha=0.2); axs[1].set_title("ACF de los residuos")
stats.probplot(z, dist="norm", plot=axs[2]); axs[2].set_title("QQ-plot")
plt.tight_layout(); plt.show()
""")
md("""
Resultados: ARMA(1,1) con m = 10 → Q = [[diag.q10_arma|2]], p = **[[diag.p10_arma|3]]** (si no se restan los 2 parámetros, p = [[diag.p10_arma_sin_gl|3]]: el test sería demasiado permisivo); con m = 20, p = [[diag.p20_arma|3]]. El AR(1) tampoco se rechaza a m = 10 (p = [[diag.p10_ar1|3]]): con un ρ₁ de 0.27 el Ljung-Box **no distingue** bien entre candidatos; por eso se usan además el AIC y la parsimonia.

Normalidad: el K-S con μ y σ estimados da p = [[diag.ks_p|2]], pero ese p está **inflado** (el test exige parámetros conocidos); el test correcto, **Lilliefors**, da p = [[diag.lil_p|3]], y Shapiro-Wilk p = [[diag.sw_p|3]] (colas algo más pesadas que la normal). Para los intervalos de predicción no es grave con n = 493, pero hay que decirlo.

✍️ **Ejercicio B5.** Calcula el valor p de Ljung-Box para m = 1, …, 24 del ARMA(1,1) y cuenta cuántos son menores que 0.05. ¿Qué te dice? (Usa `ddof = 2`, y solo m > 2.)
""")
code("""
# tu código aquí
""")
solution("""
ps = np.array([ljung_box(e, m_, ddof=2)[1] for m_ in range(3, 25)])
print("p mínimo:", ps.min().round(3), "| m con p < 0.05:", int((ps < 0.05).sum()), "de", len(ps))
""", "Ninguno es menor que 0.05: el modelo no deja autocorrelación sin explicar. (Al probar muchos m, un p < 0.05 aislado sería esperable por azar.)")

md("""
### B6 · Pronóstico, intervalos y validación a ciegas (20 min)
El mejor predictor es la esperanza condicional. Para el ARMA(1,1) de los anillos: pronóstico a 10 años con intervalos del 80 y 95 % (la diapositiva 60). Después, **validación temporal**: entrenar con los primeros 453 años y pronosticar los últimos 40 sin mirarlos.
""")
code("""
fc = m11.get_forecast(10)
ic95, ic80 = np.asarray(fc.conf_int(alpha=0.05)), np.asarray(fc.conf_int(alpha=0.20))
tab = pd.DataFrame({"pronóstico": fc.predicted_mean, "bajo 95": ic95[:, 0], "bajo 80": ic80[:, 0], "alto 80": ic80[:, 1], "alto 95": ic95[:, 1]}, index=np.arange(tree["Year"].iloc[-1] + 1, tree["Year"].iloc[-1] + 11))
display(tab.round(4).head(4))
print("ancho del IC 95 %: h=1:", round(ic95[0, 1] - ic95[0, 0], 3), "| h=10:", round(ic95[-1, 1] - ic95[-1, 0], 3), "| 2·1.96·sd serie:", round(2 * 1.96 * tr_y.std(ddof=1), 3), "| media μ:", round(float(m11.params[0]), 4))
h = 40
tr_, te_ = tr_y[:-h], tr_y[-h:]
res = {}
for nm, (p, q) in {"AR(1)": (1, 0), "AR(2)": (2, 0), "MA(3)": (0, 3), "ARMA(1,1)": (1, 1)}.items():
    res[nm] = mae(te_, ARIMA(tr_, order=(p, 0, q), trend="c").fit().forecast(h))
res["media del entrenamiento"] = mae(te_, np.full(h, tr_.mean()))
res["ingenuo (último valor)"] = mae(te_, np.full(h, tr_[-1]))
print(pd.Series(res).round(4).to_string())
""")
md("""
La primera predicción (año 1985) es [[pron.h1|4]] con IC 95 % [[[pron.lo95_h1|4]]; [[pron.hi95_h1|4]]] — los mismos de la diapositiva 60 (1.1328; 0.6122 y 1.6534). A 10 años el pronóstico ya es [[pron.h10|3]] ≈ la media y el intervalo mide [[pron.ancho95_h10|2]], casi 2·1.96 veces la desviación de la serie ([[pron.z95_sd_serie|2]]): **un ARMA a largo plazo solo dice «la media»** (memoria corta, el profesor lo explicó con las 50 predicciones).

En la validación a ciegas de 40 años, **el ARMA(1,1) apenas le gana a la media del entrenamiento** (MAE [[MAE.ARMA(1,1)|4]]… ver la tabla): con ACF(1) = 0.27 hay poca dependencia que explotar. El **ingenuo** pierde claramente. Un buen modelo se mide así: contra referencias simples en datos no vistos.

⚠️ El profesor y el cuaderno llaman a estos intervalos «de confianza»; son intervalos de **predicción** (para un valor futuro, no para un parámetro). Y en la transcripción el «75 %» es el 80 % que dibuja `forecast`.

✍️ **Ejercicio B6.** Calcula a mano el pronóstico AR(1) a h = 1, 5 y 10 años del modelo AR(1) (μ = 1.0, φ = 0.2723, σ = 0.2678, último valor observado) con `ar1_pronostico` y verifica contra `get_forecast`.
""")
code("""
# tu código aquí
""")
solution("""
m_ar1 = ARIMA(tr_y, order=(1, 0, 0), trend="c").fit()
mu_, phi_, s_ = m_ar1.params[0], m_ar1.params[1], np.sqrt(m_ar1.params[2])
punt, sd = ar1_pronostico(tr_y[-1], mu_, phi_, s_, 10)
ref = m_ar1.get_forecast(10)
print("a mano h=1,5,10:", punt[[0, 4, 9]].round(4), "| sd:", sd[[0, 4, 9]].round(4))
print("statsmodels   :", np.asarray(ref.predicted_mean)[[0, 4, 9]].round(4), "| sd:", np.asarray(ref.se_mean)[[0, 4, 9]].round(4))
""", "Coinciden. La sd del error sube de σ = 0.268 hacia σ/√(1 − φ²) = 0.278 (la sd de la serie) y el pronóstico converge a μ = 1.")

# ================================================================== PARTE C
md("""
## Parte C · Clase 27: estacionalidad, diferenciación, Dickey-Fuller y cierre

### C1 · Operador de rezago y diferencias (15 min)
`B·Yₜ = Yₜ₋₁`, `Bʲ·Yₜ = Yₜ₋ⱼ`, `∇ = 1 − B`, `∇²Yₜ = Yₜ − 2Yₜ₋₁ + Yₜ₋₂` y `∇ₛ = 1 − Bˢ` (diferencia estacional). Los polinomios en B se multiplican como polinomios de números reales.
""")
code("""
y6 = pd.Series([20, 24, 30, 26, 32, 37, 43, 38, 44, 49, 56, 50], dtype=float)
d1 = y6.diff().dropna()
d4 = y6.diff(4).dropna()
dd = y6.diff().diff(4).dropna()
print("∇Y   :", d1.values, "| var =", round(d1.var(), 2))
print("∇4 Y :", d4.values, "| var =", round(d4.var(), 2))
print("∇∇4 Y:", dd.values, "(", len(dd), "datos: se pierden 1 + 4 )")
print("(1-B)(1-B^4) = ", np.polynomial.polynomial.polymul([1, -1], [1, 0, 0, 0, -1]).astype(int), "-> Y_t - Y_(t-1) - Y_(t-4) + Y_(t-5)")
e_ = np.random.default_rng(2703).normal(size=200000)
x = e_.copy()
for k in range(4):
    print(f"d = {k}: Var(∇^d ruido blanco) = {x.var():.2f}")
    x = np.diff(x)
""")
md("""
Para ruido blanco, **Var(∇ᵈε) = C(2d, d)·σ² = 1, 2, 6, 20 σ²**. En clase se dijo que con tres diferencias la varianza era «tres veces σ²»: no es así. Moraleja: **no se diferencia de más** (d = 1 casi siempre; d = 2 en casos extremos).

✍️ **Ejercicio C1.** Demuestra con simulación que diferenciar una **recta con ruido** (tendencia determinística) deja una serie con ACF(1) cercana a −0.5 (un MA(1) con θ = 1, no invertible) y varianza mayor que la del ruido original.
""")
code("""
# tu código aquí
""")
solution("""
rng4 = np.random.default_rng(3)
n4 = 5000
lin = 2 + 0.2 * np.arange(n4) + rng4.normal(0, 2, n4)
dlin = np.diff(lin)
print("ACF(1) de la diferencia:", round(acf_manual(dlin, 1)[1], 3), "| var(ruido) = 4 | var(diferencia) =", round(dlin.var(), 2), "(≈ 2σ² = 8)")
""", "ACF(1) ≈ −0.5 y varianza ≈ 8 = 2σ²: **diferenciar una tendencia determinística la sobre-diferencia**. Por eso las diapositivas dicen «es preferible ajustar la media por un polinomio».")

md("""
### C2 · AR estacional simulado (15 min)
Yₜ = 0.8·Yₜ₋₁₂ + εₜ (diapositivas 8–9): **ρ(k) = 0.8^(k/12) en los múltiplos de 12** y 0 en el resto. La PACF tiene un solo pico en el rezago 12.
""")
code("""
rng = np.random.default_rng(2701)
n = 800
e_ = rng.normal(size=n + 200); ys = np.zeros(n + 200)
for t_ in range(12, n + 200):
    ys[t_] = 0.8 * ys[t_ - 12] + e_[t_]
ys = ys[200:]
r = acf_manual(ys, 60); pc = pacf_yw(ys, 60)
print("ACF en 12, 24, 36, 48:", r[[12, 24, 36, 48]].round(3), "| teórica:", [round(0.8 ** k, 3) for k in (1, 2, 3, 4)])
print("PACF en 12:", round(pc[12], 3), "| ACF en rezago 1 y 6:", r[[1, 6]].round(3))
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(ys[:60]); axs[0].set_title("primeras 60 observaciones")
axs[1].vlines(range(1, 61), 0, r[1:], lw=2); axs[1].set_title("ACF")
axs[2].vlines(range(1, 61), 0, pc[1:], lw=2); axs[2].set_title("PACF")
plt.tight_layout(); plt.show()
""")
md("""
✍️ **Ejercicio C2.** Multiplica los polinomios (1 − 0.6B)(1 − 0.5B¹²) y escribe la ecuación en diferencias del SARMA(1,0)(1,0)₁₂ resultante. ¿Cuántos coeficientes distintos de 0 hay, comparado con un AR(13) general?
""")
code("""
# tu código aquí
""")
solution("""
poly = np.polynomial.polynomial.polymul([1, -0.6], np.r_[1, np.zeros(11), -0.5])
print({i: round(float(v), 3) for i, v in enumerate(poly) if v != 0})
""", "(1 − 0.6B)(1 − 0.5B¹²) = 1 − 0.6B − 0.5B¹² + 0.3B¹³, es decir, Yₜ = 0.6Yₜ₋₁ + 0.5Yₜ₋₁₂ − 0.3Yₜ₋₁₃ + εₜ: 3 coeficientes **generados por solo 2 parámetros** (un AR(13) general tendría 13).")

md("""
### C3 · SARMA con las muertes accidentales (25 min)
`USAccDeaths`: muertes mensuales por accidentes en EE. UU., 1973–1978 (n = [[usacc.n]]). Sin tendencia, con un máximo en julio. El profesor corre `auto.arima` y obtiene un **SARMA(1,0)(1,0)₁₂**: φ = 0.7579, Φ = 0.8501, media 9 214.71, AIC 1 074.89 (diapositiva 15). Lo reproducimos con `statsmodels`; el optimizador por defecto se detiene antes en la dirección «plana» de la media, así que se **pule** con Nelder-Mead.
""")
code("""
yu = ts(usa, "muertes")
mod_u = ARIMA(yu, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
r0 = mod_u.fit()
ru = mod_u.fit(start_params=r0.params, method_kwargs={"method": "nm", "maxiter": 3000, "xatol": 1e-8, "fatol": 1e-10})
print(ru.params.round(4).to_string())
print(f"AIC = {ru.aic:.2f} | BIC = {ru.bic:.2f} | log L = {ru.llf:.3f}   (profesor: φ 0.7579, Φ 0.8501, media 9214.71, AIC 1074.89, BIC 1084)")
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(yu); axs[0].set_title("muertes accidentales")
axs[1].vlines(range(1, 37), 0, acf_manual(yu.values, 36)[1:], lw=2); axs[1].set_title("ACF")
axs[2].vlines(range(1, 37), 0, pacf_yw(yu.values, 36)[1:], lw=2); axs[2].set_title("PACF")
plt.tight_layout(); plt.show()
""")
md("""
Resultado: φ = [[usacc.ar1|4]], Φ = [[usacc.sar1|4]], media [[usacc.media|1]], AIC **[[usacc.aic|2]]** (el del profesor es 1 074.89; la diferencia en las décimas viene del optimizador). La σ² de `statsmodels` es [[usacc.sigma2|0]] y la del profesor 132 733: usan otra convención (≈ 4 % de diferencia); lo importante, la log-verosimilitud y los coeficientes, coincide.

La ACF muestra picos en 12, 24 y 36 y la PACF en 1 y 12: sugieren AR(1) y AR estacional(1). En la práctica se prueba una **rejilla** y se elige el menor AIC (lo que hace `auto.arima`).
""")
code("""
rejilla = []
for p in range(3):
    for q in range(3):
        for P in range(2):
            for Q in range(2):
                m_ = ARIMA(yu, order=(p, 0, q), seasonal_order=(P, 0, Q, 12)).fit()
                rejilla.append(((p, q, P, Q), m_.aic, m_.bic))
rejilla.sort(key=lambda t: t[1])
display(pd.DataFrame([{"(p,q)(P,Q)": f"{t[0][:2]}x{t[0][2:]}", "AIC": t[1], "BIC": t[2]} for t in rejilla[:6]]).round(2))
e_u = np.asarray(ru.resid)
print("Ljung-Box (m=12, gl=m-2): p =", round(ljung_box(e_u, 12, 2)[1], 3), "| m=24: p =", round(ljung_box(e_u, 24, 2)[1], 3))
fc_u = ru.get_forecast(24); ci_u = np.asarray(fc_u.conf_int())
print("pronóstico enero 1979:", round(float(fc_u.predicted_mean.iloc[0])), "IC 95 %", ci_u[0].round(0), "| julio 1979:", round(float(fc_u.predicted_mean.iloc[6])))
""")
md("""
La rejilla de [[usacc.rejilla_n]] modelos da como mejor un SARMA(2,2)(1,1) (AIC [[usacc.mejor_rejilla.1|2]]) pero con el **AR(1)×AR(1) a ≈ 5 puntos**; modelos más grandes ganan por poco y son difíciles de interpretar: se prefiere el parsimonioso salvo que mejore la validación. Los residuos pasan Ljung-Box (p = [[usacc.p_lb12|2]] a m = 12).

✍️ **Ejercicio C3.** Valida con corte temporal: entrena con 1973–1977 (60 meses), pronostica 1978 y compara el MAE del SARMA(1,0)(1,0)₁₂ con el del ingenuo estacional y el de la media.
""")
code("""
# tu código aquí
""")
solution("""
tr_u, te_u = particion_temporal(yu, 12)
mm = ARIMA(tr_u, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
rr = mm.fit(); rr = mm.fit(start_params=rr.params, method_kwargs={"method": "nm", "maxiter": 3000})
print("MAE SARMA:", round(mae(te_u, rr.forecast(12)), 1), "| ingenuo estacional:", round(mae(te_u, ingenuo(tr_u.values, 12, 12)), 1), "| media:", round(mae(te_u, np.full(12, tr_u.mean())), 1))
""", "SARMA ≈ [[usacc.mae_sarma_12|0]], ingenuo estacional ≈ [[usacc.mae_ing_est_12|0]] y media ≈ [[usacc.mae_media_12|0]]: el SARMA captura la estacionalidad (la media no), pero con solo 6 años **no le gana** al ingenuo estacional. Con series cortas, la simplicidad compite.")

md("""
### C4 · Tendencia determinística y estocástica (15 min)
Determinística: la serie oscila alrededor de una recta fija. **Estocástica**: la tendencia cambia (caminata aleatoria). Las dos tienen ACF(1) ≈ 0.93, pero se tratan distinto.
""")
code("""
rng = np.random.default_rng(2702)
n = 144; tt = np.arange(n)
det = 2 + 0.2 * tt + rng.normal(0, 2, n)
est = np.cumsum(rng.normal(0.05, 1.8, n))
fig, axs = plt.subplots(2, 2, figsize=(10, 5))
for j, (s_, nm) in enumerate(((det, "determinística"), (est, "estocástica"))):
    axs[0, j].plot(s_); axs[0, j].set_title(nm); axs[1, j].plot(np.diff(s_), lw=0.8); axs[1, j].set_title("diferenciada")
plt.tight_layout(); plt.show()
print("ACF(1) serie  : det", round(acf_manual(det, 1)[1], 2), "| est", round(acf_manual(est, 1)[1], 2))
print("ACF(1) difer. : det", round(acf_manual(np.diff(det), 1)[1], 2), "| est", round(acf_manual(np.diff(est), 1)[1], 2))
print("φ̂ de la caminata (regresión de y_t sobre y_t-1):", round(float(np.sum(est[1:] * est[:-1]) / np.sum(est[:-1] ** 2)), 3))
""")
md("""
Tras diferenciar, la determinística muestra ACF(1) = [[tend.acf1_det_dif|2]] (el MA(1) con θ = 1 de C1) y la caminata ≈ [[tend.acf1_est_dif|2]] (ruido blanco). Una **caminata aleatoria** es el ARIMA(0,1,0): Yₜ = Yₜ₋₁ + εₜ, es decir, un AR(1) con φ = 1.

✍️ **Ejercicio C4.** En la serie determinística, ajusta una recta por OLS y mira los residuos: ¿su ACF(1) es cercana a 0? ¿Qué tratamiento recomienda la diapositiva 26 para este caso y por qué es mejor que diferenciar?
""")
code("""
# tu código aquí
""")
solution("""
X = sm.add_constant(tt)
rr = sm.OLS(det, X).fit()
print("ACF(1) de los residuos de la recta:", round(acf_manual(rr.resid, 1)[1], 3), "| varianza residual:", round(float(rr.resid.var()), 2), "| varianza de la diferencia:", round(float(np.diff(det).var()), 2))
""", "Los residuos son ≈ ruido blanco (ACF(1) ≈ 0) y su varianza (≈ 4) es la mitad que la de la serie diferenciada (≈ 8). Es preferible **ajustar la tendencia determinística y modelar los errores con ARMA** que diferenciar.")

md("""
### C5 · Dickey-Fuller a mano y con statsmodels (20 min)
El test parte de Δyₜ = c + γ·yₜ₋₁ + e, con H₀: γ = 0 (raíz unitaria) frente a H₁: γ < 0. El estadístico t **no sigue una t de Student**: bajo H₀ está corrido a la izquierda; con constante, su valor crítico al 5 % es ≈ −2.86.
""")
code("""
rng = np.random.default_rng(2706)
tdf = np.array([adf_regresion(np.cumsum(rng.normal(size=250)))[1] for _ in range(1500)])
print("percentil 5 % del estadístico bajo H0 (simulado, 1 500 caminatas):", round(float(np.percentile(tdf, 5)), 2), "| con una t/normal sería", -1.65)
indice = 100 * np.cumprod(1 + ipc["var_ipc"].values / 100)       # índice reconstruido acumulando la variación mensual (no es el índice oficial)
g, t_g = adf_regresion(indice)
print(f"índice de precios: γ̂ = {g:.5f}, t = {t_g:.3f} | statsmodels (sin rezagos): {adfuller(indice, regression='c', maxlag=0, autolag=None)[0]:.3f}")
lips = np.log(ipsa["ipsa"].values); ret = np.diff(lips)
for nm, s_ in (("índice de precios", indice), ("Δ índice", np.diff(indice)), ("log IPSA", lips), ("retorno log IPSA", ret)):
    a = adfuller(s_, regression="c", autolag="AIC")
    print(f"{nm:20s} estadístico = {a[0]:7.3f} | p = {a[1]:.4f} | rezagos = {a[2]} | crítico 5 % = {a[4]['5%']:.2f}")
""")
md("""
La simulación da un 5 % crítico de [[adf.crit5_sim|2]], **no** −1.65. Resultados: el índice de precios (p = [[RES.Índice de precios (reconstruido de ipc_var).p|3]]) y el log del IPSA (p = [[RES.log IPSA 1980-2000.p|3]]) **no rechazan** la raíz unitaria; sus diferencias sí. Cuidado con leerlo al revés: «p grande» significa **no poder descartar** la raíz unitaria (la diapositiva 35 lo dice como «existe evidencia de presencia de tendencia»).

La regresión DF **sin rezagos** sobre el índice da t = [[adf.t_indice|2]] (p ≈ 0.006, rechaza), mientras que el test aumentado con rezagos elegidos por AIC no: con mucha autocorrelación, el DF simple se equivoca; por eso existe el **aumentado**. (El archivo del IPC empalmado y el de Falabella de las diapositivas 35–46 no se entregaron; aquí se usan el IPC reconstruido y el IPSA.)

✍️ **Ejercicio C5.** Aplica el ADF a una AR(1) estacionaria con φ = 0.9 y n = 100 y a una caminata aleatoria de n = 100. ¿Cuántas veces de 500 rechaza H₀ cada una (potencia y tamaño)? ¿Qué te dice sobre la potencia del test con φ cerca de 1?
""")
code("""
# tu código aquí
""")
solution("""
rng5 = np.random.default_rng(11)
rech_ar = np.mean([adfuller(simular_arma([0.9], [], 100, rng5), regression="c", maxlag=0, autolag=None)[1] < 0.05 for _ in range(500)])
rech_rw = np.mean([adfuller(np.cumsum(rng5.normal(size=100)), regression="c", maxlag=0, autolag=None)[1] < 0.05 for _ in range(500)])
print("rechaza H0 con AR(1) φ=0.9 (potencia):", rech_ar, "| con caminata (error tipo I):", rech_rw)
""", "Con φ = 0.9 y n = 100 el test rechaza **solo una minoría** de las veces (potencia ≈ 0.37): un AR casi unitario se confunde con una raíz unitaria. La caminata rechaza ≈ 5 % (el nivel). Un p grande no prueba que haya raíz unitaria.")

md("""
### C6 · Retornos del IPSA y MA con parámetros fijados (15 min)
En finanzas se modelan los **retornos** log Pₜ − log Pₜ₋₁ ≈ (Pₜ − Pₜ₋₁)/Pₜ₋₁ (las diapositivas dividen por Pₜ; es Pₜ₋₁). El profesor aplicó esto a las acciones de Falabella (archivo no entregado); aquí se usa el IPSA mensual 1980–2000.
""")
code("""
n_r = len(ret)
ac = acf_manual(ret, 24)
print(f"n = {n_r} | media {ret.mean():.4f} | sd {ret.std(ddof=1):.4f} | ACF 1,2,3: {ac[1:4].round(3)} | banda ±{1.96/np.sqrt(n_r):.3f}")
print("Ljung-Box m=10:", f"p = {ljung_box(ret, 10)[1]:.2e}", "| m=20:", f"p = {ljung_box(ret, 20)[1]:.2e}")
m_ma3 = ARIMA(ret, order=(0, 0, 3), trend="c", enforce_invertibility=False)    # para poder fijar un solo ma
r_ma3 = m_ma3.fit()
print(r_ma3.summary().tables[1])
with m_ma3.fix_params({"ma.L1": 0.0}):        # como fixed = c(0, NA, NA) en R: se fija ma1 en 0 y se estima el resto
    r_fix = m_ma3.fit()
print(r_fix.summary().tables[1])
print("AIC ruido blanco con constante:", round(ARIMA(ret, order=(0, 0, 0), trend="c").fit().aic, 1), "| MA(3):", round(r_ma3.aic, 1), "| MA(3) con ma1 fijo en 0:", round(r_fix.aic, 1))
""")
md("""
Los retornos del IPSA **no son ruido blanco** (ACF(1) = [[retornos.acf1|2]], Ljung-Box p ≈ 5·10⁻⁶): el único coeficiente significativo es ma₁ = [[retornos.ma3.0|2]]; ma₂ y ma₃ no lo son. El profesor, con Falabella, hizo lo contrario (ma₁ no significativo y ma₂, ma₃ sí) y los dejó con `fixed = c(0, NA, NA)`. La idea es la misma: **quitar los parámetros no significativos** y volver a diagnosticar. (Ojo: la diapositiva 47 escribe «+0.086εₜ₋₂ + 0.086εₜ₋₃»; con la convención de R los coeficientes son **−0.0863**.)

✍️ **Ejercicio C6.** Ajusta MA(1) a los retornos del IPSA y compara AIC y Ljung-Box (m = 10, gl = m − 1) con el MA(3) y con el ruido blanco. ¿Qué modelo elegirías?
""")
code("""
# tu código aquí
""")
solution("""
r_ma1 = ARIMA(ret, order=(0, 0, 1), trend="c").fit()
print("MA(1): ma1 =", round(float(r_ma1.params[1]), 3), "| AIC", round(r_ma1.aic, 1), "| Ljung-Box m=10: p =", round(ljung_box(np.asarray(r_ma1.resid), 10, 1)[1], 3))
""", "MA(1) tiene AIC menor que el MA(3) y su Ljung-Box no rechaza: es el modelo parsimonioso para los retornos del IPSA (a diferencia del MA(3) de Falabella).")

md("""
### C7 · SARIMA de pasajeros con origen móvil (25 min)
Los pasajeros (2002-10 a 2019-09, n = [[valid.n]]) tienen tendencia, estacionalidad anual y una varianza que crece con el nivel. Se compara el **ingenuo estacional**, la regresión con dummies y un **SARIMA(0,1,1)(0,1,1)₁₂** (el «modelo de las aerolíneas»: ∇∇₁₂ Yₜ = (1 + θB)(1 + ΘB¹²)εₜ). Siempre con **partición temporal**.
""")
code("""
yp = ts(pas_todo, "INTERNATIONAL") / 1e6
h = 24
trp, tep = particion_temporal(yp, h)
dfp = pd.DataFrame({"y": yp.values, "t": np.arange(len(yp)), "mes": yp.index.month}, index=yp.index)
ols_ = sm.OLS.from_formula("y ~ t + C(mes)", dfp.iloc[:-h]).fit()
sar = ARIMA(trp, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
fc = sar.get_forecast(h)
modelos = {"ingenuo": ingenuo(trp.values, h, 1), "ingenuo estacional": ingenuo(trp.values, h, 12), "tendencia + dummies": ols_.predict(dfp.iloc[-h:]).values, "SARIMA(0,1,1)(0,1,1)12": fc.predicted_mean.values}
print(pd.Series({k: mae(tep.values, v) for k, v in modelos.items()}).round(3).to_string())
print("parámetros del SARIMA:", sar.params.round(3).to_dict(), "| AIC", round(sar.aic, 1))
ci_p = np.asarray(fc.conf_int(alpha=0.05))
print("cobertura del IC 95 % en los 24 meses de prueba:", round(float(np.mean((tep.values >= ci_p[:, 0]) & (tep.values <= ci_p[:, 1]))), 2))
# origen móvil: 5 orígenes anuales, 12 meses de prueba
orig = list(range(len(yp) - 60, len(yp) - 11, 12))
fila = {"ingenuo estacional": [], "SARIMA": [], "dummies": []}
for o in orig:
    a_, b_ = yp.iloc[:o], yp.iloc[o:o + 12]
    fila["ingenuo estacional"].append(mae(b_.values, ingenuo(a_.values, 12, 12)))
    fila["SARIMA"].append(mae(b_.values, ARIMA(a_, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit().forecast(12).values))
    fila["dummies"].append(mae(b_.values, sm.OLS.from_formula("y ~ t + C(mes)", dfp.iloc[:o]).fit().predict(dfp.iloc[o:o + 12]).values))
display(pd.DataFrame(fila, index=[str(yp.index[o].date()) for o in orig]).round(3).assign())
""")
md("""
Con los últimos 24 meses de prueba: ingenuo [[VM.ingenuo|2]], ingenuo estacional [[VM.ingenuo estacional|2]], dummies [[VM.tendencia + dummies (OLS)|2]] y **SARIMA [[VM.SARIMA(0,1,1)(0,1,1)₁₂|2]]** millones; con origen móvil de 5 cortes el SARIMA gana **siempre** (MAE medio [[valid.mae_prom_origen.SARIMA(0,1,1)(0,1,1)₁₂|2]] frente a [[valid.mae_prom_origen.ingenuo estacional|2]] del ingenuo estacional). La regresión con dummies **empeora** con el tiempo porque la estacionalidad crece con el nivel (modelo multiplicativo): una tendencia lineal con efectos fijos no basta.

✍️ **Ejercicio C7.** Repite el SARIMA sobre `np.log(yp)` y devuelve el pronóstico a la escala original con `np.exp` (sin corrección de sesgo). ¿Mejora el MAE de los 24 meses de prueba? Relaciona con el α de Box-Cox de A6.
""")
code("""
# tu código aquí
""")
solution("""
sar_log = ARIMA(np.log(trp), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
pred_log = np.exp(sar_log.forecast(h))
print("MAE SARIMA en logaritmos:", round(mae(tep.values, pred_log.values), 3), "| en niveles:", round(mae(tep.values, modelos["SARIMA(0,1,1)(0,1,1)12"]), 3))
""", "**No mejora** en esta ventana: MAE 0.376 con logaritmo frente a 0.342 en niveles. El logaritmo (coherente con α̂ ≈ −0.26 de A6) es una opción a **validar**, no una regla; además `np.exp(pronóstico en log)` estima la mediana, no la media (pequeño sesgo).")

md("""
### C8 · Cierre del módulo (Clase 28): CityBike (20 min)
El cuaderno `DS26_28_Cierre_de_Módulo.ipynb` aplica ARIMA, SARIMA y SARIMAX a la serie diaria de viajes de **CityBike** ([[citybike.ini]] a [[citybike.fin]], [[citybike.n]] días). Usa los últimos 30 días como prueba (**sin barajar**). La serie viene de una hoja de Google; aquí se lee la copia local `citybike_serie_diaria.csv`.
""")
code("""
cb = pd.read_csv(carpeta / "citybike_serie_diaria.csv", parse_dates=["fecha"]).sort_values("fecha").set_index("fecha")
serie = cb["total_viajes"]
h = 30
train, test = particion_temporal(serie, h)
a1 = ARIMA(train, order=(2, 0, 2)).fit()
a2 = ARIMA(train, order=(7, 1, 7)).fit()
s1 = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
X = cb[["temperatura_promedio_c"]]
sx = SARIMAX(train, exog=X.iloc[:-h], order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
pred = {"ARIMA(2,0,2)": a1.forecast(h), "ARIMA(7,1,7)": a2.forecast(h), "SARIMA(1,0,1)(1,0,1,7)": s1.forecast(h), "SARIMAX + temperatura": sx.forecast(h, exog=X.iloc[-h:])}
tabla_cb = pd.DataFrame({"MAE": {k: mae(test, v) for k, v in pred.items()}, "AIC": {"ARIMA(2,0,2)": a1.aic, "ARIMA(7,1,7)": a2.aic, "SARIMA(1,0,1)(1,0,1,7)": s1.aic, "SARIMAX + temperatura": sx.aic},
                         "profesor (MAE)": {"ARIMA(2,0,2)": 25.47, "ARIMA(7,1,7)": 25.15, "SARIMA(1,0,1)(1,0,1,7)": 21.85, "SARIMAX + temperatura": 15.08}})
display(tabla_cb.round(2))
print("referencias: ingenuo estacional (s=7)", round(mae(test, ingenuo(train.values, h, 7)), 2), "| media de 7 días", round(mae(test, np.full(h, train.iloc[-7:].mean())), 2))
print("coeficientes AR del SARIMA:", s1.params[["ar.L1", "ar.S.L7"]].round(4).to_dict(), "| media de la serie:", round(float(train.mean()), 1))
print("correlación temperatura-viajes:", round(float(np.corrcoef(cb["temperatura_promedio_c"], serie)[0, 1]), 2))
""")
md("""
Se reproducen los números del cuaderno (MAE [[CBM.ARIMA(2,0,2)|2]], [[CBM.SARIMA(1,0,1)(1,0,1)₇|2]] y [[CBM.SARIMAX + temperatura|2]]; el ARIMA(7,1,7) no converge del todo y varía entre versiones: [[CBM.ARIMA(7,1,7)|2]] aquí frente a 25.15). Lo que **hay que saber**:
- Los coeficientes AR del SARIMA salen [[citybike.sarima_ar1|3]] y [[citybike.sarima_ars|3]]: **casi raíces unitarias**. La serie tiene media ≈ [[citybike.media_train|0]] y el modelo no tiene intercepto ni diferencia, así que usa un AR casi unitario para «sostener» el nivel.
- La temperatura baja de [[citybike.temp_ini|0]] a [[citybike.temp_fin|0]] °C mientras los viajes caen de ≈ [[citybike.v_ini|0]] a ≈ [[citybike.v_fin|0]]: la correlación ([[citybike.temp_corr|2]]) viene en buena parte de la **tendencia compartida**. Y el SARIMAX usa la temperatura **real** de la prueba (conocida): es un pronóstico condicionado.
- Referencias simples: el ingenuo estacional (s = 7) da [[CBM.ingenuo estacional (s = 7)|1]] y la media de 7 días [[CBM.media de los últimos 7 días|1]]; el ARIMA(2,0,2) (25.5) **pierde con la media de 7 días**.

✍️ **Ejercicio C8.** Mejora el SARIMA: (a) con intercepto (`trend="c"`) y (b) con diferenciación estacional, SARIMA(1,0,1)(0,1,1,7). Compara el MAE y mira si los coeficientes AR dejan de ser casi unitarios.
""")
code("""
# tu código aquí
""")
solution("""
s_a = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7), trend="c").fit(disp=False, maxiter=300)
s_b = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
for nm, m_ in (("con intercepto", s_a), ("con D = 1", s_b)):
    print(nm, "| MAE:", round(mae(test, m_.forecast(h)), 2), "| AIC:", round(m_.aic, 1), "| AR:", m_.params[[c for c in m_.params.index if c.startswith("ar")]].round(3).to_dict())
""", "Con intercepto: MAE 23.4 y AIC 4 929; con D = 1: MAE 21.8 y AIC 4 856 (mejor AIC por ≈ 70 puntos). Los AR **siguen casi unitarios** (0.995 y 0.997): la serie tiene una tendencia fuerte (bajan los viajes de ≈ 400 a ≈ 250 diarios), que ni el intercepto ni la diferencia estacional absorben; haría falta diferenciar (d = 1) o un regresor de tendencia. Y el MAE es una sola ventana de 30 días: conviene un origen móvil.")

# ================================================================== PARTE D
md("""
## Parte D · Puentes y alertas

### D1 · Puentes con C17–21 y C43 (15 min)
- **C17–21**: en la regresión de A5 los residuos son autocorrelacionados; los errores estándar clásicos engañan. Las salidas: errores robustos (HAC), modelar los errores con ARMA (regresión con errores ARMA = SARIMAX con regresores) o incluir rezagos.
- **C43**: el intervalo de predicción de un ARMA es paramétrico (supone normalidad). Una alternativa es el **bootstrap de residuos**: simular trayectorias futuras remuestreando los residuos.
""")
code("""
phi_, th_, mu_ = m11.params[1], m11.params[2], m11.params[0]
res_ = np.asarray(m11.resid)
rng = np.random.default_rng(2808)
H, B = 10, 2000
sim = np.zeros((B, H))
for b in range(B):
    yy, ee = tr_y[-1], res_[-1]
    for k in range(H):
        en = rng.choice(res_)
        yy = mu_ + phi_ * (yy - mu_) + en + th_ * ee
        ee = en
        sim[b, k] = yy
q = np.percentile(sim, [2.5, 97.5], axis=0)
ic_m = np.asarray(m11.get_forecast(H).conf_int(alpha=0.05))
print("h = 10 | IC del modelo:", ic_m[-1].round(3), "ancho", round(ic_m[-1, 1] - ic_m[-1, 0], 3), "| IC bootstrap:", q[:, -1].round(3), "ancho", round(q[1, -1] - q[0, -1], 3))
""")
md("""
Los dos intervalos son casi iguales ([[puentes.ancho_modelo_h10|3]] y [[puentes.ancho_boot_h10|3]]): los residuos son bastante normales y el bootstrap no aporta nada que el modelo no diga. Si los residuos fueran muy asimétricos, el bootstrap sí cambiaría el resultado. Pero **ninguno** incluye la incertidumbre del modelo elegido ni de cambios de régimen (pandemia).

### D2 · Alertas verificadas y tabla R ↔ Python (10 min)
Lo que se dijo o aparece en las diapositivas y se **verificó con código** en este notebook:

| Dónde | Qué dice | Qué se verificó |
|---|---|---|
| Cuaderno R, IPC | lee `ipc_var.txt` por filas | mezcla los meses; ACF(1) [[ejemplos.ipc_acf1_codigo_R|2]] frente a [[ejemplos.ipc_acf1_ok|2]] (A1) |
| Diapositivas 9, 27 | PM2.5 «1990–2000»; pasajeros «2010–2019» | datos: PM2.5 2000–2009; pasajeros desde 2002-10 (el profesor corrigió en clase) |
| Diapositiva 31 | «regresión lineal simple» con 12 regresores | es múltiple; los residuos tienen DW [[regresion.dw|2]] (A5) |
| Diapositiva 50 | Yₜ = 1.0 − 0.652Yₜ₋₁ + εₜ + 0.415εₜ₋₁ | signos al revés; MLE: φ = +0.652, θ = +0.415 con εₜ − θεₜ₋₁ (B4) |
| Diapositivas 44–49 | AIC de `arma` y de `arima` | métodos distintos: ganadores AR(2) y ARMA(1,1); no se comparan (B4) |
| Diapositiva 53 | Ljung-Box sin grados de libertad | gl = m − p − q (B5) |
| Diapositivas 46 y 56 | K-S con μ, σ estimados | p inflado; usar Lilliefors ([[diag.ks_p|2]] frente a [[diag.lil_p|3]]) (B5) |
| Clase 26 y cierre | «intervalos de confianza» del pronóstico | son intervalos de predicción (B6, C3) |
| Clase 27 | «tres diferencias: 3σ²» | Var(∇ᵈε) = 1, 2, 6, 20 σ² (C1) |
| Diapositivas 28–29 | H₁: φ < 1 y luego φ ≠ 1; el test de t | DF clásico unilateral; no usa t de Student (C5) |
| Diapositiva 35 | «existe evidencia de presencia de tendencia» | p grande = no se rechaza raíz unitaria (C5) |
| Diapositiva 39 | (Pₜ − Pₜ₋₁)/Pₜ | divide por Pₜ₋₁ (C6) |
| Diapositiva 47 | MA(3) con +0.086 | R/statsmodels: −0.0863 (C6) |
| Diapositivas 6, 10 y 11 | θ(B) con + y con −; Φ(B²) | convención inconsistente; debe ser Φ(Bˢ) (B1) |
| Diapositiva 26 | «tendencia determinística: ARIMA siempre sirve» | diferenciar una recta crea un MA(1) con θ = 1 (C1, C4) |
| Cierre (C28) | SARIMA(1,0,1)(1,0,1,7) | AR ≈ 1 (sin intercepto); SARIMAX con temperatura real (C8) |
| Clase 25 | «IPSA: 50 empresas» | no verificable con los materiales; según conocimiento externo hoy son 30 acciones |

**R ↔ Python**

| Tarea | R (profesor) | Python (este notebook) |
|---|---|---|
| Leer datos | `read_xlsx()`, `read.table()` | `pd.read_csv("Guias/datos/series_tiempo/…")` |
| Serie con frecuencia | `ts(x, frequency = 12, start = c(2010, 1))` | `pd.Series(x, index=pd.DatetimeIndex(…, freq="MS"))` |
| ACF / PACF | `acf()`, `pacf()`, `ggAcf()`, `ggPacf()` | `acf_manual`, `pacf_yw`, `sm.graphics.tsa.plot_acf` |
| Simular | `arima.sim(list(order = c(1,0,0), ar = 0.8), n = 200)` | `simular_arma([0.8], [], 200, rng)` |
| Descomposición | `decompose(x, type = "additive")` | `descomponer_aditiva` / `seasonal_decompose` |
| Regresión | `lm(y ~ Tiempo + Month)` | `sm.OLS.from_formula("y ~ t + C(mes)", df)` |
| Box-Cox | (`BoxCox`) | `stats.boxcox_normmax` |
| Ajuste ARMA | `tseries::arma()`, `forecast::Arima()`, `arima()` | `arma_css` (CSS) / `ARIMA(…).fit()` (MLE) |
| AIC | `AIC()` en el resumen | `modelo.aic`, `.bic` |
| Residuos y diagnóstico | `tsdiag()`, `ggtsdiag()`, `Box.test()` | `ljung_box`, `acorr_ljungbox`, QQ con `stats.probplot` |
| Normalidad | `ks.test()`, `fitdistrplus` | `stats.kstest`, `lilliefors`, `stats.shapiro` |
| Pronóstico | `forecast(modelo, h = 10, level = 95)`, `predict()` | `modelo.get_forecast(10).conf_int()` |
| Orden automático | `auto.arima()` | rejilla con `ARIMA(…).fit().aic` |
| Diferencias | `diff(x)`, `diff(x, lag = 12)` | `np.diff`, `s.diff(12)` |
| Dickey-Fuller | `adf.test()`, `urca::ur.df()`, `fUnitRoots::adfTest()` | `adfuller`, `adf_regresion` |
| Parámetros fijos | `Arima(…, fixed = c(0, NA, NA))` | `with modelo.fix_params({"ma.L1": 0.0}): …` |
| Estacional | `Arima(order, seasonal = list(order = c(1,0,0), period = 12))` | `ARIMA(order, seasonal_order=(1, 0, 0, 12))`, `SARIMAX` |
| Regresores | `xreg =` | `SARIMAX(…, exog=X)` |
""")

# ================================================================== CIERRE
md("""
## Ejercicio de cierre · IAC del INE (40 min, sin ayuda)
`iac_ine.csv`: el **Índice de Actividad del Comercio** mensual (enero 2015 – noviembre 2024, n = [[iac.n]]). Es un dato que viene en la carpeta pero que el profesor no usó en clase. Haz un análisis completo, como lo harías ante un jefe:

1. **Mira** la serie: gráfico, tendencia, estacionalidad, varianza. ¿Aditiva o multiplicativa?
2. **Separa** los últimos **12 meses** como prueba (partición temporal).
3. **Estacionariedad**: ADF del nivel, de la diferencia estacional y de la diferencia simple + estacional. ¿Qué diferenciación usas?
4. **Identifica**: ACF y PACF de la serie diferenciada; propone al menos 3 candidatos.
5. **Estima y compara** con AIC (rejilla pequeña con d = D = 1 y s = 12).
6. **Diagnostica** el mejor (ACF de residuos y Ljung-Box con gl = m − p − q).
7. **Pronostica** 12 meses con intervalo del 95 % y compara con el **ingenuo** y el **ingenuo estacional** (MAE y RMSE). ¿Gana?
8. Escribe una conclusión de **máximo tres líneas** para un gerente comercial.

> Pistas: usa `y.diff(12)`, `adfuller`, `ARIMA(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12))` y las funciones de la Preparación.
""")
code("""
# tu análisis comienza aquí
""")
solution("""
yi = ts(iac, "iac")
h = 12
tr_i, te_i = particion_temporal(yi, h)
for nm, s_ in (("nivel", tr_i), ("∇12", tr_i.diff(12).dropna()), ("∇∇12", tr_i.diff().diff(12).dropna())):
    print(f"ADF {nm:5s}: p = {adfuller(s_, regression='c', autolag='AIC')[1]:.3f}")
dd = tr_i.diff().diff(12).dropna()
print("ACF de ∇∇12:", acf_manual(dd.values, 13)[[1, 2, 12, 13]].round(3), "| PACF:", pacf_yw(dd.values, 13)[[1, 2, 12, 13]].round(3), "| banda", round(1.96 / np.sqrt(len(dd)), 3))
tabla = []
for p in range(2):
    for q in range(2):
        for P in range(2):
            for Q in range(2):
                tabla.append(((p, 1, q, P, 1, Q), ARIMA(tr_i, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit().aic))
tabla.sort(key=lambda t: t[1])
mejor = tabla[0][0]
print("mejores por AIC:", [(t[0], round(t[1], 1)) for t in tabla[:3]])
mi = ARIMA(tr_i, order=mejor[:3], seasonal_order=mejor[3:] + (12,)).fit()
ei = np.asarray(mi.resid)[13:]
print("Ljung-Box m=12 (gl = m - p - q - P - Q):", round(ljung_box(ei, 12, ddof=int(mejor[0] + mejor[2] + mejor[3] + mejor[5]))[1], 3))
fci = mi.get_forecast(h); ci_i = np.asarray(fci.conf_int(alpha=0.05))
res_i = {"SARIMA": fci.predicted_mean.values, "ingenuo estacional": ingenuo(tr_i.values, h, 12), "ingenuo": ingenuo(tr_i.values, h, 1)}
print(pd.DataFrame({k: {"MAE": mae(te_i.values, v), "RMSE": rmse(te_i.values, v)} for k, v in res_i.items()}).round(2).to_string())
print("cobertura del IC 95 %:", round(float(np.mean((te_i.values >= ci_i[:, 0]) & (te_i.values <= ci_i[:, 1]))), 2), "| ancho h=1 y h=12:", round(ci_i[0, 1] - ci_i[0, 0], 1), round(ci_i[-1, 1] - ci_i[-1, 0], 1))
""", "Referencia: el ADF del nivel no rechaza (p ≈ [[iac.adf_nivel_p|2]]); con ∇₁₂ p ≈ [[iac.adf_d12_p|3]] y con ∇∇₁₂ p ≈ [[iac.adf_dd_p|3]]. El mejor por AIC en la rejilla es SARIMA(0,1,0)(0,1,1)₁₂ (AIC ≈ [[iac.aic_mejor|1]]; luego (0,1,0)(1,1,1) y (1,1,0)(0,1,1)); Ljung-Box (m = 12) p ≈ [[iac.p_lb12|2]]. En la prueba de 12 meses: SARIMA MAE ≈ [[iac.mae_sarima|2]] (RMSE ≈ [[iac.rmse_sarima|2]]), **ingenuo estacional MAE ≈ [[iac.mae_ing_est|2]]** e ingenuo ≈ [[iac.mae_ing|2]]: **el SARIMA no le gana al ingenuo estacional en esta ventana**. El IC al 95 % cubre [[iac.cobertura|2]] de los meses y pasa de ancho [[iac.ancho_h1|1]] a [[iac.ancho_h12|1]]. **Conclusión sugerida**: el IAC tiene un patrón anual estable (crece ≈ [[iac.crec_anual|1]] % en el último año); con 107 meses de entrenamiento un modelo SARIMA sencillo es válido (residuos limpios) pero no mejora la referencia estacional en 12 meses, así que conviene validar con **origen móvil** antes de recomendar un modelo más complejo.")

# ---------------------------------------------------------------- autoevaluación
md("""
## Autoevaluación (responde sin mirar)

1. ¿Qué tres condiciones definen el ruido blanco? ¿Por qué es más débil que la independencia? ¿Qué descartan una tendencia y una varianza creciente?
2. ¿Cómo se calcula la ACF muestral y por qué se usan las bandas ±1.96/√n? ¿Cuántas barras fuera esperas en 20 rezagos de ruido blanco?
3. ¿Qué es una descomposición aditiva? ¿Qué se pierde al estimar la tendencia con la media móvil centrada? ¿Por qué los residuos de la regresión con dummies siguen autocorrelacionados?
4. ¿Cómo se distinguen AR(p), MA(q) y ARMA en la ACF y la PACF? ¿Qué condiciones aseguran estacionariedad e invertibilidad?
5. ¿Por qué no se pueden comparar los AIC de `arma` (CSS) y de `arima` (MLE)? ¿Qué otras cosas se miran además del AIC?
6. ¿Qué hipótesis prueba Ljung-Box y por qué hay que restar p + q a los grados de libertad?
7. ¿Qué entrega un intervalo de predicción que no entrega uno de confianza? ¿Por qué se ensancha al aumentar h y por qué se estabiliza en un ARMA?
8. ¿Cómo cambia un modelo al agregar la parte estacional (P, Q, s)? ¿Qué se ve en la ACF de un AR estacional?
9. ¿Qué es una tendencia determinística y una estocástica? ¿Cuándo diferenciar y cuándo ajustar una recta?
10. ¿Por qué el estadístico de Dickey-Fuller no sigue una t de Student? ¿Qué significa un p grande?
11. ¿Por qué se parte una serie en el tiempo y no al azar? ¿Qué referencias simples debe vencer un modelo?
12. ¿Qué problemas tiene el SARIMA del cuaderno de cierre (coeficientes AR ≈ 1, temperatura futura conocida)?

<details><summary><b>Respuestas</b></summary>

1. E(εₜ) = 0, Var(εₜ) = σ², Cov(εₜ, εₛ) = 0 (s ≠ t): no pide independencia (que implica covarianza 0 pero no al revés). Tendencia viola la media constante; la varianza creciente, la varianza constante.
2. r(k) = Σ(yₜ − ȳ)(yₜ₊ₖ − ȳ)/Σ(yₜ − ȳ)² con la media global. Bajo ruido blanco r(k) ≈ N(0, 1/n): bandas ±1.96/√n. Se esperan ≈ 1 de 20 por azar (5 %).
3. Yₜ = Tₜ + Sₜ + εₜ; con media móvil 2×12 se pierden 6 meses por lado. La regresión con tendencia y dummies no modela la dependencia del error, que queda (ACF(1) y ACF(12) altas, Durbin-Watson ≈ 0.9).
4. AR: ACF decae y PACF se corta en p; MA: ACF se corta en q y PACF decae; ARMA: ambas decaen. AR(1): |φ| < 1; AR(2): φ₂ + φ₁ < 1, φ₂ − φ₁ < 1, |φ₂| < 1; MA invertible: |θ| < 1 (raíces fuera del círculo).
5. Porque cada método calcula el AIC con su función objetivo (mínimos cuadrados condicionales o verosimilitud exacta): dan números y hasta ganadores distintos. Se miran también la significancia de los parámetros, la parsimonia, los residuos y el error fuera de muestra.
6. H₀: ρ₁ = … = ρₘ = 0 (los residuos son ruido blanco). Los residuos son calculados con parámetros estimados, lo que reduce los grados de libertad a m − p − q.
7. Es sobre un valor futuro (incluye el azar del dato nuevo), no sobre un parámetro. Se ensancha porque se acumula incertidumbre; en un ARMA estacionario se estabiliza porque el pronóstico converge a la media y el error a la desviación de la serie.
8. Se agregan polinomios en Bˢ; el AR estacional(1) con s = 12 tiene ACF = Φ^(k/12) en múltiplos de 12 y 0 en los demás, y PACF con un pico en 12.
9. Determinística: oscila alrededor de una recta fija → ajustar la recta y modelar los errores con ARMA. Estocástica: la tendencia cambia (caminata) → diferenciar (d = 1). Diferenciar una recta crea un MA(1) con θ = 1 y más varianza.
10. Bajo H₀ (raíz unitaria) el regresor yₜ₋₁ no es estacionario, así que el t tiene una distribución asimétrica y corrida a la izquierda (valor crítico ≈ −2.86). Un p grande significa que no se puede descartar la raíz unitaria.
11. Para no filtrar el futuro al entrenamiento (la serie está ordenada y es dependiente). Debe vencer al ingenuo (último valor), al ingenuo estacional y, en series estacionarias, a la media.
12. Sin intercepto ni diferenciación, el SARIMA usa un AR ≈ 1 para sostener el nivel (casi raíz unitaria), y el SARIMAX usó la temperatura real del periodo de prueba (en la práctica habría que pronosticarla), además la temperatura comparte tendencia con los viajes.
</details>

---
✍️ **Mis notas de hoy** (qué entendí, qué me costó, qué preguntaría en clase):

-

---
**Próximas sesiones:** Machine Learning (Clases 30–32), evaluación y selección de modelos (Clase 38) y evaluación estadística con bootstrap (Clase 43). Revisa `Guias/apuntes/04_Ejercicios_Cuaderno_SeriesTiempo.pdf` si quedó algo pendiente de practicar a mano.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", OUT, "| celdas:", len(cells))
