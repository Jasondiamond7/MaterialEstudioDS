"""Figuras de las Clases 25-27 (Modelos predictivos de series de tiempo, Prof. Cristian Vásquez, PUC).
Fuentes: tres presentaciones (01. Clase 1, 01. Clase 2 y 01. Clase 4 Series de Tiempo.pdf), dos cuadernos en R (Código Clase 1 y Clase 2),
el cuaderno de cierre en Python (DS26_28_Cierre_de_Módulo.ipynb, Clase 28) y las tres transcripciones corregidas.
La carpeta de la clase se lee sin modificarla; los datos se copian a Guias/datos/series_tiempo/ (CSV) y de ahí leen los PDFs y el notebook.
Las funciones de la sección «NÚCLEO» se copian tal cual al notebook. Escribe c25_27_numeros.json.
Ejecutar:  python -X utf8 figuras_c25_27.py   (o con nombres de paso: python -X utf8 figuras_c25_27.py descomposicion arbol)."""
import inspect
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, pacf as sm_pacf

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C25 = "#65A30D"
C25_BG = "#F7FEE7"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
TEAL = "#0D9488"
VIOLET = "#7C3AED"
GREEN = "#16A34A"
NUM = {}
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CLASE = BASE / "Clase 25, 26, 27 - Modelos predictivos de series de tiempo (Profesor cristian vasquez)"
DATOS = HERE.parent / "datos" / "series_tiempo"
URL_CITYBIKE = "https://docs.google.com/spreadsheets/d/145W6_H1vD_nnwNvDi-Fju7071NUumpnD4Kr7mSFosnQ/export?format=csv"
# Serie USAccDeaths (muertes accidentales mensuales en EE. UU., 1973-1978): es un conjunto de datos de R (datasets) que el profesor usa en la Clase 27
# pero que NO viene en la carpeta; se incluye aquí para poder reproducir sus resultados.
USACC = [9007, 8106, 8928, 9137, 10017, 10826, 11317, 10744, 9713, 9938, 9161, 8927,
         7750, 6981, 8038, 8422, 8714, 9512, 10120, 9823, 8743, 9129, 8710, 8680,
         8162, 7306, 8124, 7870, 9387, 9556, 10093, 9620, 8285, 8466, 8160, 8034,
         7717, 7461, 7767, 7925, 8623, 8945, 10078, 9179, 8037, 8488, 7874, 8647,
         7792, 6957, 7726, 8106, 8890, 9299, 10625, 9302, 8314, 8850, 8265, 8796,
         7836, 6892, 7791, 8129, 9115, 9434, 10484, 9827, 9110, 9070, 8633, 9240]


# ============================================================================ NÚCLEO (se copia al notebook)
def acf_manual(x, nlags=20):
    """ACF muestral como en la diapositiva 21: suma de productos de desviaciones a distancia k sobre la suma de cuadrados (misma media global)."""
    x = np.asarray(x, float)
    d = x - x.mean()
    den = np.sum(d ** 2)
    return np.array([np.sum(d[: len(d) - k] * d[k:]) / den for k in range(nlags + 1)])


def pacf_yw(x, nlags=20):
    """PACF muestral con Durbin-Levinson a partir de la ACF muestral: el último coeficiente φ_kk de un AR(k) ajustado por Yule-Walker."""
    r = acf_manual(x, nlags)
    pac = [1.0, r[1]]
    phi = np.array([r[1]])
    for k in range(2, nlags + 1):
        num = r[k] - np.sum(phi * r[k - 1:0:-1])
        den = 1 - np.sum(phi * r[1:k])
        kk = num / den
        phi = np.r_[phi - kk * phi[::-1], kk]
        pac.append(kk)
    return np.array(pac)


def simular_arma(ar, ma, n, rng, sd=1.0, burn=300):
    """Simula un ARMA con la convención de R (arima.sim): y_t = Σ ar_i y_{t-i} + e_t + Σ ma_j e_{t-j}. Ojo: las diapositivas escriben MA con signo menos (θ = -ma)."""
    ar, ma = np.atleast_1d(ar).astype(float), np.atleast_1d(ma).astype(float)
    e = rng.normal(0, sd, n + burn)
    y = np.zeros(n + burn)
    for t in range(n + burn):
        y[t] = e[t]
        for i, a in enumerate(ar):
            if t - 1 - i >= 0:
                y[t] += a * y[t - 1 - i]
        for j, b in enumerate(ma):
            if t - 1 - j >= 0:
                y[t] += b * e[t - 1 - j]
    return y[burn:]


def descomponer_aditiva(y, s=12):
    """Descomposición aditiva clásica (igual a decompose() de R): tendencia = media móvil centrada 2×s, estacionalidad = media por mes de (y - tendencia) recentrada, ruido = lo que queda."""
    y = np.asarray(y, float)
    n = len(y)
    w = np.r_[0.5, np.ones(s - 1), 0.5] / s                      # pesos de la media móvil centrada de longitud s+1 (s par)
    tend = np.full(n, np.nan)
    for t in range(s // 2, n - s // 2):
        tend[t] = np.sum(w * y[t - s // 2: t + s // 2 + 1])
    desv = y - tend
    fig = np.array([np.nanmean(desv[k::s]) for k in range(s)])
    fig = fig - fig.mean()
    est = np.tile(fig, n // s + 1)[:n]
    return tend, est, y - tend - est


def arma_css(y, p, q):
    """Ajuste por suma de cuadrados condicional, igual a tseries::arma de R (la que usa el profesor):
    se condiciona en las primeras max(p, q) observaciones. Devuelve coeficientes (phi, theta, media), el «intercepto» de tseries (constante AR = media·(1-Σφ)),
    CSS, σ², AIC al estilo tseries, errores estándar (Hessiana numérica) y residuos."""
    y = np.asarray(y, float)
    n, k0 = len(y), max(p, q)

    def resid(par):
        c, phi, th = par[0], par[1:1 + p], par[1 + p:]
        e = np.zeros(n)
        for t in range(k0, n):
            ar = sum(phi[i] * (y[t - 1 - i] - c) for i in range(p) if t - 1 - i >= 0)
            ma = sum(th[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= k0)
            e[t] = (y[t] - c) - ar - ma
        return e

    def css(par):
        return np.sum(resid(par) ** 2)

    r = optimize.minimize(css, np.r_[y.mean(), np.zeros(p + q)], method="BFGS")
    e = resid(r.x)
    sse = float(np.sum(e ** 2))
    s2 = sse / (n - k0 - 1)
    k = p + q + 1
    aic = 2 * k + n * (1 + np.log(2 * np.pi) + np.log(s2))
    h = 1e-4
    H = np.zeros((k, k))
    f0 = css(r.x)
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (css(r.x + ei + ej) - css(r.x + ei - ej) - css(r.x - ei + ej) + css(r.x - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(H / 2) * s2                              # Hessiana de css/2 -> varianza aproximada
    se = np.sqrt(np.diag(cov))
    media, phi, th = r.x[0], r.x[1:1 + p], r.x[1 + p:]
    return dict(media=float(media), phi=phi, theta=th, intercepto=float(media * (1 - phi.sum())), css=sse, sigma2=float(s2), aic=float(aic), se=se, resid=e[k0:])


def ljung_box(res, m, ddof=0):
    """Estadístico Q de Ljung-Box (diapositiva 53) y su valor p con m - ddof grados de libertad (ddof = p + q de los parámetros ARMA estimados)."""
    res = np.asarray(res, float)
    n = len(res)
    r = acf_manual(res, m)[1:]
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, m + 1)))
    return Q, float(stats.chi2.sf(Q, m - ddof))


def adf_regresion(y, tendencia=False):
    """Test de Dickey-Fuller simple como regresión: Δy_t = c (+ b·t) + γ·y_{t-1} + e_t. Devuelve γ̂ y su estadístico t (que NO sigue una t de Student sino la distribución DF)."""
    y = np.asarray(y, float)
    dy, ylag = np.diff(y), y[:-1]
    cols = [np.ones(len(dy)), ylag] + ([np.arange(1, len(dy) + 1)] if tendencia else [])
    X = np.column_stack(cols)
    res = sm.OLS(dy, X).fit()
    return float(res.params[1]), float(res.tvalues[1])


def particion_temporal(y, h):
    """Partición temporal: entrena con todo menos las últimas h observaciones y evalúa en esas h (nunca se baraja el orden)."""
    return y.iloc[:-h], y.iloc[-h:]


def ingenuo(train, h, s=1):
    """Pronóstico ingenuo (s = 1: repite el último valor) o ingenuo estacional (s = 12: repite el valor de hace un año)."""
    ult = np.asarray(train)[-s:]
    return np.tile(ult, h // s + 1)[:h]


def mae(real, pred):
    return float(np.mean(np.abs(np.asarray(real) - np.asarray(pred))))


def rmse(real, pred):
    return float(np.sqrt(np.mean((np.asarray(real) - np.asarray(pred)) ** 2)))


def ar1_pronostico(y_ult, mu, phi, sigma, h):
    """Pronóstico a h pasos de un AR(1) con media mu: ŷ_{t+h} = mu + phi^h (y_t - mu); varianza del error = σ² (1 - phi^{2h})/(1 - phi²)."""
    hh = np.arange(1, h + 1)
    punt = mu + phi ** hh * (y_ult - mu)
    var = sigma ** 2 * (1 - phi ** (2 * hh)) / (1 - phi ** 2)
    return punt, np.sqrt(var)


# ============================================================================ datos y utilidades
def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


def preparar_datos():
    """Lee los archivos de la carpeta de la clase (sin modificarla) y escribe CSV ordenados en Guias/datos/series_tiempo/."""
    DATOS.mkdir(parents=True, exist_ok=True)
    pas = pd.read_excel(CLASE / "Passengers.xlsx")
    pas = pas[pas["Month"].astype(str) != "TOTAL"].copy()
    for c in ("DOMESTIC", "INTERNATIONAL", "TOTAL"):
        pas[c] = pas[c].astype(str).str.replace(",", "").astype(float)
    pas["fecha"] = pd.to_datetime(dict(year=pas["Year"].astype(int), month=pas["Month"].astype(int), day=1))
    pas[["fecha", "DOMESTIC", "INTERNATIONAL", "TOTAL"]].to_csv(DATOS / "passengers.csv", index=False)
    pd.read_excel(CLASE / "tree.xlsx").to_csv(DATOS / "tree_idaho.csv", index=False)
    pd.read_excel(CLASE / "pm25.xlsx").rename(columns={"Fecha": "fecha", "Valor": "pm25"}).to_csv(DATOS / "pm25.csv", index=False)
    pd.read_excel(CLASE / "desocupacion.xlsx").rename(columns={"Periodo": "fecha", "Tasa": "tasa"}).to_csv(DATOS / "desocupacion.csv", index=False)
    pd.read_excel(CLASE / "IAC_INE.xlsx").rename(columns={"Fecha": "fecha", "IAC_2018Lag0": "iac"}).to_csv(DATOS / "iac_ine.csv", index=False)
    ipc = pd.read_csv(CLASE / "ipc_var.txt", sep=r"\s+")                 # 12 filas (meses) x 22 columnas (años 1988-2009)
    pd.DataFrame({"fecha": pd.date_range("1988-01-01", periods=264, freq="MS"), "var_ipc": ipc.values.T.reshape(-1)}).to_csv(DATOS / "ipc_var.csv", index=False)
    ips = pd.read_csv(CLASE / "ipsa.txt", sep=r"\s+")                    # 21 filas (años 1980-2000) x 12 columnas (meses)
    pd.DataFrame({"fecha": pd.date_range("1980-01-01", periods=252, freq="MS"), "ipsa": ips.values.reshape(-1)}).to_csv(DATOS / "ipsa.csv", index=False)
    pd.DataFrame({"fecha": pd.date_range("1973-01-01", periods=72, freq="MS"), "muertes": USACC}).to_csv(DATOS / "usaccdeaths.csv", index=False)


_D = {}


def datos():
    if not _D:
        if not (DATOS / "usaccdeaths.csv").exists():
            preparar_datos()
        rd = lambda f, **kw: pd.read_csv(DATOS / f, parse_dates=["fecha"], **kw)
        pas = rd("passengers.csv")
        _D.update(pas=pas[pas["fecha"] >= "2010-01-01"].reset_index(drop=True), pas_todo=pas, tree=pd.read_csv(DATOS / "tree_idaho.csv"), pm25=rd("pm25.csv"), des=rd("desocupacion.csv"),
                  iac=rd("iac_ine.csv"), ipc=rd("ipc_var.csv"), ipsa=rd("ipsa.csv"), usa=rd("usaccdeaths.csv"))
    return _D


def ts_mensual(df, col):
    return pd.Series(df[col].values.astype(float), index=pd.DatetimeIndex(df["fecha"], freq="MS"))


def cita(ax, txt, x=0.98, y=0.96, ha="right", fs=8, col=INK):
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=fs, color=col, bbox=dict(fc="white", ec=GRID, boxstyle="round,pad=0.3"))


def barras_acf(ax, vals, n, color=BLUE, titulo_txt=None, ylim=(-0.4, 1.0), bandas=True, lag0=False, lw=3):
    k = np.arange(len(vals))
    ini = 0 if lag0 else 1
    ax.vlines(k[ini:], 0, vals[ini:], color=color, lw=lw)
    ax.axhline(0, color=INK, lw=0.8)
    if bandas:
        b = 1.96 / np.sqrt(n)
        ax.axhspan(-b, b, color=AMBER, alpha=0.18, lw=0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("rezago k")
    if titulo_txt:
        titulo(ax, titulo_txt)
    style(ax, "y")



# ============================================================================ 1. esquema de las tres clases
def esquema_fig():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    pasos = [("1 · Mirar", "gráfico de la serie\ntendencia · estacionalidad\nvarianza · cambios bruscos", "C25", C25),
             ("2 · ¿Ruido blanco?", "media 0 · varianza constante\ncovarianza 0 · ACF con\nbandas ±1.96/√n", "C25", C25),
             ("3 · Descomponer", "Y = T + S + ruido\n(regresión con dummies,\narmónicas, Box-Cox)", "C25", C25),
             ("4 · ¿Estacionaria?", "diferenciar (1−B)ᵈ y (1−Bˢ)\ntest de Dickey-Fuller\n(raíz unitaria)", "C27", TEAL),
             ("5 · Identificar", "ACF y PACF:\nAR · MA · ARMA\n+ parte estacional SARMA", "C26-27", BLUE),
             ("6 · Estimar y elegir", "máxima verosimilitud\nmenor AIC · parsimonia\nparámetros significativos", "C26", BLUE),
             ("7 · Diagnosticar", "residuos = ruido blanco:\nACF · Ljung-Box · QQ\n(si no, volver al paso 5)", "C26", AMBER),
             ("8 · Pronosticar", "predictor puntual +\nintervalo de predicción\nvalidar en el tiempo", "C26-27", RED)]
    w = 1.52
    for i, (t, txt, cl, col) in enumerate(pasos):
        x0 = 0.1 + i * 1.615
        ax.add_patch(FancyBboxPatch((x0, 0.62), w, 2.35, boxstyle="round,pad=0.03", fc="white", ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((x0, 2.3), w, 0.67, boxstyle="round,pad=0.03", fc=col, ec=col))
        ax.text(x0 + w / 2, 2.63, t, ha="center", va="center", fontsize=7.8, color="white", fontweight="bold")
        ax.text(x0 + w / 2, 1.5, txt, ha="center", va="center", fontsize=6.3, color=INK)
        ax.text(x0 + w / 2, 0.84, cl, ha="center", va="center", fontsize=8, color=col, fontweight="bold")
        if i < len(pasos) - 1:
            ax.annotate("", (x0 + w + 0.1, 1.6), (x0 + w + 0.005, 1.6), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.text(0.1, 0.22, "Box-Jenkins es un ciclo: identificar → estimar → diagnosticar (y volver si los residuos no son ruido blanco).", fontsize=8.4, color=AMBER, ha="left")
    ax.text(0.1, 0.02, "Datos: IPC, IPSA, PM2.5, desocupación, pasajeros, anillos de árbol, accidentes (EE. UU.), CityBike, IAC", fontsize=8, color=MUTED, ha="left")
    save(fig, "c25_27_esquema.png")


# ============================================================================ 2. series de ejemplo de la clase
def ejemplos_fig():
    d = datos()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 6.1))
    ipc = d["ipc"]
    ax = axs[0, 0]
    ax.plot(ipc["fecha"], ipc["var_ipc"], color=BLUE, lw=1)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set(ylabel="variación mensual del IPC (%)")
    titulo(ax, "IPC: variación mensual 1988–2009 · la inflación baja y se «calma»")
    style(ax)
    ips = d["ipsa"][d["ipsa"]["fecha"] >= "1990-01-01"]
    ax = axs[0, 1]
    ax.plot(ips["fecha"], ips["ipsa"], color=VIOLET, lw=1.3)
    ax.set(ylabel="IPSA (puntos)")
    titulo(ax, "IPSA 1990–2000: tendencia y varianza que crece con el nivel")
    style(ax)
    pm = d["pm25"]
    ax = axs[1, 0]
    ax.plot(pm["fecha"], pm["pm25"], color=TEAL, lw=0.5, alpha=0.8)
    mm = pm.set_index("fecha")["pm25"].resample("MS").mean()
    ax.plot(mm.index, mm.values, color=RED, lw=1.6, label="media mensual")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set(ylabel="PM2.5 (μg/m³)")
    titulo(ax, "PM2.5 diario 2000–2009: estacionalidad (picos de invierno) y tendencia a la baja")
    style(ax)
    ds = d["des"]
    ax = axs[1, 1]
    ax.plot(ds["fecha"], ds["tasa"], color=AMBER, lw=1.5)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"), color=RED, alpha=0.12)
    ax.text(pd.Timestamp("2020-04-01"), ds["tasa"].max() * 0.99, "pandemia\n(intervención)", fontsize=8, color=RED, va="top")
    ax.set(ylabel="tasa de desocupación (%)")
    titulo(ax, "Desocupación 2010–2026: tendencia que cambia y un salto (intervención)")
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_ejemplos.png")
    ipc_v = ipc["var_ipc"].values
    mal = ipc_v.reshape(22, 12).T.reshape(-1)                      # lo que produce el código R: lee la tabla 12×22 por filas, mezclando años y meses
    NUM["ejemplos"] = dict(ipc_n=len(ipc), ipsa_n=len(ips), pm_n=len(pm), pm_ini=str(pm["fecha"].min().date()), pm_fin=str(pm["fecha"].max().date()),
                           des_n=len(ds), des_max=float(ds["tasa"].max()), des_min=float(ds["tasa"].min()), des_max_fecha=str(ds.loc[ds["tasa"].idxmax(), "fecha"].date()),
                           ipc_acf1_ok=float(acf_manual(ipc_v, 1)[1]), ipc_acf1_codigo_R=float(acf_manual(mal, 1)[1]),
                           pm_media_ini=float(pm[pm["fecha"] < "2002-01-01"]["pm25"].mean()), pm_media_fin=float(pm[pm["fecha"] >= "2008-01-01"]["pm25"].mean()))


# ============================================================================ 3. ruido blanco y su ACF
def ruido_fig():
    rng = np.random.default_rng(250101)
    n = 144
    t = np.arange(1, n + 1)
    f = pd.date_range("2008-01-01", periods=n, freq="MS")
    tend = 2 + 4 * t / 24 + rng.normal(0, 2, n)                    # a: tendencia
    het = rng.normal(0, 0.5 + 5 * t / n)                             # b: heterocedasticidad
    rb = rng.normal(0, 2, n)                                         # c: ruido blanco
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 5.6))
    for ax, y, tt, col in ((axs[0, 0], tend, "a) Tendencia: la media sube → NO es ruido blanco", RED), (axs[0, 1], het, "b) Heterocedasticidad: la varianza crece → NO", RED),
                           (axs[0, 2], rb, "c) Media 0 y varianza constante → compatible", GREEN)):
        ax.plot(f, y, color=BLUE, lw=1.2)
        ax.axhline(0, color=INK, lw=0.7)
        titulo(ax, tt, 8.6)
        ax.title.set_color(col)
        style(ax)
    a = [acf_manual(x, 20) for x in (tend, het, rb)]
    nombres = ["ACF de a): decae muy lento", "ACF de b): sin autocorrelación, pero falla la varianza", "ACF de c): todo dentro de las bandas"]
    for ax, v, nm in zip(axs[1], a, nombres):
        barras_acf(ax, v, n, BLUE, nm, ylim=(-0.3, 1.0))
        ax.set_ylabel("ACF")
    fig.tight_layout()
    save(fig, "c25_27_ruido_blanco.png")
    b = 1.96 / np.sqrt(n)
    NUM["ruido"] = dict(n=n, banda=float(b), fuera_c=int(np.sum(np.abs(a[2][1:]) > b)), de_c=20, acf1_tend=float(a[0][1]), acf1_het=float(a[1][1]), acf1_rb=float(a[2][1]))


# ============================================================================ 4. qué es la ACF: dispersión a rezago k
def acf_idea_fig():
    rng = np.random.default_rng(2501)
    y = simular_arma([0.8], [], 200, rng)
    r = acf_manual(y, 20)
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.7))
    for ax, k, col in ((axs[0], 1, BLUE), (axs[1], 5, AMBER)):
        ax.scatter(y[:-k], y[k:], s=12, color=col, alpha=0.7)
        ax.set(xlabel="yₜ", ylabel=f"yₜ₊{k}")
        titulo(ax, f"Rezago k = {k}: correlación entre yₜ y yₜ₊{k} = {r[k]:.2f}")
        style(ax)
    barras_acf(axs[2], r, 200, BLUE, "ACF muestral (n = 200)", ylim=(-0.3, 1.0))
    axs[2].plot(np.arange(1, 21), 0.8 ** np.arange(1, 21), "o", color=RED, ms=4, label="teórica φᵏ, φ = 0.8")
    axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "c25_27_acf_idea.png")
    NUM["acf_idea"] = dict(r1=float(r[1]), r5=float(r[5]), teo5=float(0.8 ** 5), banda=float(1.96 / np.sqrt(200)))



# ============================================================================ 5. descomposición aditiva de los pasajeros (Clase 25)
def descomposicion_fig():
    d = datos()
    y = ts_mensual(d["pas"], "INTERNATIONAL")
    t, s, r = descomponer_aditiva(y.values)
    fig = plt.figure(figsize=(13, 4.9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.5, 1], hspace=0.7, wspace=0.18)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    for ax, v, nm, col in zip(axs, (y.values, t, s, r), ("observada", "tendencia (media móvil 2×12)", "estacional (12 valores que se repiten)", "ruido = observada − tendencia − estacional"), (INK, BLUE, AMBER, RED)):
        ax.plot(y.index, v / 1e6, color=col, lw=1.3)
        ax.set_ylabel("millones", fontsize=7.5)
        titulo(ax, nm, 8.2)
        style(ax)
    rr = r[~np.isnan(r)]
    ax = fig.add_subplot(gs[:2, 1])
    barras_acf(ax, acf_manual(rr, 24), len(rr), RED, "ACF del ruido: sigue habiendo estructura", ylim=(-0.5, 1.0))
    mes = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax = fig.add_subplot(gs[2:, 1])
    ax.bar(mes, s[:12] / 1e6, color=[AMBER if v > 0 else BLUE for v in s[:12]])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("efecto (millones)", fontsize=8)
    titulo(ax, "Efecto estacional de cada mes", 8.8)
    style(ax, "y")
    save(fig, "c25_27_descomposicion.png")
    NUM["descomp"] = dict(n=len(y), ini=str(y.index[0].date()), fin=str(y.index[-1].date()), mes_max=int(np.argmax(s[:12]) + 1), efecto_max=float(s[:12].max()), efecto_min=float(s[:12].min()),
                          na_ini=int(np.isnan(t).sum()), acf12_ruido=float(acf_manual(rr, 12)[12]), acf1_ruido=float(acf_manual(rr, 1)[1]), banda=float(1.96 / np.sqrt(len(rr))))


# ============================================================================ 6. regresión con tendencia y dummies mensuales (Clase 25)
def regresion_fig():
    d = datos()
    df = d["pas"].copy()
    df["t"] = np.arange(1, len(df) + 1)
    df["mes"] = df["fecha"].dt.month
    m = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df).fit()
    res = m.resid.values
    dw = float(sm.stats.durbin_watson(res))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    ax.plot(df["fecha"], df["INTERNATIONAL"] / 1e6, color=BLUE, lw=1.4, label="observada")
    ax.plot(df["fecha"], m.fittedvalues / 1e6, color=RED, lw=1.4, label=f"ajuste (R² aj. = {m.rsquared_adj:.3f})")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("millones de pasajeros")
    titulo(ax, "Yₜ = β₀ + β₁·t + Σ αⱼ·dummyⱼ + εₜ")
    style(ax)
    ax = axs[1]
    ax.plot(df["fecha"], res / 1e3, color=VIOLET, lw=1.2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("residuo (miles)")
    titulo(ax, "Residuos: primero bajan y luego suben (curvatura)")
    style(ax)
    barras_acf(axs[2], acf_manual(res, 24), len(res), VIOLET, f"ACF de los residuos · Durbin-Watson = {dw:.2f}", ylim=(-0.5, 1.0))
    fig.tight_layout()
    save(fig, "c25_27_regresion.png")
    co = m.params
    NUM["regresion"] = dict(r2=float(m.rsquared), r2aj=float(m.rsquared_adj), intercepto=float(co["Intercept"]), tiempo=float(co["t"]), m7=float(co["C(mes)[T.7]"]), m8=float(co["C(mes)[T.8]"]),
                            m6=float(co["C(mes)[T.6]"]), m2=float(co["C(mes)[T.2]"]), m11=float(co["C(mes)[T.11]"]), p_oct=float(m.pvalues["C(mes)[T.10]"]), p_sep=float(m.pvalues["C(mes)[T.9]"]),
                            dw=dw, acf1=float(acf_manual(res, 1)[1]), acf12=float(acf_manual(res, 12)[12]), n=len(df), sigma=float(np.sqrt(m.scale)),
                            n_signif=int((m.pvalues.drop("Intercept") < 0.05).sum()), n_coef=int(len(m.params) - 1))
    # tendencia lineal frente a una pandemia futura: pronóstico fuera de muestra con corte temporal
    tr, te = df.iloc[:-12], df.iloc[-12:]
    m2 = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", tr).fit()
    pr = m2.predict(te)
    NUM["regresion"]["mae_ultimos12"] = mae(te["INTERNATIONAL"], pr)
    NUM["regresion"]["mape_ultimos12"] = float(np.mean(np.abs(te["INTERNATIONAL"] - pr) / te["INTERNATIONAL"]) * 100)


# ============================================================================ 7. armónicas y Box-Cox (Clase 25)
def armonica_boxcox_fig():
    rng = np.random.default_rng(2507)
    t = np.arange(1, 401)
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.4))
    comps = [([(4, 40)], "1 frecuencia"), ([(4, 40), (3, 100)], "2 frecuencias"), ([(4, 40), (3, 100), (6, 17)], "3 frecuencias")]
    for ax, (cs, nm) in zip(axs[0], comps):
        base = sum(a * np.sin(2 * np.pi * t / p) for a, p in cs)
        ax.plot(t, base + rng.normal(0, 1.5, len(t)), color=BLUE, lw=0.8, alpha=0.7, label="datos")
        ax.plot(t, base, color=RED, lw=1.6, label="componente armónico")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        titulo(ax, f"Regresión armónica: {nm}", 8.8)
        style(ax)
    d = datos()
    y = ts_mensual(d["pas"], "INTERNATIONAL")
    ps = d["pas_todo"].copy()
    yy = ts_mensual(ps, "INTERNATIONAL")
    lam = float(stats.boxcox_normmax(yy.values, method="mle"))
    # serie con varianza creciente: IPSA en niveles, 1990-2000
    ips = ts_mensual(d["ipsa"][d["ipsa"]["fecha"] >= "1990-01-01"], "ipsa")
    lam_i = float(stats.boxcox_normmax(ips.values, method="mle"))
    ax = axs[1, 0]
    ax.plot(ips.index, ips.values, color=VIOLET, lw=1.3)
    titulo(ax, "IPSA en niveles: saltos mensuales más grandes con el nivel", 8.6)
    style(ax)
    ax = axs[1, 1]
    ax.plot(ips.index, np.log(ips.values), color=TEAL, lw=1.3)
    titulo(ax, "log(IPSA) (α = 0): saltos más parejos", 8.6)
    style(ax)
    ax = axs[1, 2]
    grid = np.linspace(-1, 1.5, 51)
    ll = [stats.boxcox_llf(a, ips.values) for a in grid]
    ax.plot(grid, ll, color=INK, lw=1.6)
    ax.axvline(lam_i, color=RED, ls="--")
    ax.axvline(0, color=TEAL, ls=":")
    ax.set(xlabel="α de Box-Cox", ylabel="log-verosimilitud")
    titulo(ax, f"α̂ = {lam_i:.2f} (cercano a 1: Box-Cox no exige transformar)", 8.6)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_armonica_boxcox.png")
    # amplitud de la oscilación según nivel (desviación de los retornos mensuales de primera y segunda mitad)
    mitad = len(ips) // 2
    NUM["boxcox"] = dict(alpha_ipsa=lam_i, alpha_pasajeros=lam, sd_dif_mitad1=float(np.diff(ips.values[:mitad]).std()), sd_dif_mitad2=float(np.diff(ips.values[mitad:]).std()),
                         sd_dlog_mitad1=float(np.diff(np.log(ips.values[:mitad])).std()), sd_dlog_mitad2=float(np.diff(np.log(ips.values[mitad:])).std()))



# ============================================================================ 8. ACF y PACF de AR, MA y ARMA (Clase 26)
def _teorica(ar, ma, nl=25):
    from statsmodels.tsa.arima_process import ArmaProcess
    ap = ArmaProcess(np.r_[1, -np.atleast_1d(ar)] if len(np.atleast_1d(ar)) else np.array([1.0]), np.r_[1, np.atleast_1d(ma)] if len(np.atleast_1d(ma)) else np.array([1.0]))
    return ap.acf(nl + 1)[:nl + 1], np.r_[1.0, ap.pacf(nl + 1)[1:nl + 1]]


def _panel_acf_pacf(titulo_fig, casos, nombre, n=500, nl=25, seed=2601):
    fig, axs = plt.subplots(2, len(casos), figsize=(4.3 * len(casos), 5.4))
    out = []
    for j, (ttl, ar, ma) in enumerate(casos):
        rng = np.random.default_rng(seed + j)
        y = simular_arma(ar, ma, n, rng)
        ta, tp = _teorica(ar, ma, nl)
        ma_a, ma_p = acf_manual(y, nl), pacf_yw(y, nl)
        for i, (muestra, teo, nm) in enumerate(((ma_a, ta, "ACF"), (ma_p, tp, "PACF"))):
            ax = axs[i, j]
            barras_acf(ax, muestra, n, BLUE if i == 0 else VIOLET, None, ylim=(-1, 1.02))
            ax.plot(np.arange(1, nl + 1), teo[1:], "o", color=RED, ms=3.2, label="teórica" if j == 0 else None)
            titulo(ax, f"{nm} · {ttl}", 8.6)
            if j == 0:
                ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        out.append(dict(caso=ttl, acf=[float(v) for v in ta[1:6]], pacf=[float(v) for v in tp[1:6]], acf_muestra=[float(v) for v in ma_a[1:6]], pacf_muestra=[float(v) for v in ma_p[1:6]]))
    fig.tight_layout()
    save(fig, nombre)
    return out


def ar_fig():
    NUM["ar"] = _panel_acf_pacf("AR", [("AR(1), φ = 0.8", [0.8], []), ("AR(1), φ = −0.75", [-0.75], []), ("AR(2), φ = (0.5, 0.3)", [0.5, 0.3], [])], "c25_27_ar.png")


def ma_fig():
    # convención de R: ma = +0.9 equivale a θ = −0.9 en la notación de las diapositivas (Yₜ = εₜ − θ εₜ₋₁)
    NUM["ma"] = _panel_acf_pacf("MA", [("MA(1), θ = −0.9", [], [0.9]), ("MA(2), θ = (−0.7, −0.2)", [], [0.7, 0.2]), ("ARMA(1,1), φ = 0.8, θ = 0.5", [0.8], [-0.5])], "c25_27_ma_arma.png", n=700, seed=2610)
    m1 = NUM["ma"][0]
    NUM["ma_teo"] = dict(rho1_theta09=float(0.9 / (1 + 0.81)), rho1_teorica=m1["acf"][0], pacf_k1=m1["pacf"][0], pacf_k2=m1["pacf"][1])


# ============================================================================ 9. cómo se identifica: tabla de patrones y condiciones de estacionariedad
def identificacion_fig():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    ax.axis("off")
    filas = [("", "ACF", "PACF"), ("AR(p)", "decae (exponencial\no sinusoidal)", f"se corta en p"), ("MA(q)", "se corta en q", "decae"), ("ARMA(p, q)", "decae", "decae"), ("ruido blanco", "nada significativo", "nada significativo")]
    for i, (a, b, c) in enumerate(filas):
        yy = 0.88 - i * 0.2
        col = INK if i else MUTED
        ax.text(0.02, yy, a, fontsize=10.5, color=BLUE if i in (1, 3) else (AMBER if i == 2 else col), fontweight="bold", va="center")
        ax.text(0.34, yy, b, fontsize=9, color=INK, va="center")
        ax.text(0.7, yy, c, fontsize=9, color=INK, va="center")
        if i:
            ax.plot([0, 1], [yy + 0.1, yy + 0.1], color=GRID, lw=0.8)
    titulo(ax, "Patrón de identificación (Box-Jenkins)")
    ax = axs[1]
    p1 = np.linspace(-2, 2, 100)
    ax.fill_between(p1, np.maximum(-1, p1 - 1), np.minimum(1, 1 - p1), where=(np.minimum(1, 1 - p1) > np.maximum(-1, p1 - 1)), color=C25, alpha=0.25)
    ax.plot([-2, 0, 2, -2], [-1, 1, -1, -1], color=C25, lw=1.5)
    for (a, b, nm, col) in ((0.5, 0.3, "(0.5, 0.3)", GREEN), (1.2, 0.0, "(1.2, 0)", RED), (0.4, 0.7, "(0.4, 0.7)", AMBER)):
        ax.plot(a, b, "o", color=col)
        ax.text(a + 0.05, b + 0.06, nm, fontsize=8, color=col)
    ax.set(xlabel="φ₁", ylabel="φ₂", xlim=(-2.2, 2.2), ylim=(-1.2, 1.2))
    titulo(ax, "AR(2) estacionario: dentro del triángulo\nφ₂+φ₁<1 · φ₂−φ₁<1 · |φ₂|<1", 8.6)
    style(ax)
    ax = axs[2]
    rng = np.random.default_rng(2611)
    for phi, col in ((0.5, GREEN), (0.95, AMBER), (1.0, RED)):
        e = rng.normal(size=250)
        y = np.zeros(250)
        for t in range(1, 250):
            y[t] = phi * y[t - 1] + e[t]
        ax.plot(y, color=col, lw=1.2, label=f"φ = {phi}" + (" (caminata aleatoria)" if phi == 1 else ""))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    titulo(ax, "AR(1): φ cerca de 1 → «memoria larga» y\nφ = 1 → no estacionaria", 8.6)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_identificacion.png")


# ============================================================================ 10. anillos de árbol: serie, ACF/PACF y cinco modelos
def _ajustes_arbol():
    y = datos()["tree"]["Idaho"].values
    out = {}
    for (p, q) in [(0, 3), (0, 2), (1, 0), (2, 0), (1, 1)]:
        c = arma_css(y, p, q)
        m = ARIMA(y, order=(p, 0, q), trend="c").fit()
        out[(p, q)] = dict(css=c, mle=m)
    return y, out


def arbol_fig():
    y, aj = _ajustes_arbol()
    n = len(y)
    d = datos()["tree"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    ax.plot(d["Year"], y, color=BLUE, lw=0.9)
    ax.axhline(y.mean(), color=RED, lw=1.2, label=f"media = {y.mean():.3f}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("ancho del anillo (mm)")
    titulo(ax, f"Anillos de un árbol de Idaho, {d['Year'].min()}–{d['Year'].max()} (n = {n})")
    style(ax)
    barras_acf(axs[1], acf_manual(y, 15), n, BLUE, "ACF: decae → parte AR (o ARMA)", ylim=(-0.2, 0.8))
    barras_acf(axs[2], pacf_yw(y, 15), n, VIOLET, "PACF: lag 1 y 2 sobresalen", ylim=(-0.2, 0.8))
    fig.tight_layout()
    save(fig, "c25_27_arbol_serie.png")
    ac, pc = acf_manual(y, 15), pacf_yw(y, 15)
    NUM["arbol"] = dict(n=n, media=float(y.mean()), mediana=float(np.median(y)), sd=float(y.std(ddof=1)), var=float(y.var(ddof=1)), iqr=float(np.subtract(*np.percentile(y, [75, 25]))),
                        minimo=float(y.min()), maximo=float(y.max()), q1=float(np.percentile(y, 25)), q3=float(np.percentile(y, 75)), acf=[float(v) for v in ac[1:6]], pacf=[float(v) for v in pc[1:6]],
                        banda=float(1.96 / np.sqrt(n)), acf_sobre_banda=int(np.sum(np.abs(ac[1:16]) > 1.96 / np.sqrt(n))), pacf_sobre_banda=int(np.sum(np.abs(pc[1:16]) > 1.96 / np.sqrt(n))))


def arbol_modelos_fig():
    y, aj = _ajustes_arbol()
    nombres = ["MA(3)", "MA(2)", "AR(1)", "AR(2)", "ARMA(1,1)"]
    claves = list(aj)
    a_css = [aj[k]["css"]["aic"] for k in claves]
    a_mle = [aj[k]["mle"].aic for k in claves]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    x = np.arange(5)
    ax.bar(x - 0.2, a_css, 0.4, color=AMBER, label="tseries::arma (CSS) – diapositivas 44-48")
    ax.bar(x + 0.2, a_mle, 0.4, color=BLUE, label="máx. verosimilitud (arima) – diapositiva 49")
    for i in range(5):
        ax.text(i - 0.2, a_css[i] + 0.3, f"{a_css[i]:.1f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, a_mle[i] + 0.3, f"{a_mle[i]:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=8)
    ax.set_ylim(90, 110)
    ax.set_ylabel("AIC")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    titulo(ax, "El AIC solo se compara dentro de un mismo método", 8.8)
    style(ax, "y")
    ax = axs[1]
    ax.axis("off")
    ax.text(0, 1.0, "Coeficientes (θ con signo de R: Yₜ = … + εₜ + θ·εₜ₋₁)", fontsize=8.4, color=MUTED, va="top")
    filas = [("modelo", "estimación (error est.)", "t")]
    c11 = aj[(1, 1)]["css"]
    for k, nm in zip(claves, nombres):
        c = aj[k]["css"]
        ests = list(c["phi"]) + list(c["theta"])
        se = c["se"][1:]
        txt = ", ".join(f"{e:+.3f}" for e in ests)
        tt = ", ".join(f"{e / s:.1f}" for e, s in zip(ests, se))
        filas.append((nm, txt, tt))
    for i, (a, b, c) in enumerate(filas):
        yy = 0.88 - i * 0.14
        ax.text(0.0, yy, a, fontsize=8.8, fontweight="bold" if i == 0 else None, va="center", color=INK)
        ax.text(0.27, yy, b, fontsize=8.2, va="center", color=INK, family="monospace" if i else None)
        ax.text(0.82, yy, c, fontsize=8, va="center", color=INK)
    titulo(ax, "Ajuste CSS (tseries)")
    ax = axs[2]
    m = aj[(1, 1)]["mle"]
    ult = 120
    yr = datos()["tree"]["Year"].values
    ax.plot(yr[-ult:], y[-ult:], color=BLUE, lw=1.1, label="observada")
    ax.plot(yr[-ult:], m.fittedvalues[-ult:], color=RED, lw=1.1, label="ajustada ARMA(1,1)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("mm")
    titulo(ax, "El ajuste sigue el nivel pero no los picos (ruido grande)", 8.8)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_arbol_modelos.png")
    NUM["modelos"] = {nm: dict(css_aic=float(aj[k]["css"]["aic"]), mle_aic=float(aj[k]["mle"].aic), css_s2=float(aj[k]["css"]["sigma2"]), css_sse=float(aj[k]["css"]["css"]),
                               phi=[float(v) for v in aj[k]["css"]["phi"]], theta=[float(v) for v in aj[k]["css"]["theta"]], intercepto=float(aj[k]["css"]["intercepto"]),
                               t=[float(v) for v in (list(aj[k]["css"]["phi"]) + list(aj[k]["css"]["theta"])) / aj[k]["css"]["se"][1:]],
                               mle_params=[float(v) for v in aj[k]["mle"].params[1:-1]], mle_media=float(aj[k]["mle"].params[0]), mle_s2=float(aj[k]["mle"].params[-1]),
                               mle_bic=float(aj[k]["mle"].bic)) for k, nm in zip(claves, nombres)}
    NUM["modelos"]["menor_aic_css"] = nombres[int(np.argmin(a_css))]
    NUM["modelos"]["menor_aic_mle"] = nombres[int(np.argmin(a_mle))]
    NUM["modelos"]["menor_bic_mle"] = nombres[int(np.argmin([aj[k]["mle"].bic for k in claves]))]


# ============================================================================ 11. diagnóstico de residuos del ARMA(1,1)
def diagnostico_fig():
    y, aj = _ajustes_arbol()
    m = aj[(1, 1)]["mle"]
    e = np.asarray(m.resid)
    z = e / np.sqrt(m.params[-1])
    n = len(z)
    m3 = aj[(1, 0)]["mle"]
    ar1_res = np.asarray(m3.resid)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.0))
    ax = axs[0, 0]
    ax.plot(z, color=BLUE, lw=0.8)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhspan(-2, 2, color=AMBER, alpha=0.15, lw=0)
    ax.set_ylabel("residuo estandarizado")
    titulo(ax, "Residuos estandarizados (casi todos entre −2 y 2)", 8.8)
    style(ax)
    barras_acf(axs[0, 1], acf_manual(e, 25), n, BLUE, "ACF de los residuos del ARMA(1,1)", ylim=(-0.3, 0.6))
    ax = axs[0, 2]
    ms = np.arange(1, 26)
    p11 = [ljung_box(e, mm_, ddof=2)[1] if mm_ > 2 else np.nan for mm_ in ms]
    p11_sin = [ljung_box(e, mm_, ddof=0)[1] for mm_ in ms]
    p10 = [ljung_box(ar1_res, mm_, ddof=1)[1] if mm_ > 1 else np.nan for mm_ in ms]
    ax.plot(ms, p11, "o-", color=GREEN, ms=3.5, label="ARMA(1,1): gl = m − 2")
    ax.plot(ms, p11_sin, "--", color=GREEN, lw=1, label="ARMA(1,1): gl = m (como el estadístico de la diapositiva)")
    ax.plot(ms, p10, "o-", color=RED, ms=3.5, label="AR(1): gl = m − 1")
    ax.axhline(0.05, color=INK, ls=":")
    ax.set(xlabel="m (rezagos incluidos)", ylabel="valor p de Ljung-Box", ylim=(-0.02, 1.02))
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    titulo(ax, "Ljung-Box: p > 0.05 = no se rechaza ruido blanco", 8.8)
    style(ax)
    ax = axs[1, 0]
    ax.hist(z, bins=25, density=True, color=BLUE, alpha=0.5, edgecolor="white")
    g = np.linspace(-4, 4, 200)
    ax.plot(g, stats.norm.pdf(g), color=RED, lw=1.5)
    titulo(ax, "Histograma vs. normal estándar", 8.8)
    style(ax)
    ax = axs[1, 1]
    (osm, osr), (sl, ic, rr) = stats.probplot(z, dist="norm")
    ax.plot(osm, osr, ".", color=BLUE, ms=3)
    ax.plot(osm, sl * osm + ic, color=RED)
    ax.set(xlabel="cuantiles teóricos", ylabel="cuantiles de la muestra")
    titulo(ax, "QQ-plot de los residuos estandarizados", 8.8)
    style(ax)
    ax = axs[1, 2]
    ax.axis("off")
    ks_mal = stats.kstest(z, stats.norm(z.mean(), z.std(ddof=1)).cdf)
    from statsmodels.stats.diagnostic import lilliefors
    lil = lilliefors(z, dist="norm")
    sw = stats.shapiro(z)
    txt = (f"Kolmogorov-Smirnov con μ y σ estimados\n   D = {ks_mal.statistic:.4f}  valor p = {ks_mal.pvalue:.3f}  (p inflado)\n\n"
           f"Lilliefors (corrige la estimación)\n   D = {lil[0]:.4f}  valor p = {lil[1]:.3f}\n\nShapiro-Wilk\n   W = {sw.statistic:.4f}  valor p = {sw.pvalue:.3f}\n\n"
           f"Jarque-Bera\n   valor p = {stats.jarque_bera(z).pvalue:.3f}")
    ax.text(0.02, 0.97, txt, fontsize=8.6, va="top", family="monospace", color=INK)
    titulo(ax, "Normalidad de los residuos", 8.8)
    fig.tight_layout()
    save(fig, "c25_27_diagnostico.png")
    q10, p10v = ljung_box(ar1_res, 10, 1)
    q11, p11v = ljung_box(e, 10, 2)
    q20, p20v = ljung_box(e, 20, 2)
    NUM["diag"] = dict(n=n, q10_arma=float(q11), p10_arma=float(p11v), q20_arma=float(q20), p20_arma=float(p20v), p10_ar1=float(p10v), q10_ar1=float(q10),
                       p10_arma_sin_gl=float(ljung_box(e, 10, 0)[1]), ks_D=float(ks_mal.statistic), ks_p=float(ks_mal.pvalue), lil_D=float(lil[0]), lil_p=float(lil[1]), sw_p=float(sw.pvalue),
                       max_abs_z=float(np.abs(z).max()), sobre2=int((np.abs(z) > 2).sum()), acf_resid_max=float(np.abs(acf_manual(e, 25)[1:]).max()))
    chk = sm.stats.acorr_ljungbox(e, lags=[10], model_df=2)
    NUM["diag"]["q10_statsmodels"] = float(chk["lb_stat"].iloc[0])
    NUM["diag"]["p10_statsmodels"] = float(chk["lb_pvalue"].iloc[0])


# ============================================================================ 12. pronóstico con intervalos y validación temporal
def pronostico_fig():
    y, aj = _ajustes_arbol()
    yr = datos()["tree"]["Year"].values
    m = aj[(1, 1)]["mle"]
    fc = m.get_forecast(10)
    ic95, ic80 = fc.conf_int(alpha=0.05), fc.conf_int(alpha=0.20)
    fc50 = m.get_forecast(50)
    mu = m.params[0]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    h = np.arange(yr[-1] + 1, yr[-1] + 11)
    ax.plot(yr[-40:], y[-40:], color=BLUE, lw=1.2, label="observada")
    ax.plot(h, fc.predicted_mean, color=RED, lw=1.8, label="pronóstico")
    ax.fill_between(h, ic95[:, 0], ic95[:, 1], color=AMBER, alpha=0.25, label="95 %")
    ax.fill_between(h, ic80[:, 0], ic80[:, 1], color=RED, alpha=0.25, label="80 %")
    ax.axhline(mu, color=INK, ls=":", lw=1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.set_ylabel("mm")
    titulo(ax, "ARMA(1,1), h = 10: converge a la media y el intervalo se estabiliza", 8.6)
    style(ax)
    ax = axs[1]
    h50 = np.arange(1, 51)
    ax.plot(h50, fc50.predicted_mean, color=RED, lw=1.8, label="pronóstico")
    ci = fc50.conf_int()
    ax.fill_between(h50, ci[:, 0], ci[:, 1], color=AMBER, alpha=0.25, label="95 %")
    ax.axhline(mu, color=INK, ls=":", lw=1, label=f"media μ = {mu:.2f}")
    ax.set(xlabel="horizonte h", ylabel="mm")
    ax.legend(frameon=False, fontsize=7.5)
    titulo(ax, "Modelo de memoria corta: más allá de ~10 pasos solo dice «la media»", 8.6)
    style(ax)
    # validación temporal: últimos 40 años como prueba
    h_t = 40
    tr, te = y[:-h_t], y[-h_t:]
    res = {}
    for nm, (p, q) in (("AR(1)", (1, 0)), ("AR(2)", (2, 0)), ("MA(3)", (0, 3)), ("ARMA(1,1)", (1, 1))):
        mm_ = ARIMA(tr, order=(p, 0, q), trend="c").fit()
        res[nm] = mae(te, mm_.forecast(h_t))
    res["media del entrenamiento"] = mae(te, np.full(h_t, tr.mean()))
    res["ingenuo (último valor)"] = mae(te, np.full(h_t, tr[-1]))
    ax = axs[2]
    nombres = list(res)
    ax.barh(nombres, [res[k] for k in nombres], color=[GREEN if k.startswith("ARMA") or k.startswith("AR") or k.startswith("MA") else MUTED for k in nombres])
    for i, k in enumerate(nombres):
        ax.text(res[k] + 0.002, i, f"{res[k]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.18, 0.30)
    ax.set_xlabel("MAE en los últimos 40 años (corte temporal)")
    titulo(ax, "Validación temporal: pronosticar 40 años a ciegas", 8.8)
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c25_27_pronostico.png")
    sl = fc.predicted_mean
    NUM["pron"] = dict(h1=float(sl[0]), lo95_h1=float(ic95[0, 0]), hi95_h1=float(ic95[0, 1]), lo80_h1=float(ic80[0, 0]), hi80_h1=float(ic80[0, 1]), h10=float(sl[-1]), lo95_h10=float(ic95[-1, 0]),
                       hi95_h10=float(ic95[-1, 1]), media=float(mu), ancho95_h1=float(ic95[0, 1] - ic95[0, 0]), ancho95_h10=float(ic95[-1, 1] - ic95[-1, 0]),
                       ancho95_h50=float(ci[-1, 1] - ci[-1, 0]), sd_serie=float(y.std(ddof=1)), z95_sd_serie=float(2 * 1.96 * y.std(ddof=1)), mae=res, h_prueba=h_t,
                       ar1_mu=float(aj[(1, 0)]["mle"].params[0]), ar1_phi=float(aj[(1, 0)]["mle"].params[1]), ar1_s=float(np.sqrt(aj[(1, 0)]["mle"].params[2])))



# ============================================================================ 13. modelos estacionales (Clase 27)
def sar_fig():
    rng = np.random.default_rng(2701)
    n = 800
    e = rng.normal(size=n + 200)
    y = np.zeros(n + 200)
    for t in range(12, n + 200):
        y[t] = 0.8 * y[t - 12] + e[t]
    y = y[200:]
    ta = np.zeros(61)
    ta[::12] = 0.8 ** (np.arange(0, 61, 12) // 12)
    r = acf_manual(y, 60)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    ax.plot(y[:60], color=BLUE, lw=1.3)
    ax.set_xlabel("t (primeras 60 observaciones)")
    titulo(ax, "Yₜ = 0.8·Yₜ₋₁₂ + εₜ: el patrón no se ve a simple vista", 8.8)
    style(ax)
    barras_acf(axs[1], r, n, BLUE, "ACF muestral: picos en 12, 24, 36, 48…", ylim=(-0.25, 0.9), lw=2)
    axs[1].plot(np.arange(12, 61, 12), ta[12::12], "o", color=RED, ms=5, label="teórica ρ(12j) = 0.8ʲ")
    axs[1].legend(frameon=False, fontsize=8)
    ax = axs[2]
    pc = pacf_yw(y, 60)
    barras_acf(ax, pc, n, VIOLET, "PACF: un solo pico en el rezago 12 (AR estacional de orden 1)", ylim=(-0.25, 0.9), lw=2)
    fig.tight_layout()
    save(fig, "c25_27_sar.png")
    NUM["sar"] = dict(acf12=float(r[12]), acf24=float(r[24]), acf36=float(r[36]), acf1=float(r[1]), pacf12=float(pc[12]), banda=float(1.96 / np.sqrt(n)), teo=[0.8, 0.64, 0.512])


def _ajustar_sarma(y, order, seas, pulir=True):
    mod = ARIMA(y, order=order, seasonal_order=seas)
    r = mod.fit()
    if pulir:                                # el optimizador por defecto se detiene antes en la dirección plana de la media: se pule con Nelder-Mead
        r = mod.fit(start_params=r.params, method_kwargs={"method": "nm", "maxiter": 3000, "xatol": 1e-8, "fatol": 1e-10})
    return r


def usacc_fig():
    d = datos()["usa"]
    y = ts_mensual(d, "muertes")
    r = _ajustar_sarma(y, (1, 0, 0), (1, 0, 0, 12))
    e = np.asarray(r.resid)
    n = len(y)
    # búsqueda en rejilla (equivalente simplificado a auto.arima: p,q ≤ 2, P,Q ≤ 1, d = D = 0)
    tabla = []
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        m = ARIMA(y, order=(p, 0, q), seasonal_order=(P, 0, Q, 12)).fit()
                        tabla.append(((p, q, P, Q), float(m.aic), float(m.bic)))
                    except Exception:
                        pass
    tabla.sort(key=lambda t: t[1])
    fc = r.get_forecast(24)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
    ax = axs[0, 0]
    ax.plot(y.index, y.values, color=BLUE, lw=1.3)
    ax.set_ylabel("muertes por mes")
    titulo(ax, "Muertes accidentales EE. UU. 1973–78: estacionalidad, sin tendencia", 8.6)
    style(ax)
    barras_acf(axs[0, 1], acf_manual(y.values, 36), n, BLUE, "ACF: picos en 12, 24, 36", ylim=(-0.6, 1.0), lw=2)
    barras_acf(axs[0, 2], pacf_yw(y.values, 36), n, VIOLET, "PACF: lag 1 y lag 12", ylim=(-0.6, 1.0), lw=2)
    ax = axs[1, 0]
    ft = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=24, freq="MS")
    ax.plot(y.index, y.values, color=BLUE, lw=1.3, label="observada")
    ax.plot(ft, fc.predicted_mean, color=RED, lw=1.6, label="pronóstico 24 meses")
    ax.fill_between(ft, ci[:, 0], ci[:, 1], color=AMBER, alpha=0.3, label="intervalo 95 %")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    titulo(ax, "SARMA(1,0)(1,0)₁₂: el patrón anual se proyecta", 8.8)
    style(ax)
    barras_acf(axs[1, 1], acf_manual(e, 24), n, GREEN, "ACF de los residuos", ylim=(-0.5, 0.6), lw=2)
    ax = axs[1, 2]
    ms = np.arange(3, 25)
    ax.plot(ms, [ljung_box(e, m_, ddof=2)[1] for m_ in ms], "o-", color=GREEN, ms=3.5)
    ax.axhline(0.05, color=INK, ls=":")
    ax.set(xlabel="m", ylabel="valor p de Ljung-Box", ylim=(0, 1.02))
    titulo(ax, "Ljung-Box (gl = m − 2): no se rechaza ruido blanco", 8.8)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_usacc.png")
    se = r.bse
    res = np.asarray(r.resid)
    ks = stats.kstest(res / res.std(ddof=1), "norm")
    NUM["usacc"] = dict(n=n, ar1=float(r.params["ar.L1"]), sar1=float(r.params["ar.S.L12"]), media=float(r.params["const"]), sigma2=float(r.params["sigma2"]), llf=float(r.llf), aic=float(r.aic),
                        bic=float(r.bic), se_ar1=float(se["ar.L1"]), se_sar1=float(se["ar.S.L12"]), se_media=float(se["const"]), z_ar1=float(r.params["ar.L1"] / se["ar.L1"]),
                        z_sar1=float(r.params["ar.S.L12"] / se["ar.S.L12"]), mejor_rejilla=[list(tabla[0][0]), tabla[0][1], tabla[0][2]], top5=[[list(t[0]), t[1]] for t in tabla[:5]],
                        rejilla_n=len(tabla), p_lb12=float(ljung_box(e, 12, 2)[1]), p_lb24=float(ljung_box(e, 24, 2)[1]), pron_h1=float(fc.predicted_mean.iloc[0]), pron_h13=float(fc.predicted_mean.iloc[12]),
                        ic_h1=[float(ci[0, 0]), float(ci[0, 1])], ic_h24=[float(ci[-1, 0]), float(ci[-1, 1])], mes_max=int(y.groupby(y.index.month).mean().idxmax()),
                        rmse_in=float(np.sqrt(np.mean(res ** 2))), ks_p=float(ks.pvalue), media_serie=float(y.mean()))
    # el mismo AR(1)x(1)12 con 12 meses de prueba: validación temporal
    tr, te = y.iloc[:-12], y.iloc[-12:]
    r2 = _ajustar_sarma(tr, (1, 0, 0), (1, 0, 0, 12))
    NUM["usacc"].update(mae_sarma_12=mae(te, r2.forecast(12)), mae_ing_est_12=mae(te, ingenuo(tr.values, 12, 12)), mae_media_12=mae(te, np.full(12, tr.mean())))


# ============================================================================ 14. tendencia determinística vs estocástica y diferenciación (Clase 27)
def tendencias_fig():
    rng = np.random.default_rng(2702)
    n = 144
    t = np.arange(n)
    f = pd.date_range("2010-01-01", periods=n, freq="MS")
    det = 2 + 0.2 * t + rng.normal(0, 2, n)
    est = np.cumsum(rng.normal(0.05, 1.8, n))
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.6))
    for j, (y, nm, col) in enumerate(((det, "Tendencia determinística: recta + ruido", BLUE), (est, "Tendencia estocástica: caminata aleatoria", RED))):
        ax = axs[j, 0]
        ax.plot(f, y, color=col, lw=1.3)
        titulo(ax, nm, 8.8)
        style(ax)
        ax = axs[j, 1]
        ax.plot(f[1:], np.diff(y), color=col, lw=1)
        ax.axhline(0, color=INK, lw=0.7)
        titulo(ax, "Serie diferenciada Δyₜ = yₜ − yₜ₋₁", 8.8)
        style(ax)
    ax = axs[0, 2]
    barras_acf(ax, acf_manual(det, 24), n, BLUE, "ACF de la serie determinística (decae lento)", ylim=(-0.3, 1.0), lw=2)
    barras_acf(axs[1, 2], acf_manual(est, 24), n, RED, "ACF de la caminata (decae lento)", ylim=(-0.3, 1.0), lw=2)
    fig.tight_layout()
    save(fig, "c25_27_tendencias.png")
    # sobre-diferenciación del ruido blanco: Var(Δ^d ε) = C(2d, d) σ²
    e = np.random.default_rng(2703).normal(size=200000)
    emp = []
    x = e.copy()
    for dd in range(4):
        emp.append(float(x.var()))
        x = np.diff(x)
    from math import comb
    NUM["tend"] = dict(var_emp=emp, var_teo=[comb(2 * k, k) for k in range(4)], acf1_det_dif=float(acf_manual(np.diff(det), 1)[1]), acf1_est_dif=float(acf_manual(np.diff(est), 1)[1]),
                       acf1_det=float(acf_manual(det, 1)[1]), acf1_est=float(acf_manual(est, 1)[1]), var_det=float(det.var()), var_det_dif=float(np.diff(det).var()),
                       var_est_dif=float(np.diff(est).var()), phi_hat_est=float(np.sum(est[1:] * est[:-1]) / np.sum(est[:-1] ** 2)))


# ============================================================================ 15. raíz unitaria: Dickey-Fuller (Clase 27)
def adf_fig():
    d = datos()
    ipc = d["ipc"]
    indice = 100 * np.cumprod(1 + ipc["var_ipc"].values / 100)      # índice reconstruido a partir de la variación mensual (no es el índice oficial)
    ips = d["ipsa"]["ipsa"].values
    lips = np.log(ips)
    ret = np.diff(lips)
    ci = d["ipc"]
    res = {}
    for nm, serie, reg in (("Índice de precios (reconstruido de ipc_var)", indice, "c"), ("Primera diferencia del índice", np.diff(indice), "c"), ("log IPSA 1980-2000", lips, "c"),
                           ("Retorno log IPSA", ret, "c"), ("Caminata aleatoria simulada", np.cumsum(np.random.default_rng(2704).normal(size=250)), "c"), ("AR(1) φ = 0.5 simulado", simular_arma([0.5], [], 250, np.random.default_rng(2705)), "c")):
        a = adfuller(serie, regression=reg, autolag="AIC")
        res[nm] = dict(stat=float(a[0]), p=float(a[1]), lags=int(a[2]), n=int(a[3]), crit5=float(a[4]["5%"]))
    # distribución del estadístico DF bajo H0 por simulación
    rng = np.random.default_rng(2706)
    tdf = np.array([adf_regresion(np.cumsum(rng.normal(size=250)))[1] for _ in range(3000)])
    gam, tgam = adf_regresion(indice)
    ref = adfuller(indice, regression="c", maxlag=0, autolag=None)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    ax.hist(tdf, bins=50, density=True, color=AMBER, alpha=0.6, edgecolor="white", label="estadístico t bajo H₀ (φ = 1)")
    g = np.linspace(-5, 3, 200)
    ax.plot(g, stats.norm.pdf(g), color=BLUE, lw=1.5, label="N(0, 1) (error común)")
    c5 = float(np.percentile(tdf, 5))
    ax.axvline(c5, color=RED, ls="--")
    ax.text(c5 - 0.05, ax.get_ylim()[1] * 0.85, f"crítico 5 % ≈ {c5:.2f}\n(no −1.65)", ha="right", fontsize=8, color=RED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_xlabel("t de γ̂ en Δyₜ = c + γ·yₜ₋₁ + e")
    titulo(ax, "El test de Dickey-Fuller no usa la t de Student", 8.8)
    style(ax)
    ax = axs[1]
    nombres = list(res)
    ax.barh(range(len(nombres)), [res[k]["p"] for k in nombres], color=[RED if res[k]["p"] > 0.05 else GREEN for k in nombres])
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels([k.replace(" (reconstruido de ipc_var)", "") for k in nombres], fontsize=7.8)
    for i, k in enumerate(nombres):
        ax.text(max(min(res[k]["p"], 0.9), 0.06) + 0.02, i, f"p = {res[k]['p']:.3f}", va="center", fontsize=8)
    ax.axvline(0.05, color=INK, ls=":")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel("valor p del ADF (H₀: raíz unitaria)")
    titulo(ax, "Rojo: no se rechaza raíz unitaria · verde: se rechaza", 8.8)
    style(ax, "x")
    ax = axs[2]
    ax.plot(indice, color=BLUE, lw=1.4, label="índice")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(indice)), np.diff(indice), color=AMBER, lw=0.9, label="Δ índice")
    ax.set_xlabel("mes")
    ax.set_ylabel("índice (base 100)", color=BLUE)
    ax2.set_ylabel("Δ índice", color=AMBER)
    titulo(ax, "Tendencia clara → el test no rechaza; la diferencia sí", 8.8)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_adf.png")
    NUM["adf"] = dict(res=res, crit5_sim=c5, gamma_indice=gam, t_indice=tgam, t_statsmodels=float(ref[0]), p_statsmodels=float(ref[1]), n_sim=3000, indice_ini=float(indice[0]), indice_fin=float(indice[-1]))


# ============================================================================ 16. retornos del IPSA: del precio al retorno, MA con parámetros fijados
def retornos_fig():
    d = datos()
    ips = ts_mensual(d["ipsa"], "ipsa")
    lips = np.log(ips.values)
    ret = np.diff(lips)
    n = len(ret)
    ac = acf_manual(ret, 24)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    ax = axs[0]
    ax.plot(ips.index[1:], ret, color=BLUE, lw=1)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("log(Pₜ) − log(Pₜ₋₁)")
    titulo(ax, "Retorno logarítmico mensual del IPSA", 8.8)
    style(ax)
    barras_acf(axs[1], acf_manual(lips, 24), len(lips), VIOLET, "ACF del log IPSA: φ ≈ 1", ylim=(-0.3, 1.0), lw=2)
    barras_acf(axs[2], ac, n, BLUE, "ACF del retorno", ylim=(-0.3, 1.0), lw=2)
    ax = axs[3]
    ms = np.arange(1, 25)
    ax.plot(ms, [ljung_box(ret, m_)[1] for m_ in ms], "o-", color=RED, ms=3.5)
    ax.axhline(0.05, color=INK, ls=":")
    ax.set(xlabel="m", ylabel="valor p de Ljung-Box", ylim=(0, 1.02))
    titulo(ax, "Ljung-Box del retorno", 8.8)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_retornos.png")
    aprox = float(np.max(np.abs(np.diff(ips.values) / ips.values[:-1] - ret)))
    sig = [int(k) for k in range(1, 13) if abs(ac[k]) > 1.96 / np.sqrt(n)]
    # MA(q) con el rezago significativo: ajuste libre y con parámetros fijados a cero, como hizo el profesor con la acción
    m3 = ARIMA(ret, order=(0, 0, 3), trend="c").fit()
    NUM["retornos"] = dict(n=n, media=float(ret.mean()), sd=float(ret.std(ddof=1)), acf1=float(ac[1]), acf2=float(ac[2]), acf3=float(ac[3]), banda=float(1.96 / np.sqrt(n)), rezagos_signif=sig,
                           p_lb10=float(ljung_box(ret, 10)[1]), p_lb20=float(ljung_box(ret, 20)[1]), dif_aprox_max=aprox, ma3=[float(v) for v in m3.params[1:4]], ma3_p=[float(v) for v in m3.pvalues[1:4]],
                           aic_ruido=float(ARIMA(ret, order=(0, 0, 0), trend="c").fit().aic), aic_ma3=float(m3.aic), ret_max=float(ret.max()), ret_min=float(ret.min()))


# ============================================================================ 17. validación temporal y modelos estacionales de pasajeros (C30/C38)
def validacion_fig():
    d = datos()
    y = ts_mensual(d["pas_todo"], "INTERNATIONAL") / 1e6
    n = len(y)
    h = 24
    tr, te = particion_temporal(y, h)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axs[0]
    rng = np.random.default_rng(2707)
    idx = np.arange(n)
    azar = rng.choice(idx, size=h, replace=False)
    ax.scatter(idx, np.full(n, 3), s=8, color=BLUE)
    ax.scatter(azar, np.full(h, 3), s=22, color=RED, zorder=3)
    ax.scatter(idx[:-h], np.full(n - h, 2), s=8, color=BLUE)
    ax.scatter(idx[-h:], np.full(h, 2), s=22, color=RED, zorder=3)
    for k, o in enumerate(range(n - 4 * 12, n - 12 + 1, 12)):
        ax.scatter(idx[:o], np.full(o, 1 - k * 0.0 - 0.0), s=0)
    for k, o in enumerate((n - 48, n - 36, n - 24)):
        yy = 1.2 - k * 0.35
        ax.scatter(idx[:o], np.full(o, yy), s=7, color=BLUE)
        ax.scatter(idx[o:o + 12], np.full(12, yy), s=18, color=RED, zorder=3)
    ax.set_yticks([3, 2, 1.2, 0.85, 0.5])
    ax.set_yticklabels(["al azar\n(¡con fuga!)", "temporal\n(corte)", "origen móvil 1", "origen móvil 2", "origen móvil 3"], fontsize=7.8)
    ax.set_xlabel("mes (azul = entrenamiento, rojo = prueba)")
    titulo(ax, "Tres formas de separar: solo las dos últimas respetan el tiempo", 8.4)
    style(ax, "x")
    # modelos
    df = pd.DataFrame({"y": y.values, "t": np.arange(n), "mes": y.index.month}, index=y.index)
    ols = sm.OLS.from_formula("y ~ t + C(mes)", df.iloc[:-h]).fit()
    p_ols = ols.predict(df.iloc[-h:])
    sar = ARIMA(tr, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
    p_sar = sar.get_forecast(h)
    modelos = {"ingenuo": ingenuo(tr.values, h, 1), "ingenuo estacional": ingenuo(tr.values, h, 12), "tendencia + dummies (OLS)": p_ols.values, "SARIMA(0,1,1)(0,1,1)₁₂": p_sar.predicted_mean.values}
    mae_m = {k: mae(te.values, v) for k, v in modelos.items()}
    ax = axs[1]
    ax.plot(y.index[-60:], y.values[-60:], color=INK, lw=1.2, label="observada")
    for (k, v), col in zip(modelos.items(), (MUTED, AMBER, VIOLET, RED)):
        ax.plot(te.index, v, color=col, lw=1.3, label=f"{k}  MAE {mae_m[k]:.2f}")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.set_ylabel("millones de pasajeros")
    titulo(ax, "Últimos 24 meses como prueba", 8.8)
    style(ax)
    # origen móvil
    resultados = {k: [] for k in ("ingenuo estacional", "SARIMA(0,1,1)(0,1,1)₁₂", "tendencia + dummies (OLS)")}
    origenes = list(range(n - 60, n - 11, 12))
    for o in origenes:
        trn, tst = y.iloc[:o], y.iloc[o:o + 12]
        resultados["ingenuo estacional"].append(mae(tst.values, ingenuo(trn.values, 12, 12)))
        s_ = ARIMA(trn, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
        resultados["SARIMA(0,1,1)(0,1,1)₁₂"].append(mae(tst.values, s_.forecast(12).values))
        o_ = sm.OLS.from_formula("y ~ t + C(mes)", df.iloc[:o]).fit()
        resultados["tendencia + dummies (OLS)"].append(mae(tst.values, o_.predict(df.iloc[o:o + 12]).values))
    ax = axs[2]
    w = 0.27
    for i, (k, col) in enumerate(zip(resultados, (AMBER, RED, VIOLET))):
        ax.bar(np.arange(len(origenes)) + (i - 1) * w, resultados[k], w, color=col, label=k)
    ax.set_xticks(range(len(origenes)))
    ax.set_xticklabels([str(y.index[o].year) + "-" + f"{y.index[o].month:02d}" for o in origenes], fontsize=7.5)
    ax.set_xlabel("origen del pronóstico (siguientes 12 meses)")
    ax.set_ylabel("MAE (millones)")
    ax.legend(frameon=False, fontsize=7)
    titulo(ax, "Origen móvil: ¿el ranking se mantiene?", 8.8)
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c25_27_validacion.png")
    # fuga: error «interpolando» con vecinos si se separa al azar vs error real de pronosticar
    vec = np.array([abs(y.values[i] - 0.5 * (y.values[i - 1] + y.values[i + 1])) for i in azar if 0 < i < n - 1])
    NUM["valid"] = dict(n=n, h=h, mae=mae_m, mae_origen={k: [float(v) for v in vs] for k, vs in resultados.items()}, origenes=[str(y.index[o].date()) for o in origenes],
                        mae_prom_origen={k: float(np.mean(v)) for k, v in resultados.items()}, fuga_mae_vecinos=float(vec.mean()), ini=str(y.index[0].date()), fin=str(y.index[-1].date()),
                        sarima_params=[float(v) for v in sar.params], sarima_aic=float(sar.aic), ic95_h12=[float(v) for v in p_sar.conf_int(alpha=0.05).iloc[11].values],
                        real_h12=float(te.values[11]), cobertura95=float(np.mean((te.values >= p_sar.conf_int(alpha=0.05).iloc[:, 0].values) & (te.values <= p_sar.conf_int(alpha=0.05).iloc[:, 1].values))))



# ============================================================================ 18. cuaderno de cierre (Clase 28): CityBike, ARIMA frente a SARIMA y SARIMAX
def citybike_serie():
    """Serie diaria de CityBike. El cuaderno de la Clase 28 la lee de una hoja de Google; aquí se guarda una copia local (citybike_serie_diaria.csv)."""
    ruta = DATOS / "citybike_serie_diaria.csv"
    if not ruta.exists():
        import urllib.request
        ruta.write_bytes(urllib.request.urlopen(URL_CITYBIKE, timeout=30).read())
    return pd.read_csv(ruta, parse_dates=["fecha"]).sort_values("fecha").set_index("fecha")


def citybike_fig():
    sd = citybike_serie()
    serie = sd["total_viajes"]
    h = 30
    train, test = particion_temporal(serie, h)
    a1 = ARIMA(train, order=(2, 0, 2)).fit()
    a2 = ARIMA(train, order=(7, 1, 7)).fit()
    s1 = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
    X = sd[["temperatura_promedio_c"]]
    sx = SARIMAX(train, exog=X.iloc[:-h], order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
    fc = {"ARIMA(2,0,2)": a1.forecast(h), "ARIMA(7,1,7)": a2.forecast(h), "SARIMA(1,0,1)(1,0,1)₇": s1.forecast(h), "SARIMAX + temperatura": sx.forecast(h, exog=X.iloc[-h:])}
    base = {"ingenuo estacional (s = 7)": ingenuo(train.values, h, 7), "media de los últimos 7 días": np.full(h, train.iloc[-7:].mean())}
    errores = {k: mae(test, v) for k, v in {**fc, **base}.items()}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axs[0]
    ax.plot(train.index[-90:], train.values[-90:], color=MUTED, lw=1)
    ax.plot(test.index, test.values, color=INK, lw=1.6, label="real")
    for (k, v), col in zip(fc.items(), (AMBER, VIOLET, BLUE, RED)):
        ax.plot(test.index, v, color=col, lw=1.2, label=k)
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
    ax.set_ylabel("viajes por día")
    titulo(ax, "Últimos 30 días como prueba (junio 2026)", 8.8)
    style(ax)
    ax = axs[1]
    nombres = list(errores)
    ax.barh(nombres, [errores[k] for k in nombres], color=[AMBER, VIOLET, BLUE, RED, MUTED, MUTED])
    for i, k in enumerate(nombres):
        ax.text(errores[k] + 0.3, i, f"{errores[k]:.1f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (viajes por día)")
    titulo(ax, "El patrón semanal es lo que baja el error", 8.8)
    style(ax, "x")
    ax = axs[2]
    barras_acf(ax, acf_manual(train.values, 35), len(train), BLUE, "ACF del entrenamiento: picos cada 7 días", ylim=(-0.3, 1.0), lw=2)
    fig.tight_layout()
    save(fig, "c25_27_citybike.png")
    pr = lambda m, nm: float(m.params[nm])
    NUM["citybike"] = dict(n=len(serie), n_train=len(train), ini=str(serie.index[0].date()), fin=str(serie.index[-1].date()), mae=errores,
                           aic=dict(a202=float(a1.aic), a717=float(a2.aic), sarima=float(s1.aic), sarimax=float(sx.aic)), profesor=dict(a202=25.47, a717=25.15, sarima=21.85, sarimax=15.08),
                           sarima_ar1=pr(s1, "ar.L1"), sarima_ars=pr(s1, "ar.S.L7"), sarima_ma1=pr(s1, "ma.L1"), sarima_mas=pr(s1, "ma.S.L7"), sarimax_temp=pr(sx, "temperatura_promedio_c"),
                           media_train=float(train.mean()), temp_corr=float(np.corrcoef(sd["temperatura_promedio_c"], serie)[0, 1]), temp_ini=float(sd["temperatura_promedio_c"].iloc[0]),
                           temp_fin=float(sd["temperatura_promedio_c"].iloc[-1]), v_ini=float(serie.iloc[:30].mean()), v_fin=float(serie.iloc[-30:].mean()))


# ============================================================================ 19. puentes: residuos autocorrelacionados (C17-21) e incertidumbre de pronóstico (C43)
def puentes_fig():
    d = datos()
    df = d["pas"].copy()
    df["t"] = np.arange(1, len(df) + 1)
    df["mes"] = df["fecha"].dt.month
    m = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df).fit()
    mh = sm.OLS.from_formula("INTERNATIONAL ~ t + C(mes)", df).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    # tendencia + AR(1) en los errores (mínimos cuadrados generalizados factibles / modelo de regresión con errores ARMA)
    sx = SARIMAX(df["INTERNATIONAL"].values, exog=sm.add_constant(df[["t"]]).values, order=(1, 0, 0), trend="n").fit(disp=False)
    # bootstrap de residuos del ARMA(1,1) del árbol
    y = d["tree"]["Idaho"].values
    mod = ARIMA(y, order=(1, 0, 1), trend="c")
    r = mod.fit()
    phi, th, mu = r.params[1], r.params[2], r.params[0]
    res = np.asarray(r.resid)
    rng = np.random.default_rng(2808)
    H, B = 10, 2000
    sim = np.zeros((B, H))
    e_last = res[-1]
    for b in range(B):
        yy, ee = y[-1], e_last
        for h in range(H):
            en = rng.choice(res)
            yy = mu + phi * (yy - mu) + en + th * ee
            ee = en
            sim[b, h] = yy
    q = np.percentile(sim, [2.5, 97.5], axis=0)
    ic = np.asarray(r.get_forecast(H).conf_int(alpha=0.05))
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axs[0]
    ax.plot(df["fecha"], m.resid / 1e3, color=VIOLET, lw=1.1)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("residuo (miles)")
    titulo(ax, f"OLS de la Clase 25: residuos con memoria (DW = {sm.stats.durbin_watson(m.resid):.2f})", 8.6)
    style(ax)
    ax = axs[1]
    nombres = ["t: error est. OLS", "t: error est. robusto (HAC)"]
    vals = [m.bse["t"], mh.bse["t"]]
    ax.bar(nombres, vals, color=[AMBER, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v:,.0f}".replace(",", " "), ha="center", fontsize=9)
    ax.set_ylabel("error estándar del coeficiente de tendencia")
    titulo(ax, "Con autocorrelación, el error estándar del OLS se subestima", 8.6)
    style(ax, "y")
    ax = axs[2]
    h = np.arange(1, H + 1)
    ax.fill_between(h, ic[:, 0], ic[:, 1], color=AMBER, alpha=0.3, label="intervalo de predicción (modelo)")
    ax.plot(h, q[0], color=BLUE, lw=1.6, ls="--", label="bootstrap de residuos 95 %")
    ax.plot(h, q[1], color=BLUE, lw=1.6, ls="--")
    ax.plot(h, r.forecast(H), color=RED, lw=1.6, label="pronóstico")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set(xlabel="horizonte h", ylabel="mm")
    titulo(ax, "Dos formas de medir la incertidumbre del pronóstico", 8.6)
    style(ax)
    fig.tight_layout()
    save(fig, "c25_27_puentes.png")
    NUM["puentes"] = dict(bse_ols=float(m.bse["t"]), bse_hac=float(mh.bse["t"]), razon=float(mh.bse["t"] / m.bse["t"]), dw=float(sm.stats.durbin_watson(m.resid)),
                          ar1_resid=float(sx.params[2]) if len(sx.params) > 2 else None, ic_modelo_h10=[float(ic[-1, 0]), float(ic[-1, 1])], ic_boot_h10=[float(q[0, -1]), float(q[1, -1])],
                          ancho_modelo_h10=float(ic[-1, 1] - ic[-1, 0]), ancho_boot_h10=float(q[1, -1] - q[0, -1]), B=B)



# ============================================================================ 20. ejercicio de cierre: IAC (Índice de Actividad del Comercio, INE)
def iac_fig():
    d = datos()["iac"]
    y = ts_mensual(d, "iac")
    h = 12
    train, test = particion_temporal(y, h)
    a = adfuller(train, regression="c", autolag="AIC")
    a_d = adfuller(train.diff(12).dropna(), regression="c", autolag="AIC")
    a_dd = adfuller(train.diff().diff(12).dropna(), regression="c", autolag="AIC")
    tabla = []
    for p in range(2):
        for q in range(2):
            for P in range(2):
                for Q in range(2):
                    try:
                        m = ARIMA(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit()
                        tabla.append(((p, 1, q, P, 1, Q), float(m.aic)))
                    except Exception:
                        pass
    tabla.sort(key=lambda t: t[1])
    mejor = tabla[0][0]
    m = ARIMA(train, order=mejor[:3], seasonal_order=mejor[3:] + (12,)).fit()
    fc = m.get_forecast(h)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    e = np.asarray(m.resid)[13:]
    ing_e = ingenuo(train.values, h, 12)
    ing = ingenuo(train.values, h, 1)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axs[0]
    ax.plot(train.index, train.values, color=BLUE, lw=1.3, label="entrenamiento")
    ax.plot(test.index, test.values, color=INK, lw=1.6, label="prueba (últimos 12 meses)")
    ax.plot(test.index, fc.predicted_mean, color=RED, lw=1.6, label="SARIMA")
    ax.fill_between(test.index, ci[:, 0], ci[:, 1], color=AMBER, alpha=0.25, label="95 %")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    titulo(ax, "IAC 2015–2024: tendencia, estacionalidad anual y pronóstico", 8.8)
    style(ax)
    dd = train.diff().diff(12).dropna()
    barras_acf(axs[1], acf_manual(dd.values, 24), len(dd), BLUE, "ACF de (1−B)(1−B¹²)·IAC", ylim=(-0.6, 0.6), lw=2)
    barras_acf(axs[2], acf_manual(e, 24), len(e), GREEN, "ACF de los residuos del mejor SARIMA", ylim=(-0.6, 0.6), lw=2)
    fig.tight_layout()
    save(fig, "c25_27_iac.png")
    NUM["iac"] = dict(n=len(y), ini=str(y.index[0].date()), fin=str(y.index[-1].date()), n_train=len(train), adf_nivel_p=float(a[1]), adf_d12_p=float(a_d[1]), adf_dd_p=float(a_dd[1]),
                      mejor=list(mejor), aic_mejor=float(tabla[0][1]), top3=[[list(t[0]), t[1]] for t in tabla[:3]], mae_sarima=mae(test.values, fc.predicted_mean.values),
                      mae_ing_est=mae(test.values, ing_e), mae_ing=mae(test.values, ing), rmse_sarima=rmse(test.values, fc.predicted_mean.values), p_lb12=float(ljung_box(e, 12, ddof=int(sum(mejor[:3:2]) + sum(mejor[3::2])))[1]),
                      cobertura=float(np.mean((test.values >= ci[:, 0]) & (test.values <= ci[:, 1]))), ancho_h1=float(ci[0, 1] - ci[0, 0]), ancho_h12=float(ci[-1, 1] - ci[-1, 0]),
                      media_train=float(train.mean()), crec_anual=float((y.iloc[-12:].mean() / y.iloc[-24:-12].mean() - 1) * 100))

PASOS = {k[:-4]: v for k, v in list(globals().items()) if k.endswith("_fig") and callable(v)}
ORDEN = list(PASOS)           # orden de definición

if __name__ == "__main__":
    import sys
    _prev = HERE / "c25_27_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or ORDEN
    for k in ORDEN:
        if k in elegidos:
            PASOS[k]()
            print("hecho:", k, flush=True)
    _prev.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")

