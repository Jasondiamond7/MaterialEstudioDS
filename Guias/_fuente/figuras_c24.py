"""Figuras de la Clase 24 (Estadística bayesiana, parte 2: inferencia posterior, predictiva, Monte Carlo y MCMC).
Fuentes: Clase 24.pdf (diapositivas), Clase24.ipynb (regresión logística bayesiana con rstanarm sobre MASS::Pima.tr, 200 mujeres) de la profesora Maritza Márquez
y la transcripción corregida. La carpeta de la clase se lee sin modificarla; los datos Pima.tr están en Guias/datos/pima_tr.csv.
Stan/rstanarm no se usa: el muestreador es un Metropolis de paseo aleatorio escrito a mano con numpy (las funciones de la sección «NÚCLEO» se copian tal cual al notebook).
Escribe c24_numeros.json (los PDFs y el notebook leen de ahí).
Ejecutar:  python -X utf8 figuras_c24.py   (o con nombres: python -X utf8 figuras_c24.py normal predictiva)."""
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

from figuras import GRID, INK, MUTED, save

warnings.filterwarnings("ignore")
C24 = "#047857"
C24_BG = "#ECFDF5"
BLUE = "#2563EB"
AMBER = "#D97706"
RED = "#DC2626"
TEAL = "#0D9488"
VIOLET = "#7C3AED"
GREEN = "#16A34A"
NUM = {}
HERE = Path(__file__).resolve().parent
PIMA = HERE.parent / "datos" / "pima_tr.csv"
COLS3 = [BLUE, AMBER, RED]


# ============================================================================ NÚCLEO (se copia al notebook)
def normal_normal(m0, s0, sigma, n, ybar):
    """Posterior de mu con varianza conocida: devuelve (media, varianza) ponderando por precisiones."""
    prec = 1 / s0 ** 2 + n / sigma ** 2
    media = (m0 / s0 ** 2 + n * ybar / sigma ** 2) / prec
    return media, 1 / prec


def log_beta_nucleo(a, b):
    """Logaritmo del núcleo de una Beta(a, b): (a - 1) log t + (b - 1) log(1 - t); -inf fuera de (0, 1)."""
    def logp(t):
        if t <= 0 or t >= 1:
            return -np.inf
        return (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    return logp


def metropolis(logp, x0, paso, n, rng):
    """Metropolis de paseo aleatorio en una dimensión: propuesta normal centrada en el valor actual."""
    x, lp = x0, logp(x0)
    cadena = np.empty(n)
    aceptadas = 0
    for t in range(n):
        prop = x + paso * rng.normal()
        lp_prop = logp(prop)
        if np.log(rng.uniform()) < lp_prop - lp:      # acepta con probabilidad min(1, razón)
            x, lp = prop, lp_prop
            aceptadas += 1
        cadena[t] = x
    return cadena, aceptadas / n


def autocorr(x, max_lag=50):
    """Autocorrelación muestral de una cadena (lags 0..max_lag)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / n
    return ac[: max_lag + 1] / ac[0]


def ess_rhat(cadenas):
    """Tamaño efectivo de muestra y R-hat dividido (cada cadena se parte en dos). cadenas: arreglo (m, n)."""
    c = np.asarray(cadenas, float)
    m, n = c.shape
    mitad = n // 2
    c = np.concatenate([c[:, :mitad], c[:, mitad: 2 * mitad]], axis=0)     # 2m cadenas de longitud n/2
    m2, n2 = c.shape
    medias = c.mean(axis=1)
    W = c.var(axis=1, ddof=1).mean()
    Bv = n2 * medias.var(ddof=1)
    var_mas = (n2 - 1) / n2 * W + Bv / n2
    rhat = float(np.sqrt(var_mas / W))
    # autocovarianza promedio de las cadenas
    ac = np.zeros(n2)
    for fila in c:
        z = fila - fila.mean()
        f = np.fft.rfft(z, 2 * n2)
        ac += np.fft.irfft(f * np.conj(f))[:n2] / n2
    ac /= m2
    rho = 1 - (W - ac) / var_mas
    suma, t = 0.0, 0
    previo = np.inf
    while t + 1 < n2:
        par = rho[t] + rho[t + 1]
        if par < 0:
            break
        par = min(par, previo)           # secuencia de pares monótona
        suma += par
        previo = par
        t += 2
    tau = max(-1 + 2 * suma, 1 / np.log10(max(n2 * m2, 10)))
    ess = float(m2 * n2 / tau)
    return ess, rhat


def log_posterior_logistica(beta, X, y, sd_coef=1.0, sd_int=2.5):
    """Log posteriori (salvo constante) de la regresión logística: Bernoulli-logit + priori Normal(0, sd_int²) en el intercepto y Normal(0, sd_coef²) en los coeficientes."""
    eta = X @ beta
    loglik = np.sum(y * eta - np.logaddexp(0, eta))
    logprior = -0.5 * (beta[0] / sd_int) ** 2 - 0.5 * np.sum((beta[1:] / sd_coef) ** 2)
    return loglik + logprior


def metropolis_bloque(logp, x0, n_calent, n_total, rng, escala0=0.1, adapta=(400, 700)):
    """Metropolis por bloques con propuesta normal multivariada. En el calentamiento ajusta la covarianza de la propuesta
    (2.38²/d veces la covarianza de los últimos valores) en las iteraciones indicadas; luego queda fija."""
    d = len(x0)
    cov = (escala0 ** 2) * np.eye(d)
    x = np.array(x0, float)
    lp = logp(x)
    cadena = np.empty((n_total, d))
    aceptadas = np.zeros(n_total, bool)
    for t in range(n_total):
        L = np.linalg.cholesky(cov)
        prop = x + L @ rng.normal(size=d)
        lp_prop = logp(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            x, lp = prop, lp_prop
            aceptadas[t] = True
        cadena[t] = x
        if t + 1 in adapta:
            ventana = cadena[max(0, (t + 1) // 2): t + 1]
            cov = (2.38 ** 2 / d) * np.cov(ventana.T) + 1e-8 * np.eye(d)
    return cadena[n_calent:], float(aceptadas[n_calent:].mean())
# ============================================================================ fin del núcleo


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def titulo(ax, t, fs=9.2):
    ax.set_title(t, loc="left", fontsize=fs, color=INK)


_C = {}


def pima():
    """Datos Pima.tr, estandarización como en el cuaderno de la profesora y ajuste con 4 cadenas (misma configuración: 2000 iteraciones, 1000 de calentamiento)."""
    if _C:
        return _C
    d = pd.read_csv(PIMA)
    d["diabetes"] = (d["type"] == "Yes").astype(int)
    medias = d[["glu", "bmi", "age"]].mean()
    sds = d[["glu", "bmi", "age"]].std(ddof=1)
    Z = (d[["glu", "bmi", "age"]] - medias) / sds
    X = np.column_stack([np.ones(len(d)), Z.to_numpy()])
    y = d["diabetes"].to_numpy()
    logp = lambda b: log_posterior_logistica(b, X, y)
    inicios = np.random.default_rng(24100).normal(0, 1.5, size=(4, 4))
    cadenas, acs = [], []
    for k in range(4):
        c, a = metropolis_bloque(logp, inicios[k], 1000, 2000, np.random.default_rng(24110 + k))
        cadenas.append(c)
        acs.append(a)
    ch = np.array(cadenas)                      # (4, 1000, 4)
    # misma cosa con corrida larga (para la comparación con rstanarm)
    largas = []
    for k in range(4):
        c, a = metropolis_bloque(logp, inicios[k], 5000, 30000, np.random.default_rng(24120 + k))
        largas.append(c)
    larga = np.array(largas)
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    _C.update(d=d, medias=medias, sds=sds, Z=Z, X=X, y=y, logp=logp, inicios=inicios, ch=ch, acs=acs, larga=larga, glm=glm)
    return _C


# ----------------------------------------------------------------------------- 1. Normal–Normal
def normal_fig():
    m0, s0, sigma, n, yb = 60.0, 15.0, 10.0, 25, 66.0
    mn, vn = normal_normal(m0, s0, sigma, n, yb)
    sn = np.sqrt(vn)
    p65 = float(stats.norm.sf(65, mn, sn))
    prec0, precd = 1 / s0 ** 2, n / sigma ** 2
    w0 = prec0 / (prec0 + precd)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    x = np.linspace(40, 85, 600)
    ax.fill_between(x, stats.norm.pdf(x, m0, s0), color=MUTED, alpha=0.25, label=f"priori N({m0:.0f}, {s0:.0f}²)")
    ax.plot(x, stats.norm.pdf(x, yb, sigma / np.sqrt(n)), color=AMBER, lw=2, label=f"datos: ȳ = {yb:.0f}, σ/√n = {sigma / np.sqrt(n):.1f}")
    ax.fill_between(x, stats.norm.pdf(x, mn, sn), color=C24, alpha=0.5, label=f"posteriori N({mn:.2f}, {sn:.2f}²)")
    ax.fill_between(x[x > 65], stats.norm.pdf(x[x > 65], mn, sn), color=RED, alpha=0.35)
    ax.axvline(65, color=RED, ls="--", lw=1)
    ax.set(xlabel="media poblacional μ del puntaje", ylabel="densidad", xlim=(40, 85))
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    titulo(ax, f"Priori × datos = posteriori; P(μ > 65) = {p65:.3f}")
    style(ax)
    ax = axs[1]
    ax.barh([0], [w0], color=MUTED, height=0.5)
    ax.barh([0], [1 - w0], left=[w0], color=AMBER, height=0.5)
    ax.text(0.5, 0.45, f"precisión priori 1/s₀² = {prec0:.4f}", ha="center", fontsize=8.6, color=INK)
    ax.text(0.5, -0.45, f"precisión de los datos n/σ² = {precd:.2f}", ha="center", fontsize=8.6, color=INK)
    ax.text(w0 + 0.01, 0.3, f"↑ priori {w0 * 100:.1f} %", ha="left", va="center", fontsize=8, color=INK)
    ax.text(w0 + (1 - w0) / 2, 0, f"datos {(1 - w0) * 100:.1f} %", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set(xlim=(0, 1), ylim=(-0.9, 0.9), yticks=[], xlabel="peso en la media posteriori")
    ax.annotate(f"m_n = {w0:.3f}·{m0:.0f} + {1 - w0:.3f}·{yb:.0f} = {mn:.2f}", (0.5, 0.78), ha="center", fontsize=9, color=C24, fontweight="bold")
    titulo(ax, "La media posteriori pesa por precisiones")
    style(ax, "x")
    ax = axs[2]
    ns = [1, 5, 25, 100, 400]
    for n_, col in zip([1, 5, 25, 100], [MUTED, BLUE, C24, AMBER]):
        m_, v_ = normal_normal(m0, s0, sigma, n_, yb)
        ax.plot(x, stats.norm.pdf(x, m_, np.sqrt(v_)), color=col, lw=1.8, label=f"n = {n_}: media {m_:.2f}, sd {np.sqrt(v_):.2f}")
    ax.axvline(yb, color=INK, ls=":", lw=1)
    ax.set(xlabel="μ", ylabel="densidad", xlim=(50, 80))
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    titulo(ax, "Más datos: la posteriori se estrecha hacia ȳ")
    style(ax)
    fig.tight_layout()
    save(fig, "c24_normal.png")
    NUM["normal"] = dict(m0=m0, s0=s0, sigma=sigma, n=n, ybar=yb, media=float(mn), var=float(vn), sd=float(sn), p65=p65, w_priori=float(w0), prec_priori=float(prec0), prec_datos=float(precd),
                         ic=[float(v) for v in stats.norm.interval(0.95, mn, sn)], se_datos=sigma / np.sqrt(n),
                         por_n={str(k): [float(normal_normal(m0, s0, sigma, k, yb)[0]), float(np.sqrt(normal_normal(m0, s0, sigma, k, yb)[1]))] for k in ns},
                         pred_sd=float(np.sqrt(sigma ** 2 + vn)), plug_sd=sigma, p_nueva_mayor_65=float(stats.norm.sf(65, mn, np.sqrt(sigma ** 2 + vn))),
                         p_plug_mayor_65=float(stats.norm.sf(65, mn, sigma)))


# ----------------------------------------------------------------------------- 2. predictiva
def predictiva_fig():
    a, b, m = 16, 8, 10
    k = np.arange(m + 1)
    bb = stats.betabinom(m, a, b)
    plug = stats.binom(m, a / (a + b))
    mn, vn = NUM["normal"]["media"], NUM["normal"]["var"]
    sig = NUM["normal"]["sigma"]
    rng = np.random.default_rng(24201)
    S = 20000
    th = stats.beta(a, b).rvs(S, random_state=rng)
    ysim = rng.binomial(m, th)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    ax.bar(k - 0.2, bb.pmf(k), width=0.4, color=C24, label=f"predictiva (beta-binomial): sd {bb.std():.2f}")
    ax.bar(k + 0.2, plug.pmf(k), width=0.4, color=AMBER, alpha=0.85, label=f"plug-in Binomial(10, 0.667): sd {plug.std():.2f}")
    for kk in k[k < 5]:
        ax.axvspan(kk - 0.5, kk + 0.5, color=RED, alpha=0.06)
    ax.set(xlabel="aprobados en la próxima cohorte de 10", ylabel="probabilidad", xticks=k)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    titulo(ax, f"P(menos de 5): predictiva {bb.cdf(4):.3f} · plug-in {plug.cdf(4):.3f}")
    style(ax, "y")
    ax = axs[1]
    x = np.linspace(40, 95, 600)
    ax.plot(x, stats.norm.pdf(x, mn, np.sqrt(vn)), color=C24, lw=2, label=f"posteriori de μ (sd {np.sqrt(vn):.2f})")
    ax.plot(x, stats.norm.pdf(x, mn, sig), color=AMBER, lw=2, ls="--", label=f"plug-in N(m_n, σ²) (sd {sig:.0f})")
    ax.plot(x, stats.norm.pdf(x, mn, np.sqrt(sig ** 2 + vn)), color=VIOLET, lw=2, label=f"predictiva N(m_n, σ² + s_n²) (sd {np.sqrt(sig ** 2 + vn):.2f})")
    ax.set(xlabel="puntaje de un estudiante nuevo", ylabel="densidad")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    titulo(ax, "Normal: la predictiva suma dos varianzas")
    style(ax)
    ax = axs[2]
    j = rng.uniform(-0.28, 0.28, 500)
    ax.scatter(th[:500], ysim[:500] + j, s=7, color=C24, alpha=0.5)
    ax.set(xlabel="θ⁽ˢ⁾ simulado de la posteriori Beta(16, 8)", ylabel="Ỹ⁽ˢ⁾ ~ Binomial(10, θ⁽ˢ⁾)", yticks=k)
    ax.text(0.03, 0.93, f"{S} pares: media de Ỹ = {ysim.mean():.3f} (exacta {bb.mean():.3f})\nP(Ỹ < 5) ≈ {np.mean(ysim < 5):.3f}", transform=ax.transAxes, fontsize=8.2, va="top", bbox=dict(fc="white", ec=GRID))
    titulo(ax, "Se simula θ y luego el dato nuevo")
    style(ax)
    fig.tight_layout()
    save(fig, "c24_predictiva.png")
    NUM["pred"] = dict(a=a, b=b, m=m, media=float(bb.mean()), var=float(bb.var()), sd=float(bb.std()), var_plug=float(plug.var()), sd_plug=float(plug.std()),
                       p_menos5=float(bb.cdf(4)), p_menos5_plug=float(plug.cdf(4)), p_menos5_sim=float(np.mean(ysim < 5)), media_sim=float(ysim.mean()),
                       pmf=[float(v) for v in bb.pmf(k)], pmf_plug=[float(v) for v in plug.pmf(k)], p_7_mas=float(bb.sf(6)), p_7_mas_plug=float(plug.sf(6)), S=S)


# ----------------------------------------------------------------------------- 3. Monte Carlo
def montecarlo_fig():
    post = stats.beta(16, 8)
    exacta = float(post.sf(0.6))
    rng = np.random.default_rng(24301)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    M = 1000
    th = post.rvs(M, random_state=rng)
    est = float(np.mean(th > 0.6))
    ax = axs[0]
    bins = np.linspace(0.3, 0.95, 40)
    ax.hist(th[th <= 0.6], bins=bins, color=MUTED, alpha=0.7, label="θ⁽ᵐ⁾ ≤ 0.6 (indicador 0)")
    ax.hist(th[th > 0.6], bins=bins, color=C24, alpha=0.8, label=f"θ⁽ᵐ⁾ > 0.6 (indicador 1): {int((th > 0.6).sum())} de {M}")
    ax.axvline(0.6, color=RED, ls="--")
    ax.set(xlabel="θ", ylabel="N.º de simulaciones")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    titulo(ax, f"P(θ > 0.6 | y) ≈ {est:.3f}  (exacta {exacta:.4f})")
    style(ax, "y")
    ax = axs[1]
    Ms = np.array([10, 30, 100, 300, 1000, 3000, 10000, 100000])
    err = []
    for M_ in Ms:
        e = [abs(np.mean(post.rvs(M_, random_state=rng) > 0.6) - exacta) for _ in range(200 if M_ <= 10000 else 40)]
        err.append(np.mean(e))
    ax.loglog(Ms, err, "o-", color=C24, label="error absoluto medio (simulado)")
    ax.loglog(Ms, np.sqrt(2 / np.pi) * np.sqrt(exacta * (1 - exacta) / Ms), "--", color=AMBER, label="teoría 0.80·√(p(1 − p)/M)")
    ax.set(xlabel="M (n.º de simulaciones)", ylabel="error de Monte Carlo")
    ax.legend(frameon=False, fontsize=7.8)
    titulo(ax, "El error baja como 1/√M: 100 veces más para dividir por 10")
    style(ax)
    ax = axs[2]
    g = {"media": th, "θ²": th ** 2, "P(θ > 0.8)": (th > 0.8).astype(float)}
    thL = post.rvs(200000, random_state=rng)
    ex = [post.mean(), post.var() + post.mean() ** 2, post.sf(0.8)]
    est_ = [th.mean(), (th ** 2).mean(), (th > 0.8).mean()]
    xs = np.arange(3)
    ax.bar(xs - 0.2, ex, width=0.4, color=C24, label="exacto")
    ax.bar(xs + 0.2, est_, width=0.4, color=AMBER, label=f"Monte Carlo, M = {M}")
    for i, (e1, e2) in enumerate(zip(ex, est_)):
        ax.text(i + 0.2, e2 + 0.02, f"{e2:.3f}", ha="center", fontsize=8)
        ax.text(i - 0.2, e1 + 0.02, f"{e1:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["E[θ | y]", "E[θ² | y]", "P(θ > 0.8 | y)"])
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=8)
    titulo(ax, "Cualquier integral se reemplaza por un promedio")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c24_montecarlo.png")
    se = float(np.sqrt(exacta * (1 - exacta) / M))
    NUM["mc"] = dict(exacta=exacta, estimado=est, M=M, se=se, err_Ms=[int(v) for v in Ms], err=[float(v) for v in err], cuenta=int((th > 0.6).sum()),
                     exacto_media=float(post.mean()), exacto_cuad=float(post.var() + post.mean() ** 2), exacto_08=float(post.sf(0.8)), estimados=[float(v) for v in est_],
                     razon_100=float(err[4] / err[6]))


# ----------------------------------------------------------------------------- 4. por qué MCMC
def porque_fig():
    c = pima()
    glm = c["glm"]
    X, y = c["X"], c["y"]
    b_hat = c["larga"].reshape(-1, 4).mean(0)
    g0 = np.linspace(-1.5, -0.35, 120)
    g1 = np.linspace(0.3, 1.7, 120)
    LP = np.array([[c["logp"](np.array([u, v, b_hat[2], b_hat[3]])) for u in g0] for v in g1])
    LP -= LP.max()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    ax.contourf(g0, g1, np.exp(LP), levels=12, cmap="Greens")
    sub = c["larga"][:, ::60, :].reshape(-1, 4)
    ax.plot(sub[:600, 0], sub[:600, 1], ".", ms=2.5, color=INK, alpha=0.35)
    ax.set(xlabel="β₀ (intercepto)", ylabel="β_glu (glucosa estandarizada)")
    titulo(ax, "Posteriori de la logística: sin fórmula de nombre")
    style(ax)
    ax = axs[1]
    d = np.arange(1, 8)
    ax.semilogy(d, 100.0 ** d, "o-", color=RED)
    ax.axvline(4, color=INK, ls=":")
    ax.text(4.1, 3e2, "Pima: 4 coeficientes\n10⁸ evaluaciones", fontsize=8.4)
    ax.set(xlabel="n.º de parámetros d", ylabel="puntos de una grilla de 100 por eje", xticks=d)
    titulo(ax, "La grilla se vuelve imposible: 100^d")
    style(ax)
    ax = axs[2]
    ax.set(xlim=(-0.03, 1.03), ylim=(0, 1))
    ax.axis("off")
    filas = [("Beta–Binomial, Normal–Normal", "posteriori conocida\n→ fórmulas (C23, hoy §1)", C24),
             ("sé simular de la posteriori", "Monte Carlo ordinario\nmuestras independientes", BLUE),
             ("solo conozco p(y|θ)·p(θ)\n(sin constante)", "MCMC: cadena dependiente\n→ revisar convergencia", AMBER)]
    for i, (izq, der, col) in enumerate(filas):
        yy = 0.78 - i * 0.33
        ax.add_patch(FancyBboxPatch((0.0, yy - 0.12), 0.44, 0.24, boxstyle="round,pad=0.01", fc="white", ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((0.56, yy - 0.12), 0.44, 0.24, boxstyle="round,pad=0.01", fc=col, ec=col, alpha=0.18))
        ax.text(0.22, yy, izq, ha="center", va="center", fontsize=8.2, color=INK)
        ax.text(0.78, yy, der, ha="center", va="center", fontsize=8.2, color=INK)
        ax.annotate("", (0.55, yy), (0.45, yy), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    titulo(ax, "¿Qué herramienta usar?")
    fig.tight_layout()
    save(fig, "c24_porque_mcmc.png")
    NUM["porque"] = dict(d4=1e8)


# ----------------------------------------------------------------------------- esquema de la clase
def esquema_fig():
    fig, ax = plt.subplots(figsize=(13, 3.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    pasos = [("1 · Posteriori", "p(θ | y) ∝ p(y | θ) p(θ)\nresumen: media, intervalo creíble,\nP(θ > c | y)", C24),
             ("2 · Predictiva", "p(ỹ | y) = ∫ p(ỹ | θ) p(θ | y) dθ\npromedia sobre la\nincertidumbre de θ", BLUE),
             ("3 · Monte Carlo", "integral → promedio de M\nmuestras independientes\nerror ∝ 1/√M", TEAL),
             ("4 · MCMC", "cadena de Markov cuya\ndistribución de equilibrio\nes la posteriori", AMBER),
             ("5 · Diagnóstico", "traza · burn-in · mezcla\nautocorrelación · ESS\nvarias cadenas · R-hat", RED)]
    for i, (t, txt, col) in enumerate(pasos):
        x0 = 0.1 + i * 2.6
        ax.add_patch(FancyBboxPatch((x0, 0.55), 2.35, 2.0, boxstyle="round,pad=0.03", fc="white", ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((x0, 2.0), 2.35, 0.55, boxstyle="round,pad=0.03", fc=col, ec=col))
        ax.text(x0 + 1.175, 2.28, t, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(x0 + 1.175, 1.25, txt, ha="center", va="center", fontsize=8.4, color=INK)
        if i < 4:
            ax.annotate("", (x0 + 2.58, 1.55), (x0 + 2.37, 1.55), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.text(6.5, 0.18, "Aplicación: regresión logística bayesiana con los datos Pima (diabetes ~ glucosa + IMC + edad), muestreada con Metropolis", ha="center", fontsize=9, color=C24, fontweight="bold")
    save(fig, "c24_esquema.png")


# ----------------------------------------------------------------------------- 5. Metropolis paso a paso
def metropolis_fig():
    logp = log_beta_nucleo(16, 8)
    ker = lambda t: np.exp(logp(t)) if 0 < t < 1 else 0.0
    rng = np.random.default_rng(24500)
    paso = 0.10
    x, pasos = 0.55, []
    for _ in range(6):
        prop = x + paso * rng.normal()
        u = rng.uniform()
        r = ker(prop) / ker(x)
        alfa = min(1.0, r)
        ok = u < alfa
        pasos.append(dict(actual=float(x), propuesta=float(prop), razon=float(r), alfa=float(alfa), u=float(u), acepta=bool(ok)))
        if ok:
            x = prop
    t = np.linspace(0.001, 0.999, 500)
    kk = np.array([ker(v) for v in t])
    kk = kk / kk.max()
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.6))
    for i, (ax, p) in enumerate(zip(axs.ravel(), pasos)):
        ax.plot(t, kk, color=C24, lw=2)
        ax.fill_between(t, kk, color=C24, alpha=0.12)
        ax.plot([p["actual"]], [ker(p["actual"]) / ker(0.68)], "o", ms=11, color=BLUE, label="valor actual", zorder=5)
        if 0 < p["propuesta"] < 1:
            ax.plot([p["propuesta"]], [ker(p["propuesta"]) / ker(0.68)], "s", ms=10, color=AMBER, label="propuesta", zorder=5)
        ax.axvline(p["actual"], color=BLUE, ls=":", lw=1)
        ax.axvline(p["propuesta"], color=AMBER, ls=":", lw=1)
        col = GREEN if p["acepta"] else RED
        ax.text(0.03, 0.95, f"paso {i + 1}: razón = p(θ*)/p(θ) = {p['razon']:.2f}\nα = min(1, razón) = {p['alfa']:.2f} · u = {p['u']:.2f}\n{'u < α → ACEPTA: la cadena se mueve' if p['acepta'] else 'u ≥ α → RECHAZA: se queda'}",
                transform=ax.transAxes, fontsize=7.9, va="top", color=col, bbox=dict(fc="white", ec=col, lw=1))
        ax.set(xlim=(0.2, 1.0), ylim=(0, 1.35), xlabel="θ", yticks=[])
        if i == 0:
            ax.legend(frameon=False, fontsize=7.6, loc="center left", bbox_to_anchor=(0.0, 0.62))
        style(ax, "x")
    fig.tight_layout()
    save(fig, "c24_metropolis.png")
    NUM["pasos"] = pasos
    NUM["pasos_paso"] = paso


# ----------------------------------------------------------------------------- 6. traza según tamaño de paso
def traza_fig():
    logp = log_beta_nucleo(16, 8)
    post = stats.beta(16, 8)
    pasos = [0.005, 0.12, 1.5]
    nombres = ["paso muy pequeño", "paso razonable", "paso muy grande"]
    n = 2000
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(3, 4)
    res = []
    for i, (st, nom) in enumerate(zip(pasos, nombres)):
        ch, ac = metropolis(logp, 0.5, st, n, np.random.default_rng(24601 + i))
        e, _ = ess_rhat(ch[None, :])
        ax = fig.add_subplot(gs[i, :3])
        ax.plot(ch, color=COLS3[i], lw=0.8)
        ax.axhline(post.mean(), color=INK, ls=":", lw=1)
        ax.set(ylim=(0.2, 1.0), ylabel="θ", xlabel="iteración" if i == 2 else "")
        titulo(ax, f"{nom} (σ_prop = {st}): aceptación {ac * 100:.0f} % · ESS {e:.0f} de {n}")
        style(ax)
        ax = fig.add_subplot(gs[i, 3])
        ax.hist(ch, bins=np.linspace(0.2, 1, 40), density=True, color=COLS3[i], alpha=0.6)
        tt = np.linspace(0.2, 0.99, 200)
        ax.plot(tt, post.pdf(tt), color=INK, lw=1.4)
        ax.set(xlabel="θ" if i == 2 else "", yticks=[])
        titulo(ax, "frente a la Beta(16, 8) exacta", fs=8.2)
        style(ax, "x")
        res.append(dict(paso=st, aceptacion=float(ac), ess=float(e), media=float(ch.mean()), media_exacta=float(post.mean()), sd=float(ch.std())))
    fig.tight_layout()
    save(fig, "c24_traza.png")
    NUM["traza"] = res


# ----------------------------------------------------------------------------- 7. burn-in
def burnin_fig():
    logp = log_beta_nucleo(16, 8)
    post = stats.beta(16, 8)
    n, quemar = 1500, 150
    ch, ac = metropolis(logp, 0.03, 0.05, n, np.random.default_rng(24701))
    ch2, _ = metropolis(logp, 0.97, 0.05, n, np.random.default_rng(24702))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    ax.plot(ch[:400], color=BLUE, lw=0.9, label="empieza en 0.03")
    ax.plot(ch2[:400], color=AMBER, lw=0.9, label="empieza en 0.97")
    ax.axvspan(0, quemar, color=RED, alpha=0.12)
    ax.text(quemar / 2, 0.08, "burn-in", ha="center", color=RED, fontsize=9)
    ax.axhline(post.mean(), color=INK, ls=":")
    ax.set(xlabel="iteración", ylabel="θ")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    titulo(ax, "Dos cadenas que parten lejos convergen a la misma zona")
    style(ax)
    ax = axs[1]
    tt = np.linspace(0, 1, 200)
    ax.hist(ch, bins=np.linspace(0, 1, 50), density=True, color=RED, alpha=0.45, label="con las iteraciones iniciales")
    ax.hist(ch[quemar:], bins=np.linspace(0, 1, 50), density=True, color=C24, alpha=0.55, label="sin burn-in")
    ax.plot(tt, post.pdf(tt), color=INK, lw=1.4)
    ax.set(xlabel="θ", yticks=[])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    titulo(ax, "Las iteraciones iniciales ensucian el histograma")
    style(ax, "x")
    ax = axs[2]
    nb = [0, 25, 50, 100, 150, 250, 400]
    R, L = 400, 700
    sesgo = np.zeros(len(nb))
    rng = np.random.default_rng(24703)
    for _ in range(R):
        c_, _a = metropolis(logp, 0.03, 0.05, L, rng)
        for i, b in enumerate(nb):
            sesgo[i] += c_[b:].mean() - post.mean()
    sesgo /= R
    ax.bar([str(b) for b in nb], sesgo, color=[RED if b < quemar else C24 for b in nb])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xlabel="iteraciones descartadas", ylabel="sesgo medio de la media (400 cadenas)")
    titulo(ax, "Descartar el inicio reduce el sesgo")
    style(ax, "y")
    fig.tight_layout()
    save(fig, "c24_burnin.png")
    NUM["burnin"] = dict(quemar=quemar, n=n, media_con=float(ch.mean()), media_sin=float(ch[quemar:].mean()), exacta=float(post.mean()), sesgo=dict(zip([str(b) for b in nb], [float(e) for e in sesgo])), R=R, L=L,
                         aceptacion=float(ac))


# ----------------------------------------------------------------------------- 8. autocorrelación y ESS
def acf_fig():
    logp = log_beta_nucleo(16, 8)
    post = stats.beta(16, 8)
    n = 5000
    pasos = [0.005, 0.12, 1.5]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    res = []
    cad = []
    for i, st in enumerate(pasos):
        ch, ac = metropolis(logp, 0.68, st, n, np.random.default_rng(24801 + i))
        cad.append(ch)
        ax.plot(autocorr(ch, 60), color=COLS3[i], lw=1.8, label=f"σ_prop = {st}")
        e, _ = ess_rhat(ch[None, :])
        res.append(dict(paso=st, ess=float(e), aceptacion=float(ac), ac1=float(autocorr(ch, 1)[1]), ac10=float(autocorr(ch, 10)[10]), ac50=float(autocorr(ch, 50)[50])))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xlabel="rezago (lag)", ylabel="autocorrelación")
    ax.legend(frameon=False, fontsize=8)
    titulo(ax, "Autocorrelación de la cadena")
    style(ax)
    ax = axs[1]
    ax.bar([f"σ_prop\n{r['paso']}" for r in res], [r["ess"] for r in res], color=COLS3)
    ax.axhline(n, color=INK, ls="--", lw=1)
    ax.text(2.45, n * 1.02, f"{n} iteraciones", ha="right", fontsize=8)
    for i, r in enumerate(res):
        ax.text(i, r["ess"] + 80, f"{r['ess']:.0f}\n({r['ess'] / n * 100:.0f} %)", ha="center", fontsize=8.4)
    ax.set(ylabel="tamaño efectivo de muestra (ESS)", ylim=(0, n * 1.12))
    titulo(ax, "Misma longitud, información muy distinta")
    style(ax, "y")
    ax = axs[2]
    ch = cad[1]
    ths = [1, 10, 50, 100]
    esss = []
    for t_ in ths:
        sub = ch[::t_]
        e, _ = ess_rhat(sub[None, :])
        esss.append((t_, len(sub), float(e)))
    ax.plot(autocorr(ch, 60), color=COLS3[1], lw=1.6, label="sin adelgazar (σ_prop = 0.12)")
    ax.plot(np.arange(0, 61, 10), autocorr(ch[::10], 6), "o-", color=VIOLET, lw=1.6, label="1 de cada 10 (thin = 10)")
    ax.set(xlabel="rezago medido en iteraciones originales", ylabel="autocorrelación")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.97, 0.7, "\n".join(f"thin = {t:>3}: {k:>4} valores, ESS = {e:.0f}" for t, k, e in esss), transform=ax.transAxes, fontsize=8, family="monospace", va="top", ha="right", bbox=dict(fc="white", ec=GRID))
    titulo(ax, "Adelgazar (thinning) no crea información")
    style(ax)
    fig.tight_layout()
    save(fig, "c24_autocorr.png")
    NUM["acf"] = dict(res=res, thin=[dict(thin=t, n=k, ess=e) for t, k, e in esss], n=n)


# ----------------------------------------------------------------------------- 9. varias cadenas y R-hat
def rhat_fig():
    logp = log_beta_nucleo(16, 8)
    post = stats.beta(16, 8)
    starts = [0.05, 0.3, 0.7, 0.95]
    n = 3000
    cad = np.array([metropolis(logp, s, 0.05, n, np.random.default_rng(24901 + i))[0] for i, s in enumerate(starts)])
    cols = [BLUE, AMBER, GREEN, RED]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, hasta, tit in ((axs[0], 120, "Muy pocas iteraciones: las cadenas aún no se mezclan"), (axs[1], 3000, "Después de muchas: se superponen")):
        for k in range(4):
            ax.plot(cad[k, :hasta], color=cols[k], lw=0.7)
        ax.axhline(post.mean(), color=INK, ls=":")
        e, r = ess_rhat(cad[:, :hasta])
        ax.set(xlabel="iteración", ylabel="θ")
        ax.text(0.97, 0.05, f"R-hat = {r:.3f}\nESS = {e:.0f}", transform=ax.transAxes, ha="right", fontsize=9, bbox=dict(fc="white", ec=GRID), color=GREEN if r < 1.01 else RED)
        titulo(ax, tit, fs=8.8)
        style(ax)
    ax = axs[2]
    largos = np.array([20, 40, 80, 160, 320, 640, 1280, 3000])
    rh = [ess_rhat(cad[:, :L])[1] for L in largos]
    ax.semilogx(largos, rh, "o-", color=C24)
    ax.axhline(1.01, color=RED, ls="--", lw=1)
    ax.text(21, 1.12, "umbral 1.01", color=RED, fontsize=8)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set(xlabel="iteraciones por cadena", ylabel="R-hat dividido")
    titulo(ax, "R-hat baja hacia 1 cuando las cadenas convergen")
    style(ax)
    fig.tight_layout()
    save(fig, "c24_rhat.png")
    NUM["rhat"] = dict(largos=[int(v) for v in largos], rhat=[float(v) for v in rh], rhat_120=float(ess_rhat(cad[:, :120])[1]), ess_120=float(ess_rhat(cad[:, :120])[0]),
                       rhat_3000=float(ess_rhat(cad)[1]), ess_3000=float(ess_rhat(cad)[0]), starts=starts)


# ----------------------------------------------------------------------------- 10-12. regresión logística bayesiana (Pima)
NOMBRES = ["(Intercept)", "glu_z", "bmi_z", "age_z"]


def pima_fig():
    c = pima()
    d, ch, larga, glm = c["d"], c["ch"], c["larga"], c["glm"]
    muestras = ch.reshape(-1, 4)
    ml = larga.reshape(-1, 4)
    # --- resúmenes
    res = {}
    for j, nm in enumerate(NOMBRES):
        el, rl = ess_rhat(larga[:, :, j])
        x = muestras[:, j]
        res[nm] = dict(media=float(x.mean()), mediana=float(np.median(x)), sd=float(x.std(ddof=1)), mad_sd=float(1.4826 * np.median(np.abs(x - np.median(x)))),
                       q025=float(np.quantile(x, 0.025)), q975=float(np.quantile(x, 0.975)), p_pos=float(np.mean(x > 0)),
                       or_med=float(np.exp(np.median(x))), or_lo=float(np.exp(np.quantile(x, 0.025))), or_hi=float(np.exp(np.quantile(x, 0.975))),
                       ess=float(ess_rhat(ch[:, :, j])[0]), rhat=float(ess_rhat(ch[:, :, j])[1]),
                       larga_ess=float(el), larga_rhat=float(rl), larga_media=float(ml[:, j].mean()), larga_mediana=float(np.median(ml[:, j])), larga_sd=float(ml[:, j].std(ddof=1)),
                       larga_q025=float(np.quantile(ml[:, j], 0.025)), larga_q975=float(np.quantile(ml[:, j], 0.975)),
                       mle=float(glm.params[j]), mle_se=float(glm.bse[j]), mle_p=float(glm.pvalues[j]), mle_lo=float(glm.conf_int()[j][0]), mle_hi=float(glm.conf_int()[j][1]))
    # --- figura de coeficientes
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    g0 = d.loc[d.diabetes == 0, "glu"]
    g1 = d.loc[d.diabetes == 1, "glu"]
    bp = ax.boxplot([g0, g1], tick_labels=["No (132)", "Sí (68)"], patch_artist=True, widths=0.55)
    for patch, col in zip(bp["boxes"], [BLUE, RED]):
        patch.set(facecolor=col, alpha=0.5)
    ax.set(ylabel="glucosa plasmática (glu)", xlabel="diagnóstico de diabetes")
    titulo(ax, f"Medianas de glucosa: {g0.median():.0f} frente a {g1.median():.0f}")
    style(ax, "y")
    ax = axs[1]
    xs = np.linspace(-2.2, 2.4, 400)
    for j, (nm, col) in enumerate(zip(NOMBRES[1:], [BLUE, AMBER, TEAL]), start=1):
        dens = stats.gaussian_kde(ml[:, j])
        ax.plot(xs, dens(xs), color=col, lw=2, label=nm)
    ax.plot(xs, stats.norm.pdf(xs, 0, 1), color=MUTED, ls="--", lw=1.4, label="priori N(0, 1)")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="coeficiente (escala log-odds)", ylabel="densidad")
    ax.legend(frameon=False, fontsize=8)
    titulo(ax, "Posterioris de los tres coeficientes")
    style(ax)
    ax = axs[2]
    for i, nm in enumerate(NOMBRES[1:]):
        r = res[nm]
        yy = 2 - i
        ax.plot([r["q025"], r["q975"]], [yy + 0.12, yy + 0.12], color=C24, lw=3, solid_capstyle="round")
        ax.plot(r["mediana"], yy + 0.12, "o", color=C24, ms=6)
        ax.plot([r["mle_lo"], r["mle_hi"]], [yy - 0.12, yy - 0.12], color=AMBER, lw=3, solid_capstyle="round")
        ax.plot(r["mle"], yy - 0.12, "s", color=AMBER, ms=6)
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.set(yticks=[2, 1, 0], yticklabels=NOMBRES[1:], xlabel="coeficiente (log-odds)")
    ax.plot([], [], color=C24, lw=3, label="bayesiano: mediana e IC creíble 95 %")
    ax.plot([], [], color=AMBER, lw=3, label="clásico (glm): estimación e IC Wald 95 %")
    ax.legend(frameon=False, fontsize=7.6, loc="center", bbox_to_anchor=(0.62, 0.5))
    titulo(ax, "Casi lo mismo, con otra lectura")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c24_regresion.png")
    # --- trazas
    fig, axs = plt.subplots(2, 2, figsize=(13, 4.8))
    cols = [BLUE, AMBER, GREEN, RED]
    for j, (ax, nm) in enumerate(zip(axs.ravel(), NOMBRES)):
        for k in range(4):
            ax.plot(ch[k, :, j], color=cols[k], lw=0.6, alpha=0.9)
        ax.set(ylabel=nm)
        ax.text(0.99, 0.04, f"R-hat {res[nm]['rhat']:.3f} · ESS {res[nm]['ess']:.0f}", transform=ax.transAxes, ha="right", fontsize=8.4, bbox=dict(fc="white", ec=GRID))
        style(ax)
    axs[1, 0].set_xlabel("iteración (después del calentamiento)")
    axs[1, 1].set_xlabel("iteración (después del calentamiento)")
    fig.tight_layout()
    save(fig, "c24_trazas_pima.png")
    NUM["pima"] = dict(res=res, n=int(len(d)), n_si=int(d.diabetes.sum()), n_no=int((d.diabetes == 0).sum()), prop_si=float(d.diabetes.mean()),
                       medias={k: float(v) for k, v in c["medias"].items()}, sds={k: float(v) for k, v in c["sds"].items()},
                       aceptacion=[float(a) for a in c["acs"]], aceptacion_media=float(np.mean(c["acs"])), med_glu=[float(g0.median()), float(g1.median())],
                       rango_z={k: [float(c["Z"][k].min()), float(c["Z"][k].max())] for k in c["Z"].columns}, desv_z={k: float(c["Z"][k].std(ddof=1)) for k in c["Z"].columns},
                       prof=dict(media_glu_z_min=float(c["Z"]["glu"].min()), media_glu_z_max=float(c["Z"]["glu"].max())),
                       devianza=float(-2 * glm.llf), null_devianza=float(256.41), aic=float(glm.aic), n_larga=int(ml.shape[0]))


def pred_fig():
    c = pima()
    d, ch, larga, glm, X, y = c["d"], c["ch"], c["larga"], c["glm"], c["X"], c["y"]
    muestras = ch.reshape(-1, 4)
    med, sd = c["medias"], c["sds"]
    nuevos = pd.DataFrame({"perfil": ["Perfil A", "Perfil B"], "glu": [100, 160], "bmi": [25, 35], "age": [30, 55]})
    Zn = np.column_stack([np.ones(2)] + [(nuevos[k] - med[k]) / sd[k] for k in ("glu", "bmi", "age")])
    pr = 1 / (1 + np.exp(-(muestras @ Zn.T)))                   # (4000, 2)
    resum = []
    for j in range(2):
        resum.append(dict(perfil=nuevos.perfil[j], media=float(pr[:, j].mean()), mediana=float(np.median(pr[:, j])), lo=float(np.quantile(pr[:, j], 0.025)), hi=float(np.quantile(pr[:, j], 0.975)),
                          mle=float(1 / (1 + np.exp(-(glm.params @ Zn[j]))))))
    # exactitud aparente
    p_i = 1 / (1 + np.exp(-(muestras @ X.T)))
    pm = p_i.mean(0)
    pred = (pm >= 0.5).astype(int)
    conf = [[int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 0) & (pred == 1)))], [int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 1) & (pred == 1)))]]
    acc = float(np.mean(pred == y))
    # pp_check: nº de diabéticos replicados
    rng = np.random.default_rng(24951)
    idx = rng.choice(muestras.shape[0], 1000, replace=False)
    rep = np.array([rng.binomial(1, p_i[i]).sum() for i in idx])
    # bootstrap clásico vs posteriori para glu_z
    rng = np.random.default_rng(24952)
    B = 2000
    bs = np.empty(B)
    for b in range(B):
        i = rng.choice(len(y), len(y), replace=True)
        try:
            bs[b] = sm.GLM(y[i], X[i], family=sm.families.Binomial()).fit().params[1]
        except Exception:
            bs[b] = np.nan
    bs = bs[~np.isnan(bs)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    for j, col in zip(range(2), [BLUE, RED]):
        ax.hist(pr[:, j], bins=40, color=col, alpha=0.6, density=True, label=f"{resum[j]['perfil']}: {resum[j]['media']:.3f} [{resum[j]['lo']:.3f}; {resum[j]['hi']:.3f}]")
    ax.set(xlabel="probabilidad de diabetes", yticks=[], xlim=(0, 1))
    ax.legend(frameon=False, fontsize=7.8, loc="upper center")
    titulo(ax, "Probabilidad posterior para dos perfiles")
    style(ax, "x")
    ax = axs[1]
    vals, cnt = np.unique(rep, return_counts=True)
    ax.bar(vals, cnt / cnt.sum(), color=C24, alpha=0.7, width=0.9)
    ax.axvline(y.sum(), color=RED, lw=2)
    ax.text(y.sum() + 1.2, ax.get_ylim()[1] * 0.9, f"observado: {int(y.sum())}", color=RED, fontsize=8.6)
    ax.set(xlabel="n.º de diabéticos en 200 mujeres simuladas", ylabel="frecuencia relativa")
    titulo(ax, "Chequeo predictivo posterior (pp_check)")
    style(ax, "y")
    ax = axs[2]
    j = 1
    bins = np.linspace(0.2, 1.9, 45)
    ax.hist(bs, bins=bins, density=True, color=AMBER, alpha=0.55, label=f"bootstrap del glm (B = {len(bs)})")
    ax.hist(muestras[:, j], bins=bins, density=True, color=C24, alpha=0.55, label="posteriori (MCMC)")
    ax.set(xlabel="coeficiente de glu_z", yticks=[])
    ax.legend(frameon=False, fontsize=8)
    titulo(ax, "Dos formas de medir la incertidumbre")
    style(ax, "x")
    fig.tight_layout()
    save(fig, "c24_predicciones.png")
    NUM["pred_pima"] = dict(perfiles=resum, confusion=conf, exactitud=acc, rep_media=float(rep.mean()), rep_lo=float(np.quantile(rep, 0.025)), rep_hi=float(np.quantile(rep, 0.975)),
                            obs=int(y.sum()), boot=dict(media=float(bs.mean()), sd=float(bs.std(ddof=1)), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), B=int(len(bs))),
                            post=dict(sd=float(muestras[:, 1].std(ddof=1)), lo=float(np.quantile(muestras[:, 1], 0.025)), hi=float(np.quantile(muestras[:, 1], 0.975))))


# ----------------------------------------------------------------------------- comprobaciones numéricas y alertas
def verificaciones():
    """Verifica en código afirmaciones de las diapositivas/clase (alertas) y reproduce los números de la profesora."""
    out = {}
    # (a) Metropolis con propuesta asimétrica sin corrección de Hastings sesga el resultado
    logp = log_beta_nucleo(16, 8)
    rng = np.random.default_rng(24990)
    n = 60000
    x = 0.6
    ch_mal = np.empty(n)
    ch_bien = np.empty(n)
    # propuesta asimétrica: x* = x + N(0.08, 0.1²) (tiene tendencia a subir)
    mu, s = 0.08, 0.1
    for corr, ch in ((False, ch_mal), (True, ch_bien)):
        x = 0.6
        for t in range(n):
            prop = x + mu + s * rng.normal()
            lp_prop, lp = logp(prop), logp(x)
            lq_ida = stats.norm.logpdf(prop, x + mu, s)
            lq_vuelta = stats.norm.logpdf(x, prop + mu, s)
            logr = lp_prop - lp + ((lq_vuelta - lq_ida) if corr else 0.0)
            if np.log(rng.uniform()) < logr:
                x = prop
            ch[t] = x
    out["asimetrica"] = dict(media_sin_correccion=float(ch_mal[2000:].mean()), media_con_correccion=float(ch_bien[2000:].mean()), exacta=float(stats.beta(16, 8).mean()))
    # (b) más iteraciones no empeoran la convergencia: error de la media según n
    # (c) la media posterior Beta(16,8) y la predictiva
    NUM["verif"] = out


PASOS = dict(normal=normal_fig, predictiva=predictiva_fig, montecarlo=montecarlo_fig, esquema=esquema_fig, porque=porque_fig, metropolis=metropolis_fig, traza=traza_fig,
             burnin=burnin_fig, acf=acf_fig, rhat=rhat_fig, pima=pima_fig, pred=pred_fig, verif=verificaciones)
ORDEN = ["normal", "predictiva", "montecarlo", "esquema", "metropolis", "traza", "burnin", "acf", "rhat", "pima", "pred", "porque", "verif"]

if __name__ == "__main__":
    import sys
    _prev = HERE / "c24_numeros.json"
    if _prev.exists():
        NUM.update(json.loads(_prev.read_text(encoding="utf-8")))
    elegidos = sys.argv[1:] or ORDEN
    for k in ORDEN:
        if k in elegidos:
            PASOS[k]()
            print("hecho:", k, flush=True)
    out = _prev
    out.write_text(json.dumps(NUM, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")
