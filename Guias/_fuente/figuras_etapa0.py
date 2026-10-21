"""Figuras de la guía de la Etapa 0 (nivelación estadística)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats

from figuras import C1719, GRID, INK, MUTED, OUT, save

E0 = "#475569"
BLUE = "#2563EB"
TEAL = "#0D9488"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#059669"
PINK = "#DB2777"

ENS = pd.read_csv(C1719 / "ENS.csv")
COL = ENS["COLESTEROL"]


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ concept chain
def mapa():
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (1.0, 2.9, "0.1\nDescribir\nlos datos"),
        (3.0, 2.9, "0.2\nDistribuciones\ny la normal"),
        (5.0, 2.9, "0.3\nMuestras y\nerror estándar"),
        (7.2, 3.35, "0.4\nIntervalo de\nconfianza"),
        (7.2, 1.55, "0.5 · 0.6\nPrueba de hipótesis\ny p-valor"),
        (9.5, 1.55, "0.7\nPruebas t,\nproporción, χ²"),
        (11.4, 2.45, "Etapa 1\nANOVA y\nregresión"),
    ]
    w, h = 1.75, 1.25
    for i, (x, y, t) in enumerate(nodes):
        last = i == len(nodes) - 1
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white" if last else E0, ec=BLUE if last else E0,
                                    lw=1.8, ls="--" if last else "-"))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.4, fontweight="bold",
                color=BLUE if last else "white")

    def arr(a, b, label=None, dy=0.0):
        (xa, ya, _), (xb, yb, _) = nodes[a], nodes[b]
        p = FancyArrowPatch((xa + w / 2, ya), (xb - w / 2, yb), arrowstyle="-|>",
                            mutation_scale=13, color=MUTED, lw=1.3)
        ax.add_patch(p)
        if label:
            ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.28 + dy, label, ha="center",
                    fontsize=7.8, color=MUTED, style="italic")

    arr(0, 1)
    arr(1, 2)
    arr(2, 3)
    arr(2, 4)
    ax.text(6.0, 3.45, "estimar", fontsize=8.2, color=MUTED, style="italic", ha="right")
    ax.text(5.95, 1.75, "decidir", fontsize=8.2, color=MUTED, style="italic", ha="right")
    arr(4, 5)
    arr(3, 6)
    arr(5, 6)
    ax.text(0.1, 0.25, "La pregunta de toda la etapa: con una MUESTRA, ¿qué puedo afirmar de la POBLACIÓN, "
            "y con cuánta seguridad?", fontsize=9.6, color=INK, style="italic")
    save(fig, "e0_mapa.png")


# ------------------------------------------------------------ 0.1
def s01():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3))
    a1.hist(COL, bins=25, color=E0, alpha=0.85, edgecolor="white")
    a1.axvline(COL.mean(), color=RED, lw=2, label=f"media = {COL.mean():.1f}")
    a1.axvline(COL.median(), color=AMBER, lw=2, ls="--", label=f"mediana = {COL.median():.1f}")
    a1.set_xlabel("Colesterol (mg/dL)")
    a1.set_ylabel("N° de personas")
    a1.set_title("Distribución del colesterol (n = 350)", loc="left", fontsize=10.5, color=INK)
    a1.legend(frameon=False, fontsize=8.5)
    style(a1)

    orden = ["Joven", "Adulto", "Adulto Mayor"]
    data = [ENS.loc[ENS.EDAD == g, "COLESTEROL"] for g in orden]
    bp = a2.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", mfc=RED, mec="white", ms=6))
    for b in bp["boxes"]:
        b.set(facecolor=E0, edgecolor=E0)
    a2.set_xticks([1, 2, 3])
    a2.set_xticklabels(orden)
    a2.set_ylabel("Colesterol")
    a2.set_title("Boxplot por grupo de edad", loc="left", fontsize=10.5, color=INK)
    q1, q3 = np.percentile(data[0], [25, 75])
    a2.annotate("caja = 50% central\n(Q1 a Q3)", xy=(1.25, q3), xytext=(1.55, q3 + 55),
                fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED))
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "e0_s01.png")


# ------------------------------------------------------------ 0.2
def s02():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3))
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)
    a1.plot(x, y, color=INK, lw=1.8)
    for k, c, lab in [(3, "#CBD5E1", "99.7%"), (2, "#94A3B8", "95%"), (1, E0, "68%")]:
        m = (x >= -k) & (x <= k)
        a1.fill_between(x[m], y[m], color=c)
    for k, lab, yy in [(1, "68%", 0.2), (2, "95%", 0.075), (3, "99.7%", 0.012)]:
        a1.annotate("", xy=(-k, yy), xytext=(k, yy),
                    arrowprops=dict(arrowstyle="<->", color="white" if k < 3 else INK, lw=1))
        if k < 3:
            a1.text(0, yy + 0.012, lab, ha="center", fontsize=8.5, color="white", fontweight="bold")
        else:
            a1.text(3.15, yy + 0.02, lab, ha="left", fontsize=8.5, color=INK, fontweight="bold")
    a1.set_xticks(range(-3, 4))
    a1.set_xticklabels([f"{k}σ" if k else "μ" for k in range(-3, 4)])
    a1.set_yticks([])
    a1.set_title("Regla 68–95–99.7 de la normal", loc="left", fontsize=10.5, color=INK)
    a1.spines["left"].set_visible(False)

    a2.hist(COL, bins=25, density=True, color=E0, alpha=0.55, edgecolor="white")
    xs = np.linspace(COL.min() - 20, COL.max() + 20, 300)
    a2.plot(xs, stats.norm.pdf(xs, COL.mean(), COL.std()), color=RED, lw=2,
            label=f"Normal(μ={COL.mean():.0f}, σ={COL.std():.0f})")
    z = (250 - COL.mean()) / COL.std()
    a2.axvline(250, color=AMBER, lw=1.6, ls="--")
    a2.text(253, a2.get_ylim()[1] * 0.85 if a2.get_ylim()[1] else 0.008,
            f"250 mg/dL\nz = {z:.2f}", fontsize=8.5, color=AMBER)
    a2.set_yticks([])
    a2.set_xlabel("Colesterol")
    a2.set_title("¿El colesterol se parece a una normal?", loc="left", fontsize=10.5, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=3)
    save(fig, "e0_s02.png")


# ------------------------------------------------------------ 0.3
def s03():
    rng = np.random.default_rng(7)
    pob = rng.exponential(scale=10, size=100_000)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].hist(pob, bins=60, color=MUTED, alpha=0.8, edgecolor="white")
    axes[0].set_title("Población (muy asimétrica)", loc="left", fontsize=10, color=INK)
    for ax, n, col in [(axes[1], 5, AMBER), (axes[2], 30, TEAL)]:
        medias = rng.choice(pob, size=(5000, n)).mean(axis=1)
        ax.hist(medias, bins=50, color=col, alpha=0.85, edgecolor="white", density=True)
        xs = np.linspace(medias.min(), medias.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, pob.mean(), pob.std() / np.sqrt(n)), color=INK, lw=1.5)
        ax.set_title(f"5.000 medias con n = {n}\nEE = σ/√n = {pob.std() / np.sqrt(n):.1f}",
                     loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 45 if ax is axes[0] else 30)
    fig.tight_layout(w_pad=2)
    save(fig, "e0_s03.png")


# ------------------------------------------------------------ 0.4
def s04():
    rng = np.random.default_rng(3)
    mu, sigma, n, k = COL.mean(), COL.std(), 30, 40
    fig, ax = plt.subplots(figsize=(10, 3.4))
    fallan = 0
    for i in range(k):
        m = rng.normal(mu, sigma, n)
        lo, hi = stats.t.interval(0.95, n - 1, loc=m.mean(), scale=stats.sem(m))
        ok = lo <= mu <= hi
        fallan += not ok
        c = TEAL if ok else RED
        ax.plot([i, i], [lo, hi], color=c, lw=2.2 if not ok else 1.6)
        ax.plot(i, m.mean(), "o", color=c, ms=3.5)
    ax.axhline(mu, color=INK, lw=1.2, ls="--")
    ax.text(-1.2, mu, "μ real", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-3.5, k)
    ax.set_xticks([])
    ax.set_xlabel("40 muestras distintas de 30 personas, cada una con su IC 95%")
    ax.set_ylabel("Colesterol")
    verbo = "falla" if fallan == 1 else "fallan"
    ax.set_title(f"{k - fallan} de {k} intervalos atrapan la media real; {fallan} {verbo} (en rojo). "
                 "En el largo plazo, el 95% acierta.", loc="left", fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout()
    save(fig, "e0_s04.png")


# ------------------------------------------------------------ 0.5
def s05():
    a = ENS.loc[ENS.FUMA == "Si", "COLESTEROL"]
    b = ENS.loc[ENS.FUMA == "No", "COLESTEROL"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color=INK, lw=1.8)
    m = np.abs(x) >= abs(t)
    ax.fill_between(x, y, where=x >= abs(t), color=AMBER, alpha=0.7)
    ax.fill_between(x, y, where=x <= -abs(t), color=AMBER, alpha=0.7)
    ax.axvline(t, color=RED, lw=2)
    ax.text(t + 0.08, 0.36, f"t observado = {t:.2f}", color=RED, fontsize=9)
    ax.text(2.4, 0.08, f"área naranja\n= p-valor = {p:.2f}", color=AMBER, fontsize=9,
            fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Estadístico t (cómo se vería si H0 fuera cierta)")
    ax.set_title("¿Fumar cambia el colesterol?  Si H0 fuera cierta, un resultado así de extremo "
                 "aparece el 30% de las veces → no hay evidencia", loc="left", fontsize=10.2, color=INK)
    fig.tight_layout()
    save(fig, "e0_s05.png")


# ------------------------------------------------------------ 0.6
def s06():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.linspace(-4, 7, 600)
    h0 = stats.norm.pdf(x, 0, 1)
    h1 = stats.norm.pdf(x, 2.5, 1)
    crit = stats.norm.ppf(0.95)
    ax.plot(x, h0, color=INK, lw=1.8, label="Si H0 es cierta")
    ax.plot(x, h1, color=BLUE, lw=1.8, label="Si H1 es cierta")
    ax.fill_between(x, h0, where=x >= crit, color=RED, alpha=0.6, label="α = error tipo I")
    ax.fill_between(x, h1, where=x < crit, color=AMBER, alpha=0.5, label="β = error tipo II")
    ax.fill_between(x, h1, where=x >= crit, color=BLUE, alpha=0.18, label="Potencia = 1 − β")
    ax.axvline(crit, color=MUTED, lw=1.2, ls="--")
    ax.text(crit - 0.08, 0.30, "umbral de\ndecisión", fontsize=8.5, color=MUTED, ha="right")
    ax.text(-3.6, 0.3, "← no rechazo H0", fontsize=9, color=INK)
    ax.text(4.6, 0.3, "rechazo H0 →", fontsize=9, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right", ncol=1)
    ax.set_xlim(-4, 7)
    ax.set_title("Los dos errores posibles: bajar α (menos falsas alarmas) sube β, salvo que aumentes n",
                 loc="left", fontsize=10.2, color=INK)
    fig.tight_layout()
    save(fig, "e0_s06.png")


# ------------------------------------------------------------ 0.7
def s07():
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def node(x, y, t, fc, tc="white", w=2.5, h=0.85, dashed=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=fc if fc != "white" else E0, lw=1.4,
                                    ls="--" if dashed else "-"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.8, color=tc, fontweight="bold")

    def arrow(p1, p2, lab=None, off=(0, 0.14)):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, lw=1.2))
        if lab:
            ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], lab,
                    ha="center", fontsize=8, color=MUTED, style="italic")

    node(1.5, 2.2, "¿Qué tipo de\nvariable respuesta?", E0, w=2.4, h=1.0)
    node(5.0, 3.5, "Numérica\n(ej. colesterol)", "white", INK, w=2.3)
    node(5.0, 0.9, "Categórica\n(ej. fuma sí/no)", "white", INK, w=2.3)
    arrow((2.7, 2.45), (3.85, 3.4))
    arrow((2.7, 1.95), (3.85, 1.0))
    node(9.4, 4.0, "1 grupo vs valor → t de una muestra", TEAL, w=4.6, h=0.6)
    node(9.4, 3.2, "2 grupos → t de dos muestras", TEAL, w=4.6, h=0.6)
    node(9.4, 2.4, "3+ grupos → ANOVA (Etapa 1)", BLUE, w=4.6, h=0.6, dashed=False)
    node(9.4, 1.3, "1 proporción vs valor → prueba z", AMBER, w=4.6, h=0.6)
    node(9.4, 0.5, "2 categóricas → chi-cuadrado (χ²)", AMBER, w=4.6, h=0.6)
    for y in (4.0, 3.2, 2.4):
        arrow((6.15, 3.5), (7.1, y))
    for y in (1.3, 0.5):
        arrow((6.15, 0.9), (7.1, y))
    save(fig, "e0_s07.png")


if __name__ == "__main__":
    mapa()
    s01()
    s02()
    s03()
    s04()
    s05()
    s06()
    s07()
    print("ok")
