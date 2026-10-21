"""Figuras de los apuntes de la Carpeta 1 (regresión, Clases 17-19)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figuras import GRID, INK, MUTED, save

C17 = "#2563EB"
C18 = "#0D9488"
C19 = "#7C3AED"
RED = "#DC2626"
AMBER = "#D97706"


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def anova_idea():
    g = {"A": [4, 5, 6], "B": [7, 8, 9], "C": [5, 6, 7]}
    gm = np.mean(sum(g.values(), []))
    cols = [C17, C18, C19]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for ax, title in [(a1, "Variación ENTRE grupos: medias vs media general"),
                      (a2, "Variación DENTRO de los grupos: datos vs su media")]:
        ax.axhline(gm, color=MUTED, ls="--", lw=1.2)
        ax.text(0.5, gm + 0.12, f"media general = {gm:.2f}", fontsize=8.3, color=MUTED, ha="center")
        for i, (k, v) in enumerate(g.items()):
            m = np.mean(v)
            ax.scatter([i] * 3, v, s=55, color=cols[i], zorder=3, alpha=0.9)
            ax.plot([i - 0.25, i + 0.25], [m, m], color=cols[i], lw=3)
            if ax is a1:
                ax.annotate("", xy=(i + 0.32, m), xytext=(i + 0.32, gm),
                            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
            else:
                for val in v:
                    ax.plot([i + 0.12, i + 0.12], [val, m], color=AMBER, lw=1.5)
        ax.set_xticks([0, 1, 2], list(g))
        ax.set_ylim(3, 10)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    a1.text(0.02, 0.92, "flechas rojas = SC entre = 14", transform=a1.transAxes, fontsize=8.5, color=RED)
    a2.text(0.02, 0.92, "líneas naranjas = SC dentro = 6", transform=a2.transAxes, fontsize=8.5, color=AMBER)
    fig.tight_layout(w_pad=3)
    save(fig, "reg_anova.png")


def minimos_cuadrados():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    b0, b1 = 2.2, 0.6
    fig, ax = plt.subplots(figsize=(10, 3.0))
    xs = np.linspace(0.5, 5.5, 50)
    ax.plot(xs, b0 + b1 * xs, color=C18, lw=2.4, label="ŷ = 2.2 + 0.6·x")
    for xi, yi in zip(x, y):
        yh = b0 + b1 * xi
        ax.plot([xi, xi], [yi, yh], color=RED, lw=1.6)
        ax.text(xi + 0.07, (yi + yh) / 2, f"e = {yi - yh:+.1f}", fontsize=8.2, color=RED, va="center")
    ax.scatter(x, y, s=60, color=INK, zorder=3)
    ax.axhline(4, color=MUTED, ls=":", lw=1)
    ax.text(5.45, 4.08, "ȳ = 4", fontsize=8.3, color=MUTED, ha="right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mínimos cuadrados: la recta que hace más chica la suma de los residuos al cuadrado (Σe² = 2.4)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    save(fig, "reg_mc.png")


def patrones_residuos():
    rng = np.random.default_rng(2)
    x = np.linspace(1, 10, 80)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    cases = [
        ("Bien: nube sin forma", rng.normal(0, 1, 80), C18),
        ("Embudo: heterocedasticidad", rng.normal(0, 1, 80) * x / 3, AMBER),
        ("Curva: falta un término no lineal", (x - 5.5) ** 2 / 4 - 2 + rng.normal(0, 0.6, 80), RED),
    ]
    for ax, (title, r, c) in zip(axes, cases):
        ax.scatter(x, r, s=12, color=c, alpha=0.8)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=9.6, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("valores ajustados", fontsize=8.5)
        ax.set_ylabel("residuo", fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2)
    save(fig, "reg_residuos.png")


if __name__ == "__main__":
    anova_idea()
    minimos_cuadrados()
    patrones_residuos()
    print("ok")
