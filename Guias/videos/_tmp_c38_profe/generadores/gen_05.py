import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, fila)

OUT = out("guion_c38_05_sesgo.json")
v = V("Sobreajuste, sesgo y varianza")
RK = {r["modelo"]: r for r in D["h_ranking"]}

v.portada("Sobreajuste, sesgo y varianza", "Ajustar mejor lo conocido no es predecir mejor lo nuevo",
          "Quinta parada: el sobreajuste, y las tres fuentes del error de predicción: sesgo, varianza y ruido.")

# ---- simulacion propia (ilustrativa), semilla fija
rng = np.random.default_rng(38)
xs = np.sort(rng.uniform(-1, 1, 24))
senal = lambda x: 1.5 - 3.0 * x ** 2 + 0.8 * x  # noqa: E731
ys = senal(xs) + rng.normal(0, 0.6, len(xs))
XX = np.linspace(xs.min(), xs.max(), 121)
c2 = np.polyfit(xs, ys, 2)
Y2 = np.polyval(c2, XX)
_V = np.polynomial.legendre.legvander(xs, 12)          # grado 12 con un freno minimo (1e-3) para que no explote en los bordes
_b = np.linalg.solve(_V.T @ _V + 1e-3 * np.eye(13), _V.T @ ys)
Y12 = np.polynomial.legendre.legvander(XX, 12) @ _b
PTS = dict(xs=[round(float(a), 3) for a in xs] + [5.0], ys=[round(float(b), 3) for b in ys] + [0.0], color="gris", linea=False,
           marcadores=8, revela=0.0, lapiz=False)


def curva(Y, col, g=6):
    return dict(xs=[round(float(a), 3) for a in XX], ys=[round(float(b), 3) for b in Y], color=col, grosor=g, revela=0.0, lapiz=False)


# ---- 1. Hook
v.b("Mira dos curvas ajustadas a los mismos puntos. [[a]]Una es un polinomio de grado dos: una curva suave. [[b]]La otra es de grado doce: se dobla para pasar cerca de cada punto. [[c]]En los datos conocidos, la de grado doce se equivoca menos. ¿Es la mejor? Veamos qué pasa con datos nuevos.",
    CL(), T("grado 2 frente a grado 12", 960, 140, tam=54, c="tinta", ent="subir"),
    G(420, 240, 1500, 860, -1, 1, -4.5, 3.5, [PTS, curva(Y2, "azul"), curva(Y12, "rojo", 5)], xticks=[-1, 0, 1], yticks=[-4, -2, 0, 2],
      yfmt="{:.0f}", xlabel="x", ylabel="y", id="gs", tam_txt=28),
    U("gs", **{"series.0.revela": 1.0}, t=0.2, dur=1.0),
    U("gs", **{"series.1.revela": 1.0}, en="a", dur=1.2), T("grado 2", 1540, 420, tam=40, c="azul", anc="lm", en="a", t=0.6),
    U("gs", **{"series.2.revela": 1.0}, en="b", dur=1.6), T("grado 12", 1540, 520, tam=40, c="rojo", anc="lm", en="b", t=0.8),
    ilustrativo(1010, texto="ejemplo ilustrativo (simulación propia, como en la diapositiva 18)"), sec="1. Dos curvas")
v.b("Las cifras de la diapositiva dieciocho lo dicen todo. [[a]]Grado dos: error de ajuste 0.67, error en datos nuevos 0.64. [[b]]Grado doce: error de ajuste 0.58, más bajo; pero en datos nuevos, 0.73, más alto. [[c]]El grado doce ajusta mejor lo conocido, y predice peor lo nuevo. Eso es sobreajuste.",
    CL(), T("RMSE de ajuste y en datos nuevos · diapositiva 18", 960, 150, tam=48, c="tinta", ent="subir"),
    fila(300, [("", 520), ("ajuste", 960), ("datos nuevos", 1400)], tams=[40] * 3, colores=["gris"] * 3),
    fila(420, [("grado 2", 520), ("0.67", 960), ("0.64", 1400)], tams=[46, 56, 56], colores=["azul", "tinta", "verde"], en="a",
         fondo="azulclaro", x0=340, x1=1580, h=100),
    fila(550, [("grado 12", 520), ("0.58", 960), ("0.73", 1400)], tams=[46, 56, 56], colores=["rojo", "verde", "rojo"], en="b",
         fondo="rojoclaro", x0=340, x1=1580, h=100),
    P("ajusta mejor lo conocido, predice peor lo nuevo: SOBREAJUSTE", 960, 740, "ambar", 44, en="c"), diapo(18, en="a"))

# ---- 2. Analogía
v.b("Piensa en un sastre que te hace un traje a la medida, pero copia cada arruga de la camisa que llevabas ese día. [[a]]Ese día, el traje calza perfecto. [[b]]Al día siguiente, con otra camisa, ya no calza. [[c]]El profesor lo definió con una frase muy buena: en el sobreajuste, «no estoy modelando el fenómeno, sino que estoy modelando mis datos».",
    CL(), T("el sastre que copia las arrugas", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "ese día", "calza perfecto", "a"), (1360, "rojo", "al día siguiente", "con otra camisa, no calza", "b")],
             y=430, w=620, h=280, tam_t=52, tam_d=40),
    cita("no estoy modelando el fenómeno, sino que estoy modelando mis datos", 730, en="c", tam=44),
    etiqueta("cita de la grabación de la clase", 820, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: el sastre")

# ---- 3. Tres fuentes de error
v.b("¿De dónde sale el error de una predicción? La diapositiva diecinueve lo descompone en tres partes. [[a]]El error cuadrático esperado es igual al sesgo al cuadrado, más la varianza, más el ruido. [[b]]El sesgo es la distancia entre la señal verdadera y el ajuste promedio. La varianza, cuánto cambia el ajuste al cambiar la muestra. [[c]]Y el ruido es la variación de la respuesta que existe incluso si fijamos los predictores: ningún modelo lo elimina.",
    CL(), T("sesgo, varianza y ruido · diapositiva 19", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("error cuadrático esperado = sesgo² + varianza + ruido", y=300, c="violeta", tam=48, w=1500, en="a"),
    tarjetas([(400, "azul", "sesgo", "señal vs. ajuste promedio", "b"), (960, "rojo", "varianza", "cuánto cambia el ajuste con otra muestra", "b"),
              (1520, "gris", "ruido", "variación que ningún modelo elimina", "c")], y=600, w=500, h=300, tam_t=48, tam_d=34),
    diapo(19, en="a"), sec="3. Tres fuentes de error")

# ---- 4. Analogía: los dardos
DIANA = []
for i, (cx, cy, puntos, col, k) in enumerate([(560, 560, [(60, -40), (75, -25), (55, -60), (80, -45), (65, -30)], "azul", "a"),
                                               (1360, 560, [(-90, 40), (70, -80), (20, 95), (-60, -70), (95, 30)], "rojo", "b")]):
    DIANA += [Bc(cx, cy, 300, 300, "blanco", "gris", 4, 150, en=k), Bc(cx, cy, 180, 180, "blanco", "gris", 3, 90, en=k),
              Bc(cx, cy, 60, 60, "ambarclaro", "ambar", 3, 30, en=k)]
    DIANA += [Bc(cx + dx, cy + dy, 22, 22, col, None, 0, 11, en=k, t=0.4 + 0.12 * j) for j, (dx, dy) in enumerate(puntos)]
v.b("Una imagen ayuda: tirar dardos a una diana. [[a]]Sesgo alto: los dardos quedan juntos, pero lejos del centro. Siempre te equivocas hacia el mismo lado. [[b]]Varianza alta: en promedio apuntas al centro, pero los dardos quedan desparramados. Cada intento sale distinto. [[c]]El modelo de grado dos se parece al primero si es demasiado simple; el de grado doce, al segundo.",
    CL(), T("la diana", 960, 150, tam=56, c="tinta", ent="subir"), DIANA,
    T("sesgo alto: juntos, pero corridos", 560, 780, tam=38, c="azul", w="sb", en="a", t=0.8),
    T("varianza alta: centrados, pero dispersos", 1360, 780, tam=38, c="rojo", w="sb", en="b", t=0.8),
    P("muy simple → sesgo · muy flexible → varianza", 960, 900, "violeta", 44, en="c"), ilustrativo(1010, texto="analogía"),
    sec="4. Analogía: la diana")

# ---- 5. Ejemplo resuelto
v.b("Hagamos la cuenta con números inventados. [[a]]En un punto, la señal verdadera vale 10, y el ajuste promedio, sobre muchas muestras, vale 12. El sesgo es 2, y al cuadrado, 4. [[b]]Las predicciones varían alrededor de 12 con una varianza de 1. [[c]]Y el ruido tiene varianza 0.25. El error cuadrático esperado es cuatro, más uno, más 0.25: 5.25.",
    CL(), T("la descomposición, con números", 960, 150, tam=54, c="tinta", ent="subir"),
    T("señal = 10   ·   ajuste promedio = 12", 960, 280, tam=44, c="tinta", w="sb", en="a"),
    T("sesgo² = (12 − 10)² =", 1000, 400, tam=46, c="tinta", w="sb", anc="rm", en="a", t=1.0), N(4, 1030, 400, "{:.0f}", 60, "azul", anc="lm", en="a", t=1.6),
    T("varianza =", 1000, 510, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(1, 1030, 510, "{:.0f}", 60, "rojo", anc="lm", en="b", t=0.5),
    T("ruido =", 1000, 620, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(0.25, 1030, 620, "{:.2f}", 60, "gris", anc="lm", en="c", t=0.5),
    T("error esperado = 4 + 1 + 0.25 =", 1000, 760, tam=50, c="tinta", w="sb", anc="rm", en="c", t=1.6), N(5.25, 1030, 760, "{:.2f}", 66, "violeta", anc="lm", en="c", t=2.2),
    ilustrativo(1000), sec="5. La cuenta con números")

# ---- 6. Flexibilidad y validación
v.b("¿Y qué se hace con esto? [[a]]Aumentar la flexibilidad suele reducir el sesgo, pero aumenta la varianza. [[b]]En la regresión ridge, la flexibilidad la controla alfa: un alfa grande restringe, más sesgo y menos varianza; un alfa pequeño libera, menos sesgo y más varianza. [[c]]No sabemos de antemano dónde está el equilibrio. Por eso la diapositiva dice: la validación permite comparar el efecto conjunto.",
    CL(), T("más flexibilidad: menos sesgo, más varianza", 960, 150, tam=52, c="tinta", ent="subir"),
    L(360, 400, 1560, 400, "gris", 6, True, en="a"), T("menos flexible", 360, 460, tam=34, c="azul", w="sb", anc="lm", en="a"),
    T("más flexible", 1560, 460, tam=34, c="rojo", w="sb", anc="rm", en="a"),
    T("α grande: más sesgo, menos varianza", 560, 580, tam=36, c="azul", w="sb", en="b"),
    T("α pequeño: menos sesgo, más varianza", 1360, 580, tam=36, c="rojo", w="sb", en="b", t=1.2),
    P("la validación compara el efecto conjunto", 960, 760, "violeta", 48, en="c"), diapo(19, en="c"), sec="6. Flexibilidad y validación")
v.b("Con los datos reales del hormigón se ve claro. [[a]]La regresión lineal es demasiado simple: su error fuera de pliegue es 10.560 megapascales. [[b]]La cuadrática con alfa igual a mil está demasiado restringida: 13.051. [[c]]Y la cuadrática con alfa igual a 0.1 logra el menor error: 8.229. Ni muy rígida, ni sin freno.",
    CL(), T("error fuera de pliegue (RMSE, MPa) · datos reales", 960, 150, tam=48, c="tinta", ent="subir"),
    *[x for (lab, val, c, k) in [("lineal (muy simple)", RK["LINEAL"]["rmse"], "azul", "a"), ("cuadrática, α = 1000 (muy restringida)", RK["Q_A1000"]["rmse"], "gris", "b"),
                                  ("cuadrática, α = 0.1", RK["Q_A0.1"]["rmse"], "verde", "c")]
      for x in [T(lab, 860, {"a": 330, "b": 470, "c": 610}[k], tam=34, c="tinta", w="sb", anc="rm", en=k),
                Bx(890, {"a": 330, "b": 470, "c": 610}[k] - 34, 890 + val * 60, {"a": 330, "b": 470, "c": 610}[k] + 34, c, None, 0, 10, grow="x", en=k, t=0.1),
                N(val, 910 + val * 60, {"a": 330, "b": 470, "c": 610}[k], "{:.3f}", 46, c, anc="lm", en=k, t=0.5)]],
    P("ni muy rígida, ni sin freno", 960, 780, "teal", 48, en="c", t=1.5), real(1050, en="a"))

# ---- 7. Ajuste frente a validación
v.b("Un detalle que adelanta la próxima parada. [[a]]Para la cuadrática con alfa igual a 0.1, el error en las mismas filas usadas para estimar es 6.958. [[b]]Y fuera de pliegue, 8.229. [[c]]La diferencia es optimismo: el modelo siempre se ve mejor en los datos que ya vio. El error de ajuste nunca sirve para elegir.",
    CL(), T("error de ajuste frente a error fuera de pliegue", 960, 150, tam=50, c="tinta", ent="subir"),
    T("ajuste", 560, 320, tam=42, c="gris", w="sb", en="a"), N(RK["Q_A0.1"]["rmse_ajuste"], 560, 420, "{:.3f}", 90, "verde", en="a", t=0.3),
    T("fuera de pliegue", 1360, 320, tam=42, c="gris", w="sb", en="b"), N(RK["Q_A0.1"]["rmse"], 1360, 420, "{:.3f}", 90, "rojo", en="b", t=0.3),
    P("la diferencia es optimismo", 960, 620, "ambar", 50, en="c"),
    T("cuadrática con ridge, α = 0.1 · RMSE en MPa", 960, 730, tam=32, c="gris", w="sb", en="c", t=0.8), real(1050, en="a"), sec="7. Ajuste frente a validación")

# ---- 8. El error típico
v.b("El error típico: [[a]]elegir el modelo con el menor error en los datos de entrenamiento. [[b]]Con ese criterio, siempre gana el más flexible: el grado doce le gana al grado dos, 0.58 contra 0.67. [[c]]Y es justamente el que peor predice casos nuevos.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("elegir por el error de entrenamiento", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("siempre gana el más flexible: 0.58 < 0.67", 960, 440, tam=46, c="tinta", w="sb", en="b"),
    P("…y en datos nuevos: 0.73 > 0.64", 960, 620, "rojo", 50, en="c"), diapo(18, en="b"), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]El modelo A tiene sesgo al cuadrado igual a 1, varianza 3 y ruido 0.5. [[b]]El modelo B tiene sesgo al cuadrado 2.5, varianza 0.5 y ruido 0.5. [[c]]¿Cuál tiene el menor error esperado?",
    CL(), T("¿cuál tiene menor error esperado?", 960, 160, tam=54, c="tinta", ent="subir"),
    P("A: sesgo² = 1 · varianza = 3 · ruido = 0.5", 960, 340, "azul", 46, en="a"),
    P("B: sesgo² = 2.5 · varianza = 0.5 · ruido = 0.5", 960, 480, "rojo", 46, en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("A: uno, más tres, más 0.5: [[a]]4.5. B: 2.5, más 0.5, más 0.5: [[b]]3.5. [[c]]Gana B, aunque tenga más sesgo: su varianza es mucho menor. Y fíjate que el ruido es el mismo para los dos, porque no depende del modelo.",
    T("A = 1 + 3 + 0.5 =", 1000, 640, tam=46, c="tinta", w="sb", anc="rm"), N(4.5, 1030, 640, "{:.1f}", 60, "azul", anc="lm", en="a"),
    T("B = 2.5 + 0.5 + 0.5 =", 1000, 750, tam=46, c="tinta", w="sb", anc="rm", en="a", t=0.8), N(3.5, 1030, 750, "{:.1f}", 60, "rojo", anc="lm", en="b"),
    P("gana B: más sesgo, pero mucha menos varianza", 960, 900, "teal", 44, en="c"), ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]ajustar mejor lo conocido no es predecir mejor lo nuevo; eso es el sobreajuste. [[b]]El error esperado es sesgo al cuadrado, más varianza, más ruido; más flexibilidad baja el sesgo y sube la varianza. [[c]]Y el equilibrio no se adivina: se busca con validación, nunca con el error de entrenamiento.",
    CL(), P("ajustar mejor lo conocido ≠ predecir mejor lo nuevo", 960, 300, "ambar", 44, en="a"),
    P("error = sesgo² + varianza + ruido", 960, 460, "violeta", 48, en="b"),
    P("el equilibrio se busca con validación", 960, 620, "teal", 48, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]la validación cruzada por dentro. Las predicciones fuera de pliegue, y por qué la receta se reaprende en cada ajuste.",
    CL(), siguiente(6, "Validación cruzada y predicciones fuera de pliegue"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
