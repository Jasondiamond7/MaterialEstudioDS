import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, matriz, alerta, fila, rh

OUT = out("guion_c37_04_umbral.json")
v = V("Umbral, odds y coeficientes")
UP = FG["umbral_prueba"]
UC = FG["umbral_cv"]
GR = FG["glm_como_R"]
LS = FG["logistica_std"]
SEP = D["separacion"]

v.portada("Umbral, odds y coeficientes", "Mover el umbral con datos reales y leer los coeficientes sin trampas",
          "Cuarta parada: el umbral con datos reales, los odds, y cómo leer los coeficientes de la logística sin caer en trampas.")

# ---- 1. Hook
v.b("¿El umbral tiene que ser siempre 0.5? [[a]]Si un falso negativo es un cáncer que no se detecta, quizás convenga dar la alarma antes. [[b]]El umbral no es una propiedad del modelo: es una decisión, y se toma pensando en el costo de cada error.",
    CL(), T("¿siempre t = 0.5?", 960, 220, tam=70, c="tinta", ent="subir"),
    P("un FN = un cáncer no detectado", 960, 420, "rojo", 52, en="a"),
    P("el umbral es una DECISIÓN, según el costo de cada error", 960, 600, "ambar", 50, en="b"), sec="1. ¿Siempre 0.5?")

# ---- 2. Analogía: el guardia
v.b("Piensa en un guardia de aeropuerto que decide a quién revisar. [[a]]Si es muy exigente, revisa a casi todos: no se le escapa nada, pero molesta a mucha gente inocente. [[b]]Si es relajado, casi no molesta, pero se le pueden escapar cosas. [[c]]Bajar el umbral es volverse exigente; subirlo, relajarse.",
    CL(), T("el guardia del aeropuerto", 960, 160, tam=54, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "umbral bajo", "revisa a casi todos: más detecciones y más falsas alarmas", "a"),
              (1360, "verde", "umbral alto", "molesta poco: menos falsas alarmas, pero se escapan casos", "b")], y=500, w=640, h=360, tam_t=50, tam_d=36),
    P("bajar t = ser exigente · subir t = relajarse", 960, 820, "ambar", 46, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: el guardia")

# ---- 3. El umbral con datos reales
XS = [380, 960, 1540]
v.b("Ahora con los datos reales: la logística de la clase en los 171 casos de prueba, con tres umbrales. [[a]]Con te igual a 0.3, detecta 59 de los 64 malignos, pero da 13 falsas alarmas. [[b]]Con 0.5, detecta 58 y da 4 falsas alarmas. [[c]]Con 0.8, detecta solo 52, y da apenas 2 falsas alarmas.",
    CL(), T("la misma logística, tres umbrales · 171 casos de prueba", 960, 140, tam=46, c="tinta", ent="subir"),
    *[matriz(x, 480, [[UP[t]["tn"], UP[t]["fp"]], [UP[t]["fn"], UP[t]["tp"]]], en=k, cel=150, tam=46, tit=f"t = {t}", c=c, etiquetas=(i == 0))
      for i, (x, t, k, c) in enumerate(zip(XS, ["0.3", "0.5", "0.8"], ["a", "b", "c"], ["rojo", "azul", "verde"]))],
    real(1050, en="a"), sec="3. El umbral con datos reales")
v.b("En métricas: [[a]]con 0.3, sensibilidad 0.922 y especificidad 0.879. [[b]]Con 0.5, 0.906 y 0.963. [[c]]Con 0.8, 0.813 y 0.981. Subir el umbral gana especificidad y pierde sensibilidad. Siempre es un intercambio.",
    *[T(f"sens {rh(UP[t]['tp'], UP[t]['tp'] + UP[t]['fn'])} · espe {rh(UP[t]['tn'], UP[t]['tn'] + UP[t]['fp'])}", x, 720, tam=34, c="tinta", w="sb", en=k) for x, t, k in zip(XS, ["0.3", "0.5", "0.8"], ["a", "b", "c"])],
    P("subir t: + especificidad, − sensibilidad", 960, 860, "ambar", 48, en="c", t=1.0))
v.b("Pero ojo: [[a]]estos números de prueba sirven para ver el efecto, no para elegir. [[b]]El umbral se elige con validación, dentro del entrenamiento. [[c]]Con validación cruzada en los 398 casos, la sensibilidad y la especificidad se igualan cerca de te igual a 0.34: ambas, alrededor de 0.93.",
    CL(), P("la prueba muestra el efecto; NO sirve para elegir", 960, 300, "rojo", 48, en="a"),
    P("el umbral se elige con validación en ENTRENAMIENTO", 960, 450, "teal", 48, en="b"),
    T("validación cruzada (398 casos): se igualan en t ≈", 1080, 620, tam=42, c="tinta", w="sb", anc="rm", en="c"),
    N(UC["cruce"]["t"], 1110, 620, "{:.2f}", 56, "violeta", anc="lm", en="c", t=0.5),
    T(f"sensibilidad {UC['cruce']['sens']:.3f} · especificidad {UC['cruce']['espe']:.3f}", 960, 720, tam=38, c="gris", w="sb", en="c", t=1.2), real(1050, en="c"))

# ---- 4. Los odds
v.b("Para leer los coeficientes, necesitamos los odds. [[a]]Los odds son pe sobre uno menos pe. [[b]]Con probabilidad 0.8, los odds son 0.8 sobre 0.2: cuatro. Cuatro a uno, como en las apuestas. [[c]]Con 0.5, los odds valen uno: es parejo.",
    CL(), T("los odds", 960, 150, tam=54, c="tinta", ent="subir"), caja_formula("odds = p / (1 − p)", y=300, c="violeta", tam=60, w=900, en="a"),
    T("p = 0.8  →  0.8 / 0.2 =", 960, 460, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(4, 990, 460, "{:.0f}", 60, "violeta", anc="lm", en="b", t=0.5),
    T("«4 a 1»", 1120, 460, tam=40, c="gris", w="sb", anc="lm", en="b", t=0.8),
    T("p = 0.5  →  0.5 / 0.5 =", 960, 580, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(1, 990, 580, "{:.0f}", 60, "gris", anc="lm", en="c", t=0.5),
    sec="4. Los odds")
v.b("Y aquí está la conexión con eta: [[a]]el logaritmo de los odds es justamente el puntaje lineal. O sea, los odds son e elevado a eta. [[b]]En el ejemplo de la diapositiva diecisiete, eta valía uno: los odds son e, 2.718, y la probabilidad, 0.731.",
    P("log(odds) = η   →   odds = e^η", 960, 740, "teal", 50, en="a"),
    T("η = 1  →  odds = e = 2.718  →  p = 0.731", 960, 860, tam=44, c="tinta", w="sb", en="b"))

# ---- 5. Interpretar un coeficiente
v.b("Ahora sí, la diapositiva dieciocho. [[a]]Si x jota sube en una unidad, con los demás predictores fijos, los odds se multiplican por e elevado a beta jota. [[b]]Con beta igual a 0.8: e a la 0.8 es 2.23. Los odds se multiplican por 2.23.",
    CL(), T("leer un coeficiente · diapositiva 18", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("odds nuevo / odds anterior = e^βⱼ", y=310, c="violeta", tam=54, w=1100, en="a", wf="sb"),
    T("β = 0.8  →  e^0.8 =", 960, 470, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(2.23, 990, 470, "{:.2f}", 62, "violeta", anc="lm", en="b", t=0.5),
    sec="5. Leer un coeficiente")
v.b("Pero la probabilidad no se multiplica por 2.23. [[a]]Sigamos el ejemplo: eta pasa de uno a 1.8, y los odds, de 2.718 a 6.05. [[b]]La probabilidad pasa de 0.731 a 0.858. [[c]]Sube, pero no se multiplica: si se multiplicara por 2.23, daría 1.63, y una probabilidad no puede pasar de uno.",
    T("odds: 2.718 × 2.23 = 6.05", 960, 610, tam=46, c="tinta", w="sb", en="a"),
    T("p: 0.731  →", 950, 720, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(0.858, 980, 720, "{:.3f}", 60, "teal", anc="lm", en="b", t=0.5),
    P("0.731 × 2.23 = 1.63 → imposible", 960, 850, "rojo", 46, en="c"), ilustrativo(1000))
v.b("Dos advertencias más de la diapositiva. [[a]]Con variables estandarizadas, una unidad es una desviación estándar del entrenamiento. [[b]]Y la interpretación es una asociación condicional, no un efecto causal.",
    CL(), P("estandarizadas: 1 unidad = 1 desviación estándar", 960, 380, "azul", 50, en="a"),
    P("asociación condicional, NO efecto causal", 960, 540, "rojo", 50, en="b"))

# ---- 6. Los coeficientes de R
COEF = [("intercepto", GR["intercepto"]), ("radio", GR["coef"]["radio"]), ("textura", GR["coef"]["textura"]),
        ("suavidad", GR["coef"]["suavidad"]), ("concavidad", GR["coef"]["concavidad"])]
v.b("La diapositiva veintiocho muestra los coeficientes del glm de R, en unidades originales. [[a]]Intercepto menos 42.25; radio, 1.34; textura, 0.442; [[b]]suavidad, 124.98; y concavidad, 21.41. [[c]]La diapositiva lee la textura: una unidad más de textura multiplica los odds por e a la 0.4421, es decir, por 1.56.",
    CL(), T("los coeficientes de glm en R · diapositiva 28", 960, 140, tam=48, c="tinta", ent="subir"),
    *[fila(270 + 90 * i, [(n, 640), (f"{c:.4f}", 1180)], tams=[40, 40], colores=["tinta", "violeta"], en=("a" if i < 3 else "b"), t=0.15 * (i % 3),
           fondo=("violetaclaro" if n == "suavidad" else None), x0=460, x1=1460, h=80) for i, (n, c) in enumerate(COEF)],
    P("textura: e^0.4421 = 1.56 → odds × 1.56 por unidad", 960, 790, "teal", 44, en="c"),
    real(1050, en="a"), sec="6. Los coeficientes de R")

# ---- 7. Cuidado con la suavidad
v.b("Cuidado si alguien aplica esa misma lectura a la suavidad. [[a]]Una unidad más de suavidad multiplicaría los odds por e elevado a 124.98: un número de 55 cifras. [[b]]No tiene sentido práctico, porque la suavidad de toda la base va de 0.053 a 0.163. Una unidad completa no existe. [[c]]La lectura útil es otra: un aumento de 0.01 multiplica los odds por e a la 1.25, o sea, por 3.49.",
    CL(), alerta(170, texto="precisión al leer la diapositiva 28"),
    T("suavidad: e^124.98 ≈ 1.9 × 10⁵⁴", 960, 330, tam=50, c="tinta", w="sb", en="a", id="sv"), U("sv", tachado=1.0, en="b", t=1.2, dur=0.8),
    T("rango real de la suavidad: 0.053 a 0.163", 960, 460, tam=44, c="rojo", w="sb", en="b"),
    T("+0.01 de suavidad:  e^(124.98 × 0.01) = e^1.25 =", 1180, 600, tam=42, c="tinta", w="sb", anc="rm", en="c"),
    N(3.49, 1210, 600, "{:.2f}", 60, "verde", anc="lm", en="c", t=0.8), P("odds × 3.49 por cada +0.01", 960, 740, "verde", 46, en="c", t=1.6),
    real(1050, en="b"), sec="7. Cuidado con la suavidad")
v.b("Y en Python los coeficientes salen distintos, porque las variables están estandarizadas: [[a]]el radio vale 4.92 por desviación estándar, en vez de 1.34 por unidad. [[b]]Pero las probabilidades son las mismas. Para el caso nuevo de la clase, las dos versiones dan 0.3919.",
    CL(), T("Python (estandarizado) frente a R (unidades originales)", 960, 160, tam=46, c="tinta", ent="subir"),
    fila(340, [("radio", 520), ("R: 1.3399 por unidad", 960), ("Python: 4.9247 por sd", 1440)], tams=[44, 40, 40], colores=["tinta", "teal", "azul"], en="a"),
    P(f"misma probabilidad para el caso nuevo: {GR['p_nuevo']:.4f}", 960, 560, "violeta", 50, en="b"), real(1050, en="a"))

# ---- 8. Alerta: ¿solución única?
BETAS = ["1", "2", "5", "10"]
v.b("Una precisión sobre lo dicho en clase. [[a]]Se dijo que el estimador de máxima verosimilitud tiene garantizado que su solución es única. [[b]]No siempre. Si las clases están perfectamente separadas, la verosimilitud sigue subiendo a medida que el coeficiente crece, y nunca llega a un máximo.",
    CL(), alerta(170), T("«tiene garantizado que tiene una solución única»", 960, 320, tam=46, c="tinta", w="sb", en="a", id="cu"),
    U("cu", tachado=1.0, en="b", dur=0.8), P("con separación perfecta NO hay máximo", 960, 470, "rojo", 50, en="b", t=0.6),
    T("grabación de la clase", 1860, 1030, tam=26, c="gris", w="sb", anc="rm", en="a"), sec="8. Alerta: ¿solución única?")
v.b("Un ejemplo pequeño: seis puntos, los tres negativos a la izquierda y los tres positivos a la derecha. [[a]]Con beta igual a uno, la log verosimilitud es menos 0.978. [[b]]Con dos, menos 0.295. Con cinco, menos 0.014. [[c]]Con diez, menos 0.00009. Siempre sube hacia cero, sin detenerse. En la práctica, el programa se detiene en un valor arbitrario.",
    *[T(f"β = {b}", 520 + 300 * i, 640, tam=40, c="gris", w="sb", en=("a" if i == 0 else ("b" if i < 3 else "c"))) for i, b in enumerate(BETAS)],
    *[T((f"{SEP[b]:.3f}" if b != "10" else f"{SEP[b]:.5f}").replace("-", "−"), 520 + 300 * i, 720, tam=44, c="violeta", w="b", en=("a" if i == 0 else ("b" if i < 3 else "c")),
        t=0.3) for i, b in enumerate(BETAS)],
    P("la log-verosimilitud sube sin detenerse → el MLE no existe", 960, 860, "ambar", 44, en="c", t=1.5), ilustrativo(1000))

# ---- 9. El error típico
v.b("El error típico de esta parte: [[a]]elegir el umbral mirando la prueba. [[b]]Si pruebas varios umbrales en los 171 casos y te quedas con el que dio mejor, ya usaste la prueba para decidir. [[c]]La diapositiva once lo dice: mirar la prueba para decidir la convierte en validación, y su número deja de ser una evaluación honesta.",
    CL(), K(330, 230, "cruz", 70, en="a"), T("elegir el umbral mirando la prueba", 410, 230, tam=48, c="rojo", anc="lm", en="a"),
    T("probar umbrales en los 171 casos y quedarse con el mejor…", 960, 400, tam=42, c="tinta", w="sb", en="b"),
    P("…convierte la prueba en validación (diapositiva 11)", 960, 560, "ambar", 48, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si un coeficiente vale menos 0.7, ¿qué les pasa a los odds cuando esa variable sube una unidad?",
    CL(), P("β = −0.7  →  los odds se multiplican por… ?", 960, 400, "violeta", 54, en="a"), sec="10. Piensa un segundo", pausa=0.8)
v.b("Se multiplican por e a la menos 0.7, [[a]]que es 0.497. [[b]]Es decir, los odds se reducen casi a la mitad. Un coeficiente negativo baja los odds; uno positivo los sube.",
    T("e^(−0.7) =", 940, 560, tam=56, c="tinta", w="sb", anc="rm", en="a"), N(0.497, 970, 560, "{:.3f}", 64, "violeta", anc="lm", en="a", t=0.4),
    P("los odds se reducen casi a la mitad", 960, 720, "teal", 48, en="b"), ilustrativo(1000))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el umbral es una decisión: bajarlo gana sensibilidad, subirlo gana especificidad, y se elige con validación. [[b]]Un coeficiente multiplica los odds por e elevado a beta, no la probabilidad. [[c]]Y hay que leerlo en la escala de su variable: para la suavidad, de a 0.01.",
    CL(), P("umbral: decisión elegida con validación", 960, 300, "ambar", 48, en="a"),
    P("e^β multiplica los ODDS, no la probabilidad", 960, 460, "violeta", 48, en="b"),
    P("leer el coeficiente en la escala de su variable", 960, 620, "teal", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]las máquinas de vectores de soporte. El hiperplano, el margen, y los puntos que lo sostienen.",
    CL(), siguiente(5, "SVM: hiperplano, margen y vectores de soporte"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
