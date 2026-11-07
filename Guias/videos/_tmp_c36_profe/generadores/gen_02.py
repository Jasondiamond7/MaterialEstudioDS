import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, k1, cien

OUT = out("guion_c36_02_bagging.json")
v = V("Bagging vs boosting")
BB = FG["bb"]
CU = D["curvas"]

v.portada("Bagging vs boosting", "Mismos árboles, filosofías opuestas",
          "Segunda parada: bagging frente a boosting. Los dos combinan árboles, pero resuelven problemas distintos.")

# ---- 1. Hook
v.b("En la clase anterior armamos un Random Forest: trescientos árboles promediados. [[a]]Boosting también combina muchos árboles. [[b]]Entonces, ¿no es lo mismo? [[c]]No. Usan los mismos ingredientes, pero con filosofías opuestas.",
    CL(), T("Random Forest: 300 árboles promediados", 960, 280, tam=52, c="azul", ent="subir"),
    T("boosting: también muchos árboles…", 960, 420, tam=52, c="ambar", en="a"), T("¿no es lo mismo?", 960, 580, tam=76, c="tinta", en="b"),
    P("mismos ingredientes · filosofías opuestas", 960, 760, "violeta", 52, en="c"), sec="1. ¿No es lo mismo?")

# ---- 2. Analogía: comité vs grupo de estudio
v.b("Una analogía. [[a]]Bagging es como un comité de expertos que trabajan por separado: cada uno opina sin mirar al otro, y al final se promedia. [[b]]Boosting es como un grupo de estudio en cadena: cada estudiante revisa los errores del anterior y se concentra en eso.",
    CL(), T("dos formas de trabajar en equipo", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "comité", "cada experto trabaja solo; al final se promedia", "a"),
              (1360, "ambar", "grupo de estudio", "cada uno revisa los errores del anterior", "b")], y=480, w=640, h=360, tam_t=60),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía")
v.b("El comité funciona en paralelo, todos a la vez. [[a]]El grupo de estudio, en secuencia: el segundo no puede empezar hasta que el primero termine.",
    P("en paralelo", 560, 780, "azul", 48), P("en secuencia", 1360, 780, "ambar", 48, en="a"),
    L(1180, 860, 1540, 860, "ambar", 6, True, en="a", t=0.4))

# ---- 3. La diapositiva 4
v.b("La diapositiva cuatro lo resume en dos columnas. [[a]]Bagging: modelos en paralelo, cada uno con una muestra distinta, y las predicciones se promedian o votan. [[b]]Boosting: modelos en secuencia, cada etapa usa el error acumulado, y las contribuciones se suman.",
    CL(), T("bagging y boosting · diapositiva 4", 960, 150, tam=50, c="tinta", ent="subir"),
    Bc(520, 520, 760, 560, "blanco", "azul", 5, 30, sombra=True, en="a"), T("BAGGING", 520, 300, tam=50, c="azul", en="a", t=0.1),
    T("modelos en paralelo", 520, 410, tam=38, c="tinta", w="sb", en="a", t=0.3), T("cada uno ve una muestra distinta", 520, 490, tam=38, c="tinta", w="sb", en="a", t=0.6),
    T("se promedian o votan", 520, 570, tam=38, c="tinta", w="sb", en="a", t=0.9),
    Bc(1400, 520, 760, 560, "blanco", "ambar", 5, 30, sombra=True, en="b"), T("BOOSTING", 1400, 300, tam=50, c="ambar", en="b", t=0.1),
    T("modelos en secuencia", 1400, 410, tam=38, c="tinta", w="sb", en="b", t=0.3), T("cada etapa usa el error acumulado", 1400, 490, tam=38, c="tinta", w="sb", en="b", t=0.6),
    T("las contribuciones se suman", 1400, 570, tam=38, c="tinta", w="sb", en="b", t=0.9), sec="3. La diapositiva 4")
v.b("Y el objetivo típico de cada uno: [[a]]bagging reduce la varianza. [[b]]Boosting reduce el sesgo, y el error en general.",
    P("objetivo: ↓ VARIANZA", 520, 700, "azul", 48, en="a"), P("objetivo: ↓ SESGO y error", 1400, 700, "ambar", 48, en="b"),
    T("diapositiva 4", 1860, 1030, tam=26, c="gris", w="sb", anc="rm"))

# ---- 4. Sesgo vs varianza: los dardos
def diana(cx, en=None):
    return [Bc(cx, 560, 420, 420, "blanco", "linea", 4, 210, en=en), Bc(cx, 560, 260, 260, "blanco", "linea", 4, 130, en=en),
            Bc(cx, 560, 100, 100, "rojoclaro", "rojo", 4, 50, en=en)]


DISP = [(-120, -60), (90, -130), (130, 80), (-60, 140), (-150, 40), (40, 30)]
SESG = [(150, -150), (180, -120), (130, -110), (170, -175), (205, -140), (145, -185)]
v.b("¿Qué significan varianza y sesgo? Piensa en dardos. [[a]]Un modelo con varianza alta tira dardos que caen alrededor del centro, pero muy dispersos. [[b]]Un modelo con sesgo alto tira dardos muy juntos, pero lejos del centro.",
    CL(), T("varianza y sesgo, con dardos", 960, 160, tam=52, c="tinta", ent="subir"),
    diana(560, en="a"), T("varianza alta", 560, 300, tam=44, c="azul", en="a"),
    *[Bc(560 + dx, 560 + dy, 30, 30, "azul", "blanco", 3, 15, en="a", t=0.15 * i, id=f"dv{i}") for i, (dx, dy) in enumerate(DISP)],
    diana(1360, en="b"), T("sesgo alto", 1360, 300, tam=44, c="ambar", en="b"),
    *[Bc(1360 + dx, 560 + dy, 30, 30, "ambar", "blanco", 3, 15, en="b", t=0.15 * i, id=f"ds{i}") for i, (dx, dy) in enumerate(SESG)],
    ilustrativo(1000, texto="analogía"), sec="4. Sesgo y varianza")
v.b("Bagging arregla el primer problema: [[a]]si promedias muchos tiros dispersos, el promedio cae muy cerca del centro. [[b]]Boosting arregla el segundo: cada corrección empuja el grupo hacia el centro.",
    *[U(f"dv{i}", x=560 + dx * 0.15, y=560 + dy * 0.15, en="a", t=0.05 * i, dur=0.8) for i, (dx, dy) in enumerate(DISP)],
    P("promediar → cerca del centro", 560, 850, "azul", 40, en="a", t=0.6),
    *[U(f"ds{i}", x=1360 + (dx - 165) * 0.6, y=560 + (dy + 147) * 0.6, en="b", t=0.05 * i, dur=1.0) for i, (dx, dy) in enumerate(SESG)],
    P("corregir → el grupo se mueve al centro", 1360, 850, "ambar", 40, en="b", t=0.8))
v.b("Por eso cada familia usa un árbol distinto. [[a]]El bosque usa árboles profundos: poco sesgo, mucha varianza, y el promedio calma la varianza. [[b]]Boosting usa árboles chicos, de profundidad dos a seis: mucho sesgo, poca varianza, y la suma va quitando el sesgo.",
    CL(), T("cada familia, su tipo de árbol", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "árboles PROFUNDOS", "poco sesgo, mucha varianza → se promedian", "a"),
              (1360, "ambar", "árboles CHICOS", "mucho sesgo, poca varianza → se suman", "b")], y=480, w=680, h=360, tam_t=52))

# ---- 5. Datos reales: la curva en Ames
xs = list(range(1, 301))
v.b("Veámoslo con datos reales: Ames, error de validación según el número de árboles. [[a]]Un solo árbol de profundidad tres, como referencia, se equivoca en 41 200 dólares. [[b]]Un Random Forest con un árbol profundo, en 39 300. Con cien árboles baja a 23 100, y ahí se estanca.",
    CL(), T("Ames · RMSE de validación según el número de árboles", 960, 150, tam=44, c="tinta", ent="subir"),
    G(300, 250, 1380, 820, 0, 300, 15, 75, [dict(xs=xs, ys=[c / 1000 for c in CU["rf"]], color="azul", grosor=6, revela=0.0),
                                            dict(xs=xs, ys=[c / 1000 for c in CU["gb"]], color="ambar", grosor=6, revela=0.0),
                                            dict(xs=[0, 300], ys=[BB["tree3"] / 1000] * 2, color="gris", grosor=4, revela=0.0, lapiz=False)],
      xticks=[1, 50, 100, 150, 200, 250, 300], yticks=[15, 30, 45, 60, 75], yfmt="{:.0f} mil", xlabel="número de árboles", id="gc"),
    U("gc", **{"series.2.revela": 1.0}, en="a", dur=0.8), T(f"1 árbol de prof. 3: {k1(BB['tree3'])}", 1660, 330, tam=34, c="gris", w="b", en="a", t=0.3),
    U("gc", **{"series.0.revela": 1.0}, en="b", dur=2.5), T(f"bosque, 1 árbol: {k1(BB['rf1'])}", 1660, 420, tam=34, c="azul", w="b", en="b", t=0.2),
    T(f"bosque, 300: {k1(BB['rf300'])}", 1660, 500, tam=34, c="azul", w="b", en="b", t=2.6), real(1050), sec="5. La curva en Ames")
v.b("Ahora boosting, con árboles de profundidad tres y eta 0.1. [[a]]Con un árbol, el error es enorme: 69 000. [[b]]Pero sigue bajando, árbol tras árbol, y pasa por debajo del bosque. [[c]]Con trescientos árboles llega a 21 800.",
    U("gc", **{"series.1.revela": 0.004}, en="a", dur=0.1), T(f"boosting, 1 árbol: {k1(BB['gb1'])}", 1660, 600, tam=34, c="ambar", w="b", en="a", t=0.2),
    U("gc", **{"series.1.revela": 1.0}, en="b", dur=2.8), T(f"boosting, 300: {k1(BB['gb300'])}", 1660, 680, tam=34, c="ambar", w="b", en="c"),
    P("el bosque se estanca · boosting sigue mejorando", 840, 960, "verde", 40, en="c", t=0.5))

# ---- 6. Piensa un segundo
v.b("Piensa un segundo. [[a]]¿Por qué boosting con un solo árbol se equivoca en 69 000 dólares, mucho peor que un árbol suelto de la misma profundidad, que se equivoca en 41 200?",
    CL(), P(f"boosting, 1 árbol: {k1(BB['gb1'])}", 960, 300, "ambar", 54, en="a"), P(f"árbol suelto, prof. 3: {k1(BB['tree3'])}", 960, 430, "gris", 54, en="a", t=0.4),
    T("¿por qué tanto peor?", 960, 620, tam=76, c="tinta", en="a", t=0.8), sec="6. Piensa un segundo", pausa=0.6)
v.b("Por eta. [[a]]Boosting parte del promedio, y al primer árbol lo suma multiplicado por 0.1: solo avanza un diez por ciento del camino. [[b]]Es un paso chico a propósito. [[c]]El árbol suelto, en cambio, da el paso completo de una vez. Boosting lo alcanza y lo supera después, sumando muchos pasos chicos.",
    CL(), caja_formula("F₁ = promedio + 0.1 · h₁", y=300, c="ambar", tam=62, w=1000),
    T("solo un 10 % del primer árbol", 960, 470, tam=48, c="tinta", w="sb", en="a"), P("un paso chico, a propósito", 960, 600, "ambar", 50, en="b"),
    T("el árbol suelto da el paso completo de una vez", 960, 740, tam=42, c="gris", w="sb", en="c"))

# ---- 7. Paralelo vs secuencial (tiempo)
v.b("La secuencia tiene un costo. [[a]]Imagina mil árboles y un computador con diez núcleos. [[b]]En bagging, los diez núcleos entrenan diez árboles a la vez: el tiempo es como el de unos cien árboles. [[c]]En boosting, el árbol dos necesita el resultado del árbol uno: se entrenan de a uno, y el tiempo es el de los mil.",
    CL(), T("mil árboles, diez núcleos", 960, 160, tam=52, c="tinta", ent="subir"), T("tiempo relativo", 960, 230, tam=34, c="gris", w="sb", en="a"),
    T("bagging", 360, 420, tam=44, c="azul", anc="rm", en="b"), Bx(400, 385, 400 + 130, 455, "azul", None, 0, 12, grow="x", en="b", t=0.2),
    T("≈ 100", 560, 420, tam=40, c="azul", w="b", anc="lm", en="b", t=0.8),
    T("boosting", 360, 560, tam=44, c="ambar", anc="rm", en="c"), Bx(400, 525, 400 + 1300, 595, "ambar", None, 0, 12, grow="x", en="c", t=0.2, dur=1.6),
    T("≈ 1.000", 1720, 640, tam=40, c="ambar", w="b", anc="rm", en="c", t=1.8), ilustrativo(1000), sec="7. El costo de la secuencia")
v.b("Un matiz importante: [[a]]XGBoost y LightGBM sí paralelizan, pero dentro de cada árbol, al buscar los cortes. Lo que no se puede paralelizar es la cadena de árboles.",
    P("XGBoost y LightGBM paralelizan DENTRO de cada árbol", 960, 780, "verde", 44, en="a"),
    T("la cadena de árboles sigue siendo secuencial", 960, 870, tam=38, c="gris", w="sb", en="a", t=0.6))

# ---- 8. El error típico
v.b("El error típico: [[a]]creer que, como en el bosque, agregar árboles a boosting nunca hace daño. [[b]]En el bosque es cierto: la curva se aplana y no sube. [[c]]En boosting, cada árbol nuevo sigue ajustando los residuos, y puede terminar ajustando ruido. En Ames, el mínimo llegó a los 250 árboles, y a los trescientos el error ya había subido un poco.",
    CL(), K(330, 280, "cruz", 80, en="a"), T("«más árboles nunca hace daño»", 420, 280, tam=48, c="rojo", anc="lm", en="a"),
    T("bosque: cierto, la curva se aplana", 420, 420, tam=42, c="azul", w="sb", anc="lm", en="b"),
    T("boosting: puede empezar a ajustar ruido", 420, 520, tam=42, c="ambar", w="sb", anc="lm", en="c"),
    P(f"Ames: mínimo en {BB['gb_best']} árboles: {round(BB['gb_min']):,} USD · con 300: {round(BB['gb300']):,} USD".replace(",", "."), 960, 680, "ambar", 40, en="c", t=1.2),
    real(1050), sec="8. El error típico")
v.b("Y la diapositiva cuatro advierte algo más: [[a]]la secuencia hace a boosting más sensible a los hiperparámetros, al ruido y a la filtración de información. [[b]]Una etiqueta mal puesta se convierte en el caso más difícil, y el modelo se obsesiona con ella.",
    P("más sensible: hiperparámetros · ruido · filtración", 960, 850, "rojo", 44, en="a"),
    T("una etiqueta mal puesta = el caso «más difícil»", 960, 950, tam=38, c="gris", w="sb", en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]bagging, en paralelo, con árboles profundos que se promedian: baja la varianza. [[b]]Boosting, en secuencia, con árboles chicos que se suman: baja el sesgo. [[c]]Boosting da pasos chicos, puede superar al bosque, y necesita que alguien le diga cuándo parar.",
    CL(), P("bagging: paralelo · profundos · promedio → ↓ varianza", 960, 300, "azul", 44, en="a"),
    P("boosting: secuencia · chicos · suma → ↓ sesgo", 960, 460, "ambar", 44, en="b"),
    P("pasos chicos · puede superar al bosque · hay que saber parar", 960, 620, "verde", 42, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el primer algoritmo de la familia, AdaBoost, y cómo usa pesos para señalar los casos difíciles.",
    CL(), siguiente(3, "AdaBoost: pesos, error y alfa"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
