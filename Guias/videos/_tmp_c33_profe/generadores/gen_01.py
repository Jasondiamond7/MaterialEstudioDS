import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util33 import D, out, tarjetas, flujo, siguiente, ilustrativo

OUT = out("guion_c33_01_introduccion.json")
v = V("Introducción: ¿con cuáles variables me quedo?")

v.portada("¿Con cuáles me quedo?", "Limpiar, construir y seleccionar variables sin hacer trampa",
          "Hola. Bienvenido a la serie de videos de la clase treinta y tres, con el profesor Eduardo Rodríguez. Hoy vas a aprender a decidir con qué variables se queda tu modelo, y a hacerlo sin hacer trampa.")

# ---- 1. Hook: el entrenador
v.b("Imagina que eres entrenador de fútbol. [[a]]Tienes cincuenta jugadores convocados, pero solo once entran a la cancha. [[b]]Si eliges a los once que mejor juegan solos, [[c]]puedes terminar con cuatro arqueros en el equipo.",
    CL(), P("50 convocados · 11 a la cancha", 960, 220, "violeta", 54, ent="subir"),
    Pt([(50, [380, 330, 1540, 620], "#94A3B8")], 16, id="jug", visibles=50, en="a", dur=1.2),
    U("jug", grupos=[dict(n=39, rect=[380, 330, 1540, 620], color="#CBD5E1"), dict(n=11, rect=[560, 720, 1360, 800], color="azul")], en="b", dur=1.2),
    U("jug", grupos=[dict(n=39, rect=[380, 330, 1540, 620], color="#CBD5E1"), dict(n=7, rect=[560, 720, 1100, 800], color="azul"),
                     dict(n=4, rect=[1180, 720, 1360, 800], color="ambar")], en="c", dur=0.8),
    P("4 arqueros", 1500, 880, "ambar", 44, en="c", t=0.5), ilustrativo(1010, texto="analogía"), sec="1. La gran idea")
v.b("Cada jugador es bueno, pero el equipo es malo. [[a]]Elegir variables para un modelo es exactamente lo mismo. [[b]]No buscas las mejores de a una: buscas el mejor conjunto.",
    CL(), T("cada jugador es bueno · el equipo es malo", 960, 300, tam=56, c="gris", ent="subir"),
    P("las mejores de a una", 560, 520, "rojo", 56, en="b"), K(560, 640, "cruz", 70, en="b", t=0.3),
    P("el mejor conjunto", 1360, 520, "verde", 56, en="b", t=0.6), K(1360, 640, "check", 70, en="b", t=0.9))
v.b("La gran idea de hoy: más variables no es un mejor modelo. [[a]]El trabajo real, que según el profesor es el setenta a ochenta por ciento de un proyecto, [[b]]es limpiar con criterio de negocio, construir variables con sentido, y seleccionar sin hacer trampa.",
    CL(), T("más variables ≠ mejor modelo", 960, 200, tam=64, c="tinta", ent="subir"),
    P("70–80 % de un proyecto", 960, 330, "ambar", 44, en="a"),
    tarjetas([(360, "teal", "limpiar", "con criterio de negocio", "b"), (960, "violeta", "construir", "variables con sentido", "b"),
              (1560, "azul", "seleccionar", "sin hacer trampa", "b")], y=620, h=340))
v.b("Y la trampa más común es invisible. [[a]]Si una selección mira el target fuera de la validación cruzada, [[b]]el AUC que reportas es inventado.",
    CL(), P("la trampa invisible", 960, 260, "rojo", 60, ent="subir"),
    T("seleccionar mirando el target fuera de la validación cruzada", 960, 470, tam=46, c="tinta", w="sb", mw=1500, en="a"),
    L(960, 540, 960, 640, "gris", 6, True, en="b"), P("AUC inventado", 960, 720, "rojo", 70, en="b", t=0.3))

# ---- 2. De dónde venimos
v.b("¿De dónde venimos? [[a]]La clase treinta preguntó si los datos eran confiables. [[b]]La treinta y uno, cómo aprende un modelo. [[c]]La treinta y dos, en qué forma se los entregamos. [[d]]Y hoy, la treinta y tres pregunta: ¿con cuáles me quedo?",
    CL(), *[[Bc(285 + 450 * i, 520, 400, 380, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"),
             T(f"Clase {n}", 285 + 450 * i, 400, tam=50, c=c, en=k, t=0.1),
             T(q, 285 + 450 * i, 560, tam=36, c="tinta", w="sb", mw=330, en=k, t=0.2)]
            for i, (n, c, q, k) in enumerate(((30, "gris", "¿los datos son confiables?", "a"), (31, "verde", "¿cómo aprende un modelo?", "b"),
                                              (32, "azul", "¿en qué forma se los entrego?", "c"), (33, "ambar", "¿con cuáles me quedo?", "d")))],
    R(1635, 520, 430, 410, "ambar", 34, en="d", t=0.4), sec="2. De dónde venimos")
v.b("La clase treinta y dos dejó dos caminos abiertos. [[a]]Uno, extraer: combinar variables en componentes nuevos, como PCA. [[b]]El otro, seleccionar: quedarse con algunas de las variables originales, que se pueden explicar. [[c]]Hoy tomamos el segundo camino.",
    CL(), tarjetas([(560, "azul", "extraer", "componentes nuevos · PCA, t-SNE", "a"), (1360, "ambar", "seleccionar", "conservar originales · filter, wrapper, embedded", "b")], w=720, h=380),
    T("clase 32", 560, 790, tam=34, c="azul", w="sb", en="a", t=0.4), P("hoy · clase 33", 1360, 790, "ambar", 40, en="c"))

# ---- 3. Los datos
v.b("Los datos son los mismos de las clases anteriores: [[a]]la base de fuga de clientes de una telco. [[b]]Tres mil ciento cincuenta clientes, [[c]]doce variables predictoras, [[d]]y cuatrocientos noventa y cinco que se fueron: el quince coma siete por ciento.",
    CL(), P("Base de datos Churn.xlsx · telco", 960, 210, "violeta", 50, ent="subir", en="a"),
    *[[Bc(x, 510, 440, 300, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), (T(val, x, 480, tam=110, c=c, en=k, t=0.15) if isinstance(val, str) else N(val, x, 480, fmt, 110, c, suf=suf, en=k, t=0.15, dur=0.9)),
       T(lab, x, 600, tam=34, c="gris", w="sb", en=k, t=0.25)]
      for x, c, val, fmt, suf, lab, k in ((440, "violeta", "3.150", "", "", "clientes", "b"), (960, "azul", 12, "{:.0f}", "", "predictoras", "c"),
                                          (1480, "rojo", D["tasa"], "{:.1f}", " %", f"se fuga · {D['n_fuga']} clientes", "d"))],
    T("datos reales de la clase (verificados)", 960, 800, tam=32, c="gris", w="sb", en="d", t=0.8), sec="3. Los datos")
v.b("Con tan poca fuga, ojo con la métrica. [[a]]Un modelo que dice «nadie se va» acierta el ochenta y cuatro coma tres por ciento, y no sirve para nada. [[b]]Por eso toda la clase mide con AUC: [[c]]cero coma cinco es azar, y uno es perfecto.",
    CL(), T("«nadie se va»", 760, 300, tam=64, c="tinta", ent="subir"), N(D["acc_nadie"], 1260, 300, "{:.1f}", 90, "rojo", suf=" %", en="a", dur=0.9),
    T("de exactitud · inútil", 1260, 390, tam=34, c="rojo", w="sb", en="a", t=0.5),
    P("se mide con AUC", 960, 520, "verde", 56, en="b"),
    L(460, 720, 1460, 720, "tinta", 5, en="c"),
    T("0.5 · azar", 460, 790, tam=40, c="gris", en="c", t=0.2), T("1.0 · perfecto", 1460, 790, tam=40, c="verde", en="c", t=0.5),
    Bc(460, 720, 26, 26, "gris", None, 0, 13, en="c", t=0.2), Bc(1460, 720, 26, 26, "verde", None, 0, 13, en="c", t=0.5))

# ---- 4. La ruta en números
v.b("La ruta de hoy, en números. [[a]]Primero, limpiar: la base trae cuarenta y tres valores imposibles. [[b]]Después, construir: de doce columnas pasamos a cincuenta candidatas. [[c]]Y al final, seleccionar.",
    CL(), T("la ruta en números", 960, 200, tam=48, c="gris", w="sb", ent="subir"),
    Bc(380, 500, 420, 300, "blanco", "teal", 5, 30, sombra=True, en="a", ent="subir"), N(D["imposibles"], 380, 470, "{:.0f}", 120, "teal", en="a", t=0.1),
    T("valores imposibles", 380, 590, tam=34, c="teal", w="sb", en="a", t=0.2),
    L(600, 500, 720, 500, "gris", 7, True, en="b"),
    Bc(960, 500, 420, 300, "blanco", "violeta", 5, 30, sombra=True, en="b", ent="subir"), T("12 → 50", 960, 470, tam=100, c="violeta", en="b", t=0.1),
    T("candidatas", 960, 590, tam=34, c="violeta", w="sb", en="b", t=0.2),
    L(1180, 500, 1300, 500, "gris", 7, True, en="c"),
    Bc(1540, 500, 420, 300, "blanco", "azul", 5, 30, sombra=True, en="c", ent="subir"), T("¿cuáles?", 1540, 470, tam=90, c="azul", en="c", t=0.1),
    T("filter · wrapper · embedded", 1540, 590, tam=27, c="azul", w="sb", en="c", t=0.2), sec="4. La ruta en números")
v.b("Y el final de la clase es una sorpresa. [[a]]Con cien clientes y mil variables de puro ruido, una selección mal ubicada da AUC cero coma ochocientos ochenta y dos. [[b]]La verdad es cero coma cinco. Treinta y ocho puntos inventados.",
    CL(), P("1.000 variables de puro ruido", 960, 230, "gris", 50, ent="subir"),
    Bx(560, 860 - D["ruido"]["mal"] * 500, 800, 860, "rojo", None, 0, 12, grow="y", en="a"), N(D["ruido"]["mal"], 680, 860 - D["ruido"]["mal"] * 500 - 45, "{:.3f}", 64, "rojo", en="a", t=0.3, dur=0.8),
    T("selección mal ubicada", 680, 910, tam=32, c="rojo", w="sb", en="a"),
    Bx(1120, 860 - 0.5 * 500, 1360, 860, "verde", None, 0, 12, grow="y", en="b"), N(0.5, 1240, 860 - 0.5 * 500 - 45, "{:.3f}", 64, "verde", en="b", t=0.3, dur=0.6),
    T("la verdad", 1240, 910, tam=32, c="verde", w="sb", en="b"), P("+38 puntos inventados", 1500, 420, "rojo", 44, en="b", t=0.9),
    L(460, 860, 1460, 860, "tinta", 4))

# ---- 5. Error típico
v.b("Antes de seguir, el primer error típico: [[a]]confundir limpiar con seleccionar. [[b]]Si una variable trae una edad de doscientos treinta y tres años, eso es calidad de datos, y va antes.",
    CL(), T("limpiar  =  seleccionar", 960, 380, tam=80, c="tinta", ent="subir", id="lims"), U("lims", tachado=1.0, color="gris", en="a", t=0.6, dur=0.6),
    P("edad = 233 años → calidad de datos, va antes", 960, 620, "ambar", 50, en="b"), sec="5. El error típico")
v.b("Si mezclas las dos cosas, [[a]]terminas descartando variables buenas solo porque estaban sucias.",
    P("descartas variables buenas por estar sucias", 960, 780, "rojo", 50, en="a"), K(1610, 780, "cruz", 60, en="a", t=0.6))

# ---- 6. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tienes cincuenta candidatas, y te quedas con las cuatro que más información aportan, cada una por separado. [[b]]¿Es ese el mejor equipo de cuatro?",
    CL(), T("50 candidatas → las 4 mejores de a una", 960, 330, tam=58, c="violeta", en="a"), T("¿es el mejor equipo de 4?", 960, 560, tam=72, c="tinta", en="b"),
    sec="6. Piensa un segundo", pausa=0.6)
v.b("No. [[a]]Las cuatro mejores de a una dan AUC cero coma ochocientos cuatro. [[b]]Cuatro elegidas evitando redundancia dan cero coma ochocientos noventa y tres. [[c]]Igual que los cuatro arqueros: lo verás en el video siete.",
    CL(), P("no", 960, 220, "rojo", 70, ent="subir"),
    Bc(560, 520, 600, 300, "blanco", "rojo", 5, 30, sombra=True, en="a", ent="subir"), N(D["top4"]["auc"], 560, 490, "{:.3f}", 110, "rojo", en="a", t=0.2),
    T("top-4 de a una", 560, 610, tam=36, c="rojo", w="sb", en="a", t=0.3),
    Bc(1360, 520, 600, 300, "blanco", "verde", 5, 30, sombra=True, en="b", ent="subir"), N(D["manual4"]["auc"], 1360, 490, "{:.3f}", 110, "verde", en="b", t=0.2),
    T("4 sin redundancia", 1360, 610, tam=36, c="verde", w="sb", en="b", t=0.3),
    T("datos reales · validación cruzada de 5 folds", 960, 760, tam=32, c="gris", w="sb", en="b", t=0.6), P("los 4 arqueros → video 7", 960, 870, "ambar", 44, en="c"))

# ---- 7. Mapa
ETQ = [("Limpiar con criterio de negocio", "teal"), ("Construir variables: 12 → 50", "violeta"), ("Irrelevante, redundante, en conjunto", "rojo"),
       ("El caso XOR", "rojo"), ("Taxonomía y métodos de filtro", "azul"), ("Cuatro contra cuatro", "azul"),
       ("Wrapper y embedded: RFE y Lasso", "verde"), ("La trampa: dónde vive la selección", "ambar"), ("Repaso y checklist", "gris")]


def tj(i, t):
    x, y = 330 + 630 * (i % 3), 300 + 230 * (i // 3)
    c = ETQ[i][1]
    return [Bc(x, y, 580, 190, "blanco", c, 5, 26, sombra=True, en="a", t=t), T(str(i + 2), x - 225, y, tam=72, c=c, en="a", t=t + 0.05),
            T(ETQ[i][0], x + 45, y, tam=32, c="tinta", w="sb", mw=430, en="a", t=t + 0.1)]


v.b("Este es el mapa de la serie. [[a]]Después de esta introducción vienen nueve paradas. [[b]]Limpiar y construir. Los tres motivos por los que una variable sobra. Filtros, wrapper y embedded. Y al final, la trampa, y el repaso.",
    CL(), [tj(i, 0.18 * i) for i in range(9)], R(960, 760, 600, 210, "ambar", 30, en="b", t=4.6), sec="7. El mapa de la serie")

# ---- 8. Para llevar
v.b("Para llevarte de esta introducción: [[a]]se busca el mejor conjunto, no las mejores de a una. [[b]]Primero se limpia, después se construye, y recién ahí se selecciona. [[c]]Y todo lo que mira el target va dentro de la validación.",
    CL(), P("el mejor conjunto, no las mejores de a una", 960, 300, "violeta", 50, en="a"), P("limpiar → construir → seleccionar", 960, 480, "teal", 54, en="b"),
    P("lo que mira el target, dentro de la validación", 960, 660, "rojo", 50, en="c"), sec="8. Para llevar")
v.b("Vamos con la primera parada: [[a]]limpiar con criterio de negocio.",
    CL(), siguiente(2, "Limpiar con criterio de negocio"), sec="9. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
