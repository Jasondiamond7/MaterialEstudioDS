import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, flujo, mils

OUT = out("guion_c35_01_inestable.json")
v = V("Un árbol inestable")
BO = FG["boot"]
RA = BO["raices"]
AP = D["prop_auc_promedio"]

v.portada("Un árbol inestable", "Por qué un solo árbol no basta, y qué vamos a construir hoy",
          "Bienvenido a la clase treinta y cinco: Random Forest y Bagging. Hoy vas a aprender a convertir un modelo nervioso, que cambia con cualquier detalle, en uno estable, combinando muchos.")

# ---- 1. Hook
v.b("Imagina que entrenas un árbol de decisión con tus datos. [[a]]La primera pregunta del árbol es: ¿la edad es menor que 35? [[b]]Ahora cambias apenas unas pocas filas, y lo vuelves a entrenar. [[c]]Y la primera pregunta ahora es otra: ¿el ingreso es menor que 820?",
    CL(), Bc(520, 470, 640, 520, "blanco", "azul", 5, 30, sombra=True, en="a", ent="subir"), T("muestra A", 520, 260, tam=40, c="azul", en="a", t=0.1),
    P("¿edad < 35?", 520, 360, "azul", 44, en="a", t=0.2), L(520, 400, 380, 500, "grisclaro", 5, en="a", t=0.3), L(520, 400, 660, 500, "grisclaro", 5, en="a", t=0.3),
    P("¿ingreso < 900?", 380, 550, "gris", 34, en="a", t=0.4), P("compra", 680, 550, "verde", 34, en="a", t=0.4),
    T("+ pocas filas\ndistintas", 960, 440, tam=30, c="gris", w="sb", en="b"), L(860, 520, 1060, 520, "gris", 6, True, en="b", t=0.2),
    Bc(1400, 470, 640, 520, "blanco", "rojo", 5, 30, sombra=True, en="c", ent="subir"), T("muestra B", 1400, 260, tam=40, c="rojo", en="c", t=0.1),
    P("¿ingreso < 820?", 1400, 360, "rojo", 44, en="c", t=0.2), L(1400, 400, 1260, 500, "grisclaro", 5, en="c", t=0.3), L(1400, 400, 1540, 500, "grisclaro", 5, en="c", t=0.3),
    P("¿antigüedad < 2?", 1260, 550, "gris", 34, en="c", t=0.4), P("compra", 1560, 550, "verde", 34, en="c", t=0.4),
    ilustrativo(1000, texto="ejemplo ilustrativo (diapositiva 3)"), sec="1. Un árbol que cambia de opinión")
v.b("Mismo problema, mismos clientes casi, [[a]]y dos árboles que razonan distinto. [[b]]Eso tiene nombre: el árbol es inestable. Y hoy vamos a ver cómo se arregla.",
    P("dos árboles, dos razonamientos", 960, 800, "violeta", 48, en="a"), T("inestable", 960, 910, tam=64, c="rojo", en="b"))

# ---- 2. Analogía
v.b("Piensa en una veleta. [[a]]Una ráfaga pequeña, y apunta para otro lado. [[b]]Un árbol profundo es una veleta: aprende tanto detalle de su muestra que cualquier cambio lo hace girar.",
    CL(), T("una veleta", 960, 200, tam=60, c="tinta", ent="subir"),
    L(960, 760, 960, 420, "tinta", 10, en="a"), L(840, 420, 1080, 420, "azul", 14, True, en="a", t=0.3, id="vel1"),
    U("vel1", x0=1060, y0=380, x1=860, y1=460, color="rojo", en="a", t=1.0, dur=0.5, ease="back"),
    T("ráfaga pequeña → otra dirección", 960, 840, tam=40, c="gris", w="sb", en="a", t=0.6),
    P("árbol profundo = veleta", 960, 940, "violeta", 48, en="b"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: la veleta")
v.b("¿Cómo sabes hacia dónde sopla de verdad el viento? [[a]]No miras una veleta: miras cien, puestas en lugares distintos, y promedias. [[b]]Esa es toda la idea de la clase: muchos modelos distintos, y luego votan o promedian.",
    CL(), T("¿hacia dónde sopla de verdad?", 960, 200, tam=54, c="tinta", ent="subir"),
    Pt([(100, [460, 300, 1460, 640], "azul")], 13, visibles=100, en="a", dur=1.4),
    P("100 veletas distintas → promedio", 960, 760, "verde", 50, en="a", t=1.0),
    P("muchos modelos distintos + votar o promediar", 960, 890, "violeta", 46, en="b"))

# ---- 3. Bajo sesgo, alta varianza
v.b("Primero, el diagnóstico. [[a]]Un árbol profundo tiene bajo sesgo: puede capturar cualquier forma, umbrales, interacciones, relaciones no lineales. [[b]]Pero tiene alta varianza: su forma depende demasiado de la muestra que le tocó.",
    CL(), T("el diagnóstico del árbol profundo", 960, 170, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "bajo sesgo", "captura umbrales, interacciones y relaciones no lineales", "a"),
              (1360, "rojo", "alta varianza", "su forma depende de la muestra que le tocó", "b")], y=520, w=680, h=380),
    sec="3. Bajo sesgo, alta varianza")
v.b("Y aquí está la pregunta guía de la clase, tal cual la plantea el profesor: [[a]]¿cómo podemos reducir la inestabilidad de un árbol, sin renunciar a su capacidad para capturar relaciones no lineales?",
    CL(), P("pregunta guía", 960, 260, "ambar", 44, ent="subir"),
    T("¿Cómo reducir la inestabilidad de un árbol sin renunciar a su capacidad para capturar relaciones no lineales?", 960, 480,
      tam=56, c="tinta", mw=1500, en="a"), T("diapositiva 2", 960, 700, tam=32, c="gris", w="sb", en="a", t=0.8))

# ---- 4. Medirlo con datos reales
v.b("Vamos a medir esa inestabilidad con datos reales. [[a]]Heart: 303 pacientes, 13 variables clínicas, y la pregunta es si hay enfermedad cardíaca. [[b]]El 46 % de los pacientes la tiene: 139 de 303.",
    CL(), T("Heart.xlsx", 960, 190, tam=60, c="teal", ent="subir"),
    T("303 pacientes · 13 variables clínicas · AHD = enfermedad cardíaca", 960, 300, tam=40, c="tinta", w="sb", en="a"),
    Pt([(139, [360, 420, 900, 760], "rojo"), (164, [1020, 420, 1560, 760], "gris")], 12, visibles=303, en="b", dur=1.4),
    T("139 sí (46 %)", 630, 820, tam=44, c="rojo", en="b", t=0.8), T("164 no", 1290, 820, tam=44, c="gris", en="b", t=0.8),
    real(1050), sec="4. Medirlo con datos reales")
v.b("Tomamos los 227 pacientes de entrenamiento. [[a]]Sacamos 200 muestras un poco distintas de ellos, y en cada una entrenamos un árbol sin podar. [[b]]Después preguntamos: ¿qué variable eligió cada árbol para su primera pregunta, la raíz?",
    CL(), T("227 pacientes de entrenamiento", 960, 200, tam=48, c="tinta", ent="subir"),
    flujo([("200 muestras distintas", "azul", "a"), ("200 árboles sin podar", "violeta", "a"), ("¿qué variable está en la raíz?", "teal", "b")],
          y=520, w=420, h=220, x0=440, paso=520, tam=42), real(1050))
RAIZ = [("Thal", RA["Thal"], "azul", "a"), ("ChestPain", RA["ChestPain"], "violeta", "b"), ("Ca", RA["Ca"], "teal", "c")]
v.b("El resultado. [[a]]En casi la mitad de los árboles, el 48.5 %, la raíz fue Thal, el resultado de una prueba de esfuerzo. [[b]]En el 30 %, fue ChestPain, el tipo de dolor de pecho. [[c]]Y en el 20 %, Ca, el número de vasos sanguíneos visibles.",
    CL(), T("variable de la raíz en 200 árboles bootstrap", 960, 170, tam=46, c="tinta", ent="subir"),
    *[[T(lab, 560, 330 + 150 * i, tam=46, c=c, anc="rm", en=m), Bx(600, 330 + 150 * i - 40, 600 + val * 1800, 330 + 150 * i + 40, c, None, 0, 12, grow="x", en=m, t=0.1),
       N(val * 100, 620 + val * 1800, 330 + 150 * i, "{:.1f}", 54, c, suf=" %", anc="lm", en=m, t=0.4)] for i, (lab, val, c, m) in enumerate(RAIZ)],
    real(1050))
v.b("Mismo hospital, mismos pacientes, solo otra muestra: [[a]]y el árbol cambia su primera pregunta en más de la mitad de los casos. [[b]]Y peor: si tomas dos de estos árboles al azar, discrepan en el diagnóstico del 26 % de los pacientes de prueba.",
    P("cambia la raíz en más de la mitad de los árboles", 960, 820, "rojo", 44, en="a"),
    P(f"dos árboles cualquiera discrepan en el {BO['desacuerdo'] * 100:.0f} % de los pacientes de prueba", 960, 930, "ambar", 40, en="b"))

# ---- 5. Ejemplo: un árbol contra el promedio
v.b("Ahora el ejemplo que resume la clase. [[a]]Medimos el AUC de cada uno de esos árboles en los 76 pacientes de prueba. [[b]]En promedio, un árbol suelto da 0.744. Y varía mucho: el peor da 0.586, el mejor 0.817.",
    CL(), T("AUC en los 76 pacientes de prueba", 960, 170, tam=48, c="tinta", ent="subir"),
    T("un árbol suelto", 560, 330, tam=44, c="gris", en="a"), N(D["prop_auc_un_arbol_media"], 560, 450, "{:.3f}", 110, "gris", en="b"),
    T(f"(entre {D['prop_auc_un_arbol_min']:.3f} y {D['prop_auc_un_arbol_max']:.3f})", 560, 560, tam=36, c="gris", w="sb", en="b", t=0.8),
    real(1050), sec="5. Un árbol contra el promedio")
v.b("Y ahora promediamos las probabilidades de esos mismos árboles. [[a]]Con 5 árboles, el AUC sube a 0.878. [[b]]Con 25, a 0.894. [[c]]Y con 100 árboles promediados, 0.905. Los mismos árboles mediocres, juntos, se vuelven buenos.",
    T("promedio de B árboles", 1360, 330, tam=44, c="verde", ent="subir"),
    T("B = 5", 1180, 440, tam=40, c="tinta", w="sb", anc="lm", en="a"), T(f"{AP['5']:.3f}", 1560, 440, tam=48, c="verde", anc="rm", en="a", t=0.2),
    T("B = 25", 1180, 530, tam=40, c="tinta", w="sb", anc="lm", en="b"), T(f"{AP['25']:.3f}", 1560, 530, tam=48, c="verde", anc="rm", en="b", t=0.2),
    T("B = 100", 1180, 620, tam=40, c="tinta", w="sb", anc="lm", en="c"), T(f"{AP['100']:.3f}", 1560, 620, tam=56, c="verde", anc="rm", en="c", t=0.2),
    P("mismos árboles mediocres → juntos, buenos", 960, 800, "violeta", 46, en="c", t=1.2))

# ---- 6. La ruta
v.b("Eso es un ensamblado: [[a]]varios modelos base, una función que agrega sus respuestas, y una sola predicción final. [[b]]Y funciona con una condición: que los modelos cometan errores distintos.",
    CL(), T("un ensamblado", 960, 170, tam=56, c="tinta", ent="subir"),
    *[[Bc(400, 330 + 150 * i, 300, 100, "blanco", "azul", 4, 20, en="a", t=0.15 * i, ent="pop"), T(f"modelo {k}", 400, 330 + 150 * i, tam=36, c="azul", en="a", t=0.15 * i + 0.05),
       L(560, 330 + 150 * i, 820, 480, "gris", 5, True, en="a", t=0.15 * i + 0.3)] for i, k in enumerate(["1", "2", "B"])],
    Bc(1000, 480, 340, 130, "violetaclaro", "violeta", 5, 26, en="a", t=0.8), T("agregación", 1000, 480, tam=40, c="violeta", en="a", t=0.85),
    L(1180, 480, 1380, 480, "gris", 6, True, en="a", t=1.0), Bc(1560, 480, 320, 130, "verdeclaro", "verde", 5, 26, en="a", t=1.1),
    T("predicción final", 1560, 480, tam=38, c="verde", en="a", t=1.15),
    P("condición: errores DISTINTOS", 960, 860, "ambar", 50, en="b"), sec="6. La ruta de la clase")
v.b("La ruta de hoy tiene cinco paradas. [[a]]El bootstrap, para fabricar muestras distintas. El error fuera de bolsa, una validación gratis. [[b]]El bagging, que promedia. La matemática de por qué promediar funciona. [[c]]Y el Random Forest, que agrega azar en las variables, más su ajuste en la práctica.",
    CL(), T("la ruta de hoy", 960, 160, tam=54, c="tinta", ent="subir"),
    flujo([("bootstrap", "azul", "a"), ("OOB", "teal", "a"), ("bagging", "violeta", "b"), ("varianza", "ambar", "b"), ("Random Forest", "verde", "c")],
          y=480, w=300, h=170, x0=260, paso=350, tam=40),
    T("muestras distintas · validación gratis · promediar · por qué funciona · azar en variables", 960, 700, tam=32, c="gris", w="sb", en="c", t=0.5))

# ---- 7. Error típico
v.b("El error típico en este punto: [[a]]creer que si un árbol tuvo buen resultado en una partición, es un buen modelo. [[b]]Con otra partición, ese mismo árbol puede caer de 0.817 a 0.586. [[c]]Un solo resultado de un modelo inestable no te dice casi nada.",
    CL(), T("«mi árbol dio 0.817: es buen modelo»", 960, 300, tam=56, c="tinta", en="a", id="e1"),
    T(f"otra muestra: {D['prop_auc_un_arbol_min']:.3f}", 960, 430, tam=50, c="rojo", w="b", en="b"), U("e1", tachado=1.0, color="gris", en="c", dur=0.6),
    P("un resultado de un modelo inestable dice poco", 960, 620, "verde", 50, en="c", t=0.4), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si los 200 árboles hubieran salido idénticos, exactamente iguales, [[b]]¿serviría de algo promediarlos?",
    CL(), P("200 árboles IDÉNTICOS", 960, 330, "violeta", 60, en="a"), T("¿sirve promediarlos?", 960, 520, tam=72, c="tinta", en="b"),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("No. El promedio de doscientas copias es la misma copia: [[a]]comete exactamente los mismos errores. [[b]]Promediar solo ayuda si los modelos son distintos. Por eso la primera parada de la clase es una máquina de fabricar diversidad: el bootstrap.",
    CL(), T("promedio de 200 copias = la misma copia", 960, 320, tam=58, c="tinta", ent="subir"),
    T("mismos errores, nada se cancela", 960, 440, tam=44, c="rojo", w="sb", en="a"),
    P("promediar ayuda solo si los modelos son DISTINTOS", 960, 620, "verde", 50, en="b"))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]un árbol profundo tiene bajo sesgo y alta varianza; con otra muestra cambia hasta su primera pregunta. [[b]]Promediar árboles distintos lo estabiliza: de 0.744 a 0.905 de AUC en Heart. [[c]]Y la condición es la diversidad.",
    CL(), P("árbol profundo: bajo sesgo, alta varianza", 960, 300, "rojo", 50, en="a"),
    P(f"1 árbol {D['prop_auc_un_arbol_media']:.3f} → 100 promediados {AP['100']:.3f}", 960, 470, "verde", 52, en="b"),
    P("la condición: árboles distintos", 960, 640, "violeta", 50, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]el bootstrap, a mano. Cómo sacar muestras distintas de los mismos datos.",
    CL(), siguiente(2, "Bootstrap a mano"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
