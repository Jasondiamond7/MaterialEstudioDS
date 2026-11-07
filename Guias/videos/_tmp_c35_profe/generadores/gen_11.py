import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util35 import D, FG, out, tarjetas, siguiente, ilustrativo, real, pct, caja_formula, codigo, alerta, k1, flujo, fila

OUT = out("guion_c35_11_repaso.json")
v = V("Repaso final y checklist")
CH, CA, MF = FG["cmp_h"], FG["cmp_a"], FG["mf"]["ames"]

v.portada("Repaso final y checklist", "Tres ideas, tres preguntas de la clase y los errores del material",
          "Última parada: el repaso. Tres ideas para llevar, las preguntas de repaso de la diapositiva dieciocho, los errores del material, y un checklist para tu próximo bosque.")

# ---- 1. Tres ideas
v.b("La diapositiva dieciocho resume la clase en tres ideas. [[a]]Una: el bootstrap crea diversidad al cambiar las filas. [[b]]Dos: el bagging reduce la varianza mediante agregación. [[c]]Tres: el Random Forest reduce también la correlación entre árboles.",
    CL(), T("tres ideas para llevar · diapositiva 18", 960, 160, tam=48, c="tinta", ent="subir"),
    tarjetas([(400, "azul", "1. bootstrap", "crea diversidad al cambiar las filas", "a"), (960, "violeta", "2. bagging", "reduce varianza mediante agregación", "b"),
              (1520, "verde", "3. Random Forest", "reduce también la correlación entre árboles", "c")], y=480, w=480, h=360, tam_t=46, tam_d=34),
    sec="1. Tres ideas")
v.b("Y los números que las sostienen: [[a]]63.2 % de filas únicas, 36.8 % fuera de bolsa. [[b]]La varianza del promedio, ro sigma al cuadrado más uno menos ro, sigma al cuadrado sobre B. [[c]]Y en Ames, bajar max_features al 10 % bajó el error de 24 300 a 22 500 dólares.",
    CL(), P("63.2 % únicas · 36.8 % OOB", 960, 280, "azul", 54, en="a"), caja_formula("Var = ρσ² + (1 − ρ)σ² / B", y=460, c="violeta", tam=56, w=1100, en="b"),
    P(f"Ames: max_features 10 % → {k1(MF['0.100'][0])} (todas: {k1(MF['1.000'][0])})", 960, 650, "verde", 48, en="c"))

# ---- 2. Pregunta 1
v.b("Ahora las preguntas de repaso de la clase. Intenta responder antes de que aparezca la respuesta. [[a]]Primera: ¿por qué aparecen filas repetidas en el bootstrap?",
    CL(), P("pregunta 1", 960, 220, "ambar", 44, ent="subir"), T("¿Por qué aparecen filas repetidas en el bootstrap?", 960, 400, tam=58, c="tinta", mw=1500, en="a"),
    sec="2. Pregunta 1", pausa=0.6)
v.b("Porque el muestreo es con reemplazo: [[a]]cada fila sacada vuelve a la tómbola, y puede salir otra vez. [[b]]Con muchas filas, cerca del 63.2 % entra al menos una vez, y el 36.8 % queda afuera.",
    P("muestreo CON reemplazo", 960, 580, "verde", 54, en="a"), T("≈ 63.2 % entra · ≈ 36.8 % queda afuera", 960, 720, tam=46, c="tinta", w="sb", en="b"))

# ---- 3. Pregunta 2
v.b("Segunda: [[a]]¿qué problema busca resolver max_features?",
    CL(), P("pregunta 2", 960, 220, "ambar", 44, ent="subir"), T("¿Qué problema busca resolver max_features?", 960, 400, tam=58, c="tinta", mw=1500, en="a"),
    sec="3. Pregunta 2", pausa=0.6)
v.b("La correlación entre árboles. [[a]]Si todos pueden usar siempre la mejor variable, se parecen, y el piso ro sigma al cuadrado queda alto. [[b]]Sorteando candidatas en cada nodo, los árboles se parecen menos, y el promedio mejora.",
    P("la correlación ρ entre árboles", 960, 580, "verde", 54, en="a"), T("sorteo en cada nodo → árboles distintos → piso ρσ² más bajo", 960, 720, tam=44, c="tinta", w="sb", en="b"))

# ---- 4. Pregunta 3
v.b("Tercera: [[a]]¿qué aporta el OOB, y qué no reemplaza siempre?",
    CL(), P("pregunta 3", 960, 220, "ambar", 44, ent="subir"), T("¿Qué aporta el OOB y qué no reemplaza siempre?", 960, 400, tam=58, c="tinta", mw=1500, en="a"),
    sec="4. Pregunta 3", pausa=0.6)
v.b("Aporta una estimación del error de generalización con un solo ajuste, sin apartar datos: [[a]]en Heart, 0.811 contra 0.798 de la validación cruzada. [[b]]Lo que no reemplaza siempre es una prueba independiente: sigue siendo la referencia final.",
    P(f"aporta: error interno con 1 ajuste (Heart {FG['oobcv']['heart']['oob']:.3f} vs CV {FG['oobcv']['heart']['cv']:.3f})", 960, 580, "verde", 44, en="a"),
    T("no reemplaza: la prueba independiente, referencia final", 960, 720, tam=46, c="tinta", w="sb", en="b"))

# ---- 5. Errores del material
v.b("Repasemos también lo que corregimos del material. [[a]]En el notebook, el Random Forest de regresión con max_features igual a uno punto cero es, en realidad, un bagging: 29 100 contra 29 300 dólares. [[b]]El class_weight balanced estaba solo en el bosque, una comparación no pareja. [[c]]Y en la búsqueda aleatoria, el bagging podía sortear max_features en sus árboles, convirtiéndose en un Random Forest disfrazado.",
    CL(), alerta(150, texto="lo que corregimos del material"),
    *[[Bc(960, 300 + 150 * i, 1600, 120, "blanco", "ambar", 4, 22, en=m, ent="subir"), T(tx, 960, 300 + 150 * i, tam=36, c="tinta", w="sb", mw=1500, en=m, t=0.1)]
      for i, (tx, m) in enumerate([("RandomForestRegressor(max_features=1.0) = bagging (29.1 mil vs 29.3 mil)", "a"),
                                   ("class_weight='balanced' solo en el bosque: comparación no pareja", "b"),
                                   ("bagging con estimator__max_features en la búsqueda = Random Forest disfrazado", "c")])],
    sec="5. Lo que corregimos")
v.b("Y dos aclaraciones de las diapositivas. [[a]]La diapositiva trece cuenta votos, pero scikit-learn promedia probabilidades, y con algunos umbrales la decisión cambia. [[b]]Y el ejemplo de imágenes, que usa CIFAR-10 con una descarga grande, lo reemplazamos por los dígitos de scikit-learn: el bosque acierta 96.9 %, pero cae a 50.4 % con un corrimiento de un píxel.",
    *[[Bc(960, 750 + 130 * i, 1600, 110, "blanco", "gris", 4, 22, en=m, ent="subir"), T(tx, 960, 750 + 130 * i, tam=34, c="tinta", w="sb", mw=1500, en=m, t=0.1)]
      for i, (tx, m) in enumerate([("diapositiva 13: votos ≠ promedio de probabilidades (scikit-learn promedia)", "a"),
                                   (f"imágenes: dígitos en vez de CIFAR-10 · test {FG['img']['test']:.3f} · corridas 1 píxel {FG['img']['shifted']:.3f}", "b")])])

# ---- 6. Checklist
CK = ["separa la prueba antes de todo (estratificada si clasificas)", "preprocesa dentro del Pipeline", "elige la métrica antes de ajustar",
      "compara con un árbol, un bagging y un modelo simple", "valida n_estimators, max_features y hoja mínima (CV u OOB)",
      "explica con permutación en datos no vistos", "mira la prueba una sola vez, al final"]
v.b("El checklist para tu próximo bosque. [[a]]Separa la prueba antes de todo. Preprocesa dentro del Pipeline, para no filtrar información. Elige la métrica antes de ajustar. [[b]]Compara con un árbol, un bagging y un modelo simple, como una logística. Valida n_estimators, max_features y el tamaño mínimo de hoja, con validación cruzada u OOB. [[c]]Explica con permutación en datos no vistos. Y mira la prueba una sola vez, al final.",
    CL(), T("checklist para entrenar un Random Forest", 960, 140, tam=48, c="tinta", ent="subir"),
    *[[K(330, 250 + 105 * i, "check", 46, en=m, t=0.3 * (i % 3)), T(tx, 380, 250 + 105 * i, tam=38, c="tinta", w="sb", anc="lm", en=m, t=0.3 * (i % 3) + 0.1)]
      for i, (tx, m) in enumerate(zip(CK, ["a", "a", "a", "b", "b", "c", "c"]))],
    sec="6. Checklist")

# ---- 7. Error típico
v.b("Y el error típico de toda la clase, en una frase: [[a]]creer que un algoritmo más sofisticado arregla una evaluación mal hecha. [[b]]Ningún bosque compensa una prueba contaminada o una métrica equivocada.",
    CL(), T("«un algoritmo mejor arregla una evaluación mal hecha»", 960, 300, tam=52, c="tinta", en="a", id="e1"),
    U("e1", tachado=1.0, color="gris", en="b", dur=0.6), P("ningún bosque compensa una prueba contaminada o una métrica equivocada", 960, 470, "verde", 42, en="b", t=0.4),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo, como pregunta final. [[a]]Tienes un bagging de árboles que casi no mejora a un árbol solo. ¿Qué sospechas, y qué cambias?",
    CL(), T("un bagging que casi no mejora a un árbol solo", 960, 320, tam=56, c="tinta", en="a"), T("¿qué sospechas? ¿qué cambias?", 960, 480, tam=64, c="violeta", en="a", t=0.8),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("Sospechas que los árboles se parecen demasiado: ro alto, el piso de la fórmula. [[a]]Más árboles no ayudan. [[b]]Lo que cambias es la diversidad: pasar a un Random Forest y validar max_features. [[c]]Y si aun así no mejora, quizás el árbol no era inestable, y el problema está en los datos o en la métrica.",
    CL(), P("sospecha: árboles parecidos → ρ alto", 960, 280, "rojo", 50, ent="subir"), T("más árboles no ayudan", 960, 400, tam=44, c="gris", w="sb", en="a"),
    P("cambia: Random Forest + validar max_features", 960, 540, "verde", 50, en="b"), T("si no mejora: revisa los datos o la métrica", 960, 680, tam=42, c="tinta", w="sb", en="c"))

# ---- 9. Cierre
v.b("Y con esto cerramos la clase treinta y cinco. [[a]]Empezamos con un árbol nervioso, que cambiaba de opinión con cualquier detalle, [[b]]y terminamos con un bosque estable, que sabe cuándo conviene y cuándo no. [[c]]La próxima clase da el paso siguiente: el boosting, árboles que se corrigen uno tras otro.",
    CL(), T("de un árbol nervioso…", 960, 300, tam=60, c="rojo", en="a"), T("…a un bosque estable", 960, 430, tam=60, c="verde", en="b"),
    P("próxima clase: boosting, árboles que se corrigen uno tras otro", 960, 620, "violeta", 46, en="c"), sec="9. Cierre")

n = v.guardar(OUT)
print(n, "beats")
