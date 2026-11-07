import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, alerta, k1, nodo, arista

OUT = out("guion_c36_08_xgblgbm.json")
v = V("XGBoost, LightGBM y CatBoost")
CA = FG["cmp_a"]          # {modelo: (rmse_test, rmse_train, segundos)}

v.portada("XGBoost, LightGBM y CatBoost", "Gradient Boosting, con ingeniería",
          "Octava parada: las implementaciones modernas. La misma idea de Gradient Boosting, con mucha ingeniería encima.")

# ---- 1. Hook: tiempos
v.b("Entrenamos trescientos cincuenta árboles sobre las casas de Ames. [[a]]El Gradient Boosting de scikit-learn tardó unos tres segundos. [[b]]XGBoost, menos de medio segundo. LightGBM, medio segundo. [[c]]Misma idea, pero seis veces más rápido. ¿Qué hay debajo?",
    CL(), T("350 árboles en Ames · segundos de entrenamiento (este PC)", 960, 150, tam=44, c="tinta", ent="subir"),
    *[[T(nm, 560, 300 + 130 * i, tam=42, c=c, anc="rm", en=m), Bx(600, 300 + 130 * i - 34, 600 + CA[nm][2] * 300, 300 + 130 * i + 34, c, None, 0, 12, grow="x", en=m, t=0.2),
       T(f"{CA[nm][2]:.2f} s", 620 + CA[nm][2] * 300, 300 + 130 * i, tam=40, c=c, w="b", anc="lm", en=m, t=0.8)]
      for i, (nm, c, m) in enumerate([("Gradient Boosting", "gris", "a"), ("XGBoost", "ambar", "b"), ("LightGBM", "verde", "b")])],
    P("misma idea, ~6 veces más rápido", 960, 760, "azul", 50, en="c"), real(1050, texto="datos reales · los tiempos dependen del equipo"),
    sec="1. Más rápido")

# ---- 2. XGBoost (diapositiva 12)
v.b("Empecemos por XGBoost: la equis viene de extremo. [[a]]La diapositiva doce lo define como un sistema optimizado para datos tabulares y producción. [[b]]Cuatro ideas: un objetivo regularizado, gradientes de primer y segundo orden, submuestreo de filas y columnas, y un tratamiento eficiente de datos dispersos y faltantes.",
    CL(), T("XGBoost · diapositiva 12", 960, 150, tam=50, c="tinta", ent="subir"),
    T("sistema optimizado para datos tabulares y producción", 960, 250, tam=40, c="gris", w="sb", en="a"),
    Tj(960, 580, 1100, 520, "XGBOOST", ["objetivo regularizado", "gradientes de 1.er y 2.º orden", "submuestreo de filas y columnas",
                                         "datos dispersos y faltantes, eficiente"], c="ambar", en="b", tam=38),
    T("núcleo en C++ · no viene en scikit-learn: se instala aparte", 960, 920, tam=34, c="gris", w="sb", en="b", t=2.0), sec="2. XGBoost")

# ---- 3. Objetivo regularizado
v.b("La idea más importante es el objetivo regularizado. [[a]]El objetivo es la pérdida de entrenamiento, más una penalización por complejidad. [[b]]La penalización tiene dos partes: gamma por el número de hojas, y lambda por el tamaño de los valores de las hojas.",
    CL(), T("el objetivo regularizado", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("Objetivo = pérdida de entrenamiento + penalización", y=300, c="ambar", tam=48, w=1400, en="a"),
    caja_formula("Ω = γ · (n.º de hojas) + ½ · λ · Σ w²", y=500, c="violeta", tam=52, w=1200, en="b"),
    T("w = valor que predice cada hoja", 960, 640, tam=38, c="gris", w="sb", en="b", t=0.8), sec="3. El objetivo regularizado")
v.b("Piensa en gamma como una multa por cada hoja. [[a]]Un corte nuevo solo vale la pena si mejora la pérdida más de lo que cuesta la multa. [[b]]Si un corte mejora la pérdida en 5 y gamma vale 3, se hace. Si la mejora en 2, no.",
    P("γ = una multa por cada hoja nueva", 960, 770, "violeta", 46, en="a"),
    T("mejora 5, multa 3 → se corta  ·  mejora 2, multa 3 → no", 960, 880, tam=40, c="tinta", w="sb", en="b"), ilustrativo(1000))

# ---- 4. Ejemplo resuelto: lambda encoge las hojas
v.b("Y lambda encoge los valores de las hojas. Hagamos la cuenta, con pérdida cuadrática. [[a]]Una hoja tiene tres casas, con residuos 3, 5 y 4. [[b]]Su valor óptimo es la suma de los residuos, dividida por el número de casas más lambda. [[c]]Con lambda igual a cero, 12 sobre 3: 4, el promedio de siempre.",
    CL(), T("λ encoge la hoja · pérdida cuadrática", 960, 150, tam=50, c="tinta", ent="subir"),
    P("residuos en la hoja: 3, 5, 4", 960, 270, "azul", 50, en="a"),
    caja_formula("w = (suma de residuos) / (n + λ)", y=420, c="violeta", tam=52, w=1000, en="b"),
    T("λ = 0:  12 / (3 + 0) =", 1000, 580, tam=48, c="tinta", w="sb", anc="rm", en="c"), N(4, 1030, 580, tam=60, c="azul", anc="lm", en="c", t=0.8),
    ilustrativo(1000), sec="4. Ejemplo: λ")
v.b("Con lambda igual a uno, 12 sobre 4: [[a]]3. [[b]]Con lambda igual a tres, 12 sobre 6: [[c]]2. Cuanto más grande lambda, más se encoge la hoja hacia cero. Y el efecto es más fuerte en hojas con pocos datos, que son las menos confiables.",
    T("λ = 1:  12 / (3 + 1) =", 1000, 690, tam=48, c="tinta", w="sb", anc="rm"), N(3, 1030, 690, tam=60, c="violeta", anc="lm", en="a"),
    T("λ = 3:  12 / (3 + 3) =", 1000, 800, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(2, 1030, 800, tam=60, c="violeta", anc="lm", en="c"),
    P("λ grande → hoja más cerca de 0 (más en hojas chicas)", 960, 920, "violeta", 40, en="c", t=1.0))

# ---- 5. Segundo orden
v.b("¿Y los gradientes de segundo orden? [[a]]Volvamos al cerro con niebla. El gradiente te dice la pendiente. [[b]]La segunda derivada te dice cuán curvo es el terreno. [[c]]Con las dos, XGBoost calcula pasos mejor calibrados: sabe no solo hacia dónde ir, sino cuánto avanzar.",
    CL(), T("gradientes de primer y segundo orden", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "1.er orden", "la pendiente: hacia dónde bajar", "a"), (1360, "violeta", "2.º orden", "la curvatura: cuánto avanzar", "b")],
             y=470, w=640, h=330, tam_t=60),
    P("pasos mejor calibrados", 960, 790, "ambar", 50, en="c"), sec="5. Segundo orden")

# ---- 6. LightGBM: histogramas (diapositiva 13)
v.b("Ahora LightGBM. Según la diapositiva trece, busca entrenar rápido y escalar a muchas filas y variables. [[a]]Su primera idea son los histogramas. [[b]]Para buscar el mejor corte de una variable, un árbol clásico prueba cada valor distinto. [[c]]LightGBM primero agrupa los valores en intervalos, los bins, y solo prueba los bordes de esos intervalos.",
    CL(), T("LightGBM · diapositiva 13: histogramas", 960, 150, tam=50, c="tinta", ent="subir"),
    *[Bc(260 + 23 * i, 330, 14, 14, "gris", None, 0, 7, en="b", t=0.01 * i) for i in range(60)],
    T("cada valor distinto = un corte candidato", 960, 400, tam=38, c="gris", w="sb", en="b", t=0.6),
    *[Bx(260 + 230 * k, 520, 260 + 230 * k + 200, 640, "verdeclaro", "verde", 4, 12, en="c", t=0.15 * k) for k in range(6)],
    *[L(260 + 230 * k + 215, 500, 260 + 230 * k + 215, 660, "verde", 6, en="c", t=1.0 + 0.1 * k) for k in range(5)],
    T("bins: solo se prueban los bordes", 960, 720, tam=40, c="verde", w="sb", en="c", t=1.2), sec="6. LightGBM: histogramas")
v.b("Con datos reales: la superficie habitable de Ames tiene 1 292 valores distintos. [[a]]LightGBM los agrupa, por defecto, en 255 bins. [[b]]Cinco veces menos cortes que probar, y mucha menos memoria.",
    T("Ames · superficie habitable: 1.292 valores distintos", 960, 830, tam=42, c="tinta", w="sb"),
    P("LightGBM: 255 bins por defecto", 960, 920, "verde", 44, en="a"), T("≈ 5 veces menos cortes que probar", 960, 1000, tam=36, c="gris", w="sb", en="b"),
    real(1050))

# ---- 7. Por nivel vs por hoja (diapositiva 14)
NV = {"R": (520, 330), "L": (370, 480), "Rr": (670, 480), "LL": (300, 630), "LR": (440, 630), "RL": (600, 630), "RR": (740, 630)}
NH = {"R": (1400, 330), "L": (1250, 480), "Rr": (1550, 480), "RL": (1480, 630), "RR": (1620, 630), "RLL": (1410, 780), "RLR": (1550, 780)}
v.b("La segunda idea: cómo crece el árbol. Es la diapositiva catorce. [[a]]Por nivel: se expande un nivel completo, todas las hojas a la vez. Es lo que suele hacer XGBoost. [[b]]Por hoja: se expande solo la hoja con mayor ganancia, una a la vez. Es lo que hace LightGBM, y el árbol queda asimétrico.",
    CL(), T("cómo crecen los árboles · diapositiva 14", 960, 150, tam=50, c="tinta", ent="subir"),
    T("POR NIVEL (level-wise)", 520, 240, tam=40, c="azul", en="a"), T("POR HOJA (leaf-wise)", 1400, 240, tam=40, c="verde", en="b"),
    *[arista(NV[a], NV[b], en="a", t=0.4) for a, b in [("R", "L"), ("R", "Rr"), ("L", "LL"), ("L", "LR"), ("Rr", "RL"), ("Rr", "RR")]],
    *[nodo(*NV[k], hoja=k in ("LL", "LR", "RL", "RR"), c="azul", en="a", t=0.2 * j) for j, k in enumerate(NV)],
    *[arista(NH[a], NH[b], en="b", t=0.4) for a, b in [("R", "L"), ("R", "Rr"), ("Rr", "RL"), ("Rr", "RR"), ("RL", "RLL"), ("RL", "RLR")]],
    *[nodo(*NH[k], hoja=k in ("L", "RR", "RLL", "RLR"), c="verde", en="b", t=0.2 * j) for j, k in enumerate(NH)],
    T("XGBoost suele crecer por nivel · LightGBM por hoja", 960, 875, tam=40, c="tinta", w="sb", en="b", t=1.8), sec="7. Por nivel vs por hoja")
v.b("Crecer por hoja es más eficiente con muchos datos: [[a]]pone las hojas donde más ganancia hay. [[b]]Pero tiene un riesgo, y la diapositiva lo advierte: con datos pequeños, puede sobreajustar si no se limita.",
    P("eficiente: las hojas van donde hay más ganancia", 960, 945, "verde", 38, en="a"),
    P("riesgo: sobreajuste con datos pequeños", 960, 1020, "rojo", 36, en="b"))

# ---- 8. Hojas y profundidad
v.b("Un poco de aritmética de hojas. [[a]]Un árbol por nivel con profundidad cuatro tiene, como máximo, dos a la cuarta: 16 hojas. [[b]]En LightGBM, con 15 hojas el árbol puede ser asimétrico: 15 hojas son 14 cortes, y en el caso extremo, una rama de profundidad 14. [[c]]Por eso la regla práctica: el número de hojas, menor que dos elevado a la profundidad máxima. Con profundidad cinco, menos de 32 hojas.",
    CL(), T("hojas y profundidad", 960, 160, tam=52, c="tinta", ent="subir"),
    T("por nivel, profundidad 4:  2⁴ =", 1000, 310, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(16, 1030, 310, tam=60, c="azul", anc="lm", en="a", t=0.8),
    T("LightGBM, 15 hojas = 14 cortes → hasta profundidad 14", 960, 450, tam=44, c="verde", w="sb", en="b"),
    caja_formula("num_leaves < 2^max_depth", y=620, c="violeta", tam=52, w=900, en="c"),
    T("max_depth = 5  →  num_leaves < 32", 960, 760, tam=46, c="tinta", w="sb", en="c", t=1.5), ilustrativo(1000, texto="ejemplo del apunte"),
    sec="8. Hojas y profundidad")

# ---- 9. Datos reales: sobreajuste
v.b("¿Se nota el riesgo en los datos de la clase? [[a]]LightGBM con 31 hojas por árbol, en Ames: su RMSE en entrenamiento es de solo 8 000 dólares. [[b]]En test, 27 000. [[c]]Más de tres veces peor fuera de los datos que vio: una señal clara de sobreajuste.",
    CL(), T("Ames · LightGBM con 31 hojas por árbol · RMSE", 960, 160, tam=46, c="tinta", ent="subir"),
    T("entrenamiento", 560, 360, tam=44, c="gris", anc="rm", en="a"), Bx(600, 326, 600 + CA["LightGBM"][1] / 30, 394, "verde", None, 0, 12, grow="x", en="a", t=0.2),
    T(k1(CA["LightGBM"][1]), 620 + CA["LightGBM"][1] / 30, 360, tam=44, c="verde", w="b", anc="lm", en="a", t=0.8),
    T("test", 560, 500, tam=44, c="gris", anc="rm", en="b"), Bx(600, 466, 600 + CA["LightGBM"][0] / 30, 534, "rojo", None, 0, 12, grow="x", en="b", t=0.2),
    T(k1(CA["LightGBM"][0]), 620 + CA["LightGBM"][0] / 30, 500, tam=44, c="rojo", w="b", anc="lm", en="b", t=0.8),
    P("más de 3 veces peor en test: sobreajuste", 960, 700, "rojo", 50, en="c"), real(1050), sec="9. El riesgo, en datos reales")

# ---- 10. GOSS y EFB
v.b("LightGBM trae dos trucos más. [[a]]GOSS: conserva las filas con gradientes grandes, las que el modelo todavía predice mal, y toma solo una muestra de las de gradiente pequeño. [[b]]Y EFB: agrupa variables dispersas que casi nunca están activas a la vez, como muchas columnas de un one-hot, en una sola.",
    CL(), T("dos trucos más de LightGBM", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "GOSS", "conserva gradientes grandes y muestrea los pequeños", "a"),
              (1360, "violeta", "EFB", "agrupa variables dispersas casi excluyentes", "b")], y=480, w=680, h=360, tam_t=64),
    sec="10. GOSS y EFB")

# ---- 11. Alerta del notebook: subsample
v.b("Y una alerta del notebook. [[a]]El LightGBM del notebook usa subsample igual a 0.8, para que cada árbol vea solo el ochenta por ciento de las filas. [[b]]Pero en LightGBM, ese submuestreo solo se activa si subsample_freq es mayor que cero, y por defecto vale cero. [[c]]Lo comprobamos: con o sin ese 0.8, las predicciones son idénticas. XGBoost, en cambio, sí lo aplica.",
    CL(), alerta(170, texto="alerta del notebook"),
    T("LGBMClassifier(subsample=0.8, ...)", 960, 320, tam=46, c="tinta", w="m", en="a"),
    T("el submuestreo solo se activa con subsample_freq > 0 (por defecto: 0)", 960, 450, tam=40, c="tinta", w="sb", en="b"),
    P("con o sin subsample = 0.8: predicciones idénticas", 960, 600, "rojo", 46, en="c"),
    T("solución: subsample_freq=1 · XGBoost sí aplica su subsample", 960, 720, tam=40, c="verde", w="sb", en="c", t=1.5),
    real(1050, texto="comprobado con LightGBM 4.7 en Heart"), sec="11. Alerta: subsample")

# ---- 12. CatBoost
v.b("¿Y CatBoost? [[a]]Es la quinta implementación de la clase, especialista en variables categóricas: las maneja de forma nativa. [[b]]La diapositiva dieciocho la recomienda cuando dominan muchas categorías. [[c]]Pero en la práctica de la clase no se usó, y no está instalada en este entorno, así que no tenemos números propios para ella.",
    CL(), T("CatBoost", 960, 200, tam=72, c="teal", ent="subir"),
    P("especialista en variables categóricas (nativas)", 960, 360, "teal", 48, en="a"),
    T("diapositiva 18: muchas categorías → CatBoost", 960, 490, tam=42, c="tinta", w="sb", en="b"),
    T("no se usó en la práctica · sin números propios", 960, 610, tam=40, c="gris", w="sb", en="c"), sec="12. CatBoost")

# ---- 13. Error típico
v.b("El error típico: [[a]]pensar que XGBoost y LightGBM son algoritmos distintos de Gradient Boosting. [[b]]No lo son: son implementaciones de la misma idea, residuos y gradientes, árbol tras árbol. [[c]]Lo que cambia es la ingeniería: regularización, histogramas, crecimiento por hoja, y velocidad.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("«son algoritmos distintos»", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("son Gradient Boosting: la misma idea", 960, 420, tam=50, c="tinta", w="sb", en="b"),
    P("cambia la ingeniería: regularización · histogramas · hojas · velocidad", 960, 600, "azul", 42, en="c"), sec="13. El error típico")

# ---- 14. Piensa un segundo
v.b("Piensa un segundo. [[a]]El XGBoost del notebook usa profundidad máxima tres. ¿Cuántas hojas puede tener cada árbol, como máximo? [[b]]¿Y cómo se compara con las 31 hojas del LightGBM de Ames?",
    CL(), P("XGBoost: max_depth = 3 → ¿hojas máximas?", 960, 300, "ambar", 52, en="a"), P("LightGBM de Ames: 31 hojas", 960, 450, "verde", 52, en="b"),
    sec="14. Piensa un segundo", pausa=0.6)
v.b("Dos elevado a tres: [[a]]8 hojas. [[b]]El LightGBM de Ames usa casi cuatro veces más hojas por árbol. Árboles mucho más complejos, y eso ayuda a explicar su brecha entre entrenamiento y test.",
    CL(), T("2³ =", 900, 320, tam=72, c="tinta", w="sb", anc="rm"), N(8, 930, 320, tam=86, c="ambar", anc="lm", en="a"),
    P("31 hojas ≈ 4 veces más: árboles más complejos", 960, 500, "verde", 48, en="b"))

# ---- 15. Para llevar
v.b("Para llevarte: [[a]]XGBoost agrega un objetivo regularizado, con gamma y lambda, y usa segundo orden. [[b]]LightGBM usa histogramas y crece hoja por hoja: muy rápido, pero cuidado con datos pequeños. [[c]]Y CatBoost es el especialista en categóricas.",
    CL(), P("XGBoost: regularización (γ, λ) + 2.º orden", 960, 300, "ambar", 48, en="a"),
    P("LightGBM: histogramas + por hoja (ojo con datos chicos)", 960, 460, "verde", 46, en="b"), P("CatBoost: especialista en categóricas", 960, 620, "teal", 48, en="c"),
    sec="15. Para llevar")
v.b("Siguiente parada: [[a]]early stopping, y los hiperparámetros que de verdad mueven el resultado.",
    CL(), siguiente(9, "Early stopping e hiperparámetros"), sec="16. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
