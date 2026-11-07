import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, k1, cien, r3

OUT = out("guion_c36_11_repaso.json")
v = V("Repaso final y checklist")
BB, ES, LR = FG["bb"], FG["es"], {d["lr"]: d for d in FG["lr"]}
CA = FG["cmp_a"]

v.portada("Repaso final", "Toda la clase en cinco minutos, y un checklist para entrenar",
          "Última parada: el repaso. Toda la clase de boosting en unos minutos, y un checklist para tu próximo modelo.")

# ---- 1. La idea
v.b("La idea central: [[a]]boosting entrena modelos simples en secuencia, cada uno sobre el error del anterior. [[b]]La predicción es una suma: efe cero más eta por cada árbol. [[c]]Y lo que distingue a cada algoritmo es cómo define el error.",
    CL(), T("1 · la idea", 960, 150, tam=52, c="tinta", ent="subir"),
    P("modelos simples EN SECUENCIA, sobre el error anterior", 960, 300, "azul", 46, en="a"),
    caja_formula("F(x) = F₀ + η·h₁(x) + η·h₂(x) + …", y=470, c="ambar", tam=56, w=1200, en="b"),
    P("cada algoritmo define el «error» a su manera", 960, 640, "violeta", 46, en="c"), sec="1. La idea")

# ---- 2. Bagging vs boosting
v.b("Frente a bagging: [[a]]el bosque promedia árboles profundos en paralelo, y baja la varianza. [[b]]Boosting suma árboles chicos en secuencia, y baja el sesgo. [[c]]En Ames, el bosque se estancó en " + cien(BB["rf300"]) + " dólares; boosting siguió bajando hasta " + cien(BB["gb300"]) + ".",
    CL(), T("2 · bagging vs boosting", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "bagging", "paralelo · profundos · promedio → ↓ varianza", "a"),
              (1360, "ambar", "boosting", "secuencia · chicos · suma → ↓ sesgo", "b")], y=450, w=680, h=320, tam_t=60),
    P(f"Ames: bosque {k1(BB['rf300'])} · boosting {k1(BB['gb300'])}", 960, 720, "verde", 46, en="c"), real(1050), sec="2. Bagging vs boosting")

# ---- 3. AdaBoost
v.b("AdaBoost: [[a]]el error épsilon es la suma de los pesos de los casos mal clasificados. [[b]]El peso del árbol es alfa, un medio del logaritmo de uno menos épsilon, sobre épsilon. [[c]]Los errores multiplican su peso por e elevado a alfa, y el voto final es el signo de la suma ponderada.",
    CL(), T("3 · AdaBoost", 960, 150, tam=52, c="tinta", ent="subir"),
    P("ε = suma de pesos de los errores", 960, 290, "rojo", 46, en="a"), P("α = ½ · ln((1 − ε)/ε)", 960, 420, "violeta", 50, en="b"),
    P("errores × e^α  ·  voto = signo(Σ α·h)", 960, 550, "ambar", 46, en="c"), sec="3. AdaBoost")
v.b("El ejemplo a mano: con épsilon 0.2, alfa vale 0.693, los errores duplican su peso, y tras normalizar quedan con la mitad del total. [[a]]Y su punto débil: el ruido, porque una etiqueta mala acumula peso ronda tras ronda.",
    T("ε = 0.2 → α = 0.693 → errores × 2 → la mitad del peso", 960, 700, tam=42, c="tinta", w="sb"),
    P("punto débil: el ruido", 960, 820, "rojo", 46, en="a"))

# ---- 4. Gradient Boosting y la pérdida
v.b("Gradient Boosting: [[a]]parte del promedio, y cada árbol aprende el gradiente negativo de la pérdida. [[b]]Con pérdida cuadrática, eso es el residuo: en la tabla de la diapositiva diez, más 2, menos 3, más 1 y cero. [[c]]Con la absoluta, aprende signos; con Huber, residuos recortados; con la logística, y menos pe.",
    CL(), T("4 · Gradient Boosting y la pérdida", 960, 150, tam=52, c="tinta", ent="subir"),
    P("cada árbol aprende −∂L/∂F", 960, 290, "ambar", 50, en="a"),
    T("cuadrática: el residuo  ·  diapositiva 10: +2, −3, +1, 0", 960, 430, tam=42, c="tinta", w="sb", en="b"),
    T("absoluta: signos · Huber: recortados · logística: y − p", 960, 540, tam=40, c="gris", w="sb", en="c"), sec="4. Gradient Boosting")
v.b("Y una regla práctica: [[a]]entrena con la misma pérdida con la que vas a juzgar. El notebook entrenó con Huber y evaluó con RMSE.",
    P("entrena con la pérdida con que vas a juzgar", 960, 720, "verde", 46, en="a"))

# ---- 5. Tasa y árboles
v.b("La tasa de aprendizaje y los árboles: [[a]]eta chico necesita más árboles. En el modelo idealizado, cuatro árboles con eta 0.5, y veintidós con eta 0.1. [[b]]En Ames: con eta uno, el mejor punto llegó con " + str(LR[1.0]["n"]) + " árboles y " + cien(LR[1.0]["best"]) +
    " dólares; con eta 0.1, con " + str(LR[0.1]["n"]) + " árboles y " + cien(LR[0.1]["best"]) + ". [[c]]Se ajustan en pareja, y eta bajo con pocos árboles subajusta.",
    CL(), T("5 · tasa y número de árboles", 960, 150, tam=52, c="tinta", ent="subir"),
    T("idealizado: η = 0.5 → 4 árboles · η = 0.1 → 22", 960, 300, tam=44, c="tinta", w="sb", en="a"),
    T(f"Ames: η = 1 → {LR[1.0]['n']} árboles, {k1(LR[1.0]['best'])} · η = 0.1 → {LR[0.1]['n']}, {k1(LR[0.1]['best'])}", 960, 420, tam=42, c="violeta", w="sb", en="b"),
    P("en pareja · η bajo con pocos árboles = subajuste", 960, 580, "ambar", 46, en="c"), real(1050), sec="5. Tasa y árboles")

# ---- 6. Implementaciones
v.b("Las implementaciones: [[a]]primero, XGBoost, que agrega un objetivo regularizado, con gamma por hoja y lambda que encoge las hojas, y usa segundo orden. [[b]]LightGBM usa histogramas y crece hoja por hoja: muy rápido, pero con riesgo de sobreajuste en datos chicos. [[c]]Y CatBoost, el especialista en categóricas.",
    CL(), T("6 · XGBoost, LightGBM y CatBoost", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "ambar", "XGBoost", "regularización γ, λ · 2.º orden", "a"), (960, "verde", "LightGBM", "histogramas · por hoja · ojo datos chicos", "b"),
              (1520, "teal", "CatBoost", "especialista en categóricas", "c")], y=470, w=480, h=330, tam_t=50, tam_d=34), sec="6. Implementaciones")
v.b("Y la alerta: [[a]]en LightGBM, subsample no hace nada sin subsample_freq.",
    P("LightGBM: subsample necesita subsample_freq > 0", 960, 790, "rojo", 46, en="a"))

# ---- 7. Early stopping
v.b("Early stopping: [[a]]mide la validación tras cada árbol, y para tras varias rondas seguidas sin mejorar. [[b]]En Ames, XGBoost se detuvo con " + str(ES["parada"]) + " árboles de mil quinientos, con un error de validación de " + cien(ES["val"]) + " dólares. [[c]]Y la validación sale del entrenamiento, nunca del test.",
    CL(), T("7 · early stopping", 960, 150, tam=52, c="tinta", ent="subir"),
    P("k rondas sin mejorar → parar · quedarse con la mejor", 960, 290, "verde", 46, en="a"),
    T(f"Ames: se detuvo con {ES['parada']} de 1.500 árboles · validación {k1(ES['val'])}", 960, 430, tam=42, c="tinta", w="sb", en="b"),
    P("validación ≠ test", 960, 570, "rojo", 52, en="c"), real(1050), sec="7. Early stopping")

# ---- 8. La comparación
v.b("La comparación: [[a]]en Heart, los cinco modelos empatan dentro del desvío. [[b]]En Ames, XGBoost fue el mejor, con " + cien(CA["XGBoost"][0]) + " dólares, y AdaBoost el peor, con " + cien(CA["AdaBoost"][0]) + ". [[c]]Una sola partición no declara ganadores.",
    CL(), T("8 · la comparación", 960, 150, tam=52, c="tinta", ent="subir"),
    P("Heart: empate dentro del desvío (0.035 a 0.044)", 960, 290, "azul", 46, en="a"),
    P(f"Ames: XGBoost {k1(CA['XGBoost'][0])} · AdaBoost {k1(CA['AdaBoost'][0])}", 960, 430, "violeta", 46, en="b"),
    P("una sola partición no declara ganadores", 960, 570, "rojo", 46, en="c"), real(1050), sec="8. La comparación")

# ---- 9. Preguntas de repaso (diapositiva 19)
v.b("La última diapositiva, la diecinueve, deja tres preguntas. Respóndelas tú primero. [[a]]Una: ¿qué cambia entre AdaBoost y Gradient Boosting? [[b]]Dos: ¿por qué el learning_rate y el número de árboles interactúan? [[c]]Tres: ¿cuándo priorizarías LightGBM?",
    CL(), T("preguntas de repaso · diapositiva 19", 960, 150, tam=50, c="tinta", ent="subir"),
    P("1. ¿qué cambia entre AdaBoost y Gradient Boosting?", 960, 320, "azul", 44, en="a"),
    P("2. ¿por qué learning_rate y n.º de árboles interactúan?", 960, 460, "ambar", 44, en="b"),
    P("3. ¿cuándo priorizarías LightGBM?", 960, 600, "verde", 44, en="c"), sec="9. Preguntas de repaso", pausa=0.6)
v.b("Respuestas. [[a]]Uno: AdaBoost cambia los pesos de los casos y vota con alfa; Gradient Boosting entrena cada árbol sobre el gradiente negativo de una pérdida. [[b]]Dos: porque cada árbol aporta solo una fracción eta; con eta más chico, hacen falta más árboles para llegar igual de lejos. [[c]]Tres: con muchas filas y variables, cuando la velocidad importa, y cuidando la complejidad si los datos son pocos.",
    CL(), T("respuestas", 960, 150, tam=52, c="tinta", ent="subir"),
    T("1. AdaBoost: pesos y voto con α · GB: gradiente negativo de una pérdida", 960, 300, tam=38, c="azul", w="sb", en="a"),
    T("2. cada árbol aporta η: η chico → más árboles para llegar igual", 960, 430, tam=38, c="ambar", w="sb", en="b"),
    T("3. muchas filas y variables, velocidad · ojo con datos chicos", 960, 560, tam=38, c="verde", w="sb", en="c"))

# ---- 10. Checklist
CHK = ["parte train / test (estratificado en clasificación)", "preprocesa dentro del Pipeline", "empieza con una línea base",
       "η bajo + early stopping con validación propia", "limita la complejidad del árbol", "compara train vs test",
       "incluye tu configuración base en la búsqueda", "reporta media ± desvío, misma partición"]
v.b("El checklist para entrenar un modelo de boosting. [[a]]Uno, parte en entrenamiento y test, estratificado si es clasificación. Dos, preprocesa dentro del pipeline. [[b]]Tres, empieza con una línea base, como AdaBoost o un bosque. Cuatro, fija un eta bajo, y deja que early stopping elija los árboles, con una validación propia. [[c]]Cinco, limita la complejidad del árbol. Y seis, compara el error de entrenamiento con el de test.",
    CL(), T("checklist de boosting", 960, 120, tam=50, c="tinta", ent="subir"),
    Tj(960, 600, 1400, 840, "ANTES DE ENTREGAR", CHK, c="navy", tam=34, visibles=0, id="chk", marcas=[dict(forma="check", color="verde", progreso=0)] * 8),
    U("chk", visibles=2, en="a", dur=1.2), U("chk", visibles=4, en="b", dur=1.4), U("chk", visibles=6, en="c", dur=1.4), sec="10. Checklist")
v.b("Siete: si buscas hiperparámetros, incluye tu configuración base en la búsqueda, para no terminar con algo peor. [[a]]Y ocho: reporta la media con su desvío, y compara siempre bajo la misma partición.",
    U("chk", visibles=7, dur=1.0), U("chk", visibles=8, en="a", dur=1.0),
    U("chk", marcas=[dict(forma="check", color="verde", progreso=1)] * 8, en="a", t=1.5, dur=1.5))

# ---- 11. Tres ideas para llevar (diapositiva 20)
v.b("Y las tres ideas para llevar de la clase: [[a]]uno, boosting aprende de forma secuencial. [[b]]Dos, la pérdida define qué corrige cada etapa. [[c]]Tres, cada biblioteca prioriza un problema distinto.",
    CL(), T("tres ideas para llevar · diapositiva 19", 960, 150, tam=50, c="tinta", ent="subir"),
    P("1. boosting aprende de forma secuencial", 960, 320, "azul", 50, en="a"), P("2. la pérdida define qué corrige cada etapa", 960, 460, "ambar", 50, en="b"),
    P("3. cada biblioteca prioriza un problema distinto", 960, 600, "verde", 50, en="c"), sec="11. Tres ideas")
v.b("El siguiente paso, según la clase, es comparar los modelos bajo el mismo pipeline, la misma validación y el mismo presupuesto. [[a]]Y ahora sí: a practicar con el notebook guiado y el cuaderno de ejercicios.",
    CL(), P("mismo pipeline · misma validación · mismo presupuesto", 960, 380, "violeta", 48),
    T("a practicar: notebook guiado + cuaderno de ejercicios", 960, 560, tam=48, c="tinta", w="sb", en="a"), sec="12. Cierre")

n = v.guardar(OUT)
print(n, "beats")
