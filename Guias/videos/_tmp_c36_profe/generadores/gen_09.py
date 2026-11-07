import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util36 import D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, codigo, alerta, k1, cien, r3

OUT = out("guion_c36_09_early.json")
v = V("Early stopping e hiperparámetros")
ES = FG["es"]
CU = D["curvas"]
NH, NHT = D["nb_heart"], D["nb_heart_tuned"]
NA, NAT = D["nb_ames"], D["nb_ames_tuned"]
YM = 60

v.portada("Early stopping", "Que la validación decida cuántos árboles, y qué ajustar primero",
          "Novena parada: early stopping. Y los hiperparámetros que de verdad mueven el resultado.")

# ---- 1. Hook
v.b("¿Cuántos árboles necesita un modelo de boosting? [[a]]¿Cien? ¿Mil? [[b]]Ya viste que demasiados pueden empezar a ajustar ruido, y muy pocos dejan el modelo corto. [[c]]La respuesta práctica es: que lo decida la validación, mientras entrenas.",
    CL(), T("¿cuántos árboles?", 960, 260, tam=80, c="tinta", ent="subir"), T("¿100? ¿1.000?", 960, 420, tam=60, c="gris", en="a"),
    T("pocos: corto  ·  demasiados: ruido", 960, 560, tam=48, c="tinta", w="sb", en="b"), P("que lo decida la validación", 960, 720, "verde", 56, en="c"),
    sec="1. ¿Cuántos árboles?")

# ---- 2. Analogía: el horno
v.b("Piensa en un queque en el horno. [[a]]No pones un temporizador a ciegas: lo revisas cada pocos minutos. [[b]]Mientras mejora, lo dejas. [[c]]Y si en dos revisiones seguidas ya no mejora, lo sacas, y te quedas con el mejor momento.",
    CL(), T("un queque en el horno", 960, 160, tam=52, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "revisar", "cada pocos minutos", "a"), (960, "verde", "mejora", "lo dejas", "b"),
              (1520, "ambar", "2 veces sin mejorar", "lo sacas: el mejor momento", "c")], y=490, w=480, h=330, tam_t=46, tam_d=36),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el horno")

# ---- 3. Definición
v.b("Eso es early stopping, en español, parada temprana. [[a]]Separas un conjunto de validación, distinto del test. [[b]]Después de cada árbol, mides el error en validación. [[c]]Y si no mejora durante varias rondas seguidas, la paciencia, el entrenamiento se detiene y se conserva la mejor ronda.",
    CL(), T("early stopping = parada temprana", 960, 160, tam=52, c="tinta", ent="subir"),
    P("1. un conjunto de validación (≠ test)", 960, 320, "teal", 48, en="a"), P("2. medir el error tras cada árbol", 960, 460, "azul", 48, en="b"),
    P("3. k rondas sin mejorar → parar y quedarse con la mejor", 960, 600, "verde", 48, en="c"), sec="3. La regla")

# ---- 4. Ejemplo resuelto (apunte)
SER = [30, 25, 22, 21, 21.4, 21.9, 22.5]
v.b("Un ejemplo del apunte, con paciencia dos. [[a]]El error de validación, ronda por ronda: 30, 25, 22, 21. [[b]]Luego 21.4, 21.9 y 22.5.",
    CL(), T("early stopping con paciencia 2 · RMSE de validación", 960, 150, tam=46, c="tinta", ent="subir"),
    *[[Bx(320 + 200 * k, 860 - (val - 15) * 30, 460 + 200 * k, 860, "azul", None, 0, 10, grow="y", en=("a" if k < 4 else "b"), t=0.3 * (k % 4), id=f"es{k}"),
       T(f"{val:g}", 390 + 200 * k, 860 - (val - 15) * 30 - 32, tam=36, c="tinta", w="b", en=("a" if k < 4 else "b"), t=0.3 * (k % 4) + 0.3),
       T(f"ronda {k + 1}", 390 + 200 * k, 900, tam=28, c="gris", w="sb", en=("a" if k < 4 else "b"))] for k, val in enumerate(SER)],
    ilustrativo(1000, texto="ejemplo del apunte"), sec="4. Ejemplo a mano")
v.b("El mínimo es 21, en la ronda cuatro. [[a]]La ronda cinco no mejora: van una. La ronda seis tampoco: van dos. [[b]]Con paciencia dos, se detiene después de la ronda seis. [[c]]Y el modelo final es el de la ronda cuatro, la mejor. La ronda siete nunca se entrena.",
    U("es3", relleno="verde"), R(990, 860 - 6 * 30 - 32, 150, 70, "verde", r=16),
    U("es4", relleno="ambar", en="a"), U("es5", relleno="ambar", en="a", t=1.2),
    T("1 sin mejorar", 1190, 300, tam=30, c="ambar", w="b", en="a"), T("2 sin mejorar", 1390, 300, tam=30, c="ambar", w="b", en="a", t=1.2),
    L(1500, 230, 1500, 900, "rojo", 6, guion=True, en="b"), T("STOP", 1520, 220, tam=40, c="rojo", anc="lm", en="b", t=0.3),
    U("es6", alfa=0.25, en="c", t=1.5), P("modelo final = ronda 4", 960, 1000, "verde", 40, en="c"))

# ---- 5. Datos reales: XGBoost en Ames
xs = list(range(1, len(CU["xgb_va"]) + 1, 3))
v.b("Con datos reales. [[a]]XGBoost en Ames, con eta 0.05, profundidad tres, hasta mil quinientos árboles, y paciencia cincuenta. [[b]]El error de entrenamiento baja y baja. [[c]]El de validación baja rápido al principio, y después se aplana.",
    CL(), T("Ames · XGBoost con early stopping · RMSE", 960, 140, tam=46, c="tinta", ent="subir"),
    T("η = 0.05 · profundidad 3 · máximo 1.500 árboles · paciencia 50", 960, 205, tam=32, c="gris", w="sb", en="a"),
    G(260, 260, 1360, 840, 0, 1500, 0, YM, [dict(xs=xs, ys=[min(CU["xgb_tr"][x - 1] / 1000, YM) for x in xs], color="gris", grosor=5, revela=0.0),
                                            dict(xs=xs, ys=[min(CU["xgb_va"][x - 1] / 1000, YM) for x in xs], color="ambar", grosor=6, revela=0.0)],
      xticks=[0, 300, 600, 900, 1200, 1500], yticks=[0, 15, 30, 45, 60], yfmt="{:.0f} mil", xlabel="ronda (árbol)", id="ge", en="a"),
    U("ge", **{"series.0.revela": 1.0}, en="b", dur=2.0), P("entrenamiento", 1620, 560, "gris", 34, en="b", t=1.0),
    U("ge", **{"series.1.revela": 1.0}, en="c", dur=2.0), P("validación", 1620, 380, "ambar", 34, en="c", t=1.0), real(1050), sec="5. XGBoost en Ames")
v.b("XGBoost reporta su mejor iteración: la 754, contando desde cero. [[a]]Como espera cincuenta rondas sin mejora, se detiene al completar 805 árboles, de los mil quinientos posibles. [[b]]En esa mejor ronda, el error de validación es de 21 000 dólares, y el de entrenamiento, 9 400.",
    L(260 + ES["best"] / 1500 * 1100, 260, 260 + ES["best"] / 1500 * 1100, 840, "verde", 5, guion=True), T(f"mejor: {ES['best']}", 260 + ES["best"] / 1500 * 1100 + 10, 290, tam=32, c="verde", w="b", anc="lm"),
    T(f"se detiene con {ES['parada']} árboles de 1.500", 1620, 680, tam=32, c="tinta", w="sb", en="a"),
    T(f"validación: {k1(ES['val'])}", 1620, 760, tam=34, c="ambar", w="b", en="b"), T(f"entrenamiento: {k1(ES['train'])}", 1620, 830, tam=34, c="gris", w="b", en="b", t=0.4))
v.b("Ahorró casi setecientos árboles, y eligió el número sola. [[a]]Fíjate en la brecha: el entrenamiento sigue mejorando, pero eso ya no sirve fuera de los datos vistos.",
    P("ahorró ~700 árboles y eligió n sola", 810, 985, "verde", 36), T("la brecha: mejorar en train ya no sirve", 1620, 920, tam=28, c="rojo", w="sb", en="a"))

# ---- 6. Validación no es test
v.b("Una regla que no se negocia: [[a]]el conjunto de validación del early stopping nunca es el test. [[b]]Si usas el test para decidir cuándo parar, el test participó en el entrenamiento, y tu estimación final deja de ser honesta. [[c]]En estos datos, la validación sale del entrenamiento, y el test queda intacto.",
    CL(), alerta(180, texto="regla de oro"), T("validación del early stopping ≠ test", 960, 330, tam=58, c="rojo", en="a"),
    T("si el test decide cuándo parar, ya no es una evaluación honesta", 960, 470, tam=42, c="tinta", w="sb", en="b"),
    P("entrenamiento → (ajuste + validación) · test intacto", 960, 630, "verde", 46, en="c"), sec="6. Validación no es test")
v.b("Y un detalle de scikit-learn: [[a]]su Gradient Boosting entrena todos los árboles que le pidas, salvo que actives su propia parada temprana. [[b]]Por defecto ese parámetro está apagado, así que no hay parada temprana automática.",
    T("GradientBoostingRegressor(n_iter_no_change=None)  # por defecto", 960, 800, tam=36, c="tinta", w="m", en="a"),
    T("sin n_iter_no_change: entrena todos los n_estimators", 960, 890, tam=38, c="gris", w="sb", en="b"))

# ---- 7. Hiperparámetros que mueven el resultado (diapositiva 15)
FIL = [("Capacidad", "n_estimators / iterations", "cantidad de etapas", "azul"), ("Paso", "learning_rate", "aporte de cada etapa", "ambar"),
       ("Árbol", "max_depth / num_leaves", "complejidad e interacciones", "violeta"), ("Regularización", "min_child_weight, min_data_in_leaf, L2", "evitar hojas frágiles", "teal"),
       ("Aleatoriedad", "subsample, colsample", "diversidad, menos sobreajuste", "verde")]
v.b("La diapositiva quince resume los hiperparámetros que realmente mueven el resultado. [[a]]Capacidad: el número de árboles. Paso: el learning_rate. [[b]]Árbol: la profundidad o el número de hojas. [[c]]Regularización: mínimos por hoja y la penalización L2. Y aleatoriedad: submuestreo de filas y de columnas.",
    CL(), T("hiperparámetros que mueven el resultado · diapositiva 15", 960, 140, tam=44, c="tinta", ent="subir"),
    *[[Bc(960, 270 + 125 * i, 1640, 105, "blanco", c, 4, 20, en=m, t=0.3 * (i % 2), ent="subir"),
       T(f, 190, 270 + 125 * i, tam=36, c=c, w="b", anc="lm", en=m, t=0.3 * (i % 2) + 0.1),
       T(pr, 560, 270 + 125 * i, tam=32, c="tinta", w="m", anc="lm", en=m, t=0.3 * (i % 2) + 0.2),
       T(ef, 1740, 270 + 125 * i, tam=32, c="gris", w="sb", anc="rm", en=m, t=0.3 * (i % 2) + 0.3)]
      for i, ((f, pr, ef, c), m) in enumerate(zip(FIL, ["a", "a", "b", "c", "c"]))],
    sec="7. Hiperparámetros")
v.b("Y su consejo: [[a]]ajusta con validación cruzada o con early stopping, y conserva el test para la evaluación final.",
    P("ajusta con CV o early stopping · el test, solo al final", 960, 940, "verde", 44, en="a"))

# ---- 8. Las búsquedas del notebook
v.b("El notebook de la clase ajusta con dos estrategias. [[a]]GridSearchCV, que prueba todas las combinaciones de una grilla: para AdaBoost, dos valores de árboles por dos de tasa, cuatro combinaciones. [[b]]Y RandomizedSearchCV, que sortea combinaciones: para XGBoost, seis de 1 728 posibles. [[c]]Las dos búsquedas usan solo el entrenamiento, con tres pliegues.",
    CL(), T("las búsquedas del notebook (Heart)", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "GridSearchCV", "AdaBoost: 2 × 2 = 4 combinaciones", "a"),
              (1360, "violeta", "RandomizedSearchCV", "XGBoost: 6 de 1.728 combinaciones", "b")], y=450, w=680, h=320, tam_t=50),
    T("solo con el entrenamiento · validación cruzada de 3 pliegues", 960, 700, tam=40, c="gris", w="sb", en="c"), sec="8. Las búsquedas del notebook")
v.b("¿Y mejoraron en test? Replicamos el notebook. [[a]]AdaBoost: AUC de 0.930 sin ajustar, 0.929 ajustado. [[b]]XGBoost: 0.937 sin ajustar, 0.929 ajustado. Peor. [[c]]Con 227 pacientes y tres pliegues, las diferencias de la validación cruzada son casi puro ruido: ajustar no garantiza ganar en test.",
    CL(), T("Heart · AUC en test: sin ajustar vs ajustado", 960, 150, tam=48, c="tinta", ent="subir"),
    T("AdaBoost:", 700, 300, tam=48, c="azul", anc="rm", en="a"), T(f"{r3(NH['AdaBoost']['auc'])}  →  {r3(NHT['ada']['auc'])}", 740, 300, tam=52, c="tinta", w="b", anc="lm", en="a", t=0.4),
    T("XGBoost:", 700, 420, tam=48, c="violeta", anc="rm", en="b"), T(f"{r3(NH['XGBoost']['auc'])}  →  {r3(NHT['xgb']['auc'])}", 740, 420, tam=52, c="rojo", w="b", anc="lm", en="b", t=0.4),
    P("227 pacientes, 3 pliegues: diferencias casi puro ruido", 960, 600, "ambar", 46, en="c"),
    real(1050, texto="réplica del notebook (FAST_MODE = True)"))
v.b("En Ames, en cambio, ajustar sí ayudó al Gradient Boosting: [[a]]de 29 200 dólares a 23 900, porque la configuración de partida era débil. [[b]]Y a LightGBM casi no le cambió nada: de 25 800 a 25 900.",
    T("Ames · RMSE en test:", 960, 740, tam=42, c="tinta", w="sb"),
    T(f"Gradient Boosting: {k1(NA['Gradient Boosting']['rmse_test'])} → {k1(NAT['gb']['rmse_test'])}", 960, 830, tam=44, c="verde", w="b", en="a"),
    T(f"LightGBM: {k1(NA['LightGBM']['rmse_test'])} → {k1(NAT['lgbm']['rmse_test'])}", 960, 920, tam=44, c="gris", w="b", en="b"))

# ---- 9. Error típico
v.b("El error típico: [[a]]elegir los hiperparámetros, o el momento de parar, mirando el test. [[b]]Parece inofensivo, pero cada decisión que tomas con el test lo contamina un poco. [[c]]El test se mira una sola vez, al final.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("ajustar o parar mirando el test", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("cada decisión tomada con el test lo contamina", 960, 420, tam=46, c="tinta", w="sb", en="b"),
    P("el test se mira UNA vez, al final", 960, 600, "verde", 54, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Paciencia dos, y la serie: 30, 25, 22, 21, 21.4, y luego 20.9. [[b]]¿Se detiene en la ronda seis?",
    CL(), P("paciencia 2 · 30, 25, 22, 21, 21.4, 20.9", 960, 300, "azul", 52, en="a"), T("¿se detiene en la ronda 6?", 960, 470, tam=70, c="tinta", en="b"),
    sec="10. Piensa un segundo", pausa=0.6)
v.b("No. [[a]]La ronda cinco no mejora: van una. Pero la ronda seis baja a 20.9, un nuevo mínimo. [[b]]El contador vuelve a cero, y el entrenamiento sigue. La paciencia cuenta rondas seguidas sin mejorar.",
    CL(), T("No", 960, 280, tam=110, c="rojo"), T("ronda 5: 1 sin mejorar  ·  ronda 6: 20.9 < 21, nuevo mínimo", 960, 450, tam=42, c="tinta", w="sb", en="a"),
    P("el contador vuelve a 0: cuenta rondas SEGUIDAS", 960, 600, "verde", 48, en="b"), ilustrativo(1000))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]early stopping mide la validación tras cada árbol y para cuando deja de mejorar durante varias rondas seguidas. [[b]]La validación sale del entrenamiento, nunca del test. [[c]]Y ajustar hiperparámetros no garantiza ganar en test: con pocos datos, las diferencias pueden ser ruido.",
    CL(), P("parar tras k rondas sin mejorar · quedarse con la mejor", 960, 300, "verde", 46, en="a"),
    P("validación ≠ test", 960, 460, "rojo", 52, en="b"), P("ajustar no garantiza ganar en test", 960, 620, "ambar", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]la comparación de los modelos con los datos de la clase, y cómo elegir.",
    CL(), siguiente(10, "La comparación y cómo elegir"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
