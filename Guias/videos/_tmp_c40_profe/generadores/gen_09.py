import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_09_oof.json")
v = V("Comparar recetas fuera de pliegue")
O, X = D["oof"], D["oof_extremos"]
REC = [("FRECUENCIA", "frecuencia de spam"), ("BERN_05", "Bernoulli, α = 0.5"), ("BERN_1", "Bernoulli, α = 1"), ("BERN_2", "Bernoulli, α = 2"),
       ("GAUSS_LOG", "gaussiana, log(1 + x)")]

v.portada("Comparar recetas fuera de pliegue", "Log loss, Brier y AUC con los mismos 3680 correos",
          "Novena parada. Tenemos cinco recetas candidatas. Las vamos a comparar con predicciones fuera de pliegue, y vamos a ver por qué un modelo que acierta bastante puede dar probabilidades muy malas.")

# ---- 1. Las recetas
v.b("La diapositiva cuarenta y uno define los candidatos, todos con las mismas cincuenta y cuatro frecuencias y los mismos cinco pliegues. [[a]]Una referencia: la frecuencia de spam, una probabilidad sin usar ninguna señal. [[b]]Tres recetas de Bernoulli, con alfa 0.5, uno y dos. [[c]]Y una receta gaussiana, con medias y dispersiones del logaritmo de uno más equis. Son cinco candidatos.",
    CL(), T("qué recetas compararemos · diapositiva 41", 960, 140, tam=50, c="tinta", ent="subir"),
    tabla([["receta", "qué aprende", "opciones"], ["frecuencia de spam", "una probabilidad sin usar señales", "una referencia"],
           ["Bernoulli", "presencias y ausencias", "α = 0.5, 1, 2"], ["gaussiana", "medias y dispersiones de log(1 + x)", "una receta"]],
          [420, 1000, 1520], 300, paso=115, tam=40, x0=170, x1=1760,
          colores=[["tinta"] * 3, ["gris", "tinta", "tinta"], ["teal", "tinta", "tinta"], ["azul", "tinta", "tinta"]], ens=[None, "a", "b", "c"]),
    P("5 candidatos: 1 referencia + 3 suavizados + 1 gaussiana", 960, 800, "violeta", 44, en="c", t=1.0), diapo(41, en="a"), sec="1. Las cinco recetas")
v.b("¿Por qué incluir una referencia que no usa señales? [[a]]Porque si una receta complicada no le gana a dividir dos números, no vale la pena. [[b]]El profesor lo dijo así: por qué complicarse la vida si con dos divisiones queda mejor. [[c]]Y la regla de elección tiene dos pasos: primero, el menor log loss; después, el umbral del ganador. Primero comparamos probabilidades; después decidimos qué hacer con ellas.",
    CL(), cita("por qué me voy a complicar la vida si simplemente puedo sacar hacer dos divisiones", y=270, tam=42, en="a"),
    flujo([("1. menor log loss", "violeta", "c"), ("2. umbral del ganador", "ambar", "c")], y=560, w=560, h=200, x0=620, paso=680, tam=46),
    grabacion("01:43:43", en="a"), diapo(41, en="c", y=1010))

# ---- 2. El log loss
v.b("Recordemos el log loss, que ya usamos en la clase treinta y ocho. [[a]]Para cada correo, miramos la probabilidad que el modelo le dio a la clase que realmente ocurrió, y tomamos menos su logaritmo. Luego promediamos. [[b]]Un cálculo adicional para sentirlo: si era spam y el modelo dijo 0.9, el castigo es 0.105. Si dijo 0.1, el castigo es 2.303. [[c]]Y si dijo 0.001, el castigo es 6.908. Equivocarse con mucha seguridad sale carísimo.",
    CL(), T("log loss: castiga la poca probabilidad a lo que ocurrió", 960, 140, tam=48, c="tinta", ent="subir"),
    caja_formula("log loss = − promedio de log P(clase real)", y=290, c="violeta", tam=50, w=1200),
    tabla([["el correo era spam y el modelo dijo", "castigo = −log p"], ["p = 0.9", f"{-math.log(0.9):.3f}"], ["p = 0.1", f"{-math.log(0.1):.3f}"],
           ["p = 0.001", f"{-math.log(0.001):.3f}"]], [760, 1300], 440, paso=100, tam=42, x0=380, x1=1560, ens=[None, "b", "b", "c"]),
    adicional(1050, en="b"), sec="2. El log loss")

# ---- 3. Fuera de pliegue
v.b("¿Con qué predicciones se calcula? La diapositiva cuarenta y dos lo dibuja. [[a]]Los tres mil seiscientos ochenta correos de desarrollo están repartidos en cinco pliegues. Se ajusta con cuatro, se predice el pliegue apartado, y se repite cinco veces. [[b]]Así, cada correo recibe una probabilidad que no participó en su propio ajuste: la predicción fuera de pliegue. [[c]]Cinco recetas por cinco pliegues: veinticinco ajustes.",
    CL(), T("una predicción sin participar en su ajuste · diapositiva 42", 960, 140, tam=48, c="tinta", ent="subir"),
    flujo([("ajustar: 4 pliegues", "azul", "a"), ("predecir: el pliegue apartado", "violeta", "a"), ("repetir: 1 probabilidad por correo", "teal", "b")],
          y=420, w=480, h=220, x0=420, paso=540, tam=40),
    T("5 recetas × 5 pliegues = 25 ajustes", 960, 660, tam=52, c="tinta", w="b", en="c"),
    P("la selección usa desarrollo; los 921 correos de prueba quedan aparte", 960, 800, "ambar", 42, en="c", t=1.0), diapo(42, en="a"), sec="3. Fuera de pliegue")
v.b("En código, la diapositiva cuarenta y tres muestra el recorrido para Bernoulli con alfa uno. [[a]]Un vector vacío para las predicciones, un ciclo sobre los pliegues, y en cada vuelta: val marca el pliegue apartado, se ajusta con el resto y se predice val. [[b]]Al final, pred reúne las tres mil seiscientos ochenta probabilidades fuera de pliegue. El script conserva también las log-probabilidades, para evaluar con precisión.",
    CL(), T("el mismo recorrido de validación · diapositiva 43", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["pred = np.empty(len(y))", "for k in sorted(np.unique(folds)):", "    val = folds == k",
            '    pred[val], _, _ = ajustar_predecir(X[~val], y[~val], X[val], "bern", 1.)'], y=400, w=1640, tam=36, c="violeta", en="a"),
    P("pred: 3680 probabilidades OOF de spam", 960, 680, "teal", 46, en="b"), diapo(43, en="a"))

# ---- 4. Alerta
v.b("Una precisión sobre la grabación. [[a]]Al presentar esta comparación, cerca de los minutos ciento cuatro y ciento nueve, se dijo que las recetas se evalúan en el conjunto de prueba. [[b]]En el material, la comparación usa las predicciones fuera de pliegue de los tres mil seiscientos ochenta correos de desarrollo. Los novecientos veintiuno de prueba se usan una sola vez, al final, con la receta y el umbral ya elegidos.",
    CL(), error_material("01:48:57", "cuando lo evalúe estas metodologías en prueba, no hay mucha discrepancia",
                         "se comparan con OOF en los 3680 de desarrollo; la prueba (921) se usa una vez, al final", en_ok="b"),
    T("también en 01:44:57: «esto lo voy a evaluar en el conjunto de prueba»", 960, 760, tam=34, c="gris", w="sb", en="a", t=1.0),
    sec="4. Una precisión")

# ---- 5. La tabla
v.b("Y estos son los resultados, de la diapositiva cuarenta y cuatro, recalculados con el script. [[a]]La frecuencia de spam: log loss 0.671, Brier 0.239 y AUC 0.499. Es la referencia: no ordena nada. [[b]]Bernoulli con alfa 0.5: log loss 0.556, Brier 0.099 y AUC 0.952. Con alfa uno y dos, casi lo mismo: 0.557 y 0.558. [[c]]Y la gaussiana: Brier 0.190 y AUC 0.940, pero un log loss de 4.931.",
    CL(), T("Bernoulli obtiene el menor error probabilístico · diapositiva 44", 960, 130, tam=46, c="tinta", ent="subir"),
    tabla([["receta", "log loss", "Brier", "AUC"]] + [[nom, f"{O[k]['log_loss']:.3f}", f"{O[k]['brier']:.3f}", f"{O[k]['auc']:.3f}"] for k, nom in REC],
          [520, 1000, 1260, 1500], 260, paso=100, tam=42, x0=220, x1=1680,
          colores=[["tinta"] * 4, ["gris"] * 4, ["verde"] * 4, ["teal"] * 4, ["teal"] * 4, ["azul", "rojo", "azul", "azul"]],
          ens=[None, "a", "b", "b", "b", "c"]),
    real(1050, en="a"), sec="5. La tabla")
v.b("Cómo se lee cada columna. [[a]]El Brier promedia el error cuadrático entre la probabilidad y la etiqueta, cero o uno: menor es mejor. [[b]]El AUC mide el ordenamiento entre clases: mayor es mejor. [[c]]Por el menor log loss se elige Bernoulli con alfa 0.5. Los tres suavizados dan resultados cercanos, como anticipamos: con tantos correos, el suavizado casi no importa.",
    CL(), P("Brier: error cuadrático entre p y la etiqueta (menor es mejor)", 960, 300, "teal", 44, en="a"),
    P("AUC: ordenamiento entre clases (mayor es mejor)", 960, 450, "azul", 44, en="b"),
    P("elegido: Bernoulli, α = 0.5 (log loss 0.556)", 960, 620, "verde", 50, en="c"),
    T("los tres suavizados: 0.556 · 0.557 · 0.558", 960, 760, tam=42, c="tinta", w="sb", en="c", t=1.0), diapo(44, en="a"))
v.b("Y la advertencia de la diapositiva: [[a]]el mínimo fuera de pliegue intervino en la selección. Elegimos al que salió mejor; por eso su número tiende a ser un poco optimista. [[b]]No constituye una evaluación independiente del ganador. Esa la intentaremos con la prueba histórica, en la parada once.",
    CL(), P("el mínimo OOF intervino en la selección", 960, 360, "ambar", 50, en="a"),
    P("no es una evaluación independiente del ganador", 960, 540, "rojo", 48, en="b"), diapo(44, en="a"))

# ---- 6. Aciertos y certeza
GB = "gb9"
v.b("La diapositiva cuarenta y cinco dibuja el log loss de cada receta. [[a]]Los tres Bernoulli quedan juntos, alrededor de 0.56; la referencia, en 0.67. [[b]]Y la gaussiana se dispara hasta casi cinco. [[c]]Lo curioso: con umbral 0.5, la gaussiana acierta el 80.8 por ciento de los correos. No es un clasificador inútil; es un clasificador demasiado seguro de sí mismo.",
    CL(), T("los aciertos pueden esconder certeza equivocada · diapositiva 45", 960, 130, tam=46, c="tinta", ent="subir"),
    barras_v([(nom.replace("Bernoulli, ", "Bern. "), O[k]["log_loss"], c, m) for (k, nom), c, m in
              zip(REC, ["gris", "verde", "teal", "teal", "rojo"], ["a", "a", "a", "a", "b"])],
             x0=380, paso=290, base=820, escala=110, w=180, fmt="{:.3f}", tam=34),
    P("gaussiana: acierta 80.8 % con umbral 0.5", 840, 330, "azul", 40, en="c"), diapo(45, en="a"), sec="6. Aciertos y certeza")
v.b("¿Cómo puede acertar tanto y tener un log loss tan malo? Un cálculo adicional lo explica. [[a]]La gaussiana pone el 98.6 por ciento de sus probabilidades fuera del intervalo de 0.01 a 0.99: casi siempre dice cero o uno. [[b]]Y se equivoca con esa seguridad: quinientos noventa y dos correos legítimos reciben probabilidad de spam de 0.999 o más. Bernoulli hace eso con veintisiete. [[c]]Cada uno de esos errores suma al menos 6.9 al log loss: el castigo de la tabla anterior.",
    CL(), T("acertar no es lo mismo que dar buenas probabilidades", 960, 140, tam=50, c="tinta", ent="subir"),
    tabla([["", "gaussiana", "Bernoulli α = 0.5"], ["probabilidades fuera de (0.01, 0.99)", f"{100 * X['gauss_fuera_001']:.1f} %", f"{100 * X['bern05_fuera_001']:.1f} %"],
           ["legítimos con p ≥ 0.999", str(X["gauss_leg_999"]), str(X["bern_leg_999"])]], [560, 1150, 1520], 300, paso=120, tam=42, x0=180, x1=1760,
          ens=[None, "a", "b"]),
    P("cada error con p ≥ 0.999 suma al menos −log(0.001) ≈ 6.9", 960, 700, "rojo", 44, en="c"), adicional(1050, en="a"))
v.b("La conclusión de la diapositiva: [[a]]el log loss penaliza los errores con certeza excesiva. [[b]]El porcentaje de aciertos no muestra esa diferencia. Por eso se elige con el log loss, y no con la exactitud.",
    CL(), P("log loss: penaliza la certeza excesiva", 960, 360, "violeta", 54, en="a"),
    P("los aciertos no ven esa diferencia", 960, 540, "ambar", 50, en="b"), diapo(45, en="a"))

# ---- 7. Analogía
v.b("Una analogía: dos pronosticadores del tiempo. [[a]]El primero siempre dice cero por ciento o cien por ciento de lluvia. Acierta muchos días, pero cuando falla, alguien sale sin paraguas bajo un aguacero. [[b]]El segundo dice setenta o veinte por ciento. Acierta un poco menos de forma tajante, pero sus números significan algo. [[c]]Para decidir, a partir de una probabilidad, nos sirve el segundo. Esa es la diferencia entre la gaussiana y Bernoulli aquí.",
    CL(), T("dos pronosticadores del tiempo", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "«0 % o 100 %»", "acierta mucho, pero falla con total seguridad", "a"),
              (1300, "verde", "«70 % o 20 %»", "sus números significan algo", "b")], y=460, w=560, h=300, tam_t=52, tam_d=38),
    P("para decidir con una probabilidad, sirve el segundo", 960, 760, "violeta", 44, en="c"), analogia(1050), sec="7. Analogía: el pronóstico")

# ---- 8. Errores típicos
v.b("Errores típicos al comparar. [[a]]Uno: elegir la receta por la exactitud, y quedarse con un modelo que acierta pero da probabilidades extremas. [[b]]Dos: leer el log loss del ganador como si fuera una evaluación independiente, cuando participó en la selección. [[c]]Y tres: comparar recetas con distintos pliegues. Todas deben usar los mismos correos y los mismos pliegues.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("elegir por exactitud", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("leer el mínimo OOF como evaluación independiente", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("comparar recetas con pliegues distintos", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="8. Errores típicos")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Dos correos. El primero es spam, y el modelo le dio 0.8 de spam. El segundo es legítimo, y el modelo le dio 0.4 de spam. [[b]]¿Cuál es el log loss promedio?",
    CL(), P("correo 1: spam, p = 0.8   ·   correo 2: legítimo, p = 0.4", 960, 330, "teal", 46, en="a"),
    P("log loss = ?", 960, 520, "violeta", 60, en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("Para el spam, la probabilidad de lo que ocurrió es 0.8: castigo 0.223. [[a]]Para el legítimo, lo que ocurrió tiene probabilidad uno menos 0.4, es decir 0.6: castigo 0.511. [[b]]El promedio: aproximadamente 0.367. El error típico aquí es usar 0.4 en el segundo correo: hay que usar la probabilidad de la clase real.",
    CL(), T("−log 0.8 = 0.223   ·   −log(1 − 0.4) = −log 0.6 = 0.511", 960, 330, tam=46, c="tinta", w="sb", en="a"),
    T("promedio ≈", 1000, 480, tam=56, c="tinta", w="sb", anc="rm", en="b"), N((-math.log(0.8) - math.log(0.6)) / 2, 1030, 480, "{:.3f}", 72, "violeta", anc="lm", en="b"),
    P("usar la probabilidad de la clase real: 1 − 0.4 en el legítimo", 960, 650, "ambar", 44, en="b", t=1.0), ilustrativo(1050))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]se comparan cinco recetas con las mismas predicciones fuera de pliegue de los tres mil seiscientos ochenta correos de desarrollo. [[b]]Gana Bernoulli con alfa 0.5, con log loss 0.556; los tres suavizados casi empatan. [[c]]Y la gaussiana acierta el 80.8 por ciento, pero su log loss es 4.931: certeza equivocada.",
    CL(), P("5 recetas · mismas predicciones OOF · 3680 correos", 960, 300, "violeta", 46, en="a"),
    P("gana Bernoulli α = 0.5: log loss 0.556", 960, 460, "verde", 48, en="b"),
    P("gaussiana: 80.8 % de aciertos, log loss 4.931", 960, 620, "rojo", 46, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]por qué las probabilidades salen tan extremas. Copiamos una señal, miramos la calibración y explicamos el correo de identificador uno, señal por señal.",
    CL(), siguiente(10, "Señales copiadas, calibración y un correo explicado"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
