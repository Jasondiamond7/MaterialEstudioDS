import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, fila, codigo, flujo)

OUT = out("guion_c38_06_oof.json")
v = V("Validación cruzada y predicciones fuera de pliegue")
HP = D["h_part"]
FQ = [f["rmse"] for f in D["h_fold"]["Q_A0.1"]]
PV = D["h_prom_vs_oof"]["Q_A0.1"]

v.portada("Predicciones fuera de pliegue", "Cada fila, predicha por un modelo que nunca la vio",
          "Sexta parada: la validación cruzada por dentro. Cómo cada medición recibe una predicción hecha por un modelo que nunca la vio.")


def rondas(k_activo, en=None, t=0.0, y=430):
    """Cinco bloques (pliegues): el activo en ambar (validacion), el resto en azul (ajuste)."""
    acts = []
    for k in range(5):
        val = (k + 1) == k_activo
        acts += [Bc(500 + 270 * k, y, 240, 120, "ambarclaro" if val else "azulclaro", "ambar" if val else "azul", 4, 18, en=en, t=t + 0.05 * k),
                 T("valida" if val else "ajusta", 500 + 270 * k, y, tam=34, c="ambar" if val else "azul", en=en, t=t + 0.05 * k + 0.05)]
    return acts


# ---- 1. Hook
v.b("Esta es la idea más importante de la validación de la clase. [[a]]Cada una de las {806|ochocientas seis} mediciones del entrenamiento recibe exactamente una predicción. [[b]]Y esa predicción la hace un modelo que nunca vio esa medición, ni ninguna otra de su misma composición. [[c]]A esas predicciones se les llama fuera de pliegue: en inglés, out of fold, o con su sigla, OOF.",
    CL(), T("una predicción por fila, de un modelo que no la vio", 960, 150, tam=50, c="tinta", ent="subir"),
    N(HP["filas_tr"], 960, 360, "{:.0f}", 130, "azul", en="a"), T("mediciones → 806 predicciones", 960, 470, tam=40, c="gris", w="sb", en="a", t=0.4),
    P("cada predicción: de un modelo que no vio esa composición", 960, 620, "violeta", 44, en="b"),
    P("fuera de pliegue = out of fold = OOF", 960, 770, "teal", 50, en="c"), diapo(20, en="a"), sec="1. Una predicción por fila")

# ---- 2. Analogía
v.b("Piensa en un examen oral por turnos, con cinco grupos de alumnos. [[a]]En cada ronda, cuatro grupos preparan el material juntos, y el quinto rinde el examen. [[b]]Después de cinco rondas, cada alumno rindió exactamente una vez. [[c]]Y siempre lo evaluaron con lo que prepararon los otros cuatro grupos, nunca con lo suyo.",
    CL(), T("el examen oral por turnos", 960, 150, tam=56, c="tinta", ent="subir"),
    rondas(1, en="a"), T("ronda 1", 200, 430, tam=34, c="gris", w="sb", anc="lm", en="a"),
    P("tras 5 rondas: cada alumno rindió una vez", 960, 640, "violeta", 46, en="b"),
    T("evaluado con lo que prepararon los otros cuatro", 960, 780, tam=40, c="tinta", w="sb", en="c"),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: el examen por turnos")

# ---- 3. Las cinco rondas
v.b("En la validación cruzada es igual. [[a]]Ronda uno: se ajusta con los pliegues dos a cinco, y se predice el pliegue uno. [[b]]Ronda dos: se ajusta sin el pliegue dos, y se predice el pliegue dos. [[c]]Y así hasta la ronda cinco. Al final, cada candidato tiene {806|ochocientas seis} predicciones, obtenidas con cinco ajustes distintos.",
    CL(), T("cinco ajustes, una predicción por fila", 960, 150, tam=52, c="tinta", ent="subir"),
    g("r1", rondas(1, en="a", y=330)), T("ronda 1", 200, 330, tam=32, c="gris", w="sb", anc="lm", en="a"),
    g("r2", rondas(2, en="b", y=480)), T("ronda 2", 200, 480, tam=32, c="gris", w="sb", anc="lm", en="b"),
    T("…", 960, 580, tam=54, c="gris", en="c"), rondas(5, en="c", y=680, t=0.3), T("ronda 5", 200, 680, tam=32, c="gris", w="sb", anc="lm", en="c", t=0.3),
    P("806 predicciones · 5 ajustes por candidato", 960, 850, "teal", 46, en="c", t=1.5), diapo(20, en="a"), sec="3. Las cinco rondas")

# ---- 4. La definición
v.b("La diapositiva veinte lo escribe con símbolos. [[a]]La predicción fuera de pliegue de la fila i usa el modelo ajustado sin el pliegue de esa fila, que aquí deja fuera todo su grupo. [[b]]Y el RMSE fuera de pliegue reúne todo: sumo los {806|ochocientos seis} errores al cuadrado, divido por {806|ochocientos seis}, y saco la raíz. [[c]]Cada medición pesa igual. Los grupos controlan la separación, pero no dan el mismo peso a cada grupo: son dos decisiones distintas.",
    CL(), T("predicciones fuera de pliegue · diapositiva 20", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("ŷᵢ(OOF) = f̂ (sin el pliegue k(i)) (xᵢ)", y=300, c="violeta", tam=48, w=1300, en="a"),
    caja_formula("RMSE(OOF) = √[ (1/806) · Σ (yᵢ − ŷᵢ(OOF))² ]", y=490, c="rojo", tam=48, w=1300, en="b"),
    P("cada medición pesa igual", 600, 690, "azul", 44, en="c"), P("los grupos controlan la separación", 1340, 690, "teal", 44, en="c", t=0.8),
    diapo(20, en="a"), sec="4. La fórmula")

# ---- 5. Promedio de RMSE frente a RMSE OOF
v.b("Ahora, una sutileza de la diapositiva veintiuno. [[a]]Dos pliegues del mismo tamaño: en el primero, el error cuadrático medio es 1; en el segundo, 9. [[b]]Opción uno, promediar los RMSE: la raíz de 1 es 1, la raíz de 9 es 3; el promedio es 2. [[c]]Opción dos, el RMSE fuera de pliegue: primero juntamos los errores, uno más nueve, sobre dos, es cinco; y la raíz de cinco es 2.236.",
    CL(), T("promedio de RMSE y RMSE OOF · diapositiva 21", 960, 150, tam=50, c="tinta", ent="subir"),
    T("pliegue 1: MSE = 1      pliegue 2: MSE = 9", 960, 280, tam=46, c="tinta", w="sb", en="a"),
    T("promedio de RMSE = (√1 + √9) / 2 = (1 + 3) / 2 =", 1180, 430, tam=42, c="tinta", w="sb", anc="rm", en="b"),
    N(2, 1210, 430, "{:.0f}", 64, "azul", anc="lm", en="b", t=1.5),
    T("RMSE OOF = √[(1 + 9) / 2] = √5 =", 1180, 570, tam=42, c="tinta", w="sb", anc="rm", en="c"),
    N(2.236, 1210, 570, "{:.3f}", 64, "rojo", anc="lm", en="c", t=1.8), diapo(21, en="a"), sec="5. Promedio de RMSE o RMSE OOF")
v.b("¿Por qué salen distintos? [[a]]Porque la raíz no es lineal: el orden de las operaciones cambia el resultado. [[b]]El promedio de RMSE le da el mismo peso a cada pliegue; el RMSE fuera de pliegue le da el mismo peso a cada observación. [[c]]Los dos son válidos, pero no son intercambiables. La clase selecciona con el RMSE fuera de pliegue.",
    P("la raíz no es lineal", 960, 700, "ambar", 46, en="a"),
    T("promedio de RMSE: cada pliegue pesa igual   ·   RMSE OOF: cada observación pesa igual", 960, 820, tam=36, c="tinta", w="sb", en="b"),
    P("se selecciona con el RMSE OOF", 960, 940, "teal", 44, en="c"))
v.b("Con los datos reales, la diferencia es chica, pero existe. [[a]]Para la cuadrática con alfa igual a 0.1, los cinco RMSE por pliegue son 7.110, 7.670, 8.806, 9.320 y 8.058. [[b]]Su promedio es 8.193. [[c]]Y el RMSE fuera de pliegue, que junta los {806|ochocientos seis} errores, es 8.229. Por eso conviene decir siempre cuál de los dos se reporta.",
    CL(), T("los dos resúmenes con datos reales (α = 0.1)", 960, 150, tam=50, c="tinta", ent="subir"),
    fila(300, [(f"pliegue {k}", 360 + 300 * (k - 1)) for k in range(1, 6)], tams=[32] * 5, colores=["gris"] * 5, en="a", x0=200, x1=1720),
    fila(390, [(f"{f:.3f}", 360 + 300 * i) for i, f in enumerate(FQ)], tams=[46] * 5, colores=["violeta"] * 5, en="a", t=0.4,
         fondo="violetaclaro", x0=200, x1=1720, h=96),
    T("promedio de los cinco RMSE", 560, 550, tam=38, c="gris", w="sb", en="b"), N(PV["prom"], 560, 640, "{:.3f}", 80, "azul", en="b", t=0.3),
    T("RMSE fuera de pliegue", 1360, 550, tam=38, c="gris", w="sb", en="c"), N(PV["oof"], 1360, 640, "{:.3f}", 80, "rojo", en="c", t=0.3),
    P("decir siempre cuál se reporta", 960, 820, "teal", 46, en="c", t=1.8), real(1050, en="a"), adicional(1010, en="b", texto="promedio por pliegue: cálculo adicional"))

# ---- 6. La receta se reaprende en cada ajuste
v.b("Segunda clave, en la diapositiva veintidós: la receta completa se reaprende en cada ajuste. [[a]]Con las filas de ajuste del pliegue, se estiman la transformación y el modelo: por ejemplo, las medias y las escalas para estandarizar. [[b]]Esos parámetros quedan fijos. [[c]]Y a las filas de validación solo se les aplica la transformación ya estimada, se predice, y se calcula el error. La validación no recalcula escalas con sus propias filas.",
    CL(), T("la receta completa se reaprende en cada ajuste · diapositiva 22", 960, 150, tam=46, c="tinta", ent="subir"),
    flujo([("filas de ajuste:\nestimar transformación\ny modelo", "azul", "a"), ("parámetros\nfijados", "violeta", "b"),
           ("filas de validación:\naplicar y predecir", "ambar", "c"), ("calcular\nel error", "teal", "c")], y=470, w=380, h=250, x0=290, paso=450, tam=32),
    diapo(22, en="a"), sec="6. La receta en cada ajuste")
v.b("Si no se hace así, aparece la fuga de datos, o data leakage. [[a]]Un ejemplo mínimo: las filas de ajuste valen 10, 20 y 30, y su media es 20. [[b]]Si estandarizo usando también la fila de validación, que vale 60, la media pasa a 30. [[c]]Esa fila ya influyó en la receta antes de ser evaluada: el error de validación sale optimista. El profesor lo llamó, otra vez, hacer trampa.",
    CL(), T("fuga de datos: un ejemplo mínimo", 960, 150, tam=52, c="tinta", ent="subir"),
    *[x for i, val in enumerate([10, 20, 30]) for x in [Bc(560 + 160 * i, 330, 130, 110, "azulclaro", "azul", 4, 18, en="a", t=0.1 * i),
                                                         T(str(val), 560 + 160 * i, 330, tam=50, c="azul", en="a", t=0.1 * i + 0.05)]],
    Bc(1300, 330, 130, 110, "ambarclaro", "ambar", 4, 18, en="b"), T("60", 1300, 330, tam=50, c="ambar", en="b", t=0.05),
    T("ajuste", 720, 420, tam=30, c="azul", w="sb", en="a"), T("validación", 1300, 420, tam=30, c="ambar", w="sb", en="b"),
    T("media correcta (solo ajuste) = (10 + 20 + 30) / 3 =", 1180, 540, tam=40, c="tinta", w="sb", anc="rm", en="a", t=0.8),
    N(20, 1210, 540, "{:.0f}", 56, "verde", anc="lm", en="a", t=1.4),
    T("media con fuga = (10 + 20 + 30 + 60) / 4 =", 1180, 650, tam=40, c="tinta", w="sb", anc="rm", en="b", t=0.6),
    N(30, 1210, 650, "{:.0f}", 56, "rojo", anc="lm", en="b", t=1.2),
    P("la validación influyó en la receta → error optimista", 960, 820, "rojo", 44, en="c"), ilustrativo(1000))

# ---- 7. El código
v.b("En el código de la sección tres se ve todo esto. [[a]]Para cada pliegue k, del uno al cinco, se marca qué filas se validan. [[b]]Se clona el modelo, para partir de cero, y se ajusta solo con las filas que no se validan. [[c]]Se predicen las filas del pliegue, y al final se calcula el RMSE con todas las predicciones juntas. El script incluso comprueba que la media usada para estandarizar es la de las filas de ajuste.",
    CL(), T("en código · sección 03 (diapositiva 25)", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["for k in range(1, 6):", "    valida = folds == k", "    ajuste = clone(modelo).fit(x.loc[~valida], y[~valida])",
            "    oof[valida] = ajuste.predict(x.loc[valida])", "rmse = np.sqrt(np.mean((y - oof)**2))"], y=420, w=1640, tam=36,
           ids=["c1", "c2", "c3", "c4", "c5"]),
    R(960, 364, 1600, 60, "ambar", r=12, en="a", id="rc1"), X("rc1", en="b"), R(960, 420, 1600, 60, "ambar", r=12, en="b", id="rc2"),
    X("rc2", en="c"), R(960, 476, 1600, 60, "ambar", r=12, en="c"),
    codigo(["assert np.allclose(ajuste.steps[0][1].mean_, x.loc[~valida].mean())"], y=760, w=1640, tam=32, c="teal", en="c", t=1.5),
    diapo(25, en="a"), sec="7. En código")

# ---- 8. El protocolo fijado
v.b("Con todo esto, el protocolo queda fijado antes de comparar, como dice la diapositiva veintitrés. [[a]]Siete candidatos, cinco pliegues por composición: 35 ajustes de validación cruzada. [[b]]Se elige el menor RMSE fuera de pliegue; un empate exacto se resuelve por el orden declarado. [[c]]El MAE y el R cuadrado describen, pero no reemplazan el criterio después. Y la prueba reservada, con su referencia, queda definida desde el inicio.",
    CL(), T("un protocolo fijado antes de comparar · diapositiva 23", 960, 150, tam=48, c="tinta", ent="subir"),
    T("• 7 candidatos · 5 pliegues por composición · 35 ajustes", 960, 300, tam=40, c="tinta", w="sb", en="a"),
    T("• menor RMSE OOF · empate exacto: orden declarado", 960, 420, tam=40, c="tinta", w="sb", en="b"),
    T("• MAE y R² describen; no cambian el criterio", 960, 540, tam=40, c="tinta", w="sb", en="c"),
    T("• prueba reservada y referencia (la media) desde el inicio", 960, 660, tam=40, c="tinta", w="sb", en="c", t=1.6),
    diapo(23, en="a"), sec="8. El protocolo")

# ---- 9. El error típico
v.b("El error típico: [[a]]estandarizar todos los datos de una vez, antes de la validación cruzada. [[b]]Parece inofensivo, pero las medias y las escalas ya incluyen las filas que después se validan. [[c]]La solución es poner la estandarización dentro de la tubería, para que se reaprenda en cada pliegue.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("estandarizar todo antes de la validación cruzada", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    codigo(["x = StandardScaler().fit_transform(x)   # con TODAS las filas"], y=430, w=1400, tam=36, c="rojo", en="a", t=0.6),
    T("las escalas ya conocen las filas de validación", 960, 560, tam=42, c="tinta", w="sb", en="b"),
    codigo(["make_pipeline(StandardScaler(), modelo)   # se reaprende en cada pliegue"], y=700, w=1500, tam=36, c="verde", en="c"),
    sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tres pliegues del mismo tamaño tienen errores cuadráticos medios de 4, 16 y 25. [[b]]¿Cuánto vale el promedio de los RMSE? ¿Y el RMSE fuera de pliegue?",
    CL(), T("MSE por pliegue: 4, 16 y 25", 960, 300, tam=60, c="tinta", w="b", en="a"),
    P("promedio de RMSE = ?", 640, 520, "azul", 50, en="b"), P("RMSE OOF = ?", 1300, 520, "rojo", 50, en="b", t=0.4), sec="10. Piensa un segundo", pausa=0.6)
v.b("Los RMSE por pliegue son 2, 4 y 5; su promedio es once tercios, [[a]]3.667. [[b]]El RMSE fuera de pliegue junta primero: cuatro más dieciséis más veinticinco son 45; sobre tres, 15; y la raíz de 15 es [[c]]3.873. Otra vez, el RMSE fuera de pliegue sale mayor, porque el pliegue con más error pesa más antes de la raíz.",
    CL(), T("promedio de RMSE = (2 + 4 + 5) / 3 =", 1150, 320, tam=46, c="tinta", w="sb", anc="rm"), N(3.667, 1180, 320, "{:.3f}", 62, "azul", anc="lm", en="a"),
    T("RMSE OOF = √[(4 + 16 + 25) / 3] = √15 =", 1150, 470, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(3.873, 1180, 470, "{:.3f}", 62, "rojo", anc="lm", en="c"),
    ilustrativo(1000))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]cada fila recibe una predicción fuera de pliegue, de un modelo que no vio su composición. [[b]]El RMSE fuera de pliegue pesa igual cada observación, y no es lo mismo que el promedio de los RMSE. [[c]]Y la receta completa, con su estandarización, se reaprende en cada ajuste: así se evita la fuga de datos.",
    CL(), P("una predicción OOF por fila", 960, 300, "violeta", 50, en="a"),
    P("RMSE OOF ≠ promedio de los RMSE", 960, 460, "rojo", 50, en="b"),
    P("la receta se reaprende en cada ajuste: sin fuga", 960, 620, "teal", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]leer el ranking de los siete candidatos, la curva de validación, y las diferencias pliegue por pliegue.",
    CL(), siguiente(7, "Leer el ranking, la curva y los pliegues"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
