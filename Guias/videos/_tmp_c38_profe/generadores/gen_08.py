import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, fila, codigo)

OUT = out("guion_c38_08_prueba.json")
v = V("Reajustar, evaluar una vez y diagnosticar")
HF = D["h_final"]
HR = D["h_resid"]
RD = {r["modelo"]: r for r in D["h_ranking"]}
PR, RE, ED = HR["pred"], HR["real"], HR["edad"]
RES = [r - p for r, p in zip(RE, PR)]

v.portada("La prueba final", "Reajustar, evaluar una sola vez y mirar los residuos",
          "Octava parada: se abre la prueba reservada. Reajustamos el modelo elegido, lo evaluamos una sola vez, y miramos dónde se equivoca.")

# ---- 1. Hook
v.b("Hasta ahora, la prueba reservada no se ha tocado. [[a]]Ni para ajustar, ni para elegir. [[b]]Ahora, con la elección cerrada, llega el momento de consultar al oráculo. [[c]]Y se consulta una sola vez: lo que salga es la estimación del desempeño de la elección final.",
    CL(), T("se abre la prueba reservada", 960, 150, tam=56, c="tinta", ent="subir"),
    Cand(960, 420, 140, c="ambar", id="cand"), T("ni para ajustar, ni para elegir", 960, 600, tam=42, c="gris", w="sb", en="a"),
    U("cand", abierto=1.0, en="b", dur=0.8),
    P("una sola vez", 960, 760, "ambar", 54, en="c"), sec="1. Abrir la prueba")

# ---- 2. Analogía
v.b("Piensa en los ensayos antes de un examen final. [[a]]Los ensayos sirven para practicar y para decidir cómo estudiar: eso fue la validación cruzada. [[b]]El examen final se rinde una vez. [[c]]Si después de ver las preguntas pides rendirlo de nuevo, la nota ya no mide lo que sabes: mide lo que viste del examen.",
    CL(), T("los ensayos y el examen final", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "violeta", "ensayos", "practicar y decidir = validación cruzada", "a"), (1360, "ambar", "examen final", "una sola vez = prueba reservada", "b")],
             y=440, w=640, h=300, tam_t=52, tam_d=38),
    P("¿repetirlo tras ver las preguntas? la nota ya no mide", 960, 760, "rojo", 44, en="c"), ilustrativo(1000, texto="analogía"),
    sec="2. Analogía: el examen final")

# ---- 3. Reajustar
v.b("Primer paso, en la diapositiva treinta: reajustar. [[a]]Reajustar significa volver a estimar: se toma la receta ganadora y se aprende de nuevo, con las {806|ochocientas seis} filas del entrenamiento, sus escalas y sus coeficientes. [[b]]Con ese único ajuste se predicen las {224|doscientas veinticuatro} mezclas reservadas. [[c]]Ojo: no se promedian los cinco modelos de la validación cruzada. Eso sería otro procedimiento.",
    CL(), T("reajustar significa volver a estimar · diapositiva 30", 960, 140, tam=48, c="tinta", ent="subir"),
    codigo(["ganador = ranking.modelo.iloc[0]", "modelo_final = clone(modelos[ganador])",
            "modelo_final.fit(entrenamiento[variables], entrenamiento.resistencia)", "pred_prueba = modelo_final.predict(prueba[variables])"],
           y=380, w=1700, tam=34, en="a"),
    P("806 filas → un ajuste → 224 predicciones", 960, 640, "azul", 46, en="b"),
    P("no se promedian los 5 modelos de la CV", 960, 780, "rojo", 46, en="c"), diapo(30, en="a"), sec="3. Reajustar")

# ---- 4. La evaluación final
v.b("Y este es el resultado, recalculado con el script de la clase. [[a]]La cuadrática con alfa igual a 0.1 obtiene en la prueba un RMSE de 8.823 megapascales, un MAE de 6.494, y un R cuadrado de 0.716. [[b]]La referencia fijada desde el inicio, la media de entrenamiento, tiene un RMSE de 16.605, un MAE de 12.826, y un R cuadrado de menos 0.007. [[c]]El RMSE baja un 46.9 por ciento frente a la referencia: uno menos 8.823 sobre 16.605.",
    CL(), T("la evaluación final se expresa en MPa · diapositiva 31", 960, 140, tam=48, c="tinta", ent="subir"),
    fila(280, [("", 480), ("RMSE", 900), ("MAE", 1220), ("R²", 1540)], tams=[38] * 4, colores=["gris"] * 4),
    fila(390, [("cuadrática, α = 0.1", 480), (f"{HF['rmse']:.3f}", 900), (f"{HF['mae']:.3f}", 1220), (f"{HF['r2']:.3f}", 1540)],
         tams=[38, 54, 54, 54], colores=["verde"] * 4, en="a", fondo="verdeclaro", x0=240, x1=1700, h=100),
    fila(510, [("media de entrenamiento", 480), (f"{HF['ref_rmse']:.3f}", 900), (f"{HF['ref_mae']:.3f}", 1220), (f"{HF['ref_r2']:.3f}", 1540)],
         tams=[36, 54, 54, 54], colores=["gris"] * 4, en="b", fondo="linea", x0=240, x1=1700, h=100),
    T("baja del RMSE = 1 − 8.823 / 16.605 =", 1080, 690, tam=46, c="tinta", w="sb", anc="rm", en="c"),
    N(100 * HF["baja"], 1110, 690, "{:.1f}", 70, "azul", anc="lm", suf=" %", en="c", t=1.6), real(1050, en="a"), sec="4. El resultado en prueba")
v.b("Comparado con la validación: [[a]]fuera de pliegue, el ganador tenía 8.229; en la prueba, 8.823. [[b]]Es un poco más alto, algo esperable: el número usado para elegir tiende a ser optimista, y además la prueba es otra muestra, de solo 86 composiciones. [[c]]Lo importante: ese 8.823 no se usa para volver a elegir. El script lo dice: no se evalúan los otros candidatos en la prueba.",
    CL(), T("validación frente a prueba", 960, 150, tam=54, c="tinta", ent="subir"),
    T("RMSE fuera de pliegue", 560, 320, tam=40, c="gris", w="sb", en="a"), N(RD["Q_A0.1"]["rmse"], 560, 420, "{:.3f}", 90, "violeta", en="a", t=0.3),
    T("RMSE en prueba", 1360, 320, tam=40, c="gris", w="sb", en="a", t=1.0), N(HF["rmse"], 1360, 420, "{:.3f}", 90, "ambar", en="a", t=1.3),
    T("optimismo de la selección · otra muestra (86 composiciones)", 960, 590, tam=38, c="tinta", w="sb", en="b"),
    P("no se evalúan los otros candidatos en la prueba", 960, 740, "rojo", 46, en="c"), real(1050, en="a"))

# ---- 5. Predicho frente a real
v.b("La diapositiva treinta y uno también muestra el gráfico de predicho frente a real. [[a]]Cada punto es una de las {224|doscientas veinticuatro} mezclas de prueba. [[b]]La diagonal representa la predicción perfecta: predicho igual a real. [[c]]Los puntos siguen la diagonal, pero con dispersión: algunos quedan lejos. Ese número, el RMSE, es un resumen del error; no es una cota para cada caso.",
    CL(), T("predicho frente a real · 224 mezclas de prueba", 960, 140, tam=48, c="tinta", ent="subir"),
    G(560, 230, 1260, 930, 0, 95, 0, 95, [dict(xs=[0, 95], ys=[0, 95], color="ambar", grosor=4, revela=0.0, lapiz=False),
                                         dict(**dict(zip(("xs", "ys"), map(list, zip(*sorted(zip(RE, PR)))))), color="azul", linea=False,
                                              marcadores=6, revela=0.0, lapiz=False)],
      xticks=[20, 40, 60, 80], yticks=[20, 40, 60, 80], yfmt="{:.0f}", xlabel="real (MPa)", ylabel="pred.", id="gp", tam_txt=28),
    U("gp", **{"series.1.revela": 1.0}, en="a", dur=1.6), U("gp", **{"series.0.revela": 1.0}, en="b", dur=1.0),
    T("diagonal = predicción perfecta", 1300, 400, tam=34, c="ambar", w="sb", anc="lm", en="b", t=0.8),
    P("el RMSE no es una cota por caso", 1560, 700, "rojo", 38, en="c"), real(1050, en="a"), sec="5. Predicho frente a real")

# ---- 6. Residuos y diagnóstico
v.b("Por eso la diapositiva treinta y dos pide diagnosticar. [[a]]El residuo es el valor real menos el predicho. Un residuo positivo indica subestimación: el modelo predijo menos de lo que resistió la mezcla. [[b]]En la prueba, el residuo más grande es más 24.39 megapascales, y el más negativo, menos 32.81. [[c]]Hay 25 mezclas con un error mayor que 15 en valor absoluto, y nueve con un error mayor que 20.",
    CL(), T("una cifra promedio no reemplaza el diagnóstico · diapositiva 32", 960, 140, tam=46, c="tinta", ent="subir"),
    G(420, 240, 1360, 860, 0, 85, -35, 30, [dict(xs=[0, 85], ys=[0, 0], color="ambar", grosor=4, revela=1.0, lapiz=False),
                                           dict(**dict(zip(("xs", "ys"), map(list, zip(*sorted(zip(PR, RES)))))), color="azul", linea=False,
                                                marcadores=6, revela=0.0, lapiz=False)],
      xticks=[20, 40, 60, 80], yticks=[-30, -15, 0, 15, 30], yfmt="{:.0f}", xlabel="predicho (MPa)", ylabel="residuo", id="gr", tam_txt=26),
    U("gr", **{"series.1.revela": 1.0}, t=0.3, dur=1.6),
    T("residuo = real − predicho", 1640, 300, tam=34, c="tinta", w="sb", en="a"), T("> 0: subestima", 1640, 360, tam=32, c="azul", w="sb", en="a", t=0.8),
    T(f"máximo: +{HR['max']:.2f}", 1640, 480, tam=36, c="tinta", w="sb", en="b"), T(f"mínimo: {HR['min']:.2f}", 1640, 540, tam=36, c="tinta", w="sb", en="b", t=0.6),
    T(f"|error| > 15: {HR['sobre15']} mezclas", 1640, 660, tam=34, c="rojo", w="sb", en="c"), T(f"|error| > 20: {HR['sobre20']} mezclas", 1640, 720, tam=34, c="rojo", w="sb", en="c", t=0.6),
    real(1050, en="a"), adicional(1010, en="b", texto="máximos y conteos: cálculo adicional"), sec="6. Los residuos")
RE_ = HR["rmse_edad"]
v.b("El otro gráfico de la diapositiva mira los errores según la edad. [[a]]Si separamos las mezclas por edad, y calculamos el RMSE de cada grupo, para las de uno a siete días es 10.247, con 57 casos. [[b]]De 14 a 28 días baja a 7.330, con 104 casos. [[c]]Y de 56 días en adelante sube a 9.649, con 63 casos. El 8.823 global esconde que el modelo se equivoca más en las edades extremas.",
    CL(), T("RMSE de prueba por edad (MPa)", 960, 150, tam=52, c="tinta", ent="subir"),
    *[x for i, (lab, k, col) in enumerate([("1 a 7 días", "1-7", "rojo"), ("14 a 28 días", "14-28", "verde"), ("56 a 365 días", "56-365", "ambar")])
      for x in [Bx(460 + 500 * i - 120, 820 - RE_[k]["rmse"] * 45, 460 + 500 * i + 120, 820, col, None, 0, 12, grow="y", en="abc"[i]),
                T(f"{RE_[k]['rmse']:.3f}", 460 + 500 * i, 820 - RE_[k]["rmse"] * 45 - 40, tam=46, c=col, w="b", en="abc"[i], t=0.5),
                T(lab, 460 + 500 * i, 870, tam=36, c="gris", w="sb", en="abc"[i]), T(f"n = {RE_[k]['n']}", 460 + 500 * i, 920, tam=30, c="gris", w="sb", en="abc"[i])]],
    L(240, 820 - HF["rmse"] * 45, 1700, 820 - HF["rmse"] * 45, "violeta", 4, guion=True, en="c", t=1.2),
    T("global 8.823", 1720, 820 - HF["rmse"] * 45, tam=30, c="violeta", w="sb", anc="lm", en="c", t=1.5),
    adicional(1050, en="a"))
v.b("Y una advertencia final de la clase: [[a]]estos gráficos describen la prueba. [[b]]Si los usamos para rediseñar el modelo, por ejemplo agregando una variable para las edades extremas, la prueba deja de estar intacta. [[c]]El rediseño exige una nueva selección y una nueva evaluación independiente. Lo veremos de nuevo al final de la clase.",
    CL(), T("los diagnósticos describen la prueba", 960, 150, tam=54, c="tinta", ent="subir"),
    P("rediseñar con lo que vimos en la prueba…", 960, 320, "ambar", 46, en="b"),
    P("…exige nueva selección y nueva evaluación", 960, 480, "rojo", 46, en="c"), diapo(58, en="c"))

# ---- 7. El error típico
v.b("El error típico: [[a]]evaluar los siete candidatos en la prueba y quedarse con el que salga mejor. [[b]]Parece honesto, porque «la prueba es independiente». Pero si la prueba se usa para elegir, ya no es independiente de la elección. [[c]]Se convierte en otra validación, y su cifra vuelve a ser optimista.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("evaluar los 7 en la prueba y elegir el mejor", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("si la prueba elige, deja de ser independiente", 960, 440, tam=44, c="tinta", w="sb", en="b"),
    P("se convierte en otra validación: cifra optimista", 960, 620, "ambar", 46, en="c"), sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un modelo tiene un RMSE de prueba de 6, y la referencia de la media tiene 15. [[b]]¿En qué porcentaje baja el RMSE? [[c]]Y con un RMSE de 6, ¿puede haber un caso con un error de 20?",
    CL(), T("modelo: RMSE = 6   ·   media: RMSE = 15", 960, 300, tam=56, c="tinta", w="b", en="a"),
    P("¿cuánto baja el RMSE?", 960, 480, "azul", 50, en="b"), P("¿puede haber un error de 20?", 960, 620, "rojo", 50, en="c"),
    sec="8. Piensa un segundo", pausa=0.6)
v.b("Baja uno menos seis sobre quince: [[a]]un 60 por ciento. [[b]]Y sí, puede haber un caso con un error de 20: el RMSE es un promedio, no una cota. [[c]]En el hormigón, con un RMSE de 8.823, el peor caso se equivocó por 32.81.",
    CL(), T("1 − 6 / 15 =", 1000, 320, tam=54, c="tinta", w="sb", anc="rm"), N(60, 1030, 320, "{:.0f}", 72, "azul", anc="lm", suf=" %", en="a"),
    P("sí: el RMSE es un promedio, no una cota", 960, 500, "rojo", 48, en="b"),
    T("hormigón: RMSE 8.823, peor caso 32.81", 960, 650, tam=42, c="gris", w="sb", en="c"), ilustrativo(1000))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]la receta elegida se reajusta con todo el entrenamiento, y se evalúa una sola vez en la prueba. [[b]]El resultado se compara con la referencia fijada desde el inicio: aquí, 8.823 contra 16.605 megapascales. [[c]]Y los residuos muestran lo que el promedio esconde, sin usarlos para volver a elegir.",
    CL(), P("reajustar con todo el entrenamiento · evaluar una vez", 960, 300, "azul", 44, en="a"),
    P("comparar con la referencia fijada: 8.823 vs 16.605", 960, 460, "verde", 46, en="b"),
    P("los residuos muestran lo que el promedio esconde", 960, 620, "ambar", 46, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]cambiamos de caso. Clasificar correos como spam, y por qué cada error tiene un costo distinto.",
    CL(), siguiente(9, "Clasificación: el costo de cada error"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
