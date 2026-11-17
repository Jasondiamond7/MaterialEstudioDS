import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo, error_material)

OUT = out("guion_c40_11_umbral.json")
v = V("Decidir: umbral por costo")
U_, PR, O = D["umbral"], D["prueba"], D["oof"]
TB = {round(r["t"], 2): r for r in U_["tabla"]}

v.portada("Decidir: umbral por costo", "De la probabilidad a la acción: bloquear o dejar pasar",
          "Undécima parada. Hasta ahora predijimos. Ahora decidimos: con qué probabilidad conviene bloquear un correo, si los dos errores no cuestan lo mismo.")

# ---- 1. Los costos
v.b("La diapositiva cincuenta y uno fija los costos del ejemplo. [[a]]Un falso positivo bloquea un correo legítimo: cuesta cinco. [[b]]Un falso negativo deja pasar un spam: cuesta uno. Los aciertos cuestan cero. [[c]]El profesor lo justificó así: bloquear un correo legítimo es mucho peor que dejar pasar un spam, que solo es tedioso. Los números son de ejemplo; en la práctica salen del contexto: dinero, tiempo, importancia.",
    CL(), T("¿cuándo conviene bloquear un correo? · diapositiva 51", 960, 140, tam=48, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "falso positivo: costo 5", "bloquear un correo legítimo", "a"), (1300, "ambar", "falso negativo: costo 1", "dejar pasar un spam", "b")],
             y=430, w=600, h=280, tam_t=46, tam_d=40),
    T("los aciertos cuestan cero  ·  los costos salen del contexto del problema", 960, 700, tam=40, c="tinta", w="sb", en="c"),
    diapo(51, en="a"), sec="1. Los costos")
v.b("Si p es la probabilidad de spam, cada acción tiene un costo esperado. [[a]]Bloquear arriesga el costo de un legítimo: cinco por uno menos p. Dejar pasar arriesga el de un spam: uno por p. [[b]]Con p igual a 0.90: bloquear cuesta cinco por 0.10, 0.50; dejar pasar cuesta 0.90. Conviene bloquear. [[c]]Con p igual a 0.70: bloquear cuesta cinco por 0.30, 1.50; dejar pasar cuesta 0.70. Conviene dejar pasar.",
    CL(), T("costo esperado de cada acción", 960, 140, tam=54, c="tinta", ent="subir"),
    P("bloquear: 5 (1 − p)   ·   dejar pasar: 1 · p", 960, 270, "violeta", 48, en="a"),
    tabla([["probabilidad", "bloquear", "dejar pasar", "menor costo"], ["p = 0.90", "5(1 − 0.90) = 0.50", "1(0.90) = 0.90", "bloquear"],
           ["p = 0.70", "5(1 − 0.70) = 1.50", "1(0.70) = 0.70", "dejar pasar"]], [380, 830, 1250, 1580], 420, paso=120, tam=42, x0=170, x1=1770,
          colores=[["tinta"] * 4, ["tinta", "tinta", "tinta", "rojo"], ["tinta", "tinta", "tinta", "verde"]], ens=[None, "b", "c"]),
    T("costos y probabilidades hipotéticos para explicar la decisión", 960, 820, tam=32, c="gris", w="sb", en="c", t=1.0), diapo(51, en="a"))
v.b("¿Dónde está la frontera? Un cálculo adicional. [[a]]Bloquear conviene cuando cinco por uno menos p es menor que p, es decir, cuando p supera cinco sextos, aproximadamente 0.833. [[b]]Pero eso vale solo si las probabilidades están bien calibradas. Y en la parada anterior vimos que no lo están: exageran. [[c]]Por eso la diapositiva propone otra cosa: comparar umbrales con los errores observados.",
    CL(), T("la frontera teórica", 960, 150, tam=56, c="tinta", ent="subir"),
    T("5 (1 − p) < p   →   p > 5/6 ≈", 1000, 330, tam=54, c="tinta", w="sb", anc="rm", en="a"), N(5 / 6, 1030, 330, "{:.3f}", 70, "violeta", anc="lm", en="a", t=1.5),
    P("solo si las probabilidades están calibradas", 960, 500, "ambar", 46, en="b"),
    P("mejor: comparar umbrales con los errores observados", 960, 660, "teal", 46, en="c"), adicional(1050, en="a"))

# ---- 2. Analogía / cita
v.b("El profesor dio un ejemplo fuera del correo. [[a]]Si un diagnóstico de cáncer tiene un costo de error muy alto, una probabilidad de setenta u ochenta por ciento puede ser baja para decidir. [[b]]Quizás se exija noventa y cinco o noventa y ocho por ciento. [[c]]El umbral no es una propiedad del modelo: sale del costo de equivocarse. El mismo modelo, con otros costos, toma otra decisión.",
    CL(), cita("si el costo de equivocarse es muy grande voy a querer estar muy muy seguro", y=270, tam=44),
    P("80 % puede ser poco si el error es grave", 620, 480, "rojo", 42, en="a"), P("exigir 95 % o 98 %", 1300, 480, "ambar", 42, en="b"),
    P("el umbral sale del costo, no del modelo", 960, 680, "violeta", 50, en="c"), grabacion("02:18:52"), sec="2. El umbral sale del costo")

# ---- 3. Buscar el umbral
v.b("La diapositiva cincuenta y dos busca el umbral en las predicciones del ganador. [[a]]La función medir cuenta falsos positivos y falsos negativos, y calcula mil por cinco FP más FN, dividido por n: el costo por cada mil correos. [[b]]Se prueban diecinueve umbrales, de 0.05 a 0.95, con las probabilidades fuera de pliegue de Bernoulli con alfa 0.5. [[c]]Se elige el de menor costo, y si dos empatan, el umbral mayor.",
    CL(), T("buscar el umbral en las predicciones del ganador · diapositiva 52", 960, 130, tam=46, c="tinta", ent="subir"),
    caja_formula("costo por 1000 = 1000 · (5 FP + FN) / n", y=270, c="violeta", tam=50, w=1200, en="a"),
    codigo(["umbrales = np.arange(1, 20)/20", "costos = pd.DataFrame([dict(umbral=t, **medir(y, oof[ganador], t)) for t in umbrales])",
            'elegido = costos.sort_values(["costo_1000", "umbral"], ascending=[True, False]).iloc[0]'], y=520, w=1760, tam=30, c="teal", en="b"),
    P("19 umbrales (0.05 a 0.95) · OOF de Bernoulli α = 0.5 · empate → el mayor", 960, 760, "ambar", 40, en="c"), diapo(52, en="a"), sec="3. Buscar el umbral")

# ---- 4. La curva de costo
GU = "gu11"
v.b("Y este es el resultado, de la diapositiva cincuenta y tres. [[a]]Con umbrales bajos, el costo es alto: con 0.05 llega a cuatrocientos cuarenta y nueve por mil correos. [[b]]A medida que el umbral sube, el costo baja. [[c]]El mínimo de la rejilla está en 0.95, con un costo de 219.0.",
    CL(), T("el umbral elegido cambia los errores del filtro · diapositiva 53", 960, 120, tam=46, c="tinta", ent="subir"),
    G(330, 220, 1240, 830, 0, 1, 200, 460, [dict(xs=[r["t"] for r in U_["tabla"]], ys=[r["costo"] for r in U_["tabla"]], color="azul", grosor=5, marcadores=9, revela=0.0)],
      xticks=[0.2, 0.4, 0.6, 0.8], yticks=[250, 300, 350, 400, 450], yfmt="{:.0f}", xlabel="umbral", id=GU, tam_txt=26), T("costo OOF por 1000 correos", 330, 180, tam=30, c="tinta", w="sb", anc="lm"),
    U(GU, **{"series.0.revela": 0.15}, en="a", dur=1.0), U(GU, **{"series.0.revela": 1.0}, en="b", dur=2.0),
    R(0.95, TB[0.95]["costo"], 60, 60, "ambar", r=30, ref=GU, en="c"),
    T("0.05: 448.9", 1520, 330, tam=40, c="azul", w="b", en="a"), T("umbral elegido: 0.95", 1520, 470, tam=42, c="ambar", w="b", en="c"),
    T("costo: 219.0", 1520, 550, tam=42, c="ambar", w="b", en="c", t=0.3), diapo(53, en="a"), real(1010, en="a"), sec="4. La curva de costo")
v.b("Con umbral 0.95 hay setenta y ocho legítimos bloqueados y cuatrocientos dieciséis spam que pasan, entre tres mil seiscientos ochenta correos. [[a]]El costo: cinco por setenta y ocho, más cuatrocientos dieciséis, dividido por tres mil seiscientos ochenta, por mil. [[b]]Es decir, ochocientos seis sobre tres mil seiscientos ochenta, por mil: 219.0.",
    CL(), T("la cuenta con umbral 0.95", 960, 150, tam=54, c="tinta", ent="subir"),
    T("FP = 78 legítimos bloqueados   ·   FN = 416 spam que pasan", 960, 300, tam=44, c="tinta", w="sb"),
    T("costo por 1000 = (5 × 78 + 416) / 3680 × 1000 = 806 / 3680 × 1000 =", 1440, 460, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(TB[0.95]["costo"], 1470, 460, "{:.1f}", 64, "ambar", anc="lm", en="b"), diapo(53, en="a"))
v.b("Comparemos algunos umbrales de la misma tabla. [[a]]Con 0.5: ciento cincuenta y dos falsos positivos y doscientos sesenta y tres falsos negativos; costo 278.0. [[b]]Con 0.8: ciento once y trescientos veintiocho; costo 239.9. Con 0.9: noventa y dos y trescientos ochenta y tres; costo 229.1. [[c]]Subir el umbral cambia falsos positivos caros por falsos negativos baratos.",
    CL(), T("cómo cambian los errores con el umbral", 960, 140, tam=54, c="tinta", ent="subir"),
    tabla([["umbral", "FP (costo 5)", "FN (costo 1)", "costo por 1000"]] + [[f"{t:.2f}", str(TB[t]["FP"]), str(TB[t]["FN"]), f"{TB[t]['costo']:.1f}"] for t in (0.5, 0.8, 0.9, 0.95)],
          [440, 820, 1180, 1520], 290, paso=105, tam=44, x0=200, x1=1740, ens=[None, "a", "b", "b", "b"]),
    P("subir el umbral: menos FP caros, más FN baratos", 960, 860, "violeta", 44, en="c"), real(1050, en="a"))

# ---- 5. Por qué no irse al extremo
v.b("El profesor explicó la forma de la curva. [[a]]Con un umbral muy bajo, se bloquea casi todo, y se bloquean muchos legítimos, que son caros. [[b]]Con un umbral muy alto, no se bloquea nada, y queda un costo base: todos los spam pasan. Con estos datos, eso cuesta mil cuatrocientos cincuenta sobre tres mil seiscientos ochenta, por mil: 394.0. [[c]]Y la diapositiva agrega una advertencia: que el mínimo esté en el borde de la rejilla no demuestra que 0.95 sea el mejor umbral posible.",
    CL(), T("ni todo bloqueado, ni nada bloqueado", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(620, "rojo", "umbral muy bajo", "se bloquean muchos legítimos (costo 5)", "a"),
              (1300, "ambar", "umbral muy alto", f"todo pasa: 1450 / 3680 × 1000 = {O['FRECUENCIA']['costo']:.1f}", "b")], y=420, w=600, h=280, tam_t=48, tam_d=38),
    P("0.95 está en el borde de la rejilla: no demuestra que sea el mejor posible", 960, 700, "violeta", 40, en="c"),
    grabacion("02:09:50", en="a"), diapo(53, en="c", y=1010), sec="5. Ni todo ni nada")
v.b("Un dato adicional que muestra que decidir no es acertar. [[a]]La exactitud fuera de pliegue con umbral 0.5 es 88.7 por ciento; con 0.95 baja a 86.6 por ciento. [[b]]El umbral que minimiza el costo no es el que maximiza los aciertos: acepta más errores baratos para evitar los caros.",
    CL(), T(f"exactitud OOF:  0.5 → {100 * TB[0.5]['acc']:.1f} %   ·   0.95 → {100 * TB[0.95]['acc']:.1f} %", 960, 360, tam=50, c="tinta", w="b", en="a"),
    P("menor costo ≠ más aciertos", 960, 540, "violeta", 56, en="b"), adicional(1050, en="a"))

# ---- 6. Comprobación histórica
v.b("Ahora sí, los novecientos veintiuno correos históricos, en la diapositiva cincuenta y cuatro. Con Bernoulli alfa 0.5 y umbral 0.95, elegidos en desarrollo. [[a]]Filas: la clase real. Columnas: la decisión. Legítimos dejados pasar: quinientos cuarenta y uno; legítimos bloqueados: diecisiete. [[b]]Spam que pasan: ochenta y nueve; spam bloqueados: doscientos setenta y cuatro. [[c]]El costo: cinco por diecisiete, más ochenta y nueve, sobre novecientos veintiuno, por mil: 188.9.",
    CL(), T("qué obtiene la regla en los 921 correos históricos · diapositiva 54", 960, 120, tam=44, c="tinta", ent="subir"),
    T("decisión: legítimo", 820, 240, tam=34, c="gris", w="b", en="a"), T("decisión: spam", 1160, 240, tam=34, c="gris", w="b", en="a"),
    T("real: legítimo", 520, 340, tam=34, c="azul", w="b", anc="rm", en="a"), T("real: spam", 520, 480, tam=34, c="rojo", w="b", anc="rm", en="b"),
    Bc(820, 340, 300, 120, "verdeclaro", None, 0, 16, en="a"), T(str(PR["VN"]), 820, 340, tam=60, c="verde", w="b", en="a"),
    Bc(1160, 340, 300, 120, "rojoclaro", None, 0, 16, en="a"), T(str(PR["FP"]), 1160, 340, tam=60, c="rojo", w="b", en="a"),
    Bc(820, 480, 300, 120, "ambarclaro", None, 0, 16, en="b"), T(str(PR["FN"]), 820, 480, tam=60, c="ambar", w="b", en="b"),
    Bc(1160, 480, 300, 120, "verdeclaro", None, 0, 16, en="b"), T(str(PR["VP"]), 1160, 480, tam=60, c="verde", w="b", en="b"),
    T("(5 × 17 + 89) / 921 × 1000 =", 1120, 660, tam=46, c="tinta", w="sb", anc="rm", en="c"), N(PR["costo"], 1150, 660, "{:.1f}", 66, "violeta", anc="lm", en="c", t=1.5),
    T("Bernoulli α = 0.5 · umbral 0.95 · log loss 0.533 · Brier 0.094", 960, 800, tam=36, c="gris", w="sb", en="c", t=2.0),
    diapo(54, en="a"), real(1010, en="a"), sec="6. Comprobación histórica")
v.b("¿Es bueno 188.9? Hay que compararlo con algo. [[a]]Si dejamos pasar todos los correos, pasan los ochenta y nueve más doscientos setenta y cuatro: trescientos sesenta y tres spam. [[b]]El costo: trescientos sesenta y tres sobre novecientos veintiuno, por mil: 394.1. [[c]]La regla reduce el costo a menos de la mitad frente a no filtrar nada. Además, el log loss en estos correos es 0.533 y el Brier, 0.094: parecidos a los de desarrollo.",
    CL(), T("compararlo con no filtrar nada", 960, 150, tam=54, c="tinta", ent="subir"),
    T("dejar pasar todo: 89 + 274 = 363 spam pasan → 363 / 921 × 1000 =", 1400, 330, tam=40, c="tinta", w="sb", anc="rm", en="a"),
    N(PR["base_costo"], 1430, 330, "{:.1f}", 60, "rojo", anc="lm", en="b"),
    barras_v([("regla (α = 0.5, umbral 0.95)", PR["costo"], "violeta", "c"), ("dejar pasar todo", PR["base_costo"], "rojo", "c")],
             x0=760, paso=400, base=860, escala=0.9, w=220, fmt="{:.1f}", tam=38), diapo(54, en="a"))
v.b("Y las advertencias de la diapositiva, que son parte del resultado. [[a]]Es una comprobación retrospectiva: estos correos ya se habían usado al comparar otros clasificadores. [[b]]El modelo y el umbral se eligieron en desarrollo; los correos de prueba no participaron en esas elecciones. [[c]]Y el resultado no acredita rendimiento en correos actuales: son correos de hace más de veinte años.",
    CL(), P("comprobación retrospectiva: ya se habían usado", 960, 320, "ambar", 46, en="a"),
    P("modelo y umbral elegidos en desarrollo", 960, 480, "teal", 46, en="b"),
    P("no acredita rendimiento en correos actuales", 960, 640, "rojo", 46, en="c"), diapo(54, en="a"))

# ---- 7. Errores típicos
v.b("Errores típicos al decidir. [[a]]Uno: usar 0.5 por costumbre, como si los dos errores costaran lo mismo. [[b]]Dos: elegir el umbral mirando los correos de prueba; entonces la prueba deja de ser una comprobación. [[c]]Y tres: usar la frontera teórica, cinco sextos, con probabilidades que no están calibradas.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("usar 0.5 por costumbre", 410, 300, tam=44, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("elegir el umbral mirando la prueba", 410, 450, tam=44, c="rojo", anc="lm", en="b"),
    K(330, 600, "cruz", 64, en="c"), T("usar 5/6 con probabilidades no calibradas", 410, 600, tam=44, c="rojo", anc="lm", en="c"),
    sec="7. Errores típicos")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si bloquear un legítimo costara dos en vez de cinco, y las probabilidades estuvieran bien calibradas, [[b]]¿a partir de qué probabilidad de spam convendría bloquear?",
    CL(), P("FP = 2   ·   FN = 1   ·   probabilidades calibradas", 960, 330, "teal", 50, en="a"),
    P("bloquear si p > ?", 960, 520, "violeta", 60, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("Bloquear conviene si dos por uno menos p es menor que p: [[a]]es decir, si p supera dos tercios, [[b]]aproximadamente 0.667. Un error menos caro baja el umbral: se bloquea con menos certeza.",
    CL(), T("2 (1 − p) < p   →   p > 2/3 ≈", 1000, 360, tam=56, c="tinta", w="sb", anc="rm", en="a"), N(2 / 3, 1030, 360, "{:.3f}", 72, "violeta", anc="lm", en="b"),
    P("error menos caro → umbral más bajo", 960, 540, "teal", 48, en="b", t=1.0), ilustrativo(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]el umbral sale de los costos: con falso positivo cinco y falso negativo uno, se compara cinco por uno menos p con p. [[b]]Con las predicciones fuera de pliegue se eligió 0.95, con costo 219.0 por mil, en el borde de la rejilla. [[c]]En los novecientos veintiuno correos históricos: diecisiete legítimos bloqueados, ochenta y nueve spam que pasan, y costo 188.9, frente a 394.1 sin filtrar.",
    CL(), P("bloquear si 5 (1 − p) < 1 · p, con costos del contexto", 960, 300, "violeta", 46, en="a"),
    P("OOF: umbral 0.95 · costo 219.0 por 1000", 960, 460, "teal", 48, en="b"),
    P("prueba histórica: 188.9 frente a 394.1 sin filtrar", 960, 620, "ambar", 48, en="c"), sec="9. Para llevar")
v.b("Siguiente y última parada: [[a]]qué revisar antes de usar el filtro en otra bandeja, las bibliotecas de R y Python, y el repaso de toda la clase.",
    CL(), siguiente(12, "Límites, bibliotecas y repaso"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
