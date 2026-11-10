import math
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import D, out, tarjetas, siguiente, ilustrativo, real, caja_formula, diapo, etiqueta, cita, alerta, fila

OUT = out("guion_c38_02_metricas.json")
v = V("MAE, RMSE y R²")
RK = {r["modelo"]: r for r in D["h_ranking"]}
HF = D["h_final"]

v.portada("MAE, RMSE y R²", "Tres formas de resumir el error de una cantidad",
          "Segunda parada: cómo resumir en una cifra los errores de un modelo que predice una cantidad. El MAE, el RMSE y el R cuadrado.")

# ---- 1. Hook
v.b("Dos modelos predicen cuatro mezclas. [[a]]El modelo A acierta exacto en tres, y en la cuarta se equivoca por 8 megapascales. [[b]]El modelo B se equivoca por 3 en las cuatro. [[c]]¿Cuál es mejor? Vas a ver que con una métrica gana A, y con otra gana B.",
    CL(), T("¿cuál es mejor?", 960, 150, tam=56, c="tinta", ent="subir"),
    fila(300, [("", 420), ("caso 1", 720), ("caso 2", 960), ("caso 3", 1200), ("caso 4", 1440)], tams=[36] * 5, colores=["gris"] * 5),
    fila(410, [("A", 420), ("0", 720), ("0", 960), ("0", 1200), ("8", 1440)], tams=[54, 54, 54, 54, 54], colores=["azul"] * 4 + ["rojo"],
         en="a", fondo="azulclaro", x0=320, x1=1560, h=100),
    fila(540, [("B", 420), ("3", 720), ("3", 960), ("3", 1200), ("3", 1440)], tams=[54] * 5, colores=["ambar"] * 5,
         en="b", fondo="ambarclaro", x0=320, x1=1560, h=100),
    T("error absoluto en MPa", 960, 640, tam=34, c="gris", w="sb", en="a", t=0.6),
    P("una métrica elige A; otra elige B", 960, 800, "violeta", 50, en="c"), diapo(10, en="a"), sec="1. Dos modelos, cuatro casos")

# ---- 2. Analogía: dos buses
v.b("Piensa en dos buses que tomas cada mañana. [[a]]El bus A llega puntual tres días, y el cuarto llega 8 minutos tarde. [[b]]El bus B llega 3 minutos tarde todos los días. [[c]]Si solo te importa el atraso promedio, prefieres A. Pero si un atraso de 8 minutos te hace perder el tren, prefieres B: los atrasos grandes te duelen más.",
    CL(), T("dos buses", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "bus A", "0, 0, 0 y 8 minutos tarde", "a"), (1360, "ambar", "bus B", "3 minutos tarde, siempre", "b")],
             y=430, w=640, h=300, tam_t=52, tam_d=40),
    T("¿importa el promedio?  →  A", 960, 680, tam=44, c="azul", w="sb", en="c"),
    T("¿los atrasos grandes duelen más?  →  B", 960, 780, tam=44, c="ambar", w="sb", en="c", t=1.5),
    ilustrativo(1000, texto="analogía"), sec="2. Analogía: dos buses")

# ---- 3. Pérdida y riesgo
v.b("Para hablar con precisión, la clase usa dos palabras. [[a]]La pérdida penaliza el error de un caso. Por ejemplo, la pérdida cuadrática: el valor real menos la predicción, al cuadrado. [[b]]El riesgo es el promedio de esa pérdida en la población que nos interesa: los casos nuevos. [[c]]Una métrica resume los errores en una muestra; el riesgo describe lo que esperamos en casos nuevos.",
    CL(), T("pérdida y riesgo · diapositiva 8", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "violeta", "pérdida", "penaliza el error de UN caso", "a"),
              (1360, "teal", "riesgo", "el promedio de la pérdida en los casos nuevos", "b")], y=440, w=680, h=320, tam_t=52, tam_d=38),
    T("pérdida cuadrática:  L(y, ŷ) = (y − ŷ)²", 560, 660, tam=36, c="violeta", w="sb", en="a", t=0.8),
    P("métrica = muestra   ·   riesgo = casos nuevos", 960, 800, "azul", 46, en="c"), diapo(8, en="a"), sec="3. Pérdida y riesgo")

# ---- 4. MAE y RMSE: las dos pérdidas
ES = [x / 4 for x in range(-16, 17)]
v.b("Las dos métricas de la diapositiva nueve nacen de dos pérdidas. [[a]]A la izquierda, la pérdida absoluta: cada megapascal adicional de error pesa lo mismo. [[b]]A la derecha, la pérdida cuadrática: un error de 4 pesa 16, cuatro veces más que un error de 2, que pesa 4. [[c]]Los errores grandes pesan mucho más.",
    CL(), T("pérdida absoluta y pérdida cuadrática · diapositiva 9", 960, 140, tam=46, c="tinta", ent="subir"),
    G(240, 280, 860, 800, -4, 4, 0, 4.5, [dict(xs=ES, ys=[abs(e) for e in ES], color="azul", grosor=6, revela=0.0)],
      xticks=[-4, -2, 0, 2, 4], yticks=[0, 1, 2, 3, 4], yfmt="{:.0f}", xlabel="error (MPa)", ylabel="|error|", id="ga", tam_txt=26),
    U("ga", **{"series.0.revela": 1.0}, en="a", dur=1.2),
    G(1120, 280, 1740, 800, -4, 4, 0, 17, [dict(xs=ES, ys=[e * e for e in ES], color="rojo", grosor=6, revela=0.0)],
      xticks=[-4, -2, 0, 2, 4], yticks=[0, 4, 8, 12, 16], yfmt="{:.0f}", xlabel="error (MPa)", ylabel="error²", id="gc", tam_txt=26, en="b"),
    U("gc", **{"series.0.revela": 1.0}, en="b", t=0.3, dur=1.2),
    P("error 2 → 4   ·   error 4 → 16", 1430, 900, "rojo", 40, en="b", t=1.6),
    P("los errores grandes pesan más", 960, 990, "ambar", 44, en="c"), sec="4. Dos pérdidas")
v.b("De ahí salen las fórmulas. [[a]]El MAE, error absoluto medio: el promedio de los errores en valor absoluto. [[b]]El RMSE, raíz del error cuadrático medio: elevo cada error al cuadrado, promedio, y saco la raíz. [[c]]La raíz devuelve las unidades: los dos se expresan en megapascales.",
    CL(), T("MAE y RMSE", 960, 150, tam=56, c="tinta", ent="subir"),
    caja_formula("MAE = (1/n) · Σ |yᵢ − ŷᵢ|", y=330, c="azul", tam=52, w=1200, en="a"),
    caja_formula("RMSE = √[ (1/n) · Σ (yᵢ − ŷᵢ)² ]", y=530, c="rojo", tam=52, w=1200, en="b"),
    P("los dos en MPa", 960, 720, "teal", 50, en="c"), diapo(9, en="a"))

# ---- 5. Ejemplo resuelto: cambia el ganador
v.b("Volvamos a los dos modelos. [[a]]MAE de A: cero más cero más cero más ocho, sobre cuatro: 2. MAE de B: tres por cuatro, sobre cuatro: 3. [[b]]Ahora el RMSE de A: ocho al cuadrado es 64; 64 sobre cuatro es 16; la raíz de 16 es 4. [[c]]Y el de B: nueve por cuatro, sobre cuatro, es nueve; la raíz es 3.",
    CL(), T("el ejemplo de la diapositiva 10, a mano", 960, 140, tam=50, c="tinta", ent="subir"),
    T("A: errores 0, 0, 0, 8", 520, 260, tam=42, c="azul"), T("B: errores 3, 3, 3, 3", 1400, 260, tam=42, c="ambar"),
    T("MAE = 8 / 4 =", 660, 400, tam=44, c="tinta", w="sb", anc="rm", en="a"), N(2, 690, 400, "{:.0f}", 64, "azul", anc="lm", en="a", t=0.4),
    T("MAE = 12 / 4 =", 1520, 400, tam=44, c="tinta", w="sb", anc="rm", en="a", t=1.5), N(3, 1550, 400, "{:.0f}", 64, "ambar", anc="lm", en="a", t=1.9),
    T("RMSE = √(64 / 4) = √16 =", 660, 560, tam=44, c="tinta", w="sb", anc="rm", en="b"), N(4, 690, 560, "{:.0f}", 64, "azul", anc="lm", en="b", t=1.5),
    T("RMSE = √(36 / 4) = √9 =", 1520, 560, tam=44, c="tinta", w="sb", anc="rm", en="c"), N(3, 1550, 560, "{:.0f}", 64, "ambar", anc="lm", en="c", t=1.2),
    diapo(10, en="a"), sec="5. El ejemplo a mano")
v.b("Resultado: [[a]]con el MAE elegimos A, porque 2 es menor que 3. [[b]]Con el RMSE elegimos B, porque 3 es menor que 4. [[c]]Por eso la diapositiva dice: el criterio se define antes de consultar los resultados. Si no, siempre podrás encontrar una métrica que favorezca al modelo que ya te gustaba.",
    P("MAE → gana A  (2 < 3)", 560, 730, "azul", 46, en="a"), P("RMSE → gana B  (3 < 4)", 1400, 730, "ambar", 46, en="b"),
    P("el criterio se define ANTES de ver los resultados", 960, 880, "violeta", 46, en="c"))

# ---- 6. R² y su referencia
v.b("La tercera métrica es el R cuadrado. [[a]]Es uno menos un cociente: arriba, la suma de los errores al cuadrado del modelo; abajo, la suma de los errores al cuadrado de predecir siempre la media del conjunto evaluado. [[b]]Compara el modelo con esa referencia. [[c]]No tiene unidades, así que no dice cuántos megapascales nos equivocamos.",
    CL(), T("el R² necesita una referencia · diapositiva 11", 960, 150, tam=50, c="tinta", ent="subir"),
    caja_formula("R² = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²", y=330, c="violeta", tam=54, w=1300, en="a"),
    T("modelo", 1080, 430, tam=32, c="violeta", w="sb", en="a", t=0.6), T("siempre la media del conjunto evaluado", 1300, 470, tam=30, c="gris", w="sb", en="a", t=0.9),
    P("compara contra esa referencia", 960, 600, "teal", 46, en="b"), P("sin unidades: no dice cuántos MPa", 960, 730, "ambar", 46, en="c"),
    diapo(11, en="a"), sec="6. El R² y su referencia")

# ---- 7. Alerta del material
v.b("Aquí hay un detalle que conviene precisar. [[a]]En la grabación, al definir el R cuadrado, se dijo: «es un número que está entre 0 y 1». [[b]]Eso vale para el ajuste de una regresión lineal con intercepto, en sus propios datos de entrenamiento. [[c]]Pero en datos de validación o de prueba, el R cuadrado puede ser negativo. Las diapositivas once y doce lo muestran.",
    CL(), alerta(170), cita("es un número que está entre 0 y 1", 330, en="a", tam=50),
    etiqueta("grabación de la clase, al definir el R²", 420, en="a", t=0.4),
    T("cierto para el ajuste lineal con intercepto, en sus propios datos", 960, 560, tam=40, c="gris", w="sb", en="b"),
    P("en validación o prueba puede ser NEGATIVO · diapositivas 11 y 12", 960, 720, "rojo", 46, en="c"), sec="7. ¿Siempre entre 0 y 1?")
v.b("El ejemplo de la diapositiva doce lo deja clarísimo. [[a]]Valores reales: uno, dos y tres. Predicciones: once, doce y trece. [[b]]La correlación es perfecta, 1: el orden se conserva. Pero cada predicción está corrida en diez, así que el RMSE es 10. [[c]]Y el R cuadrado: arriba, tres errores de diez al cuadrado, 300; abajo, la variación de uno, dos, tres respecto de su media, 2. Uno menos 300 sobre 2: menos 149.",
    CL(), T("correlación perfecta, R² muy negativo · diapositiva 12", 960, 140, tam=48, c="tinta", ent="subir"),
    T("y = (1, 2, 3)     ŷ = (11, 12, 13)", 960, 270, tam=50, c="tinta", w="sb", en="a"),
    T("correlación² = 1", 560, 410, tam=46, c="verde", w="sb", en="b"), T("RMSE = 10", 1360, 410, tam=46, c="rojo", w="sb", en="b", t=1.5),
    T("Σ (y − ŷ)² = 100 + 100 + 100 = 300      Σ (y − ȳ)² = 1 + 0 + 1 = 2", 960, 560, tam=40, c="tinta", w="sb", en="c"),
    T("R² = 1 − 300 / 2 =", 1020, 700, tam=54, c="tinta", w="sb", anc="rm", en="c", t=2.0), N(-149, 1050, 700, "{:.0f}", 72, "rojo", anc="lm", en="c", t=2.6),
    diapo(12, en="a"))
v.b("Y con los datos reales pasa lo mismo, en pequeño. [[a]]El modelo que siempre predice la media del entrenamiento tiene, fuera de pliegue, un R cuadrado de menos 0.002. [[b]]Y en la prueba, menos 0.007. [[c]]Negativo, porque la media del entrenamiento no es la media del conjunto evaluado. Más adelante en la clase se dijo bien: «al tener un número negativo significa que aquí hay problemas».",
    CL(), T("la media de entrenamiento, con datos reales", 960, 150, tam=50, c="tinta", ent="subir"),
    T("R² fuera de pliegue", 560, 320, tam=40, c="gris", w="sb", en="a"), N(RK["MEDIA"]["r2"], 560, 420, "{:.3f}", 84, "rojo", en="a", t=0.3),
    T("R² en la prueba", 1360, 320, tam=40, c="gris", w="sb", en="b"), N(HF["ref_r2"], 1360, 420, "{:.3f}", 84, "rojo", en="b", t=0.3),
    T("la media del entrenamiento ≠ la media del conjunto evaluado", 960, 590, tam=40, c="tinta", w="sb", en="c"),
    cita("al tener un número negativo significa que aquí hay problemas", 740, en="c", t=1.5, tam=40),
    real(1050, en="a"))

# ---- 8. Cómo las usa la clase
v.b("Entonces, ¿cómo las usa la clase? [[a]]El RMSE se usa para seleccionar. [[b]]El MAE y el R cuadrado, para describir. [[c]]Y la referencia se fija antes de evaluar: predecir la media del entrenamiento. Con datos reales, el modelo elegido tiene, fuera de pliegue, un RMSE de 8.229 megapascales, un MAE de 6.234, y un R cuadrado de 0.758.",
    CL(), T("el papel de cada métrica · diapositiva 11", 960, 150, tam=50, c="tinta", ent="subir"),
    P("RMSE → seleccionar", 560, 300, "rojo", 50, en="a"), P("MAE y R² → describir", 1360, 300, "azul", 50, en="b"),
    T("referencia fijada antes: la media del entrenamiento", 960, 430, tam=40, c="gris", w="sb", en="c"),
    fila(570, [("RMSE", 560), ("MAE", 960), ("R²", 1360)], tams=[40, 40, 40], colores=["gris"] * 3, en="c", t=1.5),
    N(RK["Q_A0.1"]["rmse"], 560, 670, "{:.3f}", 70, "rojo", en="c", t=2.0), N(RK["Q_A0.1"]["mae"], 960, 670, "{:.3f}", 70, "azul", en="c", t=2.6),
    N(RK["Q_A0.1"]["r2"], 1360, 670, "{:.3f}", 70, "violeta", en="c", t=3.2),
    T("cuadrática con ridge, α = 0.1 · predicciones fuera de pliegue", 960, 790, tam=32, c="gris", w="sb", en="c", t=2.0),
    real(1050, en="c"), sec="8. Cómo las usa la clase")

# ---- 9. El error típico
v.b("El error típico: [[a]]elegir la métrica después de mirar los resultados. [[b]]Primero veo que mi modelo favorito gana en MAE, y entonces decido que el MAE era lo importante. [[c]]El profesor lo resumió así: el criterio se fija antes, porque si no, «es como hacer trampa».",
    CL(), K(330, 260, "cruz", 70, en="a"), T("elegir la métrica después de ver los resultados", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("«gana en MAE… entonces el MAE era lo importante»", 960, 430, tam=42, c="gris", w="sb", en="b"),
    cita("es como hacer trampa", 600, en="c", tam=54), etiqueta("cita de la grabación de la clase", 690, en="c"), sec="9. El error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un modelo comete estos cuatro errores: 4, 0, menos 2 y 2 megapascales. [[b]]¿Cuál es su MAE? ¿Y su RMSE?",
    CL(), T("errores: 4, 0, −2, 2 (MPa)", 960, 300, tam=60, c="tinta", w="b", en="a"),
    P("MAE = ?", 700, 520, "azul", 54, en="b"), P("RMSE = ?", 1220, 520, "rojo", 54, en="b", t=0.4), sec="10. Piensa un segundo", pausa=0.6)
v.b("El MAE: cuatro, más cero, más dos, más dos, sobre cuatro: [[a]]2. [[b]]El RMSE: dieciséis, más cero, más cuatro, más cuatro, son 24; sobre cuatro, 6; y la raíz de 6, [[c]]2.449. El RMSE sale más alto que el MAE porque el error de 4 pesa más. Y si todos los errores fueran iguales, coincidirían.",
    CL(), T("MAE = (4 + 0 + 2 + 2) / 4 =", 1100, 320, tam=48, c="tinta", w="sb", anc="rm"), N(2, 1130, 320, "{:.0f}", 64, "azul", anc="lm", en="a"),
    T("RMSE = √[(16 + 0 + 4 + 4) / 4] = √6 =", 1100, 470, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(2.449, 1130, 470, "{:.3f}", 64, "rojo", anc="lm", en="c"),
    T("RMSE ≥ MAE; iguales solo si todos los errores tienen el mismo tamaño", 960, 640, tam=38, c="gris", w="sb", en="c", t=1.2),
    ilustrativo(1000))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]el MAE pesa igual cada megapascal de error; el RMSE pesa más los errores grandes, y los dos están en megapascales. [[b]]El R cuadrado compara con la media del conjunto evaluado, no tiene unidades, y puede ser negativo. [[c]]Y la métrica de selección se fija antes de mirar los resultados.",
    CL(), P("MAE: cada MPa pesa igual · RMSE: los errores grandes pesan más", 960, 300, "azul", 42, en="a"),
    P("R²: sin unidades, contra la media evaluada, puede ser < 0", 960, 460, "violeta", 44, en="b"),
    P("la métrica se fija antes de mirar", 960, 620, "teal", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]cómo partir los datos. Por qué en el hormigón no se separan filas, sino composiciones completas.",
    CL(), siguiente(3, "Separar por composición: grupos y prueba reservada"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
