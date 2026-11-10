import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, alerta, fila, m2)

OUT = out("guion_c38_09_costo.json")
v = V("Clasificación: el costo de cada error")
SP = D["s_part"]

v.portada("El costo de cada error", "Clasificar spam: no todos los errores pesan igual",
          "Novena parada: cambiamos de caso. Ahora clasificamos correos como spam, y vamos a ver por qué cada tipo de error tiene un costo distinto.")

# ---- 1. Hook
v.b("Te ofrezco un filtro de spam que acierta el 95 por ciento de los correos. [[a]]¿Lo instalas? [[b]]Antes, una pregunta: de cada cien correos llegan cinco de spam, y este filtro… nunca marca nada como spam. [[c]]Acierta 95 de cien, y detecta cero de los cinco. Un porcentaje alto de aciertos puede ser insuficiente.",
    CL(), T("un filtro que acierta el 95 %", 960, 150, tam=56, c="tinta", ent="subir"),
    P("¿lo instalas?", 960, 290, "violeta", 50, en="a"),
    *[Bc(420 + 54 * (i % 20), 420 + 60 * (i // 20), 44, 44, "rojoclaro" if i in (7, 23, 48, 66, 91) else "azulclaro",
         "rojo" if i in (7, 23, 48, 66, 91) else "azul", 3, 10, en="b", t=0.01 * i) for i in range(100)],
    T("5 de spam · 95 legítimos", 960, 760, tam=40, c="gris", w="sb", en="b", t=1.0),
    P("acierta 95 de 100 · detecta 0 de 5", 960, 880, "rojo", 50, en="c"), diapo(35, en="b"), sec="1. El filtro del 95 %")

# ---- 2. El caso spam
v.b("Este es el segundo caso de la clase, en la diapositiva treinta y tres. [[a]]La base Spambase, del repositorio UCI: 4601 correos, con 57 entradas numéricas, como frecuencias de palabras, de caracteres, y medidas de mayúsculas. [[b]]La clase positiva, y igual a uno, es spam. [[c]]Se reservan 921 correos para la prueba, y se usan 3680 para ajustar y seleccionar. En el entrenamiento hay 2230 correos legítimos, el 60.6 por ciento, y 1450 de spam, el 39.4.",
    CL(), T("caso 2 · detectar spam · diapositiva 33", 960, 140, tam=50, c="tinta", ent="subir"),
    N(SP["n"], 520, 330, "{:.0f}", 100, "azul", en="a"), T("correos", 520, 420, tam=36, c="gris", w="sb", en="a", t=0.3),
    N(SP["vars"], 1400, 330, "{:.0f}", 100, "violeta", en="a", t=0.6), T("entradas numéricas", 1400, 420, tam=36, c="gris", w="sb", en="a", t=0.9),
    P("y = 1: spam   ·   y = 0: correo legítimo", 960, 540, "rojo", 46, en="b"),
    Bx(320, 640, 320 + 1280 * SP["n0"] / SP["tr"], 740, "azulclaro", "azul", 3, 14, grow="x", en="c"),
    T(f"{SP['n0']} legítimos (60.6 %)", 320 + 640 * SP["n0"] / SP["tr"], 690, tam=34, c="azul", en="c", t=0.5),
    Bx(330 + 1280 * SP["n0"] / SP["tr"], 640, 1600, 740, "rojoclaro", "rojo", 3, 14, grow="x", en="c", t=0.3),
    T(f"{SP['n1']} spam (39.4 %)", (330 + 1280 * SP["n0"] / SP["tr"] + 1600) / 2, 690, tam=34, c="rojo", en="c", t=0.7),
    T("entrenamiento: 3680 correos   ·   prueba reservada: 921", 960, 810, tam=36, c="gris", w="sb", en="c", t=1.0),
    real(1050, en="a"), sec="2. El caso del spam")
v.b("La lógica es la misma del hormigón: [[a]]separar la selección de la evaluación, con la prueba reservada. [[b]]Pero ahora aparece algo nuevo: hay que distinguir la calidad de las probabilidades, y el costo de decidir. [[c]]Y una advertencia de la diapositiva: es un corpus histórico; no es una prueba de un filtro actual.",
    CL(), T("misma lógica, otras preguntas", 960, 150, tam=56, c="tinta", ent="subir"),
    P("separar selección y evaluación", 960, 300, "violeta", 48, en="a"),
    tarjetas([(560, "teal", "probabilidades", "¿son buenas?", "b"), (1360, "ambar", "decisiones", "¿cuánto cuestan sus errores?", "b")],
             y=520, w=600, h=260, tam_t=48, tam_d=40),
    T("corpus histórico: no prueba un filtro actual", 960, 780, tam=40, c="gris", w="sb", en="c"), diapo(33, en="a"))

# ---- 3. Dos errores
v.b("En la diapositiva treinta y cuatro aparecen los dos errores. [[a]]Falso positivo: el correo es legítimo y lo marcamos como spam. La consecuencia: ocultar un mensaje que el usuario necesitaba recibir. [[b]]Falso negativo: el correo es spam y lo dejamos pasar. La consecuencia: ruido y trabajo adicional en la bandeja de entrada. [[c]]Contar errores sin distinguir su tipo supone que equivocarse cuesta lo mismo en ambos sentidos.",
    CL(), T("dos errores con consecuencias diferentes · diapositiva 34", 960, 140, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "falso positivo (FP)", "legítimo → «spam»: se oculta un mensaje necesario", "a"),
              (1360, "ambar", "falso negativo (FN)", "spam → pasa: ruido y trabajo en la bandeja", "b")], y=450, w=680, h=340, tam_t=46, tam_d=36),
    P("contar errores sin su tipo = suponer que cuestan lo mismo", 960, 780, "violeta", 44, en="c"), diapo(34, en="a"), sec="3. Dos errores")

# ---- 4. Analogía
v.b("Piensa en el conserje de un edificio. [[a]]Si no deja entrar a un vecino que vive ahí, es grave: el vecino se queda afuera. Ese es el falso positivo. [[b]]Si deja entrar a un vendedor que nadie llamó, es una molestia, pero se resuelve. Ese es el falso negativo. [[c]]Un buen conserje no se mide solo por cuántas veces acierta, sino por cuánto cuestan sus errores.",
    CL(), T("el conserje del edificio", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "deja afuera a un vecino", "grave · falso positivo", "a"), (1360, "ambar", "deja pasar a un vendedor", "molestia · falso negativo", "b")],
             y=450, w=660, h=300, tam_t=44, tam_d=40),
    P("no solo cuántos aciertos: cuánto cuestan los errores", 960, 760, "violeta", 46, en="c"), ilustrativo(1000, texto="analogía"),
    sec="4. Analogía: el conserje")

# ---- 5. La exactitud no basta
v.b("Volvamos al filtro del inicio, el ejemplo de la diapositiva treinta y cinco. [[a]]Cinco spam reales, noventa y cinco legítimos, y una regla que siempre dice «no spam». [[b]]La exactitud, en inglés accuracy, es 95 por ciento: describe los aciertos. [[c]]Pero no demuestra que resolvamos la tarea de detectar spam. En la clase se dijo: «de 100, 95 le achuntó», pero esos cinco nunca fueron marcados.",
    CL(), T("la exactitud no basta · diapositiva 35", 960, 150, tam=50, c="tinta", ent="subir"),
    m2(640, 530, [[95, 0], [5, 0]], en="a", cel=190, tam=60),
    T("exactitud = 95 / 100 = 95 %", 1400, 420, tam=44, c="tinta", w="sb", en="b"),
    T("spam detectado = 0 / 5", 1400, 540, tam=44, c="rojo", w="sb", en="c"),
    T("«de 100, 95 le achuntó»", 1400, 680, tam=40, c="tinta", w="sb", en="c", t=1.2), T("grabación de la clase", 1400, 750, tam=26, c="gris", w="sb", en="c", t=1.5),
    diapo(35, en="a"), sec="5. La exactitud no basta")

# ---- 6. Alerta del material
v.b("Sobre este tema de las clases desbalanceadas, en la grabación hay un pequeño error de cuenta. [[a]]Se dijo: «el evento de interés son 10 de los {1000|mil} datos; entonces yo estoy interesado en 10 observaciones más que los {9990|nueve mil novecientos noventa} restantes». [[b]]Si son mil datos y diez son del evento, los restantes son 990, no {9990|nueve mil novecientos noventa}. [[c]]La idea sí es correcta: con tan pocos casos del evento, la exactitud engaña, y aprender de esos diez es difícil.",
    CL(), alerta(170), cita("el evento de interés son 10 de los 1000 datos… más que los 9.990 restantes", 330, en="a", tam=42),
    etiqueta("grabación de la clase", 420, en="a", t=0.4),
    T("1000 − 10 =", 1000, 580, tam=56, c="tinta", w="sb", anc="rm", en="b"), N(990, 1030, 580, "{:.0f}", 76, "rojo", anc="lm", en="b", t=0.6),
    P("la idea es correcta: con pocos casos del evento, la exactitud engaña", 960, 780, "teal", 42, en="c"), sec="6. Una cuenta")

# ---- 7. La métrica de costo
v.b("¿Cómo se expresa el costo? La diapositiva treinta y seis propone una métrica. [[a]]Supongamos un costo de cinco unidades por cada falso positivo, y de una unidad por cada falso negativo. Los aciertos tienen costo cero. [[b]]El costo promedio es: cinco por FP más FN, dividido por el número de correos. [[c]]Y se multiplica por mil, para comparar en una escala común: costo por cada mil correos.",
    CL(), T("la métrica puede expresar el costo · diapositiva 36", 960, 150, tam=48, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "bloquear un legítimo", "5 unidades por caso (FP)", "a"), (1360, "ambar", "dejar pasar spam", "1 unidad por caso (FN)", "a")],
             y=400, w=620, h=240, tam_t=44, tam_d=40),
    caja_formula("C = (5 · FP + FN) / n", y=640, c="violeta", tam=56, w=900, en="b"),
    T("× 1000 → costo por cada 1000 correos", 960, 790, tam=42, c="tinta", w="sb", en="c"), diapo(36, en="a"), sec="7. La métrica de costo")
v.b("Dos aclaraciones importantes. [[a]]La razón cinco a uno es una prioridad elegida antes de comparar; si cambia esa prioridad, puede cambiar la regla preferida. [[b]]Y son costos hipotéticos, para el ejemplo: no son costos medidos en los datos. [[c]]En un proyecto real, esos costos se conversan con quien usa el filtro.",
    CL(), P("5 : 1 = una prioridad fijada ANTES de comparar", 960, 320, "violeta", 48, en="a"),
    P("costos hipotéticos: no medidos en los datos", 960, 480, "gris", 46, en="b"),
    P("en la práctica, se acuerdan con quien usa el filtro", 960, 640, "teal", 46, en="c"), diapo(36, en="a"))

# ---- 8. Ejemplo resuelto: A y B
v.b("Ahora, el ejemplo de la diapositiva treinta y siete: mil correos, cien de spam. [[a]]El filtro A: 890 verdaderos negativos, 10 falsos positivos, 40 falsos negativos y 60 verdaderos positivos. [[b]]El filtro B: 860, 40, 10 y 90. [[c]]Los dos aciertan 950 de mil: 95 por ciento. Con la exactitud, empatan.",
    CL(), T("mismos aciertos, distinto valor · diapositiva 37", 960, 140, tam=48, c="tinta", ent="subir"),
    m2(560, 520, [[890, 10], [40, 60]], en="a", tit="filtro A", c="azul", cel=180, tam=52),
    m2(1380, 520, [[860, 40], [10, 90]], en="b", tit="filtro B", c="violeta", cel=180, tam=52),
    P("A: (890 + 60) / 1000 = 95 %   ·   B: (860 + 90) / 1000 = 95 %", 960, 860, "gris", 40, en="c"), diapo(37, en="a"), sec="8. Mismos aciertos")
v.b("Pero con los costos fijados, no empatan. [[a]]Costo de A: cinco por diez falsos positivos, cincuenta, más cuarenta falsos negativos: 90. [[b]]Costo de B: cinco por cuarenta, doscientos, más diez: 210. [[c]]Preferimos A, porque bloquea menos correos legítimos. Mismos aciertos, decisiones de distinto valor.",
    CL(), T("el costo los separa", 960, 150, tam=54, c="tinta", ent="subir"),
    T("A: 5 · 10 + 40 =", 900, 330, tam=52, c="tinta", w="sb", anc="rm", en="a"), N(90, 930, 330, "{:.0f}", 76, "azul", anc="lm", en="a", t=0.8),
    T("B: 5 · 40 + 10 =", 900, 480, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(210, 930, 480, "{:.0f}", 76, "violeta", anc="lm", en="b", t=0.8),
    P("preferimos A: bloquea menos correos legítimos", 960, 680, "verde", 48, en="c"), diapo(37, en="a"))

# ---- 9. F1
v.b("¿Y el F1, que vieron la clase pasada? Diapositiva treinta y ocho. [[a]]La precisión es verdaderos positivos sobre todo lo marcado como spam; el recall, verdaderos positivos sobre todo el spam real. El F1 los combina. [[b]]Para A: dos por sesenta, sobre dos por sesenta más diez más cuarenta: 120 sobre 170, 0.706. Para B: 180 sobre 230, 0.783. [[c]]El F1 prefiere B. ¿Por qué? Porque no incorpora la razón de costos cinco a uno: no sustituye nuestra función de costo.",
    CL(), T("F1 y costo pueden preferir modelos distintos · diapositiva 38", 960, 140, tam=46, c="tinta", ent="subir"),
    T("precisión = VP / (VP + FP)    ·    recall = VP / (VP + FN)", 960, 270, tam=40, c="tinta", w="sb", en="a"),
    T("F1 = 2·VP / (2·VP + FP + FN)", 960, 350, tam=44, c="violeta", w="sb", en="a", t=1.0),
    T("A: 120 / 170 =", 760, 480, tam=46, c="tinta", w="sb", anc="rm", en="b"), N(0.706, 790, 480, "{:.3f}", 60, "azul", anc="lm", en="b", t=0.6),
    T("B: 180 / 230 =", 1460, 480, tam=46, c="tinta", w="sb", anc="rm", en="b", t=1.4), N(0.783, 1490, 480, "{:.3f}", 60, "violeta", anc="lm", en="b", t=2.0),
    fila(640, [("", 560), ("F1 (mayor es mejor)", 960), ("costo (menor es mejor)", 1400)], tams=[34] * 3, colores=["gris"] * 3, en="c"),
    fila(720, [("ganador", 560), ("B", 960), ("A", 1400)], tams=[38, 54, 54], colores=["tinta", "violeta", "azul"], en="c", t=0.5,
         fondo="linea", x0=400, x1=1600, h=90),
    P("el F1 no incorpora la razón 5 : 1", 960, 870, "ambar", 46, en="c", t=1.5), diapo(38, en="a"), sec="9. El F1")

# ---- 10. El error típico
v.b("El error típico: [[a]]elegir el filtro por la exactitud o por el F1 cuando los errores tienen costos distintos. [[b]]Las dos métricas tratan igual, o casi igual, un correo legítimo bloqueado y un spam que pasa. [[c]]Si ya acordamos que bloquear un legítimo cuesta cinco veces más, la métrica de selección tiene que reflejarlo.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("elegir por exactitud o F1 con costos distintos", 410, 260, tam=46, c="rojo", anc="lm", en="a"),
    T("tratan casi igual un FP y un FN", 960, 440, tam=44, c="tinta", w="sb", en="b"),
    P("la métrica de selección debe reflejar el 5 : 1", 960, 620, "violeta", 48, en="c"), sec="10. El error típico")

# ---- 11. Piensa un segundo
v.b("Piensa un segundo. [[a]]Un filtro C, en los mismos mil correos, tiene 895 verdaderos negativos, 5 falsos positivos, 50 falsos negativos y 50 verdaderos positivos. [[b]]¿Cuál es su exactitud? ¿Y su costo? ¿Lo preferirías al filtro A?",
    CL(), m2(700, 520, [[895, 5], [50, 50]], en="a", tit="filtro C", c="teal", cel=180, tam=52),
    P("exactitud = ?", 1420, 420, "gris", 48, en="b"), P("costo = ?", 1420, 560, "violeta", 48, en="b", t=0.4),
    P("¿mejor que A (costo 90)?", 1420, 700, "azul", 44, en="b", t=0.8), sec="11. Piensa un segundo", pausa=0.6)
v.b("Exactitud: 895 más 50, sobre mil: [[a]]94.5 por ciento, menos que el 95 de A. [[b]]Costo: cinco por cinco, veinticinco, más cincuenta: [[c]]75, menos que el 90 de A. Con la razón cinco a uno, preferimos C, aunque acierte menos.",
    CL(), T("exactitud = (895 + 50) / 1000 =", 1060, 330, tam=48, c="tinta", w="sb", anc="rm"), N(94.5, 1090, 330, "{:.1f}", 64, "gris", anc="lm", suf=" %", en="a"),
    T("costo = 5 · 5 + 50 =", 1060, 480, tam=48, c="tinta", w="sb", anc="rm", en="b"), N(75, 1090, 480, "{:.0f}", 64, "violeta", anc="lm", en="c"),
    P("preferimos C: acierta menos, pero cuesta menos", 960, 680, "teal", 46, en="c", t=1.0), ilustrativo(1000))

# ---- 12. Para llevar
v.b("Para llevarte: [[a]]en clasificación, falso positivo y falso negativo tienen consecuencias distintas. [[b]]La exactitud y el F1 no saben de costos; una métrica de costo, como cinco por FP más FN, sí. [[c]]Y la razón de costos es una prioridad que se fija antes de comparar.",
    CL(), P("FP y FN: consecuencias distintas", 960, 300, "rojo", 48, en="a"),
    P("exactitud y F1 no saben de costos; (5·FP + FN)/n sí", 960, 460, "violeta", 44, en="b"),
    P("la razón de costos se fija antes", 960, 620, "teal", 48, en="c"), sec="12. Para llevar")
v.b("Siguiente parada: [[a]]las probabilidades. Cómo un umbral las convierte en acciones, y cómo se mide si una probabilidad es buena: log loss, Brier y calibración.",
    CL(), siguiente(10, "Umbral, log loss, Brier y calibración"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
