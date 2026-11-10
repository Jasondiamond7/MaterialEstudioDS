import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, caja_formula, diapo, etiqueta, cita, fila, m2, flujo)

OUT = out("guion_c38_12_umbral.json")
v = V("Umbral por costo, prueba final y repaso")
CO = D["s_costos"]
CT = {f"{c['t']:.2f}": c for c in CO}
SF = D["s_final"]
SP = D["s_part"]
c05, c06 = CT["0.50"], CT["0.60"]

v.portada("Umbral por costo y prueba final", "Cerrar la regla, evaluarla una vez y repasar la clase",
          "Última parada: elegir el umbral por costo, evaluar la regla cerrada en la prueba, y repasar toda la clase.")

# ---- 1. Hook
v.b("El mismo bosque, con las mismas probabilidades. [[a]]Con el umbral de 0.5, cuesta 149.7 unidades por cada mil correos. [[b]]Con el umbral de 0.6, cuesta 128.8. [[c]]No cambiamos el modelo: cambiamos solo la decisión, y el costo bajó. Elegir el umbral es parte de elegir la regla.",
    CL(), T("mismo modelo, otro umbral", 960, 150, tam=56, c="tinta", ent="subir"),
    T("t = 0.50", 560, 320, tam=46, c="gris", w="sb", en="a"), N(c05["c1000"], 560, 420, "{:.1f}", 96, "rojo", en="a", t=0.3),
    T("t = 0.60", 1360, 320, tam=46, c="gris", w="sb", en="b"), N(c06["c1000"], 1360, 420, "{:.1f}", 96, "verde", en="b", t=0.3),
    T("costo por cada 1000 correos (OOF)", 960, 540, tam=36, c="gris", w="sb", en="a", t=0.6),
    P("elegir el umbral es parte de elegir la regla", 960, 720, "violeta", 48, en="c"), real(1050, en="a"), sec="1. Mismo modelo, otro costo")

# ---- 2. Analogía
v.b("Piensa en un juez que decide cuánta evidencia exigir antes de condenar. [[a]]Si exige muy poca, condena a más inocentes. [[b]]Si exige demasiada, deja libres a más culpables. [[c]]Dónde poner la vara depende de cuánto pesa cada error. Con el spam, bloquear un correo legítimo pesa cinco veces más que dejar pasar un spam: la vara tiene que estar alta.",
    CL(), T("el juez y la vara", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "vara baja", "más inocentes condenados (FP)", "a"), (1360, "ambar", "vara alta", "más culpables libres (FN)", "b")],
             y=440, w=640, h=300, tam_t=52, tam_d=40),
    P("la vara depende de cuánto pesa cada error", 960, 760, "violeta", 48, en="c"), ilustrativo(1000, texto="analogía"), sec="2. Analogía: el juez")

# ---- 3. La curva de costo
TS = [c["t"] for c in CO if c["t"] >= 0.25]
CS = [c["c1000"] for c in CO if c["t"] >= 0.25]
v.b("La diapositiva cincuenta y tres muestra el costo fuera de pliegue del bosque, para cada umbral de la cuadrícula. [[a]]Los umbrales bajos bloquean demasiado y cuestan mucho: con 0.05, más de dos mil unidades por cada mil correos; aquí se ven desde 0.25. [[b]]El costo baja hasta un mínimo y vuelve a subir. [[c]]Cerca del mínimo: con 0.55, 132.3; con 0.60, 128.8; y con 0.65, 135.3. Entre los 19 umbrales ensayados, se elige t igual a 0.60.",
    CL(), T("elegimos el umbral con el costo en validación · diapositiva 53", 960, 140, tam=46, c="tinta", ent="subir"),
    G(420, 240, 1300, 840, 0.2, 1.0, 0, 500, [dict(xs=TS, ys=CS, color="azul", grosor=6, marcadores=8, revela=0.0)],
      xticks=[0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 100, 200, 300, 400, 500], yfmt="{:.0f}", xlabel="umbral de bloqueo", ylabel="costo",
      id="gco", tam_txt=26),
    U("gco", **{"series.0.revela": 1.0}, en="b", dur=1.8),
    R(420 + (0.6 - 0.2) / 0.8 * 880, 840 - 128.8 / 500 * 600, 60, 60, "verde", r=30, en="c"),
    fila(400, [("umbral", 1490), ("costo", 1720)], tams=[32, 32], colores=["gris"] * 2, en="c", x0=1380, x1=1840),
    fila(480, [("0.55", 1490), (f"{CT['0.55']['c1000']:.1f}", 1720)], tams=[38, 40], colores=["tinta", "tinta"], en="c", t=0.3, x0=1380, x1=1840),
    fila(560, [("0.60", 1490), (f"{CT['0.60']['c1000']:.1f}", 1720)], tams=[38, 44], colores=["verde", "verde"], en="c", t=1.2, x0=1380, x1=1840,
         fondo="verdeclaro", h=70),
    fila(640, [("0.65", 1490), (f"{CT['0.65']['c1000']:.1f}", 1720)], tams=[38, 40], colores=["tinta", "tinta"], en="c", t=2.0, x0=1380, x1=1840),
    T("t = 0.05: 2286.4 por 1000", 1610, 760, tam=28, c="gris", w="sb", en="a"), real(1050, en="b"), sec="3. La curva de costo")
v.b("Y aquí la grabación lo dijo bien. [[a]]«Si pongo un umbral 0,55, el costo asociado va a estar en 132; en 0,6 el costo va a ser 128; y subiendo a 0,65, el costo empezó a subir». [[b]]Coincide con la tabla recalculada. [[c]]Una advertencia de la diapositiva: el mínimo usado para elegir todavía necesita la prueba, porque también es optimista.",
    CL(), T("lo que se dijo en clase, verificado", 960, 150, tam=52, c="tinta", ent="subir"),
    cita("si pongo un umbral 0,55 el costo asociado va a estar en 132… en 0,6 el costo va a ser 128", 320, en="a", tam=40),
    etiqueta("grabación de la clase", 410, en="a", t=0.4),
    K(700, 560, "check", 70, en="b"), T("132.3 · 128.8 · 135.3: coincide", 780, 560, tam=46, c="verde", w="sb", anc="lm", en="b"),
    P("el mínimo usado para elegir también es optimista", 960, 740, "ambar", 46, en="c"), diapo(53, en="c"))

# ---- 4. El intercambio
v.b("¿Qué cambió entre 0.5 y 0.6? La diapositiva cincuenta y cuatro lo detalla. [[a]]Con 0.5: 74 falsos positivos y 181 falsos negativos; aciertos, 93.1 por ciento. [[b]]Con 0.6: 44 falsos positivos y 254 falsos negativos; aciertos, 91.9. [[c]]Son 30 falsos positivos menos y 73 falsos negativos más. Ahorro: cinco por treinta, menos 73, igual a 77 unidades. Bajan los aciertos, pero también baja el costo: evitamos los errores que pesan más.",
    CL(), T("el mismo modelo, otra combinación de errores · diapositiva 54", 960, 140, tam=46, c="tinta", ent="subir"),
    fila(280, [("umbral", 400), ("FP", 720), ("FN", 980), ("aciertos", 1260), ("costo / 1000", 1580)], tams=[34] * 5, colores=["gris"] * 5),
    fila(380, [("0.50", 400), (str(c05["FP"]), 720), (str(c05["FN"]), 980), (f"{100 * c05['acc']:.1f} %", 1260), (f"{c05['c1000']:.1f}", 1580)],
         tams=[42, 50, 50, 46, 50], colores=["tinta", "rojo", "ambar", "tinta", "rojo"], en="a", fondo="linea", x0=240, x1=1740, h=96),
    fila(490, [("0.60", 400), (str(c06["FP"]), 720), (str(c06["FN"]), 980), (f"{100 * c06['acc']:.1f} %", 1260), (f"{c06['c1000']:.1f}", 1580)],
         tams=[42, 50, 50, 46, 50], colores=["tinta", "rojo", "ambar", "tinta", "verde"], en="b", fondo="verdeclaro", x0=240, x1=1740, h=96),
    T("30 FP menos · 73 FN más", 960, 640, tam=44, c="tinta", w="sb", en="c"),
    T("ahorro = 5 · 30 − 73 =", 1000, 740, tam=46, c="tinta", w="sb", anc="rm", en="c", t=1.5), N(77, 1030, 740, "{:.0f}", 64, "verde", anc="lm", suf=" unidades", en="c", t=2.1),
    T("77 / 3680 · 1000 = 20.9 = 149.7 − 128.8", 960, 850, tam=34, c="gris", w="sb", en="c", t=3.0),
    real(1050, en="a"), sec="4. El intercambio")

# ---- 5. El umbral teórico
v.b("Los costos también sugieren un umbral teórico, en la diapositiva cincuenta y cinco. [[a]]Supongamos que p es la probabilidad correcta de que el correo sea spam. Si lo bloqueo, me equivoco cuando es legítimo: costo esperado, cinco por uno menos p. [[b]]Si lo dejo pasar, me equivoco cuando es spam: costo esperado, p. [[c]]Conviene bloquear cuando cinco por uno menos p es menor que p. Despejando: p mayor que cinco sextos, 0.833.",
    CL(), T("el umbral teórico · diapositiva 55", 960, 140, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "rojo", "bloquear", "costo esperado = 5 · (1 − p)", "a"), (1360, "ambar", "dejar pasar", "costo esperado = p", "b")],
             y=380, w=620, h=240, tam_t=46, tam_d=40),
    caja_formula("5 (1 − p) < p   →   5 < 6p   →   p > 5/6 ≈ 0.833", y=640, c="violeta", tam=46, w=1400, en="c"),
    diapo(55, en="a"), sec="5. El umbral teórico")
v.b("Probemos con dos correos. [[a]]Con p igual a 0.9: bloquear cuesta cinco por 0.1, medio punto; dejar pasar cuesta 0.9. Conviene bloquear. [[b]]Con p igual a 0.7: bloquear cuesta cinco por 0.3, 1.5; dejar pasar, 0.7. Conviene dejar pasar. [[c]]Pero el umbral elegido con los datos fue 0.60, no 0.833. ¿Por qué difieren? Porque la fórmula supone probabilidades correctas, y las del bosque no lo son: en la parada anterior vimos que, cerca de 0.67, subestima. Además, la muestra y la cuadrícula son finitas.",
    CL(), T("dos correos, a mano", 960, 140, tam=54, c="tinta", ent="subir"),
    T("p = 0.9:  bloquear 5 · 0.1 = 0.5   <   pasar 0.9", 960, 290, tam=44, c="tinta", w="sb", en="a"), P("bloquear", 1640, 290, "rojo", 38, en="a", t=1.5),
    T("p = 0.7:  bloquear 5 · 0.3 = 1.5   >   pasar 0.7", 960, 410, tam=44, c="tinta", w="sb", en="b"), P("dejar pasar", 1660, 410, "ambar", 36, en="b", t=1.5),
    T("teórico 0.833   ·   elegido con datos 0.60", 960, 570, tam=46, c="violeta", w="sb", en="c"),
    T("probabilidades imperfectas (subestima cerca de 0.67) · muestra finita · cuadrícula finita", 960, 680, tam=34, c="gris", w="sb", en="c", t=2.5),
    P("no se reemplaza el umbral elegido después de mirar la prueba", 960, 820, "ambar", 42, en="c", t=4.5), ilustrativo(1000))

# ---- 6. La regla cerrada y la prueba
v.b("Con eso, la regla completa queda cerrada antes de la prueba: diapositiva cincuenta y seis. [[a]]Modelo: el bosque de profundidad ocho. Umbral: 0.60. [[b]]Se reajusta con los 3680 correos de entrenamiento, y se evalúa en los 921 reservados. [[c]]La referencia de decisión deja pasar todos los correos: no comete falsos positivos, pero no detecta ningún spam. La prueba evalúa el procedimiento completo: preprocesamiento, modelo, umbral y costos declarados.",
    CL(), T("la regla completa queda cerrada antes de la prueba · diapositiva 56", 960, 140, tam=44, c="tinta", ent="subir"),
    flujo([("modelo\nbosque, prof. 8", "verde", "a"), ("umbral\nt = 0.60", "ambar", "a"), ("reajustar\ncon 3680", "azul", "b"),
           ("evaluar\nen 921", "violeta", "b")], y=400, w=360, h=220, x0=300, paso=440, tam=36),
    P("referencia: dejar pasar todo (0 FP, 0 spam detectado)", 960, 640, "gris", 44, en="c"),
    T("se evalúa el procedimiento completo", 960, 760, tam=40, c="tinta", w="sb", en="c", t=1.5), diapo(56, en="a"), sec="6. La regla cerrada")
v.b("Y este es el resultado en la prueba, recalculado con el script de Python, como en la diapositiva cincuenta y siete. [[a]]553 verdaderos negativos, solo 5 falsos positivos, 53 falsos negativos y 310 verdaderos positivos. [[b]]Aciertos, 93.7 por ciento; precisión, 98.4; recall, 85.4. Log loss 0.194, y Brier 0.0502. [[c]]El costo: cinco por cinco, más 53, son 78; sobre 921 y por mil, 84.7 por cada mil correos. La referencia de dejar pasar todo cuesta 363 sobre 921, por mil: 394.1.",
    CL(), T("qué obtiene la regla en datos reservados · diapositiva 57", 960, 140, tam=46, c="tinta", ent="subir"),
    m2(560, 520, [[SF["VN"], SF["FP"]], [SF["FN"], SF["VP"]]], en="a", tit="prueba: bosque, t = 0.60", c="verde", cel=180, tam=54),
    T(f"aciertos {100 * SF['acc']:.1f} %  ·  precisión {100 * SF['prec']:.1f} %  ·  recall {100 * SF['rec']:.1f} %", 1360, 330, tam=32, c="tinta", w="sb", en="b"),
    T(f"log loss {SF['log_loss']:.3f}  ·  Brier {SF['brier']:.4f}", 1360, 400, tam=32, c="tinta", w="sb", en="b", t=1.5),
    T("costo = (5 · 5 + 53) / 921 · 1000 =", 1460, 520, tam=34, c="tinta", w="sb", anc="rm", en="c"), N(SF["c1000"], 1480, 520, "{:.1f}", 54, "verde", anc="lm", en="c", t=1.2),
    T("siempre no spam = 363 / 921 · 1000 =", 1460, 620, tam=34, c="tinta", w="sb", anc="rm", en="c", t=2.0), N(SF["ref_c1000"], 1480, 620, "{:.1f}", 54, "rojo", anc="lm", en="c", t=2.8),
    T("versión en R (diapositiva 57): mismos 93.7 % de aciertos, 7 FP y 51 FN, costo 93.4", 960, 930, tam=30, c="gris", w="sb", en="c", t=3.5),
    real(1050, en="a"), sec="7. El resultado en prueba")
# ---- 7. Cambiar la receta después de la prueba
v.b("Y ahora, la regla que cierra la clase, en la diapositiva cincuenta y ocho. [[a]]Supongamos que, mirando la prueba, se nos ocurre otra variable, otro modelo u otro umbral. [[b]]La prueba anterior ayuda a formular esa hipótesis, pero ya no queda intacta para confirmarla. [[c]]El rediseño exige una nueva selección y una nueva evaluación independiente. La regla es la misma para la resistencia y para el spam: el desempeño usado para elegir no reemplaza una evaluación independiente.",
    CL(), T("cambiar la receta después de la prueba cambia la evidencia · diapositiva 58", 960, 140, tam=42, c="tinta", ent="subir"),
    flujo([("evaluamos la\nregla elegida", "verde", "a"), ("diagnóstico:\notra variable,\nmodelo o umbral", "ambar", "a"),
           ("nueva selección\ny nueva evaluación", "violeta", "c")], y=420, w=440, h=250, x0=400, paso=560, tam=34),
    T("la prueba anterior sirve para la hipótesis, no para confirmarla", 960, 650, tam=40, c="tinta", w="sb", en="b"),
    P("lo que se usó para elegir no reemplaza una evaluación independiente", 960, 800, "rojo", 42, en="c", t=1.5), diapo(58, en="a"),
    sec="8. Después de la prueba")

# ---- 8. El error típico
v.b("El error típico de esta parte: [[a]]elegir el umbral mirando el costo en la prueba. [[b]]Si en la prueba el mejor umbral resultara ser otro, la tentación es cambiarlo. [[c]]Pero entonces el 84.7 ya no estima nada: la prueba se habría usado para elegir. El umbral se elige con validación, y la prueba se mira una sola vez.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("elegir el umbral mirando la prueba", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("«en la prueba convenía otro umbral… lo cambio»", 960, 440, tam=42, c="gris", w="sb", en="b"),
    P("umbral con validación · prueba una sola vez", 960, 620, "violeta", 48, en="c"), sec="9. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Si los costos fueran otros: tres unidades por cada falso positivo y una por cada falso negativo. [[b]]¿Cuál sería el umbral teórico?",
    CL(), T("FP = 3   ·   FN = 1", 960, 300, tam=72, c="tinta", w="b", en="a"),
    P("umbral teórico = ?", 960, 500, "violeta", 56, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Bloquear cuesta tres por uno menos p; dejar pasar cuesta p. [[a]]Conviene bloquear si tres menos tres p es menor que p, es decir, si p es mayor que tres cuartos: [[b]]0.75. [[c]]En general, con costos c para el falso positivo y uno para el falso negativo, el umbral teórico es c sobre c más uno. Con cinco, cinco sextos; con tres, tres cuartos.",
    CL(), T("3 (1 − p) < p   →   p > 3/4 =", 1080, 320, tam=50, c="tinta", w="sb", anc="rm", en="a"), N(0.75, 1110, 320, "{:.2f}", 70, "violeta", anc="lm", en="b"),
    P("en general: umbral = c / (c + 1)", 960, 500, "teal", 50, en="c"),
    T("c = 5 → 5/6 ≈ 0.833   ·   c = 3 → 3/4 = 0.75", 960, 630, tam=42, c="gris", w="sb", en="c", t=1.5), ilustrativo(1000))

# ---- 10. Repaso de toda la clase
v.b("Repasemos toda la clase con la diapositiva sesenta. [[a]]Evaluar responde: ¿qué error esperamos? Para eso, se predicen casos fuera del ajuste, con una métrica acorde al uso, y se compara con una referencia. [[b]]Seleccionar responde: ¿qué receta usamos? Se comparan los candidatos en la misma validación, se aplica el criterio fijado, y se reajusta la receta elegida. [[c]]En el hormigón, se eligió por RMSE. En el spam, las probabilidades con la pérdida logarítmica, y el umbral con el costo.",
    CL(), T("¿cómo evaluamos y seleccionamos modelos? · diapositiva 60", 960, 140, tam=46, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "evaluar", "¿qué error esperamos? fuera del ajuste · métrica acorde · referencia", "a"),
              (1360, "violeta", "seleccionar", "¿qué receta? misma validación · criterio fijado · reajustar", "b")], y=420, w=680, h=320, tam_t=50, tam_d=34),
    P("hormigón: RMSE   ·   spam: log loss + umbral por costo", 960, 720, "teal", 44, en="c"),
    flujo([("definir", "azul", "c"), ("comparar", "violeta", "c"), ("elegir", "teal", "c"), ("evaluar", "ambar", "c")], y=880, w=260, h=100, x0=420, paso=360, tam=36),
    diapo(60, en="a"), sec="11. Repaso de la clase")
v.b("Y las cuatro recomendaciones de la diapositiva sesenta y uno. [[a]]Uno: definir el criterio. Fijar la métrica principal y la referencia, y en clasificación distinguir probabilidades de decisiones. Dos: validar sin fuga. Respetar los grupos o el tiempo, y aprender las transformaciones solo con las filas de ajuste. [[b]]Tres: cerrar la receta. Fijar modelo, transformaciones, hiperparámetros y umbral, y reajustar con el entrenamiento. [[c]]Cuatro: valorar la mejora. Considerar su magnitud y su estabilidad; y ante resultados parecidos, considerar la simplicidad y el costo.",
    CL(), T("recomendaciones para elegir y evaluar bien · diapositiva 61", 960, 140, tam=46, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "1 · definir el criterio", "métrica y referencia · probabilidades ≠ decisiones", "a"),
              (1360, "rojo", "2 · validar sin fuga", "grupos o tiempo · transformar solo con ajuste", "a")], y=380, w=700, h=250, tam_t=42, tam_d=32),
    tarjetas([(560, "teal", "3 · cerrar la receta", "modelo, transformaciones, hiperparámetros, umbral", "b"),
              (1360, "ambar", "4 · valorar la mejora", "magnitud, estabilidad, simplicidad y costo", "c")], y=680, w=700, h=250, tam_t=42, tam_d=32),
    diapo(61, en="a"))

# ---- 11. Para llevar
v.b("Para llevarte de esta parada: [[a]]el umbral se elige con validación, por el costo; el umbral teórico, cinco sextos, supone probabilidades correctas. [[b]]La regla completa se cierra antes de la prueba, y se evalúa una sola vez contra su referencia: 84.7 contra 394.1 por cada mil correos. [[c]]Y si se rediseña después de mirar la prueba, hace falta una nueva evaluación independiente.",
    CL(), P("umbral por costo en validación · teórico 5/6 si p es correcta", 960, 300, "ambar", 42, en="a"),
    P("regla cerrada, evaluada una vez: 84.7 vs 394.1", 960, 460, "verde", 46, en="b"),
    P("rediseñar tras la prueba → nueva evaluación", 960, 620, "rojo", 46, en="c"), sec="12. Para llevar")
v.b("Con esto termina la clase treinta y ocho. [[a]]Siguiente parada: la clase treinta y nueve, sobre clustering, donde ya no hay una respuesta que predecir.",
    CL(), siguiente(39, "Clase 39: clustering"), sec="13. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
