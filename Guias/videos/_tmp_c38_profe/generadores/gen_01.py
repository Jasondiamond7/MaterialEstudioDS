import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import D, out, tarjetas, siguiente, ilustrativo, real, flujo, diapo, etiqueta, cita

OUT = out("guion_c38_01_mejor.json")
v = V("Introducción: ¿mejor modelo para qué?")
HP = D["h_part"]
EJ = D["h_resid"]["ej"][0]          # id 2: real 61.9, modelo 59.6, media 35.5

v.portada("¿Mejor modelo para qué?", "Evaluar la salida que de verdad se va a usar",
          "Clase treinta y ocho: evaluación de modelos supervisados y selección de modelos. Antes de comparar números, hay que responder una pregunta: mejor, ¿para qué?")

# ---- 1. Hook
v.b("Tienes dos modelos que predicen la resistencia de una mezcla de hormigón. [[a]]El primero comete errores pequeños casi siempre, pero de vez en cuando se equivoca por mucho. [[b]]El segundo nunca comete errores enormes, pero rara vez da justo en el blanco. [[c]]¿Cuál es mejor? La respuesta honesta es: depende de para qué lo vas a usar.",
    CL(), T("¿cuál es mejor?", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "modelo 1", "errores pequeños casi siempre… y a veces uno enorme", "a"),
              (1360, "violeta", "modelo 2", "nunca un error enorme… pero rara vez exacto", "b")], y=470, w=660, h=340, tam_t=52, tam_d=38),
    P("depende de para qué lo vas a usar", 960, 780, "ambar", 52, en="c"), diapo(2, en="a"), sec="1. ¿Cuál es mejor?")
v.b("Y en clasificación pasa lo mismo. [[a]]Dos filtros de spam aciertan el mismo porcentaje de correos. [[b]]Uno bloquea más correos legítimos; el otro deja pasar más spam. [[c]]El profesor lo dijo así en la clase: este concepto de mejor modelo es un poco etéreo. Hay que definirlo.",
    CL(), T("dos filtros de spam", 960, 150, tam=56, c="tinta", ent="subir"),
    P("mismo porcentaje de aciertos", 960, 300, "teal", 48, en="a"),
    tarjetas([(560, "rojo", "filtro A", "bloquea más correos legítimos", "b"), (1360, "ambar", "filtro B", "deja pasar más spam", "b")],
             y=530, w=620, h=280, tam_t=50, tam_d=38),
    cita("este concepto de mejor modelo es un poco etéreo", 830, en="c", tam=46), etiqueta("cita de la grabación de la clase", 1000, en="c"),
    diapo(2, en="a"))

# ---- 2. Analogía
v.b("Piensa en alguien que te pregunta cuál es el mejor auto. [[a]]Si lo quieres para la ciudad, gana uno pequeño que gaste poco. [[b]]Si lo quieres para un camino de montaña, gana una camioneta alta. [[c]]Si es para una carrera, gana el más rápido. El auto no cambió: cambió la pregunta. Con los modelos ocurre exactamente lo mismo.",
    CL(), T("¿cuál es el mejor auto?", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(400, "teal", "ciudad", "gana el que gasta poco", "a"), (960, "ambar", "montaña", "gana el más alto", "b"),
              (1520, "rojo", "carrera", "gana el más rápido", "c")], y=480, w=480, h=320, tam_t=52, tam_d=38),
    P("el auto no cambió: cambió la pregunta", 960, 800, "violeta", 50, en="c", t=1.2), ilustrativo(1000, texto="analogía"),
    sec="2. Analogía: el mejor auto")

# ---- 3. Tres salidas distintas
v.b("Primera idea de la clase: la salida de un modelo y su uso son cosas distintas. [[a]]A veces la salida es una cantidad: una resistencia estimada de 35 {MPa|megapascales}. [[b]]Otras veces es una probabilidad: probabilidad de spam igual a 0.70. [[c]]Y otras veces es una acción: bloquear el correo, o dejarlo pasar.",
    CL(), T("la salida y su uso son cosas distintas · diapositiva 3", 960, 150, tam=48, c="tinta", ent="subir"),
    flujo([("cantidad\nŷ = 35 MPa", "azul", "a"), ("probabilidad\np = 0.70", "violeta", "b"), ("acción\nbloquear / dejar pasar", "ambar", "c")],
          y=470, w=460, h=230, x0=400, paso=560, tam=40),
    diapo(3, en="a"), sec="3. Cantidad, probabilidad y acción")
v.b("Y cada una se evalúa distinto. [[a]]La cantidad se compara con el valor real: si la mezcla resistió 40, me equivoqué por 5. [[b]]La probabilidad expresa incertidumbre: hay que ver si ese 0.70 se cumple en la práctica. [[c]]Y la acción agrega una regla de decisión, con sus propias consecuencias. Por eso la diapositiva insiste: evaluar bien exige evaluar la salida que se va a utilizar.",
    T("se compara con el valor real", 400, 680, tam=34, c="azul", w="sb", mw=440, en="a"),
    T("¿se cumple ese 0.70?", 960, 680, tam=34, c="violeta", w="sb", mw=440, en="b"),
    T("¿qué cuesta cada error?", 1520, 680, tam=34, c="ambar", w="sb", mw=440, en="c"),
    P("evaluar la salida que se va a utilizar", 960, 850, "teal", 50, en="c", t=1.6))

# ---- 4. Tres preguntas antes de elegir
v.b("Antes de elegir, la diapositiva cuatro propone tres preguntas. [[a]]Dónde se usará el modelo: qué casos futuros nos interesan. [[b]]Qué importa: la consecuencia de cada tipo de error. [[c]]Y cómo validar: con datos que imiten ese uso. La métrica representa qué errores importan; la partición representa los casos nuevos que queremos predecir.",
    CL(), T("antes de elegir: ¿mejor para qué y para quién?", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(400, "azul", "¿dónde?", "los casos futuros que nos interesan", "a"), (960, "rojo", "¿qué importa?", "la consecuencia de cada error", "b"),
              (1520, "teal", "¿cómo validar?", "con datos que imitan ese uso", "c")], y=470, w=500, h=330, tam_t=48, tam_d=36),
    T("métrica = qué errores importan   ·   partición = qué casos nuevos", 960, 800, tam=40, c="tinta", w="sb", en="c", t=1.5),
    diapo(4, en="a"), sec="4. Tres preguntas")

# ---- 5. El caso del hormigón
v.b("Ahora, el primer caso de la clase. [[a]]Queremos predecir la resistencia del hormigón, en megapascales, una unidad de presión. [[b]]Tenemos 1030 mediciones, con ocho predictores: siete ingredientes, en kilos por metro cúbico, y la edad en días. [[c]]Los ingredientes son cemento, escoria, ceniza, agua, superplastificante, y agregados grueso y fino.",
    CL(), T("caso 1 · la resistencia del hormigón", 960, 150, tam=52, c="tinta", ent="subir"),
    P("respuesta: resistencia en MPa (megapascales)", 960, 290, "azul", 46, en="a"),
    N(HP["filas"], 520, 470, "{:.0f}", 110, "azul", en="b"), T("mediciones", 520, 560, tam=38, c="gris", w="sb", en="b", t=0.3),
    N(8, 1400, 470, "{:.0f}", 110, "violeta", en="b", t=0.6), T("predictores: 7 ingredientes + edad", 1400, 560, tam=38, c="gris", w="sb", en="b", t=0.8),
    T("cemento · escoria · ceniza · agua · superplastificante · agregado grueso · agregado fino", 960, 720, tam=34, c="tinta", w="sb", mw=1700, en="c"),
    real(1050, en="b"), sec="5. El caso del hormigón")
v.b("La pregunta que guía todo el caso está en la diapositiva cinco: [[a]]¿cuánto nos equivocaremos en una composición que quedó fuera del ajuste? [[b]]Fíjate en la palabra composición. No basta con predecir una fila cualquiera: queremos predecir mezclas que el modelo no conoce. [[c]]Esa idea va a decidir cómo partimos los datos.",
    CL(), T("la pregunta del caso · diapositiva 5", 960, 150, tam=50, c="tinta", ent="subir"),
    P("¿cuánto nos equivocaremos en una composición que quedó fuera del ajuste?", 960, 360, "violeta", 44, en="a"),
    R(960, 360, 1640, 120, "ambar", r=30, en="b"),
    T("no una fila cualquiera: una mezcla que el modelo no conoce", 960, 560, tam=42, c="tinta", w="sb", en="b", t=0.5),
    P("esa idea decide cómo partimos los datos", 960, 740, "teal", 48, en="c"), diapo(5, en="a"))

# ---- 6. Ejemplo resuelto con datos reales
v.b("Veamos una predicción real, de la clase. [[a]]Una mezcla de prueba resistió 61.9 megapascales. [[b]]El modelo que vamos a terminar eligiendo predijo 59.6: se equivocó por 2.3. [[c]]Un modelo que siempre dice el promedio del entrenamiento, 35.5, se equivoca por 26.4. Esa comparación con una referencia simple es parte de evaluar.",
    CL(), T("una predicción real de la clase", 960, 150, tam=52, c="tinta", ent="subir"),
    T("real", 560, 330, tam=40, c="gris", w="sb"), N(EJ["real"], 560, 430, "{:.1f}", 96, "tinta", en="a", suf=" MPa"),
    T("modelo elegido", 1000, 330, tam=40, c="gris", w="sb", en="b"), N(EJ["pred"], 1000, 430, "{:.1f}", 96, "azul", en="b"),
    T("siempre la media", 1440, 330, tam=40, c="gris", w="sb", en="c"), N(EJ["base"], 1440, 430, "{:.1f}", 96, "ambar", en="c"),
    P("error 61.9 − 59.6 = 2.3", 1000, 620, "azul", 42, en="b", t=1.0),
    P("error 61.9 − 35.5 = 26.4", 1440, 740, "ambar", 42, en="c", t=1.0),
    real(1050, en="a"), sec="6. Una predicción real")
EJ4 = D["h_resid"]["ej"][3]         # id 20: real 42.6, modelo 54.0
v.b("Pero no siempre le va tan bien. [[a]]Otra mezcla de prueba resistió 42.6, y el mismo modelo predijo 54.0: esta vez se pasó por 11.4. [[b]]Y en la prueba hay {224|doscientas veinticuatro} mezclas, cada una con su propio error. [[c]]¿Cómo resumimos {224|doscientos veinticuatro} errores en una cifra que sirva para comparar modelos? Esa es la pregunta de la próxima parada.",
    CL(), T("otra predicción del mismo modelo", 960, 150, tam=52, c="tinta", ent="subir"),
    T("real", 700, 330, tam=40, c="gris", w="sb"), N(EJ4["real"], 700, 430, "{:.1f}", 96, "tinta", suf=" MPa"),
    T("modelo elegido", 1220, 330, tam=40, c="gris", w="sb", en="a"), N(EJ4["pred"], 1220, 430, "{:.1f}", 96, "rojo", en="a"),
    P("error 42.6 − 54.0 = −11.4", 960, 600, "rojo", 44, en="a", t=1.2),
    T("224 mezclas de prueba → 224 errores", 960, 740, tam=42, c="tinta", w="sb", en="b"),
    P("¿cómo resumirlos en una cifra?", 960, 870, "violeta", 48, en="c"), real(1050, en="a"))

# ---- 7. El plan de la clase
v.b("Con eso, el plan de la clase tiene cuatro pasos, que se repiten en los dos casos. [[a]]Definir la tarea y la métrica. [[b]]Comparar los candidatos con validación cruzada interna. [[c]]Elegir y reajustar. Y recién al final, evaluar una sola vez en la prueba reservada.",
    CL(), T("el protocolo de la clase · diapositivas 23 y 60", 960, 150, tam=50, c="tinta", ent="subir"),
    flujo([("definir\ntarea y métrica", "azul", "a"), ("comparar\nCV interna", "violeta", "b"), ("elegir\ny reajustar", "teal", "c"),
           ("evaluar\nprueba final", "ambar", "c")], y=470, w=360, h=220, x0=300, paso=440, tam=38),
    T("hormigón: regresión   ·   spam: clasificación", 960, 720, tam=40, c="gris", w="sb", en="c", t=1.2), sec="7. El plan de la clase")

# ---- 8. El error típico
v.b("El error típico al empezar: [[a]]mirar una sola cifra, por ejemplo el porcentaje de aciertos, y declarar un ganador. [[b]]Recuerda los dos filtros de spam: tenían la misma cifra, y aun así uno bloquea correos que necesitabas. [[c]]Una sola cifra no responde todas las preguntas.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("una sola cifra → un ganador", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    T("misma cifra, consecuencias distintas", 960, 450, tam=44, c="tinta", w="sb", en="b"),
    P("una sola cifra no responde todas las preguntas", 960, 640, "ambar", 50, en="c"), diapo(3, en="c"), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]¿Qué tipo de salida es cada una? Primero: «resistencia estimada, 42 megapascales». [[b]]Segundo: «probabilidad de spam, 0.85». [[c]]Tercero: «mover el correo a la carpeta de spam».",
    CL(), T("¿qué tipo de salida es cada una?", 960, 160, tam=54, c="tinta", ent="subir"),
    P("1 · resistencia estimada: 42 MPa", 960, 340, "azul", 46, en="a"), P("2 · probabilidad de spam: 0.85", 960, 480, "violeta", 46, en="b"),
    P("3 · mover el correo a la carpeta de spam", 960, 620, "ambar", 46, en="c"), ilustrativo(1000), sec="9. Piensa un segundo", pausa=0.6)
v.b("La primera es una cantidad: se compara con el valor real. [[a]]La segunda es una probabilidad: hay que revisar si se cumple. [[b]]Y la tercera es una acción: sale de aplicar una regla a la probabilidad, por ejemplo bloquear si pasa de cierto umbral. [[c]]Tres salidas, tres formas de evaluar.",
    K(1560, 340, "check", 60), T("cantidad", 1620, 340, tam=40, c="azul", anc="lm"),
    K(1560, 480, "check", 60, en="a"), T("probabilidad", 1620, 480, tam=40, c="violeta", anc="lm", en="a"),
    K(1560, 620, "check", 60, en="b"), T("acción", 1620, 620, tam=40, c="ambar", anc="lm", en="b"),
    P("tres salidas, tres formas de evaluar", 960, 800, "teal", 48, en="c"))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]«mejor modelo» necesita una definición: qué error importa, para qué uso, y en qué datos nuevos. [[b]]Hay que evaluar la salida que de verdad se va a usar: cantidad, probabilidad o acción. [[c]]Y el orden es siempre el mismo: definir, comparar, elegir, y evaluar al final.",
    CL(), P("mejor = qué error importa · para qué uso · en qué datos nuevos", 960, 300, "violeta", 44, en="a"),
    P("evaluar la salida que se va a usar", 960, 460, "azul", 48, en="b"),
    P("definir → comparar → elegir → evaluar", 960, 620, "teal", 48, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]cómo medir el error de una cantidad. El MAE, el RMSE y el R cuadrado, y por qué pueden elegir ganadores distintos.",
    CL(), siguiente(2, "MAE, RMSE y R²: medir el error de una cantidad"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
