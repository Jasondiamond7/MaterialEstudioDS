import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util40 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, casillas, barras_v, tabla, flujo)

OUT = out("guion_c40_01_preguntas.json")
v = V("Previa, condicional y posterior")

v.portada("Previa, condicional y posterior", "De buscar grupos a aprender una clase con probabilidades",
          "Clase cuarenta, primera parada. Hoy vamos a construir un filtro de spam que entrega una probabilidad. Y antes de cualquier fórmula, vamos a separar tres preguntas que parecen iguales, pero no lo son.")

# ---- 1. Puente con la clase anterior
v.b("En la clase anterior, con los vinos, buscábamos grupos sin ninguna etiqueta: los grupos eran un resultado del análisis. [[a]]Hoy la situación cambia. Tenemos correos que ya vienen marcados como spam o como legítimos. [[b]]Ya no buscamos grupos: aprendemos una clase que conocemos, para poder clasificar el siguiente correo que llegue.",
    CL(), T("ahora conocemos la clase que queremos predecir", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(560, "teal", "vinos: buscar grupos", "sin etiqueta; los grupos son un resultado", "a"),
              (1360, "violeta", "correos: aprender una clase", "cada correo ya viene etiquetado", "b")], y=520, w=640, h=360, tam_t=46, tam_d=38),
    diapo(2, en="a"), sec="1. De los vinos a los correos")
v.b("La notación es sencilla. [[a]]La respuesta es Y. Vale uno si el correo es spam, y cero si es legítimo. [[b]]Las entradas son características del mensaje; por ejemplo, qué palabras aparecen. En la clase las llamamos señales. [[c]]Y la pregunta de hoy es una sola: cómo construir una probabilidad de spam a partir de ejemplos etiquetados.",
    CL(), T("la notación del día", 960, 150, tam=54, c="tinta", ent="subir"),
    P("Y = 1: spam      Y = 0: legítimo", 960, 320, "violeta", 54, en="a"),
    P("entradas = señales del mensaje (por ejemplo, palabras)", 960, 480, "teal", 46, en="b"),
    T("¿cómo construir una probabilidad de spam a partir de ejemplos etiquetados?", 960, 690, tam=44, c="tinta", w="sb", mw=1500, en="c"),
    diapo(2, en="a"))

# ---- 2. El plan: predecir no es decidir
v.b("El recorrido tiene tres pasos. [[a]]Primero, aprender de los correos: qué señales aparecen dentro de cada clase. [[b]]Segundo, calcular y comprobar: cómo se combinan esas señales y qué tan fiables son las probabilidades que salen. [[c]]Y tercero, tomar una decisión: cuándo conviene bloquear, considerando que los dos errores no cuestan lo mismo.",
    CL(), T("construir un filtro que podamos explicar", 960, 150, tam=54, c="tinta", ent="subir"),
    flujo([("aprender de los correos", "azul", "a"), ("calcular y comprobar", "violeta", "b"), ("tomar una decisión", "ambar", "c")],
          y=470, w=480, h=230, x0=420, paso=540, tam=40),
    diapo(3, en="a"), sec="2. Predecir no es decidir")
v.b("El profesor insistió en una distinción que vamos a usar toda la clase. [[a]]Predecir es llegar a una probabilidad: este correo tiene tal probabilidad de ser spam. [[b]]Decidir es aplicar una regla sobre esa probabilidad: lo bloqueo o lo dejo pasar. [[c]]Una cosa alimenta a la otra, pero son pasos distintos. Las primeras diez paradas son sobre predecir; la parada once, sobre decidir.",
    CL(), cita("el acto de predecir y el acto de tomar una decisión son cosas distintas", y=230, tam=44),
    tarjetas([(620, "violeta", "predecir", "llegar a una probabilidad de spam", "a"), (1300, "ambar", "decidir", "aplicar una regla: bloquear o dejar pasar", "b")],
             y=560, w=560, h=300, tam_t=50, tam_d=38),
    T("paradas 1 a 10: predecir  ·  parada 11: decidir", 960, 860, tam=40, c="tinta", w="sb", en="c"),
    grabacion("00:07:10"))

# ---- 3. El nombre
v.b("El método de hoy se llama Naive Bayes. [[a]]En español, clasificador bayesiano ingenuo. [[b]]Bayes, por el teorema de probabilidad condicional que vamos a usar. [[c]]E ingenuo, por un supuesto simplificador que haremos en la parada cuatro, y que casi nunca se cumple del todo. Lo interesante es que, aun así, funciona razonablemente bien.",
    CL(), T("Naive Bayes", 960, 200, tam=110, c="violeta", ent="subir"),
    T("clasificador bayesiano ingenuo", 960, 330, tam=50, c="tinta", w="sb", en="a"),
    tarjetas([(620, "azul", "Bayes", "el teorema de probabilidad condicional", "b"), (1300, "ambar", "ingenuo", "un supuesto simplificador (parada 4)", "c")],
             y=620, w=560, h=280, tam_t=50, tam_d=38), sec="3. El nombre del método")

# ---- 4. Analogía
v.b("Antes de los números, una analogía. [[a]]Piensa en una alarma de humo. Una pregunta es: de todas las veces que hubo un incendio, ¿cuántas veces sonó la alarma? [[b]]Otra pregunta es: de todas las veces que sonó la alarma, ¿cuántas veces había un incendio? [[c]]La primera puede ser casi cien por ciento, y la segunda, bastante baja, porque la alarma también suena con una tostada quemada.",
    CL(), T("la alarma de humo", 960, 150, tam=56, c="tinta", ent="subir"),
    P("de los incendios, ¿cuántas veces sonó?", 960, 340, "azul", 48, en="a"),
    P("de las veces que sonó, ¿cuántas había incendio?", 960, 520, "ambar", 48, en="b"),
    T("la primera puede ser casi 100 %; la segunda, baja (tostadas quemadas)", 960, 720, tam=40, c="tinta", w="sb", mw=1500, en="c"),
    analogia(1050), sec="4. Analogía: la alarma de humo")

# ---- 5. Las tres preguntas
v.b("Con los correos pasa lo mismo, y la diapositiva cuatro les pone nombre. [[a]]La previa: ¿qué fracción de los correos es spam? Ahí solo miramos la clase. [[b]]La condicional: entre los spam, ¿qué fracción contiene la palabra oferta? Miramos la palabra dentro de la clase. [[c]]Y la posterior: entre los correos con oferta, ¿qué fracción es spam? Miramos la clase cuando ya conocemos la palabra.",
    CL(), T("tres preguntas distintas · diapositiva 4", 960, 140, tam=52, c="tinta", ent="subir"),
    tabla([["nombre", "pregunta en el ejemplo", "qué observamos"],
           ["previa", "¿qué fracción de los correos es spam?", "la clase"],
           ["condicional", "entre los spam, ¿qué fracción tiene oferta?", "la palabra dentro de la clase"],
           ["posterior", "entre los correos con oferta, ¿qué fracción es spam?", "la clase al conocer la palabra"]],
          [330, 900, 1560], 300, paso=130, tam=32, x0=140, x1=1800,
          colores=[["tinta"] * 3, ["azul", "tinta", "tinta"], ["teal", "tinta", "tinta"], ["violeta", "tinta", "tinta"]],
          ens=[None, "a", "b", "c"]),
    diapo(4, en="a"), sec="5. Las tres preguntas")
v.b("La idea central de Bayes es justamente esa: [[a]]pasar de lo que sabemos dentro de cada clase, la condicional, [[b]]a una probabilidad para el correo que estamos mirando, la posterior. Para clasificar necesitamos la posterior; pero los datos nos dan con facilidad la condicional.",
    P("condicional: lo que sabemos dentro de cada clase", 680, 860, "teal", 38, en="a"),
    L(1050, 860, 1190, 860, "gris", 6, True, en="b"), P("posterior", 1340, 860, "violeta", 44, en="b"))

# ---- 6. La tabla de 100 correos
v.b("Ahora sí, los números. La diapositiva cinco construye un ejemplo con cien correos. [[a]]Hay ochenta legítimos: ocho tienen la palabra oferta y setenta y dos no. [[b]]Hay veinte spam: dieciséis tienen oferta y cuatro no. [[c]]En total, veinticuatro correos traen oferta y setenta y seis no. Es un ejemplo construido para las cuentas, no son conteos de Spambase.",
    CL(), T("un ejemplo con 100 correos · diapositiva 5", 960, 140, tam=52, c="tinta", ent="subir"),
    tabla([["clase", "con «oferta»", "sin «oferta»", "total"], ["legítimo (0)", "8", "72", "80"], ["spam (1)", "16", "4", "20"],
           ["total", "24", "76", "100"]], [480, 860, 1200, 1500], 300, paso=110, tam=44, x0=240, x1=1680,
          colores=[["tinta"] * 4, ["azul"] + ["tinta"] * 3, ["rojo"] + ["tinta"] * 3, ["tinta"] * 4], ens=[None, "a", "b", "c"]),
    T("ejemplo construido para las cuentas de las diapositivas 5 a 25; no son conteos de Spambase", 960, 820, tam=32, c="gris", w="sb", mw=1500, en="c"),
    diapo(5, en="a"), sec="6. La tabla de 100 correos")
v.b("Mira esta tabla con dos lecturas. [[a]]Si recorres la fila del spam, ves que la palabra aparece en dieciséis de los veinte spam. [[b]]Si recorres la columna de oferta, ves que de los veinticuatro correos con oferta, dieciséis son spam. El mismo dieciséis, pero dividido por cosas distintas.",
    R(960, 520, 1500, 100, "rojo", r=20, en="a"), R(860, 520, 200, 340, "violeta", r=20, en="b"))

# ---- 7. La condicional
v.b("Calculemos la condicional, como en la diapositiva seis. [[a]]La probabilidad de oferta dado spam es dieciséis sobre veinte: 0.80. [[b]]La probabilidad de oferta dado legítimo es ocho sobre ochenta: 0.10. [[c]]Ojo: estas dos no suman uno, y no tienen por qué, porque cada una vive en una clase distinta. Lo que sí suma uno es oferta y no oferta dentro de la misma clase: ocho sobre ochenta, más setenta y dos sobre ochenta.",
    CL(), T("la dirección del condicionamiento · diapositiva 6", 960, 140, tam=50, c="tinta", ent="subir"),
    T("P(oferta | spam) = 16/20 =", 1000, 310, tam=50, c="tinta", w="sb", anc="rm", en="a"), N(0.80, 1030, 310, "{:.2f}", 66, "rojo", anc="lm", en="a", t=1.2),
    T("P(oferta | legítimo) = 8/80 =", 1000, 450, tam=50, c="tinta", w="sb", anc="rm", en="b"), N(0.10, 1030, 450, "{:.2f}", 66, "azul", anc="lm", en="b", t=1.2),
    P("0.80 + 0.10 no tiene por qué sumar 1: son clases distintas", 960, 610, "ambar", 40, en="c"),
    T("dentro de legítimos: 8/80 + 72/80 = 0.10 + 0.90 = 1", 960, 740, tam=40, c="teal", w="sb", en="c", t=2.5),
    diapo(6, en="a"), sec="7. La condicional")

# ---- 8. La posterior
v.b("Y ahora la posterior. [[a]]Para clasificar no tomamos como referencia todos los spam, sino todos los correos con oferta. [[b]]La probabilidad de spam dado oferta es dieciséis sobre dieciséis más ocho: dieciséis sobre veinticuatro, [[c]]dos tercios, aproximadamente 0.6667.",
    CL(), T("la posterior: referencia = los correos con oferta", 960, 140, tam=50, c="tinta", ent="subir"),
    P("referencia: los 24 correos con «oferta» (la columna)", 960, 300, "violeta", 44, en="a"),
    T("P(spam | oferta) = 16 / (16 + 8) = 16/24", 960, 420, tam=50, c="tinta", w="sb", en="b"),
    T("= 2/3 ≈", 1000, 530, tam=50, c="tinta", w="sb", anc="rm", en="c"),
    N(0.6667, 1030, 530, "{:.4f}", 70, "violeta", anc="lm", en="c", t=1.0), diapo(6, en="a"), sec="8. La posterior")
v.b("La frase que resume la diapositiva: [[a]]el ochenta por ciento de los spam contiene oferta, [[b]]pero solo el 66.7 por ciento de los correos con oferta es spam. [[c]]Y el resto, ocho de veinticuatro, un tercio, son legítimos: las dos posteriores sí suman uno, porque comparten la misma referencia.",
    T("80 % de los spam contiene «oferta»", 960, 640, tam=46, c="rojo", w="b", en="a"),
    T("pero solo 66.7 % de los correos con «oferta» es spam", 960, 740, tam=46, c="violeta", w="b", en="b"),
    T("legítimo dado oferta: 8/24 ≈ 0.3333  ·  0.6667 + 0.3333 = 1", 960, 860, tam=40, c="teal", w="sb", en="c"))

# ---- 9. Error típico
v.b("Error típico, y es el más clásico de la probabilidad. [[a]]Confundir la probabilidad de oferta dado spam con la de spam dado oferta. [[b]]Decir: como el ochenta por ciento de los spam trae oferta, un correo con oferta tiene ochenta por ciento de ser spam. [[c]]Es falso: depende también de cuántos legítimos traen oferta y de cuántos spam hay en total. Eso es lo que hace el teorema de Bayes, en la próxima parada.",
    CL(), K(330, 300, "cruz", 64, en="a"), T("P(oferta | spam) ≠ P(spam | oferta)", 410, 300, tam=48, c="rojo", anc="lm", en="a"),
    K(330, 450, "cruz", 64, en="b"), T("«80 % de los spam trae oferta → 80 % de ser spam»", 410, 450, tam=40, c="rojo", anc="lm", en="b"),
    K(330, 620, "check", 64, en="c"), T("también importan los legítimos con oferta y la cantidad de spam", 410, 620, tam=38, c="verde", anc="lm", mw=1300, en="c"),
    sec="9. Error típico")

# ---- 10. Piensa un segundo
v.b("Piensa un segundo. [[a]]Con la misma tabla: entre los correos que no traen la palabra oferta, [[b]]¿qué fracción es spam?",
    CL(), tabla([["clase", "con «oferta»", "sin «oferta»", "total"], ["legítimo (0)", "8", "72", "80"], ["spam (1)", "16", "4", "20"],
                 ["total", "24", "76", "100"]], [480, 860, 1200, 1500], 260, paso=96, tam=40, x0=240, x1=1680),
    P("entre los correos sin «oferta», ¿qué fracción es spam?", 960, 760, "violeta", 46, en="b"), sec="10. Piensa un segundo", pausa=0.6)
v.b("Miramos la columna sin oferta: [[a]]hay setenta y seis correos, y de ellos cuatro son spam. [[b]]Cuatro sobre setenta y seis, aproximadamente 0.053. Sin la palabra, la probabilidad de spam cae a cerca del cinco por ciento.",
    R(1200, 452, 190, 280, "violeta", r=20, en="a"),
    T("P(spam | sin oferta) = 4/76 ≈", 1040, 900, tam=48, c="tinta", w="sb", anc="rm", en="a"), N(0.053, 1070, 900, "{:.3f}", 64, "violeta", anc="lm", en="b"),
    ilustrativo(1050, texto="Cálculo adicional con la tabla de la diapositiva 5"))

# ---- 11. Para llevar
v.b("Para llevarte: [[a]]la previa mira la clase; la condicional, la señal dentro de la clase; la posterior, la clase cuando ya conocemos la señal. [[b]]Con los cien correos: oferta dado spam es 0.80, pero spam dado oferta es dos tercios. [[c]]Y para clasificar necesitamos la posterior: eso es lo que construye el teorema de Bayes.",
    CL(), P("previa · condicional · posterior: tres preguntas distintas", 960, 300, "violeta", 46, en="a"),
    P("P(oferta | spam) = 0.80   ·   P(spam | oferta) = 2/3", 960, 460, "teal", 46, en="b"),
    P("para clasificar: la posterior", 960, 620, "ambar", 48, en="c"), sec="11. Para llevar")
v.b("Siguiente parada: [[a]]el teorema de Bayes, que recupera esta misma cuenta, y por qué la frecuencia de spam cambia la respuesta.",
    CL(), siguiente(2, "El teorema de Bayes y la frecuencia de spam"), sec="12. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
