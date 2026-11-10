import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util38 import (D, out, tarjetas, siguiente, ilustrativo, real, adicional, diapo, etiqueta, cita, alerta, fila, codigo, casillas)

OUT = out("guion_c38_03_particion.json")
v = V("Separar por composición: grupos y prueba reservada")
HP = D["h_part"]
GE = D["h_grupo_ej"]

v.portada("Separar por composición", "Qué cuenta como un caso nuevo",
          "Tercera parada: cómo partir los datos del hormigón. La pregunta clave es qué cuenta como un caso nuevo.")

# ---- 1. Hook: la misma receta a distintas edades
F = GE["filas"]
v.b("Mira esta receta de hormigón, tal como aparece en los datos de la clase. [[a]]La misma mezcla, con los mismos siete ingredientes, se midió a distintas edades: a un día resistió 12.6 megapascales; a los 28 días, 44.1; a los 270 días, 55.2. [[b]]Son ocho filas, pero una sola composición. [[c]]Si una de esas filas cae en el entrenamiento y otra en la validación, ¿el modelo está prediciendo una mezcla nueva?",
    CL(), T("la misma receta, ocho edades", 960, 140, tam=54, c="tinta", ent="subir"),
    fila(270, [("edad (días)", 330)] + [(str(r["edad"]), 560 + 160 * i) for i, r in enumerate(F)], tams=[34] + [38] * len(F),
         colores=["gris"] * (len(F) + 1), ancl=["lm"] + ["mm"] * len(F)),
    fila(380, [("MPa", 330)] + [(f"{r['res']:.1f}", 560 + 160 * i) for i, r in enumerate(F)], tams=[34] + [40] * len(F),
         colores=["gris"] + ["azul"] * len(F), en="a", fondo="azulclaro", x0=300, x1=1820, h=96, ancl=["lm"] + ["mm"] * len(F)),
    P("8 filas = 1 composición", 960, 540, "violeta", 50, en="b"),
    T("¿una fila en entrenamiento y otra en validación = mezcla nueva?", 960, 700, tam=42, c="tinta", w="sb", en="c"),
    real(1050, en="a"), sec="1. La misma receta")

# ---- 2. Analogía
v.b("Piensa en un profesor que arma la prueba copiando los ejercicios de la guía, y solo les cambia un número. [[a]]Un alumno que memorizó la guía saca muy buena nota. [[b]]Pero esa nota no dice si sabe resolver un ejercicio realmente nuevo. [[c]]Con el hormigón pasa lo mismo: si la misma receta aparece en el ajuste y en la validación, el error medido es optimista.",
    CL(), T("la prueba copiada de la guía", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "nota alta", "memorizó la guía", "a"), (1360, "rojo", "¿sabe de verdad?", "no lo sabemos: no hubo ejercicios nuevos", "b")],
             y=460, w=640, h=320, tam_t=52, tam_d=40),
    P("misma receta en ajuste y validación → error optimista", 960, 780, "ambar", 46, en="c"), ilustrativo(1000, texto="analogía"),
    sec="2. Analogía: la prueba copiada")

# ---- 3. Grupos por igualdad exacta
v.b("Por eso la clase agrupa. [[a]]Dos filas pertenecen al mismo grupo si tienen exactamente los mismos siete ingredientes. [[b]]La edad y la respuesta no definen el grupo. [[c]]Y la partición reserva grupos completos: la composición entera, con todas sus edades, queda de un solo lado.",
    CL(), T("agrupar por composición · diapositiva 7", 960, 150, tam=50, c="tinta", ent="subir"),
    P("mismo grupo = mismos 7 ingredientes, exactos", 960, 300, "violeta", 48, en="a"),
    P("la edad y la resistencia NO definen el grupo", 960, 440, "gris", 44, en="b"),
    *[Bc(560 + 110 * i, 640, 90, 90, "azulclaro", "azul", 4, 16, en="c", t=0.08 * i) for i in range(4)],
    T("composición A: todas sus edades juntas", 780, 740, tam=32, c="azul", w="sb", en="c", t=0.4),
    *[Bc(1180 + 110 * i, 640, 90, 90, "ambarclaro", "ambar", 4, 16, en="c", t=0.5 + 0.08 * i) for i in range(3)],
    T("composición B", 1290, 740, tam=32, c="ambar", w="sb", en="c", t=0.8), diapo(7, en="a"), sec="3. Grupos por composición")
v.b("¿Importa mucho? [[a]]En estos datos, 182 composiciones tienen más de una fila. [[b]]Y esas composiciones reúnen 784 de las 1030 mediciones. [[c]]Es decir, si partiéramos fila por fila, la mayoría de las mediciones podría quedar con una hermana del otro lado.",
    CL(), T("¿cuántas filas comparten composición?", 960, 150, tam=52, c="tinta", ent="subir"),
    N(GE["grupos_con_varias"], 560, 380, "{:.0f}", 110, "violeta", en="a"), T("composiciones con más de una fila", 560, 470, tam=34, c="gris", w="sb", en="a", t=0.3),
    N(GE["filas_en_grupos_varias"], 1360, 380, "{:.0f}", 110, "azul", en="b"), T("de las 1030 mediciones", 1360, 470, tam=34, c="gris", w="sb", en="b", t=0.3),
    P("partir por filas → hermanas a ambos lados", 960, 660, "rojo", 48, en="c"), adicional(1050, en="a"))

v.b("¿Y por qué la edad no entra en la definición del grupo? [[a]]Si la edad definiera el grupo, la misma receta a los siete días y a los 28 días serían grupos distintos. [[b]]Entonces podrían quedar separadas: una en el ajuste y otra en la validación. Justo lo que queríamos evitar. [[c]]La edad sigue siendo un predictor, uno de los ocho; lo que no hace es decidir qué filas van juntas.",
    CL(), T("¿por qué la edad no define el grupo?", 960, 150, tam=52, c="tinta", ent="subir"),
    Bc(620, 400, 520, 150, "azulclaro", "azul", 4, 20, en="a"), T("receta A · 7 días", 620, 400, tam=40, c="azul", en="a", t=0.1),
    Bc(1300, 400, 520, 150, "ambarclaro", "ambar", 4, 20, en="a", t=0.4), T("receta A · 28 días", 1300, 400, tam=40, c="ambar", en="a", t=0.5),
    T("si la edad definiera el grupo, podrían quedar separadas", 960, 560, tam=40, c="rojo", w="sb", en="b"),
    P("la edad es predictor, pero no decide qué filas van juntas", 960, 730, "teal", 44, en="c"), ilustrativo(1000))

# ---- 4. La unidad de separación
v.b("La diapositiva trece muestra la partición de la clase. [[a]]{806|Ochocientas seis} mediciones, de 342 composiciones, para ajustar y seleccionar, con validación cruzada en cinco pliegues. [[b]]Y {224|doscientas veinticuatro} mediciones, de 86 composiciones, para la prueba reservada, que evalúa la elección final. [[c]]En total, 428 composiciones.",
    CL(), T("la unidad de separación · diapositiva 13", 960, 150, tam=50, c="tinta", ent="subir"),
    Bx(240, 280, 240 + 1440 * 0.8, 520, "azulclaro", "azul", 4, 20, grow="x", en="a"),
    T("806 mediciones · 342 composiciones", 240 + 720 * 0.8, 360, tam=42, c="azul", en="a", t=0.5),
    T("ajustar y seleccionar (CV, 5 pliegues)", 240 + 720 * 0.8, 440, tam=34, c="tinta", w="sb", en="a", t=0.7),
    Bx(240 + 1440 * 0.8 + 16, 280, 1696, 520, "ambarclaro", "ambar", 4, 20, grow="x", en="b"),
    T("224 · 86", 1568, 360, tam=42, c="ambar", en="b", t=0.5), T("prueba", 1568, 440, tam=32, c="tinta", w="sb", en="b", t=0.7),
    P("428 composiciones en total", 960, 640, "violeta", 50, en="c"), real(1050, en="a"), sec="4. La partición de la clase")
v.b("Un detalle fino: [[a]]se reservó el 20 por ciento de los grupos: 86 de 428 es el 20.1 por ciento. [[b]]Pero eso no exige el 20 por ciento de las filas: 224 de 1030 es el 21.7 por ciento, porque algunas composiciones tienen más edades que otras. [[c]]La partición se piensa en grupos, no en filas.",
    T("grupos: 86 / 428 = 20.1 %", 560, 790, tam=44, c="violeta", w="sb", en="a"),
    T("filas: 224 / 1030 = 21.7 %", 1360, 790, tam=44, c="gris", w="sb", en="b"),
    P("se piensa en grupos, no en filas", 960, 920, "teal", 46, en="c"))

# ---- 5. Los cinco pliegues
v.b("Dentro del entrenamiento, los 342 grupos se reparten en cinco pliegues. [[a]]Cada pliegue tiene 68 o 69 composiciones, [[b]]y entre 161 y 162 mediciones. [[c]]El script lo comprueba con una aserción: ningún grupo aparece en dos pliegues, y ninguno está a la vez en entrenamiento y en prueba.",
    CL(), T("cinco pliegues por composición", 960, 150, tam=52, c="tinta", ent="subir"),
    casillas([f"pliegue {k}" for k in range(1, 6)], 320, paso=330, w=290, h=100, c="violeta", tam=36, en="a", pre="pl"),
    fila(440, [(str(g), 300 + 330 * i) for i, g in enumerate(HP["fold_grupos"])], tams=[44] * 5, colores=["violeta"] * 5, en="a", t=0.6, x0=140, x1=1780),
    fila(520, [(str(f), 300 + 330 * i) for i, f in enumerate(HP["fold_filas"])], tams=[40] * 5, colores=["gris"] * 5, en="b", x0=140, x1=1780),
    T("composiciones", 1830, 440, tam=26, c="violeta", w="sb", anc="rm", en="a", t=0.8), T("mediciones", 1830, 520, tam=26, c="gris", w="sb", anc="rm", en="b", t=0.3),
    codigo(["assert set(entrenamiento.grupo).isdisjoint(prueba.grupo)",
            "assert entrenamiento.groupby('grupo').fold.nunique().eq(1).all()"], y=740, w=1640, tam=32, en="c"),
    real(1050, en="a"), sec="5. Cinco pliegues")

# ---- 6. Alerta del material
v.b("Una precisión sobre lo que se dijo en la grabación. [[a]]Al explicar el agrupamiento, se dijo: «al juntarlos voy a quedar con 342 composiciones distintas, que representan {806|ochocientas seis} mediciones de las {1030|mil treinta}». [[b]]Al juntar las 1030 mediciones, en realidad quedan 428 composiciones. [[c]]Las 342 son solo las del entrenamiento; las otras 86 están en la prueba. Más adelante, la misma grabación lo dice bien: 428 composiciones, y el 20 por ciento de los grupos para la prueba.",
    CL(), alerta(170, texto="precisión sobre la grabación"),
    cita("al juntarlos voy a quedar con 342 composiciones distintas, que representan 806 mediciones de las 1.030", 330, en="a", tam=40),
    etiqueta("grabación de la clase", 440, en="a", t=0.4),
    T("al juntar las 1030 mediciones quedan", 960, 570, tam=40, c="gris", w="sb", en="b"), N(HP["grupos"], 960, 660, "{:.0f}", 80, "violeta", en="b", t=0.6),
    P("342 en entrenamiento + 86 en prueba = 428", 960, 820, "teal", 46, en="c"), sec="6. Una precisión")

# ---- 7. El alcance: el oráculo
v.b("La prueba reservada cumple un papel especial. [[a]]El profesor la llamó un oráculo: se consulta una sola vez, al final. [[b]]Ninguna composición aparece a ambos lados de la partición. [[c]]Pero ojo con lo que eso no dice: no acredita que el modelo funcione en otra planta o en otro laboratorio. Solo aproxima composiciones excluidas de esta misma base histórica.",
    CL(), T("el alcance de la prueba", 960, 150, tam=54, c="tinta", ent="subir"),
    cita("va a funcionar como oráculo", 300, en="a", tam=48), etiqueta("grabación de la clase", 380, en="a", t=0.4),
    P("ninguna composición a ambos lados", 960, 500, "verde", 46, en="b"),
    P("NO acredita otra planta u otro laboratorio", 960, 640, "rojo", 46, en="c"),
    T("aproxima composiciones excluidas de la misma base histórica", 960, 760, tam=38, c="gris", w="sb", en="c", t=1.5),
    diapo(13, en="b"), sec="7. El oráculo y su alcance")

# ---- 8. El error típico
v.b("El error típico: [[a]]partir fila por fila, al azar, como si cada medición fuera independiente. [[b]]Entonces la receta que se midió a los siete días queda en el entrenamiento, y la misma receta a los 28 días queda en la validación. [[c]]El modelo parece predecir mezclas nuevas, pero solo está completando la curva de una mezcla que ya conoce.",
    CL(), K(330, 260, "cruz", 70, en="a"), T("partir fila por fila, al azar", 410, 260, tam=50, c="rojo", anc="lm", en="a"),
    Bc(620, 480, 520, 160, "azulclaro", "azul", 4, 20, en="b"), T("receta A, 7 días → ajuste", 620, 480, tam=32, c="azul", en="b", t=0.1),
    Bc(1300, 480, 520, 160, "ambarclaro", "ambar", 4, 20, en="b", t=0.6), T("receta A, 28 días → validación", 1300, 480, tam=32, c="ambar", en="b", t=0.7),
    P("no predice mezclas nuevas: completa una curva conocida", 960, 700, "ambar", 46, en="c"), ilustrativo(1000), sec="8. El error típico")

# ---- 9. Piensa un segundo
v.b("Piensa un segundo. [[a]]Tienes 50 composiciones y reservas 10 para la prueba. Esas 10 composiciones suman 60 filas, de un total de 250. [[b]]¿Qué porcentaje reservaste de los grupos? ¿Y de las filas?",
    CL(), T("50 composiciones · 250 filas", 960, 260, tam=56, c="tinta", w="b", en="a"),
    T("prueba: 10 composiciones con 60 filas", 960, 380, tam=48, c="ambar", w="sb", en="a", t=0.8),
    P("% de grupos = ?", 700, 560, "violeta", 50, en="b"), P("% de filas = ?", 1220, 560, "gris", 50, en="b", t=0.4),
    sec="9. Piensa un segundo", pausa=0.6)
v.b("De los grupos: 10 de 50, [[a]]el 20 por ciento. [[b]]De las filas: 60 de 250, [[c]]el 24 por ciento. No coinciden, y no tienen por qué coincidir: lo que se controla es que cada composición quede de un solo lado.",
    CL(), T("grupos: 10 / 50 =", 1000, 320, tam=52, c="tinta", w="sb", anc="rm"), N(20, 1030, 320, "{:.0f}", 70, "violeta", anc="lm", suf=" %", en="a"),
    T("filas: 60 / 250 =", 1000, 470, tam=52, c="tinta", w="sb", anc="rm", en="b"), N(24, 1030, 470, "{:.0f}", 70, "gris", anc="lm", suf=" %", en="c"),
    P("se controla que cada composición quede de un solo lado", 960, 660, "teal", 44, en="c", t=1.0), ilustrativo(1000))

# ---- 10. Para llevar
v.b("Para llevarte: [[a]]la unidad que separas define qué es un caso nuevo; en el hormigón, es la composición. [[b]]Se reservan grupos completos, con todas sus edades, y los pliegues también son por grupo. [[c]]Y la prueba es un oráculo que se consulta una sola vez, con un alcance limitado a esta base.",
    CL(), P("la unidad que separas define qué es un caso nuevo", 960, 300, "violeta", 46, en="a"),
    P("grupos completos en la prueba y en los pliegues", 960, 460, "azul", 46, en="b"),
    P("prueba = oráculo: una sola vez, alcance limitado", 960, 620, "ambar", 46, en="c"), sec="10. Para llevar")
v.b("Siguiente parada: [[a]]qué se ajusta y qué se selecciona, y los siete modelos candidatos que vamos a comparar.",
    CL(), siguiente(4, "Ajustar no es seleccionar: los candidatos"), sec="11. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
