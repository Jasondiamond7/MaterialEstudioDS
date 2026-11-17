import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, real, adicional, diapo, etiqueta, cita, grabacion, alerta,
                    fila, codigo, caja_formula, flujo)

OUT = out("guion_c39_12_repaso.json")
v = V("Comparar métodos, evaluar y repaso")
KM, KO, KP = D["km"], D["kmodes"], D["kproto"]

v.portada("Comparar, evaluar y repasar", "Costos que no se comparan, evaluación sin respuesta y el mapa de la clase",
          "Última parada: cómo comparar los tres métodos, cómo evaluar un agrupamiento sin una respuesta observada, y el repaso de toda la clase.")

# ---- 1. Hook
v.b("Tenemos tres soluciones de ejemplo, cada una con su costo. [[a]]K-means con tres grupos: 1267.37. K-modes con cuatro: 75. K-prototypes con tres: 326.64. [[b]]¿Gana K-modes, porque setenta y cinco es el número más chico? [[c]]No. Y entender por qué es la lección más importante de esta parada.",
    CL(), T("tres métodos, tres costos", 960, 150, tam=56, c="tinta", ent="subir"),
    fila(290, [("método", 520), ("k", 860), ("costo del ejemplo", 1200), ("significado", 1560)], tams=[32] * 4, colores=["gris"] * 4, x0=240, x1=1800),
    fila(390, [("K-means", 520), ("3", 860), (f"{KM['wss_k3']:.2f}", 1200), ("distancias al cuadrado", 1560)], tams=[40, 40, 40, 32],
         colores=["azul", "tinta", "azul", "tinta"], en="a", fondo="azulclaro", x0=240, x1=1800),
    fila(490, [("K-modes", 520), ("4", 860), (f"{KO['costo'][3]:.0f}", 1200), ("desacuerdos", 1560)], tams=[40, 40, 40, 32],
         colores=["violeta", "tinta", "violeta", "tinta"], en="a", t=1.2, fondo="violetaclaro", x0=240, x1=1800),
    fila(590, [("K-prototypes", 520), ("3", 860), (f"{KP['costo'][2]:.2f}", 1200), ("costo mixto, peso 1", 1560)], tams=[40, 40, 40, 32],
         colores=["teal", "tinta", "teal", "tinta"], en="a", t=2.4, fondo="tealclaro", x0=240, x1=1800),
    P("¿gana K-modes porque 75 es el menor?", 960, 760, "ambar", 48, en="b"), diapo(52, en="a"), sec="1. Tres costos")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Un estudiante sacó un 6 en una prueba con escala de uno a siete. Otro sacó un 80 en una prueba de cero a cien. Un tercero, un 15 en una de cero a veinte. [[b]]Nadie diría que el 6 es la peor nota porque es el número más chico. [[c]]Son escalas distintas, de pruebas distintas. Los costos de los tres métodos son exactamente así.",
    CL(), T("notas en escalas distintas", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "6", "escala de 1 a 7", "a"), (960, "violeta", "80", "escala de 0 a 100", "a"), (1440, "teal", "15", "escala de 0 a 20", "a")],
             y=420, w=400, h=260, tam_t=72, tam_d=36),
    P("el número más chico no es la peor nota", 960, 700, "ambar", 46, en="b"),
    P("escalas distintas: no se comparan directo", 960, 830, "rojo", 46, en="c"), analogia(1050), sec="2. Analogía: las notas")

# ---- 3. Por qué no se comparan
v.b("La diapositiva cincuenta y dos lo dice sin rodeos: [[a]]no elegimos K-modes porque setenta y cinco sea el menor número de la tabla. [[b]]Los datos de entrada son distintos: trece mediciones, tres categorías, o una mezcla. Las unidades son distintas: distancias al cuadrado, desacuerdos, o una suma ponderada por gamma. Y las funciones de costo son distintas. [[c]]La comparación pertinente comienza por el tipo de variables y el propósito del análisis.",
    CL(), T("comparación entre los tres métodos · diapositiva 52", 960, 140, tam=48, c="tinta", ent="subir"),
    P("no se elige K-modes porque 75 sea el menor", 960, 290, "rojo", 46, en="a"),
    T("distintos: datos de entrada · unidades · funciones de costo", 960, 440, tam=44, c="tinta", w="sb", en="b"),
    P("se compara por tipo de variables y propósito", 960, 600, "teal", 48, en="c"), diapo(52, en="a"), grabacion("2:00:40", en="a", y=1010),
    sec="3. Por qué no se comparan")
v.b("Una precisión sobre la grabación: [[a]]al pasar a K-modes se habló de un cambio en el costo, o en la varianza, básicamente. [[b]]El costo de K-modes no es una varianza: es un conteo de desacuerdos. Solo el WSS de K-means tiene la forma de una suma de cuadrados.",
    CL(), alerta(160, texto="precisión sobre la grabación"),
    cita("tuvimos un cambio en el costo, o en la varianza, básicamente", y=320, en="a"),
    P("costo de K-modes = número de desacuerdos, no una varianza", 960, 500, "verde", 44, en="b"),
    grabacion("2:00:18", en="a"))

# ---- 4. Regla de elección
v.b("Entonces, ¿cómo se elige el método? El profesor lo resumió de forma muy sencilla. [[a]]Si tienes solo variables numéricas: K-means. [[b]]Si tienes solo categóricas: K-modes. [[c]]Si tienes una mezcla: K-prototypes. Y dentro de cada método, el número de grupos se elige con los criterios internos que vimos.",
    CL(), T("¿qué método? depende de los datos", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "solo numéricas", "K-means", "a"), (960, "violeta", "solo categóricas", "K-modes", "b"), (1440, "teal", "mezcla", "K-prototypes", "c")],
             y=440, w=420, h=280, tam_t=44, tam_d=50),
    grabacion("2:00:59", en="a"), sec="4. Cómo elegir el método")

# ---- 5. Evaluación sin respuesta
v.b("Y ahora, la evaluación, de la diapositiva cincuenta y uno. [[a]]No calculamos exactitud ni área bajo la curva: no hay una clase observada que predecir. [[b]]Usamos criterios internos acordes con las variables y la disimilitud: WSS, silueta, Gap, costos. [[c]]Revisamos si los perfiles reaparecen al cambiar los inicios o al remuestrear los datos. Y contrastamos su utilidad con el problema que motivó el agrupamiento.",
    CL(), T("evaluación sin una respuesta observada · diapositiva 51", 960, 140, tam=46, c="tinta", ent="subir"),
    T("• sin exactitud ni AUC: no hay clase que predecir", 300, 290, tam=42, c="rojo", w="sb", anc="lm", en="a"),
    T("• criterios internos: WSS, silueta, Gap, costos", 300, 400, tam=42, c="azul", w="sb", anc="lm", en="b"),
    T("• estabilidad: ¿reaparecen los perfiles con otros inicios o remuestreando?", 300, 510, tam=42, c="teal", w="sb", anc="lm", en="c"),
    T("• utilidad para el problema que motivó el análisis", 300, 620, tam=42, c="violeta", w="sb", anc="lm", en="c", t=3.0),
    diapo(51, en="a"), sec="5. Evaluar sin respuesta")
v.b("Dos detalles de la misma diapositiva. [[a]]Esta exploración usa las ciento setenta y siete filas: no requiere automáticamente una división setenta treinta, porque no se está entrenando un modelo para predecir. [[b]]Pero si después quieres asignar casos futuros a estos grupos, debes fijar el escalamiento y los centros con los datos de desarrollo. Y evaluar ese uso futuro necesita un diseño separado.",
    CL(), P("exploración: las 177 filas, sin división 70/30 automática", 960, 330, "teal", 44, en="a"),
    P("asignar casos futuros: fijar escalamiento y centros", 960, 500, "ambar", 44, en="b"),
    T("y evaluar ese uso con un diseño separado", 960, 620, tam=40, c="tinta", w="sb", en="b", t=3.0),
    grabacion("1:59:49", en="a"), diapo(51, en="a", y=1010))

# ---- 6. Checklist
v.b("Antes de usar unos grupos para tomar decisiones, la diapositiva cincuenta y tres pide revisar cuatro cosas. [[a]]Uno: qué variables entraron, en qué escalas, y si hay observaciones atípicas. [[b]]Dos: los tamaños, los perfiles y la variación dentro de cada grupo. Tres: la sensibilidad a k, a la inicialización y al peso de las categorías. [[c]]Cuatro: si las conclusiones describen las mediciones o exceden la evidencia. Un agrupamiento organiza la información; su utilidad debe justificarse en el contexto.",
    CL(), T("interpretación de una segmentación · diapositiva 53", 960, 140, tam=48, c="tinta", ent="subir"),
    T("1. variables, escalas y atípicos", 330, 290, tam=44, c="azul", w="sb", anc="lm", en="a"),
    T("2. tamaños, perfiles y variación interna", 330, 400, tam=44, c="teal", w="sb", anc="lm", en="b"),
    T("3. sensibilidad a k, a los inicios y al peso γ", 330, 510, tam=44, c="violeta", w="sb", anc="lm", en="b", t=2.5),
    T("4. ¿las conclusiones exceden la evidencia?", 330, 620, tam=44, c="ambar", w="sb", anc="lm", en="c"),
    P("organiza la información; la utilidad se justifica en el contexto", 960, 790, "violeta", 40, en="c", t=3.0), diapo(53, en="a"),
    sec="6. Lista de revisión")

# ---- 7. Repaso: el mapa
v.b("Repasemos la clase completa como un solo flujo. [[a]]Primero, decidir qué significa parecerse, y estandarizar las numéricas. [[b]]Después, ajustar K-means con muchos inicios, quedándose con el menor WSS. [[c]]Elegir k con el codo, la silueta y el Gap, y aceptar que pueden discrepar. Y finalmente, describir los grupos con perfiles, dibujarlos con una proyección, y revisar sus límites.",
    CL(), T("el mapa de la clase", 960, 130, tam=56, c="tinta", ent="subir"),
    flujo([("parecido y escala", "azul", "a"), ("K-means, muchos inicios", "violeta", "b"), ("elegir k: codo, silueta, Gap", "teal", "c"),
           ("perfiles, proyección y límites", "ambar", "c")], y=430, w=380, h=230, x0=300, paso=440, tam=36),
    T("categóricas → K-modes      mixtas → K-prototypes (con γ)", 960, 720, tam=44, c="tinta", w="sb", en="c", t=3.0),
    sec="7. Repaso: el mapa")
v.b("Y los números que conviene recordar de los vinos. [[a]]Ciento setenta y siete vinos, trece mediciones estandarizadas. [[b]]Con tres grupos, un WSS de 1267.37, frente a un TSS de 2288; silueta 0.284, la mayor; el Gap prefería cuatro. [[c]]Grupos de sesenta y dos, cincuenta y sesenta y cinco vinos, y ciento setenta y uno de ciento setenta y siete coincidiendo con la variedad real, que nunca se usó para ajustar.",
    CL(), T("los vinos en cifras", 960, 150, tam=56, c="tinta", ent="subir"),
    P("177 vinos · 13 mediciones estandarizadas", 960, 300, "azul", 46, en="a"),
    P("k = 3: WSS 1267.37 de 2288 · silueta 0.284 · Gap: 4", 960, 450, "violeta", 44, en="b"),
    P("62 · 50 · 65 vinos · 171 / 177 con la variedad real", 960, 600, "teal", 44, en="c"), real(1050, en="a"))

# ---- 8. Errores típicos (resumen)
v.b("Y los errores que vimos a lo largo de la clase, en una sola lista. [[a]]Agrupar sin estandarizar. Correr K-means una sola vez. Elegir k con el menor WSS. [[b]]Leer el número del grupo como una calificación. Codificar categorías con números para usar K-means. [[c]]Comparar costos entre métodos distintos. Y reportar la coincidencia con una etiqueta como si fuera exactitud.",
    CL(), T("los errores típicos, juntos", 960, 140, tam=56, c="tinta", ent="subir"),
    [x for i, (txt, k) in enumerate([("agrupar sin estandarizar", "a"), ("un solo inicio", "a"), ("elegir k con el menor WSS", "a"),
                                    ("leer el número del grupo como nota", "b"), ("codificar categorías con números", "b"),
                                    ("comparar costos entre métodos", "c"), ("coincidencia con etiqueta = «exactitud»", "c")])
     for x in [K(300 + 760 * (i % 2), 290 + 110 * (i // 2), "cruz", 46, en=k, t=0.6 * (i % 3)),
               T(txt, 350 + 760 * (i % 2), 290 + 110 * (i // 2), tam=36, c="tinta", w="sb", anc="lm", en=k, t=0.6 * (i % 3))]],
    sec="8. Los errores típicos")

# ---- 9. Piensa
v.b("Un último piensa un segundo. [[a]]Una empresa tiene clientes con edad, gasto mensual, plan Gold o Silver, y canal App o Web. Quiere segmentarlos. [[b]]¿Qué método usas, y qué tienes que decidir antes de ajustarlo?",
    CL(), T("edad · gasto mensual · plan (Gold / Silver) · canal (App / Web)", 960, 320, tam=46, c="tinta", w="b", en="a"),
    P("¿qué método? ¿qué decidir antes?", 960, 520, "violeta", 56, en="b"), sec="9. Piensa un segundo", pausa=0.6)
v.b("K-prototypes, porque hay variables numéricas y categóricas. [[a]]Antes de ajustar: estandarizar la edad y el gasto, decidir el peso gamma, y probar varios valores de k y varios inicios. [[b]]Después, comparar los perfiles para distintos k y gamma, y quedarse con la solución estable que sirva para la decisión de la empresa.",
    CL(), P("K-prototypes: variables mixtas", 960, 280, "teal", 56, en=None),
    T("estandarizar edad y gasto · fijar γ · probar varios k e inicios", 960, 440, tam=42, c="tinta", w="sb", en="a"),
    T("comparar perfiles y elegir lo estable y útil", 960, 560, tam=42, c="tinta", w="sb", en="b"), ilustrativo(1050))

# ---- 10. Cierre
v.b("Para llevarte de toda la clase: [[a]]el clustering construye grupos sin una respuesta observada, y el método depende del tipo de variables. [[b]]Los números internos, WSS, silueta, Gap y costos, ayudan a elegir, pero no se comparan entre métodos y no reemplazan la interpretación. [[c]]Y un buen agrupamiento es el que es estable, interpretable y útil para el problema que lo motivó.",
    CL(), P("sin respuesta observada · método según las variables", 960, 300, "violeta", 46, en="a"),
    P("criterios internos ayudan, no reemplazan el juicio", 960, 460, "teal", 46, en="b"),
    P("estable, interpretable y útil", 960, 620, "ambar", 52, en="c"), sec="10. Para llevar")
v.b("Con esto termina la clase treinta y nueve. [[a]]La próxima clase vuelve al aprendizaje supervisado, con Naive Bayes y la clasificación probabilística. ¡Nos vemos ahí!",
    CL(), siguiente(40, "Naive Bayes y clasificación probabilística"), sec="11. Siguiente clase")

n = v.guardar(OUT)
print(n, "beats")
