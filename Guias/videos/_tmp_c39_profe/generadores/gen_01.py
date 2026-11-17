import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util39 import (D, out, tarjetas, siguiente, ilustrativo, analogia, simulado, diapo, etiqueta, cita, grabacion, fila,
                    grupos_series, serie_puntos, revelar, cruz, GCOL)

OUT = out("guion_c39_01_agrupar.json")
v = V("Agrupar sin etiquetas")
SB = D["sim_blobs"]

v.portada("Agrupar sin etiquetas", "Qué es el clustering y qué entrega",
          "Bienvenidos a la clase treinta y nueve: clustering, o agrupamiento. Vamos a aprender a encontrar grupos en datos que no traen ninguna etiqueta.")

# ---- 1. Hook
v.b("Imagina que trabajas en un banco. [[a]]Quieres ofrecer inversiones de riesgo bajo, medio o alto, según el perfil de cada cliente. [[b]]Pero no puedes llamar a cada persona y preguntarle: ¿usted es de riesgo bajo o de riesgo alto? Muchas ni siquiera lo saben. [[c]]Lo que sí tienes son datos: cuánto ahorran, cuánto gastan, qué productos usan. La pregunta de esta clase es: ¿podemos construir esos grupos solo con los datos?",
    CL(), T("el banco y sus clientes", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "verde", "riesgo bajo", "¿quiénes?", "a"), (960, "ambar", "riesgo medio", "¿quiénes?", "a"),
              (1440, "rojo", "riesgo alto", "¿quiénes?", "a")], y=420, w=400, h=260, tam_t=46, tam_d=40),
    P("no se puede preguntar la etiqueta", 960, 680, "rojo", 46, en="b"),
    P("sí tenemos datos: ahorro, gasto, productos", 960, 820, "teal", 46, en="c"),
    grabacion("0:04:30", en="a"), sec="1. El problema")

# ---- 2. Analogía
v.b("Una analogía. [[a]]Te entregan una caja con cientos de botones mezclados y te piden que los ordenes, sin instrucciones. [[b]]Podrías separarlos por color. O por tamaño. O por número de agujeros. [[c]]Las tres respuestas son razonables, y ninguna es la correcta: depende de para qué quieres los grupos. Eso es exactamente el clustering.",
    CL(), T("la caja de botones", 960, 150, tam=56, c="tinta", ent="subir"),
    Pt([(30, (560, 260, 1360, 470), "azul"), (25, (560, 260, 1360, 470), "rojo"), (20, (560, 260, 1360, 470), "ambar")], radio=13, en="a"),
    tarjetas([(480, "azul", "por color", "un criterio", "b"), (960, "rojo", "por tamaño", "otro criterio", "b"),
              (1440, "ambar", "por agujeros", "otro más", "b")], y=660, w=400, h=200, tam_t=42, tam_d=34),
    P("no hay una respuesta correcta: depende del propósito", 960, 880, "violeta", 44, en="c"),
    analogia(1050), sec="2. Analogía: la caja de botones")

# ---- 3. Supervisado y no supervisado
v.b("Pongamos esto junto a lo que ya sabes. [[a]]En clasificación supervisada, como la regresión logística, SVM o Naive Bayes, observamos las variables y también una clase conocida, y aprendemos a predecir esa clase en casos nuevos. [[b]]En clustering solo observamos las variables. No hay respuesta que predecir: buscamos una partición, y describimos en qué se parecen y en qué se diferencian los grupos. [[c]]Por eso, la etiqueta de un clúster se construye a partir de los datos.",
    CL(), T("clasificación y agrupación · diapositiva 2", 960, 150, tam=50, c="tinta", ent="subir"),
    tarjetas([(560, "azul", "clasificación supervisada", "variables + clase conocida → predecir la clase de casos nuevos", "a"),
              (1360, "teal", "clustering", "solo variables → buscar una partición y describir los grupos", "b")], y=470, w=700, h=380, tam_t=46, tam_d=36),
    P("la etiqueta del clúster sale de los datos", 960, 790, "violeta", 46, en="c"), diapo(2, en="a"), sec="3. Supervisado frente a no supervisado")
v.b("Y una advertencia que la diapositiva dos subraya. [[a]]El grupo uno no significa positivo, ni mejor, ni de mayor riesgo. [[b]]Es solo un nombre. Si mañana el algoritmo numera distinto, el grupo uno puede pasar a llamarse grupo tres, y nada habrá cambiado en los datos.",
    P("grupo 1 ≠ «positivo», «mejor» ni «mayor riesgo»", 960, 920, "ambar", 44, en="a"))

# ---- 4. El resultado de un agrupamiento
v.b("¿Qué entrega un agrupamiento? Miremos un ejemplo simulado, como el de la diapositiva tres. [[a]]Noventa observaciones en dos variables, sin ninguna etiqueta: todas del mismo color. [[b]]K-means las reparte en tres grupos, y marca con una cruz el centro de cada uno. [[c]]Al final obtenemos dos cosas: una etiqueta por observación, y un perfil de cada grupo, que aquí es su centro.",
    CL(), T("el resultado de un agrupamiento · diapositiva 3", 960, 140, tam=48, c="tinta", ent="subir"),
    G(390, 220, 1050, 840, -5.5, 5.5, -5, 5.5, [serie_puntos(SB["x"], SB["y"], "gris", 8)], xticks=[-4, 0, 4], yticks=[-4, 0, 4],
      yfmt="{:.0f}", xlabel="variable 1", ylabel="var. 2", id="gb1", tam_txt=26),
    revelar("gb1", 1, en="a", dur=1.2),
    G(1170, 220, 1830, 840, -5.5, 5.5, -5, 5.5, grupos_series(SB["x"], SB["y"], SB["lab"], mk=8), xticks=[-4, 0, 4], yticks=[-4, 0, 4],
      yfmt="{:.0f}", xlabel="variable 1", ylabel="", id="gb2", tam_txt=26, en="b"),
    revelar("gb2", 3, en="b", dur=1.4),
    [cruz(cx, cy, "tinta", 50, ref="gb2", en="b", t=1.6) for cx, cy in zip(SB["cx"], SB["cy"])],
    P("una etiqueta por observación + un perfil por grupo", 960, 985, "teal", 38, en="c"), simulado(1050, en="a"), sec="4. Qué entrega un agrupamiento")

v.b("¿Por qué no basta con mirar el gráfico? [[a]]Porque aquí hay solo dos variables, y los grupos se separan solos. [[b]]En la práctica tendrás diez, trece o cincuenta variables, y los grupos estarán mezclados. [[c]]Ahí nadie puede ver los grupos a ojo: necesitamos una regla para encontrar los centros, y otra para decidir cuántos grupos proponer. Esas son las dos dificultades de esta clase.",
    CL(), T("¿por qué no mirar el gráfico y listo?", 960, 150, tam=54, c="tinta", ent="subir"),
    tarjetas([(560, "verde", "2 variables", "los grupos se ven a ojo", "a"), (1360, "rojo", "13 o 50 variables", "grupos mezclados: no se ven", "b")],
             y=430, w=620, h=280, tam_t=50, tam_d=38),
    P("dificultad 1: encontrar los centros", 960, 700, "azul", 44, en="c"), P("dificultad 2: decidir cuántos grupos", 960, 820, "violeta", 44, en="c", t=1.5),
    grabacion("0:08:49", en="a"))

# ---- 5. Similitud y propósito
v.b("Antes de agrupar hay una decisión que ningún algoritmo toma por ti: qué significa que dos observaciones se parezcan. [[a]]En clientes, podrías usar la frecuencia de compra, el gasto o el canal preferido. [[b]]En vinos, sus mediciones químicas, sin ninguna nota de calidad. [[c]]En documentos, alguna representación de su contenido. Cambia las variables, y cambian los grupos.",
    CL(), T("similitud y propósito · diapositiva 4", 960, 150, tam=52, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "clientes", "frecuencia, gasto, canal", "a"), (960, "violeta", "vinos", "mediciones químicas", "b"),
              (1440, "teal", "documentos", "representación del contenido", "c")], y=470, w=420, h=300, tam_t=48, tam_d=36),
    diapo(4, en="a"), sec="5. Similitud y propósito")
v.b("Y ojo con esto: [[a]]un grupo muy compacto puede ser poco útil para el propósito del análisis. Que los puntos queden juntos no garantiza que el grupo sirva para tomar una decisión.",
    P("compacto no es lo mismo que útil", 960, 780, "ambar", 48, en="a"))

v.b("¿Dónde se usa esto? El profesor contó varios casos. [[a]]Segmentar clientes, para ofrecer productos o publicidad distinta a cada grupo. [[b]]Tractores del mismo modelo, donde algunos fallan cuatro o cinco veces más que otros: se agrupan, y después se investiga qué tienen en común. [[c]]O edificios en construcción que se comportan distinto porque están cerca del metro, una variable que nadie había registrado. El patrón es el mismo: primero se encuentran los grupos, y después se explica por qué existen.",
    CL(), T("¿dónde se usa?", 960, 150, tam=56, c="tinta", ent="subir"),
    tarjetas([(480, "azul", "clientes", "productos y publicidad por grupo", "a"), (960, "rojo", "tractores", "algunos fallan 4 o 5 veces más", "b"),
              (1440, "teal", "edificios", "cerca o lejos del metro", "c")], y=450, w=420, h=300, tam_t=48, tam_d=36),
    P("primero se encuentran los grupos; después se explica por qué existen", 960, 760, "violeta", 42, en="c", t=1.8),
    grabacion("1:16:26", en="b"))

# ---- 6. Tres métodos
v.b("La clase presenta tres métodos, y la elección depende del tipo de variables. [[a]]K-means, para variables numéricas: cada grupo se representa por la media de cada variable. [[b]]K-modes, para variables categóricas: se representa por la moda, la categoría más frecuente. [[c]]Y K-prototypes, para variables mixtas: medias para las numéricas y modas para las categóricas.",
    CL(), T("el método depende de las variables · diapositiva 5", 960, 150, tam=48, c="tinta", ent="subir"),
    fila(290, [("método", 520), ("variables", 960), ("representante", 1400)], tams=[34] * 3, colores=["gris"] * 3),
    fila(400, [("K-means", 520), ("numéricas", 960), ("media por variable", 1400)], tams=[42] * 3, colores=["azul", "tinta", "tinta"],
         en="a", fondo="azulclaro"),
    fila(520, [("K-modes", 520), ("categóricas", 960), ("moda por variable", 1400)], tams=[42] * 3, colores=["violeta", "tinta", "tinta"],
         en="b", fondo="violetaclaro"),
    fila(640, [("K-prototypes", 520), ("mixtas", 960), ("medias y modas", 1400)], tams=[42] * 3, colores=["teal", "tinta", "tinta"],
         en="c", fondo="tealclaro"),
    diapo(5, en="a"), sec="6. Tres métodos")
v.b("Los tres tienen algo en común: [[a]]hay que fijar el número de grupos, k, antes de ajustar. Y el costo que minimizan depende de cómo representamos los datos y de cómo medimos la diferencia entre ellos. Elegir k será uno de los grandes temas de la clase.",
    P("los tres requieren fijar k", 960, 800, "rojo", 50, en="a"))

# ---- 7. Error típico
v.b("El error típico al empezar: [[a]]leer el número del grupo como si fuera un orden o una calificación. Decir, por ejemplo, el grupo uno son los mejores clientes. [[b]]El número lo pone el algoritmo, y es arbitrario. [[c]]El significado de un grupo hay que construirlo después, mirando su perfil: sus medias, sus modas, su tamaño.",
    CL(), K(330, 300, "cruz", 70, en="a"), T("«el grupo 1 son los mejores clientes»", 410, 300, tam=46, c="rojo", anc="lm", en="a"),
    T("el número es arbitrario: lo pone el algoritmo", 960, 470, tam=44, c="tinta", w="sb", en="b"),
    K(330, 640, "check", 70, en="c"), T("el significado sale del perfil: medias, modas, tamaño", 410, 640, tam=44, c="verde", anc="lm", en="c"),
    sec="7. El error típico")

# ---- 8. Piensa un segundo
v.b("Piensa un segundo. [[a]]Una tienda quiere agrupar a sus clientes usando el gasto mensual, en pesos, y el canal preferido: app, web o sucursal. [[b]]¿Cuál de los tres métodos usarías?",
    CL(), T("gasto mensual (pesos) + canal preferido (app, web, sucursal)", 960, 330, tam=46, c="tinta", w="b", en="a"),
    P("¿K-means, K-modes o K-prototypes?", 960, 540, "violeta", 54, en="b"), sec="8. Piensa un segundo", pausa=0.6)
v.b("K-prototypes. [[a]]El gasto es numérico y el canal es categórico: son variables mixtas. [[b]]Con K-means tendrías que convertir el canal en números, y eso inventa distancias que no existen. Con K-modes tendrías que cortar el gasto en categorías, y perderías información.",
    CL(), P("K-prototypes: variables mixtas", 960, 330, "teal", 58, en=None),
    T("K-means: inventaría distancias entre canales", 960, 520, tam=42, c="tinta", w="sb", en="a"),
    T("K-modes: obligaría a cortar el gasto en categorías", 960, 620, tam=42, c="tinta", w="sb", en="b"), ilustrativo(1050))

# ---- 9. Para llevar
v.b("Para llevarte: [[a]]el clustering busca grupos sin una respuesta observada; la etiqueta sale de los datos. [[b]]Qué significa parecerse depende de las variables y del propósito. [[c]]Y el método se elige según el tipo de variables: K-means, K-modes o K-prototypes, todos con un número de grupos fijado de antemano.",
    CL(), P("sin respuesta observada: la etiqueta sale de los datos", 960, 300, "violeta", 46, en="a"),
    P("parecerse depende de las variables y del propósito", 960, 460, "teal", 46, en="b"),
    P("numéricas, categóricas o mixtas → K-means, K-modes, K-prototypes", 960, 620, "azul", 42, en="c"), sec="9. Para llevar")
v.b("Siguiente parada: [[a]]los datos de la clase, ciento setenta y siete vinos, y por qué hay que estandarizarlos antes de medir distancias.",
    CL(), siguiente(2, "Los vinos y la estandarización"), sec="10. Siguiente parada")

n = v.guardar(OUT)
print(n, "beats")
