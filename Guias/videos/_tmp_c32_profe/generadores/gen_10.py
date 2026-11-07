import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util32 import D, out, tarjetas, siguiente, ilustrativo, caja_formula

OUT = out("guion_c32_10_repaso.json")
v = V("Repaso final")
SS, RE = D["scree_s"], D["real"]

v.portada("Repaso final", "Las seis ideas de la clase, las confusiones a evitar y tu criterio de dominio",
          "Repaso final de la clase treinta y dos. Las seis ideas, las confusiones que conviene corregir, y cómo saber si ya dominas el tema.")

# ---- 1. Seis ideas
IDEAS = [("1", "Más columnas no es más información", "costo · multicolinealidad · ruido · varianza", "rojo"),
         ("2", "Representar antes de reducir", "escalar · codificar · construir, con fit en entrenamiento", "teal"),
         ("3", "PCA = direcciones de máxima varianza", "λ = varianza de cada componente", "azul"),
         ("4", "Elegir k con criterio", "umbral · codo · validación cruzada en un Pipeline", "violeta"),
         ("5", "Leer las cargas", "correlación = carga · √λ · el nombre lo pones tú", "ambar"),
         ("6", "t-SNE explora, PCA produce", "vecindades · sin transform · no determinista", "verde")]


def idea(i, en, t=0.0):
    n, a, b, c = IDEAS[i]
    x, y = (520 if i % 2 == 0 else 1400), 330 + 220 * (i // 2)
    return [Bc(x, y, 820, 180, "blanco", c, 5, 28, sombra=True, en=en, ent="subir", t=t), T(n, x - 350, y, tam=72, c=c, en=en, t=t + 0.05),
            T(a, x + 40, y - 30, tam=36, c=c, w="b", mw=660, en=en, t=t + 0.1), T(b, x + 40, y + 34, tam=28, c="tinta", w="sb", mw=660, en=en, t=t + 0.2)]


v.b("Repasemos la clase en seis ideas. [[a]]Una: más columnas no es más información. Las variables sobran por costo, multicolinealidad, ruido y varianza. [[b]]Dos: primero se representa, y después se reduce.",
    CL(), T("la clase en 6 ideas", 960, 180, tam=48, c="tinta", ent="subir"), idea(0, "a"), idea(1, "b"), sec="1. Seis ideas")
v.b("[[a]]Tres: PCA busca las direcciones de máxima varianza, y lambda es la varianza de cada componente. [[b]]Cuatro: el número de componentes se elige con criterio, y se confirma con validación cruzada.",
    idea(2, "a"), idea(3, "b"))
v.b("[[a]]Cinco: los componentes se leen con sus cargas. [[b]]Y seis: t-SNE sirve para explorar; para modelar y producir, PCA.",
    idea(4, "a"), idea(5, "b"))

# ---- 2. Los números de la clase
NUM = [("segundos ↔ llamadas", "r = 0.99 · VIF 40.8 y 48.3", "rojo"), ("ruido con 100 filas", "AUC prueba 0.715 → 0.569", "ambar"),
       ("PCA de la clase", "PC1 34.9 % · 8 componentes = 85.0 %", "azul"), ("validación cruzada", "k = 3: AUC 0.773 · 15 variables: 0.746", "violeta"),
       ("sin estandarizar", "segundos = 99.2 % de la varianza", "rojo"), ("base real", "7 componentes = 80.4 % · F1 0.574 → 0.548", "verde")]
v.b("Y los números que conviene recordar. [[a]]Segundos y llamadas, con correlación cero coma noventa y nueve. [[b]]El ruido, que baja el AUC de prueba de 0.715 a 0.569. [[c]]PC1 con treinta y cuatro coma nueve por ciento, y ocho componentes para el ochenta y cinco.",
    CL(), T("los números de la clase", 960, 180, tam=48, c="tinta", ent="subir"),
    *[[Bc(960, 290 + 115 * i, 1600, 95, "blanco", c, 4, 22, en=k, ent="subir"), T(a, 220, 290 + 115 * i, tam=34, c=c, w="b", anc="lm", en=k, t=0.1),
       T(b, 1720, 290 + 115 * i, tam=34, c="tinta", w="sb", anc="rm", en=k, t=0.2)] for i, ((a, b, c), k) in enumerate(zip(NUM[:3], "abc"))],
    sec="2. Los números de la clase")
v.b("[[a]]En validación cruzada, tres componentes ganan a las quince variables. [[b]]Sin estandarizar, segundos se lleva el noventa y nueve coma dos por ciento. [[c]]Y en la base real, siete componentes dan el ochenta coma cuatro.",
    *[[Bc(960, 290 + 115 * (i + 3), 1600, 95, "blanco", c, 4, 22, en=k, ent="subir"), T(a, 220, 290 + 115 * (i + 3), tam=34, c=c, w="b", anc="lm", en=k, t=0.1),
       T(b, 1720, 290 + 115 * (i + 3), tam=34, c="tinta", w="sb", anc="rm", en=k, t=0.2)] for i, ((a, b, c), k) in enumerate(zip(NUM[3:], "abc"))])

# ---- 3. Un ejercicio integrador
v.b("Un ejercicio rápido, para cerrar. [[a]]Cuatro variables estandarizadas, con lambda dos coma dos, uno, cero coma cinco y cero coma tres. ¿Cuántos componentes necesitas para al menos el ochenta por ciento? ¿Y qué dice Kaiser?",
    CL(), T("4 variables estandarizadas", 960, 230, tam=44, c="gris", w="sb", en="a"), T("λ = 2.2 · 1.0 · 0.5 · 0.3", 960, 370, tam=70, c="violeta", en="a", t=0.3),
    T("¿k para ≥ 80 %?  ¿y Kaiser?", 960, 560, tam=70, c="tinta", en="a", t=0.8), ilustrativo(990, en="a"), sec="3. Un ejercicio integrador", pausa=0.8)
v.b("La suma es cuatro. [[a]]Dos coma dos entre cuatro: cincuenta y cinco por ciento. [[b]]Más uno entre cuatro, veinticinco: ochenta justo. Como pide al menos ochenta, bastan dos. [[c]]Kaiser exige lambda mayor que uno: solo el primero. Kaiser dice uno.",
    CL(), T("Σλ = 4", 960, 210, tam=52, c="gris", w="sb"),
    *[[Bc(x, 450, 480, 260, "blanco", c, 5, 30, sombra=True, en=k, ent="subir"), T(op, x, 400, tam=40, c="tinta", w="sb", en=k, t=0.1), N(ac, x, 490, "{:.0f}", 80, c, suf=" %", en=k, t=0.3, dur=0.6)]
      for x, c, op, ac, k in ((660, "gris", "2.2 / 4", 55, "a"), (1260, "verde", "+ 1.0 / 4", 80, "b"))],
    P("≥ 80 % → k = 2", 960, 700, "verde", 54, en="b", t=0.6), P("Kaiser (λ > 1) → k = 1", 960, 830, "rojo", 50, en="c"))

# ---- 4. Confusiones
ALR = [("«PCA es un clustering»", "PCA no forma grupos: es una proyección lineal; cambia los ejes", "rojo"),
       ("«7 componentes ≈ 80 %» con tabla de 8", "en el notebook 7 = 78.7 %, 8 = 85.0 %; en la base real 7 = 80.4 %", "ambar"),
       ("«PC2 = fallas y reclamos»", "en el notebook son PC3 (0.72 y 0.50); comprueba con components_", "ambar"),
       ("«ordinal: nominal → entero»", "la codificación ordinal es para variables con orden", "violeta"),
       ("«con PCA se cede algo de desempeño»", "F1 0.270 vs 0.267 con 57 fugas es ruido; en CV, k = 3 gana", "azul"),
       ("«fallas y reclamos opuestos al uso» (caso 3.2)", "fallas correlaciona +0.49, +0.57 y +0.59 con el uso", "rojo")]


def alr(i, en, t=0.0):
    f, c_, col = ALR[i]
    y = 290 + 125 * i
    return [Bc(960, y, 1720, 108, "blanco", col, 4, 22, sombra=True, en=en, ent="subir", t=t), T(f, 140, y - 22, tam=32, c=col, w="b", anc="lm", mw=1640, en=en, t=t + 0.1),
            T(c_, 140, y + 22, tam=28, c="tinta", w="sb", anc="lm", mw=1640, en=en, t=t + 0.2)]


v.b("Ahora, seis ideas del material que conviene corregir. [[a]]PCA no es un clustering. [[b]]Siete componentes no llegan al ochenta por ciento en los datos del notebook. [[c]]Y fallas y reclamos forman PC3, no PC2.",
    CL(), P("seis correcciones al material", 960, 170, "ambar", 44, ent="subir"), alr(0, "a"), alr(1, "b"), alr(2, "c"), sec="4. Seis correcciones")
v.b("[[a]]La codificación ordinal es para variables con orden. [[b]]La diferencia de F1 entre quince variables y ocho componentes es ruido. [[c]]Y en la base real, las fallas van en la misma dirección que el uso, no en la opuesta.",
    alr(3, "a"), alr(4, "b"), alr(5, "c"))

# ---- 5. Criterio de dominio
CRI = [("Calculo a mano la varianza explicada desde los λ y elijo k para un umbral", "azul"),
       ("Diagonalizo una matriz de correlación 2×2 y proyecto un punto", "violeta"),
       ("Leo cargas y escribo qué significa un componente, sin inventar", "ambar"),
       ("Elijo ordinal, one-hot o target encoding según el tipo de variable", "teal"),
       ("Escribo un Pipeline con ColumnTransformer, StandardScaler y PCA sin fuga", "verde"),
       ("Resuelvo los bloques 1 a 7 del cuaderno con al menos 80 % de aciertos", "rojo")]


def cri(i, en, t=0.0):
    txt, c = CRI[i]
    y = 290 + 120 * i
    return [Bc(960, y, 1640, 100, "blanco", c, 4, 24, en=en, ent="subir", t=t), T(txt, 220, y, tam=32, c="tinta", w="sb", anc="lm", mw=1380, en=en, t=t + 0.1),
            Bc(1730, y, 48, 48, "blanco", c, 4, 12, en=en, ent="fade", t=t + 0.2)]


v.b("Para pasar a la siguiente clase, deberías poder hacer seis cosas. [[a]]Calcular a mano la varianza explicada, y elegir k. [[b]]Resolver una matriz de dos por dos, y proyectar un punto. [[c]]Y leer las cargas sin inventar.",
    CL(), P("criterio de dominio", 960, 170, "violeta", 46, ent="subir"), cri(0, "a"), cri(1, "b"), cri(2, "c"), sec="5. Tu criterio de dominio")
v.b("[[a]]Elegir la codificación correcta según la variable. [[b]]Escribir un Pipeline sin fuga. [[c]]Y resolver los bloques uno a siete del cuaderno, con al menos ochenta por ciento de aciertos.",
    cri(3, "a"), cri(4, "b"), cri(5, "c"))

# ---- 6. Cómo estudiar
STOPS = [("apuntes", "teal"), ("cuaderno\nbloques 1–4", "violeta"), ("notebook\npartes A–C", "azul"), ("cuaderno\nbloques 5–7", "violeta"), ("notebook\nparte D", "azul"), ("ejercicio\nde cierre", "rojo")]
v.b("¿En qué orden estudiar? [[a]]Primero, los apuntes. [[b]]Luego, los bloques uno a cuatro del cuaderno, y las partes A a C del notebook. [[c]]Después, los bloques cinco a siete, y la parte D, la de t-SNE. [[d]]Y al final, el ejercicio de cierre.",
    CL(),
    [[Bc(200 + 300 * i, 480, 250, 170, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, 200 + 300 * i, 490, tam=30, c=c, w="b", mw=220, en=k, t=0.1), T(str(i + 1), 200 + 300 * i, 390, tam=44, c=c, en=k, t=0.1)]
     for i, ((n, c), k) in enumerate(zip(STOPS, "abbccd"))],
    [L(335 + 300 * i, 480, 365 + 300 * i, 480, "grisclaro", 5, True, en=k, t=0.3, dur=0.4) for i, k in enumerate("bbccd")], sec="6. Cómo estudiar")
v.b("Y si te trabas más de diez minutos, [[a]]mira la solución, [[b]]entiéndela, [[c]]y vuelve a hacerlo sin mirar.",
    CL(), P("más de 10 minutos trabado", 960, 280, "ambar", 56, ent="subir"), P("mira la solución", 500, 520, "azul", 52, en="a"), P("entiéndela", 1000, 520, "teal", 52, en="b"), P("hazlo sin mirar", 1500, 520, "violeta", 52, en="c"),
    L(740, 520, 800, 520, "gris", 5, True, en="b"), L(1200, 520, 1260, 520, "gris", 5, True, en="c"))

# ---- 7. Cierre
v.b("Eso es todo. Ya sabes diagnosticar cuándo sobran variables, [[a]]representarlas bien, resumirlas con PCA, y explorarlas con t-SNE. [[b]]Y recuerda la frase de la profesora: reducir dimensiones es eficiente, pero no es reducir por reducir. Mucho éxito con tu estudio.",
    CL(), T("diagnosticar · representar · reducir", 960, 340, tam=70, c="violeta", ent="subir", t=0.2), P("PCA para producir · t-SNE para explorar", 960, 540, "teal", 52, en="a"),
    P("«no es reducir por reducir»", 960, 740, "ambar", 56, en="b"), sec="7. Cierre")

n = v.guardar(OUT)
print(n, "beats")
