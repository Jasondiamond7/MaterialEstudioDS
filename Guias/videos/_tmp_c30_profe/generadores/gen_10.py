import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *

OUT = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\guion_c30_10_repaso.json"
v = V("Repaso final")

v.portada("Repaso final", "Las ideas de la clase, conectadas en un solo recorrido",
          "Repaso final. Vamos a conectar todo lo que viste en un solo recorrido.")

v.b("Empecemos por la pregunta de la clase: [[a]]¿cómo sé que mi modelo aprendió, [[b]]y no solo memorizó? [[c]]Todo lo que viste sirve para responderla.",
    CL(), T("¿cómo sé que mi modelo aprendió…", 960, 330, tam=70, c="tinta", en="a"), T("…y no solo memorizó?", 960, 440, tam=70, c="violeta", en="b"),
    P("seis ideas para responderla", 960, 700, "teal", 64, en="c"), sec="1. Las seis ideas")

IDEAS = [("Tipos y calidad", "numérica o categórica; exactitud, completitud, coherencia y frescura", "teal"),
         ("Imputar con criterio", "media, mediana o moda, aprendida con el entrenamiento", "azul"),
         ("Tres conjuntos", "entrenamiento aprende, validación elige, prueba da el veredicto", "violeta"),
         ("Regularizar y estandarizar", "Ridge achica, Lasso puede anular; por eso se estandariza", "ambar"),
         ("K-Fold y estratificación", "un promedio y una dispersión; la clase rara, repartida en proporción", "verde"),
         ("Métricas y fuga", "precisión, recall y F1 según el costo; todo dentro de un Pipeline", "rojo")]
CXs = [500, 1420]
CYs = [330, 600, 870]


def idea(i, en, t=0.0):
    n, d, c = IDEAS[i]
    x, y = CXs[i % 2], CYs[i // 2]
    return [Bc(x, y, 860, 230, "blanco", c, 5, 30, sombra=True, id=f"id{i}", en=en, ent="subir", t=t),
            T(str(i + 1), x - 360, y, tam=110, c=c, id=f"in{i}", en=en, t=t + 0.1),
            T(n, x + 40, y - 52, tam=44, c=c, mw=640, id=f"it{i}", en=en, t=t + 0.15),
            T(d, x + 40, y + 36, tam=32, c="tinta", w="sb", mw=660, id=f"id_{i}", en=en, t=t + 0.25)]


def anillo(i, en=None):
    x, y = CXs[i % 2], CYs[i // 2]
    return R(x, y, 900, 270, "ambar", 36, id="ri", en=en, ent="fade")


v.b("Uno: [[a]]cada variable es numérica o categórica, y los datos se juzgan por su exactitud, completitud, coherencia y frescura.",
    CL(), idea(0, "a"), anillo(0, "a"))
v.b("Dos: [[a]]cuando faltan datos, se imputa con media, mediana o moda, según el caso. Y siempre aprendida con el entrenamiento.",
    idea(1, "a"), anillo(1, "a"))
v.b("Tres: [[a]]el entrenamiento aprende, la validación elige, [[b]]y la prueba da el veredicto final, una sola vez.",
    idea(2, "a"), anillo(2, "a"))
v.b("Cuatro: [[a]]Ridge achica los coeficientes y Lasso puede anularlos. [[b]]Y por eso se estandariza antes.",
    idea(3, "a"), anillo(3, "a"))
v.b("Cinco: [[a]]K-Fold te da un promedio y una dispersión, en vez de un número de suerte. [[b]]Y la estratificación reparte la clase rara en proporción.",
    idea(4, "a"), anillo(4, "a"))
v.b("Seis: [[a]]precisión, recall y efe uno, según el costo del error. [[b]]Y todo el preprocesamiento va dentro de un Pipeline.",
    idea(5, "a"), anillo(5, "a"))

# conexión
v.b("Fíjate cómo se conectan. [[a]]Primero preparas los datos. [[b]]Luego los divides. [[c]]Validas y eliges. [[d]]Y al final, mides, una sola vez con la prueba. [[e]]Todo dentro de un Pipeline, para no filtrar información.",
    CL(),
    [[Bc(x, 500, 340, 190, "blanco", c, 5, 28, sombra=True, en=k, ent="subir"), T(n, x, 470, tam=48, c=c, en=k, t=0.1), T(d, x, 550, tam=30, c="gris", w="sb", mw=290, en=k, t=0.2)]
     for x, c, n, d, k in ((330, "teal", "Preparar", "tipos, calidad, imputar", "a"), (780, "violeta", "Dividir", "entrenamiento, validación, prueba", "b"),
                           (1230, "azul", "Validar", "Ridge, Lasso, K-Fold", "c"), (1680, "rojo", "Medir", "métricas, prueba una vez", "d"))],
    [L(510 + 450 * i, 500, 600 + 450 * i, 500, "gris", 6, True, en=k, t=0.3) for i, k in enumerate("bcd")],
    R(1005, 500, 1820, 330, "teal", 50, id="rp", en="e", ent="fade"), P("Pipeline", 1005, 290, "teal", 56, en="e", t=0.2), sec="2. Cómo se conecta")
v.b("Piensa un segundo: [[a]]¿qué conjunto usas para elegir alfa en Ridge? [[b]]¿Y cuál para dar el veredicto final?",
    CL(), T("elegir alfa  →  ?", 960, 400, tam=80, c="violeta", en="a"), T("veredicto final  →  ?", 960, 600, tam=80, c="rojo", en="b"), pausa=0.5)
v.b("Alfa se elige con la validación. [[a]]El veredicto final lo da la prueba, una sola vez.",
    P("validación", 1500, 400, "azul", 60), P("prueba", 1500, 600, "rojo", 60, en="a"), K(1780, 400, "check", 60, en=None), K(1780, 600, "check", 60, en="a"))

# criterio de dominio
CRI = [("Clasifico variables y elijo media, mediana o moda, justificando", "teal"), ("Calculo a mano precisión, recall, F1 y accuracy desde una matriz de confusión", "azul"),
       ("Explico por qué se estandariza, y por qué el scaler se ajusta solo con entrenamiento", "ambar"), ("Calculo promedio y desviación de un K-Fold, y reparto una clase rara con estratificación", "verde"),
       ("Escribo un Pipeline con ColumnTransformer sin filtrar información de la prueba", "violeta"), ("Resuelvo los bloques 1 a 8 del cuaderno con al menos 80 % de aciertos", "rojo")]


def cri(i, en, t=0.0):
    txt, c = CRI[i]
    y = 270 + 128 * i
    return [Bc(960, y, 1640, 100, "blanco", c, 4, 24, en=en, ent="subir", t=t), T(txt, 260, y, tam=36, c="tinta", w="sb", anc="lm", mw=1300, en=en, t=t + 0.1),
            Bc(1760, y, 50, 50, "blanco", c, 4, 12, en=en, ent="fade", t=t + 0.2)]


v.b("Para pasar a la siguiente clase, deberías poder hacer seis cosas. [[a]]Clasificar variables y elegir cómo imputar. [[b]]Calcular a mano las métricas de una matriz de confusión. [[c]]Y explicar por qué se estandariza.",
    CL(), P("criterio de dominio", 960, 150, "violeta", 48, ent="subir"), cri(0, "a"), cri(1, "b"), cri(2, "c"), sec="3. Tu criterio de dominio")
v.b("[[a]]Calcular el promedio y la desviación de un K-Fold, y repartir una clase rara. [[b]]Escribir un Pipeline con ColumnTransformer sin filtrar la prueba. [[c]]Y resolver los bloques uno a ocho del cuaderno, con al menos ochenta por ciento de aciertos.",
    cri(3, "a"), cri(4, "b"), cri(5, "c"))
STOPS = ["apuntes", "cuaderno\nbloques 1–4", "notebook\npartes A y B", "cuaderno\nbloques 5–8", "notebook\npartes C y D", "ejercicio\nde cierre"]
v.b("¿Y en qué orden estudiar? [[a]]Primero, los apuntes. [[b]]Después, los bloques uno a cuatro del cuaderno, y las partes A y B del notebook. [[c]]Luego los bloques cinco a ocho, y las partes C y D. [[d]]Y al final, el ejercicio de cierre.",
    CL(),
    [[Bc(200 + 300 * i, 480, 250, 160, "blanco", c, 5, 26, sombra=True, en=k, ent="subir"), T(n, 200 + 300 * i, 480, tam=32, c=c, w="b", mw=220, en=k, t=0.1),
      T(str(i + 1), 200 + 300 * i, 380, tam=44, c=c, en=k, t=0.1)]
     for i, (n, c, k) in enumerate(zip(STOPS, ["teal", "violeta", "azul", "violeta", "azul", "rojo"], "abbccd"))],
    [L(335 + 300 * i, 480, 365 + 300 * i, 480, "grisclaro", 5, True, en=k, t=0.3, dur=0.4) for i, k in enumerate("bbccd")], sec="4. Cómo estudiar")
v.b("Y si te trabas más de diez minutos, [[a]]mira la solución, [[b]]entiéndela, [[c]]y vuelve a hacerlo sin mirar.",
    CL(), P("más de 10 minutos trabado", 960, 280, "ambar", 56, ent="subir"), P("mira la solución", 560, 520, "azul", 56, en="a"), P("entiéndela", 1060, 520, "teal", 56, en="b", t=0.0),
    P("hazlo sin mirar", 1500, 520, "violeta", 56, en="c"), L(790, 520, 840, 520, "gris", 5, True, en="b"), L(1270, 520, 1320, 520, "gris", 5, True, en="c"))
v.b("Eso es todo. Con estas reglas del juego ya puedes preparar tus datos [[a]]y medir un modelo con honestidad. [[b]]En la próxima clase empieza la introducción al Machine Learning. Mucho éxito con tu estudio.",
    CL(), T("las reglas del juego", 960, 380, tam=96, c="violeta", ent="subir", t=0.2), P("preparar bien · validar bien · medir con honestidad", 960, 600, "teal", 52, en="a"),
    P("siguiente: Introducción al Machine Learning", 960, 820, "azul", 48, en="b"), sec="5. Cierre")

print(v.guardar(OUT), "beats")
