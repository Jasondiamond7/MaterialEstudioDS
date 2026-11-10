import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dsl import *
from util37 import (D, FG, out, tarjetas, siguiente, ilustrativo, real, caja_formula, fila)

OUT = out("guion_c37_12_repaso.json")
v = V("Repaso final y checklist")
SC = D["script"]
RA = SC["SVM radial"]

v.portada("Repaso final", "Toda la clase en unos minutos, y un checklist para tu próximo clasificador",
          "Última parada: el repaso. Toda la clase de máquinas de vectores de soporte y clasificadores clásicos en unos minutos, y un checklist para tu próximo modelo.")

# ---- 1. Datos y línea base
v.b("Primero, los datos y la vara mínima. [[a]]La base de Wisconsin tiene 569 casos, con una partición fija: 398 para entrenar y 171 para la prueba. [[b]]La línea base, decir siempre benigno, acierta el 62.6 por ciento, y no detecta ni un maligno. [[c]]Todo modelo se compara contra eso.",
    CL(), T("1 · datos y línea base", 960, 150, tam=52, c="tinta", ent="subir"),
    P("569 casos · entrenamiento 398 · prueba 171", 960, 300, "teal", 48, en="a"),
    P("«siempre benigno»: 62.6 % de exactitud, sensibilidad 0", 960, 440, "ambar", 46, en="b"),
    P("todo modelo se compara con la línea base", 960, 580, "violeta", 46, en="c"), real(1050, en="a"), sec="1. Datos y línea base")

# ---- 2. Métricas
v.b("Las métricas salen de la matriz de confusión: filas, la clase real; columnas, la predicha. [[a]]Sensibilidad: verdaderos positivos sobre todos los positivos reales. Especificidad: verdaderos negativos sobre los negativos reales. [[b]]Precisión: de los que declaro positivos, cuántos lo eran. [[c]]En el ejemplo de la diapositiva nueve: 0.80, 0.80 y 0.632.",
    CL(), T("2 · la matriz de confusión", 960, 150, tam=52, c="tinta", ent="subir"),
    P("sensibilidad = VP/(VP+FN) · especificidad = VN/(VN+FP)", 960, 300, "verde", 44, en="a"),
    P("precisión = VP/(VP+FP)", 960, 440, "violeta", 48, en="b"),
    T("diapositiva 9: 0.80 · 0.80 · 0.632", 960, 580, tam=44, c="tinta", w="sb", en="c"), sec="2. Métricas")

# ---- 3. Logística
v.b("La regresión logística: [[a]]un puntaje lineal, eta, convertido en probabilidad con uno sobre uno más e a la menos eta. Con eta igual a uno, 0.731. [[b]]El umbral decide la clase, sin cambiar la probabilidad: bajarlo gana sensibilidad, subirlo gana especificidad. [[c]]Y e elevado a beta multiplica los odds, no la probabilidad.",
    CL(), T("3 · la regresión logística", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("p = 1 / (1 + e^(−η))      η = 1 → p = 0.731", y=310, c="teal", tam=50, w=1300, en="a", wf="sb"),
    P("el umbral decide la clase; la probabilidad no cambia", 960, 480, "ambar", 46, en="b"),
    P("e^β multiplica los ODDS", 960, 620, "violeta", 48, en="c"), sec="3. La logística")
v.b("Y dos trampas que vimos: [[a]]en Python, la logística clásica necesita penalty igual a None. [[b]]Y el coeficiente de la suavidad, 124.98, se lee en pasos de 0.01: multiplica los odds por 3.49.",
    T("Python: LogisticRegression(penalty=None)", 960, 780, tam=40, c="azul", w="sb", en="a"),
    T("suavidad: +0.01 → odds × 3.49", 960, 870, tam=40, c="gris", w="sb", en="b"))

# ---- 4. SVM
v.b("La SVM: [[a]]la frontera es efe de equis igual a cero, y la distancia de un punto es el valor absoluto de efe, dividido por la norma de doble u. [[b]]El ancho del margen es dos sobre la norma de doble u, y la SVM lo maximiza. [[c]]Solo los vectores de soporte deciden la frontera: en el ejemplo de treinta y dos casos, eran tres.",
    CL(), T("4 · la SVM", 960, 150, tam=52, c="tinta", ent="subir"),
    P("distancia = |f(x)| / ‖w‖", 960, 300, "ambar", 48, en="a"), P("ancho del margen = 2 / ‖w‖ → se maximiza", 960, 440, "violeta", 46, en="b"),
    P("solo los vectores de soporte deciden (3 de 32)", 960, 580, "teal", 46, en="c"), sec="4. La SVM")
v.b("Con las clases mezcladas, el margen suave: [[a]]cada punto paga una holgura, el máximo entre cero y uno menos {y·f|ye por efe}: la pérdida hinge. [[b]]Y C es el precio de cada violación: chico, calle ancha; grande, calle angosta.",
    T("ξ = máx(0, 1 − y·f) = pérdida hinge", 960, 760, tam=42, c="rojo", w="sb", en="a"),
    T("C chico → calle ancha · C grande → calle angosta", 960, 860, tam=40, c="tinta", w="sb", en="b"))

# ---- 5. Kernels, C y gamma
v.b("Los kernels: [[a]]miden la similitud como si los puntos estuvieran en un espacio más grande, sin construirlo. El radial es e a la menos gamma por la distancia al cuadrado. [[b]]Con distancia dos: 0.670 con gamma 0.1, y 0.018 con gamma uno. [[c]]Gamma chica da fronteras suaves; gamma grande, islas que memorizan. Con gamma cien, la prueba bajó al nivel de la línea base.",
    CL(), T("5 · kernels, C y γ", 960, 150, tam=52, c="tinta", ent="subir"),
    caja_formula("K(x, z) = exp(−γ‖x − z‖²)", y=300, c="rojo", tam=54, w=1000, en="a", wf="sb"),
    T("distancia 2:  γ = 0.1 → 0.670  ·  γ = 1 → 0.018", 960, 460, tam=44, c="tinta", w="sb", en="b"),
    P("γ chica → suave · γ grande → islas (γ = 100: 62.6 % en prueba)", 960, 610, "ambar", 42, en="c"), sec="5. Kernels, C y γ")

# ---- 6. Estandarizar
v.b("Estandarizar: [[a]]la SVM radial y kNN miden distancias, así que sin estandarizar mandan las variables con números grandes. [[b]]La SVM radial de la clase baja de 97.1 a 90.1 por ciento sin estandarizar. [[c]]Y la escala se ajusta solo con el entrenamiento, dentro de un Pipeline.",
    CL(), T("6 · estandarizar", 960, 150, tam=52, c="tinta", ent="subir"),
    P("distancias sin estandarizar → mandan los números grandes", 960, 300, "ambar", 46, en="a"),
    P("SVM radial: 97.1 % estandarizada · 90.1 % sin estandarizar", 960, 440, "rojo", 44, en="b"),
    P("escala: solo con entrenamiento, dentro de un Pipeline", 960, 580, "teal", 46, en="c"), real(1050, en="b"), sec="6. Estandarizar")

# ---- 7. La comparación
v.b("La comparación, con la misma partición: [[a]]la SVM radial comete cinco errores; la logística, diez; la SVM lineal, catorce. Coincide con la diapositiva cincuenta y dos. [[b]]kNN, LDA y QDA quedan entre 91 y 92 por ciento. [[c]]Y la AUC de la logística y la lineal es la misma, 0.9746: mide el orden de los puntajes, no los aciertos.",
    CL(), T("7 · la comparación", 960, 150, tam=52, c="tinta", ent="subir"),
    P("radial 5 · logística 10 · lineal 14 errores", 960, 300, "verde", 48, en="a"),
    P("kNN, LDA, QDA: entre 91 % y 92 %", 960, 440, "violeta", 46, en="b"),
    P("AUC logística = AUC lineal = 0.9746", 960, 580, "teal", 46, en="c"), real(1050, en="a"), sec="7. La comparación")

# ---- 8. Las alertas del material
v.b("Y las precisiones del material que verificamos. [[a]]La base tiene 569 casos, no 357: esos son los benignos. [[b]]La regla que acierta 107 es siempre benigno, no siempre uno. [[c]]El estimador de máxima verosimilitud no siempre tiene solución: con separación perfecta, no existe. Y aunque la diapositiva sesenta y cuatro diga lo contrario, los scripts sí calculan la AUC.",
    CL(), T("8 · las precisiones del material", 960, 150, tam=52, c="tinta", ent="subir"),
    P("569 casos (357 son los benignos)", 960, 290, "ambar", 44, en="a"),
    P("107 aciertos = «siempre benigno», no «siempre uno»", 960, 410, "ambar", 44, en="b"),
    P("MLE sin máximo con separación perfecta", 960, 530, "ambar", 44, en="c"),
    P("los scripts SÍ calculan la AUC (bloque 9)", 960, 650, "ambar", 44, en="c", t=2.5), sec="8. Las precisiones del material")

# ---- 9. Errores frecuentes (diapositiva 58)
ERR = ["no verificar cuál etiqueta es la clase positiva", "escalar toda la base antes de la validación cruzada",
       "elegir C, gamma o el umbral mirando la prueba", "pasar clases 0/1 a la AUC en vez de puntajes",
       "leer decision_function como probabilidad", "comparar modelos con particiones distintas"]
v.b("La diapositiva cincuenta y ocho junta los errores frecuentes al implementar. [[a]]No verificar cuál etiqueta es la positiva. Escalar toda la base antes de la validación cruzada. [[b]]Elegir {C|, ce,}, gamma o el umbral mirando la prueba. Pasarle clases a la AUC en vez de puntajes. [[c]]Leer el puntaje de la SVM como probabilidad. Y comparar modelos con particiones distintas.",
    CL(), T("9 · errores frecuentes · diapositiva 58", 960, 140, tam=50, c="tinta", ent="subir"),
    *[K(330, 280 + 100 * i, "cruz", 46, en=("a" if i < 2 else ("b" if i < 4 else "c")), t=0.5 * (i % 2)) for i in range(6)],
    *[T(e, 390, 280 + 100 * i, tam=40, c="tinta", w="sb", anc="lm", en=("a" if i < 2 else ("b" if i < 4 else "c")), t=0.5 * (i % 2) + 0.1) for i, e in enumerate(ERR)],
    sec="9. Errores frecuentes")

# ---- 10. Checklist (diapositiva 59)
CK = ["definir la clase positiva y los errores relevantes", "reservar la prueba y construir folds apropiados",
      "ajustar el preprocesamiento dentro de cada fold", "comparar logística, SVM lineal y SVM radial",
      "elegir configuración y umbral con validación", "evaluar una vez en prueba y documentar límites"]
v.b("Y el checklist para una aplicación nueva, de la diapositiva cincuenta y nueve. [[a]]Uno: definir la clase positiva y qué errores importan. Dos: reservar la prueba y armar folds apropiados. [[b]]Tres: ajustar el preprocesamiento dentro de cada fold. Cuatro: comparar logística, SVM lineal y SVM radial. [[c]]Cinco: elegir configuración y umbral con validación. Y seis: evaluar una sola vez en la prueba, y documentar las limitaciones.",
    CL(), Tj(960, 560, 1400, 820, "checklist · diapositiva 59", CK, c="teal", tam=38, id="ck", visibles=0.0),
    U("ck", visibles=2.0, en="a", dur=1.2), U("ck", visibles=4.0, en="b", dur=1.2), U("ck", visibles=6.0, en="c", dur=1.2),
    U("ck", marcas=[dict(progreso=1.0)] * 6, en="c", t=2.5, dur=1.0), sec="10. Checklist")
v.b("Con datos en el tiempo, o con varias filas por persona, los folds deben respetar el tiempo o los grupos. [[a]]Y recuerda: es una base histórica para aprender, no una herramienta clínica.",
    CL(), P("datos temporales o por persona → folds por tiempo o grupos", 960, 380, "violeta", 46),
    P("base histórica para aprender, no herramienta clínica", 960, 540, "rojo", 46, en="a"))

# ---- 11. Cierre
v.b("Eso es todo por esta clase. [[a]]Ahora te toca a ti: abre el cuaderno de ejercicios y el notebook de la clase treinta y siete, y repite estas cuentas con tus propias manos. [[b]]La próxima clase: evaluación de modelos supervisados y selección de modelos.",
    CL(), T("ahora te toca a ti", 960, 300, tam=70, c="tinta", ent="subir"),
    P("cuaderno de ejercicios + notebook C37_SVM.ipynb", 960, 470, "teal", 48, en="a"),
    P("próxima clase: evaluación y selección de modelos", 960, 620, "violeta", 46, en="b"), sec="11. Cierre")

n = v.guardar(OUT)
print(n, "beats")
